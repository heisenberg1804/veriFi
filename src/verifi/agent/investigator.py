"""
Forensic investigation agent: LLM-driven tool-use loop.

Save to: src/verifi/agent/investigator.py

The agent:
1. Starts with a quick_scan (Tier 1 fixed pipeline)
2. Reviews results, plans investigation
3. Calls tools iteratively (up to max_rounds)
4. Produces a forensic report with reasoning trace

Works with both Ollama (local, free) and Claude API (production).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
import structlog

from verifi.agent.planner import (
    AGENT_SYSTEM_PROMPT,
    INVESTIGATION_CONCLUDE_PROMPT,
    INVESTIGATION_CONTINUE_PROMPT,
    INVESTIGATION_START_PROMPT,
)
from verifi.tools.base import ToolRegistry, ToolResult

logger = structlog.get_logger()


@dataclass
class InvestigationStep:
    """One step in the investigation."""
    step_number: int
    tool_name: str
    tool_args: dict
    result_summary: str
    execution_time_ms: float


@dataclass
class InvestigationReport:
    """Complete output of an agentic investigation."""
    verdict: str
    confidence: float
    manipulation_type: str
    summary: str
    evidence: list[str]
    investigation_trace: list[str]
    caveats: list[str]
    recommended_action: str
    steps: list[InvestigationStep] = field(default_factory=list)
    total_time_sec: float = 0.0
    num_tool_calls: int = 0
    raw_llm_response: str = ""

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "manipulation_type": self.manipulation_type,
            "summary": self.summary,
            "evidence": self.evidence,
            "investigation_trace": self.investigation_trace,
            "caveats": self.caveats,
            "recommended_action": self.recommended_action,
            "num_tool_calls": self.num_tool_calls,
            "total_time_sec": round(self.total_time_sec, 2),
            "steps": [
                {
                    "step": s.step_number,
                    "tool": s.tool_name,
                    "args": {k: v for k, v in s.tool_args.items() if not isinstance(v, np.ndarray)},
                    "result": s.result_summary,
                    "time_ms": round(s.execution_time_ms, 1),
                }
                for s in self.steps
            ],
        }


class ForensicInvestigator:
    """
    LLM-driven forensic investigation agent.

    Uses tool-calling to iteratively investigate a video.
    The LLM decides which tools to use and in what order,
    based on intermediate findings.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        backend: str = "ollama",
        model: str = "qwen3:8b",
        ollama_url: str = "http://localhost:11434",
        max_rounds: int = 5,
    ):
        self.registry = registry
        self.backend = backend
        self.model = model
        self.ollama_url = ollama_url
        self.max_rounds = max_rounds

        # Shared context: images and data that tools produce
        # Keys like "frame_42" → np.ndarray
        self._context: dict[str, any] = {}
        self._video_path: str = ""

    async def investigate(
        self,
        video_path: str,
        output_dir: str | Path | None = None,
        scan_context: str | None = None,
    ) -> InvestigationReport:
        """
        Run a full agentic investigation on a video.

        Args:
            video_path: Path to the video file.
            output_dir: Where to save artifacts.
            scan_context: Pre-computed Tier 1 scan results as a text summary.
                          When provided, the agent skips quick_scan and starts
                          investigating specific signals directly.

        Returns:
            InvestigationReport with verdict, evidence, and reasoning trace.
        """
        self._video_path = video_path
        self._context = {"video_path": video_path}
        if output_dir:
            self._context["output_dir"] = str(output_dir)

        total_t0 = time.perf_counter()
        steps: list[InvestigationStep] = []
        findings: list[str] = []

        logger.info("investigation_start", video=video_path, backend=self.backend,
                     has_scan_context=scan_context is not None)

        # Build conversation history
        if scan_context:
            initial_prompt = (
                f"Investigate this video for authenticity: {video_path}\n\n"
                f"Tier 1 detection results are already available — do NOT call quick_scan.\n\n"
                f"{scan_context}\n\n"
                f"IMPORTANT: Frame images are NOT pre-loaded in context. To analyze "
                f"a specific frame visually, you must first call sample_more_frames "
                f"to extract frames from the video. After that, tools like detect_faces "
                f"and run_dct_analysis can use the extracted frames. Do not call "
                f"detect_faces or run_clip_detection with an image_key before "
                f"extracting frames.\n\n"
                f"Based on these findings, decide what to investigate further. "
                f"Call a tool to examine specific regions, signals, or metadata."
            )
        else:
            initial_prompt = INVESTIGATION_START_PROMPT.format(video_path=video_path)

        messages = [
            {"role": "user", "content": initial_prompt},
        ]

        for round_num in range(self.max_rounds + 1):
            # Decide prompt based on round
            if round_num > 0 and round_num < self.max_rounds:
                messages.append({
                    "role": "user",
                    "content": INVESTIGATION_CONTINUE_PROMPT.format(
                        tools_used=len(steps),
                        max_tools=self.max_rounds,
                        findings_summary="\n".join(f"- {f}" for f in findings),
                    ),
                })
            elif round_num >= self.max_rounds:
                messages.append({
                    "role": "user",
                    "content": INVESTIGATION_CONCLUDE_PROMPT.format(
                        tools_used=len(steps),
                        findings_summary="\n".join(f"- {f}" for f in findings),
                    ),
                })

            # Call LLM
            if self.backend == "ollama":
                response = await self._call_ollama(messages)
            else:
                response = await self._call_claude(messages)

            # Handle timeout — build report from what we have so far
            if response.get("timeout"):
                logger.warning("investigation_timeout", round=round_num, steps=len(steps))
                return self._build_fallback_report(steps, findings, time.perf_counter() - total_t0)

            # Check if LLM wants to call a tool or is done
            if response.get("tool_calls"):
                for tc in response["tool_calls"]:
                    tool_name = tc["name"]
                    tool_args = tc.get("arguments", {})

                    logger.info("agent_tool_call", round=round_num, tool=tool_name)

                    # Resolve image references from context
                    resolved_args = self._resolve_args(tool_name, tool_args)

                    # Execute the tool
                    result = self.registry.execute(tool_name, **resolved_args)

                    # Store any images produced by the tool in context
                    stored_keys = self._store_result_in_context(tool_name, result)

                    # Record step
                    step = InvestigationStep(
                        step_number=len(steps) + 1,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        result_summary=result.summary(),
                        execution_time_ms=result.execution_time_ms,
                    )
                    steps.append(step)
                    findings.append(result.summary())

                    # Add tool result to conversation
                    messages.append({
                        "role": "assistant",
                        "content": f"Called {tool_name}. Result: {result.summary()}",
                    })

                    # Tell the LLM exactly which image keys are now available
                    if stored_keys:
                        keys_str = ", ".join(stored_keys)
                        messages.append({
                            "role": "user",
                            "content": (
                                f"Frames extracted and available with these keys: "
                                f"{keys_str}. Use these EXACT keys as image_key "
                                f"when calling run_clip_detection, run_dct_analysis, "
                                f"run_effnet_detection, or detect_faces."
                            ),
                        })

            elif response.get("content"):
                # LLM is done — parse the final report
                report = self._parse_report(
                    response["content"], steps, time.perf_counter() - total_t0
                )
                logger.info(
                    "investigation_complete",
                    verdict=report.verdict,
                    steps=len(steps),
                    time=f"{report.total_time_sec:.1f}s",
                )
                return report

        # If we exhausted rounds without a report, force one
        return self._build_fallback_report(steps, findings, time.perf_counter() - total_t0)

    def _resolve_args(self, tool_name: str, tool_args: dict) -> dict:
        """
        Resolve image_key references to actual numpy arrays from context.
        Also inject video_path and output_dir from context.
        """
        resolved = dict(tool_args)

        # Resolve image references
        image_key = resolved.pop("image_key", None)
        if image_key and image_key in self._context:
            resolved["image"] = self._context[image_key]

        frame_a_key = resolved.pop("frame_a_key", None)
        if frame_a_key and frame_a_key in self._context:
            resolved["frame_a"] = self._context[frame_a_key]

        frame_b_key = resolved.pop("frame_b_key", None)
        if frame_b_key and frame_b_key in self._context:
            resolved["frame_b"] = self._context[frame_b_key]

        # Always inject video path and output dir
        if "video_path" not in resolved:
            resolved["video_path"] = self._context.get("video_path")
        if "output_dir" not in resolved:
            resolved["output_dir"] = self._context.get("output_dir")

        return resolved

    def _store_result_in_context(self, tool_name: str, result: ToolResult) -> list[str]:
        """Store images and data from tool results in shared context.

        Returns:
            List of context keys that were stored (for reporting to the LLM).
        """
        stored_keys: list[str] = []
        if not result.success:
            return stored_keys

        data = result.data

        # Store zoomed images
        if "image" in data and isinstance(data["image"], np.ndarray):
            key = f"zoomed_{len(self._context)}"
            self._context[key] = data["image"]
            stored_keys.append(key)

        # Store extracted frames
        if "frames" in data:
            for frame_data in data["frames"]:
                if "image" in frame_data:
                    key = f"frame_{frame_data['frame_idx']}"
                    self._context[key] = frame_data["image"]
                    stored_keys.append(key)

        # Store face crops
        if "faces" in data:
            for face in data["faces"]:
                if "crop" in face:
                    key = f"face_{face['face_id']}"
                    self._context[key] = face["crop"]
                    stored_keys.append(key)

        # Store GradCAM overlays
        if "overlay" in data and isinstance(data["overlay"], np.ndarray):
            key = f"gradcam_{data.get('model_name', 'unknown')}_{data.get('frame_idx', 0)}"
            self._context[key] = data["overlay"]
            stored_keys.append(key)

        # Store quick_scan frame scores for reference
        if "frame_scores" in data:
            self._context["frame_scores"] = data["frame_scores"]

    async def _call_ollama(self, messages: list[dict]) -> dict:
        """Call Ollama with tool-use support."""
        tools = self.registry.schemas_for_ollama()

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                *messages,
            ],
            "tools": tools,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 3000,
            },
        }

        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                resp = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()

                message = data.get("message", {})

                # Check for tool calls
                tool_calls = message.get("tool_calls", [])
                if tool_calls:
                    parsed_calls = []
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        parsed_calls.append({
                            "name": func.get("name", ""),
                            "arguments": func.get("arguments", {}),
                        })
                    return {"tool_calls": parsed_calls}

                # No tool calls — LLM is responding with text
                return {"content": message.get("content", "")}

            except httpx.ReadTimeout:
                logger.warning("ollama_timeout", timeout=300.0)
                return {"content": "", "timeout": True}

            except httpx.ConnectError:
                logger.error("ollama_not_running")
                return {"content": '{"error": "Ollama not running"}'}

    async def _call_claude(self, messages: list[dict]) -> dict:
        """Call Claude API with tool-use support."""
        import anthropic

        client = anthropic.Anthropic()
        tools = self.registry.schemas_for_claude()

        response = client.messages.create(
            model=self.model,
            max_tokens=3000,
            system=AGENT_SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        # Check for tool use
        for block in response.content:
            if block.type == "tool_use":
                return {
                    "tool_calls": [{
                        "name": block.name,
                        "arguments": block.input,
                    }]
                }

        # Text response
        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return {"content": text}

    def _parse_report(
        self,
        raw: str,
        steps: list[InvestigationStep],
        total_time: float,
    ) -> InvestigationReport:
        """Parse the LLM's final report from its text response."""
        # Try to extract JSON
        cleaned = raw.strip()
        if "```" in cleaned:
            cleaned = cleaned.split("```")[1] if "```" in cleaned else cleaned
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        # Find JSON object
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1

        if start != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start:end])
                return InvestigationReport(
                    verdict=parsed.get("verdict", "SUSPICIOUS"),
                    confidence=parsed.get("confidence", 0.5),
                    manipulation_type=parsed.get("manipulation_type", "unknown"),
                    summary=parsed.get("summary", "Investigation complete"),
                    evidence=parsed.get("evidence", []),
                    investigation_trace=parsed.get("investigation_trace", []),
                    caveats=parsed.get("caveats", []),
                    recommended_action=parsed.get("recommended_action", "Manual review"),
                    steps=steps,
                    total_time_sec=total_time,
                    num_tool_calls=len(steps),
                    raw_llm_response=raw,
                )
            except json.JSONDecodeError:
                pass

        # Fallback: couldn't parse JSON
        return self._build_fallback_report(
            steps, [s.result_summary for s in steps], total_time, raw
        )

    def _build_fallback_report(
        self,
        steps: list[InvestigationStep],
        findings: list[str],
        total_time: float,
        raw_response: str = "",
    ) -> InvestigationReport:
        """Build a report when the LLM doesn't produce valid JSON."""
        return InvestigationReport(
            verdict="SUSPICIOUS",
            confidence=0.5,
            manipulation_type="unknown",
            summary=(
                f"Investigation completed with {len(steps)} tool calls. "
                f"LLM did not produce structured output."
            ),
            evidence=findings,
            investigation_trace=[
                f"Step {s.step_number}: {s.tool_name} → {s.result_summary}"
                for s in steps
            ],
            caveats=[
                "LLM failed to produce structured JSON report"
                " — findings are raw tool outputs"
            ],
            recommended_action="Manual review of tool results recommended",
            steps=steps,
            total_time_sec=total_time,
            num_tool_calls=len(steps),
            raw_llm_response=raw_response,
        )
