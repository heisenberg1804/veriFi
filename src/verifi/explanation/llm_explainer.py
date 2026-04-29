"""
VeriFi — LLM Explainer with pluggable backends (Ollama local / Claude API).

Drop this file into: src/verifi/explanation/llm_explainer.py
It replaces the Claude-only version with a backend-agnostic design.

Usage:
    # Local development (free, no API key needed)
    explainer = create_explainer(backend="ollama", model="qwen3:8b")

    # Production (higher quality explanations)
    explainer = create_explainer(backend="claude", model="claude-sonnet-4-20250514")
"""
from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from pathlib import Path

import httpx
import structlog

from verifi.explanation.prompts import FORENSIC_SYSTEM_PROMPT, FORENSIC_USER_TEMPLATE

logger = structlog.get_logger()


# ─── Abstract interface ───────────────────────────────────────

class BaseExplainer(ABC):
    """Backend-agnostic forensic explainer interface."""

    @abstractmethod
    async def explain(
        self,
        video_metadata: dict,
        analysis_summary: dict,
        heatmap_paths: list[Path],
        signal_stats: dict,
    ) -> dict:
        ...

    def _build_text_prompt(
        self,
        video_metadata: dict,
        analysis_summary: dict,
        signal_stats: dict,
    ) -> str:
        """Build the analysis prompt text (shared across backends)."""
        return FORENSIC_USER_TEMPLATE.format(
            duration_sec=video_metadata.get("duration_sec", "unknown"),
            resolution=video_metadata.get("resolution", "unknown"),
            fps=video_metadata.get("fps", "unknown"),
            codec=video_metadata.get("codec", "unknown"),
            has_audio=video_metadata.get("has_audio", False),
            video_score=analysis_summary.get("video_score", 0),
            verdict=analysis_summary.get("verdict", "unknown"),
            manipulation_type=analysis_summary.get("manipulation_type", "unknown"),
            num_frames=analysis_summary.get("num_frames", 0),
            num_flagged=analysis_summary.get("num_flagged", 0),
            clip_mean=signal_stats.get("clip_mean", 0),
            clip_max=signal_stats.get("clip_max", 0),
            clip_std=signal_stats.get("clip_std", 0),
            effnet_mean=signal_stats.get("effnet_mean", 0),
            effnet_max=signal_stats.get("effnet_max", 0),
            effnet_std=signal_stats.get("effnet_std", 0),
            freq_score=signal_stats.get("freq_score", 0),
            hf_suppression=signal_stats.get("hf_suppression", 0),
            temporal_summary=signal_stats.get("temporal_summary", "N/A"),
            av_sync_summary=signal_stats.get("av_sync_summary", "N/A"),
            peak_timestamps=analysis_summary.get("peak_timestamps", "N/A"),
            score_pattern=analysis_summary.get("score_pattern", "N/A"),
        )

    def _parse_json_response(self, raw: str) -> dict:
        """Parse JSON from LLM response, handling common formatting issues."""
        cleaned = raw.strip()

        # Strip markdown code fences
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        # Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object within the text
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass

        # Fallback: return raw text as summary
        logger.warn("json_parse_failed", raw_length=len(raw))
        return {
            "summary": cleaned[:500],
            "evidence": [],
            "caveats": ["LLM response was not valid JSON — raw text returned"],
            "parse_error": True,
        }


# ─── Ollama backend (local, free) ────────────────────────────

class OllamaExplainer(BaseExplainer):
    """
    Local LLM explainer using Ollama.

    Requirements:
      - Ollama running: `ollama serve`
      - Model pulled: `ollama pull qwen3:8b` (or your preferred model)

    Recommended models for 16GB M3 Air:
      - llama3.1:8b      — best quality/speed tradeoff (~5GB RAM)
      - mistral:7b        — fast, decent reasoning
      - gemma2:9b         — good at structured output
      - llama3.2:3b       — fastest, lower quality (good for iteration)

    For multimodal (heatmap analysis):
      - llava:13b         — can see images but needs ~10GB RAM
      - llava:7b          — lighter vision model
    """

    def __init__(
        self,
        model: str = "qwen3:8b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    async def explain(
        self,
        video_metadata: dict,
        analysis_summary: dict,
        heatmap_paths: list[Path],
        signal_stats: dict,
    ) -> dict:
        prompt_text = self._build_text_prompt(
            video_metadata, analysis_summary, signal_stats
        )

        # Check if model supports vision (for heatmap images)
        is_vision_model = any(
            v in self.model.lower() for v in ["llava", "bakllava", "moondream"]
        )

        if is_vision_model and heatmap_paths:
            return await self._explain_with_images(
                prompt_text, heatmap_paths
            )
        return await self._explain_text_only(prompt_text)

    async def _explain_text_only(self, prompt: str) -> dict:
        """Text-only generation via Ollama /api/generate."""
        full_prompt = (
            f"SYSTEM: {FORENSIC_SYSTEM_PROMPT}\n\n"
            f"USER: {prompt}"
        )

        logger.info(
            "calling_ollama",
            model=self.model,
            mode="text",
            prompt_length=len(full_prompt),
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,     # lower = more deterministic
                            "num_predict": 2000,    # max output tokens
                        },
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                raw = data.get("response", "")

                logger.info(
                    "ollama_response",
                    model=self.model,
                    tokens=data.get("eval_count", "?"),
                    duration_ms=data.get("total_duration", 0) / 1e6,
                )

                return self._parse_json_response(raw)

            except httpx.ConnectError:
                logger.error("ollama_not_running")
                return {
                    "summary": "Ollama is not running. Start it with: ollama serve",
                    "evidence": [],
                    "caveats": ["LLM explainer unavailable"],
                    "error": "ollama_connection_refused",
                }
            except httpx.HTTPStatusError as e:
                logger.error("ollama_http_error", status=e.response.status_code)
                return {
                    "summary": f"Ollama returned HTTP {e.response.status_code}",
                    "evidence": [],
                    "caveats": ["LLM explainer error"],
                    "error": str(e),
                }

    async def _explain_with_images(
        self, prompt: str, heatmap_paths: list[Path]
    ) -> dict:
        """Multimodal generation via Ollama /api/generate (vision models)."""
        images_b64 = []
        for path in heatmap_paths[:3]:  # limit to 3 images for memory
            if path.exists():
                images_b64.append(
                    base64.standard_b64encode(path.read_bytes()).decode("utf-8")
                )

        full_prompt = (
            f"SYSTEM: {FORENSIC_SYSTEM_PROMPT}\n\n"
            f"USER: The following heatmap images show GradCAM attention "
            f"maps for flagged frames. Bright regions indicate detected "
            f"anomalies.\n\n{prompt}"
        )

        logger.info(
            "calling_ollama",
            model=self.model,
            mode="vision",
            n_images=len(images_b64),
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "images": images_b64,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 2000,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return self._parse_json_response(data.get("response", ""))

    async def check_health(self) -> dict:
        """Check if Ollama is running and model is available."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                models = resp.json().get("models", [])
                names = [m["name"] for m in models]
                has_model = any(self.model in n for n in names)
                return {
                    "ollama_running": True,
                    "available_models": names,
                    "requested_model": self.model,
                    "model_available": has_model,
                    "hint": f"Run 'ollama pull {self.model}' to download"
                    if not has_model else "ready",
                }
            except httpx.ConnectError:
                return {
                    "ollama_running": False,
                    "hint": "Start Ollama with: ollama serve",
                }


# ─── Claude backend (production) ─────────────────────────────

class ClaudeExplainer(BaseExplainer):
    """
    Claude API explainer for production-quality forensic reports.
    Requires ANTHROPIC_API_KEY environment variable.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514", max_tokens: int = 1500):
        self.model = model
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    async def explain(
        self,
        video_metadata: dict,
        analysis_summary: dict,
        heatmap_paths: list[Path],
        signal_stats: dict,
    ) -> dict:
        prompt_text = self._build_text_prompt(
            video_metadata, analysis_summary, signal_stats
        )

        # Build multimodal content
        content = []
        for i, path in enumerate(heatmap_paths[:5]):
            if path.exists():
                b64 = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
                content.append({"type": "text", "text": f"Heatmap for flagged frame {i+1}:"})
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": b64},
                })

        content.append({"type": "text", "text": prompt_text})

        logger.info("calling_claude", model=self.model, n_images=len(heatmap_paths))

        client = self._get_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=FORENSIC_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )

        raw = response.content[0].text.strip()
        return self._parse_json_response(raw)


# ─── Factory ──────────────────────────────────────────────────

def create_explainer(
    backend: str = "ollama",
    model: str | None = None,
    **kwargs,
) -> BaseExplainer:
    """
    Create an LLM explainer instance.

    Args:
        backend: "ollama" (local, free) or "claude" (API, paid)
        model: Model name. Defaults:
               - ollama: "qwen3:8b"
               - claude: "claude-sonnet-4-20250514"
        **kwargs: Passed to the backend constructor.

    Examples:
        # Development (free)
        explainer = create_explainer("ollama", "qwen3:8b")

        # Faster iteration with smaller model
        explainer = create_explainer("ollama", "llama3.2:3b")

        # Vision model (can analyze heatmap images)
        explainer = create_explainer("ollama", "llava:7b")

        # Production
        explainer = create_explainer("claude")
    """
    if backend == "ollama":
        return OllamaExplainer(model=model or "qwen3:8b", **kwargs)
    elif backend == "claude":
        return ClaudeExplainer(model=model or "claude-sonnet-4-20250514", **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'ollama' or 'claude'.")
