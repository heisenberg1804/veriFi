"""
Phase 4 integration test: Run all three tiers on a video.

Save to: scripts/test_phase4.py
Run:  python scripts/test_phase4.py data/sample_videos/YOUR_VIDEO.mp4
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def setup_pipeline():
    """Create pipeline and load models."""
    from verifi.config import AppConfig
    from verifi.pipeline.orchestrator import VeriFiPipeline

    config = AppConfig()
    config.sampling.min_laplacian_var = 50.0

    pipeline = VeriFiPipeline(config)
    pipeline.load_models()
    return pipeline


def setup_tools(pipeline):
    """Create tool registry from loaded pipeline."""
    from verifi.tools.factory import create_tool_registry, get_langchain_tools

    registry = create_tool_registry(pipeline)
    lc_tools = get_langchain_tools(registry)
    return registry, lc_tools


def check_ollama(model: str = "qwen3:8b") -> bool:
    """Check if Ollama is running and model is available."""
    import httpx
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if resp.status_code != 200:
            return False
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(model in m for m in models)
    except (httpx.ConnectError, httpx.ReadTimeout):
        return False


async def test_tier1(registry, video_path: str):
    """Tier 1: Fixed pipeline via tool interface."""
    print("\n┌─ Tier 1: Quick Scan (fixed pipeline) ────────")
    t0 = time.perf_counter()
    result = registry.execute("quick_scan", video_path=video_path)
    elapsed = time.perf_counter() - t0

    if result.success:
        d = result.data
        print(f"│ Verdict:      {d['verdict']}")
        print(f"│ Score:        {d['video_score']:.3f}")
        print(f"│ Dominant:     {d['dominant_path']} path")
        print(f"│ Manipulation: {d['manipulation_type']}")
        print(f"│ Frames:       {d['num_frames_flagged']}/{d['num_frames_analyzed']} flagged")
        print(f"│ Time:         {elapsed:.1f}s")
    else:
        print(f"│ FAILED: {result.error}")
    print("└──────────────────────────────────────────────")
    return result


async def test_tier2(scan_result, video_path: str):
    """Tier 2: Single LLM explanation call (Ollama)."""
    print("\n┌─ Tier 2: LLM Explanation (single call) ──────")

    from verifi.explanation.llm_explainer import create_explainer

    explainer = create_explainer("ollama", "qwen3:8b")
    health = await explainer.check_health()

    if not health.get("model_available"):
        print(f"│ [SKIP] Ollama not ready: {health.get('hint', 'unknown')}")
        print("└──────────────────────────────────────────────")
        return None

    t0 = time.perf_counter()
    d = scan_result.data
    result = await explainer.explain(
        video_metadata={
            "duration_sec": 8, "resolution": "1280x720",
            "fps": 24, "codec": "h264", "has_audio": True,
        },
        analysis_summary={
            "video_score": d["video_score"],
            "verdict": d["verdict"],
            "manipulation_type": d["manipulation_type"],
            "num_frames": d["num_frames_analyzed"],
            "num_flagged": d["num_frames_flagged"],
            "peak_timestamps": "N/A",
            "score_pattern": "stable",
        },
        heatmap_paths=[Path(p) for p in d.get("heatmap_paths", [])],
        signal_stats=d.get("signal_stats", {}),
    )
    elapsed = time.perf_counter() - t0

    print(f"│ Time: {elapsed:.1f}s")
    if result.get("parse_error"):
        print("│ [WARN] LLM didn't return valid JSON")
        print(f"│ Raw: {result.get('summary', '')[:200]}")
    else:
        print(f"│ Summary: {result.get('summary', 'N/A')[:200]}")
        for i, e in enumerate(result.get("evidence", [])[:3]):
            print(f"│ Evidence {i+1}: {e[:120]}")
    print("└──────────────────────────────────────────────")
    return result


async def test_tier3(registry, lc_tools: list, video_path: str, scan_result=None):
    """Tier 3: LangGraph agentic investigation."""
    print("\n┌─ Tier 3: Agentic Investigation (LangGraph) ──")

    from verifi.agent.investigator import ForensicInvestigator

    investigator = ForensicInvestigator(
        registry=registry,
        backend="ollama",
        model="qwen3:8b",
        max_rounds=4,
    )

    # Build scan context from Tier 1 results so the agent skips quick_scan
    scan_context = None
    if scan_result and scan_result.success:
        d = scan_result.data
        stats = d.get("signal_stats", {})
        scan_context = (
            f"Here are the Tier 1 detection results:\n"
            f"- Verdict: {d['verdict']} (score: {d['video_score']:.3f})\n"
            f"- Dominant path: {d['dominant_path']}\n"
            f"- Manipulation type: {d['manipulation_type']}\n"
            f"- {d['num_frames_flagged']} of {d['num_frames_analyzed']} frames flagged "
            f"by the frame-level path\n"
            f"- CLIP scored up to {stats.get('clip_max', 'N/A')}, "
            f"mean {stats.get('clip_mean', 'N/A')}\n"
            f"- EfficientNet max: {stats.get('effnet_max', 'N/A')}, "
            f"mean {stats.get('effnet_mean', 'N/A')}\n"
            f"- DCT frequency score: {stats.get('freq_score', 'N/A')}, "
            f"HF suppression: {stats.get('hf_suppression', 'N/A')}\n"
            f"- Temporal consistency: {stats.get('temporal_summary', 'N/A')}\n"
            f"\nWhat specific region or signal do you want to examine more closely?"
        )
        print("│ Using Tier 1 scan results as context (skipping quick_scan)")

    t0 = time.perf_counter()
    output_dir = "data/reports/phase4_agent_test"

    report = await investigator.investigate(
        video_path, output_dir=output_dir, scan_context=scan_context,
    )
    elapsed = time.perf_counter() - t0

    print(f"│ Verdict:       {report.verdict}")
    print(f"│ Confidence:    {report.confidence}")
    print(f"│ Manipulation:  {report.manipulation_type}")
    print(f"│ Tool calls:    {report.num_tool_calls}")
    print(f"│ Time:          {elapsed:.1f}s")
    print("│")
    print(f"│ Summary: {report.summary[:200]}")

    if report.investigation_trace:
        print("│")
        print("│ Investigation trace:")
        for t in report.investigation_trace[:5]:
            print(f"│   → {t[:120]}")

    if report.evidence:
        print("│")
        print("│ Evidence:")
        for i, e in enumerate(report.evidence[:5]):
            print(f"│   {i+1}. {e[:120]}")

    # Save report
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "investigation_report.json"
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    print("│")
    print(f"│ Report saved: {report_path}")
    print("└──────────────────────────────────────────────")

    return report


async def main(video_path: str):
    print("=" * 65)
    print("VeriFi — Phase 4 Three-Tier Integration Test")
    print("=" * 65)

    # ── Setup ──
    print("\n── Loading models + building tool registry ──")
    t0 = time.perf_counter()
    pipeline = setup_pipeline()
    registry, lc_tools = setup_tools(pipeline)
    print(f"  {len(registry)} tools registered: {registry.list_tools()}")
    print(f"  {len(lc_tools)} LangChain tools for agent")
    print(f"  Setup time: {time.perf_counter() - t0:.1f}s")

    # ── Tier 1: Quick scan ──
    scan_result = await test_tier1(registry, video_path)

    # ── Check Ollama for Tiers 2+3 ──
    ollama_ok = check_ollama("qwen3:8b")
    if not ollama_ok:
        print("\n[!] Ollama not available — skipping Tiers 2 and 3")
        print("    Start Ollama: ollama serve")
        print("    Pull model:   ollama pull qwen3:8b")
    else:
        # ── Tier 2: LLM explanation ──
        if scan_result and scan_result.success:
            await test_tier2(scan_result, video_path)

        # ── Tier 3: Agentic investigation (reuse Tier 1 results) ──
        await test_tier3(registry, lc_tools, video_path, scan_result=scan_result)

    # ── Cleanup ──
    pipeline.unload_models()

    print(f"\n{'=' * 65}")
    print("Phase 4 test complete.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sample_dir = Path("data/sample_videos")
        videos = list(sample_dir.glob("*.mp4")) + list(sample_dir.glob("*.mov"))
        if videos:
            video_path = str(videos[0])
            print(f"Auto-detected: {video_path}")
        else:
            print("Usage: python scripts/test_phase4.py <video_path>")
            sys.exit(1)
    else:
        video_path = sys.argv[1]

    asyncio.run(main(video_path))
