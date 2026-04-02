"""
Phase 3 integration test: Full pipeline on a real video.

Save to: scripts/test_phase3.py
Run:  python scripts/test_phase3.py data/sample_videos/YOUR_VIDEO.mp4
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main(video_path: str):
    from verifi.config import AppConfig
    from verifi.pipeline.orchestrator import VeriFiPipeline

    print("=" * 65)
    print("VeriFi — Phase 3 Full Pipeline Test")
    print("=" * 65)

    # ── Configure ──
    config = AppConfig()
    # Lower blur threshold for AI-generated content
    config.sampling.min_laplacian_var = 50.0

    # ── Create pipeline ──
    print("\n── Loading models ──")
    t0 = time.perf_counter()
    pipeline = VeriFiPipeline(config)
    pipeline.load_models()
    print(f"  Models loaded in {time.perf_counter() - t0:.1f}s")

    # ── Run analysis ──
    print(f"\n── Analyzing: {video_path} ──")
    try:
        report = pipeline.analyze(video_path)
    except Exception as e:
        print(f"\n  [FAIL] Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        pipeline.unload_models()
        sys.exit(1)

    # ── Print results ──
    analysis = report.analysis
    timings = report.timings

    print(f"\n── Results ──")
    print(f"  Video:            {report.video_metadata.get('filename', '?')}")
    print(f"  Duration:         {report.video_metadata.get('duration_sec', 0):.1f}s")
    print(f"  Resolution:       {report.video_metadata.get('resolution', '?')}")

    print(f"\n  ┌─ Verdict ─────────────────────────────")
    print(f"  │ Score:           {analysis.video_score:.3f}")
    print(f"  │ Verdict:         {analysis.verdict.value}")
    print(f"  │ Manipulation:    {analysis.manipulation_type.value}")
    print(f"  │ Dominant path:   {analysis.dominant_path}")
    print(f"  └────────────────────────────────────────")

    print(f"\n  ┌─ Path A: Face-level ─────────────────")
    print(f"  │ Active:          {analysis.face_path_active}")
    print(f"  │ Score:           {analysis.face_path_score:.3f}")
    print(f"  │ Face analyses:   {len(analysis.face_analyses)}")
    print(f"  │ Flagged:         {len(analysis.flagged_face_indices)}")
    print(f"  └────────────────────────────────────────")

    print(f"\n  ┌─ Path B: Frame-level ────────────────")
    print(f"  │ Active:          {analysis.frame_path_active}")
    print(f"  │ Score:           {analysis.frame_path_score:.3f}")
    print(f"  │ Frame analyses:  {len(analysis.frame_analyses)}")
    print(f"  │ Flagged:         {len(analysis.flagged_frame_indices)}")
    print(f"  └────────────────────────────────────────")

    print(f"\n  ┌─ Signal Statistics ───────────────────")
    stats = report.signal_stats
    if analysis.face_path_active:
        print(f"  │ Face CLIP:       mean={stats.get('clip_mean', 0):.3f}, max={stats.get('clip_max', 0):.3f}")
        print(f"  │ Face EfficientNet: mean={stats.get('effnet_mean', 0):.3f}, max={stats.get('effnet_max', 0):.3f}")
    print(f"  │ Frame CLIP:      mean={stats.get('frame_clip_mean', 0):.3f}, max={stats.get('frame_clip_max', 0):.3f}")
    print(f"  │ DCT Frequency:   score={stats.get('freq_score', 0):.3f}, HF suppression={stats.get('hf_suppression', 0):.1f}%")
    print(f"  │ Temporal:        {stats.get('temporal_summary', 'N/A')}")
    print(f"  └────────────────────────────────────────")

    print(f"\n  ┌─ Timings ─────────────────────────────")
    print(f"  │ Validation:      {timings.validation:.2f}s")
    print(f"  │ Scene detection:  {timings.scene_detection:.2f}s")
    print(f"  │ Frame selection:  {timings.frame_selection:.2f}s")
    print(f"  │ Face detection:   {timings.face_detection:.2f}s")
    print(f"  │ Face path:        {timings.face_path_inference:.2f}s")
    print(f"  │ Frame path:       {timings.frame_path_inference:.2f}s")
    print(f"  │ Temporal:         {timings.temporal_analysis:.2f}s")
    print(f"  │ Ensemble:         {timings.ensemble:.2f}s")
    print(f"  │ GradCAM:          {timings.gradcam:.2f}s")
    print(f"  │ Forensic views:   {timings.forensic_views:.2f}s")
    print(f"  │ ────────────────────────────────────")
    print(f"  │ TOTAL:            {timings.total:.2f}s")
    print(f"  └────────────────────────────────────────")

    print(f"\n  ┌─ Output Files ────────────────────────")
    print(f"  │ Output dir:      {report.output_dir}")
    print(f"  │ Heatmaps:        {len(report.heatmap_paths)}")
    for p in report.heatmap_paths[:5]:
        print(f"  │   {Path(p).name}")
    print(f"  │ Forensic views:  {len(report.forensic_view_paths)}")
    for p in report.forensic_view_paths[:5]:
        print(f"  │   {Path(p).name}")
    if report.timeline_path:
        print(f"  │ Timeline:        {Path(report.timeline_path).name}")
    print(f"  └────────────────────────────────────────")

    # ── Save JSON report ──
    report_path = Path(report.output_dir) / "report.json"
    report_json = report.summary()
    with open(report_path, "w") as f:
        json.dump(report_json, f, indent=2, default=str)
    print(f"\n  JSON report saved: {report_path}")

    # ── Per-frame score dump ──
    print(f"\n── Frame-level scores (top 10 by score) ──")
    sorted_frames = sorted(
        analysis.frame_analyses,
        key=lambda a: a.ensemble_score,
        reverse=True,
    )
    for fa in sorted_frames[:10]:
        signals = ", ".join(f"{s.name}={s.score:.3f}" for s in fa.signals)
        flag = " [FLAGGED]" if fa.flagged else ""
        print(f"  Frame {fa.frame_idx:4d} @ {fa.timestamp_sec:6.2f}s: "
              f"ensemble={fa.ensemble_score:.3f} ({signals}){flag}")

    # ── Cleanup ──
    pipeline.unload_models()

    print(f"\n{'=' * 65}")
    if analysis.verdict.value == "LIKELY_MANIPULATED":
        print(f"  RESULT: Video is likely AI-generated/manipulated ({analysis.video_score:.3f})")
    elif analysis.verdict.value == "SUSPICIOUS":
        print(f"  RESULT: Video is suspicious — manual review recommended ({analysis.video_score:.3f})")
    else:
        print(f"  RESULT: Video appears authentic ({analysis.video_score:.3f})")

    print(f"\n  Open {report.output_dir}/ to inspect heatmaps and forensic views.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sample_dir = Path("data/sample_videos")
        videos = list(sample_dir.glob("*.mp4")) + list(sample_dir.glob("*.mov"))
        if videos:
            video_path = str(videos[0])
            print(f"Auto-detected: {video_path}")
        else:
            print("Usage: python scripts/test_phase3.py <video_path>")
            sys.exit(1)
    else:
        video_path = sys.argv[1]

    main(video_path)