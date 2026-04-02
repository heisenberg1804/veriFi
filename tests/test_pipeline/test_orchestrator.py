
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FILE 3: tests/test_pipeline/test_orchestrator.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""Tests for pipeline orchestrator."""

from verifi.pipeline.orchestrator import ForensicReport, StageTimings, VeriFiPipeline


def test_stage_timings_to_dict():
    t = StageTimings(validation=0.5, total=3.2)
    d = t.to_dict()
    assert d["validation"] == 0.5
    assert d["total"] == 3.2


def test_forensic_report_summary():
    from verifi.ensemble.aggregator import ManipulationType, Verdict, VideoAnalysis

    analysis = VideoAnalysis(
        video_score=0.85,
        verdict=Verdict.LIKELY_MANIPULATED,
        manipulation_type=ManipulationType.FULL_SYNTHESIS,
        dominant_path="frame",
    )
    report = ForensicReport(
        video_metadata={"filename": "test.mp4", "duration_sec": 10},
        analysis=analysis,
        signal_stats={"freq_score": 0.7},
        heatmap_paths=["a.png", "b.png"],
        timings=StageTimings(total=5.0),
    )
    summary = report.summary()
    assert summary["video"] == "test.mp4"
    assert summary["verdict"] == "LIKELY_MANIPULATED"
    assert summary["heatmaps_generated"] == 2


def test_pipeline_instantiation():
    """Pipeline should instantiate without loading models."""
    from verifi.config import AppConfig
    config = AppConfig()
    pipeline = VeriFiPipeline(config)
    assert not pipeline._models_loaded
