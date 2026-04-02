
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FILE 1: tests/test_ensemble/test_aggregator.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""Tests for dual-path ensemble aggregator."""

from verifi.ensemble.aggregator import (
    FaceFrameAnalysis,
    FullFrameAnalysis,
    ManipulationType,
    SignalScore,
    Verdict,
    aggregate,
    compute_signal_stats,
)


def _make_face_analysis(clip=0.8, effnet=0.7, dct=0.5, face_id=0, frame_idx=0):
    return FaceFrameAnalysis(
        face_id=face_id,
        frame_idx=frame_idx,
        timestamp_sec=frame_idx / 30.0,
        signals=[
            SignalScore(name="clip", score=clip),
            SignalScore(name="effnet", score=effnet),
            SignalScore(name="dct", score=dct),
        ],
    )


def _make_frame_analysis(clip=0.7, dct=0.6, temporal=None, frame_idx=0):
    signals = [
        SignalScore(name="clip", score=clip),
        SignalScore(name="dct", score=dct, metadata={"hf_suppression_pct": dct * 50}),
    ]
    if temporal is not None:
        signals.append(SignalScore(name="temporal", score=temporal))
    return FullFrameAnalysis(
        frame_idx=frame_idx,
        timestamp_sec=frame_idx / 30.0,
        signals=signals,
    )


def test_high_face_score_yields_manipulated():
    faces = [_make_face_analysis(clip=0.9, effnet=0.85, dct=0.7, frame_idx=i) for i in range(10)]
    frames = [_make_frame_analysis(clip=0.3, dct=0.2, frame_idx=i) for i in range(10)]
    result = aggregate(faces, frames)
    assert result.verdict == Verdict.LIKELY_MANIPULATED
    assert result.dominant_path == "face"


def test_high_frame_score_yields_manipulated():
    faces = []  # no faces detected
    frames = [_make_frame_analysis(clip=0.85, dct=0.8, frame_idx=i) for i in range(10)]
    result = aggregate(faces, frames)
    assert result.verdict == Verdict.LIKELY_MANIPULATED
    assert result.dominant_path == "frame"
    assert not result.face_path_active


def test_low_scores_yield_authentic():
    faces = [_make_face_analysis(clip=0.1, effnet=0.15, dct=0.05, frame_idx=i) for i in range(5)]
    frames = [_make_frame_analysis(clip=0.1, dct=0.1, frame_idx=i) for i in range(5)]
    result = aggregate(faces, frames)
    assert result.verdict == Verdict.LIKELY_AUTHENTIC


def test_medium_scores_yield_suspicious():
    faces = [_make_face_analysis(clip=0.5, effnet=0.4, dct=0.3, frame_idx=i) for i in range(5)]
    frames = [_make_frame_analysis(clip=0.45, dct=0.35, frame_idx=i) for i in range(5)]
    result = aggregate(faces, frames)
    assert result.verdict == Verdict.SUSPICIOUS


def test_full_synthesis_detected():
    """High frame scores + no/low face scores → full_synthesis."""
    faces = []
    frames = [_make_frame_analysis(clip=0.8, dct=0.7, frame_idx=i) for i in range(10)]
    result = aggregate(faces, frames)
    assert result.manipulation_type == ManipulationType.FULL_SYNTHESIS


def test_face_swap_detected():
    """High face CLIP + high EfficientNet → face_swap."""
    faces = [_make_face_analysis(clip=0.85, effnet=0.75, dct=0.5, frame_idx=i) for i in range(10)]
    frames = [_make_frame_analysis(clip=0.3, dct=0.2, frame_idx=i) for i in range(10)]
    result = aggregate(faces, frames)
    assert result.manipulation_type == ManipulationType.FACE_SWAP


def test_empty_inputs():
    result = aggregate([], [])
    assert result.verdict == Verdict.LIKELY_AUTHENTIC
    assert result.video_score == 0.0


def test_flagged_indices_populated():
    frames = [_make_frame_analysis(clip=0.8, dct=0.7, frame_idx=i) for i in range(5)]
    result = aggregate([], frames)
    assert len(result.flagged_frame_indices) > 0


def test_signal_stats_keys():
    frames = [_make_frame_analysis(clip=0.5, dct=0.4, frame_idx=i) for i in range(5)]
    result = aggregate([], frames)
    stats = compute_signal_stats(result)
    assert "freq_score" in stats
    assert "hf_suppression" in stats
    assert "frame_clip_mean" in stats
    assert "temporal_summary" in stats


def test_to_dict():
    frames = [_make_frame_analysis(clip=0.5, dct=0.4, frame_idx=i) for i in range(3)]
    result = aggregate([], frames)
    d = result.to_dict()
    assert "video_score" in d
    assert "verdict" in d
    assert "face_path" in d
    assert "frame_path" in d
