"""Tests for dual-path ensemble aggregator."""

from verifi.ensemble.aggregator import (
    EnsembleWeights,
    FaceFrameAnalysis,
    FullFrameAnalysis,
    ManipulationType,
    SignalScore,
    Verdict,
    _apply_agreement_bonus,
    _compute_confidence,
    aggregate,
    compute_signal_stats,
)


def _make_face_analysis(
    clip=0.8, effnet=0.7, dct=0.5, noise_residual=0.5,
    channel_corr=0.5, face_id=0, frame_idx=0,
):
    return FaceFrameAnalysis(
        face_id=face_id,
        frame_idx=frame_idx,
        timestamp_sec=frame_idx / 30.0,
        signals=[
            SignalScore(name="clip", score=clip),
            SignalScore(name="effnet", score=effnet),
            SignalScore(name="dct", score=dct),
            SignalScore(name="noise_residual", score=noise_residual),
            SignalScore(name="channel_corr", score=channel_corr),
        ],
    )


def _make_frame_analysis(
    clip=0.7, dct=0.6, noise_residual=0.5, channel_corr=0.5,
    temporal=None, frame_idx=0,
):
    signals = [
        SignalScore(name="clip", score=clip),
        SignalScore(name="dct", score=dct, metadata={"hf_suppression_pct": dct * 50}),
        SignalScore(name="noise_residual", score=noise_residual),
        SignalScore(name="channel_corr", score=channel_corr),
    ]
    if temporal is not None:
        signals.append(SignalScore(name="temporal", score=temporal))
    return FullFrameAnalysis(
        frame_idx=frame_idx,
        timestamp_sec=frame_idx / 30.0,
        signals=signals,
    )


def test_high_face_score_yields_manipulated():
    faces = [
        _make_face_analysis(clip=0.9, effnet=0.85, dct=0.8, noise_residual=0.8, frame_idx=i)
        for i in range(10)
    ]
    frames = [
        _make_frame_analysis(clip=0.3, dct=0.2, noise_residual=0.2, frame_idx=i)
        for i in range(10)
    ]
    result = aggregate(faces, frames)
    assert result.verdict == Verdict.LIKELY_MANIPULATED
    assert result.dominant_path == "face"


def test_high_frame_score_yields_manipulated():
    faces = []
    frames = [
        _make_frame_analysis(clip=0.85, dct=0.85, noise_residual=0.85, frame_idx=i)
        for i in range(10)
    ]
    result = aggregate(faces, frames)
    assert result.verdict == Verdict.LIKELY_MANIPULATED
    assert result.dominant_path == "frame"
    assert not result.face_path_active


def test_low_scores_yield_authentic():
    faces = [
        _make_face_analysis(clip=0.1, effnet=0.15, dct=0.05, noise_residual=0.1, frame_idx=i)
        for i in range(5)
    ]
    frames = [
        _make_frame_analysis(clip=0.1, dct=0.1, noise_residual=0.1, frame_idx=i)
        for i in range(5)
    ]
    result = aggregate(faces, frames)
    assert result.verdict == Verdict.LIKELY_AUTHENTIC


def test_medium_scores_yield_suspicious():
    faces = [
        _make_face_analysis(
            clip=0.5, effnet=0.4, dct=0.5, noise_residual=0.5,
            channel_corr=0.5, frame_idx=i,
        ) for i in range(5)
    ]
    frames = [
        _make_frame_analysis(
            clip=0.5, dct=0.5, noise_residual=0.5,
            channel_corr=0.5, frame_idx=i,
        ) for i in range(5)
    ]
    result = aggregate(faces, frames)
    assert result.verdict == Verdict.SUSPICIOUS


def test_full_synthesis_detected():
    """High frame forensic signals + no faces → full_synthesis."""
    faces = []
    frames = [
        _make_frame_analysis(clip=0.8, dct=0.7, noise_residual=0.7, frame_idx=i)
        for i in range(10)
    ]
    result = aggregate(faces, frames)
    assert result.manipulation_type == ManipulationType.FULL_SYNTHESIS


def test_face_swap_detected():
    """High face CLIP + high EfficientNet → face_swap."""
    faces = [
        _make_face_analysis(
            clip=0.85, effnet=0.75, dct=0.5, noise_residual=0.5,
            channel_corr=0.5, frame_idx=i,
        ) for i in range(10)
    ]
    frames = [
        _make_frame_analysis(
            clip=0.3, dct=0.2, noise_residual=0.2,
            channel_corr=0.3, frame_idx=i,
        ) for i in range(10)
    ]
    result = aggregate(faces, frames)
    assert result.manipulation_type == ManipulationType.FACE_SWAP


def test_empty_inputs():
    result = aggregate([], [])
    assert result.verdict == Verdict.LIKELY_AUTHENTIC
    assert result.video_score == 0.0


def test_flagged_indices_populated():
    frames = [
        _make_frame_analysis(clip=0.8, dct=0.7, noise_residual=0.7, frame_idx=i)
        for i in range(5)
    ]
    result = aggregate([], frames)
    assert len(result.flagged_frame_indices) > 0


def test_signal_stats_keys():
    frames = [
        _make_frame_analysis(clip=0.5, dct=0.4, noise_residual=0.4, frame_idx=i)
        for i in range(5)
    ]
    result = aggregate([], frames)
    stats = compute_signal_stats(result)
    assert "freq_score" in stats
    assert "hf_suppression" in stats
    assert "frame_clip_mean" in stats
    assert "temporal_summary" in stats
    assert "noise_residual_mean" in stats


def test_to_dict():
    frames = [
        _make_frame_analysis(clip=0.5, dct=0.4, noise_residual=0.4, frame_idx=i)
        for i in range(3)
    ]
    result = aggregate([], frames)
    d = result.to_dict()
    assert "video_score" in d
    assert "verdict" in d
    assert "confidence" in d
    assert "face_path" in d
    assert "frame_path" in d


def test_signal_agreement_bonus():
    """Both DCT and noise_residual high → score gets a bonus."""
    signals = [
        SignalScore(name="dct", score=0.7),
        SignalScore(name="noise_residual", score=0.7),
    ]
    weights = EnsembleWeights()
    boosted = _apply_agreement_bonus(signals, 0.50, weights)
    assert boosted > 0.50


def test_signal_disagreement_penalty():
    """DCT high but noise_residual low → score gets penalized."""
    signals = [
        SignalScore(name="dct", score=0.7),
        SignalScore(name="noise_residual", score=0.3),
    ]
    weights = EnsembleWeights()
    penalized = _apply_agreement_bonus(signals, 0.50, weights)
    assert penalized < 0.50


def test_confidence_at_extremes():
    """Scores far from thresholds should have high confidence."""
    weights = EnsembleWeights()
    assert _compute_confidence(0.10, weights) > 0.5
    assert _compute_confidence(0.90, weights) > 0.5


def test_confidence_near_threshold():
    """Scores in middle of SUSPICIOUS band should have low confidence."""
    weights = EnsembleWeights()
    mid = (weights.suspicious_threshold + weights.manipulated_threshold) / 2.0
    assert _compute_confidence(mid, weights) <= 0.5
