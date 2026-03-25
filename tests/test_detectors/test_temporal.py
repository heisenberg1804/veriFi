"""Tests for temporal consistency analyzer."""
from verifi.detectors.temporal import TemporalAnalyzer


def test_temporal_returns_valid_score(dummy_frame_pair):
    analyzer = TemporalAnalyzer()
    frame_a, frame_b = dummy_frame_pair
    result = analyzer.analyze_pair(frame_a, frame_b)
    assert 0.0 <= result.score <= 1.0


def test_temporal_identical_frames():
    """Identical frames should produce low divergence."""
    import numpy as np
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    analyzer = TemporalAnalyzer()
    result = analyzer.analyze_pair(frame, frame.copy())
    assert result.score < 0.1


def test_temporal_metadata_keys(dummy_frame_pair):
    analyzer = TemporalAnalyzer()
    frame_a, frame_b = dummy_frame_pair
    result = analyzer.analyze_pair(frame_a, frame_b)
    assert "face_flow_mean" in result.metadata
    assert "bg_flow_mean" in result.metadata
    assert "divergence_sigma" in result.metadata
