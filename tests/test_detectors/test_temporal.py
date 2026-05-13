"""Tests for temporal consistency analyzer."""
import numpy as np

from verifi.detectors.temporal import TemporalAnalyzer, _ssim_grayscale


def test_temporal_returns_valid_score(dummy_frame_pair):
    analyzer = TemporalAnalyzer()
    frame_a, frame_b = dummy_frame_pair
    result = analyzer.analyze_pair(frame_a, frame_b)
    assert 0.0 <= result.score <= 1.0


def test_temporal_identical_frames_high_ssim():
    """Identical frames → SSIM=1.0, flicker=0.0, both AI-like signals."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    analyzer = TemporalAnalyzer()
    result = analyzer.analyze_pair(frame, frame.copy())
    assert result.metadata["frame_ssim"] > 0.99
    assert result.metadata["hf_flicker"] < 0.01
    assert result.metadata["divergence_sigma"] < 0.01


def test_temporal_different_frames_lower_ssim():
    """Very different frames should have low SSIM."""
    frame_a = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_b = np.full((480, 640, 3), 255, dtype=np.uint8)
    analyzer = TemporalAnalyzer()
    result = analyzer.analyze_pair(frame_a, frame_b)
    assert result.metadata["frame_ssim"] < 0.1


def test_temporal_metadata_keys(dummy_frame_pair):
    analyzer = TemporalAnalyzer()
    frame_a, frame_b = dummy_frame_pair
    result = analyzer.analyze_pair(frame_a, frame_b)
    assert "face_flow_mean" in result.metadata
    assert "bg_flow_mean" in result.metadata
    assert "divergence_sigma" in result.metadata
    assert "flow_cv" in result.metadata
    assert "frame_ssim" in result.metadata
    assert "hf_flicker" in result.metadata


def test_ssim_identical_images():
    img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    assert _ssim_grayscale(img, img.copy()) > 0.99


def test_ssim_different_images():
    a = np.zeros((256, 256), dtype=np.uint8)
    b = np.full((256, 256), 255, dtype=np.uint8)
    assert _ssim_grayscale(a, b) < 0.1
