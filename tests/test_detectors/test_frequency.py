"""Tests for DCT frequency analyzer."""
import numpy as np
from verifi.detectors.frequency import FrequencyAnalyzer


def test_frequency_returns_valid_score(dummy_face_crop):
    analyzer = FrequencyAnalyzer()
    result = analyzer.analyze(dummy_face_crop)
    assert 0.0 <= result.score <= 1.0


def test_frequency_metadata_keys(dummy_face_crop):
    analyzer = FrequencyAnalyzer()
    result = analyzer.analyze(dummy_face_crop)
    assert "hf_ratio" in result.metadata
    assert "hf_suppression_pct" in result.metadata
    assert "lf_energy" in result.metadata
    assert "hf_energy" in result.metadata


def test_frequency_spectrum_image(dummy_face_crop):
    analyzer = FrequencyAnalyzer()
    spectrum = analyzer.generate_spectrum_image(dummy_face_crop)
    assert spectrum.shape == (256, 256, 3)
    assert spectrum.dtype == np.uint8
