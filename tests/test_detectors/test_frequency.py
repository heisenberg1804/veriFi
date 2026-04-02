"""
Updated tests for multi-band DCT frequency analyzer.

Save to: tests/test_detectors/test_frequency.py (REPLACES existing file)
"""
import cv2
import numpy as np

from verifi.detectors.frequency import FrequencyAnalyzer


def test_frequency_returns_valid_score(dummy_face_crop):
    analyzer = FrequencyAnalyzer()
    result = analyzer.analyze(dummy_face_crop)
    assert 0.0 <= result.score <= 1.0


def test_frequency_metadata_keys(dummy_face_crop):
    analyzer = FrequencyAnalyzer()
    result = analyzer.analyze(dummy_face_crop)
    assert "band_score" in result.metadata
    assert "smooth_score" in result.metadata
    assert "periodic_score" in result.metadata
    assert "low_ratio" in result.metadata
    assert "mid_ratio" in result.metadata
    assert "high_ratio" in result.metadata


def test_frequency_spectrum_image(dummy_face_crop):
    analyzer = FrequencyAnalyzer()
    spectrum = analyzer.generate_spectrum_image(dummy_face_crop)
    assert spectrum.shape == (256, 256, 3)
    assert spectrum.dtype == np.uint8


def test_smooth_image_scores_high():
    """A very smooth (blurry) image should score high — looks AI-generated."""
    analyzer = FrequencyAnalyzer()
    # Create a smooth gradient image (minimal HF content)
    smooth = np.zeros((224, 224, 3), dtype=np.uint8)
    for y in range(224):
        smooth[y, :] = int(y / 224 * 255)
    # Apply heavy blur to remove any remaining edges
    smooth = cv2.GaussianBlur(smooth, (31, 31), 10)
    result = analyzer.analyze(smooth)
    assert result.score > 0.4, f"Smooth image scored {result.score}, expected > 0.4"


def test_noisy_image_scores_low():
    """Random noise has lots of HF content — should look authentic."""
    analyzer = FrequencyAnalyzer()
    noisy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = analyzer.analyze(noisy)
    assert result.score < 0.5, f"Noisy image scored {result.score}, expected < 0.5"


def test_band_ratios_sum_to_one(dummy_face_crop):
    """Low + mid + high ratios should approximately sum to 1."""
    analyzer = FrequencyAnalyzer()
    result = analyzer.analyze(dummy_face_crop)
    total = (result.metadata["low_ratio"] +
             result.metadata["mid_ratio"] +
             result.metadata["high_ratio"])
    assert abs(total - 1.0) < 0.01, f"Band ratios sum to {total}, expected ~1.0"
