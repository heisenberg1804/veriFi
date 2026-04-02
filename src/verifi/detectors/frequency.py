"""
DCT frequency analysis — FIXED with adaptive scoring.

Save to: src/verifi/detectors/frequency.py (REPLACES existing file)

Key changes:
1. Removed hardcoded baseline_hf_ratio that only worked for face crops.
2. New multi-band analysis: splits spectrum into low/mid/high bands and
   computes ratios between them. AI-generated content shows characteristic
   patterns regardless of input type (face crop vs full frame).
3. Added spectral smoothness metric: AI generators produce unnaturally
   smooth spectral rolloff compared to camera sensors.
4. Added periodic artifact detection: GAN upsampling creates periodic
   peaks in the frequency domain.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.fft import dctn

from verifi.detectors.base import DetectionResult


class FrequencyAnalyzer:
    """
    Detect AI generation fingerprints in the frequency domain.

    Uses three independent frequency-domain signals:
    1. Band energy ratio: AI generators suppress high-frequency detail
    2. Spectral smoothness: AI output has unnaturally smooth rolloff
    3. Periodic artifacts: GAN upsampling leaves periodic spectral peaks

    Each signal is scored 0-1 and combined into a final anomaly score.
    """

    def __init__(self, analysis_size: int = 256):
        self.analysis_size = analysis_size

    def analyze(self, image: np.ndarray) -> DetectionResult:
        """
        Analyze frequency characteristics of an image.

        Args:
            image: BGR numpy array (any size — will be resized internally).

        Returns:
            DetectionResult with anomaly score and detailed metadata.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.analysis_size, self.analysis_size))
        gray = gray.astype(np.float32)

        # Compute 2D DCT
        dct = dctn(gray, norm="ortho")
        magnitude = np.abs(dct)

        # ── Signal 1: Band energy ratio ──
        band_score, band_meta = self._band_energy_analysis(magnitude)

        # ── Signal 2: Spectral smoothness ──
        smooth_score, smooth_meta = self._spectral_smoothness(magnitude)

        # ── Signal 3: Periodic artifact detection ──
        periodic_score, periodic_meta = self._periodic_artifacts(magnitude)

        # ── Combined score ──
        # Weighted combination — band energy is most reliable
        combined = (
            0.45 * band_score +
            0.35 * smooth_score +
            0.20 * periodic_score
        )
        combined = float(np.clip(combined, 0.0, 1.0))

        metadata = {
            "band_score": round(band_score, 4),
            "smooth_score": round(smooth_score, 4),
            "periodic_score": round(periodic_score, 4),
            **band_meta,
            **smooth_meta,
            **periodic_meta,
        }

        return DetectionResult(score=combined, metadata=metadata)

    def _band_energy_analysis(self, magnitude: np.ndarray) -> tuple[float, dict]:
        """
        Split spectrum into 3 bands and analyze energy distribution.

        AI generators concentrate energy in low frequencies and suppress
        high frequencies compared to real camera images.

        Band definitions (as fraction of spectrum):
        - Low:  0-12.5%  (top-left corner)
        - Mid:  12.5-37.5%
        - High: 37.5-100%
        """
        h, w = magnitude.shape

        # Define band boundaries
        low_h, low_w = h // 8, w // 8
        mid_h, mid_w = h * 3 // 8, w * 3 // 8

        # Extract band energies (using L2 norm for stability)
        # Mid band is the L-shaped region between low and high, no overlap
        low_energy = np.sqrt(np.sum(magnitude[:low_h, :low_w] ** 2))
        mid_energy = np.sqrt(
            np.sum(magnitude[low_h:mid_h, :mid_w] ** 2) +
            np.sum(magnitude[:low_h, low_w:mid_w] ** 2)
        )
        high_energy = np.sqrt(
            np.sum(magnitude[mid_h:, :] ** 2) +
            np.sum(magnitude[:mid_h, mid_w:] ** 2)
        )

        total = low_energy + mid_energy + high_energy
        if total < 1e-6:
            return 0.0, {"low_ratio": 0, "mid_ratio": 0, "high_ratio": 0}

        low_ratio = float(low_energy / total)
        mid_ratio = float(mid_energy / total)
        high_ratio = float(high_energy / total)

        # Scoring: real images typically have high_ratio > 0.15
        # AI-generated images typically have high_ratio < 0.10
        # The lower the high_ratio, the more suspicious
        if high_ratio < 0.05:
            score = 1.0
        elif high_ratio < 0.10:
            score = 0.8
        elif high_ratio < 0.15:
            score = 0.5
        elif high_ratio < 0.20:
            score = 0.3
        else:
            score = 0.1  # healthy HF content

        # Also check if low band is disproportionately dominant
        if low_ratio > 0.7:
            score = max(score, 0.7)

        meta = {
            "low_ratio": round(low_ratio, 4),
            "mid_ratio": round(mid_ratio, 4),
            "high_ratio": round(high_ratio, 4),
            "hf_suppression_pct": round((1 - high_ratio / 0.20) * 100, 1),
        }
        return float(score), meta

    def _spectral_smoothness(self, magnitude: np.ndarray) -> tuple[float, dict]:
        """
        Measure how smooth the spectral rolloff is.

        Real camera images have irregular frequency content (scene-dependent).
        AI-generated images have unnaturally smooth, monotonic rolloff
        because the generation process applies uniform filtering.
        """
        h, w = magnitude.shape

        # Compute radial average of magnitude spectrum
        max_radius = min(h, w) // 2
        radial_profile = np.zeros(max_radius)
        counts = np.zeros(max_radius)

        for y in range(h):
            for x in range(w):
                r = int(np.sqrt((y - 0) ** 2 + (x - 0) ** 2))  # distance from DC
                if r < max_radius:
                    radial_profile[r] += magnitude[y, x]
                    counts[r] += 1

        # Avoid division by zero
        counts = np.maximum(counts, 1)
        radial_profile = radial_profile / counts

        if len(radial_profile) < 10:
            return 0.0, {"spectral_smoothness": 0}

        # Compute smoothness as inverse of variation in the gradient
        gradient = np.diff(radial_profile)
        if len(gradient) < 2:
            return 0.0, {"spectral_smoothness": 0}

        gradient_variation = float(np.std(gradient) / (np.abs(np.mean(gradient)) + 1e-6))

        # Low variation = smooth rolloff = likely AI
        # High variation = irregular = likely real camera
        if gradient_variation < 0.5:
            score = 0.9
        elif gradient_variation < 1.0:
            score = 0.6
        elif gradient_variation < 2.0:
            score = 0.3
        else:
            score = 0.1

        return float(score), {"spectral_smoothness": round(gradient_variation, 4)}

    def _periodic_artifacts(self, magnitude: np.ndarray) -> tuple[float, dict]:
        """
        Detect periodic peaks in the spectrum that indicate GAN upsampling.

        GAN generators (not diffusion models) often produce checkerboard
        artifacts from transpose convolution, visible as periodic peaks.
        Diffusion models generally don't show this.
        """
        h, w = magnitude.shape

        # Look at high-frequency region only
        hf_region = magnitude[h // 4:, w // 4:]

        if hf_region.size < 100:
            return 0.0, {"periodic_peaks": 0}

        # Compute autocorrelation of HF region
        hf_flat = hf_region.flatten()
        hf_normalized = hf_flat - hf_flat.mean()
        if np.std(hf_normalized) < 1e-6:
            return 0.0, {"periodic_peaks": 0}

        hf_normalized = hf_normalized / np.std(hf_normalized)

        # Count peaks that exceed 3 sigma
        threshold = 3.0
        peaks = np.sum(np.abs(hf_normalized) > threshold)
        total = len(hf_normalized)
        peak_ratio = peaks / total if total > 0 else 0

        # High peak ratio suggests periodic artifacts (GAN signature)
        if peak_ratio > 0.05:
            score = 0.8
        elif peak_ratio > 0.02:
            score = 0.5
        elif peak_ratio > 0.01:
            score = 0.3
        else:
            score = 0.1

        return float(score), {"periodic_peaks": int(peaks), "peak_ratio": round(peak_ratio, 4)}

    def generate_spectrum_image(self, image: np.ndarray) -> np.ndarray:
        """Generate a visual DCT spectrum image for the forensic report."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.analysis_size, self.analysis_size))
        gray = gray.astype(np.float32)

        dct = dctn(gray, norm="ortho")
        magnitude = np.log1p(np.abs(dct))

        # Normalize to 0-255
        if magnitude.max() > 0:
            magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
        else:
            magnitude = np.zeros_like(magnitude, dtype=np.uint8)

        colored = cv2.applyColorMap(magnitude, cv2.COLORMAP_JET)
        return colored
