"""
DCT frequency analysis with forensic scoring.

Signals:
1. Band energy ratio (sharpness-normalized) — HF suppression vs expectation
2. Spectral smoothness — AI generators produce unnaturally smooth rolloff
3. Periodic artifacts — GAN upsampling checkerboard detection
4. Cross-channel correlation — AI generators correlate RGB HF; real sensors don't
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.fft import dctn

from verifi.detectors.base import DetectionResult


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class FrequencyAnalyzer:
    """
    Detect AI generation fingerprints in the frequency domain.

    Four independent signals scored 0-1 and combined into a final anomaly score.
    All scoring uses continuous functions (no step-function quantization).
    """

    def __init__(self, analysis_size: int = 256):
        self.analysis_size = analysis_size

    def analyze(
        self, image: np.ndarray, sharpness: float | None = None
    ) -> DetectionResult:
        """
        Analyze frequency characteristics of an image.

        Args:
            image: BGR numpy array (any size — will be resized internally).
            sharpness: Laplacian variance of the original image (optional).
                       Used to normalize band energy scoring against compression.

        Returns:
            DetectionResult with anomaly score and detailed metadata.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.analysis_size, self.analysis_size))
        gray_f = resized.astype(np.float32)

        dct = dctn(gray_f, norm="ortho")
        magnitude = np.abs(dct)

        band_score, band_meta = self._band_energy_analysis(magnitude, sharpness)
        smooth_score, smooth_meta = self._spectral_smoothness(magnitude)
        periodic_score, periodic_meta = self._periodic_artifacts(magnitude)
        channel_score, channel_meta = self._cross_channel_correlation(image)

        # DCT combined does NOT include ChCorr — it's exposed separately
        # so the ensemble can weight it independently.
        combined = (
            0.50 * band_score + 0.35 * smooth_score + 0.15 * periodic_score
        )
        combined = float(np.clip(combined, 0.0, 1.0))

        metadata = {
            "band_score": round(band_score, 4),
            "smooth_score": round(smooth_score, 4),
            "periodic_score": round(periodic_score, 4),
            "channel_corr_score": round(channel_score, 4),
            **band_meta,
            **smooth_meta,
            **periodic_meta,
            **channel_meta,
        }

        return DetectionResult(score=combined, metadata=metadata)

    def _band_energy_analysis(
        self, magnitude: np.ndarray, sharpness: float | None = None
    ) -> tuple[float, dict]:
        """
        Split spectrum into 3 bands and score HF suppression.

        When sharpness (Laplacian variance) is provided, scores relative to
        what's expected for that compression level. Without sharpness, uses
        a continuous sigmoid on the raw high_ratio.
        """
        h, w = magnitude.shape

        low_h, low_w = h // 8, w // 8
        mid_h, mid_w = h * 3 // 8, w * 3 // 8

        low_energy = np.sqrt(np.sum(magnitude[:low_h, :low_w] ** 2))
        mid_energy = np.sqrt(
            np.sum(magnitude[low_h:mid_h, :mid_w] ** 2)
            + np.sum(magnitude[:low_h, low_w:mid_w] ** 2)
        )
        high_energy = np.sqrt(
            np.sum(magnitude[mid_h:, :] ** 2)
            + np.sum(magnitude[:mid_h, mid_w:] ** 2)
        )

        total = low_energy + mid_energy + high_energy
        if total < 1e-6:
            return 0.0, {"low_ratio": 0, "mid_ratio": 0, "high_ratio": 0}

        low_ratio = float(low_energy / total)
        mid_ratio = float(mid_energy / total)
        high_ratio = float(high_energy / total)

        if sharpness is not None and sharpness > 0:
            expected_hf = 0.03 + 0.18 * min(sharpness / 800.0, 1.0)
            deviation = (expected_hf - high_ratio) / max(expected_hf, 0.01)
            score = _sigmoid(deviation * 4.0)
        else:
            score = _sigmoid(-(high_ratio - 0.10) * 20.0)

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
        score = _sigmoid(-(gradient_variation - 1.0) * 2.0)

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

        score = _sigmoid((peak_ratio - 0.02) * 100.0)

        return float(score), {"periodic_peaks": int(peaks), "peak_ratio": round(peak_ratio, 4)}

    def _cross_channel_correlation(self, image_bgr: np.ndarray) -> tuple[float, dict]:
        """
        Measure correlation of HF DCT coefficients across color channels.

        Real cameras have independent sensor noise per Bayer channel.
        AI generators produce correlated artifacts from a shared latent space.
        """
        sz = self.analysis_size
        channels = cv2.split(
            cv2.resize(image_bgr, (sz, sz)).astype(np.float32)
        )
        dcts = [dctn(c, norm="ortho") for c in channels]

        hf_start = sz // 4
        hf_dcts = [np.abs(d[hf_start:, hf_start:]).flatten() for d in dcts]

        corrs = []
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = hf_dcts[i], hf_dcts[j]
                if np.std(a) < 1e-8 or np.std(b) < 1e-8:
                    corrs.append(0.0)
                else:
                    corrs.append(float(np.corrcoef(a, b)[0, 1]))

        avg_corr = float(np.mean([abs(c) for c in corrs]))
        score = float(np.clip(avg_corr * 1.5, 0.0, 1.0))

        return score, {"channel_corr_mean": round(avg_corr, 4)}

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
