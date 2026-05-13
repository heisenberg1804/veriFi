"""
Noise residual analysis — compression-aware forensic detector.

Extracts noise residuals via median-filter subtraction and analyzes
statistical properties that differ between real H.264 video and
AI-generated content:

1. Spatial autocorrelation (INVERTED for H.264): real multi-encoded video
   has HIGH autocorrelation from deblocking filters; AI single-pass
   encoding from clean latent has LOWER autocorrelation.
2. Spectral entropy of residual DCT: AI noise has higher entropy (more
   uniform spectral energy); real noise has lower/more varied entropy.
3. Block variance consistency: AI noise is more spatially uniform.

No ML model required — pure statistical analysis.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.fft import dctn
from scipy.stats import entropy

from verifi.detectors.base import DetectionResult


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class NoiseResidualAnalyzer:
    """Analyze noise residuals to distinguish camera vs AI-generated content."""

    def __init__(self, analysis_size: int = 256, median_kernel: int = 3):
        self.analysis_size = analysis_size
        self.median_kernel = median_kernel

    def analyze(self, image: np.ndarray) -> DetectionResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.analysis_size, self.analysis_size))
        gray_f = gray.astype(np.float64)

        denoised = cv2.medianBlur(gray, self.median_kernel).astype(np.float64)
        residual = gray_f - denoised

        autocorr_score, autocorr_meta = self._spatial_autocorrelation(residual)
        entropy_score, entropy_meta = self._spectral_entropy(residual)
        variance_score, variance_meta = self._block_variance_consistency(
            residual,
        )

        combined = (
            0.45 * autocorr_score
            + 0.30 * entropy_score
            + 0.25 * variance_score
        )
        combined = float(np.clip(combined, 0.0, 1.0))

        metadata = {
            "noise_autocorr_score": round(autocorr_score, 4),
            "noise_entropy_score": round(entropy_score, 4),
            "noise_variance_score": round(variance_score, 4),
            **autocorr_meta,
            **entropy_meta,
            **variance_meta,
        }

        return DetectionResult(score=combined, metadata=metadata)

    def _spatial_autocorrelation(
        self, residual: np.ndarray,
    ) -> tuple[float, dict]:
        """
        H.264-aware autocorrelation analysis (polarity INVERTED).

        Real H.264 video: deblocking filters and multi-generation
        compression add spatially correlated artifacts → HIGH autocorr.
        AI video: single-pass encoding from clean latent → LOWER autocorr.

        Score: LOW autocorrelation → HIGH score (AI-like).
        """
        h, w = residual.shape
        if h < 4 or w < 4:
            return 0.5, {"noise_autocorr": 0.0}

        center = residual[1:-1, 1:-1]
        if np.std(center) < 1e-6:
            return 0.5, {"noise_autocorr": 0.0}

        right = residual[1:-1, 2:]
        down = residual[2:, 1:-1]

        corr_h = float(np.corrcoef(center.flatten(), right.flatten())[0, 1])
        corr_v = float(np.corrcoef(center.flatten(), down.flatten())[0, 1])
        avg_autocorr = (abs(corr_h) + abs(corr_v)) / 2.0

        # INVERTED: low autocorrelation = AI-like (high score)
        score = _sigmoid((avg_autocorr - 0.50) * -8.0)

        return float(score), {"noise_autocorr": round(avg_autocorr, 4)}

    def _spectral_entropy(
        self, residual: np.ndarray,
    ) -> tuple[float, dict]:
        """
        Spectral entropy of the noise residual's DCT.

        AI noise has higher normalized entropy (more uniform spectral
        energy across frequencies). Real noise has lower entropy (more
        structured, content-dependent frequency distribution).

        Score: HIGH entropy → HIGH score (AI-like).
        """
        dct = dctn(residual, norm="ortho")
        mag = np.abs(dct).flatten()
        mag_sum = mag.sum()
        if mag_sum < 1e-10:
            return 0.5, {"noise_spectral_entropy": 0.0}

        prob = mag / mag_sum
        ent = float(entropy(prob))
        max_ent = float(np.log(len(prob)))
        normalized = ent / max_ent if max_ent > 0 else 0.0

        score = _sigmoid((normalized - 0.965) * 200.0)

        return float(score), {
            "noise_spectral_entropy": round(normalized, 4),
        }

    def _block_variance_consistency(
        self, residual: np.ndarray,
    ) -> tuple[float, dict]:
        """
        Real noise: variance varies with content complexity.
        AI noise: more uniform variance across the frame.
        """
        h, w = residual.shape
        block_size = 32
        variances = []

        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                block = residual[y : y + block_size, x : x + block_size]
                variances.append(float(np.var(block)))

        if len(variances) < 4:
            return 0.5, {"noise_var_cv": 0.0}

        variances = np.array(variances)
        mean_var = float(np.mean(variances))
        if mean_var < 1e-8:
            return 0.5, {"noise_var_cv": 0.0}

        cv = float(np.std(variances) / mean_var)

        # Low CV = uniform noise = AI-like (high score)
        score = _sigmoid((cv - 0.8) * -3.0)

        return float(score), {"noise_var_cv": round(cv, 4)}
