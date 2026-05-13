"""
Temporal consistency analysis using optical flow, SSIM, and flicker.

Multi-signal temporal analyzer that exploits AI video characteristics:

1. Flow field smoothness (CV of flow magnitude): AI video has unnaturally
   uniform motion fields — low coefficient of variation.
2. Inter-frame SSIM: AI generators produce temporally smooth output, so
   consecutive frames have higher structural similarity.
3. High-frequency temporal flicker: AI generators avoid temporal instability,
   so HF content changes less between frames.
4. Face-background flow divergence (legacy): face-swap deepfakes show motion
   discontinuities between face region and background.
"""
from __future__ import annotations

import cv2
import numpy as np

from verifi.detectors.base import DetectionResult


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _ssim_grayscale(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Compute SSIM between two grayscale uint8 images (no skimage dependency)."""
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    ksize = 11

    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)

    mu_a = cv2.GaussianBlur(a, (ksize, ksize), 1.5)
    mu_b = cv2.GaussianBlur(b, (ksize, ksize), 1.5)

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(a * a, (ksize, ksize), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(b * b, (ksize, ksize), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(a * b, (ksize, ksize), 1.5) - mu_ab

    numerator = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)

    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


class TemporalAnalyzer:
    """
    Multi-signal temporal consistency analyzer.

    Combines four sub-signals to detect AI-generated video:
    - Flow field CV (low = AI-like uniform motion)
    - Inter-frame SSIM (high = AI-like temporal smoothness)
    - HF flicker (low = AI-like temporal stability)
    - Face-bg divergence (high = face-swap motion discontinuity)
    """

    def __init__(self, divergence_threshold: float = 2.0, analysis_size: int = 256):
        self.divergence_threshold = divergence_threshold
        self.analysis_size = analysis_size

    def analyze_pair(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        face_bbox: tuple[int, int, int, int] | None = None,
    ) -> DetectionResult:
        """
        Compare temporal consistency between two consecutive frames.

        Returns a combined score from flow CV, SSIM, flicker, and divergence.
        """
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray_a, gray_b, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        flow_cv_score, flow_cv_meta = self._flow_field_cv(mag)
        ssim_score, ssim_meta = self._inter_frame_ssim(gray_a, gray_b)
        flicker_score, flicker_meta = self._hf_flicker(gray_a, gray_b)
        div_score, div_meta = self._face_bg_divergence(mag, frame_a.shape, face_bbox)

        combined = (
            0.30 * flow_cv_score
            + 0.35 * ssim_score
            + 0.25 * flicker_score
            + 0.10 * div_score
        )
        combined = float(np.clip(combined, 0.0, 1.0))

        metadata = {
            **flow_cv_meta,
            **ssim_meta,
            **flicker_meta,
            **div_meta,
            "temporal_flow_cv_score": round(flow_cv_score, 4),
            "temporal_ssim_score": round(ssim_score, 4),
            "temporal_flicker_score": round(flicker_score, 4),
            "temporal_divergence_score": round(div_score, 4),
        }

        return DetectionResult(score=combined, metadata=metadata)

    def _flow_field_cv(self, mag: np.ndarray) -> tuple[float, dict]:
        """Low CV of flow magnitude = unnaturally uniform motion = AI-like."""
        mean_mag = float(np.mean(mag))
        if mean_mag < 1e-6:
            return 0.5, {"flow_cv": 0.0, "flow_mean_mag": 0.0}

        cv = float(np.std(mag) / mean_mag)
        score = _sigmoid(-(cv - 1.2) * 3.0)

        return float(score), {
            "flow_cv": round(cv, 4),
            "flow_mean_mag": round(mean_mag, 4),
        }

    def _inter_frame_ssim(
        self, gray_a: np.ndarray, gray_b: np.ndarray,
    ) -> tuple[float, dict]:
        """High SSIM between consecutive frames = AI-like temporal smoothness."""
        size = self.analysis_size
        a = cv2.resize(gray_a, (size, size))
        b = cv2.resize(gray_b, (size, size))

        ssim_val = _ssim_grayscale(a, b)
        score = _sigmoid((ssim_val - 0.90) * 10.0)

        return float(score), {"frame_ssim": round(ssim_val, 4)}

    def _hf_flicker(
        self, gray_a: np.ndarray, gray_b: np.ndarray,
    ) -> tuple[float, dict]:
        """Low HF flicker between frames = AI-like temporal stability."""
        size = self.analysis_size
        a = cv2.resize(gray_a, (size, size)).astype(np.float64)
        b = cv2.resize(gray_b, (size, size)).astype(np.float64)

        blur_a = cv2.GaussianBlur(a, (5, 5), 1.0)
        blur_b = cv2.GaussianBlur(b, (5, 5), 1.0)

        hf_a = a - blur_a
        hf_b = b - blur_b

        flicker = float(np.mean(np.abs(hf_a - hf_b)))
        score = _sigmoid(-(flicker - 0.25) * 8.0)

        return float(score), {"hf_flicker": round(flicker, 4)}

    def _face_bg_divergence(
        self,
        mag: np.ndarray,
        frame_shape: tuple,
        face_bbox: tuple[int, int, int, int] | None,
    ) -> tuple[float, dict]:
        """Face-vs-background flow divergence for face-swap detection."""
        h, w = mag.shape
        if face_bbox:
            x, y, bw, bh = face_bbox
            face_mask = np.zeros((h, w), dtype=bool)
            face_mask[y:y + bh, x:x + bw] = True
        else:
            ch, cw = h // 5, w // 5
            face_mask = np.zeros((h, w), dtype=bool)
            face_mask[ch * 2:ch * 3, cw * 2:cw * 3] = True

        bg_mask = ~face_mask

        face_flow_mean = float(mag[face_mask].mean()) if face_mask.any() else 0.0
        bg_flow_mean = float(mag[bg_mask].mean()) if bg_mask.any() else 0.0
        bg_flow_std = float(mag[bg_mask].std()) if bg_mask.any() else 1.0

        if bg_flow_std > 0.01:
            divergence = abs(face_flow_mean - bg_flow_mean) / bg_flow_std
        else:
            divergence = 0.0

        score = min(1.0, divergence / (self.divergence_threshold * 2))

        return float(score), {
            "face_flow_mean": round(face_flow_mean, 4),
            "bg_flow_mean": round(bg_flow_mean, 4),
            "divergence_sigma": round(divergence, 4),
        }
