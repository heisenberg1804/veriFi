"""
JPEG ghost analysis — forensic detector exploiting compression history.

Real video frames in H.264 preserve block-level compression artifacts.
Re-compressing at various JPEG quality levels reveals a characteristic
error "dip" near the original compression quality.

AI-generated frames lack this compression history — their error curve
is smooth and monotonically decreasing.
"""
from __future__ import annotations

import cv2
import numpy as np

from verifi.detectors.base import DetectionResult

QUALITY_LEVELS = [50, 60, 70, 80, 90, 95]


class JPEGGhostAnalyzer:
    """Detect compression history via JPEG re-compression error analysis."""

    def __init__(self, analysis_size: int = 256):
        self.analysis_size = analysis_size

    def analyze(self, image: np.ndarray) -> DetectionResult:
        resized = cv2.resize(image, (self.analysis_size, self.analysis_size))

        errors = []
        for q in QUALITY_LEVELS:
            _, buf = cv2.imencode(
                ".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, q]
            )
            recompressed = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            mse = float(
                np.mean(
                    (resized.astype(np.float64) - recompressed.astype(np.float64)) ** 2
                )
            )
            errors.append(mse)

        errors_arr = np.array(errors)
        gradient = np.diff(errors_arr)

        sign_changes = int(np.sum(np.diff(np.sign(gradient)) != 0))
        neg_frac = float(np.sum(gradient < 0) / len(gradient)) if len(gradient) > 0 else 0.0
        error_range = float(errors_arr.max() - errors_arr.min()) if len(errors_arr) > 0 else 0.0

        # Monotonically decreasing error + few sign changes = no compression history = AI
        if neg_frac > 0.8 and sign_changes <= 1:
            score = 0.5 + 0.5 * neg_frac
        elif sign_changes >= 2:
            score = max(0.1, 0.4 - sign_changes * 0.1)
        else:
            score = 0.45

        score = float(np.clip(score, 0.0, 1.0))

        metadata = {
            "jpeg_ghost_score": round(score, 4),
            "jpeg_monotonicity": round(neg_frac, 4),
            "jpeg_sign_changes": sign_changes,
            "jpeg_error_range": round(error_range, 2),
            "jpeg_errors": [round(e, 2) for e in errors],
        }

        return DetectionResult(score=score, metadata=metadata)
