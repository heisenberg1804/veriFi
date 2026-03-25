"""Temporal consistency analysis using optical flow."""
import cv2
import numpy as np

from verifi.detectors.base import DetectionResult


class TemporalAnalyzer:
    """
    Check temporal consistency between adjacent frames.
    Deepfakes often show motion discontinuities in the face
    region that differ from natural background motion.
    """

    def __init__(self, divergence_threshold: float = 2.0):
        self.divergence_threshold = divergence_threshold

    def analyze_pair(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        face_bbox: tuple[int, int, int, int] | None = None,
    ) -> DetectionResult:
        """
        Compare optical flow consistency between two frames.

        Args:
            frame_a, frame_b: BGR frames (same resolution).
            face_bbox: (x, y, w, h) of face region. If None, uses center crop.

        Returns:
            DetectionResult with divergence score.
        """
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray_a, gray_b, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

        # Compute flow magnitude
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Split into face region and background
        h, w = mag.shape
        if face_bbox:
            x, y, bw, bh = face_bbox
            face_mask = np.zeros((h, w), dtype=bool)
            face_mask[y:y+bh, x:x+bw] = True
        else:
            # Default: center 40% as face proxy
            ch, cw = h // 5, w // 5
            face_mask = np.zeros((h, w), dtype=bool)
            face_mask[ch*2:ch*3, cw*2:cw*3] = True

        bg_mask = ~face_mask

        face_flow_mean = mag[face_mask].mean() if face_mask.any() else 0
        bg_flow_mean = mag[bg_mask].mean() if bg_mask.any() else 0
        bg_flow_std = mag[bg_mask].std() if bg_mask.any() else 1.0

        # Divergence: how many std devs does face flow differ from background
        if bg_flow_std > 0.01:
            divergence = abs(face_flow_mean - bg_flow_mean) / bg_flow_std
        else:
            divergence = 0.0

        # Map to [0, 1] score
        score = min(1.0, divergence / (self.divergence_threshold * 2))

        return DetectionResult(
            score=score,
            metadata={
                "face_flow_mean": float(face_flow_mean),
                "bg_flow_mean": float(bg_flow_mean),
                "divergence_sigma": float(divergence),
            },
        )
