"""Standalone quality filter utilities."""
import cv2
import numpy as np


def compute_blur_score(frame: np.ndarray) -> float:
    """Laplacian variance — higher means sharper."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_sharp(frame: np.ndarray, threshold: float = 100.0) -> bool:
    """Quick check if a frame passes the blur threshold."""
    return compute_blur_score(frame) >= threshold
