"""
Face alignment utilities.

Note: Alignment is integrated into face_detector.py's FaceDetectionPipeline.
This module exists for standalone use if needed.
"""
import cv2
import numpy as np


def align_face_by_eyes(
    image: np.ndarray,
    left_eye: tuple[float, float],
    right_eye: tuple[float, float],
) -> np.ndarray:
    """Rotate image so the eye line is horizontal."""
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = float(np.degrees(np.arctan2(dy, dx)))
    center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    h, w = image.shape[:2]
    return cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
