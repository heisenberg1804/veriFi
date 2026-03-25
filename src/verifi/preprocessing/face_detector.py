"""
Face detection, alignment, and cross-frame tracking.

Save to: src/verifi/preprocessing/face_detector.py
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import structlog
import torch
from facenet_pytorch import MTCNN

logger = structlog.get_logger()


@dataclass
class FaceBBox:
    """Bounding box for a detected face."""
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def area(self) -> int:
        return self.w * self.h

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


@dataclass
class DetectedFace:
    """A detected and cropped face from a frame."""
    face_id: int                  # Tracked ID across frames
    bbox: FaceBBox                # Bounding box in original frame coords
    crop: np.ndarray              # Aligned, cropped face (BGR, target_size)
    crop_raw: np.ndarray          # Cropped face without resize (BGR)
    confidence: float             # Detection confidence [0, 1]
    landmarks: np.ndarray | None  # 5-point landmarks (eyes, nose, mouth)
    frame_idx: int                # Source frame index


@dataclass
class FrameFaces:
    """All faces detected in a single frame."""
    frame_idx: int
    timestamp_sec: float
    faces: list[DetectedFace]
    original_shape: tuple[int, int]  # (height, width) of source frame

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    @property
    def has_faces(self) -> bool:
        return len(self.faces) > 0


class FaceDetectionPipeline:
    """
    End-to-end face detection: detect → crop with margin → align → resize.

    Uses MTCNN for detection (lightweight, good frontal performance).
    Adds configurable margin around face box for context.
    Tracks face IDs across frames using IoU-based matching.
    """

    def __init__(
        self,
        device: str = "cpu",
        target_size: int = 224,
        margin_ratio: float = 0.3,
        min_confidence: float = 0.95,
        min_face_size: int = 40,
    ):
        self.device = device
        self.target_size = target_size
        self.margin_ratio = margin_ratio
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size
        self._mtcnn = None
        self._next_face_id = 0
        self._prev_faces: list[tuple[int, FaceBBox]] = []  # (id, bbox)

    def load(self) -> None:
        """Initialize MTCNN detector."""
        # MTCNN runs better on CPU than MPS for small images
        mtcnn_device = torch.device("cpu")
        self._mtcnn = MTCNN(
            image_size=self.target_size,
            margin=0,  # we handle margin ourselves
            min_face_size=self.min_face_size,
            thresholds=[0.6, 0.7, 0.7],  # default MTCNN thresholds
            factor=0.709,
            post_process=False,
            device=mtcnn_device,
            keep_all=True,  # detect ALL faces, not just largest
        )
        logger.info("face_detector_loaded", backend="MTCNN")

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_sec: float = 0.0,
    ) -> FrameFaces:
        """
        Detect, crop, and align faces in a single frame.

        Args:
            frame: BGR numpy array (H, W, 3).
            frame_idx: Frame index in the video.
            timestamp_sec: Timestamp for this frame.

        Returns:
            FrameFaces with all detected faces.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MTCNN detection
        boxes, confidences, landmarks = self._mtcnn.detect(rgb, landmarks=True)

        faces: list[DetectedFace] = []

        if boxes is None or len(boxes) == 0:
            self._prev_faces = []
            return FrameFaces(
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                faces=[],
                original_shape=(h, w),
            )

        current_bboxes: list[tuple[int, FaceBBox]] = []

        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            if conf is None or conf < self.min_confidence:
                continue

            # Convert MTCNN box [x1, y1, x2, y2] to [x, y, w, h]
            x1, y1, x2, y2 = [int(v) for v in box]
            bw, bh = x2 - x1, y2 - y1

            if bw < self.min_face_size or bh < self.min_face_size:
                continue

            # Add margin
            margin_x = int(bw * self.margin_ratio)
            margin_y = int(bh * self.margin_ratio)
            x1m = max(0, x1 - margin_x)
            y1m = max(0, y1 - margin_y)
            x2m = min(w, x2 + margin_x)
            y2m = min(h, y2 + margin_y)

            bbox = FaceBBox(x=x1m, y=y1m, w=x2m - x1m, h=y2m - y1m)

            # Crop from original BGR frame
            crop_raw = frame[y1m:y2m, x1m:x2m].copy()

            if crop_raw.size == 0:
                continue

            # Resize to target size
            crop = cv2.resize(crop_raw, (self.target_size, self.target_size))

            # Align face using landmarks (if available)
            lm = landmarks[i] if landmarks is not None else None
            if lm is not None:
                crop = self._align_face(frame, lm, bbox)

            # Track face ID
            face_id = self._match_face_id(bbox)
            current_bboxes.append((face_id, bbox))

            faces.append(DetectedFace(
                face_id=face_id,
                bbox=bbox,
                crop=crop,
                crop_raw=crop_raw,
                confidence=float(conf),
                landmarks=lm,
                frame_idx=frame_idx,
            ))

        self._prev_faces = current_bboxes

        return FrameFaces(
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            faces=faces,
            original_shape=(h, w),
        )

    def detect_batch(
        self,
        frames: list[tuple[np.ndarray, int, float]],
    ) -> list[FrameFaces]:
        """
        Detect faces across multiple frames with tracking.

        Args:
            frames: List of (image, frame_idx, timestamp_sec) tuples.

        Returns:
            List of FrameFaces, one per input frame.
        """
        results = []
        for image, idx, ts in frames:
            result = self.detect_frame(image, idx, ts)
            results.append(result)

        # Log summary
        total_faces = sum(ff.num_faces for ff in results)
        frames_with_faces = sum(1 for ff in results if ff.has_faces)
        unique_ids = set()
        for ff in results:
            for f in ff.faces:
                unique_ids.add(f.face_id)

        logger.info(
            "face_detection_complete",
            frames_processed=len(results),
            frames_with_faces=frames_with_faces,
            total_detections=total_faces,
            unique_face_ids=len(unique_ids),
        )

        return results

    def _align_face(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        bbox: FaceBBox,
    ) -> np.ndarray:
        """
        Align face using eye landmarks via affine transform.
        Rotates face so eyes are horizontal, then crops and resizes.
        """
        try:
            # MTCNN landmarks: [left_eye, right_eye, nose, mouth_left, mouth_right]
            left_eye = landmarks[0]
            right_eye = landmarks[1]

            # Compute rotation angle
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = float(np.degrees(np.arctan2(dy, dx)))

            # Center of eyes
            eye_center = (
                (left_eye[0] + right_eye[0]) / 2,
                (left_eye[1] + right_eye[1]) / 2,
            )

            # Rotation matrix
            m = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)

            # Rotate entire frame
            h, w = frame.shape[:2]
            rotated = cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_LINEAR)

            # Crop with margin from rotated frame
            x, y, bw, bh = bbox.x, bbox.y, bbox.w, bbox.h
            crop = rotated[y:y+bh, x:x+bw]

            if crop.size == 0:
                return cv2.resize(
                    frame[y:y+bh, x:x+bw], (self.target_size, self.target_size)
                )

            return cv2.resize(crop, (self.target_size, self.target_size))

        except Exception:
            # Fallback: simple crop without alignment
            x, y, bw, bh = bbox.x, bbox.y, bbox.w, bbox.h
            crop = frame[y:y+bh, x:x+bw]
            if crop.size == 0:
                return np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
            return cv2.resize(crop, (self.target_size, self.target_size))

    def _match_face_id(self, bbox: FaceBBox) -> int:
        """
        Match a detected face to a previously tracked face using IoU.
        If no match found, assign a new ID.
        """
        if not self._prev_faces:
            face_id = self._next_face_id
            self._next_face_id += 1
            return face_id

        best_iou = 0.0
        best_id = -1

        for prev_id, prev_bbox in self._prev_faces:
            iou = _compute_iou(bbox, prev_bbox)
            if iou > best_iou:
                best_iou = iou
                best_id = prev_id

        if best_iou > 0.3:  # IoU threshold for matching
            return best_id

        face_id = self._next_face_id
        self._next_face_id += 1
        return face_id

    def reset_tracking(self) -> None:
        """Reset face tracking state (call between videos)."""
        self._next_face_id = 0
        self._prev_faces = []


def _compute_iou(a: FaceBBox, b: FaceBBox) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a.area + b.area - inter

    return inter / union if union > 0 else 0.0
