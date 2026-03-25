
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FILE: tests/test_preprocessing/test_face_detector.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""Tests for face detection pipeline."""
import numpy as np
import pytest

from verifi.preprocessing.face_detector import (
    FaceBBox,
    FaceDetectionPipeline,
    _compute_iou,
)


@pytest.fixture
def face_pipeline():
    """Initialized face detection pipeline."""
    pipeline = FaceDetectionPipeline(
        device="cpu",
        target_size=224,
        margin_ratio=0.3,
        min_confidence=0.90,
    )
    pipeline.load()
    return pipeline


def test_iou_identical_boxes():
    """Identical boxes should have IoU = 1.0."""
    a = FaceBBox(x=10, y=10, w=100, h=100)
    b = FaceBBox(x=10, y=10, w=100, h=100)
    assert abs(_compute_iou(a, b) - 1.0) < 0.001


def test_iou_no_overlap():
    """Non-overlapping boxes should have IoU = 0.0."""
    a = FaceBBox(x=0, y=0, w=50, h=50)
    b = FaceBBox(x=100, y=100, w=50, h=50)
    assert _compute_iou(a, b) == 0.0


def test_iou_partial_overlap():
    """Partially overlapping boxes should have 0 < IoU < 1."""
    a = FaceBBox(x=0, y=0, w=100, h=100)
    b = FaceBBox(x=50, y=50, w=100, h=100)
    iou = _compute_iou(a, b)
    assert 0 < iou < 1


def test_detect_empty_frame(face_pipeline):
    """Black frame should detect no faces."""
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    result = face_pipeline.detect_frame(black, frame_idx=0)
    assert result.num_faces == 0
    assert not result.has_faces


def test_detect_returns_correct_shape(face_pipeline):
    """Face crops should match target size."""
    # Create a frame with a simple face-like pattern
    # (MTCNN likely won't detect this, but test the pipeline structure)
    frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
    result = face_pipeline.detect_frame(frame, frame_idx=0, timestamp_sec=0.5)

    assert result.frame_idx == 0
    assert result.timestamp_sec == 0.5
    assert result.original_shape == (480, 640)

    # If any faces detected, check crop shape
    for face in result.faces:
        assert face.crop.shape == (224, 224, 3)
        assert face.confidence >= 0.9


def test_tracking_reset(face_pipeline):
    """Reset should clear face IDs."""
    face_pipeline._next_face_id = 42
    face_pipeline._prev_faces = [(1, FaceBBox(0, 0, 100, 100))]
    face_pipeline.reset_tracking()
    assert face_pipeline._next_face_id == 0
    assert len(face_pipeline._prev_faces) == 0


def test_bbox_properties():
    """BBox should compute center and area correctly."""
    bbox = FaceBBox(x=10, y=20, w=100, h=200)
    assert bbox.center == (60, 120)
    assert bbox.area == 20000
    assert bbox.to_tuple() == (10, 20, 100, 200)
