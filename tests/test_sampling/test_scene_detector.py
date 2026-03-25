#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FILE: tests/test_sampling/test_scene_detector.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""Tests for scene detection."""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from verifi.sampling.scene_detector import detect_scenes


@pytest.fixture
def synthetic_video_path():
    """Create a synthetic video with 2 distinct scenes (different colors)."""
    path = tempfile.mktemp(suffix=".mp4")
    fps = 30
    size = (320, 240)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)

    # Scene 1: 60 frames of blue
    for _ in range(60):
        frame = np.full((*size[::-1], 3), (255, 100, 50), dtype=np.uint8)
        # Add slight noise for realism
        noise = np.random.randint(-5, 5, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)

    # Scene 2: 60 frames of red (abrupt change)
    for _ in range(60):
        frame = np.full((*size[::-1], 3), (50, 50, 255), dtype=np.uint8)
        noise = np.random.randint(-5, 5, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def uniform_video_path():
    """Create a video with no scene changes."""
    path = tempfile.mktemp(suffix=".mp4")
    fps = 30
    size = (320, 240)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)

    for _ in range(90):
        frame = np.full((*size[::-1], 3), (128, 128, 128), dtype=np.uint8)
        noise = np.random.randint(-3, 3, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    yield path
    Path(path).unlink(missing_ok=True)


def test_detects_scene_boundary(synthetic_video_path):
    """Should detect the blue→red transition."""
    analysis = detect_scenes(synthetic_video_path, threshold=5.0)

    assert analysis.num_scenes >= 2, f"Expected >= 2 scenes, got {analysis.num_scenes}"
    assert len(analysis.boundaries) >= 1, "Expected at least 1 boundary"

    # Boundary should be around frame 60
    boundary = analysis.boundaries[0]
    assert 55 <= boundary.frame_idx <= 65, f"Boundary at {boundary.frame_idx}, expected ~60"


def test_no_false_boundaries(uniform_video_path):
    """Uniform video should produce 1 scene and 0 boundaries."""
    analysis = detect_scenes(uniform_video_path, threshold=20.0)

    assert analysis.num_scenes == 1, f"Expected 1 scene, got {analysis.num_scenes}"
    assert len(analysis.boundaries) == 0, f"Expected 0 boundaries, got {len(analysis.boundaries)}"


def test_scene_coverage(synthetic_video_path):
    """All frames should be covered by exactly one scene."""
    analysis = detect_scenes(synthetic_video_path, threshold=5.0)

    covered = set()
    for scene in analysis.scenes:
        for i in range(scene.start_idx, scene.end_idx + 1):
            assert i not in covered, f"Frame {i} covered by multiple scenes"
            covered.add(i)

    # Should cover most frames (some edge frames might be lost)
    assert len(covered) >= analysis.total_frames - 2


def test_scene_has_valid_fields(synthetic_video_path):
    """Each scene should have valid start/end/duration."""
    analysis = detect_scenes(synthetic_video_path)

    for scene in analysis.scenes:
        assert scene.start_idx <= scene.end_idx
        assert scene.duration_sec > 0
        assert scene.frame_count > 0
        assert scene.start_sec >= 0

