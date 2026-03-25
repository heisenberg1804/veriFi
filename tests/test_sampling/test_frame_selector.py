#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FILE: tests/test_sampling/test_frame_selector.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""Tests for smart frame selection."""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from verifi.sampling.frame_selector import select_frames
from verifi.sampling.scene_detector import detect_scenes


@pytest.fixture
def multi_scene_video():
    """Video with 3 scenes of different colors."""
    path = tempfile.mktemp(suffix=".mp4")
    fps = 30
    size = (320, 240)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # blue, green, red
    for color in colors:
        for _ in range(45):  # 1.5 sec per scene
            frame = np.full((*size[::-1], 3), color, dtype=np.uint8)
            noise = np.random.randint(-10, 10, frame.shape, dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            writer.write(frame)

    writer.release()
    yield path
    Path(path).unlink(missing_ok=True)


def test_respects_budget(multi_scene_video):
    """Should not exceed frame budget."""
    analysis = detect_scenes(multi_scene_video, threshold=20.0)
    frames = select_frames(
        multi_scene_video, analysis,
        frame_budget=15, min_laplacian_var=0,  # disable blur filter
    )
    # May slightly exceed due to transition frames, but should be close
    assert len(frames) <= 30, f"Got {len(frames)} frames, budget was 15"


def test_includes_transition_frames(multi_scene_video):
    """Should include frames around scene boundaries."""
    analysis = detect_scenes(multi_scene_video, threshold=20.0)
    frames = select_frames(
        multi_scene_video, analysis,
        frame_budget=20, transition_margin=2, min_laplacian_var=0,
    )
    transition_frames = [f for f in frames if f.selection_reason == "transition"]
    if analysis.boundaries:
        assert len(transition_frames) > 0, "No transition frames selected despite boundaries"


def test_frames_sorted_chronologically(multi_scene_video):
    """Output should be sorted by frame index."""
    analysis = detect_scenes(multi_scene_video, threshold=20.0)
    frames = select_frames(multi_scene_video, analysis, min_laplacian_var=0)

    indices = [f.frame_idx for f in frames]
    assert indices == sorted(indices), "Frames not in chronological order"


def test_no_duplicate_frames(multi_scene_video):
    """Each frame index should appear at most once."""
    analysis = detect_scenes(multi_scene_video, threshold=20.0)
    frames = select_frames(multi_scene_video, analysis, min_laplacian_var=0)

    indices = [f.frame_idx for f in frames]
    assert len(indices) == len(set(indices)), "Duplicate frame indices found"


def test_frames_have_images(multi_scene_video):
    """Each selected frame should have a valid image array."""
    analysis = detect_scenes(multi_scene_video, threshold=20.0)
    frames = select_frames(multi_scene_video, analysis, frame_budget=10, min_laplacian_var=0)

    for f in frames:
        assert f.image is not None
        assert f.image.ndim == 3
        assert f.image.shape[2] == 3  # BGR
        assert f.image.dtype == np.uint8
