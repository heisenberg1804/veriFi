"""Shared test fixtures for VeriFi."""
import numpy as np
import pytest


@pytest.fixture
def dummy_face_crop():
    """224x224 BGR face crop (random pixels)."""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def dummy_face_crops_batch():
    """Batch of 4 dummy face crops."""
    return [
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        for _ in range(4)
    ]


@pytest.fixture
def dummy_frame_pair():
    """Two consecutive 720p frames for temporal analysis."""
    a = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    b = a.copy()
    # Add slight motion to background
    b[:, 5:] = a[:, :-5]
    # Add larger motion to face region (center)
    b[260:460, 540:740] = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    return a, b
