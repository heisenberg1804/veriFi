"""Tests for dataset adapters."""
import json

import pytest

from verifi.benchmark.datasets.base import DatasetNotFoundError, VideoSample
from verifi.benchmark.datasets.faceforensics import FaceForensicsAdapter


def _create_ff_structure(root, n_real=3, n_fake_per_method=2):
    """Create a mock FF++ directory structure."""
    methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

    # Real videos
    real_dir = root / "original_sequences" / "youtube" / "c23" / "videos"
    real_dir.mkdir(parents=True)
    ids = []
    for i in range(n_real):
        vid_id = str(i)
        (real_dir / f"{vid_id}.mp4").touch()
        ids.append(vid_id)

    # Fake videos
    manip = root / "manipulated_sequences"
    for method in methods:
        method_dir = manip / method / "c23" / "videos"
        method_dir.mkdir(parents=True)
        for i in range(n_fake_per_method):
            vid_id = str(i)
            (method_dir / f"{vid_id}_{i+10}.mp4").touch()

    # Splits — pairs of IDs that appear in file stems
    splits_dir = root / "splits"
    splits_dir.mkdir()
    test_pairs = [[str(i), str(i + 100)] for i in range(n_real)]
    for split_name in ["train", "val", "test"]:
        with open(splits_dir / f"{split_name}.json", "w") as f:
            json.dump(test_pairs, f)


def test_ff_validate_missing(tmp_path):
    adapter = FaceForensicsAdapter(tmp_path / "nonexistent")
    with pytest.raises(DatasetNotFoundError):
        adapter.validate()


def test_ff_validate_present(tmp_path):
    _create_ff_structure(tmp_path)
    adapter = FaceForensicsAdapter(tmp_path)
    adapter.validate()


def test_ff_iter_samples(tmp_path):
    _create_ff_structure(tmp_path, n_real=3, n_fake_per_method=2)
    adapter = FaceForensicsAdapter(tmp_path)
    samples = list(adapter.iter_samples(split="test"))
    real = [s for s in samples if s.label == 0]
    fake = [s for s in samples if s.label == 1]
    assert len(real) == 3
    assert len(fake) == 8  # 4 methods * 2 each


def test_ff_iter_samples_method_filter(tmp_path):
    _create_ff_structure(tmp_path)
    adapter = FaceForensicsAdapter(tmp_path)
    samples = list(adapter.iter_samples(split="test", methods=["Deepfakes"]))
    fake = [s for s in samples if s.label == 1]
    assert all(s.method == "Deepfakes" for s in fake)


def test_ff_stratified_sample(tmp_path):
    _create_ff_structure(tmp_path, n_real=10, n_fake_per_method=5)
    adapter = FaceForensicsAdapter(tmp_path)
    samples = list(adapter.iter_stratified_sample(split="test", n=10))
    assert len(samples) <= 10


def test_ff_name_and_methods(tmp_path):
    adapter = FaceForensicsAdapter(tmp_path)
    assert adapter.name == "ff++"
    assert "Deepfakes" in adapter.methods


def test_video_sample_fields():
    from pathlib import Path

    s = VideoSample(path=Path("/a.mp4"), label=0, method="real")
    assert s.label == 0
    assert s.split == "test"
