"""FaceForensics++ dataset adapter."""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from verifi.benchmark.datasets.base import (
    BaseDatasetAdapter,
    DatasetNotFoundError,
    VideoSample,
)

FF_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


class FaceForensicsAdapter(BaseDatasetAdapter):
    """
    Adapter for FaceForensics++ dataset.

    Expected directory structure:
        {root}/
            original_sequences/youtube/{compression}/videos/*.mp4
            manipulated_sequences/{method}/{compression}/videos/*.mp4
            splits/
                train.json   # [[id1_a, id1_b], ...]
                val.json
                test.json
    """

    @property
    def name(self) -> str:
        return "ff++"

    @property
    def methods(self) -> list[str]:
        return list(FF_METHODS)

    def validate(self) -> None:
        orig = self.root / "original_sequences" / "youtube"
        manip = self.root / "manipulated_sequences"
        splits = self.root / "splits"

        missing = []
        if not orig.is_dir():
            missing.append(str(orig))
        if not manip.is_dir():
            missing.append(str(manip))
        if not splits.is_dir():
            missing.append(str(splits))

        if missing:
            raise DatasetNotFoundError(
                dataset_name="FaceForensics++",
                expected_path=self.root,
                instructions=(
                    "Download FF++ from https://github.com/ondyari/FaceForensics\n"
                    f"Missing directories: {', '.join(missing)}\n"
                    "Expected structure:\n"
                    "  {root}/original_sequences/youtube/c23/videos/*.mp4\n"
                    "  {root}/manipulated_sequences/Deepfakes/c23/videos/*.mp4\n"
                    "  {root}/splits/train.json"
                ),
            )

    def _load_split_ids(self, split: str) -> set[str]:
        split_file = self.root / "splits" / f"{split}.json"
        if not split_file.exists():
            return set()
        with open(split_file) as f:
            pairs = json.load(f)
        ids = set()
        for pair in pairs:
            for vid_id in pair:
                ids.add(str(vid_id))
        return ids

    def _video_id_from_path(self, path: Path) -> str:
        return path.stem.split("_")[0]

    def iter_samples(
        self,
        split: str = "test",
        methods: list[str] | None = None,
        max_per_class: int | None = None,
    ) -> Iterator[VideoSample]:
        self.validate()

        split_ids = self._load_split_ids(split)
        target_methods = methods or FF_METHODS

        # Real videos
        real_dir = (
            self.root / "original_sequences" / "youtube"
            / self.compression / "videos"
        )
        real_count = 0
        if real_dir.is_dir():
            for vpath in sorted(real_dir.glob("*.mp4")):
                vid_id = self._video_id_from_path(vpath)
                if split_ids and vid_id not in split_ids:
                    continue
                if max_per_class and real_count >= max_per_class:
                    break
                yield VideoSample(
                    path=vpath, label=0, method="real", split=split,
                )
                real_count += 1

        # Fake videos per method
        for method in target_methods:
            method_dir = (
                self.root / "manipulated_sequences" / method
                / self.compression / "videos"
            )
            if not method_dir.is_dir():
                continue
            fake_count = 0
            for vpath in sorted(method_dir.glob("*.mp4")):
                vid_id = self._video_id_from_path(vpath)
                if split_ids and vid_id not in split_ids:
                    continue
                if max_per_class and fake_count >= max_per_class:
                    break
                yield VideoSample(
                    path=vpath, label=1, method=method, split=split,
                )
                fake_count += 1
