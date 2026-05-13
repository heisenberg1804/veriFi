"""DF40 dataset adapter."""
from __future__ import annotations

import json
from collections.abc import Iterator

from verifi.benchmark.datasets.base import (
    BaseDatasetAdapter,
    DatasetNotFoundError,
    VideoSample,
)


class DF40Adapter(BaseDatasetAdapter):
    """
    Adapter for the DF40 benchmark (40 deepfake generation methods).

    Expected directory structure:
        {root}/
            real/               # Real videos
            {method_name}/      # One directory per generation method
            splits.json         # {"train": [...], "val": [...], "test": [...]}
    """

    @property
    def name(self) -> str:
        return "df40"

    @property
    def methods(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            d.name for d in self.root.iterdir()
            if d.is_dir() and d.name != "real"
        )

    def validate(self) -> None:
        real_dir = self.root / "real"
        if not real_dir.is_dir():
            raise DatasetNotFoundError(
                dataset_name="DF40",
                expected_path=self.root,
                instructions=(
                    "Download DF40 from the official repository.\n"
                    "Expected: {root}/real/ and method subdirectories."
                ),
            )

    def _load_split_ids(self, split: str) -> set[str] | None:
        splits_file = self.root / "splits.json"
        if not splits_file.exists():
            return None
        with open(splits_file) as f:
            splits = json.load(f)
        ids = splits.get(split)
        return set(ids) if ids else None

    def iter_samples(
        self,
        split: str = "test",
        methods: list[str] | None = None,
        max_per_class: int | None = None,
    ) -> Iterator[VideoSample]:
        self.validate()

        split_ids = self._load_split_ids(split)
        target_methods = methods or self.methods

        # Real videos
        real_dir = self.root / "real"
        real_count = 0
        if real_dir.is_dir():
            for vpath in sorted(real_dir.glob("*.mp4")):
                if split_ids and vpath.stem not in split_ids:
                    continue
                if max_per_class and real_count >= max_per_class:
                    break
                yield VideoSample(
                    path=vpath, label=0, method="real", split=split,
                )
                real_count += 1

        # Fake videos per method
        for method in target_methods:
            method_dir = self.root / method
            if not method_dir.is_dir():
                continue
            fake_count = 0
            for vpath in sorted(method_dir.glob("*.mp4")):
                if split_ids and vpath.stem not in split_ids:
                    continue
                if max_per_class and fake_count >= max_per_class:
                    break
                yield VideoSample(
                    path=vpath, label=1, method=method, split=split,
                )
                fake_count += 1
