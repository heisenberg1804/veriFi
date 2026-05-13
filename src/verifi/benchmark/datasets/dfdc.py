"""DFDC (Deepfake Detection Challenge) dataset adapter."""
from __future__ import annotations

import json
from collections.abc import Iterator

from verifi.benchmark.datasets.base import (
    BaseDatasetAdapter,
    DatasetNotFoundError,
    VideoSample,
)


class DFDCAdapter(BaseDatasetAdapter):
    """
    Adapter for the DFDC dataset.

    Expected directory structure:
        {root}/
            dfdc_train_part_00/
                metadata.json
                *.mp4
            dfdc_train_part_01/
                ...
            ...
            dfdc_train_part_49/

    Convention: parts 00-39 = train, 40-49 = test.
    """

    TRAIN_PARTS = range(0, 40)
    TEST_PARTS = range(40, 50)

    @property
    def name(self) -> str:
        return "dfdc"

    @property
    def methods(self) -> list[str]:
        return ["mixed"]

    def validate(self) -> None:
        parts = list(self.root.glob("dfdc_train_part_*"))
        if not parts:
            raise DatasetNotFoundError(
                dataset_name="DFDC",
                expected_path=self.root,
                instructions=(
                    "Download DFDC from "
                    "https://www.kaggle.com/c/deepfake-detection-challenge/data\n"
                    "Expected: {root}/dfdc_train_part_00/ through _49/"
                ),
            )

        has_metadata = any(
            (p / "metadata.json").exists() for p in parts
        )
        if not has_metadata:
            raise DatasetNotFoundError(
                dataset_name="DFDC",
                expected_path=self.root,
                instructions="Found part directories but no metadata.json files.",
            )

    def _get_parts_for_split(self, split: str) -> list[int]:
        if split == "test":
            return list(self.TEST_PARTS)
        if split == "train":
            return list(self.TRAIN_PARTS)
        return list(range(50))

    def iter_samples(
        self,
        split: str = "test",
        methods: list[str] | None = None,
        max_per_class: int | None = None,
    ) -> Iterator[VideoSample]:
        self.validate()

        target_parts = self._get_parts_for_split(split)
        real_count = 0
        fake_count = 0

        for part_idx in target_parts:
            part_dir = self.root / f"dfdc_train_part_{part_idx:02d}"
            if not part_dir.is_dir():
                continue

            meta_file = part_dir / "metadata.json"
            if not meta_file.exists():
                continue

            with open(meta_file) as f:
                metadata = json.load(f)

            for filename, info in sorted(metadata.items()):
                vpath = part_dir / filename
                if not vpath.exists():
                    continue

                label = 1 if info.get("label") == "FAKE" else 0
                method = "real" if label == 0 else "mixed"

                if max_per_class:
                    if label == 0 and real_count >= max_per_class:
                        continue
                    if label == 1 and fake_count >= max_per_class:
                        continue

                yield VideoSample(
                    path=vpath, label=label, method=method, split=split,
                    metadata={"original": info.get("original", "")},
                )

                if label == 0:
                    real_count += 1
                else:
                    fake_count += 1
