"""Celeb-DF v2 dataset adapter."""
from __future__ import annotations

from collections.abc import Iterator

from verifi.benchmark.datasets.base import (
    BaseDatasetAdapter,
    DatasetNotFoundError,
    VideoSample,
)


class CelebDFAdapter(BaseDatasetAdapter):
    """
    Adapter for Celeb-DF v2 dataset.

    Expected directory structure:
        {root}/
            Celeb-real/          # 590 real videos
            Celeb-synthesis/     # 5639 synthesized videos
            YouTube-real/        # 300 YouTube real videos (optional)
            List_of_testing_videos.txt
    """

    @property
    def name(self) -> str:
        return "celebdf"

    @property
    def methods(self) -> list[str]:
        return ["face_swap"]

    def validate(self) -> None:
        real_dir = self.root / "Celeb-real"
        synth_dir = self.root / "Celeb-synthesis"
        test_list = self.root / "List_of_testing_videos.txt"

        missing = []
        if not real_dir.is_dir():
            missing.append(str(real_dir))
        if not synth_dir.is_dir():
            missing.append(str(synth_dir))
        if not test_list.exists():
            missing.append(str(test_list))

        if missing:
            raise DatasetNotFoundError(
                dataset_name="Celeb-DF v2",
                expected_path=self.root,
                instructions=(
                    "Download Celeb-DF v2 from "
                    "https://github.com/yuezunli/celeb-deepfakeforensics\n"
                    f"Missing: {', '.join(missing)}"
                ),
            )

    def _load_test_list(self) -> set[str]:
        test_file = self.root / "List_of_testing_videos.txt"
        if not test_file.exists():
            return set()
        paths = set()
        with open(test_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    paths.add(parts[1])
        return paths

    def iter_samples(
        self,
        split: str = "test",
        methods: list[str] | None = None,
        max_per_class: int | None = None,
    ) -> Iterator[VideoSample]:
        self.validate()

        test_paths = self._load_test_list() if split == "test" else set()

        # Real videos
        real_count = 0
        for subdir in ["Celeb-real", "YouTube-real"]:
            real_dir = self.root / subdir
            if not real_dir.is_dir():
                continue
            for vpath in sorted(real_dir.glob("*.mp4")):
                rel = f"{subdir}/{vpath.name}"
                if test_paths and rel not in test_paths:
                    continue
                if max_per_class and real_count >= max_per_class:
                    break
                yield VideoSample(
                    path=vpath, label=0, method="real", split=split,
                )
                real_count += 1

        # Fake videos
        synth_dir = self.root / "Celeb-synthesis"
        fake_count = 0
        if synth_dir.is_dir():
            for vpath in sorted(synth_dir.glob("*.mp4")):
                rel = f"Celeb-synthesis/{vpath.name}"
                if test_paths and rel not in test_paths:
                    continue
                if max_per_class and fake_count >= max_per_class:
                    break
                yield VideoSample(
                    path=vpath, label=1, method="face_swap", split=split,
                )
                fake_count += 1
