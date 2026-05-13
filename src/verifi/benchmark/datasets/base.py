"""Abstract base class for dataset adapters."""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path


class DatasetNotFoundError(FileNotFoundError):
    """Raised when a dataset is not found at the expected path."""

    def __init__(self, dataset_name: str, expected_path: Path, instructions: str):
        self.dataset_name = dataset_name
        self.expected_path = expected_path
        self.instructions = instructions
        super().__init__(
            f"Dataset '{dataset_name}' not found at {expected_path}.\n"
            f"{instructions}"
        )


@dataclass
class VideoSample:
    """A single video with its ground truth label."""

    path: Path
    label: int  # 0 = real, 1 = fake
    method: str
    split: str = "test"
    metadata: dict = field(default_factory=dict)


class BaseDatasetAdapter(ABC):
    """Interface for loading benchmark datasets."""

    def __init__(self, root: Path, compression: str = "c23"):
        self.root = Path(root)
        self.compression = compression

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def methods(self) -> list[str]: ...

    @abstractmethod
    def validate(self) -> None:
        """Check dataset exists. Raise DatasetNotFoundError if not."""

    @abstractmethod
    def iter_samples(
        self,
        split: str = "test",
        methods: list[str] | None = None,
        max_per_class: int | None = None,
    ) -> Iterator[VideoSample]:
        """Yield video samples."""

    def iter_stratified_sample(
        self,
        split: str = "test",
        n: int = 100,
        seed: int = 42,
    ) -> Iterator[VideoSample]:
        """Yield a stratified random sample balanced across classes and methods."""
        all_samples = list(self.iter_samples(split=split))

        groups: dict[tuple[int, str], list[VideoSample]] = defaultdict(list)
        for s in all_samples:
            groups[(s.label, s.method)].append(s)

        rng = random.Random(seed)
        per_group = max(1, n // len(groups)) if groups else 0

        selected = []
        for key in sorted(groups.keys()):
            pool = groups[key]
            rng.shuffle(pool)
            selected.extend(pool[:per_group])

        rng.shuffle(selected)
        yield from selected[:n]

    def summary(self) -> dict:
        """Return dataset statistics."""
        try:
            self.validate()
        except DatasetNotFoundError:
            return {"name": self.name, "available": False}

        all_samples = list(self.iter_samples(split="test"))
        real = sum(1 for s in all_samples if s.label == 0)
        fake = sum(1 for s in all_samples if s.label == 1)
        method_counts = defaultdict(int)
        for s in all_samples:
            method_counts[s.method] += 1

        return {
            "name": self.name,
            "available": True,
            "total": len(all_samples),
            "real": real,
            "fake": fake,
            "methods": dict(method_counts),
        }
