"""Benchmark result data models and I/O (JSONL + CSV)."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class VideoResult:
    """Result of running the pipeline on one video."""

    video_path: str
    dataset: str
    label: int  # 0 = real, 1 = fake
    method: str
    predicted_score: float
    predicted_verdict: str
    confidence: float
    face_path_score: float
    frame_path_score: float
    dominant_path: str
    manipulation_type: str
    signal_scores: dict[str, float] = field(default_factory=dict)
    processing_time_sec: float = 0.0
    error: str | None = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> VideoResult:
        return cls(**d)


CSV_COLUMNS = [
    "video_path", "dataset", "label", "method",
    "predicted_score", "predicted_verdict", "confidence",
    "face_path_score", "frame_path_score", "dominant_path",
    "manipulation_type", "processing_time_sec", "error", "timestamp",
]


class ResultsWriter:
    """Append-only writer for benchmark results (JSONL + CSV)."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = run_dir / "results.jsonl"
        self._csv_path = run_dir / "results.csv"
        self._init_csv()

    def _init_csv(self) -> None:
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()

    def write(self, result: VideoResult) -> None:
        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

        row = {k: getattr(result, k) for k in CSV_COLUMNS}
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writerow(row)

    def get_processed_paths(self) -> set[str]:
        if not self._jsonl_path.exists():
            return set()
        paths = set()
        with open(self._jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    paths.add(json.loads(line)["video_path"])
        return paths

    def save_config(self, config: dict) -> None:
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)


class ResultsReader:
    """Read benchmark results from a completed or partial run."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self._jsonl_path = run_dir / "results.jsonl"

    def load_all(self) -> list[VideoResult]:
        if not self._jsonl_path.exists():
            return []
        results = []
        with open(self._jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(VideoResult.from_dict(json.loads(line)))
        return results

    def load_config(self) -> dict:
        config_path = self.run_dir / "config.json"
        if not config_path.exists():
            return {}
        with open(config_path) as f:
            return json.load(f)

    def count(self) -> int:
        if not self._jsonl_path.exists():
            return 0
        with open(self._jsonl_path) as f:
            return sum(1 for line in f if line.strip())
