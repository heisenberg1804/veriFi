"""Benchmark runner: orchestrates pipeline over datasets."""
from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import structlog
import torch

from verifi.benchmark.datasets.base import (
    BaseDatasetAdapter,
    VideoSample,
)
from verifi.benchmark.datasets.celebdf import CelebDFAdapter
from verifi.benchmark.datasets.df40 import DF40Adapter
from verifi.benchmark.datasets.dfdc import DFDCAdapter
from verifi.benchmark.datasets.faceforensics import FaceForensicsAdapter
from verifi.benchmark.results import ResultsWriter, VideoResult
from verifi.pipeline.orchestrator import VeriFiPipeline

logger = structlog.get_logger()

ADAPTER_REGISTRY: dict[str, type[BaseDatasetAdapter]] = {
    "ff++": FaceForensicsAdapter,
    "celebdf": CelebDFAdapter,
    "dfdc": DFDCAdapter,
    "df40": DF40Adapter,
}


@dataclass
class RunConfig:
    """Configuration for a benchmark run."""

    dataset_name: str
    dataset_root: str
    compression: str = "c23"
    split: str = "test"
    methods: list[str] | None = None
    max_videos: int | None = None
    stratified: bool = True
    seed: int = 42
    timeout_per_video: int = 120
    skip_heatmaps: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def _get_adapter(config: RunConfig) -> BaseDatasetAdapter:
    adapter_cls = ADAPTER_REGISTRY.get(config.dataset_name)
    if adapter_cls is None:
        raise ValueError(
            f"Unknown dataset: {config.dataset_name}. "
            f"Available: {list(ADAPTER_REGISTRY.keys())}"
        )
    return adapter_cls(
        root=Path(config.dataset_root),
        compression=config.compression,
    )


def _extract_signal_scores(analysis) -> dict[str, float]:
    """Extract per-signal mean scores from frame analyses."""
    import numpy as np

    signal_means: dict[str, list[float]] = {}
    for fa in analysis.frame_analyses:
        for sig in fa.signals:
            signal_means.setdefault(sig.name, []).append(sig.score)
    for fa in analysis.face_analyses:
        for sig in fa.signals:
            key = f"face_{sig.name}"
            signal_means.setdefault(key, []).append(sig.score)

    return {k: float(np.mean(v)) for k, v in signal_means.items()}


class BenchmarkRunner:
    """Run the VeriFi pipeline over a dataset and collect results."""

    def __init__(
        self,
        pipeline: VeriFiPipeline,
        run_config: RunConfig,
        output_dir: Path | None = None,
    ):
        self.pipeline = pipeline
        self.config = run_config
        self.adapter = _get_adapter(run_config)

        if output_dir is None:
            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            run_id = f"{run_config.dataset_name}_{run_config.compression}_{ts}"
            output_dir = Path("data/benchmarks") / run_id

        self.output_dir = output_dir
        self.writer = ResultsWriter(output_dir)

    def run(self) -> Path:
        """Execute the benchmark. Returns path to results directory."""
        self.adapter.validate()
        self.writer.save_config(self.config.to_dict())

        processed = self.writer.get_processed_paths()
        if processed:
            logger.info("resuming_benchmark", already_done=len(processed))

        if self.config.max_videos and self.config.stratified:
            samples = list(self.adapter.iter_stratified_sample(
                split=self.config.split,
                n=self.config.max_videos,
                seed=self.config.seed,
            ))
        else:
            samples = list(self.adapter.iter_samples(
                split=self.config.split,
                methods=self.config.methods,
            ))
            if self.config.max_videos:
                samples = samples[: self.config.max_videos]

        total = len(samples)
        logger.info(
            "benchmark_start",
            dataset=self.config.dataset_name,
            total=total,
            resuming=len(processed),
        )

        done = len(processed)
        errors = 0
        t_start = time.perf_counter()

        for i, sample in enumerate(samples):
            path_str = str(sample.path)
            if path_str in processed:
                continue

            result = self._analyze_one(sample)
            self.writer.write(result)
            done += 1

            if result.error:
                errors += 1

            if done % 10 == 0:
                elapsed = time.perf_counter() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                logger.info(
                    "benchmark_progress",
                    done=done,
                    total=total,
                    errors=errors,
                    rate=f"{rate:.1f} vid/min" if rate > 0 else "?",
                    eta=f"{eta / 60:.0f}min",
                )

            if done % 50 == 0:
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        elapsed = time.perf_counter() - t_start
        logger.info(
            "benchmark_complete",
            total=done,
            errors=errors,
            elapsed=f"{elapsed:.0f}s",
        )

        return self.output_dir

    def _analyze_one(self, sample: VideoSample) -> VideoResult:
        """Run pipeline on one video with error handling."""
        t0 = time.perf_counter()

        try:
            report = self.pipeline.analyze(
                str(sample.path),
                skip_explainability=self.config.skip_heatmaps,
            )
            analysis = report.analysis
            signal_scores = _extract_signal_scores(analysis)

            return VideoResult(
                video_path=str(sample.path),
                dataset=self.config.dataset_name,
                label=sample.label,
                method=sample.method,
                predicted_score=analysis.video_score,
                predicted_verdict=analysis.verdict.value,
                confidence=analysis.confidence,
                face_path_score=analysis.face_path_score,
                frame_path_score=analysis.frame_path_score,
                dominant_path=analysis.dominant_path,
                manipulation_type=analysis.manipulation_type.value,
                signal_scores=signal_scores,
                processing_time_sec=time.perf_counter() - t0,
            )

        except Exception as e:
            logger.error(
                "benchmark_video_error",
                video=str(sample.path),
                error=str(e),
            )
            return VideoResult(
                video_path=str(sample.path),
                dataset=self.config.dataset_name,
                label=sample.label,
                method=sample.method,
                predicted_score=0.5,
                predicted_verdict="ERROR",
                confidence=0.0,
                face_path_score=0.0,
                frame_path_score=0.0,
                dominant_path="none",
                manipulation_type="unknown",
                processing_time_sec=time.perf_counter() - t0,
                error=str(e),
            )
