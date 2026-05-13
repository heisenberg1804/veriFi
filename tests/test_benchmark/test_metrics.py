"""Tests for benchmark metrics computation."""
import numpy as np

from verifi.benchmark.metrics import MetricsComputer
from verifi.benchmark.results import VideoResult


def _make_results(scores_and_labels: list[tuple[float, int]]) -> list[VideoResult]:
    results = []
    for score, label in scores_and_labels:
        results.append(VideoResult(
            video_path=f"/video_{len(results)}.mp4",
            dataset="test",
            label=label,
            method="real" if label == 0 else "Deepfakes",
            predicted_score=score,
            predicted_verdict="LIKELY_MANIPULATED" if score > 0.5 else "LIKELY_AUTHENTIC",
            confidence=0.5,
            face_path_score=0.0,
            frame_path_score=score,
            dominant_path="frame",
            manipulation_type="none" if label == 0 else "full_synthesis",
            signal_scores={"dct": score, "noise_residual": score * 0.9},
        ))
    return results


def test_perfect_classifier():
    """Perfect separation should give AUC=1.0."""
    data = [(0.1, 0), (0.2, 0), (0.3, 0), (0.8, 1), (0.9, 1), (1.0, 1)]
    results = _make_results(data)
    computer = MetricsComputer(results)
    metrics = computer.compute_all()
    assert metrics.auc == 1.0
    assert metrics.n_real == 3
    assert metrics.n_fake == 3


def test_random_classifier():
    """Overlapping scores should give AUC near 0.5."""
    rng = np.random.RandomState(42)
    data = [(float(rng.uniform(0, 1)), 0) for _ in range(50)]
    data += [(float(rng.uniform(0, 1)), 1) for _ in range(50)]
    results = _make_results(data)
    computer = MetricsComputer(results)
    metrics = computer.compute_all()
    assert 0.3 < metrics.auc < 0.7


def test_threshold_metrics():
    data = [(0.1, 0), (0.2, 0), (0.6, 1), (0.8, 1)]
    results = _make_results(data)
    computer = MetricsComputer(results)
    metrics = computer.compute_all(thresholds=[0.5])
    assert len(metrics.threshold_metrics) == 1
    assert metrics.threshold_metrics[0]["threshold"] == 0.5
    assert metrics.threshold_metrics[0]["accuracy"] == 1.0


def test_per_method_metrics():
    results = []
    for i in range(5):
        results.append(VideoResult(
            video_path=f"/real_{i}.mp4", dataset="test", label=0,
            method="real", predicted_score=0.2,
            predicted_verdict="LIKELY_AUTHENTIC", confidence=0.5,
            face_path_score=0.0, frame_path_score=0.2,
            dominant_path="frame", manipulation_type="none",
        ))
    for i in range(5):
        results.append(VideoResult(
            video_path=f"/df_{i}.mp4", dataset="test", label=1,
            method="Deepfakes", predicted_score=0.8,
            predicted_verdict="LIKELY_MANIPULATED", confidence=0.5,
            face_path_score=0.0, frame_path_score=0.8,
            dominant_path="frame", manipulation_type="full_synthesis",
        ))

    computer = MetricsComputer(results)
    metrics = computer.compute_all()
    assert "Deepfakes" in metrics.per_method
    assert metrics.per_method["Deepfakes"]["auc"] == 1.0


def test_signal_aucs():
    data = [(0.1, 0), (0.2, 0), (0.8, 1), (0.9, 1)]
    results = _make_results(data)
    computer = MetricsComputer(results)
    metrics = computer.compute_all()
    assert "dct" in metrics.signal_aucs
    assert metrics.signal_aucs["dct"] == 1.0


def test_error_handling():
    results = _make_results([(0.5, 0), (0.5, 1)])
    results.append(VideoResult(
        video_path="/err.mp4", dataset="test", label=1, method="Deepfakes",
        predicted_score=0.5, predicted_verdict="ERROR", confidence=0.0,
        face_path_score=0.0, frame_path_score=0.0,
        dominant_path="none", manipulation_type="unknown",
        error="timeout",
    ))
    computer = MetricsComputer(results)
    metrics = computer.compute_all()
    assert metrics.n_errors == 1


def test_metrics_save(tmp_path):
    data = [(0.1, 0), (0.9, 1)]
    results = _make_results(data)
    computer = MetricsComputer(results)
    metrics = computer.compute_all()
    metrics.save(tmp_path / "metrics.json")
    assert (tmp_path / "metrics.json").exists()


def test_empty_results():
    computer = MetricsComputer([])
    metrics = computer.compute_all()
    assert metrics.auc == 0.0
