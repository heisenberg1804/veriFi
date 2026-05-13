"""Benchmark metrics computation: ROC, AUC, per-method breakdown."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from verifi.benchmark.results import VideoResult


@dataclass
class BenchmarkMetrics:
    """Complete metrics from a benchmark run."""

    auc: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    eer: float = 0.0
    n_real: int = 0
    n_fake: int = 0
    n_errors: int = 0

    threshold_metrics: list[dict] = field(default_factory=list)
    per_method: dict[str, dict] = field(default_factory=dict)

    fpr: list[float] = field(default_factory=list)
    tpr: list[float] = field(default_factory=list)
    thresholds: list[float] = field(default_factory=list)

    pr_precision: list[float] = field(default_factory=list)
    pr_recall: list[float] = field(default_factory=list)

    per_method_roc: dict[str, dict] = field(default_factory=dict)
    signal_aucs: dict[str, float] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "auc": round(self.auc, 4),
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "eer": round(self.eer, 4),
            "n_real": self.n_real,
            "n_fake": self.n_fake,
            "n_errors": self.n_errors,
            "threshold_metrics": self.threshold_metrics,
            "per_method": self.per_method,
            "signal_aucs": {k: round(v, 4) for k, v in self.signal_aucs.items()},
            "confusion_matrix": self.confusion,
        }

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class MetricsComputer:
    """Compute benchmark metrics from collected results."""

    def __init__(self, results: list[VideoResult]):
        valid = [r for r in results if r.error is None]
        self.results = valid
        self.n_errors = len(results) - len(valid)
        self.labels = np.array([r.label for r in valid])
        self.scores = np.array([r.predicted_score for r in valid])

    def compute_all(
        self, thresholds: list[float] | None = None,
    ) -> BenchmarkMetrics:
        if len(self.results) < 2:
            return BenchmarkMetrics(n_errors=self.n_errors)

        if thresholds is None:
            thresholds = [0.35, 0.50, 0.70]

        fpr, tpr, roc_thresholds = roc_curve(self.labels, self.scores)
        roc_auc = float(auc(fpr, tpr))
        eer = self._compute_eer(fpr, tpr)

        best_thresh = 0.5
        preds = (self.scores >= best_thresh).astype(int)

        pr_prec, pr_rec, _ = precision_recall_curve(self.labels, self.scores)

        metrics = BenchmarkMetrics(
            auc=roc_auc,
            accuracy=float(accuracy_score(self.labels, preds)),
            precision=float(precision_score(self.labels, preds, zero_division=0)),
            recall=float(recall_score(self.labels, preds, zero_division=0)),
            f1=float(f1_score(self.labels, preds, zero_division=0)),
            eer=eer,
            n_real=int((self.labels == 0).sum()),
            n_fake=int((self.labels == 1).sum()),
            n_errors=self.n_errors,
            fpr=fpr.tolist(),
            tpr=tpr.tolist(),
            thresholds=roc_thresholds.tolist(),
            pr_precision=pr_prec.tolist(),
            pr_recall=pr_rec.tolist(),
            threshold_metrics=self._threshold_metrics(thresholds),
            per_method=self._per_method_metrics(),
            per_method_roc=self._per_method_roc(),
            signal_aucs=self._signal_aucs(),
            confusion=confusion_matrix(self.labels, preds).tolist(),
        )

        return metrics

    def _compute_eer(self, fpr: np.ndarray, tpr: np.ndarray) -> float:
        fnr = 1.0 - tpr
        idx = np.nanargmin(np.abs(fpr - fnr))
        return float((fpr[idx] + fnr[idx]) / 2.0)

    def _threshold_metrics(self, thresholds: list[float]) -> list[dict]:
        results = []
        for t in thresholds:
            preds = (self.scores >= t).astype(int)
            results.append({
                "threshold": t,
                "accuracy": round(float(accuracy_score(self.labels, preds)), 4),
                "precision": round(
                    float(precision_score(self.labels, preds, zero_division=0)), 4,
                ),
                "recall": round(
                    float(recall_score(self.labels, preds, zero_division=0)), 4,
                ),
                "f1": round(
                    float(f1_score(self.labels, preds, zero_division=0)), 4,
                ),
            })
        return results

    def _per_method_metrics(self) -> dict[str, dict]:
        from collections import defaultdict

        groups: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(self.results):
            groups[r.method].append(i)

        per_method = {}
        for method, indices in sorted(groups.items()):
            if method == "real":
                continue
            fake_idx = np.array(indices)
            real_idx = np.where(self.labels == 0)[0]
            combined = np.concatenate([real_idx, fake_idx])
            sub_labels = self.labels[combined]
            sub_scores = self.scores[combined]

            if len(np.unique(sub_labels)) < 2:
                continue

            fpr, tpr, _ = roc_curve(sub_labels, sub_scores)
            method_auc = float(auc(fpr, tpr))
            per_method[method] = {
                "auc": round(method_auc, 4),
                "n_fake": len(fake_idx),
                "mean_score": round(float(sub_scores[sub_labels == 1].mean()), 4),
            }

        return per_method

    def _per_method_roc(self) -> dict[str, dict]:
        from collections import defaultdict

        groups: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(self.results):
            if r.label == 1:
                groups[r.method].append(i)

        per_method_roc = {}
        real_idx = np.where(self.labels == 0)[0]

        for method, fake_indices in sorted(groups.items()):
            combined = np.concatenate([real_idx, np.array(fake_indices)])
            sub_labels = self.labels[combined]
            sub_scores = self.scores[combined]
            if len(np.unique(sub_labels)) < 2:
                continue
            fpr, tpr, _ = roc_curve(sub_labels, sub_scores)
            per_method_roc[method] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
            }

        return per_method_roc

    def _signal_aucs(self) -> dict[str, float]:
        all_signals: set[str] = set()
        for r in self.results:
            all_signals.update(r.signal_scores.keys())

        signal_aucs = {}
        for sig_name in sorted(all_signals):
            sig_scores = []
            sig_labels = []
            for r in self.results:
                if sig_name in r.signal_scores:
                    sig_scores.append(r.signal_scores[sig_name])
                    sig_labels.append(r.label)

            if len(set(sig_labels)) < 2 or len(sig_labels) < 4:
                continue

            fpr, tpr, _ = roc_curve(
                np.array(sig_labels), np.array(sig_scores),
            )
            signal_aucs[sig_name] = float(auc(fpr, tpr))

        return signal_aucs
