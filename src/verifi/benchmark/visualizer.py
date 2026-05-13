"""Publication-quality benchmark visualization."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from verifi.benchmark.metrics import BenchmarkMetrics
from verifi.benchmark.results import VideoResult


class PlotGenerator:
    """Generate benchmark plots from metrics and results."""

    def __init__(self, metrics: BenchmarkMetrics, results: list[VideoResult]):
        self.metrics = metrics
        self.results = [r for r in results if r.error is None]

    def generate_all(self, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        paths.append(self.plot_roc(output_dir))
        paths.append(self.plot_score_distribution(output_dir))
        paths.append(self.plot_precision_recall(output_dir))
        paths.append(self.plot_confusion_matrix(output_dir))
        if self.metrics.signal_aucs:
            paths.append(self.plot_signal_bars(output_dir))
        if self.metrics.per_method_roc:
            paths.append(self.plot_per_method_roc(output_dir))
        return paths

    def plot_roc(self, output_dir: Path) -> Path:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(
            self.metrics.fpr, self.metrics.tpr,
            lw=2, label=f"VeriFi (AUC = {self.metrics.auc:.3f})",
        )
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        path = output_dir / "roc_overall.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_per_method_roc(self, output_dir: Path) -> Path:
        fig, ax = plt.subplots(figsize=(8, 6))
        for method, data in sorted(self.metrics.per_method_roc.items()):
            method_auc = self.metrics.per_method.get(method, {}).get("auc", 0)
            ax.plot(
                data["fpr"], data["tpr"],
                lw=1.5, label=f"{method} (AUC={method_auc:.3f})",
            )
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Per Method")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        path = output_dir / "roc_per_method.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_score_distribution(self, output_dir: Path) -> Path:
        real_scores = [r.predicted_score for r in self.results if r.label == 0]
        fake_scores = [r.predicted_score for r in self.results if r.label == 1]

        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(0, 1, 40)
        ax.hist(real_scores, bins=bins, alpha=0.6, label="Real", color="#2196F3")
        ax.hist(fake_scores, bins=bins, alpha=0.6, label="Fake", color="#F44336")
        ax.axvline(x=0.35, color="orange", ls="--", lw=1, label="Suspicious")
        ax.axvline(x=0.70, color="red", ls="--", lw=1, label="Manipulated")
        ax.set_xlabel("Predicted Score")
        ax.set_ylabel("Count")
        ax.set_title("Score Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = output_dir / "score_distribution.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_precision_recall(self, output_dir: Path) -> Path:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(
            self.metrics.pr_recall, self.metrics.pr_precision,
            lw=2, color="#4CAF50",
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.grid(True, alpha=0.3)
        path = output_dir / "precision_recall.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_confusion_matrix(
        self, output_dir: Path, threshold: float = 0.50,
    ) -> Path:
        labels = np.array([r.label for r in self.results])
        preds = np.array([
            1 if r.predicted_score >= threshold else 0 for r in self.results
        ])

        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(labels, preds)
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Real", "Fake"])
        ax.set_yticklabels(["Real", "Fake"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix (threshold={threshold})")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16)
        fig.colorbar(im, ax=ax)
        path = output_dir / "confusion_matrix.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_signal_bars(self, output_dir: Path) -> Path:
        signals = sorted(self.metrics.signal_aucs.items(), key=lambda x: x[1], reverse=True)
        names = [s[0] for s in signals]
        aucs = [s[1] for s in signals]

        fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.5)))
        colors = ["#4CAF50" if a > 0.7 else "#FF9800" if a > 0.5 else "#F44336" for a in aucs]
        ax.barh(names, aucs, color=colors)
        ax.set_xlabel("AUC")
        ax.set_title("Per-Signal AUC")
        ax.set_xlim(0, 1)
        ax.axvline(x=0.5, color="gray", ls="--", lw=1, alpha=0.5)
        for i, v in enumerate(aucs):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
        ax.grid(True, axis="x", alpha=0.3)
        path = output_dir / "signal_aucs.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path
