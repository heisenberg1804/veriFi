"""
Offline ensemble weight + threshold tuner.

Stage 1 — Weight search (threshold-invariant):
  Sample N=5000 weight vectors from Dirichlet distribution, score all videos,
  compute AUC for each. Report top configs + drop-one ablation for all signals.

Stage 2 — Threshold search (threshold-dependent):
  Using the best weight vector from Stage 1, sweep threshold pairs and report
  balanced accuracy, TPR @ FPR=0.10, and confusion matrices.

CPU-only, no model loading.

Usage:
    python scripts/tune_weights.py data/benchmarks/<run_id>/
    python scripts/tune_weights.py data/benchmarks/<run_id>/ --n-samples 10000
    python scripts/tune_weights.py data/benchmarks/<run_id>/ --frame-subsample 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verifi.benchmark.results import ResultsReader
from verifi.ensemble.aggregator import EnsembleWeights, score_from_signals

FRAME_SIGNAL_NAMES = ["dct", "noise_residual", "temporal", "clip", "channel_corr"]
FACE_SIGNAL_NAMES = [
    "dct", "clip", "noise_residual", "effnet", "temporal", "channel_corr",
]


def load_dataset(
    run_dir: Path, frame_subsample: int | None = None,
) -> list[dict]:
    reader = ResultsReader(run_dir)
    results = reader.load_all()

    videos = []
    skipped = 0
    for r in results:
        if r.error or not r.frame_signals:
            skipped += 1
            continue

        frame_sigs = [fs.signals for fs in r.frame_signals]
        face_sigs = [fs.signals for fs in r.face_signals]

        if frame_subsample and frame_subsample < len(frame_sigs):
            frame_sigs = frame_sigs[:frame_subsample]

        videos.append({
            "label": r.label,
            "method": r.method,
            "frame_signals": frame_sigs,
            "face_signals": face_sigs,
            "path": r.video_path,
        })

    print(f"Loaded {len(videos)} videos ({skipped} skipped)")
    real = sum(1 for v in videos if v["label"] == 0)
    fake = sum(1 for v in videos if v["label"] == 1)
    print(f"  Real: {real}, Fake: {fake}")
    return videos


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)[::-1]
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    total_pos = labels_sorted.sum()
    total_neg = len(labels_sorted) - total_pos
    if total_pos == 0 or total_neg == 0:
        return 0.5
    tpr = tp / total_pos
    fpr = fp / total_neg
    return float(np.trapezoid(tpr, fpr))


def compute_roc_curve(
    labels: np.ndarray, scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(scores)[::-1]
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    total_pos = labels_sorted.sum()
    total_neg = len(labels_sorted) - total_pos
    if total_pos == 0 or total_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
    tpr = tp / total_pos
    fpr = fp / total_neg
    return fpr, tpr


def tpr_at_fpr(
    labels: np.ndarray, scores: np.ndarray, target_fpr: float = 0.10,
) -> float:
    fpr, tpr = compute_roc_curve(labels, scores)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx])


def score_all(videos: list[dict], weights: EnsembleWeights) -> np.ndarray:
    scores = np.empty(len(videos))
    for i, v in enumerate(videos):
        result = score_from_signals(
            v["frame_signals"], v["face_signals"], weights,
        )
        scores[i] = result.video_score
    return scores


def make_weights_from_frame_vec(
    w: np.ndarray, face_vec: np.ndarray | None = None,
) -> EnsembleWeights:
    """Build EnsembleWeights from a frame-path weight vector (5 elements)."""
    return EnsembleWeights(
        frame_dct=float(w[0]),
        frame_noise_residual=float(w[1]),
        frame_temporal=float(w[2]),
        frame_clip=float(w[3]),
        frame_channel_corr=float(w[4]),
        face_dct=float(face_vec[0]) if face_vec is not None else 0.30,
        face_clip=float(face_vec[1]) if face_vec is not None else 0.20,
        face_noise_residual=float(face_vec[2]) if face_vec is not None else 0.20,
        face_effnet=float(face_vec[3]) if face_vec is not None else 0.15,
        face_temporal=float(face_vec[4]) if face_vec is not None else 0.10,
        face_channel_corr=float(face_vec[5]) if face_vec is not None else 0.05,
    )


# ─── Stage 1: Weight search via Dirichlet ───


def stage1_weight_search(
    videos: list[dict],
    n_samples: int = 5000,
    seed: int = 42,
) -> tuple[np.ndarray, float, list[tuple[float, np.ndarray]]]:
    """
    Sample weight vectors from Dirichlet, compute AUC for each.
    Returns (best_weights, best_auc, sorted_results).
    """
    labels = np.array([v["label"] for v in videos])
    rng = np.random.default_rng(seed)

    # Dirichlet alpha — uniform (alpha=1) gives unbiased random simplices
    alpha = np.ones(len(FRAME_SIGNAL_NAMES))
    all_results: list[tuple[float, np.ndarray]] = []

    for _ in range(n_samples):
        w = rng.dirichlet(alpha)
        weights = make_weights_from_frame_vec(w)
        scores = score_all(videos, weights)
        auc = compute_auc(labels, scores)
        all_results.append((auc, w.copy()))

    all_results.sort(key=lambda x: x[0], reverse=True)
    best_auc, best_w = all_results[0]
    return best_w, best_auc, all_results


def print_weight_table(results: list[tuple[float, np.ndarray]], top_n: int = 15):
    names = FRAME_SIGNAL_NAMES
    hdr = f"{'AUC':>7s}  " + "  ".join(f"{n[:5]:>5s}" for n in names)
    print(hdr)
    for i, (auc, w) in enumerate(results[:top_n]):
        row = f"{auc:7.4f}  " + "  ".join(f"{v:5.3f}" for v in w)
        if i == 0:
            row += "  <-- BEST"
        print(row)


# ─── Drop-one ablation ───


def drop_one_ablation(
    videos: list[dict],
    base_weights: np.ndarray,
    base_auc: float,
) -> list[tuple[str, float, float]]:
    """
    For each signal, zero its weight, redistribute to others, compute AUC.
    Returns list of (signal_name, ablated_auc, delta).
    """
    labels = np.array([v["label"] for v in videos])
    results = []

    for i, name in enumerate(FRAME_SIGNAL_NAMES):
        ablated = base_weights.copy()
        if ablated[i] == 0.0:
            results.append((name, base_auc, 0.0))
            continue

        ablated[i] = 0.0
        remaining = ablated.sum()
        if remaining > 0:
            ablated = ablated / remaining
        else:
            ablated = np.ones_like(ablated) / (len(ablated) - 1)
            ablated[i] = 0.0

        weights = make_weights_from_frame_vec(ablated)
        scores = score_all(videos, weights)
        auc = compute_auc(labels, scores)
        results.append((name, auc, auc - base_auc))

    return results


# ─── Stage 2: Threshold search ───


def balanced_accuracy(
    labels: np.ndarray,
    scores: np.ndarray,
    sus_t: float,
    manip_t: float,
) -> float:
    """
    Binary balanced accuracy: predict fake if score >= sus_t.
    (manip_t is used for the three-class verdict but not for this metric.)
    """
    pred = (scores >= sus_t).astype(int)
    tp = ((pred == 1) & (labels == 1)).sum()
    tn = ((pred == 0) & (labels == 0)).sum()
    total_pos = labels.sum()
    total_neg = len(labels) - total_pos
    tpr = tp / total_pos if total_pos > 0 else 0.0
    tnr = tn / total_neg if total_neg > 0 else 0.0
    return float((tpr + tnr) / 2)


def confusion_matrix(
    labels: np.ndarray,
    scores: np.ndarray,
    sus_t: float,
) -> tuple[int, int, int, int]:
    """Returns (TP, FP, TN, FN) at threshold sus_t."""
    pred = (scores >= sus_t).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    return tp, fp, tn, fn


THRESHOLD_PAIRS = [
    (0.25, 0.60), (0.25, 0.65), (0.25, 0.70),
    (0.30, 0.60), (0.30, 0.65), (0.30, 0.70),
    (0.35, 0.60), (0.35, 0.65), (0.35, 0.70),
    (0.40, 0.65), (0.40, 0.70), (0.40, 0.75),
    (0.45, 0.70), (0.45, 0.75), (0.45, 0.80),
]


def stage2_threshold_search(
    videos: list[dict],
    best_weights: np.ndarray,
) -> list[tuple[float, float, float, float, tuple[int, int, int, int]]]:
    """
    Sweep threshold pairs using best weights from Stage 1.
    Returns sorted list of (bal_acc, sus_t, manip_t, tpr_at_fpr10, confusion).
    """
    labels = np.array([v["label"] for v in videos])
    base_ew = make_weights_from_frame_vec(best_weights)
    scores = score_all(videos, base_ew)

    results = []
    for sus_t, manip_t in THRESHOLD_PAIRS:
        ba = balanced_accuracy(labels, scores, sus_t, manip_t)
        tpr10 = tpr_at_fpr(labels, scores, target_fpr=0.10)
        cm = confusion_matrix(labels, scores, sus_t)
        results.append((ba, sus_t, manip_t, tpr10, cm))

    results.sort(key=lambda x: x[0], reverse=True)
    return results


# ─── Main ───


def main():
    parser = argparse.ArgumentParser(
        description="Offline ensemble weight + threshold tuner",
    )
    parser.add_argument(
        "run_dir", type=Path, help="Path to benchmark results directory",
    )
    parser.add_argument(
        "--n-samples", type=int, default=5000,
        help="Dirichlet weight samples (default 5000)",
    )
    parser.add_argument(
        "--frame-subsample", type=int, default=None,
        help="Use only the first N frames per video",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    videos = load_dataset(args.run_dir, args.frame_subsample)
    if len(videos) < 2:
        print("ERROR: Need at least 2 videos (1 real, 1 fake)")
        sys.exit(1)

    labels = np.array([v["label"] for v in videos])

    # ── Stage 1: Weight search ──
    print(f"\n{'='*65}")
    print(f"STAGE 1: WEIGHT SEARCH (Dirichlet, N={args.n_samples})")
    print(f"{'='*65}")

    best_w, best_auc, all_results = stage1_weight_search(
        videos, n_samples=args.n_samples, seed=args.seed,
    )

    print("\n--- Top 15 weight vectors by AUC ---")
    print_weight_table(all_results, top_n=15)

    print(f"\nBest AUC: {best_auc:.4f}")
    print(f"Best weights: {dict(zip(FRAME_SIGNAL_NAMES, best_w))}")

    # ── Drop-one ablation ──
    print(f"\n{'='*65}")
    print("DROP-ONE ABLATION (all signals)")
    print(f"{'='*65}")

    ablation = drop_one_ablation(videos, best_w, best_auc)
    print(f"\n{'Signal':<16s} {'AUC w/o':>8s} {'Delta':>8s}  {'Verdict'}")
    for name, auc, delta in ablation:
        if delta < -0.01:
            verdict = "VALUABLE"
        elif delta > 0.01:
            verdict = "HARMFUL — drop improves AUC"
        else:
            verdict = "marginal"
        print(f"{name:<16s} {auc:8.4f} {delta:+8.4f}  {verdict}")

    # ── Stage 2: Threshold search ──
    print(f"\n{'='*65}")
    print("STAGE 2: THRESHOLD SEARCH (best weights from Stage 1)")
    print(f"{'='*65}")

    base_ew = make_weights_from_frame_vec(best_w)
    scores = score_all(videos, base_ew)
    auc_check = compute_auc(labels, scores)
    tpr10_global = tpr_at_fpr(labels, scores, target_fpr=0.10)
    print(f"\nUsing weights AUC={auc_check:.4f}, TPR@FPR=0.10: {tpr10_global:.4f}")

    thresh_results = stage2_threshold_search(videos, best_w)

    print(f"\n{'BalAcc':>7s}  {'Sus':>5s} {'Man':>5s}  "
          f"{'TP':>4s} {'FP':>4s} {'TN':>4s} {'FN':>4s}")
    for ba, sus_t, manip_t, tpr10, (tp, fp, tn, fn) in thresh_results[:10]:
        marker = " <-- BEST" if ba == thresh_results[0][0] else ""
        print(f"{ba:7.4f}  {sus_t:5.2f} {manip_t:5.2f}  "
              f"{tp:4d} {fp:4d} {tn:4d} {fn:4d}{marker}")

    # Best threshold summary
    best_ba, best_sus, best_man, best_tpr10, best_cm = thresh_results[0]
    print(f"\nBest threshold pair: sus={best_sus:.2f}, manip={best_man:.2f}")
    print(f"Balanced accuracy: {best_ba:.4f}")
    print(f"TPR@FPR=0.10:      {tpr10_global:.4f}")
    tp, fp, tn, fn = best_cm
    print(f"Confusion matrix:  TP={tp} FP={fp} TN={tn} FN={fn}")

    # ── Summary: recommended config ──
    print(f"\n{'='*65}")
    print("RECOMMENDED CONFIG")
    print(f"{'='*65}")
    print("Frame weights:")
    for name, w in zip(FRAME_SIGNAL_NAMES, best_w):
        print(f"  {name}: {w:.4f}")
    print("Thresholds:")
    print(f"  suspicious: {best_sus:.2f}")
    print(f"  manipulated: {best_man:.2f}")
    print("Metrics:")
    print(f"  AUC: {best_auc:.4f}")
    print(f"  Balanced accuracy: {best_ba:.4f}")
    print(f"  TPR@FPR=0.10: {tpr10_global:.4f}")


if __name__ == "__main__":
    main()
