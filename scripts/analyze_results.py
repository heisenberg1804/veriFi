"""
Recompute metrics and regenerate plots from a previous benchmark run.

Usage:
  python scripts/analyze_results.py data/benchmarks/ff++_c23_20260512_143022
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_results.py <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        sys.exit(1)

    from verifi.benchmark.metrics import MetricsComputer
    from verifi.benchmark.results import ResultsReader
    from verifi.benchmark.visualizer import PlotGenerator

    reader = ResultsReader(run_dir)
    results = reader.load_all()
    config = reader.load_config()

    print(f"Loaded {len(results)} results from {run_dir}")
    print(f"Config: {config}")

    if len(results) < 2:
        print("Not enough results to compute metrics.")
        return

    computer = MetricsComputer(results)
    metrics = computer.compute_all()
    metrics.save(run_dir / "metrics.json")

    print(f"\nAUC:       {metrics.auc:.4f}")
    print(f"EER:       {metrics.eer:.4f}")
    print(f"Accuracy:  {metrics.accuracy:.4f}")
    print(f"F1:        {metrics.f1:.4f}")
    print(f"Real/Fake: {metrics.n_real}/{metrics.n_fake}")

    if metrics.per_method:
        print("\nPer-Method:")
        for method, data in sorted(metrics.per_method.items()):
            print(f"  {method:<20} AUC={data['auc']:.4f}")

    if metrics.signal_aucs:
        print("\nPer-Signal AUC:")
        for sig, sig_auc in sorted(
            metrics.signal_aucs.items(), key=lambda x: x[1], reverse=True,
        ):
            print(f"  {sig:<20} AUC={sig_auc:.4f}")

    plots_dir = run_dir / "plots"
    plotter = PlotGenerator(metrics, results)
    plot_paths = plotter.generate_all(plots_dir)
    print(f"\nGenerated {len(plot_paths)} plots in {plots_dir}")


if __name__ == "__main__":
    main()
