"""
Run VeriFi benchmark on a deepfake detection dataset.

Usage:
  python scripts/run_benchmark.py --dataset ff++ --root /path/to/FaceForensics++
  python scripts/run_benchmark.py --dataset celebdf --root /path/to/Celeb-DF-v2 --max-videos 200
  python scripts/run_benchmark.py --dataset ff++ --root /path/to/FF++ --methods Deepfakes Face2Face
  python scripts/run_benchmark.py --resume data/benchmarks/ff++_c23_20260512_143022
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Run VeriFi benchmark")
    parser.add_argument("--dataset", type=str, help="Dataset name: ff++, celebdf, dfdc, df40")
    parser.add_argument("--root", type=str, help="Path to dataset root directory")
    parser.add_argument("--compression", type=str, default="c23", help="Compression level (ff++)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--methods", nargs="+", help="Specific methods to evaluate")
    parser.add_argument("--max-videos", type=int, help="Maximum videos to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--resume", type=str, help="Resume from a previous run directory")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--no-stratified", action="store_true", help="Disable stratified sampling")
    args = parser.parse_args()

    from verifi.benchmark.metrics import MetricsComputer
    from verifi.benchmark.results import ResultsReader
    from verifi.benchmark.runner import BenchmarkRunner, RunConfig
    from verifi.benchmark.visualizer import PlotGenerator
    from verifi.config import AppConfig
    from verifi.pipeline.orchestrator import VeriFiPipeline

    if args.resume:
        run_dir = Path(args.resume)
        reader = ResultsReader(run_dir)
        config_dict = reader.load_config()
        config = RunConfig(**config_dict)
        output_dir = run_dir
    else:
        if not args.dataset or not args.root:
            parser.error("--dataset and --root are required (or use --resume)")
        config = RunConfig(
            dataset_name=args.dataset,
            dataset_root=args.root,
            compression=args.compression,
            split=args.split,
            methods=args.methods,
            max_videos=args.max_videos,
            stratified=not args.no_stratified,
            seed=args.seed,
        )
        output_dir = Path(args.output) if args.output else None

    print("=" * 65)
    print("VeriFi — Benchmark Runner")
    print("=" * 65)
    print(f"\n  Dataset:      {config.dataset_name}")
    print(f"  Root:         {config.dataset_root}")
    print(f"  Compression:  {config.compression}")
    print(f"  Split:        {config.split}")
    print(f"  Max videos:   {config.max_videos or 'all'}")
    print(f"  Methods:      {config.methods or 'all'}")

    # Load pipeline
    print("\n── Loading models ──")
    t0 = time.perf_counter()
    app_config = AppConfig()
    app_config.sampling.min_laplacian_var = 50.0
    pipeline = VeriFiPipeline(app_config)
    pipeline.load_models()
    print(f"  Models loaded in {time.perf_counter() - t0:.1f}s")

    # Run benchmark
    print("\n── Running benchmark ──")
    runner = BenchmarkRunner(pipeline, config, output_dir)
    result_dir = runner.run()

    # Compute metrics
    print("\n── Computing metrics ──")
    reader = ResultsReader(result_dir)
    results = reader.load_all()

    if len(results) < 2:
        print("  Not enough results to compute metrics.")
        pipeline.unload_models()
        return

    computer = MetricsComputer(results)
    metrics = computer.compute_all()
    metrics.save(result_dir / "metrics.json")

    print(f"\n  AUC:          {metrics.auc:.4f}")
    print(f"  EER:          {metrics.eer:.4f}")
    print(f"  Accuracy:     {metrics.accuracy:.4f}")
    print(f"  Precision:    {metrics.precision:.4f}")
    print(f"  Recall:       {metrics.recall:.4f}")
    print(f"  F1:           {metrics.f1:.4f}")
    print(f"  Real/Fake:    {metrics.n_real}/{metrics.n_fake}")
    print(f"  Errors:       {metrics.n_errors}")

    if metrics.per_method:
        print("\n  ┌─ Per-Method AUC ──────────────")
        for method, data in sorted(metrics.per_method.items()):
            print(f"  │ {method:<20} AUC={data['auc']:.4f}  (n={data['n_fake']})")
        print("  └─────────────────────────────")

    if metrics.signal_aucs:
        print("\n  ┌─ Per-Signal AUC ──────────────")
        for sig, sig_auc in sorted(metrics.signal_aucs.items(), key=lambda x: x[1], reverse=True):
            print(f"  │ {sig:<20} AUC={sig_auc:.4f}")
        print("  └─────────────────────────────")

    # Generate plots
    print("\n── Generating plots ──")
    plots_dir = result_dir / "plots"
    plotter = PlotGenerator(metrics, results)
    plot_paths = plotter.generate_all(plots_dir)
    for p in plot_paths:
        print(f"  Saved: {p}")

    pipeline.unload_models()

    print(f"\n{'=' * 65}")
    print(f"  Results saved to: {result_dir}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
