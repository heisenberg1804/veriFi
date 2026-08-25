#!/usr/bin/env python3
"""
Calibrate physics reasoning on all test videos.
Runs PhysicsReasoner (Ollama backend) on each video and compares scores.

Usage:
    python scripts/calibrate_physics.py
    python scripts/calibrate_physics.py --backend claude
    python scripts/calibrate_physics.py --max-frames 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from verifi.detectors.physics_reasoner import PhysicsReasoner

VIDEO_DIR = Path("data/sample_videos")

GROUND_TRUTH = {
    "Capybara_Walks_Oscars_Red_Carpet.mp4": ("AI", "Veo (Google Gemini)"),
    "WasifAI - No one Will believe": ("AI", "Unknown AI generator"),
    "SocialSight - cameraman knew": ("AI", "Veo (Google Gemini)"),
    "Mansur -": ("AI", "Veo (Google Gemini)"),
    "Victoria Repa -": ("AI", "Unknown AI generator"),
    "real_sample.mp4": ("Real", "Real camera"),
    "Florentino": ("Real", "Real camera"),
    "Lionel Messi": ("Real", "Real broadcast"),
    "iPhone": ("Real", "Real iPhone"),
    "Ben 10": ("Real", "Animated TV"),
}


def match_ground_truth(filename: str) -> tuple[str, str]:
    for key, (label, source) in GROUND_TRUTH.items():
        if key in filename:
            return label, source
    return "Unknown", "Unknown"


def extract_frames(video_path: Path, num_frames: int = 8) -> list[tuple[np.ndarray, int, float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    if total <= 0:
        return []

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append((frame, int(idx), float(idx / fps)))

    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser(description="Calibrate physics reasoning")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "claude"])
    parser.add_argument("--model", default="llava:7b")
    parser.add_argument("--max-frames", type=int, default=6)
    parser.add_argument("--video", type=str, default=None, help="Single video to test")
    args = parser.parse_args()

    reasoner = PhysicsReasoner(
        backend=args.backend,
        ollama_model=args.model,
        max_frames=args.max_frames,
    )

    if args.video:
        videos = [Path(args.video)]
    else:
        if not VIDEO_DIR.exists():
            print(f"Video directory not found: {VIDEO_DIR}")
            sys.exit(1)
        videos = sorted(VIDEO_DIR.glob("*.mp4"))

    if not videos:
        print("No videos found.")
        sys.exit(1)

    print(f"\nPhysics Reasoning Calibration ({args.backend} / {args.model})")
    print(f"Max frames per video: {args.max_frames}")
    print("=" * 100)

    results = []

    for vpath in videos:
        label, source = match_ground_truth(vpath.name)
        short_name = vpath.name[:50] + "..." if len(vpath.name) > 50 else vpath.name

        print(f"\n{'─' * 80}")
        print(f"Video: {short_name}")
        print(f"Label: {label} ({source})")

        frame_data = extract_frames(vpath, num_frames=args.max_frames + 2)
        if not frame_data:
            print("  ERROR: Could not extract frames")
            continue

        frames = [f[0] for f in frame_data]
        indices = [f[1] for f in frame_data]
        timestamps = [f[2] for f in frame_data]

        t0 = time.perf_counter()
        try:
            result = reasoner.analyze(frames, indices, timestamps)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        elapsed = time.perf_counter() - t0

        print(f"  Time: {elapsed:.1f}s | Frames: {result.num_frames_analyzed}")
        agg = result.aggregate_score
        conf = result.confidence
        print(f"  Aggregate score: {agg:.3f} | Confidence: {conf:.3f}")
        print(f"  Verdict: {result.verdict_suggestion}")

        if result.per_frame:
            print("\n  Per-frame scores:")
            hdr = (
                f"  {'Frame':>8} {'Time':>6} {'Shadow':>7} "
                f"{'Reflect':>8} {'Persp':>6} {'Anat':>6} "
                f"{'Text':>6} {'Overall':>8}"
            )
            print(hdr)
            for fr in result.per_frame:
                print(
                    f"  {fr.frame_idx:>8} {fr.timestamp_sec:>5.1f}s "
                    f"{fr.shadow_score:>7.2f} {fr.reflection_score:>8.2f} "
                    f"{fr.perspective_score:>6.2f} {fr.anatomy_score:>6.2f} "
                    f"{fr.texture_score:>6.2f} {fr.overall_score:>8.2f}"
                )

        if result.evidence_summary:
            print("\n  Key evidence:")
            for ev in result.evidence_summary[:5]:
                print(f"    - {ev}")

        results.append({
            "name": short_name,
            "label": label,
            "score": result.aggregate_score,
            "confidence": result.confidence,
            "verdict": result.verdict_suggestion,
            "time": elapsed,
        })

    # Summary table
    print(f"\n\n{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")
    print(f"{'Video':<55} {'Label':>6} {'Score':>6} {'Conf':>6} {'Verdict':<20} {'Time':>6}")
    print("-" * 100)

    real_scores = []
    ai_scores = []

    for r in results:
        print(
            f"{r['name']:<55} {r['label']:>6} {r['score']:>6.3f} "
            f"{r['confidence']:>6.2f} {r['verdict']:<20} {r['time']:>5.1f}s"
        )
        if r["label"] == "AI":
            ai_scores.append(r["score"])
        elif r["label"] == "Real":
            real_scores.append(r["score"])

    if real_scores and ai_scores:
        print(f"\n{'─' * 60}")
        r_lo, r_hi = min(real_scores), max(real_scores)
        a_lo, a_hi = min(ai_scores), max(ai_scores)
        print(f"Real mean:  {np.mean(real_scores):.3f} "
              f"(range: {r_lo:.3f}-{r_hi:.3f})")
        print(f"AI mean:    {np.mean(ai_scores):.3f} "
              f"(range: {a_lo:.3f}-{a_hi:.3f})")
        gap = a_lo - r_hi
        print(f"Separation: {gap:+.3f} "
              f"({'GOOD' if gap > 0 else 'OVERLAP'})")

        correct = sum(
            1 for r in results
            if (r["label"] == "AI" and r["score"] >= 0.5)
            or (r["label"] == "Real" and r["score"] < 0.5)
        )
        total = len(results)
        print(f"Accuracy:   {correct}/{total} "
              f"({100 * correct / total:.0f}%)")


if __name__ == "__main__":
    main()
