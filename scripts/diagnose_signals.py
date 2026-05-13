"""
Signal-level diagnostic: raw score distributions across all sample videos.

Run: python scripts/diagnose_signals.py

Extracts 10 uniform frames per video and runs each detector independently.
No pipeline, no ensemble — just raw signal values for calibration.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def extract_frames(video_path: str, n: int = 10) -> list[np.ndarray]:
    """Extract n uniformly spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def get_video_meta(video_path: str) -> dict:
    """Get resolution, codec, bitrate via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        probe = json.loads(result.stdout)
    except Exception:
        return {"resolution": "?", "codec": "?", "bitrate": "?"}

    vs = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), {})
    fmt = probe.get("format", {})
    bitrate_kbps = int(fmt.get("bit_rate", 0)) // 1000
    return {
        "resolution": f"{vs.get('width', '?')}x{vs.get('height', '?')}",
        "codec": vs.get("codec_name", "?"),
        "bitrate": f"{bitrate_kbps}kbps" if bitrate_kbps else "?",
        "duration": f"{float(fmt.get('duration', 0)):.1f}s",
    }


def stats(scores: list[float]) -> dict:
    """Compute min/mean/max for a list of scores."""
    if not scores:
        return {"min": 0, "mean": 0, "max": 0}
    return {
        "min": float(np.min(scores)),
        "mean": float(np.mean(scores)),
        "max": float(np.max(scores)),
    }


def compute_sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main():
    import torch

    from verifi.config import AppConfig
    from verifi.detectors.clip_detector import CLIPDeepfakeDetector
    from verifi.detectors.effnet_detector import EfficientNetDetector
    from verifi.detectors.frequency import FrequencyAnalyzer
    from verifi.detectors.jpeg_ghost import JPEGGhostAnalyzer
    from verifi.detectors.noise_residual import NoiseResidualAnalyzer

    config = AppConfig()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # ── Load models ──
    print("Loading models...")
    clip_det = CLIPDeepfakeDetector(
        weight_path=str(config.detector.clip_weight_path), device=device
    )
    clip_det.load()

    effnet_det = EfficientNetDetector(
        weight_path=str(config.detector.effnet_weight_path), device=device
    )
    effnet_det.load()

    freq = FrequencyAnalyzer()
    noise_det = NoiseResidualAnalyzer()
    jpeg_det = JPEGGhostAnalyzer()
    print("Models loaded.\n")

    # ── Find videos ──
    sample_dir = Path("data/sample_videos")
    videos = sorted(sample_dir.glob("*.mp4")) + sorted(sample_dir.glob("*.mov"))
    if not videos:
        print("No videos found in data/sample_videos/")
        return

    all_results = []

    for vpath in videos:
        print(f"{'=' * 70}")
        print(f"  {vpath.name}")
        print(f"{'=' * 70}")

        meta = get_video_meta(str(vpath))
        print(f"  Resolution: {meta['resolution']}  Codec: {meta['codec']}  "
              f"Bitrate: {meta['bitrate']}  Duration: {meta['duration']}")

        frames = extract_frames(str(vpath), n=10)
        if not frames:
            print("  [SKIP] Could not extract frames\n")
            continue

        print(f"  Extracted {len(frames)} frames\n")

        # ── Run CLIP on each frame ──
        clip_scores = []
        clip_results = clip_det.predict(frames)
        clip_scores = [r.score for r in clip_results]

        # ── Run EfficientNet on each frame (full frame, not face crop) ──
        effnet_scores = []
        effnet_results = effnet_det.predict(frames)
        effnet_scores = [r.score for r in effnet_results]

        # ── Run DCT on each frame (with sharpness normalization) ──
        dct_scores = []
        dct_hf_ratios = []
        dct_ch_corrs = []
        for frame in frames:
            sharpness = compute_sharpness(frame)
            result = freq.analyze(frame, sharpness=sharpness)
            dct_scores.append(result.score)
            dct_hf_ratios.append(result.metadata.get("high_ratio", 0))
            dct_ch_corrs.append(result.metadata.get("channel_corr_mean", 0))

        # ── Run Noise Residual on each frame ──
        noise_scores = []
        for frame in frames:
            result = noise_det.analyze(frame)
            noise_scores.append(result.score)

        # ── Run JPEG Ghost on each frame ──
        jpeg_scores = []
        for frame in frames:
            result = jpeg_det.analyze(frame)
            jpeg_scores.append(result.score)

        cs = stats(clip_scores)
        es = stats(effnet_scores)
        ds = stats(dct_scores)
        hf = stats(dct_hf_ratios)
        ch = stats(dct_ch_corrs)
        ns = stats(noise_scores)
        js = stats(jpeg_scores)

        # ── Print per-frame table ──
        cols = ["Frame", "CLIP", "EffNet", "DCT", "HF rat", "ChCorr", "Noise", "JPEG"]
        print("  " + "  ".join(f"{c:>6}" for c in cols))
        print("  " + "  ".join("─" * 6 for _ in cols))
        for i in range(len(frames)):
            c = clip_scores[i] if i < len(clip_scores) else 0
            e = effnet_scores[i] if i < len(effnet_scores) else 0
            d = dct_scores[i] if i < len(dct_scores) else 0
            h = dct_hf_ratios[i] if i < len(dct_hf_ratios) else 0
            cc = dct_ch_corrs[i] if i < len(dct_ch_corrs) else 0
            n = noise_scores[i] if i < len(noise_scores) else 0
            j = jpeg_scores[i] if i < len(jpeg_scores) else 0
            vals = [c, e, d, h, cc, n, j]
            row = f"  {i:>6}" + "".join(f"  {v:>6.3f}" for v in vals)
            print(row)

        print()
        print(f"  {'Signal':<14}  {'Min':>6}  {'Mean':>6}  {'Max':>6}")
        print(f"  {'─' * 14}  {'─' * 6}  {'─' * 6}  {'─' * 6}")
        print(f"  {'CLIP':<14}  {cs['min']:>6.3f}  {cs['mean']:>6.3f}  {cs['max']:>6.3f}")
        print(f"  {'EfficientNet':<14}  {es['min']:>6.3f}  {es['mean']:>6.3f}  {es['max']:>6.3f}")
        print(f"  {'DCT':<14}  {ds['min']:>6.3f}  {ds['mean']:>6.3f}  {ds['max']:>6.3f}")
        print(f"  {'HF ratio':<14}  {hf['min']:>6.3f}  {hf['mean']:>6.3f}  {hf['max']:>6.3f}")
        print(f"  {'ChCorr':<14}  {ch['min']:>6.3f}  {ch['mean']:>6.3f}  {ch['max']:>6.3f}")
        print(f"  {'NoiseResidual':<14}  {ns['min']:>6.3f}  {ns['mean']:>6.3f}  {ns['max']:>6.3f}")
        print(f"  {'JPEGGhost':<14}  {js['min']:>6.3f}  {js['mean']:>6.3f}  {js['max']:>6.3f}")
        print()

        # Simulated frame-path ensemble (DCT 0.30, NR 0.25, CLIP 0.15, ChCorr 0.05)
        # No temporal in this diagnostic (requires frame pairs)
        frame_ensemble_scores = []
        for i in range(len(frames)):
            c = clip_scores[i] if i < len(clip_scores) else 0
            d = dct_scores[i] if i < len(dct_scores) else 0
            cc = dct_ch_corrs[i] if i < len(dct_ch_corrs) else 0
            n = noise_scores[i] if i < len(noise_scores) else 0
            ens = 0.30 * d + 0.25 * n + 0.15 * c + 0.05 * cc
            frame_ensemble_scores.append(ens / 0.75)  # normalize (no temporal)
        ens_stats = stats(frame_ensemble_scores)
        print(f"  Simulated ensemble (mean all frames): {ens_stats['mean']:.3f}  "
              f"(thresh: 0.35=SUSPICIOUS, 0.70=MANIPULATED)")
        print()

        all_results.append({
            "video": vpath.name,
            "meta": meta,
            "clip": cs,
            "effnet": es,
            "dct": ds,
            "hf_ratio": hf,
            "channel_corr": ch,
            "noise": ns,
            "jpeg_ghost": js,
            "ensemble": ens_stats,
        })

    # ── Summary comparison ──
    if len(all_results) > 1:
        print(f"\n{'=' * 110}")
        print("  SUMMARY — Signal Comparison Across Videos")
        print(f"{'=' * 110}\n")

        header = (
            f"  {'Video':<35} {'DCT':>7} {'NR':>7}"
            f" {'CLIP':>7} {'ChCorr':>7} {'Ens':>7} {'Verdict':>18}"
        )
        subhdr = (
            f"  {'':35} {'mean':>7} {'mean':>7}"
            f" {'mean':>7} {'mean':>7} {'mean':>7} {'':>18}"
        )
        print(header)
        print(subhdr)
        print(f"  {'─' * 100}")

        for r in all_results:
            name = r["video"][:33]
            dct_m = r["dct"]["mean"]
            nr_m = r["noise"]["mean"]
            clip_m = r["clip"]["mean"]
            ch_m = r["channel_corr"]["mean"]
            ens_mean = r["ensemble"]["mean"]
            if ens_mean >= 0.70:
                verdict = "MANIPULATED"
            elif ens_mean >= 0.35:
                verdict = "SUSPICIOUS"
            else:
                verdict = "AUTHENTIC"
            print(
                f"  {name:<35} {dct_m:>7.3f} {nr_m:>7.3f}"
                f" {clip_m:>7.3f} {ch_m:>7.3f} {ens_mean:>7.3f} {verdict:>18}"
            )

        print()


if __name__ == "__main__":
    main()
