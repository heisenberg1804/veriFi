# VeriFi

Forensic detection engine for AI-generated video. Multi-signal ensemble analysis, GradCAM explainability, and LLM-driven forensic investigation.

> **Status: pre-validation.** The pipeline is complete and instrumented, but the detector has **not yet been benchmarked on a dataset**. No AUC is claimed. See [Current status](#current-status) for an honest account of what works, what doesn't, and what's next.

---

## The problem

AI-generated video is increasingly indistinguishable from authentic footage. Most detection tools return a confidence score and nothing else — which is unusable for journalists, trust-and-safety teams, or legal reviewers who need to know *what* was manipulated, *where*, and *why* the system flagged it.

Most published detectors are also supervised classifiers trained on 2019-era face-swap datasets. They degrade sharply on generators they weren't trained against, and modern text-to-video models (Veo, Sora, Kling, Runway) are exactly that.

## Approach

VeriFi leads with **signal-processing detectors that require no training data**, and treats learned models as supporting evidence rather than the verdict. Two independent analysis paths run in parallel:

- **Path A (face-level)** — activates when faces are detected. Targets face-swap and reenactment.
- **Path B (frame-level)** — always active. Targets fully synthetic generation.

Results feed a weighted ensemble, then optionally a multimodal physics reasoner and a LangGraph agent that calls forensic tools and writes a structured report.

**Input:** video file (MP4/MOV/WebM) or URL
**Output:** three-tier verdict, per-frame confidence timeline, GradCAM heatmaps, DCT spectrum views, natural-language explanation, structured JSON

---

## Pipeline

```
Video
  └─ Validation (ffprobe) → Scene detection → Two-pass frame selection → Quality gate
       └─ MTCNN face detection
            ├─ Path A (face crops)  : DCT · CLIP · Noise residual · EfficientNet · Temporal · ChCorr
            └─ Path B (full frames) : DCT · Noise residual · Temporal · CLIP · ChCorr
                 └─ Weighted ensemble → verdict + confidence
                      ├─ Physics reasoner (multimodal LLM, Tier 2)
                      ├─ GradCAM + forensic views
                      └─ LangGraph agent → forensic report
```

### Detection signals

| Signal | Type | Premise |
|---|---|---|
| **Noise residual** | Statistical | Real sensors leave correlated noise; generative models don't. Polarity inverted for H.264. |
| **DCT frequency** | Signal processing | Multi-band energy ratios, spectral smoothness, periodic artifacts |
| **Temporal** | Optical flow | Flow coefficient of variation, SSIM, HF flicker, face/background flow divergence |
| **CLIP ViT-L/14** | Semantic (ML) | Zero-shot real-vs-AI prompt ensembling |
| **EfficientNet-B4** | Artifact (ML) | Blending boundaries, texture anomalies (DeepfakeBench FF++ weights) |
| **Cross-channel correlation** | Signal processing | Channel independence in high-frequency DCT coefficients |

Nothing here is trained on deepfake data by this project. CLIP and EfficientNet use published pretrained weights; the rest is pure signal processing.

### Frame sampling

Not uniform sampling, and not seek-based. Two sequential decode passes:

1. Decode once, retaining only per-frame statistics (histogram, Laplacian variance, scene difference). Pixels discarded immediately.
2. Run scene-proportional budgeting, greedy farthest-point diversity selection, and transition boosting against those statistics.
3. Decode again, retaining pixels only for selected indices.

Memory is bounded by the frame budget rather than video length. This replaced a seek-based implementation — see [The seek bug](#the-seek-bug).

---

## Current status

| | |
|---|---|
| Tests | 134 passing |
| Lint | clean |
| Per-video inference | ~11s (Apple Silicon M3, 60 frames) |
| Benchmark dataset | in progress |
| Reportable AUC | **none yet** |

### What works

Noise residual is the only signal currently demonstrating reliable separation on the internal 10-video smoke set. The pipeline runs end-to-end, all stages are instrumented, and per-frame per-signal scores are persisted so ensemble tuning happens offline on cached data rather than requiring re-inference.

### Known issues

Documented rather than hidden, because they're the actual state of the system:

- **DCT is inverted.** Near-perfect inverse correlation on the internal set. It holds the highest ensemble weight and points the wrong way.
- **`smooth_score` returns a constant.** Sigmoid centered at 1.0 while its input ranges 6.6–10.4, so it saturates to zero for every video. Holds 35% of DCT's weight.
- **`band_score` is clamped** to 0.700 by a `low_ratio > 0.7` override on most videos.
- **Cross-channel correlation saturates** at 1.000 via a `* 1.5` clamp. Raw discrimination is near chance.
- **EfficientNet is uncalibrated** and may be net-harmful; it scored 0.937 on a real camera interview.
- **The ensemble currently scores below its best single component.**

### The seek bug

The frame selector originally used `cv2.VideoCapture.set(CAP_PROP_POS_FRAMES)` to seek to each candidate frame. Past GOP boundaries, OpenCV returns pixel data that does not correspond to the requested index — one test video diverged completely after frame 200.

This meant the histograms driving diversity selection were computed on wrong pixels, and the frames handed to the detectors were not the frames the selector believed it had chosen. Every calibration measurement taken before the fix is void.

If you are extracting frames by index anywhere in a video ML pipeline, verify your decoder returns what you asked for.

---

## Roadmap

- [x] Multi-signal ensemble (noise residual, DCT, temporal, CLIP, EfficientNet, ChCorr)
- [x] Two-pass scene-aware frame sampling
- [x] Face detection, alignment, cross-frame tracking
- [x] GradCAM heatmaps (ViT + CNN) and three-panel forensic views
- [x] LLM forensic explainer (Ollama local / Claude API)
- [x] End-to-end pipeline orchestrator with timing instrumentation
- [x] LangGraph agentic investigation loop
- [x] FastAPI service
- [x] Benchmark harness with resume support and crash-safe result streaming
- [x] Per-frame signal persistence + offline weight tuner
- [ ] Fix DCT polarity and dead components
- [ ] Signed weight search (polarity per signal)
- [ ] Modern-generator evaluation set (Veo / Kling / Runway, matched confounds)
- [ ] Calibration run → offline tuning → frozen holdout → **reportable AUC**
- [ ] Audio-visual consistency (SyncNet)
- [ ] Confidence calibration (Platt scaling)
- [ ] Docker packaging

---

## Evaluation plan

No results yet. The protocol, fixed in advance:

1. **Pilot (n=20)** — validate the harness. Plumbing, not results.
2. **Calibration run** — per-frame scores persisted to JSONL. One GPU pass.
3. **Offline tuning** — signed weight search, threshold sweep, drop-one ablation, frame-subsample study. CPU only, no re-inference.
4. **Holdout run** — weights and thresholds frozen. AUC reported from this run only.

### On dataset choice

Celeb-DF v2 is the obvious benchmark and is deliberately **not** the primary target. Its fakes are synthesized *from* the real videos, so a real/fake pair shares an identical background and differs only in the face region. Frame-level signals — the ones working here — have almost nothing to measure, and what remains discriminative is largely the double-encode artifact of the synthesis pipeline. That is a shortcut, not manipulation detection.

The primary evaluation set is being built for current text-to-video generators, with matched confounds: uniform duration, balanced face presence across classes, content-matched pairs, ≥3 generators, and an identical re-encode applied to every video so the detector cannot learn encoder provenance instead of generation.

---

## Architecture

```
src/verifi/
├── ingestion/        # Validation, metadata, URL download
├── sampling/         # Scene detection, two-pass frame selection, quality gate
├── preprocessing/    # MTCNN detection, eye-based alignment, IoU tracking
├── detectors/        # CLIP, EfficientNet, DCT frequency, noise residual, temporal
├── ensemble/         # score_from_signals() pure function, weighted aggregation
├── explainability/   # GradCAM, forensic view rendering
├── explanation/      # LLM explainer (Ollama / Claude), physics reasoner
├── agent/            # LangGraph investigation loop, forensic tool registry
├── benchmark/        # Dataset adapters, runner, metrics, per-frame persistence
├── pipeline/         # Orchestrator
└── api/              # FastAPI (POST /analyze, GET /report/{id})
```

The ensemble scoring logic lives in a single pure function, `score_from_signals()`, called by both the live pipeline and the offline tuner — so tuned weights reproduce in production rather than optimizing a drifting reimplementation.

---

## Quick start

### Prerequisites

- Python 3.11+
- macOS (Apple Silicon MPS) or Linux (CUDA)
- ffmpeg — `brew install ffmpeg`
- Ollama for local LLM — `brew install ollama && ollama pull llama3.1`

### Setup

```bash
git clone https://github.com/heisenberg1804/verifi.git
cd verifi

python3 -m venv .venv
source .venv/bin/activate

make install      # dependencies
make weights      # CLIP detection model (~900MB)
make validate     # verify MPS, models, imports
make test         # test suite
```

### Configuration

```bash
cp .env.example .env
```

```env
VERIFI_EXPLAINER__BACKEND=ollama
VERIFI_EXPLAINER__MODEL=llama3.1

# For production-quality reports:
# VERIFI_EXPLAINER__BACKEND=claude
# ANTHROPIC_API_KEY=sk-ant-...
```

### Run

```bash
make run                                    # start API server

curl -X POST http://localhost:8000/api/v1/analyze/upload \
  -F "video=@path/to/video.mp4"
```

Skip explainability for faster analysis — it disables the physics reasoner, GradCAM, and forensic view rendering.

---

## Model weights

| Model | Source | Size | Download |
|---|---|---|---|
| CLIP ViT-L/14 (LN-tuned) | [yermandy/deepfake-detection](https://huggingface.co/yermandy/deepfake-detection) | ~900 MB | `make weights` |
| EfficientNet-B4 | [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) | ~75 MB | manual |

See `scripts/download_weights.py` for manual instructions.

---

## Tech stack

**ML:** PyTorch, OpenCLIP, timm, facenet-pytorch, pytorch-grad-cam
**Video:** OpenCV, ffmpeg, yt-dlp
**API:** FastAPI, Pydantic
**Agent:** LangGraph
**LLM:** Ollama (local) / Anthropic Claude API
**Compute:** Apple Silicon MPS / NVIDIA CUDA

---

## Development

```bash
make test       # test suite
make test-cov   # with coverage
make lint       # ruff
make format     # ruff auto-format
make smoke      # detector smoke test
make bench      # benchmark harness
```

---

## License

MIT

---

## References

- Yermakov et al., *Unlocking the Hidden Potential of CLIP in Generalizable Deepfake Detection*, WACV 2026
- Yan et al., *DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection*, NeurIPS 2023
- Yan et al., *DF40: Toward Next-Generation Deepfake Detection*, NeurIPS 2024
- Guo et al., *Rethinking Vision-Language Model in Face Forensics*, CVPR 2025
- Li et al., *Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics*, CVPR 2020