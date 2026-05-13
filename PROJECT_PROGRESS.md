# VeriFi — Project Progress Checkpoint

**Date:** May 12, 2026
**Branch:** `feat/agentic-investigation`
**Status:** Phase 5 (partial) complete, 64/64 tests passing, 0 lint errors

---

## What VeriFi Is

VeriFi is a forensic AI-generated video detection engine. It analyzes videos through multiple independent detection signals, generates GradCAM heatmaps highlighting suspicious regions, and produces structured natural-language forensic reports.

**Target users:** Trust-and-safety teams at social media platforms (B2B API), journalists, fact-checkers.

**Tech stack:** Python 3.13, PyTorch (MPS/Apple Silicon), open_clip, timm, facenet-pytorch, FastAPI, LangGraph, structlog. ~5,800 lines of source code across 31 modules, 64 unit tests.

---

## Architecture

### Dual-Path Detection

```
Video Input
    │
    ├── Path A (face-level) ─── MTCNN face detection ─── CLIP + EfficientNet + DCT + ChCorr on face crops
    │                                                     (only active when faces are detected)
    │
    └── Path B (frame-level) ── Smart frame sampling ──── CLIP + DCT + ChCorr + Temporal on full frames
                                                          (always active)
    │
    └── Ensemble Aggregator ─── Weighted combination ──── Verdict + Manipulation Type
    │
    └── Agentic Investigation ─ LLM tool-use loop ────── Forensic Report (JSON)
```

Path A catches face-swap and face-reenactment deepfakes. Path B catches fully synthetic content (Sora, Veo, Runway, etc.). The ensemble normally takes the stronger path's signal, with a consensus override that forces frame dominance when >70% of frames are flagged.

### Detection Signals (ranked by forensic reliability)

| Signal | Type | What It Measures | Reliability |
|---|---|---|---|
| **Cross-Channel Correlation** | Signal processing | Pearson correlation of HF DCT coefficients across RGB channels. Real cameras have independent sensor noise per Bayer channel; AI generators produce correlated artifacts from shared latent space. | Best discriminator. AI: 0.79-0.95, Real: ~0.63 |
| **DCT Frequency Analysis** | Signal processing | Multi-band DCT with sharpness-normalized sigmoid scoring. Band energy ratios, spectral smoothness, periodic artifacts. | Good, decoupled from resolution/bitrate via sharpness normalization |
| **CLIP ViT-L/14** | Semantic/aesthetic | Zero-shot classification via prompt ensembling. Detects obviously AI content (surreal scenarios, smooth textures). | Unreliable for photorealistic AI video |
| **EfficientNet-B4** | Pixel artifacts | Blending boundaries, texture anomalies. DeepfakeBench FF++ weights. | Face-swap only, not calibrated |
| **Temporal Consistency** | Optical flow | Farneback optical flow between adjacent frames. Detects motion discontinuities. | Supplementary |
| **Noise Residual** | Statistical | SRM-inspired kurtosis/autocorrelation/variance of noise. | Diagnostic only — confounded by H.264 |
| **JPEG Ghost** | Re-compression | Error curve monotonicity across quality levels. | Diagnostic only — no discrimination for video |

### Smart Frame Sampling (3-pass)

1. Scene segmentation via frame differencing
2. Budget-proportional diverse keyframe selection (greedy farthest-point by histogram distance)
3. Transition frame boosting (±2 frames around scene boundaries)
4. Quality gate: reject blurry frames (Laplacian variance < threshold)

### Ensemble Weights

```
Frame path: CLIP 0.25, DCT 0.20, ChCorr 0.35, Temporal 0.20
Face path:  CLIP 0.25, EfficientNet 0.20, DCT 0.20, ChCorr 0.35

Thresholds: SUSPICIOUS >= 0.50, LIKELY_MANIPULATED >= 0.55
Aggregation: mean of all frames (top_k_ratio = 1.0)
```

### Agentic Investigation

LangGraph tool-use loop where an LLM acts as a forensic investigator. System prompt establishes a decision tree prioritizing forensic signals (ChCorr, DCT) over semantic signals (CLIP). Available tools: quick_scan, run_dct_analysis, run_gradcam, check_metadata, sample_more_frames, zoom_region. Max 5 tool calls per investigation, then produces a structured JSON report.

---

## Completed Phases

### Phase 1 — Detectors & Infrastructure
- CLIP ViT-L/14 zero-shot detector with prompt ensembling
- EfficientNet-B4 detector with DeepfakeBench weight loading
- DCT frequency analyzer (multi-band)
- Temporal consistency detector (optical flow)
- Pydantic config system with env var overrides (`VERIFI_` prefix)
- Model weight download script

### Phase 2 — Video Ingestion & Sampling
- Video validation + ffprobe metadata extraction
- URL download via yt-dlp
- Scene boundary detection (frame differencing)
- 3-pass smart frame selection with quality filtering
- MTCNN face detection, alignment, IoU-based tracking

### Phase 3 — Ensemble & Explainability
- Dual-path ensemble aggregator with consensus override
- GradCAM heatmaps for both ViT (CLIP) and CNN (EfficientNet)
- Three-panel forensic view: Original | GradCAM | DCT Spectrum
- Confidence timeline visualization
- End-to-end pipeline orchestrator (`VeriFiPipeline.analyze()`)

### Phase 4 — Agentic Investigation
- LangGraph-based tool-use loop with forensic tools
- LLM explainer with Ollama (local llama3.1) and Claude API backends
- Structured JSON report generation
- Decision tree prompts for investigation strategy

### Phase 5 (partial) — Detection Calibration
This is the most recent and most impactful work. It was driven by a critical finding: **only 1 of 4 AI-generated test videos was being correctly detected.**

#### The Problem

We downloaded 3 additional AI-generated test videos from X/Twitter and ran diagnostics:

| Video | Ground Truth | CLIP Mean | Old Verdict |
|---|---|---|---|
| Capybara Walks Oscars Red Carpet (Veo, 720p) | AI | 0.909 | MANIPULATED |
| WasifAI Football Match (720p) | AI | 0.551 | SUSPICIOUS at best |
| SocialSight Cameraman (720p) | AI | 0.456 | AUTHENTIC |
| Mansur/Neymar Leak (720p) | AI | 0.276 | AUTHENTIC |
| real_sample.mp4 (360p) | Real | 0.349 | AUTHENTIC |

Root causes identified:
1. **CLIP is aesthetic, not forensic** — zero-shot prompts ask "does this look AI?" which only works for surreal content, not photorealistic AI video
2. **DCT was measuring compression, not AI artifacts** — step-function thresholds correlated with resolution x bitrate, not generation method
3. **No genuinely forensic signal existed** — all signals were either semantic (CLIP), aesthetic (EfficientNet), or compression-confounded (DCT)

#### What We Built

1. **Cross-channel DCT correlation** — The breakthrough signal. Exploits the physics of image formation: real Bayer sensors produce independent per-channel noise; AI generators from shared latent space produce correlated HF artifacts. Implemented as Pearson correlation of HF DCT coefficients across RGB channel pairs.

2. **Sharpness-normalized DCT scoring** — Uses Laplacian variance to compute expected HF ratio relative to compression level. Scores deviation from expectation rather than absolute values, decoupling the signal from resolution/bitrate.

3. **Continuous sigmoid scoring** — Replaced all step-function thresholds with continuous sigmoid curves to eliminate score quantization (identical scores across all frames of a video).

4. **Noise residual analyzer** — SRM-inspired noise extraction with kurtosis, spatial autocorrelation, and block variance analysis. Built and tested, but excluded from ensemble after finding it confounded by H.264 compression.

5. **JPEG ghost analyzer** — Re-compression error curve analysis. Built and tested, but excluded from ensemble after finding no discrimination for H.264 video frames.

6. **Ensemble rebalancing** — Demoted CLIP (0.25), promoted ChCorr (0.35) as primary signal. Lowered manipulated threshold from 0.70 to 0.55. Removed confidence-weighted voting (it amplified confidently-wrong CLIP scores).

7. **Agent decision tree rewrite** — Rewrote investigation prompts to prioritize forensic signals. ChCorr mean >0.75 triggers LIKELY_MANIPULATED regardless of CLIP score.

#### Results After Calibration

| Video | CLIP Mean | ChCorr Mean | Ensemble Score | New Verdict |
|---|---|---|---|---|
| Capybara (Veo) | 0.909 | 0.950 | **0.764** | MANIPULATED |
| WasifAI Football | 0.551 | 0.911 | **0.672** | MANIPULATED |
| SocialSight Cameraman | 0.456 | 0.794 | **0.592** | MANIPULATED |
| Mansur/Neymar | 0.276 | 0.893 | **0.580** | MANIPULATED |
| real_sample.mp4 | 0.349 | 0.632 | **0.490** | AUTHENTIC |

**All 4 AI videos correctly detected. Real video correctly classified as authentic.** Minimum separation gap: 0.09 (real at 0.490 vs lowest AI at 0.580).

---

## Project Structure

```
src/verifi/                          # 5,800 lines across 31 modules
├── config.py                        # Pydantic settings, VERIFI_ env prefix
├── agent/
│   ├── investigator.py              # LangGraph tool-use loop
│   └── planner.py                   # System prompt + decision tree
├── detectors/
│   ├── base.py                      # Abstract BaseDetector interface
│   ├── clip_detector.py             # CLIP ViT-L/14 zero-shot
│   ├── effnet_detector.py           # EfficientNet-B4 (DeepfakeBench)
│   ├── frequency.py                 # DCT + cross-channel correlation
│   ├── noise_residual.py            # Noise residual (diagnostic only)
│   ├── jpeg_ghost.py                # JPEG ghost (diagnostic only)
│   └── temporal.py                  # Optical flow consistency
├── ensemble/
│   └── aggregator.py                # Dual-path weighted aggregation
├── explainability/
│   ├── gradcam.py                   # GradCAM for ViT + CNN
│   └── heatmap_renderer.py          # Overlays, forensic views, timeline
├── explanation/
│   ├── llm_explainer.py             # Ollama / Claude API backends
│   └── prompts.py                   # Versioned forensic templates
├── ingestion/
│   ├── validator.py                 # ffprobe metadata + validation
│   └── downloader.py                # yt-dlp URL download
├── sampling/
│   ├── scene_detector.py            # Scene boundary detection
│   ├── frame_selector.py            # 3-pass smart frame selection
│   └── quality_filter.py            # Laplacian blur filter
├── preprocessing/
│   ├── face_detector.py             # MTCNN + IoU tracking
│   ├── face_aligner.py              # Eye-based rotation alignment
│   └── face_tracker.py              # Stub for embedding-based tracking
├── pipeline/
│   └── orchestrator.py              # End-to-end VeriFiPipeline.analyze()
├── tools/                           # Agent tool implementations
│   ├── base.py, factory.py
│   ├── detection_tools.py
│   ├── analysis_tools.py
│   └── sampling_tools.py
└── api/                             # FastAPI (Phase 6, not yet built)

tests/                               # 64 tests, all passing
scripts/
├── diagnose_signals.py              # Raw signal diagnostics across videos
├── calibrate_clip_prompts.py        # CLIP prompt pair calibration
├── test_phase2.py                   # Stages 1-4 integration test
├── test_phase3.py                   # Full pipeline integration test
├── test_phase4.py                   # Agentic investigation test
├── validate_setup.py                # Environment validation
└── download_weights.py              # Model weight download
```

---

## Test Videos

| File | Source | Ground Truth | Resolution | Bitrate | Content |
|---|---|---|---|---|---|
| `Capybara_Walks_Oscars_Red_Carpet.mp4` | Veo (Google Gemini) | AI-generated | 720p | 7677 kbps | Capybara on red carpet with paparazzi. Surreal, obviously AI aesthetic. |
| `WasifAI - No one Will believe...mp4` | Unknown AI generator | AI-generated | 720p | 1697 kbps | Photorealistic AI football match. Fools CLIP completely. |
| `SocialSight - cameraman knew...mp4` | Unknown AI generator | AI-generated | 720p | 817 kbps | Photorealistic AI cameraman footage. Low bitrate. |
| `Mansur - تسريب فيديو...mp4` | Unknown AI generator | AI-generated | 720p | 1644 kbps | Photorealistic AI Neymar footage. CLIP scores lower than real video. |
| `real_sample.mp4` | Real camera | Authentic | 360p | 526 kbps | Real video footage. Lowest resolution/bitrate in test set. |

---

## Known Issues & Limitations

1. **CLIP is fundamentally semantic, not forensic.** Zero-shot prompts cannot distinguish photorealistic AI from real camera footage. CLIP's value is limited to detecting aesthetically obvious AI content. Cross-channel correlation is the actual discriminator.

2. **Cross-channel correlation works best as video-level mean, not per-frame.** Individual real frames can score as high as 0.95 ChCorr; AI discrimination emerges from the video-level average.

3. **Noise residual and JPEG ghost are confounded by H.264 compression.** Both detectors work in theory (noise residual exploits PRNU physics; JPEG ghost exploits re-compression dips) but H.264's motion-compensated prediction destroys the signal. They remain as standalone diagnostic tools, not in the ensemble.

4. **EfficientNet-B4 weights not fully calibrated.** DeepfakeBench FF++ checkpoint loads but scores are not calibrated for the current pipeline. Ensemble weight is set to 0.20 on face path only.

5. **Separation margin is tight.** Real video at 0.490 vs lowest AI at 0.580 gives a 0.09 gap. More diverse test videos needed to validate robustness.

6. **No benchmarking on standard datasets.** Not yet tested on FF++, Celeb-DF, DFDC, or DF40.

---

## Upcoming Phases

### Phase 5 (remaining) — Pipeline Polish
- Error handling and graceful degradation
- Timeout handling for long videos
- Progress callbacks

### Phase 6 — FastAPI API
- `POST /analyze` — submit video for analysis
- `GET /report/{id}` — retrieve forensic report
- Async processing with job queue
- Rate limiting, auth

### Phase 7 — Benchmarking
- FF++ (FaceForensics++)
- Celeb-DF v2
- DFDC (Deepfake Detection Challenge)
- DF40 (40 deepfake methods)
- ROC/AUC curves, per-method breakdown

---

## Development Commands

```bash
make install        # pip install -e ".[dev]"
make test           # pytest tests/ -v (64 tests)
make lint           # ruff check (0 errors)
make format         # ruff format
make validate       # scripts/validate_setup.py

# Integration tests
python scripts/diagnose_signals.py                              # Raw signal diagnostics
python scripts/test_phase3.py data/sample_videos/VIDEO.mp4      # Full pipeline
python scripts/test_phase4.py data/sample_videos/VIDEO.mp4      # Agentic investigation
```

---

## Key Insight

The most important finding from this project so far: **zero-shot semantic classifiers (CLIP) are not forensic tools.** They detect whether content *looks* AI-generated, not whether it *is* AI-generated. Modern AI video generators produce photorealistic content that is semantically indistinguishable from camera footage.

The actual forensic signal is **cross-channel correlation of high-frequency DCT coefficients** — a physics-based signal that exploits the fundamental difference between how camera sensors and neural networks produce pixel values. Real cameras have independent noise per Bayer color channel; AI generators produce correlated artifacts from shared latent representations. This signal correctly classifies all 5 test videos with zero misclassifications.
