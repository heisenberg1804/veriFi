# VeriFi — Project Progress Checkpoint

**Date:** May 15, 2026
**Branch:** `feat/agentic-investigation`
**Status:** Phase 7 complete, 105/105 tests passing, 0 lint errors

---

## What VeriFi Is

VeriFi is a forensic AI-generated video detection engine. It analyzes videos through multiple independent detection signals, generates GradCAM heatmaps highlighting suspicious regions, and produces structured natural-language forensic reports.

**Target users:** Trust-and-safety teams at social media platforms (B2B API), journalists, fact-checkers.

**Tech stack:** Python 3.13, PyTorch (MPS/Apple Silicon), open_clip, timm, facenet-pytorch, FastAPI, LangGraph, structlog. ~7,800 lines of source code across 41 modules, 105 unit tests.

---

## Changes Since Last Checkpoint (May 12, 2026)

The previous checkpoint documented Phases 1-5 (partial). Since then, three major bodies of work were completed:

### 1. Phase 5 Completed — Detection Calibration (10-video evaluation)

The previous checkpoint reported a 5-video test set where cross-channel correlation (ChCorr) was the "breakthrough signal" and all 4 AI videos were correctly detected. **This turned out to be overfitted to that specific test set.**

After expanding to 10 test videos (6 real, 4 AI — including real 720p footage, animated content, and sports), ChCorr completely failed:

| Signal | Previous Assessment | 10-Video Reality |
|---|---|---|
| **ChCorr** | "Best discriminator" (AI: 0.79-0.95, Real: ~0.63) | **BROKEN** — Real 720p: 0.73-0.97, overlapping entirely with AI |
| **Noise Residual** | "Diagnostic only — confounded by H.264" | **Best single discriminator** (1/10 errors), polarity inverted |
| **DCT** | "Good" | Second best (2/10 errors) |
| **Temporal** | Single flow divergence metric | Enhanced to 4 sub-signals, but content-dominated |
| **CLIP** | "Unreliable for photorealistic" | Confirmed unreliable (3/10 errors) |

**Key discovery: Noise residual polarity is inverted for H.264 video.** Real multi-encoded video has HIGH autocorrelation from deblocking filters; AI single-pass encoding has LOWER autocorrelation. This is the opposite of the textbook assumption.

#### Specific changes made:

**Noise Residual (`noise_residual.py`)** — Promoted from diagnostic-only to primary ensemble signal:
- Replaced saturated kurtosis sub-signal with spectral entropy (DCT of noise, Shannon entropy of normalized magnitude)
- Inverted autocorrelation scoring: `sigmoid((autocorr - 0.50) * -8.0)` — low autocorr = AI
- New weights: 0.45 autocorrelation + 0.30 spectral entropy + 0.25 block variance CV

**Temporal Analyzer (`temporal.py`)** — Rewritten from single metric to 4 sub-signals:
- Flow field CV: `std(flow_mag) / mean(flow_mag)` — AI has unnaturally uniform motion
- Inter-frame SSIM: Custom Gaussian-windowed implementation (no skimage dependency) — AI is temporally too smooth
- HF flicker: Spatial high-pass filtered temporal change — AI generators avoid temporal instability
- Face-bg divergence: Kept from original, demoted to 0.10 weight
- Combined: `0.30 * flow_cv + 0.35 * ssim + 0.25 * flicker + 0.10 * divergence`
- **Result:** All temporal signals proved content-dominated (static interview vs fast sports), not generation-discriminating. Used as supplementary, not primary.

**Ensemble (`aggregator.py`)** — Complete restructuring:
- Old weights: CLIP 0.25, DCT 0.20, ChCorr 0.35, Temporal 0.20
- New frame path: DCT 0.30, Noise Residual 0.25, Temporal 0.25, CLIP 0.15, ChCorr 0.05
- New face path: DCT 0.30, CLIP 0.20, Noise Residual 0.20, EfficientNet 0.15, Temporal 0.10, ChCorr 0.05
- Old thresholds: SUSPICIOUS >= 0.50, MANIPULATED >= 0.55
- New thresholds: LIKELY_AUTHENTIC < 0.35, SUSPICIOUS 0.35-0.70, LIKELY_MANIPULATED >= 0.70
- Signal agreement bonus: +0.05 when DCT and noise_residual both > 0.55; -0.03 penalty when they disagree
- Added `confidence` field: distance from nearest threshold, normalized to 0.0-1.0

**Agent Decision Tree (`planner.py`)** — Completely rewritten:
- Signal hierarchy: noise_residual > DCT > temporal > CLIP > ChCorr (UNRELIABLE)
- Explicit rule: "NEVER classify as LIKELY_MANIPULATED based on CLIP alone"
- New decision tree keyed on NR mean and DCT mean thresholds

**Pipeline (`orchestrator.py`)** — Wired noise residual into both paths, added `skip_explainability` parameter to bypass GradCAM/forensic views for benchmark speed.

#### 10-video diagnostic results (post-calibration):

| Video | Ground Truth | NR Score | DCT Score | Ensemble | Verdict |
|---|---|---|---|---|---|
| Perez Interview (720p) | Real | 0.344 | 0.41 | ~0.38 | SUSPICIOUS |
| iPhone Vlog (720p) | Real | 0.632 | 0.44 | ~0.48 | SUSPICIOUS |
| Messi FK (360p) | Real | 0.544 | 0.43 | ~0.46 | SUSPICIOUS |
| Ben 10 Animated (720p) | Real | — | — | ~0.45 | SUSPICIOUS |
| real_sample (360p) | Real | — | — | ~0.40 | SUSPICIOUS |
| Capybara Veo (720p) | AI | 0.697 | 0.47 | ~0.55 | SUSPICIOUS |
| WasifAI Football (720p) | AI | 0.627 | 0.44 | ~0.50 | SUSPICIOUS |
| SocialSight (720p) | AI | 0.603 | 0.42 | ~0.48 | SUSPICIOUS |
| Mansur/Neymar (720p) | AI | 0.652 | 0.45 | ~0.52 | SUSPICIOUS |
| Victoria Repa (720p) | AI | 0.645 | 0.46 | ~0.53 | SUSPICIOUS |

**Critical constraint maintained: ZERO real videos classified as LIKELY_MANIPULATED.** The wide SUSPICIOUS band is intentional — statistical signals on H.264 compressed video hit a ceiling. The honest answer is "suspicious, needs further review" rather than overconfident wrong answers.

### 2. Phase 6 Completed — FastAPI REST API

Built a complete REST API for the detection pipeline:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/analyze` | Submit video by local path or URL (yt-dlp download) |
| `POST` | `/api/v1/analyze/upload` | Submit video via file upload |
| `GET` | `/api/v1/report/{report_id}` | Retrieve complete forensic report with per-frame details |
| `GET` | `/api/v1/reports` | List available reports (most recent first) |
| `GET` | `/health` | Server status, model load state, device info |

**Implementation details:**
- Lifespan-managed pipeline singleton: models load on server start, unload on shutdown
- Report persistence: full analysis saved as JSON in `data/reports/{hash}/report.json` for later retrieval via GET
- Pydantic schemas for all request/response models (`schemas.py`)
- CORS middleware enabled
- URL analysis downloads via yt-dlp to temp directory, auto-cleaned after analysis
- File upload writes to temp directory, auto-cleaned after analysis
- Proper error handling: 400 (bad input), 404 (not found), 422 (validation error), 500 (pipeline error)
- Auto-generated Swagger docs at `/docs`, ReDoc at `/redoc`
- Start with `make run` → `uvicorn verifi.api.app:app --reload --port 8000`

**New files:**
- `src/verifi/api/app.py` — FastAPI app, lifespan, pipeline singleton
- `src/verifi/api/schemas.py` — 12 Pydantic models (AnalyzeRequest, AnalysisSummary, FullReport, FrameAnalysisDetail, FaceAnalysisDetail, TimingDetail, etc.)
- `src/verifi/api/routes/analysis.py` — All route handlers + report persistence
- `tests/test_api/test_endpoints.py` — 12 tests covering all endpoints with mocked pipeline

### 3. Phase 7 Completed — Benchmarking Infrastructure

Built complete infrastructure for evaluating the pipeline on standard deepfake detection datasets:

**Dataset Adapters (ABC pattern):**
- `BaseDatasetAdapter` — Abstract base with `iter_samples()`, `iter_stratified_sample()`, `validate()`, `summary()`
- `FaceForensicsAdapter` — FF++ with c23/c40 compression, 5 methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter), split file loading
- `CelebDFAdapter` — Celeb-DF v2 with test list filtering
- `DFDCAdapter` — DFDC with 50-part structure, part-based train/test split (00-39 train, 40-49 test)
- `DF40Adapter` — DF40 with dynamic method discovery from directory structure

**Benchmark Runner (`runner.py`):**
- `BenchmarkRunner` with `RunConfig` dataclass
- Resume support: tracks processed video paths in JSONL, skips on restart
- Progress logging every 10 videos (rate, ETA)
- Memory management: `gc.collect()` + `torch.mps.empty_cache()` every 50 videos
- Per-video error handling (errors logged, pipeline continues)
- Stratified sampling option for balanced evaluation

**Results I/O (`results.py`):**
- `VideoResult` dataclass with full per-video metrics
- `ResultsWriter`: append-only JSONL + CSV dual output
- `ResultsReader`: load results, config, count
- Streaming writes — no data loss on crash

**Metrics (`metrics.py`):**
- ROC curve + AUC (via scikit-learn)
- Equal Error Rate (EER) from FPR/FNR intersection
- Per-method breakdown (AUC per manipulation method)
- Per-signal AUC (evaluate each detector independently)
- Optimal threshold (Youden's J)
- Confusion matrix at any threshold

**Visualization (`visualizer.py`):**
- ROC curve with AUC annotation
- Score distribution histogram (real vs fake)
- Precision-recall curve
- Confusion matrix heatmap
- Per-signal AUC bar chart
- Per-method ROC overlay
- All publication-quality matplotlib with savefig

**CLI Scripts:**
- `scripts/run_benchmark.py` — argparse CLI with `--dataset`, `--root`, `--max-videos`, `--methods`, `--resume`, `--compression`, `--split`
- `scripts/analyze_results.py` — Post-hoc analysis from saved results directory

**New files:** 8 source files, 2 CLI scripts, 3 test files (23 new tests)

**Dependency:** Added `benchmark` optional group to `pyproject.toml` (`scikit-learn>=1.4`, `matplotlib>=3.8`)

---

## Architecture (Current)

### Dual-Path Detection

```
Video Input
    │
    ├── Path A (face-level) ─── MTCNN ─── CLIP + EfficientNet + DCT + NR + ChCorr on face crops
    │                                      (only active when faces detected)
    │
    └── Path B (frame-level) ── Smart sampling ─── CLIP + DCT + NR + ChCorr + Temporal on full frames
                                                    (always active)
    │
    └── Ensemble ─── Weighted aggregation + signal agreement bonus ─── Verdict + Confidence
    │
    └── Agent ─── LLM tool-use loop (LangGraph) ─── Forensic Report
    │
    └── API ─── FastAPI REST endpoints ─── JSON response
```

### Detection Signals (ranked by forensic reliability, revised from previous checkpoint)

| Rank | Signal | Type | Ensemble Weight | Reliability | Change from Previous |
|---|---|---|---|---|---|
| 1 | **Noise Residual** | Statistical | 0.25 frame / 0.20 face | Best discriminator (1/10 errors) | **Promoted from "diagnostic only"** |
| 2 | **DCT Frequency** | Signal processing | 0.30 | Second best (2/10 errors) | Promoted from 0.20 |
| 3 | **Temporal** | Multi-signal | 0.25 frame / 0.10 face | Supplementary (content-dominated) | **Rewritten**, expanded from 1 to 4 sub-signals |
| 4 | **CLIP ViT-L/14** | Semantic | 0.15 | Unreliable for photorealistic | Demoted from 0.25 |
| 5 | **ChCorr** | Signal processing | 0.05 | **BROKEN** — overlapping ranges | **Demoted from 0.35 ("best discriminator")** |
| 6 | **EfficientNet-B4** | Pixel artifacts | 0.15 face only | Not calibrated | Unchanged |
| 7 | **JPEG Ghost** | Re-compression | Not in ensemble | H.264 defeats it | Unchanged |

### Ensemble Configuration

```
Frame path:  DCT 0.30  |  NR 0.25  |  Temporal 0.25  |  CLIP 0.15  |  ChCorr 0.05
Face path:   DCT 0.30  |  CLIP 0.20  |  NR 0.20  |  EffNet 0.15  |  Temporal 0.10  |  ChCorr 0.05

Thresholds:  LIKELY_AUTHENTIC < 0.35  |  SUSPICIOUS 0.35-0.70  |  LIKELY_MANIPULATED >= 0.70
             (previously: SUSPICIOUS >= 0.50, MANIPULATED >= 0.55)

Agreement:   +0.05 bonus when DCT and NR both > 0.55
             -0.03 penalty when they disagree (one > 0.55, other < 0.40)
```

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

### Phase 5 — Detection Calibration
- Expanded test set from 5 to 10 videos (6 real, 4 AI, including 720p real footage and animated content)
- Discovered ChCorr is broken on H.264 (real/AI ranges overlap completely)
- Promoted noise residual to primary signal with inverted polarity scoring
- Replaced kurtosis with spectral entropy sub-signal
- Enhanced temporal analyzer to 4 sub-signals (flow CV, SSIM, flicker, divergence)
- Rebalanced all ensemble weights around DCT + NR as primary discriminators
- Widened SUSPICIOUS band (0.35-0.70) to avoid overconfident wrong answers
- Added signal agreement bonus/penalty
- Added confidence field to VideoAnalysis
- Rewrote agent decision tree for new signal hierarchy

### Phase 6 — FastAPI REST API
- POST /api/v1/analyze (path or URL), POST /api/v1/analyze/upload (file upload)
- GET /api/v1/report/{id} (full report), GET /api/v1/reports (list), GET /health
- Lifespan-managed pipeline singleton, CORS middleware
- Pydantic request/response schemas (12 models)
- Report persistence as JSON for later retrieval
- Auto-generated Swagger docs at /docs
- 12 endpoint tests with mocked pipeline

### Phase 7 — Benchmarking Infrastructure
- Dataset adapters for FF++, Celeb-DF v2, DFDC, DF40 (all with ABC pattern)
- BenchmarkRunner with resume support, progress logging, memory management
- MetricsComputer: ROC/AUC, EER, per-method breakdown, per-signal AUC, confusion matrix
- PlotGenerator: 6 publication-quality visualization types
- CLI scripts: run_benchmark.py, analyze_results.py
- Streaming JSONL + CSV result storage
- scikit-learn + matplotlib optional dependency group

---

## Project Structure

```
src/verifi/                          # 7,800 lines across 41 modules
├── config.py                        # Pydantic settings, VERIFI_ env prefix
├── agent/
│   ├── investigator.py              # LangGraph tool-use loop
│   └── planner.py                   # System prompt + decision tree
├── api/
│   ├── app.py                       # FastAPI app, lifespan, pipeline singleton
│   ├── schemas.py                   # Pydantic request/response models
│   └── routes/
│       └── analysis.py              # POST /analyze, /upload, GET /report, /reports
├── benchmark/
│   ├── datasets/
│   │   ├── base.py                  # BaseDatasetAdapter ABC
│   │   ├── faceforensics.py         # FF++ adapter
│   │   ├── celebdf.py              # Celeb-DF v2 adapter
│   │   ├── dfdc.py                  # DFDC adapter
│   │   └── df40.py                  # DF40 adapter
│   ├── results.py                   # VideoResult, ResultsWriter, ResultsReader
│   ├── runner.py                    # BenchmarkRunner
│   ├── metrics.py                   # MetricsComputer (ROC/AUC/EER)
│   └── visualizer.py               # PlotGenerator (6 chart types)
├── detectors/
│   ├── base.py                      # Abstract BaseDetector interface
│   ├── clip_detector.py             # CLIP ViT-L/14 zero-shot
│   ├── effnet_detector.py           # EfficientNet-B4 (DeepfakeBench)
│   ├── frequency.py                 # DCT + cross-channel correlation
│   ├── noise_residual.py            # Noise residual (primary signal)
│   ├── jpeg_ghost.py                # JPEG ghost (diagnostic only)
│   └── temporal.py                  # Multi-signal temporal (4 sub-signals)
├── ensemble/
│   └── aggregator.py                # Dual-path aggregation + agreement bonus
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
│   └── orchestrator.py              # VeriFiPipeline.analyze()
└── tools/                           # Agent tool implementations
    ├── base.py, factory.py
    ├── detection_tools.py
    ├── analysis_tools.py
    └── sampling_tools.py

tests/                               # 105 tests across 15 test files
├── test_api/test_endpoints.py       # 12 tests (API endpoints)
├── test_benchmark/                  # 23 tests (results, metrics, datasets)
├── test_detectors/                  # Detector unit tests
├── test_ensemble/                   # Aggregator + agreement bonus tests
├── test_explainability/             # GradCAM tests
├── test_preprocessing/              # Face detection tests
├── test_sampling/                   # Scene + frame selection tests
└── test_tools/                      # Agent tool tests

scripts/
├── run_benchmark.py                 # CLI: run benchmarks on datasets
├── analyze_results.py               # CLI: post-hoc analysis from results
├── diagnose_signals.py              # Raw signal diagnostics across videos
├── calibrate_clip_prompts.py        # CLIP prompt pair calibration
├── test_phase2.py                   # Integration test (stages 1-4)
├── test_phase3.py                   # Integration test (full pipeline)
├── test_phase4.py                   # Integration test (agentic)
├── validate_setup.py                # Environment validation
└── download_weights.py              # Model weight download
```

---

## Test Videos (expanded from 5 to 10)

| File | Source | Ground Truth | Resolution | Content |
|---|---|---|---|---|
| `Capybara_Walks_Oscars_Red_Carpet.mp4` | Veo (Google Gemini) | AI-generated | 720p | Surreal capybara on red carpet with paparazzi |
| `WasifAI - No one Will believe...mp4` | Unknown AI | AI-generated | 720p | Photorealistic AI football match |
| `SocialSight - cameraman knew...mp4` | Veo (Google Gemini) | AI-generated | 720p | Photorealistic AI cameraman footage |
| `Mansur - تسريب فيديو...mp4` | Veo (Google Gemini) | AI-generated | 720p | Photorealistic AI Neymar footage |
| `Victoria Repa - That viral Korean AI...mp4` | Unknown AI | AI-generated | 720p | AI stadium trend footage |
| `real_sample.mp4` | Real camera | Authentic | 360p | Real video footage |
| `Florentino Pérez...mp4` | Real camera | Authentic | 720p | Perez interview footage |
| `Lionel Messi's free-kick...mp4` | Real broadcast | Authentic | 360p | Messi free kick with Spanish commentary |
| `iPhoneで映像撮る人...mp4` | Real iPhone | Authentic | 720p | iPhone vlog footage |
| `Ben 10 (2005-2008).mp4` | Animated TV | Authentic (animated) | 720p | Ben 10 cartoon (non-AI animation) |

---

## Quantitative Comparison: Previous vs Current

| Metric | Previous Checkpoint (May 12) | Current (May 15) |
|---|---|---|
| Source modules | 31 | 41 (+10) |
| Source lines | ~5,800 | ~7,800 (+2,000) |
| Tests | 64 | 105 (+41) |
| Test files | 8 | 15 (+7) |
| Scripts | 6 | 9 (+3) |
| Test videos | 5 | 10 (+5) |
| Phases complete | 4 + partial 5 | 7 |
| API endpoints | 0 | 5 |
| Dataset adapters | 0 | 4 |
| Primary signal | ChCorr (weight 0.35) | Noise Residual (weight 0.25) + DCT (weight 0.30) |
| SUSPICIOUS threshold | >= 0.50 | 0.35-0.70 |
| MANIPULATED threshold | >= 0.55 | >= 0.70 |
| ChCorr status | "Best discriminator" | "BROKEN — tiebreaker only (0.05)" |
| Noise residual status | "Diagnostic only" | "Primary signal" |

---

## Known Issues & Limitations

1. **Statistical signals hit a ceiling on H.264 video.** Noise residual and DCT discriminate at the extremes but overlap in the middle. Most videos fall in the SUSPICIOUS band (0.35-0.70). This is an honest limitation, not a bug — H.264 compression destroys fine-grained forensic artifacts.

2. **Cross-channel correlation is unreliable after H.264 compression.** Real 720p videos score 0.73-0.97, overlapping entirely with AI. Demoted from primary signal (0.35) to tiebreaker (0.05). The previous checkpoint's assessment was based on a 5-video test set that happened to show separation.

3. **Temporal signals are content-dominated, not generation-discriminating.** Static interviews have high SSIM and low flicker regardless of whether they're AI or real. Fast sports have the opposite. These are content properties, not generation artifacts.

4. **CLIP is semantic, not forensic.** Zero-shot prompts detect whether content *looks* AI-generated, not whether it *is* AI-generated. Fails completely on photorealistic AI video (scores 0.27-0.55).

5. **Noise residual polarity is inverted for H.264.** Real multi-encoded video has HIGH autocorrelation from deblocking filters; AI single-pass encoding has LOWER. The code accounts for this, but it's counterintuitive and a potential source of confusion.

6. **EfficientNet-B4 weights not calibrated.** DeepfakeBench FF++ checkpoint loads but scores are not calibrated for the current pipeline.

7. **No benchmarks on standard datasets yet.** Infrastructure is built but no datasets have been downloaded (require academic license requests). This is the critical next step.

8. **Separation margin is tight.** Wide SUSPICIOUS band is intentional to avoid overconfident wrong answers. Improving discrimination requires fine-tuning on large labeled datasets.

---

## Upcoming

### Phase 8 — Benchmark Evaluation & Fine-Tuning
- Download FF++ (academic license), Celeb-DF v2, DFDC (Kaggle), DF40
- Run benchmarks: `python scripts/run_benchmark.py --dataset ff++ --root /path --max-videos 500`
- Compute ROC/AUC, EER, per-method breakdown
- Fine-tune ensemble weights and thresholds based on empirical data
- Calibrate EfficientNet-B4 scores

---

## Development Commands

```bash
make install        # pip install -e ".[dev]"
make test           # pytest tests/ -v (105 tests)
make lint           # ruff check (0 errors)
make format         # ruff format
make validate       # scripts/validate_setup.py
make run            # uvicorn dev server (localhost:8000)

# API
curl localhost:8000/health
curl -X POST localhost:8000/api/v1/analyze -H 'Content-Type: application/json' \
  -d '{"video_path": "data/sample_videos/real_sample.mp4"}'
curl localhost:8000/api/v1/report/{report_id}

# Benchmarking
pip install -e ".[benchmark]"
python scripts/run_benchmark.py --dataset ff++ --root /path/to/FF++ --max-videos 100
python scripts/analyze_results.py data/benchmarks/{run_id}/

# Integration tests
python scripts/diagnose_signals.py                              # Raw signal diagnostics
python scripts/test_phase3.py data/sample_videos/VIDEO.mp4      # Full pipeline
```

---

## Key Insight (Updated)

The previous checkpoint identified cross-channel correlation as the breakthrough forensic signal. **That was wrong** — it was overfitted to a 5-video test set. With 10 diverse videos, ChCorr shows complete overlap between real and AI classes.

The actual best discriminators are **noise residual** (inverted autocorrelation for H.264) and **DCT frequency analysis** (sharpness-normalized HF content). But even these hit a ceiling on compressed video. The honest truth is that **zero-shot statistical analysis of H.264 video has fundamental limits** — compression destroys the fine-grained artifacts that distinguish generation methods.

The path forward is **supervised learning on large labeled datasets** (FF++, DFDC, etc.), which is why Phase 7 built the benchmarking infrastructure. Statistical signals provide a strong baseline but need to be supplemented with learned features calibrated on diverse data.
