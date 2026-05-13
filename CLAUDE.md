# CLAUDE.md — VeriFi Project Context

## What this project is

VeriFi is a forensic AI-generated video detection engine. It analyzes videos through multiple independent detection signals, generates visual heatmaps highlighting suspicious regions, and produces structured natural language forensic reports explaining findings.

Target users: trust-and-safety teams at social media platforms (B2B API), journalists, fact-checkers.

## Architecture

### Dual-path detection strategy

The pipeline runs two independent analysis paths in parallel:

**Path A (face-level):** For face-swap and face-reenactment deepfakes. Detects faces via MTCNN, crops them, runs CLIP ViT-L/14 + EfficientNet-B4 + DCT frequency analysis on face crops. Only active when faces are detected.

**Path B (frame-level):** For fully synthetic content (Sora, Veo, Runway, MidJourney). Runs CLIP + DCT on entire frames regardless of face detection. Always active. This was added after testing on a Veo-generated capybara video revealed that face-only detection completely misses fully AI-generated content.

The ensemble normally takes the stronger path's signal as the dominant verdict. However, a frame-path consensus override forces frame dominance when >70% of frames are flagged and frame_score exceeds the suspicious threshold — this prevents noisy face-path scores from small background faces from overriding strong frame-level synthetic detection.

### Detection signals (ranked by forensic reliability)

1. **Noise Residual** (statistical, no ML) — Best single discriminator (1/10 errors on test set). Extracts noise via median-filter subtraction, analyzes spatial autocorrelation, spectral entropy, and block variance. POLARITY IS INVERTED for H.264: real multi-encoded video has HIGH autocorrelation from deblocking filters; AI single-pass encoding has LOWER autocorrelation. Ensemble weight: 0.25 (frame), 0.20 (face).
2. **DCT Frequency** (signal processing, no ML) — Second best discriminator. Multi-band DCT analysis with sharpness-normalized scoring (sigmoid-based). Band energy ratios, spectral smoothness, periodic artifacts. AI videos show suppressed HF content relative to sharpness. Ensemble weight: 0.30.
3. **Temporal Consistency** (multi-signal) — Flow field CV (AI has unnaturally uniform motion), inter-frame SSIM (AI is temporally too smooth), HF flicker (AI generators avoid temporal instability), face-bg divergence (face-swap detection). Combined: 0.30*flow_cv + 0.35*ssim + 0.25*flicker + 0.10*divergence. Ensemble weight: 0.25.
4. **CLIP ViT-L/14** (semantic) — Zero-shot classification via prompt ensembling. Detects aesthetically obvious AI content but FAILS on photorealistic AI video (scores 0.27-0.55 for modern generators). Ensemble weight: 0.15 (demoted).
5. **Cross-Channel Correlation** (signal processing, no ML) — UNRELIABLE after H.264 compression. Real 720p videos score 0.73-0.97, overlapping entirely with AI. Kept as weak tiebreaker only. Ensemble weight: 0.05.
6. **EfficientNet-B4** (artifact) — Face-swap only. DeepfakeBench FF++ weights, not calibrated. Ensemble weight: 0.15 (face path only).
7. **JPEG Ghost** (diagnostic only) — Re-compression error curve analysis. Not in ensemble — H.264 defeats the JPEG-specific dip pattern.

### Smart frame sampling (3-pass)

1. Scene segmentation via frame differencing (cv2.absdiff)
2. Budget-proportional diverse key frame selection (greedy farthest-point by histogram distance)
3. Transition frame boosting (±2 frames around scene boundaries)
4. Quality gate: reject blurry frames (Laplacian variance < threshold)

### Forensic explainability

- GradCAM heatmaps from CLIP (ViT, needs reshape_transform) and EfficientNet (standard CNN)
- Three-panel forensic view: Original | GradCAM Heatmap | DCT Spectrum
- Confidence timeline visualization
- LLM-generated explanation via Ollama (local, llama3.1) or Claude API (production)

## Project structure

```
src/verifi/
├── config.py              # Pydantic settings, env var overrides (VERIFI_ prefix)
├── ingestion/
│   ├── validator.py       # Video validation + ffprobe metadata
│   └── downloader.py      # yt-dlp URL download
├── sampling/
│   ├── scene_detector.py  # Scene boundary detection
│   ├── frame_selector.py  # 3-pass smart frame selection
│   └── quality_filter.py  # Laplacian blur filter
├── preprocessing/
│   ├── face_detector.py   # MTCNN detection, alignment, IoU tracking
│   ├── face_aligner.py    # Eye-based rotation alignment
│   └── face_tracker.py    # Stub for future embedding-based tracking
├── detectors/
│   ├── base.py            # Abstract BaseDetector interface
│   ├── clip_detector.py   # CLIP ViT-L/14 zero-shot (open_clip, prompt ensembling)
│   ├── effnet_detector.py # EfficientNet-B4 via timm
│   ├── frequency.py       # DCT frequency + cross-channel correlation (no ML)
│   ├── noise_residual.py  # Noise residual analysis (SRM-inspired, no ML)
│   ├── jpeg_ghost.py      # JPEG ghost re-compression analysis (no ML)
│   └── temporal.py        # Optical flow consistency
├── ensemble/
│   └── aggregator.py      # Dual-path weighted aggregation + consensus override + verdict
├── explainability/
│   ├── gradcam.py         # GradCAM for ViT + CNN architectures
│   └── heatmap_renderer.py # Overlays, forensic views, timeline
├── explanation/
│   ├── prompts.py         # Versioned forensic prompt templates
│   └── llm_explainer.py   # Ollama (local) / Claude API (production)
├── pipeline/
│   └── orchestrator.py    # End-to-end VeriFiPipeline.analyze()
└── api/                   # FastAPI server (Phase 6, not yet built)
```

## Current state

**Completed:**
- Phase 1: All 4 detectors working, config system, weight download, validation script
- Phase 2: Video ingestion, scene detection, smart frame sampling, face detection + tracking
- Phase 3: Dual-path ensemble, GradCAM, forensic views, pipeline orchestrator, zero-shot CLIP, multi-band DCT, frame consensus override, small-face EfficientNet skip
- Phase 4: Agentic investigation with tool-use loop, LLM explainer, decision tree prompts
- Phase 5: Detection calibration — noise residual promoted to primary signal (inverted polarity for H.264), temporal analyzer enhanced (flow CV, SSIM, flicker), ensemble rebalanced (DCT 0.30, NR 0.25, temporal 0.25, CLIP 0.15, ChCorr 0.05), conservative thresholds (0.35/0.70), signal agreement bonus, confidence field, agent decision tree rewritten

**Ensemble weights:**
- Frame path: DCT 0.30, noise_residual 0.25, temporal 0.25, CLIP 0.15, channel_corr 0.05
- Face path: DCT 0.30, CLIP 0.20, noise_residual 0.20, EfficientNet 0.15, temporal 0.10, channel_corr 0.05
- Thresholds: LIKELY_AUTHENTIC < 0.35, SUSPICIOUS 0.35-0.70, LIKELY_MANIPULATED >= 0.70
- Signal agreement bonus: +0.05 when DCT and noise_residual both > 0.55; -0.03 penalty when they disagree

**Known issues:**
- CLIP is semantic/aesthetic, not forensic. Fails on photorealistic AI video. Demoted to weight 0.15.
- Cross-channel correlation is UNRELIABLE after H.264 compression — real 720p videos score 0.73-0.97, overlapping entirely with AI. Demoted to weight 0.05 (tiebreaker only).
- Noise residual polarity is INVERTED for H.264: real video has HIGH autocorrelation (from deblocking filters), AI has LOWER. The scoring accounts for this inversion.
- Temporal signals are confounded by content type (static interview vs fast sports). Used as supplementary, not primary.
- EfficientNet-B4 weights load from DeepfakeBench FF++ checkpoint but are not calibrated.
- Separation margin on 10-video test set is tight. Wide SUSPICIOUS band (0.35-0.70) is intentional to avoid overconfident wrong answers.
- Scene detection test uses threshold=5.0 (not default 30.0) because mp4v codec compression smooths synthetic test video color transitions.
- Pillow requires >=11.0 for Python 3.13 compatibility.

**Next phases:**
- Phase 6: FastAPI endpoints (POST /analyze, GET /report/{id})
- Phase 7: Benchmarking on FF++, Celeb-DF, DFDC, DF40

## Tech stack

- Python 3.13, PyTorch MPS (Apple Silicon M3 Air, 16GB)
- open_clip (CLIP ViT-L/14), timm (EfficientNet-B4), facenet-pytorch (MTCNN)
- pytorch-grad-cam (GradCAM for ViT + CNN)
- OpenCV, scipy (DCT), ffmpeg/ffprobe
- FastAPI, Pydantic, structlog
- Ollama local (llama3.1) for LLM explainer, Claude API for production
- pytest, ruff

## Development commands

```bash
make install     # pip install -e ".[dev]"
make test        # pytest tests/ -v
make validate    # scripts/validate_setup.py (checks MPS, models, imports)
make smoke       # Quick DCT analyzer test
make lint        # ruff check
make format      # ruff format
make run         # uvicorn dev server
```

## Key integration test commands

```bash
python scripts/test_phase2.py data/sample_videos/VIDEO.mp4   # Stages 1-4
python scripts/test_phase3.py data/sample_videos/VIDEO.mp4   # Full pipeline
```

## Conventions

- All detectors implement BaseDetector (src/verifi/detectors/base.py)
- Config via environment variables with VERIFI_ prefix and __ nesting (e.g., VERIFI_ENSEMBLE__CLIP_WEIGHT)
- LLM backend is pluggable: create_explainer("ollama") or create_explainer("claude")
- Tests use synthetic video fixtures (cv2.VideoWriter) to avoid requiring real video files
- structlog for all logging
- Pipeline outputs go to data/reports/<file_hash>/
- GradCAM runs on CPU (MPS has issues with PyTorch hooks), model moved back to device after

## Test video context

The primary test video is a Veo-generated (Google Gemini) capybara walking on a red carpet with paparazzi. It is 100% AI-generated. MTCNN detects small human faces in the paparazzi crowd but not the capybara (expected — MTCNN only finds human faces). DCT spectrum shows strong low-frequency concentration typical of diffusion models. This video is the reason the dual-path architecture exists.

## Model weights

- CLIP ViT-L/14: uses open_clip OpenAI pretrained weights (downloaded automatically, ~1.71GB). Zero-shot classification, no custom weights needed.
- `data/weights/model.torchscript` — Fine-tuned CLIP from huggingface (yermandy/deepfake-detection), not currently used (TorchScript fails on MPS). Could be used for LN-tuned face-swap detection in future.
- `data/weights/model.ckpt` — Full checkpoint with hyperparameters
- EfficientNet-B4 weights: manual download from DeepfakeBench (not yet downloaded, using ImageNet pretrained)
- DF40 CLIP weights: manual download from Google Drive (not yet downloaded)