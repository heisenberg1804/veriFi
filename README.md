# VeriFi

Forensic AI-generated video detection engine with multi-signal ensemble analysis, GradCAM visual explainability, and LLM-powered natural language forensic reports.

---

## The Problem

AI-generated and deepfake videos have become nearly indistinguishable from authentic content. Existing detection tools output a confidence score with no explanation — useless for journalists, trust-and-safety teams, and legal reviewers who need to understand *what* was manipulated, *where*, and *why* the system flagged it.

## What VeriFi Does

VeriFi analyzes a video through four independent detection signals, generates visual heatmaps highlighting suspicious regions, and produces a structured natural language forensic report explaining the findings. The output isn't a binary verdict — it's an investigation toolkit.

**Input:** Video file (MP4/MOV/WebM) or URL

**Output:**
- Three-tier verdict: Likely Authentic / Suspicious / Likely Manipulated
- Per-frame confidence timeline
- GradCAM attention heatmaps on flagged frames
- Side-by-side forensic view (original | heatmap | DCT spectrum)
- Natural language explanation of what was detected and why
- Structured JSON report for downstream automation

---

## Detection Pipeline

```
Video → Smart Frame Sampling → Face Detection → Multi-Signal Ensemble → Forensic Report
```

### Detection Signals

| Signal | Type | What It Catches |
|---|---|---|
| **CLIP ViT-L/14** | Semantic (ML) | Learned forgery patterns — best cross-dataset generalization |
| **EfficientNet-B4** | Artifact (ML) | Pixel-level blending boundaries, texture anomalies |
| **DCT Frequency** | Signal processing | GAN/diffusion spectral fingerprints — no ML, immune to adversarial attacks |
| **Temporal Consistency** | Optical flow | Motion discontinuities between face region and background |

### Smart Frame Sampling

Not uniform sampling. A three-pass strategy that allocates frame budget proportionally to scene duration, maximizes visual diversity within each scene via greedy farthest-point selection, and boosts sampling density around scene transitions where deepfake artifacts are most visible.

### Forensic Explainability

- **GradCAM heatmaps** from both CLIP and EfficientNet, highlighting which regions triggered each detector independently
- **DCT spectrum visualization** showing frequency-domain anomalies invisible to the naked eye
- **LLM-generated forensic explanation** that synthesizes all signals into a human-readable report with evidence, caveats, and recommended next steps

---

## Architecture

```
src/verifi/
├── ingestion/        # Video validation, metadata extraction, URL download
├── sampling/         # Scene detection, smart frame selection, quality filter
├── preprocessing/    # Face detection (MTCNN), alignment, cross-frame tracking
├── detectors/        # CLIP, EfficientNet, DCT frequency, temporal consistency
├── ensemble/         # Weighted aggregation, confidence calibration
├── explainability/   # GradCAM heatmaps, forensic view rendering
├── explanation/      # LLM explainer (Ollama local / Claude API), prompt templates
├── pipeline/         # End-to-end orchestrator
└── api/              # FastAPI server (POST /analyze, GET /report/{id})
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- macOS (Apple Silicon MPS) or Linux (CUDA)
- ffmpeg (`brew install ffmpeg`)
- Ollama for local LLM (`brew install ollama && ollama pull llama3.1`)

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/verifi.git
cd verifi

python3 -m venv .venv
source .venv/bin/activate

make install        # Install dependencies
make weights        # Download CLIP detection model (~900MB)
make validate       # Verify setup (MPS, models, imports)
make test           # Run test suite
```

### Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Key settings:

```env
# LLM backend (free local inference)
VERIFI_EXPLAINER__BACKEND=ollama
VERIFI_EXPLAINER__MODEL=llama3.1

# Switch to Claude for production-quality reports
# VERIFI_EXPLAINER__BACKEND=claude
# ANTHROPIC_API_KEY=sk-ant-...
```

### Analyze a Video

```bash
# Integration test on a sample video
python scripts/test_phase2.py data/sample_videos/your_video.mp4

# Start the API server
make run

# Analyze via API
curl -X POST http://localhost:8000/analyze \
  -F "video=@path/to/video.mp4"
```

---

## Model Weights

| Model | Source | Size | Download |
|---|---|---|---|
| CLIP ViT-L/14 (LN-tuned) | [yermandy/deepfake-detection](https://huggingface.co/yermandy/deepfake-detection) | ~900 MB | `make weights` (auto) |
| EfficientNet-B4 | [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) | ~75 MB | Manual (see docs) |
| DF40 CLIP weights | [DF40](https://github.com/YZY-stack/DF40) | ~900 MB | Manual (see docs) |

The CLIP TorchScript model downloads automatically via `make weights`. EfficientNet and DF40 weights require manual download from their respective repositories — see `scripts/download_weights.py` for instructions.

---

## Evaluation

Trained on FaceForensics++ (c23). Cross-dataset evaluation protocol:

| Dataset | Year | Videos | Purpose |
|---|---|---|---|
| FaceForensics++ | 2019 | 5,000 | Training base |
| Celeb-DF v2 | 2020 | 6,229 | Cross-dataset generalization |
| DFDC (Meta) | 2020 | 119,197 | Demographic diversity |
| DF40 | 2024 | 40 methods, million-scale | Modern generator coverage |
| OpenFake | 2025 | ~4M images | Real-world political deepfakes |

```bash
# Run evaluation benchmarks
make bench
```

---

## Tech Stack

- **ML:** PyTorch, OpenCLIP, timm, facenet-pytorch, pytorch-grad-cam
- **Video:** OpenCV, ffmpeg, yt-dlp
- **API:** FastAPI, Pydantic
- **LLM:** Ollama (local) / Anthropic Claude API (production)
- **Compute:** Apple Silicon MPS / NVIDIA CUDA

---

## Development

```bash
make test       # Run test suite
make test-cov   # Tests with coverage report
make lint       # Ruff linting
make format     # Ruff auto-format
make smoke      # Quick detector smoke test
make validate   # Full setup validation
make run        # Start dev API server
```

---

## Roadmap

- [x] Multi-signal detection ensemble (CLIP + EfficientNet + DCT + temporal)
- [x] Smart frame sampling with scene-aware budget allocation
- [x] Face detection, alignment, and cross-frame tracking
- [ ] GradCAM heatmap generation for CLIP ViT and EfficientNet
- [ ] LLM forensic explainer integration (Ollama + Claude)
- [ ] End-to-end pipeline orchestrator
- [ ] FastAPI endpoints with async processing
- [ ] Cross-dataset benchmarking (FF++, Celeb-DF, DFDC, DF40)
- [ ] Audio-visual consistency analysis (SyncNet)
- [ ] Shareable forensic report (static HTML with unique URL)
- [ ] Confidence calibration (Platt scaling)
- [ ] Docker deployment packaging

---

## License

MIT

---

## References

- Yermakov et al., *"Unlocking the Hidden Potential of CLIP in Generalizable Deepfake Detection"*, WACV 2026
- Yan et al., *"DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection"*, NeurIPS 2023
- Yan et al., *"DF40: Toward Next-Generation Deepfake Detection"*, NeurIPS 2024
- Guo et al., *"Rethinking Vision-Language Model in Face Forensics: Multi-Modal Interpretable Forged Face Detector"*, CVPR 2025 (Oral)
- Livernoche et al., *"OpenFake: An Open Dataset and Platform Toward Real-World Deepfake Detection"*, 2025