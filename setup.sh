#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# VeriFi — Phase 1 Project Setup Script
# Run: chmod +x setup.sh && ./setup.sh
# ──────────────────────────────────────────────────────────────
set -euo pipefail

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; }
step() { echo -e "\n${CYAN}━━━ $1 ━━━${NC}"; }

PROJECT="verifi"
PYTHON_MIN="3.11"

# ──────────────────────────────────────────────────────────────
# 0. Preflight checks
# ──────────────────────────────────────────────────────────────
step "Preflight checks"

# Check Python version
if ! command -v python3 &> /dev/null; then
    err "python3 not found. Install Python >= $PYTHON_MIN"
    exit 1
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJ=$(echo "$PY_VER" | cut -d. -f1)
PY_MIN=$(echo "$PY_VER" | cut -d. -f2)
REQ_MIN=$(echo "$PYTHON_MIN" | cut -d. -f2)

if [ "$PY_MAJ" -lt 3 ] || ([ "$PY_MAJ" -eq 3 ] && [ "$PY_MIN" -lt "$REQ_MIN" ]); then
    err "Python $PY_VER found, but >= $PYTHON_MIN required"
    exit 1
fi
log "Python $PY_VER detected"

# Check for git
if ! command -v git &> /dev/null; then
    warn "git not found — skipping git init"
    HAS_GIT=false
else
    HAS_GIT=true
    log "git detected"
fi

# Check for ffmpeg (needed for audio extraction)
if ! command -v ffmpeg &> /dev/null; then
    warn "ffmpeg not found — install it later for audio-visual analysis"
    warn "  macOS: brew install ffmpeg"
else
    log "ffmpeg detected"
fi

# ──────────────────────────────────────────────────────────────
# 1. Create project directory structure
# ──────────────────────────────────────────────────────────────
step "Creating project structure"

if [ -d "$PROJECT" ]; then
    warn "Directory '$PROJECT' already exists"
    read -p "Overwrite structure? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        err "Aborted."
        exit 1
    fi
fi

mkdir -p "$PROJECT"
cd "$PROJECT"

# Source tree
mkdir -p src/verifi/{ingestion,sampling,preprocessing,detectors,ensemble,explainability,explanation,pipeline,api/routes}
mkdir -p configs scripts notebooks tests/{test_sampling,test_detectors,test_ensemble,test_explainability,test_pipeline,test_api}
mkdir -p data/{weights,datasets,sample_videos,reports}

log "Directory tree created"

# ──────────────────────────────────────────────────────────────
# 2. Create pyproject.toml
# ──────────────────────────────────────────────────────────────
step "Writing pyproject.toml"

cat > pyproject.toml << 'PYPROJECT'
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "verifi"
version = "0.1.0"
description = "Forensic AI-generated video detection with explainability"
requires-python = ">=3.11"
license = {text = "MIT"}

dependencies = [
    # Core ML
    "torch>=2.2",
    "torchvision>=0.17",
    "open-clip-torch>=2.24",
    "timm>=0.9",
    "facenet-pytorch>=2.5",

    # Video + image processing
    "opencv-python-headless>=4.9",
    "ffmpeg-python>=0.2",
    "Pillow>=10.0",
    "scipy>=1.12",

    # Explainability
    "grad-cam>=1.5",

    # API + infrastructure
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "python-multipart>=0.0.6",
    "pydantic>=2.6",
    "pydantic-settings>=2.1",
    "pyyaml>=6.0",
    "httpx>=0.27",

    # Video download
    "yt-dlp>=2024.1",

    # LLM
    "anthropic>=0.40",

    # Utilities
    "structlog>=24.1",
    "numpy>=1.26",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=4.1",
    "ruff>=0.2",
    "ipykernel>=6.28",
    "matplotlib>=3.8",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
PYPROJECT

log "pyproject.toml written"

# ──────────────────────────────────────────────────────────────
# 3. Create environment + .env files
# ──────────────────────────────────────────────────────────────
step "Writing environment files"

cat > .env.example << 'ENVFILE'
# VeriFi Environment Configuration
# Copy to .env and fill in your values

# Claude API key (for forensic explainer)
ANTHROPIC_API_KEY=sk-ant-...

# Device override (auto | mps | cuda | cpu)
VERIFI_DEVICE__PREFERRED=auto

# Sampling
VERIFI_SAMPLING__FRAME_BUDGET=30
VERIFI_SAMPLING__SCENE_THRESHOLD=30.0

# Ensemble weights
VERIFI_ENSEMBLE__CLIP_WEIGHT=0.45
VERIFI_ENSEMBLE__EFFNET_WEIGHT=0.30
VERIFI_ENSEMBLE__FREQUENCY_WEIGHT=0.15
VERIFI_ENSEMBLE__TEMPORAL_WEIGHT=0.10

# Explainer
VERIFI_EXPLAINER__MODEL=claude-sonnet-4-20250514
ENVFILE

# Create actual .env from example if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    log ".env created from template (fill in ANTHROPIC_API_KEY)"
else
    log ".env already exists — skipped"
fi

# ──────────────────────────────────────────────────────────────
# 4. Create config module
# ──────────────────────────────────────────────────────────────
step "Writing core source files"

# ── Package init files ──
for dir in src/verifi src/verifi/{ingestion,sampling,preprocessing,detectors,ensemble,explainability,explanation,pipeline,api,api/routes}; do
    touch "$dir/__init__.py"
done

# ── src/verifi/config.py ──
cat > src/verifi/config.py << 'CONFIG'
"""Centralized configuration with Pydantic Settings + YAML override."""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class DeviceConfig(BaseSettings):
    preferred: str = "auto"

    def resolve(self) -> str:
        if self.preferred != "auto":
            return self.preferred
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


class SamplingConfig(BaseSettings):
    frame_budget: int = 30
    scene_threshold: float = 30.0
    transition_margin: int = 2
    min_laplacian_var: float = 100.0
    min_face_confidence: float = 0.95


class DetectorConfig(BaseSettings):
    clip_weight_path: Path = Path("data/weights/clip_vit_l14_deepfake.torchscript")
    effnet_weight_path: Path = Path("data/weights/efficientnet_b4_ff.pth")
    clip_input_size: int = 224
    effnet_input_size: int = 380


class EnsembleConfig(BaseSettings):
    clip_weight: float = 0.45
    effnet_weight: float = 0.30
    frequency_weight: float = 0.15
    temporal_weight: float = 0.10
    suspicious_threshold: float = 0.30
    manipulated_threshold: float = 0.70


class ExplainerConfig(BaseSettings):
    model: str = "claude-sonnet-4-20250514"
    max_heatmap_frames: int = 5
    max_tokens: int = 1500


class AppConfig(BaseSettings):
    project_root: Path = Path(".")
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    explainer: ExplainerConfig = Field(default_factory=ExplainerConfig)
    max_video_duration_sec: int = 600

    class Config:
        env_prefix = "VERIFI_"
        env_nested_delimiter = "__"


def load_config() -> AppConfig:
    """Load config from environment variables (+ .env file)."""
    from dotenv import load_dotenv
    load_dotenv()
    return AppConfig()
CONFIG

# ── src/verifi/detectors/base.py ──
cat > src/verifi/detectors/base.py << 'DETECTOR_BASE'
"""Abstract base class for all detection signals."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class DetectionResult:
    """Result from a single detector on a single frame."""
    score: float
    raw_logits: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


class BaseDetector(ABC):
    """Interface all detectors must implement."""

    def __init__(self, device: str):
        self.device = device
        self._model = None
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def predict(self, face_crops: list[np.ndarray]) -> list[DetectionResult]:
        ...

    @abstractmethod
    def get_gradcam_target_layer(self):
        ...

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        self._model = None
        self._loaded = False
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
DETECTOR_BASE

# ── src/verifi/detectors/clip_detector.py ──
cat > src/verifi/detectors/clip_detector.py << 'CLIP_DET'
"""CLIP ViT-L/14 deepfake detector (LN-tuned, WACV 2026)."""
import numpy as np
import structlog
import torch
from torchvision import transforms

from verifi.detectors.base import BaseDetector, DetectionResult

logger = structlog.get_logger()


class CLIPDeepfakeDetector(BaseDetector):
    """
    CLIP ViT-L/14 fine-tuned via LayerNorm tuning on FaceForensics++.
    Best cross-dataset generalization of all open-source detectors.
    """

    def __init__(self, weight_path: str, device: str, input_size: int = 224):
        super().__init__(device)
        self.weight_path = weight_path
        self.input_size = input_size
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    def load(self) -> None:
        logger.info("loading_clip_detector", path=self.weight_path, device=self.device)
        try:
            # Try TorchScript first
            self._model = torch.jit.load(self.weight_path, map_location=self.device)
            self._model.eval()
            self._model_format = "torchscript"
            logger.info("clip_loaded", format="torchscript")
        except Exception as e:
            # Fallback: load via open_clip + state dict
            logger.warn("torchscript_failed_trying_open_clip", error=str(e))
            self._load_via_open_clip()
        self._loaded = True

    def _load_via_open_clip(self) -> None:
        """Fallback: load CLIP via open_clip and apply saved weights."""
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        # If weight_path is a state dict (not torchscript), load it
        if self.weight_path.endswith(".pth"):
            state = torch.load(self.weight_path, map_location=self.device)
            model.load_state_dict(state, strict=False)

        self._model = model.visual.to(self.device)
        self._model.eval()
        self._model_format = "open_clip"

        # Add a classification head (2-class: fake/real)
        embed_dim = 768  # ViT-L/14 output dim
        self._classifier = torch.nn.Linear(embed_dim, 2).to(self.device)
        # NOTE: You'll need to load the classifier weights too
        # or train this head on FF++ — see Phase 3 benchmarking
        logger.info("clip_loaded", format="open_clip_fallback")

    def predict(self, face_crops: list[np.ndarray]) -> list[DetectionResult]:
        if not face_crops or not self._loaded:
            return []

        tensors = []
        for crop in face_crops:
            rgb = crop[:, :, ::-1].copy()
            tensors.append(self._transform(rgb))

        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            if self._model_format == "torchscript":
                logits = self._model(batch)
            else:
                features = self._model(batch)
                logits = self._classifier(features)

            probs = torch.softmax(logits, dim=1)
            # Verify label order: index 0 = fake, index 1 = real
            fake_probs = probs[:, 0].cpu().numpy()

        return [
            DetectionResult(
                score=float(p),
                raw_logits=logits[i].cpu().numpy(),
            )
            for i, p in enumerate(fake_probs)
        ]

    def get_gradcam_target_layer(self):
        if self._model_format == "torchscript":
            # TorchScript may not expose named layers — needs testing
            logger.warn("gradcam_torchscript_limited")
            return None
        # For open_clip ViT-L/14:
        return self._model.transformer.resblocks[-1].ln_2
CLIP_DET

# ── src/verifi/detectors/effnet_detector.py ──
cat > src/verifi/detectors/effnet_detector.py << 'EFFNET_DET'
"""EfficientNet-B4 artifact-focused deepfake detector."""
import numpy as np
import structlog
import timm
import torch
from torchvision import transforms

from verifi.detectors.base import BaseDetector, DetectionResult

logger = structlog.get_logger()


class EfficientNetDetector(BaseDetector):
    """
    EfficientNet-B4 trained on FF++ (DeepfakeBench weights).
    Focuses on pixel-level artifacts: blending boundaries,
    compression inconsistencies, texture anomalies.
    """

    def __init__(self, weight_path: str, device: str, input_size: int = 380):
        super().__init__(device)
        self.weight_path = weight_path
        self.input_size = input_size
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def load(self) -> None:
        logger.info("loading_effnet_detector", path=self.weight_path, device=self.device)
        self._model = timm.create_model(
            "efficientnet_b4", pretrained=False, num_classes=2
        )
        try:
            state = torch.load(self.weight_path, map_location=self.device, weights_only=True)
            self._model.load_state_dict(state, strict=False)
        except FileNotFoundError:
            logger.warn("effnet_weights_not_found_using_imagenet", path=self.weight_path)
            self._model = timm.create_model(
                "efficientnet_b4", pretrained=True, num_classes=2
            )
        self._model = self._model.to(self.device)
        self._model.eval()
        self._loaded = True
        logger.info("effnet_loaded")

    def predict(self, face_crops: list[np.ndarray]) -> list[DetectionResult]:
        if not face_crops or not self._loaded:
            return []

        tensors = [self._transform(c[:, :, ::-1].copy()) for c in face_crops]
        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            logits = self._model(batch)
            probs = torch.softmax(logits, dim=1)
            fake_probs = probs[:, 0].cpu().numpy()

        return [
            DetectionResult(score=float(p), raw_logits=logits[i].cpu().numpy())
            for i, p in enumerate(fake_probs)
        ]

    def get_gradcam_target_layer(self):
        return self._model.conv_head
EFFNET_DET

# ── src/verifi/detectors/frequency.py ──
cat > src/verifi/detectors/frequency.py << 'FREQ_DET'
"""DCT frequency analysis — pure signal processing, no ML."""
import cv2
import numpy as np
from scipy.fft import dctn

from verifi.detectors.base import DetectionResult


class FrequencyAnalyzer:
    """
    Detect GAN/diffusion fingerprints in the frequency domain.
    Generators suppress high-frequency energy and leave periodic
    patterns from upsampling layers.
    """

    def __init__(self, baseline_hf_ratio: float = 0.38):
        self.baseline_hf_ratio = baseline_hf_ratio

    def analyze(self, face_crop: np.ndarray) -> DetectionResult:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256)).astype(np.float32)

        dct = dctn(gray, norm="ortho")
        magnitude = np.abs(dct)

        h, w = magnitude.shape
        mid_h, mid_w = h // 4, w // 4
        lf_energy = magnitude[:mid_h, :mid_w].sum()
        hf_energy = magnitude[mid_h:, :].sum() + magnitude[:mid_h, mid_w:].sum()

        total = lf_energy + hf_energy
        hf_ratio = float(hf_energy / total) if total > 0 else 0.5

        suppression = max(0, self.baseline_hf_ratio - hf_ratio) / self.baseline_hf_ratio
        score = min(1.0, suppression * 2.5)

        return DetectionResult(
            score=score,
            metadata={
                "hf_ratio": hf_ratio,
                "hf_suppression_pct": round(
                    (self.baseline_hf_ratio - hf_ratio) / self.baseline_hf_ratio * 100, 1
                ),
                "lf_energy": float(lf_energy),
                "hf_energy": float(hf_energy),
            },
        )

    def generate_spectrum_image(self, face_crop: np.ndarray) -> np.ndarray:
        """Generate a visual DCT spectrum image for the forensic report."""
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256)).astype(np.float32)

        dct = dctn(gray, norm="ortho")
        magnitude = np.log1p(np.abs(dct))

        # Normalize to 0-255 for visualization
        magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
        colored = cv2.applyColorMap(magnitude, cv2.COLORMAP_JET)
        return colored
FREQ_DET

# ── src/verifi/detectors/temporal.py ──
cat > src/verifi/detectors/temporal.py << 'TEMP_DET'
"""Temporal consistency analysis using optical flow."""
import cv2
import numpy as np

from verifi.detectors.base import DetectionResult


class TemporalAnalyzer:
    """
    Check temporal consistency between adjacent frames.
    Deepfakes often show motion discontinuities in the face
    region that differ from natural background motion.
    """

    def __init__(self, divergence_threshold: float = 2.0):
        self.divergence_threshold = divergence_threshold

    def analyze_pair(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        face_bbox: tuple[int, int, int, int] | None = None,
    ) -> DetectionResult:
        """
        Compare optical flow consistency between two frames.

        Args:
            frame_a, frame_b: BGR frames (same resolution).
            face_bbox: (x, y, w, h) of face region. If None, uses center crop.

        Returns:
            DetectionResult with divergence score.
        """
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray_a, gray_b, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

        # Compute flow magnitude
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Split into face region and background
        h, w = mag.shape
        if face_bbox:
            x, y, bw, bh = face_bbox
            face_mask = np.zeros((h, w), dtype=bool)
            face_mask[y:y+bh, x:x+bw] = True
        else:
            # Default: center 40% as face proxy
            ch, cw = h // 5, w // 5
            face_mask = np.zeros((h, w), dtype=bool)
            face_mask[ch*2:ch*3, cw*2:cw*3] = True

        bg_mask = ~face_mask

        face_flow_mean = mag[face_mask].mean() if face_mask.any() else 0
        bg_flow_mean = mag[bg_mask].mean() if bg_mask.any() else 0
        bg_flow_std = mag[bg_mask].std() if bg_mask.any() else 1.0

        # Divergence: how many std devs does face flow differ from background
        if bg_flow_std > 0.01:
            divergence = abs(face_flow_mean - bg_flow_mean) / bg_flow_std
        else:
            divergence = 0.0

        # Map to [0, 1] score
        score = min(1.0, divergence / (self.divergence_threshold * 2))

        return DetectionResult(
            score=score,
            metadata={
                "face_flow_mean": float(face_flow_mean),
                "bg_flow_mean": float(bg_flow_mean),
                "divergence_sigma": float(divergence),
            },
        )
TEMP_DET

# ──────────────────────────────────────────────────────────────
# 5. Create scripts
# ──────────────────────────────────────────────────────────────
step "Writing scripts"

# ── scripts/download_weights.py ──
cat > scripts/download_weights.py << 'DOWNLOAD'
"""Download pre-trained model weights for VeriFi."""
import subprocess
import sys
from pathlib import Path

WEIGHTS_DIR = Path("data/weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS = {
    "clip_deepfake_torchscript": {
        "url": "https://github.com/yermandy/deepfake-detection/releases/download/v1.0/model.torchscript",
        "filename": "clip_vit_l14_deepfake.torchscript",
        "description": "CLIP ViT-L/14 LN-tuned (WACV 2026)",
    },
}

MANUAL_DOWNLOADS = {
    "df40_clip_weights": {
        "source": "https://github.com/YZY-stack/DF40",
        "instructions": "Download from Google Drive link in DF40 repo README",
        "save_as": "data/weights/clip_df40.pth",
    },
    "deepfakebench_effnet": {
        "source": "https://github.com/SCLBD/DeepfakeBench",
        "instructions": "Follow DeepfakeBench setup to get EfficientNet-B4 weights",
        "save_as": "data/weights/efficientnet_b4_ff.pth",
    },
}


def download(name: str, info: dict) -> bool:
    dest = WEIGHTS_DIR / info["filename"]
    if dest.exists():
        print(f"  [skip] {name}: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    print(f"  [download] {name}: {info['description']}")
    print(f"             {info['url']}")
    try:
        subprocess.run(
            ["curl", "-L", "--progress-bar", "-o", str(dest), info["url"]],
            check=True,
        )
        print(f"  [done] saved to {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"  [error] Failed to download {name}")
        return False


def main():
    print("=" * 60)
    print("VeriFi — Model Weight Downloader")
    print("=" * 60)

    print("\n── Auto-downloads ──")
    for name, info in WEIGHTS.items():
        download(name, info)

    print("\n── Manual downloads required ──")
    for name, info in MANUAL_DOWNLOADS.items():
        path = Path(info["save_as"])
        if path.exists():
            print(f"  [skip] {name}: already at {path}")
        else:
            print(f"  [TODO] {name}")
            print(f"         Source: {info['source']}")
            print(f"         {info['instructions']}")
            print(f"         Save to: {info['save_as']}")

    print("\n── Verification ──")
    for f in WEIGHTS_DIR.iterdir():
        if f.is_file():
            print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
DOWNLOAD

# ── scripts/validate_setup.py ──
cat > scripts/validate_setup.py << 'VALIDATE'
"""Phase 1 validation: check that everything works before building."""
import sys
import time
from pathlib import Path

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

errors = []


def check(name: str, fn):
    try:
        result = fn()
        if result:
            print(f"  {PASS} {name}: {result}")
        else:
            print(f"  {PASS} {name}")
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        errors.append(name)


def check_torch():
    import torch
    return f"v{torch.__version__}"


def check_device():
    import torch
    if torch.backends.mps.is_available():
        return "MPS (Apple Silicon GPU) available"
    elif torch.cuda.is_available():
        return f"CUDA available ({torch.cuda.get_device_name(0)})"
    else:
        return "CPU only (MPS/CUDA not available)"


def check_mps_inference():
    import torch
    if not torch.backends.mps.is_available():
        print(f"  {WARN} MPS not available — skipping GPU benchmark")
        return None
    device = "mps"
    x = torch.randn(1, 3, 224, 224, device=device)
    # Warm up
    for _ in range(3):
        _ = x * 2 + 1
    torch.mps.synchronize()
    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        _ = x * 2 + 1
    torch.mps.synchronize()
    elapsed = (time.perf_counter() - start) * 10  # ms per op
    return f"MPS basic op: {elapsed:.2f}ms/100 ops"


def check_open_clip():
    import open_clip
    models = open_clip.list_pretrained()
    vit_l14 = [m for m in models if "ViT-L-14" in m[0]]
    return f"open_clip loaded, {len(vit_l14)} ViT-L-14 variants available"


def check_timm():
    import timm
    assert "efficientnet_b4" in timm.list_models("efficientnet*")
    return "efficientnet_b4 available"


def check_cv2():
    import cv2
    return f"OpenCV v{cv2.__version__}"


def check_gradcam():
    from pytorch_grad_cam import GradCAM
    return "pytorch-grad-cam imported"


def check_facenet():
    from facenet_pytorch import MTCNN
    return "MTCNN available"


def check_anthropic():
    import anthropic
    return f"anthropic SDK v{anthropic.__version__}"


def check_fastapi():
    import fastapi
    return f"FastAPI v{fastapi.__version__}"


def check_weights():
    weights_dir = Path("data/weights")
    if not weights_dir.exists():
        return "data/weights/ not found — run: python scripts/download_weights.py"
    files = list(weights_dir.glob("*"))
    if not files:
        print(f"  {WARN} No weight files found — run: python scripts/download_weights.py")
        return None
    return f"{len(files)} weight file(s) found"


def check_clip_inference():
    """The big test: can we actually run CLIP inference?"""
    import torch
    from verifi.config import AppConfig

    config = AppConfig()
    device = config.device.resolve()

    weight_path = config.detector.clip_weight_path
    if not weight_path.exists():
        print(f"  {WARN} CLIP weights not found at {weight_path} — skipping inference test")
        return None

    from verifi.detectors.clip_detector import CLIPDeepfakeDetector
    import numpy as np

    detector = CLIPDeepfakeDetector(
        weight_path=str(weight_path),
        device=device,
        input_size=config.detector.clip_input_size,
    )
    detector.load()

    # Create dummy face crop (224x224 random pixels)
    dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Warm up
    detector.predict([dummy])

    # Benchmark
    start = time.perf_counter()
    results = detector.predict([dummy] * 4)  # batch of 4
    elapsed = (time.perf_counter() - start) * 1000

    detector.unload()
    return f"Batch of 4 on {device}: {elapsed:.0f}ms ({elapsed/4:.0f}ms/frame), scores: {[f'{r.score:.3f}' for r in results]}"


def main():
    print("=" * 60)
    print("VeriFi — Phase 1 Setup Validation")
    print("=" * 60)

    print("\n── Core dependencies ──")
    check("PyTorch", check_torch)
    check("Compute device", check_device)
    check("MPS inference", check_mps_inference)
    check("OpenCLIP", check_open_clip)
    check("timm", check_timm)
    check("OpenCV", check_cv2)
    check("pytorch-grad-cam", check_gradcam)
    check("MTCNN (facenet-pytorch)", check_facenet)
    check("Anthropic SDK", check_anthropic)
    check("FastAPI", check_fastapi)

    print("\n── Model weights ──")
    check("Weight files", check_weights)

    print("\n── Inference test ──")
    check("CLIP ViT-L/14 inference", check_clip_inference)

    print("\n" + "=" * 60)
    if errors:
        print(f"\033[91m{len(errors)} check(s) failed: {', '.join(errors)}\033[0m")
        print("Fix these before proceeding to Phase 2.")
        sys.exit(1)
    else:
        print(f"\033[92mAll checks passed. Ready to build.\033[0m")
        sys.exit(0)


if __name__ == "__main__":
    main()
VALIDATE

# ──────────────────────────────────────────────────────────────
# 6. Create test fixtures + initial tests
# ──────────────────────────────────────────────────────────────
step "Writing test scaffolding"

cat > tests/conftest.py << 'CONFTEST'
"""Shared test fixtures for VeriFi."""
import numpy as np
import pytest


@pytest.fixture
def dummy_face_crop():
    """224x224 BGR face crop (random pixels)."""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def dummy_face_crops_batch():
    """Batch of 4 dummy face crops."""
    return [
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        for _ in range(4)
    ]


@pytest.fixture
def dummy_frame_pair():
    """Two consecutive 720p frames for temporal analysis."""
    a = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    b = a.copy()
    # Add slight motion to background
    b[:, 5:] = a[:, :-5]
    # Add larger motion to face region (center)
    b[260:460, 540:740] = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    return a, b
CONFTEST

cat > tests/test_detectors/test_frequency.py << 'TEST_FREQ'
"""Tests for DCT frequency analyzer."""
import numpy as np
from verifi.detectors.frequency import FrequencyAnalyzer


def test_frequency_returns_valid_score(dummy_face_crop):
    analyzer = FrequencyAnalyzer()
    result = analyzer.analyze(dummy_face_crop)
    assert 0.0 <= result.score <= 1.0


def test_frequency_metadata_keys(dummy_face_crop):
    analyzer = FrequencyAnalyzer()
    result = analyzer.analyze(dummy_face_crop)
    assert "hf_ratio" in result.metadata
    assert "hf_suppression_pct" in result.metadata
    assert "lf_energy" in result.metadata
    assert "hf_energy" in result.metadata


def test_frequency_spectrum_image(dummy_face_crop):
    analyzer = FrequencyAnalyzer()
    spectrum = analyzer.generate_spectrum_image(dummy_face_crop)
    assert spectrum.shape == (256, 256, 3)
    assert spectrum.dtype == np.uint8
TEST_FREQ

cat > tests/test_detectors/test_temporal.py << 'TEST_TEMP'
"""Tests for temporal consistency analyzer."""
from verifi.detectors.temporal import TemporalAnalyzer


def test_temporal_returns_valid_score(dummy_frame_pair):
    analyzer = TemporalAnalyzer()
    frame_a, frame_b = dummy_frame_pair
    result = analyzer.analyze_pair(frame_a, frame_b)
    assert 0.0 <= result.score <= 1.0


def test_temporal_identical_frames():
    """Identical frames should produce low divergence."""
    import numpy as np
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    analyzer = TemporalAnalyzer()
    result = analyzer.analyze_pair(frame, frame.copy())
    assert result.score < 0.1


def test_temporal_metadata_keys(dummy_frame_pair):
    analyzer = TemporalAnalyzer()
    frame_a, frame_b = dummy_frame_pair
    result = analyzer.analyze_pair(frame_a, frame_b)
    assert "face_flow_mean" in result.metadata
    assert "bg_flow_mean" in result.metadata
    assert "divergence_sigma" in result.metadata
TEST_TEMP

cat > tests/test_detectors/__init__.py << 'EOF'
EOF

# ──────────────────────────────────────────────────────────────
# 7. Create Makefile
# ──────────────────────────────────────────────────────────────
step "Writing Makefile"

cat > Makefile << 'MAKEFILE'
.PHONY: setup install test lint format run validate weights clean

# Full setup from scratch
setup: install weights validate
	@echo "\n\033[92m✓ Setup complete. Run 'make test' to verify.\033[0m"

# Install dependencies
install:
	python3 -m pip install -e ".[dev]"
	python3 -m pip install python-dotenv

# Download model weights
weights:
	python3 scripts/download_weights.py

# Validate entire setup
validate:
	python3 scripts/validate_setup.py

# Run tests
test:
	python3 -m pytest tests/ -v --tb=short

# Run tests with coverage
test-cov:
	python3 -m pytest tests/ -v --cov=verifi --cov-report=term-missing

# Lint
lint:
	python3 -m ruff check src/ tests/

# Format
format:
	python3 -m ruff format src/ tests/

# Run API server (development)
run:
	python3 -m uvicorn verifi.api.app:app --reload --port 8000

# Quick smoke test: run frequency analyzer on a dummy image
smoke:
	python3 -c "import numpy as np; from verifi.detectors.frequency import FrequencyAnalyzer; \
		a = FrequencyAnalyzer(); r = a.analyze(np.random.randint(0,255,(224,224,3),dtype=np.uint8)); \
		print(f'Score: {r.score:.3f}, HF ratio: {r.metadata[\"hf_ratio\"]:.3f}')"

# Clean caches
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache build dist
MAKEFILE

# ──────────────────────────────────────────────────────────────
# 8. Create .gitignore
# ──────────────────────────────────────────────────────────────
step "Writing .gitignore"

cat > .gitignore << 'GITIGNORE'
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/

# Environment
.env

# Data (large files)
data/weights/
data/datasets/
data/sample_videos/
data/reports/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Notebooks
.ipynb_checkpoints/

# Testing
.pytest_cache/
.coverage
htmlcov/

# Ruff
.ruff_cache/
GITIGNORE

# ──────────────────────────────────────────────────────────────
# 9. Create README
# ──────────────────────────────────────────────────────────────
step "Writing README.md"

cat > README.md << 'README'
# VeriFi — Forensic AI-Generated Video Detection

Deepfake and AI-generated video detection with multi-signal ensemble analysis and natural language forensic explanations.

## Quick Start

```bash
# 1. Setup
chmod +x setup.sh && ./setup.sh   # or run steps manually:

# 2. Install dependencies
make install

# 3. Download model weights
make weights

# 4. Validate setup
make validate

# 5. Run tests
make test

# 6. Quick smoke test
make smoke
```

## Architecture

VeriFi uses a multi-signal detection ensemble:

- **CLIP ViT-L/14** — semantic deepfake detection (LN-tuned on FF++)
- **EfficientNet-B4** — pixel-level artifact detection
- **DCT Frequency Analysis** — generator fingerprint detection (no ML)
- **Temporal Consistency** — optical flow anomaly detection

Results are explained via GradCAM heatmaps and Claude API natural language reasoning.

## Configuration

Copy `.env.example` to `.env` and set your `ANTHROPIC_API_KEY`.
All settings can be overridden via environment variables with `VERIFI_` prefix.

## Development

```bash
make test       # run tests
make lint       # check code style
make format     # auto-format code
make run        # start API server
```
README

# ──────────────────────────────────────────────────────────────
# 10. Create configs/default.yaml
# ──────────────────────────────────────────────────────────────
step "Writing default config"

cat > configs/default.yaml << 'YAML'
# VeriFi default configuration
# Override via environment variables (VERIFI_ prefix)

device:
  preferred: auto  # auto | mps | cuda | cpu

sampling:
  frame_budget: 30
  scene_threshold: 30.0
  transition_margin: 2
  min_laplacian_var: 100.0
  min_face_confidence: 0.95

detector:
  clip_weight_path: data/weights/clip_vit_l14_deepfake.torchscript
  effnet_weight_path: data/weights/efficientnet_b4_ff.pth
  clip_input_size: 224
  effnet_input_size: 380

ensemble:
  clip_weight: 0.45
  effnet_weight: 0.30
  frequency_weight: 0.15
  temporal_weight: 0.10
  suspicious_threshold: 0.30
  manipulated_threshold: 0.70

explainer:
  model: claude-sonnet-4-20250514
  max_heatmap_frames: 5
  max_tokens: 1500

max_video_duration_sec: 600
YAML

# ──────────────────────────────────────────────────────────────
# 11. Initialize git repo
# ──────────────────────────────────────────────────────────────
if [ "$HAS_GIT" = true ]; then
    step "Initializing git repository"
    git init -q
    git add -A
    git commit -q -m "feat: initial project scaffold — Phase 1 setup

- Project structure with all modules stubbed
- CLIP ViT-L/14 + EfficientNet-B4 + DCT + temporal detectors
- Pydantic config with env var overrides
- Weight download script
- Setup validation script
- Test scaffolding with fixtures
- Makefile for common commands"
    log "Initial commit created"
fi

# ──────────────────────────────────────────────────────────────
# 12. Summary
# ──────────────────────────────────────────────────────────────
step "Setup complete"

echo -e "
${GREEN}Project created at: $(pwd)${NC}

${CYAN}Next steps:${NC}

  ${YELLOW}1.${NC} Install dependencies:
     ${CYAN}make install${NC}

  ${YELLOW}2.${NC} Download model weights:
     ${CYAN}make weights${NC}

  ${YELLOW}3.${NC} Set your Claude API key:
     ${CYAN}Edit .env → ANTHROPIC_API_KEY=sk-ant-...${NC}

  ${YELLOW}4.${NC} Validate everything works:
     ${CYAN}make validate${NC}

  ${YELLOW}5.${NC} Run the smoke test:
     ${CYAN}make smoke${NC}

  ${YELLOW}6.${NC} Run tests:
     ${CYAN}make test${NC}

${CYAN}Project structure:${NC}
  src/verifi/detectors/   — Detection models (CLIP, EfficientNet, DCT, temporal)
  src/verifi/sampling/    — Smart frame sampling (to be built Phase 2)
  src/verifi/pipeline/    — Orchestrator (to be built Phase 5)
  src/verifi/api/         — FastAPI server (to be built Phase 6)
  scripts/                — Setup + validation utilities
  tests/                  — Test suite
  data/weights/           — Model checkpoints (.gitignored)
"