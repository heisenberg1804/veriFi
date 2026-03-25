#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# VeriFi — Create missing files that Phase 2 code depends on
# Run from project root (where pyproject.toml is)
# ──────────────────────────────────────────────────────────────
set -euo pipefail

if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: Run from project root"
    exit 1
fi

mkdir -p src/verifi/explanation
mkdir -p src/verifi/sampling

# ── 1. Prompt templates (required by llm_explainer.py) ──
cat > src/verifi/explanation/__init__.py << 'EOF'
EOF

cat > src/verifi/explanation/prompts.py << 'ENDOFFILE'
"""Versioned prompt templates for the forensic explainer."""

FORENSIC_SYSTEM_PROMPT = """\
You are a forensic video analyst AI specializing in deepfake and \
AI-generated media detection. You receive detection signal data and \
visual heatmaps from an automated analysis pipeline, and your job is \
to produce a clear, accurate forensic explanation.

Your audience includes:
- Trust-and-safety reviewers at social media platforms
- Journalists fact-checking content
- Legal/compliance teams evaluating video evidence

Guidelines:
- Be precise and evidence-based. Reference specific signals and regions.
- Acknowledge uncertainty. Never state something is definitively fake \
  or real — use probabilistic language ("strong indicators of", \
  "consistent with", "suggestive but not conclusive").
- Distinguish between signals. Explain which detectors flagged what, \
  and whether multiple independent signals corroborate each other.
- Note caveats. Mention compression artifacts, video quality, and \
  other factors that affect detection confidence.
- Structure your output as valid JSON matching the schema the user \
  provides. Do NOT include markdown fences or any text outside the JSON."""


FORENSIC_USER_TEMPLATE = """\
Analyze the following deepfake detection results and generate a \
forensic report.

## Video Metadata
- Duration: {duration_sec}s | Resolution: {resolution} | FPS: {fps}
- Codec: {codec} | Has Audio: {has_audio}

## Detection Scores (per-frame ensemble)
- Video-level score: {video_score:.3f}
- Verdict: {verdict}
- Inferred manipulation type: {manipulation_type}
- Frames analyzed: {num_frames} | Frames flagged: {num_flagged}

## Signal Breakdown
- CLIP ViT-L/14: mean={clip_mean:.3f}, max={clip_max:.3f}, std={clip_std:.3f}
- EfficientNet-B4: mean={effnet_mean:.3f}, max={effnet_max:.3f}, std={effnet_std:.3f}
- DCT Frequency: anomaly={freq_score:.3f}, HF suppression={hf_suppression:.1f}%
- Temporal Consistency: {temporal_summary}
- AV Sync: {av_sync_summary}

## GradCAM Heatmap Analysis
The attached images show GradCAM attention maps for the top flagged \
frames. Bright regions indicate where each model detected anomalies. \
Describe what regions are highlighted and what this suggests about the \
manipulation technique.

## Frame Timeline
Peak scores at timestamps: {peak_timestamps}
Score pattern: {score_pattern}

Respond ONLY with a JSON object matching this structure:
{{
  "summary": "2-3 sentence overall assessment",
  "evidence": ["evidence point 1", "evidence point 2", "..."],
  "manipulation_type_reasoning": "why you believe it is this type",
  "caveats": ["caveat 1", "caveat 2", "..."],
  "confidence_assessment": "how confident should the user be in this result",
  "recommended_action": "what the reviewer should do next"
}}"""
ENDOFFILE

echo "[✓] src/verifi/explanation/prompts.py"

# ── 2. Ensure all __init__.py files exist ──
# These are required for Python to treat directories as packages.
for dir in \
    src/verifi \
    src/verifi/ingestion \
    src/verifi/sampling \
    src/verifi/preprocessing \
    src/verifi/detectors \
    src/verifi/ensemble \
    src/verifi/explainability \
    src/verifi/explanation \
    src/verifi/pipeline \
    src/verifi/api \
    src/verifi/api/routes
do
    mkdir -p "$dir"
    touch "$dir/__init__.py"
done

echo "[✓] All __init__.py files verified"

# ── 3. Ensure quality_filter.py exists (imported by some references) ──
cat > src/verifi/sampling/quality_filter.py << 'ENDOFFILE'
"""Standalone quality filter utilities."""
import cv2
import numpy as np


def compute_blur_score(frame: np.ndarray) -> float:
    """Laplacian variance — higher means sharper."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_sharp(frame: np.ndarray, threshold: float = 100.0) -> bool:
    """Quick check if a frame passes the blur threshold."""
    return compute_blur_score(frame) >= threshold
ENDOFFILE

echo "[✓] src/verifi/sampling/quality_filter.py"

# ── 4. Ensure downloader stub exists (referenced in project structure) ──
cat > src/verifi/ingestion/downloader.py << 'ENDOFFILE'
"""Video URL downloader using yt-dlp."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import structlog

logger = structlog.get_logger()


def download_video(url: str, output_dir: str | Path = "data/sample_videos") -> Path:
    """
    Download a video from URL using yt-dlp.

    Args:
        url: Video URL (YouTube, Twitter, TikTok, direct .mp4 link, etc.)
        output_dir: Directory to save downloaded video.

    Returns:
        Path to downloaded video file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_template = str(output_dir / "%(title)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "best[height<=720]",
        "--max-filesize", "100M",
        "-o", output_template,
        "--print", "after_move:filepath",
        url,
    ]

    logger.info("downloading_video", url=url)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr[:300]}")

        filepath = result.stdout.strip().split("\n")[-1]
        path = Path(filepath)
        logger.info("download_complete", path=str(path), size_mb=f"{path.stat().st_size/1e6:.1f}")
        return path

    except FileNotFoundError:
        raise RuntimeError("yt-dlp not found. Install: pip install yt-dlp")
ENDOFFILE

echo "[✓] src/verifi/ingestion/downloader.py"

# ── 5. Ensure face_aligner.py and face_tracker.py stubs exist ──
cat > src/verifi/preprocessing/face_aligner.py << 'ENDOFFILE'
"""
Face alignment utilities.

Note: Alignment is integrated into face_detector.py's FaceDetectionPipeline.
This module exists for standalone use if needed.
"""
import cv2
import numpy as np


def align_face_by_eyes(
    image: np.ndarray,
    left_eye: tuple[float, float],
    right_eye: tuple[float, float],
) -> np.ndarray:
    """Rotate image so the eye line is horizontal."""
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = float(np.degrees(np.arctan2(dy, dx)))
    center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    h, w = image.shape[:2]
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
ENDOFFILE

echo "[✓] src/verifi/preprocessing/face_aligner.py"

cat > src/verifi/preprocessing/face_tracker.py << 'ENDOFFILE'
"""
Cross-frame face tracking.

Note: IoU-based tracking is integrated into face_detector.py's
FaceDetectionPipeline. This module exists for advanced tracking
(e.g., embedding-based re-identification) in future phases.
"""
ENDOFFILE

echo "[✓] src/verifi/preprocessing/face_tracker.py"

# ── 6. Add python-dotenv to pyproject.toml if missing ──
if ! grep -q "python-dotenv" pyproject.toml; then
    sed -i '' 's/"structlog>=24.1",/"structlog>=24.1",\n    "python-dotenv>=1.0",/' pyproject.toml
    echo "[✓] Added python-dotenv to pyproject.toml"
else
    echo "[✓] python-dotenv already in pyproject.toml"
fi

# ── 7. Add huggingface_hub to pyproject.toml if missing ──
if ! grep -q "huggingface.hub\|huggingface-hub" pyproject.toml; then
    sed -i '' 's/"numpy>=1.26",/"numpy>=1.26",\n    "huggingface-hub>=0.20",/' pyproject.toml
    echo "[✓] Added huggingface-hub to pyproject.toml"
else
    echo "[✓] huggingface-hub already in pyproject.toml"
fi

# ── Summary ──
echo ""
echo "All missing files created. Run:"
echo "  pip install -e '.[dev]'   # pick up new deps"
echo "  make test                 # verify everything imports"