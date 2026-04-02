"""
Heatmap overlay rendering and forensic side-by-side view assembly.

Save to: src/verifi/explainability/heatmap_renderer.py

Generates:
1. Individual heatmap overlays (per-frame, per-model)
2. Three-panel forensic views: Original | GradCAM Heatmap | DCT Spectrum
3. Confidence timeline bar image
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import structlog

from verifi.detectors.frequency import FrequencyAnalyzer
from verifi.explainability.gradcam import HeatmapResult

logger = structlog.get_logger()


def save_heatmap(result: HeatmapResult, output_dir: Path) -> Path:
    """Save a single heatmap overlay to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"heatmap_{result.model_name}_frame{result.frame_idx:04d}.png"
    path = output_dir / filename
    cv2.imwrite(str(path), result.overlay)
    return path


def create_forensic_view(
    original: np.ndarray,
    heatmap_result: HeatmapResult | None,
    frame_idx: int,
    freq_analyzer: FrequencyAnalyzer | None = None,
    target_height: int = 360,
) -> np.ndarray:
    """
    Create a three-panel forensic view:
      [Original] | [GradCAM Heatmap] | [DCT Spectrum]

    If heatmap is None, shows "No heatmap" placeholder.
    If freq_analyzer is None, shows "No DCT" placeholder.

    Args:
        original: BGR image (any size, will be resized).
        heatmap_result: GradCAM output (or None).
        frame_idx: Frame index for labeling.
        freq_analyzer: FrequencyAnalyzer instance (or None).
        target_height: Height to normalize all panels to.

    Returns:
        BGR image with three panels side by side.
    """
    # Resize original to target height, preserving aspect ratio
    h, w = original.shape[:2]
    scale = target_height / h
    target_width = int(w * scale)
    orig_resized = cv2.resize(original, (target_width, target_height))

    # Panel 1: Original with label
    panel1 = _add_label(orig_resized, "Original", frame_idx)

    # Panel 2: GradCAM heatmap
    if heatmap_result is not None:
        heatmap_resized = cv2.resize(heatmap_result.overlay, (target_width, target_height))
        label = f"GradCAM ({heatmap_result.model_name})"
        panel2 = _add_label(heatmap_resized, label)
    else:
        panel2 = _placeholder(target_width, target_height, "No heatmap")

    # Panel 3: DCT spectrum
    if freq_analyzer is not None:
        spectrum = freq_analyzer.generate_spectrum_image(original)
        spectrum_resized = cv2.resize(spectrum, (target_width, target_height))
        panel3 = _add_label(spectrum_resized, "DCT spectrum")
    else:
        panel3 = _placeholder(target_width, target_height, "No DCT")

    # Add thin separator lines
    sep = np.full((target_height + 30, 2, 3), 128, dtype=np.uint8)

    # Concatenate horizontally
    forensic = np.hstack([panel1, sep, panel2, sep, panel3])
    return forensic


def create_confidence_timeline(
    scores: list[tuple[float, float]],
    width: int = 800,
    height: int = 80,
    suspicious_threshold: float = 0.3,
    manipulated_threshold: float = 0.7,
) -> np.ndarray:
    """
    Create a visual timeline bar showing per-frame confidence scores.

    Args:
        scores: List of (timestamp_sec, ensemble_score) tuples.
        width: Image width in pixels.
        height: Image height in pixels.
        suspicious_threshold: Yellow threshold.
        manipulated_threshold: Red threshold.

    Returns:
        BGR image of the timeline.
    """
    if not scores:
        return np.zeros((height, width, 3), dtype=np.uint8)

    # Create blank image
    img = np.full((height, width, 3), 32, dtype=np.uint8)

    max_time = max(t for t, _ in scores) if scores else 1.0
    if max_time <= 0:
        max_time = 1.0

    bar_top = 25
    bar_height = height - 40
    bar_left = 10
    bar_width = width - 20

    # Draw background bar
    cv2.rectangle(img, (bar_left, bar_top), (bar_left + bar_width, bar_top + bar_height),
                  (60, 60, 60), -1)

    # Draw threshold lines
    y_suspicious = bar_top + int(bar_height * (1 - suspicious_threshold))
    y_manipulated = bar_top + int(bar_height * (1 - manipulated_threshold))
    cv2.line(img, (bar_left, y_suspicious), (bar_left + bar_width, y_suspicious),
             (0, 200, 200), 1)
    cv2.line(img, (bar_left, y_manipulated), (bar_left + bar_width, y_manipulated),
             (0, 0, 200), 1)

    # Draw score points and bars
    for timestamp, score in scores:
        x = bar_left + int((timestamp / max_time) * bar_width)
        y = bar_top + int(bar_height * (1 - score))

        # Color based on score
        if score >= manipulated_threshold:
            color = (0, 0, 220)       # red
        elif score >= suspicious_threshold:
            color = (0, 180, 220)     # yellow
        else:
            color = (0, 180, 0)       # green

        # Draw vertical bar from bottom to score level
        cv2.line(img, (x, bar_top + bar_height), (x, y), color, 2)
        cv2.circle(img, (x, y), 3, color, -1)

    # Labels
    cv2.putText(img, "0s", (bar_left, height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    cv2.putText(img, f"{max_time:.1f}s", (bar_left + bar_width - 30, height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    cv2.putText(img, "Confidence timeline", (bar_left, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return img


def save_forensic_view(
    forensic_image: np.ndarray,
    output_dir: Path,
    frame_idx: int,
) -> Path:
    """Save forensic view to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"forensic_frame{frame_idx:04d}.png"
    cv2.imwrite(str(path), forensic_image)
    return path


def save_timeline(
    timeline_image: np.ndarray,
    output_dir: Path,
) -> Path:
    """Save timeline image to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "confidence_timeline.png"
    cv2.imwrite(str(path), timeline_image)
    return path


# ── Internal helpers ──

def _add_label(
    image: np.ndarray,
    text: str,
    frame_idx: int | None = None,
) -> np.ndarray:
    """Add a label bar at the top of an image."""
    h, w = image.shape[:2]
    label_h = 30
    labeled = np.zeros((h + label_h, w, 3), dtype=np.uint8)
    labeled[:label_h] = (40, 40, 40)  # dark gray bar
    labeled[label_h:] = image

    label = text
    if frame_idx is not None:
        label = f"{text} (frame {frame_idx})"

    cv2.putText(labeled, label, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    return labeled


def _placeholder(width: int, height: int, text: str) -> np.ndarray:
    """Create a gray placeholder panel with centered text."""
    total_h = height + 30
    img = np.full((total_h, width, 3), 80, dtype=np.uint8)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    x = (width - text_size[0]) // 2
    y = (total_h + text_size[1]) // 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return img
