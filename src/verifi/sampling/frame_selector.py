"""
Smart frame selection: budget-proportional diversity + transition boosting.

Save to: src/verifi/sampling/frame_selector.py
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import structlog

from verifi.sampling.scene_detector import SceneAnalysis

logger = structlog.get_logger()


@dataclass
class SelectedFrame:
    """A frame selected for analysis."""
    frame_idx: int
    timestamp_sec: float
    scene_id: int
    selection_reason: str   # "key_frame" | "transition"
    image: np.ndarray       # BGR numpy array (original resolution)
    blur_score: float = 0.0 # Laplacian variance (higher = sharper)


def select_frames(
    video_path: str,
    scene_analysis: SceneAnalysis,
    frame_budget: int = 30,
    transition_margin: int = 2,
    min_laplacian_var: float = 100.0,
) -> list[SelectedFrame]:
    """
    Three-pass smart frame selection.

    Pass 1: Allocate frame budget proportionally to scene duration.
    Pass 2: Within each scene, select maximally diverse frames
            using histogram distance (greedy farthest-point).
    Pass 3: Add ±N frames around each scene boundary.
    Quality gate: Reject blurry frames (Laplacian variance < threshold).

    Args:
        video_path: Path to video file.
        scene_analysis: Output from scene detection.
        frame_budget: Target number of frames to select.
        transition_margin: ±N frames around each scene cut.
        min_laplacian_var: Minimum blur score to keep a frame.

    Returns:
        Sorted list of SelectedFrame objects.
    """
    scenes = scene_analysis.scenes
    boundaries = scene_analysis.boundaries
    fps = scene_analysis.fps

    if not scenes:
        logger.warn("no_scenes_detected")
        return []

    # ── Calculate budget allocation ──
    transition_count = len(boundaries) * (2 * transition_margin + 1)
    key_budget = max(frame_budget - transition_count, 5)
    total_duration = sum(s.duration_sec for s in scenes)

    if total_duration == 0:
        logger.warn("zero_duration_video")
        return []

    # ── Pass 1+2: Budget-proportional diverse key frames ──
    selected: list[SelectedFrame] = []
    selected_indices: set[int] = set()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    for scene in scenes:
        # Proportional budget for this scene
        scene_budget = max(1, round(key_budget * scene.duration_sec / total_duration))

        # Generate candidate frame indices (3x oversample for diversity selection)
        scene_range = range(scene.start_idx, scene.end_idx + 1)
        if len(scene_range) <= scene_budget:
            candidate_indices = list(scene_range)
        else:
            oversample = min(len(scene_range), scene_budget * 4)
            step = max(1, len(scene_range) // oversample)
            candidate_indices = list(scene_range[::step])

        # Load candidate frames and compute histograms
        candidates = []
        for idx in candidate_indices:
            frame = _read_frame(cap, idx)
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            candidates.append((idx, frame, hist))

        if not candidates:
            continue

        # Greedy max-diversity selection (farthest-point sampling)
        chosen = _greedy_diverse_select(candidates, scene_budget)

        for idx, frame, _ in chosen:
            blur = _compute_blur_score(frame)
            selected.append(SelectedFrame(
                frame_idx=idx,
                timestamp_sec=idx / fps,
                scene_id=scene.scene_id,
                selection_reason="key_frame",
                image=frame,
                blur_score=blur,
            ))
            selected_indices.add(idx)

    # ── Pass 3: Transition boosting ──
    for boundary in boundaries:
        for offset in range(-transition_margin, transition_margin + 1):
            t_idx = boundary.frame_idx + offset
            if t_idx < 0 or t_idx >= scene_analysis.total_frames:
                continue
            if t_idx in selected_indices:
                continue

            frame = _read_frame(cap, t_idx)
            if frame is None:
                continue

            blur = _compute_blur_score(frame)
            selected.append(SelectedFrame(
                frame_idx=t_idx,
                timestamp_sec=t_idx / fps,
                scene_id=-1,  # transition frame
                selection_reason="transition",
                image=frame,
                blur_score=blur,
            ))
            selected_indices.add(t_idx)

    cap.release()

    # ── Quality gate: reject blurry frames ──
    before_filter = len(selected)
    filtered = [f for f in selected if f.blur_score >= min_laplacian_var]
    rejected = before_filter - len(filtered)

    # ── Hard cap: prevent runaway frame counts ──
    hard_cap = frame_budget * 2
    if len(filtered) > hard_cap:
        logger.warning(
            "frame_hard_cap_triggered",
            before_cap=len(filtered),
            hard_cap=hard_cap,
            budget=frame_budget,
        )
        # Keep the sharpest frames (highest blur_score)
        filtered.sort(key=lambda f: f.blur_score, reverse=True)
        filtered = filtered[:hard_cap]

    # Sort by frame index (chronological order)
    filtered.sort(key=lambda f: f.frame_idx)

    logger.info(
        "frames_selected",
        total_selected=len(filtered),
        key_frames=sum(1 for f in filtered if f.selection_reason == "key_frame"),
        transition_frames=sum(1 for f in filtered if f.selection_reason == "transition"),
        blur_rejected=rejected,
        budget=frame_budget,
    )

    return filtered


def _read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    """Read a specific frame by index."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    return frame if ret else None


def _compute_blur_score(frame: np.ndarray) -> float:
    """Compute Laplacian variance as sharpness metric."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _greedy_diverse_select(
    candidates: list[tuple[int, np.ndarray, np.ndarray]],
    k: int,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """
    Greedy farthest-point sampling by histogram distance.
    Ensures selected frames are maximally visually diverse.
    """
    if len(candidates) <= k:
        return candidates

    # Start with first candidate
    chosen = [candidates[0]]
    remaining = list(candidates[1:])

    while len(chosen) < k and remaining:
        # Find candidate with max min-distance to all chosen
        best_idx = -1
        best_dist = -1.0

        for i, (_, _, hist_r) in enumerate(remaining):
            min_dist = min(
                cv2.compareHist(hist_r, c[2], cv2.HISTCMP_BHATTACHARYYA)
                for c in chosen
            )
            if min_dist > best_dist:
                best_dist = min_dist
                best_idx = i

        if best_idx >= 0:
            chosen.append(remaining.pop(best_idx))
        else:
            break

    return chosen


# ─── Quality filter (standalone, can also be used independently) ───

def filter_by_quality(
    frames: list[SelectedFrame],
    min_laplacian_var: float = 100.0,
) -> list[SelectedFrame]:
    """
    Post-hoc quality filter. Use when you need to re-filter
    with a different threshold.
    """
    return [f for f in frames if f.blur_score >= min_laplacian_var]
