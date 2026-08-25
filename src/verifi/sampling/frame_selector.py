"""
Smart frame selection: two-pass sequential decode.

Pass 1: Single sequential decode — compute scene diffs, histograms, blur scores.
        Pixel data discarded immediately. Merges with scene detection.
Select: Scene segmentation + budget-proportional greedy farthest-point +
        transition boosting + quality gate, using retained stats.
Pass 2: Second sequential decode — retain pixels only for selected indices.

Peak memory bounded by frame budget, not video length.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import structlog

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


@dataclass
class _FrameStats:
    """Lightweight per-frame stats retained from Pass 1. No pixel data."""
    frame_idx: int
    hist: np.ndarray        # 64-bin grayscale histogram (normalized)
    blur_score: float       # Laplacian variance
    scene_diff: float = 0.0 # mean abs diff to previous frame (0 for frame 0)


@dataclass
class _SceneBoundary:
    frame_idx: int
    diff_score: float


@dataclass
class _Scene:
    scene_id: int
    start_idx: int
    end_idx: int
    duration_sec: float
    frame_count: int


@dataclass
class _Pass1Result:
    """All stats from the single sequential decode."""
    frame_stats: dict[int, _FrameStats]  # frame_idx -> stats
    boundaries: list[_SceneBoundary]
    scenes: list[_Scene]
    total_frames: int
    fps: float


def _pass1_sequential_decode(
    video_path: str,
    scene_threshold: float,
    downscale: tuple[int, int] = (320, 180),
    min_scene_frames: int = 10,
) -> _Pass1Result:
    """
    Single sequential decode. Per frame, compute and retain ONLY:
    - 64-bin grayscale histogram (at downscale resolution, matching original code)
    - Laplacian variance (blur score)
    - Scene diff value
    Pixel data discarded immediately.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_stats: dict[int, _FrameStats] = {}
    boundaries: list[_SceneBoundary] = []
    diff_scores: list[float] = []
    prev_gray_small = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Histogram at downscaled resolution (matches original frame_selector code
        # which computed histograms on full-res gray with 64 bins)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Scene diff at downscaled resolution (matches scene_detector.py)
        gray_small = cv2.resize(gray, downscale)
        scene_diff = 0.0
        if prev_gray_small is not None:
            diff = cv2.absdiff(prev_gray_small, gray_small)
            scene_diff = float(diff.mean())
            diff_scores.append(scene_diff)

        frame_stats[frame_idx] = _FrameStats(
            frame_idx=frame_idx, hist=hist, blur_score=blur, scene_diff=scene_diff,
        )

        prev_gray_small = gray_small
        frame_idx += 1

    cap.release()

    # Scene boundary detection (mirrors scene_detector.py exactly)
    for idx in range(1, frame_idx):
        if frame_stats[idx].scene_diff > scene_threshold:
            boundaries.append(_SceneBoundary(
                frame_idx=idx, diff_score=frame_stats[idx].scene_diff,
            ))

    max_boundaries = 50
    if len(boundaries) > max_boundaries:
        raised = scene_threshold * 1.5
        logger.warning(
            "scene_threshold_too_sensitive",
            boundaries=len(boundaries),
            max_boundaries=max_boundaries,
            old_threshold=scene_threshold,
            new_threshold=raised,
        )
        boundaries = [b for b in boundaries if b.diff_score > raised]
        logger.info(
            "scene_boundaries_refiltered",
            boundaries_after=len(boundaries),
            threshold=raised,
        )

    # Build scenes
    raw_scenes = _build_scenes(boundaries, total, fps)
    scenes = _merge_short_scenes(raw_scenes, min_scene_frames, fps)
    for i, s in enumerate(scenes):
        s.scene_id = i

    logger.info(
        "scenes_detected",
        total_frames=total,
        num_boundaries=len(boundaries),
        num_scenes=len(scenes),
        avg_diff=f"{np.mean(diff_scores):.1f}" if diff_scores else "N/A",
        max_diff=f"{np.max(diff_scores):.1f}" if diff_scores else "N/A",
    )

    return _Pass1Result(
        frame_stats=frame_stats,
        boundaries=boundaries,
        scenes=scenes,
        total_frames=total,
        fps=fps,
    )


@dataclass
class SelectionResult:
    """Result of frame index selection, including quality metadata."""
    indices: list[tuple[int, int, str]]  # (frame_idx, scene_id, reason)
    low_sharpness_fallback: bool = False


def _select_indices(
    p1: _Pass1Result,
    frame_budget: int,
    transition_margin: int,
    min_laplacian_var: float,
) -> SelectionResult:
    """
    Select frame indices from Pass 1 stats. Returns SelectionResult with
    list of (frame_idx, scene_id, selection_reason) and quality flags.

    If fewer than frame_budget frames pass the blur gate, falls back to
    the sharpest available frames rather than returning empty.
    """
    scenes = p1.scenes
    boundaries = p1.boundaries

    if not scenes:
        return SelectionResult(indices=[])

    transition_count = len(boundaries) * (2 * transition_margin + 1)
    key_budget = max(frame_budget - transition_count, 5)
    total_duration = sum(s.duration_sec for s in scenes)

    if total_duration == 0:
        return SelectionResult(indices=[])

    selected: list[tuple[int, int, str]] = []
    selected_indices: set[int] = set()

    # Pass 1+2: Budget-proportional diverse key frames using retained histograms
    for scene in scenes:
        scene_budget = max(1, round(key_budget * scene.duration_sec / total_duration))
        scene_range = range(scene.start_idx, scene.end_idx + 1)

        if len(scene_range) <= scene_budget:
            candidate_indices = list(scene_range)
        else:
            oversample = min(len(scene_range), scene_budget * 4)
            step = max(1, len(scene_range) // oversample)
            candidate_indices = list(scene_range[::step])

        # Build candidates from retained stats (no pixel data needed)
        candidates = []
        for idx in candidate_indices:
            if idx in p1.frame_stats:
                fs = p1.frame_stats[idx]
                candidates.append((idx, fs.hist))

        if not candidates:
            continue

        chosen = _greedy_diverse_select(candidates, scene_budget)

        for idx, _ in chosen:
            fs = p1.frame_stats[idx]
            selected.append((idx, scene.scene_id, "key_frame"))
            selected_indices.add(idx)

    # Transition boosting
    for boundary in boundaries:
        for offset in range(-transition_margin, transition_margin + 1):
            t_idx = boundary.frame_idx + offset
            if t_idx < 0 or t_idx >= p1.total_frames:
                continue
            if t_idx in selected_indices:
                continue
            if t_idx not in p1.frame_stats:
                continue
            selected.append((t_idx, -1, "transition"))
            selected_indices.add(t_idx)

    # Quality gate using retained blur scores
    before_filter = len(selected)
    filtered = [
        (idx, sid, reason) for idx, sid, reason in selected
        if p1.frame_stats[idx].blur_score >= min_laplacian_var
    ]
    rejected = before_filter - len(filtered)
    low_sharpness_fallback = False

    # Fallback: if too few frames pass the gate, take the sharpest available
    if len(filtered) < max(frame_budget, 5) and before_filter > 0:
        selected.sort(
            key=lambda x: p1.frame_stats[x[0]].blur_score, reverse=True,
        )
        filtered = selected[:frame_budget]
        low_sharpness_fallback = True
        logger.warning(
            "blur_quality_fallback",
            passed_gate=before_filter - rejected,
            total_candidates=before_filter,
            threshold=min_laplacian_var,
            fallback_count=len(filtered),
            sharpest=round(p1.frame_stats[filtered[0][0]].blur_score, 1)
            if filtered else 0,
        )

    # Hard cap
    hard_cap = frame_budget * 2
    if len(filtered) > hard_cap:
        logger.warning(
            "frame_hard_cap_triggered",
            before_cap=len(filtered),
            hard_cap=hard_cap,
            budget=frame_budget,
        )
        filtered.sort(key=lambda x: p1.frame_stats[x[0]].blur_score, reverse=True)
        filtered = filtered[:hard_cap]

    filtered.sort(key=lambda x: x[0])

    logger.info(
        "frames_selected",
        total_selected=len(filtered),
        key_frames=sum(1 for _, _, r in filtered if r == "key_frame"),
        transition_frames=sum(1 for _, _, r in filtered if r == "transition"),
        blur_rejected=rejected,
        low_sharpness_fallback=low_sharpness_fallback,
        budget=frame_budget,
    )

    return SelectionResult(
        indices=filtered,
        low_sharpness_fallback=low_sharpness_fallback,
    )


def _pass2_load_pixels(
    video_path: str,
    selected: list[tuple[int, int, str]],
    p1: _Pass1Result,
) -> list[SelectedFrame]:
    """
    Second sequential decode. Retain full frames ONLY for selected indices.
    """
    if not selected:
        return []

    target_set = {idx for idx, _, _ in selected}
    max_target = max(target_set)
    loaded: dict[int, np.ndarray] = {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_idx = 0
    while frame_idx <= max_target:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in target_set:
            loaded[frame_idx] = frame
        frame_idx += 1

    cap.release()

    frames = []
    for idx, scene_id, reason in selected:
        if idx not in loaded:
            continue
        fs = p1.frame_stats[idx]
        frames.append(SelectedFrame(
            frame_idx=idx,
            timestamp_sec=idx / p1.fps,
            scene_id=scene_id,
            selection_reason=reason,
            image=loaded[idx],
            blur_score=fs.blur_score,
        ))

    return frames


def select_frames(
    video_path: str,
    scene_analysis=None,
    frame_budget: int = 30,
    transition_margin: int = 2,
    min_laplacian_var: float = 100.0,
    scene_threshold: float = 30.0,
    _pass1_result: _Pass1Result | None = None,
) -> list[SelectedFrame]:
    """
    Two-pass smart frame selection.

    If _pass1_result is provided, skips Pass 1 (used when the pipeline
    already ran the combined scene+stats decode).

    The scene_analysis parameter is accepted for backwards compatibility
    but ignored when _pass1_result is provided.
    """
    if _pass1_result is None:
        _pass1_result = _pass1_sequential_decode(video_path, scene_threshold)

    result = _select_indices(
        _pass1_result, frame_budget, transition_margin, min_laplacian_var,
    )

    return _pass2_load_pixels(video_path, result.indices, _pass1_result)


def _greedy_diverse_select(
    candidates: list[tuple[int, np.ndarray]],
    k: int,
) -> list[tuple[int, np.ndarray]]:
    """
    Greedy farthest-point sampling by histogram distance.
    Ensures selected frames are maximally visually diverse.
    """
    if len(candidates) <= k:
        return candidates

    chosen = [candidates[0]]
    remaining = list(candidates[1:])

    while len(chosen) < k and remaining:
        best_idx = -1
        best_dist = -1.0

        for i, (_, hist_r) in enumerate(remaining):
            min_dist = min(
                cv2.compareHist(hist_r, c[1], cv2.HISTCMP_BHATTACHARYYA)
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


def _build_scenes(
    boundaries: list[_SceneBoundary],
    total_frames: int,
    fps: float,
) -> list[_Scene]:
    """Convert boundary list into scene segments."""
    scenes = []
    prev_idx = 0

    for i, b in enumerate(boundaries):
        end_idx = b.frame_idx - 1
        if end_idx >= prev_idx:
            scenes.append(_Scene(
                scene_id=i,
                start_idx=prev_idx,
                end_idx=end_idx,
                duration_sec=(end_idx - prev_idx + 1) / fps,
                frame_count=end_idx - prev_idx + 1,
            ))
        prev_idx = b.frame_idx

    if prev_idx < total_frames:
        scenes.append(_Scene(
            scene_id=len(scenes),
            start_idx=prev_idx,
            end_idx=total_frames - 1,
            duration_sec=(total_frames - prev_idx) / fps,
            frame_count=total_frames - prev_idx,
        ))

    return scenes


def _merge_short_scenes(
    scenes: list[_Scene],
    min_frames: int,
    fps: float,
) -> list[_Scene]:
    """Merge scenes shorter than min_frames into their neighbors."""
    if len(scenes) <= 1:
        return scenes

    merged = [scenes[0]]

    for scene in scenes[1:]:
        if scene.frame_count < min_frames:
            prev = merged[-1]
            merged[-1] = _Scene(
                scene_id=prev.scene_id,
                start_idx=prev.start_idx,
                end_idx=scene.end_idx,
                duration_sec=(scene.end_idx - prev.start_idx + 1) / fps,
                frame_count=scene.end_idx - prev.start_idx + 1,
            )
        else:
            merged.append(scene)

    return merged


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
