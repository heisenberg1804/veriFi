"""
Scene boundary detection using frame differencing.

Save to: src/verifi/sampling/scene_detector.py
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class SceneBoundary:
    """A detected scene cut point."""
    frame_idx: int
    timestamp_sec: float
    diff_score: float


@dataclass
class Scene:
    """A contiguous segment of visually consistent content."""
    scene_id: int
    start_idx: int
    end_idx: int
    start_sec: float
    end_sec: float
    duration_sec: float
    frame_count: int

    @property
    def duration_frames(self) -> int:
        return self.end_idx - self.start_idx + 1


@dataclass
class SceneAnalysis:
    """Complete scene structure of a video."""
    scenes: list[Scene]
    boundaries: list[SceneBoundary]
    total_frames: int
    fps: float

    @property
    def num_scenes(self) -> int:
        return len(self.scenes)


def detect_scenes(
    video_path: str,
    threshold: float = 30.0,
    min_scene_frames: int = 10,
    downscale: tuple[int, int] = (320, 180),
) -> SceneAnalysis:
    """
    Detect scene boundaries via absolute frame differencing.

    Algorithm:
    1. Decode frames, downscale to 320x180 grayscale.
    2. Compute mean absolute difference between consecutive frames.
    3. When diff exceeds threshold, mark a scene boundary.
    4. Merge very short scenes (< min_scene_frames) into neighbors.

    Args:
        video_path: Path to video file.
        threshold: Mean pixel difference threshold for a scene cut.
                   30.0 works well for most content. Lower = more sensitive.
        min_scene_frames: Minimum frames for a valid scene.
        downscale: Resolution for diff computation (speed optimization).

    Returns:
        SceneAnalysis with detected scenes and boundaries.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    boundaries: list[SceneBoundary] = []
    prev_gray = None
    frame_idx = 0
    diff_scores = []  # for logging

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, downscale)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            mean_diff = float(diff.mean())
            diff_scores.append(mean_diff)

            if mean_diff > threshold:
                boundaries.append(SceneBoundary(
                    frame_idx=frame_idx,
                    timestamp_sec=frame_idx / fps,
                    diff_score=mean_diff,
                ))

        prev_gray = gray
        frame_idx += 1

    cap.release()

    # ── Auto-raise threshold if too many boundaries ──
    max_boundaries = 50
    if len(boundaries) > max_boundaries:
        raised_threshold = threshold * 1.5
        logger.warning(
            "scene_threshold_too_sensitive",
            boundaries=len(boundaries),
            max_boundaries=max_boundaries,
            old_threshold=threshold,
            new_threshold=raised_threshold,
        )
        # Re-filter boundaries with the raised threshold
        boundaries = [b for b in boundaries if b.diff_score > raised_threshold]
        logger.info(
            "scene_boundaries_refiltered",
            boundaries_after=len(boundaries),
            threshold=raised_threshold,
        )

    # ── Build scenes from boundaries ──
    raw_scenes = _build_scenes(boundaries, total, fps)

    # ── Merge short scenes into neighbors ──
    scenes = _merge_short_scenes(raw_scenes, min_scene_frames, fps)

    # Re-number scene IDs after merging
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

    return SceneAnalysis(
        scenes=scenes,
        boundaries=boundaries,
        total_frames=total,
        fps=fps,
    )


def _build_scenes(
    boundaries: list[SceneBoundary],
    total_frames: int,
    fps: float,
) -> list[Scene]:
    """Convert boundary list into scene segments."""
    scenes = []
    prev_idx = 0

    for i, b in enumerate(boundaries):
        end_idx = b.frame_idx - 1
        if end_idx >= prev_idx:
            scenes.append(Scene(
                scene_id=i,
                start_idx=prev_idx,
                end_idx=end_idx,
                start_sec=prev_idx / fps,
                end_sec=end_idx / fps,
                duration_sec=(end_idx - prev_idx + 1) / fps,
                frame_count=end_idx - prev_idx + 1,
            ))
        prev_idx = b.frame_idx

    # Final scene
    if prev_idx < total_frames:
        scenes.append(Scene(
            scene_id=len(scenes),
            start_idx=prev_idx,
            end_idx=total_frames - 1,
            start_sec=prev_idx / fps,
            end_sec=(total_frames - 1) / fps,
            duration_sec=(total_frames - prev_idx) / fps,
            frame_count=total_frames - prev_idx,
        ))

    return scenes


def _merge_short_scenes(
    scenes: list[Scene],
    min_frames: int,
    fps: float,
) -> list[Scene]:
    """Merge scenes shorter than min_frames into their neighbors."""
    if len(scenes) <= 1:
        return scenes

    merged = [scenes[0]]

    for scene in scenes[1:]:
        if scene.frame_count < min_frames:
            # Merge into previous scene
            prev = merged[-1]
            merged[-1] = Scene(
                scene_id=prev.scene_id,
                start_idx=prev.start_idx,
                end_idx=scene.end_idx,
                start_sec=prev.start_sec,
                end_sec=scene.end_sec,
                duration_sec=(scene.end_idx - prev.start_idx + 1) / fps,
                frame_count=scene.end_idx - prev.start_idx + 1,
            )
        else:
            merged.append(scene)

    return merged
