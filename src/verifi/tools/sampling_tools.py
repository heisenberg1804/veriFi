"""
Sampling and preprocessing tools.

Save to: src/verifi/tools/sampling_tools.py

These give the agent control over WHAT to analyze:
- Sample more frames from a specific time range
- Zoom into a region at higher resolution
- Detect faces in a specific frame
"""
from __future__ import annotations

import cv2
import numpy as np

from verifi.preprocessing.face_detector import FaceDetectionPipeline
from verifi.tools.base import Tool, ToolResult


class ZoomRegionTool(Tool):
    """Crop and upscale a specific region of a frame."""

    @property
    def name(self) -> str:
        return "zoom_region"

    @property
    def description(self) -> str:
        return (
            "Crop a specific rectangular region from a frame and upscale it "
            "to a target size. Use this to get a closer look at a suspicious "
            "area — for example, zooming into a jawline to check for blending "
            "artifacts, or isolating a subject from its background. "
            "The zoomed crop can then be passed to other detection tools."
        )

    @property
    def parameters(self) -> dict:
        return {
            "image_key": {
                "type": "string",
                "description": "Key to retrieve the source frame",
            },
            "x": {"type": "integer", "description": "Left edge of crop region"},
            "y": {"type": "integer", "description": "Top edge of crop region"},
            "width": {"type": "integer", "description": "Width of crop region"},
            "height": {"type": "integer", "description": "Height of crop region"},
            "target_size": {
                "type": "integer",
                "description": "Target size for the output (square). Default 224.",
                "default": 224,
            },
        }

    def execute(
        self,
        image: np.ndarray | None = None,
        x: int = 0,
        y: int = 0,
        width: int = 224,
        height: int = 224,
        target_size: int = 224,
        **kwargs,
    ) -> ToolResult:
        if image is None:
            return ToolResult(
                tool_name=self.name, success=False,
                error="No image provided",
            )

        img_h, img_w = image.shape[:2]

        # Clamp to image bounds
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        width = min(width, img_w - x)
        height = min(height, img_h - y)

        if width < 10 or height < 10:
            return ToolResult(
                tool_name=self.name, success=False,
                error=f"Region too small: {width}x{height}",
            )

        crop = image[y:y+height, x:x+width]
        zoomed = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "image": zoomed,
                "original_region": {"x": x, "y": y, "w": width, "h": height},
                "scale_factor": round(target_size / max(width, height), 2),
                "summary": (
                    f"Zoomed into region ({x},{y},{width},{height})"
                    f" → {target_size}x{target_size}"
                ),
            },
        )


class SampleFramesTool(Tool):
    """Extract additional frames from a specific time range."""

    @property
    def name(self) -> str:
        return "sample_more_frames"

    @property
    def description(self) -> str:
        return (
            "Extract additional frames from a specific time range of the video. "
            "Use this when initial analysis flagged a specific segment as suspicious "
            "and you want denser sampling in that region. Returns a list of frames "
            "with their timestamps."
        )

    @property
    def parameters(self) -> dict:
        return {
            "start_sec": {
                "type": "number",
                "description": "Start time in seconds",
            },
            "end_sec": {
                "type": "number",
                "description": "End time in seconds",
            },
            "num_frames": {
                "type": "integer",
                "description": "Number of frames to extract. Default 5.",
                "default": 5,
            },
        }

    def execute(
        self,
        video_path: str | None = None,
        start_sec: float = 0.0,
        end_sec: float = 1.0,
        num_frames: int = 5,
        fps: float = 30.0,
        **kwargs,
    ) -> ToolResult:
        if video_path is None:
            return ToolResult(
                tool_name=self.name, success=False,
                error="No video path provided",
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return ToolResult(
                tool_name=self.name, success=False,
                error=f"Cannot open video: {video_path}",
            )

        video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        start_frame = int(start_sec * video_fps)
        end_frame = int(end_sec * video_fps)
        total_range = max(end_frame - start_frame, 1)
        step = max(total_range // num_frames, 1)

        frames = []
        for i in range(num_frames):
            frame_idx = start_frame + i * step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames.append({
                    "frame_idx": frame_idx,
                    "timestamp_sec": round(frame_idx / video_fps, 3),
                    "image": frame,
                })

        cap.release()

        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "frames": frames,
                "num_extracted": len(frames),
                "time_range": f"{start_sec:.1f}s - {end_sec:.1f}s",
                "summary": (
                    f"Extracted {len(frames)} frames"
                    f" from {start_sec:.1f}s to {end_sec:.1f}s"
                ),
            },
        )


class FaceDetectionTool(Tool):
    """Run face detection on a specific frame."""

    def __init__(self, pipeline: FaceDetectionPipeline):
        self._pipeline = pipeline

    @property
    def name(self) -> str:
        return "detect_faces"

    @property
    def description(self) -> str:
        return (
            "Run MTCNN face detection on a frame. Returns bounding boxes, "
            "confidence scores, and aligned face crops for all detected faces. "
            "Note: only detects HUMAN faces. Will not detect animals, objects, "
            "or non-human subjects. Use this when you want to check if a "
            "specific frame contains faces that weren't caught in the initial scan."
        )

    @property
    def parameters(self) -> dict:
        return {
            "image_key": {
                "type": "string",
                "description": "Key to retrieve the frame to analyze",
            },
        }

    def execute(
        self,
        image: np.ndarray | None = None,
        frame_idx: int = 0,
        **kwargs,
    ) -> ToolResult:
        if image is None:
            return ToolResult(
                tool_name=self.name, success=False,
                error="No image provided",
            )

        result = self._pipeline.detect_frame(image, frame_idx)

        faces_data = []
        for face in result.faces:
            faces_data.append({
                "face_id": face.face_id,
                "bbox": face.bbox.to_tuple(),
                "confidence": round(face.confidence, 3),
                "crop": face.crop,
                "crop_raw_size": f"{face.crop_raw.shape[1]}x{face.crop_raw.shape[0]}",
            })

        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "num_faces": result.num_faces,
                "faces": faces_data,
                "summary": f"Detected {result.num_faces} face(s)" + (
                    f" — IDs: {[f['face_id'] for f in faces_data]}" if faces_data else ""
                ),
            },
        )


class CheckMetadataTool(Tool):
    """Extract and analyze video file metadata."""

    @property
    def name(self) -> str:
        return "check_metadata"

    @property
    def description(self) -> str:
        return (
            "Extract detailed metadata from the video file using ffprobe. "
            "This reveals codec information, creation date, software used, "
            "and sometimes the generation tool (e.g., Veo watermark, RunwayML "
            "metadata). Metadata can also reveal re-encoding which may indicate "
            "the video was processed after creation."
        )

    @property
    def parameters(self) -> dict:
        return {}  # No parameters — uses the video already in context

    def execute(self, video_path: str | None = None, **kwargs) -> ToolResult:
        if video_path is None:
            return ToolResult(
                tool_name=self.name, success=False,
                error="No video path provided",
            )

        from pathlib import Path

        from verifi.ingestion.validator import probe_video

        try:
            probe = probe_video(Path(video_path))
        except Exception as e:
            return ToolResult(
                tool_name=self.name, success=False,
                error=f"ffprobe failed: {e}",
            )

        fmt = probe.get("format", {})
        streams = probe.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})

        # Look for generator signatures in metadata tags
        tags = fmt.get("tags", {})
        video_tags = video_stream.get("tags", {})
        all_tags = {**tags, **video_tags}

        # Common AI generator signatures
        suspicious_tags = {}
        for key, value in all_tags.items():
            key_lower = key.lower()
            gen_keywords = [
                "encoder", "handler", "software",
                "creator", "comment",
            ]
            if any(kw in key_lower for kw in gen_keywords):
                suspicious_tags[key] = value

        summary_parts = [
            f"Codec: {video_stream.get('codec_name', 'unknown')}",
            f"Profile: {video_stream.get('profile', 'unknown')}",
        ]
        if suspicious_tags:
            summary_parts.append(f"Tags: {suspicious_tags}")

        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "format": fmt.get("format_name", "unknown"),
                "codec": video_stream.get("codec_name", "unknown"),
                "profile": video_stream.get("profile", "unknown"),
                "pixel_format": video_stream.get("pix_fmt", "unknown"),
                "creation_time": all_tags.get("creation_time", "unknown"),
                "encoder": all_tags.get("encoder", "unknown"),
                "all_tags": all_tags,
                "suspicious_tags": suspicious_tags,
                "summary": " | ".join(summary_parts),
            },
        )
