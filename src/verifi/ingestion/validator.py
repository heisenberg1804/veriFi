"""
Video ingestion, validation, and metadata extraction.

Save to: src/verifi/ingestion/validator.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import structlog

logger = structlog.get_logger()

# Supported video formats
SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}


@dataclass
class VideoMetadata:
    """Extracted metadata from a video file."""
    path: str
    filename: str
    file_hash: str              # SHA256 of first 10MB (for dedup/caching)
    duration_sec: float
    width: int
    height: int
    fps: float
    total_frames: int
    codec: str
    has_audio: bool
    file_size_mb: float
    resolution: str = ""        # e.g., "1280x720"

    def __post_init__(self):
        self.resolution = f"{self.width}x{self.height}"

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "filename": self.filename,
            "file_hash": self.file_hash,
            "duration_sec": round(self.duration_sec, 2),
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "fps": round(self.fps, 2),
            "total_frames": self.total_frames,
            "codec": self.codec,
            "has_audio": self.has_audio,
            "file_size_mb": round(self.file_size_mb, 2),
        }


@dataclass
class ValidationResult:
    """Result of video validation."""
    valid: bool
    metadata: VideoMetadata | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def compute_file_hash(path: Path, chunk_size: int = 10 * 1024 * 1024) -> str:
    """SHA256 hash of first 10MB of file (fast for large videos)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        data = f.read(chunk_size)
        h.update(data)
    return f"sha256:{h.hexdigest()[:16]}"


def probe_video(path: Path) -> dict:
    """
    Extract video metadata using ffprobe.
    Returns raw ffprobe JSON output.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr[:200]}")
        return json.loads(result.stdout)
    except FileNotFoundError:
        raise RuntimeError(
            "ffprobe not found. Install ffmpeg: brew install ffmpeg"
        )


def validate_video(
    path: str | Path,
    max_duration_sec: int = 600,
) -> ValidationResult:
    """
    Validate a video file and extract metadata.

    Checks:
    - File exists and is readable
    - Supported format
    - Has video stream
    - Within duration limit
    - Minimum resolution

    Returns ValidationResult with metadata if valid.
    """
    path = Path(path)
    errors = []
    warnings = []

    # ── File checks ──
    if not path.exists():
        return ValidationResult(valid=False, errors=[f"File not found: {path}"])

    if not path.is_file():
        return ValidationResult(valid=False, errors=[f"Not a file: {path}"])

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return ValidationResult(
            valid=False,
            errors=[f"Unsupported format: {path.suffix}. Supported: {SUPPORTED_EXTENSIONS}"],
        )

    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb < 0.01:
        return ValidationResult(valid=False, errors=["File is empty or too small"])

    # ── Probe with ffprobe ──
    try:
        probe = probe_video(path)
    except RuntimeError as e:
        return ValidationResult(valid=False, errors=[str(e)])

    # ── Extract streams ──
    streams = probe.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    if not video_streams:
        return ValidationResult(valid=False, errors=["No video stream found"])

    vs = video_streams[0]
    fmt = probe.get("format", {})

    # ── Parse video properties ──
    width = int(vs.get("width", 0))
    height = int(vs.get("height", 0))

    # FPS: try r_frame_rate first, then avg_frame_rate
    fps_str = vs.get("r_frame_rate", vs.get("avg_frame_rate", "30/1"))
    try:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) > 0 else 30.0
    except (ValueError, ZeroDivisionError):
        fps = 30.0
        warnings.append(f"Could not parse FPS '{fps_str}', defaulting to 30")

    duration = float(fmt.get("duration", vs.get("duration", 0)))
    codec = vs.get("codec_name", "unknown")

    # Total frames: try nb_frames, otherwise estimate from duration
    nb_frames = vs.get("nb_frames")
    if nb_frames and nb_frames != "N/A":
        total_frames = int(nb_frames)
    else:
        total_frames = int(duration * fps)

    # ── Validation checks ──
    if duration > max_duration_sec:
        errors.append(
            f"Video too long: {duration:.1f}s (max: {max_duration_sec}s)"
        )

    if width < 128 or height < 128:
        errors.append(f"Resolution too low: {width}x{height} (min: 128x128)")

    if width < 480 or height < 360:
        warnings.append(
            f"Low resolution ({width}x{height}) — detection accuracy may be reduced"
        )

    if duration < 0.5:
        errors.append(f"Video too short: {duration:.2f}s (min: 0.5s)")

    # ── Build metadata ──
    file_hash = compute_file_hash(path)

    metadata = VideoMetadata(
        path=str(path),
        filename=path.name,
        file_hash=file_hash,
        duration_sec=duration,
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        codec=codec,
        has_audio=len(audio_streams) > 0,
        file_size_mb=file_size_mb,
    )

    valid = len(errors) == 0

    if valid:
        logger.info(
            "video_validated",
            filename=path.name,
            duration=f"{duration:.1f}s",
            resolution=metadata.resolution,
            fps=f"{fps:.1f}",
            codec=codec,
            has_audio=metadata.has_audio,
        )
    else:
        logger.warn("video_validation_failed", filename=path.name, errors=errors)

    return ValidationResult(
        valid=valid,
        metadata=metadata,
        errors=errors,
        warnings=warnings,
    )
