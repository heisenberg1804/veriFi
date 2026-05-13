"""Video URL downloader using yt-dlp."""
from __future__ import annotations

import subprocess
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
