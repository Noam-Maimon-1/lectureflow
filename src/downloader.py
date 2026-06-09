"""
Downloader — the EXTRACT stage of the pipeline.

Responsible for pulling raw media from a source URL.
Currently handles direct URLs and yt-dlp-compatible sources.

Future extension point: a Moodle scraper module would sit alongside
this and feed URLs into the same download_video() interface.
"""

import logging
import re
import subprocess
import time
from pathlib import Path

from src.config import PipelineConfig

logger = logging.getLogger("lecture_pipeline")


class DownloadError(Exception):
    """Raised when a video cannot be downloaded after all retries."""


def download_video(url: str, config: PipelineConfig) -> Path:
    """Download a video from a URL. Tries yt-dlp first, falls back to curl.

    Args:
        url: Direct video URL or yt-dlp-compatible URL.
        config: Pipeline configuration.

    Returns:
        Path to the downloaded video file.

    Raises:
        DownloadError: If download fails after all retries.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, config.download_retries + 1):
        logger.info("Download attempt %d/%d", attempt, config.download_retries)

        try:
            path = _try_ytdlp(url, config)
            logger.info("Download complete: %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
            return path
        except Exception as e:
            logger.warning("yt-dlp failed: %s", e)

        try:
            path = _try_curl(url, config)
            logger.info("Download complete: %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
            return path
        except Exception as e:
            logger.warning("curl failed: %s", e)

        if attempt < config.download_retries:
            wait = 2 ** attempt  # exponential backoff: 2s, 4s, 8s
            logger.info("Retrying in %ds...", wait)
            time.sleep(wait)

    raise DownloadError(f"Failed to download after {config.download_retries} attempts: {url}")


def _try_ytdlp(url: str, config: PipelineConfig) -> Path:
    """Attempt download using yt-dlp."""
    output_template = str(config.output_dir / "%(title)s.%(ext)s")

    result = subprocess.run(
        [
            "yt-dlp",
            "--no-playlist",
            "--restrict-filenames",
            "--socket-timeout", str(config.download_timeout),
            "-o", output_template,
            "--print", "after_move:filepath",
            url,
        ],
        capture_output=True,
        text=True,
        timeout=config.download_timeout + 30,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr[:300])

    filepath = result.stdout.strip().split("\n")[-1]
    return Path(filepath)


def _try_curl(url: str, config: PipelineConfig) -> Path:
    """Fallback: direct download for simple media URLs."""
    # Derive a filename from the URL
    raw_name = url.split("/")[-1].split("?")[0]
    if not raw_name or "." not in raw_name:
        raw_name = "lecture_video.mp4"
    safe_name = re.sub(r"[^\w.\-]", "_", raw_name)
    output_path = config.output_dir / safe_name

    result = subprocess.run(
        [
            "curl", "-L",
            "--max-time", str(config.download_timeout),
            "--retry", "2",
            "-o", str(output_path),
            url,
        ],
        capture_output=True,
        text=True,
        timeout=config.download_timeout + 30,
    )

    if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError(f"curl exit code {result.returncode}")

    return output_path
