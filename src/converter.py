"""
Converter — TRANSFORM stage 1.

Extracts audio from a video file using ffmpeg.
This is a "lossy reduction" transform: we discard the video stream
because downstream stages (transcription) only need audio.
"""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("lecture_pipeline")


class ConversionError(Exception):
    """Raised when audio extraction fails."""


def extract_audio(video_path: Path, keep_video: bool = False) -> Path:
    """Convert a video file to MP3 audio.

    Args:
        video_path: Path to the input video file.
        keep_video: If False, deletes the video after conversion to save disk.

    Returns:
        Path to the output MP3 file.

    Raises:
        ConversionError: If ffmpeg fails.
    """
    audio_path = video_path.with_suffix(".mp3")

    logger.info("Extracting audio: %s -> %s", video_path.name, audio_path.name)

    result = subprocess.run(
        [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",                  # discard video stream
            "-acodec", "libmp3lame",
            "-ab", "128k",         # 128kbps is fine for speech
            "-ar", "44100",        # standard sample rate
            "-y",                  # overwrite if exists (idempotency)
            str(audio_path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise ConversionError(f"ffmpeg failed: {result.stderr[:300]}")

    size_mb = audio_path.stat().st_size / 1e6
    logger.info("Audio extracted: %s (%.1f MB)", audio_path.name, size_mb)

    if not keep_video and video_path.exists():
        video_path.unlink()
        logger.debug("Removed source video: %s", video_path.name)

    return audio_path
