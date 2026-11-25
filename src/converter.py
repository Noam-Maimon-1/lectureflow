"""
Converter — EXTRACT stage.

Extracts the audio track from a lecture video using ffmpeg.
This is a "lossy reduction" step: we discard the video stream
because downstream stages (transcription) only need audio.

The user's input file is never modified or deleted.
"""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("lecture_pipeline")

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}


class ConversionError(Exception):
    """Raised when audio extraction fails."""


def extract_audio(input_path: Path, output_dir: Path) -> Path:
    """Convert a video file to MP3 audio in ``output_dir``.

    If the input is already an audio file, ffmpeg is skipped and the
    input path is returned as-is.

    Args:
        input_path: Path to the user's video (or audio) file.
        output_dir: Directory where the extracted MP3 is written.

    Returns:
        Path to the MP3 file (or the original file, if it was already audio).

    Raises:
        ConversionError: If the input does not exist or ffmpeg fails.
    """
    if not input_path.is_file():
        raise ConversionError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() in AUDIO_EXTENSIONS:
        logger.info("Input is already an audio file, skipping ffmpeg: %s", input_path.name)
        return input_path

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / f"{input_path.stem}.mp3"

    logger.info("Extracting audio: %s -> %s", input_path.name, audio_path)

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i", str(input_path),
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
    except FileNotFoundError as e:
        raise ConversionError("ffmpeg not found. Install it and make sure it is on your PATH.") from e

    if result.returncode != 0:
        raise ConversionError(f"ffmpeg failed: {result.stderr[-300:]}")

    size_mb = audio_path.stat().st_size / 1e6
    logger.info("Audio extracted: %s (%.1f MB)", audio_path.name, size_mb)

    return audio_path
