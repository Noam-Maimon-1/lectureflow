"""
Transcriber — TRANSFORM stage 2.

Converts audio to text using OpenAI's Whisper model (runs locally).
This is the most compute-intensive stage of the pipeline.

Model size trade-offs:
  tiny  — fastest, least accurate, ~75 MB download
  base  — good balance for clear English audio, ~150 MB
  small — better for accented speech, ~500 MB
  medium — recommended for non-English, ~1.5 GB
  large — best accuracy, ~3 GB, needs decent GPU
"""

import logging
from pathlib import Path

logger = logging.getLogger("lecture_pipeline")


class TranscriptionError(Exception):
    """Raised when transcription fails."""


def transcribe(audio_path: Path, model_name: str = "base") -> tuple[str, Path]:
    """Transcribe an audio file to text using Whisper.

    Args:
        audio_path: Path to the audio file (MP3, WAV, etc.).
        model_name: Whisper model size to use.

    Returns:
        Tuple of (transcript_text, path_to_saved_transcript_file).

    Raises:
        TranscriptionError: If transcription fails.
    """
    logger.info("Loading Whisper model '%s' (first run downloads the model)...", model_name)

    try:
        import whisper
    except ImportError:
        raise TranscriptionError(
            "Whisper is not installed. Run: pip install openai-whisper"
        )

    try:
        model = whisper.load_model(model_name)
    except Exception as e:
        raise TranscriptionError(f"Failed to load Whisper model '{model_name}': {e}")

    logger.info("Transcribing %s — this may take a few minutes...", audio_path.name)

    try:
        result = model.transcribe(str(audio_path), verbose=False)
    except Exception as e:
        raise TranscriptionError(f"Transcription failed: {e}")

    transcript = result["text"].strip()

    if not transcript:
        raise TranscriptionError("Transcription produced empty output — audio may be silent or corrupt.")

    # Save transcript alongside the audio file
    transcript_path = audio_path.with_suffix(".txt")
    transcript_path.write_text(transcript, encoding="utf-8")

    word_count = len(transcript.split())
    logger.info("Transcription complete: %d words -> %s", word_count, transcript_path.name)

    return transcript, transcript_path
