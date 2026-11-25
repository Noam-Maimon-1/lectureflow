"""
Transcriber — TRANSFORM stage.

Converts audio to text. Two interchangeable backends implement the same
interface (strategy pattern), and ``get_transcriber()`` picks one from config:

  local — OpenAI Whisper running on this machine (free, offline, slower).
          Model size trade-offs:
            tiny  — fastest, least accurate, ~75 MB download
            base  — good balance for clear English audio, ~150 MB
            small — better for accented speech, ~500 MB
            medium — recommended for non-English, ~1.5 GB
            large — best accuracy, ~3 GB, needs decent GPU
  groq  — Groq's hosted Whisper API (fast, runs whisper-large-v3). See groq_transcriber.py.

Every backend returns timestamped segments; the shared base class turns them
into the transcript file, so the output format is identical by construction.
"""

import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypedDict

from src.config import TRANSCRIPTION_LANGUAGE, PipelineConfig

logger = logging.getLogger("lecture_pipeline")

# Unicode bidi control characters (LRM/RLM, embeddings/overrides, isolates, Arabic letter mark).
# The transcript must contain plain text only; editors handle RTL display themselves.
_BIDI_CONTROL_CHARS = re.compile("[؜‎‏‪-‮⁦-⁩]")


class TranscriptionError(Exception):
    """Raised when transcription fails."""


class Segment(TypedDict):
    start: float  # seconds from the start of the recording
    end: float
    text: str


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def format_transcript(segments: list[Segment]) -> str:
    """Render segments as one ``[HH:MM:SS] text`` line each, without bidi control characters."""
    lines = []
    for seg in segments:
        text = " ".join(_BIDI_CONTROL_CHARS.sub("", seg["text"]).split())
        if text:
            lines.append(f"[{format_timestamp(seg['start'])}] {text}")
    return "\n".join(lines)


def normalize_segments(raw_segments) -> list[Segment]:
    """Convert backend-specific segment dicts/objects into plain ``Segment`` dicts."""
    segments: list[Segment] = []
    for raw in raw_segments or []:
        get = raw.get if isinstance(raw, dict) else lambda key, r=raw: getattr(r, key)
        segments.append({"start": float(get("start")), "end": float(get("end")), "text": str(get("text"))})
    return segments


class Transcriber(ABC):
    """Common interface for all transcription backends.

    Subclasses implement ``transcribe_segments``; this class owns writing the
    transcript so every backend produces exactly the same file format.
    """

    backend_name: str = "unknown"

    @abstractmethod
    def transcribe_segments(self, audio_path: Path) -> list[Segment]:
        """Return timestamped segments for the whole recording. Raise TranscriptionError on failure."""

    def transcribe(self, audio_path: Path, output_dir: Path | None = None) -> tuple[str, Path]:
        """Transcribe ``audio_path`` and save ``<stem>.txt`` to ``output_dir``.

        Returns:
            Tuple of (transcript_text, path_to_saved_transcript_file).
        """
        t0 = time.time()
        segments = self.transcribe_segments(audio_path)
        transcript = format_transcript(segments)

        if not transcript:
            raise TranscriptionError("Transcription produced empty output — audio may be silent or corrupt.")

        target_dir = output_dir or audio_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = target_dir / f"{audio_path.stem}.txt"
        transcript_path.write_text(transcript + "\n", encoding="utf-8")

        logger.info(
            "Transcription complete (%s): %d segments, %d words in %.1fs -> %s",
            self.backend_name, len(segments), len(transcript.split()), time.time() - t0, transcript_path.name,
        )
        return transcript, transcript_path


class LocalWhisperTranscriber(Transcriber):
    """Runs OpenAI Whisper locally."""

    backend_name = "local"

    def __init__(self, model_name: str = "base", language: str = TRANSCRIPTION_LANGUAGE):
        self.model_name = model_name
        self.language = language

    def transcribe_segments(self, audio_path: Path) -> list[Segment]:
        logger.info("Loading Whisper model '%s' (first run downloads the model)...", self.model_name)

        try:
            import whisper
        except ImportError as e:
            raise TranscriptionError("Whisper is not installed. Run: pip install openai-whisper") from e

        try:
            model = whisper.load_model(self.model_name)
        except Exception as e:
            raise TranscriptionError(f"Failed to load Whisper model '{self.model_name}': {e}") from e

        logger.info("Transcribing %s locally — this may take a few minutes...", audio_path.name)

        try:
            result = model.transcribe(str(audio_path), language=self.language, verbose=False)
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}") from e

        return normalize_segments(result.get("segments"))


def get_transcriber(config: PipelineConfig) -> Transcriber:
    """Factory: build the transcription backend selected in config."""
    backend = config.transcriber_backend
    logger.info("Transcriber backend: %s (language: %s)", backend, config.language)

    if backend == "local":
        return LocalWhisperTranscriber(model_name=config.whisper_model, language=config.language)
    if backend == "groq":
        from src.groq_transcriber import GroqTranscriber

        return GroqTranscriber(api_key=config.groq_api_key, model=config.groq_model, language=config.language)

    raise TranscriptionError(f"Unknown transcriber backend: {backend!r}")
