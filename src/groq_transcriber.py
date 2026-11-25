"""
Groq transcription backend — Whisper large-v3 via Groq's hosted API.

Uploads are size-capped (25 MB on the free tier), so long lectures go through
two steps before upload:

  1. Re-encode to mono, 16 kHz, 32 kbps MP3 (~14 MB per hour — plenty for speech).
  2. If that is still over the limit, split into ~10 minute chunks, transcribe
     each one, and shift every chunk's segment timestamps by the real duration
     of all chunks before it, so the merged transcript has continuous timestamps.

All intermediate files live in a temporary directory that is always removed.
"""

import logging
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from src.config import TRANSCRIPTION_LANGUAGE
from src.transcriber import Segment, Transcriber, TranscriptionError, normalize_segments

logger = logging.getLogger("lecture_pipeline")

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # Groq free-tier limit for direct file uploads
CHUNK_SECONDS = 600                  # ~10 minutes per chunk
MAX_ATTEMPTS = 4                     # per chunk: 1 try + 3 retries
MAX_BACKOFF_SECONDS = 120


# ── Pure helpers (unit-tested directly) ─────────────────────────────────────

def compute_offsets(durations: list[float]) -> list[float]:
    """Start time of each chunk = sum of the real durations of all chunks before it."""
    offsets, total = [], 0.0
    for duration in durations:
        offsets.append(total)
        total += duration
    return offsets


def offset_segments(segments: list[Segment], offset: float) -> list[Segment]:
    """Shift chunk-relative timestamps onto the timeline of the full recording."""
    return [{**seg, "start": seg["start"] + offset, "end": seg["end"] + offset} for seg in segments]


# ── ffmpeg helpers ──────────────────────────────────────────────────────────

def _run(cmd: list[str], what: str) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise TranscriptionError(f"{cmd[0]} not found. Install ffmpeg and make sure it is on your PATH.") from e
    if result.returncode != 0:
        raise TranscriptionError(f"{what} failed: {result.stderr[-300:]}")
    return result


def _reencode_for_upload(audio_path: Path, out_path: Path) -> Path:
    """Mono, 16 kHz, 32 kbps MP3: Whisper resamples to 16 kHz anyway, so no accuracy is lost."""
    _run(
        ["ffmpeg", "-y", "-i", str(audio_path), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", str(out_path)],
        "Re-encoding audio for upload",
    )
    return out_path


def _split_into_chunks(audio_path: Path, out_dir: Path, chunk_seconds: int) -> list[Path]:
    """Split without re-encoding. Cuts land on frame boundaries, so chunk lengths vary slightly."""
    _run(
        [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-f", "segment", "-segment_time", str(chunk_seconds), "-c", "copy",
            str(out_dir / "chunk_%03d.mp3"),
        ],
        "Splitting audio into chunks",
    )
    chunks = sorted(out_dir.glob("chunk_*.mp3"))
    if not chunks:
        raise TranscriptionError("Splitting audio produced no chunks.")
    return chunks


def _probe_duration(path: Path) -> float:
    result = _run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        f"Reading duration of {path.name}",
    )
    try:
        return float(result.stdout.strip())
    except ValueError as e:
        raise TranscriptionError(f"Could not read duration of {path.name}: {result.stdout!r}") from e


def _import_groq():
    try:
        import groq
    except ImportError as e:
        raise TranscriptionError("The Groq SDK is not installed. Run: pip install groq") from e
    return groq


# ── Backend ─────────────────────────────────────────────────────────────────

class GroqTranscriber(Transcriber):
    """Transcribes through Groq's Whisper API, chunking long recordings."""

    backend_name = "groq"

    def __init__(
        self,
        api_key: str | None,
        model: str = "whisper-large-v3",
        language: str = TRANSCRIPTION_LANGUAGE,
        max_upload_bytes: int = MAX_UPLOAD_BYTES,
        chunk_seconds: int = CHUNK_SECONDS,
        max_attempts: int = MAX_ATTEMPTS,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.api_key = api_key
        self.model = model
        self.language = language
        self.max_upload_bytes = max_upload_bytes
        self.chunk_seconds = chunk_seconds
        self.max_attempts = max_attempts
        self._sleep = sleep  # injectable so tests don't actually wait

    def transcribe_segments(self, audio_path: Path) -> list[Segment]:
        groq = _import_groq()
        # SDK retries are disabled: _call_with_retry is the single retry policy,
        # so failures aren't retried by two nested loops.
        client = groq.Groq(api_key=self.api_key, max_retries=0)
        t_start = time.time()

        with tempfile.TemporaryDirectory(prefix="lp_groq_") as tmp_name:
            tmp = Path(tmp_name)
            prepared = _reencode_for_upload(audio_path, tmp / "upload.mp3")
            size = prepared.stat().st_size

            if size <= self.max_upload_bytes:
                chunks, offsets = [prepared], [0.0]
            else:
                chunk_dir = tmp / "chunks"
                chunk_dir.mkdir()
                chunks = _split_into_chunks(prepared, chunk_dir, self.chunk_seconds)
                offsets = compute_offsets([_probe_duration(c) for c in chunks])

            logger.info(
                "Groq transcription: model=%s, upload size %.1f MB, %d chunk(s)",
                self.model, size / 1e6, len(chunks),
            )

            segments: list[Segment] = []
            for i, (chunk, offset) in enumerate(zip(chunks, offsets, strict=True), start=1):
                t0 = time.time()
                chunk_segments = self._call_with_retry(groq, lambda c=chunk: self._transcribe_file(client, c))
                segments.extend(offset_segments(chunk_segments, offset))
                logger.info(
                    "Chunk %d/%d (starts at %.0fs): %d segments in %.1fs",
                    i, len(chunks), offset, len(chunk_segments), time.time() - t0,
                )

        logger.info("Groq transcription finished in %.1fs", time.time() - t_start)
        return segments

    def _transcribe_file(self, client, path: Path) -> list[Segment]:
        response = client.audio.transcriptions.create(
            file=(path.name, path.read_bytes()),
            model=self.model,
            language=self.language,
            temperature=0,
            response_format="verbose_json",
        )
        segments = normalize_segments(getattr(response, "segments", None))
        if not segments and getattr(response, "text", "").strip():
            raise TranscriptionError("Groq returned text without segments; expected verbose_json output.")
        return segments

    def _call_with_retry(self, groq, fn: Callable[[], list[Segment]]) -> list[Segment]:
        """Retry rate limits (429), 5xx and network errors with exponential backoff.

        Auth and other client errors fail immediately — retrying cannot fix them.
        """
        retryable = (groq.RateLimitError, groq.InternalServerError, groq.APIConnectionError)

        for attempt in range(1, self.max_attempts + 1):
            try:
                return fn()
            except (groq.AuthenticationError, groq.PermissionDeniedError) as e:
                raise TranscriptionError(
                    "Groq rejected the API key. Check GROQ_API_KEY (https://console.groq.com/keys)."
                ) from e
            except retryable as e:
                if attempt == self.max_attempts:
                    raise TranscriptionError(
                        f"Groq request failed after {self.max_attempts} attempts: {e}"
                    ) from e
                wait = min(max(2**attempt, _retry_after_seconds(e)), MAX_BACKOFF_SECONDS)
                logger.warning(
                    "Groq request failed (%s), retrying in %ds (attempt %d/%d)",
                    type(e).__name__, wait, attempt, self.max_attempts,
                )
                self._sleep(wait)
            except groq.APIStatusError as e:
                raise TranscriptionError(f"Groq API error ({e.status_code}): {e}") from e

        raise AssertionError("unreachable")  # pragma: no cover


def _retry_after_seconds(error: Exception) -> float:
    """Honor the server's Retry-After header on 429s when present."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        return float(headers.get("retry-after", 0))
    except (TypeError, ValueError):
        return 0.0
