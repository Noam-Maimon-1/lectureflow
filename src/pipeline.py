"""
Pipeline orchestrator — runs the full ETL chain on a user-provided file.

This module is the "conductor" that calls each stage in order,
handles errors, tracks timing, and produces a summary. Each stage
is independent and testable on its own — this module just sequences them.

Pipeline stages:
  1. EXTRACT   — extract audio from the video     (converter)
  2. TRANSFORM — transcribe audio to text          (transcriber: local Whisper or Groq)
  3. LOAD      — send transcript to Claude API     (analyzer)
  4. RENDER    — study guide JSON to HTML          (renderer)
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.analyzer import AnalysisError, analyze
from src.config import PipelineConfig
from src.converter import ConversionError, extract_audio
from src.renderer import RenderError, render_html
from src.transcriber import TranscriptionError, get_transcriber

logger = logging.getLogger("lecture_pipeline")


@dataclass
class PipelineResult:
    """Captures the outcome of a pipeline run — useful for logging and testing."""

    success: bool = False
    input_path: Path | None = None
    audio_path: Path | None = None
    transcript_path: Path | None = None
    study_guide_path: Path | None = None
    error: str | None = None
    stage_durations: dict[str, float] = field(default_factory=dict)

    @property
    def total_duration(self) -> float:
        return sum(self.stage_durations.values())

    def summary(self) -> str:
        lines = []
        status = "SUCCESS" if self.success else "FAILED"
        lines.append(f"Pipeline {status} (total: {self.total_duration:.1f}s)")
        lines.append("")

        for stage, duration in self.stage_durations.items():
            lines.append(f"  {stage:<16} {duration:>6.1f}s")

        lines.append("")
        lines.append("Files:")
        if self.input_path:
            lines.append(f"  Input:       {self.input_path}")
        if self.audio_path:
            lines.append(f"  Audio:       {self.audio_path}")
        if self.transcript_path:
            lines.append(f"  Transcript:  {self.transcript_path}")
        if self.study_guide_path:
            lines.append(f"  Study guide: {self.study_guide_path}")

        if self.error:
            lines.append("")
            lines.append(f"Error: {self.error}")

        return "\n".join(lines)


def run(input_path: Path, config: PipelineConfig) -> PipelineResult:
    """Execute the full pipeline for a single lecture recording.

    Args:
        input_path: Path to a video or audio file on disk. It is never
            modified or deleted; all outputs go to ``config.output_dir``.
        config: Pipeline configuration.

    Returns:
        PipelineResult with paths to all outputs and timing data.
    """
    input_path = Path(input_path)
    result = PipelineResult(input_path=input_path)

    logger.info("=" * 60)
    logger.info("PIPELINE START — %s", input_path.name)
    logger.info("=" * 60)

    # ── Stage 1: Extract audio ───────────────────────────────────
    try:
        t0 = time.time()
        result.audio_path = extract_audio(input_path, config.output_dir)
        result.stage_durations["convert"] = time.time() - t0
    except ConversionError as e:
        result.error = str(e)
        logger.error("Pipeline failed at CONVERT stage: %s", e)
        return result

    # ── Stage 2: Transcribe ──────────────────────────────────────
    try:
        t0 = time.time()
        transcriber = get_transcriber(config)  # local Whisper or Groq — the pipeline doesn't care which
        transcript, result.transcript_path = transcriber.transcribe(result.audio_path, output_dir=config.output_dir)
        result.stage_durations["transcribe"] = time.time() - t0
    except TranscriptionError as e:
        result.error = str(e)
        logger.error("Pipeline failed at TRANSCRIBE stage: %s", e)
        return result

    # ── Stage 3: Analyze with Claude ─────────────────────────────
    if config.skip_analysis:
        logger.info("Skipping analysis (--skip-analysis flag set)")
    else:
        try:
            t0 = time.time()
            result.study_guide_path, parsed = analyze(
                transcript, config, output_dir=config.output_dir
            )
            result.stage_durations["analyze"] = time.time() - t0
        except AnalysisError as e:
            result.error = str(e)
            logger.error("Pipeline failed at ANALYZE stage: %s", e)
            return result

        # ── Stage 4: Render HTML ─────────────────────────────────
        if not config.skip_html:
            try:
                t0 = time.time()
                render_html(parsed, config.output_dir)
                result.stage_durations["render"] = time.time() - t0
            except RenderError as e:
                logger.warning("HTML render failed (non-fatal): %s", e)

    # ── Done ─────────────────────────────────────────────────────
    result.success = True
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("\n%s", result.summary())
    logger.info("=" * 60)

    return result
