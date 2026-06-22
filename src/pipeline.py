"""
Pipeline orchestrator — runs the full ETL chain.

This module is the "conductor" that calls each stage in order,
handles errors, tracks timing, and produces a summary. Each stage
is independent and testable on its own — this module just sequences them.

Pipeline stages:
  1. EXTRACT   — download video from URL        (downloader)
  2. TRANSFORM — convert video to audio          (converter)
  3. TRANSFORM — transcribe audio to text        (transcriber)
  4. LOAD      — send transcript to Claude API   (analyzer)
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.config import PipelineConfig
from src.downloader import download_video, DownloadError
from src.converter import extract_audio, ConversionError
from src.transcriber import transcribe, TranscriptionError
from src.analyzer import analyze, AnalysisError
from src.renderer import render_html, RenderError

logger = logging.getLogger("lecture_pipeline")


@dataclass
class PipelineResult:
    """Captures the outcome of a pipeline run — useful for logging and testing."""

    success: bool = False
    video_path: Path | None = None
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
        lines.append("Output files:")
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


def run(url: str, config: PipelineConfig) -> PipelineResult:
    """Execute the full pipeline for a single lecture URL.

    Args:
        url: Video URL to process.
        config: Pipeline configuration.

    Returns:
        PipelineResult with paths to all outputs and timing data.
    """
    result = PipelineResult()

    logger.info("=" * 60)
    logger.info("PIPELINE START — %s", url[:80])
    logger.info("=" * 60)

    # ── Stage 1: Download (Extract) ──────────────────────────────
    try:
        t0 = time.time()
        result.video_path = download_video(url, config)
        result.stage_durations["download"] = time.time() - t0
    except DownloadError as e:
        result.error = str(e)
        logger.error("Pipeline failed at DOWNLOAD stage: %s", e)
        return result

    # ── Stage 2: Convert to audio (Transform 1) ─────────────────
    try:
        t0 = time.time()
        result.audio_path = extract_audio(result.video_path, keep_video=config.keep_video)
        result.stage_durations["convert"] = time.time() - t0
    except ConversionError as e:
        result.error = str(e)
        logger.error("Pipeline failed at CONVERT stage: %s", e)
        return result

    # ── Stage 3: Transcribe (Transform 2) ────────────────────────
    try:
        t0 = time.time()
        transcript, result.transcript_path = transcribe(
            result.audio_path, model_name=config.whisper_model
        )
        result.stage_durations["transcribe"] = time.time() - t0
    except TranscriptionError as e:
        result.error = str(e)
        logger.error("Pipeline failed at TRANSCRIBE stage: %s", e)
        return result

    # ── Stage 4: Analyze with Claude (Load) ──────────────────────
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

        # ── Stage 5: Render HTML ──────────────────────────────────
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
