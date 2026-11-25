#!/usr/bin/env python3
"""
Lecture Pipeline — CLI entry point.

Usage:
  python main.py lecture.mp4
  python main.py lecture.mp4 --skip-analysis
  python main.py lecture.m4a --instructions my_prompt.txt --whisper-model medium
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from src import pipeline
from src.config import TRANSCRIBER_BACKENDS, PipelineConfig
from src.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process a lecture recording (video or audio) into an AI-generated study guide.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (audio -> transcript -> study guide):
  python main.py lecture.mp4

  # Audio files skip the ffmpeg step:
  python main.py lecture.m4a

  # Just transcribe (no Claude API key needed):
  python main.py lecture.mp4 --skip-analysis

  # Better accuracy for non-English lectures:
  python main.py lecture.mp4 --whisper-model medium

  # Transcribe in the cloud with Groq (whisper-large-v3, needs GROQ_API_KEY):
  python main.py lecture.mp4 --transcriber groq

  # Use your own instructions for how Claude should process it:
  python main.py lecture.mp4 --instructions my_instructions.txt
        """,
    )

    parser.add_argument(
        "input_file", type=Path,
        help="Path to the lecture video or audio file (never modified or deleted)",
    )

    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="Stop after transcription (no Claude API call needed)",
    )
    parser.add_argument(
        "--instructions", type=str, default=None,
        help="Path to a custom instructions file for Claude",
    )
    parser.add_argument(
        "--transcriber", type=str, default=None,
        choices=list(TRANSCRIBER_BACKENDS),
        help="Transcription backend: local Whisper or the Groq API (default: local, or LP_TRANSCRIBER_BACKEND)",
    )
    parser.add_argument(
        "--whisper-model", type=str, default=None,
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size for the local backend (default: base)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save output files (default: ~/lecture_pipeline_output)",
    )
    parser.add_argument(
        "--no-html", action="store_true",
        help="Skip HTML rendering (only JSON and Markdown outputs are saved)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Also write logs to this file",
    )

    return parser.parse_args()


def main():
    load_dotenv()  # Load environment variables from .env file if present
    args = parse_args()

    # Setup logging first
    log_file = Path(args.log_file) if args.log_file else None
    setup_logger(level=args.log_level, log_file=log_file)

    # Build config from environment, then override with CLI flags
    config = PipelineConfig.from_env()

    if args.skip_analysis:
        config.skip_analysis = True
    if args.instructions:
        config.instructions_file = Path(args.instructions)
    if args.transcriber:
        config.transcriber_backend = args.transcriber
    if args.whisper_model:
        config.whisper_model = args.whisper_model
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    if args.no_html:
        config.skip_html = True

    # Validate before running
    errors = config.validate()
    if not args.input_file.is_file():
        errors.append(f"Input file not found: {args.input_file}")
    if errors:
        for err in errors:
            print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    # Run the pipeline
    result = pipeline.run(args.input_file, config)

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
