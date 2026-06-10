#!/usr/bin/env python3
"""
Lecture Pipeline — CLI entry point.

Usage:
  python main.py "https://example.com/lecture-video.mp4"
  python main.py "URL" --skip-analysis
  python main.py "URL" --instructions my_prompt.txt --whisper-model medium
"""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if present

from src.config import PipelineConfig
from src.logger import setup_logger
from src import pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process a lecture video into an AI-generated study guide.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (download → audio → transcribe → study guide):
  python main.py "https://moodle.uni.edu/media/lecture.mp4"

  # Just download + transcribe (no Claude API key needed):
  python main.py "URL" --skip-analysis

  # Better accuracy for non-English lectures:
  python main.py "URL" --whisper-model medium

  # Use your own instructions for how Claude should process it:
  python main.py "URL" --instructions my_instructions.txt
        """,
    )

    parser.add_argument("url", help="URL of the lecture video to process")

    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="Stop after transcription (no Claude API call needed)",
    )
    parser.add_argument(
        "--instructions", type=str, default=None,
        help="Path to a custom instructions file for Claude",
    )
    parser.add_argument(
        "--whisper-model", type=str, default=None,
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save output files (default: ~/lecture_pipeline_output)",
    )
    parser.add_argument(
        "--keep-video", action="store_true",
        help="Keep the video file after converting to audio",
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
    if args.whisper_model:
        config.whisper_model = args.whisper_model
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    if args.keep_video:
        config.keep_video = True

    # Validate before running
    errors = config.validate()
    if errors:
        for err in errors:
            print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    # Run the pipeline
    result = pipeline.run(args.url, config)

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
