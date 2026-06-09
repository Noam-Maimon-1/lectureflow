"""
Analyzer — the LOAD stage of the pipeline.

Sends the processed transcript to Claude's API along with
study instructions, and saves the resulting study guide.

This is "Load" because we're delivering transformed data to
the system that will consume it (the LLM). The output is the
final deliverable: a structured study guide.
"""

import logging
from pathlib import Path

from src.config import PipelineConfig

logger = logging.getLogger("lecture_pipeline")

# Bundled with the project — user can override via --instructions
DEFAULT_INSTRUCTIONS_PATH = Path(__file__).parent.parent / "instructions" / "default.txt"


class AnalysisError(Exception):
    """Raised when the Claude API call fails."""


def analyze(transcript: str, config: PipelineConfig, output_dir: Path) -> Path:
    """Send transcript to Claude with study instructions, save the result.

    Args:
        transcript: The lecture transcript text.
        config: Pipeline configuration (API key, model, etc.).
        output_dir: Where to save the study guide.

    Returns:
        Path to the saved study guide markdown file.

    Raises:
        AnalysisError: If the API call fails.
    """
    instructions = _load_instructions(config)

    logger.info(
        "Sending %d words to Claude (%s)...",
        len(transcript.split()),
        config.claude_model,
    )

    try:
        import anthropic
    except ImportError:
        raise AnalysisError("Anthropic SDK not installed. Run: pip install anthropic")

    try:
        client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        message = client.messages.create(
            model=config.claude_model,
            max_tokens=config.claude_max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"{instructions}\n\n"
                        f"--- LECTURE TRANSCRIPT ---\n\n"
                        f"{transcript}"
                    ),
                }
            ],
        )
    except Exception as e:
        raise AnalysisError(f"Claude API call failed: {e}")

    study_guide = message.content[0].text

    # Save the study guide
    guide_path = output_dir / "study_guide.md"
    guide_path.write_text(study_guide, encoding="utf-8")

    logger.info("Study guide saved: %s (%d words)", guide_path, len(study_guide.split()))
    return guide_path


def _load_instructions(config: PipelineConfig) -> str:
    """Load study instructions from file — user-provided or default."""
    if config.instructions_file and config.instructions_file.exists():
        logger.info("Using custom instructions: %s", config.instructions_file)
        return config.instructions_file.read_text(encoding="utf-8")

    if DEFAULT_INSTRUCTIONS_PATH.exists():
        logger.debug("Using default instructions")
        return DEFAULT_INSTRUCTIONS_PATH.read_text(encoding="utf-8")

    # Hardcoded fallback (shouldn't happen if project is intact)
    logger.warning("No instructions file found, using minimal fallback")
    return "Summarize this lecture and explain the key concepts clearly."
