"""
Configuration management for the lecture pipeline.

Loads settings from environment variables and .env file.
All pipeline behavior is controlled here — no magic strings in other modules.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """Central configuration for all pipeline stages."""

    # Paths
    output_dir: Path = field(default_factory=lambda: Path.home() / "lecture_pipeline_output")
    instructions_file: Path | None = None

    # Downloader
    download_retries: int = 3
    download_timeout: int = 300  # seconds
    referer: str | None = None  # Optional Referer header for downloads

    # Transcriber
    whisper_model: str = "base"  # tiny | base | small | medium | large

    # Analyzer (Claude API)
    anthropic_api_key: str | None = None
    claude_model: str = "claude-sonnet-4-5"
    claude_max_tokens: int = 16000

    # Pipeline behavior
    keep_video: bool = False
    skip_analysis: bool = False  # True = stop after transcription
    skip_html: bool = False       # True = skip HTML rendering

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Build config from environment variables.

        This is the preferred way to create a config — it reads from
        the environment, which can be populated by a .env file, shell
        exports, Docker env, or a future scheduler.
        """
        return cls(
            output_dir=Path(os.getenv("LP_OUTPUT_DIR", str(Path.home() / "lecture_pipeline_output"))),
            instructions_file=Path(f) if (f := os.getenv("LP_INSTRUCTIONS_FILE")) else None,
            download_retries=int(os.getenv("LP_DOWNLOAD_RETRIES", "3")),
            download_timeout=int(os.getenv("LP_DOWNLOAD_TIMEOUT", "300")),
            referer=os.getenv("LP_REFERER"),
            whisper_model=os.getenv("LP_WHISPER_MODEL", "base"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            claude_model=os.getenv("LP_CLAUDE_MODEL", "claude-sonnet-4-5"),
            claude_max_tokens=int(os.getenv("LP_CLAUDE_MAX_TOKENS", "16000")),
            keep_video=os.getenv("LP_KEEP_VIDEO", "false").lower() == "true",
            skip_analysis=os.getenv("LP_SKIP_ANALYSIS", "false").lower() == "true",
            skip_html=os.getenv("LP_SKIP_HTML", "false").lower() == "true",
        )

    def validate(self) -> list[str]:
        """Check config for problems. Returns list of error messages."""
        errors = []
        if not self.skip_analysis and not self.anthropic_api_key:
            errors.append(
                "ANTHROPIC_API_KEY not set. Either set it or use --skip-analysis. "
                "Get a key at: https://console.anthropic.com/settings/keys"
            )
        if self.instructions_file and not self.instructions_file.exists():
            errors.append(f"Instructions file not found: {self.instructions_file}")
        if self.whisper_model not in ("tiny", "base", "small", "medium", "large"):
            errors.append(f"Invalid whisper model: {self.whisper_model}")
        return errors
