"""
Logging setup for the lecture pipeline.

Why structured logging instead of print()?
- Each line has a timestamp, level, and source module
- You can filter by severity (DEBUG vs INFO vs ERROR)
- Output can go to a file for debugging past runs
- Production systems expect logs, not print statements
"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "lecture_pipeline",
    level: str = "INFO",
    log_file: Path | None = None,
) -> logging.Logger:
    """Configure and return the pipeline logger.

    Args:
        name: Logger name (shows up in log lines).
        level: Minimum log level — DEBUG, INFO, WARNING, ERROR.
        log_file: If set, also writes logs to this file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s.%(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console output
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File output (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
