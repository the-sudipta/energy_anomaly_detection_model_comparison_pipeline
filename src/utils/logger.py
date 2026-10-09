"""Console and file logging."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

LOGGER_NAME = "energy_anomaly"
_FILE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure the project logger to write to the console and a timestamped file.

    Uses ``rich`` for the console when available, plain text otherwise.

    Args:
        log_dir: Directory that receives ``pipeline_<timestamp>.log``.
        level: Logging level for both handlers.

    Returns:
        The configured project logger.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    logger.propagate = False
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_dir / f"pipeline_{stamp}.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
    logger.addHandler(file_handler)
    logger.addHandler(_console_handler())
    return logger


def _console_handler() -> logging.Handler:
    """Build the console handler, preferring ``rich`` formatting.

    Returns:
        A logging handler that writes to the terminal.
    """
    try:
        from rich.logging import RichHandler

        return RichHandler(show_path=False, rich_tracebacks=True, markup=False)
    except ImportError:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
        return handler


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the project logger or one of its children.

    Args:
        name: Optional child name, usually ``__name__``.

    Returns:
        A logger under the project namespace.
    """
    if not name:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name.split('.')[-1]}")
