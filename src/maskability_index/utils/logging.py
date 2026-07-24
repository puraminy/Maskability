"""Logging configuration for command-line and file-based experiment logs."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_file: Path | str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure root logging and return the project logger."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("maskability_index")
