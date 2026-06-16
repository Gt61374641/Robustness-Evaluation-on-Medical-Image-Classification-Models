"""Logging utilities for experiment tracking."""

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_dir: Path = None, level: int = logging.INFO) -> logging.Logger:
    """Create a logger that writes to both console and file.

    Args:
        name: Logger name.
        log_dir: If provided, also write logs to {log_dir}/{name}.log.
        level: Logging level.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
