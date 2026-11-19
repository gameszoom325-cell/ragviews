"""Structured Logging Subsystem with Automatic Secret Sanitization.

Provides thread-safe loggers with redaction of API keys and credentials.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Optional

# Regex pattern matching potential OpenAI API keys or tokens
SECRET_PATTERN = re.compile(r"sk-[a-zA-Z0-9_-]{20,}")


class SecretSanitizingFormatter(logging.Formatter):
    """Logging formatter that redacts sensitive API keys and secrets."""

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        return SECRET_PATTERN.sub("sk-***[REDACTED]***", original)


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Retrieve or configure a structured logger.

    Args:
        name: Logger identifier (typically __name__).
        level: Optional log level override (e.g., 'DEBUG', 'INFO').

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Determine desired log level
    log_level_name = level or "INFO"
    log_level = getattr(logging, log_level_name.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Avoid duplicate handlers if logger was already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = SecretSanitizingFormatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
