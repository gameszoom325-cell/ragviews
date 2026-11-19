"""Security Utilities for RAG Pipeline.

Implements query sanitization, prompt injection defenses, path traversal
guards, and credential masking.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

from exceptions import InputValidationError
from logger import get_logger

logger = get_logger(__name__)

# Common injection patterns to detect and flag
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(an?\s+)?unrestricted", re.IGNORECASE),
    re.compile(r"system\s*:\s*system\s+prompt", re.IGNORECASE),
    re.compile(r"disregard\s+(the\s+)?above", re.IGNORECASE),
    re.compile(r"bypass\s+safety\s+filters", re.IGNORECASE),
]


def sanitize_query(question: str, max_length: int = 2000) -> Tuple[str, bool]:
    """Sanitize and validate an incoming user query.

    Args:
        question: User query string.
        max_length: Maximum allowed length in characters.

    Returns:
        Tuple[str, bool]: Cleaned query string and flag indicating if suspicious.

    Raises:
        InputValidationError: If input is not a string or exceeds max length.
    """
    if not isinstance(question, str):
        raise InputValidationError(f"Query must be a string, got {type(question).__name__}")

    cleaned = question.strip()
    if not cleaned:
        return "", False

    if len(cleaned) > max_length:
        raise InputValidationError(
            f"Query length ({len(cleaned)}) exceeds maximum allowed length ({max_length})."
        )

    # Strip control characters except newline and tab
    cleaned = "".join(ch for ch in cleaned if ch in ("\n", "\t") or ch.isprintable())

    # Detect potential prompt injections
    is_suspicious = False
    for pattern in INJECTION_PATTERNS:
        if pattern.search(cleaned):
            logger.warning(
                "Suspicious prompt pattern detected in query: %s",
                pattern.pattern,
            )
            is_suspicious = True
            break

    return cleaned, is_suspicious


def validate_path_traversal(base_directory: Path, target_path: Path) -> Path:
    """Validate that target_path is located strictly inside base_directory.

    Args:
        base_directory: Permitted root directory.
        target_path: Path to validate.

    Returns:
        Path: Resolved safe path.

    Raises:
        InputValidationError: If path escapes the base directory.
    """
    resolved_base = base_directory.resolve()
    resolved_target = target_path.resolve()

    try:
        resolved_target.relative_to(resolved_base)
    except ValueError as exc:
        raise InputValidationError(
            f"Path traversal detected: '{target_path}' is outside permitted directory '{base_directory}'."
        ) from exc

    return resolved_target


def mask_secret(secret: str, visible_suffix_len: int = 4) -> str:
    """Mask sensitive string, leaving only trailing characters visible."""
    if not secret:
        return ""
    if len(secret) <= visible_suffix_len:
        return "****"
    return f"****{secret[-visible_suffix_len:]}"
