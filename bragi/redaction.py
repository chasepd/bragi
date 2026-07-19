"""Shared redaction helpers for local diagnostics and logs."""

from __future__ import annotations

import re
from collections.abc import Mapping

_SECRET_TEXT_PATTERNS = (
    (re.compile(r"\bsk-[A-Za-z0-9._-]+"), "[redacted]"),
    (
        re.compile(r"\b[Bb][Ee][Aa][Rr][Ee][Rr]\s+[A-Za-z0-9._~+/=-]+"),
        "Bearer [redacted]",
    ),
    (
        re.compile(
            r"(?i)([\"']?\b(?:token|api[_-]?key|authorization)\b[\"']?"
            r"\s*[:=]\s*[\"']?)([^\s,;\"']+)"
        ),
        r"\1[redacted]",
    ),
)
_SENSITIVE_KEYS = frozenset({"api_key", "apikey", "token", "authorization"})


def redact_text(value: str | None) -> str | None:
    if value is None:
        return None
    redacted = value
    for pattern, replacement in _SECRET_TEXT_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def redact_log_value(value: object, *, key: str | None = None) -> object:
    if key is not None and _is_sensitive_key(key):
        return "[redacted]" if value is not None else None
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {
            str(item_key): redact_log_value(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [redact_log_value(item) for item in value]
    return value


def _is_sensitive_key(key: str) -> bool:
    return key.lower().replace("-", "_") in _SENSITIVE_KEYS
