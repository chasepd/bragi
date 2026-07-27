"""Provider-neutral, conservative token estimation helpers."""

from __future__ import annotations

from math import ceil


def estimate_text_tokens(text: str) -> int:
    """Estimate tokens without assuming Latin text or a provider tokenizer."""

    if not text:
        return 0
    ascii_chars = sum(1 for character in text if character.isascii())
    non_ascii_bytes = len(text.encode("utf-8")) - ascii_chars
    estimate = (ascii_chars / 4) + (non_ascii_bytes / 2)
    return max(1, ceil(estimate))
