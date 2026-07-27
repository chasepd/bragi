"""Provider-neutral, conservative token estimation helpers."""

from __future__ import annotations


def estimate_text_tokens(text: str) -> int:
    """Return a tokenizer-independent worst-case bound for provider text."""

    if not text:
        return 0
    return len(text.encode("utf-8"))
