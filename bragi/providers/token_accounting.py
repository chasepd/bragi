"""Provider-neutral, conservative token estimation helpers."""

from __future__ import annotations

import re
from math import ceil


def estimate_text_tokens(text: str) -> int:
    """Estimate tokens without assuming Latin text or a provider tokenizer."""

    if not text:
        return 0
    ascii_text = "".join(
        character if character.isascii() else " " for character in text
    )
    ascii_tokens = sum(
        ceil(len(segment) / 4) if segment.isalnum() else 1
        for segment in re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9\s]", ascii_text)
    )
    non_ascii_bytes = sum(
        len(character.encode("utf-8"))
        for character in text
        if not character.isascii()
    )
    return max(1, ascii_tokens + ceil(non_ascii_bytes / 2))
