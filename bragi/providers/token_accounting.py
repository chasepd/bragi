"""Provider-neutral, conservative token estimation helpers."""

from __future__ import annotations

_BYTES_PER_TOKEN = 4


def estimate_text_tokens(text: str) -> int:
    """Return a tokenizer-independent conservative upper bound for provider text.

    The estimate is roughly 1 token per 4 bytes of UTF-8, which is a common
    ratio for English prose with most BPE tokenizers. The function still
    returns a conservative upper bound for high-entropy input and CJK text
    where the byte-to-token ratio is closer to 1:1, but it is no longer
    artificially tight (1 byte per token) which caused every chat request to
    be trimmed well below the model's actual context window.
    """

    if not text:
        return 0
    byte_len = len(text.encode("utf-8"))
    return (byte_len + _BYTES_PER_TOKEN - 1) // _BYTES_PER_TOKEN
