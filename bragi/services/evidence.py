"""Evidence matching helpers shared by model-output validation paths."""

from __future__ import annotations

_FORMAT_NORMALIZED_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
    }
)
_MARKDOWN_QUOTE_MARKERS = ("*", "`")


def quote_matches_source(quote: str, source_body: str) -> bool:
    stripped = quote.strip()
    if not stripped:
        return False
    if stripped in source_body:
        return True
    normalized_quote = _format_normalized_quote(stripped)
    if not normalized_quote:
        return False
    return normalized_quote in _format_normalized_quote(source_body)


def _format_normalized_quote(value: str) -> str:
    normalized = value.translate(_FORMAT_NORMALIZED_QUOTE_TRANSLATION)
    for marker in _MARKDOWN_QUOTE_MARKERS:
        if normalized.count(marker) >= 2:
            normalized = normalized.replace(marker, "")
    return " ".join(normalized.split())
