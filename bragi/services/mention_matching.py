"""Boundary-aware free-text character mention matching."""

from __future__ import annotations

import re
from collections.abc import Iterable

_NAME_BOUNDARY_CHARS = r"\w-"
_POSSESSIVE_SUFFIX = r"(?:['\u2019]s)?"


def character_name_is_mentioned(
    *,
    name: str,
    aliases: Iterable[str],
    text: str,
) -> bool:
    """Return true when text contains a deliberate character name or alias."""
    return any(
        _phrase_is_mentioned(candidate, text)
        for candidate in (name, *tuple(aliases))
        if candidate.strip()
    )


def _phrase_is_mentioned(phrase: str, text: str) -> bool:
    parts = phrase.strip().split()
    if not parts:
        return False
    phrase_pattern = r"\s+".join(re.escape(part.casefold()) for part in parts)
    pattern = re.compile(
        rf"(?<![{_NAME_BOUNDARY_CHARS}])"
        rf"{phrase_pattern}"
        rf"{_POSSESSIVE_SUFFIX}"
        rf"(?![{_NAME_BOUNDARY_CHARS}])"
    )
    return pattern.search(text.casefold()) is not None
