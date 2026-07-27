"""Deterministic Unicode lexical helpers for local context recall."""

from __future__ import annotations

import unicodedata

MAX_CJK_LEXICAL_INPUT_CHARS = 16_384


def unicode_word_terms(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    terms: list[str] = []
    current: list[str] = []
    for character in normalized:
        if character.isalnum() or (character in {"'", "’"} and current):
            current.append("'" if character == "’" else character)
            continue
        if current:
            term = "".join(current).strip("'")
            if term:
                terms.append(term)
            current = []
    if current:
        term = "".join(current).strip("'")
        if term:
            terms.append(term)
    return tuple(terms)


def cjk_lexical_anchors(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize(
        "NFKC",
        value[:MAX_CJK_LEXICAL_INPUT_CHARS],
    ).casefold()
    runs: list[str] = []
    current: list[str] = []
    current_family = ""
    for character in normalized:
        family = _cjk_script_family(character)
        if family and family == current_family:
            current.append(character)
            continue
        if current:
            runs.append("".join(current))
        current = [character] if family else []
        current_family = family
    if current:
        runs.append("".join(current))

    anchors: list[str] = []
    for run in runs:
        if len(run) == 1:
            if _cjk_script_family(run) == "han":
                anchors.append(run)
            continue
        if len(run) <= 64:
            anchors.append(run)
        for width in (3, 2):
            if len(run) < width:
                continue
            anchors.extend(
                run[index : index + width]
                for index in range(len(run) - width + 1)
            )
    return tuple(dict.fromkeys(anchors))


def _cjk_script_family(character: str) -> str:
    codepoint = ord(character)
    if (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    ):
        return "han"
    if 0x3040 <= codepoint <= 0x309F:
        return "hiragana"
    if 0x30A0 <= codepoint <= 0x30FF or 0x31F0 <= codepoint <= 0x31FF:
        return "katakana"
    if (
        0x1100 <= codepoint <= 0x11FF
        or 0x3130 <= codepoint <= 0x318F
        or 0xAC00 <= codepoint <= 0xD7AF
    ):
        return "hangul"
    return ""
