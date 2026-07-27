"""Deterministic Unicode lexical helpers for local context recall."""

from __future__ import annotations

import json
import re
import unicodedata
import zlib

MAX_CJK_LEXICAL_INPUT_CHARS = 16_384
MAX_STRUCTURED_IDENTIFIER_INPUT_CHARS = 65_536
MAX_STRUCTURED_IDENTIFIER_CHARS = 512
MAX_STRUCTURED_IDENTIFIERS = 4_096
MAX_IDENTIFIER_FILTER_IDENTIFIERS = 32_768
MAX_IDENTIFIER_FILTER_UNCOMPRESSED_BYTES = 8 * (
    MAX_STRUCTURED_IDENTIFIER_INPUT_CHARS + 1
)
MAX_NORMALIZED_SEARCH_CHARS = 65_536


def _bounded_nfkc_casefold(
    value: str,
    *,
    max_input_chars: int,
    max_output_chars: int,
) -> str:
    normalized_parts: list[str] = []
    remaining = max_output_chars
    for start in range(0, min(len(value), max_input_chars), 1024):
        normalized = unicodedata.normalize(
            "NFKC",
            value[start : start + 1024],
        ).casefold()
        normalized_parts.append(normalized[:remaining])
        remaining -= min(len(normalized), remaining)
        if remaining <= 0:
            break
    return "".join(normalized_parts)


def unicode_word_terms(value: str) -> tuple[str, ...]:
    normalized = _bounded_nfkc_casefold(
        value,
        max_input_chars=MAX_NORMALIZED_SEARCH_CHARS,
        max_output_chars=MAX_NORMALIZED_SEARCH_CHARS,
    )
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


def structured_identifiers(
    value: str,
    *,
    max_input_chars: int = MAX_STRUCTURED_IDENTIFIER_INPUT_CHARS,
    max_identifiers: int = MAX_STRUCTURED_IDENTIFIERS,
) -> tuple[str, ...]:
    normalized = _bounded_nfkc_casefold(
        value,
        max_input_chars=max_input_chars,
        max_output_chars=MAX_NORMALIZED_SEARCH_CHARS,
    )
    identifiers = tuple(
        dict.fromkeys(
            identifier
            for identifier in re.findall(
                r"(?<!\w)[^\W_]+(?:[-_.][^\W_]+)+(?!\w)",
                normalized,
                flags=re.UNICODE,
            )
            if len(identifier) <= MAX_STRUCTURED_IDENTIFIER_CHARS
        )
    )
    if len(identifiers) <= max_identifiers:
        return identifiers
    edge_count = max_identifiers // 2
    if edge_count <= 0:
        return ()
    return tuple(
        dict.fromkeys(
            (
                *identifiers[:edge_count],
                *identifiers[-edge_count:],
            )
        )
    )[:max_identifiers]


def structured_identifier_filter(title: str, body: str) -> bytes:
    identifiers = tuple(
        dict.fromkeys(
            (
                *structured_identifiers(
                    title,
                    max_identifiers=MAX_IDENTIFIER_FILTER_IDENTIFIERS,
                ),
                *structured_identifiers(
                    body,
                    max_identifiers=MAX_IDENTIFIER_FILTER_IDENTIFIERS,
                ),
            )
        )
    )
    return zlib.compress("\n".join(identifiers).encode("utf-8"))


def identifier_filter_matches(value: object, identifiers_json: object) -> int:
    if not isinstance(value, bytes):
        return 0
    try:
        decompressor = zlib.decompressobj()
        decoded = decompressor.decompress(
            value,
            MAX_IDENTIFIER_FILTER_UNCOMPRESSED_BYTES + 1,
        )
        if (
            len(decoded) > MAX_IDENTIFIER_FILTER_UNCOMPRESSED_BYTES
            or decompressor.unconsumed_tail
            or not decompressor.eof
        ):
            return 0
        requested = json.loads(str(identifiers_json))
    except (UnicodeDecodeError, ValueError, zlib.error):
        return 0
    if not isinstance(requested, list):
        return 0
    padded = b"\n" + decoded + b"\n"
    return int(
        any(
            isinstance(identifier, str)
            and bool(identifier)
            and (
                b"\n"
                + identifier.casefold().encode("utf-8")
                + b"\n"
            )
            in padded
            for identifier in requested
        )
    )


def cjk_lexical_anchors(value: str) -> tuple[str, ...]:
    normalized = _bounded_nfkc_casefold(
        value,
        max_input_chars=MAX_CJK_LEXICAL_INPUT_CHARS,
        max_output_chars=MAX_CJK_LEXICAL_INPUT_CHARS,
    )
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
