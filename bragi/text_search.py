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
MAX_UNICODE_WORD_TERM_CHARS = 512
MAX_STRUCTURED_IDENTIFIER_BOUNDARY_PROBE_CHARS = (
    MAX_STRUCTURED_IDENTIFIER_CHARS * 4
)
MAX_STRUCTURED_IDENTIFIER_EDGE_SAMPLE_CHARS = (
    (MAX_STRUCTURED_IDENTIFIER_INPUT_CHARS - 1) // 2
    + MAX_STRUCTURED_IDENTIFIER_BOUNDARY_PROBE_CHARS
    + 1
)


def _bounded_nfkc_casefold(
    value: str,
    *,
    max_input_chars: int,
    max_output_chars: int,
) -> str:
    normalized = unicodedata.normalize(
        "NFKC",
        value[:max_input_chars],
    ).casefold()
    if len(normalized) <= max_output_chars:
        return normalized
    edge_chars = max(1, (max_output_chars - 1) // 2)
    return f"{normalized[:edge_chars]} {normalized[-edge_chars:]}"


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
            if term and len(term) <= MAX_UNICODE_WORD_TERM_CHARS:
                terms.append(term)
            current = []
    if current:
        term = "".join(current).strip("'")
        if term and len(term) <= MAX_UNICODE_WORD_TERM_CHARS:
            terms.append(term)
    return tuple(terms)


def structured_identifiers(
    value: str,
    *,
    max_input_chars: int = MAX_STRUCTURED_IDENTIFIER_INPUT_CHARS,
    max_identifiers: int = MAX_STRUCTURED_IDENTIFIERS,
) -> tuple[str, ...]:
    return _structured_identifiers_from_bounded_input(
        _bounded_identifier_input(value, max_input_chars=max_input_chars),
        max_identifiers=max_identifiers,
    )


def structured_identifiers_from_edges(
    prefix: str,
    suffix: str,
    *,
    total_chars: int,
    max_identifiers: int = MAX_STRUCTURED_IDENTIFIERS,
) -> tuple[str, ...]:
    bounded_input = _bounded_identifier_input_from_edges(
        prefix,
        suffix,
        total_chars=total_chars,
        max_input_chars=MAX_STRUCTURED_IDENTIFIER_INPUT_CHARS,
    )
    return _structured_identifiers_from_bounded_input(
        bounded_input,
        max_identifiers=max_identifiers,
    )


def _structured_identifiers_from_bounded_input(
    value: str,
    *,
    max_identifiers: int,
) -> tuple[str, ...]:
    normalized = unicodedata.normalize(
        "NFKC",
        value,
    ).casefold()
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


def _bounded_identifier_input(value: str, *, max_input_chars: int) -> str:
    if max_input_chars <= 0:
        return ""
    if len(value) <= max_input_chars:
        return value
    edge_chars = max(1, (max_input_chars - 1) // 2)
    prefix_end = edge_chars
    if (
        prefix_end < len(value)
        and _structured_identifier_character(value[prefix_end - 1])
        and _structured_identifier_character(value[prefix_end])
    ):
        token_span = _bounded_identifier_token_span(value, cut=prefix_end)
        if token_span is not None:
            prefix_end = token_span[1]
        else:
            while (
                prefix_end > 0
                and _structured_identifier_character(value[prefix_end - 1])
            ):
                prefix_end -= 1
    suffix_start = len(value) - edge_chars
    if (
        suffix_start > 0
        and _structured_identifier_character(value[suffix_start - 1])
        and _structured_identifier_character(value[suffix_start])
    ):
        token_span = _bounded_identifier_token_span(value, cut=suffix_start)
        if token_span is not None:
            suffix_start = token_span[0]
        else:
            while (
                suffix_start < len(value)
                and _structured_identifier_character(value[suffix_start])
            ):
                suffix_start += 1
    return f"{value[:prefix_end]} {value[suffix_start:]}"


def _bounded_identifier_input_from_edges(
    prefix: str,
    suffix: str,
    *,
    total_chars: int,
    max_input_chars: int,
) -> str:
    if max_input_chars <= 0 or total_chars <= 0:
        return ""
    if total_chars <= max_input_chars:
        overlap_chars = max(0, len(prefix) + len(suffix) - total_chars)
        return f"{prefix}{suffix[overlap_chars:]}"[:total_chars]
    edge_chars = max(1, (max_input_chars - 1) // 2)
    prefix_end = min(edge_chars, len(prefix))
    if (
        prefix_end < len(prefix)
        and _structured_identifier_character(prefix[prefix_end - 1])
        and _structured_identifier_character(prefix[prefix_end])
    ):
        token_span = _bounded_identifier_token_span(prefix, cut=prefix_end)
        if token_span is not None:
            prefix_end = token_span[1]
        else:
            while (
                prefix_end > 0
                and _structured_identifier_character(prefix[prefix_end - 1])
            ):
                prefix_end -= 1
    suffix_start = max(0, len(suffix) - edge_chars)
    if (
        suffix_start > 0
        and suffix_start < len(suffix)
        and _structured_identifier_character(suffix[suffix_start - 1])
        and _structured_identifier_character(suffix[suffix_start])
    ):
        token_span = _bounded_identifier_token_span(suffix, cut=suffix_start)
        if token_span is not None:
            suffix_start = token_span[0]
        else:
            while (
                suffix_start < len(suffix)
                and _structured_identifier_character(suffix[suffix_start])
            ):
                suffix_start += 1
    return f"{prefix[:prefix_end]} {suffix[suffix_start:]}"


def _bounded_identifier_token_span(
    value: str,
    *,
    cut: int,
) -> tuple[int, int] | None:
    token_start = cut
    while (
        token_start > 0
        and cut - token_start <= MAX_STRUCTURED_IDENTIFIER_BOUNDARY_PROBE_CHARS
        and _structured_identifier_character(value[token_start - 1])
    ):
        token_start -= 1
    if (
        cut - token_start
        > MAX_STRUCTURED_IDENTIFIER_BOUNDARY_PROBE_CHARS
    ):
        return None
    token_end = cut
    while (
        token_end < len(value)
        and token_end - token_start
        <= MAX_STRUCTURED_IDENTIFIER_BOUNDARY_PROBE_CHARS
        and _structured_identifier_character(value[token_end])
    ):
        token_end += 1
    if (
        token_end - token_start
        > MAX_STRUCTURED_IDENTIFIER_BOUNDARY_PROBE_CHARS
    ):
        return None
    normalized_token = unicodedata.normalize(
        "NFKC",
        value[token_start:token_end],
    ).casefold()
    if (
        len(normalized_token) > MAX_STRUCTURED_IDENTIFIER_CHARS
        or re.fullmatch(
            r"[^\W_]+(?:[-_.][^\W_]+)+",
            normalized_token,
            flags=re.UNICODE,
        )
        is None
    ):
        return None
    return token_start, token_end


def _structured_identifier_character(character: str) -> bool:
    normalized = unicodedata.normalize("NFKC", character)
    return bool(normalized) and all(
        item.isalnum()
        or item in {"-", "_", "."}
        or unicodedata.category(item).startswith("M")
        for item in normalized
    )


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
        for index in range(len(run) - 1):
            anchors.append(run[index : index + 2])
            if index + 3 <= len(run):
                anchors.append(run[index : index + 3])
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
