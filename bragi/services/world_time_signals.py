"""Deterministic world-time signal detection."""

from __future__ import annotations

import re
from typing import NamedTuple


class _NormalizedOccurrence(NamedTuple):
    source_start: int
    source_end: int
    normalized_start: int


class _ClockSpan(NamedTuple):
    start: int
    end: int


_CLOCK_24H_RE = re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b")
_MERIDIEM_CLOCK_RE = re.compile(
    r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)\b",
)
_READOUT_DURATION_RE = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
    r"(?:seconds?|minutes?)\b",
)
_READOUT_DEVICE_ANCHOR = (
    r"countdown|timer|scoreboard|game\s+clock|shot\s+clock|play\s+clock|"
    r"match\s+clock"
)
_READOUT_PERIOD_ANCHOR = r"round|quarter|period|half|inning"
_READOUT_CONTEXT_ANCHOR = rf"{_READOUT_DEVICE_ANCHOR}|{_READOUT_PERIOD_ANCHOR}"
_TIMER_SUFFIX_PATTERN = (
    r"[\s,;:()\[\]\-]*(?:(?:is\s+)?(?:left|remaining|remains?)"
    r"(?=$|[\s,;:()\[\]\-.]*(?:$|[.!?;,]|on\b|in\b|for\b|"
    rf"(?:{_READOUT_CONTEXT_ANCHOR}|seconds?|minutes?)\b))|"
    rf"left\s+to\s+go\b.{{0,32}}\b(?:{_READOUT_DEVICE_ANCHOR})|"
    rf"on\s+(?:the\s+)?(?:{_READOUT_DEVICE_ANCHOR})|"
    r"(?:seconds?|minutes?)\s+(?:left|remaining)|"
    r"with\s+(?:\w+\s+){0,6}(?:seconds?|minutes?)\s+"
    rf"(?:left|remaining|remain)\b.{{0,64}}\b(?:{_READOUT_DEVICE_ANCHOR}))\b"
)
_TIMER_PREFIX_RE = re.compile(
    rf"(?:\b(?:elapsed|{_READOUT_DEVICE_ANCHOR})\b|"
    rf"\b(?:{_READOUT_PERIOD_ANCHOR})\b(?=.{{0,32}}\b"
    rf"(?:{_READOUT_DEVICE_ANCHOR}|remaining|left)\b))"
    r"(?:\W+\w+){0,4}\W*$",
)
_TIMER_SUFFIX_RE = re.compile(rf"^{_TIMER_SUFFIX_PATTERN}")
_PHASE_ANCHOR = (
    r"late\s+morning|early\s+morning|morning|afternoon|evening|night|"
    r"dawn|dusk|sunrise|sunset|noon|midday|midnight|tonight|tomorrow|"
    r"overnight"
)
_DAY_ANCHOR = (
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"next\s+day|following\s+day"
)
_APPOINTMENT_ANCHOR = (
    r"class|school|work|meeting|appointment|practice|shift|breakfast|"
    r"lunch|dinner"
)
_LEADING_APPOINTMENT_ANCHOR = (
    r"class|meeting|appointment|practice|shift|breakfast|lunch|dinner"
)
_TIME_ANCHOR = (
    rf"{_PHASE_ANCHOR}|{_DAY_ANCHOR}|{_APPOINTMENT_ANCHOR}|"
    rf"{_MERIDIEM_CLOCK_RE.pattern}|{_CLOCK_24H_RE.pattern}"
)
_TRAVEL_TIME_ANCHOR = (
    rf"{_PHASE_ANCHOR}|{_DAY_ANCHOR}|"
    rf"{_MERIDIEM_CLOCK_RE.pattern}|{_CLOCK_24H_RE.pattern}"
)
_COUNTED_DURATION_AMOUNT = (
    r"(?:a\s+couple(?:\s+of)?|couple(?:\s+of)?|a\s+few|several|a|an|"
    r"one|two|three|four|five|six|seven|eight|nine|ten|\d+)"
)
_FRACTIONAL_DURATION_ANCHOR = (
    r"(?:(?:a\s+)?half\s+(?:an?\s+)?|quarter\s+(?:of\s+an?\s+)?)"
    r"(?:minutes?|hours?|days?)"
)
_DURATION_ANCHOR = (
    rf"(?:{_COUNTED_DURATION_AMOUNT}\s+(?:minutes?|hours?|days?)|"
    rf"{_FRACTIONAL_DURATION_ANCHOR})"
)
_CLOCK_ADVANCE_ANCHOR = (
    rf"(?:{_MERIDIEM_CLOCK_RE.pattern}|{_CLOCK_24H_RE.pattern})"
    rf"(?!{_TIMER_SUFFIX_PATTERN})"
)
_NON_TIMER_ADVANCE_TARGET = (
    rf"{_PHASE_ANCHOR}|{_DAY_ANCHOR}|{_APPOINTMENT_ANCHOR}|"
    rf"{_DURATION_ANCHOR}|{_CLOCK_ADVANCE_ANCHOR}"
)
_LEADING_TIME_ANCHOR = (
    rf"{_PHASE_ANCHOR}|{_DAY_ANCHOR}|{_LEADING_APPOINTMENT_ANCHOR}"
)
_TRAVEL_VERB_ANCHOR = (
    r"go|goes|went|head|heads|headed|travel|travels|traveled|travelled|"
    r"walk|walks|walked|drive|drives|drove|ride|rides|rode|return|returns|"
    r"returned|arrive|arrives|arrived|leave|leaves|left|regroup|regroups|"
    r"regrouped|meet|meets|met"
)
_DESTINATION_TRAVEL_VERB_ANCHOR = (
    r"go|goes|went|head|heads|headed|travel|travels|traveled|travelled|"
    r"walk|walks|walked|drive|drives|drove|ride|rides|rode|return|returns|"
    r"returned"
)
_WORLD_TIME_ADVANCE_PATTERNS = (
    r"\bskip(?:s|ped|ping)?\s+(?:ahead|forward|to|until|through|past)\b",
    r"\bfast[- ]?forward\s+(?:to|until|through|past)\b",
    r"\b(?:wait|waits|waited|waiting|rest|rests|rested|resting|sleep|"
    r"sleeps|slept|sleeping|stay|stays|stayed|staying)\s+"
    r"(?:until|for|through|past)\b",
    rf"\b(?:wait|waits|waited|waiting|rest|rests|rested|resting|sleep|"
    rf"sleeps|slept|sleeping|stay|stays|stayed|staying)\s+"
    rf"{_DURATION_ANCHOR}\b",
    rf"\b(?:spend|spends|spent|spending)\s+(?:the\s+)?"
    rf"(?:{_PHASE_ANCHOR}|{_DURATION_ANCHOR})\b",
    rf"\b(?:pass|passes|passed|passing)\s+{_DURATION_ANCHOR}\b",
    rf"\bafter\s+(?:the\s+)?(?:{_TIME_ANCHOR})\b",
    rf"\b(?:until|by)\s+(?:{_TIME_ANCHOR})\b",
    rf"\b(?:at|around|before)\s+(?:{_CLOCK_ADVANCE_ANCHOR})\b",
    rf"(?:^|[.!?;]\s*|\b(?:and|then|so)\s+)\b(?:at|around|before)\s+"
    rf"(?:{_LEADING_TIME_ANCHOR})\b",
    r"\b(?:the\s+)?next\s+(?:day|morning|afternoon|evening|night|week|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\bfollowing\s+(?:day|morning|afternoon|evening|night)\b",
    r"\blater\s+(?:that|this|tonight)\s+"
    r"(?:morning|afternoon|evening|night)\b",
    r"\b(?:minutes?|hours?|days?)\s+later\b",
    r"\b(?:overnight|through\s+the\s+night)\b",
    rf"\b(?:{_TRAVEL_VERB_ANCHOR})\b"
    rf".{{0,80}}\b(?:at|by|until|after|around|before)\s+"
    rf"(?:{_TRAVEL_TIME_ANCHOR})\b",
    rf"\b(?:{_TRAVEL_VERB_ANCHOR})\b"
    rf".{{0,80}}\b(?:for|over|through)\s+{_DURATION_ANCHOR}\b",
    rf"\b(?:{_DESTINATION_TRAVEL_VERB_ANCHOR})\b"
    rf".{{0,80}}\bto\s+(?:{_APPOINTMENT_ANCHOR})\b",
    r"\b(?:loop|cycle)\s+resets?\s+(?:to|at|on|into)\b",
    r"\b(?:next|new)\s+loop\s+(?:phase|cycle|reset)\b",
)
_CLEAR_NON_TIMER_ADVANCE_PATTERNS = (
    rf"\bskip(?:s|ped|ping)?\s+(?:(?:ahead|forward)\s+)?"
    rf"(?:to|until|through|past)\s+(?:{_NON_TIMER_ADVANCE_TARGET})\b",
    rf"\bfast[- ]?forward\s+(?:to|until|through|past)\s+"
    rf"(?:{_NON_TIMER_ADVANCE_TARGET})\b",
    r"\b(?:wait|waits|waited|waiting|rest|rests|rested|resting|sleep|"
    r"sleeps|slept|sleeping|stay|stays|stayed|staying)\s+"
    rf"(?:until|for|through|to|past)\s+(?:{_PHASE_ANCHOR}|{_DAY_ANCHOR}|"
    rf"{_APPOINTMENT_ANCHOR}|{_DURATION_ANCHOR}|{_CLOCK_ADVANCE_ANCHOR})\b",
    rf"\b(?:wait|waits|waited|waiting|rest|rests|rested|resting|sleep|"
    rf"sleeps|slept|sleeping|stay|stays|stayed|staying)\s+"
    rf"{_DURATION_ANCHOR}\b",
    rf"\b(?:spend|spends|spent|spending)\s+(?:the\s+)?"
    rf"(?:{_PHASE_ANCHOR}|{_DURATION_ANCHOR})\b",
    rf"\b(?:pass|passes|passed|passing)\s+{_DURATION_ANCHOR}\b",
    rf"\bafter\s+(?:the\s+)?(?:{_PHASE_ANCHOR}|{_DAY_ANCHOR}|"
    rf"{_APPOINTMENT_ANCHOR}|{_CLOCK_ADVANCE_ANCHOR})\b",
    r"\b(?:the\s+)?next\s+(?:day|morning|afternoon|evening|night|week|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\bfollowing\s+(?:day|morning|afternoon|evening|night)\b",
    r"\blater\s+(?:that|this|tonight)\s+"
    r"(?:morning|afternoon|evening|night)\b",
    r"\b(?:minutes?|hours?|days?)\s+later\b",
    r"\b(?:overnight|through\s+the\s+night)\b",
    rf"\b(?:until|by)\s+(?:{_PHASE_ANCHOR}|{_DAY_ANCHOR}|"
    rf"{_APPOINTMENT_ANCHOR})\b",
    rf"\b(?:until|by)\s+{_CLOCK_ADVANCE_ANCHOR}\b",
    rf"\b(?:at|around|before)\s+{_CLOCK_ADVANCE_ANCHOR}\b",
    rf"(?:^|[.!?;]\s*|\b(?:and|then|so)\s+)\b(?:at|around|before)\s+"
    rf"(?:{_LEADING_TIME_ANCHOR})\b",
    rf"\b(?:{_TRAVEL_VERB_ANCHOR})\b"
    rf".{{0,80}}\b(?:at|by|until|after|around|before)\s+"
    rf"(?:{_PHASE_ANCHOR}|{_DAY_ANCHOR}|{_CLOCK_ADVANCE_ANCHOR})\b",
    rf"\b(?:{_TRAVEL_VERB_ANCHOR})\b"
    rf".{{0,80}}\b(?:for|over|through)\s+{_DURATION_ANCHOR}\b",
    rf"\b(?:{_DESTINATION_TRAVEL_VERB_ANCHOR})\b"
    rf".{{0,80}}\bto\s+(?:{_APPOINTMENT_ANCHOR})\b",
    r"\b(?:loop|cycle)\s+resets?\s+(?:to|at|on|into)\b",
    r"\b(?:next|new)\s+loop\s+(?:phase|cycle|reset)\b",
)


def has_world_time_advance_signal(text: str) -> bool:
    normalized = text.casefold()
    if timer_readout_without_clock_advance(normalized):
        return False
    return any(
        re.search(pattern, normalized) is not None
        for pattern in _WORLD_TIME_ADVANCE_PATTERNS
    )


def timer_readout_without_clock_advance(text: str) -> bool:
    normalized = text.casefold()
    if not _timer_readout_near_context(normalized):
        return False
    if _has_clear_non_timer_time_advance(_without_timer_readout_clauses(normalized)):
        return False
    return True


def timer_readout_evidence_without_clock_advance(
    evidence_quote: str,
    source_body: str,
) -> bool:
    quote = _normalize_text_spacing(evidence_quote.casefold())
    source = source_body.casefold()
    normalized_source, normalized_source_spans = _normalized_text_with_source_spans(
        source,
    )
    occurrences = _normalized_occurrences(
        normalized_source,
        normalized_source_spans,
        quote,
    )
    if not _readout_value_spans(quote):
        if not occurrences:
            return False
        if all(
            _span_overlaps_timer_clause(
                source,
                start=occurrence.source_start,
                end=occurrence.source_end,
            )
            for occurrence in occurrences
        ):
            return True
        return False
    quote_without_timer_clauses = _without_timer_readout_clauses(quote)
    if _has_clear_non_clock_time_advance(quote_without_timer_clauses):
        return False
    if occurrences:
        has_timer_clock = False
        for occurrence in occurrences:
            occurrence_has_timer_clock = False
            for match in _readout_value_spans(quote):
                clock_start, clock_end = _source_span_for_normalized_range(
                    normalized_source_spans,
                    start=occurrence.normalized_start + match.start,
                    end=occurrence.normalized_start + match.end,
                )
                if _span_overlaps_timer_clause(
                    source,
                    start=clock_start,
                    end=clock_end,
                ):
                    occurrence_has_timer_clock = True
                    continue
                if _source_clock_is_clear_non_timer_advance(
                    source,
                    start=clock_start,
                    end=clock_end,
                ):
                    return False
                if _clock_has_timer_context(source, start=clock_start, end=clock_end):
                    occurrence_has_timer_clock = True
            if not occurrence_has_timer_clock:
                return False
            has_timer_clock = True
        if has_timer_clock:
            return True
    if _has_clear_non_timer_time_advance(quote):
        return False
    if _timer_readout_near_context(quote):
        return True
    if not occurrences:
        return False
    return all(
        _timer_readout_near_context(
            source[max(0, occurrence.source_start - 48) : occurrence.source_end + 48],
        )
        for occurrence in occurrences
    )


def text_without_timer_readout_clauses(text: str) -> str:
    return _without_timer_readout_clauses(text.casefold())


def first_non_timer_24h_clock(text: str) -> str:
    normalized = text.casefold()
    for match in _CLOCK_24H_RE.finditer(normalized):
        if _span_overlaps_timer_clause(
            normalized,
            start=match.start(),
            end=match.end(),
        ):
            continue
        if not _clock_has_timer_context(
            normalized,
            start=match.start(),
            end=match.end(),
        ):
            return match.group(0)
    return ""


def _find_occurrences(text: str, needle: str) -> tuple[int, ...]:
    if not needle:
        return ()
    indexes: list[int] = []
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            return tuple(indexes)
        indexes.append(index)
        start = index + max(1, len(needle))


def _normalize_text_spacing(value: str) -> str:
    return " ".join(value.strip().split())


def _normalized_text_with_source_spans(
    value: str,
) -> tuple[str, tuple[tuple[int, int], ...]]:
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    previous_end = 0
    for match in re.finditer(r"\S+", value):
        if parts:
            parts.append(" ")
            spans.append((previous_end, match.start()))
        for index, char in enumerate(match.group(0)):
            parts.append(char)
            source_index = match.start() + index
            spans.append((source_index, source_index + len(char)))
        previous_end = match.end()
    return "".join(parts), tuple(spans)


def _normalized_occurrences(
    normalized_text: str,
    source_spans: tuple[tuple[int, int], ...],
    needle: str,
) -> tuple[_NormalizedOccurrence, ...]:
    occurrences: list[_NormalizedOccurrence] = []
    for index in _find_occurrences(normalized_text, needle):
        source_start, source_end = _source_span_for_normalized_range(
            source_spans,
            start=index,
            end=index + len(needle),
        )
        occurrences.append(
            _NormalizedOccurrence(
                source_start=source_start,
                source_end=source_end,
                normalized_start=index,
            )
        )
    return tuple(occurrences)


def _source_span_for_normalized_range(
    source_spans: tuple[tuple[int, int], ...],
    *,
    start: int,
    end: int,
) -> tuple[int, int]:
    return source_spans[start][0], source_spans[end - 1][1]


def _clock_spans(text: str) -> tuple[_ClockSpan, ...]:
    meridiem_spans = tuple(
        _ClockSpan(start=match.start(), end=match.end())
        for match in _MERIDIEM_CLOCK_RE.finditer(text)
    )
    spans = list(meridiem_spans)
    for match in _CLOCK_24H_RE.finditer(text):
        if any(
            match.start() < meridiem.end and match.end() > meridiem.start
            for meridiem in meridiem_spans
        ):
            continue
        spans.append(_ClockSpan(start=match.start(), end=match.end()))
    return tuple(sorted(spans, key=lambda span: (span.start, span.end)))


def _readout_value_spans(text: str) -> tuple[_ClockSpan, ...]:
    spans = list(_clock_spans(text))
    spans.extend(
        _ClockSpan(start=match.start(), end=match.end())
        for match in _READOUT_DURATION_RE.finditer(text)
    )
    return tuple(sorted(spans, key=lambda span: (span.start, span.end)))


def _timer_readout_near_context(text: str) -> bool:
    readout_spans = _readout_value_spans(text)
    if not readout_spans:
        return False
    for match in readout_spans:
        if _clock_has_timer_context(
            text,
            start=match.start,
            end=match.end,
        ):
            return True
    return False


def _clock_has_timer_context(text: str, *, start: int, end: int) -> bool:
    prefix = text[max(0, start - 48) : start]
    suffix = text[end : end + 96]
    return _TIMER_PREFIX_RE.search(prefix) is not None or (
        _TIMER_SUFFIX_RE.search(suffix) is not None
    )


def _without_timer_readout_clauses(text: str) -> str:
    spans: list[tuple[int, int]] = []
    for match in _readout_value_spans(text):
        if _clock_has_timer_context(text, start=match.start, end=match.end):
            spans.append(
                (
                    _timer_clause_start(text, match.start),
                    _timer_clause_end(text, match.end),
                )
            )
    if not spans:
        return text
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        if start > cursor:
            parts.append(text[cursor:start])
        cursor = max(cursor, end)
    parts.append(text[cursor:])
    return " ".join(part.strip() for part in parts if part.strip())


def _span_overlaps_timer_clause(text: str, *, start: int, end: int) -> bool:
    for match in _readout_value_spans(text):
        if not _clock_has_timer_context(text, start=match.start, end=match.end):
            continue
        clause_start = _timer_clause_start(text, match.start)
        clause_end = _timer_clause_end(text, match.end)
        if start < clause_end and end > clause_start:
            return True
    return False


def _timer_clause_start(text: str, end: int) -> int:
    boundary = 0
    for match in re.finditer(
        r"(?:^|[.!?;]\s*|,\s*|\s+(?:and|then|so|while|as|when)\s+)",
        text[:end],
    ):
        boundary = match.end()
    return boundary


def _timer_clause_end(text: str, start: int) -> int:
    match = re.search(
        r"(?:[.!?;]|,\s*|"
        r"\s+(?=(?:(?:i|we|they|he|she|[a-z][\w'-]*)\s+)?"
        r"(?:wait|waits|waited|waiting|rest|rests|rested|resting|sleep|"
        r"sleeps|slept|sleeping|stay|stays|stayed|staying|skip|skips|"
        r"skipped|go|goes|went|head|heads|headed|travel|travels|traveled|"
        r"travelled|walk|walks|walked|drive|drives|drove|ride|rides|rode|"
        r"return|returns|returned|arrive|arrives|arrived|leave|leaves|left|"
        r"regroup|regroups|regrouped|meet|meets|met)\b)|"
        r"\s+(?:and|then|so|while|as|when)\s+"
        r"(?:(?:(?:i|we|they|he|she|[a-z][\w'-]*)\s+)?"
        r"(?=(?:wait|waits|waited|waiting|rest|rests|rested|resting|sleep|"
        r"sleeps|slept|sleeping|stay|stays|stayed|staying|skip|skips|"
        r"skipped|go|goes|went|head|heads|headed|travel|travels|traveled|"
        r"travelled|walk|walks|walked|drive|drives|drove|ride|rides|rode|"
        r"return|returns|returned|arrive|arrives|arrived|leave|leaves|left|"
        r"regroup|regroups|regrouped|meet|meets|met)\b)|"
        r"(?=(?:at|around|before|after|by|until|later|overnight|next|"
        r"the\s+next|following)\b)))",
        text[start:],
    )
    if match is None:
        return len(text)
    return start + match.end()


def _has_clear_non_timer_time_advance(text: str) -> bool:
    return any(
        re.search(pattern, text) is not None
        for pattern in _CLEAR_NON_TIMER_ADVANCE_PATTERNS
    )


def _has_clear_non_clock_time_advance(text: str) -> bool:
    without_clocks = _CLOCK_24H_RE.sub("", _MERIDIEM_CLOCK_RE.sub("", text))
    return _has_clear_non_timer_time_advance(without_clocks)


def _source_clock_is_clear_non_timer_advance(
    text: str,
    *,
    start: int,
    end: int,
) -> bool:
    prefix = text[max(0, start - 96) : start]
    suffix = text[end : end + 96]
    if _TIMER_SUFFIX_RE.search(suffix) is not None:
        return False
    return any(
        re.search(pattern, prefix) is not None
        for pattern in (
            r"\b(?:wait|waits|waited|waiting|rest|rests|rested|resting|sleep|"
            r"sleeps|slept|sleeping|stay|stays|stayed|staying)\s+"
            r"(?:until|for|through|to|past)\s*$",
            rf"\b(?:{_TRAVEL_VERB_ANCHOR})\b"
            r".{0,80}\b(?:at|by|until|after|around|before)\s*$",
            r"\b(?:at|around|before|after|by|until)\s*$",
        )
    )
