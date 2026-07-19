"""Deterministic guards for rolling summaries before they become context."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from re import sub

from bragi.persistence.models import MessageRecord, SummaryRecord
from bragi.services.sexual_content_safety import classify_sexual_content

_DIRECT_PROMPT_PATTERN = re.compile(
    r"\b(?:what do you|what will you|how do you|do you|will you|can you|"
    r"are you|your move|what next|what happens next)\b",
    re.IGNORECASE,
)
_SPEAKER_LABEL_PATTERN = re.compile(r"(?m)^\s*[A-Z][A-Za-z0-9 '_-]{0,32}\s*:")
_FIRST_SECOND_PERSON_PATTERN = re.compile(
    r"\b(?:I|me|my|mine|we|our|ours|you|your|yours)\b",
    re.IGNORECASE,
)
_ACTION_MARKERS = frozenset(
    {
        "answers",
        "asks",
        "continues",
        "finds",
        "hands",
        "opens",
        "reaches",
        "says",
        "steps",
        "takes",
        "tells",
        "touches",
        "walks",
    }
)
_STOPWORDS = frozenset(
    {
        "and",
        "are",
        "but",
        "for",
        "from",
        "has",
        "have",
        "her",
        "him",
        "his",
        "into",
        "not",
        "she",
        "that",
        "the",
        "their",
        "then",
        "there",
        "they",
        "this",
        "was",
        "were",
        "with",
    }
)


@dataclass(frozen=True)
class SummaryValidationResult:
    accepted: bool
    reason: str = ""


def validate_summary_output(
    body: str,
    *,
    covered_messages: Iterable[MessageRecord] = (),
    retained_recent_messages: Iterable[MessageRecord] = (),
    prior_summaries: Iterable[SummaryRecord] = (),
) -> SummaryValidationResult:
    text = _compact_text(body)
    if not text:
        return SummaryValidationResult(False, "summary is empty")
    if classify_sexual_content(text).value != "acceptable_romance":
        return SummaryValidationResult(
            False,
            "summary rejected because sexual detail must remain off-screen",
        )
    if _looks_like_direct_prompt(text):
        return SummaryValidationResult(
            False,
            "summary rejected as continuation-risk direct prompt",
        )
    if _looks_like_dialogue_beat(text):
        return SummaryValidationResult(
            False,
            "summary rejected as continuation-risk dialogue beat",
        )

    source_text = _compact_text(
        " ".join(
            (
                *(message.body for message in covered_messages),
                *(summary.body for summary in prior_summaries),
            )
        )
    )
    if _is_low_compression(text, source_text):
        return SummaryValidationResult(
            False,
            "summary rejected as continuation-risk low-compression output",
        )
    if _has_high_retained_narrator_overlap(text, tuple(retained_recent_messages)):
        return SummaryValidationResult(
            False,
            "summary rejected as continuation-risk retained narrator overlap",
        )
    if _has_unsupported_new_action(text, source_text):
        return SummaryValidationResult(
            False,
            "summary rejected as continuation-risk unsupported new action",
        )
    return SummaryValidationResult(True)


def summary_has_continuation_risk(body: str) -> bool:
    return not validate_summary_output(body).accepted


def summary_overlaps_recent_window(
    summary: SummaryRecord,
    *,
    messages: Iterable[MessageRecord],
    recent_message_limit: int,
) -> bool:
    ordered_messages = tuple(messages)
    if recent_message_limit <= 0 or not ordered_messages:
        return False
    positions = {message.id: index for index, message in enumerate(ordered_messages)}
    start = positions.get(summary.covers_message_start_id)
    end = positions.get(summary.covers_message_end_id)
    if start is None or end is None:
        return False
    if start > end:
        start, end = end, start
    recent_start = max(0, len(ordered_messages) - recent_message_limit)
    return end >= recent_start and start < len(ordered_messages)


def _looks_like_direct_prompt(text: str) -> bool:
    if _DIRECT_PROMPT_PATTERN.search(text):
        return True
    return text.rstrip().endswith("?") and bool(
        _FIRST_SECOND_PERSON_PATTERN.search(text)
    )


def _looks_like_dialogue_beat(text: str) -> bool:
    if _SPEAKER_LABEL_PATTERN.search(text):
        return True
    if '"' in text and _FIRST_SECOND_PERSON_PATTERN.search(text):
        return True
    first_words = " ".join(text.split()[:8])
    return bool(_FIRST_SECOND_PERSON_PATTERN.search(first_words))


def _is_low_compression(summary_text: str, source_text: str) -> bool:
    if len(source_text) < 600:
        return False
    return len(summary_text) >= int(len(source_text) * 0.75)


def _has_high_retained_narrator_overlap(
    summary_text: str,
    retained_recent_messages: tuple[MessageRecord, ...],
) -> bool:
    summary_shingles = _word_shingles(summary_text)
    if not summary_shingles:
        return False
    for message in retained_recent_messages:
        if message.role != "narrator":
            continue
        message_text = _compact_text(message.body)
        if len(message_text) >= 80 and message_text[:160] in summary_text:
            return True
        message_shingles = _word_shingles(message_text)
        if not message_shingles:
            continue
        overlap = len(summary_shingles & message_shingles) / len(
            summary_shingles | message_shingles
        )
        if overlap >= 0.45:
            return True
    return False


def _has_unsupported_new_action(summary_text: str, source_text: str) -> bool:
    if not source_text:
        return False
    source_terms = _meaningful_terms(source_text)
    summary_terms = _meaningful_terms(summary_text)
    if len(source_terms) < 20 or len(summary_terms) < 10:
        return False
    novel_terms = summary_terms - source_terms
    if len(novel_terms) < 8:
        return False
    if not novel_terms & _ACTION_MARKERS:
        return False
    return len(novel_terms) / max(1, len(summary_terms)) >= 0.45


def _word_shingles(text: str, size: int = 5) -> set[tuple[str, ...]]:
    words = [
        word
        for word in re.findall(r"[a-z0-9']+", text.casefold())
        if word not in _STOPWORDS
    ]
    if len(words) < size:
        return set()
    return {
        tuple(words[index : index + size]) for index in range(len(words) - size + 1)
    }


def _meaningful_terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9']{3,}", text.casefold())
        if term not in _STOPWORDS
    }


def _compact_text(text: str) -> str:
    return sub(r"\s+", " ", text.strip())
