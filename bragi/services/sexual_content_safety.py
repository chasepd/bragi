"""Deterministic application-side safety for saved narrator prose."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

from bragi.safety import (
    CONTENT_FILTER_TRANSITION,
    CONTENT_FILTER_TRANSITION_KIND,
    FADE_TO_BLACK_TRANSITION,
    FADE_TO_BLACK_TRANSITION_KIND,
)
from bragi.services.content_rating import (
    CONTENT_RATING_R,
    CONTENT_RATING_UNRATED,
    DEFAULT_ADULT_CONTENT_RATING,
    content_exceeds_rating,
    sanitize_content_rating,
)


class SexualContentClassification(StrEnum):
    """The deterministic safety class assigned to narrator prose."""

    ACCEPTABLE_ROMANCE = "acceptable_romance"
    FADE_NEEDED_SEXUAL_ESCALATION = "fade_needed_sexual_escalation"
    EXPLICIT_DISALLOWED = "explicit_disallowed"

    ACCEPTABLE = ACCEPTABLE_ROMANCE
    FADE_NEEDED = FADE_NEEDED_SEXUAL_ESCALATION
    EXPLICIT = EXPLICIT_DISALLOWED


@dataclass(frozen=True)
class SexualContentSafetyResult:
    """Safe body and non-sensitive classification for a narrator completion."""

    body: str
    classification: SexualContentClassification
    transition_applied: bool


_NON_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")

_EXPLICIT_PATTERNS = (
    re.compile(r"\bthrust(?:s|ed|ing)?\s+(?:into|inside)\b"),
    re.compile(r"\bpenetrat(?:e|ed|es|ing|ion)\b"),
    re.compile(r"\bejaculat(?:e|ed|es|ing|ion)\b"),
    re.compile(r"\bmasturbat(?:e|ed|es|ing|ion)\b"),
    re.compile(r"\b(?:orgasm(?:s)?|climax(?:es|ed|ing)?)\b"),
    re.compile(r"\b(?:cock|dick|pussy)\b"),
    re.compile(
        r"\bcame\s+(?:(?:inside|on)\s+|(?:all\s+)?over\s+)"
        r"(?:him|her|them)\b"
    ),
    re.compile(
        r"\bforc(?:e|es|ed|ing)\s+"
        r"(?:himself|herself|themself|themselves)\s+"
        r"(?:inside|into)\s+(?:him|her|them)\b"
    ),
    re.compile(r"\b(?:penis|vagina)\s+(?:entered|penetrated|thrust)\b"),
    re.compile(
        r"\b(?:had|has|have|having|began|begin|begins|beginning|started|starts|"
        r"starting|engaged|engages|engaging)\s+(?:in\s+)?"
        r"(?:\w+\s+){0,3}sex\b"
    ),
    re.compile(r"\b(?:sex|sexual)\s+acts?\b"),
    re.compile(r"\b(?:sexual\s+)?intercourse\b"),
    re.compile(r"\bcopulat(?:e|es|ed|ing|ion)\b"),
    re.compile(
        r"\b(?:had|have|having|performed|performing|gave|giving|described|"
        r"describes)\s+(?:graphic\s+)?"
        r"(?:oral sex|anal sex|blow ?job|fellatio|cunnilingus)\b"
    ),
)

_FADE_PATTERNS = (
    re.compile(r"\bhands?\s+(?:slipped|slid)\s+(?:beneath|under|inside)\b"),
    re.compile(
        r"\b(?:slipped|slid|reached|moved)\s+(?:their|his|her|the|my|your)?\s*"
        r"hands?\s+(?:beneath|under|inside)\b"
    ),
    re.compile(
        r"\b(?:slipped|slid|reached|moved)\s+"
        r"(?:beneath|under|inside)\s+(?:him|her|them|me|you)\b"
    ),
    re.compile(
        r"\b(?:entered|moved inside|went inside)\s+"
        r"(?:him|her|them)\s+(?:slowly|deeply|fully)\b"
    ),
    re.compile(r"\b(?:undressed|disrobed)\b"),
    re.compile(
        r"\b(?:unbuttoned|unzipped|removed|took off|pulled off)\s+"
        r"(?:his|her|their|my|your|the)?\s*"
        r"(?:clothes?|shirt|dress|pants|lingerie)\b"
    ),
    re.compile(
        r"\b(?:guided|placed|lowered|slid)\s+(?:his|her|their|my|your)\s+"
        r"hands?\s+(?:beneath|under|inside)\b"
    ),
    re.compile(
        r"\b(?:touched|caressed|stroked)\s+(?:his|her|their|my|your|the)?\s*"
        r"(?:breasts?|groin|crotch|inner thighs?|between)\b"
    ),
    re.compile(r"\b(?:made|make) love\b"),
    re.compile(r"\b(?:slept|sleep) together\b"),
    re.compile(r"\bgrinding\s+(?:against|into)\b"),
)


def classify_sexual_content(value: str) -> SexualContentClassification:
    """Classify narrator prose without consulting a provider or prompt text."""

    normalized = _normalize_for_matching(value)
    if any(pattern.search(normalized) for pattern in _EXPLICIT_PATTERNS):
        return SexualContentClassification.EXPLICIT
    if any(pattern.search(normalized) for pattern in _FADE_PATTERNS):
        return SexualContentClassification.FADE_NEEDED
    return SexualContentClassification.ACCEPTABLE


def sanitize_narrator_body(
    value: str,
    *,
    content_rating: str = DEFAULT_ADULT_CONTENT_RATING,
    fade_to_black_enabled: bool = True,
) -> SexualContentSafetyResult:
    """Replace sexual escalation with the fixed, non-sensitive transition."""

    classification = classify_sexual_content(value)
    rating_exceeded = content_exceeds_rating(
        value,
        allowed_rating=content_rating,
    )
    sexual_transition = (
        rating_exceeded or fade_to_black_enabled
    ) and classification is not SexualContentClassification.ACCEPTABLE
    transition_applied = rating_exceeded or (
        fade_to_black_enabled
        and classification is not SexualContentClassification.ACCEPTABLE
    )
    return SexualContentSafetyResult(
        body=(
            FADE_TO_BLACK_TRANSITION
            if sexual_transition
            else CONTENT_FILTER_TRANSITION
            if transition_applied
            else value
        ),
        classification=classification,
        transition_applied=transition_applied,
    )


def validate_image_prompt(value: str, *, allowed_rating: str | None = None) -> str:
    """Reject intimate image prompts before they reach jobs or providers."""

    sexual_classification = classify_sexual_content(value)
    normalized_allowed_rating = (
        sanitize_content_rating(allowed_rating, default="")
        if allowed_rating is not None
        else None
    )
    if (
        sexual_classification is not SexualContentClassification.ACCEPTABLE
        and normalized_allowed_rating
        not in {CONTENT_RATING_R, CONTENT_RATING_UNRATED}
    ):
        if allowed_rating is not None:
            raise ValueError("Image prompt exceeds the selected content rating")
        raise ValueError("Image prompts cannot contain intimate sexual content")
    if allowed_rating is not None and content_exceeds_rating(
        value,
        allowed_rating=allowed_rating,
    ):
        raise ValueError("Image prompt exceeds the selected content rating")
    return value


def is_fade_to_black_message(
    *,
    role: str,
    body: str,
    safety_transition: str = "",
) -> bool:
    """Return whether a narrator message is any persisted safety transition.

    The legacy name remains for compatibility with downstream guards that must
    skip both intimate fades and neutral content-filter placeholders.
    """

    return role == "narrator" and (
        safety_transition
        in {FADE_TO_BLACK_TRANSITION_KIND, CONTENT_FILTER_TRANSITION_KIND}
        or (
            not safety_transition
            and body in {FADE_TO_BLACK_TRANSITION, CONTENT_FILTER_TRANSITION}
        )
    )


def _normalize_for_matching(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return _WHITESPACE_RE.sub(" ", _NON_WORD_RE.sub(" ", normalized)).strip()
