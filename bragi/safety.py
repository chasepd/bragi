"""Dependency-neutral safety metadata helpers."""

from __future__ import annotations

FADE_TO_BLACK_TRANSITION = (
    "The intimate moment is kept off-screen. Hours later, the next scene begins."
)
FADE_TO_BLACK_TRANSITION_KIND = "fade_to_black"
CONTENT_FILTER_TRANSITION = (
    "*The harsher details stay out of view as the story moves forward.*"
)
CONTENT_FILTER_TRANSITION_KIND = "content_filter"
SUPPORTED_SAFETY_TRANSITIONS = frozenset(
    {FADE_TO_BLACK_TRANSITION_KIND, CONTENT_FILTER_TRANSITION_KIND}
)


def validate_safety_transition(value: str, *, role: str) -> str:
    """Validate the allowlisted non-sensitive message marker."""

    if value not in SUPPORTED_SAFETY_TRANSITIONS:
        if value:
            raise ValueError(f"Unsupported safety transition: {value}")
        return ""
    if role != "narrator":
        raise ValueError("Safety transitions may only be stored on narrator messages")
    return value


def normalize_message_safety(
    *,
    role: str,
    body: str,
    safety_transition: str,
) -> tuple[str, str]:
    """Keep the persisted marker and narrator body mutually consistent."""

    marker = validate_safety_transition(safety_transition, role=role)
    if role == "narrator" and (
        marker == FADE_TO_BLACK_TRANSITION_KIND
        or body == FADE_TO_BLACK_TRANSITION
    ):
        return FADE_TO_BLACK_TRANSITION, FADE_TO_BLACK_TRANSITION_KIND
    if role == "narrator" and (
        marker == CONTENT_FILTER_TRANSITION_KIND
        or body == CONTENT_FILTER_TRANSITION
    ):
        return CONTENT_FILTER_TRANSITION, CONTENT_FILTER_TRANSITION_KIND
    return body, marker
