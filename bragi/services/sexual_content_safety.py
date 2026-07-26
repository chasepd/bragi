"""Compatibility helpers for persisted content-safety transitions."""

from __future__ import annotations

from bragi.safety import (
    CONTENT_FILTER_TRANSITION,
    CONTENT_FILTER_TRANSITION_KIND,
    FADE_TO_BLACK_TRANSITION,
    FADE_TO_BLACK_TRANSITION_KIND,
)


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
