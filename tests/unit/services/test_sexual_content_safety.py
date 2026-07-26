from __future__ import annotations

import pytest

from bragi.safety import validate_safety_transition
from bragi.services.sexual_content_safety import (
    CONTENT_FILTER_TRANSITION,
    FADE_TO_BLACK_TRANSITION,
    is_fade_to_black_message,
)


def test_canonical_transition_body_is_recognized_for_legacy_records() -> None:
    assert is_fade_to_black_message(
        role="narrator",
        body=FADE_TO_BLACK_TRANSITION,
    )
    assert is_fade_to_black_message(
        role="narrator",
        body=CONTENT_FILTER_TRANSITION,
    )
    assert not is_fade_to_black_message(
        role="player",
        body=FADE_TO_BLACK_TRANSITION,
    )


def test_safety_transition_marker_is_allowlisted_for_narrators() -> None:
    assert validate_safety_transition("fade_to_black", role="narrator") == (
        "fade_to_black"
    )
    assert validate_safety_transition("content_filter", role="narrator") == (
        "content_filter"
    )
    with pytest.raises(ValueError):
        validate_safety_transition("fade_to_black", role="player")
    with pytest.raises(ValueError):
        validate_safety_transition("content_filter", role="player")
