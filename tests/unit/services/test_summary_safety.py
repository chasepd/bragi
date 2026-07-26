from __future__ import annotations

import pytest

from bragi.persistence.models import MessageRecord, SummaryRecord
from bragi.services.summary_safety import (
    summary_has_continuation_risk,
    summary_overlaps_recent_window,
    validate_summary_output,
)


def test_validate_summary_output_accepts_third_person_factual_summary() -> None:
    result = validate_summary_output(
        "Mara crossed the ash bridge, marked the oath bell, and left a sigil."
    )

    assert result.accepted is True
    assert result.reason == ""


def test_validate_summary_output_leaves_content_rating_to_safety_agent() -> None:
    result = validate_summary_output(
        "Mara and the stranger had sex, then returned to the bridge."
    )

    assert result.accepted is True
    assert result.reason == ""


def test_validate_summary_output_rejects_low_compression_source_copy() -> None:
    source_body = (
        "Mara studies the bridge, records the ash marks, and listens for the "
        "hidden bell beneath the stones. "
        * 8
    )
    summary_body = source_body[: int(len(source_body) * 0.8)]

    result = validate_summary_output(
        summary_body,
        covered_messages=(_message("m1", source_body),),
    )

    assert result.accepted is False
    assert result.reason == (
        "summary rejected as continuation-risk low-compression output"
    )


def test_validate_summary_output_rejects_retained_narrator_overlap() -> None:
    retained_body = (
        "The bridge bell hums beneath the ash stones while Mara maps each "
        "old oath mark along the parapet. "
        * 2
    )

    result = validate_summary_output(
        (
            f"{retained_body} Mara later notes that the eastern span remains "
            "quiet after sunset."
        ),
        retained_recent_messages=(_message("m1", retained_body, role="narrator"),),
    )

    assert result.accepted is False
    assert result.reason == (
        "summary rejected as continuation-risk retained narrator overlap"
    )


def test_validate_summary_output_rejects_unsupported_new_action() -> None:
    source_body = (
        "Mara studies bridge archive cinders lantern sigil compass threshold "
        "parapet bell oath keeper shrine marker map silver route dusk ember "
        "stone ledger crossing witness."
    )
    summary_body = (
        "Mara opens crimson vault beneath harbor moonlit stairwell, answers "
        "captain oracle riddle, reaches hidden engine, and takes royal crown."
    )

    result = validate_summary_output(
        summary_body,
        covered_messages=(_message("m1", source_body),),
    )

    assert result.accepted is False
    assert result.reason == (
        "summary rejected as continuation-risk unsupported new action"
    )


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("Mara records the ash marks in the bridge ledger.", False),
        ("What do you do next?", True),
    ],
)
def test_summary_has_continuation_risk_wraps_validation(
    body: str,
    expected: bool,
) -> None:
    assert summary_has_continuation_risk(body) is expected


def test_summary_overlaps_recent_window_detects_recent_covered_messages() -> None:
    messages = [_message(f"m{index}", f"Beat {index}.") for index in range(5)]
    summary = _summary(
        start_id=messages[1].id,
        end_id=messages[3].id,
    )

    assert (
        summary_overlaps_recent_window(
            summary,
            messages=messages,
            recent_message_limit=2,
        )
        is True
    )


def test_summary_overlaps_recent_window_handles_reversed_start_and_end_ids() -> None:
    messages = [_message(f"m{index}", f"Beat {index}.") for index in range(5)]
    summary = _summary(
        start_id=messages[3].id,
        end_id=messages[1].id,
    )

    assert (
        summary_overlaps_recent_window(
            summary,
            messages=messages,
            recent_message_limit=2,
        )
        is True
    )


@pytest.mark.parametrize(
    ("start_id", "end_id"),
    [
        ("missing", "m2"),
        ("m1", "missing"),
    ],
)
def test_summary_overlaps_recent_window_ignores_missing_message_ids(
    start_id: str,
    end_id: str,
) -> None:
    messages = [_message(f"m{index}", f"Beat {index}.") for index in range(4)]
    summary = _summary(start_id=start_id, end_id=end_id)

    assert (
        summary_overlaps_recent_window(
            summary,
            messages=messages,
            recent_message_limit=2,
        )
        is False
    )


@pytest.mark.parametrize("recent_message_limit", [0, -1])
def test_summary_overlaps_recent_window_ignores_empty_or_disabled_recent_window(
    recent_message_limit: int,
) -> None:
    messages = [_message(f"m{index}", f"Beat {index}.") for index in range(3)]
    summary = _summary(start_id="m1", end_id="m2")

    assert (
        summary_overlaps_recent_window(
            summary,
            messages=messages,
            recent_message_limit=recent_message_limit,
        )
        is False
    )


def test_summary_overlaps_recent_window_ignores_empty_messages() -> None:
    summary = _summary(start_id="m1", end_id="m2")

    assert (
        summary_overlaps_recent_window(
            summary,
            messages=[],
            recent_message_limit=2,
        )
        is False
    )


def _message(
    message_id: str,
    body: str,
    *,
    role: str = "player",
) -> MessageRecord:
    return MessageRecord(
        id=message_id,
        save_id="save-1",
        role=role,
        body=body,
        speaker_name="Narrator" if role == "narrator" else "Mara",
        provider="fake" if role == "narrator" else None,
        model="fake-model" if role == "narrator" else None,
        token_estimate=None,
    )


def _summary(
    *,
    start_id: str,
    end_id: str,
) -> SummaryRecord:
    return SummaryRecord(
        id="summary-1",
        save_id="save-1",
        covers_message_start_id=start_id,
        covers_message_end_id=end_id,
        body="Mara recorded the ash marks near the bridge.",
        provider="fake",
        model="fake-summary",
    )
