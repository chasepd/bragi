from __future__ import annotations

from dataclasses import dataclass

import pytest

from bragi.services.sexual_content_safety import FADE_TO_BLACK_TRANSITION
from bragi.services.world_time_signals import (
    has_world_time_advance_signal,
    timer_readout_evidence_without_clock_advance,
    timer_readout_without_clock_advance,
)


@dataclass(frozen=True)
class WorldTimeSignalCase:
    case_id: str
    text: str
    expected_status: str


def test_fade_transition_preserves_hours_later_world_time_signal() -> None:
    assert has_world_time_advance_signal(FADE_TO_BLACK_TRANSITION)


WORLD_TIME_SIGNAL_CORPUS = (
    WorldTimeSignalCase(
        case_id="explicit_wait_until_evening",
        text="We wait until evening.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="sleep_until_tomorrow",
        text="I sleep until tomorrow morning.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="travel_consumes_afternoon",
        text="We spend the afternoon traveling to the observatory.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="travel_for_duration",
        text="We travel for two hours to the observatory.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="travel_for_couple_duration",
        text="We travel for a couple of hours to the observatory.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="travel_to_place_for_duration",
        text="We travel to the observatory for two hours.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="after_meeting_travel",
        text="I head home after the meeting.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="go_to_class",
        text="I go to class.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="head_to_work",
        text="I head to work.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="walk_to_dinner",
        text="We walk to dinner.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="drive_to_breakfast",
        text="We drive to breakfast.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="pass_hours_preparing",
        text="We pass a few hours preparing the ward stones.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="wait_direct_duration",
        text="We wait two hours.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="wait_fractional_duration",
        text="We wait half an hour.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="rest_direct_duration",
        text="We rest a few hours.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="sleep_direct_duration",
        text="We sleep eight hours.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="clock_time_with_meridiem",
        text="I wait until 3:45 p.m.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="twenty_four_hour_clock",
        text="We regroup at 14:30.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="after_twenty_four_hour_clock",
        text="After 14:30, we leave.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="clock_with_remaining_prose",
        text="We wait until 14:30, remaining quiet until the guards leave.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="wait_before_timer_readout",
        text="I wait until evening while the countdown timer shows 03:45 remaining.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="wait_after_timer_readout_while_clause",
        text="The countdown timer shows 03:45 remaining while I wait until evening.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="left_at_twenty_four_hour_clock",
        text="I left at 14:30.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="quarter_location_clock",
        text="We meet in the northern quarter at 14:30.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="half_hour_until_clock",
        text="We wait half an hour until 14:30.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="clock_in_half_light",
        text="We meet at 14:30 in the half-light.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="clock_in_quarter_location",
        text="We meet at 14:30 in the quarter.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="overnight_transition",
        text="We keep watch overnight and leave at dawn.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="by_evening",
        text="By evening, we hide.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="leading_at_dawn",
        text="At dawn, I check the beacon.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="leading_at_night",
        text="At night, I return to the tower.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="leading_at_dinner",
        text="At dinner, I ask about the route.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="leading_at_work_location",
        text="At work, I sharpen the blade.",
        expected_status="skipped",
    ),
    WorldTimeSignalCase(
        case_id="next_day_transition",
        text="The next day, I check the beacon again.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="later_that_night",
        text="Later that night, I return to the tower.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="time_loop_reset",
        text="When the loop resets to Monday morning, I write down what changed.",
        expected_status="applied",
    ),
    WorldTimeSignalCase(
        case_id="decorative_evening",
        text="I look at the evening lanterns.",
        expected_status="skipped",
    ),
    WorldTimeSignalCase(
        case_id="decorative_at_evening",
        text="I look at evening lanterns.",
        expected_status="skipped",
    ),
    WorldTimeSignalCase(
        case_id="location_at_work",
        text="I look around at work.",
        expected_status="skipped",
    ),
    WorldTimeSignalCase(
        case_id="left_object_at_work",
        text="I left my notebook at work.",
        expected_status="skipped",
    ),
    WorldTimeSignalCase(
        case_id="ambiguous_later",
        text="Later, I ask whether anyone heard the bell.",
        expected_status="skipped",
    ),
    WorldTimeSignalCase(
        case_id="ambiguous_soon",
        text="Soon, I should probably check the gate.",
        expected_status="skipped",
    ),
    WorldTimeSignalCase(
        case_id="wait_to_see",
        text="I wait to see what happens.",
        expected_status="skipped",
    ),
    WorldTimeSignalCase(
        case_id="countdown_timer",
        text="I wait while the countdown timer shows 03:45 left in the round.",
        expected_status="skipped",
    ),
    WorldTimeSignalCase(
        case_id="clock_on_countdown_timer",
        text="I look at 14:30 on the countdown timer.",
        expected_status="skipped",
    ),
    WorldTimeSignalCase(
        case_id="travel_to_timer_target",
        text="We travel until 03:45 remaining on the countdown timer.",
        expected_status="skipped",
    ),
    WorldTimeSignalCase(
        case_id="skip_to_timer_target",
        text="Skip ahead to 03:45 remaining on the countdown timer.",
        expected_status="skipped",
    ),
    WorldTimeSignalCase(
        case_id="punctuated_timer_target",
        text="I wait until 03:45, remaining on the countdown timer.",
        expected_status="skipped",
    ),
    WorldTimeSignalCase(
        case_id="timer_target_with_intervening_words",
        text=(
            "I wait until 03:45, with several seconds remaining on the "
            "countdown timer."
        ),
        expected_status="skipped",
    ),
    WorldTimeSignalCase(
        case_id="timer_is_left_target",
        text="I wait until 03:45 is left on the timer.",
        expected_status="skipped",
    ),
    WorldTimeSignalCase(
        case_id="timer_left_to_go_target",
        text="I wait until 03:45 left to go on the timer.",
        expected_status="skipped",
    ),
    WorldTimeSignalCase(
        case_id="timer_remaining_until_phase",
        text="The countdown timer has 03:45 remaining until dawn.",
        expected_status="skipped",
    ),
    WorldTimeSignalCase(
        case_id="timer_remaining_by_phase",
        text="The countdown timer has 03:45 remaining by evening.",
        expected_status="skipped",
    ),
    WorldTimeSignalCase(
        case_id="timer_remaining_until_clock",
        text="The countdown timer has 03:45 remaining until 14:30.",
        expected_status="skipped",
    ),
    WorldTimeSignalCase(
        case_id="elapsed_timer",
        text="The elapsed timer reads 11:20.",
        expected_status="skipped",
    ),
    WorldTimeSignalCase(
        case_id="game_clock_readout",
        text="I wait until the game clock shows 03:45.",
        expected_status="skipped",
    ),
    WorldTimeSignalCase(
        case_id="scoreboard_readout",
        text="The scoreboard reads 03:45.",
        expected_status="skipped",
    ),
    WorldTimeSignalCase(
        case_id="shot_clock_readout",
        text="I wait until the shot clock hits 00:12.",
        expected_status="skipped",
    ),
    WorldTimeSignalCase(
        case_id="shot_clock_seconds_readout",
        text="I wait until the shot clock hits 12 seconds.",
        expected_status="skipped",
    ),
    WorldTimeSignalCase(
        case_id="play_clock_seconds_readout",
        text="The play clock shows 12 seconds remaining.",
        expected_status="skipped",
    ),
    WorldTimeSignalCase(
        case_id="game_clock_no_minute_meridiem_readout",
        text="The game clock shows 3 p.m.",
        expected_status="skipped",
    ),
    WorldTimeSignalCase(
        case_id="round_corner_clock",
        text="We round the corner at 14:30.",
        expected_status="applied",
    ),
)


@pytest.mark.parametrize(
    "case",
    WORLD_TIME_SIGNAL_CORPUS,
    ids=lambda case: case.case_id,
)
def test_world_time_signal_corpus_records_expected_status(
    case: WorldTimeSignalCase,
) -> None:
    assert case.expected_status in {"applied", "queued", "skipped", "failed"}
    assert has_world_time_advance_signal(case.text) is (
        case.expected_status != "skipped"
    )


@pytest.mark.parametrize(
    "case",
    [
        case
        for case in WORLD_TIME_SIGNAL_CORPUS
        if case.expected_status == "applied"
    ],
    ids=lambda case: case.case_id,
)
def test_clear_world_time_signals_survive_unrelated_timer_readout(
    case: WorldTimeSignalCase,
) -> None:
    assert has_world_time_advance_signal(
        f"The countdown timer shows 03:45 remaining, so {case.text}",
    )


@pytest.mark.parametrize(
    "text",
    (
        "I wait until 03:45 remains on the countdown timer.",
        "We wait until the timer hits 03:45 left.",
        "The round has 03:45 remaining.",
        "I wait until the game clock shows 03:45.",
        "The scoreboard reads 03:45.",
        "I wait until the shot clock hits 00:12.",
        "I wait until the shot clock hits 12 seconds.",
        "The play clock shows 12 seconds remaining.",
        "The game clock shows 3 p.m.",
    ),
)
def test_timer_readout_without_clock_advance_rejects_timer_values(
    text: str,
) -> None:
    assert timer_readout_without_clock_advance(text)


@pytest.mark.parametrize(
    "text",
    (
        "We wait until 3:45 p.m.",
        "We arrive at 03:45.",
        "The countdown timer shows 03:45 remaining, so I wait until 14:30.",
        "The countdown timer shows 03:45 remaining, so at 14:30 we leave.",
        "The countdown timer shows 03:45 remaining, so by 14:30 we leave.",
        "The countdown timer shows 03:45 remaining, so until 14:30 we rest.",
        "The countdown timer shows 03:45 remaining, so after 14:30 we leave.",
        "The countdown timer shows 03:45 remaining, so we travel until dawn.",
        "The countdown timer shows 03:45 remaining, so I wait until evening.",
        "The timer shows 03:45 remaining. By evening, we hide.",
        "The countdown timer shows 03:45 remaining and I wait until evening.",
        "The countdown timer shows 03:45 remaining then by evening we hide.",
        "The countdown timer shows 03:45 remaining and at 14:30 we leave.",
        "The countdown timer shows 03:45 remaining I wait until evening.",
        "The timer shows 03:45 remaining and Mara waits until evening.",
        "The countdown timer shows 03:45 remaining. At dawn, we leave.",
        "The timer shows 03:45 remaining. At night, we leave.",
        "The timer shows 03:45 remaining. Around evening, we leave.",
        "The game clock shows 03:45, so I wait until evening.",
        "The scoreboard reads 03:45, so at 14:30 we leave.",
        "The game clock shows 3 p.m., so I wait until evening.",
        "I wait until evening while the countdown timer shows 03:45 remaining.",
        "The countdown timer shows 03:45 remaining while I wait until evening.",
        "We meet in the northern quarter at 14:30.",
        "We wait half an hour until 14:30.",
        "We meet at 14:30 in the half-light.",
        "We meet at 14:30 in the quarter.",
    ),
)
def test_timer_readout_without_clock_advance_allows_real_clock_advances(
    text: str,
) -> None:
    assert not timer_readout_without_clock_advance(text)


@pytest.mark.parametrize(
    ("evidence_quote", "source_body", "expected"),
    (
        (
            "03:45 remaining",
            "The countdown timer shows 03:45 remaining, so I wait until evening.",
            True,
        ),
        (
            "Skip ahead to 03:45 remaining on the countdown timer.",
            "Skip ahead to 03:45 remaining on the countdown timer.",
            True,
        ),
        (
            "wait until 03:45",
            "I wait until 03:45, remaining on the countdown timer.",
            True,
        ),
        (
            "03:45",
            (
                "I wait until 03:45, with several seconds remaining on the "
                "countdown timer."
            ),
            True,
        ),
        (
            "The countdown timer has 03:45 remaining until dawn.",
            "The countdown timer has 03:45 remaining until dawn.",
            True,
        ),
        (
            "I wait until evening",
            "The countdown timer shows 03:45 remaining I wait until evening.",
            False,
        ),
        (
            "03:45 remaining until 14:30",
            "The countdown timer has 03:45 remaining until 14:30.",
            True,
        ),
        (
            "until dawn",
            (
                "The countdown timer has 03:45 remaining until dawn, so I wait "
                "until evening."
            ),
            True,
        ),
        (
            "until dawn",
            (
                "The countdown timer has 03:45 remaining until   dawn, so I "
                "wait until evening."
            ),
            True,
        ),
        (
            "until dawn",
            "The countdown timer has 03:45 remaining until dawn. I wait until dawn.",
            False,
        ),
        (
            "14:30",
            "The countdown timer has 03:45 remaining until 14:30. We leave at 14:30.",
            False,
        ),
        (
            "I wait until evening",
            "The countdown timer shows 03:45 remaining, so I wait until evening.",
            False,
        ),
        (
            "wait until 14:30",
            "The countdown timer shows 03:45 remaining, so I wait until 14:30.",
            False,
        ),
        (
            "14:30",
            "The countdown timer shows 03:45 remaining, so I wait until 14:30.",
            False,
        ),
        (
            "14:30",
            "The countdown timer shows 03:45 remaining, so by 14:30 we leave.",
            False,
        ),
        (
            "wait until 14:30, remaining quiet",
            "We wait until 14:30, remaining quiet until the guards leave.",
            False,
        ),
        (
            "14:30",
            "The countdown timer shows 03:45 remaining, so until 14:30 we rest.",
            False,
        ),
        (
            "after 14:30",
            "The countdown timer shows 03:45 remaining, so after 14:30 we leave.",
            False,
        ),
        (
            "The countdown timer shows 03:45 remaining, so at 14:30 we leave.",
            "The countdown timer shows 03:45 remaining, so at 14:30 we leave.",
            False,
        ),
        (
            "3:45 p.m.",
            "We arrive at 3:45 p.m.",
            False,
        ),
        (
            "03:45",
            "I wait until the game clock shows 03:45.",
            True,
        ),
        (
            "03:45",
            "The scoreboard reads 03:45.",
            True,
        ),
        (
            "00:12",
            "I wait until the shot clock hits 00:12.",
            True,
        ),
        (
            "3:45 p.m.",
            "The game clock shows 3:45 p.m., so I wait until evening.",
            True,
        ),
        (
            "3 p.m.",
            "The game clock shows 3 p.m., so I wait until evening.",
            True,
        ),
        (
            "3 p.m.",
            "We arrive at 3 p.m.",
            False,
        ),
        (
            "12 seconds",
            "I wait until the shot clock hits 12 seconds.",
            True,
        ),
        (
            "12 seconds",
            "We wait 12 seconds.",
            False,
        ),
    ),
)
def test_timer_readout_evidence_without_clock_advance_uses_quote_context(
    evidence_quote: str,
    source_body: str,
    expected: bool,
) -> None:
    assert (
        timer_readout_evidence_without_clock_advance(evidence_quote, source_body)
        is expected
    )
