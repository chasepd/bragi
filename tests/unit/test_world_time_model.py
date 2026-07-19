from __future__ import annotations

from bragi.world_time_model import (
    CanonicalWorldTime,
    canonical_world_time_from_legacy,
    format_world_time,
    world_time_display,
)


def test_format_world_time_prefers_canonical_fields_over_legacy_label() -> None:
    world_time = CanonicalWorldTime(
        day_index=5,
        day_label="friday",
        phase="evening",
        legacy_label="Monday morning",
    )

    assert format_world_time(world_time) == "Friday evening; world day index 5"


def test_format_world_time_can_preserve_consistent_legacy_detail() -> None:
    world_time = CanonicalWorldTime(
        day_index=5,
        day_label="friday",
        phase="evening",
        legacy_label="Friday evening after class",
    )

    assert (
        format_world_time(world_time, include_legacy_detail=True)
        == "Friday evening after class; world day index 5"
    )


def test_format_world_time_detail_mode_rejects_stale_legacy_label() -> None:
    world_time = CanonicalWorldTime(
        day_index=5,
        day_label="friday",
        phase="evening",
        legacy_label="Monday morning after class",
    )

    assert (
        format_world_time(world_time, include_legacy_detail=True)
        == "Friday evening; world day index 5"
    )


def test_format_world_time_falls_back_to_legacy_label_without_canonical_fields(
) -> None:
    world_time = CanonicalWorldTime(legacy_label="Friday evening after class")

    assert format_world_time(world_time) == "Friday evening after class"


def test_canonical_world_time_from_legacy_backfills_clock_minutes() -> None:
    world_time = canonical_world_time_from_legacy(
        in_world_time="Friday 9:41 PM after the festival",
        time_of_day="evening",
        day_of_week="friday",
        world_day_index=5,
    )

    assert world_time == CanonicalWorldTime(
        day_index=5,
        day_label="friday",
        phase="evening",
        clock_minutes=21 * 60 + 41,
        legacy_label="Friday 9:41 PM after the festival",
    )


def test_canonical_world_time_from_legacy_reads_parenthetical_phase() -> None:
    world_time = canonical_world_time_from_legacy(
        in_world_time="14:30 (afternoon)",
    )

    assert world_time.phase == "afternoon"
    assert world_time.clock_minutes == 14 * 60 + 30
    assert world_time.legacy_label == "14:30 (afternoon)"


def test_canonical_world_time_from_legacy_prefers_late_morning_phrase() -> None:
    world_time = canonical_world_time_from_legacy(
        in_world_time="Friday late morning after study hall",
    )

    assert world_time.phase == "late_morning"


def test_world_time_display_keeps_legacy_phase_day_format() -> None:
    assert world_time_display(time_of_day="late_morning", day_of_week="tuesday") == (
        "Tuesday late morning"
    )
