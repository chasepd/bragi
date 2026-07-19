from __future__ import annotations

import pytest

from bragi.services.scene_snapshot_locks import scene_snapshot_field_is_locked


@pytest.mark.parametrize(
    ("locked_fields", "field_path"),
    [
        (["world_time_phase"], "in_world_time"),
        (["world_time_day_label"], "in_world_time"),
        (["world_time_day_index"], "in_world_time"),
        (["time_of_day"], "in_world_time"),
        (["in_world_time"], "time_of_day"),
        (["world_time_phase"], "time_of_day"),
        (["in_world_time"], "day_of_week"),
        (["world_time_day_label"], "day_of_week"),
        (["in_world_time"], "world_time_clock_minutes"),
        (["in_world_time"], "world_time_period_label"),
    ],
)
def test_scene_snapshot_field_is_locked_matches_world_time_aliases(
    locked_fields: list[str],
    field_path: str,
) -> None:
    assert scene_snapshot_field_is_locked(locked_fields, field_path)


def test_scene_snapshot_field_is_locked_ignores_unrelated_aliases() -> None:
    assert not scene_snapshot_field_is_locked(
        ["world_time_clock_minutes"],
        "time_of_day",
    )
    assert not scene_snapshot_field_is_locked(["mood"], "time_of_day")
