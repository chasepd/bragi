"""Lock alias handling for scene snapshot fields."""

from __future__ import annotations

from collections.abc import Iterable

_ALL_WORLD_TIME_LOCK_FIELDS = frozenset(
    {
        "in_world_time",
        "time_of_day",
        "day_of_week",
        "world_day_index",
        "world_time_day_index",
        "world_time_day_label",
        "world_time_phase",
        "world_time_clock_minutes",
        "world_time_period_label",
    }
)

_SCENE_SNAPSHOT_LOCK_ALIASES: dict[str, frozenset[str]] = {
    "in_world_time": _ALL_WORLD_TIME_LOCK_FIELDS,
    "time_of_day": frozenset(
        {
            "in_world_time",
            "time_of_day",
            "world_time_phase",
        }
    ),
    "day_of_week": frozenset(
        {
            "in_world_time",
            "day_of_week",
            "world_time_day_label",
        }
    ),
    "world_day_index": frozenset(
        {
            "in_world_time",
            "world_day_index",
            "world_time_day_index",
        }
    ),
    "world_time_day_index": frozenset(
        {
            "in_world_time",
            "world_day_index",
            "world_time_day_index",
        }
    ),
    "world_time_day_label": frozenset(
        {
            "in_world_time",
            "day_of_week",
            "world_time_day_label",
        }
    ),
    "world_time_phase": frozenset(
        {
            "in_world_time",
            "time_of_day",
            "world_time_phase",
        }
    ),
    "world_time_clock_minutes": frozenset(
        {
            "in_world_time",
            "world_time_clock_minutes",
        }
    ),
    "world_time_period_label": frozenset(
        {
            "in_world_time",
            "world_time_period_label",
        }
    ),
}


def scene_snapshot_field_is_locked(
    locked_fields: Iterable[str],
    field_path: str,
) -> bool:
    """Return whether a scene field is locked, including legacy/canonical aliases."""
    locked = {str(field) for field in locked_fields}
    if field_path in locked:
        return True
    return bool(locked & _SCENE_SNAPSHOT_LOCK_ALIASES.get(field_path, frozenset()))
