"""Scenario evolution throttle and phase-gate policy helpers."""

from __future__ import annotations

from bragi.persistence.repositories import PersistenceRepositories

SCENARIO_EVOLUTION_TURN_INTERVAL_SETTING = "scenario_evolution_turn_interval"
DEFAULT_SCENARIO_EVOLUTION_TURN_INTERVAL = 8
MIN_SCENARIO_EVOLUTION_TURN_INTERVAL = 0
MAX_SCENARIO_EVOLUTION_TURN_INTERVAL = 64
_SAVE_INTERVAL_PREFIX = f"{SCENARIO_EVOLUTION_TURN_INTERVAL_SETTING}:save:"
_SCENARIO_INTERVAL_PREFIX = f"{SCENARIO_EVOLUTION_TURN_INTERVAL_SETTING}:scenario:"


def save_scenario_evolution_turn_interval_setting_key(save_id: str) -> str:
    return f"{_SAVE_INTERVAL_PREFIX}{save_id}"


def scenario_template_evolution_turn_interval_setting_key(scenario_id: str) -> str:
    return f"{_SCENARIO_INTERVAL_PREFIX}{scenario_id}"


def scenario_evolution_turn_interval(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
) -> int:
    interval = coerce_scenario_evolution_turn_interval(
        repositories.get_effective_setting(
            SCENARIO_EVOLUTION_TURN_INTERVAL_SETTING,
            save_id=save_id,
        )
    )
    if interval is not None:
        return interval
    return DEFAULT_SCENARIO_EVOLUTION_TURN_INTERVAL


def sanitize_scenario_evolution_turn_interval(value: object) -> int:
    interval = coerce_scenario_evolution_turn_interval(value)
    return (
        interval
        if interval is not None
        else DEFAULT_SCENARIO_EVOLUTION_TURN_INTERVAL
    )


def coerce_scenario_evolution_turn_interval(value: object) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return min(
        max(value, MIN_SCENARIO_EVOLUTION_TURN_INTERVAL),
        MAX_SCENARIO_EVOLUTION_TURN_INTERVAL,
    )
