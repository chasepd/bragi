"""Helpers for scenario-level action choice mode."""

from __future__ import annotations

import json
from collections.abc import Mapping

from bragi.persistence.models import ScenarioRecord

ACTION_CHOICES_ENABLED_KEY = "action_choices_enabled"
LEGACY_CHOOSE_YOUR_OWN_ADVENTURE_TYPE = "choose_your_own_adventure"


def action_choices_enabled_from_content(content: Mapping[str, object]) -> bool:
    return content.get(ACTION_CHOICES_ENABLED_KEY) is True


def scenario_action_choices_enabled(scenario: ScenarioRecord) -> bool:
    if scenario.type == LEGACY_CHOOSE_YOUR_OWN_ADVENTURE_TYPE:
        return True
    return action_choices_enabled_from_content(_scenario_content(scenario.content_json))


def content_with_action_choices_enabled(
    content: Mapping[str, object],
    *,
    enabled: bool,
) -> dict[str, object]:
    normalized = dict(content)
    if enabled:
        normalized[ACTION_CHOICES_ENABLED_KEY] = True
    else:
        normalized.pop(ACTION_CHOICES_ENABLED_KEY, None)
    return normalized


def normalize_legacy_action_choice_scenario(
    *,
    scenario_type: str,
    content: Mapping[str, object],
) -> tuple[str, dict[str, object], bool]:
    normalized = dict(content)
    if scenario_type != LEGACY_CHOOSE_YOUR_OWN_ADVENTURE_TYPE:
        return scenario_type, normalized, False
    normalized[ACTION_CHOICES_ENABLED_KEY] = True
    return "full_roleplay", normalized, True


def _scenario_content(content_json: str) -> Mapping[str, object]:
    try:
        loaded = json.loads(content_json)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}
