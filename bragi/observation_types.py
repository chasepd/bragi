"""Canonical observation types shared by extraction and persistence."""

from __future__ import annotations

import re

OBSERVATION_TYPES = (
    "identity",
    "character_voice",
    "relationship",
    "inventory",
    "location",
    "promise",
    "open_obligation",
    "player_preference",
    "character_intent",
    "open_thread",
    "scene_fact",
    "event",
    "other",
)

_OBSERVATION_TYPE_ALIASES = {
    "look_around": "scene_fact",
    "scene_detail": "scene_fact",
    "preference": "player_preference",
    "npc_intent": "character_intent",
    "character_goal": "character_intent",
    "quest": "open_obligation",
    "task": "open_obligation",
    "obligation": "open_obligation",
    "oath": "promise",
    "voice": "character_voice",
    "bond": "relationship",
    "item": "inventory",
    "place": "location",
    "character_fact": "identity",
    "memory": "event",
}


def normalize_observation_type(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    normalized = _OBSERVATION_TYPE_ALIASES.get(normalized, normalized)
    return normalized if normalized in OBSERVATION_TYPES else "other"
