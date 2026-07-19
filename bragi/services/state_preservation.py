"""Durable memory preservation for overwritten world-state facts."""

from __future__ import annotations

import json

from bragi.persistence.models import MemoryRecord, WorldStateRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.post_turn_inference import memory_fingerprint

STATE_HISTORY_TAG = "state_history"
FROM_WORLD_STATE_TAG = "from_world_state"


def preserve_replaced_world_state_memory(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    before: WorldStateRecord | None,
    after_value: dict[str, object] | None,
    source_message_id: str | None,
) -> MemoryRecord | None:
    """Persist displaced durable world-state values as retrievable memory."""
    if before is None or not _durable_state_can_be_preserved(before):
        return None
    changed_value = _changed_previous_value(before.value, after_value)
    if not changed_value:
        return None
    body = (
        f"Previous world state for {before.key}: "
        f"{_format_state_value(changed_value)}"
    )
    fingerprint = memory_fingerprint(body)
    if any(
        memory_fingerprint(memory.body) == fingerprint
        for memory in repositories.list_memories(save_id)
    ):
        return None
    source_ids = tuple(
        dict.fromkeys(
            source_id
            for source_id in (before.source_message_id, source_message_id)
            if source_id
        )
    )
    return repositories.add_memory(
        save_id=save_id,
        body=body,
        tags=_state_history_tags(before),
        importance=max(0.55, min(before.confidence, 0.85)),
        source_message_ids=source_ids,
    )


def _durable_state_can_be_preserved(state: WorldStateRecord) -> bool:
    key = state.key.strip().casefold()
    category = state.category.strip().casefold()
    if not state.value:
        return False
    if category in {"scene", "ephemeral"}:
        return False
    if key.startswith("scene.") or key.endswith(".current_emotional_state"):
        return False
    return True


def _changed_previous_value(
    before: dict[str, object],
    after: dict[str, object] | None,
) -> dict[str, object]:
    if after is None:
        return dict(before)
    return {
        key: value
        for key, value in before.items()
        if key not in after or after.get(key) != value
    }


def _format_state_value(value: dict[str, object]) -> str:
    return ", ".join(
        f"{key}: {_format_scalar(item)}" for key, item in sorted(value.items())
    )


def _format_scalar(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float) and not isinstance(value, bool):
        return str(value)
    return json.dumps(value, sort_keys=True)


def _state_history_tags(state: WorldStateRecord) -> list[str]:
    tags = [
        STATE_HISTORY_TAG,
        FROM_WORLD_STATE_TAG,
        f"state_key:{state.key}",
    ]
    category = state.category.strip()
    if category:
        tags.append(f"category:{category}")
    return tags
