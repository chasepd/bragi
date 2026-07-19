"""Manual confirmation setting helpers."""

from __future__ import annotations

from bragi.persistence.repositories import PersistenceRepositories

MANUAL_CONFIRMATION_MEMORIES_SETTING = "manual_confirmation_memories_enabled"
MANUAL_CONFIRMATION_CHARACTER_REGISTRY_SETTING = (
    "manual_confirmation_character_registry_enabled"
)
MANUAL_CONFIRMATION_STATE_CHANGES_SETTING = (
    "manual_confirmation_state_changes_enabled"
)


def manual_memory_confirmation_enabled(
    repositories: PersistenceRepositories,
    *,
    save_id: str | None = None,
) -> bool:
    return _bool_setting(
        repositories,
        MANUAL_CONFIRMATION_MEMORIES_SETTING,
        save_id=save_id,
    )


def manual_character_registry_confirmation_enabled(
    repositories: PersistenceRepositories,
    *,
    save_id: str | None = None,
) -> bool:
    return _bool_setting(
        repositories,
        MANUAL_CONFIRMATION_CHARACTER_REGISTRY_SETTING,
        save_id=save_id,
    )


def manual_state_change_confirmation_enabled(
    repositories: PersistenceRepositories,
    *,
    save_id: str | None = None,
) -> bool:
    return _bool_setting(
        repositories,
        MANUAL_CONFIRMATION_STATE_CHANGES_SETTING,
        save_id=save_id,
    )


def _bool_setting(
    repositories: PersistenceRepositories,
    key: str,
    *,
    save_id: str | None,
) -> bool:
    value = repositories.get_effective_setting(key, save_id=save_id)
    return value if isinstance(value, bool) else False
