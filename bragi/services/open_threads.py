"""Helpers for canonical open-thread state handling."""

from __future__ import annotations

from bragi.persistence.repositories import PersistenceRepositories

OPEN_THREAD_AGGREGATE_KEYS = frozenset(
    {
        "interaction.open_threads",
        "interactions.open_threads",
        "open_threads",
    }
)


def is_open_threads_aggregate_key(key: str) -> bool:
    return key.strip().casefold() in OPEN_THREAD_AGGREGATE_KEYS


def has_active_thread_records(
    repositories: PersistenceRepositories,
    save_id: str,
) -> bool:
    return repositories.has_active_threads(save_id)


def archive_open_thread_aggregate_state(
    repositories: PersistenceRepositories,
    save_id: str,
) -> int:
    archived_count = 0
    for state in repositories.list_world_state(save_id):
        if not is_open_threads_aggregate_key(state.key):
            continue
        repositories.archive_world_state(save_id=save_id, key=state.key)
        archived_count += 1
    return archived_count
