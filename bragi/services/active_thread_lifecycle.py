"""Shared lifecycle policy for active thread prompt visibility."""

from __future__ import annotations

from dataclasses import replace

from bragi.persistence.models import ActiveThreadRecord
from bragi.persistence.repositories import PersistenceRepositories

ACTIVE_THREAD_OPEN_STATUSES = frozenset({"active", "blocked", "deferred"})
ACTIVE_THREAD_INACTIVE_STATUSES = frozenset({"resolved", "abandoned"})
ACTIVE_THREAD_STATUSES = ACTIVE_THREAD_OPEN_STATUSES | ACTIVE_THREAD_INACTIVE_STATUSES
ACTIVE_THREAD_VISIBILITIES = frozenset({"public", "private", "hidden", "scene"})

_STATUS_ALIASES = {
    "": "active",
    "active": "active",
    "open": "active",
    "ongoing": "active",
    "unresolved": "active",
    "in progress": "active",
    "inprogress": "active",
    "pending": "active",
    "waiting": "blocked",
    "blocked": "blocked",
    "paused": "deferred",
    "deferred": "deferred",
    "resolved": "resolved",
    "resolve": "resolved",
    "complete": "resolved",
    "completed": "resolved",
    "done": "resolved",
    "closed": "resolved",
    "finished": "resolved",
    "fulfilled": "resolved",
    "settled": "resolved",
    "abandoned": "abandoned",
    "abandon": "abandoned",
    "cancelled": "abandoned",
    "canceled": "abandoned",
    "dropped": "abandoned",
    "inactive": "abandoned",
    "irrelevant": "abandoned",
    "obsolete": "abandoned",
    "retired": "abandoned",
    "stale": "abandoned",
    "superseded": "abandoned",
}

_VISIBILITY_ALIASES = {
    "": "public",
    "active": "public",
    "neither": "public",
    "none": "public",
    "open": "public",
    "public": "public",
    "private": "private",
    "private between": "private",
    "private between characters": "private",
    "hidden": "hidden",
    "secret": "hidden",
    "internal": "hidden",
    "gm only": "hidden",
    "scene": "scene",
    "scene local": "scene",
    "scene only": "scene",
    "current scene": "scene",
    "local": "scene",
}


def normalize_active_thread_status(status: object) -> str:
    key = _normalized_key(status)
    if key in _STATUS_ALIASES:
        return _STATUS_ALIASES[key]
    return "active"


def normalize_active_thread_visibility(visibility: object) -> str:
    key = _normalized_key(visibility)
    if key in _VISIBILITY_ALIASES:
        return _VISIBILITY_ALIASES[key]
    if "hidden" in key or "secret" in key:
        return "hidden"
    if "private" in key:
        return "private"
    if "scene" in key or "local" in key:
        return "scene"
    return "public"


def active_thread_status_is_open(status: object) -> bool:
    return normalize_active_thread_status(status) in ACTIVE_THREAD_OPEN_STATUSES


def active_thread_is_prompt_visible(thread: ActiveThreadRecord) -> bool:
    if not active_thread_status_is_open(thread.status):
        return False
    return normalize_active_thread_visibility(thread.visibility) != "hidden"


def active_thread_is_scene_local(thread: ActiveThreadRecord) -> bool:
    return normalize_active_thread_visibility(thread.visibility) == "scene"


def normalize_active_thread_record(
    thread: ActiveThreadRecord,
) -> ActiveThreadRecord:
    return replace(
        thread,
        status=normalize_active_thread_status(thread.status),
        visibility=normalize_active_thread_visibility(thread.visibility),
    )


def archive_inactive_active_threads(
    repositories: PersistenceRepositories,
    save_id: str,
) -> int:
    archived_count = 0
    for thread in repositories.list_active_threads(save_id):
        normalized = normalize_active_thread_record(thread)
        if normalized != thread:
            thread = repositories.update_active_thread(normalized)
        if active_thread_status_is_open(thread.status):
            continue
        repositories.archive_active_thread(thread.id)
        archived_count += 1
    return archived_count


def _normalized_key(value: object) -> str:
    return " ".join(
        str(value or "")
        .strip()
        .casefold()
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )
