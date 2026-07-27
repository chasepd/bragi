"""Deterministic context-source provenance merging."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import cast

MAX_CONTEXT_SOURCE_PROVENANCE_GROUPS = 64
MAX_CONTEXT_SOURCE_PROVENANCE_GROUP_MEMBERS = 64


def merge_context_source_metadata(
    first: object,
    second: object,
) -> dict[str, object]:
    """Merge alternative derivations without weakening visibility constraints."""
    loaded = [_metadata_object(value) for value in (first, second)]
    metadata: dict[str, object] = {}
    for item in loaded:
        metadata.update(item)

    provenance_overflow = False
    for field in (
        "source_message_ids",
        "audience_character_ids",
        "known_by",
        "tags",
    ):
        values = _merged_string_values(loaded, field)
        if (
            field == "source_message_ids"
            and len(values) > MAX_CONTEXT_SOURCE_PROVENANCE_GROUP_MEMBERS
        ):
            provenance_overflow = True
        metadata[field] = values

    groups = _provenance_groups(loaded)
    if not _provenance_within_bounds(groups):
        provenance_overflow = True
    selected_provenance = loaded
    if provenance_overflow:
        selected_provenance = loaded[:1]
        first_metadata = selected_provenance[0] if selected_provenance else {}
        source_ids = _string_values(first_metadata.get("source_message_ids"))
        groups = _provenance_groups(selected_provenance)
        if (
            len(source_ids) > MAX_CONTEXT_SOURCE_PROVENANCE_GROUP_MEMBERS
            or not _provenance_within_bounds(groups)
        ):
            raise ValueError("Context source provenance is too large")
        metadata["source_message_ids"] = source_ids
        for field in ("source_message_id", "last_seen_message_id"):
            value = first_metadata.get(field)
            if isinstance(value, str) and value:
                metadata[field] = value
            else:
                metadata.pop(field, None)

    metadata["source_provenance_groups"] = groups
    metadata["source_provenance_mode"] = "any"
    if any(item.get("requires_audience") is True for item in loaded):
        metadata["requires_audience"] = True
    return metadata


def _metadata_object(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {
            str(key): item
            for key, item in value.items()
            if isinstance(key, str)
        }
    try:
        loaded = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return {}
    return cast(dict[str, object], loaded) if isinstance(loaded, dict) else {}


def _merged_string_values(
    metadata_items: Iterable[Mapping[str, object]],
    field: str,
) -> list[str]:
    return list(
        dict.fromkeys(
            value
            for item in metadata_items
            for value in _string_values(item.get(field))
        )
    )


def _string_values(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        str(item)
        for item in value
        if isinstance(item, str) and item
    ]


def _provenance_groups(
    metadata_items: Iterable[Mapping[str, object]],
) -> list[list[str]]:
    groups: list[list[str]] = []
    for item in metadata_items:
        item_groups: list[list[str]] = []
        raw_groups = item.get("source_provenance_groups")
        if isinstance(raw_groups, list):
            for raw_group in raw_groups:
                group = _string_values(raw_group)
                if group and group not in item_groups:
                    item_groups.append(group)
        item_source_ids = _string_values(item.get("source_message_ids"))
        for field in ("source_message_id", "last_seen_message_id"):
            value = item.get(field)
            if isinstance(value, str) and value:
                item_source_ids.append(value)
        grouped_ids = {
            source_id
            for group in item_groups
            for source_id in group
        }
        ungrouped_ids = [
            source_id
            for source_id in dict.fromkeys(item_source_ids)
            if source_id not in grouped_ids
        ]
        if ungrouped_ids:
            item_groups.append(ungrouped_ids)
        if item.get("source_provenance_mode") == "all" and item_groups:
            item_groups = [
                list(
                    dict.fromkeys(
                        source_id
                        for group in item_groups
                        for source_id in group
                    )
                )
            ]
        for group in item_groups:
            if group not in groups:
                groups.append(group)
    return groups


def _provenance_within_bounds(groups: Iterable[list[str]]) -> bool:
    materialized = list(groups)
    return (
        len(materialized) <= MAX_CONTEXT_SOURCE_PROVENANCE_GROUPS
        and all(
            len(group) <= MAX_CONTEXT_SOURCE_PROVENANCE_GROUP_MEMBERS
            for group in materialized
        )
    )
