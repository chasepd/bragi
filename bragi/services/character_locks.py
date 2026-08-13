"""Character fact lock normalization helpers."""

from __future__ import annotations

from collections.abc import Iterable

CHARACTER_AGENCY_FIELDS = frozenset(
    {
        "goals",
        "motivations",
        "current_intent",
        "boundaries",
        "attitude_toward_player",
        "cooperation_conditions",
    }
)

CHARACTER_FACT_LOCK_FIELDS = frozenset(
    {
        "name",
        "aliases",
        "role",
        "age",
        "known_state",
        "met",
        "appearance",
        "visual_notes",
        "current_clothing",
        "personality",
        "voice",
        "texting_style",
        "relationships",
        *CHARACTER_AGENCY_FIELDS,
        "status",
        "location_id",
        "private_notes",
        "present",
    }
)

_CHARACTER_LOCK_FIELD_ALIASES = {
    "aliases_text": "aliases",
    "relationships_json": "relationships",
}


def canonical_character_fact_lock_field(field: str) -> str | None:
    normalized = _CHARACTER_LOCK_FIELD_ALIASES.get(field.strip(), field.strip())
    if normalized in CHARACTER_FACT_LOCK_FIELDS:
        return normalized
    return None


def normalize_character_fact_locks(fields: Iterable[str]) -> list[str]:
    return sorted(
        {
            canonical
            for field in fields
            if (canonical := canonical_character_fact_lock_field(field)) is not None
        }
    )


def normalize_character_locked_fields(
    fields: Iterable[str],
    *,
    preserve_unknown: bool = True,
) -> list[str]:
    normalized: set[str] = set()
    for field in fields:
        stripped = field.strip()
        if not stripped:
            continue
        canonical = canonical_character_fact_lock_field(stripped)
        if canonical is not None:
            normalized.add(canonical)
        elif preserve_unknown:
            normalized.add(stripped)
    return sorted(normalized)


def explicit_character_locked_fields(
    existing: Iterable[str],
    requested: Iterable[str],
) -> list[str]:
    existing_normalized = normalize_character_locked_fields(existing)
    preserved_non_fact = {
        field
        for field in existing_normalized
        if canonical_character_fact_lock_field(field) is None
    }
    return sorted({*preserved_non_fact, *normalize_character_fact_locks(requested)})


def merge_character_locked_fields(
    existing: Iterable[str],
    changed: Iterable[str],
) -> list[str]:
    return normalize_character_locked_fields((*existing, *changed))


def character_field_is_locked(
    locked_fields: Iterable[str],
    field_path: str,
) -> bool:
    canonical = canonical_character_fact_lock_field(field_path)
    if canonical is None:
        return field_path.strip() in {field.strip() for field in locked_fields}
    return canonical in normalize_character_fact_locks(locked_fields)


def reconcile_character_presence_locks(
    *,
    current_present_ids: Iterable[str],
    proposed_present_ids: Iterable[str],
    locked_character_ids: Iterable[str],
) -> set[str]:
    """Apply a proposed scene roster while preserving locked memberships."""
    current = set(current_present_ids)
    proposed = set(proposed_present_ids)
    locked = set(locked_character_ids)
    return (proposed - locked) | (current & locked)
