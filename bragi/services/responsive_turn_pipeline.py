"""Deterministic routing policy for the responsive turn pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass

from bragi.persistence.models import (
    CharacterRecord,
    LocationRecord,
    ScenarioRecord,
)
from bragi.retry_policy import RetryExecutionClass, current_retry_execution_class

TURN_OPERATION_NEW_PLAYER = "new_player"
TURN_OPERATION_REGENERATE = "regenerate"
TURN_OPERATION_EDIT = "edit"
TURN_OPERATION_TIMESKIP = "timeskip"
TURN_OPERATION_LOOK_AROUND = "look_around"
TURN_OPERATION_RECOVERY = "recovery"

_POTENTIAL_NAME = re.compile(
    r"(?<![\w'-])([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*)(?![\w'-])"
)
_NON_NAME_CAPITALIZED_PHRASES = frozenset(
    {
        "a",
        "an",
        "and",
        "but",
        "i",
        "if",
        "maybe",
        "no",
        "okay",
        "please",
        "so",
        "the",
        "then",
        "yes",
    }
)


@dataclass(frozen=True)
class ResponsiveFastPathEligibility:
    eligible: bool
    reasons: tuple[str, ...]


def responsive_fast_path_eligibility(
    *,
    operation: str,
    precomputed_snapshot_valid: bool,
    strong_local_recall: bool,
    character_references_resolved: bool,
    continuity_ready: bool,
    retrieval_degraded: bool,
) -> ResponsiveFastPathEligibility:
    """Require every locked deterministic gate before helpers may be skipped."""

    reasons: list[str] = []
    if (
        current_retry_execution_class()
        is not RetryExecutionClass.RESPONSIVE_FOREGROUND
    ):
        reasons.append("not_responsive_foreground")
    if operation != TURN_OPERATION_NEW_PLAYER:
        reasons.append("operation_not_new_player")
    if not precomputed_snapshot_valid:
        reasons.append("precomputed_snapshot_missing")
    if not strong_local_recall:
        reasons.append("local_recall_not_strong")
    if not character_references_resolved:
        reasons.append("character_reference_unresolved")
    if not continuity_ready:
        reasons.append("continuity_not_ready")
    if retrieval_degraded:
        reasons.append("retrieval_degraded")
    return ResponsiveFastPathEligibility(
        eligible=not reasons,
        reasons=tuple(reasons),
    )


def character_references_are_resolved(
    *,
    player_message: str,
    characters: tuple[CharacterRecord, ...],
    present_character_ids: frozenset[str],
    locations: tuple[LocationRecord, ...] = (),
    scenario: ScenarioRecord | None = None,
) -> bool:
    """Conservatively reject absent known characters and unknown proper names."""

    allowed_character_ids = present_character_ids | frozenset(
        character.id for character in characters if character.is_player_character
    )
    for character in characters:
        references = (character.name, *character.aliases, character.contact_name)
        if character.id not in allowed_character_ids and any(
            _phrase_is_present(reference, player_message) for reference in references
        ):
            return False

    known_phrases = {
        value.casefold().strip()
        for value in (
            *(
                reference
                for character in characters
                for reference in (
                    character.name,
                    *character.aliases,
                    character.contact_name,
                )
            ),
            *(location.name for location in locations),
            *((scenario.title, scenario.player_role) if scenario is not None else ()),
        )
        if value.strip()
    }
    for match in _POTENTIAL_NAME.finditer(player_message):
        phrase = match.group(1).casefold().strip()
        if phrase in _NON_NAME_CAPITALIZED_PHRASES:
            continue
        if any(
            phrase == known or phrase in known.split() or known in phrase.split()
            for known in known_phrases
        ):
            continue
        return False
    return True


def _phrase_is_present(phrase: str, text: str) -> bool:
    phrase = phrase.strip()
    if not phrase:
        return False
    return re.search(
        rf"(?<![\w'-]){re.escape(phrase)}(?:['\u2019]s)?(?![\w'-])",
        text,
        flags=re.IGNORECASE,
    ) is not None
