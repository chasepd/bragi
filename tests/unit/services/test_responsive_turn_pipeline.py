from __future__ import annotations

import pytest

from bragi.persistence.models import CharacterRecord
from bragi.retry_policy import RetryExecutionClass, retry_execution_context
from bragi.services.responsive_turn_pipeline import (
    TURN_OPERATION_EDIT,
    TURN_OPERATION_NEW_PLAYER,
    character_references_are_resolved,
    responsive_fast_path_eligibility,
)


def _character(
    character_id: str,
    name: str,
    *,
    is_player_character: bool = False,
) -> CharacterRecord:
    return CharacterRecord(
        id=character_id,
        save_id="save-1",
        name=name,
        aliases=[],
        role="player" if is_player_character else "npc",
        known_state="",
        met=True,
        appearance="",
        visual_notes="",
        current_clothing="",
        personality="",
        voice="",
        relationships={},
        status="active",
        location_id=None,
        private_notes="",
        source_message_id=None,
        locked_fields=[],
        is_player_character=is_player_character,
    )


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({"operation": TURN_OPERATION_EDIT}, "operation_not_new_player"),
        ({"precomputed_snapshot_valid": False}, "precomputed_snapshot_missing"),
        ({"strong_local_recall": False}, "local_recall_not_strong"),
        ({"character_references_resolved": False}, "character_reference_unresolved"),
        ({"continuity_ready": False}, "continuity_not_ready"),
        ({"retrieval_degraded": True}, "retrieval_degraded"),
    ],
)
def test_responsive_fast_path_requires_every_deterministic_gate(
    overrides: dict[str, object],
    expected_reason: str,
) -> None:
    inputs: dict[str, object] = {
        "operation": TURN_OPERATION_NEW_PLAYER,
        "precomputed_snapshot_valid": True,
        "strong_local_recall": True,
        "character_references_resolved": True,
        "continuity_ready": True,
        "retrieval_degraded": False,
    }
    inputs.update(overrides)

    with retry_execution_context(RetryExecutionClass.RESPONSIVE_FOREGROUND):
        result = responsive_fast_path_eligibility(**inputs)  # type: ignore[arg-type]

    assert result.eligible is False
    assert expected_reason in result.reasons


def test_responsive_fast_path_accepts_only_complete_responsive_new_player_state(
) -> None:
    with retry_execution_context(RetryExecutionClass.RESPONSIVE_FOREGROUND):
        result = responsive_fast_path_eligibility(
            operation=TURN_OPERATION_NEW_PLAYER,
            precomputed_snapshot_valid=True,
            strong_local_recall=True,
            character_references_resolved=True,
            continuity_ready=True,
            retrieval_degraded=False,
        )

    assert result.eligible is True
    assert result.reasons == ()


def test_quality_execution_never_uses_responsive_fast_path() -> None:
    result = responsive_fast_path_eligibility(
        operation=TURN_OPERATION_NEW_PLAYER,
        precomputed_snapshot_valid=True,
        strong_local_recall=True,
        character_references_resolved=True,
        continuity_ready=True,
        retrieval_degraded=False,
    )

    assert result.eligible is False
    assert result.reasons == ("not_responsive_foreground",)


def test_character_reference_gate_accepts_present_and_player_names() -> None:
    assert character_references_are_resolved(
        player_message="I tell Mira that Rowan will wait.",
        characters=(
            _character("mira", "Mira"),
            _character("rowan", "Rowan", is_player_character=True),
        ),
        present_character_ids=frozenset({"mira"}),
    )


def test_character_reference_gate_accepts_present_uncased_script_name() -> None:
    assert character_references_are_resolved(
        player_message="I ask 李梅 what she sees.",
        characters=(_character("li-mei", "李梅"),),
        present_character_ids=frozenset({"li-mei"}),
    )


@pytest.mark.parametrize(
    "player_message",
    (
        "I call for Mira.",
        "I ask Zorak for directions.",
        "I ask zorak for directions.",
        "I ask Élodie for directions.",
        "I ask 李梅 for directions.",
        "李梅 waits nearby.",
        "I speak with zorak.",
        "I approach zorak.",
    ),
)
def test_character_reference_gate_rejects_absent_or_unknown_named_characters(
    player_message: str,
) -> None:
    assert not character_references_are_resolved(
        player_message=player_message,
        characters=(_character("mira", "Mira"),),
        present_character_ids=frozenset(),
    )
