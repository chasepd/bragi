"""Unit tests for the typed TurnOutcome artifact."""

from __future__ import annotations

from bragi.services.post_turn_inference import (
    POST_TURN_DOMAIN_KNOWLEDGE,
    POST_TURN_DOMAIN_SCENE,
    POST_TURN_DOMAIN_TIME,
)
from bragi.services.turn_outcome import (
    TurnOutcome,
    TurnOutcomeEffect,
    remap_turn_outcome_payload,
    turn_outcome_coverage,
    turn_outcome_from_mapping,
)


def _effect(
    *,
    candidate_id: str = "effect:1",
    candidate_type: str = "scene_snapshot_field",
    domain: str = POST_TURN_DOMAIN_SCENE,
    application_status: str = "committed",
    state_key: str = "",
    field_path: str = "",
    value: dict[str, object] | None = None,
    evidence_source_ids: tuple[str, ...] = ("message:player-1",),
) -> TurnOutcomeEffect:
    return TurnOutcomeEffect(
        candidate_id=candidate_id,
        candidate_type=candidate_type,
        domain=domain,
        operation="upsert",
        state_key=state_key,
        field_path=field_path,
        character_id="",
        target_type="",
        target_id="",
        value=value or {},
        confidence=0.9,
        evidence_source_ids=evidence_source_ids,
        evidence_quote="the beacon wakes at ember dawn",
        verifier_status="rendered",
        safe_to_commit=True,
        application_status=application_status,
        reason="rendered in prose",
        changed=True,
    )


def _outcome(effects: tuple[TurnOutcomeEffect, ...]) -> TurnOutcome:
    return TurnOutcome(
        save_id="save-1",
        message_id="narrator-1",
        source_message_ids=("player-1", "narrator-1"),
        attempted_action="I tell Mara the phrase.",
        attempt_feasibility=("Mara is present",),
        attempt_evidence_source_ids=("message:player-1",),
        attempt_evidence_quote="I tell Mara",
        attempt_resolution="succeeded",
        effects=effects,
        verification_passed=True,
        verifier_available=True,
        post_turn_update_needed=False,
        committed_count=len(effects),
    )


def test_turn_outcome_round_trips_through_json() -> None:
    outcome = _outcome(
        (_effect(candidate_id="scene:mood", state_key="scene_snapshot.mood"),)
    )
    restored = turn_outcome_from_mapping(outcome.to_json())
    assert restored is not None
    assert restored.save_id == "save-1"
    assert restored.message_id == "narrator-1"
    assert restored.source_message_ids == ("player-1", "narrator-1")
    assert restored.attempted_action == "I tell Mara the phrase."
    assert restored.attempt_resolution == "succeeded"
    assert restored.attempt_evidence_source_ids == ("message:player-1",)
    assert restored.verification_passed is True
    assert restored.post_turn_update_needed is False
    assert len(restored.effects) == 1
    effect = restored.effects[0]
    assert effect.candidate_id == "scene:mood"
    assert effect.verifier_status == "rendered"
    assert effect.application_status == "committed"
    assert effect.evidence_source_ids == ("message:player-1",)


def test_turn_outcome_from_mapping_requires_save_and_message_ids() -> None:
    assert turn_outcome_from_mapping({}) is None
    assert turn_outcome_from_mapping({"save_id": "s", "message_id": ""}) is None


def test_turn_outcome_coverage_derives_domains_and_details() -> None:
    outcome = _outcome(
        (
            _effect(
                candidate_id="scene:mood",
                candidate_type="scene_snapshot_field",
                domain=POST_TURN_DOMAIN_SCENE,
                state_key="scene_snapshot.mood",
                field_path="mood",
                value={"mood": "uneasy"},
            ),
            _effect(
                candidate_id="memory:1",
                candidate_type="character_learned_memory",
                domain=POST_TURN_DOMAIN_KNOWLEDGE,
                state_key="memories",
                value={"body": "Mara knows the phrase"},
            ),
            _effect(
                candidate_id="queued:1",
                candidate_type="character_learned_memory",
                domain=POST_TURN_DOMAIN_KNOWLEDGE,
                application_status="confirmation_queued",
                value={"body": "Mara knows a second phrase"},
            ),
        )
    )
    coverage = turn_outcome_coverage(outcome)
    assert coverage.applied_domains == frozenset(
        {POST_TURN_DOMAIN_SCENE, POST_TURN_DOMAIN_KNOWLEDGE}
    )
    assert coverage.queued_domains == frozenset({POST_TURN_DOMAIN_KNOWLEDGE})
    assert coverage.scene_snapshot_fields == frozenset({"mood"})
    assert len(coverage.memory_fingerprints) == 2
    assert coverage.committed_count == 2
    assert coverage.confirmation_queued_count == 1
    assert coverage.source_message_ids == ("player-1", "narrator-1")
    assert coverage.metadata["planned_commit_verifier_available"] is True
    assert coverage.metadata["planned_commit_post_turn_update_needed"] is False


def test_turn_outcome_coverage_tracks_world_time_change_fields() -> None:
    outcome = _outcome(
        (
            _effect(
                candidate_id="time:1",
                candidate_type="world_time_change",
                domain=POST_TURN_DOMAIN_TIME,
                value={"time_of_day": "evening", "day_of_week": "tuesday"},
            ),
        )
    )
    coverage = turn_outcome_coverage(outcome)
    assert POST_TURN_DOMAIN_TIME in coverage.applied_domains
    assert {"in_world_time", "time_of_day", "day_of_week"} <= (
        coverage.scene_snapshot_fields
    )


def test_remap_turn_outcome_payload_remaps_message_refs() -> None:
    outcome = _outcome(
        (
            _effect(
                candidate_id="memory:1",
                candidate_type="character_learned_memory",
                domain=POST_TURN_DOMAIN_KNOWLEDGE,
                evidence_source_ids=("message:player-1", "message:narrator-1"),
            ),
        )
    )
    remapped = remap_turn_outcome_payload(
        outcome.to_json(),
        message_id_map={"player-1": "player-2", "narrator-1": "narrator-2"},
        save_id="save-2",
    )
    assert remapped["save_id"] == "save-2"
    assert remapped["source_message_ids"] == ["player-2", "narrator-2"]
    assert remapped["attempt_evidence_source_ids"] == ["message:player-2"]
    raw_effects = remapped["effects"]
    assert isinstance(raw_effects, list)
    effect = raw_effects[0]
    assert isinstance(effect, dict)
    assert effect["evidence_source_ids"] == ["message:player-2", "message:narrator-2"]


def test_remap_turn_outcome_payload_ignores_unknown_message_refs() -> None:
    outcome = _outcome((_effect(),))
    remapped = remap_turn_outcome_payload(
        outcome.to_json(),
        message_id_map={"unrelated": "x"},
    )
    raw_effects = remapped["effects"]
    assert isinstance(raw_effects, list)
    effect = raw_effects[0]
    assert isinstance(effect, dict)
    assert effect["evidence_source_ids"] == ["message:player-1"]
    assert remapped["source_message_ids"] == ["player-1", "narrator-1"]
