"""Typed TurnOutcome artifact for verified narrator turns.

A TurnOutcome records what a turn established: the player's attempted
action, its resolution, and every verified effect that was deterministically
applied (or queued) after narrator verification approved it against the
accepted prose. It is the durable, evidence-carrying artifact that lets
post-turn inference stages run only for domains the turn did not cover.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from bragi.services.post_turn_inference import (
    VerifiedPostTurnCoverage,
    memory_fingerprint,
    planned_effect_domain,
)


@dataclass(frozen=True)
class TurnOutcomeEffect:
    candidate_id: str
    candidate_type: str
    domain: str
    operation: str
    state_key: str
    field_path: str
    character_id: str
    target_type: str
    target_id: str
    value: dict[str, object]
    confidence: float
    evidence_source_ids: tuple[str, ...]
    evidence_quote: str
    verifier_status: str = ""
    safe_to_commit: bool = False
    application_status: str = ""
    reason: str = ""
    changed: bool = False

    def to_json(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_type": self.candidate_type,
            "domain": self.domain,
            "operation": self.operation,
            "state_key": self.state_key,
            "field_path": self.field_path,
            "character_id": self.character_id,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "value": self.value,
            "confidence": self.confidence,
            "evidence_source_ids": list(self.evidence_source_ids),
            "evidence_quote": self.evidence_quote,
            "verifier_status": self.verifier_status,
            "safe_to_commit": self.safe_to_commit,
            "application_status": self.application_status,
            "reason": self.reason,
            "changed": self.changed,
        }


@dataclass(frozen=True)
class TurnOutcome:
    save_id: str
    message_id: str
    source_message_ids: tuple[str, ...] = ()
    attempted_action: str = ""
    attempt_feasibility: tuple[str, ...] = ()
    attempt_evidence_source_ids: tuple[str, ...] = ()
    attempt_evidence_quote: str = ""
    attempt_resolution: str = ""
    effects: tuple[TurnOutcomeEffect, ...] = ()
    applied_domains: frozenset[str] = frozenset()
    queued_domains: frozenset[str] = frozenset()
    verification_passed: bool = False
    verifier_available: bool = False
    post_turn_update_needed: bool = True
    committed_count: int = 0
    confirmation_queued_count: int = 0

    @property
    def covered_domains(self) -> frozenset[str]:
        return frozenset(self.applied_domains) | frozenset(self.queued_domains)

    def to_json(self) -> dict[str, object]:
        return {
            "save_id": self.save_id,
            "message_id": self.message_id,
            "source_message_ids": list(self.source_message_ids),
            "attempted_action": self.attempted_action,
            "attempt_feasibility": list(self.attempt_feasibility),
            "attempt_evidence_source_ids": list(self.attempt_evidence_source_ids),
            "attempt_evidence_quote": self.attempt_evidence_quote,
            "attempt_resolution": self.attempt_resolution,
            "effects": [effect.to_json() for effect in self.effects],
            "applied_domains": sorted(self.applied_domains),
            "queued_domains": sorted(self.queued_domains),
            "verification_passed": self.verification_passed,
            "verifier_available": self.verifier_available,
            "post_turn_update_needed": self.post_turn_update_needed,
            "committed_count": self.committed_count,
            "confirmation_queued_count": self.confirmation_queued_count,
        }


def turn_outcome_from_mapping(value: object) -> TurnOutcome | None:
    if not isinstance(value, Mapping):
        return None
    save_id = _text(value.get("save_id"))
    message_id = _text(value.get("message_id"))
    if not save_id or not message_id:
        return None
    effects: list[TurnOutcomeEffect] = []
    raw_effects = value.get("effects")
    if isinstance(raw_effects, list):
        for item in raw_effects:
            effect = _effect_from_mapping(item)
            if effect is not None:
                effects.append(effect)
    return TurnOutcome(
        save_id=save_id,
        message_id=message_id,
        source_message_ids=_string_tuple(value.get("source_message_ids")),
        attempted_action=_text(value.get("attempted_action")),
        attempt_feasibility=_string_tuple(value.get("attempt_feasibility")),
        attempt_evidence_source_ids=_string_tuple(
            value.get("attempt_evidence_source_ids")
        ),
        attempt_evidence_quote=_text(value.get("attempt_evidence_quote")),
        attempt_resolution=_text(value.get("attempt_resolution")),
        effects=tuple(effects),
        applied_domains=frozenset(_string_tuple(value.get("applied_domains"))),
        queued_domains=frozenset(_string_tuple(value.get("queued_domains"))),
        verification_passed=bool(value.get("verification_passed")),
        verifier_available=bool(value.get("verifier_available")),
        post_turn_update_needed=value.get("post_turn_update_needed") is not False,
        committed_count=_nonnegative_int(value.get("committed_count")),
        confirmation_queued_count=_nonnegative_int(
            value.get("confirmation_queued_count")
        ),
    )


def _effect_from_mapping(value: object) -> TurnOutcomeEffect | None:
    if not isinstance(value, Mapping):
        return None
    candidate_id = _text(value.get("candidate_id"))
    candidate_type = _text(value.get("candidate_type"))
    if not candidate_id or not candidate_type:
        return None
    raw_value = value.get("value")
    return TurnOutcomeEffect(
        candidate_id=candidate_id,
        candidate_type=candidate_type,
        domain=_text(value.get("domain")) or planned_effect_domain(candidate_type),
        operation=_text(value.get("operation")),
        state_key=_text(value.get("state_key")),
        field_path=_text(value.get("field_path")),
        character_id=_text(value.get("character_id")),
        target_type=_text(value.get("target_type")),
        target_id=_text(value.get("target_id")),
        value=dict(raw_value) if isinstance(raw_value, dict) else {},
        confidence=_float(value.get("confidence")),
        evidence_source_ids=_string_tuple(value.get("evidence_source_ids")),
        evidence_quote=_text(value.get("evidence_quote")),
        verifier_status=_text(value.get("verifier_status")),
        safe_to_commit=bool(value.get("safe_to_commit")),
        application_status=_text(value.get("application_status")),
        reason=_text(value.get("reason")),
        changed=bool(value.get("changed")),
    )


def turn_outcome_coverage(outcome: TurnOutcome) -> VerifiedPostTurnCoverage:
    """Derive the VerifiedPostTurnCoverage a TurnOutcome establishes."""
    state_keys: set[str] = set()
    scene_fields: set[str] = set()
    scene_presence_ids: set[str] = set()
    memory_fingerprints: set[str] = set()
    knowledge_edge_targets: set[tuple[str, str, str]] = set()
    applied_domains: set[str] = set()
    queued_domains: set[str] = set()
    committed_count = 0
    confirmation_queued_count = 0
    for effect in outcome.effects:
        if effect.application_status == "committed":
            applied_domains.add(effect.domain)
            committed_count += 1
        elif effect.application_status == "confirmation_queued":
            queued_domains.add(effect.domain)
            confirmation_queued_count += 1
        _accumulate_effect_coverage(
            effect,
            state_keys=state_keys,
            scene_fields=scene_fields,
            scene_presence_ids=scene_presence_ids,
            memory_fingerprints=memory_fingerprints,
            knowledge_edge_targets=knowledge_edge_targets,
        )
    return VerifiedPostTurnCoverage(
        source_message_ids=outcome.source_message_ids,
        state_keys=frozenset(state_keys),
        scene_snapshot_fields=frozenset(scene_fields),
        scene_presence_character_ids=frozenset(scene_presence_ids),
        memory_fingerprints=frozenset(memory_fingerprints),
        knowledge_edge_targets=frozenset(knowledge_edge_targets),
        applied_domains=frozenset(applied_domains),
        queued_domains=frozenset(queued_domains),
        committed_count=committed_count,
        confirmation_queued_count=confirmation_queued_count,
        metadata={
            "planned_commit_proposed_count": len(outcome.effects),
            "planned_commit_committed_count": committed_count,
            "planned_commit_queued_count": confirmation_queued_count,
            "planned_commit_skipped_count": sum(
                1
                for effect in outcome.effects
                if effect.application_status not in {"committed", "confirmation_queued"}
            ),
            "planned_commit_contradicted_count": sum(
                1
                for effect in outcome.effects
                if effect.verifier_status == "contradicted"
            ),
            "planned_commit_rejected_count": sum(
                1
                for effect in outcome.effects
                if effect.application_status not in {"committed", "confirmation_queued"}
            ),
            "planned_commit_confirmation_queued_count": confirmation_queued_count,
            "planned_commit_verifier_available": outcome.verifier_available,
            "planned_commit_verification_passed": outcome.verification_passed,
            "planned_commit_post_turn_update_needed": outcome.post_turn_update_needed,
        },
    )


def character_physical_state_key(character_id: str) -> str:
    return f"character.{character_id}.physical_state"


def character_emotional_state_key(character_id: str) -> str:
    return f"character.{character_id}.current_emotional_state"


def character_relationships_state_key(character_id: str) -> str:
    return f"character.{character_id}.relationships"


def _accumulate_effect_coverage(
    effect: TurnOutcomeEffect,
    *,
    state_keys: set[str],
    scene_fields: set[str],
    scene_presence_ids: set[str],
    memory_fingerprints: set[str],
    knowledge_edge_targets: set[tuple[str, str, str]],
) -> None:
    if effect.application_status == "confirmation_queued":
        if effect.candidate_type == "character_learned_memory":
            body = _text(effect.value.get("body"))
            if body:
                memory_fingerprints.add(memory_fingerprint(body))
        return
    if effect.application_status != "committed":
        return
    if effect.state_key:
        state_keys.add(effect.state_key)
    if effect.candidate_type == "scene_presence":
        character_id = effect.character_id or _text(effect.value.get("character_id"))
        if character_id:
            scene_presence_ids.add(character_id)
    elif effect.candidate_type == "scene_snapshot_field":
        field_path = effect.field_path or effect.state_key.removeprefix(
            "scene_snapshot."
        )
        if field_path:
            scene_fields.add(field_path)
    elif effect.candidate_type == "character_learned_memory":
        body = _text(effect.value.get("body"))
        if body:
            memory_fingerprints.add(memory_fingerprint(body))
    elif effect.candidate_type == "character_knowledge_edge":
        character_id = effect.character_id or _text(effect.value.get("character_id"))
        target_type = effect.target_type or _text(effect.value.get("target_type"))
        target_id = effect.target_id or _text(effect.value.get("target_id"))
        if character_id and target_type and target_id:
            knowledge_edge_targets.add((character_id, target_type, target_id))
    elif effect.candidate_type == "world_time_change":
        for field in ("in_world_time", "time_of_day", "day_of_week"):
            scene_fields.add(field)
    elif effect.candidate_type in {
        "physical_change",
        "emotional_change",
        "relationship_change",
    }:
        if effect.state_key:
            return
        character_id = effect.character_id or _text(effect.value.get("character_id"))
        if not character_id:
            return
        if effect.candidate_type == "physical_change":
            state_keys.add(character_physical_state_key(character_id))
        elif effect.candidate_type == "emotional_change":
            state_keys.add(character_emotional_state_key(character_id))
        else:
            state_keys.add(character_relationships_state_key(character_id))


def remap_turn_outcome_payload(
    payload: dict[str, object],
    *,
    message_id_map: Mapping[str, str],
    save_id: str | None = None,
) -> dict[str, object]:
    """Remap message references inside a TurnOutcome payload after import."""
    remapped = dict(payload)
    if save_id is not None:
        remapped["save_id"] = save_id
    payload_message_id = payload.get("message_id")
    if isinstance(payload_message_id, str) and payload_message_id:
        remapped["message_id"] = remap_source_ref(
            payload_message_id,
            message_id_map=message_id_map,
        )

    def remap_refs(refs: object) -> object:
        if not isinstance(refs, list):
            return refs
        return [
            remap_source_ref(item, message_id_map=message_id_map)
            if isinstance(item, str)
            else item
            for item in refs
        ]

    remapped["source_message_ids"] = remap_refs(payload.get("source_message_ids"))
    remapped["attempt_evidence_source_ids"] = remap_refs(
        payload.get("attempt_evidence_source_ids")
    )
    raw_effects = payload.get("effects")
    if isinstance(raw_effects, list):
        remapped_effects: list[object] = []
        for item in raw_effects:
            if not isinstance(item, dict):
                remapped_effects.append(item)
                continue
            effect = dict(item)
            evidence = effect.get("evidence_source_ids")
            if isinstance(evidence, list):
                effect["evidence_source_ids"] = [
                    remap_source_ref(ref, message_id_map=message_id_map)
                    if isinstance(ref, str)
                    else ref
                    for ref in evidence
                ]
            remapped_effects.append(effect)
        remapped["effects"] = remapped_effects
    return remapped


def remap_source_ref(source_id: str, *, message_id_map: Mapping[str, str]) -> str:
    if source_id.startswith("message:"):
        message_id = source_id.removeprefix("message:")
        mapped = message_id_map.get(message_id)
        if mapped is None:
            return source_id
        return f"message:{mapped}"
    mapped = message_id_map.get(source_id)
    if mapped is None:
        return source_id
    return mapped


def _text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _float(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int) and value > 0:
        return value
    return 0


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


__all__ = [
    "TurnOutcome",
    "TurnOutcomeEffect",
    "character_emotional_state_key",
    "character_physical_state_key",
    "character_relationships_state_key",
    "remap_turn_outcome_payload",
    "remap_source_ref",
    "turn_outcome_coverage",
    "turn_outcome_from_mapping",
]
