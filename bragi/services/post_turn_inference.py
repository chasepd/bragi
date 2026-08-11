"""Post-turn narrator-prose inference mode helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bragi.persistence.repositories import PersistenceRepositories

POST_TURN_INFERENCE_MODE_SETTING = "post_turn_inference_mode"
POST_TURN_INFERENCE_MODE_LEGACY = "legacy"
POST_TURN_INFERENCE_MODE_HYBRID = "hybrid"
POST_TURN_INFERENCE_MODE_PLAN_OWNED = "plan_owned"
POST_TURN_INFERENCE_MODE_DEFAULT = POST_TURN_INFERENCE_MODE_PLAN_OWNED
POST_TURN_INFERENCE_MODES = frozenset(
    {
        POST_TURN_INFERENCE_MODE_LEGACY,
        POST_TURN_INFERENCE_MODE_HYBRID,
        POST_TURN_INFERENCE_MODE_PLAN_OWNED,
    }
)
POST_TURN_INFERENCE_MODE_OPTIONS = (
    POST_TURN_INFERENCE_MODE_LEGACY,
    POST_TURN_INFERENCE_MODE_HYBRID,
    POST_TURN_INFERENCE_MODE_PLAN_OWNED,
)

POST_TURN_DOMAIN_STATE = "state"
POST_TURN_DOMAIN_SCENE = "scene"
POST_TURN_DOMAIN_PHYSICAL = "physical"
POST_TURN_DOMAIN_RELATIONSHIP = "relationship"
POST_TURN_DOMAIN_EMOTIONAL = "emotional"
POST_TURN_DOMAIN_KNOWLEDGE = "knowledge"
POST_TURN_DOMAIN_THREAD_CLOCK = "thread_clock"
POST_TURN_DOMAIN_RESOURCE = "resource"
POST_TURN_DOMAIN_TIME = "time"
ALL_POST_TURN_DOMAINS = frozenset(
    {
        POST_TURN_DOMAIN_STATE,
        POST_TURN_DOMAIN_SCENE,
        POST_TURN_DOMAIN_PHYSICAL,
        POST_TURN_DOMAIN_RELATIONSHIP,
        POST_TURN_DOMAIN_EMOTIONAL,
        POST_TURN_DOMAIN_KNOWLEDGE,
        POST_TURN_DOMAIN_THREAD_CLOCK,
        POST_TURN_DOMAIN_RESOURCE,
        POST_TURN_DOMAIN_TIME,
    }
)

PLANNED_EFFECT_TYPE_TO_DOMAIN: dict[str, str] = {
    "scene_presence": POST_TURN_DOMAIN_SCENE,
    "scene_snapshot_field": POST_TURN_DOMAIN_SCENE,
    "character_learned_memory": POST_TURN_DOMAIN_KNOWLEDGE,
    "character_knowledge_edge": POST_TURN_DOMAIN_KNOWLEDGE,
    "physical_change": POST_TURN_DOMAIN_PHYSICAL,
    "relationship_change": POST_TURN_DOMAIN_RELATIONSHIP,
    "emotional_change": POST_TURN_DOMAIN_EMOTIONAL,
    "active_thread_change": POST_TURN_DOMAIN_THREAD_CLOCK,
    "resource_change": POST_TURN_DOMAIN_RESOURCE,
    "world_state_change": POST_TURN_DOMAIN_STATE,
    "world_time_change": POST_TURN_DOMAIN_TIME,
}


def planned_effect_domain(candidate_type: str) -> str:
    return PLANNED_EFFECT_TYPE_TO_DOMAIN.get(candidate_type, "unknown")


@dataclass(frozen=True)
class VerifiedPostTurnCoverage:
    source_message_ids: tuple[str, ...] = ()
    state_keys: frozenset[str] = frozenset()
    scene_snapshot_fields: frozenset[str] = frozenset()
    scene_presence_character_ids: frozenset[str] = frozenset()
    memory_fingerprints: frozenset[str] = frozenset()
    knowledge_edge_targets: frozenset[tuple[str, str, str]] = frozenset()
    applied_domains: frozenset[str] = frozenset()
    queued_domains: frozenset[str] = frozenset()
    committed_count: int = 0
    confirmation_queued_count: int = 0
    metadata: dict[str, object] = field(default_factory=dict, compare=False)

    @property
    def empty(self) -> bool:
        return not (
            self.state_keys
            or self.scene_snapshot_fields
            or self.scene_presence_character_ids
            or self.memory_fingerprints
            or self.knowledge_edge_targets
            or self.applied_domains
            or self.queued_domains
            or self.committed_count
            or self.confirmation_queued_count
        )

    def to_json(self) -> dict[str, object]:
        return {
            "source_message_ids": list(self.source_message_ids),
            "state_keys": sorted(self.state_keys),
            "scene_snapshot_fields": sorted(self.scene_snapshot_fields),
            "scene_presence_character_ids": sorted(self.scene_presence_character_ids),
            "memory_count": len(self.memory_fingerprints),
            "memory_fingerprints": sorted(self.memory_fingerprints),
            "knowledge_edge_count": len(self.knowledge_edge_targets),
            "knowledge_edge_targets": [
                list(target) for target in sorted(self.knowledge_edge_targets)
            ],
            "applied_domains": sorted(self.applied_domains),
            "queued_domains": sorted(self.queued_domains),
            "committed_count": self.committed_count,
            "confirmation_queued_count": self.confirmation_queued_count,
            **dict(self.metadata),
        }


def post_turn_inference_mode(
    repositories: PersistenceRepositories,
    *,
    save_id: str | None = None,
) -> str:
    return sanitize_post_turn_inference_mode(
        repositories.get_effective_setting(
            POST_TURN_INFERENCE_MODE_SETTING,
            save_id=save_id,
        )
    )


def sanitize_post_turn_inference_mode(value: object) -> str:
    mode = value.strip() if isinstance(value, str) else ""
    if mode in POST_TURN_INFERENCE_MODES:
        return mode
    return POST_TURN_INFERENCE_MODE_DEFAULT


def memory_fingerprint(body: str) -> str:
    normalized = re.sub(r"\s+", " ", body.casefold()).strip()
    return sha256(normalized.encode("utf-8")).hexdigest()


def verified_post_turn_coverage_from_mapping(
    value: object,
) -> VerifiedPostTurnCoverage:
    if not isinstance(value, Mapping):
        return VerifiedPostTurnCoverage()
    source_message_ids = _string_tuple(value.get("source_message_ids"))
    state_keys = frozenset(_string_tuple(value.get("state_keys")))
    scene_snapshot_fields = frozenset(_string_tuple(value.get("scene_snapshot_fields")))
    scene_presence_character_ids = frozenset(
        _string_tuple(value.get("scene_presence_character_ids"))
    )
    memory_fingerprints = frozenset(_string_tuple(value.get("memory_fingerprints")))
    knowledge_edge_targets = frozenset(
        _knowledge_edge_target_tuple(value.get("knowledge_edge_targets"))
    )
    applied_domains = frozenset(_string_tuple(value.get("applied_domains")))
    queued_domains = frozenset(_string_tuple(value.get("queued_domains")))
    return VerifiedPostTurnCoverage(
        source_message_ids=source_message_ids,
        state_keys=state_keys,
        scene_snapshot_fields=scene_snapshot_fields,
        scene_presence_character_ids=scene_presence_character_ids,
        memory_fingerprints=memory_fingerprints,
        knowledge_edge_targets=knowledge_edge_targets,
        applied_domains=applied_domains,
        queued_domains=queued_domains,
        committed_count=_nonnegative_int(value.get("committed_count")),
        confirmation_queued_count=_nonnegative_int(
            value.get("confirmation_queued_count")
        ),
        metadata=_coverage_metadata(value),
    )


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _knowledge_edge_target_tuple(value: object) -> tuple[tuple[str, str, str], ...]:
    if not isinstance(value, list):
        return ()
    targets: list[tuple[str, str, str]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 3:
            continue
        character_id, target_type, target_id = (
            str(part).strip() for part in item
        )
        if character_id and target_type and target_id:
            targets.append((character_id, target_type, target_id))
    return tuple(targets)


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int) and value > 0:
        return value
    return 0


def _coverage_metadata(value: Mapping[object, object]) -> dict[str, object]:
    known_keys = {
        "source_message_ids",
        "state_keys",
        "scene_snapshot_fields",
        "scene_presence_character_ids",
        "memory_count",
        "memory_fingerprints",
        "knowledge_edge_count",
        "knowledge_edge_targets",
        "applied_domains",
        "queued_domains",
        "committed_count",
        "confirmation_queued_count",
    }
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(key, str) and key not in known_keys
    }
