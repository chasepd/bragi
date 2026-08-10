"""Unit tests for post-turn inference modes and domain coverage."""

from __future__ import annotations

from bragi.services.post_turn_inference import (
    ALL_POST_TURN_DOMAINS,
    PLANNED_EFFECT_TYPE_TO_DOMAIN,
    POST_TURN_DOMAIN_EMOTIONAL,
    POST_TURN_DOMAIN_KNOWLEDGE,
    POST_TURN_DOMAIN_PHYSICAL,
    POST_TURN_DOMAIN_RELATIONSHIP,
    POST_TURN_DOMAIN_RESOURCE,
    POST_TURN_DOMAIN_SCENE,
    POST_TURN_DOMAIN_STATE,
    POST_TURN_DOMAIN_THREAD_CLOCK,
    POST_TURN_DOMAIN_TIME,
    POST_TURN_INFERENCE_MODE_DEFAULT,
    POST_TURN_INFERENCE_MODES,
    VerifiedPostTurnCoverage,
    planned_effect_domain,
    sanitize_post_turn_inference_mode,
    verified_post_turn_coverage_from_mapping,
)


def test_all_domains_are_covered_by_planned_effect_types() -> None:
    mapped_domains = frozenset(PLANNED_EFFECT_TYPE_TO_DOMAIN.values())
    assert mapped_domains == ALL_POST_TURN_DOMAINS


def test_planned_effect_domain_mapping() -> None:
    assert planned_effect_domain("scene_presence") == POST_TURN_DOMAIN_SCENE
    assert planned_effect_domain("scene_snapshot_field") == POST_TURN_DOMAIN_SCENE
    assert planned_effect_domain("character_learned_memory") == (
        POST_TURN_DOMAIN_KNOWLEDGE
    )
    assert planned_effect_domain("character_knowledge_edge") == (
        POST_TURN_DOMAIN_KNOWLEDGE
    )
    assert planned_effect_domain("physical_change") == POST_TURN_DOMAIN_PHYSICAL
    assert planned_effect_domain("relationship_change") == (
        POST_TURN_DOMAIN_RELATIONSHIP
    )
    assert planned_effect_domain("emotional_change") == POST_TURN_DOMAIN_EMOTIONAL
    assert planned_effect_domain("active_thread_change") == (
        POST_TURN_DOMAIN_THREAD_CLOCK
    )
    assert planned_effect_domain("resource_change") == POST_TURN_DOMAIN_RESOURCE
    assert planned_effect_domain("world_state_change") == POST_TURN_DOMAIN_STATE
    assert planned_effect_domain("world_time_change") == POST_TURN_DOMAIN_TIME
    assert planned_effect_domain("unknown_type") == "unknown"


def test_sanitize_post_turn_inference_mode() -> None:
    assert sanitize_post_turn_inference_mode("plan_owned") == "plan_owned"
    assert (
        sanitize_post_turn_inference_mode("bogus") == POST_TURN_INFERENCE_MODE_DEFAULT
    )
    assert sanitize_post_turn_inference_mode(None) == POST_TURN_INFERENCE_MODE_DEFAULT
    assert POST_TURN_INFERENCE_MODES == frozenset(
        {"legacy", "hybrid", "plan_owned"}
    )


def test_verified_coverage_round_trips_domains() -> None:
    coverage = VerifiedPostTurnCoverage(
        source_message_ids=("m1", "m2"),
        state_keys=frozenset({"character.mara.physical_state"}),
        applied_domains=frozenset({POST_TURN_DOMAIN_PHYSICAL}),
        queued_domains=frozenset({POST_TURN_DOMAIN_KNOWLEDGE}),
        committed_count=1,
        confirmation_queued_count=1,
        metadata={"planned_commit_verifier_available": True},
    )
    restored = verified_post_turn_coverage_from_mapping(coverage.to_json())
    assert restored.applied_domains == coverage.applied_domains
    assert restored.queued_domains == coverage.queued_domains
    assert restored.state_keys == coverage.state_keys
    assert restored.metadata["planned_commit_verifier_available"] is True
