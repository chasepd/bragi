from __future__ import annotations

from bragi.persistence.models import (
    CharacterKnowledgeEdgeRecord,
    CharacterRecord,
    EntityLinkRecord,
    MessageVisibilityRecord,
    SceneSnapshotRecord,
)
from bragi.services.knowledge_boundary import (
    allowed_character_scoped_targets,
    character_scope_for_turn,
    message_visible_to_character,
    message_visible_to_present_characters,
)


def test_character_scope_splits_present_and_mentioned_references() -> None:
    present = _character("character-ilyra", name="Captain Ilyra", aliases=["Ilyra"])
    absent = _character("character-ren", name="Archivist Ren", aliases=["Ren"])
    snapshot = _scene_snapshot(present_character_ids=[present.id])

    scope = character_scope_for_turn(
        scene_snapshot=snapshot,
        characters=[present, absent],
        latest_player_message="I ask Ilyra whether Ren touched the lens.",
    )

    assert scope.present_character_ids == frozenset({present.id})
    assert scope.mentioned_character_ids == frozenset({present.id, absent.id})
    assert scope.reference_character_ids == frozenset({present.id, absent.id})


def test_character_scope_has_only_mentions_without_scene_snapshot() -> None:
    absent = _character("character-ren", name="Archivist Ren", aliases=["Ren"])

    scope = character_scope_for_turn(
        scene_snapshot=None,
        characters=[absent],
        latest_player_message="Ren, what did you hide in the ledger?",
    )

    assert scope.present_character_ids == frozenset()
    assert scope.mentioned_character_ids == frozenset({absent.id})
    assert scope.reference_character_ids == frozenset({absent.id})


def test_scoped_targets_only_unlock_for_present_characters() -> None:
    present = _character("character-sienna", name="Sienna")
    absent = _character("character-ren", name="Archivist Ren", aliases=["Ren"])
    snapshot = _scene_snapshot(present_character_ids=[present.id])
    present_memory_id = "memory-sienna-current-scene"
    absent_memory_id = "memory-ren-private-ledger"

    targets = allowed_character_scoped_targets(
        scene_snapshot=snapshot,
        characters=[present, absent],
        character_knowledge_edges=[
            _knowledge_edge(
                character_id=present.id,
                target_type="memory",
                target_id=present_memory_id,
            ),
            _knowledge_edge(
                character_id=absent.id,
                target_type="memory",
                target_id=absent_memory_id,
            ),
        ],
        entity_links=[
            _entity_link(
                entity_id=absent.id,
                target_type="world_state",
                target_id="state-ren-ledger",
            )
        ],
        latest_player_message="I ask Sienna what Ren would know about this.",
    )

    assert targets.allowed == {
        ("memory", present_memory_id): ("Sienna knows",),
    }
    assert ("memory", absent_memory_id) in targets.blocked
    assert ("world_state", "state-ren-ledger") in targets.blocked


def test_scoped_targets_supersede_legacy_links_by_owner_and_target() -> None:
    present = _character("character-sienna", name="Sienna")
    absent = _character("character-ren", name="Archivist Ren")
    snapshot = _scene_snapshot(present_character_ids=[present.id])
    shared_memory_id = "memory-shared-ledger"

    targets = allowed_character_scoped_targets(
        scene_snapshot=snapshot,
        characters=[present, absent],
        character_knowledge_edges=[
            _knowledge_edge(
                character_id=absent.id,
                target_type="memory",
                target_id=shared_memory_id,
            ),
        ],
        entity_links=[
            _entity_link(
                entity_id=present.id,
                target_type="memory",
                target_id=shared_memory_id,
            ),
        ],
        latest_player_message="I ask Sienna about the ledger.",
    )

    assert targets.allowed == {
        ("memory", shared_memory_id): ("Sienna knows",),
    }
    assert ("memory", shared_memory_id) not in targets.blocked


def test_restrictive_alias_edge_dominates_for_same_character_and_target() -> None:
    present = _character("character-sienna", name="Sienna")
    snapshot = _scene_snapshot(present_character_ids=[present.id])
    target_id = "state-secret"

    targets = allowed_character_scoped_targets(
        scene_snapshot=snapshot,
        characters=[present],
        character_knowledge_edges=[
            _knowledge_edge(
                character_id=present.id,
                target_type="state",
                target_id=target_id,
            ),
            _knowledge_edge(
                character_id=present.id,
                target_type="world_state",
                target_id=target_id,
                knowledge_state="does_not_know",
            ),
        ],
        entity_links=[],
        latest_player_message="I ask Sienna about the secret.",
    )

    assert ("world_state", target_id) not in targets.allowed
    assert ("world_state", target_id) in targets.blocked


def test_legacy_plural_memory_edge_blocks_scoped_target() -> None:
    present = _character("character-sienna", name="Sienna")
    snapshot = _scene_snapshot(present_character_ids=[present.id])
    target_id = "memory-secret"

    targets = allowed_character_scoped_targets(
        scene_snapshot=snapshot,
        characters=[present],
        character_knowledge_edges=[
            _knowledge_edge(
                character_id=present.id,
                target_type="memories",
                target_id=target_id,
                knowledge_state="does_not_know",
            )
        ],
        entity_links=[],
        latest_player_message="I ask Sienna about the secret.",
    )

    assert ("memory", target_id) not in targets.allowed
    assert ("memory", target_id) in targets.blocked


def test_scoped_targets_fail_closed_when_scene_exceeds_graph_character_limit() -> None:
    characters = [
        _character(f"character-{index:02d}", name=f"Character {index}")
        for index in range(65)
    ]
    last_character = characters[-1]
    target_id = "state-private-to-oversized-scene"
    snapshot = _scene_snapshot(
        present_character_ids=[character.id for character in characters]
    )

    targets = allowed_character_scoped_targets(
        scene_snapshot=snapshot,
        characters=characters,
        character_knowledge_edges=[
            _knowledge_edge(
                character_id=last_character.id,
                target_type="state",
                target_id=target_id,
            )
        ],
        entity_links=[],
        latest_player_message="I ask about the private state.",
    )

    assert ("world_state", target_id) not in targets.allowed
    assert ("world_state", target_id) in targets.blocked


def test_scalar_knowledge_provenance_is_checked_for_visibility() -> None:
    present = _character("character-sienna", name="Sienna")
    snapshot = _scene_snapshot(present_character_ids=[present.id])
    hidden_message_id = "message-hidden"

    targets = allowed_character_scoped_targets(
        scene_snapshot=snapshot,
        characters=[present],
        character_knowledge_edges=[
            _knowledge_edge(
                character_id=present.id,
                target_type="memory",
                target_id="memory-secret",
                source_message_id=hidden_message_id,
            ),
        ],
        entity_links=[],
        latest_player_message="I ask Sienna about the secret.",
        message_visibility=[
            _message_visibility(
                message_id=hidden_message_id,
                character_id=present.id,
            )
        ],
    )

    assert ("memory", "memory-secret") not in targets.allowed
    assert ("memory", "memory-secret") in targets.blocked


def test_absent_mentions_do_not_hide_messages_from_present_scene() -> None:
    present = _character("character-sienna", name="Sienna")
    absent = _character("character-ren", name="Archivist Ren", aliases=["Ren"])
    hidden_from_absent = _message_visibility(
        message_id="message-private-current-scene",
        character_id=absent.id,
    )

    assert message_visible_to_present_characters(
        message_id=hidden_from_absent.message_id,
        present_character_ids=frozenset({present.id}),
        message_visibility=[hidden_from_absent],
    )
    assert not message_visible_to_present_characters(
        message_id=hidden_from_absent.message_id,
        present_character_ids=frozenset({absent.id}),
        message_visibility=[hidden_from_absent],
    )


def test_visibility_is_projected_per_character_in_mixed_knowledge_scene() -> None:
    secret = "message-secret"
    informed = _character("character-informed", name="Informed")
    uninformed = _character("character-uninformed", name="Uninformed")
    visibility = [
        _message_visibility(message_id=secret, character_id=uninformed.id),
    ]

    assert message_visible_to_character(
        message_id=secret,
        character_id=informed.id,
        message_visibility=visibility,
    )
    assert not message_visible_to_character(
        message_id=secret,
        character_id=uninformed.id,
        message_visibility=visibility,
    )


def test_informed_present_character_unlocks_fact_hidden_from_companion() -> None:
    secret = "message-secret"
    informed = _character("character-informed", name="Informed")
    uninformed = _character("character-uninformed", name="Uninformed")

    targets = allowed_character_scoped_targets(
        scene_snapshot=_scene_snapshot(
            present_character_ids=[informed.id, uninformed.id]
        ),
        characters=[informed, uninformed],
        character_knowledge_edges=[
            _knowledge_edge(
                character_id=informed.id,
                target_type="memory",
                target_id="memory-secret",
                source_message_id=secret,
            )
        ],
        entity_links=[],
        latest_player_message="I wait for an answer.",
        message_visibility=[
            _message_visibility(message_id=secret, character_id=uninformed.id)
        ],
    )

    assert targets.allowed == {
        ("memory", "memory-secret"): ("Informed knows",),
    }


def _character(
    character_id: str,
    *,
    name: str,
    aliases: list[str] | None = None,
) -> CharacterRecord:
    return CharacterRecord(
        id=character_id,
        save_id="save-knowledge-boundary",
        name=name,
        aliases=list(aliases or []),
        role="",
        known_state="",
        met=True,
        appearance="",
        visual_notes="",
        current_clothing="",
        personality="",
        voice="",
        relationships={},
        status="",
        location_id=None,
        private_notes="",
        source_message_id=None,
        locked_fields=[],
    )


def _scene_snapshot(*, present_character_ids: list[str]) -> SceneSnapshotRecord:
    return SceneSnapshotRecord(
        id="snapshot-knowledge-boundary",
        save_id="save-knowledge-boundary",
        current_location_id=None,
        situation="",
        objective="",
        in_world_time="",
        time_of_day="",
        day_of_week="",
        weather="",
        mood="",
        nearby_objects=[],
        hazards=[],
        present_character_ids=present_character_ids,
        source_message_id=None,
        locked_fields=[],
    )


def _knowledge_edge(
    *,
    character_id: str,
    target_type: str,
    target_id: str,
    knowledge_state: str = "knows",
    source_message_id: str | None = None,
) -> CharacterKnowledgeEdgeRecord:
    return CharacterKnowledgeEdgeRecord(
        id=f"edge-{character_id}-{target_id}",
        save_id="save-knowledge-boundary",
        character_id=character_id,
        target_type=target_type,
        target_id=target_id,
        knowledge_state=knowledge_state,
        acquisition_method="witnessed",
        confidence=1.0,
        source_message_id=source_message_id,
    )


def _entity_link(
    *,
    entity_id: str,
    target_type: str,
    target_id: str,
) -> EntityLinkRecord:
    return EntityLinkRecord(
        id=f"link-{entity_id}-{target_id}",
        save_id="save-knowledge-boundary",
        entity_type="character",
        entity_id=entity_id,
        target_type=target_type,
        target_id=target_id,
        relation="knows",
    )


def _message_visibility(
    *,
    message_id: str,
    character_id: str,
) -> MessageVisibilityRecord:
    return MessageVisibilityRecord(
        id=f"visibility-{message_id}-{character_id}",
        save_id="save-knowledge-boundary",
        message_id=message_id,
        character_id=character_id,
        visibility="not_visible",
        confidence=1.0,
        source="scene_presence",
        evidence="The character was absent.",
    )
