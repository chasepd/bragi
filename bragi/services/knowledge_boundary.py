"""Helpers for per-character knowledge and message visibility boundaries."""

from __future__ import annotations

from dataclasses import dataclass

from bragi.persistence.models import (
    CharacterKnowledgeEdgeRecord,
    CharacterRecord,
    EntityLinkRecord,
    MessageVisibilityRecord,
    SceneSnapshotRecord,
)
from bragi.services.mention_matching import character_name_is_mentioned

SCOPED_MAY_KNOW_CONFIDENCE_THRESHOLD = 0.7
CHARACTER_TEXT_SOURCE_PREFIX = "character_text_message:"
MAX_SCOPED_PRESENT_CHARACTER_IDS = 64


@dataclass(frozen=True)
class ScopedTargets:
    allowed: dict[tuple[str, str], tuple[str, ...]]
    blocked: set[tuple[str, str]]


@dataclass(frozen=True)
class TurnCharacterScope:
    present_character_ids: frozenset[str]
    mentioned_character_ids: frozenset[str]

    @property
    def reference_character_ids(self) -> frozenset[str]:
        return self.present_character_ids | self.mentioned_character_ids


def character_scope_for_turn(
    *,
    scene_snapshot: SceneSnapshotRecord | None,
    characters: list[CharacterRecord],
    latest_player_message: str,
) -> TurnCharacterScope:
    snapshot_present_ids = (
        scene_snapshot.present_character_ids if scene_snapshot is not None else []
    )
    present_ids = frozenset(snapshot_present_ids)
    mentioned_ids = {
        character.id
        for character in characters
        if character_name_is_mentioned(
            name=character.name,
            aliases=character.aliases,
            text=latest_player_message,
        )
    }
    return TurnCharacterScope(
        present_character_ids=present_ids,
        mentioned_character_ids=frozenset(mentioned_ids),
    )


def allowed_character_scoped_targets(
    *,
    scene_snapshot: SceneSnapshotRecord | None,
    characters: list[CharacterRecord],
    character_knowledge_edges: list[CharacterKnowledgeEdgeRecord],
    entity_links: list[EntityLinkRecord],
    latest_player_message: str,
    message_visibility: list[MessageVisibilityRecord] | None = None,
) -> ScopedTargets:
    characters_by_id = {character.id: character for character in characters}
    present_ids = character_scope_for_turn(
        scene_snapshot=scene_snapshot,
        characters=characters,
        latest_player_message=latest_player_message,
    ).present_character_ids
    if len(present_ids) > MAX_SCOPED_PRESENT_CHARACTER_IDS:
        return ScopedTargets(
            allowed={},
            blocked=_all_character_scoped_targets(
                character_knowledge_edges=character_knowledge_edges,
                entity_links=entity_links,
            ),
        )
    allowed: dict[tuple[str, str], tuple[str, ...]] = {}
    blocked: set[tuple[str, str]] = set()
    graph_targets: set[tuple[str, str, str]] = set()
    restrictive_graph_targets = {
        (
            edge.character_id,
            normalized_knowledge_target_type(edge.target_type),
            edge.target_id,
        )
        for edge in character_knowledge_edges
        if _knowledge_edge_is_prompt_blocked(
            edge,
            characters_by_id=characters_by_id,
            present_ids=present_ids,
            message_visibility=message_visibility or [],
        )
    }
    for edge in character_knowledge_edges:
        target_type = normalized_knowledge_target_type(edge.target_type)
        if target_type not in {"memory", "world_state", "summary", "scenario_section"}:
            continue
        target = (target_type, edge.target_id)
        graph_target = (edge.character_id, *target)
        graph_targets.add(graph_target)
        if graph_target in restrictive_graph_targets:
            blocked.add(target)
            continue
        character = characters_by_id.get(edge.character_id)
        if character is not None:
            allowed[target] = _append_scope_label(
                allowed.get(target, ()),
                knowledge_edge_scope_label(edge, character),
            )
    for link in entity_links:
        if link.entity_type != "character" or link.relation != "knows":
            continue
        target_type = normalized_knowledge_target_type(link.target_type)
        if target_type not in {"memory", "world_state", "summary"}:
            continue
        target = (target_type, link.target_id)
        if (link.entity_id, *target) in graph_targets:
            continue
        if link.source_message_id is not None and not (
            message_visible_to_character(
                message_id=link.source_message_id,
                character_id=link.entity_id,
                message_visibility=message_visibility or [],
            )
        ):
            blocked.add(target)
            continue
        character = characters_by_id.get(link.entity_id)
        if character is not None and link.entity_id in present_ids:
            allowed[target] = _append_scope_label(
                allowed.get(target, ()),
                f"{character.name} knows",
            )
        else:
            blocked.add(target)
    return ScopedTargets(allowed=allowed, blocked=blocked - set(allowed))


def _all_character_scoped_targets(
    *,
    character_knowledge_edges: list[CharacterKnowledgeEdgeRecord],
    entity_links: list[EntityLinkRecord],
) -> set[tuple[str, str]]:
    targets = {
        (normalized_knowledge_target_type(edge.target_type), edge.target_id)
        for edge in character_knowledge_edges
    }
    targets.update(
        (normalized_knowledge_target_type(link.target_type), link.target_id)
        for link in entity_links
        if link.entity_type == "character" and link.relation == "knows"
    )
    return {
        target
        for target in targets
        if target[0] in {"memory", "world_state", "summary", "scenario_section"}
    }


def _knowledge_edge_is_prompt_blocked(
    edge: CharacterKnowledgeEdgeRecord,
    *,
    characters_by_id: dict[str, CharacterRecord],
    present_ids: frozenset[str],
    message_visibility: list[MessageVisibilityRecord],
) -> bool:
    source_message_ids = tuple(
        dict.fromkeys(
            (
                *edge.source_message_ids,
                *([edge.source_message_id] if edge.source_message_id else []),
            )
        )
    )
    if any(
        not message_visible_to_character(
            message_id=source_id,
            character_id=edge.character_id,
            message_visibility=message_visibility,
        )
        for source_id in source_message_ids
    ):
        return True
    character = characters_by_id.get(edge.character_id)
    if (
        character is not None
        and character.is_player_character
        and knowledge_edge_has_character_text_source(edge)
    ):
        return True
    return (
        edge.character_id not in present_ids
        or not knowledge_edge_allows_prompt_use(edge)
    )


def _append_scope_label(existing: tuple[str, ...], label: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*existing, label)))


def message_visible_to_present_characters(
    *,
    message_id: str,
    present_character_ids: frozenset[str],
    message_visibility: list[MessageVisibilityRecord],
) -> bool:
    if not present_character_ids:
        return True
    return not any(
        record.message_id == message_id
        and record.character_id in present_character_ids
        and record.visibility == "not_visible"
        for record in message_visibility
    )


def message_visible_to_character(
    *,
    message_id: str,
    character_id: str,
    message_visibility: list[MessageVisibilityRecord],
) -> bool:
    return not any(
        record.message_id == message_id
        and record.character_id == character_id
        and record.visibility == "not_visible"
        for record in message_visibility
    )


def knowledge_edge_allows_prompt_use(edge: CharacterKnowledgeEdgeRecord) -> bool:
    if edge.knowledge_state == "knows":
        return True
    return (
        edge.knowledge_state == "may_know"
        and edge.confidence >= SCOPED_MAY_KNOW_CONFIDENCE_THRESHOLD
    )


def knowledge_edge_scope_label(
    edge: CharacterKnowledgeEdgeRecord,
    character: CharacterRecord,
) -> str:
    if edge.knowledge_state == "may_know":
        return f"{character.name} may know"
    return f"{character.name} knows"


def scoped_owner_name(scope_label: str) -> str:
    for suffix in (" knows", " may know"):
        if scope_label.endswith(suffix):
            return scope_label.removesuffix(suffix)
    return scope_label


def normalized_knowledge_target_type(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized in {"state", "world_state"}:
        return "world_state"
    if normalized in {"memory", "memories"}:
        return "memory"
    return normalized


def knowledge_edge_has_character_text_source(
    edge: CharacterKnowledgeEdgeRecord,
) -> bool:
    source_ids = list(edge.source_message_ids)
    if edge.source_message_id:
        source_ids.append(edge.source_message_id)
    return any(
        source_id.startswith(CHARACTER_TEXT_SOURCE_PREFIX)
        for source_id in source_ids
    )
