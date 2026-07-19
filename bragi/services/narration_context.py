"""Shared save read snapshot for narration hot paths."""

from __future__ import annotations

from dataclasses import dataclass

from bragi.persistence.models import (
    ActiveThreadRecord,
    CharacterKnowledgeEdgeRecord,
    CharacterRecord,
    ContextObservationRecord,
    ContextSourceRecord,
    ContextUpdateSuggestionRecord,
    EntityLinkRecord,
    LocationRecord,
    MediaAssetRecord,
    MemoryRecord,
    MessageVisibilityRecord,
    SaveDetailsRecord,
    SceneSnapshotRecord,
    StateChangeRecord,
    SummaryRecord,
    WorldStateRecord,
)
from bragi.persistence.repositories import PersistenceRepositories


@dataclass(frozen=True)
class NarrationContextSnapshot:
    details: SaveDetailsRecord
    scene_snapshot: SceneSnapshotRecord | None
    locations: tuple[LocationRecord, ...]
    characters: tuple[CharacterRecord, ...]
    active_threads: tuple[ActiveThreadRecord, ...]
    character_knowledge_edges: tuple[CharacterKnowledgeEdgeRecord, ...]
    message_visibility: tuple[MessageVisibilityRecord, ...]
    entity_links: tuple[EntityLinkRecord, ...]
    world_state: tuple[WorldStateRecord, ...]
    world_state_for_scope: tuple[WorldStateRecord, ...]
    state_changes: tuple[StateChangeRecord, ...]
    media_assets: tuple[MediaAssetRecord, ...]
    memories: tuple[MemoryRecord, ...]
    summaries: tuple[SummaryRecord, ...]
    observations: tuple[ContextObservationRecord, ...]
    context_sources: tuple[ContextSourceRecord, ...]
    pending_context_suggestions: tuple[ContextUpdateSuggestionRecord, ...]


def load_narration_context_snapshot(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    details: SaveDetailsRecord | None = None,
    include_context_sources: bool = True,
) -> NarrationContextSnapshot | None:
    details = details or repositories.load_save_details(save_id)
    if details is None:
        return None
    return NarrationContextSnapshot(
        details=details,
        scene_snapshot=repositories.get_scene_snapshot(save_id),
        locations=tuple(repositories.list_locations(save_id)),
        characters=tuple(repositories.list_characters(save_id)),
        active_threads=tuple(repositories.list_active_threads(save_id)),
        character_knowledge_edges=tuple(
            repositories.list_character_knowledge_edges(save_id)
        ),
        message_visibility=tuple(repositories.list_message_visibility(save_id)),
        entity_links=tuple(repositories.list_entity_links(save_id)),
        world_state=tuple(repositories.list_world_state(save_id)),
        world_state_for_scope=tuple(
            repositories.list_world_state_including_archived(save_id)
        ),
        state_changes=tuple(repositories.list_state_changes(save_id)),
        media_assets=tuple(repositories.list_media_assets(save_id)),
        memories=tuple(repositories.list_memories(save_id)),
        summaries=tuple(repositories.list_summaries(save_id)),
        observations=tuple(repositories.list_context_observations(save_id)),
        context_sources=(
            tuple(repositories.list_context_sources(save_id))
            if include_context_sources
            else ()
        ),
        pending_context_suggestions=tuple(
            repositories.list_context_update_suggestions(save_id, status="pending")
        ),
    )
