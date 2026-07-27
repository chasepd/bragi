"""Shared save read snapshot for narration hot paths."""

from __future__ import annotations

from dataclasses import dataclass, replace

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
    raw_record_limit: int | None = None,
) -> NarrationContextSnapshot | None:
    details = details or repositories.load_save_details(save_id)
    if details is None:
        return None
    scene_snapshot = repositories.get_scene_snapshot(save_id)
    visibility_character_ids = (
        set(scene_snapshot.present_character_ids)
        if raw_record_limit is not None and scene_snapshot is not None
        else None
    )
    if raw_record_limit is not None:
        details = replace(
            details,
            messages=repositories.list_recent_messages_visible_to_characters(
                save_id,
                character_ids=visibility_character_ids or set(),
                limit=len(details.messages),
            ),
        )
    world_state = tuple(
        repositories.list_world_state(save_id, limit=raw_record_limit)
    )
    world_state_for_scope = tuple(
        repositories.list_world_state_including_archived(
            save_id,
            limit=raw_record_limit,
        )
    )
    state_changes = tuple(
        repositories.list_state_changes(save_id, limit=raw_record_limit)
    )
    media_assets = tuple(
        repositories.list_media_assets(save_id, limit=raw_record_limit)
    )
    memories = tuple(
        repositories.list_memories(save_id, limit=raw_record_limit)
    )
    summaries = tuple(
        repositories.list_summaries(save_id, limit=raw_record_limit)
    )
    observations = tuple(
        repositories.list_context_observations(
            save_id,
            limit=raw_record_limit,
        )
    )
    pending_suggestions = tuple(
        repositories.list_context_update_suggestions(
            save_id,
            status="pending",
            limit=raw_record_limit,
        )
    )
    visibility_message_ids = (
        {
            *(message.id for message in details.messages),
            *(
                state.source_message_id
                for state in (*world_state, *world_state_for_scope)
                if state.source_message_id is not None
            ),
            *(
                change.source_message_id
                for change in state_changes
                if change.source_message_id is not None
            ),
            *(
                asset.source_message_id
                for asset in media_assets
                if asset.source_message_id is not None
            ),
            *(
                source_id
                for memory in memories
                for source_id in memory.source_message_ids
            ),
            *(
                source_id
                for summary in summaries
                for source_id in (
                    summary.covers_message_start_id,
                    summary.covers_message_end_id,
                )
            ),
            *(
                source_id
                for observation in observations
                for source_id in observation.source_message_ids
            ),
            *(
                source_id
                for suggestion in pending_suggestions
                for source_id in suggestion.source_message_ids
            ),
        }
        if raw_record_limit is not None
        else None
    )
    return NarrationContextSnapshot(
        details=details,
        scene_snapshot=scene_snapshot,
        locations=tuple(repositories.list_locations(save_id)),
        characters=tuple(repositories.list_characters(save_id)),
        active_threads=tuple(repositories.list_active_threads(save_id)),
        character_knowledge_edges=tuple(
            repositories.list_character_knowledge_edges(save_id)
        ),
        message_visibility=tuple(
            repositories.list_message_visibility(
                save_id,
                character_ids=visibility_character_ids,
                message_ids=visibility_message_ids,
            )
        ),
        entity_links=tuple(repositories.list_entity_links(save_id)),
        world_state=world_state,
        world_state_for_scope=world_state_for_scope,
        state_changes=state_changes,
        media_assets=media_assets,
        memories=memories,
        summaries=summaries,
        observations=observations,
        context_sources=(
            tuple(repositories.list_context_sources(save_id))
            if include_context_sources
            else ()
        ),
        pending_context_suggestions=pending_suggestions,
    )
