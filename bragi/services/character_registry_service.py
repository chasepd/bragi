"""Import-safe character registry model and persistence service."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import cast

from bragi.content_rating_instructions import (
    content_rating_exceeds,
    maximum_content_rating,
)
from bragi.persistence.models import (
    CharacterKnowledgeEdgeRecord,
    CharacterRecord,
    EntityLinkRecord,
    MediaAssetRecord,
    MemoryRecord,
    SaveRecord,
    SceneSnapshotRecord,
    SummaryRecord,
    WorldStateRecord,
)
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.character_locks import (
    character_field_is_locked,
    explicit_character_locked_fields,
    merge_character_locked_fields,
    normalize_character_locked_fields,
)
from bragi.services.knowledge_boundary import knowledge_edge_allows_prompt_use

_KNOWS_RELATION = "knows"
_REFERENCE_IMAGE_RELATION = "reference_image"
_REGISTRY_KNOWLEDGE_TARGET_TYPES = frozenset({"memory", "world_state", "summary"})


@dataclass(frozen=True)
class CharacterRegistryReferenceImageRow:
    media_asset_id: str
    mime_type: str
    prompt_preview: str
    provider: str
    model: str
    created_at: str | None = None
    source: str | None = None
    content_rating: str = "unclassified"


@dataclass(frozen=True)
class CharacterRegistryLinkRow:
    target_type: str
    target_id: str
    title: str
    body: str
    linked_character_ids: tuple[str, ...] = ()
    value: object | None = None
    category: str = ""
    confidence: float | None = None
    tags: tuple[str, ...] = ()
    importance: float | None = None
    source_message_id: str | None = None

    @property
    def id(self) -> str:
        return f"{self.target_type}:{self.target_id}"


@dataclass(frozen=True)
class CharacterRegistryRow:
    character_id: str
    name: str
    aliases_text: str = ""
    role: str = ""
    age: str = ""
    known_state: str = ""
    history: str = ""
    met: bool = False
    appearance: str = ""
    visual_notes: str = ""
    current_clothing: str = ""
    personality: str = ""
    voice: str = ""
    texting_style: str = ""
    relationships_json: str = "{}"
    goals: str = ""
    motivations: str = ""
    current_intent: str = ""
    boundaries: str = ""
    attitude_toward_player: str = ""
    cooperation_conditions: str = ""
    status: str = ""
    location_id: str | None = None
    private_notes: str = ""
    contact_name: str = ""
    present: bool = False
    linked_memory_ids: tuple[str, ...] = ()
    linked_state_ids: tuple[str, ...] = ()
    linked_summary_ids: tuple[str, ...] = ()
    archived: bool = False
    merge_into_character_id: str | None = None
    source_message_id: str | None = None
    locked_fields: tuple[str, ...] | None = None
    protected_from_maintenance: bool = False
    is_player_character: bool = False
    reference_image: CharacterRegistryReferenceImageRow | None = None
    generated_images: tuple[CharacterRegistryReferenceImageRow, ...] = ()
    content_rating: str = "unclassified"

    @property
    def id(self) -> str:
        return self.character_id


@dataclass(frozen=True)
class CharacterRegistryModel:
    active_save_id: str | None
    save: SaveRecord | None
    characters: tuple[CharacterRegistryRow, ...] = ()
    link_targets: tuple[CharacterRegistryLinkRow, ...] = ()
    location_choices: tuple[tuple[str, str], ...] = ()
    error: str | None = None

    @property
    def save_id(self) -> str | None:
        return self.active_save_id


@dataclass(frozen=True)
class CharacterRegistryEdits:
    characters: tuple[CharacterRegistryRow, ...] = ()


@dataclass(frozen=True)
class CharacterRegistryApplyResult:
    model: CharacterRegistryModel
    created_count: int = 0
    updated_count: int = 0
    archived_count: int = 0
    created_character_ids: tuple[str, ...] = ()

    @property
    def error(self) -> str | None:
        return self.model.error


@dataclass(frozen=True)
class CharacterFieldEnhanceResult:
    model: CharacterRegistryModel
    character_id: str
    field_name: str
    created_count: int = 0
    updated_count: int = 0
    archived_count: int = 0
    field_changed: bool = True
    notice: str | None = None

    @property
    def error(self) -> str | None:
        return self.model.error


@dataclass(frozen=True)
class CharacterKnowledgeAction:
    action: str
    target_type: str = ""
    target_id: str = ""
    memory_id: str = ""
    state_id: str = ""
    body: str = ""
    tags: tuple[str, ...] = ()
    importance: float = 1.0
    key: str = ""
    category: str = ""
    confidence: float = 1.0
    value: dict[str, object] | None = None
    archived: bool = False


@dataclass(frozen=True)
class CharacterKnowledgeApplyResult:
    model: CharacterRegistryModel
    created_count: int = 0
    updated_count: int = 0
    archived_count: int = 0

    @property
    def error(self) -> str | None:
        return self.model.error


class CharacterRegistryService:
    def __init__(
        self,
        repositories: PersistenceRepositories,
        *,
        active_save_id: str | None = None,
        allowed_content_rating: str | None = None,
    ) -> None:
        self.repositories = repositories
        self.active_save_id = active_save_id
        self.allowed_content_rating = allowed_content_rating

    def build_model(
        self,
        active_save_id: str | None | object = ...,
    ) -> CharacterRegistryModel:
        requested_save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        active_save = _active_save(self.repositories, requested_save_id)
        if active_save is None:
            return CharacterRegistryModel(
                active_save_id=None,
                save=None,
                error="No save loaded",
            )
        snapshot = self.repositories.get_scene_snapshot(active_save.id)
        present_ids = set(snapshot.present_character_ids if snapshot else [])
        character_records = tuple(self.repositories.list_characters(active_save.id))
        location_records = tuple(self.repositories.list_locations(active_save.id))
        memory_records = tuple(self.repositories.list_memories(active_save.id))
        state_records = tuple(self.repositories.list_world_state(active_save.id))
        summary_records = tuple(self.repositories.list_summaries(active_save.id))
        media_assets = {
            asset.id: asset
            for asset in self.repositories.list_media_assets(active_save.id)
            if asset.type == "image" and asset.status == "succeeded"
        }
        entity_links = tuple(self.repositories.list_entity_links(active_save.id))
        knowledge_edges = tuple(
            self.repositories.list_character_knowledge_edges(active_save.id)
        )
        linked_targets = _link_targets(
            memory_records=memory_records,
            state_records=state_records,
            summary_records=summary_records,
        )
        allowed_location_ids: set[str] | None = None
        if self.allowed_content_rating is not None:
            from bragi.services.world_data_service import WorldDataService

            rated_world = WorldDataService(
                self.repositories,
                active_save_id=active_save.id,
                allowed_content_rating=self.allowed_content_rating,
            ).build_model(active_save_id=active_save.id)
            allowed_targets = {
                *(("memory", row.memory_id) for row in rated_world.memories),
                *(("world_state", row.state_id) for row in rated_world.state_rows),
                *(("summary", row.summary_id) for row in rated_world.summaries),
            }
            allowed_location_ids = {
                row.location_id for row in rated_world.locations
            }
            linked_targets = [
                target
                for target in linked_targets
                if (target.target_type, target.target_id) in allowed_targets
            ]
        valid_link_targets = {
            (target.target_type, target.target_id) for target in linked_targets
        }
        links_by_character = _character_knows_links(
            entity_links,
            knowledge_edges=knowledge_edges,
            valid_targets=valid_link_targets,
        )
        reference_images_by_character = _character_reference_images(
            character_records=character_records,
            media_assets=media_assets,
            entity_links=entity_links,
        )
        generated_images_by_character = _character_generated_images(
            character_records=character_records,
            media_assets=media_assets,
            reference_images_by_character=reference_images_by_character,
        )
        characters = tuple(
            _character_row(
                _restricted_character_record(record)
                if self.allowed_content_rating is not None
                and content_rating_exceeds(
                    minimum_rating=record.content_rating,
                    allowed_rating=self.allowed_content_rating,
                )
                else record,
                present=record.id in present_ids or record.is_player_character,
                links=links_by_character.get(record.id, frozenset()),
                reference_image=reference_images_by_character.get(record.id),
                generated_images=generated_images_by_character.get(record.id, ()),
            )
            for record in character_records
        )
        return CharacterRegistryModel(
            active_save_id=active_save.id,
            save=active_save,
            characters=characters,
            link_targets=tuple(
                replace(
                    target,
                    linked_character_ids=tuple(
                        character.id
                        for character in character_records
                        if (target.target_type, target.target_id)
                        in links_by_character.get(character.id, frozenset())
                    ),
                )
                for target in linked_targets
            ),
            location_choices=tuple(
                (location.id, location.name)
                for location in location_records
                if allowed_location_ids is None
                or location.id in allowed_location_ids
            ),
        )

    def apply_edits(
        self,
        edits: CharacterRegistryEdits,
        active_save_id: str | None | object = ...,
    ) -> CharacterRegistryApplyResult:
        model = self.build_model(active_save_id=active_save_id)
        if model.save_id is None:
            raise ValueError("No save loaded")
        save_id = model.save_id
        current = {
            character.id: character
            for character in self.repositories.list_characters(save_id)
        }
        valid_location_ids = {
            location.id for location in self.repositories.list_locations(save_id)
        }
        valid_targets = {
            (target.target_type, target.target_id) for target in model.link_targets
        }
        model_rows = {row.character_id: row for row in model.characters}
        created_count = 0
        updated_count = 0
        archived_count = 0
        merged_count = 0
        created_character_ids: list[str] = []
        archived_ids: set[str] = set()
        merged_ids: set[str] = set()
        self.repositories.begin_immediate_transaction()
        try:
            saved_rows: list[CharacterRegistryRow] = []
            merge_rows: list[CharacterRegistryRow] = []
            for row in edits.characters:
                if row.location_id and row.location_id not in valid_location_ids:
                    raise ValueError(
                        "Character location does not belong to the active save"
                    )
                requested_targets = _requested_targets(row)
                unknown_targets = requested_targets - valid_targets
                if unknown_targets:
                    raise ValueError(
                        "Character linked target does not belong to the active save"
                    )
                if row.character_id:
                    record = current.get(row.character_id)
                    if record is None:
                        raise ValueError(
                            "Character edit does not belong to the active save"
                        )
                    if row.merge_into_character_id is not None:
                        merge_rows.append(row)
                        continue
                    if row.archived:
                        self.repositories.archive_character(record.id)
                        self.repositories.delete_entity_links_for_endpoint(
                            save_id=save_id,
                            entity_type="character",
                            entity_id=record.id,
                        )
                        _remove_thread_character_references(
                            self.repositories,
                            save_id=save_id,
                            character_id=record.id,
                        )
                        archived_ids.add(record.id)
                        archived_count += 1
                        continue
                    if not row.name.strip():
                        raise ValueError("Character name must not be blank")
                    saved = self.repositories.update_character(
                        _record_from_row(row, record=record)
                    )
                    updated_count += 1
                else:
                    if row.archived or not row.name.strip():
                        continue
                    saved = self.repositories.add_character(
                        save_id=save_id,
                        name=row.name.strip(),
                        aliases=_csv(row.aliases_text),
                        role=row.role.strip(),
                        age=row.age.strip(),
                        known_state=_row_history(row),
                        history=_row_history(row),
                        met=row.met,
                        appearance=row.appearance.strip(),
                        visual_notes=row.visual_notes.strip(),
                        current_clothing=row.current_clothing.strip(),
                        personality=row.personality.strip(),
                        voice=row.voice.strip(),
                        texting_style=row.texting_style.strip(),
                        relationships=_json_object(row.relationships_json),
                        goals=row.goals.strip(),
                        motivations=row.motivations.strip(),
                        current_intent=row.current_intent.strip(),
                        boundaries=row.boundaries.strip(),
                        attitude_toward_player=row.attitude_toward_player.strip(),
                        cooperation_conditions=row.cooperation_conditions.strip(),
                        status=row.status.strip(),
                        location_id=_none_if_blank(row.location_id),
                        private_notes=row.private_notes.strip(),
                        contact_name=row.contact_name.strip(),
                        locked_fields=normalize_character_locked_fields(
                            row.locked_fields or (),
                            preserve_unknown=False,
                        ),
                        protected_from_maintenance=row.protected_from_maintenance,
                        is_player_character=row.is_player_character,
                        content_rating=row.content_rating,
                    )
                    created_count += 1
                    created_character_ids.append(saved.id)
                current_row = model_rows.get(saved.id)
                current_targets = (
                    _requested_targets(current_row)
                    if current_row is not None
                    else set()
                )
                if requested_targets != current_targets:
                    _replace_character_links(
                        repositories=self.repositories,
                        save_id=save_id,
                        character_id=saved.id,
                        targets=requested_targets,
                    )
                saved_rows.append(replace(row, character_id=saved.id))
            merge_targets = {
                row.character_id: _none_if_blank(row.merge_into_character_id)
                for row in merge_rows
            }
            merge_source_ids = set(merge_targets)
            if any(
                target_id != source_id and target_id in merge_source_ids
                for source_id, target_id in merge_targets.items()
            ):
                raise ValueError("Chained character merges are not supported")
            for row in merge_rows:
                target_id = merge_targets[row.character_id]
                if row.archived:
                    raise ValueError("Character cannot be both deleted and merged")
                if target_id is None:
                    raise ValueError("Merge target character is required")
                if row.character_id == target_id:
                    raise ValueError("Character cannot be merged into itself")
                if row.character_id in archived_ids or row.character_id in merged_ids:
                    raise ValueError("Character merge source is no longer active")
                if target_id in archived_ids or target_id in merged_ids:
                    raise ValueError("Character merge target is no longer active")
                source = self.repositories.get_character(row.character_id)
                target = self.repositories.get_character(target_id)
                if source is None or source.save_id != save_id:
                    raise ValueError(
                        "Character merge source does not belong to the active save"
                    )
                if target is None or target.save_id != save_id:
                    raise ValueError(
                        "Character merge target does not belong to the active save"
                    )
                merged = self.repositories.update_character(
                    _merged_character(target=target, source=source)
                )
                _merge_character_links(
                    repositories=self.repositories,
                    save_id=save_id,
                    source_id=source.id,
                    target_id=merged.id,
                )
                _replace_thread_character_references(
                    self.repositories,
                    save_id=save_id,
                    source_id=source.id,
                    target_id=merged.id,
                )
                self.repositories.archive_character(source.id)
                self.repositories.delete_entity_links_for_endpoint(
                    save_id=save_id,
                    entity_type="character",
                    entity_id=source.id,
                )
                merged_ids.add(source.id)
                archived_ids.add(source.id)
                merged_count += 1
            _persist_presence(
                repositories=self.repositories,
                save_id=save_id,
                rows=saved_rows,
                archived_ids=archived_ids,
                merged_ids=merged_ids,
                merge_targets=merge_targets,
            )
            self.repositories.commit_transaction()
        except Exception:
            self.repositories.rollback_transaction()
            raise
        return CharacterRegistryApplyResult(
            model=self.build_model(active_save_id=save_id),
            created_count=created_count,
            updated_count=updated_count,
            archived_count=archived_count + merged_count,
            created_character_ids=tuple(created_character_ids),
        )

    def apply_knowledge_actions(
        self,
        *,
        character_id: str,
        actions: tuple[CharacterKnowledgeAction, ...],
        active_save_id: str | None | object = ...,
    ) -> CharacterKnowledgeApplyResult:
        model = self.build_model(active_save_id=active_save_id)
        if model.save_id is None:
            raise ValueError("No save loaded")
        save_id = model.save_id
        character_ids = {
            character.character_id for character in model.characters
        }
        if character_id not in character_ids:
            raise ValueError("Character edit does not belong to the active save")

        memories = {
            memory.id: memory
            for memory in self.repositories.list_memories(save_id)
        }
        states = {
            state.id: state for state in self.repositories.list_world_state(save_id)
        }
        valid_targets = {
            (target.target_type, target.target_id) for target in model.link_targets
        }
        created_count = 0
        updated_count = 0
        archived_count = 0

        self.repositories.begin_immediate_transaction()
        try:
            for action in actions:
                kind = action.action.strip()
                if kind == "link":
                    target_type, target_id = _knowledge_target(action)
                    _validate_knowledge_target(
                        target_type=target_type,
                        target_id=target_id,
                        valid_targets=valid_targets,
                    )
                    _link_character_knowledge(
                        self.repositories,
                        save_id=save_id,
                        character_id=character_id,
                        target_type=target_type,
                        target_id=target_id,
                    )
                    continue
                if kind == "unlink":
                    target_type, target_id = _knowledge_target(action)
                    _validate_knowledge_target(
                        target_type=target_type,
                        target_id=target_id,
                        valid_targets=valid_targets,
                    )
                    _unlink_character_knowledge(
                        self.repositories,
                        save_id=save_id,
                        character_id=character_id,
                        target_type=target_type,
                        target_id=target_id,
                    )
                    continue
                if kind == "create_memory":
                    body = action.body.strip()
                    if not body:
                        raise ValueError("Memory body is required")
                    memory = self.repositories.add_memory(
                        save_id=save_id,
                        body=body,
                        tags=_clean_tags(action.tags),
                        importance=_knowledge_importance(action.importance),
                    )
                    memories[memory.id] = memory
                    valid_targets.add(("memory", memory.id))
                    _link_character_knowledge(
                        self.repositories,
                        save_id=save_id,
                        character_id=character_id,
                        target_type="memory",
                        target_id=memory.id,
                    )
                    created_count += 1
                    continue
                if kind == "update_memory":
                    memory = _knowledge_memory(action, memories)
                    if action.archived:
                        self.repositories.delete_entity_links_for_endpoint(
                            save_id=save_id,
                            entity_type="memory",
                            entity_id=memory.id,
                        )
                        _archive_character_knowledge_edges_for_target(
                            repositories=self.repositories,
                            save_id=save_id,
                            target_type="memory",
                            target_id=memory.id,
                        )
                        self.repositories.archive_memory(memory.id)
                        memories.pop(memory.id, None)
                        valid_targets.discard(("memory", memory.id))
                        archived_count += 1
                        continue
                    body = action.body.strip()
                    if not body:
                        raise ValueError("Memory body is required")
                    updated_memory = self.repositories.update_memory(
                        memory_id=memory.id,
                        body=body,
                        tags=_clean_tags(action.tags),
                        importance=_knowledge_importance(action.importance),
                        clear_source=True,
                    )
                    memories[updated_memory.id] = updated_memory
                    _link_character_knowledge(
                        self.repositories,
                        save_id=save_id,
                        character_id=character_id,
                        target_type="memory",
                        target_id=updated_memory.id,
                    )
                    updated_count += 1
                    continue
                if kind == "create_world_state":
                    key = action.key.strip()
                    if not key:
                        raise ValueError("World-state key is required")
                    if action.value is None:
                        raise ValueError("World-state value is required")
                    _reject_duplicate_world_state_key(states, key)
                    state = self.repositories.upsert_world_state(
                        save_id=save_id,
                        key=key,
                        value=dict(action.value),
                        category=action.category.strip(),
                        confidence=_knowledge_confidence(action.confidence),
                    )
                    self.repositories.add_state_change(
                        save_id=save_id,
                        operation="manual_character_knowledge_edit",
                        state_key=state.key,
                        before_json=None,
                        after_json=_dump_json(state.value),
                    )
                    states[state.id] = state
                    valid_targets.add(("world_state", state.id))
                    _link_character_knowledge(
                        self.repositories,
                        save_id=save_id,
                        character_id=character_id,
                        target_type="world_state",
                        target_id=state.id,
                    )
                    created_count += 1
                    continue
                if kind == "update_world_state":
                    state = _knowledge_state(action, states)
                    if action.archived:
                        self.repositories.delete_entity_links_for_endpoint(
                            save_id=save_id,
                            entity_type="world_state",
                            entity_id=state.id,
                        )
                        _archive_character_knowledge_edges_for_target(
                            repositories=self.repositories,
                            save_id=save_id,
                            target_type="world_state",
                            target_id=state.id,
                        )
                        self.repositories.archive_world_state(
                            save_id=save_id,
                            key=state.key,
                        )
                        self.repositories.add_state_change(
                            save_id=save_id,
                            operation="manual_character_knowledge_edit",
                            state_key=state.key,
                            before_json=_dump_json(state.value),
                            after_json=None,
                        )
                        states.pop(state.id, None)
                        valid_targets.discard(("world_state", state.id))
                        archived_count += 1
                        continue
                    key = action.key.strip()
                    if not key:
                        raise ValueError("World-state key is required")
                    if action.value is None:
                        raise ValueError("World-state value is required")
                    if key != state.key:
                        _reject_duplicate_world_state_key(states, key)
                        self.repositories.delete_entity_links_for_endpoint(
                            save_id=save_id,
                            entity_type="world_state",
                            entity_id=state.id,
                        )
                        _archive_character_knowledge_edges_for_target(
                            repositories=self.repositories,
                            save_id=save_id,
                            target_type="world_state",
                            target_id=state.id,
                        )
                        self.repositories.archive_world_state(
                            save_id=save_id,
                            key=state.key,
                        )
                        valid_targets.discard(("world_state", state.id))
                    updated_state = self.repositories.upsert_world_state(
                        save_id=save_id,
                        key=key,
                        value=dict(action.value),
                        category=action.category.strip(),
                        confidence=_knowledge_confidence(action.confidence),
                    )
                    self.repositories.add_state_change(
                        save_id=save_id,
                        operation="manual_character_knowledge_edit",
                        state_key=updated_state.key,
                        before_json=_dump_json(state.value),
                        after_json=_dump_json(updated_state.value),
                    )
                    states.pop(state.id, None)
                    states[updated_state.id] = updated_state
                    valid_targets.add(("world_state", updated_state.id))
                    _link_character_knowledge(
                        self.repositories,
                        save_id=save_id,
                        character_id=character_id,
                        target_type="world_state",
                        target_id=updated_state.id,
                    )
                    updated_count += 1
                    continue
                raise ValueError(f"Unsupported character knowledge action: {kind}")
            self.repositories.commit_transaction()
        except Exception:
            self.repositories.rollback_transaction()
            raise

        return CharacterKnowledgeApplyResult(
            model=self.build_model(active_save_id=save_id),
            created_count=created_count,
            updated_count=updated_count,
            archived_count=archived_count,
        )


def _active_save(
    repositories: PersistenceRepositories,
    active_save_id: str | None,
) -> SaveRecord | None:
    if active_save_id is not None:
        return repositories.get_save(active_save_id)
    saves = repositories.list_saves()
    return saves[0] if saves else None


def _character_row(
    record: CharacterRecord,
    *,
    present: bool,
    links: frozenset[tuple[str, str]],
    reference_image: CharacterRegistryReferenceImageRow | None,
    generated_images: tuple[CharacterRegistryReferenceImageRow, ...],
) -> CharacterRegistryRow:
    return CharacterRegistryRow(
        character_id=record.id,
        name=record.name,
        aliases_text=", ".join(record.aliases),
        role=record.role,
        age=record.age,
        known_state=record.known_state,
        history=record.history,
        met=record.met,
        appearance=record.appearance,
        visual_notes=record.visual_notes,
        current_clothing=record.current_clothing,
        personality=record.personality,
        voice=record.voice,
        texting_style=record.texting_style,
        relationships_json=_dump_json(record.relationships),
        goals=record.goals,
        motivations=record.motivations,
        current_intent=record.current_intent,
        boundaries=record.boundaries,
        attitude_toward_player=record.attitude_toward_player,
        cooperation_conditions=record.cooperation_conditions,
        status=record.status,
        location_id=record.location_id,
        private_notes=record.private_notes,
        contact_name=record.contact_name,
        present=present,
        linked_memory_ids=tuple(
            sorted(
                target_id
                for target_type, target_id in links
                if target_type == "memory"
            )
        ),
        linked_state_ids=tuple(
            sorted(
                target_id
                for target_type, target_id in links
                if target_type == "world_state"
            )
        ),
        linked_summary_ids=tuple(
            sorted(
                target_id
                for target_type, target_id in links
                if target_type == "summary"
            )
        ),
        source_message_id=record.source_message_id,
        locked_fields=tuple(normalize_character_locked_fields(record.locked_fields)),
        protected_from_maintenance=record.protected_from_maintenance,
        is_player_character=record.is_player_character,
        reference_image=reference_image,
        generated_images=generated_images,
        content_rating=record.content_rating,
    )


def _character_reference_images(
    *,
    character_records: tuple[CharacterRecord, ...],
    media_assets: dict[str, MediaAssetRecord],
    entity_links: tuple[EntityLinkRecord, ...],
) -> dict[str, CharacterRegistryReferenceImageRow]:
    rows: dict[str, CharacterRegistryReferenceImageRow] = {}
    character_ids = {character.id for character in character_records}
    for link in entity_links:
        if (
            link.entity_type == "character"
            and link.entity_id in character_ids
            and link.target_type == "media_asset"
            and link.relation == _REFERENCE_IMAGE_RELATION
            and link.entity_id not in rows
        ):
            asset = media_assets.get(link.target_id)
            if asset is not None:
                rows[link.entity_id] = _reference_image_row(asset)

    return rows


def _reference_image_row(
    asset: MediaAssetRecord,
) -> CharacterRegistryReferenceImageRow:
    metadata = _metadata(asset)
    source = metadata.get("source")
    return CharacterRegistryReferenceImageRow(
        media_asset_id=asset.id,
        mime_type=asset.mime_type,
        prompt_preview=_prompt_preview(asset.prompt),
        provider=asset.provider,
        model=asset.model,
        created_at=asset.created_at,
        source=source if isinstance(source, str) else None,
        content_rating=str(metadata.get("content_rating", "unclassified")),
    )


def _character_generated_images(
    *,
    character_records: tuple[CharacterRecord, ...],
    media_assets: dict[str, MediaAssetRecord],
    reference_images_by_character: dict[str, CharacterRegistryReferenceImageRow],
) -> dict[str, tuple[CharacterRegistryReferenceImageRow, ...]]:
    character_ids = {character.id for character in character_records}
    reference_asset_character_ids = {
        reference.media_asset_id: character_id
        for character_id, reference in reference_images_by_character.items()
    }
    rows: dict[str, list[MediaAssetRecord]] = {
        character_id: [] for character_id in character_ids
    }
    seen: set[tuple[str, str]] = set()
    for asset in media_assets.values():
        metadata = _metadata(asset)
        kind = metadata.get("kind")
        if asset.id in reference_asset_character_ids:
            continue
        if kind not in {"character_image", "character_reference"}:
            continue
        linked_character_ids: list[str] = []
        metadata_character_id = metadata.get("character_id")
        if (
            isinstance(metadata_character_id, str)
            and metadata_character_id in character_ids
        ):
            linked_character_ids.append(metadata_character_id)
        if kind == "character_image":
            source_character_id = reference_asset_character_ids.get(
                asset.source_media_asset_id or ""
            )
            if source_character_id is not None:
                linked_character_ids.append(source_character_id)
        for character_id in dict.fromkeys(linked_character_ids):
            key = (character_id, asset.id)
            if key in seen:
                continue
            rows[character_id].append(asset)
            seen.add(key)
    return {
        character_id: tuple(
            _reference_image_row(asset)
            for asset in sorted(
                assets,
                key=lambda item: item.created_at or "",
                reverse=True,
            )
        )
        for character_id, assets in rows.items()
        if assets
    }


def _metadata(asset: MediaAssetRecord) -> dict[str, object]:
    try:
        value = json.loads(asset.metadata_json)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _prompt_preview(prompt: str) -> str:
    text = " ".join(prompt.split())
    return text if len(text) <= 160 else f"{text[:157].rstrip()}..."


def _link_targets(
    *,
    memory_records: tuple[MemoryRecord, ...],
    state_records: tuple[WorldStateRecord, ...],
    summary_records: tuple[SummaryRecord, ...],
) -> list[CharacterRegistryLinkRow]:
    targets: list[CharacterRegistryLinkRow] = []
    targets.extend(
        CharacterRegistryLinkRow(
            target_type="memory",
            target_id=memory.id,
            title="Memory",
            body=memory.body,
            tags=tuple(memory.tags),
            importance=memory.importance,
            source_message_id=memory.source_message_id,
        )
        for memory in memory_records
    )
    targets.extend(
        CharacterRegistryLinkRow(
            target_type="world_state",
            target_id=state.id,
            title=state.key,
            body=_dump_json(state.value),
            value=state.value,
            category=state.category,
            confidence=state.confidence,
            source_message_id=state.source_message_id,
        )
        for state in state_records
    )
    targets.extend(
        CharacterRegistryLinkRow(
            target_type="summary",
            target_id=summary.id,
            title="Summary",
            body=summary.body,
        )
        for summary in summary_records
    )
    return targets


def _character_knows_links(
    entity_links: tuple[EntityLinkRecord, ...],
    *,
    knowledge_edges: tuple[CharacterKnowledgeEdgeRecord, ...] = (),
    valid_targets: set[tuple[str, str]] | None = None,
) -> dict[str, frozenset[tuple[str, str]]]:
    links: dict[str, set[tuple[str, str]]] = {}
    blocked_by_knowledge_edge: set[tuple[str, str, str]] = set()
    for edge in knowledge_edges:
        target_type = _normalized_target_type(edge.target_type)
        if target_type not in _REGISTRY_KNOWLEDGE_TARGET_TYPES:
            continue
        target = (target_type, edge.target_id)
        if valid_targets is not None and target not in valid_targets:
            continue
        key = (edge.character_id, target_type, edge.target_id)
        if not knowledge_edge_allows_prompt_use(edge):
            blocked_by_knowledge_edge.add(key)
            continue
        links.setdefault(edge.character_id, set()).add(target)
    for link in entity_links:
        if link.entity_type != "character" or link.relation != _KNOWS_RELATION:
            continue
        target_type = _normalized_target_type(link.target_type)
        if target_type not in _REGISTRY_KNOWLEDGE_TARGET_TYPES:
            continue
        key = (link.entity_id, target_type, link.target_id)
        if key in blocked_by_knowledge_edge:
            continue
        links.setdefault(link.entity_id, set()).add((target_type, link.target_id))
    return {key: frozenset(value) for key, value in links.items()}


def _requested_targets(row: CharacterRegistryRow) -> set[tuple[str, str]]:
    return {
        *{("memory", target_id) for target_id in row.linked_memory_ids},
        *{("world_state", target_id) for target_id in row.linked_state_ids},
        *{("summary", target_id) for target_id in row.linked_summary_ids},
    }


def _replace_character_links(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    character_id: str,
    targets: set[tuple[str, str]],
) -> None:
    for link in repositories.list_entity_links(save_id):
        if (
            link.entity_type == "character"
            and link.entity_id == character_id
            and link.relation == _KNOWS_RELATION
            and _normalized_target_type(link.target_type)
            in _REGISTRY_KNOWLEDGE_TARGET_TYPES
        ):
            repositories.delete_entity_link(link.id)
    _archive_character_knowledge_edges_for_character(
        repositories=repositories,
        save_id=save_id,
        character_id=character_id,
        target_types=_REGISTRY_KNOWLEDGE_TARGET_TYPES,
    )
    for target_type, target_id in sorted(targets):
        repositories.add_entity_link(
            save_id=save_id,
            entity_type="character",
            entity_id=character_id,
            target_type=target_type,
            target_id=target_id,
            relation=_KNOWS_RELATION,
        )
        repositories.add_character_knowledge_edge(
            save_id=save_id,
            character_id=character_id,
            target_type=target_type,
            target_id=target_id,
            knowledge_state="knows",
            acquisition_method="manual",
            confidence=1.0,
        )


def _archive_character_knowledge_edges_for_character(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    character_id: str,
    target_types: frozenset[str],
    target_id: str | None = None,
) -> None:
    for edge in repositories.list_character_knowledge_edges(
        save_id,
        character_ids={character_id},
    ):
        if not knowledge_edge_allows_prompt_use(edge):
            continue
        target_type = _normalized_target_type(edge.target_type)
        if target_type not in target_types:
            continue
        if target_id is not None and edge.target_id != target_id:
            continue
        repositories.archive_character_knowledge_edge(edge.id)


def _archive_character_knowledge_edges_for_target(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    target_type: str,
    target_id: str,
) -> None:
    normalized_target_type = _normalized_target_type(target_type)
    for edge in repositories.list_character_knowledge_edges(save_id):
        if (
            _normalized_target_type(edge.target_type) == normalized_target_type
            and edge.target_id == target_id
        ):
            repositories.archive_character_knowledge_edge(edge.id)


def _merge_character_links(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    source_id: str,
    target_id: str,
) -> None:
    for link in repositories.list_entity_links(save_id):
        if link.entity_type == "character" and link.entity_id == source_id:
            target_type = _normalized_target_type(link.target_type)
            if target_type == "character" and link.target_id == target_id:
                continue
            repositories.add_entity_link(
                save_id=save_id,
                entity_type="character",
                entity_id=target_id,
                target_type=target_type,
                target_id=link.target_id,
                relation=link.relation,
            )
        if link.target_type == "character" and link.target_id == source_id:
            if link.entity_type == "character" and link.entity_id == target_id:
                continue
            repositories.add_entity_link(
                save_id=save_id,
                entity_type=link.entity_type,
                entity_id=link.entity_id,
                target_type="character",
                target_id=target_id,
                relation=link.relation,
            )


def _remove_thread_character_references(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    character_id: str,
) -> None:
    removed = {character_id, f"character:{character_id}"}
    for thread in repositories.list_active_threads(save_id):
        related_entities = [
            item for item in thread.related_entities if item not in removed
        ]
        if related_entities != thread.related_entities:
            repositories.update_active_thread(
                replace(thread, related_entities=related_entities)
            )


def _replace_thread_character_references(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    source_id: str,
    target_id: str,
) -> None:
    source_ref = f"character:{source_id}"
    target_ref = f"character:{target_id}"
    for thread in repositories.list_active_threads(save_id):
        related_entities = _dedupe(
            _replace_character_reference(
                item,
                source_id=source_id,
                target_id=target_id,
                source_ref=source_ref,
                target_ref=target_ref,
            )
            for item in thread.related_entities
        )
        if related_entities != thread.related_entities:
            repositories.update_active_thread(
                replace(thread, related_entities=related_entities)
            )


def _replace_character_reference(
    value: str,
    *,
    source_id: str,
    target_id: str,
    source_ref: str,
    target_ref: str,
) -> str:
    if value == source_id:
        return target_id
    if value == source_ref:
        return target_ref
    return value


def _persist_presence(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    rows: list[CharacterRegistryRow],
    archived_ids: set[str],
    merged_ids: set[str] | None = None,
    merge_targets: dict[str, str | None] | None = None,
) -> SceneSnapshotRecord | None:
    snapshot = repositories.get_scene_snapshot(save_id)
    if snapshot is None and not archived_ids and not any(
        row.present or row.is_player_character for row in rows
    ):
        return None
    current_ids = set(snapshot.present_character_ids if snapshot else [])
    merged_present_ids = {
        target_id
        for source_id, target_id in (merge_targets or {}).items()
        if target_id and source_id in current_ids
    }
    edited_ids = {row.character_id for row in rows if row.character_id}
    current_ids.difference_update(
        edited_ids | archived_ids | set(merged_ids or set())
    )
    current_ids.update(
        row.character_id
        for row in rows
        if row.character_id and (row.present or row.is_player_character)
    )
    current_ids.update(merged_present_ids)
    return repositories.upsert_scene_snapshot(
        save_id=save_id,
        current_location_id=snapshot.current_location_id if snapshot else None,
        situation=snapshot.situation if snapshot else "",
        objective=snapshot.objective if snapshot else "",
        in_world_time=snapshot.in_world_time if snapshot else "",
        time_of_day=snapshot.time_of_day if snapshot else "",
        day_of_week=snapshot.day_of_week if snapshot else "",
        world_day_index=snapshot.world_day_index if snapshot else None,
        weather=snapshot.weather if snapshot else "",
        mood=snapshot.mood if snapshot else "",
        nearby_objects=snapshot.nearby_objects if snapshot else [],
        hazards=snapshot.hazards if snapshot else [],
        present_character_ids=sorted(current_ids),
        source_message_id=snapshot.source_message_id if snapshot else None,
        locked_fields=snapshot.locked_fields if snapshot else [],
        snapshot_id=snapshot.id if snapshot else None,
    )


def _row_history(row: CharacterRegistryRow) -> str:
    return (row.known_state or row.history).strip()


def _record_from_row(
    row: CharacterRegistryRow,
    *,
    record: CharacterRecord,
) -> CharacterRecord:
    changed = _changed_fields(
        _character_row(
            record,
            present=row.present,
            links=frozenset(),
            reference_image=None,
            generated_images=(),
        ),
        row,
    )
    return replace(
        record,
        name=row.name.strip(),
        aliases=_csv(row.aliases_text),
        role=row.role.strip(),
        age=row.age.strip(),
        known_state=_row_history(row),
        history=_row_history(row),
        met=row.met,
        appearance=row.appearance.strip(),
        visual_notes=row.visual_notes.strip(),
        current_clothing=row.current_clothing.strip(),
        personality=row.personality.strip(),
        voice=row.voice.strip(),
        texting_style=row.texting_style.strip(),
        relationships=_json_object(row.relationships_json),
        goals=row.goals.strip(),
        motivations=row.motivations.strip(),
        current_intent=row.current_intent.strip(),
        boundaries=row.boundaries.strip(),
        attitude_toward_player=row.attitude_toward_player.strip(),
        cooperation_conditions=row.cooperation_conditions.strip(),
        status=row.status.strip(),
        location_id=_none_if_blank(row.location_id),
        private_notes=row.private_notes.strip(),
        contact_name=row.contact_name.strip(),
        protected_from_maintenance=row.protected_from_maintenance,
        is_player_character=row.is_player_character,
        content_rating=row.content_rating,
        locked_fields=(
            merge_character_locked_fields(record.locked_fields, changed)
            if row.locked_fields is None
            else explicit_character_locked_fields(
                record.locked_fields,
                row.locked_fields,
            )
        ),
    )


def _restricted_character_record(record: CharacterRecord) -> CharacterRecord:
    from bragi.services.sexual_content_safety import CONTENT_FILTER_TRANSITION

    replacement = CONTENT_FILTER_TRANSITION
    return replace(
        record,
        name=replacement,
        aliases=[],
        role=replacement if record.role else "",
        age=replacement if record.age else "",
        known_state=replacement if record.known_state else "",
        history=replacement if record.history else "",
        appearance=replacement if record.appearance else "",
        visual_notes=replacement if record.visual_notes else "",
        current_clothing=replacement if record.current_clothing else "",
        personality=replacement if record.personality else "",
        voice=replacement if record.voice else "",
        texting_style=replacement if record.texting_style else "",
        relationships={},
        goals=replacement if record.goals else "",
        motivations=replacement if record.motivations else "",
        current_intent=replacement if record.current_intent else "",
        boundaries=replacement if record.boundaries else "",
        attitude_toward_player=(
            replacement if record.attitude_toward_player else ""
        ),
        cooperation_conditions=(
            replacement if record.cooperation_conditions else ""
        ),
        status=replacement if record.status else "",
        private_notes=replacement if record.private_notes else "",
        contact_name=replacement if record.contact_name else "",
    )


def _merged_character(
    *,
    target: CharacterRecord,
    source: CharacterRecord,
) -> CharacterRecord:
    aliases = (
        target.aliases
        if character_field_is_locked(target.locked_fields, "aliases")
        else _dedupe([*target.aliases, source.name, *source.aliases])
    )
    relationships = (
        target.relationships
        if character_field_is_locked(target.locked_fields, "relationships")
        else _merge_relationships(target.relationships, source.relationships)
    )
    locked_fields = merge_character_locked_fields(
        target.locked_fields,
        source.locked_fields,
    )
    return replace(
        target,
        aliases=[alias for alias in aliases if alias != target.name],
        role=_merge_character_field(target, source, "role"),
        age=_merge_character_age(target, source),
        known_state=_merge_character_field(target, source, "known_state"),
        history=_merge_character_field(target, source, "known_state"),
        met=(
            target.met
            if character_field_is_locked(target.locked_fields, "met")
            else target.met or source.met
        ),
        appearance=_merge_character_field(target, source, "appearance"),
        visual_notes=_merge_character_field(target, source, "visual_notes"),
        current_clothing=_merge_character_field(target, source, "current_clothing"),
        personality=_merge_character_field(target, source, "personality"),
        voice=_merge_character_field(target, source, "voice"),
        texting_style=_merge_character_field(target, source, "texting_style"),
        relationships=relationships,
        goals=_merge_character_field(target, source, "goals"),
        motivations=_merge_character_field(target, source, "motivations"),
        current_intent=_merge_character_field(target, source, "current_intent"),
        boundaries=_merge_character_field(target, source, "boundaries"),
        attitude_toward_player=_merge_character_field(
            target,
            source,
            "attitude_toward_player",
        ),
        cooperation_conditions=_merge_character_field(
            target,
            source,
            "cooperation_conditions",
        ),
        status=_merge_character_field(target, source, "status"),
        location_id=(
            target.location_id
            if character_field_is_locked(target.locked_fields, "location_id")
            else target.location_id or source.location_id
        ),
        private_notes=_merge_character_field(target, source, "private_notes"),
        protected_from_maintenance=(
            target.protected_from_maintenance or source.protected_from_maintenance
        ),
        is_player_character=target.is_player_character or source.is_player_character,
        locked_fields=locked_fields,
        content_rating=maximum_content_rating(
            (target.content_rating, source.content_rating)
        ),
    )


def _merge_character_field(
    target: CharacterRecord,
    source: CharacterRecord,
    field: str,
) -> str:
    target_value = cast(str, getattr(target, field))
    if character_field_is_locked(target.locked_fields, field):
        return target_value
    return _merge_text(target_value, cast(str, getattr(source, field)))


def _merge_character_age(target: CharacterRecord, source: CharacterRecord) -> str:
    if character_field_is_locked(target.locked_fields, "age"):
        return target.age
    return target.age.strip() or source.age.strip()


def _changed_fields(
    current: CharacterRegistryRow,
    row: CharacterRegistryRow,
) -> tuple[str, ...]:
    fields = (
        "name",
        "aliases",
        "role",
        "age",
        "known_state",
        "history",
        "met",
        "appearance",
        "visual_notes",
        "current_clothing",
        "personality",
        "voice",
        "relationships",
        "goals",
        "motivations",
        "current_intent",
        "boundaries",
        "attitude_toward_player",
        "cooperation_conditions",
        "status",
        "location_id",
        "private_notes",
        "contact_name",
        "is_player_character",
    )
    return tuple(
        field
        for field in fields
        if _character_field_value(current, field) != _character_field_value(row, field)
    )


def _character_field_value(row: CharacterRegistryRow, field: str) -> object:
    if field == "aliases":
        return row.aliases_text
    if field == "relationships":
        return row.relationships_json
    return getattr(row, field)


def _normalized_target_type(value: str) -> str:
    if value in {"state", "world_state"}:
        return "world_state"
    return value


def _knowledge_target(action: CharacterKnowledgeAction) -> tuple[str, str]:
    target_type = _normalized_target_type(action.target_type.strip())
    target_id = action.target_id.strip()
    if target_type not in _REGISTRY_KNOWLEDGE_TARGET_TYPES or not target_id:
        raise ValueError("Character linked target does not belong to the active save")
    return target_type, target_id


def _validate_knowledge_target(
    *,
    target_type: str,
    target_id: str,
    valid_targets: set[tuple[str, str]],
) -> None:
    if (target_type, target_id) not in valid_targets:
        raise ValueError("Character linked target does not belong to the active save")


def _link_character_knowledge(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    character_id: str,
    target_type: str,
    target_id: str,
) -> None:
    repositories.add_entity_link(
        save_id=save_id,
        entity_type="character",
        entity_id=character_id,
        target_type=target_type,
        target_id=target_id,
        relation=_KNOWS_RELATION,
    )
    repositories.add_character_knowledge_edge(
        save_id=save_id,
        character_id=character_id,
        target_type=target_type,
        target_id=target_id,
        knowledge_state="knows",
        acquisition_method="manual",
        confidence=1.0,
    )


def _unlink_character_knowledge(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    character_id: str,
    target_type: str,
    target_id: str,
) -> None:
    for link in repositories.list_entity_links(save_id):
        if (
            link.entity_type == "character"
            and link.entity_id == character_id
            and _normalized_target_type(link.target_type) == target_type
            and link.target_id == target_id
            and link.relation == _KNOWS_RELATION
        ):
            repositories.delete_entity_link(link.id)
    _archive_character_knowledge_edges_for_character(
        repositories=repositories,
        save_id=save_id,
        character_id=character_id,
        target_types=frozenset({target_type}),
        target_id=target_id,
    )


def _knowledge_memory(
    action: CharacterKnowledgeAction,
    memories: dict[str, MemoryRecord],
) -> MemoryRecord:
    memory_id = (action.memory_id or action.target_id).strip()
    memory = memories.get(memory_id)
    if memory is None:
        raise ValueError("Character linked target does not belong to the active save")
    return memory


def _knowledge_state(
    action: CharacterKnowledgeAction,
    states: dict[str, WorldStateRecord],
) -> WorldStateRecord:
    state_id = (action.state_id or action.target_id).strip()
    state = states.get(state_id)
    if state is None:
        raise ValueError("Character linked target does not belong to the active save")
    return state


def _clean_tags(tags: tuple[str, ...]) -> list[str]:
    return _dedupe(tags)


def _knowledge_importance(value: float) -> float:
    if value < 0 or value > 1:
        raise ValueError("Memory importance must be between 0 and 1")
    return value


def _knowledge_confidence(value: float) -> float:
    if value < 0 or value > 1:
        raise ValueError("World-state confidence must be between 0 and 1")
    return value


def _reject_duplicate_world_state_key(
    states: dict[str, WorldStateRecord],
    key: str,
) -> None:
    if any(state.key == key for state in states.values()):
        raise ValueError("World-state key already exists")


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        stripped = value.strip()
        if not stripped or stripped in seen:
            continue
        result.append(stripped)
        seen.add(stripped)
    return result


def _merge_text(target: str, source: str) -> str:
    target_text = target.strip()
    source_text = source.strip()
    if not target_text:
        return source_text
    if not source_text or source_text == target_text:
        return target_text
    return f"{target_text}\n\nMerged duplicate note: {source_text}"


def _merge_relationships(
    target: dict[str, object],
    source: dict[str, object],
) -> dict[str, object]:
    relationships = dict(target)
    for key, source_value in source.items():
        if key not in relationships:
            relationships[key] = source_value
            continue
        target_value = relationships[key]
        if target_value == source_value:
            continue
        if isinstance(target_value, str) and isinstance(source_value, str):
            relationships[key] = _merge_text(target_value, source_value)
        else:
            relationships[key] = [target_value, source_value]
    return relationships


def _none_if_blank(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _json_object(value: str) -> dict[str, object]:
    try:
        loaded = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("Character relationships must be valid JSON") from exc
    if not isinstance(loaded, dict):
        raise ValueError("Character relationships must be a JSON object")
    return cast(dict[str, object], loaded)


def _dump_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
