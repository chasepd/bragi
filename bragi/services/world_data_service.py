"""Import-safe active-save world data editor model and persistence service."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any, cast

from bragi.app_logging import log_error_event, log_event
from bragi.content_rating_instructions import (
    CONTENT_RATING_UNCLASSIFIED,
    content_rating_exceeds,
    maximum_content_rating,
)
from bragi.persistence.models import (
    ActiveThreadRecord,
    CharacterKnowledgeEdgeRecord,
    CharacterRecord,
    ContextSourceRecord,
    ContextUpdateAuditRecord,
    ContextUpdateSuggestionRecord,
    EntityLinkRecord,
    LocationRecord,
    MemoryRecord,
    MessageRecord,
    MessageVisibilityRecord,
    SaveRecord,
    ScenarioRecord,
    SceneSnapshotRecord,
    SummaryRecord,
)
from bragi.persistence.repositories import (
    PersistenceRepositories,
    canonical_claim_fingerprint,
)
from bragi.services.character_locks import merge_character_locked_fields
from bragi.services.character_profile_completion import (
    CHARACTER_STARTERS_CONTENT_KEY,
    ScenarioCharacterStarter,
    normalize_scenario_character_starters,
    scenario_character_starter_to_json,
)
from bragi.services.scenario_content_rating import (
    metadata_with_scenario_content_ratings,
    scenario_content_rating,
)
from bragi.services.scenario_service import normalize_scenario_definition
from bragi.services.sexual_content_safety import CONTENT_FILTER_TRANSITION
from bragi.services.state_preservation import preserve_replaced_world_state_memory
from bragi.world_time_model import canonical_world_time_from_legacy

STALE_PENDING_SUGGESTION_DAYS = 30
SCENE_WORLD_TIME_FIELDS = frozenset(
    {
        "in_world_time",
        "time_of_day",
        "day_of_week",
        "world_day_index",
    }
)


@dataclass(frozen=True)
class WorldDataScenarioModel:
    scenario_id: str
    scenario_type: str
    title: str
    premise: str
    player_character_name: str
    player_role: str
    content_sections: tuple[tuple[str, str], ...]
    generation_prompt: str | None = None
    character_starters: tuple[ScenarioCharacterStarter, ...] = ()


@dataclass(frozen=True)
class WorldDataStateRow:
    row_id: str
    key: str
    category: str
    confidence: float
    value_json: str
    source_message_id: str | None
    archived: bool = False
    original_key: str | None = None

    @property
    def id(self) -> str:
        return self.row_id

    @property
    def state_id(self) -> str:
        return self.row_id


@dataclass(frozen=True)
class WorldDataMemoryRow:
    memory_id: str
    body: str
    tags_text: str
    importance: float
    source_message_id: str | None
    archived: bool = False
    source_message_ids: tuple[str, ...] = ()
    consolidated: bool = False

    @property
    def source_count(self) -> int:
        return len(self.source_message_ids)


@dataclass(frozen=True)
class WorldDataContextInputRow:
    context_source_id: str
    source_type: str
    source_id: str
    title: str
    body: str
    fact_type: str
    importance: float
    source_message_count: int
    token_estimate: int | None = None

    @property
    def group_label(self) -> str:
        if self.fact_type:
            return f"{self.source_type} / {self.fact_type}"
        return self.source_type


@dataclass(frozen=True)
class WorldDataSummaryRow:
    summary_id: str
    body: str
    provider: str
    model: str
    covers_message_start_id: str
    covers_message_end_id: str
    archived: bool = False


@dataclass(frozen=True)
class WorldDataSceneRow:
    snapshot_id: str
    current_location_id: str | None
    situation: str
    objective: str
    in_world_time: str
    time_of_day: str
    day_of_week: str
    world_day_index: int | None
    weather: str
    mood: str
    nearby_objects_text: str
    hazards_text: str
    present_character_ids_text: str
    source_message_id: str | None
    locked_fields: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        return self.snapshot_id


@dataclass(frozen=True)
class WorldDataLocationRow:
    location_id: str
    name: str
    aliases_text: str
    description: str
    visual_description: str
    parent_location_id: str | None
    connections_text: str
    status: str
    hazards_text: str
    source_message_id: str | None
    locked_fields: tuple[str, ...] = ()
    archived: bool = False

    @property
    def id(self) -> str:
        return self.location_id


@dataclass(frozen=True)
class WorldDataCharacterRow:
    character_id: str
    name: str
    aliases_text: str
    role: str
    age: str
    known_state: str
    met: bool
    appearance: str
    visual_notes: str
    current_clothing: str
    personality: str
    voice: str
    texting_style: str
    relationships_json: str
    status: str
    location_id: str | None
    private_notes: str
    source_message_id: str | None
    goals: str = ""
    motivations: str = ""
    current_intent: str = ""
    boundaries: str = ""
    attitude_toward_player: str = ""
    cooperation_conditions: str = ""
    locked_fields: tuple[str, ...] = ()
    protected_from_maintenance: bool = False
    is_player_character: bool = False
    content_rating: str = "unclassified"
    history: str = ""

    @property
    def id(self) -> str:
        return self.character_id


@dataclass(frozen=True)
class WorldDataThreadRow:
    thread_id: str
    title: str
    description: str
    status: str
    priority: int
    visibility: str
    related_entities_text: str
    source_message_id: str | None
    locked_fields: tuple[str, ...] = ()
    archived: bool = False

    @property
    def id(self) -> str:
        return self.thread_id


@dataclass(frozen=True)
class WorldDataEntityLinkRow:
    link_id: str
    entity_type: str
    entity_id: str
    target_type: str
    target_id: str
    relation: str
    deleted: bool = False

    @property
    def id(self) -> str:
        return self.link_id


@dataclass(frozen=True)
class WorldDataKnowledgeEdgeRow:
    edge_id: str
    character_id: str
    target_type: str
    target_id: str
    knowledge_state: str
    acquisition_method: str
    confidence: float
    source_message_id: str | None
    source_message_ids_text: str
    evidence_quote: str
    archived: bool = False

    @property
    def id(self) -> str:
        return self.edge_id


@dataclass(frozen=True)
class WorldDataMessageVisibilityRow:
    visibility_id: str
    message_id: str
    character_id: str
    visibility: str
    confidence: float
    source: str
    evidence: str

    @property
    def id(self) -> str:
        return self.visibility_id


@dataclass(frozen=True)
class WorldDataSuggestionRow:
    suggestion_id: str
    update_type: str
    entity_type: str
    entity_id: str | None
    field_path: str
    proposed_value_json: str
    status: str
    reason: str
    confidence: float
    source_message_ids_text: str
    action: str = ""

    @property
    def id(self) -> str:
        return self.suggestion_id


@dataclass(frozen=True)
class WorldDataSuggestionGroupRow:
    group_id: str
    suggestion_ids: tuple[str, ...]
    update_type: str
    entity_type: str
    entity_id: str | None
    field_path: str
    proposed_value_json: str
    status: str
    reason: str
    confidence: float
    source_message_ids_text: str
    suggestion_count: int
    action: str = ""

    @property
    def id(self) -> str:
        return self.group_id


@dataclass(frozen=True)
class WorldDataAuditRow:
    audit_id: str
    suggestion_id: str | None
    operation: str
    entity_type: str
    entity_id: str | None
    field_path: str
    before_json: str
    after_json: str
    reason: str
    confidence: float
    source_message_ids_text: str
    created_at: str | None = None

    @property
    def id(self) -> str:
        return self.audit_id


@dataclass(frozen=True)
class WorldDataLossConditionRow:
    condition_id: str
    name: str
    description: str
    status: str
    source: str
    archived: bool = False

    @property
    def id(self) -> str:
        return self.condition_id


@dataclass(frozen=True)
class WorldDataLossOutcomeRow:
    outcome_id: str
    condition_name: str
    triggering_message_id: str
    explanation: str
    evidence_json: str
    confidence: float
    provider: str
    model: str
    epilogue_message_id: str | None
    epilogue_error: str | None

    @property
    def id(self) -> str:
        return self.outcome_id


@dataclass(frozen=True)
class WorldDataModel:
    active_save_id: str | None
    save: SaveRecord | None
    scenario: WorldDataScenarioModel | None
    world_state: tuple[WorldDataStateRow, ...] = ()
    memories: tuple[WorldDataMemoryRow, ...] = ()
    context_inputs: tuple[WorldDataContextInputRow, ...] = ()
    summaries: tuple[WorldDataSummaryRow, ...] = ()
    scene: WorldDataSceneRow | None = None
    locations: tuple[WorldDataLocationRow, ...] = ()
    characters: tuple[WorldDataCharacterRow, ...] = ()
    threads: tuple[WorldDataThreadRow, ...] = ()
    links: tuple[WorldDataEntityLinkRow, ...] = ()
    knowledge_edges: tuple[WorldDataKnowledgeEdgeRow, ...] = ()
    message_visibility: tuple[WorldDataMessageVisibilityRow, ...] = ()
    suggestions: tuple[WorldDataSuggestionRow, ...] = ()
    suggestion_groups: tuple[WorldDataSuggestionGroupRow, ...] = ()
    audit: tuple[WorldDataAuditRow, ...] = ()
    loss_conditions: tuple[WorldDataLossConditionRow, ...] = ()
    active_loss_outcome: WorldDataLossOutcomeRow | None = None
    error: str | None = None

    @property
    def save_id(self) -> str | None:
        return self.active_save_id

    @property
    def save_title(self) -> str | None:
        return self.save.title if self.save is not None else None

    @property
    def state_rows(self) -> tuple[WorldDataStateRow, ...]:
        return self.world_state

    @property
    def memory_rows(self) -> tuple[WorldDataMemoryRow, ...]:
        return self.memories

    @property
    def context_input_rows(self) -> tuple[WorldDataContextInputRow, ...]:
        return self.context_inputs

    @property
    def summary_rows(self) -> tuple[WorldDataSummaryRow, ...]:
        return self.summaries

    @property
    def location_rows(self) -> tuple[WorldDataLocationRow, ...]:
        return self.locations

    @property
    def character_rows(self) -> tuple[WorldDataCharacterRow, ...]:
        return self.characters

    @property
    def thread_rows(self) -> tuple[WorldDataThreadRow, ...]:
        return self.threads

    @property
    def link_rows(self) -> tuple[WorldDataEntityLinkRow, ...]:
        return self.links

    @property
    def suggestion_rows(self) -> tuple[WorldDataSuggestionRow, ...]:
        return self.suggestions

    @property
    def suggestion_group_rows(self) -> tuple[WorldDataSuggestionGroupRow, ...]:
        return self.suggestion_groups

    @property
    def audit_rows(self) -> tuple[WorldDataAuditRow, ...]:
        return self.audit

    @property
    def loss_condition_rows(self) -> tuple[WorldDataLossConditionRow, ...]:
        return self.loss_conditions


@dataclass(frozen=True)
class WorldDataScenarioEdit:
    title: str
    premise: str
    player_role: str
    content_sections: tuple[tuple[str, str], ...]
    player_character_name: str = ""
    character_starters: tuple[ScenarioCharacterStarter, ...] = ()
    section_content_ratings: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ScenarioEdit:
    title: str
    premise: str
    player_role: str
    content: dict[str, object]
    player_character_name: str = ""
    character_starters: tuple[ScenarioCharacterStarter, ...] = ()
    section_content_ratings: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class WorldStateEdit:
    state_id: str
    key: str
    value_json: str
    category: str
    confidence: float
    archived: bool = False
    original_key: str | None = None
    source_message_id: str | None = None


@dataclass(frozen=True)
class MemoryEdit:
    memory_id: str
    body: str
    tags: tuple[str, ...]
    importance: float
    archived: bool = False
    source_message_id: str | None = None


@dataclass(frozen=True)
class SummaryEdit:
    summary_id: str
    body: str
    archived: bool = False


@dataclass(frozen=True)
class WorldDataEdits:
    scenario: WorldDataScenarioEdit | ScenarioEdit | None
    world_state: tuple[object, ...] = ()
    memories: tuple[object, ...] = ()
    summaries: tuple[object, ...] = ()
    scene: object | None = None
    locations: tuple[object, ...] = ()
    characters: tuple[object, ...] = ()
    threads: tuple[object, ...] = ()
    links: tuple[object, ...] = ()
    suggestions: tuple[object, ...] = ()
    suggestion_groups: tuple[object, ...] = ()
    loss_conditions: tuple[object, ...] = ()

    @property
    def state_rows(self) -> tuple[object, ...]:
        return self.world_state

    @property
    def memory_rows(self) -> tuple[object, ...]:
        return self.memories

    @property
    def summary_rows(self) -> tuple[object, ...]:
        return self.summaries

    @property
    def loss_condition_rows(self) -> tuple[object, ...]:
        return self.loss_conditions


@dataclass(frozen=True)
class WorldDataApplyResult:
    model: WorldDataModel
    state_archive_count: int
    memory_archive_count: int
    summary_archive_count: int
    location_archive_count: int = 0
    thread_archive_count: int = 0

    @property
    def error(self) -> str | None:
        return self.model.error


@dataclass(frozen=True)
class ScenarioDefinitionApplyResult:
    model: WorldDataModel
    linked_save_count: int

    @property
    def error(self) -> str | None:
        return self.model.error


class WorldDataService:
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

    def build_model(self, active_save_id: str | None | object = ...) -> WorldDataModel:
        requested_save_id = (
            self.active_save_id
            if active_save_id is ...
            else cast(str | None, active_save_id)
        )
        active_save = _active_save(self.repositories, requested_save_id)
        if active_save is None:
            return WorldDataModel(
                active_save_id=None,
                save=None,
                scenario=None,
                error="No save loaded",
            )
        details = self.repositories.load_save_details(active_save.id)
        if details is None:
            return WorldDataModel(
                active_save_id=None,
                save=None,
                scenario=None,
                error="No save loaded",
            )
        log_event(
            "world_data.editor_opened",
            save_id=active_save.id,
            state_count=len(self.repositories.list_world_state(active_save.id)),
            memory_count=len(self.repositories.list_memories(active_save.id)),
            summary_count=len(self.repositories.list_summaries(active_save.id)),
        )
        _expire_stale_pending_suggestions(self.repositories, active_save.id)
        snapshot = self.repositories.get_scene_snapshot(active_save.id)
        suggestion_records = tuple(
            self.repositories.list_context_update_suggestions(active_save.id)
        )
        messages = tuple(self.repositories.list_messages(active_save.id))
        message_ratings = {
            message.id: message.content_rating
            for message in messages
        }
        allowed_rating = self.allowed_content_rating
        return WorldDataModel(
            active_save_id=active_save.id,
            save=active_save,
            scenario=_scenario_model_from_record(
                details.scenario,
                allowed_rating=self.allowed_content_rating,
            ),
            world_state=tuple(
                WorldDataStateRow(
                    row_id=record.id,
                    key=record.key,
                    original_key=record.key,
                    category=record.category,
                    confidence=record.confidence,
                    value_json=_dump_json(record.value),
                    source_message_id=record.source_message_id,
                )
                for record in self.repositories.list_world_state(active_save.id)
                if not _record_exceeds_rating(
                    record,
                    allowed_rating=allowed_rating,
                    message_ratings=message_ratings,
                )
            ),
            memories=tuple(
                WorldDataMemoryRow(
                    memory_id=record.id,
                    body=record.body,
                    tags_text=", ".join(record.tags),
                    importance=record.importance,
                    source_message_id=record.source_message_id,
                    source_message_ids=tuple(record.source_message_ids),
                    consolidated=(
                        len(record.source_message_ids) > 1
                        or any(tag.casefold() == "dossier" for tag in record.tags)
                    ),
                )
                for record in self.repositories.list_memories(active_save.id)
                if not _record_exceeds_rating(
                    record,
                    allowed_rating=allowed_rating,
                    message_ratings=message_ratings,
                )
            ),
            context_inputs=tuple(
                _context_input_row(record)
                for record in self.repositories.list_context_sources(active_save.id)
                if not _context_source_exceeds_rating(
                    record,
                    allowed_rating=allowed_rating,
                    message_ratings=message_ratings,
                )
            ),
            summaries=tuple(
                WorldDataSummaryRow(
                    summary_id=record.id,
                    body=record.body,
                    provider=record.provider,
                    model=record.model,
                    covers_message_start_id=record.covers_message_start_id,
                    covers_message_end_id=record.covers_message_end_id,
                )
                for record in self.repositories.list_summaries(active_save.id)
                if not _summary_exceeds_rating(
                    record,
                    allowed_rating=allowed_rating,
                    messages=messages,
                )
            ),
            scene=(
                _scene_row(snapshot)
                if snapshot
                and not _record_exceeds_rating(
                    snapshot,
                    allowed_rating=allowed_rating,
                    message_ratings=message_ratings,
                )
                else None
            ),
            locations=tuple(
                _location_row(record)
                for record in self.repositories.list_locations(active_save.id)
                if not _record_exceeds_rating(
                    record,
                    allowed_rating=allowed_rating,
                    message_ratings=message_ratings,
                )
            ),
            characters=tuple(
                _character_row(record)
                for record in self.repositories.list_characters(active_save.id)
                if not (
                    content_rating_exceeds(
                        minimum_rating=record.content_rating,
                        allowed_rating=allowed_rating or "unrated",
                    )
                    or _record_exceeds_rating(
                        record,
                        allowed_rating=allowed_rating,
                        message_ratings=message_ratings,
                    )
                )
            ),
            threads=tuple(
                _thread_row(record)
                for record in self.repositories.list_active_threads(active_save.id)
                if not _record_exceeds_rating(
                    record,
                    allowed_rating=allowed_rating,
                    message_ratings=message_ratings,
                )
            ),
            links=tuple(
                _link_row(record)
                for record in self.repositories.list_entity_links(active_save.id)
                if not _record_exceeds_rating(
                    record,
                    allowed_rating=allowed_rating,
                    message_ratings=message_ratings,
                )
            ),
            knowledge_edges=tuple(
                _knowledge_edge_row(record)
                for record in self.repositories.list_character_knowledge_edges(
                    active_save.id,
                    include_archived=True,
                )
                if not _record_exceeds_rating(
                    record,
                    allowed_rating=allowed_rating,
                    message_ratings=message_ratings,
                )
            ),
            message_visibility=tuple(
                _message_visibility_row(record)
                for record in self.repositories.list_message_visibility(active_save.id)
                if not _record_exceeds_rating(
                    record,
                    allowed_rating=allowed_rating,
                    message_ratings=message_ratings,
                    extra_source_ids=(record.message_id,),
                )
            ),
            suggestions=tuple(
                _suggestion_row(record)
                for record in suggestion_records
                if not _record_exceeds_rating(
                    record,
                    allowed_rating=allowed_rating,
                    message_ratings=message_ratings,
                )
            ),
            suggestion_groups=_suggestion_group_rows(
                tuple(
                    record
                    for record in suggestion_records
                    if not _record_exceeds_rating(
                        record,
                        allowed_rating=allowed_rating,
                        message_ratings=message_ratings,
                    )
                )
            ),
            audit=tuple(
                _audit_row(record)
                for record in self.repositories.list_context_update_audit(
                    active_save.id
                )
                if not _record_exceeds_rating(
                    record,
                    allowed_rating=allowed_rating,
                    message_ratings=message_ratings,
                )
            ),
            loss_conditions=(),
            active_loss_outcome=None,
        )

    def build_scenario_definition_model(self, scenario_id: str) -> WorldDataModel:
        scenario = self.repositories.get_scenario(scenario_id)
        if scenario is None:
            return WorldDataModel(
                active_save_id=None,
                save=None,
                scenario=None,
                error=f"Unknown scenario id: {scenario_id}",
            )
        return WorldDataModel(
            active_save_id=None,
            save=None,
            scenario=_scenario_model_from_record(
                scenario,
                allowed_rating=self.allowed_content_rating,
            ),
        )

    def apply_scenario_definition_edit(
        self,
        scenario_id: str,
        edit: WorldDataScenarioEdit | ScenarioEdit,
    ) -> ScenarioDefinitionApplyResult:
        scenario = self.repositories.get_scenario(scenario_id)
        if scenario is None:
            log_error_event(
                "world_data.scenario_definition_apply_failed",
                scenario_id=scenario_id,
                error="Unknown scenario id",
            )
            raise ValueError(f"Unknown scenario id: {scenario_id}")
        (
            title,
            premise,
            player_character_name,
            player_role,
            visible_content,
        ) = _scenario_edit_values(edit, scenario_type=scenario.type)
        scenario_model = _scenario_model_from_record(scenario)
        linked_save_count = self.repositories.count_saves_for_scenario(scenario_id)
        if _scenario_values_changed(
            scenario=scenario_model,
            title=title,
            premise=premise,
            player_character_name=player_character_name,
            player_role=player_role,
            content=visible_content,
        ):
            content = _content_with_preserved_metadata(
                visible_content,
                scenario,
                section_ratings=dict(edit.section_content_ratings),
            )
            self.repositories.update_scenario(
                scenario_id=scenario_id,
                title=title,
                premise=premise,
                player_role=player_role,
                content=content,
            )
        log_event(
            "world_data.scenario_definition_applied",
            scenario_id=scenario_id,
            linked_save_count=linked_save_count,
        )
        return ScenarioDefinitionApplyResult(
            model=self.build_scenario_definition_model(scenario_id),
            linked_save_count=linked_save_count,
        )

    def apply_edits(
        self,
        edits: WorldDataEdits,
        active_save_id: str | None | object = ...,
    ) -> WorldDataApplyResult:
        model = self.build_model(active_save_id=active_save_id)
        if model.save_id is None or model.scenario is None:
            log_error_event("world_data.apply_failed", error="No save loaded")
            raise ValueError("No save loaded")
        if edits.scenario is None:
            raise ValueError("Scenario edits are required")
        save_id = model.save_id
        state_archive_count = 0
        memory_archive_count = 0
        summary_archive_count = 0
        location_archive_count = 0
        thread_archive_count = 0
        try:
            _validate_world_data_ids(
                edits=edits,
                model=model,
                repositories=self.repositories,
                save_id=save_id,
            )
            _validate_world_state_key_collisions(edits=edits, model=model)
            self.repositories.begin_transaction()
            (
                scenario_title,
                scenario_premise,
                scenario_player_character_name,
                scenario_player_role,
                scenario_visible_content,
            ) = (
                _scenario_edit_values(
                    edits.scenario,
                    scenario_type=model.scenario.scenario_type,
                )
            )
            scenario_id = model.scenario.scenario_id
            if _scenario_values_changed(
                scenario=model.scenario,
                title=scenario_title,
                premise=scenario_premise,
                player_character_name=scenario_player_character_name,
                player_role=scenario_player_role,
                content=scenario_visible_content,
            ):
                source_scenario = self.repositories.get_scenario(
                    model.scenario.scenario_id
                )
                scenario_content = _content_with_preserved_metadata(
                    scenario_visible_content,
                    source_scenario,
                    section_ratings=dict(
                        edits.scenario.section_content_ratings
                    ),
                )
                scenario_id = _scenario_id_for_single_save_edit(
                    repositories=self.repositories,
                    save_id=save_id,
                    scenario=model.scenario,
                    title=scenario_title,
                    premise=scenario_premise,
                    player_role=scenario_player_role,
                    content=scenario_content,
                )
            for row in edits.world_state:
                row_data = _state_edit_values(row)
                key = row_data.key.strip()
                original_key = (row_data.original_key or row_data.key).strip()
                if original_key == "loop.current" or key == "loop.current":
                    if row_data.archived or key != "loop.current":
                        raise ValueError("The time-loop clock state cannot be modified")
                    before_loop_current = _find_state(model.state_rows, "loop.current")
                    loop_current_value = _validated_state_row(row)
                    if (
                        before_loop_current is None
                        or before_loop_current.value_json
                        != _dump_json(loop_current_value)
                        or before_loop_current.category != row_data.category.strip()
                        or before_loop_current.confidence != row_data.confidence
                    ):
                        raise ValueError("The time-loop clock state cannot be modified")
                    # Preserve the typed envelope and its source provenance
                    # when unrelated World Data edits submit an unchanged row.
                    continue
                if row_data.archived:
                    if original_key:
                        before = _find_state(model.state_rows, original_key)
                        if before is not None:
                            self.repositories.delete_entity_links_for_endpoint(
                                save_id=save_id,
                                entity_type="world_state",
                                entity_id=before.state_id,
                            )
                        self.repositories.archive_world_state(
                            save_id=save_id,
                            key=original_key,
                        )
                        self.repositories.add_state_change(
                            save_id=save_id,
                            operation="manual_world_data_edit",
                            state_key=original_key,
                            before_json=before.value_json if before else None,
                            after_json=None,
                            source_message_id=None,
                        )
                        state_archive_count += 1
                    continue
                if not key:
                    raise ValueError("World-state key is required")
                value = _validated_state_row(row)
                if original_key and original_key != key:
                    before_rename = _find_state(model.state_rows, original_key)
                    if before_rename is not None:
                        self.repositories.delete_entity_links_for_endpoint(
                            save_id=save_id,
                            entity_type="world_state",
                            entity_id=before_rename.state_id,
                        )
                    self.repositories.archive_world_state(
                        save_id=save_id,
                        key=original_key,
                    )
                    state_archive_count += 1
                before = _find_state(model.state_rows, original_key or key)
                self.repositories.upsert_world_state(
                    save_id=save_id,
                    key=key,
                    value=value,
                    category=row_data.category.strip(),
                    confidence=row_data.confidence,
                    source_message_id=None,
                )
                self.repositories.add_state_change(
                    save_id=save_id,
                    operation="manual_world_data_edit",
                    state_key=key,
                    before_json=before.value_json if before else None,
                    after_json=_dump_json(value),
                    source_message_id=None,
                )
            for row in edits.memories:
                memory_data = _memory_edit_values(row)
                if memory_data.archived:
                    if memory_data.memory_id:
                        self.repositories.delete_entity_links_for_endpoint(
                            save_id=save_id,
                            entity_type="memory",
                            entity_id=memory_data.memory_id,
                        )
                        self.repositories.archive_memory(memory_data.memory_id)
                        memory_archive_count += 1
                    continue
                body = memory_data.body.strip()
                if not body:
                    raise ValueError("Memory body is required")
                if memory_data.memory_id:
                    self.repositories.update_memory(
                        memory_id=memory_data.memory_id,
                        body=body,
                        tags=memory_data.tags,
                        importance=memory_data.importance,
                        clear_source=True,
                    )
                else:
                    self.repositories.add_memory(
                        save_id=save_id,
                        body=body,
                        tags=memory_data.tags,
                        importance=memory_data.importance,
                    )
            for row in edits.summaries:
                summary_data = _summary_edit_values(row)
                if summary_data.archived:
                    self.repositories.delete_entity_links_for_endpoint(
                        save_id=save_id,
                        entity_type="summary",
                        entity_id=summary_data.summary_id,
                    )
                    self.repositories.archive_summary(summary_data.summary_id)
                    summary_archive_count += 1
                    continue
                summary_body = summary_data.body.strip()
                if not summary_body:
                    raise ValueError("Summary body is required")
                self.repositories.update_summary(
                    summary_id=summary_data.summary_id,
                    body=summary_body,
                    content_rating="unclassified",
                )
            if edits.scene is not None:
                _upsert_scene_snapshot(
                    repositories=self.repositories,
                    values=_scene_update_values(
                        row=edits.scene, model=model, save_id=save_id
                    ),
                )
            for row in edits.locations:
                location_row = _location_edit_row(row)
                if location_row.archived:
                    if location_row.location_id:
                        _validate_location_archive_allowed(
                            model=model,
                            location_id=location_row.location_id,
                        )
                        self.repositories.delete_entity_links_for_endpoint(
                            save_id=save_id,
                            entity_type="location",
                            entity_id=location_row.location_id,
                        )
                        self.repositories.archive_location(location_row.location_id)
                        location_archive_count += 1
                    continue
                if location_row.location_id:
                    self.repositories.update_location(
                        _location_update_values(row=location_row, model=model)
                    )
                else:
                    name = location_row.name.strip()
                    if not name:
                        raise ValueError("Location name is required")
                    self.repositories.add_location(
                        save_id=save_id,
                        name=name,
                        aliases=_csv(location_row.aliases_text),
                        description=location_row.description.strip(),
                        visual_description=location_row.visual_description.strip(),
                        parent_location_id=_none_if_blank(
                            location_row.parent_location_id
                        ),
                        connections=_csv(location_row.connections_text),
                        status=location_row.status.strip(),
                        hazards=_csv(location_row.hazards_text),
                        source_message_id=None,
                        locked_fields=list(location_row.locked_fields),
                    )
            for row in edits.characters:
                self.repositories.update_character(
                    _character_update_values(row=row, model=model)
                )
            for row in edits.threads:
                thread_row = _thread_edit_row(row)
                if thread_row.archived:
                    if thread_row.thread_id:
                        self.repositories.delete_entity_links_for_endpoint(
                            save_id=save_id,
                            entity_type="active_thread",
                            entity_id=thread_row.thread_id,
                        )
                        self.repositories.archive_active_thread(thread_row.thread_id)
                        thread_archive_count += 1
                    continue
                if thread_row.thread_id:
                    self.repositories.update_active_thread(
                        _thread_update_values(row=thread_row, model=model)
                    )
                else:
                    title = thread_row.title.strip()
                    if not title:
                        raise ValueError("Thread title is required")
                    self.repositories.add_active_thread(
                        save_id=save_id,
                        title=title,
                        description=thread_row.description.strip(),
                        status=thread_row.status.strip(),
                        priority=int(thread_row.priority),
                        visibility=thread_row.visibility.strip(),
                        related_entities=_csv(thread_row.related_entities_text),
                        source_message_id=None,
                        locked_fields=list(thread_row.locked_fields),
                    )
            for row in edits.links:
                link_data = _link_edit_values(row)
                if link_data.deleted:
                    self.repositories.delete_entity_link(link_data.link_id)
                elif not link_data.link_id:
                    self.repositories.add_entity_link(
                        save_id=save_id,
                        entity_type=link_data.entity_type.strip(),
                        entity_id=link_data.entity_id.strip(),
                        target_type=link_data.target_type.strip(),
                        target_id=link_data.target_id.strip(),
                        relation=link_data.relation.strip(),
                        source_message_id=None,
                        overwrite_source=True,
                    )
            for row in edits.suggestions:
                latest_model = self.build_model(active_save_id=save_id)
                suggestion = _suggestion_from_model(
                    latest_model,
                    _suggestion_row_id(row),
                )
                action = _suggestion_action(row)
                if action == "reject":
                    _reject_suggestion_batch(
                        repositories=self.repositories,
                        save_id=save_id,
                        model=latest_model,
                        suggestions=(suggestion,),
                        operation="manual_suggestion_reject",
                        reason=suggestion.reason,
                    )
                elif action == "apply":
                    _apply_suggestion_batch(
                        repositories=self.repositories,
                        save_id=save_id,
                        model=latest_model,
                        suggestions=(suggestion,),
                        operation="manual_suggestion_apply",
                    )
            for row in edits.suggestion_groups:
                latest_model = self.build_model(active_save_id=save_id)
                group = _suggestion_group_from_model(
                    latest_model,
                    _suggestion_group_row_id(row),
                )
                action = _suggestion_group_action(row)
                if action == "apply":
                    _apply_suggestion_group(
                        repositories=self.repositories,
                        save_id=save_id,
                        model=latest_model,
                        group=group,
                    )
                elif action == "reject":
                    _resolve_suggestion_group(
                        repositories=self.repositories,
                        save_id=save_id,
                        model=latest_model,
                        group=group,
                        status="rejected",
                        operation="manual_suggestion_group_reject",
                    )
                elif action == "dismiss":
                    _resolve_suggestion_group(
                        repositories=self.repositories,
                        save_id=save_id,
                        model=latest_model,
                        group=group,
                        status="dismissed",
                        operation="manual_suggestion_group_dismiss",
                    )
            self.repositories.commit_transaction()
        except Exception:
            self.repositories.rollback_transaction()
            raise
        log_event(
            "world_data.edits_applied",
            save_id=save_id,
            scenario_id=scenario_id,
            state_row_count=len(edits.world_state),
            memory_row_count=len(edits.memories),
            summary_row_count=len(edits.summaries),
            state_archive_count=state_archive_count,
            memory_archive_count=memory_archive_count,
            summary_archive_count=summary_archive_count,
            location_archive_count=location_archive_count,
            thread_archive_count=thread_archive_count,
        )
        return WorldDataApplyResult(
            model=self.build_model(active_save_id=save_id),
            state_archive_count=state_archive_count,
            memory_archive_count=memory_archive_count,
            summary_archive_count=summary_archive_count,
            location_archive_count=location_archive_count,
            thread_archive_count=thread_archive_count,
        )

    def apply_suggestions(
        self,
        suggestion_ids: tuple[str, ...] | list[str],
        *,
        active_save_id: str | None | object = ...,
        operation: str = "agent_suggestion_apply",
        reason: str = "",
    ) -> WorldDataApplyResult:
        model = self.build_model(active_save_id=active_save_id)
        if model.save_id is None or model.scenario is None:
            log_error_event(
                "world_data.suggestion_apply_failed",
                error="No save loaded",
            )
            raise ValueError("No save loaded")
        ids = tuple(dict.fromkeys(suggestion_ids))
        if not ids:
            return WorldDataApplyResult(
                model=model,
                state_archive_count=0,
                memory_archive_count=0,
                summary_archive_count=0,
            )
        suggestions_by_id = {
            suggestion.suggestion_id: suggestion for suggestion in model.suggestions
        }
        suggestions = tuple(
            suggestions_by_id[suggestion_id] for suggestion_id in ids
        )
        if len(suggestions) != len(ids):
            missing = next(
                suggestion_id
                for suggestion_id in ids
                if suggestion_id not in suggestions_by_id
            )
            raise ValueError(f"Unknown suggestion id: {missing}")
        try:
            self.repositories.begin_transaction()
            _apply_suggestion_batch(
                repositories=self.repositories,
                save_id=model.save_id,
                model=model,
                suggestions=suggestions,
                operation=operation,
                reason=reason,
            )
            self.repositories.commit_transaction()
        except Exception:
            self.repositories.rollback_transaction()
            raise
        return WorldDataApplyResult(
            model=self.build_model(active_save_id=model.save_id),
            state_archive_count=0,
            memory_archive_count=0,
            summary_archive_count=0,
        )

    def reject_suggestions(
        self,
        suggestion_ids: tuple[str, ...] | list[str],
        *,
        active_save_id: str | None | object = ...,
        operation: str = "agent_suggestion_reject",
        reason: str = "",
    ) -> WorldDataApplyResult:
        model = self.build_model(active_save_id=active_save_id)
        if model.save_id is None or model.scenario is None:
            log_error_event(
                "world_data.suggestion_reject_failed",
                error="No save loaded",
            )
            raise ValueError("No save loaded")
        ids = tuple(dict.fromkeys(suggestion_ids))
        if not ids:
            return WorldDataApplyResult(
                model=model,
                state_archive_count=0,
                memory_archive_count=0,
                summary_archive_count=0,
            )
        suggestions_by_id = {
            suggestion.suggestion_id: suggestion for suggestion in model.suggestions
        }
        suggestions = tuple(
            suggestions_by_id[suggestion_id] for suggestion_id in ids
        )
        if len(suggestions) != len(ids):
            missing = next(
                suggestion_id
                for suggestion_id in ids
                if suggestion_id not in suggestions_by_id
            )
            raise ValueError(f"Unknown suggestion id: {missing}")
        try:
            self.repositories.begin_transaction()
            _reject_suggestion_batch(
                repositories=self.repositories,
                save_id=model.save_id,
                model=model,
                suggestions=suggestions,
                operation=operation,
                reason=reason,
            )
            self.repositories.commit_transaction()
        except Exception:
            self.repositories.rollback_transaction()
            raise
        return WorldDataApplyResult(
            model=self.build_model(active_save_id=model.save_id),
            state_archive_count=0,
            memory_archive_count=0,
            summary_archive_count=0,
        )


def _scenario_id_for_single_save_edit(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    scenario: WorldDataScenarioModel,
    title: str,
    premise: str,
    player_role: str,
    content: dict[str, object],
) -> str:
    if repositories.count_saves_for_scenario(scenario.scenario_id) > 1:
        forked = repositories.create_scenario(
            type=scenario.scenario_type,
            title=title,
            premise=premise,
            player_role=player_role,
            content=content,
        )
        repositories.update_save_scenario(
            save_id=save_id,
            scenario_id=forked.id,
        )
        log_event(
            "world_data.scenario_forked",
            save_id=save_id,
            original_scenario_id=scenario.scenario_id,
            scenario_id=forked.id,
        )
        return forked.id

    repositories.update_scenario(
        scenario_id=scenario.scenario_id,
        title=title,
        premise=premise,
        player_role=player_role,
        content=content,
    )
    return scenario.scenario_id


def _scenario_model_from_record(
    scenario: ScenarioRecord,
    *,
    allowed_rating: str | None = None,
) -> WorldDataScenarioModel:
    if allowed_rating is not None and content_rating_exceeds(
        minimum_rating=scenario_content_rating(scenario.content_json),
        allowed_rating=allowed_rating,
    ):
        return WorldDataScenarioModel(
            scenario_id=scenario.id,
            scenario_type=scenario.type,
            title=CONTENT_FILTER_TRANSITION,
            premise=CONTENT_FILTER_TRANSITION,
            player_character_name="",
            player_role=CONTENT_FILTER_TRANSITION,
            content_sections=(),
        )
    scenario_content = _scenario_content(scenario.content_json)
    return WorldDataScenarioModel(
        scenario_id=scenario.id,
        scenario_type=scenario.type,
        title=scenario.title,
        premise=scenario.premise,
        player_character_name=_scenario_player_character_name(scenario_content),
        player_role=scenario.player_role,
        content_sections=tuple(
            (key, _section_text(value))
            for key, value in scenario_content
            if key not in _SCENARIO_NON_SECTION_CONTENT_KEYS
        ),
        generation_prompt=_scenario_generation_prompt(scenario_content),
        character_starters=_scenario_character_starters(scenario_content),
    )


def _scenario_values_changed(
    *,
    scenario: WorldDataScenarioModel,
    title: str,
    premise: str,
    player_character_name: str,
    player_role: str,
    content: dict[str, object],
) -> bool:
    current_content = _sync_scenario_core_fields(
        title=scenario.title,
        premise=scenario.premise,
        player_character_name=scenario.player_character_name,
        player_role=scenario.player_role,
        content=dict(scenario.content_sections),
        scenario_type=scenario.scenario_type,
        character_starters=scenario.character_starters,
    )
    return (
        title != scenario.title
        or premise != scenario.premise
        or player_character_name != scenario.player_character_name
        or player_role != scenario.player_role
        or content != current_content
    )


def _active_save(
    repositories: PersistenceRepositories,
    active_save_id: str | None,
) -> SaveRecord | None:
    if active_save_id is None:
        return None
    return repositories.get_save(active_save_id)


def _scenario_content(content_json: str) -> tuple[tuple[str, object], ...]:
    loaded = json.loads(content_json)
    if not isinstance(loaded, dict):
        return ()
    return tuple((str(key), value) for key, value in loaded.items())


_SCENARIO_CORE_CONTENT_KEYS = frozenset(
    (
        "title",
        "premise",
        "setup_line",
        "starting_scene",
        "player_character_name",
        "player_role",
    )
)
_SCENARIO_NON_SECTION_CONTENT_KEYS = frozenset(
    (*_SCENARIO_CORE_CONTENT_KEYS, CHARACTER_STARTERS_CONTENT_KEY, "_source")
)


def _scenario_player_character_name(
    content: tuple[tuple[str, object], ...],
) -> str:
    for key, value in content:
        if key == "player_character_name" and isinstance(value, str):
            return value.strip()
    return ""


def _scenario_character_starters(
    content: tuple[tuple[str, object], ...],
) -> tuple[ScenarioCharacterStarter, ...]:
    payload = dict(content).get(CHARACTER_STARTERS_CONTENT_KEY)
    return normalize_scenario_character_starters(payload, strict=False)


def _scenario_generation_prompt(
    content: tuple[tuple[str, object], ...],
) -> str | None:
    source = dict(content).get("_source")
    if not isinstance(source, dict):
        return None
    prompt = source.get("generation_prompt")
    if not isinstance(prompt, str):
        return None
    text = prompt.strip()
    return text or None


def _content_with_preserved_metadata(
    content: dict[str, object],
    scenario: ScenarioRecord | None,
    *,
    section_ratings: Mapping[str, str] | None = None,
) -> dict[str, object]:
    normalized = dict(content)
    if scenario is None:
        return normalized
    scenario_content = dict(_scenario_content(scenario.content_json))
    source = scenario_content.get("_source")
    resolved_section_ratings = dict(section_ratings or {})
    if not resolved_section_ratings:
        existing_source = source if isinstance(source, Mapping) else {}
        existing_ratings = existing_source.get("section_content_ratings")
        prior_ratings = (
            existing_ratings if isinstance(existing_ratings, Mapping) else {}
        )
        resolved_section_ratings = {
            key: (
                str(prior_ratings.get(key, CONTENT_RATING_UNCLASSIFIED))
                if scenario_content.get(key) == value
                else CONTENT_RATING_UNCLASSIFIED
            )
            for key, value in normalized.items()
            if isinstance(value, str)
        }
    normalized["_source"] = metadata_with_scenario_content_ratings(
        source if isinstance(source, dict) else None,
        aggregate_rating=maximum_content_rating(
            tuple(resolved_section_ratings.values())
        ),
        section_ratings=resolved_section_ratings,
    )
    _preserve_scenario_starter_reference_metadata(
        content=normalized,
        existing_content=scenario_content,
    )
    return normalized


def _preserve_scenario_starter_reference_metadata(
    *,
    content: dict[str, object],
    existing_content: dict[str, object],
) -> None:
    existing_starters = normalize_scenario_character_starters(
        existing_content.get(CHARACTER_STARTERS_CONTENT_KEY),
        strict=False,
    )
    incoming_starters = normalize_scenario_character_starters(
        content.get(CHARACTER_STARTERS_CONTENT_KEY),
        strict=False,
    )
    if not existing_starters or not incoming_starters:
        return
    used_existing_indexes: set[int] = set()
    merged: list[ScenarioCharacterStarter] = []
    for incoming in incoming_starters:
        existing_index, existing = _matching_existing_starter(
            incoming,
            existing_starters,
            used_existing_indexes,
        )
        if existing is not None and existing_index is not None:
            used_existing_indexes.add(existing_index)
            incoming = replace(
                incoming,
                starter_id=incoming.starter_id or existing.starter_id,
                reference_image=(
                    incoming.reference_image
                    if incoming.reference_image is not None
                    else existing.reference_image
                ),
            )
        merged.append(incoming)
    content[CHARACTER_STARTERS_CONTENT_KEY] = [
        scenario_character_starter_to_json(starter) for starter in merged
    ]


def _matching_existing_starter(
    incoming: ScenarioCharacterStarter,
    existing_starters: tuple[ScenarioCharacterStarter, ...],
    used_existing_indexes: set[int],
) -> tuple[int | None, ScenarioCharacterStarter | None]:
    if incoming.starter_id:
        for index, existing in enumerate(existing_starters):
            if (
                index not in used_existing_indexes
                and existing.starter_id == incoming.starter_id
            ):
                return index, existing
    name = _starter_lookup_key(incoming.name)
    if not name:
        return None, None
    matches = [
        (index, existing)
        for index, existing in enumerate(existing_starters)
        if (
            index not in used_existing_indexes
            and _starter_lookup_key(existing.name) == name
        )
    ]
    if len(matches) != 1:
        return None, None
    return matches[0]


def _starter_lookup_key(value: str) -> str:
    return " ".join(value.casefold().split())


def _section_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


@dataclass(frozen=True)
class _StateEditValues:
    key: str
    original_key: str | None
    category: str
    confidence: float
    value_json: str
    source_message_id: str | None
    archived: bool


@dataclass(frozen=True)
class _MemoryEditValues:
    memory_id: str
    body: str
    tags: list[str]
    importance: float
    archived: bool


@dataclass(frozen=True)
class _SummaryEditValues:
    summary_id: str
    body: str
    archived: bool


@dataclass(frozen=True)
class _LinkEditValues:
    link_id: str
    entity_type: str
    entity_id: str
    target_type: str
    target_id: str
    relation: str
    deleted: bool


def _scene_row(record: SceneSnapshotRecord) -> WorldDataSceneRow:
    return WorldDataSceneRow(
        snapshot_id=record.id,
        current_location_id=record.current_location_id,
        situation=record.situation,
        objective=record.objective,
        in_world_time=record.in_world_time,
        time_of_day=record.time_of_day,
        day_of_week=record.day_of_week,
        world_day_index=record.world_day_index,
        weather=record.weather,
        mood=record.mood,
        nearby_objects_text=", ".join(record.nearby_objects),
        hazards_text=", ".join(record.hazards),
        present_character_ids_text=", ".join(record.present_character_ids),
        source_message_id=record.source_message_id,
        locked_fields=tuple(record.locked_fields),
    )


def _location_row(record: LocationRecord) -> WorldDataLocationRow:
    return WorldDataLocationRow(
        location_id=record.id,
        name=record.name,
        aliases_text=", ".join(record.aliases),
        description=record.description,
        visual_description=record.visual_description,
        parent_location_id=record.parent_location_id,
        connections_text=", ".join(record.connections),
        status=record.status,
        hazards_text=", ".join(record.hazards),
        source_message_id=record.source_message_id,
        locked_fields=tuple(record.locked_fields),
    )


def _character_row(record: CharacterRecord) -> WorldDataCharacterRow:
    return WorldDataCharacterRow(
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
        source_message_id=record.source_message_id,
        locked_fields=tuple(record.locked_fields),
        protected_from_maintenance=record.protected_from_maintenance,
        is_player_character=record.is_player_character,
        content_rating=record.content_rating,
    )


def _thread_row(record: ActiveThreadRecord) -> WorldDataThreadRow:
    return WorldDataThreadRow(
        thread_id=record.id,
        title=record.title,
        description=record.description,
        status=record.status,
        priority=record.priority,
        visibility=record.visibility,
        related_entities_text=", ".join(record.related_entities),
        source_message_id=record.source_message_id,
        locked_fields=tuple(record.locked_fields),
    )


def _link_row(record: EntityLinkRecord) -> WorldDataEntityLinkRow:
    return WorldDataEntityLinkRow(
        link_id=record.id,
        entity_type=record.entity_type,
        entity_id=record.entity_id,
        target_type=record.target_type,
        target_id=record.target_id,
        relation=record.relation,
    )


def _knowledge_edge_row(
    record: CharacterKnowledgeEdgeRecord,
) -> WorldDataKnowledgeEdgeRow:
    return WorldDataKnowledgeEdgeRow(
        edge_id=record.id,
        character_id=record.character_id,
        target_type=record.target_type,
        target_id=record.target_id,
        knowledge_state=record.knowledge_state,
        acquisition_method=record.acquisition_method,
        confidence=record.confidence,
        source_message_id=record.source_message_id,
        source_message_ids_text=", ".join(record.source_message_ids),
        evidence_quote=record.evidence_quote,
        archived=record.archived_at is not None,
    )


def _message_visibility_row(
    record: MessageVisibilityRecord,
) -> WorldDataMessageVisibilityRow:
    return WorldDataMessageVisibilityRow(
        visibility_id=record.id,
        message_id=record.message_id,
        character_id=record.character_id,
        visibility=record.visibility,
        confidence=record.confidence,
        source=record.source,
        evidence=record.evidence,
    )


def _suggestion_row(record: ContextUpdateSuggestionRecord) -> WorldDataSuggestionRow:
    return WorldDataSuggestionRow(
        suggestion_id=record.id,
        update_type=record.update_type,
        entity_type=record.entity_type,
        entity_id=record.entity_id,
        field_path=record.field_path,
        proposed_value_json=_dump_json(record.proposed_value),
        status=record.status,
        reason=record.reason,
        confidence=record.confidence,
        source_message_ids_text=", ".join(record.source_message_ids),
    )


def _suggestion_group_rows(
    records: tuple[ContextUpdateSuggestionRecord, ...],
) -> tuple[WorldDataSuggestionGroupRow, ...]:
    grouped: dict[tuple[str, str, str | None, str, str], list[
        ContextUpdateSuggestionRecord
    ]] = {}
    for record in records:
        if record.status != "pending":
            continue
        key = (
            record.update_type,
            record.entity_type,
            record.entity_id,
            record.field_path,
            _dump_json(record.proposed_value),
        )
        grouped.setdefault(key, []).append(record)

    rows: list[WorldDataSuggestionGroupRow] = []
    for key, members in grouped.items():
        first = members[0]
        update_type, entity_type, entity_id, field_path, proposed_value_json = key
        rows.append(
            WorldDataSuggestionGroupRow(
                group_id=_suggestion_group_id(
                    save_id=first.save_id,
                    update_type=update_type,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    field_path=field_path,
                    proposed_value_json=proposed_value_json,
                ),
                suggestion_ids=tuple(member.id for member in members),
                update_type=update_type,
                entity_type=entity_type,
                entity_id=entity_id,
                field_path=field_path,
                proposed_value_json=proposed_value_json,
                status="pending",
                reason="\n".join(
                    dict.fromkeys(
                        member.reason.strip()
                        for member in members
                        if member.reason.strip()
                    )
                ),
                confidence=max(member.confidence for member in members),
                source_message_ids_text=", ".join(
                    dict.fromkeys(
                        source_id
                        for member in members
                        for source_id in member.source_message_ids
                    )
                ),
                suggestion_count=len(members),
            )
        )
    return tuple(rows)


def _suggestion_group_id(
    *,
    save_id: str,
    update_type: str,
    entity_type: str,
    entity_id: str | None,
    field_path: str,
    proposed_value_json: str,
) -> str:
    payload = _dump_json(
        [
            save_id,
            update_type,
            entity_type,
            entity_id,
            field_path,
            proposed_value_json,
        ]
    )
    return f"sugggrp-{sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _expire_stale_pending_suggestions(
    repositories: PersistenceRepositories,
    save_id: str,
) -> None:
    expired = repositories.expire_stale_context_update_suggestions(
        save_id,
        older_than_days=STALE_PENDING_SUGGESTION_DAYS,
    )
    for suggestion in expired:
        repositories.add_context_update_audit(
            save_id=save_id,
            suggestion_id=suggestion.id,
            operation="suggestion_expired",
            entity_type=suggestion.entity_type,
            entity_id=suggestion.entity_id,
            field_path=suggestion.field_path,
            before=None,
            after=suggestion.proposed_value,
            reason=(
                "Pending suggestion expired after "
                f"{STALE_PENDING_SUGGESTION_DAYS} days without review."
            ),
            confidence=suggestion.confidence,
            source_message_ids=suggestion.source_message_ids,
        )


def _context_input_row(record: ContextSourceRecord) -> WorldDataContextInputRow:
    fact_type = str(record.metadata.get("fact_type", "")).strip()
    importance = record.metadata.get("importance", 0.0)
    source_message_ids = record.metadata.get("source_message_ids", [])
    return WorldDataContextInputRow(
        context_source_id=record.id,
        source_type=record.source_type,
        source_id=record.source_id,
        title=record.title,
        body=record.body,
        fact_type=fact_type,
        importance=float(importance) if isinstance(importance, int | float) else 0.0,
        source_message_count=(
            len(source_message_ids) if isinstance(source_message_ids, list) else 0
        ),
        token_estimate=record.token_estimate,
    )


def _record_exceeds_rating(
    record: object,
    *,
    allowed_rating: str | None,
    message_ratings: dict[str, str],
    extra_source_ids: tuple[str, ...] = (),
) -> bool:
    if allowed_rating is None:
        return False
    source_ids = list(extra_source_ids)
    for field_name in (
        "source_message_id",
        "first_seen_message_id",
        "last_updated_message_id",
        "world_time_source_message_id",
    ):
        value = getattr(record, field_name, None)
        if isinstance(value, str) and value:
            source_ids.append(value)
    multiple = getattr(record, "source_message_ids", ())
    if isinstance(multiple, list | tuple):
        source_ids.extend(
            value
            for value in multiple
            if isinstance(value, str) and value
        )
    if not source_ids:
        return content_rating_exceeds(
            minimum_rating=CONTENT_RATING_UNCLASSIFIED,
            allowed_rating=allowed_rating,
        )
    return any(
        content_rating_exceeds(
            minimum_rating=message_ratings.get(
                source_id,
                "unclassified",
            ),
            allowed_rating=allowed_rating,
        )
        for source_id in dict.fromkeys(source_ids)
    )


def _context_source_exceeds_rating(
    record: ContextSourceRecord,
    *,
    allowed_rating: str | None,
    message_ratings: dict[str, str],
) -> bool:
    source_ids = record.metadata.get("source_message_ids", [])
    normalized_ids = (
        tuple(value for value in source_ids if isinstance(value, str))
        if isinstance(source_ids, list)
        else ()
    )
    if record.source_type == "message":
        normalized_ids = (*normalized_ids, record.source_id)
    return _record_exceeds_rating(
        record,
        allowed_rating=allowed_rating,
        message_ratings=message_ratings,
        extra_source_ids=normalized_ids,
    )


def _summary_exceeds_rating(
    record: SummaryRecord,
    *,
    allowed_rating: str | None,
    messages: tuple[MessageRecord, ...],
) -> bool:
    if allowed_rating is None:
        return False
    if content_rating_exceeds(
        minimum_rating=record.content_rating,
        allowed_rating=allowed_rating,
    ):
        return True
    positions = {
        message.id: index
        for index, message in enumerate(messages)
    }
    start = positions.get(record.covers_message_start_id)
    end = positions.get(record.covers_message_end_id)
    if start is None or end is None:
        return True
    if start > end:
        start, end = end, start
    return any(
        content_rating_exceeds(
            minimum_rating=message.content_rating,
            allowed_rating=allowed_rating,
        )
        for message in messages[start : end + 1]
    )


def _audit_row(record: ContextUpdateAuditRecord) -> WorldDataAuditRow:
    return WorldDataAuditRow(
        audit_id=record.id,
        suggestion_id=record.suggestion_id,
        operation=record.operation,
        entity_type=record.entity_type,
        entity_id=record.entity_id,
        field_path=record.field_path,
        before_json=_dump_json(record.before) if record.before is not None else "",
        after_json=_dump_json(record.after) if record.after is not None else "",
        reason=record.reason,
        confidence=record.confidence,
        source_message_ids_text=", ".join(record.source_message_ids),
        created_at=record.created_at,
    )


def _scenario_edit_values(
    edit: WorldDataScenarioEdit | ScenarioEdit,
    *,
    scenario_type: str = "full_roleplay",
) -> tuple[str, str, str, str, dict[str, object]]:
    if isinstance(edit, ScenarioEdit):
        content = _sync_scenario_core_fields(
            title=edit.title,
            premise=edit.premise,
            player_character_name=edit.player_character_name,
            player_role=edit.player_role,
            content=edit.content,
            scenario_type=scenario_type,
            character_starters=edit.character_starters,
        )
        normalized_premise, normalized_content = normalize_scenario_definition(
            scenario_type=scenario_type,
            premise=edit.premise,
            content=content,
        )
        return (
            edit.title.strip(),
            normalized_premise,
            edit.player_character_name.strip(),
            edit.player_role.strip(),
            normalized_content,
        )
    content = _sync_scenario_core_fields(
        title=edit.title,
        premise=edit.premise,
        player_character_name=edit.player_character_name,
        player_role=edit.player_role,
        content={key.strip(): value for key, value in edit.content_sections},
        scenario_type=scenario_type,
        character_starters=edit.character_starters,
    )
    normalized_premise, normalized_content = normalize_scenario_definition(
        scenario_type=scenario_type,
        premise=edit.premise,
        content=content,
    )
    return (
        edit.title.strip(),
        normalized_premise,
        edit.player_character_name.strip(),
        edit.player_role.strip(),
        normalized_content,
    )


def _sync_scenario_core_fields(
    *,
    title: str,
    premise: str,
    player_character_name: str,
    player_role: str,
    content: dict[str, object],
    scenario_type: str,
    character_starters: tuple[ScenarioCharacterStarter, ...] = (),
) -> dict[str, object]:
    synced = {
        key.strip(): value
        for key, value in content.items()
        if key.strip() and key.strip() not in _SCENARIO_NON_SECTION_CONTENT_KEYS
    }
    synced["title"] = title.strip()
    synced["premise"] = premise.strip()
    synced["player_character_name"] = player_character_name.strip()
    synced["player_role"] = player_role.strip()
    if character_starters:
        synced[CHARACTER_STARTERS_CONTENT_KEY] = [
            scenario_character_starter_to_json(starter)
            for starter in character_starters
        ]
    return synced


def _state_edit_values(row: object) -> _StateEditValues:
    if not isinstance(row, WorldDataStateRow | WorldStateEdit):
        raise TypeError(f"Unsupported world-state edit row: {type(row).__name__}")
    key = row.key
    original_key = row.original_key
    return _StateEditValues(
        key=key,
        original_key=original_key or key,
        category=row.category,
        confidence=row.confidence,
        value_json=row.value_json,
        source_message_id=row.source_message_id or None,
        archived=row.archived,
    )


def _memory_edit_values(row: object) -> _MemoryEditValues:
    if isinstance(row, WorldDataMemoryRow):
        resolved_tags = _tags(row.tags_text)
        memory_id = row.memory_id
        body = row.body
        importance = row.importance
        archived = row.archived
    elif isinstance(row, MemoryEdit):
        resolved_tags = list(row.tags)
        memory_id = row.memory_id
        body = row.body
        importance = row.importance
        archived = row.archived
    else:
        raise TypeError(f"Unsupported memory edit row: {type(row).__name__}")
    return _MemoryEditValues(
        memory_id=memory_id,
        body=body,
        tags=resolved_tags,
        importance=importance,
        archived=archived,
    )


def _summary_edit_values(row: object) -> _SummaryEditValues:
    if not isinstance(row, WorldDataSummaryRow | SummaryEdit):
        raise TypeError(f"Unsupported summary edit row: {type(row).__name__}")
    return _SummaryEditValues(
        summary_id=row.summary_id,
        body=row.body,
        archived=row.archived,
    )


def _link_edit_values(row: object) -> _LinkEditValues:
    if not isinstance(row, WorldDataEntityLinkRow):
        raise TypeError(f"Unsupported entity-link edit row: {type(row).__name__}")
    return _LinkEditValues(
        link_id=row.link_id,
        entity_type=row.entity_type,
        entity_id=row.entity_id,
        target_type=row.target_type,
        target_id=row.target_id,
        relation=row.relation,
        deleted=row.deleted,
    )


def _location_edit_row(row: object) -> WorldDataLocationRow:
    if not isinstance(row, WorldDataLocationRow):
        raise TypeError(f"Unsupported location edit row: {type(row).__name__}")
    return row


def _thread_edit_row(row: object) -> WorldDataThreadRow:
    if not isinstance(row, WorldDataThreadRow):
        raise TypeError(f"Unsupported thread edit row: {type(row).__name__}")
    return row


def _scene_update_values(
    *, row: object, model: WorldDataModel, save_id: str
) -> dict[str, Any]:
    if not isinstance(row, WorldDataSceneRow):
        raise TypeError(f"Unsupported scene edit row: {type(row).__name__}")
    current = model.scene
    changed = _changed_fields(
        current,
        row,
        (
            "current_location_id",
            "situation",
            "objective",
            "in_world_time",
            "time_of_day",
            "day_of_week",
            "world_day_index",
            "weather",
            "mood",
            "nearby_objects_text",
            "hazards_text",
            "present_character_ids_text",
        ),
    )
    return {
        "save_id": save_id,
        "snapshot_id": row.snapshot_id or None,
        "current_location_id": _none_if_blank(row.current_location_id),
        "situation": row.situation.strip(),
        "objective": row.objective.strip(),
        "in_world_time": row.in_world_time.strip(),
        "time_of_day": row.time_of_day.strip(),
        "day_of_week": row.day_of_week.strip(),
        "world_day_index": row.world_day_index,
        "weather": row.weather.strip(),
        "mood": row.mood.strip(),
        "nearby_objects": _csv(row.nearby_objects_text),
        "hazards": _csv(row.hazards_text),
        "present_character_ids": _csv(row.present_character_ids_text),
        "source_message_id": None,
        "locked_fields": _locked(row.locked_fields, changed),
        "world_time_changed": any(
            field in SCENE_WORLD_TIME_FIELDS for field in changed
        ),
        "world_time_changed_fields": tuple(
            field for field in changed if field in SCENE_WORLD_TIME_FIELDS
        ),
    }


def _upsert_scene_snapshot(
    *, repositories: PersistenceRepositories, values: dict[str, Any]
) -> SceneSnapshotRecord:
    existing = repositories.get_scene_snapshot(cast(str, values["save_id"]))
    world_time_kwargs: dict[str, Any] = {}
    if values.get("world_time_changed"):
        changed_fields = set(
            cast(tuple[str, ...], values.get("world_time_changed_fields", ()))
        )
        canonical_world_time = canonical_world_time_from_legacy(
            in_world_time=values["in_world_time"],
            time_of_day=(
                ""
                if (
                    "in_world_time" in changed_fields
                    and "time_of_day" not in changed_fields
                )
                else values["time_of_day"]
            ),
            day_of_week=values["day_of_week"],
            world_day_index=values["world_day_index"],
            source_message_id=values.get(
                "world_time_source_message_id",
                values["source_message_id"],
            ),
            confidence=values.get("world_time_confidence"),
        )
        display_fields_changed = bool(
            changed_fields & {"in_world_time", "time_of_day", "day_of_week"}
        )
        world_time_kwargs = {
            "world_time_day_index": canonical_world_time.day_index,
        }
        if display_fields_changed:
            world_time_kwargs.update(
                {
                    "world_time_day_label": canonical_world_time.day_label,
                    "world_time_phase": canonical_world_time.phase,
                    "world_time_clock_minutes": (
                        canonical_world_time.clock_minutes
                        if canonical_world_time.clock_minutes is not None
                        else (
                            existing.world_time_clock_minutes
                            if existing is not None
                            else None
                        )
                    ),
                    "world_time_period_label": (
                        canonical_world_time.period_label
                        or (
                            existing.world_time_period_label
                            if existing is not None
                            else ""
                        )
                    ),
                    "world_time_source_message_id": (
                        canonical_world_time.source_message_id
                    ),
                    "world_time_confidence": canonical_world_time.confidence,
                }
            )
    saved = repositories.upsert_scene_snapshot(
        save_id=cast(str, values["save_id"]),
        current_location_id=cast(str | None, values["current_location_id"]),
        situation=cast(str, values["situation"]),
        objective=cast(str, values["objective"]),
        in_world_time=cast(str, values["in_world_time"]),
        time_of_day=cast(str, values["time_of_day"]),
        day_of_week=cast(str, values["day_of_week"]),
        world_day_index=cast(int | None, values["world_day_index"]),
        weather=cast(str, values["weather"]),
        mood=cast(str, values["mood"]),
        nearby_objects=cast(list[str], values["nearby_objects"]),
        hazards=cast(list[str], values["hazards"]),
        present_character_ids=cast(list[str], values["present_character_ids"]),
        source_message_id=cast(str | None, values["source_message_id"]),
        locked_fields=cast(list[str], values["locked_fields"]),
        snapshot_id=cast(str | None, values["snapshot_id"]),
        **world_time_kwargs,
    )
    if values.get("world_time_changed"):
        from bragi.services.time_loop_time_policy import TimeLoopTimePolicy

        policy = TimeLoopTimePolicy(repositories, save_id=saved.save_id)
        if existing is not None:
            policy.ensure_baseline(existing)
        policy.ensure_baseline(saved)
        policy.sync_current(saved, transition="manual_scene_update")
    return saved


def _location_update_values(*, row: object, model: WorldDataModel) -> LocationRecord:
    row = _location_edit_row(row)
    current = _location_from_model(model, row.location_id)
    changed = _changed_fields(
        _location_row(current),
        row,
        (
            "name",
            "aliases_text",
            "description",
            "visual_description",
            "parent_location_id",
            "connections_text",
            "status",
            "hazards_text",
        ),
    )
    return replace(
        current,
        name=row.name.strip(),
        aliases=_csv(row.aliases_text),
        description=row.description.strip(),
        visual_description=row.visual_description.strip(),
        parent_location_id=_none_if_blank(row.parent_location_id),
        connections=_csv(row.connections_text),
        status=row.status.strip(),
        hazards=_csv(row.hazards_text),
        source_message_id=None,
        locked_fields=_locked(row.locked_fields, changed),
    )


def _character_update_values(*, row: object, model: WorldDataModel) -> CharacterRecord:
    if not isinstance(row, WorldDataCharacterRow):
        raise TypeError(f"Unsupported character edit row: {type(row).__name__}")
    current = _character_from_model(model, row.character_id)
    relationships = _json_object(row.relationships_json, "relationships")
    changed = _changed_fields(
        _character_row(current),
        row,
        (
            "name",
            "aliases_text",
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
            "texting_style",
            "relationships_json",
            "goals",
            "motivations",
            "current_intent",
            "boundaries",
            "attitude_toward_player",
            "cooperation_conditions",
            "status",
            "location_id",
            "private_notes",
        ),
    )
    return replace(
        current,
        name=row.name.strip(),
        aliases=_csv(row.aliases_text),
        role=row.role.strip(),
        age=row.age.strip(),
        known_state=_row_history(row),
        history=_row_history(row),
        met=bool(row.met),
        appearance=row.appearance.strip(),
        visual_notes=row.visual_notes.strip(),
        current_clothing=row.current_clothing.strip(),
        personality=row.personality.strip(),
        voice=row.voice.strip(),
        texting_style=row.texting_style.strip(),
        relationships=relationships,
        goals=row.goals.strip(),
        motivations=row.motivations.strip(),
        current_intent=row.current_intent.strip(),
        boundaries=row.boundaries.strip(),
        attitude_toward_player=row.attitude_toward_player.strip(),
        cooperation_conditions=row.cooperation_conditions.strip(),
        status=row.status.strip(),
        location_id=_none_if_blank(row.location_id),
        private_notes=row.private_notes.strip(),
        source_message_id=None,
        protected_from_maintenance=current.protected_from_maintenance,
        is_player_character=bool(row.is_player_character),
        content_rating=(
            "unclassified" if changed else current.content_rating
        ),
        locked_fields=merge_character_locked_fields(current.locked_fields, changed),
    )


def _row_history(row: WorldDataCharacterRow) -> str:
    return (row.known_state or row.history).strip()


def _thread_update_values(*, row: object, model: WorldDataModel) -> ActiveThreadRecord:
    row = _thread_edit_row(row)
    current = _thread_from_model(model, row.thread_id)
    changed = _changed_fields(
        _thread_row(current),
        row,
        (
            "title",
            "description",
            "status",
            "priority",
            "visibility",
            "related_entities_text",
        ),
    )
    changed = tuple(
        "related_entities" if field == "related_entities_text" else field
        for field in changed
    )
    return replace(
        current,
        title=row.title.strip(),
        description=row.description.strip(),
        status=row.status.strip(),
        priority=int(row.priority),
        visibility=row.visibility.strip(),
        related_entities=_csv(row.related_entities_text),
        source_message_id=None,
        locked_fields=_locked(row.locked_fields, changed),
    )


def _validated_state_row(row: object) -> dict[str, object]:
    row_data = _state_edit_values(row)
    try:
        loaded = json.loads(row_data.value_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid world-state JSON for key '{row_data.key}'") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"World-state JSON for '{row_data.key}' must be an object")
    return cast(dict[str, object], loaded)


def _tags(value: str) -> list[str]:
    return [tag.strip() for tag in value.split(",") if tag.strip()]


def _find_state(
    rows: tuple[WorldDataStateRow, ...],
    key: str,
) -> WorldDataStateRow | None:
    for row in rows:
        if row.key == key:
            return row
    return None


def _validate_world_data_ids(
    *,
    edits: WorldDataEdits,
    model: WorldDataModel,
    repositories: PersistenceRepositories,
    save_id: str,
) -> None:
    state_ids = {row.row_id for row in model.state_rows}
    state_keys = {row.key for row in model.state_rows}
    message_ids = _all_message_ids_for_save(repositories, save_id)
    for row in edits.world_state:
        row_data = _state_edit_values(row)
        row_id = _world_state_row_id(row).strip()
        original_key = (row_data.original_key or "").strip()
        if row_id:
            if row_id not in state_ids:
                raise ValueError("World-state edit does not belong to the active save")
        elif original_key and original_key in state_keys:
            raise ValueError("World-state edit does not identify the active save row")
        if (
            row_data.source_message_id is not None
            and row_data.source_message_id not in message_ids
        ):
            raise ValueError(
                "World-state source message does not belong to the active save"
            )

    memory_ids = {row.memory_id for row in model.memory_rows}
    for row in edits.memories:
        memory_id = _memory_edit_values(row).memory_id
        if memory_id and memory_id not in memory_ids:
            raise ValueError("Memory edit does not belong to the active save")

    summary_ids = {row.summary_id for row in model.summary_rows}
    for row in edits.summaries:
        summary_id = _summary_edit_values(row).summary_id
        if summary_id not in summary_ids:
            raise ValueError("Summary edit does not belong to the active save")

    location_ids = {row.location_id for row in model.locations}
    character_ids = {row.character_id for row in model.characters}
    thread_ids = {row.thread_id for row in model.threads}
    link_ids = {row.link_id for row in model.links}
    suggestion_ids = {row.suggestion_id for row in model.suggestions}
    suggestion_group_ids = {row.group_id for row in model.suggestion_groups}
    if edits.loss_conditions:
        raise ValueError("Loss conditions are deprecated and cannot be edited")

    if edits.scene is not None:
        if not isinstance(edits.scene, WorldDataSceneRow):
            raise TypeError("Unsupported scene edit row")
        if (
            edits.scene.current_location_id
            and edits.scene.current_location_id not in location_ids
        ):
            raise ValueError(
                "Scene current location does not belong to the active save"
            )
        for character_id in _csv(edits.scene.present_character_ids_text):
            if character_id not in character_ids:
                raise ValueError(
                    "Scene present character does not belong to the active save"
                )
        if (
            edits.scene.source_message_id
            and edits.scene.source_message_id not in message_ids
        ):
            raise ValueError("Scene source message does not belong to the active save")

    for row in edits.locations:
        row = _location_edit_row(row)
        if row.location_id and row.location_id not in location_ids:
            raise ValueError("Location edit does not belong to the active save")
        if row.parent_location_id and row.parent_location_id not in location_ids:
            raise ValueError("Location parent does not belong to the active save")
        if row.source_message_id and row.source_message_id not in message_ids:
            raise ValueError(
                "Location source message does not belong to the active save"
            )

    for row in edits.characters:
        if (
            not isinstance(row, WorldDataCharacterRow)
            or row.character_id not in character_ids
        ):
            raise ValueError("Character edit does not belong to the active save")
        if row.location_id and row.location_id not in location_ids:
            raise ValueError("Character location does not belong to the active save")
        if row.source_message_id and row.source_message_id not in message_ids:
            raise ValueError(
                "Character source message does not belong to the active save"
            )

    for row in edits.threads:
        row = _thread_edit_row(row)
        if row.thread_id and row.thread_id not in thread_ids:
            raise ValueError("Thread edit does not belong to the active save")
        if row.source_message_id and row.source_message_id not in message_ids:
            raise ValueError("Thread source message does not belong to the active save")

    for row in edits.links:
        link = _link_edit_values(row)
        if link.deleted:
            if link.link_id not in link_ids:
                raise ValueError("Entity link does not belong to the active save")
            continue
        if link.link_id and link.link_id not in link_ids:
            raise ValueError("Entity link does not belong to the active save")
        _validate_link_endpoint(
            kind="entity",
            endpoint_type=link.entity_type,
            endpoint_id=link.entity_id,
            location_ids=location_ids,
            character_ids=character_ids,
            thread_ids=thread_ids,
            state_ids=state_ids,
            memory_ids=memory_ids,
            summary_ids=summary_ids,
            model=model,
        )
        _validate_link_endpoint(
            kind="target",
            endpoint_type=link.target_type,
            endpoint_id=link.target_id,
            location_ids=location_ids,
            character_ids=character_ids,
            thread_ids=thread_ids,
            state_ids=state_ids,
            memory_ids=memory_ids,
            summary_ids=summary_ids,
            model=model,
        )

    for row in edits.suggestions:
        suggestion_id = _suggestion_row_id(row)
        if suggestion_id not in suggestion_ids:
            raise ValueError("Suggestion does not belong to the active save")
        if _suggestion_action(row) not in {"", "apply", "reject"}:
            raise ValueError("Unsupported suggestion action")

    for row in edits.suggestion_groups:
        group_id = _suggestion_group_row_id(row)
        if group_id not in suggestion_group_ids:
            raise ValueError("Suggestion group does not belong to the active save")
        if _suggestion_group_action(row) not in {"", "apply", "reject", "dismiss"}:
            raise ValueError("Unsupported suggestion group action")


def _validate_world_state_key_collisions(
    *,
    edits: WorldDataEdits,
    model: WorldDataModel,
) -> None:
    active_by_key = {row.key: row for row in model.state_rows}
    target_keys: set[str] = set()
    for row in edits.world_state:
        row_data = _state_edit_values(row)
        target_key = row_data.key.strip()
        if row_data.archived or not target_key:
            continue
        row_id = _world_state_row_id(row)
        existing = active_by_key.get(target_key)
        if existing is not None and existing.row_id != row_id:
            raise ValueError(
                f"World-state key already exists in this save: {target_key}"
            )
        if target_key in target_keys:
            raise ValueError(f"World-state key appears more than once: {target_key}")
        target_keys.add(target_key)


def _validate_location_archive_allowed(
    *, model: WorldDataModel, location_id: str
) -> None:
    if model.scene is not None and model.scene.current_location_id == location_id:
        raise ValueError("Cannot archive the current scene location")
    for character in model.characters:
        if character.location_id == location_id:
            raise ValueError("Cannot archive a location used by a character")
    for location in model.locations:
        if (
            location.location_id != location_id
            and location.parent_location_id == location_id
        ):
            raise ValueError("Cannot archive a location used as a parent")


def _all_message_ids_for_save(
    repositories: PersistenceRepositories,
    save_id: str,
) -> set[str]:
    return repositories.list_message_ids(save_id)


def _world_state_row_id(row: object) -> str:
    if isinstance(row, WorldDataStateRow):
        return row.row_id
    if isinstance(row, WorldStateEdit):
        return row.state_id
    raise TypeError(f"Unsupported world-state edit row: {type(row).__name__}")


def _location_from_model(model: WorldDataModel, location_id: str) -> LocationRecord:
    record = model.save_id and model.save_id
    del record
    for location in model.locations:
        if location.location_id == location_id:
            return LocationRecord(
                id=location.location_id,
                save_id=cast(str, model.save_id),
                name=location.name,
                aliases=_csv(location.aliases_text),
                description=location.description,
                visual_description=location.visual_description,
                parent_location_id=location.parent_location_id,
                connections=_csv(location.connections_text),
                status=location.status,
                hazards=_csv(location.hazards_text),
                source_message_id=location.source_message_id,
                locked_fields=list(location.locked_fields),
            )
    raise ValueError("Location edit does not belong to the active save")


def _character_from_model(model: WorldDataModel, character_id: str) -> CharacterRecord:
    for character in model.characters:
        if character.character_id == character_id:
            return CharacterRecord(
                id=character.character_id,
                save_id=cast(str, model.save_id),
                name=character.name,
                aliases=_csv(character.aliases_text),
                role=character.role,
                age=character.age,
                known_state=character.known_state,
                met=character.met,
                appearance=character.appearance,
                visual_notes=character.visual_notes,
                current_clothing=character.current_clothing,
                personality=character.personality,
                voice=character.voice,
                texting_style=character.texting_style,
                relationships=_json_object(
                    character.relationships_json, "relationships"
                ),
                goals=character.goals,
                motivations=character.motivations,
                current_intent=character.current_intent,
                boundaries=character.boundaries,
                attitude_toward_player=character.attitude_toward_player,
                cooperation_conditions=character.cooperation_conditions,
                status=character.status,
                location_id=character.location_id,
                private_notes=character.private_notes,
                source_message_id=character.source_message_id,
                locked_fields=list(character.locked_fields),
                protected_from_maintenance=character.protected_from_maintenance,
                is_player_character=character.is_player_character,
            )
    raise ValueError("Character edit does not belong to the active save")


def _thread_from_model(model: WorldDataModel, thread_id: str) -> ActiveThreadRecord:
    for thread in model.threads:
        if thread.thread_id == thread_id:
            return ActiveThreadRecord(
                id=thread.thread_id,
                save_id=cast(str, model.save_id),
                title=thread.title,
                description=thread.description,
                status=thread.status,
                priority=thread.priority,
                visibility=thread.visibility,
                related_entities=_csv(thread.related_entities_text),
                source_message_id=thread.source_message_id,
                locked_fields=list(thread.locked_fields),
            )
    raise ValueError("Thread edit does not belong to the active save")


def _memory_from_model(model: WorldDataModel, memory_id: str) -> MemoryRecord:
    for memory in model.memories:
        if memory.memory_id == memory_id:
            return MemoryRecord(
                id=memory.memory_id,
                save_id=cast(str, model.save_id),
                body=memory.body,
                tags=_csv(memory.tags_text),
                importance=memory.importance,
                source_message_id=memory.source_message_id,
                source_message_ids=list(memory.source_message_ids),
            )
    raise ValueError("Memory edit does not belong to the active save")


def _summary_from_model(model: WorldDataModel, summary_id: str) -> SummaryRecord:
    for summary in model.summaries:
        if summary.summary_id == summary_id:
            return SummaryRecord(
                id=summary.summary_id,
                save_id=cast(str, model.save_id),
                body=summary.body,
                provider=summary.provider,
                model=summary.model,
                covers_message_start_id=summary.covers_message_start_id,
                covers_message_end_id=summary.covers_message_end_id,
            )
    raise ValueError("Summary edit does not belong to the active save")


def _suggestion_from_model(
    model: WorldDataModel,
    suggestion_id: str,
) -> WorldDataSuggestionRow:
    for suggestion in model.suggestions:
        if suggestion.suggestion_id == suggestion_id:
            return suggestion
    raise ValueError("Suggestion does not belong to the active save")


def _suggestion_group_from_model(
    model: WorldDataModel,
    group_id: str,
) -> WorldDataSuggestionGroupRow:
    for group in model.suggestion_groups:
        if group.group_id == group_id:
            return group
    raise ValueError("Suggestion group does not belong to the active save")


def _suggestion_row_id(row: object) -> str:
    if not isinstance(row, WorldDataSuggestionRow):
        raise TypeError(f"Unsupported suggestion edit row: {type(row).__name__}")
    return row.suggestion_id


def _suggestion_group_row_id(row: object) -> str:
    if not isinstance(row, WorldDataSuggestionGroupRow):
        raise TypeError(
            f"Unsupported suggestion group edit row: {type(row).__name__}"
        )
    return row.group_id


def _suggestion_action(row: object) -> str:
    if not isinstance(row, WorldDataSuggestionRow):
        raise TypeError(f"Unsupported suggestion edit row: {type(row).__name__}")
    return row.action.strip().casefold()


def _suggestion_group_action(row: object) -> str:
    if not isinstance(row, WorldDataSuggestionGroupRow):
        raise TypeError(
            f"Unsupported suggestion group edit row: {type(row).__name__}"
        )
    return row.action.strip().casefold()


def _apply_suggestion_batch(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    model: WorldDataModel,
    suggestions: tuple[WorldDataSuggestionRow, ...],
    operation: str,
    reason: str = "",
) -> None:
    if not suggestions:
        return
    _validate_suggestion_batch(suggestions)
    suggestion = suggestions[0]
    value = _json_value(suggestion.proposed_value_json, suggestion.field_path)
    before = _suggestion_before_value(model, suggestion)
    _apply_suggestion_value(
        repositories=repositories,
        save_id=save_id,
        model=model,
        suggestion=suggestion,
        value=value,
    )
    repositories.update_context_update_suggestion_status(
        suggestion.suggestion_id,
        status="applied",
    )
    if len(suggestions) > 1:
        repositories.update_context_update_suggestion_statuses(
            [member.suggestion_id for member in suggestions[1:]],
            status="applied",
        )
    for member in suggestions:
        repositories.add_context_update_audit(
            save_id=save_id,
            suggestion_id=member.suggestion_id,
            operation=operation,
            entity_type=member.entity_type,
            entity_id=member.entity_id,
            field_path=member.field_path,
            before=before,
            after=value,
            reason=reason or member.reason,
            confidence=member.confidence,
            source_message_ids=_csv(member.source_message_ids_text),
        )


def _reject_suggestion_batch(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    model: WorldDataModel,
    suggestions: tuple[WorldDataSuggestionRow, ...],
    operation: str,
    reason: str,
) -> None:
    if not suggestions:
        return
    repositories.update_context_update_suggestion_statuses(
        [suggestion.suggestion_id for suggestion in suggestions],
        status="rejected",
    )
    for suggestion in suggestions:
        try:
            before = _suggestion_before_value(model, suggestion)
        except ValueError:
            # A suggestion can outlive the entity it proposed to change. Rejection
            # must still be terminal; the absent value is useful audit information.
            before = None
        repositories.add_context_update_audit(
            save_id=save_id,
            suggestion_id=suggestion.suggestion_id,
            operation=operation,
            entity_type=suggestion.entity_type,
            entity_id=suggestion.entity_id,
            field_path=suggestion.field_path,
            before=before,
            after=_json_value(suggestion.proposed_value_json, suggestion.field_path),
            reason=reason or suggestion.reason,
            confidence=suggestion.confidence,
            source_message_ids=_csv(suggestion.source_message_ids_text),
        )


def _validate_suggestion_batch(
    suggestions: tuple[WorldDataSuggestionRow, ...],
) -> None:
    first = suggestions[0]
    key = (
        first.update_type,
        first.entity_type,
        first.entity_id,
        first.field_path,
        first.proposed_value_json,
    )
    for suggestion in suggestions[1:]:
        if (
            suggestion.update_type,
            suggestion.entity_type,
            suggestion.entity_id,
            suggestion.field_path,
            suggestion.proposed_value_json,
        ) != key:
            raise ValueError("Suggestion batch targets do not match")
    if any(suggestion.status != "pending" for suggestion in suggestions):
        raise ValueError("Only pending suggestions can be applied")


def _apply_suggestion_group(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    model: WorldDataModel,
    group: WorldDataSuggestionGroupRow,
) -> None:
    members = _suggestion_group_members(model, group)
    first = members[0]
    value = _json_value(first.proposed_value_json, first.field_path)
    before = _suggestion_before_value(model, first)
    _apply_suggestion_value(
        repositories=repositories,
        save_id=save_id,
        model=model,
        suggestion=first,
        value=value,
    )
    repositories.update_context_update_suggestion_statuses(
        list(group.suggestion_ids),
        status="applied",
    )
    for member in members:
        repositories.add_context_update_audit(
            save_id=save_id,
            suggestion_id=member.suggestion_id,
            operation="manual_suggestion_group_apply",
            entity_type=member.entity_type,
            entity_id=member.entity_id,
            field_path=member.field_path,
            before=before,
            after=value,
            reason=member.reason,
            confidence=member.confidence,
            source_message_ids=_csv(member.source_message_ids_text),
        )


def _resolve_suggestion_group(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    model: WorldDataModel,
    group: WorldDataSuggestionGroupRow,
    status: str,
    operation: str,
) -> None:
    members = _suggestion_group_members(model, group)
    repositories.update_context_update_suggestion_statuses(
        list(group.suggestion_ids),
        status=status,
    )
    for member in members:
        repositories.add_context_update_audit(
            save_id=save_id,
            suggestion_id=member.suggestion_id,
            operation=operation,
            entity_type=member.entity_type,
            entity_id=member.entity_id,
            field_path=member.field_path,
            before=None,
            after=_json_value(member.proposed_value_json, member.field_path),
            reason=member.reason,
            confidence=member.confidence,
            source_message_ids=_csv(member.source_message_ids_text),
        )


def _suggestion_group_members(
    model: WorldDataModel,
    group: WorldDataSuggestionGroupRow,
) -> tuple[WorldDataSuggestionRow, ...]:
    suggestions = {
        suggestion.suggestion_id: suggestion for suggestion in model.suggestions
    }
    members = tuple(
        suggestions[suggestion_id]
        for suggestion_id in group.suggestion_ids
        if suggestion_id in suggestions
    )
    if not members:
        raise ValueError("Suggestion group has no pending suggestions")
    return members


def _apply_suggestion_value(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    model: WorldDataModel,
    suggestion: WorldDataSuggestionRow,
    value: object,
) -> None:
    field = suggestion.field_path
    if (
        suggestion.update_type in {"archive", "delete"}
        and suggestion.entity_type != "world_state"
    ):
        _apply_destructive_suggestion(
            repositories=repositories,
            save_id=save_id,
            model=model,
            suggestion=suggestion,
        )
        return
    if suggestion.entity_type == "scene_snapshot":
        current = model.scene
        if current is None:
            raise ValueError("Scene suggestion cannot apply without a scene snapshot")
        kwargs = _scene_update_values(row=current, model=model, save_id=save_id)
        if field not in kwargs:
            raise ValueError(f"Unsupported scene suggestion field: {field}")
        kwargs[field] = _coerce_scene_value(field, value)
        if field in SCENE_WORLD_TIME_FIELDS:
            kwargs["world_time_changed"] = True
            kwargs["world_time_changed_fields"] = (field,)
            source_message_ids = _csv(suggestion.source_message_ids_text)
            kwargs["world_time_source_message_id"] = (
                source_message_ids[-1] if source_message_ids else None
            )
            kwargs["world_time_confidence"] = suggestion.confidence
        kwargs["locked_fields"] = _locked(
            cast(list[str], kwargs["locked_fields"]), (field,)
        )
        _upsert_scene_snapshot(repositories=repositories, values=kwargs)
        return
    if suggestion.entity_type == "location" and suggestion.entity_id:
        location_record = _location_from_model(model, suggestion.entity_id)
        repositories.update_location(
            _replace_location_field(location_record, field, value)
        )
        return
    if suggestion.entity_type == "character" and suggestion.entity_id:
        character_record = _character_from_model(model, suggestion.entity_id)
        repositories.update_character(
            _replace_character_field(character_record, field, value)
        )
        return
    if suggestion.entity_type == "active_thread" and suggestion.entity_id:
        thread_record = _thread_from_model(model, suggestion.entity_id)
        repositories.update_active_thread(
            _replace_thread_field(thread_record, field, value)
        )
        return
    if suggestion.entity_type == "memory" and suggestion.entity_id:
        memory_record = _memory_from_model(model, suggestion.entity_id)
        if field not in {"body", "tags", "importance"}:
            raise ValueError(f"Unsupported memory suggestion field: {field}")
        if field == "body" and (not isinstance(value, str) or not value.strip()):
            raise ValueError("Memory suggestion body is required")
        repositories.update_memory(
            memory_id=memory_record.id,
            body=(
                str(value).strip() if field == "body" else memory_record.body
            ),
            tags=(
                _string_list_value(value, "Memory tags")
                if field == "tags"
                else memory_record.tags
            ),
            importance=(
                _memory_importance_value(value)
                if field == "importance"
                else memory_record.importance
            ),
        )
        return
    if suggestion.entity_type == "summary" and suggestion.entity_id:
        summary_record = _summary_from_model(model, suggestion.entity_id)
        if field != "body":
            raise ValueError(f"Unsupported summary suggestion field: {field}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Summary suggestion body is required")
        repositories.update_summary(
            summary_id=summary_record.id,
            body=value.strip(),
        )
        return
    if suggestion.entity_type == "memory" and suggestion.update_type == "create":
        if not isinstance(value, dict):
            raise ValueError("Memory suggestion value must be a JSON object")
        body = str(value.get("body", "")).strip()
        if not body:
            raise ValueError("Memory suggestion body is required")
        importance = value.get("importance", 1.0)
        if isinstance(importance, bool) or not isinstance(importance, int | float):
            raise ValueError("Memory importance must be numeric")
        source_message_id = value.get("source_message_id")
        raw_source_message_ids = value.get("source_message_ids")
        raw_source_observation_ids = value.get("source_observation_ids")
        memory_source_message_ids = (
            _string_list_value(raw_source_message_ids, "Memory source message IDs")
            if isinstance(raw_source_message_ids, list)
            else None
        )
        memory_source_observation_ids = (
            _string_list_value(
                raw_source_observation_ids,
                "Memory source observation IDs",
            )
            if isinstance(raw_source_observation_ids, list)
            else None
        )
        tags = _string_list_value(value.get("tags", []), "Memory tags")
        fingerprint = canonical_claim_fingerprint(body)
        repositories.begin_immediate_transaction()
        try:
            existing = next(
                (
                    memory
                    for memory in repositories.list_memories(save_id)
                    if memory.claim_fingerprint == fingerprint
                ),
                None,
            )
            if existing is None:
                repositories.add_memory(
                    save_id=save_id,
                    body=body,
                    tags=tags,
                    importance=float(importance),
                    source_message_id=(
                        source_message_id
                        if isinstance(source_message_id, str)
                        else None
                    ),
                    source_message_ids=memory_source_message_ids,
                    source_observation_ids=memory_source_observation_ids,
                )
            else:
                repositories.update_memory(
                    memory_id=existing.id,
                    body=existing.body,
                    tags=list(dict.fromkeys((*existing.tags, *tags))),
                    importance=max(existing.importance, float(importance)),
                    source_message_ids=list(
                        dict.fromkeys(
                            (
                                *existing.source_message_ids,
                                *(memory_source_message_ids or ()),
                            )
                        )
                    ),
                    source_observation_ids=list(
                        dict.fromkeys(
                            (
                                *existing.source_observation_ids,
                                *(memory_source_observation_ids or ()),
                            )
                        )
                    ),
                )
            repositories.commit_transaction()
        except Exception:
            repositories.rollback_transaction()
            raise
        return
    if suggestion.entity_type == "character" and suggestion.update_type == "create":
        if not isinstance(value, dict):
            raise ValueError("Character suggestion value must be a JSON object")
        name = str(value.get("name", "")).strip()
        if not name:
            raise ValueError("Character suggestion name is required")
        source_message_id = value.get("source_message_id")
        location_id = value.get("location_id")
        repositories.add_character(
            save_id=save_id,
            name=name,
            aliases=_string_list_value(value.get("aliases", []), "Character aliases"),
            role=str(value.get("role", "")).strip(),
            age=str(value.get("age", "")).strip(),
            known_state=str(value.get("known_state", "")).strip(),
            met=bool(value.get("met", False)),
            appearance=str(value.get("appearance", "")).strip(),
            visual_notes=str(value.get("visual_notes", "")).strip(),
            current_clothing=str(value.get("current_clothing", "")).strip(),
            personality=str(value.get("personality", "")).strip(),
            voice=str(value.get("voice", "")).strip(),
            texting_style=str(value.get("texting_style", "")).strip(),
            relationships=(
                cast(dict[str, object], value.get("relationships"))
                if isinstance(value.get("relationships"), dict)
                else {}
            ),
            goals=str(value.get("goals", "")).strip(),
            motivations=str(value.get("motivations", "")).strip(),
            current_intent=str(value.get("current_intent", "")).strip(),
            boundaries=str(value.get("boundaries", "")).strip(),
            attitude_toward_player=str(
                value.get("attitude_toward_player", "")
            ).strip(),
            cooperation_conditions=str(
                value.get("cooperation_conditions", "")
            ).strip(),
            status=str(value.get("status", "")).strip(),
            location_id=location_id if isinstance(location_id, str) else None,
            private_notes=str(value.get("private_notes", "")).strip(),
            source_message_id=(
                source_message_id if isinstance(source_message_id, str) else None
            ),
        )
        return
    if suggestion.entity_type == "world_state":
        _apply_world_state_suggestion(
            repositories=repositories,
            save_id=save_id,
            model=model,
            suggestion=suggestion,
            value=value,
        )
        return
    raise ValueError("Unsupported suggestion target")


def _apply_destructive_suggestion(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    model: WorldDataModel,
    suggestion: WorldDataSuggestionRow,
) -> None:
    if not suggestion.entity_id:
        raise ValueError("Destructive suggestion target id is required")
    if suggestion.entity_type == "entity_link":
        if suggestion.update_type != "delete":
            raise ValueError("Entity-link suggestions only support delete")
        if not any(link.link_id == suggestion.entity_id for link in model.links):
            raise ValueError("Entity-link suggestion target is no longer active")
        repositories.delete_entity_link(suggestion.entity_id)
        return
    if suggestion.update_type != "archive":
        raise ValueError("Only entity links may be deleted by suggestions")
    repositories.delete_entity_links_for_endpoint(
        save_id=save_id,
        entity_type=suggestion.entity_type,
        entity_id=suggestion.entity_id,
    )
    if suggestion.entity_type == "memory":
        _memory_from_model(model, suggestion.entity_id)
        repositories.archive_memory(suggestion.entity_id)
        return
    if suggestion.entity_type == "summary":
        _summary_from_model(model, suggestion.entity_id)
        repositories.archive_summary(suggestion.entity_id)
        return
    if suggestion.entity_type == "location":
        _location_from_model(model, suggestion.entity_id)
        _validate_location_archive_allowed(
            model=model,
            location_id=suggestion.entity_id,
        )
        repositories.archive_location(suggestion.entity_id)
        return
    if suggestion.entity_type == "character":
        character = _character_from_model(model, suggestion.entity_id)
        if character.protected_from_maintenance:
            raise ValueError(
                "Character is protected from maintenance and cannot be archived"
            )
        repositories.archive_character(suggestion.entity_id)
        return
    if suggestion.entity_type == "active_thread":
        _thread_from_model(model, suggestion.entity_id)
        repositories.archive_active_thread(suggestion.entity_id)
        return
    raise ValueError("Unsupported destructive suggestion target")


def _apply_world_state_suggestion(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    model: WorldDataModel,
    suggestion: WorldDataSuggestionRow,
    value: object,
) -> None:
    if not isinstance(value, dict):
        raise ValueError("World-state suggestion value must be a JSON object")
    operation = str(value.get("operation", suggestion.update_type)).strip()
    key = str(value.get("key", suggestion.field_path)).strip()
    if not key:
        raise ValueError("World-state suggestion key is required")
    before = _find_state(model.state_rows, key)
    before_json = before.value_json if before else None
    before_record = next(
        (
            record
            for record in repositories.list_world_state(save_id)
            if record.key == key
        ),
        None,
    )
    source_message_id = value.get("source_message_id")
    source_id = source_message_id if isinstance(source_message_id, str) else None
    if operation in {"delete", "remove"}:
        if key == "loop.current":
            raise ValueError("The time-loop clock state cannot be deleted")
        if before is not None:
            repositories.delete_entity_links_for_endpoint(
                save_id=save_id,
                entity_type="world_state",
                entity_id=before.state_id,
            )
        preserve_replaced_world_state_memory(
            repositories=repositories,
            save_id=save_id,
            before=before_record,
            after_value=None,
            source_message_id=source_id,
        )
        repositories.archive_world_state(save_id=save_id, key=key)
        repositories.add_state_change(
            save_id=save_id,
            operation="manual_suggestion_apply",
            state_key=key,
            before_json=before_json,
            after_json=None,
            source_message_id=source_id,
        )
        return
    if operation != "upsert":
        raise ValueError(f"Unsupported world-state suggestion operation: {operation}")
    state_value = value.get("value")
    if not isinstance(state_value, dict):
        raise ValueError("World-state suggestion value must contain an object value")
    confidence = value.get("confidence", 1.0)
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        raise ValueError("World-state confidence must be numeric")
    preserved_value = cast(dict[str, object], state_value)
    if key == "loop.current" and before_record is not None:
        preserved_value = _merge_loop_current_summary(
            before_record.value,
            preserved_value,
        )
    preserve_replaced_world_state_memory(
        repositories=repositories,
        save_id=save_id,
        before=before_record,
        after_value=preserved_value,
        source_message_id=source_id,
    )
    repositories.upsert_world_state(
        save_id=save_id,
        key=key,
        value=preserved_value,
        category=str(value.get("category", "")).strip(),
        confidence=float(confidence),
        source_message_id=source_id,
    )
    repositories.add_state_change(
        save_id=save_id,
        operation="manual_suggestion_apply",
        state_key=key,
        before_json=before_json,
        after_json=_dump_json(preserved_value),
        source_message_id=source_id,
    )


def _merge_loop_current_summary(
    existing: dict[str, object],
    proposed: dict[str, object],
) -> dict[str, object]:
    """Keep policy-owned loop metadata when a suggestion only updates prose."""
    summary = proposed.get("summary")
    if not isinstance(summary, str):
        return existing
    merged = dict(existing)
    merged["summary"] = summary
    return merged


def _replace_location_field(
    record: LocationRecord,
    field: str,
    value: object,
) -> LocationRecord:
    locked = _locked(record.locked_fields, (field,))
    if field in {
        "name",
        "description",
        "visual_description",
        "parent_location_id",
        "status",
    }:
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Location field {field} must be a string")
        if field == "name":
            return replace(record, name=value or "", locked_fields=locked)
        if field == "description":
            return replace(record, description=value or "", locked_fields=locked)
        if field == "visual_description":
            return replace(record, visual_description=value or "", locked_fields=locked)
        if field == "parent_location_id":
            return replace(record, parent_location_id=value, locked_fields=locked)
        if field == "status":
            return replace(record, status=value or "", locked_fields=locked)
    if field in {"aliases", "connections", "hazards"}:
        string_list = _string_list_value(value, f"Location field {field}")
        if field == "aliases":
            return replace(record, aliases=string_list, locked_fields=locked)
        if field == "connections":
            return replace(record, connections=string_list, locked_fields=locked)
        if field == "hazards":
            return replace(record, hazards=string_list, locked_fields=locked)
    raise ValueError(f"Unsupported suggestion field: {field}")


def _replace_character_field(
    record: CharacterRecord,
    field: str,
    value: object,
) -> CharacterRecord:
    locked = merge_character_locked_fields(record.locked_fields, (field,))
    if field in {
        "name",
        "role",
        "age",
        "known_state",
        "history",
        "appearance",
        "visual_notes",
        "current_clothing",
        "personality",
        "voice",
        "texting_style",
        "goals",
        "motivations",
        "current_intent",
        "boundaries",
        "attitude_toward_player",
        "cooperation_conditions",
        "status",
        "location_id",
        "private_notes",
    }:
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Character field {field} must be a string")
        text_value = value or ""
        if field == "name":
            return replace(record, name=text_value, locked_fields=locked)
        if field == "role":
            return replace(record, role=text_value, locked_fields=locked)
        if field == "age":
            return replace(record, age=text_value, locked_fields=locked)
        if field in {"known_state", "history"}:
            return replace(
                record,
                known_state=text_value,
                history=text_value,
                locked_fields=locked,
            )
        if field == "appearance":
            return replace(record, appearance=text_value, locked_fields=locked)
        if field == "visual_notes":
            return replace(record, visual_notes=text_value, locked_fields=locked)
        if field == "current_clothing":
            return replace(record, current_clothing=text_value, locked_fields=locked)
        if field == "personality":
            return replace(record, personality=text_value, locked_fields=locked)
        if field == "voice":
            return replace(record, voice=text_value, locked_fields=locked)
        if field == "texting_style":
            return replace(record, texting_style=text_value, locked_fields=locked)
        if field == "goals":
            return replace(record, goals=text_value, locked_fields=locked)
        if field == "motivations":
            return replace(record, motivations=text_value, locked_fields=locked)
        if field == "current_intent":
            return replace(record, current_intent=text_value, locked_fields=locked)
        if field == "boundaries":
            return replace(record, boundaries=text_value, locked_fields=locked)
        if field == "attitude_toward_player":
            return replace(
                record,
                attitude_toward_player=text_value,
                locked_fields=locked,
            )
        if field == "cooperation_conditions":
            return replace(
                record,
                cooperation_conditions=text_value,
                locked_fields=locked,
            )
        if field == "status":
            return replace(record, status=text_value, locked_fields=locked)
        if field == "location_id":
            return replace(record, location_id=value, locked_fields=locked)
        if field == "private_notes":
            return replace(record, private_notes=text_value, locked_fields=locked)
    if field == "met":
        if not isinstance(value, bool):
            raise ValueError("Character field met must be a boolean")
        return replace(record, met=value, locked_fields=locked)
    if field == "relationships":
        if not isinstance(value, dict):
            raise ValueError("Character relationships must be a JSON object")
        return replace(
            record,
            relationships=cast(dict[str, object], value),
            locked_fields=locked,
        )
    if field == "aliases":
        return replace(
            record,
            aliases=_string_list_value(value, "Character aliases"),
            locked_fields=locked,
        )
    raise ValueError(f"Unsupported suggestion field: {field}")


def _replace_thread_field(
    record: ActiveThreadRecord,
    field: str,
    value: object,
) -> ActiveThreadRecord:
    locked = _locked(record.locked_fields, (field,))
    if field in {"title", "description", "status", "visibility"}:
        if not isinstance(value, str):
            raise ValueError(f"Thread field {field} must be a string")
        if field == "title":
            return replace(record, title=value, locked_fields=locked)
        if field == "description":
            return replace(record, description=value, locked_fields=locked)
        if field == "status":
            return replace(record, status=value, locked_fields=locked)
        if field == "visibility":
            return replace(record, visibility=value, locked_fields=locked)
    if field == "priority":
        if not isinstance(value, int):
            raise ValueError("Thread priority must be an integer")
        return replace(record, priority=value, locked_fields=locked)
    if field == "related_entities":
        return replace(
            record,
            related_entities=_string_list_value(value, "Thread related_entities"),
            locked_fields=locked,
        )
    raise ValueError(f"Unsupported suggestion field: {field}")


def _memory_importance_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("Memory importance must be numeric")
    result = float(value)
    if not 0 <= result <= 1:
        raise ValueError("Memory importance must be between 0 and 1")
    return result


def _suggestion_before_value(
    model: WorldDataModel,
    suggestion: WorldDataSuggestionRow,
) -> object | None:
    if suggestion.entity_type == "scene_snapshot" and model.scene is not None:
        return getattr(model.scene, suggestion.field_path, None)
    if suggestion.entity_type == "location" and suggestion.entity_id:
        return getattr(
            _location_from_model(model, suggestion.entity_id),
            suggestion.field_path,
            None,
        )
    if suggestion.entity_type == "character" and suggestion.entity_id:
        return getattr(
            _character_from_model(model, suggestion.entity_id),
            suggestion.field_path,
            None,
        )
    if suggestion.entity_type == "active_thread" and suggestion.entity_id:
        return getattr(
            _thread_from_model(model, suggestion.entity_id), suggestion.field_path, None
        )
    if suggestion.entity_type == "memory" and suggestion.entity_id:
        return getattr(
            _memory_from_model(model, suggestion.entity_id), suggestion.field_path, None
        )
    if suggestion.entity_type == "summary" and suggestion.entity_id:
        return getattr(
            _summary_from_model(model, suggestion.entity_id),
            suggestion.field_path,
            None,
        )
    if suggestion.entity_type == "world_state":
        state = _find_state(model.state_rows, suggestion.field_path)
        return _json_value(state.value_json, state.key) if state is not None else None
    return None


def _validate_link_endpoint(
    *,
    kind: str,
    endpoint_type: str,
    endpoint_id: str,
    location_ids: set[str],
    character_ids: set[str],
    thread_ids: set[str],
    state_ids: set[str],
    memory_ids: set[str],
    summary_ids: set[str],
    model: WorldDataModel,
) -> None:
    normalized = endpoint_type.strip()
    valid = {
        "scene_snapshot": {model.scene.snapshot_id} if model.scene else set(),
        "location": location_ids,
        "character": character_ids,
        "active_thread": thread_ids,
        "world_state": state_ids,
        "state": state_ids,
        "memory": memory_ids,
        "summary": summary_ids,
    }
    if normalized == "scenario_section":
        scenario = model.scenario
        section_ids = (
            {key for key, _value in scenario.content_sections} if scenario else set()
        )
        valid["scenario_section"] = (
            section_ids
            | {f"scenario:{scenario.scenario_id}:section:{key}" for key in section_ids}
            if scenario
            else set()
        )
    if endpoint_id not in valid.get(normalized, set()):
        raise ValueError(f"Entity link {kind} does not belong to the active save")


def _changed_fields(
    current: object | None, row: object, fields: tuple[str, ...]
) -> tuple[str, ...]:
    if current is None:
        return fields
    return tuple(
        field for field in fields if getattr(current, field) != getattr(row, field)
    )


def _locked(
    existing: tuple[str, ...] | list[str], changed: tuple[str, ...]
) -> list[str]:
    return sorted({*existing, *changed})


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _string_list_value(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a string list")
    return value


def _none_if_blank(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _json_object(value: str, label: str) -> dict[str, object]:
    loaded = _json_value(value, label)
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], loaded)


def _json_value(value: str, label: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for {label}") from exc


def _coerce_scene_value(field: str, value: object) -> object:
    if field in {"nearby_objects", "hazards", "present_character_ids"}:
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(f"Scene field {field} must be a string list")
    if (
        field == "current_location_id"
        and value is not None
        and not isinstance(value, str)
    ):
        raise ValueError("Scene current_location_id must be a string or null")
    if (
        field == "world_day_index"
        and value is not None
        and (not isinstance(value, int) or isinstance(value, bool))
    ):
        raise ValueError("Scene world_day_index must be an integer or null")
    return value


def _dump_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
