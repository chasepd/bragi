"""Persistence DTOs returned by repositories."""

from __future__ import annotations

from dataclasses import dataclass, field

from bragi.interaction_mode import InteractionMode
from bragi.retry_policy import DEFAULT_RETRY_COUNT, DEFERRED_WORK_MAX_ATTEMPTS


@dataclass(frozen=True)
class ScenarioRecord:
    id: str
    type: str
    title: str
    premise: str
    player_role: str
    content_json: str
    created_at: str | None = None
    updated_at: str | None = None
    interaction_mode: InteractionMode = InteractionMode.ROLEPLAY


@dataclass(frozen=True)
class SaveRecord:
    id: str
    scenario_id: str
    title: str
    active: bool
    scenario_title: str | None = None
    custom_instructions: str = ""
    owner_user_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    last_opened_at: str | None = None
    interaction_mode: InteractionMode = InteractionMode.ROLEPLAY


@dataclass(frozen=True)
class SaveDetailsRecord:
    save: SaveRecord
    scenario: ScenarioRecord
    messages: list[MessageRecord]
    has_more_messages_before: bool = False


@dataclass(frozen=True)
class UserRecord:
    id: str
    username: str
    username_normalized: str
    role: str
    password_hash: str
    status: str
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class UserSessionRecord:
    id: str
    user_id: str
    token_hash: str
    expires_at: str
    revoked_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class ScopedSettingRecord:
    scope: str
    scope_id: str | None
    key: str
    value: object
    value_json: str
    updated_at: str | None = None


@dataclass(frozen=True)
class SaveScenarioUpdateRecord:
    id: str
    save_id: str
    source_message_id: str | None
    title: str
    premise: str
    player_role: str
    content_json: str
    source_message_ids_json: str
    reason: str
    provider: str
    model: str
    created_at: str | None = None
    archived_at: str | None = None

    @property
    def active(self) -> bool:
        return self.archived_at is None


@dataclass(frozen=True)
class LossConditionRecord:
    id: str
    save_id: str
    name: str
    description: str
    status: str
    source: str
    key: str = ""
    label: str = ""
    severity: str = ""
    source_message_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class LossConditionChangeRecord:
    id: str
    save_id: str
    condition_id: str | None
    source_message_id: str | None
    operation: str
    before: dict[str, object] | None
    after: dict[str, object] | None
    reason: str
    provider: str
    model: str
    created_at: str | None = None
    archived_at: str | None = None

    @property
    def active(self) -> bool:
        return self.archived_at is None

    @property
    def before_json(self) -> str | None:
        import json

        if self.before is None:
            return None
        return json.dumps(self.before, sort_keys=True)

    @property
    def after_json(self) -> str | None:
        import json

        if self.after is None:
            return None
        return json.dumps(self.after, sort_keys=True)


@dataclass(frozen=True)
class LossOutcomeRecord:
    id: str
    save_id: str
    condition_id: str | None
    condition_name: str
    triggering_message_id: str
    explanation: str
    evidence: dict[str, object]
    confidence: float
    provider: str
    model: str
    epilogue_provider: str | None
    epilogue_model: str | None
    epilogue_message_id: str | None
    epilogue_error: str | None
    created_at: str | None = None
    archived_at: str | None = None
    outcome_type: str = "loss_condition"

    @property
    def active(self) -> bool:
        return self.archived_at is None

    @property
    def source_message_id(self) -> str:
        return self.triggering_message_id

    @property
    def title(self) -> str:
        return self.condition_name

    @property
    def body(self) -> str:
        return self.explanation

    @property
    def epilogue(self) -> str:
        if isinstance(self.evidence.get("epilogue"), str):
            return str(self.evidence["epilogue"])
        return self.explanation

    @property
    def evidence_json(self) -> str:
        import json

        return json.dumps(self.evidence.get("items", self.evidence), sort_keys=True)


@dataclass(frozen=True)
class MessageRecord:
    id: str
    save_id: str
    role: str
    body: str
    speaker_name: str | None
    provider: str | None
    model: str | None
    token_estimate: int | None
    deleted_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    safety_transition: str = ""
    content_rating: str = "unclassified"


@dataclass(frozen=True)
class MessagePageRecord:
    messages: list[MessageRecord]
    has_more_before: bool = False


@dataclass(frozen=True)
class MessageActionChoiceRecord:
    id: str
    save_id: str
    message_id: str
    ordinal: int
    body: str
    provider: str
    model: str
    content_rating: str = "unclassified"
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class MessageRevisionRecord:
    id: str
    save_id: str
    message_id: str
    revision_number: int
    previous_body: str
    new_body: str
    diff_unified: str
    reconciliation_status: str
    reconciliation_error: str | None = None
    created_at: str | None = None
    reconciled_at: str | None = None


@dataclass(frozen=True)
class MessageRevisionMetadataRecord:
    message_id: str
    revision_count: int
    edited_at: str | None


@dataclass(frozen=True)
class WorldStateRecord:
    id: str
    save_id: str
    key: str
    value: dict[str, object]
    category: str
    confidence: float
    source_message_id: str | None


@dataclass(frozen=True)
class ContextSourceRecord:
    id: str
    save_id: str
    source_type: str
    source_id: str
    title: str
    body: str
    metadata: dict[str, object]
    token_estimate: int | None
    scene_snapshot_id: str | None = None
    scene_generation: int | None = None
    created_turn_number: int | None = None
    expires_after_turn_number: int | None = None


@dataclass(frozen=True)
class ContextSourceSearchHit:
    record: ContextSourceRecord
    bm25_rank: float


@dataclass(frozen=True)
class SceneSnapshotRecord:
    id: str
    save_id: str
    current_location_id: str | None
    situation: str
    objective: str
    in_world_time: str
    time_of_day: str
    day_of_week: str
    weather: str
    mood: str
    nearby_objects: list[str]
    hazards: list[str]
    present_character_ids: list[str]
    source_message_id: str | None
    locked_fields: list[str]
    world_day_index: int | None = None
    world_time_day_index: int | None = None
    world_time_day_label: str = ""
    world_time_phase: str = ""
    world_time_clock_minutes: int | None = None
    world_time_period_label: str = ""
    world_time_source_message_id: str | None = None
    world_time_confidence: float | None = None
    first_seen_message_id: str | None = None
    last_updated_message_id: str | None = None
    scene_generation: int = 1


@dataclass(frozen=True)
class SceneFactProvenanceRecord:
    id: str
    save_id: str
    scene_fact_id: str
    source_message_id: str
    evidence_quote: str
    reason: str
    confidence: float
    created_at: str | None = None


@dataclass(frozen=True)
class SceneFactRecord:
    id: str
    save_id: str
    scene_snapshot_id: str
    scene_generation: int
    fact_type: str
    subject_type: str
    subject_id: str | None
    subject_label: str
    target_type: str
    target_id: str | None
    target_label: str
    aspect: str
    value: str
    conflict_key: str
    lifetime: str
    created_turn_number: int
    expires_after_turn_number: int | None
    archived_at: str | None = None
    archive_reason: str = ""
    created_at: str | None = None
    updated_at: str | None = None
    provenance: tuple[SceneFactProvenanceRecord, ...] = ()


@dataclass(frozen=True)
class DatingRouteStateRecord:
    id: str
    save_id: str
    player_character_id: str
    npc_character_id: str
    stage: str
    first_met_message_id: str | None
    first_met_world_day_index: int | None
    last_interaction_message_id: str | None
    last_interaction_world_day_index: int | None
    completed_interactions: int
    dates_completed: int
    interest_level: str
    trust_level: str
    comfort_with_intimacy: str
    pacing_preference: str
    known_boundaries: list[str]
    unresolved_questions: list[str]
    next_reasonable_step: str
    source_message_id: str | None
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class LocationRecord:
    id: str
    save_id: str
    name: str
    aliases: list[str]
    description: str
    visual_description: str
    parent_location_id: str | None
    connections: list[str]
    status: str
    hazards: list[str]
    source_message_id: str | None
    locked_fields: list[str]
    first_seen_message_id: str | None = None
    last_updated_message_id: str | None = None


@dataclass(frozen=True)
class CharacterRecord:
    id: str
    save_id: str
    name: str
    aliases: list[str]
    role: str
    known_state: str
    met: bool
    appearance: str
    visual_notes: str
    current_clothing: str
    personality: str
    voice: str
    relationships: dict[str, object]
    status: str
    location_id: str | None
    private_notes: str
    source_message_id: str | None
    locked_fields: list[str]
    age: str = ""
    goals: str = ""
    motivations: str = ""
    current_intent: str = ""
    boundaries: str = ""
    attitude_toward_player: str = ""
    cooperation_conditions: str = ""
    protected_from_maintenance: bool = False
    is_player_character: bool = False
    texting_style: str = ""
    contact_name: str = ""
    first_seen_message_id: str | None = None
    last_updated_message_id: str | None = None
    history: str = ""
    content_rating: str = "unclassified"

    def __post_init__(self) -> None:
        history = self.known_state or self.history
        if self.history != history:
            object.__setattr__(self, "history", history)
        if self.known_state != history:
            object.__setattr__(self, "known_state", history)


@dataclass(frozen=True)
class ActiveThreadRecord:
    id: str
    save_id: str
    title: str
    description: str
    status: str
    priority: int
    visibility: str
    related_entities: list[str]
    source_message_id: str | None
    locked_fields: list[str]
    first_seen_message_id: str | None = None
    last_updated_message_id: str | None = None


@dataclass(frozen=True)
class EntityLinkRecord:
    id: str
    save_id: str
    entity_type: str
    entity_id: str
    target_type: str
    target_id: str
    relation: str
    source_message_id: str | None = None


@dataclass(frozen=True)
class CharacterKnowledgeEdgeRecord:
    id: str
    save_id: str
    character_id: str
    target_type: str
    target_id: str
    knowledge_state: str
    acquisition_method: str
    confidence: float
    source_message_id: str | None = None
    source_message_ids: list[str] = field(default_factory=list)
    evidence_quote: str = ""
    created_at: str | None = None
    updated_at: str | None = None
    archived_at: str | None = None


@dataclass(frozen=True)
class MessageVisibilityRecord:
    id: str
    save_id: str
    message_id: str
    character_id: str
    visibility: str
    confidence: float
    source: str
    evidence: str
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class MessageScenePresenceRecord:
    id: str
    save_id: str
    message_id: str
    character_id: str
    source: str
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class CharacterTextThreadRecord:
    id: str
    save_id: str
    character_id: str | None
    title: str
    status: str
    kind: str = "direct"
    memory_body: str = ""
    memory_message_count: int = 0
    memory_updated_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    archived_at: str | None = None


@dataclass(frozen=True)
class CharacterTextThreadParticipantRecord:
    id: str
    save_id: str
    thread_id: str
    character_id: str
    ordinal: int
    created_at: str | None = None
    updated_at: str | None = None
    archived_at: str | None = None


@dataclass(frozen=True)
class CharacterTextMessageRecord:
    id: str
    save_id: str
    thread_id: str
    character_id: str | None
    sender: str
    body: str
    sender_character_id: str | None = None
    provider: str | None = None
    model: str | None = None
    token_estimate: int | None = None
    delivery_status: str = "sent"
    delivery_error: str | None = None
    delivery_job_id: str | None = None
    delivery_attempt: int = 0
    in_world_sent_at: str | None = None
    delivered_at: str | None = None
    read_at: str | None = None
    reply_to_message_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    deleted_at: str | None = None
    content_rating: str = "unclassified"


@dataclass(frozen=True)
class CharacterTextActivityEventRecord:
    id: str
    save_id: str
    ordinal: int
    thread_id: str
    activity_type: str
    text_message_id: str | None = None
    read_count: int = 0
    delivery_status: str = ""
    created_at: str | None = None


@dataclass(frozen=True)
class CharacterTextMessageAttachmentRecord:
    id: str
    save_id: str
    thread_id: str
    text_message_id: str
    character_id: str
    ordinal: int
    kind: str
    status: str
    media_asset_id: str | None = None
    prompt: str = ""
    error: str | None = None
    metadata_json: str = "{}"
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class CharacterTextMessageRevisionRecord:
    id: str
    save_id: str
    text_message_id: str
    revision_number: int
    previous_body: str
    new_body: str
    diff_unified: str
    reconciliation_status: str
    reconciliation_error: str | None = None
    created_at: str | None = None
    reconciled_at: str | None = None


@dataclass(frozen=True)
class CharacterTextProvenanceRecord:
    id: str
    save_id: str
    thread_id: str
    text_message_id: str
    target_type: str
    target_id: str
    operation: str
    field_path: str
    created_at: str | None = None


@dataclass(frozen=True)
class CharacterTextProactiveTriggerRecord:
    id: str
    save_id: str
    character_id: str
    trigger_key: str
    trigger_type: str
    thread_id: str | None = None
    text_message_id: str | None = None
    source_type: str = ""
    source_id: str = ""
    source_message_id: str | None = None
    reason: str = ""
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class CharacterContactStateRecord:
    id: str
    save_id: str
    player_character_id: str
    character_id: str
    player_has_character_number: bool
    character_has_player_number: bool
    source_message_id: str | None = None
    source_text_message_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    archived_at: str | None = None


@dataclass(frozen=True)
class ContextUpdateSuggestionRecord:
    id: str
    save_id: str
    update_type: str
    entity_type: str
    entity_id: str | None
    field_path: str
    proposed_value: object
    status: str
    reason: str
    confidence: float
    source_message_ids: list[str]
    created_at: str | None = None
    resolved_at: str | None = None
    review_attempt_count: int = 0
    next_review_at: str | None = None
    last_review_error: str | None = None
    max_retry_count: int = DEFAULT_RETRY_COUNT


@dataclass(frozen=True)
class ContextUpdateAuditRecord:
    id: str
    save_id: str
    suggestion_id: str | None
    operation: str
    entity_type: str
    entity_id: str | None
    field_path: str
    before: object | None
    after: object | None
    reason: str
    confidence: float
    source_message_ids: list[str]
    created_at: str | None = None


@dataclass(frozen=True)
class StateChangeRecord:
    id: str
    save_id: str
    source_message_id: str | None
    operation: str
    state_key: str
    before_json: str | None
    after_json: str | None


@dataclass(frozen=True)
class TurnOutcomeRecord:
    id: str
    save_id: str
    message_id: str | None
    payload: dict[str, object]
    created_at: str | None = None


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    save_id: str
    body: str
    tags: list[str]
    importance: float
    source_message_id: str | None
    source_message_ids: list[str] = field(default_factory=list)
    claim_fingerprint: str = ""
    source_observation_ids: list[str] = field(default_factory=list)
    epistemic_status: str = "legacy_unclassified"
    epistemic_actor_id: str | None = None
    epistemic_actor_name: str = ""


@dataclass(frozen=True)
class ContextObservationRecord:
    id: str
    save_id: str
    observation_type: str
    claim: str
    evidence_quote: str
    source_message_ids: list[str]
    scope: str
    status: str
    confidence: float
    tags: list[str]
    metadata: dict[str, object]
    created_at: str | None = None
    updated_at: str | None = None
    archived_at: str | None = None
    epistemic_status: str = "legacy_unclassified"
    epistemic_actor_id: str | None = None
    epistemic_actor_name: str = ""


@dataclass(frozen=True)
class ContextObservationCurationStateRecord:
    observation_id: str
    save_id: str
    attempt_count: int
    next_eligible_at: str | None
    lease_token: str | None
    lease_until: str | None
    last_error: str | None
    terminal_outcome: str | None
    completed_at: str | None
    created_at: str
    updated_at: str
    max_attempts: int = DEFERRED_WORK_MAX_ATTEMPTS


@dataclass(frozen=True)
class ContextObservationCurationHealthRecord:
    pending_count: int
    eligible_count: int
    leased_count: int
    oldest_pending_at: str | None
    total_attempt_count: int
    max_attempt_count: int
    terminal_failure_count: int


@dataclass(frozen=True)
class SummaryRecord:
    id: str
    save_id: str
    covers_message_start_id: str
    covers_message_end_id: str
    body: str
    provider: str
    model: str
    content_rating: str = "unclassified"
    source_message_ids: tuple[str, ...] = ()
    source_summary_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SummaryPressureStateRecord:
    save_id: str
    history_revision: int
    summarized_through_message_id: str | None
    unsummarized_message_count: int
    unsummarized_player_count: int
    unsummarized_narrator_count: int
    unsummarized_other_count: int
    unsummarized_token_estimate: int
    active_summary_count: int
    active_summary_token_estimate: int


@dataclass(frozen=True)
class ProviderModelRecord:
    id: str
    provider: str
    model_id: str
    display_name: str
    capabilities: list[str]
    context_window: int | None
    available: bool
    supported_parameters: list[str] = field(default_factory=list)
    pricing: dict[str, str] = field(default_factory=dict)
    thinking: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderCatalogEntryRecord:
    id: str
    provider: str
    slug: str
    name: str
    privacy_policy_url: str | None
    terms_of_service_url: str | None
    status_page_url: str | None
    headquarters: str | None
    datacenters: list[str]
    refreshed_at: str | None = None


@dataclass(frozen=True)
class ProviderConfigRecord:
    id: str
    provider: str
    enabled: bool
    has_api_key: bool
    last_model_refresh_at: str | None
    last_error: str | None


@dataclass(frozen=True)
class ModelPreferenceRecord:
    id: str
    task: str
    provider: str
    model_id: str


@dataclass(frozen=True)
class JobRecord:
    id: str
    save_id: str | None
    type: str
    status: str
    payload: dict[str, object]
    result: dict[str, object] | None
    error: str | None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: int | None = None
    creator_user_id: str | None = None
    diagnostics: dict[str, object] | None = None


@dataclass(frozen=True)
class ScheduledTaskRecord:
    id: str
    task_type: str
    save_id: str | None
    enabled: bool
    interval_seconds: int
    next_run_at: str
    lease_until: str | None
    last_started_at: str | None
    last_completed_at: str | None
    last_job_id: str | None
    failure_count: int
    payload: dict[str, object]
    result: dict[str, object] | None
    error: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class JobStepRecord:
    id: str
    job_id: str
    name: str
    status: str
    provider: str | None
    model: str | None
    task: str | None
    started_at: str | None
    completed_at: str | None
    duration_ms: int | None
    error: str | None
    metadata: dict[str, object]


@dataclass(frozen=True)
class RuntimePerformanceRecord:
    job_type: str | None = None
    step_name: str | None = None
    provider: str | None = None
    model: str | None = None
    task: str | None = None
    sample_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    cancelled_count: int = 0
    skipped_count: int = 0
    average_duration_ms: int | None = None
    p50_duration_ms: int | None = None
    p95_duration_ms: int | None = None
    min_duration_ms: int | None = None
    max_duration_ms: int | None = None
    latest_duration_ms: int | None = None
    average_queue_wait_ms: int | None = None
    p95_queue_wait_ms: int | None = None
    failure_rate: float = 0.0
    latest_completed_at: str | None = None


@dataclass(frozen=True)
class RuntimeSlowOperationRecord:
    job_id: str
    save_id: str | None
    job_type: str
    status: str
    started_at: str | None
    completed_at: str | None
    duration_ms: int | None
    queue_wait_ms: int | None = None
    slowest_step_name: str | None = None
    slowest_step_duration_ms: int | None = None
    provider: str | None = None
    model: str | None = None
    task: str | None = None


@dataclass(frozen=True)
class MediaAssetRecord:
    id: str
    save_id: str
    source_message_id: str | None
    type: str
    path: str
    thumbnail_path: str | None
    prompt: str
    provider: str
    model: str
    status: str
    mime_type: str = "image/png"
    metadata_json: str = "{}"
    source_media_asset_id: str | None = None
    created_at: str | None = None
    archived_at: str | None = None
