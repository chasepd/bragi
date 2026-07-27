"""SQLite schema initialization for Bragi."""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from contextvars import ContextVar
from hashlib import sha256
from pathlib import Path

from bragi.model_tasks import is_retired_model_task
from bragi.observation_types import normalize_observation_type
from bragi.persistence.context_provenance import merge_context_source_metadata
from bragi.private_files import ensure_private_file
from bragi.text_search import (
    cjk_lexical_anchors,
    structured_identifier_filter,
    structured_identifiers,
    unicode_word_terms,
)

CURRENT_SCHEMA_VERSION = 72
_MAX_CONTEXT_SOURCE_SEARCH_TEXT_CHARS = 65_536
_MAX_CONTEXT_SOURCE_INDEX_TERMS = 256
_MAX_CONTEXT_SOURCE_INDEX_IDENTIFIERS = 128
_MAX_CONTEXT_INDEX_ROWS_PER_REBUILD = 3_000_000
_MAX_CONTEXT_INDEX_TEXT_CHARS_PER_REBUILD = 32 * 1024 * 1024
_MAX_KNOWLEDGE_EDGE_SOURCE_MESSAGE_IDS = 64
_MAX_MEMORY_PROVENANCE_IDS = 64

_PRESERVE_SCHEMA_SCRIPT_TRANSACTION: ContextVar[bool] = ContextVar(
    "_PRESERVE_SCHEMA_SCRIPT_TRANSACTION",
    default=False,
)
_MIGRATION_TIME_OF_DAY_VALUES = (
    "morning",
    "late_morning",
    "afternoon",
    "evening",
    "night",
)
_MIGRATION_DAY_OF_WEEK_VALUES = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)
_MIGRATION_TIME_OF_DAY_ALIASES = {
    "late morning": "late_morning",
    "late-morning": "late_morning",
    "midday": "afternoon",
    "noon": "afternoon",
    "dusk": "evening",
    "sunset": "evening",
    "dawn": "morning",
    "sunrise": "morning",
    "midnight": "night",
    "late night": "night",
}
_MIGRATION_CLOCK_RE = re.compile(
    r"\b([01]?\d|2[0-3])(?::([0-5]\d))?\s*([ap]\.?m\.?)?\b",
    flags=re.IGNORECASE,
)
_DEPRECATED_SCENARIO_CHARACTER_SECTION_KEYS = (
    "characters",
    "romance_options",
    "suspects",
    "crew_and_command",
    "party_roster",
    "crew_and_contacts",
    "major_npcs",
    "population_and_residents",
    "rivals_and_factions",
    "traveling_party",
    "reputation_and_contacts",
)
_DEPRECATED_SCENARIO_FACTION_APPEND_KEYS = (
    "rivals_and_factions",
    "reputation_and_contacts",
)


CURRENT_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scenarios (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    title TEXT NOT NULL,
    premise TEXT NOT NULL DEFAULT '',
    player_role TEXT NOT NULL DEFAULT '',
    content_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS saves (
    id TEXT PRIMARY KEY,
    scenario_id TEXT NOT NULL REFERENCES scenarios(id),
    title TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    custom_instructions TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_opened_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    save_id TEXT NOT NULL REFERENCES saves(id),
    role TEXT NOT NULL,
    speaker_name TEXT,
    body TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    token_estimate INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at TEXT,
    safety_transition TEXT NOT NULL DEFAULT '',
    content_rating TEXT NOT NULL DEFAULT 'unclassified'
);

CREATE TABLE IF NOT EXISTS message_revisions (
    id TEXT PRIMARY KEY,
    save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
    message_id TEXT NOT NULL REFERENCES messages(id),
    revision_number INTEGER NOT NULL,
    previous_body TEXT NOT NULL,
    new_body TEXT NOT NULL,
    diff_unified TEXT NOT NULL,
    reconciliation_status TEXT NOT NULL DEFAULT 'queued',
    reconciliation_error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    reconciled_at TEXT,
    UNIQUE(message_id, revision_number)
);

CREATE TABLE IF NOT EXISTS world_state (
    id TEXT PRIMARY KEY,
    save_id TEXT NOT NULL REFERENCES saves(id),
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 1.0,
    source_message_id TEXT REFERENCES messages(id),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    archived_at TEXT,
    UNIQUE(save_id, key)
);

CREATE TABLE IF NOT EXISTS context_sources (
    id TEXT PRIMARY KEY,
    save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    body TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    token_estimate INTEGER,
    scene_snapshot_id TEXT REFERENCES scene_snapshots(id) ON DELETE SET NULL,
    scene_generation INTEGER,
    created_turn_number INTEGER,
    expires_after_turn_number INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    archived_at TEXT,
    UNIQUE(save_id, source_type, source_id)
);

CREATE TABLE IF NOT EXISTS context_observations (
    id TEXT PRIMARY KEY,
    save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
    observation_type TEXT NOT NULL,
    claim TEXT NOT NULL,
    evidence_quote TEXT NOT NULL DEFAULT '',
    source_message_ids_json TEXT NOT NULL DEFAULT '[]',
    scope TEXT NOT NULL DEFAULT 'turn',
    status TEXT NOT NULL DEFAULT 'pending',
    confidence REAL NOT NULL DEFAULT 0.0,
    tags_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    archived_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_context_observations_save_status
ON context_observations(save_id, status, archived_at);

CREATE INDEX IF NOT EXISTS idx_context_observations_save_created
ON context_observations(save_id, created_at, id);

CREATE TABLE IF NOT EXISTS context_observation_curation_state (
    observation_id TEXT PRIMARY KEY
        REFERENCES context_observations(id) ON DELETE CASCADE,
    save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    next_eligible_at TEXT,
    lease_token TEXT,
    lease_until TEXT,
    last_error TEXT,
    terminal_outcome TEXT,
    completed_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_context_observation_curation_due
ON context_observation_curation_state(
    save_id, terminal_outcome, next_eligible_at, lease_until
);

CREATE TABLE IF NOT EXISTS scene_snapshots (
    id TEXT PRIMARY KEY,
    save_id TEXT NOT NULL UNIQUE REFERENCES saves(id) ON DELETE CASCADE,
    current_location_id TEXT,
    situation TEXT NOT NULL DEFAULT '',
    objective TEXT NOT NULL DEFAULT '',
    in_world_time TEXT NOT NULL DEFAULT '',
    time_of_day TEXT NOT NULL DEFAULT '',
    day_of_week TEXT NOT NULL DEFAULT '',
    world_day_index INTEGER,
    world_time_day_index INTEGER,
    world_time_day_label TEXT NOT NULL DEFAULT '',
    world_time_phase TEXT NOT NULL DEFAULT '',
    world_time_clock_minutes INTEGER,
    world_time_period_label TEXT NOT NULL DEFAULT '',
    world_time_source_message_id TEXT REFERENCES messages(id),
    world_time_confidence REAL,
    weather TEXT NOT NULL DEFAULT '',
    mood TEXT NOT NULL DEFAULT '',
    nearby_objects_json TEXT NOT NULL DEFAULT '[]',
    hazards_json TEXT NOT NULL DEFAULT '[]',
    present_character_ids_json TEXT NOT NULL DEFAULT '[]',
    source_message_id TEXT REFERENCES messages(id),
    locked_fields_json TEXT NOT NULL DEFAULT '[]',
    first_seen_message_id TEXT REFERENCES messages(id),
    last_updated_message_id TEXT REFERENCES messages(id),
    scene_generation INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(save_id, current_location_id) REFERENCES locations(save_id, id)
);

CREATE TABLE IF NOT EXISTS locations (
    id TEXT PRIMARY KEY,
    save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    aliases_json TEXT NOT NULL DEFAULT '[]',
    description TEXT NOT NULL DEFAULT '',
    visual_description TEXT NOT NULL DEFAULT '',
    parent_location_id TEXT,
    connections_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT '',
    hazards_json TEXT NOT NULL DEFAULT '[]',
    source_message_id TEXT REFERENCES messages(id),
    locked_fields_json TEXT NOT NULL DEFAULT '[]',
    first_seen_message_id TEXT REFERENCES messages(id),
    last_updated_message_id TEXT REFERENCES messages(id),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    archived_at TEXT,
    UNIQUE(save_id, id),
    FOREIGN KEY(save_id, parent_location_id) REFERENCES locations(save_id, id)
);

CREATE TABLE IF NOT EXISTS characters (
    id TEXT PRIMARY KEY,
    save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    aliases_json TEXT NOT NULL DEFAULT '[]',
    role TEXT NOT NULL DEFAULT '',
    age TEXT NOT NULL DEFAULT '',
    known_state TEXT NOT NULL DEFAULT '',
    history TEXT NOT NULL DEFAULT '',
    met INTEGER NOT NULL DEFAULT 0,
    appearance TEXT NOT NULL DEFAULT '',
    visual_notes TEXT NOT NULL DEFAULT '',
    current_clothing TEXT NOT NULL DEFAULT '',
    personality TEXT NOT NULL DEFAULT '',
    voice TEXT NOT NULL DEFAULT '',
    relationships_json TEXT NOT NULL DEFAULT '{}',
    goals TEXT NOT NULL DEFAULT '',
    motivations TEXT NOT NULL DEFAULT '',
    current_intent TEXT NOT NULL DEFAULT '',
    boundaries TEXT NOT NULL DEFAULT '',
    attitude_toward_player TEXT NOT NULL DEFAULT '',
    cooperation_conditions TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT '',
    location_id TEXT,
    private_notes TEXT NOT NULL DEFAULT '',
    source_message_id TEXT REFERENCES messages(id),
    locked_fields_json TEXT NOT NULL DEFAULT '[]',
    protected_from_maintenance INTEGER NOT NULL DEFAULT 0,
    is_player_character INTEGER NOT NULL DEFAULT 0,
    first_seen_message_id TEXT REFERENCES messages(id),
    last_updated_message_id TEXT REFERENCES messages(id),
    content_rating TEXT NOT NULL DEFAULT 'unclassified',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    archived_at TEXT,
    FOREIGN KEY(save_id, location_id) REFERENCES locations(save_id, id)
);

CREATE TRIGGER IF NOT EXISTS null_location_references_before_location_delete
BEFORE DELETE ON locations
FOR EACH ROW
BEGIN
    UPDATE scene_snapshots
    SET current_location_id = NULL
    WHERE save_id = OLD.save_id AND current_location_id = OLD.id;

    UPDATE locations
    SET parent_location_id = NULL
    WHERE save_id = OLD.save_id AND parent_location_id = OLD.id;

    UPDATE characters
    SET location_id = NULL
    WHERE save_id = OLD.save_id AND location_id = OLD.id;
END;

CREATE TABLE IF NOT EXISTS active_threads (
    id TEXT PRIMARY KEY,
    save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    priority INTEGER NOT NULL DEFAULT 0,
    visibility TEXT NOT NULL DEFAULT 'public',
    related_entities_json TEXT NOT NULL DEFAULT '[]',
    source_message_id TEXT REFERENCES messages(id),
    locked_fields_json TEXT NOT NULL DEFAULT '[]',
    first_seen_message_id TEXT REFERENCES messages(id),
    last_updated_message_id TEXT REFERENCES messages(id),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    archived_at TEXT
);

CREATE TABLE IF NOT EXISTS entity_links (
    id TEXT PRIMARY KEY,
    save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation TEXT NOT NULL DEFAULT '',
    source_message_id TEXT REFERENCES messages(id),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(save_id, entity_type, entity_id, target_type, target_id, relation)
);

CREATE TABLE IF NOT EXISTS context_update_suggestions (
    id TEXT PRIMARY KEY,
    save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
    update_type TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT,
    field_path TEXT NOT NULL DEFAULT '',
    proposed_value_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    reason TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0.0,
    source_message_ids_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at TEXT,
    review_attempt_count INTEGER NOT NULL DEFAULT 0,
    next_review_at TEXT,
    last_review_error TEXT
);

CREATE TABLE IF NOT EXISTS context_update_audit (
    id TEXT PRIMARY KEY,
    save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
    suggestion_id TEXT REFERENCES context_update_suggestions(id) ON DELETE SET NULL,
    operation TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT,
    field_path TEXT NOT NULL DEFAULT '',
    before_json TEXT,
    after_json TEXT,
    reason TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0.0,
    source_message_ids_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS state_changes (
    id TEXT PRIMARY KEY,
    save_id TEXT NOT NULL REFERENCES saves(id),
    source_message_id TEXT REFERENCES messages(id),
    operation TEXT NOT NULL,
    state_key TEXT NOT NULL,
    before_json TEXT,
    after_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    save_id TEXT NOT NULL REFERENCES saves(id),
    body TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    importance INTEGER NOT NULL DEFAULT 1,
    source_message_id TEXT REFERENCES messages(id),
    source_message_ids_json TEXT NOT NULL DEFAULT '[]',
    claim_fingerprint TEXT NOT NULL DEFAULT '',
    source_observation_ids_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    archived_at TEXT
);

CREATE TABLE IF NOT EXISTS summaries (
    id TEXT PRIMARY KEY,
    save_id TEXT NOT NULL REFERENCES saves(id),
    covers_message_start_id TEXT NOT NULL,
    covers_message_end_id TEXT NOT NULL,
    body TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    content_rating TEXT NOT NULL DEFAULT 'unclassified',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    archived_at TEXT
);

CREATE TABLE IF NOT EXISTS save_scenario_updates (
    id TEXT PRIMARY KEY,
    save_id TEXT NOT NULL REFERENCES saves(id),
    source_message_id TEXT REFERENCES messages(id),
    title TEXT NOT NULL,
    premise TEXT NOT NULL DEFAULT '',
    player_role TEXT NOT NULL DEFAULT '',
    content_json TEXT NOT NULL,
    source_message_ids_json TEXT NOT NULL DEFAULT '[]',
    reason TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    archived_at TEXT
);

CREATE TABLE IF NOT EXISTS save_loss_conditions (
    id TEXT PRIMARY KEY,
    save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
    key TEXT NOT NULL DEFAULT '',
    label TEXT NOT NULL DEFAULT '',
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    severity TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'manual',
    source_message_id TEXT REFERENCES messages(id),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    archived_at TEXT,
    UNIQUE(save_id, key)
);

CREATE TABLE IF NOT EXISTS save_loss_condition_changes (
    id TEXT PRIMARY KEY,
    save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
    condition_id TEXT REFERENCES save_loss_conditions(id) ON DELETE SET NULL,
    source_message_id TEXT REFERENCES messages(id),
    operation TEXT NOT NULL,
    before_json TEXT,
    after_json TEXT,
    reason TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    archived_at TEXT
);

CREATE TABLE IF NOT EXISTS save_loss_outcomes (
    id TEXT PRIMARY KEY,
    save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
    condition_id TEXT REFERENCES save_loss_conditions(id) ON DELETE CASCADE,
    condition_name TEXT NOT NULL,
    triggering_message_id TEXT NOT NULL REFERENCES messages(id),
    explanation TEXT NOT NULL DEFAULT '',
    evidence_json TEXT NOT NULL DEFAULT '{}',
    confidence REAL NOT NULL DEFAULT 0.0,
    provider TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    outcome_type TEXT NOT NULL DEFAULT 'loss_condition',
    epilogue_provider TEXT,
    epilogue_model TEXT,
    epilogue_message_id TEXT REFERENCES messages(id),
    epilogue_error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    archived_at TEXT
);

CREATE TABLE IF NOT EXISTS provider_configs (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL UNIQUE,
    enabled INTEGER NOT NULL DEFAULT 0,
    has_api_key INTEGER NOT NULL DEFAULT 0,
    last_model_refresh_at TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_preferences (
    id TEXT PRIMARY KEY,
    task TEXT NOT NULL UNIQUE,
    provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS provider_models (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    capabilities_json TEXT NOT NULL,
    supported_parameters_json TEXT NOT NULL DEFAULT '[]',
    context_window INTEGER,
    available INTEGER NOT NULL DEFAULT 1,
    pricing_json TEXT NOT NULL DEFAULT '{}',
    thinking_json TEXT NOT NULL DEFAULT '{}',
    refreshed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(provider, model_id)
);

CREATE TABLE IF NOT EXISTS provider_catalog_entries (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    slug TEXT NOT NULL,
    name TEXT NOT NULL,
    privacy_policy_url TEXT,
    terms_of_service_url TEXT,
    status_page_url TEXT,
    headquarters TEXT,
    datacenters_json TEXT NOT NULL DEFAULT '[]',
    refreshed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_provider_catalog_entries_provider_slug
ON provider_catalog_entries(provider, slug);

CREATE TABLE IF NOT EXISTS media_assets (
    id TEXT PRIMARY KEY,
    save_id TEXT NOT NULL REFERENCES saves(id),
    source_message_id TEXT REFERENCES messages(id),
    type TEXT NOT NULL,
    path TEXT NOT NULL,
    thumbnail_path TEXT,
    prompt TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL,
    mime_type TEXT NOT NULL DEFAULT 'image/png',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    source_media_asset_id TEXT REFERENCES media_assets(id),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    archived_at TEXT
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    save_id TEXT REFERENCES saves(id),
    creator_user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
    type TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    result_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TEXT,
    completed_at TEXT,
    duration_ms INTEGER,
    diagnostics_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_status_type_save_completed
ON jobs(status, type, save_id, completed_at);

CREATE INDEX IF NOT EXISTS idx_jobs_save_status_completed
ON jobs(save_id, status, completed_at);

CREATE TABLE IF NOT EXISTS job_steps (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    task TEXT,
    started_at TEXT,
    completed_at TEXT,
    duration_ms INTEGER,
    error TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_job_steps_job_id
ON job_steps(job_id);

CREATE INDEX IF NOT EXISTS idx_job_steps_name_status_completed
ON job_steps(name, status, completed_at);

CREATE INDEX IF NOT EXISTS idx_job_steps_provider_model_task_status_completed
ON job_steps(provider, model, task, status, completed_at);

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

"""


def migrate_database(database_path: Path | str) -> None:
    path = Path(database_path)
    ensure_private_file(path)
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if "schema_migrations" not in table_names:
            _raise_if_baseline_blocked_by_existing_tables(
                connection,
                allowed_tables=set(),
                schema_state="Database is missing schema_migrations",
            )
            _initialize_baseline_schema(connection)
            return

        current = connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        ).fetchone()[0]
        if current == 0:
            _raise_if_baseline_blocked_by_existing_tables(
                connection,
                allowed_tables={"schema_migrations"},
                schema_state="Database schema_migrations has no applied versions",
            )
            _initialize_baseline_schema(connection)
            return
        if current < CURRENT_SCHEMA_VERSION:
            if current == 71:
                _migrate_schema_71_to_72(connection)
                current = CURRENT_SCHEMA_VERSION
            elif current == 70:
                _migrate_schema_70_to_71(connection)
                _migrate_schema_71_to_72(connection)
                current = CURRENT_SCHEMA_VERSION
            elif current == 69:
                _migrate_schema_69_to_70(connection)
                _migrate_schema_70_to_71(connection)
                _migrate_schema_71_to_72(connection)
                current = CURRENT_SCHEMA_VERSION
            elif current == 68:
                _migrate_schema_68_to_69(connection)
                _migrate_schema_69_to_70(connection)
                _migrate_schema_70_to_71(connection)
                _migrate_schema_71_to_72(connection)
                current = CURRENT_SCHEMA_VERSION
            elif current == 67:
                _migrate_schema_67_to_68(connection)
                _migrate_schema_68_to_69(connection)
                _migrate_schema_69_to_70(connection)
                _migrate_schema_70_to_71(connection)
                _migrate_schema_71_to_72(connection)
                current = CURRENT_SCHEMA_VERSION
            elif current == 66:
                _migrate_schema_66_to_67(connection)
                _migrate_schema_67_to_68(connection)
                _migrate_schema_68_to_69(connection)
                _migrate_schema_69_to_70(connection)
                _migrate_schema_70_to_71(connection)
                _migrate_schema_71_to_72(connection)
                current = CURRENT_SCHEMA_VERSION
            elif current == 65:
                _migrate_schema_65_to_66(connection)
                _migrate_schema_66_to_67(connection)
                _migrate_schema_67_to_68(connection)
                _migrate_schema_68_to_69(connection)
                _migrate_schema_69_to_70(connection)
                _migrate_schema_70_to_71(connection)
                _migrate_schema_71_to_72(connection)
                current = CURRENT_SCHEMA_VERSION
            elif current == 64:
                _migrate_schema_64_to_65(connection)
                _migrate_schema_65_to_66(connection)
                _migrate_schema_66_to_67(connection)
                _migrate_schema_67_to_68(connection)
                _migrate_schema_68_to_69(connection)
                _migrate_schema_69_to_70(connection)
                _migrate_schema_70_to_71(connection)
                _migrate_schema_71_to_72(connection)
                current = CURRENT_SCHEMA_VERSION
            elif current == 63:
                _migrate_schema_63_to_64(connection)
                _migrate_schema_64_to_65(connection)
                _migrate_schema_65_to_66(connection)
                _migrate_schema_66_to_67(connection)
                _migrate_schema_67_to_68(connection)
                _migrate_schema_68_to_69(connection)
                _migrate_schema_69_to_70(connection)
                _migrate_schema_70_to_71(connection)
                _migrate_schema_71_to_72(connection)
                current = CURRENT_SCHEMA_VERSION
            elif current == 62:
                _migrate_schema_62_to_63(connection)
                _migrate_schema_63_to_64(connection)
                _migrate_schema_64_to_65(connection)
                _migrate_schema_65_to_66(connection)
                _migrate_schema_66_to_67(connection)
                _migrate_schema_67_to_68(connection)
                _migrate_schema_68_to_69(connection)
                _migrate_schema_69_to_70(connection)
                _migrate_schema_70_to_71(connection)
                _migrate_schema_71_to_72(connection)
                current = CURRENT_SCHEMA_VERSION
            elif current == 61:
                _migrate_schema_61_to_62(connection)
                _migrate_schema_62_to_63(connection)
                _migrate_schema_63_to_64(connection)
                _migrate_schema_64_to_65(connection)
                _migrate_schema_65_to_66(connection)
                _migrate_schema_66_to_67(connection)
                _migrate_schema_67_to_68(connection)
                _migrate_schema_68_to_69(connection)
                _migrate_schema_69_to_70(connection)
                _migrate_schema_70_to_71(connection)
                _migrate_schema_71_to_72(connection)
                current = CURRENT_SCHEMA_VERSION
            else:
                raise RuntimeError(
                    "Database schema version "
                    f"{current} is no longer supported by this Bragi build. "
                    "Restore a schema-61 backup or run a historical Bragi build "
                    "to upgrade this database before starting Bragi."
                )
        if current < CURRENT_SCHEMA_VERSION:
            raise RuntimeError(
                "Database schema version "
                f"{current} is no longer supported by this Bragi build. "
                "Restore a schema-61 backup or run a historical Bragi build "
                "to upgrade this database before starting Bragi."
            )
        if current > CURRENT_SCHEMA_VERSION:
            raise RuntimeError(
                "Database schema version "
                f"{current} is newer than this Bragi build supports "
                f"({CURRENT_SCHEMA_VERSION}). Upgrade Bragi before starting."
            )
        if not _memory_claim_fingerprint_index_is_unique(connection):
            _migrate_schema_71_to_72(connection)
        _ensure_runtime_telemetry_schema(connection)
        _ensure_context_update_suggestion_review_schema(connection)
        _ensure_context_observation_curation_schema(connection)
        _ensure_character_current_clothing_schema(connection)
        _ensure_character_text_schema(connection)
        _ensure_character_text_activity_schema(connection)
        _ensure_scene_world_time_schema(connection)
        _ensure_hot_narration_query_indexes(connection)
        _ensure_continuity_index_revision_schema(connection)
        _ensure_context_source_search_terms_schema(connection)
        connection.commit()


def _initialize_baseline_schema(connection: sqlite3.Connection) -> None:
    preserve_transaction_token = _PRESERVE_SCHEMA_SCRIPT_TRANSACTION.set(True)
    try:
        _execute_schema_script(
            connection,
            f"""
            BEGIN;
            {CURRENT_SCHEMA_SQL}
            """,
        )
        _ensure_character_knowledge_schema(connection)
        _ensure_context_revision_schema(connection)
        _ensure_continuity_index_revision_schema(connection)
        _ensure_scheduled_tasks_schema(connection)
        _ensure_auth_schema(connection)
        _ensure_save_access_schema(connection)
        _ensure_scoped_settings_schema(connection)
        _migrate_app_settings_to_scoped_settings(connection)
        _ensure_player_character_schema(connection)
        _ensure_provider_catalog_schema(connection)
        _ensure_hot_narration_query_indexes(connection)
        _ensure_context_source_fts_schema(connection)
        _ensure_context_source_search_terms_schema(connection)
        _ensure_message_scene_presence_schema(connection)
        _ensure_message_action_choices_schema(connection)
        _normalize_legacy_action_choice_scenarios(connection)
        _ensure_character_agency_schema(connection)
        _ensure_character_age_schema(connection)
        _ensure_scene_world_time_schema(connection)
        _ensure_dating_route_state_schema(connection)
        _ensure_character_text_schema(connection)
        _ensure_character_text_activity_schema(connection)
        _ensure_character_contact_state_schema(connection)
        _ensure_character_text_proactive_trigger_schema(connection)
        _ensure_character_text_message_revision_schema(connection)
        _ensure_character_text_message_attachment_schema(connection)
        _ensure_turn_snapshot_schema(connection)
        _ensure_context_revision_schema(connection)
        _ensure_continuity_index_revision_schema(connection)
        _ensure_character_contact_name_schema(connection)
        _ensure_character_texting_style_schema(connection)
        _ensure_character_current_clothing_schema(connection)
        _ensure_character_text_thread_memory_schema(connection)
        _ensure_character_history_schema(connection)
        _add_column_if_missing(
            connection,
            "provider_models",
            "thinking_json",
            "TEXT NOT NULL DEFAULT '{}'",
        )
        connection.executemany(
            "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)",
            [(version,) for version in range(1, CURRENT_SCHEMA_VERSION + 1)],
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        _PRESERVE_SCHEMA_SCRIPT_TRANSACTION.reset(preserve_transaction_token)


def _raise_if_baseline_blocked_by_existing_tables(
    connection: sqlite3.Connection,
    *,
    allowed_tables: set[str],
    schema_state: str,
) -> None:
    extra_tables = _user_table_names(connection) - allowed_tables
    if not extra_tables:
        return
    table_list = ", ".join(sorted(extra_tables))
    raise RuntimeError(
        f"{schema_state}, but existing tables are present: {table_list}. "
        "Restore from backup or remove the incomplete database before starting Bragi."
    )


def _execute_schema_script(connection: sqlite3.Connection, script: str) -> None:
    if not _PRESERVE_SCHEMA_SCRIPT_TRANSACTION.get():
        connection.executescript(script)
        return

    statement_lines: list[str] = []
    for line in script.splitlines():
        statement_lines.append(line)
        statement = "\n".join(statement_lines).strip()
        if not statement:
            statement_lines.clear()
            continue
        if sqlite3.complete_statement(statement):
            connection.execute(statement)
            statement_lines.clear()
    statement = "\n".join(statement_lines).strip()
    if statement:
        connection.execute(statement)


def _user_table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            AND name NOT LIKE 'sqlite_%'
            """
        )
    }




def _ensure_character_history_schema(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "characters"):
        return
    _add_column_if_missing(
        connection,
        "characters",
        "history",
        "TEXT NOT NULL DEFAULT ''",
    )
    columns = _column_names(connection, "characters")
    if {"history", "known_state"} <= columns:
        connection.execute(
            """
            UPDATE characters
            SET history = known_state
            WHERE history = '' AND known_state != ''
            """
        )




def _ensure_provider_catalog_schema(connection: sqlite3.Connection) -> None:
    _execute_schema_script(
        connection,
        """
        CREATE TABLE IF NOT EXISTS provider_catalog_entries (
            id TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            slug TEXT NOT NULL,
            name TEXT NOT NULL,
            privacy_policy_url TEXT,
            terms_of_service_url TEXT,
            status_page_url TEXT,
            headquarters TEXT,
            datacenters_json TEXT NOT NULL DEFAULT '[]',
            refreshed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_provider_catalog_entries_provider_slug
        ON provider_catalog_entries(provider, slug);
        """
    )


def _ensure_turn_snapshot_schema(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "saves"):
        return
    _execute_schema_script(
        connection,
        """
        CREATE TABLE IF NOT EXISTS save_snapshot_objects (
            object_hash TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            encoding TEXT NOT NULL DEFAULT 'zlib-json-v1',
            payload BLOB NOT NULL,
            uncompressed_size INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CHECK (encoding = 'zlib-json-v1')
        );

        CREATE TABLE IF NOT EXISTS save_turn_snapshots (
            id TEXT PRIMARY KEY,
            save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
            message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
            parent_snapshot_id TEXT REFERENCES save_turn_snapshots(id)
                ON DELETE SET NULL,
            root_manifest_hash TEXT NOT NULL
                REFERENCES save_snapshot_objects(object_hash),
            context_revision INTEGER NOT NULL DEFAULT 0,
            reason TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_save_turn_snapshots_save_message_created
        ON save_turn_snapshots(save_id, message_id, created_at);

        CREATE INDEX IF NOT EXISTS idx_save_turn_snapshots_save_created
        ON save_turn_snapshots(save_id, created_at);

        CREATE INDEX IF NOT EXISTS idx_save_turn_snapshots_root_manifest
        ON save_turn_snapshots(root_manifest_hash);
        """
    )


def _ensure_context_observation_schema(connection: sqlite3.Connection) -> None:
    _execute_schema_script(
        connection,
        """
        CREATE TABLE IF NOT EXISTS context_observations (
            id TEXT PRIMARY KEY,
            save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
            observation_type TEXT NOT NULL,
            claim TEXT NOT NULL,
            evidence_quote TEXT NOT NULL DEFAULT '',
            source_message_ids_json TEXT NOT NULL DEFAULT '[]',
            scope TEXT NOT NULL DEFAULT 'turn',
            status TEXT NOT NULL DEFAULT 'pending',
            confidence REAL NOT NULL DEFAULT 0.0,
            tags_json TEXT NOT NULL DEFAULT '[]',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            archived_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_context_observations_save_status
        ON context_observations(save_id, status, archived_at);

        CREATE INDEX IF NOT EXISTS idx_context_observations_save_created
        ON context_observations(save_id, created_at, id);
        """
    )


def _ensure_context_observation_curation_schema(
    connection: sqlite3.Connection,
) -> None:
    if not _table_exists(connection, "context_observations") or not _table_exists(
        connection,
        "saves",
    ):
        return
    _execute_schema_script(
        connection,
        """
        CREATE TABLE IF NOT EXISTS context_observation_curation_state (
            observation_id TEXT PRIMARY KEY
                REFERENCES context_observations(id) ON DELETE CASCADE,
            save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            next_eligible_at TEXT,
            lease_token TEXT,
            lease_until TEXT,
            last_error TEXT,
            terminal_outcome TEXT,
            completed_at TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_context_observation_curation_due
        ON context_observation_curation_state(
            save_id, terminal_outcome, next_eligible_at, lease_until
        );
        """,
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO context_observation_curation_state(
            observation_id, save_id, terminal_outcome, completed_at
        )
        SELECT
            id,
            save_id,
            CASE WHEN status = 'pending' THEN NULL ELSE status END,
            CASE WHEN status = 'pending' THEN NULL ELSE updated_at END
        FROM context_observations
        """
    )
    for observation_id, observation_type, metadata_json in connection.execute(
        "SELECT id, observation_type, metadata_json FROM context_observations"
    ).fetchall():
        original_type = str(observation_type)
        normalized_type = normalize_observation_type(original_type)
        if normalized_type == original_type:
            continue
        try:
            metadata = json.loads(str(metadata_json))
        except (TypeError, ValueError):
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.setdefault("original_observation_type", original_type)
        connection.execute(
            """
            UPDATE context_observations
            SET observation_type = ?, metadata_json = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                normalized_type,
                json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                observation_id,
            ),
        )


def _ensure_message_scene_presence_schema(connection: sqlite3.Connection) -> None:
    _execute_schema_script(
        connection,
        """
        CREATE TABLE IF NOT EXISTS message_scene_presence (
            id TEXT PRIMARY KEY,
            save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
            message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            character_id TEXT NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
            source TEXT NOT NULL DEFAULT 'context_snapshot',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(save_id, message_id, character_id)
        );

        CREATE INDEX IF NOT EXISTS idx_message_scene_presence_save_message
        ON message_scene_presence(save_id, message_id);
        """
    )


def _ensure_message_action_choices_schema(connection: sqlite3.Connection) -> None:
    _execute_schema_script(
        connection,
        """
        CREATE TABLE IF NOT EXISTS message_action_choices (
            id TEXT PRIMARY KEY,
            save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
            message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            body TEXT NOT NULL,
            provider TEXT NOT NULL DEFAULT '',
            model TEXT NOT NULL DEFAULT '',
            content_rating TEXT NOT NULL DEFAULT 'unclassified',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(message_id, ordinal)
        );

        CREATE INDEX IF NOT EXISTS idx_message_action_choices_save_message_ordinal
        ON message_action_choices(save_id, message_id, ordinal);
        """
    )
    _add_column_if_missing(
        connection,
        "message_action_choices",
        "content_rating",
        "TEXT NOT NULL DEFAULT 'unclassified'",
    )


def _ensure_generated_content_rating_schema(
    connection: sqlite3.Connection,
) -> None:
    _ensure_message_action_choices_schema(connection)
    _add_column_if_missing(
        connection,
        "summaries",
        "content_rating",
        "TEXT NOT NULL DEFAULT 'unclassified'",
    )


def _normalize_legacy_action_choice_scenarios(
    connection: sqlite3.Connection,
) -> None:
    if not _table_exists(connection, "scenarios"):
        return
    scenario_columns = _column_names(connection, "scenarios")
    if not {"id", "type", "content_json"} <= scenario_columns:
        return
    rows = connection.execute(
        """
        SELECT id, content_json
        FROM scenarios
        WHERE type = 'choose_your_own_adventure'
        """
    ).fetchall()
    for scenario_id, content_json in rows:
        try:
            content = json.loads(str(content_json or "{}"))
        except json.JSONDecodeError:
            content = {}
        if not isinstance(content, dict):
            content = {}
        content["action_choices_enabled"] = True
        connection.execute(
            """
            UPDATE scenarios
            SET type = 'full_roleplay',
                content_json = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                json.dumps(content, sort_keys=True, separators=(",", ":")),
                scenario_id,
            ),
        )


def _ensure_hot_narration_query_indexes(connection: sqlite3.Connection) -> None:
    _create_index_if_columns_exist(
        connection,
        "jobs",
        {"status", "type", "save_id", "completed_at"},
        """
        CREATE INDEX IF NOT EXISTS idx_jobs_status_type_save_completed
        ON jobs(status, type, save_id, completed_at)
        """,
    )
    _create_index_if_columns_exist(
        connection,
        "jobs",
        {"save_id", "status", "completed_at"},
        """
        CREATE INDEX IF NOT EXISTS idx_jobs_save_status_completed
        ON jobs(save_id, status, completed_at)
        """,
    )
    _create_index_if_columns_exist(
        connection,
        "messages",
        {"save_id", "deleted_at"},
        """
        CREATE INDEX IF NOT EXISTS idx_messages_save_active_row_order
        ON messages(save_id)
        WHERE deleted_at IS NULL
        """,
    )
    _create_index_if_columns_exist(
        connection,
        "messages",
        {"save_id", "role", "created_at", "deleted_at"},
        """
        CREATE INDEX IF NOT EXISTS idx_messages_save_role_active_created
        ON messages(save_id, role, created_at)
        WHERE deleted_at IS NULL
        """,
    )
    _create_index_if_columns_exist(
        connection,
        "world_state",
        {"save_id", "key", "archived_at"},
        """
        CREATE INDEX IF NOT EXISTS idx_world_state_save_active_key
        ON world_state(save_id, key)
        WHERE archived_at IS NULL
        """,
    )
    _create_index_if_columns_exist(
        connection,
        "context_sources",
        {"save_id", "source_type", "title", "created_at", "archived_at"},
        """
        CREATE INDEX IF NOT EXISTS idx_context_sources_save_active_type_title_created
        ON context_sources(save_id, source_type, title, created_at)
        WHERE archived_at IS NULL
        """,
    )
    _create_index_if_columns_exist(
        connection,
        "context_observations",
        {"save_id", "created_at", "archived_at"},
        """
        CREATE INDEX IF NOT EXISTS idx_context_observations_save_active_created
        ON context_observations(save_id, created_at)
        WHERE archived_at IS NULL
        """,
    )
    _create_index_if_columns_exist(
        connection,
        "character_knowledge_edges",
        {
            "save_id",
            "target_type",
            "target_id",
            "archived_at",
            "character_id",
        },
        """
        CREATE INDEX IF NOT EXISTS idx_knowledge_edges_save_target_character
        ON character_knowledge_edges(
            save_id,
            target_type,
            target_id,
            archived_at,
            character_id
        )
        """,
    )
    _create_index_if_columns_exist(
        connection,
        "entity_links",
        {
            "save_id",
            "target_type",
            "target_id",
            "relation",
            "entity_type",
            "entity_id",
        },
        """
        CREATE INDEX IF NOT EXISTS idx_entity_links_save_target_relation_entity
        ON entity_links(
            save_id,
            target_type,
            target_id,
            relation,
            entity_type,
            entity_id
        )
        """,
    )
    _create_index_if_columns_exist(
        connection,
        "locations",
        {"save_id", "name", "created_at", "archived_at"},
        """
        CREATE INDEX IF NOT EXISTS idx_locations_save_active_name_created
        ON locations(save_id, name, created_at)
        WHERE archived_at IS NULL
        """,
    )
    _create_index_if_columns_exist(
        connection,
        "characters",
        {"save_id", "name", "created_at", "archived_at"},
        """
        CREATE INDEX IF NOT EXISTS idx_characters_save_active_name_created
        ON characters(save_id, name, created_at)
        WHERE archived_at IS NULL
        """,
    )
    _create_index_if_columns_exist(
        connection,
        "characters",
        {"save_id", "protected_from_maintenance", "archived_at"},
        """
        CREATE INDEX IF NOT EXISTS idx_characters_save_protected_active
        ON characters(save_id, protected_from_maintenance)
        WHERE archived_at IS NULL
        """,
    )
    _create_index_if_columns_exist(
        connection,
        "active_threads",
        {"save_id", "priority", "created_at", "archived_at"},
        """
        CREATE INDEX IF NOT EXISTS idx_active_threads_save_active_priority_created
        ON active_threads(save_id, priority DESC, created_at)
        WHERE archived_at IS NULL
        """,
    )
    _create_index_if_columns_exist(
        connection,
        "state_changes",
        {"save_id", "created_at"},
        """
        CREATE INDEX IF NOT EXISTS idx_state_changes_save_created
        ON state_changes(save_id, created_at)
        """,
    )
    _create_index_if_columns_exist(
        connection,
        "memories",
        {"save_id", "created_at", "archived_at"},
        """
        CREATE INDEX IF NOT EXISTS idx_memories_save_active_created
        ON memories(save_id, created_at)
        WHERE archived_at IS NULL
        """,
    )
    _create_index_if_columns_exist(
        connection,
        "memories",
        {"save_id", "claim_fingerprint", "archived_at"},
        """
        CREATE UNIQUE INDEX IF NOT EXISTS
            idx_memories_save_claim_fingerprint_active
        ON memories(save_id, claim_fingerprint)
        WHERE archived_at IS NULL AND claim_fingerprint != ''
        """,
    )
    _create_index_if_columns_exist(
        connection,
        "scheduled_tasks",
        {"task_type", "enabled", "next_run_at", "lease_until"},
        """
        CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_type_due
        ON scheduled_tasks(task_type, enabled, next_run_at, lease_until)
        """,
    )
    _create_index_if_columns_exist(
        connection,
        "summaries",
        {"save_id", "created_at", "archived_at"},
        """
        CREATE INDEX IF NOT EXISTS idx_summaries_save_active_created
        ON summaries(save_id, created_at)
        WHERE archived_at IS NULL
        """,
    )
    _create_index_if_columns_exist(
        connection,
        "media_assets",
        {"save_id", "created_at", "archived_at"},
        """
        CREATE INDEX IF NOT EXISTS idx_media_assets_save_active_created
        ON media_assets(save_id, created_at)
        WHERE archived_at IS NULL
        """,
    )
    _create_index_if_columns_exist(
        connection,
        "media_assets",
        {"save_id", "source_message_id", "type", "archived_at"},
        """
        CREATE INDEX IF NOT EXISTS idx_media_assets_save_source_type_active
        ON media_assets(save_id, source_message_id, type)
        WHERE archived_at IS NULL
        """,
    )


def _ensure_context_source_fts_schema(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "context_sources"):
        return
    _drop_context_source_fts_triggers(connection)
    connection.execute("DROP TABLE IF EXISTS context_source_fts")
    connection.execute(
        """
        CREATE VIRTUAL TABLE context_source_fts
        USING fts5(
            title,
            body,
            content='context_sources',
            content_rowid='rowid'
        )
        """
    )
    connection.execute(
        """
        INSERT INTO context_source_fts(rowid, title, body)
        SELECT rowid, title, body
        FROM context_sources
        WHERE archived_at IS NULL
        """
    )
    _execute_schema_script(
        connection,
        """
        CREATE TRIGGER IF NOT EXISTS context_sources_fts_after_insert
        AFTER INSERT ON context_sources
        WHEN NEW.archived_at IS NULL
        BEGIN
            INSERT INTO context_source_fts(rowid, title, body)
            VALUES (NEW.rowid, NEW.title, NEW.body);
        END;

        CREATE TRIGGER IF NOT EXISTS context_sources_fts_after_update
        AFTER UPDATE OF title, body, archived_at ON context_sources
        BEGIN
            INSERT INTO context_source_fts(
                context_source_fts, rowid, title, body
            )
            SELECT 'delete', OLD.rowid, OLD.title, OLD.body
            WHERE OLD.archived_at IS NULL;

            INSERT INTO context_source_fts(rowid, title, body)
            SELECT NEW.rowid, NEW.title, NEW.body
            WHERE NEW.archived_at IS NULL;
        END;

        CREATE TRIGGER IF NOT EXISTS context_sources_fts_after_delete
        AFTER DELETE ON context_sources
        BEGIN
            INSERT INTO context_source_fts(
                context_source_fts, rowid, title, body
            )
            SELECT 'delete', OLD.rowid, OLD.title, OLD.body
            WHERE OLD.archived_at IS NULL;
        END;
        """
    )


def _drop_context_source_fts_triggers(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'trigger'
          AND name GLOB 'context_sources_fts_after_*'
        """
    ).fetchall()
    for (name,) in rows:
        connection.execute(f"DROP TRIGGER IF EXISTS {_quote_identifier(name)}")


def _create_index_if_columns_exist(
    connection: sqlite3.Connection,
    table_name: str,
    column_names: set[str],
    statement: str,
) -> None:
    if column_names <= _column_names(connection, table_name):
        connection.execute(statement)


def _ensure_scoped_settings_schema(connection: sqlite3.Connection) -> None:
    _execute_schema_script(
        connection,
        """
        CREATE TABLE IF NOT EXISTS scoped_settings (
            scope TEXT NOT NULL,
            scope_id TEXT NOT NULL DEFAULT '',
            key TEXT NOT NULL,
            value_json TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(scope, scope_id, key)
        );

        CREATE INDEX IF NOT EXISTS idx_scoped_settings_scope_key
        ON scoped_settings(scope, key);
        """
    )


def _migrate_app_settings_to_scoped_settings(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "app_settings"):
        return
    _ensure_scoped_settings_schema(connection)
    rows = connection.execute(
        "SELECT key, value_json, updated_at FROM app_settings ORDER BY key"
    ).fetchall()
    for key, value_json, updated_at in rows:
        scope, scope_id, scoped_key = _legacy_app_setting_scope(key)
        connection.execute(
            """
            INSERT INTO scoped_settings(scope, scope_id, key, value_json, updated_at)
            VALUES (?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
            ON CONFLICT(scope, scope_id, key) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = excluded.updated_at
            """,
            (scope, scope_id, scoped_key, value_json, updated_at),
        )


def _legacy_app_setting_scope(key: str) -> tuple[str, str, str]:
    for prefix, scope, scoped_key in (
        ("image_style_preset:save:", "save", "image_style_preset"),
        (
            "scenario_evolution_turn_interval:save:",
            "save",
            "scenario_evolution_turn_interval",
        ),
        (
            "scenario_evolution_turn_interval:scenario:",
            "scenario",
            "scenario_evolution_turn_interval",
        ),
    ):
        if key.startswith(prefix):
            return scope, key.removeprefix(prefix), scoped_key
    return "global", "", key


def _ensure_player_character_schema(connection: sqlite3.Connection) -> None:
    _add_column_if_missing(
        connection,
        "characters",
        "is_player_character",
        "INTEGER NOT NULL DEFAULT 0",
    )
    if not _table_exists(connection, "characters"):
        return
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS
            idx_characters_single_active_player_character
        ON characters(save_id)
        WHERE is_player_character = 1 AND archived_at IS NULL
        """
    )
    _backfill_player_characters_from_scenarios(connection)


def _ensure_character_agency_schema(connection: sqlite3.Connection) -> None:
    for column_name in (
        "goals",
        "motivations",
        "current_intent",
        "boundaries",
        "attitude_toward_player",
        "cooperation_conditions",
    ):
        _add_column_if_missing(
            connection,
            "characters",
            column_name,
            "TEXT NOT NULL DEFAULT ''",
        )


def _ensure_character_age_schema(connection: sqlite3.Connection) -> None:
    _add_column_if_missing(
        connection,
        "characters",
        "age",
        "TEXT NOT NULL DEFAULT ''",
    )


def _ensure_character_contact_name_schema(connection: sqlite3.Connection) -> None:
    _add_column_if_missing(
        connection,
        "characters",
        "contact_name",
        "TEXT NOT NULL DEFAULT ''",
    )


def _ensure_character_texting_style_schema(connection: sqlite3.Connection) -> None:
    _add_column_if_missing(
        connection,
        "characters",
        "texting_style",
        "TEXT NOT NULL DEFAULT ''",
    )


def _ensure_character_current_clothing_schema(connection: sqlite3.Connection) -> None:
    _add_column_if_missing(
        connection,
        "characters",
        "current_clothing",
        "TEXT NOT NULL DEFAULT ''",
    )


def _ensure_scene_world_time_schema(connection: sqlite3.Connection) -> None:
    _add_column_if_missing(
        connection,
        "scene_snapshots",
        "time_of_day",
        "TEXT NOT NULL DEFAULT ''",
    )
    _add_column_if_missing(
        connection,
        "scene_snapshots",
        "day_of_week",
        "TEXT NOT NULL DEFAULT ''",
    )
    _add_column_if_missing(
        connection,
        "scene_snapshots",
        "world_day_index",
        "INTEGER",
    )
    _add_column_if_missing(
        connection,
        "scene_snapshots",
        "world_time_day_index",
        "INTEGER",
    )
    _add_column_if_missing(
        connection,
        "scene_snapshots",
        "world_time_day_label",
        "TEXT NOT NULL DEFAULT ''",
    )
    _add_column_if_missing(
        connection,
        "scene_snapshots",
        "world_time_phase",
        "TEXT NOT NULL DEFAULT ''",
    )
    _add_column_if_missing(
        connection,
        "scene_snapshots",
        "world_time_clock_minutes",
        "INTEGER",
    )
    _add_column_if_missing(
        connection,
        "scene_snapshots",
        "world_time_period_label",
        "TEXT NOT NULL DEFAULT ''",
    )
    _add_column_if_missing(
        connection,
        "scene_snapshots",
        "world_time_source_message_id",
        "TEXT REFERENCES messages(id)",
    )
    _add_column_if_missing(
        connection,
        "scene_snapshots",
        "world_time_confidence",
        "REAL",
    )
    _backfill_scene_world_time_schema(connection)


def _migrate_schema_61_to_62(connection: sqlite3.Connection) -> None:
    _ensure_scene_world_time_schema(connection)
    connection.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES (62)")


def _migrate_schema_62_to_63(connection: sqlite3.Connection) -> None:
    _ensure_character_text_activity_schema(connection)
    connection.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES (63)")


def _migrate_schema_63_to_64(connection: sqlite3.Connection) -> None:
    _add_column_if_missing(
        connection,
        "messages",
        "safety_transition",
        "TEXT NOT NULL DEFAULT ''",
    )
    connection.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES (64)")


def _migrate_schema_64_to_65(connection: sqlite3.Connection) -> None:
    _ensure_context_update_suggestion_review_schema(connection)
    _reject_orphaned_pending_context_suggestions(connection)
    connection.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES (65)")


def _migrate_schema_65_to_66(connection: sqlite3.Connection) -> None:
    _remove_retired_model_preferences(connection)
    _remove_retired_model_tasks_from_settings(connection)
    if _table_exists(connection, "jobs"):
        connection.execute("DELETE FROM jobs WHERE type = 'venice_character_import'")
    connection.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES (66)")


def _migrate_schema_66_to_67(connection: sqlite3.Connection) -> None:
    _strip_deprecated_scenario_character_sections(connection)
    connection.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES (67)")


def _strip_deprecated_scenario_character_sections(
    connection: sqlite3.Connection,
) -> None:
    for table_name in ("scenarios", "save_scenario_updates"):
        _strip_deprecated_scenario_character_sections_from_table(
            connection,
            table_name,
        )


def _strip_deprecated_scenario_character_sections_from_table(
    connection: sqlite3.Connection,
    table_name: str,
) -> None:
    if table_name not in {"scenarios", "save_scenario_updates"}:
        raise ValueError(f"Unsupported scenario content table: {table_name}")
    if not _table_exists(connection, table_name):
        return
    if not {"id", "content_json"} <= _column_names(connection, table_name):
        return
    rows = connection.execute(f"SELECT id, content_json FROM {table_name}").fetchall()
    for row_id, content_json in rows:
        try:
            content = json.loads(str(content_json or "{}"))
        except json.JSONDecodeError:
            continue
        if not isinstance(content, dict):
            continue
        updated = False
        faction_fragments = [_migration_stripped_text(content.get("factions"))]
        for key in _DEPRECATED_SCENARIO_CHARACTER_SECTION_KEYS:
            if key not in content:
                continue
            value = content.pop(key)
            updated = True
            if key in _DEPRECATED_SCENARIO_FACTION_APPEND_KEYS:
                text = _migration_stripped_text(value)
                if text:
                    faction_fragments.append(text)
        nonblank_factions = [fragment for fragment in faction_fragments if fragment]
        if nonblank_factions:
            merged_factions = "\n\n".join(nonblank_factions)
            if content.get("factions") != merged_factions:
                content["factions"] = merged_factions
                updated = True
        elif content.get("factions") == "":
            content.pop("factions", None)
            updated = True
        if not updated:
            continue
        connection.execute(
            f"UPDATE {table_name} SET content_json = ? WHERE id = ?",
            (
                json.dumps(content, sort_keys=True, separators=(",", ":")),
                row_id,
            ),
        )


def _migration_stripped_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _migrate_schema_67_to_68(connection: sqlite3.Connection) -> None:
    _add_column_if_missing(
        connection,
        "messages",
        "content_rating",
        "TEXT NOT NULL DEFAULT 'unrated'",
    )
    _add_column_if_missing(
        connection,
        "character_text_messages",
        "content_rating",
        "TEXT NOT NULL DEFAULT 'unrated'",
    )
    connection.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES (68)")


def _migrate_schema_68_to_69(connection: sqlite3.Connection) -> None:
    _ensure_generated_content_rating_schema(connection)
    connection.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES (69)")


def _migrate_schema_69_to_70(connection: sqlite3.Connection) -> None:
    _strip_deprecated_scenario_character_sections(connection)
    _add_column_if_missing(
        connection,
        "characters",
        "content_rating",
        "TEXT NOT NULL DEFAULT 'unclassified'",
    )
    connection.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES (70)")


def _migrate_schema_70_to_71(connection: sqlite3.Connection) -> None:
    _ensure_context_observation_curation_schema(connection)
    connection.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES (71)")


def _migrate_schema_71_to_72(connection: sqlite3.Connection) -> None:
    _add_column_if_missing(
        connection,
        "scene_snapshots",
        "scene_generation",
        "INTEGER NOT NULL DEFAULT 1",
    )
    for column_name, definition in (
        (
            "scene_snapshot_id",
            "TEXT REFERENCES scene_snapshots(id) ON DELETE SET NULL",
        ),
        ("scene_generation", "INTEGER"),
        ("created_turn_number", "INTEGER"),
        ("expires_after_turn_number", "INTEGER"),
    ):
        _add_column_if_missing(connection, "context_sources", column_name, definition)
    _add_column_if_missing(
        connection,
        "memories",
        "claim_fingerprint",
        "TEXT NOT NULL DEFAULT ''",
    )
    _add_column_if_missing(
        connection,
        "memories",
        "source_observation_ids_json",
        "TEXT NOT NULL DEFAULT '[]'",
    )
    if _table_exists(connection, "memories"):
        rows = connection.execute(
            "SELECT id, body FROM memories WHERE claim_fingerprint = ''"
        ).fetchall()
        for memory_id, body in rows:
            connection.execute(
                "UPDATE memories SET claim_fingerprint = ? WHERE id = ?",
                (_migration_claim_fingerprint(body), memory_id),
            )
        if _table_exists(connection, "context_observations"):
            observations_by_claim: dict[tuple[str, str], list[str]] = {}
            observation_rows = connection.execute(
                """
                SELECT id, save_id, claim, metadata_json
                FROM context_observations
                WHERE archived_at IS NULL AND status = 'accepted'
                ORDER BY created_at, rowid
                """
            ).fetchall()
            for observation_id, save_id, claim, metadata_json in observation_rows:
                try:
                    metadata = json.loads(metadata_json)
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(metadata, dict):
                    continue
                curation = metadata.get("curation")
                if (
                    not isinstance(curation, dict)
                    or curation.get("action") != "durable_memory"
                ):
                    continue
                memory_body = curation.get("memory_body")
                body = (
                    memory_body.strip()
                    if isinstance(memory_body, str) and memory_body.strip()
                    else str(claim)
                )
                fingerprint = _migration_claim_fingerprint(body)
                if fingerprint:
                    observations_by_claim.setdefault(
                        (str(save_id), fingerprint),
                        [],
                    ).append(str(observation_id))
            memory_rows = connection.execute(
                """
                SELECT id, save_id, claim_fingerprint,
                       source_observation_ids_json
                FROM memories
                WHERE archived_at IS NULL
                """
            ).fetchall()
            for memory_id, save_id, fingerprint, existing_json in memory_rows:
                matching_ids = observations_by_claim.get(
                    (str(save_id), str(fingerprint)),
                    [],
                )
                if not matching_ids:
                    continue
                try:
                    existing = json.loads(existing_json)
                except (json.JSONDecodeError, TypeError):
                    existing = []
                existing_ids = (
                    [str(item) for item in existing if isinstance(item, str)]
                    if isinstance(existing, list)
                    else []
                )
                merged_ids = list(
                    dict.fromkeys((*existing_ids, *matching_ids))
                )[:_MAX_MEMORY_PROVENANCE_IDS]
                connection.execute(
                    """
                    UPDATE memories
                    SET source_observation_ids_json = ?
                    WHERE id = ?
                    """,
                    (
                        json.dumps(merged_ids, separators=(",", ":")),
                        memory_id,
                    ),
                )
        duplicate_rows = connection.execute(
            """
            SELECT id, save_id, tags_json, importance, source_message_id,
                   source_message_ids_json, claim_fingerprint,
                   source_observation_ids_json
            FROM memories
            WHERE archived_at IS NULL AND claim_fingerprint != ''
            ORDER BY created_at, rowid
            """
        ).fetchall()
        grouped_rows: dict[tuple[str, str], list[tuple[object, ...]]] = {}
        for row in duplicate_rows:
            grouped_rows.setdefault(
                (str(row[1]), str(row[6])),
                [],
            ).append(row)
        for (save_id, _fingerprint), group in grouped_rows.items():
            if len(group) < 2:
                continue
            keeper = group[0]
            tags: list[str] = []
            source_message_ids: list[str] = []
            source_observation_ids: list[str] = []
            source_message_id: str | None = None
            importance = 0.0
            for row in group:
                importance = max(importance, float(str(row[3])))
                if source_message_id is None and row[4]:
                    source_message_id = str(row[4])
                for raw_json, target in (
                    (row[2], tags),
                    (row[5], source_message_ids),
                    (row[7], source_observation_ids),
                ):
                    try:
                        values = json.loads(str(raw_json))
                    except (json.JSONDecodeError, TypeError):
                        values = []
                    if isinstance(values, list):
                        target.extend(
                            str(value)
                            for value in values
                            if isinstance(value, str) and value
                        )
                if row[4]:
                    source_message_ids.append(str(row[4]))
            connection.execute(
                """
                UPDATE memories
                SET tags_json = ?, importance = ?, source_message_id = ?,
                    source_message_ids_json = ?,
                    source_observation_ids_json = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND save_id = ?
                """,
                (
                    json.dumps(list(dict.fromkeys(tags)), separators=(",", ":")),
                    importance,
                    source_message_id,
                    json.dumps(
                        list(dict.fromkeys(source_message_ids))[
                            :_MAX_MEMORY_PROVENANCE_IDS
                        ],
                        separators=(",", ":"),
                    ),
                    json.dumps(
                        list(dict.fromkeys(source_observation_ids))[
                            :_MAX_MEMORY_PROVENANCE_IDS
                        ],
                        separators=(",", ":"),
                    ),
                    keeper[0],
                    save_id,
                ),
            )
            duplicate_ids = [str(row[0]) for row in group[1:]]
            for duplicate_id in duplicate_ids:
                _remap_migrated_memory_references(
                    connection,
                    save_id=save_id,
                    duplicate_id=duplicate_id,
                    keeper_id=str(keeper[0]),
                )
            connection.execute(
                f"""
                UPDATE memories
                SET archived_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE save_id = ?
                  AND id IN ({_migration_placeholders(len(duplicate_ids))})
                """,
                (save_id, *duplicate_ids),
            )
            if _table_exists(connection, "context_sources"):
                connection.execute(
                    f"""
                    UPDATE context_sources
                    SET archived_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE save_id = ?
                      AND source_type = 'memory'
                      AND source_id IN ({_migration_placeholders(len(duplicate_ids))})
                    """,
                    (save_id, *duplicate_ids),
                )
        connection.execute(
            "DROP INDEX IF EXISTS idx_memories_save_claim_fingerprint_active"
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
                idx_memories_save_claim_fingerprint_active
            ON memories(save_id, claim_fingerprint)
            WHERE archived_at IS NULL AND claim_fingerprint != ''
            """
        )
    connection.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES (72)")


def _remap_migrated_memory_references(
    connection: sqlite3.Connection,
    *,
    save_id: str,
    duplicate_id: str,
    keeper_id: str,
) -> None:
    if _table_exists(connection, "character_knowledge_edges"):
        _merge_migrated_memory_knowledge_edge_conflicts(
            connection,
            save_id=save_id,
            duplicate_id=duplicate_id,
            keeper_id=keeper_id,
        )
        connection.execute(
            """
            DELETE FROM character_knowledge_edges AS keeper_edge
            WHERE keeper_edge.save_id = ?
              AND keeper_edge.target_type IN ('memory', 'memories')
              AND keeper_edge.target_id = ?
              AND keeper_edge.archived_at IS NOT NULL
              AND EXISTS (
                  SELECT 1
                  FROM character_knowledge_edges AS duplicate_edge
                  WHERE duplicate_edge.save_id = keeper_edge.save_id
                    AND duplicate_edge.character_id = keeper_edge.character_id
                    AND duplicate_edge.target_type = keeper_edge.target_type
                    AND duplicate_edge.target_id = ?
                    AND duplicate_edge.archived_at IS NULL
              )
            """,
            (save_id, keeper_id, duplicate_id),
        )
        connection.execute(
            """
            UPDATE OR IGNORE character_knowledge_edges
            SET target_id = ?, updated_at = CURRENT_TIMESTAMP
            WHERE save_id = ?
              AND target_type IN ('memory', 'memories')
              AND target_id = ?
            """,
            (keeper_id, save_id, duplicate_id),
        )
        connection.execute(
            """
            DELETE FROM character_knowledge_edges
            WHERE save_id = ?
              AND target_type IN ('memory', 'memories')
              AND target_id = ?
            """,
            (save_id, duplicate_id),
        )
    if _table_exists(connection, "context_sources"):
        source_rows = connection.execute(
            """
            SELECT source_id, metadata_json, token_estimate
            FROM context_sources
            WHERE save_id = ? AND source_type = 'memory'
              AND source_id IN (?, ?) AND archived_at IS NULL
            """,
            (save_id, keeper_id, duplicate_id),
        ).fetchall()
        sources_by_id = {str(row[0]): row for row in source_rows}
        keeper_source = sources_by_id.get(keeper_id)
        duplicate_source = sources_by_id.get(duplicate_id)
        if keeper_source is not None and duplicate_source is not None:
            merged_metadata = merge_context_source_metadata(
                keeper_source[1],
                duplicate_source[1],
            )
            connection.execute(
                """
                UPDATE context_sources
                SET metadata_json = ?, token_estimate = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE save_id = ? AND source_type = 'memory'
                  AND source_id = ? AND archived_at IS NULL
                """,
                (
                    json.dumps(
                        merged_metadata,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    max(
                        int(keeper_source[2] or 0),
                        int(duplicate_source[2] or 0),
                    ),
                    save_id,
                    keeper_id,
                ),
            )
        connection.execute(
            """
            DELETE FROM context_sources AS keeper_source
            WHERE keeper_source.save_id = ?
              AND keeper_source.source_type = 'memory'
              AND keeper_source.source_id = ?
              AND keeper_source.archived_at IS NOT NULL
              AND EXISTS (
                  SELECT 1
                  FROM context_sources AS duplicate_source
                  WHERE duplicate_source.save_id = keeper_source.save_id
                    AND duplicate_source.source_type = 'memory'
                    AND duplicate_source.source_id = ?
                    AND duplicate_source.archived_at IS NULL
              )
            """,
            (save_id, keeper_id, duplicate_id),
        )
        connection.execute(
            """
            UPDATE OR IGNORE context_sources
            SET source_id = ?, updated_at = CURRENT_TIMESTAMP
            WHERE save_id = ?
              AND source_type = 'memory'
              AND source_id = ?
            """,
            (keeper_id, save_id, duplicate_id),
        )
    if _table_exists(connection, "entity_links"):
        connection.execute(
            """
            UPDATE OR IGNORE entity_links
            SET entity_id = ?
            WHERE save_id = ?
              AND entity_type IN ('memory', 'memories')
              AND entity_id = ?
            """,
            (keeper_id, save_id, duplicate_id),
        )
        connection.execute(
            """
            DELETE FROM entity_links
            WHERE save_id = ?
              AND entity_type IN ('memory', 'memories')
              AND entity_id = ?
            """,
            (save_id, duplicate_id),
        )
        connection.execute(
            """
            UPDATE OR IGNORE entity_links
            SET target_id = ?
            WHERE save_id = ?
              AND target_type IN ('memory', 'memories')
              AND target_id = ?
            """,
            (keeper_id, save_id, duplicate_id),
        )
        connection.execute(
            """
            DELETE FROM entity_links
            WHERE save_id = ?
              AND target_type IN ('memory', 'memories')
              AND target_id = ?
            """,
            (save_id, duplicate_id),
        )
    for table_name in ("context_update_suggestions", "context_update_audit"):
        if not _table_exists(connection, table_name):
            continue
        connection.execute(
            f"""
            UPDATE {table_name}
            SET entity_id = ?
            WHERE save_id = ?
              AND entity_type = 'memory'
              AND entity_id = ?
            """,
            (keeper_id, save_id, duplicate_id),
        )
    if _table_exists(connection, "character_text_provenance"):
        connection.execute(
            """
            UPDATE character_text_provenance
            SET target_id = ?
            WHERE save_id = ?
              AND target_type IN ('memory', 'memories')
              AND target_id = ?
            """,
            (keeper_id, save_id, duplicate_id),
        )
    _remap_migrated_memory_proactive_triggers(
        connection,
        save_id=save_id,
        duplicate_id=duplicate_id,
        keeper_id=keeper_id,
    )


def _remap_migrated_memory_proactive_triggers(
    connection: sqlite3.Connection,
    *,
    save_id: str,
    duplicate_id: str,
    keeper_id: str,
) -> None:
    if not _table_exists(connection, "character_text_proactive_triggers"):
        return
    rows = connection.execute(
        """
        SELECT id, character_id, trigger_key, thread_id, text_message_id,
               source_type, source_id, source_message_id, reason
        FROM character_text_proactive_triggers
        WHERE save_id = ?
          AND (
                (
                    source_type IN ('memory', 'memories')
                    AND source_id = ?
                )
                OR trigger_key = 'memory:' || ?
                OR instr(trigger_key, 'memory:' || ? || ':') = 1
                OR trigger_key = 'memories:' || ?
                OR instr(trigger_key, 'memories:' || ? || ':') = 1
              )
        ORDER BY rowid
        """,
        (
            save_id,
            duplicate_id,
            duplicate_id,
            duplicate_id,
            duplicate_id,
            duplicate_id,
        ),
    ).fetchall()
    for row in rows:
        (
            trigger_id,
            character_id,
            trigger_key,
            thread_id,
            text_message_id,
            source_type,
            source_id,
            source_message_id,
            reason,
        ) = row
        key_parts = str(trigger_key).split(":")
        if (
            len(key_parts) >= 2
            and key_parts[0] in {"memory", "memories"}
            and key_parts[1] == duplicate_id
        ):
            key_parts[1] = keeper_id
        remapped_key = ":".join(key_parts)
        remapped_source_id = (
            keeper_id
            if source_type in {"memory", "memories"}
            and source_id == duplicate_id
            else source_id
        )
        existing = connection.execute(
            """
            SELECT id, thread_id, text_message_id, source_type, source_id,
                   source_message_id, reason
            FROM character_text_proactive_triggers
            WHERE save_id = ? AND character_id = ? AND trigger_key = ?
              AND id != ?
            LIMIT 1
            """,
            (save_id, character_id, remapped_key, trigger_id),
        ).fetchone()
        if existing is None:
            connection.execute(
                """
                UPDATE character_text_proactive_triggers
                SET trigger_key = ?, source_id = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (remapped_key, remapped_source_id, trigger_id),
            )
            continue
        connection.execute(
            """
            UPDATE character_text_proactive_triggers
            SET thread_id = COALESCE(?, thread_id),
                text_message_id = COALESCE(?, text_message_id),
                source_type = COALESCE(NULLIF(?, ''), source_type),
                source_id = COALESCE(NULLIF(?, ''), source_id),
                source_message_id = COALESCE(?, source_message_id),
                reason = COALESCE(NULLIF(?, ''), reason),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                thread_id,
                text_message_id,
                source_type,
                remapped_source_id,
                source_message_id,
                reason,
                existing[0],
            ),
        )
        connection.execute(
            "DELETE FROM character_text_proactive_triggers WHERE id = ?",
            (trigger_id,),
        )


def _merge_migrated_memory_knowledge_edge_conflicts(
    connection: sqlite3.Connection,
    *,
    save_id: str,
    duplicate_id: str,
    keeper_id: str,
) -> None:
    rows = connection.execute(
        """
        SELECT
            keeper_edge.id, keeper_edge.knowledge_state,
            keeper_edge.acquisition_method, keeper_edge.confidence,
            keeper_edge.source_message_id,
            keeper_edge.source_message_ids_json, keeper_edge.evidence_quote,
            duplicate_edge.id, duplicate_edge.knowledge_state,
            duplicate_edge.acquisition_method, duplicate_edge.confidence,
            duplicate_edge.source_message_id,
            duplicate_edge.source_message_ids_json,
            duplicate_edge.evidence_quote
        FROM character_knowledge_edges AS keeper_edge
        JOIN character_knowledge_edges AS duplicate_edge
          ON duplicate_edge.save_id = keeper_edge.save_id
         AND duplicate_edge.character_id = keeper_edge.character_id
         AND duplicate_edge.target_type = keeper_edge.target_type
        WHERE keeper_edge.save_id = ?
          AND keeper_edge.target_type IN ('memory', 'memories')
          AND keeper_edge.target_id = ?
          AND duplicate_edge.target_id = ?
          AND keeper_edge.archived_at IS NULL
          AND duplicate_edge.archived_at IS NULL
        """,
        (save_id, keeper_id, duplicate_id),
    ).fetchall()
    state_rank = {"knows": 0, "may_know": 1, "does_not_know": 2}
    for row in rows:
        duplicate_dominates = state_rank.get(str(row[8]), 1) > state_rank.get(
            str(row[1]),
            1,
        )
        dominant_offset = 7 if duplicate_dominates else 0
        source_ids: list[str] = []
        for raw_json in (row[5], row[12]):
            try:
                values = json.loads(str(raw_json))
            except (json.JSONDecodeError, TypeError):
                values = []
            if isinstance(values, list):
                source_ids.extend(
                    str(value)
                    for value in values
                    if isinstance(value, str) and value
                )
        source_ids = list(dict.fromkeys(source_ids))
        provenance_overflow = (
            len(source_ids) > _MAX_KNOWLEDGE_EDGE_SOURCE_MESSAGE_IDS
        )
        connection.execute(
            """
            UPDATE character_knowledge_edges
            SET knowledge_state = ?, acquisition_method = ?,
                confidence = ?, source_message_id = ?,
                source_message_ids_json = ?, evidence_quote = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                (
                    "does_not_know"
                    if provenance_overflow
                    else row[dominant_offset + 1]
                ),
                "unknown" if provenance_overflow else row[dominant_offset + 2],
                max(float(str(row[3])), float(str(row[10]))),
                None if provenance_overflow else row[dominant_offset + 4],
                json.dumps(
                    [] if provenance_overflow else source_ids,
                    separators=(",", ":"),
                ),
                (
                    "Provenance exceeded the safe bound."
                    if provenance_overflow
                    else row[dominant_offset + 6]
                ),
                row[0],
            ),
        )
        connection.execute(
            "DELETE FROM character_knowledge_edges WHERE id = ?",
            (row[7],),
        )


def _migration_claim_fingerprint(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    canonical = "".join(
        character if character.isalnum() else " "
        for character in text
    )
    canonical = " ".join(canonical.split())
    return sha256(canonical.encode("utf-8")).hexdigest() if canonical else ""


def _migration_placeholders(count: int) -> str:
    return ", ".join("?" for _ in range(count))
def _remove_retired_model_preferences(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "model_preferences"):
        return
    connection.execute(
        """
        DELETE FROM model_preferences
        WHERE task = 'chat_character_interaction'
           OR substr(task, 1, length('character_interaction_')) =
              'character_interaction_'
        """
    )


def _remove_retired_model_tasks_from_settings(
    connection: sqlite3.Connection,
) -> None:
    if _table_exists(connection, "app_settings"):
        _rewrite_retired_model_task_settings(
            connection,
            table="app_settings",
            row_id_columns=("key",),
        )
    if _table_exists(connection, "scoped_settings"):
        _rewrite_retired_model_task_settings(
            connection,
            table="scoped_settings",
            row_id_columns=("scope", "scope_id", "key"),
        )


def _rewrite_retired_model_task_settings(
    connection: sqlite3.Connection,
    *,
    table: str,
    row_id_columns: tuple[str, ...],
) -> None:
    id_list = ", ".join(row_id_columns)
    rows = connection.execute(
        f"""
        SELECT {id_list}, key, value_json
        FROM {table}
        WHERE key IN ('model_thinking_preferences', 'save_model_overrides')
        """
    ).fetchall()
    for row in rows:
        row_ids = row[: len(row_id_columns)]
        key = row[len(row_id_columns)]
        value_json = row[len(row_id_columns) + 1]
        try:
            value = json.loads(value_json)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict):
            continue
        if key == "model_thinking_preferences":
            sanitized = {
                task: config
                for task, config in value.items()
                if not is_retired_model_task(task)
            }
        else:
            sanitized = dict(value)
            for preference_key in ("preferences", "thinking"):
                preferences = sanitized.get(preference_key)
                if isinstance(preferences, dict):
                    sanitized[preference_key] = {
                        task: config
                        for task, config in preferences.items()
                        if not is_retired_model_task(task)
                    }
        if sanitized == value:
            continue
        where = " AND ".join(f"{column} = ?" for column in row_id_columns)
        connection.execute(
            f"UPDATE {table} SET value_json = ? WHERE {where}",
            (json.dumps(sanitized, sort_keys=True, separators=(",", ":")), *row_ids),
        )


def _ensure_context_update_suggestion_review_schema(
    connection: sqlite3.Connection,
) -> None:
    if not _table_exists(connection, "context_update_suggestions"):
        return
    _add_column_if_missing(
        connection,
        "context_update_suggestions",
        "review_attempt_count",
        "INTEGER NOT NULL DEFAULT 0",
    )
    _add_column_if_missing(
        connection,
        "context_update_suggestions",
        "next_review_at",
        "TEXT",
    )
    _add_column_if_missing(
        connection,
        "context_update_suggestions",
        "last_review_error",
        "TEXT",
    )
    _create_index_if_columns_exist(
        connection,
        "context_update_suggestions",
        {"status", "next_review_at", "save_id"},
        """
        CREATE INDEX IF NOT EXISTS idx_context_update_suggestions_review_due
        ON context_update_suggestions(status, next_review_at, save_id)
        """,
    )


def _reject_orphaned_pending_context_suggestions(
    connection: sqlite3.Connection,
) -> None:
    if not (
        _table_exists(connection, "context_update_suggestions")
        and _table_exists(connection, "characters")
        and _table_exists(connection, "active_threads")
    ):
        return
    reason = (
        "Suggestion rejected during review migration because its target no longer "
        "exists."
    )
    condition = """
        status = 'pending' AND (
            (entity_type = 'character' AND entity_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM characters
                WHERE characters.id = context_update_suggestions.entity_id
                  AND characters.save_id = context_update_suggestions.save_id
                  AND characters.archived_at IS NULL
            )) OR
            (entity_type = 'active_thread' AND entity_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM active_threads
                WHERE active_threads.id = context_update_suggestions.entity_id
                  AND active_threads.save_id = context_update_suggestions.save_id
                  AND active_threads.archived_at IS NULL
            )) OR EXISTS (
                SELECT 1 FROM json_each(
                    context_update_suggestions.source_message_ids_json
                )
                WHERE (
                    value LIKE 'character_text_message:%' AND NOT EXISTS (
                        SELECT 1 FROM character_text_messages
                        WHERE character_text_messages.id = substr(
                            value, length('character_text_message:') + 1
                        )
                          AND character_text_messages.save_id =
                              context_update_suggestions.save_id
                          AND character_text_messages.deleted_at IS NULL
                    )
                ) OR (
                    value NOT LIKE 'character_text_message:%' AND NOT EXISTS (
                        SELECT 1 FROM messages
                        WHERE messages.id = value
                          AND messages.save_id = context_update_suggestions.save_id
                          AND messages.deleted_at IS NULL
                    )
                )
            )
        )
    """
    connection.execute(
        f"""
        INSERT INTO context_update_audit(
            id, save_id, suggestion_id, operation, entity_type, entity_id,
            field_path, before_json, after_json, reason, confidence,
            source_message_ids_json
        )
        SELECT lower(hex(randomblob(16))), save_id, id,
               'agent_suggestion_preflight_reject', entity_type, entity_id,
               field_path, NULL, proposed_value_json, ?, confidence,
               source_message_ids_json
        FROM context_update_suggestions
        WHERE {condition}
        """,
        (reason,),
    )
    connection.execute(
        f"""
        UPDATE context_update_suggestions
        SET status = 'rejected', resolved_at = CURRENT_TIMESTAMP,
            next_review_at = NULL, last_review_error = ?
        WHERE {condition}
        """,
        (reason,),
    )


def _backfill_scene_world_time_schema(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "scene_snapshots"):
        return
    columns = _column_names(connection, "scene_snapshots")
    required = {
        "id",
        "in_world_time",
        "time_of_day",
        "day_of_week",
        "world_day_index",
        "source_message_id",
        "world_time_day_index",
        "world_time_day_label",
        "world_time_phase",
        "world_time_clock_minutes",
        "world_time_period_label",
        "world_time_source_message_id",
    }
    if not required <= columns:
        return
    rows = connection.execute(
        """
        SELECT id, in_world_time, time_of_day, day_of_week, world_day_index,
               source_message_id, world_time_day_index, world_time_day_label,
               world_time_phase, world_time_clock_minutes,
               world_time_period_label, world_time_source_message_id
        FROM scene_snapshots
        """
    ).fetchall()
    for row in rows:
        (
            snapshot_id,
            in_world_time,
            time_of_day,
            day_of_week,
            world_day_index,
            source_message_id,
            canonical_day_index,
            canonical_day_label,
            canonical_phase,
            canonical_clock_minutes,
            canonical_period_label,
            _canonical_source_message_id,
        ) = row
        has_canonical = any(
            value not in (None, "")
            for value in (
                canonical_day_index,
                canonical_day_label,
                canonical_phase,
                canonical_clock_minutes,
                canonical_period_label,
            )
        )
        if has_canonical:
            continue
        connection.execute(
            """
            UPDATE scene_snapshots
            SET world_time_day_index = ?,
                world_time_day_label = ?,
                world_time_phase = ?,
                world_time_clock_minutes = ?,
                world_time_source_message_id = ?
            WHERE id = ?
            """,
            (
                _migration_optional_nonnegative_int(world_day_index),
                _migration_day_label(day_of_week),
                _migration_phase(time_of_day)
                or _migration_phase_from_legacy_label(in_world_time),
                _migration_clock_minutes(in_world_time),
                source_message_id,
                snapshot_id,
            ),
        )


def _migration_optional_nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _migration_day_label(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.strip().split()).casefold()
    return text if text in _MIGRATION_DAY_OF_WEEK_VALUES else ""


def _migration_phase(value: object) -> str:
    if not isinstance(value, str):
        return ""
    stripped = value.strip()
    normalized = re.sub(r"[\s-]+", "_", stripped.casefold())
    normalized = _MIGRATION_TIME_OF_DAY_ALIASES.get(stripped.casefold(), normalized)
    return normalized if normalized in _MIGRATION_TIME_OF_DAY_VALUES else ""


def _migration_phase_from_legacy_label(value: object) -> str:
    direct = _migration_phase(value)
    if direct:
        return direct
    if not isinstance(value, str):
        return ""
    for parenthetical in re.findall(r"\(([^)]+)\)", value):
        phase = _migration_phase(parenthetical)
        if phase:
            return phase
    text = re.sub(r"[_-]+", " ", value.casefold())
    for phase in sorted(_MIGRATION_TIME_OF_DAY_VALUES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phase.replace('_', ' '))}\b", text):
            return phase
    for alias, phase in sorted(
        _MIGRATION_TIME_OF_DAY_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return phase
    return ""


def _migration_clock_minutes(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    for match in _MIGRATION_CLOCK_RE.finditer(text):
        meridiem = match.group(3)
        minute_text = match.group(2)
        if meridiem is None and minute_text is None:
            continue
        hour = int(match.group(1))
        minute = int(minute_text or "0")
        if meridiem is not None:
            meridiem_text = meridiem.replace(".", "").casefold()
            if meridiem_text == "am" and hour == 12:
                hour = 0
            elif meridiem_text == "pm" and hour != 12:
                hour += 12
        return hour * 60 + minute
    return None


def _ensure_dating_route_state_schema(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "saves"):
        return
    _execute_schema_script(
        connection,
        """
        CREATE TABLE IF NOT EXISTS dating_route_states (
            id TEXT PRIMARY KEY,
            save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
            player_character_id TEXT NOT NULL REFERENCES characters(id)
                ON DELETE CASCADE,
            npc_character_id TEXT NOT NULL REFERENCES characters(id)
                ON DELETE CASCADE,
            stage TEXT NOT NULL DEFAULT 'unmet',
            first_met_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
            first_met_world_day_index INTEGER,
            last_interaction_message_id TEXT REFERENCES messages(id)
                ON DELETE SET NULL,
            last_interaction_world_day_index INTEGER,
            completed_interactions INTEGER NOT NULL DEFAULT 0,
            dates_completed INTEGER NOT NULL DEFAULT 0,
            interest_level TEXT NOT NULL DEFAULT '',
            trust_level TEXT NOT NULL DEFAULT '',
            comfort_with_intimacy TEXT NOT NULL DEFAULT '',
            pacing_preference TEXT NOT NULL DEFAULT '',
            known_boundaries_json TEXT NOT NULL DEFAULT '[]',
            unresolved_questions_json TEXT NOT NULL DEFAULT '[]',
            next_reasonable_step TEXT NOT NULL DEFAULT '',
            source_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            archived_at TEXT,
            UNIQUE(save_id, player_character_id, npc_character_id)
        );

        CREATE INDEX IF NOT EXISTS idx_dating_route_states_save_active
        ON dating_route_states(save_id, archived_at, stage, updated_at);

        CREATE INDEX IF NOT EXISTS idx_dating_route_states_save_npc_active
        ON dating_route_states(save_id, npc_character_id, archived_at);
        """
    )


def _ensure_character_text_schema(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "saves"):
        return
    _execute_schema_script(
        connection,
        """
        CREATE TABLE IF NOT EXISTS character_text_threads (
            id TEXT PRIMARY KEY,
            save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
            character_id TEXT REFERENCES characters(id) ON DELETE CASCADE,
            title TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active',
            kind TEXT NOT NULL DEFAULT 'direct',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            archived_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_character_text_threads_save_active
        ON character_text_threads(save_id, archived_at, updated_at);

        CREATE TABLE IF NOT EXISTS character_text_thread_participants (
            id TEXT PRIMARY KEY,
            save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
            thread_id TEXT NOT NULL REFERENCES character_text_threads(id)
                ON DELETE CASCADE,
            character_id TEXT NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            archived_at TEXT,
            UNIQUE(thread_id, character_id)
        );

        CREATE INDEX IF NOT EXISTS idx_character_text_participants_thread
        ON character_text_thread_participants(thread_id, ordinal, character_id);

        CREATE TABLE IF NOT EXISTS character_text_messages (
            id TEXT PRIMARY KEY,
            save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
            thread_id TEXT NOT NULL REFERENCES character_text_threads(id)
                ON DELETE CASCADE,
            character_id TEXT REFERENCES characters(id) ON DELETE CASCADE,
            sender TEXT NOT NULL,
            body TEXT NOT NULL,
            sender_character_id TEXT REFERENCES characters(id) ON DELETE SET NULL,
            provider TEXT,
            model TEXT,
            token_estimate INTEGER,
            content_rating TEXT NOT NULL DEFAULT 'unclassified',
            delivery_status TEXT NOT NULL DEFAULT 'sent',
            delivery_error TEXT,
            delivery_job_id TEXT,
            delivery_attempt INTEGER NOT NULL DEFAULT 0,
            in_world_sent_at TEXT,
            delivered_at TEXT,
            read_at TEXT,
            reply_to_message_id TEXT REFERENCES character_text_messages(id)
                ON DELETE SET NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            deleted_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_character_text_messages_thread_created
        ON character_text_messages(thread_id, created_at, id);

        CREATE INDEX IF NOT EXISTS idx_character_text_messages_save_created
        ON character_text_messages(save_id, created_at, id);

        CREATE TABLE IF NOT EXISTS character_text_provenance (
            id TEXT PRIMARY KEY,
            save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
            thread_id TEXT NOT NULL REFERENCES character_text_threads(id)
                ON DELETE CASCADE,
            text_message_id TEXT NOT NULL REFERENCES character_text_messages(id)
                ON DELETE CASCADE,
            target_type TEXT NOT NULL,
            target_id TEXT NOT NULL,
            operation TEXT NOT NULL DEFAULT '',
            field_path TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_character_text_provenance_text
        ON character_text_provenance(text_message_id);

        CREATE INDEX IF NOT EXISTS idx_character_text_provenance_target
        ON character_text_provenance(save_id, target_type, target_id);
        """
    )
    _ensure_character_text_group_schema(connection)
    _add_column_if_missing(
        connection,
        "character_text_messages",
        "delivery_status",
        "TEXT NOT NULL DEFAULT 'sent'",
    )
    _add_column_if_missing(
        connection,
        "character_text_messages",
        "delivery_error",
        "TEXT",
    )
    _add_column_if_missing(
        connection,
        "character_text_messages",
        "delivery_job_id",
        "TEXT",
    )
    _add_column_if_missing(
        connection,
        "character_text_messages",
        "delivery_attempt",
        "INTEGER NOT NULL DEFAULT 0",
    )
    _ensure_character_text_thread_memory_schema(connection)
    _ensure_character_text_message_metadata_schema(connection)


def _ensure_character_text_activity_schema(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "saves"):
        return
    _ensure_character_text_schema(connection)
    _execute_schema_script(
        connection,
        """
        CREATE TABLE IF NOT EXISTS character_text_activity_events (
            id TEXT PRIMARY KEY,
            save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            thread_id TEXT NOT NULL REFERENCES character_text_threads(id)
                ON DELETE CASCADE,
            activity_type TEXT NOT NULL,
            text_message_id TEXT REFERENCES character_text_messages(id)
                ON DELETE CASCADE,
            read_count INTEGER NOT NULL DEFAULT 0,
            delivery_status TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(save_id, ordinal)
        );

        CREATE INDEX IF NOT EXISTS idx_character_text_activity_save_ordinal
        ON character_text_activity_events(save_id, ordinal);

        CREATE TABLE IF NOT EXISTS narrator_phone_activity_cursors (
            narrator_message_id TEXT PRIMARY KEY REFERENCES messages(id)
                ON DELETE CASCADE,
            save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
            last_activity_ordinal INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_narrator_phone_activity_cursors_save
        ON narrator_phone_activity_cursors(save_id, narrator_message_id);
        """,
    )


def _ensure_character_text_group_schema(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "character_text_threads"):
        return
    if _character_text_tables_need_group_rebuild(connection):
        _rebuild_character_text_tables_for_groups(connection)
    _add_column_if_missing(
        connection,
        "character_text_threads",
        "kind",
        "TEXT NOT NULL DEFAULT 'direct'",
    )
    _add_column_if_missing(
        connection,
        "character_text_messages",
        "sender_character_id",
        "TEXT REFERENCES characters(id) ON DELETE SET NULL",
    )
    _execute_schema_script(
        connection,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_character_text_threads_save_direct
        ON character_text_threads(save_id, character_id, kind)
        WHERE kind = 'direct' AND character_id IS NOT NULL;

        CREATE TABLE IF NOT EXISTS character_text_thread_participants (
            id TEXT PRIMARY KEY,
            save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
            thread_id TEXT NOT NULL REFERENCES character_text_threads(id)
                ON DELETE CASCADE,
            character_id TEXT NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            archived_at TEXT,
            UNIQUE(thread_id, character_id)
        );

        CREATE INDEX IF NOT EXISTS idx_character_text_participants_thread
        ON character_text_thread_participants(thread_id, ordinal, character_id);
        """
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO character_text_thread_participants(
            id, save_id, thread_id, character_id, ordinal, created_at, updated_at
        )
        SELECT
            lower(hex(randomblob(16))),
            save_id,
            id,
            character_id,
            0,
            COALESCE(created_at, CURRENT_TIMESTAMP),
            COALESCE(updated_at, CURRENT_TIMESTAMP)
        FROM character_text_threads
        WHERE kind = 'direct' AND character_id IS NOT NULL
        """
    )


def _character_text_tables_need_group_rebuild(
    connection: sqlite3.Connection,
) -> bool:
    thread_columns = {
        row[1]: row
        for row in connection.execute("PRAGMA table_info(character_text_threads)")
    }
    message_columns = {
        row[1]: row
        for row in connection.execute("PRAGMA table_info(character_text_messages)")
    }
    thread_character = thread_columns.get("character_id")
    message_character = message_columns.get("character_id")
    return (
        "kind" not in thread_columns
        or thread_character is None
        or bool(thread_character[3])
        or "sender_character_id" not in message_columns
        or message_character is None
        or bool(message_character[3])
    )


def _rebuild_character_text_tables_for_groups(
    connection: sqlite3.Connection,
) -> None:
    foreign_keys_enabled = bool(connection.execute("PRAGMA foreign_keys").fetchone()[0])
    legacy_alter_table = bool(
        connection.execute("PRAGMA legacy_alter_table").fetchone()[0]
    )
    connection.execute("PRAGMA foreign_keys = OFF")
    connection.execute("PRAGMA legacy_alter_table = ON")
    try:
        for index_name in (
            "idx_character_text_threads_save_active",
            "idx_character_text_threads_save_direct",
            "idx_character_text_messages_thread_created",
            "idx_character_text_messages_save_created",
            "idx_character_text_participants_thread",
        ):
            connection.execute(f"DROP INDEX IF EXISTS {index_name}")
        connection.execute("DROP TABLE IF EXISTS character_text_thread_participants")
        connection.execute(
            "ALTER TABLE character_text_messages RENAME TO character_text_messages_old"
        )
        connection.execute(
            "ALTER TABLE character_text_threads RENAME TO character_text_threads_old"
        )
        _execute_schema_script(
            connection,
            """
            CREATE TABLE character_text_threads (
                id TEXT PRIMARY KEY,
                save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
                character_id TEXT REFERENCES characters(id) ON DELETE CASCADE,
                title TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                kind TEXT NOT NULL DEFAULT 'direct',
                memory_body TEXT NOT NULL DEFAULT '',
                memory_message_count INTEGER NOT NULL DEFAULT 0,
                memory_updated_at TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                archived_at TEXT
            );

            CREATE TABLE character_text_thread_participants (
                id TEXT PRIMARY KEY,
                save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
                thread_id TEXT NOT NULL REFERENCES character_text_threads(id)
                    ON DELETE CASCADE,
                character_id TEXT NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                archived_at TEXT,
                UNIQUE(thread_id, character_id)
            );

            CREATE TABLE character_text_messages (
                id TEXT PRIMARY KEY,
                save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
                thread_id TEXT NOT NULL REFERENCES character_text_threads(id)
                    ON DELETE CASCADE,
                character_id TEXT REFERENCES characters(id) ON DELETE CASCADE,
                sender TEXT NOT NULL,
                body TEXT NOT NULL,
                sender_character_id TEXT REFERENCES characters(id) ON DELETE SET NULL,
                provider TEXT,
                model TEXT,
                token_estimate INTEGER,
                content_rating TEXT NOT NULL DEFAULT 'unclassified',
                delivery_status TEXT NOT NULL DEFAULT 'sent',
                delivery_error TEXT,
                delivery_job_id TEXT,
                delivery_attempt INTEGER NOT NULL DEFAULT 0,
                in_world_sent_at TEXT,
                delivered_at TEXT,
                read_at TEXT,
                reply_to_message_id TEXT REFERENCES character_text_messages(id)
                    ON DELETE SET NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                deleted_at TEXT
            );
            """
        )
        thread_kind_expr = _old_column_expr(
            connection,
            "character_text_threads_old",
            "kind",
            "'direct'",
        )
        thread_memory_body_expr = _old_column_expr(
            connection,
            "character_text_threads_old",
            "memory_body",
            "''",
        )
        thread_memory_count_expr = _old_column_expr(
            connection,
            "character_text_threads_old",
            "memory_message_count",
            "0",
        )
        thread_memory_updated_expr = _old_column_expr(
            connection,
            "character_text_threads_old",
            "memory_updated_at",
            "NULL",
        )
        connection.execute(
            f"""
            INSERT INTO character_text_threads(
                id, save_id, character_id, title, status, kind, memory_body,
                memory_message_count, memory_updated_at, created_at, updated_at,
                archived_at
            )
            SELECT
                id,
                save_id,
                character_id,
                title,
                status,
                {thread_kind_expr},
                {thread_memory_body_expr},
                {thread_memory_count_expr},
                {thread_memory_updated_expr},
                created_at,
                updated_at,
                archived_at
            FROM character_text_threads_old old
            """
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO character_text_thread_participants(
                id, save_id, thread_id, character_id, ordinal, created_at, updated_at
            )
            SELECT
                lower(hex(randomblob(16))),
                save_id,
                id,
                character_id,
                0,
                COALESCE(created_at, CURRENT_TIMESTAMP),
                COALESCE(updated_at, CURRENT_TIMESTAMP)
            FROM character_text_threads
            WHERE kind = 'direct' AND character_id IS NOT NULL
            """
        )
        message_delivery_status_expr = _old_column_expr(
            connection,
            "character_text_messages_old",
            "delivery_status",
            "'sent'",
        )
        message_delivery_error_expr = _old_column_expr(
            connection,
            "character_text_messages_old",
            "delivery_error",
            "NULL",
        )
        message_delivery_job_expr = _old_column_expr(
            connection,
            "character_text_messages_old",
            "delivery_job_id",
            "NULL",
        )
        message_delivery_attempt_expr = _old_column_expr(
            connection,
            "character_text_messages_old",
            "delivery_attempt",
            "0",
        )
        message_in_world_sent_expr = _old_column_expr(
            connection,
            "character_text_messages_old",
            "in_world_sent_at",
            "NULL",
        )
        message_delivered_expr = _old_column_expr(
            connection,
            "character_text_messages_old",
            "delivered_at",
            "NULL",
        )
        message_read_expr = _old_column_expr(
            connection,
            "character_text_messages_old",
            "read_at",
            "NULL",
        )
        message_reply_to_expr = _old_column_expr(
            connection,
            "character_text_messages_old",
            "reply_to_message_id",
            "NULL",
        )
        connection.execute(
            f"""
            INSERT INTO character_text_messages(
                id, save_id, thread_id, character_id, sender, body,
                sender_character_id, provider, model, token_estimate,
                delivery_status, delivery_error, delivery_job_id,
                delivery_attempt, in_world_sent_at, delivered_at, read_at,
                reply_to_message_id, created_at, updated_at, deleted_at
            )
            SELECT
                old.id,
                old.save_id,
                old.thread_id,
                old.character_id,
                old.sender,
                old.body,
                {_character_text_sender_character_id_expr(connection)},
                old.provider,
                old.model,
                old.token_estimate,
                {message_delivery_status_expr},
                {message_delivery_error_expr},
                {message_delivery_job_expr},
                {message_delivery_attempt_expr},
                {message_in_world_sent_expr},
                {message_delivered_expr},
                {message_read_expr},
                {message_reply_to_expr},
                old.created_at,
                old.updated_at,
                old.deleted_at
            FROM character_text_messages_old old
            """
        )
        connection.execute("DROP TABLE character_text_messages_old")
        connection.execute("DROP TABLE character_text_threads_old")
        _execute_schema_script(
            connection,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_character_text_threads_save_direct
            ON character_text_threads(save_id, character_id, kind)
            WHERE kind = 'direct' AND character_id IS NOT NULL;

            CREATE INDEX IF NOT EXISTS idx_character_text_threads_save_active
            ON character_text_threads(save_id, archived_at, updated_at);

            CREATE INDEX IF NOT EXISTS idx_character_text_participants_thread
            ON character_text_thread_participants(thread_id, ordinal, character_id);

            CREATE INDEX IF NOT EXISTS idx_character_text_messages_thread_created
            ON character_text_messages(thread_id, created_at, id);

            CREATE INDEX IF NOT EXISTS idx_character_text_messages_save_created
            ON character_text_messages(save_id, created_at, id);
            """
        )
    finally:
        connection.execute(
            f"PRAGMA legacy_alter_table = {1 if legacy_alter_table else 0}"
        )
        connection.execute(f"PRAGMA foreign_keys = {1 if foreign_keys_enabled else 0}")


def _old_column_expr(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    fallback: str,
) -> str:
    if _column_exists(connection, table_name, column_name):
        return f"old.{column_name}"
    return fallback


def _character_text_sender_character_id_expr(connection: sqlite3.Connection) -> str:
    if _column_exists(
        connection,
        "character_text_messages_old",
        "sender_character_id",
    ):
        return "old.sender_character_id"
    return (
        "CASE "
        "WHEN old.sender = 'character' THEN old.character_id "
        "WHEN old.sender = 'player' THEN ("
        "SELECT characters.id FROM characters "
        "WHERE characters.save_id = old.save_id "
        "AND characters.is_player_character = 1 "
        "AND characters.archived_at IS NULL "
        "ORDER BY characters.created_at, characters.rowid LIMIT 1"
        ") "
        "ELSE NULL END"
    )


def _ensure_character_text_thread_memory_schema(
    connection: sqlite3.Connection,
) -> None:
    if not _table_exists(connection, "character_text_threads"):
        return
    _add_column_if_missing(
        connection,
        "character_text_threads",
        "memory_body",
        "TEXT NOT NULL DEFAULT ''",
    )
    _add_column_if_missing(
        connection,
        "character_text_threads",
        "memory_message_count",
        "INTEGER NOT NULL DEFAULT 0",
    )
    _add_column_if_missing(
        connection,
        "character_text_threads",
        "memory_updated_at",
        "TEXT",
    )


def _ensure_character_text_message_metadata_schema(
    connection: sqlite3.Connection,
) -> None:
    if not _table_exists(connection, "character_text_messages"):
        return
    _add_column_if_missing(
        connection,
        "character_text_messages",
        "in_world_sent_at",
        "TEXT",
    )
    _add_column_if_missing(
        connection,
        "character_text_messages",
        "delivered_at",
        "TEXT",
    )
    _add_column_if_missing(
        connection,
        "character_text_messages",
        "read_at",
        "TEXT",
    )
    _add_column_if_missing(
        connection,
        "character_text_messages",
        "reply_to_message_id",
        "TEXT",
    )


def _ensure_character_text_message_revision_schema(
    connection: sqlite3.Connection,
) -> None:
    if not _table_exists(connection, "character_text_messages"):
        return
    _execute_schema_script(
        connection,
        """
        CREATE TABLE IF NOT EXISTS character_text_message_revisions (
            id TEXT PRIMARY KEY,
            save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
            text_message_id TEXT NOT NULL REFERENCES character_text_messages(id),
            revision_number INTEGER NOT NULL,
            previous_body TEXT NOT NULL,
            new_body TEXT NOT NULL,
            diff_unified TEXT NOT NULL,
            reconciliation_status TEXT NOT NULL DEFAULT 'queued',
            reconciliation_error TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            reconciled_at TEXT,
            UNIQUE(text_message_id, revision_number)
        );

        CREATE INDEX IF NOT EXISTS idx_character_text_message_revisions_save_text
        ON character_text_message_revisions(save_id, text_message_id);
        """
    )


def _ensure_character_text_message_attachment_schema(
    connection: sqlite3.Connection,
) -> None:
    if not _table_exists(connection, "character_text_messages"):
        return
    _execute_schema_script(
        connection,
        """
        CREATE TABLE IF NOT EXISTS character_text_message_attachments (
            id TEXT PRIMARY KEY,
            save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
            thread_id TEXT NOT NULL REFERENCES character_text_threads(id)
                ON DELETE CASCADE,
            text_message_id TEXT NOT NULL REFERENCES character_text_messages(id)
                ON DELETE CASCADE,
            character_id TEXT NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL DEFAULT 0,
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            media_asset_id TEXT REFERENCES media_assets(id) ON DELETE SET NULL,
            prompt TEXT NOT NULL DEFAULT '',
            error TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_character_text_attachments_text_order
        ON character_text_message_attachments(text_message_id, ordinal, created_at, id);

        CREATE INDEX IF NOT EXISTS idx_character_text_attachments_media_asset
        ON character_text_message_attachments(media_asset_id);
        """
    )


def _ensure_character_contact_state_schema(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "saves"):
        return
    _execute_schema_script(
        connection,
        """
        CREATE TABLE IF NOT EXISTS character_contact_states (
            id TEXT PRIMARY KEY,
            save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
            player_character_id TEXT NOT NULL REFERENCES characters(id)
                ON DELETE CASCADE,
            character_id TEXT NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
            player_has_character_number INTEGER NOT NULL DEFAULT 0,
            character_has_player_number INTEGER NOT NULL DEFAULT 0,
            source_message_id TEXT REFERENCES messages(id),
            source_text_message_id TEXT REFERENCES character_text_messages(id),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            archived_at TEXT,
            UNIQUE(save_id, player_character_id, character_id)
        );

        CREATE INDEX IF NOT EXISTS idx_character_contact_states_save_active
        ON character_contact_states(save_id, archived_at, updated_at);

        CREATE INDEX IF NOT EXISTS idx_character_contact_states_character_active
        ON character_contact_states(save_id, character_id, archived_at);
        """
    )


def _ensure_character_text_proactive_trigger_schema(
    connection: sqlite3.Connection,
) -> None:
    if not _table_exists(connection, "saves"):
        return
    _ensure_character_text_schema(connection)
    _execute_schema_script(
        connection,
        """
        CREATE TABLE IF NOT EXISTS character_text_proactive_triggers (
            id TEXT PRIMARY KEY,
            save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
            character_id TEXT NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
            trigger_key TEXT NOT NULL,
            trigger_type TEXT NOT NULL DEFAULT '',
            thread_id TEXT REFERENCES character_text_threads(id) ON DELETE SET NULL,
            text_message_id TEXT REFERENCES character_text_messages(id)
                ON DELETE SET NULL,
            source_type TEXT NOT NULL DEFAULT '',
            source_id TEXT NOT NULL DEFAULT '',
            source_message_id TEXT REFERENCES messages(id),
            reason TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(save_id, character_id, trigger_key)
        );

        CREATE INDEX IF NOT EXISTS idx_character_text_proactive_triggers_save_character
        ON character_text_proactive_triggers(save_id, character_id, created_at);

        CREATE INDEX IF NOT EXISTS idx_character_text_proactive_triggers_text
        ON character_text_proactive_triggers(text_message_id);
        """
    )




def _backfill_player_characters_from_scenarios(
    connection: sqlite3.Connection,
) -> None:
    if not all(
        _table_exists(connection, table)
        for table in ("characters", "saves", "scenarios")
    ):
        return
    character_columns = _column_names(connection, "characters")
    scenario_columns = _column_names(connection, "scenarios")
    if (
        "is_player_character" not in character_columns
        or "content_json" not in scenario_columns
    ):
        return
    rows = connection.execute(
        """
        SELECT saves.id, scenarios.content_json
        FROM saves
        JOIN scenarios ON scenarios.id = saves.scenario_id
        ORDER BY saves.rowid
        """
    ).fetchall()
    for save_id, content_json in rows:
        player_name = _migration_player_character_name(content_json)
        if not player_name:
            continue
        matches = _migration_matching_character_ids(
            connection,
            save_id=str(save_id),
            player_name=player_name,
        )
        if len(matches) != 1:
            continue
        connection.execute(
            """
            UPDATE characters
            SET is_player_character = 1,
                protected_from_maintenance = 1
            WHERE id = ?
            """,
            (matches[0],),
        )
        _migration_add_present_character_id(
            connection,
            save_id=str(save_id),
            character_id=matches[0],
        )


def _migration_player_character_name(content_json: object) -> str:
    try:
        content = json.loads(str(content_json or "{}"))
    except json.JSONDecodeError:
        return ""
    if not isinstance(content, dict):
        return ""
    value = content.get("player_character_name")
    return value.strip() if isinstance(value, str) else ""


def _migration_matching_character_ids(
    connection: sqlite3.Connection,
    *,
    save_id: str,
    player_name: str,
) -> list[str]:
    target = _migration_character_key(player_name)
    if not target:
        return []
    matches: list[str] = []
    for row in connection.execute(
        """
        SELECT id, name, aliases_json
        FROM characters
        WHERE save_id = ? AND archived_at IS NULL
        ORDER BY rowid
        """,
        (save_id,),
    ):
        names = [str(row[1] or "")]
        try:
            aliases = json.loads(str(row[2] or "[]"))
        except json.JSONDecodeError:
            aliases = []
        if isinstance(aliases, list):
            names.extend(alias for alias in aliases if isinstance(alias, str))
        if any(_migration_character_key(name) == target for name in names):
            matches.append(str(row[0]))
    return matches


def _migration_add_present_character_id(
    connection: sqlite3.Connection,
    *,
    save_id: str,
    character_id: str,
) -> None:
    if (
        not _table_exists(connection, "scene_snapshots")
        or "present_character_ids_json"
        not in _column_names(connection, "scene_snapshots")
    ):
        return
    row = connection.execute(
        """
        SELECT id, present_character_ids_json
        FROM scene_snapshots
        WHERE save_id = ?
        """,
        (save_id,),
    ).fetchone()
    if row is None:
        return
    try:
        present = json.loads(str(row[1] or "[]"))
    except json.JSONDecodeError:
        present = []
    if not isinstance(present, list):
        present = []
    present_ids = [item for item in present if isinstance(item, str)]
    if character_id not in present_ids:
        present_ids.append(character_id)
    connection.execute(
        """
        UPDATE scene_snapshots
        SET present_character_ids_json = ?
        WHERE id = ?
        """,
        (json.dumps(present_ids, sort_keys=True), row[0]),
    )




def _ensure_auth_schema(connection: sqlite3.Connection) -> None:
    _execute_schema_script(
        connection,
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            username_normalized TEXT NOT NULL,
            role TEXT NOT NULL
                CHECK (role IN ('admin', 'user', 'child')),
            password_hash TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active'
                CHECK (status IN ('active', 'disabled')),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username_normalized
            ON users(username_normalized);

        CREATE TABLE IF NOT EXISTS user_sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            token_hash TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            revoked_at TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_user_sessions_token_hash
            ON user_sessions(token_hash);

        CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id
            ON user_sessions(user_id, revoked_at, expires_at);
        """
    )


def _ensure_save_access_schema(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "saves"):
        return
    _ensure_auth_schema(connection)
    _add_column_if_missing(
        connection,
        "saves",
        "owner_user_id",
        "TEXT REFERENCES users(id)",
    )
    _execute_schema_script(
        connection,
        """
        CREATE INDEX IF NOT EXISTS idx_saves_owner_user_id
            ON saves(owner_user_id);

        CREATE TABLE IF NOT EXISTS save_assignments (
            id TEXT PRIMARY KEY,
            save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(save_id, user_id)
        );

        CREATE INDEX IF NOT EXISTS idx_save_assignments_save_user
            ON save_assignments(save_id, user_id);

        CREATE INDEX IF NOT EXISTS idx_save_assignments_user_id
            ON save_assignments(user_id);

        CREATE TABLE IF NOT EXISTS user_runtime_state (
            user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            active_save_id TEXT REFERENCES saves(id) ON DELETE SET NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_user_runtime_state_active_save
            ON user_runtime_state(active_save_id);
        """
    )


def _ensure_character_knowledge_schema(connection: sqlite3.Connection) -> None:
    if not (
        _table_exists(connection, "saves")
        and _table_exists(connection, "messages")
        and _table_exists(connection, "characters")
    ):
        return
    _execute_schema_script(
        connection,
        """
        CREATE TABLE IF NOT EXISTS character_knowledge_edges (
            id TEXT PRIMARY KEY,
            save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
            character_id TEXT NOT NULL REFERENCES characters(id),
            target_type TEXT NOT NULL,
            target_id TEXT NOT NULL,
            knowledge_state TEXT NOT NULL DEFAULT 'knows'
                CHECK (knowledge_state IN ('knows', 'may_know', 'does_not_know')),
            acquisition_method TEXT NOT NULL DEFAULT 'unknown'
                CHECK (
                    acquisition_method IN (
                        'witnessed',
                        'overheard',
                        'told',
                        'inferred_from_visible_consequence',
                        'background',
                        'manual',
                        'unknown'
                    )
                ),
            confidence REAL NOT NULL DEFAULT 1.0,
            source_message_id TEXT REFERENCES messages(id),
            source_message_ids_json TEXT NOT NULL DEFAULT '[]',
            evidence_quote TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            archived_at TEXT,
            UNIQUE(save_id, character_id, target_type, target_id)
        );

        CREATE INDEX IF NOT EXISTS idx_character_knowledge_edges_save_character
            ON character_knowledge_edges(save_id, character_id, archived_at);

        CREATE INDEX IF NOT EXISTS idx_character_knowledge_edges_target
            ON character_knowledge_edges(save_id, target_type, target_id, archived_at);

        CREATE TABLE IF NOT EXISTS message_visibility (
            id TEXT PRIMARY KEY,
            save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
            message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            character_id TEXT NOT NULL REFERENCES characters(id),
            visibility TEXT NOT NULL DEFAULT 'unknown'
                CHECK (visibility IN ('visible', 'not_visible', 'unknown')),
            confidence REAL NOT NULL DEFAULT 1.0,
            source TEXT NOT NULL DEFAULT 'unknown',
            evidence TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(save_id, message_id, character_id)
        );

        CREATE INDEX IF NOT EXISTS idx_message_visibility_save_message
            ON message_visibility(save_id, message_id);

        CREATE INDEX IF NOT EXISTS idx_message_visibility_save_character
            ON message_visibility(save_id, character_id);
        """
    )




def _ensure_scheduled_tasks_schema(connection: sqlite3.Connection) -> None:
    if not (
        _table_exists(connection, "saves")
        and _table_exists(connection, "jobs")
    ):
        return
    _execute_schema_script(
        connection,
        """
        CREATE TABLE IF NOT EXISTS scheduled_tasks (
            id TEXT PRIMARY KEY,
            task_type TEXT NOT NULL,
            save_id TEXT REFERENCES saves(id) ON DELETE CASCADE,
            enabled INTEGER NOT NULL DEFAULT 1,
            interval_seconds INTEGER NOT NULL DEFAULT 60,
            next_run_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            lease_until TEXT,
            last_started_at TEXT,
            last_completed_at TEXT,
            last_job_id TEXT REFERENCES jobs(id) ON DELETE SET NULL,
            failure_count INTEGER NOT NULL DEFAULT 0,
            payload_json TEXT NOT NULL DEFAULT '{}',
            result_json TEXT,
            error TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(task_type, save_id)
        );

        CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_due
        ON scheduled_tasks(enabled, next_run_at, lease_until);

        CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_save_type
        ON scheduled_tasks(save_id, task_type);
        """
    )


def _ensure_runtime_telemetry_schema(connection: sqlite3.Connection) -> None:
    _execute_schema_script(
        connection,
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            save_id TEXT REFERENCES saves(id),
            creator_user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
            type TEXT NOT NULL,
            status TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            result_json TEXT,
            error TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            started_at TEXT,
            completed_at TEXT,
            duration_ms INTEGER,
            diagnostics_json TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_jobs_status_type_save_completed
        ON jobs(status, type, save_id, completed_at);

        CREATE INDEX IF NOT EXISTS idx_jobs_save_status_completed
        ON jobs(save_id, status, completed_at);
        """
    )
    _add_column_if_missing(connection, "jobs", "duration_ms", "INTEGER")
    _add_column_if_missing(connection, "jobs", "diagnostics_json", "TEXT")
    _add_column_if_missing(
        connection,
        "jobs",
        "creator_user_id",
        "TEXT",
    )
    _execute_schema_script(
        connection,
        """
        CREATE TABLE IF NOT EXISTS job_steps (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            status TEXT NOT NULL,
            provider TEXT,
            model TEXT,
            task TEXT,
            started_at TEXT,
            completed_at TEXT,
            duration_ms INTEGER,
            error TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_job_steps_job_id
        ON job_steps(job_id);

        CREATE INDEX IF NOT EXISTS idx_job_steps_name_status_completed
        ON job_steps(name, status, completed_at);

        CREATE INDEX IF NOT EXISTS idx_job_steps_provider_model_task_status_completed
        ON job_steps(provider, model, task, status, completed_at);
        """
    )
    connection.execute(
        """
        UPDATE jobs
        SET duration_ms = MAX(
            0,
            CAST(ROUND((julianday(completed_at) - julianday(started_at)) * 86400000)
                AS INTEGER)
        )
        WHERE duration_ms IS NULL
          AND status IN ('succeeded', 'failed', 'cancelled')
          AND started_at IS NOT NULL
          AND completed_at IS NOT NULL
        """
    )


def _ensure_context_revision_schema(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "saves"):
        return
    _drop_context_revision_triggers(connection)
    if _save_context_revisions_needs_rebuild(connection):
        _rebuild_save_context_revisions_table(connection)
    _execute_schema_script(
        connection,
        """
        CREATE TABLE IF NOT EXISTS save_context_revisions (
            save_id TEXT PRIMARY KEY REFERENCES saves(id) ON DELETE CASCADE,
            revision INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        INSERT OR IGNORE INTO save_context_revisions(save_id, revision)
        SELECT id, 0 FROM saves;

        CREATE TRIGGER IF NOT EXISTS init_save_context_revision_after_save_insert
        AFTER INSERT ON saves
        FOR EACH ROW
        BEGIN
            INSERT OR IGNORE INTO save_context_revisions(save_id, revision)
            VALUES (NEW.id, 0);
        END;
        """
    )
    _ensure_message_context_revision_schema(connection)
    for table_name, save_id_ref, events in (
        ("world_state", "save_id", ("INSERT", "UPDATE", "DELETE")),
        ("state_changes", "save_id", ("INSERT", "DELETE")),
        ("media_assets", "save_id", ("INSERT", "UPDATE", "DELETE")),
        ("memories", "save_id", ("INSERT", "UPDATE", "DELETE")),
        ("summaries", "save_id", ("INSERT", "UPDATE", "DELETE")),
        ("save_scenario_updates", "save_id", ("INSERT", "UPDATE", "DELETE")),
        ("save_loss_conditions", "save_id", ("INSERT", "UPDATE", "DELETE")),
        ("save_loss_condition_changes", "save_id", ("INSERT", "UPDATE", "DELETE")),
        ("save_loss_outcomes", "save_id", ("INSERT", "UPDATE", "DELETE")),
        ("context_sources", "save_id", ("INSERT", "UPDATE", "DELETE")),
        ("context_observations", "save_id", ("INSERT", "UPDATE", "DELETE")),
        ("context_update_audit", "save_id", ("INSERT", "UPDATE", "DELETE")),
        ("scene_snapshots", "save_id", ("INSERT", "UPDATE", "DELETE")),
        ("locations", "save_id", ("INSERT", "UPDATE", "DELETE")),
        ("characters", "save_id", ("INSERT", "UPDATE", "DELETE")),
        ("active_threads", "save_id", ("INSERT", "UPDATE", "DELETE")),
        ("entity_links", "save_id", ("INSERT", "UPDATE", "DELETE")),
        ("character_knowledge_edges", "save_id", ("INSERT", "UPDATE", "DELETE")),
        ("message_visibility", "save_id", ("INSERT", "UPDATE", "DELETE")),
        ("message_scene_presence", "save_id", ("INSERT", "UPDATE", "DELETE")),
        ("message_action_choices", "save_id", ("INSERT", "UPDATE", "DELETE")),
        ("dating_route_states", "save_id", ("INSERT", "UPDATE", "DELETE")),
    ):
        if not _table_exists(connection, table_name):
            continue
        for event in events:
            row_ref = "OLD" if event == "DELETE" else "NEW"
            trigger_name = f"bump_{table_name}_context_revision_after_{event.lower()}"
            _execute_schema_script(
                connection,
                f"""
                CREATE TRIGGER IF NOT EXISTS {trigger_name}
                AFTER {event} ON {table_name}
                FOR EACH ROW
                BEGIN
                    INSERT INTO save_context_revisions(save_id, revision, updated_at)
                    VALUES ({row_ref}.{save_id_ref}, 1, CURRENT_TIMESTAMP)
                    ON CONFLICT(save_id) DO UPDATE SET
                        revision = revision + 1,
                        updated_at = CURRENT_TIMESTAMP;
                END;
                """
            )
    _ensure_message_context_revision_triggers(connection)


def _ensure_continuity_index_revision_schema(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "saves"):
        return
    _execute_schema_script(
        connection,
        """
        CREATE TABLE IF NOT EXISTS save_continuity_index_revisions (
            save_id TEXT PRIMARY KEY REFERENCES saves(id) ON DELETE CASCADE,
            revision INTEGER NOT NULL DEFAULT 0,
            indexed_revision INTEGER NOT NULL DEFAULT -1,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS continuity_index_dirty_sources (
            save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
            source_kind TEXT NOT NULL,
            source_id TEXT NOT NULL,
            dirty_generation INTEGER NOT NULL DEFAULT 1,
            queued_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(save_id, source_kind, source_id)
        );

        CREATE INDEX IF NOT EXISTS idx_continuity_dirty_save_queue
        ON continuity_index_dirty_sources(
            save_id,
            queued_at,
            source_kind,
            source_id
        );

        INSERT OR IGNORE INTO save_continuity_index_revisions(
            save_id,
            revision,
            indexed_revision
        )
        SELECT id, 0, -1 FROM saves;

        CREATE TRIGGER IF NOT EXISTS init_continuity_revision_after_save_insert
        AFTER INSERT ON saves
        BEGIN
            INSERT OR IGNORE INTO save_continuity_index_revisions(
                save_id,
                revision,
                indexed_revision
            )
            VALUES (NEW.id, 0, -1);
        END;
        """,
    )
    source_mappings = {
        "world_state": ("world_state", "id"),
        "memories": ("memory", "id"),
        "summaries": ("summary", "id"),
        "active_threads": ("active_thread", "id"),
        "locations": ("location", "id"),
        "characters": ("character", "id"),
        "save_scenario_updates": ("scenario", None),
        "character_text_threads": ("character_text_thread", "id"),
        "character_text_messages": ("character_text_thread", "thread_id"),
    }
    for table_name in source_mappings:
        if not _table_exists(connection, table_name):
            continue
        for event in ("INSERT", "UPDATE", "DELETE"):
            row_ref = "OLD" if event == "DELETE" else "NEW"
            trigger_name = (
                f"bump_{table_name}_continuity_revision_after_{event.lower()}"
            )
            connection.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")
            source_kind, source_column = source_mappings[table_name]
            source_id_sql = (
                "'scenario'"
                if source_column is None
                else f"{row_ref}.{source_column}"
            )
            _execute_schema_script(
                connection,
                f"""
                CREATE TRIGGER {trigger_name}
                AFTER {event} ON {table_name}
                BEGIN
                    INSERT INTO save_continuity_index_revisions(
                        save_id,
                        revision,
                        indexed_revision,
                        updated_at
                    )
                    VALUES ({row_ref}.save_id, 1, -1, CURRENT_TIMESTAMP)
                    ON CONFLICT(save_id) DO UPDATE SET
                        revision = revision + 1,
                        updated_at = CURRENT_TIMESTAMP;

                    INSERT INTO continuity_index_dirty_sources(
                        save_id,
                        source_kind,
                        source_id,
                        dirty_generation,
                        queued_at
                    )
                    VALUES (
                        {row_ref}.save_id,
                        '{source_kind}',
                        {source_id_sql},
                        1,
                        CURRENT_TIMESTAMP
                    )
                    ON CONFLICT(save_id, source_kind, source_id) DO UPDATE SET
                        dirty_generation = dirty_generation + 1,
                        queued_at = CURRENT_TIMESTAMP;
                END;
                """,
            )
    _ensure_scene_snapshot_continuity_triggers(connection)
    _ensure_scenario_continuity_triggers(connection)


def _ensure_scene_snapshot_continuity_triggers(
    connection: sqlite3.Connection,
) -> None:
    if not _table_exists(connection, "scene_snapshots"):
        return
    for event, references in (
        ("INSERT", ("NEW",)),
        ("UPDATE", ("OLD", "NEW")),
        ("DELETE", ("OLD",)),
    ):
        trigger_name = (
            f"bump_scene_snapshots_continuity_revision_after_{event.lower()}"
        )
        connection.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")
        row_ref = "OLD" if event == "DELETE" else "NEW"
        location_queries = " UNION ".join(
            f"SELECT {reference}.current_location_id AS source_id"
            for reference in references
        )
        character_queries = " UNION ".join(
            (
                "SELECT CAST(value AS TEXT) AS source_id "
                f"FROM json_each(COALESCE({reference}."
                "present_character_ids_json, '[]'))"
            )
            for reference in references
        )
        _execute_schema_script(
            connection,
            f"""
            CREATE TRIGGER {trigger_name}
            AFTER {event} ON scene_snapshots
            BEGIN
                INSERT INTO save_continuity_index_revisions(
                    save_id, revision, indexed_revision, updated_at
                )
                VALUES ({row_ref}.save_id, 1, -1, CURRENT_TIMESTAMP)
                ON CONFLICT(save_id) DO UPDATE SET
                    revision = revision + 1,
                    updated_at = CURRENT_TIMESTAMP;

                INSERT INTO continuity_index_dirty_sources(
                    save_id, source_kind, source_id, dirty_generation, queued_at
                )
                SELECT {row_ref}.save_id, 'location', source_id, 1,
                       CURRENT_TIMESTAMP
                FROM ({location_queries})
                WHERE source_id IS NOT NULL AND source_id != ''
                ON CONFLICT(save_id, source_kind, source_id) DO UPDATE SET
                    dirty_generation = dirty_generation + 1,
                    queued_at = CURRENT_TIMESTAMP;

                INSERT INTO continuity_index_dirty_sources(
                    save_id, source_kind, source_id, dirty_generation, queued_at
                )
                SELECT {row_ref}.save_id, 'character', source_id, 1,
                       CURRENT_TIMESTAMP
                FROM ({character_queries})
                WHERE source_id IS NOT NULL AND source_id != ''
                ON CONFLICT(save_id, source_kind, source_id) DO UPDATE SET
                    dirty_generation = dirty_generation + 1,
                    queued_at = CURRENT_TIMESTAMP;
            END;
            """,
        )


def _ensure_scenario_continuity_triggers(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "scenarios"):
        return
    trigger_name = "bump_scenarios_continuity_revision_after_update"
    connection.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")
    _execute_schema_script(
        connection,
        f"""
        CREATE TRIGGER {trigger_name}
        AFTER UPDATE ON scenarios
        BEGIN
            INSERT INTO save_continuity_index_revisions(
                save_id, revision, indexed_revision, updated_at
            )
            SELECT saves.id, 1, -1, CURRENT_TIMESTAMP
            FROM saves
            WHERE saves.scenario_id = NEW.id
            ON CONFLICT(save_id) DO UPDATE SET
                revision = revision + 1,
                updated_at = CURRENT_TIMESTAMP;

            INSERT INTO continuity_index_dirty_sources(
                save_id, source_kind, source_id, dirty_generation, queued_at
            )
            SELECT saves.id, 'scenario', 'scenario', 1, CURRENT_TIMESTAMP
            FROM saves
            WHERE saves.scenario_id = NEW.id
            ON CONFLICT(save_id, source_kind, source_id) DO UPDATE SET
                dirty_generation = dirty_generation + 1,
                queued_at = CURRENT_TIMESTAMP;
        END;
        """,
    )


def _ensure_context_source_search_terms_schema(
    connection: sqlite3.Connection,
) -> None:
    if not _table_exists(connection, "context_sources"):
        return
    _execute_schema_script(
        connection,
        """
        CREATE TABLE IF NOT EXISTS context_source_search_terms (
            context_source_id TEXT NOT NULL
                REFERENCES context_sources(id) ON DELETE CASCADE,
            save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
            term TEXT NOT NULL,
            PRIMARY KEY(context_source_id, term)
        );

        CREATE INDEX IF NOT EXISTS idx_context_source_terms_save_term
        ON context_source_search_terms(save_id, term, context_source_id);

        CREATE TABLE IF NOT EXISTS context_source_exact_identifiers (
            context_source_id TEXT NOT NULL
                REFERENCES context_sources(id) ON DELETE CASCADE,
            save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
            identifier TEXT NOT NULL,
            PRIMARY KEY(context_source_id, identifier)
        );

        CREATE INDEX IF NOT EXISTS idx_context_source_identifiers_save_value
        ON context_source_exact_identifiers(
            save_id,
            identifier,
            context_source_id
        );

        CREATE TABLE IF NOT EXISTS context_source_exact_identifier_filters (
            context_source_id TEXT PRIMARY KEY
                REFERENCES context_sources(id) ON DELETE CASCADE,
            save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
            identifiers_blob BLOB NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_context_source_identifier_filters_save
        ON context_source_exact_identifier_filters(save_id, context_source_id);

        DELETE FROM context_source_search_terms
        WHERE context_source_id NOT IN (SELECT id FROM context_sources);

        DELETE FROM context_source_exact_identifiers
        WHERE context_source_id NOT IN (SELECT id FROM context_sources);

        DELETE FROM context_source_exact_identifier_filters
        WHERE context_source_id NOT IN (SELECT id FROM context_sources);
        """,
    )
    missing_text_chars = int(
        connection.execute(
            """
            SELECT COALESCE(SUM(LENGTH(source.title) + LENGTH(source.body)), 0)
            FROM context_sources source
            WHERE source.archived_at IS NULL
              AND (
                    NOT EXISTS (
                        SELECT 1
                        FROM context_source_search_terms term
                        WHERE term.context_source_id = source.id
                    )
                    OR NOT EXISTS (
                        SELECT 1
                        FROM context_source_exact_identifiers identifier
                        WHERE identifier.context_source_id = source.id
                    )
                    OR NOT EXISTS (
                        SELECT 1
                        FROM context_source_exact_identifier_filters filter
                        WHERE filter.context_source_id = source.id
                    )
                  )
            """
        ).fetchone()[0]
    )
    if missing_text_chars > _MAX_CONTEXT_INDEX_TEXT_CHARS_PER_REBUILD:
        raise RuntimeError("Context source text is too large to index")
    missing_rows = connection.execute(
        """
        SELECT source.id, source.save_id, source.title, source.body
        FROM context_sources source
        WHERE source.archived_at IS NULL
          AND NOT EXISTS (
              SELECT 1
              FROM context_source_search_terms term
              WHERE term.context_source_id = source.id
          )
        ORDER BY source.rowid
        """
    ).fetchall()
    indexed_rows = 0
    for source_id, save_id, title, body in missing_rows:
        terms = _migration_context_source_search_terms(
            str(title or ""),
            str(body or ""),
        )
        indexed_rows += len(terms)
        if indexed_rows > _MAX_CONTEXT_INDEX_ROWS_PER_REBUILD:
            raise RuntimeError("Context source index is too large to rebuild")
        connection.executemany(
            """
            INSERT OR IGNORE INTO context_source_search_terms(
                context_source_id,
                save_id,
                term
            )
            VALUES (?, ?, ?)
            """,
            ((str(source_id), str(save_id), term) for term in terms),
        )
    missing_identifier_rows = connection.execute(
        """
        SELECT source.id, source.save_id, source.title, source.body
        FROM context_sources source
        WHERE source.archived_at IS NULL
          AND NOT EXISTS (
              SELECT 1
              FROM context_source_exact_identifiers identifier
              WHERE identifier.context_source_id = source.id
          )
        ORDER BY source.rowid
        """
    ).fetchall()
    for source_id, save_id, title, body in missing_identifier_rows:
        identifiers = _migration_context_source_exact_identifiers(
            str(title or ""),
            str(body or ""),
        )
        indexed_rows += max(1, len(identifiers))
        if indexed_rows > _MAX_CONTEXT_INDEX_ROWS_PER_REBUILD:
            raise RuntimeError("Context source index is too large to rebuild")
        connection.executemany(
            """
            INSERT OR IGNORE INTO context_source_exact_identifiers(
                context_source_id,
                save_id,
                identifier
            )
            VALUES (?, ?, ?)
            """,
            (
                (str(source_id), str(save_id), identifier)
                for identifier in identifiers or ("",)
            ),
        )
    missing_filter_rows = connection.execute(
        """
        SELECT source.id, source.save_id, source.title, source.body
        FROM context_sources source
        WHERE source.archived_at IS NULL
          AND NOT EXISTS (
              SELECT 1
              FROM context_source_exact_identifier_filters filter
              WHERE filter.context_source_id = source.id
          )
        ORDER BY source.rowid
        """
    ).fetchall()
    indexed_rows += len(missing_filter_rows)
    if indexed_rows > _MAX_CONTEXT_INDEX_ROWS_PER_REBUILD:
        raise RuntimeError("Context source index is too large to rebuild")
    connection.executemany(
        """
        INSERT INTO context_source_exact_identifier_filters(
            context_source_id,
            save_id,
            identifiers_blob
        )
        VALUES (?, ?, ?)
        """,
        (
            (
                str(source_id),
                str(save_id),
                structured_identifier_filter(
                    str(title or ""),
                    str(body or ""),
                ),
            )
            for source_id, save_id, title, body in missing_filter_rows
        ),
    )


def _migration_context_source_search_terms(
    title: str,
    body: str,
) -> tuple[str, ...]:
    bounded_title = title[:_MAX_CONTEXT_SOURCE_SEARCH_TEXT_CHARS]
    bounded_body = body[:_MAX_CONTEXT_SOURCE_SEARCH_TEXT_CHARS]
    terms = (
        *unicode_word_terms(bounded_title),
        *cjk_lexical_anchors(bounded_title),
        *unicode_word_terms(bounded_body),
        *cjk_lexical_anchors(bounded_body),
    )
    return tuple(dict.fromkeys(terms))[:_MAX_CONTEXT_SOURCE_INDEX_TERMS]


def _migration_context_source_exact_identifiers(
    title: str,
    body: str,
) -> tuple[str, ...]:
    identifiers = tuple(
        dict.fromkeys(
            (
                *structured_identifiers(
                    title,
                    max_input_chars=_MAX_CONTEXT_SOURCE_SEARCH_TEXT_CHARS,
                ),
                *structured_identifiers(
                    body,
                    max_input_chars=_MAX_CONTEXT_SOURCE_SEARCH_TEXT_CHARS,
                ),
            )
        )
    )
    if len(identifiers) <= _MAX_CONTEXT_SOURCE_INDEX_IDENTIFIERS:
        return identifiers
    edge_count = _MAX_CONTEXT_SOURCE_INDEX_IDENTIFIERS // 2
    return (
        *identifiers[:edge_count],
        *identifiers[-edge_count:],
    )


def _ensure_message_context_revision_schema(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "messages"):
        return
    if _message_context_revisions_needs_rebuild(connection):
        _rebuild_message_context_revisions_table(connection)
    _execute_schema_script(
        connection,
        """
        CREATE TABLE IF NOT EXISTS message_context_revisions (
            message_id TEXT PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
            save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
            revision INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_message_context_revisions_save_id
        ON message_context_revisions(save_id);
        """
    )
    if not _column_exists(connection, "messages", "save_id"):
        return
    if _column_exists(connection, "messages", "updated_at"):
        timestamp_expr = "COALESCE(messages.updated_at, CURRENT_TIMESTAMP)"
    elif _column_exists(connection, "messages", "created_at"):
        timestamp_expr = "COALESCE(messages.created_at, CURRENT_TIMESTAMP)"
    else:
        timestamp_expr = "CURRENT_TIMESTAMP"
    connection.execute(
        f"""
        INSERT OR IGNORE INTO message_context_revisions(
            message_id, save_id, revision, updated_at
        )
        SELECT
            messages.id,
            messages.save_id,
            0,
            {timestamp_expr}
        FROM messages
        JOIN saves ON saves.id = messages.save_id
        """
    )


def _ensure_message_context_revision_triggers(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "messages") or not _table_exists(
        connection,
        "message_context_revisions",
    ):
        return
    if not _column_exists(connection, "messages", "save_id"):
        return
    _execute_schema_script(
        connection,
        """
        CREATE TRIGGER IF NOT EXISTS bump_messages_context_revision_after_insert
        AFTER INSERT ON messages
        FOR EACH ROW
        BEGIN
            INSERT INTO save_context_revisions(save_id, revision, updated_at)
            VALUES (NEW.save_id, 1, CURRENT_TIMESTAMP)
            ON CONFLICT(save_id) DO UPDATE SET
                revision = revision + 1,
                updated_at = CURRENT_TIMESTAMP;

            INSERT INTO message_context_revisions(
                message_id, save_id, revision, updated_at
            )
            VALUES (NEW.id, NEW.save_id, 1, CURRENT_TIMESTAMP)
            ON CONFLICT(message_id) DO UPDATE SET
                save_id = excluded.save_id,
                revision = message_context_revisions.revision + 1,
                updated_at = CURRENT_TIMESTAMP;
        END;

        CREATE TRIGGER IF NOT EXISTS bump_messages_context_revision_after_update
        AFTER UPDATE ON messages
        FOR EACH ROW
        BEGIN
            INSERT INTO save_context_revisions(save_id, revision, updated_at)
            VALUES (NEW.save_id, 1, CURRENT_TIMESTAMP)
            ON CONFLICT(save_id) DO UPDATE SET
                revision = revision + 1,
                updated_at = CURRENT_TIMESTAMP;

            INSERT INTO message_context_revisions(
                message_id, save_id, revision, updated_at
            )
            VALUES (NEW.id, NEW.save_id, 1, CURRENT_TIMESTAMP)
            ON CONFLICT(message_id) DO UPDATE SET
                save_id = excluded.save_id,
                revision = message_context_revisions.revision + 1,
                updated_at = CURRENT_TIMESTAMP;
        END;

        CREATE TRIGGER IF NOT EXISTS bump_messages_context_revision_after_delete
        AFTER DELETE ON messages
        FOR EACH ROW
        BEGIN
            INSERT INTO save_context_revisions(save_id, revision, updated_at)
            VALUES (OLD.save_id, 1, CURRENT_TIMESTAMP)
            ON CONFLICT(save_id) DO UPDATE SET
                revision = revision + 1,
                updated_at = CURRENT_TIMESTAMP;

            DELETE FROM message_context_revisions
            WHERE message_id = OLD.id;
        END;
        """
    )


def _message_context_revisions_needs_rebuild(
    connection: sqlite3.Connection,
) -> bool:
    if not _table_exists(connection, "message_context_revisions"):
        return False
    foreign_keys = connection.execute(
        "PRAGMA foreign_key_list(message_context_revisions)"
    ).fetchall()
    message_id_refs = [row for row in foreign_keys if row[3] == "message_id"]
    save_id_refs = [row for row in foreign_keys if row[3] == "save_id"]
    return (
        not message_id_refs
        or any(row[2] != "messages" for row in message_id_refs)
        or not save_id_refs
        or any(row[2] != "saves" for row in save_id_refs)
    )


def _rebuild_message_context_revisions_table(connection: sqlite3.Connection) -> None:
    connection.execute("DROP INDEX IF EXISTS idx_message_context_revisions_save_id")
    connection.execute(
        "ALTER TABLE message_context_revisions RENAME TO message_context_revisions_old"
    )
    _execute_schema_script(
        connection,
        """
        CREATE TABLE message_context_revisions (
            message_id TEXT PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
            save_id TEXT NOT NULL REFERENCES saves(id) ON DELETE CASCADE,
            revision INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO message_context_revisions(
            message_id, save_id, revision, updated_at
        )
        SELECT
            old.message_id,
            old.save_id,
            old.revision,
            old.updated_at
        FROM message_context_revisions_old old
        JOIN messages ON messages.id = old.message_id
        JOIN saves ON saves.id = old.save_id
        """
    )
    connection.execute("DROP TABLE message_context_revisions_old")


def _drop_context_revision_triggers(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'trigger'
          AND (
              name = 'init_save_context_revision_after_save_insert'
              OR name GLOB 'bump_*_context_revision_after_*'
          )
        """
    ).fetchall()
    for (name,) in rows:
        connection.execute(f"DROP TRIGGER IF EXISTS {_quote_identifier(name)}")


def _save_context_revisions_needs_rebuild(connection: sqlite3.Connection) -> bool:
    if not _table_exists(connection, "save_context_revisions"):
        return False
    foreign_keys = connection.execute(
        "PRAGMA foreign_key_list(save_context_revisions)"
    ).fetchall()
    save_id_refs = [row for row in foreign_keys if row[3] == "save_id"]
    return not save_id_refs or any(row[2] != "saves" for row in save_id_refs)


def _rebuild_save_context_revisions_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        "ALTER TABLE save_context_revisions RENAME TO save_context_revisions_old"
    )
    _execute_schema_script(
        connection,
        """
        CREATE TABLE save_context_revisions (
            save_id TEXT PRIMARY KEY REFERENCES saves(id) ON DELETE CASCADE,
            revision INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO save_context_revisions(save_id, revision, updated_at)
        SELECT
            old.save_id,
            old.revision,
            old.updated_at
        FROM save_context_revisions_old old
        JOIN saves ON saves.id = old.save_id
        """
    )
    connection.execute("DROP TABLE save_context_revisions_old")




def _migration_character_key(value: str) -> str:
    return " ".join(value.split()).casefold()




def _add_column_if_missing(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_definition: str,
) -> None:
    if not _table_exists(connection, table_name):
        return
    columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table_name})")}
    if column_name in columns:
        return
    connection.execute(
        f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
    )


def _column_exists(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
) -> bool:
    if not _table_exists(connection, table_name):
        return False
    return column_name in {
        row[1] for row in connection.execute(f"PRAGMA table_info({table_name})")
    }


def _column_names(connection: sqlite3.Connection, table_name: str) -> set[str]:
    if not _table_exists(connection, table_name):
        return set()
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table_name})")}


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        is not None
    )


def _memory_claim_fingerprint_index_is_unique(
    connection: sqlite3.Connection,
) -> bool:
    if not _table_exists(connection, "memories"):
        return True
    return any(
        str(row[1]) == "idx_memories_save_claim_fingerprint_active"
        and bool(row[2])
        for row in connection.execute("PRAGMA index_list('memories')").fetchall()
    )


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'
