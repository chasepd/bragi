"""Shared declarations for tables included in turn snapshots."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SnapshotTable:
    name: str
    primary_key: str = "id"
    active_only: bool = False
    order_by: str = "rowid"


SNAPSHOT_TABLES: tuple[SnapshotTable, ...] = (
    SnapshotTable("messages", order_by="rowid"),
    SnapshotTable("world_state", active_only=True, order_by="key, rowid"),
    SnapshotTable(
        "context_sources",
        active_only=True,
        order_by="source_type, source_id, rowid",
    ),
    SnapshotTable("context_observations", active_only=True),
    SnapshotTable(
        "context_observation_curation_state",
        primary_key="observation_id",
    ),
    SnapshotTable("locations", active_only=True),
    SnapshotTable("characters", active_only=True),
    SnapshotTable("scene_snapshots"),
    SnapshotTable("scene_facts", active_only=True, order_by="created_at, rowid"),
    SnapshotTable("scene_fact_sources", order_by="created_at, rowid"),
    SnapshotTable("active_threads", active_only=True),
    SnapshotTable("entity_links"),
    SnapshotTable("context_update_suggestions"),
    SnapshotTable("context_update_audit"),
    SnapshotTable("state_changes"),
    SnapshotTable("memories", active_only=True),
    SnapshotTable("summaries"),
    SnapshotTable("save_scenario_updates", active_only=True),
    SnapshotTable("save_loss_conditions", active_only=True),
    SnapshotTable("save_loss_condition_changes", active_only=True),
    SnapshotTable("save_loss_outcomes", active_only=True),
    SnapshotTable("media_assets", active_only=True),
    SnapshotTable("character_knowledge_edges", active_only=True),
    SnapshotTable("message_visibility"),
    SnapshotTable("message_scene_presence"),
    SnapshotTable("message_action_choices"),
    SnapshotTable("dating_route_states", active_only=True),
    SnapshotTable("character_text_threads", active_only=True),
    SnapshotTable("character_text_thread_participants", active_only=True),
    SnapshotTable("character_text_messages"),
    SnapshotTable("character_text_activity_events"),
    SnapshotTable(
        "narrator_phone_activity_cursors",
        primary_key="narrator_message_id",
    ),
    SnapshotTable("character_text_message_revisions"),
    SnapshotTable("character_text_message_attachments"),
    SnapshotTable("character_text_provenance"),
    SnapshotTable("character_contact_states", active_only=True),
    SnapshotTable("character_text_proactive_triggers"),
    SnapshotTable("turn_outcomes"),
)

SNAPSHOT_TABLES_BY_NAME = {table.name: table for table in SNAPSHOT_TABLES}
SNAPSHOT_TABLE_NAMES = tuple(table.name for table in SNAPSHOT_TABLES)
