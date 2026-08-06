"""Rollback helpers for message regeneration and edit/resubmit flows."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from difflib import unified_diff
from typing import Any, cast

from bragi.persistence.models import (
    ActiveThreadRecord,
    CharacterRecord,
    ContextSourceRecord,
    ContextUpdateAuditRecord,
    EntityLinkRecord,
    LocationRecord,
    MessageRecord,
    MessageRevisionRecord,
    SceneSnapshotRecord,
    StateChangeRecord,
    SummaryRecord,
    WorldStateRecord,
)
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.turn_snapshot_service import TurnSnapshotService
from bragi.world_time_model import canonical_world_time_from_legacy

_SCENE_WORLD_TIME_FIELDS = frozenset(
    {
        "in_world_time",
        "time_of_day",
        "day_of_week",
        "world_day_index",
    }
)


@dataclass(frozen=True)
class RevisionResubmission:
    body: str
    speaker_name: str | None
    anchor_message_id: str
    deleted_messages: tuple[MessageRecord, ...]
    archived_memory_ids: frozenset[str]
    archived_summary_ids: frozenset[str]
    archived_scenario_update_ids: frozenset[str]
    archived_loss_condition_change_ids: frozenset[str] = frozenset()
    archived_loss_outcome_ids: frozenset[str] = frozenset()
    expired_context_update_suggestion_ids: frozenset[str] = frozenset()
    archived_context_source_ids: frozenset[str] = frozenset()
    archived_context_observation_ids: frozenset[str] = frozenset()
    archived_location_ids: frozenset[str] = frozenset()
    archived_character_ids: frozenset[str] = frozenset()
    archived_active_thread_ids: frozenset[str] = frozenset()
    deleted_entity_link_ids: frozenset[str] = frozenset()
    deleted_entity_links: tuple[EntityLinkRecord, ...] = ()
    scene_snapshot_before_cleanup: SceneSnapshotRecord | None = None
    scene_snapshot_cleanup_changed: bool = False
    archived_character_knowledge_edge_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class MessageRollback:
    deleted_messages: tuple[MessageRecord, ...]
    archived_memory_ids: frozenset[str]
    archived_summary_ids: frozenset[str]
    archived_scenario_update_ids: frozenset[str]
    archived_loss_condition_change_ids: frozenset[str]
    archived_loss_outcome_ids: frozenset[str]
    expired_context_update_suggestion_ids: frozenset[str]
    archived_context_source_ids: frozenset[str]
    archived_context_observation_ids: frozenset[str]
    archived_location_ids: frozenset[str]
    archived_character_ids: frozenset[str]
    archived_active_thread_ids: frozenset[str]
    deleted_entity_link_ids: frozenset[str]
    deleted_entity_links: tuple[EntityLinkRecord, ...]
    scene_snapshot_before_cleanup: SceneSnapshotRecord | None = None
    scene_snapshot_cleanup_changed: bool = False
    archived_character_knowledge_edge_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class MessageDeletion:
    anchor_message_id: str
    deleted_messages: tuple[MessageRecord, ...]
    archived_memory_ids: frozenset[str]
    archived_summary_ids: frozenset[str]
    archived_scenario_update_ids: frozenset[str]
    archived_loss_condition_change_ids: frozenset[str]
    archived_loss_outcome_ids: frozenset[str]
    archived_media_asset_ids: frozenset[str]
    expired_context_update_suggestion_ids: frozenset[str]
    archived_context_source_ids: frozenset[str]
    archived_context_observation_ids: frozenset[str]
    archived_location_ids: frozenset[str]
    archived_character_ids: frozenset[str]
    archived_active_thread_ids: frozenset[str]
    deleted_entity_link_ids: frozenset[str] = frozenset()
    deleted_scene_snapshot_id: str | None = None
    archived_character_knowledge_edge_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _WorldDataContextCleanup:
    expired_context_update_suggestion_ids: frozenset[str]
    archived_context_sources: tuple[ContextSourceRecord, ...]
    archived_context_observation_ids: frozenset[str]
    archived_locations: tuple[LocationRecord, ...]
    archived_characters: tuple[CharacterRecord, ...]
    archived_active_threads: tuple[ActiveThreadRecord, ...]
    deleted_entity_links: tuple[EntityLinkRecord, ...]
    scene_snapshot_before_cleanup: SceneSnapshotRecord | None = None
    scene_snapshot_cleanup_changed: bool = False
    archived_character_knowledge_edge_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class MessageEdit:
    message: MessageRecord
    revision: MessageRevisionRecord
    previous_body: str


class MessageRevisionService:
    def __init__(self, repositories: PersistenceRepositories) -> None:
        self.repositories = repositories

    def regenerate_message(
        self,
        *,
        save_id: str,
        message_id: str,
    ) -> RevisionResubmission:
        messages = self.repositories.list_messages(save_id)
        selected = _find_message(messages, message_id)
        if selected is None:
            raise ValueError(f"Unknown active message id: {message_id}")

        if selected.role == "player":
            anchor = selected
        elif selected.role == "narrator":
            previous_player = _previous_player_message(messages, selected)
            if previous_player is None:
                raise ValueError(
                    "Cannot regenerate a narrator message without a prior player turn"
                )
            anchor = previous_player
        else:
            raise ValueError(f"Cannot regenerate message role: {selected.role}")

        rollback = self.rollback_from_message(
            save_id=save_id,
            message_id=anchor.id,
        )
        return RevisionResubmission(
            body=anchor.body,
            speaker_name=anchor.speaker_name,
            anchor_message_id=anchor.id,
            deleted_messages=rollback.deleted_messages,
            archived_memory_ids=rollback.archived_memory_ids,
            archived_summary_ids=rollback.archived_summary_ids,
            archived_scenario_update_ids=rollback.archived_scenario_update_ids,
            archived_loss_condition_change_ids=(
                rollback.archived_loss_condition_change_ids
            ),
            archived_loss_outcome_ids=rollback.archived_loss_outcome_ids,
            expired_context_update_suggestion_ids=(
                rollback.expired_context_update_suggestion_ids
            ),
            archived_context_source_ids=rollback.archived_context_source_ids,
            archived_context_observation_ids=(
                rollback.archived_context_observation_ids
            ),
            archived_location_ids=rollback.archived_location_ids,
            archived_character_ids=rollback.archived_character_ids,
            archived_active_thread_ids=rollback.archived_active_thread_ids,
            deleted_entity_link_ids=rollback.deleted_entity_link_ids,
            deleted_entity_links=rollback.deleted_entity_links,
            scene_snapshot_before_cleanup=rollback.scene_snapshot_before_cleanup,
            scene_snapshot_cleanup_changed=rollback.scene_snapshot_cleanup_changed,
            archived_character_knowledge_edge_ids=(
                rollback.archived_character_knowledge_edge_ids
            ),
        )

    def edit_and_resubmit_message(
        self,
        *,
        save_id: str,
        message_id: str,
        body: str,
    ) -> RevisionResubmission:
        replacement = body.strip()
        if not replacement:
            raise ValueError("Message is empty")
        messages = self.repositories.list_messages(save_id)
        selected = _find_message(messages, message_id)
        if selected is None:
            raise ValueError(f"Unknown active message id: {message_id}")
        if selected.role != "player":
            raise ValueError("Only player messages can be edited and resubmitted")

        rollback = self.rollback_from_message(
            save_id=save_id,
            message_id=selected.id,
        )
        return RevisionResubmission(
            body=replacement,
            speaker_name=selected.speaker_name,
            anchor_message_id=selected.id,
            deleted_messages=rollback.deleted_messages,
            archived_memory_ids=rollback.archived_memory_ids,
            archived_summary_ids=rollback.archived_summary_ids,
            archived_scenario_update_ids=rollback.archived_scenario_update_ids,
            archived_loss_condition_change_ids=(
                rollback.archived_loss_condition_change_ids
            ),
            archived_loss_outcome_ids=rollback.archived_loss_outcome_ids,
            expired_context_update_suggestion_ids=(
                rollback.expired_context_update_suggestion_ids
            ),
            archived_context_source_ids=rollback.archived_context_source_ids,
            archived_context_observation_ids=(
                rollback.archived_context_observation_ids
            ),
            archived_location_ids=rollback.archived_location_ids,
            archived_character_ids=rollback.archived_character_ids,
            archived_active_thread_ids=rollback.archived_active_thread_ids,
            deleted_entity_link_ids=rollback.deleted_entity_link_ids,
            deleted_entity_links=rollback.deleted_entity_links,
            scene_snapshot_before_cleanup=rollback.scene_snapshot_before_cleanup,
            scene_snapshot_cleanup_changed=rollback.scene_snapshot_cleanup_changed,
            archived_character_knowledge_edge_ids=(
                rollback.archived_character_knowledge_edge_ids
            ),
        )

    def edit_narrator_message(
        self,
        *,
        save_id: str,
        message_id: str,
        body: str,
        current_user_id: str | None = None,
        content_rating: str = "unclassified",
        safety_transition: str = "",
    ) -> MessageEdit:
        return self._edit_message_body(
            save_id=save_id,
            message_id=message_id,
            body=body,
            allowed_roles=frozenset({"narrator"}),
            role_error="Only narrator messages can be edited this way",
            current_user_id=current_user_id,
            content_rating=content_rating,
            safety_transition=safety_transition,
        )

    def edit_message_without_resubmit(
        self,
        *,
        save_id: str,
        message_id: str,
        body: str,
        current_user_id: str | None = None,
        content_rating: str = "unclassified",
        safety_transition: str = "",
    ) -> MessageEdit:
        return self._edit_message_body(
            save_id=save_id,
            message_id=message_id,
            body=body,
            allowed_roles=frozenset({"player", "narrator"}),
            role_error="Only player and narrator messages can be edited this way",
            current_user_id=current_user_id,
            content_rating=content_rating,
            safety_transition=safety_transition,
        )

    def _edit_message_body(
        self,
        *,
        save_id: str,
        message_id: str,
        body: str,
        allowed_roles: frozenset[str],
        role_error: str,
        current_user_id: str | None,
        content_rating: str,
        safety_transition: str,
    ) -> MessageEdit:
        replacement = body.strip()
        if not replacement:
            raise ValueError("Message is empty")
        messages = self.repositories.list_messages(save_id)
        selected = _find_message(messages, message_id)
        if selected is None:
            raise ValueError(f"Unknown active message id: {message_id}")
        if selected.role not in allowed_roles:
            raise ValueError(role_error)
        if selected.body.strip() == replacement:
            raise ValueError("Message was not changed")

        del current_user_id
        updated = self.repositories.update_message_body(
            save_id=save_id,
            message_id=message_id,
            body=replacement,
            content_rating=content_rating,
            safety_transition=safety_transition,
        )
        revision = self.repositories.add_message_revision(
            save_id=save_id,
            message_id=message_id,
            previous_body=selected.body,
            new_body=updated.body,
            diff_unified=_message_diff(selected.body, updated.body),
            reconciliation_status="queued",
        )
        self._archive_summaries_covering_message(
            save_id=save_id,
            message_id=message_id,
        )
        return MessageEdit(
            message=updated,
            revision=revision,
            previous_body=selected.body,
        )

    def delete_suffix_from_message(
        self,
        *,
        save_id: str,
        message_id: str,
    ) -> MessageDeletion:
        messages = self.repositories.list_messages(save_id)
        selected = _find_message(messages, message_id)
        if selected is None:
            raise ValueError(f"Unknown active message id: {message_id}")
        anchor = _deletion_anchor_message(messages, selected)

        snapshot_service = TurnSnapshotService(self.repositories)
        snapshot_service.capture_current_head_if_dirty(
            save_id,
            reason="pre_delete_head",
        )
        protected_characters_before_snapshot_restore = tuple(
            character
            for character in self.repositories.list_characters(save_id)
            if character.protected_from_maintenance
        )
        snapshot_deletion = snapshot_service.restore_delete_from_message(
            save_id=save_id,
            message_id=anchor.id,
        )
        if snapshot_deletion is not None:
            self._restore_missing_protected_characters(
                save_id=save_id,
                characters=protected_characters_before_snapshot_restore,
            )
            return MessageDeletion(
                anchor_message_id=anchor.id,
                deleted_messages=snapshot_deletion.deleted_messages,
                archived_memory_ids=frozenset(),
                archived_summary_ids=frozenset(),
                archived_scenario_update_ids=frozenset(),
                archived_loss_condition_change_ids=frozenset(),
                archived_loss_outcome_ids=frozenset(),
                archived_media_asset_ids=frozenset(),
                expired_context_update_suggestion_ids=frozenset(),
                archived_context_source_ids=frozenset(),
                archived_context_observation_ids=frozenset(),
                archived_location_ids=frozenset(),
                archived_character_ids=frozenset(),
                archived_active_thread_ids=frozenset(),
            )

        rollback = self.rollback_from_message(
            save_id=save_id,
            message_id=anchor.id,
        )
        deleted_message_ids = frozenset(
            message.id for message in rollback.deleted_messages
        )
        archived_media_asset_ids = self.repositories.archive_media_assets_for_messages(
            save_id=save_id,
            message_ids=deleted_message_ids,
        )
        deleted_scene_snapshot_id = (
            rollback.scene_snapshot_before_cleanup.id
            if rollback.scene_snapshot_cleanup_changed
            and rollback.scene_snapshot_before_cleanup is not None
            and self.repositories.get_scene_snapshot(save_id) is None
            else None
        )
        snapshot_service.capture_current_head_if_dirty(
            save_id,
            reason="legacy_delete_result",
        )
        return MessageDeletion(
            anchor_message_id=anchor.id,
            deleted_messages=rollback.deleted_messages,
            archived_memory_ids=rollback.archived_memory_ids,
            archived_summary_ids=rollback.archived_summary_ids,
            archived_scenario_update_ids=rollback.archived_scenario_update_ids,
            archived_loss_condition_change_ids=(
                rollback.archived_loss_condition_change_ids
            ),
            archived_loss_outcome_ids=rollback.archived_loss_outcome_ids,
            archived_media_asset_ids=archived_media_asset_ids,
            expired_context_update_suggestion_ids=(
                rollback.expired_context_update_suggestion_ids
            ),
            archived_context_source_ids=rollback.archived_context_source_ids,
            archived_context_observation_ids=(
                rollback.archived_context_observation_ids
            ),
            archived_location_ids=rollback.archived_location_ids,
            archived_character_ids=rollback.archived_character_ids,
            archived_active_thread_ids=rollback.archived_active_thread_ids,
            deleted_entity_link_ids=rollback.deleted_entity_link_ids,
            deleted_scene_snapshot_id=deleted_scene_snapshot_id,
            archived_character_knowledge_edge_ids=(
                rollback.archived_character_knowledge_edge_ids
            ),
        )

    def rollback_from_message(
        self,
        *,
        save_id: str,
        message_id: str,
    ) -> MessageRollback:
        deleted_messages = self.repositories.archive_messages_from(
            save_id=save_id,
            message_id=message_id,
        )
        if not deleted_messages:
            raise ValueError(f"Unknown active message id: {message_id}")

        deleted_message_ids = frozenset(message.id for message in deleted_messages)
        self._restore_world_state(
            save_id=save_id,
            deleted_message_ids=deleted_message_ids,
        )
        archived_memory_ids = self._archive_memories(
            save_id=save_id,
            deleted_message_ids=deleted_message_ids,
        )
        archived_summary_ids = self._archive_summaries(
            save_id=save_id,
            deleted_message_ids=deleted_message_ids,
        )
        archived_scenario_update_ids = self._archive_scenario_updates(
            save_id=save_id,
            deleted_message_ids=deleted_message_ids,
        )
        archived_loss_condition_change_ids = self._archive_loss_condition_changes(
            save_id=save_id,
            deleted_message_ids=deleted_message_ids,
        )
        archived_loss_outcome_ids = self._archive_loss_outcomes(
            save_id=save_id,
            deleted_message_ids=deleted_message_ids,
        )
        cleanup = self._cleanup_deleted_message_context(
            save_id=save_id,
            deleted_message_ids=deleted_message_ids,
        )
        self._rebuild_loss_conditions(save_id=save_id)
        self._sync_continuity_index(save_id=save_id)
        return MessageRollback(
            deleted_messages=tuple(deleted_messages),
            archived_memory_ids=archived_memory_ids,
            archived_summary_ids=archived_summary_ids,
            archived_scenario_update_ids=archived_scenario_update_ids,
            archived_loss_condition_change_ids=archived_loss_condition_change_ids,
            archived_loss_outcome_ids=archived_loss_outcome_ids,
            expired_context_update_suggestion_ids=(
                cleanup.expired_context_update_suggestion_ids
            ),
            archived_context_source_ids=frozenset(
                source.id for source in cleanup.archived_context_sources
            ),
            archived_context_observation_ids=(
                cleanup.archived_context_observation_ids
            ),
            archived_location_ids=frozenset(
                location.id for location in cleanup.archived_locations
            ),
            archived_character_ids=frozenset(
                character.id for character in cleanup.archived_characters
            ),
            archived_active_thread_ids=frozenset(
                thread.id for thread in cleanup.archived_active_threads
            ),
            deleted_entity_link_ids=frozenset(
                link.id for link in cleanup.deleted_entity_links
            ),
            deleted_entity_links=cleanup.deleted_entity_links,
            scene_snapshot_before_cleanup=cleanup.scene_snapshot_before_cleanup,
            scene_snapshot_cleanup_changed=cleanup.scene_snapshot_cleanup_changed,
            archived_character_knowledge_edge_ids=(
                cleanup.archived_character_knowledge_edge_ids
            ),
        )

    def restore_resubmission(
        self,
        *,
        save_id: str,
        revision: RevisionResubmission,
        active_message_ids_before_resubmission: frozenset[str],
        active_summary_ids_before_resubmission: frozenset[str],
    ) -> None:
        restored_message_ids = frozenset(
            message.id for message in revision.deleted_messages
        )
        replacement_message_ids = frozenset(
            message.id
            for message in self.repositories.list_messages(save_id)
            if message.id not in active_message_ids_before_resubmission
        )

        for message_id in replacement_message_ids:
            self.repositories.archive_message(message_id)
        self._archive_memories(
            save_id=save_id,
            deleted_message_ids=replacement_message_ids,
        )
        self._archive_summaries(
            save_id=save_id,
            deleted_message_ids=replacement_message_ids,
        )
        self._archive_scenario_updates(
            save_id=save_id,
            deleted_message_ids=replacement_message_ids,
        )
        self._archive_loss_condition_changes(
            save_id=save_id,
            deleted_message_ids=replacement_message_ids,
        )
        self._archive_loss_outcomes(
            save_id=save_id,
            deleted_message_ids=replacement_message_ids,
        )
        self._archive_new_summaries(
            save_id=save_id,
            active_summary_ids_before_resubmission=(
                active_summary_ids_before_resubmission
            ),
        )
        self._restore_world_state(
            save_id=save_id,
            deleted_message_ids=replacement_message_ids,
        )
        self._cleanup_deleted_message_context(
            save_id=save_id,
            deleted_message_ids=replacement_message_ids,
        )

        self.repositories.restore_messages(restored_message_ids)
        self.repositories.restore_memories(revision.archived_memory_ids)
        self.repositories.restore_summaries(
            revision.archived_summary_ids | active_summary_ids_before_resubmission
        )
        self.repositories.restore_save_scenario_updates(
            revision.archived_scenario_update_ids
        )
        self.repositories.restore_loss_condition_changes(
            revision.archived_loss_condition_change_ids
        )
        self.repositories.restore_loss_outcomes(revision.archived_loss_outcome_ids)
        self.repositories.restore_context_update_suggestions(
            revision.expired_context_update_suggestion_ids
        )
        self.repositories.restore_context_sources(revision.archived_context_source_ids)
        self.repositories.restore_context_observations(
            revision.archived_context_observation_ids
        )
        self.repositories.restore_locations(revision.archived_location_ids)
        self.repositories.restore_characters(revision.archived_character_ids)
        self.repositories.restore_active_threads(revision.archived_active_thread_ids)
        self.repositories.restore_character_knowledge_edges(
            revision.archived_character_knowledge_edge_ids
        )
        if (
            revision.scene_snapshot_cleanup_changed
            and revision.scene_snapshot_before_cleanup is not None
        ):
            self._restore_scene_snapshot(
                save_id=save_id,
                snapshot=revision.scene_snapshot_before_cleanup,
            )
        self.repositories.restore_entity_links(revision.deleted_entity_links)
        self._restore_restored_world_state(
            save_id=save_id,
            restored_message_ids=restored_message_ids,
            excluded_message_ids=replacement_message_ids,
        )
        self._rebuild_loss_conditions(save_id=save_id)
        self._sync_continuity_index(save_id=save_id)

    def _restore_world_state(
        self,
        *,
        save_id: str,
        deleted_message_ids: frozenset[str],
    ) -> None:
        changes = self.repositories.list_state_changes(save_id)
        touched_keys = {
            change.state_key
            for change in changes
            if change.source_message_id in deleted_message_ids
        }
        if not touched_keys:
            return

        deleted_changes = [
            change
            for change in changes
            if change.source_message_id in deleted_message_ids
        ]
        remaining_changes = [
            change
            for change in changes
            if change.source_message_id not in deleted_message_ids
        ]
        for state_key in touched_keys:
            last_change = _last_change_for_key(remaining_changes, state_key)
            if last_change is not None:
                if last_change.after_json is None:
                    self.repositories.archive_world_state(
                        save_id=save_id,
                        key=state_key,
                    )
                    continue
                current_state = _active_world_state_for_key(
                    self.repositories,
                    save_id=save_id,
                    key=state_key,
                )
                if (
                    current_state is not None
                    and current_state.value == _load_state_value(
                        last_change.after_json
                    )
                ):
                    continue
                self.repositories.upsert_world_state(
                    save_id=save_id,
                    key=state_key,
                    value=_load_state_value(last_change.after_json),
                    source_message_id=last_change.source_message_id,
                )
                continue

            first_deleted_change = _first_change_for_key(deleted_changes, state_key)
            if (
                first_deleted_change is not None
                and first_deleted_change.before_json is not None
            ):
                self.repositories.upsert_world_state(
                    save_id=save_id,
                    key=state_key,
                    value=_load_state_value(first_deleted_change.before_json),
                )
                continue

            self.repositories.archive_world_state(
                save_id=save_id,
                key=state_key,
            )

    def _archive_memories(
        self,
        *,
        save_id: str,
        deleted_message_ids: frozenset[str],
    ) -> frozenset[str]:
        archived_memory_ids: set[str] = set()
        for memory in self.repositories.list_memories(save_id):
            source_ids = set(memory.source_message_ids)
            if memory.source_message_id is not None:
                source_ids.add(memory.source_message_id)
            if source_ids & deleted_message_ids:
                self.repositories.archive_memory(memory.id)
                archived_memory_ids.add(memory.id)
        return frozenset(archived_memory_ids)

    def _archive_summaries(
        self,
        *,
        save_id: str,
        deleted_message_ids: frozenset[str],
    ) -> frozenset[str]:
        archived_summary_ids: set[str] = set()
        for summary in self.repositories.list_summaries(save_id):
            if (
                summary.covers_message_start_id in deleted_message_ids
                or summary.covers_message_end_id in deleted_message_ids
            ):
                self.repositories.archive_summary(summary.id)
                archived_summary_ids.add(summary.id)
        return frozenset(archived_summary_ids)

    def _archive_summaries_covering_message(
        self,
        *,
        save_id: str,
        message_id: str,
    ) -> frozenset[str]:
        messages = tuple(self.repositories.list_messages(save_id))
        message_order = {message.id: index for index, message in enumerate(messages)}
        if message_id not in message_order:
            return frozenset()
        archived_summary_ids: set[str] = set()
        for summary in self.repositories.list_summaries(save_id):
            if not _summary_covers_message(
                summary=summary,
                message_order=message_order,
                message_id=message_id,
            ):
                continue
            self.repositories.archive_summary(summary.id)
            archived_summary_ids.add(summary.id)
        return frozenset(archived_summary_ids)

    def _archive_scenario_updates(
        self,
        *,
        save_id: str,
        deleted_message_ids: frozenset[str],
    ) -> frozenset[str]:
        return self.repositories.archive_save_scenario_updates_for_messages(
            save_id=save_id,
            message_ids=deleted_message_ids,
        )

    def _archive_loss_condition_changes(
        self,
        *,
        save_id: str,
        deleted_message_ids: frozenset[str],
    ) -> frozenset[str]:
        return self.repositories.archive_loss_condition_changes_for_messages(
            save_id=save_id,
            message_ids=deleted_message_ids,
        )

    def _archive_loss_outcomes(
        self,
        *,
        save_id: str,
        deleted_message_ids: frozenset[str],
    ) -> frozenset[str]:
        return self.repositories.archive_loss_outcomes_for_messages(
            save_id=save_id,
            message_ids=deleted_message_ids,
        )

    def _rebuild_loss_conditions(self, *, save_id: str) -> None:
        self.repositories.rebuild_save_loss_conditions_from_changes(save_id)

    def _sync_continuity_index(self, *, save_id: str) -> None:
        from bragi.services.continuity_index_service import ContinuityIndexService

        ContinuityIndexService(self.repositories).sync_save(save_id)

    def _cleanup_deleted_message_context(
        self,
        *,
        save_id: str,
        deleted_message_ids: frozenset[str],
    ) -> _WorldDataContextCleanup:
        if not deleted_message_ids:
            return _WorldDataContextCleanup(
                expired_context_update_suggestion_ids=frozenset(),
                archived_context_sources=(),
                archived_context_observation_ids=frozenset(),
                archived_locations=(),
                archived_characters=(),
                archived_active_threads=(),
                deleted_entity_links=(),
            )
        context_sources = self.repositories.list_context_sources(save_id)
        expired_suggestion_ids = (
            self.repositories.expire_context_update_suggestions_for_messages(
                save_id=save_id,
                message_ids=deleted_message_ids,
            )
        )
        archived_context_source_ids = (
            self.repositories.archive_context_sources_for_deleted_messages(
                save_id=save_id,
                message_ids=deleted_message_ids,
            )
        )
        archived_context_observation_ids = (
            self.repositories.archive_context_observations_for_deleted_messages(
                save_id=save_id,
                message_ids=deleted_message_ids,
            )
        )
        archived_context_sources = tuple(
            source
            for source in context_sources
            if source.id in archived_context_source_ids
        )
        deleted_links = list(
            self.repositories.delete_entity_links_for_source_messages(
                save_id=save_id,
                source_message_ids=deleted_message_ids,
            )
        )
        archived_locations = self._archive_deleted_source_locations(
            save_id=save_id,
            deleted_message_ids=deleted_message_ids,
            deleted_links=deleted_links,
        )
        archived_characters = self._archive_deleted_source_characters(
            save_id=save_id,
            deleted_message_ids=deleted_message_ids,
            deleted_links=deleted_links,
        )
        archived_active_threads = self._archive_deleted_source_threads(
            save_id=save_id,
            deleted_message_ids=deleted_message_ids,
            deleted_links=deleted_links,
        )
        self._revert_deleted_context_updates(
            save_id=save_id,
            deleted_message_ids=deleted_message_ids,
        )
        archived_character_knowledge_edge_ids = (
            self.repositories.archive_character_knowledge_edges_for_deleted_messages(
                save_id=save_id,
                message_ids=deleted_message_ids,
            )
        )
        snapshot_after_revert = self.repositories.get_scene_snapshot(save_id)
        scene_changed = self._cleanup_scene_snapshot_references(
            save_id=save_id,
            snapshot=snapshot_after_revert,
            deleted_message_ids=deleted_message_ids,
            archived_location_ids=frozenset(
                location.id for location in archived_locations
            ),
            archived_character_ids=frozenset(
                character.id for character in archived_characters
            ),
        )
        deleted_links.extend(
            self.repositories.delete_entity_links_for_inactive_stored_endpoints(save_id)
        )
        return _WorldDataContextCleanup(
            expired_context_update_suggestion_ids=expired_suggestion_ids,
            archived_context_sources=archived_context_sources,
            archived_context_observation_ids=archived_context_observation_ids,
            archived_locations=archived_locations,
            archived_characters=archived_characters,
            archived_active_threads=archived_active_threads,
            deleted_entity_links=_dedupe_entity_links(deleted_links),
            scene_snapshot_before_cleanup=(
                snapshot_after_revert if scene_changed else None
            ),
            scene_snapshot_cleanup_changed=scene_changed,
            archived_character_knowledge_edge_ids=(
                archived_character_knowledge_edge_ids
            ),
        )

    def _restore_missing_protected_characters(
        self,
        *,
        save_id: str,
        characters: tuple[CharacterRecord, ...],
    ) -> None:
        if not characters:
            return
        active_character_ids = {
            character.id for character in self.repositories.list_characters(save_id)
        }
        active_location_ids = {
            location.id for location in self.repositories.list_locations(save_id)
        }
        for character in characters:
            if character.id in active_character_ids:
                continue
            self.repositories.add_character(
                save_id=save_id,
                name=character.name,
                aliases=character.aliases,
                role=character.role,
                age=character.age,
                known_state=character.known_state,
                history=character.history,
                met=character.met,
                appearance=character.appearance,
                visual_notes=character.visual_notes,
                current_clothing=character.current_clothing,
                personality=character.personality,
                voice=character.voice,
                relationships=character.relationships,
                goals=character.goals,
                motivations=character.motivations,
                current_intent=character.current_intent,
                boundaries=character.boundaries,
                attitude_toward_player=character.attitude_toward_player,
                cooperation_conditions=character.cooperation_conditions,
                status=character.status,
                location_id=(
                    character.location_id
                    if character.location_id in active_location_ids
                    else None
                ),
                private_notes=character.private_notes,
                source_message_id=character.source_message_id,
                locked_fields=character.locked_fields,
                protected_from_maintenance=True,
                is_player_character=character.is_player_character,
                texting_style=character.texting_style,
                contact_name=character.contact_name,
                character_id=character.id,
                first_seen_message_id=character.first_seen_message_id,
                last_updated_message_id=character.last_updated_message_id,
                content_rating=character.content_rating,
            )
            active_character_ids.add(character.id)

    def _archive_deleted_source_locations(
        self,
        *,
        save_id: str,
        deleted_message_ids: frozenset[str],
        deleted_links: list[EntityLinkRecord],
    ) -> tuple[LocationRecord, ...]:
        archived: list[LocationRecord] = []
        links = self.repositories.list_entity_links(save_id)
        seen_link_ids = {link.id for link in deleted_links}
        for location in self.repositories.list_locations(save_id):
            if _record_first_seen_message_id(location) not in deleted_message_ids:
                continue
            for link in _links_for_endpoint(
                links,
                entity_type="location",
                entity_id=location.id,
            ):
                if link.id not in seen_link_ids:
                    deleted_links.append(link)
                    seen_link_ids.add(link.id)
            self.repositories.delete_entity_links_for_endpoint(
                save_id=save_id,
                entity_type="location",
                entity_id=location.id,
            )
            self.repositories.archive_location(location.id)
            archived.append(location)
        return tuple(archived)

    def _archive_deleted_source_characters(
        self,
        *,
        save_id: str,
        deleted_message_ids: frozenset[str],
        deleted_links: list[EntityLinkRecord],
    ) -> tuple[CharacterRecord, ...]:
        archived: list[CharacterRecord] = []
        links = self.repositories.list_entity_links(save_id)
        seen_link_ids = {link.id for link in deleted_links}
        for character in self.repositories.list_characters(save_id):
            if _record_first_seen_message_id(character) not in deleted_message_ids:
                continue
            if character.protected_from_maintenance:
                continue
            for link in _links_for_endpoint(
                links,
                entity_type="character",
                entity_id=character.id,
            ):
                if link.id not in seen_link_ids:
                    deleted_links.append(link)
                    seen_link_ids.add(link.id)
            self.repositories.delete_entity_links_for_endpoint(
                save_id=save_id,
                entity_type="character",
                entity_id=character.id,
            )
            self.repositories.archive_character(character.id)
            archived.append(character)
        return tuple(archived)

    def _archive_deleted_source_threads(
        self,
        *,
        save_id: str,
        deleted_message_ids: frozenset[str],
        deleted_links: list[EntityLinkRecord],
    ) -> tuple[ActiveThreadRecord, ...]:
        archived: list[ActiveThreadRecord] = []
        links = self.repositories.list_entity_links(save_id)
        seen_link_ids = {link.id for link in deleted_links}
        for thread in self.repositories.list_active_threads(save_id):
            if _record_first_seen_message_id(thread) not in deleted_message_ids:
                continue
            for link in _links_for_endpoint(
                links,
                entity_type="active_thread",
                entity_id=thread.id,
            ):
                if link.id not in seen_link_ids:
                    deleted_links.append(link)
                    seen_link_ids.add(link.id)
            self.repositories.delete_entity_links_for_endpoint(
                save_id=save_id,
                entity_type="active_thread",
                entity_id=thread.id,
            )
            self.repositories.archive_active_thread(thread.id)
            archived.append(thread)
        return tuple(archived)

    def _cleanup_scene_snapshot_references(
        self,
        *,
        save_id: str,
        snapshot: SceneSnapshotRecord | None,
        deleted_message_ids: frozenset[str],
        archived_location_ids: frozenset[str],
        archived_character_ids: frozenset[str],
    ) -> bool:
        if snapshot is None:
            return False
        if _record_first_seen_message_id(snapshot) in deleted_message_ids:
            self.repositories.delete_scene_snapshot(save_id)
            return True
        current_location_id = (
            None
            if snapshot.current_location_id in archived_location_ids
            else snapshot.current_location_id
        )
        present_character_ids = [
            character_id
            for character_id in snapshot.present_character_ids
            if character_id not in archived_character_ids
        ]
        if (
            current_location_id == snapshot.current_location_id
            and present_character_ids == snapshot.present_character_ids
        ):
            return False
        self.repositories.upsert_scene_snapshot(
            save_id=save_id,
            current_location_id=current_location_id,
            situation=snapshot.situation,
            objective=snapshot.objective,
            in_world_time=snapshot.in_world_time,
            time_of_day=snapshot.time_of_day,
            day_of_week=snapshot.day_of_week,
            world_day_index=snapshot.world_day_index,
            world_time_source_message_id=snapshot.world_time_source_message_id,
            world_time_confidence=snapshot.world_time_confidence,
            weather=snapshot.weather,
            mood=snapshot.mood,
            nearby_objects=snapshot.nearby_objects,
            hazards=snapshot.hazards,
            present_character_ids=present_character_ids,
            source_message_id=snapshot.source_message_id,
            locked_fields=snapshot.locked_fields,
            snapshot_id=snapshot.id,
            first_seen_message_id=snapshot.first_seen_message_id,
            last_updated_message_id=snapshot.last_updated_message_id,
        )
        return True

    def _revert_deleted_context_updates(
        self,
        *,
        save_id: str,
        deleted_message_ids: frozenset[str],
    ) -> None:
        audit_rows = self.repositories.list_context_update_audit(save_id)
        for audit in reversed(audit_rows):
            if audit.operation != "updated":
                continue
            if not set(audit.source_message_ids) & deleted_message_ids:
                continue
            self._revert_context_update_audit(
                save_id=save_id,
                audit=audit,
                deleted_message_ids=deleted_message_ids,
                audit_rows=audit_rows,
            )

    def _revert_context_update_audit(
        self,
        *,
        save_id: str,
        audit: ContextUpdateAuditRecord,
        deleted_message_ids: frozenset[str],
        audit_rows: list[ContextUpdateAuditRecord],
    ) -> None:
        if audit.entity_id is None or audit.field_path == "*":
            return
        if audit.entity_type == "location":
            location = self.repositories.get_location(audit.entity_id)
            if location is None or location.save_id != save_id:
                return
            if _record_first_seen_message_id(location) in deleted_message_ids:
                return
            if getattr(location, audit.field_path) != audit.after:
                return
            source_message_id = _latest_remaining_context_update_source(
                record=location,
                audit_rows=audit_rows,
                deleted_message_ids=deleted_message_ids,
            )
            self.repositories.update_location(
                replace(
                    location,
                    **cast(Any, {
                        audit.field_path: audit.before,
                        "source_message_id": source_message_id,
                        "last_updated_message_id": source_message_id,
                    }),
                )
            )
            return
        if audit.entity_type == "character":
            character = self.repositories.get_character(audit.entity_id)
            if character is None or character.save_id != save_id:
                return
            if _record_first_seen_message_id(character) in deleted_message_ids:
                return
            if getattr(character, audit.field_path) != audit.after:
                return
            source_message_id = _latest_remaining_context_update_source(
                record=character,
                audit_rows=audit_rows,
                deleted_message_ids=deleted_message_ids,
            )
            self.repositories.update_character(
                replace(
                    character,
                    **cast(Any, {
                        audit.field_path: audit.before,
                        "source_message_id": source_message_id,
                        "last_updated_message_id": source_message_id,
                    }),
                )
            )
            return
        if audit.entity_type == "active_thread":
            thread = self.repositories.get_active_thread(audit.entity_id)
            if thread is None or thread.save_id != save_id:
                return
            if _record_first_seen_message_id(thread) in deleted_message_ids:
                return
            if getattr(thread, audit.field_path) != audit.after:
                return
            source_message_id = _latest_remaining_context_update_source(
                record=thread,
                audit_rows=audit_rows,
                deleted_message_ids=deleted_message_ids,
            )
            self.repositories.update_active_thread(
                replace(
                    thread,
                    **cast(Any, {
                        audit.field_path: audit.before,
                        "source_message_id": source_message_id,
                        "last_updated_message_id": source_message_id,
                    }),
                )
            )
            return
        if audit.entity_type != "scene_snapshot":
            return
        snapshot = self.repositories.get_scene_snapshot(save_id)
        if snapshot is None or snapshot.id != audit.entity_id:
            return
        if _record_first_seen_message_id(snapshot) in deleted_message_ids:
            return
        if getattr(snapshot, audit.field_path) != audit.after:
            return
        source_message_id = _latest_remaining_context_update_source(
            record=snapshot,
            audit_rows=audit_rows,
            deleted_message_ids=deleted_message_ids,
        )
        reverted = _replace_scene_snapshot_field(
            snapshot,
            audit.field_path,
            audit.before,
        )
        world_time_kwargs: dict[str, Any] = {}
        if audit.field_path in _SCENE_WORLD_TIME_FIELDS:
            world_time_confidence = _latest_remaining_context_update_confidence(
                record=snapshot,
                audit_rows=audit_rows,
                deleted_message_ids=deleted_message_ids,
                field_paths=_SCENE_WORLD_TIME_FIELDS,
            )
            canonical_world_time = canonical_world_time_from_legacy(
                in_world_time=reverted.in_world_time,
                time_of_day=(
                    "" if audit.field_path == "in_world_time" else reverted.time_of_day
                ),
                day_of_week=reverted.day_of_week,
                world_day_index=reverted.world_day_index,
                source_message_id=source_message_id,
                confidence=world_time_confidence,
            )
            world_time_kwargs = {
                "world_time_day_index": canonical_world_time.day_index,
                "world_time_source_message_id": canonical_world_time.source_message_id,
                "world_time_confidence": canonical_world_time.confidence,
            }
            if audit.field_path != "world_day_index":
                world_time_kwargs.update(
                    {
                        "world_time_day_label": canonical_world_time.day_label,
                        "world_time_phase": canonical_world_time.phase,
                        "world_time_clock_minutes": (
                            canonical_world_time.clock_minutes
                            if canonical_world_time.clock_minutes is not None
                            else reverted.world_time_clock_minutes
                        ),
                        "world_time_period_label": (
                            canonical_world_time.period_label
                            or reverted.world_time_period_label
                        ),
                    }
                )
        self.repositories.upsert_scene_snapshot(
            save_id=save_id,
            current_location_id=reverted.current_location_id,
            situation=reverted.situation,
            objective=reverted.objective,
            in_world_time=reverted.in_world_time,
            time_of_day=reverted.time_of_day,
            day_of_week=reverted.day_of_week,
            world_day_index=reverted.world_day_index,
            weather=reverted.weather,
            mood=reverted.mood,
            nearby_objects=reverted.nearby_objects,
            hazards=reverted.hazards,
            present_character_ids=reverted.present_character_ids,
            source_message_id=source_message_id,
            locked_fields=reverted.locked_fields,
            snapshot_id=reverted.id,
            first_seen_message_id=reverted.first_seen_message_id,
            last_updated_message_id=source_message_id,
            **world_time_kwargs,
        )

    def _restore_scene_snapshot(
        self,
        *,
        save_id: str,
        snapshot: SceneSnapshotRecord,
    ) -> None:
        self.repositories.upsert_scene_snapshot(
            save_id=save_id,
            current_location_id=snapshot.current_location_id,
            situation=snapshot.situation,
            objective=snapshot.objective,
            in_world_time=snapshot.in_world_time,
            time_of_day=snapshot.time_of_day,
            day_of_week=snapshot.day_of_week,
            world_day_index=snapshot.world_day_index,
            world_time_day_index=snapshot.world_time_day_index,
            world_time_day_label=snapshot.world_time_day_label,
            world_time_phase=snapshot.world_time_phase,
            world_time_clock_minutes=snapshot.world_time_clock_minutes,
            world_time_period_label=snapshot.world_time_period_label,
            world_time_source_message_id=snapshot.world_time_source_message_id,
            world_time_confidence=snapshot.world_time_confidence,
            weather=snapshot.weather,
            mood=snapshot.mood,
            nearby_objects=snapshot.nearby_objects,
            hazards=snapshot.hazards,
            present_character_ids=snapshot.present_character_ids,
            source_message_id=snapshot.source_message_id,
            locked_fields=snapshot.locked_fields,
            snapshot_id=snapshot.id,
            first_seen_message_id=snapshot.first_seen_message_id,
            last_updated_message_id=snapshot.last_updated_message_id,
        )

    def _archive_new_summaries(
        self,
        *,
        save_id: str,
        active_summary_ids_before_resubmission: frozenset[str],
    ) -> None:
        for summary in self.repositories.list_summaries(save_id):
            if summary.id not in active_summary_ids_before_resubmission:
                self.repositories.archive_summary(summary.id)

    def _restore_restored_world_state(
        self,
        *,
        save_id: str,
        restored_message_ids: frozenset[str],
        excluded_message_ids: frozenset[str],
    ) -> None:
        changes = [
            change
            for change in self.repositories.list_state_changes(save_id)
            if change.source_message_id not in excluded_message_ids
        ]
        touched_keys = {
            change.state_key
            for change in changes
            if change.source_message_id in restored_message_ids
        }
        if not touched_keys:
            return

        for state_key in touched_keys:
            last_change = _last_change_for_key(changes, state_key)
            if last_change is None or last_change.after_json is None:
                self.repositories.archive_world_state(
                    save_id=save_id,
                    key=state_key,
                )
                continue
            self.repositories.upsert_world_state(
                save_id=save_id,
                key=state_key,
                value=_load_state_value(last_change.after_json),
                source_message_id=last_change.source_message_id,
            )


def _find_message(
    messages: list[MessageRecord],
    message_id: str,
) -> MessageRecord | None:
    for message in messages:
        if message.id == message_id:
            return message
    return None


def _previous_player_message(
    messages: list[MessageRecord],
    selected: MessageRecord,
) -> MessageRecord | None:
    previous: MessageRecord | None = None
    for message in messages:
        if message.id == selected.id:
            return previous
        if message.role == "player":
            previous = message
    return None


def _deletion_anchor_message(
    messages: list[MessageRecord],
    selected: MessageRecord,
) -> MessageRecord:
    if selected.role != "narrator":
        return selected
    previous_player = _previous_player_message(messages, selected)
    return previous_player or selected


def _first_change_for_key(
    changes: list[StateChangeRecord],
    state_key: str,
) -> StateChangeRecord | None:
    for change in changes:
        if change.state_key == state_key:
            return change
    return None


def _last_change_for_key(
    changes: list[StateChangeRecord],
    state_key: str,
) -> StateChangeRecord | None:
    for change in reversed(changes):
        if change.state_key == state_key:
            return change
    return None


def _load_state_value(value: str) -> dict[str, object]:
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError("State change value must be a JSON object")
    return cast(dict[str, object], loaded)


def _message_diff(previous: str, new: str) -> str:
    diff = list(
        unified_diff(
            previous.splitlines(),
            new.splitlines(),
            fromfile="previous",
            tofile="current",
            lineterm="",
        )
    )
    return "\n".join(diff) + ("\n" if diff else "")


def _summary_covers_message(
    *,
    summary: SummaryRecord,
    message_order: dict[str, int],
    message_id: str,
) -> bool:
    start = message_order.get(summary.covers_message_start_id)
    end = message_order.get(summary.covers_message_end_id)
    target = message_order.get(message_id)
    if start is None or end is None or target is None:
        return False
    lower, upper = sorted((start, end))
    return lower <= target <= upper


def _links_for_endpoint(
    links: list[EntityLinkRecord],
    *,
    entity_type: str,
    entity_id: str,
) -> tuple[EntityLinkRecord, ...]:
    return tuple(
        link
        for link in links
        if (
            (link.entity_type == entity_type and link.entity_id == entity_id)
            or (link.target_type == entity_type and link.target_id == entity_id)
        )
    )


def _dedupe_entity_links(
    links: list[EntityLinkRecord],
) -> tuple[EntityLinkRecord, ...]:
    deduped: list[EntityLinkRecord] = []
    seen: set[str] = set()
    for link in links:
        if link.id in seen:
            continue
        seen.add(link.id)
        deduped.append(link)
    return tuple(deduped)


def _active_world_state_for_key(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    key: str,
) -> WorldStateRecord | None:
    for state in repositories.list_world_state(save_id):
        if state.key == key:
            return state
    return None


def _record_first_seen_message_id(record: object) -> str | None:
    value = getattr(record, "first_seen_message_id", None)
    if isinstance(value, str):
        return value
    source = getattr(record, "source_message_id", None)
    return source if isinstance(source, str) else None


def _latest_remaining_context_update_source(
    *,
    record: object,
    audit_rows: list[ContextUpdateAuditRecord],
    deleted_message_ids: frozenset[str],
) -> str | None:
    record_id = cast(Any, record).id
    first_seen_message_id = _record_first_seen_message_id(record)
    latest_source_id = first_seen_message_id
    for audit in audit_rows:
        if audit.entity_id != record_id:
            continue
        if set(audit.source_message_ids) & deleted_message_ids:
            continue
        if audit.operation not in {"created", "updated"}:
            continue
        latest_source_id = (
            audit.source_message_ids[-1]
            if audit.source_message_ids
            else latest_source_id
        )
    return latest_source_id


def _latest_remaining_context_update_confidence(
    *,
    record: object,
    audit_rows: list[ContextUpdateAuditRecord],
    deleted_message_ids: frozenset[str],
    field_paths: frozenset[str],
) -> float | None:
    record_id = cast(Any, record).id
    latest_confidence: float | None = None
    for audit in audit_rows:
        if audit.entity_id != record_id:
            continue
        if audit.field_path not in field_paths:
            continue
        if set(audit.source_message_ids) & deleted_message_ids:
            continue
        if audit.operation not in {"created", "updated"}:
            continue
        latest_confidence = audit.confidence
    return latest_confidence


def _replace_scene_snapshot_field(
    snapshot: SceneSnapshotRecord,
    field_path: str,
    value: object,
) -> SceneSnapshotRecord:
    if field_path == "current_location_id":
        return replace(snapshot, current_location_id=cast(str | None, value))
    if field_path == "situation":
        return replace(snapshot, situation=cast(str, value))
    if field_path == "objective":
        return replace(snapshot, objective=cast(str, value))
    if field_path == "in_world_time":
        return replace(snapshot, in_world_time=cast(str, value))
    if field_path == "time_of_day":
        return replace(snapshot, time_of_day=cast(str, value))
    if field_path == "day_of_week":
        return replace(snapshot, day_of_week=cast(str, value))
    if field_path == "world_day_index":
        world_day_index = (
            value if isinstance(value, int) and not isinstance(value, bool) else None
        )
        return replace(snapshot, world_day_index=world_day_index)
    if field_path == "weather":
        return replace(snapshot, weather=cast(str, value))
    if field_path == "mood":
        return replace(snapshot, mood=cast(str, value))
    if field_path == "nearby_objects":
        return replace(snapshot, nearby_objects=cast(list[str], value))
    if field_path == "hazards":
        return replace(snapshot, hazards=cast(list[str], value))
    if field_path == "present_character_ids":
        return replace(snapshot, present_character_ids=cast(list[str], value))
    raise ValueError(f"Unsupported scene snapshot field: {field_path}")
