"""Revision helpers for side-channel character text messages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from difflib import unified_diff
from typing import TYPE_CHECKING, cast

from bragi.persistence.models import (
    CharacterTextMessageRecord,
    CharacterTextMessageRevisionRecord,
    ContextUpdateAuditRecord,
)
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import ProviderClient
from bragi.services.character_text_service import (
    CharacterTextAttachmentMediaRunner,
    CharacterTextMessage,
    CharacterTextService,
    CharacterTextThread,
    refresh_character_text_thread_memory,
)
from bragi.services.character_text_world_update_service import character_text_source_ref
from bragi.services.content_rating import effective_content_safety_policy
from bragi.services.content_safety_service import ContentSafetyService

if TYPE_CHECKING:
    from bragi.services.character_text_world_update_service import (
        CharacterTextWorldUpdateResult,
    )


_TEXT_SOURCE_PREFIX = "character_text_message:"
_RECONCILED_TEXT_TARGET_TYPES = frozenset(
    {
        "active_thread",
        "character",
        "dating_route_state",
        "character_contact_state",
    }
)


@dataclass(frozen=True)
class CharacterTextEditResult:
    message: CharacterTextMessageRecord
    revision: CharacterTextMessageRevisionRecord
    previous_body: str


@dataclass(frozen=True)
class CharacterTextResubmitResult:
    save_id: str
    thread: CharacterTextThread
    player_message: CharacterTextMessage
    reply: CharacterTextMessage
    revision: CharacterTextMessageRevisionRecord
    world_update: CharacterTextWorldUpdateResult | None = None


@dataclass(frozen=True)
class CharacterTextDeletionResult:
    save_id: str
    thread: CharacterTextThread
    deleted_messages: tuple[CharacterTextMessageRecord, ...]
    archived_memory_ids: frozenset[str]
    archived_knowledge_edge_ids: frozenset[str]
    expired_context_update_suggestion_ids: frozenset[str]
    archived_context_observation_ids: frozenset[str]


@dataclass(frozen=True)
class _TextCleanup:
    archived_memory_ids: frozenset[str]
    archived_knowledge_edge_ids: frozenset[str]
    expired_context_update_suggestion_ids: frozenset[str]
    archived_context_observation_ids: frozenset[str]


class CharacterTextRevisionService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        providers: dict[str, object],
        media_service: CharacterTextAttachmentMediaRunner | None = None,
    ) -> None:
        self.repositories = repositories
        self.providers = providers
        self.media_service = media_service

    def _raise_unless_enabled(self, save_id: str) -> None:
        CharacterTextService(
            repositories=self.repositories,
            providers=self.providers,
        ).raise_unless_enabled(save_id)

    def edit_text_without_resubmit(
        self,
        *,
        save_id: str,
        text_message_id: str,
        body: str,
    ) -> CharacterTextEditResult:
        self._raise_unless_enabled(save_id)
        return self._edit_text_body(
            save_id=save_id,
            text_message_id=text_message_id,
            body=body,
            allowed_senders=frozenset({"player"}),
            sender_error="Only player text messages can be edited without resubmit",
        )

    def correct_character_text(
        self,
        *,
        save_id: str,
        text_message_id: str,
        body: str,
    ) -> CharacterTextEditResult:
        self._raise_unless_enabled(save_id)
        return self._edit_text_body(
            save_id=save_id,
            text_message_id=text_message_id,
            body=body,
            allowed_senders=frozenset({"character"}),
            sender_error="Only character text messages can be corrected",
        )

    async def correct_character_text_with_safety(
        self,
        *,
        save_id: str,
        text_message_id: str,
        body: str,
        current_user_id: str | None,
    ) -> CharacterTextEditResult:
        self._raise_unless_enabled(save_id)
        policy = effective_content_safety_policy(
            self.repositories,
            user_id=current_user_id,
        )
        safety = await ContentSafetyService(
            repositories=self.repositories,
            providers=cast(dict[str, ProviderClient], self.providers),
        ).review_narration(
            body=body,
            content_rating=policy.rating,
            fade_to_black_enabled=policy.fade_to_black_enabled,
            save_id=save_id,
        )
        return self._edit_text_body(
            save_id=save_id,
            text_message_id=text_message_id,
            body=safety.body,
            content_rating=safety.reviewed_content_rating,
            allowed_senders=frozenset({"character"}),
            sender_error="Only character text messages can be corrected",
        )

    async def edit_text_and_resubmit(
        self,
        *,
        save_id: str,
        text_message_id: str,
        body: str,
        current_user_id: str | None = None,
    ) -> CharacterTextResubmitResult:
        self._raise_unless_enabled(save_id)
        replacement = _replacement_body(body)
        self.repositories.begin_transaction()
        try:
            selected = self._active_text_message(
                save_id=save_id,
                text_message_id=text_message_id,
            )
            if selected.sender != "player":
                raise ValueError("Only player text messages can be edited and resent")
            if selected.delivery_status in {"pending", "retrying"}:
                raise ValueError("Character text send is already pending")
            if selected.body.strip() == replacement:
                raise ValueError("Text message was not changed")
            archived_messages = self.repositories.archive_character_text_messages_after(
                save_id=save_id,
                thread_id=selected.thread_id,
                message_id=selected.id,
            )
            cleaned_text_message_ids = frozenset(
                {selected.id, *(message.id for message in archived_messages)}
            )
            cleanup = self._cleanup_text_sources(
                save_id=save_id,
                text_message_ids=cleaned_text_message_ids,
                delete_proactive_triggers=False,
            )
            updated = self.repositories.update_character_text_message_body(
                save_id=save_id,
                message_id=selected.id,
                body=replacement,
            )
            updated = self.repositories.update_character_text_delivery(
                save_id=save_id,
                message_id=updated.id,
                status="pending",
                error=None,
                job_id=None,
                attempt=0,
            )
            revision = self.repositories.add_character_text_message_revision(
                save_id=save_id,
                text_message_id=selected.id,
                previous_body=selected.body,
                new_body=replacement,
                diff_unified=_message_diff(selected.body, replacement),
                reconciliation_status="queued",
            )
            refresh_character_text_thread_memory(
                repositories=self.repositories,
                save_id=save_id,
                thread_id=selected.thread_id,
            )
            self.repositories.commit_transaction()
        except Exception:
            self.repositories.rollback_transaction()
            raise

        try:
            result = await CharacterTextService(
                repositories=self.repositories,
                providers=self.providers,
                media_service=self.media_service,
            ).complete_queued_text_send(
                save_id=save_id,
                player_message_id=updated.id,
                current_user_id=current_user_id,
            )
        except Exception:
            self._restore_failed_resubmit(
                save_id=save_id,
                selected=selected,
                archived_messages=archived_messages,
                cleaned_text_message_ids=cleaned_text_message_ids,
                cleanup=cleanup,
                revision_id=revision.id,
            )
            raise

        self.repositories.delete_character_text_proactive_triggers_for_messages(
            save_id=save_id,
            text_message_ids=cleaned_text_message_ids,
        )
        revision = self.repositories.mark_character_text_message_revision_reconciled(
            revision.id,
            status="succeeded",
        )
        return CharacterTextResubmitResult(
            save_id=save_id,
            thread=result.thread,
            player_message=result.player_message,
            reply=result.reply,
            revision=revision,
            world_update=result.world_update,
        )

    def delete_text_messages_from_here(
        self,
        *,
        save_id: str,
        text_message_id: str,
    ) -> CharacterTextDeletionResult:
        self._raise_unless_enabled(save_id)
        self.repositories.begin_transaction()
        try:
            selected = self._active_text_message(
                save_id=save_id,
                text_message_id=text_message_id,
            )
            if selected.delivery_status in {"pending", "retrying"}:
                raise ValueError("Character text send is already pending")
            deleted_messages = self.repositories.archive_character_text_messages_from(
                save_id=save_id,
                thread_id=selected.thread_id,
                message_id=selected.id,
            )
            cleanup = self._cleanup_text_sources(
                save_id=save_id,
                text_message_ids=frozenset(
                    message.id for message in deleted_messages
                ),
            )
            refresh_character_text_thread_memory(
                repositories=self.repositories,
                save_id=save_id,
                thread_id=selected.thread_id,
            )
            self.repositories.commit_transaction()
        except Exception:
            self.repositories.rollback_transaction()
            raise

        thread = CharacterTextService(
            repositories=self.repositories,
            providers=self.providers,
        ).get_thread_model(
            save_id=save_id,
            thread_id=selected.thread_id,
        )
        return CharacterTextDeletionResult(
            save_id=save_id,
            thread=thread,
            deleted_messages=deleted_messages,
            archived_memory_ids=cleanup.archived_memory_ids,
            archived_knowledge_edge_ids=cleanup.archived_knowledge_edge_ids,
            expired_context_update_suggestion_ids=(
                cleanup.expired_context_update_suggestion_ids
            ),
            archived_context_observation_ids=cleanup.archived_context_observation_ids,
        )

    def _edit_text_body(
        self,
        *,
        save_id: str,
        text_message_id: str,
        body: str,
        content_rating: str | None = None,
        allowed_senders: frozenset[str],
        sender_error: str,
    ) -> CharacterTextEditResult:
        replacement = _replacement_body(body)
        self.repositories.begin_transaction()
        try:
            selected = self._active_text_message(
                save_id=save_id,
                text_message_id=text_message_id,
            )
            if selected.sender not in allowed_senders:
                raise ValueError(sender_error)
            if selected.delivery_status in {"pending", "retrying"}:
                raise ValueError("Character text send is already pending")
            if selected.body.strip() == replacement:
                raise ValueError("Text message was not changed")
            updated = self.repositories.update_character_text_message_body(
                save_id=save_id,
                message_id=selected.id,
                body=replacement,
                content_rating=content_rating,
            )
            self._cleanup_text_sources(
                save_id=save_id,
                text_message_ids=frozenset({selected.id}),
            )
            revision = self.repositories.add_character_text_message_revision(
                save_id=save_id,
                text_message_id=selected.id,
                previous_body=selected.body,
                new_body=replacement,
                diff_unified=_message_diff(selected.body, replacement),
                reconciliation_status="succeeded",
                reconciliation_error=None,
            )
            refresh_character_text_thread_memory(
                repositories=self.repositories,
                save_id=save_id,
                thread_id=selected.thread_id,
            )
            self.repositories.commit_transaction()
        except Exception:
            self.repositories.rollback_transaction()
            raise
        return CharacterTextEditResult(
            message=updated,
            revision=revision,
            previous_body=selected.body,
        )

    def _active_text_message(
        self,
        *,
        save_id: str,
        text_message_id: str,
    ) -> CharacterTextMessageRecord:
        selected = self.repositories.get_character_text_message(
            save_id=save_id,
            message_id=text_message_id,
        )
        if selected is None:
            raise ValueError(f"Unknown character text message id: {text_message_id}")
        return selected

    def _cleanup_text_sources(
        self,
        *,
        save_id: str,
        text_message_ids: frozenset[str],
        delete_proactive_triggers: bool = True,
    ) -> _TextCleanup:
        source_refs = frozenset(
            character_text_source_ref(id_) for id_ in text_message_ids
        )
        if not source_refs:
            return _TextCleanup(
                archived_memory_ids=frozenset(),
                archived_knowledge_edge_ids=frozenset(),
                expired_context_update_suggestion_ids=frozenset(),
                archived_context_observation_ids=frozenset(),
            )
        archived_memory_ids: set[str] = set()
        for memory in self.repositories.list_memories(save_id):
            source_ids = set(memory.source_message_ids)
            if memory.source_message_id is not None:
                source_ids.add(memory.source_message_id)
            if source_ids & source_refs:
                self.repositories.archive_memory(memory.id)
                archived_memory_ids.add(memory.id)
        archived_knowledge_edge_ids: set[str] = set()
        for edge in self.repositories.list_character_knowledge_edges(save_id):
            source_ids = set(edge.source_message_ids)
            if edge.source_message_id is not None:
                source_ids.add(edge.source_message_id)
            if source_ids & source_refs:
                self.repositories.archive_character_knowledge_edge(edge.id)
                archived_knowledge_edge_ids.add(edge.id)
        expired_suggestion_ids = (
            self.repositories.expire_context_update_suggestions_for_messages(
                save_id=save_id,
                message_ids=source_refs,
            )
        )
        archived_observation_ids = (
            self.repositories.archive_context_observations_for_deleted_messages(
                save_id=save_id,
                message_ids=source_refs,
            )
        )
        self._reconcile_text_world_state(
            save_id=save_id,
            source_refs=source_refs,
        )
        if delete_proactive_triggers:
            self.repositories.delete_character_text_proactive_triggers_for_messages(
                save_id=save_id,
                text_message_ids=text_message_ids,
            )
        return _TextCleanup(
            archived_memory_ids=frozenset(archived_memory_ids),
            archived_knowledge_edge_ids=frozenset(archived_knowledge_edge_ids),
            expired_context_update_suggestion_ids=expired_suggestion_ids,
            archived_context_observation_ids=archived_observation_ids,
        )

    def _restore_failed_resubmit(
        self,
        *,
        save_id: str,
        selected: CharacterTextMessageRecord,
        archived_messages: tuple[CharacterTextMessageRecord, ...],
        cleaned_text_message_ids: frozenset[str],
        cleanup: _TextCleanup,
        revision_id: str,
    ) -> None:
        self.repositories.begin_transaction()
        try:
            replacement_messages = (
                self.repositories.archive_character_text_messages_after(
                    save_id=save_id,
                    thread_id=selected.thread_id,
                    message_id=selected.id,
                )
            )
            if replacement_messages:
                self._cleanup_text_sources(
                    save_id=save_id,
                    text_message_ids=frozenset(
                        message.id for message in replacement_messages
                    ),
                )
            self.repositories.update_character_text_message_body(
                save_id=save_id,
                message_id=selected.id,
                body=selected.body,
            )
            self.repositories.update_character_text_delivery(
                save_id=save_id,
                message_id=selected.id,
                status=selected.delivery_status,
                error=selected.delivery_error,
                job_id=selected.delivery_job_id,
                attempt=selected.delivery_attempt,
                delivered_at=selected.delivered_at,
            )
            self.repositories.restore_character_text_messages(
                frozenset(message.id for message in archived_messages)
            )
            self._restore_text_cleanup(cleanup)
            self._restore_text_world_state(
                save_id=save_id,
                source_refs=frozenset(
                    character_text_source_ref(id_) for id_ in cleaned_text_message_ids
                ),
            )
            refresh_character_text_thread_memory(
                repositories=self.repositories,
                save_id=save_id,
                thread_id=selected.thread_id,
            )
            self.repositories.mark_character_text_message_revision_reconciled(
                revision_id,
                status="failed",
                error="Text resubmit failed",
            )
            self.repositories.commit_transaction()
        except Exception:
            self.repositories.rollback_transaction()
            raise

    def _restore_text_cleanup(self, cleanup: _TextCleanup) -> None:
        self.repositories.restore_memories(cleanup.archived_memory_ids)
        self.repositories.restore_character_knowledge_edges(
            cleanup.archived_knowledge_edge_ids
        )
        self.repositories.restore_context_update_suggestions(
            cleanup.expired_context_update_suggestion_ids
        )
        self.repositories.restore_context_observations(
            cleanup.archived_context_observation_ids
        )

    def _reconcile_text_world_state(
        self,
        *,
        save_id: str,
        source_refs: frozenset[str],
    ) -> None:
        if not source_refs:
            return
        active_source_refs = {
            character_text_source_ref(message.id)
            for message in self.repositories.list_character_text_messages(
                save_id=save_id,
            )
        }
        audits_by_target: dict[
            tuple[str, str],
            list[ContextUpdateAuditRecord],
        ] = {}
        impacted_targets: set[tuple[str, str]] = set()
        for audit in self.repositories.list_context_update_audit(save_id):
            if (
                audit.entity_id is None
                or audit.entity_type not in _RECONCILED_TEXT_TARGET_TYPES
                or audit.operation not in {"created", "updated", "text_exchange"}
            ):
                continue
            text_refs = _audit_text_source_refs(audit)
            target = (audit.entity_type, audit.entity_id)
            audits_by_target.setdefault(target, []).append(audit)
            if text_refs & source_refs:
                impacted_targets.add(target)
        for target_type, target_id in sorted(impacted_targets):
            audits = audits_by_target[(target_type, target_id)]
            remaining_audits = [
                audit
                for audit in audits
                if _audit_survives_text_cleanup(
                    audit,
                    removed_source_refs=source_refs,
                    active_source_refs=active_source_refs,
                )
            ]
            restored_value = _replayed_target_value(
                audits=audits,
                remaining_audits=remaining_audits,
            )
            self._apply_reconciled_target_value(
                save_id=save_id,
                target_type=target_type,
                target_id=target_id,
                value=restored_value,
            )

    def _restore_text_world_state(
        self,
        *,
        save_id: str,
        source_refs: frozenset[str],
    ) -> None:
        if not source_refs:
            return
        active_source_refs = {
            character_text_source_ref(message.id)
            for message in self.repositories.list_character_text_messages(
                save_id=save_id,
            )
        }
        audits_by_target: dict[
            tuple[str, str],
            list[ContextUpdateAuditRecord],
        ] = {}
        impacted_targets: set[tuple[str, str]] = set()
        for audit in self.repositories.list_context_update_audit(save_id):
            if (
                audit.entity_id is None
                or audit.entity_type not in _RECONCILED_TEXT_TARGET_TYPES
                or audit.operation not in {"created", "updated", "text_exchange"}
            ):
                continue
            text_refs = _audit_text_source_refs(audit)
            target = (audit.entity_type, audit.entity_id)
            audits_by_target.setdefault(target, []).append(audit)
            if text_refs & source_refs:
                impacted_targets.add(target)
        for target_type, target_id in sorted(impacted_targets):
            audits = audits_by_target[(target_type, target_id)]
            remaining_audits = [
                audit
                for audit in audits
                if _audit_active_after_text_restore(
                    audit,
                    active_source_refs=active_source_refs,
                )
            ]
            restored_value = _replayed_target_value(
                audits=audits,
                remaining_audits=remaining_audits,
            )
            self._apply_reconciled_target_value(
                save_id=save_id,
                target_type=target_type,
                target_id=target_id,
                value=restored_value,
            )

    def _apply_reconciled_target_value(
        self,
        *,
        save_id: str,
        target_type: str,
        target_id: str,
        value: dict[str, object] | None,
    ) -> None:
        if target_type == "active_thread":
            self._apply_reconciled_active_thread(
                save_id=save_id,
                target_id=target_id,
                value=value,
            )
        elif target_type == "character":
            self._apply_reconciled_character(
                save_id=save_id,
                target_id=target_id,
                value=value,
            )
        elif target_type == "dating_route_state":
            self._apply_reconciled_route(
                save_id=save_id,
                target_id=target_id,
                value=value,
            )
        elif target_type == "character_contact_state":
            self._apply_reconciled_contact_state(
                save_id=save_id,
                target_id=target_id,
                value=value,
            )

    def _apply_reconciled_active_thread(
        self,
        *,
        save_id: str,
        target_id: str,
        value: dict[str, object] | None,
    ) -> None:
        owner_row = self.repositories.connection.execute(
            "SELECT save_id FROM active_threads WHERE id = ?",
            (target_id,),
        ).fetchone()
        if owner_row is None or str(owner_row["save_id"]) != save_id:
            return
        if value is None:
            self.repositories.archive_active_thread(target_id)
            return
        self.repositories.restore_active_threads(frozenset({target_id}))
        thread = self.repositories.get_active_thread(target_id)
        if thread is None:
            return
        self.repositories.update_active_thread(
            replace(
                thread,
                title=_text_value(value, "title", thread.title),
                description=_text_value(value, "description", thread.description),
                status=_text_value(value, "status", thread.status),
                priority=_int_value(value, "priority", thread.priority),
                visibility=_text_value(value, "visibility", thread.visibility),
                related_entities=_string_list_value(
                    value,
                    "related_entities",
                    thread.related_entities,
                ),
            )
        )

    def _apply_reconciled_character(
        self,
        *,
        save_id: str,
        target_id: str,
        value: dict[str, object] | None,
    ) -> None:
        if value is None:
            return
        character = self.repositories.get_character(target_id)
        if character is None or character.save_id != save_id:
            return
        self.repositories.update_character(
            replace(
                character,
                known_state=_text_value(value, "known_state", character.known_state),
                relationships=_object_dict_value(
                    value,
                    "relationships",
                    character.relationships,
                ),
                goals=_text_value(value, "goals", character.goals),
                motivations=_text_value(value, "motivations", character.motivations),
                current_intent=_text_value(
                    value,
                    "current_intent",
                    character.current_intent,
                ),
                boundaries=_text_value(value, "boundaries", character.boundaries),
                attitude_toward_player=_text_value(
                    value,
                    "attitude_toward_player",
                    character.attitude_toward_player,
                ),
                status=_text_value(value, "status", character.status),
            )
        )

    def _apply_reconciled_route(
        self,
        *,
        save_id: str,
        target_id: str,
        value: dict[str, object] | None,
    ) -> None:
        route = next(
            (
                candidate
                for candidate in self.repositories.list_dating_route_states(
                    save_id,
                    include_archived=True,
                )
                if candidate.id == target_id
            ),
            None,
        )
        if value is None:
            if route is not None:
                self.repositories.archive_dating_route_state(route.id)
            return
        if route is None:
            return
        self.repositories.upsert_dating_route_state(
            save_id=save_id,
            player_character_id=route.player_character_id,
            npc_character_id=route.npc_character_id,
            stage=_text_value(value, "stage", route.stage),
            first_met_message_id=route.first_met_message_id,
            first_met_world_day_index=route.first_met_world_day_index,
            last_interaction_message_id=route.last_interaction_message_id,
            last_interaction_world_day_index=route.last_interaction_world_day_index,
            completed_interactions=_int_value(
                value,
                "completed_interactions",
                route.completed_interactions,
            ),
            dates_completed=_int_value(value, "dates_completed", route.dates_completed),
            interest_level=_text_value(value, "interest_level", route.interest_level),
            trust_level=_text_value(value, "trust_level", route.trust_level),
            comfort_with_intimacy=_text_value(
                value,
                "comfort_with_intimacy",
                route.comfort_with_intimacy,
            ),
            pacing_preference=_text_value(
                value,
                "pacing_preference",
                route.pacing_preference,
            ),
            known_boundaries=_string_list_value(
                value,
                "known_boundaries",
                route.known_boundaries,
            ),
            unresolved_questions=_string_list_value(
                value,
                "unresolved_questions",
                route.unresolved_questions,
            ),
            next_reasonable_step=_text_value(
                value,
                "next_reasonable_step",
                route.next_reasonable_step,
            ),
            source_message_id=route.source_message_id,
        )

    def _apply_reconciled_contact_state(
        self,
        *,
        save_id: str,
        target_id: str,
        value: dict[str, object] | None,
    ) -> None:
        if value is None:
            self.repositories.archive_character_contact_state(target_id)
            return
        player_character_id = _text_value(value, "player_character_id", "")
        character_id = _text_value(value, "character_id", "")
        if not player_character_id or not character_id:
            return
        self.repositories.replace_character_contact_state(
            save_id=save_id,
            player_character_id=player_character_id,
            character_id=character_id,
            player_has_character_number=_bool_value(
                value,
                "player_has_character_number",
                False,
            ),
            character_has_player_number=_bool_value(
                value,
                "character_has_player_number",
                False,
            ),
            source_message_id=_optional_text_value(value, "source_message_id"),
            source_text_message_id=_optional_text_value(
                value,
                "source_text_message_id",
            ),
            state_id=target_id,
        )


def _audit_text_source_refs(audit: ContextUpdateAuditRecord) -> frozenset[str]:
    return frozenset(
        source_id
        for source_id in audit.source_message_ids
        if source_id.startswith(_TEXT_SOURCE_PREFIX)
    )


def _audit_survives_text_cleanup(
    audit: ContextUpdateAuditRecord,
    *,
    removed_source_refs: frozenset[str],
    active_source_refs: set[str],
) -> bool:
    text_refs = _audit_text_source_refs(audit)
    if not text_refs:
        return True
    return not (text_refs & removed_source_refs) and text_refs <= active_source_refs


def _audit_active_after_text_restore(
    audit: ContextUpdateAuditRecord,
    *,
    active_source_refs: set[str],
) -> bool:
    text_refs = _audit_text_source_refs(audit)
    if not text_refs:
        return True
    return text_refs <= active_source_refs


def _replayed_target_value(
    *,
    audits: list[ContextUpdateAuditRecord],
    remaining_audits: list[ContextUpdateAuditRecord],
) -> dict[str, object] | None:
    created_by_text = any(
        audit.operation == "created" and _audit_value(audit.before) is None
        for audit in audits
    )
    if created_by_text and not remaining_audits:
        return None

    value = None if created_by_text else _baseline_target_value(audits)
    if value is None and remaining_audits:
        value = _audit_value(remaining_audits[0].before)

    remaining_ids = {audit.id for audit in remaining_audits}
    for audit in audits:
        if audit.id not in remaining_ids:
            continue
        after = _audit_value(audit.after)
        if after is None:
            continue
        value = _merged_audit_value(value, after)
    return value


def _baseline_target_value(
    audits: list[ContextUpdateAuditRecord],
) -> dict[str, object] | None:
    value: dict[str, object] | None = None
    for audit in audits:
        before = _audit_value(audit.before)
        if before is None:
            continue
        if value is None:
            value = dict(before)
        else:
            value = _fill_missing_audit_value(value, before)
    return value


def _audit_value(value: object | None) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    return {str(key): item for key, item in value.items()}


def _merged_audit_value(
    base: dict[str, object] | None,
    update: Mapping[str, object],
) -> dict[str, object]:
    merged = dict(base or {})
    merged.update(update)
    return merged


def _fill_missing_audit_value(
    base: dict[str, object],
    fallback: Mapping[str, object],
) -> dict[str, object]:
    merged = dict(base)
    for key, value in fallback.items():
        merged.setdefault(key, value)
    return merged


def _text_value(
    value: Mapping[str, object],
    key: str,
    default: str,
) -> str:
    candidate = value.get(key)
    if isinstance(candidate, str):
        return candidate
    return default


def _optional_text_value(value: Mapping[str, object], key: str) -> str | None:
    candidate = value.get(key)
    if isinstance(candidate, str):
        return candidate
    return None


def _int_value(
    value: Mapping[str, object],
    key: str,
    default: int,
) -> int:
    candidate = value.get(key)
    if isinstance(candidate, bool):
        return int(candidate)
    if isinstance(candidate, int):
        return candidate
    return default


def _bool_value(
    value: Mapping[str, object],
    key: str,
    default: bool,
) -> bool:
    candidate = value.get(key)
    if isinstance(candidate, bool):
        return candidate
    return default


def _string_list_value(
    value: Mapping[str, object],
    key: str,
    default: list[str],
) -> list[str]:
    candidate = value.get(key)
    if isinstance(candidate, list):
        return [str(item) for item in candidate if str(item).strip()]
    return list(default)


def _object_dict_value(
    value: Mapping[str, object],
    key: str,
    default: dict[str, object],
) -> dict[str, object]:
    candidate = value.get(key)
    if isinstance(candidate, dict):
        return {str(item_key): item_value for item_key, item_value in candidate.items()}
    return dict(default)


def _replacement_body(body: str) -> str:
    replacement = body.strip()
    if not replacement:
        raise ValueError("Text message is empty")
    return replacement


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
