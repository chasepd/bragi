"""Structured world-data updates derived from character text exchanges."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, cast

from bragi.app_logging import exception_log_fields, log_error_event
from bragi.persistence.models import (
    ActiveThreadRecord,
    CharacterContactStateRecord,
    CharacterRecord,
    CharacterTextMessageRecord,
    DatingRouteStateRecord,
    JobRecord,
    MemoryRecord,
)
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatMessage,
    ProviderClient,
    StructuredOutputProvider,
    StructuredOutputRequest,
)
from bragi.redaction import redact_text
from bragi.retry_policy import DEFERRED_WORK_MAX_ATTEMPTS, configured_retry_count
from bragi.services.character_text_context import (
    canonical_character_text_context_messages,
    character_text_audience_character_ids,
    uploaded_photo_descriptions_by_message_id,
)
from bragi.services.job_lifecycle import JobLifecycleService
from bragi.services.mention_matching import character_name_is_mentioned
from bragi.services.model_preferences import roleplay_model_preference
from bragi.services.openrouter_routing_settings import (
    request_with_openrouter_routing,
)
from bragi.services.provider_fallbacks import structured_output_with_fallback

CHARACTER_TEXT_WORLD_UPDATE_JOB_TYPE = "character_text_world_update"
CHARACTER_TEXT_WORLD_UPDATE_RETRY_JOB_TYPE = "character_text_world_update_retry"
CHARACTER_TEXT_SOURCE_PREFIX = "character_text_message:"
_MAX_RETRY_ATTEMPTS = DEFERRED_WORK_MAX_ATTEMPTS
_RETRY_DRAIN_LIMIT = 3
_MAX_PRIOR_THREAD_CONTEXT_MESSAGES = 12


@dataclass(frozen=True)
class CharacterTextWorldUpdateResult:
    status: str
    memory_count: int = 0
    active_thread_count: int = 0
    character_count: int = 0
    dating_route_count: int = 0
    contact_permission_count: int = 0
    knowledge_edge_count: int = 0
    audit_count: int = 0
    retry_job_id: str | None = None
    error: str | None = None

    def to_json(self) -> dict[str, object]:
        result: dict[str, object] = {
            "status": self.status,
            "memory_count": self.memory_count,
            "active_thread_count": self.active_thread_count,
            "character_count": self.character_count,
            "dating_route_count": self.dating_route_count,
            "contact_permission_count": self.contact_permission_count,
            "knowledge_edge_count": self.knowledge_edge_count,
            "audit_count": self.audit_count,
        }
        if self.retry_job_id is not None:
            result["retry_job_id"] = self.retry_job_id
        if self.error:
            result["error"] = self.error
        return result


@dataclass
class _ApplyCounts:
    memory_count: int = 0
    active_thread_count: int = 0
    character_count: int = 0
    dating_route_count: int = 0
    contact_permission_count: int = 0
    knowledge_edge_count: int = 0
    audit_count: int = 0


@dataclass(frozen=True)
class _TextUpdateScope:
    text_character_ids: frozenset[str]
    player_character_ids: frozenset[str]
    audience_character_ids: frozenset[str]
    character_update_ids: frozenset[str]
    route_npc_ids: frozenset[str]


class CharacterTextWorldUpdateService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        providers: Mapping[str, object],
    ) -> None:
        self.repositories = repositories
        self.providers = providers
        self.jobs = JobLifecycleService(repositories=repositories)

    async def update_after_text_exchange(
        self,
        *,
        save_id: str,
        player_message: CharacterTextMessageRecord,
        reply: CharacterTextMessageRecord,
        queue_retry_on_failure: bool = True,
    ) -> CharacterTextWorldUpdateResult:
        return await self.update_after_text_messages(
            save_id=save_id,
            text_messages=(player_message, reply),
            queue_retry_on_failure=queue_retry_on_failure,
        )

    async def update_after_text_messages(
        self,
        *,
        save_id: str,
        text_messages: tuple[CharacterTextMessageRecord, ...],
        queue_retry_on_failure: bool = True,
    ) -> CharacterTextWorldUpdateResult:
        if not text_messages:
            return CharacterTextWorldUpdateResult(status="skipped")
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="context_update",
        )
        if preference is None:
            return CharacterTextWorldUpdateResult(status="skipped")
        provider = self.providers.get(preference.provider)
        if not isinstance(provider, StructuredOutputProvider):
            return CharacterTextWorldUpdateResult(status="skipped")
        try:
            request = self._structured_request(
                save_id=save_id,
                provider=preference.provider,
                model_id=preference.model_id,
                text_messages=text_messages,
            )
            response = await structured_output_with_fallback(
                repositories=self.repositories,
                providers=cast(dict[str, ProviderClient], self.providers),
                request=request,
                task="context_update",
                save_id=save_id,
            )
            counts = self.apply_structured_update(
                save_id=save_id,
                data=response.data,
                text_messages=text_messages,
            )
        except Exception as exc:  # noqa: BLE001 - best-effort maintenance boundary
            if queue_retry_on_failure:
                retry_job = self._queue_retry(
                    save_id=save_id,
                    text_messages=text_messages,
                    provider=preference.provider,
                    model_id=preference.model_id,
                    error=exc,
                )
                if retry_job is None:
                    return CharacterTextWorldUpdateResult(
                        status="failed",
                        error=redact_text(str(exc)) or exc.__class__.__name__,
                    )
                return CharacterTextWorldUpdateResult(
                    status="retry_queued",
                    retry_job_id=retry_job.id,
                    error=redact_text(str(exc)) or exc.__class__.__name__,
                )
            raise
        return CharacterTextWorldUpdateResult(
            status="applied",
            memory_count=counts.memory_count,
            active_thread_count=counts.active_thread_count,
            character_count=counts.character_count,
            dating_route_count=counts.dating_route_count,
            contact_permission_count=counts.contact_permission_count,
            knowledge_edge_count=counts.knowledge_edge_count,
            audit_count=counts.audit_count,
        )

    async def run_retries(self, *, save_id: str | None = None) -> int:
        retry_jobs = [
            job
            for job in self.repositories.list_jobs_by_status(("queued",))
            if job.type == CHARACTER_TEXT_WORLD_UPDATE_RETRY_JOB_TYPE
            and (save_id is None or job.save_id == save_id)
        ]
        completed = 0
        for retry_job in retry_jobs[:_RETRY_DRAIN_LIMIT]:
            running = self.jobs.start(retry_job.id, collect_provider_diagnostics=True)
            retry_save_id = running.save_id
            if retry_save_id is None:
                self.jobs.fail(
                    running.id,
                    error="Text world update retry missing save_id",
                )
                continue
            messages = _retry_text_messages(
                self.repositories,
                save_id=retry_save_id,
                payload=running.payload,
            )
            if messages is None:
                self.jobs.fail(
                    running.id,
                    error="Text world update retry missing text messages",
                )
                continue
            try:
                result = await self.update_after_text_messages(
                    save_id=retry_save_id,
                    text_messages=messages,
                    queue_retry_on_failure=False,
                )
            except Exception as exc:  # noqa: BLE001 - retry jobs record provider errors
                retry_attempt = _retry_attempt(running.payload)
                max_attempts = _retry_max_attempts(running.payload)
                retry_result: dict[str, object] = {
                    "text_message_ids": [message.id for message in messages],
                    "retry_attempt": retry_attempt,
                    "max_retry_attempts": max_attempts,
                }
                if retry_attempt < max_attempts:
                    next_job = self.jobs.create_queued(
                        save_id=retry_save_id,
                        type=CHARACTER_TEXT_WORLD_UPDATE_RETRY_JOB_TYPE,
                        payload={
                            **running.payload,
                            "retry_attempt": retry_attempt + 1,
                            "max_retry_attempts": max_attempts,
                        },
                    )
                    retry_result["next_retry_job_id"] = next_job.id
                self.jobs.fail(
                    running.id,
                    error=redact_text(str(exc)) or exc.__class__.__name__,
                    result=retry_result,
                )
                log_error_event(
                    "character_text_world_update.retry_failed",
                    save_id=retry_save_id,
                    retry_job_id=running.id,
                    **exception_log_fields(exc),
                )
                continue
            self.jobs.succeed(
                running.id,
                result={
                    **result.to_json(),
                    "text_message_ids": [message.id for message in messages],
                },
            )
            completed += 1
        return completed

    def apply_structured_update(
        self,
        *,
        save_id: str,
        data: dict[str, Any],
        text_messages: tuple[CharacterTextMessageRecord, ...],
    ) -> _ApplyCounts:
        counts = _ApplyCounts()
        messages_by_id = _source_messages_by_key(text_messages)
        characters = tuple(self.repositories.list_characters(save_id))
        characters_by_id = {
            character.id: character
            for character in characters
        }
        scope = _text_update_scope(
            repositories=self.repositories,
            save_id=save_id,
            text_messages=text_messages,
            characters=characters,
        )
        routes_by_npc_id = {
            route.npc_character_id: route
            for route in self.repositories.list_dating_route_states(save_id)
        }
        self.repositories.begin_transaction()
        try:
            for item in _objects(data.get("memories")):
                memory = self._apply_memory(
                    save_id=save_id,
                    item=item,
                    messages_by_id=messages_by_id,
                    characters_by_id=characters_by_id,
                    scope=scope,
                    counts=counts,
                )
                if memory is not None:
                    counts.memory_count += 1
            for item in _objects(data.get("active_threads")):
                thread = self._apply_active_thread(
                    save_id=save_id,
                    item=item,
                    messages_by_id=messages_by_id,
                    characters_by_id=characters_by_id,
                    scope=scope,
                    counts=counts,
                )
                if thread is not None:
                    counts.active_thread_count += 1
            for item in _objects(data.get("character_updates")):
                character = self._apply_character_update(
                    save_id=save_id,
                    item=item,
                    messages_by_id=messages_by_id,
                    characters_by_id=characters_by_id,
                    scope=scope,
                    counts=counts,
                )
                if character is not None:
                    counts.character_count += 1
                    characters_by_id[character.id] = character
            for item in _objects(data.get("dating_route_updates")):
                route = self._apply_dating_route_update(
                    save_id=save_id,
                    item=item,
                    messages_by_id=messages_by_id,
                    routes_by_npc_id=routes_by_npc_id,
                    scope=scope,
                    counts=counts,
                )
                if route is not None:
                    counts.dating_route_count += 1
                    routes_by_npc_id[route.npc_character_id] = route
            for item in _objects(data.get("contact_permissions")):
                contact_state = self._apply_contact_permission_update(
                    save_id=save_id,
                    item=item,
                    messages_by_id=messages_by_id,
                    scope=scope,
                    counts=counts,
                )
                if contact_state is not None:
                    counts.contact_permission_count += 1
            self.repositories.commit_transaction()
        except Exception:
            self.repositories.rollback_transaction()
            raise
        return counts

    def _apply_memory(
        self,
        *,
        save_id: str,
        item: dict[str, object],
        messages_by_id: dict[str, CharacterTextMessageRecord],
        characters_by_id: dict[str, CharacterRecord],
        scope: _TextUpdateScope,
        counts: _ApplyCounts,
    ) -> MemoryRecord | None:
        body = _string(item.get("body"))
        if not body:
            return None
        source_message = _source_message(item, messages_by_id)
        if source_message is None:
            return None
        source_ref = character_text_source_ref(source_message.id)
        importance = _float(item.get("importance"), default=1.0)
        tags = _string_list(item.get("tags"))
        memory = self.repositories.add_memory(
            save_id=save_id,
            body=body,
            tags=tags,
            importance=importance,
            source_message_ids=[source_ref],
        )
        self._record_applied(
            save_id=save_id,
            source_message=source_message,
            target_type="memory",
            target_id=memory.id,
            operation="created",
            field_path="*",
            before=None,
            after={"body": memory.body, "tags": memory.tags},
            reason=_string(item.get("reason")),
            confidence=importance,
            counts=counts,
        )
        character_id = _string(item.get("character_id"))
        if character_id and character_id not in scope.character_update_ids:
            self._record_rejected(
                save_id=save_id,
                source_message=source_message,
                target_type="character_knowledge_edge",
                target_id=character_id,
                field_path="character_id",
                reason="Rejected memory knowledge edge outside text participants",
                confidence=importance,
                item=item,
                details={
                    "attempted_character_id": character_id,
                    "allowed_character_ids": _sorted_strings(
                        scope.character_update_ids
                    ),
                },
                counts=counts,
            )
            return memory
        target_character_ids = (
            (character_id,)
            if character_id
            else tuple(sorted(scope.audience_character_ids))
        )
        for target_character_id in target_character_ids:
            if target_character_id not in characters_by_id:
                continue
            edge = self.repositories.add_character_knowledge_edge(
                save_id=save_id,
                character_id=target_character_id,
                target_type="memory",
                target_id=memory.id,
                knowledge_state=_knowledge_state(item.get("knowledge_state")),
                acquisition_method=_acquisition_method(item.get("acquisition_method")),
                confidence=importance,
                source_message_ids=[source_ref],
                evidence_quote=_string(item.get("evidence_quote")),
            )
            counts.knowledge_edge_count += 1
            self._record_text_provenance(
                save_id=save_id,
                source_message=source_message,
                target_type="character_knowledge_edge",
                target_id=edge.id,
                operation="created",
                field_path="*",
            )
        return memory

    def _apply_active_thread(
        self,
        *,
        save_id: str,
        item: dict[str, object],
        messages_by_id: dict[str, CharacterTextMessageRecord],
        characters_by_id: dict[str, CharacterRecord],
        scope: _TextUpdateScope,
        counts: _ApplyCounts,
    ) -> ActiveThreadRecord | None:
        title = _string(item.get("title"))
        if not title:
            return None
        source_message = _source_message(item, messages_by_id)
        if source_message is None:
            return None
        existing = _find_thread(self.repositories.list_active_threads(save_id), title)
        description = _string(item.get("description"))
        status = _string(item.get("status")) or (
            existing.status if existing else "active"
        )
        visibility = (
            _string(item.get("visibility"))
            or (existing.visibility if existing else "private")
        )
        priority = int(
            _float(
                item.get("priority"),
                default=existing.priority if existing else 0,
            )
        )
        related_entities, rejected_related_entities = _scoped_related_entities(
            _string_list(item.get("related_entities")),
            characters_by_id=characters_by_id,
            scope=scope,
        )
        if rejected_related_entities:
            self._record_rejected(
                save_id=save_id,
                source_message=source_message,
                target_type="active_thread",
                target_id=existing.id if existing else None,
                field_path="related_entities",
                reason=(
                    "Rejected active thread related_entities outside text "
                    "participants"
                ),
                confidence=_float(item.get("confidence"), default=1.0),
                item=item,
                details={
                    "rejected_related_entities": rejected_related_entities,
                    "allowed_character_ids": _sorted_strings(
                        scope.character_update_ids
                    ),
                },
                counts=counts,
            )
        if existing is None:
            thread = self.repositories.add_active_thread(
                save_id=save_id,
                title=title,
                description=description,
                status=status,
                priority=priority,
                visibility=visibility,
                related_entities=related_entities,
            )
            before = None
            operation = "created"
        else:
            before = _thread_audit_value(existing)
            thread = self.repositories.update_active_thread(
                replace(
                    existing,
                    description=description or existing.description,
                    status=status,
                    priority=priority,
                    visibility=visibility,
                    related_entities=_merge_strings(
                        existing.related_entities,
                        related_entities,
                    ),
                )
            )
            operation = "updated"
        self._record_applied(
            save_id=save_id,
            source_message=source_message,
            target_type="active_thread",
            target_id=thread.id,
            operation=operation,
            field_path="*",
            before=before,
            after=_thread_audit_value(thread),
            reason=_string(item.get("reason")),
            confidence=_float(item.get("confidence"), default=1.0),
            counts=counts,
        )
        return thread

    def _apply_character_update(
        self,
        *,
        save_id: str,
        item: dict[str, object],
        messages_by_id: dict[str, CharacterTextMessageRecord],
        characters_by_id: dict[str, CharacterRecord],
        scope: _TextUpdateScope,
        counts: _ApplyCounts,
    ) -> CharacterRecord | None:
        character_id = _string(item.get("character_id"))
        source_message = _source_message(item, messages_by_id)
        if source_message is None:
            return None
        if character_id not in scope.character_update_ids:
            self._record_rejected(
                save_id=save_id,
                source_message=source_message,
                target_type="character",
                target_id=character_id or None,
                field_path="*",
                reason="Rejected character update outside text participants",
                confidence=_float(item.get("confidence"), default=1.0),
                item=item,
                details={
                    "attempted_character_id": character_id,
                    "allowed_character_ids": _sorted_strings(
                        scope.character_update_ids
                    ),
                },
                counts=counts,
            )
            return None
        character = characters_by_id.get(character_id)
        if character is None:
            return None
        before = _character_audit_value(character)
        relationships = dict(character.relationships)
        item_relationships = item.get("relationships")
        if isinstance(item_relationships, dict):
            relationships.update(
                {
                    str(key): value
                    for key, value in item_relationships.items()
                    if str(key).strip()
                }
            )
        elif isinstance(item_relationships, list):
            relationships.update(_relationship_updates(item_relationships))
        updated = replace(
            character,
            known_state=_string(item.get("known_state")) or character.known_state,
            relationships=relationships,
            goals=_string(item.get("goals")) or character.goals,
            motivations=_string(item.get("motivations")) or character.motivations,
            current_intent=(
                _string(item.get("current_intent")) or character.current_intent
            ),
            boundaries=_string(item.get("boundaries")) or character.boundaries,
            attitude_toward_player=(
                _string(item.get("attitude_toward_player"))
                or character.attitude_toward_player
            ),
            status=_string(item.get("status")) or character.status,
        )
        if updated == character:
            return None
        saved = self.repositories.update_character(updated)
        self._record_applied(
            save_id=save_id,
            source_message=source_message,
            target_type="character",
            target_id=saved.id,
            operation="updated",
            field_path="*",
            before=before,
            after=_character_audit_value(saved),
            reason=_string(item.get("reason")),
            confidence=_float(item.get("confidence"), default=1.0),
            counts=counts,
        )
        return saved

    def _apply_dating_route_update(
        self,
        *,
        save_id: str,
        item: dict[str, object],
        messages_by_id: dict[str, CharacterTextMessageRecord],
        routes_by_npc_id: dict[str, DatingRouteStateRecord],
        scope: _TextUpdateScope,
        counts: _ApplyCounts,
    ) -> DatingRouteStateRecord | None:
        npc_character_id = _string(item.get("npc_character_id"))
        if not npc_character_id:
            return None
        source_message = _source_message(item, messages_by_id)
        if source_message is None:
            return None
        route = routes_by_npc_id.get(npc_character_id)
        if npc_character_id not in scope.route_npc_ids:
            self._record_rejected(
                save_id=save_id,
                source_message=source_message,
                target_type="dating_route_state",
                target_id=route.id if route else npc_character_id,
                field_path="*",
                reason="Rejected dating route update outside text thread character",
                confidence=_float(item.get("confidence"), default=1.0),
                item=item,
                details={
                    "attempted_npc_character_id": npc_character_id,
                    "allowed_npc_character_ids": _sorted_strings(
                        scope.route_npc_ids
                    ),
                },
                counts=counts,
            )
            return None
        if route is None:
            return None
        before = _route_audit_value(route)
        saved = self.repositories.upsert_dating_route_state(
            save_id=save_id,
            player_character_id=route.player_character_id,
            npc_character_id=route.npc_character_id,
            stage=_string(item.get("stage")) or route.stage,
            completed_interactions=route.completed_interactions,
            dates_completed=route.dates_completed,
            interest_level=_string(item.get("interest_level")) or route.interest_level,
            trust_level=_string(item.get("trust_level")) or route.trust_level,
            comfort_with_intimacy=(
                _string(item.get("comfort_with_intimacy"))
                or route.comfort_with_intimacy
            ),
            pacing_preference=(
                _string(item.get("pacing_preference")) or route.pacing_preference
            ),
            known_boundaries=(
                _merge_strings(
                    route.known_boundaries,
                    _string_list(item.get("known_boundaries")),
                )
            ),
            unresolved_questions=(
                _merge_strings(
                    route.unresolved_questions,
                    _string_list(item.get("unresolved_questions")),
                )
            ),
            next_reasonable_step=(
                _string(item.get("next_reasonable_step")) or route.next_reasonable_step
            ),
            source_message_id=route.source_message_id,
        )
        if saved == route:
            return None
        self._record_applied(
            save_id=save_id,
            source_message=source_message,
            target_type="dating_route_state",
            target_id=saved.id,
            operation="updated",
            field_path="*",
            before=before,
            after=_route_audit_value(saved),
            reason=_string(item.get("reason")),
            confidence=_float(item.get("confidence"), default=1.0),
            counts=counts,
        )
        return saved

    def _apply_contact_permission_update(
        self,
        *,
        save_id: str,
        item: dict[str, object],
        messages_by_id: dict[str, CharacterTextMessageRecord],
        scope: _TextUpdateScope,
        counts: _ApplyCounts,
    ) -> CharacterContactStateRecord | None:
        source_message = _source_message(item, messages_by_id)
        if source_message is None:
            return None
        character_id = _string(item.get("character_id"))
        if character_id not in scope.text_character_ids:
            self._record_rejected(
                save_id=save_id,
                source_message=source_message,
                target_type="character_contact_state",
                target_id=character_id or None,
                field_path="character_id",
                reason="Rejected contact permission outside text participants",
                confidence=_float(item.get("confidence"), default=1.0),
                item=item,
                details={
                    "attempted_character_id": character_id,
                    "allowed_character_ids": _sorted_strings(scope.text_character_ids),
                },
                counts=counts,
            )
            return None
        player_character_id = next(iter(sorted(scope.player_character_ids)), None)
        if player_character_id is None:
            return None
        player_has_character_number = bool(
            item.get("player_has_character_number")
        )
        character_has_player_number = bool(
            item.get("character_has_player_number")
        )
        if not (player_has_character_number or character_has_player_number):
            return None
        before = self.repositories.get_character_contact_state(
            save_id=save_id,
            player_character_id=player_character_id,
            character_id=character_id,
        )
        saved = self.repositories.upsert_character_contact_state(
            save_id=save_id,
            player_character_id=player_character_id,
            character_id=character_id,
            player_has_character_number=player_has_character_number,
            character_has_player_number=character_has_player_number,
            source_text_message_id=source_message.id,
        )
        before_value = _contact_state_audit_value(before)
        after_value = _contact_state_audit_value(saved)
        if before_value == after_value:
            return None
        self._record_applied(
            save_id=save_id,
            source_message=source_message,
            target_type="character_contact_state",
            target_id=saved.id,
            operation="created" if before is None else "updated",
            field_path="*",
            before=before_value,
            after=after_value,
            reason=_string(item.get("reason")),
            confidence=_float(item.get("confidence"), default=1.0),
            counts=counts,
        )
        return saved

    def _record_applied(
        self,
        *,
        save_id: str,
        source_message: CharacterTextMessageRecord,
        target_type: str,
        target_id: str,
        operation: str,
        field_path: str,
        before: object | None,
        after: object | None,
        reason: str,
        confidence: float,
        counts: _ApplyCounts,
    ) -> None:
        source_ref = character_text_source_ref(source_message.id)
        self.repositories.add_context_update_audit(
            save_id=save_id,
            operation=operation,
            entity_type=target_type,
            entity_id=target_id,
            field_path=field_path,
            before=before,
            after=after,
            reason=reason,
            confidence=confidence,
            source_message_ids=[source_ref],
        )
        counts.audit_count += 1
        self._record_text_provenance(
            save_id=save_id,
            source_message=source_message,
            target_type=target_type,
            target_id=target_id,
            operation=operation,
            field_path=field_path,
        )

    def _record_rejected(
        self,
        *,
        save_id: str,
        source_message: CharacterTextMessageRecord,
        target_type: str,
        target_id: str | None,
        field_path: str,
        reason: str,
        confidence: float,
        item: dict[str, object],
        details: dict[str, object],
        counts: _ApplyCounts,
    ) -> None:
        source_ref = character_text_source_ref(source_message.id)
        self.repositories.add_context_update_audit(
            save_id=save_id,
            operation="rejected",
            entity_type=target_type,
            entity_id=target_id,
            field_path=field_path,
            before=None,
            after={
                **details,
                "provider_item": _safe_audit_value(item),
            },
            reason=reason,
            confidence=confidence,
            source_message_ids=[source_ref],
        )
        counts.audit_count += 1

    def _record_text_provenance(
        self,
        *,
        save_id: str,
        source_message: CharacterTextMessageRecord,
        target_type: str,
        target_id: str,
        operation: str,
        field_path: str,
    ) -> None:
        self.repositories.add_character_text_provenance(
            save_id=save_id,
            thread_id=source_message.thread_id,
            text_message_id=source_message.id,
            target_type=target_type,
            target_id=target_id,
            operation=operation,
            field_path=field_path,
        )

    def _structured_request(
        self,
        *,
        save_id: str,
        provider: str,
        model_id: str,
        text_messages: tuple[CharacterTextMessageRecord, ...],
    ) -> StructuredOutputRequest:
        characters = tuple(self.repositories.list_characters(save_id))
        scope = _text_update_scope(
            repositories=self.repositories,
            save_id=save_id,
            text_messages=text_messages,
            characters=characters,
        )
        scoped_characters = tuple(
            character
            for character in characters
            if character.id in scope.character_update_ids
        )
        scoped_routes = tuple(
            route
            for route in self.repositories.list_dating_route_states(save_id)
            if route.npc_character_id in scope.route_npc_ids
        )
        scoped_active_threads = _active_threads_for_text_world_update(
            self.repositories,
            save_id=save_id,
            text_messages=text_messages,
            characters=characters,
            scope=scope,
        )
        return request_with_openrouter_routing(
            self.repositories,
            StructuredOutputRequest(
                provider=provider,
                model_id=model_id,
                schema_name="character_text_world_update",
                schema=_schema(text_messages, scope=scope),
                messages=_messages(
                    repositories=self.repositories,
                    save_id=save_id,
                    text_messages=text_messages,
                    scope=scope,
                    thread_memory=_thread_memory_for_text_messages(
                        self.repositories,
                        save_id=save_id,
                        text_messages=text_messages,
                    ),
                    prior_thread_messages=_prior_thread_context_messages(
                        self.repositories,
                        save_id=save_id,
                        text_messages=text_messages,
                    ),
                    characters=scoped_characters,
                    active_threads=scoped_active_threads,
                    routes=scoped_routes,
                ),
                temperature=0.0,
            ),
            task="context_update",
            save_id=save_id,
        )

    def _queue_retry(
        self,
        *,
        save_id: str,
        text_messages: tuple[CharacterTextMessageRecord, ...],
        provider: str,
        model_id: str,
        error: Exception,
    ) -> JobRecord | None:
        log_error_event(
            "character_text_world_update.failed",
            save_id=save_id,
            provider=provider,
            model=model_id,
            **exception_log_fields(error),
        )
        max_retry_count = configured_retry_count(self.repositories)
        if max_retry_count == 0:
            return None
        return self.jobs.create_queued(
            save_id=save_id,
            type=CHARACTER_TEXT_WORLD_UPDATE_RETRY_JOB_TYPE,
            payload={
                "text_message_ids": [message.id for message in text_messages],
                "provider": provider,
                "model": model_id,
                "retry_attempt": 1,
                "max_retry_attempts": max_retry_count,
                "reason": "character_text_world_update_failed",
            },
        )


def character_text_source_ref(text_message_id: str) -> str:
    return f"{CHARACTER_TEXT_SOURCE_PREFIX}{text_message_id}"


def parse_character_text_source_ref(source_ref: str) -> str | None:
    if not source_ref.startswith(CHARACTER_TEXT_SOURCE_PREFIX):
        return None
    text_id = source_ref[len(CHARACTER_TEXT_SOURCE_PREFIX) :].strip()
    return text_id or None


def _retry_text_messages(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    payload: dict[str, object],
) -> tuple[CharacterTextMessageRecord, ...] | None:
    ids = payload.get("text_message_ids")
    if not isinstance(ids, list) or not ids:
        return None
    wanted = [item for item in ids if isinstance(item, str) and item]
    if len(wanted) != len(ids):
        return None
    messages = repositories.list_character_text_messages(save_id=save_id)
    by_id = {message.id: message for message in messages}
    if any(message_id not in by_id for message_id in wanted):
        return None
    return tuple(by_id[message_id] for message_id in wanted)


def _retry_attempt(payload: dict[str, object]) -> int:
    value = payload.get("retry_attempt", 1)
    return value if isinstance(value, int) and value > 0 else 1


def _retry_max_attempts(payload: dict[str, object]) -> int:
    value = payload.get("max_retry_attempts", _MAX_RETRY_ATTEMPTS)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return _MAX_RETRY_ATTEMPTS
    return value


def _text_update_scope(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    text_messages: tuple[CharacterTextMessageRecord, ...],
    characters: tuple[CharacterRecord, ...],
) -> _TextUpdateScope:
    text_character_ids = _text_recipient_character_ids(
        repositories=repositories,
        save_id=save_id,
        text_messages=text_messages,
    )
    player_character_ids = frozenset(
        character.id for character in characters if character.is_player_character
    )
    return _TextUpdateScope(
        text_character_ids=text_character_ids,
        player_character_ids=player_character_ids,
        audience_character_ids=character_text_audience_character_ids(
            repositories=repositories,
            save_id=save_id,
            text_messages=text_messages,
        ),
        character_update_ids=text_character_ids | player_character_ids,
        route_npc_ids=text_character_ids,
    )


def _text_recipient_character_ids(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    text_messages: tuple[CharacterTextMessageRecord, ...],
) -> frozenset[str]:
    character_ids = {
        message.character_id for message in text_messages if message.character_id
    }
    for thread_id in {message.thread_id for message in text_messages}:
        thread = repositories.get_character_text_thread(
            save_id=save_id,
            thread_id=thread_id,
        )
        if thread is None or thread.kind != "group":
            continue
        character_ids.update(
            participant.character_id
            for participant in repositories.list_character_text_thread_participants(
                save_id=save_id,
                thread_id=thread.id,
            )
        )
    return frozenset(character_ids)


def _sorted_strings(values: frozenset[str]) -> list[str]:
    return sorted(value for value in values if value)


def _schema(
    text_messages: tuple[CharacterTextMessageRecord, ...],
    *,
    scope: _TextUpdateScope,
) -> dict[str, object]:
    source_ids = [
        *_text_message_aliases(text_messages),
        *(message.id for message in text_messages),
    ]
    source_field = {"type": "string", "enum": source_ids}
    character_id_field = {
        "type": "string",
        "enum": _sorted_strings(scope.character_update_ids),
    }
    contact_character_id_field = {
        "type": "string",
        "enum": _sorted_strings(scope.text_character_ids),
    }
    route_npc_id_field = {
        "type": "string",
        "enum": _sorted_strings(scope.route_npc_ids),
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "memories",
            "active_threads",
            "character_updates",
            "dating_route_updates",
            "contact_permissions",
        ],
        "properties": {
            "memories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "body",
                        "tags",
                        "importance",
                        "source_text_message_id",
                    ],
                    "properties": {
                        "body": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "importance": {"type": "number"},
                        "source_text_message_id": source_field,
                        "character_id": character_id_field,
                        "knowledge_state": {"type": "string"},
                        "acquisition_method": {"type": "string"},
                        "evidence_quote": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                },
            },
            "active_threads": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["title", "source_text_message_id"],
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "status": {"type": "string"},
                        "priority": {"type": "number"},
                        "visibility": {"type": "string"},
                        "related_entities": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "source_text_message_id": source_field,
                        "reason": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                },
            },
            "character_updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["character_id", "source_text_message_id"],
                    "properties": {
                        "character_id": character_id_field,
                        "known_state": {"type": "string"},
                        "relationships": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["name", "value"],
                                "properties": {
                                    "name": {"type": "string"},
                                    "value": {"type": "string"},
                                },
                            },
                        },
                        "goals": {"type": "string"},
                        "motivations": {"type": "string"},
                        "current_intent": {"type": "string"},
                        "boundaries": {"type": "string"},
                        "attitude_toward_player": {"type": "string"},
                        "status": {"type": "string"},
                        "source_text_message_id": source_field,
                        "reason": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                },
            },
            "dating_route_updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["npc_character_id", "source_text_message_id"],
                    "properties": {
                        "npc_character_id": route_npc_id_field,
                        "stage": {"type": "string"},
                        "interest_level": {"type": "string"},
                        "trust_level": {"type": "string"},
                        "comfort_with_intimacy": {"type": "string"},
                        "pacing_preference": {"type": "string"},
                        "known_boundaries": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "unresolved_questions": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "next_reasonable_step": {"type": "string"},
                        "source_text_message_id": source_field,
                        "reason": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                },
            },
            "contact_permissions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["character_id", "source_text_message_id"],
                    "properties": {
                        "character_id": contact_character_id_field,
                        "player_has_character_number": {"type": "boolean"},
                        "character_has_player_number": {"type": "boolean"},
                        "source_text_message_id": source_field,
                        "evidence_quote": {"type": "string"},
                        "reason": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                },
            },
        },
    }


def _messages(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    text_messages: tuple[CharacterTextMessageRecord, ...],
    scope: _TextUpdateScope,
    thread_memory: str,
    prior_thread_messages: tuple[CharacterTextMessageRecord, ...],
    characters: tuple[CharacterRecord, ...],
    active_threads: tuple[ActiveThreadRecord, ...],
    routes: tuple[DatingRouteStateRecord, ...],
) -> tuple[ChatMessage, ...]:
    aliases = _text_message_aliases(text_messages)
    uploaded_photo_descriptions = uploaded_photo_descriptions_by_message_id(
        repositories=repositories,
        save_id=save_id,
        messages=(*prior_thread_messages, *text_messages),
    )
    thread_memory_lines = (
        ["Thread memory:", thread_memory.strip()]
        if thread_memory.strip()
        else []
    )
    prior_thread_lines = (
        [
            "Prior phone thread context:",
            *(
                f"- {message.sender}: "
                + _text_message_body_for_world_context(
                    message,
                    uploaded_photo_descriptions=uploaded_photo_descriptions.get(
                        message.id,
                        (),
                    ),
                )
                for message in prior_thread_messages
            ),
        ]
        if prior_thread_messages
        else []
    )
    body = "\n".join(
        [
            f"Save id: {save_id}",
            *thread_memory_lines,
            *prior_thread_lines,
            "Text messages:",
            *(
                f"- {alias}: {message.sender}: "
                + _text_message_body_for_world_context(
                    message,
                    uploaded_photo_descriptions=uploaded_photo_descriptions.get(
                        message.id,
                        (),
                    ),
                )
                for alias, message in zip(aliases, text_messages, strict=True)
            ),
            (
                "Allowed character update and memory knowledge targets: "
                f"{', '.join(_sorted_strings(scope.character_update_ids)) or 'none'}"
            ),
            (
                "Allowed dating-route NPC targets: "
                f"{', '.join(_sorted_strings(scope.route_npc_ids)) or 'none'}"
            ),
            (
                "Allowed contact permission targets: "
                f"{', '.join(_sorted_strings(scope.text_character_ids)) or 'none'}"
            ),
            "Characters:",
            *(
                f"- {character.id}: {character.name}; role={character.role}; "
                f"relationships={character.relationships}"
                for character in characters
            ),
            "Active threads:",
            *(
                f"- {thread.id}: {thread.title}; {thread.status}"
                for thread in active_threads
            ),
            "Dating routes:",
            *(
                f"- {route.id}: npc={route.npc_character_id}; stage={route.stage}; "
                f"trust={route.trust_level}; interest={route.interest_level}"
                for route in routes
            ),
        ]
    )
    return (
        ChatMessage(
            role="system",
            body=(
                "Extract durable world updates from a side-channel in-world text "
                "conversation. Only return facts directly supported by the text. "
                "Do not include ephemeral banter. Only target the listed "
                "participant character ids, and only update dating routes for the "
                "listed text-thread NPC ids."
            ),
        ),
        ChatMessage(role="player", body=body),
    )


def _text_message_body_for_world_context(
    message: CharacterTextMessageRecord,
    *,
    uploaded_photo_descriptions: tuple[str, ...] = (),
) -> str:
    body = message.body.strip()
    photo_lines = tuple(
        f"[Attached photo visible to recipient: {description}]"
        for description in uploaded_photo_descriptions
        if description.strip()
    )
    if not photo_lines:
        return body
    return "\n".join((body, *photo_lines)).strip()


def _thread_memory_for_text_messages(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    text_messages: tuple[CharacterTextMessageRecord, ...],
) -> str:
    thread_id = _single_thread_id(text_messages)
    if thread_id is None:
        return ""
    thread = repositories.get_character_text_thread(
        save_id=save_id,
        thread_id=thread_id,
    )
    if thread is None:
        return ""
    return thread.memory_body.strip()


def _prior_thread_context_messages(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    text_messages: tuple[CharacterTextMessageRecord, ...],
) -> tuple[CharacterTextMessageRecord, ...]:
    thread_id = _single_thread_id(text_messages)
    if thread_id is None:
        return ()
    target_ids = {message.id for message in text_messages}
    thread_messages = canonical_character_text_context_messages(
        repositories=repositories,
        save_id=save_id,
        thread_id=thread_id,
    )
    first_target_index = next(
        (
            index
            for index, message in enumerate(thread_messages)
            if message.id in target_ids
        ),
        None,
    )
    if first_target_index is None:
        return ()
    prior = tuple(
        message
        for message in thread_messages[:first_target_index]
        if message.id not in target_ids
    )
    return prior[-_MAX_PRIOR_THREAD_CONTEXT_MESSAGES:]


def _active_threads_for_text_world_update(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    text_messages: tuple[CharacterTextMessageRecord, ...],
    characters: tuple[CharacterRecord, ...],
    scope: _TextUpdateScope,
) -> tuple[ActiveThreadRecord, ...]:
    active_threads = tuple(repositories.list_active_threads(save_id))
    if not active_threads:
        return ()
    text_message_ids = {message.id for message in text_messages}
    source_refs = {
        character_text_source_ref(message_id) for message_id in text_message_ids
    }
    provenance_thread_ids = {
        provenance.target_id
        for message_id in text_message_ids
        for provenance in repositories.list_character_text_provenance(
            save_id=save_id,
            text_message_id=message_id,
        )
        if provenance.target_type == "active_thread"
    }
    text_characters_by_id = {
        character.id: character
        for character in characters
        if character.id in scope.text_character_ids
    }
    return tuple(
        thread
        for thread in active_threads
        if _active_thread_matches_text_scope(
            thread,
            text_characters_by_id=text_characters_by_id,
            text_message_ids=text_message_ids,
            source_refs=source_refs,
            provenance_thread_ids=provenance_thread_ids,
        )
    )


def _active_thread_matches_text_scope(
    thread: ActiveThreadRecord,
    *,
    text_characters_by_id: dict[str, CharacterRecord],
    text_message_ids: set[str],
    source_refs: set[str],
    provenance_thread_ids: set[str],
) -> bool:
    if thread.id in provenance_thread_ids:
        return True
    if _active_thread_has_text_source(thread, text_message_ids, source_refs):
        return True
    if any(
        _matching_character_ids(entity, text_characters_by_id)
        for entity in thread.related_entities
    ):
        return True
    combined = f"{thread.title}\n{thread.description}"
    return any(
        character_name_is_mentioned(
            name=character.name,
            aliases=(character.contact_name, *character.aliases),
            text=combined,
        )
        for character in text_characters_by_id.values()
    )


def _active_thread_has_text_source(
    thread: ActiveThreadRecord,
    text_message_ids: set[str],
    source_refs: set[str],
) -> bool:
    source_values = (
        thread.source_message_id,
        thread.first_seen_message_id,
        thread.last_updated_message_id,
    )
    return any(
        value in text_message_ids or value in source_refs
        for value in source_values
        if value
    )


def _single_thread_id(
    text_messages: tuple[CharacterTextMessageRecord, ...],
) -> str | None:
    thread_ids = {message.thread_id for message in text_messages}
    if len(thread_ids) != 1:
        return None
    return next(iter(thread_ids), None)


def _source_message(
    item: dict[str, object],
    messages_by_id: dict[str, CharacterTextMessageRecord],
) -> CharacterTextMessageRecord | None:
    raw = _string(item.get("source_text_message_id"))
    return messages_by_id.get(raw)


def _source_messages_by_key(
    text_messages: tuple[CharacterTextMessageRecord, ...],
) -> dict[str, CharacterTextMessageRecord]:
    messages_by_key = {message.id: message for message in text_messages}
    if len(text_messages) == 1:
        messages_by_key["message"] = text_messages[0]
        return messages_by_key
    if len(text_messages) == 2:
        player_message = next(
            (message for message in text_messages if message.sender == "player"),
            text_messages[0],
        )
        reply_message = next(
            (message for message in text_messages if message.sender == "character"),
            text_messages[1],
        )
        messages_by_key["player"] = player_message
        messages_by_key["reply"] = reply_message
        return messages_by_key
    messages_by_key.update(
        {
            alias: message
            for alias, message in zip(
                _text_message_aliases(text_messages),
                text_messages,
                strict=True,
            )
        }
    )
    return messages_by_key


def _text_message_aliases(
    text_messages: tuple[CharacterTextMessageRecord, ...],
) -> tuple[str, ...]:
    if len(text_messages) == 1:
        return ("message",)
    if len(text_messages) == 2:
        return ("player", "reply")
    return tuple(f"message_{index + 1}" for index in range(len(text_messages)))


def _objects(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return list(
        dict.fromkeys(
            item.strip() for item in value if isinstance(item, str) and item.strip()
        )
    )


def _scoped_related_entities(
    related_entities: list[str],
    *,
    characters_by_id: dict[str, CharacterRecord],
    scope: _TextUpdateScope,
) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    rejected: list[str] = []
    for entity in related_entities:
        character_ids = _matching_character_ids(entity, characters_by_id)
        if character_ids and not character_ids <= scope.character_update_ids:
            rejected.append(entity)
            continue
        kept.append(entity)
    return kept, rejected


def _matching_character_ids(
    value: str,
    characters_by_id: dict[str, CharacterRecord],
) -> frozenset[str]:
    normalized = _normalize_character_reference(value)
    if not normalized:
        return frozenset()
    return frozenset(
        character.id
        for character in characters_by_id.values()
        if normalized in _character_reference_keys(character)
    )


def _character_reference_keys(character: CharacterRecord) -> frozenset[str]:
    values: list[str] = []
    for raw in (
        character.id,
        character.name,
        character.contact_name,
        *character.aliases,
    ):
        text = raw.strip()
        if not text:
            continue
        values.extend((text, f"character:{text}"))
    return frozenset(
        normalized
        for value in values
        if (normalized := _normalize_character_reference(value))
    )


def _normalize_character_reference(value: str) -> str:
    return " ".join(value.casefold().replace(":", " ").split())


def _safe_audit_value(value: object) -> object:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, bool | int | float) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _safe_audit_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_safe_audit_value(item) for item in value]
    return redact_text(str(value))


def _relationship_updates(value: list[object]) -> dict[str, str]:
    updates: dict[str, str] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        name = _string(item.get("name"))
        relation_value = _string(item.get("value"))
        if name and relation_value:
            updates[name] = relation_value
    return updates


def _float(value: object, *, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    return default


def _knowledge_state(value: object) -> str:
    text = _string(value)
    return text if text in {"knows", "may_know", "does_not_know"} else "knows"


def _acquisition_method(value: object) -> str:
    text = _string(value)
    allowed = {
        "witnessed",
        "overheard",
        "told",
        "inferred_from_visible_consequence",
        "background",
        "manual",
        "unknown",
    }
    return text if text in allowed else "told"


def _find_thread(
    threads: list[ActiveThreadRecord],
    title: str,
) -> ActiveThreadRecord | None:
    normalized = title.strip().casefold()
    return next(
        (
            thread
            for thread in threads
            if thread.title.casefold() == normalized
        ),
        None,
    )


def _merge_strings(existing: list[str], additions: list[str]) -> list[str]:
    return list(dict.fromkeys([*existing, *additions]))


def _thread_audit_value(thread: ActiveThreadRecord) -> dict[str, object]:
    return {
        "id": thread.id,
        "title": thread.title,
        "description": thread.description,
        "status": thread.status,
        "priority": thread.priority,
        "visibility": thread.visibility,
        "related_entities": list(thread.related_entities),
    }


def _character_audit_value(character: CharacterRecord) -> dict[str, object]:
    return {
        "id": character.id,
        "name": character.name,
        "known_state": character.known_state,
        "relationships": dict(character.relationships),
        "goals": character.goals,
        "motivations": character.motivations,
        "current_intent": character.current_intent,
        "boundaries": character.boundaries,
        "attitude_toward_player": character.attitude_toward_player,
        "status": character.status,
    }


def _route_audit_value(route: DatingRouteStateRecord) -> dict[str, object]:
    return {
        "id": route.id,
        "stage": route.stage,
        "completed_interactions": route.completed_interactions,
        "dates_completed": route.dates_completed,
        "interest_level": route.interest_level,
        "trust_level": route.trust_level,
        "comfort_with_intimacy": route.comfort_with_intimacy,
        "pacing_preference": route.pacing_preference,
        "known_boundaries": list(route.known_boundaries),
        "unresolved_questions": list(route.unresolved_questions),
        "next_reasonable_step": route.next_reasonable_step,
    }


def _contact_state_audit_value(
    state: CharacterContactStateRecord | None,
) -> dict[str, object] | None:
    if state is None:
        return None
    return {
        "id": state.id,
        "player_character_id": state.player_character_id,
        "character_id": state.character_id,
        "player_has_character_number": state.player_has_character_number,
        "character_has_player_number": state.character_has_player_number,
        "source_message_id": state.source_message_id,
        "source_text_message_id": state.source_text_message_id,
    }
