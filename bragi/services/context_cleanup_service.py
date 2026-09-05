"""Manual cleanup pass for active-save context records."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, cast

from bragi.app_logging import exception_log_fields, log_error_event, log_event
from bragi.persistence.models import (
    ActiveThreadRecord,
    CharacterRecord,
    EntityLinkRecord,
    LocationRecord,
    MemoryRecord,
    MessageRecord,
    ModelPreferenceRecord,
    SceneSnapshotRecord,
    SummaryRecord,
    WorldStateRecord,
)
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatMessage,
    ProviderClient,
    ProviderToolCall,
    StructuredOutputProvider,
    StructuredOutputRequest,
    StructuredOutputResponse,
    ToolCallMessage,
    ToolCallProvider,
    ToolCallRequest,
    ToolDefinition,
)
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.providers.system_prompt import LYRICS_INTERPRETATION_SECTION
from bragi.redaction import redact_text
from bragi.retry_policy import MODEL_OUTPUT_MAX_ATTEMPTS, configured_max_attempts
from bragi.services.active_thread_lifecycle import active_thread_is_prompt_visible
from bragi.services.character_locks import (
    character_field_is_locked,
    reconcile_character_presence_locks,
)
from bragi.services.job_lifecycle import JobLifecycleService
from bragi.services.openrouter_routing_settings import (
    request_with_openrouter_routing,
)
from bragi.services.provider_fallbacks import (
    provider_error_with_fallback_attempted,
    provider_error_with_fallback_skipped_reason,
    structured_output_with_fallback,
    tool_call_fallback_request,
    tool_call_fallback_skip_reason,
)
from bragi.services.request_budget import (
    budget_structured_output_request,
    budget_tool_call_request,
)
from bragi.services.scene_snapshot_locks import scene_snapshot_field_is_locked
from bragi.services.tool_call_helpers import (
    accepted_tool_result,
    append_tool_feedback_messages,
    invalid_tool_result,
    parse_tool_arguments_json,
    validate_tool_arguments_shape,
)
from bragi.world_time_model import (
    canonical_world_time_from_legacy,
    canonical_world_time_from_values,
    legacy_world_time_fields,
)

_TRANSCRIPT_CHUNK_SIZE = 20
_TARGET_BATCH_SIZE = 40
_ACTION_SCAN_NOTE_LIMIT = 60
_ACTION_SCAN_NOTE_CHAR_LIMIT = 360
_ACTION_MESSAGE_REFERENCE_LIMIT = 160
_MIN_ACTION_CONFIDENCE = 0.65
_MAX_CONTEXT_CLEANUP_TOOL_FEEDBACK_TURNS = MODEL_OUTPUT_MAX_ATTEMPTS - 1
CONTEXT_CLEANUP_TASK = "context_cleanup"
CONTEXT_CLEANUP_SCAN_TASK = "context_cleanup_scan"
CONTEXT_CLEANUP_ACTIONS_TASK = "context_cleanup_actions"
GUIDED_CONTEXT_CLEANUP_TASK = "guided_context_cleanup"
_ARCHIVE_LOCK_FIELDS = frozenset({"*", "__record__", "archive", "archived_at"})
_TARGET_TYPES = frozenset(
    {
        "world_state",
        "memory",
        "summary",
        "scene_snapshot",
        "location",
        "character",
        "active_thread",
        "entity_link",
    }
)
_TARGET_TYPE_ORDER = (
    "scene_snapshot",
    "world_state",
    "memory",
    "summary",
    "location",
    "character",
    "active_thread",
    "entity_link",
)
_ARCHIVE_TARGET_TYPES = frozenset(
    {"world_state", "memory", "summary", "location", "character", "active_thread"}
)
_UPDATE_FIELDS: dict[str, frozenset[str]] = {
    "world_state": frozenset({"value"}),
    "memory": frozenset({"body", "tags", "importance"}),
    "summary": frozenset({"body"}),
    "scene_snapshot": frozenset(
        {
            "current_location_id",
            "situation",
            "objective",
            "in_world_time",
            "time_of_day",
            "day_of_week",
            "weather",
            "mood",
            "nearby_objects",
            "hazards",
            "present_character_ids",
        }
    ),
    "location": frozenset(
        {
            "name",
            "aliases",
            "description",
            "visual_description",
            "parent_location_id",
            "connections",
            "status",
            "hazards",
        }
    ),
    "character": frozenset(
        {
            "name",
            "aliases",
            "role",
            "known_state",
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
        }
    ),
    "active_thread": frozenset(
        {
            "title",
            "description",
            "status",
            "priority",
            "visibility",
            "related_entities",
        }
    ),
}
_SCENE_WORLD_TIME_FIELDS = frozenset(
    {
        "in_world_time",
        "time_of_day",
        "day_of_week",
        "world_day_index",
    }
)


@dataclass(frozen=True)
class ContextCleanupResult:
    save_id: str
    scanned_messages: int
    scan_batches: int
    cleanup_target_count: int
    action_batches: int
    proposed_actions: int
    applied_actions: int
    rejected_actions: int
    archives: int
    updates: int
    deleted_links: int


@dataclass(frozen=True)
class GuidedCleanupResult:
    save_id: str
    instruction: str
    cleanup_target_count: int
    action_batches: int
    proposed_actions: int
    queued_suggestions: int
    rejected_actions: int
    suggestion_ids: tuple[str, ...]


@dataclass(frozen=True)
class CleanupAction:
    operation: str
    target_type: str
    target_id: str
    field_path: str
    value: object | None
    reason: str
    confidence: float
    evidence_message_ids: tuple[str, ...]


@dataclass
class _ApplyCounts:
    applied_actions: int = 0
    rejected_actions: int = 0
    archives: int = 0
    updates: int = 0
    deleted_links: int = 0

    def add(self, other: _ApplyCounts) -> None:
        self.applied_actions += other.applied_actions
        self.rejected_actions += other.rejected_actions
        self.archives += other.archives
        self.updates += other.updates
        self.deleted_links += other.deleted_links


@dataclass
class _GuidedQueueCounts:
    queued_suggestions: int = 0
    rejected_actions: int = 0
    suggestion_ids: tuple[str, ...] = ()

    def add(self, other: _GuidedQueueCounts) -> None:
        self.queued_suggestions += other.queued_suggestions
        self.rejected_actions += other.rejected_actions
        self.suggestion_ids = self.suggestion_ids + other.suggestion_ids


@dataclass(frozen=True)
class _AppliedAction:
    changed: bool
    deleted_links: int = 0


@dataclass(frozen=True)
class _CleanupTargetBatch:
    target_count: int
    registry_text: str


class ContextCleanupService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        provider: StructuredOutputProvider | ToolCallProvider,
        provider_name: str,
        model_id: str,
        providers: dict[str, ProviderClient] | None = None,
        prefer_tool_calls: bool = False,
        task_model_preferences: Mapping[str, ModelPreferenceRecord] | None = None,
        prefer_tool_call_tasks: frozenset[str] | None = None,
        transcript_chunk_size: int = _TRANSCRIPT_CHUNK_SIZE,
        target_batch_size: int = _TARGET_BATCH_SIZE,
        action_scan_note_limit: int = _ACTION_SCAN_NOTE_LIMIT,
        action_scan_note_char_limit: int = _ACTION_SCAN_NOTE_CHAR_LIMIT,
        action_message_reference_limit: int = _ACTION_MESSAGE_REFERENCE_LIMIT,
    ) -> None:
        self.repositories = repositories
        self.provider = provider
        self.provider_name = provider_name
        self.model_id = model_id
        self.providers = providers
        self.prefer_tool_calls = prefer_tool_calls
        self.task_model_preferences = dict(task_model_preferences or {})
        self.prefer_tool_call_tasks = prefer_tool_call_tasks
        self.transcript_chunk_size = transcript_chunk_size
        self.target_batch_size = max(1, target_batch_size)
        self.action_scan_note_limit = max(0, action_scan_note_limit)
        self.action_scan_note_char_limit = max(1, action_scan_note_char_limit)
        self.action_message_reference_limit = max(0, action_message_reference_limit)
        self.jobs = JobLifecycleService(repositories=repositories)

    def _provider_model_for_task(self, task: str) -> tuple[str, str]:
        preference = self.task_model_preferences.get(task)
        if preference is not None:
            return preference.provider, preference.model_id
        return self.provider_name, self.model_id

    def _provider_for_name(self, provider_name: str) -> object:
        if self.providers is not None:
            provider = self.providers.get(provider_name)
            if provider is None:
                raise ValueError(
                    f"Context cleanup provider is unavailable: {provider_name}"
                )
            return provider
        return self.provider

    def _task_prefers_tool_calls(self, task: str) -> bool:
        if self.prefer_tool_call_tasks is not None:
            return task in self.prefer_tool_call_tasks
        return self.prefer_tool_calls

    async def analyze_and_apply(
        self,
        save_id: str,
        *,
        apply_guard: Callable[[], AbstractAsyncContextManager[None]] | None = None,
    ) -> ContextCleanupResult:
        if self.repositories.get_save(save_id) is None:
            raise ValueError(f"Unknown save id: {save_id}")
        messages = tuple(self.repositories.list_messages(save_id))
        job = self.jobs.create_running(
            save_id=save_id,
            type="context_cleanup",
            payload={"message_count": len(messages)},
            collect_provider_diagnostics=True,
        )
        started_at = perf_counter()
        log_event(
            "job.running",
            job_id=job.id,
            job_type=job.type,
            save_id=save_id,
            message_count=len(messages),
        )
        transaction_open = False
        scan_batches = 0
        target_batches: tuple[_CleanupTargetBatch, ...] = ()
        proposed_action_count = 0
        counts = _ApplyCounts()
        completed_action_batches = 0
        result: ContextCleanupResult | None = None
        try:
            scan_notes = await self._scan_transcript_chunks(
                save_id=save_id,
                messages=messages,
            )
            scan_batches = len(_message_chunks(messages, self.transcript_chunk_size))
            allowed_evidence_message_ids = _allowed_action_evidence_message_ids(
                messages=messages,
                scan_notes=scan_notes,
                scan_note_limit=self.action_scan_note_limit,
                scan_note_char_limit=self.action_scan_note_char_limit,
                message_reference_limit=self.action_message_reference_limit,
            )
            target_batches = _cleanup_target_batches(
                self.repositories,
                save_id,
                self.target_batch_size,
            )
            for batch_index, batch in enumerate(target_batches):
                actions = await self._propose_actions(
                    save_id=save_id,
                    messages=messages,
                    scan_notes=scan_notes,
                    target_batch=batch,
                    allowed_evidence_message_ids=allowed_evidence_message_ids,
                )
                proposed_action_count += len(actions)
                async with (
                    apply_guard() if apply_guard is not None else nullcontext()
                ):
                    if self.repositories.get_save(save_id) is None:
                        raise ValueError(f"Unknown save id: {save_id}")
                    self.repositories.begin_immediate_transaction()
                    transaction_open = True
                    batch_counts = self._apply_actions(
                        save_id=save_id,
                        actions=actions,
                        allowed_evidence_message_ids=allowed_evidence_message_ids,
                    )
                    self.repositories.commit_transaction()
                    transaction_open = False
                    counts.add(batch_counts)
                    completed_action_batches += 1
                    if batch_index == len(target_batches) - 1:
                        result = ContextCleanupResult(
                            save_id=save_id,
                            scanned_messages=len(messages),
                            scan_batches=scan_batches,
                            cleanup_target_count=sum(
                                batch.target_count for batch in target_batches
                            ),
                            action_batches=len(target_batches),
                            proposed_actions=proposed_action_count,
                            applied_actions=counts.applied_actions,
                            rejected_actions=counts.rejected_actions,
                            archives=counts.archives,
                            updates=counts.updates,
                            deleted_links=counts.deleted_links,
                        )
                        self.jobs.succeed(
                            job.id,
                            result=_result_json(result),
                        )
            if result is None:
                raise RuntimeError("Context cleanup finished without a result")
        except Exception as exc:
            if transaction_open:
                self.repositories.rollback_transaction()
            try:
                self.jobs.fail(
                    job.id,
                    error=redact_text(str(exc)) or exc.__class__.__name__,
                    result={
                        "scanned_messages": len(messages),
                        "scan_batches": scan_batches,
                        "cleanup_target_count": sum(
                            batch.target_count for batch in target_batches
                        ),
                        "action_batches": len(target_batches),
                        "completed_action_batches": completed_action_batches,
                        "proposed_actions": proposed_action_count,
                        "applied_actions": counts.applied_actions,
                        "rejected_actions": counts.rejected_actions,
                        "archives": counts.archives,
                        "updates": counts.updates,
                        "deleted_links": counts.deleted_links,
                    },
                )
            except ValueError:
                if self.repositories.get_save(save_id) is not None:
                    raise
            log_error_event(
                "job.failed",
                job_id=job.id,
                job_type=job.type,
                save_id=save_id,
                duration_ms=_elapsed_ms(started_at),
                **exception_log_fields(exc),
            )
            raise
        log_event(
            "job.succeeded",
            job_id=job.id,
            job_type=job.type,
            save_id=save_id,
            duration_ms=_elapsed_ms(started_at),
            proposed_actions=result.proposed_actions,
            applied_actions=result.applied_actions,
            rejected_actions=result.rejected_actions,
        )
        return result

    async def propose_guided_cleanup(
        self,
        save_id: str,
        *,
        instruction: str,
        apply_guard: Callable[[], AbstractAsyncContextManager[None]] | None = None,
    ) -> GuidedCleanupResult:
        instruction = _normalized_instruction(instruction)
        if not instruction:
            raise ValueError("Cleanup instructions are required")
        if self.repositories.get_save(save_id) is None:
            raise ValueError(f"Unknown save id: {save_id}")
        job = self.jobs.create_running(
            save_id=save_id,
            type="guided_context_cleanup",
            payload={"instruction": redact_text(instruction)},
            collect_provider_diagnostics=True,
        )
        started_at = perf_counter()
        log_event(
            "job.running",
            job_id=job.id,
            job_type=job.type,
            save_id=save_id,
            instruction_chars=len(instruction),
        )
        transaction_open = False
        target_batches: tuple[_CleanupTargetBatch, ...] = ()
        proposed_action_count = 0
        completed_action_batches = 0
        counts = _GuidedQueueCounts()
        result: GuidedCleanupResult | None = None
        try:
            messages = tuple(self.repositories.list_messages(save_id))
            allowed_evidence_message_ids = _message_reference_ids(
                messages,
                limit=self.action_message_reference_limit,
            )
            target_batches = _cleanup_target_batches(
                self.repositories,
                save_id,
                self.target_batch_size,
            )
            for batch_index, batch in enumerate(target_batches):
                actions = await self._propose_guided_actions(
                    save_id=save_id,
                    instruction=instruction,
                    messages=messages,
                    target_batch=batch,
                    allowed_evidence_message_ids=allowed_evidence_message_ids,
                )
                proposed_action_count += len(actions)
                async with (
                    apply_guard() if apply_guard is not None else nullcontext()
                ):
                    if self.repositories.get_save(save_id) is None:
                        raise ValueError(f"Unknown save id: {save_id}")
                    self.repositories.begin_immediate_transaction()
                    transaction_open = True
                    batch_counts = self._queue_guided_actions(
                        save_id=save_id,
                        instruction=instruction,
                        actions=actions,
                        allowed_evidence_message_ids=allowed_evidence_message_ids,
                    )
                    self.repositories.commit_transaction()
                    transaction_open = False
                    counts.add(batch_counts)
                    completed_action_batches += 1
                    if batch_index == len(target_batches) - 1:
                        result = GuidedCleanupResult(
                            save_id=save_id,
                            instruction=instruction,
                            cleanup_target_count=sum(
                                target_batch.target_count
                                for target_batch in target_batches
                            ),
                            action_batches=len(target_batches),
                            proposed_actions=proposed_action_count,
                            queued_suggestions=counts.queued_suggestions,
                            rejected_actions=counts.rejected_actions,
                            suggestion_ids=counts.suggestion_ids,
                        )
                        self.jobs.succeed(
                            job.id,
                            result=_guided_result_json(result),
                        )
            if result is None:
                raise RuntimeError("Guided context cleanup finished without a result")
        except Exception as exc:
            if transaction_open:
                self.repositories.rollback_transaction()
            try:
                self.jobs.fail(
                    job.id,
                    error=redact_text(str(exc)) or exc.__class__.__name__,
                    result={
                        "cleanup_target_count": sum(
                            batch.target_count for batch in target_batches
                        ),
                        "action_batches": len(target_batches),
                        "completed_action_batches": completed_action_batches,
                        "proposed_actions": proposed_action_count,
                        "queued_suggestions": counts.queued_suggestions,
                        "rejected_actions": counts.rejected_actions,
                        "suggestion_ids": list(counts.suggestion_ids),
                    },
                )
            except ValueError:
                if self.repositories.get_save(save_id) is not None:
                    raise
            log_error_event(
                "job.failed",
                job_id=job.id,
                job_type=job.type,
                save_id=save_id,
                duration_ms=_elapsed_ms(started_at),
                **exception_log_fields(exc),
            )
            raise
        log_event(
            "job.succeeded",
            job_id=job.id,
            job_type=job.type,
            save_id=save_id,
            duration_ms=_elapsed_ms(started_at),
            proposed_actions=result.proposed_actions,
            queued_suggestions=result.queued_suggestions,
            rejected_actions=result.rejected_actions,
        )
        return result

    async def _scan_transcript_chunks(
        self,
        *,
        save_id: str,
        messages: tuple[MessageRecord, ...],
    ) -> tuple[str, ...]:
        notes: list[str] = []
        for chunk_index, chunk in enumerate(
            _message_chunks(messages, self.transcript_chunk_size)
        ):
            task = CONTEXT_CLEANUP_SCAN_TASK
            provider_name, model_id = self._provider_model_for_task(task)
            if self._task_prefers_tool_calls(task) and isinstance(
                self._provider_for_name(provider_name),
                ToolCallProvider,
            ):
                notes.extend(
                    await self._scan_transcript_chunk_with_tool_calls(
                        save_id=save_id,
                        task=task,
                        chunk_index=chunk_index,
                        messages=chunk,
                    )
                )
                continue
            response = await self._generate_structured_output(
                save_id=save_id,
                task=task,
                request=StructuredOutputRequest(
                    provider=provider_name,
                    model_id=model_id,
                    schema_name="context_cleanup_scan",
                    schema=_scan_schema(tuple(message.id for message in chunk)),
                    messages=_scan_messages(
                        save_id=save_id,
                        chunk_index=chunk_index,
                        messages=chunk,
                    ),
                    temperature=0.0,
                )
            )
            notes.extend(_scan_notes_from_data(response.data))
        return tuple(notes)

    async def _propose_actions(
        self,
        *,
        save_id: str,
        messages: tuple[MessageRecord, ...],
        scan_notes: tuple[str, ...],
        target_batch: _CleanupTargetBatch,
        allowed_evidence_message_ids: frozenset[str],
    ) -> tuple[CleanupAction, ...]:
        task = CONTEXT_CLEANUP_ACTIONS_TASK
        provider_name, model_id = self._provider_model_for_task(task)
        if self._task_prefers_tool_calls(task) and isinstance(
            self._provider_for_name(provider_name),
            ToolCallProvider,
        ):
            return await self._propose_actions_with_tool_calls(
                save_id=save_id,
                task=task,
                request=ToolCallRequest(
                    provider=provider_name,
                    model_id=model_id,
                    messages=_action_tool_messages(
                        target_batch=target_batch,
                        scan_notes=scan_notes,
                        messages=messages,
                        scan_note_limit=self.action_scan_note_limit,
                        scan_note_char_limit=self.action_scan_note_char_limit,
                        message_reference_limit=self.action_message_reference_limit,
                    ),
                    tools=_action_tool_definitions(),
                    temperature=0.0,
                ),
                allowed_evidence_message_ids=allowed_evidence_message_ids,
            )
        response = await self._generate_structured_output(
            save_id=save_id,
            task=task,
            request=StructuredOutputRequest(
                provider=provider_name,
                model_id=model_id,
                schema_name="context_cleanup_actions",
                schema=_actions_schema(),
                messages=_action_messages(
                    target_batch=target_batch,
                    scan_notes=scan_notes,
                    messages=messages,
                    scan_note_limit=self.action_scan_note_limit,
                    scan_note_char_limit=self.action_scan_note_char_limit,
                    message_reference_limit=self.action_message_reference_limit,
                ),
                temperature=0.0,
            )
        )
        return _actions_from_data(response.data)

    async def _propose_guided_actions(
        self,
        *,
        save_id: str,
        instruction: str,
        messages: tuple[MessageRecord, ...],
        target_batch: _CleanupTargetBatch,
        allowed_evidence_message_ids: frozenset[str],
    ) -> tuple[CleanupAction, ...]:
        task = GUIDED_CONTEXT_CLEANUP_TASK
        provider_name, model_id = self._provider_model_for_task(task)
        if self._task_prefers_tool_calls(task) and isinstance(
            self._provider_for_name(provider_name),
            ToolCallProvider,
        ):
            return await self._propose_actions_with_tool_calls(
                save_id=save_id,
                task=task,
                request=ToolCallRequest(
                    provider=provider_name,
                    model_id=model_id,
                    messages=_guided_action_tool_messages(
                        instruction=instruction,
                        target_batch=target_batch,
                        messages=messages,
                        message_reference_limit=self.action_message_reference_limit,
                    ),
                    tools=_action_tool_definitions(),
                    temperature=0.0,
                ),
                allowed_evidence_message_ids=allowed_evidence_message_ids,
            )
        response = await self._generate_structured_output(
            save_id=save_id,
            task=task,
            request=StructuredOutputRequest(
                provider=provider_name,
                model_id=model_id,
                schema_name="guided_context_cleanup_actions",
                schema=_actions_schema(),
                messages=_guided_action_messages(
                    instruction=instruction,
                    target_batch=target_batch,
                    messages=messages,
                    message_reference_limit=self.action_message_reference_limit,
                ),
                temperature=0.0,
            ),
        )
        return _actions_from_data(response.data)

    async def _generate_structured_output(
        self,
        *,
        save_id: str,
        task: str,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        provider = self._provider_for_name(request.provider)
        if not isinstance(provider, StructuredOutputProvider):
            raise ValueError("Context cleanup provider lacks structured output")
        request = request_with_openrouter_routing(
            self.repositories,
            request,
            task=task,
            save_id=save_id,
        )
        if self.providers is None:
            return await provider.generate_structured_output(
                budget_structured_output_request(
                    self.repositories,
                    request,
                    task=task,
                )
            )
        return await structured_output_with_fallback(
            repositories=self.repositories,
            providers=self.providers,
            request=request,
            task=task,
            save_id=save_id,
        )

    async def _scan_transcript_chunk_with_tool_calls(
        self,
        *,
        save_id: str,
        task: str,
        chunk_index: int,
        messages: tuple[MessageRecord, ...],
    ) -> tuple[str, ...]:
        provider_name, model_id = self._provider_model_for_task(task)
        request = ToolCallRequest(
            provider=provider_name,
            model_id=model_id,
            messages=_scan_tool_messages(
                save_id=save_id,
                chunk_index=chunk_index,
                messages=messages,
            ),
            tools=_scan_tool_definitions(tuple(message.id for message in messages)),
            temperature=0.0,
        )
        return await self._generate_tool_call_items(
            save_id=save_id,
            task=task,
            request=request,
            validator=lambda call, schemas: _validate_scan_note_tool_call(
                call,
                tool_schemas=schemas,
            ),
        )

    async def _propose_actions_with_tool_calls(
        self,
        *,
        save_id: str,
        task: str,
        request: ToolCallRequest,
        allowed_evidence_message_ids: frozenset[str],
    ) -> tuple[CleanupAction, ...]:
        return await self._generate_tool_call_items(
            save_id=save_id,
            task=task,
            request=request,
            validator=lambda call, schemas: _validate_cleanup_action_tool_call(
                call,
                tool_schemas=schemas,
                allowed_evidence_message_ids=allowed_evidence_message_ids,
            ),
        )

    async def _generate_tool_call_items[T](
        self,
        *,
        save_id: str,
        task: str,
        request: ToolCallRequest,
        validator: Callable[
            [ProviderToolCall, dict[str, dict[str, object]]],
            tuple[bool, dict[str, str], T | None],
        ],
    ) -> tuple[T, ...]:
        provider = self._provider_for_name(request.provider)
        if not isinstance(provider, ToolCallProvider):
            raise ValueError("Context cleanup provider lacks tool calling")
        request = request_with_openrouter_routing(
            self.repositories,
            request,
            task=task,
            save_id=save_id,
        )
        if self.providers is None:
            return await _generate_cleanup_tool_items(
                repositories=self.repositories,
                provider=provider,
                request=request,
                task=task,
                validator=validator,
            )
        try:
            return await _generate_cleanup_tool_items(
                repositories=self.repositories,
                provider=provider,
                request=request,
                task=task,
                validator=validator,
            )
        except ProviderError as exc:
            fallback_request = tool_call_fallback_request(
                repositories=self.repositories,
                providers=self.providers,
                request=request,
                save_id=save_id,
            )
            if fallback_request is None:
                reason = tool_call_fallback_skip_reason(
                    repositories=self.repositories,
                    providers=self.providers,
                    save_id=save_id,
                )
                log_event(
                    "provider.tool_call_fallback_skipped",
                    provider=request.provider,
                    model=request.model_id,
                    task=task,
                    reason=reason,
                )
                raise provider_error_with_fallback_skipped_reason(exc, reason) from exc
            fallback_provider = self.providers[fallback_request.provider]
            if not isinstance(fallback_provider, ToolCallProvider):
                reason = "fallback_provider_unavailable"
                log_event(
                    "provider.tool_call_fallback_skipped",
                    provider=request.provider,
                    model=request.model_id,
                    task=task,
                    reason=reason,
                )
                raise provider_error_with_fallback_skipped_reason(exc, reason) from exc
            log_event(
                "provider.tool_call_fallback_started",
                provider=fallback_request.provider,
                model=fallback_request.model_id,
                task=task,
            )
            try:
                return await _generate_cleanup_tool_items(
                    repositories=self.repositories,
                    provider=fallback_provider,
                    request=fallback_request,
                    task=task,
                    validator=validator,
                )
            except ProviderError as fallback_exc:
                raise provider_error_with_fallback_attempted(
                    fallback_exc,
                    provider=fallback_request.provider,
                    model_id=fallback_request.model_id,
                ) from fallback_exc

    def _apply_actions(
        self,
        *,
        save_id: str,
        actions: tuple[CleanupAction, ...],
        allowed_evidence_message_ids: frozenset[str],
    ) -> _ApplyCounts:
        counts = _ApplyCounts()
        seen: set[tuple[str, str, str, str]] = set()
        message_ids = {
            message.id for message in self.repositories.list_messages(save_id)
        }
        for action in actions:
            key = (
                action.operation,
                action.target_type,
                action.target_id,
                action.field_path,
            )
            if key in seen:
                self._reject(save_id, action, "Duplicate cleanup action")
                counts.rejected_actions += 1
                continue
            seen.add(key)
            reason = self._validate_action(
                save_id,
                action,
                message_ids,
                allowed_evidence_message_ids,
            )
            if reason is not None:
                self._reject(save_id, action, reason)
                counts.rejected_actions += 1
                continue
            applied = self._apply_valid_action(save_id, action)
            if applied.changed:
                counts.applied_actions += 1
                counts.deleted_links += applied.deleted_links
                if action.operation == "archive":
                    counts.archives += 1
                elif action.operation == "update":
                    counts.updates += 1
                elif action.operation == "delete":
                    counts.deleted_links += 1
            else:
                self._reject(save_id, action, "Cleanup action made no change")
                counts.rejected_actions += 1
        return counts

    def _queue_guided_actions(
        self,
        *,
        save_id: str,
        instruction: str,
        actions: tuple[CleanupAction, ...],
        allowed_evidence_message_ids: frozenset[str],
    ) -> _GuidedQueueCounts:
        counts = _GuidedQueueCounts()
        seen: set[tuple[str, str, str, str]] = set()
        message_ids = {
            message.id for message in self.repositories.list_messages(save_id)
        }
        for action in actions:
            key = (
                action.operation,
                action.target_type,
                action.target_id,
                action.field_path,
            )
            if key in seen:
                self._reject_guided(
                    save_id,
                    instruction,
                    action,
                    "Duplicate cleanup action",
                    message_ids,
                )
                counts.rejected_actions += 1
                continue
            seen.add(key)
            reason = self._validate_guided_action(
                save_id,
                action,
                message_ids,
                allowed_evidence_message_ids,
            )
            if reason is not None:
                self._reject_guided(
                    save_id,
                    instruction,
                    action,
                    reason,
                    message_ids,
                )
                counts.rejected_actions += 1
                continue
            suggestion_id = self._queue_guided_suggestion(
                save_id,
                instruction,
                action,
                message_ids,
            )
            if suggestion_id is None:
                continue
            counts.queued_suggestions += 1
            counts.suggestion_ids = counts.suggestion_ids + (suggestion_id,)
        return counts

    def _validate_action(
        self,
        save_id: str,
        action: CleanupAction,
        message_ids: set[str],
        allowed_evidence_message_ids: frozenset[str],
    ) -> str | None:
        if action.operation not in {"archive", "update", "delete"}:
            return f"Unsupported cleanup operation: {action.operation}"
        if action.target_type not in _TARGET_TYPES:
            return f"Unsupported cleanup target type: {action.target_type}"
        if not action.target_id:
            return "Cleanup action target_id is required"
        if action.confidence < _MIN_ACTION_CONFIDENCE:
            return "Cleanup action confidence is below threshold"
        if not action.evidence_message_ids:
            return "Cleanup action evidence_message_ids are required"
        unknown_evidence = [
            message_id
            for message_id in action.evidence_message_ids
            if message_id not in message_ids
        ]
        if unknown_evidence:
            return f"Unknown evidence message id: {unknown_evidence[0]}"
        evidence_error = _surfaced_evidence_error(
            action.evidence_message_ids,
            allowed_evidence_message_ids,
        )
        if evidence_error is not None:
            return evidence_error
        target = _target_record(
            self.repositories,
            save_id,
            action.target_type,
            action.target_id,
        )
        if target is None:
            return "Cleanup action target is unknown or not in the active save"
        if action.operation == "archive":
            if action.target_type not in _ARCHIVE_TARGET_TYPES:
                return f"Cleanup target cannot be archived: {action.target_type}"
            if _record_archive_locked(target):
                return "Cleanup target archive is locked"
            protected_error = _protected_character_archive_error(
                action.target_type,
                target,
            )
            if protected_error is not None:
                return protected_error
            if _high_value_continuity_target(action.target_type, target) and not (
                _has_explicit_contradiction_evidence(action)
            ):
                return (
                    "High-value continuity facts require explicit contradiction "
                    "evidence before archive"
                )
            return None
        if action.operation == "delete":
            if action.target_type != "entity_link":
                return "Only entity links may be deleted by context cleanup"
            return None
        if action.target_type == "entity_link":
            return "Entity links cannot be updated by context cleanup"
        if action.field_path not in _UPDATE_FIELDS.get(action.target_type, frozenset()):
            return (
                f"Unsupported cleanup field: "
                f"{action.target_type}.{action.field_path}"
            )
        if _target_field_is_locked(action.target_type, target, action.field_path):
            return "Cleanup field is locked"
        value_error = _field_value_error(
            action.target_type,
            action.field_path,
            action.value,
        )
        if value_error is not None:
            return value_error
        if (
            action.target_type == "scene_snapshot"
            and action.field_path == "present_character_ids"
            and isinstance(action.value, list)
        ):
            snapshot = cast(SceneSnapshotRecord, target)
            locked_ids = {
                character.id
                for character in self.repositories.list_characters(save_id)
                if character_field_is_locked(character.locked_fields, "present")
            }
            reconciled = reconcile_character_presence_locks(
                current_present_ids=snapshot.present_character_ids,
                proposed_present_ids=cast(list[str], action.value),
                locked_character_ids=locked_ids,
            )
            if reconciled != set(action.value):
                return "Cleanup character presence is locked"
        reference_error = _reference_validation_error(
            self.repositories,
            save_id,
            action.target_type,
            action.field_path,
            action.value,
        )
        if reference_error is not None:
            return reference_error
        return None

    def _validate_guided_action(
        self,
        save_id: str,
        action: CleanupAction,
        message_ids: set[str],
        allowed_evidence_message_ids: frozenset[str],
    ) -> str | None:
        if action.operation not in {"archive", "update", "delete"}:
            return f"Unsupported cleanup operation: {action.operation}"
        if action.target_type not in _TARGET_TYPES:
            return f"Unsupported cleanup target type: {action.target_type}"
        if not action.target_id:
            return "Cleanup action target_id is required"
        unknown_evidence = [
            message_id
            for message_id in action.evidence_message_ids
            if message_id not in message_ids
        ]
        if unknown_evidence:
            return f"Unknown evidence message id: {unknown_evidence[0]}"
        evidence_error = _surfaced_evidence_error(
            action.evidence_message_ids,
            allowed_evidence_message_ids,
        )
        if evidence_error is not None:
            return evidence_error
        target = _target_record(
            self.repositories,
            save_id,
            action.target_type,
            action.target_id,
        )
        if target is None:
            return "Cleanup action target is unknown or not in the active save"
        if action.operation == "archive":
            if action.target_type not in _ARCHIVE_TARGET_TYPES:
                return f"Cleanup target cannot be archived: {action.target_type}"
            if _record_archive_locked(target):
                return "Cleanup target archive is locked"
            protected_error = _protected_character_archive_error(
                action.target_type,
                target,
            )
            if protected_error is not None:
                return protected_error
            return None
        if action.operation == "delete":
            if action.target_type != "entity_link":
                return "Only entity links may be deleted by context cleanup"
            return None
        if action.target_type == "entity_link":
            return "Entity links cannot be updated by context cleanup"
        if action.field_path not in _UPDATE_FIELDS.get(action.target_type, frozenset()):
            return (
                f"Unsupported cleanup field: "
                f"{action.target_type}.{action.field_path}"
            )
        if _target_field_is_locked(action.target_type, target, action.field_path):
            return "Cleanup field is locked"
        value_error = _field_value_error(
            action.target_type,
            action.field_path,
            action.value,
        )
        if value_error is not None:
            return value_error
        reference_error = _reference_validation_error(
            self.repositories,
            save_id,
            action.target_type,
            action.field_path,
            action.value,
        )
        if reference_error is not None:
            return reference_error
        return None

    def _apply_valid_action(
        self,
        save_id: str,
        action: CleanupAction,
    ) -> _AppliedAction:
        target = _target_record(
            self.repositories,
            save_id,
            action.target_type,
            action.target_id,
        )
        if target is None:
            return _AppliedAction(changed=False)
        before = _action_before_value(target, action)
        if action.operation == "archive":
            deleted_links = self._archive_target(save_id, action, target)
            self._audit(
                save_id,
                action,
                before=before,
                after=None,
                operation="archived",
            )
            return _AppliedAction(changed=True, deleted_links=deleted_links)
        if action.operation == "delete":
            self.repositories.delete_entity_link(action.target_id)
            self._audit(save_id, action, before=before, after=None, operation="deleted")
            return _AppliedAction(changed=True)
        after = action.value
        if before == after:
            return _AppliedAction(changed=False)
        self._update_target(save_id, action, target)
        self._audit(save_id, action, before=before, after=after, operation="updated")
        if action.target_type == "world_state":
            state = cast(WorldStateRecord, target)
            self.repositories.add_state_change(
                save_id=save_id,
                operation="updated",
                state_key=state.key,
                before_json=_dump_json(before),
                after_json=_dump_json(after),
                source_message_id=action.evidence_message_ids[-1],
            )
        return _AppliedAction(changed=True)

    def _archive_target(
        self,
        save_id: str,
        action: CleanupAction,
        target: object,
    ) -> int:
        protected_error = _protected_character_archive_error(
            action.target_type,
            target,
        )
        if protected_error is not None:
            raise ValueError(protected_error)
        deleted_links = _delete_related_entity_links(
            self.repositories,
            save_id=save_id,
            entity_type=action.target_type,
            entity_id=action.target_id,
        )
        if action.target_type == "world_state":
            state = cast(WorldStateRecord, target)
            if state.key == "loop.current":
                raise ValueError("The time-loop clock state cannot be archived")
            self.repositories.archive_world_state(save_id=save_id, key=state.key)
            self.repositories.add_state_change(
                save_id=save_id,
                operation="archived",
                state_key=state.key,
                before_json=_dump_json(state.value),
                after_json=None,
                source_message_id=action.evidence_message_ids[-1],
            )
        elif action.target_type == "memory":
            self.repositories.archive_memory(action.target_id)
        elif action.target_type == "summary":
            self.repositories.archive_summary(action.target_id)
        elif action.target_type == "location":
            self.repositories.archive_location(action.target_id)
        elif action.target_type == "character":
            self.repositories.archive_character(action.target_id)
        elif action.target_type == "active_thread":
            self.repositories.archive_active_thread(action.target_id)
        return deleted_links

    def _update_target(
        self,
        save_id: str,
        action: CleanupAction,
        target: object,
    ) -> None:
        if action.target_type == "world_state":
            state = cast(WorldStateRecord, target)
            if state.key == "loop.current":
                raise ValueError("The time-loop clock state cannot be modified")
            self.repositories.upsert_world_state(
                save_id=save_id,
                key=state.key,
                value=(
                    cast(dict[str, object], action.value)
                    if action.field_path == "value"
                    else state.value
                ),
                category=(
                    cast(str, action.value)
                    if action.field_path == "category"
                    else state.category
                ),
                confidence=(
                    cast(float, action.value)
                    if action.field_path == "confidence"
                    else state.confidence
                ),
                source_message_id=action.evidence_message_ids[-1],
                state_id=state.id,
            )
        elif action.target_type == "memory":
            memory = cast(MemoryRecord, target)
            self.repositories.update_memory(
                memory_id=memory.id,
                body=(
                    cast(str, action.value)
                    if action.field_path == "body"
                    else memory.body
                ),
                tags=(
                    cast(list[str], action.value)
                    if action.field_path == "tags"
                    else memory.tags
                ),
                importance=(
                    cast(float, action.value)
                    if action.field_path == "importance"
                    else memory.importance
                ),
            )
        elif action.target_type == "summary":
            summary = cast(SummaryRecord, target)
            self.repositories.update_summary(
                summary_id=summary.id,
                body=cast(str, action.value),
            )
        elif action.target_type == "scene_snapshot":
            snapshot = _replace_scene_snapshot_field(
                cast(SceneSnapshotRecord, target),
                action.field_path,
                action.value,
            )
            world_time_kwargs: dict[str, Any] = {}
            if action.field_path in _SCENE_WORLD_TIME_FIELDS:
                canonical_world_time = canonical_world_time_from_legacy(
                    in_world_time=snapshot.in_world_time,
                    time_of_day=(
                        ""
                        if action.field_path == "in_world_time"
                        else snapshot.time_of_day
                    ),
                    day_of_week=snapshot.day_of_week,
                    world_day_index=snapshot.world_day_index,
                    source_message_id=action.evidence_message_ids[-1],
                    confidence=action.confidence,
                )
                world_time_kwargs = {
                    "world_time_day_index": canonical_world_time.day_index,
                    "world_time_day_label": canonical_world_time.day_label,
                    "world_time_phase": canonical_world_time.phase,
                    "world_time_clock_minutes": (
                        canonical_world_time.clock_minutes
                        if canonical_world_time.clock_minutes is not None
                        else snapshot.world_time_clock_minutes
                    ),
                    "world_time_period_label": (
                        canonical_world_time.period_label
                        or snapshot.world_time_period_label
                    ),
                    "world_time_source_message_id": (
                        canonical_world_time.source_message_id
                    ),
                    "world_time_confidence": canonical_world_time.confidence,
                }
                if action.field_path == "in_world_time":
                    display_world_time = canonical_world_time_from_values(
                        day_index=canonical_world_time.day_index,
                        day_label=canonical_world_time.day_label,
                        phase=canonical_world_time.phase,
                        clock_minutes=(
                            canonical_world_time.clock_minutes
                            if canonical_world_time.clock_minutes is not None
                            else snapshot.world_time_clock_minutes
                        ),
                        period_label=(
                            canonical_world_time.period_label
                            or snapshot.world_time_period_label
                        ),
                        source_message_id=canonical_world_time.source_message_id,
                        confidence=canonical_world_time.confidence,
                        legacy_in_world_time=snapshot.in_world_time,
                        legacy_time_of_day=snapshot.time_of_day,
                        legacy_day_of_week=snapshot.day_of_week,
                        legacy_world_day_index=snapshot.world_day_index,
                    )
                    legacy_fields = legacy_world_time_fields(display_world_time)
                    snapshot = replace(
                        snapshot,
                        in_world_time=cast(str, legacy_fields["in_world_time"]),
                        time_of_day=cast(str, legacy_fields["time_of_day"]),
                        day_of_week=cast(str, legacy_fields["day_of_week"]),
                        world_day_index=cast(
                            int | None,
                            legacy_fields["world_day_index"],
                        ),
                    )
            if action.field_path in _SCENE_WORLD_TIME_FIELDS:
                from bragi.services.time_loop_time_policy import TimeLoopTimePolicy

                loop_policy = TimeLoopTimePolicy(self.repositories, save_id=save_id)
                loop_policy.ensure_baseline(cast(SceneSnapshotRecord, target))
            saved_snapshot = self.repositories.upsert_scene_snapshot(
                save_id=save_id,
                current_location_id=snapshot.current_location_id,
                situation=snapshot.situation,
                objective=snapshot.objective,
                in_world_time=snapshot.in_world_time,
                time_of_day=snapshot.time_of_day,
                day_of_week=snapshot.day_of_week,
                world_day_index=snapshot.world_day_index,
                weather=snapshot.weather,
                mood=snapshot.mood,
                nearby_objects=snapshot.nearby_objects,
                hazards=snapshot.hazards,
                present_character_ids=snapshot.present_character_ids,
                source_message_id=action.evidence_message_ids[-1],
                locked_fields=snapshot.locked_fields,
                snapshot_id=snapshot.id,
                **world_time_kwargs,
            )
            if action.field_path in _SCENE_WORLD_TIME_FIELDS:
                loop_policy.ensure_baseline(saved_snapshot)
                loop_policy.sync_current(
                    saved_snapshot,
                    transition="cleanup_scene_update",
                    source_message_id=action.evidence_message_ids[-1],
                )
        elif action.target_type == "location":
            location = cast(LocationRecord, target)
            self.repositories.update_location(
                _replace_location_field(location, action.field_path, action.value)
            )
        elif action.target_type == "character":
            character = cast(CharacterRecord, target)
            self.repositories.update_character(
                _replace_character_field(character, action.field_path, action.value)
            )
        elif action.target_type == "active_thread":
            thread = cast(ActiveThreadRecord, target)
            self.repositories.update_active_thread(
                _replace_thread_field(thread, action.field_path, action.value)
            )

    def _queue_guided_suggestion(
        self,
        save_id: str,
        instruction: str,
        action: CleanupAction,
        message_ids: set[str],
    ) -> str | None:
        target = _target_record(
            self.repositories,
            save_id,
            action.target_type,
            action.target_id,
        )
        if target is None:
            raise ValueError(
                "Cleanup action target is unknown or not in the active save"
            )
        source_message_ids = _known_message_ids(
            action.evidence_message_ids,
            message_ids,
        )
        update_type = action.operation
        field_path = action.field_path or "*"
        proposed_value = action.value
        before = _action_before_value(target, action)
        after = action.value
        if action.target_type == "world_state":
            state = cast(WorldStateRecord, target)
            field_path = state.key
            source_message_id = source_message_ids[-1] if source_message_ids else None
            if action.operation == "update":
                update_type = "upsert"
                proposed_value = {
                    "operation": "upsert",
                    "key": state.key,
                    "value": action.value,
                    "category": state.category,
                    "confidence": action.confidence,
                    "source_message_id": source_message_id,
                }
                after = proposed_value
            elif action.operation == "archive":
                update_type = "delete"
                proposed_value = {
                    "operation": "delete",
                    "key": state.key,
                    "source_message_id": source_message_id,
                }
                after = proposed_value
        elif action.operation in {"archive", "delete"}:
            field_path = "*"
            proposed_value = None
            after = None
        reason = _guided_reason(instruction=instruction, action_reason=action.reason)
        existing = self.repositories.find_pending_context_update_suggestion(
            save_id=save_id,
            update_type=update_type,
            entity_type=action.target_type,
            entity_id=action.target_id,
            field_path=field_path,
            proposed_value=proposed_value,
        )
        if existing is not None:
            log_event(
                "context_cleanup.guided_suggestion_suppressed",
                save_id=save_id,
                entity_type=action.target_type,
                entity_id=action.target_id,
                field_path=field_path,
                suggestion_id=existing.id,
                reason="duplicate_pending",
            )
            return None
        suggestion = self.repositories.add_context_update_suggestion(
            save_id=save_id,
            update_type=update_type,
            entity_type=action.target_type,
            entity_id=action.target_id,
            field_path=field_path,
            proposed_value=proposed_value,
            status="pending",
            reason=reason,
            confidence=action.confidence,
            source_message_ids=list(source_message_ids),
        )
        self.repositories.add_context_update_audit(
            save_id=save_id,
            suggestion_id=suggestion.id,
            operation="guided_cleanup_queued",
            entity_type=action.target_type,
            entity_id=action.target_id,
            field_path=field_path,
            before=before,
            after=after,
            reason=reason,
            confidence=action.confidence,
            source_message_ids=list(source_message_ids),
        )
        return suggestion.id

    def _reject(self, save_id: str, action: CleanupAction, reason: str) -> None:
        self._audit(
            save_id,
            action,
            before=None,
            after=action.value,
            operation="rejected",
            reason=f"{reason}: {action.reason}".strip(": "),
        )

    def _reject_guided(
        self,
        save_id: str,
        instruction: str,
        action: CleanupAction,
        reason: str,
        message_ids: set[str],
    ) -> None:
        self.repositories.add_context_update_audit(
            save_id=save_id,
            operation="guided_cleanup_rejected",
            entity_type=action.target_type or "unknown",
            entity_id=action.target_id or None,
            field_path=action.field_path or "*",
            before=None,
            after=action.value,
            reason=_guided_reason(
                instruction=instruction,
                action_reason=f"{reason}: {action.reason}".strip(": "),
            ),
            confidence=action.confidence,
            source_message_ids=list(
                _known_message_ids(action.evidence_message_ids, message_ids)
            ),
        )

    def _audit(
        self,
        save_id: str,
        action: CleanupAction,
        *,
        before: object | None,
        after: object | None,
        operation: str,
        reason: str | None = None,
    ) -> None:
        self.repositories.add_context_update_audit(
            save_id=save_id,
            operation=operation,
            entity_type=action.target_type,
            entity_id=action.target_id or None,
            field_path=action.field_path or "*",
            before=before,
            after=after,
            reason=reason if reason is not None else action.reason,
            confidence=action.confidence,
            source_message_ids=list(action.evidence_message_ids),
        )


def _result_json(result: ContextCleanupResult) -> dict[str, object]:
    return {
        "scanned_messages": result.scanned_messages,
        "scan_batches": result.scan_batches,
        "cleanup_target_count": result.cleanup_target_count,
        "action_batches": result.action_batches,
        "proposed_actions": result.proposed_actions,
        "applied_actions": result.applied_actions,
        "rejected_actions": result.rejected_actions,
        "archives": result.archives,
        "updates": result.updates,
        "deleted_links": result.deleted_links,
    }


def _guided_result_json(result: GuidedCleanupResult) -> dict[str, object]:
    return {
        "cleanup_target_count": result.cleanup_target_count,
        "action_batches": result.action_batches,
        "proposed_actions": result.proposed_actions,
        "queued_suggestions": result.queued_suggestions,
        "rejected_actions": result.rejected_actions,
        "suggestion_ids": list(result.suggestion_ids),
    }


def _scan_schema(message_ids: tuple[str, ...]) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "notes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "message_id": {"type": "string", "enum": list(message_ids)},
                        "note": {"type": "string"},
                    },
                    "required": ["message_id", "note"],
                },
            }
        },
        "required": ["notes"],
    }


def _scan_tool_definitions(message_ids: tuple[str, ...]) -> tuple[ToolDefinition, ...]:
    return (
        ToolDefinition(
            name="note_cleanup_candidate",
            description="Record one transcript note about possible context cleanup.",
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "message_id": {"type": "string", "enum": list(message_ids)},
                    "note": {"type": "string"},
                },
                "required": ["message_id", "note"],
            },
        ),
    )


def _actions_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["archive", "update", "delete"],
                        },
                        "target_type": {
                            "type": "string",
                            "enum": sorted(_TARGET_TYPES),
                        },
                        "target_id": {"type": "string"},
                        "field_path": {"type": "string"},
                        "value": {
                            "type": [
                                "object",
                                "array",
                                "string",
                                "number",
                                "integer",
                                "boolean",
                                "null",
                            ]
                        },
                        "reason": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "evidence_message_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "operation",
                        "target_type",
                        "target_id",
                        "field_path",
                        "value",
                        "reason",
                        "confidence",
                        "evidence_message_ids",
                    ],
                },
            }
        },
        "required": ["actions"],
    }


def _action_tool_definitions() -> tuple[ToolDefinition, ...]:
    action_schema = cast(
        dict[str, object],
        cast(
            dict[str, object],
            cast(dict[str, object], _actions_schema()["properties"])["actions"],
        )["items"],
    )
    return (
        ToolDefinition(
            name="propose_cleanup_action",
            description="Propose one safe context cleanup action.",
            parameters=action_schema,
        ),
    )


def _scan_messages(
    *,
    save_id: str,
    chunk_index: int,
    messages: tuple[MessageRecord, ...],
) -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(
            role="system",
            body=LYRICS_INTERPRETATION_SECTION + "\n\n" + (
                "Scan this Bragi transcript chunk for context records that may be "
                "obsolete, contradicted, duplicate, stale, or safely over-specific. "
                "Return concise notes only through the enforced schema. Do not "
                "propose edits here."
            ),
        ),
        ChatMessage(
            role="user",
            body=f"Save: {save_id}\nChunk: {chunk_index}\n" + _messages_text(messages),
        ),
    )


def _scan_tool_messages(
    *,
    save_id: str,
    chunk_index: int,
    messages: tuple[MessageRecord, ...],
) -> tuple[ToolCallMessage, ...]:
    return tuple(
        ToolCallMessage(
            role=message.role,
            body=message.body.replace(
                (
                    "Return concise notes only through the enforced schema. "
                    "Do not propose edits here."
                ),
                (
                    "Use the note_cleanup_candidate tool for concise notes. "
                    "Do not propose edits here. Make no tool calls when the "
                    "chunk contains no cleanup candidates."
                ),
            ),
            speaker_name=message.speaker_name,
        )
        for message in _scan_messages(
            save_id=save_id,
            chunk_index=chunk_index,
            messages=messages,
        )
    )


def _action_messages(
    *,
    target_batch: _CleanupTargetBatch,
    scan_notes: tuple[str, ...],
    messages: tuple[MessageRecord, ...],
    scan_note_limit: int,
    scan_note_char_limit: int,
    message_reference_limit: int,
) -> tuple[ChatMessage, ...]:
    notes = _scan_notes_text(
        scan_notes,
        limit=scan_note_limit,
        char_limit=scan_note_char_limit,
    )
    return (
        ChatMessage(
            role="system",
            body=(
                "Choose safe Bragi context cleanup actions using only the enforced "
                "schema. Allowed operations are update, archive, and entity_link "
                "delete. Do not delete messages, media, or scenario definitions. "
                "Characters marked protected must not be archived. "
                "Fields listed as locked(read-only) are player-locked read-only "
                "facts and must not be updated. "
                "Prefer no action when evidence is weak."
            ),
        ),
        ChatMessage(
            role="user",
            body="\n\n".join(
                (
                    target_batch.registry_text,
                    "Transcript scan notes:\n" + notes,
                    _message_id_reference(
                        messages,
                        limit=message_reference_limit,
                    ),
                )
            ),
        ),
    )


def _action_tool_messages(
    *,
    target_batch: _CleanupTargetBatch,
    scan_notes: tuple[str, ...],
    messages: tuple[MessageRecord, ...],
    scan_note_limit: int,
    scan_note_char_limit: int,
    message_reference_limit: int,
) -> tuple[ToolCallMessage, ...]:
    return tuple(
        ToolCallMessage(
            role=message.role,
            body=message.body.replace(
                "using only the enforced schema.",
                "using only the propose_cleanup_action tool.",
            ),
            speaker_name=message.speaker_name,
        )
        for message in _action_messages(
            target_batch=target_batch,
            scan_notes=scan_notes,
            messages=messages,
            scan_note_limit=scan_note_limit,
            scan_note_char_limit=scan_note_char_limit,
            message_reference_limit=message_reference_limit,
        )
    )


def _guided_action_messages(
    *,
    instruction: str,
    target_batch: _CleanupTargetBatch,
    messages: tuple[MessageRecord, ...],
    message_reference_limit: int,
) -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(
            role="system",
            body=(
                "Choose Bragi save-state cleanup actions for the player's explicit "
                "cleanup instruction using only the enforced schema. Allowed "
                "operations are update, archive, and entity_link delete. Only target "
                "records listed in the current cleanup targets may be changed. "
                "Do not propose transcript message edits, media edits, or scenario "
                "definition edits. Characters marked protected must not be archived. "
                "Fields listed as locked(read-only) are player-locked read-only "
                "facts and must not be updated. "
                "Prefer no action when the target is uncertain."
            ),
        ),
        ChatMessage(
            role="user",
            body="\n\n".join(
                (
                    "Player cleanup instruction:\n" + instruction,
                    target_batch.registry_text,
                    _message_id_reference(
                        messages,
                        limit=message_reference_limit,
                    ),
                )
            ),
        ),
    )


def _guided_action_tool_messages(
    *,
    instruction: str,
    target_batch: _CleanupTargetBatch,
    messages: tuple[MessageRecord, ...],
    message_reference_limit: int,
) -> tuple[ToolCallMessage, ...]:
    return tuple(
        ToolCallMessage(
            role=message.role,
            body=message.body.replace(
                "using only the enforced schema.",
                "using only the propose_cleanup_action tool.",
            ),
            speaker_name=message.speaker_name,
        )
        for message in _guided_action_messages(
            instruction=instruction,
            target_batch=target_batch,
            messages=messages,
            message_reference_limit=message_reference_limit,
        )
    )


async def _generate_cleanup_tool_items[T](
    *,
    repositories: PersistenceRepositories,
    provider: ToolCallProvider,
    request: ToolCallRequest,
    task: str,
    validator: Callable[
        [ProviderToolCall, dict[str, dict[str, object]]],
        tuple[bool, dict[str, str], T | None],
    ],
) -> tuple[T, ...]:
    messages = list(request.messages)
    tool_schemas = {tool.name: tool.parameters for tool in request.tools}
    items: list[T] = []
    accepted_keys: set[str] = set()
    last_errors: list[str] = []
    max_attempt_count = configured_max_attempts(repositories)

    for _turn in range(max_attempt_count):
        turn_request = budget_tool_call_request(
            repositories,
            replace(request, messages=tuple(messages)),
            task=task,
        )
        response = await provider.generate_tool_calls(turn_request)
        errors: list[str] = []
        tool_results: list[tuple[ProviderToolCall, dict[str, str]]] = []
        for call in response.tool_calls:
            accepted, result, item = validator(call, tool_schemas)
            if accepted:
                if item is not None:
                    key = _canonical_tool_call_item(item)
                    if key not in accepted_keys:
                        accepted_keys.add(key)
                        items.append(item)
                tool_results.append((call, accepted_tool_result()))
                continue
            errors.append(result["error"])
            tool_results.append((call, result))

        if not errors:
            return tuple(items)

        last_errors = errors
        append_tool_feedback_messages(
            messages,
            assistant_body=response.body,
            tool_calls=response.tool_calls,
            tool_results=tool_results,
        )

    raise ProviderError(
        category=ProviderErrorCategory.PROVIDER_ERROR,
        message=(
            "Context cleanup tool-call validation failed after feedback: "
            + "; ".join(last_errors)
        ),
    )


def _validate_scan_note_tool_call(
    call: ProviderToolCall,
    *,
    tool_schemas: dict[str, dict[str, object]],
) -> tuple[bool, dict[str, str], str | None]:
    schema = tool_schemas.get(call.name)
    if schema is None:
        return _invalid_cleanup_tool_call(f"Unknown tool name: {call.name}")
    arguments, parse_error = parse_tool_arguments_json(call.arguments_json)
    if parse_error is not None or arguments is None:
        return _invalid_cleanup_tool_call(
            parse_error or "Tool arguments must be a JSON object"
        )
    shape_error = validate_tool_arguments_shape(arguments, schema=schema)
    if shape_error is not None:
        return _invalid_cleanup_tool_call(shape_error)
    message_id = str(arguments.get("message_id", "")).strip()
    note = str(arguments.get("note", "")).strip()
    if not note:
        return _invalid_cleanup_tool_call("note must not be blank")
    return True, accepted_tool_result(), f"{message_id}: {note}"


def _validate_cleanup_action_tool_call(
    call: ProviderToolCall,
    *,
    tool_schemas: dict[str, dict[str, object]],
    allowed_evidence_message_ids: frozenset[str],
) -> tuple[bool, dict[str, str], CleanupAction | None]:
    schema = tool_schemas.get(call.name)
    if schema is None:
        return _invalid_cleanup_tool_call(f"Unknown tool name: {call.name}")
    arguments, parse_error = parse_tool_arguments_json(call.arguments_json)
    if parse_error is not None or arguments is None:
        return _invalid_cleanup_tool_call(
            parse_error or "Tool arguments must be a JSON object"
        )
    shape_error = validate_tool_arguments_shape(arguments, schema=schema)
    if shape_error is not None:
        return _invalid_cleanup_tool_call(shape_error)
    action = _action_from_data(arguments)
    evidence_error = _surfaced_evidence_error(
        action.evidence_message_ids,
        allowed_evidence_message_ids,
    )
    if evidence_error is not None:
        return _invalid_cleanup_tool_call(evidence_error)
    return True, accepted_tool_result(), action


def _invalid_cleanup_tool_call[T](
    error: str,
) -> tuple[bool, dict[str, str], T | None]:
    return False, invalid_tool_result(error), None


def _canonical_tool_call_item(item: object) -> str:
    if isinstance(item, CleanupAction):
        return json.dumps(
            {
                "operation": item.operation,
                "target_type": item.target_type,
                "target_id": item.target_id,
                "field_path": item.field_path,
                "value": item.value,
                "reason": item.reason,
                "confidence": item.confidence,
                "evidence_message_ids": item.evidence_message_ids,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    return str(item)


def _cleanup_target_batches(
    repositories: PersistenceRepositories,
    save_id: str,
    target_batch_size: int,
) -> tuple[_CleanupTargetBatch, ...]:
    lines = _registry_target_lines(repositories, save_id)
    if not lines:
        return (
            _CleanupTargetBatch(
                target_count=0,
                registry_text="Current cleanup targets: none",
            ),
        )
    if len(lines) <= target_batch_size:
        return (
            _CleanupTargetBatch(
                target_count=len(lines),
                registry_text="Current cleanup targets:\n"
                + "\n".join(line for _target_type, line in lines),
            ),
        )
    batches: list[_CleanupTargetBatch] = []
    for target_type in _TARGET_TYPE_ORDER:
        typed_lines = [line for line_type, line in lines if line_type == target_type]
        for index, chunk in enumerate(_chunks(tuple(typed_lines), target_batch_size)):
            batches.append(
                _CleanupTargetBatch(
                    target_count=len(chunk),
                    registry_text="Current cleanup targets "
                    f"({target_type} batch {index + 1}):\n" + "\n".join(chunk),
                )
            )
    return tuple(batches)


def _registry_target_lines(
    repositories: PersistenceRepositories,
    save_id: str,
) -> tuple[tuple[str, str], ...]:
    lines: list[tuple[str, str]] = []
    snapshot = repositories.get_scene_snapshot(save_id)
    if snapshot is not None:
        lines.append(
            (
                "scene_snapshot",
                "- scene_snapshot "
                f"{snapshot.id}: situation={snapshot.situation}; "
                f"objective={snapshot.objective}; "
                f"locked(read-only)={','.join(snapshot.locked_fields)}",
            )
        )
    for state in repositories.list_world_state(save_id):
        lines.append(
            (
                "world_state",
                "- world_state "
                f"{state.id}: key={state.key}; category={state.category}; "
                f"value={_dump_json(state.value)}",
            )
        )
    for memory in repositories.list_memories(save_id):
        lines.append(
            (
                "memory",
                f"- memory {memory.id}: {memory.body}; tags={','.join(memory.tags)}",
            )
        )
    for summary in repositories.list_summaries(save_id):
        lines.append(("summary", f"- summary {summary.id}: {summary.body}"))
    for location in repositories.list_locations(save_id):
        lines.append(
            (
                "location",
                "- location "
                f"{location.id}: {location.name}; status={location.status}; "
                f"locked(read-only)={','.join(location.locked_fields)}",
            )
        )
    for character in repositories.list_characters(save_id):
        lines.append(
            (
                "character",
                "- character "
                f"{character.id}: {character.name}; status={character.status}; "
                f"age={character.age}; "
                f"current_clothing={character.current_clothing}; "
                f"protected={str(character.protected_from_maintenance).lower()}; "
                f"locked(read-only)={','.join(character.locked_fields)}",
            )
        )
    for thread in repositories.list_active_threads(save_id):
        if not active_thread_is_prompt_visible(thread):
            continue
        lines.append(
            (
                "active_thread",
                "- active_thread "
                f"{thread.id}: {thread.title}; status={thread.status}; "
                f"locked(read-only)={','.join(thread.locked_fields)}",
            )
        )
    for link in repositories.list_entity_links(save_id):
        lines.append(
            (
                "entity_link",
                "- entity_link "
                f"{link.id}: {link.entity_type}:{link.entity_id} -> "
                f"{link.target_type}:{link.target_id}; relation={link.relation}",
            )
        )
    return tuple(lines)


def _messages_text(messages: tuple[MessageRecord, ...]) -> str:
    if not messages:
        return "Transcript messages: none"
    return "Transcript messages:\n" + "\n".join(
        f"- {message.id} [{message.role}] {message.body}" for message in messages
    )


def _message_id_reference(
    messages: tuple[MessageRecord, ...],
    *,
    limit: int,
) -> str:
    if not messages:
        return "Transcript message ids: none"
    if limit <= 0 or len(messages) <= limit:
        return "Transcript message ids: " + ", ".join(
            message.id for message in messages
        )
    visible = messages[-limit:]
    return (
        f"Transcript message ids (showing last {limit} of {len(messages)}): "
        + ", ".join(message.id for message in visible)
    )


def _message_reference_ids(
    messages: tuple[MessageRecord, ...],
    *,
    limit: int,
) -> frozenset[str]:
    if not messages:
        return frozenset()
    if limit <= 0 or len(messages) <= limit:
        visible = messages
    else:
        visible = messages[-limit:]
    return frozenset(message.id for message in visible)


def _allowed_action_evidence_message_ids(
    *,
    messages: tuple[MessageRecord, ...],
    scan_notes: tuple[str, ...],
    scan_note_limit: int,
    scan_note_char_limit: int,
    message_reference_limit: int,
) -> frozenset[str]:
    return _message_reference_ids(
        messages,
        limit=message_reference_limit,
    ) | _scan_note_message_ids(
        scan_notes,
        limit=scan_note_limit,
        char_limit=scan_note_char_limit,
    )


def _scan_note_message_ids(
    scan_notes: tuple[str, ...],
    *,
    limit: int,
    char_limit: int,
) -> frozenset[str]:
    visible_notes = scan_notes if limit <= 0 else scan_notes[-limit:]
    message_ids: set[str] = set()
    for note in visible_notes:
        message_id, separator, _note_body = _trim_note(note, char_limit).partition(":")
        if separator and message_id.strip():
            message_ids.add(message_id.strip())
    return frozenset(message_ids)


def _surfaced_evidence_error(
    evidence_message_ids: tuple[str, ...],
    allowed_evidence_message_ids: frozenset[str],
) -> str | None:
    for message_id in evidence_message_ids:
        if message_id not in allowed_evidence_message_ids:
            return (
                "Evidence message id was not surfaced to context cleanup: "
                f"{message_id}"
            )
    return None


def _scan_notes_text(
    scan_notes: tuple[str, ...],
    *,
    limit: int,
    char_limit: int,
) -> str:
    if not scan_notes:
        return "- none"
    visible_notes = scan_notes if limit <= 0 else scan_notes[-limit:]
    lines: list[str] = []
    omitted_count = len(scan_notes) - len(visible_notes)
    if omitted_count > 0:
        lines.append(f"- {omitted_count} older scan notes omitted")
    lines.extend(f"- {_trim_note(note, char_limit)}" for note in visible_notes)
    return "\n".join(lines)


def _trim_note(note: str, char_limit: int) -> str:
    normalized = " ".join(note.split())
    if len(normalized) <= char_limit:
        return normalized
    return normalized[: max(0, char_limit - 3)].rstrip() + "..."


def _message_chunks(
    messages: tuple[MessageRecord, ...],
    chunk_size: int,
) -> tuple[tuple[MessageRecord, ...], ...]:
    return _chunks(messages, chunk_size)


def _chunks[T](
    values: tuple[T, ...],
    chunk_size: int,
) -> tuple[tuple[T, ...], ...]:
    size = max(1, chunk_size)
    return tuple(
        values[index : index + size] for index in range(0, len(values), size)
    )


def _scan_notes_from_data(data: dict[str, object]) -> tuple[str, ...]:
    notes = data.get("notes", [])
    if not isinstance(notes, list):
        return ()
    result: list[str] = []
    for item in notes:
        if not isinstance(item, dict):
            continue
        message_id = item.get("message_id")
        note = item.get("note")
        if isinstance(message_id, str) and isinstance(note, str) and note.strip():
            result.append(f"{message_id}: {note.strip()}")
    return tuple(result)


def _normalized_instruction(value: str) -> str:
    return " ".join(value.split())


def _guided_reason(*, instruction: str, action_reason: str) -> str:
    reason = action_reason.strip()
    if reason:
        return f"User instruction: {instruction}\nCleanup reason: {reason}"
    return f"User instruction: {instruction}"


def _known_message_ids(
    values: tuple[str, ...],
    message_ids: set[str],
) -> tuple[str, ...]:
    return tuple(message_id for message_id in values if message_id in message_ids)


def _actions_from_data(data: dict[str, object]) -> tuple[CleanupAction, ...]:
    actions = data.get("actions", [])
    if not isinstance(actions, list):
        return ()
    return tuple(
        _action_from_data(cast(dict[str, object], item))
        for item in actions
        if isinstance(item, dict)
    )


def _action_from_data(value: dict[str, object]) -> CleanupAction:
    return CleanupAction(
        operation=_string(value.get("operation")).strip(),
        target_type=_normalized_target_type(value.get("target_type")),
        target_id=_string(value.get("target_id")).strip(),
        field_path=_string(value.get("field_path")).strip(),
        value=value.get("value"),
        reason=_string(value.get("reason")).strip(),
        confidence=_confidence(value.get("confidence")),
        evidence_message_ids=_string_tuple(value.get("evidence_message_ids")),
    )


def _target_record(
    repositories: PersistenceRepositories,
    save_id: str,
    target_type: str,
    target_id: str,
) -> object | None:
    if target_type == "scene_snapshot":
        snapshot = repositories.get_scene_snapshot(save_id)
        return snapshot if snapshot is not None and snapshot.id == target_id else None
    if target_type == "location":
        location = repositories.get_location(target_id)
        return (
            location
            if location is not None and location.save_id == save_id
            else None
        )
    if target_type == "character":
        character = repositories.get_character(target_id)
        return (
            character
            if character is not None and character.save_id == save_id
            else None
        )
    if target_type == "active_thread":
        thread = repositories.get_active_thread(target_id)
        return thread if thread is not None and thread.save_id == save_id else None
    if target_type == "entity_link":
        return next(
            (
                link
                for link in repositories.list_entity_links(save_id)
                if link.id == target_id
            ),
            None,
        )
    if target_type == "world_state":
        return next(
            (
                record
                for record in repositories.list_world_state(save_id)
                if record.id == target_id
            ),
            None,
        )
    if target_type == "memory":
        return next(
            (
                record
                for record in repositories.list_memories(save_id)
                if record.id == target_id
            ),
            None,
        )
    if target_type == "summary":
        return next(
            (
                record
                for record in repositories.list_summaries(save_id)
                if record.id == target_id
            ),
            None,
        )
    return None


def _record_archive_locked(record: object) -> bool:
    locked_fields = set(getattr(record, "locked_fields", []))
    return bool(locked_fields & _ARCHIVE_LOCK_FIELDS)


def _target_field_is_locked(
    target_type: str,
    record: object,
    field_path: str,
) -> bool:
    locked_fields = getattr(record, "locked_fields", [])
    if target_type == "scene_snapshot":
        return scene_snapshot_field_is_locked(locked_fields, field_path)
    if target_type == "character":
        return character_field_is_locked(locked_fields, field_path)
    return field_path in set(locked_fields)


def _protected_character_archive_error(
    target_type: str,
    target: object,
) -> str | None:
    if (
        target_type == "character"
        and isinstance(target, CharacterRecord)
        and target.protected_from_maintenance
    ):
        return "Character is protected from maintenance and cannot be archived"
    return None


def _high_value_continuity_target(target_type: str, target: object) -> bool:
    if target_type == "character":
        return True
    if target_type == "active_thread":
        if not isinstance(target, ActiveThreadRecord):
            return False
        return active_thread_is_prompt_visible(target)
    if target_type == "memory":
        body = str(getattr(target, "body", "")).casefold()
        tags = {str(tag).casefold() for tag in getattr(target, "tags", [])}
        return bool(
            tags
            & {
                "identity",
                "inventory",
                "object",
                "promise",
                "relationship",
                "voice",
            }
        ) or any(
            term in body
            for term in (
                "promised",
                "relationship",
                "voice",
                "inventory",
                "knows",
                "swore",
            )
        )
    if target_type == "world_state":
        key = str(getattr(target, "key", "")).casefold()
        category = str(getattr(target, "category", "")).casefold()
        return any(
            term in key or term in category
            for term in (
                "identity",
                "inventory",
                "location",
                "object",
                "promise",
                "relationship",
                "voice",
            )
        )
    if target_type == "summary":
        body = str(getattr(target, "body", "")).casefold()
        return any(
            term in body
            for term in ("promised", "relationship", "voice", "inventory", "swore")
        )
    return False


def _has_explicit_contradiction_evidence(action: CleanupAction) -> bool:
    reason = action.reason.casefold()
    negated_markers = (
        "does not contradict",
        "doesn't contradict",
        "no contradiction",
        "not contradicted",
        "without contradiction",
    )
    if any(marker in reason for marker in negated_markers):
        return False
    return any(
        term in reason
        for term in (
            "contradict",
            "explicitly superseded",
            "explicitly resolved",
            "no longer true",
        )
    )


def _delete_related_entity_links(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    entity_type: str,
    entity_id: str,
) -> int:
    before = len(repositories.list_entity_links(save_id))
    repositories.delete_entity_links_for_endpoint(
        save_id=save_id,
        entity_type=entity_type,
        entity_id=entity_id,
    )
    after = len(repositories.list_entity_links(save_id))
    return max(0, before - after)


def _action_before_value(target: object, action: CleanupAction) -> object | None:
    if action.operation in {"archive", "delete"}:
        if isinstance(target, WorldStateRecord):
            return target.value
        if isinstance(target, EntityLinkRecord):
            return {
                "entity_type": target.entity_type,
                "entity_id": target.entity_id,
                "target_type": target.target_type,
                "target_id": target.target_id,
                "relation": target.relation,
            }
        return _record_json(target)
    return cast(object | None, getattr(target, action.field_path))


def _record_json(record: object) -> dict[str, object]:
    values: dict[str, object] = {}
    for key, value in vars(record).items():
        if key in {"id", "save_id"}:
            continue
        values[key] = value
    return values


def _reference_validation_error(
    repositories: PersistenceRepositories,
    save_id: str,
    target_type: str,
    field_path: str,
    value: object | None,
) -> str | None:
    if target_type == "scene_snapshot" and field_path == "current_location_id":
        return _location_reference_error(repositories, save_id, value)
    if target_type == "scene_snapshot" and field_path == "present_character_ids":
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            return "Scene present_character_ids must be a string list"
        for character_id in value:
            character = repositories.get_character(character_id)
            if character is None or character.save_id != save_id:
                return "Scene present_character_ids must belong to the active save"
    if target_type == "location" and field_path == "parent_location_id":
        return _location_reference_error(repositories, save_id, value)
    if target_type == "character" and field_path == "location_id":
        return _location_reference_error(repositories, save_id, value)
    return None


def _field_value_error(
    target_type: str,
    field_path: str,
    value: object | None,
) -> str | None:
    if target_type == "world_state" and field_path == "value":
        if isinstance(value, dict):
            return None
        return "World-state value must be an object"
    if field_path in {"current_location_id", "parent_location_id", "location_id"}:
        if isinstance(value, str) or value is None:
            return None
        return f"{field_path} must be a string id or null"
    if target_type == "memory" and field_path == "tags":
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return None
        return "Memory tags must be a string list"
    if target_type == "memory" and field_path == "importance":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if 0 <= float(value) <= 1:
                return None
            return "Memory importance must be between 0 and 1"
        return "Memory importance must be a number"
    if target_type == "character" and field_path == "met":
        if isinstance(value, bool):
            return None
        return "Character met must be a boolean"
    if target_type == "character" and field_path == "relationships":
        if isinstance(value, dict):
            return None
        return "Character relationships must be an object"
    if target_type == "active_thread" and field_path == "priority":
        if isinstance(value, int) and not isinstance(value, bool):
            return None
        return "Active-thread priority must be an integer"
    if field_path in {
        "aliases",
        "connections",
        "hazards",
        "nearby_objects",
        "present_character_ids",
        "related_entities",
    }:
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return None
        return f"{field_path} must be a string list"
    if isinstance(value, str):
        return None
    return f"{field_path} must be text"


def _location_reference_error(
    repositories: PersistenceRepositories,
    save_id: str,
    value: object | None,
) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        return "Location reference must be a string id"
    location = repositories.get_location(value)
    if location is None or location.save_id != save_id:
        return "Location reference must belong to the active save"
    return None


def _replace_scene_snapshot_field(
    snapshot: SceneSnapshotRecord,
    field_path: str,
    value: object | None,
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


def _replace_location_field(
    location: LocationRecord,
    field_path: str,
    value: object | None,
) -> LocationRecord:
    if field_path == "name":
        return replace(location, name=cast(str, value))
    if field_path == "description":
        return replace(location, description=cast(str, value))
    if field_path == "visual_description":
        return replace(location, visual_description=cast(str, value))
    if field_path == "status":
        return replace(location, status=cast(str, value))
    if field_path == "aliases":
        return replace(location, aliases=cast(list[str], value))
    if field_path == "connections":
        return replace(location, connections=cast(list[str], value))
    if field_path == "hazards":
        return replace(location, hazards=cast(list[str], value))
    if field_path == "parent_location_id":
        return replace(location, parent_location_id=cast(str | None, value))
    raise ValueError(f"Unsupported location field: {field_path}")


def _replace_character_field(
    character: CharacterRecord,
    field_path: str,
    value: object | None,
) -> CharacterRecord:
    if field_path == "name":
        return replace(character, name=cast(str, value))
    if field_path == "role":
        return replace(character, role=cast(str, value))
    if field_path == "age":
        return replace(character, age=cast(str, value))
    if field_path == "known_state":
        return replace(character, known_state=cast(str, value))
    if field_path == "appearance":
        return replace(character, appearance=cast(str, value))
    if field_path == "visual_notes":
        return replace(character, visual_notes=cast(str, value))
    if field_path == "current_clothing":
        return replace(character, current_clothing=cast(str, value))
    if field_path == "personality":
        return replace(character, personality=cast(str, value))
    if field_path == "voice":
        return replace(character, voice=cast(str, value))
    if field_path == "goals":
        return replace(character, goals=cast(str, value))
    if field_path == "motivations":
        return replace(character, motivations=cast(str, value))
    if field_path == "current_intent":
        return replace(character, current_intent=cast(str, value))
    if field_path == "boundaries":
        return replace(character, boundaries=cast(str, value))
    if field_path == "attitude_toward_player":
        return replace(character, attitude_toward_player=cast(str, value))
    if field_path == "cooperation_conditions":
        return replace(character, cooperation_conditions=cast(str, value))
    if field_path == "status":
        return replace(character, status=cast(str, value))
    if field_path == "private_notes":
        return replace(character, private_notes=cast(str, value))
    if field_path == "aliases":
        return replace(character, aliases=cast(list[str], value))
    if field_path == "relationships":
        return replace(character, relationships=cast(dict[str, object], value))
    if field_path == "location_id":
        return replace(character, location_id=cast(str | None, value))
    if field_path == "met":
        return replace(character, met=cast(bool, value))
    raise ValueError(f"Unsupported character field: {field_path}")


def _replace_thread_field(
    thread: ActiveThreadRecord,
    field_path: str,
    value: object | None,
) -> ActiveThreadRecord:
    if field_path == "title":
        return replace(thread, title=cast(str, value))
    if field_path == "description":
        return replace(thread, description=cast(str, value))
    if field_path == "status":
        return replace(thread, status=cast(str, value))
    if field_path == "visibility":
        return replace(thread, visibility=cast(str, value))
    if field_path == "related_entities":
        return replace(thread, related_entities=cast(list[str], value))
    if field_path == "priority":
        return replace(thread, priority=cast(int, value))
    raise ValueError(f"Unsupported active thread field: {field_path}")


def _normalized_target_type(value: object) -> str:
    text = _string(value).strip().casefold()
    if text in {"state", "world_state"}:
        return "world_state"
    if text in {"thread", "active_thread"}:
        return "active_thread"
    if text in {"link", "entity_link"}:
        return "entity_link"
    return text


def _string(value: object) -> str:
    return value if isinstance(value, str) else ""


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _confidence(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return 0.0
    return min(max(float(value), 0.0), 1.0)


def _dump_json(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def _elapsed_ms(started_at: float) -> int:
    return int((perf_counter() - started_at) * 1000)
