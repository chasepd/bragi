"""Deterministic world-state and memory extraction application."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Protocol, cast

from bragi.app_logging import exception_log_fields, log_error_event, log_event
from bragi.persistence.models import (
    MemoryRecord,
    MessageRecord,
    ScenarioRecord,
    StateChangeRecord,
    WorldStateRecord,
)
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatMessage,
    ProviderClient,
    ProviderToolCall,
    StructuredOutputProvider,
    StructuredOutputRequest,
    ToolCallMessage,
    ToolCallProvider,
    ToolCallRequest,
    ToolDefinition,
)
from bragi.providers.errors import (
    ProviderError,
    ProviderErrorCategory,
    provider_error_is_model_not_found,
)
from bragi.providers.structured_schema import normalize_strict_json_schema
from bragi.redaction import redact_text
from bragi.retry_policy import MODEL_OUTPUT_MAX_ATTEMPTS, configured_max_attempts
from bragi.services.job_lifecycle import JobLifecycleService
from bragi.services.manual_confirmation import (
    manual_memory_confirmation_enabled,
    manual_state_change_confirmation_enabled,
)
from bragi.services.message_correction import (
    MessageCorrectionContext,
    correction_context_text,
)
from bragi.services.open_threads import (
    archive_open_thread_aggregate_state,
    has_active_thread_records,
    is_open_threads_aggregate_key,
)
from bragi.services.openrouter_routing_settings import (
    request_with_openrouter_routing,
)
from bragi.services.post_turn_inference import memory_fingerprint
from bragi.services.prompt_inspection import PromptInspectionStore
from bragi.services.provider_fallbacks import (
    provider_error_with_fallback_attempted,
    provider_error_with_fallback_skipped_reason,
    recover_tool_call_shape_with_structured_output,
    shape_switch_diagnostics,
    structured_output_with_fallback,
    tool_call_fallback_request,
    tool_call_fallback_skip_reason,
)
from bragi.services.request_budget import (
    budget_structured_output_request,
    budget_tool_call_request,
)
from bragi.services.sexual_content_safety import is_fade_to_black_message
from bragi.services.state_preservation import preserve_replaced_world_state_memory
from bragi.services.text_script_policy import (
    DEFAULT_SCRIPT_GUARD_MODE,
    allowed_generated_scripts,
    object_text_script_violations,
    script_guard_mode,
    summarize_script_policy_violations,
)
from bragi.services.tool_call_helpers import (
    accepted_tool_result,
    invalid_tool_result,
    parse_tool_arguments_json,
    validate_tool_arguments_shape,
)

MAX_STATE_TOOL_FEEDBACK_TURNS = MODEL_OUTPUT_MAX_ATTEMPTS - 1
_FORMAT_NORMALIZED_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
    }
)
_MARKDOWN_QUOTE_MARKERS = ("*", "`")


@dataclass(frozen=True)
class ExtractedStateChange:
    operation: str
    key: str
    value: dict[str, object]
    category: str
    confidence: float
    source_message_id: str
    evidence_quote: str = ""
    persistence_scope: str = ""


@dataclass(frozen=True)
class ExtractedMemory:
    body: str
    tags: tuple[str, ...]
    importance: float
    source_message_id: str
    evidence_quote: str = ""


@dataclass(frozen=True)
class ExtractedStateConflict:
    key: str
    source_message_id: str
    new_evidence: str
    current_value: dict[str, object] | None = None
    proposed_value: dict[str, object] | None = None
    reason: str = ""
    confidence: float = 1.0


@dataclass(frozen=True)
class StateExtraction:
    state_changes: tuple[ExtractedStateChange, ...] = ()
    memories: tuple[ExtractedMemory, ...] = ()
    conflicts: tuple[ExtractedStateConflict, ...] = ()
    tool_diagnostics: dict[str, object] = field(
        default_factory=dict,
        compare=False,
    )


@dataclass(frozen=True)
class StateExtractionRequest:
    save_id: str
    messages: tuple[MessageRecord, ...]
    current_state: tuple[WorldStateRecord, ...]
    scenario_type: str = ""
    scenario_context: str = ""
    correction_context: MessageCorrectionContext | None = None
    include_memories: bool = True


class StateExtractor(Protocol):
    async def extract(self, request: StateExtractionRequest) -> StateExtraction:
        ...


STATE_EXTRACTION_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "state_changes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "operation": {"type": "string", "enum": ["upsert", "delete"]},
                    "key": {"type": "string"},
                    "value": {"type": "object"},
                    "category": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "source_message_id": {"type": "string"},
                    "evidence_quote": {"type": "string"},
                    "persistence_scope": {
                        "type": "string",
                        "enum": ["durable", "scene", "ephemeral"],
                    },
                },
                "required": [
                    "operation",
                    "key",
                    "value",
                    "category",
                    "confidence",
                    "source_message_id",
                    "evidence_quote",
                ],
            },
        },
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "body": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "importance": {"type": "number", "minimum": 0, "maximum": 1},
                    "source_message_id": {"type": "string"},
                    "evidence_quote": {"type": "string"},
                },
                "required": [
                    "body",
                    "tags",
                    "importance",
                    "source_message_id",
                    "evidence_quote",
                ],
            },
        },
        "conflicts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "key": {"type": "string"},
                    "source_message_id": {"type": "string"},
                    "new_evidence": {"type": "string"},
                    "current_value": {"type": "object"},
                    "proposed_value": {"type": "object"},
                    "reason": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "key",
                    "source_message_id",
                    "new_evidence",
                ],
            },
        },
    },
    "required": ["state_changes", "memories", "conflicts"],
}


class StructuredProviderStateExtractor:
    def __init__(
        self,
        *,
        provider: StructuredOutputProvider,
        provider_name: str,
        model_id: str,
        repositories: PersistenceRepositories | None = None,
        providers: dict[str, ProviderClient] | None = None,
    ) -> None:
        self.provider = provider
        self.provider_name = provider_name
        self.model_id = model_id
        self.repositories = repositories
        self.providers = providers

    async def extract(self, request: StateExtractionRequest) -> StateExtraction:
        structured_request = request_with_openrouter_routing(
            self.repositories,
            StructuredOutputRequest(
                provider=self.provider_name,
                model_id=self.model_id,
                schema_name="state_memory_extraction",
                schema=_state_extraction_schema(request),
                messages=_state_extraction_messages(request),
                temperature=0.0,
            ),
            task="state_memory",
            save_id=request.save_id,
        )
        if self.repositories is not None and self.providers is not None:
            response = await structured_output_with_fallback(
                repositories=self.repositories,
                providers=self.providers,
                request=structured_request,
                task="state_memory",
                save_id=request.save_id,
            )
        else:
            response = await self.provider.generate_structured_output(
                budget_structured_output_request(
                    self.repositories,
                    structured_request,
                    task="state_memory",
                )
            )
        extraction = _state_extraction_from_structured_data(
            response.data,
            include_memories=request.include_memories,
        )
        _validate_structured_extraction_grounding(
            extraction,
            source_messages=request.messages,
        )
        return extraction


@dataclass(frozen=True)
class _ValidatedStateToolCall:
    arguments: dict[str, object]
    extraction: object


class ToolCallingProviderStateExtractor:
    def __init__(
        self,
        *,
        provider: ToolCallProvider,
        provider_name: str,
        model_id: str,
        repositories: PersistenceRepositories | None = None,
        providers: dict[str, ProviderClient] | None = None,
        prompt_inspection_store: PromptInspectionStore | None = None,
    ) -> None:
        self.provider = provider
        self.provider_name = provider_name
        self.model_id = model_id
        self.repositories = repositories
        self.providers = providers
        self.prompt_inspection_store = prompt_inspection_store

    async def extract(self, request: StateExtractionRequest) -> StateExtraction:
        tool_request = request_with_openrouter_routing(
            self.repositories,
            ToolCallRequest(
                provider=self.provider_name,
                model_id=self.model_id,
                messages=_state_extraction_tool_messages(request),
                tools=_state_extraction_tool_definitions(request),
                temperature=0.0,
            ),
            task="state_memory",
            save_id=request.save_id,
        )
        self._capture_tool_call_request(
            message_id=_prompt_inspection_message_id(request.messages),
            kind="state_memory_tool_calls",
            title=(
                "State and memory tool calls"
                if request.include_memories
                else "State tool calls"
            ),
            request=tool_request,
        )
        try:
            return await self._extract_with_provider(
                provider=self.provider,
                request=tool_request,
                source_messages=request.messages,
                current_state=request.current_state,
                script_guard_mode_value=(
                    script_guard_mode(self.repositories, save_id=request.save_id)
                    if self.repositories is not None
                    else DEFAULT_SCRIPT_GUARD_MODE
                ),
            )
        except ProviderError as exc:
            if self.repositories is None or self.providers is None:
                if provider_error_is_model_not_found(exc):
                    return await self._extract_via_structured_shape(request, error=exc)
                raise
            fallback_request = tool_call_fallback_request(
                repositories=self.repositories,
                providers=self.providers,
                request=tool_request,
                save_id=request.save_id,
            )
            if fallback_request is None:
                reason = tool_call_fallback_skip_reason(
                    repositories=self.repositories,
                    providers=self.providers,
                    save_id=request.save_id,
                )
                log_event(
                    "provider.tool_call_fallback_skipped",
                    provider=tool_request.provider,
                    model=tool_request.model_id,
                    task="state_memory",
                    reason=reason,
                )
                if provider_error_is_model_not_found(exc):
                    return await self._extract_via_structured_shape(request, error=exc)
                raise provider_error_with_fallback_skipped_reason(exc, reason) from exc
            fallback_provider = self.providers[fallback_request.provider]
            if not isinstance(fallback_provider, ToolCallProvider):
                reason = "fallback_provider_unavailable"
                log_event(
                    "provider.tool_call_fallback_skipped",
                    provider=tool_request.provider,
                    model=tool_request.model_id,
                    task="state_memory",
                    reason=reason,
                )
                if provider_error_is_model_not_found(exc):
                    return await self._extract_via_structured_shape(request, error=exc)
                raise provider_error_with_fallback_skipped_reason(exc, reason) from exc
            log_event(
                "provider.tool_call_fallback_started",
                provider=fallback_request.provider,
                model=fallback_request.model_id,
                task="state_memory",
            )
            try:
                fallback_extraction = await self._extract_with_provider(
                    provider=fallback_provider,
                    request=fallback_request,
                    source_messages=request.messages,
                    current_state=request.current_state,
                    fallback_used=True,
                    script_guard_mode_value=(
                        script_guard_mode(self.repositories, save_id=request.save_id)
                        if self.repositories is not None
                        else DEFAULT_SCRIPT_GUARD_MODE
                    ),
                )
            except ProviderError as fallback_exc:
                # Recover when either tool attempt ended with model_not_found:
                # the tool shape is unavailable regardless of which attempt
                # reported it. The recovery helper re-runs through the
                # structured-output route when handed a model_not_found error.
                if provider_error_is_model_not_found(
                    exc
                ) or provider_error_is_model_not_found(fallback_exc):
                    recovery_error = (
                        exc
                        if provider_error_is_model_not_found(exc)
                        else fallback_exc
                    )
                    return await self._extract_via_structured_shape(
                        request,
                        error=recovery_error,
                    )
                raise provider_error_with_fallback_attempted(
                    fallback_exc,
                    provider=fallback_request.provider,
                    model_id=fallback_request.model_id,
                ) from fallback_exc
            primary_diagnostics = _tool_diagnostics_from_exception(exc)
            if primary_diagnostics:
                return replace(
                    fallback_extraction,
                    tool_diagnostics=_merge_tool_diagnostics(
                        primary_diagnostics,
                        fallback_extraction.tool_diagnostics,
                        fallback_used=True,
                    ),
                )
            return fallback_extraction

    async def _extract_via_structured_shape(
        self,
        request: StateExtractionRequest,
        *,
        error: ProviderError,
    ) -> StateExtraction:
        structured_extractor = StructuredProviderStateExtractor(
            provider=cast(StructuredOutputProvider, self.provider),
            provider_name=self.provider_name,
            model_id=self.model_id,
            repositories=self.repositories,
            providers=self.providers,
        )

        async def structured_run() -> StateExtraction:
            if not isinstance(self.provider, StructuredOutputProvider):
                raise ValueError("State extraction provider lacks structured output")
            extraction = await structured_extractor.extract(request)
            return replace(
                extraction,
                tool_diagnostics=shape_switch_diagnostics(
                    provider=self.provider_name,
                    model_id=self.model_id,
                ),
            )

        return await recover_tool_call_shape_with_structured_output(
            error=error,
            task="state_memory",
            provider=self.provider_name,
            model_id=self.model_id,
            structured_run=structured_run,
        )

    def _capture_tool_call_request(
        self,
        *,
        message_id: str | None,
        kind: str,
        title: str,
        request: ToolCallRequest,
    ) -> None:
        if self.prompt_inspection_store is None or message_id is None:
            return
        self.prompt_inspection_store.capture_tool_call_request(
            message_id=message_id,
            kind=kind,
            title=title,
            request=request,
        )

    async def _extract_with_provider(
        self,
        *,
        provider: ToolCallProvider,
        request: ToolCallRequest,
        source_messages: tuple[MessageRecord, ...],
        current_state: tuple[WorldStateRecord, ...],
        fallback_used: bool = False,
        script_guard_mode_value: str = DEFAULT_SCRIPT_GUARD_MODE,
    ) -> StateExtraction:
        messages = list(request.messages)
        state_by_key = {record.key: record for record in current_state}
        source_messages_by_id = {message.id: message for message in source_messages}
        tool_schemas = {tool.name: tool.parameters for tool in request.tools}
        accepted_keys: set[tuple[str, str]] = set()
        state_changes: list[ExtractedStateChange] = []
        memories: list[ExtractedMemory] = []
        conflicts: list[ExtractedStateConflict] = []
        conflict_keys: set[str] = set()
        resolved_patch_keys: set[str] = set()
        unresolved_patch_calls_by_key: dict[str, list[ProviderToolCall]] = {}
        last_unsafe_partial_state_keys: set[str] = set()
        last_unknown_partial_state_key = False
        last_errors: list[str] = []
        diagnostics = _initial_tool_diagnostics(
            provider=request.provider,
            model_id=request.model_id,
            fallback_used=fallback_used,
        )
        max_attempt_count = configured_max_attempts(self.repositories)

        for turn in range(max_attempt_count):
            turn_request = budget_tool_call_request(
                self.repositories,
                replace(request, messages=tuple(messages)),
                task="state_memory",
            )
            response = await provider.generate_tool_calls(turn_request)
            raw_calls = [_tool_call_diagnostic(call) for call in response.tool_calls]
            errors: list[str] = []
            turn_unsafe_partial_state_keys: set[str] = set()
            turn_unknown_partial_state_key = False
            tool_results: list[tuple[ProviderToolCall, dict[str, str]]] = []
            validated_calls: list[tuple[ProviderToolCall, _ValidatedStateToolCall]] = []

            for call in response.tool_calls:
                accepted, result, validated = _validate_state_tool_call(
                    call,
                    tool_schemas=tool_schemas,
                    source_messages_by_id=source_messages_by_id,
                    state_by_key=state_by_key,
                    script_guard_mode_value=script_guard_mode_value,
                )
                if accepted:
                    validated_calls.append((call, validated))
                    continue
                errors.append(result["error"])
                unsafe_key = _state_tool_call_key(call)
                if unsafe_key is not None:
                    turn_unsafe_partial_state_keys.add(unsafe_key)
                elif _is_state_keyed_tool(call.name):
                    turn_unknown_partial_state_key = True
                _append_tool_diagnostic_call(
                    diagnostics,
                    "rejected_calls",
                    call,
                    error=result["error"],
                )
                tool_results.append((call, result))

            turn_conflict_keys = {
                validated.extraction.key
                for _call, validated in validated_calls
                if isinstance(validated.extraction, ExtractedStateConflict)
            }
            combined_conflict_keys = conflict_keys | turn_conflict_keys

            for call, validated in validated_calls:
                extracted = validated.extraction
                if (
                    isinstance(extracted, ExtractedStateChange)
                    and extracted.operation == "upsert"
                    and extracted.key in combined_conflict_keys
                    and not _has_resolution_quote(
                        validated.arguments,
                        source_messages_by_id=source_messages_by_id,
                    )
                ):
                    error = (
                        "state conflict must be flagged without applying a patch "
                        f"unless a resolution_quote is grounded: {extracted.key}"
                    )
                    errors.append(error)
                    turn_unsafe_partial_state_keys.add(extracted.key)
                    _append_tool_diagnostic_call(
                        diagnostics,
                        "rejected_calls",
                        call,
                        error=error,
                    )
                    tool_results.append((call, _invalid_tool_call(error)[1]))
                    continue

                key = (call.name, _canonical_tool_arguments(validated.arguments))
                if key not in accepted_keys:
                    accepted_keys.add(key)
                    if isinstance(extracted, ExtractedStateChange):
                        state_changes.append(extracted)
                        if _has_resolution_quote(
                            validated.arguments,
                            source_messages_by_id=source_messages_by_id,
                        ):
                            resolved_patch_keys.add(extracted.key)
                        else:
                            unresolved_patch_calls_by_key.setdefault(
                                extracted.key,
                                [],
                            ).append(call)
                    elif isinstance(extracted, ExtractedMemory):
                        memories.append(extracted)
                    elif isinstance(extracted, ExtractedStateConflict):
                        conflicts.append(extracted)
                        conflict_keys.add(extracted.key)
                        if extracted.key not in resolved_patch_keys:
                            state_changes[:] = [
                                change
                                for change in state_changes
                                if change.key != extracted.key
                            ]
                            for prior_call in unresolved_patch_calls_by_key.pop(
                                extracted.key,
                                [],
                            ):
                                _append_tool_diagnostic_call(
                                    diagnostics,
                                    "rejected_calls",
                                    prior_call,
                                    error=(
                                        "state conflict superseded a previously "
                                        f"accepted patch: {extracted.key}"
                                    ),
                                )
                    _append_tool_diagnostic_call(
                        diagnostics,
                        "accepted_calls",
                        call,
                    )
                tool_results.append((call, _accepted_tool_result()))

            _append_tool_diagnostic_turn(
                diagnostics,
                turn=turn,
                raw_calls=raw_calls,
                errors=errors,
            )
            if errors:
                log_event(
                    "state_memory.tool_call_validation_failed",
                    provider=request.provider,
                    model=request.model_id,
                    turn=turn,
                    error_count=len(errors),
                )
            else:
                log_event(
                    "state_memory.tool_call_validation_succeeded",
                    provider=request.provider,
                    model=request.model_id,
                    turn=turn,
                    accepted_call_count=len(response.tool_calls),
                    conflict_count=len(conflicts),
                )
                return StateExtraction(
                    state_changes=tuple(state_changes),
                    memories=tuple(memories),
                    conflicts=tuple(conflicts),
                    tool_diagnostics=_final_tool_diagnostics(diagnostics),
                )

            last_errors = errors
            last_unsafe_partial_state_keys = turn_unsafe_partial_state_keys
            last_unknown_partial_state_key = turn_unknown_partial_state_key
            diagnostics["retry_count"] = turn + 1
            messages.append(
                ToolCallMessage(
                    role="assistant",
                    body=response.body,
                    tool_calls=response.tool_calls,
                )
            )
            for call, result in tool_results:
                messages.append(
                    ToolCallMessage(
                        role="tool",
                        body=json.dumps(result, sort_keys=True),
                        tool_call_id=call.id,
                    )
                )

        partial_extraction = _partial_state_extraction(
            state_changes=tuple(state_changes),
            memories=tuple(memories),
            conflicts=tuple(conflicts),
            unsafe_state_keys=last_unsafe_partial_state_keys,
            unknown_state_key=last_unknown_partial_state_key,
            diagnostics=diagnostics,
            last_errors=last_errors,
        )
        if partial_extraction is not None:
            return partial_extraction

        exc = ProviderError(
            category=ProviderErrorCategory.PROVIDER_ERROR,
            message=(
                "State/memory tool-call validation failed after feedback: "
                + "; ".join(last_errors)
            ),
        )
        _attach_tool_diagnostics(exc, _final_tool_diagnostics(diagnostics))
        raise exc


@dataclass(frozen=True)
class AppliedExtraction:
    world_state: tuple[WorldStateRecord, ...]
    memories: tuple[MemoryRecord, ...]
    state_changes: tuple[StateChangeRecord, ...]
    suppressed_memory_count: int = 0
    suppressed_state_change_count: int = 0


class StateService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        extractor: StateExtractor,
    ) -> None:
        self.repositories = repositories
        self.extractor = extractor
        self.jobs = JobLifecycleService(repositories=repositories)

    async def extract_and_apply_turn(
        self,
        *,
        save_id: str,
        source_message_ids: tuple[str, ...],
        include_memories: bool = True,
        suppressed_memory_fingerprints: frozenset[str] = frozenset(),
        suppressed_state_keys: frozenset[str] = frozenset(),
        source_scoped_only: bool = False,
    ) -> AppliedExtraction:
        job = self.jobs.create_running(
            save_id=save_id,
            type="state_extraction",
            payload={
                "source_message_ids": list(source_message_ids),
                "include_memories": include_memories,
                **(
                    {
                        "suppressed_memory_count": len(
                            suppressed_memory_fingerprints
                        )
                    }
                    if suppressed_memory_fingerprints
                    else {}
                ),
                **(
                    {"suppressed_state_keys": sorted(suppressed_state_keys)}
                    if suppressed_state_keys
                    else {}
                ),
                **({"source_scoped_only": True} if source_scoped_only else {}),
            },
            collect_provider_diagnostics=True,
        )
        log_event(
            "job.running",
            job_id=job.id,
            job_type=job.type,
            save_id=save_id,
            source_message_count=len(source_message_ids),
        )
        details = self.repositories.load_save_details(save_id)
        all_messages = (
            details.messages
            if details is not None
            else self.repositories.list_messages(save_id)
        )
        scenario = details.scenario if details is not None else None
        messages = tuple(
            message
            for message in all_messages
            if message.id in set(source_message_ids)
        )
        safety_transition_source_ids = _safety_transition_source_ids(messages)
        request = StateExtractionRequest(
            save_id=save_id,
            messages=messages,
            current_state=tuple(self.repositories.list_world_state(save_id)),
            scenario_type=scenario.type if scenario is not None else "",
            scenario_context=_scenario_context_text(scenario),
            include_memories=include_memories,
        )
        started_at = perf_counter()
        try:
            extraction = await self.extractor.extract(request)
            self.repositories.begin_transaction()
            applied = self.apply_extraction(
                save_id=save_id,
                extraction=extraction,
                allowed_source_message_ids=tuple(message.id for message in messages),
                safety_transition_source_message_ids=safety_transition_source_ids,
                suppressed_memory_fingerprints=suppressed_memory_fingerprints,
                suppressed_state_keys=suppressed_state_keys,
                source_scoped_only=source_scoped_only,
            )
            self.repositories.commit_transaction()
        except Exception as exc:
            self.repositories.rollback_transaction()
            failure_result = _failure_tool_diagnostics(exc)
            self.jobs.fail(
                job.id,
                error=redact_text(str(exc)) or exc.__class__.__name__,
                result=failure_result,
            )
            log_error_event(
                "job.failed",
                job_id=job.id,
                job_type=job.type,
                save_id=save_id,
                duration_ms=_elapsed_ms(started_at),
                **exception_log_fields(exc),
            )
            raise
        result: dict[str, object] = {
            "state_change_count": len(applied.state_changes),
            "memory_count": len(applied.memories),
            "conflict_count": len(extraction.conflicts),
        }
        if applied.suppressed_memory_count:
            result["suppressed_memory_count"] = applied.suppressed_memory_count
        if applied.suppressed_state_change_count:
            result["suppressed_state_change_count"] = (
                applied.suppressed_state_change_count
            )
        if extraction.tool_diagnostics:
            result["tool_diagnostics"] = extraction.tool_diagnostics
        self.jobs.succeed(job.id, result=result)
        log_event(
            "job.succeeded",
            job_id=job.id,
            job_type=job.type,
            save_id=save_id,
            duration_ms=_elapsed_ms(started_at),
            state_change_count=len(applied.state_changes),
            memory_count=len(applied.memories),
        )
        return applied

    async def extract_and_apply_message_correction(
        self,
        *,
        save_id: str,
        source_message_id: str,
        correction_context: MessageCorrectionContext,
    ) -> AppliedExtraction:
        details = self.repositories.load_save_details(save_id)
        all_messages = (
            details.messages
            if details is not None
            else self.repositories.list_messages(save_id)
        )
        scenario = details.scenario if details is not None else None
        messages = tuple(
            message for message in all_messages if message.id == source_message_id
        )
        request = StateExtractionRequest(
            save_id=save_id,
            messages=messages,
            current_state=tuple(self.repositories.list_world_state(save_id)),
            scenario_type=scenario.type if scenario is not None else "",
            scenario_context=_scenario_context_text(scenario),
            correction_context=correction_context,
        )
        extraction = await self.extractor.extract(request)
        self.repositories.begin_transaction()
        try:
            self._archive_message_correction_state(
                save_id=save_id,
                source_message_id=source_message_id,
            )
            applied = self.apply_extraction(
                save_id=save_id,
                extraction=extraction,
                allowed_source_message_ids=tuple(message.id for message in messages),
            )
            self.repositories.commit_transaction()
        except Exception:
            self.repositories.rollback_transaction()
            raise
        return applied

    def apply_extraction(
        self,
        *,
        save_id: str,
        extraction: StateExtraction,
        allowed_source_message_ids: tuple[str, ...] | None = None,
        suppressed_memory_fingerprints: frozenset[str] = frozenset(),
        suppressed_state_keys: frozenset[str] = frozenset(),
        safety_transition_source_message_ids: frozenset[str] = frozenset(),
        source_scoped_only: bool = False,
    ) -> AppliedExtraction:
        world_state_records: list[WorldStateRecord] = []
        memory_records: list[MemoryRecord] = []
        state_change_records: list[StateChangeRecord] = []
        suppressed_state_change_count = 0
        suppressed_memory_count = 0
        _validate_extraction(
            extraction,
            allowed_source_message_ids=allowed_source_message_ids,
        )
        if not safety_transition_source_message_ids and allowed_source_message_ids:
            safety_transition_source_message_ids = _safety_transition_source_ids(
                self.repositories.list_messages(save_id),
            ).intersection(allowed_source_message_ids)
        extraction = replace(
            extraction,
            conflicts=tuple(
                conflict
                for conflict in extraction.conflicts
                if conflict.source_message_id
                not in safety_transition_source_message_ids
            ),
        )
        mode = script_guard_mode(self.repositories, save_id=save_id)
        source_texts_by_id = _source_texts_by_id_for_script_policy(
            self.repositories,
            save_id=save_id,
            source_message_ids=allowed_source_message_ids,
        )
        confirm_state_changes = manual_state_change_confirmation_enabled(
            self.repositories,
            save_id=save_id,
        )
        confirm_memories = manual_memory_confirmation_enabled(
            self.repositories,
            save_id=save_id,
        )

        for change in extraction.state_changes:
            if change.source_message_id in safety_transition_source_message_ids:
                suppressed_state_change_count += 1
                continue
            if _state_change_script_policy_violations(
                change,
                source_texts_by_id=source_texts_by_id,
                mode=mode,
            ):
                suppressed_state_change_count += 1
                continue
            if change.key in suppressed_state_keys:
                suppressed_state_change_count += 1
                continue
            if _state_change_scope(change) == "ephemeral":
                continue
            if (
                change.key == "loop.current"
                and change.operation in {"remove", "delete"}
            ):
                # The typed loop envelope is policy-owned.  Extraction may
                # refresh its summary, but cannot delete its clock/baseline.
                suppressed_state_change_count += 1
                continue
            if _should_skip_state_change(
                repositories=self.repositories,
                save_id=save_id,
                change=change,
            ):
                continue
            before = (
                None
                if source_scoped_only
                else _find_world_state(
                    self.repositories.list_world_state(save_id),
                    change.key,
                )
            )
            if confirm_state_changes and not source_scoped_only:
                self._queue_state_change_confirmation(
                    save_id=save_id,
                    change=change,
                    before=before,
                )
                continue
            after_json = None
            if change.operation == "upsert":
                value = change.value
                if change.key == "loop.current" and before is not None:
                    value = _merge_loop_current_summary(before.value, change.value)
                if not source_scoped_only:
                    preserve_replaced_world_state_memory(
                        repositories=self.repositories,
                        save_id=save_id,
                        before=before,
                        after_value=value,
                        source_message_id=change.source_message_id,
                    )
                    world_state = self.repositories.upsert_world_state(
                        save_id=save_id,
                        key=change.key,
                        value=value,
                        category=change.category,
                        confidence=change.confidence,
                        source_message_id=change.source_message_id,
                    )
                    world_state_records.append(world_state)
                after_json = _dump(value)
            elif change.operation in {"remove", "delete"}:
                if not source_scoped_only:
                    preserve_replaced_world_state_memory(
                        repositories=self.repositories,
                        save_id=save_id,
                        before=before,
                        after_value=None,
                        source_message_id=change.source_message_id,
                    )
                    self.repositories.archive_world_state(
                        save_id=save_id,
                        key=change.key,
                    )
            if (
                change.operation == "upsert"
                and before is not None
                and _state_change_scope(change) == "scene"
                and before.value == value
            ):
                continue
            state_change = self.repositories.add_state_change(
                save_id=save_id,
                source_message_id=change.source_message_id,
                operation=change.operation,
                state_key=change.key,
                before_json=_dump(before.value) if before else None,
                after_json=after_json,
            )
            state_change_records.append(state_change)

        for memory in extraction.memories:
            if memory.source_message_id in safety_transition_source_message_ids:
                suppressed_memory_count += 1
                continue
            if _memory_script_policy_violations(
                memory,
                source_texts_by_id=source_texts_by_id,
                mode=mode,
            ):
                suppressed_memory_count += 1
                continue
            if memory_fingerprint(memory.body) in suppressed_memory_fingerprints:
                suppressed_memory_count += 1
                continue
            if confirm_memories:
                self._queue_memory_confirmation(save_id=save_id, memory=memory)
                continue
            memory_records.append(
                self.repositories.add_memory(
                    save_id=save_id,
                    body=memory.body,
                    tags=list(memory.tags),
                    importance=memory.importance,
                    source_message_id=memory.source_message_id,
                )
            )

        return AppliedExtraction(
            world_state=tuple(world_state_records),
            memories=tuple(memory_records),
            state_changes=tuple(state_change_records),
            suppressed_memory_count=suppressed_memory_count,
            suppressed_state_change_count=suppressed_state_change_count,
        )

    def _archive_message_correction_state(
        self,
        *,
        save_id: str,
        source_message_id: str,
    ) -> None:
        for state in self.repositories.list_world_state(save_id):
            if state.key == "loop.current":
                continue
            if state.source_message_id == source_message_id:
                self.repositories.archive_world_state(
                    save_id=save_id,
                    key=state.key,
                )
        for memory in self.repositories.list_memories(save_id):
            if source_message_id in memory.source_message_ids:
                self.repositories.archive_memory(memory.id)

    def _queue_state_change_confirmation(
        self,
        *,
        save_id: str,
        change: ExtractedStateChange,
        before: WorldStateRecord | None,
    ) -> None:
        proposed_value = {
            "operation": change.operation,
            "key": change.key,
            "value": deepcopy(change.value),
            "category": change.category,
            "confidence": change.confidence,
            "source_message_id": change.source_message_id,
        }
        suggestion = self.repositories.add_context_update_suggestion(
            save_id=save_id,
            update_type=change.operation,
            entity_type="world_state",
            entity_id=before.id if before is not None else None,
            field_path=change.key,
            proposed_value=proposed_value,
            status="pending",
            reason=f"Confirm {change.operation} for {change.key}",
            confidence=change.confidence,
            source_message_ids=[change.source_message_id],
        )
        self.repositories.add_context_update_audit(
            save_id=save_id,
            suggestion_id=suggestion.id,
            operation="queued",
            entity_type="world_state",
            entity_id=before.id if before is not None else None,
            field_path=change.key,
            before=before.value if before is not None else None,
            after=proposed_value,
            reason=f"Confirm {change.operation} for {change.key}",
            confidence=change.confidence,
            source_message_ids=[change.source_message_id],
        )

    def _queue_memory_confirmation(
        self,
        *,
        save_id: str,
        memory: ExtractedMemory,
    ) -> None:
        proposed_value = {
            "body": memory.body,
            "tags": list(memory.tags),
            "importance": memory.importance,
            "source_message_id": memory.source_message_id,
        }
        suggestion = self.repositories.add_context_update_suggestion(
            save_id=save_id,
            update_type="create",
            entity_type="memory",
            field_path="*",
            proposed_value=proposed_value,
            status="pending",
            reason="Confirm new memory",
            confidence=memory.importance,
            source_message_ids=[memory.source_message_id],
        )
        self.repositories.add_context_update_audit(
            save_id=save_id,
            suggestion_id=suggestion.id,
            operation="queued",
            entity_type="memory",
            entity_id=None,
            field_path="*",
            before=None,
            after=proposed_value,
            reason="Confirm new memory",
            confidence=memory.importance,
            source_message_ids=[memory.source_message_id],
        )


def _find_world_state(
    records: list[WorldStateRecord],
    key: str,
) -> WorldStateRecord | None:
    for record in records:
        if record.key == key:
            return record
    return None


def _merge_loop_current_summary(
    existing: dict[str, object],
    proposed: dict[str, object],
) -> dict[str, object]:
    """Allow extraction to refresh prose without replacing policy-owned time data."""
    summary = proposed.get("summary")
    if not isinstance(summary, str):
        return existing
    merged = dict(existing)
    merged["summary"] = summary
    return merged


def _state_change_scope(change: ExtractedStateChange) -> str:
    scope = change.persistence_scope.strip().casefold()
    if scope:
        return scope
    key = change.key.casefold()
    category = change.category.casefold()
    if category == "ephemeral":
        return "ephemeral"
    if key.endswith(".current_emotional_state"):
        return "scene"
    return "durable"


def _should_skip_state_change(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    change: ExtractedStateChange,
) -> bool:
    if change.operation != "upsert":
        return False
    if not change.value:
        return True
    if is_open_threads_aggregate_key(change.key) and has_active_thread_records(
        repositories,
        save_id
    ):
        archive_open_thread_aggregate_state(repositories, save_id)
        return True
    return False


def _dump(value: dict[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _validate_extraction(
    extraction: StateExtraction,
    *,
    allowed_source_message_ids: tuple[str, ...] | None = None,
) -> None:
    allowed_ids = set(allowed_source_message_ids or ())
    for change in extraction.state_changes:
        if change.operation not in {"upsert", "remove", "delete"}:
            raise ValueError(f"Unsupported state operation: {change.operation}")
        if not change.key:
            raise ValueError("State change key is required")
        if allowed_source_message_ids is not None and (
            change.source_message_id not in allowed_ids
        ):
            raise ValueError(
                f"Unknown state source_message_id: {change.source_message_id}"
            )
        _dump(change.value)
        if _state_change_scope(change) not in {"durable", "scene", "ephemeral"}:
            raise ValueError(
                f"Unsupported state persistence_scope: {change.persistence_scope}"
            )
    for memory in extraction.memories:
        if not memory.body:
            raise ValueError("Memory body is required")
        if allowed_source_message_ids is not None and (
            memory.source_message_id not in allowed_ids
        ):
            raise ValueError(
                f"Unknown memory source_message_id: {memory.source_message_id}"
            )
    for conflict in extraction.conflicts:
        if not conflict.key:
            raise ValueError("State conflict key is required")
        if not conflict.new_evidence:
            raise ValueError("State conflict new_evidence is required")
        if allowed_source_message_ids is not None and (
            conflict.source_message_id not in allowed_ids
        ):
            raise ValueError(
                f"Unknown state conflict source_message_id: "
                f"{conflict.source_message_id}"
            )


def _source_texts_by_id_for_script_policy(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    source_message_ids: tuple[str, ...] | None,
) -> dict[str, str]:
    allowed_ids = set(source_message_ids) if source_message_ids is not None else None
    return {
        message.id: message.body
        for message in repositories.list_messages(save_id)
        if allowed_ids is None or message.id in allowed_ids
    }


def _allowed_scripts_for_source_message_id(
    source_message_id: str,
    *,
    source_texts_by_id: dict[str, str],
    fallback_texts: tuple[str, ...] = (),
) -> frozenset[str]:
    source_text = source_texts_by_id.get(source_message_id)
    if source_text:
        return allowed_generated_scripts((source_text,))
    return allowed_generated_scripts(fallback_texts)


def _state_change_script_policy_violations(
    change: ExtractedStateChange,
    *,
    source_texts_by_id: dict[str, str],
    mode: str,
) -> bool:
    allowed_scripts = _allowed_scripts_for_source_message_id(
        change.source_message_id,
        source_texts_by_id=source_texts_by_id,
        fallback_texts=(change.evidence_quote,),
    )
    values: tuple[tuple[str, object], ...] = (
        ("state.key", change.key),
        ("state.category", change.category),
        ("state.value", change.value),
        ("state.evidence_quote", change.evidence_quote),
    )
    return any(
        object_text_script_violations(
            value,
            allowed_scripts=allowed_scripts,
            mode=mode,
            field_name=field_name,
        )
        for field_name, value in values
    )


def _memory_script_policy_violations(
    memory: ExtractedMemory,
    *,
    source_texts_by_id: dict[str, str],
    mode: str,
) -> bool:
    allowed_scripts = _allowed_scripts_for_source_message_id(
        memory.source_message_id,
        source_texts_by_id=source_texts_by_id,
        fallback_texts=(memory.evidence_quote,),
    )
    values: tuple[tuple[str, object], ...] = (
        ("memory.body", memory.body),
        ("memory.tags", memory.tags),
        ("memory.evidence_quote", memory.evidence_quote),
    )
    return any(
        object_text_script_violations(
            value,
            allowed_scripts=allowed_scripts,
            mode=mode,
            field_name=field_name,
        )
        for field_name, value in values
    )


def _elapsed_ms(started_at: float) -> int:
    return int((perf_counter() - started_at) * 1000)


def _validate_structured_extraction_grounding(
    extraction: StateExtraction,
    *,
    source_messages: tuple[MessageRecord, ...],
) -> None:
    source_messages_by_id = {message.id: message for message in source_messages}
    for change in extraction.state_changes:
        error = _validate_exact_quote(
            {
                "source_message_id": change.source_message_id,
                "evidence_quote": change.evidence_quote,
            },
            source_messages_by_id=source_messages_by_id,
            quote_field="evidence_quote",
        )
        if error is not None:
            raise ValueError(f"Structured state change grounding failed: {error}")
    for memory in extraction.memories:
        error = _validate_exact_quote(
            {
                "source_message_id": memory.source_message_id,
                "evidence_quote": memory.evidence_quote,
            },
            source_messages_by_id=source_messages_by_id,
            quote_field="evidence_quote",
        )
        if error is not None:
            raise ValueError(f"Structured memory grounding failed: {error}")
    for conflict in extraction.conflicts:
        error = _validate_exact_quote(
            {
                "source_message_id": conflict.source_message_id,
                "new_evidence": conflict.new_evidence,
            },
            source_messages_by_id=source_messages_by_id,
            quote_field="new_evidence",
        )
        if error is not None:
            raise ValueError(f"Structured state conflict grounding failed: {error}")


def _state_extraction_messages(
    request: StateExtractionRequest,
) -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(
            role="system",
            body=_state_extraction_instruction(
                request.scenario_type,
                include_memories=request.include_memories,
            ),
        ),
        ChatMessage(
            role="user",
            body="\n\n".join(
                item
                for item in (
                    request.scenario_context,
                    _current_state_text(request.current_state),
                    _messages_text(request.messages),
                    correction_context_text(request.correction_context),
                )
                if item
            ),
        ),
    )


def _state_extraction_tool_messages(
    request: StateExtractionRequest,
) -> tuple[ToolCallMessage, ...]:
    messages = _state_extraction_messages(request)
    tool_messages: list[ToolCallMessage] = []
    tool_instruction = (
        "Use the provided tools instead of prose. Prefer patch_world_state "
        "with minimal value_patch objects, record_memory_fact for durable "
        "memory facts, and flag_state_conflict when the completed turn "
        "contradicts current state without explicitly resolving it. Every "
        "normal fact or patch tool call must include source_message_id and "
        "evidence_quote copied exactly from that source message. Conflict "
        "calls must include source_message_id and new_evidence copied "
        "exactly from that source message. If evidence does not name a "
        "person, place, or backstory, use unknown or empty arrays instead "
        "of inventing details."
    )
    if not request.include_memories:
        tool_instruction = (
            "Use the provided tools instead of prose. Prefer patch_world_state "
            "with minimal value_patch objects and flag_state_conflict when the "
            "completed turn contradicts current state without explicitly "
            "resolving it. Every patch tool call must include source_message_id "
            "and evidence_quote copied exactly from that source message. "
            "Conflict calls must include source_message_id and new_evidence "
            "copied exactly from that source message. If evidence does not name "
            "a person, place, or backstory, use unknown or empty arrays instead "
            "of inventing details."
        )
    for message in messages:
        body = message.body.replace(
            "Use the enforced response schema.",
            tool_instruction,
        )
        tool_messages.append(
            ToolCallMessage(
                role=message.role,
                body=body,
                speaker_name=message.speaker_name,
            )
        )
    return tuple(tool_messages)


def _prompt_inspection_message_id(
    messages: tuple[MessageRecord, ...],
) -> str | None:
    for message in reversed(messages):
        if message.role != "player":
            return message.id
    return messages[-1].id if messages else None


def _state_extraction_instruction(
    scenario_type: str,
    *,
    include_memories: bool = True,
) -> str:
    target = (
        "durable Bragi world-state changes and memories"
        if include_memories
        else "durable Bragi world-state changes"
    )
    base = (
        f"Extract only {target} from the "
        "completed turn. Use the enforced response schema. Include only facts "
        "supported by the provided messages. Ignore flavor, transient motion, "
        "and anything already captured well enough by the current state. If a "
        "completed turn contradicts current state without explicitly resolving "
        "it, add a conflict for that key and do not add a state change for that "
        "same key. Every state change and memory must include evidence_quote "
        "copied exactly from the source message, and every conflict must include "
        "new_evidence copied exactly from the source message. A marked narrator "
        "safety transition is only the canonical off-screen event and elapsed "
        "time; never extract intimate detail or inferred physical facts from it."
        " Mark persistence_scope as durable for stable facts, scene for current "
        "scene status or current emotional state, and ephemeral for beat notes "
        "that should not become durable world state."
    )
    if scenario_type == "fantasy_roleplay":
        return (
            base
            + " This is a fantasy roleplay scenario, so prefer active scene "
            "status, places, quest objectives, magic rules or costs, faction "
            "pressure, mythic threats, durable NPC status, promises, constraints, "
            "inventory-like facts, and unresolved quest threads."
        )
    if scenario_type == "science_fiction_roleplay":
        return (
            base
            + " This is a science fiction roleplay scenario, so prefer active "
            "scene status, locations or vessels, mission objectives, technology "
            "constraints or failures, species or AI facts, institutional pressure, "
            "durable NPC status, equipment-like facts, and unresolved mission "
            "threads."
        )
    if scenario_type == "first_contact_exploration":
        return (
            base
            + " This is a first-contact or exploration science fiction scenario, "
            "so prefer active scene status, mission objectives, ship or base "
            "condition, crew morale or expertise, observed facts, hypotheses, "
            "misunderstandings, confirmed knowledge, unknowns, translation terms, "
            "false assumptions, confirmed meanings, alien or unknown-intelligence "
            "behavior, discoveries, samples, sensor findings, artifacts, "
            "contamination risks, environmental hazards, diplomatic tension, "
            "equipment damage, rescue windows, and mission deadlines. Prefer "
            "keys such as mission.objective, mission.constraints, ship.status, "
            "base.status, crew.<name>.status, crew.<name>.trust, "
            "site.<name>.observations, site.<name>.hazards, "
            "contact.<entity>.behaviors, contact.<entity>.relationship, "
            "translation.<signal>.hypotheses, "
            "translation.<signal>.confirmed_meanings, discovery.<item>.status, "
            "sample.<item>.contamination_risk, and escalation.<clock>."
        )
    if scenario_type == "survival_expedition":
        return (
            base
            + " This is a survival expedition scenario, so prefer expedition "
            "state: route progress, delays, detours, landmarks, retreat status, "
            "resource changes, equipment condition, injuries, illness, fatigue, "
            "morale, weather, exposure, terrain hazards, camp safety, and open "
            "survival threats. Prefer keys such as expedition.progress, "
            "expedition.route, expedition.resources, expedition.environment, "
            "expedition.hazards, expedition.camp, expedition.party, and "
            "character.<name>.survival_status."
        )
    if scenario_type == "time_loop":
        return (
            base
            + " This is a time loop scenario, so prefer explicit loop state: "
            "loop counter, current phase, reset trigger status, resettable "
            "baseline changes, schedule deviations, discoveries that persist "
            "as player/meta knowledge, persistence exceptions, NPC memory "
            "boundaries, and prior-loop summaries. Keep resettable world facts "
            "separate from persistent knowledge. Prefer keys such as loop.current, "
            "loop.schedule, loop.baseline, loop.knowledge, loop.persistence, "
            "loop.npc_memory, loop.rules, and character.<name>.loop_exception."
        )
    if scenario_type == "investigation_mystery":
        return (
            base
            + " This is an investigation mystery scenario, so prefer active "
            "scene status, discovered clues, known facts, suspect and witness "
            "state, public timeline changes, red herrings, deduction progress, "
            "case status, durable NPC status, and unresolved case threads."
        )
    if scenario_type == "heist_infiltration":
        return (
            base
            + " This is a heist or infiltration scenario, so prefer active job "
            "state: target status, objective progress, crew and contact status, "
            "intel quality, access credentials or covers, security layers, guards, "
            "locks, alarms, cameras, patrols, suspicion, alarm state, heat, loadout "
            "changes, complications, extraction route status, pursuit pressure, "
            "and aftermath consequences. Prefer keys such as heist.target, "
            "heist.objectives, heist.crew, heist.intel, heist.security, "
            "heist.alert, heist.loadout, heist.complications, heist.extraction, "
            "heist.aftermath, and character.<name>.heist_status."
        )
    if scenario_type == "political_intrigue":
        return (
            base
            + " This is a political intrigue scenario, so prefer social and "
            "strategic state: faction positions, resources, pressure points, "
            "key NPC loyalties, grudges, obligations, secrets, leverage, "
            "public reputation, faction standing, favors owed or held, bargains, "
            "promises, blackmail terms, alliances, rivalries, event calendars, "
            "timed votes or scandals, and public versus private knowledge. "
            "Prefer keys such as intrigue.factions, intrigue.npcs, "
            "intrigue.standing, intrigue.obligations, intrigue.alliances, "
            "intrigue.pressure, intrigue.calendar, intrigue.secrets, "
            "intrigue.knowledge, faction.<name>.standing, "
            "obligation.<party>.owed_to_<party>, and alliance.<name>."
        )
    if scenario_type == "settlement_builder":
        return (
            base
            + " This is a settlement builder scenario, so prefer community "
            "management state: settlement profile, residents, population groups, "
            "leaders, specialists, outside contacts, resources, morale, safety, "
            "prosperity, facilities, projects, project progress, blockers, "
            "benefits, threats, opportunities, seasonal state, deadlines, and "
            "external relationships. Prefer keys such as settlement.profile, "
            "settlement.population, settlement.resources, settlement.projects, "
            "settlement.facilities, settlement.pressures, settlement.calendar, "
            "settlement.relationships, project.<name>.status, "
            "resource.<name>, and faction.<name>.standing."
        )
    if scenario_type == "monster_hunt_bounty":
        return (
            base
            + " This is a monster hunt or bounty campaign scenario, so prefer "
            "target and investigation state: target status, abilities, "
            "weaknesses, habits, signs, territory, clues, leads, witnesses, "
            "sightings, lairs, rivals, patrons, authorities, preparation, gear, "
            "traps, debts, reward terms, and outcome state. Keep discovered "
            "evidence separate from hidden truth. Prefer keys such as "
            "hunt.profile, hunt.target, hunt.leads, hunt.locations, hunt.rivals, "
            "hunt.preparation, hunt.status, target.<name>.status, "
            "clue.<name>.status, and location.<name>.hunt_state."
        )
    if scenario_type == "road_trip_pilgrimage":
        return (
            base
            + " This is a road trip or pilgrimage scenario, so prefer journey "
            "state: route, stops, current leg, delays, detours, destination "
            "status, transport condition, supplies, money, documents, recurring "
            "pressures, pursuers, border or weather constraints, companion "
            "relationships, promises, shared memories, conflicts, and unresolved "
            "threads from prior stops. Prefer keys such as journey.profile, "
            "journey.route, journey.party, journey.supplies, journey.pressures, "
            "journey.relationships, journey.progress, stop.<name>.threads, "
            "companion.<name>.relationship, and vehicle.<name>.condition."
        )
    if scenario_type == "merchant_trade_route":
        return (
            base
            + " This is a merchant or trade route scenario, so prefer trade "
            "state: cargo, contracts, debts, deadlines, penalties, patrons, "
            "market demand and supply, route hazards, reputation, contacts, "
            "legal status, profit, loss, margins, price changes, delivery state, "
            "and obligation fulfillment. Keep accounting lightweight and "
            "explainable. Prefer keys such as trade.profile, trade.cargo, "
            "trade.markets, trade.contracts, trade.hazards, trade.reputation, "
            "trade.ledger, cargo.<item>.status, contract.<name>.status, "
            "debt.<party>.status, and market.<name>.conditions."
        )
    return (
        base
        + " This is a full roleplay scenario, so prefer active scene status, "
        "location, current objective, durable NPC status, inventory-like facts, "
        "promises, constraints, threats, and unresolved plot threads."
    )


def _state_extraction_tool_definitions(
    request: StateExtractionRequest,
) -> tuple[ToolDefinition, ...]:
    schemas = _state_extraction_tool_schemas(request)
    descriptions = {
        "patch_world_state": (
            "Patch one world-state key with a minimal supported value patch."
        ),
        "record_memory_fact": "Record one durable memory fact.",
        "flag_state_conflict": (
            "Flag a contradiction between the completed turn and current state "
            "without applying a resolving patch."
        ),
    }
    return tuple(
        ToolDefinition(
            name=name,
            description=descriptions[name],
            parameters=schema,
        )
        for name, schema in schemas.items()
    )


def _state_extraction_tool_schemas(
    request: StateExtractionRequest,
) -> dict[str, dict[str, object]]:
    source_schema: dict[str, object] = {"type": "string"}
    message_ids = [message.id for message in request.messages]
    if message_ids:
        source_schema["enum"] = message_ids
    base_grounding = {
        "source_message_id": source_schema,
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }
    schemas = {
        "patch_world_state": _tool_schema(
            required=[
                "operation",
                "key",
                "source_message_id",
                "evidence_quote",
            ],
            properties={
                **base_grounding,
                "operation": {"type": "string", "enum": ["upsert", "delete"]},
                "key": {"type": "string"},
                "value_patch": {"type": "object"},
                "category": {"type": "string"},
                "persistence_scope": {
                    "type": "string",
                    "enum": ["", "durable", "scene", "ephemeral"],
                },
                "evidence_quote": {"type": "string"},
                "resolution_quote": {
                    "type": "string",
                    "description": (
                        "Optional exact quote explaining how a contradiction was "
                        "resolved in-scene."
                    ),
                },
            },
        ),
        "record_memory_fact": _tool_schema(
            required=["body", "source_message_id", "evidence_quote"],
            properties={
                **base_grounding,
                "body": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "importance": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence_quote": {"type": "string"},
            },
        ),
        "flag_state_conflict": _tool_schema(
            required=["key", "source_message_id", "new_evidence"],
            properties={
                **base_grounding,
                "key": {"type": "string"},
                "current_value": {"type": "object"},
                "proposed_value": {"type": "object"},
                "new_evidence": {"type": "string"},
                "reason": {"type": "string"},
            },
        ),
    }
    if not request.include_memories:
        schemas.pop("record_memory_fact", None)
    return schemas


def _tool_schema(
    *,
    required: list[str],
    properties: dict[str, object],
) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _state_extraction_schema(
    request: StateExtractionRequest,
) -> dict[str, object]:
    schema = deepcopy(STATE_EXTRACTION_SCHEMA)
    properties = schema["properties"]
    if isinstance(properties, dict) and not request.include_memories:
        properties.pop("memories", None)
        required = schema.get("required")
        if isinstance(required, list):
            schema["required"] = [
                field for field in required if field != "memories"
            ]
    message_ids = [message.id for message in request.messages]
    if not message_ids:
        return normalize_strict_json_schema(schema)

    if not isinstance(properties, dict):
        return schema
    for collection_name in ("state_changes", "memories", "conflicts"):
        collection = properties.get(collection_name)
        if not isinstance(collection, dict):
            continue
        items = collection.get("items")
        if not isinstance(items, dict):
            continue
        item_properties = items.get("properties")
        if not isinstance(item_properties, dict):
            continue
        source_property = item_properties.get("source_message_id")
        if isinstance(source_property, dict):
            source_property["enum"] = message_ids
    return normalize_strict_json_schema(schema)


def _scenario_context_text(scenario: ScenarioRecord | None) -> str:
    if scenario is None:
        return "Scenario context: unavailable"
    lines = [
        "Scenario context:",
        f"- type: {scenario.type}",
        f"- title: {scenario.title}",
        f"- premise/setup: {scenario.premise}",
        f"- player role: {scenario.player_role}",
    ]
    for key, value in _scenario_content(scenario.content_json):
        if key in {
            "title",
            "premise",
            "setup_line",
            "player_character_name",
            "player_role",
            "starting_scene",
        }:
            continue
        if value:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _scenario_content(content_json: str) -> tuple[tuple[str, str], ...]:
    try:
        loaded = json.loads(content_json)
    except json.JSONDecodeError:
        return ()
    if not isinstance(loaded, dict):
        return ()
    return tuple((str(key), str(value)) for key, value in loaded.items())


def _current_state_text(records: tuple[WorldStateRecord, ...]) -> str:
    if not records:
        return "Current world state: none"
    return "Current world state:\n" + "\n".join(
        f"- {record.key}: {_dump(record.value)}" for record in records
    )


def _messages_text(messages: tuple[MessageRecord, ...]) -> str:
    if not messages:
        return "Completed turn messages: none"
    return "Completed turn messages:\n" + "\n".join(
        f"- {message.id} [{message.role}] {message.body}" for message in messages
    )


def _state_extraction_from_structured_data(
    data: dict[str, object],
    *,
    include_memories: bool = True,
) -> StateExtraction:
    raw_changes = data.get("state_changes", [])
    raw_memories = data.get("memories", []) if include_memories else []
    raw_conflicts = data.get("conflicts", [])
    if not isinstance(raw_changes, list):
        raise ValueError("Structured state extraction state_changes must be a list")
    if not isinstance(raw_memories, list):
        raise ValueError("Structured state extraction memories must be a list")
    if not isinstance(raw_conflicts, list):
        raise ValueError("Structured state extraction conflicts must be a list")
    conflicts = tuple(_state_conflict_from_data(item) for item in raw_conflicts)
    conflict_keys = {conflict.key for conflict in conflicts}
    return StateExtraction(
        state_changes=tuple(
            change
            for change in (_state_change_from_data(item) for item in raw_changes)
            if change.key not in conflict_keys
        ),
        memories=tuple(_memory_from_data(item) for item in raw_memories),
        conflicts=conflicts,
    )


def _validate_state_tool_call(
    call: ProviderToolCall,
    *,
    tool_schemas: dict[str, dict[str, object]],
    source_messages_by_id: dict[str, MessageRecord],
    state_by_key: dict[str, WorldStateRecord],
    script_guard_mode_value: str,
) -> tuple[bool, dict[str, str], _ValidatedStateToolCall]:
    schema = tool_schemas.get(call.name)
    if schema is None:
        return _invalid_tool_call(f"Unknown tool name: {call.name}")
    arguments, parse_error = parse_tool_arguments_json(call.arguments_json)
    if parse_error is not None or arguments is None:
        return _invalid_tool_call(parse_error or "Tool arguments must be a JSON object")
    error = _validate_state_tool_arguments(
        arguments,
        schema=schema,
        tool_name=call.name,
        source_messages_by_id=source_messages_by_id,
        state_by_key=state_by_key,
        script_guard_mode_value=script_guard_mode_value,
    )
    if error is not None:
        return _invalid_tool_call(error)
    return True, _accepted_tool_result(), _ValidatedStateToolCall(
        arguments=arguments,
        extraction=_state_tool_call_extraction(
            call.name,
            arguments,
            state_by_key=state_by_key,
        ),
    )


def _state_tool_call_key(call: ProviderToolCall) -> str | None:
    if not _is_state_keyed_tool(call.name):
        return None
    arguments, parse_error = parse_tool_arguments_json(call.arguments_json)
    if parse_error is not None or arguments is None:
        return None
    key = arguments.get("key")
    if not isinstance(key, str) or not key.strip():
        return None
    return key.strip()


def _is_state_keyed_tool(tool_name: str) -> bool:
    return tool_name in {"patch_world_state", "flag_state_conflict"}


def _invalid_tool_call(
    error: str,
) -> tuple[bool, dict[str, str], _ValidatedStateToolCall]:
    return (
        False,
        invalid_tool_result(error),
        _ValidatedStateToolCall(arguments={}, extraction=None),
    )


def _accepted_tool_result() -> dict[str, str]:
    return accepted_tool_result()


def _validate_state_tool_arguments(
    arguments: dict[str, object],
    *,
    schema: dict[str, object],
    tool_name: str,
    source_messages_by_id: dict[str, MessageRecord],
    state_by_key: dict[str, WorldStateRecord],
    script_guard_mode_value: str,
) -> str | None:
    shape_error = validate_tool_arguments_shape(
        arguments,
        schema=schema,
        skip_enum_fields=frozenset({"source_message_id"}),
    )
    if shape_error is not None:
        return shape_error

    quote_field = (
        "new_evidence"
        if tool_name == "flag_state_conflict"
        else "evidence_quote"
    )
    quote_error = _validate_exact_quote(
        arguments,
        source_messages_by_id=source_messages_by_id,
        quote_field=quote_field,
    )
    if quote_error is not None:
        return quote_error
    if tool_name == "patch_world_state":
        script_error = _state_tool_script_policy_error(
            arguments,
            fields=("key", "value_patch", "evidence_quote", "resolution_quote"),
            source_messages_by_id=source_messages_by_id,
            mode=script_guard_mode_value,
        )
        if script_error is not None:
            return script_error
        if "resolution_quote" in arguments:
            resolution_error = _validate_exact_quote(
                arguments,
                source_messages_by_id=source_messages_by_id,
                quote_field="resolution_quote",
            )
            if resolution_error is not None:
                return resolution_error
        return _validate_world_state_patch(arguments, state_by_key=state_by_key)
    if tool_name == "record_memory_fact":
        body = arguments.get("body")
        if not isinstance(body, str) or not body.strip():
            return "Memory body is required"
        script_error = _state_tool_script_policy_error(
            arguments,
            fields=("body", "tags", "evidence_quote"),
            source_messages_by_id=source_messages_by_id,
            mode=script_guard_mode_value,
        )
        if script_error is not None:
            return script_error
    if tool_name == "flag_state_conflict":
        key = arguments.get("key")
        if not isinstance(key, str) or not key.strip():
            return "State conflict key is required"
        script_error = _state_tool_script_policy_error(
            arguments,
            fields=("key", "new_evidence", "reason"),
            source_messages_by_id=source_messages_by_id,
            mode=script_guard_mode_value,
        )
        if script_error is not None:
            return script_error
    return None


def _validate_world_state_patch(
    arguments: dict[str, object],
    *,
    state_by_key: dict[str, WorldStateRecord],
) -> str | None:
    key = arguments.get("key")
    if not isinstance(key, str) or not key.strip():
        return "State change key is required"
    operation = arguments.get("operation")
    if operation == "delete":
        if key not in state_by_key:
            return f"state delete for {key} would not change old state"
        return None
    if operation != "upsert":
        return f"Unsupported state operation: {operation}"
    raw_patch = arguments.get("value_patch")
    if not isinstance(raw_patch, dict):
        return "value_patch is required for upsert"
    value_patch = cast(dict[str, object], raw_patch)
    if not value_patch:
        return f"state patch for {key} is empty"
    existing = state_by_key.get(key)
    if existing is None:
        return None
    unchanged_fields = {
        field
        for field, value in value_patch.items()
        if existing.value.get(field) == value
    }
    if unchanged_fields:
        for field in unchanged_fields:
            value_patch.pop(field, None)
        if not value_patch:
            return f"state patch for {key} does not change old state"
    merged = {**existing.value, **value_patch}
    if merged == existing.value:
        return f"state patch for {key} does not change old state"
    return None


def _state_tool_script_policy_error(
    arguments: dict[str, object],
    *,
    fields: tuple[str, ...],
    source_messages_by_id: dict[str, MessageRecord],
    mode: str,
) -> str | None:
    source_message_id = arguments.get("source_message_id")
    source_message = (
        source_messages_by_id.get(source_message_id)
        if isinstance(source_message_id, str)
        else None
    )
    allowed_scripts = allowed_generated_scripts(
        (source_message.body,) if source_message is not None else ()
    )
    for field_name in fields:
        if field_name not in arguments:
            continue
        violations = object_text_script_violations(
            arguments[field_name],
            allowed_scripts=allowed_scripts,
            mode=mode,
            field_name=field_name,
        )
        if violations:
            return summarize_script_policy_violations(violations)
    return None


def _state_tool_call_extraction(
    tool_name: str,
    arguments: dict[str, object],
    *,
    state_by_key: dict[str, WorldStateRecord],
) -> object:
    if tool_name == "patch_world_state":
        return _state_change_from_tool_arguments(arguments, state_by_key=state_by_key)
    if tool_name == "record_memory_fact":
        return _memory_from_tool_arguments(arguments)
    if tool_name == "flag_state_conflict":
        return _state_conflict_from_tool_arguments(arguments)
    raise ValueError(f"Unsupported state extraction tool: {tool_name}")


def _state_change_from_tool_arguments(
    arguments: dict[str, object],
    *,
    state_by_key: dict[str, WorldStateRecord],
) -> ExtractedStateChange:
    key = str(arguments.get("key", "")).strip()
    operation = str(arguments.get("operation", "upsert")).strip() or "upsert"
    existing = state_by_key.get(key)
    raw_patch = arguments.get("value_patch", {})
    value_patch = (
        cast(dict[str, object], raw_patch) if isinstance(raw_patch, dict) else {}
    )
    if operation == "upsert":
        value = {**(existing.value if existing is not None else {}), **value_patch}
    else:
        value = {}
    category = str(arguments.get("category", "")).strip()
    if not category and existing is not None:
        category = existing.category
    if not category:
        category = "world_state"
    return ExtractedStateChange(
        operation=operation,
        key=key,
        value=value,
        category=category,
        confidence=_confidence(arguments.get("confidence")),
        source_message_id=str(arguments.get("source_message_id", "")).strip(),
        evidence_quote=str(arguments.get("evidence_quote", "")).strip(),
        persistence_scope=str(arguments.get("persistence_scope", "")).strip(),
    )


def _memory_from_tool_arguments(arguments: dict[str, object]) -> ExtractedMemory:
    raw_tags = arguments.get("tags", [])
    tags = (
        tuple(
            item.strip()
            for item in raw_tags
            if isinstance(item, str) and item.strip()
        )
        if isinstance(raw_tags, list)
        else ()
    )
    return ExtractedMemory(
        body=str(arguments.get("body", "")).strip(),
        tags=tags,
        importance=_confidence(arguments.get("importance")),
        source_message_id=str(arguments.get("source_message_id", "")).strip(),
        evidence_quote=str(arguments.get("evidence_quote", "")).strip(),
    )


def _state_conflict_from_tool_arguments(
    arguments: dict[str, object],
) -> ExtractedStateConflict:
    current_value = arguments.get("current_value")
    proposed_value = arguments.get("proposed_value")
    return ExtractedStateConflict(
        key=str(arguments.get("key", "")).strip(),
        source_message_id=str(arguments.get("source_message_id", "")).strip(),
        new_evidence=str(arguments.get("new_evidence", "")).strip(),
        current_value=(
            dict(cast(dict[str, object], current_value))
            if isinstance(current_value, dict)
            else None
        ),
        proposed_value=(
            dict(cast(dict[str, object], proposed_value))
            if isinstance(proposed_value, dict)
            else None
        ),
        reason=str(arguments.get("reason", "")).strip(),
        confidence=_confidence(arguments.get("confidence")),
    )


def _state_change_from_data(value: object) -> ExtractedStateChange:
    if not isinstance(value, dict):
        raise ValueError("Structured state change must be an object")
    state_value = value.get("value", {})
    if isinstance(state_value, str):
        state_value = {"text": state_value.strip()} if state_value.strip() else {}
    if not isinstance(state_value, dict):
        raise ValueError("Structured state change value must be an object")
    return ExtractedStateChange(
        operation=str(value.get("operation", "")),
        key=str(value.get("key", "")),
        value=state_value,
        category=str(value.get("category", "")),
        confidence=float(value.get("confidence", 1.0)),
        source_message_id=str(value.get("source_message_id", "")),
        evidence_quote=str(value.get("evidence_quote", "")),
        persistence_scope=str(value.get("persistence_scope", "")),
    )


def _memory_from_data(value: object) -> ExtractedMemory:
    if not isinstance(value, dict):
        raise ValueError("Structured memory must be an object")
    raw_tags = value.get("tags", [])
    if not isinstance(raw_tags, list) or not all(
        isinstance(item, str) for item in raw_tags
    ):
        raise ValueError("Structured memory tags must be a string list")
    return ExtractedMemory(
        body=str(value.get("body", "")),
        tags=tuple(raw_tags),
        importance=float(value.get("importance", 1.0)),
        source_message_id=str(value.get("source_message_id", "")),
        evidence_quote=str(value.get("evidence_quote", "")),
    )


def _state_conflict_from_data(value: object) -> ExtractedStateConflict:
    if not isinstance(value, dict):
        raise ValueError("Structured state conflict must be an object")
    return ExtractedStateConflict(
        key=str(value.get("key", "")).strip(),
        source_message_id=str(value.get("source_message_id", "")).strip(),
        new_evidence=str(value.get("new_evidence", "")).strip(),
        current_value=_structured_state_value(value.get("current_value")),
        proposed_value=_structured_state_value(value.get("proposed_value")),
        reason=str(value.get("reason", "")).strip(),
        confidence=_confidence(value.get("confidence")),
    )


def _structured_state_value(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return dict(cast(dict[str, object], value))
    if isinstance(value, str) and value.strip():
        return {"text": value.strip()}
    return None


def _confidence(value: object) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return 1.0
    return min(max(float(value), 0.0), 1.0)


def _validate_exact_quote(
    arguments: dict[str, object],
    *,
    source_messages_by_id: dict[str, MessageRecord],
    quote_field: str,
) -> str | None:
    source_message_id = arguments.get("source_message_id")
    if not isinstance(source_message_id, str) or not source_message_id.strip():
        return "source_message_id is required"
    source = source_messages_by_id.get(source_message_id)
    if source is None:
        return f"source_message_id is not in the completed turn: {source_message_id}"
    quote = arguments.get(quote_field)
    if not isinstance(quote, str) or not quote.strip():
        return f"{quote_field} is required"
    if not _quote_matches_source(quote, source.body):
        return f"{quote_field} not found in source message {source_message_id}"
    return None


def _quote_matches_source(quote: str, source_body: str) -> bool:
    stripped = quote.strip()
    if stripped in source_body:
        return True
    normalized_quote = _format_normalized_quote(stripped)
    if not normalized_quote:
        return False
    return normalized_quote in _format_normalized_quote(source_body)


def _format_normalized_quote(value: str) -> str:
    normalized = value.translate(_FORMAT_NORMALIZED_QUOTE_TRANSLATION)
    for marker in _MARKDOWN_QUOTE_MARKERS:
        if normalized.count(marker) >= 2:
            normalized = normalized.replace(marker, "")
    return " ".join(normalized.split())


def _has_resolution_quote(
    arguments: dict[str, object],
    *,
    source_messages_by_id: dict[str, MessageRecord],
) -> bool:
    return (
        _validate_exact_quote(
            arguments,
            source_messages_by_id=source_messages_by_id,
            quote_field="resolution_quote",
        )
        is None
    )


def _canonical_tool_arguments(arguments: dict[str, object]) -> str:
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"))


def _partial_state_extraction(
    *,
    state_changes: tuple[ExtractedStateChange, ...],
    memories: tuple[ExtractedMemory, ...],
    conflicts: tuple[ExtractedStateConflict, ...],
    unsafe_state_keys: set[str],
    unknown_state_key: bool,
    diagnostics: dict[str, object],
    last_errors: list[str],
) -> StateExtraction | None:
    filtered_state_changes = tuple(
        change
        for change in state_changes
        if not unknown_state_key and change.key not in unsafe_state_keys
    )
    filtered_conflicts = tuple(
        conflict
        for conflict in conflicts
        if not unknown_state_key and conflict.key not in unsafe_state_keys
    )
    if not filtered_state_changes and not memories and not filtered_conflicts:
        return None

    suppressed_keys = sorted(
        {
            key
            for key in unsafe_state_keys
            if any(change.key == key for change in state_changes)
            or any(conflict.key == key for conflict in conflicts)
        }
    )
    if unknown_state_key:
        suppressed_keys = sorted(
            {change.key for change in state_changes}
            | {conflict.key for conflict in conflicts}
        )
    diagnostics["partial_success"] = True
    diagnostics["partial_suppressed_state_keys"] = suppressed_keys
    diagnostics["final_validation_errors"] = list(last_errors)
    return StateExtraction(
        state_changes=filtered_state_changes,
        memories=memories,
        conflicts=filtered_conflicts,
        tool_diagnostics=_final_tool_diagnostics(diagnostics),
    )


def _initial_tool_diagnostics(
    *,
    provider: str,
    model_id: str,
    fallback_used: bool,
) -> dict[str, object]:
    return {
        "provider": provider,
        "model": model_id,
        "fallback_used": fallback_used,
        "retry_count": 0,
        "turns": [],
        "accepted_calls": [],
        "rejected_calls": [],
        "validation_errors": [],
    }


def _tool_call_diagnostic(call: ProviderToolCall) -> dict[str, object]:
    return {
        "id": call.id,
        "name": call.name,
        "arguments_json": call.arguments_json,
    }


def _append_tool_diagnostic_call(
    diagnostics: dict[str, object],
    key: str,
    call: ProviderToolCall,
    *,
    error: str | None = None,
) -> None:
    calls = diagnostics.get(key)
    if not isinstance(calls, list):
        return
    entry = _tool_call_diagnostic(call)
    if error:
        entry["error"] = error
    calls.append(entry)
    if error:
        errors = diagnostics.get("validation_errors")
        if isinstance(errors, list):
            errors.append(error)


def _append_tool_diagnostic_turn(
    diagnostics: dict[str, object],
    *,
    turn: int,
    raw_calls: list[dict[str, object]],
    errors: list[str],
) -> None:
    turns = diagnostics.get("turns")
    if not isinstance(turns, list):
        return
    turns.append(
        {
            "turn": turn,
            "raw_tool_calls": raw_calls,
            "validation_errors": list(errors),
        }
    )


def _final_tool_diagnostics(
    diagnostics: dict[str, object],
) -> dict[str, object]:
    final = dict(diagnostics)
    for key in ("turns", "accepted_calls", "rejected_calls", "validation_errors"):
        value = final.get(key)
        if isinstance(value, list):
            final[key] = list(value)
    return final


def _attach_tool_diagnostics(
    exc: Exception,
    diagnostics: dict[str, object],
) -> None:
    exc.tool_diagnostics = diagnostics  # type: ignore[attr-defined]


def _tool_diagnostics_from_exception(exc: Exception) -> dict[str, object]:
    diagnostics = getattr(exc, "tool_diagnostics", None)
    return diagnostics if isinstance(diagnostics, dict) else {}


def _failure_tool_diagnostics(exc: Exception) -> dict[str, object] | None:
    diagnostics = _tool_diagnostics_from_exception(exc)
    if not diagnostics:
        return None
    return {"tool_diagnostics": diagnostics}


def _merge_tool_diagnostics(
    primary: dict[str, object],
    fallback: dict[str, object],
    *,
    fallback_used: bool,
) -> dict[str, object]:
    if not primary:
        merged = dict(fallback)
        merged["fallback_used"] = fallback_used
        return merged
    if not fallback:
        merged = dict(primary)
        merged["fallback_used"] = fallback_used
        return merged
    return {
        "fallback_used": fallback_used,
        "primary": primary,
        "fallback": fallback,
    }


def _safety_transition_source_ids(
    messages: tuple[MessageRecord, ...] | list[MessageRecord],
) -> frozenset[str]:
    return frozenset(
        message.id
        for message in messages
        if is_fade_to_black_message(
            role=message.role,
            body=message.body,
            safety_transition=message.safety_transition,
        )
    )
