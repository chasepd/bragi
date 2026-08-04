"""Model-assisted pruning for obsolete world-state rows."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Protocol

from bragi.app_logging import exception_log_fields, log_error_event, log_event
from bragi.persistence.models import (
    JobRecord,
    MessageRecord,
    ScenarioRecord,
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
from bragi.redaction import redact_text
from bragi.retry_policy import MODEL_OUTPUT_MAX_ATTEMPTS, configured_max_attempts
from bragi.services.job_lifecycle import JobLifecycleService
from bragi.services.model_capabilities import (
    STRUCTURED_OUTPUT_CAPABILITIES,
    TOOL_CALLING_CAPABILITIES,
    known_model_is_unavailable,
    model_supports_any_capability,
)
from bragi.services.model_preferences import roleplay_model_preference
from bragi.services.openrouter_routing_settings import (
    request_with_openrouter_routing,
)
from bragi.services.provider_fallbacks import (
    provider_error_with_fallback_attempted,
    provider_error_with_fallback_skipped_reason,
    recover_tool_call_shape_with_structured_output,
    shape_switch_diagnostics,
    structured_output_with_fallback,
    tool_call_fallback_request,
    tool_call_fallback_skip_reason,
)
from bragi.services.request_budget import budget_tool_call_request
from bragi.services.tool_call_helpers import (
    accepted_tool_result,
    append_tool_feedback_messages,
    invalid_tool_result,
    parse_tool_arguments_json,
    validate_tool_arguments_shape,
)

RECENT_MESSAGE_LIMIT = 24
STATE_PRUNING_BATCH_SIZE = 40
MAX_STATE_PRUNING_TOOL_FEEDBACK_TURNS = MODEL_OUTPUT_MAX_ATTEMPTS - 1


@dataclass(frozen=True)
class PrunedWorldStateFact:
    world_state_id: str
    key: str
    reason: str


@dataclass(frozen=True)
class StatePruningResult:
    proposed: tuple[PrunedWorldStateFact, ...]
    archived: tuple[PrunedWorldStateFact, ...]
    rejected: tuple[PrunedWorldStateFact, ...]
    decisions: tuple[str, ...] = ()
    active_state_count: int = 0
    batch_count: int = 0
    completed_batch_count: int = 0
    batch_size: int = STATE_PRUNING_BATCH_SIZE
    tool_diagnostics: dict[str, object] = field(default_factory=dict)


class StatePruningRunner(Protocol):
    async def prune(
        self,
        *,
        save_id: str,
        review_only: bool = False,
    ) -> StatePruningResult:
        ...


class StatePruningService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        providers: dict[str, ProviderClient],
        state_batch_size: int = STATE_PRUNING_BATCH_SIZE,
    ) -> None:
        self.repositories = repositories
        self.providers = providers
        self.state_batch_size = max(1, state_batch_size)
        self.jobs = JobLifecycleService(repositories=repositories)

    async def prune(
        self,
        *,
        save_id: str,
        review_only: bool = False,
        apply_guard: Callable[[], AbstractAsyncContextManager[None]] | None = None,
    ) -> StatePruningResult:
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="state_pruning",
        )
        if preference is None:
            raise ValueError("No state-pruning model preference configured")
        started_at = perf_counter()
        batch_results: list[StatePruningResult] = []
        active_state_count = 0
        batch_count = 0
        failed_batch_index: int | None = None
        shape_diagnostics: dict[str, object] = {}
        job: JobRecord | None = None
        try:
            async with _apply_guard_context(apply_guard):
                narrator_turn_count = sum(
                    1
                    for message in self.repositories.list_messages(save_id)
                    if message.role == "narrator"
                )
                job = self.jobs.create_running(
                    save_id=save_id,
                    type="state_pruning",
                    payload={
                        "review_only": review_only,
                        "automatic": not review_only,
                        "narrator_turn_count": narrator_turn_count,
                    },
                    collect_provider_diagnostics=True,
                )
                log_event(
                    "job.running",
                    job_id=job.id,
                    job_type=job.type,
                    save_id=save_id,
                    review_only=review_only,
                )
                active_state = tuple(self.repositories.list_world_state(save_id))
                active_state_count = len(active_state)
                state_batches = _world_state_batches(
                    active_state,
                    self.state_batch_size,
                )
                batch_count = len(state_batches)
                provider = self.providers[preference.provider]
                if known_model_is_unavailable(
                    repositories=self.repositories,
                    provider=preference.provider,
                    model_id=preference.model_id,
                ):
                    raise ValueError(
                        f"State pruning model is unavailable: {preference.model_id}"
                    )
                supports_tool_calling = (
                    isinstance(provider, ToolCallProvider)
                    and _model_supports_tool_calling(
                        repositories=self.repositories,
                        provider=preference.provider,
                        model_id=preference.model_id,
                    )
                )
                supports_structured_output = (
                    isinstance(provider, StructuredOutputProvider)
                    and _model_supports_structured_output(
                        repositories=self.repositories,
                        provider=preference.provider,
                        model_id=preference.model_id,
                    )
                )
                if not supports_tool_calling and not supports_structured_output:
                    raise ValueError(
                        "State-pruning model does not advertise structured output "
                        "or tool calling"
                    )
                details = self.repositories.load_save_details(save_id)
                scenario = details.scenario if details is not None else None
                messages = tuple(
                    self.repositories.list_messages(save_id)[-RECENT_MESSAGE_LIMIT:]
                )
            for batch_index, batch in enumerate(state_batches):
                failed_batch_index = batch_index
                if supports_tool_calling:
                    proposed, batch_diagnostics = (
                        await _select_pruned_state_with_tool_calls(
                            repositories=self.repositories,
                            providers=self.providers,
                            provider=provider,
                            provider_name=preference.provider,
                            model_id=preference.model_id,
                            save_id=save_id,
                            scenario=scenario,
                            active_state=batch,
                            recent_messages=messages,
                        )
                    )
                else:
                    proposed = await _select_pruned_state(
                        repositories=self.repositories,
                        providers=self.providers,
                        provider_name=preference.provider,
                        model_id=preference.model_id,
                        save_id=save_id,
                        scenario=scenario,
                        active_state=batch,
                        recent_messages=messages,
                    )
                    batch_diagnostics = {}
                if batch_diagnostics and not shape_diagnostics:
                    shape_diagnostics = batch_diagnostics
                batch_results.append(
                    await self._apply_proposals(
                        save_id=save_id,
                        active_state=batch,
                        proposed=proposed,
                        review_only=review_only,
                        apply_guard=apply_guard,
                    ),
                )
            failed_batch_index = None
            result = _merge_state_pruning_results(
                tuple(batch_results),
                active_state_count=active_state_count,
                batch_count=batch_count,
                completed_batch_count=len(batch_results),
                batch_size=self.state_batch_size,
                tool_diagnostics=shape_diagnostics,
            )
        except asyncio.CancelledError:
            if job is not None:
                async with _apply_guard_context(apply_guard):
                    self.jobs.cancel(
                        job.id,
                        error="State pruning cancelled",
                        result=_partial_result_json(
                            tuple(batch_results),
                            review_only=review_only,
                            active_state_count=active_state_count,
                            batch_count=batch_count,
                            completed_batch_count=len(batch_results),
                            batch_size=self.state_batch_size,
                            failed_batch_index=failed_batch_index,
                        ),
                    )
            log_event(
                "job.cancelled",
                job_id=job.id if job is not None else "",
                job_type=job.type if job is not None else "state_pruning",
                save_id=save_id,
                duration_ms=_elapsed_ms(started_at),
            )
            raise
        except Exception as exc:
            if job is not None:
                async with _apply_guard_context(apply_guard):
                    self.jobs.fail(
                        job.id,
                        error=redact_text(str(exc)) or exc.__class__.__name__,
                        result=_partial_result_json(
                            tuple(batch_results),
                            review_only=review_only,
                            active_state_count=active_state_count,
                            batch_count=batch_count,
                            completed_batch_count=len(batch_results),
                            batch_size=self.state_batch_size,
                            failed_batch_index=failed_batch_index,
                        ),
                    )
            log_error_event(
                "job.failed",
                job_id=job.id if job is not None else "",
                job_type=job.type if job is not None else "state_pruning",
                save_id=save_id,
                duration_ms=_elapsed_ms(started_at),
                **exception_log_fields(exc),
            )
            raise
        async with _apply_guard_context(apply_guard):
            self.jobs.succeed(
                job.id,
                result=_result_json(result, review_only=review_only),
            )
        log_event(
            "job.succeeded",
            job_id=job.id,
            job_type=job.type,
            save_id=save_id,
            duration_ms=_elapsed_ms(started_at),
            proposed_count=len(result.proposed),
            archived_count=len(result.archived),
            rejected_count=len(result.rejected),
            review_only=review_only,
        )
        return result

    async def _apply_proposals(
        self,
        *,
        save_id: str,
        active_state: tuple[WorldStateRecord, ...],
        proposed: tuple[PrunedWorldStateFact, ...],
        review_only: bool,
        apply_guard: Callable[[], AbstractAsyncContextManager[None]] | None,
    ) -> StatePruningResult:
        active_by_id = {record.id: record for record in active_state}
        accepted_records: list[tuple[int, PrunedWorldStateFact, WorldStateRecord]] = []
        archived: list[PrunedWorldStateFact] = []
        rejected: list[PrunedWorldStateFact] = []
        decisions = ["rejected"] * len(proposed)
        seen_ids: set[str] = set()
        for index, proposal in enumerate(proposed):
            record = active_by_id.get(proposal.world_state_id)
            if proposal.world_state_id in seen_ids:
                rejected.append(proposal)
                continue
            if (
                record is None
                or record.save_id != save_id
                or record.key != proposal.key
            ):
                rejected.append(proposal)
                continue
            seen_ids.add(proposal.world_state_id)
            if review_only:
                decisions[index] = "proposed"
            elif _high_value_world_state(record) and not _has_contradiction_reason(
                proposal.reason
            ):
                rejected.append(proposal)
            else:
                accepted_records.append((index, proposal, record))
        if not review_only:
            async with _apply_guard_context(apply_guard):
                self.repositories.begin_transaction()
                try:
                    for index, proposal, record in accepted_records:
                        archived_row = (
                            self.repositories.archive_world_state_if_unchanged(
                                save_id=save_id,
                                world_state_id=proposal.world_state_id,
                                key=proposal.key,
                                value=record.value,
                            )
                        )
                        if archived_row:
                            archived.append(proposal)
                            decisions[index] = "archived"
                        else:
                            rejected.append(proposal)
                    self.repositories.commit_transaction()
                except Exception:
                    self.repositories.rollback_transaction()
                    raise
        return StatePruningResult(
            proposed=proposed,
            archived=tuple(archived),
            rejected=tuple(rejected),
            decisions=tuple(decisions),
        )


def _apply_guard_context(
    apply_guard: Callable[[], AbstractAsyncContextManager[None]] | None,
) -> AbstractAsyncContextManager[None]:
    return apply_guard() if apply_guard is not None else nullcontext()


def _world_state_batches(
    records: tuple[WorldStateRecord, ...],
    batch_size: int,
) -> tuple[tuple[WorldStateRecord, ...], ...]:
    size = max(1, batch_size)
    return tuple(
        records[index : index + size] for index in range(0, len(records), size)
    )


def _merge_state_pruning_results(
    results: tuple[StatePruningResult, ...],
    *,
    active_state_count: int,
    batch_count: int,
    completed_batch_count: int,
    batch_size: int,
    tool_diagnostics: dict[str, object] | None = None,
) -> StatePruningResult:
    proposed: list[PrunedWorldStateFact] = []
    archived: list[PrunedWorldStateFact] = []
    rejected: list[PrunedWorldStateFact] = []
    decisions: list[str] = []
    for result in results:
        proposed.extend(result.proposed)
        archived.extend(result.archived)
        rejected.extend(result.rejected)
        decisions.extend(result.decisions)
    return StatePruningResult(
        proposed=tuple(proposed),
        archived=tuple(archived),
        rejected=tuple(rejected),
        decisions=tuple(decisions),
        active_state_count=active_state_count,
        batch_count=batch_count,
        completed_batch_count=completed_batch_count,
        batch_size=batch_size,
        tool_diagnostics=dict(tool_diagnostics or {}),
    )


def _high_value_world_state(record: WorldStateRecord) -> bool:
    key = record.key.casefold()
    category = record.category.casefold()
    return any(
        term in key or term in category
        for term in (
            "identity",
            "inventory",
            "item",
            "location",
            "object",
            "open_thread",
            "promise",
            "relationship",
            "voice",
        )
    )


def _has_contradiction_reason(reason: str) -> bool:
    normalized = reason.casefold()
    negated_markers = (
        "does not contradict",
        "doesn't contradict",
        "no contradiction",
        "not contradicted",
        "without contradiction",
    )
    if any(marker in normalized for marker in negated_markers):
        return False
    return any(
        term in normalized
        for term in (
            "contradict",
            "explicitly superseded",
            "explicitly resolved",
            "no longer true",
        )
    )


async def _select_pruned_state(
    *,
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    provider_name: str,
    model_id: str,
    save_id: str,
    scenario: ScenarioRecord | None,
    active_state: tuple[WorldStateRecord, ...],
    recent_messages: tuple[MessageRecord, ...],
) -> tuple[PrunedWorldStateFact, ...]:
    started_at = perf_counter()
    request = request_with_openrouter_routing(
        repositories,
        StructuredOutputRequest(
            provider=provider_name,
            model_id=model_id,
            schema_name="state_pruning_selection",
            schema=_state_pruning_schema(active_state),
            messages=_state_pruning_messages(
                scenario=scenario,
                active_state=active_state,
                recent_messages=recent_messages,
            ),
            temperature=0.0,
        ),
        task="state_pruning",
        save_id=save_id,
    )
    try:
        response = await structured_output_with_fallback(
            repositories=repositories,
            providers=providers,
            request=request,
            task="state_pruning",
            save_id=save_id,
        )
    except Exception as exc:
        log_error_event(
            "provider.structured_output_failed",
            provider=provider_name,
            model=model_id,
            task="state_pruning",
            duration_ms=_elapsed_ms(started_at),
            candidate_count=len(active_state),
            **exception_log_fields(exc),
        )
        raise
    log_event(
        "provider.structured_output_succeeded",
        provider=response.provider,
        model=response.model_id,
        task="state_pruning",
        duration_ms=_elapsed_ms(started_at),
        candidate_count=len(active_state),
        token_usage=response.token_usage,
    )
    return _proposals_from_structured_data(response.data)


async def _select_pruned_state_with_tool_calls(
    *,
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    provider: ProviderClient,
    provider_name: str,
    model_id: str,
    save_id: str,
    scenario: ScenarioRecord | None,
    active_state: tuple[WorldStateRecord, ...],
    recent_messages: tuple[MessageRecord, ...],
) -> tuple[tuple[PrunedWorldStateFact, ...], dict[str, object]]:
    if not isinstance(provider, ToolCallProvider):
        raise ValueError("State-pruning provider does not support tool calling")
    started_at = perf_counter()
    request = request_with_openrouter_routing(
        repositories,
        ToolCallRequest(
            provider=provider_name,
            model_id=model_id,
            messages=_state_pruning_tool_messages(
                scenario=scenario,
                active_state=active_state,
                recent_messages=recent_messages,
            ),
            tools=_state_pruning_tool_definitions(active_state),
            temperature=0.0,
        ),
        task="state_pruning",
        save_id=save_id,
    )
    try:
        result = await _select_pruned_state_with_tool_fallback(
            repositories=repositories,
            providers=providers,
            provider=provider,
            request=request,
            save_id=save_id,
            active_state=active_state,
        )
    except ProviderError as exc:
        # The tool fallback chain enriches the failing error; the enriched
        # error reports model_not_found when either the primary or the
        # fallback attempt failed with it, so recovering through the
        # structured route covers both cases.
        if provider_error_is_model_not_found(exc):
            facts = await recover_tool_call_shape_with_structured_output(
                error=exc,
                task="state_pruning",
                provider=provider_name,
                model_id=model_id,
                structured_run=lambda: _select_pruned_state(
                    repositories=repositories,
                    providers=providers,
                    provider_name=provider_name,
                    model_id=model_id,
                    save_id=save_id,
                    scenario=scenario,
                    active_state=active_state,
                    recent_messages=recent_messages,
                ),
            )
            return (
                facts,
                shape_switch_diagnostics(
                    provider=provider_name,
                    model_id=model_id,
                ),
            )
        raise
    except Exception as exc:
        log_error_event(
            "provider.tool_call_failed",
            provider=provider_name,
            model=model_id,
            task="state_pruning",
            duration_ms=_elapsed_ms(started_at),
            candidate_count=len(active_state),
            **exception_log_fields(exc),
        )
        raise
    log_event(
        "provider.tool_call_succeeded",
        provider=provider_name,
        model=model_id,
        task="state_pruning",
        duration_ms=_elapsed_ms(started_at),
        candidate_count=len(active_state),
    )
    return result, {}


async def _select_pruned_state_with_tool_fallback(
    *,
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    provider: ToolCallProvider,
    request: ToolCallRequest,
    save_id: str,
    active_state: tuple[WorldStateRecord, ...],
) -> tuple[PrunedWorldStateFact, ...]:
    try:
        return await _select_pruned_state_with_tool_feedback(
            repositories=repositories,
            provider=provider,
            request=request,
            active_state=active_state,
        )
    except ProviderError as exc:
        fallback_request = tool_call_fallback_request(
            repositories=repositories,
            providers=providers,
            request=request,
            save_id=save_id,
        )
        if fallback_request is None:
            reason = tool_call_fallback_skip_reason(
                repositories=repositories,
                providers=providers,
                save_id=save_id,
            )
            log_event(
                "provider.tool_call_fallback_skipped",
                provider=request.provider,
                model=request.model_id,
                task="state_pruning",
                reason=reason,
            )
            raise provider_error_with_fallback_skipped_reason(exc, reason) from exc
        fallback_provider = providers[fallback_request.provider]
        if not isinstance(fallback_provider, ToolCallProvider):
            reason = "fallback_provider_unavailable"
            log_event(
                "provider.tool_call_fallback_skipped",
                provider=request.provider,
                model=request.model_id,
                task="state_pruning",
                reason=reason,
            )
            raise provider_error_with_fallback_skipped_reason(exc, reason) from exc
        log_event(
            "provider.tool_call_fallback_started",
            provider=fallback_request.provider,
            model=fallback_request.model_id,
            task="state_pruning",
        )
        try:
            return await _select_pruned_state_with_tool_feedback(
                repositories=repositories,
                provider=fallback_provider,
                request=fallback_request,
                active_state=active_state,
            )
        except ProviderError as fallback_exc:
            enriched = provider_error_with_fallback_attempted(
                fallback_exc,
                provider=fallback_request.provider,
                model_id=fallback_request.model_id,
            )
            if provider_error_is_model_not_found(exc):
                enriched = replace(
                    enriched,
                    category=ProviderErrorCategory.MODEL_NOT_FOUND,
                )
            raise enriched from fallback_exc


async def _select_pruned_state_with_tool_feedback(
    *,
    repositories: PersistenceRepositories,
    provider: ToolCallProvider,
    request: ToolCallRequest,
    active_state: tuple[WorldStateRecord, ...],
) -> tuple[PrunedWorldStateFact, ...]:
    messages = list(request.messages)
    tool_schemas = {tool.name: tool.parameters for tool in request.tools}
    active_by_id = {record.id: record for record in active_state}
    proposals: list[PrunedWorldStateFact] = []
    accepted_keys: set[tuple[str, str]] = set()
    last_errors: list[str] = []
    max_attempt_count = configured_max_attempts(repositories)

    for _turn in range(max_attempt_count):
        turn_request = budget_tool_call_request(
            repositories,
            replace(request, messages=tuple(messages)),
            task="state_pruning",
        )
        response = await provider.generate_tool_calls(turn_request)
        errors: list[str] = []
        tool_results: list[tuple[ProviderToolCall, dict[str, str]]] = []
        for call in response.tool_calls:
            accepted, result, proposal = _validate_state_pruning_tool_call(
                call,
                tool_schemas=tool_schemas,
                active_by_id=active_by_id,
            )
            if accepted:
                if proposal is not None:
                    key = (proposal.world_state_id, proposal.key)
                    if key not in accepted_keys:
                        accepted_keys.add(key)
                        proposals.append(proposal)
                tool_results.append((call, accepted_tool_result()))
                continue
            errors.append(result["error"])
            tool_results.append((call, result))

        if not errors:
            return tuple(proposals)

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
            "State-pruning tool-call validation failed after feedback: "
            + "; ".join(last_errors)
        ),
    )


def _validate_state_pruning_tool_call(
    call: ProviderToolCall,
    *,
    tool_schemas: dict[str, dict[str, object]],
    active_by_id: dict[str, WorldStateRecord],
) -> tuple[bool, dict[str, str], PrunedWorldStateFact | None]:
    schema = tool_schemas.get(call.name)
    if schema is None:
        return _invalid_state_pruning_tool_call(f"Unknown tool name: {call.name}")
    arguments, parse_error = parse_tool_arguments_json(call.arguments_json)
    if parse_error is not None or arguments is None:
        return _invalid_state_pruning_tool_call(
            parse_error or "Tool arguments must be a JSON object"
        )
    shape_error = validate_tool_arguments_shape(arguments, schema=schema)
    if shape_error is not None:
        return _invalid_state_pruning_tool_call(shape_error)
    world_state_id = str(arguments.get("world_state_id", "")).strip()
    record = active_by_id.get(world_state_id)
    if record is None:
        return _invalid_state_pruning_tool_call(
            f"world_state_id is not an active fact in this batch: {world_state_id}"
        )
    key = str(arguments.get("key", "")).strip()
    if key != record.key:
        return _invalid_state_pruning_tool_call(
            f"key does not match world_state_id {world_state_id}: {key}"
        )
    reason = str(arguments.get("reason", "")).strip()
    return (
        True,
        accepted_tool_result(),
        PrunedWorldStateFact(
            world_state_id=world_state_id,
            key=key,
            reason=reason or "Selected by state pruning tool call.",
        ),
    )


def _invalid_state_pruning_tool_call(
    error: str,
) -> tuple[bool, dict[str, str], None]:
    return False, invalid_tool_result(error), None


def _state_pruning_schema(
    active_state: tuple[WorldStateRecord, ...],
) -> dict[str, object]:
    state_ids = [record.id for record in active_state]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "archives": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "world_state_id": {"type": "string", "enum": state_ids},
                        "key": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["world_state_id", "key", "reason"],
                },
            },
        },
        "required": ["archives"],
    }


def _state_pruning_messages(
    *,
    scenario: ScenarioRecord | None,
    active_state: tuple[WorldStateRecord, ...],
    recent_messages: tuple[MessageRecord, ...],
) -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(
            role="system",
            body=(
                "Select obsolete Bragi world-state facts for archival. Use the "
                "enforced response schema. Only select existing world-state IDs "
                "that are stale, contradicted, superseded, low-value for future "
                "continuity, or no longer useful. Prefer keeping uncertain facts."
            ),
        ),
        ChatMessage(
            role="user",
            body="\n\n".join(
                (
                    _scenario_context_text(scenario),
                    _active_state_text(active_state),
                    _recent_messages_text(recent_messages),
                )
            ),
        ),
    )


def _state_pruning_tool_messages(
    *,
    scenario: ScenarioRecord | None,
    active_state: tuple[WorldStateRecord, ...],
    recent_messages: tuple[MessageRecord, ...],
) -> tuple[ToolCallMessage, ...]:
    messages = _state_pruning_messages(
        scenario=scenario,
        active_state=active_state,
        recent_messages=recent_messages,
    )
    return tuple(
        ToolCallMessage(
            role=message.role,
            body=message.body.replace(
                "Use the enforced response schema.",
                (
                    "Use the archive_world_state_fact tool instead of prose. "
                    "Call it once per existing world-state fact that should be "
                    "archived. Make no tool calls when every fact should be kept."
                ),
            ),
            speaker_name=message.speaker_name,
        )
        for message in messages
    )


def _state_pruning_tool_definitions(
    active_state: tuple[WorldStateRecord, ...],
) -> tuple[ToolDefinition, ...]:
    return (
        ToolDefinition(
            name="archive_world_state_fact",
            description="Propose archiving one obsolete world-state fact.",
            parameters=_state_pruning_tool_schema(active_state),
        ),
    )


def _state_pruning_tool_schema(
    active_state: tuple[WorldStateRecord, ...],
) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "world_state_id": {
                "type": "string",
                "enum": [record.id for record in active_state],
            },
            "key": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["world_state_id", "key", "reason"],
    }


def _proposals_from_structured_data(
    data: dict[str, object],
) -> tuple[PrunedWorldStateFact, ...]:
    raw_archives = data.get("archives", [])
    if not isinstance(raw_archives, list):
        raise ValueError("Structured state pruning archives must be a list")
    proposals: list[PrunedWorldStateFact] = []
    for value in raw_archives:
        if not isinstance(value, dict):
            raise ValueError("Structured state pruning archive must be an object")
        proposals.append(
            PrunedWorldStateFact(
                world_state_id=str(value.get("world_state_id", "")),
                key=str(value.get("key", "")),
                reason=str(value.get("reason", "")).strip()
                or "Selected by state pruning model.",
            )
        )
    return tuple(proposals)


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


def _active_state_text(records: tuple[WorldStateRecord, ...]) -> str:
    return "Active world state:\n" + "\n".join(
        (
            f"- id={record.id}; key={record.key}; category={record.category}; "
            f"value={_dump(record.value)}"
        )
        for record in records
    )


def _recent_messages_text(messages: tuple[MessageRecord, ...]) -> str:
    if not messages:
        return "Recent transcript messages: none"
    return "Recent transcript messages:\n" + "\n".join(
        f"- {message.id} [{message.role}] {message.body}" for message in messages
    )


def _result_json(
    result: StatePruningResult,
    *,
    review_only: bool,
) -> dict[str, object]:
    data: dict[str, object] = {
        "active_state_count": result.active_state_count,
        "batch_count": result.batch_count,
        "completed_batch_count": result.completed_batch_count,
        "batch_size": result.batch_size,
        "proposed_count": len(result.proposed),
        "archived_count": len(result.archived),
        "rejected_count": len(result.rejected),
        "review_only": review_only,
        "proposals": [
            {
                "world_state_id": proposal.world_state_id,
                "key": proposal.key,
                "reason": redact_text(proposal.reason),
                "archived": _proposal_decision(result, index) == "archived",
                "rejected": _proposal_decision(result, index) == "rejected",
            }
            for index, proposal in enumerate(result.proposed)
        ],
    }
    if result.tool_diagnostics:
        data["tool_diagnostics"] = result.tool_diagnostics
    return data


def _partial_result_json(
    results: tuple[StatePruningResult, ...],
    *,
    review_only: bool,
    active_state_count: int,
    batch_count: int,
    completed_batch_count: int,
    batch_size: int,
    failed_batch_index: int | None,
) -> dict[str, object]:
    result = _merge_state_pruning_results(
        results,
        active_state_count=active_state_count,
        batch_count=batch_count,
        completed_batch_count=completed_batch_count,
        batch_size=batch_size,
    )
    data = _result_json(result, review_only=review_only)
    if failed_batch_index is not None:
        data["failed_batch_index"] = failed_batch_index
    return data


def _proposal_decision(result: StatePruningResult, index: int) -> str:
    if index < len(result.decisions):
        return result.decisions[index]
    proposal = result.proposed[index]
    if proposal in result.archived:
        return "archived"
    if proposal in result.rejected:
        return "rejected"
    return "proposed"


def _model_supports_structured_output(
    *,
    repositories: PersistenceRepositories,
    provider: str,
    model_id: str,
) -> bool:
    return model_supports_any_capability(
        repositories,
        provider=provider,
        model_id=model_id,
        required=STRUCTURED_OUTPUT_CAPABILITIES,
    )


def _model_supports_tool_calling(
    *,
    repositories: PersistenceRepositories,
    provider: str,
    model_id: str,
) -> bool:
    return model_supports_any_capability(
        repositories,
        provider=provider,
        model_id=model_id,
        required=TOOL_CALLING_CAPABILITIES,
    )


def _dump(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def _elapsed_ms(started_at: float) -> int:
    return int((perf_counter() - started_at) * 1000)
