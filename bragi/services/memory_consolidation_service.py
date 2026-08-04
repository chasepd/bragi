"""Best-effort memory rewrite and duplicate consolidation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, replace
from time import perf_counter

from bragi.app_logging import exception_log_fields, log_error_event, log_event
from bragi.persistence.models import MemoryRecord
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
from bragi.services.openrouter_routing_settings import (
    request_with_openrouter_routing,
)
from bragi.services.prompt_inspection import PromptInspectionStore
from bragi.services.provider_fallbacks import (
    provider_error_with_fallback_attempted,
    provider_error_with_fallback_skipped_reason,
    recover_tool_call_shape_with_structured_output,
    structured_output_with_fallback,
    tool_call_fallback_request,
    tool_call_fallback_skip_reason,
)
from bragi.services.request_budget import (
    budget_structured_output_request,
    budget_tool_call_request,
)
from bragi.services.text_script_policy import (
    DEFAULT_SCRIPT_GUARD_MODE,
    ScriptPolicyViolation,
    allowed_generated_scripts,
    object_text_script_violations,
    script_guard_mode,
    summarize_script_policy_violations,
)
from bragi.services.tool_call_helpers import (
    accepted_tool_result,
    append_tool_feedback_messages,
    invalid_tool_result,
    parse_tool_arguments_json,
    validate_tool_arguments_shape,
)

MEMORY_CONSOLIDATION_THRESHOLD = 40
MEMORY_CONSOLIDATION_BATCH_SIZE = 80
MEMORY_CONSOLIDATION_MIN_CONFIDENCE = 0.85
MAX_MEMORY_CONSOLIDATION_TOOL_FEEDBACK_TURNS = MODEL_OUTPUT_MAX_ATTEMPTS - 1

_ApplyGuardFactory = Callable[[], AbstractAsyncContextManager[None]]


@dataclass(frozen=True)
class MemoryConsolidationCluster:
    canonical_memory_id: str
    merged_memory_ids: tuple[str, ...]
    body: str
    tags: tuple[str, ...]
    importance: float
    confidence: float
    reason: str


@dataclass(frozen=True)
class MemoryConsolidationResult:
    save_id: str
    active_memory_count: int
    proposed_cluster_count: int = 0
    rewritten_count: int = 0
    archived_count: int = 0
    rejected_count: int = 0
    skipped_reason: str = ""
    batch_count: int = 0
    completed_batch_count: int = 0


def _apply_guard_context(
    apply_guard: _ApplyGuardFactory | None,
) -> AbstractAsyncContextManager[None]:
    if apply_guard is None:
        return _noop_apply_guard()
    return apply_guard()


@asynccontextmanager
async def _noop_apply_guard() -> AsyncIterator[None]:
    yield


class MemoryConsolidationService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        provider: StructuredOutputProvider | ToolCallProvider,
        provider_name: str,
        model_id: str,
        providers: dict[str, ProviderClient] | None = None,
        prompt_inspection_store: PromptInspectionStore | None = None,
        inspection_message_id: str | None = None,
        prefer_tool_calls: bool = False,
    ) -> None:
        self.repositories = repositories
        self.provider = provider
        self.provider_name = provider_name
        self.model_id = model_id
        self.providers = providers
        self.prompt_inspection_store = prompt_inspection_store
        self.inspection_message_id = inspection_message_id
        self.prefer_tool_calls = prefer_tool_calls
        self.jobs = JobLifecycleService(repositories=repositories)

    async def consolidate_if_needed(
        self,
        save_id: str,
        *,
        min_active_memories: int = MEMORY_CONSOLIDATION_THRESHOLD,
        apply_guard: _ApplyGuardFactory | None = None,
    ) -> MemoryConsolidationResult:
        memories = tuple(self.repositories.list_memories(save_id))
        if len(memories) < min_active_memories:
            return MemoryConsolidationResult(
                save_id=save_id,
                active_memory_count=len(memories),
                skipped_reason="active memory count below threshold",
            )
        job = self.jobs.create_running(
            save_id=save_id,
            type="memory_consolidation",
            payload={"active_memory_count": len(memories)},
            collect_provider_diagnostics=True,
        )
        started_at = perf_counter()
        transaction_started = False
        result: MemoryConsolidationResult
        try:
            batches = _memory_batches(memories, MEMORY_CONSOLIDATION_BATCH_SIZE)
            batch_clusters: list[
                tuple[tuple[MemoryRecord, ...], tuple[MemoryConsolidationCluster, ...]]
            ] = []
            for batch in batches:
                batch_clusters.append(
                    (
                        batch,
                        await self._propose_clusters(
                            save_id=save_id,
                            memories=batch,
                        ),
                    )
                )
            async with _apply_guard_context(apply_guard):
                batch_results: list[MemoryConsolidationResult] = []
                self.repositories.begin_immediate_transaction()
                transaction_started = True
                for batch, clusters in batch_clusters:
                    batch_results.append(
                        self._apply_clusters(
                            save_id=save_id,
                            memories=batch,
                            clusters=clusters,
                        )
                    )
                result = _merge_results(
                    save_id=save_id,
                    active_memory_count=len(memories),
                    batch_count=len(batches),
                    batch_results=tuple(batch_results),
                )
                self.repositories.commit_transaction()
                transaction_started = False
                self.jobs.succeed(
                    job.id,
                    result={
                        "active_memory_count": result.active_memory_count,
                        "proposed_cluster_count": result.proposed_cluster_count,
                        "rewritten_count": result.rewritten_count,
                        "archived_count": result.archived_count,
                        "rejected_count": result.rejected_count,
                        "batch_count": result.batch_count,
                        "completed_batch_count": result.completed_batch_count,
                    },
                )
        except Exception as exc:
            if transaction_started:
                self.repositories.rollback_transaction()
            self.jobs.fail(
                job.id,
                error=redact_text(str(exc)) or exc.__class__.__name__,
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
        log_event(
            "job.succeeded",
            job_id=job.id,
            job_type=job.type,
            save_id=save_id,
            duration_ms=_elapsed_ms(started_at),
            rewritten_count=result.rewritten_count,
            archived_count=result.archived_count,
            rejected_count=result.rejected_count,
        )
        return result

    async def _propose_clusters(
        self,
        *,
        save_id: str,
        memories: tuple[MemoryRecord, ...],
    ) -> tuple[MemoryConsolidationCluster, ...]:
        if self.prefer_tool_calls and isinstance(self.provider, ToolCallProvider):
            tool_request = request_with_openrouter_routing(
                self.repositories,
                ToolCallRequest(
                    provider=self.provider_name,
                    model_id=self.model_id,
                    messages=_memory_consolidation_tool_messages(memories),
                    tools=_memory_consolidation_tool_definitions(memories),
                    temperature=0.0,
                ),
                task="context_update",
                save_id=save_id,
            )
            if (
                self.prompt_inspection_store is not None
                and self.inspection_message_id is not None
            ):
                self.prompt_inspection_store.capture_tool_call_request(
                    message_id=self.inspection_message_id,
                    kind="memory_consolidation_tool_calls",
                    title="Memory consolidation tool calls",
                    request=tool_request,
                )
            if self.providers is not None:
                try:
                    return await _memory_clusters_with_tool_fallback(
                        repositories=self.repositories,
                        providers=self.providers,
                        provider=self.provider,
                        request=tool_request,
                        save_id=save_id,
                        memories=memories,
                        script_guard_mode_value=script_guard_mode(
                            self.repositories,
                            save_id=save_id,
                        ),
                    )
                except ProviderError as exc:
                    # The tool fallback chain enriches the failing error,
                    # which keeps the category of whichever attempt ended the
                    # tool path; either one failing with model_not_found means
                    # the tool shape is unavailable.
                    if not provider_error_is_model_not_found(exc):
                        raise
                    return await recover_tool_call_shape_with_structured_output(
                        error=exc,
                        task="context_update",
                        provider=self.provider_name,
                        model_id=self.model_id,
                        structured_run=lambda: self._propose_clusters_structured(
                            save_id=save_id,
                            memories=memories,
                        ),
                    )
            try:
                return await _memory_clusters_with_tool_feedback(
                    repositories=self.repositories,
                    provider=self.provider,
                    request=tool_request,
                    memories=memories,
                    script_guard_mode_value=script_guard_mode(
                        self.repositories,
                        save_id=save_id,
                    ),
                )
            except ProviderError as exc:
                if not provider_error_is_model_not_found(exc):
                    raise
                return await recover_tool_call_shape_with_structured_output(
                    error=exc,
                    task="context_update",
                    provider=self.provider_name,
                    model_id=self.model_id,
                    structured_run=lambda: self._propose_clusters_structured(
                        save_id=save_id,
                        memories=memories,
                    ),
                )

        return await self._propose_clusters_structured(
            save_id=save_id,
            memories=memories,
        )

    async def _propose_clusters_structured(
        self,
        *,
        save_id: str,
        memories: tuple[MemoryRecord, ...],
    ) -> tuple[MemoryConsolidationCluster, ...]:
        if not isinstance(self.provider, StructuredOutputProvider):
            raise ValueError("Memory consolidation provider lacks structured output")
        structured_request = request_with_openrouter_routing(
            self.repositories,
            StructuredOutputRequest(
                provider=self.provider_name,
                model_id=self.model_id,
                schema_name="memory_consolidation",
                schema=_memory_consolidation_schema(memories),
                messages=_memory_consolidation_messages(memories),
                temperature=0.0,
            ),
            task="context_update",
            save_id=save_id,
        )
        if (
            self.prompt_inspection_store is not None
            and self.inspection_message_id is not None
        ):
            self.prompt_inspection_store.capture_structured_request(
                    message_id=self.inspection_message_id,
                    kind="memory_consolidation",
                    title="Memory consolidation",
                    request=structured_request,
                )
        if self.providers is not None:
            response = await structured_output_with_fallback(
                repositories=self.repositories,
                providers=self.providers,
                request=structured_request,
                task="context_update",
                save_id=save_id,
            )
        else:
            response = await self.provider.generate_structured_output(
                budget_structured_output_request(
                    self.repositories,
                    structured_request,
                    task="context_update",
                ),
            )
        return _clusters_from_structured_data(response.data)

    def _apply_clusters(
        self,
        *,
        save_id: str,
        memories: tuple[MemoryRecord, ...],
        clusters: tuple[MemoryConsolidationCluster, ...],
    ) -> MemoryConsolidationResult:
        memories_by_id = {memory.id: memory for memory in memories}
        archived_ids: set[str] = set()
        rewritten_count = 0
        archived_count = 0
        rejected_count = 0
        mode = script_guard_mode(self.repositories, save_id=save_id)
        for cluster in clusters:
            canonical = memories_by_id.get(cluster.canonical_memory_id)
            merged = [
                memories_by_id[memory_id]
                for memory_id in cluster.merged_memory_ids
                if memory_id in memories_by_id
            ]
            if (
                canonical is None
                or canonical.id in archived_ids
                or cluster.confidence < MEMORY_CONSOLIDATION_MIN_CONFIDENCE
                or not cluster.body.strip()
                or not merged
                or len(merged) != len(set(cluster.merged_memory_ids))
                or any(memory.id in archived_ids for memory in merged)
                or any(memory.id == canonical.id for memory in merged)
                or any(memory.save_id != save_id for memory in (canonical, *merged))
                or _cluster_is_noop(canonical, cluster)
            ):
                rejected_count += 1
                continue
            cluster_violations = _cluster_script_policy_violations(
                cluster,
                source_memories=(canonical, *merged),
                mode=mode,
            )
            if cluster_violations:
                rejected_count += 1
                continue
            source_message_ids = _union_memory_source_ids((canonical, *merged))
            source_observation_ids = list(
                dict.fromkeys(
                    observation_id
                    for memory in (canonical, *merged)
                    for observation_id in memory.source_observation_ids
                )
            )
            before = _memory_audit_value(canonical)
            updated = self.repositories.update_memory(
                memory_id=canonical.id,
                body=cluster.body.strip(),
                tags=list(cluster.tags),
                importance=cluster.importance,
                source_message_ids=source_message_ids,
                source_observation_ids=source_observation_ids,
            )
            self.repositories.add_context_update_audit(
                save_id=save_id,
                operation="memory_consolidation_rewritten",
                entity_type="memory",
                entity_id=canonical.id,
                field_path="*",
                before=before,
                after=_memory_audit_value(updated),
                reason=cluster.reason,
                confidence=cluster.confidence,
                source_message_ids=source_message_ids,
            )
            rewritten_count += 1
            for memory in merged:
                self.repositories.archive_memory(memory.id)
                archived_ids.add(memory.id)
                self.repositories.add_context_update_audit(
                    save_id=save_id,
                    operation="memory_consolidation_archived",
                    entity_type="memory",
                    entity_id=memory.id,
                    field_path="*",
                    before=_memory_audit_value(memory),
                    after=None,
                    reason=cluster.reason,
                    confidence=cluster.confidence,
                    source_message_ids=_union_memory_source_ids((memory,)),
                )
                archived_count += 1
        return MemoryConsolidationResult(
            save_id=save_id,
            active_memory_count=len(memories),
            proposed_cluster_count=len(clusters),
            rewritten_count=rewritten_count,
            archived_count=archived_count,
            rejected_count=rejected_count,
        )


def _memory_consolidation_schema(
    memories: tuple[MemoryRecord, ...],
) -> dict[str, object]:
    memory_ids = [memory.id for memory in memories]
    memory_id_schema: dict[str, object] = {"type": "string"}
    if memory_ids:
        memory_id_schema["enum"] = memory_ids
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "clusters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "canonical_memory_id": memory_id_schema,
                        "merged_memory_ids": {
                            "type": "array",
                            "items": memory_id_schema,
                        },
                        "body": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "importance": {"type": "number", "minimum": 0, "maximum": 1},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "canonical_memory_id",
                        "merged_memory_ids",
                        "body",
                        "tags",
                        "importance",
                        "confidence",
                        "reason",
                    ],
                },
            }
        },
        "required": ["clusters"],
    }


def _cluster_script_policy_violations(
    cluster: MemoryConsolidationCluster,
    *,
    source_memories: tuple[MemoryRecord, ...],
    mode: str,
) -> tuple[ScriptPolicyViolation, ...]:
    allowed_scripts = allowed_generated_scripts(
        memory.body for memory in source_memories
    )
    values: tuple[tuple[str, object], ...] = (
        ("body", cluster.body),
        ("tags", cluster.tags),
        ("reason", cluster.reason),
    )
    violations: list[ScriptPolicyViolation] = []
    for field_name, value in values:
        violations.extend(
            object_text_script_violations(
                value,
                allowed_scripts=allowed_scripts,
                mode=mode,
                field_name=field_name,
            )
        )
    return tuple(violations)


def _memory_consolidation_messages(
    memories: tuple[MemoryRecord, ...],
) -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(
            role="system",
            body=(
                "Find duplicate or noisy Bragi memory rows that can be merged. "
                "Use the enforced schema. Emit only high-confidence clusters. "
                "Keep one canonical memory id, rewrite it into a concise durable "
                "memory, and list duplicate memory ids to archive. Dossier-style "
                "character relationship memories may use tags like dossier, "
                "relationship, and character:<name>."
            ),
        ),
        ChatMessage(role="user", body=_memory_registry_text(memories)),
    )


def _memory_consolidation_tool_messages(
    memories: tuple[MemoryRecord, ...],
) -> tuple[ToolCallMessage, ...]:
    messages = _memory_consolidation_messages(memories)
    return tuple(
        ToolCallMessage(
            role=message.role,
            body=message.body.replace(
                "Use the enforced schema.",
                (
                    "Use the merge_memory_cluster tool instead of prose. Call it "
                    "once per high-confidence duplicate or noisy cluster. Make "
                    "no tool calls when no memories should be merged."
                ),
            ),
            speaker_name=message.speaker_name,
        )
        for message in messages
    )


def _memory_consolidation_tool_definitions(
    memories: tuple[MemoryRecord, ...],
) -> tuple[ToolDefinition, ...]:
    return (
        ToolDefinition(
            name="merge_memory_cluster",
            description="Merge duplicate memories into one canonical rewritten memory.",
            parameters=_memory_consolidation_tool_schema(memories),
        ),
    )


def _memory_consolidation_tool_schema(
    memories: tuple[MemoryRecord, ...],
) -> dict[str, object]:
    memory_ids = [memory.id for memory in memories]
    memory_id_schema: dict[str, object] = {
        "type": "string",
        "enum": memory_ids,
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "canonical_memory_id": memory_id_schema,
            "merged_memory_ids": {
                "type": "array",
                "items": memory_id_schema,
            },
            "body": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "importance": {"type": "number", "minimum": 0, "maximum": 1},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
        "required": [
            "canonical_memory_id",
            "merged_memory_ids",
            "body",
            "tags",
            "importance",
            "confidence",
            "reason",
        ],
    }


async def _memory_clusters_with_tool_fallback(
    *,
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    provider: ToolCallProvider,
    request: ToolCallRequest,
    save_id: str,
    memories: tuple[MemoryRecord, ...],
    script_guard_mode_value: str,
) -> tuple[MemoryConsolidationCluster, ...]:
    try:
        return await _memory_clusters_with_tool_feedback(
            repositories=repositories,
            provider=provider,
            request=request,
            memories=memories,
            script_guard_mode_value=script_guard_mode_value,
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
                task="context_update",
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
                task="context_update",
                reason=reason,
            )
            raise provider_error_with_fallback_skipped_reason(exc, reason) from exc
        log_event(
            "provider.tool_call_fallback_started",
            provider=fallback_request.provider,
            model=fallback_request.model_id,
            task="context_update",
        )
        try:
            return await _memory_clusters_with_tool_feedback(
                repositories=repositories,
                provider=fallback_provider,
                request=fallback_request,
                memories=memories,
                script_guard_mode_value=script_guard_mode_value,
            )
        except ProviderError as fallback_exc:
            raise provider_error_with_fallback_attempted(
                fallback_exc,
                provider=fallback_request.provider,
                model_id=fallback_request.model_id,
            ) from fallback_exc


async def _memory_clusters_with_tool_feedback(
    *,
    repositories: PersistenceRepositories,
    provider: ToolCallProvider,
    request: ToolCallRequest,
    memories: tuple[MemoryRecord, ...],
    script_guard_mode_value: str,
) -> tuple[MemoryConsolidationCluster, ...]:
    messages = list(request.messages)
    tool_schemas = {tool.name: tool.parameters for tool in request.tools}
    memories_by_id = {memory.id: memory for memory in memories}
    clusters: list[MemoryConsolidationCluster] = []
    accepted_ids: set[tuple[str, tuple[str, ...]]] = set()
    last_errors: list[str] = []
    max_attempt_count = configured_max_attempts(repositories)

    for _turn in range(max_attempt_count):
        turn_request = budget_tool_call_request(
            repositories,
            replace(request, messages=tuple(messages)),
            task="context_update",
        )
        response = await provider.generate_tool_calls(turn_request)
        errors: list[str] = []
        tool_results: list[tuple[ProviderToolCall, dict[str, str]]] = []
        for call in response.tool_calls:
            accepted, result, cluster = _validate_memory_cluster_tool_call(
                call,
                tool_schemas=tool_schemas,
                memories_by_id=memories_by_id,
                script_guard_mode_value=script_guard_mode_value,
            )
            if accepted:
                if cluster is not None:
                    key = (cluster.canonical_memory_id, cluster.merged_memory_ids)
                    if key not in accepted_ids:
                        accepted_ids.add(key)
                        clusters.append(cluster)
                tool_results.append((call, accepted_tool_result()))
                continue
            errors.append(result["error"])
            tool_results.append((call, result))

        if not errors:
            return tuple(clusters)

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
            "Memory consolidation tool-call validation failed after feedback: "
            + "; ".join(last_errors)
        ),
    )


def _validate_memory_cluster_tool_call(
    call: ProviderToolCall,
    *,
    tool_schemas: dict[str, dict[str, object]],
    memories_by_id: dict[str, MemoryRecord],
    script_guard_mode_value: str = DEFAULT_SCRIPT_GUARD_MODE,
) -> tuple[bool, dict[str, str], MemoryConsolidationCluster | None]:
    schema = tool_schemas.get(call.name)
    if schema is None:
        return _invalid_memory_cluster_tool_call(f"Unknown tool name: {call.name}")
    arguments, parse_error = parse_tool_arguments_json(call.arguments_json)
    if parse_error is not None or arguments is None:
        return _invalid_memory_cluster_tool_call(
            parse_error or "Tool arguments must be a JSON object"
        )
    shape_error = validate_tool_arguments_shape(arguments, schema=schema)
    if shape_error is not None:
        return _invalid_memory_cluster_tool_call(shape_error)
    cluster = _cluster_from_data(arguments)
    if cluster.canonical_memory_id not in memories_by_id:
        return _invalid_memory_cluster_tool_call(
            "canonical_memory_id is not an active memory: "
            f"{cluster.canonical_memory_id}"
        )
    missing_merged = [
        memory_id
        for memory_id in cluster.merged_memory_ids
        if memory_id not in memories_by_id
    ]
    if missing_merged:
        return _invalid_memory_cluster_tool_call(
            f"merged_memory_id is not an active memory: {missing_merged[0]}"
        )
    if len(cluster.merged_memory_ids) != len(set(cluster.merged_memory_ids)):
        return _invalid_memory_cluster_tool_call("merged_memory_ids must be unique")
    if cluster.canonical_memory_id in set(cluster.merged_memory_ids):
        return _invalid_memory_cluster_tool_call(
            "canonical_memory_id must not appear in merged_memory_ids"
        )
    canonical = memories_by_id[cluster.canonical_memory_id]
    merged = tuple(memories_by_id[memory_id] for memory_id in cluster.merged_memory_ids)
    violations = _cluster_script_policy_violations(
        cluster,
        source_memories=(canonical, *merged),
        mode=script_guard_mode_value,
    )
    if violations:
        return _invalid_memory_cluster_tool_call(
            summarize_script_policy_violations(violations)
        )
    return True, accepted_tool_result(), cluster


def _invalid_memory_cluster_tool_call(
    error: str,
) -> tuple[bool, dict[str, str], None]:
    return False, invalid_tool_result(error), None


def _memory_registry_text(memories: tuple[MemoryRecord, ...]) -> str:
    if not memories:
        return "Active memories: none"
    lines = ["Active memories:"]
    for memory in memories:
        lines.append(
            f"- {memory.id}: {memory.body} "
            f"(tags={', '.join(memory.tags)}; importance={memory.importance:g}; "
            f"sources={len(memory.source_message_ids)})"
        )
    return "\n".join(lines)


def _memory_batches(
    memories: tuple[MemoryRecord, ...],
    batch_size: int,
) -> tuple[tuple[MemoryRecord, ...], ...]:
    size = max(1, batch_size)
    return tuple(
        memories[index : index + size] for index in range(0, len(memories), size)
    )


def _merge_results(
    *,
    save_id: str,
    active_memory_count: int,
    batch_count: int,
    batch_results: tuple[MemoryConsolidationResult, ...],
) -> MemoryConsolidationResult:
    return MemoryConsolidationResult(
        save_id=save_id,
        active_memory_count=active_memory_count,
        proposed_cluster_count=sum(
            result.proposed_cluster_count for result in batch_results
        ),
        rewritten_count=sum(result.rewritten_count for result in batch_results),
        archived_count=sum(result.archived_count for result in batch_results),
        rejected_count=sum(result.rejected_count for result in batch_results),
        batch_count=batch_count,
        completed_batch_count=len(batch_results),
    )


def _clusters_from_structured_data(
    data: dict[str, object],
) -> tuple[MemoryConsolidationCluster, ...]:
    raw_clusters = data.get("clusters", [])
    if not isinstance(raw_clusters, list):
        raise ValueError("Structured memory consolidation clusters must be a list")
    return tuple(_cluster_from_data(item) for item in raw_clusters)


def _cluster_from_data(value: object) -> MemoryConsolidationCluster:
    if not isinstance(value, dict):
        raise ValueError("Structured memory consolidation cluster must be an object")
    raw_merged = value.get("merged_memory_ids", [])
    raw_tags = value.get("tags", [])
    if not isinstance(raw_merged, list) or not all(
        isinstance(item, str) for item in raw_merged
    ):
        raise ValueError("merged_memory_ids must be a string list")
    if not isinstance(raw_tags, list) or not all(
        isinstance(item, str) for item in raw_tags
    ):
        raise ValueError("tags must be a string list")
    return MemoryConsolidationCluster(
        canonical_memory_id=str(value.get("canonical_memory_id", "")),
        merged_memory_ids=tuple(raw_merged),
        body=str(value.get("body", "")),
        tags=tuple(dict.fromkeys(tag.strip() for tag in raw_tags if tag.strip())),
        importance=_confidence(value.get("importance")),
        confidence=_confidence(value.get("confidence")),
        reason=str(value.get("reason", "")).strip(),
    )


def _confidence(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return 0.0
    return min(max(float(value), 0.0), 1.0)


def _cluster_is_noop(
    canonical: MemoryRecord,
    cluster: MemoryConsolidationCluster,
) -> bool:
    return (
        canonical.body.strip() == cluster.body.strip()
        and canonical.tags == list(cluster.tags)
        and float(canonical.importance) == cluster.importance
    )


def _union_memory_source_ids(memories: tuple[MemoryRecord, ...]) -> list[str]:
    source_ids: list[str] = []
    for memory in memories:
        if memory.source_message_id:
            source_ids.append(memory.source_message_id)
        source_ids.extend(memory.source_message_ids)
    return list(dict.fromkeys(source_id for source_id in source_ids if source_id))


def _memory_audit_value(memory: MemoryRecord) -> dict[str, object]:
    return {
        "id": memory.id,
        "body": memory.body,
        "tags": list(memory.tags),
        "importance": memory.importance,
        "source_message_id": memory.source_message_id,
        "source_message_ids": list(memory.source_message_ids),
    }


def _elapsed_ms(started_at: float) -> int:
    return int((perf_counter() - started_at) * 1000)
