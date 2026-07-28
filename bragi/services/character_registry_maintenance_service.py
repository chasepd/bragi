"""Structured duplicate and erroneous character maintenance."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Protocol

from bragi.app_logging import exception_log_fields, log_error_event, log_event
from bragi.persistence.models import (
    CharacterRecord,
    JobRecord,
    MessageRecord,
    SaveRecord,
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
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.providers.structured_schema import normalize_strict_json_schema
from bragi.redaction import redact_text
from bragi.retry_policy import MODEL_OUTPUT_MAX_ATTEMPTS
from bragi.services.character_locks import normalize_character_locked_fields
from bragi.services.character_registry_service import (
    CharacterRegistryEdits,
    CharacterRegistryRow,
    CharacterRegistryService,
)
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

CHARACTER_MAINTENANCE_TURN_CADENCE = 3
CHARACTER_MAINTENANCE_CONFIDENCE_THRESHOLD = 0.88
RECENT_MESSAGE_LIMIT = 24
CHARACTER_MAINTENANCE_MESSAGE_OVERLAP = 4
CHARACTER_MAINTENANCE_MEMORY_TEXT_LIMIT = 80
MAX_CHARACTER_MAINTENANCE_TOOL_FEEDBACK_TURNS = MODEL_OUTPUT_MAX_ATTEMPTS - 1


@dataclass(frozen=True)
class CharacterMaintenanceDecision:
    operation: str
    character_id: str
    target_character_id: str | None = None
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class CharacterMaintenanceResult:
    proposed: tuple[CharacterMaintenanceDecision, ...]
    applied: tuple[CharacterMaintenanceDecision, ...]
    rejected: tuple[CharacterMaintenanceDecision, ...]
    skipped_reason: str | None = None


class CharacterMaintenanceRunner(Protocol):
    async def maintain_if_due(
        self,
        *,
        save_id: str,
        force: bool = False,
    ) -> CharacterMaintenanceResult: ...


class CharacterRegistryMaintenanceService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        providers: dict[str, ProviderClient],
    ) -> None:
        self.repositories = repositories
        self.providers = providers
        self.jobs = JobLifecycleService(repositories=repositories)

    async def maintain_if_due(
        self,
        *,
        save_id: str,
        force: bool = False,
    ) -> CharacterMaintenanceResult:
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="character_registry_maintenance",
        )
        if preference is None:
            return CharacterMaintenanceResult(
                proposed=(),
                applied=(),
                rejected=(),
                skipped_reason=(
                    "No character-registry maintenance model preference configured"
                ),
            )
        characters = tuple(self.repositories.list_characters(save_id))
        if not characters:
            return CharacterMaintenanceResult(
                proposed=(),
                applied=(),
                rejected=(),
                skipped_reason="No active characters",
            )
        actionable_characters = tuple(
            character
            for character in characters
            if not character.protected_from_maintenance
        )
        if not actionable_characters:
            return CharacterMaintenanceResult(
                proposed=(),
                applied=(),
                rejected=(),
                skipped_reason="No unprotected characters eligible for maintenance",
            )
        if not force and not _turn_count_is_due(self.repositories, save_id):
            return CharacterMaintenanceResult(
                proposed=(),
                applied=(),
                rejected=(),
                skipped_reason="Character-registry maintenance cadence not due",
            )
        job = self.jobs.create_running(
            save_id=save_id,
            type="character_registry_maintenance",
            payload={"cadence": CHARACTER_MAINTENANCE_TURN_CADENCE},
            collect_provider_diagnostics=True,
        )
        started_at = perf_counter()
        try:
            provider = self.providers.get(preference.provider)
            if provider is None:
                raise ValueError(
                    "Character-registry maintenance provider is unavailable"
                )
            if known_model_is_unavailable(
                self.repositories,
                provider=preference.provider,
                model_id=preference.model_id,
            ):
                raise ValueError(
                    "Character-registry maintenance model is unavailable: "
                    f"{preference.model_id}"
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
                    "Character-registry maintenance model does not advertise "
                    "structured output or tool calling"
                )
            save = self.repositories.get_save(save_id)
            recent_messages = _recent_messages_for_maintenance(
                self.repositories,
                save_id=save_id,
            )
            memory_texts = tuple(
                f"{memory.id}: {memory.body}"
                for memory in self.repositories.list_memories(save_id)
            )
            if supports_tool_calling:
                select_with_tools = (
                    _select_character_maintenance_decisions_with_tool_calls
                )
                proposed = await select_with_tools(
                    repositories=self.repositories,
                    providers=self.providers,
                    provider=provider,
                    provider_name=preference.provider,
                    model_id=preference.model_id,
                    save_id=save_id,
                    save=save,
                    characters=characters,
                    recent_messages=recent_messages,
                    memory_texts=memory_texts,
                )
            else:
                proposed = await _select_character_maintenance_decisions(
                    repositories=self.repositories,
                    providers=self.providers,
                    provider_name=preference.provider,
                    model_id=preference.model_id,
                    save_id=save_id,
                    save=save,
                    characters=characters,
                    recent_messages=recent_messages,
                    memory_texts=memory_texts,
                )
            result = self._apply_decisions(
                save_id=save_id,
                characters=characters,
                proposed=proposed,
            )
        except Exception as exc:
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
        self.jobs.succeed(
            job.id,
            result={
                "proposed_count": len(result.proposed),
                "applied_count": len(result.applied),
                "rejected_count": len(result.rejected),
                "decisions": [decision.operation for decision in result.applied],
                **_message_window_metadata(recent_messages),
            },
        )
        log_event(
            "job.succeeded",
            job_id=job.id,
            job_type=job.type,
            save_id=save_id,
            duration_ms=_elapsed_ms(started_at),
            proposed_count=len(result.proposed),
            applied_count=len(result.applied),
            rejected_count=len(result.rejected),
        )
        return result

    def _apply_decisions(
        self,
        *,
        save_id: str,
        characters: tuple[CharacterRecord, ...],
        proposed: tuple[CharacterMaintenanceDecision, ...],
    ) -> CharacterMaintenanceResult:
        character_ids = {character.id for character in characters}
        protected_ids = {
            character.id
            for character in characters
            if character.protected_from_maintenance
        }
        rows_by_id = {
            character.id: _row_for_character(character) for character in characters
        }
        rejected: list[CharacterMaintenanceDecision] = []
        accepted: list[CharacterMaintenanceDecision] = []
        merge_sources: set[str] = set()
        merge_targets: set[str] = set()
        delete_ids: set[str] = set()
        chained_merge_ids = _chained_merge_ids(
            proposed,
            character_ids=character_ids,
        )
        edited_rows: list[CharacterRegistryRow] = []
        touched_ids: set[str] = set()
        active_after_edits = set(character_ids)
        for decision in proposed:
            if not _valid_decision(
                decision,
                character_ids=character_ids,
                protected_ids=protected_ids,
            ):
                rejected.append(decision)
                continue
            if decision.character_id in touched_ids:
                rejected.append(decision)
                continue
            if decision.operation == "merge":
                assert decision.target_character_id is not None
                if (
                    decision.character_id in chained_merge_ids
                    or decision.target_character_id in chained_merge_ids
                    or decision.target_character_id in delete_ids
                ):
                    rejected.append(decision)
                    continue
                if decision.target_character_id in merge_sources:
                    rejected.append(decision)
                    continue
                merge_sources.add(decision.character_id)
                merge_targets.add(decision.target_character_id)
                active_after_edits.discard(decision.character_id)
                edited_rows.append(
                    replace(
                        rows_by_id[decision.character_id],
                        merge_into_character_id=decision.target_character_id,
                    )
                )
            elif decision.operation == "delete":
                if (
                    decision.character_id in merge_targets
                    or len(active_after_edits - {decision.character_id}) == 0
                ):
                    rejected.append(decision)
                    continue
                delete_ids.add(decision.character_id)
                active_after_edits.discard(decision.character_id)
                edited_rows.append(
                    replace(
                        rows_by_id[decision.character_id],
                        archived=True,
                    )
                )
            else:
                rejected.append(decision)
                continue
            touched_ids.add(decision.character_id)
            accepted.append(decision)
        if edited_rows:
            CharacterRegistryService(self.repositories).apply_edits(
                CharacterRegistryEdits(characters=tuple(edited_rows)),
                active_save_id=save_id,
            )
        return CharacterMaintenanceResult(
            proposed=proposed,
            applied=tuple(accepted),
            rejected=tuple(rejected),
        )


async def _select_character_maintenance_decisions(
    *,
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    provider_name: str,
    model_id: str,
    save_id: str,
    save: SaveRecord | None,
    characters: tuple[CharacterRecord, ...],
    recent_messages: tuple[MessageRecord, ...],
    memory_texts: tuple[str, ...],
) -> tuple[CharacterMaintenanceDecision, ...]:
    started_at = perf_counter()
    request = request_with_openrouter_routing(
        repositories,
        StructuredOutputRequest(
            provider=provider_name,
            model_id=model_id,
            schema_name="character_registry_maintenance_decisions",
            schema=_character_maintenance_schema(characters),
            messages=_character_maintenance_messages(
                save=save,
                characters=characters,
                recent_messages=recent_messages,
                memory_texts=memory_texts,
            ),
            temperature=0.0,
        ),
        task="character_registry_maintenance",
        save_id=save_id,
    )
    try:
        response = await structured_output_with_fallback(
            repositories=repositories,
            providers=providers,
            request=request,
            task="character_registry_maintenance",
            save_id=save_id,
        )
    except Exception as exc:
        log_error_event(
            "provider.structured_output_failed",
            provider=provider_name,
            model=model_id,
            task="character_registry_maintenance",
            duration_ms=_elapsed_ms(started_at),
            candidate_count=len(characters),
            **exception_log_fields(exc),
        )
        raise
    log_event(
        "provider.structured_output_succeeded",
        provider=response.provider,
        model=response.model_id,
        task="character_registry_maintenance",
        duration_ms=_elapsed_ms(started_at),
        candidate_count=len(characters),
        token_usage=response.token_usage,
    )
    return _decisions_from_structured_data(response.data)


async def _select_character_maintenance_decisions_with_tool_calls(
    *,
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    provider: ProviderClient,
    provider_name: str,
    model_id: str,
    save_id: str,
    save: SaveRecord | None,
    characters: tuple[CharacterRecord, ...],
    recent_messages: tuple[MessageRecord, ...],
    memory_texts: tuple[str, ...],
) -> tuple[CharacterMaintenanceDecision, ...]:
    if not isinstance(provider, ToolCallProvider):
        raise ValueError(
            "Character-registry maintenance provider does not support tool calling"
        )
    started_at = perf_counter()
    request = request_with_openrouter_routing(
        repositories,
        ToolCallRequest(
            provider=provider_name,
            model_id=model_id,
            messages=_character_maintenance_tool_messages(
                save=save,
                characters=characters,
                recent_messages=recent_messages,
                memory_texts=memory_texts,
            ),
            tools=_character_maintenance_tool_definitions(characters),
            temperature=0.0,
        ),
        task="character_registry_maintenance",
        save_id=save_id,
    )
    try:
        result = await _select_character_maintenance_with_tool_fallback(
            repositories=repositories,
            providers=providers,
            provider=provider,
            request=request,
            save_id=save_id,
            characters=characters,
        )
    except Exception as exc:
        log_error_event(
            "provider.tool_call_failed",
            provider=provider_name,
            model=model_id,
            task="character_registry_maintenance",
            duration_ms=_elapsed_ms(started_at),
            candidate_count=len(characters),
            **exception_log_fields(exc),
        )
        raise
    log_event(
        "provider.tool_call_succeeded",
        provider=provider_name,
        model=model_id,
        task="character_registry_maintenance",
        duration_ms=_elapsed_ms(started_at),
        candidate_count=len(characters),
    )
    return result


async def _select_character_maintenance_with_tool_fallback(
    *,
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    provider: ToolCallProvider,
    request: ToolCallRequest,
    save_id: str,
    characters: tuple[CharacterRecord, ...],
) -> tuple[CharacterMaintenanceDecision, ...]:
    try:
        return await _select_character_maintenance_with_tool_feedback(
            repositories=repositories,
            provider=provider,
            request=request,
            characters=characters,
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
                task="character_registry_maintenance",
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
                task="character_registry_maintenance",
                reason=reason,
            )
            raise provider_error_with_fallback_skipped_reason(exc, reason) from exc
        log_event(
            "provider.tool_call_fallback_started",
            provider=fallback_request.provider,
            model=fallback_request.model_id,
            task="character_registry_maintenance",
        )
        try:
            return await _select_character_maintenance_with_tool_feedback(
                repositories=repositories,
                provider=fallback_provider,
                request=fallback_request,
                characters=characters,
            )
        except ProviderError as fallback_exc:
            raise provider_error_with_fallback_attempted(
                fallback_exc,
                provider=fallback_request.provider,
                model_id=fallback_request.model_id,
            ) from fallback_exc


async def _select_character_maintenance_with_tool_feedback(
    *,
    repositories: PersistenceRepositories,
    provider: ToolCallProvider,
    request: ToolCallRequest,
    characters: tuple[CharacterRecord, ...],
) -> tuple[CharacterMaintenanceDecision, ...]:
    messages = list(request.messages)
    tool_schemas = {tool.name: tool.parameters for tool in request.tools}
    characters_by_id = {character.id: character for character in characters}
    protected_ids = {
        character.id for character in characters if character.protected_from_maintenance
    }
    decisions: list[CharacterMaintenanceDecision] = []
    accepted_keys: set[tuple[str, str, str | None]] = set()
    last_errors: list[str] = []

    for _turn in range(MAX_CHARACTER_MAINTENANCE_TOOL_FEEDBACK_TURNS + 1):
        turn_request = budget_tool_call_request(
            repositories,
            replace(request, messages=tuple(messages)),
            task="character_registry_maintenance",
        )
        response = await provider.generate_tool_calls(turn_request)
        errors: list[str] = []
        tool_results: list[tuple[ProviderToolCall, dict[str, str]]] = []
        for call in response.tool_calls:
            accepted, result, decision = _validate_character_maintenance_tool_call(
                call,
                tool_schemas=tool_schemas,
                characters_by_id=characters_by_id,
                protected_ids=protected_ids,
            )
            if accepted:
                if decision is not None:
                    key = (
                        decision.operation,
                        decision.character_id,
                        decision.target_character_id,
                    )
                    if key not in accepted_keys:
                        accepted_keys.add(key)
                        decisions.append(decision)
                tool_results.append((call, accepted_tool_result()))
                continue
            errors.append(result["error"])
            tool_results.append((call, result))

        if not errors:
            return tuple(decisions)

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
            "Character maintenance tool-call validation failed after feedback: "
            + "; ".join(last_errors)
        ),
    )


def _validate_character_maintenance_tool_call(
    call: ProviderToolCall,
    *,
    tool_schemas: dict[str, dict[str, object]],
    characters_by_id: dict[str, CharacterRecord],
    protected_ids: set[str],
) -> tuple[bool, dict[str, str], CharacterMaintenanceDecision | None]:
    schema = tool_schemas.get(call.name)
    if schema is None:
        return _invalid_character_maintenance_tool_call(
            f"Unknown tool name: {call.name}"
        )
    arguments, parse_error = parse_tool_arguments_json(call.arguments_json)
    if parse_error is not None or arguments is None:
        return _invalid_character_maintenance_tool_call(
            parse_error or "Tool arguments must be a JSON object"
        )
    shape_error = validate_tool_arguments_shape(
        arguments,
        schema=schema,
        skip_enum_fields=frozenset({"character_id"}),
    )
    if shape_error is not None:
        return _invalid_character_maintenance_tool_call(shape_error)
    character_id = str(arguments.get("character_id", "")).strip()
    if character_id not in characters_by_id:
        return _invalid_character_maintenance_tool_call(
            f"character_id is not an active character: {character_id}"
        )
    if character_id in protected_ids:
        return _invalid_character_maintenance_tool_call(
            f"character_id is protected from maintenance: {character_id}"
        )
    confidence = _confidence(arguments.get("confidence"))
    reason = str(arguments.get("reason", "")).strip()
    if call.name == "delete_character_entry":
        return (
            True,
            accepted_tool_result(),
            CharacterMaintenanceDecision(
                operation="delete",
                character_id=character_id,
                confidence=confidence,
                reason=reason,
            ),
        )
    target_character_id = str(arguments.get("target_character_id", "")).strip()
    if target_character_id not in characters_by_id:
        return _invalid_character_maintenance_tool_call(
            f"target_character_id is not an active character: {target_character_id}"
        )
    if target_character_id == character_id:
        return _invalid_character_maintenance_tool_call(
            "target_character_id must differ from character_id"
        )
    return (
        True,
        accepted_tool_result(),
        CharacterMaintenanceDecision(
            operation="merge",
            character_id=character_id,
            target_character_id=target_character_id,
            confidence=confidence,
            reason=reason,
        ),
    )


def _invalid_character_maintenance_tool_call(
    error: str,
) -> tuple[bool, dict[str, str], None]:
    return False, invalid_tool_result(error), None


def _character_maintenance_schema(
    characters: tuple[CharacterRecord, ...],
) -> dict[str, object]:
    character_ids = [character.id for character in characters]
    actionable_character_ids = [
        character.id
        for character in characters
        if not character.protected_from_maintenance
    ]
    return normalize_strict_json_schema({
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "operation": {"type": "string", "enum": ["merge", "delete"]},
                        "character_id": {
                            "type": "string",
                            "enum": actionable_character_ids,
                        },
                        "target_character_id": {
                            "type": "string",
                            "enum": character_ids,
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reason": {"type": "string"},
                    },
                    "required": ["operation", "character_id", "confidence", "reason"],
                },
            },
        },
        "required": ["decisions"],
    })


def _character_maintenance_messages(
    *,
    save: SaveRecord | None,
    characters: tuple[CharacterRecord, ...],
    recent_messages: tuple[MessageRecord, ...],
    memory_texts: tuple[str, ...],
) -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(
            role="system",
            body=(
                "Review the Bragi character registry for duplicate characters "
                "and erroneous non-character entries. Use only the enforced "
                "response schema. Emit merge decisions only for high-confidence "
                "same-character duplicates, and delete decisions only for entries "
                "that are clearly malformed, concepts, objects, instructions, or "
                "otherwise not real characters. Characters marked protected may "
                "be used as merge targets, but must never be selected as the "
                "character_id for merge or delete decisions. Prefer no decision "
                "when uncertain."
            ),
        ),
        ChatMessage(
            role="user",
            body="\n\n".join(
                (
                    _save_text(save),
                    _characters_text(characters),
                    _memories_text(memory_texts),
                    _recent_messages_text(recent_messages),
                )
            ),
        ),
    )


def _character_maintenance_tool_messages(
    *,
    save: SaveRecord | None,
    characters: tuple[CharacterRecord, ...],
    recent_messages: tuple[MessageRecord, ...],
    memory_texts: tuple[str, ...],
) -> tuple[ToolCallMessage, ...]:
    messages = _character_maintenance_messages(
        save=save,
        characters=characters,
        recent_messages=recent_messages,
        memory_texts=memory_texts,
    )
    return tuple(
        ToolCallMessage(
            role=message.role,
            body=message.body.replace(
                "Use only the enforced response schema.",
                (
                    "Use the provided merge_character and "
                    "delete_character_entry tools instead of prose."
                ),
            ),
            speaker_name=message.speaker_name,
        )
        for message in messages
    )


def _character_maintenance_tool_definitions(
    characters: tuple[CharacterRecord, ...],
) -> tuple[ToolDefinition, ...]:
    schemas = _character_maintenance_tool_schemas(characters)
    return (
        ToolDefinition(
            name="merge_character",
            description="Merge one duplicate character into a target character.",
            parameters=schemas["merge_character"],
        ),
        ToolDefinition(
            name="delete_character_entry",
            description="Delete one erroneous non-character registry entry.",
            parameters=schemas["delete_character_entry"],
        ),
    )


def _character_maintenance_tool_schemas(
    characters: tuple[CharacterRecord, ...],
) -> dict[str, dict[str, object]]:
    character_ids = [character.id for character in characters]
    actionable_character_ids = [
        character.id
        for character in characters
        if not character.protected_from_maintenance
    ]
    base_properties: dict[str, object] = {
        "character_id": {
            "type": "string",
            "enum": actionable_character_ids,
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    }
    return {
        "merge_character": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                **base_properties,
                "target_character_id": {
                    "type": "string",
                    "enum": character_ids,
                },
            },
            "required": [
                "character_id",
                "target_character_id",
                "confidence",
                "reason",
            ],
        },
        "delete_character_entry": {
            "type": "object",
            "additionalProperties": False,
            "properties": base_properties,
            "required": ["character_id", "confidence", "reason"],
        },
    }


def _decisions_from_structured_data(
    data: dict[str, object],
) -> tuple[CharacterMaintenanceDecision, ...]:
    raw_decisions = data.get("decisions", [])
    if not isinstance(raw_decisions, list):
        raise ValueError("Structured character maintenance decisions must be a list")
    decisions: list[CharacterMaintenanceDecision] = []
    for value in raw_decisions:
        if not isinstance(value, dict):
            raise ValueError(
                "Structured character maintenance decision must be an object"
            )
        operation = str(value.get("operation", "")).strip().lower()
        target = value.get("target_character_id")
        decisions.append(
            CharacterMaintenanceDecision(
                operation=operation,
                character_id=str(value.get("character_id", "")).strip(),
                target_character_id=(str(target).strip() if target else None),
                confidence=_confidence(value.get("confidence")),
                reason=str(value.get("reason", "")).strip(),
            )
        )
    return tuple(decisions)


def _valid_decision(
    decision: CharacterMaintenanceDecision,
    *,
    character_ids: set[str],
    protected_ids: set[str],
) -> bool:
    if decision.confidence < CHARACTER_MAINTENANCE_CONFIDENCE_THRESHOLD:
        return False
    if len(decision.reason.strip()) < 12:
        return False
    if decision.character_id not in character_ids:
        return False
    if decision.character_id in protected_ids:
        return False
    if decision.operation == "delete":
        return decision.target_character_id in {None, ""}
    if decision.operation == "merge":
        return (
            decision.target_character_id in character_ids
            and decision.target_character_id != decision.character_id
        )
    return False


def _chained_merge_ids(
    decisions: tuple[CharacterMaintenanceDecision, ...],
    *,
    character_ids: set[str],
) -> set[str]:
    merge_sources = {
        decision.character_id
        for decision in decisions
        if decision.operation == "merge"
        and _valid_decision(
            decision,
            character_ids=character_ids,
            protected_ids=set(),
        )
    }
    return {
        decision_id
        for decision in decisions
        if decision.operation == "merge"
        and _valid_decision(
            decision,
            character_ids=character_ids,
            protected_ids=set(),
        )
        and decision.target_character_id in merge_sources
        for decision_id in (decision.character_id, str(decision.target_character_id))
    }


def _row_for_character(character: CharacterRecord) -> CharacterRegistryRow:
    return CharacterRegistryRow(
        character_id=character.id,
        name=character.name,
        aliases_text=", ".join(character.aliases),
        role=character.role,
        age=character.age,
        known_state=character.known_state,
        met=character.met,
        appearance=character.appearance,
        visual_notes=character.visual_notes,
        current_clothing=character.current_clothing,
        personality=character.personality,
        voice=character.voice,
        texting_style=character.texting_style,
        relationships_json=_dump_json(character.relationships),
        goals=character.goals,
        motivations=character.motivations,
        current_intent=character.current_intent,
        boundaries=character.boundaries,
        attitude_toward_player=character.attitude_toward_player,
        cooperation_conditions=character.cooperation_conditions,
        status=character.status,
        location_id=character.location_id,
        private_notes=character.private_notes,
        source_message_id=character.source_message_id,
        locked_fields=tuple(normalize_character_locked_fields(character.locked_fields)),
        protected_from_maintenance=character.protected_from_maintenance,
        is_player_character=character.is_player_character,
    )


def _turn_count_is_due(
    repositories: PersistenceRepositories,
    save_id: str,
) -> bool:
    count_messages = getattr(repositories, "count_active_messages_by_role", None)
    if callable(count_messages):
        counts = count_messages(save_id, roles=("player", "narrator"))
        completed_turn_count = min(int(counts["player"]), int(counts["narrator"]))
    else:
        completed_turn_count = sum(
            1
            for message in repositories.list_messages(save_id)
            if message.role == "player"
        )
        completed_turn_count = min(
            completed_turn_count,
            sum(
                1
                for message in repositories.list_messages(save_id)
                if message.role == "narrator"
            ),
        )
    return bool(
        completed_turn_count > 0
        and completed_turn_count % CHARACTER_MAINTENANCE_TURN_CADENCE == 0
    )


def _recent_messages_for_maintenance(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
) -> tuple[MessageRecord, ...]:
    messages = tuple(repositories.list_messages(save_id))
    last_job = _last_successful_character_maintenance_job(
        repositories,
        save_id=save_id,
    )
    if last_job is None:
        return messages[-RECENT_MESSAGE_LIMIT:]

    anchor_message_id = _message_window_end_id(last_job)
    if anchor_message_id is None:
        return messages[-RECENT_MESSAGE_LIMIT:]

    for index, message in enumerate(messages):
        if message.id != anchor_message_id:
            continue
        start_index = max(0, index + 1 - CHARACTER_MAINTENANCE_MESSAGE_OVERLAP)
        return messages[start_index:]

    return messages[-RECENT_MESSAGE_LIMIT:]


def _last_successful_character_maintenance_job(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
) -> JobRecord | None:
    jobs = repositories.list_recent_jobs(
        save_id=save_id,
        types=("character_registry_maintenance",),
        statuses=("succeeded",),
        seconds=0,
        limit=1,
    )
    return jobs[0] if jobs else None


def _message_window_end_id(job: JobRecord) -> str | None:
    for source in (job.result, job.payload):
        if not isinstance(source, dict):
            continue
        value = source.get("context_end_message_id")
        if isinstance(value, str) and value:
            return value
    return None


def _message_window_metadata(
    messages: tuple[MessageRecord, ...],
) -> dict[str, object]:
    return {
        "context_message_count": len(messages),
        "context_start_message_id": messages[0].id if messages else None,
        "context_end_message_id": messages[-1].id if messages else None,
        "context_overlap_message_count": CHARACTER_MAINTENANCE_MESSAGE_OVERLAP,
    }


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


def _save_text(save: SaveRecord | None) -> str:
    return f"Save: {save.title}" if save is not None else "Save: unknown"


def _characters_text(characters: tuple[CharacterRecord, ...]) -> str:
    lines = ["Active characters:"]
    for character in characters:
        lines.append(
            "- "
            f"{character.id}: name={character.name}; "
            f"aliases={', '.join(character.aliases)}; "
            f"protected={character.protected_from_maintenance}; "
            f"role={character.role}; age={character.age}; "
            f"known_state={character.known_state}; "
            f"appearance={character.appearance}; "
            f"current_clothing={character.current_clothing}; "
            f"personality={character.personality}; "
            f"goals={character.goals}; current_intent={character.current_intent}; "
            f"boundaries={character.boundaries}; "
            f"cooperation_conditions={character.cooperation_conditions}; "
            f"status={character.status}"
        )
    return "\n".join(lines)


def _memories_text(memory_texts: tuple[str, ...]) -> str:
    if not memory_texts:
        return "Memories: none"
    omitted = max(0, len(memory_texts) - CHARACTER_MAINTENANCE_MEMORY_TEXT_LIMIT)
    selected = memory_texts[-CHARACTER_MAINTENANCE_MEMORY_TEXT_LIMIT:]
    lines = ["Memories:"]
    if omitted:
        lines.append(f"- {omitted} older memories omitted by payload cap")
    lines.extend(f"- {text}" for text in selected)
    return "\n".join(lines)


def _recent_messages_text(messages: tuple[MessageRecord, ...]) -> str:
    if not messages:
        return "Recent chronicle: none"
    return "Recent chronicle:\n" + "\n".join(
        f"- {message.id} [{message.role}] "
        f"{message.speaker_name or message.role}: {message.body}"
        for message in messages
    )


def _confidence(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return max(0.0, min(1.0, float(value)))
    return 0.0


def _dump_json(value: object) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _elapsed_ms(started_at: float) -> int:
    return max(0, int((perf_counter() - started_at) * 1000))
