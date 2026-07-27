"""Automated review for queued world-data suggestions."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Protocol, cast

from bragi.app_logging import exception_log_fields, log_error_event, log_event
from bragi.persistence.models import (
    CharacterTextMessageRecord,
    ContextUpdateSuggestionRecord,
    MessageRecord,
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
from bragi.redaction import redact_text
from bragi.services.character_text_world_update_service import (
    parse_character_text_source_ref,
)
from bragi.services.openrouter_routing_settings import request_with_openrouter_routing
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
from bragi.services.tool_call_helpers import (
    accepted_tool_result,
    append_tool_feedback_messages,
    invalid_tool_result,
    parse_tool_arguments_json,
    validate_tool_arguments_shape,
)
from bragi.services.world_data_service import WorldDataService

MAX_REVIEW_GROUPS = 24
MAX_SOURCE_MESSAGE_CHARS = 900
MAX_WORLD_SUGGESTION_REVIEW_TOOL_FEEDBACK_TURNS = 2
MAX_AUTOMATED_REVIEW_ATTEMPTS = 3
REVIEW_RETRY_DELAYS_SECONDS = (5 * 60, 30 * 60)


class SuggestionReviewProvider(Protocol):
    provider_name: str


@dataclass(frozen=True)
class SuggestionReviewDecision:
    review_id: str
    action: str
    reason: str


@dataclass(frozen=True)
class WorldSuggestionReviewResult:
    save_id: str
    reviewed_count: int
    applied_count: int
    rejected_count: int
    deferred_count: int
    error: str | None = None

    @property
    def changed_count(self) -> int:
        return self.applied_count + self.rejected_count

    def to_json(self) -> dict[str, object]:
        result: dict[str, object] = {
            "save_id": self.save_id,
            "reviewed_count": self.reviewed_count,
            "applied_count": self.applied_count,
            "rejected_count": self.rejected_count,
            "deferred_count": self.deferred_count,
        }
        if self.error:
            result["error"] = self.error
        return result


@dataclass(frozen=True)
class _SuggestionReviewGroup:
    review_id: str
    suggestions: tuple[ContextUpdateSuggestionRecord, ...]

    @property
    def first(self) -> ContextUpdateSuggestionRecord:
        return self.suggestions[0]

    @property
    def suggestion_ids(self) -> tuple[str, ...]:
        return tuple(suggestion.id for suggestion in self.suggestions)


class WorldSuggestionReviewService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        provider: SuggestionReviewProvider,
        provider_name: str,
        model_id: str,
        providers: dict[str, ProviderClient] | None = None,
        prefer_tool_calls: bool = False,
    ) -> None:
        self.repositories = repositories
        self.provider = provider
        self.provider_name = provider_name
        self.model_id = model_id
        self.providers = providers
        self.prefer_tool_calls = prefer_tool_calls

    async def review_pending(
        self,
        save_id: str,
        *,
        due_only: bool = False,
    ) -> WorldSuggestionReviewResult:
        groups = _pending_review_groups(
            self.repositories,
            save_id,
            due_only=due_only,
        )
        invalid_suggestion_ids = self._reject_invalid_suggestions(
            save_id=save_id,
            groups=groups,
        )
        preflight_rejected = len(invalid_suggestion_ids)
        groups = tuple(
            _SuggestionReviewGroup(
                review_id=members[0].id,
                suggestions=tuple(members),
            )
            for group in groups
            if (
                members := [
                    suggestion
                    for suggestion in group.suggestions
                    if suggestion.id not in invalid_suggestion_ids
                ]
            )
        )
        if not groups:
            return WorldSuggestionReviewResult(
                save_id=save_id,
                reviewed_count=0,
                applied_count=0,
                rejected_count=preflight_rejected,
                deferred_count=0,
            )
        groups = groups[:MAX_REVIEW_GROUPS]
        try:
            decisions = await self._review_groups(save_id=save_id, groups=groups)
        except Exception as exc:  # noqa: BLE001 - provider boundary
            error = redact_text(str(exc)) or exc.__class__.__name__
            self._defer_groups(save_id=save_id, groups=groups, reason=error)
            log_error_event(
                "world_suggestion_review.failed",
                save_id=save_id,
                provider=self.provider_name,
                model=self.model_id,
                **exception_log_fields(exc),
            )
            return WorldSuggestionReviewResult(
                save_id=save_id,
                reviewed_count=len(groups),
                applied_count=0,
                rejected_count=preflight_rejected,
                deferred_count=sum(len(group.suggestions) for group in groups),
                error=error,
            )

        decisions_by_id = {decision.review_id: decision for decision in decisions}
        applied = 0
        rejected = preflight_rejected
        deferred = 0
        errors: list[str] = []
        for group in groups:
            decision = decisions_by_id.get(group.review_id)
            if decision is None:
                self._defer_groups(
                    save_id=save_id,
                    groups=(group,),
                    reason="Reviewer returned no decision for this suggestion.",
                )
                deferred += len(group.suggestions)
                continue
            if decision.action == "accept":
                try:
                    WorldDataService(
                        self.repositories,
                        active_save_id=save_id,
                    ).apply_suggestions(
                        list(group.suggestion_ids),
                        active_save_id=save_id,
                        operation="agent_suggestion_apply",
                        reason=decision.reason,
                    )
                except ValueError as exc:
                    apply_error = _review_action_error("apply", exc)
                    try:
                        WorldDataService(
                            self.repositories,
                            active_save_id=save_id,
                        ).reject_suggestions(
                            list(group.suggestion_ids),
                            active_save_id=save_id,
                            operation="agent_suggestion_reject",
                            reason=(
                                "Automated apply rejected: "
                                f"{apply_error}. Reviewer reason: {decision.reason}"
                            ),
                        )
                    except Exception as reject_exc:  # noqa: BLE001
                        error = _review_action_error("apply_reject", reject_exc)
                        errors.append(error)
                        self._defer_groups(
                            save_id=save_id,
                            groups=(group,),
                            reason=(
                                "Automated apply and rejection deferred: "
                                f"{error}. Apply error: {apply_error}. "
                                f"Reviewer reason: {decision.reason}"
                            ),
                        )
                        deferred += len(group.suggestions)
                        continue
                    rejected += len(group.suggestions)
                    continue
                except Exception as exc:  # noqa: BLE001 - persistence boundary
                    error = _review_action_error("apply", exc)
                    errors.append(error)
                    self._defer_groups(
                        save_id=save_id,
                        groups=(group,),
                        reason=(
                            f"Automated apply deferred: {error}. "
                            f"Reviewer reason: {decision.reason}"
                        ),
                    )
                    deferred += len(group.suggestions)
                    continue
                applied += len(group.suggestions)
                continue
            if decision.action == "reject":
                try:
                    WorldDataService(
                        self.repositories,
                        active_save_id=save_id,
                    ).reject_suggestions(
                        list(group.suggestion_ids),
                        active_save_id=save_id,
                        operation="agent_suggestion_reject",
                        reason=decision.reason,
                    )
                except Exception as exc:  # noqa: BLE001 - persistence boundary
                    error = _review_action_error("reject", exc)
                    errors.append(error)
                    self._defer_groups(
                        save_id=save_id,
                        groups=(group,),
                        reason=(
                            f"Automated reject deferred: {error}. "
                            f"Reviewer reason: {decision.reason}"
                        ),
                    )
                    deferred += len(group.suggestions)
                    continue
                rejected += len(group.suggestions)
                continue
            self._defer_groups(
                save_id=save_id,
                groups=(group,),
                reason=f"Reviewer returned unsupported action: {decision.action}",
            )
            deferred += len(group.suggestions)
        log_event(
            "world_suggestion_review.completed",
            save_id=save_id,
            reviewed_count=len(groups),
            applied_count=applied,
            rejected_count=rejected,
            deferred_count=deferred,
            error="; ".join(errors) if errors else None,
        )
        return WorldSuggestionReviewResult(
            save_id=save_id,
            reviewed_count=len(groups),
            applied_count=applied,
            rejected_count=rejected,
            deferred_count=deferred,
            error="; ".join(errors) if errors else None,
        )

    async def _review_groups(
        self,
        *,
        save_id: str,
        groups: tuple[_SuggestionReviewGroup, ...],
    ) -> tuple[SuggestionReviewDecision, ...]:
        if self.prefer_tool_calls and isinstance(self.provider, ToolCallProvider):
            request = _tool_request(
                repositories=self.repositories,
                provider_name=self.provider_name,
                model_id=self.model_id,
                save_id=save_id,
                groups=groups,
            )
            return await self._review_groups_with_tool_provider(
                provider=self.provider,
                request=request,
                save_id=save_id,
            )
        if not isinstance(self.provider, StructuredOutputProvider):
            if isinstance(self.provider, ToolCallProvider):
                request = _tool_request(
                    repositories=self.repositories,
                    provider_name=self.provider_name,
                    model_id=self.model_id,
                    save_id=save_id,
                    groups=groups,
                )
                return await self._review_groups_with_tool_provider(
                    provider=self.provider,
                    request=request,
                    save_id=save_id,
                )
            raise ValueError("World-suggestion reviewer lacks structured output")
        structured_request = request_with_openrouter_routing(
            self.repositories,
            StructuredOutputRequest(
                provider=self.provider_name,
                model_id=self.model_id,
                schema_name="world_suggestion_review",
                schema=_review_schema(groups),
                messages=_structured_messages(
                    save_id=save_id,
                    groups=groups,
                    messages=tuple(self.repositories.list_messages(save_id)),
                    text_messages=tuple(
                        self.repositories.list_character_text_messages(
                            save_id=save_id
                        )
                    ),
                ),
                temperature=0.0,
            ),
            task="context_update",
            save_id=save_id,
        )
        if self.providers is None:
            structured_response = await self.provider.generate_structured_output(
                budget_structured_output_request(
                    self.repositories,
                    structured_request,
                    task="context_update",
                )
            )
        else:
            structured_response = await structured_output_with_fallback(
                repositories=self.repositories,
                providers=self.providers,
                request=structured_request,
                task="context_update",
                save_id=save_id,
            )
        return _decisions_from_data(structured_response.data)

    async def _review_groups_with_tool_provider(
        self,
        *,
        provider: ToolCallProvider,
        request: ToolCallRequest,
        save_id: str,
    ) -> tuple[SuggestionReviewDecision, ...]:
        if self.providers is None:
            return await _review_groups_with_tool_feedback(
                repositories=self.repositories,
                provider=provider,
                request=request,
            )
        try:
            return await _review_groups_with_tool_feedback(
                repositories=self.repositories,
                provider=provider,
                request=request,
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
                    task="world_suggestion_review",
                    reason=reason,
                )
                raise provider_error_with_fallback_skipped_reason(
                    exc,
                    reason,
                ) from exc
            fallback_provider = self.providers[fallback_request.provider]
            if not isinstance(fallback_provider, ToolCallProvider):
                reason = "fallback_provider_unavailable"
                log_event(
                    "provider.tool_call_fallback_skipped",
                    provider=request.provider,
                    model=request.model_id,
                    task="world_suggestion_review",
                    reason=reason,
                )
                raise provider_error_with_fallback_skipped_reason(
                    exc,
                    reason,
                ) from exc
            log_event(
                "provider.tool_call_fallback_started",
                provider=fallback_request.provider,
                model=fallback_request.model_id,
                task="world_suggestion_review",
            )
            try:
                return await _review_groups_with_tool_feedback(
                    repositories=self.repositories,
                    provider=fallback_provider,
                    request=fallback_request,
                )
            except ProviderError as fallback_exc:
                raise provider_error_with_fallback_attempted(
                    fallback_exc,
                    provider=fallback_request.provider,
                    model_id=fallback_request.model_id,
                ) from fallback_exc

    def _defer_groups(
        self,
        *,
        save_id: str,
        groups: tuple[_SuggestionReviewGroup, ...],
        reason: str,
    ) -> None:
        for group in groups:
            for suggestion in group.suggestions:
                next_attempt = suggestion.review_attempt_count + 1
                if next_attempt >= MAX_AUTOMATED_REVIEW_ATTEMPTS:
                    self.repositories.update_context_update_suggestion_status(
                        suggestion.id,
                        status="rejected",
                    )
                    self.repositories.add_context_update_audit(
                        save_id=save_id,
                        suggestion_id=suggestion.id,
                        operation="agent_suggestion_review_retry_exhausted",
                        entity_type=suggestion.entity_type,
                        entity_id=suggestion.entity_id,
                        field_path=suggestion.field_path,
                        before=None,
                        after=suggestion.proposed_value,
                        reason=(
                            "Suggestion rejected after "
                            f"{MAX_AUTOMATED_REVIEW_ATTEMPTS} failed automated "
                            f"review attempts: {reason}"
                        ),
                        confidence=suggestion.confidence,
                        source_message_ids=suggestion.source_message_ids,
                    )
                    continue
                retry_after_seconds = REVIEW_RETRY_DELAYS_SECONDS[
                    suggestion.review_attempt_count
                ]
                self.repositories.defer_context_update_suggestion_review(
                    [suggestion.id],
                    error=reason,
                    retry_after_seconds=retry_after_seconds,
                )
                if suggestion.review_attempt_count == 0:
                    self.repositories.add_context_update_audit(
                        save_id=save_id,
                        suggestion_id=suggestion.id,
                        operation="agent_suggestion_review_deferred",
                        entity_type=suggestion.entity_type,
                        entity_id=suggestion.entity_id,
                        field_path=suggestion.field_path,
                        before=None,
                        after=suggestion.proposed_value,
                        reason=reason,
                        confidence=suggestion.confidence,
                        source_message_ids=suggestion.source_message_ids,
                    )

    def _reject_invalid_suggestions(
        self,
        *,
        save_id: str,
        groups: tuple[_SuggestionReviewGroup, ...],
    ) -> frozenset[str]:
        rejected_ids: set[str] = set()
        for group in groups:
            for suggestion in group.suggestions:
                reason = _suggestion_invalid_reason(self.repositories, suggestion)
                if reason is None:
                    continue
                self.repositories.update_context_update_suggestion_status(
                    suggestion.id,
                    status="rejected",
                )
                self.repositories.add_context_update_audit(
                    save_id=save_id,
                    suggestion_id=suggestion.id,
                    operation="agent_suggestion_preflight_reject",
                    entity_type=suggestion.entity_type,
                    entity_id=suggestion.entity_id,
                    field_path=suggestion.field_path,
                    before=None,
                    after=suggestion.proposed_value,
                    reason=reason,
                    confidence=suggestion.confidence,
                    source_message_ids=suggestion.source_message_ids,
                )
                rejected_ids.add(suggestion.id)
        return frozenset(rejected_ids)


def _pending_review_groups(
    repositories: PersistenceRepositories,
    save_id: str,
    *,
    due_only: bool = False,
) -> tuple[_SuggestionReviewGroup, ...]:
    grouped: dict[tuple[str, str, str | None, str, str], list[
        ContextUpdateSuggestionRecord
    ]] = {}
    for suggestion in repositories.list_context_update_suggestions(
        save_id,
        status="pending",
    ):
        if due_only and suggestion.next_review_at is not None:
            if not _review_timestamp_is_due(repositories, suggestion.next_review_at):
                continue
        key = (
            suggestion.update_type,
            suggestion.entity_type,
            suggestion.entity_id,
            suggestion.field_path,
            _json_key(suggestion.proposed_value),
        )
        grouped.setdefault(key, []).append(suggestion)
    return tuple(
        _SuggestionReviewGroup(
            review_id=members[0].id,
            suggestions=tuple(members),
        )
        for members in grouped.values()
        if members
    )


def _review_timestamp_is_due(
    repositories: PersistenceRepositories,
    next_review_at: str,
) -> bool:
    row = repositories.connection.execute(
        "SELECT ? <= CURRENT_TIMESTAMP",
        (next_review_at,),
    ).fetchone()
    return row is not None and bool(row[0])


def _suggestion_invalid_reason(
    repositories: PersistenceRepositories,
    suggestion: ContextUpdateSuggestionRecord,
) -> str | None:
    if suggestion.entity_id is not None and suggestion.entity_type == "character":
        character = repositories.get_character(suggestion.entity_id)
        if character is None or character.save_id != suggestion.save_id:
            return "Suggestion rejected because its character target no longer exists."
    if suggestion.entity_id is not None and suggestion.entity_type == "active_thread":
        thread = repositories.get_active_thread(suggestion.entity_id)
        if thread is None or thread.save_id != suggestion.save_id:
            return "Suggestion rejected because its thread target no longer exists."
    messages_by_id = {
        message.id for message in repositories.list_messages(suggestion.save_id)
    }
    text_messages_by_id = {
        message.id
        for message in repositories.list_character_text_messages(
            save_id=suggestion.save_id,
        )
    }
    for source_id in suggestion.source_message_ids:
        text_message_id = parse_character_text_source_ref(source_id)
        if text_message_id is not None and text_message_id not in text_messages_by_id:
            return "Suggestion rejected because a cited text-message source is missing."
        if text_message_id is None and source_id not in messages_by_id:
            return "Suggestion rejected because a cited message source is missing."
    return None


def _review_action_error(action: str, exc: Exception) -> str:
    message = redact_text(str(exc)) or exc.__class__.__name__
    return f"{action} failed: {message}"


def _structured_messages(
    *,
    save_id: str,
    groups: tuple[_SuggestionReviewGroup, ...],
    messages: tuple[MessageRecord, ...],
    text_messages: tuple[CharacterTextMessageRecord, ...],
) -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(
            role="system",
            body=(
                "You are reviewing queued world-data suggestions. Accept only "
                "when the proposed change is supported by the cited reason and "
                "source messages. Reject unsupported, stale, conflicting, or "
                "unsafe suggestions. Return one decision per review_id."
            ),
        ),
        ChatMessage(
            role="user",
            body=_review_body(
                save_id=save_id,
                groups=groups,
                messages=messages,
                text_messages=text_messages,
            ),
        ),
    )


def _tool_messages(
    *,
    save_id: str,
    groups: tuple[_SuggestionReviewGroup, ...],
    messages: tuple[MessageRecord, ...],
    text_messages: tuple[CharacterTextMessageRecord, ...],
) -> tuple[ToolCallMessage, ...]:
    return tuple(
        ToolCallMessage(
            role=message.role,
            body=message.body,
            speaker_name=message.speaker_name,
        )
        for message in _structured_messages(
            save_id=save_id,
            groups=groups,
            messages=messages,
            text_messages=text_messages,
        )
    )


def _tool_request(
    *,
    repositories: PersistenceRepositories,
    provider_name: str,
    model_id: str,
    save_id: str,
    groups: tuple[_SuggestionReviewGroup, ...],
) -> ToolCallRequest:
    return request_with_openrouter_routing(
        repositories,
        ToolCallRequest(
            provider=provider_name,
            model_id=model_id,
            messages=_tool_messages(
                save_id=save_id,
                groups=groups,
                messages=tuple(repositories.list_messages(save_id)),
                text_messages=tuple(
                    repositories.list_character_text_messages(save_id=save_id)
                ),
            ),
            tools=_tool_definitions(groups),
            temperature=0.0,
        ),
        task="context_update",
        save_id=save_id,
    )


def _review_body(
    *,
    save_id: str,
    groups: tuple[_SuggestionReviewGroup, ...],
    messages: tuple[MessageRecord, ...],
    text_messages: tuple[CharacterTextMessageRecord, ...],
) -> str:
    messages_by_id = {message.id: message for message in messages}
    text_messages_by_id = {message.id: message for message in text_messages}
    lines = [f"Save id: {save_id}", "Pending suggestions:"]
    for group in groups:
        suggestion = group.first
        lines.extend(
            [
                f"- review_id: {group.review_id}",
                f"  suggestion_ids: {', '.join(group.suggestion_ids)}",
                f"  target: {suggestion.entity_type}:{suggestion.entity_id or '*'}",
                f"  update_type: {suggestion.update_type}",
                f"  field_path: {suggestion.field_path}",
                f"  proposed_value: {_json_key(suggestion.proposed_value)}",
                f"  confidence: {suggestion.confidence:.3f}",
                f"  cited_reason: {suggestion.reason}",
                "  source_messages:",
            ]
        )
        if not suggestion.source_message_ids:
            lines.append("    none")
        for message_id in suggestion.source_message_ids:
            text_message_id = parse_character_text_source_ref(message_id)
            if text_message_id is not None:
                text_message = text_messages_by_id.get(text_message_id)
                if text_message is None:
                    lines.append(f"    - {message_id}: missing")
                    continue
                lines.append(
                    "    - "
                    f"{message_id} ({text_message.sender}, side-channel text): "
                    f"{_compact(text_message.body, MAX_SOURCE_MESSAGE_CHARS)}"
                )
                continue
            message = messages_by_id.get(message_id)
            if message is None:
                lines.append(f"    - {message_id}: missing")
                continue
            lines.append(
                "    - "
                f"{message.id} ({message.role}, {message.speaker_name or 'unknown'}): "
                f"{_compact(message.body, MAX_SOURCE_MESSAGE_CHARS)}"
            )
    return "\n".join(lines)


def _review_schema(groups: tuple[_SuggestionReviewGroup, ...]) -> dict[str, object]:
    review_ids = [group.review_id for group in groups]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["decisions"],
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["review_id", "action", "reason"],
                    "properties": {
                        "review_id": {"type": "string", "enum": review_ids},
                        "action": {
                            "type": "string",
                            "enum": ["accept", "reject"],
                        },
                        "reason": {"type": "string"},
                    },
                },
            }
        },
    }


def _tool_definitions(groups: tuple[_SuggestionReviewGroup, ...]) -> tuple[
    ToolDefinition,
    ...,
]:
    return (
        ToolDefinition(
            name="review_world_suggestion",
            description=(
                "Accept or reject one queued world-data suggestion review item."
            ),
            parameters=_tool_parameter_schema(groups),
        ),
    )


def _tool_parameter_schema(
    groups: tuple[_SuggestionReviewGroup, ...],
) -> dict[str, Any]:
    schema = _review_schema(groups)
    properties = cast(dict[str, object], schema["properties"])
    decisions = cast(dict[str, object], properties["decisions"])
    return cast(dict[str, Any], decisions["items"])


def _decisions_from_data(data: dict[str, Any]) -> tuple[SuggestionReviewDecision, ...]:
    decisions = data.get("decisions")
    if not isinstance(decisions, list):
        return ()
    return tuple(
        decision
        for item in decisions
        if (decision := _decision_from_mapping(item)) is not None
    )


async def _review_groups_with_tool_feedback(
    *,
    repositories: PersistenceRepositories,
    provider: ToolCallProvider,
    request: ToolCallRequest,
) -> tuple[SuggestionReviewDecision, ...]:
    messages = list(request.messages)
    tool_schemas = {tool.name: tool.parameters for tool in request.tools}
    decisions: list[SuggestionReviewDecision] = []
    decided_review_ids: set[str] = set()
    last_errors: list[str] = []

    for _turn in range(MAX_WORLD_SUGGESTION_REVIEW_TOOL_FEEDBACK_TURNS + 1):
        turn_request = budget_tool_call_request(
            repositories,
            replace(request, messages=tuple(messages)),
            task="world_suggestion_review",
        )
        response = await provider.generate_tool_calls(turn_request)
        errors: list[str] = []
        tool_results: list[tuple[ProviderToolCall, dict[str, str]]] = []
        for call in response.tool_calls:
            accepted, result, decision = _validate_review_tool_call(
                call,
                tool_schemas=tool_schemas,
            )
            if accepted:
                if (
                    decision is not None
                    and decision.review_id not in decided_review_ids
                ):
                    decided_review_ids.add(decision.review_id)
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
            "World-suggestion review tool-call validation failed after feedback: "
            + "; ".join(last_errors)
        ),
    )


def _validate_review_tool_call(
    call: ProviderToolCall,
    *,
    tool_schemas: dict[str, dict[str, object]],
) -> tuple[bool, dict[str, str], SuggestionReviewDecision | None]:
    schema = tool_schemas.get(call.name)
    if schema is None:
        return _invalid_review_tool_call(f"Unknown tool name: {call.name}")
    arguments, parse_error = parse_tool_arguments_json(call.arguments_json)
    if parse_error is not None or arguments is None:
        return _invalid_review_tool_call(
            parse_error or "Tool arguments must be a JSON object"
        )
    shape_error = validate_tool_arguments_shape(arguments, schema=schema)
    if shape_error is not None:
        return _invalid_review_tool_call(shape_error)
    decision = _decision_from_mapping(arguments)
    if decision is None:
        return _invalid_review_tool_call("Tool arguments did not form a decision")
    return True, accepted_tool_result(), decision


def _invalid_review_tool_call(
    error: str,
) -> tuple[bool, dict[str, str], None]:
    return False, invalid_tool_result(error), None


def _decision_from_mapping(value: object) -> SuggestionReviewDecision | None:
    if not isinstance(value, dict):
        return None
    review_id = value.get("review_id")
    action = value.get("action")
    reason = value.get("reason")
    if not isinstance(review_id, str) or not isinstance(action, str):
        return None
    if action not in {"accept", "reject"}:
        return None
    return SuggestionReviewDecision(
        review_id=review_id,
        action=action,
        reason=reason if isinstance(reason, str) and reason.strip() else action,
    )


def _json_key(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _compact(text: str, max_chars: int) -> str:
    compacted = " ".join(text.split())
    if len(compacted) <= max_chars:
        return compacted
    return compacted[: max_chars - 3].rstrip() + "..."
