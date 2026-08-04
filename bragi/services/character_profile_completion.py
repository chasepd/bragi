"""Scenario character starters and profile completion helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, cast

from bragi.app_logging import exception_log_fields, log_error_event, log_event
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
from bragi.retry_policy import MODEL_OUTPUT_MAX_ATTEMPTS, configured_max_attempts
from bragi.services.character_locks import (
    CHARACTER_AGENCY_FIELDS,
    normalize_character_locked_fields,
)
from bragi.services.openrouter_routing_settings import request_with_openrouter_routing
from bragi.services.phrase_denylist import (
    PhraseDenylistViolation,
    denied_phrase_violations,
    effective_generated_phrase_denylist,
    summarize_phrase_policy_violations,
)
from bragi.services.provider_fallbacks import (
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

CHARACTER_STARTERS_CONTENT_KEY = "character_starters"
CHARACTER_STARTER_IDENTITY_LOCK_FIELDS = (
    "aliases",
    "appearance",
    "known_state",
    "name",
    "personality",
    "role",
    "age",
    "visual_notes",
    "voice",
    "texting_style",
)
CHARACTER_STARTER_AGENCY_LOCK_FIELDS = (
    "goals",
    "motivations",
    "current_intent",
    "boundaries",
    "attitude_toward_player",
    "cooperation_conditions",
)
CHARACTER_PROFILE_COMPLETION_FIELDS = tuple(
    field for field in CHARACTER_STARTER_IDENTITY_LOCK_FIELDS if field != "age"
) + CHARACTER_STARTER_AGENCY_LOCK_FIELDS
CHARACTER_FIELD_ENHANCEMENT_FIELDS = (
    "appearance",
    *tuple(sorted(CHARACTER_AGENCY_FIELDS)),
    "known_state",
    "personality",
    "relationships",
    "status",
    "texting_style",
    "visual_notes",
    "voice",
)
MAX_CHARACTER_PROFILE_TOOL_FEEDBACK_TURNS = MODEL_OUTPUT_MAX_ATTEMPTS - 1
MAX_CHARACTER_FIELD_ENHANCEMENT_ATTEMPTS = MODEL_OUTPUT_MAX_ATTEMPTS
CHARACTER_STARTER_OUTPUT_TOKENS_PER_CHARACTER = 10_000
RELATIONSHIP_ENTRY_TARGET_KEYS = (
    "name",
    "character",
    "character_name",
    "target",
    "person",
    "with",
)
RELATIONSHIP_ENTRY_VALUE_KEYS = (
    "relationship",
    "note",
    "status",
    "description",
    "details",
    "value",
)
PROFILE_PHRASE_DENYLIST_FIELDS = ("voice", "texting_style")
CHARACTER_TITLE_WORDS = frozenset(
    {
        "admiral",
        "archivist",
        "brother",
        "captain",
        "commander",
        "dame",
        "doctor",
        "dr",
        "duchess",
        "duke",
        "father",
        "general",
        "guildmaster",
        "inspector",
        "king",
        "lady",
        "lieutenant",
        "lord",
        "mother",
        "prince",
        "princess",
        "prof",
        "professor",
        "queen",
        "sergeant",
        "sir",
        "sister",
        "warden",
    }
)


@dataclass(frozen=True)
class ScenarioStarterReferenceImage:
    id: str
    path: str
    thumbnail_path: str | None = None
    mime_type: str = "image/png"
    prompt_preview: str = "Uploaded character reference image"
    source: str = "uploaded"
    created_at: str | None = None
    bundle_path: str | None = None
    content_rating: str = "unclassified"


@dataclass(frozen=True)
class ScenarioCharacterStarter:
    name: str
    starter_id: str = ""
    aliases: tuple[str, ...] = ()
    role: str = ""
    age: str = ""
    known_state: str = ""
    appearance: str = ""
    visual_notes: str = ""
    personality: str = ""
    voice: str = ""
    texting_style: str = ""
    relationships: dict[str, object] = field(default_factory=dict)
    goals: str = ""
    motivations: str = ""
    current_intent: str = ""
    boundaries: str = ""
    attitude_toward_player: str = ""
    cooperation_conditions: str = ""
    status: str = ""
    met: bool = True
    locked_fields: tuple[str, ...] = ()
    evidence_source_ids: tuple[str, ...] = ()
    reference_image: ScenarioStarterReferenceImage | None = None


@dataclass(frozen=True)
class CharacterProfileCompletionRequest:
    scenario_type: str
    scenario_context: str
    starters: tuple[ScenarioCharacterStarter, ...]
    save_id: str | None = None


@dataclass(frozen=True)
class CharacterStarterGenerationRequest:
    scenario_type: str
    scenario_context: str
    content: Mapping[str, object]
    scenario_types: tuple[str, ...] = ()
    existing_starters: tuple[ScenarioCharacterStarter, ...] = ()
    count: int | None = None
    custom_description: str = ""
    name_candidate_context: str = ""
    save_id: str | None = None


@dataclass(frozen=True)
class CharacterFieldEnhancementRequest:
    scenario_type: str
    scenario_context: str
    character: ScenarioCharacterStarter
    field_name: str
    save_id: str | None = None
    evidence_source_ids: tuple[str, ...] = ()


class StructuredProviderCharacterProfileCompleter:
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

    async def complete(
        self,
        request: CharacterProfileCompletionRequest,
    ) -> tuple[ScenarioCharacterStarter, ...]:
        if not request.starters:
            return ()
        structured_request = StructuredOutputRequest(
            provider=self.provider_name,
            model_id=self.model_id,
            schema_name="character_profile_completion",
            schema=_profile_completion_schema(request.starters),
            messages=_profile_completion_messages(request),
            temperature=0.35,
        )
        structured_request = request_with_openrouter_routing(
            self.repositories,
            structured_request,
            task="context_update",
            save_id=request.save_id,
        )
        messages = list(structured_request.messages)
        last_error: ProviderError | None = None
        phrase_denylist = _profile_phrase_denylist(
            self.repositories,
            save_id=request.save_id,
        )
        max_attempt_count = configured_max_attempts(self.repositories)
        for _attempt in range(max_attempt_count):
            current_request = replace(structured_request, messages=tuple(messages))
            if self.repositories is not None and self.providers is not None:
                response = await structured_output_with_fallback(
                    repositories=self.repositories,
                    providers=self.providers,
                    request=current_request,
                    task="context_update",
                    save_id=request.save_id,
                )
            else:
                response = await self.provider.generate_structured_output(
                    budget_structured_output_request(
                        self.repositories,
                        current_request,
                        task="context_update",
                    )
                )
            completed = _merge_completed_starters(
                request.starters,
                _profile_completion_from_data(response.data),
            )
            violations = _generated_profile_phrase_violations(
                completed,
                base_starters=request.starters,
                phrase_denylist=phrase_denylist,
            )
            if not violations:
                return completed
            last_error = _phrase_denylist_provider_error(violations)
            messages.append(
                ChatMessage(
                    role="user",
                    body=_profile_phrase_retry_feedback(violations),
                )
            )
        if last_error is not None:
            raise last_error
        raise ProviderError(
            ProviderErrorCategory.PROVIDER_ERROR,
            "Character profile completion failed without a provider response.",
        )

    async def generate_starters(
        self,
        request: CharacterStarterGenerationRequest,
    ) -> tuple[ScenarioCharacterStarter, ...]:
        target_count = _starter_generation_target_count(request)
        structured_request = StructuredOutputRequest(
            provider=self.provider_name,
            model_id=self.model_id,
            schema_name="scenario_character_starters",
            schema=_character_starter_generation_schema(target_count),
            messages=_character_starter_generation_messages(request),
            temperature=0.35,
            max_output_tokens=(
                CHARACTER_STARTER_OUTPUT_TOKENS_PER_CHARACTER * target_count
            ),
        )
        task = _starter_generation_task(request.scenario_type)
        structured_request = request_with_openrouter_routing(
            self.repositories,
            structured_request,
            task=task,
            save_id=request.save_id,
        )
        messages = list(structured_request.messages)
        last_error: ProviderError | None = None
        phrase_denylist = _profile_phrase_denylist(
            self.repositories,
            save_id=request.save_id,
        )
        max_attempt_count = configured_max_attempts(self.repositories)
        for _attempt in range(max_attempt_count):
            current_request = replace(structured_request, messages=tuple(messages))
            attempt_error: ProviderError
            try:
                if self.repositories is not None and self.providers is not None:
                    response = await structured_output_with_fallback(
                        repositories=self.repositories,
                        providers=self.providers,
                        request=current_request,
                        task=task,
                        save_id=request.save_id,
                    )
                else:
                    response = await self.provider.generate_structured_output(
                        budget_structured_output_request(
                            self.repositories,
                            current_request,
                            task=task,
                        )
                    )
            except ProviderError as exc:
                if exc.category != ProviderErrorCategory.STRUCTURED_OUTPUT_INVALID:
                    raise
                attempt_error = exc
            else:
                try:
                    starters = _validated_generated_starters_from_data(
                        response.data,
                        request=request,
                        phrase_denylist=phrase_denylist,
                    )
                    return starters
                except ProviderError as exc:
                    attempt_error = exc
            last_error = attempt_error
            messages.append(
                ChatMessage(
                    role="user",
                    body=_starter_generation_retry_feedback(attempt_error),
                )
            )
        if last_error is not None:
            raise last_error
        raise ProviderError(
            ProviderErrorCategory.PROVIDER_ERROR,
            "Character starter generation failed without a provider response.",
        )

    async def enhance_field(
        self,
        request: CharacterFieldEnhancementRequest,
    ) -> ScenarioCharacterStarter:
        field_name = _validated_enhancement_field(request.field_name)
        request = replace(request, field_name=field_name)
        structured_request = StructuredOutputRequest(
            provider=self.provider_name,
            model_id=self.model_id,
            schema_name="character_field_enhancement",
            schema=_field_enhancement_schema(
                character=request.character,
                field_name=field_name,
                evidence_source_ids=request.evidence_source_ids,
            ),
            messages=_field_enhancement_messages(request),
            temperature=0.4,
        )
        structured_request = request_with_openrouter_routing(
            self.repositories,
            structured_request,
            task="character_enhancement",
            save_id=request.save_id,
        )
        return await self._enhance_field_with_provider(
            request=structured_request,
            base=request.character,
            field_name=field_name,
            evidence_source_ids=request.evidence_source_ids,
            save_id=request.save_id,
        )

    async def _enhance_field_with_provider(
        self,
        *,
        request: StructuredOutputRequest,
        base: ScenarioCharacterStarter,
        field_name: str,
        evidence_source_ids: tuple[str, ...],
        save_id: str | None,
    ) -> ScenarioCharacterStarter:
        messages = list(request.messages)
        last_error: ProviderError | None = None
        validation_failure_count = 0
        max_attempt_count = configured_max_attempts(self.repositories)
        for attempt_index in range(max_attempt_count):
            current_request = replace(request, messages=tuple(messages))
            if self.repositories is not None and self.providers is not None:
                response = await structured_output_with_fallback(
                    repositories=self.repositories,
                    providers=self.providers,
                    request=current_request,
                    task="character_enhancement",
                    save_id=save_id,
                )
            else:
                response = await self.provider.generate_structured_output(
                    budget_structured_output_request(
                        self.repositories,
                        current_request,
                        task="character_enhancement",
                    )
                )
            try:
                starter = _field_enhancement_from_data(
                    response.data,
                    base=base,
                    field_name=field_name,
                )
                _validate_enhanced_target_field(
                    starter,
                    field_name=field_name,
                    evidence_source_ids=evidence_source_ids,
                )
                _validate_generated_profile_phrases(
                    (starter,),
                    base_starters=(base,),
                    phrase_denylist=_profile_phrase_denylist(
                        self.repositories,
                        save_id=save_id,
                    ),
                )
                log_event(
                    "character_field_enhancement.structured_validation_succeeded",
                    provider=response.provider,
                    model=response.model_id,
                    field_name=field_name,
                    attempt=attempt_index + 1,
                    max_attempts=max_attempt_count,
                    validation_failure_count=validation_failure_count,
                )
                return starter
            except ProviderError as exc:
                last_error = exc
                validation_failure_count += 1
                log_event(
                    "character_field_enhancement.structured_validation_failed",
                    provider=response.provider,
                    model=response.model_id,
                    field_name=field_name,
                    attempt=attempt_index + 1,
                    max_attempts=max_attempt_count,
                    validation_failure_count=validation_failure_count,
                    error_code=_field_enhancement_validation_error_code(exc.message),
                )
                feedback = exc.message.strip() or str(exc)
                messages.append(
                    ChatMessage(
                        role="user",
                        body=(
                            "Previous structured response was invalid: "
                            f"{feedback}. Return the same character with "
                            f"character.{field_name} populated"
                            f"{_agency_evidence_retry_suffix(field_name)}."
                        ),
                    )
                )
        if last_error is not None:
            raise last_error
        raise ProviderError(
            ProviderErrorCategory.PROVIDER_ERROR,
            "Character field enhancement failed without a provider response.",
        )


class ToolCallingProviderCharacterProfileCompleter:
    def __init__(
        self,
        *,
        provider: ToolCallProvider,
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

    async def complete(
        self,
        request: CharacterProfileCompletionRequest,
    ) -> tuple[ScenarioCharacterStarter, ...]:
        if not request.starters:
            return ()
        tool_request = ToolCallRequest(
            provider=self.provider_name,
            model_id=self.model_id,
            messages=_profile_completion_tool_messages(request),
            tools=_profile_completion_tool_definitions(request.starters),
            temperature=0.35,
        )
        if self.repositories is not None:
            tool_request = request_with_openrouter_routing(
                self.repositories,
                tool_request,
                task="context_update",
                save_id=request.save_id,
            )
        try:
            completed = await self._complete_with_provider(
                provider=self.provider,
                request=tool_request,
                starters=request.starters,
                save_id=request.save_id,
            )
        except ProviderError as exc:
            if self.repositories is None or self.providers is None:
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
                raise ProviderError(
                    exc.category,
                    f"{exc}; fallback skipped: {reason}",
                    status_code=exc.status_code,
                    diagnostics=exc.diagnostics,
                    retry_attempt_count=exc.retry_attempt_count,
                    max_retry_attempts=exc.max_retry_attempts,
                    retry_attempts=exc.retry_attempts,
                ) from exc
            fallback_provider = self.providers[fallback_request.provider]
            if not isinstance(fallback_provider, ToolCallProvider):
                raise
            completed = await self._complete_with_provider(
                provider=fallback_provider,
                request=fallback_request,
                starters=request.starters,
                save_id=request.save_id,
            )
        return _merge_completed_starters(request.starters, completed)

    async def generate_starters(
        self,
        request: CharacterStarterGenerationRequest,
    ) -> tuple[ScenarioCharacterStarter, ...]:
        return ()

    async def enhance_field(
        self,
        request: CharacterFieldEnhancementRequest,
    ) -> ScenarioCharacterStarter:
        field_name = _validated_enhancement_field(request.field_name)
        request = replace(request, field_name=field_name)
        tool_request = ToolCallRequest(
            provider=self.provider_name,
            model_id=self.model_id,
            messages=_field_enhancement_tool_messages(request),
            tools=_field_enhancement_tool_definitions(
                character=request.character,
                field_name=field_name,
                evidence_source_ids=request.evidence_source_ids,
            ),
            temperature=0.4,
        )
        if self.repositories is not None:
            tool_request = request_with_openrouter_routing(
                self.repositories,
                tool_request,
                task="character_enhancement",
                save_id=request.save_id,
            )
        try:
            return await self._enhance_field_with_provider(
                provider=self.provider,
                request=tool_request,
                base=request.character,
                field_name=field_name,
                evidence_source_ids=request.evidence_source_ids,
                save_id=request.save_id,
            )
        except ProviderError as exc:
            if self.repositories is None or self.providers is None:
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
                raise ProviderError(
                    exc.category,
                    f"{exc}; fallback skipped: {reason}",
                    status_code=exc.status_code,
                    diagnostics=exc.diagnostics,
                    retry_attempt_count=exc.retry_attempt_count,
                    max_retry_attempts=exc.max_retry_attempts,
                    retry_attempts=exc.retry_attempts,
                ) from exc
            fallback_provider = self.providers[fallback_request.provider]
            if not isinstance(fallback_provider, ToolCallProvider):
                raise
            return await self._enhance_field_with_provider(
                provider=fallback_provider,
                request=fallback_request,
                base=request.character,
                field_name=field_name,
                evidence_source_ids=request.evidence_source_ids,
                save_id=request.save_id,
            )

    async def _complete_with_provider(
        self,
        *,
        provider: ToolCallProvider,
        request: ToolCallRequest,
        starters: tuple[ScenarioCharacterStarter, ...],
        save_id: str | None,
    ) -> tuple[ScenarioCharacterStarter, ...]:
        messages = list(request.messages)
        tool_schema = request.tools[0].parameters
        starters_by_key = {
            _character_key(starter.name): starter for starter in starters
        }
        phrase_denylist = _profile_phrase_denylist(
            self.repositories,
            save_id=save_id,
        )
        completed_by_key: dict[str, ScenarioCharacterStarter] = {}
        last_errors: list[str] = []
        max_attempt_count = configured_max_attempts(self.repositories)
        for _turn in range(max_attempt_count):
            turn_request = budget_tool_call_request(
                self.repositories,
                replace(request, messages=tuple(messages)),
                task="scenario_generation",
            )
            response = await provider.generate_tool_calls(turn_request)
            errors: list[str] = []
            tool_results: list[tuple[ProviderToolCall, dict[str, str]]] = []
            for call in response.tool_calls:
                accepted, result, starter = _validate_profile_completion_tool_call(
                    call,
                    tool_schema=tool_schema,
                    starters_by_key=starters_by_key,
                    phrase_denylist=phrase_denylist,
                )
                tool_results.append((call, result))
                if accepted and starter is not None:
                    completed_by_key[_character_key(starter.name)] = starter
                    continue
                errors.append(result["error"])
            if not errors:
                return tuple(completed_by_key.values())
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
                "Character profile completion tool-call validation failed: "
                + "; ".join(last_errors)
            ),
        )

    async def _enhance_field_with_provider(
        self,
        *,
        provider: ToolCallProvider,
        request: ToolCallRequest,
        base: ScenarioCharacterStarter,
        field_name: str,
        evidence_source_ids: tuple[str, ...],
        save_id: str | None,
    ) -> ScenarioCharacterStarter:
        messages = list(request.messages)
        tool_schema = request.tools[0].parameters
        last_errors: list[str] = []
        validation_failure_count = 0
        max_turns = configured_max_attempts(self.repositories)
        for turn_index in range(max_turns):
            turn_request = budget_tool_call_request(
                self.repositories,
                replace(request, messages=tuple(messages)),
                task="character_enhancement",
            )
            response = await provider.generate_tool_calls(turn_request)
            errors: list[str] = []
            tool_results: list[tuple[ProviderToolCall, dict[str, str]]] = []
            for call in response.tool_calls:
                accepted, result, starter = _validate_field_enhancement_tool_call(
                    call,
                    tool_schema=tool_schema,
                    base=base,
                    field_name=field_name,
                    evidence_source_ids=evidence_source_ids,
                    phrase_denylist=_profile_phrase_denylist(
                        self.repositories,
                        save_id=save_id,
                    ),
                )
                tool_results.append((call, result))
                if accepted and starter is not None:
                    log_event(
                        "character_field_enhancement.tool_call_validation_succeeded",
                        provider=response.provider,
                        model=response.model_id,
                        field_name=field_name,
                        turn=turn_index + 1,
                        max_turns=max_turns,
                        tool_call_count=len(response.tool_calls),
                        accepted_count=1,
                        error_count=len(errors),
                        validation_failure_count=validation_failure_count,
                    )
                    return starter
                errors.append(result["error"])
            if not errors:
                break
            last_errors = errors
            validation_failure_count += len(errors)
            log_event(
                "character_field_enhancement.tool_call_validation_failed",
                provider=response.provider,
                model=response.model_id,
                field_name=field_name,
                turn=turn_index + 1,
                max_turns=max_turns,
                tool_call_count=len(response.tool_calls),
                accepted_count=0,
                error_count=len(errors),
                validation_failure_count=validation_failure_count,
                error_codes=tuple(
                    _field_enhancement_validation_error_code(error)
                    for error in errors
                ),
            )
            append_tool_feedback_messages(
                messages,
                assistant_body=response.body,
                tool_calls=response.tool_calls,
                tool_results=tool_results,
            )
        raise ProviderError(
            category=ProviderErrorCategory.PROVIDER_ERROR,
            message=(
                "Character field enhancement tool-call validation failed: "
                + "; ".join(last_errors)
            ),
        )


async def complete_character_starters(
    *,
    completer: object | None,
    scenario_type: str,
    content: Mapping[str, object],
    scenario_types: Iterable[object] | None = None,
    save_id: str | None = None,
) -> tuple[ScenarioCharacterStarter, ...]:
    del scenario_types
    starters = scenario_character_starters_for_content(
        scenario_type=scenario_type,
        content=content,
    )
    if not starters or completer is None:
        return starters
    if not _starters_need_profile_completion(starters):
        return starters
    request = CharacterProfileCompletionRequest(
        scenario_type=scenario_type,
        scenario_context=scenario_context_text(
            scenario_type=scenario_type,
            content=content,
        ),
        starters=starters,
        save_id=save_id,
    )
    complete = getattr(completer, "complete", None)
    if not callable(complete):
        return starters
    try:
        completed = await complete(request)
    except Exception as exc:
        log_error_event(
            "character_profile_completion.failed",
            scenario_type=scenario_type,
            starter_count=len(starters),
            **exception_log_fields(exc),
        )
        return starters
    log_event(
        "character_profile_completion.succeeded",
        scenario_type=scenario_type,
        starter_count=len(starters),
    )
    return _merge_completed_starters(starters, tuple(completed))


def content_with_character_starters(
    *,
    scenario_type: str,
    content: Mapping[str, object],
    starters: Iterable[ScenarioCharacterStarter] | None = None,
) -> dict[str, object]:
    normalized_content = dict(content)
    normalized_starters = tuple(
        starters
        if starters is not None
        else scenario_character_starters_for_content(
            scenario_type=scenario_type,
            content=normalized_content,
        )
    )
    if normalized_starters:
        normalized_content[CHARACTER_STARTERS_CONTENT_KEY] = [
            scenario_character_starter_to_json(starter)
            for starter in normalized_starters
        ]
    else:
        normalized_content.pop(CHARACTER_STARTERS_CONTENT_KEY, None)
    return normalized_content


def scenario_character_starters_for_content(
    *,
    scenario_type: str,
    content: Mapping[str, object],
) -> tuple[ScenarioCharacterStarter, ...]:
    existing = content.get(CHARACTER_STARTERS_CONTENT_KEY)
    if existing is not None:
        starters = normalize_scenario_character_starters(existing, strict=False)
        if starters:
            return starters
    return ()


def normalize_scenario_character_starters(
    value: object,
    *,
    strict: bool,
) -> tuple[ScenarioCharacterStarter, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        if strict:
            raise TypeError("character_starters must be an array")
        return ()
    starters: list[ScenarioCharacterStarter] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        try:
            starter = _starter_from_mapping(item, path=f"character_starters[{index}]")
        except (TypeError, ValueError):
            if strict:
                raise
            continue
        key = _character_key(starter.name)
        if not key or key in seen:
            continue
        starters.append(starter)
        seen.add(key)
    return tuple(starters)


def scenario_character_starter_to_json(
    starter: ScenarioCharacterStarter,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": starter.name,
        "aliases": list(starter.aliases),
        "role": starter.role,
        "age": starter.age,
        "known_state": starter.known_state,
        "appearance": starter.appearance,
        "visual_notes": starter.visual_notes,
        "personality": starter.personality,
        "voice": starter.voice,
        "texting_style": starter.texting_style,
        "relationships": dict(starter.relationships),
        "goals": starter.goals,
        "motivations": starter.motivations,
        "current_intent": starter.current_intent,
        "boundaries": starter.boundaries,
        "attitude_toward_player": starter.attitude_toward_player,
        "cooperation_conditions": starter.cooperation_conditions,
        "status": starter.status,
        "met": starter.met,
        "locked_fields": list(starter.locked_fields),
    }
    if starter.starter_id:
        payload["starter_id"] = starter.starter_id
    if starter.reference_image is not None:
        payload["reference_image"] = scenario_starter_reference_image_to_json(
            starter.reference_image
        )
    return payload


def scenario_starter_reference_image_to_json(
    reference: ScenarioStarterReferenceImage,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": reference.id,
        "path": reference.path,
        "thumbnail_path": reference.thumbnail_path,
        "mime_type": reference.mime_type,
        "prompt_preview": reference.prompt_preview,
        "source": reference.source,
        "created_at": reference.created_at,
        "content_rating": reference.content_rating,
    }
    if reference.bundle_path is not None:
        payload["bundle_path"] = reference.bundle_path
    return payload


def scenario_context_text(
    *,
    scenario_type: str,
    content: Mapping[str, object],
) -> str:
    lines = [f"Scenario type: {scenario_type}"]
    for key, value in content.items():
        if key in {CHARACTER_STARTERS_CONTENT_KEY, "_source"}:
            continue
        if isinstance(value, str) and value.strip():
            lines.append(f"{key}: {value.strip()}")
    return "\n".join(lines)


def starter_identity_locked_fields(
    starter: ScenarioCharacterStarter,
) -> list[str]:
    default_fields = tuple(
        field
        for field in CHARACTER_STARTER_IDENTITY_LOCK_FIELDS
        if field != "age" or starter.age.strip()
    )
    generated_agency_fields = tuple(
        field
        for field in CHARACTER_STARTER_AGENCY_LOCK_FIELDS
        if getattr(starter, field).strip()
    )
    seed_lock_fields = (
        *CHARACTER_STARTER_IDENTITY_LOCK_FIELDS,
        *CHARACTER_STARTER_AGENCY_LOCK_FIELDS,
    )
    return normalize_character_locked_fields(
        (
            *default_fields,
            *generated_agency_fields,
            *(
                field
                for field in starter.locked_fields
                if field in seed_lock_fields
            ),
        ),
        preserve_unknown=False,
    )


def _starter_from_mapping(
    value: object,
    *,
    path: str,
) -> ScenarioCharacterStarter:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be an object")
    payload = cast(dict[str, object], value)
    name = _required_string(payload, "name", path)
    relationships = payload.get("relationships", {})
    if relationships is None:
        relationships = {}
    if not isinstance(relationships, dict) or any(
        not isinstance(key, str) for key in relationships
    ):
        raise TypeError(f"{path}.relationships must be an object")
    met = payload.get("met", True)
    if not isinstance(met, bool):
        raise TypeError(f"{path}.met must be a boolean")
    return ScenarioCharacterStarter(
        name=name,
        starter_id=_string(payload.get("starter_id")),
        aliases=_string_tuple(payload.get("aliases"), f"{path}.aliases"),
        role=_string(payload.get("role")),
        age=_string(payload.get("age")),
        known_state=_string(payload.get("known_state")),
        appearance=_string(payload.get("appearance")),
        visual_notes=_string(payload.get("visual_notes")),
        personality=_string(payload.get("personality")),
        voice=_string(payload.get("voice")),
        texting_style=_string(payload.get("texting_style")),
        relationships=dict(relationships),
        goals=_string(payload.get("goals")),
        motivations=_string(payload.get("motivations")),
        current_intent=_string(payload.get("current_intent")),
        boundaries=_string(payload.get("boundaries")),
        attitude_toward_player=_string(payload.get("attitude_toward_player")),
        cooperation_conditions=_string(payload.get("cooperation_conditions")),
        status=_string(payload.get("status")),
        met=met,
        locked_fields=tuple(
            normalize_character_locked_fields(
                _string_tuple(payload.get("locked_fields"), f"{path}.locked_fields"),
                preserve_unknown=False,
            )
        ),
        evidence_source_ids=_string_tuple(
            payload.get("evidence_source_ids"),
            f"{path}.evidence_source_ids",
        ),
        reference_image=_starter_reference_image_from_mapping(
            payload.get("reference_image"),
            path=f"{path}.reference_image",
        ),
    )


def _starter_reference_image_from_mapping(
    value: object,
    *,
    path: str,
) -> ScenarioStarterReferenceImage | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be an object")
    payload = cast(dict[str, object], value)
    image_id = _required_string(payload, "id", path)
    image_path = _required_string(payload, "path", path)
    thumbnail_path = payload.get("thumbnail_path")
    if thumbnail_path is not None and not isinstance(thumbnail_path, str):
        raise TypeError(f"{path}.thumbnail_path must be a string or null")
    created_at = payload.get("created_at")
    if created_at is not None and not isinstance(created_at, str):
        raise TypeError(f"{path}.created_at must be a string or null")
    bundle_path = payload.get("bundle_path")
    if bundle_path is not None and not isinstance(bundle_path, str):
        raise TypeError(f"{path}.bundle_path must be a string or null")
    return ScenarioStarterReferenceImage(
        id=image_id,
        path=image_path,
        thumbnail_path=thumbnail_path,
        mime_type=_string(payload.get("mime_type")) or "image/png",
        prompt_preview=(
            _string(payload.get("prompt_preview"))
            or "Uploaded character reference image"
        ),
        source=_string(payload.get("source")) or "uploaded",
        created_at=created_at,
        bundle_path=bundle_path,
        content_rating=_string(payload.get("content_rating")) or "unclassified",
    )


def _required_string(payload: Mapping[str, object], key: str, path: str) -> str:
    value = payload.get(key, "")
    if not isinstance(value, str):
        raise TypeError(f"{path}.{key} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{path}.{key} is required")
    return text


def _string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_tuple(value: object, path: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise TypeError(f"{path} must be an array")
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise TypeError(f"{path}[{index}] must be a string")
        text = item.strip()
        if text:
            items.append(text)
    return tuple(dict.fromkeys(items))


def _content_text(content: Mapping[str, object], key: str) -> str:
    value = content.get(key)
    return value.strip() if isinstance(value, str) else ""


def _profile_completion_schema(
    starters: tuple[ScenarioCharacterStarter, ...],
) -> dict[str, object]:
    names = [starter.name for starter in starters]
    string_array = {"type": "array", "items": {"type": "string"}}
    starter_schema: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string", "enum": names},
            "aliases": string_array,
            "role": {"type": "string"},
            "age": {"type": "string"},
            "known_state": {"type": "string"},
            "appearance": {"type": "string"},
            "visual_notes": {"type": "string"},
            "personality": {"type": "string"},
            "voice": {"type": "string"},
            "texting_style": {"type": "string"},
            "relationships": {"type": "object"},
            "goals": {"type": "string"},
            "motivations": {"type": "string"},
            "current_intent": {"type": "string"},
            "boundaries": {"type": "string"},
            "attitude_toward_player": {"type": "string"},
            "cooperation_conditions": {"type": "string"},
            "status": {"type": "string"},
            "met": {"type": "boolean"},
            "locked_fields": string_array,
        },
        "required": ["name"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "characters": {
                "type": "array",
                "items": starter_schema,
            }
        },
        "required": ["characters"],
    }


def _character_starter_generation_schema(
    target_count: int,
) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "characters": {
                "type": "array",
                "items": _character_starter_generation_item_schema(),
                "minItems": target_count,
                "maxItems": target_count,
            }
        },
        "required": ["characters"],
    }


def _character_starter_generation_item_schema() -> dict[str, object]:
    string_array = {"type": "array", "items": {"type": "string"}}
    properties: dict[str, object] = {
        "name": {"type": "string"},
        "aliases": string_array,
        "role": {"type": "string"},
        "age": {"type": "string"},
        "known_state": {"type": "string"},
        "appearance": {"type": "string"},
        "visual_notes": {"type": "string"},
        "personality": {"type": "string"},
        "voice": {"type": "string"},
        "texting_style": {"type": "string"},
        "goals": {"type": "string"},
        "motivations": {"type": "string"},
        "current_intent": {"type": "string"},
        "boundaries": {"type": "string"},
        "attitude_toward_player": {"type": "string"},
        "cooperation_conditions": {"type": "string"},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def _profile_completion_item_schema(
    starters: tuple[ScenarioCharacterStarter, ...],
) -> dict[str, object]:
    schema = _profile_completion_schema(starters)
    properties = cast(dict[str, object], schema["properties"])
    characters_schema = cast(dict[str, object], properties["characters"])
    return cast(dict[str, object], characters_schema["items"])


def _field_enhancement_schema(
    *,
    character: ScenarioCharacterStarter,
    field_name: str,
    evidence_source_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    target_schema: dict[str, object] = (
        _relationship_entries_schema()
        if field_name == "relationships"
        else {"type": "string"}
    )
    character_properties: dict[str, object] = {
        "name": {"type": "string", "enum": [character.name]},
        field_name: target_schema,
    }
    if field_name in CHARACTER_AGENCY_FIELDS:
        evidence_items: dict[str, object] = {"type": "string"}
        if evidence_source_ids:
            evidence_items["enum"] = list(evidence_source_ids)
        character_properties["evidence_source_ids"] = {
            "type": "array",
            "items": evidence_items,
        }
    character_schema: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "properties": character_properties,
        "required": ["name"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "field_name": {"type": "string", "enum": [field_name]},
            "character": character_schema,
        },
        "required": ["field_name", "character"],
    }


def _relationship_entries_schema() -> dict[str, object]:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
                "relationship": {"type": "string"},
            },
            "required": ["name", "relationship"],
        },
    }


def _profile_completion_messages(
    request: CharacterProfileCompletionRequest,
) -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(
            role="system",
            body=(
                "Complete sparse Bragi scenario character starters. Use the "
                "enforced response schema. Return only listed characters. Preserve "
                "nonblank supplied fields; fill blank identity/profile fields with "
                "concise details from the scenario context. Also fill blank agency "
                "fields with concise plausible details that fit the scenario and "
                "current profile: goals, motivations, current intent, boundaries, "
                "attitude toward the player, and cooperation conditions. Use the "
                "full spectrum of NPC stances; some characters may be naturally "
                "trusting and cooperative, while others may be guarded, hostile, "
                "self-interested, unfair, or unreasonable when that fits their "
                "role and context. Fill blank voice with "
                "cadence, diction, and 1-2 short quoted concrete examples of what "
                "the character would actually say. Fill blank texting_style "
                "with concise phone-message habits such as length, punctuation, "
                "capitalization, emoji comfort, double-texting, attachment habits, "
                "response rhythm, and 1-2 short sample texts. When context is sparse, "
                "invent plausible neutral details that fit the scenario. For "
                "character age, fill only when directly stated or clearly evidenced; "
                "otherwise leave it blank."
            ),
        ),
        ChatMessage(
            role="user",
            body="\n\n".join(
                (
                    request.scenario_context,
                    _starter_prompt_text(request.starters),
                )
            ),
        ),
    )


def _character_starter_generation_messages(
    request: CharacterStarterGenerationRequest,
) -> tuple[ChatMessage, ...]:
    target_count = _starter_generation_target_count(request)
    custom_instruction = (
        "Create exactly one character starter matching this custom description: "
        f"{request.custom_description.strip()}"
        if request.custom_description.strip()
        else f"Create exactly {target_count} new character starters."
    )
    return (
        ChatMessage(
            role="system",
            body=(
                "Create Bragi scenario character starters. Use the enforced "
                "response schema. "
                f"{custom_instruction} Do not return the player character. Do "
                "not return any existing starter or alias. Do not repeat names "
                "or first names in the generated starters. Return every schema "
                "field for every starter; use an empty string or empty aliases "
                "array when the draft does not support a value. Fill concise role, "
                "age, known_state, appearance, visual_notes, personality, "
                "voice, texting_style, and all agency fields from the draft "
                "context and request: goals, motivations, current intent, "
                "boundaries, attitude toward the player, and cooperation "
                "conditions. Use the full spectrum of NPC stances; some characters "
                "may be naturally trusting and cooperative, while others may be "
                "guarded, hostile, self-interested, unfair, or unreasonable when "
                "that fits their role and context. For voice include cadence, "
                "diction, and 1-2 short quoted examples of what the character "
                "would actually say. For texting_style include phone-message "
                "habits and 1-2 short sample texts. Fill age only when directly "
                "stated or clearly evidenced. Use ordinary name candidates only "
                "when they fit the scenario."
            ),
        ),
        ChatMessage(
            role="user",
            body=_starter_generation_context_text(request),
        ),
    )


def _starter_generation_context_text(
    request: CharacterStarterGenerationRequest,
) -> str:
    scenario_types = tuple(
        scenario_type.strip()
        for scenario_type in (request.scenario_types or (request.scenario_type,))
        if scenario_type.strip()
    )
    player_name = _content_text(request.content, "player_character_name")
    parts = [
        request.scenario_context,
        f"Scenario types: {', '.join(scenario_types)}",
        f"Player character name: {player_name or '[none specified]'}",
        _starter_prompt_text(request.existing_starters),
    ]
    if request.name_candidate_context:
        parts.append(request.name_candidate_context)
        parts.append(
            "Use the candidate names only when they fit the requested new "
            "character starters."
        )
    if request.custom_description.strip():
        parts.append(
            f"Custom character description: {request.custom_description.strip()}"
        )
    return "\n\n".join(part for part in parts if part.strip())


def _field_enhancement_messages(
    request: CharacterFieldEnhancementRequest,
) -> tuple[ChatMessage, ...]:
    field_label = request.field_name.replace("_", " ")
    current_value = _starter_field_display_value(
        request.character,
        request.field_name,
    )
    if request.field_name in CHARACTER_AGENCY_FIELDS:
        evidence_instruction = (
            "Enhance one Bragi NPC agency field from source-labeled save "
            "context. Use the enforced response schema. Include only target "
            "field details directly supported by the supplied source IDs; do "
            "not invent motives, goals, intent, boundaries, attitudes, or "
            "conditions. Preserve compatible current target details. Return "
            "only the listed character, target field, and cite "
            "evidence_source_ids that support the target field."
        )
        if request.evidence_source_ids:
            evidence_instruction += (
                " Allowed evidence_source_ids: "
                + ", ".join(request.evidence_source_ids)
                + "."
            )
        return (
            ChatMessage(
                role="system",
                body=evidence_instruction,
            ),
            ChatMessage(
                role="user",
                body="\n\n".join(
                    (
                        request.scenario_context,
                        _starter_prompt_text((request.character,)),
                        f"Target field: {request.field_name} ({field_label})",
                        f"Current target value: {current_value or '[empty]'}",
                    )
                ),
            ),
        )
    appearance_instruction = ""
    if request.field_name == "appearance":
        appearance_instruction = (
            " Always include skin tone in the appearance description. If the "
            "available details do not establish it, infer a plausible detail from "
            "the scenario context and existing character profile."
        )
    example_instruction = ""
    if request.field_name == "voice":
        example_instruction = (
            " Include cadence, diction, and 1-2 short quoted concrete examples "
            "of what this character would actually say."
        )
    elif request.field_name == "texting_style":
        example_instruction = (
            " Include phone-message habits and 1-2 short sample texts this "
            "character would send."
        )
    return (
        ChatMessage(
            role="system",
            body=(
                "Enhance one Bragi character registry field. Use the enforced "
                "response schema. Preserve every existing detail in the target "
                "field while filling gaps with concise, plausible details that fit "
                "the scenario. Add at least one concrete new detail beyond the "
                "current target value whenever any scenario-consistent detail can "
                "be inferred. Do not return the target field unchanged, lightly "
                "reworded, or merely summarized. If the target field is empty, "
                "invent useful neutral details from the scenario context. Return "
                "only the listed character and target field."
                f"{appearance_instruction}"
                f"{example_instruction}"
            ),
        ),
        ChatMessage(
            role="user",
            body="\n\n".join(
                (
                    request.scenario_context,
                    _starter_prompt_text((request.character,)),
                    f"Target field: {request.field_name} ({field_label})",
                    f"Current target value: {current_value or '[empty]'}",
                )
            ),
        ),
    )


def _profile_completion_tool_messages(
    request: CharacterProfileCompletionRequest,
) -> tuple[ToolCallMessage, ...]:
    messages = _profile_completion_messages(request)
    return tuple(
        ToolCallMessage(
            role=message.role,
            body=message.body.replace(
                "Use the enforced response schema.",
                "Use the complete_character_profile tool instead of prose.",
            ),
            speaker_name=message.speaker_name,
        )
        for message in messages
    )


def _field_enhancement_tool_messages(
    request: CharacterFieldEnhancementRequest,
) -> tuple[ToolCallMessage, ...]:
    messages = _field_enhancement_messages(request)
    return tuple(
        ToolCallMessage(
            role=message.role,
            body=message.body.replace(
                "Use the enforced response schema.",
                "Use the enhance_character_field tool instead of prose.",
            ),
            speaker_name=message.speaker_name,
        )
        for message in messages
    )


def _profile_completion_tool_definitions(
    starters: tuple[ScenarioCharacterStarter, ...],
) -> tuple[ToolDefinition, ...]:
    return (
        ToolDefinition(
            name="complete_character_profile",
            description=(
                "Fill blank profile fields for one listed scenario character starter."
            ),
            parameters=_profile_completion_item_schema(starters),
        ),
    )


def _field_enhancement_tool_definitions(
    *,
    character: ScenarioCharacterStarter,
    field_name: str,
    evidence_source_ids: tuple[str, ...] = (),
) -> tuple[ToolDefinition, ...]:
    return (
        ToolDefinition(
            name="enhance_character_field",
            description="Enhance one listed Bragi character registry field.",
            parameters=_field_enhancement_schema(
                character=character,
                field_name=field_name,
                evidence_source_ids=evidence_source_ids,
            ),
        ),
    )


def _starter_prompt_text(starters: tuple[ScenarioCharacterStarter, ...]) -> str:
    lines = ["Scenario character starters:"]
    for starter in starters:
        missing = [
            field
            for field in CHARACTER_PROFILE_COMPLETION_FIELDS
            if _starter_field_is_blank(starter, field)
        ]
        lines.append(
            f"- {starter.name}: role={starter.role}; "
            f"age={starter.age}; "
            f"known_state={starter.known_state}; "
            f"appearance={starter.appearance}; personality={starter.personality}; "
            f"voice={starter.voice}; texting_style={starter.texting_style}; "
            f"goals={starter.goals}; "
            f"motivations={starter.motivations}; "
            f"current_intent={starter.current_intent}; "
            f"boundaries={starter.boundaries}; "
            f"attitude_toward_player={starter.attitude_toward_player}; "
            f"cooperation_conditions={starter.cooperation_conditions}; "
            f"missing={', '.join(missing) or 'none'}"
        )
    return "\n".join(lines)


def _starter_field_display_value(
    starter: ScenarioCharacterStarter,
    field_name: str,
) -> str:
    value = getattr(starter, field_name)
    if isinstance(value, dict):
        if not value:
            return ""
        return "; ".join(f"{key}: {item}" for key, item in value.items())
    if isinstance(value, tuple):
        return ", ".join(value)
    return str(value).strip()


def _profile_completion_from_data(
    data: dict[str, object],
) -> tuple[ScenarioCharacterStarter, ...]:
    return normalize_scenario_character_starters(
        data.get("characters", []),
        strict=False,
    )


def _validated_generated_starters_from_data(
    data: dict[str, object],
    *,
    request: CharacterStarterGenerationRequest,
    phrase_denylist: tuple[str, ...],
) -> tuple[ScenarioCharacterStarter, ...]:
    raw_characters = data.get("characters", [])
    if not isinstance(raw_characters, list):
        raise ProviderError(
            ProviderErrorCategory.PROVIDER_ERROR,
            "Character starter generation returned invalid character data: "
            "characters must be an array.",
        )
    starters: list[ScenarioCharacterStarter] = []
    errors: list[str] = []
    for index, item in enumerate(raw_characters):
        try:
            starters.append(_starter_from_mapping(item, path=f"characters[{index}]"))
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
    target_count = _starter_generation_target_count(request)
    if len(starters) != target_count:
        errors.append(
            "Character starter generation returned "
            f"{len(starters)} characters; expected exactly {target_count}."
        )
    errors.extend(_generated_starter_name_errors(tuple(starters), request=request))
    violations = _generated_profile_phrase_violations(
        tuple(starters),
        base_starters=(),
        phrase_denylist=phrase_denylist,
    )
    if violations:
        errors.append(summarize_phrase_policy_violations(violations))
    if errors:
        raise ProviderError(
            ProviderErrorCategory.PROVIDER_ERROR,
            "Character starter generation returned invalid character data: "
            + "; ".join(errors),
        )
    return tuple(starters)


def _generated_starter_name_errors(
    starters: tuple[ScenarioCharacterStarter, ...],
    *,
    request: CharacterStarterGenerationRequest,
) -> list[str]:
    errors: list[str] = []
    player_names = (_content_text(request.content, "player_character_name"),)
    unavailable_keys = _starter_generation_unavailable_name_keys(
        player_names=player_names,
        existing_starters=request.existing_starters,
    )
    seen_keys: set[str] = set()
    seen_first_names: set[str] = set()
    for starter in starters:
        keys = _starter_seen_keys(starter)
        dedupe_keys = _starter_generated_dedupe_keys(starter)
        first_name_key = _character_first_name_key(starter.name)
        if not keys:
            errors.append("Generated starter name must not be blank.")
            continue
        if keys & unavailable_keys or dedupe_keys & unavailable_keys:
            errors.append(
                f"Generated starter name duplicates the player or existing starter: "
                f"{starter.name}."
            )
        if keys & seen_keys or dedupe_keys & seen_keys:
            errors.append(f"Generated starter name is duplicated: {starter.name}.")
        if first_name_key and first_name_key in seen_first_names:
            errors.append(
                f"Generated starter first name is duplicated: {starter.name}."
            )
        seen_keys.update(dedupe_keys)
        if first_name_key:
            seen_first_names.add(first_name_key)
    return errors


def _starter_generation_unavailable_name_keys(
    *,
    player_names: Iterable[str],
    existing_starters: tuple[ScenarioCharacterStarter, ...],
) -> set[str]:
    keys: set[str] = set()
    for name in player_names:
        keys.update(_character_generated_dedupe_keys(name))
    for starter in existing_starters:
        keys.update(_starter_generated_dedupe_keys(starter))
    return {key for key in keys if key}


def _starter_generation_target_count(
    request: CharacterStarterGenerationRequest,
) -> int:
    if request.custom_description.strip():
        return 1
    if request.count is None or isinstance(request.count, bool):
        raise ProviderError(
            ProviderErrorCategory.PROVIDER_ERROR,
            "Character starter generation count is required.",
        )
    if request.count < 1 or request.count > 12:
        raise ProviderError(
            ProviderErrorCategory.PROVIDER_ERROR,
            "Character starter generation count must be between 1 and 12.",
        )
    return request.count


def _starter_generation_retry_feedback(error: ProviderError) -> str:
    feedback = error.message.strip() or str(error)
    return (
        "Previous structured response was invalid: "
        f"{feedback}. Return corrected character starters matching the requested "
        "count with unique names that do not repeat the player or existing "
        "starters."
    )


def _starter_generation_task(scenario_type: str) -> str:
    normalized = scenario_type.strip() or "scenario"
    return f"{normalized}_context_update"


def _field_enhancement_from_data(
    data: dict[str, object],
    *,
    base: ScenarioCharacterStarter,
    field_name: str,
) -> ScenarioCharacterStarter:
    if data.get("field_name") != field_name:
        raise ProviderError(
            ProviderErrorCategory.PROVIDER_ERROR,
            "Character field enhancement returned the wrong field.",
        )
    payload = data.get("character")
    try:
        starter = _field_enhancement_starter_from_mapping(
            payload,
            path="character",
            base=base,
            field_name=field_name,
        )
    except (TypeError, ValueError) as exc:
        raise ProviderError(
            ProviderErrorCategory.PROVIDER_ERROR,
            f"Character field enhancement returned invalid character data: {exc}",
        ) from exc
    if _character_key(starter.name) != _character_key(base.name):
        raise ProviderError(
            ProviderErrorCategory.PROVIDER_ERROR,
            "Character field enhancement returned the wrong character.",
        )
    return starter


def _field_enhancement_starter_from_mapping(
    value: object,
    *,
    path: str,
    base: ScenarioCharacterStarter,
    field_name: str,
) -> ScenarioCharacterStarter:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be an object")
    payload = cast(dict[str, object], value)
    starter = replace(
        base,
        name=_required_string(payload, "name", path),
    )
    if field_name == "relationships":
        return replace(
            starter,
            relationships=_enhancement_relationships_from_value(
                payload.get("relationships"),
                path=f"{path}.relationships",
            ),
        )
    evidence_source_ids = (
        _string_tuple(
            payload.get("evidence_source_ids"),
            f"{path}.evidence_source_ids",
        )
        if field_name in CHARACTER_AGENCY_FIELDS
        else ()
    )
    return _replace_enhanced_starter_text_field(
        starter,
        field_name=field_name,
        value=_string(payload.get(field_name)),
        evidence_source_ids=evidence_source_ids,
    )


def _enhancement_relationships_from_value(
    value: object,
    *,
    path: str,
) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{path} must be an object")
        return dict(value)
    if isinstance(value, list):
        relationships: dict[str, object] = {}
        for item in value:
            entry = _enhancement_relationship_entry(item)
            if entry is None:
                continue
            target, relationship = entry
            relationships[target] = relationship
        return relationships
    if isinstance(value, str):
        return _enhancement_relationships_from_text(value)
    raise TypeError(f"{path} must be an object")


def _enhancement_relationship_entry(
    value: object,
) -> tuple[str, object] | None:
    if isinstance(value, dict):
        payload = cast(dict[str, object], value)
        if len(payload) == 1:
            key, relationship = next(iter(payload.items()))
            if isinstance(key, str):
                return _relationship_entry_pair(key, relationship)
        target = _first_relationship_entry_text(
            payload,
            RELATIONSHIP_ENTRY_TARGET_KEYS,
        )
        relationship = _first_relationship_entry_value(
            payload,
            RELATIONSHIP_ENTRY_VALUE_KEYS,
        )
        if target and relationship is not None:
            return target, relationship
        return None
    if isinstance(value, list | tuple) and len(value) == 2:
        target, relationship = value
        if isinstance(target, str):
            return _relationship_entry_pair(target, relationship)
        return None
    if isinstance(value, str):
        relationships = _enhancement_relationships_from_text(value)
        if len(relationships) == 1:
            return next(iter(relationships.items()))
    return None


def _first_relationship_entry_text(
    payload: Mapping[str, object],
    keys: tuple[str, ...],
) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return ""


def _first_relationship_entry_value(
    payload: Mapping[str, object],
    keys: tuple[str, ...],
) -> object | None:
    for key in keys:
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
            continue
        if value is not None:
            return value
    return None


def _enhancement_relationships_from_text(value: str) -> dict[str, object]:
    relationships: dict[str, object] = {}
    for segment in re.split(r"(?:\n+|;)", value):
        entry = _relationship_text_entry(segment)
        if entry is None:
            continue
        target, relationship = entry
        relationships[target] = relationship
    return relationships


def _relationship_text_entry(value: str) -> tuple[str, object] | None:
    text = value.strip()
    if not text:
        return None
    match = re.match(
        r"^(?P<target>[^:\n]{1,120}?)(?:\s*:\s+|\s+(?:--|-)\s+|\s*[\u2013\u2014]\s*)"
        r"(?P<relationship>.+)$",
        text,
    )
    if match is None:
        return None
    return _relationship_entry_pair(
        match.group("target"),
        match.group("relationship"),
    )


def _relationship_entry_pair(
    target: str,
    relationship: object,
) -> tuple[str, object] | None:
    normalized_target = target.strip()
    if not normalized_target:
        return None
    if isinstance(relationship, str):
        text = relationship.strip()
        if not text:
            return None
        return normalized_target, text
    if relationship is None:
        return None
    return normalized_target, relationship


def _replace_enhanced_starter_text_field(
    starter: ScenarioCharacterStarter,
    *,
    field_name: str,
    value: str,
    evidence_source_ids: tuple[str, ...],
) -> ScenarioCharacterStarter:
    if (
        field_name in CHARACTER_FIELD_ENHANCEMENT_FIELDS
        and field_name != "relationships"
    ):
        return replace(
            starter,
            **cast(
                Any,
                {
                    field_name: value,
                    "evidence_source_ids": evidence_source_ids,
                },
            ),
        )
    raise ValueError(f"Unsupported character enhancement field: {field_name}")


def _validate_enhanced_target_field(
    starter: ScenarioCharacterStarter,
    *,
    field_name: str,
    evidence_source_ids: tuple[str, ...] = (),
) -> None:
    if field_name == "relationships":
        if not starter.relationships:
            raise ProviderError(
                ProviderErrorCategory.PROVIDER_ERROR,
                "character.relationships must not be empty.",
            )
        return
    value = getattr(starter, field_name)
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        raise ProviderError(
            ProviderErrorCategory.PROVIDER_ERROR,
            f"character.{field_name} must not be blank.",
        )
    if field_name in CHARACTER_AGENCY_FIELDS:
        if not starter.evidence_source_ids:
            raise ProviderError(
                ProviderErrorCategory.PROVIDER_ERROR,
                "character.evidence_source_ids must cite supporting agency evidence.",
            )
        allowed = set(evidence_source_ids)
        if allowed:
            unknown = [
                source_id
                for source_id in starter.evidence_source_ids
                if source_id not in allowed
            ]
            if unknown:
                raise ProviderError(
                    ProviderErrorCategory.PROVIDER_ERROR,
                    "character.evidence_source_ids contains unknown source IDs: "
                    + ", ".join(unknown),
                )


def _profile_phrase_denylist(
    repositories: PersistenceRepositories | None,
    *,
    save_id: str | None,
) -> tuple[str, ...]:
    if repositories is None or not hasattr(repositories, "get_scoped_setting"):
        return ()
    return effective_generated_phrase_denylist(repositories, save_id=save_id)


def _validate_generated_profile_phrases(
    starters: tuple[ScenarioCharacterStarter, ...],
    *,
    base_starters: tuple[ScenarioCharacterStarter, ...],
    phrase_denylist: tuple[str, ...],
) -> None:
    violations = _generated_profile_phrase_violations(
        starters,
        base_starters=base_starters,
        phrase_denylist=phrase_denylist,
    )
    if violations:
        raise _phrase_denylist_provider_error(violations)


def _generated_profile_phrase_violations(
    starters: tuple[ScenarioCharacterStarter, ...],
    *,
    base_starters: tuple[ScenarioCharacterStarter, ...],
    phrase_denylist: tuple[str, ...],
) -> tuple[PhraseDenylistViolation, ...]:
    if not phrase_denylist:
        return ()
    base_by_key = {
        _character_key(starter.name): starter for starter in base_starters
    }
    violations: list[PhraseDenylistViolation] = []
    for starter in starters:
        base = base_by_key.get(_character_key(starter.name))
        for field_name in PROFILE_PHRASE_DENYLIST_FIELDS:
            value = getattr(starter, field_name)
            base_value = getattr(base, field_name) if base is not None else ""
            generated_value = _generated_profile_field_value(
                before=base_value,
                after=value,
            )
            violations.extend(
                denied_phrase_violations(
                    generated_value,
                    phrases=phrase_denylist,
                    field_name=f"character.{field_name}",
                )
            )
    return tuple(violations)


def _generated_profile_field_value(*, before: str, after: str) -> str:
    if not after:
        return ""
    if before and after == before:
        return ""
    if before and after.startswith(before):
        return after[len(before) :]
    return after


def _phrase_denylist_provider_error(
    violations: tuple[PhraseDenylistViolation, ...],
) -> ProviderError:
    return ProviderError(
        ProviderErrorCategory.PROVIDER_ERROR,
        summarize_phrase_policy_violations(violations),
    )


def _profile_phrase_retry_feedback(
    violations: tuple[PhraseDenylistViolation, ...],
) -> str:
    return (
        "Previous structured response was invalid: "
        f"{summarize_phrase_policy_violations(violations)}. Return corrected "
        "voice and texting_style examples without denied stock phrases or close "
        "variants."
    )


def _field_enhancement_validation_error_code(error: str) -> str:
    normalized = error.casefold()
    if "must not be blank" in normalized:
        return "blank_target_field"
    if "relationships must not be empty" in normalized:
        return "empty_relationships"
    if "evidence_source_ids must cite" in normalized:
        return "missing_evidence_source_ids"
    if "evidence_source_ids contains unknown source ids" in normalized:
        return "unknown_evidence_source_ids"
    if (
        "wrong field" in normalized
        or "field_name is not the requested field" in normalized
    ):
        return "wrong_field"
    if (
        "wrong character" in normalized
        or "name is not the requested character" in normalized
        or "name is not a listed starter" in normalized
    ):
        return "wrong_character"
    if "unknown tool name" in normalized:
        return "unknown_tool"
    if "malformed json arguments" in normalized:
        return "malformed_tool_arguments"
    if "tool arguments must be a json object" in normalized:
        return "invalid_tool_arguments"
    if "invalid character data" in normalized:
        return "invalid_character_data"
    if "missing required field" in normalized:
        return "missing_required_field"
    if "unexpected field" in normalized:
        return "unexpected_field"
    if "must be a" in normalized or "must be an" in normalized:
        return "invalid_field_type"
    return "validation_error"


def _validate_profile_completion_tool_call(
    call: ProviderToolCall,
    *,
    tool_schema: dict[str, object],
    starters_by_key: dict[str, ScenarioCharacterStarter],
    phrase_denylist: tuple[str, ...],
) -> tuple[bool, dict[str, str], ScenarioCharacterStarter | None]:
    if call.name != "complete_character_profile":
        return False, invalid_tool_result(f"Unknown tool name: {call.name}"), None
    arguments, parse_error = parse_tool_arguments_json(call.arguments_json)
    if parse_error is not None or arguments is None:
        return (
            False,
            invalid_tool_result(parse_error or "Tool arguments must be a JSON object"),
            None,
        )
    shape_error = validate_tool_arguments_shape(arguments, schema=tool_schema)
    if shape_error is not None:
        return False, invalid_tool_result(shape_error), None
    try:
        starter = _starter_from_mapping(arguments, path="complete_character_profile")
    except (TypeError, ValueError) as exc:
        return False, invalid_tool_result(str(exc)), None
    if _character_key(starter.name) not in starters_by_key:
        return (
            False,
            invalid_tool_result(f"name is not a listed starter: {starter.name}"),
            None,
        )
    base = starters_by_key[_character_key(starter.name)]
    violations = _generated_profile_phrase_violations(
        (starter,),
        base_starters=(base,),
        phrase_denylist=phrase_denylist,
    )
    if violations:
        return (
            False,
            invalid_tool_result(summarize_phrase_policy_violations(violations)),
            None,
        )
    return True, accepted_tool_result(), starter


def _validate_field_enhancement_tool_call(
    call: ProviderToolCall,
    *,
    tool_schema: dict[str, object],
    base: ScenarioCharacterStarter,
    field_name: str,
    evidence_source_ids: tuple[str, ...] = (),
    phrase_denylist: tuple[str, ...] = (),
) -> tuple[bool, dict[str, str], ScenarioCharacterStarter | None]:
    if call.name != "enhance_character_field":
        return False, invalid_tool_result(f"Unknown tool name: {call.name}"), None
    arguments, parse_error = parse_tool_arguments_json(call.arguments_json)
    if parse_error is not None or arguments is None:
        return (
            False,
            invalid_tool_result(parse_error or "Tool arguments must be a JSON object"),
            None,
        )
    shape_error = validate_tool_arguments_shape(arguments, schema=tool_schema)
    if shape_error is not None:
        return False, invalid_tool_result(shape_error), None
    if arguments.get("field_name") != field_name:
        return (
            False,
            invalid_tool_result("field_name is not the requested field"),
            None,
        )
    try:
        starter = _field_enhancement_starter_from_mapping(
            arguments.get("character"),
            path="enhance_character_field.character",
            base=base,
            field_name=field_name,
        )
    except (TypeError, ValueError) as exc:
        return False, invalid_tool_result(str(exc)), None
    if _character_key(starter.name) != _character_key(base.name):
        return (
            False,
            invalid_tool_result(f"name is not the requested character: {starter.name}"),
            None,
        )
    try:
        _validate_enhanced_target_field(
            starter,
            field_name=field_name,
            evidence_source_ids=evidence_source_ids,
        )
    except ProviderError as exc:
        return False, invalid_tool_result(exc.message or str(exc)), None
    violations = _generated_profile_phrase_violations(
        (starter,),
        base_starters=(base,),
        phrase_denylist=phrase_denylist,
    )
    if violations:
        return (
            False,
            invalid_tool_result(summarize_phrase_policy_violations(violations)),
            None,
        )
    return True, accepted_tool_result(), starter


def _agency_evidence_retry_suffix(field_name: str) -> str:
    if field_name not in CHARACTER_AGENCY_FIELDS:
        return ""
    return " and character.evidence_source_ids populated with supporting source IDs"


def _merge_completed_starters(
    starters: tuple[ScenarioCharacterStarter, ...],
    completed: tuple[ScenarioCharacterStarter, ...],
) -> tuple[ScenarioCharacterStarter, ...]:
    completed_by_key = {_character_key(starter.name): starter for starter in completed}
    return tuple(
        _merge_starter(base, completed_by_key.get(_character_key(base.name)))
        for base in starters
    )


def _merge_starter(
    base: ScenarioCharacterStarter,
    completed: ScenarioCharacterStarter | None,
) -> ScenarioCharacterStarter:
    if completed is None:
        return base
    return ScenarioCharacterStarter(
        name=base.name,
        starter_id=base.starter_id,
        aliases=base.aliases or completed.aliases,
        role=base.role or completed.role,
        age=base.age or completed.age,
        known_state=base.known_state or completed.known_state,
        appearance=base.appearance or completed.appearance,
        visual_notes=base.visual_notes or completed.visual_notes,
        personality=base.personality or completed.personality,
        voice=base.voice or completed.voice,
        texting_style=base.texting_style or completed.texting_style,
        relationships=base.relationships or completed.relationships,
        goals=base.goals or completed.goals,
        motivations=base.motivations or completed.motivations,
        current_intent=base.current_intent or completed.current_intent,
        boundaries=base.boundaries or completed.boundaries,
        attitude_toward_player=(
            base.attitude_toward_player or completed.attitude_toward_player
        ),
        cooperation_conditions=(
            base.cooperation_conditions or completed.cooperation_conditions
        ),
        status=base.status or completed.status,
        met=base.met,
        locked_fields=tuple(
            normalize_character_locked_fields(
                (*base.locked_fields, *completed.locked_fields),
                preserve_unknown=False,
            )
        ),
        evidence_source_ids=base.evidence_source_ids,
        reference_image=base.reference_image,
    )


def _starter_field_is_blank(starter: ScenarioCharacterStarter, field_name: str) -> bool:
    value: object = getattr(starter, field_name)
    return value == "" or value == () or value == [] or value == {}


def _starters_need_profile_completion(
    starters: tuple[ScenarioCharacterStarter, ...],
) -> bool:
    completion_fields = tuple(
        field for field in CHARACTER_PROFILE_COMPLETION_FIELDS if field != "aliases"
    )
    return any(
        _starter_field_is_blank(starter, field)
        for starter in starters
        for field in completion_fields
    )


def _character_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _validated_enhancement_field(field_name: str) -> str:
    normalized = field_name.strip()
    if normalized not in CHARACTER_FIELD_ENHANCEMENT_FIELDS:
        raise ValueError(f"Unsupported character enhancement field: {field_name}")
    return normalized


def _starter_seen_keys(starter: ScenarioCharacterStarter) -> set[str]:
    return {
        key
        for key in (
            _character_key(starter.name),
            *(_character_key(alias) for alias in starter.aliases),
        )
        if key
    }


def _starter_generated_dedupe_keys(starter: ScenarioCharacterStarter) -> set[str]:
    return _starter_seen_keys(starter) | _character_generated_dedupe_keys(
        starter.name,
        *starter.aliases,
    )


def _character_generated_dedupe_keys(*values: str) -> set[str]:
    return {
        key
        for value in values
        for key in (_character_key(value), _character_first_name_key(value))
        if key
    }


def _character_first_name_key(value: str) -> str:
    words = _character_key(value).split()
    if not words:
        return ""
    first_word = re.sub(r"[^a-z]", "", words[0])
    if first_word in CHARACTER_TITLE_WORDS and len(words) > 1:
        return re.sub(r"[^a-z'-]", "", words[1])
    return re.sub(r"[^a-z'-]", "", words[0])
