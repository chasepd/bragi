"""Save-specific scenario detail evolution from structured provider output."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Protocol, cast

from bragi.app_logging import exception_log_fields, log_error_event, log_event
from bragi.persistence.models import (
    JobRecord,
    MessageRecord,
    SaveScenarioUpdateRecord,
    ScenarioRecord,
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
from bragi.services.job_diagnostics import build_job_diagnostic_snapshot
from bragi.services.job_lifecycle import JobLifecycleService
from bragi.services.openrouter_routing_settings import (
    request_with_openrouter_routing,
)
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
from bragi.services.sexual_content_safety import is_fade_to_black_message
from bragi.services.tool_call_helpers import (
    accepted_tool_result,
    append_tool_feedback_messages,
    invalid_tool_result,
    parse_tool_arguments_json,
    validate_tool_arguments_shape,
)

FULL_ROLEPLAY_EVOLVABLE_SECTIONS = frozenset(
    (
        "current_scene",
        "worldbuilding",
        "lore",
        "locations",
        "factions",
    )
)
DATING_SIM_EVOLVABLE_SECTIONS = frozenset(
    (
        "current_scene",
        "player_character_profile",
    )
)
FANTASY_ROLEPLAY_EVOLVABLE_SECTIONS = frozenset(
    (
        "current_scene",
        "magic_system",
        "realms_and_places",
        "factions_and_orders",
        "myths_and_creatures",
        "quest_stakes",
    )
)
SCIENCE_FICTION_ROLEPLAY_EVOLVABLE_SECTIONS = frozenset(
    (
        "current_scene",
        "technology_level",
        "setting_scope",
        "species_and_intelligences",
        "factions_and_institutions",
        "mission_stakes",
    )
)
FIRST_CONTACT_EXPLORATION_EVOLVABLE_SECTIONS = frozenset(
    (
        "current_scene",
        "mission_profile",
        "ship_or_base_status",
        "exploration_target",
        "unknown_intelligence",
        "knowledge_state",
        "translation_progress",
        "discoveries_and_samples",
        "hazards_and_escalation",
    )
)
SURVIVAL_EXPEDITION_EVOLVABLE_SECTIONS = frozenset(
    (
        "current_scene",
        "expedition_goal",
        "route_options",
        "resource_inventory",
        "environmental_conditions",
        "hazards_and_events",
        "camp_status",
        "travel_progress",
    )
)
TIME_LOOP_EVOLVABLE_SECTIONS = frozenset(
    (
        "current_scene",
        "loop_premise",
        "reset_trigger",
        "loop_duration",
        "starting_state",
        "objective",
        "failure_conditions",
        "baseline_world_state",
        "loop_schedule",
        "persistent_knowledge",
        "persistence_exceptions",
        "npc_memory_rules",
        "current_loop_state",
    )
)
INVESTIGATION_MYSTERY_EVOLVABLE_SECTIONS = frozenset(
    (
        "current_scene",
        "locations",
        "factions",
        "case_facts",
        "clues",
        "timeline",
        "red_herrings",
        "hidden_truth",
        "case_status",
    )
)
HEIST_INFILTRATION_EVOLVABLE_SECTIONS = frozenset(
    (
        "current_scene",
        "target_location",
        "objectives_and_stakes",
        "intel_and_access",
        "security_model",
        "alert_and_heat",
        "loadout_and_tools",
        "complications",
        "extraction_routes",
        "aftermath",
    )
)
POLITICAL_INTRIGUE_EVOLVABLE_SECTIONS = frozenset(
    (
        "current_scene",
        "locations",
        "factions",
        "political_arena",
        "political_factions",
        "central_conflict",
        "secrets_and_leverage",
        "reputation_and_standing",
        "obligations_and_favors",
        "alliances_and_rivalries",
        "event_calendar",
        "political_pressure",
        "public_private_knowledge",
    )
)
_SCENARIO_CORE_CONTENT_KEYS = frozenset(
    (
        "title",
        "premise",
        "setup_line",
        "starting_scene",
        "player_character_name",
        "player_role",
    )
)
_SCENARIO_EVOLUTION_CHANGE_TYPES = (
    "phase_shift",
    "no_phase_shift",
    "turn_level_change",
)
_SCENARIO_EVOLUTION_SEMANTIC_SKIP_REASONS = frozenset(
    ("no_phase_shift", "turn_level_change")
)
SCENARIO_EVOLUTION_SECTION_MAX_CHARS = 1200
MAX_SCENARIO_EVOLUTION_TOOL_FEEDBACK_TURNS = MODEL_OUTPUT_MAX_ATTEMPTS - 1


@dataclass(frozen=True)
class ScenarioSectionUpdate:
    section_id: str
    text: str
    reason: str
    source_message_id: str


@dataclass(frozen=True)
class ScenarioEvolution:
    updates: tuple[ScenarioSectionUpdate, ...] = ()
    skip_reason: str | None = None


@dataclass(frozen=True)
class ScenarioEvolutionRequest:
    save_id: str
    messages: tuple[MessageRecord, ...]


class ScenarioEvolver(Protocol):
    async def evolve(
        self,
        request: ScenarioEvolutionRequest,
        *,
        repositories: PersistenceRepositories,
    ) -> ScenarioEvolution: ...


class StructuredProviderScenarioEvolver:
    def __init__(
        self,
        *,
        provider: StructuredOutputProvider,
        provider_name: str,
        model_id: str,
        providers: dict[str, ProviderClient] | None = None,
    ) -> None:
        self.provider = provider
        self.provider_name = provider_name
        self.model_id = model_id
        self.providers = providers

    async def evolve(
        self,
        request: ScenarioEvolutionRequest,
        *,
        repositories: PersistenceRepositories,
    ) -> ScenarioEvolution:
        details = repositories.load_save_details(request.save_id)
        if details is None:
            raise ValueError(f"Unknown save id: {request.save_id}")
        allowed_sections = _evolvable_sections(
            scenario_type=details.scenario.type,
            content=details.scenario.content_json,
        )
        if not allowed_sections:
            return ScenarioEvolution()
        structured_request = request_with_openrouter_routing(
            repositories,
            StructuredOutputRequest(
                provider=self.provider_name,
                model_id=self.model_id,
                schema_name="scenario_evolution",
                schema=_scenario_evolution_schema(
                    allowed_sections=allowed_sections,
                    messages=request.messages,
                ),
                messages=_scenario_evolution_messages(
                    scenario_type=details.scenario.type,
                    scenario_context=_scenario_context_text(details.scenario),
                    messages=request.messages,
                    allowed_sections=allowed_sections,
                ),
                temperature=0.0,
            ),
            task="scenario_evolution",
            save_id=request.save_id,
        )
        if self.providers is None:
            response = await self.provider.generate_structured_output(
                budget_structured_output_request(
                    repositories,
                    structured_request,
                    task="scenario_evolution",
                )
            )
        else:
            response = await structured_output_with_fallback(
                repositories=repositories,
                providers=self.providers,
                request=structured_request,
                task="scenario_evolution",
                save_id=request.save_id,
            )
        return _scenario_evolution_from_structured_data(
            response.data,
            allowed_sections=allowed_sections,
        )


class ToolCallingProviderScenarioEvolver:
    def __init__(
        self,
        *,
        provider: ToolCallProvider,
        provider_name: str,
        model_id: str,
        providers: dict[str, ProviderClient] | None = None,
    ) -> None:
        self.provider = provider
        self.provider_name = provider_name
        self.model_id = model_id
        self.providers = providers

    async def evolve(
        self,
        request: ScenarioEvolutionRequest,
        *,
        repositories: PersistenceRepositories,
    ) -> ScenarioEvolution:
        details = repositories.load_save_details(request.save_id)
        if details is None:
            raise ValueError(f"Unknown save id: {request.save_id}")
        allowed_sections = _evolvable_sections(
            scenario_type=details.scenario.type,
            content=details.scenario.content_json,
        )
        if not allowed_sections:
            return ScenarioEvolution()
        tool_request = request_with_openrouter_routing(
            repositories,
            ToolCallRequest(
                provider=self.provider_name,
                model_id=self.model_id,
                messages=_scenario_evolution_tool_messages(
                    scenario_type=details.scenario.type,
                    scenario_context=_scenario_context_text(details.scenario),
                    messages=request.messages,
                    allowed_sections=allowed_sections,
                ),
                tools=_scenario_evolution_tool_definitions(
                    allowed_sections=allowed_sections,
                    messages=request.messages,
                ),
                temperature=0.0,
            ),
            task="scenario_evolution",
            save_id=request.save_id,
        )
        try:
            if self.providers is None:
                return await _scenario_evolution_with_tool_feedback(
                    repositories=repositories,
                    provider=self.provider,
                    request=tool_request,
                    allowed_sections=allowed_sections,
                    source_message_ids=tuple(
                        message.id for message in request.messages
                    ),
                )
            return await _scenario_evolution_with_tool_fallback(
                repositories=repositories,
                providers=self.providers,
                provider=self.provider,
                request=tool_request,
                save_id=request.save_id,
                allowed_sections=allowed_sections,
                source_message_ids=tuple(
                    message.id for message in request.messages
                ),
            )
        except ProviderError as exc:
            # The tool fallback chain enriches the failing error; the enriched
            # error reports model_not_found when either the primary or the
            # fallback attempt failed with it, so recovering through the
            # structured route covers both cases.
            if not provider_error_is_model_not_found(exc):
                raise
            return await self._evolve_via_structured_shape(
                request=request,
                repositories=repositories,
                error=exc,
            )

    async def _evolve_via_structured_shape(
        self,
        *,
        request: ScenarioEvolutionRequest,
        repositories: PersistenceRepositories,
        error: ProviderError,
    ) -> ScenarioEvolution:
        structured_evolver = StructuredProviderScenarioEvolver(
            provider=cast(StructuredOutputProvider, self.provider),
            provider_name=self.provider_name,
            model_id=self.model_id,
            providers=self.providers,
        )

        async def structured_run() -> ScenarioEvolution:
            if not isinstance(self.provider, StructuredOutputProvider):
                raise ValueError("Scenario evolution provider lacks structured output")
            return await structured_evolver.evolve(
                request,
                repositories=repositories,
            )

        return await recover_tool_call_shape_with_structured_output(
            error=error,
            task="scenario_evolution",
            provider=self.provider_name,
            model_id=self.model_id,
            structured_run=structured_run,
        )


class ScenarioEvolutionService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        evolver: ScenarioEvolver,
        provider_name: str,
        model_id: str,
    ) -> None:
        self.repositories = repositories
        self.evolver = evolver
        self.provider_name = provider_name
        self.model_id = model_id
        self.jobs = JobLifecycleService(repositories=repositories)

    async def evolve_after_turn(
        self,
        *,
        save_id: str,
        source_message_ids: tuple[str, ...],
    ) -> SaveScenarioUpdateRecord | None:
        job = self.jobs.create_running(
            save_id=save_id,
            type="scenario_evolution",
            payload={"source_message_ids": list(source_message_ids)},
            collect_provider_diagnostics=True,
        )
        log_event(
            "job.running",
            job_id=job.id,
            job_type=job.type,
            save_id=save_id,
            source_message_count=len(source_message_ids),
        )
        messages = tuple(
            message
            for message in self.repositories.list_messages(save_id)
            if message.id in set(source_message_ids)
        )
        started_at = perf_counter()
        if any(
            is_fade_to_black_message(
                role=message.role,
                body=message.body,
                safety_transition=message.safety_transition,
            )
            for message in messages
        ):
            self.jobs.succeed(
                job.id,
                result={
                    "scenario_update_id": None,
                    "section_update_count": 0,
                    "skip_reason": "safety_transition",
                },
            )
            return None
        try:
            evolution = await self.evolver.evolve(
                ScenarioEvolutionRequest(save_id=save_id, messages=messages),
                repositories=self.repositories,
            )
            update = self.apply_evolution(
                save_id=save_id,
                evolution=evolution,
                allowed_source_message_ids=tuple(message.id for message in messages),
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
        result: dict[str, object] = {
            "scenario_update_id": update.id if update is not None else None,
            "section_update_count": len(evolution.updates),
        }
        skip_reason = _evolution_skip_reason(evolution=evolution, update=update)
        if skip_reason is not None:
            result["skip_reason"] = skip_reason
        self.jobs.succeed(job.id, result=result)
        log_event(
            "job.succeeded",
            job_id=job.id,
            job_type=job.type,
            save_id=save_id,
            duration_ms=_elapsed_ms(started_at),
            section_update_count=len(evolution.updates),
            scenario_update_id=update.id if update is not None else None,
            skip_reason=skip_reason,
        )
        return update

    def apply_evolution(
        self,
        *,
        save_id: str,
        evolution: ScenarioEvolution,
        allowed_source_message_ids: tuple[str, ...] | None = None,
    ) -> SaveScenarioUpdateRecord | None:
        details = self.repositories.load_save_details(save_id)
        if details is None:
            raise ValueError(f"Unknown save id: {save_id}")
        allowed_sections = _evolvable_sections(
            scenario_type=details.scenario.type,
            content=details.scenario.content_json,
        )
        evolution = _repair_evolution_source_message_ids(
            evolution,
            allowed_source_message_ids=allowed_source_message_ids,
        )
        _validate_evolution(
            evolution,
            allowed_sections=allowed_sections,
            allowed_source_message_ids=allowed_source_message_ids,
        )
        if not evolution.updates:
            return None

        content = _scenario_content(details.scenario.content_json)
        changed_sections: list[ScenarioSectionUpdate] = []
        for update in evolution.updates:
            if content.get(update.section_id) == update.text:
                continue
            content[update.section_id] = update.text
            changed_sections.append(update)
        if not changed_sections:
            return None

        reason = "; ".join(
            f"{update.section_id}: {update.reason}".strip()
            for update in changed_sections
            if update.reason.strip()
        )
        source_message_id = changed_sections[-1].source_message_id
        source_message_ids = tuple(
            dict.fromkeys(update.source_message_id for update in changed_sections)
        )
        return self.repositories.add_save_scenario_update(
            save_id=save_id,
            title=details.scenario.title,
            premise=details.scenario.premise,
            player_role=details.scenario.player_role,
            content=content,
            reason=reason,
            provider=self.provider_name,
            model=self.model_id,
            source_message_id=source_message_id,
            source_message_ids=source_message_ids,
        )


def record_scenario_evolution_skip(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    source_message_ids: tuple[str, ...],
    skip_reason: str,
    result: dict[str, object] | None = None,
) -> JobRecord:
    job = repositories.create_job(
        save_id=save_id,
        type="scenario_evolution",
        status="running",
        payload={"source_message_ids": list(source_message_ids)},
    )
    log_event(
        "job.running",
        job_id=job.id,
        job_type=job.type,
        save_id=save_id,
        source_message_count=len(source_message_ids),
        skip_reason=skip_reason,
    )
    job_result: dict[str, object] = {
        "scenario_update_id": None,
        "section_update_count": 0,
        "skip_reason": skip_reason,
        **(result or {}),
    }
    updated = repositories.update_job(
        job.id,
        status="succeeded",
        result=job_result,
    )
    updated = repositories.set_job_diagnostics(
        updated.id,
        build_job_diagnostic_snapshot(updated, result=job_result),
    )
    log_event(
        "job.succeeded",
        job_id=job.id,
        job_type=job.type,
        save_id=save_id,
        section_update_count=0,
        scenario_update_id=None,
        skip_reason=skip_reason,
    )
    return updated


def _scenario_evolution_schema(
    *,
    allowed_sections: tuple[str, ...],
    messages: tuple[MessageRecord, ...],
) -> dict[str, object]:
    message_ids = [message.id for message in messages]
    source_schema: dict[str, object] = {"type": "string"}
    if message_ids:
        source_schema["enum"] = message_ids
    return normalize_strict_json_schema({
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "change_type": {
                "type": "string",
                "enum": list(_SCENARIO_EVOLUTION_CHANGE_TYPES),
            },
            "skip_reason": {
                "type": "string",
                "enum": sorted(_SCENARIO_EVOLUTION_SEMANTIC_SKIP_REASONS),
            },
            "content": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    section_id: {"type": "string"} for section_id in allowed_sections
                },
            },
            "reason": {"type": "string"},
            "source_message_id": source_schema,
            "updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "section_id": {
                            "type": "string",
                            "enum": list(allowed_sections),
                        },
                        "text": {"type": "string"},
                        "reason": {"type": "string"},
                        "source_message_id": source_schema,
                    },
                    "required": [
                        "section_id",
                        "text",
                        "reason",
                        "source_message_id",
                    ],
                },
            },
        },
        "required": ["change_type", "updates"],
    })


def _scenario_evolution_messages(
    *,
    scenario_type: str,
    scenario_context: str,
    messages: tuple[MessageRecord, ...],
    allowed_sections: tuple[str, ...],
) -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(
            role="system",
            body=_scenario_evolution_instruction(scenario_type),
        ),
        ChatMessage(
            role="user",
            body="\n\n".join(
                (
                    scenario_context,
                    "Evolvable sections: " + ", ".join(allowed_sections),
                    _messages_text(messages),
                )
            ),
        ),
    )


def _scenario_evolution_tool_messages(
    *,
    scenario_type: str,
    scenario_context: str,
    messages: tuple[MessageRecord, ...],
    allowed_sections: tuple[str, ...],
) -> tuple[ToolCallMessage, ...]:
    return tuple(
        ToolCallMessage(
            role=message.role,
            body=message.body.replace(
                "Use the enforced schema.",
                (
                    "Use update_scenario_section for durable phase-shift "
                    "section updates, or skip_scenario_evolution when no "
                    "phase-shift update is appropriate. Do not write prose "
                    "outside tool calls."
                ),
            ),
            speaker_name=message.speaker_name,
        )
        for message in _scenario_evolution_messages(
            scenario_type=scenario_type,
            scenario_context=scenario_context,
            messages=messages,
            allowed_sections=allowed_sections,
        )
    )


def _scenario_evolution_tool_definitions(
    *,
    allowed_sections: tuple[str, ...],
    messages: tuple[MessageRecord, ...],
) -> tuple[ToolDefinition, ...]:
    message_ids = [message.id for message in messages]
    source_schema: dict[str, object] = {"type": "string"}
    if message_ids:
        source_schema["enum"] = message_ids
    return (
        ToolDefinition(
            name="update_scenario_section",
            description="Replace one evolvable scenario section after a phase shift.",
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "section_id": {
                        "type": "string",
                        "enum": list(allowed_sections),
                    },
                    "text": {"type": "string"},
                    "reason": {"type": "string"},
                    "source_message_id": source_schema,
                },
                "required": ["section_id", "text", "reason", "source_message_id"],
            },
        ),
        ToolDefinition(
            name="skip_scenario_evolution",
            description="Record that this turn does not require scenario evolution.",
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "change_type": {
                        "type": "string",
                        "enum": list(_SCENARIO_EVOLUTION_SEMANTIC_SKIP_REASONS),
                    },
                    "skip_reason": {
                        "type": "string",
                        "enum": sorted(_SCENARIO_EVOLUTION_SEMANTIC_SKIP_REASONS),
                    },
                },
                "required": ["change_type", "skip_reason"],
            },
        ),
    )


async def _scenario_evolution_with_tool_fallback(
    *,
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    provider: ToolCallProvider,
    request: ToolCallRequest,
    save_id: str,
    allowed_sections: tuple[str, ...],
    source_message_ids: tuple[str, ...],
) -> ScenarioEvolution:
    try:
        return await _scenario_evolution_with_tool_feedback(
            repositories=repositories,
            provider=provider,
            request=request,
            allowed_sections=allowed_sections,
            source_message_ids=source_message_ids,
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
            raise provider_error_with_fallback_skipped_reason(exc, reason) from exc
        fallback_provider = providers[fallback_request.provider]
        if not isinstance(fallback_provider, ToolCallProvider):
            reason = "fallback_provider_unavailable"
            raise provider_error_with_fallback_skipped_reason(exc, reason) from exc
        try:
            return await _scenario_evolution_with_tool_feedback(
                repositories=repositories,
                provider=fallback_provider,
                request=fallback_request,
                allowed_sections=allowed_sections,
                source_message_ids=source_message_ids,
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


async def _scenario_evolution_with_tool_feedback(
    *,
    repositories: PersistenceRepositories,
    provider: ToolCallProvider,
    request: ToolCallRequest,
    allowed_sections: tuple[str, ...],
    source_message_ids: tuple[str, ...],
) -> ScenarioEvolution:
    messages = list(request.messages)
    tool_schemas = {tool.name: tool.parameters for tool in request.tools}
    updates: list[ScenarioSectionUpdate] = []
    accepted_sections: set[str] = set()
    skip_reason: str | None = None
    last_errors: list[str] = []
    max_attempt_count = configured_max_attempts(repositories)

    for _turn in range(max_attempt_count):
        turn_request = budget_tool_call_request(
            repositories,
            replace(request, messages=tuple(messages)),
            task="scenario_evolution",
        )
        response = await provider.generate_tool_calls(turn_request)
        errors: list[str] = []
        turn_results: list[
            tuple[
                ProviderToolCall,
                dict[str, str],
                ScenarioSectionUpdate | str | None,
                bool,
            ]
        ] = []
        turn_updates: list[ScenarioSectionUpdate] = []
        turn_has_skip_call = any(
            call.name == "skip_scenario_evolution" for call in response.tool_calls
        )
        for call in response.tool_calls:
            accepted, result, item = _validate_scenario_evolution_tool_call(
                call,
                tool_schemas=tool_schemas,
                allowed_sections=allowed_sections,
                source_message_ids=source_message_ids,
            )
            if accepted:
                if isinstance(item, ScenarioSectionUpdate):
                    turn_updates.append(item)
                turn_results.append((call, accepted_tool_result(), item, True))
                continue
            errors.append(result["error"])
            turn_results.append((call, result, None, False))

        conflict_error: str | None = None
        if turn_has_skip_call and (
            updates or turn_updates or len(response.tool_calls) > 1
        ):
            conflict_error = "Scenario evolution cannot both skip and update sections"
        elif skip_reason is not None and turn_updates:
            conflict_error = "Scenario evolution cannot both skip and update sections"
        if conflict_error is not None:
            errors.append(conflict_error)

        tool_results: list[tuple[ProviderToolCall, dict[str, str]]] = []
        if conflict_error is None:
            for call, result, item, accepted in turn_results:
                if accepted:
                    if isinstance(item, ScenarioSectionUpdate):
                        if item.section_id not in accepted_sections:
                            accepted_sections.add(item.section_id)
                            updates.append(item)
                    elif isinstance(item, str):
                        skip_reason = item
                tool_results.append((call, result))
        else:
            for call, _result, _item, _accepted in turn_results:
                tool_results.append(
                    (
                        call,
                        invalid_tool_result(conflict_error),
                    )
                )

        if not response.tool_calls and skip_reason is None and not updates:
            skip_reason = "no_phase_shift"
        if not errors:
            evolution = ScenarioEvolution(
                updates=tuple(updates),
                skip_reason=skip_reason,
            )
            _validate_evolution(
                evolution,
                allowed_sections=allowed_sections,
                allowed_source_message_ids=source_message_ids,
            )
            return evolution

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
            "Scenario evolution tool-call validation failed after feedback: "
            + "; ".join(last_errors)
        ),
    )


def _validate_scenario_evolution_tool_call(
    call: ProviderToolCall,
    *,
    tool_schemas: dict[str, dict[str, object]],
    allowed_sections: tuple[str, ...],
    source_message_ids: tuple[str, ...],
) -> tuple[bool, dict[str, str], ScenarioSectionUpdate | str | None]:
    schema = tool_schemas.get(call.name)
    if schema is None:
        return _invalid_scenario_evolution_tool_call(f"Unknown tool name: {call.name}")
    arguments, parse_error = parse_tool_arguments_json(call.arguments_json)
    if parse_error is not None or arguments is None:
        return _invalid_scenario_evolution_tool_call(
            parse_error or "Tool arguments must be a JSON object"
        )
    shape_error = validate_tool_arguments_shape(arguments, schema=schema)
    if shape_error is not None:
        return _invalid_scenario_evolution_tool_call(shape_error)
    if call.name == "skip_scenario_evolution":
        skip_reason = str(
            arguments.get("skip_reason") or arguments.get("change_type") or ""
        ).strip()
        if skip_reason not in _SCENARIO_EVOLUTION_SEMANTIC_SKIP_REASONS:
            return _invalid_scenario_evolution_tool_call(
                f"Unknown scenario evolution skip_reason: {skip_reason}"
            )
        return True, accepted_tool_result(), skip_reason
    update = _section_update_from_data(arguments)
    if update.section_id not in set(allowed_sections):
        return _invalid_scenario_evolution_tool_call(
            f"Scenario section cannot evolve: {update.section_id}"
        )
    if update.source_message_id not in set(source_message_ids):
        return _invalid_scenario_evolution_tool_call(
            f"Unknown scenario update source_message_id: {update.source_message_id}"
        )
    if len(update.text) > SCENARIO_EVOLUTION_SECTION_MAX_CHARS:
        return _invalid_scenario_evolution_tool_call(
            "Scenario section update text is too long"
        )
    if not update.text:
        return _invalid_scenario_evolution_tool_call(
            "Scenario section update text is required"
        )
    if not update.reason:
        return _invalid_scenario_evolution_tool_call(
            "Scenario section update reason is required"
        )
    return True, accepted_tool_result(), update


def _invalid_scenario_evolution_tool_call(
    error: str,
) -> tuple[bool, dict[str, str], None]:
    return False, invalid_tool_result(error), None


def _scenario_evolution_instruction(scenario_type: str) -> str:
    base = (
        "Detect whether the completed turn makes save-specific scenario details "
        "stale. Use the enforced schema. Set change_type to phase_shift only "
        "when durable setup or campaign phase has changed enough to supersede "
        "future narrator scenario context. Set change_type to turn_level_change "
        "for ordinary emotional, relationship, tactical, conversational, mood, "
        "or momentary scene beats that belong in character, thread, memory, "
        "world state, or scene snapshot records. Set change_type to "
        "no_phase_shift when nothing durable changed. Return section updates "
        "only when change_type is phase_shift and only for sections whose "
        "operational context is clearly superseded by the messages. Do not "
        "rewrite title, player role, core premise, tone/style, or opening "
        "message. Preserve the scenario identity and update only durable "
        "context needed by future narrator prompts."
    )
    if scenario_type == "dating_sim":
        return (
            base + " For dating sims, update only durable changes to the player "
            "profile, romance option profiles, relationship route baselines, or "
            "current scene setup that have truly changed."
        )
    if scenario_type == "fantasy_roleplay":
        return (
            base + " For fantasy roleplays, update only durable changes to magic "
            "rules, realms or places, factions or orders, myths or creatures, "
            "quest stakes, or current scene setup that have truly changed."
        )
    if scenario_type == "science_fiction_roleplay":
        return (
            base + " For science fiction roleplays, update only durable changes "
            "to technology constraints, setting scope, species or intelligences, "
            "factions or institutions, mission stakes, or current scene setup "
            "that have truly changed."
        )
    if scenario_type == "first_contact_exploration":
        return (
            base + " For first-contact and exploration roleplays, update only "
            "durable changes to mission status, crew or command constraints, "
            "ship or base condition, exploration target observations, unknown "
            "intelligence behavior, observed facts, hypotheses, misunderstandings, "
            "confirmed knowledge, translation progress, discoveries, samples, "
            "hazards, escalation clocks, or current scene setup that have truly "
            "changed. Preserve uncertainty and avoid premature exposition."
        )
    if scenario_type == "survival_expedition":
        return (
            base + " For survival expeditions, update only durable changes to "
            "expedition goal, route options, party roster or survival status, "
            "resources, environmental conditions, hazards, camp status, travel "
            "progress, or current scene setup that have truly changed."
        )
    if scenario_type == "time_loop":
        return (
            base + " For time loops, update only durable changes to loop rules, "
            "reset trigger or duration, starting state, objective, failure "
            "conditions, resettable baseline state, schedules, persistent "
            "player/meta knowledge, persistence exceptions, NPC memory rules, "
            "current loop counter or phase, prior-loop summary, deviations, or "
            "current scene setup that have truly changed. Keep resettable world "
            "state separate from persistent knowledge."
        )
    if scenario_type == "investigation_mystery":
        return (
            base + " For investigation mysteries, update only durable changes "
            "to discovered clues, suspects, known public timeline, intentional "
            "red herrings, case status, faction, location, or character case "
            "context, or current scene setup that have truly changed. Do not "
            "rewrite hidden truth unless completed play has explicitly "
            "established that the prior hidden truth was incomplete or wrong."
        )
    if scenario_type == "heist_infiltration":
        return (
            base + " For heist and infiltration roleplays, update only durable "
            "changes to target access, objectives, crew or contact status, intel, "
            "security model, alert or heat state, loadout, complications, "
            "extraction routes, aftermath consequences, or current scene setup "
            "that have truly changed. Use world state for tactical moment-to-moment "
            "guard movement, alarm toggles, inventory deltas, and pursuit pressure."
        )
    if scenario_type == "political_intrigue":
        return (
            base + " For political intrigue roleplays, update only durable "
            "changes to the political arena, faction positions, notable "
            "loyalties or grudges, central conflict, secrets or leverage, "
            "reputation or standing, obligations, favors, alliances, rivalries, "
            "event calendar, timed pressure, public/private knowledge boundaries, "
            "or current scene setup that have truly changed. Use world state for "
            "concrete favor debts, faction standing deltas, promises, blackmail "
            "terms, and timed pressure changes."
        )
    return (
        base + " For generic roleplays, update locations, factions, lore, "
        "worldbuilding, or starting/current scene setup when play has clearly "
        "moved beyond the original setup. Prefer world-data records for concrete "
        "facts; use scenario section updates as short high-level rollups only."
    )


def _scenario_context_text(scenario: ScenarioRecord) -> str:
    title = scenario.title
    premise = scenario.premise
    player_role = scenario.player_role
    content_json = scenario.content_json
    lines = [
        "Current save-specific scenario context:",
        f"- title: {title}",
        f"- premise/setup: {premise}",
        f"- player role: {player_role}",
    ]
    for key, value in _scenario_content(content_json).items():
        if key in _SCENARIO_CORE_CONTENT_KEYS:
            continue
        if value:
            lines.append(
                f"- {key}: "
                f"{_compact_text(str(value), SCENARIO_EVOLUTION_SECTION_MAX_CHARS)}"
            )
    return "\n".join(lines)


def _messages_text(messages: tuple[MessageRecord, ...]) -> str:
    if not messages:
        return "Completed turn messages: none"
    return "Completed turn messages:\n" + "\n".join(
        f"- {message.id} [{message.role}] {message.body}" for message in messages
    )


def _scenario_content(content_json: str) -> dict[str, object]:
    try:
        loaded = json.loads(content_json)
    except json.JSONDecodeError:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return deepcopy(loaded)


def _compact_text(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "..."


def _evolvable_sections(
    *,
    scenario_type: str,
    content: str,
) -> tuple[str, ...]:
    if scenario_type == "dating_sim":
        allowed = DATING_SIM_EVOLVABLE_SECTIONS
    elif scenario_type == "fantasy_roleplay":
        allowed = FANTASY_ROLEPLAY_EVOLVABLE_SECTIONS
    elif scenario_type == "science_fiction_roleplay":
        allowed = SCIENCE_FICTION_ROLEPLAY_EVOLVABLE_SECTIONS
    elif scenario_type == "first_contact_exploration":
        allowed = FIRST_CONTACT_EXPLORATION_EVOLVABLE_SECTIONS
    elif scenario_type == "survival_expedition":
        allowed = SURVIVAL_EXPEDITION_EVOLVABLE_SECTIONS
    elif scenario_type == "time_loop":
        allowed = TIME_LOOP_EVOLVABLE_SECTIONS
    elif scenario_type == "investigation_mystery":
        allowed = INVESTIGATION_MYSTERY_EVOLVABLE_SECTIONS
    elif scenario_type == "heist_infiltration":
        allowed = HEIST_INFILTRATION_EVOLVABLE_SECTIONS
    elif scenario_type == "political_intrigue":
        allowed = POLITICAL_INTRIGUE_EVOLVABLE_SECTIONS
    else:
        allowed = FULL_ROLEPLAY_EVOLVABLE_SECTIONS
    return tuple(sorted(allowed))


def _scenario_evolution_from_structured_data(
    data: dict[str, object],
    *,
    allowed_sections: tuple[str, ...],
) -> ScenarioEvolution:
    change_type = _change_type_from_data(data)
    if change_type in _SCENARIO_EVOLUTION_SEMANTIC_SKIP_REASONS:
        return ScenarioEvolution(skip_reason=change_type)

    content_updates = _content_updates_from_data(
        data,
        allowed_sections=allowed_sections,
    )
    if content_updates:
        evolution = ScenarioEvolution(updates=tuple(content_updates))
        _validate_evolution(
            evolution,
            allowed_sections=allowed_sections,
            allowed_source_message_ids=None,
        )
        return evolution

    raw_updates = data.get("updates", [])
    if not isinstance(raw_updates, list):
        raise ValueError("Structured scenario evolution updates must be a list")
    updates = tuple(_section_update_from_data(item) for item in raw_updates)
    skip_reason = _semantic_skip_reason_from_data(data)
    if not updates and skip_reason is None:
        skip_reason = "no_phase_shift"
    evolution = ScenarioEvolution(
        updates=updates,
        skip_reason=skip_reason,
    )
    _validate_evolution(
        evolution,
        allowed_sections=allowed_sections,
        allowed_source_message_ids=None,
    )
    return evolution


def _change_type_from_data(data: dict[str, object]) -> str | None:
    value = data.get("change_type")
    if value is None:
        return None
    change_type = str(value).strip()
    if change_type not in _SCENARIO_EVOLUTION_CHANGE_TYPES:
        raise ValueError(f"Unknown scenario evolution change_type: {change_type}")
    return change_type


def _semantic_skip_reason_from_data(data: dict[str, object]) -> str | None:
    change_type = _change_type_from_data(data)
    if change_type in _SCENARIO_EVOLUTION_SEMANTIC_SKIP_REASONS:
        return change_type
    value = data.get("skip_reason")
    if value is None:
        return None
    skip_reason = str(value).strip()
    if skip_reason not in _SCENARIO_EVOLUTION_SEMANTIC_SKIP_REASONS:
        raise ValueError(f"Unknown scenario evolution skip_reason: {skip_reason}")
    return skip_reason


def _content_updates_from_data(
    data: dict[str, object],
    *,
    allowed_sections: tuple[str, ...],
) -> list[ScenarioSectionUpdate]:
    raw_content = data.get("content")
    if not isinstance(raw_content, dict):
        return []
    unknown_sections = set(str(section_id) for section_id in raw_content) - set(
        allowed_sections
    )
    if unknown_sections:
        raise ValueError(
            f"Scenario section cannot evolve: {sorted(unknown_sections)[0]}"
        )
    reason = str(data.get("reason", "")).strip()
    source_message_id = str(data.get("source_message_id", ""))
    allowed = set(allowed_sections)
    updates: list[ScenarioSectionUpdate] = []
    for section_id, value in raw_content.items():
        section = str(section_id)
        if section not in allowed:
            continue
        updates.append(
            ScenarioSectionUpdate(
                section_id=section,
                text=str(value).strip(),
                reason=reason,
                source_message_id=source_message_id,
            )
        )
    return updates


def _section_update_from_data(value: object) -> ScenarioSectionUpdate:
    if not isinstance(value, dict):
        raise ValueError("Structured scenario section update must be an object")
    return ScenarioSectionUpdate(
        section_id=str(value.get("section_id", "")),
        text=str(value.get("text", "")).strip(),
        reason=str(value.get("reason", "")).strip(),
        source_message_id=str(value.get("source_message_id", "")),
    )


def _validate_evolution(
    evolution: ScenarioEvolution,
    *,
    allowed_sections: tuple[str, ...],
    allowed_source_message_ids: tuple[str, ...] | None,
) -> None:
    if (
        evolution.skip_reason is not None
        and evolution.skip_reason not in _SCENARIO_EVOLUTION_SEMANTIC_SKIP_REASONS
    ):
        raise ValueError(
            f"Unknown scenario evolution skip_reason: {evolution.skip_reason}"
        )
    allowed_section_set = set(allowed_sections)
    allowed_message_set = set(allowed_source_message_ids or ())
    seen_sections: set[str] = set()
    for update in evolution.updates:
        if update.section_id not in allowed_section_set:
            raise ValueError(f"Scenario section cannot evolve: {update.section_id}")
        if update.section_id in seen_sections:
            raise ValueError(f"Duplicate scenario section update: {update.section_id}")
        seen_sections.add(update.section_id)
        if not update.text:
            raise ValueError("Scenario section update text is required")
        if len(update.text) > SCENARIO_EVOLUTION_SECTION_MAX_CHARS:
            raise ValueError("Scenario section update text is too long")
        if not update.reason:
            raise ValueError("Scenario section update reason is required")
        if allowed_source_message_ids is not None and (
            update.source_message_id not in allowed_message_set
        ):
            raise ValueError(
                f"Unknown scenario update source_message_id: {update.source_message_id}"
            )


def _repair_evolution_source_message_ids(
    evolution: ScenarioEvolution,
    *,
    allowed_source_message_ids: tuple[str, ...] | None,
) -> ScenarioEvolution:
    if allowed_source_message_ids is None:
        return evolution
    allowed = set(allowed_source_message_ids)
    fallback_source_id = (
        allowed_source_message_ids[-1] if allowed_source_message_ids else ""
    )
    updates: list[ScenarioSectionUpdate] = []
    for update in evolution.updates:
        if update.source_message_id in allowed:
            updates.append(update)
        elif fallback_source_id:
            updates.append(
                ScenarioSectionUpdate(
                    section_id=update.section_id,
                    text=update.text,
                    reason=update.reason,
                    source_message_id=fallback_source_id,
                )
            )
    return ScenarioEvolution(updates=tuple(updates), skip_reason=evolution.skip_reason)


def _evolution_skip_reason(
    *,
    evolution: ScenarioEvolution,
    update: SaveScenarioUpdateRecord | None,
) -> str | None:
    if update is not None:
        return None
    return evolution.skip_reason or "no_phase_shift"


def _elapsed_ms(started_at: float) -> int:
    return int((perf_counter() - started_at) * 1000)
