"""Per-character pre-narrator action planning."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, cast

from bragi.app_logging import exception_log_fields, log_error_event
from bragi.interaction_mode import InteractionMode
from bragi.persistence.models import (
    ActiveThreadRecord,
    CharacterRecord,
    DatingRouteStateRecord,
    MessageRecord,
    ModelPreferenceRecord,
)
from bragi.persistence.repositories import (
    CHARACTER_KNOWLEDGE_ACQUISITION_METHODS,
    CHARACTER_KNOWLEDGE_STATES,
    PersistenceRepositories,
)
from bragi.providers.contracts import (
    ChatMessage,
    ProviderClient,
    StructuredOutputProvider,
    StructuredOutputRequest,
)
from bragi.services.active_thread_lifecycle import (
    active_thread_is_prompt_visible,
    normalize_active_thread_status,
)
from bragi.services.context_assembly import scenario_section_candidates
from bragi.services.dating_route_policy import (
    escalation_policy_for_stage,
    intimacy_profile_guidance,
)
from bragi.services.evidence import quote_matches_source
from bragi.services.knowledge_boundary import message_visible_to_present_characters
from bragi.services.mention_matching import character_name_is_mentioned
from bragi.services.model_capabilities import (
    MODEL_LACKS_CAPABILITY_REASON,
    MODEL_MISSING_REASON,
    MODEL_UNAVAILABLE_REASON,
    STRUCTURED_OUTPUT_CAPABILITIES,
    check_model_capabilities,
)
from bragi.services.model_preferences import (
    CHARACTER_INTENT_PLANNING_PURPOSE,
    CHARACTER_PRESENCE_ASSESSMENT_PURPOSE,
    roleplay_model_preference_with_fallbacks,
)
from bragi.services.provider_fallbacks import structured_output_with_fallback
from bragi.world_time_model import format_world_time_from_snapshot

CHARACTER_ACTION_PLANNING_TASK = "character_action_planning"
CHARACTER_PRESENCE_ASSESSMENT_TASK = CHARACTER_PRESENCE_ASSESSMENT_PURPOSE
CHARACTER_INTENT_PLANNING_TASK = CHARACTER_INTENT_PLANNING_PURPOSE
CHARACTER_ACTION_PLANNING_ENABLED_SETTING = "character_action_planning_enabled"
CHARACTER_ACTION_PLANNING_ENABLED_DEFAULT = True
CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY_SETTING = (
    "character_action_planning_max_concurrency"
)
DEFAULT_CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY = 20
MIN_CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY = 1
MAX_CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY = 20
CHARACTER_ACTION_PLANNING_RECENT_MESSAGE_LIMIT = 10


@dataclass(frozen=True)
class CharacterActionPlan:
    character_id: str
    character_name: str
    action: str
    intent: str = ""
    reason: str = ""
    confidence: float = 0.0
    evidence_source_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CharacterLearnedMemoryCandidate:
    body: str
    tags: tuple[str, ...] = ()
    knowledge_state: str = "knows"
    acquisition_method: str = "unknown"
    reason: str = ""
    confidence: float = 0.0
    evidence_source_ids: tuple[str, ...] = ()
    evidence_quote: str = ""


@dataclass(frozen=True)
class CharacterKnowledgeEdgeCandidate:
    target_type: str
    target_id: str
    knowledge_state: str = "knows"
    acquisition_method: str = "unknown"
    reason: str = ""
    confidence: float = 0.0
    evidence_source_ids: tuple[str, ...] = ()
    evidence_quote: str = ""


@dataclass(frozen=True)
class CharacterTurnAssessment:
    character_id: str
    character_name: str
    present: bool
    action: str = ""
    intent: str = ""
    reason: str = ""
    confidence: float = 0.0
    evidence_source_ids: tuple[str, ...] = ()
    evidence_quote: str = ""
    presence_evidence_source_ids: tuple[str, ...] = ()
    presence_evidence_quote: str = ""
    enters_scene: bool = False
    leaves_scene: bool = False
    learned_memory_candidates: tuple[CharacterLearnedMemoryCandidate, ...] = ()
    knowledge_edge_candidates: tuple[CharacterKnowledgeEdgeCandidate, ...] = ()
    needs_review_notes: tuple[str, ...] = ()


CharacterPresenceDecision = CharacterTurnAssessment


@dataclass(frozen=True)
class CharacterActionPlanningResult:
    plans: tuple[CharacterActionPlan, ...] = ()
    decisions: tuple[CharacterTurnAssessment, ...] = ()
    failed_character_ids: tuple[str, ...] = ()
    skipped_reason: str = ""
    applied_presence_update: bool = False

    @property
    def assessments(self) -> tuple[CharacterTurnAssessment, ...]:
        return self.decisions


class CharacterActionPlanningService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        providers: dict[str, ProviderClient],
    ) -> None:
        self.repositories = repositories
        self.providers = providers

    async def plan_for_turn(
        self,
        *,
        save_id: str,
        player_message_id: str,
        apply_presence_updates: bool = True,
    ) -> CharacterActionPlanningResult:
        if not character_action_planning_enabled(
            self.repositories,
            save_id=save_id,
        ):
            return CharacterActionPlanningResult(skipped_reason="disabled")
        details = self.repositories.load_save_details(save_id)
        if details is None:
            raise ValueError(f"Unknown save id: {save_id}")
        source_message = _message_by_id(details.messages, player_message_id)
        if source_message is None or source_message.role not in {"player", "system"}:
            raise ValueError(f"Unknown active source message id: {player_message_id}")
        presence_preference = _character_presence_model_preference(
            repositories=self.repositories,
            save_id=save_id,
        )
        if presence_preference is None:
            return CharacterActionPlanningResult(skipped_reason="no_model_preference")
        intent_preference = _character_intent_model_preference(
            repositories=self.repositories,
            save_id=save_id,
        )
        if intent_preference is None:
            return CharacterActionPlanningResult(skipped_reason="no_model_preference")
        presence_provider, presence_skip_reason = self._structured_provider(
            presence_preference
        )
        if presence_skip_reason is not None:
            return CharacterActionPlanningResult(skipped_reason=presence_skip_reason)
        intent_provider, intent_skip_reason = self._structured_provider(
            intent_preference
        )
        if intent_skip_reason is not None:
            return CharacterActionPlanningResult(skipped_reason=intent_skip_reason)
        if not any(
            not character.is_player_character
            for character in self.repositories.list_characters(save_id)
        ):
            return CharacterActionPlanningResult(skipped_reason="no_npc_characters")
        characters = _planning_characters_for_turn(
            repositories=self.repositories,
            save_id=save_id,
            source_message=source_message,
        )
        if not characters:
            return CharacterActionPlanningResult(
                skipped_reason="no_scoped_npc_characters",
            )
        valid_knowledge_targets = _valid_knowledge_edge_targets(
            repositories=self.repositories,
            save_id=save_id,
            scenario=details.scenario,
        )
        semaphore = asyncio.Semaphore(
            character_action_planning_max_concurrency(
                self.repositories,
                save_id=save_id,
            )
        )

        async def assess_presence(
            character: CharacterRecord,
        ) -> CharacterTurnAssessment | None:
            async with semaphore:
                try:
                    return await self._assess_character_presence(
                        save_id=save_id,
                        preference=presence_preference,
                        provider=cast(StructuredOutputProvider, presence_provider),
                        character=character,
                        source_message=source_message,
                        messages=tuple(details.messages),
                    )
                except Exception as exc:
                    log_error_event(
                        "chat.character_presence_assessment_failed",
                        save_id=save_id,
                        character_id=character.id,
                        character_name=character.name,
                        **exception_log_fields(exc),
                    )
                    return None

        raw_presence_decisions = await asyncio.gather(
            *(assess_presence(character) for character in characters)
        )
        presence_decisions = tuple(
            decision for decision in raw_presence_decisions if decision is not None
        )
        failed_ids = [
            character.id
            for character, decision in zip(
                characters,
                raw_presence_decisions,
                strict=True,
            )
            if decision is None
        ]
        characters_by_id = {character.id: character for character in characters}
        intent_candidates = tuple(
            decision
            for decision in presence_decisions
            if _assessment_has_grounded_presence(decision)
            and (
                decision.present
                or decision.enters_scene
                or decision.leaves_scene
            )
        )

        async def plan_intent(
            assessment: CharacterTurnAssessment,
        ) -> CharacterTurnAssessment | None:
            character = characters_by_id[assessment.character_id]
            async with semaphore:
                try:
                    return await self._plan_character_intent(
                        save_id=save_id,
                        preference=intent_preference,
                        provider=cast(StructuredOutputProvider, intent_provider),
                        character=character,
                        presence_assessment=assessment,
                        source_message=source_message,
                        messages=tuple(details.messages),
                        valid_knowledge_targets=valid_knowledge_targets,
                    )
                except Exception as exc:
                    log_error_event(
                        "chat.character_intent_planning_failed",
                        save_id=save_id,
                        character_id=character.id,
                        character_name=character.name,
                        **exception_log_fields(exc),
                    )
                    return None

        raw_intent_decisions = await asyncio.gather(
            *(plan_intent(assessment) for assessment in intent_candidates)
        )
        intent_by_character_id = {
            decision.character_id: decision
            for decision in raw_intent_decisions
            if decision is not None
        }
        failed_ids.extend(
            assessment.character_id
            for assessment, decision in zip(
                intent_candidates,
                raw_intent_decisions,
                strict=True,
            )
            if decision is None
        )
        decisions = tuple(
            intent_by_character_id.get(decision.character_id, decision)
            for decision in presence_decisions
        )
        applied = (
            _apply_presence_decisions(
                repositories=self.repositories,
                save_id=save_id,
                player_message=source_message,
                decisions=decisions,
            )
            if apply_presence_updates
            else False
        )
        plans = tuple(
            _plan_from_assessment(assessment)
            for assessment in decisions
            if _assessment_has_action_plan(assessment)
        )
        return CharacterActionPlanningResult(
            plans=plans,
            decisions=decisions,
            failed_character_ids=tuple(failed_ids),
            applied_presence_update=applied,
        )

    def _structured_provider(
        self,
        preference: ModelPreferenceRecord,
    ) -> tuple[StructuredOutputProvider | None, str | None]:
        provider = self.providers.get(preference.provider)
        if not isinstance(cast(object, provider), StructuredOutputProvider):
            return None, "provider_unavailable"
        check = check_model_capabilities(
            self.repositories,
            provider=preference.provider,
            model_id=preference.model_id,
            required=STRUCTURED_OUTPUT_CAPABILITIES,
        )
        if check.reason == MODEL_MISSING_REASON:
            return None, "model_missing"
        if check.reason == MODEL_UNAVAILABLE_REASON:
            return None, "model_unavailable"
        if check.reason == MODEL_LACKS_CAPABILITY_REASON:
            return None, "model_lacks_structured_output"
        return cast(StructuredOutputProvider, provider), None

    async def _assess_character_presence(
        self,
        *,
        save_id: str,
        preference: ModelPreferenceRecord,
        provider: StructuredOutputProvider,
        character: CharacterRecord,
        source_message: MessageRecord,
        messages: tuple[MessageRecord, ...],
    ) -> CharacterTurnAssessment:
        evidence_sources = _planning_evidence_sources(
            repositories=self.repositories,
            save_id=save_id,
            character=character,
            source_message=source_message,
            messages=messages,
        )
        request = StructuredOutputRequest(
            provider=preference.provider,
            model_id=preference.model_id,
            schema_name="character_presence_assessment",
            schema=_character_presence_schema(
                character,
                evidence_source_ids=tuple(evidence_sources),
            ),
            messages=_character_presence_messages(
                repositories=self.repositories,
                save_id=save_id,
                character=character,
                source_message=source_message,
                messages=messages,
                evidence_sources=evidence_sources,
            ),
            temperature=0.2,
            max_output_tokens=10_000,
        )
        response = await structured_output_with_fallback(
            repositories=self.repositories,
            providers=self.providers,
            request=request,
            task=CHARACTER_PRESENCE_ASSESSMENT_TASK,
            save_id=save_id,
            diagnostic_context={
                "character_id": character.id,
                "source_message_id": source_message.id,
            },
        )
        return _presence_assessment_from_data(
            response.data,
            character=character,
            evidence_sources=evidence_sources,
        )

    async def _plan_character_intent(
        self,
        *,
        save_id: str,
        preference: ModelPreferenceRecord,
        provider: StructuredOutputProvider,
        character: CharacterRecord,
        presence_assessment: CharacterTurnAssessment,
        source_message: MessageRecord,
        messages: tuple[MessageRecord, ...],
        valid_knowledge_targets: Mapping[str, frozenset[str]],
    ) -> CharacterTurnAssessment:
        evidence_sources = _planning_evidence_sources(
            repositories=self.repositories,
            save_id=save_id,
            character=character,
            source_message=source_message,
            messages=messages,
        )
        request = StructuredOutputRequest(
            provider=preference.provider,
            model_id=preference.model_id,
            schema_name="character_intent_plan",
            schema=_character_intent_schema(
                character,
                evidence_source_ids=tuple(evidence_sources),
            ),
            messages=_character_intent_messages(
                repositories=self.repositories,
                save_id=save_id,
                character=character,
                presence_assessment=presence_assessment,
                source_message=source_message,
                messages=messages,
                valid_knowledge_targets=valid_knowledge_targets,
                evidence_sources=evidence_sources,
            ),
            temperature=0.2,
            max_output_tokens=10_000,
        )
        response = await structured_output_with_fallback(
            repositories=self.repositories,
            providers=self.providers,
            request=request,
            task=CHARACTER_INTENT_PLANNING_TASK,
            save_id=save_id,
            diagnostic_context={
                "character_id": character.id,
                "source_message_id": source_message.id,
            },
        )
        return _intent_assessment_from_data(
            response.data,
            presence_assessment=presence_assessment,
            character=character,
            valid_knowledge_targets=valid_knowledge_targets,
            evidence_sources=evidence_sources,
        )


def character_action_planning_enabled(
    repositories: PersistenceRepositories,
    *,
    save_id: str | None = None,
) -> bool:
    value = repositories.get_effective_setting(
        CHARACTER_ACTION_PLANNING_ENABLED_SETTING,
        save_id=save_id,
    )
    return (
        bool(value)
        if value is not None
        else CHARACTER_ACTION_PLANNING_ENABLED_DEFAULT
    )


def character_action_planning_max_concurrency(
    repositories: PersistenceRepositories,
    *,
    save_id: str | None = None,
) -> int:
    value = repositories.get_effective_setting(
        CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY_SETTING,
        save_id=save_id,
    )
    return sanitize_character_action_planning_max_concurrency(value)


def sanitize_character_action_planning_max_concurrency(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        return DEFAULT_CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY
    return min(
        max(value, MIN_CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY),
        MAX_CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY,
    )


def format_character_action_plan(plan: CharacterActionPlan) -> str:
    parts = [
        f"[character_action:{plan.character_id}] {plan.character_name}",
        f"intent: {plan.intent}" if plan.intent else "",
        f"next action: {plan.action}",
        f"reason: {plan.reason}" if plan.reason else "",
        f"confidence: {round(plan.confidence * 100)}%",
        (
            "evidence: " + ", ".join(plan.evidence_source_ids)
            if plan.evidence_source_ids
            else ""
        ),
    ]
    return " | ".join(part for part in parts if part)


def character_turn_assessment_has_prompt_guidance(
    assessment: CharacterTurnAssessment,
) -> bool:
    return bool(
        _assessment_has_action_plan(assessment)
        or assessment.learned_memory_candidates
        or assessment.knowledge_edge_candidates
        or assessment.needs_review_notes
    )


def format_character_turn_assessment(assessment: CharacterTurnAssessment) -> str:
    if not _assessment_has_richer_guidance(assessment):
        return format_character_action_plan(_plan_from_assessment(assessment))
    has_action_plan = _assessment_has_action_plan(assessment)
    parts = [
        f"[character_action:{assessment.character_id}] {assessment.character_name}",
        f"present: {'yes' if assessment.present else 'no'}"
        if has_action_plan
        else "",
        "enters scene: yes"
        if has_action_plan and assessment.enters_scene
        else "",
        "leaves scene after beat: yes"
        if has_action_plan and assessment.leaves_scene
        else "",
        f"intent: {assessment.intent}" if has_action_plan and assessment.intent else "",
        f"next action: {assessment.action}"
        if has_action_plan and assessment.action
        else "",
        f"reason: {assessment.reason}" if has_action_plan and assessment.reason else "",
        f"confidence: {round(assessment.confidence * 100)}%"
        if has_action_plan
        else "",
        (
            "evidence: " + ", ".join(assessment.evidence_source_ids)
            if has_action_plan and assessment.evidence_source_ids
            else ""
        ),
    ]
    parts.extend(
        _format_learned_memory_candidate(candidate)
        for candidate in assessment.learned_memory_candidates
    )
    parts.extend(
        _format_knowledge_edge_candidate(candidate)
        for candidate in assessment.knowledge_edge_candidates
    )
    if assessment.needs_review_notes:
        parts.append("needs review: " + "; ".join(assessment.needs_review_notes))
    return " | ".join(part for part in parts if part)


def _assessment_has_action_plan(assessment: CharacterTurnAssessment) -> bool:
    return bool(
        assessment.action
        and _assessment_has_grounded_presence(assessment)
        and assessment.evidence_source_ids
        and assessment.evidence_quote.strip()
        and (assessment.present or assessment.enters_scene or assessment.leaves_scene)
    )


def _assessment_has_grounded_presence(assessment: CharacterTurnAssessment) -> bool:
    return bool(
        assessment.presence_evidence_source_ids
        and assessment.presence_evidence_quote.strip()
    )


def _assessment_has_richer_guidance(assessment: CharacterTurnAssessment) -> bool:
    return bool(
        not _assessment_has_action_plan(assessment)
        or assessment.enters_scene
        or assessment.leaves_scene
        or assessment.learned_memory_candidates
        or assessment.knowledge_edge_candidates
        or assessment.needs_review_notes
    )


def _format_learned_memory_candidate(
    candidate: CharacterLearnedMemoryCandidate,
) -> str:
    parts = [
        "learned memory candidate (do not persist automatically): " + candidate.body,
        "tags: " + ", ".join(candidate.tags) if candidate.tags else "",
        f"knowledge: {candidate.knowledge_state}",
        (
            f"acquired: {candidate.acquisition_method}"
            if candidate.acquisition_method != "unknown"
            else ""
        ),
        f"reason: {candidate.reason}" if candidate.reason else "",
        f"confidence: {round(candidate.confidence * 100)}%",
        (
            "evidence: " + ", ".join(candidate.evidence_source_ids)
            if candidate.evidence_source_ids
            else ""
        ),
        f"quote: {candidate.evidence_quote}" if candidate.evidence_quote else "",
    ]
    return "; ".join(part for part in parts if part)


def _format_knowledge_edge_candidate(
    candidate: CharacterKnowledgeEdgeCandidate,
) -> str:
    parts = [
        "knowledge edge candidate (do not persist automatically): "
        f"target: {candidate.target_type}:{candidate.target_id}",
        f"knowledge: {candidate.knowledge_state}",
        (
            f"acquired: {candidate.acquisition_method}"
            if candidate.acquisition_method != "unknown"
            else ""
        ),
        f"reason: {candidate.reason}" if candidate.reason else "",
        f"confidence: {round(candidate.confidence * 100)}%",
        (
            "evidence: " + ", ".join(candidate.evidence_source_ids)
            if candidate.evidence_source_ids
            else ""
        ),
        f"quote: {candidate.evidence_quote}" if candidate.evidence_quote else "",
    ]
    return "; ".join(part for part in parts if part)


def _character_presence_model_preference(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
) -> ModelPreferenceRecord | None:
    return roleplay_model_preference_with_fallbacks(
        repositories=repositories,
        save_id=save_id,
        purposes=(
            CHARACTER_PRESENCE_ASSESSMENT_TASK,
            CHARACTER_ACTION_PLANNING_TASK,
        ),
    )


def _character_intent_model_preference(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
) -> ModelPreferenceRecord | None:
    return roleplay_model_preference_with_fallbacks(
        repositories=repositories,
        save_id=save_id,
        purposes=(
            CHARACTER_INTENT_PLANNING_TASK,
            CHARACTER_ACTION_PLANNING_TASK,
        ),
    )


def _evidence_source_ids_schema(evidence_source_ids: tuple[str, ...]) -> dict[str, Any]:
    items: dict[str, object] = {"type": "string"}
    if evidence_source_ids:
        items["enum"] = sorted(evidence_source_ids)
    return {"type": "array", "items": items, "maxItems": 8}


def _character_presence_schema(
    character: CharacterRecord,
    *,
    evidence_source_ids: tuple[str, ...],
) -> dict[str, Any]:
    string_array = _evidence_source_ids_schema(evidence_source_ids)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "character_id": {"type": "string", "enum": [character.id]},
            "present": {"type": "boolean"},
            "enters_scene": {"type": "boolean"},
            "leaves_scene": {"type": "boolean"},
            "reason": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_source_ids": string_array,
            "evidence_quote": {"type": "string"},
        },
        "required": [
            "character_id",
            "present",
            "enters_scene",
            "leaves_scene",
            "reason",
            "confidence",
            "evidence_source_ids",
            "evidence_quote",
        ],
    }


def _character_intent_schema(
    character: CharacterRecord,
    *,
    evidence_source_ids: tuple[str, ...],
) -> dict[str, Any]:
    string_array = _evidence_source_ids_schema(evidence_source_ids)
    tags_array = {"type": "array", "items": {"type": "string"}, "maxItems": 8}
    knowledge_state = {
        "type": "string",
        "enum": sorted(CHARACTER_KNOWLEDGE_STATES),
    }
    acquisition_method = {
        "type": "string",
        "enum": sorted(CHARACTER_KNOWLEDGE_ACQUISITION_METHODS),
    }
    learned_memory_candidate = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "body": {"type": "string"},
            "tags": tags_array,
            "knowledge_state": knowledge_state,
            "acquisition_method": acquisition_method,
            "reason": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_source_ids": string_array,
            "evidence_quote": {"type": "string"},
        },
        "required": [
            "body",
            "tags",
            "knowledge_state",
            "acquisition_method",
            "reason",
            "confidence",
            "evidence_source_ids",
            "evidence_quote",
        ],
    }
    knowledge_edge_candidate = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "target_type": {
                "type": "string",
                "enum": ["memory", "world_state", "summary", "scenario_section"],
            },
            "target_id": {"type": "string"},
            "knowledge_state": knowledge_state,
            "acquisition_method": acquisition_method,
            "reason": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_source_ids": string_array,
            "evidence_quote": {"type": "string"},
        },
        "required": [
            "target_type",
            "target_id",
            "knowledge_state",
            "acquisition_method",
            "reason",
            "confidence",
            "evidence_source_ids",
            "evidence_quote",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "character_id": {"type": "string", "enum": [character.id]},
            "action": {"type": "string"},
            "intent": {"type": "string"},
            "reason": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_source_ids": string_array,
            "evidence_quote": {"type": "string"},
            "learned_memory_candidates": {
                "type": "array",
                "items": learned_memory_candidate,
                "maxItems": 6,
            },
            "knowledge_edge_candidates": {
                "type": "array",
                "items": knowledge_edge_candidate,
                "maxItems": 8,
            },
            "needs_review_notes": string_array,
        },
        "required": [
            "character_id",
            "action",
            "intent",
            "reason",
            "confidence",
            "evidence_source_ids",
            "evidence_quote",
            "learned_memory_candidates",
            "knowledge_edge_candidates",
            "needs_review_notes",
        ],
    }


def _character_presence_messages(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    character: CharacterRecord,
    source_message: MessageRecord,
    messages: tuple[MessageRecord, ...],
    evidence_sources: Mapping[str, str],
) -> tuple[ChatMessage, ...]:
    details = repositories.load_save_details(save_id)
    scenario = details.scenario if details is not None else None
    storyteller_mode = (
        details is not None
        and details.save.interaction_mode is InteractionMode.STORYTELLER
    )
    snapshot = repositories.get_scene_snapshot(save_id)
    recent_messages = _visible_recent_messages_for_character(
        repositories=repositories,
        save_id=save_id,
        character=character,
        source_message=source_message,
        messages=messages,
    )
    return (
        ChatMessage(
            role="system",
            body=(
                (
                    "Assess one narrator-controlled character's scene presence "
                    if storyteller_mode
                    else "Assess one non-player character's scene presence "
                )
                + (
                    "for the next Bragi narrator turn. Decide only whether this "
                    "character is present, entering, or leaving the current scene. "
                    "Use the enforced structured schema only. Do not plan their "
                    "next action and never plan or control the player character. "
                    "Use only evidence_source_ids listed in Evidence sources, and "
                    "copy evidence_quote exactly from one cited evidence source."
                )
            ),
        ),
        ChatMessage(
            role="user",
            body="\n".join(
                part
                for part in (
                    _scenario_text(scenario),
                    _scene_text(snapshot),
                    (
                        ""
                        if storyteller_mode
                        else _player_character_text(repositories, save_id=save_id)
                    ),
                    _active_threads_text(repositories.list_active_threads(save_id)),
                    _character_text(character),
                    (
                        ""
                        if storyteller_mode
                        else _dating_route_text(repositories, save_id, character)
                    ),
                    _evidence_sources_text(evidence_sources),
                    "Recent chronicle:",
                    *(
                        _planning_message_line(
                            message,
                            storyteller_mode=storyteller_mode,
                        )
                        for message in recent_messages
                    ),
                    "",
                    (
                        "Latest non-diegetic story direction (guidance only; "
                        "not canonical evidence):"
                        if storyteller_mode
                        else "Latest turn source message:"
                    ),
                    source_message.body,
                )
                if part != ""
            ),
        ),
    )


def _character_intent_messages(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    character: CharacterRecord,
    presence_assessment: CharacterTurnAssessment,
    source_message: MessageRecord,
    messages: tuple[MessageRecord, ...],
    valid_knowledge_targets: Mapping[str, frozenset[str]],
    evidence_sources: Mapping[str, str],
) -> tuple[ChatMessage, ...]:
    details = repositories.load_save_details(save_id)
    scenario = details.scenario if details is not None else None
    storyteller_mode = (
        details is not None
        and details.save.interaction_mode is InteractionMode.STORYTELLER
    )
    snapshot = repositories.get_scene_snapshot(save_id)
    recent_messages = _visible_recent_messages_for_character(
        repositories=repositories,
        save_id=save_id,
        character=character,
        source_message=source_message,
        messages=messages,
    )
    return (
        ChatMessage(
            role="system",
            body=(
                (
                    "Plan one narrator-controlled character's next visible intent and "
                    if storyteller_mode
                    else "Plan one non-player character's next visible intent and "
                )
                + (
                "action for the next Bragi narrator turn. Use the enforced "
                "structured schema only. The character was already assessed "
                "as present, entering, or leaving, so choose only the action "
                "this character would plausibly take. Favor visible initiative "
                "over waiting for the player: choose an action that changes the "
                "situation, applies pressure, sets a boundary, or can "
                "interrupt, demand, refuse, leave, escalate, advance a clock, "
                "or otherwise advance the NPC's agenda when plausible. Preserve "
                "the full spectrum of NPC stances: a trusting character may "
                "cooperate, while a hostile, self-interested, unfair, or "
                "unreasonable character should act from that profile when "
                "evidence supports it. Restraint "
                "is valid only when supported by evidence, boundaries, or route "
                "pacing. Learned memories and knowledge edges are candidates only; "
                "never treat them as "
                "committed canon. Keep source or evidence ids on every "
                "proposed memory or knowledge candidate. Never plan or "
                "control the player character. Use only evidence_source_ids "
                "listed in Evidence sources, copy top-level evidence_quote "
                "exactly from one cited source, and copy evidence_quote exactly "
                "from one cited evidence source for every learned memory or "
                "knowledge edge candidate."
                )
            ),
        ),
        ChatMessage(
            role="user",
            body="\n".join(
                part
                for part in (
                    _scenario_text(scenario),
                    _scene_text(snapshot),
                    (
                        ""
                        if storyteller_mode
                        else _player_character_text(repositories, save_id=save_id)
                    ),
                    _active_threads_text(repositories.list_active_threads(save_id)),
                    _character_text(character),
                    _presence_assessment_text(presence_assessment),
                    (
                        ""
                        if storyteller_mode
                        else _dating_route_text(repositories, save_id, character)
                    ),
                    _linkable_knowledge_targets_text(valid_knowledge_targets),
                    _evidence_sources_text(evidence_sources),
                    "Recent chronicle:",
                    *(
                        _planning_message_line(
                            message,
                            storyteller_mode=storyteller_mode,
                        )
                        for message in recent_messages
                    ),
                    "",
                    (
                        "Latest non-diegetic story direction (guidance only; "
                        "not canonical evidence):"
                        if storyteller_mode
                        else "Latest turn source message:"
                    ),
                    source_message.body,
                )
                if part != ""
            ),
        ),
    )


def _active_threads_text(threads: list[ActiveThreadRecord]) -> str:
    visible = [
        thread
        for thread in threads
        if active_thread_is_prompt_visible(thread)
        and "director_pressure" in thread.related_entities
    ]
    if not visible:
        return ""
    return "Active threads: " + "; ".join(
        f"{thread.title} "
        f"({normalize_active_thread_status(thread.status)}, "
        f"priority {thread.priority}): {thread.description}"
        for thread in visible
    )


def _dating_route_text(
    repositories: PersistenceRepositories,
    save_id: str,
    character: CharacterRecord,
) -> str:
    route = _dating_route_for_character(repositories, save_id, character.id)
    if route is None:
        return ""
    snapshot = repositories.get_scene_snapshot(save_id)
    known_days = _known_world_day_count(
        first_day=route.first_met_world_day_index,
        current_day=snapshot.world_day_index if snapshot is not None else None,
    )
    policy = escalation_policy_for_stage(route.stage)
    parts = [
        f"stage {route.stage.replace('_', ' ')}",
        f"completed interactions {route.completed_interactions}",
        f"dates completed {route.dates_completed}",
    ]
    if known_days is not None:
        parts.append(f"known for {known_days} in-world days")
    if route.pacing_preference:
        parts.append(f"pacing {route.pacing_preference}")
    if route.next_reasonable_step:
        parts.append(f"next plausible step {route.next_reasonable_step}")
    if route.known_boundaries:
        parts.append("known boundaries " + "; ".join(route.known_boundaries))
    if route.unresolved_questions:
        parts.append("unresolved questions " + "; ".join(route.unresolved_questions))
    parts.append(f"max plausible escalation {policy.max_plausible_escalation}")
    parts.append(
        "intimacy profile "
        + intimacy_profile_guidance(
            comfort_with_intimacy=route.comfort_with_intimacy,
            pacing_preference=route.pacing_preference,
            known_boundaries=route.known_boundaries,
        )
    )
    if policy.allowed_progress:
        parts.append("allowed now " + "; ".join(policy.allowed_progress))
    if policy.needs_explicit_support:
        parts.append(
            "needs explicit support " + "; ".join(policy.needs_explicit_support)
        )
    if policy.premature_escalations:
        parts.append("premature now " + "; ".join(policy.premature_escalations))
    return (
        "Dating route pacing for this character is deterministic state: "
        + "; ".join(parts)
        + ". Keep the next action proportionate to this route state."
    )


def _dating_route_for_character(
    repositories: PersistenceRepositories,
    save_id: str,
    character_id: str,
) -> DatingRouteStateRecord | None:
    for route in repositories.list_dating_route_states(save_id):
        if route.npc_character_id == character_id:
            return route
    return None


def _known_world_day_count(
    *,
    first_day: int | None,
    current_day: int | None,
) -> int | None:
    if first_day is None or current_day is None:
        return None
    return max(0, current_day - first_day)


def _valid_knowledge_edge_targets(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    scenario: object | None,
) -> dict[str, frozenset[str]]:
    scenario_targets: set[str] = set()
    for source_id, section_id, _text in scenario_section_candidates(
        cast(Any, scenario)
    ):
        scenario_targets.add(source_id)
        scenario_targets.add(section_id)
    return {
        "memory": frozenset(
            memory.id for memory in repositories.list_memories(save_id)
        ),
        "world_state": frozenset(
            state.id for state in repositories.list_world_state(save_id)
        ),
        "summary": frozenset(
            summary.id for summary in repositories.list_summaries(save_id)
        ),
        "scenario_section": frozenset(scenario_targets),
    }


def _planning_characters_for_turn(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    source_message: MessageRecord,
) -> tuple[CharacterRecord, ...]:
    snapshot = repositories.get_scene_snapshot(save_id)
    present_ids = set(snapshot.present_character_ids if snapshot else ())
    source_text = source_message.body
    all_characters = tuple(repositories.list_characters(save_id))
    characters = tuple(
        character
        for character in all_characters
        if not character.is_player_character
    )
    save = repositories.get_save(save_id)
    if (
        save is not None
        and save.interaction_mode is InteractionMode.STORYTELLER
        and not any(character.is_player_character for character in all_characters)
    ):
        return characters
    present = tuple(
        character for character in characters if character.id in present_ids
    )
    referenced = tuple(
        character
        for character in characters
        if character.id not in present_ids
        and character_name_is_mentioned(
            name=character.name,
            aliases=character.aliases,
            text=source_text,
        )
    )
    return present + referenced


def _linkable_knowledge_targets_text(
    valid_targets: Mapping[str, frozenset[str]],
) -> str:
    lines: list[str] = []
    for target_type in ("memory", "world_state", "summary", "scenario_section"):
        target_ids = sorted(valid_targets.get(target_type, frozenset()))
        if not target_ids:
            continue
        shown = target_ids[:25]
        suffix = (
            f"; {len(target_ids) - len(shown)} more"
            if len(target_ids) > 25
            else ""
        )
        lines.append(f"- {target_type}: {', '.join(shown)}{suffix}")
    if not lines:
        return ""
    return "\n".join(
        (
            "Existing linkable knowledge target IDs for candidate edges only. "
            "Use exact target_type and target_id values; do not invent target IDs.",
            *lines,
        )
    )


def _planning_evidence_sources(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    character: CharacterRecord,
    source_message: MessageRecord,
    messages: tuple[MessageRecord, ...],
) -> dict[str, str]:
    sources: dict[str, str] = {}

    def add(source_id: str, text: str) -> None:
        normalized = " ".join(text.split())
        if source_id and normalized:
            sources.setdefault(source_id, normalized)

    snapshot = repositories.get_scene_snapshot(save_id)
    if snapshot is not None:
        add(f"scene_snapshot:{snapshot.id}", _scene_text(snapshot))
    add(f"character:{character.id}", _character_text(character))
    for thread in repositories.list_active_threads(save_id):
        if (
            active_thread_is_prompt_visible(thread)
            and "director_pressure" in thread.related_entities
        ):
            add(
                f"active_thread:{thread.id}",
                " ".join(
                    part
                    for part in (
                        thread.title,
                        normalize_active_thread_status(thread.status),
                        thread.description,
                    )
                    if part
                ),
            )

    recent_messages = _visible_recent_messages_for_character(
        repositories=repositories,
        save_id=save_id,
        character=character,
        source_message=source_message,
        messages=messages,
    )
    save = repositories.get_save(save_id)
    storyteller_mode = (
        save is not None
        and save.interaction_mode is InteractionMode.STORYTELLER
    )
    for message in (*recent_messages, source_message):
        if storyteller_mode and message.role != "narrator":
            continue
        add(f"message:{message.id}", message.body)
    return sources


def _evidence_sources_text(evidence_sources: Mapping[str, str]) -> str:
    if not evidence_sources:
        return ""
    return "\n".join(
        (
            "Evidence sources:",
            *(
                f"- {source_id}: {source_text}"
                for source_id, source_text in evidence_sources.items()
            ),
        )
    )


def _presence_assessment_text(assessment: CharacterTurnAssessment) -> str:
    parts = [
        f"present: {'yes' if assessment.present else 'no'}",
        f"enters scene: {'yes' if assessment.enters_scene else 'no'}",
        f"leaves scene: {'yes' if assessment.leaves_scene else 'no'}",
        f"reason: {assessment.reason}" if assessment.reason else "",
        f"confidence: {round(assessment.confidence * 100)}%",
        (
            "evidence: " + ", ".join(assessment.evidence_source_ids)
            if assessment.evidence_source_ids
            else ""
        ),
        f"quote: {assessment.evidence_quote}" if assessment.evidence_quote else "",
    ]
    return "Presence assessment: " + " | ".join(part for part in parts if part)


def _presence_assessment_from_data(
    data: Mapping[str, object],
    *,
    character: CharacterRecord,
    evidence_sources: Mapping[str, str],
) -> CharacterTurnAssessment:
    character_id = _string(data.get("character_id"))
    if character_id != character.id:
        raise ValueError(
            "Character presence assessment returned the wrong character_id"
        )
    present = data.get("present")
    if not isinstance(present, bool):
        raise ValueError("Character presence assessment present must be a boolean")
    enters_scene = _bool(data.get("enters_scene"))
    leaves_scene = _bool(data.get("leaves_scene"))
    evidence_source_ids, evidence_quote = _grounded_assessment_evidence(
        data,
        evidence_sources=evidence_sources,
    )
    return CharacterTurnAssessment(
        character_id=character.id,
        character_name=character.name,
        present=present,
        reason=_string(data.get("reason")),
        confidence=_float(data.get("confidence")),
        evidence_source_ids=evidence_source_ids,
        evidence_quote=evidence_quote,
        presence_evidence_source_ids=evidence_source_ids,
        presence_evidence_quote=evidence_quote,
        enters_scene=enters_scene,
        leaves_scene=leaves_scene,
    )


def _intent_assessment_from_data(
    data: Mapping[str, object],
    *,
    presence_assessment: CharacterTurnAssessment,
    character: CharacterRecord,
    valid_knowledge_targets: Mapping[str, frozenset[str]],
    evidence_sources: Mapping[str, str],
) -> CharacterTurnAssessment:
    character_id = _string(data.get("character_id"))
    if character_id != character.id:
        raise ValueError("Character intent plan returned the wrong character_id")
    action = _string(data.get("action"))
    if not action:
        raise ValueError("Active character intent plan must include an action")
    evidence_source_ids, evidence_quote = _grounded_assessment_evidence(
        data,
        evidence_sources=evidence_sources,
    )
    has_grounded_intent = bool(evidence_source_ids)
    learned_memory_candidates = _learned_memory_candidates_from_data(
        data.get("learned_memory_candidates"),
        evidence_sources=evidence_sources,
    )
    knowledge_edge_candidates = _knowledge_edge_candidates_from_data(
        data.get("knowledge_edge_candidates"),
        valid_targets=valid_knowledge_targets,
        evidence_sources=evidence_sources,
    )
    if has_grounded_intent:
        reason = _string(data.get("reason")) or presence_assessment.reason
        confidence = (
            _float(data.get("confidence"))
            if isinstance(data.get("confidence"), (int, float))
            else presence_assessment.confidence
        )
        intent = _string(data.get("intent"))
        needs_review_notes = _string_tuple(data.get("needs_review_notes"))
    else:
        action = ""
        intent = ""
        reason = presence_assessment.reason
        confidence = presence_assessment.confidence
        evidence_source_ids = presence_assessment.evidence_source_ids
        evidence_quote = presence_assessment.evidence_quote
        needs_review_notes = (
            _string_tuple(data.get("needs_review_notes"))
            if learned_memory_candidates or knowledge_edge_candidates
            else ()
        )
    return replace(
        presence_assessment,
        action=action,
        intent=intent,
        reason=reason,
        confidence=confidence,
        evidence_source_ids=evidence_source_ids,
        evidence_quote=evidence_quote,
        learned_memory_candidates=learned_memory_candidates,
        knowledge_edge_candidates=knowledge_edge_candidates,
        needs_review_notes=needs_review_notes,
    )


def _apply_presence_decisions(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    player_message: MessageRecord,
    decisions: tuple[CharacterTurnAssessment, ...],
) -> bool:
    if not decisions:
        return False
    snapshot = repositories.get_scene_snapshot(save_id)
    present_ids = set(snapshot.present_character_ids if snapshot else [])
    changed = False
    for decision in decisions:
        if not _assessment_has_grounded_presence(decision):
            continue
        before_ids = set(present_ids)
        if decision.leaves_scene:
            present_ids.discard(decision.character_id)
        elif decision.present or decision.enters_scene:
            present_ids.add(decision.character_id)
        else:
            present_ids.discard(decision.character_id)
        changed = changed or present_ids != before_ids
    if not changed:
        return False
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        current_location_id=snapshot.current_location_id if snapshot else None,
        situation=snapshot.situation if snapshot else "",
        objective=snapshot.objective if snapshot else "",
        in_world_time=snapshot.in_world_time if snapshot else "",
        time_of_day=snapshot.time_of_day if snapshot else "",
        day_of_week=snapshot.day_of_week if snapshot else "",
        world_day_index=snapshot.world_day_index if snapshot else None,
        weather=snapshot.weather if snapshot else "",
        mood=snapshot.mood if snapshot else "",
        nearby_objects=snapshot.nearby_objects if snapshot else [],
        hazards=snapshot.hazards if snapshot else [],
        present_character_ids=sorted(present_ids),
        source_message_id=player_message.id,
        locked_fields=snapshot.locked_fields if snapshot else [],
        snapshot_id=snapshot.id if snapshot else None,
        first_seen_message_id=snapshot.first_seen_message_id if snapshot else None,
        last_updated_message_id=player_message.id,
    )
    return True


def _plan_from_assessment(assessment: CharacterTurnAssessment) -> CharacterActionPlan:
    return CharacterActionPlan(
        character_id=assessment.character_id,
        character_name=assessment.character_name,
        action=assessment.action,
        intent=assessment.intent,
        reason=assessment.reason,
        confidence=assessment.confidence,
        evidence_source_ids=assessment.evidence_source_ids,
    )


def _allowed_evidence_source_ids(
    value: object,
    *,
    evidence_sources: Mapping[str, str],
    fallback_evidence_source_ids: tuple[str, ...] = (),
) -> tuple[str, ...]:
    raw_ids = _string_tuple(value) or fallback_evidence_source_ids
    allowed = set(evidence_sources)
    return tuple(
        dict.fromkeys(source_id for source_id in raw_ids if source_id in allowed)
    )


def _grounded_assessment_evidence(
    data: Mapping[str, object],
    *,
    evidence_sources: Mapping[str, str],
) -> tuple[tuple[str, ...], str]:
    evidence_source_ids = _allowed_evidence_source_ids(
        data.get("evidence_source_ids"),
        evidence_sources=evidence_sources,
    )
    quote = _string(data.get("evidence_quote"))
    if not evidence_source_ids or not quote:
        return (), ""
    if not any(
        quote_matches_source(quote, evidence_sources[source_id])
        for source_id in evidence_source_ids
    ):
        return (), ""
    return evidence_source_ids, quote


def _grounded_candidate_evidence_source_ids(
    item: Mapping[str, object],
    *,
    evidence_sources: Mapping[str, str],
) -> tuple[str, ...]:
    evidence_source_ids = _allowed_evidence_source_ids(
        item.get("evidence_source_ids"),
        evidence_sources=evidence_sources,
    )
    quote = _string(item.get("evidence_quote"))
    if not evidence_source_ids or not quote:
        return ()
    if not any(
        quote_matches_source(quote, evidence_sources[source_id])
        for source_id in evidence_source_ids
    ):
        return ()
    return evidence_source_ids


def _learned_memory_candidates_from_data(
    value: object,
    *,
    evidence_sources: Mapping[str, str],
) -> tuple[CharacterLearnedMemoryCandidate, ...]:
    if not isinstance(value, list):
        return ()
    candidates: list[CharacterLearnedMemoryCandidate] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        body = _string(item.get("body"))
        if not body:
            continue
        evidence_source_ids = _grounded_candidate_evidence_source_ids(
            item,
            evidence_sources=evidence_sources,
        )
        if not evidence_source_ids:
            continue
        candidates.append(
            CharacterLearnedMemoryCandidate(
                body=body,
                tags=_string_tuple(item.get("tags")),
                knowledge_state=_knowledge_state(item.get("knowledge_state")),
                acquisition_method=_acquisition_method(
                    item.get("acquisition_method")
                ),
                reason=_string(item.get("reason")),
                confidence=_float(item.get("confidence")),
                evidence_source_ids=evidence_source_ids,
                evidence_quote=_string(item.get("evidence_quote")),
            )
        )
    return tuple(candidates)


def _knowledge_edge_candidates_from_data(
    value: object,
    *,
    valid_targets: Mapping[str, frozenset[str]],
    evidence_sources: Mapping[str, str],
) -> tuple[CharacterKnowledgeEdgeCandidate, ...]:
    if not isinstance(value, list):
        return ()
    candidates: list[CharacterKnowledgeEdgeCandidate] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        target_type = _normalized_knowledge_target_type(item.get("target_type"))
        target_id = _string(item.get("target_id"))
        if (
            not target_type
            or not target_id
            or target_id not in valid_targets.get(target_type, frozenset())
        ):
            continue
        evidence_source_ids = _grounded_candidate_evidence_source_ids(
            item,
            evidence_sources=evidence_sources,
        )
        if not evidence_source_ids:
            continue
        candidates.append(
            CharacterKnowledgeEdgeCandidate(
                target_type=target_type,
                target_id=target_id,
                knowledge_state=_knowledge_state(item.get("knowledge_state")),
                acquisition_method=_acquisition_method(
                    item.get("acquisition_method")
                ),
                reason=_string(item.get("reason")),
                confidence=_float(item.get("confidence")),
                evidence_source_ids=evidence_source_ids,
                evidence_quote=_string(item.get("evidence_quote")),
            )
        )
    return tuple(candidates)


def _message_by_id(
    messages: tuple[MessageRecord, ...] | list[MessageRecord],
    message_id: str,
) -> MessageRecord | None:
    return next((message for message in messages if message.id == message_id), None)


def _scenario_text(scenario: object | None) -> str:
    if scenario is None:
        return "Scenario: unavailable"
    title = getattr(scenario, "title", "")
    premise = getattr(scenario, "premise", "")
    player_role = getattr(scenario, "player_role", "")
    return "\n".join(
        part
        for part in (
            "Scenario:",
            f"Title: {title}" if title else "",
            f"Premise: {premise}" if premise else "",
            f"Player role: {player_role}" if player_role else "",
        )
        if part
    )


def _scene_text(snapshot: object | None) -> str:
    if snapshot is None:
        return "Current scene: no scene snapshot"
    parts = [
        f"situation: {getattr(snapshot, 'situation', '')}",
        f"objective: {getattr(snapshot, 'objective', '')}",
        f"time: {format_world_time_from_snapshot(snapshot)}",
        f"weather: {getattr(snapshot, 'weather', '')}",
        f"mood: {getattr(snapshot, 'mood', '')}",
        "nearby objects: " + ", ".join(getattr(snapshot, "nearby_objects", []) or []),
        "hazards: " + ", ".join(getattr(snapshot, "hazards", []) or []),
        "present character ids: "
        + ", ".join(getattr(snapshot, "present_character_ids", []) or []),
    ]
    return "Current scene:\n" + "\n".join(
        part for part in parts if not part.endswith(": ")
    )


def _player_character_text(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
) -> str:
    player = _player_character(repositories, save_id=save_id)
    if player is None:
        return "Player character: unknown"
    return f"Player character: {player.name}"


def _visible_recent_messages_for_character(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    character: CharacterRecord,
    source_message: MessageRecord,
    messages: tuple[MessageRecord, ...],
) -> tuple[MessageRecord, ...]:
    target_ids = {character.id}
    player = _player_character(repositories, save_id=save_id)
    if player is not None:
        target_ids.add(player.id)
    present_character_ids = frozenset(target_ids)
    visibility = repositories.list_message_visibility(
        save_id,
        character_ids=present_character_ids,
    )
    visible_messages = [
        message
        for message in messages
        if message.id != source_message.id
        if message_visible_to_present_characters(
            message_id=message.id,
            present_character_ids=present_character_ids,
            message_visibility=visibility,
        )
    ]
    return tuple(visible_messages[-CHARACTER_ACTION_PLANNING_RECENT_MESSAGE_LIMIT:])


def _player_character(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
) -> CharacterRecord | None:
    return next(
        (
            character
            for character in repositories.list_characters(save_id)
            if character.is_player_character
        ),
        None,
    )


def _character_text(character: CharacterRecord) -> str:
    details = [
        f"Character: {character.name}",
        f"character_id: {character.id}",
        f"aliases: {', '.join(character.aliases)}" if character.aliases else "",
        f"role: {character.role}" if character.role else "",
        f"age: {character.age}" if character.age else "",
        f"known state: {character.known_state}" if character.known_state else "",
        f"status: {character.status}" if character.status else "",
        f"personality: {character.personality}" if character.personality else "",
        f"voice: {character.voice}" if character.voice else "",
        f"goals: {character.goals}" if character.goals else "",
        f"motivations: {character.motivations}" if character.motivations else "",
        (
            f"current intent: {character.current_intent}"
            if character.current_intent
            else ""
        ),
        f"boundaries: {character.boundaries}" if character.boundaries else "",
        (
            f"attitude toward player: {character.attitude_toward_player}"
            if character.attitude_toward_player
            else ""
        ),
        (
            f"cooperation conditions: {character.cooperation_conditions}"
            if character.cooperation_conditions
            else ""
        ),
        _relationships_text(character.relationships),
        (
            "narrator-only private notes for this character; do not treat as "
            f"known by other characters: {character.private_notes}"
            if character.private_notes
            else ""
        ),
    ]
    return "\n".join(part for part in details if part)


def _relationships_text(value: Mapping[str, object]) -> str:
    if not value:
        return ""
    return "relationships: " + json.dumps(value, sort_keys=True)


def _message_line(message: MessageRecord) -> str:
    speaker = message.speaker_name or message.role
    return f"- {speaker}: {message.body}"


def _planning_message_line(
    message: MessageRecord,
    *,
    storyteller_mode: bool,
) -> str:
    if storyteller_mode and message.role == "player":
        return f"- Non-diegetic story direction (not canonical): {message.body}"
    return _message_line(message)


def _string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _bool(value: object) -> bool:
    return bool(value) if isinstance(value, bool) else False


def _knowledge_state(value: object) -> str:
    text = _string(value)
    return text if text in CHARACTER_KNOWLEDGE_STATES else "knows"


def _acquisition_method(value: object) -> str:
    text = _string(value)
    return text if text in CHARACTER_KNOWLEDGE_ACQUISITION_METHODS else "unknown"


def _normalized_knowledge_target_type(value: object) -> str:
    text = _string(value)
    if text in {"memory", "memories"}:
        return "memory"
    if text in {"world_state", "state"}:
        return "world_state"
    if text in {"summary", "summaries"}:
        return "summary"
    if text in {"scenario", "scenario_section"}:
        return "scenario_section"
    return ""


def _float(value: object) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return 0.0
