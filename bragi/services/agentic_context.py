"""Agentic observation, curation, planning, and verification services."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Protocol

from bragi.persistence.models import ContextObservationRecord, MessageRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.chat_rendering import rendered_chat_request_text
from bragi.providers.contracts import (
    ChatMessage,
    ChatRequest,
    ProviderClient,
    StructuredOutputProvider,
    StructuredOutputRequest,
    StructuredOutputResponse,
)
from bragi.services.evidence import quote_matches_source
from bragi.services.manual_confirmation import manual_memory_confirmation_enabled
from bragi.services.npc_knowledge_audit_service import NpcKnowledgeLeak
from bragi.services.openrouter_routing_settings import request_with_openrouter_routing
from bragi.services.provider_fallbacks import structured_output_with_fallback
from bragi.services.sexual_content_safety import is_fade_to_black_message
from bragi.services.text_script_policy import (
    DEFAULT_SCRIPT_GUARD_MODE,
    ScriptPolicyViolation,
    allowed_generated_scripts,
    first_violation_diagnostic,
    object_text_script_violations,
    script_guard_mode,
    summarize_script_policy_violations,
    text_script_violations,
)

AGENTIC_CONTEXT_PIPELINE_DEFAULT = True
AGENTIC_CONTEXT_PIPELINE_SETTING = "agentic_context_pipeline_enabled"
PLAN_FIRST_NARRATOR_DEFAULT = True
PLAN_FIRST_NARRATOR_SETTING = "plan_first_narrator_enabled"
RESPONSE_VERIFICATION_MODE_SETTING = "response_verification_mode"
RESPONSE_VERIFICATION_MODE_DIAGNOSTIC = "diagnostic"
RESPONSE_VERIFICATION_MODE_RETRY_ONCE = "retry_once"
RESPONSE_VERIFICATION_MODES = frozenset(
    {
        RESPONSE_VERIFICATION_MODE_DIAGNOSTIC,
        RESPONSE_VERIFICATION_MODE_RETRY_ONCE,
    }
)

OBSERVATION_STATUSES_FOR_CURATION = ("pending",)


@dataclass(frozen=True)
class ExtractedObservation:
    observation_type: str
    claim: str
    evidence_quote: str
    source_message_ids: tuple[str, ...]
    scope: str
    confidence: float
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ObservationResult:
    save_id: str
    observed_count: int
    observations: tuple[ContextObservationRecord, ...] = ()
    skipped_reason: str = ""


@dataclass(frozen=True)
class CurationDecision:
    observation_id: str
    action: str
    reason: str
    confidence: float
    memory_body: str = ""
    context_title: str = ""
    context_body: str = ""
    tags: tuple[str, ...] = ()
    script_policy_violations: tuple[ScriptPolicyViolation, ...] = ()


@dataclass(frozen=True)
class CurationResult:
    save_id: str
    considered_count: int
    accepted_count: int = 0
    discarded_count: int = 0
    confirmation_count: int = 0
    skipped_reason: str = ""


@dataclass(frozen=True)
class NpcIntent:
    character_name: str
    stance: str
    current_goal: str
    next_action: str
    should_comply: bool
    cooperation_conditions: tuple[str, ...] = ()
    boundaries: tuple[str, ...] = ()
    route_stage: str = ""
    max_plausible_escalation: str = ""
    reason: str = ""
    evidence_source_ids: tuple[str, ...] = ()
    character_id: str = ""


@dataclass(frozen=True)
class NarrativeBeat:
    description: str
    evidence_source_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RequiredFact:
    fact: str
    evidence_source_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlayerAgencyConstraint:
    constraint: str
    reason: str = ""
    evidence_source_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class StateCommitCandidate:
    operation: str
    state_key: str
    value: dict[str, object]
    reason: str
    confidence: float
    evidence_source_ids: tuple[str, ...] = ()
    evidence_quote: str = ""
    candidate_id: str = ""
    candidate_type: str = ""
    field_path: str = ""
    character_id: str = ""
    target_type: str = ""
    target_id: str = ""
    safe_without_narration_allowed: bool = False


@dataclass(frozen=True)
class NarratorCommitDecision:
    candidate_id: str
    candidate_type: str
    status: str
    safe_to_commit: bool
    reason: str
    evidence_quote: str = ""


@dataclass(frozen=True)
class DatingRouteStageViolation:
    character_name: str
    character_id: str
    route_stage: str
    escalation: str
    reason: str
    evidence_quote: str = ""


@dataclass(frozen=True)
class NarratorMessageSpec:
    intent: str
    thesis: str
    must_say: tuple[str, ...]
    avoid: tuple[str, ...]
    tone: str
    uncertainties: tuple[str, ...]
    evidence_source_ids: tuple[str, ...]
    npc_intents: tuple[NpcIntent, ...] = ()
    narrative_beats: tuple[NarrativeBeat, ...] = ()
    required_facts: tuple[RequiredFact, ...] = ()
    agency_constraints: tuple[PlayerAgencyConstraint, ...] = ()
    state_commit_candidates: tuple[StateCommitCandidate, ...] = ()


@dataclass(frozen=True)
class NarratorVerificationResult:
    passed: bool
    issues: tuple[str, ...] = ()
    retry_feedback: str = ""
    confidence: float = 0.0
    post_turn_update_needed: bool = True
    npc_agency_issues: tuple[str, ...] = ()
    npc_passivity_issues: tuple[str, ...] = ()
    npc_knowledge_leaks: tuple[NpcKnowledgeLeak, ...] = ()
    commit_decisions: tuple[NarratorCommitDecision, ...] = ()
    dating_route_stage_violations: tuple[DatingRouteStageViolation, ...] = ()


class ObservationExtractor(Protocol):
    async def extract(
        self,
        *,
        save_id: str,
        messages: tuple[MessageRecord, ...],
    ) -> tuple[ExtractedObservation, ...]:
        ...


class ContextCurator(Protocol):
    async def curate(
        self,
        *,
        save_id: str,
        observations: tuple[ContextObservationRecord, ...],
    ) -> tuple[CurationDecision, ...]:
        ...


class NarratorPlanner(Protocol):
    async def plan(
        self,
        *,
        save_id: str,
        request: ChatRequest,
    ) -> NarratorMessageSpec:
        ...


class NarratorVerifier(Protocol):
    async def verify(
        self,
        *,
        save_id: str,
        source_request: ChatRequest,
        spec: NarratorMessageSpec,
        narrator_body: str,
    ) -> NarratorVerificationResult:
        ...


class StructuredProviderObservationExtractor:
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

    async def extract(
        self,
        *,
        save_id: str,
        messages: tuple[MessageRecord, ...],
    ) -> tuple[ExtractedObservation, ...]:
        request = request_with_openrouter_routing(
            self.repositories,
            StructuredOutputRequest(
                provider=self.provider_name,
                model_id=self.model_id,
                schema_name="context_observation_extraction",
                schema=_observation_schema(messages),
                messages=_observation_messages(messages),
                temperature=0.0,
            ),
            task="fact_observation",
            save_id=save_id,
        )
        mode = (
            script_guard_mode(self.repositories, save_id=save_id)
            if self.repositories is not None
            else DEFAULT_SCRIPT_GUARD_MODE
        )
        messages_by_id = {message.id: message.body for message in messages}
        for attempt in range(2):
            response = await _structured_response(
                provider=self.provider,
                repositories=self.repositories,
                providers=self.providers,
                request=request,
                task="fact_observation",
                save_id=save_id,
            )
            observations = _observations_from_data(response.data, messages)
            accepted, rejected = _filter_observations_by_script_policy(
                observations,
                messages_by_id=messages_by_id,
                mode=mode,
            )
            if not rejected or attempt == 1:
                return accepted
            request = _structured_request_with_script_policy_feedback(
                request,
                rejected,
            )
        return ()


class ObservationService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        extractor: ObservationExtractor,
    ) -> None:
        self.repositories = repositories
        self.extractor = extractor

    async def observe_turn(
        self,
        *,
        save_id: str,
        source_message_ids: tuple[str, ...],
    ) -> ObservationResult:
        messages = tuple(
            message
            for message in self.repositories.list_messages(save_id)
            if message.id in set(source_message_ids)
        )
        if not messages:
            return ObservationResult(
                save_id=save_id,
                observed_count=0,
                skipped_reason="no source messages",
            )
        extracted = await self.extractor.extract(save_id=save_id, messages=messages)
        mode = script_guard_mode(self.repositories, save_id=save_id)
        messages_by_id = {message.id: message.body for message in messages}
        blocked_source_ids = frozenset(
            message.id
            for message in messages
            if is_fade_to_black_message(
                role=message.role,
                body=message.body,
                safety_transition=message.safety_transition,
            )
        )
        records = tuple(
            self.repositories.add_context_observation(
                save_id=save_id,
                observation_type=observation.observation_type,
                claim=observation.claim,
                evidence_quote=observation.evidence_quote,
                source_message_ids=observation.source_message_ids,
                scope=observation.scope,
                confidence=observation.confidence,
                tags=observation.tags,
                metadata={"observer": "structured_output"},
            )
            for observation in extracted
            if observation.claim.strip()
            and _observation_evidence_is_grounded(
                observation,
                messages_by_id=messages_by_id,
            )
            and not _observation_script_policy_violations(
                observation,
                messages_by_id=messages_by_id,
                mode=mode,
            )
            and not blocked_source_ids.intersection(observation.source_message_ids)
        )
        return ObservationResult(
            save_id=save_id,
            observed_count=len(records),
            observations=records,
        )


class StructuredProviderContextCurator:
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

    async def curate(
        self,
        *,
        save_id: str,
        observations: tuple[ContextObservationRecord, ...],
    ) -> tuple[CurationDecision, ...]:
        request = request_with_openrouter_routing(
            self.repositories,
            StructuredOutputRequest(
                provider=self.provider_name,
                model_id=self.model_id,
                schema_name="context_observation_curation",
                schema=_curation_schema(observations),
                messages=_curation_messages(observations),
                temperature=0.0,
            ),
            task="memory_curation",
            save_id=save_id,
        )
        response = await _structured_response(
            provider=self.provider,
            repositories=self.repositories,
            providers=self.providers,
            request=request,
            task="memory_curation",
            save_id=save_id,
        )
        mode = (
            script_guard_mode(self.repositories, save_id=save_id)
            if self.repositories is not None
            else DEFAULT_SCRIPT_GUARD_MODE
        )
        source_texts_by_observation = _curation_source_texts_by_observation(
            repositories=self.repositories,
            save_id=save_id,
            observations=observations,
        )
        decisions = _curation_decisions_from_data(response.data, observations)
        decisions = _mark_curation_decision_script_policy_violations(
            decisions,
            observations=observations,
            source_texts_by_observation=source_texts_by_observation,
            mode=mode,
        )
        if not any(decision.script_policy_violations for decision in decisions):
            return decisions
        retry_request = _structured_request_with_script_policy_feedback(
            request,
            tuple(
                violation
                for decision in decisions
                for violation in decision.script_policy_violations
            ),
        )
        retry_response = await _structured_response(
            provider=self.provider,
            repositories=self.repositories,
            providers=self.providers,
            request=retry_request,
            task="memory_curation",
            save_id=save_id,
        )
        retry_decisions = _curation_decisions_from_data(
            retry_response.data,
            observations,
        )
        return _mark_curation_decision_script_policy_violations(
            retry_decisions,
            observations=observations,
            source_texts_by_observation=source_texts_by_observation,
            mode=mode,
        )


class ContextCurationService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        curator: ContextCurator,
    ) -> None:
        self.repositories = repositories
        self.curator = curator

    async def curate_pending(self, save_id: str) -> CurationResult:
        all_observations = tuple(
            self.repositories.list_context_observations(
                save_id,
                statuses=OBSERVATION_STATUSES_FOR_CURATION,
            )
        )
        if not all_observations:
            return CurationResult(
                save_id=save_id,
                considered_count=0,
                skipped_reason="no pending observations",
            )
        source_texts_by_observation = _curation_source_texts_by_observation(
            repositories=self.repositories,
            save_id=save_id,
            observations=all_observations,
        )
        observations = tuple(
            observation
            for observation in all_observations
            if _context_observation_evidence_is_grounded(
                observation,
                source_texts_by_observation=source_texts_by_observation,
            )
        )
        discarded_count = 0
        for observation in all_observations:
            if observation in observations:
                continue
            discarded_count += 1
            self._mark_observation_evidence_rejected(observation)
        if not observations:
            return CurationResult(
                save_id=save_id,
                considered_count=len(all_observations),
                discarded_count=discarded_count,
            )
        decisions = await self.curator.curate(
            save_id=save_id,
            observations=observations,
        )
        mode = script_guard_mode(self.repositories, save_id=save_id)
        decisions = _mark_curation_decision_script_policy_violations(
            decisions,
            observations=observations,
            source_texts_by_observation=source_texts_by_observation,
            mode=mode,
        )
        observations_by_id = {
            observation.id: observation for observation in observations
        }
        accepted_count = 0
        confirmation_count = 0
        for decision in decisions:
            curated_observation: ContextObservationRecord | None = (
                observations_by_id.get(decision.observation_id)
            )
            if curated_observation is None:
                continue
            if decision.script_policy_violations:
                discarded_count += 1
                self._mark_observation_script_policy_rejected(
                    curated_observation,
                    decision.script_policy_violations,
                )
                continue
            if decision.action == "durable_memory":
                body = decision.memory_body.strip() or curated_observation.claim
                if body.strip():
                    source_message_ids = tuple(curated_observation.source_message_ids)
                    tags = tuple(decision.tags or curated_observation.tags)
                    if _curated_memory_exists(
                        self.repositories,
                        save_id=save_id,
                        body=body,
                        source_message_ids=source_message_ids,
                    ):
                        self._mark_observation(
                            curated_observation,
                            decision,
                            "accepted",
                        )
                        continue
                    if manual_memory_confirmation_enabled(
                        self.repositories,
                        save_id=save_id,
                    ):
                        self._queue_memory_confirmation(
                            save_id=save_id,
                            observation=curated_observation,
                            body=body,
                            tags=tags,
                            confidence=decision.confidence,
                            reason=decision.reason,
                        )
                        confirmation_count += 1
                        self._mark_observation(
                            curated_observation,
                            decision,
                            "needs_confirmation",
                        )
                        continue
                    self.repositories.add_memory(
                        save_id=save_id,
                        body=body.strip(),
                        tags=list(tags),
                        importance=decision.confidence,
                        source_message_id=(
                            curated_observation.source_message_ids[0]
                            if curated_observation.source_message_ids
                            else None
                        ),
                        source_message_ids=source_message_ids,
                    )
                    accepted_count += 1
                    self._mark_observation(curated_observation, decision, "accepted")
            elif decision.action in {"save_context", "scene_scratch"}:
                body = decision.context_body.strip() or curated_observation.claim
                self.repositories.upsert_context_source(
                    save_id=save_id,
                    source_type="observation",
                    source_id=curated_observation.id,
                    title=(
                        decision.context_title.strip() or curated_observation.claim
                    ),
                    body=body,
                    metadata={
                        "observation_id": curated_observation.id,
                        "observation_type": curated_observation.observation_type,
                        "fact_type": curated_observation.observation_type,
                        "scope": curated_observation.scope,
                        "source_message_ids": (
                            curated_observation.source_message_ids
                        ),
                        "evidence_quote": curated_observation.evidence_quote,
                        "curation_action": decision.action,
                        "importance": decision.confidence,
                    },
                )
                accepted_count += 1
                self._mark_observation(curated_observation, decision, "accepted")
            elif decision.action == "needs_confirmation":
                self.repositories.add_context_update_suggestion(
                    save_id=save_id,
                    update_type="review",
                    entity_type="observation",
                    entity_id=curated_observation.id,
                    field_path=decision.action,
                    proposed_value=_decision_metadata(decision),
                    reason=decision.reason,
                    confidence=decision.confidence,
                    source_message_ids=curated_observation.source_message_ids,
                )
                confirmation_count += 1
                self._mark_observation(
                    curated_observation,
                    decision,
                    "needs_confirmation",
                )
            elif decision.action == "discard":
                discarded_count += 1
                self._mark_observation(curated_observation, decision, "discarded")
        return CurationResult(
            save_id=save_id,
            considered_count=len(all_observations),
            accepted_count=accepted_count,
            discarded_count=discarded_count,
            confirmation_count=confirmation_count,
        )

    def _mark_observation(
        self,
        observation: ContextObservationRecord,
        decision: CurationDecision,
        status: str,
    ) -> None:
        self.repositories.update_context_observation(
            observation.id,
            status=status,
            metadata={"curation": _decision_metadata(decision)},
        )

    def _mark_observation_script_policy_rejected(
        self,
        observation: ContextObservationRecord,
        violations: tuple[ScriptPolicyViolation, ...],
    ) -> None:
        diagnostic = first_violation_diagnostic(violations)
        self.repositories.update_context_observation(
            observation.id,
            status="discarded",
            metadata={
                "script_policy_rejected": diagnostic,
                "curation": {
                    "action": "discard",
                    "reason": summarize_script_policy_violations(violations),
                    "confidence": 0.0,
                    "script_policy_rejected": diagnostic,
                },
            },
        )

    def _mark_observation_evidence_rejected(
        self,
        observation: ContextObservationRecord,
    ) -> None:
        self.repositories.update_context_observation(
            observation.id,
            status="discarded",
            metadata={
                "evidence_rejected": {
                    "reason": "evidence_quote not found in source messages",
                },
                "curation": {
                    "action": "discard",
                    "reason": "Evidence quote is not grounded in source messages.",
                    "confidence": 0.0,
                },
            },
        )

    def _queue_memory_confirmation(
        self,
        *,
        save_id: str,
        observation: ContextObservationRecord,
        body: str,
        tags: tuple[str, ...],
        confidence: float,
        reason: str,
    ) -> None:
        proposed_value = _curated_memory_proposed_value(
            body=body,
            tags=tags,
            importance=confidence,
            source_message_ids=tuple(observation.source_message_ids),
            source_observation_id=observation.id,
        )
        suggestion = self.repositories.add_context_update_suggestion(
            save_id=save_id,
            update_type="create",
            entity_type="memory",
            field_path="*",
            proposed_value=proposed_value,
            status="pending",
            reason=reason or "Confirm curated memory",
            confidence=confidence,
            source_message_ids=list(observation.source_message_ids),
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
            reason=reason or "Confirm curated memory",
            confidence=confidence,
            source_message_ids=list(observation.source_message_ids),
        )


class StructuredProviderNarratorPlanner:
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

    async def plan(
        self,
        *,
        save_id: str,
        request: ChatRequest,
    ) -> NarratorMessageSpec:
        structured_request = request_with_openrouter_routing(
            self.repositories,
            StructuredOutputRequest(
                provider=self.provider_name,
                model_id=self.model_id,
                schema_name="narrator_message_plan",
                schema=_planner_schema(),
                messages=_planner_messages(request),
                temperature=0.0,
            ),
            task="response_planning",
            save_id=save_id,
        )
        response = await _structured_response(
            provider=self.provider,
            repositories=self.repositories,
            providers=self.providers,
            request=structured_request,
            task="response_planning",
            save_id=save_id,
        )
        return _narrator_message_spec_from_data(response.data)


class StructuredProviderNarratorVerifier:
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

    async def verify(
        self,
        *,
        save_id: str,
        source_request: ChatRequest,
        spec: NarratorMessageSpec,
        narrator_body: str,
    ) -> NarratorVerificationResult:
        structured_request = request_with_openrouter_routing(
            self.repositories,
            StructuredOutputRequest(
                provider=self.provider_name,
                model_id=self.model_id,
                schema_name="narrator_message_verification",
                schema=_verifier_schema(),
                messages=_verifier_messages(
                    spec=spec,
                    request=source_request,
                    narrator_body=narrator_body,
                ),
                temperature=0.0,
            ),
            task="response_verification",
            save_id=save_id,
        )
        response = await _structured_response(
            provider=self.provider,
            repositories=self.repositories,
            providers=self.providers,
            request=structured_request,
            task="response_verification",
            save_id=save_id,
        )
        return _verification_result_from_data(response.data)


def format_narrator_message_spec(spec: NarratorMessageSpec) -> str:
    parts = [
        "Narration turn plan",
        f"Intent: {spec.intent}",
        f"Thesis: {spec.thesis}",
        f"Tone: {spec.tone}",
    ]
    if spec.narrative_beats:
        parts.append("Narrative beats:")
        for index, beat in enumerate(spec.narrative_beats, start=1):
            parts.append(
                f"{index}. {beat.description}"
                + _format_evidence_suffix(beat.evidence_source_ids)
            )
    if spec.required_facts:
        parts.append("Required facts/reveals:")
        for fact in spec.required_facts:
            parts.append(_format_required_fact(fact))
    if spec.must_say:
        parts.append("Must say: " + "; ".join(spec.must_say))
    if spec.avoid:
        parts.append("Avoid: " + "; ".join(spec.avoid))
    if spec.agency_constraints:
        parts.append("Player-agency constraints:")
        for constraint in spec.agency_constraints:
            parts.append(_format_agency_constraint(constraint))
    if spec.uncertainties:
        parts.append("Uncertainties: " + "; ".join(spec.uncertainties))
    if spec.npc_intents:
        parts.append("Character intent/action beats:")
        for intent in spec.npc_intents:
            parts.append(_format_npc_intent(intent))
    if spec.state_commit_candidates:
        parts.append("State commit candidates (do not persist automatically):")
        for candidate in spec.state_commit_candidates:
            parts.append(_format_state_commit_candidate(candidate))
    if spec.evidence_source_ids:
        parts.append("Evidence source IDs: " + ", ".join(spec.evidence_source_ids))
    return "\n".join(part for part in parts if part.strip())


def narration_evidence_source_ids(spec: NarratorMessageSpec) -> tuple[str, ...]:
    evidence: list[str] = []
    evidence.extend(spec.evidence_source_ids)
    for beat in spec.narrative_beats:
        evidence.extend(beat.evidence_source_ids)
    for fact in spec.required_facts:
        evidence.extend(fact.evidence_source_ids)
    for constraint in spec.agency_constraints:
        evidence.extend(constraint.evidence_source_ids)
    for intent in spec.npc_intents:
        evidence.extend(intent.evidence_source_ids)
    for candidate in spec.state_commit_candidates:
        evidence.extend(candidate.evidence_source_ids)
    return tuple(dict.fromkeys(item for item in evidence if item.strip()))


def _format_required_fact(fact: RequiredFact) -> str:
    return "- " + fact.fact + _format_evidence_suffix(fact.evidence_source_ids)


def _format_agency_constraint(constraint: PlayerAgencyConstraint) -> str:
    parts = [
        constraint.constraint,
        f"reason: {constraint.reason}" if constraint.reason else "",
        (
            "evidence: " + ", ".join(constraint.evidence_source_ids)
            if constraint.evidence_source_ids
            else ""
        ),
    ]
    return "- " + " | ".join(part for part in parts if part)


def _format_npc_intent(intent: NpcIntent) -> str:
    parts = [
        intent.character_name,
        f"id: {intent.character_id}" if intent.character_id else "",
        f"stance: {intent.stance}" if intent.stance else "",
        f"goal: {intent.current_goal}" if intent.current_goal else "",
        f"next action: {intent.next_action}" if intent.next_action else "",
        f"should comply: {'yes' if intent.should_comply else 'no'}",
        (
            "cooperation conditions: "
            + "; ".join(intent.cooperation_conditions)
            if intent.cooperation_conditions
            else ""
        ),
        (
            "boundaries: " + "; ".join(intent.boundaries)
            if intent.boundaries
            else ""
        ),
        f"route stage: {intent.route_stage}" if intent.route_stage else "",
        (
            f"max plausible escalation: {intent.max_plausible_escalation}"
            if intent.max_plausible_escalation
            else ""
        ),
        f"reason: {intent.reason}" if intent.reason else "",
        (
            "evidence: " + ", ".join(intent.evidence_source_ids)
            if intent.evidence_source_ids
            else ""
        ),
    ]
    return "- " + " | ".join(part for part in parts if part)


def _format_state_commit_candidate(candidate: StateCommitCandidate) -> str:
    parts = [
        "candidate only",
        f"id: {candidate.candidate_id}" if candidate.candidate_id else "",
        f"type: {candidate.candidate_type}" if candidate.candidate_type else "",
        f"operation: {candidate.operation}",
        f"key: {candidate.state_key}",
        f"field: {candidate.field_path}" if candidate.field_path else "",
        (
            f"character_id: {candidate.character_id}"
            if candidate.character_id
            else ""
        ),
        (
            f"target: {candidate.target_type}:{candidate.target_id}"
            if candidate.target_type and candidate.target_id
            else ""
        ),
        f"value: {_compact_json(candidate.value)}",
        f"confidence: {candidate.confidence:.2g}",
        (
            "safe without explicit narration: yes"
            if candidate.safe_without_narration_allowed
            else ""
        ),
        f"reason: {candidate.reason}" if candidate.reason else "",
        (
            "evidence: " + ", ".join(candidate.evidence_source_ids)
            if candidate.evidence_source_ids
            else ""
        ),
        f"quote: {candidate.evidence_quote}" if candidate.evidence_quote else "",
    ]
    return "- " + " | ".join(part for part in parts if part)


def _format_evidence_suffix(evidence_source_ids: tuple[str, ...]) -> str:
    if not evidence_source_ids:
        return ""
    return " [evidence: " + ", ".join(evidence_source_ids) + "]"


def _compact_json(value: dict[str, object]) -> str:
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)


def agentic_context_pipeline_enabled(
    repositories: PersistenceRepositories,
    *,
    save_id: str | None = None,
) -> bool:
    value = repositories.get_effective_setting(
        AGENTIC_CONTEXT_PIPELINE_SETTING,
        save_id=save_id,
    )
    if value is None:
        return AGENTIC_CONTEXT_PIPELINE_DEFAULT
    return bool(value)


def plan_first_narrator_enabled(
    repositories: PersistenceRepositories,
    *,
    save_id: str | None = None,
) -> bool:
    value = repositories.get_effective_setting(
        PLAN_FIRST_NARRATOR_SETTING,
        save_id=save_id,
    )
    if value is None:
        return PLAN_FIRST_NARRATOR_DEFAULT
    return bool(value)


def response_verification_mode(
    repositories: PersistenceRepositories,
    *,
    save_id: str | None = None,
) -> str:
    value = repositories.get_effective_setting(
        RESPONSE_VERIFICATION_MODE_SETTING,
        save_id=save_id,
    )
    mode = value if isinstance(value, str) else RESPONSE_VERIFICATION_MODE_DIAGNOSTIC
    if mode not in RESPONSE_VERIFICATION_MODES:
        return RESPONSE_VERIFICATION_MODE_DIAGNOSTIC
    return mode


async def _structured_response(
    *,
    provider: StructuredOutputProvider,
    repositories: PersistenceRepositories | None,
    providers: dict[str, ProviderClient] | None,
    request: StructuredOutputRequest,
    task: str,
    save_id: str,
) -> StructuredOutputResponse:
    if repositories is not None and providers is not None:
        return await structured_output_with_fallback(
            repositories=repositories,
            providers=providers,
            request=request,
            task=task,
            save_id=save_id,
        )
    return await provider.generate_structured_output(request)


def _observation_schema(messages: tuple[MessageRecord, ...]) -> dict[str, object]:
    source_schema: dict[str, object] = {"type": "string"}
    message_ids = [message.id for message in messages]
    if message_ids:
        source_schema["enum"] = message_ids
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "observations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "observation_type": {"type": "string"},
                        "claim": {"type": "string"},
                        "evidence_quote": {"type": "string"},
                        "source_message_ids": {
                            "type": "array",
                            "items": source_schema,
                        },
                        "scope": {
                            "type": "string",
                            "enum": ["turn", "scene", "save", "durable"],
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "observation_type",
                        "claim",
                        "evidence_quote",
                        "source_message_ids",
                        "scope",
                        "confidence",
                        "tags",
                    ],
                },
            }
        },
        "required": ["observations"],
    }


def _observation_messages(
    messages: tuple[MessageRecord, ...],
) -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(
            role="system",
            body=(
                "Extract high-recall candidate Bragi observations from the "
                "completed turn. Use the enforced schema. Do not decide final "
                "importance. Include only claims with explicit source evidence."
                " A marked narrator safety transition is only an off-screen "
                "event and elapsed time; do not infer intimate details from it."
            ),
        ),
        ChatMessage(role="user", body=_messages_text(messages)),
    )


def _observations_from_data(
    data: dict[str, object],
    messages: tuple[MessageRecord, ...],
) -> tuple[ExtractedObservation, ...]:
    raw_items = data.get("observations", [])
    if not isinstance(raw_items, list):
        raise ValueError("Observation extraction observations must be a list")
    allowed_ids = {message.id for message in messages}
    message_bodies_by_id = {message.id: message.body for message in messages}
    observations: list[ExtractedObservation] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise ValueError("Observation extraction item must be an object")
        source_ids = tuple(
            source_id
            for source_id in _string_tuple(raw.get("source_message_ids"))
            if source_id in allowed_ids
        )
        claim = _string(raw.get("claim"))
        evidence_quote = _string(raw.get("evidence_quote"))
        if not claim or not evidence_quote or not source_ids:
            continue
        if not any(
            quote_matches_source(evidence_quote, message_bodies_by_id[source_id])
            for source_id in source_ids
        ):
            continue
        observations.append(
            ExtractedObservation(
                observation_type=_string(raw.get("observation_type"))
                or "observation",
                claim=claim,
                evidence_quote=evidence_quote,
                source_message_ids=source_ids,
                scope=_string(raw.get("scope")) or "turn",
                confidence=_float(raw.get("confidence")),
                tags=_string_tuple(raw.get("tags")),
            )
        )
    return tuple(observations)


def _observation_evidence_is_grounded(
    observation: ExtractedObservation,
    *,
    messages_by_id: dict[str, str],
) -> bool:
    if not observation.evidence_quote.strip():
        return False
    return any(
        source_body is not None
        and quote_matches_source(observation.evidence_quote, source_body)
        for source_body in (
            messages_by_id.get(source_id)
            for source_id in observation.source_message_ids
        )
    )


def _curation_schema(
    observations: tuple[ContextObservationRecord, ...],
) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "observation_id": {
                            "type": "string",
                            "enum": [observation.id for observation in observations],
                        },
                        "action": {
                            "type": "string",
                            "enum": [
                                "durable_memory",
                                "save_context",
                                "scene_scratch",
                                "discard",
                                "needs_confirmation",
                            ],
                        },
                        "reason": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "memory_body": {"type": "string"},
                        "context_title": {"type": "string"},
                        "context_body": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "observation_id",
                        "action",
                        "reason",
                        "confidence",
                        "memory_body",
                        "context_title",
                        "context_body",
                        "tags",
                    ],
                },
            }
        },
        "required": ["decisions"],
    }


def _curation_messages(
    observations: tuple[ContextObservationRecord, ...],
) -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(
            role="system",
            body=(
                "Curate candidate Bragi observations into durable memory, save "
                "context, scene scratchpad, discard, or needs confirmation. Use "
                "the enforced schema. Preserve provenance and do not treat "
                "importance as permanent when the observation may be future-dependent."
            ),
        ),
        ChatMessage(role="user", body=_observations_text(observations)),
    )


def _curation_decisions_from_data(
    data: dict[str, object],
    observations: tuple[ContextObservationRecord, ...],
) -> tuple[CurationDecision, ...]:
    raw_items = data.get("decisions", [])
    if not isinstance(raw_items, list):
        raise ValueError("Curation decisions must be a list")
    allowed_ids = {observation.id for observation in observations}
    decisions: list[CurationDecision] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise ValueError("Curation decision must be an object")
        observation_id = _string(raw.get("observation_id"))
        if observation_id not in allowed_ids:
            continue
        decisions.append(
            CurationDecision(
                observation_id=observation_id,
                action=_string(raw.get("action")),
                reason=_string(raw.get("reason")),
                confidence=_float(raw.get("confidence")),
                memory_body=_string(raw.get("memory_body")),
                context_title=_string(raw.get("context_title")),
                context_body=_string(raw.get("context_body")),
                tags=_string_tuple(raw.get("tags")),
            )
        )
    return tuple(decisions)


def _filter_observations_by_script_policy(
    observations: tuple[ExtractedObservation, ...],
    *,
    messages_by_id: dict[str, str],
    mode: str,
) -> tuple[tuple[ExtractedObservation, ...], tuple[ScriptPolicyViolation, ...]]:
    accepted: list[ExtractedObservation] = []
    rejected: list[ScriptPolicyViolation] = []
    for observation in observations:
        violations = _observation_script_policy_violations(
            observation,
            messages_by_id=messages_by_id,
            mode=mode,
        )
        if violations:
            rejected.extend(violations)
        else:
            accepted.append(observation)
    return tuple(accepted), tuple(rejected)


def _observation_script_policy_violations(
    observation: ExtractedObservation,
    *,
    messages_by_id: dict[str, str],
    mode: str,
) -> tuple[ScriptPolicyViolation, ...]:
    allowed_scripts = _allowed_scripts_for_message_ids(
        observation.source_message_ids,
        messages_by_id=messages_by_id,
        fallback_texts=(observation.evidence_quote,),
    )
    violations: list[ScriptPolicyViolation] = []
    for field_name, value in (
        ("claim", observation.claim),
        ("evidence_quote", observation.evidence_quote),
    ):
        violations.extend(
            text_script_violations(
                value,
                allowed_scripts=allowed_scripts,
                mode=mode,
                field_name=field_name,
            )
        )
    violations.extend(
        object_text_script_violations(
            observation.tags,
            allowed_scripts=allowed_scripts,
            mode=mode,
            field_name="tags",
        )
    )
    return tuple(violations)


def _mark_curation_decision_script_policy_violations(
    decisions: tuple[CurationDecision, ...],
    *,
    observations: tuple[ContextObservationRecord, ...],
    source_texts_by_observation: dict[str, tuple[str, ...]],
    mode: str,
) -> tuple[CurationDecision, ...]:
    observations_by_id = {
        observation.id: observation for observation in observations
    }
    marked: list[CurationDecision] = []
    for decision in decisions:
        observation = observations_by_id.get(decision.observation_id)
        if observation is None:
            marked.append(decision)
            continue
        marked.append(
            replace(
                decision,
                script_policy_violations=_curation_decision_script_policy_violations(
                    decision,
                    observation=observation,
                    source_texts=source_texts_by_observation.get(
                        observation.id,
                        (observation.evidence_quote,),
                    ),
                    mode=mode,
                ),
            )
        )
    return tuple(marked)


def _curation_decision_script_policy_violations(
    decision: CurationDecision,
    *,
    observation: ContextObservationRecord,
    source_texts: tuple[str, ...],
    mode: str,
) -> tuple[ScriptPolicyViolation, ...]:
    allowed_scripts = allowed_generated_scripts(source_texts)
    values: list[tuple[str, object]] = [("reason", decision.reason)]
    if decision.action == "durable_memory":
        values.append(
            ("memory_body", decision.memory_body.strip() or observation.claim)
        )
    elif decision.action in {"save_context", "scene_scratch"}:
        values.extend(
            (
                ("context_title", decision.context_title.strip() or observation.claim),
                ("context_body", decision.context_body.strip() or observation.claim),
            )
        )
    values.append(("tags", decision.tags))
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


def _allowed_scripts_for_message_ids(
    source_message_ids: tuple[str, ...],
    *,
    messages_by_id: dict[str, str],
    fallback_texts: tuple[str, ...] = (),
) -> frozenset[str]:
    texts = tuple(
        messages_by_id[source_message_id]
        for source_message_id in source_message_ids
        if source_message_id in messages_by_id
    )
    if not texts:
        texts = fallback_texts
    return allowed_generated_scripts(texts)


def _curation_source_texts_by_observation(
    *,
    repositories: PersistenceRepositories | None,
    save_id: str,
    observations: tuple[ContextObservationRecord, ...],
) -> dict[str, tuple[str, ...]]:
    if repositories is None:
        return {
            observation.id: (
                (observation.evidence_quote,) if observation.evidence_quote else ()
            )
            for observation in observations
        }
    messages_by_id = {
        message.id: message.body for message in repositories.list_messages(save_id)
    }
    texts_by_observation: dict[str, tuple[str, ...]] = {}
    for observation in observations:
        texts = tuple(
            body
            for source_message_id in observation.source_message_ids
            if (body := messages_by_id.get(source_message_id))
        )
        texts_by_observation[observation.id] = texts
    return texts_by_observation


def _context_observation_evidence_is_grounded(
    observation: ContextObservationRecord,
    *,
    source_texts_by_observation: dict[str, tuple[str, ...]],
) -> bool:
    if not observation.evidence_quote.strip() or not observation.source_message_ids:
        return False
    return any(
        quote_matches_source(observation.evidence_quote, source_text)
        for source_text in source_texts_by_observation.get(observation.id, ())
    )


def _structured_request_with_script_policy_feedback(
    request: StructuredOutputRequest,
    violations: tuple[ScriptPolicyViolation, ...],
) -> StructuredOutputRequest:
    return replace(
        request,
        messages=(
            *request.messages,
            ChatMessage(
                role="user",
                body=(
                    "The previous structured output was rejected by Bragi's "
                    "generated-text script policy. Retry without introducing "
                    "scripts that are absent from the source text. "
                    f"{summarize_script_policy_violations(violations)}."
                ),
            ),
        ),
    )


def _planner_schema() -> dict[str, object]:
    string_array = {"type": "array", "items": {"type": "string"}}
    narrative_beat = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "description": {"type": "string"},
            "evidence_source_ids": string_array,
        },
        "required": ["description", "evidence_source_ids"],
    }
    required_fact = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "fact": {"type": "string"},
            "evidence_source_ids": string_array,
        },
        "required": ["fact", "evidence_source_ids"],
    }
    agency_constraint = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "constraint": {"type": "string"},
            "reason": {"type": "string"},
            "evidence_source_ids": string_array,
        },
        "required": ["constraint", "reason", "evidence_source_ids"],
    }
    npc_intent = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "character_id": {"type": "string"},
            "character_name": {"type": "string"},
            "stance": {"type": "string"},
            "current_goal": {"type": "string"},
            "next_action": {"type": "string"},
            "should_comply": {"type": "boolean"},
            "cooperation_conditions": string_array,
            "boundaries": string_array,
            "route_stage": {"type": "string"},
            "max_plausible_escalation": {"type": "string"},
            "reason": {"type": "string"},
            "evidence_source_ids": string_array,
        },
        "required": [
            "character_id",
            "character_name",
            "stance",
            "current_goal",
            "next_action",
            "should_comply",
            "cooperation_conditions",
            "boundaries",
            "route_stage",
            "max_plausible_escalation",
            "reason",
            "evidence_source_ids",
        ],
    }
    state_commit_candidate = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "candidate_id": {"type": "string"},
            "candidate_type": {
                "type": "string",
                "enum": [
                    "scene_presence",
                    "character_learned_memory",
                    "character_knowledge_edge",
                    "scene_snapshot_field",
                ],
            },
            "operation": {
                "type": "string",
                "enum": ["create", "update", "upsert", "delete"],
            },
            "state_key": {"type": "string"},
            "field_path": {"type": "string"},
            "character_id": {"type": "string"},
            "target_type": {"type": "string"},
            "target_id": {"type": "string"},
            "value": {"type": "object"},
            "safe_without_narration_allowed": {"type": "boolean"},
            "reason": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_source_ids": string_array,
            "evidence_quote": {"type": "string"},
        },
        "required": [
            "candidate_id",
            "candidate_type",
            "operation",
            "state_key",
            "field_path",
            "character_id",
            "target_type",
            "target_id",
            "value",
            "safe_without_narration_allowed",
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
            "intent": {"type": "string"},
            "thesis": {"type": "string"},
            "narrative_beats": {
                "type": "array",
                "items": narrative_beat,
                "maxItems": 12,
            },
            "required_facts": {
                "type": "array",
                "items": required_fact,
                "maxItems": 12,
            },
            "must_say": string_array,
            "avoid": string_array,
            "agency_constraints": {
                "type": "array",
                "items": agency_constraint,
                "maxItems": 12,
            },
            "tone": {"type": "string"},
            "uncertainties": string_array,
            "evidence_source_ids": string_array,
            "npc_intents": {"type": "array", "items": npc_intent, "maxItems": 12},
            "state_commit_candidates": {
                "type": "array",
                "items": state_commit_candidate,
                "maxItems": 12,
            },
        },
        "required": [
            "intent",
            "thesis",
            "narrative_beats",
            "required_facts",
            "must_say",
            "avoid",
            "agency_constraints",
            "tone",
            "uncertainties",
            "evidence_source_ids",
            "npc_intents",
            "state_commit_candidates",
        ],
    }


def _planner_messages(request: ChatRequest) -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(
            role="system",
            body=(
                "Create a compact message spec for the next Bragi narrator "
                "response. Decide what the response should accomplish, but do "
                "not write final prose. Use the enforced schema. Include "
                "ordered narrative beats, required facts or reveals, player "
                "agency constraints, unresolved uncertainties, and any "
                "state-affecting commit candidates as candidates only. Give "
                "each commit candidate a stable candidate_id, candidate_type, "
                "valid evidence_source_ids, and evidence_quote copied exactly "
                "from a cited source. For "
                "each present non-player character with meaningful agency in "
                "this beat, include an npc_intents item grounded in evidence. "
                "Favor visible initiative: present NPCs should interrupt, "
                "demand, refuse, leave, escalate, advance clocks, or otherwise "
                "change the situation when supported; leave them restrained "
                "only when evidence or route pacing supports restraint. "
                "Preserve the full spectrum of NPC stances: a naturally "
                "trusting character may cooperate, while a hostile, "
                "self-interested, unfair, or unreasonable character should act "
                "from that profile when evidence supports it. "
                "When the source request includes deterministic dating-route "
                "pacing, copy the route stage and maximum plausible escalation "
                "into that character's npc_intents item and keep planned "
                "actions within that bound. Use empty strings for those route "
                "fields when no dating route is provided for the character. "
                "Player agency does not imply NPC compliance."
            ),
        ),
        ChatMessage(role="user", body=rendered_chat_request_text(request)),
    )


def _narrator_message_spec_from_data(data: dict[str, object]) -> NarratorMessageSpec:
    return NarratorMessageSpec(
        intent=_string(data.get("intent")),
        thesis=_string(data.get("thesis")),
        must_say=_string_tuple(data.get("must_say")),
        avoid=_string_tuple(data.get("avoid")),
        tone=_string(data.get("tone")),
        uncertainties=_string_tuple(data.get("uncertainties")),
        evidence_source_ids=_string_tuple(data.get("evidence_source_ids")),
        narrative_beats=_narrative_beats_from_data(data.get("narrative_beats")),
        required_facts=_required_facts_from_data(data.get("required_facts")),
        agency_constraints=_agency_constraints_from_data(
            data.get("agency_constraints")
        ),
        state_commit_candidates=_state_commit_candidates_from_data(
            data.get("state_commit_candidates")
        ),
        npc_intents=_npc_intents_from_data(data.get("npc_intents")),
    )


def _narrative_beats_from_data(value: object) -> tuple[NarrativeBeat, ...]:
    if not isinstance(value, list):
        return ()
    beats: list[NarrativeBeat] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        description = _string(item.get("description"))
        if not description:
            continue
        beats.append(
            NarrativeBeat(
                description=description,
                evidence_source_ids=_string_tuple(item.get("evidence_source_ids")),
            )
        )
    return tuple(beats)


def _required_facts_from_data(value: object) -> tuple[RequiredFact, ...]:
    if not isinstance(value, list):
        return ()
    facts: list[RequiredFact] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        fact = _string(item.get("fact"))
        if not fact:
            continue
        facts.append(
            RequiredFact(
                fact=fact,
                evidence_source_ids=_string_tuple(item.get("evidence_source_ids")),
            )
        )
    return tuple(facts)


def _agency_constraints_from_data(
    value: object,
) -> tuple[PlayerAgencyConstraint, ...]:
    if not isinstance(value, list):
        return ()
    constraints: list[PlayerAgencyConstraint] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        constraint = _string(item.get("constraint"))
        if not constraint:
            continue
        constraints.append(
            PlayerAgencyConstraint(
                constraint=constraint,
                reason=_string(item.get("reason")),
                evidence_source_ids=_string_tuple(item.get("evidence_source_ids")),
            )
        )
    return tuple(constraints)


def _state_commit_candidates_from_data(
    value: object,
) -> tuple[StateCommitCandidate, ...]:
    if not isinstance(value, list):
        return ()
    candidates: list[StateCommitCandidate] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        state_key = _string(item.get("state_key"))
        if not state_key:
            continue
        evidence_source_ids = _string_tuple(item.get("evidence_source_ids"))
        evidence_quote = _string(item.get("evidence_quote"))
        if not evidence_source_ids or not evidence_quote:
            continue
        raw_value = item.get("value")
        candidate_type = _string(item.get("candidate_type"))
        candidate_id = _string(item.get("candidate_id"))
        candidates.append(
            StateCommitCandidate(
                operation=_string(item.get("operation")) or "upsert",
                state_key=state_key,
                value=dict(raw_value) if isinstance(raw_value, dict) else {},
                reason=_string(item.get("reason")),
                confidence=_float(item.get("confidence")),
                evidence_source_ids=evidence_source_ids,
                evidence_quote=evidence_quote,
                candidate_id=candidate_id,
                candidate_type=candidate_type,
                field_path=_string(item.get("field_path")),
                character_id=_string(item.get("character_id")),
                target_type=_string(item.get("target_type")),
                target_id=_string(item.get("target_id")),
                safe_without_narration_allowed=bool(
                    item.get("safe_without_narration_allowed")
                ),
            )
        )
    return tuple(candidates)


def _npc_intents_from_data(value: object) -> tuple[NpcIntent, ...]:
    if not isinstance(value, list):
        return ()
    intents: list[NpcIntent] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        character_name = _string(item.get("character_name"))
        if not character_name:
            continue
        intents.append(
            NpcIntent(
                character_name=character_name,
                stance=_string(item.get("stance")),
                current_goal=_string(item.get("current_goal")),
                next_action=_string(item.get("next_action")),
                should_comply=bool(item.get("should_comply")),
                cooperation_conditions=_string_tuple(
                    item.get("cooperation_conditions")
                ),
                boundaries=_string_tuple(item.get("boundaries")),
                route_stage=_string(item.get("route_stage")),
                max_plausible_escalation=_string(
                    item.get("max_plausible_escalation")
                ),
                reason=_string(item.get("reason")),
                evidence_source_ids=_string_tuple(item.get("evidence_source_ids")),
                character_id=_string(item.get("character_id")),
            )
        )
    return tuple(intents)


def _verifier_schema() -> dict[str, object]:
    commit_decision = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "candidate_id": {"type": "string"},
            "candidate_type": {
                "type": "string",
                "enum": [
                    "scene_presence",
                    "character_learned_memory",
                    "character_knowledge_edge",
                    "scene_snapshot_field",
                ],
            },
            "status": {
                "type": "string",
                "enum": [
                    "rendered",
                    "contradicted",
                    "omitted",
                    "unclear",
                    "safe_without_narration",
                ],
            },
            "safe_to_commit": {"type": "boolean"},
            "reason": {"type": "string"},
            "evidence_quote": {"type": "string"},
        },
        "required": [
            "candidate_id",
            "candidate_type",
            "status",
            "safe_to_commit",
            "reason",
            "evidence_quote",
        ],
    }
    leak = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "speaker_name": {"type": "string"},
            "claim": {"type": "string"},
            "reason": {"type": "string"},
            "target_type": {"type": "string"},
            "target_id": {"type": "string"},
        },
        "required": [
            "speaker_name",
            "claim",
            "reason",
            "target_type",
            "target_id",
        ],
    }
    dating_route_stage_violation = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "character_name": {"type": "string"},
            "character_id": {"type": "string"},
            "route_stage": {"type": "string"},
            "escalation": {"type": "string"},
            "reason": {"type": "string"},
            "evidence_quote": {"type": "string"},
        },
        "required": [
            "character_name",
            "character_id",
            "route_stage",
            "escalation",
            "reason",
            "evidence_quote",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "passed": {"type": "boolean"},
            "issues": {"type": "array", "items": {"type": "string"}},
            "retry_feedback": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "post_turn_update_needed": {"type": "boolean"},
            "npc_agency_issues": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
            "npc_passivity_issues": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
            "npc_knowledge_leaks": {
                "type": "array",
                "items": leak,
                "maxItems": 8,
            },
            "commit_decisions": {
                "type": "array",
                "items": commit_decision,
                "maxItems": 24,
            },
            "dating_route_stage_violations": {
                "type": "array",
                "items": dating_route_stage_violation,
                "maxItems": 8,
            },
        },
        "required": [
            "passed",
            "issues",
            "retry_feedback",
            "confidence",
            "post_turn_update_needed",
            "npc_agency_issues",
            "npc_passivity_issues",
            "npc_knowledge_leaks",
            "commit_decisions",
            "dating_route_stage_violations",
        ],
    }


def _verifier_messages(
    *,
    spec: NarratorMessageSpec,
    request: ChatRequest,
    narrator_body: str,
) -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(
            role="system",
            body=(
                "Verify whether the narrator response follows the message spec. "
                "Use the enforced schema. Flag unsupported additions, missed "
                "must-say beats, tone drift, player-agency violations, and "
                "NPC knowledge leaks. Also flag unearned NPC compliance in "
                "npc_agency_issues: helping, revealing, forgiving, joining, "
                "surrendering, flirting, or changing allegiance without support "
                "from motives, relationship, leverage, pressure, or recent "
                "events. Also flag passive NPC/world handling in "
                "npc_passivity_issues: present NPCs, factions, hazards, clocks, "
                "or environmental pressure merely wait, give space, look to the "
                "player, or fail to take plausible visible initiative from "
                "planned intents, motives, leverage, pressure, or recent events. "
                "Preserve the full spectrum of NPC stances: do not treat a "
                "hostile, self-interested, unfair, or unreasonable response as "
                "an issue merely because it frustrates the player, and do flag "
                "drafts that smooth such a character into helpful cooperation "
                "without support. "
                "Also flag stage-aware dating-route violations in "
                "dating_route_stage_violations when a response exceeds a "
                "provided route stage or max plausible escalation, such as "
                "premature exclusivity, commitment, future-locking, or domestic "
                "escalation. Treat physical and sexual intimacy as "
                "character-specific: flag it only when it conflicts with the "
                "provided route intimacy profile, known boundaries, consent, or "
                "established characterization. Allow proportionate early "
                "progress such as warmth, curiosity, guarded vulnerability, "
                "contact exchange, or scheduling another interaction when the "
                "route state supports it. A knowledge leak exists only when an NPC "
                "uses or reacts to a concrete fact that the provided context "
                "does not establish for that NPC. For planned state commit candidates, "
                "return one commit_decisions item for each candidate_id. Mark whether "
                "each candidate is rendered, contradicted, omitted, unclear, or safe "
                "without narration, and set safe_to_commit only when the accepted "
                "narrator response or the candidate's explicit safe-without-narration "
                "policy makes it safe. Set post_turn_update_needed to false only "
                "when no deterministic or legacy post-turn state/context inference "
                "is needed for this response."
            ),
        ),
        ChatMessage(
            role="user",
            body="\n\n".join(
                (
                    format_narrator_message_spec(spec),
                    "Source request:\n" + rendered_chat_request_text(request),
                    "Narrator response:\n" + narrator_body,
                )
            ),
        ),
    )


def _verification_result_from_data(
    data: dict[str, object],
) -> NarratorVerificationResult:
    npc_agency_issues = _string_tuple(data.get("npc_agency_issues"))
    npc_passivity_issues = _string_tuple(data.get("npc_passivity_issues"))
    dating_route_stage_violations = _dating_route_stage_violations_from_data(
        data.get("dating_route_stage_violations")
    )
    return NarratorVerificationResult(
        passed=bool(data.get("passed"))
        and not npc_agency_issues
        and not npc_passivity_issues
        and not dating_route_stage_violations,
        issues=_string_tuple(data.get("issues")),
        retry_feedback=_string(data.get("retry_feedback")),
        confidence=_float(data.get("confidence")),
        post_turn_update_needed=data.get("post_turn_update_needed") is not False,
        npc_agency_issues=npc_agency_issues,
        npc_passivity_issues=npc_passivity_issues,
        npc_knowledge_leaks=_npc_knowledge_leaks_from_data(
            data.get("npc_knowledge_leaks"),
        ),
        commit_decisions=_commit_decisions_from_data(data.get("commit_decisions")),
        dating_route_stage_violations=dating_route_stage_violations,
    )


def _commit_decisions_from_data(
    value: object,
) -> tuple[NarratorCommitDecision, ...]:
    if not isinstance(value, list):
        return ()
    decisions: list[NarratorCommitDecision] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        candidate_id = _string(item.get("candidate_id"))
        candidate_type = _string(item.get("candidate_type"))
        status = _string(item.get("status"))
        reason = _string(item.get("reason"))
        if not candidate_id or not candidate_type or not status:
            continue
        decisions.append(
            NarratorCommitDecision(
                candidate_id=candidate_id,
                candidate_type=candidate_type,
                status=status,
                safe_to_commit=bool(item.get("safe_to_commit")),
                reason=reason,
                evidence_quote=_string(item.get("evidence_quote")),
            )
        )
    return tuple(decisions)


def _npc_knowledge_leaks_from_data(value: object) -> tuple[NpcKnowledgeLeak, ...]:
    if not isinstance(value, list):
        return ()
    leaks: list[NpcKnowledgeLeak] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        speaker_name = _string(item.get("speaker_name"))
        claim = _string(item.get("claim"))
        reason = _string(item.get("reason"))
        if not speaker_name or not claim or not reason:
            continue
        leaks.append(
            NpcKnowledgeLeak(
                speaker_name=speaker_name,
                claim=claim,
                reason=reason,
                target_type=_string(item.get("target_type")),
                target_id=_string(item.get("target_id")),
            )
        )
    return tuple(leaks)


def _dating_route_stage_violations_from_data(
    value: object,
) -> tuple[DatingRouteStageViolation, ...]:
    if not isinstance(value, list):
        return ()
    violations: list[DatingRouteStageViolation] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        character_name = _string(item.get("character_name"))
        route_stage = _string(item.get("route_stage"))
        escalation = _string(item.get("escalation"))
        reason = _string(item.get("reason"))
        if not character_name or not route_stage or not escalation or not reason:
            continue
        violations.append(
            DatingRouteStageViolation(
                character_name=character_name,
                character_id=_string(item.get("character_id")),
                route_stage=route_stage,
                escalation=escalation,
                reason=reason,
                evidence_quote=_string(item.get("evidence_quote")),
            )
        )
    return tuple(violations)


def _messages_text(messages: tuple[MessageRecord, ...]) -> str:
    if not messages:
        return "Source messages: none"
    return "\n\n".join(
        f"[{message.id}] {message.role}"
        + (f" ({message.speaker_name})" if message.speaker_name else "")
        + f":\n{message.body}"
        for message in messages
    )


def _observations_text(observations: tuple[ContextObservationRecord, ...]) -> str:
    if not observations:
        return "Observations: none"
    lines = ["Observations:"]
    for observation in observations:
        lines.append(
            f"- [{observation.id}] type={observation.observation_type}; "
            f"scope={observation.scope}; confidence={observation.confidence:.2g}; "
            f"claim={observation.claim}; evidence={observation.evidence_quote}; "
            f"sources={', '.join(observation.source_message_ids)}"
        )
    return "\n".join(lines)


def _decision_metadata(decision: CurationDecision) -> dict[str, object]:
    return {
        "action": decision.action,
        "reason": decision.reason,
        "confidence": decision.confidence,
        "memory_body": decision.memory_body,
        "context_title": decision.context_title,
        "context_body": decision.context_body,
        "tags": list(decision.tags),
    }


def _curated_memory_exists(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    body: str,
    source_message_ids: tuple[str, ...],
) -> bool:
    normalized_body = _normalize_memory_body(body)
    normalized_sources = frozenset(source_message_ids)
    for memory in repositories.list_memories(save_id):
        if (
            _normalize_memory_body(memory.body) == normalized_body
            and frozenset(memory.source_message_ids) == normalized_sources
        ):
            return True
    for suggestion in repositories.list_context_update_suggestions(
        save_id,
        status="pending",
    ):
        if suggestion.entity_type != "memory" or suggestion.update_type != "create":
            continue
        proposed_value = suggestion.proposed_value
        if not isinstance(proposed_value, dict):
            continue
        if (
            _normalize_memory_body(proposed_value.get("body")) == normalized_body
            and frozenset(_string_tuple(proposed_value.get("source_message_ids")))
            == normalized_sources
        ):
            return True
    return False


def _curated_memory_proposed_value(
    *,
    body: str,
    tags: tuple[str, ...],
    importance: float,
    source_message_ids: tuple[str, ...],
    source_observation_id: str,
) -> dict[str, object]:
    source_message_id = source_message_ids[0] if source_message_ids else None
    return {
        "body": body.strip(),
        "tags": list(tags),
        "importance": importance,
        "source_message_id": source_message_id,
        "source_message_ids": list(source_message_ids),
        "source_observation_id": source_observation_id,
    }


def _normalize_memory_body(value: object) -> str:
    return " ".join(value.strip().casefold().split()) if isinstance(value, str) else ""


def _string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _float(value: object) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return 0.0
