"""Structured generation of per-character dating-route pacing profiles."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from bragi.persistence.models import (
    CharacterRecord,
    DatingRouteStateRecord,
    ModelPreferenceRecord,
    ScenarioRecord,
)
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatMessage,
    ProviderClient,
    StructuredOutputProvider,
    StructuredOutputRequest,
)
from bragi.services.dating_route_service import DatingRouteService
from bragi.services.model_capabilities import (
    MODEL_LACKS_CAPABILITY_REASON,
    MODEL_MISSING_REASON,
    MODEL_UNAVAILABLE_REASON,
    STRUCTURED_OUTPUT_CAPABILITIES,
    check_model_capabilities,
)
from bragi.services.model_preferences import (
    CHARACTER_INTENT_PLANNING_PURPOSE,
    DATING_ROUTE_PROFILE_PURPOSE,
    roleplay_model_preference_with_fallbacks,
)
from bragi.services.provider_fallbacks import structured_output_with_fallback

DATING_ROUTE_PROFILE_TASK = DATING_ROUTE_PROFILE_PURPOSE


@dataclass(frozen=True)
class DatingRouteProfileResult:
    status: str = "skipped"
    updated_count: int = 0
    requested_count: int = 0
    skipped_reason: str = ""

    def to_json(self) -> dict[str, object]:
        return {
            "status": self.status,
            "updated_count": self.updated_count,
            "requested_count": self.requested_count,
            "skipped_reason": self.skipped_reason,
        }


class DatingRouteProfileService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        providers: dict[str, ProviderClient],
    ) -> None:
        self.repositories = repositories
        self.providers = providers

    async def ensure_profiles_for_save(
        self,
        *,
        save_id: str,
        source_message_id: str | None = None,
    ) -> DatingRouteProfileResult:
        supports_dating_routes = _save_supports_dating_routes(
            self.repositories,
            save_id,
        )
        if supports_dating_routes:
            DatingRouteService(self.repositories).seed_routes_for_save(
                save_id,
                source_message_id=source_message_id,
            )
        routes = tuple(self.repositories.list_dating_route_states(save_id))
        if not routes and not supports_dating_routes:
            return DatingRouteProfileResult(skipped_reason="no_dating_routes")
        routes = tuple(
            route
            for route in routes
            if _route_needs_profile(route)
        )
        if not routes:
            return DatingRouteProfileResult(skipped_reason="no_missing_profiles")
        details = self.repositories.load_save_details(save_id)
        if details is None:
            return DatingRouteProfileResult(skipped_reason="unknown_save")
        preference = _dating_route_profile_model_preference(
            repositories=self.repositories,
            save_id=save_id,
        )
        if preference is None:
            return DatingRouteProfileResult(
                requested_count=len(routes),
                skipped_reason="no_model_preference",
            )
        _provider, skipped_reason = self._structured_provider(preference)
        if skipped_reason is not None:
            return DatingRouteProfileResult(
                requested_count=len(routes),
                skipped_reason=skipped_reason,
            )
        characters = self.repositories.list_characters(save_id)
        request = StructuredOutputRequest(
            provider=preference.provider,
            model_id=preference.model_id,
            schema_name="dating_route_profile",
            schema=_dating_route_profile_schema(routes),
            messages=_dating_route_profile_messages(
                scenario=details.scenario,
                characters=characters,
                routes=routes,
            ),
            temperature=0.2,
            max_output_tokens=max(600, 280 * len(routes)),
        )
        response = await structured_output_with_fallback(
            repositories=self.repositories,
            providers=self.providers,
            request=request,
            task=DATING_ROUTE_PROFILE_TASK,
            save_id=save_id,
        )
        profiles_by_npc_id, skipped_reason = _validated_profiles(
            routes=routes,
            data=response.data,
        )
        if skipped_reason:
            return DatingRouteProfileResult(
                requested_count=len(routes),
                skipped_reason=skipped_reason,
            )
        updated_count = self._apply_profiles(
            save_id=save_id,
            routes=routes,
            profiles_by_npc_id=profiles_by_npc_id,
            source_message_id=source_message_id,
        )
        return DatingRouteProfileResult(
            status="succeeded",
            updated_count=updated_count,
            requested_count=len(routes),
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

    def _apply_profiles(
        self,
        *,
        save_id: str,
        routes: tuple[DatingRouteStateRecord, ...],
        profiles_by_npc_id: Mapping[str, Mapping[str, object]],
        source_message_id: str | None,
    ) -> int:
        updated_count = 0
        for route in routes:
            item = profiles_by_npc_id[route.npc_character_id]
            comfort = route.comfort_with_intimacy or _string(
                item.get("comfort_with_intimacy")
            )
            pacing = route.pacing_preference or _string(
                item.get("pacing_preference")
            )
            known_boundaries = _merge_strings(
                route.known_boundaries,
                _string_list(item.get("known_boundaries")),
            )
            unresolved_questions = _merge_strings(
                route.unresolved_questions,
                _string_list(item.get("unresolved_questions")),
            )
            updated = self.repositories.upsert_dating_route_state(
                save_id=save_id,
                player_character_id=route.player_character_id,
                npc_character_id=route.npc_character_id,
                stage=route.stage,
                comfort_with_intimacy=comfort,
                pacing_preference=pacing,
                known_boundaries=known_boundaries,
                unresolved_questions=unresolved_questions,
                source_message_id=source_message_id or route.source_message_id,
            )
            if updated != route:
                updated_count += 1
        return updated_count


def _route_needs_profile(route: DatingRouteStateRecord) -> bool:
    return (
        not route.comfort_with_intimacy.strip()
        or not route.pacing_preference.strip()
    )


def _save_supports_dating_routes(
    repositories: PersistenceRepositories,
    save_id: str,
) -> bool:
    save = repositories.get_save(save_id)
    if save is None:
        return False
    scenario = repositories.get_scenario(save.scenario_id)
    if scenario is None:
        return False
    if scenario.type == "dating_sim":
        return True
    try:
        content = json.loads(scenario.content_json)
    except json.JSONDecodeError:
        return False
    if not isinstance(content, Mapping):
        return False
    genres = content.get("_scenario_genres")
    return isinstance(genres, list) and "dating_sim" in genres


def _dating_route_profile_model_preference(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
) -> ModelPreferenceRecord | None:
    return roleplay_model_preference_with_fallbacks(
        repositories=repositories,
        save_id=save_id,
        purposes=(
            DATING_ROUTE_PROFILE_TASK,
            CHARACTER_INTENT_PLANNING_PURPOSE,
            "character_action_planning",
            "context_update",
        ),
    )


def _dating_route_profile_schema(
    routes: tuple[DatingRouteStateRecord, ...],
) -> dict[str, Any]:
    npc_ids = [route.npc_character_id for route in routes]
    npc_id_field: dict[str, object] = {"type": "string"}
    if npc_ids:
        npc_id_field["enum"] = npc_ids
    profile_item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "npc_character_id": npc_id_field,
            "comfort_with_intimacy": {"type": "string"},
            "pacing_preference": {"type": "string"},
            "known_boundaries": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
            "unresolved_questions": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 6,
            },
            "reason": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": [
            "npc_character_id",
            "comfort_with_intimacy",
            "pacing_preference",
            "known_boundaries",
            "unresolved_questions",
            "reason",
            "confidence",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "profiles": {
                "type": "array",
                "items": profile_item,
                "minItems": len(routes),
                "maxItems": len(routes),
            },
        },
        "required": ["profiles"],
    }


def _validated_profiles(
    *,
    routes: tuple[DatingRouteStateRecord, ...],
    data: Mapping[str, object],
) -> tuple[dict[str, dict[str, object]], str]:
    if set(data) != {"profiles"}:
        return {}, "incomplete_profile_response"
    requested_npc_ids = {route.npc_character_id for route in routes}
    profiles = data.get("profiles")
    if not isinstance(profiles, list):
        return {}, "incomplete_profile_response"

    profiles_by_npc_id: dict[str, dict[str, object]] = {}
    for item in profiles:
        if not isinstance(item, dict):
            return {}, "incomplete_profile_response"
        if not _profile_item_has_required_shape(item):
            return {}, "incomplete_profile_response"
        npc_id = _string(item.get("npc_character_id"))
        if npc_id not in requested_npc_ids:
            return {}, "out_of_scope_profile_response"
        if npc_id in profiles_by_npc_id:
            return {}, "duplicate_profile_response"
        profiles_by_npc_id[npc_id] = item

    if len(profiles) != len(routes):
        return {}, "incomplete_profile_response"
    if set(profiles_by_npc_id) != requested_npc_ids:
        return {}, "incomplete_profile_response"

    routes_by_npc_id = {route.npc_character_id: route for route in routes}
    for npc_id, item in profiles_by_npc_id.items():
        route = routes_by_npc_id[npc_id]
        comfort = route.comfort_with_intimacy or _string(
            item.get("comfort_with_intimacy")
        )
        pacing = route.pacing_preference or _string(item.get("pacing_preference"))
        if not comfort or not pacing:
            return {}, "incomplete_profile_response"
        if not _generated_profile_text_is_in_scope(item=item):
            return {}, "out_of_scope_profile_response"

    return profiles_by_npc_id, ""


def _profile_item_has_required_shape(item: Mapping[str, object]) -> bool:
    expected_keys = {
        "npc_character_id",
        "comfort_with_intimacy",
        "pacing_preference",
        "known_boundaries",
        "unresolved_questions",
        "reason",
        "confidence",
    }
    return (
        set(item) == expected_keys
        and isinstance(item.get("npc_character_id"), str)
        and bool(_string(item.get("npc_character_id")))
        and isinstance(item.get("comfort_with_intimacy"), str)
        and isinstance(item.get("pacing_preference"), str)
        and _is_string_list_value(item.get("known_boundaries"), max_items=8)
        and _is_string_list_value(item.get("unresolved_questions"), max_items=6)
        and isinstance(item.get("reason"), str)
        and _is_number_value(item.get("confidence"))
    )


def _is_string_list_value(value: object, *, max_items: int) -> bool:
    return isinstance(value, list) and len(value) <= max_items and all(
        isinstance(item, str) for item in value
    )


def _is_number_value(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


_COMMITMENT_ESCALATION_TERMS = (
    "exclusive",
    "exclusivity",
    "monogamy",
    "monogamous",
    "commitment",
    "committed relationship",
    "serious relationship",
    "long term relationship",
    "long term commitment",
    "official relationship",
    "relationship official",
    "officially together",
    "make it official",
    "make things official",
    "make the relationship official",
    "define the relationship",
    "relationship label",
    "relationship labels",
    "label the relationship",
    "label things",
    "relationship status",
    "partner status",
    "girlfriend",
    "boyfriend",
    "romantic partner",
    "life partner",
    "partner",
    "couple",
    "go steady",
    "engaged",
    "engagement",
    "marriage",
    "married",
    "marry",
    "wife",
    "husband",
    "spouse",
    "wedding",
    "fiance",
    "fiancee",
    "bride",
    "groom",
    "move in",
    "moving in",
    "live together",
    "living together",
    "cohabit",
    "cohabitation",
    "share an apartment",
    "share apartment",
    "shared apartment",
    "apartment together",
    "share a home",
    "shared home",
    "home together",
    "share a place",
    "place together",
    "house together",
    "same roof",
    "roommate",
    "roommates",
    "joint lease",
    "sign a lease",
    "lease together",
    "domestic partnership",
    "domestic life",
    "domestic planning",
    "major life plan",
    "major life plans",
    "start a family",
    "family planning",
    "have a family",
    "having a family",
    "baby",
    "babies",
    "children",
    "child",
    "kids",
    "pregnancy",
    "pregnant",
    "parenthood",
    "be parents",
    "become parents",
)
_COMMITMENT_ESCALATION_INTENT_TERMS = (
    "want",
    "wants",
    "wanted",
    "expect",
    "expects",
    "expected",
    "seek",
    "seeks",
    "seeking",
    "pursue",
    "pursues",
    "pursuing",
    "ask",
    "asks",
    "asking",
    "push",
    "pushes",
    "pushing",
    "prefer",
    "prefers",
    "preferred",
    "desire",
    "desires",
    "desired",
    "need",
    "needs",
    "needed",
    "require",
    "requires",
    "required",
    "looking for",
    "hope",
    "hopes",
    "hoping",
    "plan",
    "plans",
    "planning",
    "ready",
)
_NEAR_TERM_ESCALATION_TIMING_TERMS = (
    "first date",
    "tonight",
    "this date",
    "this turn",
    "right away",
    "immediate",
    "immediately",
    "now",
    "early",
    "soon",
)
_COMMITMENT_TIMELINE_MARKER_TERMS = (
    "timeline",
    "timetable",
    "schedule",
)
_COMMITMENT_TIMELINE_LINK_TERMS = (
    "after",
    "before",
    "by",
    "within",
)
_COMMITMENT_TIMELINE_CONTEXT_TERMS = (
    "date",
    "dates",
    "day",
    "days",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
    "winter",
    "spring",
    "summer",
    "fall",
    "autumn",
    "graduation",
    "semester",
    "college",
    "school year",
    "holiday",
    "birthday",
    "anniversary",
)
_PHYSICAL_INTIMACY_TERMS = (
    "physical intimacy",
    "sexual intimacy",
    "intimacy",
    "sex",
    "sexual",
    "kiss",
    "kissing",
    "touch",
    "touching",
    "affection",
    "make out",
    "sleep together",
    "sleep with",
)
_PHYSICAL_BOUNDARY_MARKERS = (
    "only",
    "not",
    "no ",
    "never",
    "until",
    "after",
    "before",
    "requires",
    "require",
    "wait",
    "without",
)


def _generated_profile_text_is_in_scope(
    *,
    item: Mapping[str, object],
) -> bool:
    texts = [
        _string(item.get("comfort_with_intimacy")),
        _string(item.get("pacing_preference")),
        _string(item.get("reason")),
    ]
    texts.extend(_string_list(item.get("known_boundaries")))
    texts.extend(_string_list(item.get("unresolved_questions")))
    return not any(
        _mentions_character_specific_commitment_escalation(text)
        for text in texts
    )


def _mentions_character_specific_commitment_escalation(text: str) -> bool:
    normalized = _normalize_scope_text(text)
    if not normalized:
        return False
    has_commitment_term = _contains_scope_term(
        normalized,
        _COMMITMENT_ESCALATION_TERMS,
    )
    if not has_commitment_term:
        return False
    has_intent_term = _contains_scope_term(
        normalized,
        _COMMITMENT_ESCALATION_INTENT_TERMS,
    )
    has_near_term_timing = _contains_scope_term(
        normalized,
        _NEAR_TERM_ESCALATION_TIMING_TERMS,
    )
    has_commitment_timeline = _mentions_commitment_timeline(normalized)
    if _mentions_physical_intimacy_boundary(normalized):
        if has_intent_term and not _only_requires_physical_boundary(normalized):
            return True
        if has_near_term_timing or has_commitment_timeline:
            return True
        return False
    if has_intent_term or has_near_term_timing or has_commitment_timeline:
        return True
    return True


def _normalize_scope_text(text: str) -> str:
    characters = []
    for character in unicodedata.normalize("NFKD", text.casefold()):
        if unicodedata.combining(character):
            continue
        characters.append(character if character.isalnum() else " ")
    return " ".join("".join(characters).split())


def _contains_scope_term(normalized: str, terms: tuple[str, ...]) -> bool:
    padded = f" {normalized} "
    return any(f" {_normalize_scope_text(term)} " in padded for term in terms)


def _mentions_commitment_timeline(normalized: str) -> bool:
    if _contains_scope_term(
        normalized,
        _COMMITMENT_TIMELINE_MARKER_TERMS,
    ):
        return True
    for link in _COMMITMENT_TIMELINE_LINK_TERMS:
        index = normalized.find(f"{link} ")
        if index == -1:
            continue
        window = normalized[index : index + 80]
        if _contains_scope_term(window, _COMMITMENT_TIMELINE_CONTEXT_TERMS):
            return True
    return False


def _mentions_physical_intimacy_boundary(normalized: str) -> bool:
    return _contains_scope_term(
        normalized,
        _PHYSICAL_INTIMACY_TERMS,
    ) and _contains_scope_term(normalized, _PHYSICAL_BOUNDARY_MARKERS)


def _only_requires_physical_boundary(normalized: str) -> bool:
    pressure_terms = (
        "want",
        "wants",
        "wanted",
        "desire",
        "desires",
        "desired",
        "hope",
        "hopes",
        "hoping",
        "plan",
        "plans",
        "planning",
        "expect",
        "expects",
        "expected",
        "looking for",
        "ask",
        "asks",
        "asking",
        "push",
        "pushes",
        "pushing",
        "prefer",
        "prefers",
        "preferred",
    )
    return not _contains_scope_term(normalized, pressure_terms)


def _dating_route_profile_messages(
    *,
    scenario: ScenarioRecord,
    characters: list[CharacterRecord],
    routes: tuple[DatingRouteStateRecord, ...],
) -> tuple[ChatMessage, ...]:
    characters_by_id = {character.id: character for character in characters}
    lines = [
        f"Scenario title: {scenario.title}",
        f"Scenario type: {scenario.type}",
        f"Premise: {scenario.premise}",
        f"Player role: {scenario.player_role}",
        *_scenario_content_lines(scenario),
        "Dating routes to profile:",
    ]
    for route in routes:
        player = characters_by_id.get(route.player_character_id)
        npc = characters_by_id.get(route.npc_character_id)
        if npc is None:
            continue
        lines.extend(
            (
                f"- npc_character_id: {npc.id}",
                f"  npc name: {npc.name}",
                f"  player name: {player.name if player is not None else 'unknown'}",
                f"  route stage: {route.stage}",
                f"  completed interactions: {route.completed_interactions}",
                f"  dates completed: {route.dates_completed}",
                f"  next reasonable step: {route.next_reasonable_step}",
                f"  existing comfort with intimacy: {route.comfort_with_intimacy}",
                f"  existing pacing preference: {route.pacing_preference}",
                "  existing known boundaries: "
                + "; ".join(route.known_boundaries),
                f"  role: {npc.role}",
                f"  age: {npc.age}",
                f"  personality: {npc.personality}",
                f"  goals: {npc.goals}",
                f"  motivations: {npc.motivations}",
                f"  current intent: {npc.current_intent}",
                f"  boundaries: {npc.boundaries}",
                f"  attitude toward player: {npc.attitude_toward_player}",
                f"  cooperation conditions: {npc.cooperation_conditions}",
                f"  status: {npc.status}",
                f"  relationships: {npc.relationships}",
            )
        )
    return (
        ChatMessage(
            role="system",
            body=(
                "Generate durable per-character dating-route pacing profiles using "
                "the enforced schema. Focus only on non-commitment relationship "
                "boundaries: physical intimacy comfort, pacing preferences, and "
                "open questions. Do not create a universal number-of-dates rule. "
                "Do not loosen or rewrite existing explicit fields. Do not make "
                "character-specific desires, timelines, or requests for "
                "commitment, exclusivity, living together, marriage, or major "
                "life plans; those remain stage-gated by deterministic route "
                "policy. If evidence supports a physical-intimacy boundary that "
                "references a later commitment milestone, phrase it only as an "
                "intimacy boundary, not as pressure to escalate that commitment. "
                "If evidence is weak, write that the preference is not "
                "established rather than inventing detail."
            ),
        ),
        ChatMessage(role="user", body="\n".join(line for line in lines if line)),
    )


def _scenario_content_lines(scenario: ScenarioRecord) -> tuple[str, ...]:
    try:
        content = json.loads(scenario.content_json)
    except json.JSONDecodeError:
        return ()
    if not isinstance(content, Mapping):
        return ()
    keys = (
        "player_character_name",
        "player_character_profile",
        "current_scene",
        "tone_genre",
    )
    lines: list[str] = []
    for key in keys:
        value = _string(content.get(key))
        if value:
            lines.append(f"{key}: {value}")
    return tuple(lines)


def _string(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    values: list[str] = []
    for item in value:
        text = _string(item)
        if text:
            values.append(text)
    return values


def _merge_strings(existing: list[str], generated: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in (*existing, *generated):
        stripped = value.strip()
        key = stripped.casefold()
        if not stripped or key in seen:
            continue
        seen.add(key)
        merged.append(stripped)
    return merged
