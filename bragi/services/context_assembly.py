"""Deterministic context assembly and metadata-only diagnostics."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import cast

from bragi.persistence.models import (
    ActiveThreadRecord,
    CharacterKnowledgeEdgeRecord,
    CharacterRecord,
    ContextUpdateSuggestionRecord,
    DatingRouteStateRecord,
    EntityLinkRecord,
    LocationRecord,
    MemoryRecord,
    MessageRecord,
    MessageVisibilityRecord,
    SaveDetailsRecord,
    ScenarioRecord,
    SceneSnapshotRecord,
    SummaryRecord,
    WorldStateRecord,
)
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.action_choice_flags import scenario_action_choices_enabled
from bragi.services.active_thread_lifecycle import (
    active_thread_is_prompt_visible,
    normalize_active_thread_status,
    normalize_active_thread_visibility,
)
from bragi.services.dating_route_policy import (
    escalation_policy_for_stage,
    intimacy_profile_guidance,
)
from bragi.services.knowledge_boundary import (
    character_scope_for_turn,
    knowledge_edge_has_character_text_source,
    message_visible_to_present_characters,
)
from bragi.services.mention_matching import character_name_is_mentioned
from bragi.services.open_threads import is_open_threads_aggregate_key
from bragi.services.summary_safety import validate_summary_output
from bragi.world_time_model import format_world_time_from_snapshot

CONTEXT_BUDGET_MODE_DIAGNOSTICS_ONLY = "diagnostics_only"
CONTEXT_BUDGET_MODE_FIXED_CHARS = "fixed_chars"
CONTEXT_BUDGET_MODE_ADAPTIVE_TIERS = "adaptive_tiers"
CONTEXT_BUDGET_MODES = frozenset(
    {
        CONTEXT_BUDGET_MODE_DIAGNOSTICS_ONLY,
        CONTEXT_BUDGET_MODE_FIXED_CHARS,
        CONTEXT_BUDGET_MODE_ADAPTIVE_TIERS,
    }
)
DEFAULT_CONTEXT_BUDGET_MODE = CONTEXT_BUDGET_MODE_DIAGNOSTICS_ONLY
DEFAULT_CONTEXT_BUDGET_FIXED_TOTAL_CHARS = 24_000
DEFAULT_CONTEXT_BUDGET_ADAPTIVE_FRACTION = 0.35
CHARACTER_VISUAL_DETAIL_MAX_CHARS = 320
PENDING_CONTEXT_SUGGESTION_LIMIT = 6
PENDING_CONTEXT_VALUE_MAX_CHARS = 240
PENDING_CONTEXT_SUGGESTION_MAX_AGE_HOURS = 12
PENDING_CONTEXT_SUGGESTION_MIN_CONFIDENCE = 0.5
PRE_TURN_SCENE_HINT_LIMIT = 8
SCOPED_MAY_KNOW_CONFIDENCE_THRESHOLD = 0.7

SCENARIO_CORE_CONTENT_KEYS = frozenset(
    (
        "title",
        "premise",
        "setup_line",
        "player_character_name",
        "player_role",
        "tone_genre",
        "starting_scene",
        "current_scene",
        "relationship_seed",
        "case_facts",
        "case_status",
    )
)
_MISSING = object()


@dataclass(frozen=True)
class ContextBudgetSettings:
    mode: str = DEFAULT_CONTEXT_BUDGET_MODE
    fixed_total_chars: int = DEFAULT_CONTEXT_BUDGET_FIXED_TOTAL_CHARS
    adaptive_fraction: float = DEFAULT_CONTEXT_BUDGET_ADAPTIVE_FRACTION

    @classmethod
    def defaults(cls) -> ContextBudgetSettings:
        return cls()


@dataclass(frozen=True)
class ContextSource:
    tier: str
    source_type: str
    source_id: str
    text: str
    reason: str = ""
    always_include: bool = False
    relevance_query: str = ""
    trimmable: bool = False


@dataclass(frozen=True)
class _PendingContextSuggestionGroup:
    update_type: str
    entity_type: str
    entity_id: str | None
    field_path: str
    proposed_value: object
    suggestion_ids: tuple[str, ...]
    confidence: float
    created_at: str


@dataclass(frozen=True)
class ContextSourceBreakdown:
    tier: str
    source_type: str
    source_id: str
    char_count: int
    included: bool
    reason: str


@dataclass(frozen=True)
class ContextAssemblyBreakdown:
    budget_mode: str
    budget_limit_chars: int | None
    total_chars: int
    included_chars: int
    sources: tuple[ContextSourceBreakdown, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "budget_mode": self.budget_mode,
            "budget_limit_chars": self.budget_limit_chars,
            "total_chars": self.total_chars,
            "included_chars": self.included_chars,
            "sources": [
                {
                    "tier": source.tier,
                    "source_type": source.source_type,
                    "source_id": source.source_id,
                    "char_count": source.char_count,
                    "included": source.included,
                    "reason": source.reason,
                }
                for source in self.sources
            ],
        }


@dataclass(frozen=True)
class ContextAssemblyResult:
    scenario_instructions: str
    current_scene_context: tuple[str, ...]
    breakdown: ContextAssemblyBreakdown


class ContextAssemblyService:
    def __init__(self, repositories: PersistenceRepositories) -> None:
        self.repositories = repositories

    def assemble_narrator_context(self, save_id: str) -> ContextAssemblyResult:
        details = self.repositories.load_save_details(save_id)
        if details is None:
            raise ValueError(f"Unknown save id: {save_id}")
        sources = (
            *deterministic_context_sources(
                repositories=self.repositories,
                save_id=save_id,
            ),
            *pending_context_suggestion_sources(
                repositories=self.repositories,
                save_id=save_id,
            ),
        )
        selected_sources, breakdown = apply_context_budget(
            sources,
            settings=context_budget_settings(self.repositories, save_id=save_id),
        )
        return ContextAssemblyResult(
            scenario_instructions=compact_scenario_instructions(details.scenario),
            current_scene_context=tuple(source.text for source in selected_sources),
            breakdown=breakdown,
        )

    def build_image_scene_context(
        self,
        *,
        save_id: str,
        source_message_id: str | None = None,
        selected_scenario_sections: tuple[str, ...] = (),
    ) -> tuple[str, ContextAssemblyBreakdown]:
        details = self.repositories.load_save_details(save_id)
        if details is None:
            raise ValueError(f"Unknown save id: {save_id}")
        source_message, context_messages = _image_context_messages(
            messages=details.messages,
            source_message_id=source_message_id,
        )
        sources = (
            ContextSource(
                tier="rules",
                source_type="scenario",
                source_id=details.scenario.id,
                text="Generate a scene image for this roleplay moment.",
                reason="image task",
            ),
            ContextSource(
                tier="scenario_header",
                source_type="scenario",
                source_id=details.scenario.id,
                text="\n".join(
                    part
                    for part in (
                        details.scenario.title,
                        details.scenario.premise,
                        details.scenario.player_role,
                    )
                    if part
                ),
                reason="image scenario header",
            ),
            *(
                ContextSource(
                    tier="scenario_section",
                    source_type="scenario_section",
                    source_id=f"{details.scenario.id}:{index}",
                    text=section,
                    reason="selected scenario section",
                )
                for index, section in enumerate(selected_scenario_sections)
            ),
            *deterministic_context_sources(
                repositories=self.repositories,
                save_id=save_id,
                mode="image",
                source_message_id=source_message_id,
            ),
            *_prior_image_continuity_sources(
                repositories=self.repositories,
                save_id=save_id,
                messages=details.messages,
                source_message_id=source_message_id,
            ),
            *_message_sources(
                context_messages,
                tier=(
                    "chronicle_before_selected"
                    if source_message is not None
                    else "recent_chronicle"
                ),
            ),
            *(
                (
                    ContextSource(
                        tier="selected_message",
                        source_type="message",
                        source_id=source_message.id,
                        text="Selected scene message:\n"
                        + _format_message(source_message),
                        reason="image source message",
                        always_include=True,
                    ),
                )
                if source_message is not None
                else ()
            ),
        )
        selected_sources, breakdown = apply_context_budget(
            sources,
            settings=context_budget_settings(self.repositories, save_id=save_id),
        )
        return (
            "\n".join(source.text for source in selected_sources if source.text),
            breakdown,
        )


def context_budget_settings(
    repositories: PersistenceRepositories,
    *,
    save_id: str | None = None,
) -> ContextBudgetSettings:
    mode_value = repositories.get_effective_setting(
        "context_budget_mode",
        save_id=save_id,
    )
    mode = mode_value if isinstance(mode_value, str) else DEFAULT_CONTEXT_BUDGET_MODE
    if mode not in CONTEXT_BUDGET_MODES:
        mode = DEFAULT_CONTEXT_BUDGET_MODE
    fixed = _positive_int_setting(
        repositories.get_effective_setting(
            "context_budget_fixed_total_chars",
            save_id=save_id,
        ),
        DEFAULT_CONTEXT_BUDGET_FIXED_TOTAL_CHARS,
    )
    adaptive = _fraction_setting(
        repositories.get_effective_setting(
            "context_budget_adaptive_fraction",
            save_id=save_id,
        ),
        DEFAULT_CONTEXT_BUDGET_ADAPTIVE_FRACTION,
    )
    return ContextBudgetSettings(
        mode=mode,
        fixed_total_chars=fixed,
        adaptive_fraction=adaptive,
    )


def apply_context_budget(
    sources: tuple[ContextSource, ...],
    *,
    settings: ContextBudgetSettings,
) -> tuple[tuple[ContextSource, ...], ContextAssemblyBreakdown]:
    limit = _budget_limit(settings)
    included: list[ContextSource] = []
    breakdown: list[ContextSourceBreakdown] = []
    included_chars = 0
    total_chars = sum(len(source.text) for source in sources)
    for source in sources:
        delivered_source = source
        char_count = len(delivered_source.text)
        include = (
            delivered_source.always_include
            or limit is None
            or included_chars + char_count <= limit
        )
        trimmed = False
        if (
            not include
            and limit is not None
            and delivered_source.trimmable
        ):
            delivered_source = _trim_context_source_to_budget(
                delivered_source,
                limit=max(0, limit - included_chars),
            )
            char_count = len(delivered_source.text)
            include = bool(delivered_source.text)
            trimmed = include
        if include:
            included.append(delivered_source)
            included_chars += char_count
        breakdown.append(
            ContextSourceBreakdown(
                tier=delivered_source.tier,
                source_type=delivered_source.source_type,
                source_id=delivered_source.source_id,
                char_count=char_count,
                included=include,
                reason=(
                    "budget_trimmed"
                    if trimmed
                    else (
                        (delivered_source.reason or "included")
                        if include
                        else "budget_skipped"
                    )
                ),
            )
        )
    return (
        tuple(included),
        ContextAssemblyBreakdown(
            budget_mode=settings.mode,
            budget_limit_chars=limit,
            total_chars=total_chars,
            included_chars=included_chars,
            sources=tuple(breakdown),
        ),
    )


def _trim_context_source_to_budget(
    source: ContextSource,
    *,
    limit: int,
) -> ContextSource:
    if limit <= 0:
        return replace(source, text="")
    text = source.text
    if len(text) <= limit:
        return source
    marker = _context_source_marker(text)
    if marker and limit <= len(marker) + 4:
        return replace(source, text="")
    body = text[len(marker) :].lstrip() if marker else text
    available = limit - len(marker)
    if marker:
        available -= 1
    if available < 8:
        return replace(source, text="")
    excerpt = _relevance_centered_excerpt(
        body,
        query=source.relevance_query,
        limit=available,
    )
    trimmed_text = f"{marker} {excerpt}" if marker else excerpt
    return replace(source, text=trimmed_text)


def _context_source_marker(text: str) -> str:
    stripped = text.lstrip()
    if not stripped.startswith("[") or "]" not in stripped:
        return ""
    marker_end = stripped.index("]") + 1
    return stripped[:marker_end]


def _relevance_centered_excerpt(
    text: str,
    *,
    query: str,
    limit: int,
) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    if limit <= 3:
        return "." * limit
    lowered = compact.casefold()
    positions = [
        position
        for term in _context_relevance_terms(query)
        if (position := lowered.find(term)) >= 0
    ]
    anchor = (
        sum(positions) // len(positions)
        if positions
        else len(compact) // 2
    )
    leading_marker = "..."
    trailing_marker = "..."
    content_limit = limit - len(leading_marker) - len(trailing_marker)
    if content_limit <= 0:
        return compact[:limit]
    start = max(0, min(anchor - (content_limit // 2), len(compact) - content_limit))
    end = min(len(compact), start + content_limit)
    if start > 0:
        next_space = compact.find(" ", start)
        if 0 <= next_space < end:
            start = next_space + 1
    if end < len(compact):
        previous_space = compact.rfind(" ", start, end)
        if previous_space > start:
            end = previous_space
    excerpt = compact[start:end].strip()
    return (
        (leading_marker if start > 0 else "")
        + excerpt
        + (trailing_marker if end < len(compact) else "")
    )[:limit]


def _context_relevance_terms(query: str) -> tuple[str, ...]:
    return tuple(
        term
        for term in re.findall(r"[a-z0-9']{3,}", query.casefold())
        if term
        not in {
            "and",
            "for",
            "the",
            "that",
            "this",
            "with",
        }
    )


def compact_scenario_instructions(
    scenario: ScenarioRecord,
    *,
    include_setup: bool = True,
) -> str:
    dating_sim_identity_parts = _dating_sim_identity_lines(
        scenario,
        include_setup=include_setup,
    )
    fantasy_identity_parts = _fantasy_identity_lines(
        scenario,
        include_setup=include_setup,
    )
    science_fiction_identity_parts = _science_fiction_identity_lines(
        scenario,
        include_setup=include_setup,
    )
    first_contact_identity_parts = _first_contact_identity_lines(
        scenario,
        include_setup=include_setup,
    )
    survival_expedition_identity_parts = _survival_expedition_identity_lines(
        scenario,
        include_setup=include_setup,
    )
    time_loop_identity_parts = _time_loop_identity_lines(
        scenario,
        include_setup=include_setup,
    )
    mystery_identity_parts = _investigation_mystery_identity_lines(
        scenario,
        include_setup=include_setup,
    )
    heist_identity_parts = _heist_infiltration_identity_lines(
        scenario,
        include_setup=include_setup,
    )
    intrigue_identity_parts = _political_intrigue_identity_lines(
        scenario,
        include_setup=include_setup,
    )
    settlement_identity_parts = _settlement_builder_identity_lines(
        scenario,
        include_setup=include_setup,
    )
    hunt_identity_parts = _monster_hunt_bounty_identity_lines(
        scenario,
        include_setup=include_setup,
    )
    journey_identity_parts = _road_trip_pilgrimage_identity_lines(
        scenario,
        include_setup=include_setup,
    )
    trade_identity_parts = _merchant_trade_route_identity_lines(
        scenario,
        include_setup=include_setup,
    )
    cyoa_identity_parts = _choose_your_own_adventure_identity_lines(scenario)
    full_setup_parts = (
        f"Premise/setup: {scenario.premise}",
        _player_character_name_line(scenario),
        *fantasy_identity_parts,
        *science_fiction_identity_parts,
        *first_contact_identity_parts,
        *survival_expedition_identity_parts,
        *time_loop_identity_parts,
        *mystery_identity_parts,
        *heist_identity_parts,
        *intrigue_identity_parts,
        *settlement_identity_parts,
        *hunt_identity_parts,
        *journey_identity_parts,
        *trade_identity_parts,
        *dating_sim_identity_parts,
        *cyoa_identity_parts,
        _scenario_content_line(scenario, "tone_genre", "Tone/style"),
        _scenario_content_line(scenario, "current_scene", "Current scene"),
        f"Player role: {scenario.player_role}",
    )
    lean_identity_parts = (
        _player_character_name_line(scenario),
        _scenario_content_line(scenario, "current_scene", "Current scene"),
        *fantasy_identity_parts,
        *science_fiction_identity_parts,
        *first_contact_identity_parts,
        *survival_expedition_identity_parts,
        *time_loop_identity_parts,
        *mystery_identity_parts,
        *heist_identity_parts,
        *intrigue_identity_parts,
        *settlement_identity_parts,
        *hunt_identity_parts,
        *journey_identity_parts,
        *trade_identity_parts,
        *dating_sim_identity_parts,
        *cyoa_identity_parts,
    )
    header = (
        (
            "Scenario header. Treat this as identity and setup only; current "
            "scene context, selected retrieval, and chronicle messages are "
            "authoritative when they diverge."
        )
        if include_setup
        else (
            "Scenario header. Treat this as durable identity only; initial "
            "setup is omitted because the opening is no longer in the recent "
            "chronicle window. Current scene context, selected retrieval, and "
            "chronicle messages are authoritative when they diverge."
        )
    )
    return "\n".join(
        part
        for part in (
            header,
            (
                "Narrator control rule: never write dialogue, actions, thoughts, "
                "intentions, or choices for the player character. Only the user's "
                "submitted player messages define what the player character says "
                "or does. Preserve player agency by leaving those choices "
                "unresolved, not by making the world passive. NPCs, factions, "
                "hazards, clocks, and environments should keep pursuing goals; "
                "they may interrupt, demand, refuse, leave, escalate, advance "
                "consequences, reveal visible changes, or create pressure when "
                "consistent with established context. Do not advance time in "
                "ways that make the player character act, such as texting, "
                "sleeping, traveling, arriving, or carrying out a plan, unless "
                "the player explicitly submitted that action or gave permission "
                "to skip ahead. Treat stated intent, future-tense plans, "
                "NPC-provided directions, and in-progress movement as not enough "
                "to complete the player's travel, arrival, entry, knock, touch, "
                "or other next action."
            ),
            (
                "Choose-your-own-adventure control rule: write book-like "
                "narration that creates a concrete changed situation suitable "
                "for player action choices. Do not include numbered options, "
                "bullet options, or action-choice lists in narrator prose; "
                "Bragi generates those separately."
                if scenario_action_choices_enabled(scenario)
                else ""
            ),
            f"Title: {scenario.title}",
            *(full_setup_parts if include_setup else lean_identity_parts),
        )
        if part
    )


def _player_character_name_line(scenario: ScenarioRecord) -> str:
    value = _scenario_content_text(scenario, "player_character_name")
    if not value:
        return ""
    return f"Player character name: {value}"


def _dating_sim_identity_lines(
    scenario: ScenarioRecord,
    *,
    include_setup: bool = True,
) -> tuple[str, ...]:
    if scenario.type != "dating_sim":
        return ()
    if not include_setup:
        return (
            _scenario_content_line(
                scenario,
                "player_character_profile",
                "Player profile",
            ),
        )
    return tuple(
        line
        for line in (
            _scenario_content_line(
                scenario,
                "player_character_profile",
                "Player profile",
            ),
        )
        if line
    )


def _fantasy_identity_lines(
    scenario: ScenarioRecord,
    *,
    include_setup: bool = True,
) -> tuple[str, ...]:
    if scenario.type != "fantasy_roleplay":
        return ()
    if not include_setup:
        return tuple(
            line
            for line in (
                _scenario_content_line(scenario, "magic_system", "Magic constraints"),
            )
            if line
        )
    return tuple(
        line
        for line in (
            _scenario_content_line(scenario, "magic_system", "Magic constraints"),
            _scenario_content_line(scenario, "realms_and_places", "Realms/places"),
            _scenario_content_line(
                scenario,
                "factions_and_orders",
                "Factions/orders",
            ),
            _scenario_content_line(
                scenario,
                "myths_and_creatures",
                "Myths/creatures",
            ),
            _scenario_content_line(scenario, "quest_stakes", "Quest stakes"),
        )
        if line
    )


def _science_fiction_identity_lines(
    scenario: ScenarioRecord,
    *,
    include_setup: bool = True,
) -> tuple[str, ...]:
    if scenario.type != "science_fiction_roleplay":
        return ()
    lean_lines = (
        _scenario_content_line(scenario, "technology_level", "Technology constraints"),
        _scenario_content_line(scenario, "setting_scope", "Setting scope"),
    )
    if not include_setup:
        return tuple(line for line in lean_lines if line)
    return tuple(
        line
        for line in (
            *lean_lines,
            _scenario_content_line(
                scenario,
                "species_and_intelligences",
                "Species/intelligences",
            ),
            _scenario_content_line(
                scenario,
                "factions_and_institutions",
                "Factions/institutions",
            ),
            _scenario_content_line(scenario, "mission_stakes", "Mission stakes"),
        )
        if line
    )


def _first_contact_identity_lines(
    scenario: ScenarioRecord,
    *,
    include_setup: bool = True,
) -> tuple[str, ...]:
    if scenario.type != "first_contact_exploration":
        return ()
    lean_lines = (
        _scenario_content_line(scenario, "mission_profile", "Mission"),
        _scenario_content_line(scenario, "ship_or_base_status", "Ship/base status"),
        _scenario_content_line(scenario, "knowledge_state", "Knowledge state"),
        _scenario_content_line(
            scenario,
            "translation_progress",
            "Translation progress",
        ),
    )
    if not include_setup:
        return tuple(line for line in lean_lines if line)
    return tuple(
        line
        for line in (
            *lean_lines,
            _scenario_content_line(
                scenario,
                "exploration_target",
                "Exploration target",
            ),
            _scenario_content_line(
                scenario,
                "unknown_intelligence",
                "Unknown intelligence",
            ),
            _scenario_content_line(
                scenario,
                "discoveries_and_samples",
                "Discoveries/samples",
            ),
            _scenario_content_line(
                scenario,
                "hazards_and_escalation",
                "Hazards/escalation",
            ),
        )
        if line
    )


def _investigation_mystery_identity_lines(
    scenario: ScenarioRecord,
    *,
    include_setup: bool = True,
) -> tuple[str, ...]:
    if scenario.type != "investigation_mystery":
        return ()
    if not include_setup:
        return tuple(
            line
            for line in (
                _scenario_content_line(scenario, "case_status", "Case status"),
            )
            if line
        )
    return tuple(
        line
        for line in (
            _scenario_content_line(scenario, "case_facts", "Case facts"),
            _scenario_content_line(scenario, "case_status", "Case status"),
        )
        if line
    )


def _heist_infiltration_identity_lines(
    scenario: ScenarioRecord,
    *,
    include_setup: bool = True,
) -> tuple[str, ...]:
    if scenario.type != "heist_infiltration":
        return ()
    lean_lines = (
        _scenario_content_line(scenario, "objectives_and_stakes", "Objectives/stakes"),
        _scenario_content_line(scenario, "security_model", "Security model"),
        _scenario_content_line(scenario, "alert_and_heat", "Alert/heat"),
        _scenario_content_line(scenario, "extraction_routes", "Extraction routes"),
    )
    if not include_setup:
        return tuple(line for line in lean_lines if line)
    return tuple(
        line
        for line in (
            _scenario_content_line(scenario, "target_location", "Target location"),
            *lean_lines,
            _scenario_content_line(scenario, "intel_and_access", "Intel/access"),
            _scenario_content_line(scenario, "loadout_and_tools", "Loadout/tools"),
            _scenario_content_line(scenario, "complications", "Complications"),
            _scenario_content_line(scenario, "aftermath", "Aftermath"),
        )
        if line
    )


def _political_intrigue_identity_lines(
    scenario: ScenarioRecord,
    *,
    include_setup: bool = True,
) -> tuple[str, ...]:
    if scenario.type != "political_intrigue":
        return ()
    lean_lines = (
        _scenario_content_line(scenario, "central_conflict", "Central conflict"),
        _scenario_content_line(
            scenario,
            "reputation_and_standing",
            "Reputation/standing",
        ),
        _scenario_content_line(
            scenario,
            "obligations_and_favors",
            "Obligations/favors",
        ),
        _scenario_content_line(
            scenario,
            "political_pressure",
            "Political pressure",
        ),
        _scenario_content_line(
            scenario,
            "public_private_knowledge",
            "Public/private knowledge",
        ),
    )
    if not include_setup:
        return tuple(line for line in lean_lines if line)
    return tuple(
        line
        for line in (
            _scenario_content_line(scenario, "political_arena", "Political arena"),
            *lean_lines,
            _scenario_content_line(
                scenario,
                "political_factions",
                "Political factions",
            ),
            _scenario_content_line(
                scenario,
                "secrets_and_leverage",
                "Secrets/leverage",
            ),
            _scenario_content_line(
                scenario,
                "alliances_and_rivalries",
                "Alliances/rivalries",
            ),
            _scenario_content_line(
                scenario,
                "event_calendar",
                "Event calendar",
            ),
        )
        if line
    )


def _settlement_builder_identity_lines(
    scenario: ScenarioRecord,
    *,
    include_setup: bool = True,
) -> tuple[str, ...]:
    if scenario.type != "settlement_builder":
        return ()
    lean_lines = (
        _scenario_content_line(scenario, "resources_and_indicators", "Resources"),
        _scenario_content_line(scenario, "projects_and_facilities", "Projects"),
        _scenario_content_line(
            scenario,
            "threats_and_opportunities",
            "Threats/opportunities",
        ),
        _scenario_content_line(
            scenario,
            "calendar_and_deadlines",
            "Calendar/deadlines",
        ),
    )
    if not include_setup:
        return tuple(line for line in lean_lines if line)
    return tuple(
        line
        for line in (
            _scenario_content_line(
                scenario,
                "settlement_profile",
                "Settlement profile",
            ),
            *lean_lines,
        )
        if line
    )


def _monster_hunt_bounty_identity_lines(
    scenario: ScenarioRecord,
    *,
    include_setup: bool = True,
) -> tuple[str, ...]:
    if scenario.type != "monster_hunt_bounty":
        return ()
    lean_lines = (
        _scenario_content_line(scenario, "target_profile", "Target profile"),
        _scenario_content_line(scenario, "leads_and_clues", "Leads/clues"),
        _scenario_content_line(scenario, "preparation_state", "Preparation"),
        _scenario_content_line(scenario, "hunt_status", "Hunt status"),
    )
    if not include_setup:
        return tuple(line for line in lean_lines if line)
    return tuple(
        line
        for line in (
            _scenario_content_line(scenario, "hunt_profile", "Hunt profile"),
            *lean_lines,
            _scenario_content_line(scenario, "hunt_locations", "Hunt locations"),
        )
        if line
    )


def _road_trip_pilgrimage_identity_lines(
    scenario: ScenarioRecord,
    *,
    include_setup: bool = True,
) -> tuple[str, ...]:
    if scenario.type != "road_trip_pilgrimage":
        return ()
    lean_lines = (
        _scenario_content_line(scenario, "journey_progress", "Journey progress"),
        _scenario_content_line(
            scenario,
            "relationship_threads",
            "Relationship threads",
        ),
        _scenario_content_line(
            scenario,
            "transport_and_supplies",
            "Transport/supplies",
        ),
        _scenario_content_line(
            scenario,
            "recurring_pressures",
            "Recurring pressures",
        ),
    )
    if not include_setup:
        return tuple(line for line in lean_lines if line)
    return tuple(
        line
        for line in (
            _scenario_content_line(scenario, "journey_profile", "Journey profile"),
            _scenario_content_line(scenario, "route_and_stops", "Route/stops"),
            *lean_lines,
        )
        if line
    )


def _merchant_trade_route_identity_lines(
    scenario: ScenarioRecord,
    *,
    include_setup: bool = True,
) -> tuple[str, ...]:
    if scenario.type != "merchant_trade_route":
        return ()
    lean_lines = (
        _scenario_content_line(scenario, "cargo_inventory", "Cargo inventory"),
        _scenario_content_line(scenario, "contracts_and_debts", "Contracts/debts"),
        _scenario_content_line(scenario, "profit_and_loss", "Profit/loss"),
    )
    if not include_setup:
        return tuple(line for line in lean_lines if line)
    return tuple(
        line
        for line in (
            _scenario_content_line(scenario, "trade_profile", "Trade profile"),
            *lean_lines,
            _scenario_content_line(scenario, "markets_and_stops", "Markets/stops"),
            _scenario_content_line(scenario, "route_hazards", "Route hazards"),
        )
        if line
    )


def _survival_expedition_identity_lines(
    scenario: ScenarioRecord,
    *,
    include_setup: bool = True,
) -> tuple[str, ...]:
    if scenario.type != "survival_expedition":
        return ()
    lean_lines = (
        _scenario_content_line(scenario, "expedition_goal", "Expedition goal"),
        _scenario_content_line(scenario, "travel_progress", "Travel progress"),
        _scenario_content_line(
            scenario,
            "resource_inventory",
            "Resource inventory",
        ),
        _scenario_content_line(
            scenario,
            "environmental_conditions",
            "Environmental conditions",
        ),
    )
    if not include_setup:
        return tuple(line for line in lean_lines if line)
    return tuple(
        line
        for line in (
            *lean_lines,
            _scenario_content_line(scenario, "route_options", "Route options"),
            _scenario_content_line(
                scenario,
                "hazards_and_events",
                "Hazards/events",
            ),
            _scenario_content_line(scenario, "camp_status", "Camp status"),
        )
        if line
    )


def _time_loop_identity_lines(
    scenario: ScenarioRecord,
    *,
    include_setup: bool = True,
) -> tuple[str, ...]:
    if scenario.type != "time_loop":
        return ()
    lean_lines = (
        _scenario_content_line(scenario, "objective", "Loop objective"),
        _scenario_content_line(
            scenario,
            "persistent_knowledge",
            "Persistent player/meta knowledge",
        ),
        _scenario_content_line(scenario, "npc_memory_rules", "NPC memory rules"),
    )
    if not include_setup:
        return tuple(line for line in lean_lines if line)
    return tuple(
        line
        for line in (
            _scenario_content_line(scenario, "loop_premise", "Loop premise"),
            _scenario_content_line(scenario, "reset_trigger", "Reset trigger"),
            _scenario_content_line(scenario, "loop_duration", "Loop duration"),
            _scenario_content_line(scenario, "starting_state", "Starting state"),
            *lean_lines,
            _scenario_content_line(
                scenario,
                "failure_conditions",
                "Failure conditions",
            ),
            _scenario_content_line(
                scenario,
                "baseline_world_state",
                "Reset baseline",
            ),
            _scenario_content_line(scenario, "loop_schedule", "Loop schedule"),
            _scenario_content_line(
                scenario,
                "persistence_exceptions",
                "Persistence exceptions",
            ),
        )
        if line
    )


def _choose_your_own_adventure_identity_lines(
    scenario: ScenarioRecord,
) -> tuple[str, ...]:
    if not scenario_action_choices_enabled(scenario):
        return ()
    choice_style = _scenario_content_line(scenario, "choice_style", "Choice style")
    return (choice_style,) if choice_style else ()


def _scenario_content_line(scenario: ScenarioRecord, key: str, label: str) -> str:
    value = _scenario_content_text(scenario, key)
    if not value:
        return ""
    return f"{label}: {value}"


def _scenario_content_text(scenario: ScenarioRecord, key: str) -> str:
    try:
        content = json.loads(scenario.content_json)
    except json.JSONDecodeError:
        return ""
    if not isinstance(content, dict):
        return ""
    value = content.get(key)
    return value.strip() if isinstance(value, str) else ""


def scenario_section_candidates(
    scenario: ScenarioRecord | None,
) -> tuple[tuple[str, str, str], ...]:
    if scenario is None:
        return ()
    try:
        loaded = json.loads(scenario.content_json)
    except json.JSONDecodeError:
        return ()
    if not isinstance(loaded, dict):
        return ()
    candidates: list[tuple[str, str, str]] = []
    for key, value in loaded.items():
        section_id = str(key)
        if (
            section_id.startswith("_")
            or section_id in SCENARIO_CORE_CONTENT_KEYS
            or not value
        ):
            continue
        text = _section_text(value)
        candidates.append(
            (
                f"scenario:{scenario.id}:section:{section_id}",
                section_id,
                text,
            )
        )
    return tuple(candidates)


def deterministic_context_sources(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    mode: str = "narrator",
    source_message_id: str | None = None,
    focus_message: MessageRecord | None = None,
    details: SaveDetailsRecord | None = None,
    scene_snapshot: SceneSnapshotRecord | None | object = _MISSING,
    locations: tuple[LocationRecord, ...] | list[LocationRecord] | None = None,
    characters: tuple[CharacterRecord, ...] | list[CharacterRecord] | None = None,
    active_threads: (
        tuple[ActiveThreadRecord, ...] | list[ActiveThreadRecord] | None
    ) = None,
    character_knowledge_edges: (
        tuple[CharacterKnowledgeEdgeRecord, ...]
        | list[CharacterKnowledgeEdgeRecord]
        | None
    ) = None,
    entity_links: tuple[EntityLinkRecord, ...] | list[EntityLinkRecord] | None = None,
    memories: tuple[MemoryRecord, ...] | list[MemoryRecord] | None = None,
    world_state: tuple[WorldStateRecord, ...] | list[WorldStateRecord] | None = None,
    summaries: tuple[SummaryRecord, ...] | list[SummaryRecord] | None = None,
    message_visibility: (
        tuple[MessageVisibilityRecord, ...] | list[MessageVisibilityRecord] | None
    ) = None,
) -> tuple[ContextSource, ...]:
    details = (
        details
        if details is not None
        else repositories.load_save_details(save_id)
    )
    scenario = details.scenario if details is not None else None
    message_positions = (
        {message.id: index for index, message in enumerate(details.messages)}
        if details is not None and source_message_id is not None
        else None
    )
    snapshot = (
        repositories.get_scene_snapshot(save_id)
        if scene_snapshot is _MISSING
        else cast(SceneSnapshotRecord | None, scene_snapshot)
    )
    if message_positions is not None and not _record_is_at_or_before(
        snapshot,
        source_message_id=source_message_id,
        message_positions=message_positions,
    ):
        snapshot = None
    location_records = (
        tuple(locations)
        if locations is not None
        else tuple(repositories.list_locations(save_id))
    )
    character_records = (
        tuple(characters)
        if characters is not None
        else tuple(repositories.list_characters(save_id))
    )
    active_thread_records = (
        tuple(active_threads)
        if active_threads is not None
        else tuple(repositories.list_active_threads(save_id))
    )
    present_character_ids = (
        set(snapshot.present_character_ids) if snapshot is not None else set()
    )
    if message_visibility is not None:
        message_visibility_records = tuple(message_visibility)
    elif present_character_ids:
        message_visibility_records = tuple(
            repositories.list_message_visibility(
                save_id,
                character_ids=present_character_ids,
            )
        )
    else:
        message_visibility_records = ()
    world_state_records = (
        list(world_state)
        if world_state is not None
        else repositories.list_world_state(save_id)
    )
    location_map = {
        location.id: location
        for location in location_records
        if _record_is_at_or_before(
            location,
            source_message_id=source_message_id,
            message_positions=message_positions,
        )
    }
    character_map = {
        character.id: character
        for character in character_records
        if _record_is_at_or_before(
            character,
            source_message_id=source_message_id,
            message_positions=message_positions,
        )
    }
    raw_threads = [
        thread
        for thread in active_thread_records
        if _record_is_at_or_before(
            thread,
            source_message_id=source_message_id,
            message_positions=message_positions,
        )
        if _active_thread_source_visible_to_present_characters(
            thread,
            present_character_ids=present_character_ids,
            message_visibility=message_visibility_records,
        )
    ]
    threads = _context_active_threads(
        raw_threads,
        scene_snapshot=snapshot,
        characters=list(character_map.values()),
        focus_message=focus_message,
    )
    sources: list[ContextSource] = []
    world_time_text = format_world_time_from_snapshot(
        snapshot,
        include_legacy_detail=True,
    )
    if snapshot is not None and world_time_text:
        sources.append(
            ContextSource(
                tier="current_scene",
                source_type="scene_snapshot",
                source_id=f"{snapshot.id}:world_time",
                text=(
                    f"Current world time: {world_time_text}. Keep the "
                    "response consistent with this unless the player explicitly "
                    "advances time."
                ),
                reason="current world time",
                always_include=True,
            )
        )
    sources.append(
        ContextSource(
            tier="current_scene",
            source_type="rules",
            source_id="current_scene_authority",
            text=(
                "Current-scene context is deterministic application state. Treat "
                "it as authoritative over stale scenario setup."
            ),
            reason="scene authority",
            always_include=mode == "narrator",
        )
    )
    if snapshot is not None:
        sources.extend(
            _scene_snapshot_sources(snapshot, location_map, character_map, mode)
        )
        sources.extend(
            _dating_route_context_sources(
                repositories=repositories,
                save_id=save_id,
                snapshot=snapshot,
                characters=character_map,
                mode=mode,
                focus_message=focus_message,
            )
        )
    elif message_positions is None:
        sources.extend(
            _legacy_scene_sources(
                world_state_records,
                always_include=mode == "narrator",
            )
        )
    sources.extend(
        _first_contact_exploration_context_sources(
            scenario,
            world_state_records,
            always_include=mode == "narrator",
        )
    )
    sources.extend(
        _survival_expedition_context_sources(
            scenario,
            world_state_records,
            always_include=mode == "narrator",
        )
    )
    sources.extend(
        _time_loop_context_sources(
            scenario,
            world_state_records,
            always_include=mode == "narrator",
        )
    )
    sources.extend(
        _heist_infiltration_context_sources(
            scenario,
            world_state_records,
            always_include=mode == "narrator",
        )
    )
    sources.extend(
        _political_intrigue_context_sources(
            scenario,
            world_state_records,
            always_include=mode == "narrator",
        )
    )
    sources.extend(
        _management_template_context_sources(
            scenario,
            world_state_records,
            always_include=mode == "narrator",
        )
    )
    if threads:
        sources.append(_thread_source(threads, always_include=mode == "narrator"))
    sources.extend(
        _active_linked_fact_sources(
            repositories=repositories,
            save_id=save_id,
            scenario=scenario,
            snapshot=snapshot,
            threads=threads,
            active_thread_records_exist=bool(raw_threads),
            source_message_id=source_message_id,
            message_positions=message_positions,
            entity_links=entity_links,
            character_knowledge_edges=character_knowledge_edges,
            memories=memories,
            world_state=world_state_records,
            summaries=summaries,
            characters=tuple(character_records),
            message_visibility=message_visibility_records,
            always_include=mode == "narrator",
        )
    )
    if mode == "narrator":
        sources.extend(
            _active_participant_state_sources(
                repositories=repositories,
                save_id=save_id,
                snapshot=snapshot,
                characters=character_map,
                active_thread_records_exist=bool(raw_threads),
                source_message_id=source_message_id,
                message_positions=message_positions,
                world_state=world_state_records,
                message_visibility=message_visibility_records,
            )
        )
    return _dedupe_context_sources(tuple(sources))


def pre_turn_scene_hint_sources(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    player_message: MessageRecord,
    limit: int = PRE_TURN_SCENE_HINT_LIMIT,
    scene_snapshot: SceneSnapshotRecord | None | object = _MISSING,
    characters: tuple[CharacterRecord, ...] | list[CharacterRecord] | None = None,
) -> tuple[ContextSource, ...]:
    if limit <= 0 or not player_message.body.strip():
        return ()
    snapshot = (
        repositories.get_scene_snapshot(save_id)
        if scene_snapshot is _MISSING
        else cast(SceneSnapshotRecord | None, scene_snapshot)
    )
    if snapshot is None:
        return ()
    present_ids = set(snapshot.present_character_ids)
    sources: list[ContextSource] = []
    character_records = (
        tuple(characters)
        if characters is not None
        else tuple(repositories.list_characters(save_id))
    )
    for character in character_records:
        if not character_name_is_mentioned(
            name=character.name,
            aliases=character.aliases,
            text=player_message.body,
        ):
            continue
        present = character.id in present_ids
        source_kind = "present_character" if present else "non_present_character"
        if present:
            text = (
                "Pre-turn scene hint: latest player message mentions present "
                f"character {character.name}; keep their current scene status "
                "in focus."
            )
        else:
            text = (
                "Pre-turn scene hint: latest player message mentions known "
                f"character {character.name}, who is not marked present in the "
                "current scene; do not treat them as present unless the narrator "
                "response establishes it."
            )
        sources.append(
            ContextSource(
                tier="pre_turn_scene_hints",
                source_type="pre_turn_scene_hint",
                source_id=(
                    f"pre_turn_scene_hint:{player_message.id}:"
                    f"{source_kind}:{character.id}"
                ),
                text=text,
                reason="read-only latest player mention",
                always_include=True,
            )
        )
        if len(sources) >= limit:
            return tuple(sources)
    for field_name, label, values in (
        ("nearby_objects", "nearby object", snapshot.nearby_objects),
        ("hazards", "hazard", snapshot.hazards),
    ):
        for index, value in enumerate(values):
            if not _phrase_is_mentioned(value, player_message.body):
                continue
            sources.append(
                ContextSource(
                    tier="pre_turn_scene_hints",
                    source_type="pre_turn_scene_hint",
                    source_id=(
                        f"pre_turn_scene_hint:{player_message.id}:"
                        f"scene_detail:{field_name}:{index}"
                    ),
                    text=(
                        "Pre-turn scene hint: latest player message references "
                        f"current {label}: {value}."
                    ),
                    reason="read-only latest player scene-detail reference",
                    always_include=True,
                )
            )
            if len(sources) >= limit:
                return tuple(sources)
    return tuple(sources)


def pending_context_suggestion_sources(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    limit: int = PENDING_CONTEXT_SUGGESTION_LIMIT,
    suggestions: (
        tuple[ContextUpdateSuggestionRecord, ...]
        | list[ContextUpdateSuggestionRecord]
        | None
    ) = None,
) -> tuple[ContextSource, ...]:
    if limit <= 0:
        return ()
    suggestion_records = (
        list(suggestions)
        if suggestions is not None
        else repositories.list_context_update_suggestions(
            save_id,
            status="pending",
        )
    )
    suggestion_records = [
        suggestion
        for suggestion in suggestion_records
        if _pending_context_suggestion_prompt_eligible(suggestion)
    ]
    groups = _pending_context_suggestion_groups(suggestion_records)
    return tuple(
        ContextSource(
            tier="pending_context_suggestions",
            source_type="context_update_suggestion",
            source_id=",".join(group.suggestion_ids),
            text=_pending_context_suggestion_text(group),
            reason=_pending_context_suggestion_reason(group),
        )
        for group in _sort_pending_context_suggestion_groups(groups)[:limit]
    )


def _pending_context_suggestion_groups(
    suggestions: list[ContextUpdateSuggestionRecord],
) -> tuple[_PendingContextSuggestionGroup, ...]:
    grouped: dict[
        tuple[str, str, str | None, str, str],
        list[ContextUpdateSuggestionRecord],
    ] = {}
    for suggestion in suggestions:
        key = (
            suggestion.update_type,
            suggestion.entity_type,
            suggestion.entity_id,
            suggestion.field_path,
            _pending_suggestion_value_key(suggestion.proposed_value),
        )
        grouped.setdefault(key, []).append(suggestion)

    groups: list[_PendingContextSuggestionGroup] = []
    for members in grouped.values():
        first = members[0]
        groups.append(
            _PendingContextSuggestionGroup(
                update_type=first.update_type,
                entity_type=first.entity_type,
                entity_id=first.entity_id,
                field_path=first.field_path,
                proposed_value=first.proposed_value,
                suggestion_ids=tuple(member.id for member in members),
                confidence=max(member.confidence for member in members),
                created_at=max(member.created_at or "" for member in members),
            )
        )
    return tuple(groups)


def _sort_pending_context_suggestion_groups(
    groups: tuple[_PendingContextSuggestionGroup, ...],
) -> list[_PendingContextSuggestionGroup]:
    ordered = list(groups)
    ordered.sort(key=lambda group: group.suggestion_ids[0])
    ordered.sort(key=lambda group: group.created_at, reverse=True)
    ordered.sort(key=lambda group: group.confidence, reverse=True)
    return ordered


def _pending_context_suggestion_prompt_eligible(
    suggestion: ContextUpdateSuggestionRecord,
) -> bool:
    if suggestion.confidence < PENDING_CONTEXT_SUGGESTION_MIN_CONFIDENCE:
        return False
    created_at = _parse_context_timestamp(suggestion.created_at)
    if created_at is None:
        return True
    cutoff = datetime.now(UTC) - timedelta(
        hours=PENDING_CONTEXT_SUGGESTION_MAX_AGE_HOURS,
    )
    return created_at > cutoff


def _parse_context_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _pending_context_suggestion_text(group: _PendingContextSuggestionGroup) -> str:
    entity = group.entity_type
    if group.entity_id:
        entity = f"{entity}/{group.entity_id}"
    value = _compact_context_detail(
        _section_text(group.proposed_value),
        max_chars=PENDING_CONTEXT_VALUE_MAX_CHARS,
    )
    parts = [f"confidence={_confidence_percent(group.confidence)}"]
    if len(group.suggestion_ids) > 1:
        parts.append(f"grouped={len(group.suggestion_ids)}")
    return (
        f"Pending review (not canon yet): {group.update_type} {entity} "
        f"{group.field_path} -> {value}; {'; '.join(parts)}"
    )


def _pending_context_suggestion_reason(
    group: _PendingContextSuggestionGroup,
) -> str:
    parts = [
        "pending context review",
        f"confidence={_confidence_percent(group.confidence)}",
    ]
    if len(group.suggestion_ids) > 1:
        parts.append(f"grouped={len(group.suggestion_ids)}")
    return "; ".join(parts)


def _confidence_percent(value: float) -> str:
    return f"{round(value * 100)}%"


def _pending_suggestion_value_key(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def _dedupe_context_sources(
    sources: tuple[ContextSource, ...],
) -> tuple[ContextSource, ...]:
    deduped: list[ContextSource] = []
    seen: dict[tuple[str, str], int] = {}
    for source in sources:
        key = (source.source_type, source.source_id)
        if key in seen:
            existing_index = seen[key]
            if source.always_include and not deduped[existing_index].always_include:
                deduped[existing_index] = source
            continue
        seen[key] = len(deduped)
        deduped.append(source)
    return tuple(deduped)


def _active_participant_state_sources(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    snapshot: SceneSnapshotRecord | None,
    characters: dict[str, CharacterRecord],
    active_thread_records_exist: bool,
    source_message_id: str | None,
    message_positions: dict[str, int] | None,
    world_state: tuple[WorldStateRecord, ...] | list[WorldStateRecord] | None = None,
    message_visibility: tuple[MessageVisibilityRecord, ...] = (),
) -> tuple[ContextSource, ...]:
    if snapshot is None:
        return ()
    active_slug_names = _active_participant_slug_names(snapshot, characters)
    active_slugs = set(active_slug_names)
    if not active_slugs:
        return ()
    sources: list[ContextSource] = []
    world_state_records = (
        tuple(world_state)
        if world_state is not None
        else tuple(repositories.list_world_state(save_id))
    )
    for state in world_state_records:
        if active_thread_records_exist and is_open_threads_aggregate_key(state.key):
            continue
        if not _record_is_at_or_before(
            state,
            source_message_id=source_message_id,
            message_positions=message_positions,
        ):
            continue
        if not _source_message_ids_visible_to_active_characters(
            (state.source_message_id,),
            active_character_ids=set(snapshot.present_character_ids),
            message_visibility=message_visibility,
        ):
            continue
        if not _active_participant_state_key_matches(state.key, active_slugs):
            continue
        sources.append(
            ContextSource(
                tier="active_participant_facts",
                source_type="world_state",
                source_id=state.id,
                text=_active_participant_fact_text(state, active_slug_names),
                reason="active participant continuity",
                always_include=True,
            )
        )
    return tuple(sources)


def _active_participant_slug_names(
    snapshot: SceneSnapshotRecord,
    characters: dict[str, CharacterRecord],
) -> dict[str, str]:
    slug_names: dict[str, str] = {}
    for character_id in snapshot.present_character_ids:
        character = characters.get(character_id)
        if character is None:
            continue
        for slug in (
            _continuity_key_slug(character.name),
            *(_continuity_key_slug(alias) for alias in character.aliases),
        ):
            if slug:
                slug_names.setdefault(slug, character.name)
    return slug_names


def _active_participant_fact_text(
    state: WorldStateRecord,
    slug_names: dict[str, str],
) -> str:
    emotion_slug = _character_emotion_slug(state.key)
    if emotion_slug and emotion_slug in slug_names:
        return (
            "Active participant continuity: "
            f"{slug_names[emotion_slug]}'s current emotional state is "
            f"{_emotion_state_value_text(state.value)}"
        )
    return (
        f"Active participant continuity: {state.key}: "
        f"{_format_state_value(state.value)}"
    )


def _character_emotion_slug(key: str) -> str:
    normalized = key.strip().casefold()
    prefix = "character."
    suffix = ".current_emotional_state"
    if not normalized.startswith(prefix) or not normalized.endswith(suffix):
        return ""
    return normalized.removeprefix(prefix).removesuffix(suffix)


def _emotion_state_value_text(value: object) -> str:
    if isinstance(value, dict) and len(value) == 1:
        for key, item in value.items():
            if str(key).casefold() in {"mood", "emotion", "state"}:
                return str(item)
    return _format_state_value(value)


def _active_participant_state_key_matches(key: str, active_slugs: set[str]) -> bool:
    normalized = key.strip().casefold()
    relationship_key = _relationship_key(normalized)
    if relationship_key:
        if relationship_key.startswith("player_to_"):
            return relationship_key.removeprefix("player_to_") in active_slugs
        if relationship_key.endswith("_to_player"):
            return relationship_key.removesuffix("_to_player") in active_slugs
        return any(
            relationship_key.startswith(f"{slug}_to_")
            or relationship_key.endswith(f"_to_{slug}")
            for slug in active_slugs
        )
    return any(
        _active_participant_character_key_matches(normalized, slug)
        for slug in active_slugs
    )


def _active_participant_character_key_matches(
    normalized_key: str,
    slug: str,
) -> bool:
    prefix = f"character.{slug}."
    if not normalized_key.startswith(prefix):
        return False
    suffix = normalized_key.removeprefix(prefix)
    if _state_key_suffix_matches(suffix, "revealed_traits") or suffix.startswith(
        "known_about_"
    ):
        return True
    return any(
        _state_key_suffix_matches(suffix, namespace)
        for namespace in (
            "preferences",
            "boundaries",
            "current_emotional_state",
        )
    )


def _state_key_suffix_matches(suffix: str, namespace: str) -> bool:
    return suffix == namespace or suffix.startswith(f"{namespace}.")


def _relationship_key(normalized_key: str) -> str:
    if not normalized_key.startswith("relationship."):
        return ""
    return normalized_key.removeprefix("relationship.").split(".", 1)[0]


def _continuity_key_slug(value: str) -> str:
    parts: list[str] = []
    current: list[str] = []
    for char in value.casefold():
        if char.isalnum():
            current.append(char)
            continue
        if current:
            parts.append("".join(current))
            current.clear()
    if current:
        parts.append("".join(current))
    return "_".join(parts)


def _phrase_is_mentioned(phrase: str, text: str) -> bool:
    parts = phrase.strip().split()
    if not parts:
        return False
    phrase_pattern = r"\s+".join(re.escape(part.casefold()) for part in parts)
    pattern = re.compile(
        rf"(?<![\w-]){phrase_pattern}(?:['\u2019]s)?(?![\w-])"
    )
    return pattern.search(text.casefold()) is not None


def _record_is_at_or_before(
    record: object | None,
    *,
    source_message_id: str | None,
    message_positions: dict[str, int] | None,
) -> bool:
    if record is None or source_message_id is None or message_positions is None:
        return True
    record_source = getattr(record, "first_seen_message_id", None) or getattr(
        record,
        "source_message_id",
        None,
    )
    selected_position = message_positions.get(source_message_id)
    if selected_position is None:
        return True
    if not isinstance(record_source, str) or record_source not in message_positions:
        latest_position = max(message_positions.values(), default=selected_position)
        return selected_position == latest_position
    return message_positions[record_source] <= selected_position


def _summary_is_at_or_before(
    summary: SummaryRecord,
    *,
    source_message_id: str | None,
    message_positions: dict[str, int] | None,
) -> bool:
    if source_message_id is None or message_positions is None:
        return True
    selected_position = message_positions.get(source_message_id)
    summary_end_position = message_positions.get(summary.covers_message_end_id)
    if selected_position is None or summary_end_position is None:
        return True
    return summary_end_position <= selected_position


def _active_linked_fact_sources(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    scenario: ScenarioRecord | None,
    snapshot: SceneSnapshotRecord | None,
    threads: list[ActiveThreadRecord],
    active_thread_records_exist: bool,
    source_message_id: str | None = None,
    message_positions: dict[str, int] | None = None,
    entity_links: tuple[EntityLinkRecord, ...] | list[EntityLinkRecord] | None = None,
    character_knowledge_edges: (
        tuple[CharacterKnowledgeEdgeRecord, ...]
        | list[CharacterKnowledgeEdgeRecord]
        | None
    ) = None,
    memories: tuple[MemoryRecord, ...] | list[MemoryRecord] | None = None,
    world_state: tuple[WorldStateRecord, ...] | list[WorldStateRecord] | None = None,
    summaries: tuple[SummaryRecord, ...] | list[SummaryRecord] | None = None,
    characters: tuple[CharacterRecord, ...] | list[CharacterRecord] | None = None,
    message_visibility: (
        tuple[MessageVisibilityRecord, ...] | list[MessageVisibilityRecord] | None
    ) = None,
    always_include: bool = False,
) -> tuple[ContextSource, ...]:
    active_entities = _active_scene_entities(snapshot=snapshot, threads=threads)
    if not active_entities:
        return ()
    active_character_ids = {
        entity_id
        for entity_type, entity_id in active_entities
        if entity_type == "character"
    }
    visibility_records = tuple(message_visibility or ())
    link_records = (
        tuple(entity_links)
        if entity_links is not None
        else tuple(repositories.list_entity_links(save_id))
    )
    links = [
        link
        for link in link_records
        if (link.entity_type, link.entity_id) in active_entities
    ]
    character_records = (
        tuple(characters)
        if characters is not None
        else tuple(repositories.list_characters(save_id))
    )
    player_character_ids = {
        character.id
        for character in character_records
        if character.is_player_character
    }
    edge_records = (
        tuple(character_knowledge_edges)
        if character_knowledge_edges is not None
        else tuple(
            repositories.list_character_knowledge_edges(
                save_id,
                character_ids=active_character_ids,
            )
        )
    )
    knowledge_edges = tuple(
        edge
        for edge in edge_records
        if edge.character_id in active_character_ids
        if not (
            edge.character_id in player_character_ids
            and knowledge_edge_has_character_text_source(edge)
        )
        if _knowledge_edge_source_visible_to_active_characters(
            edge,
            active_character_ids=active_character_ids,
            message_visibility=visibility_records,
        )
    )
    if not links and not knowledge_edges:
        return ()
    memory_records = (
        tuple(memories)
        if memories is not None
        else tuple(repositories.list_memories(save_id))
    )
    world_state_records = (
        tuple(world_state)
        if world_state is not None
        else tuple(repositories.list_world_state(save_id))
    )
    summary_records = (
        tuple(summaries)
        if summaries is not None
        else tuple(repositories.list_summaries(save_id))
    )
    memory_by_id = {
        memory.id: memory
        for memory in memory_records
        if _record_is_at_or_before(
            memory,
            source_message_id=source_message_id,
            message_positions=message_positions,
        )
        if _source_message_ids_visible_to_active_characters(
            (
                *memory.source_message_ids,
                *([memory.source_message_id] if memory.source_message_id else []),
            ),
            active_character_ids=active_character_ids,
            message_visibility=visibility_records,
        )
    }
    state_by_id = {
        state.id: state
        for state in world_state_records
        if not (
            active_thread_records_exist and is_open_threads_aggregate_key(state.key)
        )
        if _source_message_ids_visible_to_active_characters(
            (state.source_message_id,),
            active_character_ids=active_character_ids,
            message_visibility=visibility_records,
        )
        if _record_is_at_or_before(
            state,
            source_message_id=source_message_id,
            message_positions=message_positions,
        )
    }
    summary_by_id = {
        summary.id: summary
        for summary in summary_records
        if validate_summary_output(summary.body).accepted
        if _source_message_ids_visible_to_active_characters(
            (summary.covers_message_start_id, summary.covers_message_end_id),
            active_character_ids=active_character_ids,
            message_visibility=visibility_records,
        )
        if _summary_is_at_or_before(
            summary,
            source_message_id=source_message_id,
            message_positions=message_positions,
        )
    }
    scenario_candidates = scenario_section_candidates(scenario)
    scenario_sections = {
        source_id: (section_id, text)
        for source_id, section_id, text in scenario_candidates
    }
    scenario_sections_by_key = {
        section_id: (source_id, text)
        for source_id, section_id, text in scenario_candidates
    }
    character_names = {
        character.id: character.name
        for character in character_records
    }
    sources: list[ContextSource] = []
    seen: set[tuple[str, str]] = set()
    knowledge_edge_targets: set[tuple[str, str, str]] = set()
    for edge in knowledge_edges:
        target_type = _normalized_link_type(edge.target_type)
        knowledge_edge_targets.add((edge.character_id, target_type, edge.target_id))
        source = _knowledge_edge_source(
            edge=edge,
            memory_by_id=memory_by_id,
            state_by_id=state_by_id,
            summary_by_id=summary_by_id,
            scenario_sections=scenario_sections,
            scenario_sections_by_key=scenario_sections_by_key,
            character_names=character_names,
        )
        if source is None:
            continue
        key = (source.source_type, source.source_id)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            replace(source, always_include=True) if always_include else source
        )
    for link in links:
        target_type = _normalized_link_type(link.target_type)
        if not _link_source_visible_to_active_characters(
            link,
            active_character_ids=active_character_ids,
            message_visibility=visibility_records,
        ):
            continue
        if (
            link.entity_type == "character"
            and link.relation == "knows"
            and (link.entity_id, target_type, link.target_id) in knowledge_edge_targets
        ):
            continue
        source = _linked_fact_source(
            link=link,
            memory_by_id=memory_by_id,
            state_by_id=state_by_id,
            summary_by_id=summary_by_id,
            scenario_sections=scenario_sections,
            scenario_sections_by_key=scenario_sections_by_key,
            character_names=character_names,
        )
        if source is None:
            continue
        key = (source.source_type, source.source_id)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            replace(source, always_include=True) if always_include else source
        )
    return tuple(sources)


def _active_scene_entities(
    *,
    snapshot: SceneSnapshotRecord | None,
    threads: list[ActiveThreadRecord],
) -> set[tuple[str, str]]:
    entities = {("active_thread", thread.id) for thread in threads}
    if snapshot is None:
        return entities
    entities.add(("scene_snapshot", snapshot.id))
    if snapshot.current_location_id is not None:
        entities.add(("location", snapshot.current_location_id))
    entities.update(
        ("character", character_id) for character_id in snapshot.present_character_ids
    )
    return entities


def _linked_fact_source(
    *,
    link: EntityLinkRecord,
    memory_by_id: dict[str, MemoryRecord],
    state_by_id: dict[str, WorldStateRecord],
    summary_by_id: dict[str, SummaryRecord],
    scenario_sections: dict[str, tuple[str, str]],
    scenario_sections_by_key: dict[str, tuple[str, str]],
    character_names: dict[str, str],
) -> ContextSource | None:
    target_type = _normalized_link_type(link.target_type)
    prefix = _linked_fact_prefix(link, character_names)
    if target_type == "memory":
        memory = memory_by_id.get(link.target_id)
        if memory is None:
            return None
        return ContextSource(
            tier="active_linked_facts",
            source_type="memory",
            source_id=memory.id,
            text=f"{prefix}memory: {memory.body}",
            reason=_link_reason(link),
        )
    if target_type == "world_state":
        state = state_by_id.get(link.target_id)
        if state is None:
            return None
        return ContextSource(
            tier="active_linked_facts",
            source_type="world_state",
            source_id=state.id,
            text=(
                f"{prefix}world state: {state.key}: "
                f"{_format_state_value(state.value)}"
            ),
            reason=_link_reason(link),
        )
    if target_type == "summary":
        summary = summary_by_id.get(link.target_id)
        if summary is None:
            return None
        return ContextSource(
            tier="active_linked_facts",
            source_type="summary",
            source_id=summary.id,
            text=f"{prefix}summary: {summary.body}",
            reason=_link_reason(link),
        )
    if target_type == "scenario_section":
        section = scenario_sections.get(link.target_id)
        source_id = link.target_id
        if section is None:
            by_key = scenario_sections_by_key.get(link.target_id)
            if by_key is None:
                return None
            source_id, text = by_key
            section_id = link.target_id
        else:
            section_id, text = section
        return ContextSource(
            tier="active_linked_facts",
            source_type="scenario_section",
            source_id=source_id,
            text=f"{prefix}scenario section ({section_id}): {text}",
            reason=_link_reason(link),
        )
    return None


def _knowledge_edge_source(
    *,
    edge: CharacterKnowledgeEdgeRecord,
    memory_by_id: dict[str, MemoryRecord],
    state_by_id: dict[str, WorldStateRecord],
    summary_by_id: dict[str, SummaryRecord],
    scenario_sections: dict[str, tuple[str, str]],
    scenario_sections_by_key: dict[str, tuple[str, str]],
    character_names: dict[str, str],
) -> ContextSource | None:
    if not _knowledge_edge_allows_prompt_use(edge):
        return None
    target_type = _normalized_link_type(edge.target_type)
    prefix = _knowledge_edge_prefix(edge, character_names)
    reason = _knowledge_edge_reason(edge)
    if target_type == "memory":
        memory = memory_by_id.get(edge.target_id)
        if memory is None:
            return None
        return ContextSource(
            tier="active_linked_facts",
            source_type="memory",
            source_id=memory.id,
            text=f"{prefix}memory: {memory.body}",
            reason=reason,
        )
    if target_type == "world_state":
        state = state_by_id.get(edge.target_id)
        if state is None:
            return None
        return ContextSource(
            tier="active_linked_facts",
            source_type="world_state",
            source_id=state.id,
            text=(
                f"{prefix}world state: {state.key}: "
                f"{_format_state_value(state.value)}"
            ),
            reason=reason,
        )
    if target_type == "summary":
        summary = summary_by_id.get(edge.target_id)
        if summary is None:
            return None
        return ContextSource(
            tier="active_linked_facts",
            source_type="summary",
            source_id=summary.id,
            text=f"{prefix}summary: {summary.body}",
            reason=reason,
        )
    if target_type == "scenario_section":
        section = scenario_sections.get(edge.target_id)
        source_id = edge.target_id
        if section is None:
            by_key = scenario_sections_by_key.get(edge.target_id)
            if by_key is None:
                return None
            source_id, text = by_key
            section_id = edge.target_id
        else:
            section_id, text = section
        return ContextSource(
            tier="active_linked_facts",
            source_type="scenario_section",
            source_id=source_id,
            text=f"{prefix}scenario section ({section_id}): {text}",
            reason=reason,
        )
    return None


def _knowledge_edge_allows_prompt_use(edge: CharacterKnowledgeEdgeRecord) -> bool:
    if edge.knowledge_state == "knows":
        return True
    return (
        edge.knowledge_state == "may_know"
        and edge.confidence >= SCOPED_MAY_KNOW_CONFIDENCE_THRESHOLD
    )


def _knowledge_edge_prefix(
    edge: CharacterKnowledgeEdgeRecord,
    character_names: dict[str, str],
) -> str:
    name = character_names.get(edge.character_id, edge.character_id)
    relation = "may know" if edge.knowledge_state == "may_know" else "knows"
    return f"Character-scoped knowledge ({name} {relation}) linked "


def _knowledge_edge_reason(edge: CharacterKnowledgeEdgeRecord) -> str:
    method = (
        f"; {edge.acquisition_method}"
        if edge.acquisition_method and edge.acquisition_method != "unknown"
        else ""
    )
    return f"knowledge graph edge ({edge.knowledge_state}{method})"


def _linked_fact_prefix(
    link: EntityLinkRecord,
    character_names: dict[str, str],
) -> str:
    if link.entity_type == "character" and link.relation == "knows":
        name = character_names.get(link.entity_id, link.entity_id)
        return f"Character-scoped knowledge ({name} knows) linked "
    return "Linked "


def _normalized_link_type(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized in {"state", "world_state"}:
        return "world_state"
    if normalized in {"memory", "memories"}:
        return "memory"
    if normalized in {"scenario", "scenario_section"}:
        return "scenario_section"
    return normalized


def _knowledge_edge_source_visible_to_active_characters(
    edge: CharacterKnowledgeEdgeRecord,
    *,
    active_character_ids: set[str],
    message_visibility: tuple[MessageVisibilityRecord, ...],
) -> bool:
    return _source_message_ids_visible_to_active_characters(
        (
            *edge.source_message_ids,
            *([edge.source_message_id] if edge.source_message_id else []),
        ),
        active_character_ids=active_character_ids,
        message_visibility=message_visibility,
    )


def _active_thread_source_visible_to_present_characters(
    thread: ActiveThreadRecord,
    *,
    present_character_ids: set[str],
    message_visibility: tuple[MessageVisibilityRecord, ...],
) -> bool:
    return _source_message_ids_visible_to_active_characters(
        (
            thread.source_message_id,
            thread.first_seen_message_id,
            thread.last_updated_message_id,
        ),
        active_character_ids=present_character_ids,
        message_visibility=message_visibility,
    )


def _source_message_ids_visible_to_active_characters(
    source_message_ids: tuple[str | None, ...],
    *,
    active_character_ids: set[str],
    message_visibility: tuple[MessageVisibilityRecord, ...],
) -> bool:
    if not active_character_ids:
        return True
    unique_source_message_ids = tuple(
        dict.fromkeys(source_id for source_id in source_message_ids if source_id)
    )
    return all(
        message_visible_to_present_characters(
            message_id=source_message_id,
            present_character_ids=frozenset(active_character_ids),
            message_visibility=[*message_visibility],
        )
        for source_message_id in unique_source_message_ids
    )


def _link_source_visible_to_active_characters(
    link: EntityLinkRecord,
    *,
    active_character_ids: set[str],
    message_visibility: tuple[MessageVisibilityRecord, ...],
) -> bool:
    if link.source_message_id is None:
        return True
    return _source_message_ids_visible_to_active_characters(
        (link.source_message_id,),
        active_character_ids=active_character_ids,
        message_visibility=message_visibility,
    )


def _link_reason(link: EntityLinkRecord) -> str:
    relation = f" ({link.relation})" if link.relation else ""
    return f"linked to active {link.entity_type}{relation}"


def _scene_snapshot_sources(
    snapshot: SceneSnapshotRecord,
    locations: dict[str, LocationRecord],
    characters: dict[str, CharacterRecord],
    mode: str,
) -> tuple[ContextSource, ...]:
    sources: list[ContextSource] = []
    scene_parts = [
        ("situation", snapshot.situation),
        ("objective", snapshot.objective),
        ("weather", snapshot.weather),
        ("mood", snapshot.mood),
    ]
    if (
        snapshot.in_world_time
        and not snapshot.time_of_day
        and not snapshot.day_of_week
        and snapshot.world_day_index is None
    ):
        scene_parts.append(("in-world time", snapshot.in_world_time))
    nearby = ", ".join(snapshot.nearby_objects)
    hazards = ", ".join(snapshot.hazards)
    if nearby:
        scene_parts.append(("nearby objects", nearby))
    if hazards:
        scene_parts.append(("hazards", hazards))
    scene_text = "; ".join(f"{label}: {value}" for label, value in scene_parts if value)
    if scene_text:
        sources.append(
            ContextSource(
                tier="current_scene",
                source_type="scene_snapshot",
                source_id=snapshot.id,
                text=f"Scene snapshot: {scene_text}",
                reason="current scene snapshot",
                always_include=mode == "narrator",
            )
        )
    location = (
        locations.get(snapshot.current_location_id)
        if snapshot.current_location_id is not None
        else None
    )
    if location is not None:
        location_text = _location_text(location, mode)
        if location_text:
            sources.append(
                ContextSource(
                    tier="current_location",
                    source_type="location",
                    source_id=location.id,
                    text=location_text,
                    reason="current location",
                    always_include=mode == "narrator",
                )
            )
    present = [
        characters[character_id]
        for character_id in snapshot.present_character_ids
        if character_id in characters
    ]
    if present:
        sources.append(
            ContextSource(
                tier="present_characters",
                source_type="character",
                source_id=",".join(character.id for character in present),
                text=_characters_text(present, mode),
                reason="present characters",
                always_include=mode == "narrator",
            )
        )
    return tuple(sources)


def _dating_route_context_sources(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    snapshot: SceneSnapshotRecord,
    characters: dict[str, CharacterRecord],
    mode: str,
    focus_message: MessageRecord | None = None,
) -> tuple[ContextSource, ...]:
    if mode != "narrator":
        return ()
    present_ids = set(snapshot.present_character_ids)
    focus_text = focus_message.body if focus_message is not None else ""
    player = next(
        (
            character
            for character in characters.values()
            if character.is_player_character
        ),
        None,
    )
    sources: list[ContextSource] = []
    for route in repositories.list_dating_route_states(save_id):
        npc = characters.get(route.npc_character_id)
        if npc is None:
            continue
        route_is_relevant = npc.id in present_ids or (
            bool(focus_text.strip())
            and character_name_is_mentioned(
                name=npc.name,
                aliases=npc.aliases,
                text=focus_text,
            )
        )
        if not route_is_relevant:
            continue
        sources.append(
            ContextSource(
                tier="dating_route_pacing",
                source_type="dating_route_state",
                source_id=route.id,
                text=_dating_route_context_text(
                    route=route,
                    player=player,
                    npc=npc,
                    world_day_index=snapshot.world_day_index,
                ),
                reason="current dating route pacing",
                always_include=True,
            )
        )
    return tuple(sources)


def _dating_route_context_text(
    *,
    route: DatingRouteStateRecord,
    player: CharacterRecord | None,
    npc: CharacterRecord,
    world_day_index: int | None,
) -> str:
    relationship = f" with {player.name}" if player is not None else ""
    parts = [
        f"Dating route pacing for {npc.name}{relationship}",
        f"stage: {route.stage.replace('_', ' ')}",
    ]
    known_days = _known_world_day_count(
        first_day=route.first_met_world_day_index,
        current_day=world_day_index,
    )
    if known_days is not None:
        parts.append(f"known for {known_days} in-world days")
    parts.append(f"completed interactions: {route.completed_interactions}")
    parts.append(f"dates completed: {route.dates_completed}")
    if route.interest_level:
        parts.append(f"interest: {route.interest_level}")
    if route.trust_level:
        parts.append(f"trust: {route.trust_level}")
    if route.comfort_with_intimacy:
        parts.append(f"comfort with intimacy: {route.comfort_with_intimacy}")
    if route.pacing_preference:
        parts.append(f"pacing: {route.pacing_preference}")
    if route.known_boundaries:
        parts.append("known boundaries: " + "; ".join(route.known_boundaries))
    if route.unresolved_questions:
        parts.append("unresolved questions: " + "; ".join(route.unresolved_questions))
    if route.next_reasonable_step:
        parts.append(f"next plausible step: {route.next_reasonable_step}")
    policy = escalation_policy_for_stage(route.stage)
    parts.append(f"max plausible escalation: {policy.max_plausible_escalation}")
    parts.append(
        "intimacy profile: "
        + intimacy_profile_guidance(
            comfort_with_intimacy=route.comfort_with_intimacy,
            pacing_preference=route.pacing_preference,
            known_boundaries=route.known_boundaries,
        )
    )
    if policy.allowed_progress:
        parts.append("allowed now: " + "; ".join(policy.allowed_progress))
    if policy.needs_explicit_support:
        parts.append(
            "needs explicit support: " + "; ".join(policy.needs_explicit_support)
        )
    if policy.premature_escalations:
        parts.append("premature now: " + "; ".join(policy.premature_escalations))
    return "; ".join(parts) + "."


def _known_world_day_count(
    *,
    first_day: int | None,
    current_day: int | None,
) -> int | None:
    if first_day is None or current_day is None:
        return None
    return max(0, current_day - first_day)


def _legacy_scene_sources(
    world_state: list[WorldStateRecord],
    *,
    always_include: bool = False,
) -> tuple[ContextSource, ...]:
    scene_records = [
        record
        for record in world_state
        if record.key.startswith("scene.") or record.category in {"scene", "location"}
    ]
    if not scene_records:
        return ()
    return (
        ContextSource(
            tier="legacy_scene_state",
            source_type="world_state",
            source_id=",".join(record.id for record in scene_records),
            text="Legacy scene state: "
            + "; ".join(
                f"{record.key}: {_format_state_value(record.value)}"
                for record in scene_records
            ),
            reason="legacy scene state",
            always_include=always_include,
        ),
    )


_SURVIVAL_EXPEDITION_STATE_KEY_ORDER = {
    key: index
    for index, key in enumerate(
        (
            "expedition.goal",
            "expedition.route",
            "expedition.party",
            "expedition.resources",
            "expedition.environment",
            "expedition.hazards",
            "expedition.camp",
            "expedition.progress",
        )
    )
}


_FIRST_CONTACT_EXPLORATION_STATE_KEY_ORDER = {
    key: index
    for index, key in enumerate(
        (
            "contact.mission",
            "contact.crew",
            "contact.base",
            "contact.target",
            "contact.intelligence",
            "contact.knowledge",
            "contact.translation",
            "contact.discoveries",
            "contact.hazards",
            "mission.objective",
            "mission.constraints",
            "ship.status",
            "base.status",
        )
    )
}

_FIRST_CONTACT_EXPLORATION_STATE_KEY_PREFIXES = (
    "contact.",
    "mission.",
    "ship.",
    "base.",
    "crew.",
    "site.",
    "translation.",
    "discovery.",
    "sample.",
    "escalation.",
)


_TIME_LOOP_STATE_KEY_ORDER = {
    key: index
    for index, key in enumerate(
        (
            "loop.rules",
            "loop.starting_state",
            "loop.objective",
            "loop.baseline",
            "loop.schedule",
            "loop.knowledge",
            "loop.persistence",
            "loop.npc_memory",
            "loop.current",
        )
    )
}


_HEIST_INFILTRATION_STATE_KEY_ORDER = {
    key: index
    for index, key in enumerate(
        (
            "heist.target",
            "heist.objectives",
            "heist.crew",
            "heist.intel",
            "heist.security",
            "heist.alert",
            "heist.loadout",
            "heist.complications",
            "heist.extraction",
            "heist.aftermath",
        )
    )
}

_POLITICAL_INTRIGUE_STATE_KEY_ORDER = {
    key: index
    for index, key in enumerate(
        (
            "intrigue.arena",
            "intrigue.factions",
            "intrigue.npcs",
            "intrigue.conflict",
            "intrigue.secrets",
            "intrigue.standing",
            "intrigue.obligations",
            "intrigue.alliances",
            "intrigue.calendar",
            "intrigue.pressure",
            "intrigue.knowledge",
        )
    )
}

_POLITICAL_INTRIGUE_STATE_KEY_PREFIXES = (
    "intrigue.",
    "faction.",
    "obligation.",
    "alliance.",
)

_POLITICAL_INTRIGUE_STATE_CATEGORIES = frozenset(
    {
        "faction",
        "reputation",
        "obligation",
        "relationship",
        "leverage",
        "knowledge_boundary",
    }
)

_MANAGEMENT_TEMPLATE_CONTEXT_CONFIG = {
    "settlement_builder": (
        "Current settlement state",
        "settlement state",
        {
            key: index
            for index, key in enumerate(
                (
                    "settlement.profile",
                    "settlement.population",
                    "settlement.resources",
                    "settlement.projects",
                    "settlement.facilities",
                    "settlement.pressures",
                    "settlement.calendar",
                    "settlement.relationships",
                )
            )
        },
        ("settlement.", "project.", "resource."),
        frozenset(
            {
                "settlement",
                "resource",
                "project",
                "schedule",
                "threat",
            }
        ),
    ),
    "monster_hunt_bounty": (
        "Current hunt state",
        "hunt state",
        {
            key: index
            for index, key in enumerate(
                (
                    "hunt.profile",
                    "hunt.target",
                    "hunt.leads",
                    "hunt.locations",
                    "hunt.rivals",
                    "hunt.preparation",
                    "hunt.status",
                )
            )
        },
        ("hunt.", "target.", "clue."),
        frozenset({"hunt", "threat", "clue", "location", "faction", "inventory"}),
    ),
    "road_trip_pilgrimage": (
        "Current journey state",
        "journey state",
        {
            key: index
            for index, key in enumerate(
                (
                    "journey.profile",
                    "journey.route",
                    "journey.party",
                    "journey.supplies",
                    "journey.pressures",
                    "journey.relationships",
                    "journey.progress",
                )
            )
        },
        ("journey.", "stop.", "companion.", "vehicle."),
        frozenset({"journey", "location", "inventory", "threat", "relationship"}),
    ),
    "merchant_trade_route": (
        "Current trade state",
        "trade state",
        {
            key: index
            for index, key in enumerate(
                (
                    "trade.profile",
                    "trade.cargo",
                    "trade.markets",
                    "trade.contracts",
                    "trade.hazards",
                    "trade.reputation",
                    "trade.ledger",
                )
            )
        },
        ("trade.", "cargo.", "contract.", "debt.", "market."),
        frozenset(
            {"trade", "inventory", "contract", "threat", "reputation", "finance"}
        ),
    ),
}


def _first_contact_exploration_context_sources(
    scenario: ScenarioRecord | None,
    world_state: list[WorldStateRecord],
    *,
    always_include: bool = False,
) -> tuple[ContextSource, ...]:
    if scenario is None or scenario.type != "first_contact_exploration":
        return ()
    contact_records = [
        record
        for record in world_state
        if record.key.startswith(_FIRST_CONTACT_EXPLORATION_STATE_KEY_PREFIXES)
    ]
    if not contact_records:
        return ()
    contact_records.sort(
        key=lambda record: (
            _FIRST_CONTACT_EXPLORATION_STATE_KEY_ORDER.get(record.key, 999),
            record.key,
        )
    )
    return (
        ContextSource(
            tier="current_scene",
            source_type="world_state",
            source_id=",".join(record.id for record in contact_records),
            text="Current first-contact state: "
            + "; ".join(
                f"{record.key}: {_format_state_value(record.value)}"
                for record in contact_records
            ),
            reason="first-contact state",
            always_include=always_include,
        ),
    )


def _survival_expedition_context_sources(
    scenario: ScenarioRecord | None,
    world_state: list[WorldStateRecord],
    *,
    always_include: bool = False,
) -> tuple[ContextSource, ...]:
    if scenario is None or scenario.type != "survival_expedition":
        return ()
    expedition_records = [
        record for record in world_state if record.key.startswith("expedition.")
    ]
    if not expedition_records:
        return ()
    expedition_records.sort(
        key=lambda record: (
            _SURVIVAL_EXPEDITION_STATE_KEY_ORDER.get(record.key, 999),
            record.key,
        )
    )
    return (
        ContextSource(
            tier="current_scene",
            source_type="world_state",
            source_id=",".join(record.id for record in expedition_records),
            text="Current expedition state: "
            + "; ".join(
                f"{record.key}: {_format_state_value(record.value)}"
                for record in expedition_records
            ),
            reason="survival expedition state",
            always_include=always_include,
        ),
    )


def _heist_infiltration_context_sources(
    scenario: ScenarioRecord | None,
    world_state: list[WorldStateRecord],
    *,
    always_include: bool = False,
) -> tuple[ContextSource, ...]:
    if scenario is None or scenario.type != "heist_infiltration":
        return ()
    heist_records = [
        record for record in world_state if record.key.startswith("heist.")
    ]
    if not heist_records:
        return ()
    heist_records.sort(
        key=lambda record: (
            _HEIST_INFILTRATION_STATE_KEY_ORDER.get(record.key, 999),
            record.key,
        )
    )
    return (
        ContextSource(
            tier="current_scene",
            source_type="world_state",
            source_id=",".join(record.id for record in heist_records),
            text="Current heist state: "
            + "; ".join(
                f"{record.key}: {_format_state_value(record.value)}"
                for record in heist_records
            ),
            reason="heist state",
            always_include=always_include,
        ),
    )


def _political_intrigue_context_sources(
    scenario: ScenarioRecord | None,
    world_state: list[WorldStateRecord],
    *,
    always_include: bool = False,
) -> tuple[ContextSource, ...]:
    if scenario is None or scenario.type != "political_intrigue":
        return ()
    intrigue_records = [
        record for record in world_state if _is_political_intrigue_state(record)
    ]
    if not intrigue_records:
        return ()
    intrigue_records.sort(
        key=lambda record: (
            _POLITICAL_INTRIGUE_STATE_KEY_ORDER.get(record.key, 999),
            record.key,
        )
    )
    return (
        ContextSource(
            tier="current_scene",
            source_type="world_state",
            source_id=",".join(record.id for record in intrigue_records),
            text="Current political intrigue state: "
            + "; ".join(
                f"{record.key}: {_format_state_value(record.value)}"
                for record in intrigue_records
            ),
            reason="political intrigue state",
            always_include=always_include,
        ),
    )


def _is_political_intrigue_state(record: WorldStateRecord) -> bool:
    return record.key.startswith(_POLITICAL_INTRIGUE_STATE_KEY_PREFIXES) or (
        record.category in _POLITICAL_INTRIGUE_STATE_CATEGORIES
    )


def _management_template_context_sources(
    scenario: ScenarioRecord | None,
    world_state: list[WorldStateRecord],
    *,
    always_include: bool = False,
) -> tuple[ContextSource, ...]:
    if scenario is None:
        return ()
    config = _MANAGEMENT_TEMPLATE_CONTEXT_CONFIG.get(scenario.type)
    if config is None:
        return ()
    label, reason, key_order, key_prefixes, categories = config
    records = [
        record
        for record in world_state
        if record.key.startswith(key_prefixes) or record.category in categories
    ]
    if not records:
        return ()
    records.sort(key=lambda record: (key_order.get(record.key, 999), record.key))
    return (
        ContextSource(
            tier="current_scene",
            source_type="world_state",
            source_id=",".join(record.id for record in records),
            text=f"{label}: "
            + "; ".join(
                f"{record.key}: {_format_state_value(record.value)}"
                for record in records
            ),
            reason=reason,
            always_include=always_include,
        ),
    )


def _time_loop_context_sources(
    scenario: ScenarioRecord | None,
    world_state: list[WorldStateRecord],
    *,
    always_include: bool = False,
) -> tuple[ContextSource, ...]:
    if scenario is None or scenario.type != "time_loop":
        return ()
    current = next(
        (record for record in world_state if record.key == "loop.current"),
        None,
    )
    loop_records = [
        record
        for record in world_state
        if record.key.startswith("loop.") and record.key != "loop.current"
    ]
    if not loop_records and current is None:
        return ()
    loop_records.sort(
        key=lambda record: (
            _TIME_LOOP_STATE_KEY_ORDER.get(record.key, 999),
            record.key,
        )
    )
    status_parts: list[str] = []
    if current is not None:
        iteration = current.value.get("iteration")
        if isinstance(iteration, int) and not isinstance(iteration, bool):
            status_parts.append(f"loop iteration {iteration}")
        transition = current.value.get("last_transition")
        if isinstance(transition, str) and transition.strip():
            status_parts.append(f"last transition {transition.replace('_', ' ')}")
        summary = current.value.get("summary")
        if isinstance(summary, str) and summary.strip():
            status_parts.append(f"summary: {summary.strip()}")
    details = "; ".join(
        f"{record.key}: {_format_state_value(record.value)}"
        for record in loop_records
    )
    text_parts = [part for part in ("; ".join(status_parts), details) if part]
    source_records = [*loop_records]
    if current is not None:
        source_records.append(current)
    return (
        ContextSource(
            tier="current_scene",
            source_type="world_state",
            source_id=",".join(record.id for record in source_records),
            text="Current time-loop state: " + "; ".join(text_parts),
            reason="time-loop state",
            always_include=always_include,
        ),
    )


def _thread_source(
    threads: list[ActiveThreadRecord],
    *,
    always_include: bool = False,
) -> ContextSource:
    return ContextSource(
        tier="active_threads",
        source_type="active_thread",
        source_id=",".join(thread.id for thread in threads),
        text="Active threads: "
        + "; ".join(
            f"{thread.title} "
            f"({normalize_active_thread_status(thread.status)}, "
            f"priority {thread.priority}): {thread.description}"
            for thread in threads
        ),
        reason="active threads",
        always_include=always_include,
    )


def _context_active_threads(
    threads: list[ActiveThreadRecord],
    *,
    scene_snapshot: SceneSnapshotRecord | None,
    characters: list[CharacterRecord],
    focus_message: MessageRecord | None,
) -> list[ActiveThreadRecord]:
    latest_player_message = focus_message.body if focus_message is not None else ""
    reference_character_ids = character_scope_for_turn(
        scene_snapshot=scene_snapshot,
        characters=characters,
        latest_player_message=latest_player_message,
    ).reference_character_ids
    known_character_ids = frozenset(character.id for character in characters)
    visible: list[ActiveThreadRecord] = []
    for thread in threads:
        if not active_thread_is_prompt_visible(thread):
            continue
        if normalize_active_thread_visibility(thread.visibility) != "private":
            visible.append(thread)
            continue
        audience_ids = _active_thread_audience_character_ids(
            thread,
            known_character_ids=known_character_ids,
        )
        if audience_ids & reference_character_ids:
            visible.append(thread)
    return visible


def _active_thread_audience_character_ids(
    thread: ActiveThreadRecord,
    *,
    known_character_ids: frozenset[str],
) -> frozenset[str]:
    character_ids: set[str] = set()
    for item in thread.related_entities:
        if item in known_character_ids:
            character_ids.add(item)
            continue
        entity_type, separator, entity_id = item.partition(":")
        if (
            separator
            and entity_type == "character"
            and entity_id in known_character_ids
        ):
            character_ids.add(entity_id)
    return frozenset(character_ids)


def _location_text(location: LocationRecord, mode: str) -> str:
    parts = [f"Current location: {location.name}"]
    detail = location.visual_description if mode == "image" else location.description
    if detail:
        parts.append(detail)
    if location.status:
        parts.append(f"status: {location.status}")
    if location.hazards:
        parts.append("hazards: " + ", ".join(location.hazards))
    return "; ".join(parts)


def _characters_text(characters: list[CharacterRecord], mode: str) -> str:
    parts: list[str] = []
    for character in characters:
        if mode == "image":
            details = [character.name]
            details.extend(
                part
                for part in (
                    f"age: {character.age}" if character.age else "",
                    character.appearance,
                    character.visual_notes,
                    character.current_clothing,
                    f"status: {character.status}" if character.status else "",
                )
                if part
            )
        else:
            details = [_character_name_text(character)]
            details.extend(
                part
                for part in (
                    f"role: {character.role}" if character.role else "",
                    f"age: {character.age}" if character.age else "",
                    (
                        f"known state: {character.known_state}"
                        if character.known_state
                        else ""
                    ),
                    f"status: {character.status}" if character.status else "",
                    (
                        "appearance: "
                        + _compact_context_detail(character.appearance)
                        if character.appearance
                        else ""
                    ),
                    (
                        "visual notes: "
                        + _compact_context_detail(character.visual_notes)
                        if character.visual_notes
                        else ""
                    ),
                    (
                        "current clothing: "
                        + _compact_context_detail(character.current_clothing)
                        if character.current_clothing
                        else ""
                    ),
                    (
                        f"personality: {character.personality}"
                        if character.personality
                        else ""
                    ),
                    f"voice: {character.voice}" if character.voice else "",
                    (
                        "relationships: "
                        + _format_relationships(character.relationships)
                        if character.relationships
                        else ""
                    ),
                    *_character_agency_details(character),
                    (
                        "narrator-only private notes for this character; do not "
                        f"treat as known by other characters: {character.private_notes}"
                        if character.private_notes
                        else ""
                    ),
                )
                if part
            )
        parts.append(" - ".join(details))
    return "Present characters: " + "; ".join(parts)


def _character_agency_details(character: CharacterRecord) -> tuple[str, ...]:
    if character.is_player_character:
        return ()
    return tuple(
        detail
        for detail in (
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
        )
        if detail
    )


def _compact_context_detail(
    value: str,
    max_chars: int = CHARACTER_VISUAL_DETAIL_MAX_CHARS,
) -> str:
    compacted = " ".join(value.split())
    if len(compacted) <= max_chars:
        return compacted
    marker = " ..."
    if max_chars <= len(marker):
        return compacted[:max_chars].rstrip()
    return compacted[: max_chars - len(marker)].rstrip() + marker


def _character_name_text(character: CharacterRecord) -> str:
    if not character.aliases:
        return character.name
    return f"{character.name} (aliases: {', '.join(character.aliases)})"


def _format_relationships(value: dict[str, object]) -> str:
    return ", ".join(
        f"{key}: {_format_state_value(item)}"
        for key, item in sorted(value.items(), key=lambda item: item[0])
    )


def _message_sources(
    messages: tuple[MessageRecord, ...],
    *,
    tier: str,
) -> tuple[ContextSource, ...]:
    if not messages:
        return ()
    heading = (
        "Chronicle before the selected moment:"
        if tier == "chronicle_before_selected"
        else "Recent chronicle:"
    )
    return (
        ContextSource(
            tier=tier,
            source_type="message",
            source_id=",".join(message.id for message in messages),
            text=heading
            + "\n"
            + "\n".join(_format_message(message) for message in messages),
            reason=tier,
        ),
    )


def _image_context_messages(
    *,
    messages: list[MessageRecord],
    source_message_id: str | None,
) -> tuple[MessageRecord | None, tuple[MessageRecord, ...]]:
    if source_message_id is None:
        return None, tuple(messages[-8:])
    source_index = next(
        (
            index
            for index, message in enumerate(messages)
            if message.id == source_message_id
        ),
        None,
    )
    if source_index is None:
        raise ValueError(f"Unknown source message id: {source_message_id}")
    return (
        messages[source_index],
        tuple(messages[max(0, source_index - 7) : source_index]),
    )


def _prior_image_continuity_sources(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    messages: list[MessageRecord],
    source_message_id: str | None,
) -> tuple[ContextSource, ...]:
    message_positions = {message.id: index for index, message in enumerate(messages)}
    source_position: int
    if source_message_id is None:
        source_position = len(messages)
    elif source_message_id in message_positions:
        source_position = message_positions[source_message_id]
    else:
        return ()
    prior_assets = [
        asset
        for asset in repositories.list_media_assets(save_id)
        if asset.type == "image"
        and asset.status == "succeeded"
        and asset.source_message_id is not None
        and message_positions.get(asset.source_message_id, len(messages))
        < source_position
    ]
    if not prior_assets:
        return ()
    asset = max(
        enumerate(prior_assets),
        key=lambda item: (
            message_positions[item[1].source_message_id or ""],
            item[0],
        ),
    )[1]
    parts = [
        "Prior image continuity before selected moment:",
        f"source_message_id: {asset.source_message_id}",
        (
            "Reuse only stable visual continuity from the prior image, such as "
            "recurring character identity, persistent objects, and broad visual "
            "style. Do not copy its prior location, action, composition, "
            "weather, time of day, or one-off scene details unless the selected "
            "scene context repeats them."
        ),
    ]
    if asset.provider or asset.model:
        parts.append(
            "prior image model: "
            + "/".join(part for part in (asset.provider, asset.model) if part)
        )
    return (
        ContextSource(
            tier="prior_image_continuity",
            source_type="media_asset",
            source_id=asset.id,
            text="\n".join(parts),
            reason="latest prior image before selected moment",
        ),
    )


def _format_message(message: MessageRecord) -> str:
    speaker = message.speaker_name or message.role.title()
    return f"{speaker}: {message.body}"


def _positive_int_setting(value: object | None, default: int) -> int:
    if isinstance(value, int) and value > 0:
        return value
    return default


def _fraction_setting(value: object | None, default: float) -> float:
    if isinstance(value, int | float) and 0 < float(value) <= 1:
        return float(value)
    return default


def _budget_limit(settings: ContextBudgetSettings) -> int | None:
    if settings.mode == CONTEXT_BUDGET_MODE_FIXED_CHARS:
        return settings.fixed_total_chars
    if settings.mode == CONTEXT_BUDGET_MODE_ADAPTIVE_TIERS:
        return max(1, int(settings.fixed_total_chars * settings.adaptive_fraction))
    return None


def _section_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _format_state_value(value: object) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{key}: {item}" for key, item in value.items())
    return str(value)
