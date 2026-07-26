"""Import-safe scenario wizard view model."""

from __future__ import annotations

from dataclasses import dataclass

from bragi.services.scenario_service import (
    DATING_SIM_SECTIONS,
    FANTASY_ROLEPLAY_SECTIONS,
    FIRST_CONTACT_EXPLORATION_SECTIONS,
    FULL_ROLEPLAY_SECTIONS,
    HEIST_INFILTRATION_SECTIONS,
    INVESTIGATION_MYSTERY_SECTIONS,
    MERCHANT_TRADE_ROUTE_SECTIONS,
    MONSTER_HUNT_BOUNTY_SECTIONS,
    POLITICAL_INTRIGUE_SECTIONS,
    ROAD_TRIP_PILGRIMAGE_SECTIONS,
    SCIENCE_FICTION_ROLEPLAY_SECTIONS,
    SETTLEMENT_BUILDER_SECTIONS,
    SURVIVAL_EXPEDITION_SECTIONS,
    TIME_LOOP_SECTIONS,
    ScenarioType,
)


@dataclass(frozen=True)
class ScenarioReviewGroup:
    label: str
    section_ids: tuple[str, ...]


@dataclass(frozen=True)
class ScenarioFlowModel:
    flow_id: str
    label: str
    seed_prompt: str
    editable_section_ids: tuple[str, ...]
    review_groups: tuple[ScenarioReviewGroup, ...]


@dataclass(frozen=True)
class ScenarioWizardModel:
    flows: tuple[ScenarioFlowModel, ...]


def build_scenario_wizard_model() -> ScenarioWizardModel:
    return ScenarioWizardModel(
        flows=(
            ScenarioFlowModel(
                flow_id=ScenarioType.FULL_ROLEPLAY.value,
                label="Generic Roleplay",
                seed_prompt=(
                    "Describe the genre, premise, player role, tone, and "
                    "visible opening narration. Leave room for the world to "
                    "emerge in play."
                ),
                editable_section_ids=FULL_ROLEPLAY_SECTIONS,
                review_groups=(
                    ScenarioReviewGroup(
                        label="Core",
                        section_ids=(
                            "title",
                            "premise",
                            "player_character_name",
                            "player_role",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Opening",
                        section_ids=(
                            "tone_genre",
                            "opening_message",
                        ),
                    ),
                ),
            ),
            ScenarioFlowModel(
                flow_id=ScenarioType.FANTASY_ROLEPLAY.value,
                label="Fantasy",
                seed_prompt=(
                    "Describe the fantasy premise, player role, magic, realms, "
                    "factions, myths or creatures, quest stakes, tone, and "
                    "visible opening narration."
                ),
                editable_section_ids=FANTASY_ROLEPLAY_SECTIONS,
                review_groups=(
                    ScenarioReviewGroup(
                        label="Core",
                        section_ids=(
                            "title",
                            "premise",
                            "player_character_name",
                            "player_role",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Fantasy World",
                        section_ids=(
                            "magic_system",
                            "realms_and_places",
                            "factions_and_orders",
                            "myths_and_creatures",
                            "quest_stakes",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Opening",
                        section_ids=(
                            "tone_genre",
                            "opening_message",
                        ),
                    ),
                ),
            ),
            ScenarioFlowModel(
                flow_id=ScenarioType.SCIENCE_FICTION_ROLEPLAY.value,
                label="Science Fiction",
                seed_prompt=(
                    "Describe the science fiction premise, player role, "
                    "technology, setting scope, species or intelligences, "
                    "factions or institutions, mission stakes, tone, and "
                    "visible opening narration."
                ),
                editable_section_ids=SCIENCE_FICTION_ROLEPLAY_SECTIONS,
                review_groups=(
                    ScenarioReviewGroup(
                        label="Core",
                        section_ids=(
                            "title",
                            "premise",
                            "player_character_name",
                            "player_role",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Science Fiction World",
                        section_ids=(
                            "technology_level",
                            "setting_scope",
                            "species_and_intelligences",
                            "factions_and_institutions",
                            "mission_stakes",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Opening",
                        section_ids=(
                            "tone_genre",
                            "opening_message",
                        ),
                    ),
                ),
            ),
            ScenarioFlowModel(
                flow_id=ScenarioType.FIRST_CONTACT_EXPLORATION.value,
                label="First Contact / Exploration",
                seed_prompt=(
                    "Describe the first contact or exploration mission, unknown "
                    "world or anomaly, ship/base status, alien or "
                    "ambiguous intelligence, translation progress, discoveries, "
                    "hazards, tone, and visible opening narration."
                ),
                editable_section_ids=FIRST_CONTACT_EXPLORATION_SECTIONS,
                review_groups=(
                    ScenarioReviewGroup(
                        label="Core",
                        section_ids=(
                            "title",
                            "premise",
                            "player_character_name",
                            "player_role",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Mission",
                        section_ids=(
                            "mission_profile",
                            "ship_or_base_status",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Discovery",
                        section_ids=(
                            "exploration_target",
                            "knowledge_state",
                            "discoveries_and_samples",
                            "hazards_and_escalation",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Contact",
                        section_ids=(
                            "unknown_intelligence",
                            "translation_progress",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Opening",
                        section_ids=(
                            "tone_genre",
                            "opening_message",
                        ),
                    ),
                ),
            ),
            ScenarioFlowModel(
                flow_id=ScenarioType.SURVIVAL_EXPEDITION.value,
                label="Survival Expedition",
                seed_prompt=(
                    "Describe the survival expedition premise, player role, "
                    "goal, route options, supplies, environmental "
                    "conditions, hazards, camp status, travel progress, tone, "
                    "and visible opening narration."
                ),
                editable_section_ids=SURVIVAL_EXPEDITION_SECTIONS,
                review_groups=(
                    ScenarioReviewGroup(
                        label="Core",
                        section_ids=(
                            "title",
                            "premise",
                            "player_character_name",
                            "player_role",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Expedition",
                        section_ids=(
                            "expedition_goal",
                            "route_options",
                            "travel_progress",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Supplies",
                        section_ids=("resource_inventory",),
                    ),
                    ScenarioReviewGroup(
                        label="Conditions",
                        section_ids=(
                            "environmental_conditions",
                            "hazards_and_events",
                            "camp_status",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Opening",
                        section_ids=(
                            "tone_genre",
                            "opening_message",
                        ),
                    ),
                ),
            ),
            ScenarioFlowModel(
                flow_id=ScenarioType.TIME_LOOP.value,
                label="Time Loop",
                seed_prompt=(
                    "Describe the time loop premise, reset trigger, loop duration, "
                    "starting state, objective, failure conditions, baseline world "
                    "state, schedule, persistent knowledge, persistence exceptions, "
                    "NPC memory rules, tone, and visible opening narration."
                ),
                editable_section_ids=TIME_LOOP_SECTIONS,
                review_groups=(
                    ScenarioReviewGroup(
                        label="Core",
                        section_ids=(
                            "title",
                            "premise",
                            "player_character_name",
                            "player_role",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Loop Rules",
                        section_ids=(
                            "loop_premise",
                            "reset_trigger",
                            "loop_duration",
                            "objective",
                            "failure_conditions",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Reset State",
                        section_ids=(
                            "starting_state",
                            "baseline_world_state",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Schedule",
                        section_ids=(
                            "loop_schedule",
                            "current_loop_state",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Persistence",
                        section_ids=(
                            "persistent_knowledge",
                            "persistence_exceptions",
                            "npc_memory_rules",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Opening",
                        section_ids=(
                            "tone_genre",
                            "opening_message",
                        ),
                    ),
                ),
            ),
            ScenarioFlowModel(
                flow_id=ScenarioType.INVESTIGATION_MYSTERY.value,
                label="Investigation Mystery",
                seed_prompt=(
                    "Describe the mystery premise, case facts, clues, "
                    "timeline, red herrings, hidden truth, case status, tone, "
                    "and visible opening narration."
                ),
                editable_section_ids=INVESTIGATION_MYSTERY_SECTIONS,
                review_groups=(
                    ScenarioReviewGroup(
                        label="Core",
                        section_ids=(
                            "title",
                            "premise",
                            "player_character_name",
                            "player_role",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Case",
                        section_ids=(
                            "case_facts",
                            "case_status",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Evidence",
                        section_ids=(
                            "clues",
                            "timeline",
                            "red_herrings",
                            "hidden_truth",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Opening",
                        section_ids=(
                            "tone_genre",
                            "opening_message",
                        ),
                    ),
                ),
            ),
            ScenarioFlowModel(
                flow_id=ScenarioType.HEIST_INFILTRATION.value,
                label="Heist / Infiltration",
                seed_prompt=(
                    "Describe the heist or infiltration target, objectives, "
                    "intel, access, security model, alert or "
                    "heat state, loadout, complications, extraction, aftermath, "
                    "tone, and visible opening narration."
                ),
                editable_section_ids=HEIST_INFILTRATION_SECTIONS,
                review_groups=(
                    ScenarioReviewGroup(
                        label="Core",
                        section_ids=(
                            "title",
                            "premise",
                            "player_character_name",
                            "player_role",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Target & Objectives",
                        section_ids=(
                            "target_location",
                            "objectives_and_stakes",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Intel",
                        section_ids=("intel_and_access",),
                    ),
                    ScenarioReviewGroup(
                        label="Security",
                        section_ids=(
                            "security_model",
                            "alert_and_heat",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Tools & Complications",
                        section_ids=(
                            "loadout_and_tools",
                            "complications",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Exit & Consequences",
                        section_ids=(
                            "extraction_routes",
                            "aftermath",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Opening",
                        section_ids=(
                            "tone_genre",
                            "opening_message",
                        ),
                    ),
                ),
            ),
            ScenarioFlowModel(
                flow_id=ScenarioType.POLITICAL_INTRIGUE.value,
                label="Political Intrigue",
                seed_prompt=(
                    "Describe the political arena, factions, "
                    "central conflict, secrets, reputation or standing, favors "
                    "and obligations, alliances, timed political pressure, "
                    "public versus private knowledge, tone, and visible "
                    "opening narration."
                ),
                editable_section_ids=POLITICAL_INTRIGUE_SECTIONS,
                review_groups=(
                    ScenarioReviewGroup(
                        label="Core",
                        section_ids=(
                            "title",
                            "premise",
                            "player_character_name",
                            "player_role",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Arena",
                        section_ids=(
                            "political_arena",
                            "central_conflict",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Factions",
                        section_ids=(
                            "political_factions",
                            "alliances_and_rivalries",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Leverage",
                        section_ids=(
                            "secrets_and_leverage",
                            "reputation_and_standing",
                            "obligations_and_favors",
                            "public_private_knowledge",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Pressure",
                        section_ids=(
                            "event_calendar",
                            "political_pressure",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Opening",
                        section_ids=(
                            "tone_genre",
                            "opening_message",
                        ),
                    ),
                ),
            ),
            ScenarioFlowModel(
                flow_id=ScenarioType.SETTLEMENT_BUILDER.value,
                label="Settlement Builder",
                seed_prompt=(
                    "Describe the settlement premise, resources, "
                    "projects, facilities, threats, opportunities, calendar "
                    "pressure, tone, and visible opening narration."
                ),
                editable_section_ids=SETTLEMENT_BUILDER_SECTIONS,
                review_groups=(
                    ScenarioReviewGroup(
                        label="Core",
                        section_ids=(
                            "title",
                            "premise",
                            "player_character_name",
                            "player_role",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Community",
                        section_ids=("settlement_profile",),
                    ),
                    ScenarioReviewGroup(
                        label="Operations",
                        section_ids=(
                            "resources_and_indicators",
                            "projects_and_facilities",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Pressure",
                        section_ids=(
                            "threats_and_opportunities",
                            "calendar_and_deadlines",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Opening",
                        section_ids=(
                            "tone_genre",
                            "opening_message",
                        ),
                    ),
                ),
            ),
            ScenarioFlowModel(
                flow_id=ScenarioType.MONSTER_HUNT_BOUNTY.value,
                label="Monster Hunt / Bounty",
                seed_prompt=(
                    "Describe the hunt or bounty premise, target, clues, "
                    "locations, preparation state, current hunt status, "
                    "tone, and visible opening narration."
                ),
                editable_section_ids=MONSTER_HUNT_BOUNTY_SECTIONS,
                review_groups=(
                    ScenarioReviewGroup(
                        label="Core",
                        section_ids=(
                            "title",
                            "premise",
                            "player_character_name",
                            "player_role",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Hunt",
                        section_ids=(
                            "hunt_profile",
                            "target_profile",
                            "hunt_status",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Investigation",
                        section_ids=(
                            "leads_and_clues",
                            "hunt_locations",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Pressure",
                        section_ids=("preparation_state",),
                    ),
                    ScenarioReviewGroup(
                        label="Opening",
                        section_ids=(
                            "tone_genre",
                            "opening_message",
                        ),
                    ),
                ),
            ),
            ScenarioFlowModel(
                flow_id=ScenarioType.ROAD_TRIP_PILGRIMAGE.value,
                label="Road Trip / Pilgrimage",
                seed_prompt=(
                    "Describe the journey premise, route, stops, "
                    "transport, supplies, recurring pressures, "
                    "relationships, progress, tone, and visible opening "
                    "narration."
                ),
                editable_section_ids=ROAD_TRIP_PILGRIMAGE_SECTIONS,
                review_groups=(
                    ScenarioReviewGroup(
                        label="Core",
                        section_ids=(
                            "title",
                            "premise",
                            "player_character_name",
                            "player_role",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Journey",
                        section_ids=(
                            "journey_profile",
                            "route_and_stops",
                            "journey_progress",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Relationship Threads",
                        section_ids=("relationship_threads",),
                    ),
                    ScenarioReviewGroup(
                        label="Road Pressure",
                        section_ids=(
                            "transport_and_supplies",
                            "recurring_pressures",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Opening",
                        section_ids=(
                            "tone_genre",
                            "opening_message",
                        ),
                    ),
                ),
            ),
            ScenarioFlowModel(
                flow_id=ScenarioType.MERCHANT_TRADE_ROUTE.value,
                label="Merchant / Trade Route",
                seed_prompt=(
                    "Describe the trade premise, route, cargo, markets, "
                    "contracts, debts, route hazards, "
                    "profit and loss pressure, tone, and visible opening "
                    "narration."
                ),
                editable_section_ids=MERCHANT_TRADE_ROUTE_SECTIONS,
                review_groups=(
                    ScenarioReviewGroup(
                        label="Core",
                        section_ids=(
                            "title",
                            "premise",
                            "player_character_name",
                            "player_role",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Trade Route",
                        section_ids=(
                            "trade_profile",
                            "markets_and_stops",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Cargo & Contracts",
                        section_ids=(
                            "cargo_inventory",
                            "contracts_and_debts",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Risk & Standing",
                        section_ids=(
                            "route_hazards",
                            "profit_and_loss",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Opening",
                        section_ids=(
                            "tone_genre",
                            "opening_message",
                        ),
                    ),
                ),
            ),
            ScenarioFlowModel(
                flow_id=ScenarioType.DATING_SIM.value,
                label="Dating Sim",
                seed_prompt=(
                    "Describe the player character, dating sim premise, tone, "
                    "and visible opening narration."
                ),
                editable_section_ids=DATING_SIM_SECTIONS,
                review_groups=(
                    ScenarioReviewGroup(
                        label="Core",
                        section_ids=(
                            "title",
                            "premise",
                            "player_character_name",
                            "player_character_profile",
                            "player_role",
                        ),
                    ),
                    ScenarioReviewGroup(
                        label="Opening",
                        section_ids=(
                            "tone_genre",
                            "opening_message",
                        ),
                    ),
                ),
            ),
        )
    )
