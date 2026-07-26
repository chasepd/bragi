"""Scenario draft generation, editing, and persistence."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from time import perf_counter
from types import MappingProxyType

from bragi.app_logging import exception_log_fields, log_error_event, log_event
from bragi.content_rating_instructions import maximum_content_rating
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatMessage,
    ChatPromptPurpose,
    ChatRequest,
    ChatResponse,
    ProviderClient,
)
from bragi.redaction import redact_text
from bragi.services.action_choice_flags import (
    content_with_action_choices_enabled,
    normalize_legacy_action_choice_scenario,
)
from bragi.services.character_profile_completion import (
    ScenarioCharacterStarter,
    content_with_character_starters,
)
from bragi.services.content_rating import effective_content_safety_policy
from bragi.services.content_safety_service import ContentSafetyService
from bragi.services.job_lifecycle import JobLifecycleService
from bragi.services.model_preferences import (
    scenario_generation_section_model_preference,
)
from bragi.services.provider_fallbacks import chat_with_fallback
from bragi.services.scenario_content_rating import (
    metadata_with_scenario_content_ratings,
)
from bragi.services.scenario_name_sources import (
    ordinary_name_candidate_context,
    repeated_first_names_for_section,
)


class ScenarioType(StrEnum):
    FULL_ROLEPLAY = "full_roleplay"
    FANTASY_ROLEPLAY = "fantasy_roleplay"
    SCIENCE_FICTION_ROLEPLAY = "science_fiction_roleplay"
    FIRST_CONTACT_EXPLORATION = "first_contact_exploration"
    SURVIVAL_EXPEDITION = "survival_expedition"
    TIME_LOOP = "time_loop"
    INVESTIGATION_MYSTERY = "investigation_mystery"
    HEIST_INFILTRATION = "heist_infiltration"
    POLITICAL_INTRIGUE = "political_intrigue"
    SETTLEMENT_BUILDER = "settlement_builder"
    MONSTER_HUNT_BOUNTY = "monster_hunt_bounty"
    ROAD_TRIP_PILGRIMAGE = "road_trip_pilgrimage"
    MERCHANT_TRADE_ROUTE = "merchant_trade_route"
    DATING_SIM = "dating_sim"
    CHOOSE_YOUR_OWN_ADVENTURE = "choose_your_own_adventure"


SCENARIO_GENRES_CONTENT_KEY = "_scenario_genres"
RETIRED_SCENARIO_TYPE = "character_interaction"
RETIRED_SCENARIO_REASON = (
    "The character_interaction scenario type is no longer supported"
)
_OPENING_SECTION_IDS = ("tone_genre", "choice_style", "opening_message")
DEPRECATED_CHARACTER_LIST_SECTION_IDS = frozenset(
    {
        "characters",
        "romance_options",
        "suspects",
        "crew_and_command",
        "party_roster",
        "crew_and_contacts",
        "major_npcs",
        "population_and_residents",
        "rivals_and_factions",
        "traveling_party",
        "reputation_and_contacts",
    }
)
_DEPRECATED_FACTION_APPEND_SECTION_IDS = (
    "rivals_and_factions",
    "reputation_and_contacts",
)


FULL_ROLEPLAY_SECTIONS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "tone_genre",
    "opening_message",
)

FULL_ROLEPLAY_ALLOWED_SECTIONS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "choice_style",
    "worldbuilding",
    "lore",
    "locations",
    "factions",
    "tone_genre",
    "opening_message",
    "current_scene",
)

FANTASY_ROLEPLAY_SECTIONS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "magic_system",
    "realms_and_places",
    "factions_and_orders",
    "myths_and_creatures",
    "quest_stakes",
    "tone_genre",
    "opening_message",
)

FANTASY_ROLEPLAY_ALLOWED_SECTIONS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "choice_style",
    "magic_system",
    "realms_and_places",
    "factions_and_orders",
    "myths_and_creatures",
    "quest_stakes",
    "tone_genre",
    "opening_message",
    "current_scene",
)

SCIENCE_FICTION_ROLEPLAY_SECTIONS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "technology_level",
    "setting_scope",
    "species_and_intelligences",
    "factions_and_institutions",
    "mission_stakes",
    "tone_genre",
    "opening_message",
)

SCIENCE_FICTION_ROLEPLAY_ALLOWED_SECTIONS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "choice_style",
    "technology_level",
    "setting_scope",
    "species_and_intelligences",
    "factions_and_institutions",
    "mission_stakes",
    "tone_genre",
    "opening_message",
    "current_scene",
)

FIRST_CONTACT_EXPLORATION_SECTIONS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "mission_profile",
    "ship_or_base_status",
    "exploration_target",
    "unknown_intelligence",
    "knowledge_state",
    "translation_progress",
    "discoveries_and_samples",
    "hazards_and_escalation",
    "tone_genre",
    "opening_message",
)

FIRST_CONTACT_EXPLORATION_ALLOWED_SECTIONS = (
    *FIRST_CONTACT_EXPLORATION_SECTIONS,
    "choice_style",
    "current_scene",
)

SURVIVAL_EXPEDITION_SECTIONS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "expedition_goal",
    "route_options",
    "resource_inventory",
    "environmental_conditions",
    "hazards_and_events",
    "camp_status",
    "travel_progress",
    "tone_genre",
    "opening_message",
)

SURVIVAL_EXPEDITION_ALLOWED_SECTIONS = (
    *SURVIVAL_EXPEDITION_SECTIONS,
    "choice_style",
    "worldbuilding",
    "lore",
    "locations",
    "factions",
    "current_scene",
)

TIME_LOOP_SECTIONS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
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
    "tone_genre",
    "opening_message",
)

TIME_LOOP_ALLOWED_SECTIONS = (
    *TIME_LOOP_SECTIONS,
    "choice_style",
    "worldbuilding",
    "lore",
    "locations",
    "factions",
    "current_scene",
)

INVESTIGATION_MYSTERY_SECTIONS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "case_facts",
    "clues",
    "timeline",
    "red_herrings",
    "hidden_truth",
    "case_status",
    "tone_genre",
    "opening_message",
)

INVESTIGATION_MYSTERY_ALLOWED_SECTIONS = (
    *INVESTIGATION_MYSTERY_SECTIONS,
    "choice_style",
    "locations",
    "factions",
    "current_scene",
)

HEIST_INFILTRATION_SECTIONS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "target_location",
    "objectives_and_stakes",
    "intel_and_access",
    "security_model",
    "alert_and_heat",
    "loadout_and_tools",
    "complications",
    "extraction_routes",
    "aftermath",
    "tone_genre",
    "opening_message",
)

HEIST_INFILTRATION_ALLOWED_SECTIONS = (
    *HEIST_INFILTRATION_SECTIONS,
    "choice_style",
    "locations",
    "factions",
    "current_scene",
)

POLITICAL_INTRIGUE_SECTIONS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
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
    "tone_genre",
    "opening_message",
)

POLITICAL_INTRIGUE_ALLOWED_SECTIONS = (
    *POLITICAL_INTRIGUE_SECTIONS,
    "choice_style",
    "worldbuilding",
    "lore",
    "locations",
    "factions",
    "current_scene",
)

SETTLEMENT_BUILDER_SECTIONS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "settlement_profile",
    "resources_and_indicators",
    "projects_and_facilities",
    "threats_and_opportunities",
    "calendar_and_deadlines",
    "tone_genre",
    "opening_message",
)

SETTLEMENT_BUILDER_ALLOWED_SECTIONS = (
    *SETTLEMENT_BUILDER_SECTIONS,
    "choice_style",
    "worldbuilding",
    "lore",
    "locations",
    "factions",
    "current_scene",
)

MONSTER_HUNT_BOUNTY_SECTIONS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "hunt_profile",
    "target_profile",
    "leads_and_clues",
    "hunt_locations",
    "preparation_state",
    "hunt_status",
    "tone_genre",
    "opening_message",
)

MONSTER_HUNT_BOUNTY_ALLOWED_SECTIONS = (
    *MONSTER_HUNT_BOUNTY_SECTIONS,
    "choice_style",
    "locations",
    "factions",
    "current_scene",
)

ROAD_TRIP_PILGRIMAGE_SECTIONS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "journey_profile",
    "route_and_stops",
    "transport_and_supplies",
    "recurring_pressures",
    "relationship_threads",
    "journey_progress",
    "tone_genre",
    "opening_message",
)

ROAD_TRIP_PILGRIMAGE_ALLOWED_SECTIONS = (
    *ROAD_TRIP_PILGRIMAGE_SECTIONS,
    "choice_style",
    "worldbuilding",
    "lore",
    "locations",
    "factions",
    "current_scene",
)

MERCHANT_TRADE_ROUTE_SECTIONS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "trade_profile",
    "cargo_inventory",
    "markets_and_stops",
    "contracts_and_debts",
    "route_hazards",
    "profit_and_loss",
    "tone_genre",
    "opening_message",
)

MERCHANT_TRADE_ROUTE_ALLOWED_SECTIONS = (
    *MERCHANT_TRADE_ROUTE_SECTIONS,
    "choice_style",
    "worldbuilding",
    "lore",
    "locations",
    "factions",
    "current_scene",
)

DATING_SIM_SECTIONS = (
    "title",
    "premise",
    "player_character_name",
    "player_character_profile",
    "player_role",
    "tone_genre",
    "opening_message",
)

DATING_SIM_ALLOWED_SECTIONS = (
    *DATING_SIM_SECTIONS,
    "choice_style",
    "worldbuilding",
    "lore",
    "locations",
    "factions",
    "current_scene",
)

CHOOSE_YOUR_OWN_ADVENTURE_SECTIONS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "tone_genre",
    "choice_style",
    "opening_message",
)

CHOOSE_YOUR_OWN_ADVENTURE_ALLOWED_SECTIONS = (
    *CHOOSE_YOUR_OWN_ADVENTURE_SECTIONS,
    "worldbuilding",
    "lore",
    "locations",
    "factions",
    "current_scene",
)

_SECTION_GUIDANCE = {
    "title": "Write a short, evocative title.",
    "premise": (
        "Summarize the durable premise and dramatic promise. Do not write the "
        "first narrated scene beat or narrator prose; put that in the opening "
        "message."
    ),
    "player_character_name": (
        "Write the player character's in-world name only, or leave blank if the "
        "scenario should not prescribe one."
    ),
    "player_character_profile": (
        "Describe the generated player character as a durable in-world profile. "
        "Include gender/pronouns when specified or implied, personality, current "
        "situation, romantic availability, and the central choice pressure."
    ),
    "player_role": "Describe who the player is and why they matter.",
    "worldbuilding": (
        "Write a concise high-level world summary only when play has made it "
        "durably relevant."
    ),
    "lore": (
        "Write concise lore only when it has become relevant during play. Leave "
        "unknown history as useful blank space."
    ),
    "locations": (
        "Summarize important established places only; do not invent a travel guide."
    ),
    "factions": (
        "Summarize important established groups only; do not invent extra politics."
    ),
    "tone_genre": "Define the genre, mood, pacing, and content style.",
    "choice_style": (
        "Describe the flavor of the four action choices Bragi should suggest "
        "after each narrator beat: scope, specificity, risk level, tone, and "
        "how strongly they should diverge."
    ),
    "magic_system": (
        "Describe how magic works in play, including its sources, costs, limits, "
        "risks, social consequences, and what it cannot solve."
    ),
    "realms_and_places": (
        "Describe the most important fantasy realms, regions, and starting "
        "places that shape the opening situation."
    ),
    "factions_and_orders": (
        "Describe the fantasy factions, orders, courts, guilds, cults, or "
        "armies that create pressure in the scenario."
    ),
    "myths_and_creatures": (
        "Describe myths, monsters, spirits, gods, ancestral powers, or legendary "
        "creatures that may matter in play."
    ),
    "quest_stakes": (
        "Describe the central quest pressure, what is at risk, and why the "
        "player character matters now."
    ),
    "technology_level": (
        "Describe the available technology, scientific assumptions, constraints, "
        "failure modes, and what technology cannot conveniently solve."
    ),
    "setting_scope": (
        "Describe the science fiction setting scale: ship, station, city, "
        "planet, system, polity, or frontier, and the places that matter first."
    ),
    "species_and_intelligences": (
        "Describe important species, artificial intelligences, uploaded minds, "
        "posthumans, or other intelligences that shape play."
    ),
    "factions_and_institutions": (
        "Describe the corporations, governments, crews, research bodies, "
        "militaries, movements, or institutions creating pressure."
    ),
    "mission_stakes": (
        "Describe the central mission, investigation, survival pressure, or "
        "contact problem, including what happens if it goes wrong."
    ),
    "mission_profile": (
        "Describe the mission premise, objective, mandate, constraints, "
        "command limits, and what would count as a responsible success or "
        "failure."
    ),
    "ship_or_base_status": (
        "Describe the ship, station, habitat, shuttle, or field base status. "
        "Include systems, supplies, equipment damage, rescue windows, and "
        "operational constraints."
    ),
    "exploration_target": (
        "Describe the unknown world, site, anomaly, artifact, biosphere, or "
        "signal source. Separate known observations, hazards, resources, and "
        "unanswered questions."
    ),
    "unknown_intelligence": (
        "Describe any alien species, unknown intelligence, artificial mind, "
        "ecosystem behavior, or ambiguous agency. Include behaviors, contact "
        "attempts, cultural assumptions, relationship state, and what remains "
        "uncertain."
    ),
    "knowledge_state": (
        "Track partial knowledge in natural prose. Distinguish observed facts, "
        "hypotheses, misunderstood signals, confirmed knowledge, and unknowns "
        "without forcing premature exposition."
    ),
    "translation_progress": (
        "Track communication or understanding progress in natural prose. "
        "Include terms learned, open hypotheses, false assumptions, confirmed "
        "meanings, and what cannot yet be translated."
    ),
    "discoveries_and_samples": (
        "Describe scientific discoveries, samples, sensor findings, artifacts, "
        "research questions, contamination risks, and handling constraints."
    ),
    "hazards_and_escalation": (
        "Describe environmental danger, diplomatic tension, equipment damage, "
        "contamination risk, mission deadlines, rescue windows, and other "
        "escalation clocks."
    ),
    "case_facts": (
        "Describe the public case premise, victim or inciting event, stakes, "
        "jurisdiction, and facts available at the start. Do not reveal hidden "
        "truth or the culprit's secret plan here."
    ),
    "clues": (
        "Track starter clues in natural prose. Include source location, "
        "discovery status, reliability, and connections to suspects, events, "
        "or timeline contradictions."
    ),
    "timeline": (
        "Summarize public known events and hidden true events clearly enough "
        "for continuity, while preserving which parts are not known to the "
        "player."
    ),
    "red_herrings": (
        "Describe misleading evidence, false leads, or mistaken assumptions "
        "that should be tracked intentionally and reconciled consistently."
    ),
    "hidden_truth": (
        "Record the actual culprit, cause, secret sequence, or concealed "
        "answer. This truth is not known to the player at the start and must "
        "not be revealed prematurely in narration."
    ),
    "case_status": (
        "Describe the current investigation state, such as unresolved, "
        "narrowed, solved, falsely accused, escaped culprit, or cold case. "
        "Include what is known versus only suspected."
    ),
    "target_location": (
        "Describe the heist or infiltration target location in playable terms: "
        "layout, public and restricted areas, access points, chokepoints, cover, "
        "important rooms, valuable assets, and visible security posture."
    ),
    "objectives_and_stakes": (
        "Describe the primary objective, optional objectives, non-negotiable "
        "constraints, deadlines, and what success, partial success, or failure "
        "would change."
    ),
    "intel_and_access": (
        "Describe known intel, unknowns, access credentials, covers, schedules, "
        "routes, social openings, technical entry points, and assumptions that "
        "may be wrong."
    ),
    "security_model": (
        "Describe security in layers: guards, patrols, locks, alarms, cameras, "
        "wards, scanners, checkpoints, response teams, and likely escalation "
        "steps. Keep it concrete enough for tactical play."
    ),
    "alert_and_heat": (
        "Describe current suspicion, alarm state, witness risk, law enforcement "
        "or faction heat, and how noisy actions may escalate consequences."
    ),
    "loadout_and_tools": (
        "Describe important gear, disguises, credentials, transport, weapons, "
        "hacking tools, magical tools, supplies, limits, and what is unavailable "
        "or risky to use."
    ),
    "complications": (
        "Describe likely complications such as rival crews, unexpected audits, "
        "betrayal, moving targets, security upgrades, moral costs, hostages, "
        "timers, or environmental pressure."
    ),
    "extraction_routes": (
        "Describe primary and fallback exit plans, rally points, escape windows, "
        "transport, pursuit risks, and what conditions can close each route."
    ),
    "aftermath": (
        "Describe consequences after success, partial success, failure, capture, "
        "betrayal, collateral damage, or exposed identities. Include how heat "
        "or leverage may carry forward."
    ),
    "political_arena": (
        "Describe the political arena where play begins: court, council, "
        "guild, district, noble house, starship command, diplomatic summit, "
        "or other power structure. Include rules, visible institutions, and "
        "who can apply pressure there."
    ),
    "political_factions": (
        "Describe the main factions in natural prose. Include visible public "
        "positions, hidden goals where useful for continuity, resources, "
        "pressure points, rivalries, and alliances."
    ),
    "central_conflict": (
        "Describe the central political conflict, decision, succession fight, "
        "vote, negotiation, scandal, coup risk, or policy struggle that makes "
        "the scenario playable."
    ),
    "secrets_and_leverage": (
        "Track secrets, blackmail material, hidden loyalties, compromising "
        "evidence, rumors, and leverage. Mark what is public, suspected, "
        "private, or known only to specific NPCs."
    ),
    "reputation_and_standing": (
        "Describe the player's current reputation or standing with factions, "
        "social circles, and influential NPCs in explainable prose. Avoid "
        "opaque hidden meters."
    ),
    "obligations_and_favors": (
        "Track favors owed, favors held, debts, promises, bargains, blackmail "
        "terms, and outstanding obligations. Include who owes whom and what "
        "would settle or escalate the obligation."
    ),
    "alliances_and_rivalries": (
        "Describe current alliances, rivalries, fragile coalitions, likely "
        "betrayals, and relationship pressure between factions and NPCs."
    ),
    "event_calendar": (
        "Describe scheduled political events, votes, ceremonies, hearings, "
        "deadlines, scandals, negotiations, public appearances, or coup "
        "windows that should create timed pressure."
    ),
    "political_pressure": (
        "Describe at least one active timed political pressure, tension track, "
        "deadline, or public event that will force choices soon."
    ),
    "public_private_knowledge": (
        "Separate public knowledge from private knowledge. Record what the "
        "player knows, what specific NPCs or factions know, and what the "
        "narrator must not leak prematurely."
    ),
    "settlement_profile": (
        "Describe the settlement identity, location, theme, founding problem, "
        "leadership situation, and long-term objective in natural prose."
    ),
    "resources_and_indicators": (
        "Track important resources and high-level indicators in prose: food, "
        "supplies, money, tools, defenses, medicine, fuel, trade goods, "
        "reputation, morale, safety, prosperity, scarcity, and risks."
    ),
    "projects_and_facilities": (
        "Describe facilities or projects with status, costs, blockers, benefits, "
        "completion progress, who is responsible, and what choices can change."
    ),
    "threats_and_opportunities": (
        "Describe threats and opportunities such as raids, shortages, disease, "
        "politics, trade, discovery, migration, internal conflict, and timed "
        "openings."
    ),
    "calendar_and_deadlines": (
        "Describe seasonal state, project timing, harvests, travel windows, "
        "elections, festivals, deadlines, and the next pressure that will arrive."
    ),
    "hunt_profile": (
        "Describe the hunt or bounty premise, target type, known threat, patron, "
        "reward, stakes, moral pressure, and what would count as success."
    ),
    "target_profile": (
        "Describe the target's abilities, weaknesses, habits, signs, territory, "
        "current status, hidden truth, and what remains uncertain or unproven."
    ),
    "leads_and_clues": (
        "Track leads and clues in natural prose. Include source, discovery "
        "status, reliability, connection to the target, and what follow-up each "
        "lead enables."
    ),
    "hunt_locations": (
        "Describe locations tied to sightings, lairs, witnesses, victims, danger "
        "level, environmental conditions, and likely encounters."
    ),
    "preparation_state": (
        "Track preparation in prose: gear, research, traps, allies, debts, "
        "special requirements, missing tools, and costs of being unprepared."
    ),
    "hunt_status": (
        "Describe current outcome state such as unresolved, discovered, cornered, "
        "captured, killed, escaped, redeemed, exposed, paid, betrayed, or delayed."
    ),
    "journey_profile": (
        "Describe the journey premise, destination, reason for travel, deadline "
        "or pressure, and what arrival or failure would change."
    ),
    "route_and_stops": (
        "Describe the route, expected stops, detours, borders, travel constraints, "
        "opportunities, dangers, contacts, and unresolved threads at stops."
    ),
    "transport_and_supplies": (
        "Track transport and travel resources in prose: vehicle condition, mounts, "
        "tickets, supplies, money, fuel, documents, maps, and limits."
    ),
    "recurring_pressures": (
        "Describe recurring threats or pressures such as pursuers, weather, "
        "customs, borders, rival travelers, spiritual trials, and deadlines."
    ),
    "relationship_threads": (
        "Track relationship changes, promises, shared memories, conflicts, trust, "
        "resentments, and open interpersonal questions that should follow the party."
    ),
    "journey_progress": (
        "Describe current leg, distance or time remaining, delays, detours, "
        "destination status, current stop, and immediate travel decision."
    ),
    "trade_profile": (
        "Describe the trade premise, operating region, vehicle, caravan or ship, "
        "home base, business goal, constraints, and commercial pressure."
    ),
    "cargo_inventory": (
        "Track cargo, special goods, passengers, documents, legal status, storage "
        "constraints, losses, spoilage, and what inventory decisions matter."
    ),
    "markets_and_stops": (
        "Describe markets, ports, stops, demand, supply, contacts, local risks, "
        "known price tendencies, tariffs, and what each stop can change."
    ),
    "contracts_and_debts": (
        "Track contracts, patrons, deadlines, penalties, delivery terms, favors, "
        "debts, collateral, and what would fulfill or breach each obligation."
    ),
    "route_hazards": (
        "Describe route hazards such as weather, pirates, inspections, tariffs, "
        "bandits, blockades, shortages, rivals, and timed risks."
    ),
    "profit_and_loss": (
        "Describe profit, loss, margins, debt pressure, risk exposure, known price "
        "changes, and economic consequences transparently in prose."
    ),
    "expedition_goal": (
        "Describe the expedition objective, destination or rescue condition, "
        "why it matters, and what success or failure would change."
    ),
    "route_options": (
        "Describe the viable routes, detours, retreat paths, landmarks, travel "
        "constraints, and tradeoffs the party can understand at the start."
    ),
    "resource_inventory": (
        "Track important supplies and equipment in prose: food, water, shelter, "
        "tools, medicine, transport, weapons, signal gear, condition, scarcity, "
        "and risks of loss or depletion."
    ),
    "environmental_conditions": (
        "Describe the current weather, terrain, exposure, season, visibility, "
        "temperature, water access, and other environmental pressures that shape "
        "survival decisions."
    ),
    "hazards_and_events": (
        "Describe active and plausible hazards, including terrain, predators, "
        "illness, injury, hostile forces, equipment failures, disasters, and "
        "time-sensitive events."
    ),
    "camp_status": (
        "Describe camp, shelter, watch, fire, security, rest, sanitation, and "
        "recovery conditions when relevant; otherwise describe why there is no "
        "safe camp yet."
    ),
    "travel_progress": (
        "Describe current route progress, distance or landmarks reached, delays, "
        "detours, retreat status, fatigue, and what immediate travel decision is "
        "pressing."
    ),
    "loop_premise": (
        "Describe the time-loop premise in natural prose: what repeats, who or "
        "what is caught in the loop, what remains mysterious, and why the loop "
        "creates playable pressure."
    ),
    "reset_trigger": (
        "Describe the event, condition, death, deadline, ritual, device, or "
        "choice that resets the loop. Include whether the reset trigger is known "
        "to the player at the start."
    ),
    "loop_duration": (
        "Describe how long each loop lasts, how time is measured, and any known "
        "phase boundaries or deadline windows."
    ),
    "starting_state": (
        "Describe the reset starting conditions: location, time, character "
        "positions, inventory, immediate situation, and what the player reliably "
        "returns to at the start of each loop."
    ),
    "objective": (
        "Describe the main loop objective and success condition clearly enough "
        "to guide repeated attempts without solving the scenario up front."
    ),
    "failure_conditions": (
        "Describe what counts as failure or forced reset, including death, missed "
        "deadlines, irreversible outcomes, wrong choices, or escalating loop costs."
    ),
    "baseline_world_state": (
        "Describe resettable baseline world state: places, NPC routines, item "
        "positions, relationships, public facts, and other conditions restored at "
        "the beginning of each loop."
    ),
    "loop_schedule": (
        "Describe scheduled events, NPC routines, known windows of opportunity, "
        "hidden events, deadlines, and repeatable timing clues in natural prose."
    ),
    "persistent_knowledge": (
        "Track knowledge that persists for the player/meta layer across loops. "
        "Distinguish player/meta knowledge from in-world character memory when "
        "the premise separates them, and do not imply NPCs reset with this knowledge."
    ),
    "persistence_exceptions": (
        "Describe anything that may persist across resets besides player/meta "
        "knowledge: artifacts, marks, unlocked locations, outside observers, "
        "relationship changes, injuries, or other exceptions allowed by the premise."
    ),
    "npc_memory_rules": (
        "Define who remembers what after a reset. State that NPCs reset to their "
        "baseline memories unless the premise explicitly marks an exception, and "
        "separate in-world character memory from player/meta knowledge."
    ),
    "current_loop_state": (
        "Track loop counter, current phase, prior-loop summary, important "
        "deviations from baseline, active discoveries, and any persistent changes."
    ),
    "opening_message": (
        "Write the narrator's first in-character message to the player. This is "
        "the only generated setup field that becomes visible chronicle text."
    ),
    "character_name": "Write the character's name only.",
    "character_description": (
        "Describe the character's role, situation, history, and non-visual context."
    ),
    "character_physical_description": (
        "Write a detailed reusable physical appearance description for character "
        "consistency in prose and image prompts. Include age impression, build, "
        "skin tone, complexion or undertone, face shape and features, hair color "
        "and texture, eyes, distinctive visual traits, clothing or styling, "
        "posture/body language, and overall presence. Preserve explicitly "
        "provided or clearly established race, ethnicity, ancestry, or cultural "
        "appearance details when the seed or imported character data includes "
        "them; do not omit those details from the physical description."
    ),
    "character_personality": (
        "Describe the character's motivations, habits, and emotional texture."
    ),
    "character_voice": (
        "Describe how the character speaks and what their dialogue feels like."
    ),
    "relationship_seed": (
        "Describe the character's current emotional baseline toward the player "
        "and the relationship tension, trust, affection, or distance at the start."
    ),
    "current_scene": "Summarize the durable current scene after play has moved on.",
}


@dataclass(frozen=True)
class ScenarioDraft:
    type: ScenarioType
    sections: Mapping[str, str]
    scenario_types: tuple[ScenarioType, ...] = ()
    metadata: Mapping[str, object] | None = None
    regeneration_seed: str = ""
    action_choices_enabled: bool = False
    character_starters: tuple[ScenarioCharacterStarter, ...] = ()

    def __post_init__(self) -> None:
        normalized_type = ScenarioType(self.type)
        normalized_genres = _validate_scenario_type_tuple(
            normalized_type,
            self.scenario_types or (normalized_type,),
        )
        object.__setattr__(self, "type", normalized_type)
        object.__setattr__(self, "scenario_types", normalized_genres)
        object.__setattr__(
            self,
            "sections",
            MappingProxyType(dict(self.sections)),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata or {})),
        )
        object.__setattr__(
            self,
            "character_starters",
            tuple(self.character_starters),
        )

    @property
    def title(self) -> str:
        return self.sections["title"]

    @property
    def premise(self) -> str:
        return self.sections.get("premise") or self.sections.get("setup_line", "")

    @property
    def player_role(self) -> str:
        return self.sections["player_role"]

    @property
    def player_character_name(self) -> str:
        return self.sections.get("player_character_name", "")


@dataclass(frozen=True)
class ScenarioGenerationProgress:
    scenario_type: ScenarioType
    section_id: str
    status: str
    completed_sections: Mapping[str, str]
    completed_count: int
    total_count: int
    scenario_types: tuple[ScenarioType, ...] = ()
    action_choices_enabled: bool = False
    error: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scenario_types",
            _validate_scenario_type_tuple(
                self.scenario_type,
                self.scenario_types or (self.scenario_type,),
            ),
        )
        object.__setattr__(
            self,
            "completed_sections",
            MappingProxyType(dict(self.completed_sections)),
        )


ScenarioGenerationProgressCallback = Callable[
    [ScenarioGenerationProgress],
    Awaitable[None] | None,
]


@dataclass(frozen=True)
class ScenarioSectionGenerationResult:
    body: str
    minimum_rating: str


class ScenarioService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        provider: ProviderClient,
        provider_name: str,
        model_id: str,
        providers: dict[str, ProviderClient] | None = None,
        current_user_id: str | None = None,
        content_safety_service: ContentSafetyService | None = None,
    ) -> None:
        self.repositories = repositories
        self.provider = provider
        self.provider_name = provider_name
        self.model_id = model_id
        self.providers = providers or {provider_name: provider}
        self.current_user_id = current_user_id
        self.content_safety_service = content_safety_service or ContentSafetyService(
            repositories=repositories,
            providers=self.providers,
        )
        self.jobs = JobLifecycleService(repositories=repositories)

    async def generate_draft(
        self,
        *,
        scenario_type: ScenarioType | str,
        scenario_types: Iterable[ScenarioType | str] | None = None,
        seed: str,
        action_choices_enabled: bool = False,
        section_ids: tuple[str, ...] | None = None,
        metadata: Mapping[str, object] | None = None,
        progress_callback: ScenarioGenerationProgressCallback | None = None,
    ) -> ScenarioDraft:
        normalized_type, normalized_genres, action_choices_enabled = (
            normalized_scenario_types_and_flag(
                scenario_type,
                scenario_types=scenario_types,
                action_choices_enabled=action_choices_enabled,
            )
        )
        resolved_section_ids = (
            section_ids
            if section_ids is not None
            else _generated_section_ids(
                normalized_type,
                scenario_types=normalized_genres,
                action_choices_enabled=action_choices_enabled,
            )
        )
        _validate_section_ids(
            normalized_type,
            resolved_section_ids,
            scenario_types=normalized_genres,
        )
        job = self.jobs.create_running(
            type="scenario_generation",
            payload={
                "scenario_type": normalized_type.value,
                "scenario_types": [genre.value for genre in normalized_genres],
                "seed_chars": len(seed),
                "provider": self.provider_name,
                "model": self.model_id,
                "section_model_overrides": self._section_model_override_payload(
                    resolved_section_ids
                ),
            },
            collect_provider_diagnostics=True,
        )
        sections: dict[str, str] = {}
        section_content_ratings: dict[str, str] = {}
        log_event(
            "scenario_generation.started",
            scenario_type=normalized_type.value,
            scenario_types=tuple(genre.value for genre in normalized_genres),
            seed_chars=len(seed),
            provider=self.provider_name,
            model=self.model_id,
        )
        try:
            for section_id in resolved_section_ids:
                await _notify_generation_progress(
                    progress_callback,
                    ScenarioGenerationProgress(
                        scenario_type=normalized_type,
                        scenario_types=normalized_genres,
                        action_choices_enabled=action_choices_enabled,
                        section_id=section_id,
                        status="generating",
                        completed_sections=sections,
                        completed_count=len(sections),
                        total_count=len(resolved_section_ids),
                    ),
                )
                section_result = await self._generate_section(
                    scenario_type=normalized_type,
                    scenario_types=normalized_genres,
                    action_choices_enabled=action_choices_enabled,
                    seed=seed,
                    section_id=section_id,
                    sections=sections,
                )
                section_value = section_result.body
                sections[section_id] = section_value
                section_content_ratings[section_id] = section_result.minimum_rating
                log_event(
                    "scenario_generation.field_succeeded",
                    scenario_type=normalized_type.value,
                    scenario_types=tuple(genre.value for genre in normalized_genres),
                    section_id=section_id,
                    generated_chars=len(section_value),
                    completed_fields=len(sections),
                )
                await _notify_generation_progress(
                    progress_callback,
                    ScenarioGenerationProgress(
                        scenario_type=normalized_type,
                        scenario_types=normalized_genres,
                        action_choices_enabled=action_choices_enabled,
                        section_id=section_id,
                        status="completed",
                        completed_sections=sections,
                        completed_count=len(sections),
                        total_count=len(resolved_section_ids),
                    ),
                )
        except Exception as exc:
            self.jobs.fail(
                job.id,
                error=redact_text(str(exc)) or exc.__class__.__name__,
            )
            await _notify_generation_progress(
                progress_callback,
                ScenarioGenerationProgress(
                    scenario_type=normalized_type,
                    scenario_types=normalized_genres,
                    action_choices_enabled=action_choices_enabled,
                    section_id=section_id,
                    status="failed",
                    completed_sections=sections,
                    completed_count=len(sections),
                    total_count=len(resolved_section_ids),
                    error=redact_text(str(exc)) or exc.__class__.__name__,
                ),
            )
            raise
        self.jobs.succeed(
            job.id,
            result={"field_count": len(sections)},
        )
        log_event(
            "scenario_generation.succeeded",
            scenario_type=normalized_type.value,
            scenario_types=tuple(genre.value for genre in normalized_genres),
            field_count=len(sections),
        )
        draft_metadata = metadata_with_scenario_content_ratings(
            _draft_metadata_with_generation_prompt(metadata, seed),
            aggregate_rating=maximum_content_rating(
                tuple(section_content_ratings.values())
            ),
            section_ratings=section_content_ratings,
        )
        return ScenarioDraft(
            type=normalized_type,
            scenario_types=normalized_genres,
            sections=sections,
            metadata=draft_metadata,
            regeneration_seed=seed,
            action_choices_enabled=action_choices_enabled,
            character_starters=(),
        )

    async def regenerate_section(
        self,
        *,
        scenario_type: ScenarioType | str,
        scenario_types: Iterable[ScenarioType | str] | None = None,
        seed: str,
        section_id: str,
        sections: Mapping[str, str],
        action_choices_enabled: bool = False,
    ) -> ScenarioSectionGenerationResult:
        normalized_type, normalized_genres, action_choices_enabled = (
            normalized_scenario_types_and_flag(
                scenario_type,
                scenario_types=scenario_types,
                action_choices_enabled=action_choices_enabled,
            )
        )
        if section_id not in _allowed_section_ids(
            normalized_type,
            scenario_types=normalized_genres,
        ):
            raise ValueError(f"Unknown scenario section: {section_id}")
        provider_name, model_id = self._effective_section_model(section_id)
        job = self.jobs.create_running(
            type="scenario_section_generation",
            payload={
                "scenario_type": normalized_type.value,
                "scenario_types": [genre.value for genre in normalized_genres],
                "section_id": section_id,
                "seed_chars": len(seed),
                "provider": provider_name,
                "model": model_id,
                "default_provider": self.provider_name,
                "default_model": self.model_id,
            },
            collect_provider_diagnostics=True,
        )
        try:
            section_result = await self._generate_section(
                scenario_type=normalized_type,
                scenario_types=normalized_genres,
                action_choices_enabled=action_choices_enabled,
                seed=seed,
                section_id=section_id,
                sections={
                    key: value
                    for key, value in sections.items()
                    if key != section_id
                },
            )
        except Exception as exc:
            self.jobs.fail(
                job.id,
                error=redact_text(str(exc)) or exc.__class__.__name__,
            )
            raise
        self.jobs.succeed(
            job.id,
            result={
                "section_id": section_id,
                "generated_chars": len(section_result.body),
            },
        )
        log_event(
            "scenario_generation.field_regenerated",
            scenario_type=normalized_type.value,
            scenario_types=tuple(genre.value for genre in normalized_genres),
            section_id=section_id,
            generated_chars=len(section_result.body),
        )
        return section_result

    async def _generate_section(
        self,
        *,
        scenario_type: ScenarioType,
        scenario_types: tuple[ScenarioType, ...],
        action_choices_enabled: bool,
        seed: str,
        section_id: str,
        sections: Mapping[str, str],
    ) -> ScenarioSectionGenerationResult:
        started_at = perf_counter()
        provider_name, model_id = self._effective_section_model(section_id)
        if provider_name not in self.providers:
            raise ValueError(f"Scenario provider is unavailable: {provider_name}")
        content_safety = effective_content_safety_policy(
            self.repositories,
            user_id=self.current_user_id,
        )
        request = ChatRequest(
            provider=provider_name,
            model_id=model_id,
            prompt_purpose=ChatPromptPurpose.SCENARIO_GENERATION,
            messages=(
                ChatMessage(
                    role="system",
                    body=_generation_instruction(
                        scenario_type,
                        scenario_types=scenario_types,
                        action_choices_enabled=action_choices_enabled,
                    ),
                ),
                ChatMessage(
                    role="player",
                    body=_section_generation_prompt(
                        scenario_type=scenario_type,
                        scenario_types=scenario_types,
                        action_choices_enabled=action_choices_enabled,
                        section_id=section_id,
                        seed=seed,
                        sections=sections,
                    ),
                ),
            ),
            content_rating=content_safety.rating,
            fade_to_black_enabled=content_safety.fade_to_black_enabled,
        )
        response = await self._chat_for_section(
            request=request,
            scenario_type=scenario_type,
            section_id=section_id,
            started_at=started_at,
        )
        section_value = _section_value_from_response(
            response.body,
            scenario_type=scenario_type,
            scenario_types=scenario_types,
            section_id=section_id,
        )
        safety = await self.content_safety_service.review_narration(
            body=section_value,
            source_request=replace(
                request,
                provider=response.provider,
                model_id=response.model_id,
            ),
            content_rating=content_safety.rating,
            fade_to_black_enabled=content_safety.fade_to_black_enabled,
            roleplay_type=scenario_type.value,
        )
        section_value = safety.body
        repeated_names = repeated_first_names_for_section(
            scenario_type=scenario_types or (scenario_type,),
            section_id=section_id,
            text=section_value,
        )
        if repeated_names:
            retry_request = replace(
                request,
                messages=(
                    *request.messages,
                    ChatMessage(
                        role="player",
                        body=_name_dedup_retry_prompt(
                            repeated_names=repeated_names,
                            previous_value=section_value,
                        ),
                    ),
                ),
            )
            retry_response = await self._chat_for_section(
                request=retry_request,
                scenario_type=scenario_type,
                section_id=section_id,
                started_at=started_at,
            )
            retry_value = _section_value_from_response(
                retry_response.body,
                scenario_type=scenario_type,
                scenario_types=scenario_types,
                section_id=section_id,
            )
            retry_safety = await self.content_safety_service.review_narration(
                body=retry_value,
                source_request=replace(
                    retry_request,
                    provider=retry_response.provider,
                    model_id=retry_response.model_id,
                ),
                content_rating=content_safety.rating,
                fade_to_black_enabled=content_safety.fade_to_black_enabled,
                roleplay_type=scenario_type.value,
            )
            return ScenarioSectionGenerationResult(
                body=retry_safety.body,
                minimum_rating=retry_safety.reviewed_content_rating,
            )
        return ScenarioSectionGenerationResult(
            body=section_value,
            minimum_rating=safety.reviewed_content_rating,
        )

    async def _chat_for_section(
        self,
        *,
        request: ChatRequest,
        scenario_type: ScenarioType,
        section_id: str,
        started_at: float,
    ) -> ChatResponse:
        try:
            response = await chat_with_fallback(
                repositories=self.repositories,
                providers=self.providers,
                request=request,
                task="scenario_generation",
                diagnostic_context={"section_id": section_id},
            )
        except Exception as exc:
            log_error_event(
                "provider.chat_failed",
                provider=request.provider,
                model=request.model_id,
                task="scenario_generation",
                scenario_type=scenario_type.value,
                section_id=section_id,
                duration_ms=_elapsed_ms(started_at),
                **exception_log_fields(exc),
            )
            raise
        log_event(
            "provider.chat_succeeded",
            provider=response.provider,
            model=response.model_id,
            task="scenario_generation",
            scenario_type=scenario_type.value,
            section_id=section_id,
            duration_ms=_elapsed_ms(started_at),
            message_count=len(request.messages),
            response_chars=len(response.body),
            token_usage=response.token_usage,
        )
        return response

    def _effective_section_model(self, section_id: str) -> tuple[str, str]:
        preference = scenario_generation_section_model_preference(
            self.repositories,
            section_id=section_id,
        )
        if preference is None:
            return self.provider_name, self.model_id
        return preference.provider, preference.model_id

    def _section_model_override_payload(
        self,
        section_ids: tuple[str, ...],
    ) -> dict[str, dict[str, str]]:
        overrides: dict[str, dict[str, str]] = {}
        for section_id in section_ids:
            preference = scenario_generation_section_model_preference(
                self.repositories,
                section_id=section_id,
            )
            if preference is not None:
                overrides[section_id] = {
                    "provider": preference.provider,
                    "model": preference.model_id,
                }
        return overrides

    def apply_edits(
        self,
        draft: ScenarioDraft,
        edits: dict[str, str],
    ) -> ScenarioDraft:
        allowed = set(
            _allowed_section_ids(draft.type, scenario_types=draft.scenario_types)
        )
        unknown = set(edits) - allowed
        if unknown:
            raise ValueError(f"Unknown scenario section edits: {sorted(unknown)}")
        return ScenarioDraft(
            type=draft.type,
            scenario_types=draft.scenario_types,
            sections={**draft.sections, **edits},
            metadata=draft.metadata,
            regeneration_seed=draft.regeneration_seed,
            action_choices_enabled=draft.action_choices_enabled,
            character_starters=draft.character_starters,
        )

    def save_draft(self, draft: ScenarioDraft) -> str:
        normalized_payload = normalize_scenario_draft_sections(
            draft.type,
            draft.sections,
        )
        sections = _select_sections(
            draft.type,
            normalized_payload,
            scenario_types=draft.scenario_types,
            action_choices_enabled=draft.action_choices_enabled,
        )
        allowed = set(
            _allowed_section_ids(draft.type, scenario_types=draft.scenario_types)
        )
        extra = set(normalized_payload) - allowed
        if extra:
            raise ValueError(f"Scenario draft has unknown sections: {sorted(extra)}")
        content = content_with_action_choices_enabled(
            _scenario_content_with_metadata(
                _content_with_scenario_genres(sections, draft.scenario_types),
                draft.metadata,
            ),
            enabled=draft.action_choices_enabled,
        )
        scenario = self.repositories.create_scenario(
            type=draft.type.value,
            title=sections["title"],
            premise=sections.get("premise", ""),
            player_role=sections["player_role"],
            content=content_with_character_starters(
                scenario_type=draft.type.value,
                content=content,
                starters=draft.character_starters,
            ),
        )
        log_event(
            "scenario.saved",
            scenario_id=scenario.id,
            scenario_type=draft.type.value,
            scenario_types=tuple(genre.value for genre in draft.scenario_types),
            section_count=len(sections),
        )
        return scenario.id

def _generation_instruction(
    scenario_type: ScenarioType,
    *,
    scenario_types: tuple[ScenarioType, ...] = (),
    action_choices_enabled: bool = False,
) -> str:
    normalized_genres = _validate_scenario_type_tuple(
        scenario_type,
        scenario_types or (scenario_type,),
    )
    scenario_label = _scenario_type_label_for_genres(normalized_genres)
    if scenario_type is ScenarioType.FULL_ROLEPLAY:
        scope = (
            " For generic roleplay scenarios, seed only what is needed to start "
            "play; leave nonessential world details to emerge through play."
        )
    elif scenario_type is ScenarioType.FANTASY_ROLEPLAY:
        scope = (
            " For fantasy roleplay scenarios, establish the magic, realms, "
            "factions, myths, and quest pressure needed to start play, while "
            "leaving room for discoveries and legends to emerge later."
        )
    elif scenario_type is ScenarioType.SCIENCE_FICTION_ROLEPLAY:
        scope = (
            " For science fiction roleplay scenarios, establish the technology "
            "constraints, setting scale, intelligences, institutions, and "
            "mission pressure needed to start play, while leaving room for "
            "unknowns and discoveries."
        )
    elif scenario_type is ScenarioType.FIRST_CONTACT_EXPLORATION:
        scope = (
            " For first-contact and exploration scenarios, establish the mission, "
            "crew, ship or base constraints, unknown environment, contact problem, "
            "partial knowledge, translation progress, discoveries, hazards, and "
            "escalation pressure needed to start play. Preserve uncertainty and "
            "avoid explaining unknown intelligences or discoveries before play "
            "reveals them."
        )
    elif scenario_type is ScenarioType.SURVIVAL_EXPEDITION:
        scope = (
            " For survival expedition scenarios, establish the expedition goal, "
            "route options, party status, resources, environmental pressure, "
            "hazards, camp status, and travel progress needed to start play, "
            "while leaving room for discovery and hard choices."
        )
    elif scenario_type is ScenarioType.TIME_LOOP:
        scope = (
            " For time loop scenarios, establish the loop premise, reset trigger, "
            "duration, starting state, objective, failure conditions, baseline "
            "world state, schedule, persistent player/meta knowledge, persistence "
            "exceptions, NPC memory rules, and current loop state needed to start "
            "play. Preserve a strict boundary between resettable world facts and "
            "knowledge or exceptions that persist across loops."
        )
    elif scenario_type is ScenarioType.INVESTIGATION_MYSTERY:
        scope = (
            " For investigation mystery scenarios, establish one central case, "
            "a small suspect set, clues, timeline pressure, red herrings, hidden "
            "truth, and case status. Preserve a strict boundary between known "
            "facts and hidden truth; do not reveal hidden truth in opening "
            "narration."
        )
    elif scenario_type is ScenarioType.HEIST_INFILTRATION:
        scope = (
            " For heist and infiltration scenarios, establish the target, "
            "objectives, crew, contacts, intel, access paths, security layers, "
            "alert or heat state, loadout, complications, extraction routes, and "
            "aftermath consequences needed to start play. Preserve tactical "
            "uncertainty; do not solve the job in setup or reveal every hidden "
            "security response before play discovers it."
        )
    elif scenario_type is ScenarioType.POLITICAL_INTRIGUE:
        scope = (
            " For political intrigue scenarios, establish a political arena, "
            "factions, key NPCs, central conflict, secrets, reputation or "
            "standing, obligations and favors, shifting alliances, a timed "
            "event calendar, and public versus private knowledge. Preserve "
            "secret boundaries and make reputation, favors, and alliance "
            "consequences explainable in prose rather than opaque meters."
        )
    elif scenario_type is ScenarioType.SETTLEMENT_BUILDER:
        scope = (
            " For settlement builder scenarios, establish the settlement "
            "identity, residents, resources, indicators, projects, facilities, "
            "threats, opportunities, calendar pressure, and long-term objective "
            "needed to start play. Keep mechanics lightweight and express "
            "projects, resources, morale, safety, and prosperity in explainable "
            "prose rather than opaque simulation."
        )
    elif scenario_type is ScenarioType.MONSTER_HUNT_BOUNTY:
        scope = (
            " For monster hunt or bounty campaign scenarios, establish the hunt "
            "premise, target, known threat, leads, clues, locations, rivals, "
            "preparation state, reward, stakes, and current hunt status. Preserve "
            "the boundary between discovered evidence and hidden truth, and do "
            "not solve the hunt during setup."
        )
    elif scenario_type is ScenarioType.ROAD_TRIP_PILGRIMAGE:
        scope = (
            " For road trip or pilgrimage scenarios, establish the destination, "
            "reason for travel, route, stops, traveling party, transport, "
            "supplies, recurring pressures, relationship threads, and current "
            "journey progress. Keep the route broad enough for detours and make "
            "relationships and prior stops durable."
        )
    elif scenario_type is ScenarioType.MERCHANT_TRADE_ROUTE:
        scope = (
            " For merchant and trade route scenarios, establish the trade "
            "premise, operating region, cargo, markets, contracts, debts, route "
            "hazards, reputation, contacts, and profit or loss pressure. Keep "
            "the economy lightweight and make risk, obligation, and reputation "
            "consequences transparent in prose."
        )
    elif scenario_type is ScenarioType.DATING_SIM:
        scope = (
            " For dating sim scenarios, create a central player character and a "
            "cast of romance options with distinct romantic routes, while "
            "preserving player agency and avoiding a predetermined choice."
        )
    else:
        scope = (
            " For choose your own adventure scenarios, write vivid book-like "
            "setup that creates concrete changed situations for player choices. "
            "The narrator text must not include numbered options; Bragi "
            "generates action choices in a separate structured step."
        )
    if (
        action_choices_enabled
        and scenario_type is not ScenarioType.CHOOSE_YOUR_OWN_ADVENTURE
    ):
        scope += (
            " Action choices are enabled, so opening narrator text should "
            "create a concrete changed situation with immediate pressure or "
            "opportunity for choices without including numbered options or "
            "action-choice lists."
        )
    return (
        f"You are helping draft a {scenario_label} for Bragi. "
        "Generate only the requested field as natural prose. "
        "Do not include JSON, Markdown fences, headings, labels, or commentary."
        f"{scope}"
    )


async def _notify_generation_progress(
    callback: ScenarioGenerationProgressCallback | None,
    progress: ScenarioGenerationProgress,
) -> None:
    if callback is None:
        return
    try:
        result = callback(progress)
        if result is not None:
            await result
    except Exception as exc:
        log_error_event(
            "scenario_generation.progress_callback_failed",
            scenario_type=progress.scenario_type.value,
            section_id=progress.section_id,
            status=progress.status,
            **exception_log_fields(exc),
        )


def _section_ids(scenario_type: ScenarioType) -> tuple[str, ...]:
    return _generated_section_ids(scenario_type)


def _generated_section_ids(
    scenario_type: ScenarioType,
    *,
    scenario_types: tuple[ScenarioType, ...] = (),
    action_choices_enabled: bool = False,
) -> tuple[str, ...]:
    normalized_genres = _validate_scenario_type_tuple(
        scenario_type,
        scenario_types or (scenario_type,),
    )
    if len(normalized_genres) > 1:
        return _with_choice_style(
            _merged_section_ids(
                _single_generated_section_ids(genre)
                for genre in normalized_genres
            ),
            action_choices_enabled=action_choices_enabled,
        )
    return _with_choice_style(
        _single_generated_section_ids(scenario_type),
        action_choices_enabled=action_choices_enabled,
    )


def _single_generated_section_ids(scenario_type: ScenarioType) -> tuple[str, ...]:
    if scenario_type is ScenarioType.FULL_ROLEPLAY:
        return FULL_ROLEPLAY_SECTIONS
    if scenario_type is ScenarioType.FANTASY_ROLEPLAY:
        return FANTASY_ROLEPLAY_SECTIONS
    if scenario_type is ScenarioType.SCIENCE_FICTION_ROLEPLAY:
        return SCIENCE_FICTION_ROLEPLAY_SECTIONS
    if scenario_type is ScenarioType.FIRST_CONTACT_EXPLORATION:
        return FIRST_CONTACT_EXPLORATION_SECTIONS
    if scenario_type is ScenarioType.SURVIVAL_EXPEDITION:
        return SURVIVAL_EXPEDITION_SECTIONS
    if scenario_type is ScenarioType.TIME_LOOP:
        return TIME_LOOP_SECTIONS
    if scenario_type is ScenarioType.INVESTIGATION_MYSTERY:
        return INVESTIGATION_MYSTERY_SECTIONS
    if scenario_type is ScenarioType.HEIST_INFILTRATION:
        return HEIST_INFILTRATION_SECTIONS
    if scenario_type is ScenarioType.POLITICAL_INTRIGUE:
        return POLITICAL_INTRIGUE_SECTIONS
    if scenario_type is ScenarioType.SETTLEMENT_BUILDER:
        return SETTLEMENT_BUILDER_SECTIONS
    if scenario_type is ScenarioType.MONSTER_HUNT_BOUNTY:
        return MONSTER_HUNT_BOUNTY_SECTIONS
    if scenario_type is ScenarioType.ROAD_TRIP_PILGRIMAGE:
        return ROAD_TRIP_PILGRIMAGE_SECTIONS
    if scenario_type is ScenarioType.MERCHANT_TRADE_ROUTE:
        return MERCHANT_TRADE_ROUTE_SECTIONS
    if scenario_type is ScenarioType.DATING_SIM:
        return DATING_SIM_SECTIONS
    return CHOOSE_YOUR_OWN_ADVENTURE_SECTIONS


def _with_choice_style(
    section_ids: tuple[str, ...],
    *,
    action_choices_enabled: bool,
) -> tuple[str, ...]:
    if not action_choices_enabled or "choice_style" in section_ids:
        return section_ids
    if "opening_message" not in section_ids:
        return (*section_ids, "choice_style")
    return tuple(
        item
        for section_id in section_ids
        for item in (
            ("choice_style", section_id)
            if section_id == "opening_message"
            else (section_id,)
        )
    )


def _merged_section_ids(section_groups: Iterable[tuple[str, ...]]) -> tuple[str, ...]:
    body: list[str] = []
    opening: list[str] = []
    seen: set[str] = set()
    for section_ids in section_groups:
        for section_id in section_ids:
            if section_id in seen:
                continue
            seen.add(section_id)
            if section_id in _OPENING_SECTION_IDS:
                opening.append(section_id)
            else:
                body.append(section_id)
    ordered_opening = [
        section_id for section_id in _OPENING_SECTION_IDS if section_id in opening
    ]
    return (*body, *ordered_opening)


def _allowed_section_ids(
    scenario_type: ScenarioType,
    *,
    scenario_types: tuple[ScenarioType, ...] = (),
) -> tuple[str, ...]:
    normalized_genres = _validate_scenario_type_tuple(
        scenario_type,
        scenario_types or (scenario_type,),
    )
    if len(normalized_genres) > 1:
        return _merged_section_ids(
            _single_allowed_section_ids(genre) for genre in normalized_genres
        )
    return _single_allowed_section_ids(scenario_type)


def _single_allowed_section_ids(scenario_type: ScenarioType) -> tuple[str, ...]:
    if scenario_type is ScenarioType.FULL_ROLEPLAY:
        return FULL_ROLEPLAY_ALLOWED_SECTIONS
    if scenario_type is ScenarioType.FANTASY_ROLEPLAY:
        return FANTASY_ROLEPLAY_ALLOWED_SECTIONS
    if scenario_type is ScenarioType.SCIENCE_FICTION_ROLEPLAY:
        return SCIENCE_FICTION_ROLEPLAY_ALLOWED_SECTIONS
    if scenario_type is ScenarioType.FIRST_CONTACT_EXPLORATION:
        return FIRST_CONTACT_EXPLORATION_ALLOWED_SECTIONS
    if scenario_type is ScenarioType.SURVIVAL_EXPEDITION:
        return SURVIVAL_EXPEDITION_ALLOWED_SECTIONS
    if scenario_type is ScenarioType.TIME_LOOP:
        return TIME_LOOP_ALLOWED_SECTIONS
    if scenario_type is ScenarioType.INVESTIGATION_MYSTERY:
        return INVESTIGATION_MYSTERY_ALLOWED_SECTIONS
    if scenario_type is ScenarioType.HEIST_INFILTRATION:
        return HEIST_INFILTRATION_ALLOWED_SECTIONS
    if scenario_type is ScenarioType.POLITICAL_INTRIGUE:
        return POLITICAL_INTRIGUE_ALLOWED_SECTIONS
    if scenario_type is ScenarioType.SETTLEMENT_BUILDER:
        return SETTLEMENT_BUILDER_ALLOWED_SECTIONS
    if scenario_type is ScenarioType.MONSTER_HUNT_BOUNTY:
        return MONSTER_HUNT_BOUNTY_ALLOWED_SECTIONS
    if scenario_type is ScenarioType.ROAD_TRIP_PILGRIMAGE:
        return ROAD_TRIP_PILGRIMAGE_ALLOWED_SECTIONS
    if scenario_type is ScenarioType.MERCHANT_TRADE_ROUTE:
        return MERCHANT_TRADE_ROUTE_ALLOWED_SECTIONS
    if scenario_type is ScenarioType.DATING_SIM:
        return DATING_SIM_ALLOWED_SECTIONS
    return CHOOSE_YOUR_OWN_ADVENTURE_ALLOWED_SECTIONS


def _validate_section_ids(
    scenario_type: ScenarioType,
    section_ids: tuple[str, ...],
    *,
    scenario_types: tuple[ScenarioType, ...] = (),
) -> None:
    allowed = set(_allowed_section_ids(scenario_type, scenario_types=scenario_types))
    unknown = [section_id for section_id in section_ids if section_id not in allowed]
    if unknown:
        raise ValueError(f"Unknown scenario section ids: {unknown}")


def _section_value_from_response(
    value: str,
    *,
    scenario_type: ScenarioType,
    scenario_types: tuple[ScenarioType, ...] = (),
    section_id: str,
) -> str:
    section_value = value.strip()
    if not section_value:
        if section_id in _optional_section_ids(
            scenario_type,
            scenario_types=scenario_types,
        ):
            return ""
        label = _section_label(section_id)
        raise ValueError(f"Scenario provider returned empty {label}")
    return section_value


def _name_dedup_retry_prompt(
    *,
    repeated_names: tuple[str, ...],
    previous_value: str,
) -> str:
    return (
        "The previous draft for this field repeated first names: "
        f"{', '.join(repeated_names)}.\n\n"
        "Previous draft:\n"
        f"{previous_value}\n\n"
        "Regenerate the complete value for this same requested field as natural "
        "prose. Keep the same roles, relationships, and scenario intent where "
        "possible, but use distinct first names unless the repetition is "
        "intentional and meaningful. Return only the field value."
    )


def _section_generation_prompt(
    *,
    scenario_type: ScenarioType,
    scenario_types: tuple[ScenarioType, ...] = (),
    action_choices_enabled: bool,
    section_id: str,
    seed: str,
    sections: Mapping[str, str],
) -> str:
    context = _generated_context(sections)
    guidance = _section_guidance(section_id)
    scenario_label = _scenario_type_label_for_genres(
        scenario_types or (scenario_type,),
    )
    name_context = ordinary_name_candidate_context(
        scenario_type=scenario_types or (scenario_type,),
        section_id=section_id,
        seed=seed,
        sections=sections,
    )
    parts = [
        f"User request:\n{seed.strip()}\n\n"
        f"Scenario type: {scenario_label}\n"
        f"Action choices: {'enabled' if action_choices_enabled else 'disabled'}\n"
        f"Requested field: {_section_label(section_id)}\n"
        f"Field guidance: {guidance}\n"
        f"{context}"
    ]
    if name_context:
        parts.append(name_context)
    parts.append("Return the complete value for this field only.")
    return "\n".join(parts)


def _generated_context(sections: Mapping[str, str]) -> str:
    if not sections:
        return "Generated context so far: none."
    lines = ["Generated context so far:"]
    lines.extend(
        f"- {_section_label(section_id)}: {value}"
        for section_id, value in sections.items()
    )
    return "\n".join(lines)


def _scenario_type_label(scenario_type: ScenarioType) -> str:
    if scenario_type is ScenarioType.FULL_ROLEPLAY:
        return "generic roleplay scenario"
    if scenario_type is ScenarioType.FANTASY_ROLEPLAY:
        return "fantasy roleplay scenario"
    if scenario_type is ScenarioType.SCIENCE_FICTION_ROLEPLAY:
        return "science fiction roleplay scenario"
    if scenario_type is ScenarioType.FIRST_CONTACT_EXPLORATION:
        return "first contact / exploration science fiction scenario"
    if scenario_type is ScenarioType.SURVIVAL_EXPEDITION:
        return "survival expedition scenario"
    if scenario_type is ScenarioType.TIME_LOOP:
        return "time loop scenario"
    if scenario_type is ScenarioType.INVESTIGATION_MYSTERY:
        return "investigation mystery scenario"
    if scenario_type is ScenarioType.HEIST_INFILTRATION:
        return "heist / infiltration scenario"
    if scenario_type is ScenarioType.POLITICAL_INTRIGUE:
        return "political intrigue scenario"
    if scenario_type is ScenarioType.SETTLEMENT_BUILDER:
        return "settlement builder scenario"
    if scenario_type is ScenarioType.MONSTER_HUNT_BOUNTY:
        return "monster hunt / bounty campaign scenario"
    if scenario_type is ScenarioType.ROAD_TRIP_PILGRIMAGE:
        return "road trip / pilgrimage scenario"
    if scenario_type is ScenarioType.MERCHANT_TRADE_ROUTE:
        return "merchant / trade route scenario"
    if scenario_type is ScenarioType.DATING_SIM:
        return "dating sim scenario"
    return "choose your own adventure scenario"


def _scenario_type_label_for_genres(
    scenario_types: tuple[ScenarioType, ...],
) -> str:
    if len(scenario_types) == 1:
        return _scenario_type_label(scenario_types[0])
    labels = [
        _scenario_type_label(genre)
        .removesuffix(" scenario")
        .removesuffix(" roleplay")
        for genre in scenario_types
    ]
    return f"{' / '.join(labels)} hybrid scenario"


def _section_label(section_id: str) -> str:
    return section_id.replace("_", " ")


def _section_guidance(section_id: str) -> str:
    return _SECTION_GUIDANCE.get(section_id, "Write concise, vivid scenario text.")


def _select_sections(
    scenario_type: ScenarioType,
    payload: dict[str, str],
    *,
    scenario_types: tuple[ScenarioType, ...] = (),
    action_choices_enabled: bool = False,
) -> dict[str, str]:
    payload = normalize_scenario_draft_sections(scenario_type, payload)
    normalized_genres = _validate_scenario_type_tuple(
        scenario_type,
        scenario_types or (scenario_type,),
    )
    generated_section_ids = _generated_section_ids(
        scenario_type,
        scenario_types=normalized_genres,
        action_choices_enabled=action_choices_enabled,
    )
    allowed_section_ids = _allowed_section_ids(
        scenario_type,
        scenario_types=normalized_genres,
    )
    missing = [
        section_id
        for section_id in generated_section_ids
        if section_id not in payload
        and section_id
        not in _optional_section_ids(scenario_type, scenario_types=normalized_genres)
    ]
    if missing:
        raise ValueError(f"Scenario draft missing sections: {missing}")
    sections = {
        section_id: payload.get(section_id, "")
        for section_id in generated_section_ids
    }
    for section_id in allowed_section_ids:
        if section_id not in sections and payload.get(section_id, "").strip():
            sections[section_id] = payload[section_id]
    return sections


def _scenario_content_with_metadata(
    sections: Mapping[str, object],
    metadata: Mapping[str, object] | None,
) -> dict[str, object]:
    content: dict[str, object] = dict(sections)
    if metadata:
        source_metadata = dict(metadata)
        source_metadata.pop("loss_conditions", None)
        if source_metadata:
            content["_source"] = source_metadata
    return content


def _draft_metadata_with_generation_prompt(
    metadata: Mapping[str, object] | None,
    seed: str,
) -> dict[str, object]:
    normalized = dict(metadata or {})
    if "generation_prompt" in normalized:
        return normalized
    if normalized.get("origin") == "save_continuation":
        return normalized
    prompt = seed.strip()
    if not prompt:
        return normalized
    normalized.setdefault("origin", "ai_draft")
    normalized["generation_prompt"] = prompt
    return normalized


def _content_with_scenario_genres(
    sections: Mapping[str, str],
    scenario_types: tuple[ScenarioType, ...],
) -> dict[str, str | list[str]]:
    content: dict[str, str | list[str]] = dict(sections)
    if len(scenario_types) > 1:
        content[SCENARIO_GENRES_CONTENT_KEY] = [
            scenario_type.value for scenario_type in scenario_types
        ]
    return content


def normalize_scenario_draft_sections(
    scenario_type: ScenarioType | str,
    sections: Mapping[str, str],
) -> dict[str, str]:
    normalized_type, legacy_action_choices_enabled = _normalized_scenario_type_and_flag(
        scenario_type,
        action_choices_enabled=False,
    )
    normalized = {key: value.strip() for key, value in sections.items()}
    normalized = strip_deprecated_scenario_character_sections_from_text(normalized)
    if legacy_action_choices_enabled:
        normalized["choice_style"] = normalized.get("choice_style", "")
    if _uses_opening_message_legacy_setup(normalized_type):
        starting_scene = normalized.pop("starting_scene", "").strip()
        if starting_scene:
            normalized["opening_message"] = _join_unique_paragraphs(
                starting_scene,
                normalized.get("opening_message", ""),
            )
    else:
        setup_line = normalized.pop("setup_line", "").strip()
        if setup_line:
            normalized["premise"] = _join_unique_paragraphs(
                normalized.get("premise", ""),
                setup_line,
            )
    return normalized


def normalize_scenario_definition(
    *,
    scenario_type: ScenarioType | str,
    premise: str,
    content: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    normalized_type, normalized, _legacy_action_choices_enabled = (
        _normalize_definition_type_and_content(
            scenario_type=scenario_type,
            content=content,
        )
    )
    normalized = strip_deprecated_scenario_character_sections(normalized)
    normalized_premise = premise.strip()
    if _uses_opening_message_legacy_setup(normalized_type):
        starting_scene = _object_text(normalized.pop("starting_scene", None))
        if starting_scene:
            normalized["opening_message"] = _join_unique_paragraphs(
                starting_scene,
                _object_text(normalized.get("opening_message")),
            )
    else:
        setup_line = _object_text(normalized.pop("setup_line", None))
        content_premise = _object_text(normalized.get("premise"))
        normalized_premise = _join_unique_paragraphs(
            normalized_premise,
            content_premise,
            setup_line,
        )
        if normalized_premise:
            normalized["premise"] = normalized_premise
        else:
            normalized.pop("premise", None)
    return normalized_premise, normalized


def strip_deprecated_scenario_character_sections(
    content: Mapping[str, object],
) -> dict[str, object]:
    normalized = dict(content)
    deprecated_values: dict[str, object] = {}
    faction_fragments = [_object_text(normalized.get("factions"))]
    for key in DEPRECATED_CHARACTER_LIST_SECTION_IDS:
        if key in normalized:
            deprecated_values[key] = normalized.pop(key)
    for key in _DEPRECATED_FACTION_APPEND_SECTION_IDS:
        text = _object_text(deprecated_values.get(key))
        if text:
            faction_fragments.append(text)
    nonblank_factions = [fragment for fragment in faction_fragments if fragment]
    if nonblank_factions:
        normalized["factions"] = "\n\n".join(nonblank_factions)
    return normalized


def strip_deprecated_scenario_character_sections_from_text(
    content: Mapping[str, str],
) -> dict[str, str]:
    stripped = strip_deprecated_scenario_character_sections(content)
    return {
        key: value
        for key, value in stripped.items()
        if isinstance(value, str)
    }


def _normalized_scenario_type_and_flag(
    scenario_type: ScenarioType | str,
    *,
    action_choices_enabled: bool,
) -> tuple[ScenarioType, bool]:
    normalized_type = ScenarioType(scenario_type)
    if normalized_type is ScenarioType.CHOOSE_YOUR_OWN_ADVENTURE:
        return ScenarioType.FULL_ROLEPLAY, True
    return normalized_type, action_choices_enabled


def normalized_scenario_types_and_flag(
    scenario_type: ScenarioType | str,
    *,
    scenario_types: Iterable[ScenarioType | str] | None = None,
    action_choices_enabled: bool,
) -> tuple[ScenarioType, tuple[ScenarioType, ...], bool]:
    raw_genres = tuple(scenario_types or (scenario_type,))
    if is_retired_scenario_type(scenario_type, raw_genres):
        raise ValueError(RETIRED_SCENARIO_REASON)
    normalized_type, action_choices_enabled = _normalized_scenario_type_and_flag(
        scenario_type,
        action_choices_enabled=action_choices_enabled,
    )
    normalized_genres: list[ScenarioType] = []
    for raw_genre in raw_genres:
        genre, genre_action_choices_enabled = _normalized_scenario_type_and_flag(
            raw_genre,
            action_choices_enabled=action_choices_enabled,
        )
        action_choices_enabled = genre_action_choices_enabled
        normalized_genres.append(genre)
    if not normalized_genres:
        normalized_genres = [normalized_type]
    return (
        normalized_type,
        _validate_scenario_type_tuple(normalized_type, normalized_genres),
        action_choices_enabled,
    )


def is_retired_scenario_type(
    scenario_type: object,
    scenario_types: Iterable[object] = (),
) -> bool:
    return scenario_type == RETIRED_SCENARIO_TYPE or any(
        item == RETIRED_SCENARIO_TYPE for item in scenario_types
    )


def scenario_record_is_retired(
    scenario_type: object,
    content: Mapping[str, object] | None = None,
) -> bool:
    raw_genres = content.get(SCENARIO_GENRES_CONTENT_KEY) if content else None
    return is_retired_scenario_type(
        scenario_type,
        raw_genres if isinstance(raw_genres, list) else (),
    )


def _validate_scenario_type_tuple(
    scenario_type: ScenarioType,
    scenario_types: Iterable[ScenarioType | str],
) -> tuple[ScenarioType, ...]:
    normalized = tuple(ScenarioType(item) for item in scenario_types)
    if not normalized:
        normalized = (scenario_type,)
    if len(normalized) > 2:
        raise ValueError("Hybrid scenarios support at most two scenario genres")
    seen: set[ScenarioType] = set()
    duplicates: list[str] = []
    for genre in normalized:
        if genre in seen:
            duplicates.append(genre.value)
        seen.add(genre)
    if duplicates:
        raise ValueError(f"Duplicate scenario genres: {duplicates}")
    if normalized[0] is not scenario_type:
        raise ValueError("Primary scenario type must be the first selected genre")
    return normalized


def _uses_opening_message_legacy_setup(scenario_type: ScenarioType) -> bool:
    return scenario_type in {
        ScenarioType.FULL_ROLEPLAY,
        ScenarioType.FANTASY_ROLEPLAY,
        ScenarioType.SCIENCE_FICTION_ROLEPLAY,
        ScenarioType.FIRST_CONTACT_EXPLORATION,
        ScenarioType.SURVIVAL_EXPEDITION,
        ScenarioType.TIME_LOOP,
        ScenarioType.INVESTIGATION_MYSTERY,
        ScenarioType.HEIST_INFILTRATION,
        ScenarioType.POLITICAL_INTRIGUE,
        ScenarioType.SETTLEMENT_BUILDER,
        ScenarioType.MONSTER_HUNT_BOUNTY,
        ScenarioType.ROAD_TRIP_PILGRIMAGE,
        ScenarioType.MERCHANT_TRADE_ROUTE,
        ScenarioType.CHOOSE_YOUR_OWN_ADVENTURE,
    }


def _normalize_definition_type_and_content(
    *,
    scenario_type: ScenarioType | str,
    content: Mapping[str, object],
) -> tuple[ScenarioType, dict[str, object], bool]:
    requested_type = ScenarioType(scenario_type)
    normalized_type, normalized_content, legacy_action_choices_enabled = (
        normalize_legacy_action_choice_scenario(
            scenario_type=requested_type.value,
            content=content,
        )
    )
    return (
        ScenarioType(normalized_type),
        normalized_content,
        legacy_action_choices_enabled,
    )


def _join_unique_paragraphs(*values: str) -> str:
    paragraphs: list[str] = []
    folded = ""
    for value in values:
        text = value.strip()
        if not text:
            continue
        folded_casefold = folded.casefold()
        text_casefold = text.casefold()
        if text_casefold in folded_casefold:
            continue
        if folded_casefold and folded_casefold in text_casefold:
            paragraphs = [text]
        else:
            paragraphs.append(text)
        folded = "\n\n".join(paragraphs)
    return folded


def _object_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _optional_section_ids(
    scenario_type: ScenarioType,
    *,
    scenario_types: tuple[ScenarioType, ...] = (),
) -> frozenset[str]:
    normalized_genres = _validate_scenario_type_tuple(
        scenario_type,
        scenario_types or (scenario_type,),
    )
    optional = {"relationship_seed"}
    if ScenarioType.DATING_SIM not in normalized_genres:
        optional.add("player_character_name")
    return frozenset(optional)


def _elapsed_ms(started_at: float) -> int:
    return int((perf_counter() - started_at) * 1000)
