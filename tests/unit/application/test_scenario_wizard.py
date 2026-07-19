from __future__ import annotations

import builtins
import importlib
import sys
from collections.abc import Iterable, Mapping
from typing import Any

from pytest import MonkeyPatch

_MISSING = object()


def test_wizard_model_exposes_roleplay_flows_without_gtk(
    monkeypatch: MonkeyPatch,
) -> None:
    scenario_wizard = _import_scenario_wizard_without_gtk(monkeypatch)

    model = scenario_wizard.build_scenario_wizard_model()
    flows = _flows_by_identifier(_value(model, "flows"))

    assert list(flows) == [
        "full_roleplay",
        "fantasy_roleplay",
        "science_fiction_roleplay",
        "first_contact_exploration",
        "survival_expedition",
        "time_loop",
        "investigation_mystery",
        "heist_infiltration",
        "political_intrigue",
        "settlement_builder",
        "monster_hunt_bounty",
        "road_trip_pilgrimage",
        "merchant_trade_route",
        "dating_sim",
    ]
    assert _value(flows["full_roleplay"], "label") == "Generic Roleplay"
    assert _value(flows["fantasy_roleplay"], "label") == "Fantasy"
    assert _value(flows["science_fiction_roleplay"], "label") == "Science Fiction"
    assert _value(flows["first_contact_exploration"], "label") == (
        "First Contact / Exploration"
    )
    assert _value(flows["survival_expedition"], "label") == "Survival Expedition"
    assert _value(flows["time_loop"], "label") == "Time Loop"
    assert _value(flows["investigation_mystery"], "label") == (
        "Investigation Mystery"
    )
    assert _value(flows["heist_infiltration"], "label") == "Heist / Infiltration"
    assert _value(flows["political_intrigue"], "label") == "Political Intrigue"
    assert _value(flows["settlement_builder"], "label") == "Settlement Builder"
    assert _value(flows["monster_hunt_bounty"], "label") == (
        "Monster Hunt / Bounty"
    )
    assert _value(flows["road_trip_pilgrimage"], "label") == (
        "Road Trip / Pilgrimage"
    )
    assert _value(flows["merchant_trade_route"], "label") == (
        "Merchant / Trade Route"
    )
    assert _value(flows["dating_sim"], "label") == "Dating Sim"
    assert "character_interaction" not in flows


def test_wizard_model_exposes_editable_sections_for_each_flow(
    monkeypatch: MonkeyPatch,
) -> None:
    scenario_wizard = _import_scenario_wizard_without_gtk(monkeypatch)

    model = scenario_wizard.build_scenario_wizard_model()
    flows = _flows_by_identifier(_value(model, "flows"))

    assert _section_ids(flows["full_roleplay"]) == [
        "title",
        "premise",
        "player_character_name",
        "player_role",
        "tone_genre",
        "opening_message",
    ]
    assert _section_ids(flows["fantasy_roleplay"]) == [
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
    ]
    assert _section_ids(flows["science_fiction_roleplay"]) == [
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
    ]
    assert _section_ids(flows["first_contact_exploration"]) == [
        "title",
        "premise",
        "player_character_name",
        "player_role",
        "mission_profile",
        "crew_and_command",
        "ship_or_base_status",
        "exploration_target",
        "unknown_intelligence",
        "knowledge_state",
        "translation_progress",
        "discoveries_and_samples",
        "hazards_and_escalation",
        "tone_genre",
        "opening_message",
    ]
    assert _section_ids(flows["survival_expedition"]) == [
        "title",
        "premise",
        "player_character_name",
        "player_role",
        "expedition_goal",
        "route_options",
        "party_roster",
        "resource_inventory",
        "environmental_conditions",
        "hazards_and_events",
        "camp_status",
        "travel_progress",
        "tone_genre",
        "opening_message",
    ]
    assert _section_ids(flows["time_loop"]) == [
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
    ]
    assert _section_ids(flows["investigation_mystery"]) == [
        "title",
        "premise",
        "player_character_name",
        "player_role",
        "case_facts",
        "suspects",
        "clues",
        "timeline",
        "red_herrings",
        "hidden_truth",
        "case_status",
        "tone_genre",
        "opening_message",
    ]
    assert _section_ids(flows["heist_infiltration"]) == [
        "title",
        "premise",
        "player_character_name",
        "player_role",
        "target_location",
        "objectives_and_stakes",
        "crew_and_contacts",
        "intel_and_access",
        "security_model",
        "alert_and_heat",
        "loadout_and_tools",
        "complications",
        "extraction_routes",
        "aftermath",
        "tone_genre",
        "opening_message",
    ]
    assert _section_ids(flows["political_intrigue"]) == [
        "title",
        "premise",
        "player_character_name",
        "player_role",
        "political_arena",
        "political_factions",
        "major_npcs",
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
    ]
    assert _section_ids(flows["settlement_builder"]) == [
        "title",
        "premise",
        "player_character_name",
        "player_role",
        "settlement_profile",
        "population_and_residents",
        "resources_and_indicators",
        "projects_and_facilities",
        "threats_and_opportunities",
        "calendar_and_deadlines",
        "tone_genre",
        "opening_message",
    ]
    assert _section_ids(flows["monster_hunt_bounty"]) == [
        "title",
        "premise",
        "player_character_name",
        "player_role",
        "hunt_profile",
        "target_profile",
        "leads_and_clues",
        "hunt_locations",
        "rivals_and_factions",
        "preparation_state",
        "hunt_status",
        "tone_genre",
        "opening_message",
    ]
    assert _section_ids(flows["road_trip_pilgrimage"]) == [
        "title",
        "premise",
        "player_character_name",
        "player_role",
        "journey_profile",
        "route_and_stops",
        "traveling_party",
        "transport_and_supplies",
        "recurring_pressures",
        "relationship_threads",
        "journey_progress",
        "tone_genre",
        "opening_message",
    ]
    assert _section_ids(flows["merchant_trade_route"]) == [
        "title",
        "premise",
        "player_character_name",
        "player_role",
        "trade_profile",
        "cargo_inventory",
        "markets_and_stops",
        "contracts_and_debts",
        "route_hazards",
        "reputation_and_contacts",
        "profit_and_loss",
        "tone_genre",
        "opening_message",
    ]
    assert _section_ids(flows["dating_sim"]) == [
        "title",
        "premise",
        "player_character_name",
        "player_character_profile",
        "player_role",
        "romance_options",
        "tone_genre",
        "opening_message",
    ]


def test_wizard_model_exposes_guided_seed_and_review_metadata_without_gtk(
    monkeypatch: MonkeyPatch,
) -> None:
    scenario_wizard = _import_scenario_wizard_without_gtk(monkeypatch)

    model = scenario_wizard.build_scenario_wizard_model()
    flows = _flows_by_identifier(_value(model, "flows"))

    full_roleplay = flows["full_roleplay"]
    full_seed = _seed_prompt(full_roleplay).casefold()
    assert "genre" in full_seed
    assert "premise" in full_seed
    assert "player role" in full_seed
    assert "opening narration" in full_seed
    assert _review_groups(full_roleplay) == {
        "Core": ["title", "premise", "player_character_name", "player_role"],
        "Opening": ["tone_genre", "opening_message"],
    }

    fantasy = flows["fantasy_roleplay"]
    fantasy_seed = _seed_prompt(fantasy).casefold()
    assert "fantasy" in fantasy_seed
    assert "magic" in fantasy_seed
    assert "quest" in fantasy_seed
    assert _review_groups(fantasy) == {
        "Core": ["title", "premise", "player_character_name", "player_role"],
        "Fantasy World": [
            "magic_system",
            "realms_and_places",
            "factions_and_orders",
            "myths_and_creatures",
            "quest_stakes",
        ],
        "Opening": ["tone_genre", "opening_message"],
    }

    science_fiction = flows["science_fiction_roleplay"]
    science_seed = _seed_prompt(science_fiction).casefold()
    assert "science fiction" in science_seed
    assert "technology" in science_seed
    assert "mission" in science_seed
    assert _review_groups(science_fiction) == {
        "Core": ["title", "premise", "player_character_name", "player_role"],
        "Science Fiction World": [
            "technology_level",
            "setting_scope",
            "species_and_intelligences",
            "factions_and_institutions",
            "mission_stakes",
        ],
        "Opening": ["tone_genre", "opening_message"],
    }

    first_contact = flows["first_contact_exploration"]
    first_contact_seed = _seed_prompt(first_contact).casefold()
    assert "first contact" in first_contact_seed
    assert "unknown world" in first_contact_seed
    assert "translation" in first_contact_seed
    assert "hazards" in first_contact_seed
    assert _review_groups(first_contact) == {
        "Core": ["title", "premise", "player_character_name", "player_role"],
        "Mission": [
            "mission_profile",
            "crew_and_command",
            "ship_or_base_status",
        ],
        "Discovery": [
            "exploration_target",
            "knowledge_state",
            "discoveries_and_samples",
            "hazards_and_escalation",
        ],
        "Contact": ["unknown_intelligence", "translation_progress"],
        "Opening": ["tone_genre", "opening_message"],
    }

    survival = flows["survival_expedition"]
    survival_seed = _seed_prompt(survival).casefold()
    assert "survival expedition" in survival_seed
    assert "route" in survival_seed
    assert "supplies" in survival_seed
    assert "hazards" in survival_seed
    assert _review_groups(survival) == {
        "Core": ["title", "premise", "player_character_name", "player_role"],
        "Expedition": ["expedition_goal", "route_options", "travel_progress"],
        "Party & Supplies": ["party_roster", "resource_inventory"],
        "Conditions": [
            "environmental_conditions",
            "hazards_and_events",
            "camp_status",
        ],
        "Opening": ["tone_genre", "opening_message"],
    }

    time_loop = flows["time_loop"]
    loop_seed = _seed_prompt(time_loop).casefold()
    assert "time loop" in loop_seed
    assert "reset trigger" in loop_seed
    assert "persistent knowledge" in loop_seed
    assert _review_groups(time_loop) == {
        "Core": ["title", "premise", "player_character_name", "player_role"],
        "Loop Rules": [
            "loop_premise",
            "reset_trigger",
            "loop_duration",
            "objective",
            "failure_conditions",
        ],
        "Reset State": ["starting_state", "baseline_world_state"],
        "Schedule": ["loop_schedule", "current_loop_state"],
        "Persistence": [
            "persistent_knowledge",
            "persistence_exceptions",
            "npc_memory_rules",
        ],
        "Opening": ["tone_genre", "opening_message"],
    }

    mystery = flows["investigation_mystery"]
    mystery_seed = _seed_prompt(mystery).casefold()
    assert "mystery" in mystery_seed
    assert "clues" in mystery_seed
    assert "suspects" in mystery_seed
    assert "hidden truth" in mystery_seed
    assert _review_groups(mystery) == {
        "Core": ["title", "premise", "player_character_name", "player_role"],
        "Case": ["case_facts", "suspects", "case_status"],
        "Evidence": ["clues", "timeline", "red_herrings", "hidden_truth"],
        "Opening": ["tone_genre", "opening_message"],
    }

    heist = flows["heist_infiltration"]
    heist_seed = _seed_prompt(heist).casefold()
    assert "heist" in heist_seed
    assert "security" in heist_seed
    assert "extraction" in heist_seed
    assert _review_groups(heist) == {
        "Core": ["title", "premise", "player_character_name", "player_role"],
        "Target & Objectives": [
            "target_location",
            "objectives_and_stakes",
        ],
        "Crew & Intel": ["crew_and_contacts", "intel_and_access"],
        "Security": ["security_model", "alert_and_heat"],
        "Tools & Complications": ["loadout_and_tools", "complications"],
        "Exit & Consequences": ["extraction_routes", "aftermath"],
        "Opening": ["tone_genre", "opening_message"],
    }

    intrigue = flows["political_intrigue"]
    intrigue_seed = _seed_prompt(intrigue).casefold()
    assert "political" in intrigue_seed
    assert "factions" in intrigue_seed
    assert "timed political pressure" in intrigue_seed
    assert _review_groups(intrigue) == {
        "Core": ["title", "premise", "player_character_name", "player_role"],
        "Arena": ["political_arena", "central_conflict"],
        "Factions & NPCs": [
            "political_factions",
            "major_npcs",
            "alliances_and_rivalries",
        ],
        "Leverage": [
            "secrets_and_leverage",
            "reputation_and_standing",
            "obligations_and_favors",
            "public_private_knowledge",
        ],
        "Pressure": [
            "event_calendar",
            "political_pressure",
        ],
        "Opening": ["tone_genre", "opening_message"],
    }

    settlement = flows["settlement_builder"]
    settlement_seed = _seed_prompt(settlement).casefold()
    assert "settlement" in settlement_seed
    assert "projects" in settlement_seed
    assert "resources" in settlement_seed
    assert _review_groups(settlement) == {
        "Core": ["title", "premise", "player_character_name", "player_role"],
        "Community": ["settlement_profile", "population_and_residents"],
        "Operations": ["resources_and_indicators", "projects_and_facilities"],
        "Pressure": ["threats_and_opportunities", "calendar_and_deadlines"],
        "Opening": ["tone_genre", "opening_message"],
    }

    hunt = flows["monster_hunt_bounty"]
    hunt_seed = _seed_prompt(hunt).casefold()
    assert "hunt" in hunt_seed
    assert "target" in hunt_seed
    assert "clues" in hunt_seed
    assert _review_groups(hunt) == {
        "Core": ["title", "premise", "player_character_name", "player_role"],
        "Hunt": ["hunt_profile", "target_profile", "hunt_status"],
        "Investigation": ["leads_and_clues", "hunt_locations"],
        "Pressure": ["rivals_and_factions", "preparation_state"],
        "Opening": ["tone_genre", "opening_message"],
    }

    journey = flows["road_trip_pilgrimage"]
    journey_seed = _seed_prompt(journey).casefold()
    assert "journey" in journey_seed
    assert "route" in journey_seed
    assert "relationships" in journey_seed
    assert _review_groups(journey) == {
        "Core": ["title", "premise", "player_character_name", "player_role"],
        "Journey": ["journey_profile", "route_and_stops", "journey_progress"],
        "Party": ["traveling_party", "relationship_threads"],
        "Road Pressure": ["transport_and_supplies", "recurring_pressures"],
        "Opening": ["tone_genre", "opening_message"],
    }

    trade = flows["merchant_trade_route"]
    trade_seed = _seed_prompt(trade).casefold()
    assert "trade" in trade_seed
    assert "cargo" in trade_seed
    assert "contracts" in trade_seed
    assert _review_groups(trade) == {
        "Core": ["title", "premise", "player_character_name", "player_role"],
        "Trade Route": ["trade_profile", "markets_and_stops"],
        "Cargo & Contracts": ["cargo_inventory", "contracts_and_debts"],
        "Risk & Standing": [
            "route_hazards",
            "reputation_and_contacts",
            "profit_and_loss",
        ],
        "Opening": ["tone_genre", "opening_message"],
    }

    dating_sim = flows["dating_sim"]
    dating_seed = _seed_prompt(dating_sim).casefold()
    assert "player character" in dating_seed
    assert "romance option" in dating_seed
    assert "four" in dating_seed
    assert _review_groups(dating_sim) == {
        "Core": [
            "title",
            "premise",
            "player_character_name",
            "player_character_profile",
            "player_role",
        ],
        "Romance Options": ["romance_options"],
        "Opening": ["tone_genre", "opening_message"],
    }

    for flow in flows.values():
        grouped_sections = {
            section_id
            for section_ids in _review_groups(flow).values()
            for section_id in section_ids
        }
        assert grouped_sections == set(_section_ids(flow))


def _import_scenario_wizard_without_gtk(monkeypatch: MonkeyPatch) -> Any:
    original_import = builtins.__import__

    def import_without_gtk(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "gi" or name.startswith("gi."):
            raise AssertionError(
                "bragi.application.scenario_wizard model must not import GTK/PyGObject"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_gtk)
    sys.modules.pop("bragi.application.scenario_wizard", None)
    return importlib.import_module("bragi.application.scenario_wizard")


def _flows_by_identifier(flows: Iterable[object]) -> dict[str, object]:
    return {_value(flow, "flow_id", "identifier", "id"): flow for flow in flows}


def _section_ids(flow: object) -> list[str]:
    sections = _value(
        flow,
        "editable_section_ids",
        "editable_sections",
        "sections",
    )
    return [
        section
        if isinstance(section, str)
        else _value(section, "section_id", "identifier", "id")
        for section in sections
    ]


def _seed_prompt(flow: object) -> str:
    prompt = _value(flow, "seed_prompt", "seed_description", "description")
    assert isinstance(prompt, str)
    assert prompt.strip()
    return prompt


def _review_groups(flow: object) -> dict[str, list[str]]:
    groups = _value(flow, "review_groups", "section_groups")
    assert isinstance(groups, Iterable)
    grouped_sections: dict[str, list[str]] = {}
    for group in groups:
        label = _value(group, "label", "title", "name")
        assert isinstance(label, str)
        section_ids = _value(group, "section_ids", "sections")
        assert isinstance(section_ids, Iterable)
        grouped_sections[label] = [
            section
            if isinstance(section, str)
            else _value(section, "section_id", "identifier", "id")
            for section in section_ids
        ]
    return grouped_sections


def _value(
    item: object,
    *names: str,
    default: object = _MISSING,
) -> Any:
    for name in names:
        if isinstance(item, Mapping):
            if name in item:
                return item[name]
        elif hasattr(item, name):
            return getattr(item, name)

    if default is not _MISSING:
        return default

    raise AssertionError(f"{item!r} does not expose any of {names!r}")
