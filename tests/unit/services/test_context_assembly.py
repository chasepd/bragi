from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.models import LocationRecord, SaveRecord, ScenarioRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.context_assembly import (
    CONTEXT_BUDGET_MODE_ADAPTIVE_TIERS,
    CONTEXT_BUDGET_MODE_DIAGNOSTICS_ONLY,
    CONTEXT_BUDGET_MODE_FIXED_CHARS,
    ContextAssemblyService,
    ContextBudgetSettings,
    ContextSource,
    apply_context_budget,
    compact_scenario_instructions,
    deterministic_context_sources,
    pending_context_suggestion_sources,
    scenario_section_candidates,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_compact_scenario_instructions_keeps_setup_compact() -> None:
    scenario = _scenario(
        content={
            "tone_genre": "Tense frontier mystery.",
            "starting_scene": "The beacon gutters as the storm wall closes in.",
            "current_scene": "The warden stands at the lower gate.",
            "lore": "Ancient beacon lore should be retrieved only when selected.",
        }
    )

    instructions = compact_scenario_instructions(scenario)

    assert "Title: Ashfall Keep" in instructions
    assert "Premise/setup: A border keep is cut off by ash storms." in instructions
    assert "Player role: Signal warden" in instructions
    assert "Narrator control rule" in instructions
    assert (
        "Do not advance time in ways that make the player character act"
        in instructions
    )
    assert "texting, sleeping, traveling, arriving" in instructions
    assert (
        "Treat stated intent, future-tense plans, NPC-provided directions, and "
        "in-progress movement as not enough to complete the player's travel"
    ) in instructions
    assert "arrival, entry, knock, touch, or other next action" in instructions
    assert "Tone/style: Tense frontier mystery." in instructions
    assert "Starting scene:" not in instructions
    assert "Current scene: The warden stands at the lower gate." in instructions
    assert "Ancient beacon lore" not in instructions


def test_retired_scenario_type_has_no_compact_instruction_specialization() -> None:
    scenario = _scenario(
        scenario_type="character_interaction",
        content={
            "character_name": "Oracle of Glass",
            "character_description": "A seer in the mirrored arcade.",
            "character_physical_description": (
                "Silver eyes, glass-dusted robes, and still hands."
            ),
            "relationship_seed": "The oracle is wary of the petitioner.",
        },
    )

    instructions = compact_scenario_instructions(scenario)

    assert "Primary character:" not in instructions
    assert "Primary character context:" not in instructions
    assert "Primary character physical description:" not in instructions
    assert "Relationship baseline:" not in instructions


def test_compact_scenario_instructions_keeps_mystery_truth_retrieval_only() -> None:
    scenario = _scenario(
        scenario_type="investigation_mystery",
        content={
            "player_character_name": "Inspector Mara Voss",
            "case_facts": "Curator Elian Vale vanished from a sealed gallery.",
            "case_status": "Unresolved; only public facts are known.",
            "clues": "Watch log gap from 9:10 to 9:18 remains undiscovered.",
            "hidden_truth": "Sera hid the smuggling ledger in the restoration lift.",
            "current_scene": "Mara stands outside the east gallery.",
        },
    )

    instructions = compact_scenario_instructions(scenario)

    assert "Player character name: Inspector Mara Voss" in instructions
    assert "Case facts: Curator Elian Vale vanished from a sealed gallery." in (
        instructions
    )
    assert "Case status: Unresolved; only public facts are known." in instructions
    assert "Current scene: Mara stands outside the east gallery." in instructions
    assert "Watch log gap" not in instructions
    assert "smuggling ledger" not in instructions


def test_compact_scenario_instructions_guides_cyoa_changed_situations() -> None:
    scenario = _scenario(
        scenario_type="full_roleplay",
        content={
            "action_choices_enabled": True,
            "choice_style": "Four terse choices: cautious, bold, social, strange.",
            "tone_genre": "Book-like fantasy suspense.",
        },
    )

    instructions = compact_scenario_instructions(scenario)

    assert "Choose-your-own-adventure control rule" in instructions
    assert "concrete changed situation" in instructions
    assert "suitable for player action choices" in instructions
    assert "ends at a decision point" not in instructions
    assert "Do not include numbered options" in instructions
    assert "Bragi generates those separately" in instructions
    assert "Choice style: Four terse choices" in instructions


def test_compact_scenario_instructions_includes_survival_expedition_setup() -> None:
    scenario = _scenario(
        scenario_type="survival_expedition",
        content={
            "expedition_goal": "Reach Northwatch before the medicine spoils.",
            "route_options": "Cliff road, glacier basin, or old mine tunnel.",
            "resource_inventory": "Food: 9 days. Water: 6 skins.",
            "environmental_conditions": "Late winter whiteouts and brittle ice.",
            "hazards_and_events": "Avalanches, frostbite, and lost trail markers.",
            "camp_status": "Two canvas tents; one stove sputters.",
            "travel_progress": "0 of 80 miles traveled; retreat remains possible.",
        },
    )

    instructions = compact_scenario_instructions(scenario)

    assert "Expedition goal: Reach Northwatch" in instructions
    assert "Route options: Cliff road" in instructions
    assert "Resource inventory: Food: 9 days" in instructions
    assert "Environmental conditions: Late winter" in instructions
    assert "Hazards/events: Avalanches" in instructions
    assert "Camp status: Two canvas tents" in instructions
    assert "Travel progress: 0 of 80 miles" in instructions


def test_compact_scenario_instructions_includes_time_loop_boundaries() -> None:
    scenario = _scenario(
        scenario_type="time_loop",
        content={
            "loop_premise": "The festival day repeats until the bell is saved.",
            "reset_trigger": "The drowned bell tolls at midnight.",
            "loop_duration": "Twenty-four hours.",
            "starting_state": "Mara wakes in the archive loft.",
            "objective": "Prevent the bell from sinking.",
            "failure_conditions": "The bell sinks or midnight arrives.",
            "baseline_world_state": "The harbor resets to dawn.",
            "loop_schedule": "09:00 parade; 23:45 sabotage.",
            "persistent_knowledge": "Tower code persists for the player.",
            "persistence_exceptions": "A salt mark persists.",
            "npc_memory_rules": "NPCs reset unless excepted.",
            "current_loop_state": "Loop 1, dawn phase.",
        },
    )

    instructions = compact_scenario_instructions(scenario)

    assert "Loop premise: The festival day repeats" in instructions
    assert "Reset trigger: The drowned bell" in instructions
    assert "Loop duration: Twenty-four hours" in instructions
    assert "Reset baseline: The harbor resets" in instructions
    assert "Persistent player/meta knowledge: Tower code" in instructions
    assert "NPC memory rules: NPCs reset" in instructions
    assert "Current loop state: Loop 1" not in instructions


def test_compact_scenario_instructions_includes_political_intrigue_setup() -> None:
    scenario = _scenario(
        scenario_type="political_intrigue",
        content={
            "political_arena": "The harbor council and its public galleries.",
            "political_factions": "Guilds, Old Families, and reform pamphleteers.",
            "central_conflict": "A midnight no-confidence vote can replace the regent.",
            "secrets_and_leverage": "Only Mara knows Orro moved missing silver.",
            "reputation_and_standing": "Mara is trusted by reformers.",
            "obligations_and_favors": "Orro owes Mara one public endorsement.",
            "alliances_and_rivalries": "Reformers court Mara; old houses resist.",
            "event_calendar": "Dawn hearing; noon procession; midnight vote.",
            "political_pressure": "The midnight vote proceeds unless delayed.",
            "public_private_knowledge": (
                "The public knows the vote is close; only Mara knows the favor."
            ),
        },
    )

    instructions = compact_scenario_instructions(scenario)

    assert "Political arena: The harbor council" in instructions
    assert "Political factions: Guilds" in instructions
    assert "Central conflict: A midnight no-confidence vote" in instructions
    assert "Secrets/leverage: Only Mara knows Orro moved" in instructions
    assert "Reputation/standing: Mara is trusted" in instructions
    assert "Obligations/favors: Orro owes Mara" in instructions
    assert "Alliances/rivalries: Reformers court Mara" in instructions
    assert "Event calendar: Dawn hearing" in instructions
    assert "Political pressure: The midnight vote proceeds" in instructions
    assert "Public/private knowledge: The public knows the vote is close" in (
        instructions
    )


def test_compact_scenario_instructions_can_omit_aged_setup_fields() -> None:
    scenario = ScenarioRecord(
        id="scenario-aged",
        type="full_roleplay",
        title="Lantern Archive Arrival",
        premise=(
            "A sprawling initial setup about lantern ferries, archive districts, "
            "and a disputed star map."
        ),
        player_role=(
            "The player is Avery Quill, a fictional archive courier whose initial "
            "biography should not ride along after setup is no longer recent."
        ),
        content_json=json.dumps(
            {
                "player_character_name": "Avery Quill",
                "tone_genre": (
                    "Warm archival mystery with a long initial style brief "
                    "that belongs near setup, not every late narrator prompt."
                ),
                "current_scene": (
                    "Avery is reviewing a map with Nira in the archive atrium."
                ),
            }
        ),
    )

    instructions = compact_scenario_instructions(scenario, include_setup=False)

    assert "Title: Lantern Archive Arrival" in instructions
    assert "Player character name: Avery Quill" in instructions
    assert (
        "Current scene: Avery is reviewing a map with Nira in the archive atrium."
        in instructions
    )
    assert "Narrator control rule" in instructions
    assert "Premise/setup:" not in instructions
    assert "Tone/style:" not in instructions
    assert "Player role:" not in instructions
    assert "disputed star map" not in instructions
    assert "long initial style brief" not in instructions
    assert "long initial biography" not in instructions


def test_scenario_section_candidates_exclude_core_content_fields() -> None:
    scenario = _scenario(
        content={
            "title": "Duplicate title",
            "premise": "Duplicate premise",
            "setup_line": "Duplicate setup",
            "player_role": "Duplicate role",
            "tone_genre": "Duplicate tone",
            "starting_scene": "Duplicate opening scene",
            "current_scene": "Duplicate current scene",
            "relationship_seed": "Duplicate relationship",
            "locations": "The beacon tower leans over the ash gate.",
            "cast": {"Captain Ilyra": "watch captain"},
        }
    )

    candidates = scenario_section_candidates(scenario)

    assert candidates == (
        (
            "scenario:scenario-1:section:locations",
            "locations",
            "The beacon tower leans over the ash gate.",
        ),
        (
            "scenario:scenario-1:section:cast",
            "cast",
            '{"Captain Ilyra": "watch captain"}',
        ),
    )


def test_scenario_section_candidates_include_mystery_hidden_truth() -> None:
    scenario = _scenario(
        scenario_type="investigation_mystery",
        content={
            "case_facts": "Curator Elian Vale vanished from a sealed gallery.",
            "case_status": "Unresolved.",
            "hidden_truth": "Sera hid the ledger in the restoration lift.",
            "current_scene": "Mara stands outside the east gallery.",
        },
    )

    candidates = scenario_section_candidates(scenario)

    assert candidates == (
        (
            "scenario:scenario-1:section:hidden_truth",
            "hidden_truth",
            "Sera hid the ledger in the restoration lift.",
        ),
    )


def test_apply_context_budget_reports_metadata_and_skips_over_budget_sources() -> None:
    sources = (
        ContextSource(
            tier="rules",
            source_type="scenario",
            source_id="scenario-1",
            text="abc",
        ),
        ContextSource(
            tier="current_scene",
            source_type="scene_snapshot",
            source_id="snapshot-1",
            text="defgh",
            reason="current scene snapshot",
        ),
        ContextSource(
            tier="active_threads",
            source_type="active_thread",
            source_id="thread-1",
            text="ij",
        ),
    )

    diagnostics_sources, diagnostics = apply_context_budget(
        sources,
        settings=ContextBudgetSettings(mode=CONTEXT_BUDGET_MODE_DIAGNOSTICS_ONLY),
    )
    fixed_sources, fixed = apply_context_budget(
        sources,
        settings=ContextBudgetSettings(
            mode=CONTEXT_BUDGET_MODE_FIXED_CHARS,
            fixed_total_chars=7,
        ),
    )
    adaptive_sources, adaptive = apply_context_budget(
        sources,
        settings=ContextBudgetSettings(
            mode=CONTEXT_BUDGET_MODE_ADAPTIVE_TIERS,
            fixed_total_chars=10,
            adaptive_fraction=0.5,
        ),
    )

    assert diagnostics_sources == sources
    assert diagnostics.budget_limit_chars is None
    assert diagnostics.total_chars == 10
    assert diagnostics.included_chars == 10
    assert all(source.included for source in diagnostics.sources)

    assert [source.source_id for source in fixed_sources] == [
        "scenario-1",
        "thread-1",
    ]
    assert fixed.budget_limit_chars == 7
    assert fixed.included_chars == 5
    assert [(source.source_id, source.included) for source in fixed.sources] == [
        ("scenario-1", True),
        ("snapshot-1", False),
        ("thread-1", True),
    ]
    assert fixed.sources[1].reason == "budget_skipped"

    assert [source.source_id for source in adaptive_sources] == [
        "scenario-1",
        "thread-1",
    ]
    assert adaptive.budget_limit_chars == 5


def test_apply_context_budget_relevance_trims_selected_canonical_source() -> None:
    source = ContextSource(
        tier="retrieved_memories",
        source_type="memory",
        source_id="memory-1",
        text=(
            "[memory:memory-1] Opening context. "
            + ("unrelated filler " * 30)
            + "The cracked bell hides the evacuation key. "
            + ("closing filler " * 20)
        ),
        reason="selected by context search",
        relevance_query="cracked bell evacuation key",
        trimmable=True,
    )

    selected, breakdown = apply_context_budget(
        (source,),
        settings=ContextBudgetSettings(
            mode=CONTEXT_BUDGET_MODE_FIXED_CHARS,
            fixed_total_chars=140,
        ),
    )

    assert len(selected) == 1
    assert len(selected[0].text) <= 140
    assert selected[0].text.startswith("[memory:memory-1]")
    assert "cracked bell hides the evacuation key" in selected[0].text
    assert breakdown.included_chars == len(selected[0].text)
    assert breakdown.sources[0].included is True
    assert breakdown.sources[0].reason == "budget_trimmed"


def test_narrator_context_always_includes_current_in_world_time_under_tight_budget(
    repositories: PersistenceRepositories,
) -> None:
    _scenario, save, current_location = _create_context_save(
        repositories,
        scenario_id="scenario-current-time-anchor",
        save_id="save-current-time-anchor",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=current_location.id,
        situation="This verbose situation should be skipped under the tiny budget.",
        in_world_time="late morning",
        snapshot_id="snapshot-current-time-anchor",
    )
    repositories.set_app_setting("context_budget_mode", CONTEXT_BUDGET_MODE_FIXED_CHARS)
    repositories.set_app_setting("context_budget_fixed_total_chars", 1)

    assembled = ContextAssemblyService(repositories).assemble_narrator_context(save.id)

    assert (
        "Current world time: late morning. Keep the response consistent with this "
        "unless the player explicitly advances time."
    ) in assembled.current_scene_context
    assert any(
        "Scene snapshot: situation: This verbose situation" in item
        for item in assembled.current_scene_context
    )
    assert any("Current location:" in item for item in assembled.current_scene_context)
    included = [
        (source.source_type, source.source_id, source.included)
        for source in assembled.breakdown.sources
    ]
    assert (
        "scene_snapshot",
        "snapshot-current-time-anchor:world_time",
        True,
    ) in included
    assert ("scene_snapshot", "snapshot-current-time-anchor", True) in included
    assert ("location", current_location.id, True) in included


def test_narrator_context_always_includes_structured_world_time_under_tight_budget(
    repositories: PersistenceRepositories,
) -> None:
    _scenario, save, current_location = _create_context_save(
        repositories,
        scenario_id="scenario-structured-time-anchor",
        save_id="save-structured-time-anchor",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=current_location.id,
        situation="This verbose situation should be skipped under the tiny budget.",
        in_world_time="Monday morning",
        time_of_day="evening",
        day_of_week="tuesday",
        world_day_index=2,
        snapshot_id="snapshot-structured-time-anchor",
    )
    repositories.set_app_setting("context_budget_mode", CONTEXT_BUDGET_MODE_FIXED_CHARS)
    repositories.set_app_setting("context_budget_fixed_total_chars", 1)

    assembled = ContextAssemblyService(repositories).assemble_narrator_context(save.id)

    assert (
        "Current world time: Tuesday evening; world day index 2. Keep the "
        "response consistent with this unless the player explicitly advances time."
    ) in assembled.current_scene_context
    assert not any("Monday morning" in item for item in assembled.current_scene_context)
    included = [
        (source.source_type, source.source_id, source.included)
        for source in assembled.breakdown.sources
    ]
    assert (
        "scene_snapshot",
        "snapshot-structured-time-anchor:world_time",
        True,
    ) in included


def test_narrator_context_preserves_matching_legacy_world_time_detail(
    repositories: PersistenceRepositories,
) -> None:
    _scenario, save, current_location = _create_context_save(
        repositories,
        scenario_id="scenario-rich-time-anchor",
        save_id="save-rich-time-anchor",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=current_location.id,
        situation="This verbose situation should be skipped under the tiny budget.",
        in_world_time="Tuesday evening",
        time_of_day="evening",
        day_of_week="tuesday",
        world_day_index=2,
        snapshot_id="snapshot-rich-time-anchor",
    )
    repositories.connection.execute(
        """
        UPDATE scene_snapshots
        SET in_world_time = 'Tuesday evening after class'
        WHERE id = 'snapshot-rich-time-anchor'
        """
    )
    repositories.set_app_setting("context_budget_mode", CONTEXT_BUDGET_MODE_FIXED_CHARS)
    repositories.set_app_setting("context_budget_fixed_total_chars", 1)

    assembled = ContextAssemblyService(repositories).assemble_narrator_context(save.id)

    assert (
        "Current world time: Tuesday evening after class; world day index 2. Keep "
        "the response consistent with this unless the player explicitly advances "
        "time."
    ) in assembled.current_scene_context


def test_narrator_context_always_includes_present_dating_route_anchor(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Last Summer",
        premise="A summer of route choices.",
        player_role="Transfer student",
        content={},
        scenario_id="scenario-dating-route-anchor",
    )
    save = repositories.create_save(
        scenario_id=scenario.id,
        title="Summer Save",
        save_id="save-dating-route-anchor",
    )
    player = repositories.add_character(
        save_id=save.id,
        name="Lio Takahashi",
        is_player_character=True,
        met=True,
    )
    npc = repositories.add_character(
        save_id=save.id,
        name="Mika Arai",
        relationships={player.name: "romance option for Lio Takahashi"},
        met=True,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        present_character_ids=[npc.id],
        world_day_index=2,
        snapshot_id="snapshot-dating-route-anchor",
    )
    repositories.upsert_dating_route_state(
        save_id=save.id,
        player_character_id=player.id,
        npc_character_id=npc.id,
        stage="contact_exchanged",
        first_met_world_day_index=0,
        last_interaction_world_day_index=2,
        completed_interactions=1,
        dates_completed=0,
        interest_level="curious",
        trust_level="guarded",
        comfort_with_intimacy="none yet",
        pacing_preference="slow burn",
        known_boundaries=["no instant commitment"],
        next_reasonable_step="schedule a first date",
        route_id="route-mika",
    )
    repositories.set_app_setting("context_budget_mode", CONTEXT_BUDGET_MODE_FIXED_CHARS)
    repositories.set_app_setting("context_budget_fixed_total_chars", 1)

    assembled = ContextAssemblyService(repositories).assemble_narrator_context(save.id)

    route_context = "\n".join(assembled.current_scene_context)
    assert "Dating route pacing for Mika Arai" in route_context
    assert "stage: contact exchanged" in route_context
    assert "known for 2 in-world days" in route_context
    assert "completed interactions: 1" in route_context
    assert "interest: curious" in route_context
    assert "trust: guarded" in route_context
    assert "comfort with intimacy: none yet" in route_context
    assert "next plausible step: schedule a first date" in route_context
    assert "max plausible escalation: follow-up interaction" in route_context
    assert "needs explicit support: guarded vulnerability" in route_context
    assert "premature now: exclusivity or commitment language" in route_context
    included = [
        (source.source_type, source.source_id, source.included)
        for source in assembled.breakdown.sources
    ]
    assert ("dating_route_state", "route-mika", True) in included


def test_narrator_context_includes_pending_review_suggestions_but_image_does_not(
    repositories: PersistenceRepositories,
) -> None:
    _scenario, save, current_location = _create_context_save(
        repositories,
        scenario_id="scenario-pending-review-context",
        save_id="save-pending-review-context",
    )
    source_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The storm around the beacon seems wary now.",
        provider="fake",
        model="fake-chat",
    )
    suggestion = repositories.add_context_update_suggestion(
        save_id=save.id,
        update_type="update",
        entity_type="world_state",
        entity_id="state-storm-mood",
        field_path="storm.mood",
        proposed_value={"mood": "wary"},
        reason="The narrator described the storm as wary.",
        confidence=0.91,
        source_message_ids=[source_message.id],
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=current_location.id,
        situation="The red lens ticks under stress.",
    )

    assembled = ContextAssemblyService(repositories).assemble_narrator_context(save.id)
    narrator_context = "\n".join(assembled.current_scene_context)
    image_context, image_breakdown = ContextAssemblyService(
        repositories
    ).build_image_scene_context(save_id=save.id)

    assert "Pending review (not canon yet)" in narrator_context
    assert "update world_state/state-storm-mood storm.mood" in narrator_context
    assert '"mood": "wary"' in narrator_context
    assert "confidence=91%" in narrator_context
    assert source_message.id not in narrator_context
    assert "The narrator described the storm as wary" not in narrator_context
    assert any(
        source.tier == "pending_context_suggestions"
        and source.source_type == "context_update_suggestion"
        and source.source_id == suggestion.id
        and source.included
        for source in assembled.breakdown.sources
    )
    assert "Pending review" not in image_context
    assert all(
        source.tier != "pending_context_suggestions"
        for source in image_breakdown.sources
    )


def test_pending_context_suggestion_sources_group_and_cap_rows(
    repositories: PersistenceRepositories,
) -> None:
    _scenario, save, _current_location = _create_context_save(
        repositories,
        scenario_id="scenario-pending-review-groups",
        save_id="save-pending-review-groups",
    )
    top = repositories.add_context_update_suggestion(
        save_id=save.id,
        update_type="update",
        entity_type="world_state",
        entity_id="state-top",
        field_path="storm.top",
        proposed_value={"mood": "urgent"},
        reason="Highest confidence should appear first.",
        confidence=0.99,
        suggestion_id="suggestion-top",
    )
    first_duplicate = repositories.add_context_update_suggestion(
        save_id=save.id,
        update_type="update",
        entity_type="world_state",
        entity_id="state-duplicate",
        field_path="storm.duplicate",
        proposed_value={"mood": "wary"},
        reason="First duplicate reason.",
        confidence=0.95,
        source_message_ids=["message-1"],
        suggestion_id="suggestion-duplicate-1",
    )
    repositories.add_context_update_suggestion(
        save_id=save.id,
        update_type="update",
        entity_type="world_state",
        entity_id="state-duplicate",
        field_path="storm.duplicate",
        proposed_value={"mood": "wary"},
        reason="Second duplicate reason.",
        confidence=0.81,
        source_message_ids=["message-1", "message-2"],
        suggestion_id="suggestion-duplicate-2",
    )
    stale = repositories.add_context_update_suggestion(
        save_id=save.id,
        update_type="update",
        entity_type="world_state",
        entity_id="state-stale",
        field_path="storm.stale",
        proposed_value={"mood": "obsolete"},
        reason="This high confidence suggestion is too old for prompts.",
        confidence=1.0,
        suggestion_id="suggestion-stale",
    )
    for index in range(4):
        repositories.add_context_update_suggestion(
            save_id=save.id,
            update_type="update",
            entity_type="world_state",
            entity_id=f"state-mid-{index}",
            field_path=f"storm.mid.{index}",
            proposed_value={"index": index},
            confidence=0.65,
            suggestion_id=f"suggestion-mid-{index}",
        )
    for index in range(5):
        repositories.add_context_update_suggestion(
            save_id=save.id,
            update_type="update",
            entity_type="world_state",
            entity_id=f"state-low-{index}",
            field_path=f"storm.low.{index}",
            proposed_value={"index": index},
            confidence=0.2,
            suggestion_id=f"suggestion-low-{index}",
        )
    repositories.connection.execute(
        """
        UPDATE context_update_suggestions
        SET created_at = datetime('now', '-13 hours')
        WHERE id = ?
        """,
        (stale.id,),
    )
    repositories.connection.commit()

    sources = pending_context_suggestion_sources(
        repositories=repositories,
        save_id=save.id,
    )

    assert len(sources) == 6
    assert sources[0].source_id == top.id
    assert sources[1].source_id == (
        f"{first_duplicate.id},suggestion-duplicate-2"
    )
    assert "suggestion-stale" not in {source.source_id for source in sources}
    assert all("suggestion-low" not in source.source_id for source in sources)
    assert "grouped=2" in sources[1].text
    assert "message-1" not in sources[1].text
    assert "message-2" not in sources[1].text
    assert "First duplicate reason" not in sources[1].text
    assert "Second duplicate reason" not in sources[1].text


def test_pending_context_suggestion_sources_exclude_non_pending_rows(
    repositories: PersistenceRepositories,
) -> None:
    _scenario, save, _current_location = _create_context_save(
        repositories,
        scenario_id="scenario-pending-review-status",
        save_id="save-pending-review-status",
    )
    repositories.add_context_update_suggestion(
        save_id=save.id,
        update_type="update",
        entity_type="world_state",
        entity_id="state-pending",
        field_path="storm.pending",
        proposed_value={"mood": "pending"},
        confidence=0.9,
        status="pending",
        suggestion_id="suggestion-pending",
    )
    for status in ("applied", "rejected", "dismissed"):
        repositories.add_context_update_suggestion(
            save_id=save.id,
            update_type="update",
            entity_type="world_state",
            entity_id=f"state-{status}",
            field_path=f"storm.{status}",
            proposed_value={"mood": status},
            confidence=0.99,
            status=status,
            suggestion_id=f"suggestion-{status}",
        )

    sources = pending_context_suggestion_sources(
        repositories=repositories,
        save_id=save.id,
    )
    assembled = ContextAssemblyService(repositories).assemble_narrator_context(save.id)
    narrator_context = "\n".join(assembled.current_scene_context)

    assert [source.source_id for source in sources] == ["suggestion-pending"]
    assert "storm.pending" in narrator_context
    assert "storm.applied" not in narrator_context
    assert "storm.rejected" not in narrator_context
    assert "storm.dismissed" not in narrator_context


def test_deterministic_context_sources_include_active_linked_facts(
    repositories: PersistenceRepositories,
) -> None:
    active_memory_text = "Captain Ilyra knows the lens-key phrase: ember dawn."
    active_summary_text = (
        "Ilyra and the warden discovered the red lens will shatter soon."
    )
    active_scenario_text = "The beacon gallery has a cracked red lens."
    active_world_state_text = "beacon.lens: color: red, failsafe: copper notch"
    inactive_memory_text = "Archivist Lio hid a map in the flooded crypt."
    inactive_summary_text = "Lio cataloged the crypt drains far from the current scene."
    inactive_scenario_text = "The flooded crypt should stay inactive."
    inactive_world_state_key = "crypt.waterline"

    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "beacon_gallery": active_scenario_text,
            "flooded_crypt": inactive_scenario_text,
        },
        scenario_id="scenario-linked-facts",
    )
    save = repositories.create_save(
        scenario_id=scenario.id,
        title="Lens Watch",
        save_id="save-linked-facts",
    )
    current_location = repositories.add_location(
        save_id=save.id,
        name="Beacon Gallery",
        description="A high room of red glass and ash-stained brass.",
        location_id="location-beacon-gallery",
    )
    inactive_location = repositories.add_location(
        save_id=save.id,
        name="Flooded Crypt",
        description="A drowned archive below the keep.",
        location_id="location-flooded-crypt",
    )
    present_character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        role="Watch captain",
        location_id=current_location.id,
        met=True,
        character_id="character-ilyra",
    )
    non_present_character = repositories.add_character(
        save_id=save.id,
        name="Archivist Lio",
        role="Archivist",
        location_id=inactive_location.id,
        met=True,
        character_id="character-lio",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=current_location.id,
        situation="The red lens ticks under stress.",
        present_character_ids=[present_character.id],
        snapshot_id="snapshot-linked-facts",
    )
    first_message = repositories.append_message(
        save_id=save.id,
        role="player",
        body="I check the lens housing.",
        message_id="message-first",
    )
    last_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The brass notch is warm.",
        message_id="message-last",
    )
    active_memory = repositories.add_memory(
        save_id=save.id,
        body=active_memory_text,
        tags=["ilyra", "lens"],
        memory_id="memory-active-lens-key",
    )
    inactive_memory = repositories.add_memory(
        save_id=save.id,
        body=inactive_memory_text,
        tags=["lio", "crypt"],
        memory_id="memory-inactive-crypt-map",
    )
    active_world_state = repositories.upsert_world_state(
        save_id=save.id,
        key="beacon.lens",
        value={"color": "red", "failsafe": "copper notch"},
        category="artifact",
        state_id="world-state-active-lens",
    )
    inactive_world_state = repositories.upsert_world_state(
        save_id=save.id,
        key=inactive_world_state_key,
        value={"level": "high"},
        category="location",
        state_id="world-state-inactive-crypt",
    )
    active_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=first_message.id,
        covers_message_end_id=last_message.id,
        body=active_summary_text,
        provider="fake",
        model="fake-model",
        summary_id="summary-active-lens",
    )
    inactive_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=first_message.id,
        covers_message_end_id=last_message.id,
        body=inactive_summary_text,
        provider="fake",
        model="fake-model",
        summary_id="summary-inactive-crypt",
    )
    active_scenario_target = f"scenario:{scenario.id}:section:beacon_gallery"
    inactive_scenario_target = f"scenario:{scenario.id}:section:flooded_crypt"

    for entity_type, entity_id in (
        ("location", current_location.id),
        ("character", present_character.id),
    ):
        for target_type, target_id in (
            ("memory", active_memory.id),
            ("world_state", active_world_state.id),
            ("summary", active_summary.id),
            ("scenario_section", active_scenario_target),
        ):
            repositories.add_entity_link(
                save_id=save.id,
                entity_type=entity_type,
                entity_id=entity_id,
                target_type=target_type,
                target_id=target_id,
            )
    for entity_type, entity_id in (
        ("location", inactive_location.id),
        ("character", non_present_character.id),
    ):
        for target_type, target_id in (
            ("memory", inactive_memory.id),
            ("world_state", inactive_world_state.id),
            ("summary", inactive_summary.id),
            ("scenario_section", inactive_scenario_target),
        ):
            repositories.add_entity_link(
                save_id=save.id,
                entity_type=entity_type,
                entity_id=entity_id,
                target_type=target_type,
                target_id=target_id,
            )

    sources = deterministic_context_sources(repositories=repositories, save_id=save.id)

    active_linked_text = "\n".join(
        source.text for source in sources if source.tier == "active_linked_facts"
    )
    assert active_linked_text
    for expected in (
        active_memory_text,
        active_world_state_text,
        active_summary_text,
        active_scenario_text,
    ):
        assert expected in active_linked_text
    for excluded in (
        inactive_memory_text,
        inactive_world_state_key,
        inactive_summary_text,
        inactive_scenario_text,
    ):
        assert excluded not in active_linked_text

    assembled = ContextAssemblyService(repositories).assemble_narrator_context(save.id)
    current_scene_text = "\n".join(assembled.current_scene_context)
    for expected in (
        active_memory_text,
        active_world_state_text,
        active_summary_text,
        active_scenario_text,
    ):
        assert expected in current_scene_text
    for excluded in (
        inactive_memory_text,
        inactive_world_state_key,
        inactive_summary_text,
        inactive_scenario_text,
    ):
        assert excluded not in current_scene_text


def test_narrator_context_includes_rich_present_character_details(
    repositories: PersistenceRepositories,
) -> None:
    _scenario, save, current_location = _create_context_save(
        repositories,
        scenario_id="scenario-rich-present-character",
        save_id="save-rich-present-character",
    )
    elsewhere = repositories.add_location(
        save_id=save.id,
        name="Flooded Crypt",
        description="A drowned archive below the keep.",
        location_id="location-rich-crypt",
    )
    present_character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        aliases=["Ashknife", "Glass-Eye"],
        role="Watch captain",
        known_state="Guarding the cracked red lens",
        met=True,
        appearance="Tall captain in a scorched blue watchcoat",
        visual_notes="Copper lens-key on a black cord",
        current_clothing="Borrowed green raincoat over a linen shirt",
        personality="Dry humor under pressure",
        voice="Low, clipped commands",
        relationships={
            "Archivist Lio": "keeps him at arm's length",
            "Signal warden": "trusts them with the lens key",
        },
        goals="Keep the red lens under control until dawn",
        motivations="Protect the lower village from ash riders",
        current_intent="Demand proof before unlocking the failsafe",
        boundaries="Will not leave the tower while the lens is unstable",
        attitude_toward_player="Wary trust after the last repair",
        cooperation_conditions="Shares the failsafe if Mara shows the brass warrant",
        status="Bleeding from a brass-cut palm",
        location_id=current_location.id,
        private_notes="Conceals that the lens key is a family heirloom",
        character_id="character-rich-ilyra",
    )
    repositories.add_character(
        save_id=save.id,
        name="Archivist Lio",
        aliases=["Inkhand"],
        role="Archivist",
        known_state="Searching the drowned stacks",
        met=True,
        appearance="Ink-stained archivist in a drowned velvet coat",
        visual_notes="Silver map tube hidden under one sleeve",
        personality="Needles every uncertain claim",
        voice="Soft and precise",
        relationships={"Captain Ilyra": "resents her secrecy"},
        status="Missing below the keep",
        location_id=elsewhere.id,
        private_notes="Plans to sell the flooded-map cipher",
        character_id="character-rich-lio",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=current_location.id,
        situation="The red lens ticks under stress.",
        present_character_ids=[present_character.id],
        snapshot_id="snapshot-rich-present-character",
    )

    assembled = ContextAssemblyService(repositories).assemble_narrator_context(save.id)
    current_scene_text = "\n".join(assembled.current_scene_context)

    for expected in (
        "Captain Ilyra (aliases: Ashknife, Glass-Eye)",
        "role: Watch captain",
        "known state: Guarding the cracked red lens",
        "status: Bleeding from a brass-cut palm",
        "appearance: Tall captain in a scorched blue watchcoat",
        "visual notes: Copper lens-key on a black cord",
        "current clothing: Borrowed green raincoat over a linen shirt",
        "personality: Dry humor under pressure",
        "voice: Low, clipped commands",
        "relationships: Archivist Lio: keeps him at arm's length",
        "Signal warden: trusts them with the lens key",
        "goals: Keep the red lens under control until dawn",
        "motivations: Protect the lower village from ash riders",
        "current intent: Demand proof before unlocking the failsafe",
        "boundaries: Will not leave the tower while the lens is unstable",
        "attitude toward player: Wary trust after the last repair",
        "cooperation conditions: Shares the failsafe if Mara shows the brass warrant",
        "narrator-only private notes for this character; do not treat as known "
        "by other characters: Conceals that the lens key is a family heirloom",
    ):
        assert expected in current_scene_text
    for excluded in (
        "aliases: Inkhand",
        "role: Archivist",
        "Searching the drowned stacks",
        "Ink-stained archivist",
        "Silver map tube",
        "Needles every uncertain claim",
        "Plans to sell the flooded-map cipher",
    ):
        assert excluded not in current_scene_text


def test_character_knows_linked_facts_are_attributed_as_character_scoped(
    repositories: PersistenceRepositories,
) -> None:
    _scenario, save, current_location = _create_context_save(
        repositories,
        scenario_id="scenario-character-knows-linked-facts",
        save_id="save-character-knows-linked-facts",
    )
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        role="Watch captain",
        location_id=current_location.id,
        met=True,
        character_id="character-knows-ilyra",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=current_location.id,
        situation="Ilyra studies the beacon lens.",
        present_character_ids=[character.id],
        snapshot_id="snapshot-character-knows",
    )
    first_message = repositories.append_message(
        save_id=save.id,
        role="player",
        body="I ask Ilyra what she knows.",
        message_id="message-knows-first",
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="The lens-key phrase is ember dawn.",
        tags=["ilyra"],
        memory_id="memory-character-knows",
    )
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="beacon.lens",
        value={"failsafe": "copper notch"},
        state_id="world-state-character-knows",
    )
    summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=first_message.id,
        covers_message_end_id=first_message.id,
        body="Ilyra admitted the red lens can be cooled with the hidden notch.",
        provider="fake",
        model="fake-summary",
        summary_id="summary-character-knows",
    )
    for target_type, target_id in (
        ("memory", memory.id),
        ("world_state", state.id),
        ("summary", summary.id),
    ):
        repositories.add_entity_link(
            save_id=save.id,
            entity_type="character",
            entity_id=character.id,
            target_type=target_type,
            target_id=target_id,
            relation="knows",
        )

    sources = deterministic_context_sources(repositories=repositories, save_id=save.id)

    linked_text = "\n".join(
        source.text for source in sources if source.tier == "active_linked_facts"
    )
    assert (
        "Character-scoped knowledge (Captain Ilyra knows) linked memory: "
        "The lens-key phrase is ember dawn."
    ) in linked_text
    assert (
        "Character-scoped knowledge (Captain Ilyra knows) linked world state: "
        "beacon.lens: failsafe: copper notch"
    ) in linked_text
    assert (
        "Character-scoped knowledge (Captain Ilyra knows) linked summary: "
        "Ilyra admitted the red lens can be cooled with the hidden notch."
    ) in linked_text


def test_character_knowledge_edges_are_attributed_as_character_scoped(
    repositories: PersistenceRepositories,
) -> None:
    _scenario, save, current_location = _create_context_save(
        repositories,
        scenario_id="scenario-character-knowledge-edges",
        save_id="save-character-knowledge-edges",
    )
    present = repositories.add_character(
        save_id=save.id,
        name="Nira",
        role="Archive guide",
        location_id=current_location.id,
        met=True,
        character_id="character-knowledge-nira",
    )
    absent = repositories.add_character(
        save_id=save.id,
        name="Tarin",
        role="Mapmaker",
        met=True,
        character_id="character-knowledge-tarin",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=current_location.id,
        situation="Nira has just arrived.",
        present_character_ids=[present.id],
        snapshot_id="snapshot-character-knowledge",
    )
    first_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="Tarin heard Avery make the archive-code joke earlier.",
        message_id="message-knowledge-first",
    )
    visible_memory = repositories.add_memory(
        save_id=save.id,
        body="Nira knows Avery invited her into the chart room.",
        tags=["nira"],
        memory_id="memory-nira-invite",
    )
    hidden_memory = repositories.add_memory(
        save_id=save.id,
        body="Tarin knows Avery made the archive-code joke within five minutes.",
        tags=["tarin"],
        memory_id="memory-tarin-archive-code-joke",
    )
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="archive.door",
        value={"status": "open"},
        state_id="world-state-nira-door",
    )
    summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=first_message.id,
        covers_message_end_id=first_message.id,
        body="Nira arrived after the earlier archive-code joke.",
        provider="fake",
        model="fake-summary",
        summary_id="summary-nira-arrival",
    )
    for target_type, target_id in (
        ("memory", visible_memory.id),
        ("world_state", state.id),
        ("summary", summary.id),
    ):
        repositories.add_character_knowledge_edge(
            save_id=save.id,
            character_id=present.id,
            target_type=target_type,
            target_id=target_id,
            knowledge_state="knows",
            acquisition_method="witnessed",
        )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=absent.id,
        target_type="memory",
        target_id=hidden_memory.id,
        knowledge_state="knows",
        acquisition_method="witnessed",
    )

    sources = deterministic_context_sources(repositories=repositories, save_id=save.id)

    linked_text = "\n".join(
        source.text for source in sources if source.tier == "active_linked_facts"
    )
    assert (
        "Character-scoped knowledge (Nira knows) linked memory: "
        "Nira knows Avery invited her into the chart room."
    ) in linked_text
    assert (
        "Character-scoped knowledge (Nira knows) linked world state: "
        "archive.door: status: open"
    ) in linked_text
    assert (
        "Character-scoped knowledge (Nira knows) linked summary: "
        "Nira arrived after the earlier archive-code joke."
    ) in linked_text
    assert "Tarin knows Avery made the archive-code joke" not in linked_text


def test_character_knowledge_edges_hidden_from_present_scene_are_not_linked_facts(
    repositories: PersistenceRepositories,
) -> None:
    _scenario, save, current_location = _create_context_save(
        repositories,
        scenario_id="scenario-hidden-knowledge-edge-linked-facts",
        save_id="save-hidden-knowledge-edge-linked-facts",
    )
    present = repositories.add_character(
        save_id=save.id,
        name="Nira",
        role="Archive guide",
        location_id=current_location.id,
        met=True,
        character_id="character-hidden-knowledge-nira",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=current_location.id,
        situation="Nira has just arrived.",
        present_character_ids=[present.id],
        snapshot_id="snapshot-hidden-knowledge-edge",
    )
    hidden_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="Tarin whispered that the moonstone opens the cobalt ledger.",
        message_id="message-hidden-knowledge-edge",
    )
    hidden_memory = repositories.add_memory(
        save_id=save.id,
        body="Tarin knows the moonstone opens the cobalt ledger.",
        tags=["tarin"],
        memory_id="memory-hidden-knowledge-edge",
        source_message_id=hidden_message.id,
    )
    repositories.add_message_visibility(
        save_id=save.id,
        message_id=hidden_message.id,
        character_id=present.id,
        visibility="not_visible",
        source="scene_presence",
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=present.id,
        target_type="memory",
        target_id=hidden_memory.id,
        knowledge_state="knows",
        acquisition_method="witnessed",
        source_message_id=hidden_message.id,
    )

    sources = deterministic_context_sources(repositories=repositories, save_id=save.id)

    linked_text = "\n".join(
        source.text for source in sources if source.tier == "active_linked_facts"
    )
    assert "Tarin knows the moonstone opens the cobalt ledger" not in linked_text


def test_source_less_character_link_does_not_hydrate_hidden_source_memory(
    repositories: PersistenceRepositories,
) -> None:
    _scenario, save, current_location = _create_context_save(
        repositories,
        scenario_id="scenario-hidden-link-target",
        save_id="save-hidden-link-target",
    )
    present = repositories.add_character(
        save_id=save.id,
        name="Nira",
        role="Archive guide",
        location_id=current_location.id,
        met=True,
        character_id="character-hidden-link-nira",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=current_location.id,
        situation="Nira has just arrived.",
        present_character_ids=[present.id],
        snapshot_id="snapshot-hidden-link-target",
    )
    hidden_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="Tarin whispered that the moonstone opens the cobalt ledger.",
        message_id="message-hidden-link-target",
    )
    hidden_memory = repositories.add_memory(
        save_id=save.id,
        body="Tarin knows the moonstone opens the cobalt ledger.",
        tags=["tarin"],
        memory_id="memory-hidden-link-target",
        source_message_id=hidden_message.id,
    )
    repositories.add_message_visibility(
        save_id=save.id,
        message_id=hidden_message.id,
        character_id=present.id,
        visibility="not_visible",
        source="scene_presence",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=present.id,
        target_type="memories",
        target_id=hidden_memory.id,
        relation="knows",
    )

    sources = deterministic_context_sources(repositories=repositories, save_id=save.id)

    linked_text = "\n".join(
        source.text for source in sources if source.tier == "active_linked_facts"
    )
    assert "Tarin knows the moonstone opens the cobalt ledger" not in linked_text


def test_active_thread_hidden_from_present_scene_is_omitted(
    repositories: PersistenceRepositories,
) -> None:
    _scenario, save, current_location = _create_context_save(
        repositories,
        scenario_id="scenario-hidden-thread-source",
        save_id="save-hidden-thread-source",
    )
    present = repositories.add_character(
        save_id=save.id,
        name="Nira",
        role="Archive guide",
        location_id=current_location.id,
        met=True,
        character_id="character-hidden-thread-nira",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=current_location.id,
        situation="Nira has just arrived.",
        present_character_ids=[present.id],
        snapshot_id="snapshot-hidden-thread-source",
    )
    hidden_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The hidden thread is to retrieve Tarin's cobalt ledger.",
        message_id="message-hidden-thread-source",
    )
    repositories.add_message_visibility(
        save_id=save.id,
        message_id=hidden_message.id,
        character_id=present.id,
        visibility="not_visible",
        source="scene_presence",
    )
    repositories.add_active_thread(
        save_id=save.id,
        title="Retrieve Tarin's cobalt ledger",
        description="The hidden thread description should not enter context.",
        status="active",
        priority=9,
        source_message_id=hidden_message.id,
    )

    sources = deterministic_context_sources(repositories=repositories, save_id=save.id)

    active_thread_text = "\n".join(
        source.text for source in sources if source.tier == "active_threads"
    )
    assert "Retrieve Tarin's cobalt ledger" not in active_thread_text


def test_private_active_threads_are_limited_to_turn_audience(
    repositories: PersistenceRepositories,
) -> None:
    _scenario, save, current_location = _create_context_save(
        repositories,
        scenario_id="scenario-private-active-thread-audience",
        save_id="save-private-active-thread-audience",
    )
    rowan = repositories.add_character(
        save_id=save.id,
        name="Rowan",
        role="repair club",
        location_id=current_location.id,
        met=True,
        character_id="character-private-thread-rowan",
    )
    cass = repositories.add_character(
        save_id=save.id,
        name="Cass",
        role="club president",
        location_id=current_location.id,
        met=True,
        character_id="character-private-thread-cass",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=current_location.id,
        situation="Rowan waits by the beacon lens while Cass is away.",
        present_character_ids=[rowan.id],
    )
    repositories.add_active_thread(
        save_id=save.id,
        title="Rowan repair follow-up",
        description="Rowan still needs to bring the repair notes.",
        status="active",
        priority=4,
        visibility="public",
        related_entities=[f"character:{rowan.id}"],
    )
    repositories.add_active_thread(
        save_id=save.id,
        title="Cass private festival letter",
        description="Cass is deciding whether to send a private festival letter.",
        status="active",
        priority=8,
        visibility="private",
        related_entities=[f"character:{cass.id}"],
    )

    sources = deterministic_context_sources(repositories=repositories, save_id=save.id)

    active_thread_text = "\n".join(
        source.text for source in sources if source.tier == "active_threads"
    )
    assert "Rowan repair follow-up" in active_thread_text
    assert "Cass private festival letter" not in active_thread_text


def test_active_participant_relationship_state_is_deterministic_context(
    repositories: PersistenceRepositories,
) -> None:
    _scenario, save, current_location = _create_context_save(
        repositories,
        scenario_id="scenario-active-participant-state",
        save_id="save-active-participant-state",
    )
    player_character = repositories.add_character(
        save_id=save.id,
        name="Mara",
        role="Signal warden",
        character_id="character-active-mara",
    )
    present_character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        aliases=["Ilyra"],
        role="Watch captain",
        character_id="character-active-ilyra",
    )
    repositories.add_character(
        save_id=save.id,
        name="Archivist Lio",
        role="Archivist",
        character_id="character-active-lio",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=current_location.id,
        situation="Ilyra watches Mara clean ash from the beacon rifle.",
        present_character_ids=[player_character.id, present_character.id],
        snapshot_id="snapshot-active-participant-state",
    )
    prior_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="Ilyra watched Mara stop the ash raiders.",
        message_id="message-active-participant-prior",
    )
    active_trait = repositories.upsert_world_state(
        save_id=save.id,
        key="character.captain_ilyra.revealed_traits.about_mara",
        value={
            "knows": (
                "Ilyra already watched Mara kill the ash raiders during the "
                "gate rescue."
            )
        },
        category="relationship",
        source_message_id=prior_message.id,
        state_id="world-state-ilyra-knows-mara",
    )
    active_relationship = repositories.upsert_world_state(
        save_id=save.id,
        key="relationship.player_to_captain_ilyra.current_standing",
        value={
            "trust": (
                "Ilyra knows Mara is dangerous in a fight and trusts her with "
                "the beacon rifle."
            )
        },
        category="relationship",
        source_message_id=prior_message.id,
        state_id="world-state-player-ilyra-standing",
    )
    active_preferences = repositories.upsert_world_state(
        save_id=save.id,
        key="character.captain_ilyra.preferences",
        value={"likes": "Plain warnings before clever plans."},
        category="relationship",
        source_message_id=prior_message.id,
        state_id="world-state-ilyra-preferences",
    )
    active_boundaries = repositories.upsert_world_state(
        save_id=save.id,
        key="character.captain_ilyra.boundaries",
        value={"refuses": "Will not abandon wounded watch members."},
        category="relationship",
        source_message_id=prior_message.id,
        state_id="world-state-ilyra-boundaries",
    )
    active_emotion = repositories.upsert_world_state(
        save_id=save.id,
        key="character.captain_ilyra.current_emotional_state",
        value={"mood": "Wary but warming to Mara."},
        category="scene",
        source_message_id=prior_message.id,
        state_id="world-state-ilyra-emotion",
    )
    inactive_trait = repositories.upsert_world_state(
        save_id=save.id,
        key="character.archivist_lio.revealed_traits.about_mara",
        value={"knows": "Lio heard only rumors about Mara."},
        category="relationship",
        source_message_id=prior_message.id,
        state_id="world-state-lio-knows-mara",
    )
    inactive_preference = repositories.upsert_world_state(
        save_id=save.id,
        key="character.archivist_lio.preferences",
        value={"likes": "Unrelated crypt cataloging."},
        category="relationship",
        source_message_id=prior_message.id,
        state_id="world-state-lio-preferences",
    )

    sources = deterministic_context_sources(repositories=repositories, save_id=save.id)

    participant_text = "\n".join(
        source.text
        for source in sources
        if source.tier == "active_participant_facts"
    )
    assert (
        "Active participant continuity: "
        "character.captain_ilyra.revealed_traits.about_mara: "
        "knows: Ilyra already watched Mara kill the ash raiders during the "
        "gate rescue."
    ) in participant_text
    assert (
        "Active participant continuity: "
        "relationship.player_to_captain_ilyra.current_standing: "
        "trust: Ilyra knows Mara is dangerous in a fight and trusts her with "
        "the beacon rifle."
    ) in participant_text
    assert (
        "Active participant continuity: "
        "character.captain_ilyra.preferences: "
        "likes: Plain warnings before clever plans."
    ) in participant_text
    assert (
        "Active participant continuity: "
        "character.captain_ilyra.boundaries: "
        "refuses: Will not abandon wounded watch members."
    ) in participant_text
    assert (
        "Active participant continuity: Captain Ilyra's current emotional state "
        "is Wary but warming to Mara."
    ) in participant_text
    assert inactive_trait.key not in participant_text
    assert inactive_preference.key not in participant_text
    assert {source.source_id for source in sources} >= {
        active_trait.id,
        active_relationship.id,
        active_preferences.id,
        active_boundaries.id,
        active_emotion.id,
    }

    assembled = ContextAssemblyService(repositories).assemble_narrator_context(save.id)
    current_scene_text = "\n".join(assembled.current_scene_context)
    assert "Ilyra already watched Mara kill the ash raiders" in current_scene_text
    assert "trusts her with the beacon rifle" in current_scene_text
    assert "Plain warnings before clever plans" in current_scene_text
    assert "Will not abandon wounded watch members" in current_scene_text
    assert "Captain Ilyra's current emotional state is Wary but warming to Mara" in (
        current_scene_text
    )
    assert "Lio heard only rumors" not in current_scene_text
    assert "Unrelated crypt cataloging" not in current_scene_text


def test_active_participant_state_hidden_from_present_scene_is_omitted(
    repositories: PersistenceRepositories,
) -> None:
    _scenario, save, current_location = _create_context_save(
        repositories,
        scenario_id="scenario-hidden-participant-state",
        save_id="save-hidden-participant-state",
    )
    present = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        aliases=["Ilyra"],
        role="Watch captain",
        character_id="character-hidden-participant-ilyra",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=current_location.id,
        situation="Ilyra watches the bridge.",
        present_character_ids=[present.id],
        snapshot_id="snapshot-hidden-participant-state",
    )
    hidden_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="Ilyra privately learned that Mara betrayed the bridge watch.",
        message_id="message-hidden-participant-state",
    )
    repositories.add_message_visibility(
        save_id=save.id,
        message_id=hidden_message.id,
        character_id=present.id,
        visibility="not_visible",
        source="scene_presence",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="character.captain_ilyra.revealed_traits.about_mara",
        value={"knows": "Ilyra knows Mara betrayed the bridge watch."},
        category="relationship",
        source_message_id=hidden_message.id,
    )

    sources = deterministic_context_sources(repositories=repositories, save_id=save.id)

    participant_text = "\n".join(
        source.text
        for source in sources
        if source.tier == "active_participant_facts"
    )
    assert "Ilyra knows Mara betrayed the bridge watch" not in participant_text


def test_narrator_context_does_not_render_player_agency_as_npc_guidance(
    repositories: PersistenceRepositories,
) -> None:
    _scenario, save, current_location = _create_context_save(
        repositories,
        scenario_id="scenario-player-agency-context",
        save_id="save-player-agency-context",
    )
    player_character = repositories.add_character(
        save_id=save.id,
        name="Mara",
        role="Signal warden",
        goals="Find the red lens failsafe.",
        current_intent="Ask Ilyra for the failsafe phrase.",
        cooperation_conditions="Not applicable to the player.",
        location_id=current_location.id,
        character_id="character-player-mara",
        is_player_character=True,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=current_location.id,
        situation="The red lens trembles.",
        present_character_ids=[player_character.id],
        snapshot_id="snapshot-player-agency-context",
    )

    assembled = ContextAssemblyService(repositories).assemble_narrator_context(save.id)
    current_scene_text = "\n".join(assembled.current_scene_context)

    assert "Mara" in current_scene_text
    assert "goals: Find the red lens failsafe." not in current_scene_text
    assert (
        "current intent: Ask Ilyra for the failsafe phrase." not in current_scene_text
    )
    assert (
        "cooperation conditions: Not applicable to the player."
        not in current_scene_text
    )


def test_image_context_present_characters_stays_visual_focused(
    repositories: PersistenceRepositories,
) -> None:
    _scenario, save, current_location = _create_context_save(
        repositories,
        scenario_id="scenario-image-present-character",
        save_id="save-image-present-character",
    )
    present_character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        aliases=["Ashknife"],
        role="Secret oathkeeper",
        known_state="Knows the oath phrase ember dawn",
        met=True,
        appearance="Tall captain in a scorched blue watchcoat",
        visual_notes="Copper lens-key on a black cord",
        current_clothing="Borrowed green raincoat over a linen shirt",
        personality="Dry humor under pressure",
        voice="Low, clipped commands",
        relationships={"Signal warden": "trusts them with the lens key"},
        goals="Keep the red lens under control until dawn",
        motivations="Protect the lower village from ash riders",
        current_intent="Demand proof before unlocking the failsafe",
        boundaries="Will not leave the tower while the lens is unstable",
        attitude_toward_player="Wary trust after the last repair",
        cooperation_conditions="Shares the failsafe if Mara shows the brass warrant",
        status="Blood on her brass-cut palm",
        location_id=current_location.id,
        private_notes="Conceals that the lens key is a family heirloom",
        character_id="character-image-ilyra",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=current_location.id,
        situation="The red lens throws harsh light across the gallery.",
        present_character_ids=[present_character.id],
        snapshot_id="snapshot-image-present-character",
    )

    image_context, _breakdown = ContextAssemblyService(
        repositories
    ).build_image_scene_context(save_id=save.id)

    for expected in (
        "Captain Ilyra",
        "Tall captain in a scorched blue watchcoat",
        "Copper lens-key on a black cord",
        "Borrowed green raincoat over a linen shirt",
        "status: Blood on her brass-cut palm",
    ):
        assert expected in image_context
    for excluded in (
        "role: Secret oathkeeper",
        "Knows the oath phrase ember dawn",
        "personality: Dry humor under pressure",
        "voice: Low, clipped commands",
        "relationships:",
        "Keep the red lens under control",
        "Demand proof before unlocking the failsafe",
        "Shares the failsafe",
        "aliases: Ashknife",
        "Conceals that the lens key is a family heirloom",
    ):
        assert excluded not in image_context


def test_resolved_and_abandoned_threads_are_excluded_from_context(
    repositories: PersistenceRepositories,
) -> None:
    _scenario, save, current_location = _create_context_save(
        repositories,
        scenario_id="scenario-thread-filtering",
        save_id="save-thread-filtering",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=current_location.id,
        situation="The warden listens for unfinished crises.",
        present_character_ids=[],
        snapshot_id="snapshot-thread-filtering",
    )
    active_thread = repositories.add_active_thread(
        save_id=save.id,
        title="Stabilize the red lens",
        description="The active thread description keeps the cracked lens alive.",
        status="active",
        priority=10,
        thread_id="thread-active-lens",
    )
    resolved_thread = repositories.add_active_thread(
        save_id=save.id,
        title="Repair the old gate",
        description="The resolved thread description should not enter context.",
        status="resolved",
        priority=9,
        thread_id="thread-resolved-gate",
    )
    abandoned_thread = repositories.add_active_thread(
        save_id=save.id,
        title="Search the west quarry",
        description="The abandoned thread description should not enter context.",
        status="abandoned",
        priority=8,
        thread_id="thread-abandoned-quarry",
    )
    completed_thread = repositories.add_active_thread(
        save_id=save.id,
        title="Settle the dinner promise",
        description="The completed thread description should not enter context.",
        status="Completed",
        priority=7,
        thread_id="thread-completed-dinner",
    )
    active_memory = repositories.add_memory(
        save_id=save.id,
        body="The red lens hums only while the copper notch is pressed.",
        tags=[],
        memory_id="memory-active-thread-lens",
    )
    resolved_memory = repositories.add_memory(
        save_id=save.id,
        body="The old gate was repaired before the ash storm changed direction.",
        tags=[],
        memory_id="memory-resolved-thread-gate",
    )
    abandoned_memory = repositories.add_memory(
        save_id=save.id,
        body="The west quarry lead went cold after the trail collapsed.",
        tags=[],
        memory_id="memory-abandoned-thread-quarry",
    )
    completed_memory = repositories.add_memory(
        save_id=save.id,
        body="Mara settled the dinner promise before leaving the steakhouse.",
        tags=[],
        memory_id="memory-completed-thread-dinner",
    )
    for thread, memory in (
        (active_thread, active_memory),
        (resolved_thread, resolved_memory),
        (abandoned_thread, abandoned_memory),
        (completed_thread, completed_memory),
    ):
        repositories.add_entity_link(
            save_id=save.id,
            entity_type="active_thread",
            entity_id=thread.id,
            target_type="memory",
            target_id=memory.id,
        )

    deterministic_text = "\n".join(
        source.text
        for source in deterministic_context_sources(
            repositories=repositories,
            save_id=save.id,
        )
    )
    narrator_text = "\n".join(
        ContextAssemblyService(repositories)
        .assemble_narrator_context(save.id)
        .current_scene_context
    )
    image_context, _breakdown = ContextAssemblyService(
        repositories
    ).build_image_scene_context(save_id=save.id)

    for context_text in (deterministic_text, narrator_text, image_context):
        assert "Stabilize the red lens" in context_text
        assert (
            "The active thread description keeps the cracked lens alive."
            in context_text
        )
        assert active_memory.body in context_text
        assert "Repair the old gate" not in context_text
        assert (
            "The resolved thread description should not enter context."
            not in context_text
        )
        assert resolved_memory.body not in context_text
        assert "Search the west quarry" not in context_text
        assert (
            "The abandoned thread description should not enter context."
            not in context_text
        )
        assert abandoned_memory.body not in context_text
        assert "Settle the dinner promise" not in context_text
        assert (
            "The completed thread description should not enter context."
            not in context_text
        )
        assert completed_memory.body not in context_text


def test_active_thread_context_excludes_linked_aggregate_open_thread_state(
    repositories: PersistenceRepositories,
) -> None:
    _scenario, save, current_location = _create_context_save(
        repositories,
        scenario_id="scenario-thread-aggregate-dedupe",
        save_id="save-thread-aggregate-dedupe",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=current_location.id,
        situation="The warden listens for unfinished promises.",
        snapshot_id="snapshot-thread-aggregate-dedupe",
    )
    thread = repositories.add_active_thread(
        save_id=save.id,
        title="Dinner promise",
        description="Mara still owes Ilyra dinner after the beacon is safe.",
        status="active",
        priority=4,
        thread_id="thread-dinner-promise",
    )
    aggregate = repositories.upsert_world_state(
        save_id=save.id,
        key="interaction.open_threads",
        value={"dinner": "This aggregate should not enter prompt context."},
        category="open_threads",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="active_thread",
        entity_id=thread.id,
        target_type="world_state",
        target_id=aggregate.id,
    )

    context_text = "\n".join(
        source.text
        for source in deterministic_context_sources(
            repositories=repositories,
            save_id=save.id,
        )
    )

    assert "Dinner promise" in context_text
    assert "Mara still owes Ilyra dinner" in context_text
    assert "interaction.open_threads" not in context_text
    assert "This aggregate should not enter prompt context" not in context_text


def test_deterministic_context_sources_include_survival_expedition_state(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="survival_expedition",
        title="Whiteout Pass",
        premise="A rescue caravan must cross a frozen mountain pass.",
        player_role="Expedition lead",
        content={"expedition_goal": "Reach Northwatch."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Northwatch Run")
    repositories.upsert_world_state(
        save_id=save.id,
        key="expedition.resources",
        value={"summary": "Food: 2 days. Water: 1 skin."},
        category="inventory",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="expedition.progress",
        value={"summary": "18 of 80 miles traveled; blizzard delay active."},
        category="objective",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="inventory.unrelated",
        value={"summary": "This row is not part of the expedition ledger."},
        category="inventory",
    )

    sources = deterministic_context_sources(repositories=repositories, save_id=save.id)
    assembled = ContextAssemblyService(repositories).assemble_narrator_context(save.id)
    source_text = "\n".join(source.text for source in sources)
    narrator_context = "\n".join(assembled.current_scene_context)

    assert "Current expedition state" in source_text
    assert "expedition.resources: summary: Food: 2 days" in narrator_context
    assert "expedition.progress: summary: 18 of 80 miles" in narrator_context
    assert "inventory.unrelated" not in narrator_context


def test_deterministic_context_sources_include_first_contact_state(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="first_contact_exploration",
        title="Songs Under Europa",
        premise="A survey crew finds patterned signals beneath the ice.",
        player_role="Mission linguist",
        content={"mission_profile": "Survey the hidden ocean."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Europa Contact")
    repositories.upsert_world_state(
        save_id=save.id,
        key="contact.translation",
        value={"summary": "Three descending pulses may mean open water."},
        category="translation",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="contact.hazards",
        value={"summary": "Thermal fissures are spreading."},
        category="threat",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="sample.spores.contamination_risk",
        value={"status": "quarantined"},
        category="sample",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="inventory.unrelated",
        value={"summary": "This row is not part of the contact ledger."},
        category="inventory",
    )

    sources = deterministic_context_sources(repositories=repositories, save_id=save.id)
    assembled = ContextAssemblyService(repositories).assemble_narrator_context(save.id)
    source_text = "\n".join(source.text for source in sources)
    narrator_context = "\n".join(assembled.current_scene_context)

    assert "Current first-contact state" in source_text
    assert "contact.translation: summary: Three descending pulses" in narrator_context
    assert "contact.hazards: summary: Thermal fissures" in narrator_context
    assert "sample.spores.contamination_risk: status: quarantined" in narrator_context
    assert "inventory.unrelated" not in narrator_context


def test_deterministic_context_sources_include_heist_state(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="heist_infiltration",
        title="Skybank Treaty Job",
        premise="A crew must steal a treaty from a floating bank.",
        player_role="Crew planner",
        content={"security_model": "Clockwork cameras and warded locks."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Treaty Job")
    repositories.upsert_world_state(
        save_id=save.id,
        key="heist.security",
        value={"summary": "Clockwork cameras active; west lock disabled."},
        category="security",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="heist.alert",
        value={"level": "suspicious", "alarm": "inactive"},
        category="threat",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="security.unrelated",
        value={"summary": "This row is not part of the heist ledger."},
        category="security",
    )

    sources = deterministic_context_sources(repositories=repositories, save_id=save.id)
    assembled = ContextAssemblyService(repositories).assemble_narrator_context(save.id)
    source_text = "\n".join(source.text for source in sources)
    narrator_context = "\n".join(assembled.current_scene_context)

    assert "Current heist state" in source_text
    assert "heist.security: summary: Clockwork cameras active" in narrator_context
    assert "heist.alert:" in narrator_context
    assert "level: suspicious" in narrator_context
    assert "alarm: inactive" in narrator_context
    assert "security.unrelated" not in narrator_context


def test_deterministic_context_sources_include_time_loop_state(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="time_loop",
        title="Bellwether Day",
        premise="A harbor festival repeats until the drowned bell is saved.",
        player_role="Archivist who remembers the repeats.",
        content={"loop_premise": "The same dawn repeats."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Bell Loop")
    repositories.upsert_world_state(
        save_id=save.id,
        key="loop.knowledge",
        value={"summary": "Tower code and Mira's warning persist for the player."},
        category="loop_persistent",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="loop.npc_memory",
        value={"summary": "NPCs reset to dawn memories unless excepted."},
        category="loop_boundary",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="loop.current",
        value={
            "version": 1,
            "iteration": 2,
            "last_transition": "phase_advance",
            "current_time": {"phase": "evening", "clock_minutes": 1140},
            "summary": "The archive has been searched.",
        },
        category="loop_status",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="memory.unrelated",
        value={"summary": "This row is not part of the loop ledger."},
        category="memory",
    )

    sources = deterministic_context_sources(repositories=repositories, save_id=save.id)
    assembled = ContextAssemblyService(repositories).assemble_narrator_context(save.id)
    source_text = "\n".join(source.text for source in sources)
    narrator_context = "\n".join(assembled.current_scene_context)

    assert "Current time-loop state" in source_text
    assert "loop.knowledge: summary: Tower code" in narrator_context
    assert "loop.npc_memory: summary: NPCs reset" in narrator_context
    assert "loop iteration 2" in narrator_context
    assert "current_time" not in narrator_context
    assert "memory.unrelated" not in narrator_context


def test_deterministic_context_sources_include_political_intrigue_state(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="political_intrigue",
        title="Council of Ash",
        premise="A city council vote will decide who controls the harbor.",
        player_role="Envoy holding the swing vote.",
        content={"political_arena": "The harbor council chamber."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="First Vote")
    repositories.upsert_world_state(
        save_id=save.id,
        key="intrigue.obligations",
        value={"summary": "Orro owes Mara one public endorsement before midnight."},
        category="obligation",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="intrigue.standing",
        value={"summary": "Reformers trust Mara; Old Families distrust her."},
        category="reputation",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="faction.harbor_guild.standing",
        value={"toward_mara": "ally"},
        category="reputation",
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="memory.unrelated",
        value={"summary": "This row is outside the intrigue ledger."},
        category="memory",
    )

    sources = deterministic_context_sources(repositories=repositories, save_id=save.id)
    assembled = ContextAssemblyService(repositories).assemble_narrator_context(save.id)
    source_text = "\n".join(source.text for source in sources)
    narrator_context = "\n".join(assembled.current_scene_context)

    assert "Current political intrigue state" in source_text
    assert "intrigue.standing: summary: Reformers trust Mara" in narrator_context
    assert "intrigue.obligations: summary: Orro owes Mara" in narrator_context
    assert "faction.harbor_guild.standing: toward_mara: ally" in narrator_context
    assert "memory.unrelated" not in narrator_context


def _create_context_save(
    repositories: PersistenceRepositories,
    *,
    scenario_id: str,
    save_id: str,
) -> tuple[ScenarioRecord, SaveRecord, LocationRecord]:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
        scenario_id=scenario_id,
    )
    save = repositories.create_save(
        scenario_id=scenario.id,
        title="Lens Watch",
        save_id=save_id,
    )
    location = repositories.add_location(
        save_id=save.id,
        name="Beacon Gallery",
        description="A high room of red glass and ash-stained brass.",
        visual_description="Red glass gallery with ash-stained brass machinery.",
        location_id=f"location-{save_id}",
    )
    return scenario, save, location


def _scenario(
    content: dict[str, object],
    *,
    scenario_type: str = "full_roleplay",
) -> ScenarioRecord:
    return ScenarioRecord(
        id="scenario-1",
        type=scenario_type,
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content_json=json.dumps(content),
    )
