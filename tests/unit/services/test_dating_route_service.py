from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.interaction_mode import InteractionMode
from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.dating_route_service import DatingRouteService


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_seed_routes_for_dating_sim_existing_romance_characters(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, npc_id = _dating_save_with_romance_option(repositories)
    service = DatingRouteService(repositories)

    seeded = service.seed_routes_for_save(save_id)

    routes = repositories.list_dating_route_states(save_id)
    assert seeded == 1
    assert len(routes) == 1
    route = routes[0]
    assert route.npc_character_id == npc_id
    assert route.stage == "introduced"
    assert route.completed_interactions == 0
    assert route.dates_completed == 0
    assert route.next_reasonable_step == "build early interest or exchange contact info"


def test_storyteller_mode_skips_dating_route_seeding_and_progression(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, _npc_id = _dating_save_with_romance_option(
        repositories,
        interaction_mode=InteractionMode.STORYTELLER,
    )
    service = DatingRouteService(repositories)

    assert service.seed_routes_for_save(save_id) == 0
    player_message = repositories.append_message(
        save_id=save_id,
        role="player",
        body="Have them exchange numbers.",
    )
    narrator_message = repositories.append_message(
        save_id=save_id,
        role="narrator",
        body="Mika gives Ren her number.",
    )
    result = service.update_after_turn(
        save_id=save_id,
        player_message_id=player_message.id,
        narrator_message_id=narrator_message.id,
    )
    assert result.seeded_count == 0
    assert result.updated_count == 0
    assert repositories.list_dating_route_states(save_id) == []


def test_update_after_turn_advances_explicit_contact_exchange_and_counts_once(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_id, npc_id = _dating_save_with_romance_option(repositories)
    service = DatingRouteService(repositories)
    service.seed_routes_for_save(save_id)
    player_message = repositories.append_message(
        save_id=save_id,
        role="player",
        speaker_name="Ren",
        body="I ask Mika if we can exchange numbers before the festival ends.",
    )
    narrator_message = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Mika smiles and gives Ren her number, telling him to text her later.",
    )
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        present_character_ids=[npc_id],
        world_day_index=0,
        source_message_id=narrator_message.id,
    )

    service.update_after_turn(
        save_id=save_id,
        player_message_id=player_message.id,
        narrator_message_id=narrator_message.id,
    )
    service.update_after_turn(
        save_id=save_id,
        player_message_id=player_message.id,
        narrator_message_id=narrator_message.id,
    )

    route = repositories.list_dating_route_states(save_id)[0]
    assert route.stage == "contact_exchanged"
    assert route.completed_interactions == 1
    assert route.last_interaction_message_id == narrator_message.id
    assert route.last_interaction_world_day_index == 0
    assert (
        route.next_reasonable_step
        == "schedule a first date or follow-up interaction"
    )
    state = repositories.get_character_contact_state(
        save_id=save_id,
        player_character_id=player_id,
        character_id=npc_id,
    )
    assert state is not None
    assert state.player_has_character_number is True
    assert state.character_has_player_number is False


def test_update_after_turn_grants_reciprocal_contact_state_for_route_exchange(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_id, npc_id = _dating_save_with_romance_option(repositories)
    service = DatingRouteService(repositories)
    service.seed_routes_for_save(save_id)
    player_message = repositories.append_message(
        save_id=save_id,
        role="player",
        speaker_name="Ren",
        body="I tell Mika the festival was more fun with her there.",
    )
    narrator_message = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Mika Arai smiles at Ren Takahashi, and they exchange numbers.",
    )
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        present_character_ids=[npc_id],
        world_day_index=0,
        source_message_id=narrator_message.id,
    )

    service.update_after_turn(
        save_id=save_id,
        player_message_id=player_message.id,
        narrator_message_id=narrator_message.id,
    )

    state = repositories.get_character_contact_state(
        save_id=save_id,
        player_character_id=player_id,
        character_id=npc_id,
    )
    assert state is not None
    assert state.player_has_character_number is True
    assert state.character_has_player_number is True
    assert state.source_message_id == narrator_message.id


def test_update_after_turn_grants_character_inbound_only_contact_state(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_id, npc_id = _dating_save_with_romance_option(repositories)
    service = DatingRouteService(repositories)
    service.seed_routes_for_save(save_id)
    player_message = repositories.append_message(
        save_id=save_id,
        role="player",
        speaker_name="Ren",
        body="I say I hope we can talk again after the festival.",
    )
    narrator_message = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Ren Takahashi gives Mika Arai his number before heading home.",
    )
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        present_character_ids=[npc_id],
        world_day_index=0,
        source_message_id=narrator_message.id,
    )

    service.update_after_turn(
        save_id=save_id,
        player_message_id=player_message.id,
        narrator_message_id=narrator_message.id,
    )

    state = repositories.get_character_contact_state(
        save_id=save_id,
        player_character_id=player_id,
        character_id=npc_id,
    )
    assert state is not None
    assert state.player_has_character_number is False
    assert state.character_has_player_number is True
    assert state.source_message_id == narrator_message.id


def test_update_after_turn_does_not_treat_warmth_as_commitment(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, npc_id = _dating_save_with_romance_option(repositories)
    service = DatingRouteService(repositories)
    service.seed_routes_for_save(save_id)
    player_message = repositories.append_message(
        save_id=save_id,
        role="player",
        speaker_name="Ren",
        body="I tell Mika she made the whole day feel brighter.",
    )
    narrator_message = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Mika blushes, clearly touched, and admits she wants to talk again.",
    )
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        present_character_ids=[npc_id],
        world_day_index=0,
        source_message_id=narrator_message.id,
    )

    service.update_after_turn(
        save_id=save_id,
        player_message_id=player_message.id,
        narrator_message_id=narrator_message.id,
    )

    route = repositories.list_dating_route_states(save_id)[0]
    assert route.stage == "introduced"
    assert route.completed_interactions == 1
    assert route.dates_completed == 0
    assert "exclusive" not in route.next_reasonable_step


def test_update_after_turn_advances_route_through_deterministic_stages(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, npc_id = _dating_save_with_romance_option(repositories)
    service = DatingRouteService(repositories)
    service.seed_routes_for_save(save_id)

    def make_turn(player_body: str, narrator_body: str, world_day_index: int) -> None:
        player_message = repositories.append_message(
            save_id=save_id,
            role="player",
            speaker_name="Ren",
            body=player_body,
        )
        narrator_message = repositories.append_message(
            save_id=save_id,
            role="narrator",
            speaker_name="Narrator",
            body=narrator_body,
        )
        repositories.upsert_scene_snapshot(
            save_id=save_id,
            present_character_ids=[npc_id],
            world_day_index=world_day_index,
            source_message_id=narrator_message.id,
        )
        service.update_after_turn(
            save_id=save_id,
            player_message_id=player_message.id,
            narrator_message_id=narrator_message.id,
        )

    make_turn(
        "Would you like to go on a date this weekend, Mika?",
        "The invitation hangs in the air as Mika considers your words.",
        0,
    )
    route = repositories.list_dating_route_states(save_id)[0]
    assert route.stage == "introduced"
    assert route.completed_interactions == 1

    make_turn(
        "Can we exchange numbers so we can continue this?",
        "Mika smiles and gives you her number.",
        1,
    )
    route = repositories.list_dating_route_states(save_id)[0]
    assert route.stage == "contact_exchanged"
    assert route.completed_interactions == 2

    make_turn(
        "Let's plan a first date on Friday night.",
        "Mika laughs and agrees to meet on Friday.",
        1,
    )
    route = repositories.list_dating_route_states(save_id)[0]
    assert route.stage == "first_date_planned"
    assert route.completed_interactions == 3

    make_turn(
        "I can't wait. I'll be there at seven.",
        "The first date begins at the little Italian place.",
        2,
    )
    route = repositories.list_dating_route_states(save_id)[0]
    assert route.stage == "first_date_in_progress"
    assert route.completed_interactions == 4

    make_turn(
        "The evening wrapped up with a moonlit walk.",
        "After the first date, Mika thanks you for the evening.",
        3,
    )
    route = repositories.list_dating_route_states(save_id)[0]
    assert route.stage == "early_dating"
    assert route.completed_interactions == 5
    assert route.dates_completed == 1

    make_turn(
        "This feels like something real.",
        "Mika says you are exclusive.",
        4,
    )
    route = repositories.list_dating_route_states(save_id)[0]
    assert route.stage == "exclusive"
    assert route.completed_interactions == 6

    make_turn(
        "I'm committed to making this work for the long term.",
        "Mika says she is committed too.",
        5,
    )
    route = repositories.list_dating_route_states(save_id)[0]
    assert route.stage == "committed"
    assert route.completed_interactions == 7


def test_update_after_turn_reseed_does_not_reset_state(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, npc_id = _dating_save_with_romance_option(repositories)
    service = DatingRouteService(repositories)
    service.seed_routes_for_save(save_id)
    player_message = repositories.append_message(
        save_id=save_id,
        role="player",
        speaker_name="Ren",
        body="Can we exchange numbers so we can continue this?",
    )
    narrator_message = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Mika smiles and gives you her number.",
    )
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        present_character_ids=[npc_id],
        world_day_index=0,
        source_message_id=narrator_message.id,
    )
    service.update_after_turn(
        save_id=save_id,
        player_message_id=player_message.id,
        narrator_message_id=narrator_message.id,
    )
    assert (
        repositories.list_dating_route_states(save_id)[0].stage
        == "contact_exchanged"
    )

    assert service.seed_routes_for_save(save_id) == 0
    route = repositories.list_dating_route_states(save_id)[0]
    assert route.stage == "contact_exchanged"


def test_update_after_turn_uses_current_world_day_index_for_route_anchors(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, npc_id = _dating_save_with_romance_option(repositories)
    service = DatingRouteService(repositories)
    service.seed_routes_for_save(save_id)
    player_message = repositories.append_message(
        save_id=save_id,
        role="player",
        speaker_name="Ren",
        body="I ask Mika to meet after school.",
    )
    narrator_message = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Mika agrees to meet Ren by the station after school.",
    )
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        present_character_ids=[npc_id],
        world_day_index=4,
        source_message_id=narrator_message.id,
    )

    service.update_after_turn(
        save_id=save_id,
        player_message_id=player_message.id,
        narrator_message_id=narrator_message.id,
    )

    route = repositories.list_dating_route_states(save_id)[0]
    assert route.first_met_world_day_index == 4
    assert route.last_interaction_world_day_index == 4


def _dating_save_with_romance_option(
    repositories: PersistenceRepositories,
    *,
    interaction_mode: InteractionMode = InteractionMode.ROLEPLAY,
) -> tuple[str, str, str]:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Last Summer",
        premise="A summer of route choices.",
        player_role="Transfer student",
        interaction_mode=interaction_mode,
        content={
            "player_character_name": "Ren Takahashi",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Summer Save")
    player = repositories.add_character(
        save_id=save.id,
        name="Ren Takahashi",
        is_player_character=True,
        met=True,
    )
    npc = repositories.add_character(
        save_id=save.id,
        name="Mika Arai",
        relationships={player.name: "romance option for Ren Takahashi"},
        status="available romance option at scenario start",
        met=True,
    )
    return save.id, player.id, npc.id
