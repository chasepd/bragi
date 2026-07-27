from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.knowledge_boundary import allowed_character_scoped_targets
from bragi.services.narration_context import load_narration_context_snapshot


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_bounded_snapshot_keeps_hidden_graph_provenance_fail_closed(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Archive",
        premise="An archive holds unevenly shared secrets.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Index")
    hidden = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="Only absent witnesses learned the cobalt cipher.",
    )
    player = repositories.append_message(
        save_id=save.id,
        role="player",
        body="I ask Nira about the cipher.",
    )
    nira = repositories.add_character(save_id=save.id, name="Nira", met=True)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        present_character_ids=[nira.id],
    )
    repositories.add_message_visibility(
        save_id=save.id,
        message_id=hidden.id,
        character_id=nira.id,
        visibility="not_visible",
    )
    typed_memory = repositories.add_memory(
        save_id=save.id,
        body="The cobalt cipher opens the west ledger.",
        tags=["cipher"],
        source_message_id=hidden.id,
    )
    legacy_memory = repositories.add_memory(
        save_id=save.id,
        body="The cobalt seal opens the east ledger.",
        tags=["seal"],
        source_message_id=hidden.id,
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=nira.id,
        target_type="memory",
        target_id=typed_memory.id,
        knowledge_state="knows",
        acquisition_method="told",
        source_message_id=hidden.id,
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=nira.id,
        target_type="memory",
        target_id=legacy_memory.id,
        relation="knows",
        source_message_id=hidden.id,
    )

    snapshot = load_narration_context_snapshot(
        repositories,
        save_id=save.id,
        raw_record_limit=512,
    )

    assert snapshot is not None
    assert {edge.target_id for edge in snapshot.character_knowledge_edges} == {
        typed_memory.id
    }
    assert {link.target_id for link in snapshot.entity_links} == {legacy_memory.id}
    assert any(
        visibility.message_id == hidden.id
        and visibility.visibility == "not_visible"
        for visibility in snapshot.message_visibility
    )
    scoped = allowed_character_scoped_targets(
        scene_snapshot=snapshot.scene_snapshot,
        characters=list(snapshot.characters),
        character_knowledge_edges=list(snapshot.character_knowledge_edges),
        entity_links=list(snapshot.entity_links),
        latest_player_message=player.body,
        message_visibility=list(snapshot.message_visibility),
    )
    assert ("memory", typed_memory.id) in scoped.blocked
    assert ("memory", legacy_memory.id) in scoped.blocked
    assert scoped.allowed == {}


def test_bounded_snapshot_filters_private_threads_before_limit(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Archive",
        premise="An archive holds unevenly shared obligations.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Index")
    present = repositories.add_character(save_id=save.id, name="Nira", met=True)
    absent = repositories.add_character(save_id=save.id, name="Lio", met=True)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        present_character_ids=[present.id],
    )
    visible = repositories.add_active_thread(
        save_id=save.id,
        title="Keep the public lantern lit",
        description="The gallery depends on it.",
        priority=1,
    )
    for index in range(600):
        repositories.add_active_thread(
            save_id=save.id,
            title=f"Private absent obligation {index}",
            description="Only Lio may see this.",
            priority=100,
            visibility="private",
            related_entities=[f"character:{absent.id}"],
        )

    snapshot = load_narration_context_snapshot(
        repositories,
        save_id=save.id,
        raw_record_limit=1,
    )

    assert snapshot is not None
    assert [thread.id for thread in snapshot.active_threads] == [visible.id]
