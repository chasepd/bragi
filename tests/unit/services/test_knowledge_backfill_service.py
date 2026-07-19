from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.knowledge_backfill_service import KnowledgeBackfillService


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_knowledge_backfill_applies_legacy_links_and_visibility(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Archive Arrival",
        premise="An archive scene with uneven knowledge.",
        player_role="Avery",
        content={"starting_scene": "The archive door is open."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Boundary Test")
    tarin_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Tarin hears Avery make the archive-code joke while Nira is away.",
    )
    nira_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Nira",
        body="Nira arrives later and asks what she missed.",
    )
    scene_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Nira is now standing at the archive door.",
    )
    tarin = repositories.add_character(
        save_id=save.id,
        name="Tarin",
        aliases=["Tari"],
    )
    nira = repositories.add_character(
        save_id=save.id,
        name="Nira",
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="Tarin knows Avery made the archive-code joke within five minutes.",
        tags=["archive-code-joke"],
        source_message_id=tarin_message.id,
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=tarin.id,
        target_type="memory",
        target_id=memory.id,
        relation="knows",
        source_message_id=tarin_message.id,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Nira is now at the archive door.",
        present_character_ids=[nira.id],
        source_message_id=scene_message.id,
    )
    service = KnowledgeBackfillService(repositories)

    result = service.backfill_save(save.id)

    edges = repositories.list_character_knowledge_edges(save.id)
    visibility = repositories.list_message_visibility(save.id)
    assert result.character_knowledge_edges_applied == 1
    assert result.message_visibility_applied == 2
    assert [(edge.character_id, edge.target_id) for edge in edges] == [
        (tarin.id, memory.id)
    ]
    assert edges[0].source_message_ids == [tarin_message.id]
    assert {
        (row.message_id, row.character_id, row.source)
        for row in visibility
    } == {
        (nira_message.id, nira.id, "speaker_name"),
        (scene_message.id, nira.id, "scene_snapshot"),
    }
    assert repositories.list_context_update_suggestions(save.id) == []
