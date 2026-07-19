from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.application.scene_presence import (
    build_scene_presence_model,
    character_image_eligible_message_ids,
)
from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_legacy_save_reference_is_not_character_image_eligible(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Oracle of Glass",
        premise="A private audience with the oracle.",
        player_role="Petitioner",
        content={"character_name": "Oracle of Glass"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Audience")
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The oracle waits beside the silver basin.",
    )
    character = repositories.add_character(
        save_id=save.id,
        name="Oracle of Glass",
        status="beside the basin",
    )
    reference = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=message.id,
        type="image",
        path=f"{save.id}/{message.id}/oracle-reference.png",
        thumbnail_path=None,
        prompt="Uploaded character reference image",
        provider="local",
        model="upload",
        status="succeeded",
        mime_type="image/png",
        metadata={"kind": "character_reference", "source": "uploaded"},
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="save",
        entity_id=save.id,
        target_type="media_asset",
        target_id=reference.id,
        relation="reference_image",
    )
    repositories.replace_message_scene_presence(
        save.id,
        message.id,
        [character.id],
        source="manual",
    )

    model = build_scene_presence_model(
        repositories,
        save_id=save.id,
        message_id=message.id,
    )

    assert character_image_eligible_message_ids(repositories, save_id=save.id) == (
        frozenset()
    )
    assert len(model.characters) == 1
    [row] = model.characters
    assert row.present is True
    assert row.has_reference_image is False
    assert row.reference_image is None
