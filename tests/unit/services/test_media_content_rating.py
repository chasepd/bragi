from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.media_content_rating import (
    media_asset_content_rating,
    media_asset_exceeds_rating,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_media_rating_includes_source_message_and_all_reference_assets(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Harbor",
        premise="A harbor watch.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Watch")
    restricted_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="Restricted source",
        content_rating="r",
    )
    restricted_reference = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=restricted_message.id,
        type="image",
        path="restricted.png",
        prompt="Restricted reference",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={"content_rating": "g"},
    )
    derived = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=None,
        type="image",
        path="derived.png",
        prompt="Benign prompt",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={
            "content_rating": "g",
            "source_character_reference_asset_ids": [
                restricted_reference.id,
            ],
        },
    )
    assets = {asset.id: asset for asset in repositories.list_media_assets(save.id)}
    messages = {
        message.id: message for message in repositories.list_messages(save.id)
    }

    assert media_asset_content_rating(
        derived,
        media_assets_by_id=assets,
        source_messages=messages,
    ) == "r"
    assert media_asset_exceeds_rating(
        derived,
        allowed_rating="pg-13",
        media_assets_by_id=assets,
        source_messages=messages,
    )


def test_media_without_rating_provenance_is_unclassified(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Harbor",
        premise="A harbor watch.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Watch")
    asset = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=None,
        type="image",
        path="legacy.png",
        prompt="Legacy prompt",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )

    assert media_asset_content_rating(
        asset,
        media_assets_by_id={asset.id: asset},
        source_messages={},
    ) == "unclassified"
    assert media_asset_exceeds_rating(
        asset,
        allowed_rating="r",
        media_assets_by_id={asset.id: asset},
        source_messages={},
    )


def test_media_rating_includes_character_text_and_character_sources(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Harbor",
        premise="A harbor watch.",
        player_role="Warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Watch")
    character = repositories.add_character(
        save_id=save.id,
        name="Ilyra",
        content_rating="pg-13",
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=character.id,
        title=character.name,
    )
    text_message = repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=character.id,
        sender="character",
        body="Restricted source",
        content_rating="r",
    )
    derived = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=None,
        type="image",
        path="derived.png",
        prompt="Benign prompt",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={
            "content_rating": "g",
            "text_message_id": text_message.id,
            "character_id": character.id,
        },
    )

    assert media_asset_content_rating(
        derived,
        media_assets_by_id={derived.id: derived},
        source_messages={},
        character_text_messages={text_message.id: text_message},
        characters={character.id: character},
    ) == "r"
    assert media_asset_exceeds_rating(
        derived,
        allowed_rating="pg-13",
        media_assets_by_id={derived.id: derived},
        source_messages={},
        character_text_messages={text_message.id: text_message},
        characters={character.id: character},
    )
