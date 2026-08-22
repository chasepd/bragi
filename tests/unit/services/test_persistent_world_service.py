from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.fake import FakeProviderClient
from bragi.services.persistent_world_service import (
    PERSISTENT_WORLD_CONTENT_PREFIX,
    PersistentWorldService,
)
from bragi.services.save_service import SaveService


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_new_save_materializes_world_and_later_world_edits_do_not_change_it(
    repositories: PersistenceRepositories,
) -> None:
    world = repositories.create_persistent_world(
        title="The Salt Marches",
        description="A lowland setting shaped by salt and old rivers.",
        sections={"cultures": "The river clans trade by oath."},
    )
    scenario = repositories.create_scenario(
        type="fantasy_roleplay",
        title="A Lantern at Low Water",
        premise="A courier arrives before the river dries.",
        player_role="River courier",
        content={"opening_message": "The lantern burns blue."},
        persistent_world_id=world.id,
    )
    service = SaveService(repositories)

    first_save = service.create_save(scenario_id=scenario.id, title="First crossing")
    first_details = repositories.load_save_details(first_save.id)
    assert first_details is not None
    first_content = json.loads(
        first_details.scenario.content_json
    )

    repositories.update_persistent_world(
        world_id=world.id,
        title=world.title,
        description=world.description,
        sections={"cultures": "The river clans now answer to a crowned admiral."},
    )
    second_save = service.create_save(scenario_id=scenario.id, title="Second crossing")
    second_details = repositories.load_save_details(second_save.id)
    assert second_details is not None
    second_content = json.loads(
        second_details.scenario.content_json
    )

    assert first_content[f"{PERSISTENT_WORLD_CONTENT_PREFIX}cultures"] == (
        "The river clans trade by oath."
    )
    assert second_content[f"{PERSISTENT_WORLD_CONTENT_PREFIX}cultures"] == (
        "The river clans now answer to a crowned admiral."
    )
    base_scenario = repositories.get_scenario(scenario.id)
    assert base_scenario is not None
    assert json.loads(base_scenario.content_json) == {
        "opening_message": "The lantern burns blue."
    }


def test_world_service_draft_uses_plain_provider_prose(
    repositories: PersistenceRepositories,
) -> None:
    service = PersistentWorldService(repositories)
    provider = FakeProviderClient()

    draft = asyncio.run(
        service.generate_draft(
            seed="A desert confederacy built around singing wells.",
            title="The Singing Wells",
            description="",
            provider=provider,
            provider_name="fake",
            model_id="fake-chat",
            section_ids=("overview",),
        )
    )

    assert draft.sections["overview"].startswith("echo: ")
    assert "{" not in draft.sections["overview"]
    assert draft.source_metadata["origin"] == "ai_draft"
