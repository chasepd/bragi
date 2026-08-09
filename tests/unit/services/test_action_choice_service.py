from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from bragi.interaction_mode import InteractionMode
from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.fake import FakeProviderClient
from bragi.safety import CONTENT_FILTER_TRANSITION
from bragi.services.action_choice_service import (
    ACTION_CHOICE_GENERATION_TASK,
    ActionChoiceService,
)
from bragi.services.character_action_planning_service import (
    CHARACTER_ACTION_PLANNING_TASK,
)
from bragi.services.content_safety_service import (
    ContentSafetyAction,
    ContentSafetyResult,
    ContentSafetyService,
)


class BlockingContentSafetyService:
    def __init__(self) -> None:
        self.fade_settings: list[bool] = []

    async def review_narration(self, **kwargs: object) -> ContentSafetyResult:
        self.fade_settings.append(bool(kwargs["fade_to_black_enabled"]))
        return ContentSafetyResult(
            body=CONTENT_FILTER_TRANSITION,
            action=ContentSafetyAction.BLOCK,
            minimum_rating="r",
            transition_applied=True,
            agent_ran=True,
        )


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        repositories = PersistenceRepositories(connection)
        repositories.set_app_setting("content_filter_rating", "unrated")
        yield repositories


def test_action_choice_service_requests_structured_four_choice_schema_and_persists(
    repositories: PersistenceRepositories,
) -> None:
    save_id, narrator_id = _create_cyoa_save(repositories)
    _save_model(
        repositories,
        model_id="fake-chat",
        capabilities=["structured_output"],
    )
    provider = FakeProviderClient(
        structured_output={
            "choices": [
                {"body": "Open the brass atlas."},
                {"body": "Question the librarian."},
                {"body": "Hide the index under your coat."},
                {"body": "Step through the blue shelf-door."},
            ]
        }
    )
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="fake",
        model_id="fake-chat",
    )

    records = asyncio.run(
        ActionChoiceService(
            repositories=repositories,
            providers={"fake": provider},
        ).generate_for_message(save_id=save_id, narrator_message_id=narrator_id)
    )

    assert [record.ordinal for record in records] == [1, 2, 3, 4]
    assert [record.body for record in records] == [
        "Open the brass atlas.",
        "Question the librarian.",
        "Hide the index under your coat.",
        "Step through the blue shelf-door.",
    ]
    assert {record.provider for record in records} == {"fake"}
    assert {record.model for record in records} == {"fake-chat"}
    assert [
        record.body
        for record in repositories.latest_message_action_choices(save_id)
    ] == [
        "Open the brass atlas.",
        "Question the librarian.",
        "Hide the index under your coat.",
        "Step through the blue shelf-door.",
    ]
    request = provider.structured_output_requests[0]
    assert request.schema_name == "action_choices"
    choices_schema = request.schema["properties"]["choices"]
    assert choices_schema["minItems"] == 4
    assert choices_schema["maxItems"] == 4
    assert "Latest narrator message:" in request.messages[-1].body
    assert "Player character:" in request.messages[-1].body
    assert "Ily" in request.messages[-1].body
    assert "Current scene:" in request.messages[-1].body
    assert "Archivist Ren" in request.messages[-1].body
    assert "Ren intends to steal the atlas" not in request.messages[-1].body
    assert "numbering" in request.messages[0].body


def test_action_choice_service_forwards_retry_progress_callback(
    repositories: PersistenceRepositories,
) -> None:
    save_id, narrator_id = _create_cyoa_save(repositories)
    _save_model(
        repositories,
        model_id="fake-chat",
        capabilities=["structured_output"],
    )
    provider = FakeProviderClient(
        structured_output={
            "choices": [
                {"body": "Follow the lantern trail."},
                {"body": "Question the bridge keeper."},
                {"body": "Wait for the fog to lift."},
                {"body": "Search the abandoned cart."},
            ]
        }
    )
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="fake",
        model_id="fake-chat",
    )

    def retry_progress(_progress: object) -> None:
        return

    asyncio.run(
        ActionChoiceService(
            repositories=repositories,
            providers={"fake": provider},
        ).generate_for_message(
            save_id=save_id,
            narrator_message_id=narrator_id,
            retry_progress_callback=retry_progress,
        )
    )

    assert (
        provider.structured_output_requests[0].retry_progress_callback
        is retry_progress
    )


def test_action_choice_service_skips_storyteller_saves(
    repositories: PersistenceRepositories,
) -> None:
    save_id, narrator_id = _create_cyoa_save(
        repositories,
        interaction_mode=InteractionMode.STORYTELLER,
    )
    provider = FakeProviderClient()

    records = asyncio.run(
        ActionChoiceService(
            repositories=repositories,
            providers={"fake": provider},
        ).generate_for_message(save_id=save_id, narrator_message_id=narrator_id)
    )

    assert records == []
    assert provider.structured_output_requests == []


def test_action_choice_service_uses_dedicated_model_preference(
    repositories: PersistenceRepositories,
) -> None:
    save_id, narrator_id = _create_cyoa_save(repositories)
    _save_model(
        repositories,
        model_id="fake-chat",
        capabilities=["structured_output"],
    )
    provider = FakeProviderClient(
        structured_output={
            "choices": [
                {"body": "Open the brass atlas."},
                {"body": "Question the librarian."},
                {"body": "Hide the index under your coat."},
                {"body": "Step through the blue shelf-door."},
            ]
        }
    )
    repositories.set_model_preference(
        task=ACTION_CHOICE_GENERATION_TASK,
        provider="fake",
        model_id="fake-chat",
    )

    records = asyncio.run(
        ActionChoiceService(
            repositories=repositories,
            providers={"fake": provider},
        ).generate_for_message(save_id=save_id, narrator_message_id=narrator_id)
    )

    assert [record.body for record in records] == [
        "Open the brass atlas.",
        "Question the librarian.",
        "Hide the index under your coat.",
        "Step through the blue shelf-door.",
    ]
    assert provider.structured_output_requests[0].model_id == "fake-chat"


def test_action_choice_service_safety_reviews_and_rates_generated_choices(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_app_setting("content_filter_rating", "pg")
    save_id, narrator_id = _create_cyoa_save(repositories)
    _save_model(
        repositories,
        model_id="fake-chat",
        capabilities=["structured_output"],
    )
    choice_provider = FakeProviderClient(
        structured_output={
            "choices": [
                {"body": "Choice one."},
                {"body": "Choice two."},
                {"body": "Choice three."},
                {"body": "Choice four."},
            ]
        }
    )
    safety = BlockingContentSafetyService()
    repositories.set_model_preference(
        task=ACTION_CHOICE_GENERATION_TASK,
        provider="fake",
        model_id="fake-chat",
    )

    records = asyncio.run(
        ActionChoiceService(
            repositories=repositories,
            providers={"fake": choice_provider},
            content_safety_service=cast(ContentSafetyService, safety),
        ).generate_for_message(save_id=save_id, narrator_message_id=narrator_id)
    )

    assert [record.body for record in records] == [CONTENT_FILTER_TRANSITION] * 4
    assert [record.content_rating for record in records] == ["g"] * 4
    assert safety.fade_settings == [False] * 4


def test_action_choice_service_uses_roleplay_specific_action_choice_model_preference(
    repositories: PersistenceRepositories,
) -> None:
    save_id, narrator_id = _create_cyoa_save(repositories)
    _save_model(
        repositories,
        model_id="fake-chat",
        capabilities=["structured_output"],
    )
    provider = FakeProviderClient(
        structured_output={
            "choices": [
                {"body": "Open the brass atlas."},
                {"body": "Question the librarian."},
                {"body": "Hide the index under your coat."},
                {"body": "Step through the blue shelf-door."},
            ]
        }
    )
    repositories.set_app_setting("use_shared_roleplay_models", False)
    repositories.set_model_preference(
        task=f"full_roleplay_{ACTION_CHOICE_GENERATION_TASK}",
        provider="fake",
        model_id="fake-chat",
    )

    records = asyncio.run(
        ActionChoiceService(
            repositories=repositories,
            providers={"fake": provider},
        ).generate_for_message(save_id=save_id, narrator_message_id=narrator_id)
    )

    assert [record.body for record in records] == [
        "Open the brass atlas.",
        "Question the librarian.",
        "Hide the index under your coat.",
        "Step through the blue shelf-door.",
    ]
    assert provider.structured_output_requests[0].model_id == "fake-chat"


def test_action_choice_service_falls_back_to_legacy_character_agent_preference(
    repositories: PersistenceRepositories,
) -> None:
    save_id, narrator_id = _create_cyoa_save(repositories)
    _save_model(
        repositories,
        model_id="fake-chat",
        capabilities=["structured_output"],
    )
    provider = FakeProviderClient(
        structured_output={
            "choices": [
                {"body": "Open the brass atlas."},
                {"body": "Question the librarian."},
                {"body": "Hide the index under your coat."},
                {"body": "Step through the blue shelf-door."},
            ]
        }
    )
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="fake",
        model_id="fake-chat",
    )

    records = asyncio.run(
        ActionChoiceService(
            repositories=repositories,
            providers={"fake": provider},
        ).generate_for_message(save_id=save_id, narrator_message_id=narrator_id)
    )

    assert [record.body for record in records] == [
        "Open the brass atlas.",
        "Question the librarian.",
        "Hide the index under your coat.",
        "Step through the blue shelf-door.",
    ]
    assert provider.structured_output_requests[0].schema_name == "action_choices"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "choices": [
                    {"body": "Open the brass atlas."},
                    {"body": "Open the brass atlas."},
                    {"body": "Question the librarian."},
                    {"body": "Step through the blue shelf-door."},
                ]
            },
            "unique",
        ),
        (
            {
                "choices": [
                    {"body": "Open the brass atlas."},
                    {"body": ""},
                    {"body": "Question the librarian."},
                    {"body": "Step through the blue shelf-door."},
                ]
            },
            "non-empty",
        ),
        (
            {
                "choices": [
                    {"body": "Open the brass atlas."},
                    {"body": "Question the librarian."},
                    {"body": "Step through the blue shelf-door."},
                ]
            },
            "exactly four",
        ),
    ],
)
def test_action_choice_service_rejects_invalid_structured_choices(
    repositories: PersistenceRepositories,
    payload: dict[str, object],
    message: str,
) -> None:
    save_id, narrator_id = _create_cyoa_save(repositories)
    _save_model(
        repositories,
        model_id="fake-chat",
        capabilities=["structured_output"],
    )
    provider = FakeProviderClient(structured_output=payload)
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="fake",
        model_id="fake-chat",
    )

    with pytest.raises(ValueError, match=message):
        asyncio.run(
            ActionChoiceService(
                repositories=repositories,
                providers={"fake": provider},
            ).generate_for_message(save_id=save_id, narrator_message_id=narrator_id)
        )

    assert repositories.list_message_action_choices(save_id) == []


def test_action_choice_service_rejects_missing_catalog_row(
    repositories: PersistenceRepositories,
) -> None:
    save_id, narrator_id = _create_cyoa_save(repositories)
    provider = FakeProviderClient(
        structured_output={
            "choices": [
                {"body": "Open the brass atlas."},
                {"body": "Question the librarian."},
                {"body": "Hide the index under your coat."},
                {"body": "Step through the blue shelf-door."},
            ]
        }
    )
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="fake",
        model_id="unsynced-action-choice",
    )

    with pytest.raises(ValueError, match="not in the provider model catalog"):
        asyncio.run(
            ActionChoiceService(
                repositories=repositories,
                providers={"fake": provider},
            ).generate_for_message(save_id=save_id, narrator_message_id=narrator_id)
        )

    assert provider.structured_output_requests == []
    assert repositories.list_message_action_choices(save_id) == []


def test_action_choice_service_rejects_known_model_without_structured_output(
    repositories: PersistenceRepositories,
) -> None:
    save_id, narrator_id = _create_cyoa_save(repositories)
    _save_model(
        repositories,
        model_id="fake-chat-only",
        capabilities=["chat"],
    )
    provider = FakeProviderClient(
        structured_output={
            "choices": [
                {"body": "Open the brass atlas."},
                {"body": "Question the librarian."},
                {"body": "Hide the index under your coat."},
                {"body": "Step through the blue shelf-door."},
            ]
        }
    )
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="fake",
        model_id="fake-chat-only",
    )

    with pytest.raises(ValueError, match="does not advertise structured output"):
        asyncio.run(
            ActionChoiceService(
                repositories=repositories,
                providers={"fake": provider},
            ).generate_for_message(save_id=save_id, narrator_message_id=narrator_id)
        )

    assert provider.structured_output_requests == []
    assert repositories.list_message_action_choices(save_id) == []


def test_action_choice_service_rechecks_prepared_model_capability(
    repositories: PersistenceRepositories,
) -> None:
    save_id, narrator_id = _create_cyoa_save(repositories)
    _save_model(
        repositories,
        model_id="fake-chat",
        capabilities=["structured_output"],
    )
    provider = FakeProviderClient(
        structured_output={
            "choices": [
                {"body": "Open the brass atlas."},
                {"body": "Question the librarian."},
                {"body": "Hide the index under your coat."},
                {"body": "Step through the blue shelf-door."},
            ]
        }
    )
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="fake",
        model_id="fake-chat",
    )
    service = ActionChoiceService(
        repositories=repositories,
        providers={"fake": provider},
    )
    prepared = service.prepare_for_message(
        save_id=save_id,
        narrator_message_id=narrator_id,
    )
    assert prepared is not None
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-chat",
        display_name="Fake Chat",
        capabilities=["chat"],
    )

    with pytest.raises(ValueError, match="does not advertise structured output"):
        asyncio.run(service.generate_prepared(prepared))

    assert provider.structured_output_requests == []
    assert repositories.list_message_action_choices(save_id) == []


def test_action_choice_service_discards_prepared_choices_after_head_advances(
    repositories: PersistenceRepositories,
) -> None:
    save_id, narrator_id = _create_cyoa_save(repositories)
    _save_model(
        repositories,
        model_id="fake-chat",
        capabilities=["structured_output"],
    )
    provider = FakeProviderClient(
        structured_output={
            "choices": [
                {"body": "Open the brass atlas."},
                {"body": "Question the librarian."},
                {"body": "Hide the index under your coat."},
                {"body": "Step through the blue shelf-door."},
            ]
        }
    )
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="fake",
        model_id="fake-chat",
    )
    service = ActionChoiceService(
        repositories=repositories,
        providers={"fake": provider},
    )
    prepared = service.prepare_for_message(
        save_id=save_id,
        narrator_message_id=narrator_id,
    )
    assert prepared is not None
    repositories.append_message(
        save_id=save_id,
        role="player",
        speaker_name="Ily",
        body="I choose my own path.",
    )

    records = asyncio.run(service.generate_prepared(prepared))

    assert records == []
    assert repositories.list_message_action_choices(save_id) == []


def test_action_choice_service_skips_saves_without_action_choices_enabled(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A tower in fog.",
        player_role="Keeper",
        content={"opening_message": "The lamp hisses."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The lamp hisses.",
    )
    _save_model(
        repositories,
        model_id="fake-chat",
        capabilities=["structured_output"],
    )
    provider = FakeProviderClient(
        structured_output={
            "choices": [
                {"body": "A"},
                {"body": "B"},
                {"body": "C"},
                {"body": "D"},
            ]
        }
    )
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="fake",
        model_id="fake-chat",
    )

    records = asyncio.run(
        ActionChoiceService(
            repositories=repositories,
            providers={"fake": provider},
        ).generate_for_message(save_id=save.id, narrator_message_id=narrator.id)
    )

    assert records == []
    assert provider.structured_output_requests == []


def _create_cyoa_save(
    repositories: PersistenceRepositories,
    *,
    interaction_mode: InteractionMode = InteractionMode.ROLEPLAY,
) -> tuple[str, str]:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Library of Falling Doors",
        premise="Every shelf is a door.",
        player_role="Courier",
        interaction_mode=interaction_mode,
        content={
            "action_choices_enabled": True,
            "title": "Library of Falling Doors",
            "premise": "Every shelf is a door.",
            "player_character_name": "Ily",
            "player_role": "Courier",
            "tone_genre": "Bookish fantasy suspense.",
            "choice_style": "Four concrete choices with different risk profiles.",
            "opening_message": "The blue shelf opens.",
        },
    )
    save = repositories.create_save(
        scenario_id=scenario.id,
        title="Library of Falling Doors",
    )
    player = repositories.add_character(
        save_id=save.id,
        name="Ily",
        role="Courier",
        known_state="A courier carrying a brass atlas.",
        personality="Careful, curious, and quick to improvise.",
        goals="Deliver the atlas before the doors rearrange.",
        met=True,
        is_player_character=True,
    )
    ren = repositories.add_character(
        save_id=save.id,
        name="Archivist Ren",
        role="Librarian",
        known_state="Catalogs shelf-doors and watches visitors closely.",
        private_notes="Ren intends to steal the atlas if Ily looks away.",
        met=True,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Ily stands before the open blue shelf-door.",
        present_character_ids=[player.id, ren.id],
    )
    narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The blue shelf opens.",
    )
    return save.id, narrator.id


def _save_model(
    repositories: PersistenceRepositories,
    *,
    model_id: str,
    capabilities: list[str],
) -> None:
    repositories.save_provider_model(
        provider="fake",
        model_id=model_id,
        display_name=model_id.replace("-", " ").title(),
        capabilities=capabilities,
    )
