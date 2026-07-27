from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.interaction_mode import InteractionMode
from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.chat_rendering import provider_chat_messages
from bragi.providers.contracts import ChatPromptPurpose, ChatRequest, ChatResponse
from bragi.providers.fake import FakeProviderClient
from bragi.providers.system_prompt import DEFAULT_PROSE_SAFETY_SECTION
from bragi.services.continuation_scenario_service import (
    MAX_CHARACTERS,
    MAX_LOCATIONS,
    MAX_MEMORIES,
    MAX_WORLD_STATE,
    ContinuationScenarioService,
    seed_continuation_characters,
)
from bragi.services.scenario_service import ScenarioService


class RecordingContinuationProvider(FakeProviderClient):
    def __init__(self) -> None:
        super().__init__()
        self.chat_requests: list[ChatRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        return ChatResponse(
            body="A safe continuation scenario field.",
            provider=request.provider,
            model_id=request.model_id,
        )


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_snapshot_includes_current_state_without_copying_transcript(
    repositories: PersistenceRepositories,
) -> None:
    save_id, mara_id = _create_continuation_save(repositories)

    snapshot = ContinuationScenarioService(
        repositories=repositories,
    ).build_snapshot(save_id)

    assert "Create a clean chapter/continuation scenario" in snapshot.seed
    assert "Scenario title: First Harbor" in snapshot.seed
    assert "Situation: Mara faces the tide court after the reveal." in snapshot.seed
    assert "Mara Voss" in snapshot.seed
    assert "voice: clipped, dry, careful with promises" in snapshot.seed
    assert "current clothing: Borrowed green raincoat over a linen shirt." in (
        snapshot.seed
    )
    assert "relationship.player_to_mara" in snapshot.seed
    assert "The tide-court reveal changed Mara's bargain." in snapshot.seed
    assert "Transcript-only secret wording" not in snapshot.seed
    assert snapshot.metadata["origin"] == "save_continuation"
    assert snapshot.metadata["source_save_id"] == save_id
    assert snapshot.metadata["source_message_count"] == 2
    continuity = snapshot.metadata["character_continuity"]
    assert isinstance(continuity, list)
    assert continuity[0]["name"] == "Mara Voss"
    assert continuity[0]["voice"] == "clipped, dry, careful with promises"
    assert continuity[0]["current_clothing"] == (
        "Borrowed green raincoat over a linen shirt."
    )
    assert continuity[0]["relationships"] == {"Ren": "owes him the bell-key"}
    assert continuity[0]["private_notes"] == "Knows the bell is a prison."
    scene = repositories.get_scene_snapshot(save_id)
    assert scene is not None
    assert mara_id in scene.present_character_ids


def test_snapshot_prioritizes_latest_state_over_original_scenario(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _mara_id = _create_continuation_save(repositories)

    snapshot = ContinuationScenarioService(
        repositories=repositories,
    ).build_snapshot(save_id)

    assert "Latest-state authority" in snapshot.seed
    assert "Do not regress to the original scenario starting state" in snapshot.seed
    assert "Original scenario baseline" in snapshot.seed
    assert "Outstanding unresolved threads and obligations" in snapshot.seed
    assert "related: Ren, Bell Court" in snapshot.seed


def test_snapshot_records_chapter_start_as_generation_prompt_without_raw_metadata(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _mara_id = _create_continuation_save(repositories)

    snapshot = ContinuationScenarioService(
        repositories=repositories,
    ).build_snapshot(
        save_id,
        chapter_start_instructions=(
            "Characters are going to bed; start the next chapter after sunrise."
        ),
    )

    assert "Chapter start instructions" in snapshot.seed
    assert (
        "Characters are going to bed; start the next chapter after sunrise."
        in snapshot.seed
    )
    assert snapshot.metadata["generation_prompt"] == (
        "Characters are going to bed; start the next chapter after sunrise."
    )
    assert "chapter_start_instructions" not in snapshot.metadata


def test_generate_draft_uses_shared_prose_safety_prompt(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _mara_id = _create_continuation_save(repositories)
    provider = RecordingContinuationProvider()
    scenario_service = ScenarioService(
        repositories=repositories,
        provider=provider,
        provider_name="fake",
        model_id="fake-chat",
    )

    asyncio.run(
        ContinuationScenarioService(
            repositories=repositories,
            scenario_service=scenario_service,
        ).generate_draft(save_id=save_id)
    )

    assert provider.chat_requests
    assert all(
        request.prompt_purpose is ChatPromptPurpose.SCENARIO_GENERATION
        for request in provider.chat_requests
    )
    assert all(
        DEFAULT_PROSE_SAFETY_SECTION
        in provider_chat_messages(request)[0]["content"]
        for request in provider.chat_requests
    )


def test_generate_draft_preserves_saved_storyteller_mode(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="The Ceremony",
        premise="A rival waits in the wings.",
        player_role="",
        content={"opening_message": "The orchestra falls silent."},
        interaction_mode=InteractionMode.STORYTELLER,
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Act One")
    repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The rival steps through the gallery doors.",
    )
    provider = RecordingContinuationProvider()
    service = ContinuationScenarioService(
        repositories=repositories,
        scenario_service=ScenarioService(
            repositories=repositories,
            provider=provider,
            provider_name="fake",
            model_id="fake-chat",
        ),
    )

    draft = asyncio.run(service.generate_draft(save_id=save.id))

    assert draft.interaction_mode is InteractionMode.STORYTELLER
    assert provider.chat_requests
    assert all(
        request.interaction_mode is InteractionMode.STORYTELLER
        for request in provider.chat_requests
    )


def test_generate_draft_rejects_retired_character_interaction_type(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="character_interaction",
        title="Retired audience",
        premise="Historical recovery content.",
        player_role="Petitioner",
        content={"character_name": "The Keeper"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Recovery save")
    provider = RecordingContinuationProvider()
    scenario_service = ScenarioService(
        repositories=repositories,
        provider=provider,
        provider_name="fake",
        model_id="fake-chat",
    )

    with pytest.raises(
        ValueError,
        match="no longer supported",
    ):
        asyncio.run(
            ContinuationScenarioService(
                repositories=repositories,
                scenario_service=scenario_service,
            ).generate_draft(save_id=save.id)
        )

    assert provider.chat_requests == []


def test_generate_draft_rejects_retired_character_interaction_hybrid(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Retired hybrid",
        premise="Historical recovery content.",
        player_role="Player",
        content={
            "_scenario_genres": ["dating_sim", "character_interaction"],
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Recovery save")
    provider = RecordingContinuationProvider()
    scenario_service = ScenarioService(
        repositories=repositories,
        provider=provider,
        provider_name="fake",
        model_id="fake-chat",
    )

    with pytest.raises(ValueError, match="no longer supported"):
        asyncio.run(
            ContinuationScenarioService(
                repositories=repositories,
                scenario_service=scenario_service,
            ).generate_draft(save_id=save.id)
        )

    assert provider.chat_requests == []


def test_snapshot_keeps_current_scene_records_when_caps_are_full(
    repositories: PersistenceRepositories,
) -> None:
    save_id, mara_id = _create_continuation_save(repositories)

    for index in range(MAX_LOCATIONS):
        repositories.add_location(
            save_id=save_id,
            name=f"Aardvark Archive {index:02d}",
            description="Old map room.",
        )
    for index in range(MAX_CHARACTERS):
        repositories.add_character(
            save_id=save_id,
            name=f"Aardvark Archivist {index:02d}",
            known_state="Cataloguing old business.",
        )

    snapshot = ContinuationScenarioService(
        repositories=repositories,
    ).build_snapshot(save_id)

    assert "Location: Bell Court" in snapshot.seed
    assert "Mara Voss" in snapshot.seed
    assert "Aardvark Archive 19" not in snapshot.seed
    assert "Aardvark Archivist 19" not in snapshot.seed
    scene = repositories.get_scene_snapshot(save_id)
    assert scene is not None
    assert mara_id in scene.present_character_ids


def test_snapshot_keeps_thread_related_world_state_when_caps_are_full(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _mara_id = _create_continuation_save(repositories)
    for index in range(MAX_WORLD_STATE):
        repositories.upsert_world_state(
            save_id=save_id,
            key=f"aaa.filler.{index:02d}",
            value={"note": "stale background"},
            category="background",
            confidence=0.2,
        )
    repositories.upsert_world_state(
        save_id=save_id,
        key="obligation.bell_key_debt",
        value={"owed_to": "Ren", "due": "before sunrise"},
        category="thread",
        confidence=0.99,
    )

    snapshot = ContinuationScenarioService(
        repositories=repositories,
    ).build_snapshot(save_id)

    assert "obligation.bell_key_debt" in snapshot.seed
    assert "aaa.filler.59" not in snapshot.seed


def test_snapshot_caps_memories_by_importance(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _mara_id = _create_continuation_save(repositories)
    for index in range(MAX_MEMORIES + 5):
        repositories.add_memory(
            save_id=save_id,
            body=f"memory-{index}",
            tags=["beat"],
            importance=index / 100,
        )

    snapshot = ContinuationScenarioService(
        repositories=repositories,
    ).build_snapshot(save_id)

    assert "memory-34" in snapshot.seed
    assert "memory-6" in snapshot.seed
    assert "memory-5" not in snapshot.seed


def test_seed_continuation_characters_creates_portable_character_records(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Chapter Two",
        premise="The bell bargain continues.",
        player_role="Harbor warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Chapter Two")
    opening = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The next chapter begins.",
    )

    created = seed_continuation_characters(
        repositories=repositories,
        save_id=save.id,
        source_message_id=opening.id,
        metadata={
            "character_continuity": [
                {
                    "name": "Mara Voss",
                    "aliases": ["Mara"],
                    "role": "Harbor warden",
                    "known_state": "Carrying the bell-key debt.",
                    "met": True,
                    "appearance": "Salt-stained blue coat.",
                    "visual_notes": "Keeps one glove buttoned.",
                    "current_clothing": "Borrowed green raincoat over a linen shirt.",
                    "personality": "Stubborn, tender under pressure.",
                    "voice": "clipped, dry, careful with promises",
                    "relationships": {"Ren": "owes him the bell-key"},
                    "status": "alive and negotiating",
                    "private_notes": "Knows the bell is a prison.",
                }
            ]
        },
    )

    assert created == 1
    characters = repositories.list_characters(save.id)
    assert len(characters) == 1
    assert characters[0].name == "Mara Voss"
    assert characters[0].aliases == ["Mara"]
    assert characters[0].voice == "clipped, dry, careful with promises"
    assert characters[0].current_clothing == (
        "Borrowed green raincoat over a linen shirt."
    )
    assert characters[0].relationships == {"Ren": "owes him the bell-key"}
    assert characters[0].private_notes == "Knows the bell is a prison."
    assert characters[0].protected_from_maintenance is True
    assert characters[0].source_message_id == opening.id


def _create_continuation_save(
    repositories: PersistenceRepositories,
) -> tuple[str, str]:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="First Harbor",
        premise="A drowned harbor rings at low tide.",
        player_role="Harbor warden",
        content={
            "player_character_name": "Mara Voss",
            "tone_genre": "Maritime mystery.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="First Harbor")
    repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="Transcript-only secret wording should not leak.",
    )
    narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Transcript-only secret wording should not leak either.",
    )
    location = repositories.add_location(
        save_id=save.id,
        name="Bell Court",
        description="A stone court below the drowned quay.",
        status="exposed by low tide",
    )
    character = repositories.add_character(
        save_id=save.id,
        name="Mara Voss",
        aliases=["Mara"],
        role="Harbor warden",
        known_state="Revealed the bell bargain and survived.",
        met=True,
        appearance="Salt-stained blue coat.",
        visual_notes="Keeps one glove buttoned.",
        current_clothing="Borrowed green raincoat over a linen shirt.",
        personality="Stubborn, tender under pressure.",
        voice="clipped, dry, careful with promises",
        relationships={"Ren": "owes him the bell-key"},
        status="alive and negotiating",
        location_id=location.id,
        private_notes="Knows the bell is a prison.",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
        situation="Mara faces the tide court after the reveal.",
        objective="Negotiate chapter two without freeing the bell.",
        present_character_ids=[character.id],
        source_message_id=narrator.id,
    )
    repositories.add_active_thread(
        save_id=save.id,
        title="Bell-key debt",
        description="Ren expects payment before sunrise.",
        priority=9,
        related_entities=["Ren", "Bell Court"],
    )
    repositories.upsert_world_state(
        save_id=save.id,
        key="relationship.player_to_mara",
        value={"status": "fragile trust"},
        category="relationship",
    )
    repositories.add_memory(
        save_id=save.id,
        body="The tide-court reveal changed Mara's bargain.",
        tags=["reveal"],
        importance=0.95,
    )
    repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id="beat-1",
        title="Promise to Ren",
        body="Mara promised Ren the next truthful bell toll.",
        metadata={"fact_type": "story_beat", "importance": 0.9},
    )
    repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=narrator.id,
        covers_message_end_id=narrator.id,
        body="Mara exposed the bell court and learned the bell is a prison.",
        provider="fake",
        model="summary",
    )
    return save.id, character.id
