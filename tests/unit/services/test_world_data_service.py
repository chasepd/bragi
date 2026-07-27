from __future__ import annotations

import asyncio
import builtins
import importlib
import json
import sqlite3
import sys
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from pytest import MonkeyPatch

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatRequest,
    ChatResponse,
    ImageRequest,
    ImageResponse,
    ProviderCapability,
    ProviderConfigStatus,
    ProviderModel,
    StructuredOutputRequest,
    StructuredOutputResponse,
)
from bragi.services.context_search_service import (
    RECENT_MESSAGE_CANDIDATE_LIMIT,
    ContextSearchService,
)
from bragi.services.sexual_content_safety import CONTENT_FILTER_TRANSITION
from bragi.services.world_data_service import (
    WorldDataAuditRow,
    WorldDataCharacterRow,
    WorldDataEntityLinkRow,
    WorldDataLocationRow,
    WorldDataMemoryRow,
    WorldDataStateRow,
    WorldDataSuggestionRow,
    WorldDataSummaryRow,
    WorldDataThreadRow,
)

_MISSING = object()


class RecordingWorldDataContextProvider:
    provider_name = "fake"

    def __init__(self) -> None:
        self.chat_requests: list[ChatRequest] = []
        self.structured_output_requests: list[StructuredOutputRequest] = []

    async def validate_config(self) -> ProviderConfigStatus:
        return ProviderConfigStatus(
            provider=self.provider_name,
            configured=True,
            authenticated=True,
        )

    async def list_models(self) -> list[ProviderModel]:
        return [
            ProviderModel(
                provider=self.provider_name,
                model_id="fake-context",
                display_name="Fake Context",
                capabilities=frozenset({ProviderCapability.STRUCTURED_OUTPUT}),
            )
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        raise AssertionError("world-data context test must use structured output")

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        if request.schema_name == "context_retrieval_expansion":
            return StructuredOutputResponse(
                data={"terms": [], "phrases": [], "entity_ids": []},
                provider=request.provider,
                model_id=request.model_id,
            )
        self.structured_output_requests.append(request)
        prompt = "\n".join(message.body for message in request.messages)
        assert "The archived bell memory should not enter narrator context." not in (
            prompt
        )
        assert "The archived bell summary should not enter narrator context." not in (
            prompt
        )
        assert "clue.bell" not in prompt
        assert "scene.location: name: Beacon tower" in prompt
        assert "Captain Ilyra abandoned the east stair." in prompt
        assert "watch moved to the beacon tower" not in prompt
        return StructuredOutputResponse(
            data={
                "selections": _world_data_context_selections(prompt),
            },
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 9},
        )

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        raise AssertionError("world-data context test must not generate images")


def _world_data_context_selections(prompt: str) -> list[dict[str, str]]:
    selections: list[dict[str, str]] = []
    for line in prompt.splitlines():
        if "scene.location: name: Beacon tower" in line:
            selections.append(
                {
                    "source_type": "world_state",
                    "source_id": _candidate_id(line, "world_state"),
                    "relevance_note": "The edited location matters.",
                }
            )
        elif "Captain Ilyra abandoned the east stair." in line:
            selections.append(
                {
                    "source_type": "memory",
                    "source_id": _candidate_id(line, "memory"),
                    "relevance_note": "The edited promise matters.",
                }
            )
    return selections


def _candidate_id(line: str, source_type: str) -> str:
    marker = f"[{source_type}:"
    if marker not in line:
        raise AssertionError(f"missing {source_type} candidate in line: {line}")
    return line.split(marker, maxsplit=1)[1].split("]", maxsplit=1)[0]


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_world_data_service_model_is_import_safe_and_exposes_active_save_data(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_world_data_fixture(repositories)
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )

    model = service.build_model()

    assert _value(model, "active_save_id", "save_id") == save_id
    assert _value(model, "save_title") == "Night Watch"
    scenario = _value(model, "scenario")
    assert _value(scenario, "title") == "Ashfall Keep"
    assert _value(scenario, "premise") == "A border keep is cut off by ash storms."
    assert _value(scenario, "player_character_name") == "Mara Voss"
    assert _value(scenario, "player_role") == "Signal warden"
    assert _value(scenario, "content_sections") == ()

    world_state = _items_by_id(_value(model, "world_state", "state_rows"))
    assert list(world_state) == [ids["state"]]
    assert _value(world_state[ids["state"]], "key") == "scene.location"
    assert _value(world_state[ids["state"]], "value_json") == (
        '{"name":"Gatehouse","threat":"ash storm"}'
    )

    memories = _items_by_id(_value(model, "memories", "memory_rows"))
    assert list(memories) == [ids["memory"]]
    assert _value(memories[ids["memory"]], "body") == (
        "Captain Ilyra promised to hold the east stair."
    )

    summaries = _items_by_id(_value(model, "summaries", "summary_rows"))
    assert list(summaries) == [ids["summary"]]
    assert _value(summaries[ids["summary"]], "body") == (
        "The watch began as the tower beacon started failing."
    )
    assert _value(summaries[ids["summary"]], "source_message_ids") == (
        ids["message"],
    )
    assert _value(summaries[ids["summary"]], "source_summary_ids") == ()
    assert _error_text(model) == ""


def test_world_data_service_exposes_character_starters_as_scenario_field(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "rumor_board": "The east stair has fresh ash marks.",
            "character_starters": [
                {
                    "name": "Captain Ilyra",
                    "aliases": ["Ilyra"],
                    "role": "Watch captain",
                    "known_state": "She keeps the east stair.",
                    "appearance": "Bronze cloak clasp and salt-stained boots.",
                    "visual_notes": "Straight silhouette in lighthouse glare.",
                    "personality": "Decisive and guarded.",
                    "voice": "Low clipped orders.",
                    "relationships": {"Mara": "wary ally"},
                    "status": "waiting at the beacon",
                    "met": True,
                    "locked_fields": ["appearance"],
                }
            ],
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save.id,
    )

    model = service.build_model()

    scenario_model = _value(model, "scenario")
    assert _value(scenario_model, "content_sections") == (
        ("rumor_board", "The east stair has fresh ash marks."),
    )
    starters = _value(scenario_model, "character_starters")
    assert [starter.name for starter in starters] == ["Captain Ilyra"]
    assert starters[0].aliases == ("Ilyra",)
    assert starters[0].relationships == {"Mara": "wary ally"}


def test_world_data_service_exposes_generation_prompt_without_editable_source(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "rumor_board": "The east stair has fresh ash marks.",
            "_source": {
                "origin": "ai_draft",
                "generation_prompt": "A keep where ash storms answer bells.",
            },
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save.id,
    )

    model = service.build_model()

    scenario_model = _value(model, "scenario")
    assert _value(scenario_model, "generation_prompt") == (
        "A keep where ash storms answer bells."
    )
    assert _value(scenario_model, "content_sections") == (
        ("rumor_board", "The east stair has fresh ash marks."),
    )


def test_world_data_service_requires_an_active_save_and_returns_empty_state(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    _persist_world_data_fixture(repositories)
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=None,
    )

    model = service.build_model()

    assert _value(model, "active_save_id", "save_id") is None
    assert _value(model, "scenario", default=None) is None
    assert tuple(_value(model, "world_state", "state_rows")) == ()
    assert tuple(_value(model, "memories", "memory_rows")) == ()
    assert tuple(_value(model, "summaries", "summary_rows")) == ()
    assert "No save loaded" in _error_text(model)


def test_world_data_service_exposes_character_knowledge_graph_rows(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_normalized_world_data_fixture(repositories)
    edge = repositories.add_character_knowledge_edge(
        save_id=save_id,
        character_id=ids["character"],
        target_type="memory",
        target_id=ids["memory"],
        knowledge_state="knows",
        acquisition_method="witnessed",
        confidence=1.0,
        source_message_id=ids["message"],
        source_message_ids=[ids["message"]],
        evidence_quote="Ilyra heard the promise.",
        edge_id="edge-ilyra-promise",
    )
    visibility = repositories.add_message_visibility(
        save_id=save_id,
        message_id=ids["message"],
        character_id=ids["character"],
        visibility="visible",
        confidence=0.9,
        source="scene_snapshot",
        evidence="Ilyra was present.",
        visibility_id="visibility-ilyra-message",
    )
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )

    model = service.build_model()

    knowledge_rows = _items_by_id(_value(model, "knowledge_edges"))
    visibility_rows = _items_by_id(_value(model, "message_visibility"))
    assert list(knowledge_rows) == [edge.id]
    assert _value(knowledge_rows[edge.id], "character_id") == ids["character"]
    assert _value(knowledge_rows[edge.id], "target_id") == ids["memory"]
    assert _value(knowledge_rows[edge.id], "source_message_ids_text") == (
        ids["message"]
    )
    assert _value(knowledge_rows[edge.id], "evidence_quote") == (
        "Ilyra heard the promise."
    )
    assert list(visibility_rows) == [visibility.id]
    assert _value(visibility_rows[visibility.id], "message_id") == ids["message"]
    assert _value(visibility_rows[visibility.id], "source") == "scene_snapshot"


def test_world_data_service_hides_deprecated_loss_condition_rows(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "title": "Ashfall Keep",
            "premise": "A border keep is cut off by ash storms.",
            "player_role": "Signal warden",
            "starting_scene": "The beacon gutters in the tower.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    source_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens begins to crack.",
        provider="fake",
        model="fake-chat",
    )
    condition = repositories.add_loss_condition(
        condition_id="loss-beacon-collapse",
        save_id=save.id,
        name="Beacon collapse",
        description="The beacon lens is cracked but stable.",
        status="active",
        source="structured",
        source_message_id=source_message.id,
    )
    change = repositories.add_loss_condition_change(
        save_id=save.id,
        condition_id=condition.id,
        operation="add",
        before=None,
        after={
            "id": condition.id,
            "name": condition.name,
            "description": condition.description,
            "status": condition.status,
            "source": condition.source,
        },
        reason="The turn established the risk.",
        provider="fake",
        model="fake-loss-model",
        source_message_id=source_message.id,
    )
    outcome = repositories.create_loss_outcome(
        save_id=save.id,
        condition_id=condition.id,
        condition_name=condition.name,
        triggering_message_id=source_message.id,
        explanation="The beacon falls and the watch is lost.",
        confidence=0.93,
        evidence={
            "items": [
                {
                    "source_message_id": source_message.id,
                    "quote": "beacon lens begins to crack",
                }
            ]
        },
        provider="fake",
        model="fake-loss-model",
    )
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save.id,
    )
    model = service.build_model()

    assert model.loss_conditions == ()
    assert model.active_loss_outcome is None
    with pytest.raises(ValueError, match="deprecated"):
        service.apply_edits(
            world_data.WorldDataEdits(
                scenario=_scenario_edit_from_model(world_data, model),
                loss_conditions=(
                    world_data.WorldDataLossConditionRow(
                        condition_id=condition.id,
                        name=condition.name,
                        description=condition.description,
                        status=condition.status,
                        source=condition.source,
                        archived=True,
                    ),
                ),
            )
        )

    assert repositories.list_loss_conditions(save.id) == [condition]
    assert repositories.list_loss_condition_changes(save.id) == [change]
    assert repositories.get_active_loss_outcome(save.id) == outcome
    result = service.apply_edits(
        world_data.WorldDataEdits(
            scenario=_scenario_edit_from_model(world_data, model),
        )
    )
    assert result.model.loss_conditions == ()
    assert result.model.active_loss_outcome is None
    assert repositories.list_loss_conditions(save.id) == [condition]
    assert repositories.list_loss_condition_changes(save.id) == [change]
    assert repositories.get_active_loss_outcome(save.id) == outcome


def test_world_data_service_model_exposes_normalized_context_surfaces(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_normalized_world_data_fixture(repositories)
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )

    model = service.build_model()

    assert model.scene is not None
    assert model.scene.current_location_id == ids["location"]
    assert model.scene.present_character_ids_text == ids["character"]
    locations = cast(dict[str, WorldDataLocationRow], _items_by_id(model.locations))
    characters = cast(dict[str, WorldDataCharacterRow], _items_by_id(model.characters))
    threads = cast(dict[str, WorldDataThreadRow], _items_by_id(model.threads))
    links = cast(dict[str, WorldDataEntityLinkRow], _items_by_id(model.links))
    suggestions = cast(
        dict[str, WorldDataSuggestionRow],
        _items_by_id(model.suggestions),
    )
    audit = cast(dict[str, WorldDataAuditRow], _items_by_id(model.audit))
    assert locations[ids["location"]].name == "Beacon Gallery"
    assert characters[ids["character"]].name == "Captain Ilyra"
    assert characters[ids["character"]].goals == "Keep the beacon lit."
    assert characters[ids["character"]].texting_style == (
        "Crisp one-line replies, no emoji."
    )
    assert characters[ids["character"]].cooperation_conditions == (
        "Helps after proof the lens can hold."
    )
    assert threads[ids["thread"]].title == "Repair the beacon"
    assert links[ids["link"]].target_id == ids["memory"]
    assert suggestions[ids["suggestion"]].field_path == ("description")
    assert audit[ids["audit"]].operation == ("context_update_detected")


def test_world_data_service_persists_normalized_edits_and_extends_locked_fields(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_normalized_world_data_fixture(repositories)
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )
    model = service.build_model()
    assert model.scene is not None
    location = cast(dict[str, WorldDataLocationRow], _items_by_id(model.locations))[
        ids["location"]
    ]
    character = cast(dict[str, WorldDataCharacterRow], _items_by_id(model.characters))[
        ids["character"]
    ]
    thread = cast(dict[str, WorldDataThreadRow], _items_by_id(model.threads))[
        ids["thread"]
    ]
    link = cast(dict[str, WorldDataEntityLinkRow], _items_by_id(model.links))[
        ids["link"]
    ]

    service.apply_edits(
        world_data.WorldDataEdits(
            scenario=_scenario_edit_from_model(world_data, model),
            scene=replace(
                model.scene,
                situation="The red lens rings under pressure.",
                nearby_objects_text="lens crank, signal slate",
            ),
            locations=(
                replace(
                    location,
                    description="A high gallery of red glass and ringing brass.",
                    hazards_text="cracked lens, ash leak",
                ),
            ),
            characters=(
                replace(
                    character,
                    status="guarding the lens",
                    visual_notes="Ash on her epaulets",
                    texting_style="Short status pings, signs messages -I.",
                ),
            ),
            threads=(
                replace(
                    thread,
                    priority=9,
                    related_entities_text=(
                        f"location:{ids['location']}, character:{ids['character']}"
                    ),
                ),
            ),
            links=(
                replace(link, deleted=True),
                world_data.WorldDataEntityLinkRow(
                    link_id="",
                    entity_type="character",
                    entity_id=ids["character"],
                    target_type="world_state",
                    target_id=ids["state"],
                    relation="knows",
                ),
            ),
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert snapshot.situation == "The red lens rings under pressure."
    assert snapshot.nearby_objects == ["lens crank", "signal slate"]
    assert {"situation", "nearby_objects_text"} <= set(snapshot.locked_fields)
    saved_location = repositories.get_location(ids["location"])
    assert saved_location is not None
    assert (
        saved_location.description == "A high gallery of red glass and ringing brass."
    )
    assert saved_location.hazards == ["cracked lens", "ash leak"]
    assert {"description", "hazards_text"} <= set(saved_location.locked_fields)
    saved_character = repositories.get_character(ids["character"])
    assert saved_character is not None
    assert saved_character.status == "guarding the lens"
    assert saved_character.visual_notes == "Ash on her epaulets"
    assert saved_character.texting_style == "Short status pings, signs messages -I."
    assert {"status", "visual_notes", "texting_style"} <= set(
        saved_character.locked_fields
    )
    saved_thread = repositories.get_active_thread(ids["thread"])
    assert saved_thread is not None
    assert saved_thread.priority == 9
    assert saved_thread.related_entities == [
        f"location:{ids['location']}",
        f"character:{ids['character']}",
    ]
    assert {"priority", "related_entities"} <= set(saved_thread.locked_fields)
    assert [
        (item.entity_type, item.entity_id, item.target_type, item.target_id)
        for item in repositories.list_entity_links(save_id)
    ] == [("character", ids["character"], "world_state", ids["state"])]
    saved_link = repositories.list_entity_links(save_id)[0]
    assert saved_link.source_message_id is None


def test_world_data_service_syncs_canonical_scene_time_on_manual_edit(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, _ids = _persist_normalized_world_data_fixture(repositories)
    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        current_location_id=snapshot.current_location_id,
        situation=snapshot.situation,
        objective=snapshot.objective,
        in_world_time="Monday morning",
        time_of_day="morning",
        day_of_week="monday",
        world_day_index=2,
        world_time_clock_minutes=9 * 60 + 30,
        world_time_period_label="festival week",
        weather=snapshot.weather,
        mood=snapshot.mood,
        nearby_objects=snapshot.nearby_objects,
        hazards=snapshot.hazards,
        present_character_ids=snapshot.present_character_ids,
        source_message_id=snapshot.source_message_id,
        locked_fields=snapshot.locked_fields,
        snapshot_id=snapshot.id,
    )
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )
    model = service.build_model()
    assert model.scene is not None

    service.apply_edits(
        world_data.WorldDataEdits(
            scenario=_scenario_edit_from_model(world_data, model),
            scene=replace(
                model.scene,
                in_world_time="Tuesday evening",
                time_of_day="evening",
                day_of_week="tuesday",
                world_day_index=3,
            ),
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert snapshot.in_world_time == "Tuesday evening"
    assert snapshot.time_of_day == "evening"
    assert snapshot.day_of_week == "tuesday"
    assert snapshot.world_day_index == 3
    assert snapshot.world_time_phase == "evening"
    assert snapshot.world_time_day_label == "tuesday"
    assert snapshot.world_time_day_index == 3
    assert snapshot.world_time_clock_minutes == 9 * 60 + 30
    assert snapshot.world_time_period_label == "festival week"
    assert snapshot.world_time_source_message_id is None


def test_world_data_service_preserves_legacy_scene_time_for_day_index_edit(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, _ids = _persist_normalized_world_data_fixture(repositories)
    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        current_location_id=snapshot.current_location_id,
        situation=snapshot.situation,
        objective=snapshot.objective,
        in_world_time="Friday festival week evening",
        time_of_day="evening",
        day_of_week="friday",
        world_day_index=8,
        world_time_day_index=8,
        world_time_day_label="friday",
        world_time_phase="evening",
        world_time_period_label="festival week",
        weather=snapshot.weather,
        mood=snapshot.mood,
        nearby_objects=snapshot.nearby_objects,
        hazards=snapshot.hazards,
        present_character_ids=snapshot.present_character_ids,
        source_message_id=snapshot.source_message_id,
        locked_fields=snapshot.locked_fields,
        snapshot_id=snapshot.id,
    )
    repositories.connection.execute(
        """
        UPDATE scene_snapshots
        SET in_world_time = 'Friday evening after the festival'
        WHERE save_id = ?
        """,
        (save_id,),
    )
    repositories.connection.commit()
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )
    model = service.build_model()
    assert model.scene is not None

    service.apply_edits(
        world_data.WorldDataEdits(
            scenario=_scenario_edit_from_model(world_data, model),
            scene=replace(model.scene, world_day_index=9),
        )
    )

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert snapshot.in_world_time == "Friday evening after the festival"
    assert snapshot.time_of_day == "evening"
    assert snapshot.day_of_week == "friday"
    assert snapshot.world_day_index == 9
    assert snapshot.world_time_day_index == 9
    assert snapshot.world_time_day_label == "friday"
    assert snapshot.world_time_phase == "evening"
    assert snapshot.world_time_period_label == "festival week"


def test_world_data_service_explicitly_unlocks_location_and_thread_fields(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_normalized_world_data_fixture(repositories)
    location = repositories.get_location(ids["location"])
    thread = repositories.get_active_thread(ids["thread"])
    assert location is not None
    assert thread is not None
    repositories.update_location(
        replace(location, locked_fields=["description", "status"])
    )
    repositories.update_active_thread(
        replace(thread, locked_fields=["description", "priority"])
    )
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )
    model = service.build_model()
    location_row = cast(
        dict[str, WorldDataLocationRow],
        _items_by_id(model.locations),
    )[ids["location"]]
    thread_row = cast(
        dict[str, WorldDataThreadRow],
        _items_by_id(model.threads),
    )[ids["thread"]]

    service.apply_edits(
        world_data.WorldDataEdits(
            scenario=_scenario_edit_from_model(world_data, model),
            locations=(replace(location_row, locked_fields=("description",)),),
            threads=(replace(thread_row, locked_fields=("priority",)),),
        )
    )

    unlocked_location = repositories.get_location(ids["location"])
    unlocked_thread = repositories.get_active_thread(ids["thread"])
    assert unlocked_location is not None
    assert unlocked_location.locked_fields == ["description"]
    assert unlocked_thread is not None
    assert unlocked_thread.locked_fields == ["priority"]


def test_world_data_service_manual_context_edits_clear_message_provenance(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_normalized_world_data_fixture(repositories)
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )
    model = service.build_model()
    assert model.scene is not None
    state = _items_by_id(model.state_rows)[ids["state"]]
    memory = _items_by_id(model.memory_rows)[ids["memory"]]
    location = _items_by_id(model.locations)[ids["location"]]
    character = _items_by_id(model.characters)[ids["character"]]
    thread = _items_by_id(model.threads)[ids["thread"]]

    service.apply_edits(
        world_data.WorldDataEdits(
            scenario=_scenario_edit_from_model(world_data, model),
            world_state=(
                replace(
                    cast(WorldDataStateRow, state),
                    value_json='{"name":"Beacon Gallery","threat":"stable"}',
                ),
            ),
            memories=(
                replace(
                    cast(WorldDataMemoryRow, memory),
                    body="Manual correction: Ilyra holds the stair.",
                ),
            ),
            scene=replace(
                model.scene,
                situation="Manual correction: the lens is stable.",
            ),
            locations=(
                replace(
                    cast(WorldDataLocationRow, location),
                    description="Manual correction: stable red glass.",
                ),
            ),
            characters=(
                replace(
                    cast(WorldDataCharacterRow, character),
                    status="manually confirmed present",
                    current_intent="Stay beside the lens until it cools.",
                ),
            ),
            threads=(
                replace(
                    cast(WorldDataThreadRow, thread),
                    description="Manual correction: the repair is under control.",
                ),
            ),
        )
    )

    saved_state = repositories.list_world_state(save_id)[0]
    saved_memory = repositories.list_memories(save_id)[0]
    saved_snapshot = repositories.get_scene_snapshot(save_id)
    saved_location = repositories.get_location(ids["location"])
    saved_character = repositories.get_character(ids["character"])
    saved_thread = repositories.get_active_thread(ids["thread"])
    assert saved_state.source_message_id is None
    assert saved_memory.source_message_id is None
    assert saved_memory.source_message_ids == []
    assert saved_snapshot is not None
    assert saved_snapshot.source_message_id is None
    assert saved_location is not None
    assert saved_location.source_message_id is None
    assert saved_character is not None
    assert saved_character.source_message_id is None
    assert saved_character.current_intent == "Stay beside the lens until it cools."
    assert "current_intent" in saved_character.locked_fields
    assert saved_thread is not None
    assert saved_thread.source_message_id is None


def test_world_data_service_validates_scene_references_and_entity_links(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_normalized_world_data_fixture(repositories)
    other_save_id, other_ids = _persist_normalized_world_data_fixture(repositories)
    foreign_link = repositories.add_entity_link(
        save_id=other_save_id,
        entity_type="location",
        entity_id=other_ids["location"],
        target_type="memory",
        target_id=other_ids["memory"],
    )
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )
    model = service.build_model()
    assert model.scene is not None

    with pytest.raises(ValueError, match="Scene current location.*active save"):
        service.apply_edits(
            world_data.WorldDataEdits(
                scenario=_scenario_edit_from_model(world_data, model),
                scene=replace(model.scene, current_location_id=other_ids["location"]),
            )
        )

    with pytest.raises(ValueError, match="Scene present character.*active save"):
        service.apply_edits(
            world_data.WorldDataEdits(
                scenario=_scenario_edit_from_model(world_data, model),
                scene=replace(
                    model.scene,
                    present_character_ids_text=(
                        f"{ids['character']}, {other_ids['character']}"
                    ),
                ),
            )
        )

    with pytest.raises(ValueError, match="Entity link.*active save"):
        service.apply_edits(
            world_data.WorldDataEdits(
                scenario=_scenario_edit_from_model(world_data, model),
                links=(
                    world_data.WorldDataEntityLinkRow(
                        link_id=foreign_link.id,
                        entity_type=foreign_link.entity_type,
                        entity_id=foreign_link.entity_id,
                        target_type=foreign_link.target_type,
                        target_id=foreign_link.target_id,
                        relation=foreign_link.relation,
                        deleted=True,
                    ),
                ),
            )
        )

    with pytest.raises(ValueError, match="Entity link target.*active save"):
        service.apply_edits(
            world_data.WorldDataEdits(
                scenario=_scenario_edit_from_model(world_data, model),
                links=(
                    world_data.WorldDataEntityLinkRow(
                        link_id="",
                        entity_type="location",
                        entity_id=ids["location"],
                        target_type="artifact",
                        target_id=ids["state"],
                        relation="mentions",
                    ),
                ),
            )
        )


def test_world_data_service_validates_context_edit_ids_and_row_shapes(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    active_save_id, active_ids = _persist_normalized_world_data_fixture(repositories)
    other_save_id, other_ids = _persist_normalized_world_data_fixture(repositories)
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=active_save_id,
    )
    active_model = service.build_model()
    other_model = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=other_save_id,
    ).build_model()
    active_scenario = _scenario_edit_from_model(world_data, active_model)
    active_characters = cast(
        dict[str, WorldDataCharacterRow],
        _items_by_id(active_model.characters),
    )
    active_threads = cast(
        dict[str, WorldDataThreadRow],
        _items_by_id(active_model.threads),
    )
    other_locations = cast(
        dict[str, WorldDataLocationRow],
        _items_by_id(other_model.locations),
    )
    other_characters = cast(
        dict[str, WorldDataCharacterRow],
        _items_by_id(other_model.characters),
    )
    other_threads = cast(
        dict[str, WorldDataThreadRow],
        _items_by_id(other_model.threads),
    )

    with pytest.raises(ValueError, match="Location edit.*active save"):
        service.apply_edits(
            world_data.WorldDataEdits(
                scenario=active_scenario,
                locations=(other_locations[other_ids["location"]],),
            )
        )

    with pytest.raises(ValueError, match="Character edit.*active save"):
        service.apply_edits(
            world_data.WorldDataEdits(
                scenario=active_scenario,
                characters=(other_characters[other_ids["character"]],),
            )
        )

    with pytest.raises(ValueError, match="Thread edit.*active save"):
        service.apply_edits(
            world_data.WorldDataEdits(
                scenario=active_scenario,
                threads=(other_threads[other_ids["thread"]],),
            )
        )

    with pytest.raises(ValueError, match="Character location.*active save"):
        service.apply_edits(
            world_data.WorldDataEdits(
                scenario=active_scenario,
                characters=(
                    replace(
                        active_characters[active_ids["character"]],
                        location_id=other_ids["location"],
                    ),
                ),
            )
        )

    with pytest.raises(ValueError, match="Thread source message.*active save"):
        service.apply_edits(
            world_data.WorldDataEdits(
                scenario=active_scenario,
                threads=(
                    replace(
                        active_threads[active_ids["thread"]],
                        source_message_id=other_ids["message"],
                    ),
                ),
            )
        )

    with pytest.raises(ValueError, match="relationships must be a JSON object"):
        service.apply_edits(
            world_data.WorldDataEdits(
                scenario=active_scenario,
                characters=(
                    replace(
                        active_characters[active_ids["character"]],
                        relationships_json='["ally"]',
                    ),
                ),
            )
        )

    with pytest.raises(TypeError, match="Unsupported location edit row"):
        service.apply_edits(
            world_data.WorldDataEdits(
                scenario=active_scenario,
                locations=(object(),),
            )
        )


def test_world_data_service_rejects_unsupported_suggestion_actions(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_normalized_world_data_fixture(repositories)
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )
    model = service.build_model()
    suggestions = cast(
        dict[str, WorldDataSuggestionRow],
        _items_by_id(model.suggestions),
    )

    with pytest.raises(ValueError, match="Unsupported suggestion action"):
        service.apply_edits(
            world_data.WorldDataEdits(
                scenario=_scenario_edit_from_model(world_data, model),
                suggestions=(replace(suggestions[ids["suggestion"]], action="defer"),),
            )
        )

    assert repositories.list_context_update_suggestions(save_id)[0].status == "pending"


def test_world_data_service_applies_and_rejects_suggestions(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_normalized_world_data_fixture(repositories)
    rejected = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="field_update",
        entity_type="active_thread",
        entity_id=ids["thread"],
        field_path="title",
        proposed_value="Ignore the beacon",
        reason="Low confidence branch",
        confidence=0.2,
        source_message_ids=[ids["message"]],
    )
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )
    model = service.build_model()
    suggestions = cast(
        dict[str, WorldDataSuggestionRow],
        _items_by_id(model.suggestions),
    )

    service.apply_edits(
        world_data.WorldDataEdits(
            scenario=_scenario_edit_from_model(world_data, model),
            suggestions=(
                replace(suggestions[ids["suggestion"]], action="apply"),
                replace(suggestions[rejected.id], action="reject"),
            ),
        )
    )

    location = repositories.get_location(ids["location"])
    assert location is not None
    assert location.description == "The gallery glass burns with a fresh red warning."
    assert "description" in location.locked_fields
    statuses = {
        item.id: item.status
        for item in repositories.list_context_update_suggestions(save_id)
    }
    assert statuses[ids["suggestion"]] == "applied"
    assert statuses[rejected.id] == "rejected"
    applied_audit = [
        item
        for item in repositories.list_context_update_audit(save_id)
        if item.suggestion_id == ids["suggestion"]
    ]
    manual_apply_audit = [
        item for item in applied_audit if item.operation == "manual_suggestion_apply"
    ]
    assert len(manual_apply_audit) == 1
    assert manual_apply_audit[0].before == "The beacon lens overlooks the ash gate."
    assert manual_apply_audit[0].after == (
        "The gallery glass burns with a fresh red warning."
    )


def test_world_data_service_applies_entity_link_delete_suggestion(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_normalized_world_data_fixture(repositories)
    suggestion = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="delete",
        entity_type="entity_link",
        entity_id=ids["link"],
        field_path="*",
        proposed_value=None,
        reason="Cleanup suggested removing this stale relationship.",
        confidence=0.87,
        source_message_ids=[ids["message"]],
    )
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )

    service.apply_suggestions((suggestion.id,))

    assert ids["link"] not in {
        item.id for item in repositories.list_entity_links(save_id)
    }
    stored = {
        item.id: item for item in repositories.list_context_update_suggestions(save_id)
    }
    assert stored[suggestion.id].status == "applied"
    audit = [
        item
        for item in repositories.list_context_update_audit(save_id)
        if item.suggestion_id == suggestion.id
    ][0]
    assert audit.operation == "agent_suggestion_apply"
    assert audit.entity_type == "entity_link"
    assert audit.entity_id == ids["link"]


def test_world_data_service_applies_memory_suggestion_with_observation_provenance(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_normalized_world_data_fixture(repositories)
    observation = repositories.add_context_observation(
        save_id=save_id,
        observation_type="promise",
        claim="Ilyra promised to guard the stair.",
        evidence_quote="Ilyra holds the stair",
        source_message_ids=[ids["message"]],
        scope="durable",
        status="needs_confirmation",
    )
    fingerprint = "ilyra-promised-to-guard-the-stair"
    suggestion = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="create",
        entity_type="memory",
        field_path="*",
        proposed_value={
            "body": observation.claim,
            "tags": ["ilyra", "promise"],
            "importance": 0.9,
            "source_message_id": ids["message"],
            "source_message_ids": [ids["message"]],
            "source_observation_ids": [observation.id],
            "claim_fingerprint": fingerprint,
        },
        source_message_ids=[ids["message"]],
    )
    existing = repositories.add_memory(
        save_id=save_id,
        body=observation.claim,
        tags=["existing"],
        importance=0.4,
        source_message_ids=[],
        source_observation_ids=["earlier-observation"],
    )
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )

    service.apply_suggestions((suggestion.id,))

    [memory] = [
        memory
        for memory in repositories.list_memories(save_id)
        if memory.body == observation.claim
    ]
    assert memory.id == existing.id
    assert memory.tags == ["existing", "ilyra", "promise"]
    assert memory.importance == 0.9
    assert memory.source_message_ids == [ids["message"]]
    assert memory.source_observation_ids == [
        "earlier-observation",
        observation.id,
    ]
    assert memory.claim_fingerprint != fingerprint
    assert memory.claim_fingerprint


def test_world_data_service_applies_scene_time_suggestion_with_canonical_provenance(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_normalized_world_data_fixture(repositories)
    suggestion = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="field_update",
        entity_type="scene_snapshot",
        entity_id=ids["snapshot"],
        field_path="time_of_day",
        proposed_value="evening",
        reason="Narration moved the scene into evening.",
        confidence=0.91,
        source_message_ids=[ids["message"]],
    )
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )

    service.apply_suggestions((suggestion.id,))

    snapshot = repositories.get_scene_snapshot(save_id)
    assert snapshot is not None
    assert snapshot.time_of_day == "evening"
    assert snapshot.world_time_phase == "evening"
    assert snapshot.world_time_source_message_id == ids["message"]
    assert snapshot.world_time_confidence == 0.91


def test_world_data_service_accepts_locked_transition_without_advancing_twice(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_normalized_world_data_fixture(repositories)
    next_location = repositories.add_location(
        save_id=save_id,
        name="Gatehouse",
        source_message_id=ids["message"],
    )
    before_transition = repositories.get_scene_snapshot(save_id)
    assert before_transition is not None
    transitioned = repositories.advance_scene_generation(
        save_id=save_id,
        source_message_id=ids["message"],
    )
    repositories.add_context_update_audit(
        save_id=save_id,
        operation="scene_generation_advanced",
        entity_type="scene_snapshot",
        entity_id=transitioned.id,
        field_path="scene_generation",
        before=before_transition.scene_generation,
        after=transitioned.scene_generation,
        source_message_ids=[ids["message"]],
    )
    scratch = repositories.upsert_context_source(
        save_id=save_id,
        source_type="observation",
        source_id="new-scene-scratch",
        title="Gatehouse arrival",
        body="The gatehouse doors are closing.",
        metadata={"curation_action": "scene_scratch"},
        scene_snapshot_id=transitioned.id,
        scene_generation=transitioned.scene_generation,
        created_turn_number=1,
        expires_after_turn_number=13,
    )
    scene_thread = repositories.add_active_thread(
        save_id=save_id,
        title="Gatehouse countdown",
        visibility="scene local",
    )
    public_thread = repositories.add_active_thread(
        save_id=save_id,
        title="Reach the capital",
        visibility="public",
    )
    suggestion = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="field_update",
        entity_type="scene_snapshot",
        entity_id=transitioned.id,
        field_path="current_location_id",
        proposed_value=next_location.id,
        source_message_ids=[ids["message"]],
    )
    repositories.add_context_update_audit(
        save_id=save_id,
        suggestion_id=suggestion.id,
        operation="queued",
        entity_type="scene_snapshot",
        entity_id=transitioned.id,
        field_path="current_location_id",
        before=before_transition.current_location_id,
        after=next_location.id,
        source_message_ids=[ids["message"]],
    )
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )

    service.apply_suggestions((suggestion.id,))

    updated = repositories.get_scene_snapshot(save_id)
    assert updated is not None
    assert updated.current_location_id == next_location.id
    assert updated.scene_generation == transitioned.scene_generation
    assert repositories.get_context_source(scratch.id) is not None
    assert repositories.get_active_thread(scene_thread.id) is None
    assert repositories.get_active_thread(public_thread.id) is not None


def test_world_data_service_rejects_protected_character_archive_suggestion(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_normalized_world_data_fixture(repositories)
    character = repositories.get_character(ids["character"])
    assert character is not None
    repositories.update_character(replace(character, protected_from_maintenance=True))
    archive_suggestion = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="archive",
        entity_type="character",
        entity_id=ids["character"],
        field_path="*",
        proposed_value=None,
        reason="Cleanup suggested archiving this protected character.",
        confidence=0.95,
        source_message_ids=[ids["message"]],
    )
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )
    model = service.build_model()
    suggestions = cast(
        dict[str, WorldDataSuggestionRow],
        _items_by_id(model.suggestions),
    )

    with pytest.raises(ValueError, match="protected from maintenance"):
        service.apply_edits(
            world_data.WorldDataEdits(
                scenario=_scenario_edit_from_model(world_data, model),
                suggestions=(
                    replace(suggestions[archive_suggestion.id], action="apply"),
                ),
            )
        )

    assert repositories.get_character(ids["character"]) is not None
    assert repositories.list_context_update_suggestions(save_id)[0].status == "pending"


@pytest.mark.parametrize(
    ("entity_type", "id_key"),
    (
        ("memory", "memory"),
        ("summary", "summary"),
        ("world_state", "state"),
        ("location", "location"),
        ("character", "character"),
        ("active_thread", "thread"),
    ),
)
def test_world_data_service_archive_suggestion_deletes_related_entity_links(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
    entity_type: str,
    id_key: str,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_normalized_world_data_fixture(repositories)
    if entity_type == "location":
        target_id = repositories.add_location(
            save_id=save_id,
            name="Old Watchroom",
        ).id
    else:
        target_id = ids[id_key]
    proposed_value = None
    update_type = "archive"
    field_path = "*"
    if entity_type == "world_state":
        state = repositories.list_world_state(save_id)[0]
        proposed_value = {"operation": "delete", "key": state.key}
        update_type = "field_update"
        field_path = state.key
    link = repositories.add_entity_link(
        save_id=save_id,
        entity_type=entity_type,
        entity_id=target_id,
        target_type="memory",
        target_id=ids["memory"],
        relation="references",
        link_id=f"link-archive-{entity_type}",
    )
    if entity_type == "memory":
        repositories.add_entity_link(
            save_id=save_id,
            entity_type="character",
            entity_id=ids["character"],
            target_type="memory",
            target_id=target_id,
            relation="knows",
            link_id="link-character-archived-memory",
        )
    suggestion = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type=update_type,
        entity_type=entity_type,
        entity_id=target_id,
        field_path=field_path,
        proposed_value=proposed_value,
        reason="Cleanup suggested archiving stale world data.",
        confidence=0.91,
        source_message_ids=[ids["message"]],
    )
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )
    model = service.build_model()
    suggestions = cast(
        dict[str, WorldDataSuggestionRow],
        _items_by_id(model.suggestions),
    )

    service.apply_edits(
        world_data.WorldDataEdits(
            scenario=_scenario_edit_from_model(world_data, model),
            suggestions=(replace(suggestions[suggestion.id], action="apply"),),
        )
    )

    remaining_link_ids = {item.id for item in repositories.list_entity_links(save_id)}
    assert link.id not in remaining_link_ids
    if entity_type == "memory":
        assert "link-character-archived-memory" not in remaining_link_ids
    assert repositories.list_context_update_suggestions(save_id)[-1].status == "applied"


def test_world_data_service_groups_pending_suggestions_by_target_and_value(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_normalized_world_data_fixture(repositories)
    duplicate = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="field_update",
        entity_type="location",
        entity_id=ids["location"],
        field_path="description",
        proposed_value="The gallery glass burns with a fresh red warning.",
        reason="Second extractor saw the same red warning.",
        confidence=0.7,
        source_message_ids=[ids["message"]],
    )
    repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="field_update",
        entity_type="location",
        entity_id=ids["location"],
        field_path="description",
        proposed_value="The gallery glass is quiet again.",
        reason="A different proposed value must stay separate.",
        confidence=0.4,
        source_message_ids=[ids["message"]],
    )
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )

    model = service.build_model()
    groups = list(model.suggestion_group_rows)
    duplicate_group = next(
        group
        for group in groups
        if group.suggestion_ids == (ids["suggestion"], duplicate.id)
    )

    assert len(groups) == 2
    assert duplicate_group.group_id.startswith("sugggrp-")
    assert duplicate_group.suggestion_count == 2
    assert duplicate_group.status == "pending"
    assert duplicate_group.proposed_value_json == (
        '"The gallery glass burns with a fresh red warning."'
    )
    assert duplicate_group.reason == (
        "Narration moved the lens state forward.\n"
        "Second extractor saw the same red warning."
    )
    assert service.build_model().suggestion_group_rows[0].group_id == groups[0].group_id


def test_world_data_service_applies_grouped_suggestions_once_with_group_audit(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_normalized_world_data_fixture(repositories)
    duplicate = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="field_update",
        entity_type="location",
        entity_id=ids["location"],
        field_path="description",
        proposed_value="The gallery glass burns with a fresh red warning.",
        reason="Second extractor saw the same red warning.",
        confidence=0.7,
        source_message_ids=[ids["message"]],
    )
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )
    model = service.build_model()
    group = model.suggestion_group_rows[0]

    service.apply_edits(
        world_data.WorldDataEdits(
            scenario=_scenario_edit_from_model(world_data, model),
            suggestion_groups=(replace(group, action="apply"),),
        )
    )

    location = repositories.get_location(ids["location"])
    assert location is not None
    assert location.description == "The gallery glass burns with a fresh red warning."
    statuses = {
        item.id: item.status
        for item in repositories.list_context_update_suggestions(save_id)
    }
    assert statuses[ids["suggestion"]] == "applied"
    assert statuses[duplicate.id] == "applied"
    group_audit = [
        item
        for item in repositories.list_context_update_audit(save_id)
        if item.operation == "manual_suggestion_group_apply"
    ]
    assert {item.suggestion_id for item in group_audit} == {
        ids["suggestion"],
        duplicate.id,
    }


def test_world_data_service_applies_grouped_archive_suggestions_once(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_normalized_world_data_fixture(repositories)
    first = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="archive",
        entity_type="memory",
        entity_id=ids["memory"],
        field_path="*",
        proposed_value=None,
        reason="Cleanup detected the memory is obsolete.",
        confidence=0.91,
        source_message_ids=[ids["message"]],
    )
    second = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="archive",
        entity_type="memory",
        entity_id=ids["memory"],
        field_path="*",
        proposed_value=None,
        reason="A second pass detected the same obsolete memory.",
        confidence=0.73,
        source_message_ids=[ids["message"]],
    )
    repositories.add_entity_link(
        save_id=save_id,
        entity_type="character",
        entity_id=ids["character"],
        target_type="memory",
        target_id=ids["memory"],
        relation="knows",
        link_id="link-character-memory-archive",
    )
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )
    model = service.build_model()
    group = next(
        group
        for group in model.suggestion_group_rows
        if group.suggestion_ids == (first.id, second.id)
    )

    service.apply_edits(
        world_data.WorldDataEdits(
            scenario=_scenario_edit_from_model(world_data, model),
            suggestion_groups=(replace(group, action="apply"),),
        )
    )

    assert repositories.list_memories(save_id) == []
    statuses = {
        item.id: item.status
        for item in repositories.list_context_update_suggestions(save_id)
    }
    assert statuses[first.id] == "applied"
    assert statuses[second.id] == "applied"
    remaining_link_ids = {link.id for link in repositories.list_entity_links(save_id)}
    assert ids["link"] not in remaining_link_ids
    assert "link-character-memory-archive" not in remaining_link_ids
    group_audit = [
        item
        for item in repositories.list_context_update_audit(save_id)
        if item.operation == "manual_suggestion_group_apply"
        and item.entity_type == "memory"
    ]
    assert {item.suggestion_id for item in group_audit} == {first.id, second.id}


def test_world_data_service_rejects_and_dismisses_grouped_suggestions(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_normalized_world_data_fixture(repositories)
    dismissed = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="field_update",
        entity_type="active_thread",
        entity_id=ids["thread"],
        field_path="title",
        proposed_value="Ignore the beacon",
        reason="Low confidence branch",
        confidence=0.2,
        source_message_ids=[ids["message"]],
    )
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )
    model = service.build_model()
    groups = {
        group.entity_type: group for group in model.suggestion_group_rows
    }

    service.apply_edits(
        world_data.WorldDataEdits(
            scenario=_scenario_edit_from_model(world_data, model),
            suggestion_groups=(
                replace(groups["location"], action="reject"),
                replace(groups["active_thread"], action="dismiss"),
            ),
        )
    )

    statuses = {
        item.id: item.status
        for item in repositories.list_context_update_suggestions(save_id)
    }
    assert statuses[ids["suggestion"]] == "rejected"
    assert statuses[dismissed.id] == "dismissed"
    audit_operations = {
        item.suggestion_id: item.operation
        for item in repositories.list_context_update_audit(save_id)
        if item.operation.startswith("manual_suggestion_group_")
    }
    assert audit_operations[ids["suggestion"]] == "manual_suggestion_group_reject"
    assert audit_operations[dismissed.id] == "manual_suggestion_group_dismiss"


def test_world_data_service_expires_stale_pending_suggestions_when_opened(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_normalized_world_data_fixture(repositories)
    repositories.connection.execute(
        """
        UPDATE context_update_suggestions
        SET created_at = datetime('now', '-31 days')
        WHERE id = ?
        """,
        (ids["suggestion"],),
    )
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )

    model = service.build_model()

    assert model.suggestion_group_rows == ()
    suggestion = repositories.list_context_update_suggestions(save_id)[0]
    assert suggestion.status == "expired"
    assert suggestion.resolved_at is not None
    audit = repositories.list_context_update_audit(save_id)[-1]
    assert audit.operation == "suggestion_expired"
    assert audit.suggestion_id == ids["suggestion"]


def test_world_data_service_applies_suggestions_without_clobbering_manual_edits(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_normalized_world_data_fixture(repositories)
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )
    model = service.build_model()
    location = cast(dict[str, WorldDataLocationRow], _items_by_id(model.locations))[
        ids["location"]
    ]
    suggestions = cast(
        dict[str, WorldDataSuggestionRow],
        _items_by_id(model.suggestions),
    )

    service.apply_edits(
        world_data.WorldDataEdits(
            scenario=_scenario_edit_from_model(world_data, model),
            locations=(replace(location, status="manually stabilized"),),
            suggestions=(replace(suggestions[ids["suggestion"]], action="apply"),),
        )
    )

    saved_location = repositories.get_location(ids["location"])
    assert saved_location is not None
    assert saved_location.status == "manually stabilized"
    assert saved_location.description == (
        "The gallery glass burns with a fresh red warning."
    )
    assert {"status", "description"} <= set(saved_location.locked_fields)


def test_world_data_service_creates_and_archives_locations_and_threads(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_normalized_world_data_fixture(repositories)
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )
    model = service.build_model()
    location = cast(dict[str, WorldDataLocationRow], _items_by_id(model.locations))[
        ids["location"]
    ]
    thread = cast(dict[str, WorldDataThreadRow], _items_by_id(model.threads))[
        ids["thread"]
    ]
    orphan = repositories.add_location(
        save_id=save_id,
        name="Old Watchroom",
    )
    orphan_thread = repositories.add_active_thread(
        save_id=save_id,
        title="Old lead",
    )
    link = repositories.add_entity_link(
        save_id=save_id,
        entity_type="location",
        entity_id=orphan.id,
        target_type="active_thread",
        target_id=orphan_thread.id,
        relation="mentions",
    )
    model = service.build_model()
    rows_by_id = cast(dict[str, WorldDataLocationRow], _items_by_id(model.locations))
    threads_by_id = cast(dict[str, WorldDataThreadRow], _items_by_id(model.threads))

    result = service.apply_edits(
        world_data.WorldDataEdits(
            scenario=_scenario_edit_from_model(world_data, model),
            locations=(
                location,
                replace(rows_by_id[orphan.id], archived=True),
                world_data.WorldDataLocationRow(
                    location_id="",
                    name="Glass Stair",
                    aliases_text="spiral stair",
                    description="A stair packed with red lens dust.",
                    visual_description="Ruby dust on brass rails.",
                    parent_location_id=ids["location"],
                    connections_text="Beacon Gallery",
                    status="new",
                    hazards_text="slippery glass",
                    source_message_id=None,
                ),
            ),
            threads=(
                thread,
                replace(threads_by_id[orphan_thread.id], archived=True),
                world_data.WorldDataThreadRow(
                    thread_id="",
                    title="Reach the lower lens",
                    description="The lower housing may still be intact.",
                    status="active",
                    priority=4,
                    visibility="public",
                    related_entities_text=f"location:{ids['location']}",
                    source_message_id=None,
                ),
            ),
        )
    )

    assert result.location_archive_count == 1
    assert result.thread_archive_count == 1
    assert repositories.get_location(orphan.id) is None
    assert repositories.get_active_thread(orphan_thread.id) is None
    assert link.id not in {item.id for item in repositories.list_entity_links(save_id)}
    created_locations = {
        location.name: location for location in repositories.list_locations(save_id)
    }
    assert created_locations["Glass Stair"].parent_location_id == ids["location"]
    assert created_locations["Glass Stair"].hazards == ["slippery glass"]
    created_threads = {
        thread.title: thread for thread in repositories.list_active_threads(save_id)
    }
    assert created_threads["Reach the lower lens"].priority == 4


def test_world_data_service_archiving_context_rows_deletes_related_entity_links(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_normalized_world_data_fixture(repositories)
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )
    state = repositories.upsert_world_state(
        save_id=save_id,
        key="beacon.lens",
        value={"name": "Beacon Lens"},
    )
    memory = repositories.add_memory(
        save_id=save_id,
        body="The lens accepts copper keys.",
        tags=["lens"],
    )
    summary = repositories.add_summary(
        save_id=save_id,
        covers_message_start_id=ids["message"],
        covers_message_end_id=ids["message"],
        body="The beacon lens was stabilized.",
        provider="fake",
        model="fake-summary",
    )
    repositories.add_entity_link(
        save_id=save_id,
        entity_type="world_state",
        entity_id=state.id,
        target_type="memory",
        target_id=memory.id,
        relation="evidenced_by",
        link_id="link-state-memory",
    )
    repositories.add_entity_link(
        save_id=save_id,
        entity_type="summary",
        entity_id=summary.id,
        target_type="world_state",
        target_id=state.id,
        relation="summarizes",
        link_id="link-summary-state",
    )
    repositories.add_entity_link(
        save_id=save_id,
        entity_type="character",
        entity_id=ids["character"],
        target_type="summary",
        target_id=summary.id,
        relation="knows",
        link_id="link-character-summary",
    )
    model = service.build_model()
    state_rows = cast(dict[str, WorldDataStateRow], _items_by_id(model.state_rows))
    memory_rows = cast(dict[str, WorldDataMemoryRow], _items_by_id(model.memory_rows))
    summary_rows = cast(
        dict[str, WorldDataSummaryRow],
        _items_by_id(model.summary_rows),
    )

    result = service.apply_edits(
        world_data.WorldDataEdits(
            scenario=_scenario_edit_from_model(world_data, model),
            world_state=(replace(state_rows[state.id], archived=True),),
            memories=(
                replace(
                    memory_rows[memory.id],
                    archived=True,
                ),
            ),
            summaries=(
                replace(
                    summary_rows[summary.id],
                    archived=True,
                ),
            ),
        )
    )

    assert result.state_archive_count == 1
    assert result.memory_archive_count == 1
    assert result.summary_archive_count == 1
    remaining_link_ids = {link.id for link in repositories.list_entity_links(save_id)}
    assert "link-state-memory" not in remaining_link_ids
    assert "link-summary-state" not in remaining_link_ids
    assert "link-character-summary" not in remaining_link_ids
    assert ids["link"] in remaining_link_ids


def test_world_data_service_rejects_archiving_referenced_location(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_normalized_world_data_fixture(repositories)
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )
    model = service.build_model()
    location = cast(dict[str, WorldDataLocationRow], _items_by_id(model.locations))[
        ids["location"]
    ]

    with pytest.raises(ValueError, match="current scene location"):
        service.apply_edits(
            world_data.WorldDataEdits(
                scenario=_scenario_edit_from_model(world_data, model),
                locations=(replace(location, archived=True),),
            )
        )


def test_world_data_service_rejects_blank_created_location_and_thread(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, _ids = _persist_normalized_world_data_fixture(repositories)
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )
    model = service.build_model()

    with pytest.raises(ValueError, match="Location name is required"):
        service.apply_edits(
            world_data.WorldDataEdits(
                scenario=_scenario_edit_from_model(world_data, model),
                locations=(
                    world_data.WorldDataLocationRow(
                        location_id="",
                        name=" ",
                        aliases_text="",
                        description="",
                        visual_description="",
                        parent_location_id=None,
                        connections_text="",
                        status="",
                        hazards_text="",
                        source_message_id=None,
                    ),
                ),
            )
        )

    with pytest.raises(ValueError, match="Thread title is required"):
        service.apply_edits(
            world_data.WorldDataEdits(
                scenario=_scenario_edit_from_model(world_data, model),
                threads=(
                    world_data.WorldDataThreadRow(
                        thread_id="",
                        title=" ",
                        description="",
                        status="active",
                        priority=0,
                        visibility="public",
                        related_entities_text="",
                        source_message_id=None,
                    ),
                ),
            )
        )


def test_world_data_service_applies_manual_confirmation_create_suggestions(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_world_data_fixture(repositories)
    extra_source_message = repositories.append_message(
        save_id=save_id,
        role="player",
        speaker_name="Mara",
        body="I swear I will keep the beacon lit.",
    )
    memory_suggestion = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="create",
        entity_type="memory",
        field_path="*",
        proposed_value={
            "body": "Mara promised to relight the beacon.",
            "tags": ["promise", "beacon"],
            "importance": 0.74,
            "source_message_id": ids["message"],
            "source_message_ids": [ids["message"], extra_source_message.id],
        },
        reason="The player made a durable promise.",
        confidence=0.91,
        source_message_ids=[ids["message"], extra_source_message.id],
    )
    character_suggestion = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="create",
        entity_type="character",
        field_path="*",
        proposed_value={
            "name": "Captain Ilyra",
            "aliases": ["captain"],
            "role": "Watch captain",
            "known_state": "guarding the lens",
            "met": True,
            "appearance": "",
            "visual_notes": "",
            "personality": "",
            "voice": "",
            "texting_style": "Brief formal acknowledgements, no emoji.",
            "relationships": {"player": "ally"},
            "goals": "Keep the beacon lit.",
            "motivations": "Protect the village.",
            "current_intent": "Guard the stair.",
            "boundaries": "Will not leave the gallery.",
            "attitude_toward_player": "Cautiously allied.",
            "cooperation_conditions": "Helps after proof the lens can hold.",
            "status": "present",
            "location_id": None,
            "private_notes": "",
            "source_message_id": ids["message"],
        },
        reason="The narrator introduced Ilyra.",
        confidence=0.88,
        source_message_ids=[ids["message"]],
    )
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )
    model = service.build_model()
    suggestions = cast(
        dict[str, WorldDataSuggestionRow],
        _items_by_id(model.suggestions),
    )

    service.apply_edits(
        world_data.WorldDataEdits(
            scenario=_scenario_edit_from_model(world_data, model),
            suggestions=(
                replace(suggestions[memory_suggestion.id], action="apply"),
                replace(suggestions[character_suggestion.id], action="apply"),
            ),
        )
    )

    memories = repositories.list_memories(save_id)
    assert [memory.body for memory in memories] == [
        "Captain Ilyra promised to hold the east stair.",
        "Mara promised to relight the beacon.",
    ]
    assert memories[1].tags == ["promise", "beacon"]
    assert memories[1].importance == 0.74
    assert memories[1].source_message_ids == [
        ids["message"],
        extra_source_message.id,
    ]
    characters = repositories.list_characters(save_id)
    assert len(characters) == 1
    assert characters[0].name == "Captain Ilyra"
    assert characters[0].aliases == ["captain"]
    assert characters[0].relationships == {"player": "ally"}
    assert characters[0].texting_style == "Brief formal acknowledgements, no emoji."
    assert characters[0].goals == "Keep the beacon lit."
    assert characters[0].cooperation_conditions == (
        "Helps after proof the lens can hold."
    )
    statuses = {
        item.id: item.status
        for item in repositories.list_context_update_suggestions(save_id)
    }
    assert statuses[memory_suggestion.id] == "applied"
    assert statuses[character_suggestion.id] == "applied"


def test_world_data_service_applies_manual_confirmation_world_state_suggestions(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_world_data_fixture(repositories)
    state_suggestion = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="upsert",
        entity_type="world_state",
        entity_id=ids["state"],
        field_path="scene.location",
        proposed_value={
            "operation": "upsert",
            "key": "scene.location",
            "value": {"name": "Beacon lens", "danger": "high"},
            "category": "scene",
            "confidence": 0.92,
            "source_message_id": ids["message"],
        },
        reason="The narrator moved the action to the lens.",
        confidence=0.92,
        source_message_ids=[ids["message"]],
    )
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )
    model = service.build_model()
    suggestions = cast(
        dict[str, WorldDataSuggestionRow],
        _items_by_id(model.suggestions),
    )

    service.apply_edits(
        world_data.WorldDataEdits(
            scenario=_scenario_edit_from_model(world_data, model),
            suggestions=(replace(suggestions[state_suggestion.id], action="apply"),),
        )
    )

    world_state = repositories.list_world_state(save_id)
    assert [(state.key, state.value) for state in world_state] == [
        ("scene.location", {"name": "Beacon lens", "danger": "high"}),
    ]
    changes = list(
        repositories.connection.execute(
            """
            SELECT operation, state_key, before_json, after_json
            FROM state_changes
            WHERE save_id = ?
            """,
            (save_id,),
        )
    )
    assert changes[-1]["operation"] == "manual_suggestion_apply"
    assert changes[-1]["state_key"] == "scene.location"
    assert json.loads(changes[-1]["before_json"]) == {
        "name": "Gatehouse",
        "threat": "ash storm",
    }
    assert json.loads(changes[-1]["after_json"]) == {
        "name": "Beacon lens",
        "danger": "high",
    }
    status = repositories.list_context_update_suggestions(save_id)[0].status
    assert status == "applied"


def test_world_data_service_preserves_durable_state_when_applying_suggestion(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_world_data_fixture(repositories)
    state = repositories.upsert_world_state(
        save_id=save_id,
        key="character.ilyra.revealed_traits",
        value={
            "trusts_mara": True,
            "secret": "Knows the cracked red lens is a family heirloom.",
        },
        category="character",
        confidence=0.9,
        source_message_id=ids["message"],
    )
    state_suggestion = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="upsert",
        entity_type="world_state",
        entity_id=state.id,
        field_path=state.key,
        proposed_value={
            "operation": "upsert",
            "key": state.key,
            "value": {
                "trusts_mara": True,
                "secret": "Revealed the heirloom lens to Mara.",
            },
            "category": "character",
            "confidence": 0.92,
            "source_message_id": ids["message"],
        },
        reason="The narrator updated Ilyra's profile state.",
        confidence=0.92,
        source_message_ids=[ids["message"]],
    )
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )
    model = service.build_model()
    suggestions = cast(
        dict[str, WorldDataSuggestionRow],
        _items_by_id(model.suggestions),
    )

    service.apply_edits(
        world_data.WorldDataEdits(
            scenario=_scenario_edit_from_model(world_data, model),
            suggestions=(replace(suggestions[state_suggestion.id], action="apply"),),
        )
    )

    preservation_memories = [
        memory
        for memory in repositories.list_memories(save_id)
        if "state_history" in memory.tags
    ]
    assert len(preservation_memories) == 1
    assert preservation_memories[0].body == (
        "Previous world state for character.ilyra.revealed_traits: "
        "secret: Knows the cracked red lens is a family heirloom."
    )


def test_world_data_service_rejects_invalid_world_state_json_without_persisting(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_world_data_fixture(repositories)
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )
    model = service.build_model()
    assert model.scenario is not None

    edits = world_data.WorldDataEdits(
        scenario=world_data.WorldDataScenarioEdit(
            title=model.scenario.title,
            premise=model.scenario.premise,
            player_role=model.scenario.player_role,
            content_sections=model.scenario.content_sections,
        ),
        world_state=(
            world_data.WorldDataStateRow(
                row_id=ids["state"],
                key="scene.location",
                value_json="{not valid json",
                category="scene",
                confidence=0.9,
                source_message_id=ids["message"],
                archived=False,
                original_key="scene.location",
            ),
        ),
        memories=(),
        summaries=(),
    )

    with pytest.raises(ValueError, match="Invalid JSON|world-state JSON"):
        service.apply_edits(edits)

    state = repositories.list_world_state(save_id)[0]
    assert state.value == {"name": "Gatehouse", "threat": "ash storm"}


def test_world_data_service_archives_state_without_parsing_archived_value_json(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_world_data_fixture(repositories)
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )
    model = service.build_model()
    assert model.scenario is not None

    result = service.apply_edits(
        world_data.WorldDataEdits(
            scenario=world_data.WorldDataScenarioEdit(
                title=model.scenario.title,
                premise=model.scenario.premise,
                player_role=model.scenario.player_role,
                content_sections=model.scenario.content_sections,
            ),
            world_state=(
                world_data.WorldDataStateRow(
                    row_id=ids["state"],
                    key="scene.location",
                    value_json="{not valid json",
                    category="scene",
                    confidence=0.9,
                    source_message_id=ids["message"],
                    archived=True,
                    original_key="scene.location",
                ),
            ),
        )
    )

    assert result.state_archive_count == 1
    assert repositories.list_world_state(save_id) == []
    archived_row = repositories.connection.execute(
        "SELECT value_json, archived_at FROM world_state WHERE id = ?",
        (ids["state"],),
    ).fetchone()
    assert archived_row is not None
    assert json.loads(archived_row["value_json"]) == {
        "name": "Gatehouse",
        "threat": "ash storm",
    }
    assert archived_row["archived_at"] is not None
    changes = repositories.list_state_changes(save_id)
    assert len(changes) == 1
    assert changes[0].operation == "manual_world_data_edit"
    assert changes[0].state_key == "scene.location"
    assert json.loads(changes[0].before_json or "") == {
        "name": "Gatehouse",
        "threat": "ash storm",
    }
    assert changes[0].after_json is None
    assert changes[0].source_message_id is None


def test_world_data_service_updates_existing_memory_edit(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_world_data_fixture(repositories)
    proxied_repositories = _MethodOnlyRepositoryProxy(repositories)
    assert not hasattr(proxied_repositories.connection, "execute")
    service = world_data.WorldDataService(
        repositories=cast(PersistenceRepositories, proxied_repositories),
        active_save_id=save_id,
    )
    model = service.build_model()
    assert model.scenario is not None

    result = service.apply_edits(
        world_data.WorldDataEdits(
            scenario=world_data.WorldDataScenarioEdit(
                title=model.scenario.title,
                premise=model.scenario.premise,
                player_role=model.scenario.player_role,
                content_sections=model.scenario.content_sections,
            ),
            memories=(
                world_data.WorldDataMemoryRow(
                    memory_id=ids["memory"],
                    body="Captain Ilyra now holds the beacon lens.",
                    tags_text="npc, beacon, revised",
                    importance=0.95,
                    source_message_id=ids["message"],
                    archived=False,
                ),
            ),
        )
    )

    assert result.memory_archive_count == 0
    assert _error_text(result) == ""
    memories = repositories.list_memories(save_id)
    assert len(memories) == 1
    assert memories[0].id == ids["memory"]
    assert memories[0].body == "Captain Ilyra now holds the beacon lens."
    assert memories[0].tags == ["npc", "beacon", "revised"]
    assert memories[0].importance == pytest.approx(0.95)


def test_world_data_service_rejects_renaming_active_state_to_existing_active_key(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_world_data_fixture(repositories)
    repositories.upsert_world_state(
        save_id=save_id,
        key="scene.weather",
        value={"condition": "ash fall"},
        category="scene",
        confidence=0.6,
        source_message_id=ids["message"],
    )
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )
    model = service.build_model()
    assert model.scenario is not None
    rows_by_key = {row.key: row for row in model.world_state}
    location = rows_by_key["scene.location"]
    weather = rows_by_key["scene.weather"]

    edits = world_data.WorldDataEdits(
        scenario=world_data.WorldDataScenarioEdit(
            title=model.scenario.title,
            premise=model.scenario.premise,
            player_role=model.scenario.player_role,
            content_sections=model.scenario.content_sections,
        ),
        world_state=(
            world_data.WorldDataStateRow(
                row_id=location.row_id,
                key="scene.weather",
                value_json=location.value_json,
                category=location.category,
                confidence=location.confidence,
                source_message_id=location.source_message_id,
                archived=False,
                original_key="scene.location",
            ),
            weather,
        ),
        memories=(),
        summaries=(),
    )

    with pytest.raises(ValueError, match="existing|duplicate|already exists"):
        service.apply_edits(edits)

    assert {
        item.key: item.value for item in repositories.list_world_state(save_id)
    } == {
        "scene.location": {"name": "Gatehouse", "threat": "ash storm"},
        "scene.weather": {"condition": "ash fall"},
    }


def test_world_data_service_rejects_world_state_row_from_different_save(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    active_save_id, _active_ids = _persist_world_data_fixture(repositories)
    other_save_id, other_ids = _persist_world_data_fixture(repositories)
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=active_save_id,
    )
    model = service.build_model()
    assert model.scenario is not None

    with pytest.raises(ValueError, match="World-state edit.*active save"):
        service.apply_edits(
            world_data.WorldDataEdits(
                scenario=world_data.WorldDataScenarioEdit(
                    title=model.scenario.title,
                    premise=model.scenario.premise,
                    player_role=model.scenario.player_role,
                    content_sections=model.scenario.content_sections,
                ),
                world_state=(
                    world_data.WorldDataStateRow(
                        row_id=other_ids["state"],
                        key="scene.location",
                        value_json='{"name":"Foreign mutation"}',
                        category="scene",
                        confidence=0.1,
                        source_message_id=other_ids["message"],
                        archived=False,
                        original_key="scene.location",
                    ),
                ),
            )
        )

    assert {
        item.key: item.value for item in repositories.list_world_state(active_save_id)
    } == {"scene.location": {"name": "Gatehouse", "threat": "ash storm"}}
    assert {
        item.key: item.value for item in repositories.list_world_state(other_save_id)
    } == {"scene.location": {"name": "Gatehouse", "threat": "ash storm"}}


def test_world_data_service_rejects_world_state_source_message_from_different_save(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    active_save_id, active_ids = _persist_world_data_fixture(repositories)
    other_save_id, other_ids = _persist_world_data_fixture(repositories)
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=active_save_id,
    )
    model = service.build_model()
    assert model.scenario is not None

    with pytest.raises(ValueError, match="source message.*active save"):
        service.apply_edits(
            world_data.WorldDataEdits(
                scenario=world_data.WorldDataScenarioEdit(
                    title=model.scenario.title,
                    premise=model.scenario.premise,
                    player_role=model.scenario.player_role,
                    content_sections=model.scenario.content_sections,
                ),
                world_state=(
                    world_data.WorldDataStateRow(
                        row_id=active_ids["state"],
                        key="scene.location",
                        value_json='{"name":"Illicit foreign source"}',
                        category="scene",
                        confidence=0.2,
                        source_message_id=other_ids["message"],
                        archived=False,
                        original_key="scene.location",
                    ),
                ),
            )
        )

    assert {
        item.key: item.value for item in repositories.list_world_state(active_save_id)
    } == {"scene.location": {"name": "Gatehouse", "threat": "ash storm"}}
    assert {
        item.key: item.value for item in repositories.list_world_state(other_save_id)
    } == {"scene.location": {"name": "Gatehouse", "threat": "ash storm"}}


def test_world_data_service_applies_edits_and_hides_archived_context_inputs(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_world_data_fixture(repositories)
    archived_state = repositories.upsert_world_state(
        save_id=save_id,
        key="clue.bell",
        value={"sound": "archived bell"},
        category="clue",
    )
    archived_memory = repositories.add_memory(
        save_id=save_id,
        body="The archived bell memory should not enter narrator context.",
        tags=["bell"],
    )
    archived_summary = repositories.add_summary(
        save_id=save_id,
        covers_message_start_id=ids["message"],
        covers_message_end_id=ids["message"],
        body="The archived bell summary should not enter narrator context.",
        provider="fake",
        model="fake-summary",
    )
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )

    result = service.apply_edits(
        edits=world_data.WorldDataEdits(
            scenario=world_data.WorldDataScenarioEdit(
                title="Ashfall Spire",
                premise="The storm has reached the upper beacon.",
                player_character_name="Mara Voss",
                player_role="Beacon keeper",
                content_sections=(
                    ("opening_message", "The lens turns red."),
                    ("tone_genre", "siege mystery"),
                ),
            ),
            world_state=(
                world_data.WorldDataStateRow(
                    row_id=ids["state"],
                    key="scene.location",
                    value_json='{"name":"Beacon tower","threat":"critical"}',
                    category="scene",
                    confidence=0.95,
                    source_message_id=ids["message"],
                    archived=False,
                    original_key="scene.location",
                ),
                world_data.WorldDataStateRow(
                    row_id=archived_state.id,
                    key="clue.bell",
                    value_json='{"sound":"archived bell"}',
                    category="clue",
                    confidence=1.0,
                    source_message_id=None,
                    archived=True,
                    original_key="clue.bell",
                ),
            ),
            memories=(
                world_data.WorldDataMemoryRow(
                    memory_id=ids["memory"],
                    body="Captain Ilyra abandoned the east stair.",
                    tags_text="npc, broken-promise",
                    importance=0.9,
                    source_message_id=ids["message"],
                    archived=False,
                ),
                world_data.WorldDataMemoryRow(
                    memory_id=archived_memory.id,
                    body="The archived bell memory should not enter narrator context.",
                    tags_text="bell",
                    importance=1.0,
                    source_message_id=None,
                    archived=True,
                ),
            ),
            summaries=(
                world_data.WorldDataSummaryRow(
                    summary_id=ids["summary"],
                    body="The watch moved to the beacon tower after the lens cracked.",
                    provider="fake",
                    model="fake-summary",
                    covers_message_start_id=ids["message"],
                    covers_message_end_id=ids["message"],
                    archived=False,
                ),
                world_data.WorldDataSummaryRow(
                    summary_id=archived_summary.id,
                    body="The archived bell summary should not enter narrator context.",
                    provider="fake",
                    model="fake-summary",
                    covers_message_start_id=ids["message"],
                    covers_message_end_id=ids["message"],
                    archived=True,
                ),
            ),
        ),
    )

    save = repositories.get_save(save_id)
    assert save is not None
    scenario = repositories.get_scenario(save.scenario_id)
    assert scenario is not None
    assert scenario.title == "Ashfall Spire"
    assert scenario.premise == "The storm has reached the upper beacon."
    assert scenario.player_role == "Beacon keeper"
    assert json.loads(scenario.content_json) == {
        "title": "Ashfall Spire",
        "premise": "The storm has reached the upper beacon.",
        "player_character_name": "Mara Voss",
        "player_role": "Beacon keeper",
        "opening_message": "The lens turns red.",
        "tone_genre": "siege mystery",
        "_source": {
            "content_rating": "unclassified",
            "section_content_ratings": {
                "title": "unclassified",
                "premise": "unclassified",
                "player_character_name": "unclassified",
                "player_role": "unclassified",
                "opening_message": "unclassified",
                "tone_genre": "unclassified",
            },
        },
    }
    assert _error_text(result) == ""

    world_state_items = [
        (item.key, item.value) for item in repositories.list_world_state(save_id)
    ]
    assert world_state_items == [
        ("scene.location", {"name": "Beacon tower", "threat": "critical"}),
    ]
    assert [item.body for item in repositories.list_memories(save_id)] == [
        "Captain Ilyra abandoned the east stair.",
    ]
    assert [item.body for item in repositories.list_summaries(save_id)] == [
        "The watch moved to the beacon tower after the lens cracked.",
    ]

    for index in range(RECENT_MESSAGE_CANDIDATE_LIMIT):
        repositories.append_message(
            save_id=save_id,
            role="narrator",
            speaker_name="Narrator",
            body=f"Beacon tower interim beat {index}.",
        )
    player_message = repositories.append_message(
        save_id=save_id,
        role="player",
        speaker_name="Mara",
        body=(
            "I ignore the archived bell and ask why Ilyra left the east stair."
        ),
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=["structured_output"],
    )
    provider = RecordingWorldDataContextProvider()
    context_result = asyncio.run(
        ContextSearchService(
            repositories=repositories,
            providers={"fake": provider},
        ).search(save_id=save_id, player_message_id=player_message.id)
    )

    assert provider.chat_requests == []
    assert len(provider.structured_output_requests) == 1
    selected_ids = {
        item.source_id
        for bucket in (
            context_result.selected_state,
            context_result.selected_memories,
            context_result.selected_summaries,
        )
        for item in bucket
    }
    assert archived_state.id not in selected_ids
    assert archived_memory.id not in selected_ids
    assert archived_summary.id not in selected_ids


def test_world_data_service_forks_shared_scenario_before_applying_save_edits(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Shared Keep",
        premise="Two saves begin from the same storm-bound keep.",
        player_role="Signal warden",
        content={
            "title": "Shared Keep",
            "premise": "Two saves begin from the same storm-bound keep.",
            "player_role": "Signal warden",
            "starting_scene": "The storm climbs the outer wall.",
            "_source": {
                "origin": "ai_draft",
                "generation_prompt": "A shared keep in an ash storm.",
            },
        },
    )
    edited_save = repositories.create_save(
        scenario_id=scenario.id,
        title="Edited branch",
    )
    untouched_save = repositories.create_save(
        scenario_id=scenario.id,
        title="Untouched branch",
    )
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=edited_save.id,
    )
    model = service.build_model()
    assert model.scenario is not None

    result = service.apply_edits(
        world_data.WorldDataEdits(
            scenario=world_data.WorldDataScenarioEdit(
                title="Forked Keep",
                premise="Only the edited save reaches the beacon tower.",
                player_character_name="Mara Voss",
                player_role="Beacon keeper",
                content_sections=(("opening_message", "The beacon lens turns red."),),
            ),
        )
    )

    edited_after = repositories.get_save(edited_save.id)
    untouched_after = repositories.get_save(untouched_save.id)
    assert edited_after is not None
    assert untouched_after is not None
    assert edited_after.scenario_id != scenario.id
    assert untouched_after.scenario_id == scenario.id

    edited_scenario = repositories.get_scenario(edited_after.scenario_id)
    untouched_scenario = repositories.get_scenario(untouched_after.scenario_id)
    original_scenario = repositories.get_scenario(scenario.id)
    result_scenario = result.model.scenario
    assert edited_scenario is not None
    assert untouched_scenario is not None
    assert original_scenario is not None
    assert result_scenario is not None
    assert result_scenario.player_character_name == "Mara Voss"
    assert result_scenario.content_sections == (
        ("opening_message", "The beacon lens turns red."),
    )
    assert edited_scenario.title == "Forked Keep"
    assert edited_scenario.premise == "Only the edited save reaches the beacon tower."
    assert edited_scenario.player_role == "Beacon keeper"
    assert json.loads(edited_scenario.content_json)["player_character_name"] == (
        "Mara Voss"
    )
    assert json.loads(edited_scenario.content_json)["opening_message"] == (
        "The beacon lens turns red."
    )
    assert json.loads(edited_scenario.content_json)["_source"] == {
        "origin": "ai_draft",
        "generation_prompt": "A shared keep in an ash storm.",
        "content_rating": "unclassified",
        "section_content_ratings": {
            "title": "unclassified",
            "premise": "unclassified",
            "player_character_name": "unclassified",
            "player_role": "unclassified",
            "opening_message": "unclassified",
        },
    }
    assert result_scenario.generation_prompt == "A shared keep in an ash storm."
    assert untouched_scenario.title == "Shared Keep"
    assert original_scenario.title == "Shared Keep"
    assert json.loads(original_scenario.content_json)["starting_scene"] == (
        "The storm climbs the outer wall."
    )


def test_world_data_service_redacts_scenario_definition_above_viewer_rating(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Restricted Keep",
        premise="Restricted premise.",
        player_role="Signal warden",
        content={
            "opening_message": "Restricted opening.",
            "_source": {"content_rating": "r"},
        },
    )

    model = world_data.WorldDataService(
        repositories,
        allowed_content_rating="pg",
    ).build_scenario_definition_model(scenario.id)

    assert model.scenario is not None
    assert model.scenario.title == CONTENT_FILTER_TRANSITION
    assert model.scenario.premise == CONTENT_FILTER_TRANSITION
    assert model.scenario.content_sections == ()


def test_world_data_service_filters_derived_records_by_source_message_rating(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Shared Keep",
        premise="A shared keep.",
        player_role="Warden",
        content={"_source": {"content_rating": "g"}},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Watch")
    safe_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="A lantern glows.",
        content_rating="g",
    )
    restricted_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="Restricted source material.",
        content_rating="r",
    )
    for label, message in (
        ("safe", safe_message),
        ("restricted", restricted_message),
    ):
        repositories.upsert_world_state(
            save_id=save.id,
            key=f"scene.{label}",
            value={"detail": label},
            category="scene",
            source_message_id=message.id,
        )
        repositories.add_memory(
            save_id=save.id,
            body=f"{label} memory",
            tags=[label],
            source_message_ids=[message.id],
        )
        repositories.add_summary(
            save_id=save.id,
            covers_message_start_id=message.id,
            covers_message_end_id=message.id,
            body=f"{label} summary",
            provider="fake",
            model="fake-summary",
            content_rating="g" if label == "safe" else "r",
        )
        repositories.upsert_context_source(
            save_id=save.id,
            source_type="message",
            source_id=message.id,
            title=f"{label} context",
            body=f"{label} context body",
            metadata={"source_message_ids": [message.id]},
        )
        repositories.add_character(
            save_id=save.id,
            name=f"{label.title()} Character",
            source_message_id=message.id,
        )
        repositories.add_active_thread(
            save_id=save.id,
            title=f"{label} thread",
            source_message_id=message.id,
        )
    repositories.add_memory(
        save_id=save.id,
        body="legacy memory without rating provenance",
        tags=["legacy"],
    )

    model = world_data.WorldDataService(
        repositories,
        active_save_id=save.id,
        allowed_content_rating="pg",
    ).build_model()

    assert [row.key for row in model.world_state] == ["scene.safe"]
    assert [row.body for row in model.memories] == ["safe memory"]
    assert [row.body for row in model.summaries] == ["safe summary"]
    assert [row.body for row in model.context_inputs] == ["safe context body"]
    assert [row.name for row in model.characters] == ["Safe Character"]
    assert [row.title for row in model.threads] == ["safe thread"]


def test_world_data_service_applies_shared_scenario_definition_edit_in_place(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Shared Keep",
        premise="Two saves begin from the same storm-bound keep.",
        player_role="Signal warden",
        content={
            "title": "Shared Keep",
            "premise": "Two saves begin from the same storm-bound keep.",
            "player_character_name": "Mara Voss",
            "player_role": "Signal warden",
            "starting_scene": "The storm climbs the outer wall.",
            "tone_genre": "ashfall mystery",
            "_source": {
                "origin": "ai_draft",
                "generation_prompt": "A shared storm keep prompt.",
            },
        },
    )
    first_save = repositories.create_save(
        scenario_id=scenario.id,
        title="First branch",
    )
    second_save = repositories.create_save(
        scenario_id=scenario.id,
        title="Second branch",
    )
    first_message = repositories.append_message(
        save_id=first_save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Ash scratches the glass in the first branch.",
    )
    second_message = repositories.append_message(
        save_id=second_save.id,
        role="player",
        speaker_name="Mara",
        body="I keep the second branch watch.",
    )
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=first_save.id,
    )

    result = service.apply_scenario_definition_edit(
        scenario.id,
        world_data.WorldDataScenarioEdit(
            title="Beacon Spire",
            premise="Both saves now share the revised storm premise.",
            player_character_name="Nia Vale",
            player_role="Beacon keeper",
            content_sections=(
                ("opening_message", "The beacon lens turns red."),
                ("tone_genre", "siege mystery"),
            ),
        ),
    )

    first_after = repositories.get_save(first_save.id)
    second_after = repositories.get_save(second_save.id)
    scenario_after = repositories.get_scenario(scenario.id)
    result_scenario = result.model.scenario
    assert first_after is not None
    assert second_after is not None
    assert scenario_after is not None
    assert result_scenario is not None
    assert first_after.scenario_id == scenario.id
    assert second_after.scenario_id == scenario.id
    assert result.linked_save_count == 2
    assert scenario_after.title == "Beacon Spire"
    assert scenario_after.premise == "Both saves now share the revised storm premise."
    assert scenario_after.player_role == "Beacon keeper"
    assert json.loads(scenario_after.content_json) == {
        "opening_message": "The beacon lens turns red.",
        "tone_genre": "siege mystery",
        "title": "Beacon Spire",
        "premise": "Both saves now share the revised storm premise.",
        "player_character_name": "Nia Vale",
        "player_role": "Beacon keeper",
        "_source": {
            "origin": "ai_draft",
            "generation_prompt": "A shared storm keep prompt.",
            "content_rating": "unclassified",
            "section_content_ratings": {
                "title": "unclassified",
                "premise": "unclassified",
                "player_character_name": "unclassified",
                "player_role": "unclassified",
                "opening_message": "unclassified",
                "tone_genre": "unclassified",
            },
        },
    }
    assert result_scenario.scenario_id == scenario.id
    assert result_scenario.title == "Beacon Spire"
    assert result_scenario.premise == "Both saves now share the revised storm premise."
    assert result_scenario.player_character_name == "Nia Vale"
    assert result_scenario.player_role == "Beacon keeper"
    assert result_scenario.generation_prompt == "A shared storm keep prompt."
    assert result_scenario.content_sections == (
        ("opening_message", "The beacon lens turns red."),
        ("tone_genre", "siege mystery"),
    )
    assert repositories.list_messages(first_save.id) == [first_message]
    assert repositories.list_messages(second_save.id) == [second_message]


def test_world_data_service_scenario_definition_edit_preserves_starter_reference(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Shared Beacon",
        premise="A failing signal tower.",
        player_role="Signal warden",
        content={
            "title": "Shared Beacon",
            "premise": "A failing signal tower.",
            "player_character_name": "Mara Voss",
            "player_role": "Signal warden",
            "tone_genre": "ashfall mystery",
            "character_starters": [
                {
                    "starter_id": "starter-ilyra",
                    "name": "Captain Ilyra",
                    "role": "Watch captain",
                    "reference_image": {
                        "id": "starter-ref-ilyra",
                        "path": "scenario-starters/scenario-1/ilyra.png",
                        "thumbnail_path": (
                            "scenario-starters/scenario-1/thumbnails/ilyra.png"
                        ),
                        "mime_type": "image/png",
                        "prompt_preview": "Uploaded character reference image",
                        "source": "uploaded",
                    },
                }
            ],
        },
    )
    service = world_data.WorldDataService(repositories=repositories)

    result = service.apply_scenario_definition_edit(
        scenario.id,
        world_data.ScenarioEdit(
            title="Shared Beacon Revised",
            premise="A retuned signal tower.",
            player_character_name="Mara Voss",
            player_role="Beacon keeper",
            content={"tone_genre": "siege mystery"},
            character_starters=(
                world_data.ScenarioCharacterStarter(
                    name="Captain Ilyra",
                    role="Gate captain",
                ),
            ),
        ),
    )

    scenario_after = repositories.get_scenario(scenario.id)
    assert scenario_after is not None
    content = json.loads(scenario_after.content_json)
    starter = content["character_starters"][0]
    assert starter["starter_id"] == "starter-ilyra"
    assert starter["role"] == "Gate captain"
    assert starter["reference_image"]["id"] == "starter-ref-ilyra"
    assert result.model.scenario is not None
    assert result.model.scenario.character_starters[0].reference_image is not None


def test_world_data_service_scenario_definition_edit_preserves_overrides(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    original_opening = "The old beacon wakes before Mara reaches the stair."
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Shared Beacon",
        premise="Two saves begin from the same failing signal tower.",
        player_role="Signal warden",
        content={
            "title": "Shared Beacon",
            "premise": "Two saves begin from the same failing signal tower.",
            "player_character_name": "Mara Voss",
            "player_role": "Signal warden",
            "starting_scene": "The tower lens clicks in the dark.",
            "opening_message": original_opening,
            "tone_genre": "ashfall mystery",
        },
    )
    updated_save = repositories.create_save(
        scenario_id=scenario.id,
        title="Save with evolved premise",
    )
    base_save = repositories.create_save(
        scenario_id=scenario.id,
        title="Save still on base premise",
    )
    evolved_update = repositories.add_save_scenario_update(
        save_id=updated_save.id,
        title="Evolved Beacon",
        premise="This save already moved the beacon conflict underground.",
        player_role="Underground signal warden",
        content={
            "title": "Evolved Beacon",
            "premise": "This save already moved the beacon conflict underground.",
            "player_character_name": "Mara Voss",
            "player_role": "Underground signal warden",
            "starting_scene": "The old stairs are sealed behind ash.",
            "opening_message": "The save-level opening has already diverged.",
            "tone_genre": "underground mystery",
        },
        reason="The chronicle established a save-specific branch.",
        provider="fake",
        model="fake-scenario-evolution",
    )
    updated_opening_message = repositories.append_message(
        save_id=updated_save.id,
        role="narrator",
        speaker_name="Narrator",
        body=original_opening,
    )
    base_opening_message = repositories.append_message(
        save_id=base_save.id,
        role="narrator",
        speaker_name="Narrator",
        body=original_opening,
    )
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=updated_save.id,
    )

    result = service.apply_scenario_definition_edit(
        scenario.id,
        world_data.WorldDataScenarioEdit(
            title="Retuned Beacon",
            premise="The shared base scenario now starts at the upper lens.",
            player_character_name="Nia Vale",
            player_role="Beacon keeper",
            content_sections=(
                ("opening_message", "The new base opening stays in scenario content."),
                ("tone_genre", "siege mystery"),
            ),
        ),
    )

    scenario_after = repositories.get_scenario(scenario.id)
    result_scenario = result.model.scenario
    assert scenario_after is not None
    assert result_scenario is not None
    assert result.linked_save_count == 2
    assert scenario_after.title == "Retuned Beacon"
    assert scenario_after.premise == (
        "The shared base scenario now starts at the upper lens."
    )
    assert scenario_after.player_role == "Beacon keeper"
    assert json.loads(scenario_after.content_json) == {
        "opening_message": "The new base opening stays in scenario content.",
        "tone_genre": "siege mystery",
        "title": "Retuned Beacon",
        "premise": "The shared base scenario now starts at the upper lens.",
        "player_character_name": "Nia Vale",
        "player_role": "Beacon keeper",
        "_source": {
            "content_rating": "unclassified",
            "section_content_ratings": {
                "opening_message": "unclassified",
                "tone_genre": "unclassified",
                "title": "unclassified",
                "premise": "unclassified",
                "player_character_name": "unclassified",
                "player_role": "unclassified",
            },
        },
    }
    assert result_scenario.title == "Retuned Beacon"
    assert result_scenario.premise == (
        "The shared base scenario now starts at the upper lens."
    )
    assert result_scenario.player_character_name == "Nia Vale"
    assert result_scenario.player_role == "Beacon keeper"
    assert dict(result_scenario.content_sections) == {
        "opening_message": "The new base opening stays in scenario content.",
        "tone_genre": "siege mystery",
    }

    updates_after = repositories.list_save_scenario_updates(updated_save.id)
    assert len(updates_after) == 1
    update_after = updates_after[0]
    assert update_after.id == evolved_update.id
    assert update_after.title == "Evolved Beacon"
    assert update_after.premise == (
        "This save already moved the beacon conflict underground."
    )
    assert update_after.player_role == "Underground signal warden"
    assert json.loads(update_after.content_json)["opening_message"] == (
        "The save-level opening has already diverged."
    )
    assert update_after.active
    updated_details = repositories.load_save_details(updated_save.id)
    base_details = repositories.load_save_details(base_save.id)
    assert updated_details is not None
    assert base_details is not None
    assert updated_details.scenario.title == "Evolved Beacon"
    assert updated_details.scenario.premise == (
        "This save already moved the beacon conflict underground."
    )
    assert updated_details.scenario.player_role == "Underground signal warden"
    assert json.loads(updated_details.scenario.content_json)["opening_message"] == (
        "The save-level opening has already diverged."
    )
    assert base_details.scenario.title == "Retuned Beacon"
    assert json.loads(base_details.scenario.content_json)["opening_message"] == (
        "The new base opening stays in scenario content."
    )

    assert repositories.list_messages(updated_save.id) == [updated_opening_message]
    assert repositories.list_messages(base_save.id) == [base_opening_message]


def test_world_data_service_rejects_memory_and_summary_edits_from_different_save(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    active_save_id, _active_ids = _persist_world_data_fixture(repositories)
    other_save_id, other_ids = _persist_world_data_fixture(repositories)
    foreign_memory_to_archive = repositories.add_memory(
        save_id=other_save_id,
        body="Foreign archive target must stay visible.",
        tags=["foreign"],
    )
    foreign_summary_to_archive = repositories.add_summary(
        save_id=other_save_id,
        covers_message_start_id=other_ids["message"],
        covers_message_end_id=other_ids["message"],
        body="Foreign summary archive target must stay visible.",
        provider="fake",
        model="fake-summary",
    )
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=active_save_id,
    )
    model = service.build_model()
    assert model.scenario is not None

    with pytest.raises(ValueError, match="save|foreign|active"):
        service.apply_edits(
            world_data.WorldDataEdits(
                scenario=world_data.WorldDataScenarioEdit(
                    title=model.scenario.title,
                    premise=model.scenario.premise,
                    player_role=model.scenario.player_role,
                    content_sections=model.scenario.content_sections,
                ),
                memories=(
                    world_data.WorldDataMemoryRow(
                        memory_id=other_ids["memory"],
                        body="Cross-save memory mutation",
                        tags_text="foreign, mutated",
                        importance=0.1,
                        source_message_id=other_ids["message"],
                        archived=False,
                    ),
                    world_data.WorldDataMemoryRow(
                        memory_id=foreign_memory_to_archive.id,
                        body=foreign_memory_to_archive.body,
                        tags_text=", ".join(foreign_memory_to_archive.tags),
                        importance=foreign_memory_to_archive.importance,
                        source_message_id=None,
                        archived=True,
                    ),
                ),
                summaries=(
                    world_data.WorldDataSummaryRow(
                        summary_id=other_ids["summary"],
                        body="Cross-save summary mutation",
                        provider="fake",
                        model="fake-summary",
                        covers_message_start_id=other_ids["message"],
                        covers_message_end_id=other_ids["message"],
                        archived=False,
                    ),
                    world_data.WorldDataSummaryRow(
                        summary_id=foreign_summary_to_archive.id,
                        body=foreign_summary_to_archive.body,
                        provider=foreign_summary_to_archive.provider,
                        model=foreign_summary_to_archive.model,
                        covers_message_start_id=other_ids["message"],
                        covers_message_end_id=other_ids["message"],
                        archived=True,
                    ),
                ),
            )
        )

    assert [item.body for item in repositories.list_memories(other_save_id)] == [
        "Captain Ilyra promised to hold the east stair.",
        "Foreign archive target must stay visible.",
    ]
    assert [item.body for item in repositories.list_summaries(other_save_id)] == [
        "The watch began as the tower beacon started failing.",
        "Foreign summary archive target must stay visible.",
    ]


def test_world_data_service_rejects_blank_active_summary_edit(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    world_data = _import_world_data_service_without_gtk(monkeypatch)
    save_id, ids = _persist_world_data_fixture(repositories)
    service = world_data.WorldDataService(
        repositories=repositories,
        active_save_id=save_id,
    )
    model = service.build_model()
    assert model.scenario is not None

    with pytest.raises(ValueError, match="Summary|summary|required|blank"):
        service.apply_edits(
            world_data.WorldDataEdits(
                scenario=world_data.WorldDataScenarioEdit(
                    title=model.scenario.title,
                    premise=model.scenario.premise,
                    player_role=model.scenario.player_role,
                    content_sections=model.scenario.content_sections,
                ),
                summaries=(
                    world_data.WorldDataSummaryRow(
                        summary_id=ids["summary"],
                        body=" \n\t ",
                        provider="fake",
                        model="fake-summary",
                        covers_message_start_id=ids["message"],
                        covers_message_end_id=ids["message"],
                        archived=False,
                    ),
                ),
            )
        )

    assert [item.body for item in repositories.list_summaries(save_id)] == [
        "The watch began as the tower beacon started failing.",
    ]


def _import_world_data_service_without_gtk(monkeypatch: MonkeyPatch) -> Any:
    original_import = builtins.__import__

    def import_without_gtk(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "gi" or name.startswith("gi."):
            raise AssertionError(
                "bragi.services.world_data_service must not import GTK/PyGObject"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_gtk)
    sys.modules.pop("bragi.services.world_data_service", None)
    return importlib.import_module("bragi.services.world_data_service")


def _persist_world_data_fixture(
    repositories: PersistenceRepositories,
) -> tuple[str, dict[str, str]]:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "title": "Stale embedded title",
            "premise": "Stale embedded premise",
            "player_character_name": "Mara Voss",
            "player_role": "Stale embedded role",
            "starting_scene": "The beacon gutters in the tower.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Ash scratches the glass as the stair shakes.",
    )
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="scene.location",
        value={"name": "Gatehouse", "threat": "ash storm"},
        category="scene",
        confidence=0.8,
        source_message_id=message.id,
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="Captain Ilyra promised to hold the east stair.",
        tags=["npc", "promise"],
        importance=0.7,
        source_message_id=message.id,
    )
    summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=message.id,
        covers_message_end_id=message.id,
        body="The watch began as the tower beacon started failing.",
        provider="fake",
        model="fake-summary",
        source_message_ids=(message.id,),
    )
    return save.id, {
        "message": message.id,
        "state": state.id,
        "memory": memory.id,
        "summary": summary.id,
    }


def _persist_normalized_world_data_fixture(
    repositories: PersistenceRepositories,
) -> tuple[str, dict[str, str]]:
    save_id, ids = _persist_world_data_fixture(repositories)
    location = repositories.add_location(
        save_id=save_id,
        name="Beacon Gallery",
        aliases=["red lens room"],
        description="The beacon lens overlooks the ash gate.",
        visual_description="Red glass and brass gears under drifting ash.",
        status="unstable",
        hazards=["cracked lens"],
        source_message_id=ids["message"],
        locked_fields=["name"],
    )
    character = repositories.add_character(
        save_id=save_id,
        name="Captain Ilyra",
        aliases=["captain"],
        role="Watch captain",
        known_state="holding the stair",
        met=True,
        appearance="Ash-stained officer coat",
        visual_notes="Silver braid dulled by soot",
        personality="direct",
        voice="clipped",
        texting_style="Crisp one-line replies, no emoji.",
        relationships={"warden": "trusts under pressure"},
        goals="Keep the beacon lit.",
        motivations="Protect the village.",
        current_intent="Guard the stair.",
        boundaries="Will not leave the gallery.",
        attitude_toward_player="Cautiously allied.",
        cooperation_conditions="Helps after proof the lens can hold.",
        status="present",
        location_id=location.id,
        private_notes="Hides a cracked compass.",
        source_message_id=ids["message"],
        locked_fields=["role"],
    )
    snapshot = repositories.upsert_scene_snapshot(
        save_id=save_id,
        current_location_id=location.id,
        situation="The red lens ticks under stress.",
        objective="Keep the beacon alive.",
        in_world_time="midnight",
        weather="ash storm",
        mood="urgent",
        nearby_objects=["lens crank"],
        hazards=["falling glass"],
        present_character_ids=[character.id],
        source_message_id=ids["message"],
        locked_fields=["weather"],
    )
    thread = repositories.add_active_thread(
        save_id=save_id,
        title="Repair the beacon",
        description="The lens has to survive the storm.",
        status="active",
        priority=5,
        visibility="public",
        related_entities=[f"location:{location.id}"],
        source_message_id=ids["message"],
        locked_fields=["status"],
    )
    link = repositories.add_entity_link(
        save_id=save_id,
        entity_type="location",
        entity_id=location.id,
        target_type="memory",
        target_id=ids["memory"],
        relation="established_by",
    )
    suggestion = repositories.add_context_update_suggestion(
        save_id=save_id,
        update_type="field_update",
        entity_type="location",
        entity_id=location.id,
        field_path="description",
        proposed_value="The gallery glass burns with a fresh red warning.",
        reason="Narration moved the lens state forward.",
        confidence=0.8,
        source_message_ids=[ids["message"]],
    )
    audit = repositories.add_context_update_audit(
        save_id=save_id,
        suggestion_id=suggestion.id,
        operation="context_update_detected",
        entity_type="location",
        entity_id=location.id,
        field_path="description",
        before="The beacon lens overlooks the ash gate.",
        after="The gallery glass burns with a fresh red warning.",
        reason="Narration moved the lens state forward.",
        confidence=0.8,
        source_message_ids=[ids["message"]],
    )
    return save_id, {
        **ids,
        "snapshot": snapshot.id,
        "location": location.id,
        "character": character.id,
        "thread": thread.id,
        "link": link.id,
        "suggestion": suggestion.id,
        "audit": audit.id,
    }


def _scenario_edit_from_model(world_data: Any, model: Any) -> Any:
    assert model.scenario is not None
    return world_data.WorldDataScenarioEdit(
        title=model.scenario.title,
        premise=model.scenario.premise,
        player_character_name=model.scenario.player_character_name,
        player_role=model.scenario.player_role,
        content_sections=model.scenario.content_sections,
        character_starters=model.scenario.character_starters,
    )


class _MethodOnlyRepositoryProxy:
    def __init__(self, repositories: PersistenceRepositories) -> None:
        self._repositories = repositories

    @property
    def connection(self) -> object:
        def connection_wrapper() -> sqlite3.Connection:
            return self._repositories.connection

        return connection_wrapper

    def __getattr__(self, name: str) -> object:
        value = getattr(self._repositories, name)
        if not callable(value):
            return value

        def delegated_method(*args: object, **kwargs: object) -> object:
            return value(*args, **kwargs)

        return delegated_method


def _items_by_id(items: Iterable[object]) -> dict[str, object]:
    return {
        _value(item, "id", "row_id", "state_id", "memory_id", "summary_id"): item
        for item in items
    }


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


def _error_text(model: object) -> str:
    error = _value(model, "error", default="")
    return "" if error is None else str(error)
