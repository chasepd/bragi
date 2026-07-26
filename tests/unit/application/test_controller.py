from __future__ import annotations

import asyncio
import builtins
import importlib
import json
import re
import shutil
import sqlite3
import sys
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pytest import MonkeyPatch

from bragi.persistence.models import SaveRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatRequest,
    ChatResponse,
    ImageRequest,
    ImageResponse,
    ProviderCapability,
    ProviderConfigStatus,
    ProviderModel,
    ProviderRetryProgress,
    ProviderToolCall,
    StructuredOutputRequest,
    StructuredOutputResponse,
    ToolCallRequest,
    ToolCallResponse,
    VideoRequest,
    VideoResponse,
)
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.services.character_action_planning_service import (
    CHARACTER_ACTION_PLANNING_TASK,
)
from bragi.services.chat_history_settings import (
    narrator_planner_chat_history_window_settings,
)
from bragi.services.context_search_service import ContextSearchResult
from bragi.services.continuation_scenario_service import CONTINUATION_SECTION_IDS
from bragi.services.model_preferences import scenario_generation_section_model_task
from bragi.services.sexual_content_safety import CONTENT_FILTER_TRANSITION

_MISSING = object()


class RuntimeFakeProvider:
    provider_name = "fake"

    def __init__(self) -> None:
        self.chat_requests: list[ChatRequest] = []
        self.image_requests: list[ImageRequest] = []
        self.scenario_sections: dict[str, str] = {
            "title": "Glass Harbor",
            "premise": "A drowned harbor rings its bell at low tide.",
            "player_character_name": "Mara Voss",
            "player_role": "Harbor warden",
            "tone_genre": "Maritime mystery.",
            "opening_message": "The harbor bell rings from beneath the mud.",
        }

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
                model_id="fake-chat",
                display_name="Fake Chat",
                capabilities=frozenset({ProviderCapability.CHAT}),
                context_window=8192,
            ),
            ProviderModel(
                provider=self.provider_name,
                model_id="fake-image",
                display_name="Fake Image",
                capabilities=frozenset({ProviderCapability.IMAGE_GENERATION}),
            ),
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        first_message = request.messages[0].body
        if "image" in first_message.casefold() and "prompt" in first_message.casefold():
            body = "cinematic drafted image prompt"
        elif first_message.startswith("You are helping draft"):
            section_id = _requested_scenario_section(request.messages[-1].body)
            body = self.scenario_sections[section_id]
        else:
            body = f"fake narrator: {request.messages[-1].body}"
        return ChatResponse(
            body=body,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 21},
        )

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        self.image_requests.append(request)
        return ImageResponse(
            provider=request.provider,
            model_id=request.model_id,
            image_bytes=b"runtime fake scene image",
        )


class RuntimeAnimationProvider(RuntimeFakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.video_requests: list[VideoRequest] = []

    async def generate_video(self, request: VideoRequest) -> VideoResponse:
        self.video_requests.append(request)
        return VideoResponse(
            provider=request.provider,
            model_id=request.model_id,
            mime_type="video/mp4",
            video_bytes=b"\x00\x00\x00\x18ftypmp42runtime-video",
        )


class FailingRuntimeChatProvider(RuntimeFakeProvider):
    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        raise RuntimeError("chat backend leaked sk-live-secret")


class RuntimeSummaryProvider(RuntimeFakeProvider):
    def __init__(self, responses: list[str]) -> None:
        super().__init__()
        self.responses = responses

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected summary request")
        return ChatResponse(
            body=self.responses.pop(0),
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 21},
        )


class RuntimeStructuredCleanupProvider(RuntimeFakeProvider):
    def __init__(self, responses: list[dict[str, object]]) -> None:
        super().__init__()
        self.responses = responses
        self.structured_requests: list[StructuredOutputRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        raise AssertionError("context cleanup must not use normal chat")

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_requests.append(request)
        if not self.responses:
            raise AssertionError(
                f"unexpected structured request: {request.schema_name}"
            )
        return StructuredOutputResponse(
            data=self.responses.pop(0),
            provider=request.provider,
            model_id=request.model_id,
        )


class RuntimeDualCharacterEnhancementProvider(RuntimeFakeProvider):
    def __init__(self, response: dict[str, object]) -> None:
        super().__init__()
        self.response = response
        self.structured_requests: list[StructuredOutputRequest] = []
        self.tool_requests: list[ToolCallRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        raise AssertionError("character enhancement must not use chat prose")

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_requests.append(request)
        return StructuredOutputResponse(
            data=dict(self.response),
            provider=request.provider,
            model_id=request.model_id,
        )

    async def generate_tool_calls(self, request: ToolCallRequest) -> ToolCallResponse:
        self.tool_requests.append(request)
        raise AssertionError("structured-capable enhancement model used tool calls")


class RuntimeActionChoiceProvider(RuntimeFakeProvider):
    def __init__(
        self,
        *,
        response: dict[str, object] | None = None,
        error: Exception | None = None,
    ) -> None:
        super().__init__()
        self.response = response or {
            "choices": [
                {"body": "Open the brass atlas."},
                {"body": "Question the librarian."},
                {"body": "Hide the index under your coat."},
                {"body": "Step through the blue shelf-door."},
            ]
        }
        self.error = error
        self.structured_requests: list[StructuredOutputRequest] = []

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_requests.append(request)
        if self.error is not None:
            raise self.error
        return StructuredOutputResponse(
            data=dict(self.response),
            provider=request.provider,
            model_id=request.model_id,
        )


class RuntimeStructuredReconciliationProvider(RuntimeFakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.structured_requests: list[StructuredOutputRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        raise AssertionError("narrator edit reconciliation must not use chat prose")

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_requests.append(request)
        source_message_id = _structured_request_source_message_id(request)
        evidence_quote = _structured_request_source_message_quote(
            request,
            source_message_id,
        )
        if request.schema_name == "state_memory_extraction":
            data: dict[str, object] = {
                "state_changes": [
                    {
                        "operation": "upsert",
                        "key": "scene.corridor",
                        "value": {"status": "stable"},
                        "category": "scene",
                        "confidence": 0.9,
                        "source_message_id": source_message_id,
                        "evidence_quote": evidence_quote,
                        "persistence_scope": "durable",
                    }
                ],
                "memories": [],
            }
        elif request.schema_name == "context_update_extraction":
            data = {
                "scene": {
                    "source_message_id": source_message_id,
                    "situation": "The corridor remains stable.",
                    "reason": "The edited narration corrected the corridor state.",
                    "confidence": 0.9,
                },
                "locations": [],
                "characters": [],
                "active_threads": [],
                "entity_links": [],
            }
        else:
            raise AssertionError(
                f"unexpected structured request: {request.schema_name}"
            )
        return StructuredOutputResponse(
            data=data,
            provider=request.provider,
            model_id=request.model_id,
        )


class RuntimeToolReconciliationProvider(RuntimeFakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.tool_requests: list[ToolCallRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        raise AssertionError("narrator edit reconciliation must not use chat prose")

    async def generate_tool_calls(self, request: ToolCallRequest) -> ToolCallResponse:
        self.tool_requests.append(request)
        source_message_id = _tool_request_source_message_id(request)
        tool_names = {tool.name for tool in request.tools}
        if "patch_world_state" in tool_names:
            calls = (
                ProviderToolCall(
                    id="tool-state-1",
                    name="patch_world_state",
                    arguments_json=json.dumps(
                        {
                            "operation": "upsert",
                            "key": "scene.corridor",
                            "value_patch": {"status": "stable"},
                            "category": "scene",
                            "source_message_id": source_message_id,
                            "evidence_quote": "The corridor holds steady",
                            "confidence": 0.9,
                            "persistence_scope": "durable",
                        }
                    ),
                ),
            )
        elif "update_scene_snapshot" in tool_names:
            calls = (
                ProviderToolCall(
                    id="tool-context-1",
                    name="update_scene_snapshot",
                    arguments_json=json.dumps(
                        {
                            "source_message_id": source_message_id,
                            "evidence_quote": "The corridor holds steady",
                            "situation": "The corridor remains stable.",
                            "reason": (
                                "The edited narration corrected the corridor state."
                            ),
                            "confidence": 0.9,
                        }
                    ),
                ),
            )
        else:
            raise AssertionError(
                f"unexpected tool request: {sorted(tool_names)}"
            )
        return ToolCallResponse(
            tool_calls=calls,
            body="",
            provider=request.provider,
            model_id=request.model_id,
        )


class BlockingRuntimeStructuredReconciliationProvider(
    RuntimeStructuredReconciliationProvider
):
    def __init__(self) -> None:
        super().__init__()
        self.first_request_started = asyncio.Event()
        self.release_first_request = asyncio.Event()
        self._request_count = 0

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self._request_count += 1
        if self._request_count == 1:
            self.first_request_started.set()
            await self.release_first_request.wait()
        return await super().generate_structured_output(request)


def _structured_request_source_message_id(request: StructuredOutputRequest) -> str:
    schema_text = json.dumps(request.schema)
    for message in request.messages:
        if message.role == "user":
            for line in message.body.splitlines():
                match = re.match(r"^- ([^ ]+) \[[^\]]+\]", line)
                if match:
                    return match.group(1)
    match = re.search(r'"source_message_id"[^{}]*"enum":\s*\[\s*"([^"]+)"', schema_text)
    if match:
        return match.group(1)
    raise AssertionError("structured request did not expose a source message id")


def _structured_request_source_message_quote(
    request: StructuredOutputRequest,
    source_message_id: str,
) -> str:
    for message in request.messages:
        if message.role == "user":
            for line in message.body.splitlines():
                match = re.match(
                    rf"^- {re.escape(source_message_id)} \[[^\]]+\] (.*)$",
                    line,
                )
                if match:
                    return match.group(1)
    raise AssertionError("structured request did not expose source message text")


def _tool_request_source_message_id(request: ToolCallRequest) -> str:
    for message in request.messages:
        if message.role == "user":
            for line in message.body.splitlines():
                match = re.match(r"^- ([^ ]+) \[[^\]]+\]", line)
                if match:
                    return match.group(1)
    raise AssertionError("tool request did not expose a source message id")


class FailingRuntimeStructuredProvider(RuntimeFakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.structured_requests: list[StructuredOutputRequest] = []

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_requests.append(request)
        raise RuntimeError("structured backend leaked sk-live-secret")


class FailingRuntimeImageProvider(RuntimeFakeProvider):
    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        self.image_requests.append(request)
        raise RuntimeError("image backend leaked sk-live-secret")


class NoopContextSearch:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def search(
        self,
        *,
        save_id: str,
        player_message_id: str,
    ) -> ContextSearchResult:
        self.calls.append((save_id, player_message_id))
        return ContextSearchResult()


@pytest.fixture
def repositories(
    tmp_path: Path,
    migrated_database_template: Path,
) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    shutil.copy2(migrated_database_template, database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_runtime_model_is_import_safe_and_exposes_active_save_state(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, narrator_id = _persist_runtime_save(repositories)
    repositories.create_media_asset(
        save_id=save_id,
        source_message_id=narrator_id,
        type="image",
        path="media/night-watch/scene.png",
        thumbnail_path=None,
        prompt="The beacon lens catches fire.",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = controller.load_save(save_id)

    saves = list(_value(model, "save_list", "saves"))
    assert [_value(save, "title") for save in saves] == ["Night Watch"]
    assert _value(saves[0], "active") is True
    assert _value(saves[0], "id", "save_id") == save_id
    assert _value(saves[0], "scenario_title") == "Ashfall Keep"
    assert _value(saves[0], "created_at")
    assert _value(saves[0], "updated_at")
    assert _value(saves[0], "last_opened_at")
    assert _value(model, "active_save_id") == save_id
    assert _value(model, "active_save_title", "save_title") == "Night Watch"
    assert _value(model, "active_scenario_title", "scenario_title") == "Ashfall Keep"

    messages = list(_chronicle_messages(model))
    assert [_value(message, "role") for message in messages] == [
        "player",
        "narrator",
    ]
    assert [_value(message, "role_label") for message in messages] == [
        "Mara",
        "Narrator",
    ]
    assert [_value(message, "body") for message in messages] == [
        "I climb toward the beacon lens.",
        "Ash scratches the glass as the stair shakes.",
    ]

    latest_image = _latest_image(model)
    assert _value(latest_image, "path") == "media/night-watch/scene.png"
    assert _value(latest_image, "source_message_id") == narrator_id
    assert "fake-chat" in _value(model, "model_indicator")
    assert isinstance(_status_text(model), str)


def test_runtime_model_exposes_latest_action_choices(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Library of Falling Doors",
        premise="Every shelf is a door.",
        player_role="Courier",
        content={
            "action_choices_enabled": True,
            "choice_style": "Four concrete choices.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Library")
    narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The blue shelf opens.",
    )
    repositories.replace_message_action_choices(
        save_id=save.id,
        message_id=narrator.id,
        choices=(
            "Open the atlas",
            "Question the librarian",
            "Hide the index",
            "Step through",
        ),
        provider="fake",
        model="choice-model",
    )
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = controller.load_save(save.id)
    choices = _value(model, "action_choices")

    assert _value(model, "action_choices_enabled") is True
    assert _value(choices, "narrator_message_id") == narrator.id
    assert [_value(choice, "body") for choice in _value(choices, "choices")] == [
        "Open the atlas",
        "Question the librarian",
        "Hide the index",
        "Step through",
    ]


def test_manual_action_choice_scenario_generates_opening_action_choices(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    provider = RuntimeActionChoiceProvider()
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="fake",
        model_id="fake-chat",
    )
    _save_fake_provider_model(
        repositories,
        model_id="fake-chat",
        capabilities=["chat", "structured_output"],
    )
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        providers={"fake": provider},
    )

    model = controller.create_manual_scenario(
        runtime.ManualScenarioInput(
            scenario_type="full_roleplay",
            action_choices_enabled=True,
            title="Library of Falling Doors",
            premise="Every shelf is a door.",
            player_role="Courier",
            choice_style="Four concrete choices with different risks.",
            opening_message="The blue shelf opens.",
        )
    )

    save_id = _value(model, "active_save_id")
    assert save_id is not None
    details = repositories.load_save_details(save_id)
    assert details is not None
    scenario = details.scenario
    assert scenario.type == "full_roleplay"
    assert json.loads(scenario.content_json)["action_choices_enabled"] is True
    assert [
        choice.body for choice in repositories.latest_message_action_choices(save_id)
    ] == [
        "Open the brass atlas.",
        "Question the librarian.",
        "Hide the index under your coat.",
        "Step through the blue shelf-door.",
    ]
    choices = _value(model, "action_choices")
    assert _value(choices, "narrator_message_id")
    assert [_value(choice, "body") for choice in _value(choices, "choices")] == [
        "Open the brass atlas.",
        "Question the librarian.",
        "Hide the index under your coat.",
        "Step through the blue shelf-door.",
    ]
    assert provider.structured_requests
    assert _value(model, "error") is None


def test_save_action_choice_scenario_draft_returns_opening_action_choices(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    provider = RuntimeActionChoiceProvider()
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="fake",
        model_id="fake-chat",
    )
    _save_fake_provider_model(
        repositories,
        model_id="fake-chat",
        capabilities=["chat", "structured_output"],
    )
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        providers={"fake": provider},
    )

    model = asyncio.run(
        _save_scenario_draft(
            controller,
            scenario_type="full_roleplay",
            sections=_reviewed_cyoa_sections(),
            action_choices_enabled=True,
            save_title="Reviewed Library",
            request_initial_image=False,
        )
    )

    choices = _value(model, "action_choices")
    assert [_value(choice, "body") for choice in _value(choices, "choices")] == [
        "Open the brass atlas.",
        "Question the librarian.",
        "Hide the index under your coat.",
        "Step through the blue shelf-door.",
    ]
    assert provider.structured_requests
    assert _value(model, "error") is None


def test_start_saved_action_choice_scenario_returns_opening_action_choices(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    provider = RuntimeActionChoiceProvider()
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="fake",
        model_id="fake-chat",
    )
    _save_fake_provider_model(
        repositories,
        model_id="fake-chat",
        capabilities=["chat", "structured_output"],
    )
    scenario_content: dict[str, object] = dict(_reviewed_cyoa_sections())
    scenario_content["action_choices_enabled"] = True
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Library of Falling Doors",
        premise="Every shelf is a door.",
        player_role="Courier",
        content=scenario_content,
    )
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        providers={"fake": provider},
    )

    model = controller.start_saved_scenario(
        scenario_id=scenario.id,
        save_title="Second Library",
    )

    choices = _value(model, "action_choices")
    assert [_value(choice, "body") for choice in _value(choices, "choices")] == [
        "Open the brass atlas.",
        "Question the librarian.",
        "Hide the index under your coat.",
        "Step through the blue shelf-door.",
    ]
    assert provider.structured_requests
    assert _value(model, "error") is None


def test_regenerate_action_choices_replaces_latest_options(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    provider = RuntimeActionChoiceProvider(
        response={
            "choices": [
                {"body": "Circle around the mirrored shelves."},
                {"body": "Ask the librarian about the blue door."},
                {"body": "Slip the brass atlas into your satchel."},
                {"body": "Brace the shelf-door before it closes."},
            ]
        }
    )
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="fake",
        model_id="fake-chat",
    )
    _save_fake_provider_model(
        repositories,
        model_id="fake-chat",
        capabilities=["chat", "structured_output"],
    )
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Library of Falling Doors",
        premise="Every shelf is a door.",
        player_role="Courier",
        content={
            "action_choices_enabled": True,
            "choice_style": "Four concrete choices with different risks.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Library")
    narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The blue shelf opens.",
    )
    repositories.replace_message_action_choices(
        save_id=save.id,
        message_id=narrator.id,
        choices=(
            "Old option one.",
            "Old option two.",
            "Old option three.",
            "Old option four.",
        ),
        provider="fake",
        model="old-choice-model",
    )
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        providers={"fake": provider},
    )

    model = asyncio.run(
        controller.regenerate_action_choices(
            narrator_message_id=narrator.id,
            active_save_id=save.id,
        )
    )

    assert _status_text(model) == "Action choices regenerated"
    assert [
        choice.body for choice in repositories.latest_message_action_choices(save.id)
    ] == [
        "Circle around the mirrored shelves.",
        "Ask the librarian about the blue door.",
        "Slip the brass atlas into your satchel.",
        "Brace the shelf-door before it closes.",
    ]
    choices = _value(model, "action_choices")
    assert _value(choices, "narrator_message_id") == narrator.id
    assert [_value(choice, "body") for choice in _value(choices, "choices")] == [
        "Circle around the mirrored shelves.",
        "Ask the librarian about the blue door.",
        "Slip the brass atlas into your satchel.",
        "Brace the shelf-door before it closes.",
    ]
    assert provider.structured_requests
    assert _value(model, "error") is None


def test_manual_action_choice_generation_failure_is_nonfatal(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    provider = RuntimeActionChoiceProvider(error=RuntimeError("choice model down"))
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="fake",
        model_id="fake-chat",
    )
    _save_fake_provider_model(
        repositories,
        model_id="fake-chat",
        capabilities=["chat", "structured_output"],
    )
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        providers={"fake": provider},
    )

    model = controller.create_manual_scenario(
        runtime.ManualScenarioInput(
            scenario_type="full_roleplay",
            action_choices_enabled=True,
            title="Library of Falling Doors",
            premise="Every shelf is a door.",
            player_role="Courier",
            choice_style="Four concrete choices with different risks.",
            opening_message="The blue shelf opens.",
        )
    )

    save_id = _value(model, "active_save_id")
    assert save_id is not None
    assert _value(model, "error") is None
    assert repositories.latest_message_action_choices(save_id) == []


def test_runtime_delete_media_asset_archives_asset_and_refreshes_model(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, narrator_id = _persist_runtime_save(repositories)
    first_asset = repositories.create_media_asset(
        save_id=save_id,
        source_message_id=narrator_id,
        type="image",
        path="media/night-watch/scene-1.png",
        thumbnail_path=None,
        prompt="The beacon gutters.",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    latest_asset = repositories.create_media_asset(
        save_id=save_id,
        source_message_id=narrator_id,
        type="image",
        path="media/night-watch/scene-2.png",
        thumbnail_path=None,
        prompt="The beacon lens catches fire.",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    controller = _runtime_controller(runtime, repositories, tmp_path)
    controller.load_save(save_id)

    model = controller.delete_media_asset(latest_asset.id)

    latest_image = _latest_image(model)
    assert _value(model, "status") == "Media deleted"
    assert _value(latest_image, "id") == first_asset.id
    assert [asset.id for asset in repositories.list_media_assets(save_id)] == [
        first_asset.id
    ]


def test_runtime_set_character_reference_image_refreshes_model(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Oracle of Glass",
        premise="A private audience with the oracle.",
        player_role="A careful petitioner",
        content={"character_name": "Oracle of Glass"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Audience")
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The oracle studies your reflection.",
        provider="fake",
        model="fake-chat",
    )
    character = repositories.add_character(
        save_id=save.id,
        name="Oracle of Glass",
        role="An ancient diviner bound to mirrored halls.",
    )
    current_reference = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=message.id,
        type="image",
        path=f"{save.id}/current-reference.png",
        thumbnail_path=None,
        prompt="current reference",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={"kind": "character_reference", "character_id": character.id},
    )
    candidate = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=message.id,
        type="image",
        path=f"{save.id}/candidate-reference.png",
        thumbnail_path=None,
        prompt="candidate character image",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={"kind": "character_image"},
    )
    media_dir = tmp_path / "media"
    for asset in (current_reference, candidate):
        path = media_dir / asset.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake image bytes")
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=character.id,
        target_type="media_asset",
        target_id=current_reference.id,
        relation="reference_image",
    )
    controller = _runtime_controller(runtime, repositories, tmp_path)
    controller.load_save(save.id)

    model = controller.set_character_reference_image(candidate.id)

    assert _status_text(model) == "Character reference image updated"
    reference = _value(_value(model, "media"), "character_reference_image")
    assert _value(reference, "id") == candidate.id
    links = [
        link
        for link in repositories.list_entity_links(save.id)
        if link.relation == "reference_image"
    ]
    assert sorted(
        (link.entity_type, link.entity_id, link.target_id) for link in links
    ) == sorted(
        [
            ("character", character.id, candidate.id),
        ]
    )


def test_runtime_animate_media_asset_creates_video_with_source_provenance(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, narrator_id = _persist_runtime_save(repositories)
    media_dir = tmp_path / "media"
    source_path = media_dir / "night-watch" / "scene.png"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"runtime source image")
    source_asset = repositories.create_media_asset(
        save_id=save_id,
        source_message_id=narrator_id,
        type="image",
        path="night-watch/scene.png",
        thumbnail_path=None,
        prompt="The beacon lens catches fire.",
        provider="fake",
        model="fake-image",
        status="succeeded",
        mime_type="image/png",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-image-video",
        display_name="Fake Image Video",
        capabilities=[ProviderCapability.IMAGE_PLUS_TEXT_TO_VIDEO.value],
    )
    repositories.set_model_preference(
        task="image_animation",
        provider="fake",
        model_id="fake-image-video",
    )
    provider = RuntimeAnimationProvider()
    controller = _runtime_controller(runtime, repositories, tmp_path, provider=provider)
    controller.load_save(save_id)

    def save_operation_lock(_save_id: str) -> object:
        raise AssertionError("image animation must not hold the save operation lock")

    monkeypatch.setattr(controller, "_save_operation_lock", save_operation_lock)

    model = asyncio.run(
        controller.animate_media_asset(
            source_asset.id,
            motion_prompt="make the beacon flame gutter",
        )
    )

    assert _value(model, "status") == "Image animated"
    assets = repositories.list_media_assets(save_id)
    assert [asset.type for asset in assets] == ["image", "video"]
    video_asset = assets[-1]
    assert video_asset.source_media_asset_id == source_asset.id
    assert video_asset.source_message_id == narrator_id
    assert len(provider.video_requests) == 1
    request = provider.video_requests[0]
    assert request.source_media_asset_id == source_asset.id
    assert request.source_media_path == source_path
    assert "make the beacon flame gutter" in request.prompt


def test_runtime_animate_media_asset_reports_unavailable_provider(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, narrator_id = _persist_runtime_save(repositories)
    repositories.create_media_asset(
        save_id=save_id,
        source_message_id=narrator_id,
        type="image",
        path="night-watch/scene.png",
        thumbnail_path=None,
        prompt="The beacon lens catches fire.",
        provider="fake",
        model="fake-image",
        status="succeeded",
        mime_type="image/png",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-image-video",
        display_name="Fake Image Video",
        capabilities=[ProviderCapability.IMAGE_TO_VIDEO.value],
    )
    repositories.set_model_preference(
        task="image_animation",
        provider="fake",
        model_id="fake-image-video",
    )
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        providers={"other": RuntimeFakeProvider()},
    )
    controller.load_save(save_id)

    model = asyncio.run(
        controller.animate_media_asset("missing-source-does-not-matter")
    )

    assert _value(model, "error") == "Image Animation provider is unavailable: fake"
    assert [asset.type for asset in repositories.list_media_assets(save_id)] == [
        "image"
    ]


def test_runtime_animate_media_asset_allows_missing_catalog_row_for_selected_model(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, narrator_id = _persist_runtime_save(repositories)
    media_dir = tmp_path / "media"
    source_path = media_dir / "night-watch" / "scene.png"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"source image bytes")
    source_asset = repositories.create_media_asset(
        save_id=save_id,
        source_message_id=narrator_id,
        type="image",
        path="night-watch/scene.png",
        thumbnail_path=None,
        prompt="The beacon lens catches fire.",
        provider="fake",
        model="fake-image",
        status="succeeded",
        mime_type="image/png",
    )
    repositories.set_model_preference(
        task="image_animation",
        provider="fake",
        model_id="fake-unsynced-image-video",
    )
    provider = RuntimeAnimationProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
    )
    controller.load_save(save_id)

    model = asyncio.run(controller.animate_media_asset(source_asset.id))

    assert _value(model, "status") == "Image animated"
    assert len(provider.video_requests) == 1
    assert provider.video_requests[0].model_id == "fake-unsynced-image-video"


def test_build_model_exposes_scenario_wizard_flows(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = controller.build_model()

    scenario_wizard = _value(model, "scenario_wizard")
    flows = _flows_by_identifier(_value(scenario_wizard, "flows"))
    assert list(flows) == [
        "full_roleplay",
        "fantasy_roleplay",
        "science_fiction_roleplay",
        "first_contact_exploration",
        "survival_expedition",
        "time_loop",
        "investigation_mystery",
        "heist_infiltration",
        "political_intrigue",
        "settlement_builder",
        "monster_hunt_bounty",
        "road_trip_pilgrimage",
        "merchant_trade_route",
        "dating_sim",
    ]
    assert _value(flows["full_roleplay"], "label") == "Generic Roleplay"
    assert _value(flows["fantasy_roleplay"], "label") == "Fantasy"
    assert _value(flows["science_fiction_roleplay"], "label") == "Science Fiction"
    assert _value(flows["first_contact_exploration"], "label") == (
        "First Contact / Exploration"
    )
    assert _value(flows["survival_expedition"], "label") == "Survival Expedition"
    assert _value(flows["time_loop"], "label") == "Time Loop"
    assert _value(flows["investigation_mystery"], "label") == (
        "Investigation Mystery"
    )
    assert _value(flows["heist_infiltration"], "label") == "Heist / Infiltration"
    assert _value(flows["political_intrigue"], "label") == "Political Intrigue"
    assert _value(flows["settlement_builder"], "label") == "Settlement Builder"
    assert _value(flows["monster_hunt_bounty"], "label") == (
        "Monster Hunt / Bounty"
    )
    assert _value(flows["road_trip_pilgrimage"], "label") == (
        "Road Trip / Pilgrimage"
    )
    assert _value(flows["merchant_trade_route"], "label") == (
        "Merchant / Trade Route"
    )
    assert _value(flows["dating_sim"], "label") == "Dating Sim"
    assert "character_interaction" not in flows
    assert _section_ids(flows["full_roleplay"]) == [
        "title",
        "premise",
        "player_character_name",
        "player_role",
        "tone_genre",
        "opening_message",
    ]
    assert _section_ids(flows["fantasy_roleplay"]) == [
        "title",
        "premise",
        "player_character_name",
        "player_role",
        "magic_system",
        "realms_and_places",
        "factions_and_orders",
        "myths_and_creatures",
        "quest_stakes",
        "tone_genre",
        "opening_message",
    ]
    assert _section_ids(flows["science_fiction_roleplay"]) == [
        "title",
        "premise",
        "player_character_name",
        "player_role",
        "technology_level",
        "setting_scope",
        "species_and_intelligences",
        "factions_and_institutions",
        "mission_stakes",
        "tone_genre",
        "opening_message",
    ]
    assert _section_ids(flows["first_contact_exploration"]) == [
        "title",
        "premise",
        "player_character_name",
        "player_role",
        "mission_profile",
        "ship_or_base_status",
        "exploration_target",
        "unknown_intelligence",
        "knowledge_state",
        "translation_progress",
        "discoveries_and_samples",
        "hazards_and_escalation",
        "tone_genre",
        "opening_message",
    ]
    assert _section_ids(flows["survival_expedition"]) == [
        "title",
        "premise",
        "player_character_name",
        "player_role",
        "expedition_goal",
        "route_options",
        "resource_inventory",
        "environmental_conditions",
        "hazards_and_events",
        "camp_status",
        "travel_progress",
        "tone_genre",
        "opening_message",
    ]
    assert _section_ids(flows["time_loop"]) == [
        "title",
        "premise",
        "player_character_name",
        "player_role",
        "loop_premise",
        "reset_trigger",
        "loop_duration",
        "starting_state",
        "objective",
        "failure_conditions",
        "baseline_world_state",
        "loop_schedule",
        "persistent_knowledge",
        "persistence_exceptions",
        "npc_memory_rules",
        "current_loop_state",
        "tone_genre",
        "opening_message",
    ]
    assert _section_ids(flows["investigation_mystery"]) == [
        "title",
        "premise",
        "player_character_name",
        "player_role",
        "case_facts",
        "clues",
        "timeline",
        "red_herrings",
        "hidden_truth",
        "case_status",
        "tone_genre",
        "opening_message",
    ]
    assert _section_ids(flows["heist_infiltration"]) == [
        "title",
        "premise",
        "player_character_name",
        "player_role",
        "target_location",
        "objectives_and_stakes",
        "intel_and_access",
        "security_model",
        "alert_and_heat",
        "loadout_and_tools",
        "complications",
        "extraction_routes",
        "aftermath",
        "tone_genre",
        "opening_message",
    ]
    assert _section_ids(flows["political_intrigue"]) == [
        "title",
        "premise",
        "player_character_name",
        "player_role",
        "political_arena",
        "political_factions",
        "central_conflict",
        "secrets_and_leverage",
        "reputation_and_standing",
        "obligations_and_favors",
        "alliances_and_rivalries",
        "event_calendar",
        "political_pressure",
        "public_private_knowledge",
        "tone_genre",
        "opening_message",
    ]
    assert _section_ids(flows["settlement_builder"]) == [
        "title",
        "premise",
        "player_character_name",
        "player_role",
        "settlement_profile",
        "resources_and_indicators",
        "projects_and_facilities",
        "threats_and_opportunities",
        "calendar_and_deadlines",
        "tone_genre",
        "opening_message",
    ]
    assert _section_ids(flows["monster_hunt_bounty"]) == [
        "title",
        "premise",
        "player_character_name",
        "player_role",
        "hunt_profile",
        "target_profile",
        "leads_and_clues",
        "hunt_locations",
        "preparation_state",
        "hunt_status",
        "tone_genre",
        "opening_message",
    ]
    assert _section_ids(flows["road_trip_pilgrimage"]) == [
        "title",
        "premise",
        "player_character_name",
        "player_role",
        "journey_profile",
        "route_and_stops",
        "transport_and_supplies",
        "recurring_pressures",
        "relationship_threads",
        "journey_progress",
        "tone_genre",
        "opening_message",
    ]
    assert _section_ids(flows["merchant_trade_route"]) == [
        "title",
        "premise",
        "player_character_name",
        "player_role",
        "trade_profile",
        "cargo_inventory",
        "markets_and_stops",
        "contracts_and_debts",
        "route_hazards",
        "profit_and_loss",
        "tone_genre",
        "opening_message",
    ]
    assert _section_ids(flows["dating_sim"]) == [
        "title",
        "premise",
        "player_character_name",
        "player_character_profile",
        "player_role",
        "tone_genre",
        "opening_message",
    ]


def test_load_save_persists_active_controller_state_by_default(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories)
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = controller.load_save(save_id)

    assert _value(model, "active_save_id") == save_id
    assert controller.active_save_id == save_id


def test_build_model_does_not_adopt_default_save_without_active_selection(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    _persist_runtime_save(repositories)
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = controller.build_model()

    assert _value(model, "active_save_id") is None
    assert controller.active_save_id is None


def test_load_save_can_return_selected_model_without_process_global_selection(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories)
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = controller.load_save(save_id, remember_process_active_save=False)

    assert _value(model, "active_save_id") == save_id
    assert controller.active_save_id is None


def test_rename_save_updates_title_without_switching_active_save(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    first_save_id, _ = _persist_runtime_save(repositories, title="Lantern Keep")
    second_save_id, _ = _persist_runtime_save(
        repositories,
        title="Signal Tower Old",
    )
    controller = _runtime_controller(runtime, repositories, tmp_path)
    controller.load_save(first_save_id)

    model = controller.rename_save(
        save_id=second_save_id,
        title="  Signal Tower  ",
        active_save_id=first_save_id,
    )

    renamed_save = repositories.get_save(second_save_id)
    assert renamed_save is not None
    assert renamed_save.title == "Signal Tower"
    assert _value(model, "active_save_id") == first_save_id
    assert _value(model, "active_save_title") == "Lantern Keep"
    assert controller.active_save_id == first_save_id


def test_runtime_export_and_active_import_run_inside_save_operation_lock(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories)
    controller = _runtime_controller(runtime, repositories, tmp_path)
    controller.load_save(save_id)
    lock_depth = 0
    events: list[tuple[str, str]] = []

    class InstrumentedLock:
        def __init__(self, locked_save_id: str) -> None:
            self.locked_save_id = locked_save_id

        def __enter__(self) -> None:
            nonlocal lock_depth
            assert lock_depth == 0
            lock_depth += 1
            events.append(("enter", self.locked_save_id))

        def __exit__(
            self,
            _exc_type: object,
            _exc: object,
            _traceback: object,
        ) -> None:
            nonlocal lock_depth
            assert lock_depth == 1
            events.append(("exit", self.locked_save_id))
            lock_depth -= 1

    class FakeChatBundleService:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def export_save(
            self,
            requested_save_id: str,
            _bundle_path: Path,
            *,
            include_message_revisions: bool = False,
        ) -> object:
            assert requested_save_id == save_id
            assert include_message_revisions is False
            assert lock_depth == 1
            events.append(("export", requested_save_id))
            return SimpleNamespace(
                title="Night Watch",
                message_count=2,
                media_count=1,
            )

        def import_save(self, _bundle_path: Path) -> object:
            assert lock_depth == 1
            events.append(("import", save_id))
            return SimpleNamespace(
                save_id="imported-save",
                scenario_id="imported-scenario",
                title="Imported Night Watch",
                message_count=2,
                media_count=1,
                skipped_media_count=0,
            )

    def thread_save_operation_lock(requested_save_id: str) -> InstrumentedLock:
        assert requested_save_id == save_id
        events.append(("request", requested_save_id))
        return InstrumentedLock(requested_save_id)

    monkeypatch.setattr(runtime, "ChatBundleService", FakeChatBundleService)
    monkeypatch.setattr(
        controller,
        "_thread_save_operation_lock",
        thread_save_operation_lock,
    )

    export_model = controller.export_active_save(tmp_path / "export.bragi-chat")
    import_model = controller.import_save_bundle(tmp_path / "import.bragi-chat")

    assert _error_text(export_model) == ""
    assert _error_text(import_model) == ""
    assert events == [
        ("request", save_id),
        ("enter", save_id),
        ("export", save_id),
        ("exit", save_id),
        ("request", save_id),
        ("enter", save_id),
        ("import", save_id),
        ("exit", save_id),
    ]


def test_runtime_scenario_bundle_import_export_preserves_active_save(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories)
    save = repositories.get_save(save_id)
    assert save is not None
    controller = _runtime_controller(runtime, repositories, tmp_path)
    controller.load_save(save_id)
    events: list[tuple[str, str]] = []

    class FakeScenarioBundleService:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def preview_import(self, bundle_path: Path) -> object:
            events.append(("preview", bundle_path.name))
            return SimpleNamespace(
                scenario_id="scenario-imported",
                title="Imported Ashfall",
                scenario_type="full_roleplay",
                bundle_version=1,
                exported_at="2026-05-29T00:00:00+00:00",
            )

        def export_scenario(
            self,
            requested_scenario_id: str,
            bundle_path: Path,
        ) -> object:
            events.append(("export", requested_scenario_id))
            assert bundle_path.name == "export.bragi-scenario"
            return SimpleNamespace(
                title="Ashfall Keep",
                scenario_type="full_roleplay",
            )

        def import_scenario(self, bundle_path: Path) -> object:
            events.append(("import", bundle_path.name))
            return SimpleNamespace(
                scenario_id="scenario-imported",
                title="Imported Ashfall",
                scenario_type="full_roleplay",
            )

    monkeypatch.setattr(runtime, "ScenarioBundleService", FakeScenarioBundleService)

    preview = controller.preview_import_scenario_bundle(
        tmp_path / "import.bragi-scenario"
    )
    export_model = controller.export_saved_scenario(
        save.scenario_id,
        tmp_path / "export.bragi-scenario",
    )
    import_model = controller.import_scenario_bundle(
        tmp_path / "import.bragi-scenario"
    )

    assert _value(preview, "title") == "Imported Ashfall"
    assert _error_text(export_model) == ""
    assert _error_text(import_model) == ""
    assert controller.active_save_id == save_id
    assert _value(export_model, "active_save_id") == save_id
    assert _value(import_model, "active_save_id") == save_id
    assert _status_text(export_model) == "Exported scenario: Ashfall Keep"
    assert _status_text(import_model) == "Imported scenario: Imported Ashfall"
    assert events == [
        ("preview", "import.bragi-scenario"),
        ("export", save.scenario_id),
        ("import", "import.bragi-scenario"),
    ]


def test_create_manual_full_roleplay_save_sets_active_and_appends_opening(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)

    fields = {
        "title": "Frostglass Hall",
        "premise": "A sealed hall is thawing after a century.",
        "player_character_name": "Mara Voss",
        "player_role": "Relic hunter",
        "worldbuilding": "Mirrors remember every visitor.",
        "lore": "The hall was frozen by a failed oath.",
        "locations": "Mirror nave, reliquary, thawing vault.",
        "factions": "The Thawing Choir.",
        "tone_genre": "Gothic exploration.",
        "opening_message": "The frost cracks across the mirror floor.",
        "save_title": "First Thaw",
    }

    model = _create_manual_full_roleplay(runtime, controller, fields)

    save = repositories.list_saves()[0]
    scenario = repositories.get_scenario(save.scenario_id)
    assert scenario is not None
    assert save.title == "First Thaw"
    assert scenario.type == "full_roleplay"
    assert scenario.title == "Frostglass Hall"
    assert scenario.player_role == "Relic hunter"
    scenario_content = json.loads(scenario.content_json)
    assert scenario_content["player_character_name"] == "Mara Voss"
    assert scenario_content["worldbuilding"] == "Mirrors remember every visitor."
    assert scenario_content["lore"] == "The hall was frozen by a failed oath."
    assert scenario_content["tone_genre"] == "Gothic exploration."

    messages = repositories.list_messages(save.id)
    message_rows = [
        (message.role, message.speaker_name, message.body) for message in messages
    ]
    assert message_rows == [
        ("narrator", "Narrator", "The frost cracks across the mirror floor."),
    ]
    assert _value(model, "active_save_id") == save.id
    model_message_bodies = [
        _value(message, "body") for message in _chronicle_messages(model)
    ]
    assert model_message_bodies == [
        "The frost cracks across the mirror floor.",
    ]


@pytest.mark.parametrize(
    ("scenario_type", "genre_fields", "unexpected_fields"),
    [
        (
            "fantasy_roleplay",
            {
                "magic_system": "Oath-magic spends memory when vows break.",
                "realms_and_places": "The Thorn March and its moonlit roads.",
                "factions_and_orders": "The Glass Court and the Ashen Abbey.",
                "myths_and_creatures": "Hollow saints haunt broken wells.",
                "quest_stakes": "Restore the road before winter eats the villages.",
            },
            ("technology_level", "setting_scope", "worldbuilding"),
        ),
        (
            "science_fiction_roleplay",
            {
                "technology_level": "FTL exists, but jump drives forget coordinates.",
                "setting_scope": "A failing relay station above a rogue planet.",
                "species_and_intelligences": "Archivist AIs and vacuum-adapted crews.",
                "factions_and_institutions": "The Charter Fleet and survey ministry.",
                "mission_stakes": "Recover the relay key before the system goes dark.",
            },
            ("magic_system", "realms_and_places", "worldbuilding"),
        ),
    ],
)
def test_create_manual_genre_roleplay_save_persists_unique_fields(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    scenario_type: str,
    genre_fields: dict[str, str],
    unexpected_fields: tuple[str, ...],
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = controller.create_manual_scenario(
        runtime.ManualScenarioInput(
            scenario_type=scenario_type,
            title="Signal Under Snow",
            premise="A border settlement receives an impossible warning.",
            player_character_name="Mara Voss",
            player_role="Warden of the last safe gate",
            tone_genre="Tense, mythic, and intimate.",
            opening_message="The warning bell rings in a language you know.",
            save_title="First Signal",
            **genre_fields,
        )
    )

    save = repositories.list_saves()[0]
    scenario = repositories.get_scenario(save.scenario_id)
    assert scenario is not None
    assert save.title == "First Signal"
    assert scenario.type == scenario_type
    assert scenario.title == "Signal Under Snow"
    assert scenario.player_role == "Warden of the last safe gate"

    scenario_content = json.loads(scenario.content_json)
    assert scenario_content["player_character_name"] == "Mara Voss"
    assert scenario_content["tone_genre"] == "Tense, mythic, and intimate."
    for field, expected in genre_fields.items():
        assert scenario_content[field] == expected
    for field in unexpected_fields:
        assert field not in scenario_content

    messages = repositories.list_messages(save.id)
    assert [(message.role, message.body) for message in messages] == [
        ("narrator", "The warning bell rings in a language you know."),
    ]
    assert _value(model, "active_save_id") == save.id


def test_create_manual_hybrid_scenario_persists_fields_and_seeds_routes(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = controller.create_manual_scenario(
        runtime.ManualScenarioInput(
            scenario_type="science_fiction_roleplay",
            scenario_types=("science_fiction_roleplay", "dating_sim"),
            title="Orbital Hearts",
            premise="A diplomatic academy turns courtship into first contact.",
            player_character_name="Ren Takahashi",
            player_character_profile=(
                "Ren is a courier-pilot trying to keep a fragile treaty alive "
                "while deciding who to trust."
            ),
            player_role="Envoy-pilot and romantic lead.",
            technology_level="Near-future orbital habitat and alien translators.",
            setting_scope="A disputed academy station above a treaty moon.",
            species_and_intelligences="Human students, uplift envoys, and station AIs.",
            factions_and_institutions="The academy council and rival charter fleets.",
            mission_stakes="Keep the treaty delegation alive through festival week.",
            tone_genre="Warm science-fiction romance with diplomatic pressure.",
            opening_message="The airlock opens on the first reception.",
            save_title="Orbital Hearts Save",
        )
    )

    active_save_id = _value(model, "active_save_id")
    save = repositories.get_save(active_save_id)
    assert save is not None
    assert save.title == "Orbital Hearts Save"
    scenario = repositories.get_scenario(save.scenario_id)
    assert scenario is not None
    assert scenario.type == "science_fiction_roleplay"
    content = json.loads(scenario.content_json)
    assert content["_scenario_genres"] == [
        "science_fiction_roleplay",
        "dating_sim",
    ]
    assert content["technology_level"] == (
        "Near-future orbital habitat and alien translators."
    )

    characters = repositories.list_characters(active_save_id)
    assert {character.name for character in characters} == {"Ren Takahashi"}
    routes = repositories.list_dating_route_states(active_save_id)
    assert routes == []


def test_create_manual_first_contact_exploration_save_persists_contact_fields(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = controller.create_manual_scenario(
        runtime.ManualScenarioInput(
            scenario_type="first_contact_exploration",
            title="Songs Under Europa",
            premise="A survey crew finds patterned signals under the ice.",
            player_character_name="Dr. Mara Voss",
            player_role="Mission linguist and acting contact lead",
            mission_profile="Survey the hidden ocean and avoid hostile contact.",
            ship_or_base_status="Habitat Kestrel has 42 hours of stable heat.",
            exploration_target="A black-water cavern beneath the ice shelf.",
            unknown_intelligence=(
                "An unseen whale-like intelligence answers sonar with pressure songs."
            ),
            knowledge_state=(
                "Observed: repeating pressure-wave songs. Hypothesis: cadence maps "
                "safe passages. Unknown: whether the singers know the crew is present."
            ),
            translation_progress=(
                "Learned term: three descending pulses may mean open water. "
                "False assumption: louder pulses are threats. Confirmed: blue "
                "light flashes mark attention."
            ),
            discoveries_and_samples=(
                "Ice cores contain living metallic spores under quarantine."
            ),
            hazards_and_escalation=(
                "Thermal fissures spread and the rescue window closes in two days."
            ),
            tone_genre="Hopeful, tense exploration science fiction.",
            opening_message="Blue light pulses under the ice before sonar speaks back.",
            save_title="Europa Contact",
        )
    )

    save = repositories.list_saves()[0]
    scenario = repositories.get_scenario(save.scenario_id)
    assert scenario is not None
    assert save.title == "Europa Contact"
    assert scenario.type == "first_contact_exploration"
    content = json.loads(scenario.content_json)
    assert content["mission_profile"] == (
        "Survey the hidden ocean and avoid hostile contact."
    )
    assert content["ship_or_base_status"] == (
        "Habitat Kestrel has 42 hours of stable heat."
    )
    assert content["translation_progress"] == (
        "Learned term: three descending pulses may mean open water. "
        "False assumption: louder pulses are threats. Confirmed: blue light "
        "flashes mark attention."
    )

    messages = repositories.list_messages(save.id)
    assert [(message.role, message.body) for message in messages] == [
        ("narrator", "Blue light pulses under the ice before sonar speaks back."),
    ]
    state = {row.key: row for row in repositories.list_world_state(save.id)}
    assert tuple(state) == (
        "contact.base",
        "contact.discoveries",
        "contact.hazards",
        "contact.intelligence",
        "contact.knowledge",
        "contact.mission",
        "contact.target",
        "contact.translation",
    )
    assert state["contact.mission"].value == {
        "summary": "Survey the hidden ocean and avoid hostile contact."
    }
    assert state["contact.base"].value == {
        "summary": "Habitat Kestrel has 42 hours of stable heat."
    }
    assert state["contact.target"].category == "location"
    assert state["contact.translation"].value == {
        "summary": (
            "Learned term: three descending pulses may mean open water. "
            "False assumption: louder pulses are threats. Confirmed: blue "
            "light flashes mark attention."
        )
    }
    assert state["contact.hazards"].category == "threat"
    assert {
        row.source_message_id for row in state.values()
    } == {messages[0].id}
    assert _value(model, "active_save_id") == save.id


def test_create_manual_investigation_mystery_save_persists_case_fields(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = controller.create_manual_scenario(
        runtime.ManualScenarioInput(
            scenario_type="investigation_mystery",
            title="Museum of Broken Hours",
            premise="A curator disappears during a public gala.",
            player_character_name="Inspector Mara Voss",
            player_role="The investigator assigned to the impossible case",
            case_facts="Curator Elian Vale vanished from a sealed gallery.",
            clues="Watch log gap from 9:10 to 9:18; undiscovered and reliable.",
            timeline="Public alarm at 9:21; hidden lift movement at 9:12.",
            red_herrings="A bloody glove belongs to a mannequin repair.",
            hidden_truth="Sera hid a smuggling ledger in the restoration lift.",
            case_status="Unresolved; public facts only.",
            tone_genre="Quiet investigative noir with careful clue continuity.",
            opening_message="Rain taps the museum glass as the gallery unlocks.",
            save_title="Broken Hours",
        )
    )

    save = repositories.list_saves()[0]
    scenario = repositories.get_scenario(save.scenario_id)
    assert scenario is not None
    assert save.title == "Broken Hours"
    assert scenario.type == "investigation_mystery"
    content = json.loads(scenario.content_json)
    assert content["case_facts"] == "Curator Elian Vale vanished from a sealed gallery."
    assert content["clues"] == (
        "Watch log gap from 9:10 to 9:18; undiscovered and reliable."
    )
    assert content["hidden_truth"] == (
        "Sera hid a smuggling ledger in the restoration lift."
    )
    assert content["case_status"] == "Unresolved; public facts only."
    assert [
        (message.role, message.body)
        for message in repositories.list_messages(save.id)
    ] == [
        ("narrator", "Rain taps the museum glass as the gallery unlocks."),
    ]
    assert _value(model, "active_save_id") == save.id


def test_create_manual_survival_expedition_seeds_expedition_state(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = controller.create_manual_scenario(
        runtime.ManualScenarioInput(
            scenario_type="survival_expedition",
            title="Whiteout Pass",
            premise="A rescue caravan must cross a frozen mountain pass.",
            player_character_name="Mara Voss",
            player_role="Expedition lead",
            expedition_goal="Reach Northwatch before the fever medicine spoils.",
            route_options="Cliff road, glacier basin, or old mine tunnel.",
            resource_inventory="Food: 9 days. Water: 6 skins. Medicine: 3 doses.",
            environmental_conditions="Late winter whiteouts and brittle ice.",
            hazards_and_events="Avalanche risk, frostbite, and wolves near timberline.",
            camp_status="Emergency bivouac on a sheltered ledge.",
            travel_progress="0 of 80 miles traveled; retreat remains possible.",
            tone_genre="Grounded alpine survival with human cost.",
            opening_message="Snow erases the mule tracks behind you.",
            save_title="Northwatch Run",
        )
    )

    save = repositories.list_saves()[0]
    scenario = repositories.get_scenario(save.scenario_id)
    assert scenario is not None
    assert save.title == "Northwatch Run"
    assert scenario.type == "survival_expedition"
    assert scenario.title == "Whiteout Pass"
    assert _value(model, "active_save_id") == save.id

    content = json.loads(scenario.content_json)
    assert content["expedition_goal"] == (
        "Reach Northwatch before the fever medicine spoils."
    )
    assert content["resource_inventory"] == (
        "Food: 9 days. Water: 6 skins. Medicine: 3 doses."
    )
    assert content["travel_progress"] == (
        "0 of 80 miles traveled; retreat remains possible."
    )
    assert repositories.list_messages(save.id)[0].body == (
        "Snow erases the mule tracks behind you."
    )

    state_by_key = {
        state.key: state for state in repositories.list_world_state(save.id)
    }
    assert set(state_by_key) >= {
        "expedition.goal",
        "expedition.route",
        "expedition.resources",
        "expedition.environment",
        "expedition.hazards",
        "expedition.camp",
        "expedition.progress",
    }
    assert state_by_key["expedition.resources"].category == "inventory"
    assert state_by_key["expedition.resources"].value == {
        "summary": "Food: 9 days. Water: 6 skins. Medicine: 3 doses."
    }
    assert state_by_key["expedition.progress"].category == "objective"
    assert state_by_key["expedition.progress"].value == {
        "summary": "0 of 80 miles traveled; retreat remains possible."
    }


def test_create_manual_heist_infiltration_seeds_heist_state(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = controller.create_manual_scenario(
        runtime.ManualScenarioInput(
            scenario_type="heist_infiltration",
            title="Skybank Treaty Job",
            premise="A crew must steal a treaty from a floating bank.",
            player_character_name="Mara Voss",
            player_role="Crew planner and face.",
            target_location="Skybank vault above the storm moorings.",
            objectives_and_stakes=(
                "Recover the treaty; copy the ledger if possible; avoid war."
            ),
            intel_and_access="Guard shift changes at bell three; lift code is split.",
            security_model=(
                "Clockwork cameras, badge checkpoints, two warded locks, and "
                "a silent alarm."
            ),
            alert_and_heat="Suspicion low; alarm inactive; public heat minimal.",
            loadout_and_tools="Forged badges, lockpicks, smoke pellets, storm skiff.",
            complications="A rival crew shadows the job.",
            extraction_routes="Primary storm skiff; fallback service stairs.",
            aftermath="Clean success keeps heat low; partial success starts a hunt.",
            tone_genre="Tense caper with careful consequences.",
            opening_message="The skybank bell strikes three.",
            save_title="Treaty Job",
        )
    )

    save = repositories.list_saves()[0]
    scenario = repositories.get_scenario(save.scenario_id)
    assert scenario is not None
    assert save.title == "Treaty Job"
    assert scenario.type == "heist_infiltration"
    assert _value(model, "active_save_id") == save.id

    content = json.loads(scenario.content_json)
    assert content["security_model"] == (
        "Clockwork cameras, badge checkpoints, two warded locks, and a silent alarm."
    )
    assert content["alert_and_heat"] == (
        "Suspicion low; alarm inactive; public heat minimal."
    )
    assert repositories.list_messages(save.id)[0].body == (
        "The skybank bell strikes three."
    )

    state_by_key = {
        state.key: state for state in repositories.list_world_state(save.id)
    }
    assert set(state_by_key) >= {
        "heist.target",
        "heist.objectives",
        "heist.intel",
        "heist.security",
        "heist.alert",
        "heist.loadout",
        "heist.complications",
        "heist.extraction",
        "heist.aftermath",
    }
    assert state_by_key["heist.security"].category == "security"
    assert state_by_key["heist.security"].value == {
        "summary": (
            "Clockwork cameras, badge checkpoints, two warded locks, and a "
            "silent alarm."
        )
    }
    assert state_by_key["heist.alert"].category == "threat"
    assert state_by_key["heist.alert"].value == {
        "summary": "Suspicion low; alarm inactive; public heat minimal."
    }


def test_create_manual_political_intrigue_seeds_social_state(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = controller.create_manual_scenario(
        runtime.ManualScenarioInput(
            scenario_type="political_intrigue",
            title="Council of Ash",
            premise="A city council vote will decide who controls the harbor.",
            player_character_name="Mara Voss",
            player_role="Envoy holding the swing vote",
            political_arena="The harbor council chamber and public galleries.",
            political_factions="Guilds, Old Families, and dock unions.",
            central_conflict="A midnight no-confidence vote can replace the regent.",
            secrets_and_leverage="Only Mara knows Orro moved missing silver.",
            reputation_and_standing="Mara is trusted by reformers.",
            obligations_and_favors="Orro owes Mara one public endorsement.",
            alliances_and_rivalries="Reformers court Mara; old houses resist.",
            event_calendar="Dawn hearing; noon procession; midnight vote.",
            political_pressure="The midnight vote proceeds unless delayed.",
            public_private_knowledge=(
                "The public knows the vote is close; only Mara knows the favor."
            ),
            tone_genre="Tense council intrigue.",
            opening_message="The council bell rings.",
            save_title="Ash Council",
        )
    )

    save = repositories.list_saves()[0]
    scenario = repositories.get_scenario(save.scenario_id)
    assert scenario is not None
    assert save.title == "Ash Council"
    assert scenario.type == "political_intrigue"
    assert _value(model, "active_save_id") == save.id

    content = json.loads(scenario.content_json)
    assert content["political_factions"] == "Guilds, Old Families, and dock unions."
    assert content["political_pressure"] == (
        "The midnight vote proceeds unless delayed."
    )
    assert repositories.list_messages(save.id)[0].body == "The council bell rings."

    state_by_key = {
        state.key: state for state in repositories.list_world_state(save.id)
    }
    assert set(state_by_key) >= {
        "intrigue.arena",
        "intrigue.factions",
        "intrigue.conflict",
        "intrigue.secrets",
        "intrigue.standing",
        "intrigue.obligations",
        "intrigue.alliances",
        "intrigue.calendar",
        "intrigue.pressure",
        "intrigue.knowledge",
    }
    assert state_by_key["intrigue.obligations"].category == "obligation"
    assert state_by_key["intrigue.obligations"].value == {
        "summary": "Orro owes Mara one public endorsement."
    }
    assert state_by_key["intrigue.standing"].category == "reputation"
    assert state_by_key["intrigue.standing"].value == {
        "summary": "Mara is trusted by reformers."
    }
    assert state_by_key["intrigue.pressure"].category == "deadline"
    assert state_by_key["intrigue.pressure"].value == {
        "summary": "The midnight vote proceeds unless delayed."
    }

def test_create_manual_time_loop_seeds_loop_state(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = controller.create_manual_scenario(
        runtime.ManualScenarioInput(
            scenario_type="time_loop",
            title="Bellwether Day",
            premise="A harbor festival repeats until the drowned bell is saved.",
            player_character_name="Mara Voss",
            player_role="Archivist who notices the day repeating.",
            loop_premise=(
                "The same festival day resets after the harbor bell sinks beneath "
                "the tide."
            ),
            reset_trigger="Reset occurs when the drowned bell tolls at midnight.",
            loop_duration="Twenty-four hours, dawn bell to dawn bell.",
            starting_state="Mara wakes in the archive loft with a wet matchbook.",
            objective="Prevent the bell from sinking and identify the saboteur.",
            failure_conditions="The bell sinks, Mara dies, or midnight arrives.",
            baseline_world_state=(
                "At dawn the harbor is intact, the tower is locked, and Mira is "
                "skeptical."
            ),
            loop_schedule="09:00 parade; 18:00 storm tide; 23:45 sabotage window.",
            persistent_knowledge=(
                "Player/meta knowledge persists: tower code, Mira's warning, and "
                "the tunnel route."
            ),
            persistence_exceptions="A salt mark and wet matchbook persist.",
            npc_memory_rules="NPCs reset unless an exception says otherwise.",
            current_loop_state="Loop 1, dawn phase, no deviations confirmed.",
            tone_genre="Clockwork mystery with coastal urgency.",
            opening_message="The same bell rings dawn again.",
            save_title="Bell Loop",
        )
    )

    save = repositories.list_saves()[0]
    scenario = repositories.get_scenario(save.scenario_id)
    assert scenario is not None
    assert save.title == "Bell Loop"
    assert scenario.type == "time_loop"
    assert _value(model, "active_save_id") == save.id
    content = json.loads(scenario.content_json)
    assert content["persistent_knowledge"] == (
        "Player/meta knowledge persists: tower code, Mira's warning, and the "
        "tunnel route."
    )
    assert repositories.list_messages(save.id)[0].body == (
        "The same bell rings dawn again."
    )

    state_by_key = {
        state.key: state for state in repositories.list_world_state(save.id)
    }
    assert set(state_by_key) >= {
        "loop.rules",
        "loop.starting_state",
        "loop.objective",
        "loop.baseline",
        "loop.schedule",
        "loop.knowledge",
        "loop.persistence",
        "loop.npc_memory",
        "loop.current",
    }
    assert state_by_key["loop.knowledge"].category == "loop_persistent"
    assert state_by_key["loop.knowledge"].value == {
        "summary": (
            "Player/meta knowledge persists: tower code, Mira's warning, and "
            "the tunnel route."
        )
    }
    assert state_by_key["loop.baseline"].category == "loop_resettable"
    assert state_by_key["loop.current"].value == {
        "summary": "Loop 1, dawn phase, no deviations confirmed."
    }


def test_create_manual_scenario_can_skip_process_global_selection(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = controller.create_manual_scenario(
        runtime.ManualScenarioInput(
            scenario_type="full_roleplay",
            title="Frostglass Hall",
            premise="A sealed hall is thawing after a century.",
            player_role="Relic hunter",
            opening_message="The frost cracks across the mirror floor.",
        ),
        remember_process_active_save=False,
    )

    save = repositories.list_saves()[0]
    assert _value(model, "active_save_id") == save.id
    assert controller.active_save_id is None


@pytest.mark.parametrize(
    ("configure_scenario_generation", "expected_model_id"),
    [
        (True, "fake-scenario"),
        (False, "fake-chat"),
    ],
)
def test_generate_scenario_draft_uses_scenario_preference_or_chat_fallback_without_save(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    configure_scenario_generation: bool,
    expected_model_id: str,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    if configure_scenario_generation:
        repositories.set_model_preference(
            task="scenario_generation",
            provider="fake",
            model_id="fake-scenario",
        )
    provider = RuntimeFakeProvider()
    controller = _runtime_controller(runtime, repositories, tmp_path, provider=provider)

    model = asyncio.run(
        controller.generate_scenario_draft(
            scenario_type="full_roleplay",
            seed="A harbor that keeps old promises.",
        )
    )

    assert repositories.list_saves() == []
    assert len(provider.chat_requests) == len(provider.scenario_sections)
    assert {request.model_id for request in provider.chat_requests} == {
        expected_model_id
    }
    assert all(
        "User request:\nA harbor that keeps old promises." in request.messages[-1].body
        for request in provider.chat_requests
    )
    assert [
        _requested_scenario_section(request.messages[-1].body)
        for request in provider.chat_requests
    ] == list(provider.scenario_sections)

    draft = _value(model, "scenario_draft")
    assert isinstance(draft, runtime.ScenarioDraftModel)
    assert _value(draft, "scenario_type") == "full_roleplay"
    assert dict(_value(draft, "sections")) == provider.scenario_sections
    assert dict(_value(draft, "source_metadata")) == {
        "origin": "ai_draft",
        "generation_prompt": "A harbor that keeps old promises.",
    }
    assert _status_text(model) == "Scenario draft generated"


def test_generate_scenario_draft_uses_section_model_override(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    repositories.set_model_preference(
        task="scenario_generation",
        provider="fake",
        model_id="fake-scenario",
    )
    repositories.set_model_preference(
        task=scenario_generation_section_model_task("opening_message"),
        provider="deep",
        model_id="opening-drafter",
    )
    default_provider = RuntimeFakeProvider()
    override_provider = RuntimeFakeProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        providers={
            "fake": default_provider,
            "deep": override_provider,
        },
    )

    model = asyncio.run(
        controller.generate_scenario_draft(
            scenario_type="full_roleplay",
            seed="A harbor that keeps old promises.",
        )
    )

    assert _error_text(model) == ""
    assert [
        _requested_scenario_section(request.messages[-1].body)
        for request in override_provider.chat_requests
    ] == ["opening_message"]
    assert {request.model_id for request in override_provider.chat_requests} == {
        "opening-drafter"
    }
    assert "opening_message" not in {
        _requested_scenario_section(request.messages[-1].body)
        for request in default_provider.chat_requests
    }
    assert {request.model_id for request in default_provider.chat_requests} == {
        "fake-scenario"
    }
    draft = _value(model, "scenario_draft")
    assert dict(_value(draft, "sections")) == default_provider.scenario_sections


def test_generate_continuation_scenario_draft_uses_current_save_state(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    repositories.set_model_preference(
        task="scenario_generation",
        provider="fake",
        model_id="fake-scenario",
    )
    provider = RuntimeFakeProvider()
    provider.scenario_sections = {
        section_id: f"Continuation {section_id}"
        for section_id in CONTINUATION_SECTION_IDS
    }
    save = _create_save_with_current_state(repositories)
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
    )
    controller.active_save_id = save.id

    model = asyncio.run(
        controller.generate_continuation_scenario_draft(
            chapter_start_instructions=(
                "Characters are going to bed; start the next chapter after sunrise."
            )
        )
    )

    assert _error_text(model) == ""
    assert [
        _requested_scenario_section(request.messages[-1].body)
        for request in provider.chat_requests
    ] == list(CONTINUATION_SECTION_IDS)
    assert all(
        "Create a clean chapter/continuation scenario" in request.messages[-1].body
        for request in provider.chat_requests
    )
    assert "Transcript-only old chat" not in provider.chat_requests[0].messages[-1].body
    assert "Mara promised Ren the next truthful bell toll" in (
        provider.chat_requests[0].messages[-1].body
    )
    assert "Chapter start instructions" in provider.chat_requests[0].messages[-1].body
    assert (
        "Characters are going to bed; start the next chapter after sunrise."
        in provider.chat_requests[0].messages[-1].body
    )
    draft = _value(model, "scenario_draft")
    assert isinstance(draft, runtime.ScenarioDraftModel)
    assert dict(_value(draft, "sections")) == provider.scenario_sections
    metadata = dict(_value(draft, "source_metadata"))
    assert metadata["origin"] == "save_continuation"
    assert metadata["source_save_id"] == save.id
    assert metadata["source_save_title"] == "First Harbor"
    assert metadata["source_message_count"] == 2
    assert metadata["generation_prompt"] == (
        "Characters are going to bed; start the next chapter after sunrise."
    )
    assert "chapter_start_instructions" not in metadata
    assert _status_text(model) == "Continuation draft generated"


def test_save_scenario_draft_seeds_continuation_character_metadata(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = asyncio.run(
        _save_scenario_draft(
            controller,
            scenario_type="full_roleplay",
            sections={
                **_reviewed_full_roleplay_sections(),
            },
            save_title="Chapter Two",
            request_initial_image=False,
            source_metadata={
                "origin": "save_continuation",
                "character_continuity": [
                    {
                        "name": "Mara Voss",
                        "aliases": ["Mara"],
                        "role": "Harbor warden",
                        "known_state": "Carrying the bell-key debt.",
                        "met": True,
                        "appearance": "Salt-stained blue coat.",
                        "visual_notes": "Keeps one glove buttoned.",
                        "personality": "Stubborn, tender under pressure.",
                        "voice": "clipped, dry, careful with promises",
                        "relationships": {"Ren": "owes him the bell-key"},
                        "status": "alive and negotiating",
                        "private_notes": "Knows the bell is a prison.",
                    }
                ],
            },
        )
    )

    active_save_id = _value(model, "active_save_id")
    characters = repositories.list_characters(active_save_id)
    mara = next(character for character in characters if character.name == "Mara Voss")
    assert mara.voice == "clipped, dry, careful with promises"
    assert mara.relationships == {"Ren": "owes him the bell-key"}
    assert mara.private_notes == "Knows the bell is a prison."
    assert mara.protected_from_maintenance is True


def test_generate_scenario_draft_forwards_progress_models(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    repositories.set_model_preference(
        task="scenario_generation",
        provider="fake",
        model_id="fake-scenario",
    )
    provider = RuntimeFakeProvider()
    controller = _runtime_controller(runtime, repositories, tmp_path, provider=provider)
    progress_updates: list[object] = []

    model = asyncio.run(
        controller.generate_scenario_draft(
            scenario_type="full_roleplay",
            seed="A harbor that keeps old promises.",
            progress_callback=progress_updates.append,
        )
    )

    assert _error_text(model) == ""
    assert _status_text(model) == "Scenario draft generated"
    total_sections = len(provider.scenario_sections)
    assert len(progress_updates) == len(provider.scenario_sections) * 2
    assert isinstance(progress_updates[0], runtime.ScenarioDraftProgressModel)
    assert [
        (
            _value(progress, "scenario_type"),
            _value(progress, "section_id"),
            _value(progress, "status"),
            _value(progress, "completed_count"),
            _value(progress, "total_count"),
            dict(_value(progress, "completed_sections")),
        )
        for progress in progress_updates[:4]
    ] == [
        ("full_roleplay", "title", "generating", 0, total_sections, {}),
        (
            "full_roleplay",
            "title",
            "completed",
            1,
            total_sections,
            {"title": "Glass Harbor"},
        ),
        (
            "full_roleplay",
            "premise",
            "generating",
            1,
            total_sections,
            {"title": "Glass Harbor"},
        ),
        (
            "full_roleplay",
            "premise",
            "completed",
            2,
            total_sections,
            {
                "title": "Glass Harbor",
                "premise": "A drowned harbor rings its bell at low tide.",
            },
        ),
    ]
    final_progress = progress_updates[-1]
    assert _value(final_progress, "section_id") == "opening_message"
    assert _value(final_progress, "status") == "completed"
    assert _value(final_progress, "completed_count") == total_sections
    assert _value(final_progress, "total_count") == total_sections
    assert dict(_value(final_progress, "completed_sections")) == (
        provider.scenario_sections
    )


def test_generate_scenario_draft_character_starters_appends_structured_results(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    repositories.set_model_preference(
        task="dating_sim_context_update",
        provider="fake",
        model_id="fake-structured",
    )
    _save_fake_provider_model(
        repositories,
        model_id="fake-structured",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    provider = RuntimeStructuredCleanupProvider(
        [
            {
                "characters": [
                    {
                        "name": "Avery Quinn",
                        "role": "Emergency lighting technician.",
                        "known_state": "Avery restores light during the blackout.",
                        "appearance": "Tool belt and bright vest.",
                        "visual_notes": "Portable work lamp.",
                        "personality": "Practical and warm.",
                        "voice": 'Plainspoken. Example: "Hold this."',
                        "texting_style": "Brief logistics. Sample text: Found it.",
                        "goals": "Keep the reception safe.",
                        "motivations": "Prove competence under pressure.",
                        "boundaries": "Will not ignore safety hazards.",
                    }
                ]
            }
        ]
    )
    controller = _runtime_controller(runtime, repositories, tmp_path, provider=provider)

    model = asyncio.run(
        controller.generate_scenario_draft_character_starters(
            scenario_type="dating_sim",
            sections=_reviewed_dating_sim_sections(),
            character_starters=[
                {
                    "name": "Mika Arai",
                    "role": "Student council president",
                }
            ],
            count=1,
        )
    )

    assert _error_text(model) == ""
    assert _status_text(model) == "Character starters generated"
    draft = _value(model, "scenario_draft")
    starters = _value(draft, "character_starters")
    assert [starter["name"] for starter in starters] == ["Mika Arai", "Avery Quinn"]
    request = provider.structured_requests[0]
    assert request.schema_name == "scenario_character_starters"
    assert request.provider == "fake"
    assert request.model_id == "fake-structured"
    assert "Create exactly 1 new character starters" in request.messages[0].body
    request_body = request.messages[1].body
    assert "Ren Takahashi" in request_body
    assert "Mika Arai" in request_body
    assert "Ordinary contemporary name candidates" in request_body


def test_generate_scenario_draft_character_starters_requires_count_or_description(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = asyncio.run(
        controller.generate_scenario_draft_character_starters(
            scenario_type="dating_sim",
            sections=_reviewed_dating_sim_sections(),
        )
    )

    assert _error_text(model) == (
        "Number of characters or custom character description is required"
    )


def test_regenerate_scenario_section_preserves_other_draft_sections(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RuntimeFakeProvider()
    provider.scenario_sections["locations"] = "Regenerated lighthouse cavern."
    controller = _runtime_controller(runtime, repositories, tmp_path, provider=provider)
    sections = {
        "title": "Reviewed Glass Harbor",
        "premise": "A revised drowned harbor mystery.",
        "player_character_name": "Mara Voss",
        "player_role": "Reef investigator",
        "worldbuilding": "Reviewed worldbuilding.",
        "lore": "Reviewed lore.",
        "locations": "Old bell tower.",
        "factions": "Reviewed Tidemarked Guild.",
        "tone_genre": "Reviewed nautical noir.",
        "opening_message": "Reviewed opening bell.",
    }

    model = asyncio.run(
        controller.regenerate_scenario_section(
            scenario_type="full_roleplay",
            seed="A harbor that keeps old promises.",
            section_id="locations",
            sections=sections,
        )
    )

    assert len(provider.chat_requests) == 1
    request = provider.chat_requests[0]
    assert request.model_id == "fake-chat"
    assert _requested_scenario_section(request.messages[-1].body) == "locations"

    draft = _value(model, "scenario_draft")
    assert _value(draft, "scenario_type") == "full_roleplay"
    updated_sections = dict(_value(draft, "sections"))
    assert updated_sections == {
        **sections,
        "locations": "Regenerated lighthouse cavern.",
    }
    assert _status_text(model) == "Section regenerated"


def test_regenerate_scenario_section_uses_section_model_override(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    repositories.set_model_preference(
        task="scenario_generation",
        provider="fake",
        model_id="fake-scenario",
    )
    repositories.set_model_preference(
        task=scenario_generation_section_model_task("locations"),
        provider="deep",
        model_id="deep-locations",
    )
    default_provider = RuntimeFakeProvider()
    override_provider = RuntimeFakeProvider()
    override_provider.scenario_sections["locations"] = "Regenerated lighthouse cavern."
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        providers={
            "fake": default_provider,
            "deep": override_provider,
        },
    )
    sections = {
        "title": "Reviewed Glass Harbor",
        "premise": "A revised drowned harbor mystery.",
        "player_character_name": "Mara Voss",
        "player_role": "Reef investigator",
        "worldbuilding": "Reviewed worldbuilding.",
        "lore": "Reviewed lore.",
        "locations": "Old bell tower.",
        "factions": "Reviewed Tidemarked Guild.",
        "tone_genre": "Reviewed nautical noir.",
        "opening_message": "Reviewed opening bell.",
    }

    model = asyncio.run(
        controller.regenerate_scenario_section(
            scenario_type="full_roleplay",
            seed="A harbor that keeps old promises.",
            section_id="locations",
            sections=sections,
        )
    )

    assert default_provider.chat_requests == []
    assert len(override_provider.chat_requests) == 1
    request = override_provider.chat_requests[0]
    assert request.provider == "deep"
    assert request.model_id == "deep-locations"
    assert _requested_scenario_section(request.messages[-1].body) == "locations"
    draft = _value(model, "scenario_draft")
    assert dict(_value(draft, "sections")) == {
        **sections,
        "locations": "Regenerated lighthouse cavern.",
    }


def test_save_scenario_draft_persists_reviewed_sections_creates_active_save_and_opening(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)
    reviewed_sections = _reviewed_full_roleplay_sections()

    model = asyncio.run(
        _save_scenario_draft(
            controller,
            scenario_type="full_roleplay",
            sections=reviewed_sections,
            save_title="Reviewed Save",
            request_initial_image=False,
        )
    )

    saves = repositories.list_saves()
    assert len(saves) == 1
    save = saves[0]
    assert save.title == "Reviewed Save"
    assert _value(model, "active_save_id") == save.id

    scenario = repositories.get_scenario(save.scenario_id)
    assert scenario is not None
    assert scenario.type == "full_roleplay"
    assert scenario.title == "Reviewed Glass Harbor"
    assert scenario.premise == "A revised drowned harbor mystery."
    assert scenario.player_role == "Reef investigator"
    assert json.loads(scenario.content_json) == reviewed_sections

    messages = repositories.list_messages(save.id)
    message_rows = [
        (message.role, message.speaker_name, message.body) for message in messages
    ]
    assert message_rows == [
        ("narrator", "Narrator", "Reviewed opening bell."),
    ]
    model_message_bodies = [
        _value(message, "body") for message in _chronicle_messages(model)
    ]
    assert model_message_bodies == ["Reviewed opening bell."]
    assert _status_text(model) == "Created save: Reviewed Save"


def test_save_scenario_draft_preserves_opening_at_adult_rating(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)
    adult = repositories.create_user(
        username="Ilyra",
        role="user",
        password_hash="hash",
    )
    repositories.set_scoped_setting(
        scope="user",
        scope_id=adult.id,
        key="content_filter_rating",
        value="r",
    )
    repositories.set_scoped_setting(
        scope="user",
        scope_id=adult.id,
        key="fade_to_black_enabled",
        value=False,
    )
    opening = "He murdered the guard before the alarm sounded."
    reviewed_sections = {
        **_reviewed_full_roleplay_sections(),
        "opening_message": opening,
    }

    model = asyncio.run(
        controller.save_scenario_draft(
            scenario_type="full_roleplay",
            sections=reviewed_sections,
            save_title="Reviewed Save",
            current_user_id=adult.id,
        )
    )

    save_id = _value(model, "active_save_id")
    assert save_id is not None
    assert repositories.list_messages(save_id)[0].body == opening


def test_start_saved_scenario_filters_opening_for_child_account(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)
    child = repositories.create_user(
        username="Ilyra",
        role="child",
        password_hash="hash",
    )
    repositories.set_scoped_setting(
        scope="user",
        scope_id=child.id,
        key="content_filter_rating",
        value="g",
    )
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "opening_message": "Blood covered the floor after the attack.",
        },
    )

    model = controller.start_saved_scenario(
        scenario_id=scenario.id,
        current_user_id=child.id,
    )

    save_id = _value(model, "active_save_id")
    assert save_id is not None
    assert repositories.list_messages(save_id)[0].body == CONTENT_FILTER_TRANSITION


def test_start_saved_scenario_filters_graphic_gore_for_default_child_rating(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)
    child = repositories.create_user(
        username="Ilyra",
        role="child",
        password_hash="hash",
    )
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "opening_message": (
                "He disemboweled the guard, spilling his organs across the floor."
            ),
        },
    )

    model = controller.start_saved_scenario(
        scenario_id=scenario.id,
        current_user_id=child.id,
    )

    save_id = _value(model, "active_save_id")
    assert save_id is not None
    assert repositories.list_messages(save_id)[0].body == CONTENT_FILTER_TRANSITION


def test_save_scenario_draft_persists_generation_prompt_metadata(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)
    reviewed_sections = _reviewed_full_roleplay_sections()

    model = asyncio.run(
        _save_scenario_draft(
            controller,
            scenario_type="full_roleplay",
            sections=reviewed_sections,
            save_title="Reviewed Save",
            request_initial_image=False,
            source_metadata={
                "origin": "ai_draft",
                "generation_prompt": "A drowned harbor with a bell mystery.",
            },
        )
    )

    save_id = _value(model, "active_save_id")
    save = repositories.get_save(save_id)
    assert save is not None
    scenario = repositories.get_scenario(save.scenario_id)
    assert scenario is not None
    content = json.loads(scenario.content_json)
    assert content["_source"] == {
        "origin": "ai_draft",
        "generation_prompt": "A drowned harbor with a bell mystery.",
    }


def test_save_first_contact_exploration_draft_seeds_contact_state(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)
    sections = _reviewed_first_contact_exploration_sections()

    model = asyncio.run(
        _save_scenario_draft(
            controller,
            scenario_type="first_contact_exploration",
            sections=sections,
            save_title="Reviewed Europa Contact",
            request_initial_image=False,
        )
    )

    save = repositories.list_saves()[0]
    assert _value(model, "active_save_id") == save.id
    scenario = repositories.get_scenario(save.scenario_id)
    assert scenario is not None
    assert scenario.type == "first_contact_exploration"
    state = {row.key: row for row in repositories.list_world_state(save.id)}
    assert state["contact.mission"].value == {
        "summary": "Survey the hidden ocean and keep the crew alive."
    }
    assert state["contact.base"].category == "base"
    assert state["contact.target"].value == {
        "summary": "A black-water cavern under the ice shelf."
    }
    assert state["contact.intelligence"].category == "contact"
    assert state["contact.translation"].category == "translation"
    assert state["contact.hazards"].value == {
        "summary": "Thermal fissures spread while the rescue window narrows."
    }


def test_save_survival_expedition_draft_seeds_expedition_state(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)
    sections = _reviewed_survival_expedition_sections()

    model = asyncio.run(
        _save_scenario_draft(
            controller,
            scenario_type="survival_expedition",
            sections=sections,
            save_title="Relay Crossing",
            request_initial_image=False,
        )
    )

    save = repositories.list_saves()[0]
    assert _value(model, "active_save_id") == save.id
    scenario = repositories.get_scenario(save.scenario_id)
    assert scenario is not None
    assert scenario.type == "survival_expedition"
    state = {row.key: row for row in repositories.list_world_state(save.id)}
    assert state["expedition.route"].value == {
        "summary": "North wells, glass canyon, or direct salt road."
    }
    assert "expedition.party" not in state
    assert state["expedition.environment"].category == "expedition"
    assert state["expedition.hazards"].category == "threat"


def test_save_time_loop_draft_seeds_loop_state(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)
    sections = _reviewed_time_loop_sections()

    model = asyncio.run(
        _save_scenario_draft(
            controller,
            scenario_type="time_loop",
            sections=sections,
            save_title="Reviewed Bell Loop",
            request_initial_image=False,
        )
    )

    save = repositories.list_saves()[0]
    assert _value(model, "active_save_id") == save.id
    scenario = repositories.get_scenario(save.scenario_id)
    assert scenario is not None
    assert scenario.type == "time_loop"
    state = {row.key: row for row in repositories.list_world_state(save.id)}
    assert state["loop.rules"].value == {
        "summary": (
            "The same festival day resets after the harbor bell sinks beneath "
            "the tide.\n\n"
            "Reset trigger: Reset occurs when the drowned bell tolls at midnight.\n\n"
            "Loop duration: Twenty-four hours, dawn festival bell to dawn festival "
            "bell.\n\n"
            "Failure conditions: The bell sinks, Mara dies, or the day reaches "
            "midnight."
        )
    }
    assert state["loop.objective"].category == "objective"
    assert state["loop.knowledge"].category == "loop_persistent"
    assert state["loop.npc_memory"].value == {
        "summary": (
            "NPCs reset to dawn memories unless a persistence exception says "
            "otherwise."
        )
    }


def test_save_political_intrigue_draft_seeds_intrigue_state(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)
    sections = _reviewed_political_intrigue_sections()

    model = asyncio.run(
        _save_scenario_draft(
            controller,
            scenario_type="political_intrigue",
            sections=sections,
            save_title="Reviewed Ash Council",
            request_initial_image=False,
        )
    )

    save = repositories.list_saves()[0]
    assert _value(model, "active_save_id") == save.id
    scenario = repositories.get_scenario(save.scenario_id)
    assert scenario is not None
    assert scenario.type == "political_intrigue"
    state = {row.key: row for row in repositories.list_world_state(save.id)}
    assert state["intrigue.factions"].value == {
        "summary": "Guilds, Old Families, and dock unions."
    }
    assert state["intrigue.obligations"].value == {
        "summary": "Orro owes Mara one public endorsement."
    }
    assert state["intrigue.standing"].category == "reputation"
    assert state["intrigue.pressure"].category == "deadline"


@pytest.mark.parametrize(
    ("scenario_type", "section_factory", "expected_state"),
    [
        (
            "settlement_builder",
            "_reviewed_settlement_builder_sections",
            (
                ("settlement.resources", "resource", "resources_and_indicators"),
                ("settlement.projects", "project", "projects_and_facilities"),
            ),
        ),
        (
            "monster_hunt_bounty",
            "_reviewed_monster_hunt_bounty_sections",
            (
                ("hunt.leads", "clue", "leads_and_clues"),
                ("hunt.status", "objective", "hunt_status"),
            ),
        ),
        (
            "road_trip_pilgrimage",
            "_reviewed_road_trip_pilgrimage_sections",
            (
                ("journey.progress", "objective", "journey_progress"),
                ("journey.relationships", "relationship", "relationship_threads"),
            ),
        ),
        (
            "merchant_trade_route",
            "_reviewed_merchant_trade_route_sections",
            (
                ("trade.cargo", "inventory", "cargo_inventory"),
                ("trade.contracts", "contract", "contracts_and_debts"),
            ),
        ),
    ],
)
def test_save_management_scenario_draft_seeds_template_state(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    scenario_type: str,
    section_factory: str,
    expected_state: tuple[tuple[str, str, str], ...],
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)
    section_factory_func = cast(
        Callable[[], dict[str, str]],
        globals()[section_factory],
    )
    sections = section_factory_func()

    model = asyncio.run(
        _save_scenario_draft(
            controller,
            scenario_type=scenario_type,
            sections=sections,
            save_title=sections["title"],
            request_initial_image=False,
        )
    )

    save = repositories.list_saves()[0]
    assert _value(model, "active_save_id") == save.id
    scenario = repositories.get_scenario(save.scenario_id)
    assert scenario is not None
    assert scenario.type == scenario_type
    state = {row.key: row for row in repositories.list_world_state(save.id)}
    for key, category, section_id in expected_state:
        assert state[key].category == category
        assert state[key].value == {"summary": sections[section_id]}


@pytest.mark.parametrize(
    ("scenario_type", "section_factory", "expected_state_key"),
    [
        (
            "settlement_builder",
            "_reviewed_settlement_builder_sections",
            "settlement.projects",
        ),
        (
            "monster_hunt_bounty",
            "_reviewed_monster_hunt_bounty_sections",
            "hunt.leads",
        ),
        (
            "road_trip_pilgrimage",
            "_reviewed_road_trip_pilgrimage_sections",
            "journey.progress",
        ),
        (
            "merchant_trade_route",
            "_reviewed_merchant_trade_route_sections",
            "trade.contracts",
        ),
    ],
)
def test_create_manual_management_scenario_keeps_template_fields_and_state(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    scenario_type: str,
    section_factory: str,
    expected_state_key: str,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)
    section_factory_func = cast(
        Callable[[], dict[str, str]],
        globals()[section_factory],
    )
    sections = section_factory_func()

    model = controller.create_manual_scenario(
        runtime.ManualScenarioInput(
            scenario_type=scenario_type,
            save_title=f"{sections['title']} Save",
            **sections,
        )
    )

    save = repositories.list_saves()[0]
    scenario = repositories.get_scenario(save.scenario_id)
    assert scenario is not None
    assert _value(model, "active_save_id") == save.id
    assert scenario.type == scenario_type
    content = json.loads(scenario.content_json)
    for key, value in sections.items():
        assert content[key] == value
    assert expected_state_key in {
        row.key for row in repositories.list_world_state(save.id)
    }


def test_save_scenario_draft_defers_generated_full_roleplay_outcome_conditions(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    _configure_scenario_outcome_model(repositories)
    provider = RuntimeStructuredCleanupProvider(
        [
            {
                "loss_conditions": [
                    {
                        "name": "Harbor swallowed",
                        "description": (
                            "The scenario fails if the drowned harbor fully "
                            "claims the player."
                        ),
                    }
                ]
            }
        ]
    )
    controller = _runtime_controller(runtime, repositories, tmp_path, provider=provider)

    model = asyncio.run(
        _save_scenario_draft(
            controller,
            scenario_type="full_roleplay",
            sections=_reviewed_full_roleplay_sections(),
            save_title="Reviewed Save",
            request_initial_image=False,
        )
    )

    active_save_id = _value(model, "active_save_id")
    assert repositories.list_loss_conditions(active_save_id) == []
    assert repositories.list_loss_condition_changes(active_save_id) == []
    assert provider.chat_requests == []
    assert provider.structured_requests == []
    assert _status_text(model) == "Created save: Reviewed Save"


def test_save_scenario_draft_skips_structured_outcome_seed_when_provider_would_fail(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    _configure_scenario_outcome_model(repositories)
    provider = FailingRuntimeStructuredProvider()
    controller = _runtime_controller(runtime, repositories, tmp_path, provider=provider)

    model = asyncio.run(
        _save_scenario_draft(
            controller,
            scenario_type="full_roleplay",
            sections=_reviewed_full_roleplay_sections(),
            save_title="Reviewed Save",
            request_initial_image=False,
        )
    )

    saves = repositories.list_saves()
    assert len(saves) == 1
    save = saves[0]
    assert _value(model, "active_save_id") == save.id
    assert repositories.list_loss_conditions(save.id) == []
    assert [message.body for message in repositories.list_messages(save.id)] == [
        "Reviewed opening bell."
    ]
    assert provider.structured_requests == []
    assert _error_text(model) == ""
    assert _status_text(model) == "Created save: Reviewed Save"
    assert "sk-live-secret" not in (_status_text(model) or "")


def test_save_scenario_draft_ignores_legacy_loss_condition_metadata(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    _configure_scenario_outcome_model(repositories)
    provider = RuntimeStructuredCleanupProvider(
        [
            {
                "loss_conditions": [
                    {
                        "name": "Should not be used",
                        "description": "Structured seeding should not run.",
                    }
                ]
            }
        ]
    )
    controller = _runtime_controller(runtime, repositories, tmp_path, provider=provider)

    model = asyncio.run(
        _save_scenario_draft(
            controller,
            scenario_type="full_roleplay",
            sections=_reviewed_full_roleplay_sections(),
            save_title="Reviewed Save",
            request_initial_image=False,
            source_metadata={
                "loss_conditions": [
                    {
                        "name": "Manual ending",
                        "description": "The ritual ends badly if the bell breaks.",
                    }
                ]
            },
        )
    )

    active_save_id = _value(model, "active_save_id")
    assert repositories.list_loss_conditions(active_save_id) == []
    assert provider.structured_requests == []
    details = repositories.load_save_details(active_save_id)
    assert details is not None
    assert "loss_conditions" not in details.scenario.content_json
    assert _status_text(model) == "Created save: Reviewed Save"


def test_save_scenario_draft_seeds_reviewed_full_roleplay_starting_npcs(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)
    reviewed_sections = _reviewed_full_roleplay_sections()
    character_starters: list[dict[str, object]] = [
        {
            "name": "Captain Ilyra",
            "role": "Exiled commander",
            "known_state": "Captain Ilyra watches the reef gate.",
        },
        {
            "name": "Brother Senn",
            "role": "Informant",
            "known_state": "Brother Senn carries harbor rumors.",
        },
        {
            "name": "Vey the outrider",
            "aliases": ["Vey"],
            "role": "Outrider",
            "known_state": "Vey scouts the drowned road.",
        },
    ]

    model = asyncio.run(
        _save_scenario_draft(
            controller,
            scenario_type="full_roleplay",
            sections=reviewed_sections,
            character_starters=character_starters,
            save_title="Reviewed Save",
            request_initial_image=False,
        )
    )

    active_save_id = _value(model, "active_save_id")
    save = repositories.get_save(active_save_id)
    assert save is not None
    scenario = repositories.get_scenario(save.scenario_id)
    assert scenario is not None
    content = json.loads(scenario.content_json)
    assert [starter["name"] for starter in content["character_starters"]] == [
        "Captain Ilyra",
        "Brother Senn",
        "Vey the outrider",
    ]
    characters = repositories.list_characters(active_save_id)
    character_names = {character.name for character in characters}
    assert character_names == {
        "Mara Voss",
        "Captain Ilyra",
        "Brother Senn",
        "Vey the outrider",
    }
    player = next(
        character for character in characters if character.name == "Mara Voss"
    )
    assert player.is_player_character is True
    assert all(character.protected_from_maintenance for character in characters)
    identity_locks = {
        "name",
        "aliases",
        "role",
        "known_state",
        "appearance",
        "visual_notes",
        "personality",
        "voice",
    }
    assert all(
        identity_locks <= set(character.locked_fields)
        for character in characters
        if not character.is_player_character
    )
    assert all("status" not in character.locked_fields for character in characters)
    assert all(
        "relationships" not in character.locked_fields for character in characters
    )
    vey = next(
        character for character in characters if character.name == "Vey the outrider"
    )
    assert vey.aliases == ["Vey"]
    assert all(character.save_id == active_save_id for character in characters)


def test_save_scenario_draft_persists_reviewed_investigation_mystery_sections(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = asyncio.run(
        _save_scenario_draft(
            controller,
            scenario_type="investigation_mystery",
            sections=_reviewed_investigation_mystery_sections(),
            save_title="Reviewed Mystery Save",
            request_initial_image=False,
        )
    )

    active_save_id = _value(model, "active_save_id")
    save = repositories.get_save(active_save_id)
    assert save is not None
    scenario = repositories.get_scenario(save.scenario_id)
    assert scenario is not None
    assert scenario.type == "investigation_mystery"
    content = json.loads(scenario.content_json)
    assert content["case_facts"] == (
        "Curator Elian Vale vanished from the sealed east gallery during a gala."
    )
    assert content["clues"] == (
        "Broken display dust found outside the gallery door; undiscovered. "
        "Watch log gap from 9:10 to 9:18; reliable and tied to Sera's alibi."
    )
    assert content["hidden_truth"] == (
        "Sera staged the vanishing to hide a smuggling ledger in the restoration lift."
    )
    assert repositories.list_messages(active_save_id)[0].body == (
        "Rain taps the museum glass as the east gallery unlocks."
    )


def test_save_scenario_draft_seeds_dating_sim_player_and_explicit_starters(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    provider = RuntimeStructuredCleanupProvider([])
    controller = _runtime_controller(runtime, repositories, tmp_path, provider=provider)
    reviewed_sections = _reviewed_dating_sim_sections()
    character_starters: list[dict[str, object]] = [
        {
            "name": "Mika Arai",
            "role": "Female class president and precise festival organizer.",
            "known_state": (
                "Mika Arai is ambitious, secretly lonely, and drawn to Ren's patience."
            ),
            "appearance": "Sharp blazer, neat braid, and a silver student council pin.",
            "visual_notes": (
                "Always carrying a festival clipboard marked in three ink colors."
            ),
            "personality": "Disciplined, lonely, and quietly protective.",
            "voice": "Crisp formal phrasing that softens when surprised.",
            "goals": "Keep the festival schedule from collapsing.",
            "motivations": "Prove she can be trusted beyond student council work.",
            "boundaries": "Will not let Ren take blame for her mistakes.",
            "relationships": {
                "Ren Takahashi": "romance option for Ren Takahashi",
            },
            "status": "available romance option at scenario start",
        },
        {
            "name": "Sora Minase",
            "role": "Female swimmer with bright competitive energy.",
            "known_state": "Sora is terrified of leaving home after summer.",
            "appearance": "Sun-browned shoulders and wind-tossed hair.",
            "visual_notes": "Always smells faintly of pool water.",
            "personality": "Bright, competitive, and anxious.",
            "voice": "Fast, teasing, and earnest under pressure.",
            "goals": "Win the summer meet before leaving home.",
            "motivations": "Prove she can choose her own next step.",
            "boundaries": "Will not be pushed to abandon her team.",
            "relationships": {
                "Ren Takahashi": "romance option for Ren Takahashi",
            },
            "status": "available romance option at scenario start",
        },
        {
            "name": "Yuna Kisaragi",
            "role": "Female art-club dreamer drawn to storms.",
            "known_state": "Yuna sketches Ren during a rain delay.",
            "appearance": "Paint-stained fingers and storm-gray eyes.",
            "visual_notes": "Keeps a sketchbook hugged to her chest.",
            "personality": "Gentle, strange, and observant.",
            "voice": "Soft, elliptical, and vivid.",
            "goals": "Capture the storm festival in a final sketch.",
            "motivations": "Find someone who sees the world slantwise.",
            "boundaries": "Will not let her art be mocked as childish.",
            "relationships": {
                "Ren Takahashi": "romance option for Ren Takahashi",
            },
            "status": "available romance option at scenario start",
        },
        {
            "name": "Nozomi Vale",
            "role": "Female childhood friend tired of being overlooked.",
            "known_state": "Nozomi knows Ren well enough to call out his evasions.",
            "appearance": "Short dark curls and a crooked grin.",
            "visual_notes": "Wears a worn festival volunteer sash.",
            "personality": "Funny, guarded, and loyal.",
            "voice": "Dry jokes with sudden emotional honesty.",
            "goals": "Make Ren stop dodging their shared history.",
            "motivations": "Protect old loyalty without disappearing inside it.",
            "boundaries": "Will not be treated as a fallback choice.",
            "relationships": {
                "Ren Takahashi": "romance option for Ren Takahashi",
            },
            "status": "available romance option at scenario start",
        },
    ]

    model = asyncio.run(
        _save_scenario_draft(
            controller,
            scenario_type="dating_sim",
            sections=reviewed_sections,
            character_starters=character_starters,
            save_title="Reviewed Dating Save",
            request_initial_image=False,
        )
    )

    active_save_id = _value(model, "active_save_id")
    characters = repositories.list_characters(active_save_id)
    assert {character.name for character in characters} == {
        "Ren Takahashi",
        "Mika Arai",
        "Sora Minase",
        "Yuna Kisaragi",
        "Nozomi Vale",
    }
    player = next(
        character for character in characters if character.name == "Ren Takahashi"
    )
    assert player.is_player_character is True
    assert "thoughtful male transfer student" in player.known_state
    mika = next(character for character in characters if character.name == "Mika Arai")
    assert mika.protected_from_maintenance is True
    assert mika.relationships == {
        "Ren Takahashi": "romance option for Ren Takahashi"
    }
    routes = repositories.list_dating_route_states(active_save_id)
    route_names = {
        next(
            character.name
            for character in characters
            if character.id == route.npc_character_id
        )
        for route in routes
    }
    assert route_names == {"Mika Arai", "Sora Minase", "Yuna Kisaragi", "Nozomi Vale"}
    assert all(route.stage == "introduced" for route in routes)
    assert all(
        route.next_reasonable_step == "build early interest or exchange contact info"
        for route in routes
    )
    assert mika.status == "available romance option at scenario start"
    assert mika.role == (
        "Female class president and precise festival organizer."
    )
    assert mika.known_state == (
        "Mika Arai is ambitious, secretly lonely, and drawn to Ren's patience."
    )
    assert mika.appearance == (
        "Sharp blazer, neat braid, and a silver student council pin."
    )
    assert mika.visual_notes == (
        "Always carrying a festival clipboard marked in three ink colors."
    )
    assert mika.personality == "Disciplined, lonely, and quietly protective."
    assert mika.voice == "Crisp formal phrasing that softens when surprised."
    assert mika.goals == "Keep the festival schedule from collapsing."
    assert mika.motivations == (
        "Prove she can be trusted beyond student council work."
    )
    assert mika.boundaries == "Will not let Ren take blame for her mistakes."
    identity_locks = {
        "name",
        "aliases",
        "role",
        "known_state",
        "appearance",
        "visual_notes",
        "personality",
        "voice",
    }
    assert identity_locks <= set(mika.locked_fields)
    assert {"goals", "motivations", "boundaries"} <= set(mika.locked_fields)
    assert "relationships" not in mika.locked_fields
    assert "status" not in mika.locked_fields
    save = repositories.get_save(active_save_id)
    assert save is not None
    scenario = repositories.get_scenario(save.scenario_id)
    assert scenario is not None
    content = json.loads(scenario.content_json)
    starter = next(
        item for item in content["character_starters"] if item["name"] == "Mika Arai"
    )
    assert starter["appearance"] == mika.appearance
    assert starter["personality"] == mika.personality
    assert starter["goals"] == mika.goals
    assert starter["motivations"] == mika.motivations
    assert starter["boundaries"] == mika.boundaries
    assert provider.structured_requests == []


def test_save_hybrid_science_fiction_dating_sim_draft_seeds_both_genres(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)
    reviewed_sections = {
        **_reviewed_dating_sim_sections(),
        "title": "Orbital Hearts",
        "premise": "A courier enters an orbital academy during a diplomatic crisis.",
        "technology_level": "Jump ships are common, but station AIs ration access.",
        "setting_scope": "A disputed academy station above a treaty moon.",
        "species_and_intelligences": "Human students, uplift envoys, and station AIs.",
        "factions_and_institutions": "The academy council and rival charter fleets.",
        "mission_stakes": "Keep the treaty delegation alive through festival week.",
    }

    model = asyncio.run(
        _save_scenario_draft(
            controller,
            scenario_type="science_fiction_roleplay",
            scenario_types=("science_fiction_roleplay", "dating_sim"),
            sections=reviewed_sections,
            save_title="Orbital Hearts",
            request_initial_image=False,
        )
    )

    active_save_id = _value(model, "active_save_id")
    save = repositories.get_save(active_save_id)
    assert save is not None
    scenario = repositories.get_scenario(save.scenario_id)
    assert scenario is not None
    assert scenario.type == "science_fiction_roleplay"
    content = json.loads(scenario.content_json)
    assert content["_scenario_genres"] == [
        "science_fiction_roleplay",
        "dating_sim",
    ]
    assert content["technology_level"] == reviewed_sections["technology_level"]

    characters = repositories.list_characters(active_save_id)
    assert {character.name for character in characters} == {"Ren Takahashi"}
    routes = repositories.list_dating_route_states(active_save_id)
    assert routes == []


def test_runtime_completes_sparse_character_profiles_without_overwriting_user_fields(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    repositories.set_model_preference(
        task="context_update",
        provider="fake",
        model_id="fake-structured",
    )
    _save_fake_provider_model(
        repositories,
        model_id="fake-structured",
        capabilities=["structured_output"],
    )
    provider = RuntimeStructuredCleanupProvider(
        [
            {
                "characters": [
                    {
                        "name": "Mara",
                        "aliases": ["Red Signal"],
                        "role": "Generated role should not replace user role.",
                        "appearance": "Red signal cloak and salt-stained boots.",
                        "visual_notes": "A fast silhouette against beacon flare.",
                        "personality": "Restless and brave.",
                        "voice": "Quick, clipped field reports.",
                        "relationships": {"Captain Ilyra": "trusted ally"},
                        "status": "present at the beacon",
                    }
                ]
            }
        ]
    )
    controller = _runtime_controller(runtime, repositories, tmp_path, provider=provider)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower at the edge of a storm sea.",
        player_role="Keeper",
        content={"opening_message": "The beacon snaps awake."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    character = repositories.add_character(
        save_id=save.id,
        name="Mara",
        role="Signal runner",
    )

    updated_count = controller.complete_sparse_character_profiles(
        active_save_id=save.id,
        character_ids=(character.id,),
    )

    assert updated_count == 1
    saved = repositories.get_character(character.id)
    assert saved is not None
    assert saved.role == "Signal runner"
    assert saved.aliases == ["Red Signal"]
    assert saved.appearance == "Red signal cloak and salt-stained boots."
    assert saved.visual_notes == "A fast silhouette against beacon flare."
    assert saved.personality == "Restless and brave."
    assert saved.voice == "Quick, clipped field reports."
    assert saved.relationships == {"Captain Ilyra": "trusted ally"}
    assert saved.status == "present at the beacon"
    assert {
        "aliases",
        "appearance",
        "visual_notes",
        "personality",
        "voice",
    } <= set(saved.locked_fields)
    assert "role" not in saved.locked_fields
    assert "relationships" not in saved.locked_fields
    assert "status" not in saved.locked_fields
    assert [request.schema_name for request in provider.structured_requests] == [
        "character_profile_completion"
    ]


def test_runtime_completes_new_character_agency_without_overwriting_user_fields(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    repositories.set_model_preference(
        task="context_update",
        provider="fake",
        model_id="fake-structured",
    )
    _save_fake_provider_model(
        repositories,
        model_id="fake-structured",
        capabilities=["structured_output"],
    )
    provider = RuntimeStructuredCleanupProvider(
        [
            {
                "characters": [
                    {
                        "name": "Mara",
                        "goals": "Generated goal should not replace user goal.",
                        "motivations": "Protect the lower village from ash riders.",
                        "boundaries": (
                            "Will not leave the tower while the lens is unstable."
                        ),
                    }
                ]
            }
        ]
    )
    controller = _runtime_controller(runtime, repositories, tmp_path, provider=provider)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower at the edge of a storm sea.",
        player_role="Keeper",
        content={"opening_message": "The beacon snaps awake."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    character = repositories.add_character(
        save_id=save.id,
        name="Mara",
        role="Signal runner",
        goals="Keep the beacon lit.",
    )

    updated_count = controller.complete_new_character_agency(
        active_save_id=save.id,
        character_ids=(character.id,),
    )

    assert updated_count == 1
    saved = repositories.get_character(character.id)
    assert saved is not None
    assert saved.goals == "Keep the beacon lit."
    assert saved.motivations == "Protect the lower village from ash riders."
    assert saved.boundaries == "Will not leave the tower while the lens is unstable."
    assert "goals" not in saved.locked_fields
    assert {"motivations", "boundaries"} <= set(saved.locked_fields)
    assert [request.schema_name for request in provider.structured_requests] == [
        "character_profile_completion"
    ]


def test_runtime_enhances_character_text_field_and_locks_generated_canon(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    events: list[tuple[str, dict[str, object]]] = []

    def capture_log_event(event: str, **fields: object) -> None:
        events.append((event, dict(fields)))

    monkeypatch.setattr(runtime, "log_event", capture_log_event)
    repositories.set_model_preference(
        task="context_update",
        provider="fake",
        model_id="fake-structured",
    )
    _save_fake_provider_model(
        repositories,
        model_id="fake-structured",
        capabilities=["structured_output"],
    )
    provider = RuntimeStructuredCleanupProvider(
        [
            {
                "field_name": "appearance",
                "character": {
                    "name": "Mara",
                    "appearance": (
                        "Salt-stained boots and a copper lens-key on a black cord."
                    ),
                },
            }
        ]
    )
    controller = _runtime_controller(runtime, repositories, tmp_path, provider=provider)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower at the edge of a storm sea.",
        player_role="Keeper",
        content={"opening_message": "The beacon snaps awake."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    character = repositories.add_character(
        save_id=save.id,
        name="Mara",
        role="Signal runner",
        appearance="A red signal cloak.",
        locked_fields=["voice"],
    )
    rows = controller.build_character_registry_model(
        active_save_id=save.id,
    ).characters
    row = next(item for item in rows if item.character_id == character.id)

    result = controller.enhance_character_registry_field(
        active_save_id=save.id,
        character_id=character.id,
        field_name="appearance",
        row=replace(row, role="Beacon courier", locked_fields=()),
    )

    assert result.updated_count == 1
    assert result.field_changed is True
    assert result.notice is None
    saved = repositories.get_character(character.id)
    assert saved is not None
    assert saved.role == "Beacon courier"
    assert saved.appearance == (
        "A red signal cloak.\n\n"
        "Salt-stained boots and a copper lens-key on a black cord."
    )
    assert "appearance" in saved.locked_fields
    assert "voice" not in saved.locked_fields
    assert [request.schema_name for request in provider.structured_requests] == [
        "character_field_enhancement"
    ]
    assert events[-1] == (
        "runtime.character_field_enhancement_completed",
        {
            "save_id": save.id,
            "character_id": character.id,
            "provider": "fake",
            "model": "fake-structured",
            "field_name": "appearance",
            "field_changed": True,
            "created_count": 0,
            "updated_count": 1,
            "archived_count": 0,
        },
    )


def test_runtime_enhances_character_texting_style_and_locks_generated_canon(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    repositories.set_model_preference(
        task="context_update",
        provider="fake",
        model_id="fake-structured",
    )
    _save_fake_provider_model(
        repositories,
        model_id="fake-structured",
        capabilities=["structured_output"],
    )
    provider = RuntimeStructuredCleanupProvider(
        [
            {
                "field_name": "texting_style",
                "character": {
                    "name": "Mara",
                    "texting_style": (
                        "Lowercase bursts, double texts when worried, and one "
                        "sparkle emoji at most."
                    ),
                },
            }
        ]
    )
    controller = _runtime_controller(runtime, repositories, tmp_path, provider=provider)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower at the edge of a storm sea.",
        player_role="Keeper",
        content={"opening_message": "The beacon snaps awake."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    character = repositories.add_character(
        save_id=save.id,
        name="Mara",
        role="Signal runner",
        texting_style="Short replies after midnight.",
        locked_fields=["voice"],
    )
    row = next(
        item
        for item in controller.build_character_registry_model(
            active_save_id=save.id,
        ).characters
        if item.character_id == character.id
    )

    result = controller.enhance_character_registry_field(
        active_save_id=save.id,
        character_id=character.id,
        field_name="texting_style",
        row=replace(row, role="Beacon courier", locked_fields=()),
    )

    assert result.updated_count == 1
    assert result.field_changed is True
    saved = repositories.get_character(character.id)
    assert saved is not None
    assert saved.role == "Beacon courier"
    assert saved.texting_style == (
        "Short replies after midnight.\n\n"
        "Lowercase bursts, double texts when worried, and one sparkle emoji at most."
    )
    assert "texting_style" in saved.locked_fields
    assert "voice" not in saved.locked_fields
    assert [request.schema_name for request in provider.structured_requests] == [
        "character_field_enhancement"
    ]
    request = provider.structured_requests[0]
    assert "texting_style" in str(request.schema)
    assert "Target field: texting_style (texting style)" in request.messages[-1].body


def test_runtime_character_enhancement_uses_separate_model_preference(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    repositories.save_provider_model(
        provider="context",
        model_id="context-structured",
        display_name="Context Structured",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    repositories.save_provider_model(
        provider="enhance",
        model_id="enhance-structured",
        display_name="Enhancement Structured",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    repositories.set_model_preference(
        task="context_update",
        provider="context",
        model_id="context-structured",
    )
    repositories.set_model_preference(
        task="character_enhancement",
        provider="enhance",
        model_id="enhance-structured",
    )
    context_provider = RuntimeStructuredCleanupProvider([])
    enhancement_provider = RuntimeDualCharacterEnhancementProvider(
        {
            "field_name": "appearance",
            "character": {
                "name": "Mara",
                "appearance": "Copper lens-key on a black cord.",
            },
        }
    )
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        providers={
            "context": context_provider,
            "enhance": enhancement_provider,
        },
    )
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower at the edge of a storm sea.",
        player_role="Keeper",
        content={"opening_message": "The beacon snaps awake."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    character = repositories.add_character(
        save_id=save.id,
        name="Mara",
        appearance="A red signal cloak.",
    )
    row = next(
        item
        for item in controller.build_character_registry_model(
            active_save_id=save.id,
        ).characters
        if item.character_id == character.id
    )

    controller.enhance_character_registry_field(
        active_save_id=save.id,
        character_id=character.id,
        field_name="appearance",
        row=row,
    )

    assert context_provider.structured_requests == []
    assert len(enhancement_provider.structured_requests) == 1
    request = enhancement_provider.structured_requests[0]
    assert request.provider == "enhance"
    assert request.model_id == "enhance-structured"


def test_runtime_character_enhancement_prefers_structured_output_for_dual_models(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-both",
        display_name="Fake Structured And Tool",
        capabilities=[
            ProviderCapability.STRUCTURED_OUTPUT.value,
            ProviderCapability.TOOL_CALLING.value,
        ],
    )
    repositories.set_model_preference(
        task="context_update",
        provider="fake",
        model_id="fake-both",
    )
    repositories.set_model_preference(
        task="character_enhancement",
        provider="fake",
        model_id="fake-both",
    )
    provider = RuntimeDualCharacterEnhancementProvider(
        {
            "field_name": "appearance",
            "character": {
                "name": "Mara",
                "appearance": "Copper lens-key on a black cord.",
            },
        }
    )
    controller = _runtime_controller(runtime, repositories, tmp_path, provider=provider)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower at the edge of a storm sea.",
        player_role="Keeper",
        content={"opening_message": "The beacon snaps awake."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    character = repositories.add_character(
        save_id=save.id,
        name="Mara",
        appearance="A red signal cloak.",
    )
    row = next(
        item
        for item in controller.build_character_registry_model(
            active_save_id=save.id,
        ).characters
        if item.character_id == character.id
    )

    controller.enhance_character_registry_field(
        active_save_id=save.id,
        character_id=character.id,
        field_name="appearance",
        row=row,
    )

    assert len(provider.structured_requests) == 1
    assert provider.tool_requests == []


def test_runtime_character_enhancement_uses_person_scoped_context(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    repositories.set_model_preference(
        task="context_update",
        provider="fake",
        model_id="fake-structured",
    )
    _save_fake_provider_model(
        repositories,
        model_id="fake-structured",
        capabilities=["structured_output"],
    )
    provider = RuntimeStructuredCleanupProvider(
        [
            {
                "field_name": "visual_notes",
                "character": {
                    "name": "Emma Vale",
                    "visual_notes": (
                        "Grid-ruled notebook always open beside the laptop."
                    ),
                },
            }
        ]
    )
    controller = _runtime_controller(runtime, repositories, tmp_path, provider=provider)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="East Wing Study",
        premise="Civil engineering freshmen study late in the old library.",
        player_role="Chris",
        content={
            "opening_message": "Emma waves Chris toward the window seat.",
            "current_scene": "The east wing library is quiet after midnight.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Library Night")
    player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Chris",
        body="I sit at Emma's window table.",
    )
    narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Emma taps her pencil and points to the one interruption per hour rule.",
    )
    location = repositories.add_location(
        save_id=save.id,
        name="East Wing Library",
        description="A late-night study room with a tall window seat.",
        visual_description="Green lamps, rain-streaked glass, and stacked journals.",
        location_id="location-east-wing",
    )
    character = repositories.add_character(
        save_id=save.id,
        name="Emma Vale",
        aliases=["Em"],
        role="Civil engineering freshman",
        age="18",
        known_state="Invited Chris to her regular window study spot.",
        met=True,
        appearance="Medium brown shoulder-length hair, pale skin, glasses.",
        visual_notes=(
            "Constantly pushes glasses up; bundled in layers; carries a "
            "battered laptop with engineering stickers."
        ),
        personality="Quiet, precise, observant, and patient.",
        voice="Soft, exact, and dryly funny once comfortable.",
        relationships={"Chris": "trusted library study partner"},
        goals="Finish the structural analysis problem set before dawn.",
        motivations="Wants to prove she belongs in the engineering program.",
        current_intent="Keep Chris focused without making the study session awkward.",
        boundaries="One interruption per hour unless the building is on fire.",
        attitude_toward_player="Cautiously warm toward Chris.",
        cooperation_conditions="Helps when Chris respects her study rules.",
        status="Studying in the east wing.",
        location_id=location.id,
        private_notes="Secretly worried her scholarship renewal is fragile.",
        contact_name="Em",
        character_id="character-emma",
    )
    other_character = repositories.add_character(
        save_id=save.id,
        name="Mara",
        role="Unrelated signal runner",
        character_id="character-mara",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
        situation="Emma and Chris share the library window seat.",
        present_character_ids=[character.id],
        source_message_id=narrator.id,
    )
    linked_memory = repositories.add_memory(
        save_id=save.id,
        body="Emma reserves the east wing window seat for hard problem sets.",
        tags=["emma", "study"],
        source_message_id=narrator.id,
        memory_id="memory-emma-window-seat",
    )
    graph_memory = repositories.add_memory(
        save_id=save.id,
        body="Emma told Chris the one interruption per hour rule.",
        tags=["emma", "boundary"],
        source_message_id=narrator.id,
        memory_id="memory-emma-interruption-rule",
    )
    unrelated_memory = repositories.add_memory(
        save_id=save.id,
        body="Mara keeps a signal horn under her coat.",
        tags=["mara"],
        source_message_id=narrator.id,
        memory_id="memory-mara-horn",
    )
    low_confidence_memory = repositories.add_memory(
        save_id=save.id,
        body="Low-confidence rumor that Emma owns a racing motorcycle.",
        tags=["emma", "rumor"],
        source_message_id=narrator.id,
        memory_id="memory-emma-rumor",
    )
    linked_state = repositories.upsert_world_state(
        save_id=save.id,
        key="library.window_seat",
        value={"reserved_by": "Emma", "rule": "one interruption per hour"},
        category="location",
        source_message_id=narrator.id,
        state_id="state-library-window-seat",
    )
    linked_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=player.id,
        covers_message_end_id=narrator.id,
        body="Emma invited Chris to her east wing study spot and set a firm rule.",
        provider="fake",
        model="fake-summary",
        summary_id="summary-emma-study",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=character.id,
        target_type="memory",
        target_id=linked_memory.id,
        relation="knows",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=character.id,
        target_type="world_state",
        target_id=linked_state.id,
        relation="knows",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=character.id,
        target_type="summary",
        target_id=linked_summary.id,
        relation="knows",
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=character.id,
        target_type="memory",
        target_id=graph_memory.id,
        knowledge_state="may_know",
        acquisition_method="told",
        confidence=0.85,
        source_message_id=narrator.id,
        evidence_quote="one interruption per hour rule",
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=character.id,
        target_type="memory",
        target_id=low_confidence_memory.id,
        knowledge_state="may_know",
        acquisition_method="inferred_from_visible_consequence",
        confidence=0.2,
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=other_character.id,
        target_type="memory",
        target_id=unrelated_memory.id,
        knowledge_state="knows",
        acquisition_method="witnessed",
        confidence=1.0,
    )
    row = next(
        item
        for item in controller.build_character_registry_model(
            active_save_id=save.id,
        ).characters
        if item.character_id == character.id
    )

    controller.enhance_character_registry_field(
        active_save_id=save.id,
        character_id=character.id,
        field_name="visual_notes",
        row=row,
    )

    saved = repositories.get_character(character.id)
    assert saved is not None
    assert saved.visual_notes == (
        "Constantly pushes glasses up; bundled in layers; carries a "
        "battered laptop with engineering stickers.\n\n"
        "Grid-ruled notebook always open beside the laptop."
    )
    assert len(provider.structured_requests) == 1
    prompt_text = "\n".join(
        message.body for message in provider.structured_requests[0].messages
    )
    assert "contact_name: Em" in prompt_text
    assert "private_notes: Secretly worried her scholarship renewal is fragile." in (
        prompt_text
    )
    assert "met: true" in prompt_text
    assert "present: true" in prompt_text
    assert "goals: Finish the structural analysis problem set before dawn." in (
        prompt_text
    )
    assert "motivations: Wants to prove she belongs in the engineering program." in (
        prompt_text
    )
    assert "Selected location:" in prompt_text
    assert "East Wing Library" in prompt_text
    assert "Green lamps, rain-streaked glass, and stacked journals." in prompt_text
    assert "Emma reserves the east wing window seat for hard problem sets." in (
        prompt_text
    )
    assert "library.window_seat" in prompt_text
    assert "Emma invited Chris to her east wing study spot" in prompt_text
    assert "Emma told Chris the one interruption per hour rule." in prompt_text
    assert "one interruption per hour rule" in prompt_text
    assert "Mara keeps a signal horn under her coat." not in prompt_text
    assert "Low-confidence rumor that Emma owns a racing motorcycle." not in prompt_text


def test_runtime_noop_character_text_enhancement_commits_row_edits_without_lock(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    events: list[tuple[str, dict[str, object]]] = []

    def capture_log_event(event: str, **fields: object) -> None:
        events.append((event, dict(fields)))

    monkeypatch.setattr(runtime, "log_event", capture_log_event)
    repositories.set_model_preference(
        task="context_update",
        provider="fake",
        model_id="fake-structured",
    )
    _save_fake_provider_model(
        repositories,
        model_id="fake-structured",
        capabilities=["structured_output"],
    )
    provider = RuntimeStructuredCleanupProvider(
        [
            {
                "field_name": "appearance",
                "character": {
                    "name": "Mara",
                    "appearance": "A red signal cloak",
                },
            }
        ]
    )
    controller = _runtime_controller(runtime, repositories, tmp_path, provider=provider)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower at the edge of a storm sea.",
        player_role="Keeper",
        content={"opening_message": "The beacon snaps awake."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    character = repositories.add_character(
        save_id=save.id,
        name="Mara",
        role="Signal runner",
        appearance="A red signal cloak.",
        locked_fields=["voice"],
    )
    row = next(
        item
        for item in controller.build_character_registry_model(
            active_save_id=save.id,
        ).characters
        if item.character_id == character.id
    )

    result = controller.enhance_character_registry_field(
        active_save_id=save.id,
        character_id=character.id,
        field_name="appearance",
        row=replace(row, role="Beacon courier", locked_fields=()),
    )

    assert result.updated_count == 1
    assert result.field_changed is False
    assert result.notice == (
        "No new Appearance details were found; the field was left unchanged."
    )
    saved = repositories.get_character(character.id)
    assert saved is not None
    assert saved.role == "Beacon courier"
    assert saved.appearance == "A red signal cloak."
    assert "appearance" not in saved.locked_fields
    assert "voice" not in saved.locked_fields
    assert events[-1] == (
        "runtime.character_field_enhancement_completed",
        {
            "save_id": save.id,
            "character_id": character.id,
            "provider": "fake",
            "model": "fake-structured",
            "field_name": "appearance",
            "field_changed": False,
            "created_count": 0,
            "updated_count": 1,
            "archived_count": 0,
        },
    )


def test_runtime_enhances_character_relationships_without_dropping_existing(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    repositories.set_model_preference(
        task="context_update",
        provider="fake",
        model_id="fake-structured",
    )
    _save_fake_provider_model(
        repositories,
        model_id="fake-structured",
        capabilities=["structured_output"],
    )
    provider = RuntimeStructuredCleanupProvider(
        [
            {
                "field_name": "relationships",
                "character": {
                    "name": "Mara",
                    "relationships": {
                        "Captain Ilyra": "wary ally after the beacon failed",
                        "Bell Keeper": "owes them a dangerous favor",
                    },
                },
            }
        ]
    )
    controller = _runtime_controller(runtime, repositories, tmp_path, provider=provider)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower at the edge of a storm sea.",
        player_role="Keeper",
        content={"opening_message": "The beacon snaps awake."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    character = repositories.add_character(
        save_id=save.id,
        name="Mara",
        relationships={"Captain Ilyra": "trusted contact"},
    )
    rows = controller.build_character_registry_model(
        active_save_id=save.id,
    ).characters
    row = next(item for item in rows if item.character_id == character.id)

    controller.enhance_character_registry_field(
        active_save_id=save.id,
        character_id=character.id,
        field_name="relationships",
        row=row,
    )

    saved = repositories.get_character(character.id)
    assert saved is not None
    assert saved.relationships == {
        "Bell Keeper": "owes them a dangerous favor",
        "Captain Ilyra": (
            "trusted contact\n\n"
            "wary ally after the beacon failed"
        ),
    }
    assert "relationships" in saved.locked_fields


def test_runtime_enhances_agency_field_with_broad_source_labeled_context(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    repositories.set_model_preference(
        task="context_update",
        provider="fake",
        model_id="fake-structured",
    )
    _save_fake_provider_model(
        repositories,
        model_id="fake-structured",
        capabilities=["structured_output"],
    )
    provider = RuntimeStructuredCleanupProvider(
        [
            {
                "field_name": "current_intent",
                "character": {
                    "name": "Mara",
                    "current_intent": (
                        "Demand proof before sharing the red lens failsafe."
                    ),
                    "evidence_source_ids": ["memory:memory-ilyra-failsafe"],
                },
            }
        ]
    )
    controller = _runtime_controller(runtime, repositories, tmp_path, provider=provider)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower at the edge of a storm sea.",
        player_role="Keeper",
        content={"opening_message": "The beacon snaps awake."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Keeper",
        body="I ask Mara for the red lens failsafe.",
    )
    narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Mara demands proof before sharing the red lens failsafe.",
    )
    location = repositories.add_location(
        save_id=save.id,
        name="Beacon Gallery",
        description="A red glass watch room.",
        source_message_id=narrator.id,
        location_id="location-gallery",
    )
    character = repositories.add_character(
        save_id=save.id,
        name="Mara",
        role="Signal runner",
        current_intent="Guard the lens stair.",
        source_message_id=player.id,
        character_id="character-mara",
        location_id=location.id,
    )
    other_character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        role="Watch captain",
        source_message_id=player.id,
        character_id="character-ilyra",
    )
    snapshot = repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
        situation="Mara blocks the lens stair.",
        present_character_ids=[character.id, other_character.id],
        source_message_id=narrator.id,
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="Mara refuses to share the red lens failsafe without proof.",
        tags=["agency"],
        source_message_id=narrator.id,
        memory_id="memory-ilyra-failsafe",
    )
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="beacon.red_lens",
        value={"status": "unstable"},
        category="scene",
        source_message_id=narrator.id,
        state_id="state-red-lens",
    )
    summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=player.id,
        covers_message_end_id=narrator.id,
        body="Mara challenged the keeper to prove the lens could hold.",
        provider="fake",
        model="fake-summary",
        summary_id="summary-red-lens",
    )
    observation = repositories.add_context_observation(
        save_id=save.id,
        observation_type="character_intent",
        claim="Mara wants proof before sharing the failsafe.",
        evidence_quote="demands proof",
        source_message_ids=[narrator.id],
        status="accepted",
        observation_id="observation-mara-intent",
    )
    context_source = repositories.upsert_context_source(
        save_id=save.id,
        source_type="memory",
        source_id=memory.id,
        title="Mara intent",
        body="Mara's cooperation depends on proof.",
        context_source_id="context-source-mara-intent",
    )
    suggestion = repositories.add_context_update_suggestion(
        save_id=save.id,
        update_type="update",
        entity_type="character",
        entity_id=character.id,
        field_path="cooperation_conditions",
        proposed_value="Helps after seeing proof.",
        source_message_ids=[narrator.id],
    )
    rows = controller.build_character_registry_model(
        active_save_id=save.id,
    ).characters
    row = next(item for item in rows if item.character_id == character.id)

    controller.enhance_character_registry_field(
        active_save_id=save.id,
        character_id=character.id,
        field_name="current_intent",
        row=row,
    )

    saved = repositories.get_character(character.id)
    assert saved is not None
    assert saved.current_intent == (
        "Guard the lens stair.\n\n"
        "Demand proof before sharing the red lens failsafe."
    )
    assert "current_intent" in saved.locked_fields
    assert [request.schema_name for request in provider.structured_requests] == [
        "character_field_enhancement"
    ]
    request = provider.structured_requests[0]
    prompt_text = "\n".join(message.body for message in request.messages)
    assert f"[scenario:{scenario.id}]" in prompt_text
    assert f"[character:{character.id}]" in prompt_text
    assert f"[scene:{snapshot.id}]" in prompt_text
    assert f"[location:{location.id}]" in prompt_text
    assert f"[character:{other_character.id}]" in prompt_text
    assert f"[memory:{memory.id}]" in prompt_text
    assert f"[world_state:{state.id}]" in prompt_text
    assert f"[summary:{summary.id}]" in prompt_text
    assert f"[observation:{observation.id}]" in prompt_text
    assert f"[context_source:{context_source.id}]" in prompt_text
    assert f"[suggestion:{suggestion.id}]" in prompt_text
    assert f"[message:{player.id}]" in prompt_text
    assert f"[message:{narrator.id}]" in prompt_text


def test_runtime_rejects_agency_enhancement_without_evidence(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    error_events: list[tuple[str, dict[str, object]]] = []

    def capture_log_error_event(event: str, **fields: object) -> None:
        error_events.append((event, dict(fields)))

    monkeypatch.setattr(runtime, "log_error_event", capture_log_error_event)
    repositories.set_model_preference(
        task="context_update",
        provider="fake",
        model_id="fake-structured",
    )
    _save_fake_provider_model(
        repositories,
        model_id="fake-structured",
        capabilities=["structured_output"],
    )
    provider = RuntimeStructuredCleanupProvider(
        [
            {
                "field_name": "boundaries",
                "character": {
                    "name": "Mara",
                    "boundaries": "Will not abandon the tower.",
                },
            },
            {
                "field_name": "boundaries",
                "character": {
                    "name": "Mara",
                    "boundaries": "Will not abandon the tower.",
                },
            },
            {
                "field_name": "boundaries",
                "character": {
                    "name": "Mara",
                    "boundaries": "Will not abandon the tower.",
                },
            },
        ]
    )
    controller = _runtime_controller(runtime, repositories, tmp_path, provider=provider)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower at the edge of a storm sea.",
        player_role="Keeper",
        content={"opening_message": "The beacon snaps awake."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="Mara refuses to leave the tower.",
    )
    character = repositories.add_character(
        save_id=save.id,
        name="Mara",
        character_id="character-mara",
    )
    repositories.add_memory(
        save_id=save.id,
        body="Mara refuses to leave the tower.",
        tags=["agency"],
        source_message_id=narrator.id,
        memory_id="memory-mara-boundary",
    )
    row = next(
        item
        for item in controller.build_character_registry_model(
            active_save_id=save.id,
        ).characters
        if item.character_id == character.id
    )

    with pytest.raises(ValueError, match="evidence_source_ids"):
        controller.enhance_character_registry_field(
            active_save_id=save.id,
            character_id=character.id,
            field_name="boundaries",
            row=row,
        )

    saved = repositories.get_character(character.id)
    assert saved is not None
    assert saved.boundaries == ""
    assert "boundaries" not in saved.locked_fields
    assert error_events[-1][0] == "runtime.character_field_enhancement_failed"
    assert error_events[-1][1]["provider"] == "fake"
    assert error_events[-1][1]["model"] == "fake-structured"


def test_runtime_enhancement_rejects_destructive_character_rows(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower at the edge of a storm sea.",
        player_role="Keeper",
        content={"opening_message": "The beacon snaps awake."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Lantern Keep")
    character = repositories.add_character(save_id=save.id, name="Mara")
    merge_target = repositories.add_character(save_id=save.id, name="Ilyra")
    destructive_rows = controller.build_character_registry_model(
        active_save_id=save.id,
    ).characters
    row = next(item for item in destructive_rows if item.character_id == character.id)

    with pytest.raises(ValueError, match="cannot be archived"):
        controller.enhance_character_registry_field(
            active_save_id=save.id,
            character_id=character.id,
            field_name="appearance",
            row=replace(row, archived=True),
        )

    with pytest.raises(ValueError, match="Clear merge target"):
        controller.enhance_character_registry_field(
            active_save_id=save.id,
            character_id=character.id,
            field_name="appearance",
            row=replace(row, merge_into_character_id=merge_target.id),
        )


def test_save_scenario_draft_with_initial_image_failure_keeps_created_save(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    repositories.set_model_preference(
        task="chat",
        provider="chat-provider",
        model_id="chat-model",
    )
    repositories.set_model_preference(
        task="image_generation",
        provider="image-provider",
        model_id="image-model",
    )
    chat_provider = RuntimeFakeProvider()
    image_provider = FailingRuntimeImageProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        providers={
            "chat-provider": chat_provider,
            "image-provider": image_provider,
        },
    )

    model = asyncio.run(
        _save_scenario_draft(
            controller,
            scenario_type="full_roleplay",
            sections=_reviewed_full_roleplay_sections(),
            save_title="Reviewed Save",
            request_initial_image=True,
        )
    )

    saves = repositories.list_saves()
    assert len(saves) == 1
    save = saves[0]
    messages = repositories.list_messages(save.id)
    assert [(message.role, message.body) for message in messages] == [
        ("narrator", "Reviewed opening bell."),
    ]
    assert _value(model, "active_save_id") == save.id
    error_text = _error_text(model)
    assert error_text
    assert "sk-live-secret" not in error_text
    assert len(chat_provider.chat_requests) == 1
    assert len(image_provider.image_requests) == 1
    assert repositories.list_media_assets(save.id) == []


def test_save_scenario_draft_with_initial_image_uses_image_generation_preference(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    repositories.set_model_preference(
        task="chat",
        provider="chat-provider",
        model_id="chat-model",
    )
    repositories.set_model_preference(
        task="image_generation",
        provider="image-provider",
        model_id="image-model",
    )
    chat_provider = RuntimeFakeProvider()
    image_provider = RuntimeFakeProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        providers={
            "chat-provider": chat_provider,
            "image-provider": image_provider,
        },
    )

    model = asyncio.run(
        _save_scenario_draft(
            controller,
            scenario_type="full_roleplay",
            sections=_reviewed_full_roleplay_sections(),
            save_title="Reviewed Save",
            request_initial_image=True,
        )
    )

    save = repositories.list_saves()[0]
    messages = repositories.list_messages(save.id)
    assert len(messages) == 1
    opening_message = messages[0]
    assert opening_message.role == "narrator"
    assert opening_message.body == "Reviewed opening bell."
    assert len(chat_provider.chat_requests) == 1
    assert chat_provider.chat_requests[0].provider == "chat-provider"
    assert chat_provider.chat_requests[0].model_id == "chat-model"
    assert len(image_provider.image_requests) == 1
    image_request = image_provider.image_requests[0]
    assert image_request.provider == "image-provider"
    assert image_request.model_id == "image-model"
    assert image_request.source_save_id == save.id
    assert image_request.source_message_id == opening_message.id
    latest_image = _latest_image(model)
    assert _value(latest_image, "source_message_id") == opening_message.id
    assert _value(latest_image, "provider") == "image-provider"
    assert _value(latest_image, "model") == "image-model"


def test_save_scenario_draft_skips_initial_image_when_not_requested(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    repositories.set_model_preference(
        task="chat",
        provider="chat-provider",
        model_id="chat-model",
    )
    repositories.set_model_preference(
        task="image_generation",
        provider="image-provider",
        model_id="image-model",
    )
    chat_provider = RuntimeFakeProvider()
    image_provider = RuntimeFakeProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        providers={
            "chat-provider": chat_provider,
            "image-provider": image_provider,
        },
    )

    model = asyncio.run(
        _save_scenario_draft(
            controller,
            scenario_type="full_roleplay",
            sections=_reviewed_full_roleplay_sections(),
            save_title="Reviewed Save",
            request_initial_image=False,
        )
    )

    save = repositories.list_saves()[0]
    assert [
        (message.role, message.body)
        for message in repositories.list_messages(save.id)
    ] == [
        ("narrator", "Reviewed opening bell."),
    ]
    assert _value(model, "active_save_id") == save.id
    assert chat_provider.chat_requests == []
    assert image_provider.image_requests == []
    assert repositories.list_media_assets(save.id) == []


def test_save_scenario_draft_rejects_blank_required_section_values(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)
    reviewed_sections = {
        **_reviewed_full_roleplay_sections(),
        "tone_genre": "   ",
    }

    model = asyncio.run(
        _save_scenario_draft(
            controller,
            scenario_type="full_roleplay",
            sections=reviewed_sections,
            save_title="Reviewed Save",
            request_initial_image=False,
        )
    )

    assert repositories.list_saves() == []
    assert "empty required sections" in _status_text(model)
    assert "tone_genre" in _status_text(model)


def test_runtime_lists_saved_scenarios_without_gtk_or_provider_calls(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Glass Harbor",
        premise="A drowned harbor rings its bell at low tide.",
        player_role="Harbor warden",
        content={
            "opening_message": "The harbor bell rings under the mud.",
            "action_choices_enabled": True,
            "_source": {
                "origin": "ai_draft",
                "generation_prompt": "A drowned harbor at low tide.",
            },
        },
    )
    provider = RuntimeFakeProvider()
    controller = _runtime_controller(runtime, repositories, tmp_path, provider=provider)

    scenarios = list(controller.list_saved_scenarios())

    assert len(scenarios) == 1
    listed = scenarios[0]
    assert _value(listed, "scenario_id", "id") == scenario.id
    assert _value(listed, "title") == "Glass Harbor"
    assert _value(listed, "scenario_type", "type") == "full_roleplay"
    assert _value(listed, "action_choices_enabled") is True
    assert _value(listed, "has_generation_prompt") is True
    assert _value(listed, "created_at")
    assert _value(listed, "updated_at")
    assert provider.chat_requests == []


def test_runtime_lists_saved_scenarios_with_legacy_type(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    repositories.create_scenario(
        type="legacy_roleplay",
        title="Old Harbor",
        premise="An older scenario template from a previous version.",
        player_role="Keeper",
        content={"opening_message": "The old harbor bell rings."},
    )
    repositories.create_scenario(
        type="full_roleplay",
        title="Glass Harbor",
        premise="A drowned harbor rings its bell at low tide.",
        player_role="Harbor warden",
        content={"opening_message": "The harbor bell rings under the mud."},
    )
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=RuntimeFakeProvider(),
    )

    scenarios = list(controller.list_saved_scenarios())

    assert [scenario.title for scenario in scenarios] == ["Glass Harbor", "Old Harbor"]
    legacy = next(scenario for scenario in scenarios if scenario.title == "Old Harbor")
    assert legacy.scenario_type == "legacy_roleplay"
    assert legacy.scenario_types == ("legacy_roleplay",)


def test_start_saved_scenario_reuses_existing_scenario_and_opening_without_provider(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Glass Harbor",
        premise="A drowned harbor rings its bell at low tide.",
        player_role="Harbor warden",
        content={"opening_message": "The harbor bell rings under the mud."},
    )
    original_save = repositories.create_save(
        scenario_id=scenario.id,
        title="Original Crossing",
    )
    provider = RuntimeFakeProvider()
    controller = _runtime_controller(runtime, repositories, tmp_path, provider=provider)

    model = controller.start_saved_scenario(
        scenario_id=scenario.id,
        save_title="Second Crossing",
    )

    saves = repositories.list_saves()
    new_save = next(save for save in saves if save.id != original_save.id)
    assert new_save.scenario_id == scenario.id
    assert new_save.title == "Second Crossing"
    assert _value(model, "active_save_id") == new_save.id
    assert [
        (message.role, message.speaker_name, message.body)
        for message in repositories.list_messages(new_save.id)
    ] == [
        ("narrator", "Narrator", "The harbor bell rings under the mud."),
    ]
    assert provider.chat_requests == []
    assert provider.image_requests == []


def test_start_saved_scenario_seeds_registry_from_character_starters(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Glass Harbor",
        premise="A drowned harbor rings its bell at low tide.",
        player_role="Harbor warden",
        content={
            "opening_message": "The harbor bell rings under the mud.",
            "characters": "Legacy NPC",
            "character_starters": [
                {
                    "name": "Captain Ilyra",
                    "aliases": ["Ilyra"],
                    "role": "Watch captain",
                    "known_state": "She keeps the bell tower alive.",
                    "appearance": "Bronze cloak clasp and salt-stained boots.",
                    "visual_notes": "Straight silhouette in lighthouse glare.",
                    "personality": "Decisive and guarded.",
                    "voice": "Low clipped orders.",
                    "relationships": {"Mara Voss": "wary ally"},
                    "status": "waiting at the tower",
                    "met": False,
                }
            ],
        },
    )
    provider = RuntimeFakeProvider()
    controller = _runtime_controller(runtime, repositories, tmp_path, provider=provider)

    model = controller.start_saved_scenario(
        scenario_id=scenario.id,
        save_title="Second Crossing",
    )

    active_save_id = _value(model, "active_save_id")
    characters = repositories.list_characters(active_save_id)
    assert [character.name for character in characters] == ["Captain Ilyra"]
    character = characters[0]
    assert character.aliases == ["Ilyra"]
    assert character.role == "Watch captain"
    assert character.known_state == "She keeps the bell tower alive."
    assert character.appearance == "Bronze cloak clasp and salt-stained boots."
    assert character.visual_notes == "Straight silhouette in lighthouse glare."
    assert character.personality == "Decisive and guarded."
    assert character.voice == "Low clipped orders."
    assert character.relationships == {"Mara Voss": "wary ally"}
    assert character.status == "waiting at the tower"
    assert character.met is False
    assert character.protected_from_maintenance is True
    assert {
        "name",
        "aliases",
        "role",
        "known_state",
        "appearance",
        "visual_notes",
        "personality",
        "voice",
    } <= set(character.locked_fields)
    assert "relationships" not in character.locked_fields
    assert "status" not in character.locked_fields
    assert provider.chat_requests == []


def test_start_saved_scenario_seeds_character_starter_reference_image(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    media_dir = tmp_path / "media"
    source_path = media_dir / "scenario-starters" / "scenario-1" / "ilyra.png"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"\x89PNG\r\n\x1a\nstarter image bytes")
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Glass Harbor",
        premise="A drowned harbor rings its bell at low tide.",
        player_role="Harbor warden",
        content={
            "opening_message": "The harbor bell rings under the mud.",
            "character_starters": [
                {
                    "starter_id": "starter-ilyra",
                    "name": "Captain Ilyra",
                    "appearance": "Bronze cloak clasp and salt-stained boots.",
                    "reference_image": {
                        "id": "starter-ref-ilyra",
                        "path": "scenario-starters/scenario-1/ilyra.png",
                        "thumbnail_path": None,
                        "mime_type": "image/png",
                        "prompt_preview": "Uploaded character reference image",
                        "source": "uploaded",
                    },
                }
            ],
        },
    )
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = controller.start_saved_scenario(
        scenario_id=scenario.id,
        save_title="Second Crossing",
    )

    active_save_id = _value(model, "active_save_id")
    [character] = repositories.list_characters(active_save_id)
    [asset] = repositories.list_media_assets(active_save_id)
    assert asset.type == "image"
    assert asset.status == "succeeded"
    assert asset.mime_type == "image/png"
    assert json.loads(asset.metadata_json) == {
        "kind": "character_reference",
        "character_id": character.id,
        "source": "scenario_starter",
        "starter_id": "starter-ilyra",
        "starter_reference_image_id": "starter-ref-ilyra",
    }
    assert (media_dir / asset.path).read_bytes() == source_path.read_bytes()
    assert [
        (
            link.entity_type,
            link.entity_id,
            link.target_type,
            link.target_id,
            link.relation,
        )
        for link in repositories.list_entity_links(active_save_id)
    ] == [
        ("character", character.id, "media_asset", asset.id, "reference_image")
    ]


@pytest.mark.parametrize(
    (
        "scenario_type",
        "section_factory",
        "expected_state_key",
        "expected_section_id",
    ),
    [
        (
            "settlement_builder",
            "_reviewed_settlement_builder_sections",
            "settlement.projects",
            "projects_and_facilities",
        ),
        (
            "monster_hunt_bounty",
            "_reviewed_monster_hunt_bounty_sections",
            "hunt.leads",
            "leads_and_clues",
        ),
        (
            "road_trip_pilgrimage",
            "_reviewed_road_trip_pilgrimage_sections",
            "journey.progress",
            "journey_progress",
        ),
        (
            "merchant_trade_route",
            "_reviewed_merchant_trade_route_sections",
            "trade.contracts",
            "contracts_and_debts",
        ),
    ],
)
def test_start_saved_management_scenario_seeds_template_state(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    scenario_type: str,
    section_factory: str,
    expected_state_key: str,
    expected_section_id: str,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    section_factory_func = cast(
        Callable[[], dict[str, str]],
        globals()[section_factory],
    )
    sections = section_factory_func()
    scenario = repositories.create_scenario(
        type=scenario_type,
        title=sections["title"],
        premise=sections["premise"],
        player_role=sections["player_role"],
        content=cast(dict[str, object], dict(sections)),
    )
    provider = RuntimeFakeProvider()
    controller = _runtime_controller(runtime, repositories, tmp_path, provider=provider)

    model = controller.start_saved_scenario(
        scenario_id=scenario.id,
        save_title=f"{sections['title']} Replay",
    )

    active_save_id = _value(model, "active_save_id")
    state = {row.key: row for row in repositories.list_world_state(active_save_id)}
    assert state[expected_state_key].value == {
        "summary": sections[expected_section_id],
    }
    assert provider.chat_requests == []


def test_start_saved_scenario_can_skip_process_global_selection(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Glass Harbor",
        premise="A drowned harbor rings its bell at low tide.",
        player_role="Harbor warden",
        content={"opening_message": "The harbor bell rings under the mud."},
    )
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = controller.start_saved_scenario(
        scenario_id=scenario.id,
        save_title="Second Crossing",
        remember_process_active_save=False,
    )

    save = repositories.list_saves()[0]
    assert _value(model, "active_save_id") == save.id
    assert controller.active_save_id is None


def test_delete_save_clears_active_session_and_removes_save(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories)
    controller = _runtime_controller(runtime, repositories, tmp_path)
    controller.load_save(save_id)

    model = controller.delete_save(save_id)

    assert repositories.get_save(save_id) is None
    assert controller.active_save_id is None
    assert _value(model, "active_save_id") is None
    assert _status_text(model) == "Deleted save: Night Watch"


def test_delete_saved_scenario_removes_unlinked_scenario(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Glass Harbor",
        premise="A drowned harbor rings its bell at low tide.",
        player_role="Harbor warden",
        content={"opening_message": "The harbor bell rings under the mud."},
    )
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = controller.delete_saved_scenario(scenario.id)

    assert repositories.get_scenario(scenario.id) is None
    assert _status_text(model) == "Deleted scenario: Glass Harbor"


def test_delete_saved_scenario_refuses_linked_saves_without_deleting_save(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Glass Harbor",
        premise="A drowned harbor rings its bell at low tide.",
        player_role="Harbor warden",
        content={"opening_message": "The harbor bell rings under the mud."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Crossing")
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = controller.delete_saved_scenario(scenario.id)

    assert "existing saves" in _error_text(model)
    assert repositories.get_scenario(scenario.id) == scenario
    assert repositories.get_save(save.id) == save


def test_load_existing_save_returns_chronicle_and_media_for_that_save(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, narrator_id = _persist_runtime_save(repositories)
    other_save_id, other_narrator_id = _persist_runtime_save(
        repositories,
        title="Other Watch",
        player_body="This message belongs elsewhere.",
        narrator_body="This image belongs elsewhere.",
    )
    repositories.create_media_asset(
        save_id=save_id,
        source_message_id=narrator_id,
        type="image",
        path="media/night-watch/scene.png",
        thumbnail_path=None,
        prompt="The beacon lens catches fire.",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    repositories.create_media_asset(
        save_id=other_save_id,
        source_message_id=other_narrator_id,
        type="image",
        path="media/other-watch/scene.png",
        thumbnail_path=None,
        prompt="Other save image.",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = controller.load_save(save_id)

    assert _value(model, "active_save_id") == save_id
    message_bodies = [_value(message, "body") for message in _chronicle_messages(model)]
    assert message_bodies == [
        "I climb toward the beacon lens.",
        "Ash scratches the glass as the stair shakes.",
    ]
    assert [
        _value(message, "role_label") for message in _chronicle_messages(model)
    ] == [
        "Mara",
        "Narrator",
    ]
    assert _value(_latest_image(model), "path") == ("media/night-watch/scene.png")


def test_runtime_build_chat_history_model_uses_active_save_and_filter(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, narrator_id = _persist_runtime_save(repositories)
    other_save_id, other_narrator_id = _persist_runtime_save(
        repositories,
        title="Other Watch",
        player_body="This message belongs elsewhere.",
        narrator_body="This image belongs elsewhere.",
    )
    repositories.create_media_asset(
        save_id=save_id,
        source_message_id=narrator_id,
        type="image",
        path="media/night-watch/scene.png",
        thumbnail_path=None,
        prompt="The beacon lens catches fire.",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    repositories.create_media_asset(
        save_id=other_save_id,
        source_message_id=other_narrator_id,
        type="image",
        path="media/other-watch/scene.png",
        thumbnail_path=None,
        prompt="Other save image.",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    controller = _runtime_controller(runtime, repositories, tmp_path)
    controller.load_save(save_id)

    model = controller.build_chat_history_model(selected_filter="with_images")

    assert _value(model, "active_save_id") == save_id
    assert _value(model, "active_save_title") == "Night Watch"
    assert [_value(message, "body") for message in _history_messages(model)] == [
        "Ash scratches the glass as the stair shakes.",
    ]
    assert _value(_history_messages(model)[0], "image_count") == 1


def test_runtime_build_chat_history_model_handles_no_save(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = controller.build_chat_history_model()

    assert _value(model, "active_save_id") is None
    assert _history_messages(model) == ()
    assert _value(model, "empty_title") == "No save loaded"


def test_submit_player_message_uses_fake_chat_and_updates_model(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories, include_messages=False)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RuntimeFakeProvider()
    context_search = NoopContextSearch()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=context_search,
    )
    controller.load_save(save_id)

    model = asyncio.run(
        controller.submit_player_message(
            body="I touch the mirror floor.",
            speaker_name="Mara",
        )
    )

    assert len(provider.chat_requests) == 1
    assert provider.chat_requests[0].model_id == "fake-chat"
    persisted_messages = repositories.list_messages(save_id)
    assert context_search.calls == [(save_id, persisted_messages[0].id)]
    persisted_message_rows = [
        (message.role, message.body, message.model) for message in persisted_messages
    ]
    assert persisted_message_rows == [
        ("player", "I touch the mirror floor.", None),
        ("narrator", "fake narrator: I touch the mirror floor.", "fake-chat"),
    ]
    model_message_bodies = [
        _value(message, "body") for message in _chronicle_messages(model)
    ]
    assert model_message_bodies == [
        "I touch the mirror floor.",
        "fake narrator: I touch the mirror floor.",
    ]
    assert [
        _value(message, "role_label") for message in _chronicle_messages(model)
    ] == [
        "Mara",
        "Narrator",
    ]
    assert "fake-chat" in _value(model, "model_indicator")
    assert _error_text(model) == ""
    assert _status_text(model) == "Turn complete"


def test_submit_player_message_accepts_scenario_specific_chat_without_generic_chat(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories, include_messages=False)
    repositories.set_model_preference(
        task="chat_full_roleplay",
        provider="fake",
        model_id="fake-full-roleplay-chat",
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RuntimeFakeProvider()
    context_search = NoopContextSearch()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=context_search,
    )
    controller.load_save(save_id)

    model = asyncio.run(
        controller.submit_player_message(
            body="I touch the mirror floor.",
            speaker_name="Mara",
        )
    )

    assert repositories.get_model_preference("chat") is None
    assert len(provider.chat_requests) == 1
    assert provider.chat_requests[0].model_id == "fake-full-roleplay-chat"
    persisted_messages = repositories.list_messages(save_id)
    assert context_search.calls == [(save_id, persisted_messages[0].id)]
    persisted_message_rows = [
        (message.role, message.body, message.model) for message in persisted_messages
    ]
    assert persisted_message_rows == [
        ("player", "I touch the mirror floor.", None),
        (
            "narrator",
            "fake narrator: I touch the mirror floor.",
            "fake-full-roleplay-chat",
        ),
    ]
    assert _error_text(model) == ""
    assert _status_text(model) == "Turn complete"


def test_submit_player_message_reports_fallback_used_from_chat_completion_job(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories, include_messages=False)
    _configure_chat_and_context_preferences(repositories)

    class FakeChatService:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def submit_player_turn(
            self,
            *,
            save_id: str,
            body: str,
            speaker_name: str | None = None,
            run_post_turn_jobs: bool = True,
        ) -> object:
            assert run_post_turn_jobs is True
            player = repositories.append_message(
                save_id=save_id,
                role="player",
                speaker_name=speaker_name,
                body=body,
            )
            narrator = repositories.append_message(
                save_id=save_id,
                role="narrator",
                speaker_name="Narrator",
                body="fallback narrator: the glass reflects a second sky.",
                provider="fake",
                model="fake-chat",
            )
            job = repositories.create_job(
                save_id=save_id,
                type="chat_completion",
                status="running",
                payload={
                    "player_message_id": player.id,
                    "provider": "fake",
                    "model": "fake-chat",
                },
            )
            repositories.update_job(
                job.id,
                status="succeeded",
                result={
                    "narrator_message_id": narrator.id,
                    "fallback_used": True,
                },
            )
            return SimpleNamespace(
                player_message=player,
                narrator_message=narrator,
                fallback_used=False,
            )

    monkeypatch.setattr(runtime, "ChatService", FakeChatService)
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    model = asyncio.run(
        controller.submit_player_message(
            body="I touch the mirror floor.",
            speaker_name="Mara",
        )
    )

    assert _status_text(model) == "Turn complete; fallback model used"
    assert _error_text(model) == ""


def test_initial_render_submit_and_post_turn_warn_when_context_was_trimmed(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories, include_messages=False)
    _configure_chat_and_context_preferences(repositories)
    post_turn_calls: list[tuple[str, str, str]] = []

    class FakeChatService:
        def __init__(self, **_kwargs: object) -> None:
            return None

        async def submit_player_turn(
            self,
            *,
            save_id: str,
            body: str,
            speaker_name: str | None = None,
            run_post_turn_jobs: bool = True,
        ) -> object:
            assert run_post_turn_jobs is False
            player = repositories.append_message(
                save_id=save_id,
                role="player",
                speaker_name=speaker_name,
                body=body,
            )
            narrator = repositories.append_message(
                save_id=save_id,
                role="narrator",
                speaker_name="Narrator",
                body="fake narrator: the beacon answer is shorter than expected.",
                provider="fake",
                model="fake-chat",
            )
            return SimpleNamespace(
                player_message=player,
                narrator_message=narrator,
                context_trimmed=True,
            )

        async def run_post_turn_jobs(
            self,
            *,
            save_id: str,
            player_message_id: str,
            narrator_message_id: str,
        ) -> None:
            post_turn_calls.append((save_id, player_message_id, narrator_message_id))

    monkeypatch.setattr(runtime, "ChatService", FakeChatService)
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    turn = asyncio.run(
        controller.submit_player_message_for_initial_render(
            body="I touch the mirror floor.",
            speaker_name="Mara",
        )
    )

    assert _value(turn, "context_trimmed") is True
    assert _status_text(_value(turn, "delta")) == (
        "Turn complete; context budget trimmed older context"
    )

    completed_model = asyncio.run(
        controller.run_post_turn_jobs(
            save_id=save_id,
            player_message_id=_value(turn, "player_message_id"),
            narrator_message_id=_value(turn, "narrator_message_id"),
        )
    )

    assert post_turn_calls == [
        (
            save_id,
            _value(turn, "player_message_id"),
            _value(turn, "narrator_message_id"),
        )
    ]
    assert _status_text(completed_model) == (
        "Turn complete; context budget trimmed older context"
    )
    assert _error_text(completed_model) == ""


def test_initial_render_submit_passes_retry_progress_callback_to_chat_provider(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories, include_messages=False)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RuntimeFakeProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)
    progress_events: list[ProviderRetryProgress] = []

    def callback(progress: ProviderRetryProgress) -> None:
        progress_events.append(progress)

    asyncio.run(
        controller.submit_player_message_for_initial_render(
            body="I touch the mirror floor.",
            speaker_name="Mara",
            retry_progress_callback=callback,
        )
    )

    assert len(provider.chat_requests) == 1
    assert provider.chat_requests[0].retry_progress_callback is callback
    assert progress_events == []


def test_timeskip_initial_render_persists_system_request_and_defers_post_turn_jobs(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories, include_messages=False)
    _configure_chat_and_context_preferences(repositories)
    provider = RuntimeFakeProvider()
    context_search = NoopContextSearch()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=context_search,
    )
    controller.load_save(save_id)

    turn = asyncio.run(
        controller.submit_timeskip_for_initial_render(
            instruction="Skip to dawn at the city gates.",
        )
    )

    persisted_messages = repositories.list_messages(save_id)
    assert [message.role for message in persisted_messages] == ["system", "narrator"]
    assert persisted_messages[0].speaker_name == "Timeskip"
    assert persisted_messages[0].body == (
        "Timeskip request: Skip to dawn at the city gates."
    )
    assert context_search.calls == [(save_id, persisted_messages[0].id)]
    assert len(provider.chat_requests) == 1
    assert provider.chat_requests[0].turn_directive == persisted_messages[0].body
    assert _value(turn, "has_post_turn_jobs") is True
    assert _value(turn, "player_message_id") == persisted_messages[0].id
    assert _value(turn, "narrator_message_id") == persisted_messages[1].id
    assert _status_text(_value(turn, "delta")) == "Turn complete"


@pytest.mark.parametrize(
    ("player_character_name", "expected_speaker_name"),
    [
        ("Mara Voss", "Mara Voss"),
        ("Ren", "Ren"),
        ("  ", "Player"),
    ],
)
def test_submit_player_message_without_speaker_name_uses_default_player_display_name(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    player_character_name: str,
    expected_speaker_name: str,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(
        repositories,
        include_messages=False,
        player_character_name=player_character_name,
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RuntimeFakeProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    model = asyncio.run(
        controller.submit_player_message(body="I touch the mirror floor.")
    )

    persisted_messages = repositories.list_messages(save_id)
    assert persisted_messages[0].role == "player"
    assert persisted_messages[0].speaker_name == expected_speaker_name
    first_model_message = next(iter(_chronicle_messages(model)))
    assert _value(first_model_message, "role_label") == expected_speaker_name
    assert provider.chat_requests[0].messages[-1].speaker_name == expected_speaker_name


def test_submit_player_message_without_speaker_name_prefers_registry_player_character(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(
        repositories,
        include_messages=False,
        player_character_name="Scenario Mara",
    )
    repositories.add_character(
        save_id=save_id,
        name="Registry Iris",
        is_player_character=True,
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RuntimeFakeProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    model = asyncio.run(
        controller.submit_player_message(body="I touch the mirror floor.")
    )

    persisted_messages = repositories.list_messages(save_id)
    first_model_message = next(iter(_chronicle_messages(model)))
    assert persisted_messages[0].speaker_name == "Registry Iris"
    assert _value(first_model_message, "role_label") == "Registry Iris"
    assert provider.chat_requests[0].messages[-1].speaker_name == "Registry Iris"


def test_submit_player_message_does_not_capture_prompt_inspection_when_debug_disabled(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories, include_messages=False)
    _configure_chat_and_context_preferences(repositories)
    repositories.set_app_setting("debug_logging_enabled", False)
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    model = asyncio.run(
        controller.submit_player_message(
            body="I touch the mirror floor.",
            speaker_name="Mara",
        )
    )

    assert controller.prompt_inspection_store.prompts_by_message_id() == {}
    narrator_message = _chronicle_messages(model)[-1]
    assert _value(narrator_message, "role") == "narrator"
    assert _value(narrator_message, "debug_prompt") is None
    assert "inspect-debug-prompt" not in _actions_by_id(narrator_message)


def test_submit_player_message_captures_prompt_inspection_when_debug_enabled(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories, include_messages=False)
    _configure_chat_and_context_preferences(repositories)
    repositories.set_app_setting("debug_logging_enabled", True)
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    model = asyncio.run(
        controller.submit_player_message(
            body="I touch the mirror floor.",
            speaker_name="Mara",
        )
    )

    narrator_message = _chronicle_messages(model)[-1]
    prompt_text = _value(narrator_message, "debug_prompt")
    assert _value(narrator_message, "role") == "narrator"
    assert "Source cards" in prompt_text
    assert "Narrator prompt" in prompt_text
    assert "Raw requests" in prompt_text
    assert "I touch the mirror floor." in prompt_text
    assert '"model_id": "fake-chat"' in prompt_text
    assert (
        controller.prompt_inspection_store.prompt_for_message(
            _value(narrator_message, "message_id")
        )
        == prompt_text
    )
    action = _actions_by_id(narrator_message)["inspect-debug-prompt"]
    assert _value(action, "label") == "Inspect prompt"
    assert _value(action, "detail_text") == prompt_text


def test_runtime_default_summary_service_reads_updated_settings(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)

    class FakeSummaryService:
        enabled: bool
        threshold: float

        def __init__(self, **kwargs: object) -> None:
            enabled = kwargs["enabled"]
            threshold = kwargs["threshold"]
            assert isinstance(enabled, bool)
            assert isinstance(threshold, int | float)
            self.enabled = enabled
            self.threshold = float(threshold)
            constructed.append(self)

    constructed: list[FakeSummaryService] = []
    monkeypatch.setattr(runtime, "SummaryService", FakeSummaryService)
    controller = _runtime_controller(runtime, repositories, tmp_path)

    repositories.set_app_setting("automatic_summarization_enabled", False)
    repositories.set_app_setting("summarization_context_pressure_threshold", 0.4)
    first_service = controller._summary_service()

    assert first_service is constructed[0]
    assert first_service.enabled is False
    assert first_service.threshold == 0.4

    repositories.set_app_setting("automatic_summarization_enabled", True)
    repositories.set_app_setting("summarization_context_pressure_threshold", 0.9)
    second_service = controller._summary_service()

    assert second_service is constructed[1]
    assert second_service is not first_service
    assert second_service.enabled is True
    assert second_service.threshold == 0.9


def test_runtime_default_media_service_reads_updated_settings(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)

    class FakeMediaService:
        automatic_enabled: bool
        auto_frequency: int

        def __init__(self, **kwargs: object) -> None:
            automatic_enabled = kwargs["automatic_enabled"]
            auto_frequency = kwargs["auto_frequency"]
            assert isinstance(automatic_enabled, bool)
            assert isinstance(auto_frequency, int)
            self.automatic_enabled = automatic_enabled
            self.auto_frequency = auto_frequency
            constructed.append(self)

    constructed: list[FakeMediaService] = []
    monkeypatch.setattr(runtime, "MediaService", FakeMediaService)
    controller = _runtime_controller(runtime, repositories, tmp_path)

    repositories.set_app_setting("automatic_image_generation_enabled", False)
    repositories.set_app_setting("image_generation_frequency", 0)
    first_service = controller._media_service()

    assert first_service is constructed[0]
    assert first_service.automatic_enabled is False
    assert first_service.auto_frequency == 0

    repositories.set_app_setting("automatic_image_generation_enabled", True)
    repositories.set_app_setting("image_generation_frequency", 5)
    second_service = controller._media_service()

    assert second_service is constructed[1]
    assert second_service is not first_service
    assert second_service.automatic_enabled is True
    assert second_service.auto_frequency == 5


def test_regenerate_message_rolls_back_prior_player_turn_and_resubmits(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, ids = _persist_revision_save(repositories)
    _configure_chat_and_context_preferences(repositories)
    provider = RuntimeFakeProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    model = asyncio.run(controller.regenerate_message(message_id=ids["narrator_2"]))

    assert len(provider.chat_requests) == 1
    assert provider.chat_requests[0].messages[-1].body == "I open the sealed door."
    active_messages = repositories.list_messages(save_id)
    assert [message.body for message in active_messages] == [
        "I light the beacon.",
        "The beacon wakes.",
        "I open the sealed door.",
        "fake narrator: I open the sealed door.",
    ]
    assert repositories.list_world_state(save_id) == []
    assert active_messages[-2].id not in ids.values()
    assert active_messages[-1].id not in ids.values()
    _assert_messages_deleted_or_archived(
        repositories,
        [ids["player_2"], ids["narrator_2"], ids["player_3"], ids["narrator_3"]],
    )
    assert [_value(message, "body") for message in _chronicle_messages(model)] == [
        "I light the beacon.",
        "The beacon wakes.",
        "I open the sealed door.",
        "fake narrator: I open the sealed door.",
    ]


def test_edit_and_resubmit_message_rolls_back_state_context_and_submits_edit(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, ids = _persist_revision_save(repositories)
    _persist_deleted_turn_context(repositories, save_id=save_id, ids=ids)
    _configure_chat_and_context_preferences(repositories)
    provider = RuntimeFakeProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    model = asyncio.run(
        controller.edit_and_resubmit_message(
            message_id=ids["player_2"],
            body="I knock once and listen at the sealed door.",
        )
    )

    assert len(provider.chat_requests) == 1
    assert provider.chat_requests[0].messages[-1].body == (
        "I knock once and listen at the sealed door."
    )
    active_messages = repositories.list_messages(save_id)
    assert [message.body for message in active_messages] == [
        "I light the beacon.",
        "The beacon wakes.",
        "I knock once and listen at the sealed door.",
        "fake narrator: I knock once and listen at the sealed door.",
    ]
    _assert_messages_deleted_or_archived(
        repositories,
        [ids["player_2"], ids["narrator_2"], ids["player_3"], ids["narrator_3"]],
    )
    assert [memory.body for memory in repositories.list_memories(save_id)] == []
    assert [summary.body for summary in repositories.list_summaries(save_id)] == []
    assert {
        state.key: state.value for state in repositories.list_world_state(save_id)
    } == {
        "scene.location": {"name": "Beacon tower"},
    }
    assert [_value(message, "body") for message in _chronicle_messages(model)] == [
        "I light the beacon.",
        "The beacon wakes.",
        "I knock once and listen at the sealed door.",
        "fake narrator: I knock once and listen at the sealed door.",
    ]


def test_edit_message_without_resubmit_player_reconciles_world_data(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, ids = _persist_revision_save(repositories)
    _configure_narrator_edit_reconciliation_model(repositories)
    provider = RuntimeStructuredReconciliationProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    model = asyncio.run(
        controller.edit_message_without_resubmit(
            message_id=ids["player_2"],
            body="I keep the sealed door shut and listen.",
        )
    )

    assert provider.chat_requests == []
    assert [request.schema_name for request in provider.structured_requests] == [
        "state_memory_extraction",
        "context_update_extraction",
    ]
    request_text = "\n\n".join(
        message.body
        for request in provider.structured_requests
        for message in request.messages
    )
    assert "Player message correction" in request_text
    assert "Edited player text" in request_text
    assert "I keep the sealed door shut and listen." in request_text
    assert "-I open the sealed door." in request_text
    assert "+I keep the sealed door shut and listen." in request_text
    active_messages = repositories.list_messages(save_id)
    assert [message.id for message in active_messages] == [
        ids["player_1"],
        ids["narrator_1"],
        ids["player_2"],
        ids["narrator_2"],
        ids["player_3"],
        ids["narrator_3"],
    ]
    assert active_messages[2].body == "I keep the sealed door shut and listen."
    assert active_messages[3].body == "The corridor floods with ash."
    revisions = repositories.list_message_revisions(
        save_id=save_id,
        message_id=ids["player_2"],
    )
    assert len(revisions) == 1
    assert revisions[0].reconciliation_status == "succeeded"
    assert revisions[0].reconciled_at is not None
    assert {
        state.key: state.value for state in repositories.list_world_state(save_id)
    } == {"scene.corridor": {"status": "stable"}}
    edited_message = next(
        message
        for message in _chronicle_messages(model)
        if _value(message, "message_id") == ids["player_2"]
    )
    assert _value(edited_message, "body") == (
        "I keep the sealed door shut and listen."
    )
    assert _value(edited_message, "revision_count") == 1
    assert _value(edited_message, "edited_at") is not None


def test_edit_narrator_message_persists_revision_and_reconciles_world_data(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, ids = _persist_revision_save(repositories)
    _configure_narrator_edit_reconciliation_model(repositories)
    provider = RuntimeStructuredReconciliationProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    model = asyncio.run(
        controller.edit_narrator_message(
            message_id=ids["narrator_2"],
            body="The corridor holds steady as ash taps the door.",
        )
    )

    assert provider.chat_requests == []
    assert [request.schema_name for request in provider.structured_requests] == [
        "state_memory_extraction",
        "context_update_extraction",
    ]
    request_text = "\n\n".join(
        message.body
        for request in provider.structured_requests
        for message in request.messages
    )
    assert "Narrator message correction" in request_text
    assert "The corridor floods with ash." in request_text
    assert "The corridor holds steady as ash taps the door." in request_text
    assert "-The corridor floods with ash." in request_text
    assert "+The corridor holds steady as ash taps the door." in request_text
    active_messages = repositories.list_messages(save_id)
    assert [message.id for message in active_messages] == [
        ids["player_1"],
        ids["narrator_1"],
        ids["player_2"],
        ids["narrator_2"],
        ids["player_3"],
        ids["narrator_3"],
    ]
    assert active_messages[3].body == (
        "The corridor holds steady as ash taps the door."
    )
    revisions = repositories.list_message_revisions(
        save_id=save_id,
        message_id=ids["narrator_2"],
    )
    assert len(revisions) == 1
    assert revisions[0].reconciliation_status == "succeeded"
    assert revisions[0].reconciled_at is not None
    assert {
        state.key: state.value for state in repositories.list_world_state(save_id)
    } == {"scene.corridor": {"status": "stable"}}
    edited_message = next(
        message
        for message in _chronicle_messages(model)
        if _value(message, "message_id") == ids["narrator_2"]
    )
    assert _value(edited_message, "body") == (
        "The corridor holds steady as ash taps the door."
    )
    assert _value(edited_message, "revision_count") == 1
    assert _value(edited_message, "edited_at") is not None


def test_edit_narrator_message_reconciliation_prefers_tool_calls(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, ids = _persist_revision_save(repositories)
    _configure_narrator_edit_reconciliation_model(repositories)
    provider = RuntimeToolReconciliationProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    asyncio.run(
        controller.edit_narrator_message(
            message_id=ids["narrator_2"],
            body="The corridor holds steady as ash taps the door.",
        )
    )

    assert provider.chat_requests == []
    requested_tool_names = [
        tuple(tool.name for tool in request.tools)
        for request in provider.tool_requests
    ]
    assert requested_tool_names == [
        ("patch_world_state", "record_memory_fact", "flag_state_conflict"),
        (
            "update_scene_snapshot",
            "upsert_location",
            "upsert_character",
            "upsert_active_thread",
            "link_entities",
            "record_phone_number_exchange",
        ),
    ]
    request_text = "\n\n".join(
        message.body
        for request in provider.tool_requests
        for message in request.messages
    )
    assert "Narrator message correction" in request_text
    assert "-The corridor floods with ash." in request_text
    assert "+The corridor holds steady as ash taps the door." in request_text
    revisions = repositories.list_message_revisions(
        save_id=save_id,
        message_id=ids["narrator_2"],
    )
    assert revisions[0].reconciliation_status == "succeeded"
    assert {
        state.key: state.value for state in repositories.list_world_state(save_id)
    } == {"scene.corridor": {"status": "stable"}}


def test_edit_narrator_message_serializes_reconciliation_per_save(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, ids = _persist_revision_save(repositories)
    _configure_narrator_edit_reconciliation_model(repositories)
    provider = BlockingRuntimeStructuredReconciliationProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    async def run_edits() -> None:
        first = asyncio.create_task(
            controller.edit_narrator_message(
                message_id=ids["narrator_2"],
                body="The corridor holds steady.",
            )
        )
        await asyncio.wait_for(provider.first_request_started.wait(), timeout=1.0)
        second = asyncio.create_task(
            controller.edit_narrator_message(
                message_id=ids["narrator_2"],
                body="The corridor stays clear.",
            )
        )
        await asyncio.sleep(0.05)
        assert not second.done()

        provider.release_first_request.set()
        await asyncio.gather(first, second)

    asyncio.run(run_edits())

    revisions = repositories.list_message_revisions(
        save_id=save_id,
        message_id=ids["narrator_2"],
    )
    assert [revision.revision_number for revision in revisions] == [1, 2]
    assert [revision.reconciliation_status for revision in revisions] == [
        "succeeded",
        "succeeded",
    ]
    assert repositories.list_messages(save_id)[3].body == (
        "The corridor stays clear."
    )


def test_edit_narrator_message_keeps_revision_when_reconciliation_is_unavailable(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, ids = _persist_revision_save(repositories)
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=RuntimeFakeProvider(),
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    model = asyncio.run(
        controller.edit_narrator_message(
            message_id=ids["narrator_2"],
            body="The corridor holds steady as ash taps the door.",
        )
    )

    active_messages = repositories.list_messages(save_id)
    assert active_messages[3].body == (
        "The corridor holds steady as ash taps the door."
    )
    revisions = repositories.list_message_revisions(
        save_id=save_id,
        message_id=ids["narrator_2"],
    )
    assert revisions[0].reconciliation_status == "skipped"
    assert _status_text(model) == "Narrator message edited"


def test_delete_messages_from_here_soft_deletes_suffix_without_provider_preferences(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, ids = _persist_revision_save(repositories)
    _persist_deleted_turn_context(repositories, save_id=save_id, ids=ids)
    provider = RuntimeFakeProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    model = controller.delete_messages_from_here(message_id=ids["narrator_2"])

    assert provider.chat_requests == []
    assert _status_text(model) == "Messages deleted"
    assert [_value(message, "body") for message in _chronicle_messages(model)] == [
        "I light the beacon.",
        "The beacon wakes.",
    ]
    active_messages = repositories.list_messages(save_id)
    assert [message.id for message in active_messages] == [
        ids["player_1"],
        ids["narrator_1"],
    ]
    audit_messages = repositories.list_messages(save_id, include_deleted=True)
    assert [message.id for message in audit_messages] == [
        ids["player_1"],
        ids["narrator_1"],
        ids["player_2"],
        ids["narrator_2"],
        ids["player_3"],
        ids["narrator_3"],
    ]
    assert all(message.deleted_at is None for message in audit_messages[:2])
    assert all(message.deleted_at is not None for message in audit_messages[2:])
    assert {
        state.key: state.value for state in repositories.list_world_state(save_id)
    } == {
        "scene.location": {"name": "Beacon tower"},
    }
    assert repositories.list_memories(save_id) == []
    assert repositories.list_summaries(save_id) == []


@pytest.mark.parametrize(
    ("action", "expected_method", "expected_status", "replacement_body"),
    [
        (
            "regenerate",
            "submit_player_turn",
            "Message regenerated",
            "I open the sealed door.",
        ),
        (
            "edit_resubmit",
            "submit_existing_player_turn",
            "Edited message resubmitted",
            "I knock once and listen at the sealed door.",
        ),
    ],
)
def test_message_revision_initial_render_defers_post_turn_jobs_and_returns_ids(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    action: str,
    expected_method: str,
    expected_status: str,
    replacement_body: str,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, ids = _persist_revision_save(repositories)
    _configure_chat_and_context_preferences(repositories)
    calls: list[tuple[str, bool]] = []
    post_turn_calls: list[tuple[str, str, str]] = []

    class FakeChatService:
        def __init__(self, **_kwargs: object) -> None:
            return None

        async def submit_player_turn(
            self,
            *,
            save_id: str,
            body: str,
            speaker_name: str | None = None,
            run_post_turn_jobs: bool = True,
        ) -> object:
            calls.append(("submit_player_turn", run_post_turn_jobs))
            player = repositories.append_message(
                save_id=save_id,
                role="player",
                speaker_name=speaker_name,
                body=body,
            )
            narrator = repositories.append_message(
                save_id=save_id,
                role="narrator",
                speaker_name="Narrator",
                body=f"fake narrator: {body}",
                provider="fake",
                model="fake-chat",
            )
            if run_post_turn_jobs:
                post_turn_calls.append((save_id, player.id, narrator.id))
            return SimpleNamespace(player_message=player, narrator_message=narrator)

        async def submit_existing_player_turn(
            self,
            *,
            save_id: str,
            player_message_id: str,
            run_post_turn_jobs: bool = True,
        ) -> object:
            calls.append(("submit_existing_player_turn", run_post_turn_jobs))
            player = next(
                message
                for message in repositories.list_messages(save_id)
                if message.id == player_message_id
            )
            narrator = repositories.append_message(
                save_id=save_id,
                role="narrator",
                speaker_name="Narrator",
                body=f"fake narrator: {player.body}",
                provider="fake",
                model="fake-chat",
            )
            if run_post_turn_jobs:
                post_turn_calls.append((save_id, player.id, narrator.id))
            return SimpleNamespace(player_message=player, narrator_message=narrator)

        async def run_post_turn_jobs(self, **_kwargs: object) -> None:
            pytest.fail("revision initial render should not run post-turn jobs")

    monkeypatch.setattr(runtime, "ChatService", FakeChatService)
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    if action == "regenerate":
        turn = asyncio.run(
            controller.regenerate_message_for_initial_render(
                message_id=ids["narrator_2"]
            )
        )
    else:
        turn = asyncio.run(
            controller.edit_and_resubmit_message_for_initial_render(
                message_id=ids["player_2"],
                body=replacement_body,
            )
        )

    active_messages = repositories.list_messages(save_id)
    replacement_player = active_messages[-2]
    replacement_narrator = active_messages[-1]
    assert calls == [(expected_method, False)]
    assert post_turn_calls == []
    assert _value(turn, "save_id") == save_id
    assert _value(turn, "player_message_id") == replacement_player.id
    assert _value(turn, "narrator_message_id") == replacement_narrator.id
    assert _value(turn, "has_post_turn_jobs") is True
    assert _value(_value(turn, "model"), "status") == expected_status
    assert [(message.role, message.body) for message in active_messages] == [
        ("player", "I light the beacon."),
        ("narrator", "The beacon wakes."),
        ("player", replacement_body),
        ("narrator", f"fake narrator: {replacement_body}"),
    ]


def test_regenerate_message_initial_render_passes_one_shot_feedback(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, ids = _persist_revision_save(repositories)
    _configure_chat_and_context_preferences(repositories)
    calls: list[tuple[str, str, str, bool]] = []

    class FakeChatService:
        def __init__(self, **_kwargs: object) -> None:
            return None

        async def submit_player_turn(
            self,
            *,
            save_id: str,
            body: str,
            speaker_name: str | None = None,
            run_post_turn_jobs: bool = True,
            regeneration_feedback: str = "",
        ) -> object:
            calls.append(
                (
                    save_id,
                    body,
                    regeneration_feedback,
                    run_post_turn_jobs,
                )
            )
            player = repositories.append_message(
                save_id=save_id,
                role="player",
                speaker_name=speaker_name,
                body=body,
            )
            narrator = repositories.append_message(
                save_id=save_id,
                role="narrator",
                speaker_name="Narrator",
                body=f"fake narrator: {body}",
                provider="fake",
                model="fake-chat",
            )
            return SimpleNamespace(player_message=player, narrator_message=narrator)

        async def run_post_turn_jobs(self, **_kwargs: object) -> None:
            pytest.fail("revision initial render should not run post-turn jobs")

    monkeypatch.setattr(runtime, "ChatService", FakeChatService)
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    result = asyncio.run(
        controller.regenerate_message_for_initial_render(
            message_id=ids["narrator_2"],
            regeneration_feedback="  Make the answer sharper.  ",
        )
    )

    assert calls == [
        (
            save_id,
            "I open the sealed door.",
            "Make the answer sharper.",
            False,
        )
    ]
    assert result.has_post_turn_jobs is True


def test_edit_and_resubmit_reports_committed_edit_before_narrator_finishes(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, ids = _persist_revision_save(repositories)
    _configure_chat_and_context_preferences(repositories)
    replacement_body = "I knock once and listen at the sealed door."

    async def run_edit_and_resubmit() -> None:
        replacement_narrator_started = asyncio.Event()
        release_replacement_narrator = asyncio.Event()

        class SlowReplacementNarratorProvider(RuntimeFakeProvider):
            async def chat(self, request: ChatRequest) -> ChatResponse:
                self.chat_requests.append(request)
                assert request.messages[-1].body == replacement_body
                replacement_narrator_started.set()
                await release_replacement_narrator.wait()
                return ChatResponse(
                    body=f"fake narrator: {request.messages[-1].body}",
                    provider=request.provider,
                    model_id=request.model_id,
                    token_usage={"total": 21},
                )

        provider = SlowReplacementNarratorProvider()
        controller = _runtime_controller(
            runtime,
            repositories,
            tmp_path,
            provider=provider,
            context_search_service=NoopContextSearch(),
        )
        controller.load_save(save_id)
        committed_models: list[object] = []

        task = asyncio.create_task(
            controller.edit_and_resubmit_message(
                message_id=ids["player_2"],
                body=replacement_body,
                on_revision_committed=committed_models.append,
            )
        )

        try:
            await asyncio.wait_for(replacement_narrator_started.wait(), timeout=1)
        except TimeoutError as exc:
            if task.done():
                await task
            raise AssertionError("replacement narration did not start") from exc

        assert not task.done()
        assert len(committed_models) == 1
        assert len(provider.chat_requests) == 1
        assert provider.chat_requests[0].messages[-1].body == replacement_body
        active_messages = repositories.list_messages(save_id)
        assert [(message.role, message.body) for message in active_messages] == [
            ("player", "I light the beacon."),
            ("narrator", "The beacon wakes."),
            ("player", replacement_body),
        ]
        _assert_messages_deleted_or_archived(
            repositories,
            [ids["player_2"], ids["narrator_2"], ids["player_3"], ids["narrator_3"]],
        )
        in_flight_model = committed_models[0]
        assert _error_text(in_flight_model) == ""
        assert _value(in_flight_model, "active_save_id") == save_id
        assert [
            _value(message, "body") for message in _chronicle_messages(in_flight_model)
        ] == [
            "I light the beacon.",
            "The beacon wakes.",
            replacement_body,
        ]

        release_replacement_narrator.set()
        final_model = await asyncio.wait_for(task, timeout=1)

        final_messages = repositories.list_messages(save_id)
        assert [(message.role, message.body) for message in final_messages] == [
            ("player", "I light the beacon."),
            ("narrator", "The beacon wakes."),
            ("player", replacement_body),
            ("narrator", f"fake narrator: {replacement_body}"),
        ]
        assert _error_text(final_model) == ""
        assert _value(final_model, "active_save_id") == save_id
        assert [
            _value(message, "body") for message in _chronicle_messages(final_model)
        ] == [
            "I light the beacon.",
            "The beacon wakes.",
            replacement_body,
            f"fake narrator: {replacement_body}",
        ]

    asyncio.run(run_edit_and_resubmit())


def test_message_revision_releases_transaction_before_submitting_replacement_turn(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, ids = _persist_revision_save(repositories)
    _configure_chat_and_context_preferences(repositories)
    observed_transaction_depths: list[int] = []

    class FakeChatService:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def submit_player_turn(
            self,
            *,
            save_id: str,
            body: str,
            speaker_name: str | None = None,
            run_post_turn_jobs: bool = True,
        ) -> object:
            assert run_post_turn_jobs is False
            observed_transaction_depths.append(repositories._transaction_depth)
            player = repositories.append_message(
                save_id=save_id,
                role="player",
                speaker_name=speaker_name,
                body=body,
            )
            narrator = repositories.append_message(
                save_id=save_id,
                role="narrator",
                speaker_name="Narrator",
                body=f"fake narrator: {body}",
                provider="fake",
                model="fake-chat",
            )
            return SimpleNamespace(player_message=player, narrator_message=narrator)

        async def run_post_turn_jobs(self, **_kwargs: object) -> None:
            pytest.fail("revision initial render should not run post-turn jobs")

    monkeypatch.setattr(runtime, "ChatService", FakeChatService)
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    turn = asyncio.run(
        controller.regenerate_message_for_initial_render(message_id=ids["narrator_2"])
    )
    model = _value(turn, "model")

    assert observed_transaction_depths == [0]
    active_messages = repositories.list_messages(save_id)
    assert _value(turn, "save_id") == save_id
    assert _value(turn, "player_message_id") == active_messages[-2].id
    assert _value(turn, "narrator_message_id") == active_messages[-1].id
    assert _value(turn, "has_post_turn_jobs") is True
    assert [message.body for message in repositories.list_messages(save_id)] == [
        "I light the beacon.",
        "The beacon wakes.",
        "I open the sealed door.",
        "fake narrator: I open the sealed door.",
    ]
    assert [_value(message, "body") for message in _chronicle_messages(model)] == [
        "I light the beacon.",
        "The beacon wakes.",
        "I open the sealed door.",
        "fake narrator: I open the sealed door.",
    ]


def test_submit_player_message_waits_for_same_save_revision_restore(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, ids = _persist_revision_save(repositories)
    _configure_chat_and_context_preferences(repositories)
    first_chat_started = asyncio.Event()
    release_first_chat = asyncio.Event()

    class BlockingFailingFirstChatProvider(RuntimeFakeProvider):
        async def chat(self, request: ChatRequest) -> ChatResponse:
            self.chat_requests.append(request)
            if len(self.chat_requests) == 1:
                first_chat_started.set()
                await release_first_chat.wait()
                raise RuntimeError("chat backend leaked sk-live-secret")
            return ChatResponse(
                body=f"fake narrator: {request.messages[-1].body}",
                provider=request.provider,
                model_id=request.model_id,
                token_usage={"total": 21},
            )

    provider = BlockingFailingFirstChatProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)
    original_rows = [
        (message.role, message.body) for message in repositories.list_messages(save_id)
    ]

    async def run_concurrent_revision_and_submit() -> tuple[object, object]:
        revision_task = asyncio.create_task(
            controller.regenerate_message(message_id=ids["narrator_2"])
        )
        await first_chat_started.wait()

        in_flight_revision_rows = [
            (message.role, message.body)
            for message in repositories.list_messages(save_id)
        ]
        assert in_flight_revision_rows == [
            ("player", "I light the beacon."),
            ("narrator", "The beacon wakes."),
            ("player", "I open the sealed door."),
        ]

        submit_task = asyncio.create_task(
            controller.submit_player_message(
                body="I wait for the ash to settle.",
                speaker_name="Mara",
            )
        )
        await asyncio.sleep(0)

        assert not submit_task.done()
        assert len(provider.chat_requests) == 1
        assert [
            (message.role, message.body)
            for message in repositories.list_messages(save_id)
        ] == in_flight_revision_rows

        release_first_chat.set()
        return await revision_task, await submit_task

    revision_model, submit_model = asyncio.run(run_concurrent_revision_and_submit())

    assert len(provider.chat_requests) == 2
    assert provider.chat_requests[0].messages[-1].body == "I open the sealed door."
    assert provider.chat_requests[1].messages[-1].body == (
        "I wait for the ash to settle."
    )
    assert "sk-live-secret" not in _error_text(revision_model)
    assert _error_text(submit_model) == ""
    assert [
        (message.role, message.body) for message in repositories.list_messages(save_id)
    ] == [
        *original_rows,
        ("player", "I wait for the ash to settle."),
        ("narrator", "fake narrator: I wait for the ash to settle."),
    ]


@pytest.mark.parametrize(
    ("action", "expected_request_body", "replacement_body"),
    [
        ("regenerate", "I open the sealed door.", "I open the sealed door."),
        (
            "edit_resubmit",
            "I knock once and listen at the sealed door.",
            "I knock once and listen at the sealed door.",
        ),
    ],
)
def test_message_revision_provider_failure_restores_original_branch(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    action: str,
    expected_request_body: str,
    replacement_body: str,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, ids = _persist_revision_save(repositories)
    _configure_chat_and_context_preferences(repositories)
    provider = FailingRuntimeChatProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)
    original_active_ids = [
        message.id for message in repositories.list_messages(save_id)
    ]
    original_active_rows = [
        (message.role, message.body) for message in repositories.list_messages(save_id)
    ]

    if action == "regenerate":
        model = asyncio.run(controller.regenerate_message(message_id=ids["narrator_2"]))
    else:
        model = asyncio.run(
            controller.edit_and_resubmit_message(
                message_id=ids["player_2"],
                body=replacement_body,
            )
        )

    assert len(provider.chat_requests) == 1
    assert provider.chat_requests[0].messages[-1].body == expected_request_body
    active_messages_after_failure = repositories.list_messages(save_id)
    if action == "regenerate":
        assert [message.id for message in active_messages_after_failure] == (
            original_active_ids
        )
    else:
        assert [message.id for message in active_messages_after_failure] != (
            original_active_ids
        )
    replacement_rows = repositories.connection.execute(
        """
        SELECT id, deleted_at
        FROM messages
        WHERE save_id = ?
          AND id NOT IN ({})
        """.format(", ".join("?" for _ in original_active_ids)),
        (save_id, *original_active_ids),
    ).fetchall()
    assert replacement_rows
    if action == "regenerate":
        assert all(row["deleted_at"] is not None for row in replacement_rows)
        expected_active_rows = original_active_rows
    else:
        expected_active_rows = [
            ("player", "I light the beacon."),
            ("narrator", "The beacon wakes."),
            ("player", replacement_body),
        ]
    assert [
        (message.role, message.body) for message in active_messages_after_failure
    ] == expected_active_rows
    visible_messages = [
        (_value(message, "role"), _value(message, "body"))
        for message in _chronicle_messages(model)
    ]
    assert visible_messages == expected_active_rows
    if action == "regenerate":
        assert visible_messages.count(("player", "I open the sealed door.")) == 1
    else:
        assert ("player", "I open the sealed door.") not in visible_messages
        assert ("player", replacement_body) in visible_messages
    assert "sk-live-secret" not in _error_text(model)


def test_regenerate_message_restores_rollback_when_cancelled(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, ids = _persist_revision_save(repositories)
    _persist_deleted_turn_context(repositories, save_id=save_id, ids=ids)
    _configure_chat_and_context_preferences(repositories)
    provider_started = asyncio.Event()
    release_provider = asyncio.Event()

    class BlockingProvider(RuntimeFakeProvider):
        async def chat(self, request: ChatRequest) -> ChatResponse:
            self.chat_requests.append(request)
            provider_started.set()
            await release_provider.wait()
            pytest.fail("cancelled regeneration should not finish provider chat")

    provider = BlockingProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)
    original_messages = [
        (message.id, message.role, message.body)
        for message in repositories.list_messages(save_id)
    ]
    original_world_state = {
        state.key: state.value for state in repositories.list_world_state(save_id)
    }
    original_memories = [memory.body for memory in repositories.list_memories(save_id)]
    original_summaries = [
        summary.body for summary in repositories.list_summaries(save_id)
    ]

    async def cancel_regeneration() -> None:
        task = asyncio.create_task(
            controller.regenerate_message(message_id=ids["narrator_2"])
        )
        try:
            await asyncio.wait_for(provider_started.wait(), timeout=1.0)
        except TimeoutError as exc:
            if task.done():
                await task
            raise AssertionError("replacement narrator request did not start") from exc

        in_flight_messages = repositories.list_messages(save_id)
        assert [(message.role, message.body) for message in in_flight_messages] == [
            ("player", "I light the beacon."),
            ("narrator", "The beacon wakes."),
            ("player", "I open the sealed door."),
        ]
        assert in_flight_messages[-1].id not in ids.values()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_regeneration())

    assert len(provider.chat_requests) == 1
    assert provider.chat_requests[0].messages[-1].body == "I open the sealed door."
    assert [
        (message.id, message.role, message.body)
        for message in repositories.list_messages(save_id)
    ] == original_messages
    assert {
        state.key: state.value for state in repositories.list_world_state(save_id)
    } == original_world_state
    assert [memory.body for memory in repositories.list_memories(save_id)] == (
        original_memories
    )
    assert [summary.body for summary in repositories.list_summaries(save_id)] == (
        original_summaries
    )


@pytest.mark.parametrize(
    ("player_character_name", "expected_speaker_name"),
    [
        ("Mara Voss", "Mara Voss"),
        ("  ", "Player"),
    ],
)
@pytest.mark.parametrize(
    ("action", "expected_body"),
    [
        ("regenerate", "I open the sealed door."),
        ("edit_resubmit", "I knock once and listen at the sealed door."),
    ],
)
def test_message_revision_defaults_legacy_player_speaker_name_for_resubmission(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    player_character_name: str,
    expected_speaker_name: str,
    action: str,
    expected_body: str,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, ids = _persist_revision_save(
        repositories,
        player_character_name=player_character_name,
        player_2_speaker_name=None,
    )
    _configure_chat_and_context_preferences(repositories)
    provider = RuntimeFakeProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    if action == "regenerate":
        asyncio.run(controller.regenerate_message(message_id=ids["narrator_2"]))
    else:
        asyncio.run(
            controller.edit_and_resubmit_message(
                message_id=ids["player_2"],
                body=expected_body,
            )
        )

    assert len(provider.chat_requests) == 1
    request_player_message = provider.chat_requests[0].messages[-1]
    assert request_player_message.body == expected_body
    assert request_player_message.speaker_name == expected_speaker_name

    active_messages = repositories.list_messages(save_id)
    replacement_player_message = active_messages[-2]
    assert replacement_player_message.id not in ids.values()
    assert replacement_player_message.role == "player"
    assert replacement_player_message.body == expected_body
    assert replacement_player_message.speaker_name == expected_speaker_name
    assert active_messages[-1].body == f"fake narrator: {expected_body}"
    _assert_messages_deleted_or_archived(
        repositories,
        [ids["player_2"], ids["narrator_2"], ids["player_3"], ids["narrator_3"]],
    )


def test_initial_render_submit_returns_persisted_ids_without_post_jobs(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories, include_messages=False)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-chat",
    )
    calls: list[bool] = []
    post_turn_calls: list[tuple[str, str, str]] = []

    class FakeChatService:
        def __init__(self, **_kwargs: object) -> None:
            return None

        async def submit_player_turn(
            self,
            *,
            save_id: str,
            body: str,
            speaker_name: str | None = None,
            run_post_turn_jobs: bool = True,
        ) -> object:
            calls.append(run_post_turn_jobs)
            player = repositories.append_message(
                save_id=save_id,
                role="player",
                speaker_name=speaker_name,
                body=body,
            )
            narrator = repositories.append_message(
                save_id=save_id,
                role="narrator",
                speaker_name="Narrator",
                body=f"fake narrator: {body}",
                provider="fake",
                model="fake-chat",
            )
            if run_post_turn_jobs:
                post_turn_calls.append((save_id, player.id, narrator.id))
            return SimpleNamespace(player_message=player, narrator_message=narrator)

    monkeypatch.setattr(runtime, "ChatService", FakeChatService)
    controller = _runtime_controller(runtime, repositories, tmp_path)
    controller.load_save(save_id)

    turn = asyncio.run(
        controller.submit_player_message_for_initial_render(
            body="I touch the mirror floor.",
            speaker_name="Mara",
        )
    )

    assert calls == [False]
    assert post_turn_calls == []
    assert _value(turn, "save_id") == save_id
    assert _value(turn, "player_message_id") is not None
    assert _value(turn, "narrator_message_id") is not None
    assert _value(turn, "has_post_turn_jobs") is True
    persisted_messages = repositories.list_messages(save_id)
    assert [message.id for message in persisted_messages] == [
        _value(turn, "player_message_id"),
        _value(turn, "narrator_message_id"),
    ]
    delta = _value(turn, "delta")
    assert [_value(message, "body") for message in _chronicle_messages(delta)] == [
        "I touch the mirror floor.",
        "fake narrator: I touch the mirror floor.",
    ]


def test_initial_render_submit_returns_original_save_id_when_active_save_changes(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_a_id, _ = _persist_runtime_save(
        repositories,
        title="Save A",
        include_messages=False,
    )
    save_b_id, _ = _persist_runtime_save(
        repositories,
        title="Save B",
        player_body="B player message",
        narrator_body="B narrator message",
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-chat",
    )
    started = asyncio.Event()
    resume = asyncio.Event()

    class SlowFakeChatService:
        def __init__(self, **_kwargs: object) -> None:
            return None

        async def submit_player_turn(
            self,
            *,
            save_id: str,
            body: str,
            speaker_name: str | None = None,
            run_post_turn_jobs: bool = True,
        ) -> object:
            assert run_post_turn_jobs is False
            started.set()
            await resume.wait()
            player = repositories.append_message(
                save_id=save_id,
                role="player",
                speaker_name=speaker_name,
                body=body,
            )
            narrator = repositories.append_message(
                save_id=save_id,
                role="narrator",
                speaker_name="Narrator",
                body=f"fake narrator: {body}",
                provider="fake",
                model="fake-chat",
            )
            return SimpleNamespace(player_message=player, narrator_message=narrator)

    async def submit_with_save_switch() -> object:
        monkeypatch.setattr(runtime, "ChatService", SlowFakeChatService)
        controller = _runtime_controller(runtime, repositories, tmp_path)
        controller.load_save(save_a_id)
        task = asyncio.create_task(
            controller.submit_player_message_for_initial_render(
                body="A in-flight message",
                speaker_name="Mara",
            )
        )
        await started.wait()
        controller.load_save(save_b_id)
        resume.set()
        return await task

    turn = asyncio.run(submit_with_save_switch())

    assert _value(turn, "save_id") == save_a_id
    assert [
        (message.role, message.body)
        for message in repositories.list_messages(save_a_id)
    ] == [
        ("player", "A in-flight message"),
        ("narrator", "fake narrator: A in-flight message"),
    ]
    assert [
        (message.role, message.body)
        for message in repositories.list_messages(save_b_id)
    ] == [
        ("player", "B player message"),
        ("narrator", "B narrator message"),
    ]

    delta = _value(turn, "delta")
    assert _value(delta, "save_id") == save_a_id
    assert [_value(message, "body") for message in _chronicle_messages(delta)] == [
        "A in-flight message",
        "fake narrator: A in-flight message",
    ]


def test_initial_render_submit_accepts_captured_save_id_when_active_save_changed(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_a_id, _ = _persist_runtime_save(
        repositories,
        title="Save A",
        include_messages=False,
    )
    save_b_id, _ = _persist_runtime_save(
        repositories,
        title="Save B",
        player_body="B player message",
        narrator_body="B narrator message",
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RuntimeFakeProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_a_id)
    controller.load_save(save_b_id)

    turn = asyncio.run(
        controller.submit_player_message_for_initial_render(
            body="A captured message",
            speaker_name="Mara",
            active_save_id=save_a_id,
        )
    )

    assert _value(turn, "save_id") == save_a_id
    assert [
        (message.role, message.body)
        for message in repositories.list_messages(save_a_id)
    ] == [
        ("player", "A captured message"),
        ("narrator", "fake narrator: A captured message"),
    ]
    assert [
        (message.role, message.body)
        for message in repositories.list_messages(save_b_id)
    ] == [
        ("player", "B player message"),
        ("narrator", "B narrator message"),
    ]
    assert _value(_value(turn, "delta"), "save_id") == save_a_id


def test_cancel_active_submit_cancels_in_flight_initial_render_submit(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories, include_messages=False)
    _configure_chat_and_context_preferences(repositories)
    callback_ready = asyncio.Event()
    cancellation_observed = asyncio.Event()
    callbacks: list[Callable[[], bool]] = []
    calls: list[tuple[str, str | None, bool]] = []

    class CancellableFakeChatService:
        def __init__(self, **_kwargs: object) -> None:
            return None

        async def submit_player_turn(
            self,
            *,
            save_id: str,
            body: str,
            speaker_name: str | None = None,
            run_post_turn_jobs: bool = True,
            cancellation_requested: Callable[[], bool],
        ) -> object:
            calls.append((body, speaker_name, run_post_turn_jobs))
            callbacks.append(cancellation_requested)
            callback_ready.set()
            while not cancellation_requested():
                await asyncio.sleep(0)
            cancellation_observed.set()
            raise asyncio.CancelledError

    async def cancel_in_flight_submit() -> object:
        monkeypatch.setattr(runtime, "ChatService", CancellableFakeChatService)
        controller = _runtime_controller(
            runtime,
            repositories,
            tmp_path,
            context_search_service=NoopContextSearch(),
        )
        controller.load_save(save_id)
        task = asyncio.create_task(
            controller.submit_player_message_for_initial_render(
                body="I touch the mirror floor.",
                speaker_name="Mara",
            )
        )
        for _ in range(1000):
            if callback_ready.is_set():
                break
            if task.done():
                result = await task
                pytest.fail(
                    "submit completed before ChatService received "
                    f"cancellation_requested: {result!r}"
                )
            await asyncio.sleep(0)
        else:
            pytest.fail("ChatService did not receive cancellation_requested")

        assert callbacks[0]() is False
        assert controller.cancel_active_submit(save_id=save_id) is True
        assert callbacks[0]() is True
        await asyncio.wait_for(cancellation_observed.wait(), timeout=1.0)
        return await task

    turn = asyncio.run(cancel_in_flight_submit())

    assert calls == [("I touch the mirror floor.", "Mara", False)]
    assert repositories.list_messages(save_id) == []
    assert _value(turn, "save_id") == save_id
    assert _value(turn, "player_message_id") is None
    assert _value(turn, "narrator_message_id") is None
    assert _value(turn, "input_committed") is False
    assert _value(turn, "has_post_turn_jobs") is False

    model = _value(turn, "model")
    assert "cancel" in _error_text(model).casefold()
    assert _value(model, "active_save_id") == save_id


def test_pending_initial_render_cancel_does_not_poison_next_submit(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories, include_messages=False)
    _configure_chat_and_context_preferences(repositories)
    provider = RuntimeFakeProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    assert controller.cancel_active_submit(save_id=save_id) is False

    turn = asyncio.run(
        controller.submit_player_message_for_initial_render(
            body="I touch the mirror floor.",
            speaker_name="Mara",
        )
    )

    persisted_messages = repositories.list_messages(save_id)
    assert [(message.role, message.body) for message in persisted_messages] == [
        ("player", "I touch the mirror floor."),
        ("narrator", "fake narrator: I touch the mirror floor."),
    ]
    assert len(provider.chat_requests) == 1
    assert _runtime_chat_completion_job_count(repositories, save_id) == 1
    assert _value(turn, "save_id") == save_id
    assert _value(turn, "player_message_id") == persisted_messages[0].id
    assert _value(turn, "narrator_message_id") == persisted_messages[1].id
    assert _value(turn, "input_committed") is True
    assert _value(turn, "has_post_turn_jobs") is True

    delta = _value(turn, "delta")
    assert _error_text(delta) == ""
    assert _value(delta, "save_id") == save_id


def test_stale_initial_render_cancel_after_completed_submit_does_not_poison_next_submit(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories, include_messages=False)
    _configure_chat_and_context_preferences(repositories)
    provider = RuntimeFakeProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    first_turn = asyncio.run(
        controller.submit_player_message_for_initial_render(
            body="I touch the mirror floor.",
            speaker_name="Mara",
        )
    )

    assert _value(first_turn, "input_committed") is True
    assert controller.cancel_active_submit(save_id=save_id) is False

    second_turn = asyncio.run(
        controller.submit_player_message_for_initial_render(
            body="I follow the reflected stairs.",
            speaker_name="Mara",
        )
    )

    assert len(provider.chat_requests) == 2
    assert [
        (message.role, message.body)
        for message in repositories.list_messages(save_id)
    ] == [
        ("player", "I touch the mirror floor."),
        ("narrator", "fake narrator: I touch the mirror floor."),
        ("player", "I follow the reflected stairs."),
        ("narrator", "fake narrator: I follow the reflected stairs."),
    ]
    assert _value(second_turn, "input_committed") is True
    assert _value(second_turn, "has_post_turn_jobs") is True
    assert _error_text(_value(second_turn, "model")) == ""


def test_too_late_initial_render_cancel_after_token_deactivation_is_ignored(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories, include_messages=False)
    _configure_chat_and_context_preferences(repositories)
    token_deactivated = asyncio.Event()
    release_first_submit = asyncio.Event()
    bodies: list[str] = []

    class DeactivatingFakeChatService:
        def __init__(self, **_kwargs: object) -> None:
            return None

        async def submit_player_turn(
            self,
            *,
            save_id: str,
            body: str,
            speaker_name: str | None = None,
            run_post_turn_jobs: bool = True,
            cancellation_token: Any,
        ) -> object:
            assert run_post_turn_jobs is False
            bodies.append(body)
            if len(bodies) == 1:
                cancellation_token.deactivate()
                token_deactivated.set()
                await release_first_submit.wait()
            player = repositories.append_message(
                save_id=save_id,
                role="player",
                speaker_name=speaker_name,
                body=body,
            )
            narrator = repositories.append_message(
                save_id=save_id,
                role="narrator",
                speaker_name="Narrator",
                body=f"fake narrator: {body}",
                provider="fake",
                model="fake-chat",
            )
            return SimpleNamespace(player_message=player, narrator_message=narrator)

    async def run_too_late_cancel() -> tuple[object, object]:
        monkeypatch.setattr(runtime, "ChatService", DeactivatingFakeChatService)
        controller = _runtime_controller(
            runtime,
            repositories,
            tmp_path,
            context_search_service=NoopContextSearch(),
        )
        controller.load_save(save_id)
        first_task = asyncio.create_task(
            controller.submit_player_message_for_initial_render(
                body="I touch the mirror floor.",
                speaker_name="Mara",
            )
        )
        await asyncio.wait_for(token_deactivated.wait(), timeout=1.0)

        assert not first_task.done()
        assert controller.cancel_active_submit(save_id=save_id) is False

        release_first_submit.set()
        first_turn = await asyncio.wait_for(first_task, timeout=1.0)
        second_turn = await asyncio.wait_for(
            controller.submit_player_message_for_initial_render(
                body="I follow the reflected stairs.",
                speaker_name="Mara",
            ),
            timeout=1.0,
        )
        return first_turn, second_turn

    first_turn, second_turn = asyncio.run(run_too_late_cancel())

    assert bodies == [
        "I touch the mirror floor.",
        "I follow the reflected stairs.",
    ]
    assert [
        (message.role, message.body)
        for message in repositories.list_messages(save_id)
    ] == [
        ("player", "I touch the mirror floor."),
        ("narrator", "fake narrator: I touch the mirror floor."),
        ("player", "I follow the reflected stairs."),
        ("narrator", "fake narrator: I follow the reflected stairs."),
    ]
    assert _value(first_turn, "input_committed") is True
    assert _value(first_turn, "has_post_turn_jobs") is True
    assert _error_text(_value(first_turn, "model")) == ""
    assert _value(second_turn, "input_committed") is True
    assert _value(second_turn, "has_post_turn_jobs") is True
    assert _error_text(_value(second_turn, "model")) == ""


def test_cancel_queued_initial_render_submit_before_provider_call(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories, include_messages=False)
    _configure_chat_and_context_preferences(repositories)
    provider_calls: list[str] = []

    class CancellingFakeChatService:
        def __init__(self, **_kwargs: object) -> None:
            return None

        async def submit_player_turn(
            self,
            *,
            save_id: str,
            body: str,
            speaker_name: str | None = None,
            run_post_turn_jobs: bool = True,
            cancellation_token: Any,
        ) -> object:
            assert run_post_turn_jobs is False
            assert body == "queued-lock body"
            cancellation_token.throw_if_cancelled()
            provider_calls.append(body)
            player = repositories.append_message(
                save_id=save_id,
                role="player",
                speaker_name=speaker_name,
                body=body,
            )
            narrator = repositories.append_message(
                save_id=save_id,
                role="narrator",
                speaker_name="Narrator",
                body=f"fake narrator: {body}",
                provider="fake",
                model="fake-chat",
            )
            return SimpleNamespace(player_message=player, narrator_message=narrator)

    async def run_queued_cancel() -> object:
        monkeypatch.setattr(runtime, "ChatService", CancellingFakeChatService)
        controller = _runtime_controller(
            runtime,
            repositories,
            tmp_path,
            context_search_service=NoopContextSearch(),
        )
        controller.load_save(save_id)
        save_lock = asyncio.Lock()
        queued_submit_waiting = asyncio.Event()

        class InstrumentedSaveLock:
            async def __aenter__(self) -> None:
                if save_lock.locked():
                    queued_submit_waiting.set()
                await save_lock.acquire()

            async def __aexit__(
                self,
                _exc_type: object,
                _exc: object,
                _traceback: object,
            ) -> None:
                save_lock.release()

        def save_operation_lock(_save_id: str) -> InstrumentedSaveLock:
            return InstrumentedSaveLock()

        monkeypatch.setattr(controller, "_save_operation_lock", save_operation_lock)
        await save_lock.acquire()
        task = asyncio.create_task(
            controller.submit_player_message_for_initial_render(
                body="queued-lock body",
                speaker_name="Mara",
            )
        )
        await asyncio.wait_for(queued_submit_waiting.wait(), timeout=1.0)

        assert controller.cancel_active_submit(save_id=save_id) is True

        save_lock.release()
        return await asyncio.wait_for(task, timeout=1.0)

    turn = asyncio.run(run_queued_cancel())

    assert provider_calls == []
    assert repositories.list_messages(save_id) == []
    assert _value(turn, "player_message_id") is None
    assert _value(turn, "narrator_message_id") is None
    assert _value(turn, "input_committed") is False
    assert _value(turn, "has_post_turn_jobs") is False
    assert "cancel" in _error_text(_value(turn, "model")).casefold()


def test_state_pruning_provider_wait_does_not_block_new_initial_render_submit(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories, include_messages=False)
    _configure_chat_and_context_preferences(repositories)
    repositories.set_model_preference(
        task="state_pruning",
        provider="fake",
        model_id="fake-pruner",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-pruner",
        display_name="Fake Pruner",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    repositories.upsert_world_state(
        save_id=save_id,
        key="scene.old_alarm",
        value={"status": "disabled"},
        category="scene",
    )
    submitted_bodies: list[str] = []

    class BlockingStatePruningProvider(RuntimeFakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.structured_requests: list[StructuredOutputRequest] = []

        async def generate_structured_output(
            self,
            request: StructuredOutputRequest,
        ) -> StructuredOutputResponse:
            self.structured_requests.append(request)
            self.started.set()
            await self.release.wait()
            return StructuredOutputResponse(
                data={"archives": []},
                provider=request.provider,
                model_id=request.model_id,
            )

    class RecordingFakeChatService:
        def __init__(self, **_kwargs: object) -> None:
            return None

        async def submit_player_turn(
            self,
            *,
            save_id: str,
            body: str,
            speaker_name: str | None = None,
            run_post_turn_jobs: bool = True,
            cancellation_token: Any,
        ) -> object:
            assert run_post_turn_jobs is False
            cancellation_token.throw_if_cancelled()
            submitted_bodies.append(body)
            player = repositories.append_message(
                save_id=save_id,
                role="player",
                speaker_name=speaker_name,
                body=body,
            )
            narrator = repositories.append_message(
                save_id=save_id,
                role="narrator",
                speaker_name="Narrator",
                body=f"fake narrator: {body}",
                provider="fake",
                model="fake-chat",
            )
            return SimpleNamespace(player_message=player, narrator_message=narrator)

    async def run_pruning_and_submit() -> tuple[object, object]:
        monkeypatch.setattr(runtime, "ChatService", RecordingFakeChatService)
        provider = BlockingStatePruningProvider()
        controller = _runtime_controller(
            runtime,
            repositories,
            tmp_path,
            provider=provider,
            context_search_service=NoopContextSearch(),
        )
        controller.load_save(save_id)
        pruning_task = asyncio.create_task(
            controller.run_state_pruning(active_save_id=save_id)
        )
        await asyncio.wait_for(provider.started.wait(), timeout=1.0)

        turn = await asyncio.wait_for(
            controller.submit_player_message_for_initial_render(
                body="I keep moving while cleanup thinks.",
                speaker_name="Mara",
            ),
            timeout=1.0,
        )
        provider.release.set()
        pruning_model = await asyncio.wait_for(pruning_task, timeout=1.0)
        return turn, pruning_model

    turn, pruning_model = asyncio.run(run_pruning_and_submit())

    assert submitted_bodies == ["I keep moving while cleanup thinks."]
    assert _error_text(_value(turn, "delta")) == ""
    assert _status_text(pruning_model) == "World state cleanup complete."


def test_context_update_retry_wait_does_not_block_initial_render_input_save(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories, include_messages=False)
    _configure_chat_and_context_preferences(repositories)
    retry_started = asyncio.Event()
    release_retry = asyncio.Event()
    input_saved = asyncio.Event()
    post_input_barrier_released = asyncio.Event()

    class BlockingContextRetryChatService:
        def __init__(self, **_kwargs: object) -> None:
            return None

        async def run_context_update_retries(
            self,
            *,
            save_id: str | None = None,
        ) -> int:
            assert save_id == save_id_fixture
            retry_started.set()
            await release_retry.wait()
            return 1

        async def submit_player_turn(
            self,
            *,
            save_id: str,
            body: str,
            speaker_name: str | None = None,
            run_post_turn_jobs: bool = True,
            cancellation_token: Any,
            post_input_context: Callable[[], Any] | None = None,
        ) -> object:
            assert run_post_turn_jobs is False
            assert save_id == save_id_fixture
            cancellation_token.throw_if_cancelled()
            player = repositories.append_message(
                save_id=save_id,
                role="player",
                speaker_name=speaker_name,
                body=body,
            )
            input_saved.set()
            assert post_input_context is not None
            async with post_input_context():
                post_input_barrier_released.set()
            narrator = repositories.append_message(
                save_id=save_id,
                role="narrator",
                speaker_name="Narrator",
                body=f"fake narrator: {body}",
                provider="fake",
                model="fake-chat",
            )
            return SimpleNamespace(player_message=player, narrator_message=narrator)

    save_id_fixture = save_id

    async def run_retry_and_submit() -> tuple[object, object]:
        monkeypatch.setattr(runtime, "ChatService", BlockingContextRetryChatService)
        controller = _runtime_controller(
            runtime,
            repositories,
            tmp_path,
            context_search_service=NoopContextSearch(),
        )
        controller.load_save(save_id)
        retry_task = asyncio.create_task(
            controller.run_context_update_retries(active_save_id=save_id)
        )
        await asyncio.wait_for(retry_started.wait(), timeout=1.0)

        submit_task = asyncio.create_task(
            controller.submit_player_message_for_initial_render(
                body="I keep moving while context retries finish.",
                speaker_name="Mara",
            )
        )
        await asyncio.wait_for(input_saved.wait(), timeout=1.0)

        assert [
            (message.role, message.body)
            for message in repositories.list_messages(save_id)
        ] == [
            ("player", "I keep moving while context retries finish."),
        ]
        assert not post_input_barrier_released.is_set()
        assert not submit_task.done()

        release_retry.set()
        return (
            await asyncio.wait_for(submit_task, timeout=1.0),
            await asyncio.wait_for(retry_task, timeout=1.0),
        )

    turn, retry_model = asyncio.run(run_retry_and_submit())

    assert _value(turn, "input_committed") is True
    assert _error_text(_value(turn, "delta")) == ""
    assert _status_text(retry_model) == "Context update retries finished: 1 completed."
    assert [
        (message.role, message.body)
        for message in repositories.list_messages(save_id)
    ] == [
        ("player", "I keep moving while context retries finish."),
        ("narrator", "fake narrator: I keep moving while context retries finish."),
    ]


def test_cancelled_async_save_lock_waiter_does_not_leak_thread_lock(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories, include_messages=False)
    controller = _runtime_controller(runtime, repositories, tmp_path)
    controller.load_save(save_id)

    class InstrumentedThreadLock:
        def __init__(self) -> None:
            self._condition = threading.Condition()
            self._locked = True
            self.acquire_started = threading.Event()
            self.release_count = 0

        def acquire(self) -> bool:
            self.acquire_started.set()
            with self._condition:
                while self._locked:
                    self._condition.wait()
                self._locked = True
            return True

        def release(self) -> None:
            with self._condition:
                if not self._locked:
                    raise RuntimeError("cannot release an unlocked lock")
                self._locked = False
                self.release_count += 1
                self._condition.notify_all()

    lock = InstrumentedThreadLock()

    def thread_save_operation_lock(requested_save_id: str) -> InstrumentedThreadLock:
        assert requested_save_id == save_id
        return lock

    async def acquire_then_cancel() -> None:
        entered = False

        async def wait_on_save_lock() -> None:
            nonlocal entered
            async with controller._save_operation_lock(save_id):
                entered = True

        monkeypatch.setattr(
            controller,
            "_thread_save_operation_lock",
            thread_save_operation_lock,
        )
        task = asyncio.create_task(wait_on_save_lock())
        await asyncio.to_thread(lock.acquire_started.wait, 1.0)
        assert lock.acquire_started.is_set()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert entered is False

        lock.release()
        deadline = asyncio.get_running_loop().time() + 1.0
        while lock.release_count < 2 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)

        assert lock.release_count == 2
        async with controller._save_operation_lock(save_id):
            entered = True
        assert entered is True
        assert lock.release_count == 3

    asyncio.run(acquire_then_cancel())


def test_run_post_turn_jobs_returns_model_for_submitted_save_when_active_save_changes(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_a_id, narrator_a_id = _persist_runtime_save(
        repositories,
        title="Save A",
        player_body="A player message",
        narrator_body="A narrator message",
    )
    save_b_id, _ = _persist_runtime_save(
        repositories,
        title="Save B",
        player_body="B player message",
        narrator_body="B narrator message",
    )
    player_a_id = repositories.list_messages(save_a_id)[0].id
    calls: list[tuple[str, str, str]] = []

    class FakeChatService:
        def __init__(self, **_kwargs: object) -> None:
            return None

        async def run_post_turn_jobs(
            self,
            *,
            save_id: str,
            player_message_id: str,
            narrator_message_id: str,
        ) -> None:
            calls.append((save_id, player_message_id, narrator_message_id))

    monkeypatch.setattr(runtime, "ChatService", FakeChatService)
    controller = _runtime_controller(runtime, repositories, tmp_path)
    controller.load_save(save_b_id)

    model = asyncio.run(
        controller.run_post_turn_jobs(
            save_id=save_a_id,
            player_message_id=player_a_id,
            narrator_message_id=narrator_a_id,
        )
    )

    assert calls == [(save_a_id, player_a_id, narrator_a_id)]
    assert controller.active_save_id == save_b_id
    assert _value(model, "active_save_id") == save_a_id
    assert [_value(message, "body") for message in _chronicle_messages(model)] == [
        "A player message",
        "A narrator message",
    ]


def test_run_post_turn_jobs_forwards_progress_callback_when_supported(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, narrator_id = _persist_runtime_save(repositories)
    player_id = repositories.list_messages(save_id)[0].id
    calls: list[tuple[str, str, str]] = []
    progress_texts: list[str] = []

    class FakeChatService:
        def __init__(self, **_kwargs: object) -> None:
            return None

        async def run_post_turn_jobs(
            self,
            *,
            save_id: str,
            player_message_id: str,
            narrator_message_id: str,
            progress_callback: Callable[[Any], None] | None = None,
        ) -> None:
            calls.append((save_id, player_message_id, narrator_message_id))
            if progress_callback is not None:
                progress_callback(
                    SimpleNamespace(status_text="Post-turn: state running")
                )

    monkeypatch.setattr(runtime, "ChatService", FakeChatService)
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = asyncio.run(
        controller.run_post_turn_jobs(
            save_id=save_id,
            player_message_id=player_id,
            narrator_message_id=narrator_id,
            progress_callback=lambda progress: progress_texts.append(
                progress.status_text
            ),
        )
    )

    assert calls == [(save_id, player_id, narrator_id)]
    assert progress_texts == ["Post-turn: state running"]
    assert _value(model, "status") == "Turn complete"


def test_run_post_turn_jobs_waits_for_same_save_submit_lock(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, narrator_id = _persist_runtime_save(repositories)
    _configure_chat_and_context_preferences(repositories)
    player_id = repositories.list_messages(save_id)[0].id
    submit_started = asyncio.Event()
    release_submit = asyncio.Event()
    calls: list[tuple[str, str]] = []

    class FakeChatService:
        def __init__(self, **_kwargs: object) -> None:
            return None

        async def submit_player_turn(
            self,
            *,
            save_id: str,
            body: str,
            speaker_name: str | None = None,
            run_post_turn_jobs: bool = True,
        ) -> object:
            assert run_post_turn_jobs is True
            calls.append(("submit_started", save_id))
            submit_started.set()
            await release_submit.wait()
            player = repositories.append_message(
                save_id=save_id,
                role="player",
                speaker_name=speaker_name,
                body=body,
            )
            narrator = repositories.append_message(
                save_id=save_id,
                role="narrator",
                speaker_name="Narrator",
                body=f"fake narrator: {body}",
                provider="fake",
                model="fake-chat",
            )
            calls.append(("submit_finished", save_id))
            return SimpleNamespace(player_message=player, narrator_message=narrator)

        async def run_post_turn_jobs(
            self,
            *,
            save_id: str,
            player_message_id: str,
            narrator_message_id: str,
        ) -> None:
            assert player_message_id == player_id
            assert narrator_message_id == narrator_id
            calls.append(("post_jobs", save_id))

    monkeypatch.setattr(runtime, "ChatService", FakeChatService)
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    async def run_concurrent_submit_and_post_jobs() -> tuple[object, object]:
        submit_task = asyncio.create_task(
            controller.submit_player_message(
                body="I keep the beacon lit.",
                speaker_name="Mara",
            )
        )
        await submit_started.wait()

        post_jobs_task = asyncio.create_task(
            controller.run_post_turn_jobs(
                save_id=save_id,
                player_message_id=player_id,
                narrator_message_id=narrator_id,
            )
        )
        await asyncio.sleep(0)

        assert not post_jobs_task.done()
        assert calls == [("submit_started", save_id)]

        release_submit.set()
        return await submit_task, await post_jobs_task

    submit_model, post_jobs_model = asyncio.run(run_concurrent_submit_and_post_jobs())

    assert calls == [
        ("submit_started", save_id),
        ("submit_finished", save_id),
        ("post_jobs", save_id),
    ]
    assert _error_text(submit_model) == ""
    assert _error_text(post_jobs_model) == ""
    assert [
        (message.role, message.body) for message in repositories.list_messages(save_id)
    ] == [
        ("player", "I climb toward the beacon lens."),
        ("narrator", "Ash scratches the glass as the stair shakes."),
        ("player", "I keep the beacon lit."),
        ("narrator", "fake narrator: I keep the beacon lit."),
    ]


def test_generate_image_waits_for_same_save_submit_lock_before_media_service(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, narrator_id = _persist_runtime_save(repositories)
    _configure_chat_and_context_preferences(repositories)
    repositories.set_model_preference(
        task="image_generation",
        provider="fake",
        model_id="fake-image",
    )
    submit_started = asyncio.Event()
    release_submit = asyncio.Event()
    calls: list[tuple[str, str]] = []

    class FakeChatService:
        def __init__(self, **_kwargs: object) -> None:
            return None

        async def submit_player_turn(
            self,
            *,
            save_id: str,
            body: str,
            speaker_name: str | None = None,
            run_post_turn_jobs: bool = True,
        ) -> object:
            assert run_post_turn_jobs is True
            calls.append(("submit_started", save_id))
            submit_started.set()
            await release_submit.wait()
            player = repositories.append_message(
                save_id=save_id,
                role="player",
                speaker_name=speaker_name,
                body=body,
            )
            narrator = repositories.append_message(
                save_id=save_id,
                role="narrator",
                speaker_name="Narrator",
                body=f"fake narrator: {body}",
                provider="fake",
                model="fake-chat",
            )
            calls.append(("submit_finished", save_id))
            return SimpleNamespace(player_message=player, narrator_message=narrator)

    class FakeMediaService:
        def __init__(self, **_kwargs: object) -> None:
            return None

        async def generate_for_message(
            self,
            *,
            save_id: str,
            source_message_id: str,
        ) -> object:
            assert source_message_id == narrator_id
            calls.append(("media_generate", save_id))
            return SimpleNamespace()

    monkeypatch.setattr(runtime, "ChatService", FakeChatService)
    monkeypatch.setattr(runtime, "MediaService", FakeMediaService)
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    async def run_concurrent_submit_and_image() -> tuple[object, object]:
        submit_task = asyncio.create_task(
            controller.submit_player_message(
                body="I keep the beacon lit.",
                speaker_name="Mara",
            )
        )
        await submit_started.wait()

        image_task = asyncio.create_task(
            controller.generate_image(source_message_id=narrator_id)
        )
        await asyncio.sleep(0)

        assert not image_task.done()
        assert calls == [("submit_started", save_id)]

        release_submit.set()
        return await submit_task, await image_task

    submit_model, image_model = asyncio.run(run_concurrent_submit_and_image())

    assert calls == [
        ("submit_started", save_id),
        ("submit_finished", save_id),
        ("media_generate", save_id),
    ]
    assert _error_text(submit_model) == ""
    assert _error_text(image_model) == ""


def test_generate_image_for_narrator_message_updates_latest_image(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, narrator_id = _persist_runtime_save(repositories)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="image_generation",
        provider="fake",
        model_id="fake-image",
    )
    provider = RuntimeFakeProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    model = asyncio.run(_generate_image_for_message(controller, narrator_id))

    assert len(provider.chat_requests) == 1
    assert len(provider.image_requests) == 1
    assert provider.image_requests[0].prompt == "cinematic drafted image prompt"
    latest_image = _latest_image(model)
    assert _value(latest_image, "source_message_id") == narrator_id
    assert _value(latest_image, "provider") == "fake"
    assert _value(latest_image, "model") == "fake-image"
    image_path = tmp_path / "media" / _value(latest_image, "path")
    assert image_path.read_bytes() == b"runtime fake scene image"


def test_generate_image_skips_unavailable_image_prompt_model(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, narrator_id = _persist_runtime_save(repositories)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="image_prompt",
        provider="fake",
        model_id="fake-stale-prompt",
    )
    repositories.set_model_preference(
        task="image_generation",
        provider="fake",
        model_id="fake-image",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-stale-prompt",
        display_name="Fake Stale Prompt",
        capabilities=[ProviderCapability.CHAT.value],
    )
    repositories.mark_missing_provider_models_unavailable(
        provider="fake",
        available_model_ids=set(),
    )
    provider = RuntimeFakeProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    model = asyncio.run(_generate_image_for_message(controller, narrator_id))

    assert len(provider.chat_requests) == 1
    assert provider.chat_requests[0].model_id == "fake-chat"
    assert len(provider.image_requests) == 1
    latest_image = _latest_image(model)
    assert _value(latest_image, "source_message_id") == narrator_id


def test_generate_image_passes_retry_progress_callback_to_image_provider(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, narrator_id = _persist_runtime_save(repositories)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="image_generation",
        provider="fake",
        model_id="fake-image",
    )
    provider = RuntimeFakeProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)
    progress_events: list[ProviderRetryProgress] = []

    def callback(progress: ProviderRetryProgress) -> None:
        progress_events.append(progress)

    asyncio.run(
        controller.generate_image(
            source_message_id=narrator_id,
            retry_progress_callback=callback,
        )
    )

    assert len(provider.image_requests) == 1
    assert provider.image_requests[0].retry_progress_callback is callback
    assert progress_events == []


def test_generate_image_accepts_captured_save_id_when_active_save_changed(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_a_id, narrator_a_id = _persist_runtime_save(
        repositories,
        title="Save A",
    )
    save_b_id, _ = _persist_runtime_save(
        repositories,
        title="Save B",
        player_body="B player message",
        narrator_body="B narrator message",
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="image_generation",
        provider="fake",
        model_id="fake-image",
    )
    provider = RuntimeFakeProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_a_id)
    controller.load_save(save_b_id)

    model = asyncio.run(
        controller.generate_image(
            source_message_id=narrator_a_id,
            active_save_id=save_a_id,
        )
    )

    assert len(provider.chat_requests) == 1
    assert len(provider.image_requests) == 1
    assert provider.image_requests[0].prompt == "cinematic drafted image prompt"
    save_a_images = repositories.list_media_assets(save_a_id)
    save_b_images = repositories.list_media_assets(save_b_id)
    assert len(save_a_images) == 1
    assert save_a_images[0].source_message_id == narrator_a_id
    assert save_b_images == []
    assert _value(model, "active_save_id") == save_a_id
    assert _value(_latest_image(model), "source_message_id") == narrator_a_id


def test_open_world_data_editor_without_active_save_returns_error(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    _persist_runtime_save(repositories)
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = _open_world_data_editor(controller)

    assert "No save loaded" in _error_text(model)
    world_data_model = _value(model, "world_data", "world_data_editor", default=None)
    if world_data_model is not None:
        assert _value(world_data_model, "active_save_id", default=None) is None
        assert tuple(_value(world_data_model, "world_state", default=())) == ()
        assert tuple(_value(world_data_model, "memories", default=())) == ()
        assert tuple(_value(world_data_model, "summaries", default=())) == ()


def test_open_character_registry_exposes_active_save_characters(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories)
    character = repositories.add_character(
        save_id=save_id,
        name="Captain Ilyra",
        aliases=["Ashknife"],
        role="Watch captain",
    )
    controller = _runtime_controller(runtime, repositories, tmp_path)
    controller.load_save(save_id)

    model = _open_character_registry(controller)

    assert _status_text(model) == "Characters opened"
    assert _value(model, "active_save_id") == save_id
    registry_model = controller.build_character_registry_model()
    assert _value(registry_model, "save_id") == save_id
    rows = tuple(_value(registry_model, "characters"))
    assert [(row.character_id, row.name, row.aliases_text) for row in rows] == [
        (character.id, "Captain Ilyra", "Ashknife"),
    ]


@pytest.mark.parametrize(
    ("missing_task", "expected_status_text"),
    [
        ("chat", "chat"),
        ("context_search", "context"),
    ],
)
def test_submit_player_message_missing_preferences_returns_error_without_persisting(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    missing_task: str,
    expected_status_text: str,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories, include_messages=False)
    if missing_task != "chat":
        repositories.set_model_preference(
            task="chat",
            provider="fake",
            model_id="fake-chat",
        )
    if missing_task != "context_search":
        repositories.set_model_preference(
            task="context_search",
            provider="fake",
            model_id="fake-chat",
        )
    provider = RuntimeFakeProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
    )
    controller.load_save(save_id)

    model = asyncio.run(
        controller.submit_player_message(
            body="I should not be persisted on missing preferences.",
            speaker_name="Mara",
        )
    )

    assert repositories.list_messages(save_id) == []
    assert provider.chat_requests == []
    assert expected_status_text in _status_text(model).casefold()


def test_submit_player_message_provider_failure_returns_redacted_ui_error(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories, include_messages=False)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-chat",
    )
    provider = FailingRuntimeChatProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    model = asyncio.run(
        controller.submit_player_message(
            body="I touch the mirror floor.",
            speaker_name="Mara",
        )
    )

    error_text = _error_text(model)
    assert error_text
    assert "sk-live-secret" not in error_text


def test_submit_player_message_exhausted_retry_error_mentions_attempts(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories, include_messages=False)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-chat",
    )

    class ExhaustedRetryProvider(RuntimeFakeProvider):
        async def chat(self, request: ChatRequest) -> ChatResponse:
            self.chat_requests.append(request)
            raise ProviderError(
                ProviderErrorCategory.RATE_LIMITED,
                "provider said sk-live-secret is over quota",
                status_code=429,
                retry_attempt_count=3,
                max_retry_attempts=3,
                retry_attempts=(
                    {
                        "attempt": 1,
                        "error_category": "rate_limited",
                        "duration_ms": 10,
                        "http_status": 429,
                    },
                    {
                        "attempt": 2,
                        "error_category": "rate_limited",
                        "duration_ms": 11,
                        "http_status": 429,
                    },
                    {
                        "attempt": 3,
                        "error_category": "rate_limited",
                        "duration_ms": 12,
                        "http_status": 429,
                    },
                ),
            )

    provider = ExhaustedRetryProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    model = asyncio.run(
        controller.submit_player_message(
            body="I touch the mirror floor.",
            speaker_name="Mara",
        )
    )

    error_text = _error_text(model)
    assert "after 3 attempts" in error_text
    assert "sk-live-secret" not in error_text


def test_initial_render_provider_failure_returns_committed_player_message(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories, include_messages=False)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-chat",
    )
    provider = FailingRuntimeChatProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    turn = asyncio.run(
        controller.submit_player_message_for_initial_render(
            body="I touch the mirror floor.",
            speaker_name="Mara",
        )
    )

    persisted_messages = repositories.list_messages(save_id)
    assert len(provider.chat_requests) == 1
    assert [(message.role, message.body) for message in persisted_messages] == [
        ("player", "I touch the mirror floor.")
    ]
    assert _value(turn, "save_id") == save_id
    assert _value(turn, "player_message_id") == persisted_messages[0].id
    assert _value(turn, "narrator_message_id") is None
    assert _value(turn, "input_committed") is True
    assert _value(turn, "has_post_turn_jobs") is False

    model = _value(turn, "model")
    error_text = _error_text(model)
    assert error_text
    assert "sk-live-secret" not in error_text
    assert _value(model, "active_save_id") == save_id
    assert [_value(message, "body") for message in _chronicle_messages(model)] == [
        "I touch the mirror floor."
    ]


@pytest.mark.parametrize(
    ("player_character_name", "expected_speaker_name"),
    [
        ("Mara Voss", "Mara Voss"),
        ("Ren", "Ren"),
        ("  ", "Player"),
    ],
)
def test_provider_failure_without_speaker_name_commits_defaulted_player_message(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    player_character_name: str,
    expected_speaker_name: str,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(
        repositories,
        include_messages=False,
        player_character_name=player_character_name,
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-chat",
    )
    provider = FailingRuntimeChatProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    turn = asyncio.run(
        controller.submit_player_message_for_initial_render(
            body="I touch the mirror floor."
        )
    )

    persisted_messages = repositories.list_messages(save_id)
    assert len(provider.chat_requests) == 1
    assert len(persisted_messages) == 1
    persisted_message = persisted_messages[0]
    assert persisted_message.role == "player"
    assert persisted_message.body == "I touch the mirror floor."
    assert persisted_message.speaker_name == expected_speaker_name
    assert _value(turn, "save_id") == save_id
    assert _value(turn, "player_message_id") == persisted_message.id
    assert _value(turn, "narrator_message_id") is None
    assert _value(turn, "input_committed") is True
    assert _value(turn, "has_post_turn_jobs") is False

    model = _value(turn, "model")
    error_text = _error_text(model)
    assert error_text
    assert "sk-live-secret" not in error_text
    assert _value(model, "active_save_id") == save_id
    assert [
        (_value(message, "speaker_name"), _value(message, "body"))
        for message in _chronicle_messages(model)
    ] == [(expected_speaker_name, "I touch the mirror floor.")]


def test_generate_image_provider_failure_returns_redacted_ui_error(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, narrator_id = _persist_runtime_save(repositories)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="image_generation",
        provider="fake",
        model_id="fake-image",
    )
    provider = FailingRuntimeImageProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
        context_search_service=NoopContextSearch(),
    )
    controller.load_save(save_id)

    model = asyncio.run(
        controller.generate_image(
            source_message_id=narrator_id,
        )
    )

    error_text = _error_text(model)
    assert error_text
    assert "sk-live-secret" not in error_text


def test_run_context_cleanup_without_active_save_returns_error(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    controller = _runtime_controller(runtime, repositories, tmp_path)

    model = asyncio.run(controller.run_context_cleanup())

    assert _error_text(model) == "No save loaded"


def test_run_summary_backfill_compacts_save_and_can_apply_windows(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories, include_messages=False)
    for index in range(1, 7):
        role = "player" if index % 2 else "narrator"
        repositories.append_message(
            save_id=save_id,
            role=role,
            speaker_name="Mara" if role == "player" else "Narrator",
            body=(
                f"Beacon continuity event {index}: Mara tracks the lens, "
                "storm pressure, and unresolved signal duty."
            ),
            provider=None if role == "player" else "fake",
            model=None if role == "player" else "fake-chat",
            token_estimate=80,
            message_id=f"summary-backfill-message-{index}",
        )
    repositories.set_model_preference(
        task="summarization",
        provider="fake",
        model_id="fake-summary",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-summary",
        display_name="Fake Summary",
        capabilities=[ProviderCapability.CHAT.value],
        context_window=1024,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save_id,
        key="recent_player_message_window",
        value=18,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save_id,
        key="recent_narrator_message_window",
        value=16,
    )
    provider = RuntimeSummaryProvider(
        [
            (
                "Mara maintained the beacon continuity ledger while the latest "
                "exchange remains in the recent chronicle."
            )
        ]
    )
    controller = _runtime_controller(runtime, repositories, tmp_path, provider=provider)

    model = asyncio.run(
        controller.run_summary_backfill(
            active_save_id=save_id,
            apply_recommended_windows=True,
        )
    )

    summaries = repositories.list_summaries(save_id)
    assert _error_text(model) == ""
    assert "Summary backfill finished: 4 messages compacted" in _status_text(model)
    assert len(summaries) == 1
    assert summaries[0].covers_message_start_id == "summary-backfill-message-1"
    assert summaries[0].covers_message_end_id == "summary-backfill-message-4"
    assert repositories.get_effective_setting(
        "recent_player_message_window",
        save_id=save_id,
    ) == 5
    assert repositories.get_effective_setting(
        "recent_narrator_message_window",
        save_id=save_id,
    ) == 5
    planner_history = narrator_planner_chat_history_window_settings(
        repositories,
        save_id=save_id,
    )
    assert planner_history.player_messages == 24
    assert planner_history.narrator_messages == 24
    assert len(provider.chat_requests) == 1


def test_run_context_cleanup_without_model_preference_returns_error(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories)
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=RuntimeStructuredCleanupProvider([{"actions": []}]),
    )
    controller.load_save(save_id)

    model = asyncio.run(controller.run_context_cleanup())

    assert _error_text(model) == "No context cleanup model preference configured"


def test_run_context_cleanup_without_structured_provider_returns_error(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories)
    _configure_context_cleanup_model(repositories)
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=RuntimeFakeProvider(),
    )
    controller.load_save(save_id)

    model = asyncio.run(controller.run_context_cleanup())

    assert _error_text(model) == (
        "Context Cleanup provider does not support structured output or tool calling"
    )


def test_run_context_cleanup_without_structured_model_capability_returns_error(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories)
    repositories.set_model_preference(
        task="context_cleanup",
        provider="fake",
        model_id="fake-structured",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-structured",
        display_name="Fake Structured",
        capabilities=[ProviderCapability.CHAT.value],
    )
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=RuntimeStructuredCleanupProvider([{"actions": []}]),
    )
    controller.load_save(save_id)

    model = asyncio.run(controller.run_context_cleanup())

    assert _error_text(model) == (
        "Context cleanup model does not advertise structured output or tool calling"
    )


def test_run_context_cleanup_with_unavailable_model_returns_error(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories)
    _configure_context_cleanup_model(repositories)
    repositories.mark_missing_provider_models_unavailable(
        provider="fake",
        available_model_ids=set(),
    )
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=RuntimeStructuredCleanupProvider([{"actions": []}]),
    )
    controller.load_save(save_id)

    model = asyncio.run(controller.run_context_cleanup())

    assert _error_text(model) == "Context cleanup model is unavailable: fake-structured"


def test_run_context_cleanup_rejects_missing_catalog_row_for_selected_model(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories)
    repositories.set_model_preference(
        task="context_cleanup",
        provider="fake",
        model_id="fake-unsynced-structured",
    )
    provider = RuntimeStructuredCleanupProvider([{"notes": []}, {"actions": []}])
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
    )
    controller.load_save(save_id)

    model = asyncio.run(controller.run_context_cleanup())

    assert _error_text(model) == (
        "Context cleanup model does not advertise structured output or tool calling"
    )
    assert provider.structured_requests == []


def test_run_context_cleanup_reports_success_and_precomputes_context(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories)
    _configure_context_cleanup_model(repositories)
    provider = RuntimeStructuredCleanupProvider([{"notes": []}, {"actions": []}])
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
    )
    precomputed_save_ids: list[str] = []

    async def precompute(save_id: str) -> None:
        precomputed_save_ids.append(save_id)

    controller.precompute_next_turn_context = precompute
    controller.load_save(save_id)

    model = asyncio.run(controller.run_context_cleanup())

    assert _error_text(model) == ""
    assert "Context cleanup finished" in _status_text(model)
    assert precomputed_save_ids == [save_id]
    assert [request.schema_name for request in provider.structured_requests] == [
        "context_cleanup_scan",
        "context_cleanup_actions",
    ]


def test_run_context_cleanup_uses_phase_specific_model_preferences(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories)
    for task, model_id in (
        ("context_cleanup_scan", "fake-scan"),
        ("context_cleanup_actions", "fake-actions"),
    ):
        repositories.set_model_preference(
            task=task,
            provider="fake",
            model_id=model_id,
        )
        repositories.save_provider_model(
            provider="fake",
            model_id=model_id,
            display_name=model_id,
            capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
        )
    provider = RuntimeStructuredCleanupProvider([{"notes": []}, {"actions": []}])
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
    )
    controller.load_save(save_id)

    model = asyncio.run(controller.run_context_cleanup())

    assert _error_text(model) == ""
    assert [request.model_id for request in provider.structured_requests] == [
        "fake-scan",
        "fake-actions",
    ]


def test_run_guided_context_cleanup_queues_review_suggestions(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, narrator_id = _persist_runtime_save(repositories)
    state = repositories.upsert_world_state(
        save_id=save_id,
        key="scene.alert",
        value={"status": "wrong"},
        category="scene",
        source_message_id=narrator_id,
        state_id="state-alert",
    )
    _configure_context_cleanup_model(repositories)
    instruction = "Fix the stale scene alert."
    provider = RuntimeStructuredCleanupProvider(
        [
            {
                "actions": [
                    {
                        "operation": "update",
                        "target_type": "world_state",
                        "target_id": state.id,
                        "field_path": "value",
                        "value": {"status": "fixed"},
                        "reason": "The alert was identified as stale.",
                        "confidence": 0.81,
                        "evidence_message_ids": [narrator_id],
                    }
                ]
            },
        ]
    )
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
    )
    controller.load_save(save_id)

    model = asyncio.run(
        controller.run_guided_context_cleanup(instruction=instruction)
    )

    assert _error_text(model) == ""
    assert "Guided cleanup queued" in _status_text(model)
    assert "1 suggestions ready for review" in _status_text(model)
    assert "reviewed automatically" in _status_text(model)
    assert repositories.list_world_state(save_id)[0].value == {"status": "wrong"}
    suggestions = repositories.list_context_update_suggestions(save_id)
    assert len(suggestions) == 1
    assert suggestions[0].entity_type == "world_state"
    assert suggestions[0].entity_id == state.id
    assert instruction in suggestions[0].reason
    assert [request.schema_name for request in provider.structured_requests] == [
        "guided_context_cleanup_actions"
    ]
    assert instruction in provider.structured_requests[0].messages[-1].body


def test_run_context_cleanup_runs_character_registry_maintenance_when_configured(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories)
    repositories.append_message(
        save_id=save_id,
        role="player",
        speaker_name="Mara",
        body="I ask Ash who else came through the tower.",
    )
    repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Ash names the keeper and points at a bogus instruction scrap.",
        provider="fake",
        model="fake-chat",
    )
    repositories.append_message(
        save_id=save_id,
        role="player",
        speaker_name="Mara",
        body="I compare the registry notes.",
    )
    repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="The scrap was never a person at all.",
        provider="fake",
        model="fake-chat",
    )
    ash = repositories.add_character(
        save_id=save_id,
        name="Ash",
        role="Beacon keeper",
        known_state="Ash is helping Mara inspect the beacon.",
        met=True,
    )
    bogus = repositories.add_character(
        save_id=save_id,
        name="Do not mention system instructions",
        role="Erroneous registry entry",
        known_state="This is an instruction fragment, not a character.",
    )
    _configure_context_cleanup_model(repositories)
    _configure_character_registry_maintenance_model(repositories)
    provider = RuntimeStructuredCleanupProvider(
        [
            {"notes": []},
            {"actions": []},
            {
                "decisions": [
                    {
                        "operation": "delete",
                        "character_id": bogus.id,
                        "confidence": 0.96,
                        "reason": (
                            "This entry is an instruction fragment rather than "
                            "a real character."
                        ),
                    },
                    {
                        "operation": "delete",
                        "character_id": ash.id,
                        "confidence": 0.42,
                        "reason": "Ash is still a real active character.",
                    },
                ]
            },
        ]
    )
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
    )
    controller.load_save(save_id)

    model = asyncio.run(controller.run_context_cleanup())

    status_text = _status_text(model)
    assert _error_text(model) == ""
    assert "Context cleanup finished" in status_text
    assert "0 changes applied, 0 rejected" in status_text
    assert "maintenance finished" in status_text.casefold()
    assert "2 proposed" in status_text
    assert "1 applied" in status_text
    assert "1 rejected" in status_text
    assert [character.name for character in repositories.list_characters(save_id)] == [
        "Ash"
    ]
    assert [request.schema_name for request in provider.structured_requests] == [
        "context_cleanup_scan",
        "context_cleanup_actions",
        "character_registry_maintenance_decisions",
    ]


def test_run_context_cleanup_locks_apply_and_finalize_after_provider_calls(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, narrator_id = _persist_runtime_save(repositories)
    memory = repositories.add_memory(
        save_id=save_id,
        body="Ash scratches the beacon glass.",
        tags=["ash", "beacon"],
        source_message_id=narrator_id,
    )
    _configure_context_cleanup_model(repositories)
    lock_depth = 0
    events: list[str] = []

    class LockAwareStructuredCleanupProvider(RuntimeStructuredCleanupProvider):
        async def generate_structured_output(
            self,
            request: StructuredOutputRequest,
        ) -> StructuredOutputResponse:
            assert lock_depth == 0
            events.append(f"provider:{request.schema_name}")
            return await super().generate_structured_output(request)

    provider = LockAwareStructuredCleanupProvider(
        [
            {"notes": []},
            {
                "actions": [
                    {
                        "operation": "archive",
                        "target_type": "memory",
                        "target_id": memory.id,
                        "field_path": "*",
                        "value": None,
                        "reason": "Memory is redundant with the source message.",
                        "confidence": 0.92,
                        "evidence_message_ids": [narrator_id],
                    }
                ]
            },
        ]
    )
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
    )
    precomputed_save_ids: list[str] = []

    original_begin_transaction = repositories.begin_transaction
    original_update_job = repositories.update_job

    class InstrumentedSaveOperationLock:
        async def __aenter__(self) -> None:
            nonlocal lock_depth
            assert lock_depth == 0
            events.append("lock:enter")
            lock_depth += 1

        async def __aexit__(
            self,
            _exc_type: object,
            _exc: object,
            _traceback: object,
        ) -> None:
            nonlocal lock_depth
            assert lock_depth == 1
            events.append("lock:exit")
            lock_depth -= 1
            return None

    def save_operation_lock(requested_save_id: str) -> InstrumentedSaveOperationLock:
        assert requested_save_id == save_id
        events.append("lock:requested")
        return InstrumentedSaveOperationLock()

    def begin_transaction() -> None:
        assert lock_depth == 1
        events.append("apply:begin_transaction")
        original_begin_transaction()

    def update_job(
        job_id: str,
        *,
        status: str,
        result: dict[str, object] | None = None,
        error: str | None = None,
    ) -> object:
        if status == "succeeded":
            assert lock_depth == 1
            events.append("job:succeeded")
        return original_update_job(
            job_id,
            status=status,
            result=result,
            error=error,
        )

    async def precompute(save_id: str) -> None:
        assert lock_depth == 0
        precomputed_save_ids.append(save_id)

    monkeypatch.setattr(repositories, "begin_transaction", begin_transaction)
    monkeypatch.setattr(repositories, "update_job", update_job)
    monkeypatch.setattr(controller, "_save_operation_lock", save_operation_lock)
    controller.precompute_next_turn_context = precompute
    controller.load_save(save_id)

    model = asyncio.run(controller.run_context_cleanup())

    assert _error_text(model) == ""
    assert "Context cleanup finished" in _status_text(model)
    assert precomputed_save_ids == [save_id]
    assert repositories.list_memories(save_id) == []
    assert [request.schema_name for request in provider.structured_requests] == [
        "context_cleanup_scan",
        "context_cleanup_actions",
    ]
    assert events == [
        "provider:context_cleanup_scan",
        "provider:context_cleanup_actions",
        "lock:requested",
        "lock:enter",
        "apply:begin_transaction",
        "job:succeeded",
        "lock:exit",
    ]


@pytest.mark.parametrize(
    ("missing_task", "expected_status_text"),
    [
        ("chat", "chat"),
        ("context_search", "context"),
    ],
)
def test_initial_render_submit_missing_preferences_returns_submitted_save_model(
    repositories: PersistenceRepositories,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    missing_task: str,
    expected_status_text: str,
) -> None:
    runtime = _import_runtime_without_gtk(monkeypatch)
    save_id, _ = _persist_runtime_save(repositories, include_messages=False)
    if missing_task != "chat":
        repositories.set_model_preference(
            task="chat",
            provider="fake",
            model_id="fake-chat",
        )
    if missing_task != "context_search":
        repositories.set_model_preference(
            task="context_search",
            provider="fake",
            model_id="fake-chat",
        )
    provider = RuntimeFakeProvider()
    controller = _runtime_controller(
        runtime,
        repositories,
        tmp_path,
        provider=provider,
    )
    controller.load_save(save_id)

    turn = asyncio.run(
        controller.submit_player_message_for_initial_render(
            body="I should not be persisted on missing preferences.",
            speaker_name="Mara",
        )
    )

    assert _value(turn, "save_id") == save_id
    model = _value(turn, "model")
    assert _value(model, "active_save_id") == save_id
    assert repositories.list_messages(save_id) == []
    assert provider.chat_requests == []
    assert expected_status_text in _error_text(model).casefold()


def _import_runtime_without_gtk(monkeypatch: MonkeyPatch) -> Any:
    original_import = builtins.__import__

    def import_without_gtk(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "gi" or name.startswith("gi."):
            raise AssertionError(
                "bragi.application.controller must not import GTK/PyGObject"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_gtk)
    sys.modules.pop("bragi.application.controller", None)
    return importlib.import_module("bragi.application.controller")


def _runtime_controller(
    runtime: Any,
    repositories: PersistenceRepositories,
    tmp_path: Path,
    *,
    provider: RuntimeFakeProvider | None = None,
    providers: dict[str, RuntimeFakeProvider] | None = None,
    context_search_service: object | None = None,
) -> Any:
    runtime_class = (
        runtime.RuntimeController
        if hasattr(runtime, "RuntimeController")
        else runtime.BragiRuntime
    )
    return runtime_class(
        repositories=repositories,
        providers=providers or {"fake": provider or RuntimeFakeProvider()},
        media_dir=tmp_path / "media",
        context_search_service=context_search_service,
    )


def _save_fake_provider_model(
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


def _create_save_with_current_state(
    repositories: PersistenceRepositories,
) -> SaveRecord:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="First Harbor",
        premise="A drowned harbor rings at low tide.",
        player_role="Harbor warden",
        content={"player_character_name": "Mara Voss"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="First Harbor")
    repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="Transcript-only old chat.",
    )
    narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Transcript-only old chat.",
    )
    location = repositories.add_location(
        save_id=save.id,
        name="Bell Court",
        description="A stone court below the drowned quay.",
    )
    character = repositories.add_character(
        save_id=save.id,
        name="Mara Voss",
        role="Harbor warden",
        known_state="Revealed the bell bargain and survived.",
        met=True,
        personality="Stubborn, tender under pressure.",
        voice="clipped, dry, careful with promises",
        relationships={"Ren": "owes him the bell-key"},
        location_id=location.id,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
        situation="Mara faces the tide court after the reveal.",
        objective="Negotiate chapter two without freeing the bell.",
        present_character_ids=[character.id],
        source_message_id=narrator.id,
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
    return save


def _reviewed_full_roleplay_sections() -> dict[str, str]:
    return {
        "title": "Reviewed Glass Harbor",
        "premise": "A revised drowned harbor mystery.",
        "player_character_name": "Mara Voss",
        "player_role": "Reef investigator",
        "tone_genre": "Reviewed nautical noir.",
        "opening_message": "Reviewed opening bell.",
    }


def _reviewed_first_contact_exploration_sections() -> dict[str, str]:
    return {
        "title": "Reviewed Songs Under Europa",
        "premise": "A survey crew finds patterned signals beneath Europa's ice.",
        "player_character_name": "Dr. Mara Voss",
        "player_role": "Mission linguist and acting contact lead",
        "mission_profile": "Survey the hidden ocean and keep the crew alive.",
        "ship_or_base_status": "Habitat Kestrel has heat for forty-two hours.",
        "exploration_target": "A black-water cavern under the ice shelf.",
        "unknown_intelligence": "An unseen singer answers sonar with pressure songs.",
        "knowledge_state": "Observed songs; unknown intent and unknown anatomy.",
        "translation_progress": (
            "Three descending pulses may mean open water; blue flashes mean attention."
        ),
        "discoveries_and_samples": (
            "Metallic spores remain quarantined in sample bay two."
        ),
        "hazards_and_escalation": (
            "Thermal fissures spread while the rescue window narrows."
        ),
        "tone_genre": "Hopeful first-contact science fiction with practical risk.",
        "opening_message": "Blue light pulses beneath the ice.",
    }


def _reviewed_survival_expedition_sections() -> dict[str, str]:
    return {
        "title": "Red Dune Crossing",
        "premise": "A desert convoy must reach the relay tower before the wells fail.",
        "player_character_name": "Ira Sen",
        "player_role": "Caravan navigator",
        "expedition_goal": "Cross 120 miles of salt flats to repair the relay.",
        "route_options": "North wells, glass canyon, or direct salt road.",
        "resource_inventory": "Water: 18 canteens. Fuel: 4 drums. Medicine: 2 kits.",
        "environmental_conditions": "Heat haze, dust storms, and cold nights.",
        "hazards_and_events": "Sink crust, engine overheating, and raider signals.",
        "camp_status": "Canvas shade camp with one damaged condenser.",
        "travel_progress": "12 miles complete; 108 remain.",
        "tone_genre": "Desperate desert survival with practical choices.",
        "opening_message": "The salt wind scrapes across the lead truck.",
    }


def _reviewed_time_loop_sections() -> dict[str, str]:
    return {
        "title": "Reviewed Bellwether Day",
        "premise": "A revised harbor festival repeats until the bell is saved.",
        "player_character_name": "Mara Voss",
        "player_role": "Archivist who remembers the repeats.",
        "loop_premise": (
            "The same festival day resets after the harbor bell sinks beneath the tide."
        ),
        "reset_trigger": "Reset occurs when the drowned bell tolls at midnight.",
        "loop_duration": "Twenty-four hours, dawn festival bell to dawn festival bell.",
        "starting_state": "Mara wakes in the archive loft with a wet matchbook.",
        "objective": "Prevent the bell from sinking and expose the saboteur.",
        "failure_conditions": "The bell sinks, Mara dies, or the day reaches midnight.",
        "baseline_world_state": (
            "At dawn the harbor is intact, Mira is skeptical, and the tower is locked."
        ),
        "loop_schedule": "09:00 parade; 18:00 storm tide; 23:45 sabotage window.",
        "persistent_knowledge": (
            "Player/meta knowledge persists: tower code, Mira's warning, and "
            "tunnel route."
        ),
        "persistence_exceptions": "Salt mark and wet matchbook persist across resets.",
        "npc_memory_rules": (
            "NPCs reset to dawn memories unless a persistence exception says "
            "otherwise."
        ),
        "current_loop_state": "Loop 1, dawn phase, no deviations confirmed.",
        "tone_genre": "Clockwork mystery with coastal urgency.",
        "opening_message": "The same bell rings dawn again.",
    }


def _reviewed_investigation_mystery_sections() -> dict[str, str]:
    return {
        "title": "Reviewed Broken Hours",
        "premise": "A public disappearance exposes a sealed museum conspiracy.",
        "player_character_name": "Inspector Mara Voss",
        "player_role": "The investigator assigned to reopen the impossible case.",
        "case_facts": (
            "Curator Elian Vale vanished from the sealed east gallery during a gala."
        ),
        "clues": (
            "Broken display dust found outside the gallery door; undiscovered. "
            "Watch log gap from 9:10 to 9:18; reliable and tied to Sera's alibi."
        ),
        "timeline": (
            "Public: gala toast at 9:00, alarm at 9:21. "
            "Hidden: lift moved at 9:12 and ledger was swapped at 9:15."
        ),
        "red_herrings": (
            "A bloody glove belongs to an old mannequin repair, not the culprit."
        ),
        "hidden_truth": (
            "Sera staged the vanishing to hide a smuggling ledger in the restoration "
            "lift."
        ),
        "case_status": "Unresolved; the player has only public facts.",
        "tone_genre": "Quiet investigative noir with careful clue continuity.",
        "opening_message": "Rain taps the museum glass as the east gallery unlocks.",
    }


def _reviewed_political_intrigue_sections() -> dict[str, str]:
    return {
        "title": "Council of Ash",
        "premise": "A city council vote will decide who controls the harbor.",
        "player_character_name": "Mara Voss",
        "player_role": "Envoy holding the swing vote",
        "political_arena": "The harbor council chamber and public galleries.",
        "political_factions": "Guilds, Old Families, and dock unions.",
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
        "tone_genre": "Tense council intrigue.",
        "opening_message": "The council bell rings.",
    }


def _reviewed_settlement_builder_sections() -> dict[str, str]:
    return {
        "title": "Hearthstone Landing",
        "premise": "A flood-struck river town must survive its first hard year.",
        "player_character_name": "Mara Vale",
        "player_role": "Elected settlement steward",
        "settlement_profile": (
            "Hearthstone Landing is a timber-and-stone river town founded after "
            "the old bridge collapsed."
        ),
        "resources_and_indicators": (
            "Food low, lumber useful, morale fragile, defenses unfinished."
        ),
        "projects_and_facilities": (
            "Repair the palisade, build a flood gate, and reopen the mill race."
        ),
        "threats_and_opportunities": (
            "Spring floods, hungry bandits, rival ferry tolls, and a grain compact."
        ),
        "calendar_and_deadlines": "Flood season begins in sixteen days.",
        "tone_genre": "Grounded community survival with political pressure.",
        "opening_message": "The river has risen another handspan overnight.",
    }


def _reviewed_monster_hunt_bounty_sections() -> dict[str, str]:
    return {
        "title": "The Thornback Contract",
        "premise": "A bounty crew hunts a beast that learns from every failed trap.",
        "player_character_name": "Ira Voss",
        "player_role": "Licensed monster tracker",
        "hunt_profile": "Find the Thornback before the harvest road closes.",
        "target_profile": (
            "The Thornback is armored, avoids firelight, and guards the old orchard."
        ),
        "leads_and_clues": (
            "Three-toed tracks at Mill Creek and blue sap on broken arrows."
        ),
        "hunt_locations": "Mill Creek, old orchard, and collapsed toll road.",
        "preparation_state": "Silver wire, oil snares, two hounds, and one debt.",
        "hunt_status": "Unresolved; target wounded but adapting.",
        "tone_genre": "Tense investigative wilderness hunt.",
        "opening_message": "The newest tracks circle your camp twice.",
    }


def _reviewed_road_trip_pilgrimage_sections() -> dict[str, str]:
    return {
        "title": "Road to Saint Orra",
        "premise": "A divided traveling party must reach the shrine before midsummer.",
        "player_character_name": "Nell Aran",
        "player_role": "Pilgrim guide and reluctant mediator",
        "journey_profile": (
            "Carry a cracked bell relic to Saint Orra's shrine before midsummer."
        ),
        "route_and_stops": (
            "Salt road to Lantern Ford, then Crow Market, then the hill shrine."
        ),
        "transport_and_supplies": "One wagon, two mules, six days of oats.",
        "recurring_pressures": "Border patrols, summer storms, and a silent pursuer.",
        "relationship_threads": "Tom doubts Sera; the cousins blame each other.",
        "journey_progress": "Current leg: day one to Lantern Ford.",
        "tone_genre": "Warm, weary travel drama with spiritual tension.",
        "opening_message": "The shrine road starts where the city stones end.",
    }


def _reviewed_merchant_trade_route_sections() -> dict[str, str]:
    return {
        "title": "Ledger Road",
        "premise": "A caravan must turn debt into profit across dangerous markets.",
        "player_character_name": "Mara Den",
        "player_role": "Caravan factor with the final signature",
        "trade_profile": "Run cedar oil and glassware from Kesh Gate to Red Harbor.",
        "cargo_inventory": "Cedar oil: 20 jars. Glassware: 8 crates. Spare axle: 1.",
        "markets_and_stops": (
            "Kesh Gate overpays for medicine; Red Harbor needs oil; Dustwell "
            "has cheap fodder."
        ),
        "contracts_and_debts": (
            "Deliver ten jars to Red Harbor in twelve days or double the debt."
        ),
        "route_hazards": "Tariff patrols, bridge bandits, storms, and rivals.",
        "profit_and_loss": "Current margin is thin; one lost crate erases profit.",
        "tone_genre": "Economy-lite caravan drama with hard bargains.",
        "opening_message": "The creditor stamps the contract before the ink dries.",
    }


def _reviewed_cyoa_sections() -> dict[str, str]:
    return {
        "title": "Library of Falling Doors",
        "premise": "Every shelf is a door.",
        "player_character_name": "Mira Vale",
        "player_role": "Courier",
        "tone_genre": "Surreal archive adventure.",
        "choice_style": "Four concrete choices with different risks.",
        "opening_message": "The blue shelf opens.",
    }


def _reviewed_dating_sim_sections() -> dict[str, str]:
    return {
        "title": "Reviewed Saltwind Hearts",
        "premise": "Festival week begins at a seaside academy.",
        "player_character_name": "Ren Takahashi",
        "player_character_profile": (
            "Ren is a thoughtful male transfer student trying to decide which "
            "future, club, and relationship will define his last summer."
        ),
        "player_role": "The central romantic lead.",
        "tone_genre": "Warm romantic drama with comedy and school-life stakes.",
        "opening_message": "The station doors open onto salt air and festival banners.",
    }


def _character_record_text(character: object) -> str:
    fields = [
        "role",
        "known_state",
        "appearance",
        "visual_notes",
        "personality",
        "voice",
        "status",
        "private_notes",
    ]
    return "\n".join(
        value
        for field in fields
        if isinstance(value := getattr(character, field, ""), str)
    )


async def _save_scenario_draft(
    controller: Any,
    *,
    scenario_type: str,
    scenario_types: tuple[str, ...] | None = None,
    sections: dict[str, str],
    save_title: str,
    request_initial_image: bool,
    action_choices_enabled: bool = False,
    source_metadata: dict[str, object] | None = None,
    character_starters: list[dict[str, object]] | None = None,
) -> object:
    model = await controller.save_scenario_draft(
        scenario_type=scenario_type,
        scenario_types=scenario_types,
        sections=sections,
        character_starters=character_starters,
        action_choices_enabled=action_choices_enabled,
        save_title=save_title,
        source_metadata=source_metadata,
    )
    if not request_initial_image or _error_text(model):
        return model

    save_id = _value(model, "active_save_id")
    opening_message_id = next(
        (
            _value(message, "message_id")
            for message in _chronicle_messages(model)
            if _value(message, "role") == "narrator"
        ),
        None,
    )
    if save_id is None or opening_message_id is None:
        return model

    return await controller.generate_initial_scenario_image(
        source_message_id=opening_message_id,
        active_save_id=save_id,
    )


def _create_manual_full_roleplay(
    runtime: Any,
    controller: Any,
    fields: dict[str, str],
) -> object:
    if hasattr(controller, "create_manual_full_roleplay_save"):
        return controller.create_manual_full_roleplay_save(fields=fields)

    input_type = getattr(runtime, "ManualFullRoleplayForm", None)
    if input_type is not None:
        return controller.create_manual_scenario(input_type(**fields))

    manual_input_type = runtime.ManualScenarioInput
    try:
        scenario_input = manual_input_type(
            scenario_type="full_roleplay",
            **fields,
        )
    except TypeError:
        scenario_input = manual_input_type(
            scenario_type="full_roleplay",
            title=fields["title"],
            premise=fields["premise"],
            player_role=fields["player_role"],
            opening_message=fields["opening_message"],
            save_title=fields["save_title"],
        )
    return controller.create_manual_scenario(scenario_input)


async def _generate_image_for_message(
    controller: Any,
    source_message_id: str,
) -> object:
    if hasattr(controller, "generate_image_for_message"):
        return await controller.generate_image_for_message(source_message_id)
    return await controller.generate_image(source_message_id=source_message_id)


def _open_world_data_editor(controller: Any) -> object:
    if hasattr(controller, "open_world_data_editor"):
        return controller.open_world_data_editor()
    if hasattr(controller, "open_world_data"):
        return controller.open_world_data()
    raise AssertionError("Runtime controller does not expose open_world_data_editor()")


def _open_character_registry(controller: Any) -> object:
    if hasattr(controller, "open_character_registry"):
        return controller.open_character_registry()
    raise AssertionError("Runtime controller does not expose open_character_registry()")


def _persist_runtime_save(
    repositories: PersistenceRepositories,
    *,
    title: str = "Night Watch",
    player_role: str = "Signal warden",
    player_character_name: str = "Mara Voss",
    player_body: str = "I climb toward the beacon lens.",
    narrator_body: str = "Ash scratches the glass as the stair shakes.",
    include_messages: bool = True,
) -> tuple[str, str]:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role=player_role,
        content={
            "player_character_name": player_character_name,
            "opening_message": "The beacon gutters in the tower.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title=title)
    if not include_messages:
        return save.id, ""

    repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body=player_body,
    )
    narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body=narrator_body,
        provider="fake",
        model="fake-chat",
        token_estimate=21,
    )
    return save.id, narrator.id


def _persist_revision_save(
    repositories: PersistenceRepositories,
    *,
    player_role: str = "Signal warden",
    player_character_name: str = "Mara Voss",
    player_2_speaker_name: str | None = "Mara",
) -> tuple[str, dict[str, str]]:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role=player_role,
        content={
            "player_character_name": player_character_name,
            "opening_message": "The beacon gutters in the tower.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Revision Watch")
    messages = {
        "player_1": repositories.append_message(
            save_id=save.id,
            role="player",
            speaker_name="Mara",
            body="I light the beacon.",
        ),
        "narrator_1": repositories.append_message(
            save_id=save.id,
            role="narrator",
            speaker_name="Narrator",
            body="The beacon wakes.",
            provider="fake",
            model="fake-chat",
        ),
        "player_2": repositories.append_message(
            save_id=save.id,
            role="player",
            speaker_name=player_2_speaker_name,
            body="I open the sealed door.",
        ),
        "narrator_2": repositories.append_message(
            save_id=save.id,
            role="narrator",
            speaker_name="Narrator",
            body="The corridor floods with ash.",
            provider="fake",
            model="fake-chat",
        ),
        "player_3": repositories.append_message(
            save_id=save.id,
            role="player",
            speaker_name="Mara",
            body="I step through.",
        ),
        "narrator_3": repositories.append_message(
            save_id=save.id,
            role="narrator",
            speaker_name="Narrator",
            body="A shadow follows.",
            provider="fake",
            model="fake-chat",
        ),
    }
    return save.id, {name: message.id for name, message in messages.items()}


def _persist_deleted_turn_context(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    ids: dict[str, str],
) -> None:
    repositories.upsert_world_state(
        save_id=save_id,
        key="scene.location",
        value={"name": "Ash corridor"},
        category="scene",
        source_message_id=ids["narrator_2"],
    )
    repositories.add_state_change(
        save_id=save_id,
        operation="upsert",
        state_key="scene.location",
        before_json=json.dumps({"name": "Beacon tower"}),
        after_json=json.dumps({"name": "Ash corridor"}),
        source_message_id=ids["narrator_2"],
    )
    repositories.upsert_world_state(
        save_id=save_id,
        key="threat.shadow",
        value={"status": "following"},
        category="threat",
        source_message_id=ids["narrator_3"],
    )
    repositories.add_state_change(
        save_id=save_id,
        operation="upsert",
        state_key="threat.shadow",
        before_json=None,
        after_json=json.dumps({"status": "following"}),
        source_message_id=ids["narrator_3"],
    )
    repositories.add_memory(
        save_id=save_id,
        body="Mara opened the sealed ash corridor.",
        tags=["ash", "door"],
        source_message_id=ids["narrator_2"],
    )
    repositories.add_summary(
        save_id=save_id,
        covers_message_start_id=ids["player_2"],
        covers_message_end_id=ids["narrator_3"],
        body="Mara opened the door and a shadow followed.",
        provider="fake",
        model="fake-chat",
    )


def _configure_chat_and_context_preferences(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-chat",
    )


def _configure_narrator_edit_reconciliation_model(
    repositories: PersistenceRepositories,
) -> None:
    for task in ("state_memory", "context_update"):
        repositories.set_model_preference(
            task=task,
            provider="fake",
            model_id="fake-structured",
        )
    _save_fake_provider_model(
        repositories,
        model_id="fake-structured",
        capabilities=["structured_output", "tool_calling"],
    )


def _configure_context_cleanup_model(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_model_preference(
        task="context_cleanup",
        provider="fake",
        model_id="fake-structured",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-structured",
        display_name="Fake Structured",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )


def _configure_scenario_outcome_model(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_model_preference(
        task="scenario_outcome",
        provider="fake",
        model_id="fake-structured",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-structured",
        display_name="Fake Structured",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )


def _configure_character_registry_maintenance_model(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_model_preference(
        task="character_registry_maintenance",
        provider="fake",
        model_id="fake-structured",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-structured",
        display_name="Fake Structured",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )


def _runtime_chat_completion_job_count(
    repositories: PersistenceRepositories,
    save_id: str,
) -> int:
    row = repositories.connection.execute(
        """
        SELECT COUNT(*)
        FROM jobs
        WHERE save_id = ? AND type = 'chat_completion'
        """,
        (save_id,),
    ).fetchone()
    return int(row[0])


def _assert_messages_deleted_or_archived(
    repositories: PersistenceRepositories,
    message_ids: Iterable[str],
) -> None:
    ids = tuple(message_ids)
    rows = repositories.connection.execute(
        """
        SELECT id, deleted_at
        FROM messages
        WHERE id IN ({})
        """.format(", ".join("?" for _ in ids)),
        ids,
    ).fetchall()
    by_id = {row["id"]: row["deleted_at"] for row in rows}
    for message_id in ids:
        assert message_id not in by_id or by_id[message_id] is not None


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


def _requested_scenario_section(prompt: str) -> str:
    prefix = "Requested field: "
    for line in prompt.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).replace(" ", "_")
    raise AssertionError(f"Scenario section prompt did not request a field: {prompt}")


def _flows_by_identifier(flows: Iterable[object]) -> dict[str, object]:
    return {_value(flow, "flow_id", "identifier", "id"): flow for flow in flows}


def _section_ids(flow: object) -> list[str]:
    sections = _value(
        flow,
        "editable_section_ids",
        "editable_sections",
        "sections",
    )
    return [
        section
        if isinstance(section, str)
        else _value(section, "section_id", "identifier", "id")
        for section in sections
    ]


def _chronicle_messages(model: object) -> tuple[object, ...]:
    direct = _value(model, "chronicle_messages", "messages", default=None)
    if direct is not None:
        return tuple(direct)
    return tuple(_value(_value(model, "chronicle"), "messages"))


def _history_messages(model: object) -> tuple[object, ...]:
    return tuple(_value(model, "messages", "items"))


def _actions_by_id(message: object) -> dict[str, object]:
    actions = _value(message, "actions")
    return {_value(action, "action_id", "id"): action for action in actions}


def _latest_image(model: object) -> object:
    direct = _value(model, "latest_image", "latest_scene_image", default=None)
    if direct is not None:
        return direct
    media = _value(model, "media", default=None)
    if media is None:
        raise AssertionError(f"{model!r} does not expose media")
    return _value(media, "latest_image", "latest_scene_image")


def _status_text(model: object) -> str:
    error = _error_text(model)
    if error:
        return error
    status = _value(model, "status_text", "status", default="")
    return "" if status is None else str(status)


def _error_text(model: object) -> str:
    error = _value(model, "error", default="")
    return "" if error is None else str(error)
