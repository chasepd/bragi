from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
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
    ProviderToolCall,
    StructuredOutputRequest,
    StructuredOutputResponse,
    ToolCallRequest,
    ToolCallResponse,
)
from bragi.services.character_registry_maintenance_service import (
    CharacterRegistryMaintenanceService,
)


class RecordingStructuredMaintenanceProvider:
    def __init__(
        self,
        provider_name: str = "fake",
        *,
        decisions: list[dict[str, object]] | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.decisions = decisions or []
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
                model_id="fake-character-maintenance",
                display_name="Fake Character Maintenance",
                capabilities=frozenset({ProviderCapability.STRUCTURED_OUTPUT}),
                context_window=8192,
            )
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        raise AssertionError(
            "character registry maintenance must use structured output, not chat text"
        )

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        raise AssertionError("character registry maintenance must not generate images")

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_output_requests.append(request)
        return StructuredOutputResponse(
            data={"decisions": self.decisions},
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 9},
        )


class RecordingToolMaintenanceProvider:
    def __init__(
        self,
        provider_name: str = "fake",
        *,
        responses: list[tuple[ProviderToolCall, ...]],
    ) -> None:
        self.provider_name = provider_name
        self.responses = responses
        self.structured_output_requests: list[StructuredOutputRequest] = []
        self.tool_call_requests: list[ToolCallRequest] = []

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
                model_id="fake-character-maintenance",
                display_name="Fake Character Maintenance",
                capabilities=frozenset({ProviderCapability.TOOL_CALLING}),
                context_window=8192,
            )
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        raise AssertionError(
            "character registry maintenance must use provider tools, not chat text"
        )

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        raise AssertionError("character registry maintenance must not generate images")

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_output_requests.append(request)
        raise AssertionError("tool-capable character maintenance should prefer tools")

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        self.tool_call_requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected tool-call request")
        return ToolCallResponse(
            tool_calls=self.responses.pop(0),
            body="",
            provider=request.provider,
            model_id=request.model_id,
        )


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_maintain_if_due_requests_structured_character_maintenance_schema(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_due_save_with_characters(repositories)
    provider = RecordingStructuredMaintenanceProvider()
    service = CharacterRegistryMaintenanceService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(service.maintain_if_due(save_id=save.id))

    assert result.proposed == ()
    assert result.applied == ()
    assert result.rejected == ()
    assert len(provider.structured_output_requests) == 1
    request = provider.structured_output_requests[0]
    assert request.provider == "fake"
    assert request.model_id == "fake-character-maintenance"
    assert "character" in request.schema_name.casefold()
    assert "maintenance" in request.schema_name.casefold()
    decisions_schema = request.schema["properties"]["decisions"]
    decision_item_schema = decisions_schema["items"]
    decision_properties = decision_item_schema["properties"]
    assert decision_properties["operation"]["enum"] == ["merge", "delete"]
    assert set(decision_properties["character_id"]["enum"]) == {
        "character-ilyra",
        "character-ashknife",
    }
    assert set(decision_properties["target_character_id"]["enum"]) - {None} == {
        "character-ilyra",
        "character-ashknife",
    }
    prompt_text = "\n".join(message.body.casefold() for message in request.messages)
    assert "captain ilyra" in prompt_text
    assert "ashknife" in prompt_text
    assert "json" not in prompt_text


def test_maintain_if_due_prefers_tool_calls_for_tool_capable_model(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    _configure_character_maintenance_model(
        repositories,
        capabilities=[ProviderCapability.TOOL_CALLING.value],
    )
    target = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        character_id="character-ilyra",
    )
    duplicate = repositories.add_character(
        save_id=save.id,
        name="The Red Captain",
        character_id="character-red-captain",
    )
    _append_completed_turns(repositories, save_id=save.id, count=3)
    provider = RecordingToolMaintenanceProvider(
        responses=[
            (
                ProviderToolCall(
                    id="call-merge",
                    name="merge_character",
                    arguments_json=json.dumps(
                        {
                            "character_id": duplicate.id,
                            "target_character_id": target.id,
                            "confidence": 0.99,
                            "reason": (
                                "The Red Captain is clearly the same person as "
                                "Captain Ilyra."
                            ),
                        }
                    ),
                ),
            )
        ]
    )

    result = asyncio.run(
        CharacterRegistryMaintenanceService(
            repositories=repositories,
            providers={"fake": provider},
        ).maintain_if_due(save_id=save.id)
    )

    assert provider.structured_output_requests == []
    assert [decision.operation for decision in result.applied] == ["merge"]
    assert len(provider.tool_call_requests) == 1
    assert [tool.name for tool in provider.tool_call_requests[0].tools] == [
        "merge_character",
        "delete_character_entry",
    ]


def test_maintain_if_due_tool_calls_retry_invalid_protected_source(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    _configure_character_maintenance_model(
        repositories,
        capabilities=[ProviderCapability.TOOL_CALLING.value],
    )
    protected = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        protected_from_maintenance=True,
        character_id="character-ilyra",
    )
    duplicate = repositories.add_character(
        save_id=save.id,
        name="The Red Captain",
        character_id="character-red-captain",
    )
    _append_completed_turns(repositories, save_id=save.id, count=3)
    provider = RecordingToolMaintenanceProvider(
        responses=[
            (
                ProviderToolCall(
                    id="call-bad",
                    name="delete_character_entry",
                    arguments_json=json.dumps(
                        {
                            "character_id": protected.id,
                            "confidence": 0.99,
                            "reason": "Bad protected-source proposal.",
                        }
                    ),
                ),
            ),
            (
                ProviderToolCall(
                    id="call-good",
                    name="merge_character",
                    arguments_json=json.dumps(
                        {
                            "character_id": duplicate.id,
                            "target_character_id": protected.id,
                            "confidence": 0.99,
                            "reason": (
                                "The unprotected duplicate clearly refers to Ilyra."
                            ),
                        }
                    ),
                ),
            ),
        ]
    )

    result = asyncio.run(
        CharacterRegistryMaintenanceService(
            repositories=repositories,
            providers={"fake": provider},
        ).maintain_if_due(save_id=save.id)
    )

    assert [decision.character_id for decision in result.applied] == [duplicate.id]
    assert len(provider.tool_call_requests) == 2
    assert protected.id in provider.tool_call_requests[1].messages[-1].body


def test_maintain_if_due_tool_calls_apply_valid_delete(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    _configure_character_maintenance_model(
        repositories,
        capabilities=[ProviderCapability.TOOL_CALLING.value],
    )
    valid = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        character_id="character-ilyra",
    )
    malformed = repositories.add_character(
        save_id=save.id,
        name="World State: Beacon Lens",
        role="not a character",
        known_state="Erroneous registry row imported from world-state text.",
        character_id="character-malformed-state-entry",
    )
    _append_completed_turns(repositories, save_id=save.id, count=3)
    provider = RecordingToolMaintenanceProvider(
        responses=[
            (
                ProviderToolCall(
                    id="call-delete",
                    name="delete_character_entry",
                    arguments_json=json.dumps(
                        {
                            "character_id": malformed.id,
                            "confidence": 0.95,
                            "reason": (
                                "This row is a malformed world-state entry, "
                                "not a person."
                            ),
                        }
                    ),
                ),
            )
        ]
    )

    result = asyncio.run(
        CharacterRegistryMaintenanceService(
            repositories=repositories,
            providers={"fake": provider},
        ).maintain_if_due(save_id=save.id)
    )

    assert provider.structured_output_requests == []
    assert [decision.operation for decision in result.applied] == ["delete"]
    assert [decision.character_id for decision in result.applied] == [malformed.id]
    assert repositories.get_character(valid.id) is not None
    assert repositories.get_character(malformed.id) is None
    assert _character_archived_at(repositories, save.id, malformed.id) is not None
    assert [tool.name for tool in provider.tool_call_requests[0].tools] == [
        "merge_character",
        "delete_character_entry",
    ]


def test_maintain_if_due_tool_calls_retry_invalid_merge_target(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    _configure_character_maintenance_model(
        repositories,
        capabilities=[ProviderCapability.TOOL_CALLING.value],
    )
    target = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        character_id="character-ilyra",
    )
    duplicate = repositories.add_character(
        save_id=save.id,
        name="The Red Captain",
        character_id="character-red-captain",
    )
    _append_completed_turns(repositories, save_id=save.id, count=3)
    provider = RecordingToolMaintenanceProvider(
        responses=[
            (
                ProviderToolCall(
                    id="call-bad-target",
                    name="merge_character",
                    arguments_json=json.dumps(
                        {
                            "character_id": duplicate.id,
                            "target_character_id": "character-missing",
                            "confidence": 0.99,
                            "reason": "Bad target.",
                        }
                    ),
                ),
            ),
            (
                ProviderToolCall(
                    id="call-good-target",
                    name="merge_character",
                    arguments_json=json.dumps(
                        {
                            "character_id": duplicate.id,
                            "target_character_id": target.id,
                            "confidence": 0.99,
                            "reason": (
                                "The Red Captain is clearly the same person as "
                                "Captain Ilyra."
                            ),
                        }
                    ),
                ),
            ),
        ]
    )

    result = asyncio.run(
        CharacterRegistryMaintenanceService(
            repositories=repositories,
            providers={"fake": provider},
        ).maintain_if_due(save_id=save.id)
    )

    assert [decision.character_id for decision in result.applied] == [duplicate.id]
    assert len(provider.tool_call_requests) == 2
    retry_body = provider.tool_call_requests[1].messages[-1].body
    assert "target_character_id must be one of" in retry_body
    assert repositories.get_character(duplicate.id) is None
    assert repositories.get_character(target.id) is not None


def test_maintain_if_due_excludes_protected_characters_from_decision_subjects(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    _configure_character_maintenance_model(repositories)
    protected = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        protected_from_maintenance=True,
        character_id="character-ilyra",
    )
    duplicate = repositories.add_character(
        save_id=save.id,
        name="The Red Captain",
        character_id="character-red-captain",
    )
    _append_completed_turns(repositories, save_id=save.id, count=3)
    provider = RecordingStructuredMaintenanceProvider(
        decisions=[
            {
                "operation": "delete",
                "character_id": protected.id,
                "confidence": 0.99,
                "reason": "Protected characters must not be deleted automatically.",
            },
            {
                "operation": "merge",
                "character_id": duplicate.id,
                "target_character_id": protected.id,
                "confidence": 0.99,
                "reason": "The unprotected duplicate clearly refers to Ilyra.",
            },
        ],
    )
    service = CharacterRegistryMaintenanceService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(service.maintain_if_due(save_id=save.id))

    assert [decision.character_id for decision in result.rejected] == [protected.id]
    assert [decision.character_id for decision in result.applied] == [duplicate.id]
    assert repositories.get_character(duplicate.id) is None
    updated_protected = repositories.get_character(protected.id)
    assert updated_protected is not None
    assert updated_protected.protected_from_maintenance is True
    assert "The Red Captain" in updated_protected.aliases
    request = provider.structured_output_requests[0]
    decision_properties = request.schema["properties"]["decisions"]["items"][
        "properties"
    ]
    assert decision_properties["character_id"]["enum"] == [duplicate.id]
    assert set(decision_properties["target_character_id"]["enum"]) - {None} == {
        protected.id,
        duplicate.id,
    }
    prompt_text = "\n".join(message.body.casefold() for message in request.messages)
    assert "protected=true" in prompt_text


def test_maintain_if_due_uses_messages_since_previous_run_with_overlap(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    _configure_character_maintenance_model(repositories)
    repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        character_id="character-ilyra",
    )
    repositories.add_character(
        save_id=save.id,
        name="Archivist Ren",
        character_id="character-ren",
    )
    _append_labelled_completed_turns(
        repositories,
        save_id=save.id,
        labels=("pre-maintenance-zero", "pre-maintenance-one", "overlap-anchor"),
    )
    provider = RecordingStructuredMaintenanceProvider()
    service = CharacterRegistryMaintenanceService(
        repositories=repositories,
        providers={"fake": provider},
    )

    first_result = asyncio.run(service.maintain_if_due(save_id=save.id))

    assert first_result.skipped_reason is None
    _append_labelled_completed_turns(
        repositories,
        save_id=save.id,
        labels=("new-delta-zero", "new-delta-one", "new-delta-two"),
    )

    second_result = asyncio.run(service.maintain_if_due(save_id=save.id))

    assert second_result.skipped_reason is None
    assert len(provider.structured_output_requests) == 2
    prompt_text = "\n".join(
        message.body for message in provider.structured_output_requests[1].messages
    )
    assert "pre-maintenance-zero" not in prompt_text
    assert "overlap-anchor" in prompt_text
    assert "new-delta-zero" in prompt_text
    assert "new-delta-two" in prompt_text


def test_maintain_if_due_skips_when_all_characters_are_protected(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    _configure_character_maintenance_model(repositories)
    repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        protected_from_maintenance=True,
        character_id="character-ilyra",
    )
    repositories.add_character(
        save_id=save.id,
        name="Archivist Ren",
        protected_from_maintenance=True,
        character_id="character-ren",
    )
    _append_completed_turns(repositories, save_id=save.id, count=3)
    provider = RecordingStructuredMaintenanceProvider()
    service = CharacterRegistryMaintenanceService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(service.maintain_if_due(save_id=save.id))

    assert result.skipped_reason == "No unprotected characters eligible for maintenance"
    assert provider.structured_output_requests == []


@pytest.mark.parametrize(
    ("character_count", "completed_turns", "orphan_narrators", "expected_reason"),
    (
        (0, 3, 0, "No active characters"),
        (2, 2, 0, "cadence not due"),
        (2, 2, 1, "cadence not due"),
    ),
)
def test_maintain_if_due_skips_until_active_characters_and_three_completed_turns(
    repositories: PersistenceRepositories,
    character_count: int,
    completed_turns: int,
    orphan_narrators: int,
    expected_reason: str,
) -> None:
    save = _create_save(repositories)
    _configure_character_maintenance_model(repositories)
    for index in range(character_count):
        repositories.add_character(
            save_id=save.id,
            name=f"Character {index}",
            character_id=f"character-{index}",
        )
    _append_completed_turns(repositories, save_id=save.id, count=completed_turns)
    for index in range(orphan_narrators):
        repositories.append_message(
            save_id=save.id,
            role="narrator",
            speaker_name="Narrator",
            body=f"Orphan narrator beat {index}.",
            provider="fake",
            model="fake-chat",
        )
    provider = RecordingStructuredMaintenanceProvider()
    service = CharacterRegistryMaintenanceService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(service.maintain_if_due(save_id=save.id))

    assert result.skipped_reason is not None
    assert expected_reason in result.skipped_reason
    assert provider.structured_output_requests == []


def test_maintain_if_due_rejects_unavailable_selected_model(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_due_save_with_characters(repositories)
    repositories.mark_missing_provider_models_unavailable(
        provider="fake",
        available_model_ids=set(),
    )
    provider = RecordingStructuredMaintenanceProvider()
    service = CharacterRegistryMaintenanceService(
        repositories=repositories,
        providers={"fake": provider},
    )

    with pytest.raises(ValueError, match="model is unavailable"):
        asyncio.run(service.maintain_if_due(save_id=save.id))

    assert provider.structured_output_requests == []


def test_maintain_if_due_rejects_missing_catalog_row_for_selected_model(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    repositories.set_model_preference(
        task="character_registry_maintenance",
        provider="fake",
        model_id="fake-character-maintenance",
    )
    repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        character_id="character-ilyra",
    )
    repositories.add_character(
        save_id=save.id,
        name="Ashknife",
        character_id="character-ashknife",
    )
    _append_completed_turns(repositories, save_id=save.id, count=3)
    provider = RecordingStructuredMaintenanceProvider()
    service = CharacterRegistryMaintenanceService(
        repositories=repositories,
        providers={"fake": provider},
    )

    with pytest.raises(
        ValueError,
        match="does not advertise structured output or tool calling",
    ):
        asyncio.run(service.maintain_if_due(save_id=save.id))

    assert provider.structured_output_requests == []


def test_maintain_if_due_rejects_delete_that_would_leave_zero_active_characters(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    _configure_character_maintenance_model(repositories)
    malformed = repositories.add_character(
        save_id=save.id,
        name="World State: Beacon Lens",
        role="not a character",
        known_state="Erroneous registry row imported from world-state text.",
        character_id="character-malformed-state-entry",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        present_character_ids=[malformed.id],
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="The malformed registry row mentions the beacon lens.",
        tags=[],
        memory_id="memory-malformed-lone",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=malformed.id,
        target_type="memory",
        target_id=memory.id,
        relation="mentions",
    )
    thread = repositories.add_active_thread(
        save_id=save.id,
        title="Malformed state entry",
        related_entities=[malformed.id, f"character:{malformed.id}"],
        thread_id="thread-malformed-lone",
    )
    _append_completed_turns(repositories, save_id=save.id, count=3)
    provider = RecordingStructuredMaintenanceProvider(
        decisions=[
            {
                "operation": "delete",
                "character_id": malformed.id,
                "confidence": 0.95,
                "reason": "This row is a malformed world-state entry, not a person.",
            }
        ],
    )
    service = CharacterRegistryMaintenanceService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(service.maintain_if_due(save_id=save.id))

    assert result.applied == ()
    assert [decision.character_id for decision in result.rejected] == [malformed.id]
    assert len(provider.structured_output_requests) == 1
    assert repositories.get_character(malformed.id) is not None
    assert _character_archived_at(repositories, save.id, malformed.id) is None
    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.present_character_ids == [malformed.id]
    updated_thread = repositories.get_active_thread(thread.id)
    assert updated_thread is not None
    assert updated_thread.related_entities == [
        malformed.id,
        f"character:{malformed.id}",
    ]
    assert [link.entity_id for link in repositories.list_entity_links(save.id)] == [
        malformed.id
    ]


def test_maintain_if_due_rejects_invalid_merge_targets_without_editing_characters(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    _configure_character_maintenance_model(repositories)
    first = repositories.add_character(
        save_id=save.id,
        name="Ashknife",
        character_id="character-ashknife",
    )
    second = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        character_id="character-ilyra",
    )
    third = repositories.add_character(
        save_id=save.id,
        name="Archivist Ren",
        character_id="character-ren",
    )
    _append_completed_turns(repositories, save_id=save.id, count=3)
    provider = RecordingStructuredMaintenanceProvider(
        decisions=[
            {
                "operation": "merge",
                "character_id": first.id,
                "target_character_id": first.id,
                "confidence": 0.97,
                "reason": "A character cannot be merged into itself.",
            },
            {
                "operation": "merge",
                "character_id": "character-missing",
                "target_character_id": second.id,
                "confidence": 0.98,
                "reason": "The source id is not in the active character registry.",
            },
            {
                "operation": "merge",
                "character_id": first.id,
                "target_character_id": "character-missing",
                "confidence": 0.98,
                "reason": "The target id is not in the active character registry.",
            },
            {
                "operation": "merge",
                "character_id": third.id,
                "target_character_id": first.id,
                "confidence": 0.87,
                "reason": "This duplicate proposal is below the confidence threshold.",
            },
            {
                "operation": "merge",
                "character_id": first.id,
                "target_character_id": second.id,
                "confidence": 0.99,
                "reason": "This starts an unsafe chained merge sequence.",
            },
            {
                "operation": "merge",
                "character_id": second.id,
                "target_character_id": third.id,
                "confidence": 0.99,
                "reason": "This completes an unsafe chained merge sequence.",
            },
        ],
    )
    service = CharacterRegistryMaintenanceService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(service.maintain_if_due(save_id=save.id))

    assert result.applied == ()
    assert len(result.rejected) == 6
    assert {character.id for character in repositories.list_characters(save.id)} == {
        first.id,
        second.id,
        third.id,
    }


def test_maintain_if_due_rejects_delete_decisions_with_bad_confidence_or_reason(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    _configure_character_maintenance_model(repositories)
    first = repositories.add_character(
        save_id=save.id,
        name="Ashknife",
        character_id="character-ashknife",
    )
    second = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        character_id="character-ilyra",
    )
    _append_completed_turns(repositories, save_id=save.id, count=3)
    provider = RecordingStructuredMaintenanceProvider(
        decisions=[
            {
                "operation": "delete",
                "character_id": first.id,
                "confidence": 0.87,
                "reason": "This is below the confidence threshold.",
            },
            {
                "operation": "delete",
                "character_id": second.id,
                "confidence": 0.96,
                "reason": "too short",
            },
            {
                "operation": "delete",
                "character_id": first.id,
                "target_character_id": second.id,
                "confidence": 0.99,
                "reason": "Delete decisions cannot name a merge target.",
            },
        ],
    )
    service = CharacterRegistryMaintenanceService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(service.maintain_if_due(save_id=save.id))

    assert result.applied == ()
    assert len(result.rejected) == 3
    assert {character.id for character in repositories.list_characters(save.id)} == {
        first.id,
        second.id,
    }


def test_maintain_if_due_rejects_merge_into_accepted_delete_target_conservatively(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    _configure_character_maintenance_model(repositories)
    source = repositories.add_character(
        save_id=save.id,
        name="Ashknife",
        character_id="character-ashknife",
    )
    delete_target = repositories.add_character(
        save_id=save.id,
        name="World State: Beacon Lens",
        role="not a character",
        known_state="Erroneous registry row imported from world-state text.",
        character_id="character-malformed-state-entry",
    )
    _append_completed_turns(repositories, save_id=save.id, count=3)
    delete_decision = {
        "operation": "delete",
        "character_id": delete_target.id,
        "confidence": 0.95,
        "reason": "This row is a malformed world-state entry, not a person.",
    }
    merge_decision = {
        "operation": "merge",
        "character_id": source.id,
        "target_character_id": delete_target.id,
        "confidence": 0.96,
        "reason": "This duplicate points at a row already accepted for deletion.",
    }
    provider = RecordingStructuredMaintenanceProvider(
        decisions=[delete_decision, merge_decision],
    )
    service = CharacterRegistryMaintenanceService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(service.maintain_if_due(save_id=save.id))

    assert [decision.character_id for decision in result.applied] == [
        delete_target.id
    ]
    assert [decision.character_id for decision in result.rejected] == [source.id]
    assert repositories.get_character(source.id) is not None
    assert _character_archived_at(repositories, save.id, source.id) is None
    assert repositories.get_character(delete_target.id) is None
    assert _character_archived_at(repositories, save.id, delete_target.id) is not None


def test_maintain_if_due_applies_delete_batches_when_one_active_character_remains(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    _configure_character_maintenance_model(repositories)
    keeper = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        character_id="character-ilyra",
    )
    first_delete = repositories.add_character(
        save_id=save.id,
        name="World State: Beacon Lens",
        role="not a character",
        known_state="Erroneous registry row imported from world-state text.",
        character_id="character-malformed-state-entry",
    )
    second_delete = repositories.add_character(
        save_id=save.id,
        name="Location: Beacon Gallery",
        role="not a character",
        known_state="Erroneous registry row imported from location text.",
        character_id="character-malformed-location-entry",
    )
    _append_completed_turns(repositories, save_id=save.id, count=3)
    provider = RecordingStructuredMaintenanceProvider(
        decisions=[
            {
                "operation": "delete",
                "character_id": first_delete.id,
                "confidence": 0.95,
                "reason": "This row is a malformed world-state entry, not a person.",
            },
            {
                "operation": "delete",
                "character_id": second_delete.id,
                "confidence": 0.96,
                "reason": "This row is a malformed location entry, not a person.",
            },
        ],
    )
    service = CharacterRegistryMaintenanceService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(service.maintain_if_due(save_id=save.id))

    assert [decision.character_id for decision in result.applied] == [
        first_delete.id,
        second_delete.id,
    ]
    assert result.rejected == ()
    assert {character.id for character in repositories.list_characters(save.id)} == {
        keeper.id
    }
    assert repositories.get_character(keeper.id) is not None
    assert repositories.get_character(first_delete.id) is None
    assert repositories.get_character(second_delete.id) is None
    assert _character_archived_at(repositories, save.id, first_delete.id) is not None
    assert _character_archived_at(repositories, save.id, second_delete.id) is not None


def test_maintain_if_due_merges_duplicate_character_through_registry_service(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    _configure_character_maintenance_model(repositories)
    gallery = repositories.add_location(save_id=save.id, name="Beacon Gallery")
    canonical = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        aliases=["Glass-Eye"],
        known_state="Commands the watch.",
        relationships={"Mara": "trusted ally"},
        character_id="character-ilyra",
    )
    duplicate = repositories.add_character(
        save_id=save.id,
        name="Ashknife",
        aliases=["The Red Captain"],
        role="Watch captain",
        known_state="Keeps the lens-key phrase.",
        relationships={"Ren": "owes a favor"},
        texting_style="Lowercase fragments, quick double texts under pressure.",
        status="wounded",
        location_id=gallery.id,
        character_id="character-ashknife",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=gallery.id,
        present_character_ids=[duplicate.id],
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="Ashknife knows the lens-key phrase.",
        tags=[],
        memory_id="memory-ashknife",
    )
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="beacon.lens",
        value={"failsafe": "copper notch"},
        state_id="world-state-lens",
    )
    first_message = repositories.append_message(
        save_id=save.id,
        role="player",
        body="I ask about Ilyra.",
        message_id="message-summary-source",
    )
    summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=first_message.id,
        covers_message_end_id=first_message.id,
        body="Ilyra explained the beacon problem.",
        provider="fake",
        model="fake-summary",
        summary_id="summary-ilyra",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=duplicate.id,
        target_type="memory",
        target_id=memory.id,
        relation="knows",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=duplicate.id,
        target_type="state",
        target_id=state.id,
        relation="knows",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=canonical.id,
        target_type="summary",
        target_id=summary.id,
        relation="knows",
    )
    thread = repositories.add_active_thread(
        save_id=save.id,
        title="Lens-key fallout",
        related_entities=[duplicate.id, f"character:{duplicate.id}", canonical.id],
        thread_id="thread-lens-key",
    )
    _append_completed_turns(repositories, save_id=save.id, count=3)
    provider = RecordingStructuredMaintenanceProvider(
        decisions=[
            {
                "operation": "merge",
                "character_id": duplicate.id,
                "target_character_id": canonical.id,
                "confidence": 0.96,
                "reason": "The registry entries describe the same named captain.",
            }
        ],
    )
    service = CharacterRegistryMaintenanceService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(service.maintain_if_due(save_id=save.id))

    assert [decision.character_id for decision in result.applied] == [duplicate.id]
    assert repositories.get_character(duplicate.id) is None
    assert _character_archived_at(repositories, save.id, duplicate.id) is not None
    merged = repositories.get_character(canonical.id)
    assert merged is not None
    assert "Ashknife" in merged.aliases
    assert "The Red Captain" in merged.aliases
    assert merged.role == "Watch captain"
    assert "Keeps the lens-key phrase." in merged.known_state
    assert merged.relationships["Ren"] == "owes a favor"
    assert merged.texting_style == (
        "Lowercase fragments, quick double texts under pressure."
    )
    assert merged.status == "wounded"
    assert merged.location_id == gallery.id
    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.present_character_ids == [canonical.id]
    updated_thread = repositories.get_active_thread(thread.id)
    assert updated_thread is not None
    assert updated_thread.related_entities == [
        canonical.id,
        f"character:{canonical.id}",
    ]
    assert {
        (link.entity_type, link.entity_id, link.target_type, link.target_id)
        for link in repositories.list_entity_links(save.id)
    } == {
        ("character", canonical.id, "memory", memory.id),
        ("character", canonical.id, "world_state", state.id),
        ("character", canonical.id, "summary", summary.id),
    }


def test_maintain_if_due_deletes_malformed_entry_through_registry_service(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    _configure_character_maintenance_model(repositories)
    valid = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        character_id="character-ilyra",
    )
    malformed = repositories.add_character(
        save_id=save.id,
        name="World State: Beacon Lens",
        role="not a character",
        known_state="Erroneous registry row imported from world-state text.",
        character_id="character-malformed-state-entry",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        present_character_ids=[valid.id, malformed.id],
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="The malformed registry row mentions the beacon lens.",
        tags=[],
        memory_id="memory-malformed",
    )
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="beacon.lens",
        value={"status": "cracked"},
        state_id="world-state-lens",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=malformed.id,
        target_type="memory",
        target_id=memory.id,
        relation="knows",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="world_state",
        entity_id=state.id,
        target_type="character",
        target_id=malformed.id,
        relation="mentions",
    )
    thread = repositories.add_active_thread(
        save_id=save.id,
        title="Malformed state entry",
        related_entities=[malformed.id, f"character:{malformed.id}", valid.id],
        thread_id="thread-malformed",
    )
    _append_completed_turns(repositories, save_id=save.id, count=3)
    provider = RecordingStructuredMaintenanceProvider(
        decisions=[
            {
                "operation": "delete",
                "character_id": malformed.id,
                "confidence": 0.95,
                "reason": "This row is a malformed world-state entry, not a person.",
            }
        ],
    )
    service = CharacterRegistryMaintenanceService(
        repositories=repositories,
        providers={"fake": provider},
    )

    result = asyncio.run(service.maintain_if_due(save_id=save.id))

    assert [decision.character_id for decision in result.applied] == [malformed.id]
    assert repositories.get_character(valid.id) is not None
    assert repositories.get_character(malformed.id) is None
    assert _character_archived_at(repositories, save.id, malformed.id) is not None
    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.present_character_ids == [valid.id]
    updated_thread = repositories.get_active_thread(thread.id)
    assert updated_thread is not None
    assert updated_thread.related_entities == [valid.id]
    assert all(
        not (
            (link.entity_type == "character" and link.entity_id == malformed.id)
            or (link.target_type == "character" and link.target_id == malformed.id)
        )
        for link in repositories.list_entity_links(save.id)
    )


def _create_save(
    repositories: PersistenceRepositories,
    *,
    title: str = "Night Watch",
) -> SaveRecord:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    return repositories.create_save(scenario_id=scenario.id, title=title)


def _create_due_save_with_characters(
    repositories: PersistenceRepositories,
) -> SaveRecord:
    save = _create_save(repositories)
    _configure_character_maintenance_model(repositories)
    repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        character_id="character-ilyra",
    )
    repositories.add_character(
        save_id=save.id,
        name="Ashknife",
        character_id="character-ashknife",
    )
    _append_completed_turns(repositories, save_id=save.id, count=3)
    return save


def _configure_character_maintenance_model(
    repositories: PersistenceRepositories,
    *,
    capabilities: list[str] | None = None,
) -> None:
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-character-maintenance",
        display_name="Fake Character Maintenance",
        capabilities=capabilities or ["structured_output"],
        context_window=8192,
    )
    repositories.set_model_preference(
        task="character_registry_maintenance",
        provider="fake",
        model_id="fake-character-maintenance",
    )


def _append_completed_turns(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    count: int,
) -> tuple[str, str]:
    last_player_id = ""
    last_narrator_id = ""
    for index in range(count):
        player = repositories.append_message(
            save_id=save_id,
            role="player",
            speaker_name="Mara",
            body=f"I inspect the beacon lens {index}.",
        )
        narrator = repositories.append_message(
            save_id=save_id,
            role="narrator",
            speaker_name="Narrator",
            body=f"Captain Ilyra answers from the gallery {index}.",
            provider="fake",
            model="fake-chat",
        )
        last_player_id = player.id
        last_narrator_id = narrator.id
    return last_player_id, last_narrator_id


def _append_labelled_completed_turns(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    labels: tuple[str, ...],
) -> tuple[str, str]:
    last_player_id = ""
    last_narrator_id = ""
    for label in labels:
        player = repositories.append_message(
            save_id=save_id,
            role="player",
            speaker_name="Mara",
            body=f"I ask about {label}.",
        )
        narrator = repositories.append_message(
            save_id=save_id,
            role="narrator",
            speaker_name="Narrator",
            body=f"Captain Ilyra answers about {label}.",
            provider="fake",
            model="fake-chat",
        )
        last_player_id = player.id
        last_narrator_id = narrator.id
    return last_player_id, last_narrator_id


def _character_archived_at(
    repositories: PersistenceRepositories,
    save_id: str,
    character_id: str,
) -> str | None:
    row = repositories.connection.execute(
        "SELECT archived_at FROM characters WHERE save_id = ? AND id = ?",
        (save_id, character_id),
    ).fetchone()
    assert row is not None
    archived_at = row["archived_at"]
    assert archived_at is None or isinstance(archived_at, str)
    return archived_at
