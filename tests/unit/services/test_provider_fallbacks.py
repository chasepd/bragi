from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.app_logging import exception_log_fields
from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatMessage,
    ChatPromptPurpose,
    ChatRequest,
    ChatResponse,
    ImageRequest,
    ImageResponse,
    ProviderCapability,
    ProviderClient,
    ProviderConfigStatus,
    ProviderModel,
    StructuredOutputRequest,
    StructuredOutputResponse,
    ToolCallMessage,
    ToolCallRequest,
    ToolCallResponse,
    ToolDefinition,
)
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.services.generation_settings import (
    OPENROUTER_CHAT_REASONING_OVERRIDES_SETTING,
)
from bragi.services.model_preferences import (
    EXTRACTION_TOOL_FALLBACK_MODEL_ID,
    EXTRACTION_TOOL_FALLBACK_PROVIDER,
)
from bragi.services.provider_fallbacks import (
    chat_with_fallback,
    recover_tool_call_shape_with_structured_output,
    structured_output_with_fallback,
    tool_call_fallback_request,
    tool_call_fallback_skip_reason,
)


class RecordingStructuredProvider:
    def __init__(
        self,
        *,
        provider_name: str,
        response_data: dict[str, object] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.response_data = response_data or {"ok": True}
        self.error = error
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
                model_id=f"{self.provider_name}-model",
                display_name=f"{self.provider_name.title()} Model",
                capabilities=frozenset({ProviderCapability.STRUCTURED_OUTPUT}),
            )
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        raise AssertionError("structured fallback tests must not call chat")

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        raise AssertionError("structured fallback tests must not generate images")

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_output_requests.append(request)
        if self.error is not None:
            raise self.error
        return StructuredOutputResponse(
            data=self.response_data,
            provider=request.provider,
            model_id=request.model_id,
        )


class RecordingToolProvider(RecordingStructuredProvider):
    def __init__(
        self,
        *,
        provider_name: str,
        response_calls: tuple[object, ...] = (),
        error: Exception | None = None,
    ) -> None:
        super().__init__(provider_name=provider_name)
        self.response_calls = response_calls
        self.error = error
        self.tool_call_requests: list[ToolCallRequest] = []

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        self.tool_call_requests.append(request)
        if self.error is not None:
            raise self.error
        return ToolCallResponse(
            tool_calls=(),
            body="",
            provider=request.provider,
            model_id=request.model_id,
        )


class RecordingChatProvider:
    def __init__(
        self,
        *,
        provider_name: str,
        response_body: str = "The chat provider should not be called.",
        error: Exception | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.response_body = response_body
        self.error = error
        self.chat_requests: list[ChatRequest] = []

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
                model_id=f"{self.provider_name}-chat",
                display_name=f"{self.provider_name.title()} Chat",
                capabilities=frozenset({ProviderCapability.CHAT}),
            )
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        if self.error is not None:
            raise self.error
        return ChatResponse(
            body=self.response_body,
            provider=request.provider,
            model_id=request.model_id,
        )

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        raise AssertionError("chat fallback tests must not generate images")


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_chat_with_fallback_rejects_unavailable_primary_model(
    repositories: PersistenceRepositories,
) -> None:
    repositories.save_provider_model(
        provider="primary",
        model_id="primary-chat",
        display_name="Primary Chat",
        capabilities=["chat"],
    )
    repositories.mark_missing_provider_models_unavailable(
        provider="primary",
        available_model_ids=set(),
    )
    primary = RecordingChatProvider(provider_name="primary")

    with pytest.raises(ValueError, match="Chat model is unavailable"):
        asyncio.run(
            chat_with_fallback(
                repositories=repositories,
                providers={"primary": primary},
                request=ChatRequest(
                    provider="primary",
                    model_id="primary-chat",
                    messages=(ChatMessage(role="player", body="Summarize this."),),
                ),
                task="summarization",
                save_id="save-1",
            )
        )

    assert primary.chat_requests == []


def test_chat_with_fallback_reports_unavailable_fallback_model(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_app_setting("chat_fallback_enabled", True)
    repositories.set_model_preference(
        task="chat_fallback",
        provider="fallback",
        model_id="fallback-chat",
    )
    repositories.save_provider_model(
        provider="fallback",
        model_id="fallback-chat",
        display_name="Fallback Chat",
        capabilities=["chat"],
    )
    repositories.mark_missing_provider_models_unavailable(
        provider="fallback",
        available_model_ids=set(),
    )
    primary = RecordingChatProvider(
        provider_name="primary",
        error=ProviderError(
            ProviderErrorCategory.CONTENT_BLOCKED,
            "primary chat blocked",
        ),
    )
    with pytest.raises(ProviderError) as captured:
        asyncio.run(
            chat_with_fallback(
                repositories=repositories,
                providers={"primary": primary},
                request=ChatRequest(
                    provider="primary",
                    model_id="primary-chat",
                    messages=(ChatMessage(role="player", body="Summarize this."),),
                ),
                task="summarization",
                save_id="save-1",
            )
        )

    assert len(primary.chat_requests) == 1
    fields = exception_log_fields(captured.value)
    assert fields["fallback_attempted"] is False
    assert fields["fallback_skipped_reason"] == "fallback_model_unavailable"


def test_chat_with_fallback_keeps_model_available_after_model_not_found(
    repositories: PersistenceRepositories,
) -> None:
    repositories.save_provider_model(
        provider="primary",
        model_id="primary-chat",
        display_name="Primary Chat",
        capabilities=["chat"],
    )
    primary = RecordingChatProvider(
        provider_name="primary",
        error=ProviderError(
            ProviderErrorCategory.MODEL_NOT_FOUND,
            "primary chat model missing",
            status_code=404,
        ),
    )

    with pytest.raises(ProviderError):
        asyncio.run(
            chat_with_fallback(
                repositories=repositories,
                providers={"primary": primary},
                request=ChatRequest(
                    provider="primary",
                    model_id="primary-chat",
                    messages=(ChatMessage(role="player", body="Summarize this."),),
                ),
                task="summarization",
                save_id="save-1",
            )
        )

    assert [
        model.available
        for model in repositories.list_provider_models("primary")
        if model.model_id == "primary-chat"
    ] == [True]


def test_chat_with_fallback_model_not_found_keeps_fallback_available(
    repositories: PersistenceRepositories,
) -> None:
    _configure_working_chat_fallback(repositories)
    primary = RecordingChatProvider(
        provider_name="primary",
        error=ProviderError(
            ProviderErrorCategory.PROVIDER_ERROR,
            "primary chat failed",
        ),
    )
    fallback = RecordingChatProvider(
        provider_name="fallback",
        error=ProviderError(
            ProviderErrorCategory.MODEL_NOT_FOUND,
            "fallback chat model missing",
            status_code=404,
        ),
    )

    with pytest.raises(ProviderError) as captured:
        asyncio.run(
            chat_with_fallback(
                repositories=repositories,
                providers={"primary": primary, "fallback": fallback},
                request=ChatRequest(
                    provider="primary",
                    model_id="primary-chat",
                    messages=(ChatMessage(role="player", body="Summarize this."),),
                ),
                task="summarization",
                save_id="save-1",
            )
        )

    assert len(primary.chat_requests) == 1
    assert len(fallback.chat_requests) == 1
    assert "fallback_attempted=true" in str(captured.value)
    assert "fallback_provider=fallback" in str(captured.value)
    assert "fallback_model_id=fallback-chat" in str(captured.value)
    fields = exception_log_fields(captured.value)
    assert fields["fallback_attempted"] is True
    assert fields["fallback_provider"] == "fallback"
    assert fields["fallback_model_id"] == "fallback-chat"
    assert [
        model.available
        for model in repositories.list_provider_models("fallback")
        if model.model_id == "fallback-chat"
    ] == [True]


def test_chat_fallback_rebudgets_against_fallback_model_window(
    repositories: PersistenceRepositories,
) -> None:
    _configure_working_chat_fallback(repositories)
    repositories.save_provider_model(
        provider="primary",
        model_id="primary-chat",
        display_name="Primary Chat",
        capabilities=["chat"],
        context_window=100,
    )
    repositories.save_provider_model(
        provider="fallback",
        model_id="fallback-chat",
        display_name="Fallback Chat",
        capabilities=["chat"],
        context_window=4096,
    )
    primary = RecordingChatProvider(provider_name="primary")
    fallback = RecordingChatProvider(
        provider_name="fallback",
        response_body="A concise summary.",
    )

    response = asyncio.run(
        chat_with_fallback(
            repositories=repositories,
            providers={"primary": primary, "fallback": fallback},
            request=ChatRequest(
                provider="primary",
                model_id="primary-chat",
                prompt_purpose=ChatPromptPurpose.SUMMARY,
                messages=(ChatMessage(role="user", body="界" * 300),),
                max_output_tokens=64,
            ),
            task="summarization",
            save_id="save-1",
        )
    )

    assert primary.chat_requests == []
    assert len(fallback.chat_requests) == 1
    assert fallback.chat_requests[0].model_id == "fallback-chat"
    assert response.body == "A concise summary."


def test_chat_with_blank_primary_keeps_missing_fallback_model_available(
    repositories: PersistenceRepositories,
) -> None:
    _configure_working_chat_fallback(repositories)
    primary = RecordingChatProvider(
        provider_name="primary",
        response_body=" \n\t ",
    )
    fallback = RecordingChatProvider(
        provider_name="fallback",
        error=ProviderError(
            ProviderErrorCategory.MODEL_NOT_FOUND,
            "fallback chat model missing",
            status_code=404,
        ),
    )

    with pytest.raises(ProviderError) as captured:
        asyncio.run(
            chat_with_fallback(
                repositories=repositories,
                providers={"primary": primary, "fallback": fallback},
                request=ChatRequest(
                    provider="primary",
                    model_id="primary-chat",
                    messages=(ChatMessage(role="player", body="Summarize this."),),
                ),
                task="summarization",
                save_id="save-1",
            )
        )

    assert len(primary.chat_requests) == 1
    assert len(fallback.chat_requests) == 1
    assert "fallback_attempted=true" in str(captured.value)
    assert "fallback_provider=fallback" in str(captured.value)
    assert "fallback_model_id=fallback-chat" in str(captured.value)
    fields = exception_log_fields(captured.value)
    assert fields["fallback_attempted"] is True
    assert fields["fallback_provider"] == "fallback"
    assert fields["fallback_model_id"] == "fallback-chat"
    assert [
        model.available
        for model in repositories.list_provider_models("fallback")
        if model.model_id == "fallback-chat"
    ] == [True]


def test_chat_with_fallback_applies_openrouter_reasoning_overrides(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_app_setting(
        OPENROUTER_CHAT_REASONING_OVERRIDES_SETTING,
        {
            "openrouter/primary": "disabled",
            "openrouter/fallback": {"effort": "low", "exclude": True},
        },
    )
    primary = RecordingChatProvider(provider_name="openrouter")

    asyncio.run(
        chat_with_fallback(
            repositories=repositories,
            providers={"openrouter": primary},
            request=ChatRequest(
                provider="openrouter",
                model_id="openrouter/primary",
                messages=(ChatMessage(role="player", body="Summarize this."),),
            ),
            task="summarization",
            save_id="save-1",
        )
    )

    assert primary.chat_requests[0].reasoning is not None
    assert primary.chat_requests[0].reasoning.enabled is False
    assert primary.chat_requests[0].reasoning.exclude is True

    repositories.set_app_setting("chat_fallback_enabled", True)
    repositories.set_model_preference(
        task="chat_fallback",
        provider="openrouter",
        model_id="openrouter/fallback",
    )
    repositories.save_provider_model(
        provider="openrouter",
        model_id="openrouter/fallback",
        display_name="OpenRouter Fallback",
        capabilities=["chat"],
    )
    blocking = RecordingChatProvider(
        provider_name="venice",
        error=ProviderError(
            ProviderErrorCategory.CONTENT_BLOCKED,
            "primary chat blocked",
        ),
    )
    fallback = RecordingChatProvider(provider_name="openrouter")

    asyncio.run(
        chat_with_fallback(
            repositories=repositories,
            providers={"venice": blocking, "openrouter": fallback},
            request=ChatRequest(
                provider="venice",
                model_id="venice-chat",
                messages=(ChatMessage(role="player", body="Summarize this."),),
            ),
            task="summarization",
            save_id="save-1",
        )
    )

    assert blocking.chat_requests[0].reasoning is None
    assert fallback.chat_requests[0].reasoning is not None
    assert fallback.chat_requests[0].reasoning.effort == "low"
    assert fallback.chat_requests[0].reasoning.exclude is True


def test_structured_output_fallback_uses_configured_provider_when_toggle_false(
    repositories: PersistenceRepositories,
) -> None:
    _save_primary_structured_model(repositories)
    repositories.set_app_setting("structured_output_fallback_enabled", False)
    repositories.set_model_preference(
        task="structured_output_fallback",
        provider="fallback",
        model_id="fallback-structured",
    )
    repositories.save_provider_model(
        provider="fallback",
        model_id="fallback-structured",
        display_name="Fallback Structured",
        capabilities=["structured_output"],
    )
    primary = RecordingStructuredProvider(
        provider_name="primary",
        error=ProviderError(
            ProviderErrorCategory.PROVIDER_ERROR,
            "primary structured output failed",
        ),
    )
    fallback = RecordingStructuredProvider(provider_name="fallback")

    response = asyncio.run(
        structured_output_with_fallback(
            repositories=repositories,
            providers={"primary": primary, "fallback": fallback},
            request=_structured_request(),
            task="context_search",
            save_id="save-1",
        )
    )

    assert len(primary.structured_output_requests) == 1
    assert len(fallback.structured_output_requests) == 1
    assert response.provider == "fallback"
    assert response.model_id == "fallback-structured"


def test_structured_output_with_fallback_rejects_unavailable_primary_model(
    repositories: PersistenceRepositories,
) -> None:
    repositories.save_provider_model(
        provider="primary",
        model_id="primary-structured",
        display_name="Primary Structured",
        capabilities=["structured_output"],
    )
    repositories.mark_missing_provider_models_unavailable(
        provider="primary",
        available_model_ids=set(),
    )
    primary = RecordingStructuredProvider(provider_name="primary")

    with pytest.raises(ValueError, match="Structured-output model is unavailable"):
        asyncio.run(
            structured_output_with_fallback(
                repositories=repositories,
                providers={"primary": primary},
                request=StructuredOutputRequest(
                    provider="primary",
                    model_id="primary-structured",
                    schema_name="test_schema",
                    schema={"type": "object"},
                    messages=(ChatMessage(role="player", body="Select context."),),
                ),
                task="context_search",
                save_id="save-1",
            )
        )

    assert primary.structured_output_requests == []


def test_structured_output_with_fallback_rejects_missing_primary_model(
    repositories: PersistenceRepositories,
) -> None:
    primary = RecordingStructuredProvider(provider_name="primary")

    with pytest.raises(ValueError, match="not in the provider model catalog"):
        asyncio.run(
            structured_output_with_fallback(
                repositories=repositories,
                providers={"primary": primary},
                request=_structured_request(),
                task="context_search",
                save_id="save-1",
            )
        )

    assert primary.structured_output_requests == []


def test_structured_output_with_fallback_rejects_primary_without_structured_output(
    repositories: PersistenceRepositories,
) -> None:
    repositories.save_provider_model(
        provider="primary",
        model_id="primary-structured",
        display_name="Primary Structured",
        capabilities=["chat"],
    )
    primary = RecordingStructuredProvider(provider_name="primary")

    with pytest.raises(ValueError, match="does not advertise structured output"):
        asyncio.run(
            structured_output_with_fallback(
                repositories=repositories,
                providers={"primary": primary},
                request=_structured_request(),
                task="context_search",
                save_id="save-1",
            )
        )

    assert primary.structured_output_requests == []


@pytest.mark.parametrize(
    ("skip_reason", "configure_fallback"),
    [
        ("no_fallback_model", "none"),
        ("fallback_provider_unavailable", "missing_provider"),
        ("fallback_model_unavailable", "unavailable_model"),
        ("fallback_model_unavailable", "unavailable_model_missing_provider"),
        ("fallback_model_lacks_required_capabilities", "missing_capability"),
    ],
)
def test_structured_output_fallback_reports_skip_reason_when_unavailable(
    repositories: PersistenceRepositories,
    skip_reason: str,
    configure_fallback: str,
) -> None:
    _save_primary_structured_model(repositories)
    repositories.set_app_setting("structured_output_fallback_enabled", True)
    providers: dict[str, ProviderClient] = {
        "primary": RecordingStructuredProvider(
            provider_name="primary",
            error=ProviderError(
                ProviderErrorCategory.CONTENT_BLOCKED,
                "primary structured output blocked",
            ),
        ),
    }
    if configure_fallback == "missing_provider":
        repositories.set_model_preference(
            task="structured_output_fallback",
            provider="fallback",
            model_id="fallback-structured",
        )
        repositories.save_provider_model(
            provider="fallback",
            model_id="fallback-structured",
            display_name="Fallback Structured",
            capabilities=["structured_output"],
        )
    elif configure_fallback in {
        "unavailable_model",
        "unavailable_model_missing_provider",
    }:
        repositories.set_model_preference(
            task="structured_output_fallback",
            provider="fallback",
            model_id="fallback-structured",
        )
        repositories.save_provider_model(
            provider="fallback",
            model_id="fallback-structured",
            display_name="Fallback Structured",
            capabilities=["structured_output"],
        )
        repositories.mark_missing_provider_models_unavailable(
            provider="fallback",
            available_model_ids=set(),
        )
        if configure_fallback == "unavailable_model":
            providers["fallback"] = RecordingStructuredProvider(
                provider_name="fallback"
            )
    elif configure_fallback == "missing_capability":
        repositories.set_model_preference(
            task="structured_output_fallback",
            provider="fallback",
            model_id="fallback-structured",
        )
        repositories.save_provider_model(
            provider="fallback",
            model_id="fallback-structured",
            display_name="Fallback Structured",
            capabilities=["chat"],
        )
        providers["fallback"] = RecordingStructuredProvider(provider_name="fallback")

    with pytest.raises(ProviderError) as captured:
        asyncio.run(
            structured_output_with_fallback(
                repositories=repositories,
                providers=providers,
                request=_structured_request(),
                task="context_search",
                save_id="save-1",
            )
        )

    assert f"fallback_skipped_reason={skip_reason}" in str(captured.value)
    assert (
        exception_log_fields(captured.value)["fallback_skipped_reason"]
        == skip_reason
    )


def test_structured_output_fallback_attempts_configured_provider(
    repositories: PersistenceRepositories,
) -> None:
    _save_primary_structured_model(repositories)
    _configure_working_structured_fallback(repositories)
    primary = RecordingStructuredProvider(
        provider_name="primary",
        error=ProviderError(
            ProviderErrorCategory.RATE_LIMITED,
            "primary structured output rate limited",
        ),
    )
    fallback = RecordingStructuredProvider(
        provider_name="fallback",
        response_data={"selected": ["memory-1"]},
    )

    response = asyncio.run(
        structured_output_with_fallback(
            repositories=repositories,
            providers={"primary": primary, "fallback": fallback},
            request=_structured_request(),
            task="context_search",
            save_id="save-1",
        )
    )

    assert len(primary.structured_output_requests) == 1
    assert len(fallback.structured_output_requests) == 1
    assert fallback.structured_output_requests[0].provider == "fallback"
    assert fallback.structured_output_requests[0].model_id == "fallback-structured"
    assert response.provider == "fallback"
    assert response.model_id == "fallback-structured"
    assert response.data == {"selected": ["memory-1"]}


def test_structured_output_fallback_recovers_from_schema_violation(
    repositories: PersistenceRepositories,
) -> None:
    _save_primary_structured_model(repositories)
    _configure_working_structured_fallback(repositories)
    primary = RecordingStructuredProvider(
        provider_name="primary",
        error=ProviderError(
            ProviderErrorCategory.STRUCTURED_OUTPUT_INVALID,
            "Structured provider response violated its JSON Schema",
        ),
    )
    fallback = RecordingStructuredProvider(
        provider_name="fallback",
        response_data={"selected": ["memory-1"]},
    )

    response = asyncio.run(
        structured_output_with_fallback(
            repositories=repositories,
            providers={"primary": primary, "fallback": fallback},
            request=_structured_request(),
            task="context_search",
            save_id="save-1",
        )
    )

    assert len(primary.structured_output_requests) == 1
    assert len(fallback.structured_output_requests) == 1
    assert response.provider == "fallback"
    assert response.model_id == "fallback-structured"
    assert response.data == {"selected": ["memory-1"]}


def test_structured_output_fallback_keeps_model_available_after_model_not_found(
    repositories: PersistenceRepositories,
) -> None:
    _save_primary_structured_model(repositories)
    _configure_working_structured_fallback(repositories)
    primary = RecordingStructuredProvider(
        provider_name="primary",
        error=ProviderError(
            ProviderErrorCategory.MODEL_NOT_FOUND,
            "primary structured output model missing",
            status_code=404,
        ),
    )
    fallback = RecordingStructuredProvider(
        provider_name="fallback",
        response_data={"selected": ["memory-1"]},
    )

    response = asyncio.run(
        structured_output_with_fallback(
            repositories=repositories,
            providers={"primary": primary, "fallback": fallback},
            request=_structured_request(),
            task="context_search",
            save_id="save-1",
        )
    )

    assert response.provider == "fallback"
    assert [
        model.available
        for model in repositories.list_provider_models("primary")
        if model.model_id == "primary-structured"
    ] == [True]


def test_structured_output_model_not_found_keeps_fallback_available(
    repositories: PersistenceRepositories,
) -> None:
    _save_primary_structured_model(repositories)
    _configure_working_structured_fallback(repositories)
    primary = RecordingStructuredProvider(
        provider_name="primary",
        error=ProviderError(
            ProviderErrorCategory.PROVIDER_ERROR,
            "primary structured output failed",
        ),
    )
    fallback = RecordingStructuredProvider(
        provider_name="fallback",
        error=ProviderError(
            ProviderErrorCategory.MODEL_NOT_FOUND,
            "fallback structured output model missing",
            status_code=404,
        ),
    )

    with pytest.raises(ProviderError) as captured:
        asyncio.run(
            structured_output_with_fallback(
                repositories=repositories,
                providers={"primary": primary, "fallback": fallback},
                request=_structured_request(),
                task="context_search",
                save_id="save-1",
            )
        )

    assert "fallback_attempted=true" in str(captured.value)
    assert "fallback_provider=fallback" in str(captured.value)
    assert "fallback_model_id=fallback-structured" in str(captured.value)
    fields = exception_log_fields(captured.value)
    assert fields["fallback_attempted"] is True
    assert fields["fallback_provider"] == "fallback"
    assert fields["fallback_model_id"] == "fallback-structured"
    assert [
        model.available
        for model in repositories.list_provider_models("fallback")
        if model.model_id == "fallback-structured"
    ] == [True]


def test_tool_call_fallback_uses_configured_provider_when_toggle_false(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_app_setting("tool_call_fallback_enabled", False)
    repositories.set_model_preference(
        task="tool_call_fallback",
        provider="fallback",
        model_id="fallback-tools",
    )
    repositories.save_provider_model(
        provider="fallback",
        model_id="fallback-tools",
        display_name="Fallback Tools",
        capabilities=["tool_calling"],
    )
    providers: dict[str, ProviderClient] = {
        "fallback": RecordingToolProvider(provider_name="fallback"),
    }

    request = tool_call_fallback_request(
        repositories=repositories,
        providers=providers,
        request=_tool_call_request(),
        save_id="save-1",
    )

    assert request is not None
    assert request.provider == "fallback"
    assert request.model_id == "fallback-tools"


@pytest.mark.parametrize(
    ("skip_reason", "configure_fallback"),
    [
        ("no_fallback_model", "none"),
        ("fallback_provider_unavailable", "missing_provider"),
        ("fallback_model_unavailable", "unavailable_model"),
        ("fallback_model_unavailable", "unavailable_model_missing_provider"),
        ("fallback_model_lacks_required_capabilities", "missing_capability"),
    ],
)
def test_tool_call_fallback_reports_skip_reason_when_unavailable(
    repositories: PersistenceRepositories,
    skip_reason: str,
    configure_fallback: str,
) -> None:
    repositories.set_app_setting("tool_call_fallback_enabled", True)
    providers: dict[str, ProviderClient] = {}
    if configure_fallback == "missing_provider":
        repositories.set_model_preference(
            task="tool_call_fallback",
            provider="fallback",
            model_id="fallback-tools",
        )
        repositories.save_provider_model(
            provider="fallback",
            model_id="fallback-tools",
            display_name="Fallback Tools",
            capabilities=["tool_calling"],
        )
    elif configure_fallback in {
        "unavailable_model",
        "unavailable_model_missing_provider",
    }:
        repositories.set_model_preference(
            task="tool_call_fallback",
            provider="fallback",
            model_id="fallback-tools",
        )
        repositories.save_provider_model(
            provider="fallback",
            model_id="fallback-tools",
            display_name="Fallback Tools",
            capabilities=["tool_calling"],
        )
        repositories.mark_missing_provider_models_unavailable(
            provider="fallback",
            available_model_ids=set(),
        )
        if configure_fallback == "unavailable_model":
            providers["fallback"] = RecordingToolProvider(provider_name="fallback")
    elif configure_fallback == "missing_capability":
        repositories.set_model_preference(
            task="tool_call_fallback",
            provider="fallback",
            model_id="fallback-tools",
        )
        repositories.save_provider_model(
            provider="fallback",
            model_id="fallback-tools",
            display_name="Fallback Tools",
            capabilities=["structured_output"],
        )
        providers["fallback"] = RecordingToolProvider(provider_name="fallback")

    assert (
        tool_call_fallback_skip_reason(
            repositories=repositories,
            providers=providers,
            save_id="save-1",
        )
        == skip_reason
    )
    assert (
        tool_call_fallback_request(
            repositories=repositories,
            providers=providers,
            request=_tool_call_request(),
            save_id="save-1",
        )
        is None
    )


def test_tool_call_fallback_rewrites_request_for_configured_provider(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_app_setting("tool_call_fallback_enabled", True)
    repositories.set_model_preference(
        task="tool_call_fallback",
        provider="fallback",
        model_id="fallback-tools",
    )
    repositories.save_provider_model(
        provider="fallback",
        model_id="fallback-tools",
        display_name="Fallback Tools",
        capabilities=["function_calling"],
    )
    providers: dict[str, ProviderClient] = {
        "fallback": RecordingToolProvider(provider_name="fallback"),
    }

    request = tool_call_fallback_request(
        repositories=repositories,
        providers=providers,
        request=_tool_call_request(),
        save_id="save-1",
    )

    assert request is not None
    assert request.provider == "fallback"
    assert request.model_id == "fallback-tools"
    assert request.messages == _tool_call_request().messages


def test_tool_call_fallback_uses_recommended_model_when_configured_preference_missing(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_app_setting("tool_call_fallback_enabled", True)
    repositories.save_provider_model(
        provider=EXTRACTION_TOOL_FALLBACK_PROVIDER,
        model_id=EXTRACTION_TOOL_FALLBACK_MODEL_ID,
        display_name="Qwen 9B",
        capabilities=["tool_calling"],
    )
    providers: dict[str, ProviderClient] = {
        EXTRACTION_TOOL_FALLBACK_PROVIDER: RecordingToolProvider(
            provider_name=EXTRACTION_TOOL_FALLBACK_PROVIDER
        ),
    }

    request = tool_call_fallback_request(
        repositories=repositories,
        providers=providers,
        request=_tool_call_request(),
        save_id="save-1",
    )

    assert request is not None
    assert request.provider == EXTRACTION_TOOL_FALLBACK_PROVIDER
    assert request.model_id == EXTRACTION_TOOL_FALLBACK_MODEL_ID


def test_tool_call_fallback_reports_unavailable_recommended_model(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_app_setting("tool_call_fallback_enabled", True)
    repositories.save_provider_model(
        provider=EXTRACTION_TOOL_FALLBACK_PROVIDER,
        model_id=EXTRACTION_TOOL_FALLBACK_MODEL_ID,
        display_name="Qwen 9B",
        capabilities=["tool_calling"],
    )
    repositories.mark_missing_provider_models_unavailable(
        provider=EXTRACTION_TOOL_FALLBACK_PROVIDER,
        available_model_ids=set(),
    )
    providers: dict[str, ProviderClient] = {
        EXTRACTION_TOOL_FALLBACK_PROVIDER: RecordingToolProvider(
            provider_name=EXTRACTION_TOOL_FALLBACK_PROVIDER
        ),
    }

    assert (
        tool_call_fallback_request(
            repositories=repositories,
            providers=providers,
            request=_tool_call_request(),
            save_id="save-1",
        )
        is None
    )
    assert (
        tool_call_fallback_skip_reason(
            repositories=repositories,
            providers=providers,
            save_id="save-1",
        )
        == "fallback_model_unavailable"
    )


def _configure_working_structured_fallback(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_model_preference(
        task="structured_output_fallback",
        provider="fallback",
        model_id="fallback-structured",
    )
    repositories.save_provider_model(
        provider="fallback",
        model_id="fallback-structured",
        display_name="Fallback Structured",
        capabilities=["structured_output"],
    )


def _save_primary_structured_model(repositories: PersistenceRepositories) -> None:
    repositories.save_provider_model(
        provider="primary",
        model_id="primary-structured",
        display_name="Primary Structured",
        capabilities=["structured_output"],
    )


def _configure_working_chat_fallback(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_model_preference(
        task="chat_fallback",
        provider="fallback",
        model_id="fallback-chat",
    )
    repositories.save_provider_model(
        provider="fallback",
        model_id="fallback-chat",
        display_name="Fallback Chat",
        capabilities=["chat"],
    )


def _structured_request() -> StructuredOutputRequest:
    return StructuredOutputRequest(
        provider="primary",
        model_id="primary-structured",
        schema_name="test_schema",
        schema={"type": "object"},
        messages=(ChatMessage(role="user", body="Select context."),),
    )


def _tool_call_request() -> ToolCallRequest:
    return ToolCallRequest(
        provider="primary",
        model_id="primary-tools",
        messages=(ToolCallMessage(role="user", body="Extract updates."),),
        tools=(
            ToolDefinition(
                name="update_scene_snapshot",
                description="Updates the current scene.",
                parameters={
                    "type": "object",
                    "properties": {"source_message_id": {"type": "string"}},
                    "required": ["source_message_id"],
                    "additionalProperties": False,
                },
            ),
        ),
    )


def test_recover_tool_call_shape_reruns_on_model_not_found() -> None:
    async def run() -> None:
        provider = RecordingStructuredProvider(
            provider_name="primary",
            response_data={"selections": []},
        )
        result = await recover_tool_call_shape_with_structured_output(
            error=ProviderError(
                ProviderErrorCategory.MODEL_NOT_FOUND,
                "model not found",
                status_code=404,
            ),
            task="context_search",
            provider="primary",
            model_id="primary-tools",
            structured_run=lambda: _structured_selection(provider),
        )

        assert result == {"selections": []}
        assert len(provider.structured_output_requests) == 1

    asyncio.run(run())


def test_recover_tool_call_shape_skips_other_categories() -> None:
    async def run() -> None:
        error = ProviderError(
            ProviderErrorCategory.RATE_LIMITED,
            "rate limited",
            status_code=429,
        )
        with pytest.raises(ProviderError) as exc_info:
            await recover_tool_call_shape_with_structured_output(
                error=error,
                task="context_search",
                provider="primary",
                model_id="primary-tools",
                structured_run=lambda: _fail_structured_run(),
            )
        assert exc_info.value is error

    asyncio.run(run())


def test_recover_tool_call_shape_enriches_structured_failure() -> None:
    async def run() -> None:
        provider = RecordingStructuredProvider(
            provider_name="primary",
            error=ProviderError(
                ProviderErrorCategory.MODEL_NOT_FOUND,
                "structured model not found",
                status_code=404,
            ),
        )
        with pytest.raises(ProviderError) as exc_info:
            await recover_tool_call_shape_with_structured_output(
                error=ProviderError(
                    ProviderErrorCategory.MODEL_NOT_FOUND,
                    "tool model not found",
                    status_code=404,
                ),
                task="context_search",
                provider="primary",
                model_id="primary-tools",
                structured_run=lambda: provider.generate_structured_output(
                    StructuredOutputRequest(
                        provider="primary",
                        model_id="primary-tools",
                        schema_name="context_search_selection",
                        schema={"type": "object"},
                        messages=(
                            ChatMessage(role="user", body="Select context."),
                        ),
                    )
                ),
            )

        assert exc_info.value.category == ProviderErrorCategory.MODEL_NOT_FOUND
        assert exc_info.value.fallback_attempted is True
        assert exc_info.value.fallback_provider == "primary"
        assert exc_info.value.fallback_model_id == "primary-tools"

    asyncio.run(run())


async def _structured_selection(
    provider: RecordingStructuredProvider,
) -> dict[str, object]:
    response = await provider.generate_structured_output(
        StructuredOutputRequest(
            provider="primary",
            model_id="primary-tools",
            schema_name="context_search_selection",
            schema={"type": "object"},
            messages=(ChatMessage(role="user", body="Select context."),),
        )
    )
    return response.data


async def _fail_structured_run() -> dict[str, object]:
    raise AssertionError("structured route must not run")
