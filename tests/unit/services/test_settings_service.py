from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Iterator
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.paths import StoragePaths
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ProviderCapability,
    ProviderCatalogEntry,
    ProviderConfigStatus,
    ProviderGenerationParameter,
    ProviderModel,
    ProviderModelListResponse,
    ProviderModelPricing,
    ProviderThinkingLevelSupport,
    VideoRequest,
    VideoResponse,
)
from bragi.providers.errors import ProviderErrorCategory
from bragi.services.agentic_context import AGENTIC_CONTEXT_PIPELINE_SETTING
from bragi.services.character_text_service import (
    CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING,
    CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING,
    DEFAULT_CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_PERCENT,
    DEFAULT_CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_TURNS,
    MAX_CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_PERCENT,
    MAX_CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_TURNS,
)
from bragi.services.chat_history_settings import (
    DEFAULT_NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW,
    DEFAULT_NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW,
    DEFAULT_RECENT_NARRATOR_MESSAGE_WINDOW,
    DEFAULT_RECENT_PLAYER_MESSAGE_WINDOW,
    NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
    NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
    RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
    RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
    ChatHistoryWindowSettings,
)
from bragi.services.content_rating import (
    CONTENT_FILTER_RATING_SETTING,
    FADE_TO_BLACK_ENABLED_SETTING,
)
from bragi.services.generation_settings import (
    MODEL_THINKING_PREFERENCES_SETTING,
    OPENROUTER_CHAT_REASONING_OVERRIDES_SETTING,
)
from bragi.services.image_style_settings import (
    IMAGE_STYLE_PRESET_SETTING,
    save_image_style_preset_setting_key,
)
from bragi.services.model_preferences import SAVE_MODEL_OVERRIDES_SETTING
from bragi.services.phrase_denylist import (
    GENERATED_PHRASE_DENYLIST_SETTING,
    SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
)
from bragi.services.post_turn_inference import POST_TURN_INFERENCE_MODE_SETTING
from bragi.services.secrets import InMemorySecretStore, SecretStorageError
from bragi.services.settings_service import SettingsService
from bragi.services.text_script_policy import (
    DEFAULT_SCRIPT_GUARD_MODE,
    SCRIPT_GUARD_MODE_LATIN_ONLY_REJECT,
    SCRIPT_GUARD_MODE_SETTING,
)

API_KEY_SENTINEL = "sk-settings-service-sqlite-sentinel"


class RecordingProvider:
    provider_name = "openrouter"

    def __init__(
        self,
        *,
        status: ProviderConfigStatus | None = None,
        models: list[ProviderModel] | None = None,
        validate_exception: Exception | None = None,
        list_models_exception: Exception | None = None,
    ) -> None:
        self.status = status or ProviderConfigStatus(
            provider=self.provider_name,
            configured=True,
            authenticated=True,
        )
        self.validate_exception = validate_exception
        self.list_models_exception = list_models_exception
        self.models = models or [
            ProviderModel(
                provider=self.provider_name,
                model_id="anthropic/claude-3.5-sonnet",
                display_name="Claude 3.5 Sonnet",
                capabilities=frozenset({ProviderCapability.CHAT}),
                context_window=200_000,
            ),
            ProviderModel(
                provider=self.provider_name,
                model_id="google/gemini-2.5-flash-image-preview",
                display_name="Gemini 2.5 Flash Image",
                capabilities=frozenset({ProviderCapability.IMAGE_GENERATION}),
                context_window=32_768,
            ),
        ]
        self.calls: list[str] = []

    async def validate_config(self) -> ProviderConfigStatus:
        self.calls.append("validate_config")
        if self.validate_exception is not None:
            raise self.validate_exception
        return self.status

    async def list_models(self) -> list[ProviderModel]:
        self.calls.append("list_models")
        if self.list_models_exception is not None:
            raise self.list_models_exception
        return self.models


class CatalogRecordingProvider(RecordingProvider):
    def __init__(
        self,
        *,
        catalog: list[ProviderCatalogEntry] | None = None,
        catalog_exception: Exception | None = None,
    ) -> None:
        super().__init__()
        self.catalog = catalog or [
            ProviderCatalogEntry(
                slug="openai",
                name="OpenAI",
                privacy_policy_url="https://openai.com/privacy",
                terms_of_service_url="https://openai.com/terms",
                status_page_url="https://status.openai.com",
                headquarters="US",
                datacenters=("US", "IE"),
            ),
            ProviderCatalogEntry(slug="deepinfra", name="DeepInfra"),
        ]
        self.catalog_exception = catalog_exception

    async def list_providers(self) -> list[ProviderCatalogEntry]:
        self.calls.append("list_providers")
        if self.catalog_exception is not None:
            raise self.catalog_exception
        return self.catalog


class MetadataRecordingProvider(RecordingProvider):
    async def list_models_with_metadata(self) -> ProviderModelListResponse:
        self.calls.append("list_models_with_metadata")
        return ProviderModelListResponse(
            models=self.models,
            raw_metadata={
                "_bragi_retry": {
                    "attempt_count": 2,
                    "max_attempts": 3,
                    "attempts": [
                        {
                            "attempt": 1,
                            "duration_ms": 12,
                            "error_category": "network_error",
                        },
                        {
                            "attempt": 2,
                            "duration_ms": 7,
                            "error_category": None,
                        },
                    ],
                }
            },
        )


class RecordingVideoProvider(RecordingProvider):
    async def generate_video(self, request: VideoRequest) -> VideoResponse:
        return VideoResponse(
            provider=request.provider,
            model_id=request.model_id,
            mime_type="video/mp4",
            video_bytes=b"fake-video",
        )


class SecretReadingProvider:
    provider_name = "openrouter"

    def __init__(self, secret_store: SecretStorageFailureStore) -> None:
        self.secret_store = secret_store
        self.calls: list[str] = []

    async def validate_config(self) -> ProviderConfigStatus:
        self.calls.append("validate_config")
        self.secret_store.get_api_key(self.provider_name)
        return ProviderConfigStatus(
            provider=self.provider_name,
            configured=True,
            authenticated=True,
        )

    async def list_models(self) -> list[ProviderModel]:
        self.calls.append("list_models")
        return []


class SecretStorageFailureStore(InMemorySecretStore):
    def __init__(
        self,
        *,
        fail_has_api_key: bool = False,
        fail_get_api_key: bool = False,
    ) -> None:
        super().__init__()
        self.fail_has_api_key = fail_has_api_key
        self.fail_get_api_key = fail_get_api_key
        self.calls: list[str] = []

    def has_api_key(self, provider: str) -> bool:
        self.calls.append("has_api_key")
        if self.fail_has_api_key:
            raise SecretStorageError("keyring read failed")
        return super().has_api_key(provider)

    def get_api_key(self, provider: str) -> str | None:
        self.calls.append("get_api_key")
        if self.fail_get_api_key:
            raise SecretStorageError("keyring read failed")
        return super().get_api_key(provider)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_refresh_provider_models_validates_and_stores_non_secret_metadata(
    repositories: PersistenceRepositories,
) -> None:
    provider = RecordingProvider()
    secret_store = InMemorySecretStore()
    service = SettingsService(
        repositories=repositories,
        providers={"openrouter": provider},
        secret_store=secret_store,
    )
    service.set_provider_api_key("openrouter", "sk-openrouter-test-secret")

    result = asyncio.run(service.refresh_provider_models("openrouter"))

    assert provider.calls == ["validate_config", "list_models"]
    assert result.provider == "openrouter"
    assert result.configured is True
    assert result.authenticated is True
    assert result.model_count == 2
    assert result.error is None

    config = repositories.get_provider_config("openrouter")
    assert config is not None
    assert config.provider == "openrouter"
    assert config.enabled is True
    assert config.has_api_key is True
    assert config.last_error is None
    assert config.last_model_refresh_at is not None

    models = repositories.list_provider_models("openrouter")
    assert [(model.model_id, model.display_name) for model in models] == [
        ("anthropic/claude-3.5-sonnet", "Claude 3.5 Sonnet"),
        ("google/gemini-2.5-flash-image-preview", "Gemini 2.5 Flash Image"),
    ]
    assert models[0].capabilities == ["chat"]
    assert models[0].context_window == 200_000
    assert models[1].capabilities == ["image_generation"]
    assert models[1].context_window == 32_768
    assert secret_store.get_api_key("openrouter") == "sk-openrouter-test-secret"
    assert "sk-openrouter-test-secret" not in "\n".join(
        repositories.connection.iterdump()
    )


def test_refresh_provider_models_caches_openrouter_provider_catalog(
    repositories: PersistenceRepositories,
) -> None:
    provider = CatalogRecordingProvider()
    secret_store = InMemorySecretStore()
    service = SettingsService(
        repositories=repositories,
        providers={"openrouter": provider},
        secret_store=secret_store,
    )
    service.set_provider_api_key("openrouter", "sk-openrouter-test-secret")

    result = asyncio.run(service.refresh_provider_models("openrouter"))

    assert provider.calls == ["validate_config", "list_models", "list_providers"]
    assert result.error is None
    catalog = repositories.list_provider_catalog_entries("openrouter")
    assert [(entry.slug, entry.name) for entry in catalog] == [
        ("deepinfra", "DeepInfra"),
        ("openai", "OpenAI"),
    ]
    openai = next(entry for entry in catalog if entry.slug == "openai")
    assert openai.privacy_policy_url == "https://openai.com/privacy"
    assert openai.terms_of_service_url == "https://openai.com/terms"
    assert openai.status_page_url == "https://status.openai.com"
    assert openai.headquarters == "US"
    assert openai.datacenters == ["US", "IE"]
    assert openai.refreshed_at is not None


def test_refresh_provider_models_keeps_model_refresh_success_when_catalog_fetch_fails(
    repositories: PersistenceRepositories,
) -> None:
    provider = CatalogRecordingProvider(catalog_exception=RuntimeError("catalog down"))
    secret_store = InMemorySecretStore()
    service = SettingsService(
        repositories=repositories,
        providers={"openrouter": provider},
        secret_store=secret_store,
    )
    service.set_provider_api_key("openrouter", "sk-openrouter-test-secret")

    result = asyncio.run(service.refresh_provider_models("openrouter"))

    assert provider.calls == ["validate_config", "list_models", "list_providers"]
    assert result.configured is True
    assert result.authenticated is True
    assert result.model_count == 2
    assert result.error is None
    assert repositories.list_provider_models("openrouter")
    assert repositories.list_provider_catalog_entries("openrouter") == []


def test_settings_service_rejects_blank_provider_api_key(
    repositories: PersistenceRepositories,
) -> None:
    secret_store = InMemorySecretStore()
    service = SettingsService(
        repositories=repositories,
        providers={"openrouter": RecordingProvider()},
        secret_store=secret_store,
    )

    with pytest.raises(ValueError, match="API key"):
        service.set_provider_api_key("openrouter", "  ")

    assert secret_store.get_api_key("openrouter") is None
    assert repositories.get_provider_config("openrouter") is None


def test_settings_service_clears_provider_api_key_and_stale_status(
    repositories: PersistenceRepositories,
) -> None:
    secret_store = InMemorySecretStore()
    secret_store.set_api_key("openrouter", "sk-openrouter-test-secret")
    repositories.upsert_provider_config(
        provider="openrouter",
        enabled=True,
        has_api_key=True,
        last_model_refresh_at="2026-05-12T18:30:00Z",
        last_error="Invalid API key",
    )
    service = SettingsService(
        repositories=repositories,
        providers={"openrouter": RecordingProvider()},
        secret_store=secret_store,
    )

    service.clear_provider_api_key("openrouter")

    assert secret_store.get_api_key("openrouter") is None
    config = repositories.get_provider_config("openrouter")
    assert config is not None
    assert config.enabled is False
    assert config.has_api_key is False
    assert config.last_model_refresh_at is None
    assert config.last_error is None


def test_refresh_provider_models_persists_generation_parameters(
    repositories: PersistenceRepositories,
) -> None:
    provider = RecordingProvider(
        models=[
            ProviderModel(
                provider="openrouter",
                model_id="openrouter/chat",
                display_name="OpenRouter Chat",
                capabilities=frozenset({ProviderCapability.CHAT}),
                supported_parameters=frozenset(
                    {
                        ProviderGenerationParameter.TEMPERATURE,
                        ProviderGenerationParameter.MAX_OUTPUT_TOKENS,
                    }
                ),
            )
        ]
    )
    secret_store = InMemorySecretStore()
    service = SettingsService(
        repositories=repositories,
        providers={"openrouter": provider},
        secret_store=secret_store,
    )
    service.set_provider_api_key("openrouter", "sk-openrouter-test-secret")

    result = asyncio.run(service.refresh_provider_models("openrouter"))

    assert result.model_count == 1
    models = repositories.list_provider_models("openrouter")
    assert models[0].supported_parameters == [
        "max_output_tokens",
        "temperature",
    ]


def test_refresh_provider_models_persists_thinking_metadata(
    repositories: PersistenceRepositories,
) -> None:
    provider = RecordingProvider(
        models=[
            ProviderModel(
                provider="openrouter",
                model_id="openrouter/reasoning",
                display_name="OpenRouter Reasoning",
                capabilities=frozenset({ProviderCapability.CHAT}),
                thinking=ProviderThinkingLevelSupport(
                    levels=("high", "medium", "low"),
                    default_level="medium",
                    default_enabled=True,
                    mandatory=False,
                    supports_max_tokens=True,
                ),
            )
        ]
    )
    secret_store = InMemorySecretStore()
    service = SettingsService(
        repositories=repositories,
        providers={"openrouter": provider},
        secret_store=secret_store,
    )
    service.set_provider_api_key("openrouter", "sk-openrouter-test-secret")

    result = asyncio.run(service.refresh_provider_models("openrouter"))

    assert result.model_count == 1
    models = repositories.list_provider_models("openrouter")
    assert models[0].thinking == {
        "default_enabled": True,
        "default_level": "medium",
        "levels": ["high", "medium", "low"],
        "mandatory": False,
        "supports_max_tokens": True,
    }


def test_set_model_thinking_preference_persists_supported_level(
    repositories: PersistenceRepositories,
) -> None:
    repositories.save_provider_model(
        provider="openrouter",
        model_id="openai/gpt-5-mini",
        display_name="GPT-5 Mini",
        capabilities=["chat"],
        thinking={
            "levels": ["high", "low"],
            "default_level": "low",
            "mandatory": False,
        },
    )
    service = SettingsService(
        repositories=repositories,
        providers={},
        secret_store=InMemorySecretStore(),
    )

    service.set_model_thinking_preference(
        task="chat",
        provider="openrouter",
        model_id="openai/gpt-5-mini",
        level="High",
    )

    assert repositories.get_app_setting(MODEL_THINKING_PREFERENCES_SETTING) == {
        "chat": {
            "provider": "openrouter",
            "model_id": "openai/gpt-5-mini",
            "level": "high",
        }
    }


def test_set_model_thinking_preference_rejects_unsupported_model_and_level(
    repositories: PersistenceRepositories,
) -> None:
    repositories.save_provider_model(
        provider="openrouter",
        model_id="openai/gpt-5-mini",
        display_name="GPT-5 Mini",
        capabilities=["chat"],
        thinking={"levels": ["high"], "mandatory": True},
    )
    service = SettingsService(
        repositories=repositories,
        providers={},
        secret_store=InMemorySecretStore(),
    )

    with pytest.raises(ValueError, match="does not support thinking"):
        service.set_model_thinking_preference(
            task="chat",
            provider="openrouter",
            model_id="missing",
            level="high",
        )
    with pytest.raises(ValueError, match="not supported"):
        service.set_model_thinking_preference(
            task="chat",
            provider="openrouter",
            model_id="openai/gpt-5-mini",
            level="low",
        )
    with pytest.raises(ValueError, match="requires thinking"):
        service.set_model_thinking_preference(
            task="chat",
            provider="openrouter",
            model_id="openai/gpt-5-mini",
            level="off",
        )


def test_set_model_preference_clears_stale_model_thinking_preference(
    repositories: PersistenceRepositories,
) -> None:
    service = SettingsService(
        repositories=repositories,
        providers={},
        secret_store=InMemorySecretStore(),
    )
    repositories.set_app_setting(
        MODEL_THINKING_PREFERENCES_SETTING,
        {
            "chat": {
                "provider": "openrouter",
                "model_id": "old-model",
                "level": "high",
            }
        },
    )

    service.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="new-model",
    )

    assert repositories.get_app_setting(MODEL_THINKING_PREFERENCES_SETTING) == {}


def test_set_save_model_preference_stores_only_save_override(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A keep is isolated by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/server-chat",
    )
    service = _settings_service(repositories)

    service.set_model_preference(
        task="chat",
        provider="venice",
        model_id="venice/save-chat",
        save_id=save.id,
    )

    server_preference = repositories.get_model_preference("chat")
    assert server_preference is not None
    assert server_preference.model_id == "openrouter/server-chat"
    assert repositories.get_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=SAVE_MODEL_OVERRIDES_SETTING,
    ) == {
        "preferences": {
            "chat": {
                "provider": "venice",
                "model_id": "venice/save-chat",
            }
        }
    }


def test_set_save_model_preference_matching_server_value_clears_override(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A keep is isolated by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/server-chat",
    )
    service = _settings_service(repositories)
    service.set_model_preference(
        task="chat",
        provider="venice",
        model_id="venice/save-chat",
        save_id=save.id,
    )

    service.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/server-chat",
        save_id=save.id,
    )

    assert (
        repositories.get_scoped_setting(
            scope="save",
            scope_id=save.id,
            key=SAVE_MODEL_OVERRIDES_SETTING,
        )
        is None
    )


def test_set_save_model_preference_clears_stale_save_thinking_override(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A keep is isolated by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.save_provider_model(
        provider="venice",
        model_id="venice/old-save-chat",
        display_name="Old Save Chat",
        capabilities=["chat"],
        thinking={"levels": ["high"], "mandatory": False},
    )
    repositories.save_provider_model(
        provider="venice",
        model_id="venice/new-save-chat",
        display_name="New Save Chat",
        capabilities=["chat"],
        thinking={"levels": ["high"], "mandatory": False},
    )
    service = _settings_service(repositories)
    service.set_model_preference(
        task="chat",
        provider="venice",
        model_id="venice/old-save-chat",
        save_id=save.id,
    )
    service.set_model_thinking_preference(
        task="chat",
        provider="venice",
        model_id="venice/old-save-chat",
        level="high",
        save_id=save.id,
    )

    service.set_model_preference(
        task="chat",
        provider="venice",
        model_id="venice/new-save-chat",
        save_id=save.id,
    )

    assert repositories.get_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=SAVE_MODEL_OVERRIDES_SETTING,
    ) == {
        "preferences": {
            "chat": {
                "provider": "venice",
                "model_id": "venice/new-save-chat",
            }
        }
    }


def test_refresh_provider_models_persists_pricing_metadata(
    repositories: PersistenceRepositories,
) -> None:
    provider = RecordingProvider(
        models=[
            ProviderModel(
                provider="openrouter",
                model_id="openrouter/chat",
                display_name="OpenRouter Chat",
                capabilities=frozenset({ProviderCapability.CHAT}),
                pricing=ProviderModelPricing(
                    input_per_million_tokens_usd="0.15",
                    output_per_million_tokens_usd="0.6",
                    cache_read_per_million_tokens_usd="0.01",
                    cache_write_per_million_tokens_usd="0.02",
                ),
            )
        ]
    )
    secret_store = InMemorySecretStore()
    service = SettingsService(
        repositories=repositories,
        providers={"openrouter": provider},
        secret_store=secret_store,
    )
    service.set_provider_api_key("openrouter", "sk-openrouter-test-secret")

    result = asyncio.run(service.refresh_provider_models("openrouter"))

    assert result.model_count == 1
    models = repositories.list_provider_models("openrouter")
    assert models[0].pricing == {
        "input_per_million_tokens_usd": "0.15",
        "output_per_million_tokens_usd": "0.6",
        "cache_read_per_million_tokens_usd": "0.01",
        "cache_write_per_million_tokens_usd": "0.02",
    }


def test_refresh_provider_models_persists_safe_retry_metadata(
    repositories: PersistenceRepositories,
) -> None:
    provider = MetadataRecordingProvider()
    secret_store = InMemorySecretStore()
    service = SettingsService(
        repositories=repositories,
        providers={"openrouter": provider},
        secret_store=secret_store,
    )
    service.set_provider_api_key("openrouter", "sk-openrouter-test-secret")

    result = asyncio.run(service.refresh_provider_models("openrouter"))

    jobs = repositories.list_jobs_by_status(("succeeded",))
    assert result.model_count == 2
    assert provider.calls == ["validate_config", "list_models_with_metadata"]
    assert len(jobs) == 1
    assert jobs[0].type == "model_refresh"
    assert jobs[0].result == {
        "model_count": 2,
        "attempt_count": 2,
        "max_attempts": 3,
        "provider_call_count": 1,
        "provider_calls": [
            {
                "task": "model_listing",
                "provider": "openrouter",
                "attempt_count": 2,
                "max_attempts": 3,
                "retry_attempts": [
                    {
                        "attempt": 1,
                        "duration_ms": 12,
                        "error_category": "network_error",
                    },
                    {
                        "attempt": 2,
                        "duration_ms": 7,
                        "error_category": None,
                    },
                ],
            }
        ],
        "retry_attempts": [
            {
                "attempt": 1,
                "duration_ms": 12,
                "error_category": "network_error",
            },
            {
                "attempt": 2,
                "duration_ms": 7,
                "error_category": None,
            },
        ],
    }


def test_refresh_provider_models_strips_video_capabilities_without_runtime_support(
    repositories: PersistenceRepositories,
) -> None:
    provider = RecordingProvider(
        models=[
            ProviderModel(
                provider="openrouter",
                model_id="openrouter/claims-video",
                display_name="Claims Video",
                capabilities=frozenset(
                    {
                        ProviderCapability.CHAT,
                        ProviderCapability.IMAGE_GENERATION,
                        ProviderCapability.TEXT_TO_VIDEO,
                        ProviderCapability.IMAGE_TO_VIDEO,
                        ProviderCapability.IMAGE_PLUS_TEXT_TO_VIDEO,
                    }
                ),
                context_window=32_768,
            )
        ]
    )
    service = SettingsService(
        repositories=repositories,
        providers={"openrouter": provider},
        secret_store=InMemorySecretStore(),
    )

    asyncio.run(service.refresh_provider_models("openrouter"))

    models = repositories.list_provider_models("openrouter")
    assert len(models) == 1
    assert models[0].capabilities == ["chat", "image_generation"]


def test_refresh_provider_models_retains_video_capabilities_with_runtime_support(
    repositories: PersistenceRepositories,
) -> None:
    provider = RecordingVideoProvider(
        models=[
            ProviderModel(
                provider="openrouter",
                model_id="openrouter/video",
                display_name="Video",
                capabilities=frozenset(
                    {
                        ProviderCapability.CHAT,
                        ProviderCapability.IMAGE_GENERATION,
                        ProviderCapability.TEXT_TO_VIDEO,
                        ProviderCapability.IMAGE_TO_VIDEO,
                        ProviderCapability.IMAGE_PLUS_TEXT_TO_VIDEO,
                    }
                ),
                context_window=32_768,
            )
        ]
    )
    service = SettingsService(
        repositories=repositories,
        providers={"openrouter": provider},
        secret_store=InMemorySecretStore(),
    )

    asyncio.run(service.refresh_provider_models("openrouter"))

    models = repositories.list_provider_models("openrouter")
    assert len(models) == 1
    assert models[0].capabilities == [
        "chat",
        "image_generation",
        "image_plus_text_to_video",
        "image_to_video",
        "text_to_video",
    ]


def test_refresh_provider_models_stores_validation_failure_without_listing_models(
    repositories: PersistenceRepositories,
) -> None:
    provider = RecordingProvider(
        status=ProviderConfigStatus(
            provider="openrouter",
            configured=True,
            authenticated=False,
            error="Invalid API key",
        )
    )
    service = SettingsService(
        repositories=repositories,
        providers={"openrouter": provider},
        secret_store=InMemorySecretStore(),
    )

    result = asyncio.run(service.refresh_provider_models("openrouter"))

    assert provider.calls == ["validate_config"]
    assert result.provider == "openrouter"
    assert result.configured is True
    assert result.authenticated is False
    assert result.model_count == 0
    assert result.error == "Invalid API key"
    config = repositories.get_provider_config("openrouter")
    assert config is not None
    assert config.enabled is True
    assert config.has_api_key is False
    assert config.last_error == "Invalid API key"
    assert repositories.list_provider_models("openrouter") == []


@pytest.mark.parametrize(
    ("failure_stage", "expected_calls"),
    [
        ("validate_config", ["validate_config"]),
        ("list_models", ["validate_config", "list_models"]),
    ],
)
def test_refresh_provider_models_records_network_error_for_provider_exceptions(
    repositories: PersistenceRepositories,
    failure_stage: str,
    expected_calls: list[str],
) -> None:
    provider = RecordingProvider(
        validate_exception=(
            TimeoutError("validate timed out")
            if failure_stage == "validate_config"
            else None
        ),
        list_models_exception=(
            TimeoutError("model listing timed out")
            if failure_stage == "list_models"
            else None
        ),
    )
    service = SettingsService(
        repositories=repositories,
        providers={"openrouter": provider},
        secret_store=InMemorySecretStore(),
    )

    result = asyncio.run(service.refresh_provider_models("openrouter"))

    assert provider.calls == expected_calls
    assert result.provider == "openrouter"
    assert result.configured is True
    assert result.authenticated is False
    assert result.model_count == 0
    assert result.error == ProviderErrorCategory.NETWORK_ERROR.value
    config = repositories.get_provider_config("openrouter")
    assert config is not None
    assert config.last_error == ProviderErrorCategory.NETWORK_ERROR.value


@pytest.mark.parametrize("failure_method", ["has_api_key", "get_api_key"])
def test_refresh_provider_models_records_secret_storage_errors(
    repositories: PersistenceRepositories,
    failure_method: str,
) -> None:
    secret_store = SecretStorageFailureStore(
        fail_has_api_key=failure_method == "has_api_key",
        fail_get_api_key=failure_method == "get_api_key",
    )
    secret_store.set_api_key("openrouter", "sk-openrouter-test-secret")
    provider = RecordingProvider()
    service = SettingsService(
        repositories=repositories,
        providers={"openrouter": provider},
        secret_store=secret_store,
    )

    result = asyncio.run(service.refresh_provider_models("openrouter"))

    assert result.provider == "openrouter"
    assert result.configured is True
    assert result.authenticated is False
    assert result.model_count == 0
    assert result.error == ProviderErrorCategory.SECRET_STORAGE_ERROR.value
    assert result.error != ProviderErrorCategory.PROVIDER_NOT_CONFIGURED.value
    assert provider.calls == []
    config = repositories.get_provider_config("openrouter")
    assert config is not None
    assert config.enabled is True
    assert config.has_api_key is True
    assert config.last_error == ProviderErrorCategory.SECRET_STORAGE_ERROR.value
    assert repositories.list_provider_models("openrouter") == []


def test_successful_refresh_marks_missing_models_unavailable_and_keeps_preferences(
    repositories: PersistenceRepositories,
) -> None:
    missing_after_second_refresh = ProviderModel(
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
        display_name="Claude 3.5 Sonnet",
        capabilities=frozenset({ProviderCapability.CHAT}),
        context_window=200_000,
    )
    returned_after_second_refresh = ProviderModel(
        provider="openrouter",
        model_id="google/gemini-2.5-flash-image-preview",
        display_name="Gemini 2.5 Flash Image",
        capabilities=frozenset({ProviderCapability.IMAGE_GENERATION}),
        context_window=32_768,
    )
    provider = RecordingProvider(
        models=[missing_after_second_refresh, returned_after_second_refresh]
    )
    service = SettingsService(
        repositories=repositories,
        providers={"openrouter": provider},
        secret_store=InMemorySecretStore(),
    )

    asyncio.run(service.refresh_provider_models("openrouter"))
    service.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider.models = [returned_after_second_refresh]
    asyncio.run(service.refresh_provider_models("openrouter"))

    models_by_id = {
        model.model_id: model
        for model in repositories.list_provider_models("openrouter")
    }
    chat_preference = service.get_model_preference("chat")

    assert models_by_id["anthropic/claude-3.5-sonnet"].available is False
    assert models_by_id["google/gemini-2.5-flash-image-preview"].available is True
    assert chat_preference is not None
    assert chat_preference.provider == "openrouter"
    assert chat_preference.model_id == "anthropic/claude-3.5-sonnet"


@pytest.mark.parametrize(
    "failure_stage",
    ["save_provider_model", "mark_missing_models", "final_provider_config"],
)
def test_refresh_provider_models_rolls_back_model_changes_on_persistence_failure(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    repositories.upsert_provider_config(
        provider="openrouter",
        enabled=True,
        has_api_key=True,
        last_model_refresh_at="2026-05-12T18:30:00+00:00",
        last_error=None,
    )
    repositories.save_provider_model(
        provider="openrouter",
        model_id="openrouter/existing-chat",
        display_name="Existing Chat",
        capabilities=["chat"],
        context_window=8192,
    )
    provider = RecordingProvider(
        models=[
            ProviderModel(
                provider="openrouter",
                model_id="openrouter/new-chat",
                display_name="New Chat",
                capabilities=frozenset({ProviderCapability.CHAT}),
                context_window=200_000,
            ),
            ProviderModel(
                provider="openrouter",
                model_id="openrouter/new-image",
                display_name="New Image",
                capabilities=frozenset({ProviderCapability.IMAGE_GENERATION}),
                context_window=32_768,
            ),
        ]
    )
    secret_store = InMemorySecretStore()
    secret_store.set_api_key("openrouter", "sk-openrouter-test-secret")
    service = SettingsService(
        repositories=repositories,
        providers={"openrouter": provider},
        secret_store=secret_store,
    )

    if failure_stage == "save_provider_model":
        original_save_provider_model = repositories.save_provider_model

        def fail_on_second_model(**kwargs: Any) -> object:
            if kwargs["model_id"] == "openrouter/new-image":
                raise sqlite3.OperationalError("failed to persist model")
            return original_save_provider_model(**kwargs)

        monkeypatch.setattr(
            repositories,
            "save_provider_model",
            fail_on_second_model,
        )
    elif failure_stage == "mark_missing_models":

        def fail_mark_missing_provider_models_unavailable(**_kwargs: object) -> None:
            raise sqlite3.OperationalError("failed to mark missing models")

        monkeypatch.setattr(
            repositories,
            "mark_missing_provider_models_unavailable",
            fail_mark_missing_provider_models_unavailable,
        )
    else:
        original_upsert_provider_config = repositories.upsert_provider_config

        def fail_final_provider_config(**kwargs: Any) -> object:
            if kwargs.get("last_model_refresh_at") is not None:
                raise sqlite3.OperationalError("failed to record refresh")
            return original_upsert_provider_config(**kwargs)

        monkeypatch.setattr(
            repositories,
            "upsert_provider_config",
            fail_final_provider_config,
        )

    result = asyncio.run(service.refresh_provider_models("openrouter"))

    assert provider.calls == ["validate_config", "list_models"]
    assert result.error
    config = repositories.get_provider_config("openrouter")
    assert config is not None
    assert config.last_error
    models_by_id = {
        model.model_id: model
        for model in repositories.list_provider_models("openrouter")
    }
    assert set(models_by_id) == {"openrouter/existing-chat"}
    assert models_by_id["openrouter/existing-chat"].available is True


def test_settings_service_persists_summarization_visibility_preference(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        service = _settings_service(PersistenceRepositories(connection))

        assert (
            service.get_local_setting("show_summarization_activity", default=False)
            is False
        )
        service.set_provider_api_key("openrouter", API_KEY_SENTINEL)
        service.set_local_setting("show_summarization_activity", True)

    with sqlite3.connect(database_path) as connection:
        service = _settings_service(PersistenceRepositories(connection))

        assert (
            service.get_local_setting("show_summarization_activity", default=False)
            is True
        )
        assert API_KEY_SENTINEL not in "\n".join(connection.iterdump())


def test_settings_service_sanitizes_proactive_character_text_random_controls(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        service = _settings_service(PersistenceRepositories(connection))

        assert service.get_local_setting(
            CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING,
        ) == DEFAULT_CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_PERCENT
        assert service.get_local_setting(
            CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING,
        ) == DEFAULT_CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_TURNS
        service.set_local_setting(
            CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING,
            250,
        )
        service.set_local_setting(
            CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING,
            -3,
        )

    with sqlite3.connect(database_path) as connection:
        service = _settings_service(PersistenceRepositories(connection))

        assert service.get_local_setting(
            CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING,
        ) == MAX_CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_PERCENT
        assert service.get_local_setting(
            CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING,
        ) == 0
        service.set_local_setting(
            CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING,
            "invalid",
        )
        service.set_local_setting(
            CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING,
            999,
        )
        assert service.get_local_setting(
            CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING,
        ) == DEFAULT_CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_PERCENT
        assert service.get_local_setting(
            CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING,
        ) == MAX_CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_TURNS


def test_settings_service_persists_chat_fallback_preference(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        service = _settings_service(PersistenceRepositories(connection))

        assert (
            service.get_local_setting(
                "chat_fallback_enabled",
                default=False,
            )
            is False
        )
        service.set_provider_api_key("openrouter", API_KEY_SENTINEL)
        service.set_local_setting("chat_fallback_enabled", True)

    with sqlite3.connect(database_path) as connection:
        service = _settings_service(PersistenceRepositories(connection))

        assert (
            service.get_local_setting(
                "chat_fallback_enabled",
                default=False,
            )
            is True
        )
        assert API_KEY_SENTINEL not in "\n".join(connection.iterdump())


@pytest.mark.parametrize(
    "task",
    [
        "structured_output_fallback",
        "full_roleplay_structured_output_fallback",
        "fantasy_roleplay_structured_output_fallback",
        "science_fiction_roleplay_structured_output_fallback",
        "first_contact_exploration_structured_output_fallback",
        "survival_expedition_structured_output_fallback",
        "time_loop_structured_output_fallback",
        "investigation_mystery_structured_output_fallback",
        "political_intrigue_structured_output_fallback",
        "dating_sim_structured_output_fallback",
    ],
)
def test_settings_service_auto_enables_structured_output_fallback_when_model_is_set(
    tmp_path: Path,
    task: str,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        service = _settings_service(PersistenceRepositories(connection))

        assert (
            service.get_local_setting(
                "structured_output_fallback_enabled",
                default=False,
            )
            is False
        )
        service.set_model_preference(
            task=task,
            provider="openrouter",
            model_id="openrouter/structured-fallback",
        )

    with sqlite3.connect(database_path) as connection:
        service = _settings_service(PersistenceRepositories(connection))
        preference = service.get_model_preference(task)

        assert preference is not None
        assert preference.provider == "openrouter"
        assert preference.model_id == "openrouter/structured-fallback"
        assert (
            service.get_local_setting(
                "structured_output_fallback_enabled",
                default=False,
            )
            is True
        )


@pytest.mark.parametrize(
    "task",
    ["chat_character_interaction", "character_interaction_context_update"],
)
def test_settings_service_rejects_retired_model_tasks(
    tmp_path: Path,
    task: str,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        service = _settings_service(PersistenceRepositories(connection))

        with pytest.raises(ValueError, match="Model task is retired"):
            service.set_model_preference(
                task=task,
                provider="openrouter",
                model_id="openrouter/retired",
            )

        assert service.get_model_preference(task) is None


@pytest.mark.parametrize(
    "task",
    [
        "tool_call_fallback",
        "full_roleplay_tool_call_fallback",
        "fantasy_roleplay_tool_call_fallback",
        "science_fiction_roleplay_tool_call_fallback",
        "first_contact_exploration_tool_call_fallback",
        "survival_expedition_tool_call_fallback",
        "time_loop_tool_call_fallback",
        "investigation_mystery_tool_call_fallback",
        "political_intrigue_tool_call_fallback",
        "dating_sim_tool_call_fallback",
    ],
)
def test_settings_service_auto_enables_tool_call_fallback_when_model_is_set(
    tmp_path: Path,
    task: str,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        service = _settings_service(PersistenceRepositories(connection))

        assert (
            service.get_local_setting(
                "tool_call_fallback_enabled",
                default=False,
            )
            is False
        )
        service.set_model_preference(
            task=task,
            provider="openrouter",
            model_id="openrouter/tool-fallback",
        )

    with sqlite3.connect(database_path) as connection:
        service = _settings_service(PersistenceRepositories(connection))
        preference = service.get_model_preference(task)

        assert preference is not None
        assert preference.provider == "openrouter"
        assert preference.model_id == "openrouter/tool-fallback"
        assert (
            service.get_local_setting(
                "tool_call_fallback_enabled",
                default=False,
            )
            is True
        )


def test_settings_service_persists_image_generation_frequency_preference(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        service = _settings_service(PersistenceRepositories(connection))

        assert service.get_local_setting("image_generation_frequency", default=0) == 0
        service.set_provider_api_key("openrouter", API_KEY_SENTINEL)
        service.set_local_setting("image_generation_frequency", 3)

    with sqlite3.connect(database_path) as connection:
        service = _settings_service(PersistenceRepositories(connection))

        assert service.get_local_setting("image_generation_frequency", default=0) == 3
        assert API_KEY_SENTINEL not in "\n".join(connection.iterdump())


def test_settings_service_resolves_save_scoped_settings_before_global_defaults(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A beacon tower.",
        player_role="Keeper",
        content={},
    )
    first_save = repositories.create_save(
        scenario_id=scenario.id,
        title="Night Watch",
    )
    second_save = repositories.create_save(
        scenario_id=scenario.id,
        title="Signal Tower",
    )
    service = _settings_service(repositories)

    service.set_local_setting("image_generation_frequency", 2)
    service.set_local_setting(
        "image_generation_frequency",
        5,
        save_id=first_save.id,
    )

    assert (
        service.get_local_setting(
            "image_generation_frequency",
            save_id=first_save.id,
        )
        == 5
    )
    assert (
        service.get_local_setting(
            "image_generation_frequency",
            save_id=second_save.id,
        )
        == 2
    )
    assert service.get_local_setting("image_generation_frequency") == 2


def test_settings_service_sanitizes_script_guard_mode_per_save(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A beacon tower.",
        player_role="Keeper",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    service = _settings_service(repositories)

    assert service.get_local_setting(SCRIPT_GUARD_MODE_SETTING, save_id=save.id) == (
        DEFAULT_SCRIPT_GUARD_MODE
    )

    service.set_local_setting(
        SCRIPT_GUARD_MODE_SETTING,
        SCRIPT_GUARD_MODE_LATIN_ONLY_REJECT,
        save_id=save.id,
    )
    assert service.get_local_setting(SCRIPT_GUARD_MODE_SETTING, save_id=save.id) == (
        SCRIPT_GUARD_MODE_LATIN_ONLY_REJECT
    )

    service.set_local_setting(
        SCRIPT_GUARD_MODE_SETTING,
        "not-a-mode",
        save_id=save.id,
    )
    assert service.get_local_setting(SCRIPT_GUARD_MODE_SETTING, save_id=save.id) == (
        DEFAULT_SCRIPT_GUARD_MODE
    )


def test_settings_service_sanitizes_phrase_denylists(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A beacon tower.",
        player_role="Keeper",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    service = _settings_service(repositories)

    service.set_local_setting(
        GENERATED_PHRASE_DENYLIST_SETTING,
        "  That's not nothing  \nthat's NOT nothing\nthat is everything ",
    )
    service.set_local_setting(
        SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
        "  save phrase  \n\nSAVE PHRASE ",
        save_id=save.id,
    )

    assert service.get_local_setting(GENERATED_PHRASE_DENYLIST_SETTING) == (
        "That's not nothing\nthat is everything"
    )
    assert service.get_local_setting(
        SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
        save_id=save.id,
    ) == "save phrase"

    service.set_local_setting(GENERATED_PHRASE_DENYLIST_SETTING, {"not": "text"})
    assert service.get_local_setting(GENERATED_PHRASE_DENYLIST_SETTING) == ""


def test_settings_service_resolves_user_scoped_settings_before_global_defaults(
    repositories: PersistenceRepositories,
) -> None:
    first_user = repositories.create_user(
        username="Mira",
        role="user",
        password_hash="hash",
    )
    second_user = repositories.create_user(
        username="Ilyra",
        role="user",
        password_hash="hash",
    )
    service = _settings_service(repositories)

    service.set_local_setting("pending_jobs_display_mode", "expanded")
    service.set_local_setting(
        "pending_jobs_display_mode",
        "expanded_full",
        user_id=first_user.id,
    )

    assert (
        service.get_local_setting(
            "pending_jobs_display_mode",
            user_id=first_user.id,
        )
        == "expanded_full"
    )
    assert (
        service.get_local_setting(
            "pending_jobs_display_mode",
            user_id=second_user.id,
        )
        == "expanded"
    )


def test_settings_service_enforces_child_content_rating_self_service_limits(
    repositories: PersistenceRepositories,
) -> None:
    child = repositories.create_user(
        username="Ilyra",
        role="child",
        password_hash="hash",
    )
    service = _settings_service(repositories)

    service.set_local_setting(
        CONTENT_FILTER_RATING_SETTING,
        "g",
        user_id=child.id,
    )

    assert service.get_local_setting(
        CONTENT_FILTER_RATING_SETTING,
        user_id=child.id,
    ) == "g"
    with pytest.raises(ValueError, match="only G or PG"):
        service.set_local_setting(
            CONTENT_FILTER_RATING_SETTING,
            "pg-13",
            user_id=child.id,
        )


def test_settings_service_sanitizes_adult_content_safety_preferences(
    repositories: PersistenceRepositories,
) -> None:
    adult = repositories.create_user(
        username="Mira",
        role="user",
        password_hash="hash",
    )
    service = _settings_service(repositories)

    service.set_local_setting(
        CONTENT_FILTER_RATING_SETTING,
        " R ",
        user_id=adult.id,
    )
    service.set_local_setting(
        FADE_TO_BLACK_ENABLED_SETTING,
        False,
        user_id=adult.id,
    )

    assert service.get_local_setting(
        CONTENT_FILTER_RATING_SETTING,
        user_id=adult.id,
    ) == "r"
    assert service.get_local_setting(
        FADE_TO_BLACK_ENABLED_SETTING,
        user_id=adult.id,
    ) is False
    with pytest.raises(ValueError, match="boolean"):
        service.set_local_setting(
            FADE_TO_BLACK_ENABLED_SETTING,
            "false",
            user_id=adult.id,
        )


def test_settings_service_sanitizes_image_style_preset_preference(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        repositories = PersistenceRepositories(connection)
        scenario = repositories.create_scenario(
            type="full_roleplay",
            title="Lantern Keep",
            premise="A beacon tower.",
            player_role="Keeper",
            content={},
        )
        first_save = repositories.create_save(
            scenario_id=scenario.id,
            title="Night Watch",
        )
        second_save = repositories.create_save(
            scenario_id=scenario.id,
            title="Signal Tower",
        )
        service = _settings_service(repositories)
        repositories.set_app_setting(IMAGE_STYLE_PRESET_SETTING, "comic_book")

        assert service.get_local_setting(IMAGE_STYLE_PRESET_SETTING) == "none"

        assert (
            service.get_local_setting(
                IMAGE_STYLE_PRESET_SETTING,
                save_id=first_save.id,
            )
            == "none"
        )
        service.set_local_setting(
            IMAGE_STYLE_PRESET_SETTING,
            "  Anime  ",
            save_id=first_save.id,
        )
        service.set_local_setting(
            IMAGE_STYLE_PRESET_SETTING,
            "low-poly",
            save_id=second_save.id,
        )
        assert (
            service.get_local_setting(
                IMAGE_STYLE_PRESET_SETTING,
                save_id=first_save.id,
            )
            == "anime"
        )
        assert (
            service.get_local_setting(
                IMAGE_STYLE_PRESET_SETTING,
                save_id=second_save.id,
            )
            == "low_poly"
        )

        service.set_local_setting(
            IMAGE_STYLE_PRESET_SETTING,
            "oil painting",
            save_id=first_save.id,
        )
        assert (
            service.get_local_setting(
                IMAGE_STYLE_PRESET_SETTING,
                save_id=first_save.id,
            )
            == "none"
        )
        assert (
            repositories.get_app_setting(
                save_image_style_preset_setting_key(first_save.id)
            )
            == "none"
        )


def test_settings_service_sanitizes_pending_jobs_display_mode_preference(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        service = _settings_service(PersistenceRepositories(connection))

        assert service.get_local_setting("pending_jobs_display_mode") == "compact"
        service.set_local_setting("pending_jobs_display_mode", " expanded ")
        assert service.get_local_setting("pending_jobs_display_mode") == "expanded"
        service.set_local_setting("pending_jobs_display_mode", "expanded_full")
        assert service.get_local_setting("pending_jobs_display_mode") == "expanded_full"
        service.set_local_setting("pending_jobs_display_mode", "all")
        assert service.get_local_setting("pending_jobs_display_mode") == "compact"

    with sqlite3.connect(database_path) as connection:
        service = _settings_service(PersistenceRepositories(connection))

        assert service.get_local_setting("pending_jobs_display_mode") == "compact"


def test_settings_service_sanitizes_save_scoped_post_turn_inference_mode(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        repositories = PersistenceRepositories(connection)
        scenario = repositories.create_scenario(
            type="full_roleplay",
            title="Ashfall Keep",
            premise="A border keep is cut off by ash storms.",
            player_role="Signal warden",
            content={},
        )
        save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
        service = _settings_service(repositories)

        assert (
            service.get_local_setting(
                POST_TURN_INFERENCE_MODE_SETTING,
                save_id=save.id,
            )
            == "plan_owned"
        )
        service.set_local_setting(
            POST_TURN_INFERENCE_MODE_SETTING,
            " hybrid ",
            save_id=save.id,
        )
        assert (
            service.get_local_setting(
                POST_TURN_INFERENCE_MODE_SETTING,
                save_id=save.id,
            )
            == "hybrid"
        )
        service.set_local_setting(
            POST_TURN_INFERENCE_MODE_SETTING,
            "plan-owned",
            save_id=save.id,
        )
        assert (
            service.get_local_setting(
                POST_TURN_INFERENCE_MODE_SETTING,
                save_id=save.id,
            )
            == "plan_owned"
        )


def test_settings_service_sanitizes_user_narration_guidance(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        repositories = PersistenceRepositories(connection)
        user = repositories.create_user(
            username="Mira",
            role="user",
            password_hash="hash",
        )
        service = _settings_service(repositories)

        assert (
            service.get_local_setting("user_narration_guidance", user_id=user.id)
            == ""
        )
        service.set_local_setting(
            "user_narration_guidance",
            "  Keep narrator responses to two paragraphs or less.  ",
            user_id=user.id,
        )
        assert (
            service.get_local_setting("user_narration_guidance", user_id=user.id)
            == "Keep narrator responses to two paragraphs or less."
        )
        service.set_local_setting(
            "user_narration_guidance",
            {"not": "text"},
            user_id=user.id,
        )
        assert (
            service.get_local_setting("user_narration_guidance", user_id=user.id)
            == ""
        )


def test_settings_service_sanitizes_openrouter_reasoning_overrides(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        service = _settings_service(PersistenceRepositories(connection))

        assert service.get_local_setting(
            OPENROUTER_CHAT_REASONING_OVERRIDES_SETTING
        ) == {}
        service.set_local_setting(
            OPENROUTER_CHAT_REASONING_OVERRIDES_SETTING,
            {
                "z-ai/glm-4.7": "disabled",
                "openai/gpt-5-mini": {"effort": "minimal", "exclude": True},
                "bad key": "disabled",
                "meta/llama": {"enabled": "nope"},
            },
        )

    with sqlite3.connect(database_path) as connection:
        service = _settings_service(PersistenceRepositories(connection))

        assert service.get_local_setting(
            OPENROUTER_CHAT_REASONING_OVERRIDES_SETTING
        ) == {
            "z-ai/glm-4.7": {"enabled": False, "exclude": True},
            "openai/gpt-5-mini": {"effort": "minimal", "exclude": True},
        }


def test_settings_service_persists_automation_control_preferences(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        service = _settings_service(PersistenceRepositories(connection))

        assert (
            service.get_local_setting(
                "automatic_summarization_enabled",
                default=True,
            )
            is True
        )
        assert (
            service.get_local_setting(
                "automatic_image_generation_enabled",
                default=True,
            )
            is True
        )
        assert (
            service.get_local_setting(
                AGENTIC_CONTEXT_PIPELINE_SETTING,
                default=False,
            )
            is False
        )
        assert (
            service.get_local_setting(
                "summarization_context_pressure_threshold",
                default=0.75,
            )
            == 0.75
        )
        service.set_provider_api_key("openrouter", API_KEY_SENTINEL)
        service.set_local_setting("automatic_summarization_enabled", False)
        service.set_local_setting("automatic_image_generation_enabled", False)
        service.set_local_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
        service.set_local_setting("summarization_context_pressure_threshold", 0.45)

    with sqlite3.connect(database_path) as connection:
        service = _settings_service(PersistenceRepositories(connection))

        assert (
            service.get_local_setting(
                "automatic_summarization_enabled",
                default=True,
            )
            is False
        )
        assert (
            service.get_local_setting(
                "automatic_image_generation_enabled",
                default=True,
            )
            is False
        )
        assert (
            service.get_local_setting(
                AGENTIC_CONTEXT_PIPELINE_SETTING,
                default=False,
            )
            is True
        )
        assert (
            service.get_local_setting(
                "summarization_context_pressure_threshold",
                default=0.75,
            )
            == 0.45
        )
        assert API_KEY_SENTINEL not in "\n".join(connection.iterdump())


def test_settings_service_applies_debug_logging_setting_immediately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bragi.app_logging import configure_logging

    monkeypatch.delenv("BRAGI_LOG_LEVEL", raising=False)
    paths = _storage_paths(tmp_path)
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    bragi_logger = logging.getLogger("bragi")
    original_handlers = tuple(bragi_logger.handlers)
    original_level = bragi_logger.level
    original_propagate = bragi_logger.propagate

    try:
        configure_logging(paths)
        handlers = [
            handler
            for handler in bragi_logger.handlers
            if isinstance(handler, RotatingFileHandler)
            and Path(handler.baseFilename) == paths.state_dir / "logs" / "bragi.log"
        ]
        assert handlers
        assert bragi_logger.level == logging.INFO
        assert {handler.level for handler in handlers} == {logging.INFO}

        with sqlite3.connect(database_path) as connection:
            service = _settings_service(PersistenceRepositories(connection))
            service.set_local_setting("debug_logging_enabled", True)
            assert bragi_logger.level == logging.DEBUG
            assert {handler.level for handler in handlers} == {logging.DEBUG}

            service.set_local_setting("debug_logging_enabled", False)
            assert bragi_logger.level == logging.INFO
            assert {handler.level for handler in handlers} == {logging.INFO}
    finally:
        _restore_logger(
            bragi_logger,
            original_handlers,
            original_level=original_level,
            original_propagate=original_propagate,
        )


def test_settings_service_omits_default_fallback_secret_storage_diagnostic(
    repositories: PersistenceRepositories,
) -> None:
    class PlaintextSecretStore(InMemorySecretStore):
        uses_fallback_storage = True
        fallback_path = Path("/tmp/bragi/state/api_keys.json")

    service = SettingsService(
        repositories=repositories,
        providers={},
        secret_store=PlaintextSecretStore(),
    )

    assert service.secret_storage_warning() is None


def test_context_budget_settings_use_defaults_and_round_trip(
    repositories: PersistenceRepositories,
) -> None:
    from bragi.services.settings_service import ContextBudgetSettings

    service = _settings_service(repositories)

    defaults = service.get_context_budget_settings()

    assert defaults == ContextBudgetSettings.defaults()
    assert defaults.mode == "diagnostics_only"
    assert defaults.fixed_total_chars > 0
    assert 0 < defaults.adaptive_fraction <= 1

    custom = ContextBudgetSettings(
        mode="fixed_chars",
        fixed_total_chars=4_800,
        adaptive_fraction=0.25,
    )
    service.set_context_budget_settings(custom)

    assert service.get_context_budget_settings() == custom
    assert repositories.get_app_setting("context_budget_mode") == "fixed_chars"
    assert repositories.get_app_setting("context_budget_fixed_total_chars") == 4_800
    assert repositories.get_app_setting("context_budget_adaptive_fraction") == 0.25


def test_chat_history_window_settings_use_defaults_and_round_trip(
    repositories: PersistenceRepositories,
) -> None:
    service = _settings_service(repositories)

    defaults = service.get_chat_history_window_settings()

    assert defaults == ChatHistoryWindowSettings.defaults()
    assert defaults.player_messages == DEFAULT_RECENT_PLAYER_MESSAGE_WINDOW
    assert defaults.narrator_messages == DEFAULT_RECENT_NARRATOR_MESSAGE_WINDOW

    custom = ChatHistoryWindowSettings(player_messages=2, narrator_messages=1)
    service.set_chat_history_window_settings(custom)

    assert service.get_chat_history_window_settings() == custom
    assert repositories.get_app_setting(RECENT_PLAYER_MESSAGE_WINDOW_SETTING) == 2
    assert repositories.get_app_setting(RECENT_NARRATOR_MESSAGE_WINDOW_SETTING) == 1


def test_chat_history_window_settings_persist_across_reopen(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        service = _settings_service(PersistenceRepositories(connection))
        service.set_chat_history_window_settings(
            ChatHistoryWindowSettings(player_messages=3, narrator_messages=4)
        )

    with sqlite3.connect(database_path) as connection:
        service = _settings_service(PersistenceRepositories(connection))

        assert service.get_chat_history_window_settings() == ChatHistoryWindowSettings(
            player_messages=3,
            narrator_messages=4,
        )


def test_chat_history_window_settings_sanitize_invalid_and_out_of_range_values(
    repositories: PersistenceRepositories,
) -> None:
    service = _settings_service(repositories)

    repositories.set_app_setting(RECENT_PLAYER_MESSAGE_WINDOW_SETTING, "many")
    repositories.set_app_setting(RECENT_NARRATOR_MESSAGE_WINDOW_SETTING, True)

    assert service.get_chat_history_window_settings() == ChatHistoryWindowSettings(
        player_messages=DEFAULT_RECENT_PLAYER_MESSAGE_WINDOW,
        narrator_messages=DEFAULT_RECENT_NARRATOR_MESSAGE_WINDOW,
    )

    service.set_chat_history_window_settings(
        ChatHistoryWindowSettings(player_messages=-1, narrator_messages=999)
    )

    assert service.get_chat_history_window_settings() == ChatHistoryWindowSettings(
        player_messages=0,
        narrator_messages=24,
    )
    assert service.get_local_setting(RECENT_PLAYER_MESSAGE_WINDOW_SETTING) == 0
    assert service.get_local_setting(RECENT_NARRATOR_MESSAGE_WINDOW_SETTING) == 24


def test_narrator_planner_chat_history_window_settings_defaults_and_round_trip(
    repositories: PersistenceRepositories,
) -> None:
    service = _settings_service(repositories)
    service.set_chat_history_window_settings(
        ChatHistoryWindowSettings(player_messages=4, narrator_messages=3)
    )

    assert service.get_narrator_planner_chat_history_window_settings() == (
        ChatHistoryWindowSettings(
            player_messages=DEFAULT_NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW,
            narrator_messages=DEFAULT_NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW,
        )
    )

    custom = ChatHistoryWindowSettings(player_messages=9, narrator_messages=2)
    service.set_narrator_planner_chat_history_window_settings(custom)

    assert service.get_narrator_planner_chat_history_window_settings() == custom
    assert repositories.get_app_setting(
        NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW_SETTING
    ) == 9
    assert repositories.get_app_setting(
        NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW_SETTING
    ) == 2
    assert service.get_chat_history_window_settings() == ChatHistoryWindowSettings(
        player_messages=4,
        narrator_messages=3,
    )


def test_narrator_planner_chat_history_window_settings_sanitize_values(
    repositories: PersistenceRepositories,
) -> None:
    service = _settings_service(repositories)
    service.set_chat_history_window_settings(
        ChatHistoryWindowSettings(player_messages=5, narrator_messages=4)
    )
    repositories.set_app_setting(
        NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
        "many",
    )
    repositories.set_app_setting(
        NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
        True,
    )

    assert service.get_narrator_planner_chat_history_window_settings() == (
        ChatHistoryWindowSettings(
            player_messages=DEFAULT_NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW,
            narrator_messages=DEFAULT_NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW,
        )
    )

    service.set_narrator_planner_chat_history_window_settings(
        ChatHistoryWindowSettings(player_messages=-1, narrator_messages=999)
    )

    assert service.get_narrator_planner_chat_history_window_settings() == (
        ChatHistoryWindowSettings(player_messages=0, narrator_messages=24)
    )
    assert service.get_local_setting(
        NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW_SETTING
    ) == 0
    assert service.get_local_setting(
        NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW_SETTING
    ) == 24


def _settings_service(repositories: PersistenceRepositories) -> SettingsService:
    return SettingsService(
        repositories=repositories,
        providers={},
        secret_store=InMemorySecretStore(),
    )


def _storage_paths(tmp_path: Path) -> StoragePaths:
    data_dir = tmp_path / "data"
    return StoragePaths(
        data_dir=data_dir,
        database_path=data_dir / "bragi.sqlite3",
        media_dir=data_dir / "media",
        state_dir=tmp_path / "state",
        cache_dir=tmp_path / "cache",
    )


def _restore_logger(
    logger: logging.Logger,
    original_handlers: tuple[logging.Handler, ...],
    *,
    original_level: int,
    original_propagate: bool,
) -> None:
    for handler in logger.handlers:
        if handler not in original_handlers:
            handler.close()
    logger.handlers[:] = list(original_handlers)
    logger.setLevel(original_level)
    logger.propagate = original_propagate
