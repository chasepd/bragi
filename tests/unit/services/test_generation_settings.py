from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import ChatMessage, ChatRequest, StructuredOutputRequest
from bragi.services.generation_settings import (
    CHAT_MAX_OUTPUT_TOKENS_ENABLED_SETTING,
    CHAT_MAX_OUTPUT_TOKENS_SETTING,
    CHAT_TEMPERATURE_ENABLED_SETTING,
    CHAT_TEMPERATURE_SETTING,
    DEFAULT_IMAGE_DIMENSION_PRESET,
    IMAGE_DIMENSION_PRESET_SETTING,
    MODEL_THINKING_PREFERENCES_SETTING,
    OPENROUTER_CHAT_REASONING_OVERRIDES_SETTING,
    THINKING_LEVEL_OFF,
    chat_generation_settings,
    image_generation_dimensions,
    model_thinking_preference_level,
    model_thinking_reasoning_config,
    openrouter_chat_reasoning_config,
    request_with_model_thinking_preference,
    sanitize_chat_max_output_tokens,
    sanitize_chat_temperature,
    sanitize_image_dimension_preset,
    sanitize_model_thinking_preferences,
    sanitize_openrouter_chat_reasoning_overrides,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_chat_generation_settings_require_enabled_supported_parameters(
    repositories: PersistenceRepositories,
) -> None:
    repositories.save_provider_model(
        provider="openrouter",
        model_id="openrouter/chat",
        display_name="OpenRouter Chat",
        capabilities=["chat"],
        supported_parameters=["temperature", "max_output_tokens"],
    )
    repositories.set_app_setting(CHAT_TEMPERATURE_ENABLED_SETTING, True)
    repositories.set_app_setting(CHAT_TEMPERATURE_SETTING, 1.35)
    repositories.set_app_setting(CHAT_MAX_OUTPUT_TOKENS_ENABLED_SETTING, True)
    repositories.set_app_setting(CHAT_MAX_OUTPUT_TOKENS_SETTING, 2048)

    settings = chat_generation_settings(
        repositories,
        provider="openrouter",
        model_id="openrouter/chat",
    )
    unsupported = chat_generation_settings(
        repositories,
        provider="openrouter",
        model_id="missing",
    )

    assert settings.temperature == 1.35
    assert settings.max_output_tokens == 2048
    assert unsupported.temperature is None
    assert unsupported.max_output_tokens is None


def test_chat_generation_settings_sanitize_ranges() -> None:
    assert sanitize_chat_temperature(-1.0) == 0.0
    assert sanitize_chat_temperature(99.0) == 2.0
    assert sanitize_chat_temperature("nope") == 0.7
    assert sanitize_chat_max_output_tokens(12) == 64
    assert sanitize_chat_max_output_tokens(99_999) == 8192
    assert sanitize_chat_max_output_tokens(False) == 2048


def test_image_generation_dimensions_require_supported_model(
    repositories: PersistenceRepositories,
) -> None:
    repositories.save_provider_model(
        provider="venice",
        model_id="venice/image",
        display_name="Venice Image",
        capabilities=["image_generation"],
        supported_parameters=["image_dimensions"],
    )
    repositories.set_app_setting(IMAGE_DIMENSION_PRESET_SETTING, "portrait_768x1024")

    dimensions = image_generation_dimensions(
        repositories,
        provider="venice",
        model_id="venice/image",
    )
    unsupported = image_generation_dimensions(
        repositories,
        provider="venice",
        model_id="missing",
    )

    assert dimensions == (768, 1024)
    assert unsupported is None


def test_image_generation_dimensions_default_to_square(
    repositories: PersistenceRepositories,
) -> None:
    repositories.save_provider_model(
        provider="venice",
        model_id="venice/image",
        display_name="Venice Image",
        capabilities=["image_generation"],
        supported_parameters=["image_dimensions"],
    )

    dimensions = image_generation_dimensions(
        repositories,
        provider="venice",
        model_id="venice/image",
    )

    assert DEFAULT_IMAGE_DIMENSION_PRESET == "square_1024x1024"
    assert dimensions == (1024, 1024)


def test_sanitize_image_dimension_preset_rejects_unknown_values() -> None:
    assert sanitize_image_dimension_preset("wide-1024x576") == "wide_1024x576"
    assert sanitize_image_dimension_preset("nonsense") == "square_1024x1024"
    assert sanitize_image_dimension_preset(True) == "square_1024x1024"


def test_openrouter_reasoning_overrides_sanitize_to_safe_configs() -> None:
    assert sanitize_openrouter_chat_reasoning_overrides(
        {
            "z-ai/glm-4.7": "disabled",
            "openai/gpt-5-mini": {"effort": "low", "exclude": False},
            "anthropic/claude": {"max_tokens": 512, "exclude": True},
            "bad key": "disabled",
            "meta/llama": {"effort": "wat"},
            "google/gemini": {"max_tokens": False},
            "qwen/qwen": {"enabled": "nope"},
        }
    ) == {
        "z-ai/glm-4.7": {
            "enabled": False,
            "exclude": True,
        },
        "openai/gpt-5-mini": {
            "effort": "low",
            "exclude": False,
        },
        "anthropic/claude": {
            "max_tokens": 512,
            "exclude": True,
        },
    }


def test_model_thinking_preferences_sanitize_to_task_configs() -> None:
    assert sanitize_model_thinking_preferences(
        {
            "chat": {
                "provider": "OpenRouter",
                "model_id": "openai/gpt-5-mini",
                "level": "High",
            },
            "structured": {
                "provider": "venice",
                "model_id": "venice-reasoning",
                "level": "off",
            },
            "bad task": {
                "provider": "openrouter",
                "model_id": "openai/gpt-5-mini",
                "level": "low",
            },
            "bad_level": {
                "provider": "openrouter",
                "model_id": "openai/gpt-5-mini",
                "level": "ludicrous",
            },
            "bad_provider": {
                "provider": "",
                "model_id": "openai/gpt-5-mini",
                "level": "low",
            },
        }
    ) == {
        "chat": {
            "provider": "openrouter",
            "model_id": "openai/gpt-5-mini",
            "level": "high",
        },
        "structured": {
            "provider": "venice",
            "model_id": "venice-reasoning",
            "level": THINKING_LEVEL_OFF,
        },
    }


def test_model_thinking_preferences_drop_retired_character_interaction_tasks() -> None:
    assert sanitize_model_thinking_preferences(
        {
            "chat_character_interaction": {
                "provider": "openrouter",
                "model_id": "openai/gpt-5-mini",
                "level": "high",
            },
            "character_interaction_context_update": {
                "provider": "openrouter",
                "model_id": "openai/gpt-5-mini",
                "level": "low",
            },
            "character_image_description": {
                "provider": "openrouter",
                "model_id": "openai/gpt-5-mini",
                "level": "medium",
            },
            "dating_sim_context_update": {
                "provider": "openrouter",
                "model_id": "openai/gpt-5-mini",
                "level": "low",
            },
        }
    ) == {
        "character_image_description": {
            "provider": "openrouter",
            "model_id": "openai/gpt-5-mini",
            "level": "medium",
        },
        "dating_sim_context_update": {
            "provider": "openrouter",
            "model_id": "openai/gpt-5-mini",
            "level": "low",
        },
    }


def test_model_thinking_reasoning_config_requires_matching_supported_model(
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
    repositories.set_app_setting(
        MODEL_THINKING_PREFERENCES_SETTING,
        {
            "chat": {
                "provider": "openrouter",
                "model_id": "openai/gpt-5-mini",
                "level": "high",
            },
        },
    )

    config = model_thinking_reasoning_config(
        repositories,
        task="chat",
        provider="openrouter",
        model_id="openai/gpt-5-mini",
    )

    assert config is not None
    assert config.effort == "high"
    assert config.exclude is True
    assert (
        model_thinking_reasoning_config(
            repositories,
            task="chat",
            provider="openrouter",
            model_id="missing",
        )
        is None
    )
    assert (
        model_thinking_preference_level(
            repositories,
            task="chat",
            provider="openrouter",
            model_id="missing",
        )
        == "provider_default"
    )


def test_request_with_model_thinking_preference_applies_supported_config(
    repositories: PersistenceRepositories,
) -> None:
    repositories.save_provider_model(
        provider="venice",
        model_id="venice-reasoning",
        display_name="Venice Reasoning",
        capabilities=["chat"],
        thinking={"levels": ["high", "medium"], "mandatory": False},
    )
    repositories.set_app_setting(
        MODEL_THINKING_PREFERENCES_SETTING,
        {
            "chat": {
                "provider": "venice",
                "model_id": "venice-reasoning",
                "level": "medium",
            },
        },
    )
    request = ChatRequest(
        provider="venice",
        model_id="venice-reasoning",
        messages=(ChatMessage(role="player", body="Hello"),),
    )

    updated = request_with_model_thinking_preference(
        repositories,
        request,
        task="chat",
    )

    assert updated.reasoning is not None
    assert updated.reasoning.effort == "medium"
    assert updated.reasoning.exclude is True
    assert updated is not request


def test_openrouter_reasoning_config_reads_model_specific_setting(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_app_setting(
        OPENROUTER_CHAT_REASONING_OVERRIDES_SETTING,
        {
            "z-ai/glm-4.7": "disabled",
            "openai/gpt-5-mini": {"effort": "minimal", "exclude": True},
        },
    )

    disabled = openrouter_chat_reasoning_config(
        repositories,
        provider="openrouter",
        model_id="z-ai/glm-4.7",
    )
    minimal = openrouter_chat_reasoning_config(
        repositories,
        provider="openrouter",
        model_id="openai/gpt-5-mini",
    )

    assert disabled is not None
    assert disabled.enabled is False
    assert disabled.exclude is True
    assert minimal is not None
    assert minimal.effort == "minimal"
    assert minimal.exclude is True
    assert (
        openrouter_chat_reasoning_config(
            repositories,
            provider="venice",
            model_id="z-ai/glm-4.7",
        )
        is None
    )
    assert (
        openrouter_chat_reasoning_config(
            repositories,
            provider="openrouter",
            model_id="missing/model",
        )
        is None
    )


def test_model_thinking_off_for_mandatory_model_sends_effort_none(
    repositories: PersistenceRepositories,
) -> None:
    repositories.save_provider_model(
        provider="venice",
        model_id="venice-mandatory-reasoning",
        display_name="Mandatory",
        capabilities=["chat"],
        thinking={"levels": ["high", "medium", "low"], "mandatory": True},
    )
    repositories.set_app_setting(
        MODEL_THINKING_PREFERENCES_SETTING,
        {
            "chat": {
                "provider": "venice",
                "model_id": "venice-mandatory-reasoning",
                "level": THINKING_LEVEL_OFF,
            },
        },
    )

    config = model_thinking_reasoning_config(
        repositories,
        task="chat",
        provider="venice",
        model_id="venice-mandatory-reasoning",
    )

    assert config is not None
    assert config.effort == "none"
    assert config.exclude is True
    assert config.enabled is None
    assert (
        model_thinking_preference_level(
            repositories,
            task="chat",
            provider="venice",
            model_id="venice-mandatory-reasoning",
        )
        == THINKING_LEVEL_OFF
    )


def test_model_thinking_off_for_optional_model_sends_disabled(
    repositories: PersistenceRepositories,
) -> None:
    repositories.save_provider_model(
        provider="openrouter",
        model_id="optional-reasoning",
        display_name="Optional",
        capabilities=["chat"],
        thinking={"levels": ["high", "low", "none"], "mandatory": False},
    )
    repositories.set_app_setting(
        MODEL_THINKING_PREFERENCES_SETTING,
        {
            "chat": {
                "provider": "openrouter",
                "model_id": "optional-reasoning",
                "level": THINKING_LEVEL_OFF,
            },
        },
    )

    config = model_thinking_reasoning_config(
        repositories,
        task="chat",
        provider="openrouter",
        model_id="optional-reasoning",
    )

    assert config is not None
    assert config.enabled is False
    assert config.exclude is True
    assert config.effort is None


def test_model_thinking_off_propagates_to_structured_output(
    repositories: PersistenceRepositories,
) -> None:
    repositories.save_provider_model(
        provider="venice",
        model_id="venice-mandatory-reasoning",
        display_name="Mandatory",
        capabilities=["chat"],
        thinking={"levels": ["high", "medium", "low"], "mandatory": True},
    )
    repositories.set_app_setting(
        MODEL_THINKING_PREFERENCES_SETTING,
        {
            "state_memory": {
                "provider": "venice",
                "model_id": "venice-mandatory-reasoning",
                "level": THINKING_LEVEL_OFF,
            },
        },
    )
    request = StructuredOutputRequest(
        provider="venice",
        model_id="venice-mandatory-reasoning",
        messages=(ChatMessage(role="user", body="extract"),),
        schema_name="state",
        schema={"type": "object", "properties": {}},
    )

    updated = request_with_model_thinking_preference(
        repositories,
        request,
        task="state_memory",
    )

    assert updated.reasoning is not None
    assert updated.reasoning.effort == "none"
    assert updated.reasoning.exclude is True
    assert updated is not request
