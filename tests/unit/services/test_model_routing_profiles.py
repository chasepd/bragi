from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.generation_settings import MODEL_THINKING_PREFERENCES_SETTING
from bragi.services.model_preferences import ROLEPLAY_SHARED_MODE_SETTING
from bragi.services.model_routing_profiles import (
    MODEL_ROUTING_PROFILES_SETTING,
    apply_model_routing_profile,
    model_routing_profiles_model,
    sanitize_model_routing_profiles,
    save_current_model_routing_profile,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_sanitize_model_routing_profiles_keeps_valid_profiles() -> None:
    sanitized = sanitize_model_routing_profiles(
        {
            "profiles": [
                {
                    "id": "fast",
                    "name": " Fast ",
                    "roleplay_shared_models_enabled": True,
                    "preferences": [
                        {
                            "task": "chat",
                            "provider": "openrouter",
                            "model_id": "model/a",
                        },
                        {
                            "task": "full_roleplay_context_update",
                            "provider": "openrouter",
                            "model_id": "model/b",
                        },
                        {
                            "task": "scene_image_edit_generation",
                            "provider": "openrouter",
                            "model_id": "model/scene-edit",
                        },
                        {
                            "task": "character_image_edit_generation",
                            "provider": "openrouter",
                            "model_id": "model/character-edit",
                        },
                        {
                            "task": "action_choice_generation",
                            "provider": "openrouter",
                            "model_id": "model/action-choice",
                        },
                        {
                            "task": "context_cleanup_scan",
                            "provider": "openrouter",
                            "model_id": "model/cleanup-scan",
                        },
                        {
                            "task": "unknown_task",
                            "provider": "openrouter",
                            "model_id": "model/c",
                        },
                    ],
                    "thinking_preferences": [
                        {
                            "task": "chat",
                            "provider": "openrouter",
                            "model_id": "model/a",
                            "level": "high",
                        },
                        {
                            "task": "unknown_task",
                            "provider": "openrouter",
                            "model_id": "model/a",
                            "level": "low",
                        },
                    ],
                },
                {"id": "", "name": "blank"},
            ],
            "last_loaded_profile_id": "fast",
        }
    )

    assert sanitized == {
        "profiles": [
            {
                "id": "fast",
                "name": "Fast",
                "roleplay_shared_models_enabled": True,
                "preferences": [
                    {
                        "task": "chat",
                        "provider": "openrouter",
                        "model_id": "model/a",
                    },
                    {
                        "task": "full_roleplay_context_update",
                        "provider": "openrouter",
                        "model_id": "model/b",
                    },
                    {
                        "task": "scene_image_edit_generation",
                        "provider": "openrouter",
                        "model_id": "model/scene-edit",
                    },
                    {
                        "task": "character_image_edit_generation",
                        "provider": "openrouter",
                        "model_id": "model/character-edit",
                    },
                    {
                        "task": "action_choice_generation",
                        "provider": "openrouter",
                        "model_id": "model/action-choice",
                    },
                    {
                        "task": "context_cleanup_scan",
                        "provider": "openrouter",
                        "model_id": "model/cleanup-scan",
                    },
                ],
                "thinking_preferences": [
                    {
                        "task": "chat",
                        "provider": "openrouter",
                        "model_id": "model/a",
                        "level": "high",
                    }
                ],
            }
        ],
        "last_loaded_profile_id": "fast",
    }


def test_save_current_model_routing_profile_snapshots_visible_direct_overrides(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_app_setting(ROLEPLAY_SHARED_MODE_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="model/chat",
    )
    repositories.set_model_preference(
        task="full_roleplay_context_update",
        provider="openrouter",
        model_id="model/hidden",
    )
    repositories.set_model_preference(
        task="scenario_generation_section_premise",
        provider="venice",
        model_id="model/premise",
    )
    repositories.set_model_preference(
        task="scene_image_edit_generation",
        provider="openrouter",
        model_id="model/scene-edit",
    )
    repositories.set_app_setting(
        MODEL_THINKING_PREFERENCES_SETTING,
        {
            "chat": {
                "provider": "openrouter",
                "model_id": "model/chat",
                "level": "high",
            },
            "full_roleplay_context_update": {
                "provider": "openrouter",
                "model_id": "model/hidden",
                "level": "low",
            },
        },
    )

    profile = save_current_model_routing_profile(
        repositories,
        name="Shared Profile",
    )

    assert profile.name == "Shared Profile"
    assert profile.roleplay_shared_models_enabled is True
    assert [
        (preference.task, preference.provider, preference.model_id)
        for preference in profile.preferences
    ] == [
        ("chat", "openrouter", "model/chat"),
        ("scenario_generation_section_premise", "venice", "model/premise"),
        ("scene_image_edit_generation", "openrouter", "model/scene-edit"),
    ]
    saved = sanitize_model_routing_profiles(
        repositories.get_app_setting(MODEL_ROUTING_PROFILES_SETTING)
    )
    assert saved["last_loaded_profile_id"] == profile.id
    saved_profiles = saved["profiles"]
    assert isinstance(saved_profiles, list)
    saved_profile = saved_profiles[0]
    assert isinstance(saved_profile, dict)
    assert saved_profile["thinking_preferences"] == [
        {
            "task": "chat",
            "provider": "openrouter",
            "model_id": "model/chat",
            "level": "high",
        }
    ]


def test_apply_model_routing_profile_replaces_known_preferences_transactionally(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_app_setting(
        MODEL_ROUTING_PROFILES_SETTING,
        {
            "profiles": [
                {
                    "id": "profile-1",
                    "name": "Full",
                    "roleplay_shared_models_enabled": False,
                    "preferences": [
                        {
                            "task": "chat",
                            "provider": "openrouter",
                            "model_id": "model/chat",
                        },
                        {
                            "task": "full_roleplay_context_update",
                            "provider": "openrouter",
                            "model_id": "model/full-context",
                        },
                    ],
                    "thinking_preferences": [
                        {
                            "task": "chat",
                            "provider": "openrouter",
                            "model_id": "model/chat",
                            "level": "high",
                        }
                    ],
                }
            ]
        },
    )
    repositories.set_app_setting(ROLEPLAY_SHARED_MODE_SETTING, True)
    repositories.set_model_preference(
        task="context_update",
        provider="openrouter",
        model_id="model/stale-context",
    )
    repositories.set_model_preference(
        task="character_image_description",
        provider="openrouter",
        model_id="model/stale-vision",
    )
    repositories.set_model_preference(
        task="scene_image_edit_generation",
        provider="openrouter",
        model_id="model/stale-scene-edit",
    )
    repositories.set_app_setting(
        MODEL_THINKING_PREFERENCES_SETTING,
        {
            "chat": {
                "provider": "openrouter",
                "model_id": "model/stale-chat",
                "level": "low",
            },
            "character_image_description": {
                "provider": "openrouter",
                "model_id": "model/stale-vision",
                "level": "high",
            },
        },
    )

    profile = apply_model_routing_profile(repositories, "profile-1")

    assert profile.id == "profile-1"
    assert repositories.get_app_setting(ROLEPLAY_SHARED_MODE_SETTING) is False
    assert repositories.get_model_preference("context_update") is None
    assert repositories.get_model_preference("character_image_description") is None
    assert repositories.get_model_preference("scene_image_edit_generation") is None
    chat = repositories.get_model_preference("chat")
    full_context = repositories.get_model_preference("full_roleplay_context_update")
    assert chat is not None
    assert chat.model_id == "model/chat"
    assert full_context is not None
    assert full_context.model_id == "model/full-context"
    assert repositories.get_app_setting(MODEL_THINKING_PREFERENCES_SETTING) == {
        "chat": {
            "provider": "openrouter",
            "model_id": "model/chat",
            "level": "high",
        }
    }
    model = model_routing_profiles_model(repositories)
    assert model.last_loaded_profile_id == "profile-1"


def test_save_current_model_routing_profile_rejects_duplicate_names(
    repositories: PersistenceRepositories,
) -> None:
    save_current_model_routing_profile(repositories, name="Fast")

    with pytest.raises(ValueError, match="already exists"):
        save_current_model_routing_profile(repositories, name=" fast ")
