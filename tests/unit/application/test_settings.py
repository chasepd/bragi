from __future__ import annotations

import builtins
import importlib
import sqlite3
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from pytest import MonkeyPatch

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.agentic_context import (
    AGENTIC_CONTEXT_PIPELINE_SETTING,
    PLAN_FIRST_NARRATOR_SETTING,
)
from bragi.services.character_action_planning_service import (
    CHARACTER_ACTION_PLANNING_ENABLED_SETTING,
    CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY_SETTING,
)
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
)
from bragi.services.director_pressure_service import DIRECTOR_PRESSURE_ENABLED_SETTING
from bragi.services.generation_settings import MODEL_THINKING_PREFERENCES_SETTING
from bragi.services.image_style_settings import save_image_style_preset_setting_key
from bragi.services.model_preferences import (
    SAVE_MODEL_OVERRIDES_SETTING,
    scenario_generation_section_model_task,
)
from bragi.services.npc_knowledge_audit_service import (
    NPC_KNOWLEDGE_AUDIT_MODE_HARD_FAIL,
    NPC_KNOWLEDGE_AUDIT_MODE_SETTING,
    NPC_KNOWLEDGE_AUDIT_MODE_SOFT_FAIL,
)
from bragi.services.phrase_denylist import (
    GENERATED_PHRASE_DENYLIST_SETTING,
    SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
)
from bragi.services.post_turn_inference import (
    POST_TURN_INFERENCE_MODE_HYBRID,
    POST_TURN_INFERENCE_MODE_LEGACY,
    POST_TURN_INFERENCE_MODE_PLAN_OWNED,
    POST_TURN_INFERENCE_MODE_SETTING,
)
from bragi.services.secrets import InMemorySecretStore
from bragi.services.settings_service import SettingsService

_MISSING = object()
_SENTINEL_SECRET = "super-secret-token"
_SENTINEL_BEARER = f"Bearer {_SENTINEL_SECRET}"
_SENTINEL_UPPER_BEARER = f"BEARER {_SENTINEL_SECRET}"
_SENTINEL_TOKEN_PARAM = f"token={_SENTINEL_SECRET}"

EXPECTED_TASKS = {
    "chat",
    "chat_full_roleplay",
    "chat_fantasy_roleplay",
    "chat_science_fiction_roleplay",
    "chat_first_contact_exploration",
    "chat_survival_expedition",
    "chat_time_loop",
    "chat_investigation_mystery",
    "chat_heist_infiltration",
    "chat_political_intrigue",
    "chat_dating_sim",
    "narrator_fallback",
    "chat_fallback",
    "structured_output_fallback",
    "tool_call_fallback",
    "scenario_generation",
    "context_search",
    "summarization",
    "state_memory",
    "context_update",
    "character_enhancement",
    "fact_observation",
    "memory_curation",
    "response_planning",
    "response_verification",
    "director_pressure",
    "action_choice_generation",
    "character_presence_assessment",
    "character_intent_planning",
    "dating_route_profile",
    "character_action_planning",
    "character_registry_maintenance",
    "context_cleanup_scan",
    "context_cleanup_actions",
    "guided_context_cleanup",
    "context_cleanup",
    "state_pruning",
    "scenario_evolution",
    "npc_knowledge_audit",
    "image_prompt",
    "image_generation",
    "image_to_image_generation",
    "scene_image_edit_generation",
    "character_image_edit_generation",
    "text_message_image_edit_generation",
    "video_generation",
    "image_animation",
    "image_fallback",
    "image_edit_fallback",
    "video_fallback",
    "character_image_description",
}

EXPECTED_IMAGE_STYLE_PRESETS = (
    "none",
    "realistic",
    "anime",
    "cartoon",
    "cinematic",
    "concept_art",
    "digital_painting",
    "watercolor",
    "oil_painting",
    "comic_book",
    "colored_pencil",
    "sketch",
    "ink",
    "pixel_art",
    "three_d_render",
    "low_poly",
)

EXPECTED_IMAGE_DIMENSION_PRESETS = (
    "provider_default",
    "square_1024x1024",
    "landscape_1024x768",
    "portrait_768x1024",
    "wide_1024x576",
    "tall_576x1024",
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_settings_model_is_import_safe_and_exposes_provider_cards(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    _seed_settings_data(repositories)
    repositories.upsert_provider_config(
        provider="venice",
        enabled=True,
        has_api_key=False,
        last_error=f"model refresh failed for {_SENTINEL_BEARER}",
    )

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )

    cards = {
        _value(card, "provider"): card
        for card in _list(_value(model, "provider_cards", "providers"))
    }
    assert set(cards) == {"openrouter", "venice"}
    assert _value(cards["openrouter"], "enabled") is True
    assert _value(cards["openrouter"], "has_api_key") is True
    assert _value(cards["openrouter"], "model_count") == 10
    assert _value(cards["openrouter"], "last_error") is None
    assert _value(cards["venice"], "enabled") is True
    assert _value(cards["venice"], "has_api_key") is False
    assert _value(cards["venice"], "last_error") == (
        "model refresh failed for Bearer [redacted]"
    )
    assert "model refresh failed" in _value(cards["venice"], "last_error")
    assert _SENTINEL_SECRET not in _value(cards["venice"], "last_error")
    assert _SENTINEL_SECRET not in repr(cards["venice"])


def test_settings_model_provider_cards_expose_refresh_timestamp_and_status(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    repositories.upsert_provider_config(
        provider="configured",
        enabled=True,
        has_api_key=True,
    )
    repositories.upsert_provider_config(
        provider="refreshed",
        enabled=True,
        has_api_key=True,
        last_model_refresh_at="2026-05-12T18:30:00+00:00",
    )
    repositories.upsert_provider_config(
        provider="stale",
        enabled=True,
        has_api_key=True,
    )
    repositories.save_provider_model(
        provider="stale",
        model_id="stale/chat",
        display_name="Stale Chat",
        capabilities=["chat"],
    )
    repositories.upsert_provider_config(
        provider="failed",
        enabled=True,
        has_api_key=True,
        last_model_refresh_at="2026-05-12T18:30:00+00:00",
        last_error=f"refresh failed for {_SENTINEL_BEARER}",
    )

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("configured", "refreshed", "stale", "failed"),
    )

    cards = {
        _value(card, "provider"): card
        for card in _list(_value(model, "provider_cards", "providers"))
    }
    assert _value(cards["configured"], "last_model_refresh_at") is None
    assert _value(cards["configured"], "refresh_status") == (
        "Configured; not refreshed"
    )
    assert _value(cards["refreshed"], "last_model_refresh_at") == (
        "2026-05-12T18:30:00+00:00"
    )
    assert _value(cards["refreshed"], "refresh_status") == (
        "Refreshed 2026-05-12T18:30:00+00:00"
    )
    assert _value(cards["stale"], "last_model_refresh_at") is None
    assert _value(cards["stale"], "refresh_status") == (
        "Models available; refresh time unknown"
    )
    assert _value(cards["failed"], "last_model_refresh_at") == (
        "2026-05-12T18:30:00+00:00"
    )
    assert _value(cards["failed"], "refresh_status") == "Refresh failed"
    assert _value(cards["failed"], "last_error") == (
        "refresh failed for Bearer [redacted]"
    )


def test_provider_settings_model_exposes_only_provider_section(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    _seed_settings_data(repositories)

    model = settings.build_provider_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
        current_user_role="admin",
        secret_storage_warning="Secret storage unavailable.",
    )

    cards = {
        _value(card, "provider"): card
        for card in _list(_value(model, "provider_cards", "providers"))
    }
    assert set(cards) == {"openrouter", "venice"}
    assert _value(cards["openrouter"], "model_count") == 10
    assert _value(model, "secret_storage_warning") == "Secret storage unavailable."
    assert not hasattr(model, "task_model_selectors")
    assert not hasattr(model, "pending_jobs_display_mode")


def test_provider_settings_model_hides_provider_cards_for_non_admin(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    _seed_settings_data(repositories)

    model = settings.build_provider_settings_model(
        repositories=repositories,
        providers=("openrouter",),
        current_user_role="user",
        secret_storage_warning="Secret storage unavailable.",
    )

    assert _list(_value(model, "provider_cards", "providers")) == []
    assert _value(model, "secret_storage_warning") is None


def test_local_settings_model_exposes_account_controls_without_model_selectors(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    repositories.set_scoped_setting(
        scope="user",
        scope_id="user-1",
        key="pending_jobs_display_mode",
        value="expanded",
    )
    repositories.set_scoped_setting(
        scope="user",
        scope_id="user-1",
        key="user_narration_guidance",
        value="Keep narration concise.",
    )
    repositories.set_app_setting("debug_logging_enabled", True)

    model = settings.build_local_settings_model(
        repositories=repositories,
        current_user_role="admin",
        current_user_id="user-1",
    )

    pending_jobs = _value(model, "pending_jobs_display_mode")
    assert _value(pending_jobs, "selected") == "expanded"
    narration_guidance = _value(model, "user_narration_guidance")
    assert _value(narration_guidance, "value") == "Keep narration concise."
    debug_logging = _value(model, "debug_logging")
    assert _value(debug_logging, "enabled", "value") is True
    content_rating = _value(model, "content_rating")
    assert _value(content_rating, "selected") == "pg-13"
    assert _list(_value(content_rating, "options")) == [
        "g",
        "pg",
        "pg-13",
        "r",
        "unrated",
    ]
    assert _value(content_rating, "admin_granted") is False
    fade_to_black = _value(model, "fade_to_black")
    assert _value(fade_to_black, "enabled", "value") is True
    assert not hasattr(model, "provider_cards")
    assert not hasattr(model, "task_model_selectors")


def test_local_settings_model_hides_debug_logging_for_non_admin(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    repositories.set_app_setting("debug_logging_enabled", True)

    model = settings.build_local_settings_model(
        repositories=repositories,
        current_user_role="user",
        current_user_id="user-1",
    )

    assert _value(model, "debug_logging") is None


def test_local_settings_model_limits_child_content_safety_controls(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    child = repositories.create_user(
        username="child",
        role="child",
        password_hash="hash",
    )

    model = settings.build_local_settings_model(
        repositories=repositories,
        current_user_role="child",
        current_user_id=child.id,
    )

    content_rating = _value(model, "content_rating")
    assert _value(content_rating, "selected") == "pg"
    assert _list(_value(content_rating, "options")) == ["g", "pg"]
    assert _value(content_rating, "admin_granted") is False
    assert _value(model, "fade_to_black") is None


def test_local_settings_model_preserves_admin_granted_child_pg_13(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    child = repositories.create_user(
        username="child",
        role="child",
        password_hash="hash",
    )
    repositories.set_scoped_setting(
        scope="user",
        scope_id=child.id,
        key="content_filter_rating",
        value="pg-13",
    )

    model = settings.build_local_settings_model(
        repositories=repositories,
        current_user_role="child",
        current_user_id=child.id,
    )

    content_rating = _value(model, "content_rating")
    assert _value(content_rating, "selected") == "pg-13"
    assert _list(_value(content_rating, "options")) == ["g", "pg", "pg-13"]
    assert _value(content_rating, "admin_granted") is True


def test_settings_model_exposes_openrouter_routing_controls(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    repositories.set_app_setting(
        "openrouter_routing_profiles",
        {
            "global": {"sort": "price"},
            "task_overrides": {
                "narrator": {
                    "enabled": True,
                    "profile": {
                        "sort": "throughput",
                        "sort_partition": "none",
                    },
                },
            },
        },
    )
    repositories.replace_provider_catalog_entries(
        provider="openrouter",
        entries=[
            {
                "slug": "openai",
                "name": "OpenAI",
                "privacy_policy_url": "https://openai.com/privacy",
                "terms_of_service_url": "https://openai.com/terms",
                "status_page_url": "https://status.openai.com",
                "headquarters": "US",
                "datacenters": ["US", "IE"],
            }
        ],
    )

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter",),
    )

    routing = _value(model, "openrouter_routing")
    assert _value(routing, "setting_key") == "openrouter_routing_profiles"
    assert _value(routing, "global_provider_payload") == {"sort": "price"}
    overrides = {
        _value(override, "task_family"): override
        for override in _list(_value(routing, "task_overrides"))
    }
    assert _value(overrides["narrator"], "enabled") is True
    assert _value(overrides["narrator"], "effective_provider_payload") == {
        "sort": {"by": "throughput", "partition": "none"}
    }
    assert _value(overrides["background_text"], "effective_provider_payload") == {
        "sort": "price"
    }
    assert "fp8" in _value(routing, "quantization_options")
    catalog = _list(_value(routing, "provider_catalog"))
    assert [_value(entry, "slug") for entry in catalog] == ["openai"]
    assert _value(catalog[0], "name") == "OpenAI"
    assert _value(catalog[0], "datacenters") == ("US", "IE")
    assert _value(routing, "provider_catalog_refreshed_at") is not None


def test_settings_model_exposes_model_routing_profiles(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    repositories.set_app_setting(
        "model_routing_profiles",
        {
            "profiles": [
                {
                    "id": "fast",
                    "name": "Fast",
                    "roleplay_shared_models_enabled": True,
                    "preferences": [
                        {
                            "task": "chat",
                            "provider": "openrouter",
                            "model_id": "model/chat",
                        }
                    ],
                }
            ],
            "last_loaded_profile_id": "fast",
        },
    )

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter",),
    )

    profiles = _value(model, "model_routing_profiles")
    assert _value(profiles, "setting_key") == "model_routing_profiles"
    assert _value(profiles, "last_loaded_profile_id") == "fast"
    profile = _list(_value(profiles, "profiles"))[0]
    assert _value(profile, "id") == "fast"
    assert _value(profile, "name") == "Fast"
    assert _value(profile, "roleplay_shared_models_enabled") is True
    assert _value(profile, "preference_count") == 1


def test_settings_model_exposes_task_model_selectors_and_unavailable_warning(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    _seed_settings_data(repositories)

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )

    selectors = {
        _value(selector, "task"): selector
        for selector in _list(_value(model, "task_model_selectors", "model_selectors"))
    }
    assert set(selectors) == EXPECTED_TASKS
    assert _selected_model_id(selectors["chat"]) == "openrouter/chat-fast"
    assert _selected_model_id(selectors["chat_full_roleplay"]) == (
        "openrouter/chat-fast"
    )
    assert _selected_model_id(selectors["chat_heist_infiltration"]) == (
        "openrouter/chat-fast"
    )
    assert _selected_model_id(selectors["chat_political_intrigue"]) == (
        "openrouter/chat-fast"
    )
    assert _selected_model_id(selectors["chat_dating_sim"]) == "openrouter/chat-fast"
    assert _selected_model_id(selectors["scenario_generation"]) == (
        "openrouter/scenario-writer"
    )
    assert _selected_model_id(selectors["context_search"]) == (
        "openrouter/context-search"
    )
    assert _selected_model_id(selectors["summarization"]) == "openrouter/summary"
    assert _selected_model_id(selectors["state_memory"]) == "openrouter/memory"
    assert _selected_model_id(selectors["context_update"]) == (
        "openrouter/context-search"
    )
    assert _selected_model_id(selectors["character_enhancement"]) == (
        "openrouter/context-search"
    )
    assert _value(selectors["character_enhancement"], "clearable") is False
    assert _value(selectors["action_choice_generation"], "selected_model_id") is None
    assert (
        _value(selectors["character_presence_assessment"], "selected_model_id")
        is None
    )
    assert _value(selectors["character_intent_planning"], "selected_model_id") is None
    assert _selected_model_id(selectors["dating_route_profile"]) == (
        "openrouter/context-search"
    )
    assert _selected_model_id(selectors["context_cleanup"]) == (
        "openrouter/context-search"
    )
    assert _selected_model_id(selectors["context_cleanup_scan"]) == (
        "openrouter/context-search"
    )
    assert _selected_model_id(selectors["context_cleanup_actions"]) == (
        "openrouter/context-search"
    )
    assert _selected_model_id(selectors["guided_context_cleanup"]) == (
        "openrouter/context-search"
    )
    assert _selected_model_id(selectors["scenario_evolution"]) == (
        "openrouter/context-search"
    )
    assert _selected_model_id(selectors["image_generation"]) == "venice/image"
    assert _selected_model_id(selectors["image_to_image_generation"]) == (
        "openrouter/image-edit"
    )
    assert _selected_model_id(selectors["scene_image_edit_generation"]) == (
        "openrouter/image-edit"
    )
    assert _selected_model_id(selectors["character_image_edit_generation"]) == (
        "openrouter/image-edit"
    )
    assert _selected_model_id(selectors["text_message_image_edit_generation"]) == (
        "openrouter/image-edit"
    )
    assert _value(selectors["scene_image_edit_generation"], "clearable") is False
    assert _value(selectors["character_image_edit_generation"], "clearable") is False
    assert (
        _value(selectors["text_message_image_edit_generation"], "clearable")
        is False
    )
    assert _selected_model_id(selectors["character_image_description"]) == (
        "openrouter/vision"
    )
    assert _value(selectors["image_prompt"], "selected_provider") is None
    assert _value(selectors["image_prompt"], "selected_model_id") is None

    chat_options = _list(_value(selectors["chat"], "options", "models"))
    assert any(
        _value(option, "provider") == "openrouter"
        and _value(option, "model_id") == "openrouter/chat-fast"
        and _value(option, "available") is True
        for option in chat_options
    )
    for task in (
        "chat",
        "chat_full_roleplay",
        "chat_political_intrigue",
        "scenario_generation",
        "summarization",
        "image_prompt",
    ):
        model_ids = _option_model_ids(selectors[task])
        assert "openrouter/chat-fast" in model_ids
        assert "openrouter/scenario-writer" in model_ids
        assert "openrouter/summary" in model_ids
        assert "openrouter/context-search" not in model_ids
        assert "openrouter/memory" not in model_ids
        assert "openrouter/image" not in model_ids
        assert "openrouter/vision" in model_ids
        assert "venice/image" not in model_ids

    scenario_options = _list(
        _value(selectors["scenario_generation"], "options", "models")
    )
    assert any(
        _value(option, "provider") == "openrouter"
        and _value(option, "model_id") == "openrouter/scenario-writer"
        and _value(option, "available") is True
        and "chat" in _value(option, "capabilities")
        for option in scenario_options
    )
    assert "openrouter/image" not in _option_model_ids(selectors["scenario_generation"])
    assert "venice/image" not in _option_model_ids(selectors["scenario_generation"])

    image_prompt_options = _list(_value(selectors["image_prompt"], "options", "models"))
    assert any(
        _value(option, "provider") == "openrouter"
        and _value(option, "model_id") == "openrouter/chat-fast"
        and _value(option, "available") is True
        and "chat" in _value(option, "capabilities")
        for option in image_prompt_options
    )
    assert "openrouter/image" not in _option_model_ids(selectors["image_prompt"])
    assert "venice/image" not in _option_model_ids(selectors["image_prompt"])

    context_selector = selectors["context_search"]
    context_model_ids = _option_model_ids(context_selector)
    assert context_model_ids == {
        "openrouter/context-search",
        "openrouter/memory",
    }
    context_options = _list(_value(context_selector, "options", "models"))
    assert all(
        "structured_output" in _value(option, "capabilities")
        for option in context_options
    )
    assert "openrouter/chat-fast" not in context_model_ids
    assert "openrouter/scenario-writer" not in context_model_ids
    assert "openrouter/summary" not in context_model_ids
    assert _value(context_selector, "selected_provider") == "openrouter"
    assert _value(context_selector, "selected_model_id") == (
        "openrouter/context-search"
    )
    assert _value(context_selector, "selected_available") is True
    assert _value(context_selector, "warning") is None

    image_model_ids = _option_model_ids(selectors["image_generation"])
    assert "openrouter/image" in image_model_ids
    assert "venice/image" in image_model_ids
    assert "openrouter/chat-fast" not in image_model_ids
    assert "openrouter/scenario-writer" not in image_model_ids
    assert "openrouter/context-search" not in image_model_ids
    assert "openrouter/summary" not in image_model_ids
    assert "openrouter/memory" not in image_model_ids
    assert "openrouter/vision" not in image_model_ids

    image_to_image_model_ids = _option_model_ids(selectors["image_to_image_generation"])
    assert image_to_image_model_ids == {"openrouter/image-edit"}
    assert _option_model_ids(selectors["scene_image_edit_generation"]) == {
        "openrouter/image-edit"
    }
    assert _option_model_ids(selectors["character_image_edit_generation"]) == {
        "openrouter/image-edit"
    }
    assert _option_model_ids(selectors["text_message_image_edit_generation"]) == {
        "openrouter/image-edit"
    }

    video_model_ids = _option_model_ids(selectors["video_generation"])
    assert "openrouter/text-video" in video_model_ids
    assert "openrouter/image-video" not in video_model_ids
    assert "openrouter/image" not in video_model_ids
    assert "venice/image" not in video_model_ids

    animation_model_ids = _option_model_ids(selectors["image_animation"])
    assert "openrouter/image-video" in animation_model_ids
    assert "openrouter/text-video" not in animation_model_ids
    assert "openrouter/image" not in animation_model_ids
    assert "venice/image" not in animation_model_ids

    character_image_selector = selectors["character_image_description"]
    character_image_model_ids = _option_model_ids(character_image_selector)
    assert character_image_model_ids == {"openrouter/vision"}
    character_image_options = _list(
        _value(character_image_selector, "options", "models")
    )
    assert all(
        "vision" in _value(option, "capabilities")
        for option in character_image_options
    )
    assert _value(character_image_selector, "selected_provider") == "openrouter"
    assert _value(character_image_selector, "selected_model_id") == (
        "openrouter/vision"
    )
    assert _value(character_image_selector, "selected_available") is True
    assert _value(character_image_selector, "warning") is None

    state_selector = selectors["state_memory"]
    state_model_ids = _option_model_ids(state_selector)
    assert state_model_ids == {
        "openrouter/context-search",
        "openrouter/memory",
    }
    state_options = _list(_value(state_selector, "options", "models"))
    assert all(
        "structured_output" in _value(option, "capabilities")
        for option in state_options
    )
    assert _value(state_selector, "selected_provider") == "openrouter"
    assert _value(state_selector, "selected_model_id") == "openrouter/memory"
    assert _value(state_selector, "selected_available") is False
    assert "unavailable" in _value(state_selector, "warning").lower()

    context_update_selector = selectors["context_update"]
    context_update_model_ids = _option_model_ids(context_update_selector)
    assert context_update_model_ids == {
        "openrouter/context-search",
        "openrouter/memory",
    }
    context_update_options = _list(_value(context_update_selector, "options", "models"))
    assert all(
        "structured_output" in _value(option, "capabilities")
        for option in context_update_options
    )
    assert _value(context_update_selector, "selected_provider") == "openrouter"
    assert _value(context_update_selector, "selected_model_id") == (
        "openrouter/context-search"
    )
    assert _value(context_update_selector, "selected_available") is True
    assert _value(context_update_selector, "warning") is None

    enhancement_selector = selectors["character_enhancement"]
    enhancement_model_ids = _option_model_ids(enhancement_selector)
    assert enhancement_model_ids == {
        "openrouter/context-search",
        "openrouter/memory",
    }
    enhancement_options = _list(_value(enhancement_selector, "options", "models"))
    assert all(
        "structured_output" in _value(option, "capabilities")
        for option in enhancement_options
    )
    assert _value(enhancement_selector, "selected_provider") == "openrouter"
    assert _value(enhancement_selector, "selected_model_id") == (
        "openrouter/context-search"
    )
    assert _value(enhancement_selector, "selected_available") is True
    assert _value(enhancement_selector, "warning") is None

    context_cleanup_selector = selectors["context_cleanup"]
    context_cleanup_model_ids = _option_model_ids(context_cleanup_selector)
    assert context_cleanup_model_ids == {
        "openrouter/context-search",
        "openrouter/memory",
    }
    context_cleanup_options = _list(
        _value(context_cleanup_selector, "options", "models")
    )
    assert all(
        "structured_output" in _value(option, "capabilities")
        for option in context_cleanup_options
    )
    assert _value(context_cleanup_selector, "selected_provider") == "openrouter"
    assert _value(context_cleanup_selector, "selected_model_id") == (
        "openrouter/context-search"
    )
    assert _value(context_cleanup_selector, "selected_available") is True
    assert _value(context_cleanup_selector, "warning") is None

    scenario_evolution_selector = selectors["scenario_evolution"]
    scenario_evolution_model_ids = _option_model_ids(scenario_evolution_selector)
    assert scenario_evolution_model_ids == {
        "openrouter/context-search",
        "openrouter/memory",
    }
    scenario_evolution_options = _list(
        _value(scenario_evolution_selector, "options", "models")
    )
    assert all(
        "structured_output" in _value(option, "capabilities")
        for option in scenario_evolution_options
    )
    assert _value(scenario_evolution_selector, "selected_provider") == "openrouter"
    assert _value(scenario_evolution_selector, "selected_model_id") == (
        "openrouter/context-search"
    )
    assert _value(scenario_evolution_selector, "selected_available") is True
    assert _value(scenario_evolution_selector, "warning") is None

    pruning_selector = selectors["state_pruning"]
    pruning_model_ids = _option_model_ids(pruning_selector)
    assert pruning_model_ids == {
        "openrouter/context-search",
        "openrouter/memory",
    }
    pruning_options = _list(_value(pruning_selector, "options", "models"))
    assert all(
        "structured_output" in _value(option, "capabilities")
        for option in pruning_options
    )
    assert _value(pruning_selector, "selected_provider") == "openrouter"
    assert _value(pruning_selector, "selected_model_id") == (
        "openrouter/context-search"
    )
    assert _value(pruning_selector, "selected_available") is True
    assert _value(pruning_selector, "warning") is None


def test_settings_model_exposes_active_save_model_override_selectors(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    save_id = _seed_settings_data(repositories)
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save_id,
        key=SAVE_MODEL_OVERRIDES_SETTING,
        value={
            "preferences": {
                "chat": {
                    "provider": "venice",
                    "model_id": "venice/image",
                }
            }
        },
    )

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
        active_save_id=save_id,
        current_user_role="admin",
    )

    selectors = {
        _value(selector, "task"): selector
        for selector in _list(_value(model, "save_model_override_selectors"))
    }
    chat = selectors["chat"]
    assert _value(chat, "selected_provider") == "venice"
    assert _value(chat, "selected_model_id") == "venice/image"
    assert _value(chat, "inherited_provider") == "openrouter"
    assert _value(chat, "inherited_model_id") == "openrouter/chat-fast"
    assert _value(chat, "clearable") is True

    context_update = selectors["context_update"]
    assert _value(context_update, "selected_provider") == "openrouter"
    assert _value(context_update, "selected_model_id") == (
        "openrouter/context-search"
    )
    assert _value(context_update, "inherited_provider") == "openrouter"
    assert _value(context_update, "inherited_model_id") == (
        "openrouter/context-search"
    )
    assert _value(context_update, "clearable") is False


def test_settings_model_exposes_model_option_pricing(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    repositories.upsert_provider_config(
        provider="openrouter",
        enabled=True,
        has_api_key=True,
    )
    repositories.save_provider_model(
        provider="openrouter",
        model_id="openrouter/chat-fast",
        display_name="Chat Fast",
        capabilities=["chat"],
        pricing={
            "input_per_million_tokens_usd": "0.15",
            "output_per_million_tokens_usd": "0.6",
        },
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/chat-fast",
    )

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter",),
    )

    selectors = {
        _value(selector, "task"): selector
        for selector in _list(_value(model, "task_model_selectors", "model_selectors"))
    }
    chat_options = _list(_value(selectors["chat"], "options", "models"))
    chat_option = next(
        option
        for option in chat_options
        if _value(option, "model_id") == "openrouter/chat-fast"
    )
    pricing = _value(chat_option, "pricing")
    assert _value(pricing, "input_per_million_tokens_usd") == "0.15"
    assert _value(pricing, "output_per_million_tokens_usd") == "0.6"


def test_settings_model_exposes_model_thinking_control(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    repositories.upsert_provider_config(
        provider="openrouter",
        enabled=True,
        has_api_key=True,
    )
    repositories.save_provider_model(
        provider="openrouter",
        model_id="openrouter/reasoning",
        display_name="OpenRouter Reasoning",
        capabilities=["chat"],
        thinking={
            "levels": ["high", "low"],
            "default_level": "low",
            "default_enabled": True,
            "mandatory": False,
            "supports_max_tokens": True,
        },
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/reasoning",
    )
    repositories.set_app_setting(
        MODEL_THINKING_PREFERENCES_SETTING,
        {
            "chat": {
                "provider": "openrouter",
                "model_id": "openrouter/reasoning",
                "level": "high",
            }
        },
    )

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter",),
    )

    selectors = {
        _value(selector, "task"): selector
        for selector in _list(_value(model, "task_model_selectors", "model_selectors"))
    }
    control = _value(selectors["chat"], "thinking")
    assert _value(control, "supported") is True
    assert _value(control, "selected") == "high"
    assert _list(_value(control, "options")) == [
        "provider_default",
        "off",
        "high",
        "low",
    ]
    assert _value(control, "default_level") == "low"
    assert _value(control, "default_enabled") is True
    assert _value(control, "mandatory") is False
    assert _value(control, "disabled_reason") is None
    chat_options = _list(_value(selectors["chat"], "options", "models"))
    option = next(
        item
        for item in chat_options
        if _value(item, "model_id") == "openrouter/reasoning"
    )
    thinking = _value(option, "thinking")
    assert _list(_value(thinking, "levels")) == ["high", "low"]
    assert _value(thinking, "supports_max_tokens") is True


def test_settings_model_greys_out_model_thinking_for_unsupported_model(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    repositories.upsert_provider_config(
        provider="openrouter",
        enabled=True,
        has_api_key=True,
    )
    repositories.save_provider_model(
        provider="openrouter",
        model_id="openrouter/plain-chat",
        display_name="Plain Chat",
        capabilities=["chat"],
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/plain-chat",
    )

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter",),
    )

    selectors = {
        _value(selector, "task"): selector
        for selector in _list(_value(model, "task_model_selectors", "model_selectors"))
    }
    control = _value(selectors["chat"], "thinking")
    assert _value(control, "supported") is False
    assert _value(control, "selected") == "provider_default"
    assert _list(_value(control, "options")) == ["provider_default"]
    assert _value(control, "disabled_reason") == (
        "Selected model does not support thinking level"
    )


def test_settings_context_search_selector_accepts_tool_calling_models(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    _seed_settings_data(repositories)
    repositories.save_provider_model(
        provider="openrouter",
        model_id="openrouter/context-tools",
        display_name="Context Tools",
        capabilities=["tool_calling"],
        context_window=128_000,
    )

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )

    selectors = {
        _value(selector, "task"): selector
        for selector in _list(_value(model, "task_model_selectors", "model_selectors"))
    }
    context_selector = selectors["context_search"]
    context_model_ids = _option_model_ids(context_selector)
    assert "openrouter/context-search" in context_model_ids
    assert "openrouter/context-tools" in context_model_ids
    assert "openrouter/chat-fast" not in context_model_ids
    for task in (
        "context_update",
        "character_enhancement",
        "character_registry_maintenance",
        "context_cleanup_scan",
        "context_cleanup_actions",
        "guided_context_cleanup",
        "context_cleanup",
        "state_pruning",
        "scenario_evolution",
    ):
        model_ids = _option_model_ids(selectors[task])
        assert "openrouter/context-tools" in model_ids
        assert "openrouter/chat-fast" not in model_ids
    character_action_model_ids = _option_model_ids(
        selectors["character_action_planning"]
    )
    assert "openrouter/context-search" in character_action_model_ids
    assert "openrouter/context-tools" not in character_action_model_ids
    assert "openrouter/chat-fast" not in character_action_model_ids
    director_model_ids = _option_model_ids(selectors["director_pressure"])
    assert "openrouter/context-search" in director_model_ids
    assert "openrouter/context-tools" not in director_model_ids
    assert "openrouter/chat-fast" not in director_model_ids
    context_options = _list(_value(context_selector, "options", "models"))
    tool_option = next(
        option
        for option in context_options
        if _value(option, "model_id") == "openrouter/context-tools"
    )
    assert "tool_calling" in _value(tool_option, "capabilities")


def test_settings_model_exposes_optional_scenario_section_overrides(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    _seed_settings_data(repositories)
    repositories.set_model_preference(
        task=scenario_generation_section_model_task("worldbuilding"),
        provider="openrouter",
        model_id="openrouter/summary",
    )

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )

    selectors = {
        _value(selector, "section_id"): selector
        for selector in _list(_value(model, "scenario_section_model_selectors"))
    }
    assert list(selectors) == [
        "title",
        "premise",
        "player_character_name",
        "player_role",
        "choice_style",
        "opening_message",
        "worldbuilding",
        "lore",
        "locations",
        "factions",
        "tone_genre",
        "current_scene",
        "magic_system",
        "realms_and_places",
        "factions_and_orders",
        "myths_and_creatures",
        "quest_stakes",
        "technology_level",
        "setting_scope",
        "species_and_intelligences",
        "factions_and_institutions",
        "mission_stakes",
        "mission_profile",
        "ship_or_base_status",
        "exploration_target",
        "unknown_intelligence",
        "knowledge_state",
        "translation_progress",
        "discoveries_and_samples",
        "hazards_and_escalation",
        "expedition_goal",
        "route_options",
        "resource_inventory",
        "environmental_conditions",
        "hazards_and_events",
        "camp_status",
        "travel_progress",
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
        "case_facts",
        "clues",
        "timeline",
        "red_herrings",
        "hidden_truth",
        "case_status",
        "target_location",
        "objectives_and_stakes",
        "intel_and_access",
        "security_model",
        "alert_and_heat",
        "loadout_and_tools",
        "complications",
        "extraction_routes",
        "aftermath",
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
        "settlement_profile",
        "resources_and_indicators",
        "projects_and_facilities",
        "threats_and_opportunities",
        "calendar_and_deadlines",
        "hunt_profile",
        "target_profile",
        "leads_and_clues",
        "hunt_locations",
        "preparation_state",
        "hunt_status",
        "journey_profile",
        "route_and_stops",
        "transport_and_supplies",
        "recurring_pressures",
        "relationship_threads",
        "journey_progress",
        "trade_profile",
        "cargo_inventory",
        "markets_and_stops",
        "contracts_and_debts",
        "route_hazards",
        "profit_and_loss",
        "player_character_profile",
    ]

    inherited_title = selectors["title"]
    assert _value(inherited_title, "task") == (
        scenario_generation_section_model_task("title")
    )
    assert _value(inherited_title, "selected_provider") is None
    assert _value(inherited_title, "selected_model_id") is None
    assert _value(inherited_title, "inherited_provider") == "openrouter"
    assert _value(inherited_title, "inherited_model_id") == "openrouter/scenario-writer"
    assert _value(inherited_title, "clearable") is False

    worldbuilding = selectors["worldbuilding"]
    assert _value(worldbuilding, "selected_provider") == "openrouter"
    assert _value(worldbuilding, "selected_model_id") == "openrouter/summary"
    assert _value(worldbuilding, "inherited_model_id") == "openrouter/scenario-writer"
    assert _value(worldbuilding, "clearable") is True
    assert "openrouter/image" not in _option_model_ids(worldbuilding)
    assert "venice/image" not in _option_model_ids(worldbuilding)
    assert "openrouter/scenario-writer" in _option_model_ids(worldbuilding)


def test_settings_model_exposes_scenario_chat_selectors_with_generic_fallback(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    _seed_settings_data(repositories)

    fallback_model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )

    fallback_selectors = {
        _value(selector, "task"): selector
        for selector in _list(
            _value(fallback_model, "task_model_selectors", "model_selectors")
        )
    }
    for task in (
        "chat_full_roleplay",
        "chat_fantasy_roleplay",
        "chat_science_fiction_roleplay",
        "chat_first_contact_exploration",
        "chat_survival_expedition",
        "chat_time_loop",
        "chat_investigation_mystery",
        "chat_heist_infiltration",
        "chat_political_intrigue",
    ):
        selector = fallback_selectors[task]
        assert _value(selector, "selected_provider") == "openrouter"
        assert _value(selector, "selected_model_id") == "openrouter/chat-fast"
        assert _value(selector, "selected_available") is True
        assert _value(selector, "warning") is None
        assert _value(selector, "clearable") is False
        assert _option_model_ids(selector) == _option_model_ids(
            fallback_selectors["chat"]
        )
    assert _value(fallback_selectors["chat"], "clearable") is True

    service = SettingsService(
        repositories=repositories,
        providers={},
        secret_store=InMemorySecretStore(),
    )
    service.set_model_preference(
        task="chat_full_roleplay",
        provider="openrouter",
        model_id="openrouter/scenario-writer",
    )

    persisted_model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )
    persisted_selectors = {
        _value(selector, "task"): selector
        for selector in _list(
            _value(persisted_model, "task_model_selectors", "model_selectors")
        )
    }
    full_roleplay_preference = service.get_model_preference("chat_full_roleplay")

    assert full_roleplay_preference is not None
    assert full_roleplay_preference.provider == "openrouter"
    assert full_roleplay_preference.model_id == "openrouter/scenario-writer"
    assert _value(persisted_selectors["chat"], "selected_model_id") == (
        "openrouter/chat-fast"
    )
    assert _value(
        persisted_selectors["chat_full_roleplay"],
        "selected_provider",
    ) == "openrouter"
    assert _value(
        persisted_selectors["chat_full_roleplay"],
        "selected_model_id",
    ) == "openrouter/scenario-writer"
    assert _value(
        persisted_selectors["chat_full_roleplay"],
        "selected_available",
    ) is True
    assert _value(persisted_selectors["chat_full_roleplay"], "warning") is None
    assert _value(persisted_selectors["chat_full_roleplay"], "clearable") is True


def test_settings_model_defaults_to_one_shared_roleplay_model_group(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    _seed_settings_data(repositories)

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )

    shared_toggle = _value(model, "roleplay_shared_models")
    assert _value(shared_toggle, "setting_key") == "use_shared_roleplay_models"
    assert _value(shared_toggle, "enabled") is True

    groups = _list(_value(model, "roleplay_model_groups"))
    assert [_value(group, "roleplay_type") for group in groups] == ["shared"]
    assert [_value(group, "label") for group in groups] == ["Shared Roleplay"]

    selectors = {
        _value(selector, "task"): selector
        for selector in _list(_value(groups[0], "selectors"))
    }
    assert list(selectors) == [
        "chat",
        "scenario_generation",
        "context_search",
        "summarization",
        "state_memory",
        "context_update",
        "character_enhancement",
        "fact_observation",
        "memory_curation",
        "response_planning",
        "response_verification",
        "director_pressure",
        "action_choice_generation",
        "character_presence_assessment",
        "character_intent_planning",
        "dating_route_profile",
        "character_action_planning",
        "character_registry_maintenance",
        "context_cleanup_scan",
        "context_cleanup_actions",
        "guided_context_cleanup",
        "context_cleanup",
        "state_pruning",
        "scenario_evolution",
        "npc_knowledge_audit",
        "image_prompt",
        "image_generation",
        "image_to_image_generation",
        "scene_image_edit_generation",
        "character_image_edit_generation",
        "text_message_image_edit_generation",
        "video_generation",
        "image_animation",
        "narrator_fallback",
        "chat_fallback",
        "structured_output_fallback",
        "tool_call_fallback",
        "image_fallback",
        "image_edit_fallback",
        "video_fallback",
    ]
    assert "chat_full_roleplay" not in selectors
    assert "chat_fantasy_roleplay" not in selectors
    assert "chat_science_fiction_roleplay" not in selectors
    assert "chat_first_contact_exploration" not in selectors
    assert "chat_survival_expedition" not in selectors
    assert "chat_time_loop" not in selectors
    assert "chat_investigation_mystery" not in selectors
    assert "chat_heist_infiltration" not in selectors
    assert "chat_political_intrigue" not in selectors
    assert "chat_character_interaction" not in selectors
    assert "full_roleplay_scenario_generation" not in selectors
    assert "fantasy_roleplay_scenario_generation" not in selectors
    assert "science_fiction_roleplay_scenario_generation" not in selectors
    assert "first_contact_exploration_scenario_generation" not in selectors
    assert "survival_expedition_scenario_generation" not in selectors
    assert "time_loop_scenario_generation" not in selectors
    assert "investigation_mystery_scenario_generation" not in selectors
    assert "heist_infiltration_scenario_generation" not in selectors
    assert "political_intrigue_scenario_generation" not in selectors
    assert "character_interaction_scenario_generation" not in selectors
    assert _value(selectors["chat"], "selected_model_id") == "openrouter/chat-fast"
    _assert_scenario_generation_selector(selectors["scenario_generation"])
    assert _value(selectors["context_update"], "selected_model_id") == (
        "openrouter/context-search"
    )
    assert _value(selectors["character_enhancement"], "selected_model_id") == (
        "openrouter/context-search"
    )
    assert _value(selectors["character_enhancement"], "clearable") is False
    assert _value(selectors["action_choice_generation"], "selected_model_id") is None
    assert (
        _value(selectors["character_presence_assessment"], "selected_model_id")
        is None
    )
    assert _value(selectors["character_intent_planning"], "selected_model_id") is None
    assert _value(selectors["dating_route_profile"], "selected_model_id") == (
        "openrouter/context-search"
    )
    assert _value(selectors["context_cleanup"], "selected_model_id") == (
        "openrouter/context-search"
    )
    assert _value(selectors["context_cleanup_scan"], "selected_model_id") == (
        "openrouter/context-search"
    )
    assert _value(selectors["context_cleanup_actions"], "selected_model_id") == (
        "openrouter/context-search"
    )
    assert _value(selectors["guided_context_cleanup"], "selected_model_id") == (
        "openrouter/context-search"
    )
    assert _value(selectors["scenario_evolution"], "selected_model_id") == (
        "openrouter/context-search"
    )
    assert _value(selectors["image_generation"], "selected_model_id") == (
        "venice/image"
    )


def test_settings_model_exposes_separate_roleplay_groups_when_shared_is_disabled(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    _seed_settings_data(repositories)
    repositories.set_app_setting("use_shared_roleplay_models", False)
    repositories.set_model_preference(
        task="full_roleplay_context_update",
        provider="openrouter",
        model_id="openrouter/memory",
    )

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )

    shared_toggle = _value(model, "roleplay_shared_models")
    assert _value(shared_toggle, "enabled") is False

    groups = {
        _value(group, "roleplay_type"): group
        for group in _list(_value(model, "roleplay_model_groups"))
    }
    assert list(groups) == [
        "full_roleplay",
        "fantasy_roleplay",
        "science_fiction_roleplay",
        "first_contact_exploration",
        "survival_expedition",
        "time_loop",
        "investigation_mystery",
        "heist_infiltration",
        "political_intrigue",
        "dating_sim",
    ]
    assert _value(groups["full_roleplay"], "label") == "Generic Roleplay"
    assert _value(groups["fantasy_roleplay"], "label") == "Fantasy"
    assert _value(groups["science_fiction_roleplay"], "label") == "Science Fiction"
    assert _value(groups["first_contact_exploration"], "label") == (
        "First Contact / Exploration"
    )
    assert _value(groups["survival_expedition"], "label") == "Survival Expedition"
    assert _value(groups["time_loop"], "label") == "Time Loop"
    assert _value(groups["investigation_mystery"], "label") == (
        "Investigation Mystery"
    )
    assert _value(groups["heist_infiltration"], "label") == (
        "Heist / Infiltration"
    )
    assert _value(groups["political_intrigue"], "label") == "Political Intrigue"
    assert _value(groups["dating_sim"], "label") == "Dating Sim"

    full_selectors = {
        _value(selector, "task"): selector
        for selector in _list(_value(groups["full_roleplay"], "selectors"))
    }
    fantasy_selectors = {
        _value(selector, "task"): selector
        for selector in _list(_value(groups["fantasy_roleplay"], "selectors"))
    }
    science_selectors = {
        _value(selector, "task"): selector
        for selector in _list(_value(groups["science_fiction_roleplay"], "selectors"))
    }
    first_contact_selectors = {
        _value(selector, "task"): selector
        for selector in _list(
            _value(groups["first_contact_exploration"], "selectors")
        )
    }
    survival_selectors = {
        _value(selector, "task"): selector
        for selector in _list(_value(groups["survival_expedition"], "selectors"))
    }
    mystery_selectors = {
        _value(selector, "task"): selector
        for selector in _list(_value(groups["investigation_mystery"], "selectors"))
    }
    heist_selectors = {
        _value(selector, "task"): selector
        for selector in _list(_value(groups["heist_infiltration"], "selectors"))
    }
    intrigue_selectors = {
        _value(selector, "task"): selector
        for selector in _list(_value(groups["political_intrigue"], "selectors"))
    }
    dating_selectors = {
        _value(selector, "task"): selector
        for selector in _list(_value(groups["dating_sim"], "selectors"))
    }
    assert list(full_selectors) == [
        "chat_full_roleplay",
        "scenario_generation",
        "full_roleplay_context_search",
        "full_roleplay_summarization",
        "full_roleplay_state_memory",
        "full_roleplay_context_update",
        "full_roleplay_character_enhancement",
        "full_roleplay_fact_observation",
        "full_roleplay_memory_curation",
        "full_roleplay_response_planning",
        "full_roleplay_response_verification",
        "full_roleplay_director_pressure",
        "full_roleplay_action_choice_generation",
        "full_roleplay_character_presence_assessment",
        "full_roleplay_character_intent_planning",
        "full_roleplay_dating_route_profile",
        "full_roleplay_character_action_planning",
        "full_roleplay_character_registry_maintenance",
        "full_roleplay_context_cleanup_scan",
        "full_roleplay_context_cleanup_actions",
        "full_roleplay_guided_context_cleanup",
        "full_roleplay_context_cleanup",
        "full_roleplay_state_pruning",
        "full_roleplay_scenario_evolution",
        "full_roleplay_npc_knowledge_audit",
        "full_roleplay_image_prompt",
        "full_roleplay_image_generation",
        "full_roleplay_image_to_image_generation",
        "full_roleplay_scene_image_edit_generation",
        "full_roleplay_character_image_edit_generation",
        "full_roleplay_text_message_image_edit_generation",
        "full_roleplay_video_generation",
        "full_roleplay_image_animation",
        "full_roleplay_narrator_fallback",
        "full_roleplay_chat_fallback",
        "full_roleplay_structured_output_fallback",
        "full_roleplay_tool_call_fallback",
        "full_roleplay_image_fallback",
        "full_roleplay_image_edit_fallback",
        "full_roleplay_video_fallback",
    ]
    assert list(fantasy_selectors) == _expected_roleplay_selector_tasks(
        "fantasy_roleplay",
        "chat_fantasy_roleplay",
    )
    assert list(science_selectors) == _expected_roleplay_selector_tasks(
        "science_fiction_roleplay",
        "chat_science_fiction_roleplay",
    )
    assert list(first_contact_selectors) == _expected_roleplay_selector_tasks(
        "first_contact_exploration",
        "chat_first_contact_exploration",
    )
    assert list(survival_selectors) == _expected_roleplay_selector_tasks(
        "survival_expedition",
        "chat_survival_expedition",
    )
    loop_selectors = {
        _value(selector, "task"): selector
        for selector in _list(_value(groups["time_loop"], "selectors"))
    }
    assert list(loop_selectors) == _expected_roleplay_selector_tasks(
        "time_loop",
        "chat_time_loop",
    )
    assert list(mystery_selectors) == _expected_roleplay_selector_tasks(
        "investigation_mystery",
        "chat_investigation_mystery",
    )
    assert list(heist_selectors) == _expected_roleplay_selector_tasks(
        "heist_infiltration",
        "chat_heist_infiltration",
    )
    assert list(intrigue_selectors) == _expected_roleplay_selector_tasks(
        "political_intrigue",
        "chat_political_intrigue",
    )
    assert list(dating_selectors) == [
        "chat_dating_sim",
        "scenario_generation",
        "dating_sim_context_search",
        "dating_sim_summarization",
        "dating_sim_state_memory",
        "dating_sim_context_update",
        "dating_sim_character_enhancement",
        "dating_sim_fact_observation",
        "dating_sim_memory_curation",
        "dating_sim_response_planning",
        "dating_sim_response_verification",
        "dating_sim_director_pressure",
        "dating_sim_action_choice_generation",
        "dating_sim_character_presence_assessment",
        "dating_sim_character_intent_planning",
        "dating_sim_dating_route_profile",
        "dating_sim_character_action_planning",
        "dating_sim_character_registry_maintenance",
        "dating_sim_context_cleanup_scan",
        "dating_sim_context_cleanup_actions",
        "dating_sim_guided_context_cleanup",
        "dating_sim_context_cleanup",
        "dating_sim_state_pruning",
        "dating_sim_scenario_evolution",
        "dating_sim_npc_knowledge_audit",
        "dating_sim_image_prompt",
        "dating_sim_image_generation",
        "dating_sim_image_to_image_generation",
        "dating_sim_scene_image_edit_generation",
        "dating_sim_character_image_edit_generation",
        "dating_sim_text_message_image_edit_generation",
        "dating_sim_video_generation",
        "dating_sim_image_animation",
        "dating_sim_narrator_fallback",
        "dating_sim_chat_fallback",
        "dating_sim_structured_output_fallback",
        "dating_sim_tool_call_fallback",
        "dating_sim_image_fallback",
        "dating_sim_image_edit_fallback",
        "dating_sim_video_fallback",
    ]
    assert "full_roleplay_scenario_generation" not in full_selectors
    assert "fantasy_roleplay_scenario_generation" not in fantasy_selectors
    assert "science_fiction_roleplay_scenario_generation" not in science_selectors
    assert (
        "first_contact_exploration_scenario_generation"
        not in first_contact_selectors
    )
    assert "survival_expedition_scenario_generation" not in survival_selectors
    assert "time_loop_scenario_generation" not in loop_selectors
    assert "investigation_mystery_scenario_generation" not in mystery_selectors
    assert "political_intrigue_scenario_generation" not in intrigue_selectors
    assert "dating_sim_scenario_generation" not in dating_selectors

    assert _value(full_selectors["chat_full_roleplay"], "selected_model_id") == (
        "openrouter/chat-fast"
    )
    _assert_scenario_generation_selector(full_selectors["scenario_generation"])
    _assert_scenario_generation_selector(fantasy_selectors["scenario_generation"])
    _assert_scenario_generation_selector(survival_selectors["scenario_generation"])
    _assert_scenario_generation_selector(loop_selectors["scenario_generation"])
    _assert_scenario_generation_selector(mystery_selectors["scenario_generation"])
    _assert_scenario_generation_selector(intrigue_selectors["scenario_generation"])
    _assert_scenario_generation_selector(science_selectors["scenario_generation"])
    _assert_scenario_generation_selector(
        first_contact_selectors["scenario_generation"]
    )
    _assert_scenario_generation_selector(dating_selectors["scenario_generation"])
    assert _value(dating_selectors["chat_dating_sim"], "selected_model_id") == (
        "openrouter/chat-fast"
    )
    assert _value(
        full_selectors["full_roleplay_context_update"],
        "selected_model_id",
    ) == "openrouter/memory"
    assert _option_model_ids(full_selectors["full_roleplay_context_update"]) == {
        "openrouter/context-search",
        "openrouter/memory",
    }
    assert _value(
        full_selectors["full_roleplay_character_enhancement"],
        "selected_model_id",
    ) == "openrouter/memory"
    assert _value(
        full_selectors["full_roleplay_character_enhancement"],
        "clearable",
    ) is False
    assert _value(
        full_selectors["full_roleplay_scenario_evolution"],
        "selected_model_id",
    ) == "openrouter/context-search"


def test_settings_model_warns_when_scenario_chat_selection_is_unavailable(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    _seed_settings_data(repositories)
    repositories.save_provider_model(
        provider="openrouter",
        model_id="openrouter/stale-chat",
        display_name="Stale Chat",
        capabilities=["chat"],
        context_window=128_000,
    )
    repositories.mark_missing_provider_models_unavailable(
        provider="openrouter",
        available_model_ids={
            "openrouter/scenario-writer",
            "openrouter/context-search",
            "openrouter/summary",
            "openrouter/image",
        },
    )
    repositories.set_model_preference(
        task="chat_full_roleplay",
        provider="openrouter",
        model_id="openrouter/stale-chat",
    )

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )

    selectors = {
        _value(selector, "task"): selector
        for selector in _list(_value(model, "task_model_selectors", "model_selectors"))
    }
    specific_selector = selectors["chat_full_roleplay"]
    fallback_selector = selectors["chat_dating_sim"]
    assert _value(specific_selector, "selected_model_id") == "openrouter/stale-chat"
    assert _value(specific_selector, "selected_available") is False
    assert "unavailable" in _value(specific_selector, "warning").lower()
    assert _value(fallback_selector, "selected_model_id") == "openrouter/chat-fast"
    assert _value(fallback_selector, "selected_available") is False
    assert "unavailable" in _value(fallback_selector, "warning").lower()


def test_settings_model_accepts_legacy_image_capability_alias(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    _seed_settings_data(repositories)
    repositories.upsert_provider_config(
        provider="legacy",
        enabled=True,
        has_api_key=True,
    )
    repositories.save_provider_model(
        provider="legacy",
        model_id="legacy/image-v1",
        display_name="Legacy Image",
        capabilities=["image"],
        context_window=16_000,
    )

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice", "legacy"),
    )

    selectors = {
        _value(selector, "task"): selector
        for selector in _list(_value(model, "task_model_selectors", "model_selectors"))
    }
    image_options = _list(_value(selectors["image_generation"], "options", "models"))
    assert any(
        _value(option, "provider") == "legacy"
        and _value(option, "model_id") == "legacy/image-v1"
        and _value(option, "capabilities") == ("image",)
        for option in image_options
    )
    assert "legacy/image-v1" not in _option_model_ids(selectors["chat"])


def test_settings_model_exposes_fallback_selectors(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    repositories.upsert_provider_config(
        provider="catalog",
        enabled=True,
        has_api_key=True,
    )
    for model_id, display_name, capabilities in [
        (
            "catalog/chat-fallback",
            "Fallback Chat",
            ["chat", "fallback_marker"],
        ),
        (
            "catalog/chat-unmoderated",
            "Unmoderated Chat",
            ["chat", "unmoderated_fallback"],
        ),
        ("catalog/chat-only", "Chat Only", ["chat"]),
        (
            "catalog/structured-fallback",
            "Fallback Structured",
            ["structured_output", "fallback_marker"],
        ),
        (
            "catalog/structured-unmoderated",
            "Unmoderated Structured",
            ["json_schema", "unmoderated_fallback"],
        ),
        ("catalog/structured-only", "Structured Only", ["structured_output"]),
        (
            "catalog/tool-fallback",
            "Tool Fallback",
            ["tool_calling", "fallback_marker"],
        ),
        ("catalog/function-fallback", "Function Fallback", ["function_calling"]),
        (
            "catalog/image-fallback",
            "Fallback Image",
            ["image_generation", "fallback_marker"],
        ),
        (
            "catalog/image-unmoderated",
            "Unmoderated Image",
            ["image_generation", "unmoderated_fallback"],
        ),
        ("catalog/image-only", "Image Only", ["image_generation"]),
        (
            "catalog/edit-fallback",
            "Fallback Edit",
            ["image_to_image", "fallback_marker"],
        ),
        ("catalog/edit-only", "Edit Only", ["image_edit"]),
        (
            "catalog/video-fallback",
            "Fallback Video",
            ["text_to_video", "fallback_marker"],
        ),
        (
            "catalog/video-unmoderated",
            "Unmoderated Video",
            ["video_generation", "unmoderated_fallback"],
        ),
        ("catalog/video-only", "Video Only", ["text_to_video"]),
        (
            "catalog/image-video-fallback",
            "Image Video Fallback",
            ["image_plus_text_to_video", "fallback_marker"],
        ),
        (
            "catalog/fallback-only",
            "Fallback Only",
            ["fallback_marker"],
        ),
    ]:
        repositories.save_provider_model(
            provider="catalog",
            model_id=model_id,
            display_name=display_name,
            capabilities=capabilities,
            context_window=32_000,
        )
    repositories.set_model_preference(
        task="chat_fallback",
        provider="catalog",
        model_id="catalog/chat-unmoderated",
    )
    repositories.set_model_preference(
        task="narrator_fallback",
        provider="catalog",
        model_id="catalog/chat-fallback",
    )
    repositories.set_model_preference(
        task="structured_output_fallback",
        provider="catalog",
        model_id="catalog/structured-unmoderated",
    )
    repositories.set_model_preference(
        task="tool_call_fallback",
        provider="catalog",
        model_id="catalog/tool-fallback",
    )
    repositories.set_model_preference(
        task="image_fallback",
        provider="catalog",
        model_id="catalog/image-unmoderated",
    )
    repositories.set_model_preference(
        task="image_edit_fallback",
        provider="catalog",
        model_id="catalog/edit-fallback",
    )
    repositories.set_model_preference(
        task="video_fallback",
        provider="catalog",
        model_id="catalog/video-unmoderated",
    )

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("catalog",),
    )

    selectors = {
        _value(selector, "task"): selector
        for selector in _list(_value(model, "task_model_selectors", "model_selectors"))
    }
    expected_selectors = {
        "narrator_fallback": (
            "catalog/chat-fallback",
            {
                "catalog/chat-fallback",
                "catalog/chat-unmoderated",
                "catalog/chat-only",
            },
        ),
        "chat_fallback": (
            "catalog/chat-unmoderated",
            {
                "catalog/chat-fallback",
                "catalog/chat-unmoderated",
                "catalog/chat-only",
            },
        ),
        "structured_output_fallback": (
            "catalog/structured-unmoderated",
            {
                "catalog/structured-fallback",
                "catalog/structured-unmoderated",
                "catalog/structured-only",
            },
        ),
        "tool_call_fallback": (
            "catalog/tool-fallback",
            {"catalog/tool-fallback", "catalog/function-fallback"},
        ),
        "image_fallback": (
            "catalog/image-unmoderated",
            {
                "catalog/image-fallback",
                "catalog/image-unmoderated",
                "catalog/image-only",
            },
        ),
        "image_edit_fallback": (
            "catalog/edit-fallback",
            {"catalog/edit-fallback", "catalog/edit-only"},
        ),
        "video_fallback": (
            "catalog/video-unmoderated",
            {
                "catalog/video-fallback",
                "catalog/video-unmoderated",
                "catalog/video-only",
            },
        ),
    }
    for task, (selected_model_id, option_model_ids) in expected_selectors.items():
        selector = selectors[task]
        assert _value(selector, "selected_provider") == "catalog"
        assert _value(selector, "selected_model_id") == selected_model_id
        assert _option_model_ids(selector) == option_model_ids
        assert _value(selector, "selected_available") is True


def test_settings_state_and_context_update_selectors_accept_tool_models(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    repositories.upsert_provider_config(
        provider="catalog",
        enabled=True,
        has_api_key=True,
    )
    for model_id, display_name, capabilities in [
        ("catalog/structured", "Structured", ["structured_output"]),
        ("catalog/tools", "Tools", ["tool_calling"]),
        ("catalog/function-tools", "Function Tools", ["function_calling"]),
        ("catalog/chat", "Chat", ["chat"]),
    ]:
        repositories.save_provider_model(
            provider="catalog",
            model_id=model_id,
            display_name=display_name,
            capabilities=capabilities,
        )
    repositories.set_model_preference(
        task="context_update",
        provider="catalog",
        model_id="catalog/tools",
    )
    repositories.set_model_preference(
        task="character_enhancement",
        provider="catalog",
        model_id="catalog/structured",
    )
    repositories.set_model_preference(
        task="state_memory",
        provider="catalog",
        model_id="catalog/function-tools",
    )

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("catalog",),
    )

    selectors = {
        _value(selector, "task"): selector
        for selector in _list(_value(model, "task_model_selectors", "model_selectors"))
    }
    context_update_selector = selectors["context_update"]
    assert _value(context_update_selector, "selected_model_id") == "catalog/tools"
    assert _option_model_ids(context_update_selector) == {
        "catalog/structured",
        "catalog/tools",
        "catalog/function-tools",
    }
    enhancement_selector = selectors["character_enhancement"]
    assert _value(enhancement_selector, "selected_model_id") == "catalog/structured"
    assert _option_model_ids(enhancement_selector) == {
        "catalog/structured",
        "catalog/tools",
        "catalog/function-tools",
    }
    state_selector = selectors["state_memory"]
    assert _value(state_selector, "selected_model_id") == "catalog/function-tools"
    assert _option_model_ids(state_selector) == {
        "catalog/structured",
        "catalog/tools",
        "catalog/function-tools",
    }


def test_settings_character_enhancement_selector_inherits_context_update(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    repositories.upsert_provider_config(
        provider="catalog",
        enabled=True,
        has_api_key=True,
    )
    repositories.save_provider_model(
        provider="catalog",
        model_id="catalog/context",
        display_name="Context",
        capabilities=["structured_output"],
    )
    repositories.save_provider_model(
        provider="catalog",
        model_id="catalog/enhancement",
        display_name="Enhancement",
        capabilities=["structured_output"],
    )
    repositories.set_model_preference(
        task="context_update",
        provider="catalog",
        model_id="catalog/context",
    )

    inherited_model = settings.build_settings_model(
        repositories=repositories,
        providers=("catalog",),
    )
    inherited_selectors = {
        _value(selector, "task"): selector
        for selector in _list(
            _value(inherited_model, "task_model_selectors", "model_selectors")
        )
    }

    inherited_selector = inherited_selectors["character_enhancement"]
    assert _value(inherited_selector, "selected_model_id") == "catalog/context"
    assert _value(inherited_selector, "clearable") is False

    repositories.set_model_preference(
        task="character_enhancement",
        provider="catalog",
        model_id="catalog/enhancement",
    )

    direct_model = settings.build_settings_model(
        repositories=repositories,
        providers=("catalog",),
    )
    direct_selectors = {
        _value(selector, "task"): selector
        for selector in _list(
            _value(direct_model, "task_model_selectors", "model_selectors")
        )
    }
    direct_selector = direct_selectors["character_enhancement"]
    assert _value(direct_selector, "selected_model_id") == "catalog/enhancement"
    assert _value(direct_selector, "clearable") is True


def test_settings_model_automatic_media_mode_defaults_image_until_text_video_exists(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    repositories.set_app_setting("automatic_media_mode", "video")

    no_video_model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter",),
    )

    no_video_mode = _value(no_video_model, "automatic_media_mode")
    assert _value(no_video_mode, "setting_key") == "automatic_media_mode"
    assert _value(no_video_mode, "selected", "value") == "image"
    assert _value(no_video_mode, "options") == ("image",)

    repositories.upsert_provider_config(
        provider="openrouter",
        enabled=True,
        has_api_key=True,
    )
    repositories.save_provider_model(
        provider="openrouter",
        model_id="openrouter/text-video",
        display_name="Text Video",
        capabilities=["text_to_video"],
        context_window=32_000,
    )
    repositories.set_model_preference(
        task="video_generation",
        provider="openrouter",
        model_id="openrouter/text-video",
    )

    video_model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter",),
    )

    video_mode = _value(video_model, "automatic_media_mode")
    assert _value(video_mode, "setting_key") == "automatic_media_mode"
    assert _value(video_mode, "selected", "value") == "video"
    assert _value(video_mode, "options") == ("image", "video")


def test_text_task_model_selectors_exclude_legacy_broad_capability_rows(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    _seed_settings_data(repositories)
    repositories.upsert_provider_config(
        provider="capability-leak",
        enabled=True,
        has_api_key=True,
    )
    for model_id, display_name, capabilities in [
        ("capability-leak/real-chat", "Real Chat", ["chat"]),
        ("capability-leak/text", "Legacy Text", ["text"]),
        ("capability-leak/completion", "Legacy Completion", ["completion"]),
        ("capability-leak/video", "Video", ["video"]),
        ("capability-leak/tts", "TTS", ["tts"]),
        (
            "capability-leak/image-generation",
            "Image Generation",
            ["image_generation"],
        ),
    ]:
        repositories.save_provider_model(
            provider="capability-leak",
            model_id=model_id,
            display_name=display_name,
            capabilities=capabilities,
            context_window=32_000,
        )

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice", "capability-leak"),
    )

    selectors = {
        _value(selector, "task"): selector
        for selector in _list(_value(model, "task_model_selectors", "model_selectors"))
    }
    leaked_model_ids = {
        "capability-leak/text",
        "capability-leak/completion",
        "capability-leak/video",
        "capability-leak/tts",
        "capability-leak/image-generation",
    }
    for task in ("chat", "scenario_generation", "summarization"):
        model_ids = _option_model_ids(selectors[task])
        assert "capability-leak/real-chat" in model_ids
        assert model_ids.isdisjoint(leaked_model_ids)

    image_model_ids = _option_model_ids(selectors["image_generation"])
    assert "capability-leak/image-generation" in image_model_ids
    assert "capability-leak/real-chat" not in image_model_ids
    assert image_model_ids.isdisjoint(
        {
            "capability-leak/text",
            "capability-leak/completion",
            "capability-leak/video",
            "capability-leak/tts",
        }
    )


def test_settings_model_treats_blank_state_memory_preference_as_unset(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    _seed_settings_data(repositories)
    repositories.set_model_preference(
        task="state_memory",
        provider="",
        model_id="",
    )

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )

    selectors = {
        _value(selector, "task"): selector
        for selector in _list(_value(model, "task_model_selectors", "model_selectors"))
    }
    state_selector = selectors["state_memory"]
    assert _value(state_selector, "selected_provider") is None
    assert _value(state_selector, "selected_model_id") is None
    assert _value(state_selector, "warning") is None


def test_settings_model_exposes_summarization_and_image_controls(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    _seed_settings_data(repositories)

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )

    automatic_summarization = _value(model, "automatic_summarization")
    assert _value(automatic_summarization, "setting_key") == (
        "automatic_summarization_enabled"
    )
    assert _value(automatic_summarization, "enabled", "value") is True

    summarization_threshold = _value(
        model,
        "summarization_context_pressure_threshold",
        "summarization_threshold",
    )
    assert _value(summarization_threshold, "setting_key") == (
        "summarization_context_pressure_threshold"
    )
    assert _value(summarization_threshold, "value") == 0.75
    assert _value(summarization_threshold, "minimum", "min") == 0.10
    assert _value(summarization_threshold, "maximum", "max") == 1.00
    assert _value(summarization_threshold, "step") == 0.05

    summarization = _value(model, "summarization_visibility")
    assert _value(summarization, "setting_key") == "show_summarization_activity"
    assert _value(summarization, "enabled", "value") is True

    agentic_context_pipeline = _value(model, "agentic_context_pipeline")
    assert _value(agentic_context_pipeline, "setting_key") == (
        AGENTIC_CONTEXT_PIPELINE_SETTING
    )
    assert _value(agentic_context_pipeline, "enabled", "value") is True

    plan_first_narrator = _value(model, "plan_first_narrator")
    assert _value(plan_first_narrator, "setting_key") == (
        PLAN_FIRST_NARRATOR_SETTING
    )
    assert _value(plan_first_narrator, "enabled", "value") is True

    director_pressure = _value(model, "director_pressure")
    assert _value(director_pressure, "setting_key") == (
        DIRECTOR_PRESSURE_ENABLED_SETTING
    )
    assert _value(director_pressure, "enabled", "value") is True

    character_action_planning = _value(model, "character_action_planning")
    assert _value(character_action_planning, "setting_key") == (
        CHARACTER_ACTION_PLANNING_ENABLED_SETTING
    )
    assert _value(character_action_planning, "enabled", "value") is True

    character_action_planning_max_concurrency = _value(
        model,
        "character_action_planning_max_concurrency",
    )
    assert _value(character_action_planning_max_concurrency, "setting_key") == (
        CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY_SETTING
    )
    assert _value(character_action_planning_max_concurrency, "value") == 20
    assert _value(character_action_planning_max_concurrency, "minimum", "min") == 1
    assert _value(character_action_planning_max_concurrency, "maximum", "max") == 20
    assert _value(character_action_planning_max_concurrency, "step") == 1

    proactive_random_chance = _value(
        model,
        "character_text_proactive_random_chance",
    )
    assert _value(proactive_random_chance, "setting_key") == (
        CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING
    )
    assert _value(proactive_random_chance, "value") == (
        DEFAULT_CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_PERCENT
    )
    assert _value(proactive_random_chance, "minimum", "min") == 0
    assert _value(proactive_random_chance, "maximum", "max") == (
        MAX_CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_PERCENT
    )

    proactive_random_cooldown = _value(
        model,
        "character_text_proactive_random_cooldown",
    )
    assert _value(proactive_random_cooldown, "setting_key") == (
        CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING
    )
    assert _value(proactive_random_cooldown, "value") == (
        DEFAULT_CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_TURNS
    )
    assert _value(proactive_random_cooldown, "minimum", "min") == 0
    assert _value(proactive_random_cooldown, "maximum", "max") == (
        MAX_CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_TURNS
    )

    post_turn_inference_mode = _value(model, "post_turn_inference_mode")
    assert _value(post_turn_inference_mode, "setting_key") == (
        POST_TURN_INFERENCE_MODE_SETTING
    )
    assert _value(post_turn_inference_mode, "selected") == (
        POST_TURN_INFERENCE_MODE_PLAN_OWNED
    )
    assert _list(_value(post_turn_inference_mode, "options")) == [
        POST_TURN_INFERENCE_MODE_LEGACY,
        POST_TURN_INFERENCE_MODE_HYBRID,
        POST_TURN_INFERENCE_MODE_PLAN_OWNED,
    ]

    chat_history = _value(model, "chat_history")
    planner_player_messages = _value(chat_history, "planner_player_messages")
    assert _value(planner_player_messages, "setting_key") == (
        NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW_SETTING
    )
    assert _value(planner_player_messages, "value") == (
        DEFAULT_NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW
    )
    planner_narrator_messages = _value(chat_history, "planner_narrator_messages")
    assert _value(planner_narrator_messages, "setting_key") == (
        NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW_SETTING
    )
    assert _value(planner_narrator_messages, "value") == (
        DEFAULT_NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW
    )
    player_messages = _value(chat_history, "player_messages")
    assert _value(player_messages, "setting_key") == (
        RECENT_PLAYER_MESSAGE_WINDOW_SETTING
    )
    assert _value(player_messages, "value") == DEFAULT_RECENT_PLAYER_MESSAGE_WINDOW
    narrator_messages = _value(chat_history, "narrator_messages")
    assert _value(narrator_messages, "setting_key") == (
        RECENT_NARRATOR_MESSAGE_WINDOW_SETTING
    )
    assert _value(narrator_messages, "value") == DEFAULT_RECENT_NARRATOR_MESSAGE_WINDOW

    npc_knowledge_audit_mode = _value(model, "npc_knowledge_audit_mode")
    assert _value(npc_knowledge_audit_mode, "setting_key") == (
        NPC_KNOWLEDGE_AUDIT_MODE_SETTING
    )
    assert _value(npc_knowledge_audit_mode, "selected") == (
        NPC_KNOWLEDGE_AUDIT_MODE_SOFT_FAIL
    )
    assert _list(_value(npc_knowledge_audit_mode, "options")) == [
        NPC_KNOWLEDGE_AUDIT_MODE_SOFT_FAIL,
        NPC_KNOWLEDGE_AUDIT_MODE_HARD_FAIL,
    ]

    generated_phrase_denylist = _value(model, "generated_phrase_denylist")
    assert _value(generated_phrase_denylist, "setting_key") == (
        GENERATED_PHRASE_DENYLIST_SETTING
    )
    assert _value(generated_phrase_denylist, "value") == (
        "That's not nothing\nthat's actually everything"
    )

    save_generated_phrase_denylist = _value(
        model,
        "save_generated_phrase_denylist",
    )
    assert _value(save_generated_phrase_denylist, "setting_key") == (
        SAVE_GENERATED_PHRASE_DENYLIST_SETTING
    )
    assert _value(save_generated_phrase_denylist, "value") == ""

    pending_jobs_display = _value(model, "pending_jobs_display_mode")
    assert _value(pending_jobs_display, "setting_key") == "pending_jobs_display_mode"
    assert _value(pending_jobs_display, "selected") == "compact"
    assert _list(_value(pending_jobs_display, "options")) == [
        "compact",
        "expanded",
        "expanded_full",
    ]

    user_narration_guidance = _value(model, "user_narration_guidance")
    assert _value(user_narration_guidance, "setting_key") == (
        "user_narration_guidance"
    )
    assert _value(user_narration_guidance, "value") == ""

    automatic_image_generation = _value(model, "automatic_image_generation")
    assert _value(automatic_image_generation, "setting_key") == (
        "automatic_image_generation_enabled"
    )
    assert _value(automatic_image_generation, "enabled", "value") is False
    image_style_preset = _value(model, "image_style_preset")
    assert _value(image_style_preset, "setting_key") == "image_style_preset"
    assert _value(image_style_preset, "selected") == "realistic"
    assert _list(_value(image_style_preset, "options")) == list(
        EXPECTED_IMAGE_STYLE_PRESETS
    )

    image_frequency = _value(model, "image_frequency")
    assert _value(image_frequency, "setting_key") == "image_generation_frequency"
    assert _value(image_frequency, "value") == 3
    assert _value(image_frequency, "minimum", "min") == 0
    assert _value(image_frequency, "maximum", "max") is not None


def test_settings_model_exposes_persisted_summarization_and_image_controls(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A beacon tower.",
        player_role="Keeper",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting("automatic_summarization_enabled", False)
    repositories.set_app_setting("summarization_context_pressure_threshold", 0.4)
    repositories.set_app_setting("show_summarization_activity", False)
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=AGENTIC_CONTEXT_PIPELINE_SETTING,
        value=True,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=PLAN_FIRST_NARRATOR_SETTING,
        value=True,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=CHARACTER_ACTION_PLANNING_ENABLED_SETTING,
        value=True,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY_SETTING,
        value=7,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING,
        value=35,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING,
        value=9,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=DIRECTOR_PRESSURE_ENABLED_SETTING,
        value=True,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=NPC_KNOWLEDGE_AUDIT_MODE_SETTING,
        value=NPC_KNOWLEDGE_AUDIT_MODE_HARD_FAIL,
    )
    repositories.set_app_setting("automatic_image_generation_enabled", False)
    repositories.set_app_setting(
        save_image_style_preset_setting_key(save.id),
        "concept_art",
    )
    repositories.set_app_setting("image_generation_frequency", 0)

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
        active_save_id=save.id,
    )

    automatic_summarization = _value(model, "automatic_summarization")
    assert _value(automatic_summarization, "setting_key") == (
        "automatic_summarization_enabled"
    )
    assert _value(automatic_summarization, "enabled", "value") is False

    summarization_threshold = _value(
        model,
        "summarization_context_pressure_threshold",
        "summarization_threshold",
    )
    assert _value(summarization_threshold, "setting_key") == (
        "summarization_context_pressure_threshold"
    )
    assert _value(summarization_threshold, "value") == 0.4

    summarization = _value(model, "summarization_visibility")
    assert _value(summarization, "setting_key") == "show_summarization_activity"
    assert _value(summarization, "enabled", "value") is False

    agentic_context_pipeline = _value(model, "agentic_context_pipeline")
    assert _value(agentic_context_pipeline, "setting_key") == (
        AGENTIC_CONTEXT_PIPELINE_SETTING
    )
    assert _value(agentic_context_pipeline, "enabled", "value") is True

    plan_first_narrator = _value(model, "plan_first_narrator")
    assert _value(plan_first_narrator, "setting_key") == (
        PLAN_FIRST_NARRATOR_SETTING
    )
    assert _value(plan_first_narrator, "enabled", "value") is True

    director_pressure = _value(model, "director_pressure")
    assert _value(director_pressure, "setting_key") == (
        DIRECTOR_PRESSURE_ENABLED_SETTING
    )
    assert _value(director_pressure, "enabled", "value") is True

    character_action_planning = _value(model, "character_action_planning")
    assert _value(character_action_planning, "setting_key") == (
        CHARACTER_ACTION_PLANNING_ENABLED_SETTING
    )
    assert _value(character_action_planning, "enabled", "value") is True

    character_action_planning_max_concurrency = _value(
        model,
        "character_action_planning_max_concurrency",
    )
    assert _value(character_action_planning_max_concurrency, "setting_key") == (
        CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY_SETTING
    )
    assert _value(character_action_planning_max_concurrency, "value") == 7

    proactive_random_chance = _value(
        model,
        "character_text_proactive_random_chance",
    )
    assert _value(proactive_random_chance, "setting_key") == (
        CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING
    )
    assert _value(proactive_random_chance, "value") == 35

    proactive_random_cooldown = _value(
        model,
        "character_text_proactive_random_cooldown",
    )
    assert _value(proactive_random_cooldown, "setting_key") == (
        CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING
    )
    assert _value(proactive_random_cooldown, "value") == 9

    npc_knowledge_audit_mode = _value(model, "npc_knowledge_audit_mode")
    assert _value(npc_knowledge_audit_mode, "setting_key") == (
        NPC_KNOWLEDGE_AUDIT_MODE_SETTING
    )
    assert _value(npc_knowledge_audit_mode, "selected") == (
        NPC_KNOWLEDGE_AUDIT_MODE_HARD_FAIL
    )

    automatic_image_generation = _value(model, "automatic_image_generation")
    assert _value(automatic_image_generation, "setting_key") == (
        "automatic_image_generation_enabled"
    )
    assert _value(automatic_image_generation, "enabled", "value") is False
    image_style_preset = _value(model, "image_style_preset")
    assert _value(image_style_preset, "setting_key") == "image_style_preset"
    assert _value(image_style_preset, "selected") == "concept_art"

    image_frequency = _value(model, "image_frequency")
    assert _value(image_frequency, "setting_key") == "image_generation_frequency"
    assert _value(image_frequency, "value") == 0


def test_settings_model_defaults_unknown_image_style_preset_to_realistic(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A beacon tower.",
        player_role="Keeper",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(
        save_image_style_preset_setting_key(save.id),
        "oil painting",
    )

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
        active_save_id=save.id,
    )

    image_style_preset = _value(model, "image_style_preset")
    assert _value(image_style_preset, "selected") == "realistic"


def test_settings_model_reads_image_style_preset_for_active_save(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
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
    repositories.set_app_setting(
        save_image_style_preset_setting_key(first_save.id),
        "anime",
    )
    repositories.set_app_setting(
        save_image_style_preset_setting_key(second_save.id),
        "low_poly",
    )

    first_model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
        active_save_id=first_save.id,
    )
    second_model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
        active_save_id=second_save.id,
    )

    assert _value(_value(first_model, "image_style_preset"), "selected") == "anime"
    assert _value(_value(second_model, "image_style_preset"), "selected") == (
        "low_poly"
    )


def test_settings_model_exposes_generation_setting_controls(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    _seed_settings_data(repositories)
    repositories.set_app_setting("chat_temperature_enabled", True)
    repositories.set_app_setting("chat_temperature", 1.2)
    repositories.set_app_setting("chat_max_output_tokens_enabled", True)
    repositories.set_app_setting("chat_max_output_tokens", 2048)
    repositories.set_app_setting("image_dimension_preset", "landscape_1024x768")

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )

    chat_temperature = _value(model, "chat_temperature")
    assert _value(chat_temperature, "setting_key") == "chat_temperature"
    assert _value(chat_temperature, "enabled_setting_key") == (
        "chat_temperature_enabled"
    )
    assert _value(chat_temperature, "enabled") is True
    assert _value(chat_temperature, "supported") is True
    assert _value(chat_temperature, "value") == 1.2

    max_tokens = _value(model, "chat_max_output_tokens")
    assert _value(max_tokens, "setting_key") == "chat_max_output_tokens"
    assert _value(max_tokens, "enabled") is True
    assert _value(max_tokens, "supported") is True
    assert _value(max_tokens, "value") == 2048

    image_dimensions = _value(model, "image_dimension_preset")
    assert _value(image_dimensions, "setting_key") == "image_dimension_preset"
    assert _value(image_dimensions, "selected") == "landscape_1024x768"
    assert _value(image_dimensions, "supported") is True
    assert _list(_value(image_dimensions, "options")) == list(
        EXPECTED_IMAGE_DIMENSION_PRESETS
    )


def test_settings_model_defaults_image_dimension_preset_to_square(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    _seed_settings_data(repositories)

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )

    image_dimensions = _value(model, "image_dimension_preset")
    assert _value(image_dimensions, "setting_key") == "image_dimension_preset"
    assert _value(image_dimensions, "selected") == "square_1024x1024"
    assert _value(image_dimensions, "supported") is True
    assert _list(_value(image_dimensions, "options")) == list(
        EXPECTED_IMAGE_DIMENSION_PRESETS
    )


def test_settings_model_marks_generation_settings_unsupported_without_model_metadata(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    _seed_settings_data(repositories)
    repositories.save_provider_model(
        provider="openrouter",
        model_id="openrouter/no-params",
        display_name="No Params",
        capabilities=["chat"],
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/no-params",
    )

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )

    assert _value(model.chat_temperature, "supported") is False
    assert _value(model.chat_max_output_tokens, "supported") is False


def test_settings_model_exposes_manual_confirmation_controls(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )

    confirmation = _value(model, "manual_confirmation")
    assert _value(_value(confirmation, "memories"), "setting_key") == (
        "manual_confirmation_memories_enabled"
    )
    assert _value(_value(confirmation, "memories"), "enabled", "value") is False
    assert _value(_value(confirmation, "character_registry"), "setting_key") == (
        "manual_confirmation_character_registry_enabled"
    )
    assert (
        _value(_value(confirmation, "character_registry"), "enabled", "value")
        is False
    )
    assert _value(_value(confirmation, "state_changes"), "setting_key") == (
        "manual_confirmation_state_changes_enabled"
    )
    assert _value(_value(confirmation, "state_changes"), "enabled", "value") is False

    repositories.set_app_setting("manual_confirmation_memories_enabled", True)
    repositories.set_app_setting(
        "manual_confirmation_character_registry_enabled",
        True,
    )
    repositories.set_app_setting("manual_confirmation_state_changes_enabled", True)

    persisted_model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )
    persisted_confirmation = _value(persisted_model, "manual_confirmation")
    persisted_memories = _value(persisted_confirmation, "memories")
    assert _value(persisted_memories, "enabled", "value") is True
    assert (
        _value(
            _value(persisted_confirmation, "character_registry"),
            "enabled",
            "value",
        )
        is True
    )
    assert (
        _value(_value(persisted_confirmation, "state_changes"), "enabled", "value")
        is True
    )


def test_settings_model_exposes_chat_fallback_toggle(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )

    fallback_toggle = _value(model, "chat_fallback")
    assert _value(fallback_toggle, "setting_key") == (
        "chat_fallback_enabled"
    )
    assert _value(fallback_toggle, "enabled", "value") is False

    repositories.set_app_setting("chat_fallback_enabled", True)

    persisted_model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )
    persisted_toggle = _value(persisted_model, "chat_fallback")
    assert _value(persisted_toggle, "setting_key") == (
        "chat_fallback_enabled"
    )
    assert _value(persisted_toggle, "enabled", "value") is True


def test_settings_model_exposes_tool_call_fallback_toggle(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )

    fallback_toggle = _value(model, "tool_call_fallback")
    assert _value(fallback_toggle, "setting_key") == (
        "tool_call_fallback_enabled"
    )
    assert _value(fallback_toggle, "enabled", "value") is False

    repositories.set_app_setting("tool_call_fallback_enabled", True)

    persisted_model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )
    persisted_toggle = _value(persisted_model, "tool_call_fallback")
    assert _value(persisted_toggle, "setting_key") == (
        "tool_call_fallback_enabled"
    )
    assert _value(persisted_toggle, "enabled", "value") is True


def test_settings_model_exposes_image_fallback_and_venice_safe_mode_toggles(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )

    image_fallback = _value(model, "image_fallback")
    assert _value(image_fallback, "setting_key") == (
        "image_fallback_enabled"
    )
    assert _value(image_fallback, "enabled", "value") is False
    venice_safe_mode = _value(model, "venice_image_safe_mode")
    assert _value(venice_safe_mode, "setting_key") == "venice_image_safe_mode"
    assert _value(venice_safe_mode, "enabled", "value") is True

    repositories.set_app_setting("image_fallback_enabled", True)
    repositories.set_app_setting("venice_image_safe_mode", False)

    persisted_model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )
    persisted_image_fallback = _value(
        persisted_model,
        "image_fallback",
    )
    assert _value(persisted_image_fallback, "setting_key") == (
        "image_fallback_enabled"
    )
    assert _value(persisted_image_fallback, "enabled", "value") is True
    persisted_venice_safe_mode = _value(persisted_model, "venice_image_safe_mode")
    assert _value(persisted_venice_safe_mode, "setting_key") == (
        "venice_image_safe_mode"
    )
    assert _value(persisted_venice_safe_mode, "enabled", "value") is False


def test_settings_model_exposes_debug_logging_toggle_default_false_and_persisted_true(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )

    debug_logging = _value(model, "debug_logging")
    assert _value(debug_logging, "setting_key") == "debug_logging_enabled"
    assert _value(debug_logging, "enabled", "value") is False

    repositories.set_app_setting("debug_logging_enabled", True)

    persisted_model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )
    persisted_debug_logging = _value(persisted_model, "debug_logging")
    assert _value(persisted_debug_logging, "setting_key") == ("debug_logging_enabled")
    assert _value(persisted_debug_logging, "enabled", "value") is True


def test_settings_model_defaults_missing_image_generation_frequency_to_runtime_default(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )

    image_frequency = _value(model, "image_frequency")
    assert _value(image_frequency, "setting_key") == "image_generation_frequency"
    assert _value(image_frequency, "value") == 3


def test_settings_model_omits_diagnostics_entries_for_failures(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    save_id = _seed_settings_data(repositories)
    repositories.upsert_provider_config(
        provider="diagnostic-provider",
        enabled=True,
        has_api_key=True,
        last_error=f"authentication failed for {_SENTINEL_BEARER}",
    )
    repositories.upsert_provider_config(
        provider="api-key-provider",
        enabled=True,
        has_api_key=True,
        last_error=f"model refresh failed with api_key: {_SENTINEL_SECRET}",
    )
    repositories.upsert_provider_config(
        provider="jsonish-provider",
        enabled=True,
        has_api_key=True,
        last_error=(f'metadata sync failed with {{"api_key":"{_SENTINEL_SECRET}"}}'),
    )
    job = repositories.create_job(
        save_id=save_id,
        type="image_generation",
        status="running",
        payload={"provider": "venice", "api_key": _SENTINEL_SECRET},
    )
    repositories.update_job(
        job.id,
        status="failed",
        error=f"image provider rejected {_SENTINEL_TOKEN_PARAM} during generation",
        result={
            "attempt_count": 3,
            "max_attempts": 3,
            "retry_attempts": [
                {
                    "attempt": 1,
                    "duration_ms": 10,
                    "error_category": "rate_limited",
                    "http_status": 429,
                    "unsafe": _SENTINEL_SECRET,
                },
                {
                    "attempt": 2,
                    "duration_ms": 20,
                    "error_category": "rate_limited",
                    "http_status": 429,
                },
                {
                    "attempt": 3,
                    "duration_ms": 30,
                    "error_category": "rate_limited",
                    "http_status": 429,
                },
            ],
        },
    )
    bearer_job = repositories.create_job(
        save_id=save_id,
        type="chat",
        status="running",
        payload={"provider": "openrouter"},
    )
    repositories.update_job(
        bearer_job.id,
        status="failed",
        error=f"chat provider rejected {_SENTINEL_UPPER_BEARER} during reply",
    )

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )

    assert _value(model, "diagnostics", default=None) is None
    assert "model refresh failed" not in repr(model)
    assert "metadata sync failed" not in repr(model)
    assert "during generation" not in repr(model)
    assert "during reply" not in repr(model)
    assert _SENTINEL_SECRET not in repr(model)


def test_settings_model_omits_deprecated_fallback_disabled_diagnostics(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    _seed_settings_data(repositories)
    repositories.set_model_preference(
        task="structured_output_fallback",
        provider="openrouter",
        model_id="openrouter/context-search",
    )
    repositories.set_app_setting("structured_output_fallback_enabled", False)

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
    )

    assert _value(model, "diagnostics", default=None) is None
    assert "Structured output fallback model is configured" not in repr(model)
    diagnostics = _list(settings.configuration_diagnostics(repositories))
    assert not any(
        _value(entry, "kind", "type") == "configuration"
        and "fallback is disabled" in _value(entry, "error")
        for entry in diagnostics
    )


def test_settings_model_includes_supplied_secret_storage_warning(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    warning = "Secret storage health check failed"

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
        secret_storage_warning=warning,
    )

    assert _value(model, "secret_storage_warning") == warning
    assert _value(model, "diagnostics", default=None) is None


def test_settings_model_omits_log_file_path_diagnostic(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _import_settings_without_gtk(monkeypatch)
    _seed_settings_data(repositories)
    log_file_path = tmp_path / "state" / ".." / "state" / "logs" / "bragi.log"

    model = settings.build_settings_model(
        repositories=repositories,
        providers=("openrouter", "venice"),
        log_file_path=log_file_path,
    )

    assert _value(model, "diagnostics", default=None) is None
    assert str(log_file_path.resolve()) not in repr(model)


def _import_settings_without_gtk(monkeypatch: MonkeyPatch) -> Any:
    original_import = builtins.__import__

    def import_without_gtk(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "gi" or name.startswith("gi."):
            raise AssertionError(
                "bragi.application.settings must not import GTK/PyGObject"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_gtk)
    sys.modules.pop("bragi.application.settings", None)
    return importlib.import_module("bragi.application.settings")


def _seed_settings_data(repositories: PersistenceRepositories) -> str:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.upsert_provider_config(
        provider="openrouter",
        enabled=True,
        has_api_key=True,
        last_error=None,
    )
    repositories.upsert_provider_config(
        provider="venice",
        enabled=True,
        has_api_key=False,
        last_error="authentication_failed",
    )
    for provider, model_id, display_name, capabilities, supported_parameters in [
        (
            "openrouter",
            "openrouter/chat-fast",
            "Chat Fast",
            ["chat"],
            ["temperature", "max_output_tokens"],
        ),
        (
            "openrouter",
            "openrouter/scenario-writer",
            "Scenario Writer",
            ["chat"],
            ["temperature", "max_output_tokens"],
        ),
        (
            "openrouter",
            "openrouter/context-search",
            "Context Search",
            ["structured_output"],
            [],
        ),
        (
            "openrouter",
            "openrouter/summary",
            "Summary",
            ["chat"],
            ["temperature", "max_output_tokens"],
        ),
        ("openrouter", "openrouter/memory", "Memory", ["structured_output"], []),
        (
            "openrouter",
            "openrouter/image",
            "OpenRouter Image",
            ["image_generation"],
            ["image_dimensions"],
        ),
        (
            "openrouter",
            "openrouter/image-edit",
            "OpenRouter Image Edit",
            ["image_to_image"],
            ["image_dimensions"],
        ),
        (
            "openrouter",
            "openrouter/text-video",
            "OpenRouter Text Video",
            ["text_to_video"],
            [],
        ),
        (
            "openrouter",
            "openrouter/image-video",
            "OpenRouter Image Video",
            ["image_plus_text_to_video"],
            [],
        ),
        (
            "openrouter",
            "openrouter/vision",
            "OpenRouter Vision",
            ["chat", "vision"],
            ["temperature", "max_output_tokens"],
        ),
        (
            "venice",
            "venice/image",
            "Venice Image",
            ["image_generation"],
            ["image_dimensions", "image_safe_mode"],
        ),
    ]:
        repositories.save_provider_model(
            provider=provider,
            model_id=model_id,
            display_name=display_name,
            capabilities=capabilities,
            supported_parameters=supported_parameters,
            context_window=128_000,
        )
    repositories.mark_missing_provider_models_unavailable(
        provider="openrouter",
        available_model_ids={
            "openrouter/chat-fast",
            "openrouter/scenario-writer",
            "openrouter/context-search",
            "openrouter/summary",
            "openrouter/image",
            "openrouter/image-edit",
            "openrouter/text-video",
            "openrouter/image-video",
            "openrouter/vision",
        },
    )
    for task, provider, model_id in [
        ("chat", "openrouter", "openrouter/chat-fast"),
        ("scenario_generation", "openrouter", "openrouter/scenario-writer"),
        ("context_search", "openrouter", "openrouter/context-search"),
        ("summarization", "openrouter", "openrouter/summary"),
        ("state_memory", "openrouter", "openrouter/memory"),
        ("context_update", "openrouter", "openrouter/context-search"),
        ("context_cleanup", "openrouter", "openrouter/context-search"),
        ("state_pruning", "openrouter", "openrouter/context-search"),
        ("scenario_evolution", "openrouter", "openrouter/context-search"),
        ("image_generation", "venice", "venice/image"),
        ("image_to_image_generation", "openrouter", "openrouter/image-edit"),
        ("video_generation", "openrouter", "openrouter/text-video"),
        ("image_animation", "openrouter", "openrouter/image-video"),
        ("character_image_description", "openrouter", "openrouter/vision"),
    ]:
        repositories.set_model_preference(
            task=task,
            provider=provider,
            model_id=model_id,
        )
    service = SettingsService(
        repositories=repositories,
        providers={},
        secret_store=InMemorySecretStore(),
    )
    service.set_local_setting("show_summarization_activity", True)
    service.set_local_setting("image_generation_frequency", 3)
    return save.id


def _selected_model_id(selector: object) -> str:
    selected_model_id = _value(selector, "selected_model_id")
    assert isinstance(selected_model_id, str)
    return selected_model_id


def _option_model_ids(selector: object) -> set[str]:
    return {
        _value(option, "model_id")
        for option in _list(_value(selector, "options", "models"))
    }


def _assert_scenario_generation_selector(selector: object) -> None:
    assert _value(selector, "task") == "scenario_generation"
    assert _value(selector, "selected_provider") == "openrouter"
    assert _value(selector, "selected_model_id") == "openrouter/scenario-writer"
    assert _value(selector, "selected_available") is True
    assert _value(selector, "warning") is None

    options = _list(_value(selector, "options", "models"))
    assert options
    assert all("chat" in _value(option, "capabilities") for option in options)
    assert any(
        _value(option, "provider") == "openrouter"
        and _value(option, "model_id") == "openrouter/scenario-writer"
        and _value(option, "available") is True
        for option in options
    )


def _expected_roleplay_selector_tasks(roleplay_type: str, chat_task: str) -> list[str]:
    return [
        chat_task,
        "scenario_generation",
        f"{roleplay_type}_context_search",
        f"{roleplay_type}_summarization",
        f"{roleplay_type}_state_memory",
        f"{roleplay_type}_context_update",
        f"{roleplay_type}_character_enhancement",
        f"{roleplay_type}_fact_observation",
        f"{roleplay_type}_memory_curation",
        f"{roleplay_type}_response_planning",
        f"{roleplay_type}_response_verification",
        f"{roleplay_type}_director_pressure",
        f"{roleplay_type}_action_choice_generation",
        f"{roleplay_type}_character_presence_assessment",
        f"{roleplay_type}_character_intent_planning",
        f"{roleplay_type}_dating_route_profile",
        f"{roleplay_type}_character_action_planning",
        f"{roleplay_type}_character_registry_maintenance",
        f"{roleplay_type}_context_cleanup_scan",
        f"{roleplay_type}_context_cleanup_actions",
        f"{roleplay_type}_guided_context_cleanup",
        f"{roleplay_type}_context_cleanup",
        f"{roleplay_type}_state_pruning",
        f"{roleplay_type}_scenario_evolution",
        f"{roleplay_type}_npc_knowledge_audit",
        f"{roleplay_type}_image_prompt",
        f"{roleplay_type}_image_generation",
        f"{roleplay_type}_image_to_image_generation",
        f"{roleplay_type}_scene_image_edit_generation",
        f"{roleplay_type}_character_image_edit_generation",
        f"{roleplay_type}_text_message_image_edit_generation",
        f"{roleplay_type}_video_generation",
        f"{roleplay_type}_image_animation",
        f"{roleplay_type}_narrator_fallback",
        f"{roleplay_type}_chat_fallback",
        f"{roleplay_type}_structured_output_fallback",
        f"{roleplay_type}_tool_call_fallback",
        f"{roleplay_type}_image_fallback",
        f"{roleplay_type}_image_edit_fallback",
        f"{roleplay_type}_video_fallback",
    ]


def _list(value: object) -> list[Any]:
    assert isinstance(value, list | tuple), f"Expected sequence, got {value!r}"
    return list(value)


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
