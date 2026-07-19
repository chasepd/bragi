from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.model_preferences import (
    ACTION_CHOICE_GENERATION_PURPOSE,
    CHARACTER_ENHANCEMENT_PURPOSE,
    CHARACTER_IMAGE_EDIT_PURPOSE,
    CHOOSE_YOUR_OWN_ADVENTURE_TYPE,
    DATING_SIM_TYPE,
    EXTRACTION_TOOL_PRIMARY_MODEL_ID,
    EXTRACTION_TOOL_PRIMARY_PROVIDER,
    FANTASY_ROLEPLAY_TYPE,
    FIRST_CONTACT_EXPLORATION_TYPE,
    FULL_ROLEPLAY_TYPE,
    INVESTIGATION_MYSTERY_TYPE,
    POLITICAL_INTRIGUE_TYPE,
    ROLEPLAY_SHARED_MODE_SETTING,
    ROLEPLAY_SHARED_TYPE,
    SAVE_MODEL_OVERRIDES_SETTING,
    SCENE_IMAGE_EDIT_PURPOSE,
    SCIENCE_FICTION_ROLEPLAY_TYPE,
    SURVIVAL_EXPEDITION_TYPE,
    TEXT_MESSAGE_IMAGE_EDIT_PURPOSE,
    TIME_LOOP_TYPE,
    character_enhancement_model_preference,
    clear_save_model_override_preference,
    image_edit_model_preference,
    model_preference_for_selector,
    narrator_fallback_model_preference,
    roleplay_model_preference,
    roleplay_model_preference_with_fallbacks,
    roleplay_model_task,
    sanitize_save_model_overrides,
    save_model_override_preference,
    scenario_generation_model_preference,
    scenario_generation_section_model_task,
    set_save_model_override_preference,
    shared_roleplay_models_enabled,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_roleplay_model_task_maps_shared_and_per_type_purposes() -> None:
    assert roleplay_model_task(
        roleplay_type=ROLEPLAY_SHARED_TYPE,
        purpose="context_update",
    ) == "context_update"
    assert roleplay_model_task(
        roleplay_type=FULL_ROLEPLAY_TYPE,
        purpose="chat",
    ) == "chat_full_roleplay"
    assert roleplay_model_task(
        roleplay_type=FANTASY_ROLEPLAY_TYPE,
        purpose="chat",
    ) == "chat_fantasy_roleplay"
    assert roleplay_model_task(
        roleplay_type=SCIENCE_FICTION_ROLEPLAY_TYPE,
        purpose="chat",
    ) == "chat_science_fiction_roleplay"
    assert roleplay_model_task(
        roleplay_type=FIRST_CONTACT_EXPLORATION_TYPE,
        purpose="chat",
    ) == "chat_first_contact_exploration"
    assert roleplay_model_task(
        roleplay_type=SURVIVAL_EXPEDITION_TYPE,
        purpose="chat",
    ) == "chat_survival_expedition"
    assert roleplay_model_task(
        roleplay_type=TIME_LOOP_TYPE,
        purpose="chat",
    ) == "chat_time_loop"
    assert roleplay_model_task(
        roleplay_type=INVESTIGATION_MYSTERY_TYPE,
        purpose="chat",
    ) == "chat_investigation_mystery"
    assert roleplay_model_task(
        roleplay_type=POLITICAL_INTRIGUE_TYPE,
        purpose="chat",
    ) == "chat_political_intrigue"
    assert roleplay_model_task(
        roleplay_type=FULL_ROLEPLAY_TYPE,
        purpose="context_update",
    ) == "full_roleplay_context_update"
    assert roleplay_model_task(
        roleplay_type=FULL_ROLEPLAY_TYPE,
        purpose=CHARACTER_ENHANCEMENT_PURPOSE,
    ) == "full_roleplay_character_enhancement"
    assert roleplay_model_task(
        roleplay_type=FULL_ROLEPLAY_TYPE,
        purpose="narrator_fallback",
    ) == "full_roleplay_narrator_fallback"
    assert roleplay_model_task(
        roleplay_type=FULL_ROLEPLAY_TYPE,
        purpose="tool_call_fallback",
    ) == "full_roleplay_tool_call_fallback"
    assert roleplay_model_task(
        roleplay_type=DATING_SIM_TYPE,
        purpose="chat",
    ) == "chat_dating_sim"
    assert roleplay_model_task(
        roleplay_type=DATING_SIM_TYPE,
        purpose="context_update",
    ) == "dating_sim_context_update"
    assert roleplay_model_task(
        roleplay_type=CHOOSE_YOUR_OWN_ADVENTURE_TYPE,
        purpose="chat",
    ) == "chat_choose_your_own_adventure"
    assert roleplay_model_task(
        roleplay_type=CHOOSE_YOUR_OWN_ADVENTURE_TYPE,
        purpose="action_choice_generation",
    ) == "choose_your_own_adventure_action_choice_generation"


def test_save_model_overrides_drop_retired_character_interaction_tasks() -> None:
    config = {
        "provider": "openrouter",
        "model_id": "openai/gpt-5-mini",
    }
    thinking_config = {**config, "level": "low"}

    assert sanitize_save_model_overrides(
        {
            "preferences": {
                "chat_character_interaction": config,
                "character_interaction_context_update": config,
                "character_image_description": config,
                "dating_sim_context_update": config,
            },
            "thinking": {
                "chat_character_interaction": thinking_config,
                "character_interaction_context_update": thinking_config,
                "character_image_description": thinking_config,
                "dating_sim_context_update": thinking_config,
            },
        }
    ) == {
        "preferences": {
            "character_image_description": config,
            "dating_sim_context_update": config,
        },
        "thinking": {
            "character_image_description": thinking_config,
            "dating_sim_context_update": thinking_config,
        },
    }


def test_scenario_generation_section_model_task_maps_supported_sections() -> None:
    assert (
        scenario_generation_section_model_task("worldbuilding")
        == "scenario_generation_section_worldbuilding"
    )
    assert (
        scenario_generation_section_model_task("player_character_profile")
        == "scenario_generation_section_player_character_profile"
    )
    assert (
        scenario_generation_section_model_task("romance_options")
        == "scenario_generation_section_romance_options"
    )
    assert (
        scenario_generation_section_model_task("choice_style")
        == "scenario_generation_section_choice_style"
    )
    assert (
        scenario_generation_section_model_task("magic_system")
        == "scenario_generation_section_magic_system"
    )
    assert (
        scenario_generation_section_model_task("technology_level")
        == "scenario_generation_section_technology_level"
    )
    assert (
        scenario_generation_section_model_task("expedition_goal")
        == "scenario_generation_section_expedition_goal"
    )
    assert (
        scenario_generation_section_model_task("resource_inventory")
        == "scenario_generation_section_resource_inventory"
    )
    assert (
        scenario_generation_section_model_task("obligations_and_favors")
        == "scenario_generation_section_obligations_and_favors"
    )
    with pytest.raises(ValueError):
        scenario_generation_section_model_task("unknown_section")


def test_roleplay_model_preference_defaults_to_shared_base_tasks(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save(repositories, scenario_type=FULL_ROLEPLAY_TYPE)
    repositories.set_model_preference(
        task="context_update",
        provider="openrouter",
        model_id="openrouter/base-context",
    )
    repositories.set_model_preference(
        task="full_roleplay_context_update",
        provider="openrouter",
        model_id="openrouter/full-context",
    )

    assert shared_roleplay_models_enabled(repositories) is True
    preference = roleplay_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose="context_update",
    )

    assert preference is not None
    assert preference.provider == "openrouter"
    assert preference.model_id == "openrouter/base-context"


def test_save_model_override_wins_for_one_save_only(
    repositories: PersistenceRepositories,
) -> None:
    first_save_id = _create_save(repositories, scenario_type=FULL_ROLEPLAY_TYPE)
    second_save_id = _create_save(repositories, scenario_type=FULL_ROLEPLAY_TYPE)
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/server-chat",
    )

    set_save_model_override_preference(
        repositories,
        save_id=first_save_id,
        task="chat",
        provider="venice",
        model_id="venice/save-chat",
    )

    first_preference = roleplay_model_preference(
        repositories=repositories,
        save_id=first_save_id,
        purpose="chat",
    )
    second_preference = roleplay_model_preference(
        repositories=repositories,
        save_id=second_save_id,
        purpose="chat",
    )

    assert first_preference is not None
    assert first_preference.provider == "venice"
    assert first_preference.model_id == "venice/save-chat"
    assert second_preference is not None
    assert second_preference.provider == "openrouter"
    assert second_preference.model_id == "openrouter/server-chat"


def test_save_model_override_uses_roleplay_specific_task_before_server_default(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save(repositories, scenario_type=FULL_ROLEPLAY_TYPE)
    repositories.set_app_setting(ROLEPLAY_SHARED_MODE_SETTING, False)
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/server-chat",
    )
    repositories.set_model_preference(
        task="chat_full_roleplay",
        provider="openrouter",
        model_id="openrouter/server-full-chat",
    )

    set_save_model_override_preference(
        repositories,
        save_id=save_id,
        task="chat_full_roleplay",
        provider="venice",
        model_id="venice/save-full-chat",
    )

    preference = roleplay_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose="chat",
    )

    assert preference is not None
    assert preference.provider == "venice"
    assert preference.model_id == "venice/save-full-chat"


def test_save_model_override_storage_cleans_empty_setting(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save(repositories, scenario_type=FULL_ROLEPLAY_TYPE)

    set_save_model_override_preference(
        repositories,
        save_id=save_id,
        task="chat",
        provider="venice",
        model_id="venice/save-chat",
    )
    clear_save_model_override_preference(
        repositories,
        save_id=save_id,
        task="chat",
    )

    assert (
        save_model_override_preference(repositories, save_id=save_id, task="chat")
        is None
    )
    assert (
        repositories.get_scoped_setting(
            scope="save",
            scope_id=save_id,
            key=SAVE_MODEL_OVERRIDES_SETTING,
        )
        is None
    )


def test_model_preference_for_selector_can_read_save_override(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save(repositories, scenario_type=FULL_ROLEPLAY_TYPE)
    repositories.set_model_preference(
        task="context_update",
        provider="openrouter",
        model_id="openrouter/server-context",
    )
    set_save_model_override_preference(
        repositories,
        save_id=save_id,
        task="character_enhancement",
        provider="venice",
        model_id="venice/save-character",
    )

    preference = model_preference_for_selector(
        repositories,
        "character_enhancement",
        save_id=save_id,
    )
    inherited = model_preference_for_selector(
        repositories,
        "character_enhancement",
    )

    assert preference is not None
    assert preference.provider == "venice"
    assert preference.model_id == "venice/save-character"
    assert inherited is not None
    assert inherited.model_id == "openrouter/server-context"


def test_roleplay_model_preference_uses_specific_task_then_base_when_not_shared(
    repositories: PersistenceRepositories,
) -> None:
    full_save_id = _create_save(repositories, scenario_type=FULL_ROLEPLAY_TYPE)
    character_save_id = _create_save(
        repositories,
        scenario_type=DATING_SIM_TYPE,
    )
    repositories.set_app_setting(ROLEPLAY_SHARED_MODE_SETTING, False)
    repositories.set_model_preference(
        task="context_update",
        provider="openrouter",
        model_id="openrouter/base-context",
    )
    repositories.set_model_preference(
        task="full_roleplay_context_update",
        provider="openrouter",
        model_id="openrouter/full-context",
    )

    full_preference = roleplay_model_preference(
        repositories=repositories,
        save_id=full_save_id,
        purpose="context_update",
    )
    character_preference = roleplay_model_preference(
        repositories=repositories,
        save_id=character_save_id,
        purpose="context_update",
    )

    assert full_preference is not None
    assert full_preference.model_id == "openrouter/full-context"
    assert character_preference is not None
    assert character_preference.model_id == "openrouter/base-context"


def test_roleplay_model_preference_with_fallbacks_prefers_new_then_legacy_task(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save(repositories, scenario_type=FULL_ROLEPLAY_TYPE)
    repositories.set_model_preference(
        task="character_action_planning",
        provider="openrouter",
        model_id="openrouter/legacy-character",
    )

    legacy_preference = roleplay_model_preference_with_fallbacks(
        repositories=repositories,
        save_id=save_id,
        purposes=(ACTION_CHOICE_GENERATION_PURPOSE, "character_action_planning"),
    )

    assert legacy_preference is not None
    assert legacy_preference.model_id == "openrouter/legacy-character"

    repositories.set_model_preference(
        task=ACTION_CHOICE_GENERATION_PURPOSE,
        provider="openrouter",
        model_id="openrouter/action-choice",
    )

    new_preference = roleplay_model_preference_with_fallbacks(
        repositories=repositories,
        save_id=save_id,
        purposes=(ACTION_CHOICE_GENERATION_PURPOSE, "character_action_planning"),
    )

    assert new_preference is not None
    assert new_preference.model_id == "openrouter/action-choice"


def test_image_edit_model_preference_uses_flow_override_before_default(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save(repositories, scenario_type=DATING_SIM_TYPE)
    repositories.set_model_preference(
        task="image_to_image_generation",
        provider="openrouter",
        model_id="openrouter/default-edit",
    )
    repositories.set_model_preference(
        task=SCENE_IMAGE_EDIT_PURPOSE,
        provider="openrouter",
        model_id="openrouter/scene-edit",
    )
    repositories.set_model_preference(
        task=CHARACTER_IMAGE_EDIT_PURPOSE,
        provider="openrouter",
        model_id="openrouter/character-edit",
    )
    repositories.set_model_preference(
        task=TEXT_MESSAGE_IMAGE_EDIT_PURPOSE,
        provider="openrouter",
        model_id="openrouter/text-message-edit",
    )

    scene = image_edit_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose=SCENE_IMAGE_EDIT_PURPOSE,
    )
    character = image_edit_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose=CHARACTER_IMAGE_EDIT_PURPOSE,
    )
    text_message = image_edit_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose=TEXT_MESSAGE_IMAGE_EDIT_PURPOSE,
    )

    assert scene is not None
    assert scene.model_id == "openrouter/scene-edit"
    assert character is not None
    assert character.model_id == "openrouter/character-edit"
    assert text_message is not None
    assert text_message.model_id == "openrouter/text-message-edit"


def test_image_edit_model_preference_ignores_retired_type_override(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save(repositories, scenario_type="character_interaction")
    repositories.set_app_setting(ROLEPLAY_SHARED_MODE_SETTING, False)
    repositories.set_model_preference(
        task="image_to_image_generation",
        provider="openrouter",
        model_id="openrouter/shared-edit",
    )
    repositories.set_model_preference(
        task="character_interaction_image_to_image_generation",
        provider="openrouter",
        model_id="openrouter/legacy-character-edit",
    )

    scene = image_edit_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose=SCENE_IMAGE_EDIT_PURPOSE,
    )
    character = image_edit_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose=CHARACTER_IMAGE_EDIT_PURPOSE,
    )
    text_message = image_edit_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose=TEXT_MESSAGE_IMAGE_EDIT_PURPOSE,
    )

    assert scene is not None
    assert scene.model_id == "openrouter/shared-edit"
    assert character is not None
    assert character.model_id == "openrouter/shared-edit"
    assert text_message is not None
    assert text_message.model_id == "openrouter/shared-edit"


@pytest.mark.parametrize("purpose", ["state_memory", "context_update"])
def test_roleplay_model_preference_defaults_extraction_to_recommended_tool_model(
    repositories: PersistenceRepositories,
    purpose: str,
) -> None:
    save_id = _create_save(repositories, scenario_type=FULL_ROLEPLAY_TYPE)
    repositories.save_provider_model(
        provider=EXTRACTION_TOOL_PRIMARY_PROVIDER,
        model_id=EXTRACTION_TOOL_PRIMARY_MODEL_ID,
        display_name="DeepSeek V4 Flash",
        capabilities=["tool_calling"],
    )

    preference = roleplay_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose=purpose,
    )

    assert preference is not None
    assert preference.provider == EXTRACTION_TOOL_PRIMARY_PROVIDER
    assert preference.model_id == EXTRACTION_TOOL_PRIMARY_MODEL_ID


def test_roleplay_model_preference_ignores_recommended_extraction_without_tools(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save(repositories, scenario_type=FULL_ROLEPLAY_TYPE)
    repositories.save_provider_model(
        provider=EXTRACTION_TOOL_PRIMARY_PROVIDER,
        model_id=EXTRACTION_TOOL_PRIMARY_MODEL_ID,
        display_name="DeepSeek V4 Flash",
        capabilities=["chat"],
    )

    preference = roleplay_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose="state_memory",
    )

    assert preference is None


def test_character_enhancement_model_preference_prefers_separate_override(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save(repositories, scenario_type=FULL_ROLEPLAY_TYPE)
    repositories.set_model_preference(
        task="context_update",
        provider="openrouter",
        model_id="openrouter/context-update",
    )
    repositories.set_model_preference(
        task=CHARACTER_ENHANCEMENT_PURPOSE,
        provider="openrouter",
        model_id="openrouter/profile-enhancer",
    )

    preference = character_enhancement_model_preference(
        repositories=repositories,
        save_id=save_id,
    )

    assert preference is not None
    assert preference.provider == "openrouter"
    assert preference.model_id == "openrouter/profile-enhancer"


def test_character_enhancement_model_preference_falls_back_to_context_update(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save(repositories, scenario_type=FULL_ROLEPLAY_TYPE)
    repositories.set_app_setting(ROLEPLAY_SHARED_MODE_SETTING, False)
    repositories.set_model_preference(
        task="context_update",
        provider="openrouter",
        model_id="openrouter/shared-context",
    )
    repositories.set_model_preference(
        task="full_roleplay_context_update",
        provider="openrouter",
        model_id="openrouter/full-context",
    )

    preference = character_enhancement_model_preference(
        repositories=repositories,
        save_id=save_id,
    )

    assert preference is not None
    assert preference.provider == "openrouter"
    assert preference.model_id == "openrouter/full-context"


@pytest.mark.parametrize(
    "purpose",
    [
        "response_planning",
        "response_verification",
        "director_pressure",
        "character_action_planning",
    ],
)
def test_roleplay_model_preference_defaults_agentic_structured_tasks_to_context_model(
    repositories: PersistenceRepositories,
    purpose: str,
) -> None:
    save_id = _create_save(repositories, scenario_type=FULL_ROLEPLAY_TYPE)
    repositories.set_model_preference(
        task="context_update",
        provider="openrouter",
        model_id="openrouter/context-structured",
    )

    preference = roleplay_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose=purpose,
    )

    assert preference is not None
    assert preference.provider == "openrouter"
    assert preference.model_id == "openrouter/context-structured"


def test_roleplay_image_prompt_preference_preserves_chat_fallback(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save(repositories, scenario_type=DATING_SIM_TYPE)
    repositories.set_app_setting(ROLEPLAY_SHARED_MODE_SETTING, False)
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/base-chat",
    )

    preference = roleplay_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose="image_prompt",
    )

    assert preference is not None
    assert preference.provider == "openrouter"
    assert preference.model_id == "openrouter/base-chat"


def test_narrator_fallback_preference_is_separate_from_background_text_fallback(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save(repositories, scenario_type=FULL_ROLEPLAY_TYPE)
    repositories.set_app_setting(ROLEPLAY_SHARED_MODE_SETTING, False)
    repositories.set_model_preference(
        task="chat_fallback",
        provider="openrouter",
        model_id="openrouter/background-fallback",
    )
    repositories.set_model_preference(
        task="full_roleplay_narrator_fallback",
        provider="venice",
        model_id="venice/narrator-fallback",
    )

    narrator = narrator_fallback_model_preference(
        repositories=repositories,
        save_id=save_id,
    )
    background = roleplay_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose="chat_fallback",
    )

    assert narrator is not None
    assert narrator.provider == "venice"
    assert narrator.model_id == "venice/narrator-fallback"
    assert background is not None
    assert background.provider == "openrouter"
    assert background.model_id == "openrouter/background-fallback"


def test_narrator_fallback_preference_preserves_legacy_chat_fallback(
    repositories: PersistenceRepositories,
) -> None:
    save_id = _create_save(repositories, scenario_type=FULL_ROLEPLAY_TYPE)
    repositories.set_model_preference(
        task="chat_fallback",
        provider="openrouter",
        model_id="openrouter/legacy-fallback",
    )

    preference = narrator_fallback_model_preference(
        repositories=repositories,
        save_id=save_id,
    )

    assert preference is not None
    assert preference.provider == "openrouter"
    assert preference.model_id == "openrouter/legacy-fallback"


def test_scenario_generation_model_preference_uses_section_override_then_default(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/base-chat",
    )
    repositories.set_model_preference(
        task="scenario_generation",
        provider="openrouter",
        model_id="openrouter/scenario-default",
    )
    repositories.set_model_preference(
        task=scenario_generation_section_model_task("worldbuilding"),
        provider="openrouter",
        model_id="openrouter/deep-world",
    )

    overridden = scenario_generation_model_preference(
        repositories,
        section_id="worldbuilding",
    )
    inherited = scenario_generation_model_preference(
        repositories,
        section_id="title",
    )

    assert overridden is not None
    assert overridden.model_id == "openrouter/deep-world"
    assert inherited is not None
    assert inherited.model_id == "openrouter/scenario-default"


def test_scenario_generation_model_preference_preserves_chat_fallback(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/base-chat",
    )

    preference = scenario_generation_model_preference(
        repositories,
        section_id="locations",
    )

    assert preference is not None
    assert preference.provider == "openrouter"
    assert preference.model_id == "openrouter/base-chat"


def _create_save(
    repositories: PersistenceRepositories,
    *,
    scenario_type: str,
) -> str:
    scenario = repositories.create_scenario(
        type=scenario_type,
        title=f"{scenario_type} scenario",
        premise="Premise",
        player_role="Player",
        content={"starting_scene": "Start"},
    )
    save = repositories.create_save(
        scenario_id=scenario.id,
        title=f"{scenario_type} save",
    )
    return save.id
