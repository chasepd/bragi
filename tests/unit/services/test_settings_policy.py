from bragi.retry_policy import PROVIDER_CALL_DEADLINE_SETTING, RETRY_COUNT_SETTING
from bragi.services.character_text_service import (
    CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING,
    CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING,
)
from bragi.services.content_rating import (
    CONTENT_FILTER_RATING_SETTING,
    FADE_TO_BLACK_ENABLED_SETTING,
)
from bragi.services.pending_jobs_settings import PENDING_JOBS_DISPLAY_MODE_SETTING
from bragi.services.phrase_denylist import (
    GENERATED_PHRASE_DENYLIST_SETTING,
    SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
)
from bragi.services.settings_policy import (
    role_can_write_scoped_setting,
    scoped_setting_policy,
)
from bragi.services.user_narration_guidance import USER_NARRATION_GUIDANCE_SETTING


def test_child_write_permission_is_per_user_scoped_key() -> None:
    pending_jobs_policy = scoped_setting_policy(PENDING_JOBS_DISPLAY_MODE_SETTING)
    guidance_policy = scoped_setting_policy(USER_NARRATION_GUIDANCE_SETTING)

    assert pending_jobs_policy.scope == "user"
    assert pending_jobs_policy.child_allowed is True
    assert guidance_policy.scope == "user"
    assert guidance_policy.child_allowed is False
    assert role_can_write_scoped_setting(
        "child",
        PENDING_JOBS_DISPLAY_MODE_SETTING,
    )
    assert not role_can_write_scoped_setting(
        "child",
        USER_NARRATION_GUIDANCE_SETTING,
    )


def test_content_safety_preferences_are_user_scoped_with_role_specific_writes() -> None:
    rating_policy = scoped_setting_policy(CONTENT_FILTER_RATING_SETTING)
    fade_policy = scoped_setting_policy(FADE_TO_BLACK_ENABLED_SETTING)

    assert rating_policy.scope == "user"
    assert rating_policy.child_allowed is True
    assert fade_policy.scope == "user"
    assert fade_policy.child_allowed is False
    assert role_can_write_scoped_setting("child", CONTENT_FILTER_RATING_SETTING)
    assert not role_can_write_scoped_setting("child", FADE_TO_BLACK_ENABLED_SETTING)
    assert role_can_write_scoped_setting("user", CONTENT_FILTER_RATING_SETTING)
    assert role_can_write_scoped_setting("user", FADE_TO_BLACK_ENABLED_SETTING)


def test_proactive_character_text_random_controls_are_save_scoped() -> None:
    chance_policy = scoped_setting_policy(
        CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING,
    )
    cooldown_policy = scoped_setting_policy(
        CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING,
    )

    assert chance_policy.scope == "save"
    assert cooldown_policy.scope == "save"
    assert role_can_write_scoped_setting(
        "user",
        CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING,
    )
    assert role_can_write_scoped_setting(
        "user",
        CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING,
    )


def test_phrase_denylist_settings_have_global_and_save_scopes() -> None:
    global_policy = scoped_setting_policy(GENERATED_PHRASE_DENYLIST_SETTING)
    save_policy = scoped_setting_policy(SAVE_GENERATED_PHRASE_DENYLIST_SETTING)

    assert global_policy.scope == "global"
    assert global_policy.admin_only is True
    assert save_policy.scope == "save"
    assert role_can_write_scoped_setting("admin", GENERATED_PHRASE_DENYLIST_SETTING)
    assert not role_can_write_scoped_setting(
        "user",
        GENERATED_PHRASE_DENYLIST_SETTING,
    )
    assert role_can_write_scoped_setting("user", SAVE_GENERATED_PHRASE_DENYLIST_SETTING)


def test_retry_count_is_global_and_admin_only() -> None:
    policy = scoped_setting_policy(RETRY_COUNT_SETTING)

    assert policy.scope == "global"
    assert policy.admin_only is True
    assert role_can_write_scoped_setting("admin", RETRY_COUNT_SETTING)
    assert not role_can_write_scoped_setting("user", RETRY_COUNT_SETTING)
    assert not role_can_write_scoped_setting("child", RETRY_COUNT_SETTING)


def test_provider_call_deadline_is_global_and_admin_only() -> None:
    policy = scoped_setting_policy(PROVIDER_CALL_DEADLINE_SETTING)

    assert policy.scope == "global"
    assert policy.admin_only is True
    assert role_can_write_scoped_setting("admin", PROVIDER_CALL_DEADLINE_SETTING)
    assert not role_can_write_scoped_setting("user", PROVIDER_CALL_DEADLINE_SETTING)
    assert not role_can_write_scoped_setting("child", PROVIDER_CALL_DEADLINE_SETTING)
