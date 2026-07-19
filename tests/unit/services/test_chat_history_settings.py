from __future__ import annotations

from typing import cast

from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.chat_history_settings import (
    DEFAULT_NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW,
    DEFAULT_NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW,
    DEFAULT_RECENT_NARRATOR_MESSAGE_WINDOW,
    DEFAULT_RECENT_PLAYER_MESSAGE_WINDOW,
    NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
    NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
    ChatHistoryWindowSettings,
    chat_history_window_settings,
    narrator_planner_chat_history_window_settings,
    sanitize_recent_message_window,
)


class FakeRepositories:
    def __init__(self, settings: dict[str, object]) -> None:
        self.settings = settings

    def get_app_setting(self, key: str) -> object:
        return self.settings.get(key)

    def get_effective_setting(
        self,
        key: str,
        *,
        save_id: str | None = None,
        user_id: str | None = None,
        scenario_id: str | None = None,
    ) -> object:
        return self.settings.get(key)


def test_sanitize_recent_message_window_clamps_integer_values() -> None:
    assert sanitize_recent_message_window(-1, default=6) == 0
    assert sanitize_recent_message_window(12, default=6) == 12
    assert sanitize_recent_message_window(99, default=6) == 24


def test_sanitize_recent_message_window_rejects_bool_and_non_int_values() -> None:
    assert sanitize_recent_message_window(True, default=6) == 6
    assert sanitize_recent_message_window("12", default=6) == 6


def test_chat_history_window_settings_reads_and_sanitizes_repository_values() -> None:
    settings = chat_history_window_settings(
        cast(
            PersistenceRepositories,
            FakeRepositories(
                {
                    "recent_player_message_window": 30,
                    "recent_narrator_message_window": "bad",
                }
            ),
        )
    )

    assert settings == ChatHistoryWindowSettings(
        player_messages=24,
        narrator_messages=DEFAULT_RECENT_NARRATOR_MESSAGE_WINDOW,
    )
    assert ChatHistoryWindowSettings.defaults() == ChatHistoryWindowSettings(
        player_messages=DEFAULT_RECENT_PLAYER_MESSAGE_WINDOW,
        narrator_messages=DEFAULT_RECENT_NARRATOR_MESSAGE_WINDOW,
    )


def test_narrator_planner_chat_history_windows_use_planner_defaults() -> None:
    settings = narrator_planner_chat_history_window_settings(
        cast(
            PersistenceRepositories,
            FakeRepositories(
                {
                    "recent_player_message_window": 4,
                    "recent_narrator_message_window": 3,
                }
            ),
        )
    )

    assert settings == ChatHistoryWindowSettings(
        player_messages=DEFAULT_NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW,
        narrator_messages=DEFAULT_NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW,
    )


def test_narrator_planner_chat_history_windows_use_explicit_overrides() -> None:
    settings = narrator_planner_chat_history_window_settings(
        cast(
            PersistenceRepositories,
            FakeRepositories(
                {
                    "recent_player_message_window": 4,
                    "recent_narrator_message_window": 3,
                    NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW_SETTING: 9,
                    NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW_SETTING: "bad",
                }
            ),
        )
    )

    assert settings == ChatHistoryWindowSettings(
        player_messages=9,
        narrator_messages=DEFAULT_NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW,
    )
