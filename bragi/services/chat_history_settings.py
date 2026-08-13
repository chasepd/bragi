"""Configurable recent chat history windows."""

from __future__ import annotations

from dataclasses import dataclass

from bragi.persistence.repositories import PersistenceRepositories
from bragi.retry_policy import RetryExecutionClass, current_retry_execution_class
from bragi.services.turn_responsiveness import RESPONSIVE_PLANNER_MESSAGE_WINDOW

RECENT_PLAYER_MESSAGE_WINDOW_SETTING = "recent_player_message_window"
RECENT_NARRATOR_MESSAGE_WINDOW_SETTING = "recent_narrator_message_window"
NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW_SETTING = (
    "narrator_planner_recent_player_message_window"
)
NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW_SETTING = (
    "narrator_planner_recent_narrator_message_window"
)
DEFAULT_RECENT_PLAYER_MESSAGE_WINDOW = 5
DEFAULT_RECENT_NARRATOR_MESSAGE_WINDOW = 5
DEFAULT_NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW = 24
DEFAULT_NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW = 24
MIN_RECENT_MESSAGE_WINDOW = 0
MAX_RECENT_MESSAGE_WINDOW = 24


@dataclass(frozen=True)
class ChatHistoryWindowSettings:
    player_messages: int = DEFAULT_RECENT_PLAYER_MESSAGE_WINDOW
    narrator_messages: int = DEFAULT_RECENT_NARRATOR_MESSAGE_WINDOW

    @classmethod
    def defaults(cls) -> ChatHistoryWindowSettings:
        return cls()


def chat_history_window_settings(
    repositories: PersistenceRepositories,
    *,
    save_id: str | None = None,
) -> ChatHistoryWindowSettings:
    return ChatHistoryWindowSettings(
        player_messages=sanitize_recent_message_window(
            repositories.get_effective_setting(
                RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
                save_id=save_id,
            ),
            default=DEFAULT_RECENT_PLAYER_MESSAGE_WINDOW,
        ),
        narrator_messages=sanitize_recent_message_window(
            repositories.get_effective_setting(
                RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
                save_id=save_id,
            ),
            default=DEFAULT_RECENT_NARRATOR_MESSAGE_WINDOW,
        ),
    )


def narrator_planner_chat_history_window_settings(
    repositories: PersistenceRepositories,
    *,
    save_id: str | None = None,
) -> ChatHistoryWindowSettings:
    if (
        current_retry_execution_class()
        is RetryExecutionClass.RESPONSIVE_FOREGROUND
    ):
        return ChatHistoryWindowSettings(
            player_messages=RESPONSIVE_PLANNER_MESSAGE_WINDOW,
            narrator_messages=RESPONSIVE_PLANNER_MESSAGE_WINDOW,
        )
    return ChatHistoryWindowSettings(
        player_messages=sanitize_recent_message_window(
            repositories.get_effective_setting(
                NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
                save_id=save_id,
            ),
            default=DEFAULT_NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW,
        ),
        narrator_messages=sanitize_recent_message_window(
            repositories.get_effective_setting(
                NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
                save_id=save_id,
            ),
            default=DEFAULT_NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW,
        ),
    )


def sanitize_recent_message_window(value: object, *, default: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        return default
    return min(max(value, MIN_RECENT_MESSAGE_WINDOW), MAX_RECENT_MESSAGE_WINDOW)
