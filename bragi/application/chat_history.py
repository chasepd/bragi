"""Import-safe chat history view model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from bragi.application.chronicle import (
    ChronicleMarkdownBlock,
    build_chronicle_model,
)
from bragi.persistence.models import MessageRecord
from bragi.persistence.repositories import PersistenceRepositories


class ChatHistoryFilter(StrEnum):
    ALL = "all"
    PLAYER = "player"
    NARRATOR_CHARACTER = "narrator_character"
    WITH_IMAGES = "with_images"


@dataclass(frozen=True)
class ChatHistoryFilterOption:
    filter_id: str
    label: str
    active: bool


@dataclass(frozen=True)
class ChatHistoryMessageModel:
    message_id: str
    role: str
    role_label: str
    speaker_name: str | None
    body: str
    markdown_blocks: tuple[ChronicleMarkdownBlock, ...]
    style_class: str
    provider: str | None
    model: str | None
    token_estimate: int | None
    created_at: str | None
    image_count: int = 0

    @property
    def has_images(self) -> bool:
        return self.image_count > 0

    @property
    def provider_model_label(self) -> str:
        if self.provider and self.model:
            return f"{self.provider} / {self.model}"
        return self.provider or self.model or ""


@dataclass(frozen=True)
class ChatHistoryModel:
    active_save_id: str | None
    active_save_title: str | None
    selected_filter: str
    filter_options: tuple[ChatHistoryFilterOption, ...]
    messages: tuple[ChatHistoryMessageModel, ...]
    total_message_count: int
    matching_message_count: int
    has_more_before: bool
    oldest_message_id: str | None
    empty_title: str
    empty_detail: str


_FILTER_LABELS: tuple[tuple[ChatHistoryFilter, str], ...] = (
    (ChatHistoryFilter.ALL, "All"),
    (ChatHistoryFilter.PLAYER, "Player"),
    (ChatHistoryFilter.NARRATOR_CHARACTER, "Narrator/character"),
    (ChatHistoryFilter.WITH_IMAGES, "With images"),
)


def build_chat_history_model(
    *,
    repositories: PersistenceRepositories,
    save_id: str | None,
    save_title: str | None = None,
    selected_filter: str | ChatHistoryFilter | None = ChatHistoryFilter.ALL,
    player_speaker_name: str | None = None,
    before_message_id: str | None = None,
    limit: int = 80,
) -> ChatHistoryModel:
    active_filter = normalize_chat_history_filter(selected_filter)
    if save_id is None:
        return ChatHistoryModel(
            active_save_id=None,
            active_save_title=None,
            selected_filter=active_filter.value,
            filter_options=_filter_options(active_filter),
            messages=(),
            total_message_count=0,
            matching_message_count=0,
            has_more_before=False,
            oldest_message_id=None,
            empty_title="No save loaded",
            empty_detail="Start or load a save to inspect its chronicle history.",
        )

    total_message_count = repositories.count_chat_history_messages(save_id)
    matching_message_count = (
        total_message_count
        if active_filter is ChatHistoryFilter.ALL
        else repositories.count_chat_history_messages(
            save_id,
            selected_filter=active_filter.value,
        )
    )
    page = repositories.list_chat_history_message_page(
        save_id,
        selected_filter=active_filter.value,
        before_message_id=before_message_id,
        limit=limit,
    )
    messages = tuple(page.messages)
    image_counts = repositories.image_counts_for_messages(
        save_id=save_id,
        message_ids=(message.id for message in messages),
    )
    history_messages = _history_messages(
        messages,
        image_counts=image_counts,
        player_speaker_name=player_speaker_name,
    )
    return ChatHistoryModel(
        active_save_id=save_id,
        active_save_title=save_title,
        selected_filter=active_filter.value,
        filter_options=_filter_options(active_filter),
        messages=history_messages,
        total_message_count=total_message_count,
        matching_message_count=matching_message_count,
        has_more_before=page.has_more_before,
        oldest_message_id=history_messages[0].message_id if history_messages else None,
        empty_title=_empty_title(
            total_count=total_message_count,
            visible_count=matching_message_count,
        ),
        empty_detail=_empty_detail(
            total_count=total_message_count,
            visible_count=matching_message_count,
        ),
    )


def normalize_chat_history_filter(
    value: str | ChatHistoryFilter | None,
) -> ChatHistoryFilter:
    if isinstance(value, ChatHistoryFilter):
        return value
    try:
        return ChatHistoryFilter(str(value))
    except ValueError:
        return ChatHistoryFilter.ALL


def _filter_options(
    selected_filter: ChatHistoryFilter,
) -> tuple[ChatHistoryFilterOption, ...]:
    return tuple(
        ChatHistoryFilterOption(
            filter_id=filter_id.value,
            label=label,
            active=filter_id == selected_filter,
        )
        for filter_id, label in _FILTER_LABELS
    )


def _history_messages(
    messages: Sequence[MessageRecord],
    *,
    image_counts: Mapping[str, int],
    player_speaker_name: str | None,
) -> tuple[ChatHistoryMessageModel, ...]:
    chronicle = build_chronicle_model(
        list(messages),
        player_speaker_name=player_speaker_name,
    )
    return tuple(
        ChatHistoryMessageModel(
            message_id=message.id,
            role=message.role,
            role_label=chronicle_message.role_label,
            speaker_name=chronicle_message.speaker_name,
            body=message.body,
            markdown_blocks=chronicle_message.markdown_blocks,
            style_class=chronicle_message.style_class,
            provider=message.provider,
            model=message.model,
            token_estimate=message.token_estimate,
            created_at=message.created_at,
            image_count=image_counts.get(message.id, 0),
        )
        for message, chronicle_message in zip(
            messages,
            chronicle.messages,
            strict=True,
        )
    )

def _empty_title(
    *,
    total_count: int,
    visible_count: int,
) -> str:
    if visible_count > 0:
        return ""
    if total_count > 0:
        return "No matching messages"
    return "No messages yet"


def _empty_detail(
    *,
    total_count: int,
    visible_count: int,
) -> str:
    if visible_count > 0:
        return ""
    if total_count > 0:
        return "Try another history filter."
    return "Messages will appear here once this save has chronicle entries."
