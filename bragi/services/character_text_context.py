"""Shared character text context filtering."""

from __future__ import annotations

from collections.abc import Iterable

from bragi.persistence.models import (
    CharacterTextMessageRecord,
    CharacterTextThreadRecord,
)
from bragi.persistence.repositories import PersistenceRepositories


def canonical_character_text_context_messages(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    thread_id: str | None = None,
    include_message_ids: Iterable[str] = (),
) -> list[CharacterTextMessageRecord]:
    """Return messages that are canonical for model/world context.

    Failed and pending deliveries remain visible in the phone UI, but they are
    not delivered conversation history. A caller may include a specific in-flight
    message when generating the reply for that message.
    """
    included = frozenset(include_message_ids)
    return [
        message
        for message in repositories.list_character_text_messages(
            save_id=save_id,
            thread_id=thread_id,
        )
        if message.delivery_status == "sent" or message.id in included
    ]


def uploaded_photo_descriptions_by_message_id(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    messages: Iterable[CharacterTextMessageRecord],
) -> dict[str, tuple[str, ...]]:
    message_ids = tuple(message.id for message in messages)
    if not message_ids:
        return {}
    result: dict[str, list[str]] = {}
    for attachment in repositories.list_character_text_message_attachments(
        save_id=save_id,
        text_message_ids=message_ids,
    ):
        if attachment.kind != "uploaded_photo" or attachment.status != "succeeded":
            continue
        description = attachment.prompt.strip()
        if not description:
            continue
        result.setdefault(attachment.text_message_id, []).append(description)
    return {
        message_id: tuple(descriptions)
        for message_id, descriptions in result.items()
    }


def player_character_ids(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
) -> frozenset[str]:
    return frozenset(
        character.id
        for character in repositories.list_characters(save_id)
        if character.is_player_character
    )


def character_text_thread_participant_character_ids(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    thread: CharacterTextThreadRecord,
) -> frozenset[str]:
    character_ids: set[str] = set()
    if thread.character_id:
        character_ids.add(thread.character_id)
    if thread.kind == "group":
        character_ids.update(
            participant.character_id
            for participant in repositories.list_character_text_thread_participants(
                save_id=save_id,
                thread_id=thread.id,
            )
        )
    return frozenset(character_ids)


def character_text_audience_character_ids(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    text_messages: Iterable[CharacterTextMessageRecord] = (),
    thread: CharacterTextThreadRecord | None = None,
    include_player: bool = True,
) -> frozenset[str]:
    character_ids: set[str] = set()
    if include_player:
        character_ids.update(
            player_character_ids(
                repositories=repositories,
                save_id=save_id,
            )
        )
    messages = tuple(text_messages)
    thread_ids = {message.thread_id for message in messages}
    for message in messages:
        if message.character_id:
            character_ids.add(message.character_id)
        if message.sender_character_id:
            character_ids.add(message.sender_character_id)
    if thread is not None:
        thread_ids.add(thread.id)
        character_ids.update(
            character_text_thread_participant_character_ids(
                repositories=repositories,
                save_id=save_id,
                thread=thread,
            )
        )
    for thread_id in thread_ids:
        current = (
            thread
            if thread is not None and thread.id == thread_id
            else repositories.get_character_text_thread(
                save_id=save_id,
                thread_id=thread_id,
            )
        )
        if current is None:
            continue
        character_ids.update(
            character_text_thread_participant_character_ids(
                repositories=repositories,
                save_id=save_id,
                thread=current,
            )
        )
    return frozenset(character_id for character_id in character_ids if character_id)
