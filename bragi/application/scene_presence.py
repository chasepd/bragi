"""Scene-presence view models for historical message scenes."""

from __future__ import annotations

from dataclasses import dataclass

from bragi.persistence.models import CharacterRecord, MessageRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.character_registry_service import (
    CharacterRegistryReferenceImageRow,
    CharacterRegistryService,
)


@dataclass(frozen=True)
class ScenePresenceCharacter:
    character_id: str
    name: str
    present: bool
    has_reference_image: bool
    reference_image: CharacterRegistryReferenceImageRow | None
    is_player_character: bool
    status: str


@dataclass(frozen=True)
class ScenePresenceModel:
    save_id: str
    message_id: str
    latest_message: bool
    characters: tuple[ScenePresenceCharacter, ...]


def build_scene_presence_model(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    message_id: str,
) -> ScenePresenceModel:
    messages = repositories.list_messages(save_id)
    message = _active_message(messages, message_id)
    if message is None:
        raise ValueError(f"Unknown active message id: {message_id}")
    latest_message = message.id == messages[-1].id if messages else False
    present_ids = set(
        present_character_ids_for_message(
            repositories,
            save_id=save_id,
            message_id=message_id,
            messages=messages,
        )
    )
    registry = CharacterRegistryService(
        repositories,
        active_save_id=save_id,
    ).build_model(active_save_id=save_id)
    reference_images = {
        row.character_id: row.reference_image
        for row in registry.characters
        if row.reference_image is not None
    }
    characters = tuple(
        _presence_character(
            character,
            present=character.id in present_ids,
            reference_image=reference_images.get(character.id),
        )
        for character in repositories.list_characters(save_id)
    )
    return ScenePresenceModel(
        save_id=save_id,
        message_id=message_id,
        latest_message=latest_message,
        characters=characters,
    )


def present_character_ids_for_message(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    message_id: str,
    messages: list[MessageRecord] | None = None,
) -> tuple[str, ...]:
    active_messages = (
        messages if messages is not None else repositories.list_messages(save_id)
    )
    message = _active_message(active_messages, message_id)
    if message is None:
        raise ValueError(f"Unknown active message id: {message_id}")
    persisted = repositories.list_message_scene_presence(
        save_id,
        message_id=message_id,
    )
    if persisted:
        return tuple(record.character_id for record in persisted)
    latest_message_id = active_messages[-1].id if active_messages else None
    if message_id != latest_message_id:
        return ()
    snapshot = repositories.get_scene_snapshot(save_id)
    return tuple(snapshot.present_character_ids if snapshot else ())


def character_image_eligible_message_ids(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    messages: list[MessageRecord] | None = None,
) -> frozenset[str]:
    active_messages = (
        messages if messages is not None else repositories.list_messages(save_id)
    )
    if not active_messages:
        return frozenset()
    registry = CharacterRegistryService(
        repositories,
        active_save_id=save_id,
    ).build_model(active_save_id=save_id)
    reference_character_ids = frozenset(
        row.character_id
        for row in registry.characters
        if row.reference_image is not None
    )
    if not reference_character_ids:
        return frozenset()
    eligible: set[str] = set()
    for message in active_messages:
        present_ids = set(
            present_character_ids_for_message(
                repositories,
                save_id=save_id,
                message_id=message.id,
                messages=active_messages,
            )
        )
        if present_ids & reference_character_ids:
            eligible.add(message.id)
    return frozenset(eligible)


def _presence_character(
    character: CharacterRecord,
    *,
    present: bool,
    reference_image: CharacterRegistryReferenceImageRow | None,
) -> ScenePresenceCharacter:
    return ScenePresenceCharacter(
        character_id=character.id,
        name=character.name,
        present=present,
        has_reference_image=reference_image is not None,
        reference_image=reference_image,
        is_player_character=character.is_player_character,
        status=character.status,
    )


def _active_message(
    messages: list[MessageRecord],
    message_id: str,
) -> MessageRecord | None:
    for message in messages:
        if message.id == message_id:
            return message
    return None
