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

_MISSING = object()


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_chat_history_model_filters_visible_messages_and_image_sources(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    chat_history = _import_chat_history_without_gtk(monkeypatch)
    save_id, message_ids = _save_with_history_messages(repositories)
    repositories.create_media_asset(
        save_id=save_id,
        source_message_id=message_ids["narrator"],
        type="image",
        path="media/save-1/narrator.png",
        thumbnail_path=None,
        prompt="Ash claws the signal glass.",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )

    all_model = chat_history.build_chat_history_model(
        repositories=repositories,
        save_id=save_id,
        save_title="Night Watch",
        selected_filter="all",
        player_speaker_name="Mara Voss",
    )
    player_model = chat_history.build_chat_history_model(
        repositories=repositories,
        save_id=save_id,
        selected_filter="player",
    )
    narrator_character_model = chat_history.build_chat_history_model(
        repositories=repositories,
        save_id=save_id,
        selected_filter="narrator_character",
    )
    image_model = chat_history.build_chat_history_model(
        repositories=repositories,
        save_id=save_id,
        selected_filter="with_images",
    )

    assert _value(all_model, "active_save_id") == save_id
    assert _value(all_model, "active_save_title") == "Night Watch"
    assert _value(all_model, "total_message_count") == 4
    assert [_value(message, "body") for message in _messages(all_model)] == [
        "I climb toward the beacon lens.",
        "Ash claws the signal glass.",
        "Keep the autosave quiet.",
        "Captain Ilyra points toward the lower stair.",
    ]
    assert [_value(message, "role_label") for message in _messages(all_model)] == [
        "Mara Voss",
        "Narrator",
        "System",
        "Message",
    ]
    assert [_value(message, "image_count") for message in _messages(all_model)] == [
        0,
        1,
        0,
        0,
    ]
    assert all(_value(message, "created_at") for message in _messages(all_model))
    assert [_value(message, "body") for message in _messages(player_model)] == [
        "I climb toward the beacon lens.",
    ]
    assert [
        _value(message, "body") for message in _messages(narrator_character_model)
    ] == [
        "Ash claws the signal glass.",
        "Captain Ilyra points toward the lower stair.",
    ]
    assert [_value(message, "body") for message in _messages(image_model)] == [
        "Ash claws the signal glass.",
    ]
    assert _value(image_model, "matching_message_count") == 1
    assert _value(image_model, "has_more_before") is False
    assert _value(image_model, "oldest_message_id") == message_ids["narrator"]


def test_chat_history_model_paginates_filtered_history(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    chat_history = _import_chat_history_without_gtk(monkeypatch)
    save_id, message_ids = _save_with_history_messages(repositories)

    latest = chat_history.build_chat_history_model(
        repositories=repositories,
        save_id=save_id,
        selected_filter="narrator_character",
        limit=1,
    )
    previous = chat_history.build_chat_history_model(
        repositories=repositories,
        save_id=save_id,
        selected_filter="narrator_character",
        before_message_id=message_ids["character"],
        limit=1,
    )

    assert _value(latest, "total_message_count") == 4
    assert _value(latest, "matching_message_count") == 2
    assert [_value(message, "message_id") for message in _messages(latest)] == [
        message_ids["character"],
    ]
    assert _value(latest, "has_more_before") is True
    assert _value(latest, "oldest_message_id") == message_ids["character"]
    assert [_value(message, "message_id") for message in _messages(previous)] == [
        message_ids["narrator"],
    ]
    assert _value(previous, "has_more_before") is False
    assert _value(previous, "oldest_message_id") == message_ids["narrator"]


def test_chat_history_model_ignores_other_save_and_unknown_filter(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    chat_history = _import_chat_history_without_gtk(monkeypatch)
    save_id, message_ids = _save_with_history_messages(repositories)
    other_save_id, other_message_ids = _save_with_history_messages(
        repositories,
        title="Other Watch",
    )
    repositories.create_media_asset(
        save_id=other_save_id,
        source_message_id=other_message_ids["narrator"],
        type="image",
        path="media/other-save/narrator.png",
        thumbnail_path=None,
        prompt="This image belongs elsewhere.",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    repositories.archive_message(message_ids["system"])

    model = chat_history.build_chat_history_model(
        repositories=repositories,
        save_id=save_id,
        selected_filter="not-a-filter",
    )
    image_model = chat_history.build_chat_history_model(
        repositories=repositories,
        save_id=save_id,
        selected_filter="with_images",
    )

    assert _value(model, "selected_filter") == "all"
    assert [_value(message, "message_id") for message in _messages(model)] == [
        message_ids["player"],
        message_ids["narrator"],
        message_ids["character"],
    ]
    assert _messages(image_model) == ()
    assert _value(image_model, "empty_title") == "No matching messages"


def test_chat_history_model_handles_no_save_and_empty_history(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    chat_history = _import_chat_history_without_gtk(monkeypatch)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Empty Watch")

    no_save_model = chat_history.build_chat_history_model(
        repositories=repositories,
        save_id=None,
    )
    empty_model = chat_history.build_chat_history_model(
        repositories=repositories,
        save_id=save.id,
        save_title=save.title,
    )

    assert _value(no_save_model, "active_save_id") is None
    assert _messages(no_save_model) == ()
    assert _value(no_save_model, "empty_title") == "No save loaded"
    assert _value(empty_model, "active_save_id") == save.id
    assert _messages(empty_model) == ()
    assert _value(empty_model, "empty_title") == "No messages yet"


def _import_chat_history_without_gtk(monkeypatch: MonkeyPatch) -> Any:
    original_import = builtins.__import__

    def import_without_gtk(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "gi" or name.startswith("gi."):
            raise AssertionError(
                "bragi.application.chat_history must not import GTK/PyGObject"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_gtk)
    sys.modules.pop("bragi.application.chat_history", None)
    return importlib.import_module("bragi.application.chat_history")


def _save_with_history_messages(
    repositories: PersistenceRepositories,
    *,
    title: str = "Night Watch",
) -> tuple[str, dict[str, str]]:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"player_character_name": "Mara Voss"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title=title)
    player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Player",
        body="I climb toward the beacon lens.",
    )
    narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Ash claws the signal glass.",
        provider="fake",
        model="fake-chat",
        token_estimate=12,
    )
    system = repositories.append_message(
        save_id=save.id,
        role="system",
        speaker_name="System",
        body="Keep the autosave quiet.",
    )
    character = repositories.append_message(
        save_id=save.id,
        role="character",
        speaker_name="Captain Ilyra",
        body="Captain Ilyra points toward the lower stair.",
        provider="fake",
        model="fake-chat",
    )
    return save.id, {
        "player": player.id,
        "narrator": narrator.id,
        "system": system.id,
        "character": character.id,
    }


def _messages(model: object) -> tuple[object, ...]:
    return tuple(_value(model, "messages", "items"))


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
