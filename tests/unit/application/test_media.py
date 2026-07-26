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


class VideoProviderDouble:
    async def generate_video(self, _request: object) -> object:
        raise AssertionError("media model tests must not generate video")


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_media_model_exposes_latest_scene_image_for_save(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    media = _import_media_without_gtk(monkeypatch)
    save_id = _save_with_media_history(repositories)

    model = media.build_media_model(repositories=repositories, save_id=save_id)

    latest = _value(model, "latest_scene_image", "latest_image")
    assert _value(latest, "path") == "media/save-1/scene-3.png"
    assert _value(latest, "source_message_id") == "narrator-3"
    assert _value(latest, "created_at") is not None
    assert _value(latest, "source_message") == "The tower lens catches fire."
    assert _value(latest, "prompt") == "The tower lens catches fire."
    assert _value(latest, "provider") == "fake"
    assert _value(latest, "model") == "fake-image"


def test_media_model_marks_animation_eligible_images(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    media = _import_media_without_gtk(monkeypatch)
    save_id = _save_with_media_history(repositories)
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-image-video",
        display_name="Fake Image Video",
        capabilities=["image_plus_text_to_video"],
    )
    repositories.set_model_preference(
        task="image_animation",
        provider="fake",
        model_id="fake-image-video",
    )

    model = media.build_media_model(
        repositories=repositories,
        save_id=save_id,
        providers={"fake": VideoProviderDouble()},
    )

    assert _value(model, "image_animation_available") is True
    latest = _value(model, "latest_scene_image", "latest_image")
    assert _value(latest, "can_animate") is True
    assert _value(latest, "source_media_asset_id") is None


def test_media_model_disables_animation_when_provider_is_unavailable(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    media = _import_media_without_gtk(monkeypatch)
    save_id = _save_with_media_history(repositories)
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-image-video",
        display_name="Fake Image Video",
        capabilities=["image_to_video"],
    )
    repositories.set_model_preference(
        task="image_animation",
        provider="fake",
        model_id="fake-image-video",
    )

    model = media.build_media_model(
        repositories=repositories,
        save_id=save_id,
        providers={},
    )

    assert _value(model, "image_animation_available") is False
    latest = _value(model, "latest_scene_image", "latest_image")
    assert _value(latest, "can_animate") is False


def test_media_model_orders_image_history_newest_first_for_save(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    media = _import_media_without_gtk(monkeypatch)
    save_id = _save_with_media_history(repositories)

    model = media.build_media_model(repositories=repositories, save_id=save_id)

    history = list(_value(model, "image_history", "images"))
    assert [_value(item, "path") for item in history] == [
        "media/save-1/scene-3.png",
        "media/save-1/scene-2.png",
        "media/save-1/scene-1.png",
    ]
    assert [_value(item, "source_message_id") for item in history] == [
        "narrator-3",
        "narrator-2",
        "narrator-1",
    ]


def test_media_model_omits_assets_above_the_viewer_content_rating(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    media = _import_media_without_gtk(monkeypatch)
    save_id = _save_with_media_history(repositories)
    explicit_message = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="He chopped off the prisoner's head and limbs.",
    )
    explicit_asset = repositories.create_media_asset(
        save_id=save_id,
        source_message_id=explicit_message.id,
        type="image",
        path="media/save-1/explicit.png",
        thumbnail_path=None,
        prompt="He chopped off the prisoner's head and limbs.",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={"content_rating": "r"},
    )

    unfiltered = media.build_media_model(
        repositories=repositories,
        save_id=save_id,
    )
    filtered = media.build_media_model(
        repositories=repositories,
        save_id=save_id,
        allowed_rating="pg",
    )

    assert explicit_asset.id in {
        _value(item, "id") for item in _value(unfiltered, "media_history")
    }
    assert explicit_asset.id not in {
        _value(item, "id") for item in _value(filtered, "media_history")
    }


def test_media_model_omits_asset_when_its_source_message_exceeds_viewer_rating(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    media = _import_media_without_gtk(monkeypatch)
    save_id = _save_with_media_history(repositories)
    restricted_message = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Restricted source narration.",
        content_rating="r",
    )
    asset = repositories.create_media_asset(
        save_id=save_id,
        source_message_id=restricted_message.id,
        type="image",
        path="media/save-1/mild-derived-image.png",
        thumbnail_path=None,
        prompt="A quiet corridor.",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={"content_rating": "g"},
    )

    filtered = media.build_media_model(
        repositories=repositories,
        save_id=save_id,
        allowed_rating="pg",
    )

    assert asset.id not in {
        _value(item, "id") for item in _value(filtered, "media_history")
    }


def test_media_model_omits_character_text_attachment_media(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    media = _import_media_without_gtk(monkeypatch)
    save_id = _save_with_media_history(repositories)
    repositories.create_media_asset(
        save_id=save_id,
        source_message_id=None,
        type="image",
        path="media/save-1/text-attachment.png",
        thumbnail_path=None,
        prompt="creased arcade ticket stub on a dusty cabinet",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={
            "kind": "character_text_object_context_image",
            "text_message_id": "text-message-1",
        },
    )

    model = media.build_media_model(repositories=repositories, save_id=save_id)

    latest = _value(model, "latest_scene_image", "latest_image")
    assert _value(latest, "path") == "media/save-1/scene-3.png"
    assert [
        _value(item, "path")
        for item in _value(model, "image_history", "images")
    ] == [
        "media/save-1/scene-3.png",
        "media/save-1/scene-2.png",
        "media/save-1/scene-1.png",
    ]
    assert [
        _value(item, "path") for item in _value(model, "media_history")
    ] == [
        "media/save-1/scene-3.png",
        "media/save-1/scene-2.png",
        "media/save-1/scene-1.png",
    ]


def test_media_model_exposes_latest_scene_media_for_generated_video(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    media = _import_media_without_gtk(monkeypatch)
    save_id = _save_with_media_history(repositories)
    source_image = repositories.list_media_assets(save_id)[-1]
    video = repositories.create_media_asset(
        save_id=save_id,
        source_message_id="narrator-3",
        source_media_asset_id=source_image.id,
        type="video",
        path="media/save-1/scene-3-motion.mp4",
        thumbnail_path=None,
        prompt="The tower lens flares while ash wheels around it.",
        provider="fake-video",
        model="fake-video-model",
        status="succeeded",
        mime_type="video/mp4",
        metadata={"duration_seconds": 4, "source_flow": "image_animation"},
    )

    model = media.build_media_model(repositories=repositories, save_id=save_id)

    latest_media = _value(model, "latest_scene_media")
    latest_image = _value(model, "latest_scene_image", "latest_image")
    assert _value(latest_media, "id") == video.id
    assert _value(latest_media, "type") == "video"
    assert _value(latest_media, "mime_type") == "video/mp4"
    assert _value(latest_media, "metadata") == {
        "duration_seconds": 4,
        "source_flow": "image_animation",
    }
    assert _value(latest_image, "id") == source_image.id
    assert [_value(item, "path") for item in _value(model, "media_history")] == [
        "media/save-1/scene-3-motion.mp4",
        "media/save-1/scene-3.png",
        "media/save-1/scene-2.png",
        "media/save-1/scene-1.png",
    ]
    source_media = _value(latest_media, "source_media")
    assert _value(source_media, "id") == source_image.id
    assert _value(source_media, "type") == "image"
    assert _value(source_media, "prompt_preview") == "The tower lens catches fire."


def test_media_model_does_not_offer_retired_reference_promotion_action(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    media = _import_media_without_gtk(monkeypatch)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Sigil Keeper",
        premise="A private audience with the sigil keeper.",
        player_role="A careful petitioner",
        content={"character_name": "Sigil Keeper"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Audience")
    message = repositories.append_message(
        save_id=save.id,
        message_id="narrator-reference",
        role="narrator",
        speaker_name="Narrator",
        body="The sigil keeper lifts a prism.",
        provider="fake",
        model="fake-chat",
    )
    character = repositories.add_character(
        save_id=save.id,
        name="Sigil Keeper",
        role="Keeper of the glass sigils.",
    )
    reference = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=message.id,
        type="image",
        path="media/save-reference/current.png",
        thumbnail_path=None,
        prompt="current reference",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={
            "kind": "character_reference",
            "character_id": character.id,
            "character_name": "Stale Keeper",
        },
    )
    candidate = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=message.id,
        type="image",
        path="media/save-reference/candidate.png",
        thumbnail_path=None,
        prompt="generated character image",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={
            "kind": "character_image",
            "character_id": character.id,
            "character_name": "Stale Keeper",
        },
    )
    scene_candidate = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=message.id,
        type="image",
        path="media/save-reference/scene.png",
        thumbnail_path=None,
        prompt="generated scene image",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={"kind": "scene_image"},
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=character.id,
        target_type="media_asset",
        target_id=reference.id,
        relation="reference_image",
    )

    model = media.build_media_model(repositories=repositories, save_id=save.id)

    assert _value(_value(model, "character_reference_image"), "id") == reference.id
    assets = {
        _value(asset, "id"): asset
        for asset in _value(model, "media_history")
    }
    assert _value(assets[reference.id], "is_character_reference") is True
    assert _value(assets[reference.id], "can_set_character_reference") is False
    assert _value(assets[reference.id], "character_name") == "Sigil Keeper"
    assert _value(assets[candidate.id], "is_character_reference") is False
    assert _value(assets[candidate.id], "can_set_character_reference") is False
    assert _value(assets[candidate.id], "character_name") == "Sigil Keeper"
    assert _value(assets[scene_candidate.id], "can_set_character_reference") is False
    assert _value(assets[scene_candidate.id], "character_name") is None


def test_media_model_ignores_retired_save_level_character_reference(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    media = _import_media_without_gtk(monkeypatch)
    scenario = repositories.create_scenario(
        type="character_interaction",
        title="Sigil Keeper",
        premise="A private audience with the sigil keeper.",
        player_role="A careful petitioner",
        content={"character_name": "Sigil Keeper"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Audience")
    message = repositories.append_message(
        save_id=save.id,
        message_id="narrator-reference",
        role="narrator",
        speaker_name="Narrator",
        body="The sigil keeper lifts a prism.",
        provider="fake",
        model="fake-chat",
    )
    repositories.add_character(
        save_id=save.id,
        name="Another Character",
        role="Sorts before the scenario character.",
    )
    reference = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=message.id,
        type="image",
        path="media/save-reference/current.png",
        thumbnail_path=None,
        prompt="current reference",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={"kind": "scene_image"},
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="save",
        entity_id=save.id,
        target_type="media_asset",
        target_id=reference.id,
        relation="reference_image",
    )

    model = media.build_media_model(repositories=repositories, save_id=save.id)

    assert _value(model, "character_reference_image") is None
    assets = {
        _value(asset, "id"): asset
        for asset in _value(model, "media_history")
    }
    assert _value(assets[reference.id], "is_character_reference") is False
    assert _value(assets[reference.id], "can_set_character_reference") is False


def test_media_model_marks_file_availability_when_media_dir_is_supplied(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    media = _import_media_without_gtk(monkeypatch)
    save_id = _save_with_media_history(repositories)
    existing_path = tmp_path / "media" / "save-1" / "scene-3.png"
    existing_path.parent.mkdir(parents=True)
    existing_path.write_bytes(b"fake image bytes")

    model = media.build_media_model(
        repositories=repositories,
        save_id=save_id,
        media_dir=tmp_path,
    )

    availability_by_path = {
        _value(asset, "path"): _value(asset, "file_available")
        for asset in _value(model, "media_history")
    }
    assert availability_by_path == {
        "media/save-1/scene-3.png": True,
        "media/save-1/scene-2.png": False,
        "media/save-1/scene-1.png": False,
    }


def test_media_model_omits_archived_images_from_latest_and_history(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    media = _import_media_without_gtk(monkeypatch)
    save_id = _save_with_media_history(repositories)
    latest_asset = repositories.list_media_assets(save_id)[-1]

    repositories.archive_media_asset(
        save_id=save_id,
        media_asset_id=latest_asset.id,
    )
    model = media.build_media_model(repositories=repositories, save_id=save_id)

    latest = _value(model, "latest_scene_image", "latest_image")
    history = list(_value(model, "image_history", "images"))
    assert _value(latest, "path") == "media/save-1/scene-2.png"
    assert [_value(item, "path") for item in history] == [
        "media/save-1/scene-2.png",
        "media/save-1/scene-1.png",
    ]


def test_media_model_exposes_bounded_source_based_prompt_preview(
    repositories: PersistenceRepositories,
    monkeypatch: MonkeyPatch,
) -> None:
    media = _import_media_without_gtk(monkeypatch)
    save_id = _save_with_media_history(repositories)
    long_source = (
        "The tower lens catches fire while Mara shields the signal mirror. "
        * 8
    ).strip()
    internal_prompt = (
        "INTERNAL PROMPT: include scenario instructions, hidden memory, and "
        "future prompt scaffolding that players should not see."
    )
    repositories.append_message(
        save_id=save_id,
        message_id="narrator-long-preview",
        role="narrator",
        speaker_name="Narrator",
        body=long_source,
        provider="fake",
        model="fake-chat",
        token_estimate=120,
    )
    repositories.create_media_asset(
        save_id=save_id,
        source_message_id="narrator-long-preview",
        type="image",
        path="media/save-1/scene-long.png",
        thumbnail_path=None,
        prompt=internal_prompt,
        provider="fake",
        model="fake-image",
        status="succeeded",
    )

    model = media.build_media_model(repositories=repositories, save_id=save_id)

    latest = _value(model, "latest_scene_image", "latest_image")
    preview = _value(latest, "prompt_preview")
    assert preview
    assert len(preview) <= 160
    assert preview != internal_prompt
    assert "INTERNAL PROMPT" not in preview
    assert _value(latest, "prompt", default=preview) != internal_prompt
    assert long_source.startswith(preview.rstrip("."))


def _import_media_without_gtk(monkeypatch: MonkeyPatch) -> Any:
    original_import = builtins.__import__

    def import_without_gtk(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "gi" or name.startswith("gi."):
            raise AssertionError(
                "bragi.application.media must not import GTK/PyGObject"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_gtk)
    sys.modules.pop("bragi.application.media", None)
    return importlib.import_module("bragi.application.media")


def _save_with_media_history(repositories: PersistenceRepositories) -> str:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    other_save = repositories.create_save(
        scenario_id=scenario.id,
        title="Other Watch",
    )
    for message_id, body in [
        ("narrator-1", "The beacon gutters."),
        ("narrator-2", "Ash claws the glass."),
        ("narrator-3", "The tower lens catches fire."),
    ]:
        repositories.append_message(
            save_id=save.id,
            message_id=message_id,
            role="narrator",
            speaker_name="Narrator",
            body=body,
            provider="fake",
            model="fake-chat",
            token_estimate=12,
        )
    repositories.append_message(
        save_id=other_save.id,
        message_id="other-narrator",
        role="narrator",
        speaker_name="Narrator",
        body="This save should not appear.",
        provider="fake",
        model="fake-chat",
        token_estimate=12,
    )
    for index in range(1, 4):
        repositories.create_media_asset(
            save_id=save.id,
            source_message_id=f"narrator-{index}",
            type="image",
            path=f"media/save-1/scene-{index}.png",
            thumbnail_path=None,
            prompt=(
                "The tower lens catches fire."
                if index == 3
                else f"Scene prompt {index}."
            ),
            provider="fake",
            model="fake-image",
            status="succeeded",
        )
    repositories.create_media_asset(
        save_id=other_save.id,
        source_message_id="other-narrator",
        type="image",
        path="media/other-save/scene.png",
        thumbnail_path=None,
        prompt="Other save scene.",
        provider="fake",
        model="fake-image",
        status="succeeded",
    )
    return save.id


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
