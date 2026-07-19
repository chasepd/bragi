from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path

from bragi.application.media import build_media_model
from bragi.persistence.repositories import BragiRepository
from bragi.providers.contracts import (
    ChatRequest,
    ChatResponse,
    ImageRequest,
    ImageResponse,
    ProviderCapability,
    ProviderConfigStatus,
    ProviderModel,
)
from bragi.services.media_service import MediaService

_VALID_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000b49444154789c6360000200000500017a5eab3f"
    "00000000"
    "49454e44ae426082"
)


class FakeImageProvider:
    provider_name = "fake"

    def __init__(self) -> None:
        self.chat_requests: list[ChatRequest] = []
        self.image_requests: list[ImageRequest] = []

    async def validate_config(self) -> ProviderConfigStatus:
        return ProviderConfigStatus(
            provider=self.provider_name,
            configured=True,
            authenticated=True,
        )

    async def list_models(self) -> list[ProviderModel]:
        return [
            ProviderModel(
                provider=self.provider_name,
                model_id="fake-chat",
                display_name="Fake Chat",
                capabilities=frozenset({ProviderCapability.CHAT}),
                context_window=8192,
            ),
            ProviderModel(
                provider=self.provider_name,
                model_id="fake-image",
                display_name="Fake Image",
                capabilities=frozenset({ProviderCapability.IMAGE_GENERATION}),
            )
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        return ChatResponse(
            body="cinematic drafted image prompt",
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 11},
        )

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        self.image_requests.append(request)
        return ImageResponse(
            provider=request.provider,
            model_id=request.model_id,
            image_bytes=_VALID_PNG_BYTES,
        )


def test_fake_provider_image_generation_flow_persists_file_metadata_and_ui_model(
    tmp_path: Path,
) -> None:
    repository = BragiRepository(tmp_path / "bragi.sqlite3")
    try:
        scenario = repository.create_scenario(
            type="full_roleplay",
            title="Frostglass Hall",
            premise="A sealed hall is thawing after a century.",
            player_role="Relic hunter",
            content={"starting_scene": "The mirror nave begins to thaw."},
        )
        save = repository.create_save(scenario_id=scenario.id, title="First Thaw")
        source_message = repository.append_message(
            save_id=save.id,
            role="narrator",
            speaker_name="Narrator",
            body="The frost cracks across the mirror floor.",
            provider="fake",
            model="fake-chat",
            token_estimate=19,
        )
        repository.set_model_preference(
            task="chat",
            provider="fake",
            model_id="fake-chat",
        )
        repository.set_model_preference(
            task="image_generation",
            provider="fake",
            model_id="fake-image",
        )
        provider = FakeImageProvider()
        media_dir = tmp_path / "media"

        asset = asyncio.run(
            MediaService(
                repositories=repository,
                providers={"fake": provider},
                media_dir=media_dir,
                auto_frequency=3,
            ).generate_for_message(
                save_id=save.id,
                source_message_id=source_message.id,
            )
        )

        persisted_assets = repository.list_media_assets(save.id)
        model = build_media_model(repositories=repository, save_id=save.id)
        latest = model.latest_scene_image
        history = list(model.image_history)
        asset_path = _asset_path(media_dir, asset.path)
        thumbnail_path = _assert_private_thumbnail(
            media_dir=media_dir,
            image_path=asset_path,
            thumbnail_path=asset.thumbnail_path,
        )

        assert [
            (
                persisted_asset.id,
                persisted_asset.save_id,
                persisted_asset.source_message_id,
                persisted_asset.type,
                persisted_asset.path,
                persisted_asset.thumbnail_path,
                persisted_asset.prompt,
                persisted_asset.provider,
                persisted_asset.model,
                persisted_asset.status,
            )
            for persisted_asset in persisted_assets
        ] == [
            (
                asset.id,
                asset.save_id,
                asset.source_message_id,
                asset.type,
                asset.path,
                asset.thumbnail_path,
                asset.prompt,
                asset.provider,
                asset.model,
                asset.status,
            )
        ]
        assert persisted_assets[0].created_at is not None
        assert persisted_assets[0].thumbnail_path == asset.thumbnail_path
        _assert_private_thumbnail(
            media_dir=media_dir,
            image_path=asset_path,
            thumbnail_path=persisted_assets[0].thumbnail_path,
        )
        assert asset_path.read_bytes() == _VALID_PNG_BYTES
        assert asset.source_message_id == source_message.id
        assert asset.status == "succeeded"
        assert asset.provider == "fake"
        assert asset.model == "fake-image"
        assert len(provider.chat_requests) == 1
        assert len(provider.image_requests) == 1
        assert provider.image_requests[0].prompt == "cinematic drafted image prompt"
        assert asset.prompt == "cinematic drafted image prompt"
        assert latest is not None
        assert latest.path == asset.path
        assert latest.thumbnail_path == asset.thumbnail_path
        assert latest.thumbnail_path is not None
        assert _asset_path(media_dir, latest.thumbnail_path).samefile(thumbnail_path)
        assert [item.path for item in history] == [asset.path]
        assert [item.thumbnail_path for item in history] == [asset.thumbnail_path]
    finally:
        repository.connection.close()


def _asset_path(media_dir: Path, persisted_path: str) -> Path:
    path = Path(persisted_path)
    if path.is_absolute():
        return path
    return media_dir / path


def _assert_private_thumbnail(
    *,
    media_dir: Path,
    image_path: Path,
    thumbnail_path: str | None,
) -> Path:
    assert thumbnail_path is not None
    thumbnail = _asset_path(media_dir, thumbnail_path)
    assert thumbnail.resolve().is_relative_to(media_dir.resolve())
    assert thumbnail != image_path
    assert thumbnail.is_file()
    assert not thumbnail.samefile(image_path)
    if os.name != "nt":
        assert stat.S_IMODE(thumbnail.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(thumbnail.stat().st_mode) == 0o600
    return thumbnail
