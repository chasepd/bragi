"""Scene image generation and media persistence."""

from __future__ import annotations

import importlib
import json
import re
import zlib
from base64 import b64encode
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Any, cast
from urllib.parse import quote
from uuid import uuid4

from bragi.app_logging import exception_log_fields, log_error_event, log_event
from bragi.content_rating_instructions import (
    CONTENT_RATING_UNCLASSIFIED,
    content_rating_exceeds,
    maximum_content_rating,
)
from bragi.persistence.models import (
    CharacterRecord,
    CharacterTextMessageRecord,
    MediaAssetRecord,
    MessageRecord,
    ModelPreferenceRecord,
)
from bragi.persistence.repositories import PersistenceRepositories
from bragi.private_files import ensure_private_dir, write_private_bytes
from bragi.providers.contracts import (
    ChatMessage,
    ChatPromptPurpose,
    ChatRequest,
    ImageDescriptionRequest,
    ImageReferenceLimitProvider,
    ImageRequest,
    ImageResponse,
    ProviderCapability,
    ProviderClient,
    ProviderRetryProgressCallback,
    StructuredOutputRequest,
    VideoProvider,
    VideoRequest,
    VideoResponse,
)
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.providers.http_client import SAFE_PROVIDER_RESPONSE_HEADERS
from bragi.redaction import redact_text
from bragi.retry_policy import (  # noqa: F401 - test compatibility
    MODEL_OUTPUT_MAX_ATTEMPTS,
    configured_max_attempts,
)
from bragi.services.character_locks import character_field_is_locked
from bragi.services.character_profile_completion import (
    ScenarioCharacterStarter,
    ScenarioStarterReferenceImage,
    content_with_character_starters,
    scenario_character_starters_for_content,
)
from bragi.services.content_rating import effective_content_safety_policy
from bragi.services.content_safety_service import (
    ContentSafetyAction,
    ContentSafetyService,
)
from bragi.services.context_assembly import (
    ContextAssemblyBreakdown,
    ContextAssemblyService,
)
from bragi.services.generation_settings import image_generation_dimensions
from bragi.services.image_style_settings import (
    apply_image_style_preset,
    selected_image_style_preset,
)
from bragi.services.job_lifecycle import JobLifecycleService
from bragi.services.media_content_rating import media_asset_content_rating
from bragi.services.mention_matching import character_name_is_mentioned
from bragi.services.model_capabilities import (
    CHAT_CAPABILITIES,
    IMAGE_GENERATION_CAPABILITIES,
    IMAGE_TO_IMAGE_CAPABILITIES,
    IMAGE_TO_VIDEO_CAPABILITIES,
    MODEL_UNAVAILABLE_REASON,
    TEXT_TO_VIDEO_CAPABILITIES,
    VISION_CAPABILITIES,
    check_model_capabilities,
    find_provider_model,
    model_supports_any_capability,
    model_supports_any_capability_or_unknown,
)
from bragi.services.model_preferences import (
    CHARACTER_IMAGE_EDIT_PURPOSE,
    IMAGE_EDIT_FALLBACK_PURPOSE,
    IMAGE_TO_IMAGE_GENERATION_PURPOSE,
    ROLEPLAY_TYPES,
    SCENE_IMAGE_EDIT_PURPOSE,
    TEXT_MESSAGE_IMAGE_EDIT_PURPOSE,
    image_edit_model_preference,
    model_preference_for_selector,
    roleplay_model_preference,
    roleplay_model_task,
    shared_roleplay_models_enabled,
)
from bragi.services.openrouter_routing_settings import (
    OPENROUTER_PROVIDER_NAME,
    openrouter_app_title_for_task,
    openrouter_routing_payload_for_task,
    request_with_openrouter_routing,
)
from bragi.services.provider_fallbacks import (
    chat_with_fallback,
    structured_output_with_fallback,
)
from bragi.services.sexual_content_safety import (
    is_fade_to_black_message,
)

_THUMBNAIL_WIDTH = 192
_THUMBNAIL_HEIGHT = 144
_MAX_PERSISTED_IMAGE_BYTES = 25 * 1024 * 1024
_MAX_UPLOADED_IMAGE_DECODED_BYTES = 128 * 1024 * 1024
_MAX_PERSISTED_VIDEO_BYTES = 100 * 1024 * 1024
_SUSPICIOUS_FAST_RETRY_MAX_DURATION_MS = 750
_IMAGE_FALLBACK_TASK = "image_fallback"
_IMAGE_EDIT_FALLBACK_TASK = IMAGE_EDIT_FALLBACK_PURPOSE
_VIDEO_FALLBACK_TASK = "video_fallback"
_VENICE_IMAGE_SAFE_MODE_SETTING = "venice_image_safe_mode"
_VENICE_PROVIDER_NAME = "venice"
_ENFORCED_MEDIA_SAFE_MODE_REQUIRED_ERROR = (
    "This account can generate media only through a provider with enforced safe mode"
)
_VENICE_ANIMATION_PROMPT_MAX_CHARS = 2400
_CHARACTER_REFERENCE_PROMPT_MAX_CHARS = 1600
_CHARACTER_REFERENCE_SINGLE_FIELD_MAX_CHARS = 1250
_CHARACTER_REFERENCE_MULTI_FIELD_MAX_CHARS = 650
_CHARACTER_REFERENCE_FALLBACK_FIELD_MAX_CHARS = 480
_CHARACTER_IMAGE_PROMPT_MAX_CHARS = 2400
_CHARACTER_VISUAL_DIRECTION_MAX_CHARS = 700
_CHARACTER_VISUAL_DIRECTION_FIELD_MAX_CHARS = 220
_MAX_DIAGNOSTIC_ERROR_MESSAGE_CHARS = 1000
_CHARACTER_REFERENCE_RELATION = "reference_image"
_CHARACTER_REFERENCE_CANDIDATE_KINDS = frozenset(
    {"character_image", "scene_image", "character_reference"}
)
_UPLOADED_CHARACTER_REFERENCE_PROMPT = "Uploaded character reference image"
_UPLOADED_CHARACTER_TEXT_PHOTO_PROMPT = "Uploaded text photo"
_CHARACTER_TEXT_UPLOADED_PHOTO_TASK = "character_image_description"
_LOCAL_UPLOAD_PROVIDER = "local"
_LOCAL_UPLOAD_MODEL = "upload"
_REQUESTED_MODEL_METADATA_KEY = "requested_model_id"
_RESPONSE_MODEL_METADATA_KEY = "response_model_id"


@dataclass(frozen=True)
class _ImageGenerationResult:
    response: ImageResponse
    request: ImageRequest
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class _ImageFallbackRequest:
    request: ImageRequest
    task: str


@dataclass(frozen=True)
class _VideoGenerationResult:
    response: VideoResponse
    request: VideoRequest
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class _ImageRequestContext:
    preference: ModelPreferenceRecord
    source_media_asset_id: str | None = None
    source_media_path: Path | None = None
    source_media_asset_ids: tuple[str, ...] = ()
    source_media_paths: tuple[Path, ...] = ()
    metadata: dict[str, object] | None = None
    request_task: str | None = None
    character_visual_directions: str = ""


@dataclass(frozen=True)
class _SelectedImageReference:
    character_id: str
    character_name: str
    media_asset_id: str
    media_path: Path


@dataclass(frozen=True)
class PreparedAutomaticImage:
    save_id: str
    source_message_id: str
    scene_context: str
    context_breakdown_json: dict[str, object]
    provider: str
    model_id: str
    narrator_message_count: int
    media_type: str = "image"
    source_media_asset_id: str | None = None
    source_media_path: Path | None = None
    source_media_asset_ids: tuple[str, ...] = ()
    source_media_paths: tuple[Path, ...] = ()
    metadata: dict[str, object] | None = None
    request_task: str | None = None
    character_visual_directions: str = ""


@dataclass(frozen=True)
class CharacterTextUploadedPhoto:
    asset: MediaAssetRecord
    description: str


@dataclass
class _ImageGenerationFailure(ValueError):
    diagnostics: dict[str, object]
    cause: Exception

    def __str__(self) -> str:
        return str(self.cause)


class MediaService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        providers: dict[str, ProviderClient],
        media_dir: Path,
        automatic_enabled: bool = False,
        auto_frequency: int = 3,
        content_safety_service: ContentSafetyService | None = None,
    ) -> None:
        self.repositories = repositories
        self.providers = providers
        self.media_dir = media_dir
        self.automatic_enabled = automatic_enabled
        self.auto_frequency = auto_frequency
        self.content_safety_service = content_safety_service or ContentSafetyService(
            repositories=repositories,
            providers=providers,
        )
        self.jobs = JobLifecycleService(repositories=repositories)

    def upload_scenario_starter_reference_image(
        self,
        *,
        scenario_id: str,
        image_bytes: bytes,
        filename: str | None = None,
        starter_id: str | None = None,
        starter_name: str = "",
        replace_existing: bool = False,
    ) -> ScenarioStarterReferenceImage:
        scenario = self.repositories.get_scenario(scenario_id)
        if scenario is None:
            raise ValueError(f"Unknown scenario id: {scenario_id}")
        content = _scenario_content(scenario.content_json)
        starters = scenario_character_starters_for_content(
            scenario_type=scenario.type,
            content=content,
        )
        index = _scenario_starter_index(
            starters,
            starter_id=starter_id,
            starter_name=starter_name,
        )
        starter = starters[index]
        if starter.reference_image is not None and not replace_existing:
            raise ValueError(
                "Starter reference image already exists; replace it explicitly"
            )
        replaced_reference = starter.reference_image

        _assert_uploaded_image_size(len(image_bytes))
        mime_type, extension = _uploaded_image_mime_type(image_bytes)
        image_id = uuid4().hex
        relative_path = (
            Path("scenario-starters")
            / _safe_path_segment(scenario_id)
            / f"{_safe_path_segment(image_id)}{extension}"
        )
        output_path = self.media_dir / relative_path
        _assert_within_media_dir(media_dir=self.media_dir, output_path=output_path)
        thumbnail_path: str | None = None
        try:
            write_private_bytes(output_path, image_bytes)
            thumbnail_path = _persist_thumbnail(
                media_dir=self.media_dir,
                image_relative_path=relative_path,
                image_path=output_path,
            )
            reference = ScenarioStarterReferenceImage(
                id=image_id,
                path=relative_path.as_posix(),
                thumbnail_path=thumbnail_path,
                mime_type=mime_type,
                prompt_preview=_UPLOADED_CHARACTER_REFERENCE_PROMPT,
                source="uploaded",
                created_at=datetime.now(UTC).isoformat(),
            )
        except Exception:
            self._delete_persisted_files(relative_path.as_posix(), thumbnail_path)
            raise

        updated_starters = list(starters)
        updated_starters[index] = replace(
            starter,
            starter_id=starter.starter_id or f"starter-{uuid4().hex}",
            reference_image=reference,
        )
        try:
            self.repositories.update_scenario(
                scenario_id=scenario.id,
                title=scenario.title,
                premise=scenario.premise,
                player_role=scenario.player_role,
                content=content_with_character_starters(
                    scenario_type=scenario.type,
                    content=content,
                    starters=updated_starters,
                ),
            )
        except Exception:
            self._delete_persisted_files(relative_path.as_posix(), thumbnail_path)
            raise
        self._delete_scenario_starter_reference_files_if_unreferenced(
            replaced_reference,
            starters=updated_starters,
        )
        log_event(
            "media.scenario_starter_reference_uploaded",
            scenario_id=scenario.id,
            starter_id=updated_starters[index].starter_id,
            starter_name=updated_starters[index].name,
            reference_image_id=reference.id,
            mime_type=mime_type,
            byte_count=len(image_bytes),
            filename=filename,
        )
        return reference

    def remove_scenario_starter_reference_image(
        self,
        *,
        scenario_id: str,
        starter_id: str | None = None,
        starter_name: str = "",
    ) -> ScenarioStarterReferenceImage | None:
        scenario = self.repositories.get_scenario(scenario_id)
        if scenario is None:
            raise ValueError(f"Unknown scenario id: {scenario_id}")
        content = _scenario_content(scenario.content_json)
        starters = scenario_character_starters_for_content(
            scenario_type=scenario.type,
            content=content,
        )
        index = _scenario_starter_index(
            starters,
            starter_id=starter_id,
            starter_name=starter_name,
        )
        starter = starters[index]
        removed = starter.reference_image
        if removed is None:
            return None
        updated_starters = list(starters)
        updated_starters[index] = replace(starter, reference_image=None)
        self.repositories.update_scenario(
            scenario_id=scenario.id,
            title=scenario.title,
            premise=scenario.premise,
            player_role=scenario.player_role,
            content=content_with_character_starters(
                scenario_type=scenario.type,
                content=content,
                starters=updated_starters,
            ),
        )
        self._delete_scenario_starter_reference_files_if_unreferenced(
            removed,
            starters=updated_starters,
        )
        log_event(
            "media.scenario_starter_reference_removed",
            scenario_id=scenario.id,
            starter_id=starter.starter_id,
            starter_name=starter.name,
            reference_image_id=removed.id,
        )
        return removed

    def create_character_reference_from_scenario_starter(
        self,
        *,
        save_id: str,
        character_id: str,
        starter: ScenarioCharacterStarter,
    ) -> MediaAssetRecord | None:
        reference = starter.reference_image
        if reference is None:
            return None
        try:
            _assert_scenario_starter_reference_path(reference.path)
        except ValueError:
            return None
        asset_id = uuid4().hex
        path = _copy_reference_media_file(
            media_dir=self.media_dir,
            source_relative_path=reference.path,
            target_save_id=save_id,
            asset_id=asset_id,
        )
        if path is None:
            return None
        thumbnail_path: str | None = None
        asset: MediaAssetRecord | None = None
        try:
            thumbnail_path = _persist_thumbnail(
                media_dir=self.media_dir,
                image_relative_path=Path(path),
                image_path=self.media_dir / path,
            )
            asset = self.repositories.create_media_asset(
                save_id=save_id,
                source_message_id=None,
                type="image",
                path=path,
                thumbnail_path=thumbnail_path,
                prompt=reference.prompt_preview or _UPLOADED_CHARACTER_REFERENCE_PROMPT,
                provider=_LOCAL_UPLOAD_PROVIDER,
                model=_LOCAL_UPLOAD_MODEL,
                status="succeeded",
                mime_type=reference.mime_type,
                metadata={
                    "kind": "character_reference",
                    "character_id": character_id,
                    "source": "scenario_starter",
                    "starter_id": starter.starter_id,
                    "starter_reference_image_id": reference.id,
                },
                asset_id=asset_id,
            )
            _set_character_reference_link(
                repositories=self.repositories,
                save_id=save_id,
                character_id=character_id,
                media_asset_id=asset.id,
            )
        except Exception:
            if asset is not None:
                try:
                    self.repositories.archive_media_asset_only(
                        save_id=save_id,
                        media_asset_id=asset.id,
                    )
                except Exception as cleanup_exc:
                    log_error_event(
                        "media.cleanup_failed",
                        path=asset.path,
                        **exception_log_fields(cleanup_exc),
                    )
            self._delete_persisted_files(path, thumbnail_path)
            raise
        assert asset is not None
        log_event(
            "media.scenario_starter_reference_seeded",
            save_id=save_id,
            character_id=character_id,
            media_asset_id=asset.id,
            starter_id=starter.starter_id,
            reference_image_id=reference.id,
        )
        return asset

    async def generate_for_message(
        self,
        *,
        save_id: str,
        source_message_id: str,
        job_context: str | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        current_user_id: str | None = None,
    ) -> MediaAssetRecord:
        source_message = _source_message(
            messages=self.repositories.list_messages(save_id),
            source_message_id=source_message_id,
        )
        if source_message is None:
            raise ValueError(f"Unknown source message id: {source_message_id}")
        _raise_if_safety_transition_source(source_message)

        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="image_generation",
        )
        if preference is None:
            raise ValueError("No image generation model preference configured")
        scene_context, context_breakdown = self._build_scene_context_with_breakdown(
            save_id=save_id,
            source_message_id=source_message_id,
        )
        request_context = self._scene_image_request_context(
            save_id=save_id,
            source_message_id=source_message_id,
            fallback_preference=preference,
        )
        return await self._generate_for_message_with_context(
            save_id=save_id,
            source_message_id=source_message_id,
            scene_context=scene_context,
            context_breakdown_json=context_breakdown.to_json(),
            preference=request_context.preference,
            job_context=job_context,
            retry_progress_callback=retry_progress_callback,
            source_media_asset_id=request_context.source_media_asset_id,
            source_media_path=request_context.source_media_path,
            source_media_asset_ids=request_context.source_media_asset_ids,
            source_media_paths=request_context.source_media_paths,
            metadata=request_context.metadata,
            request_task=request_context.request_task,
            character_visual_directions=request_context.character_visual_directions,
            current_user_id=current_user_id,
        )

    async def regenerate_asset_with_prompt(
        self,
        *,
        save_id: str,
        media_asset_id: str,
        prompt: str,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        current_user_id: str | None = None,
    ) -> MediaAssetRecord:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Image prompt is required")
        content_safety = (
            effective_content_safety_policy(
                self.repositories,
                user_id=current_user_id,
            )
            if current_user_id is not None
            else None
        )
        original = _media_asset_by_id(
            self.repositories.list_media_assets(save_id),
            media_asset_id,
        )
        if original is None:
            raise ValueError(f"Unknown media asset id: {media_asset_id}")
        if original.type != "image":
            raise ValueError("Only image media assets can be regenerated")
        if original.provider not in self.providers:
            raise ValueError(
                f"Image generation provider is unavailable: {original.provider}"
            )
        safety = await self.content_safety_service.review_media_prompt(
            prompt=prompt,
            content_rating=(
                content_safety.rating if content_safety is not None else "pg-13"
            ),
            save_id=save_id,
            source_provider=original.provider,
            source_model_id=original.model,
        )
        if safety.action is not ContentSafetyAction.ALLOW:
            raise ValueError("Image prompt exceeds the selected content rating")
        _raise_unless_enforced_safe_mode_provider(
            provider=original.provider,
            force_safe_mode=(
                content_safety.force_venice_safe_mode
                if content_safety is not None
                else False
            ),
        )

        metadata = _media_asset_metadata(original)
        request_source_message_id = _replacement_request_source_message_id(
            original,
            metadata,
        )
        if request_source_message_id is not None:
            source_message = _source_message(
                messages=self.repositories.list_messages(save_id),
                source_message_id=request_source_message_id,
            )
            if source_message is not None:
                _raise_if_safety_transition_source(source_message)
        source_media_asset_ids = _replacement_source_media_asset_ids(
            original,
            metadata,
        )
        source_media_paths = self._replacement_source_media_paths(
            save_id=save_id,
            source_media_asset_ids=source_media_asset_ids,
        )
        primary_source_media_asset_id = (
            source_media_asset_ids[0] if source_media_asset_ids else None
        )
        primary_source_media_path = (
            source_media_paths[0] if source_media_paths else None
        )
        request_task = _replacement_image_request_task(
            original,
            metadata,
            has_source_media=bool(source_media_asset_ids),
        )
        request_model_id = _replacement_request_model_id(
            repositories=self.repositories,
            save_id=save_id,
            asset=original,
            metadata=metadata,
            request_task=request_task,
        )
        preference = ModelPreferenceRecord(
            id=f"regenerate-{original.id}",
            task=request_task,
            provider=original.provider,
            model_id=request_model_id,
        )
        job = self.jobs.create_running(
            save_id=save_id,
            type="image_regeneration",
            request_context={
                **metadata,
                "kind": "manual_image_regeneration",
                "source_message_id": request_source_message_id,
                "regenerated_from_media_asset_id": original.id,
                "provider": preference.provider,
                "model": preference.model_id,
                "task": request_task,
                "prompt": prompt,
                "prompt_chars": len(prompt),
                "source_media_asset_ids": list(source_media_asset_ids),
            },
            payload={
                "save_id": save_id,
                "source_message_id": original.source_message_id,
                "request_source_message_id": request_source_message_id,
                "regenerated_from_media_asset_id": original.id,
                "provider": preference.provider,
                "model": preference.model_id,
                "source_media_asset_id": primary_source_media_asset_id,
                "source_media_asset_ids": list(source_media_asset_ids),
                "prompt_chars": len(prompt),
                **_venice_safe_mode_diagnostics(
                    provider=preference.provider,
                    safe_mode=_request_safe_mode(
                        repositories=self.repositories,
                        provider=preference.provider,
                        save_id=save_id,
                        current_user_id=current_user_id,
                    ),
                ),
            },
        )
        started_at = perf_counter()
        path: str | None = None
        thumbnail_path: str | None = None
        replacement_committed = False
        try:
            request = await self._image_request(
                provider=preference.provider,
                model_id=preference.model_id,
                prompt=prompt,
                save_id=save_id,
                source_message_id=request_source_message_id,
                retry_progress_callback=retry_progress_callback,
                source_media_asset_id=primary_source_media_asset_id,
                source_media_path=primary_source_media_path,
                source_media_asset_ids=source_media_asset_ids,
                source_media_paths=source_media_paths,
                task=request_task,
                route_openrouter=False,
                current_user_id=current_user_id,
                reviewed_content_rating=safety.minimum_rating,
            )
            generation = await self._generate_image_without_fallback(
                save_id=save_id,
                request=request,
            )
            path, thumbnail_path = self._persist_image(
                response=generation.response,
                save_id=save_id,
                source_message_id=request_source_message_id,
                generation_id=job.id,
            )
            response = generation.response
            self.repositories.begin_immediate_transaction()
            try:
                asset = self.repositories.create_media_asset(
                    save_id=save_id,
                    source_message_id=original.source_message_id,
                    type="image",
                    path=path,
                    thumbnail_path=thumbnail_path,
                    prompt=prompt,
                    provider=response.provider,
                    model=_persisted_image_model(generation),
                    status="succeeded",
                    metadata=_image_asset_metadata(
                        {
                            **metadata,
                            "regenerated_from_media_asset_id": original.id,
                        },
                        generation=generation,
                    ),
                    source_media_asset_id=primary_source_media_asset_id,
                )
                self.repositories.replace_character_text_attachment_media_asset(
                    save_id=save_id,
                    old_media_asset_id=original.id,
                    new_media_asset_id=asset.id,
                )
                _replace_character_reference_media_links(
                    repositories=self.repositories,
                    save_id=save_id,
                    old_media_asset_id=original.id,
                    new_media_asset_id=asset.id,
                )
                self.repositories.replace_media_asset_source_references(
                    save_id=save_id,
                    old_media_asset_id=original.id,
                    new_media_asset_id=asset.id,
                )
                archived = self.repositories.archive_media_asset_only(
                    save_id=save_id,
                    media_asset_id=original.id,
                )
                if archived is None:
                    raise ValueError("Media asset disappeared during replacement")
                self.jobs.succeed(
                    job.id,
                    result={
                        "media_asset_id": asset.id,
                        "replaced_media_asset_id": original.id,
                        "path": asset.path,
                        "prompt_chars": len(prompt),
                        "provider": asset.provider,
                        "model": asset.model,
                        "source_media_asset_id": primary_source_media_asset_id,
                        "source_media_asset_ids": list(source_media_asset_ids),
                        **generation.diagnostics,
                    },
                )
                self.repositories.commit_transaction()
                replacement_committed = True
            except Exception:
                self.repositories.rollback_transaction()
                self._delete_persisted_files(path, thumbnail_path)
                path = None
                thumbnail_path = None
                raise
        except Exception as exc:
            if not replacement_committed:
                self._delete_persisted_files(path, thumbnail_path)
            self.jobs.fail(
                job.id,
                error=redact_text(str(exc)) or exc.__class__.__name__,
                result=_failed_image_result(exc=exc),
            )
            log_error_event(
                "job.failed",
                job_id=job.id,
                job_type=job.type,
                save_id=save_id,
                source_message_id=original.source_message_id,
                job_context="manual_image_regeneration",
                provider=preference.provider,
                model=preference.model_id,
                duration_ms=_elapsed_ms(started_at),
                **exception_log_fields(exc),
            )
            raise
        output_path = self.media_dir / asset.path
        log_event(
            "media.image_regenerated",
            job_id=job.id,
            save_id=save_id,
            source_message_id=original.source_message_id,
            media_asset_id=asset.id,
            replaced_media_asset_id=original.id,
            provider=asset.provider,
            model=asset.model,
            output_path=asset.path,
            byte_count=output_path.stat().st_size if output_path.is_file() else None,
            duration_ms=_elapsed_ms(started_at),
        )
        log_event(
            "job.succeeded",
            job_id=job.id,
            job_type=job.type,
            save_id=save_id,
            job_context="manual_image_regeneration",
        )
        return asset

    async def generate_character_reference(
        self,
        *,
        save_id: str,
        character_id: str | None = None,
        source_message_id: str | None = None,
        job_context: str | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        replace_existing: bool = False,
        current_user_id: str | None = None,
    ) -> MediaAssetRecord:
        character = _character_for_reference(
            repositories=self.repositories,
            save_id=save_id,
            character_id=character_id,
        )
        if character is None:
            raise ValueError("No character is available for reference generation")

        existing = _character_reference_asset(
            repositories=self.repositories,
            save_id=save_id,
            character_id=character.id,
            include_generated_fallback=False,
        )
        if existing is not None and not replace_existing:
            return existing

        resolved_source_message_id = (
            source_message_id
            or character.source_message_id
            or _first_narrator_message_id(self.repositories.list_messages(save_id))
        )
        if resolved_source_message_id is None:
            raise ValueError("Character reference generation requires a source message")
        source_message = _source_message(
            messages=self.repositories.list_messages(save_id),
            source_message_id=resolved_source_message_id,
        )
        if source_message is not None:
            _raise_if_safety_transition_source(source_message)

        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="image_generation",
        )
        if preference is None:
            raise ValueError("No image generation model preference configured")

        image_style_preset = selected_image_style_preset(
            self.repositories,
            save_id=save_id,
        )
        prompt = apply_image_style_preset(
            _character_reference_prompt(character),
            preset_id=image_style_preset,
        )
        job = self.jobs.create_running(
            save_id=save_id,
            type="character_reference_image",
            request_context={
                "kind": job_context or "manual_character_reference",
                "source_message_id": resolved_source_message_id,
                "character_id": character.id,
                "provider": preference.provider,
                "model": preference.model_id,
                "prompt": prompt,
                "prompt_chars": len(prompt),
                "image_style_preset": image_style_preset,
            },
            payload={
                "save_id": save_id,
                "source_message_id": resolved_source_message_id,
                "job_context": job_context,
                "character_id": character.id,
                "provider": preference.provider,
                "model": preference.model_id,
                "prompt_chars": len(prompt),
                "image_style_preset": image_style_preset,
                **_venice_safe_mode_diagnostics(
                    provider=preference.provider,
                    safe_mode=_request_safe_mode(
                        repositories=self.repositories,
                        provider=preference.provider,
                        save_id=save_id,
                        current_user_id=current_user_id,
                    ),
                ),
            },
        )
        path: str | None = None
        thumbnail_path: str | None = None
        asset_created = False
        try:
            requirement_error = _image_model_requirement_error(
                repositories=self.repositories,
                provider=preference.provider,
                model_id=preference.model_id,
                source_media_asset_id=None,
            )
            if requirement_error is not None:
                raise ValueError(requirement_error)
            generation = await self._generate_image_with_optional_fallback(
                save_id=save_id,
                request=await self._image_request(
                    provider=preference.provider,
                    model_id=preference.model_id,
                    prompt=prompt,
                    save_id=save_id,
                    source_message_id=resolved_source_message_id,
                    retry_progress_callback=retry_progress_callback,
                    current_user_id=current_user_id,
                ),
            )
            generation, path, thumbnail_path = (
                await self._persist_generated_image_with_optional_fallback(
                    save_id=save_id,
                    source_message_id=resolved_source_message_id,
                    generation_id=job.id,
                    generation=generation,
                )
            )
            response = generation.response
            asset = self.repositories.create_media_asset(
                save_id=save_id,
                source_message_id=resolved_source_message_id,
                type="image",
                path=path,
                thumbnail_path=thumbnail_path,
                prompt=prompt,
                provider=response.provider,
                model=_persisted_image_model(generation),
                status="succeeded",
                metadata=_image_asset_metadata(
                    {
                        "kind": "character_reference",
                        "character_id": character.id,
                    },
                    generation=generation,
                ),
            )
            asset_created = True
            _set_character_reference_link(
                repositories=self.repositories,
                save_id=save_id,
                character_id=character.id,
                media_asset_id=asset.id,
            )
            self.jobs.succeed(
                job.id,
                result={
                    "media_asset_id": asset.id,
                    "path": asset.path,
                    "character_id": character.id,
                    "provider": asset.provider,
                    "model": asset.model,
                    **generation.diagnostics,
                },
            )
        except Exception as exc:
            if not asset_created:
                self._delete_persisted_files(path, thumbnail_path)
            self.jobs.fail(
                job.id,
                error=redact_text(str(exc)) or exc.__class__.__name__,
                result=_failed_image_result(exc=exc),
            )
            raise
        return asset

    def set_character_reference_image(
        self,
        *,
        save_id: str,
        media_asset_id: str,
        character_id: str | None = None,
    ) -> MediaAssetRecord:
        character = _character_for_reference(
            repositories=self.repositories,
            save_id=save_id,
            character_id=character_id,
        )
        if character is None:
            raise ValueError("No character is available for reference image selection")

        asset = _media_asset_by_id(
            self.repositories.list_media_assets(save_id),
            media_asset_id,
        )
        if asset is None:
            raise ValueError(f"Unknown media asset id: {media_asset_id}")
        if (
            asset.type != "image"
            or asset.status != "succeeded"
            or _media_asset_kind(asset) not in _CHARACTER_REFERENCE_CANDIDATE_KINDS
        ):
            raise ValueError(
                "Only generated character, scene, or uploaded reference images "
                "can be set as character references"
            )

        image_path = self.media_dir / asset.path
        _assert_within_media_dir(media_dir=self.media_dir, output_path=image_path)
        if not image_path.is_file():
            raise ValueError("Character reference image file is unavailable")

        _set_character_reference_link(
            repositories=self.repositories,
            save_id=save_id,
            character_id=character.id,
            media_asset_id=asset.id,
        )
        log_event(
            "media.character_reference_updated",
            save_id=save_id,
            character_id=character.id,
            media_asset_id=asset.id,
        )
        return asset

    def upload_character_reference_image(
        self,
        *,
        save_id: str,
        image_bytes: bytes,
        filename: str | None = None,
        character_id: str | None = None,
        replace_existing: bool = False,
    ) -> MediaAssetRecord:
        if self.repositories.get_save(save_id) is None:
            raise ValueError(f"Unknown save id: {save_id}")

        character = _character_for_reference(
            repositories=self.repositories,
            save_id=save_id,
            character_id=character_id,
        )
        if character is None:
            raise ValueError("No character is available for reference upload")

        existing = _character_reference_asset(
            repositories=self.repositories,
            save_id=save_id,
            character_id=character.id,
            include_generated_fallback=False,
        )
        if existing is not None and not replace_existing:
            raise ValueError(
                "Character reference image already exists; replace it explicitly"
            )

        _assert_uploaded_image_size(len(image_bytes))
        mime_type, extension = _uploaded_image_mime_type(image_bytes)
        asset_id = uuid4().hex
        relative_path = (
            Path(_safe_path_segment(save_id))
            / "uploads"
            / f"{_safe_path_segment(asset_id)}{extension}"
        )
        output_path = self.media_dir / relative_path
        _assert_within_media_dir(media_dir=self.media_dir, output_path=output_path)
        thumbnail_path: str | None = None
        try:
            write_private_bytes(output_path, image_bytes)
            thumbnail_path = _persist_thumbnail(
                media_dir=self.media_dir,
                image_relative_path=relative_path,
                image_path=output_path,
            )
            asset = self.repositories.create_media_asset(
                save_id=save_id,
                source_message_id=None,
                type="image",
                path=relative_path.as_posix(),
                thumbnail_path=thumbnail_path,
                prompt=_UPLOADED_CHARACTER_REFERENCE_PROMPT,
                provider=_LOCAL_UPLOAD_PROVIDER,
                model=_LOCAL_UPLOAD_MODEL,
                status="succeeded",
                mime_type=mime_type,
                metadata={
                    "kind": "character_reference",
                    "source": "uploaded",
                    "character_id": character.id,
                },
                asset_id=asset_id,
            )
        except Exception:
            self._delete_persisted_files(relative_path.as_posix(), thumbnail_path)
            raise

        _set_character_reference_link(
            repositories=self.repositories,
            save_id=save_id,
            character_id=character.id,
            media_asset_id=asset.id,
        )
        log_event(
            "media.character_reference_uploaded",
            save_id=save_id,
            character_id=character.id,
            media_asset_id=asset.id,
            mime_type=mime_type,
            byte_count=len(image_bytes),
            replaced=existing is not None,
        )
        return asset

    def clone_character_reference_image(
        self,
        *,
        source_save_id: str,
        target_save_id: str,
    ) -> MediaAssetRecord | None:
        if self.repositories.get_save(source_save_id) is None:
            raise ValueError(f"Unknown source save id: {source_save_id}")
        if self.repositories.get_save(target_save_id) is None:
            raise ValueError(f"Unknown target save id: {target_save_id}")

        source_asset = _linked_character_reference_asset(
            repositories=self.repositories,
            save_id=source_save_id,
        )
        if source_asset is None:
            return None

        target_character = _primary_character_for_reference(
            repositories=self.repositories,
            save_id=target_save_id,
        )
        if target_character is None:
            raise ValueError("No character is available for reference clone")

        existing = _linked_character_reference_asset(
            repositories=self.repositories,
            save_id=target_save_id,
        )
        if existing is not None:
            return existing

        asset_id = uuid4().hex
        copied_path: str | None = None
        copied_thumbnail_path: str | None = None
        try:
            copied = _copy_reference_media_file(
                media_dir=self.media_dir,
                source_relative_path=source_asset.path,
                target_save_id=target_save_id,
                asset_id=asset_id,
            )
            if copied is None:
                raise ValueError("Character reference image file is unavailable")
            copied_path = copied
            if source_asset.thumbnail_path is not None:
                copied_thumbnail_path = _copy_reference_media_file(
                    media_dir=self.media_dir,
                    source_relative_path=source_asset.thumbnail_path,
                    target_save_id=target_save_id,
                    asset_id=asset_id,
                    thumbnail=True,
                    missing_allowed=True,
                )
            asset = self.repositories.create_media_asset(
                asset_id=asset_id,
                save_id=target_save_id,
                source_message_id=None,
                type=source_asset.type,
                path=copied_path,
                thumbnail_path=copied_thumbnail_path,
                prompt=source_asset.prompt,
                provider=source_asset.provider,
                model=source_asset.model,
                status=source_asset.status,
                mime_type=source_asset.mime_type,
                metadata={
                    **_media_asset_metadata(source_asset),
                    "kind": "character_reference",
                    "source": "continuation_clone",
                    "source_save_id": source_save_id,
                    "source_media_asset_id": source_asset.id,
                    "character_id": target_character.id,
                },
            )
            _set_character_reference_link(
                repositories=self.repositories,
                save_id=target_save_id,
                character_id=target_character.id,
                media_asset_id=asset.id,
            )
        except Exception:
            self._delete_persisted_files(copied_path, copied_thumbnail_path)
            raise

        log_event(
            "media.character_reference_cloned",
            source_save_id=source_save_id,
            target_save_id=target_save_id,
            source_media_asset_id=source_asset.id,
            media_asset_id=asset.id,
            character_id=target_character.id,
        )
        return asset

    def remove_character_reference_image(
        self,
        *,
        save_id: str,
        character_id: str | None = None,
    ) -> MediaAssetRecord | None:
        if self.repositories.get_save(save_id) is None:
            raise ValueError(f"Unknown save id: {save_id}")

        character = _character_for_reference(
            repositories=self.repositories,
            save_id=save_id,
            character_id=character_id,
        )
        if character is None:
            raise ValueError("No character is available for reference removal")

        media_assets = {
            asset.id: asset
            for asset in self.repositories.list_media_assets(save_id)
            if asset.type == "image" and asset.status == "succeeded"
        }
        removed: MediaAssetRecord | None = None
        for link in self.repositories.list_entity_links(save_id):
            if not (
                link.target_type == "media_asset"
                and link.relation == _CHARACTER_REFERENCE_RELATION
            ):
                continue
            is_character_link = (
                link.entity_type == "character" and link.entity_id == character.id
            )
            if is_character_link:
                removed = media_assets.get(link.target_id) or removed
                self.repositories.delete_entity_link(link.id)
        if removed is not None:
            log_event(
                "media.character_reference_removed",
                save_id=save_id,
                character_id=character.id,
                media_asset_id=removed.id,
            )
        return removed

    async def generate_character_image_for_message(
        self,
        *,
        save_id: str,
        source_message_id: str,
        character_id: str,
        job_context: str | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        current_user_id: str | None = None,
    ) -> MediaAssetRecord:
        if _scenario_type_for_save(
            repositories=self.repositories,
            save_id=save_id,
        ) not in ROLEPLAY_TYPES:
            raise ValueError("Character images require a roleplay save")
        source_message = _source_message(
            messages=self.repositories.list_messages(save_id),
            source_message_id=source_message_id,
        )
        if source_message is None:
            raise ValueError(f"Unknown source message id: {source_message_id}")
        _raise_if_safety_transition_source(source_message)

        character = self.repositories.get_character(character_id)
        if character is None or character.save_id != save_id:
            raise ValueError(f"Unknown character id: {character_id}")
        present_character_ids = _present_character_ids_for_message(
            repositories=self.repositories,
            save_id=save_id,
            source_message_id=source_message_id,
        )
        if character.id not in present_character_ids:
            raise ValueError("Selected character is not present in this scene")
        preference = self._character_image_preference(save_id=save_id)
        reference = _linked_character_reference_asset(
            repositories=self.repositories,
            save_id=save_id,
            character_id=character.id,
        )
        if reference is None:
            raise ValueError("Selected character does not have a reference image")
        request_context = self._character_image_request_context(
            preference=preference,
            reference=reference,
            character_id=character.id,
            character_name=character.name,
            origin="message_scene",
        )
        scene_context, context_breakdown = self._build_scene_context_with_breakdown(
            save_id=save_id,
            source_message_id=source_message_id,
        )
        character = (
            await self._ensure_current_clothing(
                save_id=save_id,
                characters=(character,),
                image_context=scene_context,
            )
        )[0]
        prompt = _solo_character_scene_image_prompt(
            character=character,
            character_name=character.name,
            action_context=source_message.body,
            scene_context=scene_context,
        )
        return await self._generate_character_image_asset(
            save_id=save_id,
            source_message_id=source_message_id,
            request_source_message_id=source_message_id,
            prompt=prompt,
            scene_context=scene_context,
            context_breakdown_json=context_breakdown.to_json(),
            preference=request_context.preference,
            job_context=job_context,
            retry_progress_callback=retry_progress_callback,
            source_media_asset_id=request_context.source_media_asset_id,
            source_media_path=request_context.source_media_path,
            source_media_asset_ids=request_context.source_media_asset_ids,
            source_media_paths=request_context.source_media_paths,
            metadata=request_context.metadata,
            request_task=request_context.request_task,
            job_type="character_image_generation",
            current_user_id=current_user_id,
        )

    async def generate_character_image_for_character(
        self,
        *,
        save_id: str,
        character_id: str,
        instructions: str = "",
        job_context: str | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        current_user_id: str | None = None,
    ) -> MediaAssetRecord:
        character = self.repositories.get_character(character_id)
        if character is None or character.save_id != save_id:
            raise ValueError(f"Unknown character id: {character_id}")
        preference = self._character_image_preference(save_id=save_id)
        reference = _linked_character_reference_asset(
            repositories=self.repositories,
            save_id=save_id,
            character_id=character.id,
        )
        if reference is None:
            raise ValueError("Selected character does not have a reference image")
        request_context = self._character_image_request_context(
            preference=preference,
            reference=reference,
            character_id=character.id,
            character_name=character.name,
            origin="character_registry",
        )
        character = (
            await self._ensure_current_clothing(
                save_id=save_id,
                characters=(character,),
                image_context=instructions,
            )
        )[0]
        prompt = _solo_character_registry_image_prompt(
            character=character,
            character_name=character.name,
            instructions=instructions,
        )
        request_source_message_id = (
            character.source_message_id
            or reference.source_message_id
            or f"character-{character.id}"
        )
        return await self._generate_character_image_asset(
            save_id=save_id,
            source_message_id=None,
            request_source_message_id=request_source_message_id,
            prompt=prompt,
            scene_context="",
            context_breakdown_json={},
            preference=request_context.preference,
            job_context=job_context,
            retry_progress_callback=retry_progress_callback,
            source_media_asset_id=request_context.source_media_asset_id,
            source_media_path=request_context.source_media_path,
            source_media_asset_ids=request_context.source_media_asset_ids,
            source_media_paths=request_context.source_media_paths,
            metadata=request_context.metadata,
            request_task=request_context.request_task,
            job_type="character_image_generation",
            current_user_id=current_user_id,
        )

    async def generate_character_text_character_image(
        self,
        *,
        save_id: str,
        text_message: CharacterTextMessageRecord,
        character: CharacterRecord,
        visual_prompt: str,
        scene_context: str,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        current_user_id: str | None = None,
    ) -> MediaAssetRecord:
        if character.save_id != save_id or text_message.save_id != save_id:
            raise ValueError("Character text image source does not belong to save")
        if text_message.character_id != character.id:
            raise ValueError("Character text image character mismatch")
        preference = self._image_edit_preference(
            save_id=save_id,
            purpose=TEXT_MESSAGE_IMAGE_EDIT_PURPOSE,
        )
        reference = _linked_character_reference_asset(
            repositories=self.repositories,
            save_id=save_id,
            character_id=character.id,
        )
        if reference is None:
            raise ValueError("Text character does not have a reference image")
        request_context = self._character_image_request_context(
            preference=preference,
            reference=reference,
            character_id=character.id,
            character_name=character.name,
            origin="character_text",
        )
        character = (
            await self._ensure_current_clothing(
                save_id=save_id,
                characters=(character,),
                image_context="\n\n".join(
                    part
                    for part in (text_message.body, visual_prompt, scene_context)
                    if part.strip()
                ),
            )
        )[0]
        visual_prompt = _prompt_with_current_clothing_direction(
            visual_prompt,
            character=character,
        )
        prompt = _solo_character_text_image_prompt(
            character=character,
            character_name=character.name,
            text_body=text_message.body,
            visual_prompt=visual_prompt,
            scene_context=scene_context,
        )
        metadata = {
            **(request_context.metadata or {}),
            "kind": "character_text_character_image",
            "text_message_id": text_message.id,
            "thread_id": text_message.thread_id,
            "character_id": character.id,
            "content_rating": maximum_content_rating(
                (text_message.content_rating, character.content_rating)
            ),
        }
        return await self._generate_character_image_asset(
            save_id=save_id,
            source_message_id=None,
            request_source_message_id=text_message.id,
            prompt=prompt,
            scene_context=scene_context,
            context_breakdown_json={},
            preference=request_context.preference,
            job_context="character_text_attachment",
            retry_progress_callback=retry_progress_callback,
            source_media_asset_id=request_context.source_media_asset_id,
            source_media_path=request_context.source_media_path,
            source_media_asset_ids=request_context.source_media_asset_ids,
            source_media_paths=request_context.source_media_paths,
            metadata=metadata,
            request_task=request_context.request_task,
            job_type="character_text_image_generation",
            current_user_id=current_user_id,
        )

    async def generate_character_text_object_context_image(
        self,
        *,
        save_id: str,
        text_message: CharacterTextMessageRecord,
        character: CharacterRecord,
        visual_prompt: str,
        scene_context: str,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        current_user_id: str | None = None,
    ) -> MediaAssetRecord:
        if character.save_id != save_id or text_message.save_id != save_id:
            raise ValueError("Character text image source does not belong to save")
        if text_message.character_id != character.id:
            raise ValueError("Character text image character mismatch")
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="image_generation",
        )
        if preference is None:
            raise ValueError("No image generation model preference configured")
        if preference.provider not in self.providers:
            raise ValueError(
                f"Image generation provider is unavailable: {preference.provider}"
            )
        prompt = _object_context_text_image_prompt(
            character_name=character.name,
            text_body=text_message.body,
            visual_prompt=visual_prompt,
            scene_context=scene_context,
        )
        return await self._generate_prompted_image_asset(
            save_id=save_id,
            source_message_id=None,
            request_source_message_id=text_message.id,
            prompt=prompt,
            scene_context=scene_context,
            context_breakdown_json={},
            preference=preference,
            job_context="character_text_attachment",
            retry_progress_callback=retry_progress_callback,
            metadata={
                "kind": "character_text_object_context_image",
                "text_message_id": text_message.id,
                "thread_id": text_message.thread_id,
                "character_id": character.id,
                "character_name": character.name,
                "content_rating": maximum_content_rating(
                    (text_message.content_rating, character.content_rating)
                ),
            },
            job_type="character_text_image_generation",
            request_task="image_generation",
            current_user_id=current_user_id,
        )

    async def upload_character_text_player_photo(
        self,
        *,
        save_id: str,
        text_message: CharacterTextMessageRecord,
        sender_character_id: str,
        image_bytes: bytes,
        filename: str | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
    ) -> CharacterTextUploadedPhoto:
        del filename, retry_progress_callback
        if text_message.save_id != save_id:
            raise ValueError("Character text photo source does not belong to save")
        if text_message.sender != "player":
            raise ValueError("Only player text messages can include uploaded photos")
        if text_message.sender_character_id != sender_character_id:
            raise ValueError("Text photo sender character mismatch")
        sender = self.repositories.get_character(sender_character_id)
        if sender is None or sender.save_id != save_id:
            raise ValueError("Text photo sender character is no longer available")

        _assert_uploaded_image_size(len(image_bytes))
        mime_type, extension = _uploaded_image_mime_type(image_bytes)
        description, vision_provider, vision_model = (
            await self._describe_character_text_player_photo(
                save_id=save_id,
                text_message=text_message,
                image_bytes=image_bytes,
                mime_type=mime_type,
            )
        )

        asset_id = uuid4().hex
        relative_path = (
            Path(_safe_path_segment(save_id))
            / "texts"
            / f"{_safe_path_segment(asset_id)}{extension}"
        )
        output_path = self.media_dir / relative_path
        _assert_within_media_dir(media_dir=self.media_dir, output_path=output_path)
        thumbnail_path: str | None = None
        asset: MediaAssetRecord | None = None
        try:
            write_private_bytes(output_path, image_bytes)
            thumbnail_path = _persist_thumbnail(
                media_dir=self.media_dir,
                image_relative_path=relative_path,
                image_path=output_path,
            )
            asset = self.repositories.create_media_asset(
                save_id=save_id,
                source_message_id=None,
                type="image",
                path=relative_path.as_posix(),
                thumbnail_path=thumbnail_path,
                prompt=_UPLOADED_CHARACTER_TEXT_PHOTO_PROMPT,
                provider=_LOCAL_UPLOAD_PROVIDER,
                model=_LOCAL_UPLOAD_MODEL,
                status="succeeded",
                mime_type=mime_type,
                metadata={
                    "kind": "character_text_uploaded_photo",
                    "source": "uploaded",
                    "thread_id": text_message.thread_id,
                    "text_message_id": text_message.id,
                    "sender_character_id": sender_character_id,
                    "description": description,
                    "vision_provider": vision_provider,
                    "vision_model": vision_model,
                },
                asset_id=asset_id,
            )
        except Exception:
            if asset is not None:
                try:
                    self.repositories.archive_media_asset_only(
                        save_id=save_id,
                        media_asset_id=asset.id,
                    )
                except Exception as cleanup_exc:
                    log_error_event(
                        "media.cleanup_failed",
                        path=asset.path,
                        **exception_log_fields(cleanup_exc),
                    )
            self._delete_persisted_files(relative_path.as_posix(), thumbnail_path)
            raise
        log_event(
            "media.character_text_uploaded_photo",
            save_id=save_id,
            thread_id=text_message.thread_id,
            text_message_id=text_message.id,
            media_asset_id=asset.id,
            mime_type=mime_type,
            byte_count=len(image_bytes),
        )
        return CharacterTextUploadedPhoto(asset=asset, description=description)

    async def _describe_character_text_player_photo(
        self,
        *,
        save_id: str,
        text_message: CharacterTextMessageRecord,
        image_bytes: bytes,
        mime_type: str,
    ) -> tuple[str, str, str]:
        preference = model_preference_for_selector(
            self.repositories,
            _CHARACTER_TEXT_UPLOADED_PHOTO_TASK,
            save_id=save_id,
        )
        if preference is None:
            raise ValueError("No Image Details vision model configured")
        provider = self.providers.get(preference.provider)
        describe_image = getattr(provider, "describe_image", None)
        if not callable(describe_image):
            raise ValueError("Configured Image Details provider is unavailable")
        if not model_supports_any_capability_or_unknown(
            self.repositories,
            provider=preference.provider,
            model_id=preference.model_id,
            required=VISION_CAPABILITIES,
        ):
            raise ValueError("Configured Image Details model does not support vision")
        encoded = b64encode(image_bytes).decode("ascii")
        request = ImageDescriptionRequest(
            provider=preference.provider,
            model_id=preference.model_id,
            image_url=f"data:{mime_type};base64,{encoded}",
            system_prompt=_character_text_uploaded_photo_system_prompt(),
            prompt=_character_text_uploaded_photo_description_prompt(
                text_message=text_message,
            ),
            temperature=0.1,
            max_output_tokens=10_000,
            openrouter_app_title=openrouter_app_title_for_task(
                _CHARACTER_TEXT_UPLOADED_PHOTO_TASK
            ),
        )
        request = request_with_openrouter_routing(
            self.repositories,
            request,
            task=_CHARACTER_TEXT_UPLOADED_PHOTO_TASK,
            save_id=save_id,
        )
        response = await describe_image(request)
        description = response.description.strip()
        if not description:
            raise ValueError("Image Details vision model returned an empty description")
        return description, response.provider, response.model_id

    def cleanup_character_text_uploaded_photo(
        self,
        *,
        save_id: str,
        asset: MediaAssetRecord,
    ) -> None:
        try:
            self.repositories.archive_media_asset_only(
                save_id=save_id,
                media_asset_id=asset.id,
            )
        except Exception as cleanup_exc:
            log_error_event(
                "media.cleanup_failed",
                path=asset.path,
                **exception_log_fields(cleanup_exc),
            )
        self._delete_persisted_files(asset.path, asset.thumbnail_path)

    async def generate_video_for_message(
        self,
        *,
        save_id: str,
        source_message_id: str,
        job_context: str | None = None,
        current_user_id: str | None = None,
    ) -> MediaAssetRecord:
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="video_generation",
        )
        if preference is None:
            raise ValueError("No video generation model preference configured")
        source_message = _source_message(
            messages=self.repositories.list_messages(save_id),
            source_message_id=source_message_id,
        )
        if source_message is None:
            raise ValueError(f"Unknown source message id: {source_message_id}")
        _raise_if_safety_transition_source(source_message)
        scene_context, context_breakdown = self._build_scene_context_with_breakdown(
            save_id=save_id,
            source_message_id=source_message_id,
        )
        return await self._generate_video_for_message_with_context(
            save_id=save_id,
            source_message_id=source_message_id,
            scene_context=scene_context,
            context_breakdown_json=context_breakdown.to_json(),
            preference=preference,
            job_context=job_context,
            current_user_id=current_user_id,
        )

    async def animate_image(
        self,
        *,
        save_id: str,
        media_asset_id: str,
        motion_prompt: str = "",
        job_context: str | None = None,
        current_user_id: str | None = None,
    ) -> MediaAssetRecord:
        source_asset = _media_asset_by_id(
            self.repositories.list_media_assets(save_id),
            media_asset_id,
        )
        if source_asset is None:
            raise ValueError(f"Unknown media asset id: {media_asset_id}")
        if source_asset.type != "image":
            raise ValueError("Only image media assets can be animated")
        source_message_id = source_asset.source_message_id
        if source_message_id is None:
            raise ValueError("Animated images require a source message")
        source_media_path = self.media_dir / source_asset.path
        if not source_media_path.is_file():
            raise ValueError(f"Source image is unavailable: {source_media_path}")
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="image_animation",
        )
        if preference is None:
            raise ValueError("No image animation model preference configured")
        scene_context, context_breakdown = self._build_scene_context_with_breakdown(
            save_id=save_id,
            source_message_id=source_message_id,
        )
        prompt = _animation_prompt(
            source_prompt=source_asset.prompt,
            scene_context=scene_context,
            motion_prompt=motion_prompt,
            max_chars=_animation_prompt_max_chars(preference.provider),
        )
        return await self._generate_video_asset(
            save_id=save_id,
            source_message_id=source_message_id,
            prompt=prompt,
            preference=preference,
            job_type="image_animation",
            job_context=job_context,
            context_breakdown_json=context_breakdown.to_json(),
            source_media_asset_id=source_asset.id,
            source_media_path=source_media_path,
            required_capability=ProviderCapability.IMAGE_TO_VIDEO,
            current_user_id=current_user_id,
        )

    async def _generate_for_message_with_context(
        self,
        *,
        save_id: str,
        source_message_id: str,
        scene_context: str,
        context_breakdown_json: dict[str, object],
        preference: ModelPreferenceRecord,
        job_context: str | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        source_media_asset_id: str | None = None,
        source_media_path: Path | None = None,
        source_media_asset_ids: tuple[str, ...] = (),
        source_media_paths: tuple[Path, ...] = (),
        metadata: dict[str, object] | None = None,
        job_type: str = "image_generation",
        request_task: str | None = None,
        character_visual_directions: str = "",
        current_user_id: str | None = None,
    ) -> MediaAssetRecord:
        source_message = _source_message(
            messages=self.repositories.list_messages(save_id),
            source_message_id=source_message_id,
        )
        if source_message is not None:
            _raise_if_safety_transition_source(source_message)
        scene_characters = _scene_characters(
            repositories=self.repositories,
            save_id=save_id,
            source_message_id=source_message_id,
        )
        await self._ensure_current_clothing(
            save_id=save_id,
            characters=scene_characters,
            image_context=scene_context,
        )
        character_visual_directions = _scene_character_visual_directions(
            repositories=self.repositories,
            save_id=save_id,
            source_message_id=source_message_id,
            action_context=source_message.body if source_message is not None else "",
        )
        source_media_asset_ids = _normalized_source_media_asset_ids(
            source_media_asset_id,
            source_media_asset_ids,
        )
        source_media_paths = _normalized_source_media_paths(
            source_media_path,
            source_media_paths,
        )
        primary_source_media_asset_id = (
            source_media_asset_ids[0] if source_media_asset_ids else None
        )
        primary_source_media_path = (
            source_media_paths[0] if source_media_paths else None
        )
        image_style_preset = selected_image_style_preset(
            self.repositories,
            save_id=save_id,
        )
        job = self.jobs.create_running(
            save_id=save_id,
            type=job_type,
            request_context={
                "kind": job_context or "manual_scene_image",
                "source_message_id": source_message_id,
                "provider": preference.provider,
                "model": preference.model_id,
                "task": request_task,
                "source_media_asset_ids": list(source_media_asset_ids),
                "image_style_preset": image_style_preset,
            },
            payload={
                "save_id": save_id,
                "source_message_id": source_message_id,
                "job_context": job_context,
                "provider": preference.provider,
                "model": preference.model_id,
                "source_media_asset_id": primary_source_media_asset_id,
                "source_media_asset_ids": list(source_media_asset_ids),
                "scene_context_chars": len(scene_context),
                "image_style_preset": image_style_preset,
                "context_breakdown": context_breakdown_json,
                **_venice_safe_mode_diagnostics(
                    provider=preference.provider,
                    safe_mode=_request_safe_mode(
                        repositories=self.repositories,
                        provider=preference.provider,
                        save_id=save_id,
                        current_user_id=current_user_id,
                    ),
                ),
            },
        )
        log_event(
            "job.running",
            job_id=job.id,
            job_type=job.type,
            save_id=save_id,
            source_message_id=source_message_id,
            job_context=job_context,
            provider=preference.provider,
            model=preference.model_id,
            scene_context_chars=len(scene_context),
        )
        started_at = perf_counter()
        path: str | None = None
        thumbnail_path: str | None = None
        asset_created = False
        try:
            requirement_error = _image_model_requirement_error(
                repositories=self.repositories,
                provider=preference.provider,
                model_id=preference.model_id,
                source_media_asset_id=primary_source_media_asset_id,
            )
            if requirement_error is not None:
                raise ValueError(requirement_error)
            prompt = await self._draft_image_prompt(
                save_id=save_id,
                source_message_id=source_message_id,
                scene_context=scene_context,
            )
            prompt = _prompt_with_character_visual_directions(
                prompt,
                character_visual_directions,
            )
            prompt = apply_image_style_preset(
                prompt,
                preset_id=image_style_preset,
            )
            if source_media_asset_ids:
                prompt = _image_to_image_prompt(
                    prompt,
                    reference_count=len(source_media_asset_ids),
                )
            generation = await self._generate_image_with_optional_fallback(
                save_id=save_id,
                request=await self._image_request(
                    provider=preference.provider,
                    model_id=preference.model_id,
                    prompt=prompt,
                    save_id=save_id,
                    source_message_id=source_message_id,
                    retry_progress_callback=retry_progress_callback,
                    source_media_asset_id=primary_source_media_asset_id,
                    source_media_path=primary_source_media_path,
                    source_media_asset_ids=source_media_asset_ids,
                    source_media_paths=source_media_paths,
                    task=request_task,
                    current_user_id=current_user_id,
                ),
            )
            generation, path, thumbnail_path = (
                await self._persist_generated_image_with_optional_fallback(
                    save_id=save_id,
                    source_message_id=source_message_id,
                    generation_id=job.id,
                    generation=generation,
                )
            )
            response = generation.response
            asset = self.repositories.create_media_asset(
                save_id=save_id,
                source_message_id=source_message_id,
                type="image",
                path=path,
                thumbnail_path=thumbnail_path,
                prompt=prompt,
                provider=response.provider,
                model=_persisted_image_model(generation),
                status="succeeded",
                metadata=_image_asset_metadata(metadata, generation=generation),
                source_media_asset_id=primary_source_media_asset_id,
            )
            asset_created = True
            self.jobs.succeed(
                job.id,
                result={
                    "media_asset_id": asset.id,
                    "path": asset.path,
                    "prompt_chars": len(prompt),
                    "image_style_preset": image_style_preset,
                    "context_breakdown": context_breakdown_json,
                    "provider": asset.provider,
                    "model": asset.model,
                    "source_media_asset_id": primary_source_media_asset_id,
                    "source_media_asset_ids": list(source_media_asset_ids),
                    **generation.diagnostics,
                },
            )
        except Exception as exc:
            if not asset_created:
                self._delete_persisted_files(path, thumbnail_path)
            self.jobs.fail(
                job.id,
                error=redact_text(str(exc)) or exc.__class__.__name__,
                result=_failed_image_result(
                    exc=exc,
                ),
            )
            log_error_event(
                "job.failed",
                job_id=job.id,
                job_type=job.type,
                save_id=save_id,
                source_message_id=source_message_id,
                job_context=job_context,
                provider=preference.provider,
                model=preference.model_id,
                duration_ms=_elapsed_ms(started_at),
                **exception_log_fields(exc),
            )
            raise
        output_path = self.media_dir / asset.path
        log_event(
            "media.image_succeeded",
            job_id=job.id,
            save_id=save_id,
            source_message_id=source_message_id,
            job_context=job_context,
            media_asset_id=asset.id,
            provider=asset.provider,
            model=asset.model,
            output_path=asset.path,
            byte_count=output_path.stat().st_size if output_path.is_file() else None,
            duration_ms=_elapsed_ms(started_at),
        )
        log_event(
            "job.succeeded",
            job_id=job.id,
            job_type=job.type,
            save_id=save_id,
            job_context=job_context,
        )
        return asset

    async def _ensure_current_clothing(
        self,
        *,
        save_id: str,
        characters: tuple[CharacterRecord, ...],
        image_context: str,
    ) -> tuple[CharacterRecord, ...]:
        missing = tuple(
            character
            for character in characters
            if not character.current_clothing.strip()
            and not character_field_is_locked(
                character.locked_fields,
                "current_clothing",
            )
        )
        if not missing:
            return characters
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="response_planning",
        )
        if preference is None or preference.provider not in self.providers:
            log_event(
                "media.current_clothing_completion_skipped",
                save_id=save_id,
                reason="model_unavailable",
                character_ids=[character.id for character in missing],
            )
            return characters

        messages = list(
            _current_clothing_completion_messages(
                characters=missing,
                image_context=image_context,
            )
        )
        request = StructuredOutputRequest(
            provider=preference.provider,
            model_id=preference.model_id,
            schema_name="image_current_clothing_completion",
            schema=_current_clothing_completion_schema(),
            messages=tuple(messages),
            temperature=0.2,
            max_output_tokens=10_000,
        )
        expected_ids = {character.id for character in missing}
        last_error = "unknown validation failure"
        max_attempt_count = configured_max_attempts(self.repositories)
        for attempt in range(1, max_attempt_count + 1):
            try:
                response = await structured_output_with_fallback(
                    repositories=self.repositories,
                    providers=self.providers,
                    request=replace(request, messages=tuple(messages)),
                    task="response_planning",
                    save_id=save_id,
                    diagnostic_context={
                        "character_ids": sorted(expected_ids),
                        "attempt": attempt,
                    },
                )
                completed = _current_clothing_from_data(
                    response.data,
                    expected_character_ids=expected_ids,
                    appearance_by_character_id={
                        character.id: character.appearance for character in missing
                    },
                )
            except Exception as exc:  # noqa: BLE001 - clothing is auxiliary
                last_error = str(exc) or exc.__class__.__name__
                log_event(
                    "media.current_clothing_completion_attempt_failed",
                    save_id=save_id,
                    attempt=attempt,
                    max_attempts=max_attempt_count,
                    character_ids=sorted(expected_ids),
                    **exception_log_fields(exc),
                )
                if attempt < max_attempt_count:
                    messages.append(
                        ChatMessage(
                            role="user",
                            body=(
                                "Previous structured response was invalid: "
                                f"{last_error}. Return one nonblank current_clothing "
                                "value for every requested character ID and no "
                                "other IDs."
                            ),
                        )
                    )
                continue

            updated_by_id: dict[str, CharacterRecord] = {}
            for character in missing:
                try:
                    current = (
                        self.repositories
                        .set_character_current_clothing_if_blank_and_unlocked(
                            save_id=save_id,
                            character_id=character.id,
                            current_clothing=completed[character.id],
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - clothing is auxiliary
                    log_error_event(
                        "media.current_clothing_persistence_failed",
                        save_id=save_id,
                        character_id=character.id,
                        **exception_log_fields(exc),
                    )
                    current = self.repositories.get_character(character.id)
                if current is not None:
                    updated_by_id[character.id] = current
            log_event(
                "media.current_clothing_completed",
                save_id=save_id,
                attempt=attempt,
                max_attempts=max_attempt_count,
                character_ids=sorted(expected_ids),
                provider=response.provider,
                model=response.model_id,
            )
            return tuple(
                updated_by_id.get(character.id, character) for character in characters
            )

        log_error_event(
            "media.current_clothing_completion_exhausted",
            save_id=save_id,
            max_attempts=max_attempt_count,
            character_ids=sorted(expected_ids),
            error=redact_text(last_error),
        )
        return characters

    async def _generate_prompted_image_asset(
        self,
        *,
        save_id: str,
        source_message_id: str | None,
        request_source_message_id: str,
        prompt: str,
        scene_context: str,
        context_breakdown_json: dict[str, object],
        preference: ModelPreferenceRecord,
        job_context: str | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        metadata: dict[str, object] | None = None,
        job_type: str = "image_generation",
        request_task: str | None = None,
        current_user_id: str | None = None,
    ) -> MediaAssetRecord:
        image_style_preset = selected_image_style_preset(
            self.repositories,
            save_id=save_id,
        )
        prompt = apply_image_style_preset(prompt, preset_id=image_style_preset)
        metadata_context = metadata or {}
        context_source_media_asset_id = metadata_context.get("source_media_asset_id")
        context_source_media_asset_ids = (
            [context_source_media_asset_id]
            if isinstance(context_source_media_asset_id, str)
            else []
        )
        job = self.jobs.create_running(
            save_id=save_id,
            type=job_type,
            request_context={
                **metadata_context,
                "kind": job_context
                or (
                    "character_text_attachment"
                    if job_type.startswith("character_text")
                    else "manual_scene_image"
                ),
                "source_message_id": request_source_message_id,
                "provider": preference.provider,
                "model": preference.model_id,
                "task": request_task,
                "prompt": prompt,
                "prompt_chars": len(prompt),
                "source_media_asset_ids": context_source_media_asset_ids,
                "image_style_preset": image_style_preset,
            },
            payload={
                "save_id": save_id,
                "source_message_id": source_message_id,
                "request_source_message_id": request_source_message_id,
                "job_context": job_context,
                "provider": preference.provider,
                "model": preference.model_id,
                "scene_context_chars": len(scene_context),
                "image_style_preset": image_style_preset,
                "context_breakdown": context_breakdown_json,
                **_venice_safe_mode_diagnostics(
                    provider=preference.provider,
                    safe_mode=_request_safe_mode(
                        repositories=self.repositories,
                        provider=preference.provider,
                        save_id=save_id,
                        current_user_id=current_user_id,
                    ),
                ),
            },
        )
        started_at = perf_counter()
        path: str | None = None
        thumbnail_path: str | None = None
        asset_created = False
        try:
            requirement_error = _image_model_requirement_error(
                repositories=self.repositories,
                provider=preference.provider,
                model_id=preference.model_id,
                source_media_asset_id=None,
            )
            if requirement_error is not None:
                raise ValueError(requirement_error)
            generation = await self._generate_image_with_optional_fallback(
                save_id=save_id,
                request=await self._image_request(
                    provider=preference.provider,
                    model_id=preference.model_id,
                    prompt=prompt,
                    save_id=save_id,
                    source_message_id=request_source_message_id,
                    retry_progress_callback=retry_progress_callback,
                    task=request_task,
                    current_user_id=current_user_id,
                ),
            )
            generation, path, thumbnail_path = (
                await self._persist_generated_image_with_optional_fallback(
                    save_id=save_id,
                    source_message_id=request_source_message_id,
                    generation_id=job.id,
                    generation=generation,
                )
            )
            response = generation.response
            asset = self.repositories.create_media_asset(
                save_id=save_id,
                source_message_id=source_message_id,
                type="image",
                path=path,
                thumbnail_path=thumbnail_path,
                prompt=prompt,
                provider=response.provider,
                model=_persisted_image_model(generation),
                status="succeeded",
                metadata=_image_asset_metadata(metadata, generation=generation),
            )
            asset_created = True
            self.jobs.succeed(
                job.id,
                result={
                    "media_asset_id": asset.id,
                    "path": asset.path,
                    "prompt_chars": len(prompt),
                    "image_style_preset": image_style_preset,
                    "context_breakdown": context_breakdown_json,
                    "provider": asset.provider,
                    "model": asset.model,
                    **generation.diagnostics,
                },
            )
        except Exception as exc:
            if not asset_created:
                self._delete_persisted_files(path, thumbnail_path)
            self.jobs.fail(
                job.id,
                error=redact_text(str(exc)) or exc.__class__.__name__,
                result=_failed_image_result(exc=exc),
            )
            log_error_event(
                "job.failed",
                job_id=job.id,
                job_type=job.type,
                save_id=save_id,
                source_message_id=source_message_id,
                job_context=job_context,
                provider=preference.provider,
                model=preference.model_id,
                duration_ms=_elapsed_ms(started_at),
                **exception_log_fields(exc),
            )
            raise
        output_path = self.media_dir / asset.path
        log_event(
            "media.image_succeeded",
            job_id=job.id,
            save_id=save_id,
            source_message_id=source_message_id,
            job_context=job_context,
            media_asset_id=asset.id,
            provider=asset.provider,
            model=asset.model,
            output_path=asset.path,
            byte_count=output_path.stat().st_size if output_path.is_file() else None,
            duration_ms=_elapsed_ms(started_at),
        )
        log_event(
            "job.succeeded",
            job_id=job.id,
            job_type=job.type,
            save_id=save_id,
            job_context=job_context,
        )
        return asset

    async def _generate_character_image_asset(
        self,
        *,
        save_id: str,
        source_message_id: str | None,
        request_source_message_id: str,
        prompt: str,
        scene_context: str,
        context_breakdown_json: dict[str, object],
        preference: ModelPreferenceRecord,
        job_context: str | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        source_media_asset_id: str | None = None,
        source_media_path: Path | None = None,
        source_media_asset_ids: tuple[str, ...] = (),
        source_media_paths: tuple[Path, ...] = (),
        metadata: dict[str, object] | None = None,
        job_type: str = "character_image_generation",
        request_task: str | None = None,
        current_user_id: str | None = None,
    ) -> MediaAssetRecord:
        source_media_asset_ids = _normalized_source_media_asset_ids(
            source_media_asset_id,
            source_media_asset_ids,
        )
        source_media_paths = _normalized_source_media_paths(
            source_media_path,
            source_media_paths,
        )
        primary_source_media_asset_id = (
            source_media_asset_ids[0] if source_media_asset_ids else None
        )
        primary_source_media_path = (
            source_media_paths[0] if source_media_paths else None
        )
        image_style_preset = selected_image_style_preset(
            self.repositories,
            save_id=save_id,
        )
        prompt = apply_image_style_preset(prompt, preset_id=image_style_preset)
        prompt = _image_to_image_prompt(
            prompt,
            reference_count=len(source_media_asset_ids),
        )
        job = self.jobs.create_running(
            save_id=save_id,
            type=job_type,
            request_context={
                **(metadata or {}),
                "kind": job_context
                or (
                    "character_text_attachment"
                    if job_type.startswith("character_text")
                    else "manual_scene_image"
                ),
                "source_message_id": request_source_message_id,
                "provider": preference.provider,
                "model": preference.model_id,
                "task": request_task,
                "prompt": prompt,
                "prompt_chars": len(prompt),
                "character_id": (metadata or {}).get("character_id"),
                "source_media_asset_ids": list(source_media_asset_ids),
                "image_style_preset": image_style_preset,
            },
            payload={
                "save_id": save_id,
                "source_message_id": source_message_id,
                "job_context": job_context,
                "provider": preference.provider,
                "model": preference.model_id,
                "source_media_asset_id": primary_source_media_asset_id,
                "source_media_asset_ids": list(source_media_asset_ids),
                "scene_context_chars": len(scene_context),
                "image_style_preset": image_style_preset,
                "context_breakdown": context_breakdown_json,
                **_venice_safe_mode_diagnostics(
                    provider=preference.provider,
                    safe_mode=_request_safe_mode(
                        repositories=self.repositories,
                        provider=preference.provider,
                        save_id=save_id,
                        current_user_id=current_user_id,
                    ),
                ),
            },
        )
        started_at = perf_counter()
        path: str | None = None
        thumbnail_path: str | None = None
        asset_created = False
        try:
            requirement_error = _image_model_requirement_error(
                repositories=self.repositories,
                provider=preference.provider,
                model_id=preference.model_id,
                source_media_asset_id=primary_source_media_asset_id,
            )
            if requirement_error is not None:
                raise ValueError(requirement_error)
            generation = await self._generate_image_with_optional_fallback(
                save_id=save_id,
                request=await self._image_request(
                    provider=preference.provider,
                    model_id=preference.model_id,
                    prompt=prompt,
                    save_id=save_id,
                    source_message_id=request_source_message_id,
                    retry_progress_callback=retry_progress_callback,
                    source_media_asset_id=primary_source_media_asset_id,
                    source_media_path=primary_source_media_path,
                    source_media_asset_ids=source_media_asset_ids,
                    source_media_paths=source_media_paths,
                    task=request_task,
                    current_user_id=current_user_id,
                ),
            )
            generation, path, thumbnail_path = (
                await self._persist_generated_image_with_optional_fallback(
                    save_id=save_id,
                    source_message_id=request_source_message_id,
                    generation_id=job.id,
                    generation=generation,
                )
            )
            response = generation.response
            asset = self.repositories.create_media_asset(
                save_id=save_id,
                source_message_id=source_message_id,
                type="image",
                path=path,
                thumbnail_path=thumbnail_path,
                prompt=prompt,
                provider=response.provider,
                model=_persisted_image_model(generation),
                status="succeeded",
                metadata=_image_asset_metadata(metadata, generation=generation),
                source_media_asset_id=primary_source_media_asset_id,
            )
            asset_created = True
            self.jobs.succeed(
                job.id,
                result={
                    "media_asset_id": asset.id,
                    "path": asset.path,
                    "prompt_chars": len(prompt),
                    "image_style_preset": image_style_preset,
                    "context_breakdown": context_breakdown_json,
                    "provider": asset.provider,
                    "model": asset.model,
                    "source_media_asset_id": primary_source_media_asset_id,
                    "source_media_asset_ids": list(source_media_asset_ids),
                    **generation.diagnostics,
                },
            )
        except Exception as exc:
            if not asset_created:
                self._delete_persisted_files(path, thumbnail_path)
            self.jobs.fail(
                job.id,
                error=redact_text(str(exc)) or exc.__class__.__name__,
                result=_failed_image_result(exc=exc),
            )
            log_error_event(
                "job.failed",
                job_id=job.id,
                job_type=job.type,
                save_id=save_id,
                source_message_id=source_message_id,
                job_context=job_context,
                provider=preference.provider,
                model=preference.model_id,
                duration_ms=_elapsed_ms(started_at),
                **exception_log_fields(exc),
            )
            raise
        output_path = self.media_dir / asset.path
        log_event(
            "media.image_succeeded",
            job_id=job.id,
            save_id=save_id,
            source_message_id=source_message_id,
            job_context=job_context,
            media_asset_id=asset.id,
            provider=asset.provider,
            model=asset.model,
            output_path=asset.path,
            byte_count=output_path.stat().st_size if output_path.is_file() else None,
            duration_ms=_elapsed_ms(started_at),
        )
        log_event(
            "job.succeeded",
            job_id=job.id,
            job_type=job.type,
            save_id=save_id,
            job_context=job_context,
        )
        return asset

    async def _generate_video_for_message_with_context(
        self,
        *,
        save_id: str,
        source_message_id: str,
        scene_context: str,
        context_breakdown_json: dict[str, object],
        preference: ModelPreferenceRecord,
        job_context: str | None = None,
        current_user_id: str | None = None,
    ) -> MediaAssetRecord:
        source_message = _source_message(
            messages=self.repositories.list_messages(save_id),
            source_message_id=source_message_id,
        )
        if source_message is not None:
            _raise_if_safety_transition_source(source_message)
        return await self._generate_video_asset(
            save_id=save_id,
            source_message_id=source_message_id,
            prompt_factory=lambda: self._draft_image_prompt(
                save_id=save_id,
                source_message_id=source_message_id,
                scene_context=scene_context,
            ),
            preference=preference,
            job_type="video_generation",
            job_context=job_context,
            context_breakdown_json=context_breakdown_json,
            source_media_asset_id=None,
            source_media_path=None,
            required_capability=ProviderCapability.TEXT_TO_VIDEO,
            current_user_id=current_user_id,
        )

    async def _generate_video_asset(
        self,
        *,
        save_id: str,
        source_message_id: str,
        prompt: str | None = None,
        prompt_factory: Callable[[], Awaitable[str]] | None = None,
        preference: ModelPreferenceRecord,
        job_type: str,
        job_context: str | None,
        context_breakdown_json: dict[str, object],
        source_media_asset_id: str | None,
        source_media_path: Path | None,
        required_capability: ProviderCapability,
        current_user_id: str | None = None,
    ) -> MediaAssetRecord:
        job = self.jobs.create_running(
            save_id=save_id,
            type=job_type,
            payload={
                "save_id": save_id,
                "source_message_id": source_message_id,
                "source_media_asset_id": source_media_asset_id,
                "job_context": job_context,
                "provider": preference.provider,
                "model": preference.model_id,
                "prompt_chars": len(prompt) if prompt is not None else 0,
                "context_breakdown": context_breakdown_json,
                **_venice_safe_mode_diagnostics(
                    provider=preference.provider,
                    safe_mode=_request_video_safe_mode(
                        repositories=self.repositories,
                        provider=preference.provider,
                        model_id=preference.model_id,
                        save_id=save_id,
                        current_user_id=current_user_id,
                    ),
                ),
            },
        )
        started_at = perf_counter()
        path: str | None = None
        asset_created = False
        try:
            requirement_error = _video_model_requirement_error(
                repositories=self.repositories,
                preference=preference,
                required_capability=required_capability,
            )
            if requirement_error is not None:
                raise ValueError(requirement_error)
            if prompt is None:
                if prompt_factory is None:
                    raise AssertionError("video prompt was not configured")
                prompt = await prompt_factory()
            generation = await self._generate_video_with_optional_fallback(
                save_id=save_id,
                request=await self._video_request(
                    provider=preference.provider,
                    model_id=preference.model_id,
                    prompt=prompt,
                    save_id=save_id,
                    source_message_id=source_message_id,
                    source_media_asset_id=source_media_asset_id,
                    source_media_path=source_media_path,
                    source_content_rating=_media_generation_source_rating(
                        repositories=self.repositories,
                        save_id=save_id,
                        source_message_id=source_message_id,
                        source_media_asset_id=source_media_asset_id,
                    ),
                    current_user_id=current_user_id,
                ),
                required_capability=required_capability,
            )
            response = generation.response
            path = self._persist_video(
                response=response,
                save_id=save_id,
                source_message_id=source_message_id,
                generation_id=job.id,
            )
            asset = self.repositories.create_media_asset(
                save_id=save_id,
                source_message_id=source_message_id,
                type="video",
                path=path,
                thumbnail_path=None,
                prompt=prompt,
                provider=response.provider,
                model=response.model_id,
                status="succeeded",
                mime_type=response.mime_type,
                metadata={
                    **response.raw_metadata,
                    "content_rating": generation.request.content_rating,
                },
                source_media_asset_id=source_media_asset_id,
            )
            asset_created = True
            self.jobs.succeed(
                job.id,
                result={
                    "media_asset_id": asset.id,
                    "path": asset.path,
                    "mime_type": asset.mime_type,
                    "prompt_chars": len(prompt),
                    "context_breakdown": context_breakdown_json,
                    "provider": asset.provider,
                    "model": asset.model,
                    **generation.diagnostics,
                },
            )
        except Exception as exc:
            if not asset_created:
                self._delete_persisted_files(path, None)
            self.jobs.fail(
                job.id,
                error=redact_text(str(exc)) or exc.__class__.__name__,
                result=_failed_image_result(exc=exc),
            )
            log_error_event(
                "job.failed",
                job_id=job.id,
                job_type=job.type,
                save_id=save_id,
                source_message_id=source_message_id,
                job_context=job_context,
                provider=preference.provider,
                model=preference.model_id,
                duration_ms=_elapsed_ms(started_at),
                **exception_log_fields(exc),
            )
            raise
        log_event(
            "media.video_succeeded",
            job_id=job.id,
            save_id=save_id,
            source_message_id=source_message_id,
            media_asset_id=asset.id,
            provider=asset.provider,
            model=asset.model,
            output_path=asset.path,
            duration_ms=_elapsed_ms(started_at),
        )
        return asset

    def prepare_automatic_if_due(
        self,
        *,
        save_id: str,
        source_message_id: str | None = None,
    ) -> PreparedAutomaticImage | None:
        media_type = _automatic_media_mode(self.repositories)
        source_message = self._automatic_source_if_due(
            save_id=save_id,
            source_message_id=source_message_id,
            media_type=media_type,
        )
        if source_message is None:
            return None
        if media_type == "image" and _is_text_message_beat(source_message[0].body):
            log_event(
                "media.automatic_skipped",
                save_id=save_id,
                reason="text_message_beat",
                source_message_id=source_message[0].id,
                narrator_message_count=source_message[1],
            )
            return None
        source_media_asset_id: str | None = None
        source_media_path: Path | None = None
        source_media_asset_ids: tuple[str, ...] = ()
        source_media_paths: tuple[Path, ...] = ()
        metadata: dict[str, object] | None = None
        request_task: str | None = None
        character_visual_directions = ""
        if media_type == "video":
            preference = roleplay_model_preference(
                repositories=self.repositories,
                save_id=save_id,
                purpose="video_generation",
            )
            if preference is None:
                raise ValueError("No video generation model preference configured")
        else:
            preference = roleplay_model_preference(
                repositories=self.repositories,
                save_id=save_id,
                purpose="image_generation",
            )
            if preference is None:
                raise ValueError("No image generation model preference configured")
            request_context = self._scene_image_request_context(
                save_id=save_id,
                source_message_id=source_message[0].id,
                fallback_preference=preference,
            )
            preference = request_context.preference
            source_media_asset_id = request_context.source_media_asset_id
            source_media_path = request_context.source_media_path
            source_media_asset_ids = request_context.source_media_asset_ids
            source_media_paths = request_context.source_media_paths
            metadata = request_context.metadata
            request_task = request_context.request_task
            character_visual_directions = request_context.character_visual_directions
        scene_context, context_breakdown = self._build_scene_context_with_breakdown(
            save_id=save_id,
            source_message_id=source_message[0].id,
        )
        return PreparedAutomaticImage(
            save_id=save_id,
            source_message_id=source_message[0].id,
            scene_context=scene_context,
            context_breakdown_json=context_breakdown.to_json(),
            provider=preference.provider,
            model_id=preference.model_id,
            narrator_message_count=source_message[1],
            media_type=media_type,
            source_media_asset_id=source_media_asset_id,
            source_media_path=source_media_path,
            source_media_asset_ids=source_media_asset_ids,
            source_media_paths=source_media_paths,
            metadata=metadata,
            request_task=request_task if media_type != "video" else None,
            character_visual_directions=(
                character_visual_directions if media_type != "video" else ""
            ),
        )

    async def generate_prepared_automatic(
        self,
        prepared: PreparedAutomaticImage,
        *,
        current_user_id: str | None = None,
    ) -> MediaAssetRecord | None:
        existing_source_ids = {
            asset.source_message_id
            for asset in self.repositories.list_media_assets(prepared.save_id)
            if asset.type == prepared.media_type
        }
        if prepared.source_message_id in existing_source_ids:
            log_event(
                "media.automatic_skipped",
                save_id=prepared.save_id,
                reason="already_generated",
                source_message_id=prepared.source_message_id,
            )
            return None
        source_message = _source_message(
            messages=self.repositories.list_messages(prepared.save_id),
            source_message_id=prepared.source_message_id,
        )
        if source_message is None or _is_safety_transition_source(source_message):
            log_event(
                "media.automatic_skipped",
                save_id=prepared.save_id,
                reason="safety_transition",
                source_message_id=prepared.source_message_id,
            )
            return None
        preference = ModelPreferenceRecord(
            id=f"prepared-automatic-{prepared.media_type}",
            task=(
                "video_generation"
                if prepared.media_type == "video"
                else prepared.request_task or "image_generation"
            ),
            provider=prepared.provider,
            model_id=prepared.model_id,
        )
        if prepared.media_type == "video":
            return await self._generate_video_for_message_with_context(
                save_id=prepared.save_id,
                source_message_id=prepared.source_message_id,
                scene_context=prepared.scene_context,
                context_breakdown_json=prepared.context_breakdown_json,
                preference=preference,
                job_context="automatic_post_turn",
                current_user_id=current_user_id,
            )
        return await self._generate_for_message_with_context(
            save_id=prepared.save_id,
            source_message_id=prepared.source_message_id,
            scene_context=prepared.scene_context,
            context_breakdown_json=prepared.context_breakdown_json,
            preference=preference,
            job_context="automatic_post_turn",
            source_media_asset_id=prepared.source_media_asset_id,
            source_media_path=prepared.source_media_path,
            source_media_asset_ids=prepared.source_media_asset_ids,
            source_media_paths=prepared.source_media_paths,
            metadata=prepared.metadata,
            request_task=prepared.request_task,
            character_visual_directions=prepared.character_visual_directions,
            current_user_id=current_user_id,
        )

    def _character_image_preference(
        self,
        *,
        save_id: str,
    ) -> ModelPreferenceRecord:
        return self._image_edit_preference(
            save_id=save_id,
            purpose=CHARACTER_IMAGE_EDIT_PURPOSE,
        )

    def _image_edit_preference(
        self,
        *,
        save_id: str,
        purpose: str,
    ) -> ModelPreferenceRecord:
        preference = (
            roleplay_model_preference(
                repositories=self.repositories,
                save_id=save_id,
                purpose=IMAGE_TO_IMAGE_GENERATION_PURPOSE,
            )
            if purpose == IMAGE_TO_IMAGE_GENERATION_PURPOSE
            else image_edit_model_preference(
                repositories=self.repositories,
                save_id=save_id,
                purpose=purpose,
            )
        )
        if preference is None:
            raise ValueError("No image-to-image generation model preference configured")
        if preference.provider not in self.providers:
            raise ValueError(
                "Image-to-image generation provider is unavailable: "
                f"{preference.provider}"
            )
        requirement_error = _known_media_model_requirement_error(
            repositories=self.repositories,
            preference=preference,
            required=IMAGE_TO_IMAGE_CAPABILITIES,
            unavailable_label="Image-to-image generation",
            missing_capability_message=(
                "Image-to-image generation model does not advertise "
                "image-to-image support"
            ),
        )
        if requirement_error is not None:
            raise ValueError(requirement_error)
        return preference

    def _replacement_source_media_paths(
        self,
        *,
        save_id: str,
        source_media_asset_ids: tuple[str, ...],
    ) -> tuple[Path, ...]:
        if not source_media_asset_ids:
            return ()
        active_assets = {
            asset.id: asset for asset in self.repositories.list_media_assets(save_id)
        }
        paths: list[Path] = []
        for source_media_asset_id in source_media_asset_ids:
            asset = active_assets.get(source_media_asset_id)
            if asset is None:
                raise ValueError(
                    f"Source reference image is unavailable: {source_media_asset_id}"
                )
            source_path = self.media_dir / asset.path
            _assert_within_media_dir(
                media_dir=self.media_dir,
                output_path=source_path,
            )
            if not source_path.is_file():
                raise ValueError(
                    f"Source reference image is unavailable: {source_media_asset_id}"
                )
            paths.append(source_path)
        return tuple(paths)

    def _scene_image_request_context(
        self,
        *,
        save_id: str,
        source_message_id: str,
        fallback_preference: ModelPreferenceRecord,
    ) -> _ImageRequestContext:
        source_message = _source_message(
            messages=self.repositories.list_messages(save_id),
            source_message_id=source_message_id,
        )
        action_context = source_message.body if source_message is not None else ""
        character_visual_directions = _scene_character_visual_directions(
            repositories=self.repositories,
            save_id=save_id,
            source_message_id=source_message_id,
            action_context=action_context,
        )
        references = _selected_scene_character_references(
            repositories=self.repositories,
            media_dir=self.media_dir,
            save_id=save_id,
            source_message_id=source_message_id,
        )
        if not references:
            return _ImageRequestContext(
                preference=fallback_preference,
                metadata={"kind": "scene_image"},
                request_task="image_generation",
                character_visual_directions=character_visual_directions,
            )

        preference = self._image_edit_preference(
            save_id=save_id,
            purpose=SCENE_IMAGE_EDIT_PURPOSE,
        )
        references = references[
            : _image_reference_limit(
                provider=self.providers[preference.provider],
                model_id=preference.model_id,
            )
        ]
        source_media_asset_ids = tuple(
            reference.media_asset_id for reference in references
        )
        source_media_paths = tuple(reference.media_path for reference in references)
        return _ImageRequestContext(
            preference=preference,
            source_media_asset_id=source_media_asset_ids[0],
            source_media_path=source_media_paths[0],
            source_media_asset_ids=source_media_asset_ids,
            source_media_paths=source_media_paths,
            metadata={
                "kind": "scene_image",
                "source_character_reference_asset_id": source_media_asset_ids[0],
                "source_character_reference_asset_ids": list(source_media_asset_ids),
                "source_character_reference_character_ids": [
                    reference.character_id for reference in references
                ],
                "source_character_reference_character_names": [
                    reference.character_name for reference in references
                ],
            },
            request_task=SCENE_IMAGE_EDIT_PURPOSE,
            character_visual_directions=character_visual_directions,
        )

    def _character_image_request_context(
        self,
        *,
        preference: ModelPreferenceRecord,
        reference: MediaAssetRecord,
        character_id: str,
        character_name: str,
        origin: str,
    ) -> _ImageRequestContext:
        source_path = self.media_dir / reference.path
        if not source_path.is_file():
            raise ValueError("Character reference image file is unavailable")
        return _ImageRequestContext(
            preference=preference,
            source_media_asset_id=reference.id,
            source_media_path=source_path,
            source_media_asset_ids=(reference.id,),
            source_media_paths=(source_path,),
            metadata={
                "kind": "character_image",
                "character_id": character_id,
                "character_name": character_name,
                "origin": origin,
                "source_character_reference_asset_id": reference.id,
                "source_character_reference_asset_ids": [reference.id],
            },
            request_task=CHARACTER_IMAGE_EDIT_PURPOSE,
        )

    async def _image_request(
        self,
        *,
        provider: str,
        model_id: str,
        prompt: str,
        save_id: str,
        source_message_id: str,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        source_media_asset_id: str | None = None,
        source_media_path: Path | None = None,
        source_media_asset_ids: tuple[str, ...] = (),
        source_media_paths: tuple[Path, ...] = (),
        task: str | None = None,
        route_openrouter: bool = True,
        current_user_id: str | None = None,
        reviewed_content_rating: str | None = None,
    ) -> ImageRequest:
        content_safety = effective_content_safety_policy(
            self.repositories,
            user_id=current_user_id,
        )
        minimum_rating = reviewed_content_rating
        if minimum_rating is None:
            safety = await self.content_safety_service.review_media_prompt(
                prompt=prompt,
                content_rating=content_safety.rating,
                save_id=save_id,
                source_provider=provider,
                source_model_id=model_id,
            )
            if safety.action is not ContentSafetyAction.ALLOW:
                raise ValueError("Image prompt exceeds the selected content rating")
            minimum_rating = safety.minimum_rating
        _raise_unless_enforced_safe_mode_provider(
            provider=provider,
            force_safe_mode=content_safety.force_venice_safe_mode,
        )
        resolved_task = task or (
            IMAGE_TO_IMAGE_GENERATION_PURPOSE
            if source_media_asset_id or source_media_asset_ids
            else "image_generation"
        )
        openrouter_provider_routing = None
        if provider == OPENROUTER_PROVIDER_NAME and not route_openrouter:
            openrouter_provider_routing = dict(
                openrouter_routing_payload_for_task(
                    self.repositories,
                    provider=provider,
                    task=resolved_task,
                )
                or {}
            )
            openrouter_provider_routing["allow_fallbacks"] = False
        request = ImageRequest(
            provider=provider,
            model_id=model_id,
            prompt=prompt,
            source_save_id=save_id,
            source_message_id=source_message_id,
            source_media_asset_id=source_media_asset_id,
            source_media_path=source_media_path,
            source_media_asset_ids=source_media_asset_ids,
            source_media_paths=source_media_paths,
            content_rating=minimum_rating,
            dimensions=image_generation_dimensions(
                self.repositories,
                provider=provider,
                model_id=model_id,
                save_id=save_id,
            ),
            safe_mode=_request_safe_mode(
                repositories=self.repositories,
                provider=provider,
                save_id=save_id,
                current_user_id=current_user_id,
            ),
            force_safe_mode=content_safety.force_venice_safe_mode,
            openrouter_app_title=(
                openrouter_app_title_for_task(resolved_task)
                if provider == OPENROUTER_PROVIDER_NAME
                else None
            ),
            openrouter_provider_routing=openrouter_provider_routing,
            retry_progress_callback=retry_progress_callback,
        )
        if not route_openrouter:
            return request
        return request_with_openrouter_routing(
            self.repositories,
            request,
            task=resolved_task,
            save_id=save_id,
        )

    async def _generate_image_with_optional_fallback(
        self,
        *,
        save_id: str,
        request: ImageRequest,
    ) -> _ImageGenerationResult:
        requirement_error = _image_model_requirement_error(
            repositories=self.repositories,
            provider=request.provider,
            model_id=request.model_id,
            source_media_asset_id=_primary_request_source_media_asset_id(request),
        )
        if requirement_error is not None:
            raise ValueError(requirement_error)
        primary_provider = self.providers[request.provider]
        diagnostics: dict[str, object] = {
            "original_provider": request.provider,
            "original_model": request.model_id,
            "fallback_used": False,
            **_venice_safe_mode_diagnostics(
                provider=request.provider,
                safe_mode=request.safe_mode,
                prefix="primary_",
            ),
        }
        try:
            response = await primary_provider.generate_image(request)
        except ProviderError as exc:
            diagnostics["classification"] = (
                "suspected_blocked_image_output"
                if _is_suspected_blocked_image_error(exc)
                else "primary_image_generation_failed"
            )
            diagnostics.update(_primary_error_diagnostics(exc))
            fallback = self._fallback_image_request(save_id=save_id, request=request)
            if fallback is None:
                diagnostics["fallback_skipped_reason"] = _image_fallback_skip_reason(
                    repositories=self.repositories,
                    providers=self.providers,
                    save_id=save_id,
                    required_capability=(
                        ProviderCapability.IMAGE_TO_IMAGE
                        if _image_request_has_source_media(request)
                        else ProviderCapability.IMAGE_GENERATION
                    ),
                )
                raise _ImageGenerationFailure(diagnostics, exc) from exc
            return await self._generate_fallback_image(
                fallback=fallback,
                diagnostics=diagnostics,
            )

        if not _is_suspected_blocked_image_response(response):
            return _ImageGenerationResult(
                response=response,
                request=request,
                diagnostics={
                    **diagnostics,
                    **_response_diagnostics(response.raw_metadata),
                    "final_provider": response.provider,
                    "final_model": response.model_id,
                },
            )

        diagnostics["classification"] = "suspected_blocked_image_output"
        diagnostics.update(_primary_response_diagnostics(response.raw_metadata))
        fallback = self._fallback_image_request(save_id=save_id, request=request)
        if fallback is None:
            diagnostics["fallback_skipped_reason"] = _image_fallback_skip_reason(
                repositories=self.repositories,
                providers=self.providers,
                save_id=save_id,
                required_capability=(
                    ProviderCapability.IMAGE_TO_IMAGE
                    if _image_request_has_source_media(request)
                    else ProviderCapability.IMAGE_GENERATION
                ),
            )
            log_error_event(
                "provider.image_fallback_skipped",
                provider=request.provider,
                model=request.model_id,
                task="image_generation",
                reason=diagnostics["fallback_skipped_reason"],
            )
            if _response_has_image_data(response):
                return _ImageGenerationResult(
                    response=response,
                    request=request,
                    diagnostics={
                        **diagnostics,
                        **_response_diagnostics(response.raw_metadata),
                        "final_provider": response.provider,
                        "final_model": response.model_id,
                    },
                )
            raise _ImageGenerationFailure(
                {
                    **diagnostics,
                    "final_provider": response.provider,
                    "final_model": response.model_id,
                    "final_error_category": (
                        ProviderErrorCategory.IMAGE_GENERATION_FAILED.value
                    ),
                },
                ValueError("Image provider returned no image data"),
            )
        return await self._generate_fallback_image(
            fallback=fallback,
            diagnostics=diagnostics,
        )

    async def _generate_image_without_fallback(
        self,
        *,
        save_id: str,
        request: ImageRequest,
    ) -> _ImageGenerationResult:
        requirement_error = _image_model_requirement_error(
            repositories=self.repositories,
            provider=request.provider,
            model_id=request.model_id,
            source_media_asset_id=_primary_request_source_media_asset_id(request),
        )
        if requirement_error is not None:
            raise ValueError(requirement_error)
        primary_provider = self.providers[request.provider]
        diagnostics: dict[str, object] = {
            "original_provider": request.provider,
            "original_model": request.model_id,
            "fallback_used": False,
            "fallback_skipped_reason": "disabled_for_regeneration",
            **_venice_safe_mode_diagnostics(
                provider=request.provider,
                safe_mode=request.safe_mode,
                prefix="primary_",
            ),
        }
        try:
            response = await primary_provider.generate_image(request)
        except ProviderError as exc:
            diagnostics["classification"] = (
                "suspected_blocked_image_output"
                if _is_suspected_blocked_image_error(exc)
                else "primary_image_generation_failed"
            )
            diagnostics.update(_primary_error_diagnostics(exc))
            raise _ImageGenerationFailure(diagnostics, exc) from exc

        if not _is_suspected_blocked_image_response(response):
            return _ImageGenerationResult(
                response=response,
                request=request,
                diagnostics={
                    **diagnostics,
                    **_response_diagnostics(response.raw_metadata),
                    "final_provider": response.provider,
                    "final_model": response.model_id,
                },
            )

        diagnostics["classification"] = "suspected_blocked_image_output"
        diagnostics.update(_primary_response_diagnostics(response.raw_metadata))
        if _response_has_image_data(response):
            return _ImageGenerationResult(
                response=response,
                request=request,
                diagnostics={
                    **diagnostics,
                    **_response_diagnostics(response.raw_metadata),
                    "final_provider": response.provider,
                    "final_model": response.model_id,
                },
            )
        raise _ImageGenerationFailure(
            {
                **diagnostics,
                "final_provider": response.provider,
                "final_model": response.model_id,
                "final_error_category": (
                    ProviderErrorCategory.IMAGE_GENERATION_FAILED.value
                ),
            },
            ValueError("Image provider returned no image data"),
        )

    async def _generate_fallback_image(
        self,
        *,
        fallback: _ImageFallbackRequest,
        diagnostics: dict[str, object],
    ) -> _ImageGenerationResult:
        request = fallback.request
        log_event(
            "provider.image_fallback_started",
            provider=request.provider,
            model=request.model_id,
            task=fallback.task,
        )
        try:
            response = await self.providers[request.provider].generate_image(request)
        except Exception as exc:
            raise _ImageGenerationFailure(
                {
                    **diagnostics,
                    "fallback_used": True,
                    "fallback_provider": request.provider,
                    "fallback_model": request.model_id,
                    "fallback_task": fallback.task,
                    **_venice_safe_mode_diagnostics(
                        provider=request.provider,
                        safe_mode=request.safe_mode,
                        prefix="fallback_",
                    ),
                    **_exception_diagnostics(exc),
                },
                exc,
            ) from exc
        if not _response_has_image_data(response):
            raise _ImageGenerationFailure(
                {
                    **diagnostics,
                    **_response_diagnostics(response.raw_metadata),
                    "fallback_used": True,
                    "fallback_provider": request.provider,
                    "fallback_model": request.model_id,
                    "fallback_task": fallback.task,
                    "final_provider": response.provider,
                    "final_model": response.model_id,
                    "final_error_category": (
                        ProviderErrorCategory.IMAGE_GENERATION_FAILED.value
                    ),
                    **_venice_safe_mode_diagnostics(
                        provider=request.provider,
                        safe_mode=request.safe_mode,
                        prefix="fallback_",
                    ),
                },
                ValueError("Fallback image provider returned no image data"),
            )
        return _ImageGenerationResult(
            response=response,
            request=request,
            diagnostics={
                **diagnostics,
                **_response_diagnostics(response.raw_metadata),
                "fallback_used": True,
                "fallback_provider": request.provider,
                "fallback_model": request.model_id,
                "fallback_task": fallback.task,
                "final_provider": response.provider,
                "final_model": response.model_id,
                **_venice_safe_mode_diagnostics(
                    provider=request.provider,
                    safe_mode=request.safe_mode,
                    prefix="fallback_",
                ),
            },
        )

    async def _persist_generated_image_with_optional_fallback(
        self,
        *,
        save_id: str,
        source_message_id: str,
        generation_id: str,
        generation: _ImageGenerationResult,
    ) -> tuple[_ImageGenerationResult, str, str | None]:
        try:
            path, thumbnail_path = self._persist_image(
                response=generation.response,
                save_id=save_id,
                source_message_id=source_message_id,
                generation_id=generation_id,
            )
            return generation, path, thumbnail_path
        except Exception as exc:
            if generation.diagnostics.get("fallback_used") is True:
                if _image_persistence_error_allows_fallback(exc):
                    raise _ImageGenerationFailure(
                        {
                            **generation.diagnostics,
                            **_image_persistence_error_diagnostics(
                                exc,
                                prefix="final",
                            ),
                        },
                        exc,
                    ) from exc
                raise
            if not _image_persistence_error_allows_fallback(exc):
                raise

            diagnostics = {
                **generation.diagnostics,
                "classification": "primary_image_not_stored",
                **_image_persistence_error_diagnostics(exc, prefix="primary"),
            }
            fallback = self._fallback_image_request(
                save_id=save_id,
                request=generation.request,
            )
            if fallback is None:
                diagnostics["fallback_skipped_reason"] = _image_fallback_skip_reason(
                    repositories=self.repositories,
                    providers=self.providers,
                    save_id=save_id,
                    required_capability=(
                        ProviderCapability.IMAGE_TO_IMAGE
                        if _image_request_has_source_media(generation.request)
                        else ProviderCapability.IMAGE_GENERATION
                    ),
                )
                raise _ImageGenerationFailure(diagnostics, exc) from exc

            fallback_generation = await self._generate_fallback_image(
                fallback=fallback,
                diagnostics=diagnostics,
            )
            try:
                path, thumbnail_path = self._persist_image(
                    response=fallback_generation.response,
                    save_id=save_id,
                    source_message_id=source_message_id,
                    generation_id=generation_id,
                )
            except Exception as fallback_exc:
                if _image_persistence_error_allows_fallback(fallback_exc):
                    raise _ImageGenerationFailure(
                        {
                            **fallback_generation.diagnostics,
                            **_image_persistence_error_diagnostics(
                                fallback_exc,
                                prefix="final",
                            ),
                        },
                        fallback_exc,
                    ) from fallback_exc
                raise
            return fallback_generation, path, thumbnail_path

    async def _video_request(
        self,
        *,
        provider: str,
        model_id: str,
        prompt: str,
        save_id: str,
        source_message_id: str,
        source_media_asset_id: str | None = None,
        source_media_path: Path | None = None,
        source_content_rating: str = CONTENT_RATING_UNCLASSIFIED,
        current_user_id: str | None = None,
    ) -> VideoRequest:
        content_safety = effective_content_safety_policy(
            self.repositories,
            user_id=current_user_id,
        )
        safety = await self.content_safety_service.review_media_prompt(
            prompt=prompt,
            content_rating=content_safety.rating,
            save_id=save_id,
            source_provider=provider,
            source_model_id=model_id,
        )
        if safety.action is not ContentSafetyAction.ALLOW:
            raise ValueError("Video prompt exceeds the selected content rating")
        if content_rating_exceeds(
            minimum_rating=source_content_rating,
            allowed_rating=content_safety.rating,
        ):
            raise ValueError("Source media exceeds the selected content rating")
        _raise_unless_enforced_safe_mode_provider(
            provider=provider,
            force_safe_mode=content_safety.force_venice_safe_mode,
        )
        task = (
            "image_animation"
            if source_media_asset_id or source_media_path is not None
            else "video_generation"
        )
        return VideoRequest(
            provider=provider,
            model_id=model_id,
            prompt=prompt,
            source_save_id=save_id,
            source_message_id=source_message_id,
            source_media_asset_id=source_media_asset_id,
            source_media_path=source_media_path,
            content_rating=maximum_content_rating(
                (safety.minimum_rating, source_content_rating)
            ),
            safe_mode=_request_video_safe_mode(
                repositories=self.repositories,
                provider=provider,
                model_id=model_id,
                save_id=save_id,
                current_user_id=current_user_id,
            ),
            force_safe_mode=content_safety.force_venice_safe_mode,
            openrouter_app_title=(
                openrouter_app_title_for_task(task)
                if provider == OPENROUTER_PROVIDER_NAME
                else None
            ),
        )

    async def _generate_video_with_optional_fallback(
        self,
        *,
        save_id: str,
        request: VideoRequest,
        required_capability: ProviderCapability,
    ) -> _VideoGenerationResult:
        primary_provider = self.providers[request.provider]
        if not isinstance(primary_provider, VideoProvider):
            raise ValueError(f"Provider does not support video: {request.provider}")
        diagnostics: dict[str, object] = {
            "original_provider": request.provider,
            "original_model": request.model_id,
            "fallback_used": False,
            **_venice_safe_mode_diagnostics(
                provider=request.provider,
                safe_mode=request.safe_mode,
                prefix="primary_",
            ),
        }
        try:
            response = await primary_provider.generate_video(request)
        except ProviderError as exc:
            if not _is_suspected_blocked_video_error(exc):
                raise
            diagnostics["classification"] = "suspected_blocked_video_output"
            diagnostics.update(_primary_error_diagnostics(exc))
            fallback = self._fallback_video_request(
                save_id=save_id,
                request=request,
                required_capability=required_capability,
            )
            if fallback is None:
                diagnostics["fallback_skipped_reason"] = _video_fallback_skip_reason(
                    repositories=self.repositories,
                    providers=self.providers,
                    save_id=save_id,
                    required_capability=required_capability,
                )
                raise _ImageGenerationFailure(diagnostics, exc) from exc
            return await self._generate_fallback_video(
                request=fallback,
                diagnostics=diagnostics,
            )
        if not _is_suspected_blocked_video_response(response):
            return _VideoGenerationResult(
                response=response,
                request=request,
                diagnostics={
                    **diagnostics,
                    **_response_diagnostics(response.raw_metadata),
                    "final_provider": response.provider,
                    "final_model": response.model_id,
                },
            )
        diagnostics["classification"] = "suspected_blocked_video_output"
        diagnostics.update(_primary_response_diagnostics(response.raw_metadata))
        fallback = self._fallback_video_request(
            save_id=save_id,
            request=request,
            required_capability=required_capability,
        )
        if fallback is None:
            diagnostics["fallback_skipped_reason"] = _video_fallback_skip_reason(
                repositories=self.repositories,
                providers=self.providers,
                save_id=save_id,
                required_capability=required_capability,
            )
            if _response_has_video_data(response):
                return _VideoGenerationResult(
                    response=response,
                    request=request,
                    diagnostics={
                        **diagnostics,
                        **_response_diagnostics(response.raw_metadata),
                        "final_provider": response.provider,
                        "final_model": response.model_id,
                    },
                )
            raise _ImageGenerationFailure(
                {
                    **diagnostics,
                    "final_provider": response.provider,
                    "final_model": response.model_id,
                    "final_error_category": (
                        ProviderErrorCategory.IMAGE_GENERATION_FAILED.value
                    ),
                },
                ValueError("Video provider returned no video data"),
            )
        return await self._generate_fallback_video(
            request=fallback,
            diagnostics=diagnostics,
        )

    async def _generate_fallback_video(
        self,
        *,
        request: VideoRequest,
        diagnostics: dict[str, object],
    ) -> _VideoGenerationResult:
        provider_object: object = self.providers[request.provider]
        if not _is_video_provider(provider_object):
            raise ValueError(
                f"Fallback provider does not support video: {request.provider}"
            )
        provider = cast(VideoProvider, provider_object)
        try:
            response = await provider.generate_video(request)
        except Exception as exc:
            raise _ImageGenerationFailure(
                {
                    **diagnostics,
                    "fallback_used": True,
                    "fallback_provider": request.provider,
                    "fallback_model": request.model_id,
                    **_venice_safe_mode_diagnostics(
                        provider=request.provider,
                        safe_mode=request.safe_mode,
                        prefix="fallback_",
                    ),
                    **_exception_diagnostics(exc),
                },
                exc,
            ) from exc
        if not _response_has_video_data(response):
            raise _ImageGenerationFailure(
                {
                    **diagnostics,
                    **_response_diagnostics(response.raw_metadata),
                    "fallback_used": True,
                    "fallback_provider": request.provider,
                    "fallback_model": request.model_id,
                    "final_provider": response.provider,
                    "final_model": response.model_id,
                    "final_error_category": (
                        ProviderErrorCategory.IMAGE_GENERATION_FAILED.value
                    ),
                    **_venice_safe_mode_diagnostics(
                        provider=request.provider,
                        safe_mode=request.safe_mode,
                        prefix="fallback_",
                    ),
                },
                ValueError("Fallback video provider returned no video data"),
            )
        return _VideoGenerationResult(
            response=response,
            request=request,
            diagnostics={
                **diagnostics,
                **_response_diagnostics(response.raw_metadata),
                "fallback_used": True,
                "fallback_provider": request.provider,
                "fallback_model": request.model_id,
                "final_provider": response.provider,
                "final_model": response.model_id,
                **_venice_safe_mode_diagnostics(
                    provider=request.provider,
                    safe_mode=request.safe_mode,
                    prefix="fallback_",
                ),
            },
        )

    def _fallback_video_request(
        self,
        *,
        save_id: str,
        request: VideoRequest,
        required_capability: ProviderCapability,
    ) -> VideoRequest | None:
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose=_VIDEO_FALLBACK_TASK,
        )
        if preference is None:
            return None
        if (
            request.force_safe_mode
            and preference.provider != _VENICE_PROVIDER_NAME
        ):
            return None
        provider_object: object | None = self.providers.get(preference.provider)
        if not _is_video_provider(provider_object):
            return None
        if not _model_supports_video_fallback(
            repositories=self.repositories,
            provider=preference.provider,
            model_id=preference.model_id,
            required_capability=required_capability,
        ):
            return None
        return replace(
            request,
            provider=preference.provider,
            model_id=preference.model_id,
            dimensions=image_generation_dimensions(
                self.repositories,
                provider=preference.provider,
                model_id=preference.model_id,
                save_id=request.source_save_id,
            ),
            safe_mode=_request_video_safe_mode(
                repositories=self.repositories,
                provider=preference.provider,
                model_id=preference.model_id,
                save_id=request.source_save_id,
                force_safe_mode=request.force_safe_mode,
            ),
            openrouter_app_title=(
                openrouter_app_title_for_task(_VIDEO_FALLBACK_TASK)
                if preference.provider == OPENROUTER_PROVIDER_NAME
                else None
            ),
        )

    def _fallback_image_request(
        self,
        *,
        save_id: str,
        request: ImageRequest,
    ) -> _ImageFallbackRequest | None:
        required_capability = (
            ProviderCapability.IMAGE_TO_IMAGE
            if _image_request_has_source_media(request)
            else ProviderCapability.IMAGE_GENERATION
        )
        selected_task: str | None = None
        selected_preference: ModelPreferenceRecord | None = None
        for task in _image_fallback_candidate_tasks(required_capability):
            preference = roleplay_model_preference(
                repositories=self.repositories,
                save_id=save_id,
                purpose=task,
            )
            if preference is None or preference.provider not in self.providers:
                continue
            if (
                request.force_safe_mode
                and preference.provider != _VENICE_PROVIDER_NAME
            ):
                continue
            if not _model_supports_image_fallback(
                repositories=self.repositories,
                provider=preference.provider,
                model_id=preference.model_id,
                required_capability=required_capability,
            ):
                continue
            selected_task = task
            selected_preference = preference
            break
        if selected_task is None or selected_preference is None:
            return None
        fallback = replace(
            request,
            provider=selected_preference.provider,
            model_id=selected_preference.model_id,
            openrouter_provider_routing=None,
            safe_mode=_request_safe_mode(
                repositories=self.repositories,
                provider=selected_preference.provider,
                save_id=request.source_save_id,
                force_safe_mode=request.force_safe_mode,
            ),
        )
        return _ImageFallbackRequest(
            request=request_with_openrouter_routing(
                self.repositories,
                _trim_image_request_references(
                    fallback,
                    limit=_image_reference_limit(
                        provider=self.providers[selected_preference.provider],
                        model_id=selected_preference.model_id,
                    ),
                ),
                task=selected_task,
                save_id=request.source_save_id,
            ),
            task=selected_task,
        )

    async def generate_automatic_if_due(
        self,
        *,
        save_id: str,
        source_message_id: str | None = None,
        current_user_id: str | None = None,
    ) -> MediaAssetRecord | None:
        prepared = self.prepare_automatic_if_due(
            save_id=save_id,
            source_message_id=source_message_id,
        )
        if prepared is None:
            return None
        return await self.generate_prepared_automatic(
            prepared,
            current_user_id=current_user_id,
        )

    def _automatic_source_if_due(
        self,
        *,
        save_id: str,
        source_message_id: str | None = None,
        media_type: str = "image",
    ) -> tuple[MessageRecord, int] | None:
        automatic_enabled = _automatic_image_generation_enabled(
            self.repositories,
            save_id=save_id,
            default=self.automatic_enabled,
        )
        auto_frequency = _image_generation_frequency(
            self.repositories,
            save_id=save_id,
            default=self.auto_frequency,
        )
        if not automatic_enabled:
            log_event(
                "media.automatic_skipped",
                save_id=save_id,
                reason="automatic_disabled",
                auto_frequency=auto_frequency,
            )
            return None

        if auto_frequency <= 0:
            log_event(
                "media.automatic_skipped",
                save_id=save_id,
                reason="disabled",
                auto_frequency=auto_frequency,
            )
            return None

        narrator_messages = [
            message
            for message in self.repositories.list_messages(save_id)
            if message.role == "narrator"
        ]
        if len(narrator_messages) == 0:
            log_event(
                "media.automatic_skipped",
                save_id=save_id,
                reason="no_narrator_messages",
            )
            return None
        source_message, narrator_message_count = _automatic_source_message(
            narrator_messages=narrator_messages,
            source_message_id=source_message_id,
        )
        if source_message is None:
            log_event(
                "media.automatic_skipped",
                save_id=save_id,
                reason="source_message_not_narrator",
                source_message_id=source_message_id,
            )
            return None
        if _is_safety_transition_source(source_message):
            log_event(
                "media.automatic_skipped",
                save_id=save_id,
                reason="safety_transition",
                source_message_id=source_message.id,
            )
            return None

        if narrator_message_count % auto_frequency != 0:
            log_event(
                "media.automatic_skipped",
                save_id=save_id,
                reason="frequency_not_due",
                source_message_id=source_message.id,
                narrator_message_count=narrator_message_count,
                auto_frequency=auto_frequency,
            )
            return None

        existing_source_ids = {
            asset.source_message_id
            for asset in self.repositories.list_media_assets(save_id)
            if asset.type == media_type
        }
        if source_message.id in existing_source_ids:
            log_event(
                "media.automatic_skipped",
                save_id=save_id,
                reason="already_generated",
                source_message_id=source_message.id,
            )
            return None

        log_event(
            "media.automatic_due",
            save_id=save_id,
            source_message_id=source_message.id,
            narrator_message_count=narrator_message_count,
        )
        return source_message, narrator_message_count

    async def _draft_image_prompt(
        self,
        *,
        save_id: str,
        source_message_id: str,
        scene_context: str,
    ) -> str:
        preferences = _image_prompt_preferences(
            repositories=self.repositories,
            save_id=save_id,
        )
        if not preferences:
            raise ValueError("No image prompt model preference configured")
        empty_prompt_error: str | None = None
        for preference in preferences:
            provider = self.providers.get(preference.provider)
            if provider is None:
                raise ValueError(
                    f"Image prompt provider is unavailable: {preference.provider}"
                )
            if not _model_supports_image_prompt(
                repositories=self.repositories,
                provider=preference.provider,
                model_id=preference.model_id,
            ):
                log_event(
                    "media.image_prompt_preference_skipped",
                    save_id=save_id,
                    source_message_id=source_message_id,
                    provider=preference.provider,
                    model=preference.model_id,
                    reason="model_lacks_chat_capability",
                )
                continue
            response = await chat_with_fallback(
                repositories=self.repositories,
                providers=self.providers,
                save_id=save_id,
                task="image_prompt",
                request=ChatRequest(
                    provider=preference.provider,
                    model_id=preference.model_id,
                    prompt_purpose=ChatPromptPurpose.IMAGE_PROMPT,
                    messages=(
                        ChatMessage(
                            role="system",
                            body=(
                                "Write one concise image-generation prompt for the "
                                "selected roleplay scene. Include the visible "
                                "subject, setting, action or pose, facial expression, "
                                "important objects, lighting, weather, time of "
                                "day, mood, composition, and continuity "
                                "constraints when "
                                "they are supported by the context. Treat the "
                                "selected scene message as the highest-priority "
                                "current moment for subject, action, setting, "
                                "and composition. Use "
                                "deterministic scene context, active linked facts, "
                                "older chronicle, "
                                "scenario setup, and prior image continuity only "
                                "when they describe visible details for this "
                                "moment without contradicting the selected scene "
                                "message. Reject unsupported, "
                                "internal, "
                                "private, or future details. Do not specify character "
                                "clothing or add Wearing directives; the application "
                                "adds Current Clothing separately. Return plain "
                                "prompt text with no explanation."
                            ),
                        ),
                    ),
                    current_scene_recap=(scene_context,),
                    temperature=0.4,
                    max_output_tokens=10_000,
                ),
            )
            prompt = response.body.strip()
            if not prompt:
                empty_prompt_error = (
                    "Image prompt model returned empty output: "
                    f"{response.provider}/{response.model_id}"
                )
                log_error_event(
                    "media.image_prompt_empty",
                    save_id=save_id,
                    source_message_id=source_message_id,
                    provider=response.provider,
                    model=response.model_id,
                    scene_context_chars=len(scene_context),
                )
                continue
            log_event(
                "media.image_prompt_drafted",
                save_id=save_id,
                source_message_id=source_message_id,
                provider=response.provider,
                model=response.model_id,
                scene_context_chars=len(scene_context),
                prompt_chars=len(prompt),
            )
            return prompt
        raise ValueError(empty_prompt_error or "Image prompt response was empty")

    def _build_scene_context(
        self,
        *,
        save_id: str,
        source_message_id: str | None = None,
    ) -> str:
        scene_context, _breakdown = self._build_scene_context_with_breakdown(
            save_id=save_id,
            source_message_id=source_message_id,
        )
        return scene_context

    def _build_scene_context_with_breakdown(
        self,
        *,
        save_id: str,
        source_message_id: str | None = None,
    ) -> tuple[str, ContextAssemblyBreakdown]:
        return ContextAssemblyService(self.repositories).build_image_scene_context(
            save_id=save_id,
            source_message_id=source_message_id,
        )

    def _persist_image(
        self,
        *,
        response: ImageResponse,
        save_id: str,
        source_message_id: str,
        generation_id: str,
    ) -> tuple[str, str | None]:
        relative_path = _response_relative_path(
            response=response,
            save_id=save_id,
            source_message_id=source_message_id,
            generation_id=generation_id,
        )
        output_path = self.media_dir / relative_path
        _assert_within_media_dir(media_dir=self.media_dir, output_path=output_path)
        ensure_private_dir(output_path.parent)

        if response.image_bytes is not None:
            _assert_persisted_image_size(len(response.image_bytes))
            write_private_bytes(output_path, response.image_bytes)
            thumbnail_path = _persist_thumbnail(
                media_dir=self.media_dir,
                image_relative_path=relative_path,
                image_path=output_path,
            )
            return relative_path.as_posix(), thumbnail_path
        if response.image_path is not None:
            source_path = response.image_path
            if not source_path.is_absolute():
                source_path = self.media_dir / source_path
            _assert_within_media_dir(
                media_dir=self.media_dir,
                output_path=source_path,
            )
            if not source_path.is_file():
                raise ValueError("Image provider returned a missing image file")
            _assert_persisted_image_size(source_path.stat().st_size)
            if source_path != output_path:
                write_private_bytes(output_path, source_path.read_bytes())
            else:
                output_path.chmod(0o600)
            thumbnail_path = _persist_thumbnail(
                media_dir=self.media_dir,
                image_relative_path=relative_path,
                image_path=output_path,
            )
            return relative_path.as_posix(), thumbnail_path
        raise ValueError("Image provider returned no image data")

    def _persist_video(
        self,
        *,
        response: VideoResponse,
        save_id: str,
        source_message_id: str,
        generation_id: str,
    ) -> str:
        relative_path = _video_response_relative_path(
            response=response,
            save_id=save_id,
            source_message_id=source_message_id,
            generation_id=generation_id,
        )
        output_path = self.media_dir / relative_path
        _assert_within_media_dir(media_dir=self.media_dir, output_path=output_path)
        ensure_private_dir(output_path.parent)
        if response.video_bytes is not None:
            _assert_persisted_video_size(len(response.video_bytes))
            write_private_bytes(output_path, response.video_bytes)
            return relative_path.as_posix()
        if response.video_path is not None:
            source_path = response.video_path
            if not source_path.is_absolute():
                source_path = self.media_dir / source_path
            _assert_within_media_dir(
                media_dir=self.media_dir,
                output_path=source_path,
            )
            if not source_path.is_file():
                raise ValueError("Video provider returned a missing video file")
            _assert_persisted_video_size(source_path.stat().st_size)
            if source_path != output_path:
                write_private_bytes(output_path, source_path.read_bytes())
            else:
                output_path.chmod(0o600)
            return relative_path.as_posix()
        raise ValueError("Video provider returned no video data")

    def _delete_persisted_files(
        self,
        image_path: str | None,
        thumbnail_path: str | None,
    ) -> None:
        for relative_path in (thumbnail_path, image_path):
            if not relative_path:
                continue
            path = self.media_dir / relative_path
            try:
                _assert_within_media_dir(media_dir=self.media_dir, output_path=path)
                if path.is_file():
                    path.unlink()
            except Exception as exc:
                log_error_event(
                    "media.cleanup_failed",
                    path=relative_path,
                    **exception_log_fields(exc),
                )

    def _delete_scenario_starter_reference_files_if_unreferenced(
        self,
        reference: ScenarioStarterReferenceImage | None,
        *,
        starters: Iterable[ScenarioCharacterStarter],
    ) -> None:
        if reference is None:
            return
        referenced_paths: set[str] = set()
        for starter in starters:
            current_reference = starter.reference_image
            if current_reference is None:
                continue
            referenced_paths.add(current_reference.path)
            if current_reference.thumbnail_path:
                referenced_paths.add(current_reference.thumbnail_path)
        for relative_path in (reference.thumbnail_path, reference.path):
            if (
                not relative_path
                or relative_path in referenced_paths
                or self._scenario_starter_reference_path_is_referenced(relative_path)
            ):
                continue
            try:
                _assert_scenario_starter_reference_path(relative_path)
                path = self.media_dir / relative_path
                _assert_within_media_dir(media_dir=self.media_dir, output_path=path)
                path.unlink(missing_ok=True)
            except Exception as exc:
                log_error_event(
                    "media.cleanup_failed",
                    path=relative_path,
                    **exception_log_fields(exc),
                )

    def _scenario_starter_reference_path_is_referenced(
        self,
        relative_path: str,
    ) -> bool:
        for scenario in self.repositories.list_scenarios():
            content = _scenario_content(scenario.content_json)
            starters = scenario_character_starters_for_content(
                scenario_type=scenario.type,
                content=content,
            )
            for starter in starters:
                reference = starter.reference_image
                if reference is None:
                    continue
                if reference.path == relative_path:
                    return True
                if reference.thumbnail_path == relative_path:
                    return True
        return False


def _image_prompt_preferences(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
) -> tuple[ModelPreferenceRecord, ...]:
    preferences: list[ModelPreferenceRecord] = []
    if not shared_roleplay_models_enabled(repositories):
        save = repositories.get_save(save_id)
        scenario = repositories.get_scenario(save.scenario_id) if save else None
        scenario_type = scenario.type if scenario else None
        scenario_task = (
            roleplay_model_task(roleplay_type=scenario_type, purpose="image_prompt")
            if scenario_type in ROLEPLAY_TYPES
            else None
        )
        if scenario_task is not None:
            scenario_preference = repositories.get_model_preference(scenario_task)
            if scenario_preference is not None:
                preferences.append(scenario_preference)
    preferences.extend(_shared_image_prompt_preferences(repositories))
    return tuple(_deduplicate_model_preferences(preferences))


def _shared_image_prompt_preferences(
    repositories: PersistenceRepositories,
) -> tuple[ModelPreferenceRecord, ...]:
    return tuple(
        preference
        for task in ("image_prompt", "chat")
        if (preference := repositories.get_model_preference(task)) is not None
    )


def _is_text_message_beat(body: str) -> bool:
    normalized = body.casefold()
    if not any(token in normalized for token in ("text", "phone", "screen")):
        return False
    message_markers = (
        "---",
        "**",
        ">",
        "sms",
        "message",
        "reply",
    )
    if not any(marker in normalized for marker in message_markers):
        return False
    scenic_terms = (
        "stands",
        "walks",
        "runs",
        "reaches",
        "looks",
        "room",
        "street",
        "forest",
        "battle",
        "door",
        "window",
    )
    return not any(term in normalized for term in scenic_terms)


def _deduplicate_model_preferences(
    preferences: list[ModelPreferenceRecord],
) -> list[ModelPreferenceRecord]:
    seen: set[tuple[str, str]] = set()
    unique: list[ModelPreferenceRecord] = []
    for preference in preferences:
        key = (preference.provider, preference.model_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(preference)
    return unique


def _source_message(
    *,
    messages: list[MessageRecord],
    source_message_id: str,
) -> MessageRecord | None:
    for message in messages:
        if message.id == source_message_id:
            return message
    return None


def _is_safety_transition_source(message: MessageRecord) -> bool:
    return is_fade_to_black_message(
        role=message.role,
        body=message.body,
        safety_transition=message.safety_transition,
    )


def _raise_if_safety_transition_source(message: MessageRecord) -> None:
    if _is_safety_transition_source(message):
        raise ValueError("Fade-to-black transitions cannot be media sources")


def _present_character_ids_for_message(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    source_message_id: str,
) -> frozenset[str]:
    messages = repositories.list_messages(save_id)
    if _source_message(messages=messages, source_message_id=source_message_id) is None:
        raise ValueError(f"Unknown source message id: {source_message_id}")
    records = repositories.list_message_scene_presence(
        save_id,
        message_id=source_message_id,
    )
    if records:
        return frozenset(record.character_id for record in records)
    latest_message_id = messages[-1].id if messages else None
    if source_message_id != latest_message_id:
        return frozenset()
    snapshot = repositories.get_scene_snapshot(save_id)
    return frozenset(snapshot.present_character_ids if snapshot else ())


def _media_asset_by_id(
    media_assets: list[MediaAssetRecord],
    media_asset_id: str,
) -> MediaAssetRecord | None:
    for asset in media_assets:
        if asset.id == media_asset_id:
            return asset
    return None


def _character_for_reference(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    character_id: str | None,
) -> CharacterRecord | None:
    if character_id is None:
        return _primary_character_for_reference(
            repositories=repositories,
            save_id=save_id,
        )
    character = repositories.get_character(character_id)
    if character is None or character.save_id != save_id:
        return None
    return character


def _selected_scene_character_references(
    *,
    repositories: PersistenceRepositories,
    media_dir: Path,
    save_id: str,
    source_message_id: str,
) -> tuple[_SelectedImageReference, ...]:
    characters = tuple(repositories.list_characters(save_id))
    if not characters:
        return ()
    characters_by_id = {character.id: character for character in characters}
    source_message = _source_message(
        messages=repositories.list_messages(save_id),
        source_message_id=source_message_id,
    )
    source_text = source_message.body if source_message is not None else ""
    candidate_ids: list[str] = []
    seen_character_ids: set[str] = set()

    snapshot = repositories.get_scene_snapshot(save_id)
    present_character_ids = snapshot.present_character_ids if snapshot else ()
    for character_id in present_character_ids:
        if character_id not in characters_by_id or character_id in seen_character_ids:
            continue
        candidate_ids.append(character_id)
        seen_character_ids.add(character_id)

    for character in characters:
        if character.id in seen_character_ids:
            continue
        if not character_name_is_mentioned(
            name=character.name,
            aliases=character.aliases,
            text=source_text,
        ):
            continue
        candidate_ids.append(character.id)
        seen_character_ids.add(character.id)

    references: list[_SelectedImageReference] = []
    seen_media_asset_ids: set[str] = set()
    for character_id in candidate_ids:
        candidate_character = characters_by_id.get(character_id)
        if candidate_character is None:
            continue
        asset = _linked_character_reference_asset(
            repositories=repositories,
            save_id=save_id,
            character_id=candidate_character.id,
        )
        if asset is None or asset.id in seen_media_asset_ids:
            continue
        media_path = media_dir / asset.path
        _assert_within_media_dir(media_dir=media_dir, output_path=media_path)
        if not media_path.is_file():
            continue
        references.append(
            _SelectedImageReference(
                character_id=candidate_character.id,
                character_name=candidate_character.name,
                media_asset_id=asset.id,
                media_path=media_path,
            )
        )
        seen_media_asset_ids.add(asset.id)
    return tuple(references)


def _character_reference_asset(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    character_id: str | None = None,
    include_generated_fallback: bool = True,
) -> MediaAssetRecord | None:
    media_assets = {
        asset.id: asset
        for asset in repositories.list_media_assets(save_id)
    }
    character = _character_for_reference(
        repositories=repositories,
        save_id=save_id,
        character_id=character_id,
    )
    if character is None:
        return None
    for link in repositories.list_entity_links(save_id):
        if (
            link.entity_type == "character"
            and link.entity_id == character.id
            and link.target_type == "media_asset"
            and link.relation == _CHARACTER_REFERENCE_RELATION
        ):
            asset = media_assets.get(link.target_id)
            if _is_usable_character_reference_asset(asset):
                return asset
    if not include_generated_fallback:
        return None
    fallback = _first_generated_image(media_assets.values())
    if fallback is None:
        return None
    _set_character_reference_link(
        repositories=repositories,
        save_id=save_id,
        character_id=character.id,
        media_asset_id=fallback.id,
    )
    return fallback


def _linked_character_reference_asset(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    character_id: str | None = None,
) -> MediaAssetRecord | None:
    media_assets = {
        asset.id: asset
        for asset in repositories.list_media_assets(save_id)
    }
    character = _character_for_reference(
        repositories=repositories,
        save_id=save_id,
        character_id=character_id,
    )
    if character is None:
        return None
    for link in repositories.list_entity_links(save_id):
        if (
            link.entity_type == "character"
            and link.entity_id == character.id
            and link.target_type == "media_asset"
            and link.relation == _CHARACTER_REFERENCE_RELATION
        ):
            asset = media_assets.get(link.target_id)
            if _is_usable_character_reference_asset(asset):
                return asset
    return None


def _set_character_reference_link(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    character_id: str,
    media_asset_id: str,
) -> None:
    for link in repositories.list_entity_links(save_id):
        if (
            link.entity_type == "character"
            and link.entity_id == character_id
            and link.target_type == "media_asset"
            and link.relation == _CHARACTER_REFERENCE_RELATION
        ):
            repositories.delete_entity_link(link.id)
    repositories.add_entity_link(
        save_id=save_id,
        entity_type="character",
        entity_id=character_id,
        target_type="media_asset",
        target_id=media_asset_id,
        relation=_CHARACTER_REFERENCE_RELATION,
    )


def _replace_character_reference_media_links(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    old_media_asset_id: str,
    new_media_asset_id: str,
) -> None:
    for link in tuple(repositories.list_entity_links(save_id)):
        if not (
            link.target_type == "media_asset"
            and link.target_id == old_media_asset_id
            and link.relation == _CHARACTER_REFERENCE_RELATION
        ):
            continue
        repositories.delete_entity_link(link.id)
        repositories.add_entity_link(
            save_id=save_id,
            entity_type=link.entity_type,
            entity_id=link.entity_id,
            target_type="media_asset",
            target_id=new_media_asset_id,
            relation=link.relation,
            source_message_id=link.source_message_id,
        )


def _is_usable_character_reference_asset(
    asset: MediaAssetRecord | None,
) -> bool:
    return (
        asset is not None
        and asset.type == "image"
        and asset.status == "succeeded"
    )


def _media_asset_metadata(asset: MediaAssetRecord) -> dict[str, object]:
    if not asset.metadata_json:
        return {}
    try:
        loaded = json.loads(asset.metadata_json)
    except ValueError:
        return {}
    return dict(loaded) if isinstance(loaded, dict) else {}


def _replacement_request_source_message_id(
    asset: MediaAssetRecord,
    metadata: dict[str, object],
) -> str:
    if asset.source_message_id:
        return asset.source_message_id
    for key in ("text_message_id", "request_source_message_id", "source_message_id"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return asset.id


def _replacement_source_media_asset_ids(
    asset: MediaAssetRecord,
    metadata: dict[str, object],
) -> tuple[str, ...]:
    candidates: list[str] = []
    for key in ("source_character_reference_asset_ids", "source_media_asset_ids"):
        values = _metadata_string_list(metadata.get(key))
        if values:
            candidates = values
            break
    if not candidates:
        for key in ("source_character_reference_asset_id", "source_media_asset_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                candidates = [value.strip()]
                break
    if not candidates and asset.source_media_asset_id:
        candidates = [asset.source_media_asset_id]
    return tuple(dict.fromkeys(item for item in candidates if item != asset.id))


def _metadata_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _replacement_image_request_task(
    asset: MediaAssetRecord,
    metadata: dict[str, object],
    *,
    has_source_media: bool,
) -> str:
    kind = metadata.get("kind") or _media_asset_kind(asset)
    if kind == "character_text_character_image":
        return TEXT_MESSAGE_IMAGE_EDIT_PURPOSE
    if kind == "character_image":
        return CHARACTER_IMAGE_EDIT_PURPOSE
    if kind == "scene_image" and has_source_media:
        return SCENE_IMAGE_EDIT_PURPOSE
    if has_source_media:
        return IMAGE_TO_IMAGE_GENERATION_PURPOSE
    return "image_generation"


def _replacement_request_model_id(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    asset: MediaAssetRecord,
    metadata: dict[str, object],
    request_task: str,
) -> str:
    if asset.provider == OPENROUTER_PROVIDER_NAME:
        requested_model = metadata.get(_REQUESTED_MODEL_METADATA_KEY)
        if isinstance(requested_model, str) and requested_model.strip():
            return requested_model.strip()
        catalog_model = _openrouter_catalog_request_model_for_response_model(
            repositories=repositories,
            response_model=asset.model,
        )
        if catalog_model is not None:
            return catalog_model
    preference = _replacement_request_model_preference(
        repositories=repositories,
        save_id=save_id,
        request_task=request_task,
    )
    if (
        asset.provider == OPENROUTER_PROVIDER_NAME
        and preference is not None
        and preference.provider == asset.provider
        and _openrouter_response_model_matches_request_model(
            response_model=asset.model,
            request_model=preference.model_id,
        )
    ):
        return preference.model_id
    return asset.model


def _replacement_request_model_preference(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    request_task: str,
) -> ModelPreferenceRecord | None:
    if request_task in {
        CHARACTER_IMAGE_EDIT_PURPOSE,
        SCENE_IMAGE_EDIT_PURPOSE,
        TEXT_MESSAGE_IMAGE_EDIT_PURPOSE,
    }:
        return image_edit_model_preference(
            repositories=repositories,
            save_id=save_id,
            purpose=request_task,
        )
    return roleplay_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose=request_task,
    )


def _openrouter_catalog_request_model_for_response_model(
    *,
    repositories: PersistenceRepositories,
    response_model: str,
) -> str | None:
    for model in repositories.list_provider_models(OPENROUTER_PROVIDER_NAME):
        if not model.available or model.model_id == response_model:
            continue
        if _openrouter_response_model_matches_request_model(
            response_model=response_model,
            request_model=model.model_id,
        ):
            return model.model_id
    return None


def _image_asset_metadata(
    metadata: dict[str, object] | None,
    *,
    generation: _ImageGenerationResult,
) -> dict[str, object]:
    result = dict(metadata or {})
    source_content_rating = result.get("content_rating")
    ratings = [generation.request.content_rating]
    if isinstance(source_content_rating, str):
        ratings.append(source_content_rating)
    result["content_rating"] = maximum_content_rating(tuple(ratings))
    if (
        generation.request.provider != OPENROUTER_PROVIDER_NAME
        and generation.response.provider != OPENROUTER_PROVIDER_NAME
    ):
        return result
    result[_REQUESTED_MODEL_METADATA_KEY] = generation.request.model_id
    if generation.response.model_id != generation.request.model_id:
        result[_RESPONSE_MODEL_METADATA_KEY] = generation.response.model_id
    else:
        result.pop(_RESPONSE_MODEL_METADATA_KEY, None)
    return result


def _media_generation_source_rating(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    source_message_id: str,
    source_media_asset_id: str | None,
) -> str:
    messages = {
        message.id: message
        for message in repositories.list_messages(save_id)
    }
    media_assets = {
        asset.id: asset
        for asset in repositories.list_media_assets(save_id)
    }
    if source_media_asset_id is not None:
        source_asset = media_assets.get(source_media_asset_id)
        if source_asset is None:
            return CONTENT_RATING_UNCLASSIFIED
        return media_asset_content_rating(
            source_asset,
            media_assets_by_id=media_assets,
            source_messages=messages,
        )
    source_message = messages.get(source_message_id)
    return (
        source_message.content_rating
        if source_message is not None
        else CONTENT_RATING_UNCLASSIFIED
    )


def _persisted_image_model(generation: _ImageGenerationResult) -> str:
    if (
        generation.response.provider == OPENROUTER_PROVIDER_NAME
        and _openrouter_response_model_matches_request_model(
            response_model=generation.response.model_id,
            request_model=generation.request.model_id,
        )
    ):
        return generation.request.model_id
    return generation.response.model_id


def _openrouter_response_model_matches_request_model(
    *,
    response_model: str,
    request_model: str,
) -> bool:
    if response_model == request_model:
        return True
    prefix = f"{request_model}-"
    if not response_model.startswith(prefix):
        return False
    suffix = response_model.removeprefix(prefix)
    return len(suffix) == 8 and suffix.isdigit()


def _copy_reference_media_file(
    *,
    media_dir: Path,
    source_relative_path: str,
    target_save_id: str,
    asset_id: str,
    thumbnail: bool = False,
    missing_allowed: bool = False,
) -> str | None:
    source_path = media_dir / source_relative_path
    _assert_within_media_dir(media_dir=media_dir, output_path=source_path)
    if not source_path.is_file():
        if missing_allowed:
            return None
        raise ValueError("Character reference image file is unavailable")
    suffix = "".join(Path(source_relative_path).suffixes) or ".bin"
    destination_relative = (
        Path(_safe_path_segment(target_save_id))
        / "reference-clones"
        / f"{'thumb-' if thumbnail else ''}{_safe_path_segment(asset_id)}{suffix}"
    )
    destination_path = media_dir / destination_relative
    _assert_within_media_dir(media_dir=media_dir, output_path=destination_path)
    write_private_bytes(destination_path, source_path.read_bytes())
    return destination_relative.as_posix()


def _first_generated_image(
    media_assets: Iterable[MediaAssetRecord],
) -> MediaAssetRecord | None:
    for asset in media_assets:
        if (
            asset.type == "image"
            and asset.status == "succeeded"
            and _media_asset_kind(asset) is None
        ):
            return asset
    return None


def _media_asset_kind(asset: MediaAssetRecord) -> str | None:
    if not asset.metadata_json:
        return None
    try:
        metadata = json.loads(asset.metadata_json)
    except ValueError:
        return None
    if not isinstance(metadata, dict):
        return None
    kind = metadata.get("kind")
    return kind if isinstance(kind, str) else None


def _scenario_content(content_json: str) -> dict[str, object]:
    try:
        loaded = json.loads(content_json)
    except ValueError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _scenario_starter_index(
    starters: tuple[ScenarioCharacterStarter, ...],
    *,
    starter_id: str | None,
    starter_name: str,
) -> int:
    normalized_id = (starter_id or "").strip()
    if normalized_id:
        for index, starter in enumerate(starters):
            if starter.starter_id == normalized_id:
                return index
        raise ValueError(f"Unknown character starter id: {normalized_id}")
    normalized_name = _starter_lookup_key(starter_name)
    if not normalized_name:
        raise ValueError("Character starter name is required")
    matches = [
        index
        for index, starter in enumerate(starters)
        if _starter_lookup_key(starter.name) == normalized_name
    ]
    if not matches:
        raise ValueError(f"Unknown character starter name: {starter_name}")
    if len(matches) > 1:
        raise ValueError(f"Character starter name is ambiguous: {starter_name}")
    return matches[0]


def _starter_lookup_key(value: str) -> str:
    return " ".join(value.casefold().split())


def _primary_character_for_reference(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
) -> CharacterRecord | None:
    characters = repositories.list_characters(save_id)
    if not characters:
        return None
    return characters[0]


def _scenario_type_for_save(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
) -> str | None:
    save = repositories.get_save(save_id)
    if save is None:
        return None
    scenario = repositories.get_scenario(save.scenario_id)
    return scenario.type if scenario is not None else None


def _first_narrator_message_id(messages: list[MessageRecord]) -> str | None:
    for message in messages:
        if message.role == "narrator":
            return message.id
    return None


def _scene_character_visual_directions(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    source_message_id: str,
    action_context: str,
) -> str:
    return "\n\n".join(
        part
        for character in _scene_characters(
            repositories=repositories,
            save_id=save_id,
            source_message_id=source_message_id,
        )
        if (
            part := _character_visual_direction_block(
                character,
                action_context=action_context,
            )
        )
    )


def _scene_characters(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    source_message_id: str,
) -> tuple[CharacterRecord, ...]:
    characters = tuple(repositories.list_characters(save_id))
    if not characters:
        return ()
    characters_by_id = {character.id: character for character in characters}
    source_message = _source_message(
        messages=repositories.list_messages(save_id),
        source_message_id=source_message_id,
    )
    source_text = source_message.body if source_message is not None else ""
    selected: list[CharacterRecord] = []
    seen_character_ids: set[str] = set()
    for character_id in _present_character_ids_for_message(
        repositories=repositories,
        save_id=save_id,
        source_message_id=source_message_id,
    ):
        character = characters_by_id.get(character_id)
        if character is None or character_id in seen_character_ids:
            continue
        selected.append(character)
        seen_character_ids.add(character_id)
    for character in characters:
        if character.id in seen_character_ids or not character_name_is_mentioned(
            name=character.name,
            aliases=character.aliases,
            text=source_text,
        ):
            continue
        selected.append(character)
        seen_character_ids.add(character.id)
    return tuple(selected)


def _current_clothing_completion_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "characters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "character_id": {"type": "string"},
                        "current_clothing": {"type": "string"},
                    },
                    "required": ["character_id", "current_clothing"],
                },
            }
        },
        "required": ["characters"],
    }


def _current_clothing_completion_messages(
    *,
    characters: tuple[CharacterRecord, ...],
    image_context: str,
) -> tuple[ChatMessage, ...]:
    character_lines = "\n".join(
        (
            f"Character ID: {character.id}\n"
            f"Name: {character.name}\n"
            f"Role: {character.role}\n"
            f"Stable appearance: {character.appearance}\n"
            f"Visual notes: {character.visual_notes}"
        )
        for character in characters
    )
    return (
        ChatMessage(
            role="system",
            body=(
                "Complete Current Clothing for characters depicted in a requested "
                "roleplay image using the enforced response schema. Return exactly "
                "one entry for every supplied character ID. Describe only the "
                "current outfit, clothing, armor, uniform, footwear, and worn "
                "accessories. Prefer clothing explicitly supported by the image "
                "context. When none is stated, infer a concise plausible outfit "
                "that fits the moment and character. Do not copy stable physical "
                "appearance, body, face, hair, eyes, skin, personality, or pose "
                "into Current Clothing."
            ),
        ),
        ChatMessage(
            role="user",
            body=_compact_text(
                f"Requested image context:\n{image_context}\n\n{character_lines}",
                max_chars=6000,
                label="current clothing context",
            ),
        ),
    )


def _current_clothing_from_data(
    data: object,
    *,
    expected_character_ids: set[str],
    appearance_by_character_id: dict[str, str],
) -> dict[str, str]:
    if not isinstance(data, dict):
        raise ValueError("current clothing response must be an object")
    raw_characters = data.get("characters")
    if not isinstance(raw_characters, list):
        raise ValueError("current clothing response must include characters")
    completed: dict[str, str] = {}
    for item in raw_characters:
        if not isinstance(item, dict):
            raise ValueError("current clothing character entries must be objects")
        character_id = item.get("character_id")
        clothing = item.get("current_clothing")
        if (
            not isinstance(character_id, str)
            or character_id not in expected_character_ids
        ):
            raise ValueError(
                "current clothing response contains an unknown character ID"
            )
        if character_id in completed:
            raise ValueError(
                "current clothing response contains a duplicate character ID"
            )
        if not isinstance(clothing, str) or not clothing.strip():
            raise ValueError("current clothing values must not be blank")
        normalized_clothing = _normalized_visual_comparison(clothing)
        normalized_appearance = _normalized_visual_comparison(
            appearance_by_character_id.get(character_id, "")
        )
        if (
            normalized_appearance
            and normalized_clothing == normalized_appearance
        ):
            raise ValueError(
                "current clothing must not repeat the stable appearance field"
            )
        completed[character_id] = " ".join(clothing.strip().split())
    if set(completed) != expected_character_ids:
        raise ValueError("current clothing response is missing requested character IDs")
    return completed


def _normalized_visual_comparison(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold()))


def _prompt_with_current_clothing_direction(
    prompt: str,
    *,
    character: CharacterRecord,
) -> str:
    clothing = _visual_direction_field(character.current_clothing)
    marker = f"Character visual direction for {character.name}:"
    cleaned_prompt = _without_wearing_directives(prompt)
    lines = cleaned_prompt.splitlines()
    if not clothing or marker.casefold() not in cleaned_prompt.casefold():
        return "\n".join(lines)
    for index, line in enumerate(lines):
        if line.strip().casefold() == marker.casefold():
            lines.insert(
                index + 1,
                f"Wearing: {_ensure_terminal_punctuation(clothing)}",
            )
            return "\n".join(lines)
    return "\n".join(
        (
            *lines,
            "",
            marker,
            f"Wearing: {_ensure_terminal_punctuation(clothing)}",
        )
    )


def _without_wearing_directives(value: str) -> str:
    without_wearing = re.sub(
        r"(?i)\bwearing\s*:[^\n]*",
        "",
        value,
    )
    return "\n".join(line.rstrip() for line in without_wearing.splitlines()).strip()


def _prompt_with_character_visual_directions(
    prompt: str,
    character_visual_directions: str,
) -> str:
    prompt = _without_wearing_directives(prompt)
    if not character_visual_directions.strip():
        return prompt
    if character_visual_directions.strip() in prompt:
        return prompt
    return f"{prompt.strip()}\n\n{character_visual_directions.strip()}"


def _has_character_visual_direction(text: str) -> bool:
    return "character visual direction for " in text.casefold()


def _character_visual_direction_block(
    character: CharacterRecord,
    *,
    action_context: str = "",
    expression_context: str = "",
    action_fallback: str = "natural pose supported by the current context",
    expression_fallback: str = (
        "expression grounded in this moment; infer from the selected action, "
        "mood, and dialogue without contradicting context"
    ),
    stable_identity: bool = False,
) -> str:
    wearing = (
        _character_stable_wearing_direction(character)
        if stable_identity
        else _character_wearing_direction(character)
    )
    action = _visual_direction_field(action_context) or action_fallback
    expression = _visual_direction_field(expression_context) or expression_fallback
    lines = [f"Character visual direction for {character.name}:"]
    if wearing:
        lines.append(f"Wearing: {_ensure_terminal_punctuation(wearing)}")
    lines.extend(
        (
            f"Current action/pose: {_ensure_terminal_punctuation(action)}",
            f"Facial expression: {_ensure_terminal_punctuation(expression)}",
        )
    )
    return _compact_text(
        "\n".join(lines),
        max_chars=_CHARACTER_VISUAL_DIRECTION_MAX_CHARS,
        label="character visual direction",
    )


def _character_wearing_direction(character: CharacterRecord) -> str:
    return _visual_direction_field(character.current_clothing)


def _character_stable_wearing_direction(character: CharacterRecord) -> str:
    for value in (
        character.visual_notes,
        character.appearance,
    ):
        text = _visual_direction_field(value)
        if text:
            return text
    return "consistent with established stable character appearance"


def _visual_direction_field(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.strip().split())
    if not text:
        return ""
    if len(text) <= _CHARACTER_VISUAL_DIRECTION_FIELD_MAX_CHARS:
        return text
    return (
        text[: _CHARACTER_VISUAL_DIRECTION_FIELD_MAX_CHARS - 3].rstrip()
        + "..."
    )


def _ensure_terminal_punctuation(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped[-1] in ".!?":
        return stripped
    return f"{stripped}."


def _character_reference_prompt(character: CharacterRecord) -> str:
    parts = [
        f"Character reference portrait for {character.name}.",
        "Create a consistent reusable reference image of this roleplay character.",
        "Focus on stable visible identity, face, build, hair, eyes, clothing, "
        "posture, and overall presence. Keep the background simple and avoid "
        "adding new story events.",
        _character_visual_direction_block(
            character,
            action_fallback="stable reusable reference portrait pose",
            expression_fallback="neutral, reusable character-reference expression",
            stable_identity=True,
        ),
    ]
    visual_fields = _deduplicated_character_reference_fields(
        (
            ("Age", character.age),
            ("Appearance", character.appearance),
            ("Visual notes", character.visual_notes),
        )
    )
    if not visual_fields:
        visual_fields = _deduplicated_character_reference_fields(
            (("Character cue", character.role),)
        )

    field_max_chars = (
        _CHARACTER_REFERENCE_SINGLE_FIELD_MAX_CHARS
        if len(visual_fields) <= 1
        else _CHARACTER_REFERENCE_MULTI_FIELD_MAX_CHARS
    )
    if visual_fields and visual_fields[0][0] == "Character cue":
        field_max_chars = _CHARACTER_REFERENCE_FALLBACK_FIELD_MAX_CHARS

    for label, text in visual_fields:
        parts.append(
            f"{label}: "
            f"{_compact_reference_prompt_field(text, max_chars=field_max_chars)}"
        )
    return _compact_text(
        "\n\n".join(parts),
        max_chars=_CHARACTER_REFERENCE_PROMPT_MAX_CHARS,
        label="character reference prompt",
    )


def _character_text_uploaded_photo_system_prompt() -> str:
    return (
        "Describe the uploaded photo as concise natural prose for an in-world "
        "text-message recipient. Focus on visible details the recipient could "
        "reasonably notice: people, objects, text, setting, condition, mood, "
        "and any clues relevant to a reply. Do not invent sender intent, hidden "
        "facts, or information outside the image. If image text is visible, "
        "transcribe only short readable snippets."
    )


def _character_text_uploaded_photo_description_prompt(
    *,
    text_message: CharacterTextMessageRecord,
) -> str:
    text_line = f"Text sent with the photo: {text_message.body.strip()}"
    return "\n".join(
        line
        for line in (
            "Describe this photo for the recipient of the accompanying text.",
            text_line,
        )
        if line
    )


def _solo_character_scene_image_prompt(
    *,
    character: CharacterRecord,
    character_name: str,
    action_context: str,
    scene_context: str,
) -> str:
    visual_direction = _character_visual_direction_block(
        character,
        action_context=action_context,
    )
    return _compact_text(
        "\n\n".join(
            (
                f"Create a solo image of {character_name} in this roleplay moment.",
                "Use the source reference image as the identity anchor. Preserve the "
                "same face, build, hair, visible age cues, and recognizable "
                "character identity.",
                "Show only this one character. Do not include other people, crowds, "
                "companions, duplicate versions of the character, reflections that "
                "show extra people, or background figures.",
                "Use the selected scene context only for pose, expression, action, "
                "lighting, setting, mood, weather, props, and composition. Ignore "
                "private, hidden, unsupported, or future details.",
                visual_direction,
                f"Scene context:\n{scene_context}",
            )
        ),
        max_chars=_CHARACTER_IMAGE_PROMPT_MAX_CHARS,
        label="character image prompt",
    )


def _solo_character_text_image_prompt(
    *,
    character: CharacterRecord,
    character_name: str,
    text_body: str,
    visual_prompt: str,
    scene_context: str,
) -> str:
    visual_direction = (
        ""
        if _has_character_visual_direction(visual_prompt)
        else _character_visual_direction_block(
            character,
            action_context=visual_prompt or text_body,
        )
    )
    return _compact_text(
        "\n\n".join(
            part
            for part in (
                f"Create an in-world picture text from {character_name}.",
                "Use the source reference image as the identity anchor. Preserve the "
                "same face, build, hair, visible age cues, and recognizable "
                "character identity.",
                "Show only this one character. This can be a selfie, outfit check, "
                "expression, pose, or current appearance update when supported.",
                "Do not include other people, crowds, companions, duplicate versions "
                "of the character, reflections that show extra people, or background "
                "figures.",
                "Use only visual details supported by the text conversation and "
                "local world context. Ignore private, hidden, unsupported, or future "
                "details.",
                visual_direction,
                f"NPC text message:\n{text_body.strip()}",
                f"Requested picture:\n{visual_prompt.strip()}",
                f"Context:\n{scene_context.strip()}",
            )
            if part.strip()
        ),
        max_chars=_CHARACTER_IMAGE_PROMPT_MAX_CHARS,
        label="character text image prompt",
    )


def _object_context_text_image_prompt(
    *,
    character_name: str,
    text_body: str,
    visual_prompt: str,
    scene_context: str,
) -> str:
    return _compact_text(
        "\n\n".join(
            (
                f"Create an in-world picture attachment sent by {character_name}.",
                "Depict the concrete object, clue, document, gift, location detail, "
                "food, ticket, note, or other visible subject that the text message "
                "is about.",
                "Do not depict a phone screenshot, chat UI, captions, watermarks, "
                "or readable body text unless the requested subject is itself a "
                "document or note.",
                "Ground every visual detail in the text conversation and local world "
                "context. Ignore private, hidden, unsupported, or future details.",
                f"NPC text message:\n{text_body.strip()}",
                f"Requested picture:\n{visual_prompt.strip()}",
                f"Context:\n{scene_context.strip()}",
            )
        ),
        max_chars=_CHARACTER_IMAGE_PROMPT_MAX_CHARS,
        label="character text object image prompt",
    )


def _solo_character_registry_image_prompt(
    *,
    character: CharacterRecord,
    character_name: str,
    instructions: str,
) -> str:
    cleaned_instructions = instructions.strip()
    visual_direction = _character_visual_direction_block(
        character,
        action_context=cleaned_instructions,
        action_fallback="stable solo character-picture pose",
    )
    parts = [
        f"Create a solo image of {character_name}.",
        "Use the source reference image as the identity anchor. Preserve the same "
        "face, build, hair, visible age cues, and recognizable character identity.",
        "Show only this one character. Do not include other people, crowds, "
        "companions, duplicate versions of the character, reflections that show "
        "extra people, or background figures.",
        "This is a generated character picture for the registry; do not replace or "
        "redesign the reference identity.",
        visual_direction,
    ]
    if cleaned_instructions:
        parts.append(f"User instructions: {cleaned_instructions}")
    return _compact_text(
        "\n\n".join(parts),
        max_chars=_CHARACTER_IMAGE_PROMPT_MAX_CHARS,
        label="character image prompt",
    )


def _deduplicated_character_reference_fields(
    fields: Iterable[tuple[str, object]],
) -> list[tuple[str, str]]:
    selected: list[tuple[str, str, str]] = []
    for label, value in fields:
        text = _reference_prompt_text(value)
        normalized = _normalized_reference_prompt_text(text)
        if not normalized:
            continue

        duplicate_index: int | None = None
        for index, (_existing_label, _existing_text, existing_normalized) in enumerate(
            selected
        ):
            if (
                normalized == existing_normalized
                or normalized in existing_normalized
                or existing_normalized in normalized
            ):
                duplicate_index = index
                break
        if duplicate_index is None:
            selected.append((label, text, normalized))
            continue
        if len(normalized) > len(selected[duplicate_index][2]):
            selected[duplicate_index] = (label, text, normalized)
    return [(label, text) for label, text, _normalized in selected]


def _reference_prompt_text(value: object) -> str:
    return " ".join(str(value).split())


def _normalized_reference_prompt_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _compact_reference_prompt_field(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


def _image_to_image_prompt(prompt: str, *, reference_count: int = 1) -> str:
    noun = "image" if reference_count == 1 else "images"
    return (
        f"Use the attached character reference {noun} as visual identity "
        "references. Preserve each referenced character's stable face, body type, "
        "hair, eyes, visible traits, and styling while depicting the requested "
        "scene.\n\n"
        f"{prompt.strip()}"
    )


def _normalized_source_media_asset_ids(
    source_media_asset_id: str | None,
    source_media_asset_ids: tuple[str, ...],
) -> tuple[str, ...]:
    if source_media_asset_ids:
        return tuple(item for item in source_media_asset_ids if item)
    return (source_media_asset_id,) if source_media_asset_id else ()


def _normalized_source_media_paths(
    source_media_path: Path | None,
    source_media_paths: tuple[Path, ...],
) -> tuple[Path, ...]:
    if source_media_paths:
        return source_media_paths
    return (source_media_path,) if source_media_path is not None else ()


def _primary_request_source_media_asset_id(request: ImageRequest) -> str | None:
    asset_ids = _normalized_source_media_asset_ids(
        request.source_media_asset_id,
        request.source_media_asset_ids,
    )
    return asset_ids[0] if asset_ids else None


def _image_request_has_source_media(request: ImageRequest) -> bool:
    return bool(
        _normalized_source_media_asset_ids(
            request.source_media_asset_id,
            request.source_media_asset_ids,
        )
        or _normalized_source_media_paths(
            request.source_media_path,
            request.source_media_paths,
        )
    )


def _trim_image_request_references(
    request: ImageRequest,
    *,
    limit: int,
) -> ImageRequest:
    if limit < 1 or not _image_request_has_source_media(request):
        return replace(
            request,
            source_media_asset_id=None,
            source_media_path=None,
            source_media_asset_ids=(),
            source_media_paths=(),
        )
    asset_ids = _normalized_source_media_asset_ids(
        request.source_media_asset_id,
        request.source_media_asset_ids,
    )[:limit]
    paths = _normalized_source_media_paths(
        request.source_media_path,
        request.source_media_paths,
    )[:limit]
    return replace(
        request,
        source_media_asset_id=asset_ids[0] if asset_ids else None,
        source_media_path=paths[0] if paths else None,
        source_media_asset_ids=asset_ids,
        source_media_paths=paths,
    )


def _image_reference_limit(
    *,
    provider: object,
    model_id: str,
) -> int:
    if isinstance(provider, ImageReferenceLimitProvider):
        try:
            limit = int(provider.image_reference_limit(model_id))
        except (TypeError, ValueError):
            return 1
        return max(1, limit)
    return 1


def _model_supports_image_to_image(
    *,
    repositories: PersistenceRepositories,
    provider: str,
    model_id: str,
) -> bool:
    return model_supports_any_capability_or_unknown(
        repositories,
        provider=provider,
        model_id=model_id,
        required=IMAGE_TO_IMAGE_CAPABILITIES,
    )


def _automatic_media_mode(repositories: PersistenceRepositories) -> str:
    value = repositories.get_app_setting("automatic_media_mode")
    return "video" if value == "video" else "image"


def _animation_prompt(
    *,
    source_prompt: str,
    scene_context: str,
    motion_prompt: str,
    max_chars: int | None = None,
) -> str:
    parts = [
        "Animate the source image as a short roleplay scene video.",
    ]
    if motion_prompt.strip():
        parts.append(f"Motion guidance: {motion_prompt.strip()}")
    parts.extend(
        [
            f"Image prompt: {source_prompt.strip()}",
            f"Scene context: {scene_context.strip()}",
        ]
    )
    prompt = "\n\n".join(part for part in parts if part.strip())
    if max_chars is None:
        return prompt
    return _compact_text(
        prompt,
        max_chars=max_chars,
        label="animation prompt",
    )


def _animation_prompt_max_chars(provider: str) -> int | None:
    if provider == _VENICE_PROVIDER_NAME:
        return _VENICE_ANIMATION_PROMPT_MAX_CHARS
    return None


def _compact_text(text: str, *, max_chars: int, label: str) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text

    marker = f"\n...[truncated {label}]...\n"
    available = max_chars - len(marker)
    if available <= 0:
        return text[:max_chars]

    head_chars = max(1, available // 2)
    tail_chars = max(1, available - head_chars)
    return (text[:head_chars].rstrip() + marker + text[-tail_chars:].lstrip())[
        :max_chars
    ]


def _automatic_source_message(
    *,
    narrator_messages: list[MessageRecord],
    source_message_id: str | None,
) -> tuple[MessageRecord | None, int]:
    if source_message_id is None:
        return narrator_messages[-1], len(narrator_messages)
    for index, message in enumerate(narrator_messages, start=1):
        if message.id == source_message_id:
            return message, index
    return None, 0


def _request_safe_mode(
    *,
    repositories: PersistenceRepositories,
    provider: str,
    save_id: str | None = None,
    current_user_id: str | None = None,
    force_safe_mode: bool = False,
) -> bool | None:
    if provider != _VENICE_PROVIDER_NAME:
        return None
    if force_safe_mode or effective_content_safety_policy(
        repositories,
        user_id=current_user_id,
    ).force_venice_safe_mode:
        return True
    value = repositories.get_effective_setting(
        _VENICE_IMAGE_SAFE_MODE_SETTING,
        save_id=save_id,
    )
    return value if isinstance(value, bool) else True


def _raise_unless_enforced_safe_mode_provider(
    *,
    provider: str,
    force_safe_mode: bool,
) -> None:
    if force_safe_mode and provider != _VENICE_PROVIDER_NAME:
        raise ValueError(_ENFORCED_MEDIA_SAFE_MODE_REQUIRED_ERROR)


def _request_video_safe_mode(
    *,
    repositories: PersistenceRepositories,
    provider: str,
    model_id: str,
    save_id: str | None = None,
    current_user_id: str | None = None,
    force_safe_mode: bool = False,
) -> bool | None:
    if provider != _VENICE_PROVIDER_NAME:
        return None
    if force_safe_mode or effective_content_safety_policy(
        repositories,
        user_id=current_user_id,
    ).force_venice_safe_mode:
        return True
    if not _model_supports_safe_mode_parameter(
        repositories=repositories,
        provider=provider,
        model_id=model_id,
    ):
        return None
    return _request_safe_mode(
        repositories=repositories,
        provider=provider,
        save_id=save_id,
        current_user_id=current_user_id,
        force_safe_mode=force_safe_mode,
    )


def _automatic_image_generation_enabled(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    default: bool,
) -> bool:
    value = repositories.get_effective_setting(
        "automatic_image_generation_enabled",
        save_id=save_id,
    )
    return value if isinstance(value, bool) else default


def _image_generation_frequency(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    default: int,
) -> int:
    value = repositories.get_effective_setting(
        "image_generation_frequency",
        save_id=save_id,
    )
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _venice_safe_mode_diagnostics(
    *,
    provider: str,
    safe_mode: bool | None,
    prefix: str = "",
) -> dict[str, object]:
    if provider != _VENICE_PROVIDER_NAME or safe_mode is None:
        return {}
    return {f"{prefix}venice_safe_mode": safe_mode}


def _model_supports_safe_mode_parameter(
    *,
    repositories: PersistenceRepositories,
    provider: str,
    model_id: str,
) -> bool:
    model = find_provider_model(
        repositories,
        provider=provider,
        model_id=model_id,
    )
    if model is None or not model.available:
        return False
    return any(
        value.strip().lower().replace("-", "_")
        in {"safe_mode", "image_safe_mode", "media_safe_mode", "video_safe_mode"}
        for value in model.supported_parameters
        if isinstance(value, str)
    )


def _model_supports_image_fallback(
    *,
    repositories: PersistenceRepositories,
    provider: str,
    model_id: str,
    required_capability: ProviderCapability = ProviderCapability.IMAGE_GENERATION,
) -> bool:
    required = (
        IMAGE_TO_IMAGE_CAPABILITIES
        if required_capability is ProviderCapability.IMAGE_TO_IMAGE
        else IMAGE_GENERATION_CAPABILITIES
    )
    return model_supports_any_capability(
        repositories,
        provider=provider,
        model_id=model_id,
        required=required,
    )


def _image_fallback_candidate_tasks(
    required_capability: ProviderCapability,
) -> tuple[str, ...]:
    if required_capability is ProviderCapability.IMAGE_TO_IMAGE:
        return (_IMAGE_EDIT_FALLBACK_TASK, _IMAGE_FALLBACK_TASK)
    return (_IMAGE_FALLBACK_TASK,)


def _model_supports_image_prompt(
    *,
    repositories: PersistenceRepositories,
    provider: str,
    model_id: str,
) -> bool:
    model = find_provider_model(
        repositories,
        provider=provider,
        model_id=model_id,
    )
    if model is not None:
        if not model.available:
            return False
        return model_supports_any_capability(
            repositories,
            provider=provider,
            model_id=model_id,
            required=CHAT_CAPABILITIES | {"text"},
        )
    return True


def _is_video_provider(provider: object) -> bool:
    return callable(getattr(provider, "generate_video", None))


def _model_supports_video_fallback(
    *,
    repositories: PersistenceRepositories,
    provider: str,
    model_id: str,
    required_capability: ProviderCapability,
) -> bool:
    required = (
        IMAGE_TO_VIDEO_CAPABILITIES
        if required_capability is ProviderCapability.IMAGE_TO_VIDEO
        else TEXT_TO_VIDEO_CAPABILITIES
    )
    return model_supports_any_capability(
        repositories,
        provider=provider,
        model_id=model_id,
        required=required,
    )


def _video_model_requirement_error(
    *,
    repositories: PersistenceRepositories,
    preference: ModelPreferenceRecord,
    required_capability: ProviderCapability,
) -> str | None:
    if required_capability is ProviderCapability.IMAGE_TO_VIDEO:
        return _known_media_model_requirement_error(
            repositories=repositories,
            preference=preference,
            required=IMAGE_TO_VIDEO_CAPABILITIES,
            unavailable_label="Image animation",
            missing_capability_message=(
                "Image animation model does not support image-to-video: "
                f"{preference.model_id}"
            ),
        )
    return _known_media_model_requirement_error(
        repositories=repositories,
        preference=preference,
        required=TEXT_TO_VIDEO_CAPABILITIES,
        unavailable_label="Video generation",
        missing_capability_message=(
            "Video generation model does not advertise text-to-video support"
        ),
    )


def _image_model_requirement_error(
    *,
    repositories: PersistenceRepositories,
    provider: str,
    model_id: str,
    source_media_asset_id: str | None,
) -> str | None:
    preference = ModelPreferenceRecord(
        id="image-request-model-check",
        task="image_generation",
        provider=provider,
        model_id=model_id,
    )
    if source_media_asset_id is not None:
        return _known_media_model_requirement_error(
            repositories=repositories,
            preference=preference,
            required=IMAGE_TO_IMAGE_CAPABILITIES,
            unavailable_label="Image-to-image generation",
            missing_capability_message=(
                "Image-to-image generation model does not advertise "
                "image-to-image support"
            ),
        )
    return _known_media_model_requirement_error(
        repositories=repositories,
        preference=preference,
        required=IMAGE_GENERATION_CAPABILITIES,
        unavailable_label="Image generation",
        missing_capability_message=(
            "Image generation model does not advertise image generation support"
        ),
    )


def _known_media_model_requirement_error(
    *,
    repositories: PersistenceRepositories,
    preference: ModelPreferenceRecord,
    required: frozenset[str],
    unavailable_label: str,
    missing_capability_message: str,
) -> str | None:
    check = check_model_capabilities(
        repositories,
        provider=preference.provider,
        model_id=preference.model_id,
        required=required,
    )
    if not check.found:
        return None
    if not check.available:
        return f"{unavailable_label} model is unavailable: {preference.model_id}"
    if not check.supported:
        return missing_capability_message
    return None


def _is_suspected_blocked_image_response(response: ImageResponse) -> bool:
    if not _response_has_image_data(response):
        return True
    return _metadata_indicates_blocked_output(response.raw_metadata)


def _is_suspected_blocked_video_response(response: VideoResponse) -> bool:
    if not _response_has_video_data(response):
        return True
    return _metadata_indicates_blocked_output(response.raw_metadata)


def _response_has_image_data(response: ImageResponse) -> bool:
    return response.image_bytes is not None or response.image_path is not None


def _response_has_video_data(response: VideoResponse) -> bool:
    return response.video_bytes is not None or response.video_path is not None


def _is_suspected_blocked_image_error(exc: ProviderError) -> bool:
    if exc.category is ProviderErrorCategory.CONTENT_BLOCKED:
        return True
    if exc.category in {
        ProviderErrorCategory.MODEL_NOT_FOUND,
        ProviderErrorCategory.RATE_LIMITED,
        ProviderErrorCategory.NETWORK_ERROR,
        ProviderErrorCategory.PROVIDER_ERROR,
    }:
        return True
    if (
        exc.category is ProviderErrorCategory.IMAGE_GENERATION_FAILED
        and _image_error_indicates_missing_data(exc.message)
    ):
        return True
    return _is_fast_exhausted_provider_retry(exc)


def _is_suspected_blocked_video_error(exc: ProviderError) -> bool:
    return (
        exc.category is ProviderErrorCategory.CONTENT_BLOCKED
        or exc.category
        in {
            ProviderErrorCategory.MODEL_NOT_FOUND,
            ProviderErrorCategory.RATE_LIMITED,
            ProviderErrorCategory.NETWORK_ERROR,
            ProviderErrorCategory.PROVIDER_ERROR,
        }
        or _is_fast_exhausted_provider_retry(exc)
    )


def _image_error_indicates_missing_data(message: str) -> bool:
    normalized = message.strip().casefold()
    return any(
        marker in normalized
        for marker in (
            "did not include images",
            "did not include image",
            "missing image",
            "no image data",
            "returned no image",
        )
    )


def _image_persistence_error_allows_fallback(exc: Exception) -> bool:
    if not isinstance(exc, ValueError):
        return False
    message = str(exc)
    return any(
        marker in message
        for marker in (
            "Image provider returned no image data",
            "Image provider returned a missing image file",
            "Generated image exceeded",
            "Resolved image path escapes media directory",
        )
    )


def _image_persistence_error_diagnostics(
    exc: Exception,
    *,
    prefix: str,
) -> dict[str, object]:
    message = redact_text(str(exc).strip()) or exc.__class__.__name__
    if len(message) > _MAX_DIAGNOSTIC_ERROR_MESSAGE_CHARS:
        message = f"{message[: _MAX_DIAGNOSTIC_ERROR_MESSAGE_CHARS - 3].rstrip()}..."
    return {
        f"{prefix}_error_category": ProviderErrorCategory.IMAGE_GENERATION_FAILED.value,
        f"{prefix}_error_message": message,
    }


def _is_fast_exhausted_provider_retry(exc: ProviderError) -> bool:
    if exc.category is not ProviderErrorCategory.PROVIDER_ERROR:
        return False
    if exc.retry_attempt_count is None or exc.max_retry_attempts is None:
        return False
    if exc.retry_attempt_count < exc.max_retry_attempts:
        return False

    attempts = _retry_attempts_diagnostics(exc.retry_attempts)
    failed_attempts = [
        attempt
        for attempt in attempts
        if isinstance(attempt.get("error_category"), str)
    ]
    if len(failed_attempts) < 2:
        return False
    for attempt in failed_attempts:
        duration_ms = attempt.get("duration_ms")
        if not isinstance(duration_ms, int):
            return False
        if duration_ms > _SUSPICIOUS_FAST_RETRY_MAX_DURATION_MS:
            return False
    return True


def _metadata_indicates_blocked_output(value: object) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = str(key).strip().lower().replace("-", "_")
            if normalized_key in {
                "content_filter",
                "content_filtered",
                "content_blocked",
                "refusal",
                "blocked",
                "safety",
                "x_venice_contains_minor",
                "x_venice_is_adult_model_content_violation",
                "x_venice_is_blurred",
                "x_venice_is_content_violation",
            } and _explicit_block_signal(item):
                return True
            if normalized_key in {"finish_reason", "native_finish_reason"}:
                reason = str(item).strip().lower().replace("-", "_")
                if reason in {
                    "content_filter",
                    "content_blocked",
                    "safety",
                    "refusal",
                }:
                    return True
            if _metadata_indicates_blocked_output(item):
                return True
    elif isinstance(value, list):
        return any(_metadata_indicates_blocked_output(item) for item in value)
    return False


def _truthy_block_signal(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_")
        return normalized not in {"", "false", "none", "null", "ok", "stop"}
    return True


def _explicit_block_signal(value: object) -> bool:
    return isinstance(value, bool | str) and _truthy_block_signal(value)


def _image_fallback_skip_reason(
    *,
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    save_id: str,
    required_capability: ProviderCapability = ProviderCapability.IMAGE_GENERATION,
) -> str:
    first_configured_skip_reason: str | None = None
    for task in _image_fallback_candidate_tasks(required_capability):
        preference = roleplay_model_preference(
            repositories=repositories,
            save_id=save_id,
            purpose=task,
        )
        if preference is None:
            continue
        required = (
            IMAGE_TO_IMAGE_CAPABILITIES
            if required_capability is ProviderCapability.IMAGE_TO_IMAGE
            else IMAGE_GENERATION_CAPABILITIES
        )
        check = check_model_capabilities(
            repositories,
            provider=preference.provider,
            model_id=preference.model_id,
            required=required,
        )
        if check.reason == MODEL_UNAVAILABLE_REASON:
            reason = "fallback_model_unavailable"
        elif preference.provider not in providers:
            reason = "fallback_provider_unavailable"
        elif not check.supported:
            reason = "fallback_model_lacks_required_capabilities"
        else:
            return "fallback_provider_unavailable"
        if first_configured_skip_reason is None:
            first_configured_skip_reason = reason
    if first_configured_skip_reason is not None:
        return first_configured_skip_reason
    if required_capability is ProviderCapability.IMAGE_TO_IMAGE:
        return "no_image_edit_fallback_model"
    return "no_fallback_model"


def _video_fallback_skip_reason(
    *,
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    save_id: str,
    required_capability: ProviderCapability,
) -> str:
    preference = roleplay_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose=_VIDEO_FALLBACK_TASK,
    )
    if preference is None:
        return "no_fallback_model"
    required = (
        IMAGE_TO_VIDEO_CAPABILITIES
        if required_capability is ProviderCapability.IMAGE_TO_VIDEO
        else TEXT_TO_VIDEO_CAPABILITIES
    )
    check = check_model_capabilities(
        repositories,
        provider=preference.provider,
        model_id=preference.model_id,
        required=required,
    )
    if check.reason == MODEL_UNAVAILABLE_REASON:
        return "fallback_model_unavailable"
    provider_object: object | None = providers.get(preference.provider)
    if not _is_video_provider(provider_object):
        return "fallback_provider_unavailable"
    if not check.supported:
        return "fallback_model_lacks_required_capabilities"
    return "fallback_provider_unavailable"


def _response_diagnostics(raw_metadata: dict[str, object]) -> dict[str, object]:
    return {
        **_retry_diagnostics(raw_metadata),
        **_provider_headers_diagnostics(raw_metadata),
    }


def _primary_response_diagnostics(raw_metadata: dict[str, object]) -> dict[str, object]:
    diagnostics: dict[str, object] = {}
    retry = _retry_diagnostics(raw_metadata)
    if "attempt_count" in retry:
        diagnostics["primary_attempt_count"] = retry["attempt_count"]
    if "max_attempts" in retry:
        diagnostics["primary_max_attempts"] = retry["max_attempts"]
    if "retry_attempts" in retry:
        diagnostics["primary_retry_attempts"] = retry["retry_attempts"]
    headers = _provider_headers_diagnostics(raw_metadata).get("provider_headers")
    if headers is not None:
        diagnostics["primary_provider_headers"] = headers
    return diagnostics


def _primary_error_diagnostics(exc: ProviderError) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "primary_error_category": exc.category.value,
    }
    primary_error_message = _safe_error_message(exc)
    if primary_error_message:
        diagnostics["primary_error_message"] = primary_error_message
    if exc.status_code is not None:
        diagnostics["primary_http_status"] = exc.status_code
    if exc.retry_attempt_count is not None:
        diagnostics["primary_attempt_count"] = exc.retry_attempt_count
        diagnostics["attempt_count"] = exc.retry_attempt_count
    if exc.max_retry_attempts is not None:
        diagnostics["primary_max_attempts"] = exc.max_retry_attempts
        diagnostics["max_attempts"] = exc.max_retry_attempts
    retry_attempts = _retry_attempts_diagnostics(exc.retry_attempts)
    if retry_attempts:
        diagnostics["primary_retry_attempts"] = retry_attempts
        diagnostics["retry_attempts"] = retry_attempts
    return diagnostics


def _retry_diagnostics(raw_metadata: dict[str, object]) -> dict[str, object]:
    retry = raw_metadata.get("_bragi_retry")
    if not isinstance(retry, dict):
        return {}
    attempt_count = retry.get("attempt_count")
    max_attempts = retry.get("max_attempts")
    retry_attempts = _retry_attempts_diagnostics(retry.get("attempts"))
    result: dict[str, object] = {}
    if isinstance(attempt_count, int):
        result["attempt_count"] = attempt_count
    if isinstance(max_attempts, int):
        result["max_attempts"] = max_attempts
    if retry_attempts:
        result["retry_attempts"] = retry_attempts
    return result


def _provider_headers_diagnostics(raw_metadata: dict[str, object]) -> dict[str, object]:
    headers = raw_metadata.get("_bragi_headers")
    if not isinstance(headers, dict):
        return {}
    safe_headers = {
        normalized_key: str(value)
        for key, value in headers.items()
        if isinstance(key, str)
        and isinstance(value, str)
        and (normalized_key := key.strip().lower()) in SAFE_PROVIDER_RESPONSE_HEADERS
    }
    if not safe_headers:
        return {}
    return {"provider_headers": safe_headers}


def _retry_attempts_diagnostics(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list | tuple):
        return []
    attempts: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        attempt = item.get("attempt")
        duration_ms = item.get("duration_ms")
        if not isinstance(attempt, int) or not isinstance(duration_ms, int):
            continue
        normalized: dict[str, object] = {
            "attempt": attempt,
            "duration_ms": duration_ms,
        }
        error_category = item.get("error_category")
        if isinstance(error_category, str) or error_category is None:
            normalized["error_category"] = error_category
        http_status = item.get("http_status")
        if isinstance(http_status, int):
            normalized["http_status"] = http_status
        attempts.append(normalized)
    return attempts


def _failed_image_result(
    *,
    exc: Exception,
) -> dict[str, object] | None:
    if isinstance(exc, _ImageGenerationFailure):
        return {
            **exc.diagnostics,
            **_exception_diagnostics(exc.cause),
        }
    return None


def _exception_diagnostics(exc: Exception) -> dict[str, object]:
    result: dict[str, object] = {}
    if isinstance(exc, ProviderError):
        result["final_error_category"] = exc.category.value
        final_error_message = _safe_error_message(exc)
        if final_error_message:
            result["final_error_message"] = final_error_message
        if exc.status_code is not None:
            result["final_http_status"] = exc.status_code
        if exc.retry_attempt_count is not None:
            result["attempt_count"] = exc.retry_attempt_count
        if exc.max_retry_attempts is not None:
            result["max_attempts"] = exc.max_retry_attempts
        retry_attempts = _retry_attempts_diagnostics(exc.retry_attempts)
        if retry_attempts:
            result["retry_attempts"] = retry_attempts
    return result


def _safe_error_message(exc: ProviderError) -> str:
    message = redact_text(exc.message.strip()) or ""
    if len(message) <= _MAX_DIAGNOSTIC_ERROR_MESSAGE_CHARS:
        return message
    return f"{message[: _MAX_DIAGNOSTIC_ERROR_MESSAGE_CHARS - 3].rstrip()}..."


def _response_relative_path(
    *,
    response: ImageResponse,
    save_id: str,
    source_message_id: str,
    generation_id: str,
) -> Path:
    filename = (
        f"{_safe_path_segment(source_message_id)}-"
        f"{_safe_path_segment(generation_id)}.png"
    )
    return Path(_safe_path_segment(save_id)) / filename


def _video_response_relative_path(
    *,
    response: VideoResponse,
    save_id: str,
    source_message_id: str,
    generation_id: str,
) -> Path:
    filename = (
        f"{_safe_path_segment(source_message_id)}-"
        f"{_safe_path_segment(generation_id)}{_video_extension(response.mime_type)}"
    )
    return Path(_safe_path_segment(save_id)) / filename


def _video_extension(mime_type: str) -> str:
    if mime_type == "video/mp4":
        return ".mp4"
    if mime_type == "video/webm":
        return ".webm"
    raise ValueError("Video provider returned an unsupported video MIME type")


def _thumbnail_relative_path(image_relative_path: Path) -> Path:
    return image_relative_path.parent / "thumbnails" / image_relative_path.name


def _persist_thumbnail(
    *,
    media_dir: Path,
    image_relative_path: Path,
    image_path: Path,
) -> str | None:
    thumbnail_relative_path = _thumbnail_relative_path(image_relative_path)
    thumbnail_path = media_dir / thumbnail_relative_path
    _assert_within_media_dir(media_dir=media_dir, output_path=thumbnail_path)
    ensure_private_dir(thumbnail_path.parent)
    if not _write_scaled_thumbnail(
        image_path=image_path,
        thumbnail_path=thumbnail_path,
    ):
        thumbnail_path.unlink(missing_ok=True)
        return None
    thumbnail_path.chmod(0o600)
    return thumbnail_relative_path.as_posix()


def _assert_persisted_image_size(byte_count: int) -> None:
    if byte_count > _MAX_PERSISTED_IMAGE_BYTES:
        raise ValueError(f"Generated image exceeded {_MAX_PERSISTED_IMAGE_BYTES} bytes")


def _assert_uploaded_image_size(byte_count: int) -> None:
    if byte_count > _MAX_PERSISTED_IMAGE_BYTES:
        raise ValueError(f"Uploaded image exceeded {_MAX_PERSISTED_IMAGE_BYTES} bytes")


def _assert_persisted_video_size(byte_count: int) -> None:
    if byte_count > _MAX_PERSISTED_VIDEO_BYTES:
        raise ValueError(f"Generated video exceeded {_MAX_PERSISTED_VIDEO_BYTES} bytes")


def _uploaded_image_mime_type(image_bytes: bytes) -> tuple[str, str]:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n") and _is_valid_png(image_bytes):
        png_decoded = _image_bytes_decode_with_gdkpixbuf(image_bytes, "image/png")
        if png_decoded is not False:
            return "image/png", ".png"
    jpeg_dimensions = _jpeg_dimensions(image_bytes)
    if jpeg_dimensions is not None and _decoded_dimensions_within_limit(
        jpeg_dimensions
    ):
        if _image_bytes_decode_with_gdkpixbuf(image_bytes, "image/jpeg") is True:
            return "image/jpeg", ".jpg"
    webp_dimensions = _webp_dimensions(image_bytes)
    if webp_dimensions is not None and _decoded_dimensions_within_limit(
        webp_dimensions
    ):
        if _image_bytes_decode_with_gdkpixbuf(image_bytes, "image/webp") is True:
            return "image/webp", ".webp"
    raise ValueError("Unsupported image upload type; use PNG, JPEG, or WebP")


def _decoded_dimensions_within_limit(
    dimensions: tuple[int, int],
    *,
    channels: int = 4,
) -> bool:
    width, height = dimensions
    return (
        width > 0
        and height > 0
        and width * height * channels <= _MAX_UPLOADED_IMAGE_DECODED_BYTES
    )


def _jpeg_dimensions(image_bytes: bytes) -> tuple[int, int] | None:
    if len(image_bytes) < 4 or image_bytes[:2] != b"\xff\xd8":
        return None
    offset = 2
    while offset < len(image_bytes):
        if image_bytes[offset] != 0xFF:
            return None
        while offset < len(image_bytes) and image_bytes[offset] == 0xFF:
            offset += 1
        if offset >= len(image_bytes):
            return None
        marker = image_bytes[offset]
        offset += 1
        if marker in {0x01, *range(0xD0, 0xD8)}:
            continue
        if marker in {0xD8, 0xD9}:
            return None
        if offset + 2 > len(image_bytes):
            return None
        length = int.from_bytes(image_bytes[offset : offset + 2], "big")
        if length < 2 or offset + length > len(image_bytes):
            return None
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            if length < 7:
                return None
            height = int.from_bytes(image_bytes[offset + 3 : offset + 5], "big")
            width = int.from_bytes(image_bytes[offset + 5 : offset + 7], "big")
            if width <= 0 or height <= 0:
                return None
            return width, height
        if marker == 0xDA:
            return None
        offset += length
    return None


def _webp_dimensions(image_bytes: bytes) -> tuple[int, int] | None:
    if (
        len(image_bytes) < 20
        or image_bytes[:4] != b"RIFF"
        or image_bytes[8:12] != b"WEBP"
    ):
        return None
    riff_size = int.from_bytes(image_bytes[4:8], "little") + 8
    if riff_size != len(image_bytes):
        return None
    offset = 12
    extended_dimensions: tuple[int, int] | None = None
    while offset + 8 <= len(image_bytes):
        chunk_type = image_bytes[offset : offset + 4]
        chunk_size = int.from_bytes(image_bytes[offset + 4 : offset + 8], "little")
        data_start = offset + 8
        data_end = data_start + chunk_size
        padded_end = data_end + chunk_size % 2
        if padded_end > len(image_bytes):
            return None
        chunk_data = image_bytes[data_start:data_end]
        if chunk_type == b"VP8X":
            extended_dimensions = _extended_webp_dimensions(chunk_data)
            if extended_dimensions is None:
                return None
        elif chunk_type == b"VP8 ":
            dimensions = _lossy_webp_dimensions(chunk_data)
            if dimensions is None:
                return None
            return extended_dimensions or dimensions
        elif chunk_type == b"VP8L":
            dimensions = _lossless_webp_dimensions(chunk_data)
            if dimensions is None:
                return None
            return extended_dimensions or dimensions
        offset = padded_end
    return None


def _extended_webp_dimensions(chunk_data: bytes) -> tuple[int, int] | None:
    if len(chunk_data) != 10:
        return None
    width = 1 + int.from_bytes(chunk_data[4:7], "little")
    height = 1 + int.from_bytes(chunk_data[7:10], "little")
    if width <= 0 or height <= 0:
        return None
    return width, height


def _lossy_webp_dimensions(chunk_data: bytes) -> tuple[int, int] | None:
    if len(chunk_data) < 10 or chunk_data[3:6] != b"\x9d\x01\x2a":
        return None
    width = int.from_bytes(chunk_data[6:8], "little") & 0x3FFF
    height = int.from_bytes(chunk_data[8:10], "little") & 0x3FFF
    if width <= 0 or height <= 0:
        return None
    return width, height


def _lossless_webp_dimensions(chunk_data: bytes) -> tuple[int, int] | None:
    if len(chunk_data) < 5 or chunk_data[0] != 0x2F:
        return None
    width = 1 + (chunk_data[1] | ((chunk_data[2] & 0x3F) << 8))
    height = 1 + (
        ((chunk_data[2] & 0xC0) >> 6)
        | (chunk_data[3] << 2)
        | ((chunk_data[4] & 0x0F) << 10)
    )
    if width <= 0 or height <= 0:
        return None
    return width, height


def _gdk_pixbuf_module() -> Any | None:
    try:
        gi = importlib.import_module("gi")
        require_version = getattr(gi, "require_version", None)
        if callable(require_version):
            require_version("GdkPixbuf", "2.0")
        return importlib.import_module("gi.repository.GdkPixbuf")
    except Exception:
        return None


def _image_bytes_decode_with_gdkpixbuf(
    image_bytes: bytes,
    mime_type: str,
) -> bool | None:
    gdk_pixbuf = _gdk_pixbuf_module()
    if gdk_pixbuf is None:
        return None
    try:
        loader = gdk_pixbuf.PixbufLoader.new_with_mime_type(mime_type)
        loader.write(image_bytes)
        loader.close()
        pixbuf = loader.get_pixbuf()
        return bool(
            pixbuf
            and int(pixbuf.get_width()) > 0
            and int(pixbuf.get_height()) > 0
        )
    except Exception:
        return False


def _is_valid_png(image_bytes: bytes) -> bool:
    offset = 8
    saw_ihdr = False
    saw_idat = False
    saw_plte = False
    after_idat = False
    width = 0
    height = 0
    bit_depth = 0
    color_type = 0
    interlace_method = 0
    idat_chunks: list[bytes] = []
    while offset + 12 <= len(image_bytes):
        length = int.from_bytes(image_bytes[offset : offset + 4], "big")
        chunk_type = image_bytes[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        crc_end = data_end + 4
        if crc_end > len(image_bytes):
            return False
        expected_crc = int.from_bytes(image_bytes[data_end:crc_end], "big")
        actual_crc = zlib.crc32(chunk_type + image_bytes[data_start:data_end])
        if expected_crc != actual_crc & 0xFFFFFFFF:
            return False
        if not saw_ihdr:
            if chunk_type != b"IHDR" or length != 13:
                return False
            width = int.from_bytes(image_bytes[data_start : data_start + 4], "big")
            height = int.from_bytes(image_bytes[data_start + 4 : data_start + 8], "big")
            bit_depth = image_bytes[data_start + 8]
            color_type = image_bytes[data_start + 9]
            compression_method = image_bytes[data_start + 10]
            filter_method = image_bytes[data_start + 11]
            interlace_method = image_bytes[data_start + 12]
            if (
                width <= 0
                or height <= 0
                or not _decoded_dimensions_within_limit((width, height))
                or compression_method != 0
                or filter_method != 0
                or interlace_method not in {0, 1}
                or _png_bits_per_pixel(color_type, bit_depth) is None
            ):
                return False
            saw_ihdr = True
        elif chunk_type == b"IHDR":
            return False
        elif chunk_type not in {b"PLTE", b"IDAT", b"IEND"} and not (
            chunk_type[0] & 0x20
        ):
            return False
        if chunk_type == b"PLTE":
            if saw_idat or saw_plte or color_type in {0, 4}:
                return False
            if length == 0 or length % 3 != 0 or length > 768:
                return False
            saw_plte = True
        if chunk_type == b"IDAT":
            if after_idat or (color_type == 3 and not saw_plte):
                return False
            saw_idat = True
            idat_chunks.append(image_bytes[data_start:data_end])
        elif saw_idat and chunk_type != b"IEND":
            after_idat = True
        if chunk_type == b"IEND":
            return (
                saw_ihdr
                and saw_idat
                and length == 0
                and crc_end == len(image_bytes)
                and _png_idat_decodes(
                    idat_chunks,
                    width=width,
                    height=height,
                    bits_per_pixel=cast(
                        int,
                        _png_bits_per_pixel(color_type, bit_depth),
                    ),
                    interlace_method=interlace_method,
                )
            )
        offset = crc_end
    return False


def _png_bits_per_pixel(color_type: int, bit_depth: int) -> int | None:
    valid_depths = {
        0: {1, 2, 4, 8, 16},
        2: {8, 16},
        3: {1, 2, 4, 8},
        4: {8, 16},
        6: {8, 16},
    }
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    if bit_depth not in valid_depths.get(color_type, set()):
        return None
    return channels[color_type] * bit_depth


def _png_idat_decodes(
    idat_chunks: list[bytes],
    *,
    width: int,
    height: int,
    bits_per_pixel: int,
    interlace_method: int,
) -> bool:
    scanline_layout = _png_scanline_layout(
        width=width,
        height=height,
        bits_per_pixel=bits_per_pixel,
        interlace_method=interlace_method,
    )
    if scanline_layout is None:
        return False
    expected_size, scanline_segments = scanline_layout
    if expected_size > _MAX_UPLOADED_IMAGE_DECODED_BYTES:
        return False
    try:
        decompressor = zlib.decompressobj()
        decoded = decompressor.decompress(
            b"".join(idat_chunks),
            expected_size + 1,
        )
    except zlib.error:
        return False
    if (
        len(decoded) != expected_size
        or decompressor.unconsumed_tail
        or not decompressor.eof
        or decompressor.unused_data
    ):
        return False
    return all(
        decoded[start + row * stride] <= 4
        for start, stride, row_count in scanline_segments
        for row in range(row_count)
    )


def _png_scanline_layout(
    *,
    width: int,
    height: int,
    bits_per_pixel: int,
    interlace_method: int,
) -> tuple[int, tuple[tuple[int, int, int], ...]] | None:
    if interlace_method == 0:
        row_bytes = (width * bits_per_pixel + 7) // 8
        stride = row_bytes + 1
        return height * stride, ((0, stride, height),)
    if interlace_method != 1:
        return None
    passes = (
        (0, 0, 8, 8),
        (4, 0, 8, 8),
        (0, 4, 4, 8),
        (2, 0, 4, 4),
        (0, 2, 2, 4),
        (1, 0, 2, 2),
        (0, 1, 1, 2),
    )
    offset = 0
    scanline_segments: list[tuple[int, int, int]] = []
    for x_start, y_start, x_step, y_step in passes:
        pass_width = (
            0 if width <= x_start else (width - x_start + x_step - 1) // x_step
        )
        pass_height = (
            0 if height <= y_start else (height - y_start + y_step - 1) // y_step
        )
        if pass_width == 0 or pass_height == 0:
            continue
        row_bytes = (pass_width * bits_per_pixel + 7) // 8
        stride = row_bytes + 1
        scanline_segments.append((offset, stride, pass_height))
        offset += pass_height * stride
    return offset, tuple(scanline_segments)


def _write_scaled_thumbnail(
    *,
    image_path: Path,
    thumbnail_path: Path,
) -> bool:
    try:
        gi = importlib.import_module("gi")
        require_version = getattr(gi, "require_version", None)
        if callable(require_version):
            require_version("GdkPixbuf", "2.0")
        gdk_pixbuf: Any = importlib.import_module("gi.repository.GdkPixbuf")
        pixbuf = gdk_pixbuf.Pixbuf.new_from_file_at_scale(
            str(image_path),
            _THUMBNAIL_WIDTH,
            _THUMBNAIL_HEIGHT,
            True,
        )
        pixbuf.savev(str(thumbnail_path), "png", [], [])
        return thumbnail_path.is_file()
    except Exception:
        try:
            write_private_bytes(thumbnail_path, image_path.read_bytes())
            return thumbnail_path.is_file()
        except Exception:
            return False


def _safe_path_segment(value: str) -> str:
    return quote(value, safe="").replace(".", "%2E")


def _assert_within_media_dir(
    *,
    media_dir: Path,
    output_path: Path,
) -> None:
    media_root = media_dir.resolve()
    resolved_output = output_path.resolve()
    if not resolved_output.is_relative_to(media_root):
        raise ValueError("Resolved image path escapes media directory")


def _assert_scenario_starter_reference_path(relative_path: str) -> None:
    path = PurePosixPath(relative_path)
    parts = path.parts
    if (
        path.is_absolute()
        or ".." in parts
        or len(parts) not in {3, 4}
        or parts[0] != "scenario-starters"
        or any(not part for part in parts)
    ):
        raise ValueError("Scenario starter reference image path is invalid")
    if len(parts) == 4 and parts[2] != "thumbnails":
        raise ValueError("Scenario starter reference image path is invalid")


def _elapsed_ms(started_at: float) -> int:
    return int((perf_counter() - started_at) * 1000)
