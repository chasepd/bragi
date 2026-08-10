"""FastAPI app for Bragi Web."""

from __future__ import annotations

import asyncio
import inspect
import ipaddress
import json
import mimetypes
import os
import secrets
import socket
import sqlite3
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter, time
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    NoReturn,
    TypedDict,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)
from urllib.parse import SplitResult, urlsplit
from uuid import UUID, uuid4

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from starlette.background import BackgroundTask
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from bragi_common.media_mime import safe_served_media_mime_type
from bragi_common.story_continuation import (
    STORY_CONTINUATION_DIRECTION,
    STORY_CONTINUATION_SPEAKER_NAME,
)
from bragi_web.auth_throttle import AuthAttemptThrottle
from bragi_web.bragi_adapter import (
    bragi_diagnostics_bindings,
    bragi_settings_bindings,
    lazy_bragi_api_symbol,
    resolve_lazy_symbol,
)
from bragi_web.jobs import (
    ACTIVE_JOB_STATUSES,
    CONTINUITY_READY,
    OPTIONAL_ENRICHMENTS_COMPLETE,
    PUBLIC_JOB_FAILURE_ERROR,
    RESPONSE_COMMITTED,
    TERMINAL_JOB_STATUSES,
    JobHandle,
    JobRecord,
    JobRegistryExclusiveKeyError,
    JobRegistryFullError,
    JobWorker,
    job_event_payload,
    job_summary,
)
from bragi_web.maintenance_diagnostics import maintenance_job_diagnostics
from bragi_web.observability import (
    error_fields,
    observe,
    recent_events,
    sanitize_client_fields,
)
from bragi_web.runtime import (
    WEB_RUNTIME_CHRONICLE_MESSAGE_LIMIT,
    BundlePreviewState,
    WebAppState,
    create_state,
)
from bragi_web.scheduler import WebMaintenanceScheduler
from bragi_web.serialization import to_jsonable

_CHAT_TURN_ACTIVE_DETAIL = "A chat turn is already being processed for this save."
_JOB_CANCELLATION_FAILED_DETAIL = "Job cancellation could not be requested"
_SAVE_ID_REQUIRED_DETAIL = "save_id is required for this save-scoped operation"
_RETIRED_SCENARIO_TYPE = "character_interaction"
_RETIRED_SCENARIO_DETAIL = (
    "The character_interaction scenario type is no longer supported"
)
_ASSET_CACHE_CONTROL = "public, max-age=31536000, immutable"
_MEDIA_CACHE_CONTROL = "private, max-age=31536000, immutable"
_STATIC_CACHE_CONTROL = "public, max-age=86400"
_SPA_CACHE_CONTROL = "no-cache"
_SSE_HEARTBEAT_SECONDS = 15.0
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
_COMPRESSIBLE_STATIC_SUFFIXES = frozenset({".css", ".js"})
_DIAGNOSTIC_CATEGORIES = frozenset(
    {"signals", "jobs", "performance", "scheduler", "events", "save_health"}
)
_DEFAULT_DIAGNOSTICS_LIMIT = 50
_MAX_DIAGNOSTICS_LIMIT = 200
_DEFAULT_PERFORMANCE_WINDOW_SECONDS = 15 * 60
_TERMINAL_JOB_STATUS_FILTERS = {
    "failed": ("failed",),
    "cancelled": ("cancelled",),
    "succeeded": ("succeeded",),
    "terminal": tuple(sorted(TERMINAL_JOB_STATUSES)),
}
_UNUSABLE_FALLBACK_THUMBNAIL_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000a49444154789c636000000200015e027fea00000000"
    "49454e44ae426082"
)

if TYPE_CHECKING:
    from bragi.application.controller import ManualScenarioInput
    from bragi.persistence.models import UserRecord
    from bragi.services.auth_service import AuthService
    from bragi.services.character_registry_service import (
        CharacterKnowledgeAction,
        CharacterRegistryEdits,
        CharacterRegistryRow,
        CharacterRegistryService,
    )
    from bragi.services.content_rating import ContentSafetyPolicy
    from bragi.services.world_data_service import (
        MemoryEdit,
        ScenarioEdit,
        SummaryEdit,
        WorldDataCharacterRow,
        WorldDataEdits,
        WorldDataEntityLinkRow,
        WorldDataLocationRow,
        WorldDataLossConditionRow,
        WorldDataMemoryRow,
        WorldDataSceneRow,
        WorldDataService,
        WorldDataStateRow,
        WorldDataSuggestionGroupRow,
        WorldDataSuggestionRow,
        WorldDataSummaryRow,
        WorldDataThreadRow,
    )
else:
    CharacterRegistryEdits = lazy_bragi_api_symbol("CharacterRegistryEdits")
    CharacterKnowledgeAction = lazy_bragi_api_symbol("CharacterKnowledgeAction")
    CharacterRegistryRow = lazy_bragi_api_symbol("CharacterRegistryRow")
    CharacterRegistryService = lazy_bragi_api_symbol("CharacterRegistryService")
    MemoryEdit = lazy_bragi_api_symbol("MemoryEdit")
    ScenarioEdit = lazy_bragi_api_symbol("ScenarioEdit")
    SummaryEdit = lazy_bragi_api_symbol("SummaryEdit")
    WorldDataCharacterRow = lazy_bragi_api_symbol("WorldDataCharacterRow")
    WorldDataEdits = lazy_bragi_api_symbol("WorldDataEdits")
    WorldDataEntityLinkRow = lazy_bragi_api_symbol("WorldDataEntityLinkRow")
    WorldDataLocationRow = lazy_bragi_api_symbol("WorldDataLocationRow")
    WorldDataLossConditionRow = lazy_bragi_api_symbol("WorldDataLossConditionRow")
    WorldDataMemoryRow = lazy_bragi_api_symbol("WorldDataMemoryRow")
    WorldDataSceneRow = lazy_bragi_api_symbol("WorldDataSceneRow")
    WorldDataService = lazy_bragi_api_symbol("WorldDataService")
    WorldDataStateRow = lazy_bragi_api_symbol("WorldDataStateRow")
    WorldDataSuggestionGroupRow = lazy_bragi_api_symbol("WorldDataSuggestionGroupRow")
    WorldDataSuggestionRow = lazy_bragi_api_symbol("WorldDataSuggestionRow")
    WorldDataSummaryRow = lazy_bragi_api_symbol("WorldDataSummaryRow")
    WorldDataThreadRow = lazy_bragi_api_symbol("WorldDataThreadRow")
    ManualScenarioInput = lazy_bragi_api_symbol("ManualScenarioInput")

_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_BRAGI_API_REQUEST_HEADER = "x-bragi-api-request"
_CROSS_ORIGIN_DETAIL = "Cross-origin state-changing requests are not allowed"
_INVALID_HOST_DETAIL = "Invalid host header"
_AUTH_REQUIRED_DETAIL = "Authentication required"
_DEFAULT_BACKEND_PORT = 8787
_DEFAULT_FRONTEND_PORT = 5173
_WILDCARD_HOSTS = frozenset({"", "*", "0.0.0.0", "::"})
_SESSION_COOKIE_NAME = "bragi_session"
_SESSION_COOKIE_MAX_AGE_SECONDS = 30 * 24 * 60 * 60
_BOOTSTRAP_TOKEN_ENV = "BRAGI_WEB_BOOTSTRAP_TOKEN"
_AUTH_THROTTLED_DETAIL = "Too many authentication attempts; try again later"
MAX_CHAT_BODY_CHARS = 20_000
MAX_LOOK_AROUND_QUERY_CHARS = 4_000
MAX_JSON_REQUEST_BODY_BYTES = 1024 * 1024
_AUTH_USERNAME_MAX_LENGTH = 128
_AUTH_PASSWORD_MAX_LENGTH = 1024
_BOOTSTRAP_SETUP_TOKEN_MAX_LENGTH = 256
_DRAFT_STARTER_GENERATION_MAX_SECTION_COUNT = 64
_DRAFT_STARTER_GENERATION_MAX_EXISTING_STARTERS = 24
_DRAFT_STARTER_GENERATION_MAX_JSON_CHARS = 60_000
_PUBLIC_API_PATHS = frozenset(
    {
        "/api/health",
        "/api/bootstrap/status",
        "/api/bootstrap/admin",
        "/api/auth/login",
        "/api/auth/logout",
        "/api/auth/session",
        "/api/auth/me",
    }
)
BUNDLE_UPLOAD_MAX_BYTES = 2 * 1024 * 1024 * 1024
BUNDLE_UPLOAD_CHUNK_BYTES = 1024 * 1024
CHARACTER_REFERENCE_UPLOAD_MAX_BYTES = 25 * 1024 * 1024
MULTIPART_REQUEST_OVERHEAD_BYTES = 2 * 1024 * 1024
BUNDLE_PREVIEW_TTL_SECONDS = 30 * 60.0
BUNDLE_PREVIEW_MAX_COUNT = 20
BUNDLE_PREVIEW_MAX_RETAINED_BYTES = 2 * 1024 * 1024 * 1024
SCENARIO_BUNDLE_PREVIEW_TTL_SECONDS = BUNDLE_PREVIEW_TTL_SECONDS
SCENARIO_BUNDLE_PREVIEW_MAX_COUNT = BUNDLE_PREVIEW_MAX_COUNT
CHARACTER_BUNDLE_PREVIEW_TTL_SECONDS = BUNDLE_PREVIEW_TTL_SECONDS
CHARACTER_BUNDLE_PREVIEW_MAX_COUNT = BUNDLE_PREVIEW_MAX_COUNT
_CHAT_TURN_CANCELLED_ERROR = "Chat turn cancelled"
_CHAT_JOB_TYPES = frozenset(
    {
        "chat_turn",
        "look_around",
        "chat_regenerate",
        "chat_edit",
        "message_edit",
        "narrator_edit",
        "chat_delete_from_here",
        "chat_fork_from_here",
    }
)
_POST_TURN_PROGRESS_JOB_ORDER = (
    "summary",
    "state",
    "context",
    "time_reconciliation",
    "proactive_text",
    "director",
    "scenario",
    "image",
)
_CHAT_TURN_PROGRESS_JOB_ORDER = (
    "submission",
    "classification",
    "history",
    "input",
    "character_planning",
    "context_selection",
    "prompt",
    "narrator",
    "response_checks",
    "save_narration",
    "action_choices",
)
_REQUEST_STATE: ContextVar[WebAppState | None] = ContextVar(
    "bragi_request_state",
    default=None,
)
_REQUEST_USER: ContextVar[Any | None] = ContextVar(
    "bragi_request_user",
    default=None,
)
_NO_ACCESSIBLE_SAVE_ID = "__bragi_no_accessible_save__"
_CHILD_ALLOWED_SAVE_ACTIONS = frozenset({"read", "chat", "media_generate"})
_NON_ADMIN_CHRONICLE_ACTIONS_BLOCKED = frozenset(
    {
        "inspect-debug-prompt",
        "inspect-provider-payload",
    }
)
_CHILD_CHRONICLE_ACTIONS_BLOCKED = _NON_ADMIN_CHRONICLE_ACTIONS_BLOCKED | frozenset(
    {
        "delete-messages-from-here",
        "edit-and-resubmit-message",
        "edit-narrator-message",
        "fork-from-here",
        "regenerate-message",
        "regenerate-message-with-feedback",
    }
)
_CHILD_CHARACTER_TEXT_ACTIONS_BLOCKED = frozenset(
    {
        "correct-character-text-message",
        "delete-text-messages-from-here",
        "edit-and-resubmit-text-message",
        "edit-text-message",
    }
)


class _SaveEventStreamAuthFilter(TypedDict, total=False):
    owner_user_id: str | None
    include_unowned_global: bool
    include_all_global: bool


@dataclass(frozen=True)
class HostSecurityConfig:
    allowed_hosts: frozenset[str]
    allowed_origin_ports: frozenset[int]
    allowed_origins: frozenset[tuple[str, str, int]]


class AuthCredentialsRequest(BaseModel):
    username: str = Field(default="", max_length=_AUTH_USERNAME_MAX_LENGTH)
    password: str = Field(default="", max_length=_AUTH_PASSWORD_MAX_LENGTH)


class BootstrapAdminRequest(AuthCredentialsRequest):
    setup_token: str = Field(default="", max_length=_BOOTSTRAP_SETUP_TOKEN_MAX_LENGTH)


class AdminCreateUserRequest(AuthCredentialsRequest):
    role: str = "user"


class AdminUpdateUserRequest(BaseModel):
    role: str | None = None
    status: str | None = None
    content_rating: str | None = None


class AdminResetPasswordRequest(BaseModel):
    password: str = Field(default="", max_length=_AUTH_PASSWORD_MAX_LENGTH)


class StartScenarioRequest(BaseModel):
    save_title: str = ""


class RenameSaveRequest(BaseModel):
    title: str = ""


class ScenarioDraftRequest(BaseModel):
    scenario_type: str = "full_roleplay"
    scenario_types: list[str] | None = None
    seed: str = ""
    action_choices_enabled: bool = False
    interaction_mode: str = "roleplay"


class SaveScenarioDraftRequest(BaseModel):
    scenario_type: str = "full_roleplay"
    scenario_types: list[str] | None = None
    sections: dict[str, str] = Field(default_factory=dict)
    character_starters: list[dict[str, Any]] = Field(default_factory=list)
    action_choices_enabled: bool = False
    save_title: str = ""
    source_metadata: dict[str, object] | None = None
    interaction_mode: str = "roleplay"


class ScenarioDraftCharacterStarterGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_type: str = "full_roleplay"
    scenario_types: list[str] | None = None
    sections: dict[str, str] = Field(default_factory=dict)
    character_starters: list[dict[str, Any]] = Field(default_factory=list)
    count: StrictInt | None = None
    custom_description: str = ""
    action_choices_enabled: bool = False
    interaction_mode: str = "roleplay"


class RegenerateScenarioSectionRequest(BaseModel):
    scenario_type: str = "full_roleplay"
    scenario_types: list[str] | None = None
    seed: str = ""
    section_id: str
    sections: dict[str, str] = Field(default_factory=dict)
    action_choices_enabled: bool = False
    interaction_mode: str = "roleplay"


class ChatRequest(BaseModel):
    body: str = Field(default="", max_length=MAX_CHAT_BODY_CHARS)
    speaker_name: str | None = None
    save_id: str | None = None
    client_turn_id: UUID


class LookAroundRequest(BaseModel):
    query: str = Field(default="", max_length=MAX_LOOK_AROUND_QUERY_CHARS)
    save_id: str | None = None


class CharacterTextSendRequest(BaseModel):
    save_id: str | None = None
    character_id: str
    body: str = ""


class CharacterTextThreadSendRequest(BaseModel):
    save_id: str | None = None
    body: str = ""


class CharacterTextGroupThreadCreateRequest(BaseModel):
    save_id: str | None = None
    title: str = ""
    character_ids: list[str] = Field(default_factory=list)


class CharacterTextSpontaneousRequest(BaseModel):
    save_id: str | None = None
    character_id: str


class CharacterTextThreadReadRequest(BaseModel):
    save_id: str | None = None
    through_message_id: str | None = None


class CharacterTextContactUpdateRequest(BaseModel):
    save_id: str | None = None
    player_has_character_number: bool
    character_has_player_number: bool


class CharacterTextEditRequest(BaseModel):
    save_id: str | None = None
    text_message_id: str
    body: str = ""


class CharacterTextDeleteRequest(BaseModel):
    save_id: str | None = None
    text_message_id: str


class TimeskipRequest(BaseModel):
    instruction: str = ""
    save_id: str | None = None
    client_turn_id: UUID


class ContinueStoryRequest(BaseModel):
    save_id: str | None = None
    client_turn_id: UUID


class SaveScopedRequest(BaseModel):
    save_id: str | None = None


class WorldTimeUpdateRequest(SaveScopedRequest):
    day_index: int | None = None
    day_label: str = ""
    phase: str = ""
    clock_minutes: int | None = None
    period_label: str = ""
    in_world_time: str = ""
    time_of_day: str = ""
    day_of_week: str = ""
    world_day_index: int | None = None


class CharacterReferenceRouteRequest(SaveScopedRequest):
    character_id: str | None = None


class CharacterReferenceGenerateRequest(SaveScopedRequest):
    source_message_id: str | None = None
    replace_existing: bool = False


class CharacterReferenceSetRequest(SaveScopedRequest):
    media_asset_id: str


class ScenarioStarterReferenceRemoveRequest(BaseModel):
    starter_id: str | None = None
    starter_name: str = ""


class CharacterImageGenerateRequest(SaveScopedRequest):
    instructions: str = ""


class ContinuationDraftRequest(SaveScopedRequest):
    chapter_start_instructions: str = ""


class MessageRequest(BaseModel):
    message_id: str
    save_id: str | None = None


class RegenerateMessageRequest(MessageRequest):
    regeneration_feedback: str = ""


class EditMessageRequest(MessageRequest):
    body: str = ""


class NarratorEditMessageRequest(EditMessageRequest):
    pass


class DeleteMessagesFromHereRequest(MessageRequest):
    pass


class ForkFromHereRequest(MessageRequest):
    pass


class MediaMessageRequest(MessageRequest):
    pass


class CharacterMediaMessageRequest(MessageRequest):
    character_id: str


class ScenePresenceRequest(SaveScopedRequest):
    character_ids: list[str] = Field(default_factory=list)


class InitialImageRequest(MessageRequest):
    pass


class ProviderKeyRequest(BaseModel):
    provider: str
    api_key: str = ""


class ModelPreferenceRequest(BaseModel):
    task: str
    provider: str
    model_id: str
    save_id: str | None = None


class ModelThinkingPreferenceRequest(ModelPreferenceRequest):
    level: str


class ModelRoutingProfileRequest(BaseModel):
    name: str
    profile_id: str | None = None


class ScopedSettingRequest(BaseModel):
    key: str
    value: object = None
    save_id: str | None = None


LocalSettingRequest = ScopedSettingRequest


class CustomInstructionsRequest(SaveScopedRequest):
    custom_instructions: str = ""


class ContextCleanupRequest(SaveScopedRequest):
    pass


class SummaryBackfillRequest(SaveScopedRequest):
    apply_recommended_windows: bool = False


class GuidedContextCleanupRequest(SaveScopedRequest):
    instruction: str = ""


class DatingSimMaintenanceRequest(SaveScopedRequest):
    apply: bool = False
    repair_ids: list[str] = Field(default_factory=list)
    confirm_save_id: str = ""
    include_evidence_text: bool = False


class WorldDataApplyRequest(BaseModel):
    active_save_id: str | None = None
    edits: dict[str, Any] = Field(default_factory=dict)


class ScenarioDefinitionApplyRequest(BaseModel):
    edit: dict[str, Any] = Field(default_factory=dict)


class CharacterRegistryApplyRequest(BaseModel):
    active_save_id: str | None = None
    edits: dict[str, Any] = Field(default_factory=dict)
    auto_enhance_created_agency: bool = False


class CharacterFieldEnhanceRequest(BaseModel):
    active_save_id: str | None = None
    field_name: str
    character: dict[str, Any] = Field(default_factory=dict)


class CharacterKnowledgeApplyRequest(BaseModel):
    active_save_id: str | None = None
    actions: list[dict[str, Any]] = Field(default_factory=list)


class CharacterBundleImportRequest(BaseModel):
    active_save_id: str | None = None
    name: str | None = None


class RegenerateMediaRequest(BaseModel):
    save_id: str | None = None
    prompt: str | None = None


class AnimateMediaRequest(RegenerateMediaRequest):
    motion_prompt: str = ""


class ClientLogRequest(BaseModel):
    level: str = "info"
    event: str
    fields: object = Field(default_factory=dict)


class _BundleUploadTooLarge(Exception):
    def __init__(self, max_bytes: int) -> None:
        super().__init__(f"Bundle upload exceeds {max_bytes} bytes")
        self.max_bytes = max_bytes


class _CharacterReferenceUploadTooLarge(Exception):
    def __init__(self, max_bytes: int) -> None:
        super().__init__(f"Uploaded image exceeded {max_bytes} bytes")
        self.max_bytes = max_bytes


class _JsonRequestBodyTooLarge(Exception):
    pass


class _JsonRequestBodyLimitMiddleware:
    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_limit = _request_body_limit_bytes(
            scope,
            default_limit=self.max_body_bytes,
        )
        headers = {key.lower(): value for key, value in scope.get("headers", ())}
        raw_content_length = headers.get(b"content-length")
        if raw_content_length is not None:
            try:
                content_length = int(raw_content_length)
            except ValueError:
                content_length = 0
            if content_length > request_limit:
                await self._reject(scope, receive, send)
                return

        received_bytes = 0

        async def limited_receive() -> Message:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > request_limit:
                    raise _JsonRequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _JsonRequestBodyTooLarge:
            await self._reject(scope, receive, send)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            {"detail": "Request body too large"},
            status_code=413,
        )
        await response(scope, receive, send)


def _request_body_limit_bytes(scope: Scope, *, default_limit: int) -> int:
    headers = {key.lower(): value for key, value in scope.get("headers", ())}
    raw_content_type = headers.get(b"content-type", b"")
    if not isinstance(raw_content_type, bytes):
        return default_limit
    content_type = raw_content_type.split(b";", 1)[0].strip().lower()
    if content_type != b"multipart/form-data":
        return default_limit
    path = str(scope.get("path", ""))
    bundle_paths = {
        "/api/bundles/preview",
        "/api/character-bundles/preview",
        "/api/scenario-bundles/preview",
    }
    if path in bundle_paths:
        return BUNDLE_UPLOAD_MAX_BYTES + MULTIPART_REQUEST_OVERHEAD_BYTES
    image_paths = {
        "/api/character-texts/send-image",
        "/api/media/character-reference/upload",
    }
    if (
        path in image_paths
        or (
            path.startswith("/api/character-texts/threads/")
            and path.endswith("/send-image")
        )
        or (
            path.startswith("/api/scenarios/")
            and path.endswith("/character-starters/reference-image/upload")
        )
        or (
            path.startswith("/api/characters/")
            and path.endswith("/reference-image/upload")
        )
    ):
        return (
            CHARACTER_REFERENCE_UPLOAD_MAX_BYTES
            + MULTIPART_REQUEST_OVERHEAD_BYTES
        )
    return default_limit


def create_app(state: WebAppState | None = None) -> FastAPI:
    provided_state = state is not None
    app_state = state or create_state()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        scheduler: WebMaintenanceScheduler | None = None
        if _web_maintenance_scheduler_enabled(provided_state=provided_state):
            scheduler = WebMaintenanceScheduler(app_state)
            scheduler.start()
        try:
            yield
        finally:
            if scheduler is not None:
                await scheduler.stop()
            if not provided_state:
                close = getattr(app_state, "close", None)
                if callable(close):
                    close()

    app = FastAPI(title="Bragi Web", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        _JsonRequestBodyLimitMiddleware,
        max_body_bytes=MAX_JSON_REQUEST_BODY_BYTES,
    )
    app.state.bragi = app_state
    app.dependency_overrides[state_dependency] = lambda: app.state.bragi
    host_security = _host_security_config()

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next: Any) -> Any:
        with _repository_scope_for_state(app_state):
            request_id = uuid4().hex
            started = perf_counter()
            status_code: int | None = None
            response_class: str | None = None
            state_token = _REQUEST_STATE.set(app_state)
            user_token = _REQUEST_USER.set(None)
            from bragi.services.job_diagnostics import (
                reset_job_request_context,
                set_job_request_context,
            )

            request_context_token = set_job_request_context(
                {
                    "request_id": request_id,
                }
            )
            try:
                if _reject_untrusted_host(request, host_security):
                    status_code = 400
                    response_class = "JSONResponse"
                    return JSONResponse(
                        {"detail": _INVALID_HOST_DETAIL},
                        status_code=status_code,
                    )
                if _reject_cross_origin_write(request, host_security):
                    status_code = 403
                    response_class = "JSONResponse"
                    return JSONResponse(
                        {"detail": _CROSS_ORIGIN_DETAIL},
                        status_code=status_code,
                    )
                if _requires_authenticated_session(request, app_state):
                    user = _load_request_user(request, app_state)
                    if user is None:
                        status_code = 401
                        response_class = "JSONResponse"
                        return JSONResponse(
                            {"detail": _AUTH_REQUIRED_DETAIL},
                            status_code=status_code,
                        )
                    request.state.bragi_user = user
                    _REQUEST_USER.set(user)
                response = await call_next(request)
                status_code = response.status_code
                response_class = type(response).__name__
                return response
            except Exception as exc:
                status_code = 500
                observe(
                    "web.request.failed",
                    level="error",
                    request_id=request_id,
                    method=request.method,
                    route=_route_path(request),
                    status_code=status_code,
                    duration_ms=round((perf_counter() - started) * 1000, 2),
                    **error_fields(exc),
                )
                raise
            finally:
                _REQUEST_USER.reset(user_token)
                _REQUEST_STATE.reset(state_token)
                reset_job_request_context(request_context_token)
                if status_code is not None and status_code < 500:
                    observe(
                        "web.request.completed",
                        level="info" if status_code < 400 else "error",
                        request_id=request_id,
                        method=request.method,
                        route=_route_path(request),
                        status_code=status_code,
                        duration_ms=round((perf_counter() - started) * 1000, 2),
                        response_class=response_class,
                    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/bootstrap/status")
    def bootstrap_status(request: Request, state: StateDep) -> dict[str, bool]:
        return _bootstrap_status_payload(request, state)

    @app.post("/api/bootstrap/admin")
    def bootstrap_admin(
        payload: BootstrapAdminRequest,
        request: Request,
        state: StateDep,
    ) -> JSONResponse:
        if _active_admin_exists(state):
            raise HTTPException(
                status_code=409,
                detail="First admin has already been created",
            )
        setup_token_throttle_key = _client_auth_attempt_key(
            "bootstrap_setup_token",
            request,
        )
        _raise_if_auth_throttled(state, setup_token_throttle_key)
        try:
            _require_bootstrap_setup_token(request, payload.setup_token)
        except HTTPException:
            _record_auth_failure(state, setup_token_throttle_key)
            raise
        throttle_key = _auth_attempt_key("bootstrap", request, payload.username)
        _raise_if_auth_throttled(state, throttle_key)
        service = _auth_service(state)
        from bragi.services.auth_service import FirstAdminAlreadyExistsError

        try:
            login = service.bootstrap_first_admin(
                username=payload.username,
                password=payload.password,
            )
        except FirstAdminAlreadyExistsError as exc:
            raise HTTPException(
                status_code=409,
                detail="First admin has already been created",
            ) from exc
        except ValueError as exc:
            _record_auth_failure(state, throttle_key)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _record_auth_success(state, setup_token_throttle_key)
        _record_auth_success(state, throttle_key)
        response = JSONResponse({"user": _user_json(login.user)})
        _set_session_cookie(response, login.token, request)
        return response

    @app.post("/api/auth/login")
    def auth_login(
        payload: AuthCredentialsRequest,
        request: Request,
        state: StateDep,
    ) -> JSONResponse:
        throttle_key = _auth_attempt_key("login", request, payload.username)
        _raise_if_auth_throttled(state, throttle_key)
        login = _auth_service(state).login(payload.username, payload.password)
        if login is None:
            _record_auth_failure(state, throttle_key)
            raise HTTPException(status_code=401, detail=_AUTH_REQUIRED_DETAIL)
        _record_auth_success(state, throttle_key)
        response = JSONResponse({"user": _user_json(login.user)})
        _set_session_cookie(response, login.token, request)
        return response

    @app.post("/api/auth/logout")
    def auth_logout(request: Request, state: StateDep) -> JSONResponse:
        token = request.cookies.get(_SESSION_COOKIE_NAME)
        if token:
            _auth_service(state).revoke_session(token)
        response = JSONResponse({"ok": True})
        _clear_session_cookie(response)
        return response

    @app.get("/api/auth/me")
    def auth_me(request: Request, state: StateDep) -> dict[str, Any]:
        user = _load_request_user(request, state)
        if user is None:
            raise HTTPException(status_code=401, detail=_AUTH_REQUIRED_DETAIL)
        return {"user": _user_json(user)}

    @app.get("/api/auth/session")
    def auth_session(request: Request, state: StateDep) -> dict[str, Any]:
        user = _load_request_user(request, state)
        return {
            "bootstrap": _bootstrap_status_payload(request, state),
            "user": _user_json(user) if user is not None else None,
        }

    @app.get("/api/admin/users")
    def admin_list_users(state: StateDep) -> dict[str, Any]:
        _require_admin_user()
        with state.lock:
            return {
                "users": [
                    _admin_user_json(user, state.repositories)
                    for user in _auth_service(state).list_users()
                ]
            }

    @app.post("/api/admin/users")
    def admin_create_user(
        payload: AdminCreateUserRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        _require_admin_user()
        try:
            with state.lock:
                user = _auth_service(state).create_user(
                    username=payload.username,
                    password=payload.password,
                    role=payload.role,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"user": _admin_user_json(user, state.repositories)}

    @app.patch("/api/admin/users/{user_id}")
    def admin_update_user(
        user_id: str,
        payload: AdminUpdateUserRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        current_user = _require_admin_user()
        if (
            payload.role is None
            and payload.status is None
            and payload.content_rating is None
        ):
            raise HTTPException(status_code=400, detail="No user updates provided")
        try:
            with state.lock:
                existing_user = state.repositories.get_user(user_id)
                if existing_user is None:
                    from bragi.services.auth_service import UnknownUserError

                    raise UnknownUserError(f"Unknown user id: {user_id}")
                if payload.content_rating is not None:
                    from bragi.services.content_rating import (
                        CHILD_ADMIN_CONTENT_RATING_OPTIONS,
                        sanitize_content_rating,
                    )

                    normalized_rating = sanitize_content_rating(
                        payload.content_rating,
                        default="",
                    )
                    if not normalized_rating:
                        raise ValueError("Unsupported content rating")
                    target_role = payload.role or existing_user.role
                    if (
                        target_role == "child"
                        and normalized_rating
                        not in CHILD_ADMIN_CONTENT_RATING_OPTIONS
                    ):
                        raise ValueError(
                            "Child account content rating cannot exceed PG-13"
                        )
                user = existing_user
                if payload.role is not None or payload.status is not None:
                    user = _auth_service(state).update_user(
                        user_id,
                        role=payload.role,
                        status=payload.status,
                        actor_user_id=(
                            current_user.id if current_user is not None else None
                        ),
                    )
                if (
                    payload.role == "child"
                    and existing_user.role != "child"
                    and payload.content_rating is None
                ):
                    from bragi.services.content_rating import (
                        DEFAULT_CHILD_CONTENT_RATING,
                        set_user_content_rating,
                    )

                    set_user_content_rating(
                        state.repositories,
                        user_id=user_id,
                        rating=DEFAULT_CHILD_CONTENT_RATING,
                        admin_grant=True,
                    )
                if payload.content_rating is not None:
                    from bragi.services.content_rating import set_user_content_rating

                    set_user_content_rating(
                        state.repositories,
                        user_id=user_id,
                        rating=payload.content_rating,
                        admin_grant=True,
                    )
        except Exception as exc:
            _raise_user_management_error(exc)
        return {"user": _admin_user_json(user, state.repositories)}

    @app.post("/api/admin/users/{user_id}/password")
    def admin_reset_user_password(
        user_id: str,
        payload: AdminResetPasswordRequest,
        request: Request,
        state: StateDep,
    ) -> dict[str, Any]:
        current_user = _require_admin_user()
        keep_token = (
            request.cookies.get(_SESSION_COOKIE_NAME)
            if current_user is not None and current_user.id == user_id
            else None
        )
        try:
            with state.lock:
                user = _auth_service(state).reset_user_password(
                    user_id,
                    payload.password,
                    keep_session_token=keep_token,
                )
        except Exception as exc:
            _raise_user_management_error(exc)
        return {"user": _admin_user_json(user, state.repositories)}

    @app.post("/api/admin/dating-sim-maintenance")
    async def admin_dating_sim_maintenance(
        payload: DatingSimMaintenanceRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        _require_admin_user()
        async with state.lock.async_access():
            save_id = _require_save_id(payload.save_id)
            if payload.apply:
                _raise_unless_save_action_allowed(state, save_id, "mutate")
            else:
                _raise_unknown_save_if_possible(state, save_id)

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": "Inspecting dating routes"})
            from bragi.services.dating_sim_maintenance_service import (
                DatingSimMaintenanceService,
            )

            service = DatingSimMaintenanceService(state.repositories)
            try:
                if payload.apply:
                    report = service.apply_repairs(
                        save_id,
                        repair_ids=payload.repair_ids,
                        confirm_save_id=payload.confirm_save_id,
                        include_evidence_text=payload.include_evidence_text,
                    )
                else:
                    report = service.inspect_save(
                        save_id,
                        include_evidence_text=payload.include_evidence_text,
                    )
            except ValueError as exc:
                raise RuntimeError(str(exc)) from exc
            return report.to_result()

        return await _create_job_summary(
            state,
            "dating_sim_maintenance",
            worker,
            save_id=save_id,
            operation_queue_key=save_id,
        )

    @app.get("/api/runtime")
    def runtime_model(state: StateDep, save_id: str | None = None) -> dict[str, Any]:
        with state.lock:
            if save_id is not None:
                _raise_if_save_retired(state, save_id)
            return _runtime_json_dict(
                state,
                _build_runtime_model_for_save(state, save_id),
            )

    @app.get("/api/runtime/shell")
    def runtime_shell_model(
        state: StateDep,
        save_id: str | None = None,
    ) -> dict[str, Any]:
        with state.lock:
            if save_id is not None:
                _raise_if_save_retired(state, save_id)
            return _runtime_json_dict(
                state,
                _build_runtime_shell_model_for_save(state, save_id),
            )

    @app.post("/api/runtime/world-time")
    def update_runtime_world_time(
        payload: WorldTimeUpdateRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        with state.lock:
            save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, save_id, "chat")
            _update_world_time_snapshot(state, save_id=save_id, payload=payload)
            model = _runtime_json_dict(
                state,
                _build_runtime_model_for_save(state, save_id),
            )
        _publish_save_event(
            state,
            save_id,
            "runtime_changed",
            {"reason": "world_time_corrected"},
        )
        return model

    @app.post("/api/world-data/time-loop/baseline")
    def capture_time_loop_baseline(
        payload: SaveScopedRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        from bragi.services.time_loop_time_policy import TimeLoopTimePolicy

        with state.lock:
            save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, save_id, "mutate")
            snapshot = state.repositories.get_scene_snapshot(save_id)
            if snapshot is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Set the current world time before capturing a loop baseline"
                    ),
                )
            policy = TimeLoopTimePolicy(state.repositories, save_id=save_id)
            try:
                state.repositories.begin_transaction()
                policy.capture_baseline(snapshot)
            except ValueError as exc:
                state.repositories.rollback_transaction()
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception:
                state.repositories.rollback_transaction()
                raise
            else:
                state.repositories.commit_transaction()
            model = _json_dict(_build_world_data_model_for_save(state, save_id))
        _publish_save_event(
            state,
            save_id,
            "runtime_changed",
            {"reason": "loop_baseline_captured"},
        )
        return model

    @app.post("/api/world-data/time-loop/reset")
    def reset_time_loop(
        payload: SaveScopedRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        from bragi.services.time_loop_time_policy import (
            TimeLoopTimePolicy,
            write_scene_snapshot,
        )

        with state.lock:
            save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, save_id, "mutate")
            snapshot = state.repositories.get_scene_snapshot(save_id)
            if snapshot is None:
                raise HTTPException(
                    status_code=400,
                    detail="No world time is available",
                )
            policy = TimeLoopTimePolicy(state.repositories, save_id=save_id)
            try:
                state.repositories.begin_transaction()
                result = policy.reset(snapshot)
                write_scene_snapshot(state.repositories, result.snapshot)
            except ValueError as exc:
                state.repositories.rollback_transaction()
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception:
                state.repositories.rollback_transaction()
                raise
            else:
                state.repositories.commit_transaction()
            model = _json_dict(_build_world_data_model_for_save(state, save_id))
        _publish_save_event(state, save_id, "runtime_changed", {"reason": "loop_reset"})
        return model

    @app.get("/api/saves")
    def list_saves(state: StateDep, save_id: str | None = None) -> dict[str, Any]:
        with state.lock:
            active_save_id = _resolve_runtime_save_id(state, save_id)
            return {
                "saves": _save_list_json_for_request(
                    state,
                    active_save_id=active_save_id,
                )
            }

    @app.get("/api/saves/{save_id}/media")
    def save_media_model(save_id: str, state: StateDep) -> dict[str, Any]:
        with state.lock:
            _raise_unless_save_action_allowed(state, save_id, "read")
            from bragi.application.media import build_media_model

            content_safety = _content_safety_policy_for_request(state)

            return cast(
                dict[str, Any],
                to_jsonable(
                    build_media_model(
                        repositories=state.repositories,
                        save_id=save_id,
                        providers=state.providers,
                        media_dir=state.paths.media_dir,
                        allowed_rating=content_safety.rating,
                    )
                ),
            )

    @app.get("/api/saves/{save_id}/chronicle")
    def save_chronicle_page(
        save_id: str,
        state: StateDep,
        before_message_id: str | None = None,
        limit: int = WEB_RUNTIME_CHRONICLE_MESSAGE_LIMIT,
    ) -> dict[str, Any]:
        with state.lock:
            _raise_unless_save_action_allowed(state, save_id, "read")
            page_limit = _bounded_chronicle_page_limit(limit)
            try:
                model = state.runtime.build_chronicle_page_model(
                    save_id=save_id,
                    before_message_id=before_message_id,
                    limit=page_limit,
                )
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return _chronicle_json_dict(state, model)

    @app.get("/api/saves/{save_id}/engine-health")
    def save_engine_health(save_id: str, state: StateDep) -> dict[str, Any]:
        with state.lock:
            _raise_unless_save_diagnostics_allowed(state, save_id)
            return cast(
                dict[str, Any],
                to_jsonable(
                    bragi_diagnostics_bindings().EngineHealthService(
                        state.repositories
                    ).snapshot(save_id)
                ),
            )

    @app.get("/api/chat-history")
    def chat_history(
        state: StateDep,
        filter: str = "all",
        save_id: str | None = None,
        before_message_id: str | None = None,
        limit: int = WEB_RUNTIME_CHRONICLE_MESSAGE_LIMIT,
    ) -> dict[str, Any]:
        with state.lock:
            page_limit = _bounded_chronicle_page_limit(limit)
            try:
                response_payload = _json_dict(
                    _build_chat_history_model_for_save(
                        state,
                        selected_filter=filter,
                        save_id=save_id,
                        before_message_id=before_message_id,
                        limit=page_limit,
                    )
                )
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            _scrub_response_payload_for_request(state, response_payload)
            return response_payload

    @app.get("/api/chat/submission-status")
    def chat_submission_status(
        state: StateDep,
        save_id: str | None = None,
    ) -> dict[str, Any]:
        with state.lock:
            resolved_save_id = _resolve_chat_save_id(state, save_id)
        return _chat_submission_status(state, resolved_save_id)

    @app.get("/api/character-texts")
    def character_texts(
        state: StateDep,
        save_id: str | None = None,
    ) -> dict[str, Any]:
        with state.lock:
            resolved_save_id = _require_save_id(save_id)
            _raise_unless_save_action_allowed(state, resolved_save_id, "read")
            from bragi.services.character_text_service import CharacterTextService

            service = CharacterTextService(
                repositories=state.repositories,
                providers=state.providers,
            )
            payload = _json_dict(service.build_model(resolved_save_id))
            _scrub_response_payload_for_request(
                state,
                payload,
                current_user_role=_current_request_role(state),
            )
            return payload

    @app.get("/api/character-texts/threads/{thread_id}")
    def character_text_thread(
        thread_id: str,
        state: StateDep,
        save_id: str | None = None,
    ) -> dict[str, Any]:
        with state.lock:
            resolved_save_id = _require_save_id(save_id)
            _raise_unless_save_action_allowed(state, resolved_save_id, "read")
            from bragi.services.character_text_service import CharacterTextService

            service = CharacterTextService(
                repositories=state.repositories,
                providers=state.providers,
            )
            try:
                payload = _json_dict(
                    service.get_thread_model(
                        save_id=resolved_save_id,
                        thread_id=thread_id,
                    )
                )
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            _scrub_response_payload_for_request(
                state,
                payload,
                current_user_role=_current_request_role(state),
            )
            return payload

    @app.post("/api/character-texts/groups")
    def create_character_text_group_thread(
        payload: CharacterTextGroupThreadCreateRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        with state.lock:
            resolved_save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, resolved_save_id, "chat")
            from bragi.services.character_text_service import CharacterTextService

            service = CharacterTextService(
                repositories=state.repositories,
                providers=state.providers,
            )
            try:
                thread = service.create_group_thread(
                    save_id=resolved_save_id,
                    title=payload.title,
                    character_ids=tuple(payload.character_ids),
                )
            except ValueError as exc:
                detail = str(exc)
                status_code = 400
                if detail == "Player does not have every character's number":
                    status_code = 403
                if detail.startswith("Unknown textable character id"):
                    status_code = 404
                raise HTTPException(status_code=status_code, detail=detail) from exc
            response_payload = _json_dict(
                {
                    "save_id": resolved_save_id,
                    "thread": thread,
                }
            )
            _scrub_response_payload_for_request(
                state,
                response_payload,
                current_user_role=_current_request_role(state),
            )
        _publish_save_event(
            state,
            resolved_save_id,
            "character_texts_changed",
            response_payload,
        )
        return response_payload

    @app.post("/api/character-texts/threads/{thread_id}/read")
    def mark_character_text_thread_read(
        thread_id: str,
        payload: CharacterTextThreadReadRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        with state.lock:
            resolved_save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, resolved_save_id, "read")
            from bragi.services.character_text_service import CharacterTextService

            service = CharacterTextService(
                repositories=state.repositories,
                providers=state.providers,
            )
            try:
                result = service.mark_thread_read(
                    save_id=resolved_save_id,
                    thread_id=thread_id,
                    through_message_id=payload.through_message_id,
                )
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            response_payload = _json_dict(result)
            _scrub_response_payload_for_request(
                state,
                response_payload,
                current_user_role=_current_request_role(state),
            )
        if result.updated_message_ids:
            _publish_save_event(
                state,
                resolved_save_id,
                "character_texts_changed",
                response_payload,
            )
        return response_payload

    @app.post("/api/character-texts/threads/{thread_id}/send")
    async def send_character_text_thread(
        thread_id: str,
        payload: CharacterTextThreadSendRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            submitted_save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, submitted_save_id, "chat")
            actor_user_id = _owner_user_id_for_request(state)
            from bragi.services.character_text_service import CharacterTextService

            service = CharacterTextService(
                repositories=state.repositories,
                providers=state.providers,
            )
        try:
            content_rating = await service.classify_player_text(
                save_id=submitted_save_id,
                body=payload.body,
                current_user_id=actor_user_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        async with state.lock.async_access():
            try:
                queued = service.queue_thread_text_send(
                    save_id=submitted_save_id,
                    thread_id=thread_id,
                    body=payload.body,
                    content_rating=content_rating,
                )
            except ValueError as exc:
                detail = str(exc)
                status_code = 409 if "already pending" in detail else 400
                if detail == "Player does not have every character's number":
                    status_code = 403
                if detail.startswith("Unknown character text thread id"):
                    status_code = 404
                raise HTTPException(status_code=status_code, detail=detail) from exc
            queued_payload = _json_dict(queued)
        _publish_save_event(
            state,
            submitted_save_id,
            "character_texts_changed",
            queued_payload,
        )
        job_marked = asyncio.Event()

        async def worker(handle: JobHandle) -> Any:
            await job_marked.wait()
            await handle.event("progress", {"label": "Sending group text"})
            retry_callback, flush_retry_progress = _retry_progress_callback(
                handle,
                task_label="character text",
            )

            def delivery_retry_callback(progress: object) -> None:
                retry_callback(progress)
                _publish_save_event(
                    state,
                    submitted_save_id,
                    "character_texts_changed",
                    {"message_id": queued.player_message.id, "status": "retrying"},
                )

            from bragi.services.character_text_service import CharacterTextService
            from bragi.services.media_service import MediaService

            service = CharacterTextService(
                repositories=state.repositories,
                providers=state.providers,
                media_service=MediaService(
                    repositories=state.repositories,
                    providers=state.providers,
                    media_dir=state.paths.media_dir,
                ),
                prompt_inspection_store=_prompt_inspection_store_if_enabled(state),
            )
            try:
                result = await service.complete_queued_thread_text_send(
                    save_id=submitted_save_id,
                    player_message_id=queued.player_message.id,
                    retry_progress_callback=delivery_retry_callback,
                    current_user_id=actor_user_id,
                )
                await flush_retry_progress()
            except ValueError as exc:
                _publish_save_event(
                    state,
                    submitted_save_id,
                    "character_texts_changed",
                    {"message_id": queued.player_message.id, "status": "failed"},
                )
                raise RuntimeError(str(exc)) from exc
            except Exception:
                _publish_save_event(
                    state,
                    submitted_save_id,
                    "character_texts_changed",
                    {"message_id": queued.player_message.id, "status": "failed"},
                )
                raise
            _publish_save_event(
                state,
                submitted_save_id,
                "character_texts_changed",
                result,
            )
            return result

        text_queue_key = _character_text_thread_job_key(queued.thread.id)
        try:
            created = await state.jobs.create(
                "character_text_send",
                worker,
                save_id=submitted_save_id,
                creator_user_id=_owner_user_id_for_request(state),
                exclusive_key=text_queue_key,
                operation_queue_key=text_queue_key,
            )
        except JobRegistryExclusiveKeyError as exc:
            async with state.lock.async_access():
                service = CharacterTextService(
                    repositories=state.repositories,
                    providers=state.providers,
                )
                service.repositories.update_character_text_delivery(
                    save_id=submitted_save_id,
                    message_id=queued.player_message.id,
                    status="failed",
                    error="Text delivery could not start",
                )
            _publish_save_event(
                state,
                submitted_save_id,
                "character_texts_changed",
                {"message_id": queued.player_message.id, "status": "failed"},
            )
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except JobRegistryFullError as exc:
            async with state.lock.async_access():
                service = CharacterTextService(
                    repositories=state.repositories,
                    providers=state.providers,
                )
                service.repositories.update_character_text_delivery(
                    save_id=submitted_save_id,
                    message_id=queued.player_message.id,
                    status="failed",
                    error="Text delivery could not start",
                )
            _publish_save_event(
                state,
                submitted_save_id,
                "character_texts_changed",
                {"message_id": queued.player_message.id, "status": "failed"},
            )
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        async with state.lock.async_access():
            service = CharacterTextService(
                repositories=state.repositories,
                providers=state.providers,
            )
            service.mark_text_send_job(
                save_id=submitted_save_id,
                player_message_id=queued.player_message.id,
                job_id=created.id,
            )
        job_marked.set()
        _publish_save_event(
            state,
            submitted_save_id,
            "character_texts_changed",
            {"message_id": queued.player_message.id, "job_id": created.id},
        )
        return _job_summary_for_request(state, created)

    async def _start_queued_character_text_send_job(
        *,
        state: WebAppState,
        save_id: str,
        queued: Any,
        complete_thread: bool,
        progress_label: str,
        uploaded_photo_bytes: bytes | None = None,
        uploaded_photo_filename: str | None = None,
        current_user_id: str | None = None,
    ) -> dict[str, Any]:
        job_marked = asyncio.Event()

        async def worker(handle: JobHandle) -> Any:
            await job_marked.wait()
            await handle.event("progress", {"label": progress_label})
            retry_callback, flush_retry_progress = _retry_progress_callback(
                handle,
                task_label="character text",
            )

            def delivery_retry_callback(progress: object) -> None:
                retry_callback(progress)
                _publish_save_event(
                    state,
                    save_id,
                    "character_texts_changed",
                    {"message_id": queued.player_message.id, "status": "retrying"},
                )

            from bragi.services.character_text_service import CharacterTextService
            from bragi.services.media_service import MediaService

            service = CharacterTextService(
                repositories=state.repositories,
                providers=state.providers,
                media_service=MediaService(
                    repositories=state.repositories,
                    providers=state.providers,
                    media_dir=state.paths.media_dir,
                ),
                prompt_inspection_store=_prompt_inspection_store_if_enabled(state),
            )
            try:
                result: Any
                if complete_thread:
                    result = await service.complete_queued_thread_text_send(
                        save_id=save_id,
                        player_message_id=queued.player_message.id,
                        uploaded_photo_bytes=uploaded_photo_bytes,
                        uploaded_photo_filename=uploaded_photo_filename,
                        retry_progress_callback=delivery_retry_callback,
                        current_user_id=current_user_id,
                    )
                else:
                    result = await service.complete_queued_text_send(
                        save_id=save_id,
                        player_message_id=queued.player_message.id,
                        uploaded_photo_bytes=uploaded_photo_bytes,
                        uploaded_photo_filename=uploaded_photo_filename,
                        retry_progress_callback=delivery_retry_callback,
                        current_user_id=current_user_id,
                    )
                await flush_retry_progress()
            except ValueError as exc:
                _publish_save_event(
                    state,
                    save_id,
                    "character_texts_changed",
                    {"message_id": queued.player_message.id, "status": "failed"},
                )
                raise RuntimeError(str(exc)) from exc
            except Exception:
                _publish_save_event(
                    state,
                    save_id,
                    "character_texts_changed",
                    {"message_id": queued.player_message.id, "status": "failed"},
                )
                raise
            _publish_save_event(
                state,
                save_id,
                "character_texts_changed",
                result,
            )
            return result

        text_queue_key = _character_text_thread_job_key(queued.thread.id)
        try:
            created = await state.jobs.create(
                "character_text_send",
                worker,
                save_id=save_id,
                creator_user_id=_owner_user_id_for_request(state),
                exclusive_key=text_queue_key,
                operation_queue_key=text_queue_key,
            )
        except JobRegistryExclusiveKeyError as exc:
            async with state.lock.async_access():
                service = _character_text_service_for_delivery_failure(state)
                service.repositories.update_character_text_delivery(
                    save_id=save_id,
                    message_id=queued.player_message.id,
                    status="failed",
                    error="Text delivery could not start",
                )
            _publish_save_event(
                state,
                save_id,
                "character_texts_changed",
                {"message_id": queued.player_message.id, "status": "failed"},
            )
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except JobRegistryFullError as exc:
            async with state.lock.async_access():
                service = _character_text_service_for_delivery_failure(state)
                service.repositories.update_character_text_delivery(
                    save_id=save_id,
                    message_id=queued.player_message.id,
                    status="failed",
                    error="Text delivery could not start",
                )
            _publish_save_event(
                state,
                save_id,
                "character_texts_changed",
                {"message_id": queued.player_message.id, "status": "failed"},
            )
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        async with state.lock.async_access():
            service = _character_text_service_for_delivery_failure(state)
            service.mark_text_send_job(
                save_id=save_id,
                player_message_id=queued.player_message.id,
                job_id=created.id,
            )
        job_marked.set()
        _publish_save_event(
            state,
            save_id,
            "character_texts_changed",
            {"message_id": queued.player_message.id, "job_id": created.id},
        )
        return _job_summary_for_request(state, created)

    def _character_text_service_for_delivery_failure(
        state: WebAppState,
    ) -> Any:
        from bragi.services.character_text_service import CharacterTextService

        return CharacterTextService(
            repositories=state.repositories,
            providers=state.providers,
        )

    @app.post("/api/character-texts/threads/{thread_id}/send-image")
    async def send_character_text_thread_with_image(
        thread_id: str,
        state: StateDep,
        file: Annotated[UploadFile | None, File()] = None,
        save_id: Annotated[str | None, Form()] = None,
        body: Annotated[str, Form()] = "",
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            submitted_save_id = _require_save_id(save_id)
            _raise_unless_save_action_allowed(state, submitted_save_id, "chat")
            actor_user_id = _owner_user_id_for_request(state)
            if file is not None:
                _raise_unless_save_action_allowed(
                    state,
                    submitted_save_id,
                    "media",
                )
        image_bytes: bytes | None = None
        uploaded_filename: str | None = None
        if file is not None:
            try:
                image_bytes = await _read_limited_character_reference_upload(file)
            except _CharacterReferenceUploadTooLarge as exc:
                raise HTTPException(status_code=413, detail=str(exc)) from exc
            uploaded_filename = file.filename
        from bragi.services.character_text_service import CharacterTextService

        service = CharacterTextService(
            repositories=state.repositories,
            providers=state.providers,
        )
        try:
            content_rating = await service.classify_player_text(
                save_id=submitted_save_id,
                body=body,
                current_user_id=actor_user_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        async with state.lock.async_access():
            try:
                queued = service.queue_thread_text_send(
                    save_id=submitted_save_id,
                    thread_id=thread_id,
                    body=body,
                    content_rating=content_rating,
                )
            except ValueError as exc:
                detail = str(exc)
                status_code = 409 if "already pending" in detail else 400
                if detail == "Player does not have every character's number":
                    status_code = 403
                if detail.startswith("Unknown character text thread id"):
                    status_code = 404
                raise HTTPException(status_code=status_code, detail=detail) from exc
            queued_payload = _json_dict(queued)
        _publish_save_event(
            state,
            submitted_save_id,
            "character_texts_changed",
            queued_payload,
        )
        return await _start_queued_character_text_send_job(
            state=state,
            save_id=submitted_save_id,
            queued=queued,
            complete_thread=True,
            progress_label="Sending group text",
            uploaded_photo_bytes=image_bytes,
            uploaded_photo_filename=uploaded_filename,
            current_user_id=actor_user_id,
        )

    @app.post("/api/character-texts/contacts/{character_id}")
    def update_character_text_contact(
        character_id: str,
        payload: CharacterTextContactUpdateRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        with state.lock:
            resolved_save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, resolved_save_id, "chat")
            from bragi.services.character_text_service import CharacterTextService

            service = CharacterTextService(
                repositories=state.repositories,
                providers=state.providers,
            )
            try:
                model = service.update_contact_state(
                    save_id=resolved_save_id,
                    character_id=character_id,
                    player_has_character_number=payload.player_has_character_number,
                    character_has_player_number=payload.character_has_player_number,
                )
            except ValueError as exc:
                detail = str(exc)
                status_code = (
                    404
                    if detail.startswith("Unknown textable character id")
                    else 400
                )
                raise HTTPException(status_code=status_code, detail=detail) from exc
            payload_dict = _json_dict(model)
        _publish_save_event(
            state,
            resolved_save_id,
            "character_texts_changed",
            payload_dict,
        )
        return payload_dict

    @app.post("/api/character-texts/send")
    async def send_character_text(
        payload: CharacterTextSendRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            submitted_save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, submitted_save_id, "chat")
            actor_user_id = _owner_user_id_for_request(state)
            from bragi.services.character_text_service import CharacterTextService

            service = CharacterTextService(
                repositories=state.repositories,
                providers=state.providers,
            )
        try:
            content_rating = await service.classify_player_text(
                save_id=submitted_save_id,
                body=payload.body,
                current_user_id=actor_user_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        async with state.lock.async_access():
            try:
                queued = service.queue_text_send(
                    save_id=submitted_save_id,
                    character_id=payload.character_id,
                    body=payload.body,
                    content_rating=content_rating,
                )
            except ValueError as exc:
                detail = str(exc)
                status_code = 409 if "already pending" in detail else 400
                if detail == "Player does not have this character's number":
                    status_code = 403
                if detail.startswith("Unknown textable character id"):
                    status_code = 404
                raise HTTPException(status_code=status_code, detail=detail) from exc
            queued_payload = _json_dict(queued)
        _publish_save_event(
            state,
            submitted_save_id,
            "character_texts_changed",
            queued_payload,
        )
        job_marked = asyncio.Event()

        async def worker(handle: JobHandle) -> Any:
            await job_marked.wait()
            await handle.event("progress", {"label": "Sending text"})
            retry_callback, flush_retry_progress = _retry_progress_callback(
                handle,
                task_label="character text",
            )

            def delivery_retry_callback(progress: object) -> None:
                retry_callback(progress)
                _publish_save_event(
                    state,
                    submitted_save_id,
                    "character_texts_changed",
                    {"message_id": queued.player_message.id, "status": "retrying"},
                )

            from bragi.services.character_text_service import CharacterTextService
            from bragi.services.media_service import MediaService

            service = CharacterTextService(
                repositories=state.repositories,
                providers=state.providers,
                media_service=MediaService(
                    repositories=state.repositories,
                    providers=state.providers,
                    media_dir=state.paths.media_dir,
                ),
                prompt_inspection_store=_prompt_inspection_store_if_enabled(state),
            )
            try:
                result = await service.complete_queued_text_send(
                    save_id=submitted_save_id,
                    player_message_id=queued.player_message.id,
                    retry_progress_callback=delivery_retry_callback,
                    current_user_id=actor_user_id,
                )
                await flush_retry_progress()
            except ValueError as exc:
                _publish_save_event(
                    state,
                    submitted_save_id,
                    "character_texts_changed",
                    {"message_id": queued.player_message.id, "status": "failed"},
                )
                raise RuntimeError(str(exc)) from exc
            except Exception:
                _publish_save_event(
                    state,
                    submitted_save_id,
                    "character_texts_changed",
                    {"message_id": queued.player_message.id, "status": "failed"},
                )
                raise
            _publish_save_event(
                state,
                submitted_save_id,
                "character_texts_changed",
                result,
            )
            return result

        text_queue_key = _character_text_thread_job_key(queued.thread.id)
        try:
            created = await state.jobs.create(
                "character_text_send",
                worker,
                save_id=submitted_save_id,
                creator_user_id=_owner_user_id_for_request(state),
                exclusive_key=text_queue_key,
                operation_queue_key=text_queue_key,
            )
        except JobRegistryExclusiveKeyError as exc:
            async with state.lock.async_access():
                service = CharacterTextService(
                    repositories=state.repositories,
                    providers=state.providers,
                )
                service.repositories.update_character_text_delivery(
                    save_id=submitted_save_id,
                    message_id=queued.player_message.id,
                    status="failed",
                    error="Text delivery could not start",
                )
            _publish_save_event(
                state,
                submitted_save_id,
                "character_texts_changed",
                {"message_id": queued.player_message.id, "status": "failed"},
            )
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except JobRegistryFullError as exc:
            async with state.lock.async_access():
                service = CharacterTextService(
                    repositories=state.repositories,
                    providers=state.providers,
                )
                service.repositories.update_character_text_delivery(
                    save_id=submitted_save_id,
                    message_id=queued.player_message.id,
                    status="failed",
                    error="Text delivery could not start",
                )
            _publish_save_event(
                state,
                submitted_save_id,
                "character_texts_changed",
                {"message_id": queued.player_message.id, "status": "failed"},
            )
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        async with state.lock.async_access():
            service = CharacterTextService(
                repositories=state.repositories,
                providers=state.providers,
            )
            service.mark_text_send_job(
                save_id=submitted_save_id,
                player_message_id=queued.player_message.id,
                job_id=created.id,
            )
        job_marked.set()
        _publish_save_event(
            state,
            submitted_save_id,
            "character_texts_changed",
            {"message_id": queued.player_message.id, "job_id": created.id},
        )
        return _job_summary_for_request(state, created)

    @app.post("/api/character-texts/send-image")
    async def send_character_text_with_image(
        state: StateDep,
        file: Annotated[UploadFile | None, File()] = None,
        save_id: Annotated[str | None, Form()] = None,
        character_id: Annotated[str, Form()] = "",
        body: Annotated[str, Form()] = "",
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            submitted_save_id = _require_save_id(save_id)
            _raise_unless_save_action_allowed(state, submitted_save_id, "chat")
            actor_user_id = _owner_user_id_for_request(state)
            if file is not None:
                _raise_unless_save_action_allowed(
                    state,
                    submitted_save_id,
                    "media",
                )
        image_bytes: bytes | None = None
        uploaded_filename: str | None = None
        if file is not None:
            try:
                image_bytes = await _read_limited_character_reference_upload(file)
            except _CharacterReferenceUploadTooLarge as exc:
                raise HTTPException(status_code=413, detail=str(exc)) from exc
            uploaded_filename = file.filename
        from bragi.services.character_text_service import CharacterTextService

        service = CharacterTextService(
            repositories=state.repositories,
            providers=state.providers,
        )
        try:
            content_rating = await service.classify_player_text(
                save_id=submitted_save_id,
                body=body,
                current_user_id=actor_user_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        async with state.lock.async_access():
            try:
                queued = service.queue_text_send(
                    save_id=submitted_save_id,
                    character_id=character_id,
                    body=body,
                    content_rating=content_rating,
                )
            except ValueError as exc:
                detail = str(exc)
                status_code = 409 if "already pending" in detail else 400
                if detail == "Player does not have this character's number":
                    status_code = 403
                if detail.startswith("Unknown textable character id"):
                    status_code = 404
                raise HTTPException(status_code=status_code, detail=detail) from exc
            queued_payload = _json_dict(queued)
        _publish_save_event(
            state,
            submitted_save_id,
            "character_texts_changed",
            queued_payload,
        )
        return await _start_queued_character_text_send_job(
            state=state,
            save_id=submitted_save_id,
            queued=queued,
            complete_thread=False,
            progress_label="Sending text",
            uploaded_photo_bytes=image_bytes,
            uploaded_photo_filename=uploaded_filename,
            current_user_id=actor_user_id,
        )

    @app.post("/api/character-texts/spontaneous")
    async def send_spontaneous_character_text(
        payload: CharacterTextSpontaneousRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            submitted_save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, submitted_save_id, "chat")
            actor_user_id = _owner_user_id_for_request(state)
            from bragi.services.character_text_service import CharacterTextService

            service = CharacterTextService(
                repositories=state.repositories,
                providers=state.providers,
            )
            try:
                queued = service.queue_spontaneous_text(
                    save_id=submitted_save_id,
                    character_id=payload.character_id,
                )
            except ValueError as exc:
                detail = str(exc)
                status_code = 409 if "already pending" in detail else 400
                if detail in {
                    "Player does not have this character's number",
                    "Character does not have the player's number",
                }:
                    status_code = 403
                if detail.startswith("Unknown textable character id"):
                    status_code = 404
                raise HTTPException(status_code=status_code, detail=detail) from exc
            queued_payload = _json_dict(queued)
        _publish_save_event(
            state,
            submitted_save_id,
            "character_texts_changed",
            queued_payload,
        )
        job_marked = asyncio.Event()

        async def worker(handle: JobHandle) -> Any:
            await job_marked.wait()
            await handle.event("progress", {"label": "Sending character text"})
            retry_callback, flush_retry_progress = _retry_progress_callback(
                handle,
                task_label="character text",
            )

            def delivery_retry_callback(progress: object) -> None:
                retry_callback(progress)
                _publish_save_event(
                    state,
                    submitted_save_id,
                    "character_texts_changed",
                    {"message_id": queued.message.id, "status": "retrying"},
                )

            from bragi.services.character_text_service import CharacterTextService
            from bragi.services.media_service import MediaService

            service = CharacterTextService(
                repositories=state.repositories,
                providers=state.providers,
                media_service=MediaService(
                    repositories=state.repositories,
                    providers=state.providers,
                    media_dir=state.paths.media_dir,
                ),
                prompt_inspection_store=_prompt_inspection_store_if_enabled(state),
            )
            try:
                result = await service.complete_queued_spontaneous_text(
                    save_id=submitted_save_id,
                    text_message_id=queued.message.id,
                    retry_progress_callback=delivery_retry_callback,
                    current_user_id=actor_user_id,
                )
                await flush_retry_progress()
            except ValueError as exc:
                _publish_save_event(
                    state,
                    submitted_save_id,
                    "character_texts_changed",
                    {"message_id": queued.message.id, "status": "failed"},
                )
                raise RuntimeError(str(exc)) from exc
            except Exception:
                _publish_save_event(
                    state,
                    submitted_save_id,
                    "character_texts_changed",
                    {"message_id": queued.message.id, "status": "failed"},
                )
                raise
            _publish_save_event(
                state,
                submitted_save_id,
                "character_texts_changed",
                result,
            )
            return result

        text_queue_key = _character_text_thread_job_key(queued.thread.id)
        try:
            created = await state.jobs.create(
                "character_text_spontaneous",
                worker,
                save_id=submitted_save_id,
                creator_user_id=_owner_user_id_for_request(state),
                exclusive_key=text_queue_key,
                operation_queue_key=text_queue_key,
            )
        except JobRegistryExclusiveKeyError as exc:
            async with state.lock.async_access():
                service = CharacterTextService(
                    repositories=state.repositories,
                    providers=state.providers,
                )
                service.repositories.update_character_text_delivery(
                    save_id=submitted_save_id,
                    message_id=queued.message.id,
                    status="failed",
                    error="Text delivery could not start",
                )
            _publish_save_event(
                state,
                submitted_save_id,
                "character_texts_changed",
                {"message_id": queued.message.id, "status": "failed"},
            )
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except JobRegistryFullError as exc:
            async with state.lock.async_access():
                service = CharacterTextService(
                    repositories=state.repositories,
                    providers=state.providers,
                )
                service.repositories.update_character_text_delivery(
                    save_id=submitted_save_id,
                    message_id=queued.message.id,
                    status="failed",
                    error="Text delivery could not start",
                )
            _publish_save_event(
                state,
                submitted_save_id,
                "character_texts_changed",
                {"message_id": queued.message.id, "status": "failed"},
            )
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        async with state.lock.async_access():
            service = CharacterTextService(
                repositories=state.repositories,
                providers=state.providers,
            )
            service.mark_spontaneous_text_job(
                save_id=submitted_save_id,
                text_message_id=queued.message.id,
                job_id=created.id,
            )
        job_marked.set()
        _publish_save_event(
            state,
            submitted_save_id,
            "character_texts_changed",
            {"message_id": queued.message.id, "job_id": created.id},
        )
        return _job_summary_for_request(state, created)

    @app.post("/api/character-texts/message-edit")
    async def edit_character_text_message(
        payload: CharacterTextEditRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            submitted_save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, submitted_save_id, "mutate")
            selected = state.repositories.get_character_text_message(
                save_id=submitted_save_id,
                message_id=payload.text_message_id,
            )
            if selected is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Unknown character text message id: "
                        f"{payload.text_message_id}"
                    ),
                )
            text_queue_key = _character_text_thread_job_key(selected.thread_id)
            actor_user_id = _owner_user_id_for_request(state)

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": "Saving text edit"})
            from bragi.services.character_text_revision_service import (
                CharacterTextRevisionService,
            )
            from bragi.services.character_text_service import CharacterTextService

            service = CharacterTextRevisionService(
                repositories=state.repositories,
                providers=state.providers,
            )
            selected_for_edit = state.repositories.get_character_text_message(
                save_id=submitted_save_id,
                message_id=payload.text_message_id,
            )
            if selected_for_edit is None:
                raise RuntimeError(
                    f"Unknown character text message id: {payload.text_message_id}"
            )
            if selected_for_edit.sender == "character":
                edit = await service.correct_character_text_with_safety(
                    save_id=submitted_save_id,
                    text_message_id=payload.text_message_id,
                    body=payload.body,
                    current_user_id=actor_user_id,
                )
            else:
                edit = service.edit_text_without_resubmit(
                    save_id=submitted_save_id,
                    text_message_id=payload.text_message_id,
                    body=payload.body,
                )
            thread = CharacterTextService(
                repositories=state.repositories,
                providers=state.providers,
            ).get_thread_model(
                save_id=submitted_save_id,
                thread_id=edit.message.thread_id,
            )
            result = {
                "save_id": submitted_save_id,
                "thread": thread,
                "message_id": edit.message.id,
                "revision_id": edit.revision.id,
            }
            _publish_save_event(
                state,
                submitted_save_id,
                "character_texts_changed",
                result,
            )
            return result

        return await _create_job_summary(
            state,
            "character_text_message_edit",
            worker,
            save_id=submitted_save_id,
            exclusive_key=text_queue_key,
            operation_queue_key=text_queue_key,
        )

    @app.post("/api/character-texts/delete-from-here")
    async def delete_character_texts_from_here(
        payload: CharacterTextDeleteRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            submitted_save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, submitted_save_id, "mutate")
            selected = state.repositories.get_character_text_message(
                save_id=submitted_save_id,
                message_id=payload.text_message_id,
            )
            if selected is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Unknown character text message id: "
                        f"{payload.text_message_id}"
                    ),
                )
            text_queue_key = _character_text_thread_job_key(selected.thread_id)

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": "Deleting texts"})
            from bragi.services.character_text_revision_service import (
                CharacterTextRevisionService,
            )

            result = CharacterTextRevisionService(
                repositories=state.repositories,
                providers=state.providers,
            ).delete_text_messages_from_here(
                save_id=submitted_save_id,
                text_message_id=payload.text_message_id,
            )
            result_payload = {
                "save_id": submitted_save_id,
                "thread_id": result.thread.id,
                "thread": _json_dict(result.thread),
                "deleted_message_ids": [
                    message.id for message in result.deleted_messages
                ],
                "deleted_count": len(result.deleted_messages),
            }
            _publish_save_event(
                state,
                submitted_save_id,
                "character_texts_changed",
                result_payload,
            )
            return result_payload

        return await _create_job_summary(
            state,
            "character_text_delete",
            worker,
            save_id=submitted_save_id,
            exclusive_key=text_queue_key,
            operation_queue_key=text_queue_key,
        )

    @app.post("/api/character-texts/edit")
    async def edit_and_resubmit_character_text(
        payload: CharacterTextEditRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            submitted_save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, submitted_save_id, "mutate")
            selected = state.repositories.get_character_text_message(
                save_id=submitted_save_id,
                message_id=payload.text_message_id,
            )
            if selected is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Unknown character text message id: "
                        f"{payload.text_message_id}"
                    ),
                )
            replacement = payload.body.strip()
            if not replacement:
                raise HTTPException(status_code=400, detail="Text message is empty")
            if selected.sender != "player":
                raise HTTPException(
                    status_code=400,
                    detail="Only player text messages can be edited and resent",
                )
            if selected.delivery_status in {"pending", "retrying"}:
                raise HTTPException(
                    status_code=409,
                    detail="Character text send is already pending",
                )
            if selected.body.strip() == replacement:
                raise HTTPException(
                    status_code=400,
                    detail="Text message was not changed",
                )
            text_queue_key = _character_text_thread_job_key(selected.thread_id)
            actor_user_id = _owner_user_id_for_request(state)

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": "Replaying text edit"})
            from bragi.services.character_text_revision_service import (
                CharacterTextRevisionService,
            )
            from bragi.services.media_service import MediaService

            result = await CharacterTextRevisionService(
                repositories=state.repositories,
                providers=state.providers,
                media_service=MediaService(
                    repositories=state.repositories,
                    providers=state.providers,
                    media_dir=state.paths.media_dir,
                ),
            ).edit_text_and_resubmit(
                save_id=submitted_save_id,
                text_message_id=payload.text_message_id,
                body=payload.body,
                current_user_id=actor_user_id,
            )
            _publish_save_event(
                state,
                submitted_save_id,
                "character_texts_changed",
                result,
            )
            return result

        return await _create_job_summary(
            state,
            "character_text_edit",
            worker,
            save_id=submitted_save_id,
            exclusive_key=text_queue_key,
            operation_queue_key=text_queue_key,
        )

    @app.post("/api/saves/{save_id}/load")
    def load_save(save_id: str, state: StateDep) -> dict[str, Any]:
        with state.lock:
            _raise_unless_save_action_allowed(state, save_id, "read")
            _touch_save_last_opened_if_possible(state, save_id)
            _remember_user_active_save(state, save_id)
            return _runtime_json_dict(
                state,
                _build_runtime_model_for_save(
                    state,
                    save_id,
                    status="Save loaded",
                ),
            )

    @app.post("/api/saves/{save_id}/rename")
    def rename_save(
        save_id: str,
        payload: RenameSaveRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        with state.lock:
            _raise_unless_save_action_allowed(state, save_id, "mutate")
            title = payload.title.strip()
            if not title:
                raise HTTPException(status_code=400, detail="Save title is required")
            active_save_id = _resolve_runtime_save_id(state, None)
            model = state.runtime.rename_save(
                save_id=save_id,
                title=title,
                active_save_id=active_save_id,
            )
            payload_dict = _runtime_json_dict(state, model)
        _publish_save_event(
            state,
            save_id,
            "runtime_changed",
            {"reason": "save_renamed"},
        )
        _publish_save_event(
            state,
            None,
            "saves_changed",
            {"reason": "save_renamed"},
        )
        return payload_dict

    @app.delete("/api/saves/{save_id}")
    def delete_save(save_id: str, state: StateDep) -> dict[str, Any]:
        with state.lock:
            _raise_unless_save_delete_allowed(state, save_id)
            model = state.runtime.delete_save(save_id)
            payload = _runtime_json_dict(state, model)
        if not _runtime_model_error(payload):
            _publish_save_event(state, save_id, "save_deleted", {"save_id": save_id})
            _publish_save_event(
                state,
                None,
                "saves_changed",
                {"reason": "save_deleted"},
            )
        return payload

    @app.get("/api/saves/{save_id}/events")
    async def save_events(
        save_id: str,
        request: Request,
        state: StateDep,
    ) -> StreamingResponse:
        async with state.lock.async_access():
            _raise_unless_save_action_allowed(state, save_id, "read")
        last_event_id = _save_event_cursor_from_header(
            request.headers.get("last-event-id"),
            latest_event_id=state.save_events.latest_event_id(),
        )
        observe(
            "web.save_sse.opened",
            save_id=save_id,
            last_event_id=last_event_id,
        )
        return StreamingResponse(
            _save_event_stream(
                state,
                save_id,
                last_event_id,
                current_user=_current_request_user(),
                current_user_role=_current_request_role(state),
                **_save_event_stream_auth_filter(state),
            ),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    @app.get("/api/scenarios")
    def list_scenarios(state: StateDep) -> dict[str, Any]:
        with state.lock:
            list_saved_scenarios = state.runtime.list_saved_scenarios
            kwargs: dict[str, Any] = {}
            if _call_accepts_keyword(list_saved_scenarios, "current_user_id"):
                kwargs["current_user_id"] = _owner_user_id_for_request(state)
            return {
                "scenarios": to_jsonable(list_saved_scenarios(**kwargs)),
            }

    @app.get("/api/scenarios/{scenario_id}/definition")
    def get_scenario_definition(
        scenario_id: str,
        state: StateDep,
    ) -> dict[str, Any]:
        with state.lock:
            _raise_unless_scenario_supported(state, scenario_id)
            return _json_dict(
                WorldDataService(
                    state.repositories,
                    allowed_content_rating=_content_safety_policy_for_request(
                        state
                    ).rating,
                ).build_scenario_definition_model(scenario_id)
            )

    @app.post("/api/scenarios/{scenario_id}/definition")
    async def update_scenario_definition(
        scenario_id: str,
        payload: ScenarioDefinitionApplyRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        _require_admin_user()
        try:
            edit = _scenario_edit_from_json(payload.edit)
            async with state.lock.async_access():
                _raise_unless_scenario_supported(state, scenario_id)
                scenario = state.repositories.get_scenario(scenario_id)
                if scenario is None:
                    raise ValueError(f"Unknown scenario id: {scenario_id}")
                actor_user_id = _owner_user_id_for_request(state)
                edit = await _review_scenario_edit_for_request(
                    state,
                    edit,
                    save_id=None,
                    roleplay_type=scenario.type,
                    current_user_id=actor_user_id,
                )
                return _json_dict(
                    WorldDataService(
                        state.repositories,
                        allowed_content_rating=(
                            _content_safety_policy_for_request(state).rating
                        ),
                    ).apply_scenario_definition_edit(scenario_id, edit)
                )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/scenarios/{scenario_id}/character-starters/reference-image/upload")
    async def upload_scenario_starter_reference_image(
        scenario_id: str,
        state: StateDep,
        file: Annotated[UploadFile, File()],
        starter_id: Annotated[str | None, Form()] = None,
        starter_name: Annotated[str, Form()] = "",
        replace_existing: Annotated[bool, Form()] = False,
    ) -> dict[str, Any]:
        _require_admin_user()
        async with state.lock.async_access():
            _raise_unless_scenario_supported(state, scenario_id)
        _raise_unless_unrated_reference_upload(state)
        try:
            image_bytes = await _read_limited_character_reference_upload(file)
        except _CharacterReferenceUploadTooLarge as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        try:
            async with state.lock.async_access():
                model = state.runtime.upload_scenario_starter_reference_image(
                    scenario_id=scenario_id,
                    image_bytes=image_bytes,
                    filename=file.filename,
                    starter_id=starter_id,
                    starter_name=starter_name,
                    replace_existing=replace_existing,
                )
                payload_dict = _json_dict(model)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _publish_save_event(
            state,
            None,
            "scenarios_changed",
            {"reason": "scenario_starter_reference_uploaded"},
        )
        return payload_dict

    @app.post("/api/scenarios/{scenario_id}/character-starters/reference-image/remove")
    def remove_scenario_starter_reference_image(
        scenario_id: str,
        payload: ScenarioStarterReferenceRemoveRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        _require_admin_user()
        try:
            with state.lock:
                _raise_unless_scenario_supported(state, scenario_id)
                model = state.runtime.remove_scenario_starter_reference_image(
                    scenario_id=scenario_id,
                    starter_id=payload.starter_id,
                    starter_name=payload.starter_name,
                )
                payload_dict = _json_dict(model)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _publish_save_event(
            state,
            None,
            "scenarios_changed",
            {"reason": "scenario_starter_reference_removed"},
        )
        return payload_dict

    @app.get("/api/scenarios/{scenario_id}/character-starters/reference-images/{image_id}")
    def scenario_starter_reference_image(
        scenario_id: str,
        image_id: str,
        state: StateDep,
    ) -> FileResponse:
        with state.lock:
            _raise_unless_scenario_supported(state, scenario_id)
            image = _scenario_starter_reference_image_unlocked(
                state,
                scenario_id=scenario_id,
                image_id=image_id,
                thumbnail=False,
            )
        if image is None:
            raise HTTPException(
                status_code=404,
                detail="Starter reference image not found",
            )
        path, media_type = image
        return _media_file_response(path, media_type=media_type)

    @app.get(
        "/api/scenarios/{scenario_id}/character-starters/reference-images/{image_id}/thumbnail"
    )
    def scenario_starter_reference_thumbnail(
        scenario_id: str,
        image_id: str,
        state: StateDep,
    ) -> FileResponse:
        with state.lock:
            _raise_unless_scenario_supported(state, scenario_id)
            image = _scenario_starter_reference_image_unlocked(
                state,
                scenario_id=scenario_id,
                image_id=image_id,
                thumbnail=True,
            )
        if image is None:
            raise HTTPException(
                status_code=404,
                detail="Starter reference image not found",
            )
        path, media_type = image
        return _media_file_response(path, media_type=media_type)

    @app.post("/api/scenarios/manual")
    def create_manual_scenario(
        payload: dict[str, Any],
        state: StateDep,
    ) -> dict[str, Any]:
        _raise_if_retired_scenario_request(
            payload.get("scenario_type"),
            payload.get("scenario_types"),
        )
        _raise_if_invalid_interaction_mode(payload.get("interaction_mode"))
        try:
            scenario = ManualScenarioInput(**payload)
        except TypeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        with state.lock:
            kwargs: dict[str, Any] = {}
            current_user_id = _owner_user_id_for_request(state)
            if _call_accepts_keyword(
                state.runtime.create_manual_scenario,
                "owner_user_id",
            ):
                kwargs["owner_user_id"] = current_user_id
            if _call_accepts_keyword(
                state.runtime.create_manual_scenario,
                "current_user_id",
            ):
                kwargs["current_user_id"] = current_user_id
            if _call_accepts_keyword(
                state.runtime.create_manual_scenario,
                "remember_process_active_save",
            ):
                kwargs["remember_process_active_save"] = not _auth_context_enabled(
                    state
                )
            payload_dict = _runtime_json_dict(
                state,
                state.runtime.create_manual_scenario(scenario, **kwargs),
            )
            _remember_user_active_save_from_model_result(state, payload_dict)
        _publish_runtime_changed_from_model_result(
            state,
            payload_dict,
            reason="save_created",
        )
        _publish_save_event(state, None, "saves_changed", {"reason": "save_created"})
        return payload_dict

    @app.post("/api/scenarios/{scenario_id}/start")
    def start_saved_scenario(
        scenario_id: str,
        payload: StartScenarioRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        with state.lock:
            _raise_unless_scenario_supported(state, scenario_id)
            kwargs: dict[str, Any] = {}
            current_user_id = _owner_user_id_for_request(state)
            if _call_accepts_keyword(
                state.runtime.start_saved_scenario,
                "owner_user_id",
            ):
                kwargs["owner_user_id"] = current_user_id
            if _call_accepts_keyword(
                state.runtime.start_saved_scenario,
                "current_user_id",
            ):
                kwargs["current_user_id"] = current_user_id
            if _call_accepts_keyword(
                state.runtime.start_saved_scenario,
                "remember_process_active_save",
            ):
                kwargs["remember_process_active_save"] = not _auth_context_enabled(
                    state
                )
            payload_dict = _runtime_json_dict(
                state,
                state.runtime.start_saved_scenario(
                    scenario_id=scenario_id,
                    save_title=payload.save_title,
                    **kwargs,
                ),
            )
            _remember_user_active_save_from_model_result(state, payload_dict)
        _publish_runtime_changed_from_model_result(
            state,
            payload_dict,
            reason="save_created",
        )
        _publish_save_event(state, None, "saves_changed", {"reason": "save_created"})
        return payload_dict

    @app.delete("/api/scenarios/{scenario_id}")
    def delete_saved_scenario(scenario_id: str, state: StateDep) -> dict[str, Any]:
        _require_admin_user()
        with state.lock:
            payload_dict = _runtime_json_dict(
                state,
                state.runtime.delete_saved_scenario(scenario_id),
            )
        _publish_save_event(
            state,
            None,
            "scenarios_changed",
            {"reason": "scenario_deleted"},
        )
        return payload_dict

    @app.post("/api/scenarios/draft")
    async def generate_scenario_draft(
        payload: ScenarioDraftRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        _raise_if_retired_scenario_request(
            payload.scenario_type,
            payload.scenario_types,
        )
        _raise_if_invalid_interaction_mode(payload.interaction_mode)
        async with state.lock.async_access():
            current_user_id = _owner_user_id_for_request(state)

        async def worker(handle: JobHandle) -> Any:
            async def progress(model: object) -> None:
                await handle.event("progress", model)

            kwargs: dict[str, Any] = {}
            if _call_accepts_keyword(
                state.runtime.generate_scenario_draft,
                "scenario_types",
            ):
                kwargs["scenario_types"] = payload.scenario_types
            if _call_accepts_keyword(
                state.runtime.generate_scenario_draft,
                "current_user_id",
            ):
                kwargs["current_user_id"] = current_user_id
            if _call_accepts_keyword(
                state.runtime.generate_scenario_draft,
                "interaction_mode",
            ):
                kwargs["interaction_mode"] = payload.interaction_mode
            return await state.runtime.generate_scenario_draft(
                scenario_type=payload.scenario_type,
                seed=payload.seed,
                action_choices_enabled=payload.action_choices_enabled,
                progress_callback=progress,
                **kwargs,
            )

        return await _create_job_summary(
            state,
            "scenario_draft",
            worker,
            lock_runtime=False,
        )

    @app.post("/api/scenarios/continuation-draft")
    async def generate_continuation_scenario_draft(
        payload: ContinuationDraftRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            draft_save_id = _require_save_id(payload.save_id)
            _raise_if_save_retired(state, draft_save_id)
            chapter_start_instructions = payload.chapter_start_instructions.strip()
            current_user_id = _owner_user_id_for_request(state)

        async def worker(handle: JobHandle) -> Any:
            async def progress(model: object) -> None:
                await handle.event("progress", model)

            kwargs: dict[str, Any] = {
                "active_save_id": draft_save_id,
                "chapter_start_instructions": chapter_start_instructions,
                "progress_callback": progress,
            }
            if _call_accepts_keyword(
                state.runtime.generate_continuation_scenario_draft,
                "current_user_id",
            ):
                kwargs["current_user_id"] = current_user_id
            return await state.runtime.generate_continuation_scenario_draft(**kwargs)

        return await _create_job_summary(
            state,
            "scenario_draft",
            worker,
            save_id=draft_save_id,
        )

    @app.post("/api/scenarios/draft/save")
    async def save_scenario_draft(
        payload: SaveScenarioDraftRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        _raise_if_retired_scenario_request(
            payload.scenario_type,
            payload.scenario_types,
        )
        _raise_if_invalid_interaction_mode(payload.interaction_mode)
        async with state.lock.async_access():
            kwargs: dict[str, Any] = {}
            current_user_id = _owner_user_id_for_request(state)
            if _call_accepts_keyword(
                state.runtime.save_scenario_draft,
                "owner_user_id",
            ):
                kwargs["owner_user_id"] = current_user_id
            if _call_accepts_keyword(
                state.runtime.save_scenario_draft,
                "current_user_id",
            ):
                kwargs["current_user_id"] = current_user_id
            if _call_accepts_keyword(
                state.runtime.save_scenario_draft,
                "remember_process_active_save",
            ):
                kwargs["remember_process_active_save"] = not _auth_context_enabled(
                    state
                )
            if _call_accepts_keyword(
                state.runtime.save_scenario_draft,
                "scenario_types",
            ):
                kwargs["scenario_types"] = payload.scenario_types
            if _call_accepts_keyword(
                state.runtime.save_scenario_draft,
                "character_starters",
            ):
                kwargs["character_starters"] = payload.character_starters
            if _call_accepts_keyword(
                state.runtime.save_scenario_draft,
                "defer_opening_action_choices",
            ):
                kwargs["defer_opening_action_choices"] = True
            if _call_accepts_keyword(
                state.runtime.save_scenario_draft,
                "interaction_mode",
            ):
                kwargs["interaction_mode"] = payload.interaction_mode
            result = state.runtime.save_scenario_draft(
                scenario_type=payload.scenario_type,
                sections=payload.sections,
                action_choices_enabled=payload.action_choices_enabled,
                save_title=payload.save_title,
                source_metadata=payload.source_metadata,
                **kwargs,
            )
            if inspect.isawaitable(result):
                result = await result
            payload_dict = _runtime_json_dict(state, result)
            _remember_user_active_save_from_model_result(state, payload_dict)
        _publish_runtime_changed_from_model_result(
            state,
            payload_dict,
            reason="save_created",
        )
        _publish_save_event(state, None, "saves_changed", {"reason": "save_created"})
        action_choices = payload_dict.get("action_choices")
        save_id = payload_dict.get("active_save_id")
        narrator_message_id = (
            action_choices.get("narrator_message_id")
            if isinstance(action_choices, dict)
            else None
        )
        if (
            not payload_dict.get("error")
            and payload_dict.get("action_choices_enabled") is True
            and isinstance(action_choices, dict)
            and action_choices.get("choices") == []
            and isinstance(save_id, str)
            and isinstance(narrator_message_id, str)
        ):
            try:
                action_choices["generation_job"] = (
                    await _create_action_choice_job_summary(
                        state,
                        save_id=save_id,
                        narrator_message_id=narrator_message_id,
                        current_user_id=current_user_id,
                        job_type="action_choice_generate",
                        progress_label="Generating action choices",
                    )
                )
            except HTTPException as exc:
                action_choices["generation_error"] = str(exc.detail)
        return payload_dict

    @app.post("/api/scenarios/draft/character-starters/generate")
    async def generate_scenario_draft_character_starters(
        payload: ScenarioDraftCharacterStarterGenerationRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        _raise_if_retired_scenario_request(
            payload.scenario_type,
            payload.scenario_types,
        )
        _raise_if_invalid_interaction_mode(payload.interaction_mode)
        async with state.lock.async_access():
            current_user = _save_access_user(state)
            if current_user is not None and current_user.role == "child":
                raise HTTPException(
                    status_code=403,
                    detail="Character starter generation is not allowed",
                )
            current_user_id = current_user.id if current_user is not None else None
        _raise_if_starter_generation_payload_too_large(payload)
        _raise_if_starter_generation_request_invalid(payload)

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": "Generating character starters"})
            kwargs: dict[str, Any] = {}
            if _call_accepts_keyword(
                state.runtime.generate_scenario_draft_character_starters,
                "scenario_types",
            ):
                kwargs["scenario_types"] = payload.scenario_types
            if _call_accepts_keyword(
                state.runtime.generate_scenario_draft_character_starters,
                "current_user_id",
            ):
                kwargs["current_user_id"] = current_user_id
            if _call_accepts_keyword(
                state.runtime.generate_scenario_draft_character_starters,
                "action_choices_enabled",
            ):
                kwargs["action_choices_enabled"] = payload.action_choices_enabled
            if _call_accepts_keyword(
                state.runtime.generate_scenario_draft_character_starters,
                "interaction_mode",
            ):
                kwargs["interaction_mode"] = payload.interaction_mode
            return await state.runtime.generate_scenario_draft_character_starters(
                scenario_type=payload.scenario_type,
                sections=payload.sections,
                character_starters=payload.character_starters,
                count=payload.count,
                custom_description=payload.custom_description,
                **kwargs,
            )

        return await _create_job_summary(
            state,
            "scenario_character_starters",
            worker,
            lock_runtime=False,
        )

    @app.post("/api/scenarios/draft/section")
    async def regenerate_scenario_section(
        payload: RegenerateScenarioSectionRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        _raise_if_retired_scenario_request(
            payload.scenario_type,
            payload.scenario_types,
        )
        _raise_if_invalid_interaction_mode(payload.interaction_mode)
        async with state.lock.async_access():
            current_user_id = _owner_user_id_for_request(state)

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": "Regenerating section"})
            kwargs: dict[str, Any] = {}
            if _call_accepts_keyword(
                state.runtime.regenerate_scenario_section,
                "scenario_types",
            ):
                kwargs["scenario_types"] = payload.scenario_types
            if _call_accepts_keyword(
                state.runtime.regenerate_scenario_section,
                "current_user_id",
            ):
                kwargs["current_user_id"] = current_user_id
            if _call_accepts_keyword(
                state.runtime.regenerate_scenario_section,
                "interaction_mode",
            ):
                kwargs["interaction_mode"] = payload.interaction_mode
            return await state.runtime.regenerate_scenario_section(
                scenario_type=payload.scenario_type,
                seed=payload.seed,
                section_id=payload.section_id,
                sections=payload.sections,
                action_choices_enabled=payload.action_choices_enabled,
                **kwargs,
            )

        return await _create_job_summary(
            state,
            "scenario_section",
            worker,
            lock_runtime=False,
        )

    @app.post("/api/chat")
    async def submit_chat(payload: ChatRequest, state: StateDep) -> dict[str, Any]:
        if payload.speaker_name == STORY_CONTINUATION_SPEAKER_NAME:
            raise HTTPException(
                status_code=400,
                detail="speaker_name is reserved for internal Storyteller turns",
            )
        return await _submit_chat(payload, state, operation="chat")

    async def _submit_chat(
        payload: ChatRequest,
        state: StateDep,
        *,
        operation: str,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            submitted_save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, submitted_save_id, "chat")
            current_user_id = _owner_user_id_for_request(state)
        request_fingerprint = _chat_turn_request_fingerprint(
            operation,
            {
                "body": payload.body.strip(),
                "speaker_name": (
                    payload.speaker_name.strip()
                    if payload.speaker_name and payload.speaker_name.strip()
                    else None
                ),
            },
        )
        replay = _chat_turn_submission_replay(
            state,
            save_id=submitted_save_id,
            client_turn_id=str(payload.client_turn_id),
            operation=operation,
            request_fingerprint=request_fingerprint,
        )
        if replay is not None:
            return replay
        status = _chat_submission_status(state, submitted_save_id)
        if not status["can_submit"]:
            if status["reason"] == "no_save":
                raise HTTPException(status_code=400, detail="No save loaded")
            raise HTTPException(status_code=409, detail=_CHAT_TURN_ACTIVE_DETAIL)
        await _cancel_action_choice_jobs_for_save(state, submitted_save_id)

        async def worker(handle: JobHandle) -> Any:
            initial_progress = _initial_chat_turn_progress("Submitting turn")
            await handle.event("progress", initial_progress)
            (
                turn_progress_callback,
                flush_turn_progress,
                latest_turn_progress_jobs,
            ) = _turn_progress_callback(handle, initial_progress)
            retry_callback, flush_retry_progress = _retry_progress_callback(
                handle,
                task_label="chat",
            )
            kwargs: dict[str, Any] = {
                "body": payload.body,
                "speaker_name": payload.speaker_name,
                "active_save_id": submitted_save_id,
            }
            await _wait_for_background_post_turn_catchup(
                state,
                handle,
                save_id=submitted_save_id,
            )

            if _call_accepts_keyword(
                state.runtime.submit_player_message_for_initial_render,
                "current_user_id",
            ):
                kwargs["current_user_id"] = current_user_id
            if _call_accepts_keyword(
                state.runtime.submit_player_message_for_initial_render,
                "retry_progress_callback",
            ):
                kwargs["retry_progress_callback"] = retry_callback
            if _call_accepts_keyword(
                state.runtime.submit_player_message_for_initial_render,
                "turn_progress_callback",
            ):
                kwargs["turn_progress_callback"] = turn_progress_callback
            try:
                turn = await state.runtime.submit_player_message_for_initial_render(
                    **kwargs
                )
            finally:
                await flush_retry_progress()
                await flush_turn_progress()
            _link_chat_turn_submission_messages(state, handle, turn)
            _raise_for_initial_chat_turn_failure(
                turn,
                failure_message="Chat turn did not produce a narrator response",
            )
            await _emit_initial_chat_turn_event(handle, turn)
            await handle.advance_completion_level(RESPONSE_COMMITTED)
            if turn.has_post_turn_jobs:
                queued = await _queue_post_turn_jobs_background(
                    state,
                    handle,
                    save_id=turn.save_id or "",
                    player_message_id=turn.player_message_id or "",
                    narrator_message_id=turn.narrator_message_id or "",
                    turn_revision=getattr(turn, "turn_revision", None),
                    prepared_action_choices=getattr(
                        turn,
                        "prepared_action_choices",
                        None,
                    ),
                    prior_phase_jobs=_completed_chat_turn_phase_jobs(
                        latest_turn_progress_jobs()
                    ),
                    current_user_id=current_user_id,
                )
                if queued is None:
                    return await _run_post_turn_jobs_inline_fallback(
                        state,
                        handle,
                        save_id=turn.save_id or "",
                        player_message_id=turn.player_message_id or "",
                        narrator_message_id=turn.narrator_message_id or "",
                        turn_revision=getattr(turn, "turn_revision", None),
                        prepared_action_choices=getattr(
                            turn,
                            "prepared_action_choices",
                            None,
                        ),
                        prior_phase_jobs=_completed_chat_turn_phase_jobs(
                            latest_turn_progress_jobs()
                        ),
                        current_user_id=current_user_id,
                    )
                return _initial_chat_turn_result(turn)
            return _initial_chat_turn_result(turn)

        return await _create_idempotent_chat_job_summary(
            state,
            worker,
            save_id=submitted_save_id,
            client_turn_id=str(payload.client_turn_id),
            operation=operation,
            request_fingerprint=request_fingerprint,
        )

    @app.post("/api/chat/continue")
    async def continue_story(
        payload: ContinueStoryRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            submitted_save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, submitted_save_id, "chat")
        request_fingerprint = _chat_turn_request_fingerprint(
            "continue",
            {
                "body": STORY_CONTINUATION_DIRECTION,
                "speaker_name": STORY_CONTINUATION_SPEAKER_NAME,
            },
        )
        replay = _chat_turn_submission_replay(
            state,
            save_id=submitted_save_id,
            client_turn_id=str(payload.client_turn_id),
            operation="continue",
            request_fingerprint=request_fingerprint,
        )
        if replay is not None:
            return replay
        async with state.lock.async_access():
            save = state.repositories.get_save(submitted_save_id)
            if (
                save is None
                or save.interaction_mode != "storyteller"
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Continue story is only available in Storyteller mode."
                    ),
                )
        return await _submit_chat(
            ChatRequest(
                body=STORY_CONTINUATION_DIRECTION,
                speaker_name=STORY_CONTINUATION_SPEAKER_NAME,
                save_id=submitted_save_id,
                client_turn_id=payload.client_turn_id,
            ),
            state,
            operation="continue",
        )

    @app.post("/api/chat/look-around")
    async def submit_look_around(
        payload: LookAroundRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        query = payload.query.strip()
        if not query:
            raise HTTPException(
                status_code=400,
                detail="Look Around query is required",
            )
        async with state.lock.async_access():
            submitted_save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, submitted_save_id, "chat")
            current_user_id = _owner_user_id_for_request(state)
        status = _chat_submission_status(state, submitted_save_id)
        if not status["can_submit"]:
            if status["reason"] == "no_save":
                raise HTTPException(status_code=400, detail="No save loaded")
            raise HTTPException(status_code=409, detail=_CHAT_TURN_ACTIVE_DETAIL)

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": "Looking around"})
            retry_callback, flush_retry_progress = _retry_progress_callback(
                handle,
                task_label="chat",
            )
            kwargs: dict[str, Any] = {
                "query": query,
                "active_save_id": submitted_save_id,
            }
            if _call_accepts_keyword(state.runtime.look_around, "current_user_id"):
                kwargs["current_user_id"] = current_user_id
            if _call_accepts_keyword(
                state.runtime.look_around,
                "retry_progress_callback",
            ):
                kwargs["retry_progress_callback"] = retry_callback
            try:
                return await state.runtime.look_around(**kwargs)
            finally:
                await flush_retry_progress()

        return await _create_job_summary(
            state,
            "look_around",
            worker,
            save_id=submitted_save_id,
            exclusive_key=_chat_turn_exclusive_key(submitted_save_id),
            lock_runtime=False,
        )

    @app.post("/api/chat/timeskip")
    async def submit_timeskip(
        payload: TimeskipRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        instruction = payload.instruction.strip()
        if not instruction:
            raise HTTPException(
                status_code=400,
                detail="Timeskip instruction is required",
            )
        async with state.lock.async_access():
            submitted_save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, submitted_save_id, "mutate")
            current_user_id = _owner_user_id_for_request(state)
        status = _chat_submission_status(state, submitted_save_id)
        request_fingerprint = _chat_turn_request_fingerprint(
            "timeskip",
            {"instruction": instruction},
        )
        replay = _chat_turn_submission_replay(
            state,
            save_id=submitted_save_id,
            client_turn_id=str(payload.client_turn_id),
            operation="timeskip",
            request_fingerprint=request_fingerprint,
        )
        if replay is not None:
            return replay
        if not status["can_submit"]:
            if status["reason"] == "no_save":
                raise HTTPException(status_code=400, detail="No save loaded")
            raise HTTPException(status_code=409, detail=_CHAT_TURN_ACTIVE_DETAIL)

        async def worker(handle: JobHandle) -> Any:
            initial_progress = _initial_chat_turn_progress("Submitting timeskip")
            await handle.event("progress", initial_progress)
            (
                turn_progress_callback,
                flush_turn_progress,
                latest_turn_progress_jobs,
            ) = _turn_progress_callback(handle, initial_progress)
            retry_callback, flush_retry_progress = _retry_progress_callback(
                handle,
                task_label="chat",
            )
            kwargs: dict[str, Any] = {
                "instruction": instruction,
                "active_save_id": submitted_save_id,
            }
            await _wait_for_background_post_turn_catchup(
                state,
                handle,
                save_id=submitted_save_id,
            )

            if _call_accepts_keyword(
                state.runtime.submit_timeskip_for_initial_render,
                "current_user_id",
            ):
                kwargs["current_user_id"] = current_user_id
            if _call_accepts_keyword(
                state.runtime.submit_timeskip_for_initial_render,
                "retry_progress_callback",
            ):
                kwargs["retry_progress_callback"] = retry_callback
            if _call_accepts_keyword(
                state.runtime.submit_timeskip_for_initial_render,
                "turn_progress_callback",
            ):
                kwargs["turn_progress_callback"] = turn_progress_callback
            try:
                turn = await state.runtime.submit_timeskip_for_initial_render(
                    **kwargs
                )
            finally:
                await flush_retry_progress()
                await flush_turn_progress()
            _link_chat_turn_submission_messages(state, handle, turn)
            _raise_for_initial_chat_turn_failure(
                turn,
                failure_message="Timeskip did not produce a narrator response",
            )
            await _emit_initial_chat_turn_event(handle, turn)
            await handle.advance_completion_level(RESPONSE_COMMITTED)
            if turn.has_post_turn_jobs:
                queued = await _queue_post_turn_jobs_background(
                    state,
                    handle,
                    save_id=turn.save_id or "",
                    player_message_id=turn.player_message_id or "",
                    narrator_message_id=turn.narrator_message_id or "",
                    turn_revision=getattr(turn, "turn_revision", None),
                    prepared_action_choices=getattr(
                        turn,
                        "prepared_action_choices",
                        None,
                    ),
                    prior_phase_jobs=_completed_chat_turn_phase_jobs(
                        latest_turn_progress_jobs()
                    ),
                    current_user_id=current_user_id,
                )
                if queued is None:
                    return await _run_post_turn_jobs_inline_fallback(
                        state,
                        handle,
                        save_id=turn.save_id or "",
                        player_message_id=turn.player_message_id or "",
                        narrator_message_id=turn.narrator_message_id or "",
                        turn_revision=getattr(turn, "turn_revision", None),
                        prepared_action_choices=getattr(
                            turn,
                            "prepared_action_choices",
                            None,
                        ),
                        prior_phase_jobs=_completed_chat_turn_phase_jobs(
                            latest_turn_progress_jobs()
                        ),
                        current_user_id=current_user_id,
                    )
                return _initial_chat_turn_result(turn)
            return _initial_chat_turn_result(turn)

        return await _create_idempotent_chat_job_summary(
            state,
            worker,
            save_id=submitted_save_id,
            client_turn_id=str(payload.client_turn_id),
            operation="timeskip",
            request_fingerprint=request_fingerprint,
        )

    @app.post("/api/chat/cancel")
    def cancel_chat(payload: SaveScopedRequest, state: StateDep) -> dict[str, Any]:
        save_id = _require_save_id(payload.save_id)
        with state.lock:
            _raise_unless_chat_cancel_allowed(state, save_id)
        cancelled = state.runtime.cancel_active_submit(
            save_id=save_id,
        )
        return {"cancelled": cancelled}

    @app.post("/api/action-choices/regenerate")
    async def regenerate_action_choices(
        payload: MessageRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            action_save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, action_save_id, "chat")
            status = _chat_submission_status(state, action_save_id)
            if not status["can_submit"]:
                raise HTTPException(status_code=409, detail=_CHAT_TURN_ACTIVE_DETAIL)

        await _cancel_action_choice_jobs_for_save(
            state,
            action_save_id,
            narrator_message_id=payload.message_id,
        )

        return await _create_action_choice_job_summary(
            state,
            save_id=action_save_id,
            narrator_message_id=payload.message_id,
            current_user_id=_owner_user_id_for_request(state),
            job_type="action_choice_regenerate",
            progress_label="Regenerating options",
        )


    async def _create_action_choice_job_summary(
        state: WebAppState,
        *,
        save_id: str,
        narrator_message_id: str,
        current_user_id: str | None,
        job_type: str,
        progress_label: str,
    ) -> dict[str, Any]:
        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": progress_label})
            retry_callback, flush_retry_progress = _retry_progress_callback(
                handle,
                task_label="action choices",
            )
            regenerate = state.runtime.regenerate_action_choices
            regenerate_kwargs: dict[str, object] = {
                "narrator_message_id": narrator_message_id,
                "active_save_id": save_id,
            }
            if _call_accepts_keyword(regenerate, "current_user_id"):
                regenerate_kwargs["current_user_id"] = current_user_id
            if _call_accepts_keyword(regenerate, "retry_progress_callback"):
                regenerate_kwargs["retry_progress_callback"] = retry_callback
            try:
                result = await regenerate(**regenerate_kwargs)
            finally:
                await flush_retry_progress()
            error = _runtime_model_error(result)
            if error:
                raise RuntimeError(error)
            payload_dict = _runtime_json_dict(state, result)
            _publish_runtime_changed_from_model_result(
                state,
                payload_dict,
                reason="action_choices",
            )
            return result

        return await _create_job_summary(
            state,
            job_type,
            worker,
            save_id=save_id,
            operation_queue_key=_action_choice_operation_queue_key(
                save_id,
                narrator_message_id,
            ),
        )

    @app.get("/api/messages/{message_id}/scene-presence")
    def message_scene_presence(
        message_id: str,
        state: StateDep,
        save_id: str | None = None,
    ) -> dict[str, Any]:
        with state.lock:
            resolved_save_id = _require_save_id(save_id)
            _raise_unless_save_action_allowed(state, resolved_save_id, "read")
            return _json_dict(
                _build_scene_presence_model_for_save(
                    state,
                    save_id=resolved_save_id,
                    message_id=message_id,
                )
            )

    @app.post("/api/messages/{message_id}/scene-presence")
    def update_message_scene_presence(
        message_id: str,
        payload: ScenePresenceRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        with state.lock:
            save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, save_id, "mutate")
            model = _build_scene_presence_model_for_save(
                state,
                save_id=save_id,
                message_id=message_id,
            )
            character_ids = _validated_scene_presence_character_ids(
                state,
                save_id=save_id,
                character_ids=payload.character_ids,
            )
            state.repositories.replace_message_scene_presence(
                save_id,
                message_id,
                character_ids,
                source="manual",
            )
            if bool(getattr(model, "latest_message", False)):
                _replace_current_scene_snapshot_presence(
                    state,
                    save_id=save_id,
                    message_id=message_id,
                    character_ids=character_ids,
                )
            return _json_dict(
                _build_scene_presence_model_for_save(
                    state,
                    save_id=save_id,
                    message_id=message_id,
                )
            )

    @app.post("/api/chat/regenerate")
    async def regenerate_message(
        payload: RegenerateMessageRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            action_save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, action_save_id, "mutate")
            current_user_id = _owner_user_id_for_request(state)
            status = _chat_submission_status(state, action_save_id)
            if not status["can_submit"]:
                raise HTTPException(status_code=409, detail=_CHAT_TURN_ACTIVE_DETAIL)

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": "Regenerating message"})
            retry_callback, flush_retry_progress = _retry_progress_callback(
                handle,
                task_label="chat",
            )
            kwargs: dict[str, Any] = {
                "message_id": payload.message_id,
                "active_save_id": action_save_id,
                "regeneration_feedback": payload.regeneration_feedback,
            }
            if _call_accepts_keyword(
                state.runtime.regenerate_message,
                "current_user_id",
            ):
                kwargs["current_user_id"] = current_user_id
            if _call_accepts_keyword(
                state.runtime.regenerate_message,
                "retry_progress_callback",
            ):
                kwargs["retry_progress_callback"] = retry_callback
            try:
                result = await state.runtime.regenerate_message(**kwargs)
            finally:
                await flush_retry_progress()
            _raise_for_runtime_chat_failure(
                result,
                failure_message="Regeneration did not produce a narrator response",
            )
            return result

        return await _create_job_summary(
            state,
            "chat_regenerate",
            worker,
            save_id=action_save_id,
            exclusive_key=_chat_turn_exclusive_key(action_save_id),
            operation_queue_key=action_save_id,
        )

    @app.post("/api/runtime/custom-instructions")
    def update_custom_instructions(
        payload: CustomInstructionsRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        with state.lock:
            save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, save_id, "mutate")
            model = state.runtime.update_custom_instructions(
                custom_instructions=payload.custom_instructions,
                active_save_id=save_id,
            )
            payload_dict = _runtime_json_dict(state, model)
        _publish_runtime_changed_from_model_result(
            state,
            payload_dict,
            reason="custom_instructions_updated",
        )
        return payload_dict

    @app.post("/api/chat/edit")
    async def edit_and_resubmit(
        payload: EditMessageRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            action_save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, action_save_id, "mutate")
            current_user_id = _owner_user_id_for_request(state)

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": "Replaying from edit"})
            retry_callback, flush_retry_progress = _retry_progress_callback(
                handle,
                task_label="chat",
            )
            kwargs: dict[str, Any] = {
                "message_id": payload.message_id,
                "body": payload.body,
                "active_save_id": action_save_id,
            }
            if _call_accepts_keyword(
                state.runtime.edit_and_resubmit_message,
                "current_user_id",
            ):
                kwargs["current_user_id"] = current_user_id
            if _call_accepts_keyword(
                state.runtime.edit_and_resubmit_message,
                "retry_progress_callback",
            ):
                kwargs["retry_progress_callback"] = retry_callback
            try:
                result = await state.runtime.edit_and_resubmit_message(**kwargs)
            finally:
                await flush_retry_progress()
            _raise_for_runtime_chat_failure(
                result,
                failure_message="Edited turn did not produce a narrator response",
            )
            return result

        return await _create_job_summary(
            state,
            "chat_edit",
            worker,
            save_id=action_save_id,
            exclusive_key=_chat_turn_exclusive_key(action_save_id),
            operation_queue_key=action_save_id,
        )

    @app.post("/api/chat/message-edit")
    async def edit_message_without_resubmit(
        payload: EditMessageRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            action_save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, action_save_id, "mutate")
            current_user_id = _owner_user_id_for_request(state)

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": "Saving edit"})
            revision_event_tasks: list[asyncio.Task[None]] = []

            async def publish_revision_committed(model: object) -> None:
                await handle.event("runtime", model)
                await handle.event("progress", {"label": "Reconciling world data"})

            def on_revision_committed(model: object) -> None:
                revision_event_tasks.append(
                    asyncio.create_task(publish_revision_committed(model))
                )

            try:
                kwargs: dict[str, Any] = {
                    "message_id": payload.message_id,
                    "body": payload.body,
                    "active_save_id": action_save_id,
                    "on_revision_committed": on_revision_committed,
                }
                if _call_accepts_keyword(
                    state.runtime.edit_message_without_resubmit,
                    "current_user_id",
                ):
                    kwargs["current_user_id"] = current_user_id
                result = await state.runtime.edit_message_without_resubmit(**kwargs)
            finally:
                if revision_event_tasks:
                    await asyncio.gather(*revision_event_tasks)
            error = _runtime_model_error(result)
            if error:
                raise RuntimeError(error)
            return result

        return await _create_job_summary(
            state,
            "message_edit",
            worker,
            save_id=action_save_id,
            exclusive_key=_chat_turn_exclusive_key(action_save_id),
            operation_queue_key=action_save_id,
        )

    @app.post("/api/chat/narrator-edit")
    async def edit_narrator_message(
        payload: NarratorEditMessageRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            action_save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, action_save_id, "mutate")
            current_user_id = _owner_user_id_for_request(state)

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": "Saving edit"})
            revision_event_tasks: list[asyncio.Task[None]] = []

            async def publish_revision_committed(model: object) -> None:
                await handle.event("runtime", model)
                await handle.event("progress", {"label": "Reconciling world data"})

            def on_revision_committed(model: object) -> None:
                revision_event_tasks.append(
                    asyncio.create_task(publish_revision_committed(model))
                )

            try:
                kwargs: dict[str, Any] = {
                    "message_id": payload.message_id,
                    "body": payload.body,
                    "active_save_id": action_save_id,
                    "on_revision_committed": on_revision_committed,
                }
                if _call_accepts_keyword(
                    state.runtime.edit_narrator_message,
                    "current_user_id",
                ):
                    kwargs["current_user_id"] = current_user_id
                result = await state.runtime.edit_narrator_message(**kwargs)
            finally:
                if revision_event_tasks:
                    await asyncio.gather(*revision_event_tasks)
            error = _runtime_model_error(result)
            if error:
                raise RuntimeError(error)
            return result

        return await _create_job_summary(
            state,
            "narrator_edit",
            worker,
            save_id=action_save_id,
            exclusive_key=_chat_turn_exclusive_key(action_save_id),
            operation_queue_key=action_save_id,
        )

    @app.post("/api/chat/delete-from-here")
    async def delete_messages_from_here(
        payload: DeleteMessagesFromHereRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, save_id, "mutate")

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": "Deleting messages"})
            async with state.lock.async_access():
                model = state.runtime.delete_messages_from_here(
                    message_id=payload.message_id,
                    active_save_id=save_id,
                )
                payload_dict = _runtime_json_dict(state, model)
            _publish_runtime_changed_from_model_result(
                state,
                payload_dict,
                reason="messages_deleted",
            )
            return payload_dict

        return await _create_job_summary(
            state,
            "chat_delete_from_here",
            worker,
            save_id=save_id,
            operation_queue_key=save_id,
        )

    @app.post("/api/chat/fork-from-here")
    async def fork_from_here(
        payload: ForkFromHereRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, save_id, "mutate")
            owner_user_id = _owner_user_id_for_request(state)
            remember_process_active_save = not _auth_context_enabled(state)

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": "Forking save"})
            async with state.lock.async_access():
                kwargs: dict[str, Any] = {}
                if _call_accepts_keyword(
                    state.runtime.fork_save_from_message,
                    "owner_user_id",
                ):
                    kwargs["owner_user_id"] = owner_user_id
                if _call_accepts_keyword(
                    state.runtime.fork_save_from_message,
                    "remember_process_active_save",
                ):
                    kwargs["remember_process_active_save"] = (
                        remember_process_active_save
                    )
                model = state.runtime.fork_save_from_message(
                    message_id=payload.message_id,
                    active_save_id=save_id,
                    **kwargs,
                )
                payload_dict = _runtime_json_dict(state, model)
                _remember_user_active_save_from_model_result(state, payload_dict)
            _publish_runtime_changed_from_model_result(
                state,
                payload_dict,
                reason="save_forked",
            )
            _publish_save_event(state, None, "saves_changed", {"reason": "save_forked"})
            return payload_dict

        return await _create_job_summary(
            state,
            "chat_fork_from_here",
            worker,
            save_id=save_id,
            operation_queue_key=save_id,
        )

    @app.post("/api/media/generate")
    async def generate_image(
        payload: MediaMessageRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            media_save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, media_save_id, "media_generate")
            current_user_id = _owner_user_id_for_request(state)

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": "Generating image"})
            retry_callback, flush_retry_progress = _retry_progress_callback(
                handle,
                task_label="image",
            )
            kwargs: dict[str, Any] = {
                "source_message_id": payload.message_id,
                "active_save_id": media_save_id,
            }
            if _call_accepts_keyword(
                state.runtime.generate_image,
                "retry_progress_callback",
            ):
                kwargs["retry_progress_callback"] = retry_callback
            if _call_accepts_keyword(state.runtime.generate_image, "current_user_id"):
                kwargs["current_user_id"] = current_user_id
            try:
                return _runtime_media_json_or_raise(
                    state,
                    await state.runtime.generate_image(**kwargs),
                )
            finally:
                await flush_retry_progress()

        return await _create_job_summary(
            state,
            "image_generation",
            worker,
            save_id=media_save_id,
        )

    @app.post("/api/media/generate-character-image")
    async def generate_character_image(
        payload: CharacterMediaMessageRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            media_save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, media_save_id, "media_generate")
            current_user_id = _owner_user_id_for_request(state)

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": "Generating character image"})
            retry_callback, flush_retry_progress = _retry_progress_callback(
                handle,
                task_label="image",
            )
            kwargs: dict[str, Any] = {
                "source_message_id": payload.message_id,
                "character_id": payload.character_id,
                "active_save_id": media_save_id,
            }
            if _call_accepts_keyword(
                state.runtime.generate_character_image,
                "retry_progress_callback",
            ):
                kwargs["retry_progress_callback"] = retry_callback
            if _call_accepts_keyword(
                state.runtime.generate_character_image,
                "current_user_id",
            ):
                kwargs["current_user_id"] = current_user_id
            try:
                return _runtime_media_json_or_raise(
                    state,
                    await state.runtime.generate_character_image(**kwargs),
                )
            finally:
                await flush_retry_progress()

        return await _create_job_summary(
            state,
            "character_image_generation",
            worker,
            save_id=media_save_id,
        )

    @app.post("/api/media/initial")
    async def generate_initial_image(
        payload: InitialImageRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            media_save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, media_save_id, "media_generate")
            current_user_id = _owner_user_id_for_request(state)

        async def worker(handle: JobHandle) -> Any:
            await handle.event(
                "progress",
                {"label": _initial_media_progress_label(state, media_save_id)},
            )
            retry_callback, flush_retry_progress = _retry_progress_callback(
                handle,
                task_label="image",
            )
            kwargs: dict[str, Any] = {
                "source_message_id": payload.message_id,
                "active_save_id": media_save_id,
            }
            if _call_accepts_keyword(
                state.runtime.generate_initial_scenario_image,
                "retry_progress_callback",
            ):
                kwargs["retry_progress_callback"] = retry_callback
            if _call_accepts_keyword(
                state.runtime.generate_initial_scenario_image,
                "current_user_id",
            ):
                kwargs["current_user_id"] = current_user_id
            try:
                return _runtime_media_json_or_raise(
                    state,
                    await state.runtime.generate_initial_scenario_image(**kwargs),
                )
            finally:
                await flush_retry_progress()

        return await _create_job_summary(
            state,
            "initial_image_generation",
            worker,
            save_id=media_save_id,
        )

    @app.post("/api/media/character-reference/upload")
    async def upload_character_reference_image(
        state: StateDep,
        file: Annotated[UploadFile, File()],
        save_id: Annotated[str | None, Form()] = None,
        character_id: Annotated[str | None, Form()] = None,
        replace_existing: Annotated[bool, Form()] = False,
    ) -> dict[str, Any]:
        resolved_save_id = _require_save_id(save_id)
        _raise_unless_save_action_allowed(state, resolved_save_id, "media")
        _raise_unless_unrated_reference_upload(state)
        try:
            image_bytes = await _read_limited_character_reference_upload(file)
        except _CharacterReferenceUploadTooLarge as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": "Uploading reference image"})
            async with state.lock.async_access():
                kwargs: dict[str, Any] = {}
                if character_id is not None:
                    kwargs["character_id"] = character_id
                model = state.runtime.upload_character_reference_image(
                    image_bytes=image_bytes,
                    filename=file.filename,
                    replace_existing=replace_existing,
                    active_save_id=resolved_save_id,
                    **kwargs,
                )
                error = _runtime_model_error(model)
                if error:
                    raise RuntimeError(error)
                payload = _runtime_json_dict(state, model)
            _publish_runtime_changed_from_model_result(
                state,
                payload,
                reason="character_reference_uploaded",
            )
            return payload

        return await _create_job_summary(
            state,
            "character_reference_upload",
            worker,
            save_id=resolved_save_id,
            operation_queue_key=resolved_save_id,
        )

    @app.post("/api/media/character-reference/remove")
    def remove_character_reference_image(
        payload: CharacterReferenceRouteRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        with state.lock:
            save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, save_id, "media")
            kwargs: dict[str, Any] = {}
            if payload.character_id is not None:
                kwargs["character_id"] = payload.character_id
            model = state.runtime.remove_character_reference_image(
                active_save_id=save_id,
                **kwargs,
            )
        error = _runtime_model_error(model)
        if error:
            status_code = 404 if "not found" in error.lower() else 400
            raise HTTPException(status_code=status_code, detail=error)
        payload_dict = _runtime_json_dict(state, model)
        _publish_runtime_changed_from_model_result(
            state,
            payload_dict,
            reason="character_reference_removed",
        )
        return payload_dict

    @app.get("/api/media/{asset_id}/prompt")
    async def media_asset_prompt(
        asset_id: str,
        state: StateDep,
        save_id: str | None = None,
    ) -> dict[str, str]:
        async with state.lock.async_access():
            resolved_save_id = _require_save_id(save_id)
            _raise_unless_save_action_allowed(
                state,
                resolved_save_id,
                "media_generate",
            )
            current_user = _save_access_user(state)
            asset = _find_media_asset_unlocked(
                state,
                asset_id,
                save_id=resolved_save_id,
            )
        if asset is None:
            raise HTTPException(status_code=404, detail="Unknown media asset")
        _raise_if_media_asset_exceeds_request_rating(
            state,
            asset,
            save_id=resolved_save_id,
        )
        if getattr(asset, "type", None) != "image":
            raise HTTPException(
                status_code=400,
                detail="Only image media assets expose editable prompts",
            )
        if getattr(asset, "provider", None) == "local":
            raise HTTPException(
                status_code=400,
                detail="Only generated image media assets expose editable prompts",
            )
        if getattr(asset, "status", None) != "succeeded":
            raise HTTPException(
                status_code=400,
                detail="Only succeeded image media assets expose editable prompts",
            )
        prompt = str(getattr(asset, "prompt", "") or "")
        if current_user is not None:
            from bragi.services.content_rating import effective_content_safety_policy

            policy = effective_content_safety_policy(
                state.repositories,
                user_id=current_user.id,
            )
            if _media_asset_exceeds_rating_for_request(
                state,
                asset,
                save_id=resolved_save_id,
                allowed_rating=policy.rating,
            ):
                raise HTTPException(
                    status_code=403,
                    detail="Media prompt exceeds your content rating",
                )
        return {
            "media_asset_id": asset_id,
            "prompt": prompt,
        }

    @app.post("/api/media/{asset_id}/animate")
    async def animate_image_asset(
        asset_id: str,
        payload: AnimateMediaRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            media_save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, media_save_id, "media_generate")
            current_user_id = _owner_user_id_for_request(state)
            asset = _find_media_asset_unlocked(
                state,
                asset_id,
                save_id=media_save_id,
            )
        if asset is None:
            raise HTTPException(status_code=404, detail="Unknown media asset")
        _raise_if_media_asset_exceeds_request_rating(
            state,
            asset,
            save_id=media_save_id,
        )
        if getattr(asset, "type", None) != "image":
            raise HTTPException(
                status_code=400,
                detail="Only image media assets can be animated",
            )
        if getattr(asset, "source_message_id", None) is None:
            raise HTTPException(
                status_code=400,
                detail="Media asset has no source message",
            )

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": "Animating image"})
            return _runtime_media_json_or_raise(
                state,
                await state.runtime.animate_media_asset(
                    asset_id,
                    motion_prompt=payload.motion_prompt,
                    active_save_id=media_save_id,
                    **(
                        {"current_user_id": current_user_id}
                        if _call_accepts_keyword(
                            state.runtime.animate_media_asset,
                            "current_user_id",
                        )
                        else {}
                    ),
                ),
            )

        return await _create_job_summary(
            state,
            "image_animation",
            worker,
            save_id=media_save_id,
        )

    @app.post("/api/media/{asset_id}/regenerate")
    async def regenerate_image_from_asset(
        asset_id: str,
        payload: RegenerateMediaRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            media_save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, media_save_id, "media_generate")
            current_user_id = _owner_user_id_for_request(state)
            asset = _find_media_asset_unlocked(
                state,
                asset_id,
                save_id=media_save_id,
            )
            current_user = _save_access_user(state)
            if (
                asset is not None
                and current_user is not None
                and current_user.role == "child"
                and _media_asset_is_character_reference_unlocked(
                    state,
                    save_id=media_save_id,
                    media_asset_id=asset_id,
                )
            ):
                _raise_unless_save_action_allowed(state, media_save_id, "media")
        if asset is None:
            raise HTTPException(status_code=404, detail="Unknown media asset")
        if getattr(asset, "type", None) != "image":
            raise HTTPException(
                status_code=400,
                detail="Only image media assets can be regenerated",
            )
        if getattr(asset, "provider", None) == "local":
            raise HTTPException(
                status_code=400,
                detail="Only generated image media assets can be regenerated",
            )
        if getattr(asset, "status", None) != "succeeded":
            raise HTTPException(
                status_code=400,
                detail="Only succeeded image media assets can be regenerated",
            )
        edited_prompt = payload.prompt.strip() if payload.prompt is not None else None
        if payload.prompt is not None and not edited_prompt:
            raise HTTPException(status_code=400, detail="Image prompt is required")
        if edited_prompt is None and asset.source_message_id is None:
            raise HTTPException(
                status_code=400,
                detail="Media asset has no source message",
            )

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": "Regenerating image"})
            retry_callback, flush_retry_progress = _retry_progress_callback(
                handle,
                task_label="image",
            )
            if edited_prompt is None:
                runtime_call = state.runtime.generate_image
                kwargs: dict[str, Any] = {
                    "source_message_id": asset.source_message_id,
                    "active_save_id": media_save_id,
                }
            else:
                runtime_call = state.runtime.regenerate_media_asset
                kwargs = {
                    "media_asset_id": asset_id,
                    "prompt": edited_prompt,
                    "active_save_id": media_save_id,
                }
            if _call_accepts_keyword(runtime_call, "retry_progress_callback"):
                kwargs["retry_progress_callback"] = retry_callback
            if _call_accepts_keyword(runtime_call, "current_user_id"):
                kwargs["current_user_id"] = current_user_id
            try:
                return _runtime_media_json_or_raise(
                    state,
                    await runtime_call(**kwargs),
                )
            finally:
                await flush_retry_progress()

        return await _create_job_summary(
            state,
            "image_regeneration",
            worker,
            save_id=media_save_id,
        )

    @app.post("/api/media/{asset_id}/set-character-reference")
    async def set_character_reference_image(
        asset_id: str,
        payload: CharacterReferenceRouteRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, save_id, "media")
            asset = _find_media_asset_unlocked(state, asset_id, save_id=save_id)
        if asset is None:
            raise HTTPException(status_code=404, detail="Unknown media asset")

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": "Setting reference image"})
            async with state.lock.async_access():
                kwargs: dict[str, Any] = {}
                if payload.character_id is not None:
                    kwargs["character_id"] = payload.character_id
                model = state.runtime.set_character_reference_image(
                    asset_id,
                    active_save_id=save_id,
                    **kwargs,
                )
                error = _runtime_model_error(model)
                if error:
                    raise RuntimeError(error)
                payload_dict = _runtime_json_dict(state, model)
            _publish_runtime_changed_from_model_result(
                state,
                payload_dict,
                reason="character_reference_set",
            )
            return payload_dict

        return await _create_job_summary(
            state,
            "character_reference_set",
            worker,
            save_id=save_id,
            operation_queue_key=save_id,
        )

    @app.delete("/api/media/{asset_id}")
    async def delete_media_asset(
        asset_id: str,
        state: StateDep,
        save_id: str | None = None,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            resolved_save_id = _require_save_id(save_id)
            _raise_unless_save_action_allowed(state, resolved_save_id, "media")
            asset = _find_media_asset_unlocked(
                state,
                asset_id,
                save_id=resolved_save_id,
            )
        if asset is None:
            raise HTTPException(status_code=404, detail="Unknown media asset")

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": "Deleting media"})
            async with state.lock.async_access():
                model = state.runtime.delete_media_asset(
                    asset_id,
                    active_save_id=resolved_save_id,
                )
                error = _runtime_model_error(model)
                if error:
                    raise RuntimeError(error)
                payload = _runtime_json_dict(state, model)
            _publish_runtime_changed_from_model_result(
                state,
                payload,
                reason="media_deleted",
            )
            return payload

        return await _create_job_summary(
            state,
            "media_delete",
            worker,
            save_id=resolved_save_id,
            operation_queue_key=resolved_save_id,
        )

    @app.get("/api/media/{asset_id}")
    def media_asset(
        asset_id: str,
        state: StateDep,
        save_id: str | None = None,
    ) -> FileResponse:
        with state.lock:
            resolved_save_id = _require_save_id(save_id)
            _raise_unless_save_action_allowed(state, resolved_save_id, "read")
            asset = _find_media_asset_unlocked(
                state,
                asset_id,
                save_id=resolved_save_id,
            )
            if asset is not None:
                _raise_if_media_asset_exceeds_request_rating(
                    state,
                    asset,
                    save_id=resolved_save_id,
                )
        if asset is None:
            raise HTTPException(status_code=404, detail="Unknown media asset")
        path = _media_path_within_root(state, getattr(asset, "path", ""))
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="Media file not found")
        return _media_file_response(
            path,
            media_type=safe_served_media_mime_type(asset.mime_type),
        )

    @app.get("/api/media/{asset_id}/thumbnail")
    def media_asset_thumbnail(
        asset_id: str,
        state: StateDep,
        save_id: str | None = None,
    ) -> FileResponse:
        with state.lock:
            resolved_save_id = _require_save_id(save_id)
            _raise_unless_save_action_allowed(state, resolved_save_id, "read")
            asset = _find_media_asset_unlocked(
                state,
                asset_id,
                save_id=resolved_save_id,
            )
            if asset is not None:
                _raise_if_media_asset_exceeds_request_rating(
                    state,
                    asset,
                    save_id=resolved_save_id,
                )
        if asset is None:
            raise HTTPException(status_code=404, detail="Unknown media asset")
        thumbnail_path = getattr(asset, "thumbnail_path", None)
        if isinstance(thumbnail_path, str) and thumbnail_path:
            thumbnail = _media_path_within_root(state, thumbnail_path)
            if thumbnail is None:
                raise HTTPException(
                    status_code=404,
                    detail="Media file not found",
                )
            if thumbnail.is_file() and not _is_unusable_fallback_thumbnail(
                thumbnail,
            ):
                return _media_file_response(thumbnail, media_type="image/png")

        if not _is_image_media_asset(asset):
            raise HTTPException(
                status_code=404,
                detail="Media thumbnail not found",
            )
        path = _media_path_within_root(state, getattr(asset, "path", ""))
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="Media file not found")
        return _media_file_response(
            path,
            media_type=safe_served_media_mime_type(asset.mime_type),
        )

    @app.get("/api/world-data")
    def world_data(state: StateDep, save_id: str | None = None) -> dict[str, Any]:
        with state.lock:
            return _json_dict(_build_world_data_model_for_save(state, save_id))

    @app.post("/api/world-data/apply")
    async def apply_world_data(
        payload: WorldDataApplyRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        try:
            edits = _world_data_edits_from_json(payload.edits)
            async with state.lock.async_access():
                resolved_save_id = _require_save_id(payload.active_save_id)
                _raise_unless_save_action_allowed(state, resolved_save_id, "mutate")
                actor_user_id = _owner_user_id_for_request(state)
                if edits.scenario is not None:
                    details = state.repositories.load_save_details(resolved_save_id)
                    if details is None:
                        raise ValueError(f"Unknown save id: {resolved_save_id}")
                    reviewed_scenario = await _review_scenario_edit_for_request(
                        state,
                        cast(ScenarioEdit, edits.scenario),
                        save_id=resolved_save_id,
                        roleplay_type=details.scenario.type,
                        current_user_id=actor_user_id,
                    )
                    edits = replace(edits, scenario=reviewed_scenario)
                result = WorldDataService(
                    state.repositories,
                    active_save_id=resolved_save_id,
                    allowed_content_rating=(
                        _content_safety_policy_for_request(state).rating
                    ),
                ).apply_edits(
                    edits,
                    active_save_id=payload.active_save_id
                    if payload.active_save_id is not None
                    else ...,
                )
                payload_dict = _json_dict(result)
            _publish_runtime_changed_from_model_result(
                state,
                payload_dict.get("model"),
                reason="world_data_applied",
            )
            return payload_dict
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/world-data/context-cleanup")
    async def context_cleanup(
        state: StateDep,
        payload: ContextCleanupRequest | None = None,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            save_id = _require_save_id(
                payload.save_id if payload is not None else None,
            )
            _raise_unless_save_action_allowed(state, save_id, "mutate")

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": "Cleaning context"})
            return await state.runtime.run_context_cleanup(active_save_id=save_id)

        return await _create_job_summary(
            state,
            "context_cleanup",
            worker,
            save_id=save_id,
        )

    @app.post("/api/world-data/summary-backfill")
    async def summary_backfill(
        state: StateDep,
        payload: SummaryBackfillRequest | None = None,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            save_id = _require_save_id(
                payload.save_id if payload is not None else None,
            )
            _raise_unless_save_action_allowed(state, save_id, "mutate")
            request_user = _current_request_user()
            apply_recommended_windows = (
                bool(payload.apply_recommended_windows)
                if payload is not None
                else False
            )

        async def worker(handle: JobHandle) -> Any:
            user_token = _REQUEST_USER.set(request_user)
            try:
                await handle.event("progress", {"label": "Compacting history"})
                model = await state.runtime.run_summary_backfill(
                    active_save_id=save_id,
                    apply_recommended_windows=apply_recommended_windows,
                )
                error = _runtime_model_error(model)
                if error:
                    raise RuntimeError(error)
                return _runtime_json_dict(state, model)
            finally:
                _REQUEST_USER.reset(user_token)

        return await _create_job_summary(
            state,
            "summary_backfill",
            worker,
            save_id=save_id,
        )

    @app.post("/api/world-data/suggestion-review")
    async def world_suggestion_review(
        state: StateDep,
        payload: ContextCleanupRequest | None = None,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            save_id = _require_save_id(
                payload.save_id if payload is not None else None,
            )
            _raise_unless_save_action_allowed(state, save_id, "mutate")

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": "Reviewing suggestions"})
            return await state.runtime.run_world_suggestion_review(
                active_save_id=save_id,
            )

        return await _create_job_summary(
            state,
            "world_suggestion_review",
            worker,
            save_id=save_id,
        )

    @app.post("/api/world-data/context-retention")
    async def world_context_retention(
        state: StateDep,
        payload: ContextCleanupRequest | None = None,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            save_id = _require_save_id(
                payload.save_id if payload is not None else None,
            )
            _raise_unless_save_action_allowed(state, save_id, "mutate")

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": "Pruning world history"})
            return await state.runtime.run_world_context_retention(
                active_save_id=save_id,
            )

        return await _create_job_summary(
            state,
            "world_context_retention",
            worker,
            save_id=save_id,
        )

    @app.post("/api/world-data/guided-cleanup")
    async def guided_context_cleanup(
        payload: GuidedContextCleanupRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, save_id, "mutate")

        async def worker(handle: JobHandle) -> Any:
            await handle.event(
                "progress",
                {"label": "Queueing cleanup suggestions"},
            )
            return await state.runtime.run_guided_context_cleanup(
                instruction=payload.instruction,
                active_save_id=save_id,
            )

        return await _create_job_summary(
            state,
            "guided_context_cleanup",
            worker,
            save_id=save_id,
        )

    @app.get("/api/characters")
    def characters(state: StateDep, save_id: str | None = None) -> dict[str, Any]:
        with state.lock:
            payload = _json_dict(
                _build_character_registry_model_for_save(state, save_id)
            )
            _scrub_response_payload_for_request(state, payload)
            return payload

    @app.post("/api/characters/{character_id}/reference-image/generate")
    async def generate_character_reference_image(
        character_id: str,
        payload: CharacterReferenceGenerateRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(
                state,
                save_id,
                "media" if payload.replace_existing else "media_generate",
            )
            current_user_id = _owner_user_id_for_request(state)

        async def worker(handle: JobHandle) -> Any:
            await handle.event(
                "progress",
                {"label": "Generating character reference image"},
            )
            retry_callback, flush_retry_progress = _retry_progress_callback(
                handle,
                task_label="image",
            )
            try:
                return _runtime_media_json_or_raise(
                    state,
                    await state.runtime.generate_character_reference_image(
                        character_id,
                        source_message_id=payload.source_message_id,
                        replace_existing=payload.replace_existing,
                        active_save_id=save_id,
                        retry_progress_callback=retry_callback,
                        **(
                            {"current_user_id": current_user_id}
                            if _call_accepts_keyword(
                                state.runtime.generate_character_reference_image,
                                "current_user_id",
                            )
                            else {}
                        ),
                    ),
                )
            finally:
                await flush_retry_progress()

        return await _create_job_summary(
            state,
            "character_reference_image",
            worker,
            save_id=save_id,
        )

    @app.post("/api/characters/{character_id}/image/generate")
    async def generate_character_registry_image(
        character_id: str,
        payload: CharacterImageGenerateRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, save_id, "media_generate")
            current_user_id = _owner_user_id_for_request(state)

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": "Generating character image"})
            retry_callback, flush_retry_progress = _retry_progress_callback(
                handle,
                task_label="image",
            )
            try:
                return _runtime_media_json_or_raise(
                    state,
                    await state.runtime.generate_character_registry_image(
                        character_id,
                        instructions=payload.instructions,
                        active_save_id=save_id,
                        retry_progress_callback=retry_callback,
                        **(
                            {"current_user_id": current_user_id}
                            if _call_accepts_keyword(
                                state.runtime.generate_character_registry_image,
                                "current_user_id",
                            )
                            else {}
                        ),
                    ),
                )
            finally:
                await flush_retry_progress()

        return await _create_job_summary(
            state,
            "character_image_generation",
            worker,
            save_id=save_id,
        )

    @app.post("/api/characters/{character_id}/reference-image/upload")
    async def upload_character_reference_image_for_character(
        character_id: str,
        state: StateDep,
        file: Annotated[UploadFile, File()],
        save_id: Annotated[str | None, Form()] = None,
        replace_existing: Annotated[bool, Form()] = False,
    ) -> dict[str, Any]:
        resolved_save_id = _require_save_id(save_id)
        _raise_unless_save_action_allowed(state, resolved_save_id, "media")
        _raise_unless_unrated_reference_upload(state)
        try:
            image_bytes = await _read_limited_character_reference_upload(file)
        except _CharacterReferenceUploadTooLarge as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": "Uploading reference image"})
            async with state.lock.async_access():
                model = state.runtime.upload_character_reference_image(
                    image_bytes=image_bytes,
                    filename=file.filename,
                    character_id=character_id,
                    replace_existing=replace_existing,
                    active_save_id=resolved_save_id,
                )
                error = _runtime_model_error(model)
                if error:
                    raise RuntimeError(error)
                payload_dict = _character_registry_json_or_raise(
                    state,
                    save_id=resolved_save_id,
                )
            _publish_runtime_changed_from_model_result(
                state,
                payload_dict,
                reason="character_reference_uploaded",
            )
            return payload_dict

        return await _create_job_summary(
            state,
            "character_reference_upload",
            worker,
            save_id=resolved_save_id,
            operation_queue_key=resolved_save_id,
        )

    @app.post("/api/characters/{character_id}/reference-image/set")
    async def set_character_reference_image_for_character(
        character_id: str,
        payload: CharacterReferenceSetRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            save_id = _require_save_id(payload.save_id)
            _raise_unless_save_action_allowed(state, save_id, "media")
            asset = _find_media_asset_unlocked(
                state,
                payload.media_asset_id,
                save_id=save_id,
            )
        if asset is None:
            raise HTTPException(status_code=404, detail="Unknown media asset")

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": "Setting reference image"})
            async with state.lock.async_access():
                model = state.runtime.set_character_reference_image(
                    payload.media_asset_id,
                    character_id=character_id,
                    active_save_id=save_id,
                )
                error = _runtime_model_error(model)
                if error:
                    raise RuntimeError(error)
                payload_dict = _character_registry_json_or_raise(state, save_id=save_id)
            _publish_runtime_changed_from_model_result(
                state,
                payload_dict,
                reason="character_reference_set",
            )
            return payload_dict

        return await _create_job_summary(
            state,
            "character_reference_set",
            worker,
            save_id=save_id,
            operation_queue_key=save_id,
        )

    @app.post("/api/characters/{character_id}/reference-image/remove")
    def remove_character_reference_image_for_character(
        character_id: str,
        payload: SaveScopedRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        try:
            with state.lock:
                save_id = _require_save_id(payload.save_id)
                _raise_unless_save_action_allowed(state, save_id, "media")
                model = state.runtime.remove_character_reference_image(
                    character_id=character_id,
                    active_save_id=save_id,
                )
                error = _runtime_model_error(model)
                if error:
                    raise ValueError(error)
                payload_dict = _character_registry_json_or_raise(state, save_id=save_id)
            _publish_runtime_changed_from_model_result(
                state,
                payload_dict,
                reason="character_reference_removed",
            )
            return payload_dict
        except HTTPException:
            raise
        except Exception as exc:
            detail = str(exc) or exc.__class__.__name__
            status_code = 404 if "not found" in detail.lower() else 400
            raise HTTPException(status_code=status_code, detail=detail) from exc

    @app.post("/api/characters/apply")
    async def apply_characters(
        payload: CharacterRegistryApplyRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        try:
            edits = CharacterRegistryEdits(
                characters=tuple(
                    _dataclass_from_json(CharacterRegistryRow, row)
                    for row in payload.edits.get("characters", ())
                )
            )
            async with state.lock.async_access():
                resolved_save_id = _require_save_id(payload.active_save_id)
                _raise_unless_save_action_allowed(state, resolved_save_id, "mutate")
                edits = await _review_character_edits_for_request(
                    state,
                    edits,
                    save_id=resolved_save_id,
                    current_user_id=_owner_user_id_for_request(state),
                )
                result = CharacterRegistryService(
                    state.repositories,
                    active_save_id=resolved_save_id,
                ).apply_edits(
                    edits,
                    active_save_id=payload.active_save_id
                    if payload.active_save_id is not None
                    else ...,
                )
                payload_dict = _json_dict(result)
                auto_enhanced_count = 0
                if (
                    payload.auto_enhance_created_agency
                    and result.created_character_ids
                ):
                    complete_agency = getattr(
                        state.runtime,
                        "complete_new_character_agency",
                        None,
                    )
                    if callable(complete_agency):
                        try:
                            completion_kwargs: dict[str, object] = {
                                "active_save_id": resolved_save_id,
                                "character_ids": result.created_character_ids,
                            }
                            if _call_accepts_keyword(
                                complete_agency,
                                "current_user_id",
                            ):
                                completion_kwargs["current_user_id"] = (
                                    _owner_user_id_for_request(state)
                                )
                            auto_enhanced_count = int(
                                complete_agency(**completion_kwargs)
                            )
                        except Exception:
                            observe(
                                "web.character_created_agency_enhancement_failed",
                                level="error",
                                save_id=resolved_save_id,
                                character_count=len(result.created_character_ids),
                            )
                            auto_enhanced_count = 0
                        if auto_enhanced_count:
                            payload_dict["model"] = to_jsonable(
                                CharacterRegistryService(
                                    state.repositories,
                                    active_save_id=resolved_save_id,
                                ).build_model(active_save_id=resolved_save_id)
                            )
                payload_dict["auto_enhanced_count"] = auto_enhanced_count
            _publish_runtime_changed_from_model_result(
                state,
                payload_dict.get("model"),
                reason="characters_applied",
            )
            return payload_dict
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/characters/{character_id}/enhance-field")
    def enhance_character_field(
        character_id: str,
        payload: CharacterFieldEnhanceRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        try:
            row = _dataclass_from_json(CharacterRegistryRow, payload.character)
            with state.lock:
                resolved_save_id = _require_save_id(payload.active_save_id)
                _raise_unless_save_action_allowed(state, resolved_save_id, "mutate")
                enhance_field = getattr(
                    state.runtime,
                    "enhance_character_registry_field",
                    None,
                )
                if not callable(enhance_field):
                    raise ValueError("Character enhancement model is unavailable")
                enhance_kwargs: dict[str, object] = {
                    "active_save_id": resolved_save_id,
                    "character_id": character_id,
                    "field_name": payload.field_name,
                    "row": row,
                }
                if _call_accepts_keyword(enhance_field, "current_user_id"):
                    enhance_kwargs["current_user_id"] = (
                        _owner_user_id_for_request(state)
                    )
                result = enhance_field(**enhance_kwargs)
                payload_dict = _json_dict(result)
            _publish_runtime_changed_from_model_result(
                state,
                payload_dict.get("model"),
                reason="character_field_enhanced",
            )
            return payload_dict
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/characters/{character_id}/knowledge/apply")
    def apply_character_knowledge(
        character_id: str,
        payload: CharacterKnowledgeApplyRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        try:
            actions = tuple(
                _dataclass_from_json(CharacterKnowledgeAction, action)
                for action in payload.actions
            )
            with state.lock:
                resolved_save_id = _require_save_id(payload.active_save_id)
                _raise_unless_save_action_allowed(state, resolved_save_id, "mutate")
                result = CharacterRegistryService(
                    state.repositories,
                    active_save_id=resolved_save_id,
                ).apply_knowledge_actions(
                    character_id=character_id,
                    actions=actions,
                    active_save_id=payload.active_save_id
                    if payload.active_save_id is not None
                    else ...,
                )
                payload_dict = _json_dict(result)
            _publish_runtime_changed_from_model_result(
                state,
                payload_dict.get("model"),
                reason="character_knowledge_applied",
            )
            return payload_dict
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/character-bundles/export/{character_id}")
    def export_character_bundle(
        character_id: str,
        state: StateDep,
        include_private_notes: bool = False,
    ) -> FileResponse:
        bundle_path = (
            state.paths.temp_dir
            / f"bragi-character-{uuid4().hex}.bragi-character"
        )
        try:
            with state.lock:
                character_save_id = _raise_unless_character_export_allowed(
                    state,
                    character_id,
                )
                if include_private_notes:
                    _raise_unless_character_private_notes_export_allowed(state)
                kwargs: dict[str, Any] = {}
                if (
                    character_save_id is not None
                    and _call_accepts_keyword(
                        state.runtime.export_character_bundle,
                        "active_save_id",
                    )
                ):
                    kwargs["active_save_id"] = character_save_id
                if _call_accepts_keyword(
                    state.runtime.export_character_bundle,
                    "include_private_notes",
                ):
                    kwargs["include_private_notes"] = include_private_notes
                model = state.runtime.export_character_bundle(
                    character_id,
                    bundle_path,
                    **kwargs,
                )
            error = _runtime_model_error(model)
            if error:
                raise HTTPException(status_code=400, detail=error)
            return FileResponse(
                bundle_path,
                media_type="application/octet-stream",
                filename=bundle_path.name,
                background=BackgroundTask(_unlink_file, bundle_path),
            )
        except HTTPException:
            bundle_path.unlink(missing_ok=True)
            raise
        except Exception:
            bundle_path.unlink(missing_ok=True)
            raise

    @app.post("/api/character-bundles/preview")
    async def preview_character_bundle(
        file: Annotated[UploadFile, File()],
        state: StateDep,
        active_save_id: Annotated[str | None, Form()] = None,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            _raise_unless_import_export_allowed(state)
            target_save_id = _require_save_id(active_save_id)
            _raise_unless_save_action_allowed(state, target_save_id, "mutate")
            _prune_character_bundle_previews(state)
        try:
            bundle_path = await _store_upload(file, state.paths.temp_dir)
        except _BundleUploadTooLarge as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        preview_id = uuid4().hex
        try:
            async with state.lock.async_access():
                preview = state.runtime.preview_import_character_bundle(
                    bundle_path,
                    target_save_id=target_save_id,
                )
        except HTTPException:
            bundle_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            bundle_path.unlink(missing_ok=True)
            detail = str(exc) or exc.__class__.__name__
            raise HTTPException(status_code=400, detail=detail) from exc
        async with state.lock.async_access():
            state.character_bundle_previews[preview_id] = BundlePreviewState(
                bundle_path=bundle_path,
                owner_user_id=_owner_user_id_for_request(state),
                target_save_id=target_save_id,
            )
            _prune_character_bundle_previews(state)
        return {
            "preview_id": preview_id,
            "preview": await _review_bundle_preview_for_request(
                state,
                preview,
                save_id=target_save_id,
            ),
        }

    @app.post("/api/character-bundles/import/{preview_id}")
    def import_character_bundle(
        preview_id: str,
        payload: CharacterBundleImportRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        with state.lock:
            _raise_unless_import_export_allowed(state)
            _prune_character_bundle_previews(state)
            preview = _require_owned_bundle_preview(
                state,
                state.character_bundle_previews,
                preview_id,
                detail="Unknown character bundle preview",
            )
            target_save_id = _require_save_id(payload.active_save_id)
            _raise_unless_save_action_allowed(state, target_save_id, "mutate")
            if preview.target_save_id and preview.target_save_id != target_save_id:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Character bundle preview target save does not match request"
                    ),
                )
            state.character_bundle_previews.pop(preview_id, None)
        try:
            with state.lock:
                kwargs: dict[str, Any] = {}
                if _call_accepts_keyword(
                    state.runtime.import_character_bundle,
                    "remember_process_active_save",
                ):
                    kwargs["remember_process_active_save"] = (
                        not _auth_context_enabled(state)
                    )
                model = state.runtime.import_character_bundle(
                    preview.bundle_path,
                    target_save_id=preview.target_save_id or target_save_id,
                    name=payload.name,
                    **kwargs,
                )
            error = _runtime_model_error(model)
            if error:
                raise HTTPException(status_code=400, detail=error)
            payload_dict = _runtime_json_dict(state, model)
            _publish_runtime_changed_from_model_result(
                state,
                payload_dict,
                reason="character_bundle_imported",
            )
            return payload_dict
        except HTTPException:
            raise
        except Exception as exc:
            detail = str(exc) or exc.__class__.__name__
            raise HTTPException(status_code=400, detail=detail) from exc
        finally:
            preview.bundle_path.unlink(missing_ok=True)

    @app.get("/api/diagnostics")
    def diagnostics(
        state: StateDep,
        save_id: str | None = None,
        since: str | None = None,
        limit: int = _DEFAULT_DIAGNOSTICS_LIMIT,
        category: Annotated[list[str] | None, Query()] = None,
        request_id: str | None = None,
        job_id: str | None = None,
        route: str | None = None,
        component: str | None = None,
    ) -> dict[str, Any]:
        with state.lock:
            return _diagnostics_payload_for_request(
                state,
                save_id=save_id,
                since=since,
                limit=limit,
                categories=tuple(category or ()),
                request_id=request_id,
                job_id=job_id,
                route=route,
                component=component,
            )

    @app.get("/api/settings")
    def settings(
        state: StateDep,
        save_id: str | None = None,
    ) -> dict[str, Any]:
        with state.lock:
            current_user = _save_access_user(state)
            current_user_role = current_user.role if current_user is not None else None
            is_admin = current_user_role in {None, "admin"}
            checked_save_id = (
                _require_save_id(save_id) if save_id is not None else None
            )
            settings_service = state.settings_service()
            payload = _json_dict(
                bragi_settings_bindings().build_settings_model(
                    repositories=state.repositories,
                    providers=tuple(state.providers.keys()),
                    active_save_id=checked_save_id,
                    current_user_role=current_user_role,
                    current_user_id=(
                        current_user.id if current_user is not None else None
                    ),
                    log_file_path=state.log_file_path,
                    secret_storage_warning=(
                        settings_service.secret_storage_warning() if is_admin else None
                    ),
                )
            )
            payload = {
                key: value for key, value in payload.items() if value is not None
            }
        return payload

    @app.get("/api/settings/providers")
    def provider_settings(state: StateDep) -> dict[str, Any]:
        with state.lock:
            current_user = _save_access_user(state)
            current_user_role = current_user.role if current_user is not None else None
            is_admin = current_user_role in {None, "admin"}
            settings_service = state.settings_service()
            payload = _json_dict(
                bragi_settings_bindings().build_provider_settings_model(
                    repositories=state.repositories,
                    providers=tuple(state.providers.keys()),
                    current_user_role=current_user_role,
                    secret_storage_warning=(
                        settings_service.secret_storage_warning() if is_admin else None
                    ),
                )
            )
            payload = {
                key: value for key, value in payload.items() if value is not None
            }
        return payload

    @app.get("/api/settings/local")
    def local_settings(state: StateDep) -> dict[str, Any]:
        with state.lock:
            current_user = _save_access_user(state)
            current_user_role = current_user.role if current_user is not None else None
            payload = _json_dict(
                bragi_settings_bindings().build_local_settings_model(
                    repositories=state.repositories,
                    current_user_role=current_user_role,
                    current_user_id=(
                        current_user.id if current_user is not None else None
                    ),
                )
            )
            payload = {
                key: value for key, value in payload.items() if value is not None
            }
        return payload

    @app.get("/api/settings/shell")
    def settings_shell(state: StateDep) -> dict[str, Any]:
        with state.lock:
            current_user = _save_access_user(state)
            from bragi.services.pending_jobs_settings import (
                PENDING_JOBS_DISPLAY_MODE_OPTIONS,
                PENDING_JOBS_DISPLAY_MODE_SETTING,
                sanitize_pending_jobs_display_mode,
            )

            selected = sanitize_pending_jobs_display_mode(
                state.repositories.get_effective_setting(
                    PENDING_JOBS_DISPLAY_MODE_SETTING,
                    user_id=current_user.id if current_user is not None else None,
                )
            )
        return {
            "pending_jobs_display_mode": {
                "setting_key": PENDING_JOBS_DISPLAY_MODE_SETTING,
                "selected": selected,
                "options": list(PENDING_JOBS_DISPLAY_MODE_OPTIONS),
            }
        }

    @app.post("/api/log/client")
    def client_log(payload: ClientLogRequest) -> dict[str, bool]:
        level = payload.level if payload.level in {"debug", "info", "error"} else "info"
        observe(
            payload.event,
            level=cast(Any, level),
            source="client",
            **sanitize_client_fields(payload.fields),
        )
        return {"ok": True}

    @app.post("/api/settings/provider-key")
    def set_provider_key(
        payload: ProviderKeyRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        _require_admin_user()
        if payload.provider not in state.providers:
            raise HTTPException(status_code=404, detail="Unknown provider")
        try:
            with state.lock:
                state.settings_service().set_provider_api_key(
                    payload.provider,
                    payload.api_key,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True}

    @app.delete("/api/settings/provider-key/{provider}")
    def clear_provider_key(provider: str, state: StateDep) -> dict[str, Any]:
        _require_admin_user()
        if provider not in state.providers:
            raise HTTPException(status_code=404, detail="Unknown provider")
        with state.lock:
            state.settings_service().clear_provider_api_key(provider)
        return {"ok": True}

    @app.post("/api/settings/model-refresh/{provider}")
    async def refresh_models(provider: str, state: StateDep) -> dict[str, Any]:
        _require_admin_user()
        if provider not in state.providers:
            raise HTTPException(status_code=404, detail="Unknown provider")

        async def worker(handle: JobHandle) -> Any:
            await handle.event("progress", {"label": f"Refreshing {provider}"})
            return await state.settings_service().refresh_provider_models(provider)

        return await _create_job_summary(
            state,
            "model_refresh",
            worker,
            exclusive_key=f"model_refresh:{provider}",
            lock_runtime=False,
        )

    @app.post("/api/settings/model-preference")
    def set_model_preference(
        payload: ModelPreferenceRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        _require_admin_user()
        checked_save_id = (
            _require_save_id(payload.save_id) if payload.save_id is not None else None
        )
        with state.lock:
            if checked_save_id is not None:
                _raise_unless_save_action_allowed(state, checked_save_id, "mutate")
            try:
                state.settings_service().set_model_preference(
                    task=payload.task,
                    provider=payload.provider,
                    model_id=payload.model_id,
                    save_id=checked_save_id,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        if checked_save_id is not None:
            _publish_save_event(
                state,
                checked_save_id,
                "runtime_changed",
                {"reason": "save_model_preference_updated", "task": payload.task},
            )
        return {"ok": True}

    @app.delete("/api/settings/model-preference/{task}")
    def clear_model_preference(
        task: str,
        state: StateDep,
        save_id: str | None = None,
    ) -> dict[str, Any]:
        _require_admin_user()
        checked_save_id = _require_save_id(save_id) if save_id is not None else None
        with state.lock:
            if checked_save_id is not None:
                _raise_unless_save_action_allowed(state, checked_save_id, "mutate")
            state.settings_service().clear_model_preference(
                task,
                save_id=checked_save_id,
            )
        if checked_save_id is not None:
            _publish_save_event(
                state,
                checked_save_id,
                "runtime_changed",
                {"reason": "save_model_preference_cleared", "task": task},
            )
        return {"ok": True}

    @app.post("/api/settings/model-thinking")
    def set_model_thinking_preference(
        payload: ModelThinkingPreferenceRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        _require_admin_user()
        checked_save_id = (
            _require_save_id(payload.save_id) if payload.save_id is not None else None
        )
        try:
            with state.lock:
                if checked_save_id is not None:
                    _raise_unless_save_action_allowed(
                        state,
                        checked_save_id,
                        "mutate",
                    )
                state.settings_service().set_model_thinking_preference(
                    task=payload.task,
                    provider=payload.provider,
                    model_id=payload.model_id,
                    level=payload.level,
                    save_id=checked_save_id,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if checked_save_id is not None:
            _publish_save_event(
                state,
                checked_save_id,
                "runtime_changed",
                {"reason": "save_model_thinking_updated", "task": payload.task},
            )
        return {"ok": True}

    @app.delete("/api/settings/model-thinking/{task}")
    def clear_model_thinking_preference(
        task: str,
        state: StateDep,
        save_id: str | None = None,
    ) -> dict[str, Any]:
        _require_admin_user()
        checked_save_id = _require_save_id(save_id) if save_id is not None else None
        with state.lock:
            if checked_save_id is not None:
                _raise_unless_save_action_allowed(state, checked_save_id, "mutate")
            state.settings_service().clear_model_thinking_preference(
                task,
                save_id=checked_save_id,
            )
        if checked_save_id is not None:
            _publish_save_event(
                state,
                checked_save_id,
                "runtime_changed",
                {"reason": "save_model_thinking_cleared", "task": task},
            )
        return {"ok": True}

    @app.post("/api/settings/model-routing-profiles")
    def save_model_routing_profile(
        payload: ModelRoutingProfileRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        _require_admin_user()
        from bragi.services.model_routing_profiles import (
            save_current_model_routing_profile,
        )

        try:
            with state.lock:
                profile = save_current_model_routing_profile(
                    state.repositories,
                    name=payload.name,
                    profile_id=payload.profile_id,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "profile": to_jsonable(profile)}

    @app.post("/api/settings/model-routing-profiles/{profile_id}/apply")
    def apply_saved_model_routing_profile(
        profile_id: str,
        state: StateDep,
    ) -> dict[str, Any]:
        _require_admin_user()
        from bragi.services.model_routing_profiles import apply_model_routing_profile

        try:
            with state.lock:
                profile = apply_model_routing_profile(state.repositories, profile_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"ok": True, "profile": to_jsonable(profile)}

    @app.delete("/api/settings/model-routing-profiles/{profile_id}")
    def delete_saved_model_routing_profile(
        profile_id: str,
        state: StateDep,
    ) -> dict[str, Any]:
        _require_admin_user()
        from bragi.services.model_routing_profiles import delete_model_routing_profile

        try:
            with state.lock:
                delete_model_routing_profile(state.repositories, profile_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"ok": True}

    def _set_scoped_setting(
        payload: ScopedSettingRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        with state.lock:
            from bragi.services.settings_policy import (
                role_can_write_scoped_setting,
                scoped_setting_policy,
            )

            current_user = _save_access_user(state)
            role = current_user.role if current_user is not None else "admin"
            try:
                policy = scoped_setting_policy(payload.key)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if not role_can_write_scoped_setting(role, payload.key):
                raise HTTPException(status_code=403, detail="Setting is not allowed")
            checked_save_id = (
                _require_save_id(payload.save_id)
                if policy.scope == "save"
                else None
            )
            if checked_save_id is not None:
                _raise_unless_save_action_allowed(state, checked_save_id, "mutate")
            try:
                state.settings_service().set_scoped_app_setting(
                    key=payload.key,
                    value=payload.value,
                    save_id=checked_save_id,
                    user_id=current_user.id if current_user is not None else None,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        if checked_save_id is not None:
            _publish_save_event(
                state,
                checked_save_id,
                "runtime_changed",
                {"reason": "local_setting_updated", "key": payload.key},
            )
        return {"ok": True}

    @app.post("/api/settings/scoped")
    def set_scoped_setting(
        payload: ScopedSettingRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        return _set_scoped_setting(payload, state)

    @app.post("/api/settings/local")
    def set_local_setting(
        payload: LocalSettingRequest,
        state: StateDep,
    ) -> dict[str, Any]:
        return _set_scoped_setting(payload, state)

    @app.post("/api/bundles/preview")
    async def preview_bundle(
        file: Annotated[UploadFile, File()],
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            _raise_unless_import_export_allowed(state)
            _prune_bundle_previews(state)
        try:
            bundle_path = await _store_upload(file, state.paths.temp_dir)
        except _BundleUploadTooLarge as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        preview_id = uuid4().hex
        try:
            async with state.lock.async_access():
                preview = state.runtime.preview_import_bundle(bundle_path)
        except Exception as exc:
            bundle_path.unlink(missing_ok=True)
            detail = str(exc) or exc.__class__.__name__
            raise HTTPException(status_code=400, detail=detail) from exc
        async with state.lock.async_access():
            state.bundle_previews[preview_id] = BundlePreviewState(
                bundle_path=bundle_path,
                owner_user_id=_owner_user_id_for_request(state),
            )
            _prune_bundle_previews(state)
        return {
            "preview_id": preview_id,
            "preview": await _review_bundle_preview_for_request(state, preview),
        }

    @app.post("/api/bundles/import/{preview_id}")
    def import_bundle(preview_id: str, state: StateDep) -> dict[str, Any]:
        with state.lock:
            _raise_unless_import_export_allowed(state)
            _prune_bundle_previews(state)
            preview = _require_owned_bundle_preview(
                state,
                state.bundle_previews,
                preview_id,
                detail="Unknown bundle preview",
            )
            state.bundle_previews.pop(preview_id, None)
        try:
            with state.lock:
                kwargs: dict[str, Any] = {}
                if _call_accepts_keyword(
                    state.runtime.import_save_bundle,
                    "owner_user_id",
                ):
                    kwargs["owner_user_id"] = _owner_user_id_for_request(state)
                if _call_accepts_keyword(
                    state.runtime.import_save_bundle,
                    "remember_process_active_save",
                ):
                    kwargs["remember_process_active_save"] = (
                        not _auth_context_enabled(state)
                    )
                model = state.runtime.import_save_bundle(preview.bundle_path, **kwargs)
            error = _runtime_model_error(model)
            if error:
                raise HTTPException(status_code=400, detail=error)
            payload_dict = _runtime_json_dict(state, model)
            _remember_user_active_save_from_model_result(state, payload_dict)
            _publish_runtime_changed_from_model_result(
                state,
                payload_dict,
                reason="save_imported",
            )
            _publish_save_event(
                state,
                None,
                "saves_changed",
                {"reason": "save_imported"},
            )
            return payload_dict
        except HTTPException:
            raise
        except Exception as exc:
            detail = str(exc) or exc.__class__.__name__
            raise HTTPException(status_code=400, detail=detail) from exc
        finally:
            preview.bundle_path.unlink(missing_ok=True)

    @app.post("/api/scenario-bundles/preview")
    async def preview_scenario_bundle(
        file: Annotated[UploadFile, File()],
        state: StateDep,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            _raise_unless_import_export_allowed(state)
            _prune_scenario_bundle_previews(state)
        try:
            bundle_path = await _store_upload(file, state.paths.temp_dir)
        except _BundleUploadTooLarge as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        preview_id = uuid4().hex
        try:
            async with state.lock.async_access():
                preview = state.runtime.preview_import_scenario_bundle(bundle_path)
        except Exception as exc:
            bundle_path.unlink(missing_ok=True)
            detail = str(exc) or exc.__class__.__name__
            raise HTTPException(status_code=400, detail=detail) from exc
        async with state.lock.async_access():
            state.scenario_bundle_previews[preview_id] = BundlePreviewState(
                bundle_path=bundle_path,
                owner_user_id=_owner_user_id_for_request(state),
            )
            _prune_scenario_bundle_previews(state)
        return {
            "preview_id": preview_id,
            "preview": await _review_bundle_preview_for_request(state, preview),
        }

    @app.post("/api/scenario-bundles/import/{preview_id}")
    def import_scenario_bundle(
        preview_id: str,
        state: StateDep,
    ) -> dict[str, Any]:
        with state.lock:
            _raise_unless_import_export_allowed(state)
            _prune_scenario_bundle_previews(state)
            preview = _require_owned_bundle_preview(
                state,
                state.scenario_bundle_previews,
                preview_id,
                detail="Unknown scenario bundle preview",
            )
            state.scenario_bundle_previews.pop(preview_id, None)
        try:
            with state.lock:
                model = state.runtime.import_scenario_bundle(preview.bundle_path)
            error = _runtime_model_error(model)
            if error:
                raise HTTPException(status_code=400, detail=error)
            payload_dict = _json_dict(model)
            _publish_save_event(
                state,
                None,
                "scenarios_changed",
                {"reason": "scenario_imported"},
            )
            return payload_dict
        except HTTPException:
            raise
        except Exception as exc:
            detail = str(exc) or exc.__class__.__name__
            raise HTTPException(status_code=400, detail=detail) from exc
        finally:
            preview.bundle_path.unlink(missing_ok=True)

    @app.get("/api/scenario-bundles/export/{scenario_id}")
    def export_scenario_bundle(scenario_id: str, state: StateDep) -> FileResponse:
        _raise_unless_import_export_allowed(state)
        bundle_path = (
            state.paths.temp_dir
            / f"bragi-scenario-{uuid4().hex}.bragi-scenario"
        )
        try:
            with state.lock:
                model = state.runtime.export_saved_scenario(scenario_id, bundle_path)
            error = _runtime_model_error(model)
            if error:
                raise HTTPException(status_code=400, detail=error)
            return FileResponse(
                bundle_path,
                media_type="application/octet-stream",
                filename=bundle_path.name,
                background=BackgroundTask(_unlink_file, bundle_path),
            )
        except HTTPException:
            bundle_path.unlink(missing_ok=True)
            raise
        except Exception:
            bundle_path.unlink(missing_ok=True)
            raise

    @app.get("/api/bundles/export")
    def export_bundle(
        state: StateDep,
        include_revision_history: bool = False,
        save_id: str | None = None,
    ) -> FileResponse:
        bundle_path = state.paths.temp_dir / f"bragi-export-{uuid4().hex}.bragi-chat"
        try:
            with state.lock:
                resolved_save_id = _resolve_runtime_save_id(state, save_id)
                if resolved_save_id is None:
                    raise HTTPException(
                        status_code=400,
                        detail=_SAVE_ID_REQUIRED_DETAIL,
                    )
                _raise_unless_save_action_allowed(state, resolved_save_id, "export")
                kwargs: dict[str, Any] = {}
                if _call_accepts_keyword(
                    state.runtime.export_active_save,
                    "active_save_id",
                ):
                    kwargs["active_save_id"] = resolved_save_id
                if _call_accepts_keyword(
                    state.runtime.export_active_save,
                    "include_message_revisions",
                ):
                    kwargs["include_message_revisions"] = include_revision_history
                if kwargs:
                    model = state.runtime.export_active_save(
                        bundle_path,
                        **kwargs,
                    )
                else:
                    model = state.runtime.export_active_save(bundle_path)
            error = _runtime_model_error(model)
            if error:
                raise HTTPException(status_code=400, detail=error)
            return FileResponse(
                bundle_path,
                media_type="application/octet-stream",
                filename=bundle_path.name,
                background=BackgroundTask(_unlink_file, bundle_path),
            )
        except HTTPException:
            bundle_path.unlink(missing_ok=True)
            raise
        except Exception:
            bundle_path.unlink(missing_ok=True)
            raise

    @app.get("/api/jobs")
    def list_jobs(
        state: StateDep,
        status: str = "active",
        save_id: str | None = None,
        since: str | None = None,
        limit: int = _DEFAULT_DIAGNOSTICS_LIMIT,
    ) -> dict[str, Any]:
        if status == "active":
            with state.lock:
                if save_id is not None:
                    _raise_unknown_save_if_possible(state, save_id)
                    records = [
                        record
                        for record in state.jobs.list_active(save_id=save_id)
                        if _request_user_can_access_job(state, record)
                    ]
                else:
                    records = [
                        record
                        for record in state.jobs.list_active()
                        if _request_user_can_access_job(state, record)
                    ]
            return {
                "jobs": [
                    _job_summary_for_request(state, record)
                    for record in records
                ]
            }
        terminal_statuses = _TERMINAL_JOB_STATUS_FILTERS.get(status)
        if terminal_statuses is None:
            raise HTTPException(status_code=400, detail="Unsupported job status filter")
        since_dt = _parse_diagnostics_since(since)
        bounded_limit = _bounded_diagnostics_limit(limit)
        with state.lock:
            if save_id is not None:
                _raise_unknown_save_if_possible(state, save_id)
            query_limit = (
                _MAX_DIAGNOSTICS_LIMIT
                if save_id is None and _auth_context_enabled(state)
                else bounded_limit
            )
            records = state.repositories.list_terminal_jobs(
                statuses=terminal_statuses,
                save_id=save_id,
                since=since_dt.isoformat() if since_dt is not None else None,
                limit=query_limit,
            )
            records = [
                record
                for record in records
                if _request_user_can_access_job(state, record)
            ][:bounded_limit]
            step_counts = state.repositories.count_job_steps_by_job_id(
                tuple(record.id for record in records)
            )
        return {
            "jobs": [
                _terminal_job_summary(record, step_count=step_counts.get(record.id, 0))
                for record in records
            ]
        }

    @app.get("/api/jobs/{job_id}")
    def get_job(
        job_id: str,
        state: StateDep,
        save_id: str | None = None,
    ) -> dict[str, Any]:
        with state.lock:
            record = _job_for_save_or_404(state, job_id, save_id)
        return _job_summary_for_request(state, record)

    @app.get("/api/jobs/{job_id}/steps")
    def get_job_steps(
        job_id: str,
        state: StateDep,
        save_id: str | None = None,
    ) -> dict[str, Any]:
        with state.lock:
            _persisted_job_for_save_or_404(state, job_id, save_id)
            steps = state.repositories.list_job_steps(job_id)
        return {
            "job_id": job_id,
            "steps": [_job_step_summary(step) for step in steps],
        }

    @app.get("/api/jobs/{job_id}/diagnostics")
    def get_job_diagnostics(
        job_id: str,
        state: StateDep,
        save_id: str | None = None,
    ) -> dict[str, Any]:
        with state.lock:
            record = _persisted_job_for_save_or_404(state, job_id, save_id)
            user = _save_access_user(state)
            if user is not None and user.role == "child":
                raise HTTPException(
                    status_code=403,
                    detail="Diagnostics access is not allowed",
                )
            from bragi.services.job_diagnostics import (
                build_job_diagnostic_snapshot,
                redact_job_diagnostic_snapshot,
            )

            snapshot = record.diagnostics or build_job_diagnostic_snapshot(record)
            is_admin = user is None or user.role == "admin"
            visible_snapshot = redact_job_diagnostic_snapshot(
                snapshot,
                include_prompt=is_admin,
                include_failure_detail=is_admin,
            )
            return {
                "job_id": record.id,
                "job_type": record.type,
                "save_id": record.save_id,
                "status": record.status,
                "detail_level": "admin" if is_admin else "metadata",
                "detail_available": record.diagnostics is not None,
                "diagnostics": visible_snapshot,
            }

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(
        job_id: str,
        state: StateDep,
        save_id: str | None = None,
    ) -> dict[str, Any]:
        async with state.lock.async_access():
            record = _job_for_save_or_404(state, job_id, save_id)
            _raise_unless_job_cancel_allowed(state, record)
            should_cancel_runtime_chat = _should_cancel_runtime_chat_for_job(record)
        job_cancelled = await state.jobs.cancel(job_id)
        runtime_cancelled = (
            _cancel_runtime_chat_for_job(state, record)
            if should_cancel_runtime_chat
            else False
        )
        if job_cancelled or runtime_cancelled:
            return {"cancelled": True}
        current = state.jobs.get(job_id)
        if current is not None and current.status in TERMINAL_JOB_STATUSES:
            return {"cancelled": False}
        raise HTTPException(
            status_code=409,
            detail=_JOB_CANCELLATION_FAILED_DETAIL,
        )

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(
        job_id: str,
        request: Request,
        state: StateDep,
        save_id: str | None = None,
    ) -> StreamingResponse:
        try:
            async with state.lock.async_access():
                record = _job_for_save_or_404(state, job_id, save_id)
        except HTTPException:
            observe("web.sse.unknown_job", level="error", job_id=job_id)
            raise
        observe("web.sse.opened", job_id=job_id, job_type=record.type)
        return StreamingResponse(
            _event_stream(
                state,
                job_id,
                last_event_id=_save_event_cursor_from_header(
                    request.headers.get("last-event-id"),
                    latest_event_id=record.event_offset + len(record.events),
                ),
                current_user=_current_request_user(),
                current_user_role=_current_request_role(state),
            ),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    _mount_spa(app)
    return app


def get_state(app: FastAPI) -> WebAppState:
    return cast(WebAppState, app.state.bragi)


def _repository_scope_for_state(state: Any) -> Any:
    state_scope = getattr(state, "repository_scope", None)
    if callable(state_scope):
        return state_scope()
    repositories = getattr(state, "repositories", None)
    repository_scope = getattr(repositories, "scope", None)
    if callable(repository_scope):
        return repository_scope()
    return nullcontext()


def state_dependency() -> WebAppState:
    raise RuntimeError("state_dependency override was not installed")


StateDep = Annotated[WebAppState, Depends(state_dependency)]


def _require_admin_user() -> UserRecord | None:
    state = _REQUEST_STATE.get()
    if state is not None and not _auth_context_enabled(state):
        return None
    user = _current_request_user()
    if user is None:
        raise HTTPException(status_code=401, detail=_AUTH_REQUIRED_DETAIL)
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def _raise_user_management_error(exc: Exception) -> NoReturn:
    from bragi.services.auth_service import (
        CurrentUserDisableError,
        LastActiveAdminError,
        UnknownUserError,
    )

    if isinstance(exc, UnknownUserError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, (LastActiveAdminError, CurrentUserDisableError)):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise exc


def _auth_service(state: WebAppState) -> AuthService:
    service_factory = getattr(state, "auth_service", None)
    if callable(service_factory):
        return cast("AuthService", service_factory())
    from bragi.services.auth_service import AuthService

    return AuthService(repositories=state.repositories)


def _requires_authenticated_session(
    request: Request,
    state: WebAppState,
) -> bool:
    if getattr(state, "auth_required", True) is not True:
        return False
    path = request.url.path
    return path.startswith("/api/") and path not in _PUBLIC_API_PATHS


def _load_request_user(
    request: Request,
    state: WebAppState,
) -> UserRecord | None:
    token = request.cookies.get(_SESSION_COOKIE_NAME)
    if not token:
        return None
    return _auth_service(state).load_current_user(token)


def _current_request_user() -> UserRecord | None:
    user = _REQUEST_USER.get()
    return cast("UserRecord | None", user)


def _auth_context_enabled(state: WebAppState) -> bool:
    return getattr(state, "auth_required", True) is True


def _save_access_user(state: WebAppState) -> UserRecord | None:
    if not _auth_context_enabled(state):
        return None
    return _current_request_user()


def _current_request_role(state: WebAppState) -> str | None:
    if not _auth_context_enabled(state):
        return "admin"
    user = _current_request_user()
    return user.role if user is not None else None


_CURRENT_USER_SENTINEL = object()


def _content_safety_policy_for_request(
    state: WebAppState,
    *,
    current_user: UserRecord | None | object = _CURRENT_USER_SENTINEL,
) -> Any:
    from bragi.services.content_rating import (
        DEFAULT_ADULT_CONTENT_RATING,
        DEFAULT_FADE_TO_BLACK_ENABLED,
        ContentSafetyPolicy,
        effective_content_safety_policy,
    )

    user = (
        _save_access_user(state)
        if current_user is _CURRENT_USER_SENTINEL
        else cast("UserRecord | None", current_user)
    )
    if not callable(getattr(state.repositories, "get_effective_setting", None)):
        return ContentSafetyPolicy(
            rating=DEFAULT_ADULT_CONTENT_RATING,
            fade_to_black_enabled=DEFAULT_FADE_TO_BLACK_ENABLED,
            force_venice_safe_mode=False,
        )
    return effective_content_safety_policy(
        state.repositories,
        user_id=user.id if user is not None else None,
    )


def _raise_unless_unrated_reference_upload(state: WebAppState) -> None:
    if _content_safety_policy_for_request(state).rating == "unrated":
        return
    raise HTTPException(
        status_code=400,
        detail=(
            "Uploaded images cannot be safety-reviewed. Set the content rating "
            "to Unrated before uploading a reference image."
        ),
    )


def _owner_user_id_for_request(state: WebAppState) -> str | None:
    user = _save_access_user(state)
    return user.id if user is not None else None


def _raise_if_starter_generation_payload_too_large(
    payload: ScenarioDraftCharacterStarterGenerationRequest,
) -> None:
    if len(payload.sections) > _DRAFT_STARTER_GENERATION_MAX_SECTION_COUNT:
        raise HTTPException(
            status_code=413,
            detail="Character starter generation request is too large",
        )
    if (
        len(payload.character_starters)
        > _DRAFT_STARTER_GENERATION_MAX_EXISTING_STARTERS
    ):
        raise HTTPException(
            status_code=413,
            detail="Character starter generation request is too large",
        )
    if len(payload.model_dump_json()) > _DRAFT_STARTER_GENERATION_MAX_JSON_CHARS:
        raise HTTPException(
            status_code=413,
            detail="Character starter generation request is too large",
        )


def _raise_if_starter_generation_request_invalid(
    payload: ScenarioDraftCharacterStarterGenerationRequest,
) -> None:
    from bragi.services.character_profile_completion import (
        normalize_scenario_character_starters,
    )
    from bragi.services.scenario_service import (
        normalize_scenario_draft_sections,
        normalized_scenario_types_and_flag,
    )

    try:
        draft_type, _draft_genres, _action_choices_enabled = (
            normalized_scenario_types_and_flag(
                payload.scenario_type,
                scenario_types=payload.scenario_types,
                action_choices_enabled=payload.action_choices_enabled,
            )
        )
        normalize_scenario_draft_sections(draft_type, payload.sections)
        normalize_scenario_character_starters(
            list(payload.character_starters),
            strict=True,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if payload.count is not None and (
        isinstance(payload.count, bool) or payload.count < 1 or payload.count > 12
    ):
        raise HTTPException(
            status_code=400,
            detail="Number of characters must be between 1 and 12",
        )
    if payload.custom_description.strip():
        return
    if payload.count is None:
        raise HTTPException(
            status_code=400,
            detail="Number of characters or custom character description is required",
        )


def _request_owns_bundle_preview(
    state: WebAppState,
    preview: BundlePreviewState,
) -> bool:
    if not _auth_context_enabled(state):
        return True
    user = _current_request_user()
    return user is not None and preview.owner_user_id == user.id


async def _review_bundle_preview_for_request(
    state: WebAppState,
    preview: object,
    *,
    save_id: str | None = None,
) -> object:
    serialized = to_jsonable(preview)
    if not callable(
        getattr(state.repositories, "get_effective_setting", None)
    ):
        return serialized
    from bragi.services.content_rating import effective_content_safety_policy
    from bragi.services.content_safety_service import (
        ContentSafetyAction,
        ContentSafetyService,
    )

    policy = effective_content_safety_policy(
        state.repositories,
        user_id=_owner_user_id_for_request(state),
    )
    safety = await ContentSafetyService(
        repositories=state.repositories,
        providers=state.providers,
    ).review_narration(
        body=json.dumps(serialized, ensure_ascii=False, sort_keys=True),
        content_rating=policy.rating,
        fade_to_black_enabled=policy.fade_to_black_enabled,
        save_id=save_id,
    )
    if safety.action is ContentSafetyAction.ALLOW:
        return serialized
    return _replace_untrusted_preview_text(serialized, safety.body)


def _replace_untrusted_preview_text(
    value: object,
    replacement: str,
    *,
    key: str = "",
) -> object:
    structural_keys = {
        "save_id",
        "scenario_id",
        "character_id",
        "scenario_type",
        "bundle_version",
        "created_at",
        "updated_at",
        "exported_at",
    }
    if isinstance(value, str):
        return value if key in structural_keys else replacement
    if isinstance(value, list):
        return [
            _replace_untrusted_preview_text(item, replacement, key=key)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            item_key: _replace_untrusted_preview_text(
                item,
                replacement,
                key=str(item_key),
            )
            for item_key, item in value.items()
        }
    return value


def _require_owned_bundle_preview(
    state: WebAppState,
    previews: dict[str, BundlePreviewState],
    preview_id: str,
    *,
    detail: str,
) -> BundlePreviewState:
    preview = previews.get(preview_id)
    if preview is None or not _request_owns_bundle_preview(state, preview):
        raise HTTPException(status_code=404, detail=detail)
    return preview


def _save_list_for_request(state: WebAppState) -> list[Any]:
    if not _auth_context_enabled(state):
        return list(state.repositories.list_saves())
    user = _current_request_user()
    if user is None:
        return []
    return list(state.repositories.list_saves_for_user(user))


def _save_list_for_runtime_payload(
    state: WebAppState,
    *,
    current_user: UserRecord | None | object = _CURRENT_USER_SENTINEL,
) -> list[Any]:
    if not _auth_context_enabled(state):
        return list(state.repositories.list_saves())
    user = (
        _current_request_user()
        if current_user is _CURRENT_USER_SENTINEL
        else cast("UserRecord | None", current_user)
    )
    if user is None:
        return []
    return list(state.repositories.list_saves_for_user(user))


def _save_list_json_for_request(
    state: WebAppState,
    *,
    active_save_id: object = None,
    current_user: UserRecord | None | object = _CURRENT_USER_SENTINEL,
) -> list[dict[str, object]]:
    from bragi.services.sexual_content_safety import CONTENT_FILTER_TRANSITION

    rows: list[dict[str, object]] = []
    allowed_rating = _content_safety_policy_for_request(
        state,
        current_user=current_user,
    ).rating
    for save in _save_list_for_runtime_payload(
        state,
        current_user=current_user,
    ):
        scenario = _active_scenario(state, save.id)
        supported = not _scenario_is_retired(scenario)
        restricted = _scenario_exceeds_rating(
            scenario,
            allowed_rating=allowed_rating,
        )
        rows.append(
            {
            "save_id": save.id,
            "title": (
                CONTENT_FILTER_TRANSITION
                if restricted
                else getattr(save, "title", save.id)
            ),
            "active": isinstance(active_save_id, str)
            and save.id == active_save_id,
            "scenario_id": getattr(save, "scenario_id", None),
            "scenario_title": (
                CONTENT_FILTER_TRANSITION
                if restricted
                else getattr(save, "scenario_title", None)
            ),
            "created_at": getattr(save, "created_at", None),
            "updated_at": getattr(save, "updated_at", None),
            "last_opened_at": getattr(save, "last_opened_at", None),
            "supported": supported,
            "unsupported_reason": (
                None if supported else _RETIRED_SCENARIO_DETAIL
            ),
            }
        )
    return rows


def _scenario_exceeds_rating(
    scenario: object | None,
    *,
    allowed_rating: str,
) -> bool:
    if scenario is None:
        return False
    content_json = getattr(scenario, "content_json", None)
    if not isinstance(content_json, str):
        return True
    from bragi.content_rating_instructions import content_rating_exceeds
    from bragi.services.scenario_content_rating import scenario_content_rating

    return content_rating_exceeds(
        minimum_rating=scenario_content_rating(content_json),
        allowed_rating=allowed_rating,
    )


def _request_user_can_access_save(state: WebAppState, save_id: str) -> bool:
    if not _auth_context_enabled(state):
        return state.repositories.get_save(save_id) is not None
    user = _current_request_user()
    if user is None:
        return False
    return bool(state.repositories.user_can_access_save(user, save_id))


def _resolve_runtime_save_id(
    state: WebAppState,
    requested_save_id: str | None,
) -> str | None:
    if requested_save_id is not None:
        _raise_unknown_save_if_possible(state, requested_save_id)
        return requested_save_id
    if not _auth_context_enabled(state):
        return _current_runtime_save_id(state)
    user = _current_request_user()
    if user is None:
        return None
    active_save_id = cast(
        str | None,
        state.repositories.get_user_active_save_id(user.id),
    )
    if (
        active_save_id is not None
        and state.repositories.user_can_access_save(user, active_save_id)
    ):
        return active_save_id
    saves = state.repositories.list_saves_for_user(user)
    return saves[0].id if saves else None


def _remember_user_active_save(
    state: WebAppState,
    save_id: str,
) -> None:
    user = _save_access_user(state)
    if user is None:
        return
    if not state.repositories.user_can_access_save(user, save_id):
        return
    state.repositories.set_user_active_save_id(user_id=user.id, save_id=save_id)


def _touch_save_last_opened_if_possible(state: WebAppState, save_id: str) -> None:
    touch_save_last_opened = getattr(state.repositories, "touch_save_last_opened", None)
    if callable(touch_save_last_opened):
        touch_save_last_opened(save_id)


def _remember_user_active_save_from_model_result(
    state: WebAppState,
    model: object,
) -> None:
    if not isinstance(model, dict):
        return
    save_id = model.get("active_save_id")
    if isinstance(save_id, str) and save_id:
        _remember_user_active_save(state, save_id)


def _raise_unless_save_delete_allowed(state: WebAppState, save_id: str) -> None:
    _raise_unless_save_action_allowed(state, save_id, "delete")


def _scenario_is_retired(scenario: object | None) -> bool:
    if scenario is None:
        return False
    if getattr(scenario, "type", None) == _RETIRED_SCENARIO_TYPE:
        return True
    content_json = getattr(scenario, "content_json", "")
    if not isinstance(content_json, str):
        return False
    try:
        content = json.loads(content_json)
    except (TypeError, ValueError):
        return False
    genres = content.get("_scenario_genres") if isinstance(content, dict) else None
    return isinstance(genres, list) and _RETIRED_SCENARIO_TYPE in genres


def _save_has_retired_scenario(state: WebAppState, save_id: str) -> bool:
    return _scenario_is_retired(_active_scenario(state, save_id))


def _raise_if_save_retired(state: WebAppState, save_id: str) -> None:
    if _save_has_retired_scenario(state, save_id):
        raise HTTPException(status_code=409, detail=_RETIRED_SCENARIO_DETAIL)


def _raise_unless_scenario_supported(
    state: WebAppState,
    scenario_id: str,
) -> None:
    get_scenario = getattr(state.repositories, "get_scenario", None)
    if not callable(get_scenario):
        return
    scenario = get_scenario(scenario_id)
    if scenario is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown scenario id: {scenario_id}",
        )
    if _scenario_is_retired(scenario):
        raise HTTPException(status_code=409, detail=_RETIRED_SCENARIO_DETAIL)


def _raise_unless_save_action_allowed(
    state: WebAppState,
    save_id: str,
    action: str,
) -> None:
    _raise_unknown_save_if_possible(state, save_id)
    if action not in {"delete", "export"} and _save_has_retired_scenario(
        state,
        save_id,
    ):
        raise HTTPException(status_code=409, detail=_RETIRED_SCENARIO_DETAIL)
    user = _save_access_user(state)
    if user is None:
        return
    if user.role == "admin":
        return
    if user.role == "child" and action not in _CHILD_ALLOWED_SAVE_ACTIONS:
        raise HTTPException(status_code=403, detail="Save action is not allowed")
    if action != "delete":
        return
    save = state.repositories.get_save(save_id)
    if user.role == "user" and save is not None and save.owner_user_id == user.id:
        return
    raise HTTPException(status_code=403, detail="Save deletion is not allowed")


def _raise_unless_import_export_allowed(state: WebAppState) -> None:
    user = _save_access_user(state)
    if user is not None and user.role == "child":
        raise HTTPException(status_code=403, detail="Import/export is not allowed")


def _raise_if_retired_scenario_request(
    scenario_type: object,
    scenario_types: object = None,
) -> None:
    retired = scenario_type == _RETIRED_SCENARIO_TYPE
    if not retired and isinstance(scenario_types, (list, tuple)):
        retired = _RETIRED_SCENARIO_TYPE in scenario_types
    if retired:
        raise HTTPException(status_code=400, detail=_RETIRED_SCENARIO_DETAIL)


def _raise_if_invalid_interaction_mode(value: object) -> None:
    if value is not None and not isinstance(value, str):
        raise HTTPException(
            status_code=400,
            detail="interaction_mode must be 'roleplay' or 'storyteller'",
        )
    if value is not None and value not in {"roleplay", "storyteller"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown interaction mode: {value}",
        )


def _raise_unless_save_diagnostics_allowed(
    state: WebAppState,
    save_id: str,
) -> None:
    _raise_unless_save_action_allowed(state, save_id, "read")
    user = _save_access_user(state)
    if user is not None and user.role == "child":
        raise HTTPException(status_code=403, detail="Diagnostics access is not allowed")


def _diagnostics_payload_for_request(
    state: WebAppState,
    *,
    save_id: str | None,
    since: str | None,
    limit: int,
    categories: tuple[str, ...],
    request_id: str | None,
    job_id: str | None,
    route: str | None,
    component: str | None,
) -> dict[str, Any]:
    checked_save_id = _checked_diagnostics_save_id(state, save_id)
    since_dt = _parse_diagnostics_since(since)
    bounded_limit = _bounded_diagnostics_limit(limit)
    category_set = _diagnostic_category_set(categories)
    performance_since = (
        since_dt
        if since_dt is not None
        else datetime.now(UTC) - timedelta(seconds=_DEFAULT_PERFORMANCE_WINDOW_SECONDS)
    )
    performance_since_text = performance_since.isoformat()
    current_user = _save_access_user(state)
    if current_user is not None and current_user.role == "child":
        raise HTTPException(status_code=403, detail="Diagnostics access is not allowed")
    is_admin = current_user is None or current_user.role == "admin"

    report = None
    if is_admin and (
        "signals" in category_set or "performance" in category_set
    ):
        report = bragi_diagnostics_bindings().DiagnosticsService(
            repositories=state.repositories,
            log_file_path=state.log_file_path,
        ).list_diagnostics(
            save_id=checked_save_id,
            since=performance_since_text,
            limit=bounded_limit,
        )
    signals: list[dict[str, object]] = []
    runtime_performance: object | None = None
    if is_admin and report is not None:
        if "signals" in category_set:
            secret_storage_warning = state.settings_service().secret_storage_warning()
            signal_entries = _diagnostic_signals_from_report(
                report,
                secret_storage_warning=secret_storage_warning,
            )
            signal_entries.extend(
                _diagnostic_signal_from_entry(entry)
                for entry in bragi_settings_bindings().configuration_diagnostics(
                    state.repositories
                )
            )
            signals = _filter_diagnostic_signals(
                signal_entries,
                save_id=checked_save_id,
            )
        if "performance" in category_set:
            runtime_performance = report.runtime_performance

    maintenance_jobs = (
        _filtered_maintenance_jobs(
            state,
            save_id=checked_save_id,
            since=since_dt,
            limit=bounded_limit,
        )
        if is_admin and "jobs" in category_set
        else []
    )
    scheduler_health = (
        _scheduler_health(state, save_id=checked_save_id)
        if is_admin and "scheduler" in category_set
        else _empty_scheduler_health()
    )
    events = (
        _filtered_recent_events(
            since=since_dt,
            limit=bounded_limit,
            request_id=request_id,
            job_id=job_id,
            route=route,
            component=component,
        )
        if is_admin and "events" in category_set
        else []
    )
    active_save_health = (
        _active_save_health(state, checked_save_id)
        if checked_save_id is not None and "save_health" in category_set
        else None
    )
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "filters": {
            "save_id": checked_save_id,
            "categories": list(categories),
            "limit": bounded_limit,
            "since": since,
            "request_id": request_id,
            "job_id": job_id,
            "route": route,
            "component": component,
        },
        "signals": signals,
        "maintenance_jobs": maintenance_jobs,
        "runtime_performance": runtime_performance,
        "scheduler_health": scheduler_health,
        "web_events": events,
        "active_save_health": active_save_health,
    }
    return cast(dict[str, Any], to_jsonable(payload))


def _checked_diagnostics_save_id(
    state: WebAppState,
    save_id: str | None,
) -> str | None:
    if save_id is None:
        return None
    checked_save_id = _require_save_id(save_id)
    _raise_unless_save_diagnostics_allowed(state, checked_save_id)
    return checked_save_id


def _bounded_diagnostics_limit(limit: int) -> int:
    return min(_MAX_DIAGNOSTICS_LIMIT, max(0, limit))


def _diagnostic_category_set(categories: tuple[str, ...]) -> frozenset[str]:
    if not categories:
        return _DIAGNOSTIC_CATEGORIES
    normalized = tuple(category.strip().lower() for category in categories)
    unknown = sorted(
        category
        for category in normalized
        if category not in _DIAGNOSTIC_CATEGORIES
    )
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported diagnostics category: {unknown[0]}",
        )
    return frozenset(normalized)


def _parse_diagnostics_since(since: str | None) -> datetime | None:
    if not since:
        return None
    parsed = _parse_diagnostic_timestamp(since)
    if parsed is None:
        raise HTTPException(status_code=400, detail="Invalid diagnostics since value")
    return parsed


def _diagnostic_signals_from_report(
    report: Any,
    *,
    secret_storage_warning: str | None,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for provider in getattr(report, "provider_configs", ()):
        if getattr(provider, "last_error", None):
            entries.append(
                {
                    "kind": "provider",
                    "provider": getattr(provider, "provider", None),
                    "error": getattr(provider, "last_error", None),
                }
            )
    for job in getattr(report, "failed_jobs", ()):
        entries.append(
            {
                "kind": "job",
                "job_id": getattr(job, "id", None),
                "job_type": getattr(job, "type", None),
                "save_id": getattr(job, "save_id", None),
                "provider": getattr(job, "provider", None),
                "model": getattr(job, "model", None),
                "origin": getattr(job, "origin", None),
                "detail_available": getattr(job, "detail_available", False),
                "error": getattr(job, "error", None),
                "retry_summary": getattr(job, "retry_summary", None),
            }
        )
    for group in getattr(report, "failed_job_groups", ()):
        entries.append(
            {
                "kind": "job_group",
                "job_type": getattr(group, "type", None),
                "save_id": getattr(group, "save_id", None),
                "provider": getattr(group, "provider", None),
                "error": getattr(group, "summary", None),
            }
        )
    for finding in getattr(report, "generated_text_script_findings", ()):
        entries.append(
            {
                "kind": "script_policy",
                "save_id": getattr(finding, "save_id", None),
                "path": (
                    f"{getattr(finding, 'table', '')}."
                    f"{getattr(finding, 'field', '')}"
                ),
                "error": (
                    f"{getattr(finding, 'count', 0)} active generated text "
                    f"field(s) contain {getattr(finding, 'script', '')} "
                    "script; example record "
                    f"{getattr(finding, 'example_record_id', '')}"
                ),
            }
        )
    if secret_storage_warning is not None:
        entries.append({"kind": "secret_storage", "error": secret_storage_warning})
    log_file_path = getattr(report, "log_file_path", None)
    if log_file_path is not None:
        entries.append({"kind": "log_file", "error": None, "path": log_file_path})
    return entries


def _diagnostic_signal_from_entry(entry: Any) -> dict[str, object]:
    return {
        "kind": getattr(entry, "kind", None),
        "job_id": getattr(entry, "job_id", None),
        "provider": getattr(entry, "provider", None),
        "job_type": getattr(entry, "job_type", None),
        "model": getattr(entry, "model", None),
        "origin": getattr(entry, "origin", None),
        "detail_available": getattr(entry, "detail_available", False),
        "save_id": getattr(entry, "save_id", None),
        "path": getattr(entry, "path", None),
        "error": getattr(entry, "error", None),
        "retry_summary": getattr(entry, "retry_summary", None),
    }


def _filter_diagnostic_signals(
    signals: list[dict[str, object]],
    *,
    save_id: str | None,
) -> list[dict[str, object]]:
    if save_id is None:
        return signals
    return [
        signal
        for signal in signals
        if signal.get("save_id") in {None, save_id}
    ]


def _filtered_maintenance_jobs(
    state: WebAppState,
    *,
    save_id: str | None,
    since: datetime | None,
    limit: int,
) -> list[Any]:
    jobs = list(maintenance_job_diagnostics(state.repositories, limit=limit))
    if save_id is not None:
        jobs = [job for job in jobs if job.save_id in {None, save_id}]
    if since is not None:
        jobs = [
            job
            for job in jobs
            if _diagnostic_timestamp_matches_since(
                job.completed_at or job.started_at,
                since,
            )
        ]
    return jobs[:limit]


def _filtered_recent_events(
    *,
    since: datetime | None,
    limit: int,
    request_id: str | None,
    job_id: str | None,
    route: str | None,
    component: str | None,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    entries: list[dict[str, Any]] = []
    for event in recent_events():
        if since is not None and not _diagnostic_timestamp_matches_since(
            event.get("timestamp"),
            since,
        ):
            continue
        if request_id is not None and event.get("request_id") != request_id:
            continue
        if job_id is not None and event.get("job_id") != job_id:
            continue
        if route is not None and event.get("route") != route:
            continue
        if component is not None and event.get("component") != component:
            continue
        entries.append(event)
        if len(entries) >= limit:
            break
    return entries


def _scheduler_health(
    state: WebAppState,
    *,
    save_id: str | None,
) -> dict[str, object]:
    list_tasks = getattr(state.repositories, "list_scheduled_tasks", None)
    if not callable(list_tasks):
        return _empty_scheduler_health()
    tasks = list_tasks(save_id=save_id) if save_id is not None else list_tasks()
    now = datetime.now(UTC)
    rows = [_scheduler_task_diagnostic(task, now=now) for task in tasks]
    summary = {
        "total": len(rows),
        "healthy": 0,
        "overdue": 0,
        "leased": 0,
        "failed": 0,
        "disabled": 0,
    }
    for row in rows:
        status = row["status"]
        if status in summary:
            summary[status] += 1
    return {"summary": summary, "tasks": rows}


def _empty_scheduler_health() -> dict[str, object]:
    return {
        "summary": {
            "total": 0,
            "healthy": 0,
            "overdue": 0,
            "leased": 0,
            "failed": 0,
            "disabled": 0,
        },
        "tasks": [],
    }


def _scheduler_task_diagnostic(task: Any, *, now: datetime) -> dict[str, object]:
    status = _scheduler_task_status(task, now=now)
    task_type = str(getattr(task, "task_type", ""))
    result = getattr(task, "result", None)
    skip_reason = result.get("skip_reason") if isinstance(result, dict) else None
    return {
        "task_id": str(getattr(task, "id", "")),
        "task_type": task_type,
        "save_id": getattr(task, "save_id", None),
        "status": status,
        "enabled": bool(getattr(task, "enabled", False)),
        "interval_seconds": int(getattr(task, "interval_seconds", 0)),
        "next_run_at": getattr(task, "next_run_at", None),
        "lease_until": getattr(task, "lease_until", None),
        "last_started_at": getattr(task, "last_started_at", None),
        "last_completed_at": getattr(task, "last_completed_at", None),
        "last_job_id": getattr(task, "last_job_id", None),
        "failure_count": int(getattr(task, "failure_count", 0)),
        "error": (
            None
            if task_type == "observation_curation_drain"
            else bragi_diagnostics_bindings().redact_diagnostic_text(
                getattr(task, "error", None)
            )
        ),
        "skip_reason": skip_reason if isinstance(skip_reason, str) else None,
    }


def _scheduler_task_status(task: Any, *, now: datetime) -> str:
    if not bool(getattr(task, "enabled", False)):
        return "disabled"
    if int(getattr(task, "failure_count", 0)) > 0 and getattr(task, "error", None):
        return "failed"
    lease_until = _parse_diagnostic_timestamp(getattr(task, "lease_until", None))
    if lease_until is not None and lease_until > now:
        return "leased"
    next_run_at = _parse_diagnostic_timestamp(getattr(task, "next_run_at", None))
    if next_run_at is not None and next_run_at <= now:
        return "overdue"
    return "healthy"


def _active_save_health(state: WebAppState, save_id: str) -> dict[str, object]:
    _raise_unless_save_diagnostics_allowed(state, save_id)
    snapshot = bragi_diagnostics_bindings().EngineHealthService(
        state.repositories
    ).snapshot(save_id)
    payload = cast(dict[str, object], to_jsonable(snapshot))
    payload["latest_context_search"] = _diagnostic_context_search_summary(
        payload.get("latest_context_search")
    )
    payload["latest_chat_prompt"] = _diagnostic_chat_prompt_summary(
        payload.get("latest_chat_prompt")
    )
    return payload


def _diagnostic_context_search_summary(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    diagnostics = value.get("diagnostics")
    diagnostic_keys = (
        sorted(str(key) for key in diagnostics)
        if isinstance(diagnostics, dict)
        else []
    )
    return {
        "status": value.get("status") if isinstance(value.get("status"), str) else None,
        "error_present": bool(value.get("error_present")),
        "result_counts": _diagnostic_int_mapping(value.get("result_counts")),
        "diagnostic_keys": diagnostic_keys,
        "retrieval_degraded": value.get("retrieval_degraded") is True,
        "retrieval_recovery": (
            value.get("retrieval_recovery")
            if isinstance(value.get("retrieval_recovery"), str)
            else None
        ),
    }


def _diagnostic_chat_prompt_summary(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    return {
        "context_search_failed": (
            value.get("context_search_failed")
            if isinstance(value.get("context_search_failed"), bool)
            else None
        ),
        "narrator_mode": (
            value.get("narrator_mode")
            if isinstance(value.get("narrator_mode"), str)
            else None
        ),
        "message_count": _diagnostic_int_value(value.get("message_count")),
        "baseline_recent_message_count": _diagnostic_int_value(
            value.get("baseline_recent_message_count")
        ),
        "baseline_recent_message_chars": _diagnostic_int_value(
            value.get("baseline_recent_message_chars")
        ),
        "narrator_context_withheld_counts": _diagnostic_int_mapping(
            value.get("narrator_context_withheld_counts")
        ),
        "narrator_context_withheld_chars": _diagnostic_int_mapping(
            value.get("narrator_context_withheld_chars")
        ),
        "retrieved_counts": _diagnostic_int_mapping(value.get("retrieved_counts")),
        "final_prompt_budget": _diagnostic_prompt_budget_summary(
            value.get("final_prompt_budget")
        ),
    }


def _diagnostic_prompt_budget_summary(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {
        "input_limit_tokens": _diagnostic_int_value(value.get("input_limit_tokens")),
        "reserved_output_tokens": _diagnostic_int_value(
            value.get("reserved_output_tokens")
        ),
        "estimated_tokens_before": _diagnostic_int_value(
            value.get("estimated_tokens_before")
        ),
        "estimated_tokens_after": _diagnostic_int_value(
            value.get("estimated_tokens_after")
        ),
        "trimmed": (
            value.get("trimmed")
            if isinstance(value.get("trimmed"), bool)
            else None
        ),
    }


def _diagnostic_int_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(item, int) and not isinstance(item, bool)
    }


def _diagnostic_int_value(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _diagnostic_timestamp_matches_since(
    value: object,
    since: datetime,
) -> bool:
    timestamp = _parse_diagnostic_timestamp(value)
    return timestamp is not None and timestamp >= since


def _parse_diagnostic_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _raise_unless_character_export_allowed(
    state: WebAppState,
    character_id: str,
) -> str | None:
    get_character = getattr(state.repositories, "get_character", None)
    if not callable(get_character):
        _raise_unless_import_export_allowed(state)
        return None
    character = get_character(character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Unknown character")
    try:
        _raise_unless_save_action_allowed(state, character.save_id, "export")
    except HTTPException as exc:
        if exc.status_code == 403:
            raise
        raise HTTPException(status_code=404, detail="Unknown character") from exc
    return cast(str, character.save_id)


def _raise_unless_character_private_notes_export_allowed(
    state: WebAppState,
) -> None:
    if _current_request_role(state) != "admin":
        raise HTTPException(
            status_code=403,
            detail="Private notes export requires admin access",
        )


def _active_admin_exists(state: WebAppState) -> bool:
    return any(
        user.role == "admin" and user.status == "active"
        for user in state.repositories.list_users()
    )


def _bootstrap_status_payload(request: Request, state: WebAppState) -> dict[str, bool]:
    admin_exists = _active_admin_exists(state)
    return {
        "admin_exists": admin_exists,
        "bootstrap_required": not admin_exists,
        "setup_token_required": not admin_exists
        and _remote_bootstrap_setup_token_required(request),
    }


def _require_bootstrap_setup_token(request: Request, setup_token: str) -> None:
    if not _remote_bootstrap_setup_token_required(request):
        return
    expected = os.environ.get(_BOOTSTRAP_TOKEN_ENV, "").strip()
    if not expected:
        raise HTTPException(
            status_code=403,
            detail="Remote bootstrap setup token is not configured",
        )
    if not secrets.compare_digest(setup_token.strip(), expected):
        raise HTTPException(status_code=403, detail="Setup token is invalid")


def _remote_bootstrap_setup_token_required(request: Request) -> bool:
    return not (
        _is_loopback_client(_request_client_host(request))
        and _is_loopback_client(_request_host(request))
    )


def _request_client_host(request: Request) -> str | None:
    client = request.client
    if client is None:
        return None
    return client.host


def _is_loopback_client(host: str | None) -> bool:
    if host is None:
        return False
    if host in {"localhost", "testclient", "testserver"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _auth_attempt_key(
    kind: str,
    request: Request,
    username: str,
) -> tuple[str, str, str]:
    return (
        kind,
        _request_client_host(request) or "unknown",
        _auth_attempt_username(username),
    )


def _client_auth_attempt_key(kind: str, request: Request) -> tuple[str, str, str]:
    return (kind, _request_client_host(request) or "unknown", "client")


def _auth_attempt_username(username: str) -> str:
    return username.strip().casefold()[:_AUTH_USERNAME_MAX_LENGTH]


def _auth_attempt_throttle(state: WebAppState) -> AuthAttemptThrottle:
    throttle = getattr(state, "auth_attempts", None)
    if isinstance(throttle, AuthAttemptThrottle):
        return throttle
    throttle = AuthAttemptThrottle()
    state.auth_attempts = throttle
    return throttle


def _raise_if_auth_throttled(
    state: WebAppState,
    key: tuple[str, str, str],
) -> None:
    retry_after = _auth_attempt_throttle(state).blocked_for_seconds(key)
    if retry_after is None:
        return
    raise HTTPException(
        status_code=429,
        detail=_AUTH_THROTTLED_DETAIL,
        headers={"Retry-After": str(retry_after)},
    )


def _record_auth_failure(state: WebAppState, key: tuple[str, str, str]) -> None:
    _auth_attempt_throttle(state).record_failure(key)


def _record_auth_success(state: WebAppState, key: tuple[str, str, str]) -> None:
    _auth_attempt_throttle(state).record_success(key)


def _user_json(user: UserRecord) -> dict[str, str]:
    return {
        "id": user.id,
        "username": user.username,
        "role": user.role,
        "status": user.status,
    }


def _admin_user_json(
    user: UserRecord,
    repositories: Any,
) -> dict[str, str | None]:
    from bragi.services.content_rating import effective_content_safety_policy

    return {
        "id": user.id,
        "username": user.username,
        "role": user.role,
        "status": user.status,
        "content_rating": effective_content_safety_policy(
            repositories,
            user_id=user.id,
        ).rating,
        "created_at": user.created_at,
        "updated_at": user.updated_at,
    }


def _set_session_cookie(
    response: JSONResponse,
    token: str,
    request: Request,
) -> None:
    response.set_cookie(
        _SESSION_COOKIE_NAME,
        token,
        max_age=_SESSION_COOKIE_MAX_AGE_SECONDS,
        httponly=True,
        secure=_secure_session_cookies(request),
        samesite="lax",
        path="/",
    )


def _clear_session_cookie(response: JSONResponse) -> None:
    response.delete_cookie(_SESSION_COOKIE_NAME, path="/")


def _secure_session_cookies(request: Request) -> bool:
    return request.url.scheme == "https" or os.environ.get(
        "BRAGI_WEB_SECURE_COOKIES"
    ) == "1"


def _json_dict(value: object) -> dict[str, Any]:
    payload = to_jsonable(value)
    if isinstance(payload, dict):
        _scrub_internal_save_marker(payload)
        return payload
    return {"value": payload}


def _chronicle_json_dict(state: WebAppState, value: object) -> dict[str, Any]:
    payload = _json_dict(value)
    _scrub_response_payload_for_request(
        state,
        payload,
        current_user_role=_current_request_role(state),
    )
    return payload


def _bounded_chronicle_page_limit(limit: int) -> int:
    return max(1, min(limit, WEB_RUNTIME_CHRONICLE_MESSAGE_LIMIT))


def _scrub_internal_save_marker(payload: dict[str, Any]) -> None:
    if payload.get("active_save_id") != _NO_ACCESSIBLE_SAVE_ID:
        return
    payload["active_save_id"] = None
    if "active_save_title" in payload:
        payload["active_save_title"] = None


def _runtime_json_dict(
    state: WebAppState,
    value: object,
    *,
    scrub_for_request: bool = True,
    current_user: UserRecord | None | object = _CURRENT_USER_SENTINEL,
    current_user_role: str | None = None,
) -> dict[str, Any]:
    payload = _json_dict(value)
    if not _looks_like_runtime_model(payload):
        return payload
    payload["saves"] = _save_list_json_for_request(
        state,
        active_save_id=payload.get("active_save_id"),
        current_user=current_user,
    )
    payload["active_scenario_type"] = _active_scenario_type(
        state,
        payload.get("active_save_id"),
    )
    action_choices_enabled = _active_action_choices_enabled(
        state,
        payload.get("active_save_id"),
    )
    if action_choices_enabled is None:
        payload.setdefault("action_choices_enabled", False)
    else:
        payload["action_choices_enabled"] = action_choices_enabled
    character_texts_enabled = _active_character_texts_enabled(
        state,
        payload.get("active_save_id"),
    )
    if character_texts_enabled is None:
        payload.setdefault("character_texts_enabled", False)
    else:
        payload["character_texts_enabled"] = character_texts_enabled
    payload["world_time"] = _world_time_json(
        state,
        payload.get("active_save_id"),
    )
    content_safety = _content_safety_policy_for_request(
        state,
        current_user=current_user,
    )
    if _scenario_exceeds_rating(
        _active_scenario(state, payload.get("active_save_id")),
        allowed_rating=content_safety.rating,
    ):
        from bragi.services.sexual_content_safety import CONTENT_FILTER_TRANSITION

        for key in (
            "active_save_title",
            "scenario_title",
            "scene_title",
            "custom_instructions",
        ):
            if isinstance(payload.get(key), str) and payload[key]:
                payload[key] = CONTENT_FILTER_TRANSITION
    if scrub_for_request:
        _scrub_response_payload_for_request(
            state,
            payload,
            current_user=current_user,
            current_user_role=(
                _current_request_role(state)
                if current_user_role is None
                else current_user_role
            ),
        )
    return payload


def _build_runtime_model_for_save(
    state: WebAppState,
    save_id: str | None,
    *,
    status: str | None = None,
) -> object:
    resolved_save_id = _resolve_runtime_save_id(state, save_id)
    if resolved_save_id is not None:
        _raise_if_save_retired(state, resolved_save_id)
    builder_save_id = _runtime_builder_save_id(state, resolved_save_id)
    build_model = state.runtime.build_model
    kwargs: dict[str, Any] = {}
    if (
        (save_id is not None or _save_access_user(state) is not None)
        and _call_accepts_keyword(build_model, "active_save_id")
    ):
        kwargs["active_save_id"] = builder_save_id
    if status is not None and _call_accepts_keyword(build_model, "status"):
        kwargs["status"] = status
    if kwargs:
        return build_model(**kwargs)
    return build_model()


def _build_runtime_shell_model_for_save(
    state: WebAppState,
    save_id: str | None,
    *,
    status: str | None = None,
) -> object:
    build_shell_model = getattr(state.runtime, "build_shell_model", None)
    if not callable(build_shell_model):
        return _build_runtime_model_for_save(state, save_id, status=status)
    resolved_save_id = _resolve_runtime_save_id(state, save_id)
    if resolved_save_id is not None:
        _raise_if_save_retired(state, resolved_save_id)
    builder_save_id = _runtime_builder_save_id(state, resolved_save_id)
    kwargs: dict[str, Any] = {}
    if (
        (save_id is not None or _save_access_user(state) is not None)
        and _call_accepts_keyword(build_shell_model, "active_save_id")
    ):
        kwargs["active_save_id"] = builder_save_id
    if status is not None and _call_accepts_keyword(build_shell_model, "status"):
        kwargs["status"] = status
    if _call_accepts_keyword(build_shell_model, "chronicle_message_limit"):
        kwargs["chronicle_message_limit"] = WEB_RUNTIME_CHRONICLE_MESSAGE_LIMIT
    if kwargs:
        return build_shell_model(**kwargs)
    return build_shell_model()


def _world_time_json(
    state: WebAppState,
    save_id: object,
) -> dict[str, Any] | None:
    if not isinstance(save_id, str) or not save_id:
        return None
    get_scene_snapshot = getattr(state.repositories, "get_scene_snapshot", None)
    if not callable(get_scene_snapshot):
        return None
    try:
        snapshot = get_scene_snapshot(save_id)
    except Exception:
        return None
    if snapshot is None:
        return None
    from bragi.world_time_model import (
        canonical_world_time_from_snapshot,
        format_world_time,
    )

    world_time = canonical_world_time_from_snapshot(snapshot)
    return {
        "snapshot_id": snapshot.id,
        "day_index": world_time.day_index,
        "day_label": world_time.day_label,
        "phase": world_time.phase,
        "clock_minutes": world_time.clock_minutes,
        "period_label": world_time.period_label,
        "source_message_id": world_time.source_message_id,
        "confidence": world_time.confidence,
        "display": format_world_time(world_time),
    }


def _update_world_time_snapshot(
    state: WebAppState,
    *,
    save_id: str,
    payload: WorldTimeUpdateRequest,
) -> None:
    from bragi.world_time_model import (
        canonical_world_time_from_legacy,
        canonical_world_time_from_snapshot,
        canonical_world_time_from_values,
        legacy_world_time_fields,
    )

    snapshot = state.repositories.get_scene_snapshot(save_id)
    existing_world_time = canonical_world_time_from_snapshot(snapshot)
    edited_fields = {
        field_name
        for field_name in (
            "day_index",
            "day_label",
            "phase",
            "clock_minutes",
            "period_label",
        )
        if _payload_field_was_set(payload, field_name)
    }
    legacy_edited_fields = {
        field_name
        for field_name in (
            "in_world_time",
            "time_of_day",
            "day_of_week",
            "world_day_index",
        )
        if _payload_field_was_set(payload, field_name)
    }
    legacy_world_time = None
    if legacy_edited_fields:
        if (
            "world_day_index" in legacy_edited_fields
            and payload.world_day_index is not None
            and payload.world_day_index < 0
        ):
            raise HTTPException(
                status_code=400,
                detail="World day must be zero or greater",
            )
        legacy_in_world_time = (
            payload.in_world_time
            if "in_world_time" in legacy_edited_fields
            else (snapshot.in_world_time if snapshot else "")
        )
        legacy_time_of_day = (
            ""
            if (
                "in_world_time" in legacy_edited_fields
                and "time_of_day" not in legacy_edited_fields
            )
            else payload.time_of_day
            if "time_of_day" in legacy_edited_fields
            else (snapshot.time_of_day if snapshot else "")
        )
        legacy_day_of_week = (
            payload.day_of_week
            if "day_of_week" in legacy_edited_fields
            else (snapshot.day_of_week if snapshot else "")
        )
        legacy_world_day_index = (
            payload.world_day_index
            if "world_day_index" in legacy_edited_fields
            else existing_world_time.day_index
        )
        legacy_world_time = canonical_world_time_from_legacy(
            in_world_time=legacy_in_world_time,
            time_of_day=legacy_time_of_day,
            day_of_week=legacy_day_of_week,
            world_day_index=legacy_world_day_index,
        )
    if "day_index" in edited_fields:
        day_index = payload.day_index
    elif legacy_world_time is not None and "world_day_index" in legacy_edited_fields:
        day_index = legacy_world_time.day_index
    elif snapshot is None:
        day_index = payload.day_index
    else:
        day_index = existing_world_time.day_index
    if day_index is not None and day_index < 0:
        raise HTTPException(status_code=400, detail="World day must be zero or greater")
    if "day_label" in edited_fields:
        day_label = payload.day_label
    elif (
        legacy_world_time is not None
        and {"in_world_time", "day_of_week"} & legacy_edited_fields
    ):
        day_label = legacy_world_time.day_label
    elif snapshot is None:
        day_label = payload.day_label
    else:
        day_label = existing_world_time.day_label
    if "phase" in edited_fields:
        phase = payload.phase
    elif (
        legacy_world_time is not None
        and {"in_world_time", "time_of_day"} & legacy_edited_fields
    ):
        phase = legacy_world_time.phase
    elif snapshot is None:
        phase = payload.phase
    else:
        phase = existing_world_time.phase
    clock_minutes_value = (
        payload.clock_minutes
        if "clock_minutes" in edited_fields
        else legacy_world_time.clock_minutes
        if legacy_world_time is not None and "in_world_time" in legacy_edited_fields
        else payload.clock_minutes
        if snapshot is None
        else existing_world_time.clock_minutes
    )
    clock_minutes = _validated_clock_minutes(clock_minutes_value)
    period_label = (
        payload.period_label
        if snapshot is None or "period_label" in edited_fields
        else existing_world_time.period_label
    )
    world_time = canonical_world_time_from_values(
        day_index=day_index,
        day_label=_validated_world_time_day_label(day_label),
        phase=_validated_world_time_phase(phase),
        clock_minutes=clock_minutes,
        period_label=_validated_world_time_period_label(period_label),
    )
    legacy_fields = legacy_world_time_fields(world_time)
    should_rewrite_legacy_time = snapshot is None or bool(
        edited_fields & {"day_label", "phase", "clock_minutes", "period_label"}
        or legacy_edited_fields & {"in_world_time", "time_of_day", "day_of_week"}
    )
    if legacy_edited_fields & {"in_world_time", "time_of_day", "day_of_week"}:
        legacy_fields.update(
            {
                "in_world_time": (
                    payload.in_world_time
                    if "in_world_time" in legacy_edited_fields
                    else (snapshot.in_world_time if snapshot else "")
                ),
                "time_of_day": (
                    payload.time_of_day
                    if "time_of_day" in legacy_edited_fields
                    else (snapshot.time_of_day if snapshot else "")
                ),
                "day_of_week": (
                    payload.day_of_week
                    if "day_of_week" in legacy_edited_fields
                    else (snapshot.day_of_week if snapshot else "")
                ),
            }
        )
    world_time_kwargs: dict[str, object] = {
        "world_time_day_index": world_time.day_index,
    }
    if should_rewrite_legacy_time:
        world_time_kwargs.update(
            {
                "world_time_day_label": world_time.day_label,
                "world_time_phase": world_time.phase,
                "world_time_clock_minutes": world_time.clock_minutes,
                "world_time_period_label": world_time.period_label,
            }
        )
    from bragi.services.time_loop_time_policy import TimeLoopTimePolicy

    policy = TimeLoopTimePolicy(state.repositories, save_id=save_id)
    saved = state.repositories.upsert_scene_snapshot(
        save_id=save_id,
        current_location_id=snapshot.current_location_id if snapshot else None,
        situation=snapshot.situation if snapshot else "",
        objective=snapshot.objective if snapshot else "",
        in_world_time=(
            cast(str, legacy_fields["in_world_time"])
            if should_rewrite_legacy_time
            else snapshot.in_world_time
        ),
        time_of_day=(
            cast(str, legacy_fields["time_of_day"])
            if should_rewrite_legacy_time
            else snapshot.time_of_day
        ),
        day_of_week=(
            cast(str, legacy_fields["day_of_week"])
            if should_rewrite_legacy_time
            else snapshot.day_of_week
        ),
        world_day_index=cast(int | None, legacy_fields["world_day_index"]),
        **world_time_kwargs,
        weather=snapshot.weather if snapshot else "",
        mood=snapshot.mood if snapshot else "",
        nearby_objects=snapshot.nearby_objects if snapshot else [],
        hazards=snapshot.hazards if snapshot else [],
        present_character_ids=snapshot.present_character_ids if snapshot else [],
        source_message_id=snapshot.source_message_id if snapshot else None,
        locked_fields=snapshot.locked_fields if snapshot else [],
        snapshot_id=snapshot.id if snapshot else None,
        first_seen_message_id=snapshot.first_seen_message_id if snapshot else None,
        last_updated_message_id=(
            snapshot.last_updated_message_id if snapshot else None
        ),
    )
    if snapshot is not None:
        policy.ensure_baseline(snapshot)
    policy.ensure_baseline(saved)
    policy.sync_current(saved, transition="manual_time_correction")


def _payload_field_was_set(payload: BaseModel, field_name: str) -> bool:
    fields_set = getattr(payload, "model_fields_set", None)
    if fields_set is None:
        fields_set = getattr(payload, "__fields_set__", set())
    return isinstance(fields_set, set | frozenset) and field_name in fields_set


def _validated_world_time_day_label(value: object) -> str:
    from bragi.world_time_model import normalize_world_time_day_label

    label = normalize_world_time_day_label(value)
    if str(value or "").strip() and not label:
        raise HTTPException(
            status_code=400,
            detail="Invalid world day label",
        )
    return label


def _validated_world_time_phase(value: object) -> str:
    from bragi.world_time_model import normalize_time_of_day

    text = str(value or "").strip()
    if not text:
        return ""
    normalized = normalize_time_of_day(text)
    if not normalized:
        raise HTTPException(status_code=400, detail="Invalid time of day")
    return normalized


def _validated_clock_minutes(value: object) -> int | None:
    from bragi.world_time_model import normalize_world_time_clock_minutes

    normalized = normalize_world_time_clock_minutes(value)
    if value is not None and normalized is None:
        raise HTTPException(status_code=400, detail="Invalid clock minutes")
    return normalized


def _validated_world_time_period_label(value: object) -> str:
    from bragi.world_time_model import normalize_world_time_period_label

    if isinstance(value, str) and len(" ".join(value.strip().split())) > 80:
        raise HTTPException(
            status_code=400,
            detail="World time period label must be 80 characters or fewer",
        )
    label = normalize_world_time_period_label(value)
    return label


def _build_chat_history_model_for_save(
    state: WebAppState,
    *,
    selected_filter: str,
    save_id: str | None,
    before_message_id: str | None = None,
    limit: int = WEB_RUNTIME_CHRONICLE_MESSAGE_LIMIT,
) -> object:
    build_model = state.runtime.build_chat_history_model
    resolved_save_id = _resolve_runtime_save_id(state, save_id)
    builder_save_id = _runtime_builder_save_id(state, resolved_save_id)
    kwargs: dict[str, Any] = {"selected_filter": selected_filter}
    if _call_accepts_keyword(build_model, "before_message_id"):
        kwargs["before_message_id"] = before_message_id
    if _call_accepts_keyword(build_model, "limit"):
        kwargs["limit"] = limit
    if (
        (save_id is not None or _save_access_user(state) is not None)
        and _call_accepts_keyword(build_model, "active_save_id")
    ):
        kwargs["active_save_id"] = builder_save_id
    return build_model(**kwargs)


def _build_world_data_model_for_save(
    state: WebAppState,
    save_id: str | None,
) -> object:
    resolved_save_id = _resolve_runtime_save_id(state, save_id)
    builder_save_id = _runtime_builder_save_id(state, resolved_save_id)
    return WorldDataService(
        state.repositories,
        active_save_id=builder_save_id,
        allowed_content_rating=_content_safety_policy_for_request(state).rating,
    ).build_model(active_save_id=builder_save_id if builder_save_id else ...)


def _build_character_registry_model_for_save(
    state: WebAppState,
    save_id: str | None,
) -> object:
    resolved_save_id = _resolve_runtime_save_id(state, save_id)
    builder_save_id = _runtime_builder_save_id(state, resolved_save_id)
    return CharacterRegistryService(
        state.repositories,
        active_save_id=builder_save_id,
        allowed_content_rating=_content_safety_policy_for_request(state).rating,
    ).build_model(active_save_id=builder_save_id if builder_save_id else ...)


def _build_scene_presence_model_for_save(
    state: WebAppState,
    *,
    save_id: str,
    message_id: str,
) -> object:
    from bragi.application.scene_presence import build_scene_presence_model

    try:
        return build_scene_presence_model(
            state.repositories,
            save_id=save_id,
            message_id=message_id,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail.startswith("Unknown active message id:"):
            raise HTTPException(status_code=404, detail=detail) from exc
        raise


def _validated_scene_presence_character_ids(
    state: WebAppState,
    *,
    save_id: str,
    character_ids: list[str],
) -> list[str]:
    active_character_ids = {
        character.id for character in state.repositories.list_characters(save_id)
    }
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_character_id in character_ids:
        character_id = str(raw_character_id).strip()
        if not character_id or character_id in seen:
            continue
        if character_id not in active_character_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown character id: {character_id}",
            )
        normalized.append(character_id)
        seen.add(character_id)
    return normalized


def _replace_current_scene_snapshot_presence(
    state: WebAppState,
    *,
    save_id: str,
    message_id: str,
    character_ids: list[str],
) -> None:
    snapshot = state.repositories.get_scene_snapshot(save_id)
    state.repositories.upsert_scene_snapshot(
        save_id=save_id,
        current_location_id=snapshot.current_location_id if snapshot else None,
        situation=snapshot.situation if snapshot else "",
        objective=snapshot.objective if snapshot else "",
        in_world_time=snapshot.in_world_time if snapshot else "",
        time_of_day=snapshot.time_of_day if snapshot else "",
        day_of_week=snapshot.day_of_week if snapshot else "",
        world_day_index=snapshot.world_day_index if snapshot else None,
        weather=snapshot.weather if snapshot else "",
        mood=snapshot.mood if snapshot else "",
        nearby_objects=snapshot.nearby_objects if snapshot else [],
        hazards=snapshot.hazards if snapshot else [],
        present_character_ids=character_ids,
        source_message_id=snapshot.source_message_id if snapshot else message_id,
        locked_fields=snapshot.locked_fields if snapshot else [],
        snapshot_id=snapshot.id if snapshot else None,
        first_seen_message_id=snapshot.first_seen_message_id if snapshot else None,
        last_updated_message_id=message_id,
    )


def _runtime_builder_save_id(
    state: WebAppState,
    resolved_save_id: str | None,
) -> str | None:
    if resolved_save_id is None and _save_access_user(state) is not None:
        return _NO_ACCESSIBLE_SAVE_ID
    return resolved_save_id


def _current_runtime_save_id(state: WebAppState) -> str | None:
    active_save_id = getattr(state.runtime, "active_save_id", None)
    return active_save_id if isinstance(active_save_id, str) else None


def _raise_unknown_save_if_possible(state: WebAppState, save_id: str) -> None:
    get_save = getattr(state.repositories, "get_save", None)
    if not callable(get_save):
        return
    try:
        save = get_save(save_id)
    except Exception:
        if _auth_context_enabled(state):
            raise
        return
    if save is None or not _request_user_can_access_save(state, save_id):
        raise HTTPException(status_code=404, detail=f"Unknown save id: {save_id}")


def _publish_save_event(
    state: WebAppState,
    save_id: str | None,
    event_type: str,
    payload: object | None = None,
) -> None:
    save_events = getattr(state, "save_events", None)
    publish = getattr(save_events, "publish", None)
    if callable(publish):
        publish(
            save_id,
            event_type,
            to_jsonable(payload or {}),
            owner_user_id=_owner_user_id_for_request(state),
        )


def _publish_runtime_changed_from_model_result(
    state: WebAppState,
    model: object,
    *,
    reason: str,
) -> None:
    if not isinstance(model, dict):
        return
    save_id = model.get("active_save_id")
    if not isinstance(save_id, str) or not save_id:
        return
    _publish_save_event(
        state,
        save_id,
        "runtime_changed",
        {"reason": reason},
    )


def _runtime_media_json_or_raise(
    state: WebAppState,
    value: object,
) -> dict[str, Any]:
    payload = _runtime_json_dict(state, value, scrub_for_request=False)
    error = _runtime_model_error(payload)
    if error:
        raise RuntimeError(error)
    return payload


def _scrub_response_payload_for_request(
    state: WebAppState,
    payload: object,
    *,
    current_user: UserRecord | None | object = _CURRENT_USER_SENTINEL,
    current_user_role: str | None = None,
) -> None:
    resolved_user_role = (
        _current_request_role(state)
        if current_user_role is None
        else current_user_role
    )
    _scrub_response_payload_for_role(
        payload,
        current_user_role=resolved_user_role,
        content_safety=_content_safety_policy_for_request(
            state,
            current_user=current_user,
        ),
    )


def _scrub_response_payload_for_role(
    payload: object,
    *,
    current_user_role: str | None,
    content_safety: ContentSafetyPolicy,
) -> None:
    if isinstance(payload, dict):
        _scrub_media_payload_for_rating(
            payload,
            allowed_rating=content_safety.rating,
        )
        _scrub_nested_media_references(
            payload,
            allowed_rating=content_safety.rating,
        )
        _scrub_message_payload_for_role(
            payload,
            current_user_role=current_user_role,
            content_safety=content_safety,
        )
        _scrub_character_payload_for_rating(
            payload,
            allowed_rating=content_safety.rating,
        )
        for value in payload.values():
            _scrub_response_payload_for_role(
                value,
                current_user_role=current_user_role,
                content_safety=content_safety,
            )
        return
    if isinstance(payload, list):
        payload[:] = [
            item
            for item in payload
            if not (
                isinstance(item, dict)
                and _serialized_media_reference_exceeds_rating(
                    item,
                    allowed_rating=content_safety.rating,
                )
            )
        ]
        for item in payload:
            _scrub_response_payload_for_role(
                item,
                current_user_role=current_user_role,
                content_safety=content_safety,
            )


def _scrub_message_payload_for_role(
    payload: dict[str, Any],
    *,
    current_user_role: str | None,
    content_safety: ContentSafetyPolicy,
) -> None:
    _scrub_latest_message_preview(payload, content_safety=content_safety)
    if _looks_like_chronicle_message(payload):
        blocked_action_ids = _blocked_chronicle_action_ids(current_user_role)
        if current_user_role != "admin":
            payload.pop("debug_prompt", None)
            payload.pop("debug_provider_payload", None)
    elif _looks_like_character_text_message(payload):
        blocked_action_ids = _blocked_character_text_action_ids(current_user_role)
    elif _looks_like_action_choice(payload):
        blocked_action_ids = frozenset()
    else:
        return
    body = payload.get("body")
    minimum_rating = payload.get("content_rating")
    if isinstance(body, str) and isinstance(minimum_rating, str):
        from bragi.application.chronicle import parse_message_markdown
        from bragi.content_rating_instructions import content_rating_exceeds
        from bragi.services.sexual_content_safety import CONTENT_FILTER_TRANSITION

        if content_rating_exceeds(
            minimum_rating=minimum_rating,
            allowed_rating=content_safety.rating,
        ):
            payload["body"] = CONTENT_FILTER_TRANSITION
            if "markdown_blocks" in payload:
                payload["markdown_blocks"] = to_jsonable(
                    parse_message_markdown(CONTENT_FILTER_TRANSITION)
                )
            payload["actions"] = []
            payload.pop("debug_prompt", None)
            payload.pop("debug_provider_payload", None)
    actions = payload.get("actions")
    if not isinstance(actions, list) or not blocked_action_ids:
        return
    payload["actions"] = [
        action
        for action in actions
        if not (
            isinstance(action, dict)
            and isinstance(action.get("action_id"), str)
            and action["action_id"] in blocked_action_ids
        )
    ]


def _scrub_latest_message_preview(
    payload: dict[str, Any],
    *,
    content_safety: ContentSafetyPolicy,
) -> None:
    body = payload.get("latest_message_body")
    minimum_rating = payload.get("latest_message_content_rating")
    if not isinstance(body, str) or not isinstance(minimum_rating, str):
        return
    from bragi.application.chronicle import parse_message_markdown
    from bragi.content_rating_instructions import content_rating_exceeds
    from bragi.services.sexual_content_safety import CONTENT_FILTER_TRANSITION

    if not content_rating_exceeds(
        minimum_rating=minimum_rating,
        allowed_rating=content_safety.rating,
    ):
        return
    payload["latest_message_body"] = CONTENT_FILTER_TRANSITION
    if "latest_message_markdown_blocks" in payload:
        payload["latest_message_markdown_blocks"] = to_jsonable(
            parse_message_markdown(CONTENT_FILTER_TRANSITION)
        )


def _scrub_character_payload_for_rating(
    payload: dict[str, Any],
    *,
    allowed_rating: str,
) -> None:
    if not (
        isinstance(payload.get("character_id"), str)
        and isinstance(payload.get("content_rating"), str)
        and "relationships_json" in payload
    ):
        return
    from bragi.content_rating_instructions import content_rating_exceeds
    from bragi.services.sexual_content_safety import CONTENT_FILTER_TRANSITION

    if not content_rating_exceeds(
        minimum_rating=payload["content_rating"],
        allowed_rating=allowed_rating,
    ):
        return
    for key in (
        "name",
        "aliases_text",
        "role",
        "age",
        "known_state",
        "history",
        "appearance",
        "visual_notes",
        "current_clothing",
        "personality",
        "voice",
        "texting_style",
        "goals",
        "motivations",
        "current_intent",
        "boundaries",
        "attitude_toward_player",
        "cooperation_conditions",
        "status",
        "private_notes",
        "contact_name",
    ):
        if isinstance(payload.get(key), str) and payload[key]:
            payload[key] = CONTENT_FILTER_TRANSITION
    payload["relationships_json"] = "{}"


def _scrub_nested_media_references(
    payload: dict[str, Any],
    *,
    allowed_rating: str,
) -> None:
    for key, value in tuple(payload.items()):
        if isinstance(value, dict) and _serialized_media_reference_exceeds_rating(
            value,
            allowed_rating=allowed_rating,
        ):
            payload[key] = None
        elif isinstance(value, list):
            value[:] = [
                item
                for item in value
                if not (
                    isinstance(item, dict)
                    and _serialized_media_reference_exceeds_rating(
                        item,
                        allowed_rating=allowed_rating,
                    )
                )
            ]


def _serialized_media_reference_exceeds_rating(
    payload: dict[str, Any],
    *,
    allowed_rating: str,
) -> bool:
    looks_like_media_reference = (
        "media_asset_id" in payload
        and ("prompt_preview" in payload or "prompt" in payload)
    ) or (
        "path" in payload
        and "provider" in payload
        and ("prompt_preview" in payload or "prompt" in payload)
    )
    return looks_like_media_reference and _serialized_media_exceeds_rating(
        payload,
        allowed_rating=allowed_rating,
    )


def _scrub_media_payload_for_rating(
    payload: dict[str, Any],
    *,
    allowed_rating: str,
) -> None:
    if not any(
        key in payload
        for key in (
            "latest_scene_media",
            "latest_scene_image",
            "media_history",
            "image_history",
        )
    ):
        return
    media_history = _filtered_serialized_media_history(
        payload.get("media_history"),
        allowed_rating=allowed_rating,
    )
    image_history = _filtered_serialized_media_history(
        payload.get("image_history"),
        allowed_rating=allowed_rating,
    )
    if "media_history" in payload:
        payload["media_history"] = media_history
    if "image_history" in payload:
        payload["image_history"] = image_history
    if "latest_scene_media" in payload:
        payload["latest_scene_media"] = media_history[0] if media_history else None
    if "latest_scene_image" in payload:
        payload["latest_scene_image"] = image_history[0] if image_history else None
    character_reference = payload.get("character_reference_image")
    if isinstance(character_reference, dict) and _serialized_media_exceeds_rating(
        character_reference,
        allowed_rating=allowed_rating,
    ):
        payload["character_reference_image"] = None


def _filtered_serialized_media_history(
    value: object,
    *,
    allowed_rating: str,
) -> list[object]:
    if not isinstance(value, list):
        return []
    return [
        item
        for item in value
        if not (
            isinstance(item, dict)
            and _serialized_media_exceeds_rating(
                item,
                allowed_rating=allowed_rating,
            )
        )
    ]


def _serialized_media_exceeds_rating(
    payload: dict[str, Any],
    *,
    allowed_rating: str,
) -> bool:
    from bragi.content_rating_instructions import content_rating_exceeds

    metadata = payload.get("metadata")
    minimum_rating = payload.get("content_rating")
    if not isinstance(minimum_rating, str):
        minimum_rating = (
        metadata.get("content_rating", "unclassified")
        if isinstance(metadata, dict)
        else "unclassified"
        )
    if content_rating_exceeds(
        minimum_rating=str(minimum_rating),
        allowed_rating=allowed_rating,
    ):
        return True
    source_media = payload.get("source_media")
    return isinstance(source_media, dict) and _serialized_media_exceeds_rating(
        source_media,
        allowed_rating=allowed_rating,
    )


def _looks_like_chronicle_message(payload: dict[str, Any]) -> bool:
    return (
        isinstance(payload.get("message_id"), str)
        and isinstance(payload.get("role"), str)
    )


def _looks_like_action_choice(payload: dict[str, Any]) -> bool:
    return (
        isinstance(payload.get("choice_id"), str)
        and isinstance(payload.get("ordinal"), int)
    )


def _looks_like_character_text_message(payload: dict[str, Any]) -> bool:
    return (
        isinstance(payload.get("id"), str)
        and isinstance(payload.get("thread_id"), str)
        and isinstance(payload.get("sender"), str)
    )


def _blocked_chronicle_action_ids(current_user_role: str | None) -> frozenset[str]:
    if current_user_role == "admin":
        return frozenset()
    if current_user_role == "child":
        return _CHILD_CHRONICLE_ACTIONS_BLOCKED
    return _NON_ADMIN_CHRONICLE_ACTIONS_BLOCKED


def _blocked_character_text_action_ids(
    current_user_role: str | None,
) -> frozenset[str]:
    if current_user_role == "child":
        return _CHILD_CHARACTER_TEXT_ACTIONS_BLOCKED
    return frozenset()


def _character_registry_json_or_raise(
    state: WebAppState,
    *,
    save_id: str,
) -> dict[str, Any]:
    model = CharacterRegistryService(
        state.repositories,
        active_save_id=save_id,
        allowed_content_rating=_content_safety_policy_for_request(state).rating,
    ).build_model(active_save_id=save_id)
    payload = _json_dict(model)
    error = payload.get("error")
    if isinstance(error, str) and error:
        raise RuntimeError(error)
    return payload


def _looks_like_runtime_model(payload: dict[str, Any]) -> bool:
    return "chronicle" in payload and "saves" in payload


def _initial_media_progress_label(
    state: WebAppState,
    requested_save_id: str | None,
) -> str:
    del state, requested_save_id
    return "Generating opening image"


def _active_scenario_type(state: WebAppState, save_id: object) -> str | None:
    if not isinstance(save_id, str) or not save_id:
        return None
    get_save = getattr(state.repositories, "get_save", None)
    get_scenario = getattr(state.repositories, "get_scenario", None)
    if not callable(get_save) or not callable(get_scenario):
        return None
    try:
        save = get_save(save_id)
    except Exception:
        return None
    scenario_id = getattr(save, "scenario_id", None)
    if not isinstance(scenario_id, str) or not scenario_id:
        return None
    try:
        scenario = get_scenario(scenario_id)
    except Exception:
        return None
    scenario_type = getattr(scenario, "type", None)
    return scenario_type if isinstance(scenario_type, str) else None


def _active_action_choices_enabled(
    state: WebAppState,
    save_id: object,
) -> bool | None:
    scenario = _active_scenario(state, save_id)
    if scenario is None:
        return None
    try:
        return _scenario_action_choices_enabled(scenario)
    except Exception:
        return None


def _active_character_texts_enabled(
    state: WebAppState,
    save_id: object,
) -> bool | None:
    if not isinstance(save_id, str) or not save_id:
        return None
    try:
        from bragi.services.character_text_service import CharacterTextService

        return CharacterTextService(
            repositories=state.repositories,
            providers=state.providers,
        ).is_enabled(save_id)
    except Exception:
        return None


def _prompt_inspection_store_if_enabled(state: WebAppState) -> Any | None:
    try:
        if not state.repositories.get_app_setting("debug_logging_enabled"):
            return None
    except Exception:
        return None
    return getattr(
        getattr(state, "runtime", None),
        "prompt_inspection_store",
        None,
    )


def _scenario_action_choices_enabled(scenario: object) -> bool:
    if getattr(scenario, "type", None) == "choose_your_own_adventure":
        return True
    content_json = getattr(scenario, "content_json", None)
    if not isinstance(content_json, str):
        return False
    try:
        content = json.loads(content_json)
    except json.JSONDecodeError:
        return False
    if not isinstance(content, dict):
        return False
    return content.get("action_choices_enabled") is True


def _active_scenario(state: WebAppState, save_id: object) -> object | None:
    if not isinstance(save_id, str) or not save_id:
        return None
    get_save = getattr(state.repositories, "get_save", None)
    get_scenario = getattr(state.repositories, "get_scenario", None)
    if not callable(get_save) or not callable(get_scenario):
        return None
    try:
        save = get_save(save_id)
    except Exception:
        return None
    scenario_id = getattr(save, "scenario_id", None)
    if not isinstance(scenario_id, str) or not scenario_id:
        return None
    try:
        scenario: object = get_scenario(scenario_id)
        return scenario
    except Exception:
        return None


def _resolve_chat_save_id(
    state: WebAppState,
    requested_save_id: str | None,
) -> str | None:
    if requested_save_id is not None:
        _raise_unknown_save_if_possible(state, requested_save_id)
        return requested_save_id
    return _resolve_runtime_save_id(state, None)


def _require_save_id(requested_save_id: str | None) -> str:
    if requested_save_id:
        state = _REQUEST_STATE.get()
        if state is not None:
            _raise_unknown_save_if_possible(state, requested_save_id)
        return requested_save_id
    raise HTTPException(status_code=400, detail=_SAVE_ID_REQUIRED_DETAIL)


def _job_for_save_or_404(
    state: WebAppState,
    job_id: str,
    requested_save_id: str | None,
) -> JobRecord:
    record = state.jobs.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    if record.save_id is None:
        if requested_save_id is None:
            if not _request_user_can_access_job(state, record):
                raise HTTPException(status_code=404, detail="Unknown job")
            return record
        raise HTTPException(status_code=404, detail="Unknown job")
    if requested_save_id == record.save_id:
        if not _request_user_can_access_job(state, record):
            raise HTTPException(status_code=404, detail="Unknown job")
        return record
    raise HTTPException(status_code=404, detail="Unknown job")


def _persisted_job_for_save_or_404(
    state: WebAppState,
    job_id: str,
    requested_save_id: str | None,
) -> Any:
    record = state.repositories.get_persisted_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    if record.save_id is None:
        if requested_save_id is None and _request_user_can_access_job(state, record):
            return record
        raise HTTPException(status_code=404, detail="Unknown job")
    if requested_save_id == record.save_id and _request_user_can_access_job(
        state,
        record,
    ):
        return record
    raise HTTPException(status_code=404, detail="Unknown job")


def _terminal_job_summary(record: Any, *, step_count: int) -> dict[str, Any]:
    snapshot = getattr(record, "diagnostics", None)
    if not isinstance(snapshot, dict):
        from bragi.services.job_diagnostics import build_job_diagnostic_snapshot

        snapshot = build_job_diagnostic_snapshot(record)
    request = snapshot.get("request")
    request = request if isinstance(request, dict) else {}
    provider = snapshot.get("provider")
    provider = provider if isinstance(provider, dict) else {}
    return {
        "id": record.id,
        "type": record.type,
        "save_id": record.save_id,
        "status": record.status,
        "created_at": record.created_at,
        "started_at": record.started_at,
        "completed_at": record.completed_at,
        "duration_ms": record.duration_ms,
        "queue_wait_ms": _queue_wait_ms(record),
        "step_count": step_count,
        "error": _terminal_job_error(record),
        "origin": request.get("origin"),
        "provider": provider.get("provider"),
        "model": provider.get("model"),
        "detail_available": getattr(record, "diagnostics", None) is not None,
    }


def _terminal_job_error(record: Any) -> str | None:
    if not record.error:
        return None
    if record.status == "cancelled":
        return "Job was cancelled."
    return PUBLIC_JOB_FAILURE_ERROR


def _queue_wait_ms(record: Any) -> int | None:
    return _duration_between_ms(record.created_at, record.started_at)


def _duration_between_ms(start: object, end: object) -> int | None:
    start_dt = _parse_diagnostic_timestamp(start)
    end_dt = _parse_diagnostic_timestamp(end)
    if start_dt is None or end_dt is None:
        return None
    return max(0, round((end_dt - start_dt).total_seconds() * 1000))


def _job_step_summary(step: Any) -> dict[str, Any]:
    return {
        "id": step.id,
        "name": step.name,
        "status": step.status,
        "provider": step.provider,
        "model": step.model,
        "task": step.task,
        "started_at": step.started_at,
        "completed_at": step.completed_at,
        "duration_ms": step.duration_ms,
        "metadata": step.metadata,
    }


def _raise_unless_chat_cancel_allowed(
    state: WebAppState,
    save_id: str,
) -> None:
    _raise_unless_save_action_allowed(state, save_id, "chat")
    user = _save_access_user(state)
    if user is None or user.role != "child":
        return
    active_chat = _active_chat_turn_for_save(state.jobs.list_active(), save_id)
    if active_chat is None or active_chat.creator_user_id == user.id:
        return
    raise HTTPException(status_code=403, detail="Job cancellation is not allowed")


def _raise_unless_job_cancel_allowed(
    state: WebAppState,
    record: JobRecord,
) -> None:
    if record.save_id is not None:
        _raise_if_save_retired(state, record.save_id)
    user = _save_access_user(state)
    if user is None or user.role != "child":
        return
    if record.creator_user_id == user.id:
        return
    raise HTTPException(status_code=403, detail="Job cancellation is not allowed")


def _request_user_can_access_job(state: WebAppState, record: Any) -> bool:
    if not _auth_context_enabled(state):
        return True
    user = _current_request_user()
    if user is None:
        return False
    if user.role == "admin":
        return True
    if record.save_id is not None:
        if record.creator_user_id == user.id and record.status in ACTIVE_JOB_STATUSES:
            return True
        return bool(state.repositories.user_can_access_save(user, record.save_id))
    return bool(record.creator_user_id == user.id)


def _job_summary_for_request(
    state: WebAppState,
    record: JobRecord,
    *,
    current_user: UserRecord | None | object = _CURRENT_USER_SENTINEL,
    current_user_role: str | None = None,
) -> dict[str, Any]:
    resolved_user_role = (
        _current_request_role(state)
        if current_user_role is None
        else current_user_role
    )
    payload = cast(
        dict[str, Any],
        _runtime_payload_for_request(
            state,
            job_summary(record),
            current_user=current_user,
            current_user_role=resolved_user_role,
        ),
    )
    _scrub_response_payload_for_request(
        state,
        payload,
        current_user=current_user,
        current_user_role=resolved_user_role,
    )
    return payload


def _job_event_payload_for_request(
    state: WebAppState,
    record: JobRecord,
    event: dict[str, Any],
    *,
    current_user: UserRecord | None | object = _CURRENT_USER_SENTINEL,
    current_user_role: str | None,
) -> Any:
    resolved_user_role = (
        _current_request_role(state)
        if current_user_role is None
        else current_user_role
    )
    payload = _runtime_payload_for_request(
        state,
        job_event_payload(record, event),
        current_user=current_user,
        current_user_role=resolved_user_role,
    )
    _scrub_response_payload_for_request(
        state,
        payload,
        current_user=current_user,
        current_user_role=resolved_user_role,
    )
    return payload


def _runtime_payload_for_request(
    state: WebAppState,
    payload: Any,
    *,
    current_user: UserRecord | None | object,
    current_user_role: str | None,
) -> Any:
    if isinstance(payload, dict) and _looks_like_runtime_model(payload):
        return _runtime_json_dict(
            state,
            payload,
            scrub_for_request=False,
            current_user=current_user,
            current_user_role=current_user_role,
        )
    if isinstance(payload, dict):
        return {
            key: _runtime_payload_for_request(
                state,
                value,
                current_user=current_user,
                current_user_role=current_user_role,
            )
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [
            _runtime_payload_for_request(
                state,
                value,
                current_user=current_user,
                current_user_role=current_user_role,
            )
            for value in payload
        ]
    return payload


def _chat_submission_status(
    state: WebAppState,
    save_id: str | None,
) -> dict[str, Any]:
    if save_id is None:
        return {
            "save_id": None,
            "can_submit": False,
            "reason": "no_save",
            "blocking_job_id": None,
            "blocking_job_status": None,
        }
    blocking = _active_chat_turn_for_save(
        state.jobs.list_active(save_id=save_id),
        save_id,
    )
    if blocking is not None:
        return {
            "save_id": save_id,
            "can_submit": False,
            "reason": "chat_turn_active",
            "blocking_job_id": blocking.id,
            "blocking_job_status": blocking.status,
        }
    return {
        "save_id": save_id,
        "can_submit": True,
        "reason": None,
        "blocking_job_id": None,
        "blocking_job_status": None,
    }


def _active_chat_turn_for_save(
    jobs: list[JobRecord],
    save_id: str,
) -> JobRecord | None:
    return next(
        (
            job
            for job in jobs
            if _job_blocks_chat_submission(job, save_id)
        ),
        None,
    )


def _job_blocks_chat_submission(job: JobRecord, save_id: str) -> bool:
    return job.save_id == save_id and (
        job.type in _CHAT_JOB_TYPES
        or job.exclusive_key == _chat_turn_exclusive_key(save_id)
    )


def _should_cancel_runtime_chat_for_job(job: JobRecord) -> bool:
    return (
        job.status in {"queued", "running"}
        and job.save_id is not None
        and _job_blocks_chat_submission(job, job.save_id)
    )


def _cancel_runtime_chat_for_job(state: WebAppState, job: JobRecord) -> bool:
    if job.save_id is None:
        return False
    cancel_active_submit = getattr(state.runtime, "cancel_active_submit", None)
    if not callable(cancel_active_submit):
        return False
    return bool(cancel_active_submit(save_id=job.save_id))


def _chat_turn_exclusive_key(save_id: str | None) -> str | None:
    return f"chat_turn:{save_id}" if save_id is not None else None


def _action_choice_operation_queue_key(
    save_id: str,
    narrator_message_id: str,
) -> str:
    return f"action_choices:{save_id}:{narrator_message_id}"


async def _cancel_action_choice_jobs_for_save(
    state: WebAppState,
    save_id: str,
    *,
    narrator_message_id: str | None = None,
) -> None:
    expected_queue_key = (
        _action_choice_operation_queue_key(save_id, narrator_message_id)
        if narrator_message_id is not None
        else None
    )
    for job in state.jobs.list_active(save_id=save_id):
        if job.type not in {"action_choice_generate", "action_choice_regenerate"}:
            continue
        if (
            expected_queue_key is not None
            and job.operation_queue_key != expected_queue_key
        ):
            continue
        await state.jobs.cancel(job.id)
    invalidate = getattr(
        state.repositories,
        "invalidate_message_action_choice_generations",
        None,
    )
    if callable(invalidate):
        invalidate(save_id)


def _character_text_thread_job_key(thread_id: str) -> str:
    return f"character_text_thread:{thread_id}"


def _web_maintenance_scheduler_enabled(*, provided_state: bool) -> bool:
    if os.environ.get("BRAGI_WEB_DISABLE_SCHEDULER") == "1":
        return False
    enabled_in_tests = (
        os.environ.get("BRAGI_WEB_ENABLE_SCHEDULER_IN_TESTS") == "1"
        or os.environ.get("BRAGI_WEB_ENABLE_SCHEDULER_FOR_TEST_STATE") == "1"
    )
    if (
        "PYTEST_CURRENT_TEST" in os.environ
        and not enabled_in_tests
    ):
        return False
    if provided_state:
        return enabled_in_tests
    return True


async def _create_job_summary(
    state: WebAppState,
    job_type: str,
    worker: JobWorker,
    *,
    save_id: str | None = None,
    exclusive_key: str | None = None,
    operation_queue_key: str | None = None,
    lock_runtime: bool | None = None,
) -> dict[str, Any]:
    should_lock_runtime = lock_runtime if lock_runtime is not None else save_id is None
    guarded_worker = (
        _locked_job_worker(state, worker) if should_lock_runtime else worker
    )
    try:
        return _job_summary_for_request(
            state,
            await state.jobs.create(
                job_type,
                guarded_worker,
                save_id=save_id,
                creator_user_id=_owner_user_id_for_request(state),
                exclusive_key=exclusive_key,
                operation_queue_key=operation_queue_key,
            ),
        )
    except JobRegistryExclusiveKeyError as exc:
        if job_type in _CHAT_JOB_TYPES or exc.exclusive_key.startswith("chat_turn:"):
            raise HTTPException(
                status_code=409,
                detail=_CHAT_TURN_ACTIVE_DETAIL,
            ) from exc
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except JobRegistryFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


def _chat_turn_request_fingerprint(
    operation: str,
    normalized_payload: dict[str, object],
) -> str:
    encoded = json.dumps(
        {"operation": operation, "payload": normalized_payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _chat_turn_submission_replay(
    state: WebAppState,
    *,
    save_id: str,
    client_turn_id: str,
    operation: str,
    request_fingerprint: str,
) -> dict[str, Any] | None:
    get_submission = getattr(
        state.repositories,
        "get_chat_turn_submission",
        None,
    )
    if not callable(get_submission):
        return None
    submission = get_submission(
        save_id=save_id,
        client_turn_id=client_turn_id,
    )
    if submission is None:
        return None
    if (
        submission.operation != operation
        or submission.request_fingerprint != request_fingerprint
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "client_turn_id was already used for a different chat submission."
            ),
        )
    live = state.jobs.get(submission.job.id)
    if live is not None:
        return _job_summary_for_request(state, live)
    persisted = submission.job
    result: object = persisted.result
    if persisted.status == "succeeded":
        result = {
            "kind": "chat_turn_replay",
            "save_id": save_id,
            "player_message_id": submission.player_message_id,
            "narrator_message_id": submission.narrator_message_id,
            "requires_full_refresh": True,
        }
    return {
        "id": persisted.id,
        "type": persisted.type,
        "save_id": persisted.save_id,
        "status": persisted.status,
        "completion_level": (
            RESPONSE_COMMITTED if submission.narrator_message_id else None
        ),
        "result": result,
        "error": persisted.error,
        "latest_progress": None,
    }


def _link_chat_turn_submission_messages(
    state: WebAppState,
    handle: JobHandle,
    turn: object,
) -> None:
    link_messages = getattr(
        state.repositories,
        "link_chat_turn_submission_messages",
        None,
    )
    if callable(link_messages):
        link_messages(
            job_id=handle.record.id,
            player_message_id=getattr(turn, "player_message_id", None),
            narrator_message_id=getattr(turn, "narrator_message_id", None),
        )


async def _create_idempotent_chat_job_summary(
    state: WebAppState,
    worker: JobWorker,
    *,
    save_id: str,
    client_turn_id: str,
    operation: str,
    request_fingerprint: str,
) -> dict[str, Any]:
    job_id = uuid4().hex
    creator_user_id = _owner_user_id_for_request(state)
    exclusive_key = _chat_turn_exclusive_key(save_id)

    def persist(record: JobRecord) -> None:
        create_submission = getattr(
            state.repositories,
            "create_chat_turn_submission_job",
            None,
        )
        if not callable(create_submission):
            return
        create_submission(
            save_id=save_id,
            client_turn_id=client_turn_id,
            operation=operation,
            request_fingerprint=request_fingerprint,
            creator_user_id=creator_user_id,
            job_id=record.id,
            payload={
                "source": "web",
                "exclusive_key": exclusive_key or "",
                "operation": operation,
            },
        )

    try:
        supports_durable_submission = callable(
            getattr(
                state.repositories,
                "create_chat_turn_submission_job",
                None,
            )
        )
        record = await state.jobs.create(
            "chat_turn",
            worker,
            save_id=save_id,
            creator_user_id=creator_user_id,
            exclusive_key=exclusive_key,
            job_id=job_id,
            persist_before_start=(persist if supports_durable_submission else None),
        )
        return _job_summary_for_request(state, record)
    except (sqlite3.IntegrityError, JobRegistryExclusiveKeyError) as exc:
        replay = _chat_turn_submission_replay(
            state,
            save_id=save_id,
            client_turn_id=client_turn_id,
            operation=operation,
            request_fingerprint=request_fingerprint,
        )
        if replay is not None:
            return replay
        if isinstance(exc, JobRegistryExclusiveKeyError):
            raise HTTPException(
                status_code=409,
                detail=_CHAT_TURN_ACTIVE_DETAIL,
            ) from exc
        raise
    except JobRegistryFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


def _locked_job_worker(state: WebAppState, worker: JobWorker) -> JobWorker:
    async def locked(handle: JobHandle) -> Any:
        async with state.lock.async_access():
            return await worker(handle)

    return locked


def _post_turn_operation_queue_key(save_id: str, narrator_message_id: str) -> str:
    return f"post_turn:{save_id}:{narrator_message_id}"


def _active_background_post_turn_jobs(
    state: WebAppState,
    *,
    save_id: str,
) -> list[JobRecord]:
    return [
        job
        for job in state.jobs.list_active(save_id=save_id)
        if job.type == "post_turn_background"
    ]


_POST_TURN_CATCHUP_STATUS_TEXT = {
    "waiting": "Waiting for prior turn continuity",
    "succeeded": "Prior turn continuity is ready",
    "failed": "Prior turn continuity catch-up failed; repair will retry",
    "retry_pending": (
        "Prior turn continuity is still catching up; retry pending"
    ),
    "cancelled": "Prior turn continuity catch-up cancelled",
}


def _post_turn_catchup_progress(
    status: str,
    *,
    job_ids: list[str],
) -> dict[str, object]:
    status_text = _POST_TURN_CATCHUP_STATUS_TEXT.get(status)
    if status_text is None:
        raise ValueError(f"Unsupported post-turn catch-up status: {status}")
    degraded = status in {"failed", "retry_pending"}
    return {
        "kind": "post_turn_catchup",
        "status": status,
        "status_text": status_text,
        "continuity_degraded": degraded,
        "retry_pending": degraded,
        "job_ids": list(job_ids),
        "jobs": [
            {
                "name": "post_turn_catchup",
                "status": status,
                "category": "continuity",
            }
        ],
    }


async def _wait_for_background_post_turn_catchup(
    state: WebAppState,
    handle: JobHandle,
    *,
    save_id: str,
) -> None:
    active_jobs = _active_background_post_turn_jobs(state, save_id=save_id)
    list_outbox_steps = getattr(
        state.repositories,
        "list_post_turn_outbox_steps",
        None,
    )
    incomplete_outbox = (
        list_outbox_steps(
            save_id=save_id,
            statuses=("pending", "running", "failed"),
        )
        if callable(list_outbox_steps)
        else ()
    )
    if not active_jobs and not incomplete_outbox:
        return
    job_ids = [job.id for job in active_jobs]
    await handle.event(
        "progress",
        _post_turn_catchup_progress("waiting", job_ids=job_ids),
    )
    try:
        async with asyncio.timeout(10):
            for job in active_jobs:
                try:
                    await asyncio.shield(
                        state.jobs.wait_for_completion_level(
                            job.id,
                            CONTINUITY_READY,
                        )
                    )
                except Exception:
                    # A failed prior web job still leaves durable outbox work
                    # eligible for repair within the same preflight budget.
                    continue
            recover = getattr(
                state.runtime,
                "run_post_turn_outbox_recovery",
                None,
            )
            result = (
                await recover(active_save_id=save_id)
                if callable(recover)
                else None
            )
    except asyncio.CancelledError:
        await handle.event(
            "progress",
            _post_turn_catchup_progress("cancelled", job_ids=job_ids),
        )
        raise
    except Exception as exc:
        observe(
            "web.post_turn_catchup_failed",
            level="error",
            save_id=save_id,
            prior_job_ids=job_ids,
            **error_fields(exc),
        )
        await handle.event(
            "progress",
            _post_turn_catchup_progress("failed", job_ids=job_ids),
        )
        return
    if result is not None and result.continuity_degraded:
        await handle.event(
            "progress",
            _post_turn_catchup_progress("retry_pending", job_ids=job_ids),
        )
        return
    await handle.event(
        "progress",
        _post_turn_catchup_progress("succeeded", job_ids=job_ids),
    )


async def _queue_post_turn_jobs_background(
    state: WebAppState,
    handle: JobHandle,
    *,
    save_id: str,
    player_message_id: str,
    narrator_message_id: str,
    turn_revision: object | None = None,
    prepared_action_choices: object | None = None,
    prior_phase_jobs: list[dict[str, str]] | None = None,
    current_user_id: str | None = None,
) -> JobRecord | None:
    async def worker(post_turn_handle: JobHandle) -> Any:
        await post_turn_handle.advance_completion_level(RESPONSE_COMMITTED)
        optional_jobs: list[JobRecord] = []
        await _queue_prepared_action_choices_if_available(
            state,
            post_turn_handle,
            save_id=save_id,
            narrator_message_id=narrator_message_id,
            prepared_action_choices=prepared_action_choices,
            current_user_id=current_user_id,
        )
        result = await _run_post_turn_jobs_with_ordered_progress(
            state,
            post_turn_handle,
            save_id=save_id,
            player_message_id=player_message_id,
            narrator_message_id=narrator_message_id,
            turn_revision=turn_revision,
            prior_phase_jobs=None,
            current_user_id=current_user_id,
            optional_jobs=optional_jobs,
        )
        if bool(getattr(result, "continuity_degraded", False)):
            await post_turn_handle.event("runtime", to_jsonable(result))
            raise RuntimeError("Post-turn continuity is degraded; retry pending")
        await post_turn_handle.advance_completion_level(CONTINUITY_READY)
        for optional_job in optional_jobs:
            if optional_job.task is not None:
                await asyncio.shield(optional_job.task)
        await post_turn_handle.advance_completion_level(
            OPTIONAL_ENRICHMENTS_COMPLETE
        )
        return result

    try:
        record = await state.jobs.create(
            "post_turn_background",
            worker,
            save_id=save_id,
            creator_user_id=_owner_user_id_for_request(state),
            operation_queue_key=_post_turn_operation_queue_key(
                save_id,
                narrator_message_id,
            ),
        )
    except (JobRegistryExclusiveKeyError, JobRegistryFullError) as exc:
        await handle.event(
            "post_turn_job",
            {
                "status": "inline_fallback",
                "reason": str(exc),
            },
        )
        return None
    await handle.event(
        "post_turn_job",
        {
            "status": "queued",
            "job_id": record.id,
            "save_id": save_id,
            "prior_phase_jobs": prior_phase_jobs or [],
        },
    )
    return record


async def _run_post_turn_jobs_with_ordered_progress(
    state: WebAppState,
    handle: JobHandle,
    *,
    save_id: str,
    player_message_id: str,
    narrator_message_id: str,
    turn_revision: object | None = None,
    prior_phase_jobs: list[dict[str, str]] | None = None,
    current_user_id: str | None = None,
    optional_jobs: list[JobRecord] | None = None,
) -> Any:
    done = object()
    progress_queue: asyncio.Queue[object] = asyncio.Queue()

    await handle.event("progress", _initial_post_turn_progress(prior_phase_jobs))

    async def pump_progress() -> None:
        while True:
            progress = await progress_queue.get()
            if progress is done:
                return
            await handle.event(
                "progress",
                _post_turn_progress_with_prior_jobs(progress, prior_phase_jobs),
            )

    def progress_callback(progress: object) -> None:
        progress_queue.put_nowait(progress)

    pump_task = asyncio.create_task(pump_progress())
    try:
        kwargs: dict[str, object] = {
            "save_id": save_id,
            "player_message_id": player_message_id,
            "narrator_message_id": narrator_message_id,
            "progress_callback": progress_callback,
        }
        if turn_revision is not None and _call_accepts_keyword(
            state.runtime.run_post_turn_jobs,
            "turn_revision",
        ):
            kwargs["turn_revision"] = turn_revision
        if _call_accepts_keyword(
            state.runtime.run_post_turn_jobs,
            "current_user_id",
        ):
            kwargs["current_user_id"] = current_user_id
        if _call_accepts_keyword(
            state.runtime.run_post_turn_jobs,
            "defer_image_generation",
        ):
            kwargs["defer_image_generation"] = True
        result = await state.runtime.run_post_turn_jobs(**kwargs)
        image_job = await _queue_deferred_automatic_image_if_prepared(
            state,
            save_id=save_id,
            narrator_message_id=narrator_message_id,
            current_user_id=current_user_id,
        )
        if image_job is not None and optional_jobs is not None:
            optional_jobs.append(image_job)
        return result
    finally:
        progress_queue.put_nowait(done)
        await pump_task


async def _run_post_turn_jobs_inline_fallback(
    state: WebAppState,
    handle: JobHandle,
    *,
    save_id: str,
    player_message_id: str,
    narrator_message_id: str,
    turn_revision: object | None = None,
    prepared_action_choices: object | None = None,
    prior_phase_jobs: list[dict[str, str]] | None = None,
    current_user_id: str | None = None,
) -> Any:
    optional_jobs: list[JobRecord] = []
    await _queue_prepared_action_choices_if_available(
        state,
        handle,
        save_id=save_id,
        narrator_message_id=narrator_message_id,
        prepared_action_choices=prepared_action_choices,
        current_user_id=current_user_id,
    )
    result = await _run_post_turn_jobs_with_ordered_progress(
        state,
        handle,
        save_id=save_id,
        player_message_id=player_message_id,
        narrator_message_id=narrator_message_id,
        turn_revision=turn_revision,
        prior_phase_jobs=prior_phase_jobs,
        current_user_id=current_user_id,
        optional_jobs=optional_jobs,
    )
    if bool(getattr(result, "continuity_degraded", False)):
        await handle.event("runtime", to_jsonable(result))
        raise RuntimeError("Post-turn continuity is degraded; retry pending")
    await handle.advance_completion_level(CONTINUITY_READY)

    async def finish_optional_jobs() -> None:
        for optional_job in optional_jobs:
            if optional_job.task is not None:
                await asyncio.shield(optional_job.task)
        await handle.advance_completion_level(OPTIONAL_ENRICHMENTS_COMPLETE)

    if optional_jobs:
        finalizer = asyncio.create_task(finish_optional_jobs())
        finalizer.add_done_callback(
            lambda task: None if task.cancelled() else task.exception()
        )
    else:
        await handle.advance_completion_level(OPTIONAL_ENRICHMENTS_COMPLETE)
    return result


async def _queue_prepared_action_choices_if_available(
    state: WebAppState,
    handle: JobHandle,
    *,
    save_id: str,
    narrator_message_id: str,
    prepared_action_choices: object | None,
    current_user_id: str | None = None,
) -> JobRecord | None:
    if prepared_action_choices is None:
        return None
    run_prepared = getattr(state.runtime, "run_prepared_action_choices", None)
    if not callable(run_prepared):
        return None

    async def worker(action_choice_handle: JobHandle) -> Any:
        del action_choice_handle
        kwargs: dict[str, object] = {
            "prepared_action_choices": prepared_action_choices,
        }
        if _call_accepts_keyword(run_prepared, "current_user_id"):
            kwargs["current_user_id"] = current_user_id
        return await run_prepared(**kwargs)

    try:
        record = await state.jobs.create(
            "action_choice_generate",
            worker,
            save_id=save_id,
            creator_user_id=current_user_id,
            operation_queue_key=_action_choice_operation_queue_key(
                save_id,
                narrator_message_id,
            ),
        )
    except Exception as exc:
        observe(
            "web.optional_enrichment_queue_failed",
            level="error",
            save_id=save_id,
            narrator_message_id=narrator_message_id,
            enrichment="action_choices",
            **error_fields(exc),
        )
        await handle.event(
            "optional_enrichment",
            {"name": "action_choices", "status": "queue_failed"},
        )
        return None
    await handle.event(
        "optional_enrichment",
        {"name": "action_choices", "status": "queued", "job_id": record.id},
    )
    return record


async def _queue_deferred_automatic_image_if_prepared(
    state: WebAppState,
    *,
    save_id: str,
    narrator_message_id: str,
    current_user_id: str | None = None,
) -> JobRecord | None:
    consume = getattr(
        state.runtime,
        "consume_deferred_automatic_image",
        None,
    )
    if not callable(consume):
        return None
    prepared_automatic_image = consume(
        save_id=save_id,
        narrator_message_id=narrator_message_id,
    )
    if prepared_automatic_image is None:
        return None
    run_deferred = getattr(
        state.runtime,
        "run_deferred_automatic_image",
        None,
    )
    if not callable(run_deferred):
        return None

    async def worker(post_turn_handle: JobHandle) -> Any:
        kwargs: dict[str, object] = {
            "save_id": save_id,
            "prepared_automatic_image": prepared_automatic_image,
        }
        if _call_accepts_keyword(run_deferred, "current_user_id"):
            kwargs["current_user_id"] = current_user_id
        return await run_deferred(**kwargs)

    try:
        return await state.jobs.create(
            "automatic_image_generation",
            worker,
            save_id=save_id,
            creator_user_id=current_user_id,
            operation_queue_key=f"automatic_image:{save_id}",
        )
    except Exception as exc:
        observe(
            "web.optional_enrichment_queue_failed",
            level="error",
            save_id=save_id,
            narrator_message_id=narrator_message_id,
            enrichment="image",
            **error_fields(exc),
        )
        return None


def _initial_chat_turn_progress(status_text: str) -> dict[str, object]:
    return {
        "status_text": status_text,
        "jobs": [
            {
                "name": name,
                "status": "running" if name == "submission" else "pending",
            }
            for name in _CHAT_TURN_PROGRESS_JOB_ORDER
        ],
    }


def _turn_progress_callback(
    handle: JobHandle,
    initial_progress: dict[str, object],
) -> tuple[Any, Any, Callable[[], list[dict[str, str]]]]:
    loop = asyncio.get_running_loop()
    tasks: list[asyncio.Task[None]] = []
    latest_jobs = _progress_jobs(initial_progress)

    def callback(progress: object) -> None:
        nonlocal latest_jobs
        payload = _progress_payload(progress)
        jobs = _progress_jobs(payload)
        if jobs:
            latest_jobs = jobs

        def schedule() -> None:
            tasks.append(asyncio.create_task(handle.event("progress", payload)))

        loop.call_soon_threadsafe(schedule)

    async def flush() -> None:
        await asyncio.sleep(0)
        if tasks:
            await asyncio.gather(*tasks)

    def latest() -> list[dict[str, str]]:
        return [dict(job) for job in latest_jobs]

    return callback, flush, latest


def _progress_payload(progress: object) -> dict[str, object]:
    payload = to_jsonable(progress)
    if isinstance(payload, dict):
        return payload
    return {"status_text": str(payload)}


def _progress_jobs(progress: object) -> list[dict[str, str]]:
    payload = progress if isinstance(progress, dict) else _progress_payload(progress)
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        return []
    parsed: list[dict[str, str]] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        name = job.get("name")
        status = job.get("status")
        if isinstance(name, str) and isinstance(status, str):
            parsed_job = {"name": name, "status": status}
            category = job.get("category")
            if isinstance(category, str):
                parsed_job["category"] = category
            parsed.append(parsed_job)
    return parsed


def _completed_chat_turn_phase_jobs(
    jobs: list[dict[str, str]],
) -> list[dict[str, str]]:
    statuses = {
        job["name"]: job["status"]
        for job in jobs
        if job.get("name") in _CHAT_TURN_PROGRESS_JOB_ORDER
    }
    completed: list[dict[str, str]] = []
    for name in _CHAT_TURN_PROGRESS_JOB_ORDER:
        status = statuses.get(name, "pending")
        if status in {"pending", "running"}:
            status = "succeeded"
        completed.append({"name": name, "status": status})
    return completed


def _initial_post_turn_progress(
    prior_phase_jobs: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    return {
        "status_text": "Updating world state",
        "jobs": (prior_phase_jobs or [])
        + [
            {
                "name": name,
                "status": "pending",
                "category": "optional" if name == "image" else "continuity",
            }
            for name in _POST_TURN_PROGRESS_JOB_ORDER
        ],
    }


def _post_turn_progress_with_prior_jobs(
    progress: object,
    prior_phase_jobs: list[dict[str, str]] | None,
) -> dict[str, object]:
    payload = _progress_payload(progress)
    jobs = _progress_jobs(payload)
    if not jobs:
        return payload
    return {
        **payload,
        "status_text": str(payload.get("status_text") or "Updating world state"),
        "jobs": [*(prior_phase_jobs or []), *jobs],
    }


def _retry_progress_callback(
    handle: JobHandle,
    *,
    task_label: str,
) -> tuple[Any, Any]:
    loop = asyncio.get_running_loop()
    tasks: list[asyncio.Task[None]] = []

    def callback(progress: object) -> None:
        payload = {"label": _provider_retry_status_text(progress, task_label)}

        def schedule() -> None:
            tasks.append(asyncio.create_task(handle.event("progress", payload)))

        loop.call_soon_threadsafe(schedule)

    async def flush() -> None:
        await asyncio.sleep(0)
        if tasks:
            await asyncio.gather(*tasks)

    return callback, flush


def _provider_retry_status_text(progress: object, task_label: str) -> str:
    next_attempt = getattr(progress, "next_attempt", None)
    max_attempts = getattr(progress, "max_attempts", None)
    if next_attempt is None or max_attempts is None:
        return f"Retrying {task_label} request..."
    return (
        f"Retrying {task_label} request "
        f"(attempt {next_attempt} of {max_attempts})..."
    )


def _call_accepts_keyword(callable_obj: object, keyword: str) -> bool:
    try:
        callable_value = cast(Callable[..., Any], callable_obj)
        parameters = inspect.signature(callable_value).parameters
    except (TypeError, ValueError):
        return False
    return keyword in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _runtime_model_error(model: object) -> str | None:
    attr_error = getattr(model, "error", None)
    if isinstance(attr_error, str) and attr_error:
        return attr_error
    payload = to_jsonable(model)
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if isinstance(error, str) and error:
        return error
    return None


def _is_runtime_chat_cancelled(model: object) -> bool:
    payload = to_jsonable(model)
    if not isinstance(payload, dict):
        return False
    return (
        payload.get("error") == _CHAT_TURN_CANCELLED_ERROR
        or payload.get("failure_text") == _CHAT_TURN_CANCELLED_ERROR
    )


def _raise_for_runtime_chat_failure(
    model: object,
    *,
    failure_message: str,
) -> None:
    payload = to_jsonable(model)
    if not isinstance(payload, dict):
        return
    explicit_error = payload.get("error") or payload.get("failure_text")
    if explicit_error:
        raise RuntimeError(str(explicit_error))
    if payload.get("failed_save"):
        raise RuntimeError(failure_message)
    chronicle = payload.get("chronicle")
    if not isinstance(chronicle, dict):
        return
    messages = chronicle.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RuntimeError(failure_message)
    latest = messages[-1]
    if isinstance(latest, dict) and latest.get("role") == "player":
        raise RuntimeError(failure_message)


def _initial_chat_turn_result(turn: object) -> object:
    delta = getattr(turn, "delta", None)
    if delta is not None:
        return delta
    model = getattr(turn, "model", None)
    if model is not None:
        return model
    return {
        "kind": "chat_turn_delta",
        "version": 1,
        "requires_full_refresh": True,
    }


async def _emit_initial_chat_turn_event(handle: JobHandle, turn: object) -> None:
    delta = getattr(turn, "delta", None)
    if delta is not None:
        await handle.event("chat_turn_delta", delta)
        return
    model = getattr(turn, "model", None)
    if model is not None:
        await handle.event("runtime", model)


def _raise_for_initial_chat_turn_failure(
    turn: object,
    *,
    failure_message: str,
) -> None:
    model = getattr(turn, "model", None)
    if model is not None:
        if _is_runtime_chat_cancelled(model):
            raise asyncio.CancelledError(_CHAT_TURN_CANCELLED_ERROR)
        _raise_for_runtime_chat_failure(model, failure_message=failure_message)
        return
    delta = getattr(turn, "delta", None)
    if delta is not None:
        payload = to_jsonable(delta)
        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(str(payload["error"]))
        return
    raise RuntimeError(failure_message)


async def _review_scenario_edit_for_request(
    state: WebAppState,
    edit: ScenarioEdit,
    *,
    save_id: str | None,
    roleplay_type: str,
    current_user_id: str | None,
) -> ScenarioEdit:
    from bragi.content_rating_instructions import maximum_content_rating
    from bragi.services.character_profile_completion import (
        ScenarioCharacterStarter,
    )
    from bragi.services.content_rating import effective_content_safety_policy
    from bragi.services.content_safety_service import (
        ContentSafetyAction,
        ContentSafetyService,
    )

    policy = effective_content_safety_policy(
        state.repositories,
        user_id=current_user_id,
    )
    service = ContentSafetyService(
        repositories=state.repositories,
        providers=state.providers,
    )
    ratings: dict[str, str] = {}

    async def review(section_id: str, body: str) -> str:
        if not body.strip():
            ratings[section_id] = (
                "unrated" if policy.rating == "unrated" else "g"
            )
            return body
        safety = await service.review_narration(
            body=body,
            content_rating=policy.rating,
            fade_to_black_enabled=policy.fade_to_black_enabled,
            save_id=save_id,
            roleplay_type=roleplay_type,
        )
        ratings[section_id] = safety.reviewed_content_rating
        return safety.body

    content = dict(edit.content)
    for section_id, value in tuple(content.items()):
        if isinstance(value, str):
            content[section_id] = await review(section_id, value)
    reviewed_starters: list[ScenarioCharacterStarter] = []
    for starter in edit.character_starters:
        semantic_content = {
            field.name: getattr(starter, field.name)
            for field in fields(starter)
            if field.name
            not in {
                "starter_id",
                "locked_fields",
                "evidence_source_ids",
                "reference_image",
                "met",
            }
        }
        if starter.reference_image is not None:
            semantic_content["reference_image_prompt"] = (
                starter.reference_image.prompt_preview
            )
        safety = await service.review_narration(
            body=json.dumps(
                semantic_content,
                ensure_ascii=False,
                sort_keys=True,
            ),
            content_rating=policy.rating,
            fade_to_black_enabled=policy.fade_to_black_enabled,
            save_id=save_id,
            roleplay_type=roleplay_type,
        )
        ratings["character_starters"] = maximum_content_rating(
            (
                ratings.get("character_starters", "g"),
                safety.reviewed_content_rating,
            )
        )
        reviewed_starters.append(
            starter
            if safety.action is ContentSafetyAction.ALLOW
            else _scenario_starter_with_safe_transition(
                starter,
                safety.body,
            )
        )
    return replace(
        edit,
        title=await review("title", edit.title),
        premise=await review("premise", edit.premise),
        player_character_name=await review(
            "player_character_name",
            edit.player_character_name,
        ),
        player_role=await review("player_role", edit.player_role),
        content=content,
        character_starters=tuple(reviewed_starters),
        section_content_ratings=tuple(sorted(ratings.items())),
    )


async def _review_character_edits_for_request(
    state: WebAppState,
    edits: CharacterRegistryEdits,
    *,
    save_id: str,
    current_user_id: str | None,
) -> CharacterRegistryEdits:
    from bragi.services.content_rating import effective_content_safety_policy
    from bragi.services.content_safety_service import (
        ContentSafetyAction,
        ContentSafetyService,
    )

    policy = effective_content_safety_policy(
        state.repositories,
        user_id=current_user_id,
    )
    service = ContentSafetyService(
        repositories=state.repositories,
        providers=state.providers,
    )
    reviewed_rows: list[CharacterRegistryRow] = []
    for row in edits.characters:
        existing = (
            state.repositories.get_character(row.character_id)
            if row.character_id
            else None
        )
        if row.archived or row.merge_into_character_id is not None:
            reviewed_rows.append(
                replace(
                    row,
                    content_rating=(
                        existing.content_rating
                        if existing is not None
                        else "unclassified"
                    ),
                )
            )
            continue
        semantic_content = {
            key: getattr(row, key)
            for key in (
                "name",
                "aliases_text",
                "role",
                "age",
                "known_state",
                "history",
                "appearance",
                "visual_notes",
                "current_clothing",
                "personality",
                "voice",
                "texting_style",
                "relationships_json",
                "goals",
                "motivations",
                "current_intent",
                "boundaries",
                "attitude_toward_player",
                "cooperation_conditions",
                "status",
                "private_notes",
                "contact_name",
            )
        }
        safety = await service.review_narration(
            body=json.dumps(semantic_content, ensure_ascii=False, sort_keys=True),
            content_rating=policy.rating,
            fade_to_black_enabled=policy.fade_to_black_enabled,
            save_id=save_id,
        )
        reviewed_rows.append(
            replace(row, content_rating=safety.reviewed_content_rating)
            if safety.action is ContentSafetyAction.ALLOW
            else _character_row_with_safe_transition(
                row,
                replacement=safety.body,
                content_rating=safety.reviewed_content_rating,
            )
        )
    return CharacterRegistryEdits(characters=tuple(reviewed_rows))


def _character_row_with_safe_transition(
    row: CharacterRegistryRow,
    *,
    replacement: str,
    content_rating: str,
) -> CharacterRegistryRow:
    updates: dict[str, object] = {
        key: replacement if getattr(row, key) else ""
        for key in (
            "name",
            "aliases_text",
            "role",
            "age",
            "known_state",
            "history",
            "appearance",
            "visual_notes",
            "current_clothing",
            "personality",
            "voice",
            "texting_style",
            "goals",
            "motivations",
            "current_intent",
            "boundaries",
            "attitude_toward_player",
            "cooperation_conditions",
            "status",
            "private_notes",
            "contact_name",
        )
    }
    updates["relationships_json"] = "{}"
    updates["content_rating"] = content_rating
    return replace(row, **updates)  # type: ignore[arg-type]


def _scenario_starter_with_safe_transition(
    starter: Any,
    replacement: str,
) -> Any:
    string_fields = (
        "name",
        "role",
        "age",
        "known_state",
        "appearance",
        "visual_notes",
        "personality",
        "voice",
        "texting_style",
        "goals",
        "motivations",
        "current_intent",
        "boundaries",
        "attitude_toward_player",
        "cooperation_conditions",
        "status",
    )
    updates: dict[str, object] = {
        field_name: replacement if getattr(starter, field_name, "") else ""
        for field_name in string_fields
    }
    updates["aliases"] = ()
    updates["relationships"] = {}
    return replace(starter, **updates)


def _world_data_edits_from_json(payload: dict[str, Any]) -> WorldDataEdits:
    return WorldDataEdits(
        scenario=_scenario_edit_from_json(payload["scenario"])
        if payload.get("scenario") is not None
        else None,
        world_state=tuple(
            _dataclass_from_json(WorldDataStateRow, row)
            for row in payload.get("world_state", ())
        ),
        memories=tuple(
            _memory_edit_from_json(row) for row in payload.get("memories", ())
        ),
        summaries=tuple(
            _summary_edit_from_json(row) for row in payload.get("summaries", ())
        ),
        scene=(
            _dataclass_from_json(WorldDataSceneRow, payload["scene"])
            if payload.get("scene") is not None
            else None
        ),
        locations=tuple(
            _dataclass_from_json(WorldDataLocationRow, row)
            for row in payload.get("locations", ())
        ),
        characters=tuple(
            _dataclass_from_json(WorldDataCharacterRow, row)
            for row in payload.get("characters", ())
        ),
        threads=tuple(
            _dataclass_from_json(WorldDataThreadRow, row)
            for row in payload.get("threads", ())
        ),
        links=tuple(
            _dataclass_from_json(WorldDataEntityLinkRow, row)
            for row in payload.get("links", ())
        ),
        suggestions=tuple(
            _dataclass_from_json(WorldDataSuggestionRow, row)
            for row in payload.get("suggestions", ())
        ),
        suggestion_groups=tuple(
            _dataclass_from_json(WorldDataSuggestionGroupRow, row)
            for row in payload.get("suggestion_groups", ())
        ),
        loss_conditions=tuple(
            _dataclass_from_json(WorldDataLossConditionRow, row)
            for row in payload.get("loss_conditions", ())
        ),
    )


def _scenario_edit_from_json(payload: object) -> ScenarioEdit:
    from bragi.services.character_profile_completion import (
        CHARACTER_STARTERS_CONTENT_KEY,
        normalize_scenario_character_starters,
    )

    payload = _json_object(payload, "ScenarioEdit")
    content = payload.get("content")
    if content is None:
        content = _scenario_content_sections(payload.get("content_sections", []))
    else:
        content = _strict_json_value(
            dict[str, object],
            content,
            "ScenarioEdit.content",
        )
    starter_payload = payload.get("character_starters")
    if starter_payload is None and isinstance(content, dict):
        starter_payload = content.pop(CHARACTER_STARTERS_CONTENT_KEY, None)
    character_starters = normalize_scenario_character_starters(
        [] if starter_payload is None else starter_payload,
        strict=True,
    )
    return ScenarioEdit(
        title=_string_field(payload, "title", "ScenarioEdit"),
        premise=_string_field(payload, "premise", "ScenarioEdit"),
        player_character_name=_string_field(
            payload,
            "player_character_name",
            "ScenarioEdit",
        ),
        player_role=_string_field(payload, "player_role", "ScenarioEdit"),
        content=cast(dict[str, object], content),
        character_starters=character_starters,
        interaction_mode=cast(str | None, payload.get("interaction_mode")),
    )


def _memory_edit_from_json(payload: object) -> WorldDataMemoryRow | MemoryEdit:
    payload = _json_object(payload, "MemoryEdit")
    if "tags_text" in payload:
        return cast(
            WorldDataMemoryRow,
            _dataclass_from_json(WorldDataMemoryRow, payload),
        )
    return MemoryEdit(
        memory_id=_string_field(payload, "memory_id", "MemoryEdit"),
        body=_string_field(payload, "body", "MemoryEdit"),
        tags=cast(
            tuple[str, ...],
            _strict_json_value(
                tuple[str, ...],
                payload.get("tags", []),
                "MemoryEdit.tags",
            ),
        ),
        importance=_float_field(payload, "importance", "MemoryEdit"),
        archived=_bool_field(payload, "archived", "MemoryEdit"),
        source_message_id=_optional_string_field(
            payload,
            "source_message_id",
            "MemoryEdit",
        ),
    )


def _summary_edit_from_json(
    payload: object,
) -> WorldDataSummaryRow | SummaryEdit:
    payload = _json_object(payload, "SummaryEdit")
    if "provider" in payload:
        return cast(
            WorldDataSummaryRow,
            _dataclass_from_json(WorldDataSummaryRow, payload),
        )
    return SummaryEdit(
        summary_id=_string_field(payload, "summary_id", "SummaryEdit"),
        body=_string_field(payload, "body", "SummaryEdit"),
        archived=_bool_field(payload, "archived", "SummaryEdit"),
    )


def _dataclass_from_json(model_type: type[Any], payload: object) -> Any:
    model_type = cast(type[Any], resolve_lazy_symbol(model_type))
    if not isinstance(payload, dict):
        raise TypeError(f"{model_type.__name__} payload must be an object")
    kwargs: dict[str, Any] = {}
    type_hints = get_type_hints(model_type)
    for field in fields(model_type):
        if not field.init:
            continue
        if field.name not in payload:
            continue
        annotation = type_hints.get(field.name, field.type)
        kwargs[field.name] = _strict_json_value(
            annotation,
            payload[field.name],
            f"{model_type.__name__}.{field.name}",
        )
    try:
        return model_type(**kwargs)
    except TypeError as exc:
        raise TypeError(f"Invalid {model_type.__name__}: {exc}") from exc


def _json_object(payload: object, model_name: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise TypeError(f"{model_name} payload must be an object")
    return payload


def _scenario_content_sections(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return cast(
            dict[str, object],
            _strict_json_value(
                dict[str, object],
                value,
                "ScenarioEdit.content_sections",
            ),
        )
    if not isinstance(value, list):
        raise TypeError("ScenarioEdit.content_sections must be an array")
    content: dict[str, object] = {}
    for index, item in enumerate(value):
        if not isinstance(item, list) or len(item) != 2:
            raise TypeError(
                f"ScenarioEdit.content_sections[{index}] must be a key/value pair"
            )
        key = item[0]
        if not isinstance(key, str):
            raise TypeError(
                f"ScenarioEdit.content_sections[{index}][0] must be a string"
            )
        content[key] = item[1]
    return content


def _string_field(
    payload: dict[str, object],
    field_name: str,
    model_name: str,
) -> str:
    if field_name not in payload:
        return ""
    return cast(
        str,
        _strict_json_value(str, payload[field_name], f"{model_name}.{field_name}"),
    )


def _bool_field(
    payload: dict[str, object],
    field_name: str,
    model_name: str,
) -> bool:
    if field_name not in payload:
        return False
    return cast(
        bool,
        _strict_json_value(bool, payload[field_name], f"{model_name}.{field_name}"),
    )


def _float_field(
    payload: dict[str, object],
    field_name: str,
    model_name: str,
) -> float:
    if field_name not in payload:
        return 0.0
    return cast(
        float,
        _strict_json_value(float, payload[field_name], f"{model_name}.{field_name}"),
    )


def _optional_string_field(
    payload: dict[str, object],
    field_name: str,
    model_name: str,
) -> str | None:
    if field_name not in payload:
        return None
    return cast(
        str | None,
        _strict_json_value(
            str | None,
            payload[field_name],
            f"{model_name}.{field_name}",
        ),
    )


def _strict_json_value(annotation: object, value: object, path: str) -> object:
    if annotation is Any or annotation is object:
        return value
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is tuple:
        if not isinstance(value, list):
            raise TypeError(f"{path} must be an array")
        item_type = args[0] if args and args[0] is not Ellipsis else object
        return tuple(
            _strict_json_value(item_type, item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError(f"{path} must be an object")
        key_type = args[0] if args else object
        value_type = args[1] if len(args) > 1 else object
        if key_type is str and any(not isinstance(key, str) for key in value):
            raise TypeError(f"{path} keys must be strings")
        if value_type is Any or value_type is object:
            return value
        return {
            key: _strict_json_value(value_type, item, f"{path}.{key}")
            for key, item in value.items()
        }
    if origin in {UnionType, Union}:
        non_none = [arg for arg in args if arg is not type(None)]
        if value is None:
            if len(non_none) < len(args):
                return None
            raise TypeError(f"{path} must not be null")
        if len(non_none) == 1 and len(non_none) < len(args):
            return _strict_json_value(non_none[0], value, path)
        for option in non_none:
            try:
                return _strict_json_value(option, value, path)
            except TypeError:
                continue
        raise TypeError(f"{path} does not match any allowed type")
    if annotation is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{path} must be a boolean")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{path} must be an integer")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"{path} must be a number")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise TypeError(f"{path} must be a string")
        return value
    if isinstance(annotation, type) and is_dataclass(annotation):
        return _dataclass_from_json(annotation, value)
    return value


def _route_path(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str):
        return path
    return request.url.path


def _reject_untrusted_host(
    request: Request,
    host_security: HostSecurityConfig,
) -> bool:
    try:
        host = _host_from_header(request.headers.get("host"))
    except ValueError:
        return True
    return not _is_allowed_host(host, host_security)


def _reject_cross_origin_write(
    request: Request,
    host_security: HostSecurityConfig,
) -> bool:
    if request.method.upper() not in _UNSAFE_METHODS:
        return False
    if not request.url.path.startswith("/api/"):
        return False

    origin = request.headers.get("origin")
    if origin is not None:
        return not _is_allowed_origin(
            origin,
            host_security,
            request_host=_request_host(request),
        )

    referer = request.headers.get("referer")
    if referer is not None:
        return not _is_allowed_origin(
            referer,
            host_security,
            request_host=_request_host(request),
        )

    return request.headers.get(_BRAGI_API_REQUEST_HEADER) != "1"


def _is_allowed_origin(
    value: str,
    host_security: HostSecurityConfig,
    *,
    request_host: str | None = None,
) -> bool:
    try:
        parts = _origin_parts(value)
    except ValueError:
        return False
    if parts in host_security.allowed_origins:
        return True
    scheme, host, port = parts
    return (
        scheme == "http"
        and port in host_security.allowed_origin_ports
        and request_host is not None
        and host == request_host
        and _is_allowed_host(host, host_security)
    )


def _request_host(request: Request) -> str | None:
    try:
        return _host_from_header(request.headers.get("host"))
    except ValueError:
        return None


def _is_allowed_host(host: str, host_security: HostSecurityConfig) -> bool:
    if host in host_security.allowed_hosts:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (
        not address.is_unspecified
        and (address.is_loopback or address.is_private or address.is_link_local)
    )


def _host_security_config() -> HostSecurityConfig:
    allowed_hosts = {
        "localhost",
        "127.0.0.1",
        "::1",
        "testserver",
    }
    _add_configured_host(allowed_hosts, os.environ.get("BRAGI_WEB_HOST", ""))
    _add_configured_host(allowed_hosts, os.environ.get("BRAGI_WEB_FRONTEND_HOST", ""))
    for host in _machine_hosts():
        _add_configured_host(allowed_hosts, host)
    for host in _env_values("BRAGI_WEB_ALLOWED_HOSTS"):
        _add_configured_host(allowed_hosts, host)

    allowed_origin_ports = {
        _env_int("BRAGI_WEB_PORT", _DEFAULT_BACKEND_PORT),
        _env_int("BRAGI_WEB_FRONTEND_PORT", _DEFAULT_FRONTEND_PORT),
    }
    allowed_origins = {
        _origin_parts(origin)
        for origin in _env_values("BRAGI_WEB_ALLOWED_ORIGINS")
    }
    return HostSecurityConfig(
        allowed_hosts=frozenset(allowed_hosts),
        allowed_origin_ports=frozenset(allowed_origin_ports),
        allowed_origins=frozenset(allowed_origins),
    )


def _machine_hosts() -> tuple[str, ...]:
    hosts: list[str] = []
    for resolver in (socket.gethostname, socket.getfqdn):
        try:
            host = resolver()
        except OSError:
            continue
        if host:
            hosts.append(host)
    return tuple(hosts)


def _add_configured_host(allowed_hosts: set[str], value: str | None) -> None:
    if value is None:
        return
    try:
        host = _normalize_host(_host_from_header(value))
    except ValueError:
        return
    if host not in _WILDCARD_HOSTS:
        allowed_hosts.add(host)


def _env_values(name: str) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in os.environ.get(name, "").split(",")
        if item.strip()
    )


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _host_from_header(value: str | None) -> str:
    if value is None or not value.strip():
        raise ValueError("Invalid host")
    parsed = urlsplit(f"//{value.strip()}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Invalid host")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("Invalid host") from exc
    hostname = parsed.hostname
    if hostname is None:
        raise ValueError("Invalid host")
    return _normalize_host(hostname)


def _normalize_host(hostname: str) -> str:
    host = hostname.strip().lower().rstrip(".")
    if not host:
        raise ValueError("Invalid host")
    return host


def _origin_parts(value: str) -> tuple[str, str, int]:
    parsed = urlsplit(value)
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname
    if (
        scheme not in {"http", "https"}
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("Invalid origin")
    return (scheme, _normalize_host(hostname), _origin_port(parsed))


def _origin_port(parsed: SplitResult) -> int:
    explicit_port = parsed.port
    if explicit_port is not None:
        return explicit_port
    if parsed.scheme.lower() == "https":
        return 443
    return 80


async def _event_stream(
    state: WebAppState,
    job_id: str,
    last_event_id: int = 0,
    *,
    current_user: UserRecord | None | object = _CURRENT_USER_SENTINEL,
    current_user_role: str | None = None,
) -> AsyncIterator[str]:
    index = last_event_id
    emitted = 0
    while True:
        record = state.jobs.get(job_id)
        if record is None:
            observe(
                "web.sse.disconnected",
                level="error",
                job_id=job_id,
                emitted_count=emitted,
            )
            break
        if index < record.event_offset:
            index = record.event_offset
        event_end = record.event_offset + len(record.events)
        while index < event_end:
            event = record.events[index - record.event_offset]
            index += 1
            emitted += 1
            with _repository_scope_for_state(state):
                payload = _job_event_payload_for_request(
                    state,
                    record,
                    event,
                    current_user=current_user,
                    current_user_role=current_user_role,
                )
            yield (
                f"id: {index}\n"
                f"event: {event['event']}\n"
                f"data: {json.dumps(payload)}\n\n"
            )
        if record.status in {"succeeded", "failed", "cancelled"}:
            with _repository_scope_for_state(state):
                summary = _job_summary_for_request(
                    state,
                    record,
                    current_user=current_user,
                    current_user_role=current_user_role,
                )
            yield f"event: done\ndata: {json.dumps(summary)}\n\n"
            observe(
                "web.sse.done",
                job_id=job_id,
                job_type=record.type,
                status=record.status,
                emitted_count=emitted,
            )
            break
        try:
            index = await asyncio.wait_for(
                state.jobs.wait_for_event(job_id, index),
                timeout=_SSE_HEARTBEAT_SECONDS,
            )
            current = state.jobs.get(job_id)
            if current is not None:
                event_end = current.event_offset + len(current.events)
                if index >= event_end and current.status not in {
                    "succeeded",
                    "failed",
                    "cancelled",
                }:
                    yield "event: heartbeat\ndata: {}\n\n"
        except TimeoutError:
            yield "event: heartbeat\ndata: {}\n\n"
        except asyncio.CancelledError:
            record = state.jobs.get(job_id)
            observe(
                "web.sse.disconnected",
                job_id=job_id,
                job_type=record.type if record is not None else None,
                emitted_count=emitted,
            )
            raise


def _save_event_cursor_from_header(
    value: str | None,
    *,
    latest_event_id: int,
) -> int:
    if value is None or not value.strip():
        return 0
    try:
        event_id = int(value)
    except ValueError:
        return 0
    if event_id < 0 or event_id > latest_event_id:
        return 0
    return event_id


async def _save_event_stream(
    state: WebAppState,
    save_id: str,
    last_event_id: int = 0,
    *,
    current_user: UserRecord | None = None,
    current_user_role: str | None = None,
    owner_user_id: str | None = None,
    include_unowned_global: bool = True,
    include_all_global: bool = False,
) -> AsyncIterator[str]:
    index = last_event_id
    emitted = 0
    while True:
        events = state.save_events.events_after(
            save_id,
            index,
            owner_user_id=owner_user_id,
            include_unowned_global=include_unowned_global,
            include_all_global=include_all_global,
        )
        for event in events:
            index = max(index, event.event_id)
            emitted += 1
            with _repository_scope_for_state(state):
                visible_user = current_user
                get_user = getattr(state.repositories, "get_user", None)
                if visible_user is not None and callable(get_user):
                    visible_user = get_user(visible_user.id) or visible_user
                payload = {
                    "event_id": event.event_id,
                    "save_id": event.save_id,
                    "type": event.event_type,
                    "payload": to_jsonable(event.payload),
                }
                _scrub_response_payload_for_request(
                    state,
                    payload,
                    current_user=visible_user,
                    current_user_role=(
                        visible_user.role
                        if visible_user is not None
                        else current_user_role
                    ),
                )
            yield (
                f"id: {event.event_id}\n"
                f"event: {event.event_type}\n"
                f"data: {json.dumps(payload)}\n\n"
            )
        try:
            index = await asyncio.wait_for(
                state.save_events.wait_for_event(
                    save_id,
                    index,
                    owner_user_id=owner_user_id,
                    include_unowned_global=include_unowned_global,
                    include_all_global=include_all_global,
                ),
                timeout=_SSE_HEARTBEAT_SECONDS,
            )
            if not state.save_events.events_after(
                save_id,
                index,
                owner_user_id=owner_user_id,
                include_unowned_global=include_unowned_global,
                include_all_global=include_all_global,
            ):
                yield "event: heartbeat\ndata: {}\n\n"
        except TimeoutError:
            yield "event: heartbeat\ndata: {}\n\n"
        except asyncio.CancelledError:
            observe(
                "web.save_sse.disconnected",
                save_id=save_id,
                emitted_count=emitted,
            )
            raise


def _save_event_stream_auth_filter(
    state: WebAppState,
) -> _SaveEventStreamAuthFilter:
    if not _auth_context_enabled(state):
        return {}
    user = _current_request_user()
    if user is None:
        return {"include_unowned_global": False}
    return {
        "owner_user_id": user.id,
        "include_unowned_global": False,
        "include_all_global": user.role == "admin",
    }


async def _store_upload(file: UploadFile, temp_dir: Path) -> Path:
    temp_dir.mkdir(parents=True, exist_ok=True)
    suffix = _upload_suffix(file.filename)
    bundle_path: Path | None = None
    try:
        with NamedTemporaryFile(
            prefix="bragi-web-upload-",
            suffix=suffix,
            dir=temp_dir,
            delete=False,
        ) as handle:
            bundle_path = Path(handle.name)
            written = 0
            while True:
                chunk = await file.read(BUNDLE_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if written > BUNDLE_UPLOAD_MAX_BYTES:
                    raise _BundleUploadTooLarge(BUNDLE_UPLOAD_MAX_BYTES)
                handle.write(chunk)
            return bundle_path
    except Exception:
        if bundle_path is not None:
            bundle_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


async def _read_limited_character_reference_upload(file: UploadFile) -> bytes:
    chunks: list[bytes] = []
    written = 0
    try:
        while True:
            chunk = await file.read(BUNDLE_UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            written += len(chunk)
            if written > CHARACTER_REFERENCE_UPLOAD_MAX_BYTES:
                raise _CharacterReferenceUploadTooLarge(
                    CHARACTER_REFERENCE_UPLOAD_MAX_BYTES
                )
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        await file.close()


def _upload_suffix(filename: str | None) -> str:
    if filename and filename.endswith(".bragi-chat"):
        return ".bragi-chat"
    if filename and filename.endswith(".bragi-scenario"):
        return ".bragi-scenario"
    if filename and filename.endswith(".bragi-character"):
        return ".bragi-character"
    return ".upload"


def _prune_bundle_previews(state: WebAppState) -> None:
    now = time()
    for preview_id, preview in list(state.bundle_previews.items()):
        if now - preview.created_at > BUNDLE_PREVIEW_TTL_SECONDS:
            _discard_bundle_preview(state, preview_id, preview)
    excess_count = len(state.bundle_previews) - BUNDLE_PREVIEW_MAX_COUNT
    if excess_count > 0:
        oldest = sorted(
            state.bundle_previews.items(),
            key=lambda item: item[1].created_at,
        )
        for preview_id, preview in oldest[:excess_count]:
            _discard_bundle_preview(state, preview_id, preview)
    _prune_retained_bundle_preview_bytes(state)


def _prune_scenario_bundle_previews(state: WebAppState) -> None:
    now = time()
    for preview_id, preview in list(state.scenario_bundle_previews.items()):
        if now - preview.created_at > SCENARIO_BUNDLE_PREVIEW_TTL_SECONDS:
            _discard_scenario_bundle_preview(state, preview_id, preview)
    excess_count = (
        len(state.scenario_bundle_previews) - SCENARIO_BUNDLE_PREVIEW_MAX_COUNT
    )
    if excess_count > 0:
        oldest = sorted(
            state.scenario_bundle_previews.items(),
            key=lambda item: item[1].created_at,
        )
        for preview_id, preview in oldest[:excess_count]:
            _discard_scenario_bundle_preview(state, preview_id, preview)
    _prune_retained_bundle_preview_bytes(state)


def _prune_character_bundle_previews(state: WebAppState) -> None:
    now = time()
    for preview_id, preview in list(state.character_bundle_previews.items()):
        if now - preview.created_at > CHARACTER_BUNDLE_PREVIEW_TTL_SECONDS:
            _discard_character_bundle_preview(state, preview_id, preview)
    excess_count = (
        len(state.character_bundle_previews) - CHARACTER_BUNDLE_PREVIEW_MAX_COUNT
    )
    if excess_count > 0:
        oldest = sorted(
            state.character_bundle_previews.items(),
            key=lambda item: item[1].created_at,
        )
        for preview_id, preview in oldest[:excess_count]:
            _discard_character_bundle_preview(state, preview_id, preview)
    _prune_retained_bundle_preview_bytes(state)


def _prune_retained_bundle_preview_bytes(state: WebAppState) -> None:
    if BUNDLE_PREVIEW_MAX_RETAINED_BYTES <= 0:
        return
    previews = [
        ("chat", preview_id, preview)
        for preview_id, preview in state.bundle_previews.items()
    ] + [
        ("scenario", preview_id, preview)
        for preview_id, preview in state.scenario_bundle_previews.items()
    ] + [
        ("character", preview_id, preview)
        for preview_id, preview in state.character_bundle_previews.items()
    ]
    total = sum(_bundle_preview_byte_count(preview) for _kind, _id, preview in previews)
    remaining = len(previews)
    for kind, preview_id, preview in sorted(
        previews,
        key=lambda item: item[2].created_at,
    ):
        if total <= BUNDLE_PREVIEW_MAX_RETAINED_BYTES or remaining <= 1:
            break
        total -= _bundle_preview_byte_count(preview)
        remaining -= 1
        if kind == "chat":
            _discard_bundle_preview(state, preview_id, preview)
        elif kind == "scenario":
            _discard_scenario_bundle_preview(state, preview_id, preview)
        else:
            _discard_character_bundle_preview(state, preview_id, preview)


def _bundle_preview_byte_count(preview: BundlePreviewState) -> int:
    try:
        return preview.bundle_path.stat().st_size
    except OSError:
        return 0


def _discard_bundle_preview(
    state: WebAppState,
    preview_id: str,
    preview: BundlePreviewState,
) -> None:
    state.bundle_previews.pop(preview_id, None)
    preview.bundle_path.unlink(missing_ok=True)


def _discard_scenario_bundle_preview(
    state: WebAppState,
    preview_id: str,
    preview: BundlePreviewState,
) -> None:
    state.scenario_bundle_previews.pop(preview_id, None)
    preview.bundle_path.unlink(missing_ok=True)


def _discard_character_bundle_preview(
    state: WebAppState,
    preview_id: str,
    preview: BundlePreviewState,
) -> None:
    state.character_bundle_previews.pop(preview_id, None)
    preview.bundle_path.unlink(missing_ok=True)


def _unlink_file(path: Path) -> None:
    path.unlink(missing_ok=True)


def _find_media_asset(
    state: WebAppState,
    asset_id: str,
    *,
    save_id: str | None = None,
) -> Any | None:
    with state.lock:
        return _find_media_asset_unlocked(state, asset_id, save_id=save_id)


def _find_media_asset_unlocked(
    state: WebAppState,
    asset_id: str,
    *,
    save_id: str | None = None,
) -> Any | None:
    save_ids = (
        [save_id]
        if save_id is not None
        else [save.id for save in _save_list_for_request(state)]
    )
    get_media_asset = getattr(state.repositories, "get_media_asset", None)
    if callable(get_media_asset):
        for candidate_save_id in save_ids:
            asset = get_media_asset(
                save_id=candidate_save_id,
                media_asset_id=asset_id,
            )
            if asset is not None:
                return asset
        return None
    for candidate_save_id in save_ids:
        for asset in state.repositories.list_media_assets(candidate_save_id):
            if asset.id == asset_id:
                return asset
    return None


def _raise_if_media_asset_exceeds_request_rating(
    state: WebAppState,
    asset: object,
    *,
    save_id: str,
) -> None:
    if _media_asset_exceeds_rating_for_request(
        state,
        asset,
        save_id=save_id,
        allowed_rating=_content_safety_policy_for_request(state).rating,
    ):
        raise HTTPException(
            status_code=403,
            detail="Media exceeds your content rating",
        )


def _media_asset_exceeds_rating_for_request(
    state: WebAppState,
    asset: object,
    *,
    save_id: str,
    allowed_rating: str,
) -> bool:
    from bragi.services.media_content_rating import media_asset_exceeds_rating

    list_media_assets = getattr(state.repositories, "list_media_assets", None)
    media_assets = (
        list(list_media_assets(save_id))
        if callable(list_media_assets)
        else [asset]
    )
    asset_id = getattr(asset, "id", None)
    if isinstance(asset_id, str) and all(
        getattr(candidate, "id", None) != asset_id
        for candidate in media_assets
    ):
        media_assets.append(asset)
    list_messages = getattr(state.repositories, "list_messages", None)
    messages = list(list_messages(save_id)) if callable(list_messages) else []
    list_character_text_messages = getattr(
        state.repositories,
        "list_character_text_messages",
        None,
    )
    character_text_messages = (
        list(list_character_text_messages(save_id=save_id))
        if callable(list_character_text_messages)
        else []
    )
    list_characters = getattr(state.repositories, "list_characters", None)
    characters = (
        list(list_characters(save_id))
        if callable(list_characters)
        else []
    )
    return media_asset_exceeds_rating(
        asset,
        allowed_rating=allowed_rating,
        media_assets_by_id={
            str(candidate_id): candidate
            for candidate in media_assets
            if isinstance(candidate_id := getattr(candidate, "id", None), str)
        },
        source_messages={
            str(message_id): message
            for message in messages
            if isinstance(message_id := getattr(message, "id", None), str)
        },
        character_text_messages={
            str(message_id): message
            for message in character_text_messages
            if isinstance(message_id := getattr(message, "id", None), str)
        },
        characters={
            str(character_id): character
            for character in characters
            if isinstance(character_id := getattr(character, "id", None), str)
        },
    )


def _media_asset_is_character_reference_unlocked(
    state: WebAppState,
    *,
    save_id: str,
    media_asset_id: str,
) -> bool:
    return any(
        link.target_type == "media_asset"
        and link.target_id == media_asset_id
        and link.relation == "reference_image"
        for link in state.repositories.list_entity_links(save_id)
    )


def _media_path_within_root(state: WebAppState, relative_path: str) -> Path | None:
    media_root = state.paths.media_dir.resolve()
    path = (state.paths.media_dir / relative_path).resolve()
    return path if path.is_relative_to(media_root) else None


def _scenario_starter_reference_image_unlocked(
    state: WebAppState,
    *,
    scenario_id: str,
    image_id: str,
    thumbnail: bool,
) -> tuple[Path, str] | None:
    scenario = state.repositories.get_scenario(scenario_id)
    if scenario is None:
        return None
    content_json = getattr(scenario, "content_json", "")
    if not isinstance(content_json, str):
        return None
    try:
        content = json.loads(content_json)
    except ValueError:
        return None
    if not isinstance(content, dict):
        return None
    starters = content.get("character_starters")
    if not isinstance(starters, list):
        return None
    for starter in starters:
        if not isinstance(starter, dict):
            continue
        reference = starter.get("reference_image")
        if not isinstance(reference, dict) or reference.get("id") != image_id:
            continue
        from bragi.content_rating_instructions import content_rating_exceeds

        if content_rating_exceeds(
            minimum_rating=str(
                reference.get("content_rating", "unclassified")
            ),
            allowed_rating=_content_safety_policy_for_request(state).rating,
        ):
            raise HTTPException(
                status_code=403,
                detail="Starter reference image exceeds your content rating",
            )
        mime_type = reference.get("mime_type")
        relative_path = reference.get("path")
        using_thumbnail = False
        if thumbnail:
            thumbnail_path = reference.get("thumbnail_path")
            if isinstance(thumbnail_path, str) and thumbnail_path:
                relative_path = thumbnail_path
                mime_type = "image/png"
                using_thumbnail = True
        if not isinstance(relative_path, str) or not relative_path:
            return None
        try:
            from bragi.services.media_service import (
                _assert_scenario_starter_reference_path,
            )

            _assert_scenario_starter_reference_path(relative_path)
        except ValueError:
            return None
        path = _media_path_within_root(state, relative_path)
        if path is None or not path.is_file():
            return None
        media_type = (
            "image/png"
            if using_thumbnail
            else safe_served_media_mime_type(
                mime_type if isinstance(mime_type, str) else "image/png"
            )
        )
        return path, media_type
    return None


def _media_file_response(path: Path, *, media_type: str) -> FileResponse:
    return FileResponse(
        path,
        media_type=media_type,
        headers={
            "Cache-Control": _MEDIA_CACHE_CONTROL,
            "X-Content-Type-Options": "nosniff",
        },
    )


def _is_unusable_fallback_thumbnail(path: Path) -> bool:
    try:
        return (
            path.stat().st_size == len(_UNUSABLE_FALLBACK_THUMBNAIL_PNG)
            and path.read_bytes() == _UNUSABLE_FALLBACK_THUMBNAIL_PNG
        )
    except OSError:
        return False


def _is_image_media_asset(asset: Any) -> bool:
    mime_type = getattr(asset, "mime_type", "")
    return getattr(asset, "type", None) == "image" or (
        isinstance(mime_type, str) and mime_type.startswith("image/")
    )


def _mount_spa(app: FastAPI) -> None:
    static_root = Path(__file__).resolve().parents[1] / "static"
    if not static_root.exists():
        return
    static_root = static_root.resolve()
    assets = static_root / "assets"
    if assets.exists():
        assets = assets.resolve()

        @app.api_route(
            "/assets/{path:path}",
            methods=["GET", "HEAD"],
            include_in_schema=False,
            response_model=None,
        )
        def spa_asset(path: str, request: Request) -> FileResponse | JSONResponse:
            if not path or path.endswith(".gz"):
                return JSONResponse({"detail": "Not found"}, status_code=404)
            target = (assets / path).resolve()
            if not target.is_relative_to(assets) or not target.is_file():
                return JSONResponse({"detail": "Not found"}, status_code=404)
            return _spa_asset_response(target, request)

    @app.api_route(
        "/api",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
        include_in_schema=False,
        response_model=None,
    )
    @app.api_route(
        "/api/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
        include_in_schema=False,
        response_model=None,
    )
    def missing_api(path: str = "") -> JSONResponse:
        return JSONResponse({"detail": "Not found"}, status_code=404)

    @app.get("/{path:path}", include_in_schema=False, response_model=None)
    def spa(path: str) -> FileResponse | JSONResponse:
        if path == "api" or path.startswith("api/"):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        target = (static_root / path).resolve()
        if path and not target.is_relative_to(static_root):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        if path and target.is_file():
            return _spa_static_response(
                target,
                cache_control=_static_cache_control(path),
            )
        index = (static_root / "index.html").resolve()
        if index.is_relative_to(static_root) and index.is_file():
            return _spa_static_response(index, cache_control=_SPA_CACHE_CONTROL)
        return JSONResponse({"detail": "SPA is not built"}, status_code=404)


def _spa_asset_response(target: Path, request: Request) -> FileResponse:
    headers = _static_file_headers(_ASSET_CACHE_CONTROL)
    if target.suffix in _COMPRESSIBLE_STATIC_SUFFIXES:
        headers["Vary"] = "Accept-Encoding"
        gzip_target = target.with_name(f"{target.name}.gz")
        if gzip_target.is_file() and _accepts_gzip(request):
            return _spa_static_response(
                gzip_target,
                cache_control=_ASSET_CACHE_CONTROL,
                content_encoding="gzip",
                media_path=target,
                vary="Accept-Encoding",
            )
    return FileResponse(
        target,
        media_type=_static_media_type(target),
        headers=headers,
    )


def _spa_static_response(
    target: Path,
    *,
    cache_control: str,
    content_encoding: str | None = None,
    media_path: Path | None = None,
    vary: str | None = None,
) -> FileResponse:
    headers = _static_file_headers(cache_control)
    if content_encoding is not None:
        headers["Content-Encoding"] = content_encoding
    if vary is not None:
        headers["Vary"] = vary
    return FileResponse(
        target,
        media_type=_static_media_type(media_path or target),
        headers=headers,
    )


def _static_file_headers(cache_control: str) -> dict[str, str]:
    return {
        "Cache-Control": cache_control,
        "X-Content-Type-Options": "nosniff",
    }


def _static_cache_control(path: str) -> str:
    return _SPA_CACHE_CONTROL if path == "index.html" else _STATIC_CACHE_CONTROL


def _static_media_type(path: Path) -> str | None:
    if path.name == "manifest.webmanifest":
        return "application/manifest+json"
    media_type, _encoding = mimetypes.guess_type(path.name)
    return media_type


def _accepts_gzip(request: Request) -> bool:
    accept_encoding = request.headers.get("accept-encoding", "")
    gzip_quality: float | None = None
    wildcard_quality: float | None = None
    for raw_item in accept_encoding.split(","):
        parts = [part.strip().lower() for part in raw_item.split(";")]
        if not parts or parts[0] not in {"gzip", "*"}:
            continue
        quality = _accept_encoding_quality(parts[1:])
        if parts[0] == "gzip":
            gzip_quality = quality
        else:
            wildcard_quality = quality
    if gzip_quality is not None:
        return gzip_quality > 0
    return wildcard_quality is not None and wildcard_quality > 0


def _accept_encoding_quality(parameters: list[str]) -> float:
    quality = 1.0
    for parameter in parameters:
        if not parameter.startswith("q="):
            continue
        try:
            quality = float(parameter.removeprefix("q="))
        except ValueError:
            return 0.0
    return quality
