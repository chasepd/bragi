"""Shared provider-facing request and response types."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from bragi.interaction_mode import InteractionMode
from bragi.providers.structured_schema import normalize_strict_json_schema


class ProviderCapability(StrEnum):
    CHAT = "chat"
    MODEL_LISTING = "model_listing"
    IMAGE_GENERATION = "image_generation"
    IMAGE_TO_IMAGE = "image_to_image"
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"
    IMAGE_PLUS_TEXT_TO_VIDEO = "image_plus_text_to_video"
    VISION = "vision"
    STRUCTURED_OUTPUT = "structured_output"
    TOOL_CALLING = "tool_calling"
    BLOCKED_OUTPUT_FALLBACK = "blocked_output_fallback"


class ProviderGenerationParameter(StrEnum):
    TEMPERATURE = "temperature"
    MAX_OUTPUT_TOKENS = "max_output_tokens"
    IMAGE_DIMENSIONS = "image_dimensions"
    IMAGE_SAFE_MODE = "image_safe_mode"


@dataclass(frozen=True)
class ProviderConfigStatus:
    provider: str
    configured: bool
    authenticated: bool
    error: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderModelPricing:
    input_per_million_tokens_usd: str | None = None
    output_per_million_tokens_usd: str | None = None
    cache_read_per_million_tokens_usd: str | None = None
    cache_write_per_million_tokens_usd: str | None = None
    request_usd: str | None = None
    image_usd: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class ProviderThinkingLevelSupport:
    levels: tuple[str, ...] = ()
    default_level: str | None = None
    default_enabled: bool | None = None
    mandatory: bool = False
    supports_max_tokens: bool = False


@dataclass(frozen=True)
class ProviderModel:
    provider: str
    model_id: str
    display_name: str
    capabilities: frozenset[ProviderCapability]
    context_window: int | None = None
    supported_parameters: frozenset[ProviderGenerationParameter] = field(
        default_factory=frozenset
    )
    pricing: ProviderModelPricing | None = None
    thinking: ProviderThinkingLevelSupport | None = None


@dataclass(frozen=True)
class ProviderModelListResponse:
    models: list[ProviderModel]
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderCatalogEntry:
    slug: str
    name: str
    privacy_policy_url: str | None = None
    terms_of_service_url: str | None = None
    status_page_url: str | None = None
    headquarters: str | None = None
    datacenters: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChatMessage:
    role: str
    body: str
    speaker_name: str | None = None


@dataclass(frozen=True)
class ChatReasoningConfig:
    enabled: bool | None = None
    effort: str | None = None
    max_tokens: int | None = None
    exclude: bool | None = None


NARRATOR_PROMPT_MODE_RICH_CONTEXT = "rich_context"
NARRATOR_PROMPT_MODE_PLAN_FIRST = "plan_first"
CHAT_TURN_DIRECTIVE_PURPOSE_NARRATOR = "narrator"
CHAT_TURN_DIRECTIVE_PURPOSE_CHARACTER_TEXT = "character_text"


class ChatPromptPurpose(StrEnum):
    NARRATOR = "narrator"
    CHARACTER_TEXT = "character_text"
    SCENARIO_GENERATION = "scenario_generation"
    SUMMARY = "summary"
    IMAGE_PROMPT = "image_prompt"


@dataclass(frozen=True)
class ProviderRetryProgress:
    provider: str
    task: str
    failed_attempt: int
    next_attempt: int
    max_attempts: int
    retry_delay_ms: int
    error_category: str
    http_status: int | None = None


ProviderRetryProgressCallback = Callable[[ProviderRetryProgress], None]


@dataclass(frozen=True)
class ChatRequest:
    provider: str
    model_id: str
    messages: tuple[ChatMessage, ...]
    interaction_mode: InteractionMode = InteractionMode.ROLEPLAY
    response_style_section: str | None = None
    scenario_instructions: str = ""
    user_narration_guidance: str = ""
    custom_instructions: str = ""
    regeneration_feedback: str = ""
    turn_directive: str = ""
    turn_directive_purpose: str = CHAT_TURN_DIRECTIVE_PURPOSE_NARRATOR
    prompt_purpose: ChatPromptPurpose = ChatPromptPurpose.NARRATOR
    phone_activity_context: tuple[str, ...] = ()
    phone_context: tuple[str, ...] = ()
    current_scene_recap: tuple[str, ...] = ()
    director_pressure: str = ""
    character_voice_profiles: tuple[str, ...] = ()
    character_action_plans: tuple[str, ...] = ()
    open_obligations: tuple[str, ...] = ()
    pending_context_suggestions: tuple[str, ...] = ()
    retrieved_scenario_sections: tuple[str, ...] = ()
    retrieved_state: tuple[str, ...] = ()
    retrieved_state_changes: tuple[str, ...] = ()
    retrieved_recent_messages: tuple[str, ...] = ()
    retrieved_media_assets: tuple[str, ...] = ()
    retrieved_character_text_context: tuple[str, ...] = ()
    retrieved_memories: tuple[str, ...] = ()
    retrieved_observations: tuple[str, ...] = ()
    summary: str | None = None
    narration_brief: str = ""
    narration_evidence: tuple[str, ...] = ()
    narrator_prompt_mode: str = NARRATOR_PROMPT_MODE_RICH_CONTEXT
    content_rating: str = "pg-13"
    fade_to_black_enabled: bool = True
    context_breakdown: dict[str, Any] = field(default_factory=dict)
    temperature: float | None = None
    max_output_tokens: int | None = None
    reasoning: ChatReasoningConfig | None = None
    openrouter_provider_routing: dict[str, Any] | None = None
    openrouter_app_title: str | None = None
    retry_progress_callback: ProviderRetryProgressCallback | None = field(
        default=None,
        compare=False,
        repr=False,
    )


@dataclass(frozen=True)
class ChatResponse:
    body: str
    provider: str
    model_id: str
    token_usage: dict[str, int] = field(default_factory=dict)
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    raw_request_payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class ChatStreamChunk:
    delta: str = ""
    token_usage: dict[str, int] = field(default_factory=dict)
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    done: bool = False


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ProviderToolCall:
    id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class ToolCallMessage:
    role: str
    body: str
    speaker_name: str | None = None
    tool_calls: tuple[ProviderToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ToolCallRequest:
    provider: str
    model_id: str
    messages: tuple[ToolCallMessage, ...]
    tools: tuple[ToolDefinition, ...]
    temperature: float | None = None
    max_output_tokens: int | None = None
    reasoning: ChatReasoningConfig | None = None
    openrouter_provider_routing: dict[str, Any] | None = None
    openrouter_app_title: str | None = None


@dataclass(frozen=True)
class ToolCallResponse:
    tool_calls: tuple[ProviderToolCall, ...]
    body: str
    provider: str
    model_id: str
    token_usage: dict[str, int] = field(default_factory=dict)
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    raw_request_payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class ImageRequest:
    provider: str
    model_id: str
    prompt: str
    source_save_id: str
    source_message_id: str
    source_media_asset_id: str | None = None
    source_media_path: Path | None = None
    source_media_asset_ids: tuple[str, ...] = ()
    source_media_paths: tuple[Path, ...] = ()
    content_rating: str = "unrated"
    dimensions: tuple[int, int] | None = None
    safe_mode: bool | None = None
    force_safe_mode: bool = False
    openrouter_provider_routing: dict[str, Any] | None = None
    openrouter_app_title: str | None = None
    retry_progress_callback: ProviderRetryProgressCallback | None = field(
        default=None,
        compare=False,
        repr=False,
    )


@dataclass(frozen=True)
class ImageResponse:
    provider: str
    model_id: str
    image_path: Path | None = None
    image_bytes: bytes | None = None
    revised_prompt: str | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VideoRequest:
    provider: str
    model_id: str
    prompt: str
    source_save_id: str
    source_message_id: str
    source_media_asset_id: str | None = None
    source_media_path: Path | None = None
    content_rating: str = "unrated"
    dimensions: tuple[int, int] | None = None
    safe_mode: bool | None = None
    force_safe_mode: bool = False
    openrouter_app_title: str | None = None


@dataclass(frozen=True)
class VideoResponse:
    provider: str
    model_id: str
    mime_type: str
    video_path: Path | None = None
    video_bytes: bytes | None = None
    revised_prompt: str | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageDescriptionRequest:
    provider: str
    model_id: str
    image_url: str
    prompt: str
    system_prompt: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    reasoning: ChatReasoningConfig | None = None
    openrouter_provider_routing: dict[str, Any] | None = None
    openrouter_app_title: str | None = None


@dataclass(frozen=True)
class ImageDescriptionResponse:
    description: str
    provider: str
    model_id: str
    token_usage: dict[str, int] = field(default_factory=dict)
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StructuredOutputRequest:
    provider: str
    model_id: str
    messages: tuple[ChatMessage, ...]
    schema_name: str
    schema: dict[str, Any]
    temperature: float | None = None
    max_output_tokens: int | None = None
    reasoning: ChatReasoningConfig | None = None
    openrouter_provider_routing: dict[str, Any] | None = None
    openrouter_app_title: str | None = None
    retry_progress_callback: ProviderRetryProgressCallback | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema",
            normalize_strict_json_schema(self.schema),
        )


@dataclass(frozen=True)
class StructuredOutputResponse:
    data: dict[str, Any]
    provider: str
    model_id: str
    token_usage: dict[str, int] = field(default_factory=dict)
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ProviderClient(Protocol):
    provider_name: str

    async def validate_config(self) -> ProviderConfigStatus: ...

    async def list_models(self) -> list[ProviderModel]: ...

    async def chat(self, request: ChatRequest) -> ChatResponse: ...

    async def generate_image(self, request: ImageRequest) -> ImageResponse: ...


@runtime_checkable
class ImageReferenceLimitProvider(Protocol):
    provider_name: str

    def image_reference_limit(self, model_id: str) -> int: ...


@runtime_checkable
class StreamingChatProvider(Protocol):
    provider_name: str

    def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]: ...


@runtime_checkable
class StructuredOutputProvider(Protocol):
    provider_name: str

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse: ...


@runtime_checkable
class ToolCallProvider(Protocol):
    provider_name: str

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse: ...


@runtime_checkable
class VisionProvider(Protocol):
    provider_name: str

    async def describe_image(
        self,
        request: ImageDescriptionRequest,
    ) -> ImageDescriptionResponse: ...


@runtime_checkable
class VideoProvider(Protocol):
    provider_name: str

    async def generate_video(self, request: VideoRequest) -> VideoResponse: ...
