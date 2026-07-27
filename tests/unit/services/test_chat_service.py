from __future__ import annotations

import asyncio
import json
import logging
import shutil
import sqlite3
import threading
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from bragi.persistence.models import (
    ActiveThreadRecord,
    CharacterKnowledgeEdgeRecord,
    CharacterRecord,
    ContextObservationRecord,
    ContextSourceRecord,
    ContextUpdateSuggestionRecord,
    EntityLinkRecord,
    LocationRecord,
    MediaAssetRecord,
    MemoryRecord,
    MessageRecord,
    MessageVisibilityRecord,
    SaveDetailsRecord,
    SceneSnapshotRecord,
    StateChangeRecord,
    SummaryRecord,
    WorldStateRecord,
)
from bragi.persistence.paths import StoragePaths
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ImageRequest,
    ImageResponse,
    ProviderCapability,
    ProviderConfigStatus,
    ProviderModel,
    ProviderToolCall,
    StructuredOutputRequest,
    StructuredOutputResponse,
    ToolCallRequest,
    ToolCallResponse,
)
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.providers.structured_schema import validate_strict_json_schema
from bragi.services import chat_service as chat_service_module
from bragi.services.agentic_context import (
    AGENTIC_CONTEXT_PIPELINE_SETTING,
    PLAN_FIRST_NARRATOR_SETTING,
    RESPONSE_VERIFICATION_MODE_RETRY_ONCE,
    RESPONSE_VERIFICATION_MODE_SETTING,
    CurationResult,
    DatingRouteStageViolation,
    NarrativeBeat,
    NarratorCommitDecision,
    NarratorMessageSpec,
    NarratorVerificationResult,
    NpcIntent,
    ObservationResult,
    PlayerAgencyConstraint,
    RequiredFact,
    StateCommitCandidate,
)
from bragi.services.character_action_planning_service import (
    CHARACTER_ACTION_PLANNING_ENABLED_SETTING,
    CHARACTER_ACTION_PLANNING_TASK,
    CharacterActionPlanningResult,
    CharacterTurnAssessment,
)
from bragi.services.character_text_service import (
    CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING,
)
from bragi.services.character_text_world_update_service import (
    CHARACTER_TEXT_WORLD_UPDATE_RETRY_JOB_TYPE,
)
from bragi.services.chat_history_settings import (
    NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
    NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
    RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
    RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
)
from bragi.services.chat_service import (
    CancellationToken,
    ChatService,
    _selected_context_sources,
)
from bragi.services.content_rating import (
    CONTENT_FILTER_RATING_SETTING,
    FADE_TO_BLACK_ENABLED_SETTING,
)
from bragi.services.context_search_service import (
    ContextSearchResult,
    ContextSearchService,
    SelectedContextItem,
)
from bragi.services.context_update_service import (
    ContextUpdateExtraction,
    ContextUpdateRequest,
    ContextUpdateService,
)
from bragi.services.dating_route_profile_service import (
    DATING_ROUTE_PROFILE_TASK,
    DatingRouteProfileResult,
)
from bragi.services.director_pressure_service import (
    DIRECTOR_PRESSURE_ENABLED_SETTING,
    DirectorPressureResult,
    DirectorPressureState,
)
from bragi.services.generation_settings import (
    OPENROUTER_CHAT_REASONING_OVERRIDES_SETTING,
)
from bragi.services.npc_knowledge_audit_service import (
    NPC_KNOWLEDGE_AUDIT_MODE_HARD_FAIL,
    NPC_KNOWLEDGE_AUDIT_MODE_SETTING,
    NpcKnowledgeAuditResult,
    NpcKnowledgeLeak,
)
from bragi.services.openrouter_routing_settings import (
    OPENROUTER_ROUTING_PROFILES_SETTING,
)
from bragi.services.phrase_denylist import (
    GENERATED_PHRASE_DENYLIST_SETTING,
    SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
)
from bragi.services.post_turn_inference import POST_TURN_INFERENCE_MODE_SETTING
from bragi.services.prompt_inspection import PromptInspectionStore
from bragi.services.scenario_evolution_policy import (
    SCENARIO_EVOLUTION_TURN_INTERVAL_SETTING,
)
from bragi.services.sexual_content_safety import (
    CONTENT_FILTER_TRANSITION,
    FADE_TO_BLACK_TRANSITION,
)
from bragi.services.summary_service import PendingMessageEstimate, SummaryService
from bragi.services.text_script_policy import (
    SCRIPT_GUARD_MODE_LATIN_ONLY_REJECT,
    SCRIPT_GUARD_MODE_OFF,
    SCRIPT_GUARD_MODE_SETTING,
    SCRIPT_GUARD_MODE_SOURCE_AWARE_REJECT,
)
from bragi.services.turn_snapshot_service import TurnSnapshotService
from bragi.world_time_model import canonical_world_time_from_legacy


class RecordingChatProvider:
    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name
        self.chat_requests: list[ChatRequest] = []

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
                model_id=f"{self.provider_name}-chat",
                display_name=f"{self.provider_name.title()} Chat",
                capabilities=frozenset({ProviderCapability.CHAT}),
                context_window=8192,
            )
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        latest = request.messages[-1].body
        return ChatResponse(
            body=f"{self.provider_name} narrator: {latest}",
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"prompt": 11, "completion": 23, "total": 34},
        )

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        raise AssertionError("chat turns must not request image generation")

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        return StructuredOutputResponse(
            data={
                "action": "allow",
                "category": "none",
                "reason": "The narration stays within the content ceiling.",
                "minimum_rating": "g",
            },
            provider=request.provider,
            model_id=request.model_id,
        )


class RecordingContextAndChatProvider(RecordingChatProvider):
    def __init__(self, provider_name: str) -> None:
        super().__init__(provider_name)
        self.structured_output_requests: list[StructuredOutputRequest] = []

    async def list_models(self) -> list[ProviderModel]:
        return [
            ProviderModel(
                provider=self.provider_name,
                model_id=f"{self.provider_name}-context",
                display_name=f"{self.provider_name.title()} Context",
                capabilities=frozenset({ProviderCapability.STRUCTURED_OUTPUT}),
            ),
            ProviderModel(
                provider=self.provider_name,
                model_id=f"{self.provider_name}-chat",
                display_name=f"{self.provider_name.title()} Chat",
                capabilities=frozenset({ProviderCapability.CHAT}),
                context_window=8192,
            ),
        ]

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        if request.schema_name == "content_safety_review":
            return await RecordingChatProvider.generate_structured_output(
                self,
                request,
            )
        self.structured_output_requests.append(request)
        return StructuredOutputResponse(
            data={"selections": []},
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 13},
        )


class ToolContextAndChatProvider(RecordingChatProvider):
    def __init__(
        self,
        provider_name: str,
        *,
        context_model_id: str,
        tool_calls: tuple[ProviderToolCall, ...] = (),
        tool_error: Exception | None = None,
    ) -> None:
        super().__init__(provider_name)
        self.context_model_id = context_model_id
        self.tool_calls = tool_calls
        self.tool_error = tool_error
        self.tool_call_requests: list[ToolCallRequest] = []

    async def list_models(self) -> list[ProviderModel]:
        return [
            ProviderModel(
                provider=self.provider_name,
                model_id=self.context_model_id,
                display_name=f"{self.provider_name.title()} Context",
                capabilities=frozenset({ProviderCapability.TOOL_CALLING}),
            ),
            ProviderModel(
                provider=self.provider_name,
                model_id=f"{self.provider_name}-chat",
                display_name=f"{self.provider_name.title()} Chat",
                capabilities=frozenset({ProviderCapability.CHAT}),
                context_window=8192,
            ),
        ]

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        self.tool_call_requests.append(request)
        if self.tool_error is not None:
            raise self.tool_error
        return ToolCallResponse(
            tool_calls=self.tool_calls,
            body="",
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 13},
        )


class DatingRouteProfileChatProvider(RecordingChatProvider):
    def __init__(
        self,
        provider_name: str,
        *,
        npc_id: str,
    ) -> None:
        super().__init__(provider_name)
        self.npc_id = npc_id
        self.structured_output_requests: list[StructuredOutputRequest] = []
        self.events: list[str] = []

    async def list_models(self) -> list[ProviderModel]:
        return [
            ProviderModel(
                provider=self.provider_name,
                model_id=f"{self.provider_name}-chat",
                display_name=f"{self.provider_name.title()} Chat",
                capabilities=frozenset({ProviderCapability.CHAT}),
                context_window=8192,
            ),
            ProviderModel(
                provider=self.provider_name,
                model_id=f"{self.provider_name}-profile",
                display_name=f"{self.provider_name.title()} Profile",
                capabilities=frozenset({ProviderCapability.STRUCTURED_OUTPUT}),
            ),
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.events.append("chat")
        return await super().chat(request)

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        if request.schema_name == "content_safety_review":
            return await RecordingChatProvider.generate_structured_output(
                self,
                request,
            )
        self.events.append(request.schema_name)
        self.structured_output_requests.append(request)
        return StructuredOutputResponse(
            data={
                "profiles": [
                    {
                        "npc_character_id": self.npc_id,
                        "comfort_with_intimacy": (
                            "open to physical intimacy early when chemistry "
                            "and consent are clear"
                        ),
                        "pacing_preference": "direct and chemistry-led",
                        "known_boundaries": ["no public pressure"],
                        "unresolved_questions": [],
                        "reason": "Mika is direct but private.",
                        "confidence": 0.82,
                    }
                ]
            },
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 13},
        )


class RecordingLookAroundProvider(RecordingChatProvider):
    def __init__(
        self,
        provider_name: str,
        *,
        structured_data: dict[str, object] | None = None,
        answer_body: str | None = None,
    ) -> None:
        super().__init__(provider_name)
        self.structured_data = structured_data or {"suggestions": []}
        self.structured_output_requests: list[StructuredOutputRequest] = []
        self._answer_body = answer_body

    async def list_models(self) -> list[ProviderModel]:
        return [
            ProviderModel(
                provider=self.provider_name,
                model_id=f"{self.provider_name}-chat",
                display_name=f"{self.provider_name.title()} Chat",
                capabilities=frozenset({ProviderCapability.CHAT}),
                context_window=8192,
            ),
            ProviderModel(
                provider=self.provider_name,
                model_id=f"{self.provider_name}-structured",
                display_name=f"{self.provider_name.title()} Structured",
                capabilities=frozenset({ProviderCapability.STRUCTURED_OUTPUT}),
                context_window=8192,
            ),
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        if self._answer_body is None:
            latest = request.messages[-1].body
            body = f"{self.provider_name} narrator: {latest}"
        else:
            body = self._answer_body
        return ChatResponse(
            body=body,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"prompt": 11, "completion": 23, "total": 34},
        )

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        if request.schema_name == "content_safety_review":
            return await RecordingChatProvider.generate_structured_output(
                self,
                request,
            )
        self.structured_output_requests.append(request)
        return StructuredOutputResponse(
            data=self.structured_data,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 17},
        )


class RecordingNarratorPlanningProvider(RecordingChatProvider):
    def __init__(
        self,
        provider_name: str,
        plan_payload: dict[str, object],
    ) -> None:
        super().__init__(provider_name)
        self.plan_payload = plan_payload
        self.structured_output_requests: list[StructuredOutputRequest] = []

    async def list_models(self) -> list[ProviderModel]:
        return [
            ProviderModel(
                provider=self.provider_name,
                model_id=f"{self.provider_name}-chat",
                display_name=f"{self.provider_name.title()} Chat",
                capabilities=frozenset({ProviderCapability.CHAT}),
                context_window=8192,
            ),
            ProviderModel(
                provider=self.provider_name,
                model_id=f"{self.provider_name}-planner",
                display_name=f"{self.provider_name.title()} Planner",
                capabilities=frozenset({ProviderCapability.STRUCTURED_OUTPUT}),
                context_window=8192,
            ),
        ]

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        if request.schema_name == "content_safety_review":
            return await RecordingChatProvider.generate_structured_output(
                self,
                request,
            )
        self.structured_output_requests.append(request)
        return StructuredOutputResponse(
            data=self.plan_payload,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 13},
        )


class RecordingCyoaChatProvider(RecordingChatProvider):
    def __init__(self, provider_name: str) -> None:
        super().__init__(provider_name)
        self.structured_output_requests: list[StructuredOutputRequest] = []

    async def list_models(self) -> list[ProviderModel]:
        return [
            ProviderModel(
                provider=self.provider_name,
                model_id=f"{self.provider_name}-chat",
                display_name=f"{self.provider_name.title()} Chat",
                capabilities=frozenset(
                    {
                        ProviderCapability.CHAT,
                        ProviderCapability.STRUCTURED_OUTPUT,
                    }
                ),
                context_window=8192,
            )
        ]

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        if request.schema_name == "content_safety_review":
            return await RecordingChatProvider.generate_structured_output(
                self,
                request,
            )
        self.structured_output_requests.append(request)
        return StructuredOutputResponse(
            data={
                "choices": [
                    {"body": "Open the brass atlas."},
                    {"body": "Question the librarian."},
                    {"body": "Hide the index under your coat."},
                    {"body": "Step through the blue shelf-door."},
                ]
            },
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 13},
        )


class RecordingCharacterActionChatProvider(RecordingChatProvider):
    def __init__(
        self,
        provider_name: str,
        decisions_by_name: dict[str, dict[str, object]],
    ) -> None:
        super().__init__(provider_name)
        self.decisions_by_name = decisions_by_name
        self.structured_output_requests: list[StructuredOutputRequest] = []
        self.events: list[str] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.events.append("chat")
        return await super().chat(request)

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        if request.schema_name == "content_safety_review":
            return await RecordingChatProvider.generate_structured_output(
                self,
                request,
            )
        self.events.append(request.schema_name)
        self.structured_output_requests.append(request)
        name = _requested_character_name(request.messages[-1].body)
        data = dict(self.decisions_by_name[name])
        data = _with_allowed_character_action_evidence_ids(data, request)
        data["character_id"] = request.schema["properties"]["character_id"]["enum"][0]
        return StructuredOutputResponse(
            data=data,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 13},
        )


class ScriptedDirectorPressureRunner:
    def __init__(
        self,
        result: DirectorPressureResult,
        *,
        events: list[str] | None = None,
    ) -> None:
        self.result = result
        self.events = events
        self.calls: list[tuple[str, str, str]] = []
        self.commits: list[tuple[DirectorPressureResult, str]] = []

    async def assess_completed_turn(
        self,
        *,
        save_id: str,
        player_message_id: str,
        narrator_message_id: str,
    ) -> DirectorPressureResult:
        if self.events is not None:
            self.events.append("director_pressure")
        self.calls.append((save_id, player_message_id, narrator_message_id))
        return self.result

    def commit_after_narration(
        self,
        *,
        result: DirectorPressureResult,
        narrator_message_id: str,
    ) -> None:
        self.commits.append((result, narrator_message_id))


class StaticChatProvider(RecordingChatProvider):
    def __init__(
        self,
        provider_name: str,
        response_body: str,
        *,
        raw_metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(provider_name)
        self.response_body = response_body
        self.raw_metadata = raw_metadata or {}

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        return ChatResponse(
            body=self.response_body,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"prompt": 11, "completion": 23, "total": 34},
            raw_metadata=self.raw_metadata,
        )


class ScriptedContentSafetyProvider(RecordingChatProvider):
    def __init__(self, *actions: str) -> None:
        super().__init__("safety")
        self.actions = actions or ("allow",)
        self.structured_output_requests: list[StructuredOutputRequest] = []

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        request_index = len(self.structured_output_requests)
        self.structured_output_requests.append(request)
        action = self.actions[min(request_index, len(self.actions) - 1)]
        return StructuredOutputResponse(
            data={
                "action": action,
                "category": (
                    "sexual_content"
                    if action == "fade_to_black"
                    else "violence"
                    if action == "block"
                    else "none"
                ),
                "reason": "Scripted content-safety decision.",
                "minimum_rating": (
                    "r" if action in {"block", "fade_to_black"} else "g"
                ),
            },
            provider=request.provider,
            model_id=request.model_id,
        )


def _configure_content_safety_model(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_model_preference(
        task="content_safety",
        provider="safety",
        model_id="safety-model",
    )
    repositories.save_provider_model(
        provider="safety",
        model_id="safety-model",
        display_name="Safety Model",
        capabilities=["structured_output"],
    )


class SequenceChatProvider(RecordingChatProvider):
    def __init__(self, provider_name: str, response_bodies: tuple[str, ...]) -> None:
        super().__init__(provider_name)
        self.response_bodies = response_bodies

    async def chat(self, request: ChatRequest) -> ChatResponse:
        request_index = len(self.chat_requests)
        self.chat_requests.append(request)
        body = self.response_bodies[min(request_index, len(self.response_bodies) - 1)]
        return ChatResponse(
            body=body,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"prompt": 11, "completion": 23, "total": 34},
        )


class StreamingChatProvider(RecordingChatProvider):
    def __init__(
        self,
        provider_name: str,
        chunks: tuple[ChatStreamChunk | Exception, ...],
        *,
        fallback_body: str | None = None,
        fallback_error: Exception | None = None,
    ) -> None:
        super().__init__(provider_name)
        self.chunks = chunks
        self.fallback_body = fallback_body
        self.fallback_error = fallback_error
        self.stream_requests: list[ChatRequest] = []

    async def stream_chat(self, request: ChatRequest) -> Any:
        self.stream_requests.append(request)
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        if self.fallback_error is not None:
            raise self.fallback_error
        latest = request.messages[-1].body
        return ChatResponse(
            body=self.fallback_body or f"{self.provider_name} fallback: {latest}",
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 9},
        )


class SequenceStreamingChatProvider(StreamingChatProvider):
    def __init__(
        self,
        provider_name: str,
        stream_sequences: tuple[tuple[ChatStreamChunk | Exception, ...], ...],
    ) -> None:
        super().__init__(provider_name, ())
        self.stream_sequences = stream_sequences

    async def stream_chat(self, request: ChatRequest) -> Any:
        request_index = len(self.stream_requests)
        self.stream_requests.append(request)
        chunks = self.stream_sequences[
            min(request_index, len(self.stream_sequences) - 1)
        ]
        for chunk in chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


class FailingChatProvider(RecordingChatProvider):
    def __init__(self, provider_name: str, error: Exception) -> None:
        super().__init__(provider_name)
        self.error = error

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        raise self.error


class ChatCompletesThenCancelsProvider(RecordingChatProvider):
    def __init__(self, provider_name: str) -> None:
        super().__init__(provider_name)
        self.chat_completed = False

    async def chat(self, request: ChatRequest) -> ChatResponse:
        response = await super().chat(request)
        self.chat_completed = True
        return response


class BlockingContextSearchProvider(RecordingChatProvider):
    def __init__(self, provider_name: str) -> None:
        super().__init__(provider_name)
        self.entered_structured_output = asyncio.Event()
        self.cancelled_structured_output = asyncio.Event()
        self.structured_output_requests: list[StructuredOutputRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        raise AssertionError("narrator chat must not run after context cancellation")

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_output_requests.append(request)
        self.entered_structured_output.set()
        try:
            await asyncio.Future()
            raise AssertionError(
                "blocking context search provider unexpectedly resumed"
            )
        except asyncio.CancelledError:
            self.cancelled_structured_output.set()
            raise


class BlockingNarratorPlanner:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.requests: list[ChatRequest] = []

    async def plan(
        self,
        *,
        save_id: str,
        request: ChatRequest,
    ) -> NarratorMessageSpec:
        self.requests.append(request)
        self.entered.set()
        try:
            await asyncio.Future()
            raise AssertionError("blocking narrator planner unexpectedly resumed")
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


def _provider_error_with_retry_attempts(
    *,
    category: ProviderErrorCategory,
    message: str,
    status_code: int | None = None,
    retry_attempt_count: int | None = None,
    max_retry_attempts: int | None = None,
    retry_attempts: tuple[dict[str, object], ...],
) -> ProviderError:
    error = ProviderError(
        category=category,
        message=message,
        status_code=status_code,
        retry_attempt_count=retry_attempt_count,
        max_retry_attempts=max_retry_attempts,
    )
    object.__setattr__(error, "retry_attempts", retry_attempts)
    return error


class RecordingStateMemoryProvider(RecordingChatProvider):
    def __init__(
        self,
        provider_name: str,
        *,
        events: list[str],
        structured_data: dict[str, object] | None = None,
        structured_error: Exception | None = None,
    ) -> None:
        super().__init__(provider_name)
        self.events = events
        self.structured_data = structured_data or {
            "state_changes": [],
            "memories": [],
        }
        self.structured_error = structured_error
        self.structured_output_requests: list[StructuredOutputRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.events.append("narrator_chat")
        return await super().chat(request)

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        if request.schema_name == "content_safety_review":
            return await RecordingChatProvider.generate_structured_output(
                self,
                request,
            )
        self.events.append("state_memory_extraction")
        self.structured_output_requests.append(request)
        if self.structured_error is not None:
            raise self.structured_error
        return StructuredOutputResponse(
            data=_structured_payload_with_source_ids(self.structured_data, request),
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 17},
        )


class RecordingScenarioEvolutionProvider(RecordingChatProvider):
    def __init__(
        self,
        provider_name: str,
        *,
        events: list[str],
    ) -> None:
        super().__init__(provider_name)
        self.events = events
        self.chat_response_bodies = [
            "The beacon lens answers in natural prose, washing the gallery red.",
            "The red beacon casts hard shadows over the stair.",
        ]
        self.structured_output_requests: list[StructuredOutputRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.events.append("narrator_chat")
        request_index = len(self.chat_requests)
        self.chat_requests.append(request)
        response_body = self.chat_response_bodies[
            min(request_index, len(self.chat_response_bodies) - 1)
        ]
        return ChatResponse(
            body=response_body,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"prompt": 13, "completion": 21, "total": 34},
        )

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        if request.schema_name == "content_safety_review":
            return await RecordingChatProvider.generate_structured_output(
                self,
                request,
            )
        self.structured_output_requests.append(request)
        schema_name = request.schema_name.casefold()
        data: dict[str, object]
        if "scenario" in schema_name and "evolution" in schema_name:
            self.events.append("scenario_evolution")
            data = {
                "content": {
                    "current_scene": "The beacon gallery is hot with warning light.",
                },
                "reason": "The completed turn changed the reusable setup.",
                "source_message_id": _message_id_for_role(request, "narrator"),
            }
        else:
            self.events.append("state_memory_extraction")
            data = _structured_payload_with_source_ids(
                {"state_changes": [], "memories": []},
                request,
            )
        return StructuredOutputResponse(
            data=data,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 17},
        )


class RecordingPostTurnStructuredProvider(RecordingChatProvider):
    def __init__(
        self,
        provider_name: str,
        *,
        events: list[str],
        character_maintenance_decisions: list[dict[str, object]] | None = None,
    ) -> None:
        super().__init__(provider_name)
        self.events = events
        self.character_maintenance_decisions = character_maintenance_decisions or []
        self.structured_output_requests: list[StructuredOutputRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        raise AssertionError(
            "post-turn context update must use structured output, not chat text"
        )

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        if request.schema_name == "content_safety_review":
            return await RecordingChatProvider.generate_structured_output(
                self,
                request,
            )
        self.structured_output_requests.append(request)
        schema_name = request.schema_name.casefold()
        data: dict[str, object]
        if "context" in schema_name and "update" in schema_name:
            self.events.append("context_update")
            data = {
                "scene": {},
                "locations": [],
                "characters": [],
                "active_threads": [],
                "entity_links": [],
            }
        elif "world_time" in schema_name:
            self.events.append("world_time_reconciliation")
            data = {
                "changed": False,
                "time_of_day": "",
                "day_of_week": "",
                "days_elapsed": 0,
                "evidence_source_id": "",
                "evidence_quote": "",
                "confidence": 0.0,
                "reason": "No durable world-time change.",
            }
        elif "character" in schema_name and "maintenance" in schema_name:
            self.events.append("character_maintenance")
            data = {"decisions": self.character_maintenance_decisions}
        elif "scenario" in schema_name and "evolution" in schema_name:
            self.events.append("scenario_evolution")
            data = {
                "content": {
                    "current_scene": "The beacon gallery is hot with warning light.",
                },
                "reason": "The completed turn changed the reusable setup.",
                "source_message_id": _message_id_for_role(request, "narrator"),
            }
        elif "context" in schema_name and "observation" in schema_name:
            self.events.append("context_observation_extraction")
            data = {"observations": []}
        else:
            self.events.append("state_memory_extraction")
            data = _structured_payload_with_source_ids(
                {"state_changes": [], "memories": []},
                request,
            )
        return StructuredOutputResponse(
            data=data,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 17},
        )


class ScriptedPostTurnStructuredProvider(RecordingPostTurnStructuredProvider):
    def __init__(
        self,
        provider_name: str,
        *,
        response_bodies: tuple[str, ...],
        events: list[str],
        state_data: dict[str, object] | None = None,
    ) -> None:
        super().__init__(provider_name, events=events)
        self.response_bodies = response_bodies
        self.state_data = state_data or {"state_changes": [], "memories": []}

    async def chat(self, request: ChatRequest) -> ChatResponse:
        request_index = len(self.chat_requests)
        self.chat_requests.append(request)
        self.events.append("narrator_chat")
        body = self.response_bodies[min(request_index, len(self.response_bodies) - 1)]
        return ChatResponse(
            body=body,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"prompt": 11, "completion": 23, "total": 34},
        )

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        if request.schema_name == "state_memory_extraction":
            self.structured_output_requests.append(request)
            self.events.append("state_memory_extraction")
            return StructuredOutputResponse(
                data=_structured_payload_with_source_ids(self.state_data, request),
                provider=request.provider,
                model_id=request.model_id,
                token_usage={"total": 17},
            )
        return await super().generate_structured_output(request)


class RecordingPostTurnToolProvider(RecordingPostTurnStructuredProvider):
    def __init__(
        self,
        provider_name: str,
        *,
        events: list[str],
    ) -> None:
        super().__init__(provider_name, events=events)
        self.tool_call_requests: list[ToolCallRequest] = []

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        tool_names = {tool.name for tool in request.tools}
        self.events.append(
            "state_memory_tool_call"
            if "patch_world_state" in tool_names
            else "context_update_tool_call"
        )
        self.tool_call_requests.append(request)
        return ToolCallResponse(
            tool_calls=(),
            body="",
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 19},
        )


class RecordingAgenticPipelineProvider(RecordingChatProvider):
    def __init__(
        self,
        provider_name: str,
        *,
        events: list[str],
        state_data: dict[str, object],
        curation_memory_body: str,
    ) -> None:
        super().__init__(provider_name)
        self.events = events
        self.state_data = state_data
        self.curation_memory_body = curation_memory_body
        self.structured_output_requests: list[StructuredOutputRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.events.append("narrator_chat")
        return await super().chat(request)

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        if request.schema_name == "content_safety_review":
            return await RecordingChatProvider.generate_structured_output(
                self,
                request,
            )
        self.structured_output_requests.append(request)
        schema_name = request.schema_name.casefold()
        if schema_name == "state_memory_extraction":
            self.events.append("state_memory_extraction")
            data = _structured_payload_with_source_ids(self.state_data, request)
        elif schema_name == "context_observation_extraction":
            self.events.append("fact_observation")
            source_ids = _observation_source_message_ids(request)
            data = {
                "observations": [
                    {
                        "observation_type": "player_preference",
                        "claim": "Mara likes concise narration.",
                        "evidence_quote": "I climb toward the beacon lens.",
                        "source_message_ids": source_ids,
                        "scope": "durable",
                        "confidence": 0.9,
                        "tags": ["tone"],
                    }
                ]
            }
        elif schema_name == "context_observation_curation":
            self.events.append("memory_curation")
            observation_id = _first_curation_observation_id(request)
            data = {
                "decisions": [
                    {
                        "observation_id": observation_id,
                        "action": "durable_memory",
                        "reason": "Stable narrator preference.",
                        "confidence": 0.88,
                        "memory_body": self.curation_memory_body,
                        "context_title": "",
                        "context_body": "",
                        "tags": ["tone"],
                    }
                ]
            }
        else:
            raise AssertionError(
                f"unexpected structured request: {request.schema_name}"
            )
        return StructuredOutputResponse(
            data=data,
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 17},
        )


def _structured_payload_with_source_ids(
    data: dict[str, object],
    request: StructuredOutputRequest,
) -> dict[str, object]:
    payload = deepcopy(data)
    source_message_id = _message_id_for_role(
        request, "narrator"
    ) or _message_id_for_role(request, "player")
    for key in ("state_changes", "memories", "conflicts"):
        raw_items = payload.get(key)
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            if isinstance(item, dict) and "source_message_id" not in item:
                item["source_message_id"] = source_message_id
    return payload


def _first_curation_observation_id(request: StructuredOutputRequest) -> str:
    decisions = request.schema["properties"]["decisions"]
    items = decisions["items"]
    observation_id = items["properties"]["observation_id"]
    return str(observation_id["enum"][0])


def _observation_source_message_ids(request: StructuredOutputRequest) -> list[str]:
    observations = request.schema["properties"]["observations"]
    items = observations["items"]
    source_message_ids = items["properties"]["source_message_ids"]
    source_item = source_message_ids["items"]
    return [str(message_id) for message_id in source_item.get("enum", [])]


def _with_allowed_character_action_evidence_ids(
    value: object,
    request: StructuredOutputRequest,
) -> dict[str, object]:
    source_texts = _character_action_evidence_source_texts(request)
    allowed = set(source_texts)

    def normalize(source_id: str) -> str:
        if source_id in allowed:
            return source_id
        if source_id in {"message:latest", "message:player"}:
            return next(
                (
                    allowed_id
                    for allowed_id in reversed(tuple(source_texts))
                    if allowed_id.startswith("message:")
                ),
                source_id,
            )
        placeholder_prefixes = {
            "scene_snapshot:snapshot-1": "scene_snapshot:",
            "character:lio": "character:",
        }
        prefix = placeholder_prefixes.get(source_id)
        if prefix is None:
            return source_id
        return next(
            (allowed_id for allowed_id in allowed if allowed_id.startswith(prefix)),
            source_id,
        )

    def rewrite(item: object) -> object:
        if isinstance(item, dict):
            rewritten: dict[str, object] = {}
            normalized_ids: list[str] | None = None
            for key, raw_value in item.items():
                if key == "evidence_source_ids" and isinstance(raw_value, list):
                    normalized_ids = [
                        normalize(str(source_id)) for source_id in raw_value
                    ]
                    rewritten[key] = normalized_ids
                else:
                    rewritten[key] = rewrite(raw_value)
            if normalized_ids and "evidence_quote" not in rewritten:
                quote = next(
                    (
                        source_texts[source_id]
                        for source_id in normalized_ids
                        if source_id in source_texts
                    ),
                    "",
                )
                if quote:
                    rewritten["evidence_quote"] = quote
            return rewritten
        if isinstance(item, list):
            return [rewrite(child) for child in item]
        return item

    rewritten = rewrite(value)
    assert isinstance(rewritten, dict)
    return rewritten


def _character_action_evidence_source_texts(
    request: StructuredOutputRequest,
) -> dict[str, str]:
    body = request.messages[-1].body
    sources: dict[str, str] = {}
    for line in body.splitlines():
        if not line.startswith("- ") or ": " not in line:
            continue
        source_id, source_text = line.removeprefix("- ").split(": ", 1)
        if ":" in source_id:
            sources[source_id] = source_text
    return sources


def _character_action_evidence_source_ids(
    request: StructuredOutputRequest,
) -> list[str]:
    evidence_source_ids = request.schema["properties"]["evidence_source_ids"]
    source_item = evidence_source_ids["items"]
    return [str(source_id) for source_id in source_item.get("enum", [])]


def _message_id_for_role(request: StructuredOutputRequest, role: str) -> str:
    marker = f" [{role}] "
    for message in request.messages:
        for line in message.body.splitlines():
            if line.startswith("- ") and marker in line:
                return line[2:].split(" ", 1)[0]
    return ""


def _only_request_matching(
    requests: list[StructuredOutputRequest],
    *schema_name_parts: str,
) -> StructuredOutputRequest:
    matches = [
        request
        for request in requests
        if all(part in request.schema_name.casefold() for part in schema_name_parts)
    ]
    assert len(matches) == 1
    return matches[0]


class ScriptedContextSearch:
    def __init__(
        self,
        result: ContextSearchResult,
        events: list[str] | None = None,
    ) -> None:
        self.result = result
        self.events = events if events is not None else []
        self.calls: list[tuple[str, str]] = []

    async def search(
        self,
        *,
        save_id: str,
        player_message_id: str,
    ) -> ContextSearchResult:
        self.events.append("context_search")
        self.calls.append((save_id, player_message_id))
        return self.result


class ScriptedWorldTimeRunner:
    def __init__(
        self,
        repositories: PersistenceRepositories,
        *,
        events: list[str] | None = None,
    ) -> None:
        self.repositories = repositories
        self.events = events if events is not None else []
        self.calls: list[tuple[str, str]] = []

    async def advance_time_if_supported(
        self,
        *,
        save_id: str,
        latest_message_id: str,
    ) -> dict[str, object]:
        self.events.append("world_time")
        self.calls.append((save_id, latest_message_id))
        snapshot = self.repositories.get_scene_snapshot(save_id)
        canonical_world_time = canonical_world_time_from_legacy(
            in_world_time="Tuesday evening",
            time_of_day="evening",
            day_of_week="tuesday",
            world_day_index=snapshot.world_day_index if snapshot else None,
            source_message_id=latest_message_id,
        )
        self.repositories.upsert_scene_snapshot(
            save_id=save_id,
            current_location_id=snapshot.current_location_id if snapshot else None,
            situation=snapshot.situation if snapshot else "",
            objective=snapshot.objective if snapshot else "",
            in_world_time="Tuesday evening",
            time_of_day="evening",
            day_of_week="tuesday",
            world_day_index=snapshot.world_day_index if snapshot else None,
            world_time_day_index=canonical_world_time.day_index,
            world_time_day_label=canonical_world_time.day_label,
            world_time_phase=canonical_world_time.phase,
            world_time_clock_minutes=canonical_world_time.clock_minutes,
            world_time_period_label=canonical_world_time.period_label,
            world_time_source_message_id=canonical_world_time.source_message_id,
            world_time_confidence=canonical_world_time.confidence,
            weather=snapshot.weather if snapshot else "",
            mood=snapshot.mood if snapshot else "",
            nearby_objects=snapshot.nearby_objects if snapshot else [],
            hazards=snapshot.hazards if snapshot else [],
            present_character_ids=snapshot.present_character_ids if snapshot else [],
            source_message_id=latest_message_id,
            locked_fields=snapshot.locked_fields if snapshot else [],
            snapshot_id=snapshot.id if snapshot else None,
            first_seen_message_id=snapshot.first_seen_message_id if snapshot else None,
            last_updated_message_id=latest_message_id,
        )
        return {"status": "applied"}

    async def reconcile_completed_turn(
        self,
        *,
        save_id: str,
        player_message_id: str,
        narrator_message_id: str,
    ) -> dict[str, object]:
        self.events.append("world_time_reconciliation")
        return {
            "status": "applied",
            "source_message_ids": [player_message_id, narrator_message_id],
        }


class FailingContextSearch:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls: list[tuple[str, str]] = []

    async def search(
        self,
        *,
        save_id: str,
        player_message_id: str,
    ) -> ContextSearchResult:
        self.calls.append((save_id, player_message_id))
        raise self.error


class ScriptedNarratorPlanner:
    def __init__(self, spec: NarratorMessageSpec) -> None:
        self.spec = spec
        self.calls: list[tuple[str, ChatRequest]] = []

    async def plan(
        self,
        *,
        save_id: str,
        request: ChatRequest,
    ) -> NarratorMessageSpec:
        self.calls.append((save_id, request))
        return self.spec


class FailingNarratorPlanner:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls: list[tuple[str, ChatRequest]] = []

    async def plan(
        self,
        *,
        save_id: str,
        request: ChatRequest,
    ) -> NarratorMessageSpec:
        self.calls.append((save_id, request))
        raise self.error


class ScriptedNarratorVerifier:
    def __init__(
        self,
        result: (
            NarratorVerificationResult
            | Exception
            | tuple[NarratorVerificationResult | Exception, ...]
        ),
    ) -> None:
        self.results = result if isinstance(result, tuple) else (result,)
        self.calls: list[tuple[str, ChatRequest, NarratorMessageSpec, str]] = []

    async def verify(
        self,
        *,
        save_id: str,
        source_request: ChatRequest,
        spec: NarratorMessageSpec,
        narrator_body: str,
    ) -> NarratorVerificationResult:
        self.calls.append((save_id, source_request, spec, narrator_body))
        index = min(len(self.calls) - 1, len(self.results) - 1)
        result = self.results[index]
        if isinstance(result, Exception):
            raise result
        return result


def _scene_presence_candidate(
    character_id: str,
    *,
    action: str,
    candidate_id: str | None = None,
    evidence_quote: str = "Lio leaves",
) -> StateCommitCandidate:
    return StateCommitCandidate(
        operation="update",
        state_key="scene.presence",
        value={"action": action, "evidence_quote": evidence_quote},
        reason=f"Character should {action} if the narrator renders it.",
        confidence=0.88,
        evidence_source_ids=("message:latest",),
        evidence_quote=evidence_quote,
        candidate_id=candidate_id or f"scene_presence:{character_id}:{action}",
        candidate_type="scene_presence",
        field_path="present_character_ids",
        character_id=character_id,
    )


def _learned_memory_candidate(
    character_id: str,
    *,
    body: str,
    candidate_id: str = "character_learned_memory:mara:0",
) -> StateCommitCandidate:
    return StateCommitCandidate(
        operation="create",
        state_key="character.learned_memory",
        value={
            "body": body,
            "tags": ["mara", "beacon"],
            "knowledge_state": "knows",
            "acquisition_method": "told",
            "evidence_quote": "ember dawn wakes the beacon",
        },
        reason="Mara learned the phrase from the rendered turn.",
        confidence=0.86,
        evidence_source_ids=("message:latest",),
        evidence_quote="ember dawn wakes the beacon",
        candidate_id=candidate_id,
        candidate_type="character_learned_memory",
        character_id=character_id,
    )


def _knowledge_edge_candidate(
    character_id: str,
    *,
    target_type: str,
    target_id: str,
    candidate_id: str = "character_knowledge_edge:mara:memory",
) -> StateCommitCandidate:
    return StateCommitCandidate(
        operation="upsert",
        state_key="character.knowledge_edge",
        value={
            "target_type": target_type,
            "target_id": target_id,
            "knowledge_state": "knows",
            "acquisition_method": "told",
            "evidence_quote": "ember dawn",
        },
        reason="Mara can know this existing evidence after the turn.",
        confidence=0.84,
        evidence_source_ids=("message:latest",),
        evidence_quote="ember dawn",
        candidate_id=candidate_id,
        candidate_type="character_knowledge_edge",
        character_id=character_id,
        target_type=target_type,
        target_id=target_id,
        safe_without_narration_allowed=True,
    )


def _scene_snapshot_field_candidate(
    *,
    field_path: str,
    value: object,
    candidate_id: str = "scene_snapshot:mood",
    evidence_quote: str | None = None,
) -> StateCommitCandidate:
    return StateCommitCandidate(
        operation="update",
        state_key=f"scene_snapshot.{field_path}",
        value={
            field_path: value,
            "evidence_quote": (
                evidence_quote
                if evidence_quote is not None
                else str(value)
            ),
        },
        reason=f"Update scene {field_path} if rendered.",
        confidence=0.82,
        evidence_source_ids=("message:latest",),
        evidence_quote=(
            evidence_quote
            if evidence_quote is not None
            else str(value)
        ),
        candidate_id=candidate_id,
        candidate_type="scene_snapshot_field",
        field_path=field_path,
    )


def _commit_decision(
    candidate: StateCommitCandidate,
    *,
    status: str = "rendered",
    safe_to_commit: bool = True,
    reason: str = "Verified in accepted narrator text.",
) -> NarratorCommitDecision:
    return NarratorCommitDecision(
        candidate_id=candidate.candidate_id,
        candidate_type=candidate.candidate_type,
        status=status,
        safe_to_commit=safe_to_commit,
        reason=reason,
        evidence_quote="verified phrase" if safe_to_commit else "",
    )


def _passing_verification(
    *decisions: NarratorCommitDecision,
) -> NarratorVerificationResult:
    return NarratorVerificationResult(
        passed=True,
        issues=(),
        retry_feedback="",
        confidence=0.92,
        commit_decisions=decisions,
    )


def _create_dating_chat_save(
    repositories: PersistenceRepositories,
    *,
    stage: str,
) -> tuple[str, str, str]:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Last Summer",
        premise="A summer of route choices.",
        player_role="Transfer student",
        content={"player_character_name": "Lio Takahashi"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Summer Save")
    player = repositories.add_character(
        save_id=save.id,
        name="Lio Takahashi",
        met=True,
        is_player_character=True,
    )
    npc = repositories.add_character(
        save_id=save.id,
        name="Mika Arai",
        met=True,
        relationships={player.name: "romance option for Lio Takahashi"},
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Lio and Mika linger near the festival gate.",
        present_character_ids=[npc.id],
        world_day_index=2,
    )
    repositories.upsert_dating_route_state(
        save_id=save.id,
        player_character_id=player.id,
        npc_character_id=npc.id,
        stage=stage,
        first_met_world_day_index=0,
        completed_interactions=3 if stage == "exclusive" else 1,
        dates_completed=2 if stage == "exclusive" else 0,
        next_reasonable_step=(
            "deepen the established exclusive relationship"
            if stage == "exclusive"
            else "build early interest or exchange contact info"
        ),
        route_id="route-mika",
    )
    return save.id, player.id, npc.id


class FailingContextUpdateService:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def update_after_turn(
        self,
        *,
        save_id: str,
        source_message_ids: tuple[str, ...],
    ) -> object:
        self.calls.append((save_id, source_message_ids))
        raise self.error


class RecordingContextUpdateService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def update_after_turn(
        self,
        *,
        save_id: str,
        source_message_ids: tuple[str, ...],
    ) -> object:
        self.calls.append((save_id, source_message_ids))
        return object()


class BlockingContextUpdateExtractor:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.requests: list[ContextUpdateRequest] = []

    async def extract(self, request: ContextUpdateRequest) -> ContextUpdateExtraction:
        self.requests.append(request)
        self.started.set()
        try:
            await asyncio.Future()
            raise AssertionError("blocking context update extractor resumed")
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class ScriptedNpcKnowledgeAuditor:
    def __init__(self, results: tuple[NpcKnowledgeAuditResult, ...]) -> None:
        self.results = results
        self.calls: list[tuple[str, str]] = []

    async def audit_response(
        self,
        *,
        save_id: str,
        player_message: MessageRecord,
        narrator_body: str,
        request: ChatRequest,
    ) -> NpcKnowledgeAuditResult:
        self.calls.append((save_id, narrator_body))
        index = min(len(self.calls) - 1, len(self.results) - 1)
        return self.results[index]


class ResultContextUpdateService:
    class Result:
        def __init__(self, job_result: dict[str, object]) -> None:
            self.job_result = job_result

    def __init__(self, result: dict[str, object]) -> None:
        self.result = result
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def update_after_turn(
        self,
        *,
        save_id: str,
        source_message_ids: tuple[str, ...],
    ) -> object:
        self.calls.append((save_id, source_message_ids))
        return self.Result(self.result)


class MutatingContextUpdateService:
    def __init__(self, repositories: PersistenceRepositories) -> None:
        self.repositories = repositories
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def update_after_turn(
        self,
        *,
        save_id: str,
        source_message_ids: tuple[str, ...],
    ) -> object:
        self.calls.append((save_id, source_message_ids))
        self.repositories.upsert_world_state(
            save_id=save_id,
            key="post_turn.marker",
            value={"status": "mutated after narrator"},
            category="debug",
            source_message_id=source_message_ids[-1],
        )
        return object()


class RecordingSummaryService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        events: list[str],
        covers_message_start_id: str,
        covers_message_end_id: str,
    ) -> None:
        self.repositories = repositories
        self.events = events
        self.covers_message_start_id = covers_message_start_id
        self.covers_message_end_id = covers_message_end_id
        self.calls: list[tuple[str, int | None]] = []
        self.pending_messages: list[PendingMessageEstimate | None] = []
        self.summary_id: str | None = None

    async def summarize_if_needed(
        self,
        *,
        save_id: str,
        model_context_window: int | None,
        pending_message: PendingMessageEstimate | None = None,
        current_user_id: str | None = None,
    ) -> object:
        self.events.append("summarization")
        self.calls.append((save_id, model_context_window))
        self.pending_messages.append(pending_message)
        summary = self.repositories.add_summary(
            save_id=save_id,
            covers_message_start_id=self.covers_message_start_id,
            covers_message_end_id=self.covers_message_end_id,
            body="Mara crossed the ash bridge before hearing the windless bell.",
            provider="fake",
            model="fake-summary",
        )
        self.summary_id = summary.id
        return summary


class FailingSummaryService:
    def __init__(self, *, events: list[str], error: Exception) -> None:
        self.events = events
        self.error = error
        self.calls: list[tuple[str, int | None]] = []
        self.pending_messages: list[PendingMessageEstimate | None] = []

    async def summarize_if_needed(
        self,
        *,
        save_id: str,
        model_context_window: int | None,
        pending_message: PendingMessageEstimate | None = None,
        current_user_id: str | None = None,
    ) -> object:
        self.events.append("summarization")
        self.calls.append((save_id, model_context_window))
        self.pending_messages.append(pending_message)
        raise self.error


class RecordingMediaService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        events: list[str],
    ) -> None:
        self.repositories = repositories
        self.events = events
        self.calls: list[str] = []
        self.source_message_ids_at_call: list[str | None] = []
        self.message_roles_at_call: list[list[str]] = []
        self.narrator_bodies_at_call: list[list[str]] = []

    async def generate_automatic_if_due(
        self,
        *,
        save_id: str,
        source_message_id: str | None = None,
        current_user_id: str | None = None,
    ) -> object:
        self.events.append("automatic_media")
        self.calls.append(save_id)
        self.source_message_ids_at_call.append(source_message_id)
        messages = self.repositories.list_messages(save_id)
        self.message_roles_at_call.append([message.role for message in messages])
        self.narrator_bodies_at_call.append(
            [message.body for message in messages if message.role == "narrator"]
        )
        return None


class RecordingPreparedMediaService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        events: list[str],
        generated: object | None = "generated",
    ) -> None:
        self.repositories = repositories
        self.events = events
        self.generated = generated
        self.prepared: list[dict[str, object]] = []
        self.generated_prepared: list[object] = []
        self.prepare_started: asyncio.Event | None = None

    def prepare_automatic_if_due(
        self,
        *,
        save_id: str,
        source_message_id: str | None = None,
    ) -> object | None:
        self.events.append("automatic_media_prepare")
        if self.prepare_started is not None:
            self.prepare_started.set()
        prepared: dict[str, object] = {
            "save_id": save_id,
            "source_message_id": source_message_id,
            "world_state_keys": [
                state.key for state in self.repositories.list_world_state(save_id)
            ],
        }
        self.prepared.append(prepared)
        return prepared

    async def generate_prepared_automatic(
        self,
        prepared: object,
        *,
        current_user_id: str | None = None,
    ) -> object | None:
        self.events.append("automatic_media")
        self.generated_prepared.append(prepared)
        return self.generated

    async def generate_automatic_if_due(
        self,
        *,
        save_id: str,
        source_message_id: str | None = None,
        current_user_id: str | None = None,
    ) -> object:
        raise AssertionError("prepared automatic image path should be used")


class RecordingStatePruningService:
    def __init__(self, *, events: list[str]) -> None:
        self.events = events
        self.calls: list[tuple[str, bool]] = []

    async def prune(
        self,
        *,
        save_id: str,
        review_only: bool = False,
    ) -> object:
        self.events.append("state_pruning")
        self.calls.append((save_id, review_only))
        return None


class RecordingWorldContextRetentionService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def prune(self, save_id: str) -> object:
        self.calls.append(save_id)
        return object()


@pytest.fixture
def repositories(
    tmp_path: Path,
    migrated_database_template: Path,
) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    shutil.copy2(migrated_database_template, database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


class CountingNarrationPersistenceRepositories(PersistenceRepositories):
    def __init__(self, connection: sqlite3.Connection) -> None:
        super().__init__(connection)
        self.read_counts: dict[str, int] = {}

    def _count(self, name: str) -> None:
        self.read_counts[name] = self.read_counts.get(name, 0) + 1

    def load_save_details(
        self,
        save_id: str,
        *,
        message_limit: int | None = None,
        before_message_id: str | None = None,
    ) -> SaveDetailsRecord | None:
        self._count("load_save_details")
        return super().load_save_details(
            save_id,
            message_limit=message_limit,
            before_message_id=before_message_id,
        )

    def list_messages(
        self,
        save_id: str,
        *,
        include_deleted: bool = False,
    ) -> list[MessageRecord]:
        if not include_deleted:
            self._count("messages")
        return super().list_messages(save_id, include_deleted=include_deleted)

    def get_scene_snapshot(self, save_id: str) -> SceneSnapshotRecord | None:
        self._count("scene_snapshot")
        return super().get_scene_snapshot(save_id)

    def list_locations(self, save_id: str) -> list[LocationRecord]:
        self._count("locations")
        return super().list_locations(save_id)

    def list_characters(self, save_id: str) -> list[CharacterRecord]:
        self._count("characters")
        return super().list_characters(save_id)

    def list_active_threads(self, save_id: str) -> list[ActiveThreadRecord]:
        self._count("active_threads")
        return super().list_active_threads(save_id)

    def list_character_knowledge_edges(
        self,
        save_id: str,
        *,
        character_ids: set[str] | frozenset[str] | tuple[str, ...] | None = None,
        include_archived: bool = False,
    ) -> list[CharacterKnowledgeEdgeRecord]:
        self._count("character_knowledge_edges")
        return super().list_character_knowledge_edges(
            save_id,
            character_ids=character_ids,
            include_archived=include_archived,
        )

    def list_message_visibility(
        self,
        save_id: str,
        *,
        message_id: str | None = None,
        character_ids: set[str] | frozenset[str] | tuple[str, ...] | None = None,
    ) -> list[MessageVisibilityRecord]:
        self._count("message_visibility")
        return super().list_message_visibility(
            save_id,
            message_id=message_id,
            character_ids=character_ids,
        )

    def list_entity_links(self, save_id: str) -> list[EntityLinkRecord]:
        self._count("entity_links")
        return super().list_entity_links(save_id)

    def list_world_state(self, save_id: str) -> list[WorldStateRecord]:
        self._count("world_state")
        return super().list_world_state(save_id)

    def list_world_state_including_archived(
        self,
        save_id: str,
    ) -> list[WorldStateRecord]:
        self._count("world_state_including_archived")
        return super().list_world_state_including_archived(save_id)

    def list_state_changes(self, save_id: str) -> list[StateChangeRecord]:
        self._count("state_changes")
        return super().list_state_changes(save_id)

    def list_media_assets(self, save_id: str) -> list[MediaAssetRecord]:
        self._count("media_assets")
        return super().list_media_assets(save_id)

    def list_memories(self, save_id: str) -> list[MemoryRecord]:
        self._count("memories")
        return super().list_memories(save_id)

    def list_summaries(self, save_id: str) -> list[SummaryRecord]:
        self._count("summaries")
        return super().list_summaries(save_id)

    def list_context_observations(
        self,
        save_id: str,
        *,
        statuses: set[str] | frozenset[str] | tuple[str, ...] | None = None,
        include_archived: bool = False,
    ) -> list[ContextObservationRecord]:
        self._count("context_observations")
        return super().list_context_observations(
            save_id,
            statuses=statuses,
            include_archived=include_archived,
        )

    def list_context_sources(
        self,
        save_id: str,
        *,
        source_type: str | None = None,
    ) -> list[ContextSourceRecord]:
        self._count("context_sources")
        return super().list_context_sources(save_id, source_type=source_type)

    def list_context_update_suggestions(
        self,
        save_id: str,
        *,
        status: str | None = None,
    ) -> list[ContextUpdateSuggestionRecord]:
        self._count("context_update_suggestions")
        return super().list_context_update_suggestions(save_id, status=status)


def test_submit_player_turn_persists_messages_and_uses_active_chat_model(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "starting_scene": "The beacon gutters in the tower.",
            "lore": "The buried lens code hums under the ash.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(CONTENT_FILTER_RATING_SETTING, "unrated")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert len(provider.chat_requests) == 1
    request = provider.chat_requests[0]
    assert request.provider == "openrouter"
    assert request.model_id == "anthropic/claude-3.5-sonnet"
    assert request.messages[-1].role == "player"
    assert request.messages[-1].speaker_name == "Mara"
    assert request.messages[-1].body == "I climb toward the beacon lens."

    persisted_messages = repositories.list_messages(save.id)
    assert result.player_message == persisted_messages[0]
    assert result.narrator_message == persisted_messages[1]
    assert persisted_messages[0].role == "player"
    assert persisted_messages[0].provider is None
    assert persisted_messages[0].model is None
    assert persisted_messages[1].role == "narrator"
    assert persisted_messages[1].speaker_name == "Narrator"
    assert persisted_messages[1].body == (
        "openrouter narrator: I climb toward the beacon lens."
    )
    assert persisted_messages[1].provider == "openrouter"
    assert persisted_messages[1].model == "anthropic/claude-3.5-sonnet"
    assert persisted_messages[1].token_estimate == 34


@pytest.mark.parametrize(
    ("mode", "expected_requests", "expected_body"),
    [
        (SCRIPT_GUARD_MODE_SOURCE_AWARE_REJECT, 2, "The lantern holds."),
        (SCRIPT_GUARD_MODE_LATIN_ONLY_REJECT, 2, "The lantern holds."),
        (SCRIPT_GUARD_MODE_OFF, 1, "玩家喜欢简洁叙事。"),
    ],
)
def test_submit_player_turn_applies_script_guard_before_persisting_narrator(
    repositories: PersistenceRepositories,
    mode: str,
    expected_requests: int,
    expected_body: str,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(SCRIPT_GUARD_MODE_SETTING, mode)
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = SequenceChatProvider(
        "openrouter",
        ("玩家喜欢简洁叙事。", "The lantern holds."),
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
        )
    )

    assert len(provider.chat_requests) == expected_requests
    if expected_requests == 2:
        assert "unsupported writing script" in (
            provider.chat_requests[1].regeneration_feedback
        )
    assert result.narrator_message.body == expected_body
    persisted = repositories.list_messages(save.id)
    assert [message.body for message in persisted] == [
        "I climb toward the beacon lens.",
        expected_body,
    ]


def test_submit_player_turn_applies_phrase_guard_before_persisting_narrator(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_scoped_setting(
        scope="global",
        key=GENERATED_PHRASE_DENYLIST_SETTING,
        value="",
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
        value="save-only phrase",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = SequenceChatProvider(
        "openrouter",
        ("The save-only phrase lands flat.", "The lantern holds."),
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
        )
    )

    assert len(provider.chat_requests) == 2
    assert "save-only phrase" in provider.chat_requests[1].regeneration_feedback
    assert result.narrator_message.body == "The lantern holds."
    persisted = repositories.list_messages(save.id)
    assert [message.body for message in persisted] == [
        "I climb toward the beacon lens.",
        "The lantern holds.",
    ]


def test_submit_player_turn_phrase_guard_fails_after_four_total_attempts(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
        value="save-only phrase",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = SequenceChatProvider(
        "openrouter",
        (
            "The save-only phrase lands flat.",
            "Still a save-only phrase.",
            "Another save-only phrase.",
            "Final save-only phrase.",
        ),
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    with pytest.raises(ValueError, match="generated text phrase denylist"):
        asyncio.run(
            service.submit_player_turn(
                save_id=save.id,
                body="I climb toward the beacon lens.",
                speaker_name="Mara",
                run_post_turn_jobs=False,
            )
        )

    assert len(provider.chat_requests) == 4
    persisted = repositories.list_messages(save.id)
    assert [message.role for message in persisted] == ["player"]


def test_submit_player_turn_final_guard_rejects_script_violation_from_phrase_retry(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(
        SCRIPT_GUARD_MODE_SETTING,
        SCRIPT_GUARD_MODE_LATIN_ONLY_REJECT,
    )
    repositories.set_scoped_setting(
        scope="global",
        key=GENERATED_PHRASE_DENYLIST_SETTING,
        value="",
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
        value="save-only phrase",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = SequenceChatProvider(
        "openrouter",
        ("The save-only phrase lands flat.", "玩家喜欢简洁叙事。"),
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    with pytest.raises(ValueError, match="script policy violation"):
        asyncio.run(
            service.submit_player_turn(
                save_id=save.id,
                body="I climb toward the beacon lens.",
                speaker_name="Mara",
                run_post_turn_jobs=False,
            )
        )

    assert len(provider.chat_requests) == 2
    assert "save-only phrase" in provider.chat_requests[1].regeneration_feedback
    persisted = repositories.list_messages(save.id)
    assert [message.role for message in persisted] == ["player"]


def test_submit_player_turn_final_guard_rejects_phrase_from_verifier_retry(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_app_setting(
        RESPONSE_VERIFICATION_MODE_SETTING,
        RESPONSE_VERIFICATION_MODE_RETRY_ONCE,
    )
    repositories.set_scoped_setting(
        scope="global",
        key=GENERATED_PHRASE_DENYLIST_SETTING,
        value="",
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
        value="save-only phrase",
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    provider = SequenceChatProvider(
        "fake",
        (
            "The stair is quiet.",
            "The save-only phrase lands flat.",
        ),
    )
    spec = NarratorMessageSpec(
        intent="Answer the player's move.",
        thesis="The beacon warning must be visible.",
        must_say=("The lens burns red.",),
        avoid=(),
        tone="tense and grounded",
        uncertainties=(),
        evidence_source_ids=("observation:warning",),
    )
    verifier = ScriptedNarratorVerifier(
        NarratorVerificationResult(
            passed=False,
            issues=("Missed the red lens warning.",),
            retry_feedback="Mention the red lens warning before ending the beat.",
            confidence=0.92,
        )
    )
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        narrator_planner=ScriptedNarratorPlanner(spec),
        narrator_verifier=verifier,
    )

    with pytest.raises(ValueError, match="generated text phrase denylist"):
        asyncio.run(
            service.submit_player_turn(
                save_id=save.id,
                body="I climb toward the beacon lens.",
                speaker_name="Mara",
                run_post_turn_jobs=False,
            )
        )

    assert len(provider.chat_requests) == 2
    persisted = repositories.list_messages(save.id)
    assert [message.role for message in persisted] == ["player"]


def test_submit_player_turn_buffers_streamed_narrator_until_script_guard_passes(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(
        SCRIPT_GUARD_MODE_SETTING,
        SCRIPT_GUARD_MODE_SOURCE_AWARE_REJECT,
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = SequenceStreamingChatProvider(
        "openrouter",
        (
            (
                ChatStreamChunk(delta="玩家喜欢"),
                ChatStreamChunk(delta="简洁叙事。", token_usage={"total": 12}),
                ChatStreamChunk(token_usage={"total": 12}, done=True),
            ),
            (
                ChatStreamChunk(delta="The lantern"),
                ChatStreamChunk(delta=" holds.", token_usage={"total": 12}),
                ChatStreamChunk(token_usage={"total": 12}, done=True),
            ),
        ),
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )
    drafts: list[str] = []

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
            narrator_stream_callback=drafts.append,
        )
    )

    assert drafts == ["The lantern holds."]
    assert len(provider.stream_requests) == 2
    assert provider.chat_requests == []
    assert (
        "unsupported writing script"
        in provider.stream_requests[1].regeneration_feedback
    )
    assert result.narrator_message.body == "The lantern holds."
    persisted = repositories.list_messages(save.id)
    assert [message.body for message in persisted] == [
        "I climb toward the beacon lens.",
        "The lantern holds.",
    ]


def test_submit_player_turn_reports_pre_narrator_progress_phases(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )
    progress_events: list[Any] = []

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
            turn_progress_callback=progress_events.append,
        )
    )

    status_texts = [event.status_text for event in progress_events]
    assert status_texts[0] == "Submitting turn"
    assert "Checking history" in status_texts
    assert "Saving player input" in status_texts
    assert "World time unchanged" in status_texts
    assert "Dating route profile skipped" in status_texts
    assert "Selecting context" in status_texts
    assert "Preparing narrator prompt" in status_texts
    assert "Writing narrator response" in status_texts
    assert "Saving narrator response" in status_texts

    final_statuses = {
        job.name: job.status
        for job in progress_events[-1].jobs
    }
    assert final_statuses == {
        "submission": "succeeded",
        "history": "succeeded",
        "input": "succeeded",
        "time_state": "skipped",
        "dating_route_profile": "skipped",
        "character_planning": "skipped",
        "context_selection": "succeeded",
        "prompt": "succeeded",
        "narrator": "succeeded",
        "response_checks": "succeeded",
        "save_narration": "succeeded",
        "action_choices": "skipped",
    }


def test_submit_player_turn_enters_post_input_context_after_input_saved(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": RecordingChatProvider("openrouter")},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )
    observations: list[list[tuple[str, str]]] = []

    class PostInputContext:
        async def __aenter__(self) -> None:
            observations.append(
                [
                    (message.role, message.body)
                    for message in repositories.list_messages(save.id)
                ]
            )

        async def __aexit__(
            self,
            _exc_type: object,
            _exc: object,
            _traceback: object,
        ) -> None:
            return None

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
            post_input_context=PostInputContext,
        )
    )

    assert observations == [[("player", "I climb toward the beacon lens.")]]
    assert [
        (message.role, message.body)
        for message in repositories.list_messages(save.id)
    ] == [
        ("player", "I climb toward the beacon lens."),
        ("narrator", result.narrator_message.body),
    ]


def test_submit_timeskip_turn_enters_post_input_context_after_input_saved(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": RecordingChatProvider("openrouter")},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )
    observations: list[list[tuple[str, str]]] = []

    class PostInputContext:
        async def __aenter__(self) -> None:
            observations.append(
                [
                    (message.role, message.body)
                    for message in repositories.list_messages(save.id)
                ]
            )

        async def __aexit__(
            self,
            _exc_type: object,
            _exc: object,
            _traceback: object,
        ) -> None:
            return None

    result = asyncio.run(
        service.submit_timeskip_turn(
            save_id=save.id,
            instruction="Skip to dawn at the city gates.",
            run_post_turn_jobs=False,
            post_input_context=PostInputContext,
        )
    )

    expected_input = "Timeskip request: Skip to dawn at the city gates."
    assert observations == [[("system", expected_input)]]
    assert [
        (message.role, message.body)
        for message in repositories.list_messages(save.id)
    ] == [
        ("system", expected_input),
        ("narrator", result.narrator_message.body),
    ]


def test_submit_player_turn_adds_agentic_narrator_message_spec(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingChatProvider("fake")
    spec = NarratorMessageSpec(
        intent="Answer the player's move.",
        thesis="The beacon warning should shape the next beat.",
        narrative_beats=(
            NarrativeBeat(
                description="Mara reaches the beacon gallery.",
                evidence_source_ids=("message:player-1",),
            ),
            NarrativeBeat(
                description="The lens burns red before anyone explains why.",
                evidence_source_ids=("observation:warning",),
            ),
        ),
        required_facts=(
            RequiredFact(
                fact="The beacon lens is red.",
                evidence_source_ids=("observation:warning",),
            ),
        ),
        must_say=("The lens burns red.",),
        avoid=("Do not move Mara without consent.",),
        agency_constraints=(
            PlayerAgencyConstraint(
                constraint="Mara decides whether to show the brass warrant.",
                reason="The player has not chosen that action.",
                evidence_source_ids=("message:player-1",),
            ),
        ),
        tone="tense and grounded",
        uncertainties=("Whether the riders saw the tower.",),
        evidence_source_ids=("message:source-narrator", "observation:warning"),
        npc_intents=(
            NpcIntent(
                character_name="Captain Ilyra",
                stance="wary ally",
                current_goal="Keep control of the red lens.",
                next_action="Demand proof before sharing the failsafe.",
                should_comply=False,
                cooperation_conditions=("Mara shows the brass warrant.",),
                boundaries=("Will not abandon the tower.",),
                reason="Her stored motive prioritizes the village.",
                evidence_source_ids=("character:ilyra",),
                character_id="character:ilyra",
            ),
        ),
        state_commit_candidates=(
            StateCommitCandidate(
                operation="upsert",
                state_key="scene.beacon_lens",
                value={"status": "red"},
                reason="The next turn may establish the warning as active.",
                confidence=0.82,
                evidence_source_ids=("state:beacon.lens",),
                evidence_quote="The beacon lens is red.",
            ),
        ),
    )
    planner = ScriptedNarratorPlanner(spec)
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(
                selected_memories=(
                    SelectedContextItem(
                        source_type="memory",
                        source_id="ilyra-duty",
                        text="Ilyra prioritizes the village over visitors.",
                        relevance_note="grounds Ilyra's caution",
                    ),
                ),
            )
        ),
        narrator_planner=planner,
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert len(planner.calls) == 1
    assert planner.calls[0][0] == save.id
    assert planner.calls[0][1].narration_brief == ""
    assert len(provider.chat_requests) == 1
    request = provider.chat_requests[0]
    assert request.retrieved_memories == (
        "[memory:ilyra-duty] Ilyra prioritizes the village over visitors.",
    )
    assert "grounds Ilyra's caution" not in "\n".join(request.retrieved_memories)
    assert "Narration turn plan" in request.narration_brief
    assert "1. Mara reaches the beacon gallery." in request.narration_brief
    assert "The beacon lens is red." in request.narration_brief
    assert "Mara decides whether to show the brass warrant." in (
        request.narration_brief
    )
    assert "The lens burns red." in request.narration_brief
    assert "Do not move Mara without consent." in request.narration_brief
    assert "Captain Ilyra" in request.narration_brief
    assert "id: character:ilyra" in request.narration_brief
    assert "Mara shows the brass warrant." in request.narration_brief
    assert "State commit candidates (do not persist automatically):" in (
        request.narration_brief
    )
    assert request.narration_evidence == (
        "message:source-narrator",
        "observation:warning",
        "message:player-1",
        "character:ilyra",
        "state:beacon.lens",
    )


def test_submit_player_turn_falls_back_when_narrator_planner_fails(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingChatProvider("fake")
    planner = FailingNarratorPlanner(RuntimeError("planner is unavailable"))
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        narrator_planner=planner,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
        )
    )

    assert len(planner.calls) == 1
    assert len(provider.chat_requests) == 1
    assert provider.chat_requests[0].narration_brief == ""
    assert provider.chat_requests[0].narration_evidence == ()
    assert result.narrator_message.body == (
        "fake narrator: I climb toward the beacon lens."
    )


def test_submit_player_turn_uses_plan_first_narrator_request_when_enabled(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "starting_scene": "The beacon gutters in the tower.",
            "tone_genre": "tense frontier fantasy",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="The red lens waits above the beacon gallery.",
        objective="Reach the beacon lens without losing the signal.",
    )
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_app_setting(PLAN_FIRST_NARRATOR_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.append_message(
        save_id=save.id,
        role="player",
        body="I checked the fuse before climbing.",
        speaker_name="Mara",
    )
    repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="*The fuse smelled sharply of ozone.*",
        speaker_name="Narrator",
    )
    spec = NarratorMessageSpec(
        intent="Answer the climb without moving Mara past the choice.",
        thesis="The red lens dominates the next beat.",
        must_say=("The lens burns red.",),
        avoid=("Do not decide whether Mara touches the lens.",),
        tone="tense and grounded",
        uncertainties=("Whether the riders saw the tower.",),
        evidence_source_ids=("message:latest",),
        narrative_beats=(
            NarrativeBeat(
                description="Mara reaches the beacon gallery.",
                evidence_source_ids=("message:latest",),
            ),
        ),
        required_facts=(
            RequiredFact(
                fact="The beacon lens is red.",
                evidence_source_ids=("world_state:state-lens",),
            ),
        ),
    )
    planner = ScriptedNarratorPlanner(spec)
    provider = RecordingChatProvider("fake")
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(
                selected_state=(
                    SelectedContextItem(
                        source_type="world_state",
                        source_id="state-lens",
                        text="beacon.lens: red warning",
                        relevance_note="The player is climbing to the lens.",
                    ),
                ),
                selected_memories=(
                    SelectedContextItem(
                        source_type="memory",
                        source_id="memory-fuse",
                        text="Mara already checked the fuse.",
                        relevance_note="Avoid repeating the fuse inspection.",
                    ),
                ),
                selected_recent_messages=(
                    SelectedContextItem(
                        source_type="message",
                        source_id="older-narrator",
                        text="The fuse smelled sharply of ozone.",
                        relevance_note="Recent sensory continuity.",
                    ),
                ),
            )
        ),
        narrator_planner=planner,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
        )
    )

    assert len(planner.calls) == 1
    planner_request = planner.calls[0][1]
    assert planner_request.narrator_prompt_mode == "rich_context"
    assert planner_request.retrieved_state
    assert planner_request.retrieved_memories
    assert planner_request.retrieved_recent_messages

    request = provider.chat_requests[0]
    assert request.narrator_prompt_mode == "plan_first"
    assert request.messages[-1] == ChatMessage(
        role="player",
        body="I climb toward the beacon lens.",
        speaker_name="Mara",
    )
    assert len(request.messages) >= 3
    assert "Narration turn plan" in request.narration_brief
    assert "Mara reaches the beacon gallery." in request.narration_brief
    assert "The beacon lens is red." in request.narration_brief
    assert "Tone/style: tense frontier fantasy" in request.scenario_instructions
    assert request.retrieved_state
    assert request.retrieved_memories
    assert request.retrieved_recent_messages
    assert request.current_scene_recap
    assert request.character_action_plans == ()
    assert "message:latest" in request.narration_evidence
    assert "world_state:state-lens" in request.narration_evidence

    job = _chat_completion_jobs(repositories, save.id)[-1]
    assert job["result"]["narrator_mode"] == "plan_first"
    assert job["result"]["prompt_context_diagnostics"]["context_breakdown"][
        "narrator_context_policy"
    ] == "plan_plus_context"
    withheld_counts = job["result"]["narrator_context_withheld_counts"]
    assert withheld_counts["retrieved_state"] == 0
    assert withheld_counts["retrieved_memories"] == 0
    assert withheld_counts["retrieved_recent_messages"] == 0
    assert withheld_counts["baseline_recent_messages"] == 0
    prompt_diagnostics = job["result"]["prompt_context_diagnostics"]
    assert prompt_diagnostics["narrator_mode"] == "plan_first"
    assert prompt_diagnostics["narrator_context_withheld_counts"] == withheld_counts
    assert result.narrator_message.body == (
        "fake narrator: I climb toward the beacon lens."
    )


def test_submit_player_turn_uses_rich_request_when_plan_first_plan_is_empty(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_app_setting(PLAN_FIRST_NARRATOR_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingChatProvider("fake")
    planner = ScriptedNarratorPlanner(
        NarratorMessageSpec(
            intent="",
            thesis="",
            must_say=(),
            avoid=(),
            tone="",
            uncertainties=(),
            evidence_source_ids=(),
        )
    )
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(
                selected_memories=(
                    SelectedContextItem(
                        source_type="memory",
                        source_id="memory-fuse",
                        text="Mara already checked the fuse.",
                        relevance_note="Avoid repeating the fuse inspection.",
                    ),
                ),
            )
        ),
        narrator_planner=planner,
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
        )
    )

    request = provider.chat_requests[0]
    assert request.narrator_prompt_mode == "rich_context"
    assert request.narration_brief == ""
    assert request.retrieved_memories == (
        "[memory:memory-fuse] Mara already checked the fuse.",
    )
    job = _chat_completion_jobs(repositories, save.id)[-1]
    assert job["result"]["narrator_mode"] == "rich_context"
    assert job["result"]["narrator_mode_reason"] == "invalid_turn_plan"


def test_submit_player_turn_uses_response_planning_model_preference(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="response_planning",
        provider="fake",
        model_id="fake-planner",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-planner",
        display_name="Fake Planner",
        capabilities=["structured_output"],
    )
    provider = RecordingNarratorPlanningProvider(
        "fake",
        {
            "intent": "Answer the player's move.",
            "thesis": "The beacon warning should shape the next beat.",
            "narrative_beats": [
                {
                    "description": "The red lens dominates the beacon gallery.",
                    "evidence_source_ids": ["message:latest"],
                }
            ],
            "required_facts": [
                {
                    "fact": "The beacon lens is red.",
                    "evidence_source_ids": ["message:latest"],
                }
            ],
            "must_say": ["The lens burns red."],
            "avoid": ["Do not move Mara without consent."],
            "agency_constraints": [
                {
                    "constraint": "Mara chooses whether to show the warrant.",
                    "reason": "The player has not chosen that action.",
                    "evidence_source_ids": ["message:latest"],
                }
            ],
            "tone": "tense and grounded",
            "uncertainties": ["Whether the riders saw the tower."],
            "evidence_source_ids": ["message:latest"],
            "npc_intents": [],
            "state_commit_candidates": [
                {
                    "candidate_id": "scene_snapshot:mood",
                    "candidate_type": "scene_snapshot_field",
                    "operation": "update",
                    "state_key": "scene_snapshot.mood",
                    "field_path": "mood",
                    "character_id": "",
                    "target_type": "",
                    "target_id": "",
                    "value": {"mood": "uneasy"},
                    "safe_without_narration_allowed": False,
                    "reason": "The turn may establish the warning.",
                    "confidence": 0.82,
                    "evidence_source_ids": ["message:latest"],
                    "evidence_quote": "The beacon lens is red.",
                }
            ],
        },
    )
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
        )
    )

    assert len(provider.structured_output_requests) == 1
    structured_request = provider.structured_output_requests[0]
    assert structured_request.schema_name == "narrator_message_plan"
    assert structured_request.provider == "fake"
    assert structured_request.model_id == "fake-planner"
    assert len(provider.chat_requests) == 1
    request = provider.chat_requests[0]
    assert request.model_id == "fake-chat"
    assert "The red lens dominates the beacon gallery." in request.narration_brief
    assert "The beacon lens is red." in request.narration_brief
    assert "State commit candidates (do not persist automatically):" in (
        request.narration_brief
    )
    assert request.narration_evidence == ("message:latest",)


def test_submit_player_turn_retries_once_after_narrator_verifier_failure(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_app_setting(
        RESPONSE_VERIFICATION_MODE_SETTING,
        RESPONSE_VERIFICATION_MODE_RETRY_ONCE,
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    provider = SequenceChatProvider(
        "fake",
        (
            "The stair is quiet.",
            "The lens burns red above the stair.",
        ),
    )
    spec = NarratorMessageSpec(
        intent="Answer the player's move.",
        thesis="The beacon warning must be visible.",
        must_say=("The lens burns red.",),
        avoid=(),
        tone="tense and grounded",
        uncertainties=(),
        evidence_source_ids=("observation:warning",),
    )
    planner = ScriptedNarratorPlanner(spec)
    verifier = ScriptedNarratorVerifier(
        NarratorVerificationResult(
            passed=False,
            issues=(),
            npc_agency_issues=(
                "Ilyra reveals the failsafe without proof or pressure.",
            ),
            retry_feedback="",
            confidence=0.92,
        )
    )
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        narrator_planner=planner,
        narrator_verifier=verifier,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert len(provider.chat_requests) == 2
    assert len(verifier.calls) == 1
    assert verifier.calls[0][0] == save.id
    assert verifier.calls[0][2:] == (spec, "The stair is quiet.")
    assert "The lens burns red." in verifier.calls[0][1].narration_brief
    retry_request = provider.chat_requests[1]
    assert "Ilyra reveals the failsafe" in retry_request.regeneration_feedback
    assert "The lens burns red." in retry_request.narration_brief
    assert retry_request.narration_evidence == ("observation:warning",)
    assert result.narrator_message.body == "The lens burns red above the stair."


def test_submit_player_turn_retries_once_after_narrator_passivity_issue(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_app_setting(
        RESPONSE_VERIFICATION_MODE_SETTING,
        RESPONSE_VERIFICATION_MODE_RETRY_ONCE,
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    provider = SequenceChatProvider(
        "fake",
        (
            "Mara gives you space to decide what to do next.",
            "Mara cuts across the stair and demands the beacon key.",
        ),
    )
    spec = NarratorMessageSpec(
        intent="Answer the player's move.",
        thesis="Mara should put pressure on the beacon choice.",
        must_say=(),
        avoid=(),
        tone="tense and grounded",
        uncertainties=(),
        evidence_source_ids=("character:mara",),
    )
    verifier = ScriptedNarratorVerifier(
        NarratorVerificationResult(
            passed=True,
            issues=(),
            npc_passivity_issues=(
                "Mara has urgent leverage but only gives the player space.",
            ),
            retry_feedback="",
            confidence=0.88,
        )
    )
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        narrator_planner=ScriptedNarratorPlanner(spec),
        narrator_verifier=verifier,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
        )
    )

    assert result.narrator_message.body == (
        "Mara cuts across the stair and demands the beacon key."
    )
    assert len(provider.chat_requests) == 2
    retry_feedback = provider.chat_requests[1].regeneration_feedback
    assert "NPC passivity" in retry_feedback
    assert "gives the player space" in retry_feedback
    job_result = _chat_completion_jobs(repositories, save.id)[-1]["result"]
    assert job_result["narrator_verifier"]["npc_passivity_issue_count"] == 1
    assert job_result["narrator_verifier_retry_used"] is True


def test_submit_player_turn_keeps_early_dating_warmth_without_retry(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, _npc_id = _create_dating_chat_save(
        repositories,
        stage="introduced",
    )
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_app_setting(
        RESPONSE_VERIFICATION_MODE_SETTING,
        RESPONSE_VERIFICATION_MODE_RETRY_ONCE,
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    provider = SequenceChatProvider(
        "fake",
        ("Mika blushes and says she would like to exchange numbers.",),
    )
    spec = NarratorMessageSpec(
        intent="Answer Lio's affection.",
        thesis="Mika can show interest without overcommitting.",
        must_say=(),
        avoid=("Do not make Mika exclusive yet.",),
        tone="warm and grounded",
        uncertainties=(),
        evidence_source_ids=("dating_route_state:route-mika",),
        npc_intents=(
            NpcIntent(
                character_name="Mika Arai",
                character_id="mika",
                stance="interested",
                current_goal="Decide whether to exchange numbers.",
                next_action="Respond warmly without overcommitting.",
                should_comply=True,
                route_stage="introduced",
                max_plausible_escalation=(
                    "warmth, curiosity, light flirtation, and contact exchange"
                ),
            ),
        ),
    )
    verifier = ScriptedNarratorVerifier(_passing_verification())
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        narrator_planner=ScriptedNarratorPlanner(spec),
        narrator_verifier=verifier,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save_id,
            body="I tell Mika she makes the festival feel brighter.",
            speaker_name="Lio",
            run_post_turn_jobs=False,
        )
    )

    assert len(provider.chat_requests) == 1
    assert len(verifier.calls) == 1
    assert result.narrator_message.body == (
        "Mika blushes and says she would like to exchange numbers."
    )


def test_submit_player_turn_runs_proactive_text_after_route_updates(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_id, npc_id = _create_dating_chat_save(
        repositories,
        stage="introduced",
    )
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        situation="Lio leaves the festival gate after Mika heads home.",
        present_character_ids=[],
        world_day_index=2,
    )
    repositories.upsert_character_contact_state(
        save_id=save_id,
        player_character_id=player_id,
        character_id=npc_id,
        character_has_player_number=True,
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingChatProvider("fake")
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save_id,
            body="I wave to Mika from the festival gate.",
            speaker_name="Lio",
        )
    )

    assert result.narrator_message.body == (
        "fake narrator: I wave to Mika from the festival gate."
    )
    text_messages = repositories.list_character_text_messages(save_id=save_id)
    assert [(message.sender, message.character_id) for message in text_messages] == [
        ("character", npc_id),
    ]
    details = repositories.load_save_details(save_id)
    assert details is not None
    assert details.messages == repositories.list_messages(save_id)
    coordinator = _post_turn_jobs(repositories, save_id)[-1]
    assert _post_turn_child_status(coordinator, "proactive_text") == "succeeded"
    text_result = _post_turn_child_result(coordinator, "proactive_text")
    assert text_result["status"] == "sent"
    assert text_result["message_id"] == text_messages[0].id


def test_submit_player_turn_skips_proactive_text_when_chance_zero(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_id, npc_id = _create_dating_chat_save(
        repositories,
        stage="introduced",
    )
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        situation="Lio leaves the festival gate after Mika heads home.",
        present_character_ids=[],
        world_day_index=2,
    )
    repositories.upsert_character_contact_state(
        save_id=save_id,
        player_character_id=player_id,
        character_id=npc_id,
        character_has_player_number=True,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save_id,
        key=CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING,
        value=0,
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingChatProvider("fake")
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save_id,
            body="I wave to Mika from the festival gate.",
            speaker_name="Lio",
        )
    )

    assert result.narrator_message.body == (
        "fake narrator: I wave to Mika from the festival gate."
    )
    assert repositories.list_character_text_messages(save_id=save_id) == []
    assert len(provider.chat_requests) == 1
    coordinator = _post_turn_jobs(repositories, save_id)[-1]
    assert _post_turn_child_status(coordinator, "proactive_text") == "skipped"
    text_result = _post_turn_child_result(coordinator, "proactive_text")
    assert text_result == {
        "save_id": save_id,
        "status": "skipped",
        "reason": "proactive_texts_disabled",
        "candidate_count": 0,
    }


def test_submit_player_turn_skips_proactive_text_when_text_world_retry_pending(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_id, npc_id = _create_dating_chat_save(
        repositories,
        stage="introduced",
    )
    repositories.upsert_character_contact_state(
        save_id=save_id,
        player_character_id=player_id,
        character_id=npc_id,
        character_has_player_number=True,
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.create_job(
        save_id=save_id,
        type=CHARACTER_TEXT_WORLD_UPDATE_RETRY_JOB_TYPE,
        status="queued",
        payload={"text_message_ids": ["text-message-1"]},
    )
    provider = RecordingChatProvider("fake")
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save_id,
            body="I wave to Mika from the festival gate.",
            speaker_name="Lio",
        )
    )

    assert result.narrator_message.body == (
        "fake narrator: I wave to Mika from the festival gate."
    )
    assert len(provider.chat_requests) == 1
    assert repositories.list_character_text_messages(save_id=save_id) == []
    coordinator = _post_turn_jobs(repositories, save_id)[-1]
    assert _post_turn_child_status(coordinator, "proactive_text") == "skipped"
    text_result = _post_turn_child_result(coordinator, "proactive_text")
    assert text_result == {
        "status": "skipped",
        "reason": "pending_text_world_update_retry",
    }


def test_submit_player_turn_runs_proactive_text_after_text_world_retry_terminal(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_id, npc_id = _create_dating_chat_save(
        repositories,
        stage="introduced",
    )
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        situation="Lio leaves the festival gate after Mika heads home.",
        present_character_ids=[],
        world_day_index=2,
    )
    repositories.upsert_character_contact_state(
        save_id=save_id,
        player_character_id=player_id,
        character_id=npc_id,
        character_has_player_number=True,
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="chat_dating_sim",
        provider="fake",
        model_id="fake-chat",
    )
    retry_job = repositories.create_job(
        save_id=save_id,
        type=CHARACTER_TEXT_WORLD_UPDATE_RETRY_JOB_TYPE,
        status="queued",
        payload={"text_message_ids": ["text-message-1"]},
    )
    repositories.update_job(retry_job.id, status="failed", error="terminal failure")
    provider = RecordingChatProvider("fake")
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save_id,
            body="I wave to Mika from the festival gate.",
            speaker_name="Lio",
        )
    )

    assert result.narrator_message.body == (
        "fake narrator: I wave to Mika from the festival gate."
    )
    text_messages = repositories.list_character_text_messages(save_id=save_id)
    assert [(message.sender, message.character_id) for message in text_messages] == [
        ("character", npc_id),
    ]
    coordinator = _post_turn_jobs(repositories, save_id)[-1]
    assert _post_turn_child_status(coordinator, "proactive_text") == "succeeded"


def test_submit_player_turn_retries_once_after_dating_stage_violation(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, _npc_id = _create_dating_chat_save(
        repositories,
        stage="introduced",
    )
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_app_setting(
        RESPONSE_VERIFICATION_MODE_SETTING,
        RESPONSE_VERIFICATION_MODE_RETRY_ONCE,
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    provider = SequenceChatProvider(
        "fake",
        (
            "Mika says she wants to be exclusive forever.",
            "Mika blushes and suggests trading numbers first.",
        ),
    )
    spec = NarratorMessageSpec(
        intent="Answer Lio's affection.",
        thesis="Mika can show interest without overcommitting.",
        must_say=(),
        avoid=("Do not make Mika exclusive yet.",),
        tone="warm and grounded",
        uncertainties=(),
        evidence_source_ids=("dating_route_state:route-mika",),
        npc_intents=(
            NpcIntent(
                character_name="Mika Arai",
                character_id="mika",
                stance="interested",
                current_goal="Decide whether to exchange numbers.",
                next_action="Respond warmly without overcommitting.",
                should_comply=True,
                route_stage="introduced",
                max_plausible_escalation=(
                    "warmth, curiosity, light flirtation, and contact exchange"
                ),
            ),
        ),
    )
    verifier = ScriptedNarratorVerifier(
        NarratorVerificationResult(
            passed=False,
            issues=(),
            retry_feedback="Keep Mika to warmth and contact exchange.",
            confidence=0.91,
            dating_route_stage_violations=(
                DatingRouteStageViolation(
                    character_name="Mika Arai",
                    character_id="mika",
                    route_stage="introduced",
                    escalation="exclusivity or commitment language",
                    reason="Introduced routes cannot jump to exclusivity.",
                    evidence_quote="exclusive forever",
                ),
            ),
        )
    )
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        narrator_planner=ScriptedNarratorPlanner(spec),
        narrator_verifier=verifier,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save_id,
            body="I tell Mika she makes the festival feel brighter.",
            speaker_name="Lio",
            run_post_turn_jobs=False,
        )
    )

    assert len(provider.chat_requests) == 2
    assert "Keep Mika to warmth and contact exchange." in (
        provider.chat_requests[1].regeneration_feedback
    )
    assert result.narrator_message.body == (
        "Mika blushes and suggests trading numbers first."
    )
    job_result = _chat_completion_jobs(repositories, save_id)[-1]["result"]
    assert job_result["narrator_verifier"]["dating_route_stage_violation_count"] == 1
    assert job_result["narrator_verifier_retry_used"] is True


def test_submit_player_turn_allows_later_stage_commitment_language(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, _npc_id = _create_dating_chat_save(
        repositories,
        stage="exclusive",
    )
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_app_setting(
        RESPONSE_VERIFICATION_MODE_SETTING,
        RESPONSE_VERIFICATION_MODE_RETRY_ONCE,
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    provider = SequenceChatProvider(
        "fake",
        ("Mika squeezes Lio's hand and calls him her boyfriend.",),
    )
    spec = NarratorMessageSpec(
        intent="Answer Lio's affection.",
        thesis="The exclusive route supports direct affection.",
        must_say=(),
        avoid=(),
        tone="warm and grounded",
        uncertainties=(),
        evidence_source_ids=("dating_route_state:route-mika",),
        npc_intents=(
            NpcIntent(
                character_name="Mika Arai",
                character_id="mika",
                stance="committed girlfriend",
                current_goal="Reassure Lio.",
                next_action="Respond with established exclusive affection.",
                should_comply=True,
                route_stage="exclusive",
                max_plausible_escalation=(
                    "exclusive relationship language and intimacy consistent "
                    "with known boundaries"
                ),
            ),
        ),
    )
    verifier = ScriptedNarratorVerifier(_passing_verification())
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        narrator_planner=ScriptedNarratorPlanner(spec),
        narrator_verifier=verifier,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save_id,
            body="I tell Mika I am glad we are together.",
            speaker_name="Lio",
            run_post_turn_jobs=False,
        )
    )

    assert len(provider.chat_requests) == 1
    assert result.narrator_message.body == (
        "Mika squeezes Lio's hand and calls him her boyfriend."
    )


def test_submit_player_turn_profiles_dating_route_before_narrator_prompt(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, npc_id = _create_dating_chat_save(
        repositories,
        stage="introduced",
    )
    provider = DatingRouteProfileChatProvider("fake", npc_id=npc_id)
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-chat",
        display_name="Fake Chat",
        capabilities=[ProviderCapability.CHAT.value],
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-profile",
        display_name="Fake Profile",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task=DATING_ROUTE_PROFILE_TASK,
        provider="fake",
        model_id="fake-profile",
    )
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save_id,
            body="I ask Mika if she'd like to keep talking tonight.",
            speaker_name="Lio",
            run_post_turn_jobs=False,
        )
    )

    route = repositories.list_dating_route_states(save_id)[0]
    assert provider.events[:2] == ["dating_route_profile", "chat"]
    assert route.comfort_with_intimacy.startswith("open to physical intimacy")
    assert route.pacing_preference == "direct and chemistry-led"
    request = provider.chat_requests[0]
    route_context = "\n".join(request.current_scene_recap)
    assert "comfort with intimacy: open to physical intimacy" in route_context
    assert "pacing: direct and chemistry-led" in route_context
    assert "sexual escalation" not in route_context


def test_submit_player_turn_captures_profiled_route_state_in_player_snapshot(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, _npc_id = _create_dating_chat_save(
        repositories,
        stage="introduced",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-chat",
        display_name="Fake Chat",
        capabilities=[ProviderCapability.CHAT.value],
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )

    class MutatingProfileRunner:
        async def ensure_profiles_for_save(
            self,
            *,
            save_id: str,
            source_message_id: str | None = None,
        ) -> DatingRouteProfileResult:
            route = repositories.list_dating_route_states(save_id)[0]
            repositories.upsert_dating_route_state(
                save_id=save_id,
                player_character_id=route.player_character_id,
                npc_character_id=route.npc_character_id,
                stage=route.stage,
                comfort_with_intimacy="profiled before narrator prompt",
                pacing_preference="profiled pacing",
                source_message_id=source_message_id,
                route_id=route.id,
            )
            return DatingRouteProfileResult(
                status="succeeded",
                updated_count=1,
                requested_count=1,
            )

    service = ChatService(
        repositories=repositories,
        providers={"fake": RecordingChatProvider("fake")},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        dating_route_profile_service=MutatingProfileRunner(),
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save_id,
            body="I ask Mika if she wants to talk by the gate.",
            speaker_name="Lio",
            run_post_turn_jobs=False,
        )
    )

    snapshot = TurnSnapshotService(repositories).latest_snapshot_for_message(
        save_id=save_id,
        message_id=result.player_message.id,
    )
    route = repositories.list_dating_route_states(save_id)[0]
    assert snapshot is not None
    assert snapshot.reason == "pre_turn_dating_route_profile"
    assert route.comfort_with_intimacy == "profiled before narrator prompt"
    assert route.pacing_preference == "profiled pacing"


def test_submit_player_turn_keeps_dating_route_anchor_after_setup_ages_out(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Last Summer",
        premise=(
            "A long initial dating-sim setup with romance options and dorm "
            "politics that should age out of the normal scenario header."
        ),
        player_role="Transfer student",
        content={
            "player_character_name": "Lio Takahashi",
            "current_scene": "Lio and Mika talk after the festival.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Summer Save")
    player = repositories.add_character(
        save_id=save.id,
        name="Lio Takahashi",
        is_player_character=True,
        met=True,
    )
    npc = repositories.add_character(
        save_id=save.id,
        name="Mika Arai",
        relationships={player.name: "romance option for Lio Takahashi"},
        met=True,
    )
    repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Opening romance-option setup that has aged out.",
    )
    recent_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Mika waits by the lanterns.",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Lio and Mika linger near the festival gate.",
        present_character_ids=[npc.id],
        world_day_index=2,
        snapshot_id="snapshot-dating-aged-out",
    )
    repositories.upsert_dating_route_state(
        save_id=save.id,
        player_character_id=player.id,
        npc_character_id=npc.id,
        stage="introduced",
        first_met_world_day_index=2,
        completed_interactions=1,
        dates_completed=0,
        interest_level="curious",
        trust_level="guarded",
        comfort_with_intimacy="none yet",
        next_reasonable_step="exchange contact info or plan another interaction",
        route_id="route-mika",
    )
    repositories.set_app_setting(RECENT_NARRATOR_MESSAGE_WINDOW_SETTING, 1)
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingChatProvider("fake")
    planner = ScriptedNarratorPlanner(
        NarratorMessageSpec(
            intent="Answer Lio.",
            thesis="Mika can show early interest without commitment.",
            must_say=(),
            avoid=("Do not make Mika exclusive.",),
            tone="warm and grounded",
            uncertainties=(),
            evidence_source_ids=("dating_route_state:route-mika",),
        )
    )
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(
                selected_state=(
                    SelectedContextItem(
                        source_type="dating_route_state",
                        source_id="route-mika",
                        text="Stale duplicate route state.",
                        relevance_note="Selected by retrieval but deterministic.",
                    ),
                ),
            )
        ),
        narrator_planner=planner,
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I tell Mika I would like to keep talking after tonight.",
            speaker_name="Lio",
            run_post_turn_jobs=False,
        )
    )

    assert len(planner.calls) == 1
    planner_request = planner.calls[0][1]
    request = provider.chat_requests[0]
    for seen_request in (planner_request, request):
        route_context = "\n".join(seen_request.current_scene_recap)
        assert "Dating route pacing for Mika Arai with Lio Takahashi" in route_context
        assert "stage: introduced" in route_context
        assert "known for 0 in-world days" in route_context
        assert "interest: curious" in route_context
        assert "trust: guarded" in route_context
        assert "comfort with intimacy: none yet" in route_context
        assert "premature now: exclusivity or commitment language" in route_context
    assert "Premise/setup:" not in request.scenario_instructions
    assert "Opening romance-option setup that has aged out." not in [
        message.body for message in request.messages
    ]
    assert [
        message.body for message in request.messages if message.role == "narrator"
    ] == [recent_narrator.body]
    assert request.retrieved_state == ()
    assert request.context_breakdown["suppressed_duplicate_retrieval_keys"] == [
        "dating_route_state:route-mika"
    ]


def test_submit_player_turn_includes_present_and_mentioned_dating_routes_only(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Last Summer",
        premise="A summer of route choices.",
        player_role="Transfer student",
        content={"player_character_name": "Lio Takahashi"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Summer Save")
    player = repositories.add_character(
        save_id=save.id,
        name="Lio Takahashi",
        is_player_character=True,
        met=True,
    )
    mika = repositories.add_character(
        save_id=save.id,
        name="Mika Arai",
        relationships={player.name: "romance option for Lio Takahashi"},
        met=True,
    )
    hana = repositories.add_character(
        save_id=save.id,
        name="Hana Mori",
        aliases=["Hana"],
        relationships={player.name: "romance option for Lio Takahashi"},
        met=True,
    )
    yui = repositories.add_character(
        save_id=save.id,
        name="Yui Sato",
        relationships={player.name: "romance option for Lio Takahashi"},
        met=True,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Lio and Mika talk near the festival gate.",
        present_character_ids=[mika.id],
        world_day_index=3,
    )
    for npc, route_id, stage in (
        (mika, "route-mika", "contact_exchanged"),
        (hana, "route-hana", "introduced"),
        (yui, "route-yui", "introduced"),
    ):
        repositories.upsert_dating_route_state(
            save_id=save.id,
            player_character_id=player.id,
            npc_character_id=npc.id,
            stage=stage,
            first_met_world_day_index=1,
            completed_interactions=1,
            dates_completed=0,
            next_reasonable_step="build proportionate relationship progress",
            route_id=route_id,
        )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingChatProvider("fake")
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I ask Mika whether Hana might join us later.",
            speaker_name="Lio",
            run_post_turn_jobs=False,
        )
    )

    route_context = "\n".join(provider.chat_requests[0].current_scene_recap)
    assert "Dating route pacing for Mika Arai with Lio Takahashi" in route_context
    assert "Dating route pacing for Hana Mori with Lio Takahashi" in route_context
    assert "Dating route pacing for Yui Sato" not in route_context


def test_plan_first_verifier_receives_rich_reference_request(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_app_setting(PLAN_FIRST_NARRATOR_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    provider = SequenceChatProvider("fake", ("The lens burns red.",))
    spec = NarratorMessageSpec(
        intent="Answer the player's move.",
        thesis="The beacon warning must be visible.",
        must_say=("The lens burns red.",),
        avoid=(),
        tone="tense and grounded",
        uncertainties=(),
        evidence_source_ids=("observation:warning",),
    )
    verifier = ScriptedNarratorVerifier(
        NarratorVerificationResult(
            passed=True,
            issues=(),
            retry_feedback="",
            confidence=0.92,
        )
    )
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(
                selected_memories=(
                    SelectedContextItem(
                        source_type="memory",
                        source_id="memory-warning",
                        text="The red lens means riders are close.",
                        relevance_note="Verifier needs the hidden source context.",
                    ),
                ),
            )
        ),
        narrator_planner=ScriptedNarratorPlanner(spec),
        narrator_verifier=verifier,
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
        )
    )

    assert provider.chat_requests[0].narrator_prompt_mode == "plan_first"
    verifier_request = verifier.calls[0][1]
    assert verifier_request.narrator_prompt_mode == "rich_context"
    assert verifier_request.retrieved_memories == (
        "[memory:memory-warning] The red lens means riders are close.",
    )


def test_submit_player_turn_commits_rendered_planned_scene_presence(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "Mara and Lio watch the beacon lens."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    lio = repositories.add_character(save_id=save.id, name="Lio", met=True)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara and Lio stand in the beacon gallery.",
        objective="Keep the lens lit.",
        present_character_ids=[mara.id, lio.id],
    )
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    candidate = _scene_presence_candidate(lio.id, action="leave")
    spec = NarratorMessageSpec(
        intent="Answer the player move.",
        thesis="Lio leaves if rendered.",
        must_say=(),
        avoid=(),
        tone="grounded",
        uncertainties=(),
        evidence_source_ids=(),
        state_commit_candidates=(candidate,),
    )
    provider = SequenceChatProvider(
        "fake",
        ("Lio leaves the beacon gallery to fetch the archive map.",),
    )
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        narrator_planner=ScriptedNarratorPlanner(spec),
        narrator_verifier=ScriptedNarratorVerifier(
            _passing_verification(_commit_decision(candidate))
        ),
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I ask who should fetch the map.",
            speaker_name="Ily",
            run_post_turn_jobs=False,
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert set(snapshot.present_character_ids) == {mara.id}
    assert snapshot.situation == "Mara and Lio stand in the beacon gallery."
    assert snapshot.objective == "Keep the lens lit."
    assert snapshot.last_updated_message_id == result.narrator_message.id
    planned = _chat_completion_jobs(repositories, save.id)[-1]["result"][
        "planned_commits"
    ]
    assert planned["proposed_count"] == 1
    assert planned["committed_count"] == 1
    assert planned["skipped_count"] == 0


def test_submit_player_turn_skips_scene_presence_with_ungrounded_evidence(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "Mara and Lio watch the beacon lens."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    lio = repositories.add_character(save_id=save.id, name="Lio", met=True)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara and Lio stand in the beacon gallery.",
        present_character_ids=[mara.id, lio.id],
    )
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    candidate = _scene_presence_candidate(
        lio.id,
        action="leave",
        evidence_quote="ruby library",
    )
    spec = NarratorMessageSpec(
        intent="Answer the player move.",
        thesis="Lio leaves if rendered.",
        must_say=(),
        avoid=(),
        tone="grounded",
        uncertainties=(),
        evidence_source_ids=(),
        state_commit_candidates=(candidate,),
    )

    asyncio.run(
        ChatService(
            repositories=repositories,
            providers={
                "fake": SequenceChatProvider(
                    "fake",
                    ("Lio leaves the beacon gallery to fetch the archive map.",),
                )
            },
            context_search_service=ScriptedContextSearch(ContextSearchResult()),
            narrator_planner=ScriptedNarratorPlanner(spec),
            narrator_verifier=ScriptedNarratorVerifier(
                _passing_verification(_commit_decision(candidate))
            ),
        ).submit_player_turn(
            save_id=save.id,
            body="I ask who should fetch the map.",
            speaker_name="Ily",
            run_post_turn_jobs=False,
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert set(snapshot.present_character_ids) == {mara.id, lio.id}
    planned = _chat_completion_jobs(repositories, save.id)[-1]["result"][
        "planned_commits"
    ]
    assert planned["skipped_count"] == 1
    assert planned["decisions"][0]["reason"] == "ungrounded_evidence_metadata"


def test_submit_player_turn_commits_verified_scene_snapshot_field(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "Mara watches the beacon lens."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara stands by the beacon lens.",
        objective="Keep the lens lit.",
        mood="steady",
        present_character_ids=[mara.id],
    )
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    candidate = _scene_snapshot_field_candidate(
        field_path="mood",
        value="uneasy",
    )
    spec = NarratorMessageSpec(
        intent="Answer the player move.",
        thesis="The scene mood turns uneasy if rendered.",
        must_say=(),
        avoid=(),
        tone="grounded",
        uncertainties=(),
        evidence_source_ids=(),
        state_commit_candidates=(candidate,),
    )

    result = asyncio.run(
        ChatService(
            repositories=repositories,
            providers={
                "fake": SequenceChatProvider(
                    "fake",
                    ("The lens hums, and the room settles into uneasy quiet.",),
                )
            },
            context_search_service=ScriptedContextSearch(ContextSearchResult()),
            narrator_planner=ScriptedNarratorPlanner(spec),
            narrator_verifier=ScriptedNarratorVerifier(
                _passing_verification(_commit_decision(candidate))
            ),
        ).submit_player_turn(
            save_id=save.id,
            body="I listen to the beacon lens.",
            speaker_name="Ily",
            run_post_turn_jobs=False,
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.mood == "uneasy"
    assert snapshot.situation == "Mara stands by the beacon lens."
    assert snapshot.objective == "Keep the lens lit."
    assert snapshot.present_character_ids == [mara.id]
    assert snapshot.last_updated_message_id == result.narrator_message.id
    planned = _chat_completion_jobs(repositories, save.id)[-1]["result"][
        "planned_commits"
    ]
    assert planned["committed_count"] == 1
    assert planned["by_type"]["scene_snapshot_field"]["committed"] == 1


def test_submit_player_turn_commits_plan_grounded_in_assembled_context(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "Mara watches the beacon lens."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="The red beacon lens warns of riders in the ash.",
        mood="steady",
    )
    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    evidence_source_id = f"scene_snapshot:{snapshot.id}"
    candidate = replace(
        _scene_snapshot_field_candidate(
            field_path="mood",
            value="uneasy",
            evidence_quote="red beacon lens warns",
        ),
        evidence_source_ids=(evidence_source_id,),
    )
    spec = NarratorMessageSpec(
        intent="Answer the player move.",
        thesis="The warning makes the room uneasy.",
        must_say=(),
        avoid=(),
        tone="grounded",
        uncertainties=(),
        evidence_source_ids=(evidence_source_id,),
        state_commit_candidates=(candidate,),
        evidence_source_text_by_id={
            evidence_source_id: (
                "Scene snapshot: situation: The red beacon lens warns of riders "
                "in the ash.; mood: steady"
            )
        },
    )

    asyncio.run(
        ChatService(
            repositories=repositories,
            providers={
                "fake": SequenceChatProvider(
                    "fake",
                    ("Uneasy silence gathers around the warning light.",),
                )
            },
            context_search_service=ScriptedContextSearch(ContextSearchResult()),
            narrator_planner=ScriptedNarratorPlanner(spec),
            narrator_verifier=ScriptedNarratorVerifier(
                _passing_verification(_commit_decision(candidate))
            ),
        ).submit_player_turn(
            save_id=save.id,
            body="I study the warning.",
            speaker_name="Ily",
            run_post_turn_jobs=False,
        )
    )

    updated = repositories.get_scene_snapshot(save.id)
    assert updated is not None
    assert updated.mood == "uneasy"
    planned = _chat_completion_jobs(repositories, save.id)[-1]["result"][
        "planned_commits"
    ]
    assert planned["committed_count"] == 1
    assert planned["by_domain"]["scene_snapshot"]["committed"] == 1
    assert planned["rejected_count"] == 0


def test_submit_player_turn_syncs_canonical_time_for_planned_scene_time_field(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "Mara watches the beacon lens."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara stands by the beacon lens.",
        objective="Keep the lens lit.",
        in_world_time="Monday morning",
        time_of_day="morning",
        day_of_week="monday",
        world_time_clock_minutes=21 * 60 + 15,
        world_time_period_label="festival week",
    )
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    candidate = _scene_snapshot_field_candidate(
        field_path="time_of_day",
        value="night",
        candidate_id="scene_snapshot:time_of_day",
        evidence_quote="Night settles",
    )
    spec = NarratorMessageSpec(
        intent="Answer the player move.",
        thesis="Night settles if rendered.",
        must_say=(),
        avoid=(),
        tone="grounded",
        uncertainties=(),
        evidence_source_ids=(),
        state_commit_candidates=(candidate,),
    )

    result = asyncio.run(
        ChatService(
            repositories=repositories,
            providers={
                "fake": SequenceChatProvider(
                    "fake",
                    ("Night settles over the beacon lens.",),
                )
            },
            context_search_service=ScriptedContextSearch(ContextSearchResult()),
            narrator_planner=ScriptedNarratorPlanner(spec),
            narrator_verifier=ScriptedNarratorVerifier(
                _passing_verification(_commit_decision(candidate))
            ),
        ).submit_player_turn(
            save_id=save.id,
            body="I wait by the beacon lens.",
            speaker_name="Ily",
            run_post_turn_jobs=False,
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.in_world_time == "Monday festival week night at 21:15"
    assert snapshot.time_of_day == "night"
    assert snapshot.world_time_phase == "night"
    assert snapshot.world_time_day_label == "monday"
    assert snapshot.world_time_clock_minutes == 21 * 60 + 15
    assert snapshot.world_time_period_label == "festival week"
    assert snapshot.world_time_source_message_id == result.narrator_message.id


def test_submit_player_turn_preserves_detail_for_planned_in_world_time_field(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "Mara watches the beacon lens."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara stands by the beacon lens.",
        objective="Keep the lens lit.",
        in_world_time="Monday morning",
        time_of_day="morning",
        day_of_week="monday",
        world_time_clock_minutes=21 * 60 + 15,
        world_time_period_label="festival week",
    )
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    candidate = _scene_snapshot_field_candidate(
        field_path="in_world_time",
        value="night",
        candidate_id="scene_snapshot:in_world_time",
        evidence_quote="Night settles",
    )
    spec = NarratorMessageSpec(
        intent="Answer the player move.",
        thesis="Night settles if rendered.",
        must_say=(),
        avoid=(),
        tone="grounded",
        uncertainties=(),
        evidence_source_ids=(),
        state_commit_candidates=(candidate,),
    )

    asyncio.run(
        ChatService(
            repositories=repositories,
            providers={
                "fake": SequenceChatProvider(
                    "fake",
                    ("Night settles over the beacon lens.",),
                )
            },
            context_search_service=ScriptedContextSearch(ContextSearchResult()),
            narrator_planner=ScriptedNarratorPlanner(spec),
            narrator_verifier=ScriptedNarratorVerifier(
                _passing_verification(_commit_decision(candidate))
            ),
        ).submit_player_turn(
            save_id=save.id,
            body="I wait by the beacon lens.",
            speaker_name="Ily",
            run_post_turn_jobs=False,
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.in_world_time == "Monday festival week night at 21:15"
    assert snapshot.time_of_day == "night"
    assert snapshot.world_time_phase == "night"
    assert snapshot.world_time_clock_minutes == 21 * 60 + 15
    assert snapshot.world_time_period_label == "festival week"


def test_submit_player_turn_skips_planned_scene_time_when_canonical_field_locked(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "Mara watches the beacon lens."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara stands by the beacon lens.",
        objective="Keep the lens lit.",
        in_world_time="Monday morning",
        time_of_day="morning",
        day_of_week="monday",
        world_time_phase="morning",
        locked_fields=["world_time_phase"],
    )
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    candidate = _scene_snapshot_field_candidate(
        field_path="time_of_day",
        value="night",
        candidate_id="scene_snapshot:time_of_day",
        evidence_quote="Night settles",
    )
    spec = NarratorMessageSpec(
        intent="Answer the player move.",
        thesis="Night settles if rendered.",
        must_say=(),
        avoid=(),
        tone="grounded",
        uncertainties=(),
        evidence_source_ids=(),
        state_commit_candidates=(candidate,),
    )

    asyncio.run(
        ChatService(
            repositories=repositories,
            providers={
                "fake": SequenceChatProvider(
                    "fake",
                    ("Night settles over the beacon lens.",),
                )
            },
            context_search_service=ScriptedContextSearch(ContextSearchResult()),
            narrator_planner=ScriptedNarratorPlanner(spec),
            narrator_verifier=ScriptedNarratorVerifier(
                _passing_verification(_commit_decision(candidate))
            ),
        ).submit_player_turn(
            save_id=save.id,
            body="I wait by the beacon lens.",
            speaker_name="Ily",
            run_post_turn_jobs=False,
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.time_of_day == "morning"
    assert snapshot.world_time_phase == "morning"
    planned = _chat_completion_jobs(repositories, save.id)[-1]["result"][
        "planned_commits"
    ]
    assert planned["committed_count"] == 0
    assert planned["skipped_count"] == 1
    assert planned["decisions"][0]["reason"] == "locked_scene_snapshot_field"


def test_submit_player_turn_skips_scene_snapshot_field_with_ungrounded_evidence(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "Mara watches the beacon lens."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara stands by the beacon lens.",
        objective="Keep the lens lit.",
        mood="steady",
        present_character_ids=[mara.id],
    )
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    candidate = _scene_snapshot_field_candidate(
        field_path="mood",
        value="uneasy",
        evidence_quote="ruby library",
    )
    spec = NarratorMessageSpec(
        intent="Answer the player move.",
        thesis="The scene mood turns uneasy if rendered.",
        must_say=(),
        avoid=(),
        tone="grounded",
        uncertainties=(),
        evidence_source_ids=(),
        state_commit_candidates=(candidate,),
    )

    asyncio.run(
        ChatService(
            repositories=repositories,
            providers={
                "fake": SequenceChatProvider(
                    "fake",
                    ("The lens hums, and the room settles into uneasy quiet.",),
                )
            },
            context_search_service=ScriptedContextSearch(ContextSearchResult()),
            narrator_planner=ScriptedNarratorPlanner(spec),
            narrator_verifier=ScriptedNarratorVerifier(
                _passing_verification(_commit_decision(candidate))
            ),
        ).submit_player_turn(
            save_id=save.id,
            body="I listen to the beacon lens.",
            speaker_name="Ily",
            run_post_turn_jobs=False,
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.mood == "steady"
    planned = _chat_completion_jobs(repositories, save.id)[-1]["result"][
        "planned_commits"
    ]
    assert planned["skipped_count"] == 1
    assert planned["decisions"][0]["reason"] == "ungrounded_evidence_metadata"


def test_submit_player_turn_applies_world_time_before_prompt_context(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "Mara watches the beacon lens."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara stands by the beacon lens.",
        objective="Keep the lens lit.",
        in_world_time="Monday morning",
        time_of_day="morning",
        day_of_week="monday",
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    events: list[str] = []
    provider = RecordingChatProvider("fake")

    asyncio.run(
        ChatService(
            repositories=repositories,
            providers={"fake": provider},
            context_search_service=ScriptedContextSearch(
                ContextSearchResult(),
                events=events,
            ),
            world_time_service=ScriptedWorldTimeRunner(
                repositories,
                events=events,
            ),
        ).submit_player_turn(
            save_id=save.id,
            body="We wait until evening.",
            speaker_name="Ily",
            run_post_turn_jobs=False,
        )
    )

    assert events == ["world_time", "context_search"]
    assert provider.chat_requests
    assert "Current world time: Tuesday evening." in "\n".join(
        provider.chat_requests[0].current_scene_recap
    )
    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.time_of_day == "evening"
    assert snapshot.day_of_week == "tuesday"


@pytest.mark.parametrize(
    ("status", "expected_key"),
    (
        ("omitted", "skipped_count"),
        ("contradicted", "contradicted_count"),
    ),
)
def test_submit_player_turn_skips_unverified_planned_scene_presence(
    repositories: PersistenceRepositories,
    status: str,
    expected_key: str,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "Mara and Lio watch the beacon lens."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    lio = repositories.add_character(save_id=save.id, name="Lio", met=True)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara and Lio stand in the beacon gallery.",
        present_character_ids=[mara.id, lio.id],
    )
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    candidate = _scene_presence_candidate(lio.id, action="leave")
    spec = NarratorMessageSpec(
        intent="Answer the player move.",
        thesis="Lio might leave.",
        must_say=(),
        avoid=(),
        tone="grounded",
        uncertainties=(),
        evidence_source_ids=(),
        state_commit_candidates=(candidate,),
    )
    service = ChatService(
        repositories=repositories,
        providers={
            "fake": SequenceChatProvider(
                "fake",
                ("Lio stays at the beacon lens and keeps watching.",),
            )
        },
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        narrator_planner=ScriptedNarratorPlanner(spec),
        narrator_verifier=ScriptedNarratorVerifier(
            _passing_verification(
                _commit_decision(
                    candidate,
                    status=status,
                    safe_to_commit=False,
                    reason=f"Verifier marked the candidate {status}.",
                )
            )
        ),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I ask whether Lio should leave.",
            speaker_name="Ily",
            run_post_turn_jobs=False,
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert set(snapshot.present_character_ids) == {mara.id, lio.id}
    planned = _chat_completion_jobs(repositories, save.id)[-1]["result"][
        "planned_commits"
    ]
    assert planned["proposed_count"] == 1
    assert planned["committed_count"] == 0
    assert planned[expected_key] == 1
    assert planned["decisions"][0]["status"] == status


def test_submit_player_turn_skips_planned_commits_when_verifier_unavailable(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "Mara and Lio watch the beacon lens."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    lio = repositories.add_character(save_id=save.id, name="Lio", met=True)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        present_character_ids=[mara.id, lio.id],
    )
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    candidate = _scene_presence_candidate(lio.id, action="leave")
    spec = NarratorMessageSpec(
        intent="Answer the player move.",
        thesis="Lio leaves if rendered.",
        must_say=(),
        avoid=(),
        tone="grounded",
        uncertainties=(),
        evidence_source_ids=(),
        state_commit_candidates=(candidate,),
    )

    result = asyncio.run(
        ChatService(
            repositories=repositories,
            providers={"fake": SequenceChatProvider("fake", ("Lio leaves.",))},
            context_search_service=ScriptedContextSearch(ContextSearchResult()),
            narrator_planner=ScriptedNarratorPlanner(spec),
            narrator_verifier=ScriptedNarratorVerifier(
                RuntimeError("verifier unavailable")
            ),
        ).submit_player_turn(
            save_id=save.id,
            body="I ask whether Lio should leave.",
            speaker_name="Ily",
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert set(snapshot.present_character_ids) == {mara.id, lio.id}
    assert result.narrator_message.body == "Lio leaves."
    assert [
        job.status
        for job in repositories.list_jobs_by_status(("succeeded",))
        if job.save_id == save.id and job.type == "post_turn_jobs"
    ] == ["succeeded"]
    planned = _chat_completion_jobs(repositories, save.id)[-1]["result"][
        "planned_commits"
    ]
    assert planned["skipped_count"] == 1
    assert planned["decisions"][0]["application_status"] == "skipped"
    assert planned["decisions"][0]["reason"] == "verifier_unavailable"


def test_submit_player_turn_commits_planned_effect_after_retry_success(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "Mara and Lio watch the beacon lens."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    lio = repositories.add_character(save_id=save.id, name="Lio", met=True)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        present_character_ids=[mara.id, lio.id],
    )
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_app_setting(
        RESPONSE_VERIFICATION_MODE_SETTING,
        RESPONSE_VERIFICATION_MODE_RETRY_ONCE,
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    candidate = _scene_presence_candidate(lio.id, action="leave")
    spec = NarratorMessageSpec(
        intent="Answer the player move.",
        thesis="Lio leaves if rendered.",
        must_say=("Lio leaves.",),
        avoid=(),
        tone="grounded",
        uncertainties=(),
        evidence_source_ids=(),
        state_commit_candidates=(candidate,),
    )
    provider = SequenceChatProvider(
        "fake",
        (
            "Lio studies the lens in silence.",
            "Lio leaves the beacon gallery to fetch the archive map.",
        ),
    )
    verifier = ScriptedNarratorVerifier(
        (
            NarratorVerificationResult(
                passed=False,
                issues=("Missed Lio leaving.",),
                retry_feedback="Write Lio leaving the gallery.",
                confidence=0.88,
                commit_decisions=(
                    _commit_decision(
                        candidate,
                        status="omitted",
                        safe_to_commit=False,
                        reason="The first draft does not show Lio leaving.",
                    ),
                ),
            ),
            _passing_verification(_commit_decision(candidate)),
        )
    )

    result = asyncio.run(
        ChatService(
            repositories=repositories,
            providers={"fake": provider},
            context_search_service=ScriptedContextSearch(ContextSearchResult()),
            narrator_planner=ScriptedNarratorPlanner(spec),
            narrator_verifier=verifier,
        ).submit_player_turn(
            save_id=save.id,
            body="I ask Lio to fetch the map.",
            speaker_name="Ily",
            run_post_turn_jobs=False,
        )
    )

    assert result.narrator_message.body == (
        "Lio leaves the beacon gallery to fetch the archive map."
    )
    assert len(verifier.calls) == 2
    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert set(snapshot.present_character_ids) == {mara.id}
    planned = _chat_completion_jobs(repositories, save.id)[-1]["result"][
        "planned_commits"
    ]
    assert planned["committed_count"] == 1
    assert planned["decisions"][0]["status"] == "rendered"


def test_character_assessment_scene_presence_candidate_uses_presence_evidence(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    repositories.upsert_scene_snapshot(save_id=save.id, present_character_ids=[])

    candidates = chat_service_module._character_assessment_commit_candidates(
        repositories=repositories,
        save_id=save.id,
        result=CharacterActionPlanningResult(
            decisions=(
                CharacterTurnAssessment(
                    character_id=mara.id,
                    character_name="Mara",
                    present=True,
                    enters_scene=True,
                    action="Mara checks the corridor.",
                    intent="inspect the corridor",
                    reason="Mara enters only if presence evidence is grounded.",
                    confidence=0.84,
                    evidence_source_ids=("message:intent",),
                    evidence_quote="intent quote",
                    presence_evidence_source_ids=("message:presence",),
                    presence_evidence_quote="presence quote",
                ),
            )
        ),
        include_presence=True,
    )

    assert len(candidates) == 1
    assert candidates[0].evidence_source_ids == ("message:presence",)
    assert candidates[0].evidence_quote == "presence quote"
    assert candidates[0].value["evidence_quote"] == "presence quote"


def test_submit_player_turn_queues_verified_learned_memory_when_confirmation_enabled(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "Mara watches the beacon lens."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_app_setting("manual_confirmation_memories_enabled", True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    candidate = _learned_memory_candidate(
        mara.id,
        body="Mara learned that ember dawn wakes the beacon.",
    )
    candidate = replace(
        candidate,
        value={
            key: item_value
            for key, item_value in candidate.value.items()
            if key != "evidence_quote"
        },
    )
    spec = NarratorMessageSpec(
        intent="Answer the player move.",
        thesis="Mara learns the phrase if rendered.",
        must_say=(),
        avoid=(),
        tone="grounded",
        uncertainties=(),
        evidence_source_ids=(),
        state_commit_candidates=(candidate,),
    )

    asyncio.run(
        ChatService(
            repositories=repositories,
            providers={
                "fake": SequenceChatProvider(
                    "fake",
                    ("Mara repeats that ember dawn wakes the beacon.",),
                )
            },
            context_search_service=ScriptedContextSearch(ContextSearchResult()),
            narrator_planner=ScriptedNarratorPlanner(spec),
            narrator_verifier=ScriptedNarratorVerifier(
                _passing_verification(_commit_decision(candidate))
            ),
        ).submit_player_turn(
            save_id=save.id,
            body="I tell Mara the beacon phrase.",
            speaker_name="Ily",
            run_post_turn_jobs=False,
        )
    )

    assert repositories.list_memories(save.id) == []
    suggestions = repositories.list_context_update_suggestions(save.id)
    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert suggestion.entity_type == "memory"
    assert suggestion.update_type == "create"
    assert suggestion.status == "pending"
    proposed_value = cast(dict[str, object], suggestion.proposed_value)
    assert proposed_value["body"] == (
        "Mara learned that ember dawn wakes the beacon."
    )
    assert proposed_value["character_id"] == mara.id
    planned = _chat_completion_jobs(repositories, save.id)[-1]["result"][
        "planned_commits"
    ]
    assert planned["confirmation_queued_count"] == 1
    assert planned["committed_count"] == 0
    assert planned["coverage"]["memory_count"] == 0
    assert planned["coverage"]["applied_domains"] == []
    assert planned["coverage"]["queued_domains"] == ["memories"]


def test_submit_player_turn_skips_learned_memory_with_ungrounded_evidence(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "Mara watches the beacon lens."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    base_invalid_quote = _learned_memory_candidate(
        mara.id,
        body="Mara learned an unsupported library clue.",
        candidate_id="character_learned_memory:mara:bad_quote",
    )
    invalid_quote = replace(
        base_invalid_quote,
        value={
            **base_invalid_quote.value,
            "body": "Mara learned an unsupported library clue.",
            "evidence_quote": "ruby library",
        },
    )
    unknown_source = replace(
        _learned_memory_candidate(
            mara.id,
            body="Mara learned from a missing source.",
            candidate_id="character_learned_memory:mara:missing_source",
        ),
        evidence_source_ids=("message:missing",),
    )
    spec = NarratorMessageSpec(
        intent="Answer the player move.",
        thesis="Mara learns the phrase if rendered.",
        must_say=(),
        avoid=(),
        tone="grounded",
        uncertainties=(),
        evidence_source_ids=(),
        state_commit_candidates=(invalid_quote, unknown_source),
    )

    asyncio.run(
        ChatService(
            repositories=repositories,
            providers={
                "fake": SequenceChatProvider(
                    "fake",
                    ("Mara repeats that ember dawn wakes the beacon.",),
                )
            },
            context_search_service=ScriptedContextSearch(ContextSearchResult()),
            narrator_planner=ScriptedNarratorPlanner(spec),
            narrator_verifier=ScriptedNarratorVerifier(
                _passing_verification(
                    _commit_decision(invalid_quote),
                    _commit_decision(unknown_source),
                )
            ),
        ).submit_player_turn(
            save_id=save.id,
            body="I tell Mara the beacon phrase.",
            speaker_name="Ily",
            run_post_turn_jobs=False,
        )
    )

    assert repositories.list_memories(save.id) == []
    assert repositories.list_context_update_suggestions(save.id) == []
    planned = _chat_completion_jobs(repositories, save.id)[-1]["result"][
        "planned_commits"
    ]
    assert planned["skipped_count"] == 2
    assert {
        decision["reason"] for decision in planned["decisions"]
    } == {"ungrounded_evidence_metadata"}
    assert planned["rejected_count"] == 2
    assert planned["by_reason"]["ungrounded_evidence_metadata"] == 2
    assert planned["by_domain"]["memories"]["rejected"] == 2


def test_hybrid_post_turn_mode_does_not_duplicate_verified_planned_memory(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "Mara watches the beacon lens."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=POST_TURN_INFERENCE_MODE_SETTING,
        value="hybrid",
    )
    for task, model_id in (
        ("chat", "fake-chat"),
        ("state_memory", "fake-state-memory"),
    ):
        repositories.save_provider_model(
            provider="fake",
            model_id=model_id,
            display_name=model_id,
            capabilities=["chat"] if task == "chat" else ["structured_output"],
            context_window=8192,
        )
        repositories.set_model_preference(
            task=task,
            provider="fake",
            model_id=model_id,
        )
    memory_body = "Mara learned that ember dawn wakes the beacon."
    candidate = _learned_memory_candidate(mara.id, body=memory_body)
    spec = NarratorMessageSpec(
        intent="Answer the player move.",
        thesis="Mara learns the phrase if rendered.",
        must_say=(),
        avoid=(),
        tone="grounded",
        uncertainties=(),
        evidence_source_ids=(),
        state_commit_candidates=(candidate,),
    )
    events: list[str] = []
    provider = ScriptedPostTurnStructuredProvider(
        "fake",
        response_bodies=("Mara repeats that ember dawn wakes the beacon.",),
        events=events,
        state_data={
            "state_changes": [],
            "memories": [
                {
                    "body": memory_body,
                    "tags": ["mara", "beacon"],
                    "importance": 0.86,
                    "evidence_quote": "ember dawn wakes the beacon",
                }
            ],
        },
    )

    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        narrator_planner=ScriptedNarratorPlanner(spec),
        narrator_verifier=ScriptedNarratorVerifier(
            _passing_verification(_commit_decision(candidate))
        ),
    )

    async def submit_and_run_post_turn_jobs() -> None:
        turn = await service.submit_player_turn(
            save_id=save.id,
            body="I tell Mara the beacon phrase.",
            speaker_name="Ily",
            run_post_turn_jobs=False,
        )
        await service.run_post_turn_jobs(
            save_id=save.id,
            player_message_id=turn.player_message.id,
            narrator_message_id=turn.narrator_message.id,
        )

    asyncio.run(submit_and_run_post_turn_jobs())

    assert events.count("state_memory_extraction") == 1
    memories = repositories.list_memories(save.id)
    assert [memory.body for memory in memories] == [memory_body]
    coordinator = _post_turn_jobs(repositories, save.id)[-1]
    assert coordinator["payload"]["post_turn_inference_mode"] == "hybrid"
    assert coordinator["result"]["post_turn_inference_mode"] == "hybrid"
    assert _post_turn_child_status(coordinator, "state") == "narrowed"
    state_result = _post_turn_child_result(coordinator, "state")
    assert state_result["suppressed_memory_count"] == 1
    assert state_result["verified_plan_coverage"]["memory_count"] == 1


def test_plan_owned_post_turn_mode_keeps_hybrid_fallback_for_partial_coverage(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "Mara watches the beacon lens."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara watches the beacon lens.",
        present_character_ids=[mara.id],
    )
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=POST_TURN_INFERENCE_MODE_SETTING,
        value="plan_owned",
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=DIRECTOR_PRESSURE_ENABLED_SETTING,
        value=False,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=CHARACTER_ACTION_PLANNING_ENABLED_SETTING,
        value=False,
    )
    for task, model_id in (
        ("chat", "fake-chat"),
        ("state_memory", "fake-state-memory"),
        ("context_update", "fake-context-update"),
        ("scenario_evolution", "fake-scenario-evolution"),
    ):
        repositories.save_provider_model(
            provider="fake",
            model_id=model_id,
            display_name=model_id,
            capabilities=["chat"] if task == "chat" else ["structured_output"],
            context_window=8192,
        )
        repositories.set_model_preference(
            task=task,
            provider="fake",
            model_id=model_id,
        )
    candidate = _scene_snapshot_field_candidate(
        field_path="mood",
        value="uneasy",
    )
    spec = NarratorMessageSpec(
        intent="Answer the player move.",
        thesis="The room turns uneasy.",
        must_say=(),
        avoid=(),
        tone="grounded",
        uncertainties=(),
        evidence_source_ids=(),
        state_commit_candidates=(candidate,),
    )
    events: list[str] = []
    provider = ScriptedPostTurnStructuredProvider(
        "fake",
        response_bodies=("The room settles into uneasy quiet.",),
        events=events,
        state_data={
            "state_changes": [
                {
                    "operation": "upsert",
                    "key": "legacy.should_not_run",
                    "value": {"status": "bad"},
                    "category": "debug",
                    "confidence": 0.8,
                    "evidence_quote": "uneasy quiet",
                }
            ],
            "memories": [],
        },
    )
    media_service = RecordingMediaService(
        repositories=repositories,
        events=events,
    )

    asyncio.run(
        ChatService(
            repositories=repositories,
            providers={"fake": provider},
            context_search_service=ScriptedContextSearch(ContextSearchResult()),
            media_service=media_service,
            narrator_planner=ScriptedNarratorPlanner(spec),
            narrator_verifier=ScriptedNarratorVerifier(
                NarratorVerificationResult(
                    passed=True,
                    issues=(),
                    retry_feedback="",
                    confidence=0.92,
                    post_turn_update_needed=False,
                    commit_decisions=(_commit_decision(candidate),),
                )
            ),
        ).submit_player_turn(
            save_id=save.id,
            body="I listen to the beacon lens.",
            speaker_name="Ily",
        )
    )

    assert "state_memory_extraction" in events
    assert "context_update" in events
    assert "scenario_evolution" in events
    assert "automatic_media" in events
    assert [state.key for state in repositories.list_world_state(save.id)] == [
        "legacy.should_not_run"
    ]
    coordinator = _post_turn_jobs(repositories, save.id)[-1]
    assert coordinator["payload"]["post_turn_inference_mode"] == "plan_owned"
    assert coordinator["payload"]["effective_post_turn_inference_mode"] == "hybrid"
    assert coordinator["result"]["post_turn_inference_mode_reason"] == (
        "plan_owned_partial_domain_fallback"
    )
    coverage = coordinator["result"]["verified_plan_coverage"]
    assert coverage["applied_domains"] == ["scene_snapshot"]
    assert _post_turn_child_status(coordinator, "state") == "succeeded"
    assert _post_turn_child_status(coordinator, "context") == "succeeded"


def test_plan_owned_post_turn_mode_falls_back_when_commits_still_need_updates(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "Mara watches the beacon lens."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara watches the beacon lens.",
        present_character_ids=[mara.id],
    )
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=POST_TURN_INFERENCE_MODE_SETTING,
        value="plan_owned",
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=DIRECTOR_PRESSURE_ENABLED_SETTING,
        value=False,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=CHARACTER_ACTION_PLANNING_ENABLED_SETTING,
        value=False,
    )
    for task, model_id in (
        ("chat", "fake-chat"),
        ("state_memory", "fake-state-memory"),
        ("context_update", "fake-context-update"),
    ):
        repositories.save_provider_model(
            provider="fake",
            model_id=model_id,
            display_name=model_id,
            capabilities=["chat"] if task == "chat" else ["structured_output"],
            context_window=8192,
        )
        repositories.set_model_preference(
            task=task,
            provider="fake",
            model_id=model_id,
        )
    candidate = _scene_snapshot_field_candidate(
        field_path="mood",
        value="uneasy",
    )
    spec = NarratorMessageSpec(
        intent="Answer the player move.",
        thesis="The room turns uneasy.",
        must_say=(),
        avoid=(),
        tone="grounded",
        uncertainties=(),
        evidence_source_ids=(),
        state_commit_candidates=(candidate,),
    )
    events: list[str] = []
    provider = ScriptedPostTurnStructuredProvider(
        "fake",
        response_bodies=("The room settles into uneasy quiet.",),
        events=events,
        state_data={
            "state_changes": [
                {
                    "operation": "upsert",
                    "key": "fallback.repaired_state",
                    "value": {"status": "ran"},
                    "category": "debug",
                    "confidence": 0.8,
                    "evidence_quote": "uneasy quiet",
                }
            ],
            "memories": [],
        },
    )

    asyncio.run(
        ChatService(
            repositories=repositories,
            providers={"fake": provider},
            context_search_service=ScriptedContextSearch(ContextSearchResult()),
            narrator_planner=ScriptedNarratorPlanner(spec),
            narrator_verifier=ScriptedNarratorVerifier(
                NarratorVerificationResult(
                    passed=True,
                    issues=(),
                    retry_feedback="",
                    confidence=0.92,
                    post_turn_update_needed=True,
                    commit_decisions=(_commit_decision(candidate),),
                )
            ),
        ).submit_player_turn(
            save_id=save.id,
            body="I listen to the beacon lens.",
            speaker_name="Ily",
        )
    )

    assert "state_memory_extraction" in events
    assert "context_update" in events
    assert [state.key for state in repositories.list_world_state(save.id)] == [
        "fallback.repaired_state"
    ]
    coordinator = _post_turn_jobs(repositories, save.id)[-1]
    assert coordinator["payload"]["post_turn_inference_mode"] == "plan_owned"
    assert coordinator["payload"]["effective_post_turn_inference_mode"] == "hybrid"
    assert coordinator["result"]["effective_post_turn_inference_mode"] == "hybrid"
    assert coordinator["result"]["post_turn_inference_mode_reason"] == (
        "plan_owned_safety_fallback"
    )
    assert _post_turn_child_status(coordinator, "state") == "succeeded"
    assert _post_turn_child_status(coordinator, "context") == "succeeded"
    coverage = coordinator["result"]["verified_plan_coverage"]
    assert coverage["committed_count"] == 1
    assert coverage["planned_commit_post_turn_update_needed"] is True


def test_plan_owned_post_turn_mode_skips_legacy_inference_for_verified_noop(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "Mara watches the beacon lens."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=POST_TURN_INFERENCE_MODE_SETTING,
        value="plan_owned",
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=DIRECTOR_PRESSURE_ENABLED_SETTING,
        value=False,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=CHARACTER_ACTION_PLANNING_ENABLED_SETTING,
        value=False,
    )
    for task, model_id in (
        ("chat", "fake-chat"),
        ("state_memory", "fake-state-memory"),
        ("context_update", "fake-context-update"),
    ):
        repositories.save_provider_model(
            provider="fake",
            model_id=model_id,
            display_name=model_id,
            capabilities=["chat"] if task == "chat" else ["structured_output"],
            context_window=8192,
        )
        repositories.set_model_preference(
            task=task,
            provider="fake",
            model_id=model_id,
        )
    spec = NarratorMessageSpec(
        intent="Answer the player move.",
        thesis="The room stays stable.",
        must_say=(),
        avoid=(),
        tone="grounded",
        uncertainties=(),
        evidence_source_ids=(),
        state_commit_candidates=(),
    )
    events: list[str] = []
    provider = ScriptedPostTurnStructuredProvider(
        "fake",
        response_bodies=("The room stays quiet.",),
        events=events,
        state_data={
            "state_changes": [
                {
                    "operation": "upsert",
                    "key": "legacy.should_not_run",
                    "value": {"status": "bad"},
                    "category": "debug",
                    "confidence": 0.8,
                    "evidence_quote": "legacy",
                }
            ],
            "memories": [],
        },
    )

    asyncio.run(
        ChatService(
            repositories=repositories,
            providers={"fake": provider},
            context_search_service=ScriptedContextSearch(ContextSearchResult()),
            narrator_planner=ScriptedNarratorPlanner(spec),
            narrator_verifier=ScriptedNarratorVerifier(
                NarratorVerificationResult(
                    passed=True,
                    issues=(),
                    retry_feedback="",
                    confidence=0.92,
                    post_turn_update_needed=False,
                )
            ),
        ).submit_player_turn(
            save_id=save.id,
            body="I listen to the beacon lens.",
            speaker_name="Ily",
        )
    )

    assert "state_memory_extraction" not in events
    assert "context_update" not in events
    assert repositories.list_world_state(save.id) == []
    coordinator = _post_turn_jobs(repositories, save.id)[-1]
    assert coordinator["payload"]["effective_post_turn_inference_mode"] == (
        "plan_owned"
    )
    assert _post_turn_child_status(coordinator, "state") == "skipped"
    assert _post_turn_child_status(coordinator, "context") == "skipped"
    coverage = coordinator["result"]["verified_plan_coverage"]
    assert coverage["planned_commit_post_turn_update_needed"] is False


def test_plan_owned_post_turn_mode_falls_back_to_hybrid_for_weak_plan_coverage(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "Mara watches the beacon lens."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara watches the beacon lens.",
        present_character_ids=[mara.id],
    )
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=POST_TURN_INFERENCE_MODE_SETTING,
        value="plan_owned",
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=DIRECTOR_PRESSURE_ENABLED_SETTING,
        value=False,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=CHARACTER_ACTION_PLANNING_ENABLED_SETTING,
        value=False,
    )
    for task, model_id in (
        ("chat", "fake-chat"),
        ("state_memory", "fake-state-memory"),
        ("context_update", "fake-context-update"),
    ):
        repositories.save_provider_model(
            provider="fake",
            model_id=model_id,
            display_name=model_id,
            capabilities=["chat"] if task == "chat" else ["structured_output"],
            context_window=8192,
        )
        repositories.set_model_preference(
            task=task,
            provider="fake",
            model_id=model_id,
        )
    candidate = _scene_snapshot_field_candidate(
        field_path="current_round.status.timer",
        value="1:00",
        candidate_id="scene_snapshot:round_timer",
    )
    spec = NarratorMessageSpec(
        intent="Answer the player move.",
        thesis="The round timer advances.",
        must_say=(),
        avoid=(),
        tone="grounded",
        uncertainties=(),
        evidence_source_ids=(),
        state_commit_candidates=(candidate,),
    )
    events: list[str] = []
    provider = ScriptedPostTurnStructuredProvider(
        "fake",
        response_bodies=("The room settles as the timer reaches one minute.",),
        events=events,
        state_data={
            "state_changes": [
                {
                    "operation": "upsert",
                    "key": "fallback.repaired_state",
                    "value": {"status": "ran"},
                    "category": "debug",
                    "confidence": 0.8,
                    "evidence_quote": "timer reaches one minute",
                }
            ],
            "memories": [],
        },
    )

    asyncio.run(
        ChatService(
            repositories=repositories,
            providers={"fake": provider},
            context_search_service=ScriptedContextSearch(ContextSearchResult()),
            narrator_planner=ScriptedNarratorPlanner(spec),
            narrator_verifier=ScriptedNarratorVerifier(
                _passing_verification(_commit_decision(candidate))
            ),
        ).submit_player_turn(
            save_id=save.id,
            body="I listen to the beacon lens.",
            speaker_name="Ily",
        )
    )

    assert "state_memory_extraction" in events
    assert "context_update" in events
    assert [state.key for state in repositories.list_world_state(save.id)] == [
        "fallback.repaired_state"
    ]
    planned = _chat_completion_jobs(repositories, save.id)[-1]["result"][
        "planned_commits"
    ]
    assert planned["skipped_count"] == 1
    coordinator = _post_turn_jobs(repositories, save.id)[-1]
    assert coordinator["payload"]["post_turn_inference_mode"] == "plan_owned"
    assert coordinator["payload"]["effective_post_turn_inference_mode"] == "hybrid"
    assert coordinator["result"]["post_turn_inference_mode"] == "plan_owned"
    assert coordinator["result"]["effective_post_turn_inference_mode"] == "hybrid"
    assert coordinator["result"]["post_turn_inference_mode_reason"] == (
        "plan_owned_safety_fallback"
    )
    assert _post_turn_child_status(coordinator, "state") == "succeeded"


def test_submit_player_turn_creates_verified_character_knowledge_edge(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "Mara watches the beacon lens."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara watches the beacon lens.",
        present_character_ids=[mara.id],
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="The beacon answers to ember dawn.",
        tags=["beacon"],
        memory_id="memory-beacon-key",
    )
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    candidate = _knowledge_edge_candidate(
        mara.id,
        target_type="memory",
        target_id=memory.id,
    )
    candidate = replace(
        candidate,
        value={
            key: item_value
            for key, item_value in candidate.value.items()
            if key != "evidence_quote"
        },
    )
    spec = NarratorMessageSpec(
        intent="Answer the player move.",
        thesis="Mara may connect the phrase to existing evidence.",
        must_say=(),
        avoid=(),
        tone="grounded",
        uncertainties=(),
        evidence_source_ids=(),
        state_commit_candidates=(candidate,),
    )

    result = asyncio.run(
        ChatService(
            repositories=repositories,
            providers={
                "fake": SequenceChatProvider(
                    "fake",
                    ("Mara nods at ember dawn, filing the beacon phrase away.",),
                )
            },
            context_search_service=ScriptedContextSearch(ContextSearchResult()),
            narrator_planner=ScriptedNarratorPlanner(spec),
            narrator_verifier=ScriptedNarratorVerifier(
                _passing_verification(
                    _commit_decision(
                        candidate,
                        status="safe_without_narration",
                        reason="The linked edge is metadata for a rendered fact.",
                    )
                )
            ),
        ).submit_player_turn(
            save_id=save.id,
            body="I tell Mara the beacon phrase.",
            speaker_name="Ily",
            run_post_turn_jobs=False,
        )
    )

    edges = repositories.list_character_knowledge_edges(save.id)
    assert len(edges) == 1
    assert edges[0].character_id == mara.id
    assert edges[0].target_type == "memory"
    assert edges[0].target_id == memory.id
    assert edges[0].source_message_id == result.narrator_message.id
    planned = _chat_completion_jobs(repositories, save.id)[-1]["result"][
        "planned_commits"
    ]
    assert planned["committed_count"] == 1
    assert planned["by_type"]["character_knowledge_edge"]["committed"] == 1


def test_submit_player_turn_skips_knowledge_edge_with_ungrounded_evidence(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "Mara watches the beacon lens."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara watches the beacon lens.",
        present_character_ids=[mara.id],
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="The beacon answers to ember dawn.",
        tags=["beacon"],
        memory_id="memory-beacon-key",
    )
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    base = _knowledge_edge_candidate(
        mara.id,
        target_type="memory",
        target_id=memory.id,
    )
    invalid_quote = replace(
        base,
        candidate_id="character_knowledge_edge:mara:bad_quote",
        value={**base.value, "evidence_quote": "ruby library"},
    )
    unknown_source = replace(
        _knowledge_edge_candidate(
            mara.id,
            target_type="memory",
            target_id=memory.id,
            candidate_id="character_knowledge_edge:mara:missing_source",
        ),
        evidence_source_ids=("message:missing",),
    )
    spec = NarratorMessageSpec(
        intent="Answer the player move.",
        thesis="Mara may connect the phrase to existing evidence.",
        must_say=(),
        avoid=(),
        tone="grounded",
        uncertainties=(),
        evidence_source_ids=(),
        state_commit_candidates=(invalid_quote, unknown_source),
    )

    asyncio.run(
        ChatService(
            repositories=repositories,
            providers={
                "fake": SequenceChatProvider(
                    "fake",
                    ("Mara nods at the ember dawn phrase.",),
                )
            },
            context_search_service=ScriptedContextSearch(ContextSearchResult()),
            narrator_planner=ScriptedNarratorPlanner(spec),
            narrator_verifier=ScriptedNarratorVerifier(
                _passing_verification(
                    _commit_decision(
                        invalid_quote,
                        status="safe_without_narration",
                    ),
                    _commit_decision(
                        unknown_source,
                        status="safe_without_narration",
                    ),
                )
            ),
        ).submit_player_turn(
            save_id=save.id,
            body="I tell Mara the beacon phrase.",
            speaker_name="Ily",
            run_post_turn_jobs=False,
        )
    )

    assert repositories.list_character_knowledge_edges(save.id) == []
    planned = _chat_completion_jobs(repositories, save.id)[-1]["result"][
        "planned_commits"
    ]
    assert planned["skipped_count"] == 2
    assert {
        decision["reason"] for decision in planned["decisions"]
    } == {"ungrounded_evidence_metadata"}


def test_submit_player_turn_skips_authoritative_knowledge_edge_for_absent_character(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "Mara watches the beacon lens."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    lio = repositories.add_character(save_id=save.id, name="Lio", met=True)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara watches the beacon lens while Lio is elsewhere.",
        present_character_ids=[mara.id],
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="The beacon answers to ember dawn.",
        tags=["beacon"],
        memory_id="memory-beacon-key",
    )
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    candidate = _knowledge_edge_candidate(
        lio.id,
        target_type="memory",
        target_id=memory.id,
        candidate_id="character_knowledge_edge:lio:memory",
    )
    candidate = StateCommitCandidate(
        operation=candidate.operation,
        state_key=candidate.state_key,
        value={
            **candidate.value,
            "knowledge_state": "knows",
            "acquisition_method": "witnessed",
        },
        reason=candidate.reason,
        confidence=candidate.confidence,
        evidence_source_ids=candidate.evidence_source_ids,
        evidence_quote=candidate.evidence_quote,
        candidate_id=candidate.candidate_id,
        candidate_type=candidate.candidate_type,
        field_path=candidate.field_path,
        character_id=candidate.character_id,
        target_type=candidate.target_type,
        target_id=candidate.target_id,
        safe_without_narration_allowed=candidate.safe_without_narration_allowed,
    )
    spec = NarratorMessageSpec(
        intent="Answer the player move.",
        thesis="Only present characters should gain witnessed knowledge.",
        must_say=(),
        avoid=(),
        tone="grounded",
        uncertainties=(),
        evidence_source_ids=(),
        state_commit_candidates=(candidate,),
    )

    asyncio.run(
        ChatService(
            repositories=repositories,
            providers={
                "fake": SequenceChatProvider(
                    "fake",
                    ("Mara nods at the ember dawn phrase.",),
                )
            },
            context_search_service=ScriptedContextSearch(ContextSearchResult()),
            narrator_planner=ScriptedNarratorPlanner(spec),
            narrator_verifier=ScriptedNarratorVerifier(
                _passing_verification(
                    _commit_decision(
                        candidate,
                        status="safe_without_narration",
                        reason="The edge was judged metadata-only.",
                    )
                )
            ),
        ).submit_player_turn(
            save_id=save.id,
            body="I tell Mara the beacon phrase.",
            speaker_name="Ily",
            run_post_turn_jobs=False,
        )
    )

    assert repositories.list_character_knowledge_edges(save.id) == []
    planned = _chat_completion_jobs(repositories, save.id)[-1]["result"][
        "planned_commits"
    ]
    assert planned["skipped_count"] == 1
    assert planned["decisions"][0]["reason"] == (
        "character_not_present_for_authoritative_knowledge_edge"
    )


def test_submit_player_turn_creates_verified_scenario_section_knowledge_edge(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "starting_scene": "Mara watches the beacon lens.",
            "lore": "The old keep bells only ring for ash-signed oaths.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    candidate = _knowledge_edge_candidate(
        mara.id,
        target_type="scenario_section",
        target_id="lore",
        candidate_id="character_knowledge_edge:mara:lore",
    )
    candidate = replace(
        candidate,
        value={**candidate.value, "evidence_quote": "ash-signed oaths"},
    )
    spec = NarratorMessageSpec(
        intent="Answer the player move.",
        thesis="Mara may connect the oath to existing scenario lore.",
        must_say=(),
        avoid=(),
        tone="grounded",
        uncertainties=(),
        evidence_source_ids=(),
        state_commit_candidates=(candidate,),
    )

    result = asyncio.run(
        ChatService(
            repositories=repositories,
            providers={
                "fake": SequenceChatProvider(
                    "fake",
                    ("Mara lowers her voice at the mention of ash-signed oaths.",),
                )
            },
            context_search_service=ScriptedContextSearch(ContextSearchResult()),
            narrator_planner=ScriptedNarratorPlanner(spec),
            narrator_verifier=ScriptedNarratorVerifier(
                _passing_verification(
                    _commit_decision(
                        candidate,
                        status="safe_without_narration",
                        reason="The edge links rendered knowledge to scenario lore.",
                    )
                )
            ),
        ).submit_player_turn(
            save_id=save.id,
            body="I tell Mara the oath phrase.",
            speaker_name="Ily",
            run_post_turn_jobs=False,
        )
    )

    edges = repositories.list_character_knowledge_edges(save.id)
    assert len(edges) == 1
    assert edges[0].character_id == mara.id
    assert edges[0].target_type == "scenario_section"
    assert edges[0].target_id == "lore"
    assert edges[0].source_message_id == result.narrator_message.id
    planned = _chat_completion_jobs(repositories, save.id)[-1]["result"][
        "planned_commits"
    ]
    assert planned["committed_count"] == 1
    assert planned["by_type"]["character_knowledge_edge"]["committed"] == 1


def test_plan_first_character_presence_remains_tentative_without_verification(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"player_character_name": "Ily"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    mara = repositories.add_character(
        save_id=save.id,
        name="Mara",
        met=True,
        is_player_character=True,
    )
    lio = repositories.add_character(save_id=save.id, name="Lio", met=True)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        present_character_ids=[mara.id, lio.id],
    )
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_app_setting(PLAN_FIRST_NARRATOR_SETTING, True)
    repositories.set_app_setting(CHARACTER_ACTION_PLANNING_ENABLED_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="fake",
        model_id="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-chat",
        display_name="Fake Chat",
        capabilities=["chat", "structured_output"],
    )
    provider = RecordingCharacterActionChatProvider(
        "fake",
        {
            "Lio": {
                "present": False,
                "action": "Lio leaves the beacon gallery.",
                "intent": "fetch the archive map",
                "reason": "Lio should leave only if narrated.",
                "confidence": 0.85,
                "evidence_source_ids": ["message:latest"],
                "leaves_scene": True,
            }
        },
    )
    planner = ScriptedNarratorPlanner(
        NarratorMessageSpec(
            intent="Answer the player move.",
            thesis="Lio leaves only if the accepted response renders it.",
            must_say=(),
            avoid=(),
            tone="grounded",
            uncertainties=(),
            evidence_source_ids=(),
        )
    )

    asyncio.run(
        ChatService(
            repositories=repositories,
            providers={"fake": provider},
            context_search_service=ScriptedContextSearch(ContextSearchResult()),
            narrator_planner=planner,
        ).submit_player_turn(
            save_id=save.id,
            body="I ask Lio for the archive map.",
            speaker_name="Ily",
            run_post_turn_jobs=False,
        )
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert set(snapshot.present_character_ids) == {mara.id, lio.id}
    assert provider.chat_requests[0].narrator_prompt_mode == "plan_first"
    planned = _chat_completion_jobs(repositories, save.id)[-1]["result"][
        "planned_commits"
    ]
    assert planned["proposed_count"] == 1
    assert planned["skipped_count"] == 1
    assert planned["decisions"][0]["reason"] == "verifier_unavailable"


def test_submit_player_turn_audits_final_response_after_generic_verifier_retry(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "Nira arrives at the beacon stair."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_app_setting(
        RESPONSE_VERIFICATION_MODE_SETTING,
        RESPONSE_VERIFICATION_MODE_RETRY_ONCE,
    )
    repositories.add_character(save_id=save.id, name="Nira", met=True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    provider = SequenceChatProvider(
        "fake",
        (
            "The stair is quiet.",
            "Nira says, \"The lens burns red above the stair.\"",
        ),
    )
    spec = NarratorMessageSpec(
        intent="Answer the player's move.",
        thesis="The beacon warning must be visible.",
        must_say=("The lens burns red.",),
        avoid=(),
        tone="tense and grounded",
        uncertainties=(),
        evidence_source_ids=("observation:warning",),
    )
    verifier = ScriptedNarratorVerifier(
        NarratorVerificationResult(
            passed=False,
            issues=("Missed the red lens warning.",),
            retry_feedback="Mention the red lens warning before ending the beat.",
            confidence=0.92,
        )
    )
    auditor = ScriptedNpcKnowledgeAuditor((NpcKnowledgeAuditResult(enabled=True),))
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        narrator_planner=ScriptedNarratorPlanner(spec),
        narrator_verifier=verifier,
        npc_knowledge_audit_service=auditor,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert result.narrator_message.body == (
        "Nira says, \"The lens burns red above the stair.\""
    )
    assert auditor.calls == [(save.id, result.narrator_message.body)]


def test_submit_player_turn_uses_legacy_audit_when_second_verifier_fails(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "Nira arrives at the beacon stair."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_app_setting(
        RESPONSE_VERIFICATION_MODE_SETTING,
        RESPONSE_VERIFICATION_MODE_RETRY_ONCE,
    )
    repositories.add_character(save_id=save.id, name="Nira", met=True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    provider = SequenceChatProvider(
        "fake",
        (
            "Nira says, \"The stair is quiet.\"",
            "Nira says, \"The lens burns red above the stair.\"",
        ),
    )
    spec = NarratorMessageSpec(
        intent="Answer the player's move.",
        thesis="The beacon warning must be visible.",
        must_say=("The lens burns red.",),
        avoid=(),
        tone="tense and grounded",
        uncertainties=(),
        evidence_source_ids=("observation:warning",),
    )
    verifier = ScriptedNarratorVerifier(
        (
            NarratorVerificationResult(
                passed=False,
                issues=("Missed the red lens warning.",),
                retry_feedback="Mention the red lens warning before ending the beat.",
                confidence=0.92,
            ),
            RuntimeError("verifier unavailable on retry"),
        )
    )
    auditor = ScriptedNpcKnowledgeAuditor((NpcKnowledgeAuditResult(enabled=True),))
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        narrator_planner=ScriptedNarratorPlanner(spec),
        narrator_verifier=verifier,
        npc_knowledge_audit_service=auditor,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert len(verifier.calls) == 2
    assert result.narrator_message.body == (
        "Nira says, \"The lens burns red above the stair.\""
    )
    assert auditor.calls == [(save.id, result.narrator_message.body)]


def test_submit_player_turn_agentic_verifier_npc_leak_retries_and_skips_legacy_auditor(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Archive Arrival",
        premise="An archive scene with uneven knowledge.",
        player_role="Avery",
        content={"starting_scene": "Nira arrives after the earlier archive-code joke."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Boundary Test")
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.add_character(save_id=save.id, name="Nira", met=True)
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/generic-chat",
    )
    provider = SequenceChatProvider(
        "openrouter",
        (
            "Nira says, \"You made the archive-code joke within five minutes.\"",
            "Nira says, \"I missed the beginning, so I am catching up.\"",
        ),
    )
    leak = NpcKnowledgeLeak(
        speaker_name="Nira",
        claim="Avery made the archive-code joke within five minutes.",
        reason="Nira was not present for that exchange.",
    )
    spec = NarratorMessageSpec(
        intent="Answer the player's move.",
        thesis="Nira should only know what she witnessed.",
        must_say=(),
        avoid=(),
        tone="grounded",
        uncertainties=(),
        evidence_source_ids=(),
    )
    verifier = ScriptedNarratorVerifier(
        (
            NarratorVerificationResult(
                passed=True,
                npc_knowledge_leaks=(leak,),
                retry_feedback="",
                confidence=0.91,
            ),
            NarratorVerificationResult(
                passed=True,
                npc_knowledge_leaks=(),
                retry_feedback="",
                confidence=0.93,
            ),
        )
    )
    legacy_auditor = ScriptedNpcKnowledgeAuditor(
        (NpcKnowledgeAuditResult(enabled=True, leaks=(leak,)),)
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        narrator_planner=ScriptedNarratorPlanner(spec),
        narrator_verifier=verifier,
        npc_knowledge_audit_service=legacy_auditor,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I ask Nira if she wants to enter the chart room.",
            speaker_name="Avery",
        )
    )

    assert len(provider.chat_requests) == 2
    assert len(verifier.calls) == 2
    assert legacy_auditor.calls == []
    assert "NPC knowledge leak" in provider.chat_requests[1].regeneration_feedback
    assert result.narrator_message.body == (
        "Nira says, \"I missed the beginning, so I am catching up.\""
    )
    succeeded_chat_jobs = [
        job
        for job in repositories.list_jobs_by_status(("succeeded",))
        if job.save_id == save.id and job.type == "chat_completion"
    ]
    chat_result = succeeded_chat_jobs[0].result
    assert chat_result is not None
    audit_diagnostics = cast(dict[str, Any], chat_result["npc_knowledge_audit"])
    first_audit = cast(dict[str, Any], audit_diagnostics["first"])
    second_audit = cast(dict[str, Any], audit_diagnostics["second"])
    assert audit_diagnostics["source"] == "narrator_verifier"
    assert first_audit["leak_count"] == 1
    assert second_audit["leak_count"] == 0
    assert audit_diagnostics["auto_retry_used"] is True
    assert audit_diagnostics["suspicious"] is False


def test_submit_player_turn_agentic_verifier_hard_fails_when_retry_still_leaks(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Archive Arrival",
        premise="An archive scene with uneven knowledge.",
        player_role="Avery",
        content={"starting_scene": "Nira arrives after the earlier archive-code joke."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Boundary Test")
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=NPC_KNOWLEDGE_AUDIT_MODE_SETTING,
        value=NPC_KNOWLEDGE_AUDIT_MODE_HARD_FAIL,
    )
    repositories.add_character(save_id=save.id, name="Nira", met=True)
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/generic-chat",
    )
    provider = SequenceChatProvider(
        "openrouter",
        (
            "Nira says, \"You made the archive-code joke within five minutes.\"",
            "Nira says, \"Still, the archive-code joke happened immediately.\"",
        ),
    )
    leak = NpcKnowledgeLeak(
        speaker_name="Nira",
        claim="Nira references the archive-code joke.",
        reason="No knowledge edge lets Nira know the joke.",
    )
    spec = NarratorMessageSpec(
        intent="Answer the player's move.",
        thesis="Nira should only know what she witnessed.",
        must_say=(),
        avoid=(),
        tone="grounded",
        uncertainties=(),
        evidence_source_ids=(),
    )
    verifier = ScriptedNarratorVerifier(
        (
            NarratorVerificationResult(
                passed=True,
                npc_knowledge_leaks=(leak,),
                confidence=0.91,
            ),
            NarratorVerificationResult(
                passed=True,
                npc_knowledge_leaks=(leak,),
                confidence=0.93,
            ),
        )
    )
    legacy_auditor = ScriptedNpcKnowledgeAuditor(
        (NpcKnowledgeAuditResult(enabled=True),)
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        narrator_planner=ScriptedNarratorPlanner(spec),
        narrator_verifier=verifier,
        npc_knowledge_audit_service=legacy_auditor,
    )

    with pytest.raises(ValueError, match="NPC knowledge audit"):
        asyncio.run(
            service.submit_player_turn(
                save_id=save.id,
                body="I ask Nira if she wants to enter the chart room.",
                speaker_name="Avery",
            )
        )

    assert len(provider.chat_requests) == 2
    assert len(verifier.calls) == 2
    assert legacy_auditor.calls == []
    assert [
        message.body
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    ] == []


def test_submit_player_turn_falls_back_to_legacy_npc_audit_when_verifier_unavailable(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Archive Arrival",
        premise="An archive scene with uneven knowledge.",
        player_role="Avery",
        content={"starting_scene": "Nira arrives after the earlier archive-code joke."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Boundary Test")
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.add_character(save_id=save.id, name="Nira", met=True)
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/generic-chat",
    )
    provider = SequenceChatProvider(
        "openrouter",
        (
            "Nira says, \"You made the archive-code joke within five minutes.\"",
            "Nira says, \"I missed the beginning, so I am catching up.\"",
        ),
    )
    leak = NpcKnowledgeLeak(
        speaker_name="Nira",
        claim="Avery made the archive-code joke within five minutes.",
        reason="Nira was not present for that exchange.",
    )
    auditor = ScriptedNpcKnowledgeAuditor(
        (
            NpcKnowledgeAuditResult(enabled=True, leaks=(leak,)),
            NpcKnowledgeAuditResult(enabled=True),
        )
    )
    spec = NarratorMessageSpec(
        intent="Answer the player's move.",
        thesis="Nira should only know what she witnessed.",
        must_say=(),
        avoid=(),
        tone="grounded",
        uncertainties=(),
        evidence_source_ids=(),
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        narrator_planner=ScriptedNarratorPlanner(spec),
        npc_knowledge_audit_service=auditor,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I ask Nira if she wants to enter the chart room.",
            speaker_name="Avery",
        )
    )

    assert len(provider.chat_requests) == 2
    assert [body for _save_id, body in auditor.calls] == [
        "Nira says, \"You made the archive-code joke within five minutes.\"",
        "Nira says, \"I missed the beginning, so I am catching up.\"",
    ]
    assert result.narrator_message.body == (
        "Nira says, \"I missed the beginning, so I am catching up.\""
    )


def test_submit_player_turn_includes_pending_context_suggestions_without_applying(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    source_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The storm around the beacon seems wary now.",
        provider="fake",
        model="fake-chat",
    )
    suggestion = repositories.add_context_update_suggestion(
        save_id=save.id,
        update_type="update",
        entity_type="world_state",
        entity_id="state-storm-mood",
        field_path="storm.mood",
        proposed_value={"mood": "wary"},
        reason="The narrator described the storm as wary.",
        confidence=0.91,
        source_message_ids=[source_message.id],
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I ask what changed in the storm.",
            speaker_name="Mara",
        )
    )

    request = provider.chat_requests[0]
    assert request.pending_context_suggestions == (
        "Pending review (not canon yet): update world_state/state-storm-mood "
        'storm.mood -> {"mood": "wary"}; confidence=91%',
    )
    assert source_message.id not in "\n".join(request.pending_context_suggestions)
    assert "The narrator described the storm as wary" not in "\n".join(
        request.pending_context_suggestions
    )
    assert repositories.list_world_state(save.id) == []
    assert repositories.list_context_update_suggestions(save.id, status="pending") == [
        suggestion
    ]
    assert any(
        source["tier"] == "pending_context_suggestions"
        and source["source_type"] == "context_update_suggestion"
        and source["source_id"] == suggestion.id
        and source["included"] is True
        for source in request.context_breakdown["sources"]
    )


def test_submit_timeskip_turn_persists_system_request_without_player_message(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    context_search = ScriptedContextSearch(ContextSearchResult())
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=context_search,
    )

    result = asyncio.run(
        service.submit_timeskip_turn(
            save_id=save.id,
            instruction="Skip to dawn when the patrol reaches the city gates.",
            run_post_turn_jobs=False,
        )
    )

    persisted_messages = repositories.list_messages(save.id)
    assert [message.role for message in persisted_messages] == ["system", "narrator"]
    timeskip_message = persisted_messages[0]
    assert result.player_message == timeskip_message
    assert timeskip_message.speaker_name == "Timeskip"
    assert timeskip_message.body == (
        "Timeskip request: Skip to dawn when the patrol reaches the city gates."
    )
    assert context_search.calls == [(save.id, timeskip_message.id)]
    assert len(provider.chat_requests) == 1
    request = provider.chat_requests[0]
    assert request.turn_directive == (
        "Timeskip request: Skip to dawn when the patrol reaches the city gates."
    )
    assert request.messages[-1] == ChatMessage(
        role="system",
        speaker_name="Timeskip",
        body=timeskip_message.body,
    )
    assert result.narrator_message.body == (
        f"openrouter narrator: {timeskip_message.body}"
    )


def test_submit_timeskip_turn_runs_post_turn_jobs_with_system_source_message(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    post_turn_calls: list[tuple[str, str, str]] = []

    class RecordingPostTurnChatService(ChatService):
        async def run_post_turn_jobs(
            self,
            *,
            save_id: str,
            player_message_id: str,
            narrator_message_id: str,
            **_kwargs: object,
        ) -> None:
            post_turn_calls.append((save_id, player_message_id, narrator_message_id))

    service = RecordingPostTurnChatService(
        repositories=repositories,
        providers={"openrouter": RecordingChatProvider("openrouter")},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    result = asyncio.run(
        service.submit_timeskip_turn(
            save_id=save.id,
            instruction="Advance until the ferry reaches the far bank.",
        )
    )

    assert post_turn_calls == [
        (save.id, result.player_message.id, result.narrator_message.id)
    ]
    assert result.player_message.role == "system"


def test_submit_player_turn_streams_narrator_drafts_and_persists_final_body(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(SCRIPT_GUARD_MODE_SETTING, SCRIPT_GUARD_MODE_OFF)
    repositories.set_scoped_setting(
        scope="global",
        key=GENERATED_PHRASE_DENYLIST_SETTING,
        value="",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = StreamingChatProvider(
        "openrouter",
        (
            ChatStreamChunk(delta="The beacon"),
            ChatStreamChunk(delta=" answers.", token_usage={"total": 12}),
            ChatStreamChunk(token_usage={"total": 12}, done=True),
        ),
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )
    drafts: list[str] = []

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I strike the lens.",
            narrator_stream_callback=drafts.append,
        )
    )

    assert drafts == ["The beacon answers."]
    assert provider.stream_requests
    assert provider.chat_requests == []
    persisted_messages = repositories.list_messages(save.id)
    assert persisted_messages[1].id == result.narrator_message.id
    assert persisted_messages[1].body == "The beacon answers."
    assert persisted_messages[1].token_estimate == 12


def test_rated_streaming_never_publishes_draft_rejected_by_safety_agent(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(SCRIPT_GUARD_MODE_SETTING, SCRIPT_GUARD_MODE_OFF)
    repositories.set_scoped_setting(
        scope="global",
        key=GENERATED_PHRASE_DENYLIST_SETTING,
        value="",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="streaming-model",
    )
    _configure_content_safety_model(repositories)
    rejected_draft = "A lingering explicit description."
    narrator_provider = StreamingChatProvider(
        "openrouter",
        (
            ChatStreamChunk(delta="A lingering explicit"),
            ChatStreamChunk(
                delta=" description.",
                token_usage={"total": 12},
            ),
            ChatStreamChunk(token_usage={"total": 12}, done=True),
        ),
    )
    service = ChatService(
        repositories=repositories,
        providers={
            "openrouter": narrator_provider,
            "safety": ScriptedContentSafetyProvider("block"),
        },
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )
    drafts: list[str] = []

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I close the tower door.",
            run_post_turn_jobs=False,
            narrator_stream_callback=drafts.append,
        )
    )

    assert drafts == [CONTENT_FILTER_TRANSITION]
    assert rejected_draft not in repr(drafts)
    assert result.narrator_message.body == CONTENT_FILTER_TRANSITION
    assert result.narrator_message.content_rating == "g"


def test_submit_player_turn_fades_explicit_narrator_body_before_observers(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(SCRIPT_GUARD_MODE_SETTING, SCRIPT_GUARD_MODE_OFF)
    repositories.set_scoped_setting(
        scope="global",
        key=GENERATED_PHRASE_DENYLIST_SETTING,
        value="",
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    _configure_content_safety_model(repositories)
    rejected_draft = "He thrust into her as she cried out."
    prompt_store = PromptInspectionStore()
    debug_requests: list[ChatRequest] = []

    def capture_debug_prompt(*, message_id: str, request: ChatRequest) -> None:
        del message_id
        debug_requests.append(request)

    provider = StaticChatProvider("fake", rejected_draft)
    safety_provider = ScriptedContentSafetyProvider("fade_to_black")
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider, "safety": safety_provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        prompt_inspection_store=prompt_store,
        debug_prompt_capture=capture_debug_prompt,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I close the tower door.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
        )
    )

    assert result.narrator_message.body == FADE_TO_BLACK_TRANSITION
    assert result.narrator_message.safety_transition == "fade_to_black"
    assert result.narrator_message.content_rating == "g"
    assert rejected_draft not in repr(repositories.list_messages(save.id))
    assert rejected_draft not in repr(prompt_store.prompts_by_message_id())
    assert rejected_draft not in repr(debug_requests)
    job = _chat_completion_jobs(repositories, save.id)[0]
    assert job["status"] == "succeeded"
    assert job["result"]["classification"] == "content_safety_transition_applied"
    assert job["result"]["content_safety"] == {
        "action": "fade_to_black",
        "minimum_rating": "r",
        "category": "sexual_content",
        "transition_applied": True,
        "agent_ran": True,
        "skipped_reason": "",
        "provider": "safety",
        "model": "safety-model",
    }


def test_submit_player_turn_honors_adult_rating_and_disabled_fade_classifier(
    repositories: PersistenceRepositories,
) -> None:
    adult = repositories.create_user(
        username="Mira",
        role="user",
        password_hash="hash",
    )
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_scoped_setting(
        scope="user",
        scope_id=adult.id,
        key=CONTENT_FILTER_RATING_SETTING,
        value="r",
    )
    repositories.set_scoped_setting(
        scope="user",
        scope_id=adult.id,
        key=FADE_TO_BLACK_ENABLED_SETTING,
        value=False,
    )
    repositories.set_app_setting(SCRIPT_GUARD_MODE_SETTING, SCRIPT_GUARD_MODE_OFF)
    repositories.set_scoped_setting(
        scope="global",
        key=GENERATED_PHRASE_DENYLIST_SETTING,
        value="",
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    _configure_content_safety_model(repositories)
    body = "They had sex after returning to the inn."
    provider = StaticChatProvider("fake", body)
    safety_provider = ScriptedContentSafetyProvider("allow")
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider, "safety": safety_provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I close the tower door.",
            current_user_id=adult.id,
            run_post_turn_jobs=False,
        )
    )

    assert provider.chat_requests[0].content_rating == "r"
    assert provider.chat_requests[0].fade_to_black_enabled is False
    assert len(safety_provider.structured_output_requests) == 2
    assert result.narrator_message.body == body
    assert result.narrator_message.safety_transition == ""


def test_submit_player_turn_persists_safety_agent_rating_for_player_input(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    _configure_content_safety_model(repositories)
    provider = StaticChatProvider("fake", "The beacon answers.")
    safety_provider = ScriptedContentSafetyProvider("allow")

    asyncio.run(
        ChatService(
            repositories=repositories,
            providers={"fake": provider, "safety": safety_provider},
            context_search_service=ScriptedContextSearch(ContextSearchResult()),
        ).submit_player_turn(
            save_id=save.id,
            body="I close the tower door.",
            run_post_turn_jobs=False,
        )
    )

    [player_message, narrator_message] = repositories.list_messages(save.id)
    assert player_message.content_rating == "g"
    assert narrator_message.content_rating == "g"
    assert len(safety_provider.structured_output_requests) == 2


def test_submit_player_turn_does_not_override_safety_agent_allow_with_regexes(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(SCRIPT_GUARD_MODE_SETTING, SCRIPT_GUARD_MODE_OFF)
    repositories.set_scoped_setting(
        scope="global",
        key=GENERATED_PHRASE_DENYLIST_SETTING,
        value="",
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    _configure_content_safety_model(repositories)
    body = "The safety agent, not an isolated word match, owns this decision."
    provider = StaticChatProvider("fake", body)
    safety_provider = ScriptedContentSafetyProvider("allow")

    result = asyncio.run(
        ChatService(
            repositories=repositories,
            providers={"fake": provider, "safety": safety_provider},
            context_search_service=ScriptedContextSearch(ContextSearchResult()),
        ).submit_player_turn(
            save_id=save.id,
            body="I close the tower door.",
            run_post_turn_jobs=False,
        )
    )

    assert result.narrator_message.body == body
    assert result.narrator_message.safety_transition == ""


def test_nonsexual_content_filter_transition_is_skipped_by_post_turn_jobs(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(SCRIPT_GUARD_MODE_SETTING, SCRIPT_GUARD_MODE_OFF)
    repositories.set_scoped_setting(
        scope="global",
        key=GENERATED_PHRASE_DENYLIST_SETTING,
        value="",
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    _configure_content_safety_model(repositories)
    provider = StaticChatProvider(
        "fake",
        "The blast dismembered the soldier in graphic detail.",
    )
    safety_provider = ScriptedContentSafetyProvider("block")
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider, "safety": safety_provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I shield my eyes from the blast.",
        )
    )

    assert result.narrator_message.body == CONTENT_FILTER_TRANSITION
    assert result.narrator_message.safety_transition == "content_filter"
    assert len(provider.chat_requests) == 1
    coordinator = _post_turn_jobs(repositories, save.id)[-1]
    assert _post_turn_child_result(coordinator, "proactive_text") == {
        "status": "skipped",
        "reason": "safety_transition",
    }


def test_submit_player_turn_streams_only_fade_transition_for_rejected_body(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(SCRIPT_GUARD_MODE_SETTING, SCRIPT_GUARD_MODE_OFF)
    repositories.set_scoped_setting(
        scope="global",
        key=GENERATED_PHRASE_DENYLIST_SETTING,
        value="",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    _configure_content_safety_model(repositories)
    provider = SequenceStreamingChatProvider(
        "openrouter",
        (
            (
                ChatStreamChunk(delta="He thrust into "),
                ChatStreamChunk(
                    delta="her as she cried out.",
                    token_usage={"total": 12},
                ),
                ChatStreamChunk(token_usage={"total": 12}, done=True),
            ),
        ),
    )
    service = ChatService(
        repositories=repositories,
        providers={
            "openrouter": provider,
            "safety": ScriptedContentSafetyProvider("fade_to_black"),
        },
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )
    drafts: list[str] = []

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I close the tower door.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
            narrator_stream_callback=drafts.append,
        )
    )

    assert drafts == [FADE_TO_BLACK_TRANSITION]
    assert result.narrator_message.body == FADE_TO_BLACK_TRANSITION
    assert "thrust" not in repr(drafts)


def test_submit_player_turn_falls_back_when_stream_fails_before_text(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(SCRIPT_GUARD_MODE_SETTING, SCRIPT_GUARD_MODE_OFF)
    repositories.set_scoped_setting(
        scope="global",
        key=GENERATED_PHRASE_DENYLIST_SETTING,
        value="",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = StreamingChatProvider(
        "openrouter",
        (ProviderError(ProviderErrorCategory.NETWORK_ERROR, "stream broke"),),
        fallback_body="Fallback narrator response.",
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )
    drafts: list[str] = []

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I strike the lens.",
            narrator_stream_callback=drafts.append,
        )
    )

    assert drafts == ["Fallback narrator response."]
    assert len(provider.stream_requests) == 1
    assert len(provider.chat_requests) == 1
    assert result.narrator_message.body == "Fallback narrator response."


def test_submit_player_turn_uses_configured_fallback_after_streaming_retry_failure(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(SCRIPT_GUARD_MODE_SETTING, SCRIPT_GUARD_MODE_OFF)
    repositories.set_scoped_setting(
        scope="global",
        key=GENERATED_PHRASE_DENYLIST_SETTING,
        value="",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    repositories.set_model_preference(
        task="chat_fallback",
        provider="venice",
        model_id="venice/fallback-chat",
    )
    repositories.save_provider_model(
        provider="venice",
        model_id="venice/fallback-chat",
        display_name="Venice Fallback Chat",
        capabilities=["chat", "fallback_marker"],
        context_window=8192,
    )
    repositories.set_app_setting("chat_fallback_enabled", True)
    _configure_content_safety_model(repositories)
    primary = StreamingChatProvider(
        "openrouter",
        (ProviderError(ProviderErrorCategory.NETWORK_ERROR, "stream broke"),),
        fallback_error=ProviderError(
            ProviderErrorCategory.CONTENT_BLOCKED,
            "primary retry blocked",
        ),
    )
    fallback = StaticChatProvider(
        "venice",
        "He thrust into her as she cried out.",
    )
    service = ChatService(
        repositories=repositories,
        providers={
            "openrouter": primary,
            "venice": fallback,
            "safety": ScriptedContentSafetyProvider("fade_to_black"),
        },
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )
    drafts: list[str] = []

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I strike the lens.",
            narrator_stream_callback=drafts.append,
        )
    )

    assert drafts == [FADE_TO_BLACK_TRANSITION]
    assert len(primary.stream_requests) == 1
    assert len(primary.chat_requests) == 1
    assert len(fallback.chat_requests) == 1
    assert result.narrator_message.body == FADE_TO_BLACK_TRANSITION
    assert result.narrator_message.provider == "venice"
    assert result.narrator_message.model == "venice/fallback-chat"
    jobs = _chat_completion_jobs(repositories, save.id)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["status"] == "succeeded"
    assert job["result"]["original_provider"] == "openrouter"
    assert job["result"]["original_model"] == "anthropic/claude-3.5-sonnet"
    assert job["result"]["streaming_retry_used"] is True
    assert job["result"]["primary_error_category"] == "content_blocked"
    assert job["result"]["fallback_used"] is True
    assert job["result"]["fallback_provider"] == "venice"
    assert job["result"]["fallback_model"] == "venice/fallback-chat"
    assert job["result"]["final_provider"] == "venice"
    assert job["result"]["final_model"] == "venice/fallback-chat"
    assert job["result"]["classification"] == "content_safety_transition_applied"


def test_submit_player_turn_records_streaming_retry_failure_without_fallback(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(SCRIPT_GUARD_MODE_SETTING, SCRIPT_GUARD_MODE_OFF)
    repositories.set_scoped_setting(
        scope="global",
        key=GENERATED_PHRASE_DENYLIST_SETTING,
        value="",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    primary = StreamingChatProvider(
        "openrouter",
        (ProviderError(ProviderErrorCategory.NETWORK_ERROR, "stream broke"),),
        fallback_error=ProviderError(
            ProviderErrorCategory.CONTENT_BLOCKED,
            "primary retry blocked",
        ),
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": primary},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )
    drafts: list[str] = []

    with pytest.raises(Exception, match="primary retry blocked"):
        asyncio.run(
            service.submit_player_turn(
                save_id=save.id,
                body="I strike the lens.",
                narrator_stream_callback=drafts.append,
            )
        )

    assert drafts == []
    assert len(primary.stream_requests) == 1
    assert len(primary.chat_requests) == 1
    persisted_messages = repositories.list_messages(save.id)
    assert [message.role for message in persisted_messages] == ["player"]
    jobs = _chat_completion_jobs(repositories, save.id)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["status"] == "failed"
    assert job["result"]["original_provider"] == "openrouter"
    assert job["result"]["original_model"] == "anthropic/claude-3.5-sonnet"
    assert job["result"]["streaming_retry_used"] is True
    assert job["result"]["primary_error_category"] == "content_blocked"
    assert job["result"]["fallback_used"] is False
    assert job["result"]["fallback_skipped_reason"] == "no_fallback_model"
    assert job["result"]["classification"] == "suspected_blocked_output"
    assert job["error"] is not None
    assert "primary retry blocked" in job["error"]


def test_submit_player_turn_does_not_persist_partial_narrator_after_stream_failure(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(SCRIPT_GUARD_MODE_SETTING, SCRIPT_GUARD_MODE_OFF)
    repositories.set_scoped_setting(
        scope="global",
        key=GENERATED_PHRASE_DENYLIST_SETTING,
        value="",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = StreamingChatProvider(
        "openrouter",
        (
            ChatStreamChunk(delta="The beacon"),
            ProviderError(ProviderErrorCategory.NETWORK_ERROR, "stream broke"),
        ),
        fallback_body="Different fallback response.",
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )
    drafts: list[str] = []

    with pytest.raises(Exception, match="stream broke"):
        asyncio.run(
            service.submit_player_turn(
                save_id=save.id,
                body="I strike the lens.",
                narrator_stream_callback=drafts.append,
            )
        )

    assert drafts == []
    assert provider.chat_requests == []
    persisted_messages = repositories.list_messages(save.id)
    assert [message.role for message in persisted_messages] == ["player"]


def test_submit_player_turn_reuses_fresh_context_search_continuity_sync(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    sync_calls: list[str] = []
    original_sync_save = chat_service_module.ContinuityIndexService.sync_save

    def recording_sync_save(
        self: chat_service_module.ContinuityIndexService,
        save_id: str,
    ) -> object:
        sync_calls.append(save_id)
        return original_sync_save(self, save_id)

    monkeypatch.setattr(
        chat_service_module.ContinuityIndexService,
        "sync_save",
        recording_sync_save,
    )

    class SyncingContextSearch(ScriptedContextSearch):
        async def search(
            self,
            *,
            save_id: str,
            player_message_id: str,
        ) -> ContextSearchResult:
            chat_service_module.ContinuityIndexService(repositories).sync_save(
                save_id
            )
            return ContextSearchResult(continuity_index_synced=True)

    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=SyncingContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
        )
    )

    assert sync_calls == [save.id]


def test_submit_player_turn_syncs_continuity_for_unsynced_context_result(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    sync_calls: list[str] = []
    original_sync_save = chat_service_module.ContinuityIndexService.sync_save

    def recording_sync_save(
        self: chat_service_module.ContinuityIndexService,
        save_id: str,
    ) -> object:
        sync_calls.append(save_id)
        return original_sync_save(self, save_id)

    monkeypatch.setattr(
        chat_service_module.ContinuityIndexService,
        "sync_save",
        recording_sync_save,
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
        )
    )

    assert sync_calls == [save.id]


def test_submit_player_turn_applies_supported_chat_generation_settings(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "starting_scene": "The beacon gutters in the tower.",
            "lore": "The buried lens code hums under the ash.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.save_provider_model(
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
        display_name="Claude 3.5 Sonnet",
        capabilities=["chat"],
        supported_parameters=["temperature", "max_output_tokens"],
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    repositories.set_app_setting("chat_temperature_enabled", True)
    repositories.set_app_setting("chat_temperature", 1.1)
    repositories.set_app_setting("chat_max_output_tokens_enabled", True)
    repositories.set_app_setting("chat_max_output_tokens", 1800)
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert len(provider.chat_requests) == 1
    request = provider.chat_requests[0]
    assert request.temperature == 1.1
    assert request.max_output_tokens == 1800


def test_submit_player_turn_applies_openrouter_reasoning_override(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="z-ai/glm-4.7",
    )
    repositories.set_app_setting(
        OPENROUTER_CHAT_REASONING_OVERRIDES_SETTING,
        {"z-ai/glm-4.7": "disabled"},
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
        )
    )

    assert len(provider.chat_requests) == 1
    reasoning = provider.chat_requests[0].reasoning
    assert reasoning is not None
    assert reasoning.enabled is False
    assert reasoning.exclude is True


def test_submit_player_turn_applies_openrouter_routing_profile(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openai/gpt-5-mini",
    )
    repositories.set_app_setting(
        OPENROUTER_ROUTING_PROFILES_SETTING,
        {
            "global": {"sort": "price"},
            "task_overrides": {
                "narrator": {
                    "enabled": True,
                    "profile": {
                        "sort": "latency",
                        "only": ["azure"],
                    },
                }
            },
        },
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
        )
    )

    assert len(provider.chat_requests) == 1
    assert provider.chat_requests[0].openrouter_provider_routing == {
        "only": ["azure"],
        "sort": "latency",
    }


def test_chat_fallback_applies_openrouter_reasoning_override(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.save_provider_model(
        provider="openrouter",
        model_id="z-ai/glm-4.7",
        display_name="GLM 4.7",
        capabilities=["chat"],
    )
    repositories.set_model_preference(
        task="chat",
        provider="venice",
        model_id="venice/chat",
    )
    repositories.set_model_preference(
        task="chat_fallback",
        provider="openrouter",
        model_id="z-ai/glm-4.7",
    )
    repositories.set_app_setting("chat_fallback_enabled", True)
    repositories.set_app_setting(
        OPENROUTER_CHAT_REASONING_OVERRIDES_SETTING,
        {"z-ai/glm-4.7": {"effort": "low", "exclude": True}},
    )
    primary = FailingChatProvider(
        "venice",
        ProviderError(ProviderErrorCategory.CONTENT_BLOCKED, "blocked"),
    )
    fallback = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"venice": primary, "openrouter": fallback},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
        )
    )

    assert len(primary.chat_requests) == 1
    assert primary.chat_requests[0].reasoning is None
    assert len(fallback.chat_requests) == 1
    reasoning = fallback.chat_requests[0].reasoning
    assert reasoning is not None
    assert reasoning.effort == "low"
    assert reasoning.exclude is True


def test_chat_fallback_filters_unsupported_generation_settings(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "starting_scene": "The beacon gutters in the tower.",
            "lore": "The buried lens code hums under the ash.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.save_provider_model(
        provider="primary",
        model_id="primary/chat",
        display_name="Primary Chat",
        capabilities=["chat"],
        supported_parameters=["temperature", "max_output_tokens"],
    )
    repositories.save_provider_model(
        provider="fallback",
        model_id="fallback/chat",
        display_name="Fallback Chat",
        capabilities=["chat"],
        supported_parameters=[],
    )
    repositories.set_model_preference(
        task="chat",
        provider="primary",
        model_id="primary/chat",
    )
    repositories.set_model_preference(
        task="chat_fallback",
        provider="fallback",
        model_id="fallback/chat",
    )
    repositories.set_app_setting("chat_fallback_enabled", True)
    repositories.set_app_setting("chat_temperature_enabled", True)
    repositories.set_app_setting("chat_temperature", 1.1)
    repositories.set_app_setting("chat_max_output_tokens_enabled", True)
    repositories.set_app_setting("chat_max_output_tokens", 1800)
    primary = FailingChatProvider(
        "primary",
        ProviderError(ProviderErrorCategory.CONTENT_BLOCKED, "blocked"),
    )
    fallback = RecordingChatProvider("fallback")
    service = ChatService(
        repositories=repositories,
        providers={"primary": primary, "fallback": fallback},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert len(primary.chat_requests) == 1
    assert primary.chat_requests[0].temperature == 1.1
    assert primary.chat_requests[0].max_output_tokens == 1800
    assert len(fallback.chat_requests) == 1
    assert fallback.chat_requests[0].temperature is None
    assert fallback.chat_requests[0].max_output_tokens is None


def test_submit_player_turn_includes_save_custom_instructions(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "starting_scene": "The beacon gutters in the tower.",
            "lore": "The buried lens code hums under the ash.",
        },
    )
    save = repositories.create_save(
        scenario_id=scenario.id,
        title="Night Watch",
        custom_instructions="  Keep player-facing choices tense and brief.  ",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert len(provider.chat_requests) == 1
    assert provider.chat_requests[0].custom_instructions == (
        "Keep player-facing choices tense and brief."
    )
    assert provider.chat_requests[0].regeneration_feedback == ""


def test_submit_player_turn_includes_user_narration_guidance_when_save_guidance_blank(
    repositories: PersistenceRepositories,
) -> None:
    user = repositories.create_user(
        username="Mira",
        role="user",
        password_hash="hash",
    )
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "starting_scene": "The beacon gutters in the tower.",
            "lore": "The buried lens code hums under the ash.",
        },
    )
    save = repositories.create_save(
        scenario_id=scenario.id,
        title="Night Watch",
        custom_instructions="",
    )
    repositories.set_scoped_setting(
        scope="user",
        scope_id=user.id,
        key="user_narration_guidance",
        value="  Keep narrator responses to two paragraphs or less.  ",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
            current_user_id=user.id,
        )
    )

    assert len(provider.chat_requests) == 1
    assert provider.chat_requests[0].user_narration_guidance == (
        "Keep narrator responses to two paragraphs or less."
    )
    assert provider.chat_requests[0].custom_instructions == ""


def test_submit_player_turn_save_guidance_overrides_user_narration_guidance(
    repositories: PersistenceRepositories,
) -> None:
    user = repositories.create_user(
        username="Mira",
        role="user",
        password_hash="hash",
    )
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "starting_scene": "The beacon gutters in the tower.",
            "lore": "The buried lens code hums under the ash.",
        },
    )
    save = repositories.create_save(
        scenario_id=scenario.id,
        title="Night Watch",
        custom_instructions="Keep this save tense and clipped.",
    )
    repositories.set_scoped_setting(
        scope="user",
        scope_id=user.id,
        key="user_narration_guidance",
        value="Keep narrator responses to two paragraphs or less.",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
            current_user_id=user.id,
        )
    )

    assert len(provider.chat_requests) == 1
    assert provider.chat_requests[0].custom_instructions == (
        "Keep this save tense and clipped."
    )
    assert provider.chat_requests[0].user_narration_guidance == (
        "Keep narrator responses to two paragraphs or less."
    )


def test_submit_player_turn_ignores_legacy_active_loss_outcome(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "starting_scene": "The beacon gutters in the tower.",
            "lore": "The buried lens code hums under the ash.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon tower has already collapsed.",
        provider="fake",
        model="fake-chat",
    )
    condition = repositories.add_loss_condition(
        save_id=save.id,
        name="Beacon collapse",
        description="The beacon tower collapsed.",
        status="triggered",
        source="structured",
    )
    repositories.create_loss_outcome(
        save_id=save.id,
        condition_id=condition.id,
        condition_name=condition.name,
        triggering_message_id=narrator_message.id,
        explanation="The tower collapsed into ash.",
        confidence=0.96,
        evidence={
            "items": [
                {
                    "source_message_id": narrator_message.id,
                    "quote": "collapsed",
                }
            ]
        },
        provider="fake",
        model="fake-loss-model",
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingChatProvider("fake")
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I try to stand back up.",
            speaker_name="Mara",
        )
    )

    assert len(provider.chat_requests) == 1
    assert result.player_message.body == "I try to stand back up."
    assert result.narrator_message.role == "narrator"
    assert [message.role for message in repositories.list_messages(save.id)] == [
        "narrator",
        "player",
        "narrator",
    ]
    assert repositories.get_active_loss_outcome(save.id) is not None


def test_submit_player_turn_does_not_append_loss_epilogue(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "starting_scene": "The beacon gutters in the tower.",
            "lore": "The buried lens code hums under the ash.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingChatProvider("fake")
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I keep climbing after the lens cracks.",
            speaker_name="Mara",
        )
    )

    messages = repositories.list_messages(save.id)
    assert [message.role for message in messages] == ["player", "narrator"]
    assert messages[0] == result.player_message
    assert messages[1] == result.narrator_message
    assert repositories.get_active_loss_outcome(save.id) is None


@pytest.mark.parametrize(
    (
        "scenario_type",
        "specific_task",
        "specific_provider",
        "specific_model",
        "content",
    ),
    [
        (
            "full_roleplay",
            "chat_full_roleplay",
            "fullroleplay",
            "fullroleplay/narrator",
            {"starting_scene": "The beacon gutters in the tower."},
        ),
        (
            "fantasy_roleplay",
            "chat_fantasy_roleplay",
            "fantasychat",
            "fantasychat/narrator",
            {"magic_system": "Oath-magic always asks a price."},
        ),
        (
            "science_fiction_roleplay",
            "chat_science_fiction_roleplay",
            "sciencechat",
            "sciencechat/narrator",
            {"technology_level": "FTL exists, but causality monitors bite."},
        ),
        (
            "first_contact_exploration",
            "chat_first_contact_exploration",
            "contactchat",
            "contactchat/narrator",
            {"translation_progress": "The second pulse may mean safe passage."},
        ),
        (
            "survival_expedition",
            "chat_survival_expedition",
            "survivalchat",
            "survivalchat/narrator",
            {"travel_progress": "Three miles gained before the whiteout."},
        ),
        (
            "time_loop",
            "chat_time_loop",
            "loopchat",
            "loopchat/narrator",
            {"current_loop_state": "Loop 2, storm phase, one deviation known."},
        ),
        (
            "investigation_mystery",
            "chat_investigation_mystery",
            "mysterychat",
            "mysterychat/narrator",
            {"case_status": "Unresolved; public facts only."},
        ),
        (
            "political_intrigue",
            "chat_political_intrigue",
            "intriguechat",
            "intriguechat/narrator",
            {"obligations_and_favors": "Orro owes Mara a public endorsement."},
        ),
    ],
)
def test_submit_player_turn_uses_scenario_specific_chat_model(
    repositories: PersistenceRepositories,
    scenario_type: str,
    specific_task: str,
    specific_provider: str,
    specific_model: str,
    content: dict[str, object],
) -> None:
    scenario = repositories.create_scenario(
        type=scenario_type,
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content=content,
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/generic-chat",
    )
    repositories.set_model_preference(
        task=specific_task,
        provider=specific_provider,
        model_id=specific_model,
    )
    generic_provider = RecordingChatProvider("openrouter")
    scenario_provider = RecordingChatProvider(specific_provider)
    service = ChatService(
        repositories=repositories,
        providers={
            "openrouter": generic_provider,
            specific_provider: scenario_provider,
        },
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert generic_provider.chat_requests == []
    assert len(scenario_provider.chat_requests) == 1
    request = scenario_provider.chat_requests[0]
    assert request.provider == specific_provider
    assert request.model_id == specific_model
    assert result.narrator_message.provider == specific_provider
    assert result.narrator_message.model == specific_model


def test_submit_existing_player_turn_uses_scenario_specific_chat_model(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/generic-chat",
    )
    repositories.set_model_preference(
        task="chat_full_roleplay",
        provider="fullroleplay",
        model_id="fullroleplay/narrator",
    )
    generic_provider = RecordingChatProvider("openrouter")
    scenario_provider = RecordingChatProvider("fullroleplay")
    service = ChatService(
        repositories=repositories,
        providers={
            "openrouter": generic_provider,
            "fullroleplay": scenario_provider,
        },
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    result = asyncio.run(
        service.submit_existing_player_turn(
            save_id=save.id,
            player_message_id=player_message.id,
            run_post_turn_jobs=False,
        )
    )

    assert generic_provider.chat_requests == []
    assert len(scenario_provider.chat_requests) == 1
    request = scenario_provider.chat_requests[0]
    assert request.provider == "fullroleplay"
    assert request.model_id == "fullroleplay/narrator"
    assert result.player_message == player_message
    assert result.narrator_message.provider == "fullroleplay"
    assert result.narrator_message.model == "fullroleplay/narrator"


def test_submit_existing_player_turn_includes_one_shot_regeneration_feedback(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/generic-chat",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_existing_player_turn(
            save_id=save.id,
            player_message_id=player_message.id,
            run_post_turn_jobs=False,
            regeneration_feedback="  Make the replacement more ominous.  ",
        )
    )

    assert len(provider.chat_requests) == 1
    assert provider.chat_requests[0].regeneration_feedback == (
        "Make the replacement more ominous."
    )
    refreshed_save = repositories.get_save(save.id)
    assert refreshed_save is not None
    assert refreshed_save.custom_instructions == ""


def test_submit_player_turn_retries_once_after_npc_knowledge_leak(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Archive Arrival",
        premise="An archive scene with uneven knowledge.",
        player_role="Avery",
        content={"starting_scene": "Nira arrives after the earlier archive-code joke."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Boundary Test")
    repositories.add_character(save_id=save.id, name="Nira", met=True)
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/generic-chat",
    )
    provider = SequenceChatProvider(
        "openrouter",
        (
            "Nira says, \"You made the archive-code joke within five minutes.\"",
            "Nira says, \"I missed the beginning, so I am catching up.\"",
        ),
    )
    leak = NpcKnowledgeLeak(
        speaker_name="Nira",
        claim="Avery made the archive-code joke within five minutes.",
        reason="Nira was not present for that exchange.",
    )
    auditor = ScriptedNpcKnowledgeAuditor(
        (
            NpcKnowledgeAuditResult(enabled=True, leaks=(leak,)),
            NpcKnowledgeAuditResult(enabled=True),
        )
    )
    context_update = RecordingContextUpdateService()
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        context_update_service=context_update,
        npc_knowledge_audit_service=auditor,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I ask Nira if she wants to enter the chart room.",
            speaker_name="Avery",
        )
    )

    assert len(provider.chat_requests) == 2
    assert "NPC knowledge leak" in provider.chat_requests[1].regeneration_feedback
    assert result.narrator_message.body == (
        "Nira says, \"I missed the beginning, so I am catching up.\""
    )
    assert [body for _save_id, body in auditor.calls] == [
        "Nira says, \"You made the archive-code joke within five minutes.\"",
        "Nira says, \"I missed the beginning, so I am catching up.\"",
    ]
    assert context_update.calls == [
        (save.id, (result.player_message.id, result.narrator_message.id))
    ]


def test_submit_player_turn_soft_fails_by_default_when_retry_still_leaks(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Archive Arrival",
        premise="An archive scene with uneven knowledge.",
        player_role="Avery",
        content={"starting_scene": "Nira arrives after the earlier archive-code joke."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Boundary Test")
    repositories.add_character(save_id=save.id, name="Nira", met=True)
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/generic-chat",
    )
    provider = SequenceChatProvider(
        "openrouter",
        (
            "Nira says, \"You made the archive-code joke within five minutes.\"",
            "Nira says, \"Still, the archive-code joke happened immediately.\"",
        ),
    )
    leak = NpcKnowledgeLeak(
        speaker_name="Nira",
        claim="Nira references the archive-code joke.",
        reason="No knowledge edge lets Nira know the joke.",
    )
    auditor = ScriptedNpcKnowledgeAuditor(
        (
            NpcKnowledgeAuditResult(enabled=True, leaks=(leak,)),
            NpcKnowledgeAuditResult(enabled=True, leaks=(leak,)),
        )
    )
    context_update = RecordingContextUpdateService()
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        context_update_service=context_update,
        npc_knowledge_audit_service=auditor,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I ask Nira if she wants to enter the chart room.",
            speaker_name="Avery",
        )
    )

    assert len(provider.chat_requests) == 2
    assert result.narrator_message.body == (
        "Nira says, \"Still, the archive-code joke happened immediately.\""
    )
    assert context_update.calls == [
        (save.id, (result.player_message.id, result.narrator_message.id))
    ]
    assert [
        message.body
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    ] == ["Nira says, \"Still, the archive-code joke happened immediately.\""]
    failed_chat_jobs = [
        job
        for job in repositories.list_jobs_by_status(("failed",))
        if job.save_id == save.id and job.type == "chat_completion"
    ]
    assert failed_chat_jobs == []
    succeeded_chat_jobs = [
        job
        for job in repositories.list_jobs_by_status(("succeeded",))
        if job.save_id == save.id and job.type == "chat_completion"
    ]
    assert len(succeeded_chat_jobs) == 1
    diagnostics = succeeded_chat_jobs[0].result or {}
    audit_diagnostics = diagnostics["npc_knowledge_audit"]
    assert isinstance(audit_diagnostics, dict)
    assert audit_diagnostics["suspicious"] is True
    assert audit_diagnostics["soft_failed"] is True
    assert audit_diagnostics["auto_retry_used"] is True


def test_submit_player_turn_hard_fails_when_retry_still_leaks(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Archive Arrival",
        premise="An archive scene with uneven knowledge.",
        player_role="Avery",
        content={"starting_scene": "Nira arrives after the earlier archive-code joke."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Boundary Test")
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=NPC_KNOWLEDGE_AUDIT_MODE_SETTING,
        value=NPC_KNOWLEDGE_AUDIT_MODE_HARD_FAIL,
    )
    repositories.add_character(save_id=save.id, name="Nira", met=True)
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/generic-chat",
    )
    provider = SequenceChatProvider(
        "openrouter",
        (
            "Nira says, \"You made the archive-code joke within five minutes.\"",
            "Nira says, \"Still, the archive-code joke happened immediately.\"",
        ),
    )
    leak = NpcKnowledgeLeak(
        speaker_name="Nira",
        claim="Nira references the archive-code joke.",
        reason="No knowledge edge lets Nira know the joke.",
    )
    auditor = ScriptedNpcKnowledgeAuditor(
        (
            NpcKnowledgeAuditResult(enabled=True, leaks=(leak,)),
            NpcKnowledgeAuditResult(enabled=True, leaks=(leak,)),
        )
    )
    context_update = RecordingContextUpdateService()
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        context_update_service=context_update,
        npc_knowledge_audit_service=auditor,
    )

    with pytest.raises(ValueError, match="NPC knowledge audit"):
        asyncio.run(
            service.submit_player_turn(
                save_id=save.id,
                body="I ask Nira if she wants to enter the chart room.",
                speaker_name="Avery",
            )
        )

    assert len(provider.chat_requests) == 2
    assert context_update.calls == []
    assert [
        message.body
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    ] == []
    failed_chat_jobs = [
        job
        for job in repositories.list_jobs_by_status(("failed",))
        if job.save_id == save.id and job.type == "chat_completion"
    ]
    assert len(failed_chat_jobs) == 1


def test_submit_player_turn_buffers_streaming_until_npc_audit_passes(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Archive Arrival",
        premise="An archive scene with uneven knowledge.",
        player_role="Avery",
        content={"starting_scene": "Nira arrives at the archive."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Boundary Test")
    repositories.add_character(save_id=save.id, name="Nira", met=True)
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/generic-chat",
    )
    provider = StreamingChatProvider(
        "openrouter",
        (
            ChatStreamChunk(delta="Nira"),
            ChatStreamChunk(delta=" nods.", token_usage={"total": 12}),
            ChatStreamChunk(token_usage={"total": 12}, done=True),
        ),
    )
    auditor = ScriptedNpcKnowledgeAuditor((NpcKnowledgeAuditResult(enabled=True),))
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        npc_knowledge_audit_service=auditor,
    )
    drafts: list[str] = []

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I ask Nira if she wants to enter the chart room.",
            speaker_name="Avery",
            narrator_stream_callback=drafts.append,
        )
    )

    assert result.narrator_message.body == "Nira nods."
    assert drafts == ["Nira nods."]
    assert auditor.calls == [(save.id, "Nira nods.")]


def test_submit_player_turn_audits_action_only_reply_with_active_scoped_context(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Archive Arrival",
        premise="An archive scene with uneven knowledge.",
        player_role="Avery",
        content={"starting_scene": "Nira arrives at the archive."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Boundary Test")
    source_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Tarin hears Avery make the archive-code joke while Nira is away.",
    )
    nira = repositories.add_character(
        save_id=save.id,
        name="Nira",
        met=True,
        character_id="character-nira-action-audit",
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="Nira knows Avery invited her inside.",
        tags=["nira"],
        source_message_id=source_message.id,
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=nira.id,
        target_type="memory",
        target_id=memory.id,
        knowledge_state="knows",
        acquisition_method="witnessed",
        source_message_id=source_message.id,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Nira waits near the apartment door.",
        present_character_ids=[nira.id],
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/generic-chat",
    )
    provider = StaticChatProvider(
        "openrouter",
        "She stiffens at the old joke.",
    )
    auditor = ScriptedNpcKnowledgeAuditor((NpcKnowledgeAuditResult(enabled=True),))
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        npc_knowledge_audit_service=auditor,
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I ask Nira if she wants to enter the chart room.",
            speaker_name="Avery",
        )
    )

    assert auditor.calls == [(save.id, "She stiffens at the old joke.")]


def test_submit_player_turn_omits_recent_message_hidden_from_active_npc(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Archive Arrival",
        premise="An archive scene with uneven knowledge.",
        player_role="Avery",
        content={"starting_scene": "Nira arrives at the archive."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Boundary Test")
    hidden = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Tarin hears Avery make the archive-code joke while Nira is away.",
    )
    nira = repositories.add_character(
        save_id=save.id,
        name="Nira",
        met=True,
        character_id="character-nira-transcript-filter",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Nira has just arrived at the archive.",
        present_character_ids=[nira.id],
    )
    repositories.add_message_visibility(
        save_id=save.id,
        message_id=hidden.id,
        character_id=nira.id,
        visibility="not_visible",
        confidence=1.0,
        source="scene_presence",
        evidence="Nira was not present for this exchange.",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/generic-chat",
    )
    provider = StaticChatProvider("openrouter", "The room goes quiet.")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I ask Nira if she wants to enter the chart room.",
            speaker_name="Avery",
        )
    )

    request_text = "\n".join(
        message.body for message in provider.chat_requests[0].messages
    )
    assert hidden.body not in request_text
    assert "I ask Nira if she wants to enter the chart room." in request_text


def test_submit_player_turn_keeps_message_hidden_only_from_absent_mention(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Archive Arrival",
        premise="An archive scene with uneven knowledge.",
        player_role="Avery",
        content={"starting_scene": "Nira arrives at the archive."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Boundary Test")
    current_scene_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Nira lowers her voice and gives Avery the vault password.",
    )
    nira = repositories.add_character(
        save_id=save.id,
        name="Nira",
        met=True,
        character_id="character-nira-present-transcript-filter",
    )
    lio = repositories.add_character(
        save_id=save.id,
        name="Archivist Lio",
        aliases=["Lio"],
        met=True,
        character_id="character-lio-absent-transcript-filter",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Nira is speaking with Avery while Lio is away.",
        present_character_ids=[nira.id],
    )
    repositories.add_message_visibility(
        save_id=save.id,
        message_id=current_scene_message.id,
        character_id=lio.id,
        visibility="not_visible",
        confidence=1.0,
        source="scene_presence",
        evidence="Lio was not present for this exchange.",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/generic-chat",
    )
    provider = StaticChatProvider("openrouter", "Nira glances at the door.")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I ask Nira whether Lio heard the vault password.",
            speaker_name="Avery",
        )
    )

    request_text = "\n".join(
        message.body for message in provider.chat_requests[0].messages
    )
    assert current_scene_message.body in request_text
    assert "I ask Nira whether Lio heard the vault password." in request_text


def test_submit_player_turn_omits_latest_summary_blocked_for_active_npc(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Archive Arrival",
        premise="An archive scene with uneven knowledge.",
        player_role="Avery",
        content={"starting_scene": "Nira arrives at the archive."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Boundary Test")
    source_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Tarin hears Avery make the archive-code joke while Nira is away.",
    )
    nira = repositories.add_character(
        save_id=save.id,
        name="Nira",
        met=True,
        character_id="character-nira-summary-filter",
    )
    tarin = repositories.add_character(
        save_id=save.id,
        name="Tarin",
        met=True,
        character_id="character-tarin-summary-filter",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Nira has just arrived at the archive.",
        present_character_ids=[nira.id],
    )
    summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=source_message.id,
        covers_message_end_id=source_message.id,
        body=(
            "Tarin privately heard Avery make the archive-code joke before Nira "
            "arrived."
        ),
        provider="fake",
        model="fake-summary",
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=tarin.id,
        target_type="summary",
        target_id=summary.id,
        knowledge_state="knows",
        acquisition_method="witnessed",
        source_message_id=source_message.id,
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/generic-chat",
    )
    provider = StaticChatProvider("openrouter", "The room goes quiet.")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I ask Nira if she wants to enter the chart room.",
            speaker_name="Avery",
        )
    )

    assert provider.chat_requests[0].summary is None


def test_submit_player_turn_does_not_include_summary_for_absent_mention(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Archive Arrival",
        premise="An archive scene with uneven knowledge.",
        player_role="Avery",
        content={"starting_scene": "Nira arrives at the archive."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Boundary Test")
    source_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Lio privately found the ledger before Nira arrived.",
    )
    nira = repositories.add_character(
        save_id=save.id,
        name="Nira",
        met=True,
        character_id="character-nira-summary-present",
    )
    lio = repositories.add_character(
        save_id=save.id,
        name="Archivist Lio",
        aliases=["Lio"],
        met=True,
        character_id="character-lio-summary-absent",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Nira has just arrived at the archive.",
        present_character_ids=[nira.id],
    )
    summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=source_message.id,
        covers_message_end_id=source_message.id,
        body="Lio knows the ledger is hidden under the drowned floorboard.",
        provider="fake",
        model="fake-summary",
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=lio.id,
        target_type="summary",
        target_id=summary.id,
        knowledge_state="knows",
        acquisition_method="witnessed",
        source_message_id=source_message.id,
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/generic-chat",
    )
    provider = StaticChatProvider("openrouter", "Nira frowns.")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I ask Nira what Lio knows about the ledger.",
            speaker_name="Avery",
        )
    )

    assert provider.chat_requests[0].summary is None


@pytest.mark.parametrize(
    ("scenario_type", "unrelated_specific_task", "content"),
    [
        (
            "full_roleplay",
            "chat_dating_sim",
            {"starting_scene": "The beacon gutters in the tower."},
        ),
        (
            "dating_sim",
            "chat_full_roleplay",
            {"opening_message": "Mael slides the forbidden index across."},
        ),
    ],
)
def test_submit_player_turn_falls_back_to_generic_chat_model(
    repositories: PersistenceRepositories,
    scenario_type: str,
    unrelated_specific_task: str,
    content: dict[str, object],
) -> None:
    scenario = repositories.create_scenario(
        type=scenario_type,
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content=content,
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter/generic-chat",
    )
    repositories.set_model_preference(
        task=unrelated_specific_task,
        provider="venice",
        model_id="venice/unused-specific-chat",
    )
    generic_provider = RecordingChatProvider("openrouter")
    unused_specific_provider = RecordingChatProvider("venice")
    service = ChatService(
        repositories=repositories,
        providers={
            "openrouter": generic_provider,
            "venice": unused_specific_provider,
        },
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert len(generic_provider.chat_requests) == 1
    assert unused_specific_provider.chat_requests == []
    request = generic_provider.chat_requests[0]
    assert request.provider == "openrouter"
    assert request.model_id == "openrouter/generic-chat"
    assert result.narrator_message.provider == "openrouter"
    assert result.narrator_message.model == "openrouter/generic-chat"


def test_submit_player_turn_captures_debug_prompt_without_persisting_it(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    prompt_only_context = "SENSITIVE CAPTURE CONTEXT SHOULD NOT BE PERSISTED"
    captured_requests: list[tuple[str, ChatRequest]] = []

    def capture_debug_prompt(
        *,
        message_id: str,
        request: ChatRequest,
    ) -> None:
        captured_requests.append((message_id, request))

    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(
                selected_memories=(
                    SelectedContextItem(
                        source_type="memory",
                        source_id="memory-sensitive-debug-context",
                        text=prompt_only_context,
                        relevance_note="Needed only for prompt inspection.",
                    ),
                ),
            )
        ),
        debug_prompt_capture=capture_debug_prompt,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert captured_requests == [
        (result.narrator_message.id, provider.chat_requests[0])
    ]
    captured_request = captured_requests[0][1]
    assert captured_request.retrieved_memories == (
        "[memory:memory-sensitive-debug-context] "
        "SENSITIVE CAPTURE CONTEXT SHOULD NOT BE PERSISTED",
    )
    assert captured_request.messages[-1].body == "I climb toward the beacon lens."

    persisted_text = _database_text(repositories)
    assert result.narrator_message.id in persisted_text
    assert prompt_only_context not in persisted_text


def test_submit_player_turn_persists_successful_chat_completion_job(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    jobs = _chat_completion_jobs(repositories, save.id)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["save_id"] == save.id
    assert job["status"] == "succeeded"
    assert job["error"] is None
    assert job["payload"]["provider"] == "openrouter"
    assert job["payload"]["model"] == "anthropic/claude-3.5-sonnet"
    assert job["payload"]["player_message_id"] == result.player_message.id
    assert job["payload"]["player_speaker_name"] == "Mara"
    assert job["result"]["narrator_message_id"] == result.narrator_message.id
    assert job["result"]["narrator_speaker_name"] == "Narrator"
    assert job["result"]["provider"] == "openrouter"
    assert job["result"]["model"] == "anthropic/claude-3.5-sonnet"
    assert job["result"]["token_usage"] == {
        "prompt": 11,
        "completion": 23,
        "total": 34,
    }


def test_submit_player_turn_persists_chat_transport_diagnostics(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = StreamingChatProvider(
        "openrouter",
        (
            ChatStreamChunk(delta="The streamed bell answers."),
            ChatStreamChunk(token_usage={"total": 12}, done=True),
        ),
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
        )
    )
    drafts: list[str] = []
    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I listen for the bell.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
            narrator_stream_callback=drafts.append,
        )
    )

    jobs = _chat_completion_jobs(repositories, save.id)
    assert len(jobs) == 2
    assert jobs[0]["result"]["transport_mode"] == "non_streaming"
    assert jobs[0]["result"]["streaming_used"] is False
    assert jobs[1]["result"]["transport_mode"] == "streaming"
    assert jobs[1]["result"]["streaming_used"] is True
    assert jobs[1]["result"]["context_search_failed"] is False
    assert jobs[1]["result"]["context_search_selected_counts"] == {
        "character_text_context": 0,
        "character_voice": 0,
        "memories": 0,
        "open_obligations": 0,
        "observations": 0,
        "recent_messages": 0,
        "scenario_sections": 0,
        "state": 0,
        "state_changes": 0,
        "summaries": 0,
        "media_assets": 0,
    }
    assert jobs[1]["result"]["prompt_context_trimmed"] is False
    assert drafts == ["The streamed bell answers."]


def test_submit_player_turn_marks_job_failed_when_finalization_rolls_back(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    debug_failure_text = "debug prompt capture exploded after narrator append"

    def fail_debug_prompt_capture(
        *,
        message_id: str,
        request: ChatRequest,
    ) -> None:
        raise RuntimeError(debug_failure_text)

    service = ChatService(
        repositories=repositories,
        providers={"openrouter": RecordingChatProvider("openrouter")},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        debug_prompt_capture=fail_debug_prompt_capture,
    )

    with pytest.raises(RuntimeError, match=debug_failure_text):
        asyncio.run(
            service.submit_player_turn(
                save_id=save.id,
                body="I climb toward the beacon lens.",
                speaker_name="Mara",
            )
        )

    persisted_messages = repositories.list_messages(save.id)
    assert [(message.role, message.body) for message in persisted_messages] == [
        ("player", "I climb toward the beacon lens.")
    ]

    jobs = _chat_completion_jobs(repositories, save.id)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["status"] == "failed"
    assert job["payload"]["player_message_id"] == persisted_messages[0].id
    assert job["result"]["player_message_id"] == persisted_messages[0].id
    assert debug_failure_text in job["error"]


def test_submit_player_turn_cancellation_after_chat_job_creation_cancels_job(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    context_search = ScriptedContextSearch(ContextSearchResult())
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=context_search,
    )
    cancelled_player_message_id: str | None = None

    def cancellation_requested() -> bool:
        nonlocal cancelled_player_message_id
        player_messages = [
            message
            for message in repositories.list_messages(save.id)
            if message.role == "player"
            and message.body == "I climb toward the beacon lens."
        ]
        jobs = _chat_completion_jobs(repositories, save.id)
        if not player_messages or not jobs:
            return False
        assert len(player_messages) == 1
        cancelled_player_message_id = player_messages[0].id
        return True

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            service.submit_player_turn(
                save_id=save.id,
                body="I climb toward the beacon lens.",
                speaker_name="Mara",
                cancellation_requested=cancellation_requested,
            )
        )

    assert cancelled_player_message_id is not None
    persisted_messages = repositories.list_messages(save.id)
    assert len(persisted_messages) == 1
    player_message = persisted_messages[0]
    assert player_message.id == cancelled_player_message_id
    assert player_message.role == "player"
    assert player_message.body == "I climb toward the beacon lens."
    assert player_message.speaker_name == "Mara"
    assert context_search.calls == []
    assert provider.chat_requests == []

    jobs = _chat_completion_jobs(repositories, save.id)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["status"] == "cancelled"
    assert job["payload"]["player_message_id"] == player_message.id
    assert job["result"] == {"player_message_id": player_message.id}
    assert job["error"] == "Chat turn cancelled"


def test_submit_player_turn_cancellation_after_chat_cancels_job_without_narrator(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = ChatCompletesThenCancelsProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            service.submit_player_turn(
                save_id=save.id,
                body="I climb toward the beacon lens.",
                speaker_name="Mara",
                cancellation_requested=lambda: provider.chat_completed,
            )
        )

    assert provider.chat_completed is True
    persisted_messages = repositories.list_messages(save.id)
    assert [(message.role, message.body) for message in persisted_messages] == [
        ("player", "I climb toward the beacon lens.")
    ]

    jobs = _chat_completion_jobs(repositories, save.id)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["status"] == "cancelled"
    assert job["payload"]["player_message_id"] == persisted_messages[0].id
    assert job["result"]["player_message_id"] == persisted_messages[0].id
    assert job["result"].get("narrator_message_id") is None


def test_submit_player_turn_cancels_running_context_search_child_task(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "starting_scene": "The beacon gutters in the tower.",
            "lore": "The buried lens code hums under the ash.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(CONTENT_FILTER_RATING_SETTING, "unrated")
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-chat",
        display_name="Fake Chat",
        capabilities=[ProviderCapability.CHAT.value],
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    provider = BlockingContextSearchProvider("fake")
    context_search = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=context_search,
    )
    cancellation_token = CancellationToken()

    async def submit_and_cancel() -> None:
        task = asyncio.create_task(
            service.submit_player_turn(
                save_id=save.id,
                body="I climb toward the beacon lens.",
                speaker_name="Mara",
                run_post_turn_jobs=False,
                cancellation_token=cancellation_token,
            )
        )
        try:
            await asyncio.wait_for(
                provider.entered_structured_output.wait(),
                timeout=1.0,
            )
            assert cancellation_token.cancel() is True
            await asyncio.wait_for(
                provider.cancelled_structured_output.wait(),
                timeout=1.0,
            )
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1.0)
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    asyncio.run(submit_and_cancel())

    assert [request.schema_name for request in provider.structured_output_requests] == [
        "context_search_selection"
    ]
    assert provider.chat_requests == []
    persisted_messages = repositories.list_messages(save.id)
    assert [(message.role, message.body) for message in persisted_messages] == [
        ("player", "I climb toward the beacon lens.")
    ]

    chat_jobs = _chat_completion_jobs(repositories, save.id)
    assert len(chat_jobs) == 1
    chat_job = chat_jobs[0]
    assert chat_job["status"] == "cancelled"
    assert chat_job["payload"]["player_message_id"] == persisted_messages[0].id
    assert chat_job["result"] == {"player_message_id": persisted_messages[0].id}
    assert chat_job["error"] == "Chat turn cancelled"

    context_jobs = _context_search_jobs(repositories, save.id)
    assert len(context_jobs) == 1
    assert context_jobs[0]["status"] == "cancelled"
    assert context_jobs[0]["error"] == "Context search cancelled"
    assert context_jobs[0]["result"] is None


def test_submit_player_turn_cancels_context_search_from_another_thread(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "starting_scene": "The beacon gutters in the tower.",
            "lore": "The buried lens code hums under the ash.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(CONTENT_FILTER_RATING_SETTING, "unrated")
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-chat",
        display_name="Fake Chat",
        capabilities=[ProviderCapability.CHAT.value],
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=[ProviderCapability.STRUCTURED_OUTPUT.value],
    )
    provider = BlockingContextSearchProvider("fake")
    context_search = ContextSearchService(
        repositories=repositories,
        providers={"fake": provider},
    )
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=context_search,
    )
    cancellation_token = CancellationToken()
    cancel_results: list[bool] = []

    async def submit_and_cancel_from_thread() -> None:
        task = asyncio.create_task(
            service.submit_player_turn(
                save_id=save.id,
                body="I climb toward the beacon lens.",
                speaker_name="Mara",
                run_post_turn_jobs=False,
                cancellation_token=cancellation_token,
            )
        )
        try:
            await asyncio.wait_for(
                provider.entered_structured_output.wait(),
                timeout=1.0,
            )

            cancel_thread = threading.Thread(
                target=lambda: cancel_results.append(cancellation_token.cancel())
            )
            cancel_thread.start()
            cancel_thread.join(timeout=1.0)

            assert not cancel_thread.is_alive()
            assert cancel_results == [True]
            await asyncio.wait_for(
                provider.cancelled_structured_output.wait(),
                timeout=1.0,
            )
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1.0)
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    asyncio.run(submit_and_cancel_from_thread())

    assert [request.schema_name for request in provider.structured_output_requests] == [
        "context_search_selection"
    ]
    assert provider.chat_requests == []
    persisted_messages = repositories.list_messages(save.id)
    assert [(message.role, message.body) for message in persisted_messages] == [
        ("player", "I climb toward the beacon lens.")
    ]

    chat_jobs = _chat_completion_jobs(repositories, save.id)
    assert len(chat_jobs) == 1
    chat_job = chat_jobs[0]
    assert chat_job["status"] == "cancelled"
    assert chat_job["payload"]["player_message_id"] == persisted_messages[0].id
    assert chat_job["result"] == {"player_message_id": persisted_messages[0].id}
    assert chat_job["error"] == "Chat turn cancelled"

    context_jobs = _context_search_jobs(repositories, save.id)
    assert len(context_jobs) == 1
    assert context_jobs[0]["status"] == "cancelled"
    assert context_jobs[0]["error"] == "Context search cancelled"
    assert context_jobs[0]["result"] is None


def test_submit_player_turn_cancels_chat_completion_job_when_task_is_cancelled(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingChatProvider("fake")
    planner = BlockingNarratorPlanner()
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        narrator_planner=planner,
    )

    async def submit_and_cancel_task() -> None:
        task = asyncio.create_task(
            service.submit_player_turn(
                save_id=save.id,
                body="I climb toward the beacon lens.",
                speaker_name="Mara",
                run_post_turn_jobs=False,
            )
        )
        try:
            await asyncio.wait_for(planner.entered.wait(), timeout=1.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1.0)
            await asyncio.wait_for(planner.cancelled.wait(), timeout=1.0)
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    asyncio.run(submit_and_cancel_task())

    persisted_messages = repositories.list_messages(save.id)
    assert [(message.role, message.body) for message in persisted_messages] == [
        ("player", "I climb toward the beacon lens.")
    ]
    assert planner.requests
    assert provider.chat_requests == []

    chat_jobs = _chat_completion_jobs(repositories, save.id)
    assert len(chat_jobs) == 1
    chat_job = chat_jobs[0]
    assert chat_job["status"] == "cancelled"
    assert chat_job["payload"]["player_message_id"] == persisted_messages[0].id
    assert chat_job["result"] == {"player_message_id": persisted_messages[0].id}
    assert chat_job["error"] == "Chat turn cancelled"


def test_submit_player_turn_logs_metadata_without_story_text(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    from bragi.app_logging import configure_logging

    paths = _storage_paths(tmp_path)
    log_path = paths.state_dir / "logs" / "bragi.log"
    bragi_logger = logging.getLogger("bragi")
    original_handlers = tuple(bragi_logger.handlers)
    original_level = bragi_logger.level
    original_propagate = bragi_logger.propagate

    scenario_title = "SENSITIVE SCENARIO TITLE"
    scenario_premise = "SENSITIVE SCENARIO PREMISE"
    player_role = "SENSITIVE PLAYER ROLE"
    opening_scene = "SENSITIVE OPENING SCENE"
    player_body = "SENSITIVE PLAYER ACTION"
    context_text = "SENSITIVE RETRIEVED MEMORY"
    narrator_body = "SENSITIVE NARRATOR RESPONSE"

    try:
        configure_logging(paths)

        scenario = repositories.create_scenario(
            type="full_roleplay",
            title=scenario_title,
            premise=scenario_premise,
            player_role=player_role,
            content={"starting_scene": opening_scene},
        )
        save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
        repositories.set_model_preference(
            task="chat",
            provider="openrouter",
            model_id="anthropic/claude-3.5-sonnet",
        )
        provider = StaticChatProvider("openrouter", narrator_body)
        service = ChatService(
            repositories=repositories,
            providers={"openrouter": provider},
            context_search_service=ScriptedContextSearch(
                ContextSearchResult(
                    selected_memories=(
                        SelectedContextItem(
                            source_type="memory",
                            source_id="memory-1",
                            text=context_text,
                            relevance_note="SENSITIVE RELEVANCE NOTE",
                        ),
                    ),
                )
            ),
        )

        asyncio.run(
            service.submit_player_turn(
                save_id=save.id,
                body=player_body,
                speaker_name="Mara",
            )
        )
        _flush_handlers(bragi_logger.handlers)

        log_text = log_path.read_text(encoding="utf-8")
        assert "openrouter" in log_text
        assert "anthropic/claude-3.5-sonnet" in log_text
        assert "chat" in log_text
        assert "34" in log_text
        for sensitive_text in (
            scenario_title,
            scenario_premise,
            player_role,
            opening_scene,
            player_body,
            context_text,
            "SENSITIVE RELEVANCE NOTE",
            narrator_body,
        ):
            assert sensitive_text not in log_text
    finally:
        _restore_logger(
            bragi_logger,
            original_handlers,
            original_level=original_level,
            original_propagate=original_propagate,
        )


def test_submit_player_turn_debug_logging_records_stage_metadata_without_story_text(
    repositories: PersistenceRepositories,
    tmp_path: Path,
) -> None:
    from bragi.app_logging import configure_logging, set_debug_logging_enabled

    paths = _storage_paths(tmp_path)
    log_path = paths.state_dir / "logs" / "bragi.log"
    bragi_logger = logging.getLogger("bragi")
    original_handlers = tuple(bragi_logger.handlers)
    original_level = bragi_logger.level
    original_propagate = bragi_logger.propagate

    scenario_title = "SENSITIVE DEBUG SCENARIO TITLE"
    scenario_premise = "SENSITIVE DEBUG SCENARIO PREMISE"
    player_role = "SENSITIVE DEBUG PLAYER ROLE"
    opening_scene = "SENSITIVE DEBUG OPENING SCENE"
    older_player_body = "SENSITIVE DEBUG OLDER PLAYER BODY"
    older_narrator_body = "SENSITIVE DEBUG OLDER NARRATOR BODY"
    player_body = "SENSITIVE DEBUG PLAYER ACTION"
    context_text = "SENSITIVE DEBUG RETRIEVED MEMORY"
    context_note = "SENSITIVE DEBUG RELEVANCE NOTE"
    narrator_body = "SENSITIVE DEBUG NARRATOR RESPONSE"
    summary_body = "Mara crossed the ash bridge before hearing the windless bell."

    try:
        configure_logging(paths)
        set_debug_logging_enabled(True)

        scenario = repositories.create_scenario(
            type="full_roleplay",
            title=scenario_title,
            premise=scenario_premise,
            player_role=player_role,
            content={"starting_scene": opening_scene},
        )
        save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
        older_player = repositories.append_message(
            save_id=save.id,
            role="player",
            speaker_name="Mara",
            body=older_player_body,
        )
        older_narrator = repositories.append_message(
            save_id=save.id,
            role="narrator",
            speaker_name="Narrator",
            body=older_narrator_body,
            provider="openrouter",
            model="anthropic/claude-3.5-sonnet",
            token_estimate=14,
        )
        repositories.save_provider_model(
            provider="openrouter",
            model_id="anthropic/claude-3.5-sonnet",
            display_name="Claude 3.5 Sonnet",
            capabilities=["chat"],
            context_window=8192,
        )
        repositories.set_model_preference(
            task="chat",
            provider="openrouter",
            model_id="anthropic/claude-3.5-sonnet",
        )
        events: list[str] = []
        provider = StaticChatProvider("openrouter", narrator_body)
        summary_service = RecordingSummaryService(
            repositories=repositories,
            events=events,
            covers_message_start_id=older_player.id,
            covers_message_end_id=older_narrator.id,
        )
        context_search = ScriptedContextSearch(
            ContextSearchResult(
                selected_recent_messages=(
                    SelectedContextItem(
                        source_type="message",
                        source_id=older_narrator.id,
                        text=older_narrator_body,
                        relevance_note=context_note,
                    ),
                ),
                selected_memories=(
                    SelectedContextItem(
                        source_type="memory",
                        source_id="memory-1",
                        text=context_text,
                        relevance_note=context_note,
                    ),
                ),
            ),
            events=events,
        )
        service = ChatService(
            repositories=repositories,
            providers={"openrouter": provider},
            context_search_service=context_search,
            summary_service=summary_service,
        )

        asyncio.run(
            service.submit_player_turn(
                save_id=save.id,
                body=player_body,
                speaker_name="Mara",
            )
        )
        _flush_handlers(bragi_logger.handlers)

        log_text = log_path.read_text(encoding="utf-8")
        records = [json.loads(line) for line in log_text.splitlines() if line.strip()]
        debug_records = [record for record in records if record.get("level") == "DEBUG"]
        assert debug_records
        assert any(
            _debug_record_mentions(record, "summarization") for record in debug_records
        )
        assert any(
            _debug_record_mentions(record, "player")
            and _debug_record_mentions(record, "persist")
            for record in debug_records
        )
        assert any(
            _debug_record_mentions(record, "context")
            and _debug_record_mentions(record, "search")
            for record in debug_records
        )
        assert any(
            _debug_record_mentions(record, "provider")
            and _debug_record_mentions(record, "chat")
            for record in debug_records
        )
        assert any(
            _debug_record_mentions(record, "narrator")
            and _debug_record_mentions(record, "persist")
            for record in debug_records
        )
        assert any(
            _debug_record_mentions(record, "turn") and "duration_ms" in record
            for record in debug_records
        )
        assert any(record.get("save_id") == save.id for record in debug_records)
        assert any(
            record.get("provider") == "openrouter"
            and record.get("model") == "anthropic/claude-3.5-sonnet"
            for record in debug_records
        )

        for sensitive_text in (
            scenario_title,
            scenario_premise,
            player_role,
            opening_scene,
            older_player_body,
            older_narrator_body,
            player_body,
            context_text,
            context_note,
            narrator_body,
            summary_body,
        ):
            assert sensitive_text not in log_text
    finally:
        set_debug_logging_enabled(False)
        _restore_logger(
            bragi_logger,
            original_handlers,
            original_level=original_level,
            original_propagate=original_propagate,
        )


def test_submit_player_turn_runs_automatic_media_after_narrator_is_persisted(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    events: list[str] = []
    provider = RecordingChatProvider("openrouter")
    original_chat = provider.chat

    async def recording_chat(request: ChatRequest) -> ChatResponse:
        events.append("narrator_chat")
        response = await original_chat(request)
        events.append("narrator_chat_completed")
        return response

    provider.chat = recording_chat  # type: ignore[method-assign]
    media_service = RecordingMediaService(
        repositories=repositories,
        events=events,
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(),
            events=events,
        ),
        media_service=media_service,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert events == [
        "context_search",
        "narrator_chat",
        "narrator_chat_completed",
        "automatic_media",
    ]
    assert media_service.calls == [save.id]
    assert media_service.source_message_ids_at_call == [result.narrator_message.id]
    assert media_service.message_roles_at_call == [["player", "narrator"]]
    assert media_service.narrator_bodies_at_call == [
        ["openrouter narrator: I climb toward the beacon lens."]
    ]
    assert repositories.list_messages(save.id)[1] == result.narrator_message


def test_submit_player_turn_can_return_before_post_turn_jobs(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    events: list[str] = []
    provider = RecordingChatProvider("openrouter")
    media_service = RecordingMediaService(
        repositories=repositories,
        events=events,
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(),
            events=events,
        ),
        media_service=media_service,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
        )
    )

    assert events == ["context_search"]
    assert media_service.calls == []
    persisted_messages = repositories.list_messages(save.id)
    assert persisted_messages == [result.player_message, result.narrator_message]
    later_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="A later narrator beat arrives before deferred jobs run.",
        provider="openrouter",
        model="anthropic/claude-3.5-sonnet",
        token_estimate=12,
    )

    asyncio.run(
        service.run_post_turn_jobs(
            save_id=save.id,
            player_message_id=result.player_message.id,
            narrator_message_id=result.narrator_message.id,
        )
    )

    assert events == ["context_search", "automatic_media"]
    assert media_service.calls == [save.id]
    assert media_service.source_message_ids_at_call == [result.narrator_message.id]
    assert media_service.narrator_bodies_at_call == [
        [result.narrator_message.body, later_narrator.body]
    ]


def test_submit_player_turn_can_background_post_turn_jobs_when_requested(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": RecordingChatProvider("openrouter")},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )
    post_turn_started = asyncio.Event()
    release_post_turn = asyncio.Event()
    post_turn_finished = asyncio.Event()

    async def post_turn_step(**_kwargs: object) -> None:
        post_turn_started.set()
        await release_post_turn.wait()
        post_turn_finished.set()

    monkeypatch.setattr(service, "run_post_turn_jobs", post_turn_step)

    async def submit_turn() -> Any:
        result = await service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
            await_post_turn_jobs=False,
        )
        await asyncio.wait_for(post_turn_started.wait(), timeout=1)
        assert not post_turn_finished.is_set()
        release_post_turn.set()
        await asyncio.wait_for(post_turn_finished.wait(), timeout=1)
        return result

    result = asyncio.run(submit_turn())

    assert repositories.list_messages(save.id) == [
        result.player_message,
        result.narrator_message,
    ]


def test_run_post_turn_jobs_does_not_prune_state_after_extraction(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens hums awake.",
        provider="fake",
        model="fake-chat",
        token_estimate=7,
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-state-memory",
        display_name="Fake State Memory",
        capabilities=["structured_output"],
        context_window=8192,
    )
    repositories.set_model_preference(
        task="state_memory",
        provider="fake",
        model_id="fake-state-memory",
    )
    repositories.set_model_preference(
        task="state_pruning",
        provider="fake",
        model_id="fake-pruner",
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=DIRECTOR_PRESSURE_ENABLED_SETTING,
        value=False,
    )
    events: list[str] = []
    provider = RecordingStateMemoryProvider("fake", events=events)
    media_service = RecordingMediaService(
        repositories=repositories,
        events=events,
    )
    state_pruning_service = RecordingStatePruningService(events=events)
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=None,
        media_service=media_service,
        state_pruning_service=state_pruning_service,
    )

    asyncio.run(
        service.run_post_turn_jobs(
            save_id=save.id,
            player_message_id=player_message.id,
            narrator_message_id=narrator_message.id,
        )
    )

    assert "state_memory_extraction" in events
    assert "automatic_media" in events
    assert "state_pruning" not in events
    assert state_pruning_service.calls == []
    assert media_service.calls == [save.id]
    assert media_service.source_message_ids_at_call == [narrator_message.id]


def test_run_post_turn_jobs_does_not_hold_save_lock_during_slow_provider_steps(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens hums awake.",
        provider="fake",
        model="fake-chat",
        token_estimate=7,
    )
    repositories.set_model_preference(
        task="state_pruning",
        provider="fake",
        model_id="fake-pruner",
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=DIRECTOR_PRESSURE_ENABLED_SETTING,
        value=False,
    )
    events: list[str] = []
    lock_depth = 0

    class InstrumentedSaveLock:
        async def __aenter__(self) -> None:
            nonlocal lock_depth
            events.append("lock:enter")
            lock_depth += 1

        async def __aexit__(
            self,
            _exc_type: object,
            _exc: object,
            _traceback: object,
        ) -> None:
            nonlocal lock_depth
            events.append("lock:exit")
            lock_depth -= 1

    class LockAwareStatePruningService(RecordingStatePruningService):
        async def prune(
            self,
            *,
            save_id: str,
            review_only: bool = False,
        ) -> object:
            assert lock_depth == 0
            return await super().prune(save_id=save_id, review_only=review_only)

    service = ChatService(
        repositories=repositories,
        providers={},
        context_search_service=None,
        media_service=RecordingPreparedMediaService(
            repositories=repositories,
            events=events,
        ),
        state_pruning_service=LockAwareStatePruningService(events=events),
    )

    asyncio.run(
        service.run_post_turn_jobs(
            save_id=save.id,
            player_message_id=player_message.id,
            narrator_message_id=narrator_message.id,
            world_update_context=lambda: InstrumentedSaveLock(),
        )
    )

    assert "state_pruning" not in events
    assert events.count("lock:enter") == 2
    assert events.count("lock:exit") == 2
    assert lock_depth == 0


def test_run_post_turn_jobs_runs_context_update_after_state_memory_before_later_jobs(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Captain Ilyra steadies Mara in the beacon gallery.",
        provider="fake",
        model="fake-chat",
        token_estimate=7,
    )
    for task, model_id, display_name in (
        ("state_memory", "fake-state-memory", "Fake State Memory"),
        ("context_update", "fake-context-update", "Fake Context Update"),
        ("scenario_evolution", "fake-scenario-evolution", "Fake Scenario Evolution"),
    ):
        repositories.save_provider_model(
            provider="fake",
            model_id=model_id,
            display_name=display_name,
            capabilities=["structured_output"],
            context_window=8192,
        )
        repositories.set_model_preference(
            task=task,
            provider="fake",
            model_id=model_id,
        )
    repositories.set_model_preference(
        task="state_pruning",
        provider="fake",
        model_id="fake-pruner",
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=DIRECTOR_PRESSURE_ENABLED_SETTING,
        value=False,
    )
    events: list[str] = []
    provider = RecordingPostTurnStructuredProvider("fake", events=events)
    media_service = RecordingMediaService(
        repositories=repositories,
        events=events,
    )
    state_pruning_service = RecordingStatePruningService(events=events)
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=None,
        media_service=media_service,
        state_pruning_service=state_pruning_service,
    )

    asyncio.run(
        service.run_post_turn_jobs(
            save_id=save.id,
            player_message_id=player_message.id,
            narrator_message_id=narrator_message.id,
        )
    )

    assert sorted(events) == sorted(
        [
            "state_memory_extraction",
            "context_update",
            "context_observation_extraction",
            "scenario_evolution",
            "automatic_media",
        ]
    )
    _assert_event_before(events, "state_memory_extraction", "context_update")
    assert state_pruning_service.calls == []
    assert media_service.calls == [save.id]
    assert len(provider.structured_output_requests) == 4

    context_request = _only_request_matching(
        provider.structured_output_requests,
        "context",
        "update",
    )
    assert context_request.provider == "fake"
    assert context_request.model_id == "fake-context-update"
    assert "context" in context_request.schema_name.casefold()
    assert "update" in context_request.schema_name.casefold()
    schema_text = json.dumps(context_request.schema, sort_keys=True).casefold()
    assert "source_message_id" in schema_text
    prompt_text = "\n".join(
        message.body.casefold() for message in context_request.messages
    )
    assert "i climb toward the beacon lens." in prompt_text
    assert "captain ilyra steadies mara" in prompt_text
    assert "json" not in prompt_text


def test_run_post_turn_jobs_starts_independent_jobs_before_state_finishes(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens hums awake.",
        provider="fake",
        model="fake-chat",
    )
    events: list[str] = []
    state_started = asyncio.Event()
    scenario_started = asyncio.Event()
    image_started = asyncio.Event()
    service = ChatService(
        repositories=repositories,
        providers={},
        context_search_service=None,
    )

    async def state_step(**_kwargs: object) -> str:
        events.append("state_started")
        state_started.set()
        await scenario_started.wait()
        await image_started.wait()
        events.append("state_finished")
        return "succeeded"

    async def context_step(**_kwargs: object) -> str:
        events.append("context")
        assert "state_finished" in events
        return "succeeded"

    async def scenario_step(**_kwargs: object) -> str:
        await state_started.wait()
        events.append("scenario_started")
        scenario_started.set()
        return "succeeded"

    def prepare_image_step(**_kwargs: object) -> object:
        return object()

    async def image_step(
        _prepared_image: object,
        *,
        current_user_id: str | None = None,
    ) -> str:
        await state_started.wait()
        events.append("image_started")
        image_started.set()
        return "succeeded"

    monkeypatch.setattr(service, "_extract_state_and_memory_if_configured", state_step)
    monkeypatch.setattr(service, "_update_context_if_configured", context_step)
    monkeypatch.setattr(service, "_evolve_scenario_if_configured", scenario_step)
    monkeypatch.setattr(service, "_prepare_automatic_image_if_due", prepare_image_step)
    monkeypatch.setattr(
        service,
        "_generate_prepared_automatic_image_if_due",
        image_step,
    )

    asyncio.run(
        asyncio.wait_for(
            service.run_post_turn_jobs(
                save_id=save.id,
                player_message_id=player_message.id,
                narrator_message_id=narrator_message.id,
            ),
            timeout=1.0,
        )
    )

    assert events.index("scenario_started") < events.index("state_finished")
    assert events.index("image_started") < events.index("state_finished")
    assert events.index("context") > events.index("state_finished")


def test_run_post_turn_jobs_cancels_started_child_jobs(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens hums awake.",
        provider="fake",
        model="fake-chat",
    )
    state_started = asyncio.Event()
    state_cancelled = asyncio.Event()
    scenario_started = asyncio.Event()
    scenario_cancelled = asyncio.Event()
    service = ChatService(
        repositories=repositories,
        providers={},
        context_search_service=None,
    )

    async def state_step(**_kwargs: object) -> str:
        state_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            state_cancelled.set()
            raise
        raise AssertionError("state step resumed unexpectedly")

    async def scenario_step(**_kwargs: object) -> str:
        scenario_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            scenario_cancelled.set()
            raise
        raise AssertionError("scenario step resumed unexpectedly")

    monkeypatch.setattr(service, "_extract_state_and_memory_if_configured", state_step)
    monkeypatch.setattr(service, "_evolve_scenario_if_configured", scenario_step)

    async def run_and_cancel() -> None:
        task = asyncio.create_task(
            service.run_post_turn_jobs(
                save_id=save.id,
                player_message_id=player_message.id,
                narrator_message_id=narrator_message.id,
            )
        )
        await asyncio.wait_for(state_started.wait(), timeout=1.0)
        await asyncio.wait_for(scenario_started.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(state_cancelled.wait(), timeout=1.0)
        await asyncio.wait_for(scenario_cancelled.wait(), timeout=1.0)

    asyncio.run(run_and_cancel())


def test_run_post_turn_jobs_prefers_tool_calling_for_context_update(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Captain Ilyra steadies Mara in the beacon gallery.",
        provider="fake",
        model="fake-chat",
        token_estimate=7,
    )
    for task, model_id, display_name, capabilities in (
        (
            "state_memory",
            "fake-state-memory",
            "Fake State Memory",
            ["structured_output"],
        ),
        (
            "context_update",
            "fake-context-update",
            "Fake Context Update",
            ["structured_output", "tool_calling"],
        ),
        (
            "scenario_evolution",
            "fake-scenario-evolution",
            "Fake Scenario Evolution",
            ["structured_output"],
        ),
    ):
        repositories.save_provider_model(
            provider="fake",
            model_id=model_id,
            display_name=display_name,
            capabilities=capabilities,
            context_window=8192,
        )
        repositories.set_model_preference(
            task=task,
            provider="fake",
            model_id=model_id,
        )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=DIRECTOR_PRESSURE_ENABLED_SETTING,
        value=False,
    )
    events: list[str] = []
    provider = RecordingPostTurnToolProvider("fake", events=events)
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=None,
        media_service=None,
    )

    asyncio.run(
        service.run_post_turn_jobs(
            save_id=save.id,
            player_message_id=player_message.id,
            narrator_message_id=narrator_message.id,
        )
    )

    assert "context_update_tool_call" in events
    assert "context_update" not in events
    assert len(provider.tool_call_requests) == 1
    assert provider.tool_call_requests[0].model_id == "fake-context-update"
    assert sorted(
        request.schema_name for request in provider.structured_output_requests
    ) == sorted(
        [
            "state_memory_extraction",
            "context_observation_extraction",
            "scenario_evolution",
        ]
    )


def test_run_post_turn_jobs_prefers_tool_calling_for_state_memory(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Captain Ilyra steadies Mara in the beacon gallery.",
        provider="fake",
        model="fake-chat",
        token_estimate=7,
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-state-memory",
        display_name="Fake State Memory",
        capabilities=["structured_output", "tool_calling"],
        context_window=8192,
    )
    repositories.set_model_preference(
        task="state_memory",
        provider="fake",
        model_id="fake-state-memory",
    )
    events: list[str] = []
    provider = RecordingPostTurnToolProvider("fake", events=events)
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=None,
        media_service=None,
    )

    asyncio.run(
        service.run_post_turn_jobs(
            save_id=save.id,
            player_message_id=player_message.id,
            narrator_message_id=narrator_message.id,
        )
    )

    assert events == ["state_memory_tool_call"]
    assert provider.structured_output_requests == []
    assert [tool.name for tool in provider.tool_call_requests[0].tools] == [
        "patch_world_state",
        "record_memory_fact",
        "flag_state_conflict",
    ]


def test_run_post_turn_jobs_skips_context_update_without_structured_output_capability(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens hums awake.",
        provider="fake",
        model="fake-chat",
        token_estimate=7,
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-state-memory",
        display_name="Fake State Memory",
        capabilities=["structured_output"],
        context_window=8192,
    )
    repositories.set_model_preference(
        task="state_memory",
        provider="fake",
        model_id="fake-state-memory",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-context-update",
        display_name="Fake Context Update",
        capabilities=["chat"],
        context_window=8192,
    )
    repositories.set_model_preference(
        task="context_update",
        provider="fake",
        model_id="fake-context-update",
    )
    events: list[str] = []
    provider = RecordingPostTurnStructuredProvider("fake", events=events)
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=None,
        media_service=None,
    )

    asyncio.run(
        service.run_post_turn_jobs(
            save_id=save.id,
            player_message_id=player_message.id,
            narrator_message_id=narrator_message.id,
        )
    )

    assert events == ["state_memory_extraction"]
    assert len(provider.structured_output_requests) == 1
    assert "state" in provider.structured_output_requests[0].schema_name.casefold()


def test_run_post_turn_jobs_skips_context_update_with_missing_catalog_row(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens hums awake.",
        provider="fake",
        model="fake-chat",
        token_estimate=7,
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-state-memory",
        display_name="Fake State Memory",
        capabilities=["structured_output"],
        context_window=8192,
    )
    repositories.set_model_preference(
        task="state_memory",
        provider="fake",
        model_id="fake-state-memory",
    )
    repositories.set_model_preference(
        task="context_update",
        provider="fake",
        model_id="unsynced-context-update",
    )
    events: list[str] = []
    provider = RecordingPostTurnStructuredProvider("fake", events=events)
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=None,
        media_service=None,
    )

    asyncio.run(
        service.run_post_turn_jobs(
            save_id=save.id,
            player_message_id=player_message.id,
            narrator_message_id=narrator_message.id,
        )
    )

    assert events == ["state_memory_extraction"]
    assert len(provider.structured_output_requests) == 1
    assert "state" in provider.structured_output_requests[0].schema_name.casefold()


def test_run_post_turn_jobs_leaves_character_maintenance_for_scheduler(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        character_id="character-ilyra",
    )
    repositories.add_character(
        save_id=save.id,
        name="Ashknife",
        character_id="character-ashknife",
    )
    player_message_id, narrator_message_id = _append_completed_turns(
        repositories,
        save_id=save.id,
        count=3,
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-character-maintenance",
        display_name="Fake Character Maintenance",
        capabilities=["structured_output"],
        context_window=8192,
    )
    repositories.set_model_preference(
        task="character_registry_maintenance",
        provider="fake",
        model_id="fake-character-maintenance",
    )
    events: list[str] = []
    provider = RecordingPostTurnStructuredProvider("fake", events=events)
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=None,
        media_service=None,
    )

    asyncio.run(
        service.run_post_turn_jobs(
            save_id=save.id,
            player_message_id=player_message_id,
            narrator_message_id=narrator_message_id,
        )
    )

    assert events == []
    assert provider.structured_output_requests == []
    jobs = _post_turn_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert [job["name"] for job in jobs[0]["result"]["jobs"]] == [
        "state",
        "context",
        "time_reconciliation",
        "proactive_text",
        "director",
        "scenario",
        "image",
    ]


def test_run_post_turn_jobs_does_not_review_outcomes(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message_id, narrator_message_id = _append_completed_turns(
        repositories,
        save_id=save.id,
        count=1,
    )
    repositories.set_model_preference(
        task="scenario_outcome",
        provider="fake",
        model_id="fake-outcome",
    )
    events: list[str] = []
    provider = RecordingPostTurnStructuredProvider("fake", events=events)
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=None,
        media_service=None,
    )

    asyncio.run(
        service.run_post_turn_jobs(
            save_id=save.id,
            player_message_id=player_message_id,
            narrator_message_id=narrator_message_id,
        )
    )

    assert events == []
    assert provider.structured_output_requests == []
    jobs = _post_turn_jobs(repositories, save.id)
    assert len(jobs) == 1
    child_names = [child["name"] for child in jobs[0]["result"]["jobs"]]
    assert "outcome" not in child_names


def test_run_post_turn_jobs_records_coordinator_dependencies_and_child_statuses(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens hums awake.",
        provider="fake",
        model="fake-chat",
    )
    service = ChatService(
        repositories=repositories,
        providers={},
        context_search_service=None,
    )
    progress_updates: list[Any] = []

    async def state_step(**_kwargs: object) -> str:
        return "succeeded"

    async def context_step(**_kwargs: object) -> str:
        raise RuntimeError("context updater unavailable")

    async def scenario_step(**_kwargs: object) -> str:
        return "skipped"

    prepared_image = object()

    def prepare_image_step(**_kwargs: object) -> object:
        return prepared_image

    async def image_step(
        _prepared_image: object,
        *,
        current_user_id: str | None = None,
    ) -> str:
        assert _prepared_image is prepared_image
        raise RuntimeError("image provider unavailable")

    monkeypatch.setattr(service, "_extract_state_and_memory_if_configured", state_step)
    monkeypatch.setattr(service, "_update_context_if_configured", context_step)
    monkeypatch.setattr(service, "_evolve_scenario_if_configured", scenario_step)
    monkeypatch.setattr(service, "_prepare_automatic_image_if_due", prepare_image_step)
    monkeypatch.setattr(
        service,
        "_generate_prepared_automatic_image_if_due",
        image_step,
    )
    asyncio.run(
        service.run_post_turn_jobs(
            save_id=save.id,
            player_message_id=player_message.id,
            narrator_message_id=narrator_message.id,
            progress_callback=progress_updates.append,
        )
    )

    final_statuses = {
        job.name: job.status for job in progress_updates[-1].jobs
    }
    assert final_statuses == {
        "state": "succeeded",
        "context": "failed",
        "time_reconciliation": "blocked_dependency",
        "proactive_text": "blocked_dependency",
        "director": "blocked_dependency",
        "scenario": "skipped",
        "image": "failed",
    }
    assert progress_updates[-1].status_text == (
        "Post-turn: state succeeded, context failed, "
        "time_reconciliation blocked_dependency, proactive_text blocked_dependency, "
        "director blocked_dependency, scenario skipped, image failed"
    )
    assert all(
        [job.name for job in progress.jobs]
        == [
            "state",
            "context",
            "time_reconciliation",
            "proactive_text",
            "director",
            "scenario",
            "image",
        ]
        for progress in progress_updates
    )
    seen_statuses = {
        job.status for progress in progress_updates for job in progress.jobs
    }
    assert {
        "pending",
        "running",
        "succeeded",
        "failed",
        "skipped",
        "blocked_dependency",
    } <= seen_statuses

    jobs = _post_turn_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "succeeded"
    expected_dependencies = {
        "state": [],
        "context": ["state"],
        "time_reconciliation": ["context"],
        "proactive_text": ["time_reconciliation"],
        "director": ["time_reconciliation"],
        "scenario": [],
        "image": [],
    }
    assert jobs[0]["payload"]["dependencies"] == expected_dependencies
    assert jobs[0]["result"]["dependencies"] == expected_dependencies
    assert jobs[0]["payload"]["image_context_semantics"] == "pre_post_turn_updates"
    assert jobs[0]["result"]["image_context_semantics"] == "pre_post_turn_updates"
    result_jobs = jobs[0]["result"]["jobs"]
    assert [(job["name"], job["status"]) for job in result_jobs] == [
        ("state", "succeeded"),
        ("context", "failed"),
        ("time_reconciliation", "blocked_dependency"),
        ("proactive_text", "blocked_dependency"),
        ("director", "blocked_dependency"),
        ("scenario", "skipped"),
        ("image", "failed"),
    ]
    assert result_jobs[2]["result"]["blocked_by"] == "context"
    assert result_jobs[3]["result"]["blocked_by"] == "time_reconciliation"
    steps = repositories.list_job_steps(jobs[0]["id"])
    step_by_name = {step.name: step for step in steps}
    assert {
        name: (step.status, step.task)
        for name, step in step_by_name.items()
    } == {
        "state": ("succeeded", "state_memory"),
        "context": ("failed", "context_update"),
        "time_reconciliation": ("blocked_dependency", "context_update"),
        "proactive_text": ("blocked_dependency", "chat"),
        "director": ("blocked_dependency", "director_pressure"),
        "scenario": ("skipped", "scenario_evolution"),
        "image": ("failed", "image_generation"),
    }
    assert all(step.duration_ms is not None for step in steps)
    assert step_by_name["context"].error == "context updater unavailable"
    assert step_by_name["image"].error == "image provider unavailable"


def test_run_post_turn_jobs_records_world_time_reconciliation_metadata(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens hums awake.",
    )

    class RecordingPostTurnWorldTimeRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []

        async def advance_time_if_supported(
            self,
            *,
            save_id: str,
            latest_message_id: str,
        ) -> dict[str, object]:
            raise AssertionError("pre-turn world time should not run")

        async def reconcile_completed_turn(
            self,
            *,
            save_id: str,
            player_message_id: str,
            narrator_message_id: str,
        ) -> dict[str, object]:
            self.calls.append((save_id, player_message_id, narrator_message_id))
            return {
                "status": "queued",
                "skipped_reason": "narrator_only_ambiguous",
                "queued_count": 2,
                "source_message_ids": [player_message_id, narrator_message_id],
                "reason": "Mara says a private phrase in the tower.",
                "provider_payload": {"api_key": "secret"},
            }

    world_time = RecordingPostTurnWorldTimeRunner()
    service = ChatService(
        repositories=repositories,
        providers={},
        context_search_service=None,
        world_time_service=world_time,
    )

    async def state_step(**_kwargs: object) -> str:
        return "succeeded"

    async def context_step(**_kwargs: object) -> str:
        return "succeeded"

    monkeypatch.setattr(service, "_extract_state_and_memory_if_configured", state_step)
    monkeypatch.setattr(service, "_update_context_if_configured", context_step)

    asyncio.run(
        service.run_post_turn_jobs(
            save_id=save.id,
            player_message_id=player_message.id,
            narrator_message_id=narrator_message.id,
        )
    )

    assert world_time.calls == [(save.id, player_message.id, narrator_message.id)]
    coordinator = _post_turn_jobs(repositories, save.id)[0]
    result = _post_turn_child_result(coordinator, "time_reconciliation")
    assert result == {
        "status": "queued",
        "skipped_reason": "narrator_only_ambiguous",
        "queued_count": 2,
        "source_message_ids": [player_message.id, narrator_message.id],
    }
    diagnostics = coordinator["diagnostics"]
    assert diagnostics["bragi"]["world_time"] == result
    assert "private phrase" not in repr(coordinator["result"])
    assert "api_key" not in repr(coordinator["result"])
    assert _post_turn_child_status(coordinator, "time_reconciliation") == "queued"
    step_by_name = {
        step.name: step for step in repositories.list_job_steps(coordinator["id"])
    }
    assert step_by_name["time_reconciliation"].status == "succeeded"
    assert step_by_name["time_reconciliation"].metadata == result


def test_run_post_turn_jobs_defers_context_update_after_budget(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens hums awake.",
    )
    service = ChatService(
        repositories=repositories,
        providers={},
        context_search_service=None,
    )
    context_cancelled = asyncio.Event()

    async def state_step(**_kwargs: object) -> str:
        return "succeeded"

    async def context_step(**_kwargs: object) -> str:
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            context_cancelled.set()
            raise
        raise AssertionError("context step unexpectedly resumed")

    monkeypatch.setattr(service, "_extract_state_and_memory_if_configured", state_step)
    monkeypatch.setattr(service, "_update_context_if_configured", context_step)
    monkeypatch.setattr(
        chat_service_module,
        "POST_TURN_CONTEXT_UPDATE_BUDGET_SECONDS",
        0.01,
    )

    asyncio.run(
        service.run_post_turn_jobs(
            save_id=save.id,
            player_message_id=player_message.id,
            narrator_message_id=narrator_message.id,
        )
    )

    assert context_cancelled.is_set()
    job = _post_turn_jobs(repositories, save.id)[0]
    assert _post_turn_child_status(job, "context") == "deferred"
    context_result = _post_turn_child_result(job, "context")
    assert context_result["deferred"] is True
    assert context_result["deferred_reason"] == "timeout"
    retry_jobs = [
        retry_job
        for retry_job in repositories.list_jobs_by_status(("queued",))
        if retry_job.type == "context_update_retry"
    ]
    assert [retry.payload["reason"] for retry in retry_jobs] == [
        "post_turn_context_update_timeout"
    ]
    step_by_name = {
        step.name: step for step in repositories.list_job_steps(job["id"])
    }
    assert step_by_name["context"].status == "deferred"


def test_run_post_turn_jobs_runs_director_after_context_before_later_jobs(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(DIRECTOR_PRESSURE_ENABLED_SETTING, True)
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I ask whether anything changes.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon room stays quiet for another beat.",
        provider="fake",
        model="fake-chat",
    )
    events: list[str] = []
    director_result = DirectorPressureResult(
        applied=True,
        pressure_kind="external_complication",
        directive="Raise stakes: guards start searching this floor.",
        assessment="The scene has stalled.",
        active_thread_title="Guards search the tower floor",
        active_thread_description="The guard sweep is moving toward the beacon.",
        active_thread_priority=3,
        state=DirectorPressureState(
            dramatic_questions=("Will Mara warn the lower village?",),
            tension_level=3,
            tension_trend="stalled",
            stall_turns=0,
            cooldown_turns=2,
        ),
    )
    director = ScriptedDirectorPressureRunner(director_result, events=events)
    service = ChatService(
        repositories=repositories,
        providers={},
        context_search_service=None,
        director_pressure_service=director,
    )

    async def state_step(**_kwargs: object) -> str:
        events.append("state")
        return "succeeded"

    async def context_step(**_kwargs: object) -> str:
        events.append("context")
        return "succeeded"

    async def scenario_step(**_kwargs: object) -> str:
        events.append("scenario")
        return "skipped"

    prepared_image = object()

    def prepare_image_step(**_kwargs: object) -> object:
        return prepared_image

    async def image_step(
        _prepared_image: object,
        *,
        current_user_id: str | None = None,
    ) -> str:
        assert _prepared_image is prepared_image
        events.append("image")
        return "skipped"

    monkeypatch.setattr(service, "_extract_state_and_memory_if_configured", state_step)
    monkeypatch.setattr(service, "_update_context_if_configured", context_step)
    monkeypatch.setattr(service, "_evolve_scenario_if_configured", scenario_step)
    monkeypatch.setattr(service, "_prepare_automatic_image_if_due", prepare_image_step)
    monkeypatch.setattr(
        service,
        "_generate_prepared_automatic_image_if_due",
        image_step,
    )

    asyncio.run(
        service.run_post_turn_jobs(
            save_id=save.id,
            player_message_id=player_message.id,
            narrator_message_id=narrator_message.id,
        )
    )

    _assert_event_before(events, "state", "context")
    _assert_event_before(events, "context", "director_pressure")
    assert set(events) == {"state", "context", "director_pressure", "scenario", "image"}
    assert director.calls == [
        (save.id, player_message.id, narrator_message.id)
    ]
    assert director.commits == [(director_result, narrator_message.id)]
    job = _post_turn_jobs(repositories, save.id)[0]
    assert _post_turn_child_status(job, "director") == "succeeded"


def test_run_post_turn_jobs_persists_scene_presence_for_turn_messages(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens hums awake.",
    )
    mara = repositories.add_character(save_id=save.id, name="Mara")
    lio = repositories.add_character(save_id=save.id, name="Archivist Lio")
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara climbs alone.",
        present_character_ids=[mara.id],
    )
    service = ChatService(
        repositories=repositories,
        providers={},
        context_search_service=None,
    )

    async def state_step(**_kwargs: object) -> str:
        return "succeeded"

    async def context_step(**_kwargs: object) -> str:
        repositories.upsert_scene_snapshot(
            save_id=save.id,
            situation="Lio joins Mara at the beacon lens.",
            present_character_ids=[lio.id],
        )
        return "succeeded"

    monkeypatch.setattr(service, "_extract_state_and_memory_if_configured", state_step)
    monkeypatch.setattr(service, "_update_context_if_configured", context_step)

    asyncio.run(
        service.run_post_turn_jobs(
            save_id=save.id,
            player_message_id=player_message.id,
            narrator_message_id=narrator_message.id,
        )
    )

    for message in (player_message, narrator_message):
        rows = repositories.list_message_scene_presence(
            save.id,
            message_id=message.id,
        )
        assert [(row.character_id, row.source) for row in rows] == [
            (lio.id, "post_turn_context")
        ]


def test_run_post_turn_jobs_leaves_world_context_retention_for_scheduler(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens hums awake.",
        provider="fake",
        model="fake-chat",
    )
    retention = RecordingWorldContextRetentionService()
    service = ChatService(
        repositories=repositories,
        providers={},
        context_search_service=None,
        world_context_retention_service=retention,
    )

    asyncio.run(
        service.run_post_turn_jobs(
            save_id=save.id,
            player_message_id=player_message.id,
            narrator_message_id=narrator_message.id,
        )
    )

    assert retention.calls == []
    coordinator = _post_turn_jobs(repositories, save.id)[0]
    assert [job["name"] for job in coordinator["result"]["jobs"]] == [
        "state",
        "context",
        "time_reconciliation",
        "proactive_text",
        "director",
        "scenario",
        "image",
    ]


def test_run_post_turn_jobs_queues_retry_when_continuity_update_fails(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens hums awake.",
    )
    mara = repositories.add_character(save_id=save.id, name="Mara")
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara climbs alone.",
        present_character_ids=[mara.id],
    )
    context_update = FailingContextUpdateService(RuntimeError("extractor down"))
    service = ChatService(
        repositories=repositories,
        providers={},
        context_search_service=None,
        context_update_service=context_update,
    )

    asyncio.run(
        service.run_post_turn_jobs(
            save_id=save.id,
            player_message_id=player_message.id,
            narrator_message_id=narrator_message.id,
        )
    )

    assert context_update.calls == [
        (save.id, (player_message.id, narrator_message.id))
    ]
    jobs = _post_turn_jobs(repositories, save.id)
    context_result = next(
        job for job in jobs[0]["result"]["jobs"] if job["name"] == "context"
    )["result"]
    assert jobs[0]["result"]["maintenance_degraded"] is True
    assert jobs[0]["result"]["maintenance_failed_jobs"] == ["context"]
    retry_job_id = context_result["retry_job_id"]
    retry_jobs = [
        job
        for job in repositories.list_jobs_by_status(("queued",))
        if job.type == "context_update_retry"
    ]
    assert [job.id for job in retry_jobs] == [retry_job_id]
    assert retry_jobs[0].payload["source_message_ids"] == [
        player_message.id,
        narrator_message.id,
    ]
    assert context_result["continuity_update_failed"] is True
    for message in (player_message, narrator_message):
        rows = repositories.list_message_scene_presence(
            save.id,
            message_id=message.id,
        )
        assert [(row.character_id, row.source) for row in rows] == [
            (mara.id, "post_turn_context")
        ]


def test_run_post_turn_jobs_queues_state_retry_and_blocks_dependents(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens hums awake.",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-state-memory",
        display_name="Fake State Memory",
        capabilities=["structured_output"],
        context_window=8192,
    )
    repositories.set_model_preference(
        task="state_memory",
        provider="fake",
        model_id="fake-state-memory",
    )
    events: list[str] = []
    provider = RecordingStateMemoryProvider(
        "fake",
        events=events,
        structured_error=RuntimeError("state extractor unavailable"),
    )
    context_update = RecordingContextUpdateService()
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=None,
        context_update_service=context_update,
    )

    asyncio.run(
        service.run_post_turn_jobs(
            save_id=save.id,
            player_message_id=player_message.id,
            narrator_message_id=narrator_message.id,
        )
    )

    assert events == ["state_memory_extraction"]
    assert context_update.calls == []
    coordinator = _post_turn_jobs(repositories, save.id)[0]
    assert _post_turn_child_status(coordinator, "state") == "failed"
    assert _post_turn_child_status(coordinator, "context") == "blocked_dependency"
    assert (
        _post_turn_child_status(coordinator, "time_reconciliation")
        == "blocked_dependency"
    )
    blocked_context = _post_turn_child_result(coordinator, "context")
    assert blocked_context == {
        "blocked_by": "state",
        "blocked_dependency_status": "failed",
        "source_message_ids": [player_message.id, narrator_message.id],
    }
    state_retry_jobs = [
        job
        for job in repositories.list_jobs_by_status(("queued",))
        if job.type == "state_extraction_retry"
    ]
    assert len(state_retry_jobs) == 1
    assert state_retry_jobs[0].payload["source_message_ids"] == [
        player_message.id,
        narrator_message.id,
    ]
    assert state_retry_jobs[0].payload["reason"] == "post_turn_state_failed"
    assert coordinator["result"]["maintenance_failed_jobs"] == ["state"]
    assert not [
        job
        for job in repositories.list_jobs_by_status(("queued",))
        if job.type == "context_update_retry"
    ]


def test_run_post_turn_jobs_queues_state_retry_when_extractor_unavailable(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens hums awake.",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-state-memory",
        display_name="Fake State Memory",
        capabilities=["chat"],
        context_window=8192,
    )
    repositories.set_model_preference(
        task="state_memory",
        provider="fake",
        model_id="fake-state-memory",
    )
    context_update = RecordingContextUpdateService()
    service = ChatService(
        repositories=repositories,
        providers={},
        context_search_service=None,
        context_update_service=context_update,
    )

    asyncio.run(
        service.run_post_turn_jobs(
            save_id=save.id,
            player_message_id=player_message.id,
            narrator_message_id=narrator_message.id,
        )
    )

    assert context_update.calls == []
    coordinator = _post_turn_jobs(repositories, save.id)[0]
    assert _post_turn_child_status(coordinator, "state") == "failed"
    assert _post_turn_child_status(coordinator, "context") == "blocked_dependency"
    state_result = _post_turn_child_result(coordinator, "state")
    retry_jobs = [
        job
        for job in repositories.list_jobs_by_status(("queued",))
        if job.type == "state_extraction_retry"
    ]
    assert len(retry_jobs) == 1
    assert state_result["retry_job_id"] == retry_jobs[0].id
    assert retry_jobs[0].payload["reason"] == "state_extraction_unavailable"
    assert retry_jobs[0].payload["source_message_ids"] == [
        player_message.id,
        narrator_message.id,
    ]


def test_run_state_extraction_retries_applies_once_and_queues_context_retry(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens hums awake.",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-state-memory",
        display_name="Fake State Memory",
        capabilities=["structured_output"],
        context_window=8192,
    )
    repositories.set_model_preference(
        task="state_memory",
        provider="fake",
        model_id="fake-state-memory",
    )
    repositories.create_job(
        save_id=save.id,
        type="state_extraction_retry",
        status="queued",
        payload={
            "source_message_ids": [player_message.id, narrator_message.id],
            "reason": "post_turn_state_failed",
            "retry_attempt": 1,
            "max_retry_attempts": 3,
            "provider": "fake",
            "model": "fake-state-memory",
            "include_memories": True,
        },
    )
    events: list[str] = []
    provider = RecordingStateMemoryProvider(
        "fake",
        events=events,
        structured_data={
            "state_changes": [
                {
                    "operation": "upsert",
                    "key": "scene.location",
                    "value": {"name": "Beacon gallery"},
                    "category": "scene",
                    "confidence": 0.87,
                    "evidence_quote": "The beacon lens hums awake.",
                }
            ],
            "memories": [
                {
                    "body": "Mara promised Elian she would keep the beacon lit.",
                    "tags": ["beacon", "promise"],
                    "importance": 0.91,
                    "evidence_quote": "The beacon lens hums awake.",
                }
            ],
        },
    )
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=None,
    )

    completed = asyncio.run(service.run_state_extraction_retries(save_id=save.id))

    assert completed == 1
    assert events == ["state_memory_extraction"]
    assert [
        (state.key, state.value)
        for state in repositories.list_world_state(save.id)
    ] == [("scene.location", {"name": "Beacon gallery"})]
    assert [memory.body for memory in repositories.list_memories(save.id)] == [
        "Mara promised Elian she would keep the beacon lit."
    ]
    context_retries = [
        job
        for job in repositories.list_jobs_by_status(("queued",))
        if job.type == "context_update_retry"
    ]
    assert len(context_retries) == 1
    assert context_retries[0].payload["reason"] == "state_extraction_retry_succeeded"
    assert context_retries[0].payload["run_full_post_turn_context"] is True
    assert context_retries[0].payload["source_message_ids"] == [
        player_message.id,
        narrator_message.id,
    ]
    for index in range(51):
        filler_job = repositories.create_job(
            save_id=save.id,
            type="state_extraction",
            status="running",
            payload={
                "source_message_ids": [
                    f"other-player-{index}",
                    f"other-narrator-{index}",
                ]
            },
        )
        repositories.update_job(filler_job.id, status="succeeded")
    repositories.create_job(
        save_id=save.id,
        type="state_extraction_retry",
        status="queued",
        payload={
            "source_message_ids": [player_message.id, narrator_message.id],
            "reason": "post_turn_state_failed",
            "retry_attempt": 1,
            "max_retry_attempts": 3,
            "provider": "fake",
            "model": "fake-state-memory",
            "include_memories": True,
        },
    )

    completed_again = asyncio.run(service.run_state_extraction_retries(save_id=save.id))

    assert completed_again == 1
    assert events == ["state_memory_extraction"]
    assert [memory.body for memory in repositories.list_memories(save.id)] == [
        "Mara promised Elian she would keep the beacon lit."
    ]
    assert [
        job
        for job in repositories.list_jobs_by_status(("queued",))
        if job.type == "context_update_retry"
    ] == context_retries


def test_run_post_turn_jobs_defers_low_priority_work_after_provider_pressure(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="context_update",
        provider="fake",
        model_id="fake-context",
    )
    repositories.set_model_preference(
        task="scenario_evolution",
        provider="fake",
        model_id="fake-scenario",
    )
    repositories.set_model_preference(
        task="state_pruning",
        provider="fake",
        model_id="fake-pruning",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens hums awake.",
    )
    calls: list[str] = []

    class UnexpectedWorldTimeRunner:
        def __init__(self) -> None:
            self.called = False

        async def advance_time_if_supported(
            self,
            *,
            save_id: str,
            latest_message_id: str,
        ) -> dict[str, object]:
            raise AssertionError("pre-turn world time should not run")

        async def reconcile_completed_turn(
            self,
            *,
            save_id: str,
            player_message_id: str,
            narrator_message_id: str,
        ) -> dict[str, object]:
            self.called = True
            raise AssertionError("world time should defer after provider pressure")

    world_time = UnexpectedWorldTimeRunner()
    service = ChatService(
        repositories=repositories,
        providers={},
        context_search_service=None,
        world_time_service=world_time,
    )

    async def state_step(**_kwargs: object) -> str:
        calls.append("state")
        raise _provider_error_with_retry_attempts(
            category=ProviderErrorCategory.RATE_LIMITED,
            message="provider is throttling maintenance",
            status_code=429,
            retry_attempt_count=3,
            max_retry_attempts=3,
            retry_attempts=(
                {"attempt": 1, "error_category": "rate_limited", "duration_ms": 20},
                {"attempt": 2, "error_category": "rate_limited", "duration_ms": 20},
                {"attempt": 3, "error_category": "rate_limited", "duration_ms": 20},
            ),
        )

    async def unexpected_step(**_kwargs: object) -> str:
        calls.append("unexpected")
        return "succeeded"

    monkeypatch.setattr(service, "_extract_state_and_memory_if_configured", state_step)
    monkeypatch.setattr(service, "_update_context_if_configured", unexpected_step)
    monkeypatch.setattr(service, "_evolve_scenario_if_configured", unexpected_step)

    asyncio.run(
        service.run_post_turn_jobs(
            save_id=save.id,
            player_message_id=player_message.id,
            narrator_message_id=narrator_message.id,
        )
    )

    assert calls == ["state"]
    assert world_time.called is False
    job = _post_turn_jobs(repositories, save.id)[0]
    assert _post_turn_child_status(job, "state") == "failed"
    assert _post_turn_child_status(job, "context") == "blocked_dependency"
    assert (
        _post_turn_child_status(job, "time_reconciliation")
        == "blocked_dependency"
    )
    assert _post_turn_child_status(job, "director") == "blocked_dependency"
    assert _post_turn_child_status(job, "scenario") == "skipped_provider_pressure"
    assert _post_turn_child_result(job, "context") == {
        "blocked_by": "state",
        "blocked_dependency_status": "failed",
        "source_message_ids": [player_message.id, narrator_message.id],
    }
    assert _post_turn_child_result(job, "time_reconciliation") == {
        "blocked_by": "context",
        "blocked_dependency_status": "blocked_dependency",
        "source_message_ids": [player_message.id, narrator_message.id],
    }
    retry_jobs = [
        job
        for job in repositories.list_jobs_by_status(("queued",))
        if job.type == "context_update_retry"
    ]
    assert retry_jobs == []


def test_run_post_turn_jobs_finishes_sibling_already_started_before_pressure(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="context_update",
        provider="fake",
        model_id="fake-context",
    )
    repositories.set_model_preference(
        task="scenario_evolution",
        provider="fake",
        model_id="fake-scenario",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens hums awake.",
    )
    service = ChatService(
        repositories=repositories,
        providers={},
        context_search_service=None,
    )
    calls: list[str] = []
    state_started = asyncio.Event()
    scenario_started = asyncio.Event()

    async def state_step(**_kwargs: object) -> str:
        calls.append("state")
        state_started.set()
        await scenario_started.wait()
        raise _provider_error_with_retry_attempts(
            category=ProviderErrorCategory.RATE_LIMITED,
            message="provider is throttling maintenance",
            status_code=429,
            retry_attempt_count=3,
            max_retry_attempts=3,
            retry_attempts=(
                {"attempt": 1, "error_category": "rate_limited", "duration_ms": 20},
                {"attempt": 2, "error_category": "rate_limited", "duration_ms": 20},
                {"attempt": 3, "error_category": "rate_limited", "duration_ms": 20},
            ),
        )

    async def scenario_step(**_kwargs: object) -> str:
        await state_started.wait()
        calls.append("scenario")
        scenario_started.set()
        return "succeeded"

    async def unexpected_step(**_kwargs: object) -> str:
        calls.append("unexpected")
        return "succeeded"

    monkeypatch.setattr(service, "_extract_state_and_memory_if_configured", state_step)
    monkeypatch.setattr(service, "_evolve_scenario_if_configured", scenario_step)
    monkeypatch.setattr(service, "_update_context_if_configured", unexpected_step)

    asyncio.run(
        asyncio.wait_for(
            service.run_post_turn_jobs(
                save_id=save.id,
                player_message_id=player_message.id,
                narrator_message_id=narrator_message.id,
            ),
            timeout=1.0,
        )
    )

    assert calls == ["state", "scenario"]
    job = _post_turn_jobs(repositories, save.id)[0]
    assert _post_turn_child_status(job, "state") == "failed"
    assert _post_turn_child_status(job, "scenario") == "succeeded"
    assert _post_turn_child_status(job, "context") == "blocked_dependency"


def test_run_post_turn_jobs_gates_low_priority_work_after_context_pressure_result(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="scenario_evolution",
        provider="fake",
        model_id="fake-scenario",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens hums awake.",
    )
    context_update = ResultContextUpdateService(
        {
            "provider_pressure": {
                "reason": "provider_pressure",
                "error_category": "rate_limited",
                "http_status": 429,
                "retry_attempt_count": 3,
                "max_retry_attempts": 3,
            }
        }
    )
    service = ChatService(
        repositories=repositories,
        providers={},
        context_search_service=None,
        context_update_service=context_update,
    )
    calls: list[str] = []

    async def state_step(**_kwargs: object) -> str:
        calls.append("state")
        return "succeeded"

    async def unexpected_step(**_kwargs: object) -> str:
        calls.append("unexpected")
        return "succeeded"

    monkeypatch.setattr(service, "_extract_state_and_memory_if_configured", state_step)
    monkeypatch.setattr(service, "_evolve_scenario_if_configured", unexpected_step)

    asyncio.run(
        service.run_post_turn_jobs(
            save_id=save.id,
            player_message_id=player_message.id,
            narrator_message_id=narrator_message.id,
        )
    )

    assert calls == ["state", "unexpected"]
    assert context_update.calls == [
        (save.id, (player_message.id, narrator_message.id))
    ]
    job = _post_turn_jobs(repositories, save.id)[0]
    assert _post_turn_child_status(job, "context") == "succeeded"
    assert _post_turn_child_status(job, "director") == "skipped"
    assert _post_turn_child_status(job, "scenario") == "succeeded"
    assert _post_turn_child_result(job, "context")["provider_pressure"] == {
        "reason": "provider_pressure",
        "error_category": "rate_limited",
        "http_status": 429,
        "retry_attempt_count": 3,
        "max_retry_attempts": 3,
    }


def test_run_post_turn_jobs_leaves_queued_context_retry_for_scheduler(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="scenario_evolution",
        provider="fake",
        model_id="fake-scenario",
    )
    old_player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climbed toward the beacon lens.",
    )
    old_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens hummed awake.",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I keep moving.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The gallery answers with a low red pulse.",
    )
    retry_job = repositories.create_job(
        save_id=save.id,
        type="context_update_retry",
        status="queued",
        payload={
            "source_message_ids": [old_player.id, old_narrator.id],
            "reason": "post_turn_context_update_failed",
            "retry_attempt": 1,
            "max_retry_attempts": 3,
        },
    )
    for index in range(60):
        filler = repositories.create_job(
            save_id=save.id,
            type=f"completed_noise_{index}",
            status="running",
            payload={},
        )
        repositories.update_job(
            filler.id,
            status="succeeded",
            result={"noise": index},
        )
        repositories.connection.execute(
            """
            UPDATE jobs
            SET completed_at = datetime('now', '-1 minute')
            WHERE id = ?
            """,
            (filler.id,),
        )
    repositories.commit()
    context_update = ResultContextUpdateService(
        {
            "provider_pressure": {
                "reason": "provider_pressure",
                "error_category": "rate_limited",
                "http_status": 429,
                "retry_attempt_count": 3,
                "max_retry_attempts": 3,
            }
        }
    )
    service = ChatService(
        repositories=repositories,
        providers={},
        context_search_service=None,
        context_update_service=context_update,
    )
    calls: list[str] = []

    async def state_step(**_kwargs: object) -> str:
        calls.append("state")
        return "succeeded"

    async def unexpected_step(**_kwargs: object) -> str:
        calls.append("unexpected")
        return "succeeded"

    monkeypatch.setattr(service, "_extract_state_and_memory_if_configured", state_step)
    monkeypatch.setattr(service, "_evolve_scenario_if_configured", unexpected_step)

    asyncio.run(
        service.run_post_turn_jobs(
            save_id=save.id,
            player_message_id=player_message.id,
            narrator_message_id=narrator_message.id,
        )
    )

    assert calls == ["state", "unexpected"]
    assert context_update.calls == [
        (save.id, (player_message.id, narrator_message.id))
    ]
    queued_retry = next(
        job
        for job in repositories.list_jobs_by_status(("queued",))
        if job.id == retry_job.id
    )
    assert queued_retry.payload["source_message_ids"] == [
        old_player.id,
        old_narrator.id,
    ]
    coordinator = _post_turn_jobs(repositories, save.id)[0]
    assert _post_turn_child_status(coordinator, "context") == "succeeded"
    assert _post_turn_child_status(coordinator, "scenario") == "succeeded"


def test_run_post_turn_jobs_reuses_existing_context_update_retry(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens hums awake.",
    )
    context_update = FailingContextUpdateService(RuntimeError("extractor down"))
    service = ChatService(
        repositories=repositories,
        providers={},
        context_search_service=None,
        context_update_service=context_update,
    )

    async def run_twice() -> tuple[object, object]:
        first = await service._update_context_if_configured(
            save_id=save.id,
            player_message_id=player_message.id,
            narrator_message_id=narrator_message.id,
        )
        second = await service._update_context_if_configured(
            save_id=save.id,
            player_message_id=player_message.id,
            narrator_message_id=narrator_message.id,
        )
        return first, second

    first_result, second_result = asyncio.run(run_twice())

    retry_jobs = [
        job
        for job in repositories.list_jobs_by_status(("queued",))
        if job.type == "context_update_retry"
    ]
    first_step = cast(Any, first_result)
    second_step = cast(Any, second_result)
    assert len(retry_jobs) == 1
    assert first_step.status == "failed"
    assert second_step.status == "failed"
    assert first_step.result["retry_job_id"] == retry_jobs[0].id
    assert second_step.result["retry_job_id"] == retry_jobs[0].id


def test_update_context_runs_curation_when_continuity_provider_unavailable(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_model_preference(
        task="context_update",
        provider="missing",
        model_id="missing-context",
    )
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens hums awake.",
    )
    events: list[str] = []

    class RecordingObservationService:
        async def observe_turn(
            self,
            *,
            save_id: str,
            source_message_ids: tuple[str, ...],
        ) -> ObservationResult:
            events.append("observation")
            return ObservationResult(save_id=save_id, observed_count=0)

    class RecordingCurationService:
        async def curate_pending(self, save_id: str) -> CurationResult:
            events.append("curation")
            return CurationResult(save_id=save_id, considered_count=0)

    service = ChatService(
        repositories=repositories,
        providers={},
        context_search_service=None,
        observation_service=RecordingObservationService(),
        context_curation_service=RecordingCurationService(),
    )

    result = asyncio.run(
        service._update_context_if_configured(
            save_id=save.id,
            player_message_id=player_message.id,
            narrator_message_id=narrator_message.id,
        )
    )

    assert result == "skipped"
    assert events == ["observation", "curation"]


def test_run_context_update_retries_replays_full_context_after_state_retry(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The beacon lens hums awake.",
    )
    lio = repositories.add_character(save_id=save.id, name="Archivist Lio")
    retry_job = repositories.create_job(
        save_id=save.id,
        type="context_update_retry",
        status="queued",
        payload={
            "source_message_ids": [player_message.id, narrator_message.id],
            "reason": "state_extraction_retry_succeeded",
            "run_full_post_turn_context": True,
        },
    )
    events: list[str] = []

    class RecordingObservationService:
        async def observe_turn(
            self,
            *,
            save_id: str,
            source_message_ids: tuple[str, ...],
        ) -> ObservationResult:
            assert source_message_ids == (player_message.id, narrator_message.id)
            events.append("observation")
            return ObservationResult(save_id=save_id, observed_count=0)

    class RecordingCurationService:
        async def curate_pending(self, save_id: str) -> CurationResult:
            events.append("curation")
            return CurationResult(save_id=save_id, considered_count=0)

    class ScenePresenceContextUpdateService:
        async def update_after_turn(
            self,
            *,
            save_id: str,
            source_message_ids: tuple[str, ...],
        ) -> object:
            assert source_message_ids == (player_message.id, narrator_message.id)
            events.append("context_update")
            repositories.upsert_scene_snapshot(
                save_id=save_id,
                situation="Lio joins Mara at the beacon lens.",
                present_character_ids=[lio.id],
            )
            return object()

    service = ChatService(
        repositories=repositories,
        providers={},
        context_search_service=None,
        context_update_service=ScenePresenceContextUpdateService(),
        observation_service=RecordingObservationService(),
        context_curation_service=RecordingCurationService(),
    )

    completed = asyncio.run(service.run_context_update_retries(save_id=save.id))

    assert completed == 1
    assert events == ["observation", "context_update", "curation"]
    succeeded_retry = next(
        job
        for job in repositories.list_jobs_by_status(("succeeded",))
        if job.id == retry_job.id
    )
    assert succeeded_retry.result is not None
    assert succeeded_retry.result["source_message_ids"] == [
        player_message.id,
        narrator_message.id,
    ]
    assert succeeded_retry.result["full_post_turn_context"] is True
    assert succeeded_retry.result["context_status"] == "succeeded"
    context_result = cast(dict[str, object], succeeded_retry.result["context_result"])
    assert "agentic_context" in context_result
    for message in (player_message, narrator_message):
        rows = repositories.list_message_scene_presence(
            save.id,
            message_id=message.id,
        )
        assert [(row.character_id, row.source) for row in rows] == [
            (lio.id, "context_retry")
        ]


def test_run_context_update_retries_defers_when_recent_provider_pressure(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        body="I climbed toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The beacon lens hummed awake.",
    )
    pressure_job = repositories.create_job(
        save_id=save.id,
        type="context_update",
        status="running",
        payload={"source_message_ids": [player_message.id, narrator_message.id]},
    )
    repositories.update_job(
        pressure_job.id,
        status="failed",
        error="rate_limited: provider is throttling maintenance",
        result={
            "error_category": "rate_limited",
            "http_status": 429,
            "attempt_count": 3,
            "max_attempts": 3,
        },
    )
    retry_job = repositories.create_job(
        save_id=save.id,
        type="context_update_retry",
        status="queued",
        payload={
            "source_message_ids": [player_message.id, narrator_message.id],
            "reason": "post_turn_context_update_failed",
            "retry_attempt": 1,
            "max_retry_attempts": 3,
        },
    )
    context_update = RecordingContextUpdateService()
    service = ChatService(
        repositories=repositories,
        providers={},
        context_search_service=None,
        context_update_service=context_update,
    )

    completed = asyncio.run(service.run_context_update_retries(save_id=save.id))

    assert completed == 0
    assert context_update.calls == []
    deferred_retry = next(
        job
        for job in repositories.list_jobs_by_status(("queued",))
        if job.id == retry_job.id
    )
    assert deferred_retry.payload["deferred_count"] == 1
    assert deferred_retry.payload["last_deferred_reason"] == "provider_pressure"
    assert deferred_retry.payload["last_pressure_category"] == "rate_limited"


def test_run_context_update_retries_processes_queued_retry_jobs(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The beacon lens hums awake.",
    )
    retry_job = repositories.create_job(
        save_id=save.id,
        type="context_update_retry",
        status="queued",
        payload={
            "source_message_ids": [player_message.id, narrator_message.id],
            "reason": "post_turn_context_update_failed",
        },
    )
    context_update = RecordingContextUpdateService()
    service = ChatService(
        repositories=repositories,
        providers={},
        context_search_service=None,
        context_update_service=context_update,
    )

    completed = asyncio.run(service.run_context_update_retries(save_id=save.id))

    assert completed == 1
    assert context_update.calls == [
        (save.id, (player_message.id, narrator_message.id))
    ]
    jobs = repositories.list_jobs_by_status(("succeeded",))
    succeeded_retry = next(job for job in jobs if job.id == retry_job.id)
    assert succeeded_retry.result == {
        "source_message_ids": [player_message.id, narrator_message.id]
    }


def test_run_context_update_retries_replaces_scene_presence_rows(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The beacon lens hums awake.",
    )
    mara = repositories.add_character(save_id=save.id, name="Mara")
    lio = repositories.add_character(save_id=save.id, name="Archivist Lio")
    for message in (player_message, narrator_message):
        repositories.replace_message_scene_presence(
            save.id,
            message.id,
            [mara.id],
            source="post_turn_context",
        )
    retry_job = repositories.create_job(
        save_id=save.id,
        type="context_update_retry",
        status="queued",
        payload={
            "source_message_ids": [player_message.id, narrator_message.id],
            "reason": "post_turn_context_update_failed",
        },
    )

    class ScenePresenceContextUpdateService:
        async def update_after_turn(
            self,
            *,
            save_id: str,
            source_message_ids: tuple[str, ...],
        ) -> object:
            assert save_id == save.id
            assert source_message_ids == (player_message.id, narrator_message.id)
            repositories.upsert_scene_snapshot(
                save_id=save.id,
                situation="Lio joins Mara at the beacon lens.",
                present_character_ids=[lio.id],
            )
            return object()

    service = ChatService(
        repositories=repositories,
        providers={},
        context_search_service=None,
        context_update_service=ScenePresenceContextUpdateService(),
    )

    completed = asyncio.run(service.run_context_update_retries(save_id=save.id))

    assert completed == 1
    assert next(
        job for job in repositories.list_jobs_by_status(("succeeded",))
        if job.id == retry_job.id
    ).result == {
        "source_message_ids": [player_message.id, narrator_message.id]
    }
    for message in (player_message, narrator_message):
        rows = repositories.list_message_scene_presence(
            save.id,
            message_id=message.id,
        )
        assert [(row.character_id, row.source) for row in rows] == [
            (lio.id, "context_retry")
        ]


def test_run_context_update_retries_cancels_started_child_jobs(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The beacon lens hums awake.",
    )
    repositories.create_job(
        save_id=save.id,
        type="context_update_retry",
        status="queued",
        payload={
            "source_message_ids": [player_message.id, narrator_message.id],
            "reason": "post_turn_context_update_failed",
            "retry_attempt": 2,
            "max_retry_attempts": 4,
        },
    )
    extractor = BlockingContextUpdateExtractor()
    context_update = ContextUpdateService(
        repositories=repositories,
        extractor=extractor,
    )
    service = ChatService(
        repositories=repositories,
        providers={},
        context_search_service=None,
        context_update_service=context_update,
    )

    async def run_and_cancel() -> None:
        task = asyncio.create_task(service.run_context_update_retries(save_id=save.id))
        try:
            await asyncio.wait_for(extractor.started.wait(), timeout=1.0)
            task.cancel()
            await asyncio.wait_for(extractor.cancelled.wait(), timeout=1.0)
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1.0)
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

    asyncio.run(run_and_cancel())

    jobs = repositories.connection.execute(
        """
        SELECT type, status, result_json, error
        FROM jobs
        WHERE save_id = ? AND type IN ('context_update_retry', 'context_update')
        ORDER BY created_at, rowid
        """,
        (save.id,),
    ).fetchall()
    assert [(job["type"], job["status"], job["error"]) for job in jobs] == [
        (
            "context_update_retry",
            "cancelled",
            "Context update retry drain cancelled",
        ),
        ("context_update", "cancelled", "Context update cancelled"),
    ]
    assert json.loads(jobs[0]["result_json"]) == {
        "source_message_ids": [player_message.id, narrator_message.id],
        "retry_attempt": 2,
        "max_retry_attempts": 4,
    }
    assert jobs[1]["result_json"] is None
    assert [
        job
        for job in repositories.list_jobs_by_status(("queued", "running"))
        if job.type in {"context_update_retry", "context_update"}
    ] == []


def test_run_context_update_retries_processes_bounded_batch(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    retry_count = chat_service_module.CONTEXT_UPDATE_RETRY_DRAIN_LIMIT + 1
    retry_ids: list[str] = []
    expected_calls: list[tuple[str, tuple[str, ...]]] = []
    for index in range(retry_count):
        player_message = repositories.append_message(
            save_id=save.id,
            role="player",
            body=f"I climb toward beacon lens {index}.",
        )
        narrator_message = repositories.append_message(
            save_id=save.id,
            role="narrator",
            body=f"The beacon lens hums awake {index}.",
        )
        retry = repositories.create_job(
            save_id=save.id,
            type="context_update_retry",
            status="queued",
            payload={
                "source_message_ids": [player_message.id, narrator_message.id],
                "reason": "post_turn_context_update_failed",
            },
        )
        retry_ids.append(retry.id)
        expected_calls.append(
            (save.id, (player_message.id, narrator_message.id))
        )
    context_update = RecordingContextUpdateService()
    service = ChatService(
        repositories=repositories,
        providers={},
        context_search_service=None,
        context_update_service=context_update,
    )

    completed = asyncio.run(service.run_context_update_retries(save_id=save.id))

    retry_limit = chat_service_module.CONTEXT_UPDATE_RETRY_DRAIN_LIMIT
    assert completed == retry_limit
    assert context_update.calls == expected_calls[:retry_limit]
    succeeded_ids = {
        job.id
        for job in repositories.list_jobs_by_status(("succeeded",))
        if job.type == "context_update_retry"
    }
    queued_ids = {
        job.id
        for job in repositories.list_jobs_by_status(("queued",))
        if job.type == "context_update_retry"
    }
    assert succeeded_ids == set(retry_ids[:retry_limit])
    assert queued_ids == {retry_ids[-1]}


def test_run_context_update_retries_stops_after_retry_budget_exhausted(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        body="I climbed toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The beacon lens hummed awake.",
    )
    retry_job = repositories.create_job(
        save_id=save.id,
        type="context_update_retry",
        status="queued",
        payload={
            "source_message_ids": [player_message.id, narrator_message.id],
            "reason": "post_turn_context_update_failed",
            "retry_attempt": 3,
            "max_retry_attempts": 3,
        },
    )
    context_update = FailingContextUpdateService(
        _provider_error_with_retry_attempts(
            category=ProviderErrorCategory.RATE_LIMITED,
            message="provider is still throttling maintenance",
            status_code=429,
            retry_attempt_count=3,
            max_retry_attempts=3,
            retry_attempts=(
                {"attempt": 1, "error_category": "rate_limited", "duration_ms": 20},
                {"attempt": 2, "error_category": "rate_limited", "duration_ms": 20},
                {"attempt": 3, "error_category": "rate_limited", "duration_ms": 20},
            ),
        )
    )
    service = ChatService(
        repositories=repositories,
        providers={},
        context_search_service=None,
        context_update_service=context_update,
    )

    completed = asyncio.run(service.run_context_update_retries(save_id=save.id))

    assert completed == 0
    assert context_update.calls == [
        (save.id, (player_message.id, narrator_message.id))
    ]
    assert [
        job
        for job in repositories.list_jobs_by_status(("queued",))
        if job.type == "context_update_retry"
    ] == []
    failed_retry = next(
        job
        for job in repositories.list_jobs_by_status(("failed",))
        if job.id == retry_job.id
    )
    assert failed_retry.result == {
        "source_message_ids": [player_message.id, narrator_message.id],
        "retry_attempt": 3,
        "max_retry_attempts": 3,
        "retry_budget_exhausted": True,
        "provider_pressure": {
            "reason": "provider_pressure",
            "error_category": "rate_limited",
            "http_status": 429,
            "retry_attempt_count": 3,
            "max_retry_attempts": 3,
        },
    }


def test_run_post_turn_jobs_does_not_drain_queued_context_update_retries(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    old_player = repositories.append_message(
        save_id=save.id,
        role="player",
        body="I climbed toward the beacon lens.",
    )
    old_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The beacon lens hummed awake.",
    )
    new_player = repositories.append_message(
        save_id=save.id,
        role="player",
        body="I brace the cracked lens housing.",
    )
    new_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The tower answers with a red pulse.",
    )
    retry_job = repositories.create_job(
        save_id=save.id,
        type="context_update_retry",
        status="queued",
        payload={
            "source_message_ids": [old_player.id, old_narrator.id],
            "reason": "provider_pressure_deferred",
            "retry_attempt": 1,
            "max_retry_attempts": 3,
            "last_deferred_reason": "provider_pressure",
            "last_pressure_category": "rate_limited",
            "last_pressure_http_status": 429,
        },
    )
    context_update = RecordingContextUpdateService()
    service = ChatService(
        repositories=repositories,
        providers={},
        context_search_service=None,
        context_update_service=context_update,
    )

    asyncio.run(
        service.run_post_turn_jobs(
            save_id=save.id,
            player_message_id=new_player.id,
            narrator_message_id=new_narrator.id,
        )
    )

    assert context_update.calls == [
        (save.id, (new_player.id, new_narrator.id)),
    ]
    queued_retries = [
        job
        for job in repositories.list_jobs_by_status(("queued",))
        if job.type == "context_update_retry"
    ]
    assert [job.id for job in queued_retries] == [retry_job.id]


def test_submit_player_turn_keeps_open_obligations_and_exact_facts_under_budget(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"lore": "The buried legion built the beacon."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    repositories.set_app_setting("context_budget_mode", "fixed_chars")
    repositories.set_app_setting("context_budget_fixed_total_chars", 1)
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(
                selected_open_obligations=(
                    SelectedContextItem(
                        source_type="open_obligation",
                        source_id="thread-lens",
                        text="Keep the copper notch pressed until Ilyra returns.",
                        relevance_note="Open obligation.",
                    ),
                ),
                selected_memories=(
                    SelectedContextItem(
                        source_type="memory",
                        source_id="memory-promise",
                        text=(
                            "Mara promised Ilyra she would keep the beacon lit."
                        ),
                        relevance_note="Exact promise.",
                    ),
                ),
                selected_scenario_sections=(
                    SelectedContextItem(
                        source_type="scenario_section",
                        source_id=f"scenario:{scenario.id}:section:lore",
                        text="The buried legion built the beacon.",
                        relevance_note="Lower priority lore.",
                    ),
                ),
            )
        ),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I keep my hand on the notch.",
            speaker_name="Mara",
        )
    )

    request = provider.chat_requests[0]
    assert "Keep the copper notch pressed" in "\n".join(request.open_obligations)
    assert "Mara promised Ilyra" in "\n".join(request.retrieved_memories)
    assert request.retrieved_scenario_sections == ()


def test_submit_player_turn_suppresses_open_obligation_covered_by_active_thread(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    thread = repositories.add_active_thread(
        save_id=save.id,
        title="Keep the lens notch pressed",
        description="Keep the copper notch pressed until Ilyra returns.",
        status="active",
        priority=5,
        visibility="public",
        thread_id="thread-lens",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(
                selected_open_obligations=(
                    SelectedContextItem(
                        source_type="open_obligation",
                        source_id=thread.id,
                        text="Keep the copper notch pressed until Ilyra returns.",
                        relevance_note="Open obligation.",
                    ),
                ),
            )
        ),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I keep my hand on the notch.",
            speaker_name="Mara",
        )
    )

    request = provider.chat_requests[0]
    assert "Keep the copper notch pressed" in "\n".join(
        request.current_scene_recap
    )
    assert request.open_obligations == ()


def test_run_post_turn_jobs_prepares_automatic_image_before_state_updates(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb toward the beacon lens.",
    )
    narrator_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The beacon lens hums awake.",
        provider="fake",
        model="fake-chat",
    )
    events: list[str] = []
    media_service = RecordingPreparedMediaService(
        repositories=repositories,
        events=events,
    )
    service = ChatService(
        repositories=repositories,
        providers={},
        context_search_service=None,
        media_service=media_service,
    )
    prepare_started = asyncio.Event()
    media_service.prepare_started = prepare_started

    async def state_step(**_kwargs: object) -> str:
        events.append("state_started")
        await prepare_started.wait()
        repositories.upsert_world_state(
            save_id=save.id,
            key="POST_PREPARE_STATE_beacon_lens",
            value={"status": "mutated after image prepare"},
            category="scene",
            source_message_id=narrator_message.id,
        )
        events.append("state_finished")
        return "succeeded"

    async def context_step(**_kwargs: object) -> str:
        events.append("context_update")
        return "succeeded"

    async def scenario_step(**_kwargs: object) -> str:
        events.append("scenario_evolution")
        return "skipped"

    monkeypatch.setattr(service, "_extract_state_and_memory_if_configured", state_step)
    monkeypatch.setattr(service, "_update_context_if_configured", context_step)
    monkeypatch.setattr(service, "_evolve_scenario_if_configured", scenario_step)

    asyncio.run(
        asyncio.wait_for(
            service.run_post_turn_jobs(
                save_id=save.id,
                player_message_id=player_message.id,
                narrator_message_id=narrator_message.id,
            ),
            timeout=1.0,
        )
    )

    assert media_service.prepared == [
        {
            "save_id": save.id,
            "source_message_id": narrator_message.id,
            "world_state_keys": [],
        }
    ]
    assert media_service.generated_prepared == [media_service.prepared[0]]
    assert "POST_PREPARE_STATE_beacon_lens" in [
        state.key for state in repositories.list_world_state(save.id)
    ]
    _assert_event_before(events, "automatic_media_prepare", "state_finished")


def test_submit_player_turn_allows_missing_context_search_service_for_narrator_turns(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=None,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert len(provider.chat_requests) == 1
    request = provider.chat_requests[0]
    assert request.retrieved_scenario_sections == ()
    assert request.retrieved_state == ()
    assert request.retrieved_memories == ()
    assert request.summary is None
    persisted_messages = repositories.list_messages(save.id)
    assert persisted_messages == [result.player_message, result.narrator_message]
    assert persisted_messages[0].role == "player"
    assert persisted_messages[0].speaker_name == "Mara"
    assert persisted_messages[0].body == "I climb toward the beacon lens."
    assert persisted_messages[1].role == "narrator"
    assert persisted_messages[1].speaker_name == "Narrator"
    assert persisted_messages[1].body == (
        "openrouter narrator: I climb toward the beacon lens."
    )


def test_submit_player_turn_generates_action_choices_after_narration(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Library of Falling Doors",
        premise="Every shelf is a door.",
        player_role="Courier",
        content={
            "action_choices_enabled": True,
            "choice_style": "Four concrete choices with different risks.",
            "opening_message": "The blue shelf opens.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Library")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter-chat",
    )
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="openrouter",
        model_id="openrouter-chat",
    )
    repositories.save_provider_model(
        provider="openrouter",
        model_id="openrouter-chat",
        display_name="OpenRouter Chat",
        capabilities=["chat", "structured_output"],
    )
    provider = RecordingCyoaChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=None,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I read the silver card.",
            speaker_name="Ily",
            run_post_turn_jobs=False,
        )
    )

    assert result.narrator_message.id
    assert [
        choice.body for choice in repositories.latest_message_action_choices(save.id)
    ] == [
        "Open the brass atlas.",
        "Question the librarian.",
        "Hide the index under your coat.",
        "Step through the blue shelf-door.",
    ]
    assert provider.structured_output_requests[-1].schema_name == "action_choices"


def test_submit_player_turn_runs_action_choices_alongside_post_turn_jobs(
    repositories: PersistenceRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Library of Falling Doors",
        premise="Every shelf is a door.",
        player_role="Courier",
        content={
            "action_choices_enabled": True,
            "choice_style": "Four concrete choices with different risks.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Library")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter-chat",
    )
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="openrouter",
        model_id="openrouter-chat",
    )
    repositories.save_provider_model(
        provider="openrouter",
        model_id="openrouter-chat",
        display_name="OpenRouter Chat",
        capabilities=["chat", "structured_output"],
    )
    action_choices_started = asyncio.Event()
    post_turn_started = asyncio.Event()

    class BlockingCyoaChatProvider(RecordingCyoaChatProvider):
        async def generate_structured_output(
            self,
            request: StructuredOutputRequest,
        ) -> StructuredOutputResponse:
            if request.schema_name == "action_choices":
                action_choices_started.set()
                await post_turn_started.wait()
            return await super().generate_structured_output(request)

    provider = BlockingCyoaChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    async def post_turn_step(**_kwargs: object) -> None:
        post_turn_started.set()
        await action_choices_started.wait()

    monkeypatch.setattr(service, "run_post_turn_jobs", post_turn_step)

    asyncio.run(
        asyncio.wait_for(
            service.submit_player_turn(
                save_id=save.id,
                body="I inspect the nearest blue shelf.",
                speaker_name="Courier",
            ),
            timeout=1.0,
        )
    )

    assert action_choices_started.is_set()
    assert post_turn_started.is_set()
    assert [
        choice.body for choice in repositories.latest_message_action_choices(save.id)
    ] == [
        "Open the brass atlas.",
        "Question the librarian.",
        "Hide the index under your coat.",
        "Step through the blue shelf-door.",
    ]


def test_submit_player_turn_does_not_generate_action_choices_for_non_cyoa_saves(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"opening_message": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="openrouter-chat",
    )
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="openrouter",
        model_id="openrouter-chat",
    )
    provider = RecordingCyoaChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=None,
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
        )
    )

    assert provider.structured_output_requests == []
    assert repositories.latest_message_action_choices(save.id) == []


def test_submit_player_turn_preserves_player_message_when_provider_chat_fails(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = FailingChatProvider(
        "openrouter",
        RuntimeError("narrator backend is down"),
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    with pytest.raises(RuntimeError, match="narrator backend is down"):
        asyncio.run(
            service.submit_player_turn(
                save_id=save.id,
                body="I climb toward the beacon lens.",
                speaker_name="Mara",
            )
        )

    assert len(provider.chat_requests) == 1
    persisted_messages = repositories.list_messages(save.id)
    assert len(persisted_messages) == 1
    assert persisted_messages[0].role == "player"
    assert persisted_messages[0].speaker_name == "Mara"
    assert persisted_messages[0].body == "I climb toward the beacon lens."
    assert [
        message
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    ] == []


def test_submit_player_turn_persists_failed_chat_completion_job_without_secret(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = FailingChatProvider(
        "openrouter",
        RuntimeError("narrator backend rejected sk-live-secret"),
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    with pytest.raises(RuntimeError, match="narrator backend rejected"):
        asyncio.run(
            service.submit_player_turn(
                save_id=save.id,
                body="I climb toward the beacon lens.",
                speaker_name="Mara",
            )
        )

    persisted_messages = repositories.list_messages(save.id)
    assert len(persisted_messages) == 1
    assert persisted_messages[0].role == "player"
    assert persisted_messages[0].body == "I climb toward the beacon lens."
    assert [
        message
        for message in repositories.list_messages(save.id)
        if message.role == "narrator"
    ] == []

    jobs = _chat_completion_jobs(repositories, save.id)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["status"] == "failed"
    assert job["result"] == {
        "original_provider": "openrouter",
        "original_model": "anthropic/claude-3.5-sonnet",
        "fallback_used": False,
    }
    assert job["payload"]["provider"] == "openrouter"
    assert job["payload"]["model"] == "anthropic/claude-3.5-sonnet"
    assert job["payload"]["player_message_id"] == persisted_messages[0].id
    assert "narrator backend rejected" in job["error"]
    assert "[redacted]" in job["error"]
    assert "sk-live-secret" not in job["error"]
    assert "sk-live-secret" not in repr(job["result"])


def test_submit_player_turn_records_exhausted_retry_diagnostics(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = FailingChatProvider(
        "openrouter",
        _provider_error_with_retry_attempts(
            category=ProviderErrorCategory.RATE_LIMITED,
            message="rate_limited (429)",
            status_code=429,
            retry_attempt_count=3,
            max_retry_attempts=3,
            retry_attempts=(
                {
                    "attempt": 1,
                    "error_category": "rate_limited",
                    "duration_ms": 141,
                    "http_status": 429,
                },
                {
                    "attempt": 2,
                    "error_category": "rate_limited",
                    "duration_ms": 153,
                },
                {
                    "attempt": 3,
                    "error_category": "rate_limited",
                    "duration_ms": 167,
                    "http_status": 429,
                },
            ),
        ),
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    with pytest.raises(ProviderError):
        asyncio.run(
            service.submit_player_turn(
                save_id=save.id,
                body="I climb toward the beacon lens.",
                speaker_name="Mara",
            )
        )

    jobs = _chat_completion_jobs(repositories, save.id)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["status"] == "failed"
    assert job["result"] == {
        "original_provider": "openrouter",
        "original_model": "anthropic/claude-3.5-sonnet",
        "fallback_used": False,
        "final_error_category": "rate_limited",
        "final_http_status": 429,
        "attempt_count": 3,
        "max_attempts": 3,
        "retry_attempts": [
            {
                "attempt": 1,
                "duration_ms": 141,
                "error_category": "rate_limited",
                "http_status": 429,
            },
            {
                "attempt": 2,
                "duration_ms": 153,
                "error_category": "rate_limited",
            },
            {
                "attempt": 3,
                "duration_ms": 167,
                "error_category": "rate_limited",
                "http_status": 429,
            },
        ],
    }


def test_submit_player_turn_fails_blank_narrator_completion_without_persisting_it(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = StaticChatProvider("openrouter", " \n\t ")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    with pytest.raises(Exception) as exc_info:
        asyncio.run(
            service.submit_player_turn(
                save_id=save.id,
                body="I climb toward the beacon lens.",
                speaker_name="Mara",
            )
        )

    assert len(provider.chat_requests) == 1
    assert any(word in str(exc_info.value).casefold() for word in ("blank", "empty"))
    persisted_messages = repositories.list_messages(save.id)
    assert len(persisted_messages) == 1
    assert persisted_messages[0].role == "player"
    assert persisted_messages[0].speaker_name == "Mara"
    assert persisted_messages[0].body == "I climb toward the beacon lens."
    assert [
        message for message in persisted_messages if message.role == "narrator"
    ] == []

    jobs = _chat_completion_jobs(repositories, save.id)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["status"] == "failed"
    assert job["result"]["original_provider"] == "openrouter"
    assert job["result"]["original_model"] == "anthropic/claude-3.5-sonnet"
    assert job["result"]["fallback_used"] is False
    assert job["result"]["fallback_skipped_reason"] == "no_fallback_model"
    assert job["result"]["final_provider"] == "openrouter"
    assert job["result"]["final_model"] == "anthropic/claude-3.5-sonnet"
    assert job["result"]["classification"] == "suspected_blocked_output"
    assert job["payload"]["player_message_id"] == persisted_messages[0].id
    assert job["error"] is not None
    assert any(word in job["error"].casefold() for word in ("blank", "empty"))


def test_submit_player_turn_uses_fallback_for_blank_primary_response(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    repositories.set_model_preference(
        task="chat_fallback",
        provider="venice",
        model_id="venice/fallback-chat",
    )
    repositories.save_provider_model(
        provider="venice",
        model_id="venice/fallback-chat",
        display_name="Venice Fallback Chat",
        capabilities=["chat", "fallback_marker"],
        context_window=8192,
    )
    repositories.set_app_setting("chat_fallback_enabled", True)
    primary = StaticChatProvider("openrouter", " \n\t ")
    fallback = StaticChatProvider(
        "venice",
        "The fallback narrator answers in natural prose.",
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": primary, "venice": fallback},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert len(primary.chat_requests) == 1
    assert len(fallback.chat_requests) == 1
    fallback_request = fallback.chat_requests[0]
    assert fallback_request.provider == "venice"
    assert fallback_request.model_id == "venice/fallback-chat"
    assert result.narrator_message.body == (
        "The fallback narrator answers in natural prose."
    )
    assert result.narrator_message.provider == "venice"
    assert result.narrator_message.model == "venice/fallback-chat"

    jobs = _chat_completion_jobs(repositories, save.id)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["status"] == "succeeded"
    assert job["payload"]["provider"] == "openrouter"
    assert job["payload"]["model"] == "anthropic/claude-3.5-sonnet"
    assert job["result"]["provider"] == "venice"
    assert job["result"]["model"] == "venice/fallback-chat"
    assert job["result"]["original_provider"] == "openrouter"
    assert job["result"]["original_model"] == "anthropic/claude-3.5-sonnet"
    assert job["result"]["fallback_used"] is True
    assert job["result"]["fallback_provider"] == "venice"
    assert job["result"]["fallback_model"] == "venice/fallback-chat"
    assert job["result"]["final_provider"] == "venice"
    assert job["result"]["final_model"] == "venice/fallback-chat"
    assert job["result"]["classification"] == "suspected_blocked_output"


def test_plan_first_narrator_request_is_preserved_for_fallback_model(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_app_setting(PLAN_FIRST_NARRATOR_SETTING, True)
    repositories.set_app_setting("chat_fallback_enabled", True)
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    repositories.set_model_preference(
        task="chat_fallback",
        provider="venice",
        model_id="venice/fallback-chat",
    )
    repositories.save_provider_model(
        provider="venice",
        model_id="venice/fallback-chat",
        display_name="Venice Fallback Chat",
        capabilities=["chat", "fallback_marker"],
        context_window=8192,
    )
    planner = ScriptedNarratorPlanner(
        NarratorMessageSpec(
            intent="Answer the climb.",
            thesis="The lens warning dominates the beat.",
            must_say=("The lens burns red.",),
            avoid=(),
            tone="tense",
            uncertainties=(),
            evidence_source_ids=("message:latest",),
        )
    )
    primary = StaticChatProvider("openrouter", " \n\t ")
    fallback = StaticChatProvider(
        "venice",
        "The fallback narrator answers in natural prose.",
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": primary, "venice": fallback},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(
                selected_memories=(
                    SelectedContextItem(
                        source_type="memory",
                        source_id="memory-fuse",
                        text="Mara already checked the fuse.",
                        relevance_note="Avoid repeating the fuse inspection.",
                    ),
                ),
            )
        ),
        narrator_planner=planner,
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
        )
    )

    assert primary.chat_requests[0].narrator_prompt_mode == "plan_first"
    fallback_request = fallback.chat_requests[0]
    assert fallback_request.narrator_prompt_mode == "plan_first"
    assert fallback_request.narration_brief
    assert fallback_request.retrieved_memories
    assert fallback_request.context_breakdown["narrator_context_policy"] == (
        "plan_plus_context"
    )
    assert fallback_request.provider == "venice"
    assert fallback_request.model_id == "venice/fallback-chat"
    job = _chat_completion_jobs(repositories, save.id)[-1]
    assert job["result"]["fallback_used"] is True
    assert job["result"]["narrator_mode"] == "plan_first"


def test_chat_fallback_rebudgets_from_untrimmed_primary_request(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.save_provider_model(
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
        display_name="Claude 3.5 Sonnet",
        capabilities=["chat"],
        context_window=1300,
    )
    repositories.save_provider_model(
        provider="venice",
        model_id="venice/fallback-chat",
        display_name="Venice Fallback Chat",
        capabilities=["chat", "fallback_marker"],
        context_window=8192,
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    repositories.set_model_preference(
        task="chat_fallback",
        provider="venice",
        model_id="venice/fallback-chat",
    )
    repositories.set_app_setting("chat_fallback_enabled", True)
    selected_state = (
        "[world_state:state-lens-fuse] "
        "beacon.fuse: The spare fuse is under the red lens."
    )
    primary = StaticChatProvider("openrouter", " \n\t ")
    fallback = StaticChatProvider(
        "venice",
        "The fallback narrator answers in natural prose.",
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": primary, "venice": fallback},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(
                selected_state=(
                    SelectedContextItem(
                        source_type="world_state",
                        source_id="state-lens-fuse",
                        text="beacon.fuse: The spare fuse is under the red lens.",
                        relevance_note="The player is asking about the lens.",
                    ),
                ),
            ),
        ),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert primary.chat_requests[0].retrieved_state == ()
    primary_budget = primary.chat_requests[0].context_breakdown["final_prompt_budget"]
    assert primary_budget["model_context_window"] == 1300
    assert primary_budget["trimmed"] is True
    assert fallback.chat_requests[0].retrieved_state == (selected_state,)
    fallback_budget = fallback.chat_requests[0].context_breakdown[
        "final_prompt_budget"
    ]
    assert fallback_budget["model_context_window"] == 8192
    assert fallback_budget["trimmed"] is False


def test_submit_player_turn_prefers_narrator_fallback_over_text_fallback(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    repositories.set_model_preference(
        task="narrator_fallback",
        provider="narrator",
        model_id="narrator/fallback-chat",
    )
    repositories.set_model_preference(
        task="chat_fallback",
        provider="background",
        model_id="background/fallback-chat",
    )
    for provider, model_id in [
        ("narrator", "narrator/fallback-chat"),
        ("background", "background/fallback-chat"),
    ]:
        repositories.save_provider_model(
            provider=provider,
            model_id=model_id,
            display_name="Fallback Chat",
            capabilities=["chat"],
            context_window=8192,
        )
    repositories.set_app_setting("chat_fallback_enabled", True)
    primary = StaticChatProvider("openrouter", " \n\t ")
    narrator_fallback = StaticChatProvider(
        "narrator",
        "The narrator fallback answers in natural prose.",
    )
    background_fallback = StaticChatProvider(
        "background",
        "The background fallback should not narrate.",
    )
    service = ChatService(
        repositories=repositories,
        providers={
            "openrouter": primary,
            "narrator": narrator_fallback,
            "background": background_fallback,
        },
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert len(narrator_fallback.chat_requests) == 1
    assert background_fallback.chat_requests == []
    assert result.narrator_message.provider == "narrator"
    assert result.narrator_message.model == "narrator/fallback-chat"


def test_submit_player_turn_uses_fallback_for_fast_provider_retries(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    repositories.set_model_preference(
        task="chat_fallback",
        provider="venice",
        model_id="venice/fallback-chat",
    )
    repositories.save_provider_model(
        provider="venice",
        model_id="venice/fallback-chat",
        display_name="Venice Fallback Chat",
        capabilities=["chat", "fallback_marker"],
        context_window=8192,
    )
    repositories.set_app_setting("chat_fallback_enabled", True)
    primary = FailingChatProvider(
        "openrouter",
        _provider_error_with_retry_attempts(
            category=ProviderErrorCategory.PROVIDER_ERROR,
            message="provider_error (500)",
            status_code=500,
            retry_attempt_count=3,
            max_retry_attempts=3,
            retry_attempts=(
                {
                    "attempt": 1,
                    "error_category": "provider_error",
                    "duration_ms": 28,
                    "http_status": 500,
                },
                {
                    "attempt": 2,
                    "error_category": "provider_error",
                    "duration_ms": 34,
                    "http_status": 500,
                },
                {
                    "attempt": 3,
                    "error_category": "provider_error",
                    "duration_ms": 31,
                    "http_status": 500,
                },
            ),
        ),
    )
    fallback = StaticChatProvider(
        "venice",
        "The fallback narrator answers in natural prose.",
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": primary, "venice": fallback},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert len(primary.chat_requests) == 1
    assert len(fallback.chat_requests) == 1
    assert result.narrator_message.body == (
        "The fallback narrator answers in natural prose."
    )
    assert result.narrator_message.provider == "venice"
    assert result.narrator_message.model == "venice/fallback-chat"

    jobs = _chat_completion_jobs(repositories, save.id)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["status"] == "succeeded"
    assert job["result"]["fallback_used"] is True
    assert job["result"]["fallback_provider"] == "venice"
    assert job["result"]["fallback_model"] == "venice/fallback-chat"
    assert job["result"]["classification"] == "suspected_blocked_output"
    assert job["result"]["attempt_count"] == 3
    assert job["result"]["max_attempts"] == 3


@pytest.mark.parametrize(
    ("capabilities", "case_id"),
    [
        (["fallback_marker"], "missing_chat"),
    ],
    ids=["missing-chat"],
)
def test_submit_player_turn_skips_fallback_for_missing_base_capability(
    repositories: PersistenceRepositories,
    capabilities: list[str],
    case_id: str,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    repositories.set_model_preference(
        task="chat_fallback",
        provider="venice",
        model_id="venice/fallback-chat",
    )
    repositories.save_provider_model(
        provider="venice",
        model_id="venice/fallback-chat",
        display_name=f"Venice Fallback Chat {case_id}",
        capabilities=capabilities,
        context_window=8192,
    )
    repositories.set_app_setting("chat_fallback_enabled", True)
    primary = StaticChatProvider("openrouter", " \n\t ")
    fallback = StaticChatProvider(
        "venice",
        "The fallback narrator answers in natural prose.",
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": primary, "venice": fallback},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    with pytest.raises(Exception) as exc_info:
        asyncio.run(
            service.submit_player_turn(
                save_id=save.id,
                body="I climb toward the beacon lens.",
                speaker_name="Mara",
            )
        )

    assert len(primary.chat_requests) == 1
    assert any(word in str(exc_info.value).casefold() for word in ("blank", "empty"))
    persisted_messages = repositories.list_messages(save.id)
    assert len(persisted_messages) == 1
    assert persisted_messages[0].role == "player"
    assert [
        message for message in persisted_messages if message.role == "narrator"
    ] == []

    jobs = _chat_completion_jobs(repositories, save.id)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["status"] == "failed"
    assert job["result"]["original_provider"] == "openrouter"
    assert job["result"]["original_model"] == "anthropic/claude-3.5-sonnet"
    assert job["result"]["fallback_used"] is False
    assert job["result"]["fallback_skipped_reason"] == (
        "fallback_model_lacks_required_capabilities"
    )
    assert job["result"]["final_provider"] == "openrouter"
    assert job["result"]["final_model"] == "anthropic/claude-3.5-sonnet"
    assert job["result"]["classification"] == "suspected_blocked_output"


def test_submit_player_turn_skips_unavailable_fallback_model(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    repositories.set_model_preference(
        task="chat_fallback",
        provider="venice",
        model_id="venice/fallback-chat",
    )
    repositories.save_provider_model(
        provider="venice",
        model_id="venice/fallback-chat",
        display_name="Venice Fallback Chat",
        capabilities=["chat", "fallback_marker"],
        context_window=8192,
    )
    repositories.mark_missing_provider_models_unavailable(
        provider="venice",
        available_model_ids=set(),
    )
    repositories.set_app_setting("chat_fallback_enabled", True)
    primary = StaticChatProvider("openrouter", " \n\t ")
    fallback = StaticChatProvider(
        "venice",
        "The fallback narrator answers in natural prose.",
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": primary, "venice": fallback},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    with pytest.raises(Exception) as exc_info:
        asyncio.run(
            service.submit_player_turn(
                save_id=save.id,
                body="I climb toward the beacon lens.",
                speaker_name="Mara",
            )
        )

    assert len(primary.chat_requests) == 1
    assert any(word in str(exc_info.value).casefold() for word in ("blank", "empty"))
    jobs = _chat_completion_jobs(repositories, save.id)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["status"] == "failed"
    assert job["result"]["fallback_used"] is False
    assert job["result"]["fallback_skipped_reason"] == "fallback_model_unavailable"


def test_submit_player_turn_records_failed_fallback_marker(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    repositories.set_model_preference(
        task="chat_fallback",
        provider="venice",
        model_id="venice/fallback-chat",
    )
    repositories.save_provider_model(
        provider="venice",
        model_id="venice/fallback-chat",
        display_name="Venice Fallback Chat",
        capabilities=["chat", "fallback_marker"],
        context_window=8192,
    )
    repositories.set_app_setting("chat_fallback_enabled", True)
    primary = StaticChatProvider("openrouter", " \n\t ")
    fallback = FailingChatProvider(
        "venice",
        ProviderError(
            category=ProviderErrorCategory.PROVIDER_ERROR,
            message="provider_error (500)",
            status_code=500,
        ),
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": primary, "venice": fallback},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    with pytest.raises(Exception, match="provider_error"):
        asyncio.run(
            service.submit_player_turn(
                save_id=save.id,
                body="I climb toward the beacon lens.",
                speaker_name="Mara",
            )
        )

    assert len(primary.chat_requests) == 1
    assert len(fallback.chat_requests) == 1
    jobs = _chat_completion_jobs(repositories, save.id)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["status"] == "failed"
    assert job["result"]["original_provider"] == "openrouter"
    assert job["result"]["original_model"] == "anthropic/claude-3.5-sonnet"
    assert job["result"]["fallback_used"] is True
    assert job["result"]["fallback_provider"] == "venice"
    assert job["result"]["fallback_model"] == "venice/fallback-chat"
    assert job["result"]["classification"] == "suspected_blocked_output"
    assert job["result"]["final_error_category"] == "provider_error"
    assert job["result"]["final_http_status"] == 500


def test_submit_player_turn_uses_fallback_for_content_safety_header(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    repositories.set_model_preference(
        task="chat_fallback",
        provider="venice",
        model_id="venice/fallback-chat",
    )
    repositories.save_provider_model(
        provider="venice",
        model_id="venice/fallback-chat",
        display_name="Venice Fallback Chat",
        capabilities=["chat", "fallback_marker"],
        context_window=8192,
    )
    repositories.set_app_setting("chat_fallback_enabled", True)
    primary = StaticChatProvider(
        "openrouter",
        "The primary narrator returns text flagged by provider safety metadata.",
        raw_metadata={
            "_bragi_headers": {"x-venice-is-content-violation": "true"},
        },
    )
    fallback = StaticChatProvider(
        "venice",
        "The fallback narrator answers in natural prose.",
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": primary, "venice": fallback},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert len(primary.chat_requests) == 1
    assert len(fallback.chat_requests) == 1
    assert result.narrator_message.body == (
        "The fallback narrator answers in natural prose."
    )
    assert result.narrator_message.provider == "venice"
    assert result.narrator_message.model == "venice/fallback-chat"

    jobs = _chat_completion_jobs(repositories, save.id)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["status"] == "succeeded"
    assert job["result"]["original_provider"] == "openrouter"
    assert job["result"]["original_model"] == "anthropic/claude-3.5-sonnet"
    assert job["result"]["fallback_used"] is True
    assert job["result"]["fallback_provider"] == "venice"
    assert job["result"]["fallback_model"] == "venice/fallback-chat"
    assert job["result"]["classification"] == "suspected_blocked_output"


def test_submit_player_turn_uses_fallback_marker_when_toggle_false(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    repositories.set_model_preference(
        task="chat_fallback",
        provider="venice",
        model_id="venice/fallback-chat",
    )
    repositories.save_provider_model(
        provider="venice",
        model_id="venice/fallback-chat",
        display_name="Venice Fallback Chat",
        capabilities=["chat", "fallback_marker"],
    )
    repositories.set_app_setting("chat_fallback_enabled", False)
    primary = StaticChatProvider("openrouter", " \n\t ")
    fallback = StaticChatProvider(
        "venice",
        "The fallback narrator answers in natural prose.",
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": primary, "venice": fallback},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert len(primary.chat_requests) == 1
    assert len(fallback.chat_requests) == 1
    assert result.narrator_message.body == (
        "The fallback narrator answers in natural prose."
    )
    assert result.narrator_message.provider == "venice"
    assert result.narrator_message.model == "venice/fallback-chat"

    jobs = _chat_completion_jobs(repositories, save.id)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["status"] == "succeeded"
    assert job["result"]["original_provider"] == "openrouter"
    assert job["result"]["original_model"] == "anthropic/claude-3.5-sonnet"
    assert job["result"]["fallback_used"] is True
    assert job["result"]["fallback_provider"] == "venice"
    assert job["result"]["fallback_model"] == "venice/fallback-chat"
    assert job["result"]["classification"] == "suspected_blocked_output"


def test_submit_player_turn_skips_fallback_marker_without_preference(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    repositories.set_app_setting("chat_fallback_enabled", True)
    primary = StaticChatProvider("openrouter", " \n\t ")
    fallback = StaticChatProvider(
        "venice",
        "The fallback narrator answers in natural prose.",
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": primary, "venice": fallback},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    with pytest.raises(Exception) as exc_info:
        asyncio.run(
            service.submit_player_turn(
                save_id=save.id,
                body="I climb toward the beacon lens.",
                speaker_name="Mara",
            )
        )

    assert len(primary.chat_requests) == 1
    assert fallback.chat_requests == []
    assert any(word in str(exc_info.value).casefold() for word in ("blank", "empty"))
    persisted_messages = repositories.list_messages(save.id)
    assert len(persisted_messages) == 1
    assert persisted_messages[0].role == "player"
    assert [
        message for message in persisted_messages if message.role == "narrator"
    ] == []

    jobs = _chat_completion_jobs(repositories, save.id)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["status"] == "failed"
    assert job["result"]["original_provider"] == "openrouter"
    assert job["result"]["original_model"] == "anthropic/claude-3.5-sonnet"
    assert job["result"]["fallback_used"] is False
    assert job["result"]["fallback_skipped_reason"] == "no_fallback_model"
    assert job["result"]["final_provider"] == "openrouter"
    assert job["result"]["final_model"] == "anthropic/claude-3.5-sonnet"
    assert job["result"]["classification"] == "suspected_blocked_output"


def test_submit_player_turn_continues_when_optional_context_search_fails(
    repositories: PersistenceRepositories,
) -> None:
    long_lore = "The buried legion names every signal warden in a copper ledger."
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "starting_scene": "The beacon gutters in the tower.",
            "lore": long_lore,
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    context_search = FailingContextSearch(RuntimeError("context index unavailable"))
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=context_search,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert len(context_search.calls) == 1
    assert len(provider.chat_requests) == 1
    request = provider.chat_requests[0]
    assert "Ashfall Keep" in request.scenario_instructions
    assert "A border keep is cut off by ash storms." in request.scenario_instructions
    assert "Signal warden" in request.scenario_instructions
    assert long_lore not in request.scenario_instructions
    assert "Starting scene:" not in request.scenario_instructions
    persisted_messages = repositories.list_messages(save.id)
    assert len(persisted_messages) == 2
    assert persisted_messages[0].role == "player"
    assert persisted_messages[0].speaker_name == "Mara"
    assert persisted_messages[0].body == "I climb toward the beacon lens."
    assert persisted_messages[1] == result.narrator_message


def test_submit_player_turn_omits_initial_setup_after_opening_leaves_recent_window(
    repositories: PersistenceRepositories,
) -> None:
    long_premise = (
        "A long initial setup about lantern ferries, archive districts, and "
        "the discovery of a disputed star map."
    )
    long_tone = (
        "Warm archival mystery with a lengthy initial tone brief full of "
        "ferry bells, catalog disputes, and understated romance."
    )
    long_player_role = (
        "The player is Avery Quill, a fictional archive courier with an "
        "initial biography that should not ride along forever after setup."
    )
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Archive Arrival",
        premise=long_premise,
        player_role=long_player_role,
        content={
            "player_character_name": "Avery Quill",
            "tone_genre": long_tone,
            "current_scene": (
                "Avery is reviewing a map with Nira in the archive atrium."
            ),
        },
    )
    save = repositories.create_save(
        scenario_id=scenario.id,
        title="First Crossing",
    )
    repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Opening setup chronicle that has now aged out.",
    )
    repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Avery",
        body="I ask Nira what the map's missing mark means.",
    )
    recent_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Nira sets the brass compass on the archive table.",
    )
    repositories.set_app_setting(RECENT_NARRATOR_MESSAGE_WINDOW_SETTING, 1)
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I answer Nira honestly.",
            speaker_name="Avery",
        )
    )

    request = provider.chat_requests[0]
    assert "Title: Lantern Archive Arrival" in request.scenario_instructions
    assert "Player character name: Avery Quill" in request.scenario_instructions
    assert (
        "Current scene: Avery is reviewing a map with Nira in the archive atrium."
        in request.scenario_instructions
    )
    assert "Premise/setup:" not in request.scenario_instructions
    assert "Tone/style:" not in request.scenario_instructions
    assert "Player role:" not in request.scenario_instructions
    assert long_premise not in request.scenario_instructions
    assert long_tone not in request.scenario_instructions
    assert long_player_role not in request.scenario_instructions
    assert [
        message.body for message in request.messages if message.role == "narrator"
    ] == [recent_narrator.body]


def test_submit_player_turn_continues_when_context_search_is_rate_limited(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    context_search = FailingContextSearch(
        ProviderError(
            category=ProviderErrorCategory.RATE_LIMITED,
            message="rate_limited (429)",
            status_code=429,
        )
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=context_search,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert len(context_search.calls) == 1
    assert len(provider.chat_requests) == 1
    persisted_messages = repositories.list_messages(save.id)
    assert [message.role for message in persisted_messages] == ["player", "narrator"]
    assert persisted_messages[1] == result.narrator_message


def test_submit_player_turn_uses_degraded_recovered_context(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    source_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="A silver bell rings beneath the bridge.",
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="Mara distrusts bells that ring without wind.",
        tags=["bells"],
        source_message_id=source_message.id,
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-chat",
        display_name="Fake Chat",
        capabilities=[ProviderCapability.CHAT.value],
    )
    repositories.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=[ProviderCapability.TOOL_CALLING.value],
    )
    repositories.set_model_preference(
        task="tool_call_fallback",
        provider="fallback",
        model_id="fallback-tools",
    )
    repositories.save_provider_model(
        provider="fallback",
        model_id="fallback-tools",
        display_name="Fallback Tools",
        capabilities=[ProviderCapability.TOOL_CALLING.value],
    )
    primary = ToolContextAndChatProvider(
        "fake",
        context_model_id="fake-context",
        tool_error=ProviderError(
            ProviderErrorCategory.MODEL_NOT_FOUND,
            "model not found",
            status_code=404,
        ),
    )
    fallback = ToolContextAndChatProvider(
        "fallback",
        context_model_id="fallback-tools",
        tool_calls=(
            ProviderToolCall(
                id="fallback-memory",
                name="select_context_source",
                arguments_json=json.dumps(
                    {
                        "source_id": memory.id,
                        "relevance_note": "The bell concern matters now.",
                    }
                ),
            ),
        ),
    )
    context_search = ContextSearchService(
        repositories=repositories,
        providers={"fake": primary, "fallback": fallback},
    )
    service = ChatService(
        repositories=repositories,
        providers={"fake": primary, "fallback": fallback},
        context_search_service=context_search,
    )
    progress_events: list[Any] = []

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I listen for the bell.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
            turn_progress_callback=progress_events.append,
        )
    )

    assert len(primary.tool_call_requests) == 1
    assert len(fallback.tool_call_requests) == 1
    assert len(primary.chat_requests) == 1
    narrator_request = primary.chat_requests[0]
    assert any(memory.body in item for item in narrator_request.retrieved_memories)
    status_texts = [event.status_text for event in progress_events]
    assert "Context selected with degraded retrieval" in status_texts
    final_statuses = {
        job.name: job.status
        for job in progress_events[-1].jobs
    }
    assert final_statuses["context_selection"] == "degraded"
    job_result = _chat_completion_jobs(repositories, save.id)[-1]["result"]
    assert job_result["context_search_failed"] is False
    assert job_result["context_search_degraded"] is True
    assert job_result["context_search_recovery"] == "provider_fallback"
    assert job_result["context_search_selected_counts"]["memories"] == 1
    prompt_diagnostics = job_result["prompt_context_diagnostics"]
    assert prompt_diagnostics["context_search_failed"] is False
    assert prompt_diagnostics["context_search_degraded"] is True
    assert prompt_diagnostics["context_search_recovery"] == "provider_fallback"


def test_submit_player_turn_propagates_context_search_configuration_errors(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=FailingContextSearch(
            ValueError("Context-search model does not advertise structured output")
        ),
    )

    with pytest.raises(ValueError, match="structured output"):
        asyncio.run(
            service.submit_player_turn(
                save_id=save.id,
                body="I climb toward the beacon lens.",
                speaker_name="Mara",
            )
        )

    assert provider.chat_requests == []
    assert [message.role for message in repositories.list_messages(save.id)] == [
        "player"
    ]
    jobs = _chat_completion_jobs(repositories, save.id)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "failed"
    assert jobs[0]["result"] == {
        "player_message_id": repositories.list_messages(save.id)[0].id
    }
    assert "structured output" in jobs[0]["error"]


def test_existing_message_metadata_survives_chat_model_preference_change(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="The Sealed Archivist",
        premise="A forbidden archive interview.",
        player_role="Investigator",
        content={"opening_message": "Mael slides the forbidden index across."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Interview")
    openrouter = RecordingChatProvider("openrouter")
    venice = RecordingChatProvider("venice")
    service = ChatService(
        repositories=repositories,
        providers={
            "openrouter": openrouter,
            "venice": venice,
        },
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    first_turn = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="What is sealed in the red index?",
            speaker_name="Mara",
        )
    )

    repositories.set_model_preference(
        task="chat",
        provider="venice",
        model_id="venice/llama-3.3-70b",
    )
    second_turn = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="Then show me the stacks.",
            speaker_name="Mara",
        )
    )

    persisted_messages = repositories.list_messages(save.id)
    narrator_messages = [
        message for message in persisted_messages if message.role == "narrator"
    ]

    assert first_turn.narrator_message.provider == "openrouter"
    assert first_turn.narrator_message.model == "anthropic/claude-3.5-sonnet"
    assert second_turn.narrator_message.provider == "venice"
    assert second_turn.narrator_message.model == "venice/llama-3.3-70b"
    assert [(message.provider, message.model) for message in narrator_messages] == [
        ("openrouter", "anthropic/claude-3.5-sonnet"),
        ("venice", "venice/llama-3.3-70b"),
    ]
    assert [request.model_id for request in openrouter.chat_requests] == [
        "anthropic/claude-3.5-sonnet"
    ]
    assert [request.model_id for request in venice.chat_requests] == [
        "venice/llama-3.3-70b"
    ]


def test_submit_player_turn_runs_context_search_before_narrator_and_injects_context(
    repositories: PersistenceRepositories,
) -> None:
    selected_scenario_text = "The hidden lens code answers only a sung oath."
    starting_scene_text = "The beacon gutters in the tower."
    unselected_scenario_text = "The storm teeth scrape the shutters."
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "starting_scene": starting_scene_text,
            "lore": selected_scenario_text,
            "opening_message": unselected_scenario_text,
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    unselected_message = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I check the cold storeroom.",
    )
    selected_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Ash claws at the beacon lens.",
        provider="openrouter",
        model="anthropic/claude-3.5-sonnet",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    events: list[str] = []
    provider = RecordingChatProvider("openrouter")

    original_chat = provider.chat

    async def recording_chat(request: ChatRequest) -> ChatResponse:
        events.append("narrator_chat")
        return await original_chat(request)

    provider.chat = recording_chat  # type: ignore[method-assign]
    context_search = ScriptedContextSearch(
        ContextSearchResult(
            selected_state=(
                SelectedContextItem(
                    source_type="world_state",
                    source_id="state-scene-location",
                    text="scene.location: Beacon tower",
                    relevance_note="The player is interacting with the beacon.",
                ),
            ),
            selected_memories=(
                SelectedContextItem(
                    source_type="memory",
                    source_id="memory-promise",
                    text="Mara promised Elian she would keep the beacon lit.",
                    relevance_note="The new turn tests that promise.",
                ),
            ),
            selected_summaries=(
                SelectedContextItem(
                    source_type="summary",
                    source_id="summary-ash-storm",
                    text="The ash storm isolated the keep before dawn.",
                    relevance_note="Explains the immediate stakes.",
                ),
            ),
            selected_scenario_sections=(
                SelectedContextItem(
                    source_type="scenario_section",
                    source_id=f"scenario:{scenario.id}:section:lore",
                    text=selected_scenario_text,
                    relevance_note="The player is relighting the oath-bound beacon.",
                ),
            ),
            selected_recent_messages=(
                SelectedContextItem(
                    source_type="message",
                    source_id=selected_message.id,
                    text=selected_message.body,
                    relevance_note="The new turn responds to the lens.",
                ),
            ),
        ),
        events=events,
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=context_search,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I relight the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert events == ["context_search", "narrator_chat"]
    assert context_search.calls == [(save.id, result.player_message.id)]
    request = provider.chat_requests[0]
    assert "Ashfall Keep" in request.scenario_instructions
    assert selected_scenario_text not in request.scenario_instructions
    assert starting_scene_text not in request.scenario_instructions
    assert unselected_scenario_text not in request.scenario_instructions
    assert "never write dialogue, actions" in request.scenario_instructions
    assert "Only the user's submitted player messages define" in (
        request.scenario_instructions
    )
    assert "Preserve player agency by leaving those choices unresolved" in (
        request.scenario_instructions
    )
    assert "interrupt, demand, refuse, leave, escalate" in (
        request.scenario_instructions
    )
    assert "stop at the decision point" not in request.scenario_instructions
    assert request.retrieved_state == (
        "[world_state:state-scene-location] scene.location: Beacon tower",
    )
    assert request.retrieved_memories == (
        "[memory:memory-promise] Mara promised Elian she would keep the beacon lit.",
    )
    assert request.summary == (
        "[summary:summary-ash-storm] The ash storm isolated the keep before dawn."
    )
    assert request.retrieved_scenario_sections == (
        f"[scenario_section:scenario:{scenario.id}:section:lore] "
        "The hidden lens code answers only a sung oath.",
    )
    assert [message.body for message in request.messages] == [
        unselected_message.body,
        selected_message.body,
        "I relight the beacon lens.",
    ]


def test_submit_player_turn_injects_recent_character_texts_as_phone_context(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="First Frame",
        premise="Two students are figuring out where they stand.",
        player_role="James",
        content={
            "character_name": "Clara",
            "starting_scene": "Morning light spills into the dorm room.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="First Frame")
    player = repositories.add_character(
        save_id=save.id,
        name="James",
        role="player",
        is_player_character=True,
    )
    clara = repositories.add_character(
        save_id=save.id,
        name="Clara",
        role="roommate",
        met=True,
        contact_name="Clara",
    )
    repositories.upsert_character_contact_state(
        save_id=save.id,
        player_character_id=player.id,
        character_id=clara.id,
        player_has_character_number=True,
        character_has_player_number=True,
    )
    previous_player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="James",
        body="I head back to my room.",
    )
    previous_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Clara says goodnight at the door.",
        provider="openrouter",
        model="anthropic/claude-3.5-sonnet",
    )
    thread = repositories.get_or_create_character_text_thread(
        save_id=save.id,
        character_id=clara.id,
        title="Clara",
    )
    repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=clara.id,
        sender="player",
        body="Made it home safe. Sleep well.",
        in_world_sent_at="Sunday 11:47 PM",
    )
    repositories.append_character_text_message(
        save_id=save.id,
        thread_id=thread.id,
        character_id=clara.id,
        sender="character",
        body="Good. Text me in the morning, okay?",
        provider="fake",
        model="fake-chat",
        in_world_sent_at="Sunday 11:49 PM",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I wake up and check my phone for the time.",
            speaker_name="James",
        )
    )

    request = provider.chat_requests[0]
    assert [message.body for message in request.messages] == [
        previous_player.body,
        previous_narrator.body,
        result.player_message.body,
    ]
    phone_context = "\n".join(request.phone_context)
    assert "Phone thread: Clara" in phone_context
    assert "Made it home safe. Sleep well." in phone_context
    assert "Good. Text me in the morning, okay?" in phone_context
    assert "Narrator-only side-channel context" in phone_context
    assert request.context_breakdown["phone_context_thread_count"] == 1
    assert request.context_breakdown["phone_context_message_count"] == 2
    job_result = _chat_completion_jobs(repositories, save.id)[0]["result"]
    diagnostics = job_result["prompt_context_diagnostics"]
    assert diagnostics["retrieved_counts"]["phone_context"] == len(
        request.phone_context
    )
    assert diagnostics["phone_context_chars"] == sum(
        len(line) for line in request.phone_context
    )


def test_submit_player_turn_injects_selected_old_messages_as_retrieved_chronicle(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    selected_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="A prior lens flare revealed the hidden oath sigil.",
    )
    repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I check the cold storeroom.",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    repositories.set_app_setting(RECENT_PLAYER_MESSAGE_WINDOW_SETTING, 0)
    repositories.set_app_setting(RECENT_NARRATOR_MESSAGE_WINDOW_SETTING, 0)
    provider = RecordingChatProvider("openrouter")
    context_search = ScriptedContextSearch(
        ContextSearchResult(
            selected_recent_messages=(
                SelectedContextItem(
                    source_type="message",
                    source_id=selected_message.id,
                    text="Narrator: A prior lens flare revealed the hidden oath sigil.",
                    relevance_note="The new turn responds to the sigil.",
                ),
            ),
        )
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=context_search,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I relight the beacon lens.",
            speaker_name="Mara",
        )
    )

    request = provider.chat_requests[0]
    assert context_search.calls == [(save.id, result.player_message.id)]
    assert [message.body for message in request.messages] == [
        "I relight the beacon lens.",
    ]
    assert request.retrieved_recent_messages == (
        "[message:"
        f"{selected_message.id}] Narrator: A prior lens flare revealed the "
        "hidden oath sigil.",
    )


def test_submit_player_turn_injects_selected_state_changes_and_media_assets(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Blackwater Bridge",
        premise="A bridge changes shape when moon gates open.",
        player_role="Gatefinder",
        content={"starting_scene": "Black water moves below the bridge."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Gate Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    context_search = ScriptedContextSearch(
        ContextSearchResult(
            selected_state_changes=(
                SelectedContextItem(
                    source_type="state_change",
                    source_id="change-moon-gate",
                    text="scene.exit changed from Smoke Alley to Moon Gate",
                    relevance_note="The player is choosing that exit.",
                ),
            ),
            selected_media_assets=(
                SelectedContextItem(
                    source_type="media_asset",
                    source_id="media-bridge-lights",
                    text="Image prompt: gold bridge lights over black water",
                    relevance_note="The player referenced the latest image.",
                ),
            ),
        )
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=context_search,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I step toward the lit arch from the image.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
        )
    )

    assert context_search.calls == [(save.id, result.player_message.id)]
    request = provider.chat_requests[0]
    assert request.retrieved_state_changes == (
        "[state_change:change-moon-gate] "
        "scene.exit changed from Smoke Alley to Moon Gate",
    )
    assert request.retrieved_media_assets == (
        "[media_asset:media-bridge-lights] "
        "Image prompt: gold bridge lights over black water",
    )


def test_submit_player_turn_includes_active_linked_facts_in_narrator_recap(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    location = repositories.add_location(
        save_id=save.id,
        name="Lens Gallery",
        description="A brass-ringed gallery above the beacon chamber.",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
        situation="Mara studies the beacon controls.",
    )
    linked_fact = "Captain Ilyra knows the lens-key phrase: ember dawn."
    memory = repositories.add_memory(
        save_id=save.id,
        body=linked_fact,
        tags=["ilyra", "lens-key"],
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="location",
        entity_id=location.id,
        target_type="memory",
        target_id=memory.id,
        relation="location clue",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I inspect the lens housing.",
            speaker_name="Mara",
        )
    )

    request = provider.chat_requests[0]
    assert f"Linked memory: {linked_fact}" in "\n".join(request.current_scene_recap)
    assert request.retrieved_memories == ()
    linked_fact_breakdown = next(
        source
        for source in request.context_breakdown["sources"]
        if source["source_id"] == memory.id
    )
    assert linked_fact_breakdown["tier"] == "active_linked_facts"
    assert linked_fact_breakdown["included"] is True


def test_submit_player_turn_adds_read_only_pre_turn_scene_hints(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    present = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        aliases=["Ilyra"],
        role="Watch captain",
        status="guarding the cracked lens",
        met=True,
    )
    absent = repositories.add_character(
        save_id=save.id,
        name="Archivist Lio",
        aliases=["Lio"],
        role="Archivist",
        status="missing from the gallery",
        met=True,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="The beacon lens hums under stress.",
        nearby_objects=["signal horn"],
        hazards=["cracked lens"],
        present_character_ids=[present.id],
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body=(
                "I ask Ilyra if Lio touched the signal horn beside the "
                "cracked lens."
            ),
            speaker_name="Mara",
            run_post_turn_jobs=False,
        )
    )

    request = provider.chat_requests[0]
    current_scene_text = "\n".join(request.current_scene_recap)
    assert "mentions present character Captain Ilyra" in current_scene_text
    assert "mentions known character Archivist Lio" in current_scene_text
    assert "not marked present in the current scene" in current_scene_text
    assert "references current nearby object: signal horn" in current_scene_text
    assert "references current hazard: cracked lens" in current_scene_text
    assert repositories.list_world_state(save.id) == []
    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.present_character_ids == [present.id]
    assert absent.id not in snapshot.present_character_ids
    hint_breakdowns = [
        source
        for source in request.context_breakdown["sources"]
        if source["tier"] == "pre_turn_scene_hints"
    ]
    assert len(hint_breakdowns) == 4
    assert all(
        source["source_type"] == "pre_turn_scene_hint"
        for source in hint_breakdowns
    )


def test_submit_player_turn_runs_character_action_planning_before_prompt(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"player_character_name": "Ily"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player = repositories.add_character(
        save_id=save.id,
        name="Ily",
        met=True,
        is_player_character=True,
    )
    mara = repositories.add_character(
        save_id=save.id,
        name="Mara",
        met=True,
        current_intent="guard the signal lantern",
    )
    lio = repositories.add_character(
        save_id=save.id,
        name="Archivist Lio",
        aliases=["Lio"],
        met=True,
        voice="careful, archival precision",
        personality="curious but wary",
    )
    original_snapshot = repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara waits beside the beacon controls.",
        present_character_ids=[mara.id, lio.id],
    )
    assert original_snapshot is not None
    repositories.set_app_setting(PLAN_FIRST_NARRATOR_SETTING, False)
    repositories.set_app_setting(CHARACTER_ACTION_PLANNING_ENABLED_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="fake",
        model_id="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-chat",
        display_name="Fake Chat",
        capabilities=["chat", "structured_output"],
    )
    provider = RecordingCharacterActionChatProvider(
        "fake",
        {
            "Mara": {
                "present": True,
                "action": "Mara lowers the lantern and listens for the reply.",
                "intent": "protect the lens crew",
                "reason": "Mara is in the current scene by the controls.",
                "confidence": 0.9,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
            },
            "Archivist Lio": {
                "present": False,
                "action": "",
                "intent": "",
                "reason": "Lio is not in the gallery.",
                "confidence": 0.8,
                "evidence_source_ids": ["character:lio"],
            },
        },
    )
    context_search = ScriptedContextSearch(
        ContextSearchResult(
            selected_character_voice=(
                SelectedContextItem(
                    source_type="character_voice",
                    source_id=lio.id,
                    text=(
                        "Archivist Lio voice profile: voice: careful archival "
                        "precision."
                    ),
                    relevance_note="Lio was mentioned.",
                ),
            ),
        )
    )

    asyncio.run(
        ChatService(
            repositories=repositories,
            providers={"fake": provider},
            context_search_service=context_search,
        ).submit_player_turn(
            save_id=save.id,
            body="I ask Mara what she sees and whether Lio is nearby.",
            speaker_name="Ily",
            run_post_turn_jobs=False,
        )
    )

    request = provider.chat_requests[0]
    assert provider.events == [
        "character_presence_assessment",
        "character_presence_assessment",
        "character_intent_plan",
        "chat",
    ]
    assert request.character_action_plans == (
        "[character_action:"
        f"{mara.id}] Mara | intent: protect the lens crew | next action: "
        "Mara lowers the lantern and listens for the reply. | reason: Mara is "
        "in the current scene by the controls. | confidence: 90% | evidence: "
        f"scene_snapshot:{original_snapshot.id}",
    )
    current_snapshot = repositories.get_scene_snapshot(save.id)
    assert current_snapshot is not None
    assert set(current_snapshot.present_character_ids) == {player.id, mara.id}
    assert lio.id not in current_snapshot.present_character_ids
    assert request.character_voice_profiles == ()


def test_submit_player_turn_keeps_voice_profile_for_ungrounded_absence(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"player_character_name": "Ily"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player = repositories.add_character(
        save_id=save.id,
        name="Ily",
        met=True,
        is_player_character=True,
    )
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    lio = repositories.add_character(
        save_id=save.id,
        name="Archivist Lio",
        aliases=["Lio"],
        met=True,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara waits beside the beacon controls.",
        present_character_ids=[mara.id],
    )
    repositories.set_app_setting(PLAN_FIRST_NARRATOR_SETTING, False)
    repositories.set_app_setting(CHARACTER_ACTION_PLANNING_ENABLED_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="fake",
        model_id="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-chat",
        display_name="Fake Chat",
        capabilities=["chat", "structured_output"],
    )
    provider = RecordingCharacterActionChatProvider(
        "fake",
        {
            "Mara": {
                "present": True,
                "action": "Mara listens for the reply.",
                "intent": "protect the lens crew",
                "reason": "Mara is in the current scene by the controls.",
                "confidence": 0.9,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
            },
            "Archivist Lio": {
                "present": False,
                "action": "",
                "intent": "",
                "reason": "Lio is claimed absent without grounded evidence.",
                "confidence": 0.8,
                "evidence_source_ids": ["scene_snapshot:snapshot-1"],
                "evidence_quote": "ruby library",
            },
        },
    )
    context_search = ScriptedContextSearch(
        ContextSearchResult(
            selected_character_voice=(
                SelectedContextItem(
                    source_type="character_voice",
                    source_id=lio.id,
                    text="Archivist Lio voice profile: careful archival precision.",
                    relevance_note="Lio was mentioned.",
                ),
            ),
        )
    )

    asyncio.run(
        ChatService(
            repositories=repositories,
            providers={"fake": provider},
            context_search_service=context_search,
        ).submit_player_turn(
            save_id=save.id,
            body="I ask Mara what she sees and whether Lio is nearby.",
            speaker_name="Ily",
            run_post_turn_jobs=False,
        )
    )

    request = provider.chat_requests[0]
    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert player.id in snapshot.present_character_ids
    assert any(
        "Archivist Lio voice profile" in item
        for item in request.character_voice_profiles
    )


def test_submit_player_turn_overlaps_plan_first_character_planning_and_context(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"player_character_name": "Ily"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.add_character(
        save_id=save.id,
        name="Ily",
        met=True,
        is_player_character=True,
    )
    repositories.add_character(save_id=save.id, name="Mara", met=True)
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_app_setting(PLAN_FIRST_NARRATOR_SETTING, True)
    repositories.set_app_setting(CHARACTER_ACTION_PLANNING_ENABLED_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    planning_started = asyncio.Event()
    context_started = asyncio.Event()

    class BlockingPlanFirstCharacterPlanner:
        def __init__(self) -> None:
            self.apply_presence_updates: list[bool] = []

        async def plan_for_turn(
            self,
            *,
            save_id: str,
            player_message_id: str,
            apply_presence_updates: bool = True,
        ) -> CharacterActionPlanningResult:
            self.apply_presence_updates.append(apply_presence_updates)
            planning_started.set()
            await context_started.wait()
            return CharacterActionPlanningResult(skipped_reason="test")

    class BlockingPlanFirstContextSearch:
        async def search(
            self,
            *,
            save_id: str,
            player_message_id: str,
        ) -> ContextSearchResult:
            context_started.set()
            await planning_started.wait()
            return ContextSearchResult()

    character_planner = BlockingPlanFirstCharacterPlanner()
    provider = RecordingChatProvider("fake")
    narrator_planner = ScriptedNarratorPlanner(
        NarratorMessageSpec(
            intent="Answer the player.",
            thesis="Mara responds.",
            must_say=(),
            avoid=(),
            tone="grounded",
            uncertainties=(),
            evidence_source_ids=(),
        )
    )

    asyncio.run(
        asyncio.wait_for(
            ChatService(
                repositories=repositories,
                providers={"fake": provider},
                context_search_service=BlockingPlanFirstContextSearch(),
                narrator_planner=narrator_planner,
                character_action_planning_service=character_planner,
            ).submit_player_turn(
                save_id=save.id,
                body="I ask Mara what she sees.",
                speaker_name="Ily",
                run_post_turn_jobs=False,
            ),
            timeout=1.0,
        )
    )

    assert character_planner.apply_presence_updates == [False]
    assert len(narrator_planner.calls) == 1
    assert provider.chat_requests[0].narrator_prompt_mode == "plan_first"


def test_submit_player_turn_feeds_richer_character_assessments_to_planner(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"player_character_name": "Ily"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.add_character(
        save_id=save.id,
        name="Ily",
        met=True,
        is_player_character=True,
    )
    mara = repositories.add_character(
        save_id=save.id,
        name="Mara",
        met=True,
        current_intent="guard the signal lantern",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara waits beside the beacon controls.",
        present_character_ids=[mara.id],
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="The beacon lens answers to ember dawn.",
        tags=["beacon"],
        memory_id="memory-beacon-key",
    )
    repositories.set_app_setting(CHARACTER_ACTION_PLANNING_ENABLED_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="fake",
        model_id="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-chat",
        display_name="Fake Chat",
        capabilities=["chat", "structured_output"],
    )
    provider = RecordingCharacterActionChatProvider(
        "fake",
        {
            "Mara": {
                "present": True,
                "action": "Mara pockets the phrase and watches the lens.",
                "intent": "remember the beacon phrase",
                "reason": "The player directly tells Mara the phrase.",
                "confidence": 0.89,
                "evidence_source_ids": ["message:latest"],
                "learned_memory_candidates": [
                    {
                        "body": "Mara learned that ember dawn wakes the beacon lens.",
                        "tags": ["mara", "beacon"],
                        "knowledge_state": "knows",
                        "acquisition_method": "told",
                        "reason": "The player told Mara directly.",
                        "confidence": 0.87,
                        "evidence_source_ids": ["message:latest"],
                        "evidence_quote": "ember dawn wakes the beacon lens",
                    }
                ],
                "knowledge_edge_candidates": [
                    {
                        "target_type": "memory",
                        "target_id": memory.id,
                        "knowledge_state": "knows",
                        "acquisition_method": "told",
                        "reason": "The source message teaches this memory.",
                        "confidence": 0.84,
                        "evidence_source_ids": ["message:latest"],
                        "evidence_quote": "ember dawn",
                    }
                ],
                "needs_review_notes": [
                    "Review before making Mara's learned phrase durable."
                ],
            }
        },
    )
    planner = ScriptedNarratorPlanner(
        NarratorMessageSpec(
            intent="Answer the player's move.",
            thesis="Mara reacts to the beacon phrase.",
            must_say=(),
            avoid=(),
            tone="grounded",
            uncertainties=(),
            evidence_source_ids=(),
        )
    )

    asyncio.run(
        ChatService(
            repositories=repositories,
            providers={"fake": provider},
            context_search_service=ScriptedContextSearch(ContextSearchResult()),
            narrator_planner=planner,
        ).submit_player_turn(
            save_id=save.id,
            body="I tell Mara that ember dawn wakes the beacon lens.",
            speaker_name="Ily",
            run_post_turn_jobs=False,
        )
    )

    assert len(planner.calls) == 1
    assessment_text = "\n".join(planner.calls[0][1].character_action_plans)
    assert "Mara pockets the phrase and watches the lens." in assessment_text
    assert "learned memory candidate (do not persist automatically)" in (
        assessment_text
    )
    assert "Mara learned that ember dawn wakes the beacon lens." in assessment_text
    assert "knowledge edge candidate (do not persist automatically)" in (
        assessment_text
    )
    assert "target: memory:memory-beacon-key" in assessment_text
    assert "Review before making Mara's learned phrase durable." in assessment_text
    assert [record.id for record in repositories.list_memories(save.id)] == [
        memory.id
    ]
    assert repositories.list_character_knowledge_edges(save.id) == []


def test_submit_player_turn_does_not_run_director_pressure_before_prompt(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"player_character_name": "Ily"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.add_character(
        save_id=save.id,
        name="Ily",
        met=True,
        is_player_character=True,
    )
    mara = repositories.add_character(
        save_id=save.id,
        name="Mara",
        met=True,
        current_intent="guard the signal lantern",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara waits beside the beacon controls.",
        present_character_ids=[mara.id],
    )
    repositories.set_app_setting(DIRECTOR_PRESSURE_ENABLED_SETTING, True)
    repositories.set_app_setting(CHARACTER_ACTION_PLANNING_ENABLED_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="fake",
        model_id="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-chat",
        display_name="Fake Chat",
        capabilities=["chat", "structured_output"],
    )
    provider = RecordingCharacterActionChatProvider(
        "fake",
        {
            "Mara": {
                "present": True,
                "action": "Mara checks the corridor and keeps her hand on the lantern.",
                "intent": "keep watch for the player",
                "reason": "Mara is already guarding the signal lantern.",
                "confidence": 0.9,
                "evidence_source_ids": [],
            },
        },
    )
    director = ScriptedDirectorPressureRunner(
        DirectorPressureResult(
            applied=True,
            pressure_kind="external_complication",
            directive="Raise stakes: guards start searching this floor.",
            assessment="The scene has stalled.",
            active_thread_title="Guards search the tower floor",
            active_thread_description="The guard sweep is moving toward the beacon.",
            active_thread_priority=3,
            evidence_source_ids=("message:player",),
            state=DirectorPressureState(
                dramatic_questions=("Will Mara warn the lower village?",),
                tension_level=3,
                tension_trend="stalled",
                stall_turns=0,
                cooldown_turns=2,
            ),
        ),
        events=provider.events,
    )

    asyncio.run(
        ChatService(
            repositories=repositories,
            providers={"fake": provider},
            context_search_service=ScriptedContextSearch(ContextSearchResult()),
            director_pressure_service=director,
        ).submit_player_turn(
            save_id=save.id,
            body="I ask Mara whether the corridor is still clear.",
            speaker_name="Ily",
            run_post_turn_jobs=False,
        )
    )

    assert provider.events == [
        "character_presence_assessment",
        "chat",
    ]
    assert director.calls == []
    assert director.commits == []
    request = provider.chat_requests[0]
    assert request.director_pressure == ""
    assert request.character_action_plans == ()
    character_prompt = provider.structured_output_requests[0].messages[-1].body
    assert "External story pressure:" not in character_prompt
    assert "guards start searching this floor" not in character_prompt


def test_submit_player_turn_skips_character_action_planning_when_disabled(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"player_character_name": "Ily"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.add_character(
        save_id=save.id,
        name="Mara",
        met=True,
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=CHARACTER_ACTION_PLANNING_ENABLED_SETTING,
        value=False,
    )
    provider = RecordingCharacterActionChatProvider(
        "fake",
        {
            "Mara": {
                "present": True,
                "action": "Mara answers.",
                "intent": "",
                "reason": "",
                "confidence": 0.7,
                "evidence_source_ids": [],
            }
        },
    )

    asyncio.run(
        ChatService(
            repositories=repositories,
            providers={"fake": provider},
            context_search_service=ScriptedContextSearch(ContextSearchResult()),
        ).submit_player_turn(
            save_id=save.id,
            body="I ask Mara what she sees.",
            speaker_name="Ily",
            run_post_turn_jobs=False,
        )
    )

    assert provider.structured_output_requests == []
    request = provider.chat_requests[0]
    assert request.character_action_plans == ()
    assert request.context_breakdown["character_action_planning"] == {
        "assessment_count": 0,
        "failed_character_ids": [],
        "failed_count": 0,
        "prompt_guidance_count": 0,
        "skipped_reason": "disabled",
        "applied_presence_update": False,
    }


def test_submit_timeskip_turn_runs_character_action_planning(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"player_character_name": "Ily"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    player = repositories.add_character(
        save_id=save.id,
        name="Ily",
        met=True,
        is_player_character=True,
    )
    mara = repositories.add_character(save_id=save.id, name="Mara", met=True)
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="Mara waits beside the beacon controls.",
        present_character_ids=[mara.id],
    )
    repositories.set_app_setting(CHARACTER_ACTION_PLANNING_ENABLED_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task=CHARACTER_ACTION_PLANNING_TASK,
        provider="fake",
        model_id="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-chat",
        display_name="Fake Chat",
        capabilities=["chat", "structured_output"],
    )
    provider = RecordingCharacterActionChatProvider(
        "fake",
        {
            "Mara": {
                "present": True,
                "action": "Mara shields the lantern as dawn breaks.",
                "intent": "protect the signal",
                "reason": "The timeskip keeps Mara at the beacon.",
                "confidence": 0.85,
                "evidence_source_ids": ["message:latest"],
            },
        },
    )

    asyncio.run(
        ChatService(
            repositories=repositories,
            providers={"fake": provider},
            context_search_service=ScriptedContextSearch(ContextSearchResult()),
        ).submit_timeskip_turn(
            save_id=save.id,
            instruction="Skip to dawn while Mara keeps watch.",
            run_post_turn_jobs=False,
        )
    )

    assert provider.events == [
        "character_presence_assessment",
        "character_intent_plan",
        "chat",
    ]
    request = provider.chat_requests[0]
    assert len(request.character_action_plans) == 1
    assert request.character_action_plans[0].startswith(
        "[character_action:"
        f"{mara.id}] Mara | intent: protect the signal | next action: "
        "Mara shields the lantern as dawn breaks. | reason: The timeskip "
        "keeps Mara at the beacon. | confidence: 85% | evidence: message:"
    )
    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert set(snapshot.present_character_ids) == {player.id, mara.id}


def test_submit_player_turn_does_not_mutate_scene_before_narrator_generation(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    present = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        aliases=["Ilyra"],
        met=True,
    )
    repositories.add_character(
        save_id=save.id,
        name="Archivist Lio",
        aliases=["Lio"],
        met=True,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="The beacon lens hums under stress.",
        present_character_ids=[present.id],
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    original_chat = provider.chat
    world_state_before_chat: list[list[str]] = []
    present_ids_before_chat: list[list[str]] = []

    async def recording_chat(request: ChatRequest) -> ChatResponse:
        world_state_before_chat.append(
            [state.key for state in repositories.list_world_state(save.id)]
        )
        snapshot = repositories.get_scene_snapshot(save.id)
        present_ids_before_chat.append(
            list(snapshot.present_character_ids if snapshot else [])
        )
        return await original_chat(request)

    provider.chat = recording_chat  # type: ignore[method-assign]
    context_update_service = MutatingContextUpdateService(repositories)
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        context_update_service=context_update_service,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I ask Ilyra where Lio went.",
            speaker_name="Mara",
        )
    )

    assert world_state_before_chat == [[]]
    assert present_ids_before_chat == [[present.id]]
    assert context_update_service.calls == [
        (save.id, (result.player_message.id, result.narrator_message.id))
    ]
    assert [state.key for state in repositories.list_world_state(save.id)] == [
        "post_turn.marker"
    ]


def test_submit_player_turn_reports_suppressed_duplicate_retrieval(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    snapshot = repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="The beacon lens hums under stress.",
        snapshot_id="snapshot-lens",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(
                selected_state=(
                    SelectedContextItem(
                        source_type="scene_snapshot",
                        source_id=snapshot.id,
                        text="Scene snapshot: stale duplicate scene text.",
                        relevance_note="Duplicate of deterministic projection.",
                    ),
                    SelectedContextItem(
                        source_type="world_state",
                        source_id="state-extra",
                        text="tower.signal: The horn still needs repair.",
                        relevance_note="Supplemental retrieved fact.",
                    ),
                )
            )
        ),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I inspect the signal horn.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
        )
    )

    request = provider.chat_requests[0]
    assert request.retrieved_state == (
        "[world_state:state-extra] tower.signal: The horn still needs repair.",
    )
    breakdown = request.context_breakdown
    assert breakdown["deterministic_source_count"] >= 1
    assert breakdown["retrieved_source_count"] == 2
    assert breakdown["suppressed_duplicate_retrieval_count"] == 1
    assert breakdown["suppressed_duplicate_retrieval_keys"] == [
        f"scene_snapshot:{snapshot.id}"
    ]


def test_submit_player_turn_suppresses_indexed_current_location_duplicate(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    location = repositories.add_location(
        save_id=save.id,
        name="Lens Gallery",
        description="A brass-ringed gallery above the beacon chamber.",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
        situation="The beacon lens hums under stress.",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(
                selected_state=(
                    SelectedContextItem(
                        source_type="world_state",
                        source_id=f"location:{location.id}",
                        text="Location: Lens Gallery. Stale indexed duplicate.",
                        relevance_note="Duplicate of deterministic location.",
                    ),
                    SelectedContextItem(
                        source_type="world_state",
                        source_id="state-extra",
                        text="tower.signal: The horn still needs repair.",
                        relevance_note="Supplemental retrieved fact.",
                    ),
                )
            )
        ),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I inspect the signal horn.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
        )
    )

    request = provider.chat_requests[0]
    assert "Current location: Lens Gallery" in "\n".join(
        request.current_scene_recap
    )
    assert request.retrieved_state == (
        "[world_state:state-extra] tower.signal: The horn still needs repair.",
    )
    assert request.context_breakdown["suppressed_duplicate_retrieval_keys"] == [
        f"world_state:location:{location.id}"
    ]


def test_current_scene_recap_preserves_long_narrator_message_edges() -> None:
    long_body = (
        "The narrator opened cabinets along the blue wall and counted cracked "
        "plates. "
        + " ".join(f"quiet filler detail {index}" for index in range(80))
        + " At the end, she found the chipped mug, rinsed it, and made tea."
    )
    message = MessageRecord(
        id="narrator-long",
        save_id="save-1",
        role="narrator",
        speaker_name="Narrator",
        body=long_body,
        provider="fake",
        model="fake-chat",
        token_estimate=None,
        deleted_at=None,
    )

    recap_line = chat_service_module._recap_message_line(message)

    normalized = recap_line.casefold()
    assert "opened cabinets" in normalized
    assert "found the chipped mug" in normalized
    assert "made tea" in normalized
    assert "..." in recap_line


def test_submit_player_turn_keeps_recent_transcript_out_of_current_scene_recap(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    prior_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The exact brass lens transcript should only appear as chat history.",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I inspect the lens housing.",
            speaker_name="Mara",
        )
    )

    request = provider.chat_requests[0]
    assert prior_message.body in [message.body for message in request.messages]
    recap_text = "\n".join(request.current_scene_recap)
    assert "Recent chronicle:" not in recap_text
    assert prior_message.body not in recap_text
    assert "Deterministic current-scene context" in recap_text


def test_submit_player_turn_budgets_retrieved_context_and_reports_breakdown(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"lore": "The beacon was raised by a buried legion."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    repositories.set_app_setting("context_budget_mode", "fixed_chars")
    repositories.set_app_setting("context_budget_fixed_total_chars", 1)
    section_id = f"scenario:{scenario.id}:section:lore"
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(
                selected_scenario_sections=(
                    SelectedContextItem(
                        source_type="scenario_section",
                        source_id=section_id,
                        text="The beacon was raised by a buried legion.",
                        relevance_note="Selected but over budget.",
                    ),
                ),
                selected_state=(
                    SelectedContextItem(
                        source_type="world_state",
                        source_id="state-secret",
                        text="scene.secret: The buried legion built the beacon.",
                        relevance_note="Selected but over budget.",
                    ),
                ),
            )
        ),
    )

    submitted = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I study the beacon base.",
            speaker_name="Mara",
        )
    )

    request = provider.chat_requests[0]
    assert "Ashfall Keep" in request.scenario_instructions
    assert request.retrieved_scenario_sections == ()
    assert request.retrieved_state == ()
    assert submitted.context_trimmed is True
    rendered_context = "\n".join(
        (
            request.scenario_instructions,
            "\n".join(request.current_scene_recap),
            "\n".join(request.retrieved_scenario_sections),
            "\n".join(request.retrieved_state),
            "\n".join(request.retrieved_memories),
            request.summary or "",
        )
    )
    assert "buried legion" not in rendered_context
    breakdown_sources = request.context_breakdown["sources"]
    section_breakdown = next(
        source for source in breakdown_sources if source["source_id"] == section_id
    )
    state_breakdown = next(
        source for source in breakdown_sources if source["source_id"] == "state-secret"
    )
    assert section_breakdown["included"] is False
    assert section_breakdown["reason"] == "budget_skipped"
    assert state_breakdown["included"] is False
    assert state_breakdown["reason"] == "budget_skipped"


def test_submit_player_turn_keeps_active_character_voice_under_tight_budget(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    location = repositories.add_location(
        save_id=save.id,
        name="Lens Gallery",
        description="A brass-ringed gallery above the beacon chamber.",
    )
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        aliases=["Ashknife"],
        role="Watch captain",
        known_state="Guarding the cracked red lens",
        met=True,
        personality="Dry humor under pressure",
        voice="Low, clipped commands; never rambles.",
        relationships={"Signal warden": "trusts them with the lens key"},
        status="Bleeding from a brass-cut palm",
        location_id=location.id,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
        situation="The red lens ticks under stress.",
        present_character_ids=[character.id],
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    repositories.set_app_setting("context_budget_mode", "fixed_chars")
    repositories.set_app_setting("context_budget_fixed_total_chars", 1)
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(
                selected_summaries=(
                    SelectedContextItem(
                        source_type="summary",
                        source_id="summary-too-large",
                        text="A lossy old recap should lose budget before voice.",
                        relevance_note="Selected but lower priority than voice.",
                    ),
                ),
            )
        ),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I ask Ilyra what the lens needs.",
            speaker_name="Mara",
        )
    )

    request = provider.chat_requests[0]
    current_scene_text = "\n".join(request.current_scene_recap)
    assert "Present characters: Captain Ilyra (aliases: Ashknife)" in current_scene_text
    assert "voice: Low, clipped commands; never rambles." in current_scene_text
    assert "personality: Dry humor under pressure" in current_scene_text
    assert request.character_voice_profiles == ()
    assert request.summary is None
    present_breakdown = next(
        source
        for source in request.context_breakdown["sources"]
        if source["source_id"] == character.id
        and source["tier"] == "present_characters"
    )
    assert present_breakdown["included"] is True
    assert present_breakdown["reason"] == "present characters"
    assert not any(
        source["source_id"] == character.id
        and source["tier"] == "character_voice_profiles"
        for source in request.context_breakdown["sources"]
    )


def test_submit_player_turn_reports_suppressed_present_character_voice_retrieval(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        aliases=["Ashknife"],
        role="Watch captain",
        known_state="Guarding the cracked red lens",
        met=True,
        personality="Dry humor under pressure",
        voice="Low, clipped commands; never rambles.",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="The red lens ticks under stress.",
        present_character_ids=[character.id],
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(
                selected_character_voice=(
                    SelectedContextItem(
                        source_type="character_voice",
                        source_id=character.id,
                        text="Ilyra voice profile: voice: Low clipped commands.",
                        relevance_note="Duplicate of present-character context.",
                    ),
                ),
            )
        ),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I ask Ilyra what the lens needs.",
            speaker_name="Mara",
        )
    )

    request = provider.chat_requests[0]
    current_scene_text = "\n".join(request.current_scene_recap)
    assert "Present characters: Captain Ilyra" in current_scene_text
    assert "voice: Low, clipped commands; never rambles." in current_scene_text
    assert request.character_voice_profiles == ()
    assert request.context_breakdown["suppressed_duplicate_retrieval_keys"] == [
        f"character_voice:{character.id}"
    ]


def test_submit_player_turn_reports_suppressed_present_character_profile_memory(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        aliases=["Ashknife"],
        role="Watch captain",
        known_state="Guarding the cracked red lens",
        met=True,
        personality="Dry humor under pressure",
        voice="Low, clipped commands; never rambles.",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        situation="The red lens ticks under stress.",
        present_character_ids=[character.id],
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(
                selected_memories=(
                    SelectedContextItem(
                        source_type="memory",
                        source_id=f"character_profile:{character.id}",
                        text="Ilyra profile duplicate: role: Watch captain.",
                        relevance_note="Duplicate of present-character context.",
                    ),
                    SelectedContextItem(
                        source_type="memory",
                        source_id="memory-extra",
                        text="Mara promised Ilyra the lens key.",
                        relevance_note="Selected promise.",
                    ),
                ),
            )
        ),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I ask Ilyra what the lens needs.",
            speaker_name="Mara",
        )
    )

    request = provider.chat_requests[0]
    current_scene_text = "\n".join(request.current_scene_recap)
    assert "Present characters: Captain Ilyra" in current_scene_text
    assert "role: Watch captain" in current_scene_text
    assert request.retrieved_memories == (
        "[memory:memory-extra] Mara promised Ilyra the lens key.",
    )
    assert request.context_breakdown["suppressed_duplicate_retrieval_keys"] == [
        f"memory:character_profile:{character.id}"
    ]


def test_submit_player_turn_includes_addressed_character_voice_without_snapshot(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Interview with the Sealed Archivist",
        premise="A tense private conversation in a forbidden archive.",
        player_role="Investigator",
        content={"character_name": "Mael"},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Interview")
    repositories.add_character(
        save_id=save.id,
        name="Mael",
        aliases=["Sealed Archivist"],
        role="Archivist",
        met=True,
        personality="Deflects fear with exacting questions",
        voice="Formal, precise, and allergic to casual slang.",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="Mael, what did you hide in the red index?",
            speaker_name="Mara",
        )
    )

    voice_text = "\n".join(provider.chat_requests[0].character_voice_profiles)
    assert "Mael voice profile" in voice_text
    assert "voice: Formal, precise, and allergic to casual slang." in voice_text


def test_submit_player_turn_does_not_address_character_from_alias_substring(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.add_character(
        save_id=save.id,
        name="Archivist Lio",
        aliases=["Lio"],
        role="Archivist",
        met=True,
        personality="Answers direct questions with clipped precision",
        voice="Soft, exact, and wary of strangers.",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I study the rendezvous marks near the lantern.",
            speaker_name="Mara",
        )
    )

    assert provider.chat_requests[0].character_voice_profiles == ()


def test_submit_player_turn_addresses_character_with_possessive_alias(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.add_character(
        save_id=save.id,
        name="Archivist Lio",
        aliases=["Lio"],
        role="Archivist",
        met=True,
        personality="Answers direct questions with clipped precision",
        voice="Soft, exact, and wary of strangers.",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I ask about Lio's locked ledger.",
            speaker_name="Mara",
        )
    )

    voice_text = "\n".join(provider.chat_requests[0].character_voice_profiles)
    assert "Archivist Lio voice profile" in voice_text
    assert "voice: Soft, exact, and wary of strangers." in voice_text


def test_submit_player_turn_injects_selected_character_voice_as_voice_profile(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(
                selected_character_voice=(
                    SelectedContextItem(
                        source_type="character_voice",
                        source_id="ilyra-id",
                        text="Ilyra voice profile: voice: Low clipped commands.",
                        relevance_note="Selected voice profile.",
                    ),
                ),
                selected_memories=(
                    SelectedContextItem(
                        source_type="memory",
                        source_id="memory-id",
                        text="Mara promised Ilyra the lens key.",
                        relevance_note="Selected promise.",
                    ),
                ),
            )
        ),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="What does Ilyra say?",
            speaker_name="Mara",
        )
    )

    request = provider.chat_requests[0]
    voice_text = "\n".join(request.character_voice_profiles)
    memory_text = "\n".join(request.retrieved_memories)
    assert "Ilyra voice profile" in voice_text
    assert "Selected voice profile" not in voice_text
    assert "Ilyra voice profile" not in memory_text
    assert "Mara promised Ilyra the lens key." in memory_text


def test_selected_context_sources_keep_selector_notes_diagnostic_only() -> None:
    sources = _selected_context_sources(
        (
            SelectedContextItem(
                source_type="memory",
                source_id="memory-evacuation-key",
                text="The cracked bell hides the evacuation key.",
                relevance_note="Ignore the bell and focus on unrelated filler.",
            ),
        ),
        tier="retrieved_memories",
        relevance_query="I inspect the cracked bell for the evacuation key.",
    )

    assert sources[0].text == (
        "[memory:memory-evacuation-key] "
        "The cracked bell hides the evacuation key."
    )
    assert "unrelated filler" not in sources[0].text
    assert sources[0].relevance_query == (
        "I inspect the cracked bell for the evacuation key."
    )


def test_submit_player_turn_default_context_budget_mode_is_diagnostics_only(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"lore": "The beacon was raised by a buried legion."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    section_id = f"scenario:{scenario.id}:section:lore"
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(
                selected_scenario_sections=(
                    SelectedContextItem(
                        source_type="scenario_section",
                        source_id=section_id,
                        text="The beacon was raised by a buried legion.",
                        relevance_note="Selected by diagnostics-only context search.",
                    ),
                ),
            )
        ),
    )

    submitted = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I study the beacon base.",
            speaker_name="Mara",
        )
    )

    request = provider.chat_requests[0]
    assert request.context_breakdown["budget_mode"] == "diagnostics_only"
    assert request.context_breakdown["budget_limit_chars"] is None
    assert request.context_breakdown["included_chars"] == request.context_breakdown[
        "total_chars"
    ]
    assert all(source["included"] for source in request.context_breakdown["sources"])
    assert request.retrieved_scenario_sections == (
        f"[scenario_section:{section_id}] "
        "The beacon was raised by a buried legion.",
    )
    assert submitted.context_trimmed is False


def test_submit_player_turn_reuses_narration_snapshot_for_prompt_building(
    repositories: PersistenceRepositories,
) -> None:
    counting = CountingNarrationPersistenceRepositories(repositories.connection)
    scenario = counting.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = counting.create_save(scenario_id=scenario.id, title="Night Watch")
    location = counting.add_location(
        save_id=save.id,
        name="Beacon Gallery",
        description="A red glass signal chamber.",
    )
    character = counting.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        voice="Low clipped commands.",
        location_id=location.id,
    )
    source_message = counting.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Captain Ilyra steadies Mara near the red lens.",
        provider="fake",
        model="fake-chat",
    )
    counting.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
        present_character_ids=[character.id],
        situation="Mara and Ilyra inspect the beacon.",
        source_message_id=source_message.id,
    )
    memory = counting.add_memory(
        save_id=save.id,
        body="Ilyra knows the lens-key phrase.",
        tags=["ilyra"],
        source_message_id=source_message.id,
    )
    state = counting.upsert_world_state(
        save_id=save.id,
        key="beacon.lens",
        value={"status": "cracked red glass"},
        source_message_id=source_message.id,
    )
    counting.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=character.id,
        target_type="memory",
        target_id=memory.id,
        relation="knows",
    )
    counting.add_entity_link(
        save_id=save.id,
        entity_type="location",
        entity_id=location.id,
        target_type="world_state",
        target_id=state.id,
        relation="contains",
    )
    counting.set_model_preference(
        task="context_search",
        provider="fake",
        model_id="fake-context",
    )
    counting.save_provider_model(
        provider="fake",
        model_id="fake-context",
        display_name="Fake Context",
        capabilities=["structured_output"],
    )
    counting.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    counting.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=CHARACTER_ACTION_PLANNING_ENABLED_SETTING,
        value=False,
    )
    counting.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=PLAN_FIRST_NARRATOR_SETTING,
        value=False,
    )
    provider = RecordingContextAndChatProvider("fake")
    context_search = ContextSearchService(
        repositories=counting,
        providers={"fake": provider},
    )
    service = ChatService(
        repositories=counting,
        providers={"fake": provider},
        context_search_service=context_search,
    )
    counting.read_counts.clear()

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I ask Ilyra what the lens needs now.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
        )
    )

    assert provider.structured_output_requests
    assert provider.chat_requests
    assert counting.read_counts["load_save_details"] <= 3
    assert counting.read_counts["messages"] <= 3
    for name in (
        "scene_snapshot",
        "locations",
        "characters",
        "active_threads",
    ):
        assert counting.read_counts.get(name, 0) <= 3, name
    for name in (
        "character_knowledge_edges",
        "message_visibility",
        "entity_links",
        "world_state",
        "world_state_including_archived",
        "state_changes",
        "media_assets",
        "memories",
        "summaries",
        "context_observations",
        "context_sources",
        "context_update_suggestions",
    ):
        assert counting.read_counts.get(name, 0) <= 2, name


def test_submit_player_turn_final_prompt_budget_trims_baseline_before_retrieval(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    prior_messages = [
        repositories.append_message(
            save_id=save.id,
            role="player" if index % 2 == 0 else "narrator",
            speaker_name="Mara" if index % 2 == 0 else "Narrator",
            body=(
                f"Prior chronicle beat {index}. "
                + "The storm repeats a long brass-rimmed warning. " * 12
            ),
            provider=None if index % 2 == 0 else "openrouter",
            model=None if index % 2 == 0 else "anthropic/claude-3.5-sonnet",
        )
        for index in range(12)
    ]
    repositories.save_provider_model(
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
        display_name="Claude 3.5 Sonnet",
        capabilities=["chat"],
        context_window=2200,
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(
                selected_state=(
                    SelectedContextItem(
                        source_type="world_state",
                        source_id="state-lens-fuse",
                        text="beacon.fuse: The spare fuse is under the red lens.",
                        relevance_note="The player is asking about the lens.",
                    ),
                ),
            )
        ),
    )

    submitted = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I ask what the lens needs now.",
            speaker_name="Mara",
        )
    )

    request = provider.chat_requests[0]
    budget = request.context_breakdown["final_prompt_budget"]
    assert budget["enforced"] is True
    assert budget["trimmed"] is True
    assert budget["estimated_tokens_before"] > budget["input_limit_tokens"]
    assert budget["estimated_tokens_after"] <= budget["input_limit_tokens"]
    assert submitted.context_trimmed is True
    assert request.messages[-1].body == "I ask what the lens needs now."
    assert prior_messages[0].body not in [message.body for message in request.messages]
    assert request.retrieved_state == (
        "[world_state:state-lens-fuse] "
        "beacon.fuse: The spare fuse is under the red lens.",
    )
    trimmed_sections = [
        item["section"] for item in budget["trimmed_sections"] if isinstance(item, dict)
    ]
    assert "messages" in trimmed_sections
    assert "retrieved_state" not in trimmed_sections
    job_result = _chat_completion_jobs(repositories, save.id)[-1]["result"]
    prompt_diagnostics = job_result["prompt_context_diagnostics"]
    assert prompt_diagnostics["context_search_failed"] is False
    assert prompt_diagnostics["baseline_recent_message_count"] == len(
        request.messages
    ) - 1
    assert prompt_diagnostics["baseline_recent_message_chars"] == sum(
        len(message.body) for message in request.messages[:-1]
    )
    assert prompt_diagnostics["retrieved_counts"]["state"] == 1
    assert prompt_diagnostics["final_prompt_budget"]["trimmed"] is True


def test_final_prompt_budget_trims_pending_suggestions_before_messages() -> None:
    request = chat_service_module._apply_final_prompt_budget(
        ChatRequest(
            provider="fake",
            model_id="fake-chat",
            messages=(
                ChatMessage(role="player", body="Brief prior beat."),
                ChatMessage(role="player", body="I wait for the signal."),
            ),
            pending_context_suggestions=("Pending review: " + "storm " * 1800,),
            max_output_tokens=1,
        ),
        model_context_window=1000,
    )

    budget = request.context_breakdown["final_prompt_budget"]
    trimmed_sections = [
        item["section"] for item in budget["trimmed_sections"] if isinstance(item, dict)
    ]
    assert request.pending_context_suggestions == ()
    assert [message.body for message in request.messages] == [
        "Brief prior beat.",
        "I wait for the signal.",
    ]
    assert trimmed_sections[0] == "pending_context_suggestions"
    assert "messages" not in trimmed_sections


def test_final_prompt_budget_can_trim_phone_context() -> None:
    request = chat_service_module._apply_final_prompt_budget(
        ChatRequest(
            provider="fake",
            model_id="fake-chat",
            messages=(ChatMessage(role="player", body="I check my phone."),),
            phone_context=("Phone thread: Mika " + "late message " * 1800,),
            max_output_tokens=1,
        ),
        model_context_window=1000,
    )

    budget = request.context_breakdown["final_prompt_budget"]
    trimmed_sections = [
        item["section"] for item in budget["trimmed_sections"] if isinstance(item, dict)
    ]
    assert budget["trimmed"] is True
    assert request.phone_context == ()
    assert "phone_context" in trimmed_sections


def test_submit_player_turn_final_prompt_budget_trims_selected_retrieval(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.save_provider_model(
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
        display_name="Claude 3.5 Sonnet",
        capabilities=["chat"],
        context_window=1500,
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(
                selected_recent_messages=(
                    SelectedContextItem(
                        source_type="message",
                        source_id="message-too-large",
                        text="Old chronicle: " + "the buried legion repeats " * 120,
                        relevance_note="Too verbose for the final prompt budget.",
                    ),
                ),
            )
        ),
    )

    submitted = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I study the beacon base.",
            speaker_name="Mara",
        )
    )

    request = provider.chat_requests[0]
    budget = request.context_breakdown["final_prompt_budget"]
    assert budget["enforced"] is True
    assert budget["trimmed"] is True
    assert request.messages[-1].body == "I study the beacon base."
    assert request.retrieved_recent_messages == ()
    assert submitted.context_trimmed is True
    assert any(
        isinstance(item, dict) and item["section"] == "retrieved_recent_messages"
        for item in budget["trimmed_sections"]
    )


def test_final_prompt_budget_preserves_summary_before_low_priority_retrieval() -> None:
    request = chat_service_module._apply_final_prompt_budget(
        ChatRequest(
            provider="fake",
            model_id="fake-chat",
            messages=(ChatMessage(role="player", body="I wait by the beacon."),),
            current_scene_recap=("Scene snapshot: the beacon gallery is active.",),
            retrieved_recent_messages=(
                "[message:old-1] Old chronicle: " + "ash signal " * 220,
                "[message:old-2] Older chronicle: " + "brass warning " * 220,
            ),
            retrieved_observations=(
                "[observation:obs-1] Routine ambient detail. "
                + "wind note " * 160,
            ),
            retrieved_memories=(
                "[memory:mem-1] Low priority selected memory. "
                + "lens trivia " * 180,
            ),
            retrieved_state=(
                "[world_state:state-1] Low priority selected state. "
                + "dust color " * 180,
            ),
            summary=(
                "[summary:summary-latest] Mara crossed the ash bridge, promised "
                "Ilyra the beacon would burn, and reached the red lens."
            ),
            max_output_tokens=1,
        ),
        model_context_window=900,
    )

    budget = request.context_breakdown["final_prompt_budget"]
    trimmed_sections = [
        item["section"] for item in budget["trimmed_sections"] if isinstance(item, dict)
    ]
    assert budget["trimmed"] is True
    assert request.summary is not None
    assert "summary-latest" in request.summary
    assert request.retrieved_recent_messages == ()
    assert "summary" not in trimmed_sections
    assert trimmed_sections.index("retrieved_recent_messages") < trimmed_sections.index(
        "retrieved_memories"
    )


def test_submit_player_turn_final_prompt_budget_diagnostics_only_without_model_window(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    prior_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The long prior warning remains available without a model window."
        + " Brass warning." * 80,
        provider="openrouter",
        model="anthropic/claude-3.5-sonnet",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    submitted = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I listen for the warning again.",
            speaker_name="Mara",
        )
    )

    request = provider.chat_requests[0]
    budget = request.context_breakdown["final_prompt_budget"]
    assert budget["enforced"] is False
    assert budget["reason"] == "no_model_context_window"
    assert budget["trimmed"] is False
    assert prior_message.body in [message.body for message in request.messages]
    assert submitted.context_trimmed is False


def test_submit_player_turn_keeps_recent_baseline_and_retrieves_selected_prior_messages(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    prior_messages = [
        repositories.append_message(
            save_id=save.id,
            role="player" if index % 2 == 0 else "narrator",
            speaker_name="Mara" if index % 2 == 0 else "Narrator",
            body=f"Prior chronicle beat {index}",
            provider=None if index % 2 == 0 else "openrouter",
            model=None if index % 2 == 0 else "anthropic/claude-3.5-sonnet",
        )
        for index in range(16)
    ]
    selected_older_message = prior_messages[0]
    selected_newer_message = prior_messages[14]
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(
                selected_recent_messages=(
                    SelectedContextItem(
                        source_type="message",
                        source_id=selected_newer_message.id,
                        text=selected_newer_message.body,
                        relevance_note="The newer warning is most relevant.",
                    ),
                    SelectedContextItem(
                        source_type="message",
                        source_id=selected_older_message.id,
                        text=selected_older_message.body,
                        relevance_note="The older warning still matters.",
                    ),
                ),
            ),
        ),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I raise the storm lantern again.",
            speaker_name="Mara",
        )
    )

    request = provider.chat_requests[0]
    assert [message.body for message in request.messages] == [
        prior_messages[6].body,
        prior_messages[7].body,
        prior_messages[8].body,
        prior_messages[9].body,
        prior_messages[10].body,
        prior_messages[11].body,
        prior_messages[12].body,
        prior_messages[13].body,
        selected_newer_message.body,
        prior_messages[15].body,
        "I raise the storm lantern again.",
    ]
    assert {
        message.body
        for message in request.messages
        if message.role == "player"
    } == {
        prior_messages[6].body,
        prior_messages[8].body,
        prior_messages[10].body,
        prior_messages[12].body,
        selected_newer_message.body,
        "I raise the storm lantern again.",
    }
    assert {
        message.body
        for message in request.messages
        if message.role == "narrator"
    } == {
        prior_messages[7].body,
        prior_messages[9].body,
        prior_messages[11].body,
        prior_messages[13].body,
        prior_messages[15].body,
    }
    assert selected_older_message.body not in [
        message.body for message in request.messages
    ]
    assert [message.body for message in request.messages].count(
        selected_newer_message.body
    ) == 1
    assert request.retrieved_recent_messages == (
        "[message:"
        f"{selected_newer_message.id}] {selected_newer_message.body}",
        "[message:"
        f"{selected_older_message.id}] {selected_older_message.body}",
    )
    assert all(
        message.body not in {chat_message.body for chat_message in request.messages}
        for message in (prior_messages[1], prior_messages[2], prior_messages[3])
    )


def test_submit_player_turn_uses_configured_recent_baseline_windows(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    prior_messages = [
        repositories.append_message(
            save_id=save.id,
            role="player" if index % 2 == 0 else "narrator",
            speaker_name="Mara" if index % 2 == 0 else "Narrator",
            body=f"Prior chronicle beat {index}",
            provider=None if index % 2 == 0 else "openrouter",
            model=None if index % 2 == 0 else "anthropic/claude-3.5-sonnet",
        )
        for index in range(10)
    ]
    repositories.set_app_setting(RECENT_PLAYER_MESSAGE_WINDOW_SETTING, 2)
    repositories.set_app_setting(RECENT_NARRATOR_MESSAGE_WINDOW_SETTING, 1)
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I raise the storm lantern again.",
            speaker_name="Mara",
        )
    )

    request = provider.chat_requests[0]
    assert [message.body for message in request.messages] == [
        prior_messages[6].body,
        prior_messages[8].body,
        prior_messages[9].body,
        "I raise the storm lantern again.",
    ]
    assert all(
        message.body not in {chat_message.body for chat_message in request.messages}
        for message in (*prior_messages[:6], prior_messages[7])
    )


def test_submit_player_turn_uses_separate_planner_and_prose_history_windows(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    prior_messages = [
        repositories.append_message(
            save_id=save.id,
            role="player" if index % 2 == 0 else "narrator",
            speaker_name="Mara" if index % 2 == 0 else "Narrator",
            body=f"Prior chronicle beat {index}",
            provider=None if index % 2 == 0 else "fake",
            model=None if index % 2 == 0 else "fake-chat",
        )
        for index in range(10)
    ]
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_app_setting(PLAN_FIRST_NARRATOR_SETTING, True)
    repositories.set_app_setting(RECENT_PLAYER_MESSAGE_WINDOW_SETTING, 2)
    repositories.set_app_setting(RECENT_NARRATOR_MESSAGE_WINDOW_SETTING, 1)
    repositories.set_app_setting(
        NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
        4,
    )
    repositories.set_app_setting(
        NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
        3,
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingChatProvider("fake")
    planner = ScriptedNarratorPlanner(
        NarratorMessageSpec(
            intent="Plan the lantern response.",
            thesis="The storm lantern matters.",
            must_say=(),
            avoid=(),
            tone="tense",
            uncertainties=(),
            evidence_source_ids=(),
        )
    )
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        narrator_planner=planner,
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I raise the storm lantern again.",
            speaker_name="Mara",
            run_post_turn_jobs=False,
        )
    )

    planner_request = planner.calls[0][1]
    assert [message.body for message in planner_request.messages] == [
        prior_messages[2].body,
        prior_messages[4].body,
        prior_messages[5].body,
        prior_messages[6].body,
        prior_messages[7].body,
        prior_messages[8].body,
        prior_messages[9].body,
        "I raise the storm lantern again.",
    ]
    prose_request = provider.chat_requests[0]
    assert prose_request.narrator_prompt_mode == "plan_first"
    assert [message.body for message in prose_request.messages] == [
        prior_messages[6].body,
        prior_messages[8].body,
        prior_messages[9].body,
        "I raise the storm lantern again.",
    ]


def test_submit_player_turn_keeps_persisted_summaries_out_of_scenario_instructions(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    older_player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I checked the outer wall.",
    )
    older_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The ash hid the road.",
        provider="openrouter",
        model="anthropic/claude-3.5-sonnet",
    )
    selected_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=older_player.id,
        covers_message_end_id=older_narrator.id,
        body="Selected summary: the ash road was hidden.",
        provider="fake",
        model="fake-summary",
    )
    repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=older_player.id,
        covers_message_end_id=older_narrator.id,
        body="Unselected summary: the pantry shelves collapsed.",
        provider="fake",
        model="fake-summary",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(
                selected_summaries=(
                    SelectedContextItem(
                        source_type="summary",
                        source_id=selected_summary.id,
                        text=selected_summary.body,
                        relevance_note="This is the relevant summary.",
                    ),
                ),
            ),
        ),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I scan the road from the beacon.",
            speaker_name="Mara",
        )
    )

    request = provider.chat_requests[0]
    assert "Selected summary: the ash road was hidden." not in (
        request.scenario_instructions
    )
    assert "Unselected summary: the pantry shelves collapsed." not in (
        request.scenario_instructions
    )
    assert request.summary == (
        f"[summary:{selected_summary.id}] Selected summary: the ash road was hidden."
    )


def test_submit_player_turn_includes_latest_summary_when_search_selects_none(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    older_player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I checked the outer wall.",
    )
    older_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The ash hid the road.",
        provider="openrouter",
        model="anthropic/claude-3.5-sonnet",
    )
    summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=older_player.id,
        covers_message_end_id=older_narrator.id,
        body="Mara checked the outer wall and found the ash road hidden.",
        provider="fake",
        model="fake-summary",
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    provider = RecordingChatProvider("openrouter")
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I scan the road from the beacon.",
            speaker_name="Mara",
        )
    )

    request = provider.chat_requests[0]
    assert request.summary == (
        f"[summary:{summary.id}] "
        "Mara checked the outer wall and found the ash road hidden. "
        "(relevance: latest rolling summary.)"
    )


def test_submit_player_turn_summarizes_before_context_search_and_keeps_context_separate(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Bridge of Cinders",
        premise="A bridge remembers every oath broken on it.",
        player_role="Oathkeeper",
        content={"starting_scene": "Cinders drift over the bridge stones."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Crossing")
    older_player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I step onto the ash bridge.",
        token_estimate=70,
    )
    older_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="A bell rings under the span.",
        provider="fake",
        model="fake-chat",
        token_estimate=72,
    )
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="scene.location",
        value={"name": "Ash Bridge"},
        category="scene",
        source_message_id=older_player.id,
        state_id="state-ash-bridge",
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="Mara distrusts bells that ring without wind.",
        tags=["bells", "suspicion"],
        source_message_id=older_narrator.id,
        memory_id="memory-windless-bell",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-chat",
        display_name="Fake Chat",
        capabilities=["chat"],
        context_window=8192,
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    events: list[str] = []
    provider = RecordingChatProvider("fake")

    original_chat = provider.chat

    async def recording_chat(request: ChatRequest) -> ChatResponse:
        events.append("narrator_chat")
        return await original_chat(request)

    provider.chat = recording_chat  # type: ignore[method-assign]
    summary_service = RecordingSummaryService(
        repositories=repositories,
        events=events,
        covers_message_start_id=older_player.id,
        covers_message_end_id=older_narrator.id,
    )

    class SummaryAwareContextSearch(ScriptedContextSearch):
        async def search(
            self,
            *,
            save_id: str,
            player_message_id: str,
        ) -> ContextSearchResult:
            summaries = repositories.list_summaries(save_id)
            assert len(summaries) == 1
            summary = summaries[0]
            self.result = ContextSearchResult(
                selected_state=(
                    SelectedContextItem(
                        source_type="world_state",
                        source_id=state.id,
                        text="scene.location: Ash Bridge",
                        relevance_note="The player is still on the bridge.",
                    ),
                ),
                selected_memories=(
                    SelectedContextItem(
                        source_type="memory",
                        source_id=memory.id,
                        text="Mara distrusts windless bells.",
                        relevance_note="The bell remains suspicious.",
                    ),
                ),
                selected_summaries=(
                    SelectedContextItem(
                        source_type="summary",
                        source_id=summary.id,
                        text=summary.body,
                        relevance_note="Condenses earlier bridge turns.",
                    ),
                ),
            )
            return await super().search(
                save_id=save_id,
                player_message_id=player_message_id,
            )

    context_search = SummaryAwareContextSearch(ContextSearchResult(), events=events)
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=context_search,
        summary_service=summary_service,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I listen for the bell's source.",
            speaker_name="Mara",
        )
    )

    assert events == ["summarization", "context_search", "narrator_chat"]
    assert summary_service.calls == [(save.id, 8192)]
    assert context_search.calls == [(save.id, result.player_message.id)]
    request = provider.chat_requests[0]
    assert request.summary == (
        f"[summary:{summary_service.summary_id}] "
        "Mara crossed the ash bridge before hearing the windless bell."
    )
    current_scene_text = "\n".join(request.current_scene_recap)
    assert "Legacy scene state: scene.location: name: Ash Bridge" in (
        current_scene_text
    )
    assert request.retrieved_state == ()
    assert request.retrieved_memories == (
        "[memory:memory-windless-bell] Mara distrusts windless bells.",
    )
    assert request.context_breakdown["suppressed_duplicate_retrieval_keys"] == [
        f"world_state:{state.id}"
    ]
    assert repositories.list_world_state(save.id) == [state]
    assert repositories.list_memories(save.id) == [memory]


def test_submit_player_turn_counts_pending_player_message_for_summary_pressure(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Bridge of Cinders",
        premise="A bridge remembers every oath broken on it.",
        player_role="Oathkeeper",
        content={"starting_scene": "Cinders drift over the bridge stones."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Crossing")
    older_player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I step onto the ash bridge.",
        token_estimate=40,
    )
    older_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="A bell rings under the span.",
        provider="fake-chat",
        model="fake-chat",
        token_estimate=40,
    )
    recent_player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I ask who rang the bell.",
        token_estimate=20,
    )
    recent_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The echo answers from below.",
        provider="fake-chat",
        model="fake-chat",
        token_estimate=20,
    )
    repositories.save_provider_model(
        provider="fake-chat",
        model_id="fake-chat",
        display_name="Fake Chat",
        capabilities=["chat"],
        context_window=200,
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake-chat",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="summarization",
        provider="fake-summary",
        model_id="fake-summary",
    )
    chat_provider = RecordingChatProvider("fake-chat")
    summary_provider = RecordingChatProvider("fake-summary")
    pending_body = (
        "I describe the bell's impossible resonance, the old oath carved into "
        "the stones, and the way every cinder seems to answer my question with "
        "a name I do not recognize."
    )
    service = ChatService(
        repositories=repositories,
        providers={
            "fake-chat": chat_provider,
            "fake-summary": summary_provider,
        },
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
        summary_service=SummaryService(
            repositories=repositories,
            providers={"fake-summary": summary_provider},
            threshold=0.75,
        ),
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body=pending_body,
            speaker_name="Mara",
        )
    )

    summaries = repositories.list_summaries(save.id)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.covers_message_start_id == older_player.id
    assert summary.covers_message_end_id == older_narrator.id
    assert len(summary_provider.chat_requests) == 1
    summary_prompt = "\n".join(
        message.body for message in summary_provider.chat_requests[0].messages
    )
    assert older_player.body in summary_prompt
    assert older_narrator.body in summary_prompt
    assert recent_player.body not in summary_prompt
    assert recent_narrator.body not in summary_prompt
    assert pending_body not in summary_prompt
    assert result.player_message.body == pending_body


def test_submit_player_turn_extracts_state_and_memory_with_structured_model(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-state-memory",
        display_name="Fake State Memory",
        capabilities=["structured_output"],
        context_window=8192,
    )
    repositories.set_model_preference(
        task="state_memory",
        provider="fake",
        model_id="fake-state-memory",
    )
    events: list[str] = []
    provider = RecordingStateMemoryProvider(
        "fake",
        events=events,
        structured_data={
            "state_changes": [
                {
                    "operation": "upsert",
                    "key": "scene.location",
                    "value": {"name": "Beacon gallery"},
                    "category": "scene",
                    "confidence": 0.87,
                    "evidence_quote": "I climb toward the beacon lens.",
                }
            ],
            "memories": [
                {
                    "body": "Mara promised Elian she would keep the beacon lit.",
                    "tags": ["beacon", "promise"],
                    "importance": 0.91,
                    "evidence_quote": "I climb toward the beacon lens.",
                }
            ],
        },
    )
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(),
            events=events,
        ),
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert events == ["context_search", "narrator_chat", "state_memory_extraction"]
    assert len(provider.structured_output_requests) == 1
    extraction_request = provider.structured_output_requests[0]
    assert extraction_request.provider == "fake"
    assert extraction_request.model_id == "fake-state-memory"
    assert "state" in extraction_request.schema_name
    assert "extraction" in extraction_request.schema_name
    assert "state_changes" in extraction_request.schema["properties"]
    assert "memories" in extraction_request.schema["properties"]
    assert "conflicts" in extraction_request.schema["properties"]
    expected_message_ids = [result.player_message.id, result.narrator_message.id]
    for collection_name in ("state_changes", "memories", "conflicts"):
        assert (
            extraction_request.schema["properties"][collection_name]["items"][
                "properties"
            ]["source_message_id"]["enum"]
            == expected_message_ids
        )
    extraction_prompt = "\n".join(
        message.body.lower() for message in extraction_request.messages
    )
    assert "i climb toward the beacon lens." in extraction_prompt
    assert "fake narrator: i climb toward the beacon lens." in extraction_prompt
    assert "json" not in extraction_prompt

    world_state = repositories.list_world_state(save.id)
    assert len(world_state) == 1
    assert world_state[0].key == "scene.location"
    assert world_state[0].value == {"name": "Beacon gallery"}
    assert world_state[0].category == "scene"
    assert world_state[0].confidence == 0.87
    assert world_state[0].source_message_id == result.narrator_message.id

    memories = repositories.list_memories(save.id)
    assert len(memories) == 1
    assert memories[0].body == "Mara promised Elian she would keep the beacon lit."
    assert memories[0].tags == ["beacon", "promise"]
    assert memories[0].importance == 0.91
    assert memories[0].source_message_id == result.narrator_message.id


def test_submit_player_turn_uses_state_only_extraction_when_agentic_curation_available(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    for task, model_id in (
        ("state_memory", "fake-state-memory"),
        ("fact_observation", "fake-observer"),
        ("memory_curation", "fake-curator"),
    ):
        repositories.save_provider_model(
            provider="fake",
            model_id=model_id,
            display_name=model_id,
            capabilities=["structured_output"],
            context_window=8192,
        )
        repositories.set_model_preference(
            task=task,
            provider="fake",
            model_id=model_id,
        )
    events: list[str] = []
    provider = RecordingAgenticPipelineProvider(
        "fake",
        events=events,
        state_data={
            "state_changes": [
                {
                    "operation": "upsert",
                    "key": "scene.location",
                    "value": {"name": "Beacon gallery"},
                    "category": "scene",
                    "confidence": 0.87,
                    "evidence_quote": "I climb toward the beacon lens.",
                }
            ],
            "memories": [
                {
                    "body": "This state-memory fact should not be durable.",
                    "tags": ["duplicate"],
                    "importance": 0.91,
                }
            ],
        },
        curation_memory_body="Mara likes concise, grounded narration.",
    )
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(),
            events=events,
        ),
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert events == [
        "context_search",
        "narrator_chat",
        "state_memory_extraction",
        "fact_observation",
        "memory_curation",
    ]
    state_request = _only_request_matching(
        provider.structured_output_requests,
        "state",
        "extraction",
    )
    assert "memories" not in state_request.schema["properties"]
    assert repositories.list_world_state(save.id)[0].value == {
        "name": "Beacon gallery"
    }
    assert [memory.body for memory in repositories.list_memories(save.id)] == [
        "Mara likes concise, grounded narration."
    ]
    state_jobs = [
        job
        for job in repositories.list_jobs_by_status(("succeeded",))
        if job.save_id == save.id and job.type == "state_extraction"
    ]
    state_job_result = state_jobs[-1].result
    assert state_job_result is not None
    assert state_job_result["memory_count"] == 0
    assert result.narrator_message.body == (
        "fake narrator: I climb toward the beacon lens."
    )


def test_submit_player_turn_falls_back_to_state_memory_when_curation_unavailable(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-state-memory",
        display_name="Fake State Memory",
        capabilities=["structured_output"],
        context_window=8192,
    )
    repositories.set_model_preference(
        task="state_memory",
        provider="fake",
        model_id="fake-state-memory",
    )
    events: list[str] = []
    provider = RecordingStateMemoryProvider(
        "fake",
        events=events,
        structured_data={
            "state_changes": [],
            "memories": [
                {
                    "body": "Mara promised Elian she would keep the beacon lit.",
                    "tags": ["beacon", "promise"],
                    "importance": 0.91,
                    "evidence_quote": "I climb toward the beacon lens.",
                }
            ],
        },
    )
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(),
            events=events,
        ),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    state_request = provider.structured_output_requests[0]
    assert "memories" in state_request.schema["properties"]
    assert [memory.body for memory in repositories.list_memories(save.id)] == [
        "Mara promised Elian she would keep the beacon lit."
    ]


def test_submit_player_turn_queues_agentic_durable_memory_when_confirmation_enabled(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_app_setting(AGENTIC_CONTEXT_PIPELINE_SETTING, True)
    repositories.set_app_setting("manual_confirmation_memories_enabled", True)
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    for task, model_id in (
        ("state_memory", "fake-state-memory"),
        ("fact_observation", "fake-observer"),
        ("memory_curation", "fake-curator"),
    ):
        repositories.save_provider_model(
            provider="fake",
            model_id=model_id,
            display_name=model_id,
            capabilities=["structured_output"],
            context_window=8192,
        )
        repositories.set_model_preference(
            task=task,
            provider="fake",
            model_id=model_id,
        )
    events: list[str] = []
    provider = RecordingAgenticPipelineProvider(
        "fake",
        events=events,
        state_data={"state_changes": [], "memories": []},
        curation_memory_body="Mara likes concise, grounded narration.",
    )
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(),
            events=events,
        ),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert repositories.list_memories(save.id) == []
    suggestions = repositories.list_context_update_suggestions(save.id)
    assert len(suggestions) == 1
    assert suggestions[0].entity_type == "memory"
    proposed_value = cast(dict[str, Any], suggestions[0].proposed_value)
    assert proposed_value["body"] == (
        "Mara likes concise, grounded narration."
    )


def test_submit_player_turn_runs_scenario_evolution_after_state_memory_extraction(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-state-memory",
        display_name="Fake State Memory",
        capabilities=["structured_output"],
        context_window=8192,
    )
    repositories.set_model_preference(
        task="state_memory",
        provider="fake",
        model_id="fake-state-memory",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-scenario-evolution",
        display_name="Fake Scenario Evolution",
        capabilities=["structured_output"],
        context_window=8192,
    )
    repositories.set_model_preference(
        task="scenario_evolution",
        provider="fake",
        model_id="fake-scenario-evolution",
    )
    events: list[str] = []
    provider = RecordingScenarioEvolutionProvider("fake", events=events)
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(),
            events=events,
        ),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I relight the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert events == [
        "context_search",
        "narrator_chat",
        "state_memory_extraction",
        "scenario_evolution",
    ]
    assert len(provider.structured_output_requests) == 2
    evolution_request = provider.structured_output_requests[1]
    assert evolution_request.provider == "fake"
    assert evolution_request.model_id == "fake-scenario-evolution"
    assert "scenario" in evolution_request.schema_name.casefold()
    assert "evolution" in evolution_request.schema_name.casefold()
    schema_text = json.dumps(evolution_request.schema, sort_keys=True).casefold()
    for field_name in ("content", "current_scene"):
        assert field_name in schema_text
    assert "starting_scene" not in schema_text
    for blocked_field_name in ("title", "premise", "player_role"):
        assert blocked_field_name not in schema_text
    evolution_prompt = "\n".join(
        message.body.casefold() for message in evolution_request.messages
    )
    assert "i relight the beacon lens." in evolution_prompt
    assert "the beacon lens answers in natural prose" in evolution_prompt
    assert "json" not in evolution_prompt

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I shield my eyes from the red light.",
            speaker_name="Mara",
        )
    )

    second_narrator_request = provider.chat_requests[1]
    assert "Current scene: The beacon gallery is hot with warning light." in (
        second_narrator_request.scenario_instructions
    )
    assert "Starting scene:" not in second_narrator_request.scenario_instructions
    assert len(provider.structured_output_requests) == 3
    assert (
        provider.structured_output_requests[2].schema_name
        == "state_memory_extraction"
    )


def test_scenario_evolution_not_due_records_configured_interval_skip_job(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={
            "starting_scene": "The beacon gutters in the tower.",
            "current_scene": "The lower gate waits under ash.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    previous_player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I climb the lower stair.",
    )
    previous_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="The lower gate groans shut behind Mara.",
        provider="fake",
        model="fake-chat",
    )
    repositories.record_save_scenario_evolution(
        save_id=save.id,
        title=scenario.title,
        premise=scenario.premise,
        player_role=scenario.player_role,
        content={
            "starting_scene": "The beacon gutters in the tower.",
            "current_scene": "The lower gate waits under ash.",
        },
        reason="current_scene: The lower gate became the durable setup.",
        provider="fake",
        model="fake-scenario-evolution",
        source_message_id=previous_narrator.id,
        source_message_ids=(previous_player.id, previous_narrator.id),
    )
    current_player = repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Mara",
        body="I pause at the next landing.",
    )
    current_narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name="Narrator",
        body="Red light leaks down from the beacon gallery.",
        provider="fake",
        model="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-scenario-evolution",
        display_name="Fake Scenario Evolution",
        capabilities=["structured_output"],
        context_window=8192,
    )
    repositories.set_model_preference(
        task="scenario_evolution",
        provider="fake",
        model_id="fake-scenario-evolution",
    )
    repositories.set_app_setting(SCENARIO_EVOLUTION_TURN_INTERVAL_SETTING, 2)
    events: list[str] = []
    provider = RecordingScenarioEvolutionProvider("fake", events=events)
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(ContextSearchResult()),
    )

    skipped = asyncio.run(
        service._evolve_scenario_if_configured(
            save_id=save.id,
            player_message_id=current_player.id,
            narrator_message_id=current_narrator.id,
        )
    )

    assert not isinstance(skipped, str)
    assert skipped.status == "skipped"
    assert skipped.result == {
        "scenario_update_id": None,
        "section_update_count": 0,
        "skip_reason": "not_due",
        "turn_interval": 2,
        "narrator_turns_since_update": 1,
    }
    scenario_jobs = [
        job
        for job in repositories.list_jobs_by_status(("succeeded",))
        if job.type == "scenario_evolution"
    ]
    assert len(scenario_jobs) == 1
    assert scenario_jobs[0].result == skipped.result
    assert provider.structured_output_requests == []

    repositories.set_app_setting(SCENARIO_EVOLUTION_TURN_INTERVAL_SETTING, 1)

    due = asyncio.run(
        service._evolve_scenario_if_configured(
            save_id=save.id,
            player_message_id=current_player.id,
            narrator_message_id=current_narrator.id,
        )
    )

    assert due == "succeeded"
    assert len(provider.structured_output_requests) == 1


def test_retired_character_interaction_state_extraction_has_no_special_guidance(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="character_interaction",
        title="Interview with the Sealed Archivist",
        premise="A tense private conversation in a forbidden archive.",
        player_role="Investigator",
        content={
            "featured_character": "Mael",
            "opening_message": "Mael slides the red index across the table.",
            "relationship": "Mael distrusts the investigator but needs help.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Interview")
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-state-memory",
        display_name="Fake State Memory",
        capabilities=["structured_output"],
        context_window=8192,
    )
    repositories.set_model_preference(
        task="state_memory",
        provider="fake",
        model_id="fake-state-memory",
    )
    events: list[str] = []
    provider = RecordingStateMemoryProvider("fake", events=events)
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(),
            events=events,
        ),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="What did you hide in the red index?",
            speaker_name="Mara",
        )
    )

    assert events == ["context_search", "narrator_chat", "state_memory_extraction"]
    assert len(provider.structured_output_requests) == 1
    extraction_request = provider.structured_output_requests[0]
    assert extraction_request.provider == "fake"
    assert extraction_request.model_id == "fake-state-memory"
    prompt = "\n".join(message.body.lower() for message in extraction_request.messages)
    assert "- type: character_interaction" in prompt
    assert "featured_character: mael" in prompt
    assert "mael distrusts the investigator but needs help" in prompt
    assert "revealed character traits" not in prompt
    assert "relationship dynamics" not in prompt
    assert "avoid broad world-state" not in prompt


def test_first_contact_state_extraction_prompt_tracks_partial_knowledge(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="first_contact_exploration",
        title="Songs Under Europa",
        premise="A survey crew finds patterned signals under the ice.",
        player_role="Mission linguist",
        content={
            "mission_profile": "Survey the hidden ocean and avoid hostile contact.",
            "exploration_target": "A black-water cavern beneath the ice shelf.",
            "translation_progress": (
                "Three descending pulses may mean open water; louder pulses "
                "were falsely assumed to be threats."
            ),
            "opening_message": "Blue light pulses beneath the ice.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Europa Contact")
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-state-memory",
        display_name="Fake State Memory",
        capabilities=["structured_output"],
        context_window=8192,
    )
    repositories.set_model_preference(
        task="state_memory",
        provider="fake",
        model_id="fake-state-memory",
    )
    events: list[str] = []
    provider = RecordingStateMemoryProvider("fake", events=events)
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(),
            events=events,
        ),
    )

    asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I answer with three descending pulses and wait.",
            speaker_name="Mara",
        )
    )

    assert events == ["context_search", "narrator_chat", "state_memory_extraction"]
    assert len(provider.structured_output_requests) == 1
    extraction_request = provider.structured_output_requests[0]
    assert extraction_request.provider == "fake"
    assert extraction_request.model_id == "fake-state-memory"
    prompt = "\n".join(message.body.lower() for message in extraction_request.messages)
    assert "- type: first_contact_exploration" in prompt
    assert "translation_progress" in prompt
    assert "observed facts" in prompt
    assert "hypotheses" in prompt
    assert "confirmed meanings" in prompt
    assert "false assumptions" in prompt
    assert "contamination" in prompt
    assert "mission.objective" in prompt
    assert "translation.<signal>.confirmed_meanings" in prompt


def test_submit_player_turn_does_not_fail_when_state_memory_extraction_fails(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-state-memory",
        display_name="Fake State Memory",
        capabilities=["structured_output"],
        context_window=8192,
    )
    repositories.set_model_preference(
        task="state_memory",
        provider="fake",
        model_id="fake-state-memory",
    )
    events: list[str] = []
    provider = RecordingStateMemoryProvider(
        "fake",
        events=events,
        structured_error=RuntimeError("structured extraction unavailable"),
    )
    service = ChatService(
        repositories=repositories,
        providers={"fake": provider},
        context_search_service=ScriptedContextSearch(
            ContextSearchResult(),
            events=events,
        ),
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert events == ["context_search", "narrator_chat", "state_memory_extraction"]
    assert len(provider.structured_output_requests) == 1
    persisted_messages = repositories.list_messages(save.id)
    assert [message.role for message in persisted_messages] == ["player", "narrator"]
    assert persisted_messages[0] == result.player_message
    assert persisted_messages[1] == result.narrator_message
    assert result.narrator_message.body == (
        "fake narrator: I climb toward the beacon lens."
    )
    assert repositories.list_world_state(save.id) == []
    assert repositories.list_memories(save.id) == []


def test_submit_player_turn_continues_when_summarization_fails(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.save_provider_model(
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
        display_name="Claude 3.5 Sonnet",
        capabilities=["chat"],
        context_window=8192,
    )
    repositories.set_model_preference(
        task="chat",
        provider="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
    )
    events: list[str] = []
    provider = RecordingChatProvider("openrouter")

    original_chat = provider.chat

    async def recording_chat(request: ChatRequest) -> ChatResponse:
        events.append("narrator_chat")
        return await original_chat(request)

    provider.chat = recording_chat  # type: ignore[method-assign]

    class PersistedMessageContextSearch(ScriptedContextSearch):
        async def search(
            self,
            *,
            save_id: str,
            player_message_id: str,
        ) -> ContextSearchResult:
            assert any(
                message.id == player_message_id
                for message in repositories.list_messages(save_id)
            )
            return await super().search(
                save_id=save_id,
                player_message_id=player_message_id,
            )

    context_search = PersistedMessageContextSearch(
        ContextSearchResult(),
        events=events,
    )
    summary_service = FailingSummaryService(
        events=events,
        error=RuntimeError("summary backend is down"),
    )
    service = ChatService(
        repositories=repositories,
        providers={"openrouter": provider},
        context_search_service=context_search,
        summary_service=summary_service,
    )

    result = asyncio.run(
        service.submit_player_turn(
            save_id=save.id,
            body="I climb toward the beacon lens.",
            speaker_name="Mara",
        )
    )

    assert events == ["summarization", "context_search", "narrator_chat"]
    assert summary_service.calls == [(save.id, 8192)]
    assert summary_service.pending_messages[0] == PendingMessageEstimate(
        body="I climb toward the beacon lens."
    )
    assert context_search.calls == [(save.id, result.player_message.id)]
    persisted_messages = repositories.list_messages(save.id)
    assert [message.role for message in persisted_messages] == ["player", "narrator"]
    assert persisted_messages[0] == result.player_message
    assert persisted_messages[1] == result.narrator_message
    assert result.narrator_message.body == (
        "openrouter narrator: I climb toward the beacon lens."
    )


def _append_completed_turns(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    count: int,
) -> tuple[str, str]:
    last_player_id = ""
    last_narrator_id = ""
    for index in range(count):
        player = repositories.append_message(
            save_id=save_id,
            role="player",
            speaker_name="Mara",
            body=f"I inspect the beacon lens {index}.",
        )
        narrator = repositories.append_message(
            save_id=save_id,
            role="narrator",
            speaker_name="Narrator",
            body=f"Captain Ilyra answers from the gallery {index}.",
            provider="fake",
            model="fake-chat",
        )
        last_player_id = player.id
        last_narrator_id = narrator.id
    return last_player_id, last_narrator_id


def _storage_paths(tmp_path: Path) -> StoragePaths:
    data_dir = tmp_path / "data"
    return StoragePaths(
        data_dir=data_dir,
        database_path=data_dir / "bragi.sqlite3",
        media_dir=data_dir / "media",
        state_dir=tmp_path / "state",
        cache_dir=tmp_path / "cache",
    )


def _flush_handlers(handlers: list[logging.Handler]) -> None:
    for handler in handlers:
        handler.flush()


def _chat_completion_jobs(
    repositories: PersistenceRepositories,
    save_id: str,
) -> list[dict[str, Any]]:
    rows = repositories.connection.execute(
        """
        SELECT id, save_id, status, payload_json, result_json, error,
               diagnostics_json
        FROM jobs
        WHERE save_id = ? AND type = 'chat_completion'
        ORDER BY created_at, rowid
        """,
        (save_id,),
    )
    jobs: list[dict[str, Any]] = []
    for row in rows:
        jobs.append(
            {
                "id": row["id"],
                "save_id": row["save_id"],
                "status": row["status"],
                "payload": json.loads(row["payload_json"]),
                "result": (
                    json.loads(row["result_json"])
                    if row["result_json"] is not None
                    else None
                ),
                "error": row["error"],
                "diagnostics": (
                    json.loads(row["diagnostics_json"])
                    if row["diagnostics_json"] is not None
                    else None
                ),
            }
        )
    return jobs


def _context_search_jobs(
    repositories: PersistenceRepositories,
    save_id: str,
) -> list[dict[str, Any]]:
    rows = repositories.connection.execute(
        """
        SELECT id, save_id, status, payload_json, result_json, error
        FROM jobs
        WHERE save_id = ? AND type = 'context_search'
        ORDER BY created_at, rowid
        """,
        (save_id,),
    )
    jobs: list[dict[str, Any]] = []
    for row in rows:
        jobs.append(
            {
                "id": row["id"],
                "save_id": row["save_id"],
                "status": row["status"],
                "payload": json.loads(row["payload_json"]),
                "result": (
                    json.loads(row["result_json"])
                    if row["result_json"] is not None
                    else None
                ),
                "error": row["error"],
            }
        )
    return jobs


def _post_turn_jobs(
    repositories: PersistenceRepositories,
    save_id: str,
) -> list[dict[str, Any]]:
    rows = repositories.connection.execute(
        """
        SELECT id, save_id, status, payload_json, result_json, error,
               diagnostics_json
        FROM jobs
        WHERE save_id = ? AND type = 'post_turn_jobs'
        ORDER BY created_at, rowid
        """,
        (save_id,),
    )
    jobs: list[dict[str, Any]] = []
    for row in rows:
        jobs.append(
            {
                "id": row["id"],
                "save_id": row["save_id"],
                "status": row["status"],
                "payload": json.loads(row["payload_json"]),
                "result": (
                    json.loads(row["result_json"])
                    if row["result_json"] is not None
                    else None
                ),
                "error": row["error"],
                "diagnostics": (
                    json.loads(row["diagnostics_json"])
                    if row["diagnostics_json"] is not None
                    else None
                ),
            }
        )
    return jobs


def _post_turn_child_status(job: dict[str, Any], name: str) -> str:
    result = job["result"]
    assert result is not None
    for child_job in result["jobs"]:
        if child_job["name"] == name:
            return str(child_job["status"])
    raise AssertionError(f"Missing post-turn child job: {name}")


def _post_turn_child_result(job: dict[str, Any], name: str) -> dict[str, Any]:
    result = job["result"]
    assert result is not None
    for child_job in result["jobs"]:
        if child_job["name"] == name:
            child_result = child_job["result"]
            assert isinstance(child_result, dict)
            return child_result
    raise AssertionError(f"Missing post-turn child job: {name}")


def _assert_event_before(events: list[str], earlier: str, later: str) -> None:
    assert earlier in events
    assert later in events
    assert events.index(earlier) < events.index(later)


def _requested_character_name(body: str) -> str:
    for line in body.splitlines():
        if line.startswith("Character: "):
            return line.removeprefix("Character: ").strip()
    raise AssertionError("request did not include a Character line")


def _database_text(repositories: PersistenceRepositories) -> str:
    return "\n".join(repositories.connection.iterdump())


def _debug_record_mentions(record: dict[str, Any], text: str) -> bool:
    needle = text.casefold()
    for key in ("event", "stage", "operation", "step", "message"):
        value = record.get(key)
        if isinstance(value, str) and needle in value.casefold():
            return True
    return False


def _restore_logger(
    logger: logging.Logger,
    original_handlers: tuple[logging.Handler, ...],
    *,
    original_level: int,
    original_propagate: bool,
) -> None:
    for handler in logger.handlers:
        if handler not in original_handlers:
            handler.close()
    logger.handlers[:] = list(original_handlers)
    logger.setLevel(original_level)
    logger.propagate = original_propagate


def test_look_around_returns_answer_without_appending_chronicle(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower at the sea edge.",
        player_role="Keeper",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name=None,
        body="Mara holds a brass lens beside the beacon controls.",
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    provider = RecordingLookAroundProvider("fake")

    result = asyncio.run(
        ChatService(
            repositories=repositories,
            providers={"fake": provider},
        ).look_around(
            save_id=save.id,
            query="Inspect the brass lens.",
        )
    )

    assert result.answer == (
        "fake narrator: Look around request: Inspect the brass lens."
    )
    assert isinstance(result.markdown_blocks, tuple)
    assert [message.id for message in repositories.list_messages(save.id)] == [
        narrator.id
    ]
    assert result.latest_narrator_message_id == narrator.id
    assert result.context_observation_id is not None
    [observation] = repositories.list_context_observations(save.id)
    assert observation.observation_type == "look_around"
    assert observation.claim == result.answer
    assert observation.source_message_ids == [narrator.id]
    assert observation.metadata["query"] == "Inspect the brass lens."
    request = provider.chat_requests[-1]
    assert request.messages[-1].role == "player"
    assert request.messages[-1].body == "Look around request: Inspect the brass lens."
    assert "Do not advance the chronicle" in request.turn_directive


def test_look_around_applies_child_content_rating_to_provider_output(
    repositories: PersistenceRepositories,
) -> None:
    child = repositories.create_user(
        username="Ilyra",
        role="child",
        password_hash="hash",
    )
    repositories.set_scoped_setting(
        scope="user",
        scope_id=child.id,
        key=CONTENT_FILTER_RATING_SETTING,
        value="g",
    )
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower at the sea edge.",
        player_role="Keeper",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name=None,
        body="Mara studies the beacon controls.",
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    _configure_content_safety_model(repositories)
    provider = RecordingLookAroundProvider(
        "fake",
        answer_body="Blood streaks the floor after the guard is stabbed.",
    )
    safety_provider = ScriptedContentSafetyProvider("block")

    result = asyncio.run(
        ChatService(
            repositories=repositories,
            providers={"fake": provider, "safety": safety_provider},
        ).look_around(
            save_id=save.id,
            query="Inspect the floor.",
            current_user_id=child.id,
        )
    )

    assert result.answer == CONTENT_FILTER_TRANSITION
    [observation] = repositories.list_context_observations(save.id)
    assert observation.claim == CONTENT_FILTER_TRANSITION


def test_look_around_requires_latest_narrator_state(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower at the sea edge.",
        player_role="Keeper",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.append_message(
        save_id=save.id,
        role="player",
        speaker_name="Keeper",
        body="I inspect the brass lens.",
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )

    with pytest.raises(ValueError, match="latest narrator"):
        asyncio.run(
            ChatService(
                repositories=repositories,
                providers={"fake": RecordingLookAroundProvider("fake")},
            ).look_around(
                save_id=save.id,
                query="Inspect the brass lens.",
            )
        )


def test_look_around_rejects_fade_to_black_state(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower at the sea edge.",
        player_role="Keeper",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name=None,
        body=FADE_TO_BLACK_TRANSITION,
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )

    with pytest.raises(ValueError, match="fade-to-black"):
        asyncio.run(
            ChatService(
                repositories=repositories,
                providers={"fake": RecordingLookAroundProvider("fake")},
            ).look_around(
                save_id=save.id,
                query="Inspect the room.",
            )
        )
    assert repositories.list_context_observations(save.id) == []
    assert repositories.list_context_update_suggestions(save.id) == []


def test_look_around_queues_structured_world_update_suggestions(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower at the sea edge.",
        player_role="Keeper",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    narrator = repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name=None,
        body="A locked prism rests inside the brass lens.",
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    repositories.set_model_preference(
        task="context_update",
        provider="fake",
        model_id="fake-structured",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-structured",
        display_name="Fake Structured",
        capabilities=["structured_output"],
    )
    provider = RecordingLookAroundProvider(
        "fake",
        structured_data={
            "suggestions": [
                {
                    "update_type": "upsert",
                    "entity_type": "world_state",
                    "field_path": "scene.brass_lens",
                    "proposed_value": "locked prism inside",
                    "reason": "The look-around answer established the lens contents.",
                    "confidence": 0.82,
                }
            ]
        },
    )

    result = asyncio.run(
        ChatService(
            repositories=repositories,
            providers={"fake": provider},
        ).look_around(
            save_id=save.id,
            query="What is inside the brass lens?",
        )
    )

    assert result.update_counts["suggestions"] == 1
    assert [message.id for message in repositories.list_messages(save.id)] == [
        narrator.id
    ]
    [suggestion] = repositories.list_context_update_suggestions(save.id)
    assert suggestion.status == "pending"
    assert suggestion.entity_type == "world_state"
    assert suggestion.field_path == "scene.brass_lens"
    assert suggestion.source_message_ids == [narrator.id]
    assert suggestion.proposed_value == "locked prism inside"
    [structured_request] = provider.structured_output_requests
    assert structured_request.schema_name == "look_around_updates"
    validate_strict_json_schema(structured_request.schema)
    request_properties = cast(dict[str, Any], structured_request.schema["properties"])
    suggestions_schema = cast(dict[str, Any], request_properties["suggestions"])
    suggestion_items = cast(dict[str, Any], suggestions_schema["items"])
    suggestion_properties = cast(dict[str, Any], suggestion_items["properties"])
    proposed_value_schema = cast(
        dict[str, Any],
        suggestion_properties["proposed_value"],
    )
    proposed_value_branches = cast(list[dict[str, Any]], proposed_value_schema["anyOf"])
    assert {"type": "object", "additionalProperties": True} not in (
        proposed_value_branches
    )
    assert {"type": "array", "items": {}} not in proposed_value_branches
    assert "What is inside the brass lens?" in structured_request.messages[-1].body


def test_look_around_derives_markdown_blocks_from_answer(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="A watchtower at the sea edge.",
        player_role="Keeper",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.append_message(
        save_id=save.id,
        role="narrator",
        speaker_name=None,
        body="Mara holds a brass lens beside the beacon controls.",
    )
    repositories.set_model_preference(
        task="chat",
        provider="fake",
        model_id="fake-chat",
    )
    answer_body = (
        "The brass lens rests on a velvet pad.\n"
        "- faintly etched runes\n"
        "- a small keyhole\n"
        "Use the **iron key** or the `silver key` to open it."
    )
    provider = RecordingLookAroundProvider("fake", answer_body=answer_body)

    result = asyncio.run(
        ChatService(
            repositories=repositories,
            providers={"fake": provider},
        ).look_around(
            save_id=save.id,
            query="Inspect the brass lens.",
        )
    )

    assert result.answer == answer_body
    assert result.markdown_blocks, "expected markdown blocks to be parsed"
    block_kinds = [block.kind.value for block in result.markdown_blocks]
    assert "paragraph" in block_kinds
    assert "bullet_item" in block_kinds
    span_kinds = [
        span.kind.value
        for block in result.markdown_blocks
        for span in block.spans
    ]
    assert "strong" in span_kinds
    assert "inline_code" in span_kinds
    strong_texts = [
        span.text
        for block in result.markdown_blocks
        for span in block.spans
        if span.kind.value == "strong"
    ]
    assert strong_texts == ["iron key"]
    code_texts = [
        span.text
        for block in result.markdown_blocks
        for span in block.spans
        if span.kind.value == "inline_code"
    ]
    assert code_texts == ["silver key"]
    [observation] = repositories.list_context_observations(save.id)
    assert observation.claim == answer_body
