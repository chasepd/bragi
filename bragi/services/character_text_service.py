"""Side-channel character texting service."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from hashlib import sha256
from random import Random
from typing import TYPE_CHECKING, Protocol, cast

from bragi.app_logging import (
    exception_log_fields,
    log_debug_event,
    log_error_event,
    log_event,
)
from bragi.persistence.models import (
    ActiveThreadRecord,
    CharacterContactStateRecord,
    CharacterRecord,
    CharacterTextMessageAttachmentRecord,
    CharacterTextMessageRecord,
    CharacterTextProactiveTriggerRecord,
    CharacterTextThreadRecord,
    DatingRouteStateRecord,
    MediaAssetRecord,
    MemoryRecord,
    MessageRecord,
    MessageVisibilityRecord,
    SceneSnapshotRecord,
    SummaryRecord,
    WorldStateRecord,
)
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    CHAT_TURN_DIRECTIVE_PURPOSE_CHARACTER_TEXT,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ProviderClient,
    ProviderRetryProgress,
    ProviderRetryProgressCallback,
    StructuredOutputProvider,
    StructuredOutputRequest,
)
from bragi.providers.system_prompt import CHARACTER_TEXT_RESPONSE_STYLE_SECTION
from bragi.services.active_thread_lifecycle import (
    active_thread_is_prompt_visible,
    normalize_active_thread_visibility,
)
from bragi.services.character_registry_service import (
    CharacterRegistryReferenceImageRow,
    CharacterRegistryService,
)
from bragi.services.character_text_context import (
    canonical_character_text_context_messages,
    uploaded_photo_descriptions_by_message_id,
)
from bragi.services.character_text_world_update_service import (
    CharacterTextWorldUpdateResult,
    CharacterTextWorldUpdateService,
    character_text_source_ref,
)
from bragi.services.context_assembly import compact_scenario_instructions
from bragi.services.dating_route_policy import (
    intimacy_profile_guidance,
    next_reasonable_step,
)
from bragi.services.dating_route_profile_service import DatingRouteProfileService
from bragi.services.dating_route_service import DatingRouteService
from bragi.services.generation_settings import (
    chat_generation_settings,
    chat_request_with_reasoning_override,
)
from bragi.services.knowledge_boundary import (
    knowledge_edge_allows_prompt_use,
    message_visible_to_present_characters,
    normalized_knowledge_target_type,
)
from bragi.services.media_service import CharacterTextUploadedPhoto
from bragi.services.model_capabilities import (
    STRUCTURED_OUTPUT_CAPABILITIES,
    known_model_is_unavailable,
    model_supports_any_capability,
)
from bragi.services.model_preferences import (
    roleplay_model_preference,
    roleplay_model_task,
)
from bragi.services.openrouter_routing_settings import request_with_openrouter_routing
from bragi.services.phrase_denylist import (
    GENERATED_PHRASE_GUARD_MAX_ATTEMPTS,
    PhraseDenylistViolation,
    denied_phrase_violations,
    effective_generated_phrase_denylist,
    first_phrase_violation_diagnostic,
    summarize_phrase_policy_violations,
)
from bragi.services.provider_fallbacks import structured_output_with_fallback
from bragi.services.sexual_content_safety import is_fade_to_black_message
from bragi.services.text_script_policy import (
    ScriptPolicyViolation,
    allowed_generated_scripts,
    first_violation_diagnostic,
    script_guard_mode,
    summarize_script_policy_violations,
    text_script_violations,
)
from bragi.world_time_model import format_world_time_from_snapshot

if TYPE_CHECKING:
    from bragi.application.chronicle import ChronicleMarkdownBlock
    from bragi.services.prompt_inspection import PromptInspectionStore

CHARACTER_TEXTS_ENABLED_SETTING = "character_texts_enabled"
CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING = (
    "character_text_proactive_random_chance_percent"
)
DEFAULT_CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_PERCENT = 15
MIN_CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_PERCENT = 0
MAX_CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_PERCENT = 100
CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING = (
    "character_text_proactive_random_cooldown_turns"
)
DEFAULT_CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_TURNS = 4
MIN_CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_TURNS = 0
MAX_CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_TURNS = 50
CHARACTER_TEXT_SEND_JOB_TYPE = "character_text_send"
CHARACTER_TEXT_ATTACHMENT_DECISION_TASK = "response_planning"
CHARACTER_TEXT_GROUP_RESPONSE_ASSESSMENT_TASK = "response_planning"
CHARACTER_TEXT_GROUP_RESPONSE_ASSESSMENT_SCHEMA_NAME = (
    "character_text_group_response_assessment"
)
CHARACTER_TEXT_GROUP_RESPONSE_MAX_REPLIES = 3
_CHARACTER_TEXT_PROMPT_KIND = "character_text_prompt"
_CHARACTER_TEXT_PROMPT_TITLE = "Character text prompt"
_MAX_THREAD_CONTEXT_MESSAGES = 30
_MAX_THREAD_MEMORY_MESSAGES = 18
_MAX_THREAD_MEMORY_CHARS = 1800
_MAX_THREAD_MEMORY_LINE_CHARS = 180
_MAX_CHARACTER_TEXT_SOURCE_CHRONICLE_MESSAGES = 2
_MAX_CHARACTER_TEXT_RECENT_CHRONICLE_MESSAGES = 4
_MAX_CHARACTER_TEXT_CHRONICLE_SCAN_MESSAGES = 12
_MAX_CHARACTER_TEXT_CHRONICLE_LINE_CHARS = 360
_THREAD_MEMORY_KEYWORDS = (
    "?",
    "ask",
    "asked",
    "bring",
    "joke",
    "meet",
    "nickname",
    "owe",
    "plan",
    "promise",
    "promised",
    "same place",
    "still",
)
_TEXT_NATIVE_THREAD_TERMS = frozenset(
    {
        "text",
        "texts",
        "texting",
        "message",
        "messages",
        "dm",
        "dms",
        "sms",
        "phone",
    }
)
_UNAVAILABLE_CHARACTER_PHRASES = (
    "asleep",
    "sleeping",
    "unconscious",
    "offline",
    "phone off",
    "no signal",
    "unavailable",
)
_NEAR_DUPLICATE_MIN_NORMALIZED_LENGTH = 24
_NEAR_DUPLICATE_RATIO = 0.9
_LEAKED_SENT_AT_PREFIX_RE = re.compile(
    r"^\s*(?:>\s*)?sent\s+at\s+(?:"
    r"(?=[^:\n]{1,80}:)"
    r"(?=[^:\n]*(?:early\s+morning|late\s+morning|morning|afternoon|evening|"
    r"night|overnight|dawn|dusk|noon|midday|midnight|monday|tuesday|"
    r"wednesday|thursday|friday|saturday|sunday|\b\d{1,2}(?::\d{2})?"
    r"\s*(?:am|pm)\b|\bday\s+\d+\b|\b\d{4}-\d{1,2}-\d{1,2}\b))"
    r"[^:\n]{1,80}:|"
    r"(?:early\s+morning|late\s+morning|morning|afternoon|evening|night|"
    r"overnight|dawn|dusk|noon|midday|midnight)\s*(?:-|,)"
    r")\s*",
    re.IGNORECASE,
)
_IDENTITY_TEXT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
    }
)
_IDENTITY_DECLARATION_NAME_BOUNDARY_RE = r"(?![\w'-])"


@dataclass(frozen=True)
class CharacterTextContactPermission:
    allowed: bool
    source: str
    reason: str
    source_message_id: str | None = None
    source_text_message_id: str | None = None


@dataclass(frozen=True)
class CharacterTextContact:
    id: str
    name: str
    role: str
    status: str
    is_player_character: bool
    player_has_character_number: bool
    character_has_player_number: bool
    player_number_permission: CharacterTextContactPermission
    character_number_permission: CharacterTextContactPermission
    contact_name: str = ""
    thread_id: str | None = None
    latest_message_id: str | None = None
    latest_message_body: str = ""
    latest_message_markdown_blocks: tuple[ChronicleMarkdownBlock, ...] = ()
    latest_message_sender: str | None = None
    latest_message_at: str | None = None
    latest_message_read_at: str | None = None
    reference_image: CharacterRegistryReferenceImageRow | None = None


@dataclass(frozen=True)
class CharacterTextMessageAction:
    action_id: str
    label: str


@dataclass(frozen=True)
class CharacterTextMessageAttachment:
    id: str
    kind: str
    status: str
    media_asset_id: str | None
    mime_type: str | None
    provider: str | None
    model: str | None
    prompt_preview: str
    error: str | None
    created_at: str | None


@dataclass(frozen=True)
class CharacterTextMessage:
    id: str
    thread_id: str
    character_id: str | None
    sender: str
    sender_character_id: str | None
    sender_display_name: str
    body: str
    markdown_blocks: tuple[ChronicleMarkdownBlock, ...]
    attachments: tuple[CharacterTextMessageAttachment, ...]
    actions: tuple[CharacterTextMessageAction, ...]
    provider: str | None
    model: str | None
    token_estimate: int | None
    delivery_status: str
    delivery_error: str | None
    delivery_job_id: str | None
    delivery_attempt: int
    created_at: str | None
    in_world_sent_at: str | None = None
    delivered_at: str | None = None
    read_at: str | None = None
    reply_to_message_id: str | None = None
    proactive_reason: str = ""
    proactive_trigger_type: str = ""
    revision_count: int = 0
    edited_at: str | None = None


@dataclass(frozen=True)
class CharacterTextThreadParticipant:
    character_id: str
    name: str
    ordinal: int


@dataclass(frozen=True)
class CharacterTextThread:
    id: str
    character_id: str | None
    title: str
    status: str
    kind: str
    participants: tuple[CharacterTextThreadParticipant, ...]
    created_at: str | None
    updated_at: str | None
    messages: tuple[CharacterTextMessage, ...] = ()


@dataclass(frozen=True)
class CharacterTextsModel:
    save_id: str
    enabled: bool
    contacts: tuple[CharacterTextContact, ...]
    repair_contacts: tuple[CharacterTextContact, ...]
    threads: tuple[CharacterTextThread, ...]


@dataclass(frozen=True)
class CharacterTextSendResult:
    save_id: str
    thread: CharacterTextThread
    player_message: CharacterTextMessage
    reply: CharacterTextMessage
    world_update: CharacterTextWorldUpdateResult | None = None


@dataclass(frozen=True)
class CharacterTextThreadSendResult:
    save_id: str
    thread: CharacterTextThread
    player_message: CharacterTextMessage
    replies: tuple[CharacterTextMessage, ...]
    world_update: CharacterTextWorldUpdateResult | None = None


@dataclass(frozen=True)
class CharacterTextQueuedSendResult:
    save_id: str
    thread: CharacterTextThread
    player_message: CharacterTextMessage


@dataclass(frozen=True)
class CharacterTextQueuedSpontaneousResult:
    save_id: str
    thread: CharacterTextThread
    message: CharacterTextMessage


@dataclass(frozen=True)
class CharacterTextReadResult:
    save_id: str
    thread: CharacterTextThread
    updated_message_ids: tuple[str, ...]


@dataclass(frozen=True)
class CharacterTextProactiveResult:
    save_id: str
    status: str
    reason: str = ""
    candidate_count: int = 0
    trigger_key: str = ""
    thread: CharacterTextThread | None = None
    message: CharacterTextMessage | None = None
    world_update: CharacterTextWorldUpdateResult | None = None

    def to_json(self) -> dict[str, object]:
        result: dict[str, object] = {
            "save_id": self.save_id,
            "status": self.status,
            "reason": self.reason,
            "candidate_count": self.candidate_count,
        }
        if self.trigger_key:
            result["trigger_key"] = self.trigger_key
        if self.thread is not None:
            result["thread_id"] = self.thread.id
            result["character_id"] = self.thread.character_id
        if self.message is not None:
            result["message_id"] = self.message.id
        if self.world_update is not None:
            result["world_update"] = self.world_update.to_json()
        return result


@dataclass(frozen=True)
class CharacterTextSpontaneousResult:
    save_id: str
    thread: CharacterTextThread
    message: CharacterTextMessage
    world_update: CharacterTextWorldUpdateResult | None = None


@dataclass(frozen=True)
class _CharacterTextGenerationResult:
    request: ChatRequest
    response: ChatResponse
    body: str


@dataclass(frozen=True)
class _CharacterTextIdentity:
    character_name: str
    player_name: str


@dataclass(frozen=True)
class _CharacterTextIdentityViolation:
    reason: str


@dataclass(frozen=True)
class _ProactiveTextCandidate:
    character: CharacterRecord
    trigger_key: str
    trigger_type: str
    source_type: str
    source_id: str
    source_message_id: str | None
    reason: str
    context_lines: tuple[str, ...]
    priority: int


@dataclass(frozen=True)
class _ProactiveCandidateGate:
    candidate: _ProactiveTextCandidate | None
    reason: str = ""


class CharacterTextAttachmentMediaRunner(Protocol):
    async def upload_character_text_player_photo(
        self,
        *,
        save_id: str,
        text_message: CharacterTextMessageRecord,
        sender_character_id: str,
        image_bytes: bytes,
        filename: str | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
    ) -> CharacterTextUploadedPhoto: ...

    def cleanup_character_text_uploaded_photo(
        self,
        *,
        save_id: str,
        asset: MediaAssetRecord,
    ) -> None: ...

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
    ) -> MediaAssetRecord: ...

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
    ) -> MediaAssetRecord: ...


@dataclass(frozen=True)
class _AttachmentDecision:
    kind: str
    visual_prompt: str
    reason: str
    wearing: str = ""
    current_action: str = ""
    facial_expression: str = ""


@dataclass(frozen=True)
class _GroupResponseAssessment:
    character: CharacterRecord
    should_respond: bool
    response_intent: str
    reason: str
    confidence: float
    priority: int


class CharacterTextService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        providers: Mapping[str, object],
        media_service: CharacterTextAttachmentMediaRunner | None = None,
        prompt_inspection_store: PromptInspectionStore | None = None,
    ) -> None:
        self.repositories = repositories
        self.providers = providers
        self.media_service = media_service
        self.prompt_inspection_store = prompt_inspection_store

    def is_enabled(self, save_id: str) -> bool:
        configured = self.repositories.get_effective_setting(
            CHARACTER_TEXTS_ENABLED_SETTING,
            save_id=save_id,
        )
        if isinstance(configured, bool):
            return configured
        details = self.repositories.load_save_details(save_id)
        return details is not None and details.scenario.type == "dating_sim"

    def can_player_text_character(self, *, save_id: str, character_id: str) -> bool:
        if not self.is_enabled(save_id):
            return False
        character = self.repositories.get_character(character_id)
        if (
            character is None
            or character.save_id != save_id
            or character.is_player_character
        ):
            return False
        return self.repositories.character_text_outbound_allowed(
            save_id=save_id,
            character_id=character_id,
        )

    def prepare_spontaneous_text(
        self,
        *,
        save_id: str,
        character_id: str,
    ) -> CharacterTextThread:
        if not self.is_enabled(save_id):
            raise ValueError("Character texts are not enabled for this save")
        character = self.repositories.get_character(character_id)
        if (
            character is None
            or character.save_id != save_id
            or character.is_player_character
        ):
            raise ValueError(f"Unknown textable character id: {character_id}")
        if not self.repositories.character_text_outbound_allowed(
            save_id=save_id,
            character_id=character.id,
        ):
            raise ValueError("Player does not have this character's number")
        if not self.repositories.can_character_proactively_text(
            save_id=save_id,
            character_id=character.id,
        ):
            raise ValueError("Character does not have the player's number")
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="chat",
        )
        if preference is None:
            raise ValueError("No chat model configured")
        provider = self.providers.get(preference.provider)
        if provider is None or not callable(getattr(provider, "chat", None)):
            raise ValueError("Configured chat provider is unavailable")
        thread = self.repositories.get_or_create_character_text_thread(
            save_id=save_id,
            character_id=character.id,
            title=character.name,
        )
        if self.repositories.has_active_character_text_delivery(
            save_id=save_id,
            thread_id=thread.id,
        ):
            raise ValueError("A text send is already pending for this conversation")
        return _thread_model(
            thread,
            messages=_message_models(
                self.repositories,
                save_id=save_id,
                messages=tuple(
                    self.repositories.list_character_text_messages(
                        save_id=save_id,
                        thread_id=thread.id,
                    )
                ),
            ),
        )

    def queue_spontaneous_text(
        self,
        *,
        save_id: str,
        character_id: str,
    ) -> CharacterTextQueuedSpontaneousResult:
        prepared = self.prepare_spontaneous_text(
            save_id=save_id,
            character_id=character_id,
        )
        if prepared.character_id is None:
            raise ValueError("Spontaneous text thread is missing a character")
        character = self.repositories.get_character(prepared.character_id)
        if character is None or character.save_id != save_id:
            raise ValueError("Text character is no longer available")
        message = self.repositories.append_character_text_message(
            save_id=save_id,
            thread_id=prepared.id,
            character_id=character.id,
            sender="character",
            body="",
            delivery_status="pending",
            in_world_sent_at=_current_character_text_in_world_timestamp(
                repositories=self.repositories,
                save_id=save_id,
            ),
        )
        thread = self.repositories.get_character_text_thread(
            thread_id=prepared.id,
            save_id=save_id,
        )
        if thread is None:
            raise ValueError("Character text thread disappeared during queue")
        return CharacterTextQueuedSpontaneousResult(
            save_id=save_id,
            thread=_thread_model(
                thread,
                messages=_message_models(
                    self.repositories,
                    save_id=save_id,
                    messages=tuple(
                        self.repositories.list_character_text_messages(
                            save_id=save_id,
                            thread_id=thread.id,
                        )
                    ),
                ),
            ),
            message=_message_model(message),
        )

    def build_model(self, save_id: str) -> CharacterTextsModel:
        enabled = self.is_enabled(save_id)
        if not enabled:
            return CharacterTextsModel(
                save_id=save_id,
                enabled=False,
                contacts=(),
                repair_contacts=(),
                threads=(),
            )
        characters = self.repositories.list_characters(save_id)
        player = _player_character(save_id, self.repositories)
        contact_state_by_character_id = {
            state.character_id: state
            for state in self.repositories.list_character_contact_states(save_id)
            if player is not None and state.player_character_id == player.id
        }
        threads = self.repositories.list_character_text_threads(save_id)
        latest_by_thread = {
            thread.id: latest
            for thread in threads
            if (
                latest := _latest_contact_preview_message(
                    self.repositories.list_character_text_messages(
                        save_id=save_id,
                        thread_id=thread.id,
                    )
                )
            )
            is not None
        }
        thread_by_character_id = {
            thread.character_id: thread
            for thread in threads
            if thread.kind == "direct" and thread.character_id is not None
        }
        reference_images_by_character_id = _character_reference_images_by_id(
            self.repositories,
            save_id=save_id,
            character_ids=tuple(character.id for character in characters),
        )
        repair_contacts = tuple(
            _contact_model(
                character,
                contact_state=contact_state_by_character_id.get(character.id),
                thread=thread_by_character_id.get(character.id),
                latest=latest_by_thread.get(
                    thread_by_character_id[character.id].id
                )
                if character.id in thread_by_character_id
                else None,
                reference_image=reference_images_by_character_id.get(character.id),
            )
            for character in characters
            if (
                player is not None
                and not character.is_player_character
            )
        )
        contacts = tuple(
            contact
            for contact in repair_contacts
            if (
                contact.player_has_character_number
                or contact.character_has_player_number
                or contact.thread_id is not None
            )
        )
        return CharacterTextsModel(
            save_id=save_id,
            enabled=True,
            contacts=contacts,
            repair_contacts=repair_contacts,
            threads=tuple(
                _thread_model(
                    thread,
                    participants=_thread_participant_models(
                        self.repositories,
                        save_id=save_id,
                        thread_id=thread.id,
                    ),
                    messages=(),
                )
                for thread in threads
            ),
        )

    def update_contact_state(
        self,
        *,
        save_id: str,
        character_id: str,
        player_has_character_number: bool,
        character_has_player_number: bool,
    ) -> CharacterTextsModel:
        if not self.is_enabled(save_id):
            raise ValueError("Character texts are not enabled for this save")
        character = self.repositories.get_character(character_id)
        if (
            character is None
            or character.save_id != save_id
            or character.is_player_character
        ):
            raise ValueError(f"Unknown textable character id: {character_id}")
        player = _player_character(save_id, self.repositories)
        if player is None:
            raise ValueError("Player character is required for character texts")
        self.repositories.set_character_contact_state(
            save_id=save_id,
            player_character_id=player.id,
            character_id=character.id,
            player_has_character_number=player_has_character_number,
            character_has_player_number=character_has_player_number,
        )
        return self.build_model(save_id)

    def get_thread_model(self, *, save_id: str, thread_id: str) -> CharacterTextThread:
        thread = self.repositories.get_character_text_thread(
            thread_id=thread_id,
            save_id=save_id,
        )
        if thread is None:
            raise ValueError(f"Unknown character text thread id: {thread_id}")
        return _thread_model(
            thread,
            participants=_thread_participant_models(
                self.repositories,
                save_id=save_id,
                thread_id=thread.id,
            ),
            messages=_message_models(
                self.repositories,
                save_id=save_id,
                messages=tuple(
                    self.repositories.list_character_text_messages(
                        save_id=save_id,
                        thread_id=thread.id,
                    )
                ),
            ),
        )

    def create_group_thread(
        self,
        *,
        save_id: str,
        title: str,
        character_ids: tuple[str, ...],
    ) -> CharacterTextThread:
        if not self.is_enabled(save_id):
            raise ValueError("Character texts are not enabled for this save")
        normalized_character_ids = tuple(
            dict.fromkeys(character_id.strip() for character_id in character_ids)
        )
        if len(normalized_character_ids) < 2:
            raise ValueError("Group texts require at least two contacts")
        for character_id in normalized_character_ids:
            character = self.repositories.get_character(character_id)
            if (
                character is None
                or character.save_id != save_id
                or character.is_player_character
            ):
                raise ValueError(f"Unknown textable character id: {character_id}")
            if not self.repositories.character_text_outbound_allowed(
                save_id=save_id,
                character_id=character.id,
            ):
                raise ValueError("Player does not have every character's number")
        thread = self.repositories.create_character_text_group_thread(
            save_id=save_id,
            title=title,
            character_ids=normalized_character_ids,
        )
        return _thread_model(
            thread,
            participants=_thread_participant_models(
                self.repositories,
                save_id=save_id,
                thread_id=thread.id,
            ),
            messages=(),
        )

    def mark_thread_read(
        self,
        *,
        save_id: str,
        thread_id: str,
        through_message_id: str | None = None,
    ) -> CharacterTextReadResult:
        updated = self.repositories.mark_character_text_thread_read(
            save_id=save_id,
            thread_id=thread_id,
            through_message_id=through_message_id,
        )
        return CharacterTextReadResult(
            save_id=save_id,
            thread=self.get_thread_model(save_id=save_id, thread_id=thread_id),
            updated_message_ids=tuple(message.id for message in updated),
        )

    def queue_thread_text_send(
        self,
        *,
        save_id: str,
        thread_id: str,
        body: str,
    ) -> CharacterTextQueuedSendResult:
        if not self.is_enabled(save_id):
            raise ValueError("Character texts are not enabled for this save")
        text = body.strip()
        if not text:
            raise ValueError("Character text body is required")
        thread = self.repositories.get_character_text_thread(
            thread_id=thread_id,
            save_id=save_id,
        )
        if thread is None:
            raise ValueError(f"Unknown character text thread id: {thread_id}")
        if thread.kind == "direct":
            if thread.character_id is None:
                raise ValueError("Direct text thread is missing a character")
            return self.queue_text_send(
                save_id=save_id,
                character_id=thread.character_id,
                body=text,
            )
        if thread.kind != "group":
            raise ValueError(f"Unsupported character text thread kind: {thread.kind}")
        if self.repositories.has_active_character_text_delivery(
            save_id=save_id,
            thread_id=thread.id,
        ):
            raise ValueError("A text send is already pending for this conversation")
        player = _player_character(save_id, self.repositories)
        if player is None:
            raise ValueError("Player character is required for character texts")
        participants = _group_thread_participant_characters(
            repositories=self.repositories,
            save_id=save_id,
            thread_id=thread.id,
        )
        if len(participants) < 2:
            raise ValueError("Group text thread has fewer than two participants")
        for participant in participants:
            if not self.repositories.character_text_outbound_allowed(
                save_id=save_id,
                character_id=participant.id,
            ):
                raise ValueError("Player does not have every character's number")
        player_message = self.repositories.append_character_text_message(
            save_id=save_id,
            thread_id=thread.id,
            character_id=None,
            sender="player",
            sender_character_id=player.id,
            body=text,
            delivery_status="pending",
            in_world_sent_at=_current_character_text_in_world_timestamp(
                repositories=self.repositories,
                save_id=save_id,
            ),
        )
        return CharacterTextQueuedSendResult(
            save_id=save_id,
            thread=_thread_model(
                thread,
                participants=_thread_participant_models(
                    self.repositories,
                    save_id=save_id,
                    thread_id=thread.id,
                ),
                messages=_message_models(
                    self.repositories,
                    save_id=save_id,
                    messages=tuple(
                        self.repositories.list_character_text_messages(
                            save_id=save_id,
                            thread_id=thread.id,
                        )
                    ),
                ),
            ),
            player_message=_message_model(
                player_message,
                sender_display_name=player.name,
            ),
        )

    def queue_text_send(
        self,
        *,
        save_id: str,
        character_id: str,
        body: str,
    ) -> CharacterTextQueuedSendResult:
        if not self.is_enabled(save_id):
            raise ValueError("Character texts are not enabled for this save")
        text = body.strip()
        if not text:
            raise ValueError("Character text body is required")
        character = self.repositories.get_character(character_id)
        if (
            character is None
            or character.save_id != save_id
            or character.is_player_character
        ):
            raise ValueError(f"Unknown textable character id: {character_id}")
        if not self.repositories.character_text_outbound_allowed(
            save_id=save_id,
            character_id=character.id,
        ):
            raise ValueError("Player does not have this character's number")
        details = self.repositories.load_save_details(save_id)
        if details is None:
            raise ValueError(f"Unknown save id: {save_id}")
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="chat",
        )
        if preference is None:
            raise ValueError("No chat model configured")
        provider = self.providers.get(preference.provider)
        if provider is None or not callable(getattr(provider, "chat", None)):
            raise ValueError("Configured chat provider is unavailable")
        thread = self.repositories.get_or_create_character_text_thread(
            save_id=save_id,
            character_id=character.id,
            title=character.name,
        )
        if self.repositories.has_active_character_text_delivery(
            save_id=save_id,
            thread_id=thread.id,
        ):
            raise ValueError("A text send is already pending for this conversation")
        player_message = self.repositories.append_character_text_message(
            save_id=save_id,
            thread_id=thread.id,
            character_id=character.id,
            sender="player",
            body=text,
            delivery_status="pending",
            in_world_sent_at=_current_character_text_in_world_timestamp(
                repositories=self.repositories,
                save_id=save_id,
            ),
        )
        return CharacterTextQueuedSendResult(
            save_id=save_id,
            thread=_thread_model(
                thread,
                messages=_message_models(
                    self.repositories,
                    save_id=save_id,
                    messages=tuple(
                        self.repositories.list_character_text_messages(
                            save_id=save_id,
                            thread_id=thread.id,
                        )
                    ),
                ),
            ),
            player_message=_message_model(player_message),
        )

    async def send_text(
        self,
        *,
        save_id: str,
        character_id: str,
        body: str,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        current_user_id: str | None = None,
    ) -> CharacterTextSendResult:
        queued = self.queue_text_send(
            save_id=save_id,
            character_id=character_id,
            body=body,
        )
        return await self.complete_queued_text_send(
            save_id=save_id,
            player_message_id=queued.player_message.id,
            retry_progress_callback=retry_progress_callback,
            current_user_id=current_user_id,
        )

    async def send_thread_text(
        self,
        *,
        save_id: str,
        thread_id: str,
        body: str,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        current_user_id: str | None = None,
    ) -> CharacterTextThreadSendResult:
        queued = self.queue_thread_text_send(
            save_id=save_id,
            thread_id=thread_id,
            body=body,
        )
        return await self.complete_queued_thread_text_send(
            save_id=save_id,
            player_message_id=queued.player_message.id,
            retry_progress_callback=retry_progress_callback,
            current_user_id=current_user_id,
        )

    async def complete_queued_thread_text_send(
        self,
        *,
        save_id: str,
        player_message_id: str,
        uploaded_photo_bytes: bytes | None = None,
        uploaded_photo_filename: str | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        current_user_id: str | None = None,
    ) -> CharacterTextThreadSendResult:
        player_message = self.repositories.get_character_text_message(
            save_id=save_id,
            message_id=player_message_id,
        )
        if player_message is None or player_message.sender != "player":
            raise ValueError(f"Unknown queued character text id: {player_message_id}")
        thread = self.repositories.get_character_text_thread(
            save_id=save_id,
            thread_id=player_message.thread_id,
        )
        if thread is None:
            raise ValueError("Character text thread disappeared during send")
        if thread.kind == "direct":
            result = await self.complete_queued_text_send(
                save_id=save_id,
                player_message_id=player_message_id,
                uploaded_photo_bytes=uploaded_photo_bytes,
                uploaded_photo_filename=uploaded_photo_filename,
                retry_progress_callback=retry_progress_callback,
                current_user_id=current_user_id,
            )
            return CharacterTextThreadSendResult(
                save_id=result.save_id,
                thread=result.thread,
                player_message=result.player_message,
                replies=(result.reply,),
                world_update=result.world_update,
            )
        if thread.kind != "group":
            raise ValueError(f"Unsupported character text thread kind: {thread.kind}")
        if player_message.delivery_status not in {"pending", "retrying"}:
            raise ValueError("Character text send is not pending")
        details = self.repositories.load_save_details(save_id)
        if details is None:
            raise ValueError(f"Unknown save id: {save_id}")
        if uploaded_photo_bytes is not None:
            try:
                await self._attach_uploaded_player_photo(
                    save_id=save_id,
                    player_message=player_message,
                    uploaded_photo_bytes=uploaded_photo_bytes,
                    uploaded_photo_filename=uploaded_photo_filename,
                )
            except Exception as exc:
                self.repositories.update_character_text_delivery(
                    save_id=save_id,
                    message_id=player_message.id,
                    status="failed",
                    error=str(exc) or exc.__class__.__name__,
                )
                raise
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="chat",
        )
        if preference is None:
            player_message = self.repositories.update_character_text_delivery(
                save_id=save_id,
                message_id=player_message.id,
                status="sent",
                error=None,
            )
            return CharacterTextThreadSendResult(
                save_id=save_id,
                thread=self.get_thread_model(save_id=save_id, thread_id=thread.id),
                player_message=_message_models(
                    self.repositories,
                    save_id=save_id,
                    messages=(player_message,),
                )[0],
                replies=(),
            )
        provider = self.providers.get(preference.provider)
        chat = getattr(provider, "chat", None)
        if not callable(chat):
            player_message = self.repositories.update_character_text_delivery(
                save_id=save_id,
                message_id=player_message.id,
                status="sent",
                error=None,
            )
            return CharacterTextThreadSendResult(
                save_id=save_id,
                thread=self.get_thread_model(save_id=save_id, thread_id=thread.id),
                player_message=_message_models(
                    self.repositories,
                    save_id=save_id,
                    messages=(player_message,),
                )[0],
                replies=(),
            )
        participants = _group_thread_participant_characters(
            repositories=self.repositories,
            save_id=save_id,
            thread_id=thread.id,
        )
        assessments = await self._group_response_assessments(
            save_id=save_id,
            thread=thread,
            player_message=player_message,
            participants=participants,
        )
        responders = tuple(
            assessment
            for assessment in assessments
            if assessment.should_respond
        )[:CHARACTER_TEXT_GROUP_RESPONSE_MAX_REPLIES]
        if not responders:
            player_message = self.repositories.update_character_text_delivery(
                save_id=save_id,
                message_id=player_message.id,
                status="sent",
                error=None,
            )
            refresh_character_text_thread_memory(
                repositories=self.repositories,
                save_id=save_id,
                thread_id=thread.id,
            )
            return CharacterTextThreadSendResult(
                save_id=save_id,
                thread=self.get_thread_model(save_id=save_id, thread_id=thread.id),
                player_message=_message_models(
                    self.repositories,
                    save_id=save_id,
                    messages=(player_message,),
                )[0],
                replies=(),
            )
        settings = chat_generation_settings(
            self.repositories,
            provider=preference.provider,
            model_id=preference.model_id,
            save_id=save_id,
        )
        task = roleplay_model_task(roleplay_type=details.scenario.type, purpose="chat")
        replies: list[CharacterTextMessageRecord] = []
        for assessment in responders:
            character = assessment.character
            identity = _character_text_identity(
                save_id=save_id,
                character=character,
                repositories=self.repositories,
            )
            thread_memory = refresh_character_text_thread_memory(
                repositories=self.repositories,
                save_id=save_id,
                thread_id=thread.id,
            )
            history = canonical_character_text_context_messages(
                repositories=self.repositories,
                save_id=save_id,
                thread_id=thread.id,
                include_message_ids=(player_message.id,),
            )[-_MAX_THREAD_CONTEXT_MESSAGES:]
            request = ChatRequest(
                provider=preference.provider,
                model_id=preference.model_id,
                messages=_group_history_chat_messages(
                    save_id=save_id,
                    repositories=self.repositories,
                    history=tuple(history),
                ),
                response_style_section=CHARACTER_TEXT_RESPONSE_STYLE_SECTION,
                scenario_instructions="\n\n".join(
                    part
                    for part in (
                        compact_scenario_instructions(details.scenario),
                        f"Write one in-world group text from {character.name}.",
                        "Do not narrate actions, quote JSON, or include sender labels.",
                    )
                    if part
                ),
                custom_instructions=details.save.custom_instructions,
                turn_directive=(
                    f"Reply as {character.name} in this group phone thread. "
                    f"Intent: {assessment.response_intent or assessment.reason}. "
                    "Use natural prose only."
                ),
                turn_directive_purpose=CHAT_TURN_DIRECTIVE_PURPOSE_CHARACTER_TEXT,
                phone_context=tuple(
                    [
                        *_character_text_identity_context_lines(
                            identity=identity,
                            group_participants=participants,
                            target_character_id=character.id,
                        ),
                        *_phone_context_lines(
                            save_id=save_id,
                            character=character,
                            repositories=self.repositories,
                        ),
                        _group_participant_context_line(participants),
                        *_thread_memory_context_lines(thread_memory),
                    ]
                ),
                current_scene_recap=tuple(
                    _context_lines(
                        save_id=save_id,
                        character=character,
                        repositories=self.repositories,
                    )
                ),
                retrieved_recent_messages=_character_chronicle_context_lines(
                    save_id=save_id,
                    character=character,
                    repositories=self.repositories,
                ),
                temperature=settings.temperature,
                max_output_tokens=settings.max_output_tokens,
                retry_progress_callback=_text_delivery_retry_callback(
                    repositories=self.repositories,
                    save_id=save_id,
                    message_id=player_message.id,
                    callback=retry_progress_callback,
                ),
            )
            request = chat_request_with_reasoning_override(
                self.repositories,
                request_with_openrouter_routing(
                    self.repositories,
                    request,
                    task=task,
                    save_id=save_id,
                ),
                task=task,
                save_id=save_id,
            )
            try:
                generation = await self._generate_character_text_with_script_guard(
                    save_id=save_id,
                    message_id=player_message.id,
                    request=request,
                    chat=chat,
                    identity=identity,
                )
                request = generation.request
                response = generation.response
                reply_body = generation.body
            except Exception as exc:  # noqa: BLE001 - group replies are optional
                log_error_event(
                    "character_text.group_reply_failed",
                    save_id=save_id,
                    thread_id=thread.id,
                    character_id=character.id,
                    text_message_id=player_message.id,
                    **exception_log_fields(exc),
                )
                break
            reply = self.repositories.append_character_text_message(
                save_id=save_id,
                thread_id=thread.id,
                character_id=character.id,
                sender="character",
                sender_character_id=character.id,
                body="",
                delivery_status="pending",
                reply_to_message_id=player_message.id,
            )
            reply_sent_at = _current_character_text_in_world_timestamp(
                repositories=self.repositories,
                save_id=save_id,
            )
            reply_for_effects = _completed_character_text_message(
                reply,
                body=reply_body,
                provider=response.provider,
                model=response.model_id,
                token_estimate=response.token_usage.get("total"),
                in_world_sent_at=reply_sent_at,
            )
            await self._generate_attachment_for_character_message(
                save_id=save_id,
                message=reply_for_effects,
                character=character,
                history=tuple(history),
                current_user_id=current_user_id,
            )
            reply = self.repositories.complete_character_text_message_delivery(
                save_id=save_id,
                message_id=reply.id,
                body=reply_body,
                provider=response.provider,
                model=response.model_id,
                token_estimate=response.token_usage.get("total"),
                in_world_sent_at=reply_sent_at,
            )
            _capture_character_text_prompt(
                prompt_inspection_store=self.prompt_inspection_store,
                message_id=reply.id,
                request=request,
                response=response,
            )
            _grant_player_has_character_number_from_inbound_text(
                repositories=self.repositories,
                save_id=save_id,
                character_id=character.id,
                source_text_message_id=reply.id,
            )
            replies.append(reply)
        if not replies:
            player_message = self.repositories.update_character_text_delivery(
                save_id=save_id,
                message_id=player_message.id,
                status="sent",
                error=None,
            )
            refresh_character_text_thread_memory(
                repositories=self.repositories,
                save_id=save_id,
                thread_id=thread.id,
            )
            return CharacterTextThreadSendResult(
                save_id=save_id,
                thread=self.get_thread_model(save_id=save_id, thread_id=thread.id),
                player_message=_message_models(
                    self.repositories,
                    save_id=save_id,
                    messages=(player_message,),
                )[0],
                replies=(),
            )
        player_message = self.repositories.update_character_text_delivery(
            save_id=save_id,
            message_id=player_message.id,
            status="sent",
            error=None,
        )
        world_update = await CharacterTextWorldUpdateService(
            repositories=self.repositories,
            providers=self.providers,
        ).update_after_text_messages(
            save_id=save_id,
            text_messages=(player_message, *replies),
        )
        refresh_character_text_thread_memory(
            repositories=self.repositories,
            save_id=save_id,
            thread_id=thread.id,
        )
        updated_thread = self.repositories.get_character_text_thread(
            thread_id=thread.id,
            save_id=save_id,
        )
        if updated_thread is None:
            raise ValueError("Character text thread disappeared during send")
        player_message_model, *reply_models = _message_models(
            self.repositories,
            save_id=save_id,
            messages=(player_message, *replies),
        )
        return CharacterTextThreadSendResult(
            save_id=save_id,
            thread=_thread_model(
                updated_thread,
                participants=_thread_participant_models(
                    self.repositories,
                    save_id=save_id,
                    thread_id=updated_thread.id,
                ),
                messages=_message_models(
                    self.repositories,
                    save_id=save_id,
                    messages=tuple(
                        self.repositories.list_character_text_messages(
                            save_id=save_id,
                            thread_id=updated_thread.id,
                        )
                    ),
                ),
            ),
            player_message=player_message_model,
            replies=tuple(reply_models),
            world_update=world_update,
        )

    async def complete_queued_text_send(
        self,
        *,
        save_id: str,
        player_message_id: str,
        uploaded_photo_bytes: bytes | None = None,
        uploaded_photo_filename: str | None = None,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        current_user_id: str | None = None,
    ) -> CharacterTextSendResult:
        player_message = self.repositories.get_character_text_message(
            save_id=save_id,
            message_id=player_message_id,
        )
        if player_message is None or player_message.sender != "player":
            raise ValueError(f"Unknown queued character text id: {player_message_id}")
        if player_message.delivery_status not in {"pending", "retrying"}:
            raise ValueError("Character text send is not pending")
        if player_message.character_id is None:
            raise ValueError("Queued direct character text is missing a character")
        character = self.repositories.get_character(player_message.character_id)
        if character is None or character.save_id != save_id:
            self.repositories.update_character_text_delivery(
                save_id=save_id,
                message_id=player_message.id,
                status="failed",
                error="Text character is no longer available",
            )
            raise ValueError("Text character is no longer available")
        details = self.repositories.load_save_details(save_id)
        if details is None:
            self.repositories.update_character_text_delivery(
                save_id=save_id,
                message_id=player_message.id,
                status="failed",
                error=f"Unknown save id: {save_id}",
            )
            raise ValueError(f"Unknown save id: {save_id}")
        if uploaded_photo_bytes is not None:
            try:
                await self._attach_uploaded_player_photo(
                    save_id=save_id,
                    player_message=player_message,
                    uploaded_photo_bytes=uploaded_photo_bytes,
                    uploaded_photo_filename=uploaded_photo_filename,
                )
            except Exception as exc:
                self.repositories.update_character_text_delivery(
                    save_id=save_id,
                    message_id=player_message.id,
                    status="failed",
                    error=str(exc) or exc.__class__.__name__,
                )
                raise
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="chat",
        )
        if preference is None:
            self.repositories.update_character_text_delivery(
                save_id=save_id,
                message_id=player_message.id,
                status="failed",
                error="No chat model configured",
            )
            raise ValueError("No chat model configured")
        provider = self.providers.get(preference.provider)
        chat = getattr(provider, "chat", None)
        if not callable(chat):
            self.repositories.update_character_text_delivery(
                save_id=save_id,
                message_id=player_message.id,
                status="failed",
                error="Configured chat provider is unavailable",
            )
            raise ValueError("Configured chat provider is unavailable")
        thread_memory = refresh_character_text_thread_memory(
            repositories=self.repositories,
            save_id=save_id,
            thread_id=player_message.thread_id,
        )
        history = canonical_character_text_context_messages(
            repositories=self.repositories,
            save_id=save_id,
            thread_id=player_message.thread_id,
            include_message_ids=(player_message.id,),
        )[-_MAX_THREAD_CONTEXT_MESSAGES:]
        settings = chat_generation_settings(
            self.repositories,
            provider=preference.provider,
            model_id=preference.model_id,
            save_id=save_id,
        )
        task = roleplay_model_task(roleplay_type=details.scenario.type, purpose="chat")
        uploaded_photo_descriptions = _uploaded_photo_descriptions_for_messages(
            repositories=self.repositories,
            save_id=save_id,
            messages=tuple(history),
        )
        identity = _character_text_identity(
            save_id=save_id,
            character=character,
            repositories=self.repositories,
        )
        history_messages = tuple(
            _history_chat_message(
                message,
                player_name=_player_name(save_id, self.repositories),
                character_name=character.name,
                uploaded_photo_descriptions=uploaded_photo_descriptions.get(
                    message.id,
                    (),
                ),
            )
            for message in history
        )
        request = ChatRequest(
            provider=preference.provider,
            model_id=preference.model_id,
            messages=history_messages,
            response_style_section=CHARACTER_TEXT_RESPONSE_STYLE_SECTION,
            scenario_instructions="\n\n".join(
                part
                for part in (
                    compact_scenario_instructions(details.scenario),
                    f"Return one in-world text reply as {character.name}.",
                    "Do not narrate actions, quote JSON, or include sender labels.",
                )
                if part
            ),
            custom_instructions=details.save.custom_instructions,
            turn_directive=(
                f"Reply as {character.name} in a phone text conversation. "
                "Use natural prose only."
            ),
            turn_directive_purpose=CHAT_TURN_DIRECTIVE_PURPOSE_CHARACTER_TEXT,
            phone_context=tuple(
                [
                    *_character_text_identity_context_lines(identity=identity),
                    *_phone_context_lines(
                        save_id=save_id,
                        character=character,
                        repositories=self.repositories,
                    ),
                    *_thread_memory_context_lines(thread_memory),
                ]
            ),
            current_scene_recap=tuple(
                _context_lines(
                    save_id=save_id,
                    character=character,
                    repositories=self.repositories,
                )
            ),
            retrieved_recent_messages=_character_chronicle_context_lines(
                save_id=save_id,
                character=character,
                repositories=self.repositories,
            ),
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
            retry_progress_callback=_text_delivery_retry_callback(
                repositories=self.repositories,
                save_id=save_id,
                message_id=player_message.id,
                callback=retry_progress_callback,
            ),
        )
        request = chat_request_with_reasoning_override(
            self.repositories,
            request_with_openrouter_routing(
                self.repositories,
                request,
                task=task,
                save_id=save_id,
            ),
            task=task,
            save_id=save_id,
        )
        try:
            generation = await self._generate_character_text_with_script_guard(
                save_id=save_id,
                message_id=player_message.id,
                request=request,
                chat=chat,
                identity=identity,
            )
            request = generation.request
            response = generation.response
            reply_body = generation.body
        except Exception as exc:
            self.repositories.update_character_text_delivery(
                save_id=save_id,
                message_id=player_message.id,
                status="failed",
                error=str(exc) or exc.__class__.__name__,
            )
            raise
        reply = self.repositories.append_character_text_message(
            save_id=save_id,
            thread_id=player_message.thread_id,
            character_id=character.id,
            sender="character",
            body="",
            delivery_status="pending",
            reply_to_message_id=player_message.id,
        )
        reply_sent_at = _current_character_text_in_world_timestamp(
            repositories=self.repositories,
            save_id=save_id,
        )
        reply_for_effects = _completed_character_text_message(
            reply,
            body=reply_body,
            provider=response.provider,
            model=response.model_id,
            token_estimate=response.token_usage.get("total"),
            in_world_sent_at=reply_sent_at,
        )
        updated_thread = self.repositories.get_character_text_thread(
            thread_id=player_message.thread_id,
            save_id=save_id,
        )
        if updated_thread is None:
            raise ValueError("Character text thread disappeared during send")
        await self._generate_attachment_for_character_message(
            save_id=save_id,
            message=reply_for_effects,
            character=character,
            history=tuple(history),
            current_user_id=current_user_id,
        )
        player_message = self.repositories.update_character_text_delivery(
            save_id=save_id,
            message_id=player_message.id,
            status="sent",
            error=None,
        )
        reply = self.repositories.complete_character_text_message_delivery(
            save_id=save_id,
            message_id=reply.id,
            body=reply_body,
            provider=response.provider,
            model=response.model_id,
            token_estimate=response.token_usage.get("total"),
            in_world_sent_at=reply_sent_at,
        )
        _capture_character_text_prompt(
            prompt_inspection_store=self.prompt_inspection_store,
            message_id=reply.id,
            request=request,
            response=response,
        )
        player = _player_character(save_id, self.repositories)
        if player is not None:
            self.repositories.upsert_character_contact_state(
                save_id=save_id,
                player_character_id=player.id,
                character_id=character.id,
                character_has_player_number=True,
                source_text_message_id=player_message.id,
            )
        _update_text_route(
            repositories=self.repositories,
            save_id=save_id,
            character=character,
            player_message=player_message,
            reply=reply,
        )
        world_update = await CharacterTextWorldUpdateService(
            repositories=self.repositories,
            providers=self.providers,
        ).update_after_text_exchange(
            save_id=save_id,
            player_message=player_message,
            reply=reply,
        )
        refresh_character_text_thread_memory(
            repositories=self.repositories,
            save_id=save_id,
            thread_id=updated_thread.id,
        )
        player_message_model, reply_model = _message_models(
            self.repositories,
            save_id=save_id,
            messages=(player_message, reply),
        )
        return CharacterTextSendResult(
            save_id=save_id,
            thread=_thread_model(
                updated_thread,
                messages=_message_models(
                    self.repositories,
                    save_id=save_id,
                    messages=tuple(
                        self.repositories.list_character_text_messages(
                            save_id=save_id,
                            thread_id=updated_thread.id,
                        )
                    ),
                ),
            ),
            player_message=player_message_model,
            reply=reply_model,
            world_update=world_update,
        )

    async def _generate_character_text_with_script_guard(
        self,
        *,
        save_id: str,
        message_id: str,
        request: ChatRequest,
        chat: Callable[[ChatRequest], Awaitable[ChatResponse]],
        identity: _CharacterTextIdentity,
    ) -> _CharacterTextGenerationResult:
        current_request = request
        phrase_denylist = effective_generated_phrase_denylist(
            self.repositories,
            save_id=save_id,
        )
        script_violations: tuple[ScriptPolicyViolation, ...] = ()
        phrase_violations: tuple[PhraseDenylistViolation, ...] = ()
        identity_violation: _CharacterTextIdentityViolation | None = None
        response: ChatResponse | None = None
        body = ""
        for attempt in range(1, GENERATED_PHRASE_GUARD_MAX_ATTEMPTS + 1):
            response = await chat(current_request)
            body = _validated_character_text_response_body(response.body)
            script_violations = _character_text_script_violations(
                repositories=self.repositories,
                save_id=save_id,
                request=current_request,
                body=body,
            )
            phrase_violations = denied_phrase_violations(
                body,
                phrases=phrase_denylist,
                field_name="character_text",
            )
            identity_violation = _character_text_identity_violation(
                body=body,
                identity=identity,
            )
            if (
                not script_violations
                and not phrase_violations
                and identity_violation is None
            ):
                return _CharacterTextGenerationResult(
                    request=current_request,
                    response=response,
                    body=body,
                )
            _log_character_text_guard_violations(
                save_id=save_id,
                message_id=message_id,
                response=response,
                script_violations=script_violations,
                phrase_violations=phrase_violations,
                identity_violation=identity_violation,
                identity=identity,
                retry=attempt > 1,
            )
            max_attempts = (
                GENERATED_PHRASE_GUARD_MAX_ATTEMPTS if phrase_violations else 2
            )
            if attempt >= max_attempts:
                break
            self.repositories.update_character_text_delivery(
                save_id=save_id,
                message_id=message_id,
                status="retrying",
                error=None,
                attempt=attempt + 1,
            )
            current_request = replace(
                request,
                regeneration_feedback=_combine_regeneration_feedback(
                    request.regeneration_feedback,
                    _character_text_guard_retry_feedback(
                        script_violations=script_violations,
                        phrase_violations=phrase_violations,
                        identity_violation=identity_violation,
                        identity=identity,
                    ),
                ),
            )
        if identity_violation is not None:
            raise ValueError(_character_text_identity_error(identity))
        if script_violations:
            raise ValueError(
                summarize_script_policy_violations(script_violations)
            )
        if phrase_violations:
            raise ValueError(
                summarize_phrase_policy_violations(phrase_violations)
            )
        raise ValueError("Character text generation failed validation.")

    def mark_text_send_job(
        self,
        *,
        save_id: str,
        player_message_id: str,
        job_id: str,
    ) -> CharacterTextMessage:
        return _message_model(
            self.repositories.update_character_text_delivery(
                save_id=save_id,
                message_id=player_message_id,
                status="pending",
                error=None,
                job_id=job_id,
            )
        )

    def mark_spontaneous_text_job(
        self,
        *,
        save_id: str,
        text_message_id: str,
        job_id: str,
    ) -> CharacterTextMessage:
        return _message_model(
            self.repositories.update_character_text_delivery(
                save_id=save_id,
                message_id=text_message_id,
                status="pending",
                error=None,
                job_id=job_id,
            )
        )

    async def send_spontaneous_text(
        self,
        *,
        save_id: str,
        character_id: str,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        current_user_id: str | None = None,
    ) -> CharacterTextSpontaneousResult:
        queued = self.queue_spontaneous_text(
            save_id=save_id,
            character_id=character_id,
        )
        return await self.complete_queued_spontaneous_text(
            save_id=save_id,
            text_message_id=queued.message.id,
            retry_progress_callback=retry_progress_callback,
            current_user_id=current_user_id,
        )

    async def complete_queued_spontaneous_text(
        self,
        *,
        save_id: str,
        text_message_id: str,
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        current_user_id: str | None = None,
    ) -> CharacterTextSpontaneousResult:
        queued_message = self.repositories.get_character_text_message(
            save_id=save_id,
            message_id=text_message_id,
        )
        if queued_message is None or queued_message.sender != "character":
            raise ValueError(f"Unknown queued character text id: {text_message_id}")
        if queued_message.delivery_status not in {"pending", "retrying"}:
            raise ValueError("Character text send is not pending")
        if queued_message.character_id is None:
            raise ValueError("Queued character text is missing a character")
        character = self.repositories.get_character(queued_message.character_id)
        if character is None or character.save_id != save_id:
            self.repositories.update_character_text_delivery(
                save_id=save_id,
                message_id=queued_message.id,
                status="failed",
                error="Text character is no longer available",
            )
            raise ValueError("Text character is no longer available")
        details = self.repositories.load_save_details(save_id)
        if details is None:
            self.repositories.update_character_text_delivery(
                save_id=save_id,
                message_id=queued_message.id,
                status="failed",
                error=f"Unknown save id: {save_id}",
            )
            raise ValueError(f"Unknown save id: {save_id}")
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="chat",
        )
        if preference is None:
            self.repositories.update_character_text_delivery(
                save_id=save_id,
                message_id=queued_message.id,
                status="failed",
                error="No chat model configured",
            )
            raise ValueError("No chat model configured")
        provider = self.providers.get(preference.provider)
        chat = getattr(provider, "chat", None)
        if not callable(chat):
            self.repositories.update_character_text_delivery(
                save_id=save_id,
                message_id=queued_message.id,
                status="failed",
                error="Configured chat provider is unavailable",
            )
            raise ValueError("Configured chat provider is unavailable")
        thread_memory = refresh_character_text_thread_memory(
            repositories=self.repositories,
            save_id=save_id,
            thread_id=queued_message.thread_id,
        )
        history = canonical_character_text_context_messages(
            repositories=self.repositories,
            save_id=save_id,
            thread_id=queued_message.thread_id,
        )[-_MAX_THREAD_CONTEXT_MESSAGES:]
        settings = chat_generation_settings(
            self.repositories,
            provider=preference.provider,
            model_id=preference.model_id,
            save_id=save_id,
        )
        task = roleplay_model_task(roleplay_type=details.scenario.type, purpose="chat")
        uploaded_photo_descriptions = _uploaded_photo_descriptions_for_messages(
            repositories=self.repositories,
            save_id=save_id,
            messages=tuple(history),
        )
        identity = _character_text_identity(
            save_id=save_id,
            character=character,
            repositories=self.repositories,
        )
        history_messages = tuple(
            _history_chat_message(
                message,
                player_name=_player_name(save_id, self.repositories),
                character_name=character.name,
                uploaded_photo_descriptions=uploaded_photo_descriptions.get(
                    message.id,
                    (),
                ),
            )
            for message in history
        )
        if not history_messages:
            history_messages = (
                ChatMessage(
                    role="player",
                    body=(
                        "The phone thread is quiet. Choose a natural in-world "
                        "reason to text now."
                    ),
                    speaker_name=_player_name(save_id, self.repositories),
                ),
            )
        request = ChatRequest(
            provider=preference.provider,
            model_id=preference.model_id,
            messages=history_messages,
            response_style_section=CHARACTER_TEXT_RESPONSE_STYLE_SECTION,
            scenario_instructions="\n\n".join(
                part
                for part in (
                    compact_scenario_instructions(details.scenario),
                    f"Write one in-world phone text from {character.name}.",
                    (
                        "Use natural prose only; do not include sender labels "
                        "or narration."
                    ),
                )
                if part
            ),
            custom_instructions=details.save.custom_instructions,
            turn_directive=(
                f"Send a concise spontaneous phone text as {character.name}. "
                "Choose a topic grounded in current context and recent phone "
                "thread history."
            ),
            turn_directive_purpose=CHAT_TURN_DIRECTIVE_PURPOSE_CHARACTER_TEXT,
            phone_context=tuple(
                [
                    *_character_text_identity_context_lines(identity=identity),
                    *_phone_context_lines(
                        save_id=save_id,
                        character=character,
                        repositories=self.repositories,
                    ),
                    *_thread_memory_context_lines(thread_memory),
                ]
            ),
            current_scene_recap=tuple(
                _context_lines(
                    save_id=save_id,
                    character=character,
                    repositories=self.repositories,
                )
            ),
            retrieved_recent_messages=_character_chronicle_context_lines(
                save_id=save_id,
                character=character,
                repositories=self.repositories,
            ),
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
            retry_progress_callback=_text_delivery_retry_callback(
                repositories=self.repositories,
                save_id=save_id,
                message_id=queued_message.id,
                callback=retry_progress_callback,
            ),
        )
        request = chat_request_with_reasoning_override(
            self.repositories,
            request_with_openrouter_routing(
                self.repositories,
                request,
                task=task,
                save_id=save_id,
            ),
            task=task,
            save_id=save_id,
        )
        try:
            generation = await self._generate_character_text_with_script_guard(
                save_id=save_id,
                message_id=queued_message.id,
                request=request,
                chat=chat,
                identity=identity,
            )
            request = generation.request
            response = generation.response
            response_body = generation.body
        except Exception as exc:
            self.repositories.update_character_text_delivery(
                save_id=save_id,
                message_id=queued_message.id,
                status="failed",
                error=str(exc) or exc.__class__.__name__,
            )
            raise
        message_sent_at = _current_character_text_in_world_timestamp(
            repositories=self.repositories,
            save_id=save_id,
        )
        message_for_effects = _completed_character_text_message(
            queued_message,
            body=response_body,
            provider=response.provider,
            model=response.model_id,
            token_estimate=response.token_usage.get("total"),
            in_world_sent_at=message_sent_at,
        )
        updated_thread = self.repositories.get_character_text_thread(
            thread_id=queued_message.thread_id,
            save_id=save_id,
        )
        if updated_thread is None:
            raise ValueError("Character text thread disappeared during send")
        await self._generate_attachment_for_character_message(
            save_id=save_id,
            message=message_for_effects,
            character=character,
            history=tuple(history),
            current_user_id=current_user_id,
        )
        message = self.repositories.complete_character_text_message_delivery(
            save_id=save_id,
            message_id=queued_message.id,
            body=response_body,
            provider=response.provider,
            model=response.model_id,
            token_estimate=response.token_usage.get("total"),
            in_world_sent_at=message_sent_at,
        )
        _capture_character_text_prompt(
            prompt_inspection_store=self.prompt_inspection_store,
            message_id=message.id,
            request=request,
            response=response,
        )
        _grant_player_has_character_number_from_inbound_text(
            repositories=self.repositories,
            save_id=save_id,
            character_id=character.id,
            source_text_message_id=message.id,
        )
        world_update = await CharacterTextWorldUpdateService(
            repositories=self.repositories,
            providers=self.providers,
        ).update_after_text_messages(
            save_id=save_id,
            text_messages=(message,),
        )
        refresh_character_text_thread_memory(
            repositories=self.repositories,
            save_id=save_id,
            thread_id=updated_thread.id,
        )
        message_model = _message_models(
            self.repositories,
            save_id=save_id,
            messages=(message,),
        )[0]
        return CharacterTextSpontaneousResult(
            save_id=save_id,
            thread=_thread_model(
                updated_thread,
                messages=_message_models(
                    self.repositories,
                    save_id=save_id,
                    messages=tuple(
                        self.repositories.list_character_text_messages(
                            save_id=save_id,
                            thread_id=updated_thread.id,
                        )
                    ),
                ),
            ),
            message=message_model,
            world_update=world_update,
        )

    async def _attach_uploaded_player_photo(
        self,
        *,
        save_id: str,
        player_message: CharacterTextMessageRecord,
        uploaded_photo_bytes: bytes,
        uploaded_photo_filename: str | None,
    ) -> CharacterTextMessageAttachmentRecord:
        if self.media_service is None:
            raise ValueError("Text photo upload is unavailable")
        if player_message.sender_character_id is None:
            raise ValueError("Queued character text is missing a sender character")
        upload = await self.media_service.upload_character_text_player_photo(
            save_id=save_id,
            text_message=player_message,
            sender_character_id=player_message.sender_character_id,
            image_bytes=uploaded_photo_bytes,
            filename=uploaded_photo_filename,
        )
        try:
            attachment = self.repositories.add_character_text_message_attachment(
                save_id=save_id,
                thread_id=player_message.thread_id,
                text_message_id=player_message.id,
                character_id=player_message.sender_character_id,
                kind="uploaded_photo",
                status="succeeded",
                media_asset_id=upload.asset.id,
                prompt=upload.description,
                metadata={
                    "source": "uploaded",
                    "media_asset_id": upload.asset.id,
                    "description": upload.description,
                },
            )
        except Exception:
            self.media_service.cleanup_character_text_uploaded_photo(
                save_id=save_id,
                asset=upload.asset,
            )
            raise
        log_event(
            "character_text.uploaded_photo_attached",
            save_id=save_id,
            thread_id=player_message.thread_id,
            text_message_id=player_message.id,
            media_asset_id=upload.asset.id,
        )
        return attachment

    async def _generate_attachment_for_character_message(
        self,
        *,
        save_id: str,
        message: CharacterTextMessageRecord,
        character: CharacterRecord,
        history: tuple[CharacterTextMessageRecord, ...],
        current_user_id: str | None = None,
    ) -> tuple[CharacterTextMessageAttachmentRecord, ...]:
        if self.media_service is None or message.sender != "character":
            return ()
        media_service = self.media_service
        decision = await self._attachment_decision(
            save_id=save_id,
            message=message,
            character=character,
            history=history,
        )
        if decision is None or decision.kind == "none":
            return ()
        scene_context = _text_attachment_scene_context(
            save_id=save_id,
            character=character,
            message=message,
            history=history,
            repositories=self.repositories,
        )
        visual_prompt = _attachment_visual_prompt(
            decision,
            character=character,
        )
        try:
            if decision.kind == "character_image":
                asset = await media_service.generate_character_text_character_image(
                    save_id=save_id,
                    text_message=message,
                    character=character,
                    visual_prompt=visual_prompt,
                    scene_context=scene_context,
                    current_user_id=current_user_id,
                )
            else:
                asset = (
                    await media_service.generate_character_text_object_context_image(
                        save_id=save_id,
                        text_message=message,
                        character=character,
                        visual_prompt=visual_prompt,
                        scene_context=scene_context,
                        current_user_id=current_user_id,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - text replies must survive media failures
            log_error_event(
                "character_text.attachment_generation_failed",
                save_id=save_id,
                text_message_id=message.id,
                character_id=character.id,
                kind=decision.kind,
                **exception_log_fields(exc),
            )
            attachment = self.repositories.add_character_text_message_attachment(
                save_id=save_id,
                thread_id=message.thread_id,
                text_message_id=message.id,
                character_id=character.id,
                kind=decision.kind,
                status="failed",
                prompt=decision.visual_prompt,
                error=str(exc) or exc.__class__.__name__,
                metadata={"decision_reason": decision.reason},
            )
            return (attachment,)
        attachment = self.repositories.add_character_text_message_attachment(
            save_id=save_id,
            thread_id=message.thread_id,
            text_message_id=message.id,
            character_id=character.id,
            kind=decision.kind,
            status="succeeded",
            media_asset_id=asset.id,
            prompt=decision.visual_prompt,
            metadata={
                "decision_reason": decision.reason,
                "media_asset_id": asset.id,
            },
        )
        log_event(
            "character_text.attachment_generated",
            save_id=save_id,
            text_message_id=message.id,
            character_id=character.id,
            kind=decision.kind,
            media_asset_id=asset.id,
        )
        return (attachment,)

    async def _attachment_decision(
        self,
        *,
        save_id: str,
        message: CharacterTextMessageRecord,
        character: CharacterRecord,
        history: tuple[CharacterTextMessageRecord, ...],
    ) -> _AttachmentDecision | None:
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose=CHARACTER_TEXT_ATTACHMENT_DECISION_TASK,
        )
        if preference is None:
            return None
        provider = self.providers.get(preference.provider)
        if not isinstance(provider, StructuredOutputProvider):
            return None
        if known_model_is_unavailable(
            self.repositories,
            provider=preference.provider,
            model_id=preference.model_id,
        ):
            return None
        if not model_supports_any_capability(
            self.repositories,
            provider=preference.provider,
            model_id=preference.model_id,
            required=STRUCTURED_OUTPUT_CAPABILITIES,
        ):
            return None
        request = StructuredOutputRequest(
            provider=preference.provider,
            model_id=preference.model_id,
            schema_name="character_text_image_attachment_decision",
            schema=_attachment_decision_schema(),
            messages=_attachment_decision_messages(
                save_id=save_id,
                repositories=self.repositories,
                character=character,
                message=message,
                history=history,
            ),
            temperature=0.1,
            max_output_tokens=400,
        )
        try:
            response = await structured_output_with_fallback(
                repositories=self.repositories,
                providers=cast(dict[str, ProviderClient], self.providers),
                request=request,
                task=CHARACTER_TEXT_ATTACHMENT_DECISION_TASK,
                save_id=save_id,
                diagnostic_context={
                    "text_message_id": message.id,
                    "character_id": character.id,
                },
            )
        except Exception as exc:  # noqa: BLE001 - attachment choice is auxiliary
            log_error_event(
                "character_text.attachment_decision_failed",
                save_id=save_id,
                text_message_id=message.id,
                character_id=character.id,
                **exception_log_fields(exc),
            )
            return None
        return _attachment_decision_from_data(response.data)

    async def _group_response_assessments(
        self,
        *,
        save_id: str,
        thread: CharacterTextThreadRecord,
        player_message: CharacterTextMessageRecord,
        participants: tuple[CharacterRecord, ...],
    ) -> tuple[_GroupResponseAssessment, ...]:
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose=CHARACTER_TEXT_GROUP_RESPONSE_ASSESSMENT_TASK,
        )
        if preference is None:
            return ()
        provider = self.providers.get(preference.provider)
        if not isinstance(provider, StructuredOutputProvider):
            return ()
        if known_model_is_unavailable(
            self.repositories,
            provider=preference.provider,
            model_id=preference.model_id,
        ):
            return ()
        if not model_supports_any_capability(
            self.repositories,
            provider=preference.provider,
            model_id=preference.model_id,
            required=STRUCTURED_OUTPUT_CAPABILITIES,
        ):
            return ()
        assessments: list[_GroupResponseAssessment] = []
        for character in participants:
            request = StructuredOutputRequest(
                provider=preference.provider,
                model_id=preference.model_id,
                schema_name=CHARACTER_TEXT_GROUP_RESPONSE_ASSESSMENT_SCHEMA_NAME,
                schema=_group_response_assessment_schema(),
                messages=_group_response_assessment_messages(
                    save_id=save_id,
                    repositories=self.repositories,
                    thread=thread,
                    character=character,
                    player_message=player_message,
                    participants=participants,
                ),
                temperature=0.1,
                max_output_tokens=400,
            )
            try:
                response = await structured_output_with_fallback(
                    repositories=self.repositories,
                    providers=cast(dict[str, ProviderClient], self.providers),
                    request=request,
                    task=CHARACTER_TEXT_GROUP_RESPONSE_ASSESSMENT_TASK,
                    save_id=save_id,
                    diagnostic_context={
                        "thread_id": thread.id,
                        "character_id": character.id,
                        "text_message_id": player_message.id,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - group replies are optional
                log_error_event(
                    "character_text.group_response_assessment_failed",
                    save_id=save_id,
                    thread_id=thread.id,
                    character_id=character.id,
                    text_message_id=player_message.id,
                    **exception_log_fields(exc),
                )
                return ()
            assessment = _group_response_assessment_from_data(
                response.data,
                character=character,
            )
            if assessment is not None:
                assessments.append(assessment)
        return tuple(assessments)

    async def send_proactive_text_after_turn(
        self,
        *,
        save_id: str,
        source_message_ids: tuple[str, ...] = (),
        retry_progress_callback: ProviderRetryProgressCallback | None = None,
        current_user_id: str | None = None,
    ) -> CharacterTextProactiveResult:
        if not self.is_enabled(save_id):
            return CharacterTextProactiveResult(
                save_id=save_id,
                status="skipped",
                reason="texts_disabled",
            )
        details = self.repositories.load_save_details(save_id)
        if details is None:
            return CharacterTextProactiveResult(
                save_id=save_id,
                status="skipped",
                reason="unknown_save",
            )
        source_messages = {
            message.id: message for message in details.messages
        }
        source_ids = source_message_ids or tuple(
            message.id
            for message in reversed(details.messages)
            if message.role == "narrator"
        )[:1]
        if any(
            is_fade_to_black_message(
                role=source_messages[source_id].role,
                body=source_messages[source_id].body,
                safety_transition=source_messages[source_id].safety_transition,
            )
            for source_id in source_ids
            if source_id in source_messages
        ):
            return CharacterTextProactiveResult(
                save_id=save_id,
                status="skipped",
                reason="safety_transition",
            )
        if (
            character_text_proactive_random_chance_percent(
                self.repositories,
                save_id=save_id,
            )
            <= 0
        ):
            return CharacterTextProactiveResult(
                save_id=save_id,
                status="skipped",
                reason="proactive_texts_disabled",
            )
        try:
            await DatingRouteProfileService(
                repositories=self.repositories,
                providers=cast(dict[str, ProviderClient], self.providers),
            ).ensure_profiles_for_save(
                save_id=save_id,
                source_message_id=source_message_ids[-1]
                if source_message_ids
                else None,
            )
        except Exception as exc:  # noqa: BLE001 - best-effort text context
            log_error_event(
                "character_text.dating_route_profile_failed",
                save_id=save_id,
                **exception_log_fields(exc),
            )
        candidates = _proactive_text_candidates(
            save_id=save_id,
            repositories=self.repositories,
            fallback_source_message_ids=source_message_ids,
        )
        if not candidates:
            ambient_candidate = _ambient_random_proactive_text_candidate(
                save_id=save_id,
                repositories=self.repositories,
                source_message_ids=source_message_ids,
            )
            if ambient_candidate is None:
                return CharacterTextProactiveResult(
                    save_id=save_id,
                    status="skipped",
                    reason="no_candidate",
                )
            candidates = [ambient_candidate]
        pending_candidates = [
            candidate
            for candidate in candidates
            if self.repositories.get_character_text_proactive_trigger(
                save_id=save_id,
                character_id=candidate.character.id,
                trigger_key=candidate.trigger_key,
            )
            is None
        ]
        if not pending_candidates:
            return CharacterTextProactiveResult(
                save_id=save_id,
                status="skipped",
                reason="duplicate_suppressed",
                candidate_count=len(candidates),
            )
        gate = _select_proactive_text_candidate(
            repositories=self.repositories,
            save_id=save_id,
            candidates=pending_candidates,
        )
        if gate.candidate is None:
            return CharacterTextProactiveResult(
                save_id=save_id,
                status="skipped",
                reason=gate.reason,
                candidate_count=len(pending_candidates),
            )
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="chat",
        )
        if preference is None:
            return CharacterTextProactiveResult(
                save_id=save_id,
                status="skipped",
                reason="no_chat_model",
                candidate_count=len(pending_candidates),
            )
        provider = self.providers.get(preference.provider)
        chat = getattr(provider, "chat", None)
        if not callable(chat):
            return CharacterTextProactiveResult(
                save_id=save_id,
                status="skipped",
                reason="provider_unavailable",
                candidate_count=len(pending_candidates),
            )
        candidate = gate.candidate
        character = candidate.character
        thread = self.repositories.get_or_create_character_text_thread(
            save_id=save_id,
            character_id=character.id,
            title=character.name,
        )
        if self.repositories.has_active_character_text_delivery(
            save_id=save_id,
            thread_id=thread.id,
        ):
            updated_thread = self.repositories.get_character_text_thread(
                thread_id=thread.id,
                save_id=save_id,
            )
            return CharacterTextProactiveResult(
                save_id=save_id,
                status="skipped",
                reason="thread_busy",
                candidate_count=len(pending_candidates),
                trigger_key=candidate.trigger_key,
                thread=(
                    _thread_model(
                        updated_thread,
                        messages=_message_models(
                            self.repositories,
                            save_id=save_id,
                            messages=tuple(
                                self.repositories.list_character_text_messages(
                                    save_id=save_id,
                                    thread_id=updated_thread.id,
                                )
                            ),
                        ),
                    )
                    if updated_thread is not None
                    else None
                ),
            )
        pending_message = self.repositories.append_character_text_message(
            save_id=save_id,
            thread_id=thread.id,
            character_id=character.id,
            sender="character",
            body="",
            delivery_status="pending",
            in_world_sent_at=_current_character_text_in_world_timestamp(
                repositories=self.repositories,
                save_id=save_id,
            ),
        )
        thread_memory = refresh_character_text_thread_memory(
            repositories=self.repositories,
            save_id=save_id,
            thread_id=thread.id,
        )
        history = canonical_character_text_context_messages(
            repositories=self.repositories,
            save_id=save_id,
            thread_id=thread.id,
        )[-_MAX_THREAD_CONTEXT_MESSAGES:]
        settings = chat_generation_settings(
            self.repositories,
            provider=preference.provider,
            model_id=preference.model_id,
            save_id=save_id,
        )
        task = roleplay_model_task(roleplay_type=details.scenario.type, purpose="chat")
        uploaded_photo_descriptions = _uploaded_photo_descriptions_for_messages(
            repositories=self.repositories,
            save_id=save_id,
            messages=tuple(history),
        )
        identity = _character_text_identity(
            save_id=save_id,
            character=character,
            repositories=self.repositories,
        )
        history_messages = tuple(
            _history_chat_message(
                message,
                player_name=_player_name(save_id, self.repositories),
                character_name=character.name,
                uploaded_photo_descriptions=uploaded_photo_descriptions.get(
                    message.id,
                    (),
                ),
            )
            for message in history
        )
        if not history_messages:
            history_messages = (
                ChatMessage(
                    role="player",
                    body=f"Proactive text reason: {candidate.reason}",
                    speaker_name=_player_name(save_id, self.repositories),
                ),
            )
        request = ChatRequest(
            provider=preference.provider,
            model_id=preference.model_id,
            messages=history_messages,
            response_style_section=CHARACTER_TEXT_RESPONSE_STYLE_SECTION,
            scenario_instructions="\n\n".join(
                part
                for part in (
                    compact_scenario_instructions(details.scenario),
                    f"Write one in-world phone text from {character.name}.",
                    (
                        "Use natural prose only; do not include sender labels "
                        "or narration."
                    ),
                )
                if part
            ),
            custom_instructions=details.save.custom_instructions,
            turn_directive=(
                f"Send a concise proactive phone text as {character.name}. "
                f"Reason to text now: {candidate.reason}"
            ),
            turn_directive_purpose=CHAT_TURN_DIRECTIVE_PURPOSE_CHARACTER_TEXT,
            phone_context=tuple(
                [
                    *_character_text_identity_context_lines(identity=identity),
                    *_phone_context_lines(
                        save_id=save_id,
                        character=character,
                        repositories=self.repositories,
                    ),
                    *_thread_memory_context_lines(thread_memory),
                    *candidate.context_lines,
                ]
            ),
            current_scene_recap=tuple(
                _context_lines(
                    save_id=save_id,
                    character=character,
                    repositories=self.repositories,
                )
            ),
            retrieved_recent_messages=_character_chronicle_context_lines(
                save_id=save_id,
                character=character,
                repositories=self.repositories,
                source_message_ids=(
                    (candidate.source_message_id,)
                    if candidate.source_message_id is not None
                    else ()
                ),
            ),
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
            retry_progress_callback=_text_delivery_retry_callback(
                repositories=self.repositories,
                save_id=save_id,
                message_id=pending_message.id,
                callback=retry_progress_callback,
            ),
        )
        request = chat_request_with_reasoning_override(
            self.repositories,
            request_with_openrouter_routing(
                self.repositories,
                request,
                task=task,
                save_id=save_id,
            ),
            task=task,
            save_id=save_id,
        )
        try:
            generation = await self._generate_character_text_with_script_guard(
                save_id=save_id,
                message_id=pending_message.id,
                request=request,
                chat=chat,
                identity=identity,
            )
            request = generation.request
            response = generation.response
            response_body = generation.body
        except Exception as exc:
            self.repositories.update_character_text_delivery(
                save_id=save_id,
                message_id=pending_message.id,
                status="failed",
                error=str(exc) or exc.__class__.__name__,
            )
            raise
        recent_history = canonical_character_text_context_messages(
            repositories=self.repositories,
            save_id=save_id,
            thread_id=thread.id,
        )[-_MAX_THREAD_CONTEXT_MESSAGES:]
        if _is_duplicate_recent_character_text_body(
            body=response_body,
            character_id=character.id,
            history=recent_history,
        ):
            self.repositories.archive_character_text_messages_from(
                save_id=save_id,
                thread_id=thread.id,
                message_id=pending_message.id,
            )
            self.repositories.add_character_text_proactive_trigger(
                save_id=save_id,
                character_id=character.id,
                trigger_key=candidate.trigger_key,
                trigger_type=candidate.trigger_type,
                thread_id=thread.id,
                text_message_id=None,
                source_type=candidate.source_type,
                source_id=candidate.source_id,
                source_message_id=candidate.source_message_id,
                reason=candidate.reason,
            )
            updated_thread = self.repositories.get_character_text_thread(
                thread_id=thread.id,
                save_id=save_id,
            )
            if updated_thread is None:
                raise ValueError(
                    "Character text thread disappeared during duplicate suppression"
                )
            return CharacterTextProactiveResult(
                save_id=save_id,
                status="skipped",
                reason="duplicate_body_suppressed",
                candidate_count=len(pending_candidates),
                trigger_key=candidate.trigger_key,
                thread=_thread_model(
                    updated_thread,
                    messages=_message_models(
                        self.repositories,
                        save_id=save_id,
                        messages=tuple(
                            self.repositories.list_character_text_messages(
                                save_id=save_id,
                                thread_id=updated_thread.id,
                            )
                        ),
                    ),
                ),
            )
        message_sent_at = _current_character_text_in_world_timestamp(
            repositories=self.repositories,
            save_id=save_id,
        )
        message_for_effects = _completed_character_text_message(
            pending_message,
            body=response_body,
            provider=response.provider,
            model=response.model_id,
            token_estimate=response.token_usage.get("total"),
            in_world_sent_at=message_sent_at,
        )
        updated_thread = self.repositories.get_character_text_thread(
            thread_id=thread.id,
            save_id=save_id,
        )
        if updated_thread is None:
            raise ValueError("Character text thread disappeared during proactive send")
        await self._generate_attachment_for_character_message(
            save_id=save_id,
            message=message_for_effects,
            character=character,
            history=tuple(recent_history),
            current_user_id=current_user_id,
        )
        message = self.repositories.complete_character_text_message_delivery(
            save_id=save_id,
            message_id=pending_message.id,
            body=response_body,
            provider=response.provider,
            model=response.model_id,
            token_estimate=response.token_usage.get("total"),
            in_world_sent_at=message_sent_at,
        )
        _capture_character_text_prompt(
            prompt_inspection_store=self.prompt_inspection_store,
            message_id=message.id,
            request=request,
            response=response,
        )
        self.repositories.add_character_text_proactive_trigger(
            save_id=save_id,
            character_id=character.id,
            trigger_key=candidate.trigger_key,
            trigger_type=candidate.trigger_type,
            thread_id=thread.id,
            text_message_id=message.id,
            source_type=candidate.source_type,
            source_id=candidate.source_id,
            source_message_id=candidate.source_message_id,
            reason=candidate.reason,
        )
        _grant_player_has_character_number_from_inbound_text(
            repositories=self.repositories,
            save_id=save_id,
            character_id=character.id,
            source_text_message_id=message.id,
        )
        world_update = await CharacterTextWorldUpdateService(
            repositories=self.repositories,
            providers=self.providers,
        ).update_after_text_messages(
            save_id=save_id,
            text_messages=(message,),
        )
        refresh_character_text_thread_memory(
            repositories=self.repositories,
            save_id=save_id,
            thread_id=updated_thread.id,
        )
        message_model = _message_models(
            self.repositories,
            save_id=save_id,
            messages=(message,),
        )[0]
        return CharacterTextProactiveResult(
            save_id=save_id,
            status="sent",
            reason=candidate.reason,
            candidate_count=len(pending_candidates),
            trigger_key=candidate.trigger_key,
            thread=_thread_model(
                updated_thread,
                messages=_message_models(
                    self.repositories,
                    save_id=save_id,
                    messages=tuple(
                        self.repositories.list_character_text_messages(
                            save_id=save_id,
                            thread_id=updated_thread.id,
                        )
                    ),
                ),
            ),
            message=message_model,
            world_update=world_update,
        )

    def can_character_proactively_text(
        self,
        *,
        save_id: str,
        character_id: str,
    ) -> bool:
        if not self.is_enabled(save_id):
            return False
        return self.repositories.can_character_proactively_text(
            save_id=save_id,
            character_id=character_id,
        )


def _proactive_text_candidates(
    *,
    save_id: str,
    repositories: PersistenceRepositories,
    fallback_source_message_ids: tuple[str, ...],
) -> list[_ProactiveTextCandidate]:
    fallback_source_message_id = _proactive_fallback_source_message_id(
        repositories=repositories,
        save_id=save_id,
        fallback_source_message_ids=fallback_source_message_ids,
    )
    characters = _proactive_text_eligible_characters(
        save_id=save_id,
        repositories=repositories,
    )
    active_threads = [
        thread
        for thread in repositories.list_active_threads(save_id)
        if active_thread_is_prompt_visible(thread)
    ]
    routes = repositories.list_dating_route_states(save_id)
    candidates: list[_ProactiveTextCandidate] = []
    for character in characters:
        candidates.extend(
            _active_thread_candidates(
                character=character,
                active_threads=active_threads,
                fallback_source_message_id=fallback_source_message_id,
            )
        )
        route = next(
            (route for route in routes if route.npc_character_id == character.id),
            None,
        )
        if route is not None:
            candidate = _dating_route_candidate(
                character=character,
                route=route,
                fallback_source_message_id=fallback_source_message_id,
            )
            if candidate is not None:
                candidates.append(candidate)
        intent_candidate = _character_intent_candidate(
            character=character,
            fallback_source_message_id=fallback_source_message_id,
        )
        if intent_candidate is not None:
            candidates.append(intent_candidate)
    return sorted(
        candidates,
        key=lambda candidate: (
            -candidate.priority,
            candidate.character.name.casefold(),
            candidate.character.id,
            candidate.trigger_key,
        ),
    )


def _proactive_text_eligible_characters(
    *,
    save_id: str,
    repositories: PersistenceRepositories,
) -> list[CharacterRecord]:
    return [
        character
        for character in repositories.list_characters(save_id)
        if not character.is_player_character
        and repositories.can_character_proactively_text(
            save_id=save_id,
            character_id=character.id,
        )
    ]


def _proactive_fallback_source_message_id(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    fallback_source_message_ids: tuple[str, ...],
) -> str | None:
    if fallback_source_message_ids:
        return fallback_source_message_ids[-1]
    return _latest_narrator_message_id(repositories.list_messages(save_id))


def _latest_narrator_message_id(messages: list[MessageRecord]) -> str | None:
    for message in reversed(messages):
        if message.role == "narrator":
            return message.id
    return None


def _select_proactive_text_candidate(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    candidates: list[_ProactiveTextCandidate],
) -> _ProactiveCandidateGate:
    first_gate_reason = ""
    threads_by_character_id = {
        thread.character_id: thread
        for thread in repositories.list_character_text_threads(save_id)
        if thread.kind == "direct" and thread.character_id is not None
    }
    snapshot = repositories.get_scene_snapshot(save_id)
    cooldown_turns = character_text_proactive_random_cooldown_turns(
        repositories,
        save_id=save_id,
    )
    for candidate in candidates:
        gate_reason = _proactive_candidate_gate_reason(
            repositories=repositories,
            save_id=save_id,
            candidate=candidate,
            threads_by_character_id=threads_by_character_id,
            snapshot=snapshot,
            cooldown_turns=cooldown_turns,
        )
        if not gate_reason:
            return _ProactiveCandidateGate(candidate=candidate)
        if not first_gate_reason:
            first_gate_reason = gate_reason
    return _ProactiveCandidateGate(
        candidate=None,
        reason=first_gate_reason or "no_candidate",
    )


def _proactive_candidate_gate_reason(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    candidate: _ProactiveTextCandidate,
    threads_by_character_id: Mapping[str, CharacterTextThreadRecord],
    snapshot: SceneSnapshotRecord | None,
    cooldown_turns: int,
) -> str:
    thread = threads_by_character_id.get(candidate.character.id)
    if thread is not None and repositories.has_active_character_text_delivery(
        save_id=save_id,
        thread_id=thread.id,
    ):
        return "thread_busy"
    if _character_unavailable_for_proactive_text(candidate.character):
        return "character_unavailable"
    present_character_ids = (
        set(snapshot.present_character_ids) if snapshot is not None else set()
    )
    if (
        candidate.character.id in present_character_ids
        and not _proactive_candidate_allows_present_character(candidate)
    ):
        return "present_character_suppressed"
    if _proactive_candidate_cooldown_active(
        repositories=repositories,
        save_id=save_id,
        character_id=candidate.character.id,
        source_message_id=candidate.source_message_id,
        cooldown_turns=cooldown_turns,
    ):
        return "cooldown_active"
    return ""


def _proactive_candidate_allows_present_character(
    candidate: _ProactiveTextCandidate,
) -> bool:
    if candidate.trigger_type != "active_thread":
        return False
    thread_text = " ".join(
        line
        for line in candidate.context_lines
        if not line.casefold().startswith("proactive text trigger")
    )
    return any(
        token in _TEXT_NATIVE_THREAD_TERMS
        for token in _normalized_text_tokens(thread_text)
    )


def _character_unavailable_for_proactive_text(character: CharacterRecord) -> bool:
    availability_text = " ".join(
        _normalized_text_tokens(
            "\n".join(
                (
                    character.status,
                    character.known_state,
                )
            )
        )
    )
    return any(phrase in availability_text for phrase in _UNAVAILABLE_CHARACTER_PHRASES)


def _proactive_candidate_cooldown_active(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    character_id: str,
    source_message_id: str | None,
    cooldown_turns: int,
) -> bool:
    if cooldown_turns <= 0 or source_message_id is None:
        return False
    messages = repositories.list_messages(save_id)
    message_index = {message.id: index for index, message in enumerate(messages)}
    current_index = message_index.get(source_message_id)
    if current_index is None:
        return False
    last_trigger_index: int | None = None
    for trigger in repositories.list_character_text_proactive_triggers(save_id):
        if trigger.character_id != character_id:
            continue
        if trigger.text_message_id is None or trigger.source_message_id is None:
            continue
        trigger_index = message_index.get(trigger.source_message_id)
        if trigger_index is None or trigger_index > current_index:
            continue
        if last_trigger_index is None or trigger_index > last_trigger_index:
            last_trigger_index = trigger_index
    if last_trigger_index is None:
        return False
    elapsed_narrator_turns = sum(
        1
        for index, message in enumerate(messages)
        if last_trigger_index < index <= current_index
        if message.role == "narrator"
    )
    return elapsed_narrator_turns <= cooldown_turns


def _ambient_random_proactive_text_candidate(
    *,
    save_id: str,
    repositories: PersistenceRepositories,
    source_message_ids: tuple[str, ...],
) -> _ProactiveTextCandidate | None:
    source_message_id = source_message_ids[-1] if source_message_ids else None
    if source_message_id is None:
        return None
    chance_percent = character_text_proactive_random_chance_percent(
        repositories,
        save_id=save_id,
    )
    if chance_percent <= 0:
        return None
    cooldown_turns = character_text_proactive_random_cooldown_turns(
        repositories,
        save_id=save_id,
    )
    rng = Random(f"{save_id}:{source_message_id}:ambient_character_text")
    if rng.randrange(100) >= chance_percent:
        return None
    characters = _ambient_random_eligible_characters(
        save_id=save_id,
        repositories=repositories,
        source_message_id=source_message_id,
        cooldown_turns=cooldown_turns,
    )
    if not characters:
        return None
    character = rng.choice(characters)
    reason = "They thought of the player and decided to check in."
    return _ProactiveTextCandidate(
        character=character,
        trigger_key=f"ambient_random:{source_message_id}:{character.id}",
        trigger_type="ambient_random",
        source_type="message",
        source_id=source_message_id,
        source_message_id=source_message_id,
        reason=reason,
        context_lines=(
            "Proactive text trigger: ambient random check-in",
            _join_label("Reason", reason),
        ),
        priority=50,
    )


def _ambient_random_eligible_characters(
    *,
    save_id: str,
    repositories: PersistenceRepositories,
    source_message_id: str,
    cooldown_turns: int,
) -> list[CharacterRecord]:
    snapshot = repositories.get_scene_snapshot(save_id)
    present_character_ids = (
        set(snapshot.present_character_ids) if snapshot is not None else set()
    )
    threads_by_character_id = {
        thread.character_id: thread
        for thread in repositories.list_character_text_threads(save_id)
        if thread.kind == "direct" and thread.character_id is not None
    }
    characters: list[CharacterRecord] = []
    for character in _proactive_text_eligible_characters(
        save_id=save_id,
        repositories=repositories,
    ):
        if character.id in present_character_ids:
            continue
        thread = threads_by_character_id.get(character.id)
        if thread is not None and repositories.has_active_character_text_delivery(
            save_id=save_id,
            thread_id=thread.id,
        ):
            continue
        if _proactive_candidate_cooldown_active(
            repositories=repositories,
            save_id=save_id,
            character_id=character.id,
            source_message_id=source_message_id,
            cooldown_turns=cooldown_turns,
        ):
            continue
        characters.append(character)
    return sorted(
        characters,
        key=lambda character: (character.name.casefold(), character.id),
    )


def _active_thread_candidates(
    *,
    character: CharacterRecord,
    active_threads: list[ActiveThreadRecord],
    fallback_source_message_id: str | None,
) -> list[_ProactiveTextCandidate]:
    candidates: list[_ProactiveTextCandidate] = []
    for thread in active_threads:
        if not _active_thread_allows_proactive_text(thread, character):
            continue
        source_message_id = (
            thread.last_updated_message_id
            or thread.source_message_id
            or fallback_source_message_id
        )
        basis = thread.last_updated_message_id or thread.source_message_id or (
            _fingerprint(
                "\n".join(
                    (
                        thread.title,
                        thread.description,
                        thread.status,
                        str(thread.priority),
                    )
                )
            )
        )
        reason = f"Follow up on {thread.title.strip() or 'an active thread'}."
        candidates.append(
            _ProactiveTextCandidate(
                character=character,
                trigger_key=f"active_thread:{thread.id}:{basis}",
                trigger_type="active_thread",
                source_type="active_thread",
                source_id=thread.id,
                source_message_id=source_message_id,
                reason=reason,
                context_lines=(
                    "Proactive text trigger: active thread",
                    _join_label("Thread title", thread.title),
                    _join_label("Thread status", thread.status),
                    _join_label("Thread description", thread.description),
                ),
                priority=300 + max(thread.priority, 0),
            )
        )
    return candidates


def _active_thread_allows_proactive_text(
    thread: ActiveThreadRecord,
    character: CharacterRecord,
) -> bool:
    if not _thread_related_to_character(thread, character):
        return False
    if not _active_thread_text_mentions_character(thread, character):
        return False
    visibility = normalize_active_thread_visibility(thread.visibility)
    if visibility == "public":
        return True
    if visibility == "scene":
        return _active_thread_is_text_native(thread)
    return False


def _dating_route_candidate(
    *,
    character: CharacterRecord,
    route: DatingRouteStateRecord,
    fallback_source_message_id: str | None,
) -> _ProactiveTextCandidate | None:
    unresolved = "; ".join(route.unresolved_questions)
    reason = route.next_reasonable_step.strip() or unresolved
    if not reason:
        return None
    source_message_id = (
        route.last_interaction_message_id
        or route.source_message_id
        or fallback_source_message_id
    )
    basis = _fingerprint(
        "\n".join(
            (
                route.stage,
                route.next_reasonable_step,
                unresolved,
            )
        )
    )
    return _ProactiveTextCandidate(
        character=character,
        trigger_key=f"dating_route:{route.id}:{basis}",
        trigger_type="dating_route",
        source_type="dating_route_state",
        source_id=route.id,
        source_message_id=source_message_id,
        reason=reason,
        context_lines=(
            "Proactive text trigger: dating route follow-up",
            _join_label("Route stage", route.stage),
            _join_label("Comfort with intimacy", route.comfort_with_intimacy),
            _join_label("Pacing", route.pacing_preference),
            _join_label("Known boundaries", "; ".join(route.known_boundaries)),
            _join_label("Next step", route.next_reasonable_step),
            _join_label("Unresolved questions", unresolved),
        ),
        priority=200,
    )


def _character_intent_candidate(
    *,
    character: CharacterRecord,
    fallback_source_message_id: str | None,
) -> _ProactiveTextCandidate | None:
    intent = character.current_intent.strip()
    if not intent:
        return None
    basis = (
        character.last_updated_message_id
        or character.source_message_id
        or _fingerprint(intent)
    )
    return _ProactiveTextCandidate(
        character=character,
        trigger_key=f"character_intent:{character.id}:{basis}",
        trigger_type="character_intent",
        source_type="character",
        source_id=character.id,
        source_message_id=(
            character.last_updated_message_id
            or character.source_message_id
            or fallback_source_message_id
        ),
        reason=intent,
        context_lines=(
            "Proactive text trigger: character current intent",
            _join_label("Current intent", intent),
        ),
        priority=100,
    )


def _thread_related_to_character(
    thread: ActiveThreadRecord,
    character: CharacterRecord,
) -> bool:
    wanted = set()
    for reference in _character_reference_values(character):
        normalized = _normalize_match_key(reference)
        wanted.add(normalized)
        wanted.add(_normalize_match_key(f"character:{normalized}"))
    for value in thread.related_entities:
        if _normalize_match_key(value) in wanted:
            return True
    combined = f"{thread.title}\n{thread.description}"
    return any(
        _contains_token(combined, reference)
        for reference in _character_reference_values(character)
    )


def _active_thread_text_mentions_character(
    thread: ActiveThreadRecord,
    character: CharacterRecord,
) -> bool:
    combined = f"{thread.title}\n{thread.description}"
    return any(
        _contains_token(combined, reference)
        for reference in _character_reference_values(character)
    )


def _active_thread_is_text_native(thread: ActiveThreadRecord) -> bool:
    return any(
        token in _TEXT_NATIVE_THREAD_TERMS
        for token in _normalized_text_tokens(
            "\n".join((thread.title, thread.description, thread.status))
        )
    )


def _character_reference_values(character: CharacterRecord) -> tuple[str, ...]:
    values = [character.id, character.name, character.contact_name, *character.aliases]
    return tuple(value for value in values if value.strip())


def _contains_token(text: str, token: str) -> bool:
    normalized_text = f" {_normalize_match_key(text)} "
    normalized_token = _normalize_match_key(token)
    return bool(normalized_token) and f" {normalized_token} " in normalized_text


def _normalize_match_key(value: str) -> str:
    return " ".join(re.sub(r"[\W_]+", " ", value.casefold()).split())


def character_text_proactive_random_chance_percent(
    repositories: PersistenceRepositories,
    *,
    save_id: str | None = None,
) -> int:
    value = repositories.get_effective_setting(
        CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING,
        save_id=save_id,
    )
    return sanitize_character_text_proactive_random_chance_percent(value)


def sanitize_character_text_proactive_random_chance_percent(
    value: object,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        return DEFAULT_CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_PERCENT
    return min(
        max(value, MIN_CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_PERCENT),
        MAX_CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_PERCENT,
    )


def character_text_proactive_random_cooldown_turns(
    repositories: PersistenceRepositories,
    *,
    save_id: str | None = None,
) -> int:
    value = repositories.get_effective_setting(
        CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING,
        save_id=save_id,
    )
    return sanitize_character_text_proactive_random_cooldown_turns(value)


def sanitize_character_text_proactive_random_cooldown_turns(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        return DEFAULT_CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_TURNS
    return min(
        max(value, MIN_CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_TURNS),
        MAX_CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_TURNS,
    )


def _fingerprint(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()[:16]


def _contact_model(
    character: CharacterRecord,
    *,
    contact_state: CharacterContactStateRecord | None,
    thread: CharacterTextThreadRecord | None,
    latest: CharacterTextMessageRecord | None,
    reference_image: CharacterRegistryReferenceImageRow | None = None,
) -> CharacterTextContact:
    from bragi.application.chronicle import parse_message_markdown

    if latest is not None:
        latest_markdown_blocks = parse_message_markdown(latest.body)
    else:
        latest_markdown_blocks = ()
    return CharacterTextContact(
        id=character.id,
        name=character.name,
        contact_name=character.contact_name,
        role=character.role,
        status=character.status,
        is_player_character=character.is_player_character,
        player_has_character_number=(
            contact_state.player_has_character_number
            if contact_state is not None
            else False
        ),
        character_has_player_number=(
            contact_state.character_has_player_number
            if contact_state is not None
            else False
        ),
        player_number_permission=_contact_permission(
            contact_state=contact_state,
            allowed=(
                contact_state.player_has_character_number
                if contact_state is not None
                else False
            ),
            direction="player_has_character",
        ),
        character_number_permission=_contact_permission(
            contact_state=contact_state,
            allowed=(
                contact_state.character_has_player_number
                if contact_state is not None
                else False
            ),
            direction="character_has_player",
        ),
        thread_id=thread.id if thread is not None else None,
        latest_message_id=latest.id if latest is not None else None,
        latest_message_body=latest.body if latest is not None else "",
        latest_message_markdown_blocks=latest_markdown_blocks,
        latest_message_sender=latest.sender if latest is not None else None,
        latest_message_at=latest.created_at if latest is not None else None,
        latest_message_read_at=latest.read_at if latest is not None else None,
        reference_image=reference_image,
    )


def _latest_contact_preview_message(
    messages: list[CharacterTextMessageRecord],
) -> CharacterTextMessageRecord | None:
    for message in reversed(messages):
        if message.sender == "character" and message.delivery_status in {
            "pending",
            "retrying",
        }:
            continue
        return message
    return None


def _contact_permission(
    *,
    contact_state: CharacterContactStateRecord | None,
    allowed: bool,
    direction: str,
) -> CharacterTextContactPermission:
    if not allowed:
        if direction == "player_has_character":
            return CharacterTextContactPermission(
                allowed=False,
                source="none",
                reason="You do not have this character's number.",
            )
        return CharacterTextContactPermission(
            allowed=False,
            source="none",
            reason="They cannot text you yet.",
        )
    if (
        direction == "character_has_player"
        and contact_state is not None
        and contact_state.source_text_message_id
    ):
        return CharacterTextContactPermission(
            allowed=True,
            source="text_message",
            reason="They can text you. You texted them first.",
            source_text_message_id=contact_state.source_text_message_id,
        )
    if contact_state is not None and contact_state.source_message_id:
        return CharacterTextContactPermission(
            allowed=True,
            source="chronicle",
            reason=(
                "You can text them. Detected in the Chronicle."
                if direction == "player_has_character"
                else "They can text you. Detected in the Chronicle."
            ),
            source_message_id=contact_state.source_message_id,
        )
    if contact_state is not None and contact_state.source_text_message_id:
        return CharacterTextContactPermission(
            allowed=True,
            source="text_message",
            reason=(
                "You can text them. Based on text message history."
                if direction == "player_has_character"
                else "They can text you. You texted them first."
            ),
            source_text_message_id=contact_state.source_text_message_id,
        )
    return CharacterTextContactPermission(
        allowed=True,
        source="manual_or_legacy",
        reason=(
            "You can text them. Saved manually or restored from existing save data."
            if direction == "player_has_character"
            else (
                "They can text you. Saved manually or restored from "
                "existing save data."
            )
        ),
    )


def _character_reference_images_by_id(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    character_ids: tuple[str, ...],
) -> dict[str, CharacterRegistryReferenceImageRow]:
    if not character_ids:
        return {}
    registry = CharacterRegistryService(
        repositories,
        active_save_id=save_id,
    ).build_model(active_save_id=save_id)
    return {
        row.character_id: row.reference_image
        for row in registry.characters
        if row.reference_image is not None
    }


def _thread_model(
    thread: CharacterTextThreadRecord,
    *,
    participants: tuple[CharacterTextThreadParticipant, ...] = (),
    messages: tuple[CharacterTextMessage, ...],
) -> CharacterTextThread:
    return CharacterTextThread(
        id=thread.id,
        character_id=thread.character_id,
        title=thread.title,
        status=thread.status,
        kind=thread.kind,
        participants=participants,
        created_at=thread.created_at,
        updated_at=thread.updated_at,
        messages=messages,
    )


def _thread_participant_models(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    thread_id: str,
) -> tuple[CharacterTextThreadParticipant, ...]:
    characters_by_id = {
        character.id: character for character in repositories.list_characters(save_id)
    }
    return tuple(
        CharacterTextThreadParticipant(
            character_id=participant.character_id,
            name=(
                characters_by_id[participant.character_id].name
                if participant.character_id in characters_by_id
                else "Character"
            ),
            ordinal=participant.ordinal,
        )
        for participant in repositories.list_character_text_thread_participants(
            save_id=save_id,
            thread_id=thread_id,
        )
    )


def _message_models(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    messages: tuple[CharacterTextMessageRecord, ...],
) -> tuple[CharacterTextMessage, ...]:
    characters_by_id = {
        character.id: character for character in repositories.list_characters(save_id)
    }
    revision_metadata = repositories.character_text_message_revision_metadata(save_id)
    message_ids = tuple(message.id for message in messages)
    attachments = repositories.list_character_text_message_attachments(
        save_id=save_id,
        text_message_ids=message_ids,
    )
    attachments_by_message: dict[str, list[CharacterTextMessageAttachmentRecord]] = {}
    for attachment in attachments:
        attachments_by_message.setdefault(attachment.text_message_id, []).append(
            attachment
        )
    media_assets = {
        asset.id: asset for asset in repositories.list_media_assets(save_id)
    }
    proactive_by_message: dict[str, CharacterTextProactiveTriggerRecord] = {}
    for trigger in repositories.list_character_text_proactive_triggers(save_id):
        if (
            trigger.text_message_id is not None
            and trigger.text_message_id in message_ids
        ):
            proactive_by_message[trigger.text_message_id] = trigger
    return tuple(
        _message_model(
            message,
            revision_metadata=revision_metadata,
            attachments=tuple(attachments_by_message.get(message.id, ())),
            media_assets=media_assets,
            proactive_trigger=proactive_by_message.get(message.id),
            sender_display_name=_message_sender_display_name(
                message,
                characters_by_id=characters_by_id,
            ),
        )
        for message in messages
    )


def _message_model(
    message: CharacterTextMessageRecord,
    *,
    revision_metadata: Mapping[str, object] | None = None,
    attachments: tuple[CharacterTextMessageAttachmentRecord, ...] = (),
    media_assets: Mapping[str, MediaAssetRecord] | None = None,
    proactive_trigger: CharacterTextProactiveTriggerRecord | None = None,
    sender_display_name: str = "",
) -> CharacterTextMessage:
    from bragi.application.chronicle import parse_message_markdown

    metadata = (revision_metadata or {}).get(message.id)
    revision_count = int(getattr(metadata, "revision_count", 0) or 0)
    edited_at = getattr(metadata, "edited_at", None)
    return CharacterTextMessage(
        id=message.id,
        thread_id=message.thread_id,
        character_id=message.character_id,
        sender=message.sender,
        sender_character_id=message.sender_character_id,
        sender_display_name=sender_display_name,
        body=message.body,
        markdown_blocks=parse_message_markdown(message.body),
        attachments=_attachment_models(
            attachments,
            media_assets=media_assets or {},
        ),
        actions=_message_actions(message),
        provider=message.provider,
        model=message.model,
        token_estimate=message.token_estimate,
        delivery_status=message.delivery_status,
        delivery_error=message.delivery_error,
        delivery_job_id=message.delivery_job_id,
        delivery_attempt=message.delivery_attempt,
        created_at=message.created_at,
        in_world_sent_at=message.in_world_sent_at,
        delivered_at=message.delivered_at,
        read_at=message.read_at,
        reply_to_message_id=message.reply_to_message_id,
        proactive_reason=proactive_trigger.reason if proactive_trigger else "",
        proactive_trigger_type=(
            proactive_trigger.trigger_type if proactive_trigger else ""
        ),
        revision_count=revision_count,
        edited_at=edited_at if isinstance(edited_at, str) else None,
    )


def _message_sender_display_name(
    message: CharacterTextMessageRecord,
    *,
    characters_by_id: Mapping[str, CharacterRecord],
) -> str:
    if message.sender_character_id:
        character = characters_by_id.get(message.sender_character_id)
        if character is not None:
            return character.name
    if message.character_id and message.sender == "character":
        character = characters_by_id.get(message.character_id)
        if character is not None:
            return character.name
    if message.sender == "player":
        return "Player"
    return "Character"


def _completed_character_text_message(
    message: CharacterTextMessageRecord,
    *,
    body: str,
    provider: str | None,
    model: str | None,
    token_estimate: int | None,
    in_world_sent_at: str | None,
) -> CharacterTextMessageRecord:
    return replace(
        message,
        body=body.strip(),
        provider=provider,
        model=model,
        token_estimate=token_estimate,
        delivery_status="sent",
        delivery_error=None,
        in_world_sent_at=in_world_sent_at or message.in_world_sent_at,
    )


def _attachment_models(
    attachments: tuple[CharacterTextMessageAttachmentRecord, ...],
    *,
    media_assets: Mapping[str, MediaAssetRecord],
) -> tuple[CharacterTextMessageAttachment, ...]:
    return tuple(
        _attachment_model(attachment, media_assets=media_assets)
        for attachment in attachments
    )


def _attachment_model(
    attachment: CharacterTextMessageAttachmentRecord,
    *,
    media_assets: Mapping[str, MediaAssetRecord],
) -> CharacterTextMessageAttachment:
    asset = (
        media_assets.get(attachment.media_asset_id)
        if attachment.media_asset_id is not None
        else None
    )
    return CharacterTextMessageAttachment(
        id=attachment.id,
        kind=attachment.kind,
        status=attachment.status,
        media_asset_id=attachment.media_asset_id,
        mime_type=asset.mime_type if asset is not None else None,
        provider=asset.provider if asset is not None else None,
        model=asset.model if asset is not None else None,
        prompt_preview=_attachment_prompt_preview(attachment, asset),
        error=attachment.error,
        created_at=attachment.created_at,
    )


def _attachment_prompt_preview(
    attachment: CharacterTextMessageAttachmentRecord,
    asset: MediaAssetRecord | None,
) -> str:
    text = asset.prompt if asset is not None else attachment.prompt
    collapsed = " ".join(text.strip().split())
    if len(collapsed) <= 160:
        return collapsed
    return f"{collapsed[:157].rstrip()}..."


def _attachment_decision_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "attachment_kind": {
                "type": "string",
                "enum": ["none", "character_image", "object_context_image"],
            },
            "visual_prompt": {"type": "string"},
            "wearing": {"type": "string"},
            "current_action": {"type": "string"},
            "facial_expression": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": [
            "attachment_kind",
            "visual_prompt",
            "wearing",
            "current_action",
            "facial_expression",
            "reason",
        ],
        "additionalProperties": False,
    }


def _attachment_decision_messages(
    *,
    save_id: str,
    repositories: PersistenceRepositories,
    character: CharacterRecord,
    message: CharacterTextMessageRecord,
    history: tuple[CharacterTextMessageRecord, ...],
) -> tuple[ChatMessage, ...]:
    uploaded_photo_descriptions = _uploaded_photo_descriptions_for_messages(
        repositories=repositories,
        save_id=save_id,
        messages=history[-8:],
    )
    body = "\n\n".join(
        part
        for part in (
            f"NPC: {character.name}",
            f"NPC text reply: {message.body}",
            "Recent phone thread:\n"
            + "\n".join(
                _text_history_line(
                    item,
                    player_name=_player_name(save_id, repositories),
                    character_name=character.name,
                    uploaded_photo_descriptions=uploaded_photo_descriptions.get(
                        item.id,
                        (),
                    ),
                )
                for item in history[-8:]
            ),
            "Local context:\n"
            + "\n".join(
                _context_lines(
                    save_id=save_id,
                    character=character,
                    repositories=repositories,
                )
            ),
        )
        if part.strip()
    )
    return (
        ChatMessage(
            role="system",
            body=(
                "Decide whether this NPC phone text should include one generated "
                "picture attachment. Most replies should choose none. Choose a "
                "character image only for a plausible selfie, outfit, expression, "
                "pose, or appearance update from the NPC. Choose an object/context "
                "image only for a concrete visible object, clue, gift, document, "
                "location detail, food, ticket, note, or scene detail the NPC is "
                "texting about. The visual prompt must be concise and grounded in "
                "the provided conversation and local context. For character images, "
                "also specify what the character is wearing, what the character is "
                "currently doing or posing as, and the character's facial expression. "
                "Use concise grounded phrases; leave those fields empty for none or "
                "object/context images."
            ),
        ),
        ChatMessage(role="user", body=body),
    )


def _attachment_decision_from_data(
    data: Mapping[str, object],
) -> _AttachmentDecision | None:
    raw_kind = data.get("attachment_kind")
    kind = raw_kind.strip() if isinstance(raw_kind, str) else "none"
    if kind not in {"none", "character_image", "object_context_image"}:
        return None
    raw_prompt = data.get("visual_prompt")
    prompt = raw_prompt.strip() if isinstance(raw_prompt, str) else ""
    raw_reason = data.get("reason")
    reason = raw_reason.strip() if isinstance(raw_reason, str) else ""
    if kind != "none" and not prompt:
        return None
    wearing = _string_field(data.get("wearing"))
    current_action = _string_field(data.get("current_action"))
    facial_expression = _string_field(data.get("facial_expression"))
    return _AttachmentDecision(
        kind=kind,
        visual_prompt=prompt,
        reason=reason,
        wearing=wearing,
        current_action=current_action,
        facial_expression=facial_expression,
    )


def _attachment_visual_prompt(
    decision: _AttachmentDecision,
    *,
    character: CharacterRecord,
) -> str:
    if decision.kind != "character_image":
        return decision.visual_prompt
    field_lines = [
        _detail_line("Wearing", decision.wearing),
        _detail_line("Current action/pose", decision.current_action),
        _detail_line("Facial expression", decision.facial_expression),
    ]
    field_lines = [line for line in field_lines if line]
    if not field_lines:
        return decision.visual_prompt
    detail_lines = [
        f"Character visual direction for {character.name}:",
        *field_lines,
    ]
    details = "\n".join(line for line in detail_lines if line.strip())
    return f"{decision.visual_prompt.strip()}\n\n{details}"


def _detail_line(label: str, value: str) -> str:
    text = _string_field(value)
    if not text:
        return ""
    return f"{label}: {text}"


def _string_field(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _group_response_assessment_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "character_id": {"type": "string"},
            "should_respond": {"type": "boolean"},
            "response_intent": {"type": "string"},
            "reason": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "priority": {"type": "integer", "minimum": 0, "maximum": 100},
        },
        "required": [
            "character_id",
            "should_respond",
            "response_intent",
            "reason",
            "confidence",
            "priority",
        ],
        "additionalProperties": False,
    }


def _group_response_assessment_messages(
    *,
    save_id: str,
    repositories: PersistenceRepositories,
    thread: CharacterTextThreadRecord,
    character: CharacterRecord,
    player_message: CharacterTextMessageRecord,
    participants: tuple[CharacterRecord, ...],
) -> tuple[ChatMessage, ...]:
    history = tuple(
        repositories.list_character_text_messages(
            save_id=save_id,
            thread_id=thread.id,
        )
    )[-8:]
    uploaded_photo_descriptions = _uploaded_photo_descriptions_for_messages(
        repositories=repositories,
        save_id=save_id,
        messages=(*history, player_message),
    )
    body = "\n\n".join(
        part
        for part in (
            _join_label("Group thread", thread.title or "Group text"),
            _group_participant_context_line(participants),
            f"Character being assessed: {character.name} ({character.id})",
            "Latest player text: "
            + _character_text_message_body_for_context(
                player_message,
                uploaded_photo_descriptions=uploaded_photo_descriptions.get(
                    player_message.id,
                    (),
                ),
            ),
            "Recent group thread:\n"
            + "\n".join(
                _group_text_history_line(
                    message,
                    player_name=_player_name(save_id, repositories),
                    characters_by_id={
                        participant.id: participant for participant in participants
                    },
                    uploaded_photo_descriptions=uploaded_photo_descriptions.get(
                        message.id,
                        (),
                    ),
                )
                for message in history
            ),
            "Local context for assessed character:\n"
            + "\n".join(
                _context_lines(
                    save_id=save_id,
                    character=character,
                    repositories=repositories,
                )
            ),
        )
        if part.strip()
    )
    return (
        ChatMessage(
            role="system",
            body=(
                "Decide whether the assessed NPC would naturally respond to the "
                "latest player message in this group text thread. Consider whether "
                "they have something relevant to add, whether another participant "
                "would more likely answer, and whether silence is more in character."
            ),
        ),
        ChatMessage(role="user", body=body),
    )


def _group_response_assessment_from_data(
    data: Mapping[str, object],
    *,
    character: CharacterRecord,
) -> _GroupResponseAssessment | None:
    raw_character_id = data.get("character_id")
    if not isinstance(raw_character_id, str) or raw_character_id != character.id:
        return None
    raw_should_respond = data.get("should_respond")
    should_respond = bool(raw_should_respond)
    raw_intent = data.get("response_intent")
    intent = raw_intent.strip() if isinstance(raw_intent, str) else ""
    raw_reason = data.get("reason")
    reason = raw_reason.strip() if isinstance(raw_reason, str) else ""
    raw_confidence = data.get("confidence")
    confidence = (
        float(raw_confidence)
        if isinstance(raw_confidence, int | float)
        and not isinstance(raw_confidence, bool)
        else 0.0
    )
    raw_priority = data.get("priority")
    priority = (
        int(raw_priority)
        if isinstance(raw_priority, int)
        and not isinstance(raw_priority, bool)
        else 0
    )
    return _GroupResponseAssessment(
        character=character,
        should_respond=should_respond,
        response_intent=intent,
        reason=reason,
        confidence=min(max(confidence, 0.0), 1.0),
        priority=min(max(priority, 0), 100),
    )


def _text_attachment_scene_context(
    *,
    save_id: str,
    character: CharacterRecord,
    message: CharacterTextMessageRecord,
    history: tuple[CharacterTextMessageRecord, ...],
    repositories: PersistenceRepositories,
) -> str:
    player_name = _player_name(save_id, repositories)
    uploaded_photo_descriptions = _uploaded_photo_descriptions_for_messages(
        repositories=repositories,
        save_id=save_id,
        messages=(*history[-12:], message),
    )
    lines = [
        "Phone thread:",
        *(
            _text_history_line(
                item,
                player_name=player_name,
                character_name=character.name,
                uploaded_photo_descriptions=uploaded_photo_descriptions.get(
                    item.id,
                    (),
                ),
            )
            for item in history[-12:]
        ),
        _text_history_line(
            message,
            player_name=player_name,
            character_name=character.name,
            uploaded_photo_descriptions=uploaded_photo_descriptions.get(
                message.id,
                (),
            ),
        ),
        "",
        "Local context:",
        *_context_lines(
            save_id=save_id,
            character=character,
            repositories=repositories,
        ),
    ]
    return "\n".join(line for line in lines if line)


def _text_history_line(
    message: CharacterTextMessageRecord,
    *,
    player_name: str,
    character_name: str,
    uploaded_photo_descriptions: tuple[str, ...] = (),
) -> str:
    speaker = player_name if message.sender == "player" else character_name
    return (
        f"{speaker}: "
        + _character_text_message_body_for_context(
            message,
            uploaded_photo_descriptions=uploaded_photo_descriptions,
        )
    )


def _message_actions(
    message: CharacterTextMessageRecord,
) -> tuple[CharacterTextMessageAction, ...]:
    if _active_character_text_delivery(message):
        return ()
    if message.sender == "player":
        return (
            CharacterTextMessageAction(
                action_id="edit-text-message",
                label="Edit without Resubmit",
            ),
            CharacterTextMessageAction(
                action_id="edit-and-resubmit-text-message",
                label="Edit and Resubmit",
            ),
            CharacterTextMessageAction(
                action_id="delete-text-messages-from-here",
                label="Delete from here",
            ),
        )
    if message.sender == "character":
        return (
            CharacterTextMessageAction(
                action_id="correct-character-text-message",
                label="Correct Text",
            ),
            CharacterTextMessageAction(
                action_id="delete-text-messages-from-here",
                label="Delete from here",
            ),
        )
    return ()


def _active_character_text_delivery(message: CharacterTextMessageRecord) -> bool:
    return message.delivery_status in {"pending", "retrying"}


def refresh_character_text_thread_memory(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    thread_id: str,
    exclude_message_ids: frozenset[str] | None = None,
) -> CharacterTextThreadRecord:
    thread = repositories.get_character_text_thread(
        save_id=save_id,
        thread_id=thread_id,
    )
    if thread is None:
        raise ValueError(f"Unknown character text thread id: {thread_id}")
    character = (
        repositories.get_character(thread.character_id)
        if thread.character_id is not None
        else None
    )
    character_name = (
        character.name
        if character is not None and character.save_id == save_id
        else (thread.title or "Group")
    )
    messages = tuple(
        message
        for message in canonical_character_text_context_messages(
            repositories=repositories,
            save_id=save_id,
            thread_id=thread_id,
        )
        if exclude_message_ids is None or message.id not in exclude_message_ids
    )
    uploaded_photo_descriptions = _uploaded_photo_descriptions_for_messages(
        repositories=repositories,
        save_id=save_id,
        messages=messages,
    )
    characters_by_id = {
        character.id: character for character in repositories.list_characters(save_id)
    }
    body = _character_text_thread_memory_body(
        messages,
        player_name=_player_name(save_id, repositories),
        character_name=character_name,
        characters_by_id=characters_by_id,
        uploaded_photo_descriptions=uploaded_photo_descriptions,
    )
    return repositories.update_character_text_thread_memory(
        save_id=save_id,
        thread_id=thread_id,
        body=body,
        message_count=len(messages) if body else 0,
    )


def _character_text_thread_memory_body(
    messages: tuple[CharacterTextMessageRecord, ...],
    *,
    player_name: str,
    character_name: str,
    characters_by_id: Mapping[str, CharacterRecord] | None = None,
    uploaded_photo_descriptions: Mapping[str, tuple[str, ...]] | None = None,
) -> str:
    if len(messages) <= _MAX_THREAD_CONTEXT_MESSAGES:
        return ""
    descriptions = uploaded_photo_descriptions or {}
    omitted = messages[: -_MAX_THREAD_CONTEXT_MESSAGES]
    important = tuple(
        message
        for message in omitted
        if _thread_memory_message_is_relevant(
            _character_text_message_body_for_context(
                message,
                uploaded_photo_descriptions=descriptions.get(message.id, ()),
            )
        )
    )
    selected = important[-_MAX_THREAD_MEMORY_MESSAGES:] or omitted[
        -_MAX_THREAD_MEMORY_MESSAGES:
    ]
    lines = [
        "Phone thread memory (older than the recent prompt window):",
        f"Older messages omitted from live history: {len(omitted)}.",
    ]
    character_lookup = characters_by_id or {}
    for message in selected:
        speaker = _thread_memory_speaker_name(
            message,
            player_name=player_name,
            character_name=character_name,
            characters_by_id=character_lookup,
        )
        lines.append(
            f"- {speaker}: "
            + _compact_thread_memory_text(
                _character_text_message_body_for_context(
                    message,
                    uploaded_photo_descriptions=descriptions.get(message.id, ()),
                )
            )
        )
    return _truncate_thread_memory_body("\n".join(lines))


def _thread_memory_speaker_name(
    message: CharacterTextMessageRecord,
    *,
    player_name: str,
    character_name: str,
    characters_by_id: Mapping[str, CharacterRecord],
) -> str:
    if message.sender == "player":
        return player_name
    for character_id in (message.sender_character_id, message.character_id):
        if not character_id:
            continue
        character = characters_by_id.get(character_id)
        if character is not None:
            return character.name
    return character_name


def _thread_memory_message_is_relevant(body: str) -> bool:
    normalized = " ".join(body.casefold().split())
    return any(keyword in normalized for keyword in _THREAD_MEMORY_KEYWORDS)


def _compact_thread_memory_text(body: str) -> str:
    compact = " ".join(body.split())
    if len(compact) <= _MAX_THREAD_MEMORY_LINE_CHARS:
        return compact
    return f"{compact[: _MAX_THREAD_MEMORY_LINE_CHARS - 3].rstrip()}..."


def _truncate_thread_memory_body(body: str) -> str:
    if len(body) <= _MAX_THREAD_MEMORY_CHARS:
        return body
    truncated = body[: _MAX_THREAD_MEMORY_CHARS - 3].rstrip()
    return f"{truncated}..."


def _thread_memory_context_lines(
    thread: CharacterTextThreadRecord,
) -> tuple[str, ...]:
    body = thread.memory_body.strip()
    if not body:
        return ()
    return (body,)


def _group_thread_participant_characters(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    thread_id: str,
) -> tuple[CharacterRecord, ...]:
    characters_by_id = {
        character.id: character for character in repositories.list_characters(save_id)
    }
    participants: list[CharacterRecord] = []
    for participant in repositories.list_character_text_thread_participants(
        save_id=save_id,
        thread_id=thread_id,
    ):
        character = characters_by_id.get(participant.character_id)
        if character is None or character.is_player_character:
            continue
        participants.append(character)
    return tuple(participants)


def _group_history_chat_messages(
    *,
    save_id: str,
    repositories: PersistenceRepositories,
    history: tuple[CharacterTextMessageRecord, ...],
) -> tuple[ChatMessage, ...]:
    player_name = _player_name(save_id, repositories)
    characters_by_id = {
        character.id: character for character in repositories.list_characters(save_id)
    }
    uploaded_photo_descriptions = _uploaded_photo_descriptions_for_messages(
        repositories=repositories,
        save_id=save_id,
        messages=history,
    )
    return tuple(
        ChatMessage(
            role="player" if message.sender == "player" else "narrator",
            body=_text_history_body(
                message,
                uploaded_photo_descriptions=uploaded_photo_descriptions.get(
                    message.id,
                    (),
                ),
            ),
            speaker_name=_group_message_speaker_name(
                message,
                player_name=player_name,
                characters_by_id=characters_by_id,
            ),
        )
        for message in history
    )


def _group_participant_context_line(participants: tuple[CharacterRecord, ...]) -> str:
    names = ", ".join(character.name for character in participants)
    return f"Group text participants: {names}" if names else ""


def _history_chat_message(
    message: CharacterTextMessageRecord,
    *,
    player_name: str,
    character_name: str,
    uploaded_photo_descriptions: tuple[str, ...] = (),
) -> ChatMessage:
    body = _text_history_body(
        message,
        uploaded_photo_descriptions=uploaded_photo_descriptions,
    )
    if message.sender == "player":
        return ChatMessage(role="player", body=body, speaker_name=player_name)
    return ChatMessage(role="narrator", body=body, speaker_name=character_name)


def _text_history_body(
    message: CharacterTextMessageRecord,
    *,
    uploaded_photo_descriptions: tuple[str, ...] = (),
) -> str:
    return _character_text_message_body_for_context(
        message,
        uploaded_photo_descriptions=uploaded_photo_descriptions,
    )


def _group_text_history_line(
    message: CharacterTextMessageRecord,
    *,
    player_name: str,
    characters_by_id: Mapping[str, CharacterRecord],
    uploaded_photo_descriptions: tuple[str, ...] = (),
) -> str:
    speaker = _group_message_speaker_name(
        message,
        player_name=player_name,
        characters_by_id=characters_by_id,
    )
    return (
        f"{speaker}: "
        + _character_text_message_body_for_context(
            message,
            uploaded_photo_descriptions=uploaded_photo_descriptions,
        )
    )


def _group_message_speaker_name(
    message: CharacterTextMessageRecord,
    *,
    player_name: str,
    characters_by_id: Mapping[str, CharacterRecord],
) -> str:
    if message.sender == "player":
        return player_name
    if message.sender_character_id:
        character = characters_by_id.get(message.sender_character_id)
        if character is not None:
            return character.name
    if message.character_id:
        character = characters_by_id.get(message.character_id)
        if character is not None:
            return character.name
    return "Character"


def _character_text_message_body_for_context(
    message: CharacterTextMessageRecord,
    *,
    uploaded_photo_descriptions: tuple[str, ...] = (),
) -> str:
    if message.sender == "character":
        body = _character_text_body_without_leaked_metadata(message.body)
    else:
        body = message.body.strip()
    photo_lines = tuple(
        f"[Attached photo visible to recipient: {description}]"
        for description in uploaded_photo_descriptions
        if description.strip()
    )
    if not photo_lines:
        return body
    return "\n".join((body, *photo_lines)).strip()


def _uploaded_photo_descriptions_for_messages(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    messages: Iterable[CharacterTextMessageRecord],
) -> dict[str, tuple[str, ...]]:
    return uploaded_photo_descriptions_by_message_id(
        repositories=repositories,
        save_id=save_id,
        messages=messages,
    )


def _character_text_body_without_leaked_metadata(body: str) -> str:
    cleaned = _stripped_character_text_body_without_leaked_metadata(body)
    return cleaned or body.strip()


def _validated_character_text_response_body(body: str) -> str:
    cleaned = _stripped_character_text_body_without_leaked_metadata(body)
    if not cleaned:
        raise ValueError("Text provider returned an empty reply")
    return cleaned


def _stripped_character_text_body_without_leaked_metadata(body: str) -> str:
    original = body.strip()
    cleaned = original
    while cleaned:
        next_cleaned = _LEAKED_SENT_AT_PREFIX_RE.sub("", cleaned, count=1).strip()
        if next_cleaned == cleaned:
            break
        cleaned = next_cleaned
    return cleaned


def _current_character_text_in_world_timestamp(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
) -> str | None:
    snapshot = repositories.get_scene_snapshot(save_id)
    if snapshot is None:
        return None
    return _character_text_scene_time_display(snapshot) or None


def _is_duplicate_recent_character_text_body(
    *,
    body: str,
    character_id: str,
    history: list[CharacterTextMessageRecord],
) -> bool:
    normalized = _normalized_character_text_body(body)
    return any(
        message.sender == "character"
        and message.character_id == character_id
        and _character_text_bodies_duplicate(
            left=normalized,
            right=_normalized_character_text_body(message.body),
        )
        for message in history
    )


def _character_text_bodies_duplicate(*, left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    if min(len(left), len(right)) < _NEAR_DUPLICATE_MIN_NORMALIZED_LENGTH:
        return False
    return SequenceMatcher(None, left, right).ratio() >= _NEAR_DUPLICATE_RATIO


def _normalized_character_text_body(body: str) -> str:
    return " ".join(_normalized_text_tokens(body))


def _normalized_text_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


def _player_name(save_id: str, repositories: PersistenceRepositories) -> str:
    player = _player_character(save_id, repositories)
    if player is not None:
        return player.name
    return "Player"


def _player_character(
    save_id: str,
    repositories: PersistenceRepositories,
) -> CharacterRecord | None:
    for character in repositories.list_characters(save_id):
        if character.is_player_character:
            return character
    return None


def _grant_player_has_character_number_from_inbound_text(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    character_id: str,
    source_text_message_id: str,
) -> None:
    player = _player_character(save_id, repositories)
    if player is None:
        return
    repositories.upsert_character_contact_state(
        save_id=save_id,
        player_character_id=player.id,
        character_id=character_id,
        player_has_character_number=True,
        source_text_message_id=source_text_message_id,
    )


def _capture_character_text_prompt(
    *,
    prompt_inspection_store: PromptInspectionStore | None,
    message_id: str,
    request: ChatRequest,
    response: object,
) -> None:
    if prompt_inspection_store is None:
        return
    provider_payload = getattr(response, "raw_request_payload", None)
    prompt_inspection_store.capture_chat_request(
        message_id=message_id,
        request=request,
        provider_payload=(
            provider_payload if isinstance(provider_payload, dict) else None
        ),
        kind=_CHARACTER_TEXT_PROMPT_KIND,
        title=_CHARACTER_TEXT_PROMPT_TITLE,
    )


def _text_delivery_retry_callback(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    message_id: str,
    callback: ProviderRetryProgressCallback | None,
) -> ProviderRetryProgressCallback:
    def update(progress: ProviderRetryProgress) -> None:
        try:
            repositories.update_character_text_delivery(
                save_id=save_id,
                message_id=message_id,
                status="retrying",
                error=None,
                attempt=progress.next_attempt,
            )
        finally:
            if callback is not None:
                callback(progress)

    return update


def _character_text_script_violations(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    request: ChatRequest,
    body: str,
) -> tuple[ScriptPolicyViolation, ...]:
    mode = script_guard_mode(repositories, save_id=save_id)
    return text_script_violations(
        body,
        allowed_scripts=allowed_generated_scripts(
            _script_guard_source_texts_from_chat_request(request)
        ),
        mode=mode,
        field_name="character_text_message",
    )


def _script_guard_source_texts_from_chat_request(
    request: ChatRequest,
) -> tuple[str, ...]:
    texts: list[str] = [
        request.scenario_instructions,
        request.custom_instructions,
        request.turn_directive,
        request.summary or "",
    ]
    texts.extend(message.body for message in request.messages)
    texts.extend(request.phone_context)
    texts.extend(request.current_scene_recap)
    texts.extend(request.retrieved_recent_messages)
    texts.extend(request.retrieved_scenario_sections)
    texts.extend(request.retrieved_state)
    texts.extend(request.retrieved_state_changes)
    texts.extend(request.retrieved_character_text_context)
    texts.extend(request.retrieved_memories)
    texts.extend(request.retrieved_observations)
    return tuple(text for text in texts if text.strip())


def _script_guard_retry_feedback(
    violations: tuple[ScriptPolicyViolation, ...],
) -> str:
    scripts = ", ".join(
        sorted({violation.script for violation in violations})
    ) or "an unsupported writing script"
    return (
        "The previous generated text used an unsupported writing script "
        f"({scripts}). Regenerate the text using only the allowed writing "
        "script for the scenario and conversation context."
    )


def _character_text_guard_retry_feedback(
    *,
    script_violations: tuple[ScriptPolicyViolation, ...],
    phrase_violations: tuple[PhraseDenylistViolation, ...],
    identity_violation: _CharacterTextIdentityViolation | None,
    identity: _CharacterTextIdentity,
) -> str:
    feedback: list[str] = []
    if identity_violation is not None:
        feedback.append(_character_text_identity_retry_feedback(identity))
    if script_violations:
        feedback.append(_script_guard_retry_feedback(script_violations))
    if phrase_violations:
        feedback.append(_phrase_denylist_retry_feedback(phrase_violations))
    return "\n\n".join(feedback)


def _phrase_denylist_retry_feedback(
    violations: tuple[PhraseDenylistViolation, ...],
) -> str:
    phrases = ", ".join(
        repr(phrase)
        for phrase in sorted({violation.phrase for violation in violations})
    ) or "a denied phrase"
    return (
        "The previous generated text used denied repeated phrasing "
        f"({phrases}). Regenerate the text without those phrases, and do not "
        "substitute close variants of the same stock phrasing."
    )


def _log_character_text_guard_violations(
    *,
    save_id: str,
    message_id: str,
    response: ChatResponse,
    script_violations: tuple[ScriptPolicyViolation, ...],
    phrase_violations: tuple[PhraseDenylistViolation, ...],
    identity_violation: _CharacterTextIdentityViolation | None,
    identity: _CharacterTextIdentity,
    retry: bool,
) -> None:
    suffix = "_retry" if retry else ""
    if script_violations:
        log_debug_event(
            f"character_text.script_guard{suffix}_violation",
            save_id=save_id,
            text_message_id=message_id,
            provider=response.provider,
            model=response.model_id,
            **first_violation_diagnostic(script_violations),
        )
    if phrase_violations:
        log_debug_event(
            f"character_text.phrase_denylist{suffix}_violation",
            save_id=save_id,
            text_message_id=message_id,
            provider=response.provider,
            model=response.model_id,
            **first_phrase_violation_diagnostic(phrase_violations),
        )
    if identity_violation is not None:
        log_debug_event(
            f"character_text.identity_guard{suffix}_violation",
            save_id=save_id,
            text_message_id=message_id,
            provider=response.provider,
            model=response.model_id,
            target_character=identity.character_name,
            player_character=identity.player_name,
            reason=identity_violation.reason,
        )


def _character_text_identity_retry_feedback(
    identity: _CharacterTextIdentity,
) -> str:
    return (
        "The previous generated text used the player character identity. "
        f"Regenerate exactly one phone text as {identity.character_name}. "
        f"Do not write as {identity.player_name}, include "
        f"{identity.player_name} as a sender label, or claim "
        f"{identity.player_name}'s first-person identity."
    )


def _character_text_identity_violation(
    *,
    body: str,
    identity: _CharacterTextIdentity,
) -> _CharacterTextIdentityViolation | None:
    normalized_body = _normalized_identity_text(body)
    player_name = _normalized_identity_text(identity.player_name).strip()
    character_name = _normalized_identity_text(identity.character_name).strip()
    if (
        not player_name
        or not character_name
        or player_name.casefold() == character_name.casefold()
    ):
        return None
    player_pattern = _literal_name_pattern(player_name)
    if not player_pattern:
        return None
    if re.match(
        rf"^\s*(?:>\s*)?{player_pattern}(?:\s*:\s*|\s+-\s+)",
        normalized_body,
        flags=re.IGNORECASE,
    ):
        return _CharacterTextIdentityViolation(reason="player_sender_label")
    declaration_pattern = re.compile(
        rf"\b(?:i\s+am|i['`]m|this\s+is|it['`]s|my\s+name\s+is)\s+"
        rf"{player_pattern}{_IDENTITY_DECLARATION_NAME_BOUNDARY_RE}",
        flags=re.IGNORECASE,
    )
    if declaration_pattern.search(normalized_body):
        return _CharacterTextIdentityViolation(reason="player_first_person_identity")
    return None


def _normalized_identity_text(value: str) -> str:
    return value.translate(_IDENTITY_TEXT_TRANSLATION)


def _literal_name_pattern(name: str) -> str:
    parts = tuple(part for part in name.split() if part)
    return r"\s+".join(re.escape(part) for part in parts)


def _character_text_identity_error(identity: _CharacterTextIdentity) -> str:
    return (
        "Text provider returned player character identity "
        f"({identity.player_name}) instead of replying as "
        f"{identity.character_name}"
    )


def _combine_regeneration_feedback(existing: str, addition: str) -> str:
    existing = existing.strip()
    addition = addition.strip()
    if existing and addition:
        return f"{existing}\n\n{addition}"
    return existing or addition


def _character_chronicle_context_lines(
    *,
    save_id: str,
    character: CharacterRecord,
    repositories: PersistenceRepositories,
    source_message_ids: tuple[str, ...] = (),
) -> tuple[str, ...]:
    message_visibility = repositories.list_message_visibility(
        save_id,
        character_ids={character.id},
    )
    lines: list[str] = []
    seen_message_ids: set[str] = set()
    for message_id in _bounded_source_chronicle_message_ids(source_message_ids):
        message = repositories.get_message(save_id=save_id, message_id=message_id)
        if message is None or not _message_visible_to_texting_character(
            message,
            character=character,
            message_visibility=message_visibility,
        ):
            continue
        lines.append(
            _character_chronicle_context_line(
                message,
                relevance_note=(
                    f"source event for this text visible to {character.name}"
                ),
            )
        )
        seen_message_ids.add(message.id)

    recent_messages = repositories.list_message_page(
        save_id,
        limit=_MAX_CHARACTER_TEXT_CHRONICLE_SCAN_MESSAGES,
    ).messages
    selected_recent: list[MessageRecord] = []
    for message in reversed(recent_messages):
        if len(selected_recent) >= _MAX_CHARACTER_TEXT_RECENT_CHRONICLE_MESSAGES:
            break
        if message.id in seen_message_ids:
            continue
        if not _message_visible_to_texting_character(
            message,
            character=character,
            message_visibility=message_visibility,
        ):
            continue
        selected_recent.append(message)
        seen_message_ids.add(message.id)
    for message in reversed(selected_recent):
        lines.append(
            _character_chronicle_context_line(
                message,
                relevance_note=f"recent chronicle context visible to {character.name}",
            )
        )
    return tuple(lines)


def _bounded_source_chronicle_message_ids(
    source_message_ids: tuple[str, ...],
) -> tuple[str, ...]:
    ordered = tuple(
        dict.fromkeys(message_id.strip() for message_id in source_message_ids)
    )
    return tuple(
        message_id
        for message_id in ordered
        if message_id
    )[-_MAX_CHARACTER_TEXT_SOURCE_CHRONICLE_MESSAGES:]


def _message_visible_to_texting_character(
    message: MessageRecord,
    *,
    character: CharacterRecord,
    message_visibility: list[MessageVisibilityRecord],
) -> bool:
    return message_visible_to_present_characters(
        message_id=message.id,
        present_character_ids=frozenset({character.id}),
        message_visibility=message_visibility,
    )


def _character_chronicle_context_line(
    message: MessageRecord,
    *,
    relevance_note: str,
) -> str:
    speaker = message.speaker_name or message.role.title()
    body = _compact_character_chronicle_text(message.body)
    return (
        f"[message:{message.id}] {speaker} ({message.role}): {body} "
        f"(relevance: {relevance_note})"
    )


def _compact_character_chronicle_text(text: str) -> str:
    compact = " ".join(text.split())
    if len(compact) <= _MAX_CHARACTER_TEXT_CHRONICLE_LINE_CHARS:
        return compact
    return (
        compact[: max(0, _MAX_CHARACTER_TEXT_CHRONICLE_LINE_CHARS - 3)].rstrip()
        + "..."
    )


def _context_lines(
    *,
    save_id: str,
    character: CharacterRecord,
    repositories: PersistenceRepositories,
) -> list[str]:
    lines = [
        f"Character text contact: {character.name}",
        _join_label("Character profile role", character.role),
        _join_label("Character profile known state", character.known_state),
        _join_label("Character profile personality", character.personality),
        _join_label("Character profile voice", character.voice),
        _join_label("Character texting style", character.texting_style),
        _join_label("Character profile status", character.status),
        _join_label("Character profile goals", character.goals),
        _join_label("Character profile motivations", character.motivations),
        _join_label("Character profile current intent", character.current_intent),
        _join_label("Character profile boundaries", character.boundaries),
        _join_label(
            "Character profile attitude toward player",
            character.attitude_toward_player,
        ),
        _route_context(save_id=save_id, character=character, repositories=repositories),
        *_character_scoped_knowledge_context(
            save_id=save_id,
            character=character,
            repositories=repositories,
        ),
        _scene_context(
            save_id=save_id,
            character=character,
            repositories=repositories,
        ),
    ]
    return [line for line in lines if line]


def _character_text_identity(
    *,
    save_id: str,
    character: CharacterRecord,
    repositories: PersistenceRepositories,
) -> _CharacterTextIdentity:
    return _CharacterTextIdentity(
        character_name=character.name,
        player_name=_player_name(save_id, repositories),
    )


def _character_text_identity_context_lines(
    *,
    identity: _CharacterTextIdentity,
    group_participants: tuple[CharacterRecord, ...] = (),
    target_character_id: str | None = None,
) -> tuple[str, ...]:
    lines = [
        f"Target text character: {identity.character_name}",
        f"Player character (do not portray): {identity.player_name}",
        _character_text_identity_instruction(identity),
    ]
    other_participants = tuple(
        participant.name
        for participant in group_participants
        if target_character_id is None or participant.id != target_character_id
    )
    if other_participants:
        lines.append(
            "Other group participants (context only): "
            + ", ".join(other_participants)
        )
    return tuple(lines)


def _character_text_identity_instruction(identity: _CharacterTextIdentity) -> str:
    return (
        f"Only write as {identity.character_name}. "
        f"Do not write as {identity.player_name}, speak from "
        f"{identity.player_name}'s first-person perspective, or claim "
        f"{identity.player_name}'s name, history, actions, or thoughts as your own."
    )


def _phone_context_lines(
    *,
    save_id: str,
    character: CharacterRecord,
    repositories: PersistenceRepositories,
) -> list[str]:
    snapshot = repositories.get_scene_snapshot(save_id)
    lines = [
        f"Phone context contact: {character.name}",
        _phone_scene_presence_line(snapshot=snapshot, character=character),
        *_phone_time_context_lines(snapshot),
        _join_label("Known character status", character.status),
        _join_label("Known character current intent", character.current_intent),
        _phone_location_context_line(character=character, repositories=repositories),
        (
            "Active-scene details omitted from phone context are not known to "
            "this character unless they are present in the scene or established "
            "knowledge says otherwise."
        ),
    ]
    return [line for line in lines if line]


def _phone_scene_presence_line(
    *,
    snapshot: SceneSnapshotRecord | None,
    character: CharacterRecord,
) -> str:
    if snapshot is None:
        return "Phone scene presence: active scene unknown"
    if character.id in set(snapshot.present_character_ids):
        return "Phone scene presence: present in the active scene"
    return "Phone scene presence: off-scene from the active scene"


def _phone_time_context_lines(snapshot: SceneSnapshotRecord | None) -> list[str]:
    if snapshot is None:
        return []
    world_time = format_world_time_from_snapshot(snapshot)
    return [f"Current world time: {world_time}"] if world_time else []


def _phone_location_context_line(
    *,
    character: CharacterRecord,
    repositories: PersistenceRepositories,
) -> str:
    if character.location_id is None:
        return ""
    location = repositories.get_location(character.location_id)
    if location is None or location.save_id != character.save_id:
        return ""
    parts = [location.name.strip()] if location.name.strip() else []
    if location.status.strip():
        parts.append(f"status={location.status.strip()}")
    if not parts:
        return ""
    return "Known character location: " + "; ".join(parts)


def _join_label(label: str, value: str) -> str:
    stripped = value.strip()
    return f"{label}: {stripped}" if stripped else ""


def _route_context(
    *,
    save_id: str,
    character: CharacterRecord,
    repositories: PersistenceRepositories,
) -> str:
    route = _route_for_character(
        save_id=save_id,
        character=character,
        repositories=repositories,
    )
    if route is None:
        return ""
    parts = [
        f"stage={route.stage}",
        f"interactions={route.completed_interactions}",
        _join_label("interest", route.interest_level),
        _join_label("trust", route.trust_level),
        _join_label("comfort with intimacy", route.comfort_with_intimacy),
        _join_label("pacing", route.pacing_preference),
        (
            "known boundaries: " + "; ".join(route.known_boundaries)
            if route.known_boundaries
            else ""
        ),
        (
            "intimacy profile: "
            + intimacy_profile_guidance(
                comfort_with_intimacy=route.comfort_with_intimacy,
                pacing_preference=route.pacing_preference,
                known_boundaries=route.known_boundaries,
            )
        ),
        _join_label("next", route.next_reasonable_step),
    ]
    return "Character dating route: " + "; ".join(part for part in parts if part)


def _route_for_character(
    *,
    save_id: str,
    character: CharacterRecord,
    repositories: PersistenceRepositories,
) -> DatingRouteStateRecord | None:
    for candidate in repositories.list_dating_route_states(save_id):
        if candidate.npc_character_id == character.id:
            return candidate
    return None


def _character_scoped_knowledge_context(
    *,
    save_id: str,
    character: CharacterRecord,
    repositories: PersistenceRepositories,
) -> list[str]:
    edges = [
        edge
        for edge in repositories.list_character_knowledge_edges(
            save_id,
            character_ids={character.id},
        )
        if knowledge_edge_allows_prompt_use(edge)
    ]
    if not edges:
        return []
    memory_by_id = {memory.id: memory for memory in repositories.list_memories(save_id)}
    state_by_id = {state.id: state for state in repositories.list_world_state(save_id)}
    summary_by_id = {
        summary.id: summary for summary in repositories.list_summaries(save_id)
    }
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        target_type = normalized_knowledge_target_type(edge.target_type)
        key = (target_type, edge.target_id)
        if key in seen:
            continue
        seen.add(key)
        line = _character_scoped_knowledge_line(
            character=character,
            target_type=target_type,
            target_id=edge.target_id,
            memory_by_id=memory_by_id,
            state_by_id=state_by_id,
            summary_by_id=summary_by_id,
            may_know=edge.knowledge_state == "may_know",
        )
        if line:
            lines.append(line)
    return lines


def _character_scoped_knowledge_line(
    *,
    character: CharacterRecord,
    target_type: str,
    target_id: str,
    memory_by_id: dict[str, MemoryRecord],
    state_by_id: dict[str, WorldStateRecord],
    summary_by_id: dict[str, SummaryRecord],
    may_know: bool,
) -> str:
    relation = "may know" if may_know else "knows"
    prefix = f"Character-scoped knowledge ({character.name} {relation})"
    if target_type == "memory":
        memory = memory_by_id.get(target_id)
        return f"{prefix} memory: {memory.body}" if memory is not None else ""
    if target_type == "world_state":
        state = state_by_id.get(target_id)
        if state is None:
            return ""
        return (
            f"{prefix} world state: {state.key}: "
            f"{_format_knowledge_value(state.value)}"
        )
    if target_type == "summary":
        summary = summary_by_id.get(target_id)
        return f"{prefix} summary: {summary.body}" if summary is not None else ""
    return ""


def _format_knowledge_value(value: object) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _scene_context(
    *,
    save_id: str,
    character: CharacterRecord,
    repositories: PersistenceRepositories,
) -> str:
    snapshot = repositories.get_scene_snapshot(save_id)
    if snapshot is None or character.id not in set(snapshot.present_character_ids):
        return ""
    parts = [
        _join_label("Situation", snapshot.situation),
        _join_label("Objective", snapshot.objective),
        _join_label("In-world time", _character_text_scene_time_display(snapshot)),
        _join_label("Mood", snapshot.mood),
    ]
    scene_text = "; ".join(part for part in parts if part)
    return f"Visible scene context: {scene_text}" if scene_text else ""


def _character_text_scene_time_display(snapshot: SceneSnapshotRecord) -> str:
    return (
        format_world_time_from_snapshot(snapshot, include_legacy_detail=True)
        or snapshot.in_world_time.strip()
    )


def _update_text_route(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    character: CharacterRecord,
    player_message: CharacterTextMessageRecord,
    reply: CharacterTextMessageRecord,
) -> None:
    details = repositories.load_save_details(save_id)
    if details is None or details.scenario.type != "dating_sim":
        return
    DatingRouteService(repositories).seed_routes_for_save(save_id)
    route = _route_for_character(
        save_id=save_id,
        character=character,
        repositories=repositories,
    )
    if route is None:
        return
    if _text_route_interaction_already_recorded(
        repositories=repositories,
        save_id=save_id,
        reply=reply,
        route_id=route.id,
    ):
        return
    before = _text_route_audit_value(route)
    updated = repositories.upsert_dating_route_state(
        save_id=save_id,
        player_character_id=route.player_character_id,
        npc_character_id=route.npc_character_id,
        stage=route.stage,
        first_met_message_id=route.first_met_message_id,
        first_met_world_day_index=route.first_met_world_day_index,
        last_interaction_message_id=route.last_interaction_message_id,
        last_interaction_world_day_index=route.last_interaction_world_day_index,
        completed_interactions=route.completed_interactions + 1,
        dates_completed=route.dates_completed,
        interest_level=route.interest_level,
        trust_level=route.trust_level,
        comfort_with_intimacy=route.comfort_with_intimacy,
        pacing_preference=route.pacing_preference,
        known_boundaries=list(route.known_boundaries),
        unresolved_questions=list(route.unresolved_questions),
        next_reasonable_step=route.next_reasonable_step
        or next_reasonable_step(route.stage),
        source_message_id=route.source_message_id,
    )
    source_ref = character_text_source_ref(reply.id)
    repositories.add_context_update_audit(
        save_id=save_id,
        operation="text_exchange",
        entity_type="dating_route_state",
        entity_id=updated.id,
        field_path="completed_interactions",
        before=before,
        after=_text_route_audit_value(updated),
        reason="Character text exchange counted for dating route.",
        confidence=1.0,
        source_message_ids=[source_ref],
    )
    consumed_candidate = _dating_route_candidate(
        character=character,
        route=updated,
        fallback_source_message_id=None,
    )
    if consumed_candidate is not None:
        repositories.add_character_text_proactive_trigger(
            save_id=save_id,
            character_id=character.id,
            trigger_key=consumed_candidate.trigger_key,
            trigger_type=consumed_candidate.trigger_type,
            thread_id=reply.thread_id,
            text_message_id=reply.id,
            source_type=consumed_candidate.source_type,
            source_id=consumed_candidate.source_id,
            source_message_id=consumed_candidate.source_message_id,
            reason=consumed_candidate.reason,
        )
    repositories.add_character_text_provenance(
        save_id=save_id,
        thread_id=reply.thread_id,
        text_message_id=reply.id,
        target_type="dating_route_state",
        target_id=updated.id,
        operation="text_exchange",
        field_path="completed_interactions",
    )


def _text_route_interaction_already_recorded(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    reply: CharacterTextMessageRecord,
    route_id: str,
) -> bool:
    return any(
        provenance.target_type == "dating_route_state"
        and provenance.target_id == route_id
        and provenance.operation == "text_exchange"
        and provenance.field_path == "completed_interactions"
        for provenance in repositories.list_character_text_provenance(
            save_id=save_id,
            text_message_id=reply.id,
        )
    )


def _text_route_audit_value(route: DatingRouteStateRecord) -> dict[str, object]:
    return {
        "id": route.id,
        "stage": route.stage,
        "completed_interactions": route.completed_interactions,
        "dates_completed": route.dates_completed,
        "interest_level": route.interest_level,
        "trust_level": route.trust_level,
        "comfort_with_intimacy": route.comfort_with_intimacy,
        "pacing_preference": route.pacing_preference,
        "known_boundaries": list(route.known_boundaries),
        "unresolved_questions": list(route.unresolved_questions),
        "next_reasonable_step": route.next_reasonable_step,
    }
