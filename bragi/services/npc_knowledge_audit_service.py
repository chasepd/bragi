"""Structured audit for NPC knowledge-boundary leaks in narrator replies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

from bragi.persistence.models import MessageRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatMessage,
    ChatRequest,
    ProviderClient,
    StructuredOutputProvider,
    StructuredOutputRequest,
)
from bragi.services.model_preferences import roleplay_model_preference
from bragi.services.openrouter_routing_settings import request_with_openrouter_routing

NPC_KNOWLEDGE_AUDIT_MODE_SETTING = "npc_knowledge_audit_mode"
NPC_KNOWLEDGE_AUDIT_MODE_SOFT_FAIL = "soft_fail"
NPC_KNOWLEDGE_AUDIT_MODE_HARD_FAIL = "hard_fail"
NPC_KNOWLEDGE_AUDIT_MODE_OPTIONS = (
    NPC_KNOWLEDGE_AUDIT_MODE_SOFT_FAIL,
    NPC_KNOWLEDGE_AUDIT_MODE_HARD_FAIL,
)
NPC_KNOWLEDGE_AUDIT_MODES = frozenset(
    NPC_KNOWLEDGE_AUDIT_MODE_OPTIONS
)


@dataclass(frozen=True)
class NpcKnowledgeLeak:
    speaker_name: str
    claim: str
    reason: str
    target_type: str = ""
    target_id: str = ""

    def to_json(self) -> dict[str, object]:
        return {
            "speaker_name": self.speaker_name,
            "claim": self.claim,
            "reason": self.reason,
            "target_type": self.target_type,
            "target_id": self.target_id,
        }


@dataclass(frozen=True)
class NpcKnowledgeAuditResult:
    enabled: bool
    leaks: tuple[NpcKnowledgeLeak, ...] = ()
    skipped_reason: str = ""
    provider: str = ""
    model_id: str = ""
    error: str = ""

    @property
    def leak_count(self) -> int:
        return len(self.leaks)

    @property
    def suspicious(self) -> bool:
        return bool(self.leaks or self.error)

    def to_json(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "leak_count": self.leak_count,
            "leaks": [leak.to_json() for leak in self.leaks],
            "skipped_reason": self.skipped_reason,
            "provider": self.provider,
            "model": self.model_id,
            "error": self.error,
        }


class NpcKnowledgeAuditor(Protocol):
    async def audit_response(
        self,
        *,
        save_id: str,
        player_message: MessageRecord,
        narrator_body: str,
        request: ChatRequest,
    ) -> NpcKnowledgeAuditResult: ...


class NpcKnowledgeAuditService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        providers: dict[str, ProviderClient],
    ) -> None:
        self.repositories = repositories
        self.providers = providers

    async def audit_response(
        self,
        *,
        save_id: str,
        player_message: MessageRecord,
        narrator_body: str,
        request: ChatRequest,
    ) -> NpcKnowledgeAuditResult:
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="npc_knowledge_audit",
        )
        if preference is None:
            return NpcKnowledgeAuditResult(
                enabled=False,
                skipped_reason="no_model_preference",
            )
        provider_candidate = cast(object, self.providers.get(preference.provider))
        if not isinstance(provider_candidate, StructuredOutputProvider):
            return NpcKnowledgeAuditResult(
                enabled=False,
                skipped_reason="provider_lacks_structured_output",
                provider=preference.provider,
                model_id=preference.model_id,
            )
        provider = provider_candidate
        audit_request = request_with_openrouter_routing(
            self.repositories,
            StructuredOutputRequest(
                provider=preference.provider,
                model_id=preference.model_id,
                schema_name="npc_knowledge_audit",
                schema=_npc_knowledge_audit_schema(),
                messages=_npc_knowledge_audit_messages(
                    player_message=player_message,
                    narrator_body=narrator_body,
                    request=request,
                ),
                temperature=0,
                max_output_tokens=1200,
            ),
            task="npc_knowledge_audit",
            save_id=save_id,
        )
        response = await provider.generate_structured_output(audit_request)
        return NpcKnowledgeAuditResult(
            enabled=True,
            leaks=_npc_knowledge_leaks_from_data(response.data),
            provider=response.provider,
            model_id=response.model_id,
        )


def npc_knowledge_audit_mode(
    repositories: PersistenceRepositories,
    *,
    save_id: str | None = None,
) -> str:
    value = repositories.get_effective_setting(
        NPC_KNOWLEDGE_AUDIT_MODE_SETTING,
        save_id=save_id,
    )
    mode = value if isinstance(value, str) else NPC_KNOWLEDGE_AUDIT_MODE_SOFT_FAIL
    if mode not in NPC_KNOWLEDGE_AUDIT_MODES:
        return NPC_KNOWLEDGE_AUDIT_MODE_SOFT_FAIL
    return mode


def _npc_knowledge_audit_messages(
    *,
    player_message: MessageRecord,
    narrator_body: str,
    request: ChatRequest,
) -> tuple[ChatMessage, ...]:
    context = "\n".join(
        part
        for part in (
            "Current scene recap:",
            *request.current_scene_recap,
            "Retrieved memories:",
            *request.retrieved_memories,
            "Retrieved character text context:",
            *request.retrieved_character_text_context,
            "Retrieved state:",
            *request.retrieved_state,
        )
        if part
    )
    user = (
        "Audit the draft narrator reply for NPC knowledge leaks. A leak exists "
        "only when an NPC uses or reacts to a concrete fact that the provided "
        "context does not establish for that NPC.\n\n"
        f"Latest player message ({player_message.speaker_name or 'Player'}):\n"
        f"{player_message.body}\n\n"
        f"{context}\n\n"
        "Draft narrator reply:\n"
        f"{narrator_body}"
    )
    return (
        ChatMessage(
            role="system",
            body=(
                "You are a strict continuity auditor. Return only structured "
                "schema data through the structured-output API."
            ),
        ),
        ChatMessage(role="user", body=user),
    )


def _npc_knowledge_audit_schema() -> dict[str, Any]:
    leak = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "speaker_name": {"type": "string"},
            "claim": {"type": "string"},
            "reason": {"type": "string"},
            "target_type": {"type": "string"},
            "target_id": {"type": "string"},
        },
        "required": [
            "speaker_name",
            "claim",
            "reason",
            "target_type",
            "target_id",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "leaks": {
                "type": "array",
                "items": leak,
                "maxItems": 8,
            }
        },
        "required": ["leaks"],
    }


def _npc_knowledge_leaks_from_data(
    data: dict[str, Any],
) -> tuple[NpcKnowledgeLeak, ...]:
    raw_leaks = data.get("leaks")
    if not isinstance(raw_leaks, list):
        return ()
    leaks: list[NpcKnowledgeLeak] = []
    for item in raw_leaks:
        if not isinstance(item, dict):
            continue
        speaker_name = str(item.get("speaker_name", "")).strip()
        claim = str(item.get("claim", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if not speaker_name or not claim or not reason:
            continue
        leaks.append(
            NpcKnowledgeLeak(
                speaker_name=speaker_name,
                claim=claim,
                reason=reason,
                target_type=str(item.get("target_type", "")).strip(),
                target_id=str(item.get("target_id", "")).strip(),
            )
        )
    return tuple(leaks)
