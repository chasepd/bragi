"""In-memory prompt inspection support for debug UI."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from bragi.providers.contracts import (
    ChatRequest,
    StructuredOutputRequest,
    ToolCallMessage,
    ToolCallRequest,
)


@dataclass(frozen=True)
class PromptInspectionSourceCard:
    group: str
    title: str
    body: str
    metadata: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromptInspectionEntry:
    kind: str
    title: str
    raw_text: str
    source_cards: tuple[PromptInspectionSourceCard, ...] = ()


class PromptInspectionStore:
    """Keeps captured model requests local to the running app process."""

    def __init__(self) -> None:
        self._requests_by_message_id: dict[str, str] = {}
        self._provider_payloads_by_message_id: dict[str, str] = {}
        self._entries_by_message_id: dict[str, list[PromptInspectionEntry]] = {}

    def capture_chat_request(
        self,
        *,
        message_id: str,
        request: ChatRequest,
        provider_payload: dict[str, object] | None = None,
        kind: str = "narrator_prompt",
        title: str = "Narrator prompt",
    ) -> None:
        self.capture_entry(
            message_id=message_id,
            entry=PromptInspectionEntry(
                kind=kind,
                title=title,
                raw_text=format_chat_request(request),
                source_cards=_chat_request_source_cards(request),
            ),
        )
        if provider_payload is not None:
            self._provider_payloads_by_message_id[message_id] = format_provider_payload(
                provider_payload
            )

    def capture_structured_request(
        self,
        *,
        message_id: str,
        kind: str,
        title: str,
        request: StructuredOutputRequest,
    ) -> None:
        self.capture_entry(
            message_id=message_id,
            entry=PromptInspectionEntry(
                kind=kind,
                title=title,
                raw_text=format_structured_output_request(request),
                source_cards=_structured_request_source_cards(request),
            ),
        )

    def capture_tool_call_request(
        self,
        *,
        message_id: str,
        kind: str,
        title: str,
        request: ToolCallRequest,
    ) -> None:
        self.capture_entry(
            message_id=message_id,
            entry=PromptInspectionEntry(
                kind=kind,
                title=title,
                raw_text=format_tool_call_request(request),
                source_cards=_tool_call_request_source_cards(request),
            ),
        )

    def capture_entry(
        self,
        *,
        message_id: str,
        entry: PromptInspectionEntry,
    ) -> None:
        entries = self._entries_by_message_id.setdefault(message_id, [])
        entries.append(entry)
        self._requests_by_message_id[message_id] = format_prompt_inspection_entries(
            tuple(entries)
        )

    def prompt_for_message(self, message_id: str) -> str | None:
        return self._requests_by_message_id.get(message_id)

    def provider_payload_for_message(self, message_id: str) -> str | None:
        return self._provider_payloads_by_message_id.get(message_id)

    def prompts_by_message_id(self) -> dict[str, str]:
        return dict(self._requests_by_message_id)

    def provider_payloads_by_message_id(self) -> dict[str, str]:
        return dict(self._provider_payloads_by_message_id)

    def entries_for_message(self, message_id: str) -> tuple[PromptInspectionEntry, ...]:
        return tuple(self._entries_by_message_id.get(message_id, ()))


def format_chat_request(request: ChatRequest) -> str:
    payload = asdict(request)
    payload.pop("retry_progress_callback", None)
    return json.dumps(payload, indent=2, sort_keys=True)


def format_structured_output_request(request: StructuredOutputRequest) -> str:
    return json.dumps(asdict(request), indent=2, sort_keys=True)


def format_tool_call_request(request: ToolCallRequest) -> str:
    return json.dumps(asdict(request), indent=2, sort_keys=True)


def format_provider_payload(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def format_prompt_inspection_entries(
    entries: tuple[PromptInspectionEntry, ...],
) -> str:
    if not entries:
        return ""
    lines = ["Source cards"]
    for entry in entries:
        lines.append("")
        lines.append(entry.title)
        if not entry.source_cards:
            lines.append("- No compact source cards available.")
            continue
        current_group = ""
        for card in entry.source_cards:
            if card.group != current_group:
                current_group = card.group
                lines.append(current_group)
            metadata = f" ({'; '.join(card.metadata)})" if card.metadata else ""
            lines.append(f"- {card.title}{metadata}")
            if card.body:
                lines.append(f"  {_compact_text(card.body, 900)}")

    lines.append("")
    lines.append("Raw requests")
    for entry in entries:
        lines.append("")
        lines.append(entry.title)
        lines.append(entry.raw_text)
    return "\n".join(lines)


def _chat_request_source_cards(
    request: ChatRequest,
) -> tuple[PromptInspectionSourceCard, ...]:
    cards: list[PromptInspectionSourceCard] = [
        PromptInspectionSourceCard(
            group="Request",
            title="Provider and model",
            body=f"{request.provider} / {request.model_id}",
        ),
        PromptInspectionSourceCard(
            group="Request",
            title="Narrator mode",
            body=request.narrator_prompt_mode,
            metadata=_narrator_mode_metadata(request),
        )
    ]
    _append_text_card(
        cards,
        "Request",
        "Character planning diagnostics",
        _character_planning_diagnostics_text(request),
    )
    _append_text_card(cards, "Instructions", "Scenario", request.scenario_instructions)
    if request.custom_instructions.strip():
        _append_text_card(
            cards,
            "Instructions",
            "Save response guidance",
            request.custom_instructions,
        )
    else:
        _append_text_card(
            cards,
            "Instructions",
            "User narration guidance",
            request.user_narration_guidance,
        )
    _append_text_card(
        cards,
        "Instructions",
        "Regeneration feedback",
        request.regeneration_feedback,
    )
    for index, message in enumerate(request.messages, start=1):
        title = f"{index}. {message.role}"
        if message.speaker_name:
            title = f"{title} ({message.speaker_name})"
        cards.append(
            PromptInspectionSourceCard(
                group="Conversation",
                title=title,
                body=message.body,
            )
        )
    _append_tuple_cards(cards, "Phone activity", request.phone_activity_context)
    _append_tuple_cards(cards, "Phone context", request.phone_context)
    _append_tuple_cards(cards, "Current scene recap", request.current_scene_recap)
    _append_tuple_cards(cards, "Character voice", request.character_voice_profiles)
    _append_tuple_cards(cards, "Character action plans", request.character_action_plans)
    _append_tuple_cards(cards, "Open obligations", request.open_obligations)
    _append_tuple_cards(
        cards,
        "Pending context review",
        request.pending_context_suggestions,
    )
    _append_tuple_cards(cards, "Scenario sections", request.retrieved_scenario_sections)
    _append_tuple_cards(cards, "World state", request.retrieved_state)
    _append_tuple_cards(cards, "State changes", request.retrieved_state_changes)
    _append_tuple_cards(
        cards,
        "Retrieved chronicle",
        request.retrieved_recent_messages,
    )
    _append_tuple_cards(cards, "Media", request.retrieved_media_assets)
    _append_tuple_cards(
        cards,
        "Character text context",
        request.retrieved_character_text_context,
    )
    _append_tuple_cards(cards, "Memories", request.retrieved_memories)
    _append_tuple_cards(cards, "Observations", request.retrieved_observations)
    _append_text_card(cards, "Summary", "Rolling summary", request.summary or "")
    _append_text_card(cards, "Narration", "Brief", request.narration_brief)
    _append_tuple_cards(cards, "Narration evidence", request.narration_evidence)
    return tuple(cards)


def _structured_request_source_cards(
    request: StructuredOutputRequest,
) -> tuple[PromptInspectionSourceCard, ...]:
    cards = [
        PromptInspectionSourceCard(
            group="Request",
            title=request.schema_name,
            body=_schema_summary(request.schema),
            metadata=(f"provider={request.provider}", f"model={request.model_id}"),
        )
    ]
    for index, message in enumerate(request.messages, start=1):
        cards.append(
            PromptInspectionSourceCard(
                group="Structured messages",
                title=f"{index}. {message.role}",
                body=message.body,
            )
        )
    return tuple(cards)


def _tool_call_request_source_cards(
    request: ToolCallRequest,
) -> tuple[PromptInspectionSourceCard, ...]:
    cards: list[PromptInspectionSourceCard] = [
        PromptInspectionSourceCard(
            group="Request",
            title="Provider and model",
            body=f"{request.provider} / {request.model_id}",
        )
    ]
    for tool in request.tools:
        cards.append(
            PromptInspectionSourceCard(
                group="Tools",
                title=tool.name,
                body="\n".join(
                    part
                    for part in (
                        tool.description,
                        _tool_schema_summary(tool.parameters),
                    )
                    if part
                ),
            )
        )
    for index, message in enumerate(request.messages, start=1):
        title = f"{index}. {message.role}"
        if message.speaker_name:
            title = f"{title} ({message.speaker_name})"
        cards.append(
            PromptInspectionSourceCard(
                group="Tool messages",
                title=title,
                body=_tool_message_body(message),
                metadata=(
                    (f"tool_call_id={message.tool_call_id}",)
                    if message.tool_call_id
                    else ()
                ),
            )
        )
    return tuple(cards)


def _narrator_mode_metadata(request: ChatRequest) -> tuple[str, ...]:
    counts = request.context_breakdown.get("narrator_context_withheld_counts")
    chars = request.context_breakdown.get("narrator_context_withheld_chars")
    metadata: list[str] = []
    if isinstance(counts, dict):
        metadata.append(f"withheld_count={_sum_int_values(counts)}")
    if isinstance(chars, dict):
        metadata.append(f"withheld_chars={_sum_int_values(chars)}")
    return tuple(metadata)


def _character_planning_diagnostics_text(request: ChatRequest) -> str:
    diagnostics = request.context_breakdown.get("character_action_planning")
    if not isinstance(diagnostics, dict):
        return ""
    return json.dumps(diagnostics, indent=2, sort_keys=True)


def _sum_int_values(values: dict[object, object]) -> int:
    return sum(
        value
        for value in values.values()
        if isinstance(value, int) and not isinstance(value, bool)
    )


def _tool_message_body(message: ToolCallMessage) -> str:
    parts = [message.body] if message.body else []
    if message.tool_calls:
        parts.append("Tool calls:")
        for call in message.tool_calls:
            parts.append(f"- {call.name}: {_compact_text(call.arguments_json, 300)}")
    return "\n".join(parts)


def _append_text_card(
    cards: list[PromptInspectionSourceCard],
    group: str,
    title: str,
    body: str,
) -> None:
    if body.strip():
        cards.append(PromptInspectionSourceCard(group=group, title=title, body=body))


def _append_tuple_cards(
    cards: list[PromptInspectionSourceCard],
    group: str,
    values: tuple[str, ...],
) -> None:
    for index, value in enumerate(values, start=1):
        if value.strip():
            cards.append(
                PromptInspectionSourceCard(
                    group=group,
                    title=f"{group} {index}",
                    body=value,
                )
            )


def _schema_summary(schema: dict[str, object]) -> str:
    properties = schema.get("properties")
    if isinstance(properties, dict) and properties:
        return "Structured schema fields: " + ", ".join(str(key) for key in properties)
    return "Structured output schema attached. See raw request for exact JSON."


def _tool_schema_summary(schema: dict[str, object]) -> str:
    properties = schema.get("properties")
    if isinstance(properties, dict) and properties:
        return "Tool schema fields: " + ", ".join(str(key) for key in properties)
    return "Tool schema attached. See raw request for exact JSON."


def _compact_text(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "..."
