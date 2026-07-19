from __future__ import annotations

import json

from bragi.providers.contracts import (
    ChatMessage,
    ChatRequest,
    StructuredOutputRequest,
    ToolCallMessage,
    ToolCallRequest,
    ToolDefinition,
)
from bragi.services.prompt_inspection import (
    PromptInspectionStore,
    format_chat_request,
    format_provider_payload,
    format_structured_output_request,
    format_tool_call_request,
)


def test_format_chat_request_outputs_stable_pretty_json() -> None:
    request = ChatRequest(
        provider="fake",
        model_id="fake-chat",
        messages=(ChatMessage(role="player", body="Look north."),),
        retrieved_state=("scene.location=Gate",),
    )

    rendered = format_chat_request(request)

    assert json.loads(rendered)["messages"] == [
        {"role": "player", "body": "Look north.", "speaker_name": None}
    ]
    assert rendered.splitlines()[0] == "{"
    assert "  " in rendered


def test_format_chat_request_excludes_retry_progress_callback() -> None:
    request = ChatRequest(
        provider="fake",
        model_id="fake-chat",
        messages=(ChatMessage(role="player", body="Look north."),),
        retry_progress_callback=lambda _progress: None,
    )

    rendered = json.loads(format_chat_request(request))

    assert "retry_progress_callback" not in rendered


def test_prompt_inspection_store_captures_prompt_and_provider_payload() -> None:
    store = PromptInspectionStore()
    request = ChatRequest(
        provider="fake",
        model_id="fake-chat",
        messages=(ChatMessage(role="player", body="Wait."),),
        pending_context_suggestions=(
            "Pending review (not canon yet): update world_state/storm.mood -> wary",
        ),
    )

    store.capture_chat_request(
        message_id="message-1",
        request=request,
        provider_payload={"model": "fake-chat", "stream": False},
    )

    rendered = store.prompt_for_message("message-1") or ""
    assert "Source cards" in rendered
    assert "Narrator prompt" in rendered
    assert "Conversation" in rendered
    assert "Pending context review" in rendered
    assert "not canon yet" in rendered
    assert "Raw requests" in rendered
    assert '"model_id": "fake-chat"' in rendered
    assert json.loads(store.provider_payload_for_message("message-1") or "{}") == {
        "model": "fake-chat",
        "stream": False,
    }
    assert store.prompt_for_message("missing") is None
    assert store.provider_payload_for_message("missing") is None
    assert set(store.prompts_by_message_id()) == {"message-1"}
    assert set(store.provider_payloads_by_message_id()) == {"message-1"}
    assert [entry.kind for entry in store.entries_for_message("message-1")] == [
        "narrator_prompt"
    ]


def test_prompt_inspection_store_allows_custom_chat_entry_title() -> None:
    store = PromptInspectionStore()
    request = ChatRequest(
        provider="fake",
        model_id="fake-chat",
        messages=(ChatMessage(role="player", body="Meet me after class."),),
        phone_context=("Phone context contact: Rowan",),
        current_scene_recap=("Character profile: Rowan is guarded.",),
    )

    store.capture_chat_request(
        message_id="text-message-1",
        request=request,
        kind="character_text_prompt",
        title="Character text prompt",
    )

    rendered = store.prompt_for_message("text-message-1") or ""
    assert "Character text prompt" in rendered
    assert "Phone context" in rendered
    assert "Phone context contact: Rowan" in rendered
    assert "Character profile: Rowan is guarded." in rendered
    assert [entry.kind for entry in store.entries_for_message("text-message-1")] == [
        "character_text_prompt"
    ]


def test_prompt_inspection_source_cards_include_narrator_mode_diagnostics() -> None:
    store = PromptInspectionStore()
    request = ChatRequest(
        provider="fake",
        model_id="fake-chat",
        messages=(ChatMessage(role="player", body="Wait."),),
        narrator_prompt_mode="plan_first",
        context_breakdown={
            "narrator_context_withheld_counts": {
                "baseline_recent_messages": 2,
                "retrieved_memories": 1,
            },
            "narrator_context_withheld_chars": {
                "baseline_recent_messages": 42,
                "retrieved_memories": 80,
            },
            "character_action_planning": {
                "assessment_count": 0,
                "failed_character_ids": [],
                "failed_count": 0,
                "prompt_guidance_count": 0,
                "skipped_reason": "disabled",
                "applied_presence_update": False,
            },
        },
    )

    store.capture_chat_request(message_id="message-1", request=request)

    rendered = store.prompt_for_message("message-1") or ""
    assert "Narrator mode" in rendered
    assert "plan_first" in rendered
    assert "withheld_count=3" in rendered
    assert "withheld_chars=122" in rendered
    assert "Character planning diagnostics" in rendered
    assert '"skipped_reason": "disabled"' in rendered


def test_prompt_inspection_store_groups_structured_requests_with_raw_fallback() -> None:
    store = PromptInspectionStore()
    request = StructuredOutputRequest(
        provider="fake",
        model_id="fake-structured",
        schema_name="context_update_context_selection",
        schema={"type": "object", "properties": {"selections": {}}},
        messages=(
            ChatMessage(role="system", body="Select prior context."),
            ChatMessage(role="user", body="Prior context candidates: memory:1"),
        ),
    )

    store.capture_structured_request(
        message_id="message-1",
        kind="context_selection",
        title="Context selection",
        request=request,
    )

    rendered = store.prompt_for_message("message-1") or ""
    assert "Context selection" in rendered
    assert "Structured messages" in rendered
    assert "Prior context candidates" in rendered
    assert "Raw requests" in rendered
    assert '"schema_name": "context_update_context_selection"' in rendered
    assert [entry.kind for entry in store.entries_for_message("message-1")] == [
        "context_selection"
    ]


def test_prompt_inspection_store_captures_tool_call_requests() -> None:
    store = PromptInspectionStore()
    request = ToolCallRequest(
        provider="fake",
        model_id="fake-tools",
        messages=(
            ToolCallMessage(role="system", body="Use tools."),
            ToolCallMessage(role="user", body="Completed turn: beacon lens."),
        ),
        tools=(
            ToolDefinition(
                name="record_memory_fact",
                description="Record one durable memory fact.",
                parameters={
                    "type": "object",
                    "properties": {
                        "body": {"type": "string"},
                        "source_message_id": {"type": "string"},
                    },
                    "required": ["body", "source_message_id"],
                },
            ),
        ),
        temperature=0.0,
        parallel_tool_calls=False,
    )

    store.capture_tool_call_request(
        message_id="message-1",
        kind="state_memory_tool_calls",
        title="State and memory tool calls",
        request=request,
    )

    rendered = store.prompt_for_message("message-1") or ""
    assert "State and memory tool calls" in rendered
    assert "Tool messages" in rendered
    assert "Tools" in rendered
    assert "record_memory_fact" in rendered
    assert "Tool schema fields: body, source_message_id" in rendered
    assert "Raw requests" in rendered
    assert '"parallel_tool_calls": false' in rendered
    assert [entry.kind for entry in store.entries_for_message("message-1")] == [
        "state_memory_tool_calls"
    ]


def test_format_provider_payload_sorts_keys() -> None:
    assert format_provider_payload({"z": 1, "a": 2}).splitlines()[1] == '  "a": 2,'


def test_format_structured_output_request_outputs_stable_pretty_json() -> None:
    request = StructuredOutputRequest(
        provider="fake",
        model_id="fake-structured",
        schema_name="memory_consolidation",
        schema={"type": "object"},
        messages=(ChatMessage(role="user", body="Active memories: none"),),
    )

    rendered = format_structured_output_request(request)

    assert json.loads(rendered)["schema_name"] == "memory_consolidation"
    assert rendered.splitlines()[0] == "{"


def test_format_tool_call_request_outputs_stable_pretty_json() -> None:
    request = ToolCallRequest(
        provider="fake",
        model_id="fake-tools",
        messages=(ToolCallMessage(role="user", body="Select useful context."),),
        tools=(
            ToolDefinition(
                name="select_context_source",
                description="Select one offered context source.",
                parameters={"type": "object"},
            ),
        ),
    )

    rendered = format_tool_call_request(request)

    parsed = json.loads(rendered)
    assert parsed["messages"] == [
        {
            "body": "Select useful context.",
            "role": "user",
            "speaker_name": None,
            "tool_call_id": None,
            "tool_calls": [],
        }
    ]
    assert parsed["tools"][0]["name"] == "select_context_source"
    assert rendered.splitlines()[0] == "{"
