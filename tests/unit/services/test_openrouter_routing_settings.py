from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatMessage,
    ChatRequest,
    StructuredOutputRequest,
    ToolCallMessage,
    ToolCallRequest,
    ToolDefinition,
)
from bragi.services.generation_settings import MODEL_THINKING_PREFERENCES_SETTING
from bragi.services.openrouter_routing_settings import (
    OPENROUTER_ROUTING_PROFILES_SETTING,
    openrouter_app_title_for_task,
    openrouter_routing_payload_for_task,
    request_with_openrouter_routing,
    sanitize_openrouter_routing_profiles,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_sanitize_openrouter_routing_profiles_keeps_safe_documented_fields() -> None:
    assert sanitize_openrouter_routing_profiles(
        {
            "global": {
                "order": ["deepinfra/turbo", "bad slug", "deepinfra/turbo"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
                "zdr": True,
                "enforce_distillable_text": True,
                "only": ["anthropic"],
                "ignore": ["openai"],
                "quantizations": ["fp8", "wat"],
                "sort": "throughput",
                "sort_partition": "none",
                "preferred_min_throughput": {"p90": 50, "p50": -1},
                "preferred_max_latency": {"p90": 3, "p99": "slow"},
                "max_price": {"prompt": 1, "completion": 2, "request": -1},
            },
            "task_overrides": {
                "narrator": {
                    "enabled": True,
                    "profile": {"sort": "latency"},
                },
                "missing": {
                    "enabled": True,
                    "profile": {"sort": "price"},
                },
            },
        }
    ) == {
        "global": {
            "order": ["deepinfra/turbo"],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
            "zdr": True,
            "enforce_distillable_text": True,
            "only": ["anthropic"],
            "ignore": ["openai"],
            "quantizations": ["fp8"],
            "sort": "throughput",
            "sort_partition": "none",
            "preferred_min_throughput": {"p90": 50.0},
            "preferred_max_latency": {"p90": 3.0},
            "max_price": {"prompt": 1.0, "completion": 2.0},
        },
        "task_overrides": {
            "narrator": {
                "enabled": True,
                "profile": {
                    "order": [],
                    "allow_fallbacks": None,
                    "require_parameters": False,
                    "data_collection": "allow",
                    "zdr": False,
                    "enforce_distillable_text": False,
                    "only": [],
                    "ignore": [],
                    "quantizations": [],
                    "sort": "latency",
                    "sort_partition": "model",
                    "preferred_min_throughput": {},
                    "preferred_max_latency": {},
                    "max_price": {},
                },
            },
            "background_text": {
                "enabled": False,
                "profile": {
                    "order": [],
                    "allow_fallbacks": None,
                    "require_parameters": False,
                    "data_collection": "allow",
                    "zdr": False,
                    "enforce_distillable_text": False,
                    "only": [],
                    "ignore": [],
                    "quantizations": [],
                    "sort": "default",
                    "sort_partition": "model",
                    "preferred_min_throughput": {},
                    "preferred_max_latency": {},
                    "max_price": {},
                },
            },
            "structured_tool": {
                "enabled": False,
                "profile": {
                    "order": [],
                    "allow_fallbacks": None,
                    "require_parameters": False,
                    "data_collection": "allow",
                    "zdr": False,
                    "enforce_distillable_text": False,
                    "only": [],
                    "ignore": [],
                    "quantizations": [],
                    "sort": "default",
                    "sort_partition": "model",
                    "preferred_min_throughput": {},
                    "preferred_max_latency": {},
                    "max_price": {},
                },
            },
            "media": {
                "enabled": False,
                "profile": {
                    "order": [],
                    "allow_fallbacks": None,
                    "require_parameters": False,
                    "data_collection": "allow",
                    "zdr": False,
                    "enforce_distillable_text": False,
                    "only": [],
                    "ignore": [],
                    "quantizations": [],
                    "sort": "default",
                    "sort_partition": "model",
                    "preferred_min_throughput": {},
                    "preferred_max_latency": {},
                    "max_price": {},
                },
            },
        },
    }


def test_openrouter_routing_payload_uses_global_and_task_override(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_app_setting(
        OPENROUTER_ROUTING_PROFILES_SETTING,
        {
            "global": {
                "sort": "price",
                "allow_fallbacks": False,
            },
            "task_overrides": {
                "narrator": {
                    "enabled": True,
                    "profile": {
                        "sort": "throughput",
                        "sort_partition": "none",
                        "only": ["deepinfra/turbo"],
                    },
                },
                "media": {
                    "enabled": True,
                    "profile": {
                        "sort": "latency",
                    },
                },
            },
        },
    )

    assert openrouter_routing_payload_for_task(
        repositories,
        provider="openrouter",
        task="chat",
    ) == {
        "only": ["deepinfra/turbo"],
        "sort": {"by": "throughput", "partition": "none"},
    }
    assert openrouter_routing_payload_for_task(
        repositories,
        provider="openrouter",
        task="summarization",
    ) == {
        "allow_fallbacks": False,
        "sort": "price",
    }
    for task in (
        "action_choice_generation",
        "character_presence_assessment",
        "character_intent_planning",
        "dating_route_profile",
        "context_cleanup_scan",
        "context_cleanup_actions",
        "guided_context_cleanup",
    ):
        assert openrouter_routing_payload_for_task(
            repositories,
            provider="openrouter",
            task=task,
        ) == {
            "allow_fallbacks": False,
            "sort": "price",
        }
    for task in (
        "scene_image_edit_generation",
        "character_image_edit_generation",
        "text_message_image_edit_generation",
    ):
        assert openrouter_routing_payload_for_task(
            repositories,
            provider="openrouter",
            task=task,
        ) == {"sort": "latency"}
    assert (
        openrouter_routing_payload_for_task(
            repositories,
            provider="venice",
            task="chat",
        )
        is None
    )


def test_request_with_openrouter_routing_sets_and_clears_provider_payload(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_app_setting(
        OPENROUTER_ROUTING_PROFILES_SETTING,
        {"global": {"data_collection": "deny"}},
    )
    request = ChatRequest(
        provider="openrouter",
        model_id="openai/gpt-5-mini",
        messages=(ChatMessage(role="player", body="Hello"),),
    )

    routed = request_with_openrouter_routing(
        repositories,
        request,
        task="chat",
    )
    cleared = request_with_openrouter_routing(
        repositories,
        ChatRequest(
            provider="venice",
            model_id="venice/uncensored",
            messages=request.messages,
            openrouter_provider_routing={"sort": "price"},
            openrouter_app_title="Bragi",
        ),
        task="chat",
    )

    assert routed.openrouter_provider_routing == {"data_collection": "deny"}
    assert routed.openrouter_app_title == "Bragi"
    assert cleared.openrouter_provider_routing is None
    assert cleared.openrouter_app_title is None


def test_request_with_openrouter_routing_applies_thinking_for_structured_output(
    repositories: PersistenceRepositories,
) -> None:
    repositories.save_provider_model(
        provider="openrouter",
        model_id="openai/gpt-5-mini",
        display_name="GPT-5 Mini",
        capabilities=["structured_output"],
        thinking={"levels": ["high", "low"], "mandatory": False},
    )
    repositories.set_app_setting(
        MODEL_THINKING_PREFERENCES_SETTING,
        {
            "context_update": {
                "provider": "openrouter",
                "model_id": "openai/gpt-5-mini",
                "level": "high",
            }
        },
    )
    request = StructuredOutputRequest(
        provider="openrouter",
        model_id="openai/gpt-5-mini",
        messages=(ChatMessage(role="player", body="Summarize facts"),),
        schema_name="facts",
        schema={
            "type": "object",
            "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
            "required": ["facts"],
        },
    )

    routed = request_with_openrouter_routing(
        repositories,
        request,
        task="context_update",
    )

    assert routed.reasoning is not None
    assert routed.reasoning.effort == "high"
    assert routed.reasoning.exclude is True


def test_request_with_openrouter_routing_skips_thinking_for_unsupported_model(
    repositories: PersistenceRepositories,
) -> None:
    repositories.save_provider_model(
        provider="openrouter",
        model_id="openai/gpt-5-mini",
        display_name="GPT-5 Mini",
        capabilities=["structured_output"],
    )
    repositories.set_app_setting(
        MODEL_THINKING_PREFERENCES_SETTING,
        {
            "context_update": {
                "provider": "openrouter",
                "model_id": "openai/gpt-5-mini",
                "level": "high",
            }
        },
    )
    request = StructuredOutputRequest(
        provider="openrouter",
        model_id="openai/gpt-5-mini",
        messages=(ChatMessage(role="player", body="Summarize facts"),),
        schema_name="facts",
        schema={
            "type": "object",
            "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
            "required": ["facts"],
        },
    )

    routed = request_with_openrouter_routing(
        repositories,
        request,
        task="context_update",
    )

    assert routed.reasoning is None


def test_request_with_openrouter_routing_applies_thinking_for_chat(
    repositories: PersistenceRepositories,
) -> None:
    repositories.save_provider_model(
        provider="openrouter",
        model_id="openai/gpt-5-mini",
        display_name="GPT-5 Mini",
        capabilities=["chat"],
        thinking={"levels": ["high", "low"], "mandatory": False},
    )
    repositories.set_app_setting(
        MODEL_THINKING_PREFERENCES_SETTING,
        {
            "chat": {
                "provider": "openrouter",
                "model_id": "openai/gpt-5-mini",
                "level": "high",
            }
        },
    )
    request = ChatRequest(
        provider="openrouter",
        model_id="openai/gpt-5-mini",
        messages=(ChatMessage(role="player", body="Continue the scene"),),
    )

    routed = request_with_openrouter_routing(
        repositories,
        request,
        task="chat",
    )

    assert routed.reasoning is not None
    assert routed.reasoning.effort == "high"
    assert routed.reasoning.exclude is True


def test_request_with_openrouter_routing_applies_thinking_for_tool_call(
    repositories: PersistenceRepositories,
) -> None:
    repositories.save_provider_model(
        provider="openrouter",
        model_id="openai/gpt-5-mini",
        display_name="GPT-5 Mini",
        capabilities=["tool_calling"],
        thinking={"levels": ["high", "low"], "mandatory": False},
    )
    repositories.set_app_setting(
        MODEL_THINKING_PREFERENCES_SETTING,
        {
            "context_search": {
                "provider": "openrouter",
                "model_id": "openai/gpt-5-mini",
                "level": "high",
            }
        },
    )
    request = ToolCallRequest(
        provider="openrouter",
        model_id="openai/gpt-5-mini",
        messages=(ToolCallMessage(role="user", body="Select context"),),
        tools=(
            ToolDefinition(
                name="select_context_source",
                description="Select a context source.",
                parameters={"type": "object", "properties": {}},
            ),
        ),
    )

    routed = request_with_openrouter_routing(
        repositories,
        request,
        task="context_search",
    )

    assert routed.reasoning is not None
    assert routed.reasoning.effort == "high"
    assert routed.reasoning.exclude is True


def test_openrouter_app_title_for_task_uses_single_app_name() -> None:
    assert openrouter_app_title_for_task("chat") == "Bragi"
    assert openrouter_app_title_for_task("npc_knowledge_audit") == "Bragi"
    assert openrouter_app_title_for_task("full_roleplay_context_update") == "Bragi"
    assert (
        openrouter_app_title_for_task("scenario_generation_section_opening_message")
        == "Bragi"
    )
