from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import ChatMessage, ChatRequest, ChatResponse
from bragi.providers.fake import FakeProviderClient
from bragi.services.runtime_telemetry import (
    runtime_telemetry_context,
    wrap_provider_clients_for_telemetry,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_provider_wrapper_records_model_backed_calls(
    repositories: PersistenceRepositories,
) -> None:
    async def run_test() -> None:
        job = repositories.create_job(
            type="chat_turn",
            status="running",
            payload={},
        )
        provider = wrap_provider_clients_for_telemetry(
            {"fake": FakeProviderClient()},
            repositories=repositories,
        )["fake"]

        with runtime_telemetry_context(
            repositories=repositories,
            job_id=job.id,
            task="chat",
        ):
            await provider.chat(
                ChatRequest(
                    provider="fake",
                    model_id="fake-chat",
                    messages=(ChatMessage(role="user", body="hello"),),
                    max_output_tokens=32,
                )
            )

        steps = repositories.list_job_steps(job.id)
        assert len(steps) == 1
        assert steps[0].name == "provider.chat"
        assert steps[0].status == "succeeded"
        assert steps[0].provider == "fake"
        assert steps[0].model == "fake-chat"
        assert steps[0].task == "chat"
        assert steps[0].duration_ms is not None
        assert steps[0].metadata["max_output_tokens"] == 32
        assert "hello" not in repr(steps[0])

    asyncio.run(run_test())


def test_provider_wrapper_records_openrouter_backend_provider_metadata(
    repositories: PersistenceRepositories,
) -> None:
    class OpenRouterMetadataProvider:
        provider_name = "openrouter"

        async def chat(self, request: ChatRequest) -> ChatResponse:
            return ChatResponse(
                body="The routed provider answers.",
                provider=request.provider,
                model_id=request.model_id,
                token_usage={"total": 9},
                raw_metadata={
                    "openrouter_metadata": {
                        "endpoints": {
                            "available": [
                                {
                                    "provider": "Together",
                                    "model": "qwen/qwen3-235b-a22b-2507",
                                    "selected": False,
                                },
                                {
                                    "provider": "DeepInfra",
                                    "model": "qwen/qwen3-235b-a22b-2507",
                                    "selected": True,
                                },
                            ],
                            "total": 2,
                        },
                        "attempts": [
                            {
                                "provider": "Together",
                                "model": "qwen/qwen3-235b-a22b-2507",
                                "status": 529,
                            },
                            {
                                "provider": "DeepInfra",
                                "model": "qwen/qwen3-235b-a22b-2507",
                                "status": 200,
                            },
                        ],
                        "summary": "available=2, selected=DeepInfra",
                    },
                    "unsafe_string": "do not persist this",
                },
            )

    async def run_test() -> None:
        job = repositories.create_job(
            type="chat_turn",
            status="running",
            payload={},
        )
        provider = wrap_provider_clients_for_telemetry(
            {"openrouter": OpenRouterMetadataProvider()},
            repositories=repositories,
        )["openrouter"]

        with runtime_telemetry_context(
            repositories=repositories,
            job_id=job.id,
            task="chat",
        ):
            await provider.chat(
                ChatRequest(
                    provider="openrouter",
                    model_id="qwen/qwen3-235b-a22b-2507",
                    messages=(ChatMessage(role="user", body="hello"),),
                )
            )

        steps = repositories.list_job_steps(job.id)
        assert len(steps) == 1
        assert steps[0].provider == "openrouter"
        assert steps[0].metadata == {
            "openrouter_available_provider_count": 2,
            "openrouter_provider_attempt_statuses": [529, 200],
            "openrouter_provider_attempts": ["Together", "DeepInfra"],
            "openrouter_selected_model": "qwen/qwen3-235b-a22b-2507",
            "openrouter_selected_provider": "DeepInfra",
            "token_total": 9,
        }

    asyncio.run(run_test())
