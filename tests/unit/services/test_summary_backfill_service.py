from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.models import SaveRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatRequest,
    ChatResponse,
    ImageRequest,
    ImageResponse,
    ProviderCapability,
    ProviderConfigStatus,
    ProviderModel,
)
from bragi.services.summary_backfill_service import SummaryBackfillService


class SequenceSummaryProvider:
    provider_name = "fake"

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
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
                model_id="fake-summary",
                display_name="Fake Summary",
                capabilities=frozenset({ProviderCapability.CHAT}),
                context_window=1024,
            )
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected summary request")
        return ChatResponse(
            body=self.responses.pop(0),
            provider=request.provider,
            model_id=request.model_id,
            token_usage={"total": 32},
        )

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        raise AssertionError("summary backfill must not generate images")


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_summary_backfill_batches_long_save_and_preserves_recent_window(
    repositories: PersistenceRepositories,
) -> None:
    save = _long_save_with_summary_preference(repositories, message_count=12)
    provider = SequenceSummaryProvider(
        [
            "Mara kept the beacon watch through the first long arc.",
            "Mara preserved the beacon watch and tracked the second arc.",
            (
                "Mara maintained the beacon watch through two arcs while the "
                "latest exchange stays in the recent chronicle."
            ),
        ]
    )
    service = SummaryBackfillService(
        repositories=repositories,
        providers={"fake": provider},
        batch_token_limit=220,
    )

    result = asyncio.run(service.backfill_save(save.id))

    summaries = repositories.list_summaries(save.id)
    assert len(summaries) == 1
    assert summaries[0].id == result.summary_id
    assert summaries[0].covers_message_start_id == "message-01"
    assert summaries[0].covers_message_end_id == "message-10"
    assert summaries[0].body == (
        "Mara maintained the beacon watch through two arcs while the latest "
        "exchange stays in the recent chronicle."
    )
    assert result.batch_count == 3
    assert result.summarized_message_count == 10
    assert result.retained_recent_message_count == 2
    assert result.archived_summary_count == 0
    assert len(provider.chat_requests) == 3
    assert provider.responses == []


def test_summary_backfill_keeps_existing_summary_and_settings_when_rejected(
    repositories: PersistenceRepositories,
) -> None:
    save = _long_save_with_summary_preference(repositories, message_count=6)
    prior_summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id="message-01",
        covers_message_end_id="message-02",
        body="Mara held the beacon watch before the storm intensified.",
        provider="fake",
        model="fake-summary",
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key="recent_player_message_window",
        value=18,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key="recent_narrator_message_window",
        value=16,
    )
    provider = SequenceSummaryProvider(
        [
            'Mara says, "What do you do next?"',
            'Mara asks, "What do you do next?"',
        ]
    )
    service = SummaryBackfillService(
        repositories=repositories,
        providers={"fake": provider},
        batch_token_limit=400,
    )

    with pytest.raises(ValueError, match="continuation"):
        asyncio.run(
            service.backfill_save(save.id, apply_recommended_windows=True)
        )

    assert repositories.list_summaries(save.id) == [prior_summary]
    assert repositories.get_effective_setting(
        "recent_player_message_window",
        save_id=save.id,
    ) == 18
    assert repositories.get_effective_setting(
        "recent_narrator_message_window",
        save_id=save.id,
    ) == 16


def test_summary_backfill_applies_recommended_windows_only_when_requested(
    repositories: PersistenceRepositories,
) -> None:
    save = _long_save_with_summary_preference(repositories, message_count=8)
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key="recent_player_message_window",
        value=18,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key="recent_narrator_message_window",
        value=14,
    )
    provider = SequenceSummaryProvider(
        [
            "Mara compacted the early beacon watch into a factual ledger.",
            "Mara preserved the beacon watch while recent turns remain live.",
        ]
    )
    service = SummaryBackfillService(
        repositories=repositories,
        providers={"fake": provider},
        batch_token_limit=400,
    )

    preview = asyncio.run(service.backfill_save(save.id))

    assert preview.recommended_player_window == 5
    assert preview.recommended_narrator_window == 5
    assert preview.applied_window_changes == {}
    assert repositories.get_effective_setting(
        "recent_player_message_window",
        save_id=save.id,
    ) == 18
    assert repositories.get_effective_setting(
        "recent_narrator_message_window",
        save_id=save.id,
    ) == 14

    second_save = _long_save_with_summary_preference(
        repositories,
        title="Second Watch",
        message_count=8,
        message_prefix="second-message",
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=second_save.id,
        key="recent_player_message_window",
        value=18,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=second_save.id,
        key="recent_narrator_message_window",
        value=14,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=second_save.id,
        key="narrator_planner_recent_player_message_window",
        value=20,
    )
    repositories.set_scoped_setting(
        scope="save",
        scope_id=second_save.id,
        key="narrator_planner_recent_narrator_message_window",
        value=15,
    )
    applied_provider = SequenceSummaryProvider(
        [
            "Mara compacted another early beacon watch into a factual ledger.",
            "Mara preserved another watch while recent turns remain live.",
        ]
    )
    applied_service = SummaryBackfillService(
        repositories=repositories,
        providers={"fake": applied_provider},
        batch_token_limit=400,
    )

    applied = asyncio.run(
        applied_service.backfill_save(
            second_save.id,
            apply_recommended_windows=True,
        )
    )

    assert applied.applied_window_changes == {
        "recent_player_message_window": 5,
        "recent_narrator_message_window": 5,
    }
    assert repositories.get_effective_setting(
        "recent_player_message_window",
        save_id=second_save.id,
    ) == 5
    assert repositories.get_effective_setting(
        "recent_narrator_message_window",
        save_id=second_save.id,
    ) == 5
    assert repositories.get_effective_setting(
        "narrator_planner_recent_player_message_window",
        save_id=second_save.id,
    ) == 20
    assert repositories.get_effective_setting(
        "narrator_planner_recent_narrator_message_window",
        save_id=second_save.id,
    ) == 15


def _long_save_with_summary_preference(
    repositories: PersistenceRepositories,
    *,
    title: str = "Long Watch",
    message_count: int,
    message_prefix: str = "message",
) -> SaveRecord:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title=title,
        premise="A beacon must stay lit through an ash storm.",
        player_role="Signal keeper",
        content={"starting_scene": "The beacon lens glows above the valley."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title=title)
    repositories.set_model_preference(
        task="summarization",
        provider="fake",
        model_id="fake-summary",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-summary",
        display_name="Fake Summary",
        capabilities=["chat"],
        context_window=1024,
    )
    for index in range(1, message_count + 1):
        role = "player" if index % 2 else "narrator"
        repositories.append_message(
            save_id=save.id,
            role=role,
            speaker_name="Mara" if role == "player" else "Narrator",
            body=(
                f"Beacon watch event {index:02d}: the ash storm pressure "
                "changes around the lens and Mara records the state."
            ),
            provider=None if role == "player" else "fake",
            model=None if role == "player" else "fake-chat",
            token_estimate=50,
            message_id=f"{message_prefix}-{index:02d}",
        )
    return save
