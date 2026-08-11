from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from bragi.content_rating_instructions import content_rating_exceeds
from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatRequest,
    ProviderClient,
    StructuredOutputRequest,
    StructuredOutputResponse,
)
from bragi.services.content_safety_service import (
    ContentSafetyAction,
    ContentSafetyService,
)
from bragi.services.sexual_content_safety import (
    CONTENT_FILTER_TRANSITION,
    FADE_TO_BLACK_TRANSITION,
)


class ScriptedSafetyProvider:
    provider_name = "fake"

    def __init__(
        self,
        action: str,
        *,
        minimum_rating: str | None = None,
        category: str = "sexual_content",
    ) -> None:
        self.action = action
        self.category = category
        self.minimum_rating = minimum_rating or (
            "r" if action in {"block", "fade_to_black"} else "g"
        )
        self.structured_output_requests: list[StructuredOutputRequest] = []

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_output_requests.append(request)
        return StructuredOutputResponse(
            data={
                "action": self.action,
                "category": self.category,
                "reason": "The narration crosses the configured ceiling.",
                "minimum_rating": self.minimum_rating,
            },
            provider=request.provider,
            model_id=request.model_id,
        )


class ScriptedBatchSafetyProvider:
    provider_name = "fake"

    def __init__(self, reviews: list[dict[str, object]]) -> None:
        self.reviews = reviews
        self.structured_output_requests: list[StructuredOutputRequest] = []

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.structured_output_requests.append(request)
        return StructuredOutputResponse(
            data={"reviews": self.reviews},
            provider=request.provider,
            model_id=request.model_id,
        )


def test_agent_rating_comparison_fails_closed_for_unclassified_legacy_content() -> None:
    assert content_rating_exceeds(
        minimum_rating="unclassified",
        allowed_rating="g",
    )
    assert content_rating_exceeds(
        minimum_rating="unrated",
        allowed_rating="r",
    )
    assert content_rating_exceeds(
        minimum_rating="malformed-import-value",
        allowed_rating="r",
    )
    assert content_rating_exceeds(
        minimum_rating="r",
        allowed_rating="pg-13",
    )


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def _configure_safety_model(repositories: PersistenceRepositories) -> None:
    repositories.set_model_preference(
        task="content_safety",
        provider="fake",
        model_id="fake-safety",
    )
    repositories.save_provider_model(
        provider="fake",
        model_id="fake-safety",
        display_name="Fake Safety",
        capabilities=["structured_output"],
    )


def test_review_uses_typed_output_and_rating_specific_system_instructions(
    repositories: PersistenceRepositories,
) -> None:
    _configure_safety_model(repositories)
    provider = ScriptedSafetyProvider("block")
    service = ContentSafetyService(
        repositories=repositories,
        providers={"fake": cast(ProviderClient, provider)},
    )

    result = asyncio.run(
        service.review_narration(
            body="A frightening draft.",
            content_rating="g",
            fade_to_black_enabled=True,
        )
    )

    assert result.action is ContentSafetyAction.BLOCK
    assert result.body == CONTENT_FILTER_TRANSITION
    assert result.minimum_rating == "r"
    assert result.agent_ran is True
    assert len(provider.structured_output_requests) == 1
    request = provider.structured_output_requests[0]
    assert request.schema_name == "content_safety_review"
    assert request.schema["properties"]["action"]["enum"] == [
        "allow",
        "block",
        "fade_to_black",
    ]
    system_message = request.messages[0]
    assert system_message.role == "system"
    assert "G — General audiences" in system_message.body
    assert "Any profanity beyond extremely mild exclamations" in system_message.body
    assert "safe for a young child without parental explanation" in (
        system_message.body.lower()
    )
    assert request.messages[-1].body == "A frightening draft."


@pytest.mark.parametrize(
    ("rating", "distinctive_instruction"),
    (
        ("g", "Any profanity beyond extremely mild exclamations"),
        ("pg", "Strong profanity or slurs"),
        ("pg-13", "Explicitly described sexual activity"),
        ("r", "Pornographic or explicitly erotic depictions"),
    ),
)
def test_each_rated_ceiling_has_distinct_safety_agent_instructions(
    repositories: PersistenceRepositories,
    rating: str,
    distinctive_instruction: str,
) -> None:
    _configure_safety_model(repositories)
    provider = ScriptedSafetyProvider("allow")
    service = ContentSafetyService(
        repositories=repositories,
        providers={"fake": cast(ProviderClient, provider)},
    )

    result = asyncio.run(
        service.review_narration(
            body="The reviewed narration.",
            content_rating=rating,
            fade_to_black_enabled=True,
        )
    )

    assert result.body == "The reviewed narration."
    assert distinctive_instruction in provider.structured_output_requests[0].messages[
        0
    ].body


def test_fade_to_black_is_an_agent_decision(
    repositories: PersistenceRepositories,
) -> None:
    _configure_safety_model(repositories)
    provider = ScriptedSafetyProvider("fade_to_black")
    service = ContentSafetyService(
        repositories=repositories,
        providers={"fake": cast(ProviderClient, provider)},
    )

    result = asyncio.run(
        service.review_narration(
            body="The intimacy escalates beyond the selected ceiling.",
            content_rating="pg-13",
            fade_to_black_enabled=True,
        )
    )

    assert result.action is ContentSafetyAction.FADE_TO_BLACK
    assert result.body == FADE_TO_BLACK_TRANSITION
    assert result.transition_applied is True


def test_disabled_fade_converts_an_invalid_fade_decision_to_neutral_block(
    repositories: PersistenceRepositories,
) -> None:
    _configure_safety_model(repositories)
    provider = ScriptedSafetyProvider("fade_to_black")
    service = ContentSafetyService(
        repositories=repositories,
        providers={"fake": cast(ProviderClient, provider)},
    )

    result = asyncio.run(
        service.review_narration(
            body="The intimacy escalates beyond the selected ceiling.",
            content_rating="pg",
            fade_to_black_enabled=False,
        )
    )

    assert result.action is ContentSafetyAction.BLOCK
    assert result.body == CONTENT_FILTER_TRANSITION
    assert "Fade-to-black is disabled" in (
        provider.structured_output_requests[0].messages[0].body
    )


def test_fade_decision_is_limited_to_sexual_or_romantic_escalation(
    repositories: PersistenceRepositories,
) -> None:
    _configure_safety_model(repositories)
    provider = ScriptedSafetyProvider("fade_to_black", category="violence")
    service = ContentSafetyService(
        repositories=repositories,
        providers={"fake": cast(ProviderClient, provider)},
    )

    result = asyncio.run(
        service.review_narration(
            body="A violent draft.",
            content_rating="pg",
            fade_to_black_enabled=True,
        )
    )

    assert result.action is ContentSafetyAction.BLOCK
    assert result.body == CONTENT_FILTER_TRANSITION


def test_allow_decision_cannot_exceed_the_selected_ceiling(
    repositories: PersistenceRepositories,
) -> None:
    _configure_safety_model(repositories)
    provider = ScriptedSafetyProvider("allow", minimum_rating="r")
    service = ContentSafetyService(
        repositories=repositories,
        providers={"fake": cast(ProviderClient, provider)},
    )

    result = asyncio.run(
        service.review_narration(
            body="A mislabeled draft.",
            content_rating="pg",
            fade_to_black_enabled=True,
        )
    )

    assert result.action is ContentSafetyAction.BLOCK
    assert result.body == CONTENT_FILTER_TRANSITION


def test_unrated_bypasses_safety_agent_and_never_fades(
    repositories: PersistenceRepositories,
) -> None:
    _configure_safety_model(repositories)
    provider = ScriptedSafetyProvider("fade_to_black")
    service = ContentSafetyService(
        repositories=repositories,
        providers={"fake": cast(ProviderClient, provider)},
    )
    body = "The original narration remains untouched."

    result = asyncio.run(
        service.review_narration(
            body=body,
            content_rating="unrated",
            fade_to_black_enabled=True,
        )
    )

    assert result.action is ContentSafetyAction.ALLOW
    assert result.body == body
    assert result.agent_ran is False
    assert result.transition_applied is False
    assert provider.structured_output_requests == []


def test_batch_review_uses_one_bounded_request_and_preserves_ordinal_results(
    repositories: PersistenceRepositories,
) -> None:
    _configure_safety_model(repositories)
    provider = ScriptedBatchSafetyProvider(
        [
            {
                "ordinal": 2,
                "action": "block",
                "category": "violence",
                "reason": "Too violent.",
                "minimum_rating": "r",
            },
            {
                "ordinal": 1,
                "action": "allow",
                "category": "none",
                "reason": "Suitable.",
                "minimum_rating": "g",
            },
        ]
    )
    service = ContentSafetyService(
        repositories=repositories,
        providers={"fake": cast(ProviderClient, provider)},
    )

    results = asyncio.run(
        service.review_narrations(
            bodies=("Open the door.", "Strike the guard."),
            content_rating="pg",
            fade_to_black_enabled=False,
        )
    )

    assert [result.body for result in results] == [
        "Open the door.",
        CONTENT_FILTER_TRANSITION,
    ]
    assert len(provider.structured_output_requests) == 1
    request = provider.structured_output_requests[0]
    assert request.schema_name == "content_safety_batch_review"
    reviews_schema = request.schema["properties"]["reviews"]
    assert reviews_schema["minItems"] == 2
    assert reviews_schema["maxItems"] == 2
    assert "Draft 1:\nOpen the door." in request.messages[-1].body
    assert "Draft 2:\nStrike the guard." in request.messages[-1].body


def test_batch_review_fails_closed_for_duplicate_ordinals(
    repositories: PersistenceRepositories,
) -> None:
    _configure_safety_model(repositories)
    provider = ScriptedBatchSafetyProvider(
        [
            {
                "ordinal": 1,
                "action": "allow",
                "category": "none",
                "reason": "Suitable.",
                "minimum_rating": "g",
            },
            {
                "ordinal": 1,
                "action": "allow",
                "category": "none",
                "reason": "Suitable.",
                "minimum_rating": "g",
            },
        ]
    )
    service = ContentSafetyService(
        repositories=repositories,
        providers={"fake": cast(ProviderClient, provider)},
    )

    results = asyncio.run(
        service.review_narrations(
            bodies=("First draft.", "Second draft."),
            content_rating="pg",
            fade_to_black_enabled=False,
        )
    )

    assert [result.body for result in results] == [CONTENT_FILTER_TRANSITION] * 2
    assert all(result.category == "safety_agent_error" for result in results)


def test_rated_review_fails_closed_without_a_safety_model(
    repositories: PersistenceRepositories,
) -> None:
    service = ContentSafetyService(repositories=repositories, providers={})

    result = asyncio.run(
        service.review_narration(
            body="An unreviewed rated draft.",
            content_rating="pg",
            fade_to_black_enabled=True,
        )
    )

    assert result.action is ContentSafetyAction.BLOCK
    assert result.body == CONTENT_FILTER_TRANSITION
    assert result.minimum_rating == "g"
    assert result.transition_applied is True
    assert result.agent_ran is False
    assert result.skipped_reason == "no_model_preference"


@pytest.mark.parametrize(
    ("providers", "skipped_reason"),
    (
        ({}, "provider_unavailable"),
        ({"fake": object()}, "provider_lacks_structured_output"),
    ),
)
def test_rated_review_fails_closed_without_a_usable_structured_provider(
    repositories: PersistenceRepositories,
    providers: dict[str, object],
    skipped_reason: str,
) -> None:
    _configure_safety_model(repositories)
    service = ContentSafetyService(
        repositories=repositories,
        providers=cast(dict[str, ProviderClient], providers),
    )

    result = asyncio.run(
        service.review_narration(
            body="An unreviewed rated draft.",
            content_rating="pg-13",
            fade_to_black_enabled=True,
        )
    )

    assert result.action is ContentSafetyAction.BLOCK
    assert result.body == CONTENT_FILTER_TRANSITION
    assert result.minimum_rating == "g"
    assert result.transition_applied is True
    assert result.agent_ran is False
    assert result.skipped_reason == skipped_reason


def test_review_context_does_not_ask_a_chat_model_for_structured_text(
    repositories: PersistenceRepositories,
) -> None:
    _configure_safety_model(repositories)
    provider = ScriptedSafetyProvider("allow")
    service = ContentSafetyService(
        repositories=repositories,
        providers={"fake": cast(ProviderClient, provider)},
    )

    asyncio.run(
        service.review_narration(
            body="The reviewed narration.",
            content_rating="pg-13",
            fade_to_black_enabled=True,
            source_request=ChatRequest(
                provider="fake",
                model_id="fake-chat",
                messages=(),
            ),
        )
    )

    assert provider.structured_output_requests


def test_implicit_safety_model_rejects_over_budget_before_dispatch(
    repositories: PersistenceRepositories,
) -> None:
    repositories.save_provider_model(
        provider="fake",
        model_id="tiny",
        display_name="Tiny",
        capabilities=["chat", "structured_output"],
        context_window=256,
    )
    provider = ScriptedSafetyProvider("allow")
    service = ContentSafetyService(
        repositories=repositories,
        providers={"fake": cast(ProviderClient, provider)},
    )

    result = asyncio.run(
        service.review_narration(
            body="Large rated draft. " * 200,
            content_rating="pg-13",
            fade_to_black_enabled=True,
            source_request=ChatRequest(
                provider="fake",
                model_id="tiny",
                messages=(),
            ),
        )
    )

    assert result.action is ContentSafetyAction.BLOCK
    assert result.category == "safety_agent_error"
    assert provider.structured_output_requests == []
