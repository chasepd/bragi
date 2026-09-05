from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.chat_rendering import rendered_chat_request_text
from bragi.providers.contracts import ChatRequest, ChatResponse, ProviderClient
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.services.image_prompt_service import (
    ImagePromptBrief,
    ImagePromptReference,
    ImagePromptService,
    ImagePromptSubject,
)


class ProseProvider:
    def __init__(self, *responses: str | ProviderError) -> None:
        self.responses = iter(responses)
        self.requests: list[ChatRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        response = next(self.responses)
        if isinstance(response, ProviderError):
            raise response
        return ChatResponse(
            body=response,
            provider=request.provider,
            model_id=request.model_id,
        )


@pytest.fixture
def repositories(
    tmp_path: Path, migrated_database_template: Path
) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    shutil.copy2(migrated_database_template, database_path)
    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def _service(
    repositories: PersistenceRepositories, provider: ProseProvider
) -> ImagePromptService:
    repositories.set_model_preference(
        task="image_prompt", provider="fake", model_id="drafter"
    )
    return ImagePromptService(
        repositories=repositories,
        providers={"fake": cast(ProviderClient, provider)},
    )


def test_brief_round_trips_textual_snapshot_without_live_registry() -> None:
    brief = ImagePromptBrief(
        purpose="character_attachment",
        source_moment="Mira holds a phone at arm's length beside the copper door.",
        scene_context="Soft rain glistens on the stone steps.",
        subjects=(
            ImagePromptSubject(
                character_id="mira", name="Mira", appearance="Copper curls",
                visual_notes="A pale crescent scar below the left eye", age="adult",
                current_clothing="A green wool coat", current_action="holds a phone",
                facial_expression="a restrained smile",
            ),
        ),
        references=(ImagePromptReference("mira", "Mira", "portrait-1"),),
        intent="Show the new doorway behind me.",
        style_preset="watercolor",
    )

    restored = ImagePromptBrief.from_json(json.loads(json.dumps(brief.to_json())))

    assert restored == brief
    assert ImagePromptBrief.from_json({}) == ImagePromptBrief()


@pytest.mark.parametrize("payload", [{"subjects": "Mira"}, {"source_moment": 42}])
def test_brief_rejects_nontext_snapshot_fields(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="image prompt brief"):
        ImagePromptBrief.from_json(payload)


def test_required_text_separates_clothes_and_ordered_reference_identity() -> None:
    brief = ImagePromptBrief(
        subjects=(
            ImagePromptSubject("mira", "Mira", appearance="Copper curls",
                               current_clothing="green wool coat"),
            ImagePromptSubject("oren", "Oren", appearance="Silver scales",
                               current_clothing="navy coveralls"),
        ),
        references=(
            ImagePromptReference("oren", "Oren", "portrait-2"),
            ImagePromptReference("mira", "Mira", "portrait-1"),
        ),
    )

    tail = brief.required_text()

    assert "Mira: green wool coat" in tail
    assert "Oren: navy coveralls" in tail
    assert "Attached image 1" in tail and "Attached image 2" in tail
    assert tail.index("Oren", tail.index("Attached image 1")) < tail.index(
        "Attached image 2"
    )
    assert "overrides" in tail and "reference" in tail
    assert "Copper curls" not in tail and "Silver scales" not in tail
    assert tail.count("Style preset:") == 1
    assert "infer" not in tail


def test_draft_preserves_full_profiles_and_distinct_actions_in_plain_prose(
    repositories: PersistenceRepositories,
) -> None:
    long_appearance = "Dense emerald scales with gold flecks. " * 110
    provider = ProseProvider(
        "Mira kneels to the left of a brass chest, holding its lid up. "
        "Oren stands on the right, lifting a blue glass key from the chest."
    )
    service = _service(repositories, provider)
    brief = ImagePromptBrief(
        source_moment="Mira opens the chest while Oren retrieves the key.",
        scene_context="A cold storeroom with a barred east window.",
        subjects=(
            ImagePromptSubject("mira", "Mira", current_action="holds the lid",
                               current_clothing="green coat"),
            ImagePromptSubject("oren", "Oren", appearance=long_appearance,
                               current_action="lifts the key"),
        ),
    )

    prompt = asyncio.run(service.draft(save_id="save", source_message_id="moment",
                                      brief=brief))

    assert prompt.startswith("Mira kneels to the left")
    assert prompt.endswith(brief.required_text())
    assert prompt.count("Mira: green coat") == 1
    request = provider.requests[0]
    input_text = "\n".join(message.body for message in request.messages)
    input_text += "\n".join(request.current_scene_recap)
    assert long_appearance in input_text
    assert "holds the lid" in input_text and "lifts the key" in input_text
    assert "distinct" in input_text and "one coherent frame" in input_text
    assert "pronouns" in input_text and "internal thoughts" in input_text
    assert "JSON" not in input_text
    assert request.max_output_tokens == 10_000


@pytest.mark.parametrize(
    ("purpose", "direction"),
    [
        ("scene", "complete scene"),
        ("character_reference", "simple background"),
        ("solo_character", "exactly one subject"),
        ("character_attachment", "arm's-length"),
        ("object_attachment", "requested object"),
    ],
)
def test_each_purpose_uses_the_shared_prose_drafter(
    repositories: PersistenceRepositories, purpose: str, direction: str
) -> None:
    provider = ProseProvider("A detailed visual frame.")
    service = _service(repositories, provider)

    prompt = asyncio.run(service.draft(
        save_id="save", source_message_id="moment",
        brief=ImagePromptBrief(purpose=purpose, style_preset="none"),
    ))

    assert prompt == "A detailed visual frame."
    assert direction in provider.requests[0].messages[0].body


def test_draft_reserves_required_tail_and_compresses_once(
    repositories: PersistenceRepositories,
) -> None:
    provider = ProseProvider("A very elaborate frame. " * 100, "Mira lifts the lid.")
    service = _service(repositories, provider)
    brief = ImagePromptBrief(
        subjects=(ImagePromptSubject("mira", "Mira", current_clothing="green coat"),),
        style_preset="none",
    )
    maximum = len(brief.required_text()) + 50

    prompt = asyncio.run(service.draft(
        save_id="save", source_message_id="moment", brief=brief,
        max_prompt_chars=maximum,
    ))

    assert len(prompt) <= maximum
    assert prompt == "Mira lifts the lid.\n\n" + brief.required_text()
    assert len(provider.requests) == 2
    assert "48 characters" in provider.requests[0].messages[0].body
    assert "48 characters" in provider.requests[1].messages[0].body


def test_fit_preserves_tail_when_actual_fallback_model_has_smaller_limit(
    repositories: PersistenceRepositories,
) -> None:
    provider = ProseProvider("Mira holds the lid; Oren lifts the blue key.")
    service = _service(repositories, provider)
    required = "Mira: green coat. Oren: navy coveralls."
    original = "The ornate chamber is richly detailed. " * 100 + "\n\n" + required

    prompt = asyncio.run(service.fit_prompt(
        save_id="save", source_message_id="moment", prompt=original,
        required_text=required, max_prompt_chars=120,
    ))

    assert len(prompt) <= 120
    assert prompt.endswith(required)
    assert prompt.count(required) == 1
    assert len(provider.requests) == 1


def test_compression_failure_is_explicit_and_never_silently_truncates(
    repositories: PersistenceRepositories,
) -> None:
    provider = ProseProvider("A frame described in too much detail. " * 100)
    service = _service(repositories, provider)

    with pytest.raises(ValueError, match="exceeds.*100.*after.*compression"):
        asyncio.run(service.fit_prompt(
            save_id="save", source_message_id="moment",
            prompt="A sprawling frame. " * 100, max_prompt_chars=100,
        ))
    assert len(provider.requests) == 1


def test_required_text_too_large_fails_before_request(
    repositories: PersistenceRepositories,
) -> None:
    provider = ProseProvider()
    service = _service(repositories, provider)
    with pytest.raises(ValueError, match="required.*exceeds"):
        asyncio.run(service.draft(
            save_id="save", source_message_id="moment", brief=ImagePromptBrief(),
            max_prompt_chars=20,
        ))
    assert not provider.requests


def test_no_known_limit_keeps_full_prompt_without_compression(
    repositories: PersistenceRepositories,
) -> None:
    provider = ProseProvider()
    service = _service(repositories, provider)
    original = "Detailed stonework and individual leaves. " * 500

    prompt = asyncio.run(service.fit_prompt(
        save_id="save", source_message_id="moment", prompt=original,
    ))

    assert prompt == original
    assert not provider.requests


def test_empty_image_prompt_response_continues_to_chat_preference(
    repositories: PersistenceRepositories,
) -> None:
    provider = ProseProvider("   ", "Mira stands on a sunlit stair.")
    service = _service(repositories, provider)
    repositories.set_model_preference(task="chat", provider="fake", model_id="chat")

    result = asyncio.run(service.draft(
        save_id="save", source_message_id="moment",
        brief=ImagePromptBrief(style_preset="none"),
    ))

    assert result == "Mira stands on a sunlit stair."
    assert [request.model_id for request in provider.requests] == ["drafter", "chat"]


def test_non_chat_image_model_is_skipped(
    repositories: PersistenceRepositories,
) -> None:
    provider = ProseProvider("Mira stands on a sunlit stair.")
    service = _service(repositories, provider)
    repositories.save_provider_model(
        provider="fake", model_id="drafter", display_name="Image model",
        capabilities=["image_generation"],
    )
    repositories.set_model_preference(task="chat", provider="fake", model_id="chat")

    asyncio.run(service.draft(
        save_id="save", source_message_id="moment", brief=ImagePromptBrief(),
    ))

    assert [request.model_id for request in provider.requests] == ["chat"]


def test_configured_missing_provider_fails_with_actionable_error(
    repositories: PersistenceRepositories,
) -> None:
    repositories.set_model_preference(
        task="image_prompt", provider="unconfigured", model_id="drafter"
    )
    service = ImagePromptService(repositories=repositories, providers={})
    with pytest.raises(ValueError, match="provider is unavailable: unconfigured"):
        asyncio.run(service.draft(
            save_id="save", source_message_id="moment", brief=ImagePromptBrief(),
        ))


def test_draft_keeps_long_prose_when_no_model_limit_is_known(
    repositories: PersistenceRepositories,
) -> None:
    prose = "The brass chest has delicate etched concentric rings. " * 100
    provider = ProseProvider(prose)
    service = _service(repositories, provider)

    result = asyncio.run(service.draft(
        save_id="save", source_message_id="moment",
        brief=ImagePromptBrief(style_preset="none"),
    ))

    assert result == prose.strip()
    assert len(result) > 2400
    assert len(provider.requests) == 1


def test_context_budget_preserves_source_and_full_subject_before_older_context(
    repositories: PersistenceRepositories,
) -> None:
    provider = ProseProvider("Mira holds the chest lid.")
    service = _service(repositories, provider)
    repositories.save_provider_model(
        provider="fake", model_id="drafter", display_name="Draft model",
        capabilities=["chat"], context_window=14000,
    )
    appearance = "Copper curls with pale gold streaks. " * 30 + "A crescent scar."
    brief = ImagePromptBrief(
        source_moment="Mira holds the chest lid with her right palm.",
        subjects=(ImagePromptSubject("mira", "Mira", appearance=appearance),),
        scene_context=(
            "The chest is on the bench beside the barred window.\n"
            + "Older background: the stone walls have faded paint.\n" * 1200
            + "The oldest scene is set at a railway platform."
        ),
    )

    asyncio.run(service.draft(
        save_id="save", source_message_id="moment", brief=brief,
    ))

    text = rendered_chat_request_text(provider.requests[0])
    assert brief.source_moment in text
    assert appearance in text
    assert "chest is on the bench beside the barred window" in text
    assert "oldest scene" not in text


def test_drafter_uses_configured_chat_fallback_on_provider_error(
    repositories: PersistenceRepositories,
) -> None:
    provider = ProseProvider(
        ProviderError(category=ProviderErrorCategory.NETWORK_ERROR, message="Offline"),
        "Mira braces the lid with her right palm.",
    )
    service = _service(repositories, provider)
    repositories.set_model_preference(
        task="chat_fallback", provider="fake", model_id="backup"
    )
    repositories.save_provider_model(
        provider="fake", model_id="backup", display_name="Backup",
        capabilities=["chat"],
    )

    result = asyncio.run(service.draft(
        save_id="save", source_message_id="moment",
        brief=ImagePromptBrief(style_preset="none"),
    ))

    assert result == "Mira braces the lid with her right palm."
    assert [request.model_id for request in provider.requests] == ["drafter", "backup"]


def test_context_budget_drops_an_oversized_fact_instead_of_cutting_its_meaning(
    repositories: PersistenceRepositories,
) -> None:
    provider = ProseProvider("Mira holds the chest lid.")
    service = _service(repositories, provider)
    repositories.save_provider_model(
        provider="fake", model_id="drafter", display_name="Draft model",
        capabilities=["chat"], context_window=14000,
    )
    brief = ImagePromptBrief(
        source_moment="Mira holds the chest lid.",
        scene_context="A supporting fact with a long qualification: " * 2000,
    )

    asyncio.run(service.draft(
        save_id="save", source_message_id="moment", brief=brief,
    ))

    assert not provider.requests[0].current_scene_recap


def test_empty_compression_does_not_start_a_second_compression_attempt(
    repositories: PersistenceRepositories,
) -> None:
    provider = ProseProvider(" ")
    service = _service(repositories, provider)
    repositories.set_model_preference(task="chat", provider="fake", model_id="chat")

    with pytest.raises(ValueError, match="empty output"):
        asyncio.run(service.fit_prompt(
            save_id="save", source_message_id="moment",
            prompt="A detailed room. " * 100, max_prompt_chars=80,
        ))

    assert len(provider.requests) == 1


def test_fit_rejects_missing_authoritative_suffix_without_rewriting(
    repositories: PersistenceRepositories,
) -> None:
    provider = ProseProvider()
    service = _service(repositories, provider)

    with pytest.raises(ValueError, match="required text suffix"):
        asyncio.run(service.fit_prompt(
            save_id="save", source_message_id="moment",
            prompt="A detailed room. " * 100,
            required_text="Mira: green coat", max_prompt_chars=80,
        ))

    assert not provider.requests


def test_character_attachment_framing_follows_intent_instead_of_forcing_selfie(
    repositories: PersistenceRepositories,
) -> None:
    provider = ProseProvider("Mira holds a large brass clock with both hands.")
    service = _service(repositories, provider)
    brief = ImagePromptBrief(
        purpose="character_attachment",
        intent="A full-length mirror outfit picture; both hands hold the clock.",
        subjects=(ImagePromptSubject(
            character_id="mira", name="Mira",
            current_action="holding a large brass clock with both hands",
        ),),
        style_preset="none",
    )

    prompt = asyncio.run(service.draft(
        save_id="save", source_message_id="moment", brief=brief,
    ))

    context = rendered_chat_request_text(provider.requests[0])
    assert brief.intent in context
    assert "both hands" in context
    assert "When it asks for a selfie" in context
    assert "need not be a selfie" in context
    assert prompt == "Mira holds a large brass clock with both hands."


@pytest.mark.parametrize("allowed", [True, False])
def test_brief_preserves_captured_clothing_completion_eligibility(
    allowed: bool,
) -> None:
    brief = ImagePromptBrief(subjects=(ImagePromptSubject(
        "mira", "Mira", clothing_completion_allowed=allowed,
    ),))

    restored = ImagePromptBrief.from_json(json.loads(json.dumps(brief.to_json())))

    assert restored == brief
    assert restored.subjects[0].clothing_completion_allowed is allowed


def test_legacy_brief_defaults_to_allowing_missing_clothing_completion() -> None:
    restored = ImagePromptBrief.from_json({
        "subjects": [{"character_id": "mira", "name": "Mira"}],
    })

    assert restored.subjects[0].clothing_completion_allowed is True


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_brief_rejects_nonboolean_clothing_completion_eligibility(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="clothing_completion_allowed.*boolean"):
        ImagePromptBrief.from_json({
            "subjects": [{
                "character_id": "mira", "name": "Mira",
                "clothing_completion_allowed": value,
            }],
        })
