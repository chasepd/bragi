from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import cast

import pytest

import bragi.services.character_profile_completion as completion
from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ProviderToolCall,
    StructuredOutputRequest,
    StructuredOutputResponse,
    ToolCallRequest,
    ToolCallResponse,
)
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.services.character_profile_completion import (
    CHARACTER_STARTERS_CONTENT_KEY,
    CharacterFieldEnhancementRequest,
    CharacterProfileCompletionRequest,
    CharacterStarterGenerationRequest,
    ScenarioCharacterStarter,
    ScenarioStarterReferenceImage,
    StructuredProviderCharacterProfileCompleter,
    ToolCallingProviderCharacterProfileCompleter,
    content_with_character_starters,
    normalize_scenario_character_starters,
    scenario_character_starter_to_json,
    scenario_character_starters_for_content,
)
from bragi.services.phrase_denylist import SAVE_GENERATED_PHRASE_DENYLIST_SETTING


class RecordingStructuredProfileProvider:
    provider_name = "fake"

    def __init__(
        self,
        data: dict[str, object] | list[dict[str, object]],
    ) -> None:
        self.responses = list(data) if isinstance(data, list) else [data]
        self.requests: list[StructuredOutputRequest] = []

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected structured-output request")
        return StructuredOutputResponse(
            data=self.responses.pop(0),
            provider=request.provider,
            model_id=request.model_id,
        )


class SequencedStructuredProfileProvider:
    provider_name = "fake"

    def __init__(
        self,
        responses: list[dict[str, object] | ProviderError],
    ) -> None:
        self.responses = responses
        self.requests: list[StructuredOutputRequest] = []

    async def generate_structured_output(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected structured-output request")
        response = self.responses.pop(0)
        if isinstance(response, ProviderError):
            raise response
        return StructuredOutputResponse(
            data=response,
            provider=request.provider,
            model_id=request.model_id,
        )


class RecordingToolCallProfileProvider:
    provider_name = "fake"

    def __init__(self, responses: list[tuple[ProviderToolCall, ...]]) -> None:
        self.responses = responses
        self.requests: list[ToolCallRequest] = []

    async def generate_tool_calls(
        self,
        request: ToolCallRequest,
    ) -> ToolCallResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected tool-call request")
        return ToolCallResponse(
            tool_calls=self.responses.pop(0),
            body="",
            provider=request.provider,
            model_id=request.model_id,
        )


def _repositories(tmp_path: Path) -> PersistenceRepositories:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)
    return PersistenceRepositories(sqlite3.connect(database_path))


def test_retired_character_interaction_type_does_not_infer_character_starter() -> None:
    starters = scenario_character_starters_for_content(
        scenario_type="character_interaction",
        content={
            "character_name": "Archivist Len",
            "character_description": "A gate scholar hiding a broken oath.",
            "character_physical_description": (
                "Ink-stained gloves and silver spectacles."
            ),
            "character_personality": "Precise, guarded, curious.",
            "character_voice": "Soft questions with hard edges.",
            "relationship_seed": "Len owes Mara a dangerous favor.",
            "player_character_name": "Mara Voss",
            "premise": "A sealed archive wakes up.",
        },
    )

    assert starters == ()


def test_deprecated_prose_fields_do_not_infer_character_starters() -> None:
    content = {
        "player_character_name": "Mara Voss",
        "characters": "Captain Ilyra, Vey the outrider, and Brother Senn.",
        "romance_options": "Mika Arai - student council president.",
        "crew_and_command": "Commander Reyes - cautious mission commander.",
        "major_npcs": "Duchess Salen - regent who needs Mara's vote.",
    }

    assert scenario_character_starters_for_content(
        scenario_type="full_roleplay",
        content=content,
    ) == ()
    assert scenario_character_starters_for_content(
        scenario_type="dating_sim",
        content=content,
    ) == ()
    assert scenario_character_starters_for_content(
        scenario_type="political_intrigue",
        content=content,
    ) == ()


def test_explicit_character_starters_preserve_duplicate_first_names() -> None:
    starters = scenario_character_starters_for_content(
        scenario_type="dating_sim",
        content={
            CHARACTER_STARTERS_CONTENT_KEY: [
                {"name": "Emily Carter"},
                {"name": "Emily Brooks"},
            ],
        },
    )

    assert [starter.name for starter in starters] == [
        "Emily Carter",
        "Emily Brooks",
    ]


def test_structured_starter_generation_honors_count_and_context() -> None:
    provider = RecordingStructuredProfileProvider(
        {
            "characters": [
                {
                    "name": "Emily Carter",
                    "role": "Elementary school teacher with a nurturing heart.",
                    "age": "28",
                    "known_state": "Emily lights up when James mentions design.",
                    "appearance": "Warm brown eyes and freckles across her nose.",
                    "visual_notes": "Soft smile and relaxed cardigans.",
                    "personality": "Warm, attentive, and quietly creative.",
                    "voice": "Soft, lilting, and curious.",
                    "goals": "Find a partner who values her creative life.",
                    "motivations": "Share warmth without losing her independence.",
                    "current_intent": "Test whether the player respects her time.",
                    "boundaries": "Will not rush intimacy before trust is earned.",
                    "attitude_toward_player": (
                        "Openly kind but not automatically trusting."
                    ),
                    "cooperation_conditions": (
                        "Cooperates when the player is direct and respectful."
                    ),
                },
                {
                    "name": "Lily Chen",
                    "role": "Librarian with razor-sharp wit.",
                    "age": "29",
                    "known_state": "Lily is at speed dating on a dare.",
                    "appearance": "Dark hair, glasses, sleeve tattoos, cardigan.",
                    "visual_notes": "Vintage cardigan against bold tattoos.",
                    "personality": "Cynical, dry, and secretly hopeful.",
                    "voice": "Low, dry, and measured.",
                    "goals": "Find someone who enjoys her wit without dismissing it.",
                    "motivations": "Test whether hope is worth the risk.",
                    "current_intent": (
                        "Push back before deciding whether the player is worth time."
                    ),
                    "boundaries": (
                        "Will not tolerate condescension about her guardedness."
                    ),
                    "attitude_toward_player": (
                        "Guarded, skeptical, and willing to be unfair at first."
                    ),
                    "cooperation_conditions": (
                        "Helps only after the player treats her skepticism seriously."
                    ),
                },
            ]
        }
    )
    completer = StructuredProviderCharacterProfileCompleter(
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
    )

    starters = asyncio.run(
        completer.generate_starters(
            CharacterStarterGenerationRequest(
                scenario_type="dating_sim",
                scenario_types=("dating_sim", "heist_infiltration"),
                scenario_context=(
                    "Title: The Speed of Love\n"
                    "Premise: A speed dating night starts with a power outage."
                ),
                content={
                    "title": "The Speed of Love",
                    "player_character_name": "James Mitchell",
                    "premise": "A speed dating night starts with a power outage.",
                },
                existing_starters=(ScenarioCharacterStarter(name="Mika Arai"),),
                count=2,
                name_candidate_context=(
                    "Ordinary contemporary name candidates for new character "
                    "starters:\nFeminine: Emily, Lily\nMasculine: Noah\nNeutral: "
                    "Avery"
                ),
            )
        )
    )

    assert [starter.name for starter in starters] == ["Emily Carter", "Lily Chen"]
    request = provider.requests[0]
    assert request.schema_name == "scenario_character_starters"
    assert request.max_output_tokens == 2048
    assert "Create exactly 2 new character starters" in request.messages[0].body
    request_body = request.messages[1].body
    assert "Scenario types: dating_sim, heist_infiltration" in request_body
    assert "James Mitchell" in request_body
    assert "Mika Arai" in request_body
    assert "Ordinary contemporary name candidates" in request_body
    assert "A speed dating night starts with a power outage." in request_body
    assert "James Mitchell" not in {starter.name for starter in starters}
    assert [starter.age for starter in starters] == ["28", "29"]
    assert starters[1].attitude_toward_player == (
        "Guarded, skeptical, and willing to be unfair at first."
    )
    assert starters[1].cooperation_conditions == (
        "Helps only after the player treats her skepticism seriously."
    )
    character_schema = request.schema["properties"]["characters"]
    assert isinstance(character_schema, dict)
    assert character_schema["minItems"] == 2
    assert character_schema["maxItems"] == 2
    item_schema = character_schema["items"]
    assert isinstance(item_schema, dict)
    item_properties = item_schema["properties"]
    assert isinstance(item_properties, dict)
    assert set(item_schema["required"]) == set(item_properties)
    assert {
        "goals",
        "motivations",
        "current_intent",
        "boundaries",
        "attitude_toward_player",
        "cooperation_conditions",
    } <= set(item_properties)
    system_body = request.messages[0].body
    assert "full spectrum" in system_body
    assert "hostile" in system_body
    assert "unreasonable" in system_body
    assert len(provider.requests) == 1


def test_structured_starter_generation_scales_budget_to_maximum_count() -> None:
    names = (
        "Avery Example",
        "Blake Example",
        "Casey Example",
        "Drew Example",
        "Emery Example",
        "Flynn Example",
        "Gray Example",
        "Harper Example",
        "Indigo Example",
        "Jules Example",
        "Kai Example",
        "Lane Example",
    )
    provider = RecordingStructuredProfileProvider(
        {
            "characters": [
                {"name": name}
                for name in names
            ]
        }
    )
    completer = StructuredProviderCharacterProfileCompleter(
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
    )

    starters = asyncio.run(
        completer.generate_starters(
            CharacterStarterGenerationRequest(
                scenario_type="full_roleplay",
                scenario_context="Title: The Twelve",
                content={"player_character_name": "Mara Voss"},
                count=12,
            )
        )
    )

    assert len(starters) == 12
    request = provider.requests[0]
    assert request.max_output_tokens == 12_288
    characters_schema = request.schema["properties"]["characters"]
    assert isinstance(characters_schema, dict)
    assert characters_schema["minItems"] == 12
    assert characters_schema["maxItems"] == 12


def test_structured_starter_generation_honors_custom_description() -> None:
    provider = RecordingStructuredProfileProvider(
        {
            "characters": [
                {
                    "name": "Avery Quinn",
                    "role": "Emergency lighting technician.",
                    "known_state": "Avery restores light during the blackout.",
                    "appearance": "Tool belt, cropped hair, and a bright vest.",
                    "visual_notes": "Portable work lamp and grease-smudged hands.",
                    "personality": "Practical, warm, and quick to act.",
                    "voice": 'Plainspoken. Example: "Hold this while I reset it."',
                    "texting_style": (
                        "Brief logistics with dry humor. Sample text: Found it."
                    ),
                    "goals": "Keep everyone safe until the lights return.",
                    "motivations": "Prove competence under pressure.",
                    "boundaries": "Will not ignore a safety hazard for romance.",
                }
            ]
        }
    )
    completer = StructuredProviderCharacterProfileCompleter(
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
    )

    starters = asyncio.run(
        completer.generate_starters(
            CharacterStarterGenerationRequest(
                scenario_type="dating_sim",
                scenario_context="Title: Blackout Bell",
                content={
                    "player_character_name": "James Mitchell",
                },
                custom_description=(
                    "An emergency lighting technician who meets the player "
                    "during a blackout."
                ),
            )
        )
    )

    request_body = provider.requests[0].messages[1].body
    assert (
        "Create exactly one character starter"
        in provider.requests[0].messages[0].body
    )
    assert "Custom character description" in request_body
    assert "emergency lighting technician" in request_body
    assert [starter.name for starter in starters] == ["Avery Quinn"]
    assert len(provider.requests) == 1


def test_structured_starter_generation_retries_duplicate_names() -> None:
    provider = RecordingStructuredProfileProvider(
        [
            {"characters": [{"name": "James Mitchell", "role": "Player copy."}]},
            {"characters": [{"name": "Avery Quinn", "role": "Gallery docent."}]},
        ]
    )
    completer = StructuredProviderCharacterProfileCompleter(
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
    )

    starters = asyncio.run(
        completer.generate_starters(
            CharacterStarterGenerationRequest(
                scenario_type="dating_sim",
                scenario_context="Title: Gallery Night",
                content={
                    "player_character_name": "James Mitchell",
                },
                existing_starters=(ScenarioCharacterStarter(name="Mika Arai"),),
                count=1,
            )
        )
    )

    assert [starter.name for starter in starters] == ["Avery Quinn"]
    assert len(provider.requests) == 2
    assert "duplicates the player or existing starter" in (
        provider.requests[1].messages[-1].body
    )


def test_structured_starter_generation_retries_provider_schema_violation() -> None:
    provider = SequencedStructuredProfileProvider(
        [
            ProviderError(
                ProviderErrorCategory.STRUCTURED_OUTPUT_INVALID,
                "Structured provider response violated its JSON Schema",
            ),
            {"characters": [{"name": "Avery Quinn", "role": "Gallery docent."}]},
        ]
    )
    completer = StructuredProviderCharacterProfileCompleter(
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
    )

    starters = asyncio.run(
        completer.generate_starters(
            CharacterStarterGenerationRequest(
                scenario_type="full_roleplay",
                scenario_context="Title: Gallery Night",
                content={"player_character_name": "James Mitchell"},
                count=1,
            )
        )
    )

    assert [starter.name for starter in starters] == ["Avery Quinn"]
    assert len(provider.requests) == 2
    assert "violated its JSON Schema" in provider.requests[1].messages[-1].body


def test_structured_starter_generation_bounds_provider_schema_retries() -> None:
    provider = SequencedStructuredProfileProvider(
        [
            ProviderError(
                ProviderErrorCategory.STRUCTURED_OUTPUT_INVALID,
                "Structured provider response violated its JSON Schema",
            )
            for _attempt in range(3)
        ]
    )
    completer = StructuredProviderCharacterProfileCompleter(
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
    )

    with pytest.raises(ProviderError) as captured:
        asyncio.run(
            completer.generate_starters(
                CharacterStarterGenerationRequest(
                    scenario_type="full_roleplay",
                    scenario_context="Title: Gallery Night",
                    content={"player_character_name": "James Mitchell"},
                    count=1,
                )
            )
        )

    assert (
        captured.value.category
        == ProviderErrorCategory.STRUCTURED_OUTPUT_INVALID
    )
    assert len(provider.requests) == 3


def test_tool_starter_generation_is_disabled() -> None:
    provider = RecordingToolCallProfileProvider([])
    completer = ToolCallingProviderCharacterProfileCompleter(
        provider=provider,
        provider_name="fake",
        model_id="fake-tools",
    )

    starters = asyncio.run(
        completer.generate_starters(
            CharacterStarterGenerationRequest(
                scenario_type="dating_sim",
                scenario_context="Title: Gallery Night",
                content={
                    "player_character_name": "James Mitchell",
                },
                count=1,
            )
        )
    )

    assert starters == ()
    assert provider.requests == []


def test_structured_dating_starter_generation_retries_denied_texting_style(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="The Speed of Love",
        premise="James meets several romance options.",
        player_role="Designer",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="First Bell")
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
        value="that's not nothing",
    )
    provider = RecordingStructuredProfileProvider(
        [
            {
                "characters": [
                    {
                        "name": "Emily",
                        "role": "Elementary school teacher.",
                        "texting_style": (
                            "Warm check-ins. Sample text: That's not nothing."
                        ),
                    }
                ]
            },
            {
                "characters": [
                    {
                        "name": "Emily",
                        "role": "Elementary school teacher.",
                        "texting_style": (
                            "Warm check-ins. Sample text: Tell me when you arrive."
                        ),
                    }
                ]
            },
        ]
    )
    completer = StructuredProviderCharacterProfileCompleter(
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
        repositories=repositories,
    )

    starters = asyncio.run(
        completer.generate_starters(
            CharacterStarterGenerationRequest(
                scenario_type="dating_sim",
                scenario_context="Emily is a teacher waiting near the gallery.",
                content={
                    "player_character_name": "James Mitchell",
                },
                count=1,
                save_id=save.id,
            )
        )
    )

    assert len(provider.requests) == 2
    assert "That's not nothing" in provider.requests[1].messages[-1].body
    assert starters[0].name == "Emily"
    assert starters[0].texting_style == (
        "Warm check-ins. Sample text: Tell me when you arrive."
    )


def test_content_with_character_starters_prefers_existing_explicit_payload() -> None:
    content = content_with_character_starters(
        scenario_type="full_roleplay",
        content={
            CHARACTER_STARTERS_CONTENT_KEY: [
                {
                    "name": "Custom Ilyra",
                    "role": "Manually reviewed role.",
                    "met": False,
                }
            ],
        },
    )

    assert content[CHARACTER_STARTERS_CONTENT_KEY] == [
        {
            "name": "Custom Ilyra",
            "aliases": [],
            "role": "Manually reviewed role.",
            "age": "",
            "known_state": "",
            "appearance": "",
            "visual_notes": "",
            "personality": "",
            "voice": "",
            "texting_style": "",
            "relationships": {},
            "goals": "",
            "motivations": "",
            "current_intent": "",
            "boundaries": "",
            "attitude_toward_player": "",
            "cooperation_conditions": "",
            "status": "",
            "met": False,
            "locked_fields": [],
        }
    ]


def test_scenario_character_starter_preserves_reference_image_metadata() -> None:
    reference = ScenarioStarterReferenceImage(
        id="starter-ref-1",
        path="scenario-starters/scenario-1/starter-ref-1.png",
        thumbnail_path="scenario-starters/scenario-1/thumbnails/starter-ref-1.png",
        mime_type="image/png",
        prompt_preview="Uploaded character reference image",
        source="uploaded",
        created_at="2026-07-12T00:00:00+00:00",
    )
    payload = scenario_character_starter_to_json(
        ScenarioCharacterStarter(
            starter_id="starter-ilyra",
            name="Captain Ilyra",
            appearance="Bronze cloak clasp and salt-stained boots.",
            reference_image=reference,
        )
    )

    assert payload["starter_id"] == "starter-ilyra"
    assert payload["reference_image"] == {
        "id": "starter-ref-1",
        "path": "scenario-starters/scenario-1/starter-ref-1.png",
        "thumbnail_path": "scenario-starters/scenario-1/thumbnails/starter-ref-1.png",
        "mime_type": "image/png",
        "prompt_preview": "Uploaded character reference image",
        "source": "uploaded",
        "created_at": "2026-07-12T00:00:00+00:00",
        "content_rating": "unclassified",
    }

    [normalized] = normalize_scenario_character_starters([payload], strict=True)

    assert normalized.starter_id == "starter-ilyra"
    assert normalized.reference_image == reference


def test_normalize_scenario_character_starters_rejects_invalid_strict_payload() -> None:
    with pytest.raises(TypeError, match="character_starters must be an array"):
        normalize_scenario_character_starters({"name": "Ilyra"}, strict=True)

    with pytest.raises(TypeError, match=r"character_starters\[0\]\.met"):
        normalize_scenario_character_starters(
            [{"name": "Ilyra", "met": "yes"}],
            strict=True,
        )


def test_structured_profile_completion_fills_blanks_without_overwriting() -> None:
    provider = RecordingStructuredProfileProvider(
        {
            "characters": [
                {
                    "name": "Captain Ilyra",
                    "aliases": ["The Captain"],
                    "role": "Generated role should not replace existing.",
                    "appearance": "Bronze cloak clasp and signal-horn scars.",
                    "voice": "Low, clipped commands.",
                    "relationships": {"Mara Voss": "guarded ally"},
                    "goals": "Keep the lower beacon burning through the storm.",
                    "motivations": "Protect the village below the cliff path.",
                    "current_intent": "Demand proof before unlocking the failsafe.",
                    "boundaries": (
                        "Will not abandon the tower while the lens is unstable."
                    ),
                    "attitude_toward_player": (
                        "Wary and unfairly suspicious after the last breach."
                    ),
                    "cooperation_conditions": (
                        "Shares the failsafe only after Mara shows the warrant."
                    ),
                    "status": "waiting near the beacon",
                    "met": True,
                    "locked_fields": [
                        "role",
                        "goals",
                        "motivations",
                        "current_intent",
                        "boundaries",
                        "attitude_toward_player",
                        "cooperation_conditions",
                    ],
                }
            ]
        }
    )
    completer = StructuredProviderCharacterProfileCompleter(
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
    )

    completed = asyncio.run(
        completer.complete(
            request=CharacterProfileCompletionRequest(
                scenario_type="full_roleplay",
                scenario_context="Captain Ilyra guards the beacon.",
                starters=(
                    ScenarioCharacterStarter(
                        name="Captain Ilyra",
                        role="Reviewed watch captain.",
                        goals="Keep the watch orderly.",
                        met=False,
                    ),
                ),
            )
        )
    )

    assert len(provider.requests) == 1
    assert completed == (
        ScenarioCharacterStarter(
            name="Captain Ilyra",
            aliases=("The Captain",),
            role="Reviewed watch captain.",
            appearance="Bronze cloak clasp and signal-horn scars.",
            voice="Low, clipped commands.",
            relationships={"Mara Voss": "guarded ally"},
            goals="Keep the watch orderly.",
            motivations="Protect the village below the cliff path.",
            current_intent="Demand proof before unlocking the failsafe.",
            boundaries="Will not abandon the tower while the lens is unstable.",
            attitude_toward_player=(
                "Wary and unfairly suspicious after the last breach."
            ),
            cooperation_conditions=(
                "Shares the failsafe only after Mara shows the warrant."
            ),
            status="waiting near the beacon",
            met=False,
            locked_fields=(
                "attitude_toward_player",
                "boundaries",
                "cooperation_conditions",
                "current_intent",
                "goals",
                "motivations",
                "role",
            ),
        ),
    )
    schema = provider.requests[0].schema
    properties = schema["properties"]
    assert isinstance(properties, dict)
    characters_schema = properties["characters"]
    assert isinstance(characters_schema, dict)
    item_schema = characters_schema["items"]
    assert isinstance(item_schema, dict)
    item_properties = item_schema["properties"]
    assert isinstance(item_properties, dict)
    assert {
        "goals",
        "motivations",
        "current_intent",
        "boundaries",
        "attitude_toward_player",
        "cooperation_conditions",
    } <= set(item_properties)
    assert "full spectrum" in provider.requests[0].messages[0].body
    assert "hostile" in provider.requests[0].messages[0].body


def test_structured_profile_completion_retries_denied_generated_voice(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="Captain Ilyra guards the beacon.",
        player_role="Keeper",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
        value="that's not nothing",
    )
    provider = RecordingStructuredProfileProvider(
        [
            {
                "characters": [
                    {
                        "name": "Captain Ilyra",
                        "voice": (
                            "Low clipped commands. Example: \"That's not nothing.\""
                        ),
                    }
                ]
            },
            {
                "characters": [
                    {
                        "name": "Captain Ilyra",
                        "voice": "Low clipped commands. Example: \"Hold the stair.\"",
                    }
                ]
            },
        ]
    )
    completer = StructuredProviderCharacterProfileCompleter(
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
        repositories=repositories,
    )

    completed = asyncio.run(
        completer.complete(
            request=CharacterProfileCompletionRequest(
                scenario_type="full_roleplay",
                scenario_context="Captain Ilyra guards the beacon.",
                starters=(ScenarioCharacterStarter(name="Captain Ilyra"),),
                save_id=save.id,
            )
        )
    )

    assert completed[0].voice == 'Low clipped commands. Example: "Hold the stair."'
    assert len(provider.requests) == 2
    assert "concrete examples" in provider.requests[0].messages[0].body
    assert "sample texts" in provider.requests[0].messages[0].body
    assert "That's not nothing" in provider.requests[1].messages[-1].body


def test_structured_profile_completion_retries_denied_generated_texting_style(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="Captain Ilyra guards the beacon.",
        player_role="Keeper",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
        value="that's not nothing",
    )
    provider = RecordingStructuredProfileProvider(
        [
            {
                "characters": [
                    {
                        "name": "Captain Ilyra",
                        "texting_style": (
                            "Curt logistics. Sample text: That's not nothing."
                        ),
                    }
                ]
            },
            {
                "characters": [
                    {
                        "name": "Captain Ilyra",
                        "texting_style": (
                            "Curt logistics. Sample text: Hold the stair."
                        ),
                    }
                ]
            },
        ]
    )
    completer = StructuredProviderCharacterProfileCompleter(
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
        repositories=repositories,
    )

    completed = asyncio.run(
        completer.complete(
            request=CharacterProfileCompletionRequest(
                scenario_type="full_roleplay",
                scenario_context="Captain Ilyra guards the beacon.",
                starters=(ScenarioCharacterStarter(name="Captain Ilyra"),),
                save_id=save.id,
            )
        )
    )

    assert completed[0].texting_style == (
        "Curt logistics. Sample text: Hold the stair."
    )
    assert len(provider.requests) == 2
    assert "That's not nothing" in provider.requests[1].messages[-1].body


def test_tool_profile_completion_retries_denied_generated_voice(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="Captain Ilyra guards the beacon.",
        player_role="Keeper",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
        value="that's not nothing",
    )
    provider = RecordingToolCallProfileProvider(
        [
            (
                ProviderToolCall(
                    id="tool-ilyra-1",
                    name="complete_character_profile",
                    arguments_json=json.dumps(
                        {
                            "name": "Captain Ilyra",
                            "voice": (
                                "Low clipped commands. Example: "
                                "\"That's not nothing.\""
                            ),
                        }
                    ),
                ),
            ),
            (
                ProviderToolCall(
                    id="tool-ilyra-2",
                    name="complete_character_profile",
                    arguments_json=json.dumps(
                        {
                            "name": "Captain Ilyra",
                            "voice": (
                                "Low clipped commands. Example: "
                                "\"Hold the stair.\""
                            ),
                        }
                    ),
                ),
            ),
        ]
    )
    completer = ToolCallingProviderCharacterProfileCompleter(
        provider=provider,
        provider_name="fake",
        model_id="fake-tools",
        repositories=repositories,
    )

    completed = asyncio.run(
        completer.complete(
            request=CharacterProfileCompletionRequest(
                scenario_type="full_roleplay",
                scenario_context="Captain Ilyra guards the beacon.",
                starters=(ScenarioCharacterStarter(name="Captain Ilyra"),),
                save_id=save.id,
            )
        )
    )

    assert completed[0].voice == 'Low clipped commands. Example: "Hold the stair."'
    assert len(provider.requests) == 2
    assert "That's not nothing" in provider.requests[1].messages[-1].body


def test_character_starter_json_round_trips_agency_fields() -> None:
    content = content_with_character_starters(
        scenario_type="full_roleplay",
        content={"title": "Lantern Keep"},
        starters=(
            ScenarioCharacterStarter(
                name="Captain Ilyra",
                goals="Hold the beacon until dawn.",
                motivations="Keep the lower village safe.",
                current_intent="Demand proof before opening the failsafe.",
                boundaries="Will not leave the tower during a lens breach.",
                attitude_toward_player=(
                    "Suspicious of Mara until she proves authority."
                ),
                cooperation_conditions=(
                    "Helps only after Mara shows the brass warrant."
                ),
            ),
        ),
    )

    starters = scenario_character_starters_for_content(
        scenario_type="full_roleplay",
        content=content,
    )

    assert starters == (
        ScenarioCharacterStarter(
            name="Captain Ilyra",
            goals="Hold the beacon until dawn.",
            motivations="Keep the lower village safe.",
            current_intent="Demand proof before opening the failsafe.",
            boundaries="Will not leave the tower during a lens breach.",
            attitude_toward_player=(
                "Suspicious of Mara until she proves authority."
            ),
            cooperation_conditions=(
                "Helps only after Mara shows the brass warrant."
            ),
        ),
    )


def test_structured_profile_completion_labels_direct_openrouter_request() -> None:
    provider = RecordingStructuredProfileProvider(
        {
            "characters": [
                {
                    "name": "Captain Ilyra",
                    "role": "Watch captain.",
                }
            ]
        }
    )
    completer = StructuredProviderCharacterProfileCompleter(
        provider=provider,
        provider_name="openrouter",
        model_id="openai/gpt-4o-mini",
    )

    asyncio.run(
        completer.complete(
            request=CharacterProfileCompletionRequest(
                scenario_type="full_roleplay",
                scenario_context="Captain Ilyra guards the beacon.",
                starters=(ScenarioCharacterStarter(name="Captain Ilyra"),),
            )
        )
    )

    assert provider.requests[0].openrouter_app_title == "Bragi"


def test_structured_profile_completer_routes_generation_and_enhancement_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routed_tasks: list[tuple[str, str, str | None]] = []

    def capture_routing(
        repositories: PersistenceRepositories | None,
        request: StructuredOutputRequest,
        *,
        task: str,
        save_id: str | None = None,
    ) -> StructuredOutputRequest:
        del repositories
        routed_tasks.append((request.schema_name, task, save_id))
        return request

    monkeypatch.setattr(
        completion,
        "request_with_openrouter_routing",
        capture_routing,
    )
    provider = RecordingStructuredProfileProvider(
        [
            {"characters": [{"name": "Captain Ilyra", "role": "Watch captain"}]},
            {"characters": [{"name": "Emily"}]},
            {
                "field_name": "appearance",
                "character": {
                    "name": "Captain Ilyra",
                    "appearance": "Fog-wet watchcoat.",
                },
            },
        ]
    )
    completer = StructuredProviderCharacterProfileCompleter(
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
    )

    asyncio.run(
        completer.complete(
            CharacterProfileCompletionRequest(
                scenario_type="full_roleplay",
                scenario_context="Captain Ilyra guards the beacon.",
                starters=(ScenarioCharacterStarter(name="Captain Ilyra"),),
                save_id="save-1",
            )
        )
    )
    asyncio.run(
        completer.generate_starters(
            CharacterStarterGenerationRequest(
                scenario_type="dating_sim",
                scenario_context="Emily waits by the gallery.",
                content={"player_character_name": "James Mitchell"},
                count=1,
                save_id="save-1",
            )
        )
    )
    asyncio.run(
        completer.enhance_field(
            CharacterFieldEnhancementRequest(
                scenario_type="full_roleplay",
                scenario_context="Captain Ilyra guards the beacon.",
                character=ScenarioCharacterStarter(
                    name="Captain Ilyra",
                    appearance="Bronze cloak clasp.",
                ),
                field_name="appearance",
                save_id="save-1",
            )
        )
    )

    assert routed_tasks == [
        ("character_profile_completion", "context_update", "save-1"),
        ("scenario_character_starters", "dating_sim_context_update", "save-1"),
        ("character_field_enhancement", "character_enhancement", "save-1"),
    ]


def test_tool_profile_completer_routes_generation_and_enhancement_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routed_tasks: list[tuple[str, str | None]] = []

    def capture_routing(
        repositories: PersistenceRepositories | None,
        request: ToolCallRequest,
        *,
        task: str,
        save_id: str | None = None,
    ) -> ToolCallRequest:
        del repositories
        routed_tasks.append((task, save_id))
        return request

    monkeypatch.setattr(
        completion,
        "request_with_openrouter_routing",
        capture_routing,
    )
    monkeypatch.setattr(
        completion,
        "budget_tool_call_request",
        lambda _repositories, request, *, task: request,
    )
    provider = RecordingToolCallProfileProvider([])
    completer = ToolCallingProviderCharacterProfileCompleter(
        provider=provider,
        provider_name="fake",
        model_id="fake-tool",
        repositories=cast(PersistenceRepositories, object()),
    )

    with pytest.raises(AssertionError, match="unexpected tool-call request"):
        asyncio.run(
            completer.complete(
                CharacterProfileCompletionRequest(
                    scenario_type="full_roleplay",
                    scenario_context="Captain Ilyra guards the beacon.",
                    starters=(ScenarioCharacterStarter(name="Captain Ilyra"),),
                    save_id="save-1",
                )
            )
        )
    assert (
        asyncio.run(
            completer.generate_starters(
                CharacterStarterGenerationRequest(
                    scenario_type="dating_sim",
                    scenario_context="Emily waits by the gallery.",
                    content={"player_character_name": "James Mitchell"},
                    count=1,
                    save_id="save-1",
                )
            )
        )
        == ()
    )
    with pytest.raises(AssertionError, match="unexpected tool-call request"):
        asyncio.run(
            completer.enhance_field(
                CharacterFieldEnhancementRequest(
                    scenario_type="full_roleplay",
                    scenario_context="Captain Ilyra guards the beacon.",
                    character=ScenarioCharacterStarter(
                        name="Captain Ilyra",
                        appearance="Bronze cloak clasp.",
                    ),
                    field_name="appearance",
                    save_id="save-1",
                )
            )
        )

    assert routed_tasks == [
        ("context_update", "save-1"),
        ("character_enhancement", "save-1"),
    ]


def test_structured_field_enhancement_uses_schema_for_existing_details() -> None:
    provider = RecordingStructuredProfileProvider(
        {
            "field_name": "appearance",
            "character": {
                "name": "Captain Ilyra",
                "appearance": (
                    "Bronze cloak clasp, fog-wet watchcoat, and old "
                    "signal-horn scars."
                ),
            },
        }
    )
    completer = StructuredProviderCharacterProfileCompleter(
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
    )

    enhanced = asyncio.run(
        completer.enhance_field(
            request=CharacterFieldEnhancementRequest(
                scenario_type="full_roleplay",
                scenario_context="Captain Ilyra guards the beacon.",
                character=ScenarioCharacterStarter(
                    name="Captain Ilyra",
                    appearance="Bronze cloak clasp.",
                    status="present at the beacon",
                ),
                field_name="appearance",
            )
        )
    )

    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.schema_name == "character_field_enhancement"
    assert "appearance" in str(request.schema)
    schema = request.schema
    schema_properties = schema["properties"]
    assert isinstance(schema_properties, dict)
    character_schema = schema_properties["character"]
    assert isinstance(character_schema, dict)
    character_properties = character_schema["properties"]
    assert isinstance(character_properties, dict)
    assert "relationships" not in character_properties
    assert "Preserve every existing detail" in request.messages[0].body
    assert "Add at least one concrete new detail" in request.messages[0].body
    assert "Do not return the target field unchanged" in request.messages[0].body
    assert "Always include skin tone" in request.messages[0].body
    assert "infer a plausible detail" in request.messages[0].body
    assert enhanced.name == "Captain Ilyra"
    assert enhanced.appearance == (
        "Bronze cloak clasp, fog-wet watchcoat, and old signal-horn scars."
    )


def test_structured_texting_style_enhancement_retries_denied_phrase(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Lantern Keep",
        premise="Captain Ilyra guards the beacon.",
        player_role="Keeper",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    repositories.set_scoped_setting(
        scope="save",
        scope_id=save.id,
        key=SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
        value="that's not nothing",
    )
    provider = RecordingStructuredProfileProvider(
        [
            {
                "field_name": "texting_style",
                "character": {
                    "name": "Captain Ilyra",
                    "texting_style": (
                        "Sends curt logistics. Sample text: That's not nothing."
                    ),
                },
            },
            {
                "field_name": "texting_style",
                "character": {
                    "name": "Captain Ilyra",
                    "texting_style": (
                        "Sends curt logistics. Sample text: Hold the stair."
                    ),
                },
            },
        ]
    )
    completer = StructuredProviderCharacterProfileCompleter(
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
        repositories=repositories,
    )

    enhanced = asyncio.run(
        completer.enhance_field(
            request=CharacterFieldEnhancementRequest(
                scenario_type="full_roleplay",
                scenario_context="Captain Ilyra guards the beacon.",
                character=ScenarioCharacterStarter(
                    name="Captain Ilyra",
                    texting_style="Sends curt logistics.",
                ),
                field_name="texting_style",
                save_id=save.id,
            )
        )
    )

    assert len(provider.requests) == 2
    assert "That's not nothing" in provider.requests[1].messages[-1].body
    assert enhanced.texting_style == (
        "Sends curt logistics. Sample text: Hold the stair."
    )


def test_structured_field_enhancement_ignores_bad_unrelated_relationships() -> None:
    provider = RecordingStructuredProfileProvider(
        {
            "field_name": "appearance",
            "character": {
                "name": "Captain Ilyra",
                "appearance": "Fog-wet watchcoat and signal-horn scars.",
                "relationships": "Mara Voss: trusted contact",
            },
        }
    )
    completer = StructuredProviderCharacterProfileCompleter(
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
    )

    enhanced = asyncio.run(
        completer.enhance_field(
            request=CharacterFieldEnhancementRequest(
                scenario_type="full_roleplay",
                scenario_context="Captain Ilyra guards the beacon.",
                character=ScenarioCharacterStarter(
                    name="Captain Ilyra",
                    appearance="Bronze cloak clasp.",
                    relationships={"Mara Voss": "trusted contact"},
                ),
                field_name="appearance",
            )
        )
    )

    assert len(provider.requests) == 1
    assert enhanced.appearance == "Fog-wet watchcoat and signal-horn scars."
    assert enhanced.relationships == {"Mara Voss": "trusted contact"}


@pytest.mark.parametrize(
    ("relationships_payload", "expected_relationships"),
    [
        (
            [
                {
                    "name": "Bell Keeper",
                    "relationship": "owes them a dangerous favor",
                }
            ],
            {"Bell Keeper": "owes them a dangerous favor"},
        ),
        (
            "Bell Keeper: owes them a dangerous favor",
            {"Bell Keeper": "owes them a dangerous favor"},
        ),
    ],
)
def test_structured_relationships_enhancement_accepts_common_payload_shapes(
    relationships_payload: object,
    expected_relationships: dict[str, object],
) -> None:
    provider = RecordingStructuredProfileProvider(
        {
            "field_name": "relationships",
            "character": {
                "name": "Captain Ilyra",
                "relationships": relationships_payload,
            },
        }
    )
    completer = StructuredProviderCharacterProfileCompleter(
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
    )

    enhanced = asyncio.run(
        completer.enhance_field(
            request=CharacterFieldEnhancementRequest(
                scenario_type="full_roleplay",
                scenario_context="Captain Ilyra guards the beacon.",
                character=ScenarioCharacterStarter(
                    name="Captain Ilyra",
                    relationships={"Mara Voss": "trusted contact"},
                ),
                field_name="relationships",
            )
        )
    )

    assert enhanced.relationships == expected_relationships
    schema_properties = provider.requests[0].schema["properties"]
    assert isinstance(schema_properties, dict)
    character_schema = schema_properties["character"]
    assert isinstance(character_schema, dict)
    character_properties = character_schema["properties"]
    assert isinstance(character_properties, dict)
    relationships_schema = character_properties["relationships"]
    assert isinstance(relationships_schema, dict)
    relationship_type = relationships_schema["type"]
    relationship_types = (
        relationship_type
        if isinstance(relationship_type, list)
        else [relationship_type]
    )
    assert "array" in relationship_types


def test_structured_field_enhancement_retries_empty_target_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []

    def capture_log_event(event: str, **fields: object) -> None:
        events.append((event, dict(fields)))

    monkeypatch.setattr(completion, "log_event", capture_log_event)
    provider = RecordingStructuredProfileProvider(
        [
            {
                "field_name": "appearance",
                "character": {
                    "name": "Captain Ilyra",
                    "appearance": "",
                },
            },
            {
                "field_name": "appearance",
                "character": {
                    "name": "Captain Ilyra",
                    "appearance": "Fog-wet watchcoat and signal-horn scars.",
                },
            },
        ]
    )
    completer = StructuredProviderCharacterProfileCompleter(
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
    )

    enhanced = asyncio.run(
        completer.enhance_field(
            request=CharacterFieldEnhancementRequest(
                scenario_type="full_roleplay",
                scenario_context="Captain Ilyra guards the beacon.",
                character=ScenarioCharacterStarter(
                    name="Captain Ilyra",
                    appearance="Bronze cloak clasp.",
                ),
                field_name="appearance",
            )
        )
    )

    assert len(provider.requests) == 2
    assert provider.requests[1].messages[-1].role == "user"
    assert "character.appearance must not be blank" in (
        provider.requests[1].messages[-1].body
    )
    assert enhanced.appearance == "Fog-wet watchcoat and signal-horn scars."
    assert events == [
        (
            "character_field_enhancement.structured_validation_failed",
            {
                "provider": "fake",
                "model": "fake-structured",
                "field_name": "appearance",
                "attempt": 1,
                "max_attempts": 3,
                "validation_failure_count": 1,
                "error_code": "blank_target_field",
            },
        ),
        (
            "character_field_enhancement.structured_validation_succeeded",
            {
                "provider": "fake",
                "model": "fake-structured",
                "field_name": "appearance",
                "attempt": 2,
                "max_attempts": 3,
                "validation_failure_count": 1,
            },
        ),
    ]
    serialized_events = repr(events)
    assert "Captain Ilyra" not in serialized_events
    assert "Bronze cloak clasp" not in serialized_events
    assert "Fog-wet watchcoat" not in serialized_events


def test_tool_field_enhancement_ignores_unrelated_malformed_relationships() -> None:
    provider = RecordingToolCallProfileProvider(
        [
            (
                ProviderToolCall(
                    id="call-1",
                    name="enhance_character_field",
                    arguments_json=json.dumps(
                        {
                            "field_name": "appearance",
                            "character": {
                                "name": "Captain Ilyra",
                                "appearance": "Fog-wet watchcoat and signal scars.",
                                "relationships": "Mara Voss: trusted contact",
                            },
                        }
                    ),
                ),
            ),
        ]
    )
    completer = ToolCallingProviderCharacterProfileCompleter(
        provider=provider,
        provider_name="fake",
        model_id="fake-tool",
    )

    enhanced = asyncio.run(
        completer.enhance_field(
            request=CharacterFieldEnhancementRequest(
                scenario_type="full_roleplay",
                scenario_context="Captain Ilyra guards the beacon.",
                character=ScenarioCharacterStarter(
                    name="Captain Ilyra",
                    appearance="Bronze cloak clasp.",
                    relationships={"Mara Voss": "trusted contact"},
                ),
                field_name="appearance",
            )
        )
    )

    assert len(provider.requests) == 1
    assert enhanced.appearance == "Fog-wet watchcoat and signal scars."
    assert enhanced.relationships == {"Mara Voss": "trusted contact"}


def test_structured_agency_field_enhancement_requires_evidence_source_ids() -> None:
    provider = RecordingStructuredProfileProvider(
        [
            {
                "field_name": "current_intent",
                "character": {
                    "name": "Captain Ilyra",
                    "current_intent": "Demand proof before opening the failsafe.",
                },
            },
            {
                "field_name": "current_intent",
                "character": {
                    "name": "Captain Ilyra",
                    "current_intent": "Demand proof before opening the failsafe.",
                    "evidence_source_ids": ["memory:memory-1"],
                },
            },
        ]
    )
    completer = StructuredProviderCharacterProfileCompleter(
        provider=provider,
        provider_name="fake",
        model_id="fake-structured",
    )

    enhanced = asyncio.run(
        completer.enhance_field(
            request=CharacterFieldEnhancementRequest(
                scenario_type="full_roleplay",
                scenario_context=(
                    "[memory:memory-1] Ilyra refuses to open the failsafe "
                    "without proof."
                ),
                character=ScenarioCharacterStarter(
                    name="Captain Ilyra",
                    current_intent="Guard the lens stair.",
                ),
                field_name="current_intent",
                evidence_source_ids=("memory:memory-1", "message:message-1"),
            )
        )
    )

    assert len(provider.requests) == 2
    request = provider.requests[0]
    character_schema = request.schema["properties"]["character"]
    character_properties = character_schema["properties"]
    assert "current_intent" in character_properties
    assert character_properties["evidence_source_ids"]["items"]["enum"] == [
        "memory:memory-1",
        "message:message-1",
    ]
    assert "cite evidence_source_ids" in request.messages[0].body
    assert "do not invent" in request.messages[0].body
    assert "invent useful neutral details" not in request.messages[0].body
    assert "character.evidence_source_ids" in provider.requests[1].messages[-1].body
    assert enhanced.current_intent == "Demand proof before opening the failsafe."
    assert enhanced.evidence_source_ids == ("memory:memory-1",)


def test_tool_field_enhancement_retries_empty_target_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []

    def capture_log_event(event: str, **fields: object) -> None:
        events.append((event, dict(fields)))

    monkeypatch.setattr(completion, "log_event", capture_log_event)
    provider = RecordingToolCallProfileProvider(
        [
            (
                ProviderToolCall(
                    id="call-1",
                    name="enhance_character_field",
                    arguments_json=json.dumps(
                        {
                            "field_name": "appearance",
                            "character": {
                                "name": "Captain Ilyra",
                                "appearance": "",
                            },
                        }
                    ),
                ),
            ),
            (
                ProviderToolCall(
                    id="call-2",
                    name="enhance_character_field",
                    arguments_json=json.dumps(
                        {
                            "field_name": "appearance",
                            "character": {
                                "name": "Captain Ilyra",
                                "appearance": "Fog-wet watchcoat and signal scars.",
                            },
                        }
                    ),
                ),
            ),
        ]
    )
    completer = ToolCallingProviderCharacterProfileCompleter(
        provider=provider,
        provider_name="fake",
        model_id="fake-tool",
    )

    enhanced = asyncio.run(
        completer.enhance_field(
            request=CharacterFieldEnhancementRequest(
                scenario_type="full_roleplay",
                scenario_context="Captain Ilyra guards the beacon.",
                character=ScenarioCharacterStarter(
                    name="Captain Ilyra",
                    appearance="Bronze cloak clasp.",
                ),
                field_name="appearance",
            )
        )
    )

    assert len(provider.requests) == 2
    assert "Always include skin tone" in provider.requests[0].messages[0].body
    assert "infer a plausible detail" in provider.requests[0].messages[0].body
    assert provider.requests[1].messages[-1].role == "tool"
    assert "character.appearance must not be blank" in (
        provider.requests[1].messages[-1].body
    )
    assert enhanced.appearance == "Fog-wet watchcoat and signal scars."
    assert events == [
        (
            "character_field_enhancement.tool_call_validation_failed",
            {
                "provider": "fake",
                "model": "fake-tool",
                "field_name": "appearance",
                "turn": 1,
                "max_turns": 3,
                "tool_call_count": 1,
                "accepted_count": 0,
                "error_count": 1,
                "validation_failure_count": 1,
                "error_codes": ("blank_target_field",),
            },
        ),
        (
            "character_field_enhancement.tool_call_validation_succeeded",
            {
                "provider": "fake",
                "model": "fake-tool",
                "field_name": "appearance",
                "turn": 2,
                "max_turns": 3,
                "tool_call_count": 1,
                "accepted_count": 1,
                "error_count": 0,
                "validation_failure_count": 1,
            },
        ),
    ]
    serialized_events = repr(events)
    assert "Captain Ilyra" not in serialized_events
    assert "Bronze cloak clasp" not in serialized_events
    assert "Fog-wet watchcoat" not in serialized_events
