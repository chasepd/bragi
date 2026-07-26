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
from bragi.services.character_profile_completion import (
    CHARACTER_STARTERS_CONTENT_KEY,
    CharacterFieldEnhancementRequest,
    CharacterProfileCompletionRequest,
    CharacterStarterGenerationRequest,
    ScenarioCharacterStarter,
    ScenarioStarterReferenceImage,
    StructuredProviderCharacterProfileCompleter,
    ToolCallingProviderCharacterProfileCompleter,
    complete_character_starters,
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


def test_full_roleplay_starters_skip_player_and_alias_duplicates() -> None:
    starters = scenario_character_starters_for_content(
        scenario_type="full_roleplay",
        content={
            "player_character_name": "Mara Voss",
            "characters": (
                "Mara Voss, Captain Ilyra, Vey the outrider, "
                "Vey the scout, and Brother Senn."
            ),
        },
    )

    assert [starter.name for starter in starters] == [
        "Captain Ilyra",
        "Vey the outrider",
        "Brother Senn",
    ]
    vey = next(starter for starter in starters if starter.name == "Vey the outrider")
    assert vey.aliases == ("Vey",)
    assert vey.role == "outrider"


def test_generated_starters_skip_duplicate_first_names() -> None:
    starters = scenario_character_starters_for_content(
        scenario_type="dating_sim",
        content={
            "player_character_name": "James Mitchell",
            "romance_options": (
                "Emily Carter - violinist with a guarded smile.\n"
                "Emily Brooks - chef who knows the host.\n"
                "Lily Chen - photographer watching the door."
            ),
        },
    )

    assert [starter.name for starter in starters] == ["Emily Carter", "Lily Chen"]


def test_generated_starters_do_not_treat_titles_as_duplicate_first_names() -> None:
    starters = scenario_character_starters_for_content(
        scenario_type="first_contact_exploration",
        content={
            "player_character_name": "Dr. Mara Voss",
            "crew_and_command": (
                "Commander Reyes - cautious mission commander; "
                "Dr. Nia Sol - xenobiologist tracking contamination risk"
            ),
        },
    )

    assert [starter.name for starter in starters] == [
        "Commander Reyes",
        "Dr. Nia Sol",
    ]


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


def test_political_intrigue_starters_extract_major_npcs() -> None:
    starters = scenario_character_starters_for_content(
        scenario_type="political_intrigue",
        content={
            "player_character_name": "Mara Voss",
            "major_npcs": (
                "Mara Voss - the player should not become an NPC; "
                "Duchess Salen - regent who needs Mara's vote; "
                "Guildmaster Orro - charming broker who owes Mara one favor; "
                "Duchess Salen - duplicate line should be ignored."
            ),
        },
    )

    assert [starter.name for starter in starters] == [
        "Duchess Salen",
        "Guildmaster Orro",
    ]
    assert starters[0].role == "regent who needs Mara's vote"
    assert starters[1].known_state == (
        "Guildmaster Orro - charming broker who owes Mara one favor"
    )


def test_dating_sim_starters_extract_romance_options_from_natural_prose() -> None:
    starters = scenario_character_starters_for_content(
        scenario_type="dating_sim",
        content={
            "player_character_name": "Ren Takahashi",
            "romance_options": (
                "Mika Arai - female class president; precise, ambitious, and "
                "secretly lonely.\n"
                "Sora Minase - female swimmer; bright, competitive, and terrified "
                "of leaving home.\n"
                "Ren Takahashi - the player character should not become an NPC.\n"
                "Mika Arai - duplicate line should be ignored."
            ),
        },
    )

    assert [starter.name for starter in starters] == ["Mika Arai", "Sora Minase"]
    mika = starters[0]
    assert mika.role == (
        "female class president; precise, ambitious, and secretly lonely"
    )
    assert mika.known_state == (
        "Mika Arai - female class president; precise, ambitious, and secretly lonely"
    )
    assert mika.relationships == {
        "Ren Takahashi": "romance option for Ren Takahashi"
    }
    assert mika.status == "available romance option at scenario start"
    assert mika.texting_style == (
        "Distinct casual texting style for a romance route; use their role, "
        "personality, punctuation, emoji comfort, and response rhythm from the "
        "romance option notes."
    )


def test_dating_sim_starters_extract_labeled_option_blocks() -> None:
    starters = scenario_character_starters_for_content(
        scenario_type="dating_sim",
        content={
            "player_character_name": "Avery Quill",
            "romance_options": (
                "Here are four options for the player to choose from:\n\n"
                'Option One: Sable "Sab" Venn\n'
                "Sable Venn is a sky-market guide who pulls Avery into "
                "the floating festival.\n\n"
                "Option Two: Nira Sol\n"
                "Nira Sol is a signal cartographer who makes Avery's "
                "quiet archive life feel electric.\n\n"
                "Option Three: Ione Rook\n"
                "Ione Rook is a moon-library keeper whose steadiness "
                "matches the life Avery imagined.\n\n"
                'Option Four: Lark "Kestrel" Voss\n'
                "Lark Voss goes by Kestrel in the flight guild, and she "
                "challenges Avery's assumptions about commitment."
            ),
        },
    )

    assert [starter.name for starter in starters] == [
        "Sable Venn",
        "Nira Sol",
        "Ione Rook",
        "Lark Voss",
    ]
    assert "Option One" not in {starter.name for starter in starters}
    assert starters[0].aliases == ("Sab",)
    assert starters[3].aliases == ("Kestrel",)
    assert "sky-market guide" in starters[0].known_state
    assert "goes by Kestrel" in starters[3].known_state
    assert starters[0].relationships == {
        "Avery Quill": "romance option for Avery Quill"
    }


def test_dating_sim_starters_extract_names_from_prose_paragraphs() -> None:
    starters = scenario_character_starters_for_content(
        scenario_type="dating_sim",
        content={
            "player_character_name": "Avery Quill",
            "romance_options": (
                "Sable Venn is a sky-market guide who pulls Avery into "
                "the floating festival.\n"
                "Nira Sol is a signal cartographer who shares Avery's "
                "quiet archive life."
            ),
        },
    )

    assert [starter.name for starter in starters] == [
        "Sable Venn",
        "Nira Sol",
    ]
    assert all(len(starter.name) < 80 for starter in starters)


def test_dating_sim_starters_extract_names_from_em_dash_bio_paragraphs() -> None:
    starters = scenario_character_starters_for_content(
        scenario_type="dating_sim",
        content={
            "player_character_name": "Avery Quill",
            "romance_options": (
                "Tarin Vale\u2014a 28-year-old observatory mechanic "
                "volunteering at a star-chart workshop, she/her. Tarin has "
                "short dark hair, silver gloves, and a habit of marking "
                "constellations on her sleeves. Her hook: after watching Avery "
                "repair a portable planetarium, she asks them to align a "
                "stubborn lens array.\n\n"
                "Lio Maren\u2014a 29-year-old mapmaker designing an archive "
                "exhibit, she/her. She wears her copper hair in a long braid "
                "woven with signal ribbons. Her hook: she notices Avery guide "
                "a lost visitor through a rotating gallery with calm "
                "encouragement."
            ),
        },
    )

    assert [starter.name for starter in starters] == [
        "Tarin Vale",
        "Lio Maren",
    ]
    assert all(len(starter.name) < 80 for starter in starters)
    assert starters[0].role.startswith(
        "a 28-year-old observatory mechanic volunteering at a star-chart workshop"
    )
    assert "Her hook: after watching Avery repair" in starters[0].known_state
    assert starters[0].relationships == {
        "Avery Quill": "romance option for Avery Quill"
    }
    assert starters[0].status == "available romance option at scenario start"


def test_dating_sim_starters_reject_bio_prefix_before_hook_as_name() -> None:
    starters = scenario_character_starters_for_content(
        scenario_type="dating_sim",
        content={
            "player_character_name": "Avery Quill",
            "romance_options": (
                "A clever observatory mechanic with silver gloves. Her hook: "
                "she asks Avery to repair a star-chart projector after the workshop."
            ),
        },
    )

    assert starters == ()


def test_dating_sim_starters_can_be_generated_from_unstructured_romance_prose() -> None:
    provider = RecordingStructuredProfileProvider(
        {
            "characters": [
                    {
                        "name": "Emily",
                        "role": "Elementary school teacher with a nurturing heart.",
                        "age": "28",
                        "known_state": "Emily lights up when James mentions design.",
                    "appearance": "Warm brown eyes and freckles across her nose.",
                    "visual_notes": "Soft smile and relaxed cardigans.",
                    "personality": "Warm, attentive, and quietly creative.",
                    "voice": "Soft, lilting, and curious.",
                    "goals": "Find a partner who values her creative life.",
                    "motivations": "Share warmth without losing her independence.",
                    "boundaries": "Will not rush intimacy before trust is earned.",
                },
                    {
                        "name": "Lily",
                        "role": "Librarian with razor-sharp wit.",
                        "age": "29",
                        "known_state": "Lily is at speed dating on a dare.",
                    "appearance": "Dark hair, glasses, sleeve tattoos, cardigan.",
                    "visual_notes": "Vintage cardigan against bold tattoos.",
                    "personality": "Cynical, dry, and secretly hopeful.",
                    "voice": "Low, dry, and measured.",
                    "goals": "Find someone who enjoys her wit without dismissing it.",
                    "motivations": "Test whether hope is worth the risk.",
                    "boundaries": (
                        "Will not tolerate condescension about her guardedness."
                    ),
                },
                    {
                        "name": "Olivia",
                        "role": "Marketing director who wants direct honesty.",
                        "age": "31",
                        "known_state": "Olivia challenges James to say something real.",
                    "appearance": "Tall, sharp cheekbones, blond bob.",
                    "visual_notes": "Confident posture and precise gestures.",
                    "personality": "Direct, competitive, and protective.",
                    "voice": "Clear, confident, and warm when sincere.",
                    "goals": "Meet someone honest enough to challenge her.",
                    "motivations": "Escape the performance of always closing the deal.",
                    "boundaries": (
                        "Will not soften herself for someone who fears candor."
                    ),
                },
                    {
                        "name": "Chloe",
                        "role": "Freelance photographer chasing spontaneous sparks.",
                        "age": "27",
                        "known_state": "Chloe notices James watching the room.",
                    "appearance": "Copper hair and tiny silver earrings.",
                    "visual_notes": "Camera bag slung over one shoulder.",
                    "personality": "Quick, curious, and impulsively kind.",
                    "voice": "Fast, enthusiastic, and playful.",
                    "goals": "Find a spark that feels unscripted.",
                    "motivations": "Follow curiosity toward real connection.",
                    "boundaries": "Will not be treated as a disposable adventure.",
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
        complete_character_starters(
            completer=completer,
            scenario_type="dating_sim",
            content={
                "title": "The Speed of Love",
                "player_character_name": "James Mitchell",
                "romance_options": (
                    "Emily is a 28-year-old elementary school teacher with warm "
                    "brown eyes and freckles. The hook: she asks about James's "
                    "graphic design work.\n\n"
                    "Lily, 29, is a librarian with a razor-sharp wit. The hook: "
                    "she is here on a dare from her book club.\n\n"
                    "Olivia, 31, is a marketing director who closes deals for a "
                    "living. The hook: she challenges James to say something "
                    "real.\n\n"
                    "Chloe, 27, is a freelance photographer with a camera bag "
                    "over one shoulder. The hook: she notices James watching "
                    "the room."
                ),
            },
        )
    )

    assert [starter.name for starter in starters] == [
        "Emily",
        "Lily",
        "Olivia",
        "Chloe",
    ]
    assert provider.requests[0].schema_name == "dating_sim_character_starters"
    assert "James Mitchell" not in {starter.name for starter in starters}
    assert starters[1].relationships == {
        "James Mitchell": "romance option for James Mitchell"
    }
    assert starters[2].status == "available romance option at scenario start"
    assert starters[3].appearance == "Copper hair and tiny silver earrings."
    assert [starter.age for starter in starters] == ["28", "29", "31", "27"]
    assert len(provider.requests) == 1


def test_structured_dating_starter_generation_gets_name_candidates() -> None:
    provider = RecordingStructuredProfileProvider(
        {
            "characters": [
                {
                    "name": "Avery",
                    "role": "Photographer chasing unscripted sparks.",
                    "known_state": "Avery asks James about the gallery light.",
                    "appearance": "Copper hair and tiny silver earrings.",
                    "visual_notes": "Camera strap and quick, curious glances.",
                    "personality": "Quick, curious, and impulsively kind.",
                    "voice": 'Bright and candid. Example: "Show me what you see."',
                    "texting_style": (
                        "Short excited bursts. Sample text: Send the photo?"
                    ),
                    "goals": "Find a connection that feels unscripted.",
                    "motivations": "Follow curiosity toward real trust.",
                    "boundaries": "Will not be treated as a disposable adventure.",
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
        complete_character_starters(
            completer=completer,
            scenario_type="dating_sim",
            content={
                "player_character_name": "James Mitchell",
                "romance_options": (
                    "A photographer with a camera bag asks about James's "
                    "favorite gallery light."
                ),
            },
        )
    )

    request_body = provider.requests[0].messages[1].body
    assert "Ordinary contemporary name candidates" in request_body
    assert "Feminine:" in request_body
    assert "Masculine:" in request_body
    assert "Neutral:" in request_body
    assert "Preserve explicit scenario names" in request_body
    assert [starter.name for starter in starters] == ["Avery"]
    assert len(provider.requests) == 1


def test_dating_sim_starters_can_be_generated_from_tool_calls() -> None:
    provider = RecordingToolCallProfileProvider(
        [
            (
                ProviderToolCall(
                    id="tool-emily",
                    name="create_dating_sim_character_starter",
                    arguments_json=json.dumps(
                        {
                            "name": "Emily",
                            "role": "Elementary school teacher.",
                            "age": "",
                            "known_state": "Emily asks James about his designs.",
                            "appearance": "Warm brown eyes and freckles.",
                            "visual_notes": "Soft cardigans and quick smiles.",
                            "personality": "Warm and attentive.",
                            "voice": "Soft, lilting, and curious.",
                            "goals": "Find a kind design-minded partner.",
                            "motivations": "Share warmth with someone attentive.",
                            "boundaries": "Will not rush past trust.",
                        }
                    ),
                ),
                ProviderToolCall(
                    id="tool-lily",
                    name="create_dating_sim_character_starter",
                    arguments_json=json.dumps(
                        {
                            "name": "Lily",
                            "role": "Librarian with dry wit.",
                            "age": "29",
                            "known_state": "Lily is here on a book-club dare.",
                            "appearance": "Dark hair, glasses, and sleeve tattoos.",
                            "visual_notes": "Vintage cardigan and tattoo sleeves.",
                            "personality": "Cynical but secretly hopeful.",
                            "voice": "Low, dry, and measured.",
                            "goals": "Find someone who keeps up with her wit.",
                            "motivations": "See whether hope is worth the risk.",
                            "boundaries": "Will not tolerate condescension.",
                        }
                    ),
                ),
            )
        ]
    )
    completer = ToolCallingProviderCharacterProfileCompleter(
        provider=provider,
        provider_name="fake",
        model_id="fake-tools",
    )

    starters = asyncio.run(
        complete_character_starters(
            completer=completer,
            scenario_type="dating_sim",
            content={
                "player_character_name": "James Mitchell",
                "romance_options": (
                    "Emily is a teacher. Lily, 29, is a librarian."
                ),
            },
        )
    )

    assert [starter.name for starter in starters] == ["Emily", "Lily"]
    assert provider.requests[0].tools[0].name == (
        "create_dating_sim_character_starter"
    )
    assert starters[0].relationships == {
        "James Mitchell": "romance option for James Mitchell"
    }
    assert starters[1].status == "available romance option at scenario start"
    assert len(provider.requests) == 1


def test_tool_dating_starter_generation_gets_name_candidates() -> None:
    provider = RecordingToolCallProfileProvider(
        [
            (
                ProviderToolCall(
                    id="tool-avery",
                    name="create_dating_sim_character_starter",
                    arguments_json=json.dumps(
                        {
                            "name": "Avery",
                            "role": "Photographer chasing unscripted sparks.",
                            "known_state": (
                                "Avery asks James about the gallery light."
                            ),
                            "appearance": "Copper hair and tiny silver earrings.",
                            "visual_notes": (
                                "Camera strap and quick, curious glances."
                            ),
                            "personality": (
                                "Quick, curious, and impulsively kind."
                            ),
                            "voice": (
                                'Bright and candid. Example: "Show me what you see."'
                            ),
                            "texting_style": (
                                "Short excited bursts. Sample text: Send the photo?"
                            ),
                            "goals": "Find a connection that feels unscripted.",
                            "motivations": "Follow curiosity toward real trust.",
                            "boundaries": (
                                "Will not be treated as a disposable adventure."
                            ),
                        }
                    ),
                ),
            )
        ]
    )
    completer = ToolCallingProviderCharacterProfileCompleter(
        provider=provider,
        provider_name="fake",
        model_id="fake-tools",
    )

    starters = asyncio.run(
        complete_character_starters(
            completer=completer,
            scenario_type="dating_sim",
            content={
                "player_character_name": "James Mitchell",
                "romance_options": (
                    "A photographer with a camera bag asks about James's "
                    "favorite gallery light."
                ),
            },
        )
    )

    request_body = provider.requests[0].messages[1].body
    assert "Ordinary contemporary name candidates" in request_body
    assert "Preserve explicit scenario names" in request_body
    assert [starter.name for starter in starters] == ["Avery"]
    assert len(provider.requests) == 1


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
                    "romance_options": "Emily is a teacher waiting near the gallery.",
                },
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
    assert starters[0].relationships == {
        "James Mitchell": "romance option for James Mitchell"
    }


def test_content_with_character_starters_prefers_existing_explicit_payload() -> None:
    content = content_with_character_starters(
        scenario_type="full_roleplay",
        content={
            "characters": "Captain Ilyra",
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
            "boundaries": "",
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
                    "boundaries": (
                        "Will not abandon the tower while the lens is unstable."
                    ),
                    "status": "waiting near the beacon",
                    "met": True,
                    "locked_fields": ["role", "goals", "motivations", "boundaries"],
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
            boundaries="Will not abandon the tower while the lens is unstable.",
            status="waiting near the beacon",
            met=False,
            locked_fields=("boundaries", "goals", "motivations", "role"),
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
    assert {"goals", "motivations", "boundaries"} <= set(item_properties)


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
                boundaries="Will not leave the tower during a lens breach.",
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
            boundaries="Will not leave the tower during a lens breach.",
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
        repositories=cast(PersistenceRepositories, object()),
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
                content={"romance_options": "Emily is waiting."},
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
        ("dating_sim_character_starters", "context_update", "save-1"),
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
    with pytest.raises(AssertionError, match="unexpected tool-call request"):
        asyncio.run(
            completer.generate_starters(
                CharacterStarterGenerationRequest(
                    scenario_type="dating_sim",
                    scenario_context="Emily waits by the gallery.",
                    content={"romance_options": "Emily is waiting."},
                    save_id="save-1",
                )
            )
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
