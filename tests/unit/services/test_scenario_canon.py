import asyncio
import json

import pytest

from bragi.providers.fake import FakeProviderClient
from bragi.services.scenario_canon import (
    CANON_CONTENT_KEY,
    ScenarioCanonCompiler,
    scenario_canon_claims,
    scenario_canon_is_current,
    scenario_canon_source_sections,
)


def _output() -> dict[str, object]:
    return {
        "sections": [
            {
                "section_id": "lore",
                "claims": [
                    {
                        "claim": "The beacon consumes one memory per use.",
                        "evidence_quote": "The beacon consumes one memory per use.",
                        "entity_anchors": [
                            {
                                "entity_type": "object",
                                "entity_key": "beacon",
                                "display_name": "the beacon",
                            }
                        ],
                        "fact_type": "rule",
                        "fact_key": "memory-cost",
                        "authority": "canonical",
                        "temporal_status": "durable",
                        "reveal_policy": "open",
                        "known_by": [],
                    },
                    {
                        "claim": "The keeper suspects the lens is alive.",
                        "evidence_quote": "The keeper suspects the lens is alive.",
                        "entity_anchors": [
                            {
                                "entity_type": "character",
                                "entity_key": "keeper",
                                "display_name": "the keeper",
                            }
                        ],
                        "fact_type": "knowledge",
                        "fact_key": "lens-suspicion",
                        "authority": "hypothesis",
                        "temporal_status": "current_at_scenario_start",
                        "reveal_policy": "narrator_only",
                        "known_by": ["keeper"],
                    },
                ],
            }
        ]
    }


def test_compiler_preserves_source_and_builds_atomic_grounded_claims() -> None:
    provider = FakeProviderClient(structured_output=_output())
    compiler = ScenarioCanonCompiler(
        provider=provider,
        provider_name="fake",
        model_id="canon-model",
    )
    content = {
        "title": "The Last Beacon",
        "premise": "A keeper guards the final light.",
        "lore": (
            "The beacon consumes one memory per use. "
            "The keeper suspects the lens is alive."
        ),
    }

    compiled = asyncio.run(
        compiler.compile(scenario_type="full_roleplay", content=content)
    )

    assert compiled["lore"] == content["lore"]
    assert CANON_CONTENT_KEY in compiled
    claims = scenario_canon_claims(compiled)
    assert {claim.claim for claim in claims} == {
        "The beacon consumes one memory per use.",
        "The keeper suspects the lens is alive.",
    }
    by_text = {claim.claim: claim for claim in claims}
    assert by_text["The beacon consumes one memory per use."].importance == 0.85
    hypothesis = by_text["The keeper suspects the lens is alive."]
    assert hypothesis.importance == 0.25
    assert hypothesis.reveal_policy == "narrator_only"
    assert len({claim.claim_key for claim in claims}) == 2
    request = provider.structured_output_requests[0]
    assert request.schema_name == "scenario_canon_claims"
    assert json.loads(json.dumps(request.schema))["additionalProperties"] is False


def test_compiler_normalizes_punctuated_evidence_into_atomic_claim() -> None:
    output = _output()
    sections = output["sections"]
    assert isinstance(sections, list)
    section = sections[0]
    assert isinstance(section, dict)
    section["claims"] = [
        {
            "claim": "Mira knows the route.",
            "evidence_quote": "Mira, the captain, knows the route.",
            "entity_anchors": [
                {
                    "entity_type": "character",
                    "entity_key": "mira",
                    "display_name": "Mira",
                }
            ],
            "fact_type": "knowledge",
            "fact_key": "route-knowledge",
            "authority": "canonical",
            "temporal_status": "durable",
            "reveal_policy": "player_known",
            "known_by": ["mira"],
        }
    ]

    compiled = asyncio.run(
        ScenarioCanonCompiler(
            provider=FakeProviderClient(structured_output=output),
            provider_name="fake",
            model_id="canon-model",
        ).compile(
            scenario_type="full_roleplay",
            content={"lore": "Mira, the captain, knows the route."},
        )
    )

    claims = scenario_canon_claims(compiled)
    assert [claim.claim for claim in claims] == ["Mira knows the route."]
    assert claims[0].evidence_quote == "Mira, the captain, knows the route."


@pytest.mark.parametrize(
    ("evidence", "claim"),
    [
        ("The beacon is not cold.", "The beacon is cold."),
        ("Mira follows Rowan.", "Rowan follows Mira."),
        ("It is false that Rowan is guilty.", "Rowan is guilty."),
        ("Mira believes Rowan left.", "Rowan left."),
        ("Rowan guarded the gate last winter.", "Rowan guards the gate."),
        ("The beacon glows. The keeper is blind.", "The beacon glows."),
    ],
)
def test_compiler_rejects_meaning_changing_evidence_normalization(
    evidence: str,
    claim: str,
) -> None:
    output = _output()
    sections = output["sections"]
    assert isinstance(sections, list)
    section = sections[0]
    assert isinstance(section, dict)
    claims = section["claims"]
    assert isinstance(claims, list)
    first_claim = claims[0]
    assert isinstance(first_claim, dict)
    first_claim["claim"] = claim
    first_claim["evidence_quote"] = evidence
    section["claims"] = [first_claim]

    with pytest.raises(ValueError, match="adds facts absent"):
        asyncio.run(
            ScenarioCanonCompiler(
                provider=FakeProviderClient(structured_output=output),
                provider_name="fake",
                model_id="canon-model",
            ).compile(
                scenario_type="full_roleplay",
                content={"lore": evidence},
            )
        )


def test_compiler_reuses_matching_compilation_without_provider_call() -> None:
    provider = FakeProviderClient(structured_output=_output())
    compiler = ScenarioCanonCompiler(
        provider=provider,
        provider_name="fake",
        model_id="canon-model",
    )
    content = {
        "title": "The Last Beacon",
        "lore": (
            "The beacon consumes one memory per use. "
            "The keeper suspects the lens is alive."
        ),
    }
    compiled = asyncio.run(
        compiler.compile(scenario_type="full_roleplay", content=content)
    )

    reused = asyncio.run(
        compiler.compile(scenario_type="full_roleplay", content=compiled)
    )

    assert reused == compiled
    assert len(provider.structured_output_requests) == 1


def test_config_fields_are_not_compiled_as_canon_sections() -> None:
    assert scenario_canon_source_sections(
        {
            "opening_message": "The bell rings.",
            "action_choices_enabled": True,
            "choice_style": "Four risky options.",
            "character_starters": [{"name": "Mara"}],
            "lore": "The old bell answers only at dusk.",
        }
    ) == {"lore": "The old bell answers only at dusk."}


def test_mystery_case_fields_are_compiled_as_canon_sections() -> None:
    assert scenario_canon_source_sections(
        {
            "case_facts": "Curator Vale vanished from a sealed gallery.",
            "case_status": "The disappearance remains unresolved.",
            "current_scene": "Mara stands outside the gallery.",
        }
    ) == {
        "case_facts": "Curator Vale vanished from a sealed gallery.",
        "case_status": "The disappearance remains unresolved.",
    }


def test_edit_invalidates_compilation_and_regenerates_claims() -> None:
    provider = FakeProviderClient(structured_output=_output())
    compiler = ScenarioCanonCompiler(
        provider=provider,
        provider_name="fake",
        model_id="canon-model",
    )
    original = asyncio.run(
        compiler.compile(
            scenario_type="full_roleplay",
            content={"lore": "The beacon consumes one memory per use. "
            "The keeper suspects the lens is alive."},
        )
    )
    edited = {**original, "lore": "The beacon is cold."}
    replacement = _output()
    replacement_sections = replacement["sections"]
    assert isinstance(replacement_sections, list)
    replacement_sections[0] = {
        "section_id": "lore",
        "claims": [
            {
                "claim": "The beacon is cold.",
                "evidence_quote": "The beacon is cold.",
                "entity_anchors": [
                    {
                        "entity_type": "object",
                        "entity_key": "beacon",
                        "display_name": "the beacon",
                    }
                ],
                "fact_type": "state",
                "fact_key": "temperature",
                "authority": "canonical",
                "temporal_status": "current_at_scenario_start",
                "reveal_policy": "open",
                "known_by": [],
            }
        ],
    }
    provider.structured_output = replacement

    regenerated = asyncio.run(
        compiler.compile(scenario_type="full_roleplay", content=edited)
    )

    assert not scenario_canon_is_current(edited)
    assert scenario_canon_is_current(regenerated)
    assert [claim.claim for claim in scenario_canon_claims(regenerated)] == [
        "The beacon is cold."
    ]
    assert len(provider.structured_output_requests) == 2


def test_stored_claims_are_rejected_when_provenance_is_tampered() -> None:
    provider = FakeProviderClient(structured_output=_output())
    compiled = asyncio.run(
        ScenarioCanonCompiler(
            provider=provider,
            provider_name="fake",
            model_id="canon-model",
        ).compile(
            scenario_type="full_roleplay",
            content={"lore": "The beacon consumes one memory per use. "
            "The keeper suspects the lens is alive."},
        )
    )
    payload = compiled[CANON_CONTENT_KEY]
    assert isinstance(payload, dict)
    claims = payload["claims"]
    assert isinstance(claims, list)
    assert isinstance(claims[0], dict)
    claims[0]["evidence_quote"] = "This never appeared in the source."

    assert not scenario_canon_is_current(compiled)
    assert scenario_canon_claims(compiled) == ()

    repaired = asyncio.run(
        ScenarioCanonCompiler(
            provider=provider,
            provider_name="fake",
            model_id="canon-model",
        ).compile(scenario_type="full_roleplay", content=compiled)
    )

    assert scenario_canon_is_current(repaired)
    assert len(provider.structured_output_requests) == 2


def test_empty_matching_payload_is_recompiled() -> None:
    provider = FakeProviderClient(structured_output=_output())
    compiler = ScenarioCanonCompiler(
        provider=provider,
        provider_name="fake",
        model_id="canon-model",
    )
    content = asyncio.run(
        compiler.compile(
            scenario_type="full_roleplay",
            content={"lore": "The beacon consumes one memory per use. "
            "The keeper suspects the lens is alive."},
        )
    )
    payload = content[CANON_CONTENT_KEY]
    assert isinstance(payload, dict)
    payload["claims"] = []

    assert not scenario_canon_is_current(content)
    repaired = asyncio.run(
        compiler.compile(scenario_type="full_roleplay", content=content)
    )
    assert scenario_canon_is_current(repaired)
    assert len(provider.structured_output_requests) == 2


def test_compiler_rejects_claim_without_exact_source_evidence() -> None:
    output = _output()
    sections = output["sections"]
    assert isinstance(sections, list)
    section = sections[0]
    assert isinstance(section, dict)
    claims = section["claims"]
    assert isinstance(claims, list)
    claim = claims[0]
    assert isinstance(claim, dict)
    claim["evidence_quote"] = "The beacon grants every wish."
    provider = FakeProviderClient(structured_output=output)
    compiler = ScenarioCanonCompiler(
        provider=provider,
        provider_name="fake",
        model_id="canon-model",
    )

    with pytest.raises(ValueError, match="exact source evidence"):
        asyncio.run(
            compiler.compile(
                scenario_type="full_roleplay",
                content={"title": "The Last Beacon", "lore": "A costly beacon."},
            )
        )


@pytest.mark.parametrize(
    ("claim", "evidence", "error"),
    [
        (
            "The beacon grants every wish.",
            "The beacon consumes one memory per use.",
            "adds facts absent",
        ),
        (
            "The duke is dead and the royal seal is missing.",
            "The duke is dead and the royal seal is missing.",
            "one atomic sentence",
        ),
    ],
)
def test_compiler_rejects_unsupported_or_compound_claims(
    claim: str,
    evidence: str,
    error: str,
) -> None:
    output = _output()
    sections = output["sections"]
    assert isinstance(sections, list)
    section = sections[0]
    assert isinstance(section, dict)
    claims = section["claims"]
    assert isinstance(claims, list)
    first_claim = claims[0]
    assert isinstance(first_claim, dict)
    first_claim["claim"] = claim
    first_claim["evidence_quote"] = evidence
    provider = FakeProviderClient(structured_output=output)

    with pytest.raises(ValueError, match=error):
        asyncio.run(
            ScenarioCanonCompiler(
                provider=provider,
                provider_name="fake",
                model_id="canon-model",
            ).compile(
                scenario_type="full_roleplay",
                content={
                    "lore": (
                        "The beacon consumes one memory per use. "
                        "The duke is dead and the royal seal is missing. "
                        "The keeper suspects the lens is alive."
                    )
                },
            )
        )


def test_compiler_rejects_partial_section_coverage() -> None:
    output = _output()
    sections = output["sections"]
    assert isinstance(sections, list)
    section = sections[0]
    assert isinstance(section, dict)
    claims = section["claims"]
    assert isinstance(claims, list)
    section["claims"] = claims[:1]

    with pytest.raises(ValueError, match="do not cover"):
        asyncio.run(
            ScenarioCanonCompiler(
                provider=FakeProviderClient(structured_output=output),
                provider_name="fake",
                model_id="canon-model",
            ).compile(
                scenario_type="full_roleplay",
                content={
                    "lore": (
                        "The beacon consumes one memory per use. "
                        "The keeper suspects the lens is alive."
                    )
                },
            )
        )


def test_compiler_assigns_repeated_evidence_to_distinct_occurrences() -> None:
    output = _output()
    sections = output["sections"]
    assert isinstance(sections, list)
    section = sections[0]
    assert isinstance(section, dict)
    claims = section["claims"]
    assert isinstance(claims, list)
    first_claim = claims[0]
    assert isinstance(first_claim, dict)
    section["claims"] = [dict(first_claim), dict(first_claim)]

    compiled = asyncio.run(
        ScenarioCanonCompiler(
            provider=FakeProviderClient(structured_output=output),
            provider_name="fake",
            model_id="canon-model",
        ).compile(
            scenario_type="full_roleplay",
            content={
                "lore": (
                    "The beacon consumes one memory per use. "
                    "The beacon consumes one memory per use."
                )
            },
        )
    )

    assert len(scenario_canon_claims(compiled)) == 2


def test_compiler_rejects_known_by_outside_entity_anchors() -> None:
    output = _output()
    sections = output["sections"]
    assert isinstance(sections, list)
    section = sections[0]
    assert isinstance(section, dict)
    claims = section["claims"]
    assert isinstance(claims, list)
    first_claim = claims[0]
    assert isinstance(first_claim, dict)
    first_claim["known_by"] = ["unanchored-character"]

    with pytest.raises(ValueError, match="entity anchor keys"):
        asyncio.run(
            ScenarioCanonCompiler(
                provider=FakeProviderClient(structured_output=output),
                provider_name="fake",
                model_id="canon-model",
            ).compile(
                scenario_type="full_roleplay",
                content={"lore": "The beacon consumes one memory per use. "
                "The keeper suspects the lens is alive."},
            )
        )


def test_compiler_rejects_restricted_claim_without_known_by() -> None:
    output = _output()
    sections = output["sections"]
    assert isinstance(sections, list)
    section = sections[0]
    assert isinstance(section, dict)
    claims = section["claims"]
    assert isinstance(claims, list)
    first_claim = claims[0]
    assert isinstance(first_claim, dict)
    first_claim["reveal_policy"] = "restricted"
    first_claim["known_by"] = []

    with pytest.raises(ValueError, match="require known_by"):
        asyncio.run(
            ScenarioCanonCompiler(
                provider=FakeProviderClient(structured_output=output),
                provider_name="fake",
                model_id="canon-model",
            ).compile(
                scenario_type="full_roleplay",
                content={"lore": "The beacon consumes one memory per use. "
                "The keeper suspects the lens is alive."},
            )
        )
