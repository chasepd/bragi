import asyncio
import json

import pytest

from bragi.providers.fake import FakeProviderClient
from bragi.services.scenario_canon import (
    CANON_CONTENT_KEY,
    ScenarioCanonCompiler,
    scenario_canon_claims,
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
