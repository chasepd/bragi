"""Schema-enforced compilation of scenario prose into atomic canon claims."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from bragi.providers.contracts import (
    ChatMessage,
    ProviderClient,
    StructuredOutputProvider,
    StructuredOutputRequest,
)

CANON_CONTENT_KEY = "_canon_claims"
CANON_COMPILER_VERSION = 1

SCENARIO_CORE_CONTENT_KEYS = frozenset(
    {
        "title",
        "premise",
        "setup_line",
        "player_character_name",
        "player_role",
        "tone_genre",
        "starting_scene",
        "opening_message",
        "current_scene",
        "relationship_seed",
        "character_starters",
        "action_choices_enabled",
        "choice_style",
    }
)

FACT_TYPES = (
    "identity",
    "relationship",
    "setting",
    "rule",
    "capability",
    "constraint",
    "location",
    "resource",
    "objective",
    "obligation",
    "event",
    "knowledge",
    "state",
    "other",
)
AUTHORITIES = ("canonical", "reported_belief", "rumor", "hypothesis")
TEMPORAL_STATUSES = (
    "durable",
    "current_at_scenario_start",
    "historical",
    "prospective_or_conditional",
    "timeless",
)
REVEAL_POLICIES = ("open", "player_known", "narrator_only", "restricted")
ENTITY_TYPES = (
    "character",
    "location",
    "faction",
    "group",
    "object",
    "event",
    "concept",
    "other",
)


@dataclass(frozen=True)
class ScenarioCanonClaim:
    claim_key: str
    source_section: str
    source_sha256: str
    claim: str
    evidence_quote: str
    entity_anchors: tuple[dict[str, str], ...]
    fact_type: str
    fact_key: str
    authority: str
    temporal_status: str
    reveal_policy: str
    known_by: tuple[str, ...]
    importance: float


class ScenarioCanonCompiler:
    def __init__(
        self,
        *,
        provider: StructuredOutputProvider,
        provider_name: str,
        model_id: str,
    ) -> None:
        self.provider = provider
        self.provider_name = provider_name
        self.model_id = model_id

    async def compile(
        self,
        *,
        scenario_type: str,
        content: Mapping[str, object],
    ) -> dict[str, object]:
        source_sections = scenario_canon_source_sections(content)
        digest = _source_digest(source_sections)
        if scenario_canon_is_current(content):
            return dict(content)
        if not source_sections:
            return {**content, CANON_CONTENT_KEY: _empty_payload(digest)}
        request = StructuredOutputRequest(
            provider=self.provider_name,
            model_id=self.model_id,
            schema_name="scenario_canon_claims",
            schema=_compiler_schema(tuple(source_sections)),
            messages=(
                ChatMessage(
                    role="system",
                    body=(
                        "Compile scenario prose into concise atomic canon claims. "
                        "Each claim may normalize its exact evidence quote into one "
                        "atomic statement, but must not add factual words absent from "
                        "that evidence. Use only exact evidence from the supplied "
                        "source sections. "
                        "Separate established facts, beliefs, rumors, and hypotheses; "
                        "mark time and reveal boundaries conservatively. For known_by, "
                        "use only entity_key values from that claim's entity_anchors."
                    ),
                ),
                ChatMessage(
                    role="user",
                    body=json.dumps(
                        {
                            "scenario_type": scenario_type,
                            "sections": source_sections,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                ),
            ),
        )
        response = await self.provider.generate_structured_output(request)
        payload = _validated_payload(
            response.data,
            source_sections=source_sections,
            digest=digest,
            provider=response.provider,
            model_id=response.model_id,
        )
        return {**content, CANON_CONTENT_KEY: payload}


def scenario_canon_source_sections(
    content: Mapping[str, object],
) -> dict[str, str]:
    sections: dict[str, str] = {}
    for key, value in content.items():
        section_id = str(key)
        if section_id.startswith("_") or section_id in SCENARIO_CORE_CONTENT_KEYS:
            continue
        text = _section_text(value)
        if text.strip():
            sections[section_id] = text.strip()
    return sections


def scenario_canon_claims(
    content: Mapping[str, object],
) -> tuple[ScenarioCanonClaim, ...]:
    if not scenario_canon_is_current(content):
        return ()
    payload = content.get(CANON_CONTENT_KEY)
    if not isinstance(payload, Mapping):
        return ()
    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list):
        return ()
    claims: list[ScenarioCanonClaim] = []
    source_sections = scenario_canon_source_sections(content)
    for raw in raw_claims:
        if not isinstance(raw, Mapping):
            continue
        section_id = str(raw.get("source_section", ""))
        source = source_sections.get(section_id)
        if source is None:
            continue
        normalized = _validated_claim(
            raw,
            section_id=section_id,
            source=source,
            source_sha256=_sha256(source),
        )
        anchors = normalized["entity_anchors"]
        known_by = normalized["known_by"]
        claims.append(
            ScenarioCanonClaim(
                claim_key=str(normalized["claim_key"]),
                source_section=str(normalized["source_section"]),
                source_sha256=str(normalized["source_sha256"]),
                claim=str(normalized["claim"]),
                evidence_quote=str(normalized["evidence_quote"]),
                entity_anchors=tuple(
                    {
                        "entity_type": str(anchor.get("entity_type", "")),
                        "entity_key": str(anchor.get("entity_key", "")),
                        "display_name": str(anchor.get("display_name", "")),
                    }
                    for anchor in anchors if isinstance(anchor, Mapping)
                ) if isinstance(anchors, list) else (),
                fact_type=str(normalized["fact_type"]),
                fact_key=str(normalized["fact_key"]),
                authority=str(normalized["authority"]),
                temporal_status=str(normalized["temporal_status"]),
                reveal_policy=str(normalized["reveal_policy"]),
                known_by=tuple(str(value) for value in known_by)
                if isinstance(known_by, list)
                else (),
                importance=cast(float, normalized["importance"]),
            )
        )
    return tuple(claims)


def scenario_canon_is_current(content: Mapping[str, object]) -> bool:
    source_sections = scenario_canon_source_sections(content)
    payload = content.get(CANON_CONTENT_KEY)
    return _compilation_matches(payload, digest=_source_digest(source_sections)) and (
        _stored_claims_are_grounded(payload, source_sections=source_sections)
    )


async def ensure_scenario_canon_for_save(
    *,
    repositories: Any,
    providers: Mapping[str, ProviderClient],
    save_id: str,
    details: Any | None = None,
) -> bool:
    """Compile a legacy or changed effective scenario before retrieval."""

    if details is None:
        details = repositories.load_save_details(save_id, message_limit=1)
    if details is None:
        raise ValueError(f"Unknown save id: {save_id}")
    content = _loaded_content(details.scenario.content_json)
    if scenario_canon_is_current(content):
        return False
    if not scenario_canon_source_sections(content):
        return False
    from bragi.services.model_preferences import roleplay_model_preference

    preference = roleplay_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose="context_update",
    )
    if preference is None:
        raise ValueError(
            "A Context Update model is required to compile scenario canon"
        )
    provider: object = providers.get(preference.provider)
    if not isinstance(provider, StructuredOutputProvider):
        raise ValueError(
            "The Context Update provider must support structured output"
        )
    compiled = await ScenarioCanonCompiler(
        provider=provider,
        provider_name=preference.provider,
        model_id=preference.model_id,
    ).compile(scenario_type=details.scenario.type, content=content)
    if repositories.get_active_save_scenario_update(save_id) is not None:
        repositories.update_active_save_scenario_content(
            save_id=save_id,
            content=compiled,
        )
    else:
        scenario = repositories.get_scenario(details.save.scenario_id)
        if scenario is None:
            raise ValueError(f"Unknown scenario id: {details.save.scenario_id}")
        repositories.update_scenario(
            scenario_id=scenario.id,
            title=scenario.title,
            premise=scenario.premise,
            player_role=scenario.player_role,
            interaction_mode=scenario.interaction_mode,
            content=compiled,
        )
    return True


def _validated_payload(
    data: Mapping[str, object],
    *,
    source_sections: Mapping[str, str],
    digest: str,
    provider: str,
    model_id: str,
) -> dict[str, object]:
    raw_sections = data.get("sections")
    if not isinstance(raw_sections, list):
        raise ValueError("Scenario canon compilation did not return sections")
    claims: list[dict[str, object]] = []
    returned_sections: set[str] = set()
    for raw_section in raw_sections:
        if not isinstance(raw_section, Mapping):
            raise ValueError("Scenario canon section must be an object")
        section_id = str(raw_section.get("section_id", "")).strip()
        source = source_sections.get(section_id)
        if source is None or section_id in returned_sections:
            raise ValueError("Scenario canon compilation returned an unknown section")
        returned_sections.add(section_id)
        raw_claims = raw_section.get("claims")
        if not isinstance(raw_claims, list) or not raw_claims:
            raise ValueError("Scenario canon compilation omitted a non-empty section")
        source_sha256 = _sha256(source)
        for raw_claim in raw_claims:
            claim = _validated_claim(
                raw_claim,
                section_id=section_id,
                source=source,
                source_sha256=source_sha256,
            )
            claims.append(claim)
        if not _claims_cover_source(source, claims, section_id=section_id):
            raise ValueError("Scenario canon claims do not cover their source section")
    if returned_sections != set(source_sections):
        raise ValueError("Scenario canon compilation omitted a non-empty section")
    claims.sort(key=lambda item: (str(item["source_section"]), str(item["claim_key"])))
    return {
        "version": CANON_COMPILER_VERSION,
        "source_digest": digest,
        "provider": provider,
        "model": model_id,
        "claims": claims,
    }


def _validated_claim(
    raw: object,
    *,
    section_id: str,
    source: str,
    source_sha256: str,
) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError("Scenario canon claim must be an object")
    claim = str(raw.get("claim", "")).strip()
    evidence_quote = str(raw.get("evidence_quote", "")).strip()
    if not claim or not evidence_quote or evidence_quote not in source:
        raise ValueError("Scenario canon claim lacks exact source evidence")
    if not _claim_is_atomic(claim):
        raise ValueError("Scenario canon claim must contain one atomic sentence")
    if not _claim_is_grounded(claim, evidence_quote):
        raise ValueError("Scenario canon claim adds facts absent from its evidence")
    fact_type = _enum(raw, "fact_type", FACT_TYPES)
    fact_key = str(raw.get("fact_key", "")).strip()
    if not fact_key:
        raise ValueError("Scenario canon fact_key is required")
    authority = _enum(raw, "authority", AUTHORITIES)
    temporal_status = _enum(raw, "temporal_status", TEMPORAL_STATUSES)
    reveal_policy = _enum(raw, "reveal_policy", REVEAL_POLICIES)
    anchors = _validated_anchors(raw.get("entity_anchors"))
    known_by_value = raw.get("known_by")
    if not isinstance(known_by_value, list):
        raise ValueError("Scenario canon known_by must be an array")
    known_by = sorted(
        {
            str(value).strip()
            for value in known_by_value
            if str(value).strip()
        }
    )
    anchor_keys = {anchor["entity_key"] for anchor in anchors}
    if not set(known_by) <= anchor_keys:
        raise ValueError("Scenario canon known_by must reference entity anchor keys")
    if reveal_policy == "restricted" and not known_by:
        raise ValueError("Restricted scenario canon claims require known_by")
    canonical = {
        "source_section": section_id,
        "source_sha256": source_sha256,
        "claim": claim,
        "evidence_quote": evidence_quote,
        "entity_anchors": anchors,
        "fact_type": fact_type,
        "fact_key": fact_key,
        "authority": authority,
        "temporal_status": temporal_status,
        "reveal_policy": reveal_policy,
        "known_by": known_by,
    }
    return {
        "claim_key": _sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False)),
        **canonical,
        "importance": _importance(
            fact_type=fact_type,
            authority=authority,
            temporal_status=temporal_status,
        ),
    }


def _validated_anchors(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("Scenario canon entity_anchors must be a non-empty array")
    anchors: list[dict[str, str]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError("Scenario canon entity anchor must be an object")
        entity_type = _enum(raw, "entity_type", ENTITY_TYPES)
        entity_key = str(raw.get("entity_key", "")).strip()
        display_name = str(raw.get("display_name", "")).strip()
        if not entity_key or not display_name:
            raise ValueError("Scenario canon entity anchor is incomplete")
        anchors.append(
            {
                "entity_type": entity_type,
                "entity_key": entity_key,
                "display_name": display_name,
            }
        )
    return sorted(anchors, key=lambda item: (item["entity_type"], item["entity_key"]))


def _claim_is_atomic(value: str) -> bool:
    if "\n" in value or ";" in value or "," in value:
        return False
    without_terminal = value.rstrip().rstrip(".!?")
    if re.search(r"[.!?]\s", without_terminal) is not None:
        return False
    return not _contains_coordinated_clauses(without_terminal)


def _claim_is_grounded(claim: str, evidence_quote: str) -> bool:
    claim_tokens = re.findall(r"[a-z0-9]+", claim.casefold())
    evidence_tokens = re.findall(r"[a-z0-9]+", evidence_quote.casefold())
    if claim_tokens == evidence_tokens:
        return True
    without_appositives = re.sub(r",[^,]+,", " ", evidence_quote)
    normalized_evidence = re.findall(r"[a-z0-9]+", without_appositives.casefold())
    return claim_tokens == normalized_evidence


def _contains_coordinated_clauses(value: str) -> bool:
    return re.search(
        r"\s+(?:and|but|while|whereas)\s+",
        value,
        re.IGNORECASE,
    ) is not None


def _claims_cover_source(
    source: str,
    claims: list[dict[str, object]],
    *,
    section_id: str,
) -> bool:
    intervals = sorted(
        (
            match.start(),
            match.end(),
        )
        for claim in claims
        if claim.get("source_section") == section_id
        for match in [re.search(re.escape(str(claim["evidence_quote"])), source)]
        if match is not None
    )
    cursor = 0
    uncovered: list[str] = []
    for start, end in intervals:
        if end <= cursor:
            continue
        uncovered.append(source[cursor:start])
        cursor = max(cursor, end)
    uncovered.append(source[cursor:])
    residue = " ".join(uncovered)
    residue = re.sub(r"\b(?:and|but|while|whereas)\b", " ", residue, flags=re.I)
    return re.sub(r"[^\w]+", "", residue) == ""


def _enum(raw: Mapping[str, object], key: str, values: tuple[str, ...]) -> str:
    value = str(raw.get(key, "")).strip()
    if value not in values:
        raise ValueError(f"Unknown scenario canon {key}: {value}")
    return value


def _compiler_schema(section_ids: tuple[str, ...]) -> dict[str, Any]:
    def string_enum(values: tuple[str, ...]) -> dict[str, object]:
        return {"type": "string", "enum": list(values)}

    claim_schema: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "claim": {"type": "string"},
            "evidence_quote": {"type": "string"},
            "entity_anchors": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "entity_type": string_enum(ENTITY_TYPES),
                        "entity_key": {"type": "string"},
                        "display_name": {"type": "string"},
                    },
                    "required": ["entity_type", "entity_key", "display_name"],
                },
            },
            "fact_type": string_enum(FACT_TYPES),
            "fact_key": {"type": "string"},
            "authority": string_enum(AUTHORITIES),
            "temporal_status": string_enum(TEMPORAL_STATUSES),
            "reveal_policy": string_enum(REVEAL_POLICIES),
            "known_by": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "claim",
            "evidence_quote",
            "entity_anchors",
            "fact_type",
            "fact_key",
            "authority",
            "temporal_status",
            "reveal_policy",
            "known_by",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "section_id": string_enum(section_ids),
                        "claims": {"type": "array", "items": claim_schema},
                    },
                    "required": ["section_id", "claims"],
                },
            }
        },
        "required": ["sections"],
    }


def _importance(*, fact_type: str, authority: str, temporal_status: str) -> float:
    if authority != "canonical":
        return 0.25
    if temporal_status == "current_at_scenario_start":
        return 0.45
    if fact_type in {"identity", "rule", "constraint"} and temporal_status in {
        "durable",
        "timeless",
    }:
        return 0.85
    return 0.65


def _compilation_matches(value: object, *, digest: str) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("version") == CANON_COMPILER_VERSION
        and value.get("source_digest") == digest
        and isinstance(value.get("claims"), list)
    )


def _stored_claims_are_grounded(
    value: object,
    *,
    source_sections: Mapping[str, str],
) -> bool:
    if not isinstance(value, Mapping):
        return False
    raw_claims = value.get("claims")
    if not isinstance(raw_claims, list):
        return False
    validated_claims: list[dict[str, object]] = []
    for raw in raw_claims:
        if not isinstance(raw, Mapping):
            return False
        source_section = str(raw.get("source_section", ""))
        source = source_sections.get(source_section)
        if source is None or raw.get("source_sha256") != _sha256(source):
            return False
        evidence_quote = str(raw.get("evidence_quote", "")).strip()
        if not evidence_quote or evidence_quote not in source:
            return False
        try:
            validated = _validated_claim(
                raw,
                section_id=source_section,
                source=source,
                source_sha256=_sha256(source),
            )
        except (TypeError, ValueError):
            return False
        validated_claims.append(validated)
    if source_sections and not validated_claims:
        return False
    for section_id, source in source_sections.items():
        if not _claims_cover_source(
            source,
            validated_claims,
            section_id=section_id,
        ):
            return False
    return True


def _empty_payload(digest: str) -> dict[str, object]:
    return {
        "version": CANON_COMPILER_VERSION,
        "source_digest": digest,
        "provider": "local",
        "model": "none",
        "claims": [],
    }


def _source_digest(sections: Mapping[str, str]) -> str:
    return _sha256(json.dumps(sections, ensure_ascii=False, sort_keys=True))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _section_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loaded_content(value: str) -> dict[str, object]:
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("Scenario content is invalid") from exc
    if not isinstance(loaded, dict):
        raise ValueError("Scenario content must be an object")
    return cast(dict[str, object], loaded)
