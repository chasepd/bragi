"""Reusable setting prose shared by multiple scenario templates."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from time import perf_counter

from bragi.app_logging import exception_log_fields, log_event
from bragi.content_rating_instructions import maximum_content_rating
from bragi.interaction_mode import InteractionMode
from bragi.persistence.models import PersistentWorldRecord, ScenarioRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatMessage,
    ChatPromptPurpose,
    ChatRequest,
    ProviderClient,
)
from bragi.services.content_rating import (
    effective_content_safety_policy,
    sanitize_content_rating,
)
from bragi.services.content_safety_service import ContentSafetyService
from bragi.services.provider_fallbacks import chat_with_fallback

PERSISTENT_WORLD_CONTENT_PREFIX = "persistent_world__"
PERSISTENT_WORLD_SNAPSHOT_PROVIDER = "bragi"
PERSISTENT_WORLD_SNAPSHOT_MODEL = "persistent_world_snapshot"
PERSISTENT_WORLD_SNAPSHOT_REASON = "persistent_world_snapshot"

WORLD_SECTION_DEFINITIONS: tuple[tuple[str, str, str], ...] = (
    (
        "overview",
        "Setting overview",
        "Summarize the world's central premise, era, and defining pressures.",
    ),
    (
        "cultures",
        "Cultures and peoples",
        "Describe the major cultures, peoples, customs, and social tensions.",
    ),
    (
        "geography",
        "Geography and places",
        (
            "Describe the important regions, settlements, landscapes, and travel "
            "constraints."
        ),
    ),
    (
        "factions",
        "Factions and powers",
        "Describe the major factions, institutions, alliances, and rivalries.",
    ),
    (
        "history_and_myths",
        "History and myths",
        (
            "Describe formative history, myths, legends, and unresolved historical "
            "questions."
        ),
    ),
    (
        "magic_or_technology",
        "Magic or technology",
        (
            "Describe the world's magic, science, technology, and the rules or "
            "costs that constrain them."
        ),
    ),
    (
        "tone",
        "Tone and themes",
        (
            "Describe the intended tone, themes, imagery, and boundaries for stories "
            "in this setting."
        ),
    ),
)
WORLD_SECTION_IDS = frozenset(
    section_id for section_id, _label, _prompt in WORLD_SECTION_DEFINITIONS
)


@dataclass(frozen=True)
class PersistentWorldDraft:
    title: str
    description: str
    sections: dict[str, str]
    source_metadata: dict[str, object]
    content_rating: str = "pg-13"


class PersistentWorldService:
    def __init__(
        self,
        repositories: PersistenceRepositories,
        *,
        current_user_id: str | None = None,
    ) -> None:
        self.repositories = repositories
        self.current_user_id = current_user_id

    def list_worlds(
        self,
        *,
        allowed_content_rating: str | None = None,
    ) -> list[dict[str, object]]:
        counts = self.repositories.count_scenarios_by_persistent_world()
        worlds: list[dict[str, object]] = []
        for world in self.repositories.list_persistent_worlds():
            if (
                allowed_content_rating is not None
                and _content_rating_exceeds(
                    world.content_rating,
                    allowed_content_rating,
                )
            ):
                continue
            worlds.append(self.as_json(world, scenario_count=counts.get(world.id, 0)))
        return worlds

    def get_world(
        self,
        world_id: str,
        *,
        allowed_content_rating: str | None = None,
    ) -> dict[str, object]:
        world = self.repositories.get_persistent_world(world_id)
        if world is None:
            raise ValueError(f"Unknown persistent world id: {world_id}")
        if (
            allowed_content_rating is not None
            and _content_rating_exceeds(world.content_rating, allowed_content_rating)
        ):
            raise ValueError("Persistent world exceeds your content rating")
        return self.as_json(
            world,
            scenario_count=self.repositories.count_scenarios_for_persistent_world(world_id),
        )

    def create_world(
        self,
        *,
        title: str,
        description: str = "",
        sections: Mapping[str, object],
        source_metadata: Mapping[str, object] | None = None,
        content_rating: str = "pg-13",
    ) -> PersistentWorldRecord:
        normalized_title = _required_text(title, "Persistent world title")
        normalized_sections = normalize_world_sections(sections)
        return self.repositories.create_persistent_world(
            title=normalized_title,
            description=description.strip(),
            sections=normalized_sections,
            source_metadata=_normalized_source_metadata(source_metadata),
            content_rating=sanitize_content_rating(content_rating),
        )

    def update_world(
        self,
        *,
        world_id: str,
        title: str,
        description: str = "",
        sections: Mapping[str, object],
        source_metadata: Mapping[str, object] | None = None,
        content_rating: str | None = None,
    ) -> PersistentWorldRecord:
        normalized_title = _required_text(title, "Persistent world title")
        normalized_sections = normalize_world_sections(sections)
        return self.repositories.update_persistent_world(
            world_id=world_id,
            title=normalized_title,
            description=description.strip(),
            sections=normalized_sections,
            source_metadata=(
                _normalized_source_metadata(source_metadata)
                if source_metadata is not None
                else None
            ),
            content_rating=(
                sanitize_content_rating(content_rating)
                if content_rating is not None
                else None
            ),
        )

    def delete_world(self, world_id: str) -> bool:
        return self.repositories.delete_persistent_world(world_id)

    def link_scenario(
        self,
        *,
        scenario_id: str,
        world_id: str | None,
    ) -> ScenarioRecord:
        return self.repositories.set_scenario_persistent_world(
            scenario_id=scenario_id,
            persistent_world_id=world_id,
        )

    def compose_scenario_content(
        self,
        scenario: ScenarioRecord,
        world: PersistentWorldRecord | None = None,
    ) -> dict[str, object]:
        try:
            loaded = json.loads(scenario.content_json)
        except json.JSONDecodeError:
            loaded = {}
        content = dict(loaded) if isinstance(loaded, dict) else {}
        resolved_world = world
        if resolved_world is None and scenario.persistent_world_id is not None:
            resolved_world = self.repositories.get_persistent_world(
                scenario.persistent_world_id
            )
        if resolved_world is None:
            return content
        for section_id, value in world_sections(resolved_world).items():
            content[f"{PERSISTENT_WORLD_CONTENT_PREFIX}{section_id}"] = value
        return content

    def materialize_save_snapshot(self, save_id: str) -> None:
        save = self.repositories.get_save(save_id)
        if save is None:
            raise ValueError(f"Unknown save id: {save_id}")
        scenario = self.repositories.get_scenario(save.scenario_id)
        if scenario is None or scenario.persistent_world_id is None:
            return
        world = self.repositories.get_persistent_world(scenario.persistent_world_id)
        if world is None:
            raise ValueError(
                f"Unknown persistent world id: {scenario.persistent_world_id}"
            )
        content = self.compose_scenario_content(scenario, world)
        source = _json_object(scenario.content_json).get("_source")
        snapshot_source = dict(source) if isinstance(source, dict) else {}
        snapshot_source["persistent_world_id"] = world.id
        snapshot_source["persistent_world_title"] = world.title
        content["_source"] = snapshot_source
        self.repositories.add_save_scenario_update(
            save_id=save.id,
            title=scenario.title,
            premise=scenario.premise,
            player_role=scenario.player_role,
            content=content,
            reason=PERSISTENT_WORLD_SNAPSHOT_REASON,
            provider=PERSISTENT_WORLD_SNAPSHOT_PROVIDER,
            model=PERSISTENT_WORLD_SNAPSHOT_MODEL,
        )

    async def generate_draft(
        self,
        *,
        seed: str,
        title: str = "",
        description: str = "",
        provider: ProviderClient,
        provider_name: str,
        model_id: str,
        providers: dict[str, ProviderClient] | None = None,
        section_ids: Iterable[str] | None = None,
    ) -> PersistentWorldDraft:
        normalized_seed = _required_text(seed, "World seed")
        selected_ids = tuple(
            section_ids
            if section_ids is not None
            else (
                section_id
                for section_id, _label, _prompt in WORLD_SECTION_DEFINITIONS
            )
        )
        unknown = sorted(set(selected_ids) - WORLD_SECTION_IDS)
        if unknown:
            raise ValueError(f"Unknown persistent world sections: {unknown}")
        provider_map = providers or {provider_name: provider}
        policy = effective_content_safety_policy(
            self.repositories,
            user_id=self.current_user_id,
        )
        safety_service = ContentSafetyService(
            repositories=self.repositories,
            providers=provider_map,
        )
        sections: dict[str, str] = {}
        ratings: list[str] = []
        started_at = perf_counter()
        for section_id in selected_ids:
            label, prompt = _section_definition(section_id)
            previous = "\n".join(
                f"{key}: {value}" for key, value in sections.items()
            )
            request = ChatRequest(
                provider=provider_name,
                model_id=model_id,
                interaction_mode=InteractionMode.ROLEPLAY,
                prompt_purpose=ChatPromptPurpose.SCENARIO_GENERATION,
                content_rating=policy.rating,
                fade_to_black_enabled=policy.fade_to_black_enabled,
                messages=(
                    ChatMessage(
                        role="system",
                        body=(
                            "Write one concise, vivid prose section for a reusable "
                            "roleplaying setting. Return ordinary prose only. Do not "
                            "use JSON, YAML, headings, or a preamble. "
                            f"Section focus: {prompt}"
                        ),
                    ),
                    ChatMessage(
                        role="player",
                        body=(
                            f"World seed: {normalized_seed}\n"
                            f"Section: {label}\n"
                            f"Already established sections:\n{previous or '(none)'}"
                        ),
                    ),
                ),
            )
            try:
                response = await chat_with_fallback(
                    repositories=self.repositories,
                    providers=provider_map,
                    request=request,
                    task="scenario_generation",
                    diagnostic_context={
                        "section_id": section_id,
                        "feature": "persistent_world",
                    },
                )
                safety = await safety_service.review_narration(
                    body=response.body.strip(),
                    content_rating=policy.rating,
                    fade_to_black_enabled=policy.fade_to_black_enabled,
                    source_request=request,
                    roleplay_type="full_roleplay",
                )
                sections[section_id] = safety.body.strip()
                ratings.append(safety.reviewed_content_rating)
            except Exception as exc:
                log_event(
                    "persistent_world.draft_failed",
                    section_id=section_id,
                    provider=provider_name,
                    model=model_id,
                    duration_ms=round((perf_counter() - started_at) * 1000),
                    **exception_log_fields(exc),
                )
                raise
        return PersistentWorldDraft(
            title=title.strip() or "Untitled persistent world",
            description=description.strip(),
            sections=sections,
            source_metadata={
                "origin": "ai_draft",
                "generation_prompt": normalized_seed,
                "provider": provider_name,
                "model": model_id,
            },
            content_rating=maximum_content_rating(
                tuple(ratings),
                default=policy.rating,
            ),
        )

    @staticmethod
    def as_json(
        world: PersistentWorldRecord,
        *,
        scenario_count: int = 0,
    ) -> dict[str, object]:
        return {
            "world_id": world.id,
            "title": world.title,
            "description": world.description,
            "sections": world_sections(world),
            "source_metadata": _json_object(world.source_metadata_json),
            "content_rating": world.content_rating,
            "scenario_count": scenario_count,
            "created_at": world.created_at,
            "updated_at": world.updated_at,
        }


def normalize_world_sections(sections: Mapping[str, object]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_key, raw_value in sections.items():
        key = str(raw_key).strip()
        if not key:
            continue
        value = (
            raw_value.strip()
            if isinstance(raw_value, str)
            else str(raw_value).strip()
        )
        if value:
            normalized[key] = value
    return normalized


def world_sections(world: PersistentWorldRecord) -> dict[str, str]:
    return normalize_world_sections(_json_object(world.content_json))


def _section_definition(section_id: str) -> tuple[str, str]:
    for candidate_id, label, prompt in WORLD_SECTION_DEFINITIONS:
        if candidate_id == section_id:
            return label, prompt
    raise ValueError(f"Unknown persistent world section: {section_id}")


def _required_text(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} is required")
    return normalized


def _json_object(value: str) -> dict[str, object]:
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _normalized_source_metadata(
    value: Mapping[str, object] | None,
) -> dict[str, object]:
    return dict(value or {})


def _content_rating_exceeds(value: str, allowed: str) -> bool:
    from bragi.content_rating_instructions import content_rating_exceeds

    return content_rating_exceeds(minimum_rating=value, allowed_rating=allowed)
