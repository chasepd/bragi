"""Draft reusable continuation scenarios from active save state."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

from bragi.persistence.models import (
    ActiveThreadRecord,
    CharacterRecord,
    LocationRecord,
    SceneSnapshotRecord,
    WorldStateRecord,
)
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.scenario_service import (
    DATING_SIM_SECTIONS,
    FANTASY_ROLEPLAY_SECTIONS,
    FIRST_CONTACT_EXPLORATION_SECTIONS,
    HEIST_INFILTRATION_SECTIONS,
    INVESTIGATION_MYSTERY_SECTIONS,
    MERCHANT_TRADE_ROUTE_SECTIONS,
    MONSTER_HUNT_BOUNTY_SECTIONS,
    POLITICAL_INTRIGUE_SECTIONS,
    RETIRED_SCENARIO_REASON,
    ROAD_TRIP_PILGRIMAGE_SECTIONS,
    SCIENCE_FICTION_ROLEPLAY_SECTIONS,
    SETTLEMENT_BUILDER_SECTIONS,
    SURVIVAL_EXPEDITION_SECTIONS,
    TIME_LOOP_SECTIONS,
    ScenarioDraft,
    ScenarioGenerationProgressCallback,
    ScenarioService,
    ScenarioType,
    scenario_record_is_retired,
)

CONTINUATION_SECTION_IDS = (
    "title",
    "premise",
    "player_character_name",
    "player_role",
    "worldbuilding",
    "lore",
    "locations",
    "factions",
    "tone_genre",
    "current_scene",
    "opening_message",
)

DATING_SIM_CONTINUATION_SECTION_IDS = (
    *DATING_SIM_SECTIONS,
    "current_scene",
)

FANTASY_CONTINUATION_SECTION_IDS = (
    *FANTASY_ROLEPLAY_SECTIONS,
    "current_scene",
)

SCIENCE_FICTION_CONTINUATION_SECTION_IDS = (
    *SCIENCE_FICTION_ROLEPLAY_SECTIONS,
    "current_scene",
)

FIRST_CONTACT_CONTINUATION_SECTION_IDS = (
    *FIRST_CONTACT_EXPLORATION_SECTIONS,
    "current_scene",
)

SURVIVAL_EXPEDITION_CONTINUATION_SECTION_IDS = (
    *SURVIVAL_EXPEDITION_SECTIONS,
    "worldbuilding",
    "lore",
    "locations",
    "factions",
    "current_scene",
)

TIME_LOOP_CONTINUATION_SECTION_IDS = (
    *TIME_LOOP_SECTIONS,
    "worldbuilding",
    "lore",
    "locations",
    "factions",
    "current_scene",
)

INVESTIGATION_MYSTERY_CONTINUATION_SECTION_IDS = (
    *INVESTIGATION_MYSTERY_SECTIONS,
    "locations",
    "factions",
    "current_scene",
)

HEIST_INFILTRATION_CONTINUATION_SECTION_IDS = (
    *HEIST_INFILTRATION_SECTIONS,
    "locations",
    "factions",
    "current_scene",
)

POLITICAL_INTRIGUE_CONTINUATION_SECTION_IDS = (
    *POLITICAL_INTRIGUE_SECTIONS,
    "worldbuilding",
    "lore",
    "locations",
    "factions",
    "current_scene",
)

SETTLEMENT_BUILDER_CONTINUATION_SECTION_IDS = (
    *SETTLEMENT_BUILDER_SECTIONS,
    "worldbuilding",
    "lore",
    "locations",
    "factions",
    "current_scene",
)

MONSTER_HUNT_BOUNTY_CONTINUATION_SECTION_IDS = (
    *MONSTER_HUNT_BOUNTY_SECTIONS,
    "locations",
    "factions",
    "current_scene",
)

ROAD_TRIP_PILGRIMAGE_CONTINUATION_SECTION_IDS = (
    *ROAD_TRIP_PILGRIMAGE_SECTIONS,
    "worldbuilding",
    "lore",
    "locations",
    "factions",
    "current_scene",
)

MERCHANT_TRADE_ROUTE_CONTINUATION_SECTION_IDS = (
    *MERCHANT_TRADE_ROUTE_SECTIONS,
    "worldbuilding",
    "lore",
    "locations",
    "factions",
    "current_scene",
)

MAX_SEED_CHARS = 40_000
MAX_CHARACTERS = 20
MAX_LOCATIONS = 20
MAX_MEMORIES = 30
MAX_CONTEXT_INPUTS = 20
MAX_WORLD_STATE = 60
MAX_SUMMARIES = 8


@dataclass(frozen=True)
class ContinuationScenarioSnapshot:
    seed: str
    metadata: dict[str, object]


class ContinuationScenarioService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        scenario_service: ScenarioService | None = None,
    ) -> None:
        self.repositories = repositories
        self.scenario_service = scenario_service

    async def generate_draft(
        self,
        *,
        save_id: str,
        chapter_start_instructions: str = "",
        progress_callback: ScenarioGenerationProgressCallback | None = None,
    ) -> ScenarioDraft:
        if self.scenario_service is None:
            raise ValueError("Scenario generation service is required")
        snapshot = self.build_snapshot(
            save_id,
            chapter_start_instructions=chapter_start_instructions,
        )
        save = self.repositories.get_save(save_id)
        if save is None:
            raise ValueError(f"Unknown save id: {save_id}")
        source_type = _text(snapshot.metadata.get("source_scenario_type"))
        section_ids: tuple[str, ...]
        if source_type == ScenarioType.DATING_SIM.value:
            scenario_type = ScenarioType.DATING_SIM
            section_ids = DATING_SIM_CONTINUATION_SECTION_IDS
        elif source_type == ScenarioType.FANTASY_ROLEPLAY.value:
            scenario_type = ScenarioType.FANTASY_ROLEPLAY
            section_ids = FANTASY_CONTINUATION_SECTION_IDS
        elif source_type == ScenarioType.SCIENCE_FICTION_ROLEPLAY.value:
            scenario_type = ScenarioType.SCIENCE_FICTION_ROLEPLAY
            section_ids = SCIENCE_FICTION_CONTINUATION_SECTION_IDS
        elif source_type == ScenarioType.FIRST_CONTACT_EXPLORATION.value:
            scenario_type = ScenarioType.FIRST_CONTACT_EXPLORATION
            section_ids = FIRST_CONTACT_CONTINUATION_SECTION_IDS
        elif source_type == ScenarioType.SURVIVAL_EXPEDITION.value:
            scenario_type = ScenarioType.SURVIVAL_EXPEDITION
            section_ids = SURVIVAL_EXPEDITION_CONTINUATION_SECTION_IDS
        elif source_type == ScenarioType.TIME_LOOP.value:
            scenario_type = ScenarioType.TIME_LOOP
            section_ids = TIME_LOOP_CONTINUATION_SECTION_IDS
        elif source_type == ScenarioType.INVESTIGATION_MYSTERY.value:
            scenario_type = ScenarioType.INVESTIGATION_MYSTERY
            section_ids = INVESTIGATION_MYSTERY_CONTINUATION_SECTION_IDS
        elif source_type == ScenarioType.HEIST_INFILTRATION.value:
            scenario_type = ScenarioType.HEIST_INFILTRATION
            section_ids = HEIST_INFILTRATION_CONTINUATION_SECTION_IDS
        elif source_type == ScenarioType.POLITICAL_INTRIGUE.value:
            scenario_type = ScenarioType.POLITICAL_INTRIGUE
            section_ids = POLITICAL_INTRIGUE_CONTINUATION_SECTION_IDS
        elif source_type == ScenarioType.SETTLEMENT_BUILDER.value:
            scenario_type = ScenarioType.SETTLEMENT_BUILDER
            section_ids = SETTLEMENT_BUILDER_CONTINUATION_SECTION_IDS
        elif source_type == ScenarioType.MONSTER_HUNT_BOUNTY.value:
            scenario_type = ScenarioType.MONSTER_HUNT_BOUNTY
            section_ids = MONSTER_HUNT_BOUNTY_CONTINUATION_SECTION_IDS
        elif source_type == ScenarioType.ROAD_TRIP_PILGRIMAGE.value:
            scenario_type = ScenarioType.ROAD_TRIP_PILGRIMAGE
            section_ids = ROAD_TRIP_PILGRIMAGE_CONTINUATION_SECTION_IDS
        elif source_type == ScenarioType.MERCHANT_TRADE_ROUTE.value:
            scenario_type = ScenarioType.MERCHANT_TRADE_ROUTE
            section_ids = MERCHANT_TRADE_ROUTE_CONTINUATION_SECTION_IDS
        else:
            scenario_type = ScenarioType.FULL_ROLEPLAY
            section_ids = CONTINUATION_SECTION_IDS
        return await self.scenario_service.generate_draft(
            scenario_type=scenario_type,
            interaction_mode=save.interaction_mode,
            seed=snapshot.seed,
            section_ids=section_ids,
            metadata=snapshot.metadata,
            progress_callback=progress_callback,
        )

    def build_snapshot(
        self,
        save_id: str,
        *,
        chapter_start_instructions: str = "",
    ) -> ContinuationScenarioSnapshot:
        details = self.repositories.load_save_details(save_id)
        if details is None:
            raise ValueError(f"Unknown save id: {save_id}")
        scenario_content = _json_object(details.scenario.content_json)
        if scenario_record_is_retired(details.scenario.type, scenario_content):
            raise ValueError(RETIRED_SCENARIO_REASON)
        messages = self.repositories.list_messages(save_id)
        scene = self.repositories.get_scene_snapshot(save_id)
        threads = self.repositories.list_active_threads(save_id)
        locations = _select_locations(
            self.repositories.list_locations(save_id),
            scene=scene,
            threads=threads,
        )
        characters = _select_characters(
            self.repositories.list_characters(save_id),
            scene=scene,
            threads=threads,
        )
        world_state = _select_world_state(
            self.repositories.list_world_state(save_id),
            scene=scene,
            locations=locations,
            characters=characters,
            threads=threads,
        )
        memories = sorted(
            self.repositories.list_memories(save_id),
            key=lambda memory: (-memory.importance, memory.body.casefold()),
        )[:MAX_MEMORIES]
        context_inputs = sorted(
            self.repositories.list_context_sources(save_id),
            key=lambda source: (
                -_importance(source.metadata),
                source.source_type,
                source.title.casefold(),
            ),
        )[:MAX_CONTEXT_INPUTS]
        summaries = self.repositories.list_summaries(save_id)[-MAX_SUMMARIES:]
        chapter_start = chapter_start_instructions.strip()

        sections = [
            (
                "Continuation goal",
                "Create a clean chapter/continuation scenario from this save. "
                "Preserve continuity, character voices, relationships, major "
                "story beats, current obligations, durable facts, and the current "
                "situation. Do not copy transcript text or write a recap-only "
                "reset. Draft natural prose fields for a reusable Bragi scenario.",
            ),
            (
                "Latest-state authority",
                "Use the current scene, present characters, outstanding unresolved "
                "threads, durable world facts, important memories, and recent "
                "summaries as the authoritative continuation point. Do not regress "
                "to the original scenario starting state when the latest save state "
                "contradicts or advances beyond it.",
            ),
            *(
                [
                    (
                        "Chapter start instructions",
                        "Player preference for where the new chapter should "
                        "open. Honor this when it fits established continuity; "
                        "if it conflicts with latest durable state, preserve "
                        "continuity and adapt the preference.\n"
                        + chapter_start,
                    )
                ]
                if chapter_start
                else []
            ),
            (
                "Source save",
                "\n".join(
                    part
                    for part in (
                        f"Save title: {details.save.title}",
                        f"Scenario title: {details.scenario.title}",
                        f"Scenario premise: {details.scenario.premise}",
                        "Player character: "
                        + _text(scenario_content.get("player_character_name")),
                        f"Player role: {details.scenario.player_role}",
                        f"Chronicle message count: {len(messages)}",
                    )
                    if part.strip()
                ),
            ),
            (
                "Original scenario baseline",
                "Historical setup only; keep tone, premise, and stable identity, "
                "but do not reset the new chapter to this baseline if later save "
                "state moved past it.\n"
                + _scenario_sections_text(scenario_content),
            ),
            ("Current scene", _scene_text(scene, locations, characters)),
            ("Characters", _characters_text(characters)),
            ("Locations", _locations_text(locations)),
            (
                "Outstanding unresolved threads and obligations",
                _threads_text(threads),
            ),
            ("Durable world facts", _world_state_text(world_state)),
            ("Important memories", _memories_text(memories)),
            ("Context inputs", _context_inputs_text(context_inputs)),
            ("Story so far", _summaries_text(summaries)),
        ]
        seed = _cap_seed(
            "\n\n".join(
                f"## {title}\n{body}"
                for title, body in sections
                if body.strip()
            )
        )
        metadata: dict[str, object] = {
            "origin": "save_continuation",
            "source_save_id": details.save.id,
            "source_save_title": details.save.title,
            "source_scenario_title": details.scenario.title,
            "source_scenario_type": details.scenario.type,
            "source_message_count": len(messages),
            "generated_at": datetime.now(UTC).isoformat(),
            "character_continuity": [
                _character_metadata(character) for character in characters
            ],
        }
        if chapter_start:
            metadata["generation_prompt"] = chapter_start
        return ContinuationScenarioSnapshot(seed=seed, metadata=metadata)


def seed_continuation_characters(
    *,
    repositories: PersistenceRepositories,
    save_id: str,
    metadata: Mapping[str, object],
    source_message_id: str | None,
    include_player_character: bool = True,
) -> int:
    content_rating = metadata.get("content_rating")
    normalized_content_rating = (
        content_rating.strip()
        if isinstance(content_rating, str) and content_rating.strip()
        else "unclassified"
    )
    raw_items = metadata.get("character_continuity")
    if not isinstance(raw_items, list):
        return 0
    existing_keys = {
        _character_key(name)
        for character in repositories.list_characters(save_id)
        for name in (character.name, *character.aliases)
        if _character_key(name)
    }
    created_count = 0
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        item = cast(dict[str, object], raw_item)
        if not include_player_character and item.get("is_player_character") is True:
            continue
        name = _text(item.get("name"))
        key = _character_key(name)
        if not key or key in existing_keys:
            continue
        aliases = _string_list(item.get("aliases"))
        alias_keys = {_character_key(alias) for alias in aliases}
        if alias_keys & existing_keys:
            continue
        repositories.add_character(
            save_id=save_id,
            name=name,
            aliases=aliases,
            role=_text(item.get("role")),
            age=_text(item.get("age")),
            known_state=_text(item.get("known_state")),
            met=bool(item.get("met", True)),
            appearance=_text(item.get("appearance")),
            visual_notes=_text(item.get("visual_notes")),
            current_clothing=_text(item.get("current_clothing")),
            personality=_text(item.get("personality")),
            voice=_text(item.get("voice")),
            relationships=_object(item.get("relationships")),
            goals=_text(item.get("goals")),
            motivations=_text(item.get("motivations")),
            current_intent=_text(item.get("current_intent")),
            boundaries=_text(item.get("boundaries")),
            attitude_toward_player=_text(item.get("attitude_toward_player")),
            cooperation_conditions=_text(item.get("cooperation_conditions")),
            status=_text(item.get("status")) or "present at continuation start",
            private_notes=_text(item.get("private_notes")),
            source_message_id=source_message_id,
            protected_from_maintenance=True,
            content_rating=normalized_content_rating,
        )
        existing_keys.add(key)
        existing_keys.update(alias_keys)
        created_count += 1
    return created_count


def _cap_seed(seed: str) -> str:
    if len(seed) <= MAX_SEED_CHARS:
        return seed
    marker = "\n\n[Lower-priority continuation source material omitted for length.]"
    return seed[: MAX_SEED_CHARS - len(marker)].rstrip() + marker


def _select_locations(
    locations: list[LocationRecord],
    *,
    scene: SceneSnapshotRecord | None,
    threads: Sequence[ActiveThreadRecord],
) -> list[LocationRecord]:
    current_location_id = scene.current_location_id if scene is not None else None
    thread_terms = _thread_terms(threads)

    def priority(location: LocationRecord) -> tuple[int, str]:
        score = 0
        if location.id == current_location_id:
            score += 100
        if _record_matches_terms(
            (location.id, location.name, *location.aliases),
            thread_terms,
        ):
            score += 30
        if location.status:
            score += 8
        if location.hazards:
            score += 6
        if location.description or location.visual_description:
            score += 2
        return (-score, location.name.casefold())

    return sorted(locations, key=priority)[:MAX_LOCATIONS]


def _select_characters(
    characters: list[CharacterRecord],
    *,
    scene: SceneSnapshotRecord | None,
    threads: Sequence[ActiveThreadRecord],
) -> list[CharacterRecord]:
    present_ids = set(scene.present_character_ids) if scene is not None else set()
    current_location_id = scene.current_location_id if scene is not None else None
    thread_terms = _thread_terms(threads)

    def priority(character: CharacterRecord) -> tuple[int, str]:
        score = 0
        if character.id in present_ids:
            score += 100
        if (
            current_location_id is not None
            and character.location_id == current_location_id
        ):
            score += 35
        if character.protected_from_maintenance:
            score += 30
        if _record_matches_terms(
            (character.id, character.name, *character.aliases),
            thread_terms,
        ):
            score += 25
        if character.met:
            score += 10
        if character.known_state or character.status:
            score += 6
        if character.relationships:
            score += 4
        if character.voice or character.personality:
            score += 2
        return (-score, character.name.casefold())

    return sorted(characters, key=priority)[:MAX_CHARACTERS]


def _select_world_state(
    records: list[WorldStateRecord],
    *,
    scene: SceneSnapshotRecord | None,
    locations: Sequence[LocationRecord],
    characters: Sequence[CharacterRecord],
    threads: Sequence[ActiveThreadRecord],
) -> list[WorldStateRecord]:
    relevance_terms = _world_state_terms(
        scene=scene,
        locations=locations,
        characters=characters,
        threads=threads,
    )
    high_value_words = (
        "current",
        "scene",
        "objective",
        "status",
        "relationship",
        "thread",
        "obligation",
        "promise",
        "debt",
        "unresolved",
    )
    high_value_categories = {"scene", "location", "character", "relationship", "thread"}

    def priority(record: WorldStateRecord) -> tuple[int, float, str]:
        key = record.key.casefold()
        category = record.category.casefold()
        score = 0
        if category in high_value_categories:
            score += 35
        if any(word in key or word in category for word in high_value_words):
            score += 30
        if _record_matches_terms(
            (record.key, record.category, _format_object(record.value)),
            relevance_terms,
        ):
            score += 25
        return (-score, -record.confidence, record.key.casefold())

    return sorted(records, key=priority)[:MAX_WORLD_STATE]


def _scene_text(
    scene: object | None,
    locations: Sequence[object],
    characters: Sequence[CharacterRecord],
) -> str:
    if scene is None:
        return ""
    location_name = ""
    current_location_id = getattr(scene, "current_location_id", None)
    for location in locations:
        if getattr(location, "id", None) == current_location_id:
            location_name = getattr(location, "name", "")
            break
    present_names = [
        character.name
        for character in characters
        if character.id in getattr(scene, "present_character_ids", [])
    ]
    return "\n".join(
        part
        for part in (
            f"Location: {location_name}" if location_name else "",
            f"Situation: {getattr(scene, 'situation', '')}",
            f"Objective: {getattr(scene, 'objective', '')}",
            f"In-world time: {getattr(scene, 'in_world_time', '')}",
            f"Weather: {getattr(scene, 'weather', '')}",
            f"Mood: {getattr(scene, 'mood', '')}",
            f"Present characters: {', '.join(present_names)}" if present_names else "",
            _labeled_list("Nearby objects", getattr(scene, "nearby_objects", [])),
            _labeled_list("Hazards", getattr(scene, "hazards", [])),
        )
        if part.strip()
    )


def _characters_text(characters: list[CharacterRecord]) -> str:
    lines = []
    for character in characters:
        parts = [
            character.name,
            f"aliases: {', '.join(character.aliases)}" if character.aliases else "",
            f"role: {character.role}" if character.role else "",
            f"known state: {character.known_state}" if character.known_state else "",
            f"status: {character.status}" if character.status else "",
            f"appearance: {character.appearance}" if character.appearance else "",
            f"visual notes: {character.visual_notes}" if character.visual_notes else "",
            (
                f"current clothing: {character.current_clothing}"
                if character.current_clothing
                else ""
            ),
            f"personality: {character.personality}" if character.personality else "",
            f"voice: {character.voice}" if character.voice else "",
            (
                "relationships: " + _format_object(character.relationships)
                if character.relationships
                else ""
            ),
            f"goals: {character.goals}" if character.goals else "",
            f"motivations: {character.motivations}" if character.motivations else "",
            (
                f"current intent: {character.current_intent}"
                if character.current_intent
                else ""
            ),
            f"boundaries: {character.boundaries}" if character.boundaries else "",
            (
                f"attitude toward player: {character.attitude_toward_player}"
                if character.attitude_toward_player
                else ""
            ),
            (
                f"cooperation conditions: {character.cooperation_conditions}"
                if character.cooperation_conditions
                else ""
            ),
            (
                "narrator-only private notes: " + character.private_notes
                if character.private_notes
                else ""
            ),
        ]
        lines.append("- " + "; ".join(part for part in parts if part))
    return "\n".join(lines)


def _locations_text(locations: Sequence[object]) -> str:
    lines = []
    for location in locations:
        parts = [
            getattr(location, "name", ""),
            _labeled_list("aliases", getattr(location, "aliases", [])),
            getattr(location, "description", ""),
            (
                "visual: " + getattr(location, "visual_description", "")
                if getattr(location, "visual_description", "")
                else ""
            ),
            (
                "status: " + getattr(location, "status", "")
                if getattr(location, "status", "")
                else ""
            ),
            _labeled_list("hazards", getattr(location, "hazards", [])),
            _labeled_list("connections", getattr(location, "connections", [])),
        ]
        lines.append("- " + "; ".join(part for part in parts if part))
    return "\n".join(lines)


def _threads_text(threads: Sequence[object]) -> str:
    return "\n".join(
        "- "
        + "; ".join(
            part
            for part in (
                getattr(thread, "title", ""),
                getattr(thread, "description", ""),
                f"status: {getattr(thread, 'status', '')}",
                f"priority: {getattr(thread, 'priority', 0)}",
                _labeled_list("related", getattr(thread, "related_entities", [])),
            )
            if str(part).strip()
        )
        for thread in threads
    )


def _world_state_text(records: Sequence[object]) -> str:
    return "\n".join(
        "- "
        + f"{getattr(record, 'key', '')}: "
        + _format_object(getattr(record, "value", {}))
        for record in records
    )


def _memories_text(records: Sequence[object]) -> str:
    return "\n".join(
        f"- ({getattr(record, 'importance', 0):.2f}) {getattr(record, 'body', '')}"
        for record in records
    )


def _context_inputs_text(records: Sequence[object]) -> str:
    lines = []
    for record in records:
        fact_type = _text(getattr(record, "metadata", {}).get("fact_type", ""))
        title = getattr(record, "title", "")
        body = getattr(record, "body", "")
        lines.append(
            f"- {title} ({fact_type}): {body}" if fact_type else f"- {title}: {body}"
        )
    return "\n".join(lines)


def _summaries_text(records: Sequence[object]) -> str:
    return "\n".join(f"- {getattr(record, 'body', '')}" for record in records)


def _scenario_sections_text(content: dict[str, object]) -> str:
    return "\n".join(
        f"- {key}: {_text(value)}"
        for key, value in content.items()
        if not key.startswith("_") and _text(value)
    )


def _character_metadata(character: CharacterRecord) -> dict[str, object]:
    return {
        "name": character.name,
        "aliases": list(character.aliases),
        "role": character.role,
        "known_state": character.known_state,
        "met": character.met,
        "appearance": character.appearance,
        "visual_notes": character.visual_notes,
        "current_clothing": character.current_clothing,
        "personality": character.personality,
        "voice": character.voice,
        "relationships": dict(character.relationships),
        "goals": character.goals,
        "motivations": character.motivations,
        "current_intent": character.current_intent,
        "boundaries": character.boundaries,
        "attitude_toward_player": character.attitude_toward_player,
        "cooperation_conditions": character.cooperation_conditions,
        "status": character.status,
        "private_notes": character.private_notes,
        "is_player_character": character.is_player_character,
    }


def _json_object(value: str) -> dict[str, object]:
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    if isinstance(value, dict | list):
        return json.dumps(value, sort_keys=True)
    return str(value).strip()


def _object(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _importance(metadata: Mapping[str, object]) -> float:
    value = metadata.get("importance", 0.0)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)


def _thread_terms(threads: Sequence[ActiveThreadRecord]) -> set[str]:
    terms: set[str] = set()
    for thread in threads:
        terms.update(_search_terms(thread.title))
        terms.update(_search_terms(thread.description))
        for entity in thread.related_entities:
            terms.update(_search_terms(entity))
    return terms


def _world_state_terms(
    *,
    scene: SceneSnapshotRecord | None,
    locations: Sequence[LocationRecord],
    characters: Sequence[CharacterRecord],
    threads: Sequence[ActiveThreadRecord],
) -> set[str]:
    terms = _thread_terms(threads)
    if scene is not None:
        terms.update(_search_terms(scene.situation))
        terms.update(_search_terms(scene.objective))
        terms.update(_search_terms(scene.current_location_id))
        for character_id in scene.present_character_ids:
            terms.update(_search_terms(character_id))
    for location in locations:
        terms.update(_search_terms(location.id))
        terms.update(_search_terms(location.name))
        for alias in location.aliases:
            terms.update(_search_terms(alias))
    for character in characters:
        terms.update(_search_terms(character.id))
        terms.update(_search_terms(character.name))
        for alias in character.aliases:
            terms.update(_search_terms(alias))
    return terms


def _record_matches_terms(values: Sequence[object], terms: set[str]) -> bool:
    if not terms:
        return False
    haystack = " ".join(_text(value).casefold() for value in values if _text(value))
    return any(term in haystack for term in terms)


def _search_terms(value: object) -> set[str]:
    text = _text(value).casefold()
    if not text:
        return set()
    compact = " ".join(text.replace("_", " ").replace("-", " ").split())
    terms = {text, compact}
    terms.update(part for part in compact.split() if len(part) >= 3)
    return terms


def _format_object(value: object) -> str:
    if isinstance(value, dict | list):
        return json.dumps(value, sort_keys=True)
    return _text(value)


def _labeled_list(label: str, values: object) -> str:
    if not isinstance(values, list) or not values:
        return ""
    return f"{label}: " + ", ".join(_text(value) for value in values if _text(value))


def _character_key(value: str) -> str:
    return " ".join(value.strip().casefold().split())
