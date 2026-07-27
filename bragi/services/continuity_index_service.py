"""Canonical continuity index backed by persisted context sources."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from bragi.persistence.models import (
    CharacterRecord,
    CharacterTextMessageRecord,
    CharacterTextThreadRecord,
    ContextObservationRecord,
    ContextSourceRecord,
    LocationRecord,
    MemoryRecord,
    ScenarioRecord,
    SceneSnapshotRecord,
    WorldStateRecord,
)
from bragi.persistence.repositories import (
    PersistenceRepositories,
    canonical_claim_fingerprint,
)
from bragi.services.active_thread_lifecycle import (
    active_thread_is_prompt_visible,
    normalize_active_thread_status,
    normalize_active_thread_visibility,
)
from bragi.services.character_text_context import (
    canonical_character_text_context_messages,
    character_text_audience_character_ids,
)
from bragi.services.character_text_world_update_service import (
    character_text_source_ref,
)
from bragi.services.context_assembly import scenario_section_candidates
from bragi.services.open_threads import (
    has_active_thread_records,
    is_open_threads_aggregate_key,
)
from bragi.services.summary_safety import validate_summary_output

HIGH_VALUE_FACT_TYPES = frozenset(
    {
        "character_voice",
        "identity",
        "inventory",
        "location",
        "open_obligation",
        "promise",
        "relationship",
    }
)

DEFAULT_WORLD_STATE_INDEX_LIMIT = 250
DEFAULT_MEMORY_INDEX_LIMIT = 250
DEFAULT_SUMMARY_INDEX_LIMIT = 80
DEFAULT_ACTIVE_THREAD_INDEX_LIMIT = 512
CHARACTER_PROFILE_DETAIL_MAX_CHARS = 320
CHARACTER_TEXT_THREAD_RECENT_MESSAGE_LIMIT = 4
CHARACTER_TEXT_THREAD_LINE_MAX_CHARS = 220


@dataclass(frozen=True)
class ContinuityIndexSyncResult:
    indexed_count: int
    skipped_counts: dict[str, int] = field(default_factory=dict)


class ContinuityIndexService:
    def __init__(
        self,
        repositories: PersistenceRepositories,
        *,
        world_state_limit: int | None = DEFAULT_WORLD_STATE_INDEX_LIMIT,
        memory_limit: int | None = DEFAULT_MEMORY_INDEX_LIMIT,
        summary_limit: int | None = DEFAULT_SUMMARY_INDEX_LIMIT,
        active_thread_limit: int | None = DEFAULT_ACTIVE_THREAD_INDEX_LIMIT,
    ) -> None:
        self.repositories = repositories
        self.world_state_limit = (
            None if world_state_limit is None else max(0, world_state_limit)
        )
        self.memory_limit = None if memory_limit is None else max(0, memory_limit)
        self.summary_limit = None if summary_limit is None else max(0, summary_limit)
        self.active_thread_limit = (
            None
            if active_thread_limit is None
            else max(0, active_thread_limit)
        )

    def sync_save(self, save_id: str) -> ContinuityIndexSyncResult:
        self.repositories.begin_transaction()
        try:
            result = self._sync_save(save_id)
            self.repositories.commit_transaction()
            return result
        except Exception:
            self.repositories.rollback_transaction()
            raise

    def _sync_save(self, save_id: str) -> ContinuityIndexSyncResult:
        details = self.repositories.load_save_details(save_id)
        if details is None:
            raise ValueError(f"Unknown save id: {save_id}")
        records: list[ContextSourceRecord] = []
        skipped_counts: dict[str, int] = {}
        scenario = details.scenario
        snapshot = self.repositories.get_scene_snapshot(save_id)
        locations = self.repositories.list_locations(save_id)
        characters = self.repositories.list_characters(save_id)
        message_order = {
            message.id: index for index, message in enumerate(details.messages)
        }
        records.extend(self._sync_scenario_sections(save_id, scenario))
        world_state_records, skipped = self._sync_world_state(
            save_id,
            message_order=message_order,
        )
        records.extend(world_state_records)
        skipped_counts["world_state"] = skipped
        memory_records, skipped = self._sync_memories(save_id)
        records.extend(memory_records)
        skipped_counts["memory"] = skipped
        summary_records, skipped = self._sync_summaries(save_id)
        records.extend(summary_records)
        skipped_counts["summary"] = skipped
        thread_records, skipped = self._sync_active_threads(save_id, characters)
        records.extend(thread_records)
        skipped_counts["active_thread"] = skipped
        records.extend(self._sync_character_text_threads(save_id, characters))
        records.extend(self._sync_locations(save_id, locations, snapshot))
        records.extend(self._sync_characters(save_id, characters, snapshot))
        self._archive_stale_index_rows(save_id, records)
        return ContinuityIndexSyncResult(
            indexed_count=len(records),
            skipped_counts=skipped_counts,
        )

    def _sync_scenario_sections(
        self,
        save_id: str,
        scenario: ScenarioRecord,
    ) -> list[ContextSourceRecord]:
        records: list[ContextSourceRecord] = []
        for source_id, section_id, text in scenario_section_candidates(scenario):
            records.append(
                self.repositories.upsert_context_source(
                    save_id=save_id,
                    source_type="scenario_section",
                    source_id=source_id,
                    title=section_id,
                    body=text,
                    metadata={
                        "indexed_by": "continuity_index",
                        "scenario_id": scenario.id,
                        "fact_type": "scenario_section",
                        "importance": 0.35,
                    },
                    token_estimate=_estimate_tokens(text),
                )
            )
        return records

    def _sync_world_state(
        self,
        save_id: str,
        *,
        message_order: dict[str, int],
    ) -> tuple[list[ContextSourceRecord], int]:
        records: list[ContextSourceRecord] = []
        active_threads_exist = has_active_thread_records(self.repositories, save_id)
        active_state = tuple(self.repositories.list_world_state(save_id))
        indexed_state = _bounded_records(
            active_state,
            self.world_state_limit,
            priority=_world_state_index_priority,
            recency=lambda state: (
                message_order.get(state.source_message_id)
                if state.source_message_id is not None
                else None
            ),
        )
        for state in indexed_state:
            if active_threads_exist and is_open_threads_aggregate_key(state.key):
                continue
            body = f"{state.key}: {_format_state_value(state.value)}"
            metadata = _metadata(
                fact_type=_world_state_fact_type(state),
                source_message_ids=(state.source_message_id,),
                importance=_world_state_importance(state),
                confidence=state.confidence,
                entity_ids=(),
                last_seen_message_id=state.source_message_id,
            )
            records.append(
                self.repositories.upsert_context_source(
                    save_id=save_id,
                    source_type="world_state",
                    source_id=state.id,
                    title=state.key,
                    body=body,
                    metadata=metadata,
                    token_estimate=_estimate_tokens(body),
                )
            )
        return records, max(0, len(active_state) - len(indexed_state))

    def _sync_memories(self, save_id: str) -> tuple[list[ContextSourceRecord], int]:
        records: list[ContextSourceRecord] = []
        active_count = self.repositories.count_active_memories(save_id)
        if self.memory_limit is None:
            indexed_memories = tuple(self.repositories.list_memories(save_id))
        elif self.memory_limit <= 0:
            indexed_memories = ()
        else:
            indexed_memories = tuple(
                self.repositories.list_memories_for_continuity_index(
                    save_id,
                    limit=self.memory_limit,
                )
            )
        selected_observation_ids = {
            observation_id
            for memory in indexed_memories
            for observation_id in memory.source_observation_ids
        }
        observations_by_id = {
            observation.id: observation
            for observation in self.repositories.list_context_observations_by_ids(
                save_id,
                selected_observation_ids,
            )
        }
        for memory in indexed_memories:
            fact_type = _memory_fact_type(memory)
            source_message_ids = tuple(_memory_source_message_ids(memory))
            provenance_groups: list[list[str]] = []
            grouped_source_ids: set[str] = set()
            for observation_id in memory.source_observation_ids:
                observation = observations_by_id.get(observation_id)
                if observation is None or not observation.source_message_ids:
                    continue
                group = list(dict.fromkeys(observation.source_message_ids))
                provenance_groups.append(group)
                grouped_source_ids.update(group)
            ungrouped_source_ids = [
                source_id
                for source_id in source_message_ids
                if source_id not in grouped_source_ids
            ]
            if ungrouped_source_ids:
                provenance_groups.append(ungrouped_source_ids)
            elif not provenance_groups and source_message_ids:
                provenance_groups.append(list(source_message_ids))
            records.append(
                self.repositories.upsert_context_source(
                    save_id=save_id,
                    source_type="memory",
                    source_id=memory.id,
                    title=", ".join(memory.tags) or "memory",
                    body=memory.body,
                    metadata=_metadata(
                        fact_type=fact_type,
                        source_message_ids=source_message_ids,
                        importance=max(
                            memory.importance,
                            _fact_type_importance(fact_type),
                        ),
                        confidence=1.0,
                        entity_ids=(),
                        last_seen_message_id=memory.source_message_id,
                    )
                    | {
                        "tags": list(memory.tags),
                        "source_provenance_groups": provenance_groups,
                        "source_provenance_mode": _memory_provenance_mode(
                            memory,
                            observations_by_id=observations_by_id,
                        ),
                    },
                    token_estimate=_estimate_tokens(memory.body),
                )
            )
        return records, max(0, active_count - len(indexed_memories))

    def _sync_summaries(self, save_id: str) -> tuple[list[ContextSourceRecord], int]:
        records: list[ContextSourceRecord] = []
        active_summaries = tuple(
            summary
            for summary in self.repositories.list_summaries(save_id)
            if validate_summary_output(summary.body).accepted
        )
        indexed_summaries = _bounded_records(
            active_summaries,
            self.summary_limit,
            priority=lambda _summary: 0.0,
        )
        for summary in indexed_summaries:
            records.append(
                self.repositories.upsert_context_source(
                    save_id=save_id,
                    source_type="summary",
                    source_id=summary.id,
                    title="summary",
                    body=summary.body,
                    metadata={
                        "indexed_by": "continuity_index",
                        "fact_type": "summary",
                        "source_message_ids": [
                            summary.covers_message_start_id,
                            summary.covers_message_end_id,
                        ],
                        "importance": 0.2,
                        "confidence": 0.65,
                        "last_seen_message_id": summary.covers_message_end_id,
                        "authoritative": False,
                    },
                    token_estimate=_estimate_tokens(summary.body),
                )
            )
        return records, max(0, len(active_summaries) - len(indexed_summaries))

    def _archive_stale_index_rows(
        self,
        save_id: str,
        active_records: list[ContextSourceRecord],
    ) -> None:
        active_keys = {
            (record.source_type, record.source_id) for record in active_records
        }
        for record in self.repositories.list_context_sources(save_id):
            if record.metadata.get("indexed_by") != "continuity_index":
                continue
            if (record.source_type, record.source_id) not in active_keys:
                self.repositories.archive_context_source(record.id)

    def _sync_active_threads(
        self,
        save_id: str,
        characters: list[CharacterRecord],
    ) -> tuple[list[ContextSourceRecord], int]:
        records: list[ContextSourceRecord] = []
        characters_by_id = {character.id: character for character in characters}
        prompt_visible_threads = tuple(
            thread
            for thread in self.repositories.list_active_threads(save_id)
            if active_thread_is_prompt_visible(thread)
        )
        indexed_threads = _bounded_records(
            prompt_visible_threads,
            self.active_thread_limit,
            priority=lambda thread: float(thread.priority),
        )
        for thread in indexed_threads:
            status = normalize_active_thread_status(thread.status)
            body = (
                f"{thread.title} ({status}, priority "
                f"{thread.priority}): {thread.description}"
            )
            metadata = _metadata(
                fact_type="open_obligation",
                source_message_ids=(thread.source_message_id,),
                importance=min(1.0, 0.55 + (thread.priority / 20)),
                confidence=1.0,
                entity_ids=(thread.id,),
                last_seen_message_id=thread.source_message_id,
            ) | {
                "status": status,
                "always_include_reason": "open obligation",
            }
            if normalize_active_thread_visibility(thread.visibility) == "private":
                audience_ids = _active_thread_audience_character_ids(
                    thread.related_entities,
                    known_character_ids=frozenset(characters_by_id),
                )
                metadata["requires_audience"] = True
                if audience_ids:
                    metadata |= {
                        "audience_character_ids": sorted(audience_ids),
                        "known_by": [
                            characters_by_id[character_id].name
                            for character_id in sorted(audience_ids)
                            if character_id in characters_by_id
                        ],
                    }
            records.append(
                self.repositories.upsert_context_source(
                    save_id=save_id,
                    source_type="open_obligation",
                    source_id=thread.id,
                    title=thread.title,
                    body=body,
                    metadata=metadata,
                    token_estimate=_estimate_tokens(body),
                )
            )
        return records, max(0, len(prompt_visible_threads) - len(indexed_threads))

    def _sync_character_text_threads(
        self,
        save_id: str,
        characters: list[CharacterRecord],
    ) -> list[ContextSourceRecord]:
        records: list[ContextSourceRecord] = []
        characters_by_id = {character.id: character for character in characters}
        for thread in self.repositories.list_character_text_threads(save_id):
            messages = tuple(
                canonical_character_text_context_messages(
                    repositories=self.repositories,
                    save_id=save_id,
                    thread_id=thread.id,
                )
            )
            body = _character_text_thread_body(
                thread,
                messages=messages,
                characters_by_id=characters_by_id,
            )
            if not body:
                continue
            audience_ids = character_text_audience_character_ids(
                repositories=self.repositories,
                save_id=save_id,
                text_messages=messages,
                thread=thread,
            )
            if not audience_ids:
                continue
            audience_names = tuple(
                characters_by_id[character_id].name
                for character_id in sorted(audience_ids)
                if character_id in characters_by_id
            )
            source_refs = tuple(
                character_text_source_ref(message.id) for message in messages
            )
            metadata = _metadata(
                fact_type="character_text_thread",
                source_message_ids=source_refs,
                importance=0.58,
                confidence=1.0,
                entity_ids=(thread.id,),
                last_seen_message_id=None,
            ) | {
                "audience_character_ids": sorted(audience_ids),
                "known_by": list(audience_names),
                "thread_id": thread.id,
            }
            records.append(
                self.repositories.upsert_context_source(
                    save_id=save_id,
                    source_type="character_text_thread",
                    source_id=thread.id,
                    title=_character_text_thread_title(
                        thread,
                        characters_by_id=characters_by_id,
                    ),
                    body=body,
                    metadata=metadata,
                    token_estimate=_estimate_tokens(body),
                )
            )
        return records

    def _sync_locations(
        self,
        save_id: str,
        locations: list[LocationRecord],
        snapshot: SceneSnapshotRecord | None,
    ) -> list[ContextSourceRecord]:
        current_location_id = snapshot.current_location_id if snapshot else None
        records: list[ContextSourceRecord] = []
        for location in locations:
            body = _location_body(location)
            if not body:
                continue
            records.append(
                self.repositories.upsert_context_source(
                    save_id=save_id,
                    source_type="world_state",
                    source_id=f"location:{location.id}",
                    title=f"location:{location.name}",
                    body=body,
                    metadata=_metadata(
                        fact_type="location",
                        source_message_ids=(location.source_message_id,),
                        importance=0.85 if location.id == current_location_id else 0.45,
                        confidence=1.0,
                        entity_ids=(location.id,),
                        last_seen_message_id=location.source_message_id,
                    ),
                    token_estimate=_estimate_tokens(body),
                )
            )
        return records

    def _sync_characters(
        self,
        save_id: str,
        characters: list[CharacterRecord],
        snapshot: SceneSnapshotRecord | None,
    ) -> list[ContextSourceRecord]:
        present_ids = set(snapshot.present_character_ids if snapshot else [])
        records: list[ContextSourceRecord] = []
        for character in characters:
            profile = _character_profile_body(character)
            if profile:
                records.append(
                    self.repositories.upsert_context_source(
                        save_id=save_id,
                        source_type="memory",
                        source_id=f"character_profile:{character.id}",
                        title=f"character:{character.name}",
                        body=profile,
                        metadata=_metadata(
                            fact_type="identity",
                            source_message_ids=(character.source_message_id,),
                            importance=0.8 if character.id in present_ids else 0.5,
                            confidence=1.0,
                            entity_ids=(character.id,),
                            last_seen_message_id=character.source_message_id,
                        ),
                        token_estimate=_estimate_tokens(profile),
                    )
                )
            voice = _character_voice_body(character)
            if voice:
                records.append(
                    self.repositories.upsert_context_source(
                        save_id=save_id,
                        source_type="character_voice",
                        source_id=character.id,
                        title=f"voice:{character.name}",
                        body=voice,
                        metadata=_metadata(
                            fact_type="character_voice",
                            source_message_ids=(character.source_message_id,),
                            importance=0.95 if character.id in present_ids else 0.7,
                            confidence=1.0,
                            entity_ids=(character.id,),
                            last_seen_message_id=character.source_message_id,
                        )
                        | {"always_include_reason": "character voice"},
                        token_estimate=_estimate_tokens(voice),
                    )
                )
            for relationship_name, relationship in sorted(
                character.relationships.items(),
                key=lambda item: item[0],
            ):
                body = (
                    f"{character.name} relationship to {relationship_name}: "
                    f"{relationship}"
                )
                records.append(
                    self.repositories.upsert_context_source(
                        save_id=save_id,
                        source_type="memory",
                        source_id=(
                            f"relationship:{character.id}:"
                            f"{_stable_key(relationship_name)}"
                        ),
                        title=f"relationship:{character.name}:{relationship_name}",
                        body=body,
                        metadata=_metadata(
                            fact_type="relationship",
                            source_message_ids=(character.source_message_id,),
                            importance=0.82 if character.id in present_ids else 0.55,
                            confidence=1.0,
                            entity_ids=(character.id,),
                            last_seen_message_id=character.source_message_id,
                        ),
                        token_estimate=_estimate_tokens(body),
                    )
                )
        return records


def _metadata(
    *,
    fact_type: str,
    source_message_ids: tuple[str | None, ...],
    importance: float,
    confidence: float,
    entity_ids: tuple[str, ...],
    last_seen_message_id: str | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "indexed_by": "continuity_index",
        "fact_type": fact_type,
        "source_message_ids": [item for item in source_message_ids if item],
        "entity_ids": list(entity_ids),
        "importance": importance,
        "confidence": confidence,
        "status": "active",
    }
    if last_seen_message_id:
        payload["last_seen_message_id"] = last_seen_message_id
    if fact_type in HIGH_VALUE_FACT_TYPES:
        payload["always_include_reason"] = fact_type
    return payload


def _bounded_records[T](
    records: tuple[T, ...],
    limit: int | None,
    *,
    priority: Callable[[T], float],
    prefer_recent: bool = False,
    recency: Callable[[T], int | None] | None = None,
) -> tuple[T, ...]:
    if limit is None:
        return records
    if limit <= 0:
        return ()
    if len(records) <= limit:
        return records
    if recency is not None:
        ranked = sorted(
            enumerate(records),
            key=lambda item: (
                -float(priority(item[1])),
                -_optional_recency_value(recency(item[1])),
                item[0],
            ),
        )
    elif not prefer_recent:
        ranked = sorted(
            enumerate(records),
            key=lambda item: (-float(priority(item[1])), item[0]),
        )
    else:
        ranked = sorted(
            enumerate(records),
            key=lambda item: (-float(priority(item[1])), -item[0]),
        )
    selected_indexes = {index for index, _record in ranked[:limit]}
    return tuple(
        record for index, record in enumerate(records) if index in selected_indexes
    )


def _optional_recency_value(value: int | None) -> int:
    return value if value is not None else -1


def _world_state_index_priority(state: WorldStateRecord) -> float:
    fact_type = _world_state_fact_type(state)
    return (10.0 if fact_type in HIGH_VALUE_FACT_TYPES else 0.0) + (
        _world_state_importance(state) * 2.0
    )


def _memory_index_priority(memory: MemoryRecord) -> float:
    fact_type = _memory_fact_type(memory)
    return (10.0 if fact_type in HIGH_VALUE_FACT_TYPES else 0.0) + (
        max(memory.importance, _fact_type_importance(fact_type)) * 2.0
    )


def _world_state_fact_type(state: WorldStateRecord) -> str:
    key = state.key.casefold()
    if "inventory" in key or "item" in key:
        return "inventory"
    if "location" in key or state.category == "location":
        return "location"
    if "promise" in key or "oath" in key or "obligation" in key:
        return "promise"
    return state.category or "world_fact"


def _world_state_importance(state: WorldStateRecord) -> float:
    fact_type = _world_state_fact_type(state)
    return max(0.45, _fact_type_importance(fact_type))


def _memory_fact_type(memory: MemoryRecord) -> str:
    tags = {tag.casefold() for tag in memory.tags}
    text = memory.body.casefold()
    if "dossier" in tags and (
        "relationship" in tags or any(tag.startswith("character:") for tag in tags)
    ):
        return "relationship"
    if tags & {"promise", "oath", "obligation", "quest", "task"}:
        return "promise"
    if any(term in text for term in ("promised", "swore", "owes", "must")):
        return "promise"
    if tags & {"relationship", "trust", "rivalry"}:
        return "relationship"
    if tags & {"voice", "speech", "diction"}:
        return "character_voice"
    if tags & {"inventory", "item", "object"}:
        return "inventory"
    return "memory"


def _memory_source_message_ids(memory: MemoryRecord) -> tuple[str, ...]:
    source_ids = list(memory.source_message_ids)
    if memory.source_message_id:
        source_ids.insert(0, memory.source_message_id)
    return tuple(dict.fromkeys(source_id for source_id in source_ids if source_id))


def _memory_provenance_mode(
    memory: MemoryRecord,
    *,
    observations_by_id: dict[str, ContextObservationRecord],
) -> str:
    memory_fingerprint = (
        memory.claim_fingerprint or canonical_claim_fingerprint(memory.body)
    )
    for observation_id in memory.source_observation_ids:
        observation = observations_by_id.get(observation_id)
        if observation is None:
            return "all"
        curation = observation.metadata.get("curation")
        curated_body = (
            curation.get("memory_body")
            if isinstance(curation, dict)
            else None
        )
        supporting_body = (
            curated_body.strip()
            if isinstance(curated_body, str) and curated_body.strip()
            else observation.claim
        )
        if canonical_claim_fingerprint(supporting_body) != memory_fingerprint:
            return "all"
    return "any"


def _fact_type_importance(fact_type: str) -> float:
    return {
        "character_voice": 0.95,
        "identity": 0.8,
        "inventory": 0.85,
        "location": 0.75,
        "open_obligation": 0.95,
        "promise": 0.9,
        "relationship": 0.82,
    }.get(fact_type, 0.45)


def _location_body(location: LocationRecord) -> str:
    parts = [f"Location {location.name}"]
    parts.extend(
        part
        for part in (
            f"aliases: {', '.join(location.aliases)}" if location.aliases else "",
            location.description,
            f"status: {location.status}" if location.status else "",
            f"hazards: {', '.join(location.hazards)}" if location.hazards else "",
        )
        if part
    )
    return "; ".join(parts)


def _character_profile_body(character: CharacterRecord) -> str:
    parts = [f"Character {character.name}"]
    parts.extend(
        part
        for part in (
            f"aliases: {', '.join(character.aliases)}" if character.aliases else "",
            f"role: {character.role}" if character.role else "",
            f"age: {character.age}" if character.age else "",
            f"known state: {character.known_state}" if character.known_state else "",
            f"status: {character.status}" if character.status else "",
            (
                "appearance: " + _compact_profile_detail(character.appearance)
                if character.appearance
                else ""
            ),
            (
                "visual notes: " + _compact_profile_detail(character.visual_notes)
                if character.visual_notes
                else ""
            ),
            (
                "current clothing: "
                + _compact_profile_detail(character.current_clothing)
                if character.current_clothing
                else ""
            ),
        )
        if part
    )
    return "; ".join(parts)


def _compact_profile_detail(
    value: str,
    max_chars: int = CHARACTER_PROFILE_DETAIL_MAX_CHARS,
) -> str:
    compacted = " ".join(value.split())
    if len(compacted) <= max_chars:
        return compacted
    marker = " ..."
    if max_chars <= len(marker):
        return compacted[:max_chars].rstrip()
    return compacted[: max_chars - len(marker)].rstrip() + marker


def _character_voice_body(character: CharacterRecord) -> str:
    parts = [
        part
        for part in (
            f"{character.name} voice: {character.voice}" if character.voice else "",
            (
                f"{character.name} personality: {character.personality}"
                if character.personality
                else ""
            ),
        )
        if part
    ]
    return "; ".join(parts)


def _active_thread_audience_character_ids(
    related_entities: tuple[str, ...] | list[str],
    *,
    known_character_ids: frozenset[str],
) -> frozenset[str]:
    character_ids: set[str] = set()
    for item in related_entities:
        if item in known_character_ids:
            character_ids.add(item)
            continue
        entity_type, separator, entity_id = item.partition(":")
        if (
            separator
            and entity_type == "character"
            and entity_id in known_character_ids
        ):
            character_ids.add(entity_id)
    return frozenset(character_ids)


def _character_text_thread_body(
    thread: CharacterTextThreadRecord,
    *,
    messages: tuple[CharacterTextMessageRecord, ...],
    characters_by_id: dict[str, CharacterRecord],
) -> str:
    lines = [
        (
            "Phone text thread: "
            + _character_text_thread_title(thread, characters_by_id=characters_by_id)
        )
    ]
    memory_body = " ".join(thread.memory_body.split())
    if memory_body:
        lines.append(
            "Thread memory: "
            + _compact_text_line(
                memory_body,
                CHARACTER_TEXT_THREAD_LINE_MAX_CHARS,
            )
        )
    recent_messages = messages[-CHARACTER_TEXT_THREAD_RECENT_MESSAGE_LIMIT:]
    if recent_messages:
        lines.append("Recent delivered text messages:")
        for message in recent_messages:
            lines.append(
                "- "
                + _compact_text_line(
                    (
                        f"{_character_text_message_time(message)} "
                        f"{_character_text_sender(message, characters_by_id)}: "
                        f"{message.body}"
                    ),
                    CHARACTER_TEXT_THREAD_LINE_MAX_CHARS,
                )
            )
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def _character_text_thread_title(
    thread: CharacterTextThreadRecord,
    *,
    characters_by_id: dict[str, CharacterRecord],
) -> str:
    if thread.character_id:
        character = characters_by_id.get(thread.character_id)
        if character is not None:
            return character.contact_name.strip() or character.name.strip()
    return thread.title.strip() or "Unknown contact"


def _character_text_sender(
    message: CharacterTextMessageRecord,
    characters_by_id: dict[str, CharacterRecord],
) -> str:
    if message.sender == "player":
        return "Player"
    for character_id in (message.sender_character_id, message.character_id):
        if not character_id:
            continue
        character = characters_by_id.get(character_id)
        if character is not None:
            return character.contact_name.strip() or character.name.strip()
    return "Character"


def _character_text_message_time(message: CharacterTextMessageRecord) -> str:
    return (
        message.in_world_sent_at
        or message.delivered_at
        or message.created_at
        or "time unknown"
    )


def _compact_text_line(value: str, limit: int) -> str:
    compacted = " ".join(value.split())
    if len(compacted) <= limit:
        return compacted
    marker = "..."
    return compacted[: max(0, limit - len(marker))].rstrip() + marker


def _format_state_value(value: object) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{key}: {item}" for key, item in value.items())
    return json.dumps(value, sort_keys=True) if isinstance(value, list) else str(value)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _stable_key(value: str) -> str:
    return "-".join(value.casefold().split())
