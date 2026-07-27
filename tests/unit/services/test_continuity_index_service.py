from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Never

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.models import ContextObservationRecord, MemoryRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.continuity_index_service import ContinuityIndexService


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_continuity_index_syncs_atomic_facts_with_evidence_metadata(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"lore": "The red lens was forged under the old tower."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    source_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="Ilyra promises to hold the east stair.",
    )
    location = repositories.add_location(
        save_id=save.id,
        name="Beacon Gallery",
        description="A high room of red glass.",
        source_message_id=source_message.id,
    )
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        role="Watch captain",
        appearance="Tall captain in a scorched blue watchcoat",
        visual_notes="Copper lens-key on a black cord",
        personality="Dry humor under pressure",
        voice="Low, clipped commands.",
        relationships={"Signal warden": "trusts them with the lens key"},
        location_id=location.id,
        source_message_id=source_message.id,
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=location.id,
        situation="The lens ticks under stress.",
        present_character_ids=[character.id],
        source_message_id=source_message.id,
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="Captain Ilyra promised to hold the east stair.",
        tags=["promise"],
        importance=0.5,
        source_message_id=source_message.id,
    )
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="inventory.lens_key",
        value={"holder": "Mara"},
        category="inventory",
        source_message_id=source_message.id,
    )
    thread = repositories.add_active_thread(
        save_id=save.id,
        title="Keep the beacon lit",
        description="The lens will fail if the copper notch is released.",
        priority=8,
        source_message_id=source_message.id,
    )

    result = ContinuityIndexService(repositories).sync_save(save.id)

    rows = repositories.list_context_sources(save.id)
    assert result.indexed_count == len(rows)
    by_source = {(row.source_type, row.source_id): row for row in rows}
    assert by_source[("memory", memory.id)].metadata["fact_type"] == "promise"
    assert by_source[("memory", memory.id)].metadata["indexed_by"] == (
        "continuity_index"
    )
    assert by_source[("world_state", state.id)].metadata["fact_type"] == "inventory"
    assert by_source[("open_obligation", thread.id)].metadata["fact_type"] == (
        "open_obligation"
    )
    assert by_source[("character_voice", character.id)].metadata[
        "always_include_reason"
    ] == "character voice"
    character_profile = by_source[("memory", f"character_profile:{character.id}")]
    assert "Tall captain in a scorched blue watchcoat" in character_profile.body
    assert "Copper lens-key on a black cord" in character_profile.body
    character_voice = by_source[("character_voice", character.id)]
    assert "Tall captain in a scorched blue watchcoat" not in character_voice.body
    assert "Copper lens-key on a black cord" not in character_voice.body
    source_message_ids = by_source[("memory", memory.id)].metadata[
        "source_message_ids"
    ]
    assert isinstance(source_message_ids, list)
    assert source_message.id in source_message_ids
    scenario_section_ids = [
        source_id
        for source_type, source_id in by_source
        if source_type == "scenario_section"
    ]
    assert scenario_section_ids == [f"scenario:{scenario.id}:section:lore"]

    repositories.archive_memory(memory.id)
    ContinuityIndexService(repositories).sync_save(save.id)

    remaining_keys = {
        (row.source_type, row.source_id)
        for row in repositories.list_context_sources(save.id)
    }
    assert ("memory", memory.id) not in remaining_keys


def test_continuity_index_skips_resync_until_source_data_changes(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    source = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The lens glows red.",
    )
    repositories.add_memory(
        save_id=save.id,
        body="The lens glows red.",
        tags=["lens"],
        source_message_id=source.id,
    )
    service = ContinuityIndexService(repositories)

    first = service.sync_save(save.id)
    second = service.sync_save(save.id)

    assert first.indexed_count > 0
    assert second.indexed_count == 0
    repositories.add_memory(
        save_id=save.id,
        body="The lens hums at dawn.",
        tags=["lens"],
        source_message_id=source.id,
    )
    assert repositories.continuity_index_needs_sync(save.id)


def test_continuity_index_updates_one_dirty_memory_without_broad_scans(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    ContinuityIndexService(repositories).sync_save(save.id)
    memory = repositories.add_memory(
        save_id=save.id,
        body="The moonstone opens the sealed archive.",
        tags=["moonstone"],
    )

    class PointOnlyRepositories(PersistenceRepositories):
        def load_save_details(
            self,
            save_id: str,
            *,
            message_limit: int | None = None,
            before_message_id: str | None = None,
        ) -> Never:
            raise AssertionError("dirty sync must not load the full save")

        def list_memories(
            self,
            save_id: str,
            *,
            limit: int | None = None,
        ) -> Never:
            raise AssertionError("dirty sync must not list every memory")

        def list_context_sources(
            self,
            save_id: str,
            *,
            source_type: str | None = None,
        ) -> Never:
            raise AssertionError("dirty sync must not list every context source")

    point_only = PointOnlyRepositories(repositories.connection)
    result = ContinuityIndexService(point_only).sync_save(save.id)

    assert result.complete
    assert result.processed_dirty_count == 1
    indexed = point_only.search_context_sources(
        save.id,
        query_terms={"moonstone"},
        source_types={"memory"},
        limit=1,
    )
    assert [hit.record.source_id for hit in indexed] == [memory.id]


def test_continuity_index_caps_dirty_work_per_sync(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    service = ContinuityIndexService(repositories)
    service.sync_save(save.id)
    for index in range(129):
        repositories.add_memory(
            save_id=save.id,
            body=f"Bounded dirty memory {index}.",
            tags=["bounded"],
            memory_id=f"bounded-memory-{index}",
        )

    first = service.sync_save(save.id)
    second = service.sync_save(save.id)

    assert not first.complete
    assert first.processed_dirty_count == 128
    assert second.complete
    assert second.processed_dirty_count == 1


def test_continuity_index_treats_consolidated_dossiers_as_high_value_context(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    source_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="Ilyra's trust in Mara has become explicit.",
    )
    dossier = repositories.add_memory(
        save_id=save.id,
        body="Captain Ilyra trusts Mara with the beacon lens.",
        tags=["dossier", "relationship", "character:ilyra"],
        importance=0.45,
        source_message_id=source_message.id,
        source_message_ids=[source_message.id],
    )
    archived = repositories.add_memory(
        save_id=save.id,
        body="Archived duplicate relationship note.",
        tags=["relationship"],
        source_message_id=source_message.id,
    )
    repositories.archive_memory(archived.id)

    ContinuityIndexService(repositories).sync_save(save.id)

    by_source = {
        (row.source_type, row.source_id): row
        for row in repositories.list_context_sources(save.id)
    }
    indexed = by_source[("memory", dossier.id)]
    assert indexed.metadata["fact_type"] == "relationship"
    assert indexed.metadata["importance"] == 0.82
    assert indexed.metadata["always_include_reason"] == "relationship"
    assert indexed.metadata["tags"] == ["dossier", "relationship", "character:ilyra"]
    assert ("memory", archived.id) not in by_source


def test_continuity_index_caps_low_value_memories_but_keeps_high_value_facts(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    source_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="Ilyra promises to hold the east stair.",
    )
    promise = repositories.add_memory(
        save_id=save.id,
        body="Captain Ilyra promised to hold the east stair.",
        tags=["promise"],
        importance=0.2,
        source_message_id=source_message.id,
        memory_id="memory-promise",
    )
    low_value = [
        repositories.add_memory(
            save_id=save.id,
            body=f"Low-value transient scene detail {index}.",
            tags=["detail"],
            importance=0.1,
            source_message_id=source_message.id,
            memory_id=f"memory-low-{index}",
        )
        for index in range(4)
    ]

    result = ContinuityIndexService(repositories, memory_limit=2).sync_save(save.id)

    indexed_memory_ids = {
        row.source_id
        for row in repositories.list_context_sources(save.id, source_type="memory")
    }
    assert promise.id in indexed_memory_ids
    assert low_value[-1].id in indexed_memory_ids
    assert low_value[0].id not in indexed_memory_ids
    assert low_value[1].id not in indexed_memory_ids
    assert low_value[2].id not in indexed_memory_ids
    assert result.skipped_counts["memory"] == 3


def test_continuity_index_bounds_memory_and_observation_hydration(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    source = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The watch records several details.",
    )
    observations = [
        repositories.add_context_observation(
            save_id=save.id,
            observation_type="world_fact",
            claim=f"Detail {index}.",
            evidence_quote="records several details",
            source_message_ids=[source.id],
            scope="durable",
            status="accepted",
            confidence=0.8,
            tags=["detail"],
        )
        for index in range(5)
    ]
    for index, observation in enumerate(observations):
        repositories.add_memory(
            save_id=save.id,
            body=f"Recorded detail {index}.",
            tags=["detail"],
            importance=float(index) / 10,
            source_message_id=source.id,
            source_observation_ids=[observation.id],
        )

    class BoundedRepositories(PersistenceRepositories):
        selected_observation_ids: set[str] = set()

        def list_memories(
            self,
            save_id: str,
            *,
            limit: int | None = None,
        ) -> list[MemoryRecord]:
            raise AssertionError("continuity sync must not load every memory")

        def list_context_observations(
            self,
            save_id: str,
            *,
            statuses: set[str] | frozenset[str] | tuple[str, ...] | None = None,
            include_archived: bool = False,
            limit: int | None = None,
        ) -> list[ContextObservationRecord]:
            raise AssertionError("continuity sync must not load every observation")

        def list_context_observations_by_ids(
            self,
            save_id: str,
            observation_ids: set[str] | frozenset[str],
        ) -> list[ContextObservationRecord]:
            self.selected_observation_ids = set(observation_ids)
            return super().list_context_observations_by_ids(
                save_id,
                observation_ids,
            )

    bounded = BoundedRepositories(repositories.connection)
    result = ContinuityIndexService(bounded, memory_limit=2).sync_save(save.id)

    assert result.skipped_counts["memory"] == 3
    assert len(bounded.selected_observation_ids) == 2
    indexed = bounded.list_context_sources(save.id, source_type="memory")
    assert len(indexed) == 2


def test_continuity_index_distinguishes_conjunctive_and_alternative_provenance(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    first_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The vault code is amber.",
    )
    second_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="The beacon is lit.",
    )
    first = repositories.add_context_observation(
        save_id=save.id,
        observation_type="world_fact",
        claim="The vault code is amber.",
        evidence_quote="The vault code is amber",
        source_message_ids=[first_message.id],
        scope="durable",
        status="accepted",
        confidence=0.9,
        tags=["vault"],
        metadata={
            "curation": {
                "action": "durable_memory",
                "memory_body": "The vault code is amber.",
            }
        },
    )
    second = repositories.add_context_observation(
        save_id=save.id,
        observation_type="world_fact",
        claim="The beacon is lit.",
        evidence_quote="The beacon is lit",
        source_message_ids=[second_message.id],
        scope="durable",
        status="accepted",
        confidence=0.9,
        tags=["beacon"],
        metadata={
            "curation": {
                "action": "durable_memory",
                "memory_body": "The beacon is lit.",
            }
        },
    )
    combined = repositories.add_memory(
        save_id=save.id,
        body="The vault code is amber and the beacon is lit.",
        tags=["vault", "beacon"],
        importance=0.9,
        source_message_ids=[first_message.id, second_message.id],
        source_observation_ids=[first.id, second.id],
    )
    duplicate_evidence = repositories.add_context_observation(
        save_id=save.id,
        observation_type="world_fact",
        claim="The beacon is lit.",
        evidence_quote="The beacon is lit",
        source_message_ids=[first_message.id],
        scope="durable",
        status="accepted",
        confidence=0.9,
        tags=["beacon"],
        metadata={
            "curation": {
                "action": "durable_memory",
                "memory_body": "The beacon is lit.",
            }
        },
    )
    repeated = repositories.add_memory(
        save_id=save.id,
        body="The beacon is lit.",
        tags=["beacon"],
        importance=0.9,
        source_message_ids=[first_message.id, second_message.id],
        source_observation_ids=[second.id, duplicate_evidence.id],
    )

    ContinuityIndexService(repositories).sync_save(save.id)

    by_source = {
        source.source_id: source
        for source in repositories.list_context_sources(
            save.id,
            source_type="memory",
        )
    }
    assert by_source[combined.id].metadata["source_provenance_mode"] == "all"
    assert by_source[repeated.id].metadata["source_provenance_mode"] == "any"


def test_continuity_index_reserves_provenance_group_for_scalar_source(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    messages = [
        repositories.append_message(
            save_id=save.id,
            role="narrator",
            body=f"Evidence {index}.",
        ).id
        for index in range(65)
    ]
    observation_ids = [
        repositories.add_context_observation(
            save_id=save.id,
            observation_type="world_fact",
            claim=f"Claim {index}.",
            source_message_ids=[messages[index]],
            status="accepted",
        ).id
        for index in range(64)
    ]
    memory = repositories.add_memory(
        save_id=save.id,
        body="Bounded provenance memory.",
        tags=["fact"],
        source_message_id=messages[64],
        source_message_ids=[messages[64]],
        source_observation_ids=observation_ids,
    )

    ContinuityIndexService(repositories).sync_save(save.id)

    indexed = next(
        source
        for source in repositories.list_context_sources(
            save.id,
            source_type="memory",
        )
        if source.source_id == memory.id
    )
    groups = indexed.metadata["source_provenance_groups"]
    assert isinstance(groups, list)
    assert len(groups) == 64
    assert any(messages[64] in group for group in groups)


def test_continuity_index_world_state_cap_prefers_recent_ties(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    old_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="Old detail: the ash gathered near the outer stair.",
    )
    recent_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="Recent detail: the inner lens started ticking.",
    )
    old_state = repositories.upsert_world_state(
        save_id=save.id,
        key="a.old_detail",
        value={"detail": "Ash gathered near the outer stair."},
        category="detail",
        source_message_id=old_message.id,
    )
    recent_state = repositories.upsert_world_state(
        save_id=save.id,
        key="z.recent_detail",
        value={"detail": "The inner lens started ticking."},
        category="detail",
        source_message_id=recent_message.id,
    )

    result = ContinuityIndexService(repositories, world_state_limit=1).sync_save(
        save.id
    )

    indexed_state_ids = {
        row.source_id
        for row in repositories.list_context_sources(save.id, source_type="world_state")
    }
    assert recent_state.id in indexed_state_ids
    assert old_state.id not in indexed_state_ids
    assert result.skipped_counts["world_state"] == 1


def test_continuity_index_world_state_cap_keeps_high_value_old_facts(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    old_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="Ilyra promises to hold the east stair.",
    )
    recent_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="Recent low-value detail: ash gathers near the blue tile.",
    )
    promise = repositories.upsert_world_state(
        save_id=save.id,
        key="a.promise.ilyra",
        value={"promise": "Ilyra will hold the east stair."},
        category="promise",
        source_message_id=old_message.id,
    )
    recent_state = repositories.upsert_world_state(
        save_id=save.id,
        key="z.recent_detail",
        value={"detail": "Ash gathers near the blue tile."},
        category="detail",
        source_message_id=recent_message.id,
    )

    ContinuityIndexService(repositories, world_state_limit=1).sync_save(save.id)

    indexed_state_ids = {
        row.source_id
        for row in repositories.list_context_sources(save.id, source_type="world_state")
    }
    assert promise.id in indexed_state_ids
    assert recent_state.id not in indexed_state_ids


def test_continuity_index_prefers_active_threads_over_aggregate_open_thread_state(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    source_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="Mara still owes Ilyra dinner after the beacon is safe.",
    )
    aggregate = repositories.upsert_world_state(
        save_id=save.id,
        key="interaction.open_threads",
        value={"dinner": "Mara owes Ilyra dinner after the beacon is safe."},
        category="open_threads",
        source_message_id=source_message.id,
    )
    stale_index = repositories.upsert_context_source(
        save_id=save.id,
        source_type="world_state",
        source_id=aggregate.id,
        title=aggregate.key,
        body="interaction.open_threads: stale aggregate",
        metadata={"indexed_by": "continuity_index", "fact_type": "open_obligation"},
    )
    thread = repositories.add_active_thread(
        save_id=save.id,
        title="Dinner promise",
        description="Mara still owes Ilyra dinner after the beacon is safe.",
        priority=4,
        source_message_id=source_message.id,
    )

    ContinuityIndexService(repositories).sync_save(save.id)

    by_source = {
        (row.source_type, row.source_id): row
        for row in repositories.list_context_sources(save.id)
    }
    assert ("open_obligation", thread.id) in by_source
    assert ("world_state", aggregate.id) not in by_source
    assert repositories.get_context_source(stale_index.id) is None


def test_continuity_index_excludes_completed_active_threads(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    source_message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="Mara settled the dinner promise before leaving the steakhouse.",
    )
    thread = repositories.add_active_thread(
        save_id=save.id,
        title="Dinner promise",
        description="Mara settled the dinner promise.",
        status="Completed",
        priority=8,
        source_message_id=source_message.id,
    )
    stale_index = repositories.upsert_context_source(
        save_id=save.id,
        source_type="open_obligation",
        source_id=thread.id,
        title=thread.title,
        body=thread.description,
        metadata={"indexed_by": "continuity_index", "fact_type": "open_obligation"},
    )

    ContinuityIndexService(repositories).sync_save(save.id)

    keys = {
        (row.source_type, row.source_id)
        for row in repositories.list_context_sources(save.id)
    }
    assert ("open_obligation", thread.id) not in keys
    assert repositories.get_context_source(stale_index.id) is None


def test_continuity_index_private_thread_with_unknown_audience_fails_closed(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    thread = repositories.add_active_thread(
        save_id=save.id,
        title="Private signal",
        description="The hidden lens code must not leave the tower.",
        priority=10,
        visibility="private",
        related_entities=["character:missing-character"],
    )

    result = ContinuityIndexService(repositories).sync_save(save.id)
    source = next(
        source
        for source in repositories.list_context_sources(save.id)
        if source.source_type == "open_obligation" and source.source_id == thread.id
    )
    hits = repositories.search_context_sources(
        save.id,
        query_terms={"lens"},
        source_types={"open_obligation"},
        limit=8,
        reference_character_ids=set(),
    )

    assert result.skipped_counts["active_thread"] == 0
    assert source.metadata["requires_audience"] is True
    assert hits == []


def test_continuity_index_does_not_cap_active_threads_before_retrieval(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    threads = [
        repositories.add_active_thread(
            save_id=save.id,
            title=f"Obligation {index:02d}",
            description=f"Resolve signal duty {index:02d}.",
            priority=100 - index,
        )
        for index in range(513)
    ]

    result = ContinuityIndexService(repositories).sync_save(save.id)
    indexed_thread_ids = {
        source.source_id
        for source in repositories.list_context_sources(save.id)
        if source.source_type == "open_obligation"
    }

    assert result.skipped_counts["active_thread"] == 0
    assert indexed_thread_ids == {thread.id for thread in threads}


def test_scene_and_base_scenario_changes_queue_exact_continuity_sources(
    repositories: PersistenceRepositories,
) -> None:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A keep in the ash.",
        player_role="Warden",
        content={"lore": "The old lens is red."},
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Night Watch")
    old_location = repositories.add_location(
        save_id=save.id,
        name="Old Gallery",
    )
    new_location = repositories.add_location(
        save_id=save.id,
        name="New Gallery",
    )
    old_character = repositories.add_character(
        save_id=save.id,
        name="Old Warden",
    )
    new_character = repositories.add_character(
        save_id=save.id,
        name="New Warden",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=old_location.id,
        present_character_ids=[old_character.id],
    )
    while not ContinuityIndexService(repositories).sync_save(save.id).complete:
        pass

    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=new_location.id,
        present_character_ids=[new_character.id],
    )
    dirty = {
        (source_kind, source_id)
        for source_kind, source_id, _generation in (
            repositories.list_continuity_index_dirty_sources(save.id, limit=32)
        )
    }
    assert {
        ("location", old_location.id),
        ("location", new_location.id),
        ("character", old_character.id),
        ("character", new_character.id),
    } <= dirty

    while not ContinuityIndexService(repositories).sync_save(save.id).complete:
        pass
    repositories.update_scenario(
        scenario_id=scenario.id,
        title=scenario.title,
        premise=scenario.premise,
        player_role=scenario.player_role,
        content={"lore": "The repaired lens is blue."},
    )

    assert (
        "scenario",
        "scenario",
    ) in {
        (source_kind, source_id)
        for source_kind, source_id, _generation in (
            repositories.list_continuity_index_dirty_sources(save.id, limit=8)
        )
    }
