from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.models import (
    CharacterKnowledgeEdgeRecord,
    CharacterRecord,
    EntityLinkRecord,
    LocationRecord,
    MemoryRecord,
    SummaryRecord,
    WorldStateRecord,
)
from bragi.persistence.repositories import PersistenceRepositories
from bragi.safety import CONTENT_FILTER_TRANSITION
from bragi.services.character_registry_service import (
    CharacterKnowledgeAction,
    CharacterRegistryEdits,
    CharacterRegistryRow,
    CharacterRegistryService,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


class CountingPersistenceRepositories(PersistenceRepositories):
    def __init__(self, connection: sqlite3.Connection) -> None:
        super().__init__(connection)
        self.list_counts: dict[str, int] = {}

    def list_characters(self, save_id: str) -> list[CharacterRecord]:
        self.list_counts["characters"] = self.list_counts.get("characters", 0) + 1
        return super().list_characters(save_id)

    def list_locations(self, save_id: str) -> list[LocationRecord]:
        self.list_counts["locations"] = self.list_counts.get("locations", 0) + 1
        return super().list_locations(save_id)

    def list_memories(
        self,
        save_id: str,
        *,
        limit: int | None = None,
    ) -> list[MemoryRecord]:
        self.list_counts["memories"] = self.list_counts.get("memories", 0) + 1
        return super().list_memories(save_id, limit=limit)

    def list_world_state(
        self,
        save_id: str,
        *,
        limit: int | None = None,
    ) -> list[WorldStateRecord]:
        self.list_counts["world_state"] = self.list_counts.get("world_state", 0) + 1
        return super().list_world_state(save_id, limit=limit)

    def list_summaries(
        self,
        save_id: str,
        *,
        limit: int | None = None,
    ) -> list[SummaryRecord]:
        self.list_counts["summaries"] = self.list_counts.get("summaries", 0) + 1
        return super().list_summaries(save_id, limit=limit)

    def list_entity_links(self, save_id: str) -> list[EntityLinkRecord]:
        self.list_counts["entity_links"] = self.list_counts.get("entity_links", 0) + 1
        return super().list_entity_links(save_id)

    def list_character_knowledge_edges(
        self,
        save_id: str,
        *,
        character_ids: set[str] | frozenset[str] | tuple[str, ...] | None = None,
        include_archived: bool = False,
    ) -> list[CharacterKnowledgeEdgeRecord]:
        self.list_counts["character_knowledge_edges"] = (
            self.list_counts.get("character_knowledge_edges", 0) + 1
        )
        return super().list_character_knowledge_edges(
            save_id,
            character_ids=character_ids,
            include_archived=include_archived,
        )


def test_build_model_ignores_retired_save_level_character_reference(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    repositories.add_character(save_id=save.id, name="First Character")
    repositories.add_character(save_id=save.id, name="Second Character")
    reference = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=None,
        type="image",
        path="references/legacy.png",
        thumbnail_path=None,
        prompt="Legacy save-level reference",
        provider="local",
        model="upload",
        status="succeeded",
        mime_type="image/png",
        metadata={"kind": "character_reference"},
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="save",
        entity_id=save.id,
        target_type="media_asset",
        target_id=reference.id,
        relation="reference_image",
    )

    model = CharacterRegistryService(repositories).build_model(
        active_save_id=save.id,
    )

    assert [row.reference_image for row in model.characters] == [None, None]


def test_build_model_replaces_character_above_viewer_rating(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    character = repositories.add_character(
        save_id=save.id,
        name="The Ash Warden",
        appearance="A lingering graphic injury.",
        relationships={"player": "A prolonged frightening threat."},
        content_rating="r",
    )

    model = CharacterRegistryService(
        repositories,
        allowed_content_rating="pg",
    ).build_model(active_save_id=save.id)

    row = next(item for item in model.characters if item.character_id == character.id)
    assert row.name == CONTENT_FILTER_TRANSITION
    assert row.appearance == CONTENT_FILTER_TRANSITION
    assert row.relationships_json == "{}"
    assert row.content_rating == "r"


def test_build_model_exposes_characters_presence_locations_and_known_links(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    gallery = repositories.add_location(
        save_id=save.id,
        name="Beacon Gallery",
        location_id="location-beacon",
    )
    first_message = repositories.append_message(
        save_id=save.id,
        role="player",
        body="I check the lens.",
        message_id="message-first",
    )
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        aliases=["Ashknife", "Glass-Eye"],
        role="Watch captain",
        known_state="Guarding the cracked red lens",
        met=True,
        relationships={"Mara": "trusts her with the lens key"},
        goals="Keep the red lens under control until dawn.",
        motivations="Protect the lower village from ash riders.",
        current_intent="Demand proof before sharing the failsafe.",
        boundaries="Will not abandon the tower.",
        attitude_toward_player="Wary trust after the last repair.",
        cooperation_conditions="Shares the failsafe after seeing the brass warrant.",
        status="present",
        location_id=gallery.id,
        private_notes="Conceals the lens-key lineage.",
        source_message_id=first_message.id,
        locked_fields=["status"],
        character_id="character-ilyra",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=gallery.id,
        present_character_ids=[character.id],
        snapshot_id="snapshot-beacon",
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="Ilyra knows the lens-key phrase.",
        tags=["ilyra"],
        memory_id="memory-lens-key",
    )
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="beacon.lens",
        value={"failsafe": "copper notch"},
        state_id="world-state-lens",
    )
    summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=first_message.id,
        covers_message_end_id=first_message.id,
        body="Ilyra warned Mara about the red lens.",
        provider="fake",
        model="fake-summary",
        summary_id="summary-lens",
    )
    for target_type, target_id in (
        ("memory", memory.id),
        ("state", state.id),
        ("summary", summary.id),
    ):
        repositories.add_entity_link(
            save_id=save.id,
            entity_type="character",
            entity_id=character.id,
            target_type=target_type,
            target_id=target_id,
            relation="knows",
        )
    reference = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=None,
        type="image",
        path="references/ilyra.png",
        thumbnail_path=None,
        prompt="Uploaded character reference image",
        provider="local",
        model="upload",
        status="succeeded",
        mime_type="image/png",
        metadata={"kind": "character_reference", "source": "uploaded"},
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=character.id,
        target_type="media_asset",
        target_id=reference.id,
        relation="reference_image",
    )
    source_linked = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=None,
        type="image",
        path="generated/ilyra-source.png",
        thumbnail_path=None,
        prompt="Ilyra source-linked picture",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={"kind": "character_image"},
        source_media_asset_id=reference.id,
    )
    metadata_linked = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=None,
        type="image",
        path="generated/ilyra-metadata.png",
        thumbnail_path=None,
        prompt="Ilyra metadata-linked picture",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={"kind": "character_image", "character_id": character.id},
    )
    repositories.connection.execute(
        "UPDATE media_assets SET created_at = ? WHERE id = ?",
        ("2025-01-01T00:00:00+00:00", source_linked.id),
    )
    repositories.connection.execute(
        "UPDATE media_assets SET created_at = ? WHERE id = ?",
        ("2025-01-02T00:00:00+00:00", metadata_linked.id),
    )
    repositories.commit()

    model = CharacterRegistryService(repositories).build_model(
        active_save_id=save.id,
    )

    assert model.error is None
    assert model.save_id == save.id
    assert model.location_choices == ((gallery.id, "Beacon Gallery"),)
    assert len(model.characters) == 1
    row = model.characters[0]
    assert row.character_id == character.id
    assert row.aliases_text == "Ashknife, Glass-Eye"
    assert row.relationships_json == '{"Mara":"trusts her with the lens key"}'
    assert row.goals == "Keep the red lens under control until dawn."
    assert row.motivations == "Protect the lower village from ash riders."
    assert row.current_intent == "Demand proof before sharing the failsafe."
    assert row.boundaries == "Will not abandon the tower."
    assert row.attitude_toward_player == "Wary trust after the last repair."
    assert row.cooperation_conditions == (
        "Shares the failsafe after seeing the brass warrant."
    )
    assert row.present is True
    assert row.linked_memory_ids == (memory.id,)
    assert row.linked_state_ids == (state.id,)
    assert row.linked_summary_ids == (summary.id,)
    assert row.source_message_id == first_message.id
    assert row.locked_fields == ("status",)
    assert row.is_player_character is False
    assert row.reference_image is not None
    assert row.reference_image.media_asset_id == reference.id
    assert row.reference_image.source == "uploaded"
    assert row.reference_image.prompt_preview == "Uploaded character reference image"
    assert [image.media_asset_id for image in row.generated_images] == [
        metadata_linked.id,
        source_linked.id,
    ]
    assert {
        (target.target_type, target.target_id, target.linked_character_ids)
        for target in model.link_targets
    } == {
        ("memory", memory.id, (character.id,)),
        ("world_state", state.id, (character.id,)),
        ("summary", summary.id, (character.id,)),
    }
    world_state_target = next(
        target for target in model.link_targets if target.target_id == state.id
    )
    assert world_state_target.value == {"failsafe": "copper notch"}
    assert world_state_target.category == ""
    assert world_state_target.confidence == 1.0
    memory_target = next(
        target for target in model.link_targets if target.target_id == memory.id
    )
    assert memory_target.tags == ("ilyra",)
    assert memory_target.importance == 1.0


def test_build_model_shows_previous_reference_as_character_image_after_swap(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        character_id="character-ilyra",
    )
    previous_reference = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=None,
        type="image",
        path="references/ilyra-old.png",
        thumbnail_path=None,
        prompt="Old Ilyra reference",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={"kind": "character_reference", "character_id": character.id},
    )
    current_reference = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=None,
        type="image",
        path="generated/ilyra-current.png",
        thumbnail_path=None,
        prompt="New Ilyra portrait",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={"kind": "character_image", "character_id": character.id},
    )
    normal_image = repositories.create_media_asset(
        save_id=save.id,
        source_message_id=None,
        type="image",
        path="generated/ilyra-normal.png",
        thumbnail_path=None,
        prompt="Ilyra in the beacon room",
        provider="fake",
        model="fake-image",
        status="succeeded",
        metadata={"kind": "character_image", "character_id": character.id},
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=character.id,
        target_type="media_asset",
        target_id=current_reference.id,
        relation="reference_image",
    )
    repositories.connection.execute(
        "UPDATE media_assets SET created_at = ? WHERE id = ?",
        ("2025-01-03T00:00:00+00:00", previous_reference.id),
    )
    repositories.connection.execute(
        "UPDATE media_assets SET created_at = ? WHERE id = ?",
        ("2025-01-02T00:00:00+00:00", current_reference.id),
    )
    repositories.connection.execute(
        "UPDATE media_assets SET created_at = ? WHERE id = ?",
        ("2025-01-01T00:00:00+00:00", normal_image.id),
    )
    repositories.commit()

    model = CharacterRegistryService(repositories).build_model(
        active_save_id=save.id,
    )

    row = model.characters[0]
    assert row.reference_image is not None
    assert row.reference_image.media_asset_id == current_reference.id
    assert [image.media_asset_id for image in row.generated_images] == [
        previous_reference.id,
        normal_image.id,
    ]


def test_build_model_uses_character_knowledge_edges_as_linked_knowledge(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    message = repositories.append_message(
        save_id=save.id,
        role="narrator",
        body="Ilyra studies the lens.",
        message_id="message-knowledge",
    )
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        character_id="character-ilyra",
    )
    known_memory = repositories.add_memory(
        save_id=save.id,
        body="Ilyra knows the lens key phrase.",
        tags=["ilyra"],
        memory_id="memory-known",
    )
    likely_memory = repositories.add_memory(
        save_id=save.id,
        body="Ilyra likely knows Mara carries the key.",
        tags=["ilyra", "mara"],
        memory_id="memory-likely",
    )
    uncertain_memory = repositories.add_memory(
        save_id=save.id,
        body="Ilyra might know an unverified rumor.",
        tags=["ilyra", "rumor"],
        memory_id="memory-uncertain",
    )
    hidden_memory = repositories.add_memory(
        save_id=save.id,
        body="Ilyra does not know the sealed vault phrase.",
        tags=["ilyra", "secret"],
        memory_id="memory-hidden",
    )
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="beacon.lens",
        value={"phrase": "ember dawn"},
        state_id="state-lens",
    )
    summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=message.id,
        covers_message_end_id=message.id,
        body="Ilyra learned how the red lens responds.",
        provider="fake",
        model="fake-summary",
        summary_id="summary-lens",
    )
    archived_edge = repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=character.id,
        target_type="memory",
        target_id=hidden_memory.id,
        knowledge_state="knows",
        acquisition_method="witnessed",
        confidence=1.0,
    )
    repositories.archive_character_knowledge_edge(archived_edge.id)
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=character.id,
        target_type="memory",
        target_id=known_memory.id,
        knowledge_state="knows",
        acquisition_method="witnessed",
        confidence=1.0,
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=character.id,
        target_type="memory",
        target_id=likely_memory.id,
        knowledge_state="may_know",
        acquisition_method="inferred_from_visible_consequence",
        confidence=0.75,
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=character.id,
        target_type="memory",
        target_id=uncertain_memory.id,
        knowledge_state="may_know",
        acquisition_method="unknown",
        confidence=0.69,
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=character.id,
        target_type="memory",
        target_id=hidden_memory.id,
        knowledge_state="does_not_know",
        acquisition_method="unknown",
        confidence=1.0,
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=character.id,
        target_type="memory",
        target_id=hidden_memory.id,
        relation="knows",
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=character.id,
        target_type="state",
        target_id=state.id,
        knowledge_state="knows",
        acquisition_method="told",
        confidence=0.9,
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=character.id,
        target_type="summary",
        target_id=summary.id,
        knowledge_state="knows",
        acquisition_method="background",
        confidence=1.0,
    )

    model = CharacterRegistryService(repositories).build_model(active_save_id=save.id)

    row = model.characters[0]
    assert set(row.linked_memory_ids) == {known_memory.id, likely_memory.id}
    assert row.linked_state_ids == (state.id,)
    assert row.linked_summary_ids == (summary.id,)
    targets = {
        (target.target_type, target.target_id): target
        for target in model.link_targets
    }
    assert targets[("memory", known_memory.id)].linked_character_ids == (character.id,)
    assert targets[("memory", likely_memory.id)].linked_character_ids == (
        character.id,
    )
    assert targets[("memory", uncertain_memory.id)].linked_character_ids == ()
    assert targets[("memory", hidden_memory.id)].linked_character_ids == ()
    assert targets[("world_state", state.id)].linked_character_ids == (character.id,)
    assert targets[("summary", summary.id)].linked_character_ids == (character.id,)


def test_build_model_uses_bounded_world_data_list_calls(
    repositories: PersistenceRepositories,
) -> None:
    counting = CountingPersistenceRepositories(repositories.connection)
    save = _create_save(counting)
    message = counting.append_message(
        save_id=save.id,
        role="player",
        body="I check the lens.",
    )
    characters = tuple(
        counting.add_character(
            save_id=save.id,
            name=f"Keeper {index}",
            character_id=f"character-{index}",
        )
        for index in range(4)
    )
    memories = tuple(
        counting.add_memory(
            save_id=save.id,
            body=f"Keeper memory {index}.",
            tags=["keeper"],
            memory_id=f"memory-{index}",
        )
        for index in range(3)
    )
    states = tuple(
        counting.upsert_world_state(
            save_id=save.id,
            key=f"beacon.lens.{index}",
            value={"index": index},
            state_id=f"state-{index}",
        )
        for index in range(3)
    )
    summary = counting.add_summary(
        save_id=save.id,
        covers_message_start_id=message.id,
        covers_message_end_id=message.id,
        body="The keepers know the lens.",
        provider="fake",
        model="fake-summary",
        summary_id="summary-lens",
    )
    for character in characters:
        for target_type, target_id in (
            ("memory", memories[0].id),
            ("world_state", states[0].id),
            ("summary", summary.id),
        ):
            counting.add_entity_link(
                save_id=save.id,
                entity_type="character",
                entity_id=character.id,
                target_type=target_type,
                target_id=target_id,
                relation="knows",
            )
    counting.list_counts.clear()

    model = CharacterRegistryService(counting).build_model(active_save_id=save.id)

    assert len(model.characters) == 4
    assert len(model.link_targets) == 7
    assert counting.list_counts == {
        "characters": 1,
        "locations": 1,
        "memories": 1,
        "world_state": 1,
        "summaries": 1,
        "entity_links": 1,
        "character_knowledge_edges": 1,
    }


def test_apply_edits_creates_updates_archives_and_replaces_known_links(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    gallery = repositories.add_location(save_id=save.id, name="Beacon Gallery")
    existing = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        aliases=["Ashknife"],
        role="Watch captain",
        goals="Hold the shattered gallery.",
        status="wounded",
        location_id=gallery.id,
        character_id="character-ilyra",
    )
    archived = repositories.add_character(
        save_id=save.id,
        name="Archivist Ren",
        character_id="character-ren",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=gallery.id,
        present_character_ids=[existing.id, archived.id],
    )
    first_message = repositories.append_message(
        save_id=save.id,
        role="player",
        body="I ask what Ilyra knows.",
    )
    old_memory = repositories.add_memory(
        save_id=save.id,
        body="Old Ilyra fact.",
        tags=[],
    )
    new_memory = repositories.add_memory(
        save_id=save.id,
        body="Ilyra knows the updated lens-key phrase.",
        tags=[],
    )
    hidden_memory = repositories.add_memory(
        save_id=save.id,
        body="Ilyra does not know Mara hid a spare key.",
        tags=[],
    )
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="beacon.lens",
        value={"failsafe": "copper notch"},
    )
    summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=first_message.id,
        covers_message_end_id=first_message.id,
        body="Ilyra explained the beacon problem.",
        provider="fake",
        model="fake-summary",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=existing.id,
        target_type="memory",
        target_id=old_memory.id,
        relation="knows",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=archived.id,
        target_type="memory",
        target_id=old_memory.id,
        relation="knows",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="memory",
        entity_id=old_memory.id,
        target_type="character",
        target_id=archived.id,
        relation="mentions",
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=existing.id,
        target_type="memory",
        target_id=hidden_memory.id,
        knowledge_state="does_not_know",
        acquisition_method="unknown",
        confidence=1.0,
    )

    result = CharacterRegistryService(repositories).apply_edits(
        CharacterRegistryEdits(
            characters=(
                CharacterRegistryRow(
                    character_id=existing.id,
                    name=" Captain Ilyra ",
                    aliases_text="Ashknife, Glass-Eye",
                    role="Watch captain",
                    age="late 40s",
                    goals="Keep the red lens under control until dawn.",
                    current_intent="Recover while guarding the lens.",
                    status="recovering",
                    location_id=gallery.id,
                    present=False,
                    linked_memory_ids=(new_memory.id,),
                    linked_state_ids=(state.id,),
                    linked_summary_ids=(summary.id,),
                ),
                CharacterRegistryRow(
                    character_id="",
                    name="Signal Warden",
                    role="Player ally",
                    met=True,
                    present=True,
                    linked_memory_ids=(new_memory.id,),
                ),
                CharacterRegistryRow(
                    character_id=archived.id,
                    name="Archivist Ren",
                    archived=True,
                ),
            ),
        ),
        active_save_id=save.id,
    )

    assert result.created_count == 1
    assert result.updated_count == 1
    assert result.archived_count == 1

    updated = repositories.get_character(existing.id)
    assert updated is not None
    assert updated.aliases == ["Ashknife", "Glass-Eye"]
    assert updated.age == "late 40s"
    assert updated.goals == "Keep the red lens under control until dawn."
    assert updated.current_intent == "Recover while guarding the lens."
    assert updated.status == "recovering"
    assert set(updated.locked_fields) >= {
        "aliases",
        "age",
        "goals",
        "current_intent",
        "status",
    }
    assert repositories.get_character(archived.id) is None

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert existing.id not in snapshot.present_character_ids
    assert archived.id not in snapshot.present_character_ids
    present_names = {
        character.name
        for character in repositories.list_characters(save.id)
        if character.id in snapshot.present_character_ids
    }
    assert present_names == {"Signal Warden"}

    new_character_id = next(
        character.id
        for character in repositories.list_characters(save.id)
        if character.name == "Signal Warden"
    )
    links = repositories.list_entity_links(save.id)
    assert {
        (link.entity_id, link.target_type, link.target_id, link.relation)
        for link in links
    } == {
        (existing.id, "memory", new_memory.id, "knows"),
        (existing.id, "world_state", state.id, "knows"),
        (existing.id, "summary", summary.id, "knows"),
        (new_character_id, "memory", new_memory.id, "knows"),
    }
    negative_edges = [
        edge
        for edge in repositories.list_character_knowledge_edges(save.id)
        if edge.target_id == hidden_memory.id
    ]
    assert len(negative_edges) == 1
    assert negative_edges[0].knowledge_state == "does_not_know"
    assert negative_edges[0].archived_at is None


def test_apply_edits_does_not_materialize_legacy_links_for_edge_derived_knowledge(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        role="Watch captain",
        character_id="character-ilyra",
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="Ilyra knows the lens phrase.",
        tags=["ilyra"],
        memory_id="memory-lens",
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=character.id,
        target_type="memory",
        target_id=memory.id,
        knowledge_state="knows",
        acquisition_method="witnessed",
        confidence=1.0,
    )

    result = CharacterRegistryService(repositories).apply_edits(
        CharacterRegistryEdits(
            characters=(
                CharacterRegistryRow(
                    character_id=character.id,
                    name=character.name,
                    role="Signal captain",
                    linked_memory_ids=(memory.id,),
                ),
            ),
        ),
        active_save_id=save.id,
    )

    assert result.updated_count == 1
    updated = repositories.get_character(character.id)
    assert updated is not None
    assert updated.role == "Signal captain"
    assert repositories.list_entity_links(save.id) == []
    edges = repositories.list_character_knowledge_edges(save.id)
    assert len(edges) == 1
    assert edges[0].target_id == memory.id
    assert result.model.characters[0].linked_memory_ids == (memory.id,)


def test_apply_knowledge_actions_creates_updates_and_links_character_knowledge(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        character_id="character-ilyra",
    )
    old_memory = repositories.add_memory(
        save_id=save.id,
        body="Ilyra knows an outdated lens phrase.",
        tags=["old"],
        memory_id="memory-old",
    )
    summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=repositories.append_message(
            save_id=save.id,
            role="player",
            body="I ask Ilyra what she knows.",
            message_id="message-first",
        ).id,
        covers_message_end_id="message-first",
        body="Ilyra warned Mara about the red lens.",
        provider="fake",
        model="fake-summary",
        summary_id="summary-lens",
    )
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="beacon.lens",
        value={"failsafe": "copper notch"},
        category="artifact",
        confidence=0.7,
        state_id="state-lens",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=character.id,
        target_type="memory",
        target_id=old_memory.id,
        relation="knows",
    )
    repositories.add_character_knowledge_edge(
        save_id=save.id,
        character_id=character.id,
        target_type="memory",
        target_id=old_memory.id,
        knowledge_state="knows",
        acquisition_method="manual",
        confidence=1.0,
    )

    result = CharacterRegistryService(repositories).apply_knowledge_actions(
        character_id=character.id,
        actions=(
            CharacterKnowledgeAction(
                action="unlink",
                target_type="memory",
                target_id=old_memory.id,
            ),
            CharacterKnowledgeAction(
                action="link",
                target_type="summary",
                target_id=summary.id,
            ),
            CharacterKnowledgeAction(
                action="create_memory",
                body="Ilyra knows Mara carries the lens key.",
                tags=("ilyra", "mara"),
                importance=0.65,
            ),
            CharacterKnowledgeAction(
                action="update_world_state",
                state_id=state.id,
                key="beacon.lens",
                category="artifact",
                confidence=0.92,
                value={"failsafe": "copper notch", "keeper": "Mara"},
            ),
            CharacterKnowledgeAction(
                action="create_world_state",
                key="character.ilyra.revealed_traits.about_mara",
                category="character",
                confidence=0.8,
                value={"text": "Ilyra knows Mara carries the lens key."},
            ),
        ),
        active_save_id=save.id,
    )

    assert result.created_count == 2
    assert result.updated_count == 1
    assert result.archived_count == 0
    assert result.model.error is None
    new_memory = next(
        memory
        for memory in repositories.list_memories(save.id)
        if memory.body == "Ilyra knows Mara carries the lens key."
    )
    assert new_memory.tags == ["ilyra", "mara"]
    assert new_memory.importance == 0.65
    updated_state = next(
        row for row in repositories.list_world_state(save.id) if row.id == state.id
    )
    assert updated_state.value == {"failsafe": "copper notch", "keeper": "Mara"}
    created_state = next(
        row
        for row in repositories.list_world_state(save.id)
        if row.key == "character.ilyra.revealed_traits.about_mara"
    )
    assert {
        (link.target_type, link.target_id, link.relation)
        for link in repositories.list_entity_links(save.id)
        if link.entity_type == "character" and link.entity_id == character.id
    } == {
        ("summary", summary.id, "knows"),
        ("memory", new_memory.id, "knows"),
        ("world_state", state.id, "knows"),
        ("world_state", created_state.id, "knows"),
    }
    assert {
        (
            edge.target_type,
            edge.target_id,
            edge.knowledge_state,
            edge.acquisition_method,
        )
        for edge in repositories.list_character_knowledge_edges(save.id)
        if edge.character_id == character.id
    } == {
        ("summary", summary.id, "knows", "manual"),
        ("memory", new_memory.id, "knows", "manual"),
        ("world_state", state.id, "knows", "manual"),
        ("world_state", created_state.id, "knows", "manual"),
    }
    archived_edges = repositories.list_character_knowledge_edges(
        save.id,
        include_archived=True,
    )
    assert any(
        edge.target_id == old_memory.id and edge.archived_at is not None
        for edge in archived_edges
    )


def test_apply_knowledge_actions_archives_matching_edges_when_targets_are_archived(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        character_id="character-ilyra",
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="Ilyra knows an obsolete phrase.",
        tags=["ilyra"],
        memory_id="memory-obsolete",
    )
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="beacon.obsolete",
        value={"phrase": "old dawn"},
        state_id="state-obsolete",
    )
    for target_type, target_id in (("memory", memory.id), ("world_state", state.id)):
        repositories.add_entity_link(
            save_id=save.id,
            entity_type="character",
            entity_id=character.id,
            target_type=target_type,
            target_id=target_id,
            relation="knows",
        )
        repositories.add_character_knowledge_edge(
            save_id=save.id,
            character_id=character.id,
            target_type=target_type,
            target_id=target_id,
            knowledge_state="knows",
            acquisition_method="manual",
            confidence=1.0,
        )

    result = CharacterRegistryService(repositories).apply_knowledge_actions(
        character_id=character.id,
        actions=(
            CharacterKnowledgeAction(
                action="update_memory",
                memory_id=memory.id,
                archived=True,
            ),
            CharacterKnowledgeAction(
                action="update_world_state",
                state_id=state.id,
                archived=True,
            ),
        ),
        active_save_id=save.id,
    )

    assert result.archived_count == 2
    assert repositories.list_entity_links(save.id) == []
    assert repositories.list_character_knowledge_edges(save.id) == []
    archived_targets = {
        (edge.target_type, edge.target_id)
        for edge in repositories.list_character_knowledge_edges(
            save.id,
            include_archived=True,
        )
        if edge.archived_at is not None
    }
    assert archived_targets == {("memory", memory.id), ("world_state", state.id)}


def test_apply_knowledge_actions_rejects_foreign_targets(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories, title="Active Save")
    other_save = _create_save(repositories, title="Other Save")
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        character_id="character-ilyra",
    )
    foreign_memory = repositories.add_memory(
        save_id=other_save.id,
        body="Foreign knowledge.",
        tags=[],
        memory_id="memory-foreign",
    )

    with pytest.raises(ValueError, match="linked target"):
        CharacterRegistryService(repositories).apply_knowledge_actions(
            character_id=character.id,
            actions=(
                CharacterKnowledgeAction(
                    action="link",
                    target_type="memory",
                    target_id=foreign_memory.id,
                ),
            ),
            active_save_id=save.id,
        )

    assert repositories.list_entity_links(save.id) == []


def test_apply_edits_uses_explicit_locked_fields_as_player_selected_fact_locks(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        aliases=["Ashknife"],
        status="present",
        locked_fields=["aliases_text", "voice", "archive"],
        character_id="character-ilyra",
    )

    CharacterRegistryService(repositories).apply_edits(
        CharacterRegistryEdits(
            characters=(
                CharacterRegistryRow(
                    character_id=character.id,
                    name="Captain Ilyra",
                    aliases_text="Ashknife, Glass-Eye",
                    status="recovering",
                    relationships_json="{}",
                    locked_fields=("appearance", "relationships_json"),
                ),
            ),
        ),
        active_save_id=save.id,
    )

    updated = repositories.get_character(character.id)
    assert updated is not None
    assert updated.aliases == ["Ashknife", "Glass-Eye"]
    assert updated.status == "recovering"
    assert updated.locked_fields == ["appearance", "archive", "relationships"]


def test_apply_edits_omitted_locked_fields_preserves_auto_locking(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        aliases=["Ashknife"],
        status="present",
        locked_fields=["aliases_text"],
        character_id="character-ilyra",
    )

    CharacterRegistryService(repositories).apply_edits(
        CharacterRegistryEdits(
            characters=(
                CharacterRegistryRow(
                    character_id=character.id,
                    name="Captain Ilyra",
                    aliases_text="Ashknife, Glass-Eye",
                    status="recovering",
                    relationships_json="{}",
                ),
            ),
        ),
        active_save_id=save.id,
    )

    updated = repositories.get_character(character.id)
    assert updated is not None
    assert updated.locked_fields == ["aliases", "status"]


def test_apply_edits_persists_present_as_explicit_character_lock(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        character_id="character-ilyra",
    )

    CharacterRegistryService(repositories).apply_edits(
        CharacterRegistryEdits(
            characters=(
                CharacterRegistryRow(
                    character_id=character.id,
                    name=character.name,
                    present=True,
                    locked_fields=("present",),
                ),
            ),
        ),
        active_save_id=save.id,
    )

    updated = repositories.get_character(character.id)
    snapshot = repositories.get_scene_snapshot(save.id)
    assert updated is not None
    assert updated.locked_fields == ["present"]
    assert snapshot is not None
    assert character.id in snapshot.present_character_ids

    CharacterRegistryService(repositories).apply_edits(
        CharacterRegistryEdits(
            characters=(
                CharacterRegistryRow(
                    character_id=character.id,
                    name=character.name,
                    present=False,
                    locked_fields=("present",),
                ),
            ),
        ),
        active_save_id=save.id,
    )

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert character.id not in snapshot.present_character_ids
    updated = repositories.get_character(character.id)
    assert updated is not None
    assert updated.locked_fields == ["present"]


def test_apply_edits_rejects_blank_existing_character_name(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        character_id="character-ilyra",
    )

    with pytest.raises(ValueError, match="Character name must not be blank"):
        CharacterRegistryService(repositories).apply_edits(
            CharacterRegistryEdits(
                characters=(
                    CharacterRegistryRow(
                        character_id=character.id,
                        name="   ",
                    ),
                ),
            ),
            active_save_id=save.id,
        )

    unchanged = repositories.get_character(character.id)
    assert unchanged is not None
    assert unchanged.name == "Captain Ilyra"


def test_apply_edits_persists_character_contact_name_on_create_and_update(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    service = CharacterRegistryService(repositories)

    create_result = service.apply_edits(
        CharacterRegistryEdits(
            characters=(
                CharacterRegistryRow(
                    character_id="",
                    name="Captain Ilyra",
                    role="Watch captain",
                    contact_name="Ily",
                ),
            ),
        ),
        active_save_id=save.id,
    )
    created_id = create_result.created_character_ids[0]
    created = repositories.get_character(created_id)
    assert created is not None
    assert created.contact_name == "Ily"

    update_result = service.apply_edits(
        CharacterRegistryEdits(
            characters=(
                CharacterRegistryRow(
                    character_id=created_id,
                    name="Captain Ilyra",
                    role="Watch captain",
                    contact_name="  Glass-Eye  ",
                ),
            ),
        ),
        active_save_id=save.id,
    )
    assert update_result.updated_count == 1
    updated = repositories.get_character(created_id)
    assert updated is not None
    assert updated.contact_name == "Glass-Eye"

    clear_result = service.apply_edits(
        CharacterRegistryEdits(
            characters=(
                CharacterRegistryRow(
                    character_id=created_id,
                    name="Captain Ilyra",
                    role="Watch captain",
                    contact_name="",
                ),
            ),
        ),
        active_save_id=save.id,
    )
    assert clear_result.updated_count == 1
    cleared = repositories.get_character(created_id)
    assert cleared is not None
    assert cleared.contact_name == ""


def test_build_model_exposes_character_contact_name(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        contact_name="Glass-Eye",
        character_id="character-ilyra",
    )

    model = CharacterRegistryService(repositories).build_model(active_save_id=save.id)

    row = next(row for row in model.characters if row.character_id == character.id)
    assert row.contact_name == "Glass-Eye"


def test_apply_edits_persists_character_texting_style_on_create_and_update(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    service = CharacterRegistryService(repositories)

    create_result = service.apply_edits(
        CharacterRegistryEdits(
            characters=(
                CharacterRegistryRow(
                    character_id="",
                    name="Captain Ilyra",
                    role="Watch captain",
                    texting_style=(
                        "Brief tactical check-ins, proper capitalization, no emoji."
                    ),
                ),
            ),
        ),
        active_save_id=save.id,
    )
    created_id = create_result.created_character_ids[0]
    created = repositories.get_character(created_id)
    assert created is not None
    assert created.texting_style == (
        "Brief tactical check-ins, proper capitalization, no emoji."
    )

    update_result = service.apply_edits(
        CharacterRegistryEdits(
            characters=(
                CharacterRegistryRow(
                    character_id=created_id,
                    name="Captain Ilyra",
                    role="Watch captain",
                    texting_style="  Two-word replies unless worried.  ",
                ),
            ),
        ),
        active_save_id=save.id,
    )
    assert update_result.updated_count == 1
    updated = repositories.get_character(created_id)
    assert updated is not None
    assert updated.texting_style == "Two-word replies unless worried."

    clear_result = service.apply_edits(
        CharacterRegistryEdits(
            characters=(
                CharacterRegistryRow(
                    character_id=created_id,
                    name="Captain Ilyra",
                    role="Watch captain",
                    texting_style="",
                ),
            ),
        ),
        active_save_id=save.id,
    )
    assert clear_result.updated_count == 1
    cleared = repositories.get_character(created_id)
    assert cleared is not None
    assert cleared.texting_style == ""


def test_build_model_exposes_character_texting_style(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    character = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        texting_style="Slow replies, polished punctuation, signs off with -I.",
        character_id="character-ilyra",
    )

    model = CharacterRegistryService(repositories).build_model(active_save_id=save.id)

    row = next(row for row in model.characters if row.character_id == character.id)
    assert row.texting_style == "Slow replies, polished punctuation, signs off with -I."


def test_apply_edits_merges_duplicate_character_into_active_save_target(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    gallery = repositories.add_location(save_id=save.id, name="Beacon Gallery")
    target = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        aliases=["Glass-Eye"],
        known_state="Commands the watch.",
        relationships={"Mara": "trusted ally"},
        character_id="character-ilyra",
    )
    duplicate = repositories.add_character(
        save_id=save.id,
        name="Ashknife",
        aliases=["The Red Captain"],
        role="Watch captain",
        known_state="Keeps the lens-key phrase.",
        relationships={
            "Mara": "rivals her for beacon command",
            "Ren": "owes a favor",
        },
        status="wounded",
        location_id=gallery.id,
        character_id="character-ashknife",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        current_location_id=gallery.id,
        present_character_ids=[duplicate.id],
    )
    memory = repositories.add_memory(
        save_id=save.id,
        body="Ashknife knows the lens-key phrase.",
        tags=[],
        memory_id="memory-ashknife",
    )
    state = repositories.upsert_world_state(
        save_id=save.id,
        key="beacon.lens",
        value={"failsafe": "copper notch"},
        state_id="world-state-lens",
    )
    first_message = repositories.append_message(
        save_id=save.id,
        role="player",
        body="I ask about Ilyra.",
    )
    summary = repositories.add_summary(
        save_id=save.id,
        covers_message_start_id=first_message.id,
        covers_message_end_id=first_message.id,
        body="Ilyra explained the beacon problem.",
        provider="fake",
        model="fake-summary",
        summary_id="summary-ilyra",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=duplicate.id,
        target_type="memory",
        target_id=memory.id,
        relation="knows",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=duplicate.id,
        target_type="state",
        target_id=state.id,
        relation="knows",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=target.id,
        target_type="summary",
        target_id=summary.id,
        relation="knows",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="location",
        entity_id=gallery.id,
        target_type="character",
        target_id=duplicate.id,
        relation="hosts",
    )

    result = CharacterRegistryService(repositories).apply_edits(
        CharacterRegistryEdits(
            characters=(
                CharacterRegistryRow(
                    character_id=duplicate.id,
                    name=duplicate.name,
                    merge_into_character_id=target.id,
                ),
            ),
        ),
        active_save_id=save.id,
    )

    assert result.archived_count == 1
    assert repositories.get_character(duplicate.id) is None
    merged = repositories.get_character(target.id)
    assert merged is not None
    assert "The Red Captain" in merged.aliases
    mara_relationship = merged.relationships["Mara"]
    assert isinstance(mara_relationship, str)
    assert "trusted ally" in mara_relationship
    assert "rivals her for beacon command" in mara_relationship
    assert merged.relationships["Ren"] == "owes a favor"
    assert merged.role == "Watch captain"
    assert "Keeps the lens-key phrase." in merged.known_state
    assert merged.status == "wounded"
    assert merged.location_id == gallery.id

    snapshot = repositories.get_scene_snapshot(save.id)
    assert snapshot is not None
    assert snapshot.present_character_ids == [target.id]

    links = repositories.list_entity_links(save.id)
    assert all(
        not (
            (link.entity_type == "character" and link.entity_id == duplicate.id)
            or (link.target_type == "character" and link.target_id == duplicate.id)
        )
        for link in links
    )
    assert {
        (
            link.entity_type,
            link.entity_id,
            link.target_type,
            link.target_id,
            link.relation,
        )
        for link in links
    } == {
        ("character", target.id, "memory", memory.id, "knows"),
        ("character", target.id, "world_state", state.id, "knows"),
        ("character", target.id, "summary", summary.id, "knows"),
        ("location", gallery.id, "character", target.id, "hosts"),
    }


def test_apply_edits_preserves_protected_source_on_manual_merge(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    target = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        protected_from_maintenance=False,
        character_id="character-ilyra",
    )
    protected_source = repositories.add_character(
        save_id=save.id,
        name="Ashknife",
        protected_from_maintenance=True,
        character_id="character-ashknife",
    )

    CharacterRegistryService(repositories).apply_edits(
        CharacterRegistryEdits(
            characters=(
                CharacterRegistryRow(
                    character_id=protected_source.id,
                    name=protected_source.name,
                    merge_into_character_id=target.id,
                ),
            ),
        ),
        active_save_id=save.id,
    )

    merged = repositories.get_character(target.id)
    assert merged is not None
    assert merged.protected_from_maintenance is True


def test_apply_edits_preserves_locked_target_fields_during_merge(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    target = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        appearance="Ash-stained officer coat.",
        voice="Clipped and dry.",
        goals="Protect the beacon.",
        current_intent="Hold the gallery.",
        status="present",
        locked_fields=["appearance", "voice", "current_intent"],
        character_id="character-ilyra",
    )
    duplicate = repositories.add_character(
        save_id=save.id,
        name="The Red Captain",
        appearance="A red cloak and molten-glass saber.",
        voice="Booming theatrical commands.",
        goals="Take command of the beacon.",
        current_intent="Demand surrender.",
        cooperation_conditions="Will help only if named captain.",
        status="wounded",
        character_id="character-red-captain",
    )

    CharacterRegistryService(repositories).apply_edits(
        CharacterRegistryEdits(
            characters=(
                CharacterRegistryRow(
                    character_id=duplicate.id,
                    name=duplicate.name,
                    merge_into_character_id=target.id,
                ),
            ),
        ),
        active_save_id=save.id,
    )

    merged = repositories.get_character(target.id)
    assert merged is not None
    assert merged.appearance == "Ash-stained officer coat."
    assert merged.voice == "Clipped and dry."
    assert merged.goals == (
        "Protect the beacon.\n\nMerged duplicate note: Take command of the beacon."
    )
    assert merged.current_intent == "Hold the gallery."
    assert merged.cooperation_conditions == "Will help only if named captain."
    assert merged.status == "present\n\nMerged duplicate note: wounded"
    assert repositories.get_character(duplicate.id) is None


def test_apply_edits_archiving_character_removes_thread_related_entity_references(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    kept = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        character_id="character-ilyra",
    )
    archived = repositories.add_character(
        save_id=save.id,
        name="Archivist Ren",
        character_id="character-ren",
    )
    thread = repositories.add_active_thread(
        save_id=save.id,
        title="Lens-key fallout",
        related_entities=[
            archived.id,
            f"character:{archived.id}",
            kept.id,
            "memory:memory-lens-key",
        ],
        thread_id="thread-lens-key",
    )

    CharacterRegistryService(repositories).apply_edits(
        CharacterRegistryEdits(
            characters=(
                CharacterRegistryRow(
                    character_id=archived.id,
                    name=archived.name,
                    archived=True,
                ),
            ),
        ),
        active_save_id=save.id,
    )

    updated_thread = repositories.get_active_thread(thread.id)
    assert updated_thread is not None
    assert updated_thread.related_entities == [kept.id, "memory:memory-lens-key"]


def test_apply_edits_merging_character_replaces_thread_related_entity_references(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    target = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        character_id="character-ilyra",
    )
    duplicate = repositories.add_character(
        save_id=save.id,
        name="Ashknife",
        character_id="character-ashknife",
    )
    thread = repositories.add_active_thread(
        save_id=save.id,
        title="Lens-key fallout",
        related_entities=[
            duplicate.id,
            f"character:{duplicate.id}",
            "location:beacon-gallery",
            target.id,
            f"character:{target.id}",
        ],
        thread_id="thread-lens-key",
    )

    CharacterRegistryService(repositories).apply_edits(
        CharacterRegistryEdits(
            characters=(
                CharacterRegistryRow(
                    character_id=duplicate.id,
                    name=duplicate.name,
                    merge_into_character_id=target.id,
                ),
            ),
        ),
        active_save_id=save.id,
    )

    updated_thread = repositories.get_active_thread(thread.id)
    assert updated_thread is not None
    assert updated_thread.related_entities == [
        target.id,
        f"character:{target.id}",
        "location:beacon-gallery",
    ]


def test_apply_edits_merging_character_skips_canonical_self_links(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    target = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        character_id="character-ilyra",
    )
    duplicate = repositories.add_character(
        save_id=save.id,
        name="Ashknife",
        character_id="character-ashknife",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=duplicate.id,
        target_type="character",
        target_id=target.id,
        relation="alias-of",
    )
    repositories.add_entity_link(
        save_id=save.id,
        entity_type="character",
        entity_id=target.id,
        target_type="character",
        target_id=duplicate.id,
        relation="alias-of",
    )

    CharacterRegistryService(repositories).apply_edits(
        CharacterRegistryEdits(
            characters=(
                CharacterRegistryRow(
                    character_id=duplicate.id,
                    name=duplicate.name,
                    merge_into_character_id=target.id,
                ),
            ),
        ),
        active_save_id=save.id,
    )

    assert {
        (link.entity_type, link.entity_id, link.target_type, link.target_id)
        for link in repositories.list_entity_links(save.id)
    } == set()


def test_apply_edits_rejects_chained_merge_batches_and_leaves_characters_active(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    first = repositories.add_character(
        save_id=save.id,
        name="Ashknife",
        character_id="character-ashknife",
    )
    second = repositories.add_character(
        save_id=save.id,
        name="Captain Ilyra",
        character_id="character-ilyra",
    )
    third = repositories.add_character(
        save_id=save.id,
        name="Archivist Ren",
        character_id="character-ren",
    )

    with pytest.raises(ValueError, match="Chained character merges"):
        CharacterRegistryService(repositories).apply_edits(
            CharacterRegistryEdits(
                characters=(
                    CharacterRegistryRow(
                        character_id=first.id,
                        name=first.name,
                        merge_into_character_id=second.id,
                    ),
                    CharacterRegistryRow(
                        character_id=second.id,
                        name=second.name,
                        merge_into_character_id=third.id,
                    ),
                ),
            ),
            active_save_id=save.id,
        )

    assert repositories.get_character(first.id) is not None
    assert repositories.get_character(second.id) is not None
    assert repositories.get_character(third.id) is not None


@pytest.mark.parametrize(
    ("target_id", "expected_message"),
    (
        ("self", "itself"),
        ("character-other-save", "active save"),
        ("character-missing", "active save"),
    ),
)
def test_apply_edits_rejects_invalid_merge_targets(
    repositories: PersistenceRepositories,
    target_id: str,
    expected_message: str,
) -> None:
    save = _create_save(repositories, title="Active Save")
    other_save = _create_save(repositories, title="Other Save")
    duplicate = repositories.add_character(
        save_id=save.id,
        name="Ashknife",
        character_id="character-ashknife",
    )
    repositories.add_character(
        save_id=other_save.id,
        name="Other Ilyra",
        character_id="character-other-save",
    )
    merge_target_id = duplicate.id if target_id == "self" else target_id

    with pytest.raises(ValueError, match=expected_message):
        CharacterRegistryService(repositories).apply_edits(
            CharacterRegistryEdits(
                characters=(
                    CharacterRegistryRow(
                        character_id=duplicate.id,
                        name=duplicate.name,
                        merge_into_character_id=merge_target_id,
                    ),
                ),
            ),
            active_save_id=save.id,
        )

    assert repositories.get_character(duplicate.id) is not None


def test_apply_edits_switches_single_player_character_and_forces_presence(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories)
    mara = repositories.add_character(
        save_id=save.id,
        name="Mara Voss",
        character_id="character-mara",
        is_player_character=True,
    )
    iris = repositories.add_character(
        save_id=save.id,
        name="Iris Vale",
        character_id="character-iris",
    )
    repositories.upsert_scene_snapshot(
        save_id=save.id,
        present_character_ids=[mara.id],
        snapshot_id="snapshot-main",
    )

    result = CharacterRegistryService(repositories).apply_edits(
        CharacterRegistryEdits(
            characters=(
                CharacterRegistryRow(
                    character_id=iris.id,
                    name=iris.name,
                    relationships_json="{}",
                    present=False,
                    is_player_character=True,
                ),
            ),
        ),
        active_save_id=save.id,
    )

    updated_mara = repositories.get_character(mara.id)
    updated_iris = repositories.get_character(iris.id)
    snapshot = repositories.get_scene_snapshot(save.id)
    assert updated_mara is not None
    assert updated_iris is not None
    assert snapshot is not None
    assert updated_mara.is_player_character is False
    assert updated_iris.is_player_character is True
    assert updated_iris.protected_from_maintenance is True
    assert set(snapshot.present_character_ids) == {mara.id, iris.id}
    rows = {row.character_id: row for row in result.model.characters}
    assert rows[iris.id].present is True
    assert rows[iris.id].is_player_character is True


def test_apply_edits_rejects_link_targets_outside_active_save(
    repositories: PersistenceRepositories,
) -> None:
    save = _create_save(repositories, title="Active Save")
    other_save = _create_save(repositories, title="Other Save")
    character = repositories.add_character(save_id=save.id, name="Captain Ilyra")
    other_memory = repositories.add_memory(
        save_id=other_save.id,
        body="This belongs to another save.",
        tags=[],
    )

    with pytest.raises(ValueError, match="linked target"):
        CharacterRegistryService(repositories).apply_edits(
            CharacterRegistryEdits(
                characters=(
                    CharacterRegistryRow(
                        character_id=character.id,
                        name="Captain Ilyra",
                        linked_memory_ids=(other_memory.id,),
                    ),
                ),
            ),
            active_save_id=save.id,
        )


def _create_save(
    repositories: PersistenceRepositories,
    *,
    title: str = "Night Watch",
) -> Any:
    scenario = repositories.create_scenario(
        type="full_roleplay",
        title="Ashfall Keep",
        premise="A border keep is cut off by ash storms.",
        player_role="Signal warden",
        content={"starting_scene": "The beacon gutters in the tower."},
    )
    return repositories.create_save(scenario_id=scenario.id, title=title)
