"""Deterministic backfill for character knowledge graph rows."""

from __future__ import annotations

from dataclasses import dataclass

from bragi.persistence.models import CharacterRecord
from bragi.persistence.repositories import PersistenceRepositories


@dataclass(frozen=True)
class KnowledgeBackfillResult:
    save_id: str
    character_knowledge_edges_applied: int = 0
    message_visibility_applied: int = 0

    def to_json(self) -> dict[str, object]:
        return {
            "save_id": self.save_id,
            "character_knowledge_edges_applied": (
                self.character_knowledge_edges_applied
            ),
            "message_visibility_applied": self.message_visibility_applied,
        }


class KnowledgeBackfillService:
    def __init__(self, repositories: PersistenceRepositories) -> None:
        self.repositories = repositories

    def backfill_save(self, save_id: str) -> KnowledgeBackfillResult:
        if self.repositories.get_save(save_id) is None:
            raise ValueError(f"Unknown save id: {save_id}")
        characters = self.repositories.list_characters(save_id)
        character_ids = {character.id for character in characters}
        name_index = _character_name_index(characters)
        edge_count = self._backfill_legacy_knows_links(
            save_id=save_id,
            character_ids=character_ids,
        )
        visibility_count = self._backfill_message_visibility(
            save_id=save_id,
            name_index=name_index,
            character_ids=character_ids,
        )
        return KnowledgeBackfillResult(
            save_id=save_id,
            character_knowledge_edges_applied=edge_count,
            message_visibility_applied=visibility_count,
        )

    def _backfill_legacy_knows_links(
        self,
        *,
        save_id: str,
        character_ids: set[str],
    ) -> int:
        count = 0
        for link in self.repositories.list_entity_links(save_id):
            if (
                link.entity_type != "character"
                or link.relation != "knows"
                or link.entity_id not in character_ids
            ):
                continue
            source_ids = [link.source_message_id] if link.source_message_id else []
            self.repositories.add_character_knowledge_edge(
                save_id=save_id,
                character_id=link.entity_id,
                target_type=link.target_type,
                target_id=link.target_id,
                knowledge_state="knows",
                acquisition_method="unknown",
                confidence=1.0,
                source_message_id=link.source_message_id,
                source_message_ids=source_ids,
                evidence_quote="Backfilled from legacy character knows link.",
            )
            count += 1
        return count

    def _backfill_message_visibility(
        self,
        *,
        save_id: str,
        name_index: dict[str, str],
        character_ids: set[str],
    ) -> int:
        count = 0
        for message in self.repositories.list_messages(save_id):
            if not message.speaker_name:
                continue
            character_id = name_index.get(_name_key(message.speaker_name))
            if character_id is None:
                continue
            self.repositories.add_message_visibility(
                save_id=save_id,
                message_id=message.id,
                character_id=character_id,
                visibility="visible",
                confidence=1.0,
                source="speaker_name",
                evidence=f"{message.speaker_name} is the message speaker.",
            )
            count += 1
        snapshot = self.repositories.get_scene_snapshot(save_id)
        if snapshot is None or snapshot.source_message_id is None:
            return count
        for character_id in snapshot.present_character_ids:
            if character_id not in character_ids:
                continue
            self.repositories.add_message_visibility(
                save_id=save_id,
                message_id=snapshot.source_message_id,
                character_id=character_id,
                visibility="visible",
                confidence=0.95,
                source="scene_snapshot",
                evidence="Character was present in the scene snapshot.",
            )
            count += 1
        return count


def _character_name_index(characters: list[CharacterRecord]) -> dict[str, str]:
    index: dict[str, str] = {}
    for character in characters:
        for value in (character.name, *character.aliases):
            key = _name_key(value)
            if key:
                index.setdefault(key, character.id)
    return index


def _name_key(value: str) -> str:
    return " ".join(value.strip().casefold().split())
