"""Deterministic dating-sim route state maintenance."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from bragi.interaction_mode import InteractionMode
from bragi.persistence.models import (
    CharacterRecord,
    DatingRouteStateRecord,
    MessageRecord,
    ScenarioRecord,
)
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.dating_route_policy import (
    ROUTE_STAGE_RANK,
    next_reasonable_step,
)
from bragi.services.mention_matching import character_name_is_mentioned
from bragi.services.phone_number_exchange import (
    PHONE_EXCHANGE_BOTH,
    PHONE_EXCHANGE_CHARACTER_HAS_PLAYER_NUMBER,
    PHONE_EXCHANGE_PLAYER_HAS_CHARACTER_NUMBER,
    infer_phone_number_exchanges,
)


@dataclass(frozen=True)
class DatingRouteUpdateResult:
    seeded_count: int = 0
    updated_count: int = 0


class DatingRouteService:
    def __init__(self, repositories: PersistenceRepositories) -> None:
        self.repositories = repositories

    def seed_routes_for_save(
        self,
        save_id: str,
        *,
        source_message_id: str | None = None,
    ) -> int:
        details = self.repositories.load_save_details(save_id)
        if (
            details is None
            or details.save.interaction_mode is InteractionMode.STORYTELLER
            or not _scenario_supports_dating_routes(details.scenario)
        ):
            return 0
        characters = self.repositories.list_characters(save_id)
        player = _player_character(characters)
        if player is None:
            return 0
        snapshot = self.repositories.get_scene_snapshot(save_id)
        seeded = 0
        for character in characters:
            if character.is_player_character:
                continue
            if not _is_romance_option(
                character=character,
                player_keys=_romance_player_keys(player),
            ):
                continue
            if (
                self.repositories.get_dating_route_state_for_pair(
                    save_id,
                    player.id,
                    character.id,
                )
                is not None
            ):
                continue
            stage = "introduced" if character.met else "unmet"
            self.repositories.upsert_dating_route_state(
                save_id=save_id,
                player_character_id=player.id,
                npc_character_id=character.id,
                stage=stage,
                first_met_message_id=source_message_id if character.met else None,
                first_met_world_day_index=(
                    snapshot.world_day_index
                    if character.met and snapshot is not None
                    else None
                ),
                completed_interactions=0,
                dates_completed=0,
                known_boundaries=_known_boundaries(character),
                interest_level="available" if stage == "introduced" else "",
                pacing_preference="",
                next_reasonable_step=next_reasonable_step(stage),
                source_message_id=source_message_id,
            )
            seeded += 1
        return seeded

    def update_after_turn(
        self,
        *,
        save_id: str,
        player_message_id: str,
        narrator_message_id: str,
    ) -> DatingRouteUpdateResult:
        details = self.repositories.load_save_details(save_id)
        if (
            details is None
            or details.save.interaction_mode is InteractionMode.STORYTELLER
            or not _scenario_supports_dating_routes(details.scenario)
        ):
            return DatingRouteUpdateResult()
        messages = {message.id: message for message in details.messages}
        player_message = messages.get(player_message_id)
        narrator_message = messages.get(narrator_message_id)
        if player_message is None or narrator_message is None:
            return DatingRouteUpdateResult()
        seeded = self.seed_routes_for_save(save_id)
        characters = self.repositories.list_characters(save_id)
        characters_by_id = {character.id: character for character in characters}
        snapshot = self.repositories.get_scene_snapshot(save_id)
        present_ids = set(snapshot.present_character_ids if snapshot else [])
        text = f"{player_message.body}\n{narrator_message.body}"
        normalized_text = _normalized_text(text)
        updated = 0
        for route in self.repositories.list_dating_route_states(save_id):
            character = characters_by_id.get(route.npc_character_id)
            if character is None:
                continue
            if not _route_is_active_this_turn(
                route=route,
                character=character,
                present_ids=present_ids,
                text=normalized_text,
            ):
                continue
            repeated_turn = route.last_interaction_message_id == narrator_message_id
            completed_interactions = route.completed_interactions + (
                0 if repeated_turn else 1
            )
            date_completed = (
                _date_completed(normalized_text) and not repeated_turn
            )
            dates_completed = route.dates_completed + (1 if date_completed else 0)
            next_stage = _stage_after_turn(
                route=route,
                text=normalized_text,
                date_completed=date_completed,
                dates_completed=dates_completed,
            )
            self.repositories.upsert_dating_route_state(
                save_id=save_id,
                player_character_id=route.player_character_id,
                npc_character_id=route.npc_character_id,
                stage=next_stage,
                first_met_message_id=route.first_met_message_id
                or narrator_message_id,
                first_met_world_day_index=(
                    route.first_met_world_day_index
                    if route.first_met_world_day_index is not None
                    else snapshot.world_day_index if snapshot is not None else None
                ),
                last_interaction_message_id=narrator_message_id,
                last_interaction_world_day_index=(
                    snapshot.world_day_index if snapshot is not None else None
                ),
                completed_interactions=completed_interactions,
                dates_completed=dates_completed,
                next_reasonable_step=next_reasonable_step(next_stage),
                source_message_id=narrator_message_id,
            )
            player = characters_by_id.get(route.player_character_id)
            if (
                player is not None
                and ROUTE_STAGE_RANK.get(next_stage, 0)
                >= ROUTE_STAGE_RANK["contact_exchanged"]
            ):
                _apply_contact_state_for_route_exchange(
                    self.repositories,
                    save_id=save_id,
                    player=player,
                    npc=character,
                    completed_messages=(player_message, narrator_message),
                )
            updated += 0 if repeated_turn and next_stage == route.stage else 1
        return DatingRouteUpdateResult(seeded_count=seeded, updated_count=updated)


def _player_character(characters: list[CharacterRecord]) -> CharacterRecord | None:
    for character in characters:
        if character.is_player_character:
            return character
    return None


def _apply_contact_state_for_route_exchange(
    repositories: PersistenceRepositories,
    *,
    save_id: str,
    player: CharacterRecord,
    npc: CharacterRecord,
    completed_messages: tuple[MessageRecord, ...],
) -> None:
    for exchange in infer_phone_number_exchanges(
        completed_messages=completed_messages,
        player=player,
        npcs=(npc,),
    ):
        if exchange.character_id != npc.id:
            continue
        direction = exchange.direction.strip()
        repositories.upsert_character_contact_state(
            save_id=save_id,
            player_character_id=player.id,
            character_id=npc.id,
            player_has_character_number=direction
            in {PHONE_EXCHANGE_PLAYER_HAS_CHARACTER_NUMBER, PHONE_EXCHANGE_BOTH},
            character_has_player_number=direction
            in {PHONE_EXCHANGE_CHARACTER_HAS_PLAYER_NUMBER, PHONE_EXCHANGE_BOTH},
            source_message_id=exchange.source_message_id,
        )


def _scenario_supports_dating_routes(scenario: ScenarioRecord) -> bool:
    if scenario.type == "dating_sim":
        return True
    try:
        content = json.loads(scenario.content_json)
    except json.JSONDecodeError:
        return False
    if not isinstance(content, dict):
        return False
    genres = content.get("_scenario_genres")
    return isinstance(genres, list) and "dating_sim" in genres


def _romance_player_keys(player: CharacterRecord) -> set[str]:
    return {
        player.name.casefold(),
        "player",
        *{alias.casefold() for alias in player.aliases},
    }


def _is_romance_option(character: CharacterRecord, player_keys: set[str]) -> bool:
    relationship_text = " ".join(
        str(value) for value in character.relationships.values()
    ).casefold()
    status = character.status.casefold()
    relationship_keys = {key.casefold() for key in character.relationships}
    return (
        "romance option" in relationship_text
        or "romance option" in status
    ) and bool(player_keys & relationship_keys)


def _route_is_active_this_turn(
    *,
    route: DatingRouteStateRecord,
    character: CharacterRecord,
    present_ids: set[str],
    text: str,
) -> bool:
    if route.npc_character_id in present_ids:
        return True
    return character_name_is_mentioned(
        name=character.name,
        aliases=character.aliases,
        text=text,
    )


def _stage_after_turn(
    *,
    route: DatingRouteStateRecord,
    text: str,
    date_completed: bool,
    dates_completed: int,
) -> str:
    next_stage = route.stage
    if next_stage == "unmet":
        next_stage = "introduced"
    if (
        _contact_exchanged(text)
        and ROUTE_STAGE_RANK["contact_exchanged"] > ROUTE_STAGE_RANK[next_stage]
    ):
        next_stage = "contact_exchanged"
    if (
        _date_planned(text)
        and ROUTE_STAGE_RANK[next_stage] >= ROUTE_STAGE_RANK["contact_exchanged"]
    ):
        next_stage = _max_stage(next_stage, "first_date_planned")
    if (
        _date_started(text)
        and ROUTE_STAGE_RANK[next_stage] >= ROUTE_STAGE_RANK["first_date_planned"]
    ):
        next_stage = _max_stage(next_stage, "first_date_in_progress")
    if (
        date_completed
        and ROUTE_STAGE_RANK[next_stage] >= ROUTE_STAGE_RANK["first_date_in_progress"]
    ):
        next_stage = _max_stage(next_stage, "early_dating")
    if (
        _exclusive(text)
        and route.dates_completed >= 1
        and ROUTE_STAGE_RANK[next_stage] >= ROUTE_STAGE_RANK["early_dating"]
    ):
        next_stage = _max_stage(next_stage, "exclusive")
    if (
        _committed(text)
        and ROUTE_STAGE_RANK[route.stage] >= ROUTE_STAGE_RANK["exclusive"]
    ):
        next_stage = _max_stage(next_stage, "committed")
    if date_completed and dates_completed > route.dates_completed:
        next_stage = _max_stage(next_stage, "early_dating")
    return next_stage


def _max_stage(left: str, right: str) -> str:
    if ROUTE_STAGE_RANK.get(right, 0) > ROUTE_STAGE_RANK.get(left, 0):
        return right
    return left


def _contact_exchanged(text: str) -> bool:
    return bool(
        re.search(
            r"\b(exchange|gives?|gave|shares?|shared|swap|swapped)\b"
            r".{0,80}\b(numbers?|phone|contact|text|handle|dm)\b",
            text,
        )
        or re.search(
            r"\b(numbers?|phone|contact|text|handle|dm)\b"
            r".{0,80}\b(exchange|gives?|gave|shares?|shared|swap|swapped)\b",
            text,
        )
    )


def _date_planned(text: str) -> bool:
    return bool(
        re.search(
            r"\b(plan|plans|planned|schedule|scheduled|ask|asks|asked|invite|invites|invited)\b"
            r".{0,80}\b(date|meet up|go out)\b",
            text,
        )
        or re.search(
            r"\b(date|meet up|go out)\b.{0,80}\b(tomorrow|tonight|after school|"
            r"this weekend|saturday|sunday|planned|scheduled)\b",
            text,
        )
    )


def _date_started(text: str) -> bool:
    return bool(
        re.search(r"\b(first )?date (begins|starts|is underway)\b", text)
        or re.search(r"\bon (their|a|the) (first )?date\b", text)
    )


def _date_completed(text: str) -> bool:
    return bool(
        re.search(
            r"\b(date|first date).{0,40}\b(ends|ended|complete|completed)\b",
            text,
        )
        or re.search(r"\bafter (their|the|a) (first )?date\b", text)
    )


def _exclusive(text: str) -> bool:
    return bool(
        re.search(r"\b(exclusive|exclusivity|officially together)\b", text)
        or re.search(
            r"\b(?:i|you|we|she|he|they)\s+(?:are|was|'re)?\s+(?:my|our|your)\s+"
            r"(?:girlfriend|boyfriend|partner)\b",
            text,
        )
    )


def _committed(text: str) -> bool:
    return bool(
        re.search(r"\b(committed|commitment|long-term relationship)\b", text)
        and not re.search(
            r"\b(?:not|never|not yet|don't|do not)\b.{0,40}"
            r"\b(?:commit|committed|commitment)\b",
            text,
        )
    )


def _known_boundaries(character: CharacterRecord) -> list[str]:
    raw_entries = character.boundaries.replace(";", "\n").splitlines()
    values = []
    for value in raw_entries:
        stripped = value.strip()
        if stripped and stripped.lower() not in {"", "none", "n/a", "na"}:
            values.append(stripped)
    return values


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())
