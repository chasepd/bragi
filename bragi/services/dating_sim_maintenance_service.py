"""Reviewable maintenance for existing dating-sim pacing state."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, cast

from bragi.persistence.models import (
    CharacterRecord,
    DatingRouteStateRecord,
    MessageRecord,
)
from bragi.persistence.repositories import PersistenceRepositories
from bragi.redaction import redact_text
from bragi.services.dating_route_policy import ROUTE_STAGE_RANK, next_reasonable_step
from bragi.services.dating_route_profile_service import (
    enqueue_dating_route_profile_enrichment,
)
from bragi.services.dating_route_service import (
    _date_completed,
    _date_planned,
    _date_started,
    _is_romance_option,
    _known_boundaries,
    _player_character,
    _romance_player_keys,
)
from bragi.services.mention_matching import character_name_is_mentioned
from bragi.services.world_time_service import world_time_display
from bragi.world_time_model import (
    canonical_world_time_from_legacy,
    canonical_world_time_from_values,
    legacy_world_time_fields,
)

_CLOCK_READOUT_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
_TIMER_CONTEXT_RE = re.compile(
    r"\b(countdown|elapsed|left|remaining|round|timer|minutes?|seconds?)\b",
    re.IGNORECASE,
)
_CONTACT_EXCHANGE_RE = re.compile(
    r"\b(exchange|exchanged|gives?|gave|shares?|shared|swap|swapped)\b"
    r".{0,80}\b(numbers?|phone|contact|text|handle|dm)\b"
    r"|"
    r"\b(numbers?|phone|contact|text|handle|dm)\b"
    r".{0,80}\b(exchange|exchanged|gives?|gave|shares?|shared|swap|swapped)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DatingSimMaintenanceRepair:
    repair_id: str
    category: str
    target_type: str
    target_id: str | None
    field_path: str
    proposed_value: object
    reason: str
    confidence: float
    source_message_ids: tuple[str, ...] = ()
    evidence_text: str = ""
    npc_character_id: str | None = None
    stage: str = ""
    first_met_message_id: str | None = None
    last_interaction_message_id: str | None = None
    completed_interactions: int = 0
    dates_completed: int = 0

    def to_result(self) -> dict[str, object]:
        result: dict[str, object] = {
            "repair_id": self.repair_id,
            "category": self.category,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "field_path": self.field_path,
            "proposed_value": self.proposed_value,
            "reason": self.reason,
            "confidence": self.confidence,
            "source_message_ids": list(self.source_message_ids),
        }
        if self.npc_character_id is not None:
            result["npc_character_id"] = self.npc_character_id
        if self.stage:
            result["stage"] = self.stage
        if self.first_met_message_id is not None:
            result["first_met_message_id"] = self.first_met_message_id
        if self.last_interaction_message_id is not None:
            result["last_interaction_message_id"] = self.last_interaction_message_id
        if self.completed_interactions:
            result["completed_interactions"] = self.completed_interactions
        if self.dates_completed:
            result["dates_completed"] = self.dates_completed
        if self.evidence_text:
            result["evidence_text"] = self.evidence_text
        return result


@dataclass(frozen=True)
class DatingSimMaintenanceReport:
    save_id: str
    status: str
    skipped_reason: str = ""
    deterministic_repairs: tuple[DatingSimMaintenanceRepair, ...] = ()
    reviewable_repairs: tuple[DatingSimMaintenanceRepair, ...] = ()
    ambiguous_suggestions: tuple[DatingSimMaintenanceRepair, ...] = ()
    applied_repair_ids: tuple[str, ...] = ()
    applied_count: int = 0
    scanned_messages: int = 0
    romance_option_count: int = 0

    def to_result(self) -> dict[str, object]:
        return {
            "save_id": self.save_id,
            "status": self.status,
            "skipped_reason": self.skipped_reason,
            "deterministic_repairs": [
                repair.to_result() for repair in self.deterministic_repairs
            ],
            "reviewable_repairs": [
                repair.to_result() for repair in self.reviewable_repairs
            ],
            "ambiguous_suggestions": [
                repair.to_result() for repair in self.ambiguous_suggestions
            ],
            "applied_repair_ids": list(self.applied_repair_ids),
            "applied_count": self.applied_count,
            "scanned_messages": self.scanned_messages,
            "romance_option_count": self.romance_option_count,
            "deterministic_repair_count": len(self.deterministic_repairs),
            "reviewable_repair_count": len(self.reviewable_repairs),
            "ambiguous_suggestion_count": len(self.ambiguous_suggestions),
        }


@dataclass(frozen=True)
class _RouteEvidence:
    stage: str
    first_met_message_id: str | None
    last_interaction_message_id: str | None
    completed_interactions: int
    dates_completed: int
    source_message_ids: tuple[str, ...]
    evidence_text: str = ""


class DatingSimMaintenanceService:
    def __init__(self, repositories: PersistenceRepositories) -> None:
        self.repositories = repositories

    def inspect_save(
        self,
        save_id: str,
        *,
        include_evidence_text: bool = False,
    ) -> DatingSimMaintenanceReport:
        details = self.repositories.load_save_details(save_id)
        if details is None:
            return DatingSimMaintenanceReport(
                save_id=save_id,
                status="skipped",
                skipped_reason="unknown_save",
            )
        if details.scenario.type != "dating_sim":
            return DatingSimMaintenanceReport(
                save_id=save_id,
                status="skipped",
                skipped_reason="not_dating_sim",
            )
        messages = list(details.messages)
        characters = self.repositories.list_characters(save_id)
        player = _player_character(characters)
        if player is None:
            return DatingSimMaintenanceReport(
                save_id=save_id,
                status="skipped",
                skipped_reason="missing_player_character",
                scanned_messages=len(messages),
            )
        romance_options = _romance_options(characters, player=player)
        deterministic = list(
            self._scene_time_repairs(
                save_id,
                messages=messages,
                include_evidence_text=include_evidence_text,
            )
        )
        reviewable: list[DatingSimMaintenanceRepair] = []
        for npc in romance_options:
            evidence = _route_evidence(
                npc,
                messages=messages,
                include_evidence_text=include_evidence_text,
            )
            existing = self.repositories.get_dating_route_state_for_pair(
                save_id,
                player.id,
                npc.id,
            )
            repair = _route_repair(
                save_id=save_id,
                player=player,
                npc=npc,
                existing=existing,
                evidence=evidence,
            )
            if repair is not None:
                reviewable.append(repair)
        status = "ready" if deterministic or reviewable else "no_repairs"
        return DatingSimMaintenanceReport(
            save_id=save_id,
            status=status,
            deterministic_repairs=tuple(deterministic),
            reviewable_repairs=tuple(reviewable),
            scanned_messages=len(messages),
            romance_option_count=len(romance_options),
        )

    def apply_repairs(
        self,
        save_id: str,
        *,
        repair_ids: list[str] | tuple[str, ...],
        confirm_save_id: str,
        include_evidence_text: bool = False,
    ) -> DatingSimMaintenanceReport:
        if confirm_save_id != save_id:
            raise ValueError("confirm_save_id must match save_id")
        selected = tuple(dict.fromkeys(repair_ids))
        if not selected:
            raise ValueError("repair_ids are required")
        report = self.inspect_save(
            save_id,
            include_evidence_text=include_evidence_text,
        )
        repairs = {
            repair.repair_id: repair
            for repair in (*report.deterministic_repairs, *report.reviewable_repairs)
        }
        missing = [repair_id for repair_id in selected if repair_id not in repairs]
        if missing:
            raise ValueError(f"Unknown repair id: {missing[0]}")
        applied: list[str] = []
        for repair_id in selected:
            repair = repairs[repair_id]
            if repair.category == "deterministic":
                self._apply_scene_time_repair(save_id, repair)
            elif repair.category == "reviewable_route_backfill":
                self._apply_route_repair(save_id, repair)
            else:
                continue
            applied.append(repair_id)
        return DatingSimMaintenanceReport(
            save_id=save_id,
            status="applied",
            deterministic_repairs=report.deterministic_repairs,
            reviewable_repairs=report.reviewable_repairs,
            ambiguous_suggestions=report.ambiguous_suggestions,
            applied_repair_ids=tuple(applied),
            applied_count=len(applied),
            scanned_messages=report.scanned_messages,
            romance_option_count=report.romance_option_count,
        )

    def _scene_time_repairs(
        self,
        save_id: str,
        *,
        messages: list[MessageRecord],
        include_evidence_text: bool,
    ) -> tuple[DatingSimMaintenanceRepair, ...]:
        snapshot = self.repositories.get_scene_snapshot(save_id)
        if snapshot is None or not snapshot.in_world_time:
            return ()
        if _CLOCK_READOUT_RE.search(snapshot.in_world_time) is None:
            return ()
        source = _message_by_id(messages, snapshot.source_message_id)
        if source is None or not _timer_readout_near_context(source.body):
            return ()
        proposed = world_time_display(
            time_of_day=snapshot.time_of_day,
            day_of_week=snapshot.day_of_week,
        )
        repair = DatingSimMaintenanceRepair(
            repair_id=f"scene-time-cleanup:{snapshot.id}",
            category="deterministic",
            target_type="scene_snapshot",
            target_id=snapshot.id,
            field_path="in_world_time",
            proposed_value=proposed,
            reason=(
                "Scene display time appears to use a countdown timer readout rather "
                "than an in-world clock time."
            ),
            confidence=0.98,
            source_message_ids=(source.id,),
            evidence_text=_evidence_text(source, include_evidence_text),
        )
        return (repair,)

    def _apply_scene_time_repair(
        self,
        save_id: str,
        repair: DatingSimMaintenanceRepair,
    ) -> None:
        snapshot = self.repositories.get_scene_snapshot(save_id)
        if snapshot is None:
            raise ValueError("Scene repair cannot apply without a scene snapshot")
        after = str(repair.proposed_value)
        source_message_id = (
            repair.source_message_ids[-1]
            if repair.source_message_ids
            else snapshot.source_message_id
        )
        canonical_world_time = canonical_world_time_from_legacy(
            in_world_time=after,
            time_of_day="",
            day_of_week=snapshot.day_of_week,
            world_day_index=snapshot.world_day_index,
            source_message_id=source_message_id,
            confidence=repair.confidence,
        )
        display_world_time = canonical_world_time_from_values(
            day_index=canonical_world_time.day_index,
            day_label=canonical_world_time.day_label,
            phase=canonical_world_time.phase,
            clock_minutes=canonical_world_time.clock_minutes,
            period_label=(
                canonical_world_time.period_label or snapshot.world_time_period_label
            ),
            source_message_id=canonical_world_time.source_message_id,
            confidence=canonical_world_time.confidence,
            legacy_in_world_time=after,
            legacy_time_of_day=snapshot.time_of_day,
            legacy_day_of_week=snapshot.day_of_week,
            legacy_world_day_index=snapshot.world_day_index,
        )
        legacy_fields = legacy_world_time_fields(display_world_time)
        self.repositories.upsert_scene_snapshot(
            save_id=save_id,
            current_location_id=snapshot.current_location_id,
            situation=snapshot.situation,
            objective=snapshot.objective,
            in_world_time=cast(str, legacy_fields["in_world_time"]),
            time_of_day=cast(str, legacy_fields["time_of_day"]),
            day_of_week=cast(str, legacy_fields["day_of_week"]),
            world_day_index=cast(int | None, legacy_fields["world_day_index"]),
            world_time_day_index=canonical_world_time.day_index,
            world_time_day_label=canonical_world_time.day_label,
            world_time_phase=canonical_world_time.phase,
            world_time_clock_minutes=canonical_world_time.clock_minutes,
            world_time_period_label=(
                canonical_world_time.period_label or snapshot.world_time_period_label
            ),
            world_time_source_message_id=canonical_world_time.source_message_id,
            world_time_confidence=canonical_world_time.confidence,
            weather=snapshot.weather,
            mood=snapshot.mood,
            nearby_objects=snapshot.nearby_objects,
            hazards=snapshot.hazards,
            present_character_ids=snapshot.present_character_ids,
            source_message_id=snapshot.source_message_id,
            locked_fields=snapshot.locked_fields,
            snapshot_id=snapshot.id,
            first_seen_message_id=snapshot.first_seen_message_id,
            last_updated_message_id=snapshot.last_updated_message_id,
        )
        self.repositories.add_context_update_audit(
            save_id=save_id,
            operation="dating_sim_maintenance_apply",
            entity_type="scene_snapshot",
            entity_id=snapshot.id,
            field_path=repair.field_path,
            before=snapshot.in_world_time,
            after=after,
            reason=repair.reason,
            confidence=repair.confidence,
            source_message_ids=list(repair.source_message_ids),
        )

    def _apply_route_repair(
        self,
        save_id: str,
        repair: DatingSimMaintenanceRepair,
    ) -> None:
        proposed = cast(dict[str, object], repair.proposed_value)
        player_id = str(proposed["player_character_id"])
        npc_id = str(proposed["npc_character_id"])
        existing = self.repositories.get_dating_route_state_for_pair(
            save_id,
            player_id,
            npc_id,
        )
        route = self.repositories.upsert_dating_route_state(
            save_id=save_id,
            player_character_id=player_id,
            npc_character_id=npc_id,
            stage=repair.stage,
            first_met_message_id=repair.first_met_message_id,
            last_interaction_message_id=repair.last_interaction_message_id,
            completed_interactions=repair.completed_interactions,
            dates_completed=repair.dates_completed,
            known_boundaries=_string_list(proposed.get("known_boundaries")),
            next_reasonable_step=next_reasonable_step(repair.stage),
            source_message_id=repair.last_interaction_message_id
            or repair.first_met_message_id,
        )
        if existing is None:
            enqueue_dating_route_profile_enrichment(
                self.repositories,
                save_id=save_id,
                force_due=True,
            )
        self.repositories.add_context_update_audit(
            save_id=save_id,
            operation="dating_sim_maintenance_apply",
            entity_type="dating_route_state",
            entity_id=route.id,
            field_path="route_state",
            before=existing.stage if existing is not None else None,
            after=repair.stage,
            reason=repair.reason,
            confidence=repair.confidence,
            source_message_ids=list(repair.source_message_ids),
        )


def _romance_options(
    characters: list[CharacterRecord],
    *,
    player: CharacterRecord,
) -> list[CharacterRecord]:
    player_keys = _romance_player_keys(player)
    return [
        character
        for character in characters
        if not character.is_player_character
        and _is_romance_option(character=character, player_keys=player_keys)
    ]


def _route_repair(
    *,
    save_id: str,
    player: CharacterRecord,
    npc: CharacterRecord,
    existing: DatingRouteStateRecord | None,
    evidence: _RouteEvidence,
) -> DatingSimMaintenanceRepair | None:
    if evidence.first_met_message_id is None and existing is not None:
        return None
    stage = evidence.stage
    if existing is not None:
        existing_rank = ROUTE_STAGE_RANK.get(existing.stage, 0)
        evidence_rank = ROUTE_STAGE_RANK.get(stage, 0)
        if (
            existing.first_met_message_id
            and existing.completed_interactions >= evidence.completed_interactions
            and existing.dates_completed >= evidence.dates_completed
            and existing_rank >= evidence_rank
        ):
            return None
    proposed_value: dict[str, Any] = {
        "save_id": save_id,
        "player_character_id": player.id,
        "npc_character_id": npc.id,
        "stage": stage,
        "completed_interactions": evidence.completed_interactions,
        "dates_completed": evidence.dates_completed,
        "known_boundaries": _known_boundaries(npc),
    }
    return DatingSimMaintenanceRepair(
        repair_id=f"route-backfill:{npc.id}",
        category="reviewable_route_backfill",
        target_type="dating_route_state",
        target_id=existing.id if existing is not None else None,
        field_path="route_state",
        proposed_value=proposed_value,
        reason="Clear transcript evidence can backfill dating route pacing state.",
        confidence=0.84 if existing is None else 0.78,
        source_message_ids=evidence.source_message_ids,
        evidence_text=evidence.evidence_text,
        npc_character_id=npc.id,
        stage=stage,
        first_met_message_id=evidence.first_met_message_id
        or (existing.first_met_message_id if existing else None),
        last_interaction_message_id=evidence.last_interaction_message_id,
        completed_interactions=evidence.completed_interactions,
        dates_completed=evidence.dates_completed,
    )


def _route_evidence(
    npc: CharacterRecord,
    *,
    messages: list[MessageRecord],
    include_evidence_text: bool,
) -> _RouteEvidence:
    stage = "unmet"
    first_met_message_id: str | None = None
    last_interaction_message_id: str | None = None
    completed_interactions = 0
    dates_completed = 0
    source_message_ids: list[str] = []
    evidence_lines: list[str] = []
    for message in messages:
        if not _message_mentions_character(message, npc):
            continue
        if first_met_message_id is None:
            first_met_message_id = message.id
            stage = _max_stage(stage, "introduced")
        last_interaction_message_id = message.id
        completed_interactions += 1
        text = _normalized_text(message.body)
        if _contact_exchanged(text):
            stage = _max_stage(stage, "contact_exchanged")
        if _date_planned(text):
            stage = _max_stage(stage, "first_date_planned")
        if _date_started(text):
            stage = _max_stage(stage, "first_date_in_progress")
        if _date_completed(text):
            dates_completed += 1
            stage = _max_stage(stage, "early_dating")
        source_message_ids.append(message.id)
        if include_evidence_text:
            evidence_lines.append(f"[{message.id}] {redact_text(message.body) or ''}")
    return _RouteEvidence(
        stage=stage,
        first_met_message_id=first_met_message_id,
        last_interaction_message_id=last_interaction_message_id,
        completed_interactions=completed_interactions,
        dates_completed=dates_completed,
        source_message_ids=tuple(source_message_ids),
        evidence_text="\n".join(evidence_lines),
    )


def _message_mentions_character(
    message: MessageRecord,
    character: CharacterRecord,
) -> bool:
    return character_name_is_mentioned(
        name=character.name,
        aliases=character.aliases,
        text=message.body,
    )


def _message_by_id(
    messages: list[MessageRecord],
    message_id: str | None,
) -> MessageRecord | None:
    if message_id is None:
        return None
    for message in messages:
        if message.id == message_id:
            return message
    return None


def _timer_readout_near_context(text: str) -> bool:
    for match in _CLOCK_READOUT_RE.finditer(text):
        window = text[max(0, match.start() - 48) : match.end() + 48]
        if _TIMER_CONTEXT_RE.search(window):
            return True
    return False


def _contact_exchanged(text: str) -> bool:
    return _CONTACT_EXCHANGE_RE.search(text) is not None


def _evidence_text(message: MessageRecord, include: bool) -> str:
    if not include:
        return ""
    return f"[{message.id}] {redact_text(message.body) or ''}"


def _max_stage(left: str, right: str) -> str:
    if ROUTE_STAGE_RANK.get(right, 0) > ROUTE_STAGE_RANK.get(left, 0):
        return right
    return left


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]
