"""Pre-narration deterministic world time maintenance."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, cast

from bragi.app_logging import exception_log_fields, log_error_event
from bragi.persistence.models import MessageRecord, SceneSnapshotRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatMessage,
    ProviderClient,
    StructuredOutputProvider,
    StructuredOutputRequest,
)
from bragi.services.openrouter_routing_settings import request_with_openrouter_routing
from bragi.services.provider_fallbacks import structured_output_with_fallback
from bragi.services.sexual_content_safety import is_fade_to_black_message
from bragi.services.time_loop_time_policy import TimeLoopTimePolicy
from bragi.services.world_time_model import (
    DAY_OF_WEEK_VALUES,
    TIME_OF_DAY_VALUES,
    canonical_world_time_from_values,
    format_world_time_from_snapshot,
    normalize_day_of_week,
    normalize_time_of_day,
    world_time_display,
)
from bragi.services.world_time_signals import (
    has_world_time_advance_signal,
    timer_readout_evidence_without_clock_advance,
)

WORLD_TIME_TASK = "context_update"
WORLD_TIME_SCHEMA_NAME = "world_time_advance"
WORLD_TIME_RECONCILIATION_SCHEMA_NAME = "world_time_reconciliation"
WORLD_TIME_CONFIDENCE_THRESHOLD = 0.65
NARRATOR_ONLY_WORLD_TIME_CONFIDENCE_THRESHOLD = 0.9


@dataclass(frozen=True)
class WorldTimeAssessment:
    changed: bool
    time_of_day: str = ""
    day_of_week: str = ""
    days_elapsed: int = 0
    loop_transition: str = ""
    clock_minutes: int | None = None
    period_label: str = ""
    evidence_source_id: str = ""
    evidence_quote: str = ""
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class WorldTimeResult:
    status: str
    changed: bool = False
    skipped_reason: str = ""
    queued_count: int = 0
    queued_suggestion_ids: tuple[str, ...] = ()
    source_message_ids: tuple[str, ...] = ()
    evidence_source_id: str = ""
    evidence_quote: str = ""
    confidence: float = 0.0
    reason: str = ""
    updated_fields: tuple[str, ...] = ()
    before: dict[str, object] = field(default_factory=dict)
    proposed: dict[str, object] = field(default_factory=dict)
    after: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        result: dict[str, object] = {
            "status": self.status,
            "changed": self.changed,
        }
        if self.skipped_reason:
            result["skipped_reason"] = self.skipped_reason
        if self.queued_count:
            result["queued_count"] = self.queued_count
        if self.queued_suggestion_ids:
            result["queued_suggestion_ids"] = list(self.queued_suggestion_ids)
        if self.source_message_ids:
            result["source_message_ids"] = list(self.source_message_ids)
        if self.evidence_source_id:
            result["evidence_source_id"] = self.evidence_source_id
        if self.evidence_quote:
            result["evidence_quote"] = self.evidence_quote
        if self.confidence:
            result["confidence"] = self.confidence
        if self.reason:
            result["reason"] = self.reason
        if self.updated_fields:
            result["updated_fields"] = list(self.updated_fields)
        if self.before:
            result["before"] = dict(self.before)
        if self.proposed:
            result["proposed"] = dict(self.proposed)
        if self.after:
            result["after"] = dict(self.after)
        return result


class WorldTimeChecker(Protocol):
    async def assess(
        self,
        *,
        save_id: str,
        latest_message: MessageRecord,
        snapshot: SceneSnapshotRecord,
    ) -> WorldTimeAssessment: ...


class CompletedTurnWorldTimeChecker(Protocol):
    async def assess_completed_turn(
        self,
        *,
        save_id: str,
        player_message: MessageRecord,
        narrator_message: MessageRecord,
        snapshot: SceneSnapshotRecord,
    ) -> WorldTimeAssessment: ...


class StructuredProviderWorldTimeChecker:
    def __init__(
        self,
        *,
        provider: StructuredOutputProvider,
        provider_name: str,
        model_id: str,
        repositories: PersistenceRepositories | None = None,
        providers: dict[str, ProviderClient] | None = None,
    ) -> None:
        self.provider = provider
        self.provider_name = provider_name
        self.model_id = model_id
        self.repositories = repositories
        self.providers = providers

    async def assess(
        self,
        *,
        save_id: str,
        latest_message: MessageRecord,
        snapshot: SceneSnapshotRecord,
    ) -> WorldTimeAssessment:
        request = request_with_openrouter_routing(
            self.repositories,
            StructuredOutputRequest(
                provider=self.provider_name,
                model_id=self.model_id,
                schema_name=WORLD_TIME_SCHEMA_NAME,
                schema=_world_time_schema(latest_message),
                messages=_world_time_messages(
                    latest_message=latest_message,
                    snapshot=snapshot,
                ),
                temperature=0.0,
            ),
            task=WORLD_TIME_TASK,
            save_id=save_id,
        )
        if self.repositories is not None and self.providers is not None:
            response = await structured_output_with_fallback(
                repositories=self.repositories,
                providers=self.providers,
                request=request,
                task=WORLD_TIME_TASK,
                save_id=save_id,
            )
        else:
            response = await self.provider.generate_structured_output(request)
        return _assessment_from_data(response.data)

    async def assess_completed_turn(
        self,
        *,
        save_id: str,
        player_message: MessageRecord,
        narrator_message: MessageRecord,
        snapshot: SceneSnapshotRecord,
    ) -> WorldTimeAssessment:
        request = request_with_openrouter_routing(
            self.repositories,
            StructuredOutputRequest(
                provider=self.provider_name,
                model_id=self.model_id,
                schema_name=WORLD_TIME_RECONCILIATION_SCHEMA_NAME,
                schema=_completed_world_time_schema(
                    player_message=player_message,
                    narrator_message=narrator_message,
                ),
                messages=_completed_world_time_messages(
                    player_message=player_message,
                    narrator_message=narrator_message,
                    snapshot=snapshot,
                ),
                temperature=0.0,
            ),
            task=WORLD_TIME_TASK,
            save_id=save_id,
        )
        if self.repositories is not None and self.providers is not None:
            response = await structured_output_with_fallback(
                repositories=self.repositories,
                providers=self.providers,
                request=request,
                task=WORLD_TIME_TASK,
                save_id=save_id,
            )
        else:
            response = await self.provider.generate_structured_output(request)
        return _assessment_from_data(response.data)


class WorldTimeService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        checker: WorldTimeChecker | None,
    ) -> None:
        self.repositories = repositories
        self.checker = checker

    async def advance_time_if_supported(
        self,
        *,
        save_id: str,
        latest_message_id: str,
    ) -> WorldTimeResult:
        latest_message = _active_message(self.repositories, save_id, latest_message_id)
        if latest_message is None:
            return WorldTimeResult("skipped", skipped_reason="unknown_latest_message")
        snapshot = self.repositories.get_scene_snapshot(save_id)
        if snapshot is None:
            return WorldTimeResult("skipped", skipped_reason="missing_scene_snapshot")
        if not latest_message_may_advance_time(latest_message.body):
            return WorldTimeResult("skipped", skipped_reason="no_time_advance_signal")
        if self.checker is None:
            return WorldTimeResult("skipped", skipped_reason="checker_unavailable")
        try:
            assessment = await self.checker.assess(
                save_id=save_id,
                latest_message=latest_message,
                snapshot=snapshot,
            )
        except Exception as exc:
            log_error_event(
                "world_time.assessment_failed",
                save_id=save_id,
                latest_message_id=latest_message_id,
                **exception_log_fields(exc),
            )
            return WorldTimeResult("failed", skipped_reason="checker_failed")
        return self._apply_assessment(
            save_id=save_id,
            snapshot=snapshot,
            assessment=assessment,
            candidate_messages=(latest_message,),
            source_message_ids=(latest_message.id,),
            policy="pre_turn",
        )

    async def reconcile_completed_turn(
        self,
        *,
        save_id: str,
        player_message_id: str,
        narrator_message_id: str,
    ) -> WorldTimeResult:
        player_message = _active_message(self.repositories, save_id, player_message_id)
        if player_message is None:
            return WorldTimeResult("skipped", skipped_reason="unknown_player_message")
        narrator_message = _active_message(
            self.repositories,
            save_id,
            narrator_message_id,
        )
        if narrator_message is None:
            return WorldTimeResult("skipped", skipped_reason="unknown_narrator_message")
        snapshot = self.repositories.get_scene_snapshot(save_id)
        if snapshot is None:
            return WorldTimeResult("skipped", skipped_reason="missing_scene_snapshot")
        if self.checker is None:
            return WorldTimeResult("skipped", skipped_reason="checker_unavailable")
        assess_completed_turn = getattr(self.checker, "assess_completed_turn", None)
        if not callable(assess_completed_turn):
            return WorldTimeResult("skipped", skipped_reason="checker_unavailable")
        try:
            assessment = await cast(
                CompletedTurnWorldTimeChecker,
                self.checker,
            ).assess_completed_turn(
                save_id=save_id,
                player_message=player_message,
                narrator_message=narrator_message,
                snapshot=snapshot,
            )
        except Exception as exc:
            log_error_event(
                "world_time.reconciliation_failed",
                save_id=save_id,
                player_message_id=player_message_id,
                narrator_message_id=narrator_message_id,
                **exception_log_fields(exc),
            )
            return WorldTimeResult("failed", skipped_reason="checker_failed")
        return self._apply_assessment(
            save_id=save_id,
            snapshot=snapshot,
            assessment=assessment,
            candidate_messages=(player_message, narrator_message),
            source_message_ids=(player_message.id, narrator_message.id),
            policy="completed_turn",
        )

    def _apply_assessment(
        self,
        *,
        save_id: str,
        snapshot: SceneSnapshotRecord,
        assessment: WorldTimeAssessment,
        candidate_messages: tuple[MessageRecord, ...],
        source_message_ids: tuple[str, ...],
        policy: str,
    ) -> WorldTimeResult:
        before_values = _snapshot_values(snapshot, {})
        source_message = _assessment_source_message(
            assessment,
            candidate_messages=candidate_messages,
        )

        def skipped_result(
            skipped_reason: str,
            *,
            proposed: dict[str, object] | None = None,
        ) -> WorldTimeResult:
            return WorldTimeResult(
                "skipped",
                skipped_reason=skipped_reason,
                source_message_ids=source_message_ids,
                evidence_source_id=assessment.evidence_source_id,
                evidence_quote=assessment.evidence_quote,
                confidence=assessment.confidence,
                reason=assessment.reason,
                before=before_values,
                proposed=proposed or {},
            )

        if not assessment.changed:
            return skipped_result("assessment_unchanged")
        if assessment.confidence < WORLD_TIME_CONFIDENCE_THRESHOLD:
            return skipped_result("low_confidence")
        if source_message is None:
            return skipped_result("invalid_evidence_source")
        if is_fade_to_black_message(
            role=source_message.role,
            body=source_message.body,
            safety_transition=source_message.safety_transition,
        ) and not _fade_transition_has_elapsed_time_evidence(
            assessment.evidence_quote,
        ):
            return skipped_result("safety_transition")
        if not _quote_matches_source(assessment.evidence_quote, source_message.body):
            return skipped_result("invalid_evidence_quote")
        if timer_readout_evidence_without_clock_advance(
            assessment.evidence_quote,
            source_message.body,
        ):
            return skipped_result("timer_readout_not_clock")
        loop_policy = TimeLoopTimePolicy(self.repositories, save_id=save_id)
        is_time_loop = loop_policy.is_time_loop
        changes: dict[str, object]
        loop_transition = (
            _loop_transition(assessment.loop_transition) if is_time_loop else ""
        )
        if is_time_loop and loop_transition == "reset":
            reset_locked_fields = (
                "time_of_day",
                "day_of_week",
                "world_day_index",
                "in_world_time",
                "world_time_clock_minutes",
                "world_time_period_label",
            )
            if any(
                _world_time_field_is_locked(snapshot, field)
                for field in reset_locked_fields
            ):
                return skipped_result("locked_fields")
            try:
                reset_snapshot = loop_policy.preview_reset(snapshot).snapshot
            except ValueError:
                return skipped_result("loop_baseline_unavailable")
            time_of_day = reset_snapshot.time_of_day
            day_of_week = reset_snapshot.day_of_week
            world_day_index = reset_snapshot.world_day_index
            changes = _time_snapshot_changes(snapshot, reset_snapshot)
            target_clock_minutes = reset_snapshot.world_time_clock_minutes
            target_period_label = reset_snapshot.world_time_period_label
        else:
            time_of_day = normalize_time_of_day(assessment.time_of_day)
            day_of_week = _target_day_of_week(snapshot, assessment)
            if loop_transition == "phase_advance":
                # A loop phase change moves only the clock.  In particular it
                # must not advance a calendar label while retaining the same
                # day index.
                day_of_week = snapshot.day_of_week
            changes = {}
            if time_of_day and time_of_day != snapshot.time_of_day:
                changes["time_of_day"] = time_of_day
            if day_of_week and day_of_week != snapshot.day_of_week:
                changes["day_of_week"] = day_of_week
            world_day_index = _target_world_day_index(
                snapshot=snapshot,
                assessment=assessment,
                day_of_week=day_of_week,
            )
            if loop_transition == "phase_advance":
                world_day_index = snapshot.world_day_index
            if world_day_index != snapshot.world_day_index:
                changes["world_day_index"] = world_day_index
            if any(field in changes for field in ("time_of_day", "day_of_week")):
                display_time = world_time_display(
                    time_of_day=time_of_day or snapshot.time_of_day,
                    day_of_week=day_of_week or snapshot.day_of_week,
                )
                if display_time and display_time != snapshot.in_world_time:
                    changes["in_world_time"] = display_time
            target_clock_minutes = (
                assessment.clock_minutes
                if assessment.clock_minutes is not None
                else snapshot.world_time_clock_minutes
            )
            target_period_label = (
                assessment.period_label or snapshot.world_time_period_label
            )
            if target_clock_minutes != snapshot.world_time_clock_minutes:
                changes["world_time_clock_minutes"] = target_clock_minutes
            if target_period_label != snapshot.world_time_period_label:
                changes["world_time_period_label"] = target_period_label
        proposed_values = _snapshot_values(snapshot, changes)
        if not changes and not (is_time_loop and loop_transition == "reset"):
            return skipped_result(
                "no_effective_change",
                proposed=proposed_values,
            )
        review_reason = _completed_turn_review_reason(
            policy=policy,
            assessment=assessment,
            source_message=source_message,
            candidate_messages=candidate_messages,
            proposed_values=proposed_values,
        )
        if review_reason:
            return self._queue_changes(
                save_id=save_id,
                snapshot=snapshot,
                source_message=source_message,
                source_message_ids=source_message_ids,
                assessment=assessment,
                changes=changes,
                before_values=before_values,
                proposed_values=proposed_values,
                skipped_reason=review_reason,
            )
        locked_changes = [
            field for field in changes if _world_time_field_is_locked(snapshot, field)
        ]
        if locked_changes:
            return self._queue_changes(
                save_id=save_id,
                snapshot=snapshot,
                source_message=source_message,
                source_message_ids=source_message_ids,
                assessment=assessment,
                changes=changes,
                before_values=before_values,
                proposed_values=proposed_values,
                skipped_reason="locked_fields",
            )
        updated = _snapshot_values(snapshot, changes)
        display_fields_changed = any(
            field in changes
            for field in (
                "time_of_day",
                "day_of_week",
                "world_time_clock_minutes",
                "world_time_period_label",
            )
        )
        canonical_world_time = canonical_world_time_from_values(
            day_index=updated["world_day_index"],
            day_label=updated["day_of_week"] or snapshot.world_time_day_label,
            phase=updated["time_of_day"],
            clock_minutes=target_clock_minutes,
            period_label=target_period_label,
            source_message_id=source_message.id,
            confidence=assessment.confidence,
            legacy_in_world_time=updated["in_world_time"],
            legacy_time_of_day=updated["time_of_day"],
            legacy_day_of_week=updated["day_of_week"],
            legacy_world_day_index=updated["world_day_index"],
        )
        world_time_kwargs: dict[str, object] = {
            "world_time_day_index": canonical_world_time.day_index,
            "world_time_source_message_id": canonical_world_time.source_message_id,
            "world_time_confidence": canonical_world_time.confidence,
        }
        if display_fields_changed:
            world_time_kwargs.update(
                {
                    "world_time_day_label": canonical_world_time.day_label,
                    "world_time_phase": canonical_world_time.phase,
                    "world_time_clock_minutes": canonical_world_time.clock_minutes,
                    "world_time_period_label": canonical_world_time.period_label,
                }
            )
        if is_time_loop:
            self.repositories.begin_transaction()
        try:
            if is_time_loop:
                # Capture the pre-turn time lazily, only after all review and
                # lock gates accept the proposed change.
                loop_policy.ensure_baseline(snapshot)
                if loop_transition == "reset":
                    loop_policy.reset(
                        snapshot,
                        source_message_id=source_message.id,
                    )
            self.repositories.upsert_scene_snapshot(
                save_id=save_id,
                current_location_id=snapshot.current_location_id,
                situation=snapshot.situation,
                objective=snapshot.objective,
                in_world_time=cast(str, updated["in_world_time"]),
                time_of_day=cast(str, updated["time_of_day"]),
                day_of_week=cast(str, updated["day_of_week"]),
                world_day_index=cast(int | None, updated["world_day_index"]),
                **world_time_kwargs,
                weather=snapshot.weather,
                mood=snapshot.mood,
                nearby_objects=snapshot.nearby_objects,
                hazards=snapshot.hazards,
                present_character_ids=snapshot.present_character_ids,
                source_message_id=source_message.id,
                locked_fields=snapshot.locked_fields,
                snapshot_id=snapshot.id,
                first_seen_message_id=snapshot.first_seen_message_id,
                last_updated_message_id=source_message.id,
            )
            for field_path, after in changes.items():
                before = getattr(snapshot, field_path)
                self.repositories.add_context_update_audit(
                    save_id=save_id,
                    operation="updated",
                    entity_type="scene_snapshot",
                    entity_id=snapshot.id,
                    field_path=field_path,
                    before=before,
                    after=after,
                    reason=assessment.reason,
                    confidence=assessment.confidence,
                    source_message_ids=list(source_message_ids),
                )
            if is_time_loop:
                updated_snapshot = self.repositories.get_scene_snapshot(save_id)
                if updated_snapshot is not None:
                    loop_policy.ensure_baseline(updated_snapshot)
                    loop_policy.sync_current(
                        updated_snapshot,
                        transition=loop_transition or "ordinary_elapsed",
                        source_message_id=source_message.id,
                    )
        except Exception:
            if is_time_loop:
                self.repositories.rollback_transaction()
            raise
        else:
            if is_time_loop:
                self.repositories.commit_transaction()
        return WorldTimeResult(
            "applied",
            changed=True,
            source_message_ids=source_message_ids,
            evidence_source_id=assessment.evidence_source_id,
            evidence_quote=assessment.evidence_quote,
            confidence=assessment.confidence,
            reason=assessment.reason,
            updated_fields=tuple(changes),
            before=before_values,
            proposed=proposed_values,
            after=updated,
        )

    def _queue_changes(
        self,
        *,
        save_id: str,
        snapshot: SceneSnapshotRecord,
        source_message: MessageRecord,
        source_message_ids: tuple[str, ...],
        assessment: WorldTimeAssessment,
        changes: dict[str, object],
        before_values: dict[str, object],
        proposed_values: dict[str, object],
        skipped_reason: str,
    ) -> WorldTimeResult:
        suggestion_ids: list[str] = []
        for field_path, proposed_value in changes.items():
            suggestion = self.repositories.add_context_update_suggestion(
                save_id=save_id,
                update_type="update",
                entity_type="scene_snapshot",
                entity_id=snapshot.id,
                field_path=field_path,
                proposed_value=proposed_value,
                status="pending",
                reason=assessment.reason,
                confidence=assessment.confidence,
                source_message_ids=list(source_message_ids),
            )
            suggestion_ids.append(suggestion.id)
            self.repositories.add_context_update_audit(
                save_id=save_id,
                suggestion_id=suggestion.id,
                operation="queued",
                entity_type="scene_snapshot",
                entity_id=snapshot.id,
                field_path=field_path,
                before=getattr(snapshot, field_path),
                after=proposed_value,
                reason=assessment.reason,
                confidence=assessment.confidence,
                source_message_ids=list(source_message_ids),
            )
        return WorldTimeResult(
            "queued",
            skipped_reason=skipped_reason,
            queued_count=len(changes),
            queued_suggestion_ids=tuple(suggestion_ids),
            source_message_ids=source_message_ids,
            evidence_source_id=assessment.evidence_source_id or source_message.id,
            evidence_quote=assessment.evidence_quote,
            confidence=assessment.confidence,
            reason=assessment.reason,
            updated_fields=tuple(changes),
            before=before_values,
            proposed=proposed_values,
            after=before_values,
        )


def latest_message_may_advance_time(text: str) -> bool:
    return has_world_time_advance_signal(text)


def _fade_transition_has_elapsed_time_evidence(evidence_quote: str) -> bool:
    normalized = " ".join(evidence_quote.casefold().split())
    return "hours later" in normalized and "next scene begins" in normalized


def _world_time_schema(latest_message: MessageRecord) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "changed": {"type": "boolean"},
            "time_of_day": {"type": "string", "enum": ["", *TIME_OF_DAY_VALUES]},
            "day_of_week": {"type": "string", "enum": ["", *DAY_OF_WEEK_VALUES]},
            "days_elapsed": {"type": "integer", "minimum": 0, "maximum": 6},
            "loop_transition": {
                "type": "string",
                "enum": ["", "phase_advance", "ordinary_elapsed", "reset"],
            },
            "clock_minutes": {
                "type": ["integer", "null"],
                "minimum": 0,
                "maximum": 1439,
            },
            "period_label": {"type": "string"},
            "evidence_source_id": {
                "type": "string",
                "enum": ["", latest_message.id],
            },
            "evidence_quote": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
        "required": [
            "changed",
            "time_of_day",
            "day_of_week",
            "days_elapsed",
            "loop_transition",
            "clock_minutes",
            "period_label",
            "evidence_source_id",
            "evidence_quote",
            "confidence",
            "reason",
        ],
    }


def _completed_world_time_schema(
    *,
    player_message: MessageRecord,
    narrator_message: MessageRecord,
) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "changed": {"type": "boolean"},
            "time_of_day": {"type": "string", "enum": ["", *TIME_OF_DAY_VALUES]},
            "day_of_week": {"type": "string", "enum": ["", *DAY_OF_WEEK_VALUES]},
            "days_elapsed": {"type": "integer", "minimum": 0, "maximum": 6},
            "loop_transition": {
                "type": "string",
                "enum": ["", "phase_advance", "ordinary_elapsed", "reset"],
            },
            "clock_minutes": {
                "type": ["integer", "null"],
                "minimum": 0,
                "maximum": 1439,
            },
            "period_label": {"type": "string"},
            "evidence_source_id": {
                "type": "string",
                "enum": ["", player_message.id, narrator_message.id],
            },
            "evidence_quote": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
        "required": [
            "changed",
            "time_of_day",
            "day_of_week",
            "days_elapsed",
            "loop_transition",
            "clock_minutes",
            "period_label",
            "evidence_source_id",
            "evidence_quote",
            "confidence",
            "reason",
        ],
    }


def _world_time_messages(
    *,
    latest_message: MessageRecord,
    snapshot: SceneSnapshotRecord,
) -> tuple[ChatMessage, ...]:
    display = format_world_time_from_snapshot(snapshot) or "unknown"
    return (
        ChatMessage(
            role="system",
            body=(
                "Assess whether the latest Bragi message explicitly advances "
                "the current in-world time. Use the enforced schema. Return "
                "changed=false when time does not clearly advance. For a time-loop "
                "save, classify a confirmed reset as reset, a same-loop clock change "
                "as phase_advance, and a normal calendar jump as ordinary_elapsed."
            ),
        ),
        ChatMessage(
            role="user",
            body=(
                "Current world time:\n"
                f"- day_of_week: {snapshot.day_of_week or 'unknown'}\n"
                f"- time_of_day: {snapshot.time_of_day or 'unknown'}\n"
                f"- world_day_index: {_world_day_index_text(snapshot)}\n"
                f"- display: {display}\n\n"
                "Latest message:\n"
                f"[{latest_message.id}] {latest_message.role}: "
                f"{latest_message.body}"
            ),
        ),
    )


def _completed_world_time_messages(
    *,
    player_message: MessageRecord,
    narrator_message: MessageRecord,
    snapshot: SceneSnapshotRecord,
) -> tuple[ChatMessage, ...]:
    display = format_world_time_from_snapshot(snapshot) or "unknown"
    return (
        ChatMessage(
            role="system",
            body=(
                "Assess whether the completed Bragi turn establishes a durable "
                "in-world time change. Use the enforced schema. Prefer player "
                "authorization when present, and return changed=false when the "
                "turn does not clearly establish time. For time-loop saves, classify "
                "a confirmed reset as reset, a same-loop clock change as "
                "phase_advance, and a normal calendar jump as ordinary_elapsed. "
                "A marked narrator safety transition is canonical evidence that "
                "the intimate moment stayed off-screen and time passed; use it "
                "only to support elapsed time, without inventing exact details."
            ),
        ),
        ChatMessage(
            role="user",
            body=(
                "Current world time before the turn:\n"
                f"- day_of_week: {snapshot.day_of_week or 'unknown'}\n"
                f"- time_of_day: {snapshot.time_of_day or 'unknown'}\n"
                f"- world_day_index: {_world_day_index_text(snapshot)}\n"
                f"- display: {display}\n\n"
                "Completed turn messages:\n"
                f"[{player_message.id}] {player_message.role}: "
                f"{player_message.body}\n"
                f"[{narrator_message.id}] {narrator_message.role}: "
                f"{narrator_message.body}"
            ),
        ),
    )


def _assessment_from_data(data: dict[str, object]) -> WorldTimeAssessment:
    return WorldTimeAssessment(
        changed=bool(data.get("changed")),
        time_of_day=normalize_time_of_day(data.get("time_of_day")),
        day_of_week=normalize_day_of_week(data.get("day_of_week")),
        days_elapsed=_bounded_days_elapsed(data.get("days_elapsed")),
        loop_transition=_loop_transition(data.get("loop_transition")),
        clock_minutes=_clock_minutes(data.get("clock_minutes")),
        period_label=_string(data.get("period_label")),
        evidence_source_id=_string(data.get("evidence_source_id")),
        evidence_quote=_string(data.get("evidence_quote")),
        confidence=_confidence(data.get("confidence")),
        reason=_string(data.get("reason")),
    )


def _target_day_of_week(
    snapshot: SceneSnapshotRecord,
    assessment: WorldTimeAssessment,
) -> str:
    if assessment.day_of_week:
        return assessment.day_of_week
    if not snapshot.day_of_week or assessment.days_elapsed <= 0:
        return ""
    try:
        index = DAY_OF_WEEK_VALUES.index(snapshot.day_of_week)
    except ValueError:
        return ""
    return DAY_OF_WEEK_VALUES[
        (index + assessment.days_elapsed) % len(DAY_OF_WEEK_VALUES)
    ]


def _target_world_day_index(
    *,
    snapshot: SceneSnapshotRecord,
    assessment: WorldTimeAssessment,
    day_of_week: str,
) -> int | None:
    if assessment.days_elapsed > 0:
        base = snapshot.world_day_index if snapshot.world_day_index is not None else 0
        return base + assessment.days_elapsed
    if day_of_week and snapshot.day_of_week and day_of_week != snapshot.day_of_week:
        try:
            current_index = DAY_OF_WEEK_VALUES.index(snapshot.day_of_week)
            target_index = DAY_OF_WEEK_VALUES.index(day_of_week)
        except ValueError:
            return snapshot.world_day_index
        base = snapshot.world_day_index if snapshot.world_day_index is not None else 0
        return base + ((target_index - current_index) % len(DAY_OF_WEEK_VALUES))
    return snapshot.world_day_index


def _world_time_field_is_locked(
    snapshot: SceneSnapshotRecord,
    field_path: str,
) -> bool:
    locked_fields = set(snapshot.locked_fields)
    if field_path in locked_fields:
        return True
    aliases = {
        "time_of_day": ("world_time_phase",),
        "day_of_week": ("world_time_day_label",),
        "world_day_index": ("world_time_day_index",),
    }
    return any(alias in locked_fields for alias in aliases.get(field_path, ()))


def _snapshot_values(
    snapshot: SceneSnapshotRecord,
    changes: dict[str, object],
) -> dict[str, object]:
    return {
        "in_world_time": changes.get("in_world_time", snapshot.in_world_time),
        "time_of_day": changes.get("time_of_day", snapshot.time_of_day),
        "day_of_week": changes.get("day_of_week", snapshot.day_of_week),
        "world_day_index": changes.get("world_day_index", snapshot.world_day_index),
    }


def _time_snapshot_changes(
    before: SceneSnapshotRecord,
    after: SceneSnapshotRecord,
) -> dict[str, object]:
    return {
        field: getattr(after, field)
        for field in (
            "in_world_time",
            "time_of_day",
            "day_of_week",
            "world_day_index",
            "world_time_clock_minutes",
            "world_time_period_label",
        )
        if getattr(before, field) != getattr(after, field)
    }


def _assessment_source_message(
    assessment: WorldTimeAssessment,
    *,
    candidate_messages: tuple[MessageRecord, ...],
) -> MessageRecord | None:
    for message in candidate_messages:
        if message.id == assessment.evidence_source_id:
            return message
    return None


def _completed_turn_review_reason(
    *,
    policy: str,
    assessment: WorldTimeAssessment,
    source_message: MessageRecord,
    candidate_messages: tuple[MessageRecord, ...],
    proposed_values: dict[str, object],
) -> str:
    if policy != "completed_turn":
        return ""
    player_message = next(
        (message for message in candidate_messages if message.role == "player"),
        None,
    )
    if player_message is None:
        return ""
    source_is_narrator = source_message.role == "narrator"
    player_authorized = latest_message_may_advance_time(player_message.body)
    if any(
        _message_time_conflicts_with_proposal(
            message.body,
            proposed_values=proposed_values,
        )
        for message in candidate_messages
    ):
        return "conflicting_time_evidence"
    if is_fade_to_black_message(
        role=source_message.role,
        body=source_message.body,
        safety_transition=source_message.safety_transition,
    ) and not _fade_transition_has_elapsed_time_evidence(
        assessment.evidence_quote,
    ):
        return "safety_transition"
    if (
        source_is_narrator
        and not player_authorized
        and not is_fade_to_black_message(
            role=source_message.role,
            body=source_message.body,
            safety_transition=source_message.safety_transition,
        )
        and assessment.confidence < NARRATOR_ONLY_WORLD_TIME_CONFIDENCE_THRESHOLD
    ):
        return "narrator_only_ambiguous"
    return ""


def _message_time_conflicts_with_proposal(
    text: str,
    *,
    proposed_values: dict[str, object],
) -> bool:
    proposed_time = proposed_values.get("time_of_day")
    proposed_day = proposed_values.get("day_of_week")
    mentioned_times = _mentioned_time_of_day_values(text)
    if (
        mentioned_times
        and isinstance(proposed_time, str)
        and proposed_time
        and proposed_time not in mentioned_times
    ):
        return True
    mentioned_days = _mentioned_day_of_week_values(text)
    return (
        bool(mentioned_days)
        and isinstance(proposed_day, str)
        and bool(proposed_day)
        and proposed_day not in mentioned_days
    )


def _mentioned_time_of_day_values(text: str) -> frozenset[str]:
    normalized = text.casefold()
    matches: set[str] = set()
    phrase_values = {
        "late morning": "late_morning",
        "late-morning": "late_morning",
        "morning": "morning",
        "dawn": "morning",
        "sunrise": "morning",
        "noon": "afternoon",
        "midday": "afternoon",
        "afternoon": "afternoon",
        "dusk": "evening",
        "sunset": "evening",
        "evening": "evening",
        "midnight": "night",
        "late night": "night",
        "night": "night",
    }
    for phrase, value in phrase_values.items():
        if re.search(rf"(?<![\w-]){re.escape(phrase)}(?![\w-])", normalized):
            matches.add(value)
    return frozenset(matches)


def _mentioned_day_of_week_values(text: str) -> frozenset[str]:
    normalized = text.casefold()
    return frozenset(
        day
        for day in DAY_OF_WEEK_VALUES
        if re.search(rf"(?<![\w-]){day}(?![\w-])", normalized)
    )


def _active_message(
    repositories: PersistenceRepositories,
    save_id: str,
    message_id: str,
) -> MessageRecord | None:
    for message in repositories.list_messages(save_id):
        if message.id == message_id:
            return message
    return None


def _quote_matches_source(quote: str, source_body: str) -> bool:
    normalized_quote = _normalized_text(quote)
    if not normalized_quote:
        return False
    return normalized_quote in _normalized_text(source_body)


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def _bounded_days_elapsed(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, min(6, value))
    return 0


def _loop_transition(value: object) -> str:
    return value if value in {"phase_advance", "ordinary_elapsed", "reset"} else ""


def _clock_minutes(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 24 * 60:
        return value
    return None


def _confidence(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return 0.0


def _string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _world_day_index_text(snapshot: SceneSnapshotRecord) -> str:
    if snapshot.world_day_index is None:
        return "unknown"
    return str(snapshot.world_day_index)
