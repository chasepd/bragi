"""Deterministic time-loop clock and selective reset policy."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import cast

from bragi.persistence.models import SceneSnapshotRecord, WorldStateRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.world_time_model import (
    CanonicalWorldTime,
    canonical_world_time_from_snapshot,
    canonical_world_time_from_values,
    legacy_world_time_fields,
)

LOOP_CURRENT_KEY = "loop.current"
LOOP_TIME_VERSION = 1
_RESETTABLE_CATEGORY = "loop_resettable"


@dataclass(frozen=True)
class LoopResetResult:
    snapshot: SceneSnapshotRecord
    iteration: int


class TimeLoopTimePolicy:
    """Owns typed time data embedded in the existing ``loop.current`` record."""

    def __init__(self, repositories: PersistenceRepositories, *, save_id: str) -> None:
        self.repositories = repositories
        self.save_id = save_id

    @property
    def is_time_loop(self) -> bool:
        details = self.repositories.load_save_details(self.save_id, message_limit=1)
        return details is not None and details.scenario.type == "time_loop"

    def capture_baseline(self, snapshot: SceneSnapshotRecord) -> None:
        if not self.is_time_loop:
            raise ValueError("Time-loop controls require a time-loop save")
        baseline_time = _time_payload(snapshot)
        if not any(baseline_time.values()):
            raise ValueError("Set a usable world time before capturing a loop baseline")
        current = self._current_record()
        value = self._envelope(current, snapshot=snapshot)
        value["baseline_time"] = baseline_time
        value["resettable_baseline"] = {
            record.key: _state_payload(record)
            for record in self._resettable_records()
        }
        value["last_transition"] = "baseline_captured"
        self._save_current(value, current)
        self._record_state_change(
            operation="upsert",
            key=LOOP_CURRENT_KEY,
            before=current.value if current is not None else None,
            after=value,
        )

    def ensure_baseline(self, snapshot: SceneSnapshotRecord) -> None:
        if not self.is_time_loop:
            return
        current = self._current_record()
        current_value = current.value if current is not None else {}
        if _time_payload_from_value(current_value) is not None:
            return
        if not any(_time_payload(snapshot).values()):
            return
        self.capture_baseline(snapshot)

    def sync_current(
        self,
        snapshot: SceneSnapshotRecord,
        *,
        transition: str,
        source_message_id: str | None = None,
    ) -> None:
        if not self.is_time_loop:
            return
        current = self._current_record()
        value = self._envelope(current, snapshot=snapshot)
        value["current_time"] = _time_payload(snapshot)
        value["last_transition"] = transition
        self._save_current(value, current, source_message_id=source_message_id)

    def baseline_time(self) -> CanonicalWorldTime | None:
        current = self._current_record()
        return _time_payload_from_value(current.value if current is not None else {})

    def reset(
        self,
        snapshot: SceneSnapshotRecord,
        *,
        source_message_id: str | None = None,
    ) -> LoopResetResult:
        if not self.is_time_loop:
            raise ValueError("Time-loop controls require a time-loop save")
        preview = self.preview_reset(snapshot)
        current = self._current_record()
        value = self._envelope(current, snapshot=snapshot)
        baseline_state = cast(dict[object, object], value["resettable_baseline"])
        for operation, key, before, after in self._restore_resettable_state(
            baseline_state
        ):
            self._record_state_change(
                operation=operation,
                key=key,
                before=before,
                after=after,
                source_message_id=source_message_id,
            )
        iteration = preview.iteration
        value["iteration"] = iteration
        value["current_time"] = _time_payload(preview.snapshot)
        value["last_transition"] = "reset"
        self._save_current(value, current, source_message_id=source_message_id)
        self._record_state_change(
            operation="upsert",
            key=LOOP_CURRENT_KEY,
            before=current.value if current is not None else None,
            after=value,
            source_message_id=source_message_id,
        )
        return preview

    def preview_reset(self, snapshot: SceneSnapshotRecord) -> LoopResetResult:
        if not self.is_time_loop:
            raise ValueError("Time-loop controls require a time-loop save")
        current = self._current_record()
        value = self._envelope(current, snapshot=snapshot)
        baseline = _time_payload_from_value(value)
        if baseline is None or not isinstance(value.get("resettable_baseline"), dict):
            raise ValueError("Capture a reset baseline before resetting the loop")
        return LoopResetResult(
            snapshot=_snapshot_with_world_time(snapshot, baseline),
            iteration=_positive_int(value.get("iteration")) + 1,
        )

    def _current_record(self) -> WorldStateRecord | None:
        return next(
            (
                record
                for record in self.repositories.list_world_state(self.save_id)
                if record.key == LOOP_CURRENT_KEY
            ),
            None,
        )

    def _resettable_records(self) -> tuple[WorldStateRecord, ...]:
        return tuple(
            record
            for record in self.repositories.list_world_state(self.save_id)
            if record.key != LOOP_CURRENT_KEY
            and record.key.startswith("loop.")
            and record.category == _RESETTABLE_CATEGORY
        )

    def _envelope(
        self,
        current: WorldStateRecord | None,
        *,
        snapshot: SceneSnapshotRecord,
    ) -> dict[str, object]:
        raw = current.value if current is not None else {}
        summary = raw.get("summary") if isinstance(raw.get("summary"), str) else ""
        value: dict[str, object] = {
            "version": LOOP_TIME_VERSION,
            "iteration": _positive_int(raw.get("iteration")),
            "summary": summary,
            "current_time": _time_payload(snapshot),
        }
        baseline = _time_payload_from_value(raw)
        if baseline is not None:
            value["baseline_time"] = _time_payload_from_canonical(baseline)
        baseline_state = raw.get("resettable_baseline")
        if isinstance(baseline_state, dict):
            value["resettable_baseline"] = baseline_state
        return value

    def _save_current(
        self,
        value: dict[str, object],
        current: WorldStateRecord | None,
        *,
        source_message_id: str | None = None,
    ) -> None:
        self.repositories.upsert_world_state(
            save_id=self.save_id,
            key=LOOP_CURRENT_KEY,
            value=value,
            category=current.category if current is not None else "loop_status",
            confidence=current.confidence if current is not None else 1.0,
            source_message_id=(
                source_message_id
                if source_message_id is not None
                else current.source_message_id if current is not None else None
            ),
        )

    def _restore_resettable_state(
        self,
        baseline_state: dict[object, object],
    ) -> list[tuple[str, str, dict[str, object] | None, dict[str, object] | None]]:
        changes: list[
            tuple[str, str, dict[str, object] | None, dict[str, object] | None]
        ] = []
        live_by_key = {
            record.key: record
            for record in self.repositories.list_world_state(self.save_id)
        }
        existing_by_key = {
            record.key: record.value for record in self._resettable_records()
        }
        baseline_keys = {
            key
            for key, value in baseline_state.items()
            if isinstance(key, str) and isinstance(value, dict)
        }
        for record in self._resettable_records():
            if record.key not in baseline_keys:
                self.repositories.archive_world_state(
                    save_id=self.save_id,
                    key=record.key,
                )
                changes.append(("delete", record.key, record.value, None))
        for key in sorted(baseline_keys):
            live_record = live_by_key.get(key)
            if (
                live_record is not None
                and live_record.category != _RESETTABLE_CATEGORY
            ):
                continue
            payload = cast(dict[str, object], baseline_state[key])
            value = payload.get("value")
            if not isinstance(value, dict):
                continue
            restored_source_message_id = payload.get("source_message_id")
            self.repositories.upsert_world_state(
                save_id=self.save_id,
                key=key,
                value=value,
                category=_string(payload.get("category")) or _RESETTABLE_CATEGORY,
                confidence=_confidence(payload.get("confidence")),
                source_message_id=(
                    restored_source_message_id
                    if isinstance(restored_source_message_id, str)
                    else None
                ),
            )
            changes.append(("upsert", key, existing_by_key.get(key), value))
        return changes

    def _record_state_change(
        self,
        *,
        operation: str,
        key: str,
        before: dict[str, object] | None,
        after: dict[str, object] | None,
        source_message_id: str | None = None,
    ) -> None:
        self.repositories.add_state_change(
            save_id=self.save_id,
            operation=operation,
            state_key=key,
            before_json=_state_json(before),
            after_json=_state_json(after),
            source_message_id=source_message_id,
        )


def _time_payload(snapshot: SceneSnapshotRecord) -> dict[str, object]:
    return _time_payload_from_canonical(canonical_world_time_from_snapshot(snapshot))


def write_scene_snapshot(
    repositories: PersistenceRepositories,
    snapshot: SceneSnapshotRecord,
) -> SceneSnapshotRecord:
    """Persist a snapshot value produced by the loop policy without changing locks."""
    return repositories.upsert_scene_snapshot(
        save_id=snapshot.save_id,
        current_location_id=snapshot.current_location_id,
        situation=snapshot.situation,
        objective=snapshot.objective,
        in_world_time=snapshot.in_world_time,
        time_of_day=snapshot.time_of_day,
        day_of_week=snapshot.day_of_week,
        world_day_index=snapshot.world_day_index,
        world_time_day_index=snapshot.world_time_day_index,
        world_time_day_label=snapshot.world_time_day_label,
        world_time_phase=snapshot.world_time_phase,
        world_time_clock_minutes=snapshot.world_time_clock_minutes,
        world_time_period_label=snapshot.world_time_period_label,
        world_time_source_message_id=snapshot.world_time_source_message_id,
        world_time_confidence=snapshot.world_time_confidence,
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


def _time_payload_from_canonical(time: CanonicalWorldTime) -> dict[str, object]:
    return {
        "day_index": time.day_index,
        "day_label": time.day_label,
        "phase": time.phase,
        "clock_minutes": time.clock_minutes,
        "period_label": time.period_label,
    }


def _time_payload_from_value(value: dict[str, object]) -> CanonicalWorldTime | None:
    payload = value.get("baseline_time")
    if not isinstance(payload, dict):
        return None
    time = canonical_world_time_from_values(
        day_index=payload.get("day_index"),
        day_label=payload.get("day_label"),
        phase=payload.get("phase"),
        clock_minutes=payload.get("clock_minutes"),
        period_label=payload.get("period_label"),
    )
    return time if any(_time_payload_from_canonical(time).values()) else None


def _state_payload(record: WorldStateRecord) -> dict[str, object]:
    return {
        "value": record.value,
        "category": record.category,
        "confidence": record.confidence,
        "source_message_id": record.source_message_id,
    }


def _snapshot_with_world_time(
    snapshot: SceneSnapshotRecord,
    time: CanonicalWorldTime,
) -> SceneSnapshotRecord:
    fields = legacy_world_time_fields(time)
    return replace(
        snapshot,
        in_world_time=cast(str, fields["in_world_time"]),
        time_of_day=cast(str, fields["time_of_day"]),
        day_of_week=cast(str, fields["day_of_week"]),
        world_day_index=cast(int | None, fields["world_day_index"]),
        world_time_day_index=time.day_index,
        world_time_day_label=time.day_label,
        world_time_phase=time.phase,
        world_time_clock_minutes=time.clock_minutes,
        world_time_period_label=time.period_label,
    )


def _positive_int(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 1


def _string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _confidence(value: object) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return max(0.0, min(1.0, float(value)))
    return 1.0


def _state_json(value: dict[str, object] | None) -> str | None:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"))
        if value is not None
        else None
    )
