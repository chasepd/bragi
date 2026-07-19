"""Canonical world-time value helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

TIME_OF_DAY_VALUES = (
    "morning",
    "late_morning",
    "afternoon",
    "evening",
    "night",
)
DAY_OF_WEEK_VALUES = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)

_TIME_OF_DAY_ALIASES = {
    "late morning": "late_morning",
    "late-morning": "late_morning",
    "midday": "afternoon",
    "noon": "afternoon",
    "dusk": "evening",
    "sunset": "evening",
    "dawn": "morning",
    "sunrise": "morning",
    "midnight": "night",
    "late night": "night",
}
_MAX_LABEL_LENGTH = 80
_CLOCK_RE = re.compile(
    r"\b([01]?\d|2[0-3])(?::([0-5]\d))?\s*([ap]\.?m\.?)?\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class CanonicalWorldTime:
    day_index: int | None = None
    day_label: str = ""
    phase: str = ""
    clock_minutes: int | None = None
    period_label: str = ""
    source_message_id: str | None = None
    confidence: float | None = None
    legacy_label: str = ""


def normalize_time_of_day(value: object) -> str:
    if not isinstance(value, str):
        return ""
    stripped = value.strip()
    normalized = re.sub(r"[\s-]+", "_", stripped.casefold())
    normalized = _TIME_OF_DAY_ALIASES.get(stripped.casefold(), normalized)
    return normalized if normalized in TIME_OF_DAY_VALUES else ""


def normalize_day_of_week(value: object) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip().casefold()
    return normalized if normalized in DAY_OF_WEEK_VALUES else ""


def normalize_world_time_day_label(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = _compact_text(value)
    if not text:
        return ""
    day_of_week = normalize_day_of_week(text)
    return day_of_week or text[:_MAX_LABEL_LENGTH]


def normalize_world_time_period_label(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return _compact_text(value)[:_MAX_LABEL_LENGTH]


def normalize_world_time_clock_minutes(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and 0 <= value < 24 * 60:
        return value
    return None


def normalize_world_time_confidence(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return max(0.0, min(1.0, float(value)))
    return None


def canonical_world_time_from_snapshot(snapshot: object | None) -> CanonicalWorldTime:
    if snapshot is None:
        return CanonicalWorldTime()
    return canonical_world_time_from_values(
        day_index=getattr(snapshot, "world_time_day_index", None),
        day_label=getattr(snapshot, "world_time_day_label", ""),
        phase=getattr(snapshot, "world_time_phase", ""),
        clock_minutes=getattr(snapshot, "world_time_clock_minutes", None),
        period_label=getattr(snapshot, "world_time_period_label", ""),
        source_message_id=getattr(snapshot, "world_time_source_message_id", None),
        confidence=getattr(snapshot, "world_time_confidence", None),
        legacy_in_world_time=getattr(snapshot, "in_world_time", ""),
        legacy_time_of_day=getattr(snapshot, "time_of_day", ""),
        legacy_day_of_week=getattr(snapshot, "day_of_week", ""),
        legacy_world_day_index=getattr(snapshot, "world_day_index", None),
    )


def canonical_world_time_from_values(
    *,
    day_index: object = None,
    day_label: object = "",
    phase: object = "",
    clock_minutes: object = None,
    period_label: object = "",
    source_message_id: object = None,
    confidence: object = None,
    legacy_in_world_time: object = "",
    legacy_time_of_day: object = "",
    legacy_day_of_week: object = "",
    legacy_world_day_index: object = None,
) -> CanonicalWorldTime:
    canonical = CanonicalWorldTime(
        day_index=_optional_nonnegative_int(day_index),
        day_label=normalize_world_time_day_label(day_label),
        phase=normalize_time_of_day(phase),
        clock_minutes=normalize_world_time_clock_minutes(clock_minutes),
        period_label=normalize_world_time_period_label(period_label),
        source_message_id=(
            source_message_id.strip()
            if isinstance(source_message_id, str) and source_message_id.strip()
            else None
        ),
        confidence=normalize_world_time_confidence(confidence),
        legacy_label=_compact_text(legacy_in_world_time),
    )
    if _has_canonical_value(canonical):
        return canonical
    return canonical_world_time_from_legacy(
        in_world_time=legacy_in_world_time,
        time_of_day=legacy_time_of_day,
        day_of_week=legacy_day_of_week,
        world_day_index=legacy_world_day_index,
        source_message_id=source_message_id,
        confidence=confidence,
    )


def canonical_world_time_from_legacy(
    *,
    in_world_time: object = "",
    time_of_day: object = "",
    day_of_week: object = "",
    world_day_index: object = None,
    source_message_id: object = None,
    confidence: object = None,
) -> CanonicalWorldTime:
    legacy_label = _compact_text(in_world_time)
    phase = normalize_time_of_day(time_of_day) or _phase_from_legacy_label(
        legacy_label
    )
    return CanonicalWorldTime(
        day_index=_optional_nonnegative_int(world_day_index),
        day_label=normalize_world_time_day_label(day_of_week),
        phase=phase,
        clock_minutes=clock_minutes_from_text(in_world_time),
        source_message_id=(
            source_message_id.strip()
            if isinstance(source_message_id, str) and source_message_id.strip()
            else None
        ),
        confidence=normalize_world_time_confidence(confidence),
        legacy_label=legacy_label,
    )


def format_world_time(
    world_time: CanonicalWorldTime,
    *,
    include_legacy_detail: bool = False,
) -> str:
    parts: list[str] = []
    day_label = _display_label(world_time.day_label)
    if day_label:
        parts.append(day_label)
    if world_time.period_label:
        parts.append(world_time.period_label)
    time_text = _time_display(world_time)
    if time_text:
        parts.append(time_text)
    display = " ".join(parts).strip()
    legacy_detail = (
        _legacy_detail_label(world_time, display)
        if include_legacy_detail
        else ""
    )
    if world_time.day_index is not None:
        day_index_text = f"world day index {world_time.day_index}"
        if legacy_detail:
            return f"{legacy_detail}; {day_index_text}"
        if display:
            return f"{display}; {day_index_text}"
        return day_index_text
    if legacy_detail:
        return legacy_detail
    if display:
        return display
    return world_time.legacy_label


def format_world_time_from_snapshot(
    snapshot: object | None,
    *,
    include_legacy_detail: bool = False,
) -> str:
    return format_world_time(
        canonical_world_time_from_snapshot(snapshot),
        include_legacy_detail=include_legacy_detail,
    )


def _legacy_detail_label(world_time: CanonicalWorldTime, display: str) -> str:
    legacy_label = " ".join(world_time.legacy_label.split()).strip()
    if not legacy_label or legacy_label == display:
        return ""
    normalized = legacy_label.casefold().replace("_", " ")
    required_parts = [
        part.casefold().replace("_", " ")
        for part in (
            _display_label(world_time.day_label),
            normalize_time_of_day(world_time.phase).replace("_", " "),
        )
        if part
    ]
    if required_parts and not all(part in normalized for part in required_parts):
        return ""
    return legacy_label


def world_time_display(*, time_of_day: str, day_of_week: str) -> str:
    parts = [
        normalize_world_time_day_label(day_of_week).replace("_", " ").title(),
        normalize_time_of_day(time_of_day).replace("_", " "),
    ]
    return " ".join(part for part in parts if part.strip()).strip()


def legacy_world_time_fields(world_time: CanonicalWorldTime) -> dict[str, object]:
    legacy_label = format_world_time(
        CanonicalWorldTime(
            day_label=world_time.day_label,
            phase=world_time.phase,
            clock_minutes=world_time.clock_minutes,
            period_label=world_time.period_label,
            legacy_label=world_time.legacy_label,
        )
    )
    return {
        "in_world_time": legacy_label,
        "time_of_day": world_time.phase,
        "day_of_week": (
            world_time.day_label if world_time.day_label in DAY_OF_WEEK_VALUES else ""
        ),
        "world_day_index": world_time.day_index,
    }


def clock_minutes_from_text(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    for match in _CLOCK_RE.finditer(text):
        meridiem = match.group(3)
        minute_text = match.group(2)
        if meridiem is None and minute_text is None:
            continue
        hour = int(match.group(1))
        minute = int(minute_text or "0")
        if meridiem is not None:
            meridiem_text = meridiem.replace(".", "").casefold()
            if meridiem_text == "am" and hour == 12:
                hour = 0
            elif meridiem_text == "pm" and hour != 12:
                hour += 12
        return hour * 60 + minute
    return None


def _time_display(world_time: CanonicalWorldTime) -> str:
    phase = world_time.phase.replace("_", " ")
    clock = _clock_display(world_time.clock_minutes)
    if phase and clock:
        return f"{phase} at {clock}"
    return phase or clock


def _clock_display(clock_minutes: int | None) -> str:
    if clock_minutes is None:
        return ""
    hour = clock_minutes // 60
    minute = clock_minutes % 60
    return f"{hour:02d}:{minute:02d}"


def _display_label(value: str) -> str:
    if value in DAY_OF_WEEK_VALUES:
        return value.title()
    return value


def _phase_from_legacy_label(value: str) -> str:
    direct = normalize_time_of_day(value)
    if direct:
        return direct
    for parenthetical in re.findall(r"\(([^)]+)\)", value):
        phase = normalize_time_of_day(parenthetical)
        if phase:
            return phase
    text = re.sub(r"[_-]+", " ", value.casefold())
    for phase in sorted(TIME_OF_DAY_VALUES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phase.replace('_', ' '))}\b", text):
            return phase
    for alias, phase in sorted(
        _TIME_OF_DAY_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return phase
    return ""


def _compact_text(value: object) -> str:
    return " ".join(value.strip().split()) if isinstance(value, str) else ""


def _optional_nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _has_canonical_value(world_time: CanonicalWorldTime) -> bool:
    return any(
        value not in (None, "")
        for value in (
            world_time.day_index,
            world_time.day_label,
            world_time.phase,
            world_time.clock_minutes,
            world_time.period_label,
        )
    )
