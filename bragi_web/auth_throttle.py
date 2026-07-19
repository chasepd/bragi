"""In-memory throttling for local web authentication attempts."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from time import time


@dataclass(frozen=True)
class AuthThrottleConfig:
    max_failures: int = 5
    window_seconds: float = 10 * 60.0
    lockout_seconds: float = 10 * 60.0
    max_keys: int = 4096


@dataclass
class _AuthAttemptRecord:
    failures: int
    first_failure_at: float
    last_seen_at: float
    locked_until: float | None = None


class AuthAttemptThrottle:
    def __init__(
        self,
        *,
        config: AuthThrottleConfig | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.config = config or AuthThrottleConfig()
        self._clock = clock or time
        self._records: dict[tuple[str, str, str], _AuthAttemptRecord] = {}
        self._lock = threading.RLock()

    def blocked_for_seconds(self, key: tuple[str, str, str]) -> int | None:
        now = self._clock()
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return None
            if self._record_expired(record, now):
                self._records.pop(key, None)
                return None
            record.last_seen_at = now
            if record.locked_until is None or record.locked_until <= now:
                return None
            return max(1, int(record.locked_until - now))

    def record_failure(self, key: tuple[str, str, str]) -> None:
        now = self._clock()
        with self._lock:
            self._prune(now)
            record = self._records.get(key)
            if record is None or self._record_expired(record, now):
                record = _AuthAttemptRecord(
                    failures=0,
                    first_failure_at=now,
                    last_seen_at=now,
                )
                self._records[key] = record
            record.failures += 1
            record.last_seen_at = now
            if record.failures >= self.config.max_failures:
                record.locked_until = now + self.config.lockout_seconds

    def record_success(self, key: tuple[str, str, str]) -> None:
        with self._lock:
            self._records.pop(key, None)

    def _record_expired(self, record: _AuthAttemptRecord, now: float) -> bool:
        if record.locked_until is not None and record.locked_until > now:
            return False
        return now - record.first_failure_at >= self.config.window_seconds

    def _prune(self, now: float) -> None:
        expired = [
            key
            for key, record in self._records.items()
            if self._record_expired(record, now)
        ]
        for key in expired:
            self._records.pop(key, None)
        if len(self._records) <= self.config.max_keys:
            return
        overflow = len(self._records) - self.config.max_keys
        oldest = sorted(
            self._records,
            key=lambda key: self._records[key].last_seen_at,
        )
        for key in oldest[:overflow]:
            self._records.pop(key, None)
