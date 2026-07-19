from __future__ import annotations

import asyncio
import threading

import pytest

from bragi_web.runtime import RuntimeAccessLock


def test_runtime_access_lock_excludes_sync_access_during_async_access() -> None:
    async def run_test() -> None:
        lock = RuntimeAccessLock()
        async_entered = threading.Event()
        release_async = threading.Event()
        sync_entered = threading.Event()

        async def hold_async_access() -> None:
            async with lock.async_access():
                async_entered.set()
                await asyncio.to_thread(release_async.wait)

        task = asyncio.create_task(hold_async_access())
        assert await asyncio.to_thread(async_entered.wait, 1.0)

        sync_thread = threading.Thread(
            target=lambda: _enter_sync_lock(lock, sync_entered),
        )
        sync_thread.start()
        await asyncio.sleep(0.05)

        assert not sync_entered.is_set()
        release_async.set()
        await task
        sync_thread.join(timeout=1.0)

        assert sync_entered.is_set()

    asyncio.run(run_test())


def _enter_sync_lock(lock: RuntimeAccessLock, entered: threading.Event) -> None:
    with lock:
        entered.set()


def test_runtime_access_lock_does_not_block_event_loop_when_contended() -> None:
    async def run_test() -> None:
        lock = RuntimeAccessLock()
        sync_entered = threading.Event()
        release_sync = threading.Event()

        thread = threading.Thread(
            target=lambda: _hold_sync_lock(lock, sync_entered, release_sync),
        )
        thread.start()
        assert await asyncio.to_thread(sync_entered.wait, 1.0)

        with pytest.raises(RuntimeError, match="event loop"):
            with lock:
                raise AssertionError("contended event-loop acquire should fail")

        await asyncio.wait_for(asyncio.sleep(0), timeout=0.1)
        release_sync.set()
        thread.join(timeout=1.0)

        assert not thread.is_alive()

    asyncio.run(run_test())


def _hold_sync_lock(
    lock: RuntimeAccessLock,
    entered: threading.Event,
    release: threading.Event,
) -> None:
    with lock:
        entered.set()
        release.wait()
