from __future__ import annotations

import asyncio
import threading

import pytest

from bragi_web.runtime import RuntimeAccessLock, SaveEventHub


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
        lock.close()

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
        lock.close()

    asyncio.run(run_test())


def test_runtime_access_lock_async_access_uses_private_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def blocked_to_thread(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("runtime lock waits must not use the default executor")

    async def run_test() -> None:
        lock = RuntimeAccessLock()
        monkeypatch.setattr("bragi_web.runtime.asyncio.to_thread", blocked_to_thread)
        try:
            async with lock.async_access():
                pass
        finally:
            lock.close()

    asyncio.run(run_test())


def test_runtime_access_lock_close_does_not_disable_sync_access() -> None:
    lock = RuntimeAccessLock()
    lock.close()

    with lock:
        pass


def test_runtime_access_lock_close_does_not_strand_queued_async_access() -> None:
    async def enter_async_lock(lock: RuntimeAccessLock) -> None:
        async with lock.async_access():
            pass

    async def run_test() -> None:
        lock = RuntimeAccessLock()
        first_entered = asyncio.Event()
        release_first = asyncio.Event()

        async def hold_first() -> None:
            async with lock.async_access():
                first_entered.set()
                await release_first.wait()

        first = asyncio.create_task(hold_first())
        await asyncio.wait_for(first_entered.wait(), timeout=1.0)

        second = asyncio.create_task(enter_async_lock(lock))
        await asyncio.sleep(0)
        lock.close()

        release_first.set()
        await asyncio.wait_for(first, timeout=1.0)
        await asyncio.wait_for(second, timeout=1.0)
        lock.close()

    asyncio.run(run_test())


def test_save_event_hub_wait_for_event_does_not_use_default_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def blocked_to_thread(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("save event waits must not use the default executor")

    async def run_test() -> None:
        hub = SaveEventHub()
        monkeypatch.setattr("bragi_web.runtime.asyncio.to_thread", blocked_to_thread)

        waiter = asyncio.create_task(hub.wait_for_event("save-1", 0))
        await asyncio.sleep(0)
        hub.publish("save-1", "runtime_changed", {})

        assert await asyncio.wait_for(waiter, timeout=1.0) == 0

    asyncio.run(run_test())


def _hold_sync_lock(
    lock: RuntimeAccessLock,
    entered: threading.Event,
    release: threading.Event,
) -> None:
    with lock:
        entered.set()
        release.wait()
