"""Monotonic high-resolution clock used for every timestamp in the library.

``time.monotonic()`` has ~15.6 ms resolution on Windows, which is too coarse for
latency metrics, so all timestamps use :func:`time.perf_counter` instead.
"""

from __future__ import annotations

import asyncio
import time

__all__ = ["now", "sleep_for", "sleep_until"]


def now() -> float:
    """Seconds from an arbitrary, monotonic, high-resolution origin (``perf_counter``)."""
    return time.perf_counter()


async def sleep_until(deadline: float) -> None:
    """Sleep until ``now() >= deadline`` — never earlier.

    The event loop runs on ``time.monotonic()``, which ticks every ~15.6 ms on Windows,
    and asyncio fires timers up to one clock resolution early, so a plain
    ``asyncio.sleep(0.05)`` can return after ~35 ms of :func:`now` time there. The
    remainder (if any) is slept with the high-resolution ``time.sleep`` in a worker
    thread. Always yields to the event loop at least once.
    """
    delay = deadline - now()
    if delay <= 0:
        await asyncio.sleep(0)
        return
    await asyncio.sleep(delay)
    remaining = deadline - now()
    if remaining > 0:
        await asyncio.to_thread(time.sleep, remaining)


async def sleep_for(delay: float) -> None:
    """``asyncio.sleep(delay)`` that lasts at least ``delay`` seconds of :func:`now` time
    (see :func:`sleep_until`)."""
    await sleep_until(now() + delay)
