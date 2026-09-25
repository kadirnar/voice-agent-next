"""Timing helpers for tests that assert on latencies measured in real time.

A latency measured on a CI runner is the designed delay plus however late the event loop
got around to it (a 15.6 ms timer tick on Windows, a stalled or overloaded macOS runner).
:class:`LoopLag` measures that lateness *in the same run*, so an upper bound can allow for
it instead of guessing a tolerance that a slow runner eventually exceeds.
"""

from __future__ import annotations

import asyncio
import contextlib
from types import TracebackType

from voice_agent_next.utils import now


class LoopLag:
    """Measures how late the running event loop wakes up while the block runs.

    A probe task sleeps ``tick`` seconds over and over; :attr:`max` is the worst lateness
    of those wake-ups, i.e. how much later than planned any timer of the loop could have
    fired. Use it as the tolerance of an upper bound::

        async with LoopLag() as lag:
            ...  # run the scenario
        assert measured <= designed + 0.05 + lag.max
    """

    def __init__(self, tick: float = 0.005) -> None:
        self.tick = tick
        self.max = 0.0
        self._task: asyncio.Task[None] | None = None

    async def _probe(self) -> None:
        while True:
            t = now()
            await asyncio.sleep(self.tick)
            self.max = max(self.max, now() - t - self.tick)

    async def __aenter__(self) -> LoopLag:
        self._task = asyncio.create_task(self._probe())
        await asyncio.sleep(0)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        assert self._task is not None
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
