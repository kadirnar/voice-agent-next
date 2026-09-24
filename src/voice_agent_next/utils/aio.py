"""asyncio helpers: a closable channel, background task bookkeeping, iterator merging.

All objects here are *not* thread-safe (like asyncio itself). Producers running on
other threads (e.g. audio device callbacks) must hand items over with
``loop.call_soon_threadsafe(chan.send_nowait, item)``.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Coroutine
from typing import Any, Generic, TypeVar

from .log import logger

__all__ = ["BackgroundTasks", "Chan", "ChanClosed", "cancel_and_wait", "merge_async_iterators"]

T = TypeVar("T")


class ChanClosed(Exception):
    """Raised by :meth:`Chan.recv` / :meth:`Chan.send_nowait` on a closed, drained channel."""


class Chan(Generic[T]):
    """An unbounded multi-producer/multi-consumer async channel that can be closed.

    Closing a channel lets consumers drain the remaining items; after that
    :meth:`recv` raises :class:`ChanClosed` and ``async for`` loops stop.
    """

    def __init__(self) -> None:
        self._items: deque[T] = deque()
        self._getters: deque[asyncio.Future[None]] = deque()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def qsize(self) -> int:
        return len(self._items)

    def empty(self) -> bool:
        return not self._items

    def send_nowait(self, item: T) -> None:
        if self._closed:
            raise ChanClosed("send on closed channel")
        self._items.append(item)
        self._wake_one()

    async def send(self, item: T) -> None:
        self.send_nowait(item)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        while self._getters:
            fut = self._getters.popleft()
            if not fut.done():
                fut.set_result(None)

    def clear(self) -> list[T]:
        """Drop and return all pending items (used for interruptions)."""
        items = list(self._items)
        self._items.clear()
        return items

    def recv_nowait(self) -> T:
        if self._items:
            return self._items.popleft()
        if self._closed:
            raise ChanClosed("channel closed")
        raise asyncio.QueueEmpty

    async def recv(self) -> T:
        while not self._items:
            if self._closed:
                raise ChanClosed("channel closed")
            fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._getters.append(fut)
            try:
                await fut
            except asyncio.CancelledError:
                with contextlib.suppress(ValueError):
                    self._getters.remove(fut)
                # pass the wake-up on to another getter if we consumed it
                if fut.done() and not fut.cancelled() and self._items:
                    self._wake_one()
                raise
        return self._items.popleft()

    def _wake_one(self) -> None:
        while self._getters:
            fut = self._getters.popleft()
            if not fut.done():
                fut.set_result(None)
                return

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        try:
            return await self.recv()
        except ChanClosed:
            raise StopAsyncIteration from None


async def cancel_and_wait(*tasks: asyncio.Task[Any] | None) -> None:
    """Cancel tasks and wait until they are finished, swallowing their CancelledError.

    If the *calling* task is cancelled while waiting, the cancellation propagates.
    """
    pending = [t for t in tasks if t is not None and not t.done()]
    for t in pending:
        t.cancel()
    if not pending:
        return
    done_waiting = asyncio.gather(*pending, return_exceptions=True)
    try:
        await asyncio.shield(done_waiting)
    except asyncio.CancelledError:
        # we were cancelled ourselves; still make sure children finish before re-raising
        await asyncio.gather(*pending, return_exceptions=True)
        raise


class BackgroundTasks:
    """Keeps strong references to fire-and-forget tasks and logs their failures."""

    def __init__(self, name: str = "tasks") -> None:
        self._name = name
        self._tasks: set[asyncio.Task[Any]] = set()

    def spawn(self, coro: Coroutine[Any, Any, T], *, name: str | None = None) -> asyncio.Task[T]:
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._done)
        return task

    def _done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("%s: background task %s failed", self._name, task.get_name(), exc_info=exc)

    def __len__(self) -> int:
        return len(self._tasks)

    async def cancel_all(self) -> None:
        """Cancel every task except the calling one (a task may cancel its siblings)."""
        current = asyncio.current_task()
        others = [t for t in self._tasks if t is not current]
        await cancel_and_wait(*others)
        self._tasks.difference_update(others)

    async def wait_all(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)


async def merge_async_iterators(*iterators: AsyncIterator[T]) -> AsyncIterator[T]:
    """Yield items from several async iterators as they arrive (order across sources not kept)."""
    chan: Chan[T] = Chan()
    remaining = len(iterators)

    async def pump(it: AsyncIterator[T]) -> None:
        nonlocal remaining
        try:
            async for item in it:
                chan.send_nowait(item)
        finally:
            remaining -= 1
            if remaining == 0:
                chan.close()

    tasks = [asyncio.create_task(pump(it)) for it in iterators]
    if not tasks:
        return
    try:
        async for item in chan:
            yield item
        for t in tasks:  # surface pump errors
            if t.done() and not t.cancelled() and t.exception() is not None:
                raise t.exception()  # type: ignore[misc]
    finally:
        await cancel_and_wait(*tasks)


async def wait_first(*aws: Awaitable[Any]) -> None:
    """Wait until the first awaitable finishes; cancel the rest."""
    tasks = [asyncio.ensure_future(a) for a in aws]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        await cancel_and_wait(*tasks)
