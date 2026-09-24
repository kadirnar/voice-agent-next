"""A tiny synchronous/asynchronous event emitter.

Handlers can be plain functions or coroutine functions. Coroutine handlers are
scheduled as tasks (references are kept until they finish). Exceptions raised by
handlers are logged and never propagate into the emitter's caller: a buggy
application callback must not break the audio pipeline.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, TypeVar, overload

from .log import logger

__all__ = ["EventEmitter"]

F = TypeVar("F", bound=Callable[..., Any])


class EventEmitter:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[..., Any]]] = {}
        self._handler_tasks: set[asyncio.Task[Any]] = set()

    @overload
    def on(self, event: str) -> Callable[[F], F]: ...

    @overload
    def on(self, event: str, callback: F) -> F: ...

    def on(self, event: str, callback: F | None = None) -> F | Callable[[F], F]:
        """Register ``callback`` for ``event``. Usable as a decorator: ``@x.on("metrics")``."""
        if callback is None:

            def decorator(fn: F) -> F:
                self._handlers.setdefault(event, []).append(fn)
                return fn

            return decorator
        self._handlers.setdefault(event, []).append(callback)
        return callback

    def once(self, event: str, callback: Callable[..., Any]) -> Callable[..., Any]:
        """Register a handler that is removed after its first invocation."""

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.off(event, wrapper)
            return callback(*args, **kwargs)

        self._handlers.setdefault(event, []).append(wrapper)
        return wrapper

    def off(self, event: str, callback: Callable[..., Any]) -> None:
        handlers = self._handlers.get(event)
        if handlers and callback in handlers:
            handlers.remove(callback)

    def has_listeners(self, event: str) -> bool:
        return bool(self._handlers.get(event))

    def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        for handler in list(self._handlers.get(event, ())):
            try:
                result = handler(*args, **kwargs)
            except Exception:
                logger.exception("error in %r handler for event %r", handler, event)
                continue
            if inspect.isawaitable(result):
                try:
                    task = asyncio.ensure_future(result)
                except RuntimeError:  # no running loop
                    logger.error("async handler for %r emitted outside of an event loop", event)
                    continue
                self._handler_tasks.add(task)
                task.add_done_callback(self._on_handler_done)

    def _on_handler_done(self, task: asyncio.Task[Any]) -> None:
        self._handler_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("async event handler failed", exc_info=task.exception())
