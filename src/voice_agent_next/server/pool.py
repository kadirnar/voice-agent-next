"""Prewarmed engines for serving: models load once, new calls skip connection setup.

A cold call pays for everything before its first audio: loading models (``warmup()``) and
opening the engine connection (a WebSocket to a cloud API, VAD/STT state for a cascade).
:class:`EnginePool` moves that work out of the call path:

* **shared** (an engine instance): the engine is warmed up once per process, and every
  session opens its own ``connect()`` on it. ``size`` connections are opened ahead of time
  with the options sessions are expected to use (the agent's instructions, tools, voice...).
* **per session** (a factory): every session gets its own engine instance (isolation for
  engines that are not safe to share). ``size`` instances are built, warmed up and
  pre-connected ahead of time.

:meth:`EnginePool.lease` returns a :class:`PooledEngine`, a per-session stand-in for the
engine. Its first ``connect()`` hands over the prewarmed connection when the session's
options match the prewarm options (or can be applied with ``EngineConnection.update``),
and opens a fresh one otherwise. Closing it returns nothing to the pool: prewarmed
connections and per-session engines are used once, so no state leaks between calls. The
pool refills itself in the background and recycles prewarmed connections older than
``max_idle`` (cloud sessions expire).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

from ..engine import EngineConnection, EngineOptions, S2SEngine
from ..errors import ConfigurationError
from ..utils.clock import now
from ..utils.log import logger

__all__ = ["EngineFactory", "EnginePool", "PoolStats", "PooledEngine", "options_match"]

EngineFactory = Callable[[], "S2SEngine | Awaitable[S2SEngine]"]
"""Builds one engine (per-session pools)."""

ConnectObserver = Callable[[float, bool], None]
"""``(seconds, prewarmed)`` for every session's first engine connection."""


@dataclass
class PoolStats:
    """Counters of an :class:`EnginePool` (see :meth:`EnginePool.stats`)."""

    size: int
    """Target number of prewarmed items."""
    idle: int
    """Prewarmed items ready for the next session."""
    leased: int
    """Engines currently used by sessions."""
    hits: int
    """Sessions that got a prewarmed item."""
    misses: int
    """Sessions that had to start cold (the pool was empty)."""
    reused: int
    """Sessions whose first ``connect()`` used a prewarmed connection."""
    errors: int
    """Failed prewarm attempts."""
    warmup_seconds: float | None
    """How long the initial warm-up took (models + the first ``size`` items)."""


@dataclass(eq=False)
class _Item:
    engine: S2SEngine
    conn: EngineConnection | None
    created: float


def _tools_key(options: EngineOptions) -> tuple[Any, ...]:
    return tuple(
        sorted(
            (t.name, t.description, json.dumps(t.parameters, sort_keys=True, default=str))
            for t in options.tools
        )
    )


def _rest_key(options: EngineOptions) -> tuple[Any, ...] | None:
    """What a prewarmed connection cannot change after the fact."""
    if options.chat_ctx is not None and options.chat_ctx.items:
        return None  # history is seeded at connect time
    extra = json.dumps(options.extra, sort_keys=True, default=str)
    return (options.voice, options.language, options.temperature, options.turn_detection, extra)


def options_match(prewarmed: EngineOptions, wanted: EngineOptions) -> bool:
    """``True`` when a connection opened with ``prewarmed`` serves ``wanted`` as is."""
    rest = _rest_key(wanted)
    return (
        rest is not None
        and rest == _rest_key(prewarmed)
        and prewarmed.instructions == wanted.instructions
        and _tools_key(prewarmed) == _tools_key(wanted)
    )


def _can_update(conn: EngineConnection) -> bool:
    """The connection applies ``update(instructions=, tools=)`` (the base one ignores it)."""
    return type(conn).update is not EngineConnection.update


async def _close_quietly(obj: Any) -> None:
    if obj is None:
        return
    with contextlib.suppress(Exception):
        await obj.aclose()


class PooledEngine(S2SEngine):
    """A session's engine, leased from an :class:`EnginePool` (close it when done).

    It behaves like the engine it wraps: same model, capabilities and sample rates, and
    ``"metrics"`` from the engine are re-emitted. Its first :meth:`connect` uses the
    prewarmed connection when it can.
    """

    def __init__(self, pool: EnginePool, item: _Item, *, prewarmed: bool) -> None:
        engine = item.engine
        super().__init__(
            model=engine.model,
            capabilities=engine.capabilities,
            input_sample_rate=engine.input_sample_rate,
            output_sample_rate=engine.output_sample_rate,
        )
        self.provider = engine.provider  # type: ignore[misc]
        self.inner = engine
        """The wrapped engine."""
        self.prewarmed = prewarmed
        """The lease got a prewarmed item (``False``: a cold start)."""
        self._pool = pool
        self._conn = item.conn
        self._connected = False
        self._closed = False
        engine.on("metrics", self._forward_metrics)

    def __getattr__(self, name: str) -> Any:  # engine-specific attributes (e.g. MockEngine.llm)
        if name.startswith("__") or name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def _forward_metrics(self, m: Any) -> None:
        self.emit("metrics", m)

    async def warmup(self) -> None:
        """No-op: the pool warmed the engine up (sessions call this on start)."""
        await self._pool.ensure_warm()

    async def connect(self, options: EngineOptions) -> EngineConnection:
        if self._closed:
            raise ConfigurationError("this pooled engine was released")
        started = now()
        conn, self._conn = self._conn, None
        reused = False
        if conn is not None and not conn.closed:
            prewarmed = conn.options
            if options_match(prewarmed, options):
                reused = True
            elif (
                _rest_key(options) is not None
                and _rest_key(options) == _rest_key(prewarmed)
                and _can_update(conn)
            ):
                try:
                    await conn.update(instructions=options.instructions, tools=list(options.tools))
                    reused = True
                except Exception as exc:
                    logger.debug("prewarmed connection could not be updated: %s", exc)
        if reused and conn is not None:
            conn.options = options
        else:
            await _close_quietly(conn)
            conn = await self.inner.connect(options)
        if not self._connected:
            self._connected = True
            self._pool._connected(now() - started, reused)
        return conn

    async def aclose(self) -> None:
        """Release the lease: unused prewarmed connections and per-session engines close."""
        if self._closed:
            return
        self._closed = True
        self.inner.off("metrics", self._forward_metrics)
        conn, self._conn = self._conn, None
        await _close_quietly(conn)
        await self._pool._release(self)


class EnginePool:
    """Keeps ``size`` prewarmed engines or engine connections ready for new sessions.

    Args:
        source: an engine instance (shared: warmed once, one connection per session) or a
            factory (one engine per session: built and warmed ahead of time).
        size: prewarmed items to keep ready (0: warm the shared engine only).
        options: engine options sessions are expected to connect with (the agent's
            instructions, tools, voice, language). ``None``: items are not pre-connected
            (per-session engines are still built and warmed).
        max_idle: recycle prewarmed connections older than this many seconds (``None``:
            never). Cloud sessions expire and idle sockets get dropped by proxies.
        owned: close the shared engine in :meth:`aclose`.
        warmup: warm the shared engine up in :meth:`start`.
        on_connect: called with ``(seconds, prewarmed)`` for every session's first
            engine connection (metrics).
        name: for logs.
    """

    def __init__(
        self,
        source: S2SEngine | EngineFactory,
        *,
        size: int = 0,
        options: EngineOptions | None = None,
        max_idle: float | None = 300.0,
        owned: bool = False,
        warmup: bool = True,
        on_connect: ConnectObserver | None = None,
        name: str = "default",
    ) -> None:
        if size < 0:
            raise ConfigurationError("the prewarm pool size must be >= 0")
        if isinstance(source, S2SEngine):
            self.engine: S2SEngine | None = source
            self._factory: EngineFactory | None = None
        elif callable(source):
            self.engine, self._factory = None, source
        else:
            raise ConfigurationError(f"cannot pool {type(source).__name__}: pass an engine")
        self.size = size
        self.options = options
        self.max_idle = max_idle
        self.owned = owned
        self.warmup = warmup
        self.on_connect = on_connect
        self.name = name
        self._idle: deque[_Item] = deque()
        self._leases: set[PooledEngine] = set()
        self._creating = 0
        self._hits = self._misses = self._reused = self._errors = 0
        self._warm = asyncio.Event()
        self._warm_task: asyncio.Task[None] | None = None
        self._warmup_seconds: float | None = None
        self._wake = asyncio.Event()
        self._maintainer: asyncio.Task[None] | None = None
        self._closing = False

    @property
    def per_session(self) -> bool:
        """Every session gets its own engine instance."""
        return self._factory is not None

    @property
    def idle(self) -> int:
        """Prewarmed items ready now."""
        return len(self._idle)

    @property
    def leased(self) -> int:
        """Engines currently used by sessions."""
        return len(self._leases)

    @property
    def warm(self) -> bool:
        """The initial warm-up finished."""
        return self._warm.is_set()

    @property
    def ready(self) -> bool:
        """Warm, and a prewarmed item is available (always, when ``size`` is 0)."""
        return self.warm and not self._closing and (self.size == 0 or bool(self._idle))

    def stats(self) -> PoolStats:
        return PoolStats(
            size=self.size, idle=self.idle, leased=self.leased, hits=self._hits,
            misses=self._misses, reused=self._reused, errors=self._errors,
            warmup_seconds=self._warmup_seconds,
        )  # fmt: skip

    # ------------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        """Warm up (models, then ``size`` items) and keep the pool filled. Idempotent."""
        if self._warm_task is None:
            self._warm_task = asyncio.create_task(self._initial_warmup(), name="pool-warmup")
        await asyncio.shield(self._warm_task)

    async def ensure_warm(self) -> None:
        """Wait for the initial warm-up (starting it if needed)."""
        if not self._warm.is_set():
            await self.start()

    async def _initial_warmup(self) -> None:
        started = now()
        if self.engine is not None and self.warmup:
            try:
                await self.engine.warmup()
            except Exception as exc:
                logger.warning("warmup of %s failed (continuing): %s", self.engine.provider, exc)
        for _ in range(self.size):
            if self._closing:
                break
            await self._add_item()
        self._warmup_seconds = now() - started
        self._warm.set()
        logger.info(
            "engine pool %s warm in %.2fs (%d prewarmed%s)",
            self.name, self._warmup_seconds, len(self._idle),
            ", one engine per session" if self.per_session else "",
        )  # fmt: skip
        if self.size and not self._closing:
            self._maintainer = asyncio.create_task(self._maintain(), name="pool-maintain")

    async def aclose(self) -> None:
        """Stop refilling and close prewarmed items (and the shared engine when owned).

        Leased engines stay usable; per-session engines close when their lease ends.
        """
        if self._closing:
            return
        self._closing = True
        self._wake.set()
        for task in (self._maintainer, self._warm_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task
        while self._idle:
            await self._discard(self._idle.popleft())
        if self.owned and self.engine is not None:
            await _close_quietly(self.engine)

    # --------------------------------------------------------------------- leasing
    async def lease(self) -> PooledEngine:
        """The engine for a new session (close it when the session ends)."""
        if self._closing:
            raise ConfigurationError("the engine pool is closed")
        await self.ensure_warm()
        item = await self._take()
        prewarmed = item is not None
        if item is None:
            if self.size:
                self._misses += 1
            engine = self.engine if self.engine is not None else await self._build()
            item = _Item(engine, None, now())
        else:
            self._hits += 1
        self._wake.set()
        lease = PooledEngine(self, item, prewarmed=prewarmed)
        self._leases.add(lease)
        return lease

    async def _take(self) -> _Item | None:
        while self._idle:
            item = self._idle.popleft()
            if item.conn is not None and item.conn.closed:  # e.g. the provider hung up
                await self._discard(item)
                continue
            return item
        return None

    async def _release(self, lease: PooledEngine) -> None:
        self._leases.discard(lease)
        if self.per_session:
            await _close_quietly(lease.inner)

    def _connected(self, seconds: float, reused: bool) -> None:
        if reused:
            self._reused += 1
        if self.on_connect is not None:
            try:
                self.on_connect(seconds, reused)
            except Exception:
                logger.exception("pool on_connect observer failed")

    # ------------------------------------------------------------------ prewarming
    async def _build(self) -> S2SEngine:
        assert self._factory is not None
        result = self._factory()
        engine = await result if inspect.isawaitable(result) else result
        if not isinstance(engine, S2SEngine):
            raise ConfigurationError(f"engine factory returned {type(engine).__name__}")
        if self.warmup:
            try:
                await engine.warmup()
            except Exception as exc:
                logger.warning("warmup of %s failed (continuing): %s", engine.provider, exc)
        return engine

    async def _make_item(self) -> _Item:
        engine = self.engine if self.engine is not None else await self._build()
        conn: EngineConnection | None = None
        if self.options is not None:
            try:
                conn = await engine.connect(replace(self.options))
            except BaseException:
                if self.per_session:
                    await _close_quietly(engine)
                raise
        return _Item(engine, conn, now())

    async def _add_item(self) -> bool:
        self._creating += 1
        try:
            item = await self._make_item()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._errors += 1
            logger.warning("engine pool %s: prewarming failed: %s", self.name, exc)
            return False
        finally:
            self._creating -= 1
        if self._closing:
            await self._discard(item)
            return False
        self._idle.append(item)
        return True

    async def _discard(self, item: _Item) -> None:
        await _close_quietly(item.conn)
        if self.per_session:
            await _close_quietly(item.engine)

    async def _evict_stale(self) -> None:
        limit = None if self.max_idle is None else now() - self.max_idle
        stale = [
            i
            for i in self._idle
            if i.conn is not None and (i.conn.closed or (limit is not None and i.created < limit))
        ]
        for item in stale:
            self._idle.remove(item)
            await self._discard(item)

    async def _maintain(self) -> None:
        """Refill after leases, recycle stale connections, back off on failures."""
        interval = 30.0 if self.max_idle is None else max(0.05, min(30.0, self.max_idle / 4))
        failures = 0
        while not self._closing:
            await self._evict_stale()
            while not self._closing and len(self._idle) + self._creating < self.size:
                if await self._add_item():
                    failures = 0
                else:
                    failures += 1
                    await asyncio.sleep(min(30.0, 0.5 * 2 ** min(failures, 6)))
            self._wake.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), interval)
