"""Production serving for every protocol: prewarm pools, admission, health, drain, workers.

``van serve`` builds a :class:`Served` — a protocol server wired to :class:`EnginePool`\\ s
and a :class:`~voice_agent_next.server.ops.ServeState` — and runs it with
:func:`run_served` (one process) or :func:`run_workers` (a supervisor with worker processes
sharing the port through ``SO_REUSEPORT``). See ``docs/deploy/serving.md``.

Protocols:

* ``openai-realtime`` — :class:`~voice_agent_next.server.RealtimeServer` (``/v1/realtime``);
* ``websocket`` — :class:`~voice_agent_next.transports.websocket.WebSocketAgentServer`
  (``van-ws/1``);
* ``webrtc`` — :class:`~voice_agent_next.transports.webrtc.WebRTCAgentServer` (HTTP
  signalling, ``POST /offer``);
* ``twilio`` / ``telnyx`` / ``vonage`` / ``plivo`` —
  :class:`~voice_agent_next.transports.telephony.TelephonyServer` media streams.

Every server answers ``GET /health``, ``/ready`` and ``/metrics`` on its own port, and
refuses new sessions while draining or at the session limit in the protocol's own way
(HTTP 503 before the WebSocket upgrade, HTTP 503 to a WebRTC offer).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import multiprocessing
import os
import signal
import socket
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, Final
from urllib.parse import urlsplit

from websockets.asyncio.server import ServerConnection
from websockets.http11 import Request, Response

from ..engine import EngineOptions, S2SEngine
from ..errors import ConfigurationError
from ..utils.log import logger
from .ops import ServeState, set_session_id
from .pool import EnginePool
from .realtime import RealtimeModel, RealtimeServer

__all__ = [
    "AGENT_PROTOCOLS",
    "PROTOCOLS",
    "TELEPHONY_PROTOCOLS",
    "Served",
    "build_agent_served",
    "build_realtime_served",
    "free_port",
    "reuse_port_supported",
    "run_served",
    "run_workers",
]

TELEPHONY_PROTOCOLS: Final = ("twilio", "telnyx", "vonage", "plivo")
AGENT_PROTOCOLS: Final = ("websocket", "webrtc", *TELEPHONY_PROTOCOLS)
"""Protocols that run an :class:`~voice_agent_next.AgentSession` per connection."""
PROTOCOLS: Final = ("openai-realtime", *AGENT_PROTOCOLS)
_OPS_PATHS: Final = frozenset({"/health", "/healthz", "/livez", "/ready", "/readyz", "/metrics"})


def reuse_port_supported() -> bool:
    """``SO_REUSEPORT`` is available (Linux, macOS, BSDs; not Windows)."""
    return sys.platform != "win32" and hasattr(socket, "SO_REUSEPORT")


def free_port(host: str = "127.0.0.1") -> int:
    """A free TCP port on ``host`` (for ``--port 0`` with several workers)."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


# ----------------------------------------------------------------------------- Served
@dataclass
class Served:
    """A protocol server with its engine pools and serving state."""

    server: Any
    """The protocol server (``start``, ``serve_forever``, ``aclose``, ``url``, ``port``)."""
    state: ServeState
    pools: list[EnginePool] = field(default_factory=list)

    @property
    def url(self) -> str:
        url: str = self.server.url
        return f"{url}/realtime" if isinstance(self.server, RealtimeServer) else url

    async def start(self) -> None:
        """Listen (``/health`` answers at once, ``/ready`` once warm), then warm the pools."""
        await self.server.start()
        await asyncio.gather(*(pool.start() for pool in self.pools))
        self.state.started = True

    async def drain(self, timeout: float, force: asyncio.Event | None = None) -> None:
        """Refuse new sessions and wait up to ``timeout`` s for live ones to finish."""
        state = self.state
        state.draining = True
        deadline = time.monotonic() + max(0.0, timeout)
        if state.active:
            logger.info("draining: waiting for %d session(s), up to %.0fs", state.active, timeout)
        while state.active and time.monotonic() < deadline:
            if force is not None and force.is_set():
                break
            await asyncio.sleep(0.05)
        if state.active:
            logger.warning("drain timeout: closing %d live session(s)", state.active)

    async def aclose(self) -> None:
        """Close the server (live sessions are closed) and the pools."""
        self.state.draining = True
        with contextlib.suppress(Exception):
            await self.server.aclose()
        for pool in self.pools:
            with contextlib.suppress(Exception):
                await pool.aclose()


# ----------------------------------------------------------------- openai-realtime
class _ServedRealtimeServer(RealtimeServer):
    """A :class:`RealtimeServer` with ``/ready`` and ``/metrics``, drain-aware admission
    and per-session accounting."""

    state: ServeState

    def health(self) -> dict[str, Any]:
        doc = super().health()
        state = getattr(self, "state", None)
        if state is not None:
            doc.update({k: v for k, v in state.health().items() if k != "sessions"})
            if state.draining:
                doc["status"] = "draining"
        return doc

    def _process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        path = urlsplit(request.path).path.rstrip("/") or "/"
        if path in _OPS_PATHS and path not in ("/health", "/healthz"):
            routed = self.state.http(path)
            if routed is not None:
                return _ws_response(connection, *routed)
        if path.rsplit("/", 1)[-1] == "realtime" and not self._closing:
            refusal = self.state.refusal()
            if refusal is not None:
                status, reason, message = refusal
                self.state.reject(reason)
                body = {
                    "error": {
                        "type": "invalid_request_error",
                        "code": "server_draining"
                        if reason == "draining"
                        else "session_limit_reached",
                        "message": message,
                    }
                }
                response = _ws_response(connection, status, "application/json",
                                        (json.dumps(body) + "\n").encode())  # fmt: skip
                response.headers["Retry-After"] = "1"
                return response
        return super()._process_request(connection, request)

    async def _handle(self, websocket: ServerConnection) -> None:
        from ._session import RealtimeSession

        request = websocket.request
        model = self._resolve_model(request) if request is not None else None
        if request is None or model is None:
            await websocket.close(1008, "unknown model")
            return
        if self.max_sessions is not None and len(self._sessions) >= self.max_sessions:
            self.state.reject("busy")
            await websocket.close(1013, "session limit reached")
            return
        session = RealtimeSession(self, websocket, model, dialect=self._dialect(request))
        set_session_id(session.id)
        self.state.session_started()
        self._sessions.add(session)
        try:
            await session.run()
        finally:
            self._sessions.discard(session)
            self.state.session_ended(error=bool(session.stats.get("errors")))


def _realtime_options(model: RealtimeModel) -> EngineOptions:
    """The options a Realtime session connects with before the client changes them."""
    cfg = model.session_config()
    return EngineOptions(
        instructions=cfg.instructions,
        tools=cfg.function_tools(),
        voice=cfg.voice,
        language=cfg.language or model.language,
        turn_detection=cfg.vad,
    )


def build_realtime_served(
    models: Mapping[str, Any],
    *,
    prewarm: int = 0,
    preconnect: bool = True,
    max_idle: float | None = 300.0,
    warmup: bool = True,
    worker: int | None = None,
    reuse_port: bool = False,
    **server_options: Any,
) -> Served:
    """An OpenAI-Realtime :class:`Served` for ``models`` (see :class:`RealtimeServer`).

    Every model gets an :class:`EnginePool` of ``prewarm`` items: an engine instance is
    shared by the model's sessions, an engine factory gives every session its own engine.
    ``server_options`` are :class:`RealtimeServer` arguments (``host``, ``port``,
    ``api_keys``, ``max_sessions``, ``ssl``...).
    """
    if reuse_port:
        server_options["reuse_port"] = True
    server = _ServedRealtimeServer(models=dict(models), warmup=False, **server_options)
    state = ServeState(
        "openai-realtime",
        max_sessions=server.max_sessions,
        worker=worker,
        active=lambda: len(server._sessions),
    )
    server.state = state
    pools: list[EnginePool] = []
    for name, model in server.models.items():
        pool = EnginePool(
            model.engine,  # type: ignore[arg-type]  # an engine or a factory by now
            size=prewarm,
            options=_realtime_options(model) if preconnect else None,
            max_idle=max_idle,
            owned=model.owned,
            warmup=warmup,
            on_connect=state.observe_connect,
            name=name,
        )
        model.engine, model.owned = pool.lease, False  # the pool closes what it owns
        pools.append(pool)
    state.pools = pools
    return Served(server, state, pools)


# ------------------------------------------------------------ AgentSession protocols
def build_agent_served(
    protocol: str,
    config: Any,
    *,
    engine_per_session: bool = False,
    prewarm: int = 0,
    preconnect: bool = True,
    max_idle: float | None = 300.0,
    warmup: bool = True,
    max_sessions: int | None = None,
    worker: int | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    reuse_port: bool = False,
    **server_options: Any,
) -> Served:
    """A :class:`Served` running one :class:`~voice_agent_next.AgentSession` per connection.

    Args:
        protocol: ``websocket``, ``webrtc``, ``twilio``, ``telnyx``, ``vonage`` or ``plivo``.
        config: an :class:`~voice_agent_next.config.AppConfig` (or a mapping / file path):
            the engine or cascade, the agent and the session options.
        engine_per_session: one engine instance per session (default: one engine per
            process, one connection per session).
        prewarm: prewarmed engines (connections) to keep ready.
        preconnect: open prewarmed connections with the agent's options.
        server_options: extra arguments of the protocol server (``ssl``, ``ice_servers``,
            ``serializer_options``...).
    """
    from ..app import build_agent
    from ..config import AppConfig, load_config
    from ..metrics import TurnMetrics
    from ..session import AgentSession, SessionOptions

    if protocol not in AGENT_PROTOCOLS:
        raise ConfigurationError(
            f"unknown protocol {protocol!r}; expected one of {', '.join(PROTOCOLS)}"
        )
    cfg = config if isinstance(config, AppConfig) else load_config(config)
    cfg.validate_components()
    SessionOptions(**cfg.session)  # validate early

    def make_engine() -> S2SEngine:
        return engine_from_app_config(cfg)

    probe = build_agent(cfg)
    options = EngineOptions(
        instructions=probe.instructions,
        tools=list(probe.tools),
        chat_ctx=probe.chat_ctx,
        voice=probe.voice,
        language=probe.language,
    )
    state = ServeState(protocol, max_sessions=max_sessions, worker=worker)
    pool = EnginePool(
        make_engine if engine_per_session else make_engine(),
        size=prewarm,
        options=options if preconnect else None,
        max_idle=max_idle,
        owned=not engine_per_session,
        warmup=warmup,
        on_connect=state.observe_connect,
        name="agent",
    )
    state.pools = [pool]

    async def session_factory(transport: Any) -> AgentSession:
        set_session_id(getattr(transport, "session_id", None))
        refusal = state.refusal()
        if refusal is not None:  # raced with the handshake check
            state.reject(refusal[1])
            raise ConfigurationError(refusal[2])
        engine = await pool.lease()
        session = AgentSession(engine, options=SessionOptions(**cfg.session))
        state.session_started()
        failed: list[bool] = []

        def on_metrics(m: Any) -> None:
            if isinstance(m, TurnMetrics):
                state.observe_turn(m)

        def on_error(ev: Any) -> None:
            if not ev.recoverable:
                failed.append(True)

        async def on_close(ev: Any) -> None:
            state.session_ended(error=bool(failed) or ev.reason == "error")
            await engine.aclose()

        session.on("metrics", on_metrics)
        session.on("error", on_error)
        session.on("close", on_close)
        return session

    def agent_factory() -> Any:
        return build_agent(cfg)

    server: Any
    if protocol == "webrtc":
        from ..transports.webrtc import WebRTCAgentServer

        server = WebRTCAgentServer(
            session_factory, agent_factory, host=host, port=port, max_sessions=max_sessions,
            **server_options,
        )  # fmt: skip
        _mount_webrtc_ops(server, state, reuse_port=reuse_port)
    else:
        serve_options = dict(server_options)
        if reuse_port:
            serve_options["reuse_port"] = True
        user_process_request = serve_options.pop("process_request", None)
        serve_options["process_request"] = _ws_process_request(state, user_process_request)
        if protocol == "websocket":
            from ..transports.websocket import WebSocketAgentServer

            server = WebSocketAgentServer(
                session_factory, agent_factory, host=host, port=port,
                max_sessions=max_sessions, **serve_options,
            )  # fmt: skip
        else:
            from ..transports.telephony import TelephonyServer

            server = TelephonyServer(
                session_factory, agent_factory, provider=protocol, host=host, port=port,
                max_sessions=max_sessions, **serve_options,
            )  # fmt: skip
    return Served(server, state, [pool])


def engine_from_app_config(cfg: Any) -> S2SEngine:
    """The engine of an :class:`~voice_agent_next.config.AppConfig` (engine or cascade)."""
    from ..engines.cascade import CascadeEngine, CascadeOptions
    from ..registry import create

    if cfg.engine is not None:
        engine: S2SEngine = create("engine", cfg.engine)
        return engine
    return CascadeEngine(
        stt=cfg.stt, llm=cfg.llm, tts=cfg.tts, vad=cfg.vad, turn_detector=cfg.turn_detector,
        options=CascadeOptions(**cfg.cascade),
    )  # fmt: skip


def _ws_response(
    connection: ServerConnection, status: HTTPStatus, content_type: str, body: bytes
) -> Response:
    response = connection.respond(status, "")
    response.body = body
    del response.headers["Content-Type"]
    del response.headers["Content-Length"]
    response.headers["Content-Type"] = content_type
    response.headers["Content-Length"] = str(len(body))
    response.headers["Cache-Control"] = "no-store"
    return response


def _ws_process_request(state: ServeState, user: Callable[..., Any] | None) -> Callable[..., Any]:
    """Ops routes and admission in front of a WebSocket server's handshake."""

    async def process_request(connection: ServerConnection, request: Request) -> Any:
        routed = state.http(request.path)
        if routed is not None:
            return _ws_response(connection, *routed)
        refusal = state.refusal()
        if refusal is not None:
            status, reason, message = refusal
            state.reject(reason)
            body = json.dumps({"type": "error", "code": f"server_{reason}", "message": message})
            response = _ws_response(connection, status, "application/json", body.encode())
            response.headers["Retry-After"] = "1"
            return response
        if user is not None:
            result = user(connection, request)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        return None

    return process_request


def _mount_webrtc_ops(server: Any, state: ServeState, *, reuse_port: bool) -> None:
    """Add the ops routes and admission to a WebRTC signalling server."""
    from ..transports.webrtc import _HttpError, _Response

    http = server._http
    if http is None:  # serve_http=False: the application mounts routes itself
        return
    http.reuse_port = reuse_port
    route = http._route

    async def routed(method: str, path: str, headers: dict[str, str], body: bytes) -> Any:
        clean = path.split("?", 1)[0]
        if method == "GET":
            ops = state.http(clean)
            if ops is not None:
                status, content_type, payload = ops
                return _Response(status, payload, content_type)
        if method == "POST" and clean == http.offer_path:
            refusal = state.refusal()
            if refusal is not None:
                state.reject(refusal[1])
                raise _HttpError(refusal[0], refusal[2])
        return await route(method, path, headers, body)

    http._route = routed


# --------------------------------------------------------------------------- running
async def run_served(
    served: Served,
    *,
    drain_timeout: float = 30.0,
    handle_signals: bool = True,
    handle_sigint: bool = True,
    on_started: Callable[[Served], Any] | None = None,
) -> None:
    """Serve until the server closes or a signal arrives, then drain and close.

    The first SIGTERM / SIGINT (Ctrl+C) drains: new sessions are refused (``/ready`` turns
    503) and live ones get up to ``drain_timeout`` seconds to finish. A second signal
    closes them at once. ``handle_sigint=False``: SIGTERM only (worker processes, whose
    supervisor forwards Ctrl+C as SIGTERM).
    """
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    force = asyncio.Event()

    def on_signal() -> None:
        if stop.is_set():
            force.set()
        else:
            logger.info("shutdown requested: draining (signal again to stop at once)")
            stop.set()

    restore = (
        _install_signal_handlers(loop, on_signal, sigint=handle_sigint)
        if handle_signals
        else (lambda: None)
    )
    serve_task: asyncio.Task[None] | None = None
    try:
        await served.start()
        if on_started is not None:
            on_started(served)
        serve_task = asyncio.create_task(served.server.serve_forever(), name="serve-forever")
        stop_task = asyncio.create_task(stop.wait(), name="serve-stop")
        await asyncio.wait({serve_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        stop_task.cancel()
        if stop.is_set() and not serve_task.done():
            await served.drain(drain_timeout, force)
    finally:
        restore()
        await served.aclose()
        if serve_task is not None:
            if not serve_task.done():
                serve_task.cancel()
            with contextlib.suppress(BaseException):
                await serve_task


def _install_signal_handlers(
    loop: asyncio.AbstractEventLoop, callback: Callable[[], None], *, sigint: bool = True
) -> Callable[[], None]:
    """SIGTERM/SIGINT call ``callback`` on the loop. Returns the undo function."""
    if threading.current_thread() is not threading.main_thread():
        return lambda: None
    installed: list[Any] = []
    names = ["SIGTERM"] if sys.platform != "win32" else ["SIGBREAK"]
    if sigint:
        names.insert(0, "SIGINT")
    for name in names:
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        if sys.platform != "win32":
            try:
                loop.add_signal_handler(sig, callback)
            except (NotImplementedError, RuntimeError, ValueError):
                continue
            installed.append(("loop", sig, None))
        else:  # the Proactor loop has no add_signal_handler

            def handler(signum: int, frame: Any) -> None:
                loop.call_soon_threadsafe(callback)

            previous = signal.signal(sig, handler)
            installed.append(("signal", sig, previous))

    def restore() -> None:
        for kind, sig, previous in installed:
            with contextlib.suppress(Exception):
                if kind == "loop":
                    loop.remove_signal_handler(sig)
                else:
                    signal.signal(sig, previous)

    return restore


# --------------------------------------------------------------------------- workers
def run_workers(
    target: Callable[..., Any],
    args: Sequence[Any],
    workers: int,
    *,
    drain_timeout: float = 30.0,
    restart: bool = True,
    max_quick_failures: int = 5,
    fatal_exit_codes: Sequence[int] = (),
) -> int:
    """Run ``target(*args, index)`` in ``workers`` processes and supervise them.

    Workers are started with the ``spawn`` method (no inherited event loop or model
    threads) and bind the same port with ``SO_REUSEPORT``: the kernel spreads new
    connections over them. A worker that dies is restarted, unless it keeps dying right
    after starting. SIGTERM / SIGINT drain every worker (a second signal stops them at
    once); the call returns once they all exited. A worker exiting with one of
    ``fatal_exit_codes`` (e.g. a configuration error) stops everything. Returns the exit
    code.
    """
    if workers < 1:
        raise ConfigurationError("workers must be >= 1")
    ctx = multiprocessing.get_context("spawn")
    procs: dict[int, Any] = {}
    started: dict[int, float] = {}
    signals: list[int] = []
    quick_failures = 0
    exit_code = 0

    def spawn(index: int) -> None:
        proc = ctx.Process(target=target, args=(*args, index), name=f"van-worker-{index}")
        proc.start()
        procs[index], started[index] = proc, time.monotonic()
        logger.info("worker %d started (pid %s)", index, proc.pid)

    def on_signal(signum: int, frame: Any) -> None:
        signals.append(signum)

    previous = {
        sig: signal.signal(sig, on_signal)
        for sig in (signal.SIGINT, signal.SIGTERM)
        if threading.current_thread() is threading.main_thread()
    }
    forwarded = 0
    stopping_since: float | None = None
    try:
        for index in range(workers):
            spawn(index)
        while True:
            if len(signals) > forwarded:  # forward every signal: 1st drains, 2nd stops
                forwarded = len(signals)
                for proc in procs.values():
                    if proc.is_alive() and proc.pid is not None:
                        with contextlib.suppress(OSError):
                            os.kill(proc.pid, signal.SIGTERM)
            if signals:
                if stopping_since is None:
                    stopping_since = time.monotonic()
                if not any(p.is_alive() for p in procs.values()):
                    break
                if time.monotonic() - stopping_since > drain_timeout + 15:
                    for proc in procs.values():
                        if proc.is_alive():
                            proc.kill()
                    exit_code = 1
                    break
            else:
                for index, proc in list(procs.items()):
                    if proc.is_alive():
                        continue
                    proc.join()
                    lived = time.monotonic() - started[index]
                    logger.warning("worker %d exited with code %s", index, proc.exitcode)
                    quick_failures = quick_failures + 1 if lived < 10 else 0
                    fatal = proc.exitcode in fatal_exit_codes
                    if not restart or fatal or quick_failures >= max_quick_failures:
                        logger.error("worker %d failed; stopping", index)
                        signals.append(signal.SIGTERM)
                        exit_code = 1
                        break
                    spawn(index)
            time.sleep(0.1)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        for proc in procs.values():
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
    return exit_code
