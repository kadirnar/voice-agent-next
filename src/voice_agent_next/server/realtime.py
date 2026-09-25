"""OpenAI-Realtime-compatible WebSocket server: serve any engine at ``/v1/realtime``.

Every engine of this library — a cascade of local models, Gemini Live, a scripted mock —
becomes an OpenAI Realtime endpoint, so existing Realtime clients (the official ``openai``
SDK, the OpenAI Agents SDK, LiveKit/Pipecat OpenAI-realtime plugins, and our own
:class:`~voice_agent_next.providers.openai.realtime.OpenAIRealtimeEngine`) run on it
unchanged::

    import asyncio

    from voice_agent_next.engines.cascade import CascadeEngine
    from voice_agent_next.server import RealtimeServer

    async def main() -> None:
        engine = CascadeEngine(stt="faster_whisper", llm="ollama/qwen3", tts="kokoro",
                               vad="silero")
        server = RealtimeServer(engine, model="local", host="127.0.0.1", port=8000)
        await server.serve_forever()   # clients connect to ws://127.0.0.1:8000/v1/realtime

    asyncio.run(main())

or from the command line: ``van serve --engine agent.yaml --port 8000``.

What the server does (see ``docs/deploy/realtime-server.md`` for the event mapping):

* one engine connection per client session, opened lazily with the session
  configuration the client sent (instructions, tools, voice, turn detection, input
  transcription language); ``?model=`` selects the engine when several are served;
* audio format conversion at the edges: PCM16 at any rate (24 kHz is the standard),
  G.711 μ-law / A-law at 8 kHz, resampled to and from the engine's rates;
* the Realtime event protocol, GA names by default and the beta dialect for clients that
  send ``OpenAI-Beta: realtime=v1`` (or a beta-shaped ``session.update``);
* optional bearer-token authentication (``Authorization: Bearer``, ``api-key`` or the
  browser ``openai-insecure-api-key.<key>`` subprotocol; constant-time comparison);
* secure defaults: browser pages from other websites are refused (``Origin`` allow-list,
  ``allowed_origins``), sessions are limited in number, duration and idle time, and
  clients get generic error messages with a correlation id (details go to the logs);
* backpressure in both directions, ``GET /health`` and ``GET /v1/models``, clean shutdown
  (clients are closed with 1001, engines closed).

The client plays the audio, so it — not the server — knows what the user heard: barge-in
truncation arrives from the client as ``conversation.item.truncate``, exactly like with
OpenAI's API over WebSocket.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from typing import Any, Final, TypeAlias
from urllib.parse import parse_qsl, urlsplit

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.http11 import Request, Response

from ..engine import S2SEngine
from ..errors import ConfigurationError
from ..registry import create
from ..session.agent import DEFAULT_INSTRUCTIONS
from ..utils.clock import now
from ..utils.log import logger
from ._protocol import Dialect, SessionConfig
from ._session import RealtimeSession
from .security import (
    DEFAULT_IDLE_TIMEOUT,
    DEFAULT_MAX_SESSION_DURATION,
    DEFAULT_MAX_SESSIONS,
    MAX_SEND_BUFFER,
    OriginPolicy,
    exposure_warning,
    header_origin,
)

__all__ = [
    "EngineFactory",
    "EngineSource",
    "RealtimeModel",
    "RealtimeServer",
    "engine_from_config",
    "serve_realtime",
]

PROTOCOL: Final = "openai-realtime"
EngineFactory: TypeAlias = Callable[[], "S2SEngine | Awaitable[S2SEngine]"]
"""Builds a new engine for each session (closed when the session ends)."""
EngineSource: TypeAlias = "S2SEngine | str | Mapping[str, Any] | RealtimeModel | EngineFactory"
"""An engine instance (shared by all sessions), a registry spec such as ``"mock"`` or
``{"provider": "openai/gpt-realtime", ...}``, a :class:`RealtimeModel`, or a factory."""

_HEALTH_PATHS: Final = frozenset({"/health", "/healthz"})
_MODELS_PATHS: Final = frozenset({"/v1/models", "/models"})
_BETA_SUBPROTOCOL: Final = "openai-beta.realtime-v1"
_KEY_SUBPROTOCOL: Final = "openai-insecure-api-key."


@dataclass
class RealtimeModel:
    """An engine served under a model name (``wss://host/v1/realtime?model=<name>``).

    Args:
        engine: an :class:`~voice_agent_next.engine.S2SEngine` shared by every session
            (``connect()`` opens one connection per session; models load once), a registry
            spec (created by the server), or a factory building one engine per session.
        instructions: default session instructions (clients usually replace them).
        voice: default voice (``None``: the engine's default).
        language: default input language, passed to the engine (e.g. the STT language).
        name: the served model name (set by :class:`RealtimeServer`).
    """

    engine: S2SEngine | EngineFactory | str | Mapping[str, Any]
    instructions: str = DEFAULT_INSTRUCTIONS
    voice: str | None = None
    language: str | None = None
    name: str = ""
    owned: bool = field(default=False, repr=False)
    """The server created the engine and closes it on shutdown."""

    def session_config(self) -> SessionConfig:
        """The configuration a new session starts with."""
        return SessionConfig(instructions=self.instructions, voice=self.voice)

    async def acquire(self) -> tuple[S2SEngine, bool]:
        """The engine for a new session and whether the session owns (closes) it."""
        if isinstance(self.engine, S2SEngine):
            return self.engine, False
        if isinstance(self.engine, (str, Mapping)):
            raise ConfigurationError("serve RealtimeModel specs through RealtimeServer")
        result = self.engine()
        engine = await result if inspect.isawaitable(result) else result
        if not isinstance(engine, S2SEngine):
            raise ConfigurationError(f"engine factory returned {type(engine).__name__}")
        return engine, True


def engine_from_config(config: Any) -> RealtimeModel:
    """A :class:`RealtimeModel` from an agent config (:class:`~voice_agent_next.config.AppConfig`,
    a mapping, or a YAML/TOML/JSON file path): its ``engine`` or cascade components, plus
    the agent's instructions, voice and language as session defaults."""
    from ..config import AppConfig, load_config
    from ..engines.cascade import CascadeEngine, CascadeOptions

    cfg = config if isinstance(config, AppConfig) else load_config(config)
    engine: S2SEngine
    if cfg.engine is not None:
        engine = create("engine", cfg.engine)
    else:
        engine = CascadeEngine(
            stt=cfg.stt, llm=cfg.llm, tts=cfg.tts, vad=cfg.vad,
            turn_detector=cfg.turn_detector, options=CascadeOptions(**cfg.cascade),
        )  # fmt: skip
    if cfg.agent.tools:
        logger.warning(
            "agent tools in the config are not served: Realtime clients declare and run "
            "their own tools"
        )
    return RealtimeModel(
        engine,
        instructions=cfg.agent.instructions or DEFAULT_INSTRUCTIONS,
        voice=cfg.agent.voice,
        language=cfg.agent.language,
        owned=True,
    )


def _default_name(source: Any) -> str:
    if isinstance(source, RealtimeModel) and source.name:
        return source.name
    if isinstance(source, str):
        return source.strip()
    if isinstance(source, Mapping):
        target = str(source.get("provider") or source.get("use") or "default")
        model = source.get("model")
        return f"{target}/{model}" if model and "/" not in target else target
    engine = source.engine if isinstance(source, RealtimeModel) else source
    if isinstance(engine, S2SEngine):
        return engine.model or engine.provider
    return "default"


class RealtimeServer:
    """A WebSocket server speaking the OpenAI Realtime protocol (see the module docs).

    Args:
        engine: the engine to serve (see :data:`EngineSource`), under the name ``model``.
        models: several engines by model name (instead of ``engine``); clients pick one
            with ``?model=<name>``.
        model: served name of ``engine`` (default: the spec string or the engine's model).
        default_model: model used when the client sends no ``?model=`` (default: the first).
        accept_any_model: serve the default model for unknown ``?model=`` names instead of
            rejecting the connection (HTTP 404), for clients with a hard-coded model name
            (``gpt-realtime``...). ``None`` (default): only when a single model is served.
        host / port: listening address (``port=0`` picks a free port; see :attr:`port`).
        api_keys: accepted bearer tokens (``None``: no authentication — a warning is
            logged when the server listens beyond this machine). Compared in constant
            time; never logged.
        allowed_origins: browser origins allowed besides this machine's own pages
            (``http://localhost:*``...) and clients without an ``Origin`` header; others
            get HTTP 403. See :class:`~voice_agent_next.server.security.OriginPolicy`
            (``"https://app.example.com"``, ``"https://*.example.com"``, ``"*"``).
        max_sessions: refuse clients beyond this many live sessions (HTTP 503; default
            64, ``None``: no limit).
        warmup: ``engine.warmup()`` shared engines before accepting clients.
        max_session_duration: close sessions after this many seconds with a
            ``session_expired`` error, like OpenAI (default one hour; ``None``: no limit).
        idle_timeout: close sessions after this many seconds without a client event with
            a ``session_idle`` error (default 5 minutes; ``None``: never).
        max_message_size: largest client message accepted, in bytes.
        max_send_buffer: bytes of server events queued for a client that does not read
            them before the connection is closed (1008).
        engine_connect_timeout: seconds to wait for ``engine.connect()``.
        serve_options: extra ``websockets.asyncio.server.serve`` arguments, e.g. ``ssl``
            (TLS). Passing ``origins`` (the ``websockets`` allow-list) replaces
            ``allowed_origins``.
    """

    def __init__(
        self,
        engine: EngineSource | None = None,
        *,
        models: Mapping[str, EngineSource] | None = None,
        model: str | None = None,
        default_model: str | None = None,
        accept_any_model: bool | None = None,
        host: str = "127.0.0.1",
        port: int = 8000,
        api_keys: str | Sequence[str] | None = None,
        allowed_origins: str | Sequence[str] | None = (),
        max_sessions: int | None = DEFAULT_MAX_SESSIONS,
        warmup: bool = True,
        max_session_duration: float | None = DEFAULT_MAX_SESSION_DURATION,
        idle_timeout: float | None = DEFAULT_IDLE_TIMEOUT,
        max_message_size: int = 16 * 2**20,
        max_send_buffer: int = MAX_SEND_BUFFER,
        engine_connect_timeout: float = 30.0,
        **serve_options: Any,
    ) -> None:
        if engine is not None and models:
            raise ConfigurationError("pass engine=... or models=..., not both")
        if engine is not None:
            models = {model or _default_name(engine): engine}
        if not models:
            raise ConfigurationError("RealtimeServer needs an engine to serve")
        self.models: dict[str, RealtimeModel] = {
            name: self._model(name, source) for name, source in models.items()
        }
        self.default_model = default_model or next(iter(self.models))
        if self.default_model not in self.models:
            raise ConfigurationError(f"default_model {self.default_model!r} is not served")
        if max_sessions is not None and max_sessions < 1:
            raise ConfigurationError("max_sessions must be >= 1")
        for option, value in (
            ("max_session_duration", max_session_duration),
            ("idle_timeout", idle_timeout),
        ):
            if value is not None and value <= 0:
                raise ConfigurationError(f"{option} must be > 0 (None: no limit)")
        try:
            self.origin_policy = OriginPolicy(allowed_origins)
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from None
        keys = [api_keys] if isinstance(api_keys, str) else list(api_keys or [])
        if any(not isinstance(k, str) or not k for k in keys):
            raise ConfigurationError("api_keys must be non-empty strings")
        self._keys = [k.encode() for k in keys]
        self.accept_any_model = (
            len(self.models) == 1 if accept_any_model is None else accept_any_model
        )
        self.host = host
        self.port = port
        self.max_sessions = max_sessions
        self.warmup = warmup
        self.max_session_duration = max_session_duration
        self.idle_timeout = idle_timeout
        self.max_message_size = max_message_size
        self.max_send_buffer = max_send_buffer
        self.engine_connect_timeout = engine_connect_timeout
        self.serve_options = serve_options
        self._server: Server | None = None
        self._sessions: set[RealtimeSession] = set()
        self._closing = False
        self._closed = asyncio.Event()
        self._started_at: float | None = None

    @staticmethod
    def _model(name: str, source: EngineSource) -> RealtimeModel:
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError(f"invalid model name {name!r}")
        if isinstance(source, RealtimeModel):
            model = replace(source, name=name)
            if isinstance(model.engine, (str, Mapping)):
                model.engine, model.owned = create("engine", model.engine), True
            return model
        if isinstance(source, S2SEngine):
            return RealtimeModel(source, name=name)
        if isinstance(source, (str, Mapping)):
            return RealtimeModel(create("engine", source), name=name, owned=True)
        if callable(source):
            return RealtimeModel(source, name=name)
        raise ConfigurationError(f"cannot serve {type(source).__name__} as an engine")

    # ------------------------------------------------------------------ properties
    @property
    def url(self) -> str:
        """Base URL for Realtime clients (``ws://host:port/v1``; they append ``/realtime``)."""
        scheme = "wss" if self.serve_options.get("ssl") is not None else "ws"
        host = self.host or "localhost"
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1" if host == "0.0.0.0" else "::1"
        if ":" in host:
            host = f"[{host}]"
        return f"{scheme}://{host}:{self.port}/v1"

    @property
    def sessions(self) -> list[RealtimeSession]:
        """Live sessions."""
        return list(self._sessions)

    def health(self) -> dict[str, Any]:
        """The ``GET /health`` document."""
        return {
            "status": "closing" if self._closing else "ok",
            "protocol": PROTOCOL,
            "sessions": len(self._sessions),
            "max_sessions": self.max_sessions,
            "models": list(self.models),
            "default_model": self.default_model,
            "uptime": 0.0 if self._started_at is None else round(now() - self._started_at, 3),
        }

    # ------------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        """Warm the engines up and start listening (idempotent)."""
        if self._server is not None:
            return
        if self.warmup:
            shared = {id(m.engine): m.engine for m in self.models.values()}
            engines = [e for e in shared.values() if isinstance(e, S2SEngine)]
            results = await asyncio.gather(*(e.warmup() for e in engines), return_exceptions=True)
            for engine, result in zip(engines, results, strict=True):
                if isinstance(result, Exception):
                    logger.warning("warmup of %s failed (continuing): %s", engine.provider, result)
        options: dict[str, Any] = {
            "compression": None,  # base64 audio does not compress; save the CPU
            "max_size": self.max_message_size,
            "process_request": self._process_request,
            "select_subprotocol": _select_subprotocol,
            **self.serve_options,
        }
        self._server = await serve(self._handle, self.host, self.port, **options)
        for sock in self._server.sockets:
            self.port = int(sock.getsockname()[1])
            break
        self._started_at = now()
        warning = exposure_warning(
            self.host, authenticated=bool(self._keys), what="the OpenAI Realtime server"
        )
        if warning is not None:
            logger.warning(warning)
        logger.info(
            "serving %s on %s/realtime (models: %s)", PROTOCOL, self.url, ", ".join(self.models)
        )

    async def serve_forever(self) -> None:
        """Serve until :meth:`aclose` is called or this coroutine is cancelled."""
        await self.start()
        try:
            await self._closed.wait()
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        """Stop accepting clients, close every session (1001) and the engines it created."""
        if self._closing:
            await self._closed.wait()
            return
        self._closing = True
        server, self._server = self._server, None
        if server is not None:
            server.close()  # closes connections with 1001 and waits for their handlers
            await server.wait_closed()
        for model in self.models.values():
            if model.owned and isinstance(model.engine, S2SEngine):
                with contextlib.suppress(Exception):
                    await model.engine.aclose()
        self._closed.set()

    async def __aenter__(self) -> RealtimeServer:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------- handshake
    def _process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        path = urlsplit(request.path).path.rstrip("/") or "/"
        if path in _HEALTH_PATHS:
            status = HTTPStatus.SERVICE_UNAVAILABLE if self._closing else HTTPStatus.OK
            return _json_response(connection, status, self.health())
        if path in _MODELS_PATHS:
            data = [
                {"id": name, "object": "model", "created": 0, "owned_by": "voice-agent-next"}
                for name in self.models
            ]
            return _json_response(connection, HTTPStatus.OK, {"object": "list", "data": data})
        if path.rsplit("/", 1)[-1] != "realtime":
            return _error_response(
                connection, HTTPStatus.NOT_FOUND, "not_found",
                f"Unknown path {path!r}: connect to /v1/realtime.",
            )  # fmt: skip
        if self._closing:
            return _error_response(connection, HTTPStatus.SERVICE_UNAVAILABLE, "server_closing",
                                   "The server is shutting down.")  # fmt: skip
        if not self._origin_allowed(request):
            return _error_response(
                connection, HTTPStatus.FORBIDDEN, "origin_not_allowed",
                "This origin may not connect to this server.",
                error_type="invalid_request_error",
            )  # fmt: skip
        if not self._authorized(request):
            return _error_response(
                connection, HTTPStatus.UNAUTHORIZED, "invalid_api_key",
                "Incorrect or missing API key (send 'Authorization: Bearer <key>').",
                error_type="authentication_error",
            )  # fmt: skip
        if self._resolve_model(request) is None:
            name = dict(parse_qsl(urlsplit(request.path).query)).get("model", "")
            return _error_response(
                connection, HTTPStatus.NOT_FOUND, "model_not_found",
                f"The model {name!r} is not served here. Available: {', '.join(self.models)}.",
            )  # fmt: skip
        if self.max_sessions is not None and len(self._sessions) >= self.max_sessions:
            response = _error_response(
                connection, HTTPStatus.SERVICE_UNAVAILABLE, "session_limit_reached",
                f"The server already runs {self.max_sessions} sessions; try again later.",
            )  # fmt: skip
            response.headers["Retry-After"] = "1"
            return response
        return None

    def _origin_allowed(self, request: Request) -> bool:
        if "origins" in self.serve_options:  # the websockets allow-list decides
            return True
        origin = header_origin(request.headers.get_all("Origin"))
        if self.origin_policy.allows(origin):
            return True
        logger.warning(
            "refused a Realtime client from origin %r (allowed_origins / --allowed-origin)",
            origin,
        )
        return False

    def _authorized(self, request: Request) -> bool:
        if not self._keys:
            return True
        headers = request.headers
        candidates: list[str] = []
        for value in headers.get_all("Authorization"):
            scheme, _, token = value.strip().partition(" ")
            if scheme.lower() == "bearer" and token.strip():
                candidates.append(token.strip())
        candidates.extend(v.strip() for v in headers.get_all("api-key") if v.strip())
        for proto in _subprotocols(request):
            if proto.startswith(_KEY_SUBPROTOCOL):
                candidates.append(proto[len(_KEY_SUBPROTOCOL) :])
        return any(self._key_matches(c) for c in candidates)

    def _key_matches(self, candidate: str) -> bool:
        given = candidate.encode()
        matched = False
        for key in self._keys:  # compare with every key: no early exit
            matched |= hmac.compare_digest(given, key)
        return matched

    def _resolve_model(self, request: Request) -> RealtimeModel | None:
        name = dict(parse_qsl(urlsplit(request.path).query)).get("model", "").strip()
        if not name:
            return self.models[self.default_model]
        model = self.models.get(name)
        if model is None and self.accept_any_model:
            model = self.models[self.default_model]
        return model

    @staticmethod
    def _dialect(request: Request) -> Dialect:
        beta = any("realtime=v1" in v for v in request.headers.get_all("OpenAI-Beta"))
        return "beta" if beta or _BETA_SUBPROTOCOL in _subprotocols(request) else "ga"

    async def _handle(self, websocket: ServerConnection) -> None:
        request = websocket.request
        model = self._resolve_model(request) if request is not None else None
        if request is None or model is None:
            await websocket.close(1008, "unknown model")
            return
        if self.max_sessions is not None and len(self._sessions) >= self.max_sessions:
            await websocket.close(1013, "session limit reached")
            return
        session = RealtimeSession(self, websocket, model, dialect=self._dialect(request))
        self._sessions.add(session)
        try:
            await session.run()
        finally:
            self._sessions.discard(session)


async def serve_realtime(
    engine: EngineSource | None = None, *, host: str = "127.0.0.1", port: int = 8000, **options: Any
) -> RealtimeServer:
    """Start a :class:`RealtimeServer` and return it (``await server.serve_forever()``)."""
    server = RealtimeServer(engine, host=host, port=port, **options)
    await server.start()
    return server


# ----------------------------------------------------------------------------- helpers
def _subprotocols(request: Request) -> list[str]:
    return [
        p.strip()
        for value in request.headers.get_all("Sec-WebSocket-Protocol")
        for p in value.split(",")
        if p.strip()
    ]


def _select_subprotocol(connection: ServerConnection, subprotocols: Sequence[str]) -> Any:
    """Browsers pass ``realtime`` (plus the key and beta markers) as subprotocols."""
    return "realtime" if "realtime" in subprotocols else None


def _json_response(connection: ServerConnection, status: HTTPStatus, body: Any) -> Response:
    response = connection.respond(status, json.dumps(body) + "\n")
    del response.headers["Content-Type"]
    response.headers["Content-Type"] = "application/json"
    return response


def _error_response(
    connection: ServerConnection,
    status: HTTPStatus,
    code: str,
    message: str,
    *,
    error_type: str = "invalid_request_error",
) -> Response:
    body = {"error": {"type": error_type, "code": code, "message": message}}
    return _json_response(connection, status, body)
