"""``van serve`` — serve voice agents over the network, ready for production.

One command for every protocol::

    van serve --protocol openai-realtime --engine agent.yaml     # ws://host:port/v1/realtime
    van serve --protocol websocket --preset local-cpu            # van-ws/1
    van serve --protocol twilio --config agent.yaml --prewarm 2 --max-sessions 20 --workers 4

with prewarmed engines (``--prewarm``), a per-process session limit (``--max-sessions``),
worker processes sharing the port (``--workers``), graceful drain (``--drain-timeout``) and
``/health``, ``/ready`` and ``/metrics`` endpoints. See ``docs/deploy/serving.md`` and
``docs/deploy/realtime-server.md``.

Secure by default: only this machine's pages and native clients may connect
(``--allowed-origin`` adds browser origins), sessions are limited (``--max-sessions`` 64,
``--max-session-duration`` 1 h, ``--idle-timeout`` 5 min), and ``--protocol
openai-realtime`` refuses to listen beyond this machine without ``--api-key`` unless
``--insecure`` is given (the other protocols have no built-in authentication: they warn).

The command is built in steps: :func:`build_models` (OpenAI Realtime) or
:func:`build_app_config` (the other protocols) turn the engine options into what is
served, :func:`build_served` turns that and :class:`ServeOptions` into a
:class:`~voice_agent_next.server.serving.Served` (:func:`build_server` returns just its
server), and :func:`run` runs it in one process or in a worker pool.
"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import signal
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape

app = typer.Typer(
    name="serve",
    help="Serve voice agents over the network (OpenAI Realtime, WebSocket, WebRTC, "
    "telephony) with prewarm, limits, workers and health checks.",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
console = Console(stderr=True)

PROTOCOLS = ("openai-realtime", "websocket", "webrtc", "twilio", "telnyx", "vonage", "plivo")
"""Served protocols (``voice_agent_next.server.serving.PROTOCOLS``, without the import)."""
TELEPHONY_PROTOCOLS = ("twilio", "telnyx", "vonage", "plivo")
ANSWER_PATH = "/answer"
"""Where ``van serve`` serves the telephony answer webhook (``GET``)."""
SECRET_ENV_NAME = "VAN_TELEPHONY_SECRET"
_CONFIG_SUFFIXES = (".yaml", ".yml", ".toml", ".json")
_NAMED = re.compile(r"^([A-Za-z0-9][\w.:\-]*)=(.+)$", re.S)
_DEFAULT_PORTS = {"openai-realtime": 8000, "webrtc": 8080}
_CONFIG_ERROR_EXIT = 2
_DEFAULT_MAX_SESSIONS = 64  # voice_agent_next.server.security.DEFAULT_MAX_SESSIONS
_DEFAULT_MAX_SESSION_DURATION = 3600.0
_DEFAULT_IDLE_TIMEOUT = 300.0


@dataclass
class ServeOptions:
    """Server settings shared by every protocol."""

    protocol: str = "openai-realtime"
    host: str = "127.0.0.1"
    port: int = 8000
    api_keys: list[str] = field(default_factory=list)
    """Accepted API keys: bearer tokens of ``openai-realtime``, ``websocket`` and
    ``webrtc`` clients; the telephony answer webhook's ``?key=``."""
    allowed_origins: list[str] = field(default_factory=list)
    """Browser origins allowed besides this machine's pages (``*``: any)."""
    insecure: bool = False
    """Allow listening beyond this machine without authentication."""
    public_url: str | None = None
    """Telephony: the public ``wss://``/``https://`` base URL of the server (carrier
    signature checks for Twilio and Plivo, stream URL of the served markup)."""
    stream_secret: str | None = None
    """Telephony: the stream-token secret (``None``: ``VAN_TELEPHONY_SECRET`` or generated
    by :func:`resolve_auth`)."""
    signature_secret: str | None = None
    """Vonage: the account signature secret that verifies Vonage's JWTs."""
    generated: list[str] = field(default_factory=list)
    """What :func:`resolve_auth` generated for this run (``"api key"``, ``"stream secret"``)."""
    max_sessions: int | None = _DEFAULT_MAX_SESSIONS
    accept_any_model: bool | None = None
    warmup: bool = True
    max_session_duration: float | None = _DEFAULT_MAX_SESSION_DURATION
    idle_timeout: float | None = _DEFAULT_IDLE_TIMEOUT
    """Close sessions after this many seconds without client messages (``None``: never)."""
    prewarm: int = 0
    """Prewarmed engines (connections) kept ready per model."""
    engine_per_session: bool = False
    """Give every session its own engine instance (default: one engine per process)."""
    preconnect: bool = True
    """Prewarmed items include an open engine connection."""
    prewarm_max_idle: float | None = 300.0
    """Recycle prewarmed connections older than this (seconds)."""
    workers: int = 1
    drain_timeout: float = 30.0
    log_format: str = "text"
    verbose: bool = False


@dataclass
class SourceOptions:
    """What to serve: engines, a preset, a config file or cascade components."""

    engines: list[str] = field(default_factory=list)
    preset: str | None = None
    config: str | None = None
    stt: str | None = None
    llm: str | None = None
    tts: str | None = None
    vad: str | None = None
    turn_detector: str | None = None
    name: str | None = None
    instructions: str | None = None
    voice: str | None = None
    language: str | None = None


def _source_name(source: str) -> str:
    path = Path(source)
    if path.suffix.lower() in _CONFIG_SUFFIXES:
        return path.stem
    if source.strip().startswith("{"):
        from ..bench.system import parse_component_spec

        spec = parse_component_spec(source)
        if isinstance(spec, dict):
            target = str(spec.get("provider") or spec.get("use") or "engine")
            model = spec.get("model")
            return f"{target}/{model}" if model and "/" not in target else target
    return source.strip()


def _is_config(source: str) -> bool:
    return Path(source).suffix.lower() in _CONFIG_SUFFIXES


def _load_source(source: str) -> Any:
    """A config file becomes a :class:`RealtimeModel`; anything else an engine spec."""
    from ..bench.system import parse_component_spec
    from ..errors import ConfigurationError
    from ..server import engine_from_config

    path = Path(source)
    if _is_config(source):
        if not path.is_file():
            raise ConfigurationError(f"config file not found: {path}")
        return engine_from_config(path)
    return parse_component_spec(source)


def _first_then(first: Any, source: str) -> Callable[[], Any]:
    """A builder returning ``first`` once, then loading ``source`` again on every call."""
    pending = [first]

    def build() -> Any:
        return pending.pop() if pending else _load_source(source)

    return build


def _model_builders(
    engines: list[str] | None,
    *,
    stt: str | None,
    llm: str | None,
    tts: str | None,
    vad: str | None,
    turn_detector: str | None,
    name: str | None,
    preset: str | None = None,
    config: str | None = None,
) -> list[tuple[str, Callable[[], Any]]]:
    """``(model name, builder)`` pairs; a builder returns a fresh model source."""
    from ..bench.system import parse_component_spec
    from ..engines.cascade import CascadeEngine
    from ..errors import ConfigurationError
    from ..server import RealtimeModel, engine_from_config

    builders: list[tuple[str, Callable[[], Any]]] = []
    for value in engines or []:
        match = _NAMED.match(value.strip())
        model_name, source = (match.group(1), match.group(2)) if match else (None, value)
        first = _load_source(source)  # validates now (missing file, bad inline spec)
        builders.append((model_name or _source_name(source), _first_then(first, source)))
    cascade = {"stt": stt, "llm": llm, "tts": tts, "turn_detector": turn_detector}
    if preset is not None or config is not None:
        # one model: --preset < --config < cascade flags, like `van run`
        overrides = {
            key: parse_component_spec(value)
            for key, value in {**cascade, "vad": vad}.items()
            if value is not None
        }
        layered = layered_config(preset=preset, config=config, overrides=overrides)
        base_name = preset if preset is not None else _source_name(str(config))
        model_name = name if name is not None and not engines else base_name
        builders.append((model_name, lambda: engine_from_config(layered)))
        return builders
    if any(v is not None for v in cascade.values()):
        if llm is None:  # --tts is optional for audio-output LLMs (the cascade checks)
            raise ConfigurationError("a cascade needs at least --llm and --tts (and --stt)")

        def build_cascade() -> Any:
            engine = CascadeEngine(
                stt=parse_component_spec(stt), llm=parse_component_spec(llm),
                tts=parse_component_spec(tts), vad=parse_component_spec(vad),
                turn_detector=parse_component_spec(turn_detector),
            )  # fmt: skip
            return RealtimeModel(engine, owned=True)

        builders.append((name or "cascade", build_cascade))
    elif vad is not None:
        raise ConfigurationError("--vad belongs to a cascade (--stt/--llm/--tts)")
    if not builders:
        builders.append((name or "mock", lambda: "mock"))
    elif name is not None and len(builders) == 1 and engines and not _NAMED.match(engines[0]):
        builders[0] = (name, builders[0][1])  # --name renames a single --engine
    return builders


def build_models(
    engines: list[str] | None = None,
    *,
    stt: str | None = None,
    llm: str | None = None,
    tts: str | None = None,
    vad: str | None = None,
    turn_detector: str | None = None,
    name: str | None = None,
    instructions: str | None = None,
    voice: str | None = None,
    language: str | None = None,
    preset: str | None = None,
    config: str | None = None,
    per_session: bool = False,
) -> dict[str, Any]:
    """Served models from CLI options.

    ``engines`` entries are ``[NAME=]SOURCE`` where SOURCE is a registry spec (``mock``,
    ``openai/gpt-realtime``), an inline mapping (``{provider: mock, response_delay: 0.2}``)
    or an agent config file (engine or cascade; its agent instructions/voice/language
    become session defaults). ``preset`` and/or ``config`` serve one more model, layered
    like ``van run``: ``preset`` < ``config`` < the cascade options
    (``stt``/``llm``/``tts``/``vad``/``turn_detector``), named after the preset or the
    file (or ``name``). Without them, the cascade options build one more model, a cascade
    (named ``name`` or ``"cascade"``). Nothing given: the mock engine.

    ``per_session``: every model's engine is a factory (one engine per session).
    """
    from ..errors import ConfigurationError
    from ..registry import create
    from ..server import RealtimeModel

    builders = _model_builders(
        engines, stt=stt, llm=llm, tts=tts, vad=vad, turn_detector=turn_detector, name=name,
        preset=preset, config=config,
    )  # fmt: skip
    models: dict[str, Any] = {}
    for model_name, builder in builders:
        if model_name in models:
            raise ConfigurationError(f"model name {model_name!r} is used twice; use NAME=SOURCE")
        source = builder()
        if not isinstance(source, RealtimeModel):
            source = RealtimeModel(source)
        if per_session:

            def factory(builder: Callable[[], Any] = builder) -> Any:
                built = builder()
                engine = built.engine if isinstance(built, RealtimeModel) else built
                return create("engine", engine) if isinstance(engine, str | Mapping) else engine

            source.engine, source.owned = factory, False  # the probe engine is never started
        if instructions is not None:
            source.instructions = instructions
        if voice is not None:
            source.voice = voice
        if language is not None:
            source.language = language
        models[model_name] = source
    return models


def layered_config(*, preset: str | None, config: str | None, overrides: Mapping[str, Any]) -> Any:
    """The validated :class:`~voice_agent_next.config.AppConfig` of ``preset`` < ``config``
    file < ``overrides`` (the component flags), like ``van run``. A preset is checked for
    readiness on this machine (with the layers on top)."""
    from ..config import AppConfig, layer_config

    raw = layer_config(preset=preset, file=config, overrides=dict(overrides))
    if preset is not None:
        from ..presets import load_preset

        cfg = load_preset(preset, **{k: v for k, v in raw.items() if k != "extends"})
    else:
        cfg = AppConfig.model_validate(raw)
    cfg.validate_components()
    return cfg


def build_app_config(sources: SourceOptions) -> Any:
    """The :class:`~voice_agent_next.config.AppConfig` served by the AgentSession
    protocols (``websocket``, ``webrtc``, telephony): one agent, layered like ``van run``
    as ``--preset`` < ``--config`` < ``--engine`` or cascade flags and validated once
    (nothing given: the mock engine). A ``--preset`` is checked for readiness."""
    from ..bench.system import parse_component_spec
    from ..errors import ConfigurationError

    engines = list(sources.engines)
    configs = [e for e in engines if _is_config(e)]
    if sources.config is not None:
        configs.insert(0, sources.config)
    specs = [e for e in engines if not _is_config(e)]
    cascade = any(
        v is not None for v in (sources.stt, sources.llm, sources.tts, sources.turn_detector)
    )
    base = sources.preset is not None or bool(configs)
    if len(configs) > 1 or len(specs) > 1:
        raise ConfigurationError(
            "this protocol serves one agent: pass --preset and/or one --config, then "
            "--engine or cascade flags on top (layered in that order)"
        )
    if specs and cascade:
        raise ConfigurationError(
            "this protocol serves one agent: --engine or cascade flags (--stt/--llm/--tts), "
            "not both"
        )
    if any(_NAMED.match(e.strip()) for e in specs):
        raise ConfigurationError("NAME=SOURCE model names only apply to --protocol openai-realtime")
    overrides: dict[str, Any] = {}
    if specs:
        overrides["engine"] = parse_component_spec(specs[0])
    for key in ("stt", "llm", "tts", "vad", "turn_detector"):
        value = getattr(sources, key)
        if value is not None:
            overrides[key] = parse_component_spec(value)
    if not base:  # the flags alone make the agent
        if cascade and sources.llm is None:  # --tts is optional for audio-output LLMs
            raise ConfigurationError("a cascade needs at least --llm and --tts (and --stt)")
        if sources.vad is not None and not cascade:
            raise ConfigurationError("--vad belongs to a cascade (--stt/--llm/--tts)")
        if not overrides:
            overrides["engine"] = "mock"
    cfg = layered_config(
        preset=sources.preset, config=configs[0] if configs else None, overrides=overrides
    )
    if sources.instructions is not None:
        cfg.agent.instructions = sources.instructions
    if sources.voice is not None:
        cfg.agent.voice = sources.voice
    if sources.language is not None:
        cfg.agent.language = sources.language
    cfg.validate_components()
    return cfg


def build_served(
    sources: SourceOptions, options: ServeOptions, *, worker: int | None = None
) -> Any:
    """The :class:`~voice_agent_next.server.serving.Served` for ``options.protocol``."""
    from ..errors import ConfigurationError
    from ..server.serving import (
        AGENT_PROTOCOLS,
        build_agent_served,
        build_realtime_served,
        reuse_port_supported,
    )

    resolve_auth(options)  # no-op when run() did it already (workers share its result)
    reuse_port = options.workers > 1 and reuse_port_supported()
    pool: dict[str, Any] = {
        "prewarm": options.prewarm,
        "preconnect": options.preconnect,
        "max_idle": options.prewarm_max_idle,
        "warmup": options.warmup,
        "worker": worker,
        "reuse_port": reuse_port,
    }
    if options.protocol == "openai-realtime":
        models = build_models(
            sources.engines, config=sources.config,
            stt=sources.stt, llm=sources.llm, tts=sources.tts, vad=sources.vad,
            turn_detector=sources.turn_detector, name=sources.name,
            instructions=sources.instructions, voice=sources.voice, language=sources.language,
            preset=sources.preset, per_session=options.engine_per_session,
        )  # fmt: skip
        return build_realtime_served(
            models, host=options.host, port=options.port, api_keys=options.api_keys or None,
            max_sessions=options.max_sessions, accept_any_model=options.accept_any_model,
            max_session_duration=options.max_session_duration,
            idle_timeout=options.idle_timeout, allowed_origins=options.allowed_origins,
            **pool,
        )  # fmt: skip
    if options.protocol in AGENT_PROTOCOLS:
        return build_agent_served(
            options.protocol, build_app_config(sources),
            engine_per_session=options.engine_per_session, max_sessions=options.max_sessions,
            host=options.host, port=options.port, allowed_origins=options.allowed_origins,
            max_session_duration=options.max_session_duration,
            idle_timeout=options.idle_timeout, **_protocol_options(options), **pool,
        )  # fmt: skip
    raise ConfigurationError(
        f"unknown protocol {options.protocol!r}; expected one of {', '.join(PROTOCOLS)}"
    )


def build_server(models: dict[str, Any], options: ServeOptions) -> Any:
    """The OpenAI Realtime server for ``models`` (other protocols: :func:`build_served`)."""
    from ..errors import ConfigurationError
    from ..server.serving import build_realtime_served

    if options.protocol != "openai-realtime":
        raise ConfigurationError(
            f"unknown protocol {options.protocol!r} for build_server(); expected "
            "openai-realtime (use build_served() for the others)"
        )
    return build_realtime_served(
        models, host=options.host, port=options.port, api_keys=options.api_keys or None,
        max_sessions=options.max_sessions, accept_any_model=options.accept_any_model,
        max_session_duration=options.max_session_duration, idle_timeout=options.idle_timeout,
        allowed_origins=options.allowed_origins, prewarm=options.prewarm,
        preconnect=options.preconnect, max_idle=options.prewarm_max_idle,
        warmup=options.warmup,
    ).server  # fmt: skip


# --------------------------------------------------------------------------- running
def _configure_logging(options: ServeOptions, worker: int | None = None) -> None:
    from ..server.ops import configure_logging

    configure_logging(
        level=logging.DEBUG if options.verbose else logging.INFO,
        fmt=options.log_format,
        worker=worker,
    )
    if not options.verbose:  # one line per handshake and health probe otherwise
        logging.getLogger("websockets").setLevel(logging.WARNING)


def _banner(served: Any, options: ServeOptions, workers: int = 1) -> None:
    extra = []
    models = getattr(served.server, "models", None)
    if isinstance(models, dict):
        extra.append(f"models: {', '.join(models)}")
    if options.protocol in ("openai-realtime", "websocket", "webrtc"):
        extra.append("API key" if options.api_keys else "no authentication")
    if options.allowed_origins:
        extra.append(f"origins: {', '.join(options.allowed_origins)}")
    if options.prewarm:
        extra.append(f"prewarm {options.prewarm}")
    if options.max_sessions:
        extra.append(f"max {options.max_sessions} sessions")
    if workers > 1:
        extra.append(f"{workers} workers")
    console.print(
        f"[bold]van serve[/bold] · {options.protocol} · [bold]{escape(served.url)}[/bold]"
        + "".join(f" · {escape(x)}" for x in extra),
        highlight=False,
    )
    _print_credentials(options, answer_url=getattr(served.server, "answer_url", None))


def _print_credentials(options: ServeOptions, *, answer_url: Callable[[], Any] | None) -> None:
    """Tell the operator what was generated and where the carrier's webhook points."""
    key = options.api_keys[0] if options.api_keys else None
    if "api key" in options.generated and key is not None:
        if options.protocol in TELEPHONY_PROTOCOLS:
            what = "the answer webhook"
        else:
            what = "clients (Authorization: Bearer <key>)"
        console.print(
            f"[yellow]API key for {escape(what)}, generated for this run:[/yellow] "
            f"[bold]{escape(key)}[/bold] [dim](set --api-key or VAN_SERVER_API_KEY to keep "
            "one; --insecure turns it off beyond loopback)[/dim]",
            highlight=False,
        )
    if "stream secret" in options.generated:
        console.print(
            "[dim]stream tokens use a secret generated for this run: only the served "
            f"answer webhook can issue them (set {SECRET_ENV_NAME} to share it with your "
            "own webhook)[/dim]",
            highlight=False,
        )
    if options.protocol in TELEPHONY_PROTOCOLS and answer_url is not None:
        url = answer_url()
        if url:
            if not _carrier_checked(options) and key is not None:
                url += "?key=" + (key if "api key" in options.generated else "<api key>")
            console.print(
                f"answer webhook (HTTP GET): [bold]{escape(url)}[/bold]"
                + ("" if options.public_url else " [dim](behind your public https:// host)[/dim]"),
                highlight=False,
            )


def _worker_main(sources: SourceOptions, options: ServeOptions, index: int) -> None:
    """Entry point of a worker process (spawned by :func:`run`)."""
    from ..errors import VoiceAgentError
    from ..server.serving import run_served

    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the supervisor forwards SIGTERM
    _configure_logging(options, worker=index)
    log = logging.getLogger("voice_agent_next")
    try:
        served = build_served(sources, options, worker=index)
    except (VoiceAgentError, ValueError) as exc:
        log.error("worker %d: %s", index, exc)
        sys.exit(_CONFIG_ERROR_EXIT)

    def started(served: Any) -> None:
        log.info("worker %d serving %s on %s", index, options.protocol, served.url)

    asyncio.run(
        run_served(
            served, drain_timeout=options.drain_timeout, on_started=started, handle_sigint=False
        )
    )


def check_exposure(options: ServeOptions) -> str | None:
    """Refuse (return the error) or warn about listening beyond this machine unauthenticated.

    ``openai-realtime`` refuses a non-loopback bind without ``--api-key`` unless
    ``--insecure`` says the network or a proxy protects the server. ``websocket`` and
    ``webrtc`` get an API key from :func:`resolve_auth` (called first by :func:`run`); when
    they have none, they get a warning, unless ``--insecure``. Telephony media streams
    are authenticated by their stream token.
    """
    from ..server.security import is_loopback_host

    if options.insecure or is_loopback_host(options.host):
        return None
    if options.protocol in TELEPHONY_PROTOCOLS or options.api_keys:
        return None
    if options.protocol == "openai-realtime":
        if options.api_keys:
            return None
        return (
            f"refusing to serve openai-realtime on {options.host} without authentication: "
            "anyone who can reach the port could use (and pay for) the engines. Pass "
            "--api-key KEY (or VAN_SERVER_API_KEY), bind --host 127.0.0.1, or add "
            "--insecure if a firewall or an authenticating proxy protects the port."
        )
    console.print(
        f"[yellow]warning: {escape(options.protocol)} on {escape(options.host)} has no "
        "authentication: anyone who can reach the port can open sessions. Put an "
        "authenticating proxy in front (see docs/deploy/serving.md), or pass --insecure to "
        "silence this warning.[/yellow]"
    )
    return None


def resolve_auth(options: ServeOptions) -> None:
    """Fill in the credentials this run needs and nobody gave (in place; see
    :attr:`ServeOptions.generated`). Secure by default:

    * ``websocket`` / ``webrtc`` beyond this machine: an API key clients must send, unless
      ``--api-key`` gives one or ``--insecure`` turns authentication off;
    * telephony: the stream secret (``--stream-secret`` or ``VAN_TELEPHONY_SECRET``,
      else a random one: the served answer webhook hands out the tokens), and an API key
      for the answer webhook when the carrier's signature is not checked (Twilio/Plivo
      need ``--public-url``, Vonage ``--vonage-signature-secret``).

    ``openai-realtime`` never gets a generated key: it refuses to listen beyond this
    machine without one (:func:`check_exposure`).
    """
    from ..server.security import generate_api_key, is_loopback_host
    from ..transports.telephony.auth import stream_secret_from_env

    protocol = options.protocol
    if protocol in TELEPHONY_PROTOCOLS:
        if not options.stream_secret:
            options.stream_secret = stream_secret_from_env()
        if not options.stream_secret:
            options.stream_secret = secrets.token_urlsafe(32)
            options.generated.append("stream secret")
        if not options.api_keys and not _carrier_checked(options):
            options.api_keys = [generate_api_key()]
            options.generated.append("api key")
    elif protocol in ("websocket", "webrtc"):
        exposed = not is_loopback_host(options.host)
        if exposed and not options.api_keys and not options.insecure:
            options.api_keys = [generate_api_key()]
            options.generated.append("api key")


def _answer_url(options: ServeOptions) -> str:
    """The answer webhook URL (``--public-url`` host, else the listening address)."""
    from ..transports.websocket import _url_host

    if options.public_url is not None:
        from ..transports.telephony.webhook import public_base

        base = public_base(options.public_url)
    else:
        base = f"http://{_url_host(options.host)}:{options.port}"
    return base.rstrip("/") + ANSWER_PATH


def _carrier_checked(options: ServeOptions) -> bool:
    """The telephony server checks the carrier's signature (webhook and upgrades)."""
    if options.protocol in ("twilio", "plivo"):
        return options.public_url is not None
    return options.protocol == "vonage" and bool(options.signature_secret)


def _protocol_options(options: ServeOptions) -> dict[str, Any]:
    """Protocol-specific server arguments of :func:`build_agent_served`."""
    from ..errors import ConfigurationError

    keys = options.api_keys or None
    if options.protocol in TELEPHONY_PROTOCOLS:
        out: dict[str, Any] = {
            "answer_path": ANSWER_PATH,
            "api_keys": keys,
            "public_url": options.public_url,
        }
        if options.stream_secret:
            out["stream_secret"] = options.stream_secret
        if options.protocol == "vonage" and options.signature_secret:
            out["signature_secret"] = options.signature_secret
        return out
    if options.public_url is not None:
        raise ConfigurationError("--public-url applies to the telephony protocols")
    return {"api_keys": keys}


def run(sources: SourceOptions, options: ServeOptions) -> int:
    """Run ``van serve``: one process, or a supervisor and ``options.workers`` workers."""
    from ..errors import VoiceAgentError
    from ..server.serving import free_port, reuse_port_supported, run_served, run_workers

    resolve_auth(options)  # before the workers start: they share what is generated
    refusal = check_exposure(options)
    if refusal is not None:
        console.print(f"[red]error:[/red] {escape(refusal)}")
        return _CONFIG_ERROR_EXIT
    _configure_logging(options)
    workers = options.workers
    if workers > 1 and not reuse_port_supported():
        console.print(
            "[yellow]--workers needs SO_REUSEPORT (Linux, macOS); running one process. "
            "Run several instances behind a load balancer instead.[/yellow]"
        )
        workers = options.workers = 1
    served = None
    try:  # with workers, validate in the parent too: bad options fail fast and clearly
        if workers == 1:
            served = build_served(sources, options)
        else:
            _validate(sources, options)
    except (VoiceAgentError, ValueError) as exc:
        console.print(f"[red]error:[/red] {escape(str(exc))}")
        return _CONFIG_ERROR_EXIT
    if not (sources.engines or sources.llm or sources.preset or sources.config):
        console.print("[yellow]no engine given; serving the mock engine[/yellow]")
    if served is not None:
        try:
            asyncio.run(
                run_served(
                    served,
                    drain_timeout=options.drain_timeout,
                    on_started=lambda s: _banner(s, options),
                )
            )
        except KeyboardInterrupt:
            pass
        except OSError as exc:  # e.g. the port is in use
            console.print(f"[red]error:[/red] {escape(str(exc))}")
            return 1
        console.print("[dim]bye[/dim]")
        return 0
    if options.port == 0:
        options.port = free_port(options.host)
    console.print(
        f"[bold]van serve[/bold] · {options.protocol} · {workers} workers on "
        f"{escape(options.host)}:{options.port} (SO_REUSEPORT)",
        highlight=False,
    )
    _print_credentials(options, answer_url=lambda: _answer_url(options))
    code = run_workers(
        _worker_main,
        (sources, options),
        workers,
        drain_timeout=options.drain_timeout,
        fatal_exit_codes=(_CONFIG_ERROR_EXIT,),
    )
    console.print("[dim]bye[/dim]")
    return code


def _validate(sources: SourceOptions, options: ServeOptions) -> None:
    """Check the options without starting anything (the workers build the real thing)."""
    from ..errors import ConfigurationError

    if options.protocol == "openai-realtime":
        build_models(
            sources.engines, config=sources.config,
            stt=sources.stt, llm=sources.llm, tts=sources.tts, vad=sources.vad,
            turn_detector=sources.turn_detector, name=sources.name, preset=sources.preset,
        )  # fmt: skip
    elif options.protocol in PROTOCOLS:
        build_app_config(sources)
    else:
        raise ConfigurationError(
            f"unknown protocol {options.protocol!r}; expected one of {', '.join(PROTOCOLS)}"
        )


# ------------------------------------------------------------------------------- CLI
@app.callback(invoke_without_command=True)
def serve(
    ctx: typer.Context,
    engine: Annotated[
        list[str] | None,
        typer.Option(
            "--engine",
            "-e",
            help="Engine to serve: a spec (mock, openai/gpt-realtime), an inline mapping "
            "('{provider: mock}') or an agent config file (YAML/TOML/JSON, engine or "
            "cascade). openai-realtime: NAME=SOURCE sets the model name; repeat to serve "
            "several models.",
        ),
    ] = None,
    protocol: Annotated[
        str, typer.Option("--protocol", "-p", help=f"Wire protocol: {', '.join(PROTOCOLS)}")
    ] = "openai-realtime",
    preset: Annotated[
        str | None, typer.Option("--preset", help="Serve a preset (van presets lists them)")
    ] = None,
    config: Annotated[
        str | None, typer.Option("--config", "-c", help="Agent config file (YAML/TOML/JSON)")
    ] = None,
    host: Annotated[str, typer.Option(help="Interface to bind (0.0.0.0: all)")] = "127.0.0.1",
    port: Annotated[
        int | None,
        typer.Option(
            help="TCP port (0: any free port). Default: 8000, 8080 for webrtc, "
            "8765 for websocket and telephony."
        ),
    ] = None,
    stt: Annotated[str | None, typer.Option(help="Cascade STT spec, e.g. faster_whisper")] = None,
    llm: Annotated[str | None, typer.Option(help="Cascade LLM spec, e.g. ollama/qwen3")] = None,
    tts: Annotated[str | None, typer.Option(help="Cascade TTS spec, e.g. kokoro")] = None,
    vad: Annotated[str | None, typer.Option(help="Cascade VAD spec, e.g. silero")] = None,
    turn_detector: Annotated[
        str | None, typer.Option("--turn", help="Cascade turn detector spec")
    ] = None,
    name: Annotated[
        str | None, typer.Option("--name", "-n", help="Model name of a single engine")
    ] = None,
    instructions: Annotated[str | None, typer.Option(help="Default session instructions")] = None,
    voice: Annotated[str | None, typer.Option(help="Default voice")] = None,
    language: Annotated[str | None, typer.Option(help="Default input language")] = None,
    api_key: Annotated[
        list[str] | None,
        typer.Option(
            "--api-key",
            envvar="VAN_SERVER_API_KEY",
            help="Require this API key (repeatable): a bearer token for openai-realtime, "
            "websocket and webrtc clients; ?key= of the telephony answer webhook. Default: "
            "none on loopback; beyond it openai-realtime refuses to start and websocket/"
            "webrtc/telephony generate one (see --insecure).",
        ),
    ] = None,
    public_url: Annotated[
        str | None,
        typer.Option(
            "--public-url",
            help="Telephony: the public wss:// (or https://) URL of this server, without "
            "the path. Twilio and Plivo: check the carrier's request signature on the "
            "answer webhook and every media stream. Also the stream URL of the served "
            "markup (default: from the webhook request's Host).",
        ),
    ] = None,
    stream_secret: Annotated[
        str | None,
        typer.Option(
            "--stream-secret",
            envvar="VAN_TELEPHONY_SECRET",
            help="Telephony: the secret of the per-call stream tokens (default: generated "
            "for this run; set it to share it with your own webhook).",
            show_default=False,
        ),
    ] = None,
    vonage_signature_secret: Annotated[
        str | None,
        typer.Option(
            "--vonage-signature-secret",
            envvar="VONAGE_SIGNATURE_SECRET",
            help="Vonage: your account's signature secret; the JWT of the answer webhook "
            "and of every media stream is then checked.",
            show_default=False,
        ),
    ] = None,
    allowed_origin: Annotated[
        list[str] | None,
        typer.Option(
            "--allowed-origin",
            envvar="VAN_ALLOWED_ORIGINS",
            help="Browser origin allowed to connect, e.g. https://app.example.com or "
            "https://*.example.com (repeatable; * allows any). Always allowed: this "
            "machine's pages (http://localhost:*) and clients sending no Origin.",
        ),
    ] = None,
    insecure: Annotated[
        bool,
        typer.Option(
            "--insecure",
            help="Allow listening beyond this machine without authentication (a firewall "
            "or an authenticating proxy protects the port): no API key is required or "
            "generated for websocket and webrtc.",
        ),
    ] = False,
    max_sessions: Annotated[
        int,
        typer.Option(
            min=0, help="Maximum concurrent sessions per process, then busy (0: no limit)"
        ),
    ] = _DEFAULT_MAX_SESSIONS,
    prewarm: Annotated[
        int,
        typer.Option(min=0, help="Prewarmed engines (open connections) kept ready per model"),
    ] = 0,
    engine_per_session: Annotated[
        bool,
        typer.Option(
            "--engine-per-session/--shared-engine",
            help="One engine instance per session (isolation) instead of one per process",
        ),
    ] = False,
    preconnect: Annotated[
        bool,
        typer.Option(
            "--preconnect/--no-preconnect",
            help="Prewarmed items include an open engine connection",
        ),
    ] = True,
    prewarm_max_idle: Annotated[
        float,
        typer.Option(min=0.0, help="Recycle prewarmed connections after N seconds (0: never)"),
    ] = 300.0,
    workers: Annotated[
        int,
        typer.Option(
            min=1,
            help="Worker processes sharing the port (SO_REUSEPORT; Linux, "
            "macOS). Windows: one process.",
        ),
    ] = 1,
    drain_timeout: Annotated[
        float,
        typer.Option(min=0.0, help="On SIGTERM/Ctrl+C, let live sessions finish for N s"),
    ] = 30.0,
    any_model: Annotated[
        bool | None,
        typer.Option(
            "--any-model/--strict-model",
            help="Serve the default engine for unknown ?model= names. Default: only when "
            "one model is served.",
        ),
    ] = None,
    max_session_duration: Annotated[
        float,
        typer.Option(min=0.0, help="Close sessions after this many seconds (0: no limit)"),
    ] = _DEFAULT_MAX_SESSION_DURATION,
    idle_timeout: Annotated[
        float,
        typer.Option(
            min=0.0,
            help="Close sessions after this many seconds without client messages (0: never)",
        ),
    ] = _DEFAULT_IDLE_TIMEOUT,
    warmup: Annotated[
        bool, typer.Option("--warmup/--no-warmup", help="Warm engines up before serving")
    ] = True,
    log_format: Annotated[
        str, typer.Option("--log-format", help="Log format: text or json (one object per line)")
    ] = "text",
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Serve voice agents: OpenAI Realtime, WebSocket, WebRTC or telephony media streams.

    Every protocol answers GET /health, /ready and /metrics on its port.

    Examples:

        van serve --engine mock

        van serve --engine agent.yaml --host 0.0.0.0 --api-key "$KEY" --prewarm 2

        van serve -p websocket --preset local-cpu --max-sessions 8 --workers 4

        van serve -p websocket --allowed-origin https://app.example.com

        van serve -p twilio --config agent.yaml --host 0.0.0.0 --port 8765
    """
    if ctx.invoked_subcommand is not None:
        return
    proto = protocol.strip().lower()
    fmt = log_format.strip().lower()
    if fmt not in ("text", "json"):
        console.print("[red]error:[/red] --log-format must be text or json")
        raise typer.Exit(2)
    options = ServeOptions(
        protocol=proto,
        host=host,
        port=_DEFAULT_PORTS.get(proto, 8765) if port is None else port,
        api_keys=[k for k in (api_key or []) if k],
        allowed_origins=[o for o in (allowed_origin or []) if o.strip()],
        insecure=insecure,
        public_url=public_url.strip() if public_url and public_url.strip() else None,
        stream_secret=stream_secret or None,
        signature_secret=vonage_signature_secret or None,
        max_sessions=max_sessions or None,
        accept_any_model=any_model,
        warmup=warmup,
        max_session_duration=max_session_duration or None,
        idle_timeout=idle_timeout or None,
        prewarm=prewarm,
        engine_per_session=engine_per_session,
        preconnect=preconnect,
        prewarm_max_idle=prewarm_max_idle or None,
        workers=workers,
        drain_timeout=drain_timeout,
        log_format=fmt,
        verbose=verbose,
    )
    sources = SourceOptions(
        engines=list(engine or []), preset=preset, config=config, stt=stt, llm=llm, tts=tts,
        vad=vad, turn_detector=turn_detector, name=name, instructions=instructions,
        voice=voice, language=language,
    )  # fmt: skip
    code = run(sources, options)
    if code:
        raise typer.Exit(code)
