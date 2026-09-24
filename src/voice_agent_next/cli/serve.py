"""``van serve`` — serve engines over the network.

``van serve --protocol openai-realtime --engine <spec or config>`` puts any engine behind an
OpenAI-Realtime-compatible WebSocket endpoint (``ws://host:port/v1/realtime``); see
``docs/deploy/realtime-server.md``.

The command is built in two steps so that it can grow (worker pools, prewarm and health
policies, other protocols): :func:`build_models` turns the engine options into served
models, :func:`build_server` turns the models and :class:`ServeOptions` into a server.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape

app = typer.Typer(
    name="serve",
    help="Serve engines over the network (OpenAI Realtime protocol at /v1/realtime).",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
console = Console(stderr=True)

PROTOCOLS = ("openai-realtime",)
_CONFIG_SUFFIXES = (".yaml", ".yml", ".toml", ".json")
_NAMED = re.compile(r"^([A-Za-z0-9][\w.:\-]*)=(.+)$", re.S)


@dataclass
class ServeOptions:
    """Server settings shared by every protocol."""

    protocol: str = "openai-realtime"
    host: str = "127.0.0.1"
    port: int = 8000
    api_keys: list[str] = field(default_factory=list)
    max_sessions: int | None = None
    accept_any_model: bool | None = None
    warmup: bool = True
    max_session_duration: float | None = None


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


def _load_source(source: str) -> Any:
    """A config file becomes a :class:`RealtimeModel`; anything else an engine spec."""
    from ..bench.system import parse_component_spec
    from ..errors import ConfigurationError
    from ..server import engine_from_config

    path = Path(source)
    if path.suffix.lower() in _CONFIG_SUFFIXES:
        if not path.is_file():
            raise ConfigurationError(f"config file not found: {path}")
        return engine_from_config(path)
    return parse_component_spec(source)


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
) -> dict[str, Any]:
    """Served models from CLI options.

    ``engines`` entries are ``[NAME=]SOURCE`` where SOURCE is a registry spec (``mock``,
    ``openai/gpt-realtime``), an inline mapping (``{provider: mock, response_delay: 0.2}``)
    or an agent config file (engine or cascade; its agent instructions/voice/language
    become session defaults). ``stt``/``llm``/``tts``/``vad``/``turn_detector`` build one
    more model, a cascade (named ``name`` or ``"cascade"``). Nothing given: the mock engine.
    """
    from ..bench.system import parse_component_spec
    from ..engines.cascade import CascadeEngine
    from ..errors import ConfigurationError
    from ..server import RealtimeModel

    sources: list[tuple[str, Any]] = []
    for value in engines or []:
        match = _NAMED.match(value.strip())
        model_name, source = (match.group(1), match.group(2)) if match else (None, value)
        sources.append((model_name or _source_name(source), _load_source(source)))
    cascade = {"stt": stt, "llm": llm, "tts": tts, "turn_detector": turn_detector}
    if any(v is not None for v in cascade.values()):
        if llm is None or tts is None:
            raise ConfigurationError("a cascade needs at least --llm and --tts (and --stt)")
        engine = CascadeEngine(
            stt=parse_component_spec(stt), llm=parse_component_spec(llm),
            tts=parse_component_spec(tts), vad=parse_component_spec(vad),
            turn_detector=parse_component_spec(turn_detector),
        )  # fmt: skip
        sources.append((name or "cascade", RealtimeModel(engine, owned=True)))
    elif vad is not None:
        raise ConfigurationError("--vad belongs to a cascade (--stt/--llm/--tts)")
    if not sources:
        sources.append((name or "mock", "mock"))
    elif name is not None and len(sources) == 1 and engines and not _NAMED.match(engines[0]):
        sources[0] = (name, sources[0][1])  # --name renames a single --engine
    models: dict[str, Any] = {}
    for model_name, source in sources:
        if model_name in models:
            raise ConfigurationError(f"model name {model_name!r} is used twice; use NAME=SOURCE")
        if not isinstance(source, RealtimeModel):
            source = RealtimeModel(source)
        if instructions is not None:
            source.instructions = instructions
        if voice is not None:
            source.voice = voice
        if language is not None:
            source.language = language
        models[model_name] = source
    return models


def build_server(models: dict[str, Any], options: ServeOptions) -> Any:
    """The server for ``options.protocol``."""
    from ..errors import ConfigurationError

    if options.protocol == "openai-realtime":
        from ..server import RealtimeServer

        return RealtimeServer(
            models=models,
            host=options.host,
            port=options.port,
            api_keys=options.api_keys or None,
            max_sessions=options.max_sessions,
            accept_any_model=options.accept_any_model,
            warmup=options.warmup,
            max_session_duration=options.max_session_duration,
        )
    raise ConfigurationError(
        f"unknown protocol {options.protocol!r}; expected one of {', '.join(PROTOCOLS)}"
    )


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
            "cascade). NAME=SOURCE sets the model name; repeat to serve several models.",
        ),
    ] = None,
    protocol: Annotated[
        str, typer.Option("--protocol", "-p", help="Wire protocol: openai-realtime")
    ] = "openai-realtime",
    host: Annotated[str, typer.Option(help="Interface to bind (0.0.0.0: all)")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="TCP port (0: any free port)")] = 8000,
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
            help="Require this bearer token (repeatable). Default: no authentication.",
        ),
    ] = None,
    max_sessions: Annotated[
        int | None, typer.Option(min=1, help="Maximum concurrent sessions")
    ] = None,
    any_model: Annotated[
        bool | None,
        typer.Option(
            "--any-model/--strict-model",
            help="Serve the default engine for unknown ?model= names. Default: only when "
            "one model is served.",
        ),
    ] = None,
    max_session_duration: Annotated[
        float | None, typer.Option(min=1.0, help="Close sessions after this many seconds")
    ] = None,
    warmup: Annotated[
        bool, typer.Option("--warmup/--no-warmup", help="Warm engines up before serving")
    ] = True,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Serve engines behind the OpenAI Realtime protocol.

    Examples:

        van serve --engine mock

        van serve --engine agent.yaml --host 0.0.0.0 --port 8000 --api-key "$KEY"

        van serve --stt faster_whisper --llm ollama/qwen3 --tts kokoro --vad silero

        van serve -e local=local.yaml -e cloud=openai/gpt-realtime
    """
    if ctx.invoked_subcommand is not None:
        return
    from ..errors import VoiceAgentError

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not verbose:  # one line per handshake and health probe otherwise
        logging.getLogger("websockets").setLevel(logging.WARNING)
    options = ServeOptions(
        protocol=protocol.strip().lower(),
        host=host,
        port=port,
        api_keys=[k for k in (api_key or []) if k],
        max_sessions=max_sessions,
        accept_any_model=any_model,
        warmup=warmup,
        max_session_duration=max_session_duration,
    )
    try:
        models = build_models(
            engine, stt=stt, llm=llm, tts=tts, vad=vad, turn_detector=turn_detector,
            name=name, instructions=instructions, voice=voice, language=language,
        )  # fmt: skip
        server = build_server(models, options)
    except (VoiceAgentError, ValueError) as exc:
        console.print(f"[red]error:[/red] {escape(str(exc))}")
        raise typer.Exit(2) from exc
    if not engine and llm is None:
        console.print("[yellow]no engine given; serving the mock engine[/yellow]")

    async def _serve() -> None:
        await server.start()
        auth = "bearer token" if options.api_keys else "no authentication"
        console.print(
            f"[bold]van serve[/bold] · {options.protocol} · "
            f"[bold]{escape(server.url)}/realtime[/bold] · models: "
            f"{escape(', '.join(server.models))} · {auth}",
            highlight=False,
        )
        await server.serve_forever()

    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        console.print("[dim]bye[/dim]")
    except OSError as exc:  # e.g. the port is in use
        console.print(f"[red]error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
