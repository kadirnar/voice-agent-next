"""``van`` — the voice-agent-next command line interface."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .. import __version__

app = typer.Typer(
    name="van",
    help="voice-agent-next: real-time speech-to-speech voice agents.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
console = Console()


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"voice-agent-next {__version__}")


@app.command()
def providers(
    kind: Annotated[
        str | None, typer.Option("--kind", "-k", help="stt|tts|llm|vad|turn|engine")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output")] = False,
) -> None:
    """List registered providers and whether they are ready to use."""
    from ..registry import KINDS, list_providers

    if kind is not None and kind not in KINDS:
        raise typer.BadParameter(f"kind must be one of {', '.join(KINDS)}")
    specs = list_providers(kind)
    rows: list[dict[str, Any]] = []
    for s in specs:
        missing_deps = s.missing_dependencies()
        missing_env = s.missing_env()
        if not s.supports_platform():
            status = f"unsupported on {sys.platform}"
        elif missing_deps:
            status = (
                f"pip install 'voice-agent-next[{s.extra}]'"
                if s.extra
                else f"missing {missing_deps}"
            )
        elif missing_env:
            status = f"set {' or '.join(missing_env)}"
        else:
            status = "ready"
        rows.append(
            {
                "kind": s.kind,
                "name": s.name,
                "where": "local" if s.local else "cloud",
                "default_model": s.default_model or "",
                "status": status,
                "description": s.description,
            }
        )
    if as_json:
        typer.echo(json.dumps(rows, indent=2))  # plain stdout: never colorized
        return
    table = Table(title=f"voice-agent-next providers ({len(rows)})")
    for col in ("kind", "name", "where", "default_model", "status", "description"):
        table.add_column(col, overflow="fold")
    for r in rows:
        style = "green" if r["status"] == "ready" else "yellow"
        cells = [escape(str(r[c])) for c in ("kind", "name", "where", "default_model")]
        table.add_row(*cells, f"[{style}]{escape(r['status'])}[/{style}]", escape(r["description"]))
    console.print(table)


@app.command()
def doctor(
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output")] = False,
    strict: Annotated[
        bool, typer.Option("--strict", help="Exit with 1 on warnings too, not only failures")
    ] = False,
    only: Annotated[
        list[str] | None,
        typer.Option(
            "--only",
            help="Run only these sections (repeatable): system, audio, hardware, presets, "
            "models, network, mic, echo, latency",
        ),
    ] = None,
    network: Annotated[
        bool,
        typer.Option(
            "--network",
            help="Probe the cloud endpoints of the providers whose API key is set (DNS, "
            "connect, TLS; no API call)",
        ),
    ] = False,
    endpoint: Annotated[
        list[str] | None,
        typer.Option(
            "--endpoint", help="Extra URL or host to probe (repeatable; implies --network)"
        ),
    ] = None,
    mic: Annotated[
        bool, typer.Option("--mic", help="Record the microphone and judge its level")
    ] = False,
    echo: Annotated[
        bool,
        typer.Option("--echo", help="Play a chirp: measure echo delay and loss, recommend AEC"),
    ] = False,
    latency: Annotated[
        bool, typer.Option("--latency", help="Play clicks: measure the loopback latency")
    ] = False,
    duration: Annotated[float, typer.Option(help="Seconds of the --mic recording")] = 5.0,
    input_device: Annotated[
        str | None, typer.Option(help="Microphone for --mic/--echo/--latency (index or name)")
    ] = None,
    output_device: Annotated[
        str | None, typer.Option(help="Speakers for --echo/--latency (index or name)")
    ] = None,
    timeout: Annotated[float, typer.Option(help="Network probe timeout (seconds)")] = 5.0,
) -> None:
    """Diagnose the environment: Python, audio host APIs and devices, GPUs, presets, models.

    Opt-in: --network (endpoint reachability), --mic (level meter), --echo (echo delay and
    loss), --latency (loopback latency). Exit code 1 when a check fails (--strict: or warns).
    """
    from .doctor import run_doctor

    run_doctor(
        console=console,
        as_json=as_json,
        strict=strict,
        only=only or [],
        network=network,
        endpoints=endpoint or [],
        mic=mic,
        echo=echo,
        latency=latency,
        duration=duration,
        input_device=input_device,
        output_device=output_device,
        timeout=timeout,
    )


@app.command()
def devices(
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output")] = False,
) -> None:
    """List audio devices: index, host API, channels, default rate, system defaults."""
    from ..errors import VoiceAgentError
    from ..transports.local import describe_audio_system

    try:
        info = describe_audio_system()
    except VoiceAgentError as exc:  # no sounddevice / PortAudio, or PortAudio failed
        console.print(f"[red]{escape(str(exc))}[/red]", highlight=False)
        raise typer.Exit(1) from None
    if as_json:
        typer.echo(json.dumps([dataclasses.asdict(d) for d in info.devices], indent=2))
        return
    table = Table(title=f"audio devices ({escape(info.portaudio_release)})")
    for col in ("#", "name", "host API", "in", "out", "rate", "default"):
        table.add_column(
            col, overflow="fold", justify="right" if col in ("#", "in", "out") else "left"
        )
    for d in info.devices:
        default = " + ".join(
            kind for kind, flag in (("input", d.is_default_input), ("output", d.is_default_output))
            if flag
        )  # fmt: skip
        table.add_row(
            str(d.index), escape(d.name), escape(d.hostapi), str(d.max_input_channels),
            str(d.max_output_channels), f"{d.default_samplerate:g}", default,
            style="bold" if default else None,
        )  # fmt: skip
    console.print(table)
    console.print(
        "[dim]select with LocalAudioTransport(input_device=<index or name>) or "
        "`transport: {type: local, input_device: ...}`; check the setup with `van doctor`[/dim]"
    )


@app.command()
def presets(
    name: Annotated[str | None, typer.Argument(help="Show one preset in detail")] = None,
    transport: Annotated[
        str, typer.Option(help="Transport to check for (local needs the audio extra)")
    ] = "local",
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output")] = False,
) -> None:
    """List presets and whether they are ready here (extras, API keys, platform, GPU, Ollama)."""
    from .. import presets as presets_mod
    from ..errors import ConfigurationError

    env = presets_mod.current_environment()
    try:
        chosen = [presets_mod.get_preset(name)] if name else presets_mod.list_presets()
    except ConfigurationError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]", highlight=False)
        raise typer.Exit(2) from None
    results = [presets_mod.check_preset(p, transport=transport, env=env) for p in chosen]
    if as_json:
        rows = [
            {
                "name": p.name,
                "summary": p.summary,
                "where": p.where,
                "platforms": list(p.platforms),
                "accelerator": p.accelerator,
                "extras": list(p.extras),
                "env": [list(group) for group in p.env_vars],
                "config": dict(p.config),
                "ready": r.ready,
                "problems": [dataclasses.asdict(problem) for problem in r.problems],
                "fixes": r.fixes(),
                "notes": list(r.notes),
            }
            for p, r in zip(chosen, results, strict=True)
        ]
        typer.echo(json.dumps(rows if name is None else rows[0], indent=2))
        return
    if name is not None:
        _print_preset(chosen[0], results[0])
        return
    table = Table(title="voice-agent-next presets")
    for col in ("preset", "where", "stack", "status"):
        table.add_column(col, overflow="fold")
    for p, r in zip(chosen, results, strict=True):
        status = "[green]ready[/green]" if r.ready else f"[yellow]{escape(r.summary())}[/yellow]"
        table.add_row(f"[bold]{p.name}[/bold]", p.where, escape(p.stack()), status)
    console.print(table)
    ready = [r.name for r in results if r.ready]
    if ready:
        console.print(f"`van run` picks [bold]{ready[0]}[/bold]; or: van run --preset <name>")
    else:
        console.print("[yellow]no preset is ready here[/yellow]")
    console.print("[dim]details and fixes: van presets <name>[/dim]")


def _print_preset(preset: Any, result: Any) -> None:
    import yaml

    console.print(f"[bold]{preset.name}[/bold]: {escape(preset.summary)}")
    where = f"{preset.where}; platforms: {', '.join(preset.platforms)}"
    if preset.accelerator:
        where += f"; built for: {preset.accelerator}"
    console.print(escape(where))
    if preset.extras:
        console.print(f"extras: {', '.join(preset.extras)}")
    if preset.env_vars:
        console.print("API keys: " + ", ".join(" or ".join(g) for g in preset.env_vars))
    console.print(f"\n[dim]{escape(preset.rationale)}[/dim]\n")
    console.print(escape(f"# config (use it with `extends: {preset.name}`)"))
    console.print(escape(yaml.safe_dump(dict(preset.config), sort_keys=False).rstrip()))
    console.print()
    _print_readiness(result)


def _print_readiness(result: Any) -> None:
    for note in result.notes:
        console.print(f"[dim]{escape(note)}[/dim]")
    if result.ready:
        console.print(f"[green]{escape(result.explain())}[/green]")
    else:
        console.print(f"[yellow]{escape(result.explain())}[/yellow]", highlight=False)


def _run_config(
    *,
    config: Path | None,
    preset: str | None,
    components: dict[str, str | None],
    instructions: str | None,
    greeting: str | None,
    transport: str,
    skip_checks: bool,
) -> Any:
    """The :class:`AppConfig` ``van run`` runs: preset < config file < flags, checked."""
    from .. import presets as presets_mod
    from ..config import AppConfig, layer_config, merge_config
    from ..errors import ConfigurationError

    overrides = {k: v for k, v in components.items() if v is not None}
    try:  # read raw, merge (preset < file < flags), validate once at the end
        raw = layer_config(preset=preset, file=config, overrides=overrides)
        extends = raw.pop("extends", None)
        chosen = presets_mod.get_preset(extends) if extends else None
    except ConfigurationError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]", highlight=False)
        raise typer.Exit(2) from None
    if config is None or transport != "local":
        raw["transport"] = {"type": transport}
    transport_type = str((raw.get("transport") or {"type": "local"}).get("type", "local"))
    env = presets_mod.current_environment()
    if chosen is None and config is None and not overrides:
        picked, _ = presets_mod.pick_preset(transport=transport_type, env=env)
        if picked is None:
            console.print(
                "[yellow]no preset is ready on this machine (`van presets` shows what each "
                "needs); using the mock engine[/yellow]"
            )
            raw["engine"] = "mock"
        else:
            chosen = presets_mod.get_preset(picked.name)
            console.print(
                f"no --preset or --config given: using preset [bold]{picked.name}[/bold] "
                f"({escape(chosen.stack())}), the first ready one of "
                f"{', '.join(presets_mod.AUTO_ORDER)}. Choose with --preset; see `van presets`."
            )
            raw = merge_config(dict(chosen.config), raw)
    if chosen is not None and not skip_checks:
        result = presets_mod.check_config(
            raw,
            name=chosen.name,
            platforms=chosen.platforms,
            accelerator=chosen.accelerator,
            transport=transport_type,
            env=env,
        )
        if not result.ready:
            _print_readiness(result)
            console.print("[dim](run anyway with --skip-checks)[/dim]")
            raise typer.Exit(1)
        for note in result.notes:
            console.print(f"[dim]{escape(note)}[/dim]")
        raw = result.config
    if chosen is not None:
        raw["extends"] = chosen.name
    try:
        cfg = AppConfig.model_validate(raw)
        if instructions:
            cfg.agent.instructions = instructions
        if greeting:
            cfg.agent.greeting = greeting
        if cfg.engine is None and cfg.llm is None:
            cfg.engine = "mock"
            console.print("[yellow]no engine configured; using the mock engine[/yellow]")
        cfg.validate_components()
    except (ConfigurationError, ValueError) as exc:  # pydantic's ValidationError included
        console.print(f"[red]invalid config: {escape(str(exc))}[/red]", highlight=False)
        raise typer.Exit(2) from None
    return cfg


@app.command()
def run(
    preset: Annotated[
        str | None,
        typer.Option("--preset", "-p", help="Named configuration (`van presets` lists them)"),
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="YAML/TOML/JSON config")
    ] = None,
    engine: Annotated[
        str | None, typer.Option(help="Native S2S engine, e.g. openai/gpt-realtime")
    ] = None,
    stt: Annotated[str | None, typer.Option(help="Cascade STT, e.g. deepgram/nova-3")] = None,
    llm: Annotated[str | None, typer.Option(help="Cascade LLM, e.g. openai/gpt-4.1-mini")] = None,
    tts: Annotated[str | None, typer.Option(help="Cascade TTS, e.g. cartesia/sonic-2")] = None,
    vad: Annotated[str | None, typer.Option(help="VAD, e.g. silero")] = None,
    turn_detector: Annotated[str | None, typer.Option("--turn", help="Turn detector")] = None,
    instructions: Annotated[str | None, typer.Option(help="System prompt")] = None,
    greeting: Annotated[str | None, typer.Option(help="Spoken greeting")] = None,
    transport: Annotated[
        str, typer.Option(help="local|file|websocket|webrtc|twilio|telnyx|vonage|plivo")
    ] = "local",
    input_wav: Annotated[
        Path | None, typer.Option("--input", help="Input WAV (file transport)")
    ] = None,
    output_wav: Annotated[
        Path | None, typer.Option("--output", help="Output WAV (file transport)")
    ] = None,
    skip_checks: Annotated[
        bool, typer.Option("--skip-checks", help="Run a preset without the readiness checks")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run a voice agent from a preset, a config file and/or command-line flags.

    Without any of them, the best preset that is ready on this machine is used.
    """
    import logging

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if transport == "file" and input_wav is None:
        raise typer.BadParameter("--input is required with --transport file")
    cfg = _run_config(
        config=config,
        preset=preset,
        components={
            "engine": engine,
            "stt": stt,
            "llm": llm,
            "tts": tts,
            "vad": vad,
            "turn_detector": turn_detector,
        },
        instructions=instructions,
        greeting=greeting,
        transport=transport,
        skip_checks=skip_checks,
    )
    if cfg.transport.get("type") == "file":
        if input_wav is None:
            raise typer.BadParameter("--input is required with --transport file")
        cfg.transport.update(
            input_path=str(input_wav), output_path=str(output_wav) if output_wav else None
        )
    _run_session(cfg)


def _run_session(cfg: Any) -> None:
    """Build the session, agent and transport of ``cfg`` and run until Ctrl+C."""
    from ..app import build_agent, build_session
    from ..transports import create_transport

    async def _main() -> None:
        session = build_session(cfg)
        agent = build_agent(cfg)
        _attach_console_logging(session)
        tr = create_transport(cfg.transport)
        try:
            await session.run(agent, tr)
        finally:
            await session.aclose()

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        console.print("\n[dim]bye[/dim]")


@app.command()
def demo(
    turns: Annotated[int, typer.Option(help="Number of simulated user turns")] = 3,
) -> None:
    """Offline demo: a simulated user talks to the mock speech-to-speech engine."""

    async def _demo() -> None:
        from ..audio import AudioFrame
        from ..providers.mock import MockEngine, synth_speech
        from ..session import Agent, AgentSession
        from ..transports import LoopbackTransport

        user_lines = [f"this is simulated user turn number {i + 1}" for i in range(turns)]
        engine = MockEngine(transcripts=user_lines, response_delay=0.25, realtime_factor=1.0)
        session = AgentSession(engine)
        _attach_console_logging(session)
        transport = LoopbackTransport(realtime_playout=True)
        await session.start(Agent(greeting="Hi, I am the demo agent."), transport)
        await asyncio.sleep(2.5)
        for _ in user_lines:
            await transport.play_user_audio(synth_speech(1.2, 16_000))
            await transport.play_user_audio(AudioFrame.silence(5.0, 16_000))
        await session.aclose()

    asyncio.run(_demo())


def _attach_console_logging(session: Any) -> None:
    from ..metrics import TurnMetrics

    @session.on("user_transcript")
    def _user(ev: Any) -> None:
        if ev.is_final:
            console.print(f"[bold cyan]user[/bold cyan]  {ev.text}")

    @session.on("agent_transcript")
    def _agent(ev: Any) -> None:
        console.print(f"[bold magenta]agent[/bold magenta] {ev.delta.strip()}")

    @session.on("interrupted")
    def _interrupted(ev: Any) -> None:
        console.print(f"[yellow]interrupted after {ev.played:.2f}s[/yellow]")

    @session.on("metrics")
    def _metrics(m: Any) -> None:
        if isinstance(m, TurnMetrics) and m.voice_to_voice is not None:
            console.print(f"[dim]voice-to-voice latency: {m.voice_to_voice * 1000:.0f} ms[/dim]")

    @session.on("error")
    def _error(ev: Any) -> None:
        console.print(f"[red]error: {ev.error}[/red]")


def _register_command_groups() -> None:
    """Optional command groups live in ``cli/<name>.py`` modules exposing a Typer ``app``."""
    for name in ("bench", "models", "serve"):
        module_name = f"{__package__}.{name}"
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                continue
            raise
        app.add_typer(module.app, name=name)


_register_command_groups()


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
