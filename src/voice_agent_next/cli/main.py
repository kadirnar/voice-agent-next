"""``van`` — the voice-agent-next command line interface."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import json
import os
import platform
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
def doctor() -> None:
    """Check the environment: Python, audio, ML runtimes, GPUs, API keys."""
    from ..utils.deps import is_installed

    table = Table(title="voice-agent-next doctor", show_header=False)
    table.add_column("check")
    table.add_column("result", overflow="fold")
    table.add_row("voice-agent-next", __version__)
    table.add_row("python", f"{platform.python_version()} ({sys.executable})")
    table.add_row("platform", f"{platform.system()} {platform.release()} {platform.machine()}")
    for mod in ("numpy", "soxr", "sounddevice", "onnxruntime", "torch", "mlx", "aiortc"):
        if is_installed(mod):
            try:
                ver = getattr(__import__(mod), "__version__", "installed")
            except Exception as exc:  # broken native libs (e.g. missing PortAudio)
                ver = f"installed but failed to import: {exc}"
            table.add_row(mod, str(ver))
        else:
            table.add_row(mod, "[dim]not installed[/dim]")
    if is_installed("onnxruntime"):
        try:
            import onnxruntime as ort

            table.add_row("onnxruntime providers", ", ".join(ort.get_available_providers()))
        except Exception as exc:
            table.add_row("onnxruntime providers", f"error: {exc}")
    if is_installed("torch"):
        try:
            import torch

            accel = []
            if torch.cuda.is_available():
                accel.append(f"cuda ({torch.cuda.get_device_name(0)})")
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                accel.append("mps")
            table.add_row("torch accelerators", ", ".join(accel) or "cpu only")
        except Exception as exc:
            table.add_row("torch accelerators", f"error: {exc}")
    _hardware_doctor_rows(table)
    _audio_doctor_rows(table)
    keys = [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
        "DEEPGRAM_API_KEY", "ASSEMBLYAI_API_KEY", "CARTESIA_API_KEY", "ELEVENLABS_API_KEY",
        "GROQ_API_KEY", "AWS_ACCESS_KEY_ID", "AZURE_OPENAI_API_KEY", "XAI_API_KEY",
    ]  # fmt: skip
    present = [k for k in keys if os.environ.get(k)]
    table.add_row("API keys set", ", ".join(present) or "[dim]none[/dim]")
    console.print(table)


def _hardware_doctor_rows(table: Table) -> None:
    """GPUs, CUDA libraries and where local models run on `device="auto"` (docs/hardware.md)."""
    from .. import hardware

    try:
        rows = hardware.report()
    except Exception as exc:  # report() does not raise; belt and braces for the doctor
        rows = [("hardware", f"error: {exc}")]
    for check, result in rows:
        style = "yellow" if "to use the GPU:" in result or result.startswith("error") else ""
        table.add_row(check, f"[{style}]{escape(result)}[/{style}]" if style else escape(result))


def _audio_doctor_rows(table: Table) -> None:
    """PortAudio version, host APIs, default devices and setup hints for `van doctor`."""
    from ..errors import MissingDependencyError, VoiceAgentError
    from ..transports.local import describe_audio_system

    try:
        info = describe_audio_system()
    except MissingDependencyError as exc:  # no sounddevice, or no PortAudio library
        table.add_row("audio", f"[yellow]{escape(str(exc))}[/yellow]")
        return
    except VoiceAgentError as exc:
        table.add_row("audio", f"[red]error: {escape(str(exc))}[/red]")
        return
    table.add_row("portaudio", escape(info.portaudio_version))
    table.add_row("audio host APIs", escape(", ".join(info.hostapis)) or "[dim]none[/dim]")
    for label, device in (("default input", info.default_input),
                          ("default output", info.default_output)):  # fmt: skip
        if device is None:
            table.add_row(label, "[yellow]none[/yellow]")
        else:
            table.add_row(label, escape(f"{device}, {device.default_samplerate:g} Hz"))
    n_in = sum(1 for d in info.devices if d.max_input_channels > 0)
    n_out = sum(1 for d in info.devices if d.max_output_channels > 0)
    table.add_row("audio devices", f"{n_in} input, {n_out} output (details: `van devices`)")
    for hint in info.hints():
        table.add_row("audio hint", f"[yellow]{escape(hint)}[/yellow]")


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
def run(
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
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run a voice agent from a config file and/or command-line flags."""
    import logging

    from ..app import build_agent, build_session
    from ..config import AppConfig, load_config
    from ..transports import create_transport

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(config) if config else AppConfig()
    overrides = {
        "engine": engine,
        "stt": stt,
        "llm": llm,
        "tts": tts,
        "vad": vad,
        "turn_detector": turn_detector,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    if instructions:
        cfg.agent.instructions = instructions
    if greeting:
        cfg.agent.greeting = greeting
    if config is None or transport != "local":
        cfg.transport = {"type": transport}
    if cfg.transport.get("type") == "file":
        if input_wav is None:
            raise typer.BadParameter("--input is required with --transport file")
        cfg.transport.update(
            input_path=str(input_wav), output_path=str(output_wav) if output_wav else None
        )
    if cfg.engine is None and cfg.llm is None:
        cfg.engine = "mock"
        console.print("[yellow]no engine configured; using the mock engine[/yellow]")
    cfg.validate_components()

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
