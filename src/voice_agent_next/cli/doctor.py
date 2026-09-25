"""``van doctor``: collect the diagnostics of :mod:`voice_agent_next.doctor` and print them."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .. import doctor as dx

_STYLE = {"ok": "green", "info": "dim", "warn": "yellow", "fail": "red", "skip": "dim"}
_TITLES = {
    "system": "System & Python",
    "audio": "Audio (PortAudio)",
    "hardware": "Hardware",
    "presets": "Presets",
    "models": "Models cache",
    "network": "Network",
    "mic": "Microphone level",
    "echo": "Echo test",
    "latency": "Loopback latency",
}


def audio_io() -> dx.AudioIO:
    """The device backend of the interactive checks (tests replace it)."""
    return dx.SoundDeviceIO()


def run_doctor(
    *,
    console: Console,
    as_json: bool,
    strict: bool,
    only: Sequence[str],
    network: bool,
    endpoints: Sequence[str],
    mic: bool,
    echo: bool,
    latency: bool,
    duration: float,
    input_device: str | None,
    output_device: str | None,
    timeout: float,
) -> None:
    unknown = [s for s in only if s not in dx.SECTIONS]
    if unknown:
        raise typer.BadParameter(
            f"unknown section(s) {', '.join(unknown)}; choose from {', '.join(dx.SECTIONS)}",
            param_hint="--only",
        )
    if duration <= 0:
        raise typer.BadParameter("must be > 0", param_hint="--duration")
    wanted = set(only or dx.DEFAULT_SECTIONS)
    wanted |= {
        name
        for name, flag in (
            ("network", network or bool(endpoints)),
            ("mic", mic),
            ("echo", echo),
            ("latency", latency),
        )
        if flag
    }
    live = not as_json and console.is_terminal
    report = dx.DoctorReport()

    def section(name: str, collect: Callable[[], list[dx.Check]]) -> None:
        if name not in wanted:
            return
        checks = dx.guarded(name, collect)
        report.extend(checks)
        if not as_json:
            _print_section(console, name, checks)

    section("system", dx.system_checks)
    section("audio", dx.audio_checks)
    section("hardware", dx.hardware_checks)
    section("presets", dx.preset_checks)
    section("models", dx.model_checks)
    section(
        "network",
        lambda: dx.network_checks(dx.cloud_endpoints(extra=endpoints), timeout=timeout),
    )
    interactive = [name for name in ("mic", "echo", "latency") if name in wanted]
    if interactive:
        devices: list[Any] = []

        def resolve() -> list[dx.Check]:
            devices.extend(dx.resolve_devices(_device(input_device), _device(output_device)))
            return []

        failed = dx.guarded(interactive[0], resolve)
        if failed:
            for name in interactive:
                report.extend([dx.Check(name, c.name, c.status, c.value) for c in failed])
                if not as_json:
                    _print_section(console, name, failed)
        else:
            _run_interactive(section, console, live, as_json, devices, duration)
    if as_json:
        typer.echo(json.dumps(report.to_dict(strict=strict), indent=2))
    else:
        _print_summary(console, report, strict)
    code = report.exit_code(strict=strict)
    if code:
        raise typer.Exit(code)


def _device(value: str | None) -> int | str | None:
    return int(value) if value is not None and value.strip().isdigit() else value


def _run_interactive(
    section: Callable[[str, Callable[[], list[dx.Check]]], None],
    console: Console,
    live: bool,
    as_json: bool,
    devices: list[Any],
    duration: float,
) -> None:
    mic_dev, spk_dev = devices
    io = audio_io()

    def mic() -> list[dx.Check]:
        if not as_json:
            console.print(
                f"[bold]mic[/bold]: recording {duration:g} s from {escape(str(mic_dev))}; "
                "speak normally, with pauses (nothing is saved)"
            )
        if not live:
            return dx.mic_check(io, device=mic_dev, seconds=duration)
        from rich.progress import BarColumn, Progress, TextColumn

        with Progress(
            TextColumn("level"),
            BarColumn(bar_width=40),
            TextColumn("{task.fields[db]:>6.1f} dBFS  peak {task.fields[peak]:>6.1f}"),
            TextColumn("{task.fields[t]:.1f} s"),
            console=console,
            transient=True,
        ) as bar:
            task = bar.add_task("level", total=70.0, db=-120.0, peak=-120.0, t=0.0)

            def on_level(t: float, db: float, peak: float) -> None:
                bar.update(task, completed=max(0.0, db + 70.0), db=db, peak=peak, t=t)

            return dx.mic_check(io, device=mic_dev, seconds=duration, on_level=on_level)

    def echo() -> list[dx.Check]:
        if not as_json:
            console.print(
                f"[bold]echo[/bold]: playing a 0.5 s chirp on {escape(str(spk_dev))} and "
                f"recording {escape(str(mic_dev))}; use your normal speaker volume"
            )
        return dx.echo_check(io, input_device=mic_dev, output_device=spk_dev)

    def latency() -> list[dx.Check]:
        if not as_json:
            console.print(
                f"[bold]latency[/bold]: playing 5 clicks on {escape(str(spk_dev))}, "
                f"listening on {escape(str(mic_dev))}"
            )
        return dx.latency_check(io, input_device=mic_dev, output_device=spk_dev)

    section("mic", mic)
    section("echo", echo)
    section("latency", latency)


def _print_section(console: Console, name: str, checks: list[dx.Check]) -> None:
    table = Table(title=_TITLES.get(name, name), title_justify="left", show_header=False)
    table.add_column("status", no_wrap=True)
    table.add_column("check")
    table.add_column("result", overflow="fold")
    for c in checks:
        style = _STYLE[c.status]
        result = escape(c.value)
        if c.hint:
            result += f"\n[dim]-> {escape(c.hint)}[/dim]"
        table.add_row(f"[{style}]{c.status}[/{style}]", escape(c.name), result)
    console.print(table)


def _print_summary(console: Console, report: dx.DoctorReport, strict: bool) -> None:
    counts = report.counts()
    parts = [f"[{_STYLE[k]}]{counts[k]} {k}[/{_STYLE[k]}]" for k in ("ok", "warn", "fail")]
    code = report.exit_code(strict=strict)
    console.print("summary: " + ", ".join(parts) + f" (exit code {code})")
    optional = [s for s in ("network", "mic", "echo", "latency") if s not in report.sections]
    if optional:
        flags = " ".join(f"--{s}" for s in optional)
        console.print(f"[dim]more checks (opt-in): van doctor {flags}; JSON: --json[/dim]")
