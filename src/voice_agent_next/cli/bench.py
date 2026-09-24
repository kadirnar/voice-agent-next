"""``van bench`` — benchmarks measured at the audio boundary (see ``benchmarks/README.md``)."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="bench",
    help="Benchmark engines on identical stimuli, measured on the call recording.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
console = Console()
err = Console(stderr=True)

_TABLE_METRICS = (
    ("v2v_ms", "voice-to-voice (recording)"),
    ("first_turn_v2v_ms", "first turn (cold start)"),
    ("session_v2v_ms", "session TurnMetrics v2v"),
    ("residual_ms", "residual (recording - session)"),
    ("eou_delay_ms", "end-of-turn delay"),
    ("response_ttfb_ms", "response TTFB"),
    ("llm_ttft_ms", "LLM TTFT"),
    ("tts_ttfb_ms", "TTS TTFB"),
    ("engine_ttfb_ms", "engine TTFB"),
    ("session_ready_ms", "session ready"),
)


@app.callback()
def _bench() -> None:
    """Benchmark suite: T1 latency (more tracks: see ROADMAP M6)."""


def _num(value: float | None) -> str:
    return "-" if value is None else f"{round(value) + 0.0:,.0f}"  # + 0.0: no "-0"


def _print_summary(results: object) -> None:
    from ..bench.results import RunResults

    assert isinstance(results, RunResults)
    s = results.summary
    table = Table(title=f"{s.track} · {s.system} · {s.dataset}", title_justify="left")
    for col in ("metric (ms)", "n", "mean", "p50 [95% CI]", "p90", "p95", "p99", "max"):
        table.add_column(col, justify="left" if col.startswith("metric") else "right")
    for key, label in _TABLE_METRICS:
        d = s.metrics.get(key)
        if d is None or not d.n:
            continue
        ci = d.ci95.get("p50")
        p50 = _num(d.p50) + (f" [{_num(ci[0])}, {_num(ci[1])}]" if ci and d.n > 1 else "")
        table.add_row(label, str(d.n), _num(d.mean), p50, _num(d.p90), _num(d.p95),
                      _num(d.p99), _num(d.max))  # fmt: skip
    console.print(table)
    rates = ", ".join(
        f"{k.removesuffix('_rate').replace('_', ' ')} {100 * v:.1f}%"
        for k, v in s.rates.items()
        if v is not None
    )
    console.print(f"rates: {rates or '-'}   turns measured: {s.counts.get('turns_measured', 0)}")
    if results.directory is not None:
        console.print(f"results: [bold]{results.directory}[/bold] (report.md, summary.json)")
    for note in results.manifest.notes:
        console.print(f"[yellow]note:[/yellow] {note}")


@app.command()
def latency(
    engine: Annotated[
        str | None,
        typer.Option(
            help="Native S2S engine spec: mock, openai/gpt-realtime, or an inline mapping "
            "like '{provider: mock, response_delay: 0.3}'"
        ),
    ] = None,
    stt: Annotated[str | None, typer.Option(help="Cascade STT spec, e.g. mock")] = None,
    llm: Annotated[str | None, typer.Option(help="Cascade LLM spec, e.g. mock")] = None,
    tts: Annotated[str | None, typer.Option(help="Cascade TTS spec, e.g. mock")] = None,
    vad: Annotated[str | None, typer.Option(help="Cascade VAD spec, e.g. energy")] = None,
    turn_detector: Annotated[
        str | None, typer.Option("--turn", help="Cascade turn detector spec")
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Agent config (YAML/TOML/JSON)")
    ] = None,
    scenario: Annotated[
        str, typer.Option("--scenario", "-s", help="Built-in scenario name or YAML file")
    ] = "latency-smoke",
    turns: Annotated[int, typer.Option("--turns", "-n", min=1, help="Turns per session")] = 20,
    sessions: Annotated[int, typer.Option(min=1, help="Separate sessions")] = 1,
    warmup_turns: Annotated[
        int, typer.Option(min=0, help="Leading turns per session reported as cold start only")
    ] = 1,
    out: Annotated[Path, typer.Option("--out", "-o", help="Results directory")] = Path(
        "bench-results"
    ),
    run_id: Annotated[str | None, typer.Option(help="Run directory name")] = None,
    label: Annotated[str | None, typer.Option(help="System label used in reports")] = None,
    dead_air: Annotated[float, typer.Option(min=0.01, help="Dead-air threshold in seconds")] = 2.0,
    reply_timeout: Annotated[
        float | None, typer.Option(min=0.1, help="Seconds without reply before a turn is missed")
    ] = None,
    reference_vad: Annotated[
        str,
        typer.Option(help="Reference VAD for agent onsets: rms, rms:<dBFS> or a VAD spec"),
    ] = "rms",
    audio: Annotated[
        bool, typer.Option("--audio/--no-audio", help="Save stereo recordings and labels")
    ] = True,
    engine_warmup: Annotated[
        bool, typer.Option("--engine-warmup/--no-engine-warmup", help="engine.warmup() first")
    ] = True,
    seed: Annotated[int, typer.Option(help="Bootstrap seed")] = 0,
    as_json: Annotated[bool, typer.Option("--json", help="Print summary.json to stdout")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """T1: voice-to-voice latency measured on the call recording.

    Examples:

        van bench latency --engine mock --turns 20 --out bench-results/

        van bench latency --stt mock --llm mock --tts mock --vad energy

        van bench latency --engine '{provider: mock, response_delay: 0.3}'
    """
    from ..bench.caller import TurnTiming
    from ..bench.onset import OnsetDetector, make_reference_vad
    from ..bench.stimuli import load_scenario
    from ..bench.system import BenchSystem, parse_component_spec
    from ..bench.tracks.latency import LatencyOptions, run_latency_benchmark
    from ..errors import VoiceAgentError

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        system = BenchSystem.from_options(
            config=config,
            engine=parse_component_spec(engine),
            stt=parse_component_spec(stt),
            llm=parse_component_spec(llm),
            tts=parse_component_spec(tts),
            vad=parse_component_spec(vad),
            turn_detector=parse_component_spec(turn_detector),
            label=label,
        )
        scn = load_scenario(scenario)
        detector = OnsetDetector(make_reference_vad(reference_vad))
    except (VoiceAgentError, ValueError) as exc:
        err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2) from exc
    options = LatencyOptions(
        turns=turns,
        sessions=sessions,
        warmup_turns=warmup_turns,
        dead_air_threshold=dead_air,
        reply_timeout=reply_timeout,
        save_audio=audio,
        warmup_engine=engine_warmup,
        seed=seed,
    )
    err.print(
        f"[bold]van bench latency[/bold] · {system.label} · scenario {scn.name} · "
        f"{turns} turn(s) x {sessions} session(s)"
    )

    def on_turn(session: int, turn: TurnTiming) -> None:
        if turn.reply_start is not None:
            heard = f"~{(turn.reply_start - turn.speech_end) * 1000:,.0f} ms"
        else:
            heard = "[red]no reply[/red]" if turn.stimulus.expect_reply else "-"
        err.print(
            f"  session {session + 1} turn {turn.index + 1:>3}/{turns} "
            f"{turn.stimulus.id:<12} {heard}"
        )

    try:
        results = asyncio.run(
            run_latency_benchmark(
                system, scn, options, out_dir=out, run_id=run_id, detector=detector,
                on_turn=on_turn,
            )
        )  # fmt: skip
    except VoiceAgentError as exc:
        err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc
    if as_json:
        typer.echo(json.dumps(results.summary.model_dump(mode="json"), indent=2))
    else:
        _print_summary(results)


@app.command()
def report(
    run_dir: Annotated[Path, typer.Argument(help="Run directory (contains summary.json)")],
    write: Annotated[bool, typer.Option("--write/--no-write", help="Rewrite report.md")] = True,
) -> None:
    """Re-render report.md from a run directory and print it."""
    from ..bench.results import REPORT_FILE, load_run
    from ..bench.tracks.latency import TRACK, render_latency_report

    try:
        results = load_run(run_dir)
    except (OSError, ValueError) as exc:
        err.print(f"[red]error:[/red] cannot read {run_dir}: {exc}")
        raise typer.Exit(2) from exc
    if results.manifest.track != TRACK:
        err.print(f"[red]error:[/red] unsupported track {results.manifest.track!r}")
        raise typer.Exit(2)
    text = render_latency_report(results)
    if write:
        (Path(run_dir) / REPORT_FILE).write_text(text, encoding="utf-8")
    typer.echo(text)
