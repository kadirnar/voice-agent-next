"""``van bench`` — benchmarks measured at the audio boundary (see ``benchmarks/README.md``)."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
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
    """Benchmark suite: T1 latency, T7 framework overhead (more tracks: see ROADMAP M6)."""


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
        f"{turns} turn(s) x {sessions} session(s)",
        highlight=False,
    )

    def on_turn(session: int, turn: TurnTiming) -> None:
        if turn.reply_start is not None:
            heard = f"~{(turn.reply_start - turn.speech_end) * 1000:,.0f} ms"
        else:
            heard = "[red]no reply[/red]" if turn.stimulus.expect_reply else "-"
        err.print(
            f"  session {session + 1} turn {turn.index + 1:>3}/{turns} "
            f"{turn.stimulus.id:<12} {heard}",
            highlight=False,
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
    from ..bench.tracks import latency, overhead

    renderers = {
        latency.TRACK: latency.render_latency_report,
        overhead.TRACK: overhead.render_overhead_report,
    }
    try:
        results = load_run(run_dir)
    except (OSError, ValueError) as exc:
        err.print(f"[red]error:[/red] cannot read {run_dir}: {exc}")
        raise typer.Exit(2) from exc
    render = renderers.get(results.manifest.track)
    if render is None:
        err.print(f"[red]error:[/red] unsupported track {results.manifest.track!r}")
        raise typer.Exit(2)
    text = render(results)
    if write:
        (Path(run_dir) / REPORT_FILE).write_text(text, encoding="utf-8")
    typer.echo(text)


# --------------------------------------------------------------------- T7 overhead


def _split(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(p.strip() for p in value.split(",") if p.strip())


def _print_overhead(results: object) -> None:
    from rich.markup import escape

    from ..bench.report import fmt
    from ..bench.results import RunResults

    assert isinstance(results, RunResults)
    s = results.summary
    headline = s.metrics.get("e2e.overhead_ms")
    if headline is not None and headline.n:
        console.print(
            f"[bold]framework overhead[/bold] p50 {fmt(headline.p50, 2)} ms · p90 "
            f"{fmt(headline.p90, 2)} ms · p99 {fmt(headline.p99, 2)} ms over {headline.n} "
            "turns (v2v − injected component delays)",
            highlight=False,
        )
    e2e = s.extra.get("e2e")
    if e2e:
        table = Table(title="end-to-end (ms)", title_justify="left")
        for col in ("condition", "injected", "v2v p50", "overhead p50", "p90", "p99",
                    "jitter p50", "lag p99", "CPU %"):  # fmt: skip
            table.add_column(col, justify="left" if col == "condition" else "right")
        for name, info in e2e["conditions"].items():
            v2v = s.metrics.get(f"e2e.{name}.v2v_ms")
            ovh = s.metrics.get(f"e2e.{name}.overhead_ms")
            jit = s.metrics.get(f"e2e.{name}.frame_jitter_ms")
            table.add_row(
                name, fmt(info.get("injected_ms"), 0), fmt(v2v.p50 if v2v else None, 1),
                fmt(ovh.p50 if ovh else None, 2), fmt(ovh.p90 if ovh else None, 2),
                fmt(ovh.p99 if ovh else None, 2), fmt(jit.p50 if jit else None, 3),
                fmt(info.get("loop_lag_p99_ms"), 2), fmt(info.get("cpu_pct"), 1),
            )  # fmt: skip
        console.print(table)
    flush = s.metrics.get("flush.flush_ms")
    if flush is not None and flush.n:
        console.print(
            f"flush (interrupt → last agent audio): p50 {fmt(flush.p50, 3)} ms, max "
            f"{fmt(flush.max, 3)} ms over {flush.n} interrupts; leaked frames "
            f"{s.counts.get('flush_leaked_frames', 0)}",
            highlight=False,
        )
    cap = s.extra.get("capacity")
    if cap:
        bound = "" if cap["limit_found"] else " (limit not reached)"
        console.print(
            f"capacity: {cap['sessions_per_core']} sessions per core{bound} · CPU/session "
            f"{fmt(cap.get('cpu_pct_per_session'), 2)} % · memory/session "
            f"{fmt(cap.get('rss_per_session_mb'), 2)} MB",
            highlight=False,
        )
    micro = s.extra.get("micro")
    if micro:
        table = Table(title="micro-benchmarks (µs per operation)", title_justify="left")
        for col in ("benchmark", "per", "median", "min", "IQR", "% real time"):
            table.add_column(col, justify="left" if col in ("benchmark", "per") else "right")
        for b in micro["benchmarks"]:
            budget = b.get("budget_pct")
            table.add_row(
                b["name"], b["unit"], fmt(b["median_us"], 3), fmt(b["min_us"], 3),
                fmt(b["iqr_us"], 3), "-" if budget is None else f"{budget:.3f}",
            )  # fmt: skip
        console.print(table)
    if results.directory is not None:
        console.print(f"results: [bold]{results.directory}[/bold] (report.md, summary.json)")
    for note in results.manifest.notes:
        console.print(f"[yellow]note:[/yellow] {escape(note)}")


def _print_gate(gate: object) -> None:
    from rich.markup import escape

    from ..bench.gate import ROW_HEADERS, GateReport

    assert isinstance(gate, GateReport)
    colors = {"regressed": "red", "missing": "red", "improved": "green", "new": "cyan"}
    limit = ROW_HEADERS.index("fails above")
    table = Table(title=f"regression gate ({gate.platform})", title_justify="left")
    for i, col in enumerate(ROW_HEADERS):
        if i != limit:  # printed once per rule below: keeps the table readable at 80 columns
            table.add_column(
                col, justify="right" if col in ("baseline", "run", "change") else "left"
            )
    limits: dict[str, str] = {}
    for c, cells in gate.rows():
        cells = [escape(cell) for cell in cells]
        if c.limit:
            limits[c.rule] = c.limit
        color = colors.get(c.status)
        if color:
            cells[-1] = f"[{color}]{cells[-1]}[/{color}]"
        del cells[limit]
        table.add_row(cells[0].strip("`"), *cells[1:])
    console.print(table)
    for rule, text in limits.items():
        console.print(f"  {rule} metrics fail above {escape(text)}", highlight=False)
    color = "yellow" if not gate.gated else "green" if gate.passed else "red"
    console.print(f"gate: [{color}]{escape(gate.verdict)}[/{color}]")
    for note in gate.notes:
        console.print(f"[yellow]note:[/yellow] {escape(note)}")


def _append(path: Path | None, text: str) -> None:
    if path is None:
        return
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(text.rstrip("\n") + "\n\n")
    except OSError as exc:
        err.print(f"[yellow]warning:[/yellow] cannot write the summary to {path}: {exc}")


@app.command()
def overhead(
    tier: Annotated[
        str, typer.Option(help="Preset: smoke (CI gate, ~2 min) or full (publishable)")
    ] = "smoke",
    sections: Annotated[
        str | None, typer.Option(help="Comma-separated subset of micro,e2e,flush,capacity")
    ] = None,
    conditions: Annotated[
        str | None,
        typer.Option(help="e2e conditions: engine, engine-delay, cascade, cascade-delay"),
    ] = None,
    turns: Annotated[
        int | None, typer.Option("--turns", "-n", min=2, help="Turns per e2e session")
    ] = None,
    sessions: Annotated[int | None, typer.Option(min=1, help="e2e sessions per condition")] = None,
    max_sessions: Annotated[
        int | None, typer.Option(min=1, help="Upper bound of the capacity sweep")
    ] = None,
    out: Annotated[Path, typer.Option("--out", "-o", help="Results directory")] = Path(
        "bench-results"
    ),
    run_id: Annotated[str | None, typer.Option(help="Run directory name")] = None,
    baseline: Annotated[
        Path | None,
        typer.Option(help="Baseline file to gate against (exit 1 on a regression)"),
    ] = None,
    update_baseline: Annotated[
        bool,
        typer.Option(
            "--update-baseline",
            help="Record this run in the baseline file (--baseline, default "
            "benchmarks/baselines/overhead-ci.json) instead of gating",
        ),
    ] = False,
    platform: Annotated[
        str | None, typer.Option(help="Baseline entry to use (default: the run's OS)")
    ] = None,
    retries: Annotated[
        int,
        typer.Option(min=0, help="Re-run failing sections; fail only if a regression repeats"),
    ] = 0,
    from_run: Annotated[
        Path | None, typer.Option(help="Gate/record an existing run directory instead of running")
    ] = None,
    summary: Annotated[
        Path | None,
        typer.Option(help="Append a Markdown summary to this file (e.g. $GITHUB_STEP_SUMMARY)"),
    ] = None,
    seed: Annotated[int, typer.Option(help="Bootstrap seed")] = 0,
    as_json: Annotated[bool, typer.Option("--json", help="Print summary.json to stdout")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """T7: framework overhead (v2v − injected delays, flush, jitter, loop lag, capacity,
    hot-path micro-benchmarks) and the CI regression gate.

    Examples:

        van bench overhead                                   # smoke tier, ~2 min

        van bench overhead --baseline benchmarks/baselines/overhead-ci.json --retries 1

        van bench overhead --update-baseline                 # record this machine's entry

        van bench overhead --from-run bench-results/<run-id> --update-baseline
    """
    from ..bench.gate import (
        DEFAULT_BASELINE_PATH,
        compare_to_baseline,
        load_baseline,
    )
    from ..bench.gate import (
        update_baseline as record_baseline,
    )
    from ..bench.results import RunResults, load_run
    from ..bench.tracks.overhead import (
        BUILTIN_CONDITIONS,
        TRACK,
        OverheadOptions,
        overhead_markdown_summary,
        run_overhead_benchmark,
    )
    from ..errors import VoiceAgentError

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        options = OverheadOptions.for_tier(
            tier,
            sections=_split(sections),
            conditions=_split(conditions),
            turns=turns,
            sessions=sessions,
            capacity_max_sessions=max_sessions,
            seed=seed,
        )
        options.validate(BUILTIN_CONDITIONS)
    except ValueError as exc:
        err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2) from exc

    def run(opts: OverheadOptions, rid: str | None) -> RunResults:
        err.print(
            f"[bold]van bench overhead[/bold] · {opts.tier} tier · sections "
            f"{', '.join(opts.sections)}",
            highlight=False,
        )
        try:
            return asyncio.run(
                run_overhead_benchmark(
                    opts, out_dir=out, run_id=rid,
                    progress=lambda line: err.print(f"  {line}", highlight=False),
                )
            )  # fmt: skip
        except VoiceAgentError as exc:
            err.print(f"[red]error:[/red] {exc}")
            raise typer.Exit(1) from exc

    if from_run is not None:
        try:
            results = load_run(from_run)
        except (OSError, ValueError) as exc:
            err.print(f"[red]error:[/red] cannot read {from_run}: {exc}")
            raise typer.Exit(2) from exc
        if results.manifest.track != TRACK:
            err.print(f"[red]error:[/red] {from_run} is a {results.manifest.track!r} run")
            raise typer.Exit(2)
    else:
        results = run(options, run_id)

    if as_json:
        typer.echo(json.dumps(results.summary.model_dump(mode="json"), indent=2))
    else:
        _print_overhead(results)
    run_md = overhead_markdown_summary(results)

    if update_baseline:
        path = baseline or DEFAULT_BASELINE_PATH
        try:
            _, key = record_baseline(path, results, key=platform)
        except (OSError, ValueError) as exc:
            err.print(f"[red]error:[/red] cannot update {path}: {exc}")
            raise typer.Exit(2) from exc
        console.print(f"baseline [bold]{path}[/bold] entry `{key}` <- run {results.summary.run_id}")
        _append(summary, run_md)
        return
    if baseline is None:
        _append(summary, run_md)
        return

    try:
        base = load_baseline(baseline)
    except FileNotFoundError as exc:
        err.print(
            f"[red]error:[/red] baseline {baseline} not found (create it with --update-baseline)"
        )
        raise typer.Exit(2) from exc
    except ValueError as exc:
        err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2) from exc
    gate = compare_to_baseline(base, results, key=platform, baseline_path=baseline)
    attempt = 0
    while gate.gated and not gate.passed and attempt < retries and from_run is None:
        attempt += 1
        failing = gate.failing_sections()
        err.print(
            f"[yellow]gate:[/yellow] {len(gate.failures)} regression(s) in "
            f"{', '.join(failing)}; re-running (attempt {attempt}/{retries})",
            highlight=False,
        )
        retry = run(replace(options, sections=tuple(failing)),
                    f"{results.summary.run_id}-retry{attempt}")  # fmt: skip
        gate = gate.confirm(compare_to_baseline(base, retry, key=platform, baseline_path=baseline))
    if results.directory is not None:
        (results.directory / "gate.json").write_text(
            json.dumps(gate.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (results.directory / "gate.md").write_text(gate.to_markdown(), encoding="utf-8")
    _print_gate(gate)
    _append(summary, gate.to_markdown() + "\n" + run_md)
    if not gate.passed:
        raise typer.Exit(1)
