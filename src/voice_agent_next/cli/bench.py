"""``van bench`` — benchmarks measured at the audio boundary (see ``benchmarks/README.md``)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any

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


def _tolerate_narrow_console() -> None:
    """Replace characters the console encoding lacks (e.g. ``−``, ``µ`` on a cp1252
    Windows console) instead of crashing with ``UnicodeEncodeError``."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(errors="replace")


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
    """Benchmark suite: T1 latency, T2 ASR, T3 TTS, T4 VAD / turn-taking, T7 overhead."""


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
    from ..bench.tracks import asr, latency, overhead, turn_taking, turns
    from ..bench.tracks import tts as tts_track
    from ..bench.tracks import vad as vad_track

    renderers = {
        latency.TRACK: latency.render_latency_report,
        overhead.TRACK: overhead.render_overhead_report,
        asr.TRACK: asr.render_asr_report,
        tts_track.TRACK: tts_track.render_tts_report,
        vad_track.TRACK: vad_track.render_vad_report,
        turns.TRACK: turns.render_turns_report,
        turn_taking.TRACK: turn_taking.render_turn_taking_report,
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


# -------------------------------------------------------------------------- T2 ASR


def _print_asr(results: object) -> None:
    from rich.markup import escape

    from ..bench.report import fmt
    from ..bench.results import RunResults

    assert isinstance(results, RunResults)
    s = results.summary
    table = Table(title=f"{s.track} · {s.system} · {s.transport}", title_justify="left")
    for col in ("dataset", "n", "WER", "CER", "perfect", "RTFx", "TTFS p50", "TTFS p90",
                "1st partial p50"):  # fmt: skip
        table.add_column(col, justify="left" if col == "dataset" else "right")

    def pct(x: float | None, bold: bool = False) -> str:
        text = "-" if x is None else f"{100 * x:.2f}%"
        return f"[bold]{text}[/bold]" if bold and x is not None else text

    for name, d in s.extra.get("datasets", {}).items():
        cer = d.get("metric") == "cer"
        table.add_row(
            escape(name), fmt(d.get("scored")), pct(d.get("wer"), not cer), pct(d.get("cer"), cer),
            pct(d.get("perfect_rate")), fmt(d.get("rtfx"), 1),
            fmt(d.get("ttfs_p50_ms"), 0, unit=" ms"), fmt(d.get("ttfs_p90_ms"), 0, unit=" ms"),
            fmt(d.get("first_partial_p50_ms"), 0, unit=" ms"),
        )  # fmt: skip
    console.print(table)
    revisions = s.rates.get("interim_revision_rate")
    if revisions is not None:
        console.print(f"interim revision rate: {100 * revisions:.1f}%", highlight=False)
    if results.directory is not None:
        console.print(f"results: [bold]{results.directory}[/bold] (report.md, summary.json)")
    for note in results.manifest.notes:
        console.print(f"[yellow]note:[/yellow] {escape(note)}")


@app.command()
def asr(
    stt: Annotated[
        str,
        typer.Option(
            help="STT spec: faster-whisper/base, sherpa-onnx/nemo-fastconformer-en-80ms, or "
            "an inline mapping like '{provider: faster-whisper, model: base, device: cpu}'"
        ),
    ],
    dataset: Annotated[
        list[str] | None,
        typer.Option(
            "--dataset",
            "-d",
            help="Built-in subset (librispeech-test-clean-smoke, fleurs-{en,es,de,tr,zh}-smoke) "
            "or a manifest file (.jsonl/.json/.tsv/.csv: audio + text). Repeatable.",
        ),
    ] = None,
    mode: Annotated[str, typer.Option(help="batch (transcribe()) or streaming (stream())")] = (
        "batch"
    ),
    chunk_ms: Annotated[
        float, typer.Option(min=1.0, max=1000.0, help="Streaming chunk size (ms)")
    ] = 20.0,
    realtime_factor: Annotated[
        float,
        typer.Option(
            min=0.0, help="Streaming pacing: 1 = real time, 2 = 2x, 0 = as fast as possible"
        ),
    ] = 1.0,
    normalizer: Annotated[
        str, typer.Option(help="auto, whisper-english, whisper-basic or none")
    ] = "auto",
    language: Annotated[
        str | None, typer.Option(help="Override the dataset language (e.g. for manifests)")
    ] = None,
    limit: Annotated[
        int | None, typer.Option(min=1, help="Only the first N utterances of each dataset")
    ] = None,
    vad: Annotated[
        str | None,
        typer.Option(help="VAD spec to stream a batch-only STT through StreamAdapter"),
    ] = None,
    warmup: Annotated[
        bool, typer.Option("--warmup/--no-warmup", help="stt.warmup() + one unmeasured utterance")
    ] = True,
    final_timeout: Annotated[
        float, typer.Option(min=0.1, help="Streaming: seconds to wait for the final transcript")
    ] = 30.0,
    out: Annotated[Path, typer.Option("--out", "-o", help="Results directory")] = Path(
        "bench-results"
    ),
    run_id: Annotated[str | None, typer.Option(help="Run directory name")] = None,
    label: Annotated[str | None, typer.Option(help="System label used in reports")] = None,
    seed: Annotated[int, typer.Option(help="Bootstrap seed")] = 0,
    as_json: Annotated[bool, typer.Option("--json", help="Print summary.json to stdout")] = False,
    markdown: Annotated[
        bool, typer.Option("--markdown", help="Print the per-dataset Markdown table")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """T2: ASR accuracy (WER/CER), throughput (RTFx) and latency (TTFS, first partial).

    Examples:

        van bench asr --stt faster-whisper/base

        van bench asr --stt sherpa-onnx/nemo-fastconformer-en-80ms --mode streaming

        van bench asr --stt faster-whisper/base -d fleurs-es-smoke -d fleurs-de-smoke

        van bench asr --stt mock -d my-data/manifest.jsonl --language en
    """
    from ..bench.asr_datasets import load_asr_dataset
    from ..bench.system import parse_component_spec
    from ..bench.tracks.asr import AsrItem, AsrOptions, asr_markdown_table, run_asr_benchmark
    from ..errors import VoiceAgentError

    _tolerate_narrow_console()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    options = AsrOptions(
        mode=mode,  # type: ignore[arg-type]  # validated below
        chunk_ms=chunk_ms,
        realtime_factor=realtime_factor,
        normalizer=normalizer,
        language=language,
        limit=limit,
        warmup=warmup,
        final_timeout=final_timeout,
        seed=seed,
    )
    try:
        options.validate()
        spec = parse_component_spec(stt)
        assert spec is not None  # --stt is required
        vad_spec = parse_component_spec(vad)
        progress = lambda line: err.print(f"  {line}", highlight=False)  # noqa: E731
        sets = [
            load_asr_dataset(d, language=language, progress=progress)
            for d in (dataset or ["librispeech-test-clean-smoke"])
        ]
    except (VoiceAgentError, ValueError) as exc:
        err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2) from exc
    total = sum(len(d.limit(limit).utterances) for d in sets)
    err.print(
        f"[bold]van bench asr[/bold] · {label or stt} · {mode} · "
        f"{', '.join(d.name for d in sets)} ({total} utterances)",
        highlight=False,
    )
    done = 0

    def on_item(item: AsrItem) -> None:
        nonlocal done
        done += 1
        if item.error is not None:
            detail = f"[red]error[/red] {item.error}"
        else:
            rate = item.cer if item.metric == "cer" else item.wer
            detail = f"{item.metric.upper()} {'-' if rate is None else f'{100 * rate:5.1f}%'}"
            if item.ttfs_ms is not None:
                detail += f"  TTFS {item.ttfs_ms:,.0f} ms"
        err.print(f"  {done:>4}/{total} {item.dataset:<28} {item.id:<28} {detail}",
                  highlight=False)  # fmt: skip

    try:
        results = asyncio.run(
            run_asr_benchmark(
                spec, sets, options, vad=vad_spec, out_dir=out, run_id=run_id, label=label,
                on_item=on_item,
            )
        )  # fmt: skip
    except ValueError as exc:
        err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2) from exc
    except VoiceAgentError as exc:
        err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc
    if as_json:
        typer.echo(json.dumps(results.summary.model_dump(mode="json"), indent=2))
    elif markdown:
        typer.echo(asr_markdown_table(results))
    else:
        _print_asr(results)


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

    _tolerate_narrow_console()
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


# -------------------------------------------------------------------------- T3 TTS


def _print_tts(results: object) -> None:
    from rich.markup import escape

    from ..bench.report import fmt
    from ..bench.results import RunResults
    from ..bench.tracks.tts import MODES

    assert isinstance(results, RunResults)
    s = results.summary
    table = Table(title=f"{s.track} · {escape(s.system)} · {s.dataset}", title_justify="left")
    for col in ("mode", "n", "TTFA p50", "TTFA p95", "TTFB p50", "lead sil. p50", "RTF p50",
                "underruns/min", "rt WER", "rt CER", "hard text", "DNSMOS ovrl"):  # fmt: skip
        table.add_column(col, justify="left" if col == "mode" else "right")

    def pct(x: float | None) -> str:
        return "-" if x is None else f"{100 * x:.1f}%"

    for mode in MODES:
        ttfa = s.metrics.get(f"{mode}.ttfa_ms")
        if ttfa is None:
            continue
        ttfb = s.metrics[f"{mode}.ttfb_ms"]
        lead = s.metrics[f"{mode}.leading_silence_ms"]
        rtf = s.metrics[f"{mode}.rtf"]
        ovrl = s.metrics.get(f"{mode}.dnsmos_ovrl")
        info = s.extra.get("modes", {}).get(mode, {})
        table.add_row(
            mode, str(ttfa.n), fmt(ttfa.p50, 0, unit=" ms"), fmt(ttfa.p95, 0, unit=" ms"),
            fmt(ttfb.p50, 0, unit=" ms"), fmt(lead.p50, 0, unit=" ms"), fmt(rtf.p50, 3),
            fmt(info.get("underruns_per_min"), 2), pct(s.rates.get(f"{mode}.rt_wer")),
            pct(s.rates.get(f"{mode}.rt_cer")), pct(s.rates.get(f"{mode}.hardtext_acc")),
            fmt(ovrl.mean if ovrl else None, 2),
        )  # fmt: skip
    console.print(table)
    if results.directory is not None:
        console.print(f"results: [bold]{results.directory}[/bold] (report.md, summary.json)")
    for note in results.manifest.notes:
        console.print(f"[yellow]note:[/yellow] {escape(note)}")


@app.command("tts")
def tts_command(
    tts: Annotated[
        str,
        typer.Option(
            "--tts",
            help="TTS spec: kokoro, pocket-tts, sherpa-onnx/piper-en_US-libritts_r-medium, "
            "or an inline mapping like '{provider: mock, ttfb: 0.1}'",
        ),
    ],
    stt: Annotated[
        str | None,
        typer.Option(help="STT for round-trip WER, e.g. faster-whisper/small.en (default: skip)"),
    ] = None,
    texts: Annotated[
        str, typer.Option("--texts", "-t", help="Text set: smoke or a .txt/.json/.yaml file")
    ] = "smoke",
    mode: Annotated[
        str, typer.Option(help="batch, streaming (LLM-paced text input) or both")
    ] = "both",
    repeats: Annotated[int, typer.Option(min=1, help="Requests per text and mode")] = 1,
    limit: Annotated[int | None, typer.Option(min=1, help="Use only the first N texts")] = None,
    words_per_second: Annotated[
        float, typer.Option(min=0.0, help="Streaming text pace (0: push the text at once)")
    ] = 15.0,
    warmup_requests: Annotated[
        int, typer.Option(min=0, help="Untimed warm-up requests per mode")
    ] = 1,
    mos: Annotated[
        str, typer.Option(help="MOS predictor: none or dnsmos (1.2 MB ONNX, onnx extra)")
    ] = "none",
    normalizer: Annotated[
        str, typer.Option(help="Round-trip normalizer: auto, whisper-english, basic-english, none")
    ] = "auto",
    language: Annotated[
        str | None, typer.Option(help="STT language (default: the text set's)")
    ] = None,
    timeout: Annotated[float, typer.Option(min=0.1, help="Seconds per request")] = 120.0,
    out: Annotated[Path, typer.Option("--out", "-o", help="Results directory")] = Path(
        "bench-results"
    ),
    run_id: Annotated[str | None, typer.Option(help="Run directory name")] = None,
    label: Annotated[str | None, typer.Option(help="System label used in reports")] = None,
    audio: Annotated[
        bool, typer.Option("--audio/--no-audio", help="Save every clip as WAV")
    ] = True,
    seed: Annotated[int, typer.Option(help="Bootstrap seed")] = 0,
    as_json: Annotated[bool, typer.Option("--json", help="Print summary.json to stdout")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """T3: TTS time to first audio, RTF, stalls, round-trip WER and MOS predictors.

    Examples:

        van bench tts --tts mock

        van bench tts --tts kokoro --stt faster-whisper/small.en

        van bench tts --tts pocket-tts --mode streaming --words-per-second 20 --mos dnsmos
    """
    from ..bench.system import parse_component_spec
    from ..bench.tracks.tts import MODES, TTSItem, TTSOptions, load_texts, run_tts_benchmark
    from ..errors import VoiceAgentError

    _tolerate_narrow_console()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    modes = MODES if mode == "both" else (mode,)
    try:
        text_set = load_texts(texts).limited(limit)
        options = TTSOptions(
            modes=modes,
            repeats=repeats,
            warmup_requests=warmup_requests,
            words_per_second=words_per_second,
            timeout=timeout,
            normalizer=normalizer,
            language=language,
            save_audio=audio,
            seed=seed,
        )
        options.validate()
        tts_spec = parse_component_spec(tts)
        stt_spec = parse_component_spec(stt)
    except (VoiceAgentError, ValueError, OSError) as exc:
        err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2) from exc
    assert tts_spec is not None
    err.print(
        f"[bold]van bench tts[/bold] · {tts} · {text_set.dataset_id()} "
        f"({len(text_set.texts)} texts) · {', '.join(modes)} x {repeats}",
        highlight=False,
    )

    def on_item(item: TTSItem) -> None:
        if item.error:
            status = f"error: {item.error}"
        else:
            ttfa = "-" if item.ttfa_ms is None else f"{item.ttfa_ms:,.0f} ms"
            rtf = "-" if item.rtf is None else f"{item.rtf:.3f}"
            status = f"TTFA {ttfa}  RTF {rtf}"
        err.print(f"  {item.mode:<9} {item.text_id:<18} {status}", highlight=False, markup=False)

    try:
        results = asyncio.run(
            run_tts_benchmark(
                tts_spec, text_set, options, stt=stt_spec, mos=mos, out_dir=out,
                run_id=run_id, label=label, on_item=on_item,
            )
        )  # fmt: skip
    except (VoiceAgentError, ValueError) as exc:
        err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc
    if as_json:
        typer.echo(json.dumps(results.summary.model_dump(mode="json"), indent=2))
    else:
        _print_tts(results)


# --------------------------------------------------------------- T4 VAD / turn-taking


def _print_notes(results: object) -> None:
    from rich.markup import escape

    from ..bench.results import RunResults

    assert isinstance(results, RunResults)
    if results.directory is not None:
        console.print(f"results: [bold]{results.directory}[/bold] (report.md, summary.json)")
    for note in results.manifest.notes:
        console.print(f"[yellow]note:[/yellow] {escape(note)}")


def _logging(verbose: bool) -> None:
    _tolerate_narrow_console()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command("vad")
def vad_cmd(
    vad: Annotated[
        list[str] | None,
        typer.Option(
            "--vad", help="VAD spec, repeatable: energy, silero, sherpa-onnx/ten-vad, or a mapping"
        ),
    ] = None,
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset", "-d", help="Source utterances: an ASR smoke subset or a manifest file"
        ),
    ] = "librispeech-test-clean-smoke",
    condition: Annotated[
        list[str] | None,
        typer.Option(
            "--condition",
            help="Noise condition, repeatable: clean, transient, white@10, pink@5, brown@0 "
            "(default: clean, pink@20/10/5, white@10, transient)",
        ),
    ] = None,
    limit: Annotated[int | None, typer.Option(min=1, help="Only the first N utterances")] = None,
    corpus_seed: Annotated[int, typer.Option(help="Seed of the corpus layout and noise")] = 0,
    chunk_ms: Annotated[float, typer.Option(min=1.0, max=1000.0, help="Push size (ms)")] = 20.0,
    out: Annotated[Path, typer.Option("--out", "-o", help="Results directory")] = Path(
        "bench-results"
    ),
    run_id: Annotated[str | None, typer.Option(help="Run directory name")] = None,
    label: Annotated[str | None, typer.Option(help="System label used in reports")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print summary.json to stdout")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """T4: VAD frame accuracy, onset/offset lag and false alarms in noise.

    Precision/recall/F1/AUC on 10 ms frames, onset and offset lag of the VAD's events,
    false alarms per minute and speed, on a deterministic labelled corpus.

    Examples:

        van bench vad --vad energy --vad silero --vad sherpa-onnx/ten-vad

        van bench vad --vad silero --condition clean --condition pink@0 --limit 20
    """
    from ..bench.asr_datasets import load_asr_dataset, load_audio
    from ..bench.system import parse_component_spec
    from ..bench.tracks.vad import VadClipResult, VadOptions, run_vad_benchmark, vad_markdown_table
    from ..bench.vad_corpus import DEFAULT_CONDITIONS, build_vad_corpus, parse_condition
    from ..errors import VoiceAgentError

    _logging(verbose)
    try:
        specs = [parse_component_spec(v) for v in (vad or ["energy"])]
        conditions = [parse_condition(c) for c in condition] if condition else DEFAULT_CONDITIONS
        progress = lambda line: err.print(f"  {line}", highlight=False)  # noqa: E731
        data = load_asr_dataset(dataset, progress=progress).limit(limit)
        err.print(f"building the corpus from {len(data.utterances)} utterances of {data.name}...")
        corpus = build_vad_corpus(
            [(u.id, load_audio(u.audio)) for u in data.utterances],
            conditions=conditions,
            seed=corpus_seed,
        )
    except (VoiceAgentError, ValueError) as exc:
        err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2) from exc

    def on_clip(r: VadClipResult) -> None:
        f1 = "-" if r.f1 is None else f"{100 * r.f1:.1f}%"
        err.print(f"  {r.vad:<24} {r.condition:<14} F1 {f1:>6}  FA/min "
                  f"{r.false_alarms_per_min or 0:.1f}", highlight=False)  # fmt: skip

    try:
        results = asyncio.run(
            run_vad_benchmark(
                [s for s in specs if s is not None], corpus, VadOptions(chunk_ms=chunk_ms),
                dataset=data.name, out_dir=out, run_id=run_id, label=label, on_clip=on_clip,
            )
        )  # fmt: skip
    except ValueError as exc:
        err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2) from exc
    except VoiceAgentError as exc:
        err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc
    if as_json:
        typer.echo(json.dumps(results.summary.model_dump(mode="json"), indent=2))
    else:
        typer.echo(vad_markdown_table(results))
        _print_notes(results)


@app.command()
def turns(
    detector: Annotated[
        str,
        typer.Option(
            "--detector", "--turn", help="Turn detector spec: smart_turn, mock, or a mapping"
        ),
    ] = "smart_turn",
    dataset: Annotated[
        list[str] | None,
        typer.Option(
            "--dataset",
            "-d",
            help="eot-bench-<lang> (en, de, es, fr, it, pt, nl, tr, ar, hi, id, ja, ko, zh; "
            "downloaded once, 96-166 MB) or a .jsonl manifest. Repeatable.",
        ),
    ] = None,
    limit: Annotated[int | None, typer.Option(min=1, help="Only the first N turns")] = None,
    score_point: Annotated[
        float, typer.Option(min=0.01, help="Seconds into a pause at which the detector is asked")
    ] = 0.2,
    transcript_lag: Annotated[
        float, typer.Option(min=0.0, help="Words reach text detectors this late (s)")
    ] = 0.5,
    threshold: Annotated[
        float | None, typer.Option(min=0.0, max=1.0, help="Decision threshold (default: own)")
    ] = None,
    min_endpointing_delay: Annotated[
        float, typer.Option(min=0.0, help="Configured policy: delay when the turn is complete")
    ] = 0.4,
    max_endpointing_delay: Annotated[
        float, typer.Option(min=0.0, help="Configured policy: delay otherwise (timeout)")
    ] = 2.5,
    out: Annotated[Path, typer.Option("--out", "-o", help="Results directory")] = Path(
        "bench-results"
    ),
    run_id: Annotated[str | None, typer.Option(help="Run directory name")] = None,
    label: Annotated[str | None, typer.Option(help="System label used in reports")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print summary.json to stdout")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """T4: end-of-turn detection on LiveKit's eot-bench.

    False cutoffs at 300/600 ms, latency at 5/10 % false cutoffs, the configured cascade
    policy, accuracy/F1 (complete vs incomplete) and inference time.

    Examples:

        van bench turns --detector smart_turn                  # eot-bench English, 400 turns

        van bench turns --detector '{provider: smart_turn, model: v3.2-gpu}' -d eot-bench-de

        van bench turns --detector mock -d my-turns.jsonl
    """
    from ..bench.eot_datasets import load_eot_dataset
    from ..bench.system import parse_component_spec
    from ..bench.tracks.turns import (
        TurnsOptions,
        run_turns_benchmark,
        turns_markdown_table,
    )
    from ..errors import VoiceAgentError

    _logging(verbose)
    options = TurnsOptions(
        score_point=score_point, transcript_lag=transcript_lag, threshold=threshold,
        min_endpointing_delay=min_endpointing_delay,
        max_endpointing_delay=max_endpointing_delay, limit=limit,
    )  # fmt: skip
    try:
        options.validate()
        spec = parse_component_spec(detector)
        progress = lambda line: err.print(f"  {line}", highlight=False)  # noqa: E731
        sets = [load_eot_dataset(d, progress=progress) for d in (dataset or ["eot-bench-en"])]
    except (VoiceAgentError, ValueError) as exc:
        err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2) from exc
    total = sum(len(d.limit(limit).turns) for d in sets)
    err.print(f"[bold]van bench turns[/bold] · {label or detector} · "
              f"{', '.join(d.name for d in sets)} ({total} turns)", highlight=False)  # fmt: skip
    done = 0

    def on_turn(_turn: Any, _items: Any) -> None:
        nonlocal done
        done += 1
        if done % 50 == 0 or done == total:
            err.print(f"  {done}/{total} turns", highlight=False)

    try:
        results = asyncio.run(
            run_turns_benchmark(
                spec, sets, options, out_dir=out, run_id=run_id, label=label, on_turn=on_turn
            )
        )
    except ValueError as exc:
        err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2) from exc
    except VoiceAgentError as exc:
        err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc
    if as_json:
        typer.echo(json.dumps(results.summary.model_dump(mode="json"), indent=2))
    else:
        typer.echo(turns_markdown_table(results))
        _print_notes(results)


@app.command("turn-taking")
def turn_taking_cmd(
    engine: Annotated[
        str | None,
        typer.Option(help="Native S2S engine spec (default: the mock engine with long replies)"),
    ] = None,
    stt: Annotated[str | None, typer.Option(help="Cascade STT spec")] = None,
    llm: Annotated[str | None, typer.Option(help="Cascade LLM spec")] = None,
    tts: Annotated[str | None, typer.Option(help="Cascade TTS spec")] = None,
    vad: Annotated[str | None, typer.Option(help="Cascade VAD spec")] = None,
    turn_detector: Annotated[
        str | None, typer.Option("--turn", help="Cascade turn detector spec")
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Agent config (YAML/TOML/JSON)")
    ] = None,
    scenario: Annotated[
        str, typer.Option("--scenario", "-s", help="Built-in scenario name or YAML file")
    ] = "turn-taking-smoke",
    turns: Annotated[
        int | None, typer.Option("--turns", "-n", min=1, help="Turns per session (default: all)")
    ] = None,
    sessions: Annotated[int, typer.Option(min=1, help="Separate sessions")] = 1,
    warmup_turns: Annotated[int, typer.Option(min=0, help="Leading turns not scored")] = 1,
    reply_timeout: Annotated[
        float | None, typer.Option(min=0.1, help="Seconds without reply before a turn is missed")
    ] = None,
    reference_vad: Annotated[
        str, typer.Option(help="Reference VAD for the recording: rms, rms:<dBFS> or a VAD spec")
    ] = "rms",
    out: Annotated[Path, typer.Option("--out", "-o", help="Results directory")] = Path(
        "bench-results"
    ),
    run_id: Annotated[str | None, typer.Option(help="Run directory name")] = None,
    label: Annotated[str | None, typer.Option(help="System label used in reports")] = None,
    audio: Annotated[
        bool, typer.Option("--audio/--no-audio", help="Save stereo recordings and labels")
    ] = True,
    as_json: Annotated[bool, typer.Option("--json", help="Print summary.json to stdout")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """T4: turn-taking battery on any engine (premature replies, barge-in, false barge-ins).

    Premature replies in mid-turn pauses, barge-in stop time, false barge-ins on
    backchannels and noise, resumption, missed turns and dead air, on the call recording.

    Examples:

        van bench turn-taking                                  # mock engine, smoke scenario

        van bench turn-taking --engine '{provider: mock, vad_options: {min_silence_duration: 1.0}}'

        van bench turn-taking -c agent.yaml -s benchmarks/scenarios/turn-taking-local.yaml
    """
    from ..bench.caller import TurnTiming
    from ..bench.onset import OnsetDetector, make_reference_vad
    from ..bench.stimuli import load_scenario
    from ..bench.system import BenchSystem, parse_component_spec
    from ..bench.tracks.turn_taking import (
        DEFAULT_MOCK_ENGINE,
        TurnTakingOptions,
        run_turn_taking_benchmark,
        turn_taking_markdown_table,
    )
    from ..errors import VoiceAgentError

    _logging(verbose)
    try:
        engine_spec = parse_component_spec(engine)
        cascade = any(x is not None for x in (stt, llm, tts, turn_detector))
        if engine_spec is None and not cascade and config is None:
            engine_spec = DEFAULT_MOCK_ENGINE
        system = BenchSystem.from_options(
            config=config, engine=engine_spec, stt=parse_component_spec(stt),
            llm=parse_component_spec(llm), tts=parse_component_spec(tts),
            vad=parse_component_spec(vad), turn_detector=parse_component_spec(turn_detector),
            label=label, default_engine="mock",
        )  # fmt: skip
        scn = load_scenario(scenario)
        detector = OnsetDetector(make_reference_vad(reference_vad))
    except (VoiceAgentError, ValueError) as exc:
        err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2) from exc
    options = TurnTakingOptions(
        turns=turns, sessions=sessions, warmup_turns=warmup_turns, reply_timeout=reply_timeout,
        save_audio=audio,
    )  # fmt: skip
    err.print(f"[bold]van bench turn-taking[/bold] · {system.label} · scenario {scn.name}",
              highlight=False)  # fmt: skip

    def on_turn(session: int, turn: TurnTiming) -> None:
        heard = "-" if turn.reply_start is None else (
            f"~{(turn.reply_start - turn.speech_end) * 1000:,.0f} ms")  # fmt: skip
        err.print(f"  session {session + 1} turn {turn.index + 1:>3} {turn.stimulus.id:<12} "
                  f"{heard}", highlight=False)  # fmt: skip

    try:
        results = asyncio.run(
            run_turn_taking_benchmark(
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
        typer.echo(turn_taking_markdown_table(results))
        _print_notes(results)
