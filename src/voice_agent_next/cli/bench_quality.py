"""``van bench quality`` — the T5 speech-to-speech quality track (registered by :mod:`.bench`)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer


def quality_cmd(
    engine: Annotated[str | None, typer.Option(help="Native S2S engine spec")] = None,
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
    preset: Annotated[
        str | None, typer.Option(help="A preset as the system under test (`van presets`)")
    ] = None,
    dataset: Annotated[
        list[str] | None,
        typer.Option(
            "--dataset",
            "-d",
            help="Built-in subset (big-bench-audio-smoke, voicebench-<subset>-smoke) or a "
            "manifest file; repeatable",
        ),
    ] = None,
    asr: Annotated[
        str, typer.Option(help="Fixed ASR that transcribes the agent's audio (STT spec)")
    ] = "faster-whisper/small.en",
    judge: Annotated[
        str | None,
        typer.Option(help="LLM judge spec, e.g. openai/gpt-4o-mini (default: rules only)"),
    ] = None,
    judge_temperature: Annotated[float, typer.Option(min=0.0, help="Judge temperature")] = 0.0,
    limit: Annotated[
        int | None, typer.Option(min=1, help="First N questions of every dataset")
    ] = None,
    reply_timeout: Annotated[
        float, typer.Option(min=0.1, help="Seconds without agent speech before a miss")
    ] = 20.0,
    answer_gap: Annotated[
        float, typer.Option(min=0.0, help="Silence (s) that ends an answer")
    ] = 1.5,
    max_reply: Annotated[float, typer.Option(min=1.0, help="Longest answer (s)")] = 120.0,
    reference_vad: Annotated[
        str, typer.Option(help="Reference VAD for agent onsets: rms, rms:<dBFS> or a VAD spec")
    ] = "rms",
    out: Annotated[Path, typer.Option("--out", "-o", help="Results directory")] = Path(
        "bench-results"
    ),
    run_id: Annotated[str | None, typer.Option(help="Run directory name")] = None,
    label: Annotated[str | None, typer.Option(help="System label used in reports")] = None,
    audio: Annotated[
        bool, typer.Option("--audio/--no-audio", help="Save stereo recordings per question")
    ] = True,
    engine_warmup: Annotated[
        bool, typer.Option("--engine-warmup/--no-engine-warmup", help="engine.warmup() first")
    ] = True,
    seed: Annotated[int, typer.Option(help="Bootstrap seed")] = 0,
    as_json: Annotated[bool, typer.Option("--json", help="Print summary.json to stdout")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """T5: speech-to-speech quality — spoken questions, transcribed answers, accuracy.

    Plays Big Bench Audio / VoiceBench questions (or your own recordings) to any engine,
    transcribes the agent's audio with a fixed ASR and scores it by rule, optionally
    with an LLM judge. Prints one row per dataset and category.

    Examples:

        van bench quality -c agent.yaml                       # big-bench-audio-smoke

        van bench quality -c agent.yaml -d voicebench-openbookqa-smoke -d voicebench-sd-qa-usa-smoke

        van bench quality --engine openai/gpt-realtime --judge openai/gpt-4o-mini --limit 20

        van bench quality --preset local-cpu -d my-questions.jsonl --asr faster-whisper/base.en
    """
    from rich.markup import escape

    from ..bench.onset import OnsetDetector, make_reference_vad
    from ..bench.quality_datasets import load_quality_dataset
    from ..bench.system import BenchSystem, parse_component_spec
    from ..bench.tracks.quality import (
        QualityOptions,
        QualityRecord,
        quality_markdown_table,
        run_quality_benchmark,
    )
    from ..errors import VoiceAgentError
    from ..presets import get_preset
    from ..utils.download import DownloadError
    from .bench import _logging, _print_notes, err

    _logging(verbose)
    try:
        if preset is not None and config is not None:
            raise ValueError("pass either --preset or --config, not both")
        base: Any = config
        if preset is not None:
            base = dict(get_preset(preset).config)
        system = BenchSystem.from_options(
            config=base, engine=parse_component_spec(engine), stt=parse_component_spec(stt),
            llm=parse_component_spec(llm), tts=parse_component_spec(tts),
            vad=parse_component_spec(vad), turn_detector=parse_component_spec(turn_detector),
            label=label or preset,
        )  # fmt: skip
        detector = OnsetDetector(make_reference_vad(reference_vad))
        sets = [
            load_quality_dataset(d, progress=lambda m: err.print(f"[dim]{escape(m)}[/dim]"))
            for d in (dataset or ["big-bench-audio-smoke"])
        ]
    except (VoiceAgentError, ValueError, DownloadError) as exc:
        err.print(f"[red]error:[/red] {escape(str(exc))}")
        raise typer.Exit(2) from exc
    options = QualityOptions(
        limit=limit, reply_timeout=reply_timeout, gap_after_reply=answer_gap,
        max_reply=max_reply, save_audio=audio, warmup_engine=engine_warmup,
        judge_temperature=judge_temperature, seed=seed,
    )  # fmt: skip
    total = sum(len(d.limit(limit).items) for d in sets)
    err.print(
        f"[bold]van bench quality[/bold] · {system.label} · "
        f"{', '.join(d.name for d in sets)} · {total} question(s) · ASR {asr}"
        + (f" · judge {judge}" if judge else ""),
        highlight=False,
    )

    def on_item(record: QualityRecord) -> None:
        speech = (record.answer_speech_ms or 0) / 1000
        heard = (
            "[red]no answer[/red]" if record.missed else
            f"~{record.answer_latency_ms or 0:,.0f} ms, {speech:.1f} s"
        )  # fmt: skip
        err.print(f"  {record.index + 1:>3}/{total} {record.item:<18} {heard}", highlight=False)

    try:
        results = asyncio.run(
            run_quality_benchmark(
                system, sets, options, asr=parse_component_spec(asr) or asr,
                judge=parse_component_spec(judge), out_dir=out, run_id=run_id,
                detector=detector, on_item=on_item,
            )
        )  # fmt: skip
    except VoiceAgentError as exc:
        err.print(f"[red]error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    if as_json:
        typer.echo(json.dumps(results.summary.model_dump(mode="json"), indent=2))
    else:
        typer.echo(quality_markdown_table(results))
        _print_notes(results)
