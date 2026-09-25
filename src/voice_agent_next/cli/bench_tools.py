"""``van bench tools`` — the T6 tool-use track (registered by :mod:`.bench`)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer

DEFAULT_CALLER_TTS: dict[str, Any] = {"provider": "kokoro/v1.0-fp16", "voice": "am_adam"}
"""The caller's voice for systems that transcribe speech (when the suite has none)."""


def tools_cmd(
    engine: Annotated[
        str | None,
        typer.Option(
            help="Native S2S engine spec; 'reference' (default without a system) is a "
            "scripted mock that makes exactly the expected calls: a harness check"
        ),
    ] = None,
    stt: Annotated[str | None, typer.Option(help="Cascade STT spec")] = None,
    llm: Annotated[
        str | None, typer.Option(help="Cascade LLM spec, e.g. ollama/qwen3.5:4b")
    ] = None,
    tts: Annotated[str | None, typer.Option(help="Cascade TTS spec")] = None,
    vad: Annotated[str | None, typer.Option(help="Cascade VAD spec")] = None,
    turn_detector: Annotated[
        str | None, typer.Option("--turn", help="Cascade turn detector spec")
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Agent config (YAML/TOML/JSON)")
    ] = None,
    preset: Annotated[
        str | None, typer.Option(help="Start from a preset (van presets), e.g. local-cpu")
    ] = None,
    scenarios: Annotated[
        str, typer.Option("--scenarios", "-s", help="Built-in suite (smoke) or a YAML file")
    ] = "smoke",
    only: Annotated[
        list[str] | None, typer.Option("--only", help="Run only this scenario id (repeatable)")
    ] = None,
    trials: Annotated[
        int, typer.Option("--trials", "-k", min=1, help="Calls per scenario (pass^k)")
    ] = 1,
    caller_tts: Annotated[
        str | None,
        typer.Option(
            help="TTS spec that voices the caller, or 'synthetic' (default: synthetic for the "
            "reference engine, else the suite's TTS or Kokoro)"
        ),
    ] = None,
    tool_delay_scale: Annotated[
        float, typer.Option(min=0.0, help="Multiply every mock tool's delay (0 = instant)")
    ] = 1.0,
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
    """T6: tool use on scripted spoken calls with deterministic mock tools.

    pass@1 / pass^k (final database state + facts said), tool precision / recall / F1,
    argument and entity accuracy, unnecessary calls, say-do violations, hallucinated tool
    results, tool-round latency on the recording, fillers and turns to completion.

    Examples:

        van bench tools                                   # scripted reference (harness check)

        van bench tools --preset local-cpu                # the local CPU cascade

        van bench tools --preset local-cpu --llm ollama/qwen3.5:4b -k 3

        van bench tools -c agent.yaml -s my-scenarios.yaml --only book-table
    """
    from ..bench.caller import TurnTiming
    from ..bench.onset import OnsetDetector, make_reference_vad
    from ..bench.system import BenchSystem, parse_component_spec
    from ..bench.tool_env import load_tool_suite, reference_engine
    from ..bench.tracks.tools import (
        ToolScenarioItem,
        ToolsOptions,
        run_tools_benchmark,
        tools_markdown_table,
    )
    from ..errors import ConfigurationError, VoiceAgentError
    from ..presets import get_preset
    from .bench import _logging, _print_notes, err

    _logging(verbose)
    reference = engine == "reference" or not any(
        x is not None for x in (engine, stt, llm, tts, turn_detector, config, preset)
    )
    try:
        if config is not None and preset is not None:
            raise ConfigurationError("pass either --config or --preset (a config can `extends:`)")
        system = BenchSystem.from_options(
            config=dict(get_preset(preset).config) if preset is not None else config,
            engine=None if reference else parse_component_spec(engine),
            stt=parse_component_spec(stt),
            llm=parse_component_spec(llm),
            tts=parse_component_spec(tts),
            vad=parse_component_spec(vad),
            turn_detector=parse_component_spec(turn_detector),
            label=label or ("reference" if reference else None),
            default_engine="mock",
        )
        suite = load_tool_suite(scenarios)
        if caller_tts is not None:
            voice = parse_component_spec(caller_tts)
            if isinstance(voice, list):
                raise ConfigurationError("--caller-tts takes one TTS spec")
            suite = suite.with_caller(voice)
        elif reference:
            suite = suite.with_caller("synthetic")
        elif suite.stimuli == "synthetic":
            suite = suite.with_caller(DEFAULT_CALLER_TTS)
        detector = OnsetDetector(make_reference_vad(reference_vad))
        options = ToolsOptions(
            trials=trials,
            scenarios=tuple(only or ()),
            tool_delay_scale=tool_delay_scale,
            reply_timeout=reply_timeout,
            save_audio=audio,
        )
        options.validate()
    except (VoiceAgentError, ValueError) as exc:
        err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2) from exc
    caller = "synthetic" if suite.tts is None else json.dumps(suite.tts)
    err.print(
        f"[bold]van bench tools[/bold] · {system.label} · {suite.name} · caller {caller}",
        highlight=False,
    )

    def on_turn(scenario: str, trial: int, turn: TurnTiming) -> None:
        heard = (
            "-"
            if turn.reply_start is None
            else (f"~{(turn.reply_start - turn.speech_end) * 1000:,.0f} ms")
        )
        err.print(
            f"  {scenario:<18} trial {trial + 1} turn {turn.index + 1} {heard}", highlight=False
        )

    def on_scenario(item: ToolScenarioItem) -> None:
        verdict = "[green]pass[/green]" if item.passed else "[red]FAIL[/red]"
        err.print(
            f"  {item.scenario:<18} trial {item.trial + 1} {verdict} · calls {item.calls} "
            f"· unneeded {item.unnecessary_calls}",
            highlight=False,
        )

    try:
        results = asyncio.run(
            run_tools_benchmark(
                system,
                suite,
                options,
                out_dir=out,
                run_id=run_id,
                detector=detector,
                engine_factory=reference_engine if reference else None,
                on_turn=on_turn,
                on_scenario=on_scenario,
            )
        )
    except VoiceAgentError as exc:
        err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc
    if as_json:
        typer.echo(json.dumps(results.summary.model_dump(mode="json"), indent=2))
    else:
        typer.echo(tools_markdown_table(results))
        _print_notes(results)
