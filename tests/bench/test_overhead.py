"""T7 framework-overhead track: budget math, playout analysis, probes and a tiny smoke run.

The smoke run is real time (~12 s) and shared by the tests at the bottom of this module.
Timing assertions are generous: CI runners are slow and Windows timers are coarse.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from voice_agent_next.audio import AudioFrame
from voice_agent_next.bench import Scenario, TurnSpec, load_run
from voice_agent_next.bench.microbench import run_micro_benchmarks
from voice_agent_next.bench.probes import LoopLagProbe, cpu_seconds, rss_bytes, rss_kind
from voice_agent_next.bench.results import REPORT_FILE, RunResults
from voice_agent_next.bench.tracks.overhead import (
    BUILTIN_CONDITIONS,
    CapacityStepItem,
    InterruptItem,
    OverheadOptions,
    SessionUsageItem,
    SharedClock,
    TurnOverheadItem,
    flush_duration,
    frame_gaps,
    injected_budget,
    jitter,
    run_overhead_benchmark,
    sessions_per_core,
    vad_confirmation,
)
from voice_agent_next.cli.main import app
from voice_agent_next.engine import EngineCapabilities, EngineConnection, EngineOptions, S2SEngine
from voice_agent_next.engines.cascade import CascadeEngine, CascadeOptions
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import (
    MockEngine,
    MockLLM,
    MockSTT,
    MockTTS,
    MockTurnDetector,
)
from voice_agent_next.transports.loopback import PlayedAudio
from voice_agent_next.utils import now

runner = CliRunner()
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    return ANSI.sub("", text)


def played(*spans: tuple[float, float]) -> list[PlayedAudio]:
    """``(start, duration)`` pairs as loopback playout records (24 kHz)."""
    return [PlayedAudio(AudioFrame.silence(d, 24_000), s) for s, d in spans]


# ---------------------------------------------------------------------- budget


def test_vad_confirmation_counts_whole_windows() -> None:
    assert vad_confirmation(0.4, 0.02) == pytest.approx(0.4)
    assert vad_confirmation(0.25, 0.02) == pytest.approx(0.26)  # 13 windows
    assert vad_confirmation(0.0, 0.02) == pytest.approx(0.02)
    with pytest.raises(ValueError):
        vad_confirmation(0.1, 0.0)


def build(name: str) -> S2SEngine:
    return BUILTIN_CONDITIONS[name].system(reply="Okay.", replies=1).build_engine()


def test_injected_budget_of_the_builtin_conditions() -> None:
    engine = injected_budget(build("engine"))
    assert engine.total_ms == pytest.approx(400.0)
    delayed = injected_budget(build("engine-delay"))
    assert delayed.parts == pytest.approx({"vad_silence": 0.4, "response_delay": 0.25})
    assert injected_budget(build("cascade")).total_ms == pytest.approx(600.0)  # VAD-only 0.6 s
    cascade = injected_budget(build("cascade-delay"))
    # endpointing: VAD silence 0.26 s + STT 0.1 s outlasts the 0.2 s minimum delay
    assert cascade.parts == pytest.approx({"endpointing": 0.36, "llm_ttft": 0.15, "tts_ttfb": 0.1})
    assert cascade.describe()["injected_ms"] == pytest.approx(610.0)
    # a fast STT hides behind the minimum endpointing delay
    slow_floor = CascadeEngine(
        stt=MockSTT(latency=0.05), llm=MockLLM(ttft=0.1), tts=MockTTS(), vad=EnergyVAD(),
        options=CascadeOptions(min_endpointing_delay=0.5),
    )  # fmt: skip
    assert injected_budget(slow_floor).parts["endpointing"] == pytest.approx(0.5)


class _OtherEngine(S2SEngine):
    async def connect(self, options: EngineOptions) -> EngineConnection:
        raise NotImplementedError


def test_injected_budget_needs_known_delays() -> None:
    with pytest.raises(ValueError, match="turn detector"):
        injected_budget(
            CascadeEngine(stt=MockSTT(), llm=MockLLM(), tts=MockTTS(), vad=EnergyVAD(),
                          turn_detector=MockTurnDetector(probability=1.0))
        )  # fmt: skip
    with pytest.raises(ValueError, match="token_delay"):
        injected_budget(MockEngine(token_delay=0.1))
    other = _OtherEngine(
        model="x", capabilities=EngineCapabilities(), input_sample_rate=16_000,
        output_sample_rate=24_000,
    )  # fmt: skip
    with pytest.raises(ValueError, match="mock components"):
        injected_budget(other)


# -------------------------------------------------------------------- analysis


def test_frame_gaps_and_jitter() -> None:
    log = played((0.0, 0.04), (0.04, 0.04), (0.0801, 0.04), (0.12, 0.04), (5.0, 0.04))
    gaps = frame_gaps(log, 0.0, 1.0)  # the frame at 5 s belongs to another reply
    assert gaps == pytest.approx([0.0, 0.0001, -0.0001])
    assert jitter(gaps) == pytest.approx(0.0001)
    assert frame_gaps(log, 4.0) == []
    assert jitter([]) is None and jitter([0.1]) is None


def test_flush_duration_cuts_at_the_clear_and_counts_leaks() -> None:
    # the reply's second frame was playing when the app interrupted (50 ms); the next
    # reply starts at 3 s, after the next user turn (window end 2 s)
    log = played((0.0, 0.04), (0.04, 0.04), (3.0, 0.04))
    # playback cleared 0.1 ms after the decision: the frame playing is cut there
    flush, leaked = flush_duration(log, [0.0501], 0.05, window_end=2.0)
    assert flush == pytest.approx(0.0001) and leaked == 0
    # a frame that starts after the clear leaks through
    leaky = [*log, *played((0.06, 0.04))]
    flush, leaked = flush_duration(leaky, [0.0501], 0.05, window_end=2.0)
    assert flush == pytest.approx(0.05) and leaked == 1
    assert flush_duration(log, [0.01], 0.05) == (None, 0)  # no clear after the decision
    assert flush_duration([], [0.06], 0.05) == (0.0, 0)  # nothing was playing


def step(n: int, passed: bool) -> CapacityStepItem:
    return CapacityStepItem(sessions=n, turns_measured=n, cpu_pct=n, cpu_pct_per_session=1.0,
                            wall_s=1.0, passed=passed)  # fmt: skip


def test_sessions_per_core_is_the_largest_passing_step() -> None:
    assert sessions_per_core([step(1, True), step(2, True), step(4, False), step(3, True)]) == (
        3,
        True,
    )
    assert sessions_per_core([step(1, True), step(2, True)]) == (2, False)
    assert sessions_per_core([step(1, False)]) == (0, True)


# ---------------------------------------------------------------------- probes


async def test_loop_lag_probe_sees_a_blocked_loop() -> None:
    async with LoopLagProbe(interval=0.005, rss_every=1) as probe:
        await asyncio.sleep(0.03)
        time.sleep(0.08)  # noqa: ASYNC251 - blocking the loop on purpose
        await asyncio.sleep(0.03)
    assert probe.samples and max(probe.samples) >= 0.03
    if rss_kind() != "unavailable":
        assert probe.rss_peak is not None and probe.rss_peak > 1_000_000
    with pytest.raises(ValueError):
        LoopLagProbe(interval=0)


def test_cpu_time_and_rss() -> None:
    start = cpu_seconds()
    deadline = time.perf_counter() + 0.05
    while time.perf_counter() < deadline:
        pass
    assert cpu_seconds() > start
    rss = rss_bytes()
    assert rss_kind() in ("current", "peak", "unavailable")
    assert (rss is None) == (rss_kind() == "unavailable")
    if rss is not None:
        assert rss > 1_000_000


async def test_shared_clock_coalesces_deadlines_and_never_wakes_early() -> None:
    clock = SharedClock(resolution=0.002)
    clock.start()

    async def wake(deadline: float) -> float:
        await clock.sleep_until(deadline)
        return now() - deadline

    try:
        base = now()
        late = await asyncio.gather(*(wake(base + 0.01 + i * 0.0007) for i in range(10)))
        assert await wake(now() - 1.0) >= 0  # past deadlines return at once
    finally:
        await clock.stop()
    assert min(late) >= 0.0
    assert max(late) < 0.1  # generous: Windows timers


# ----------------------------------------------------------------------- micro


def test_micro_benchmarks_report_per_operation_costs() -> None:
    results = run_micro_benchmarks(repeats=2, min_time=0.001)
    names = [r.name for r in results]
    for expected in ("resample_16k_to_24k", "resample_16k_to_24k_numpy", "frame_rms",
                     "energy_vad", "silence_trimmer", "g711_ulaw_encode", "g711_alaw_decode",
                     "sentence_segmenter", "event_emit", "chan_roundtrip"):  # fmt: skip
        assert expected in names
    for r in results:
        assert len(r.per_op_us) == 2 and r.batch >= 1
        assert 0 < r.min_us <= r.median_us and r.iqr_us >= 0
        assert (r.budget_pct is None) == (r.frame_ms is None)
        data = r.to_dict()
        assert data["kind"] == "micro" and data["median_us"] > 0
    numpy = next(r for r in results if r.name == "resample_16k_to_24k_numpy")
    assert numpy.info == {"backend": "numpy"} and numpy.unit == "frame"
    only = run_micro_benchmarks(repeats=1, min_time=0.001, only=["event_emit"])
    assert [r.name for r in only] == ["event_emit"] and only[0].unit == "event"
    with pytest.raises(ValueError, match="unknown micro-benchmark"):
        run_micro_benchmarks(only=["nope"])


# --------------------------------------------------------------------- options


def test_tiers_validation_and_config_hash() -> None:
    smoke, full = OverheadOptions.for_tier("smoke"), OverheadOptions.for_tier("full")
    assert smoke == OverheadOptions() and full.turns == 36 and full.sessions == 3
    assert OverheadOptions.for_tier("smoke", turns=None).turns == smoke.turns
    with pytest.raises(ValueError, match="unknown tier"):
        OverheadOptions.for_tier("huge")
    bad: list[dict[str, Any]] = [
        {"sections": ("e2e", "nope")},
        {"conditions": ("engine", "warp-drive")},
        {"turns": 1},
        {"capacity_turns": 1},
        {"interrupt_after": 0.0},
    ]
    for overrides in bad:
        with pytest.raises(ValueError):
            OverheadOptions(**overrides).validate(BUILTIN_CONDITIONS)
    scenario = Scenario(name="s", turns=[TurnSpec(duration=0.4)])
    digest = smoke.config_sha256(BUILTIN_CONDITIONS, scenario)
    assert (
        OverheadOptions(sections=("micro",)).config_sha256(BUILTIN_CONDITIONS, scenario) == digest
    )
    assert OverheadOptions(turns=9).config_sha256(BUILTIN_CONDITIONS, scenario) != digest


# -------------------------------------------------------------------- smoke run

TINY = OverheadOptions(
    tier="smoke",
    conditions=("engine", "cascade"),
    turns=2,
    warmup_turns=0,  # keeps the real-time run short; the mocks need no warm-up turn
    flush_conditions=("engine",),
    flush_turns=1,
    capacity_max_sessions=2,
    capacity_turns=1,
    capacity_stagger=0.1,
    micro_repeats=2,
    micro_min_time=0.001,
    micro_only=("energy_vad", "event_emit"),
    bootstrap_resamples=200,
)
TINY_SCENARIO = Scenario(
    name="tiny",
    lead_in=0.2,
    gap_after_reply=0.1,
    reply_timeout=2.0,
    turns=[TurnSpec(id="a", duration=0.3), TurnSpec(id="b", duration=0.4)],
)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory: pytest.TempPathFactory) -> RunResults:
    out = tmp_path_factory.mktemp("overhead")
    lines: list[str] = []
    results = asyncio.run(
        run_overhead_benchmark(
            TINY, scenario=TINY_SCENARIO, out_dir=out, run_id="tiny", progress=lines.append
        )
    )
    assert any(line.startswith("capacity: 2 session(s)") for line in lines)
    return results


def test_smoke_run_measures_the_framework_overhead(smoke: RunResults) -> None:
    m, extra = smoke.summary.metrics, smoke.summary.extra
    for name, injected in (("engine", 400.0), ("cascade", 600.0)):
        v2v, overhead = m[f"e2e.{name}.v2v_ms"], m[f"e2e.{name}.overhead_ms"]
        assert overhead.n == 2 and v2v.p50 is not None and overhead.p50 is not None
        assert v2v.p50 == pytest.approx(injected, abs=60)
        assert -5.0 < overhead.p50 < 60.0
        assert v2v.p50 - overhead.p50 == pytest.approx(injected, abs=0.01)  # 3-digit rounding
        assert extra["e2e"]["conditions"][name]["injected_ms"] == pytest.approx(injected)
    assert m["e2e.overhead_ms"].n == 4 and smoke.summary.n == 4
    assert m["e2e.frame_jitter_ms"].n == 4 and (m["e2e.frame_jitter_ms"].p50 or 0) < 20.0
    assert m["e2e.loop_lag_ms"].n > 10 and "p99" in m["e2e.loop_lag_ms"].ci95
    flush = m["flush.flush_ms"]
    assert flush.n == 1 and flush.p50 is not None and 0.0 <= flush.p50 < 30.0
    assert smoke.summary.counts["flush_leaked_frames"] == 0
    assert smoke.summary.rates["e2e_missed_rate"] == 0.0
    capacity = extra["capacity"]
    assert capacity["sessions_per_core"] == 2 and not capacity["limit_found"]
    assert [s["sessions"] for s in capacity["steps"]] == [1, 2]
    assert set(m) >= {"micro.energy_vad_us", "micro.event_emit_us"}
    assert extra["sections"] == ["micro", "e2e", "flush", "capacity"]


def test_smoke_run_round_trips_through_the_result_files(smoke: RunResults) -> None:
    assert smoke.directory is not None and smoke.directory.name == "tiny"
    loaded = load_run(smoke.directory)
    assert loaded.summary == smoke.summary and loaded.manifest == smoke.manifest
    kinds: dict[str, int] = {}
    models = {"turn": TurnOverheadItem, "session": SessionUsageItem,
              "interrupt": InterruptItem, "capacity_step": CapacityStepItem}  # fmt: skip
    for item in loaded.items:
        kinds[item["kind"]] = kinds.get(item["kind"], 0) + 1
        if item["kind"] in models:
            models[item["kind"]].model_validate(item)
    assert kinds == {"micro": 2, "turn": 4, "session": 3, "interrupt": 1, "capacity_step": 2}
    options = loaded.manifest.options
    assert options["tier"] == "smoke" and len(options["config_sha256"]) == 64
    assert loaded.manifest.track == "overhead" and loaded.manifest.environment["python"]
    budget = loaded.manifest.system["conditions"]["cascade"]["budget"]
    assert budget["parts_ms"] == {"endpointing": 600.0, "llm_ttft": 0.0, "tts_ttfb": 0.0}
    report = (smoke.directory / REPORT_FILE).read_text(encoding="utf-8")
    for heading in ("# Framework overhead (T7)", "## End-to-end overhead", "## Flush on interrupt",
                    "## Capacity", "## Micro-benchmarks", "## Method"):  # fmt: skip
        assert heading in report


def test_cli_records_a_baseline_and_gates_a_run(smoke: RunResults, tmp_path: Path) -> None:
    assert smoke.directory is not None
    run_dir, base = str(smoke.directory), tmp_path / "baseline.json"
    recorded = runner.invoke(
        app, ["bench", "overhead", "--from-run", run_dir, "--update-baseline", "--baseline",
              str(base)],
    )  # fmt: skip
    assert recorded.exit_code == 0, recorded.output
    data = json.loads(base.read_text(encoding="utf-8"))
    (key,) = data["entries"]
    assert "e2e.engine.overhead_ms" in data["entries"][key]["metrics"]

    summary = tmp_path / "summary.md"
    gated = runner.invoke(
        app, ["bench", "overhead", "--from-run", run_dir, "--baseline", str(base),
              "--summary", str(summary)],
    )  # fmt: skip
    assert gated.exit_code == 0, gated.output
    assert "gate: PASS" in plain(gated.stdout)
    text = summary.read_text(encoding="utf-8")
    assert "Regression gate" in text and "**PASS**" in text and "Framework overhead" in text
    assert json.loads((smoke.directory / "gate.json").read_text(encoding="utf-8"))["passed"]

    # a baseline 100 ms faster than this run: the overhead regressed
    metrics = data["entries"][key]["metrics"]
    value = metrics["e2e.engine.overhead_ms"]["value"]
    metrics["e2e.engine.overhead_ms"].update(value=value - 100.0, ci95=None)
    base.write_text(json.dumps(data), encoding="utf-8")
    failed = runner.invoke(
        app, ["bench", "overhead", "--from-run", run_dir, "--baseline", str(base),
              "--summary", str(summary)],
    )  # fmt: skip
    assert failed.exit_code == 1, failed.output
    assert "FAIL" in plain(failed.stdout) and "regressed" in plain(failed.stdout)
    assert "FAIL — 1 regression(s)" in summary.read_text(encoding="utf-8")


def test_cli_rejects_bad_input_and_rerenders_reports(smoke: RunResults, tmp_path: Path) -> None:
    assert smoke.directory is not None
    missing = runner.invoke(
        app, ["bench", "overhead", "--from-run", str(smoke.directory), "--baseline",
              str(tmp_path / "none.json")],
    )  # fmt: skip
    assert missing.exit_code == 2 and "not found" in " ".join(plain(missing.stderr).split())
    bad = runner.invoke(app, ["bench", "overhead", "--sections", "e2e,warp"])
    assert bad.exit_code == 2 and "sections" in plain(bad.stderr)
    tier = runner.invoke(app, ["bench", "overhead", "--tier", "huge"])
    assert tier.exit_code == 2 and "unknown tier" in plain(tier.stderr)
    rendered = runner.invoke(app, ["bench", "report", str(smoke.directory), "--no-write"])
    assert rendered.exit_code == 0, rendered.output
    assert plain(rendered.stdout).startswith("# Framework overhead (T7) · smoke tier")
