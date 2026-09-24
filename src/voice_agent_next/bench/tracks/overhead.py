"""T7 framework-overhead track: how much the runtime adds on top of its components.

Research note 06, §8.3 (T7). Every section runs offline on mock components whose delays
are known, so what is left is the framework:

* ``e2e`` — the T1 harness (real-time caller, loopback transport with real-time
  playout, stereo recording, reference-VAD onsets) runs mock systems. Per user turn::

      overhead_ms = v2v_ms - injected_ms

  ``v2v_ms`` is measured on the call recording exactly as in T1; ``injected_ms`` is the
  latency the configuration prescribes on the critical path (:func:`injected_budget`):
  the VAD's end-of-speech confirmation in whole windows, endpointing, STT/LLM/TTS delays
  and the engine's response delay. Per session also: ``frame_jitter_ms`` (standard
  deviation of the gaps between consecutive agent frames of a reply as played by the
  transport), event-loop lag (:class:`~voice_agent_next.bench.probes.LoopLagProbe`),
  CPU time and resident memory.
* ``flush`` — the application interrupts every reply ``interrupt_after`` seconds into
  its playback (``AgentSession.interrupt()``). ``flush_ms`` = interrupt decision -> end
  of the last agent audio played (cut by the playback clear); frames that start playing
  after the clear are counted as leaked.
* ``capacity`` — N concurrent sessions in one process (one event loop, i.e. one core)
  on one shared engine, N doubling until the overhead p95 or the event-loop lag p99
  exceeds its threshold (default 50 ms each) or a turn is missed: ``sessions_per_core``
  is the largest N that passed. CPU share and memory per session come from the steps.
* ``micro`` — per-operation cost of the hot paths
  (:mod:`voice_agent_next.bench.microbench`).

Results use the shared schema (``manifest.json``, ``items.jsonl``, ``summary.json``,
``report.md``); :mod:`voice_agent_next.bench.gate` compares a run with a baseline.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import heapq
import itertools
import json
import math
import os
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ...engine import S2SEngine
from ...engines.cascade import CascadeEngine
from ...providers.energy import EnergyVAD
from ...providers.mock import MockEngine, MockLLM, MockSTT, MockTTS
from ...session import AgentSession, AgentState
from ...session.events import AgentStateChanged
from ...transports.loopback import LoopbackTransport, PlayedAudio
from ...utils.aio import cancel_and_wait
from ...utils.clock import now
from ..caller import _sleep_until
from ..environment import collect_environment
from ..microbench import MicroResult, run_micro_benchmarks
from ..onset import OnsetDetector
from ..probes import LoopLagProbe, cpu_seconds, rss_kind
from ..report import ReportSpec, fmt, markdown_table, render_report
from ..results import (
    Distribution,
    RunManifest,
    RunResults,
    RunSummary,
    json_safe,
    new_run_id,
    utc_timestamp,
    write_run,
)
from ..stats import percentile
from ..stimuli import Scenario, Stimulus, render_stimuli
from ..system import BenchSystem, redact

# package-internal building blocks of the T1 harness (one simulated call + its analysis)
from .latency import (
    LatencyItem,
    LatencyOptions,
    _analyze_session,
    _dataset_id,
    _reserve_directory,
    _run_session,
    _SessionRun,
)

__all__ = [
    "BUILTIN_CONDITIONS",
    "OVERHEAD_SCENARIO",
    "SECTIONS",
    "TIERS",
    "TRACK",
    "CapacityStepItem",
    "InjectedBudget",
    "InterruptItem",
    "OverheadCondition",
    "OverheadOptions",
    "SessionUsageItem",
    "SharedClock",
    "TurnOverheadItem",
    "flush_duration",
    "frame_gaps",
    "injected_budget",
    "jitter",
    "overhead_markdown_summary",
    "render_overhead_report",
    "run_overhead_benchmark",
    "sessions_per_core",
    "vad_confirmation",
]

TRACK = "overhead"
SECTIONS: tuple[str, ...] = ("micro", "e2e", "flush", "capacity")
Section = Literal["micro", "e2e", "flush", "capacity"]

SHORT_REPLY = "Okay."
"""Reply of the e2e and capacity sections (0.33 s of mock speech)."""
LONG_REPLY = "Okay, let me look that up for you right now, it will only take a moment."
"""Reply of the flush section: long enough to be interrupted mid-playback."""

OVERHEAD_SCENARIO: dict[str, Any] = {
    "name": "overhead-smoke",
    "version": 1,
    "description": (
        "T7: short synthetic utterances whose durations are whole 20 ms chunks (= energy "
        "VAD windows), so the end of speech falls on a VAD window boundary and the "
        "injected-delay budget is exact."
    ),
    "sample_rate": 16_000,
    "chunk": 0.02,
    "loudness_dbfs": -20.0,
    "lead_in": 0.3,
    "stimuli": "synthetic",
    "reply_timeout": 3.0,
    "gap_after_reply": 0.15,
    "turns": [
        {"id": "u1", "duration": 0.4},
        {"id": "u2", "duration": 0.5},
        {"id": "u3", "duration": 0.6},
        {"id": "u4", "duration": 0.44},
    ],
}


# ------------------------------------------------------------------------ conditions


@dataclass(frozen=True)
class OverheadCondition:
    """A mock system with known delays: an :class:`~voice_agent_next.config.AppConfig`
    mapping (``engine: ...`` or cascade components)."""

    name: str
    description: str
    config: Mapping[str, Any]

    def system(self, *, reply: str, replies: int) -> BenchSystem:
        """The system, scripted to answer every turn with ``reply``."""
        cfg = copy.deepcopy(dict(self.config))
        key = "engine" if cfg.get("engine") is not None else "llm"
        spec = cfg.get(key)
        spec = {"provider": spec} if isinstance(spec, str) else dict(spec or {})
        spec.setdefault("responses", [reply] * replies)
        cfg[key] = spec
        return BenchSystem.from_options(config=cfg, label=self.name, default_engine=None)


BUILTIN_CONDITIONS: dict[str, OverheadCondition] = {
    c.name: c
    for c in (
        OverheadCondition(
            "engine",
            "MockEngine (native speech-to-speech), no injected delay",
            {"engine": {"provider": "mock"}},
        ),
        OverheadCondition(
            "engine-delay",
            "MockEngine, 250 ms response delay",
            {"engine": {"provider": "mock", "response_delay": 0.25}},
        ),
        OverheadCondition(
            "cascade",
            "mock STT + LLM + TTS cascade with the energy VAD, no injected delay",
            {"stt": "mock", "llm": "mock", "tts": "mock", "vad": "energy"},
        ),
        OverheadCondition(
            "cascade-delay",
            "cascade: STT 100 ms, LLM TTFT 150 ms, TTS TTFB 100 ms, endpointing 200 ms",
            {
                "stt": {"provider": "mock", "latency": 0.1},
                "llm": {"provider": "mock", "ttft": 0.15},
                "tts": {"provider": "mock", "ttfb": 0.1},
                "vad": "energy",
                "cascade": {"min_endpointing_delay": 0.2},
            },
        ),
    )
}


# --------------------------------------------------------------------------- budget


@dataclass(frozen=True)
class InjectedBudget:
    """The latency a mock configuration prescribes between the end of user speech and
    the first agent audio (critical path)."""

    parts: dict[str, float]
    """Seconds per component on the critical path."""
    formula: str
    details: dict[str, float] = field(default_factory=dict)
    """Configuration the parts were derived from (seconds)."""

    @property
    def total_ms(self) -> float:
        return 1000.0 * sum(self.parts.values())

    def describe(self) -> dict[str, Any]:
        return {
            "injected_ms": round(self.total_ms, 3),
            "formula": self.formula,
            "parts_ms": {k: round(v * 1000.0, 3) for k, v in self.parts.items()},
            "details_s": self.details,
        }


def vad_confirmation(min_silence: float, window: float) -> float:
    """Silence a :class:`~voice_agent_next.vad.VADStream` needs before it reports the end of
    speech: whole windows, accumulated exactly like the stream does (0.25 s of 20 ms
    windows is 0.26 s)."""
    if window <= 0:
        raise ValueError("window must be > 0")
    acc, n = 0.0, 0
    while True:
        acc += window
        n += 1
        if acc >= min_silence:
            return n * window


def injected_budget(engine: S2SEngine) -> InjectedBudget:
    """Critical-path delays of a :class:`~voice_agent_next.providers.mock.MockEngine` or of
    a cascade of mocks (``MockSTT`` streaming, ``MockLLM``, ``MockTTS``, a VAD, no turn
    detector). Raises ``ValueError`` for anything else: only mocks have known delays.

    * MockEngine: ``VAD silence (whole windows) + response_delay``;
    * cascade: ``max(min_endpointing_delay, VAD silence + STT latency) + LLM TTFT +
      TTS TTFB`` — the endpointing wait runs from the end of speech, concurrently with the
      VAD's silence confirmation and the STT flush.
    """
    if isinstance(engine, MockEngine):
        if engine.llm.token_delay:
            raise ValueError("MockEngine token_delay is not part of the overhead budget")
        window = EnergyVAD(
            sample_rate=engine.input_sample_rate, options=engine.vad_options
        ).window_duration
        silence = vad_confirmation(engine.vad_options.min_silence_duration, window)
        return InjectedBudget(
            {"vad_silence": silence, "response_delay": engine.response_delay},
            "VAD silence (whole windows) + response_delay",
            {
                "vad_window": window,
                "vad_min_silence": engine.vad_options.min_silence_duration,
                "response_delay": engine.response_delay,
            },
        )
    if isinstance(engine, CascadeEngine):
        vad, stt, llm, tts = engine.vad, engine.stt, engine.llm, engine.tts
        if (
            vad is None
            or engine.turn_detector is not None
            or not isinstance(stt, MockSTT)
            or not isinstance(llm, MockLLM)
            or not isinstance(tts, MockTTS)
        ):
            raise ValueError(
                "the overhead budget needs a cascade of MockSTT (streaming), MockLLM and "
                "MockTTS with a VAD and without a turn detector"
            )
        if llm.token_delay:
            raise ValueError("MockLLM token_delay is not part of the overhead budget")
        min_delay = engine.options.min_endpointing_delay
        min_delay = 0.6 if min_delay is None else min_delay  # VAD-only default
        silence = vad_confirmation(vad.options.min_silence_duration, vad.window_duration)
        return InjectedBudget(
            {
                "endpointing": max(min_delay, silence + stt.latency),
                "llm_ttft": llm.ttft,
                "tts_ttfb": tts.ttfb,
            },
            "max(min_endpointing_delay, VAD silence + STT latency) + LLM TTFT + TTS TTFB",
            {
                "min_endpointing_delay": min_delay,
                "vad_window": vad.window_duration,
                "vad_min_silence": vad.options.min_silence_duration,
                "vad_silence": silence,
                "stt_latency": stt.latency,
                "llm_ttft": llm.ttft,
                "tts_ttfb": tts.ttfb,
            },
        )
    raise ValueError(
        f"no injected-delay budget for {type(engine).__name__}: the overhead track needs "
        "mock components with known delays"
    )


# ------------------------------------------------------------------------- analysis


def frame_gaps(
    played: Sequence[PlayedAudio], start: float = -math.inf, end: float = math.inf
) -> list[float]:
    """Gaps (s) between consecutive agent frames that started playing in ``[start, end)``:
    ``start[i+1] - (start[i] + duration[i])``, 0 for seamless playback."""
    frames = sorted((p for p in played if start <= p.start_time < end), key=lambda p: p.start_time)
    return [b.start_time - (a.start_time + a.frame.duration) for a, b in itertools.pairwise(frames)]


def jitter(gaps: Sequence[float]) -> float | None:
    """Sample standard deviation of ``gaps`` (``None`` for fewer than two)."""
    if len(gaps) < 2:
        return None
    return float(np.std(np.asarray(gaps, dtype=np.float64), ddof=1))


def flush_duration(
    played: Sequence[PlayedAudio],
    clear_times: Sequence[float],
    decision: float,
    window_end: float = math.inf,
) -> tuple[float | None, int]:
    """``(flush seconds, leaked frames)`` for an interrupt decided at ``decision``.

    Flush = end of the last agent audio heard after the decision - decision. Audio playing
    when the transport cleared playback is cut at the clear; frames that *start* after
    the clear (before ``window_end``, the next user turn) are leaks. ``None`` when no
    playback clear followed the decision.
    """
    clear = next((c for c in sorted(clear_times) if c >= decision), None)
    if clear is None:
        return None, 0
    last_end, leaked = decision, 0
    for p in played:
        begin, finish = p.start_time, p.start_time + p.frame.duration
        if begin >= window_end or finish <= decision:
            continue
        if begin < clear:
            finish = min(finish, clear)
        else:
            leaked += 1
        last_end = max(last_end, finish)
    return last_end - decision, leaked


# --------------------------------------------------------------------------- items


class TurnOverheadItem(BaseModel):
    """One user turn of an e2e session (a line of ``items.jsonl``)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["turn"] = "turn"
    condition: str
    session: int
    turn: int
    stimulus: str
    warmup: bool = False
    v2v_ms: float | None = None
    injected_ms: float
    overhead_ms: float | None = None
    """``v2v_ms - injected_ms``."""
    session_v2v_ms: float | None = None
    residual_ms: float | None = None
    eou_delay_ms: float | None = None
    response_ttfb_ms: float | None = None
    frames: int = 0
    """Agent frames played for this turn's reply."""
    frame_jitter_ms: float | None = None
    frame_gap_max_ms: float | None = None
    missed: bool = False
    errors: list[str] = Field(default_factory=list)


class SessionUsageItem(BaseModel):
    """Resource usage of one session (e2e / flush)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["session"] = "session"
    section: str
    condition: str
    session: int
    wall_s: float
    cpu_ms: float
    cpu_pct: float
    """CPU time of the process while the session ran / wall time (100 = one core)."""
    turns: int
    loop_lag_p50_ms: float | None = None
    loop_lag_p99_ms: float | None = None
    loop_lag_max_ms: float | None = None
    delivery_lag_p99_ms: float | None = None
    """How late the simulated caller delivered its audio chunks (harness)."""
    rss_peak_mb: float | None = None


class InterruptItem(BaseModel):
    """One ``session.interrupt()`` of the flush section."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["interrupt"] = "interrupt"
    condition: str
    session: int
    turn: int | None = None
    warmup: bool = False
    played_ms: float | None = None
    """Reply audio played before the decision."""
    flush_ms: float | None = None
    interrupt_call_ms: float
    """Duration of the ``await session.interrupt()`` call (engine cancel included)."""
    leaked_frames: int = 0


class CapacityStepItem(BaseModel):
    """One step (N concurrent sessions) of the capacity sweep."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["capacity_step"] = "capacity_step"
    sessions: int
    turns_measured: int
    overhead_p50_ms: float | None = None
    overhead_p95_ms: float | None = None
    overhead_max_ms: float | None = None
    loop_lag_p99_ms: float | None = None
    delivery_lag_p99_ms: float | None = None
    cpu_pct: float
    cpu_pct_per_session: float
    rss_peak_mb: float | None = None
    wall_s: float
    missed: int = 0
    errors: int = 0
    passed: bool
    reasons: list[str] = Field(default_factory=list)


def sessions_per_core(steps: Iterable[CapacityStepItem]) -> tuple[int, bool]:
    """``(largest N that passed, whether a failing N was found)``; 0 if none passed."""
    steps = list(steps)
    passed = [s.sessions for s in steps if s.passed]
    return (max(passed, default=0), any(not s.passed for s in steps))


# -------------------------------------------------------------------------- options

TIERS: dict[str, dict[str, Any]] = {
    "smoke": {},
    "full": {
        "turns": 36,
        "sessions": 3,
        "flush_turns": 21,
        "capacity_max_sessions": 512,
        "capacity_turns": 5,
        "capacity_refine": 3,
        "micro_repeats": 25,
        "micro_min_time": 0.05,
    },
}
"""Presets: ``smoke`` (CI gate, ~3 min) and ``full`` (publishable numbers: >= 100
measured turns over 3 sessions per condition)."""


@dataclass
class OverheadOptions:
    """How the overhead track runs (defaults: the ``smoke`` tier)."""

    tier: str = "smoke"
    sections: tuple[str, ...] = SECTIONS
    conditions: tuple[str, ...] = ("engine", "engine-delay", "cascade", "cascade-delay")
    """e2e conditions (names of :data:`BUILTIN_CONDITIONS` or of custom conditions)."""
    turns: int = 7
    """User turns per e2e session, warm-up included."""
    sessions: int = 1
    """e2e sessions per condition (sequential, on one warmed-up engine)."""
    warmup_turns: int = 1
    """Leading turns of every session excluded from the statistics."""
    flush_conditions: tuple[str, ...] = ("engine", "cascade")
    flush_turns: int = 6
    """Interrupted replies per flush condition, warm-up included."""
    interrupt_after: float = 0.2
    """Seconds of reply playback before the application interrupts it."""
    capacity_condition: str = "engine"
    capacity_max_sessions: int = 32
    capacity_turns: int = 3
    """User turns per session in every capacity step, warm-up included."""
    capacity_refine: int = 0
    """Bisection steps between the last passing and the first failing N."""
    capacity_stagger: float = 0.5
    """Session starts are spread over this many seconds (no thundering herd)."""
    max_overhead_p95_ms: float = 50.0
    max_lag_p99_ms: float = 50.0
    micro_repeats: int = 9
    micro_min_time: float = 0.02
    micro_only: tuple[str, ...] | None = None
    lag_interval: float = 0.01
    seed: int = 0
    bootstrap_resamples: int = 2000

    @classmethod
    def for_tier(cls, tier: str = "smoke", **overrides: Any) -> OverheadOptions:
        """Options of a tier preset; ``None`` overrides are ignored."""
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}; tiers: {', '.join(TIERS)}")
        values = {**TIERS[tier], **{k: v for k, v in overrides.items() if v is not None}}
        return cls(tier=tier, **values)

    def validate(self, conditions: Mapping[str, OverheadCondition]) -> None:
        unknown = [s for s in self.sections if s not in SECTIONS]
        if unknown or not self.sections:
            raise ValueError(f"sections must be a subset of {', '.join(SECTIONS)}: {unknown}")
        names = {*self.conditions, *self.flush_conditions, self.capacity_condition}
        missing = sorted(n for n in names if n not in conditions)
        if missing:
            raise ValueError(
                f"unknown condition(s): {', '.join(missing)}; known: {', '.join(conditions)}"
            )
        if "e2e" in self.sections and not self.conditions:
            raise ValueError("the e2e section needs at least one condition")
        if self.turns <= self.warmup_turns or self.flush_turns <= self.warmup_turns:
            raise ValueError("turns and flush_turns must exceed warmup_turns")
        if self.capacity_turns <= self.warmup_turns:
            raise ValueError("capacity_turns must exceed warmup_turns")
        if self.sessions < 1 or self.capacity_max_sessions < 1 or self.warmup_turns < 0:
            raise ValueError("sessions and capacity_max_sessions must be >= 1")
        if self.interrupt_after <= 0 or self.lag_interval <= 0:
            raise ValueError("interrupt_after and lag_interval must be > 0")

    def config_sha256(self, conditions: Mapping[str, OverheadCondition], scenario: Scenario) -> str:
        """Hash of every setting that shapes the numbers (not the sections or the tier)."""
        used = sorted({*self.conditions, *self.flush_conditions, self.capacity_condition})
        payload = {
            "options": {
                k: v for k, v in asdict(self).items() if k not in ("tier", "sections", "seed")
            },
            "conditions": {n: dict(conditions[n].config) for n in used if n in conditions},
            "scenario": scenario.definition_sha256(),
            "replies": [SHORT_REPLY, LONG_REPLY],
        }
        data = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------- shared clock


class SharedClock:
    """One precise timer for many simulated callers.

    ``CallerEmulator`` normally waits for every chunk with its own precise sleep, which
    needs a worker thread wherever asyncio timers wake early (Windows). With hundreds of
    concurrent callers those threads would throttle the *harness*; this clock coalesces
    all deadlines on a ``resolution`` grid and wakes them with one timer, never early
    (at most ``resolution`` late).
    """

    def __init__(self, resolution: float = 0.001) -> None:
        if resolution <= 0:
            raise ValueError("resolution must be > 0")
        self.resolution = resolution
        self._heap: list[tuple[float, int, asyncio.Future[None]]] = []
        self._seq = itertools.count()
        self._changed = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="bench-shared-clock")

    async def stop(self) -> None:
        await cancel_and_wait(self._task)
        self._task = None
        for _, _, fut in self._heap:
            fut.cancel()
        self._heap.clear()

    async def sleep_until(self, deadline: float) -> None:
        """Return once ``now() >= deadline``."""
        if deadline <= now():
            await asyncio.sleep(0)
            return
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._heap, (deadline, next(self._seq), fut))
        self._changed.set()
        await fut

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(0)  # callers woken last round register their next deadline
            if not self._heap:
                self._changed.clear()
                await self._changed.wait()
                continue
            target = math.ceil(self._heap[0][0] / self.resolution) * self.resolution
            await _sleep_until(target)
            t = now()
            while self._heap and self._heap[0][0] <= t:
                _, _, fut = heapq.heappop(self._heap)
                if not fut.done():
                    fut.set_result(None)


# -------------------------------------------------------------------------- helpers


def _ms(seconds: float | None, digits: int = 3) -> float | None:
    return None if seconds is None else round(seconds * 1000.0, digits)


def _pct(values: Sequence[float], q: float) -> float | None:
    return float(percentile(values, q)) if values else None


def _mb(value: int | None) -> float | None:
    return None if value is None else round(value / 2**20, 2)


@dataclass
class _Section:
    items: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Distribution] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    rates: dict[str, float | None] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    headline: int = 0
    rss_peak: int | None = None


@dataclass
class _Ctx:
    options: OverheadOptions
    conditions: Mapping[str, OverheadCondition]
    scenario: Scenario
    detector: OnsetDetector
    progress: Callable[[str], None]
    stimuli: dict[int, list[Stimulus]] = field(default_factory=dict)
    systems: dict[str, dict[str, Any]] = field(default_factory=dict)

    def dist(self, values: Iterable[float | None], ci: Sequence[str] | None = None) -> Distribution:
        o = self.options
        if ci is None:
            return Distribution.of(values, seed=o.seed, n_resamples=o.bootstrap_resamples)
        return Distribution.of(values, ci=ci, seed=o.seed, n_resamples=o.bootstrap_resamples)

    async def stimuli_for(self, turns: int) -> list[Stimulus]:
        if turns not in self.stimuli:
            self.stimuli[turns] = await render_stimuli(self.scenario, turns=turns)
        return self.stimuli[turns]

    def latency_options(self, turns: int) -> LatencyOptions:
        o = self.options
        return LatencyOptions(
            turns=turns, warmup_turns=o.warmup_turns, save_audio=False, warmup_engine=False,
            seed=o.seed, bootstrap_resamples=o.bootstrap_resamples,
        )  # fmt: skip

    async def engine_for(
        self, name: str, *, reply: str, replies: int
    ) -> tuple[BenchSystem, S2SEngine, InjectedBudget]:
        cond = self.conditions[name]
        system = cond.system(reply=reply, replies=replies)
        engine = system.build_engine()
        try:
            budget = injected_budget(engine)
            await engine.warmup()
        except BaseException:
            await engine.aclose()
            raise
        if name not in self.systems:
            self.systems[name] = {
                "description": cond.description,
                "config": redact(dict(cond.config)),
                "system": system.describe(engine),
                "budget": budget.describe(),
            }
        return system, engine, budget


async def _measured_session(
    ctx: _Ctx,
    index: int,
    engine: S2SEngine,
    system: BenchSystem,
    stimuli: Sequence[Stimulus],
    opts: LatencyOptions,
    **hooks: Any,
) -> tuple[_SessionRun, LoopLagProbe, float, float]:
    """One call with a lag probe running; ``(run, probe, cpu seconds, wall seconds)``."""
    probe = LoopLagProbe(interval=ctx.options.lag_interval)
    probe.start()
    cpu0, t0 = cpu_seconds(), now()
    try:
        run = await _run_session(index, engine, system, stimuli, ctx.scenario, opts, None, **hooks)
    finally:
        cpu, wall = cpu_seconds() - cpu0, now() - t0
        await probe.stop()
    return run, probe, cpu, wall


def _usage_item(
    section: str, condition: str, run: _SessionRun, probe: LoopLagProbe, cpu: float, wall: float
) -> SessionUsageItem:
    lags = probe.samples
    push = run.call.push_lag
    return SessionUsageItem(
        section=section,
        condition=condition,
        session=run.index,
        wall_s=round(wall, 3),
        cpu_ms=round(cpu * 1000.0, 3),
        cpu_pct=round(100.0 * cpu / wall, 3) if wall > 0 else 0.0,
        turns=len(run.call.turns),
        loop_lag_p50_ms=_ms(_pct(lags, 50)),
        loop_lag_p99_ms=_ms(_pct(lags, 99)),
        loop_lag_max_ms=_ms(max(lags)) if lags else None,
        delivery_lag_p99_ms=_ms(_pct(push, 99)),
        rss_peak_mb=_mb(probe.rss_peak),
    )


def _turn_windows(turns: Sequence[Any]) -> list[tuple[float, float]]:
    """``[speech start of turn k, speech start of turn k+1)`` on the ``now()`` clock."""
    return [
        (t.speech_start, turns[k + 1].speech_start if k + 1 < len(turns) else math.inf)
        for k, t in enumerate(turns)
    ]


def _max_rss(*values: int | None) -> int | None:
    present = [v for v in values if v is not None]
    return max(present) if present else None


# ---------------------------------------------------------------------------- micro


async def _run_micro(ctx: _Ctx) -> _Section:
    o = ctx.options
    ctx.progress(f"micro: {o.micro_repeats} batches per hot path")
    # blocking on purpose: nothing else runs, and a worker thread would add GIL hand-offs
    results: list[MicroResult] = run_micro_benchmarks(
        repeats=o.micro_repeats, min_time=o.micro_min_time, only=o.micro_only
    )
    out = _Section()
    for r in results:
        out.items.append(r.to_dict())
        out.metrics[f"micro.{r.name}_us"] = Distribution.of(
            r.per_op_us, ci=("p50",), seed=o.seed, n_resamples=o.bootstrap_resamples, digits=4
        )
    out.counts["micro_benchmarks"] = len(results)
    out.extra["micro"] = {
        "repeats": o.micro_repeats,
        "min_time_s": o.micro_min_time,
        "benchmarks": [
            {k: v for k, v in r.to_dict().items() if k not in ("kind", "per_op_us")}
            for r in results
        ],
    }
    return out


# ------------------------------------------------------------------------------ e2e


async def _run_e2e(ctx: _Ctx) -> _Section:
    o = ctx.options
    out = _Section()
    stimuli = await ctx.stimuli_for(o.turns)
    opts = ctx.latency_options(o.turns)
    replies = o.turns * o.sessions + 8
    all_lags: list[float] = []
    all_push: list[float] = []
    conditions: dict[str, Any] = {}
    for name in o.conditions:
        system, engine, budget = await ctx.engine_for(name, reply=SHORT_REPLY, replies=replies)
        injected = budget.total_ms
        turns: list[TurnOverheadItem] = []
        usage: list[SessionUsageItem] = []
        lags: list[float] = []
        try:
            for s in range(o.sessions):
                ctx.progress(f"e2e {name}: session {s + 1}/{o.sessions}, {o.turns} turns")
                run, probe, cpu, wall = await _measured_session(
                    ctx, s, engine, system, stimuli, opts
                )
                usage.append(_usage_item("e2e", name, run, probe, cpu, wall))
                lags += probe.samples
                all_push += run.call.push_lag
                out.rss_peak = _max_rss(out.rss_peak, probe.rss_peak)
                analysis = await asyncio.to_thread(_analyze_session, run, ctx.detector, opts)
                played = run.transport.played_log if run.transport is not None else []
                windows = _turn_windows(run.call.turns)
                for item, (w0, w1) in zip(analysis.items, windows, strict=True):
                    turns.append(_turn_item(name, item, injected, played, w0, w1))
        finally:
            await engine.aclose()
        all_lags += lags
        measured = [t for t in turns if not t.warmup]
        overhead = [t.overhead_ms for t in measured]
        out.metrics[f"e2e.{name}.v2v_ms"] = ctx.dist(t.v2v_ms for t in measured)
        out.metrics[f"e2e.{name}.overhead_ms"] = ctx.dist(overhead)
        out.metrics[f"e2e.{name}.residual_ms"] = ctx.dist(t.residual_ms for t in measured)
        out.metrics[f"e2e.{name}.frame_jitter_ms"] = ctx.dist(t.frame_jitter_ms for t in measured)
        out.metrics[f"e2e.{name}.cpu_pct"] = ctx.dist(u.cpu_pct for u in usage)
        conditions[name] = {
            **budget.describe(),
            "description": ctx.conditions[name].description,
            "turns_measured": len(measured),
            "replies": sum(t.overhead_ms is not None for t in measured),
            "missed": sum(t.missed for t in measured),
            "loop_lag_p99_ms": _ms(_pct(lags, 99)),
            "cpu_pct": round(float(np.mean([u.cpu_pct for u in usage])), 3),
            "cpu_ms_per_turn": round(sum(u.cpu_ms for u in usage) / max(1, len(turns)), 3),
            "rss_peak_mb": max((u.rss_peak_mb or 0.0 for u in usage), default=None),
        }
        out.items += [t.model_dump(mode="json") for t in turns]
        out.items += [u.model_dump(mode="json") for u in usage]
    measured_all = [
        TurnOverheadItem.model_validate(i)
        for i in out.items
        if i["kind"] == "turn" and not i["warmup"]
    ]
    out.headline = sum(t.overhead_ms is not None for t in measured_all)
    out.metrics["e2e.overhead_ms"] = ctx.dist(t.overhead_ms for t in measured_all)
    out.metrics["e2e.frame_jitter_ms"] = ctx.dist(t.frame_jitter_ms for t in measured_all)
    out.metrics["e2e.frame_gap_max_ms"] = ctx.dist(t.frame_gap_max_ms for t in measured_all)
    out.metrics["e2e.loop_lag_ms"] = ctx.dist((v * 1000.0 for v in all_lags), ci=("p50", "p99"))
    out.metrics["e2e.delivery_lag_ms"] = ctx.dist((v * 1000.0 for v in all_push), ci=("p50", "p99"))
    missed = sum(t.missed for t in measured_all)
    out.counts.update(
        {
            "e2e_turns": sum(1 for i in out.items if i["kind"] == "turn"),
            "e2e_turns_measured": len(measured_all),
            "e2e_replies": out.headline,
            "e2e_missed": missed,
            "errors": sum(len(t.errors) for t in measured_all),
        }
    )
    out.rates["e2e_missed_rate"] = round(missed / len(measured_all), 6) if measured_all else None
    out.extra["e2e"] = {"conditions": conditions, "turns": o.turns, "sessions": o.sessions}
    if missed:
        out.notes.append(f"{missed} e2e turn(s) got no reply: their overhead is missing.")
    return out


def _turn_item(
    condition: str,
    item: LatencyItem,
    injected: float,
    played: Sequence[PlayedAudio],
    w0: float,
    w1: float,
) -> TurnOverheadItem:
    gaps = frame_gaps(played, w0, w1)
    j = jitter(gaps)
    return TurnOverheadItem(
        condition=condition,
        session=item.session,
        turn=item.turn,
        stimulus=item.stimulus,
        warmup=item.warmup,
        v2v_ms=item.v2v_ms,
        injected_ms=round(injected, 3),
        overhead_ms=None if item.v2v_ms is None else round(item.v2v_ms - injected, 3),
        session_v2v_ms=item.session_v2v_ms,
        residual_ms=item.residual_ms,
        eou_delay_ms=item.eou_delay_ms,
        response_ttfb_ms=item.response_ttfb_ms,
        frames=sum(w0 <= p.start_time < w1 for p in played),
        frame_jitter_ms=_ms(j, 4),
        frame_gap_max_ms=_ms(max(gaps), 4) if gaps else None,
        missed=item.missed,
        errors=list(item.errors),
    )


# ---------------------------------------------------------------------------- flush


@dataclass(slots=True)
class _Interrupt:
    decision: float
    returned: float


async def _interrupted_session(
    ctx: _Ctx, name: str
) -> tuple[_SessionRun, LoopLagProbe, float, float, list[_Interrupt]]:
    """One call whose every reply the application interrupts ``interrupt_after`` s in."""
    o = ctx.options
    stimuli = await ctx.stimuli_for(o.flush_turns)
    opts = ctx.latency_options(o.flush_turns)
    system, engine, _ = await ctx.engine_for(name, reply=LONG_REPLY, replies=o.flush_turns + 8)
    interrupts: list[_Interrupt] = []
    tasks: set[asyncio.Task[None]] = set()

    async def interrupt_later(session: AgentSession) -> None:
        await asyncio.sleep(o.interrupt_after)
        if session.agent_state != AgentState.SPEAKING:
            return  # the reply already ended
        decision = now()
        await session.interrupt()
        interrupts.append(_Interrupt(decision, now()))

    def on_start(session: AgentSession, transport: LoopbackTransport) -> None:
        def on_state(ev: AgentStateChanged) -> None:
            if ev.new_state == AgentState.SPEAKING:
                task = asyncio.create_task(interrupt_later(session))
                tasks.add(task)
                task.add_done_callback(tasks.discard)

        session.on("agent_state_changed", on_state)

    try:
        run, probe, cpu, wall = await _measured_session(
            ctx, 0, engine, system, stimuli, opts, on_start=on_start
        )
    finally:
        await cancel_and_wait(*tasks)
        await engine.aclose()
    return run, probe, cpu, wall, interrupts


async def _run_flush(ctx: _Ctx) -> _Section:
    o = ctx.options
    out = _Section()
    conditions: dict[str, Any] = {}
    for name in o.flush_conditions:
        ctx.progress(f"flush {name}: {o.flush_turns} interrupted replies")
        run, probe, cpu, wall, interrupts = await _interrupted_session(ctx, name)
        out.rss_peak = _max_rss(out.rss_peak, probe.rss_peak)
        usage = _usage_item("flush", name, run, probe, cpu, wall)
        items = _interrupt_items(name, run, interrupts, o.warmup_turns)
        measured = [i for i in items if not i.warmup]
        out.metrics[f"flush.{name}.flush_ms"] = ctx.dist(i.flush_ms for i in measured)
        conditions[name] = {
            "interrupts": len(measured),
            "leaked_frames": sum(i.leaked_frames for i in measured),
            "interrupt_call_p50_ms": _pct([i.interrupt_call_ms for i in measured], 50),
            "cpu_pct": usage.cpu_pct,
        }
        out.items += [i.model_dump(mode="json") for i in items]
        out.items.append(usage.model_dump(mode="json"))
    measured_all = [
        InterruptItem.model_validate(i)
        for i in out.items
        if i["kind"] == "interrupt" and not i["warmup"]
    ]
    out.metrics["flush.flush_ms"] = ctx.dist(i.flush_ms for i in measured_all)
    out.metrics["flush.interrupt_call_ms"] = ctx.dist(i.interrupt_call_ms for i in measured_all)
    leaked = sum(i.leaked_frames > 0 for i in measured_all)
    expected = len(o.flush_conditions) * (o.flush_turns - o.warmup_turns)
    out.counts.update(
        {
            "flush_interrupts": len(measured_all),
            "flush_leaked_frames": sum(i.leaked_frames for i in measured_all),
        }
    )
    out.rates["flush_leak_rate"] = round(leaked / len(measured_all), 6) if measured_all else None
    out.extra["flush"] = {
        "interrupt_after_ms": round(o.interrupt_after * 1000.0, 3),
        "conditions": conditions,
    }
    if len(measured_all) < expected:
        out.notes.append(
            f"Only {len(measured_all)} of {expected} replies were interrupted mid-playback."
        )
    if leaked:
        out.notes.append(f"{leaked} interrupt(s) let agent frames start after the flush.")
    return out


def _interrupt_items(
    condition: str, run: _SessionRun, interrupts: Sequence[_Interrupt], warmup_turns: int
) -> list[InterruptItem]:
    transport = run.transport
    played = transport.played_log if transport is not None else []
    clears = transport.clear_times if transport is not None else []
    windows = _turn_windows(run.call.turns)
    items: list[InterruptItem] = []
    for it in interrupts:
        turn = next((k for k, (a, b) in enumerate(windows) if a <= it.decision < b), None)
        w_end = windows[turn][1] if turn is not None else math.inf
        reply = run.call.turns[turn].reply_start if turn is not None else None
        flush, leaked = flush_duration(played, clears, it.decision, w_end)
        items.append(
            InterruptItem(
                condition=condition,
                session=run.index,
                turn=turn,
                warmup=turn is None or turn < warmup_turns,
                played_ms=_ms(it.decision - reply) if reply is not None else None,
                flush_ms=_ms(flush, 4),
                interrupt_call_ms=round((it.returned - it.decision) * 1000.0, 4),
                leaked_frames=leaked,
            )
        )
    return items


# ------------------------------------------------------------------------- capacity


async def _capacity_step(ctx: _Ctx, n: int) -> tuple[CapacityStepItem, int | None]:
    o = ctx.options
    stimuli = await ctx.stimuli_for(o.capacity_turns)
    opts = ctx.latency_options(o.capacity_turns)
    system, engine, budget = await ctx.engine_for(
        o.capacity_condition, reply=SHORT_REPLY, replies=n * o.capacity_turns + 8
    )
    clock = SharedClock()
    stagger = o.capacity_stagger / n

    async def call(i: int) -> _SessionRun:
        await asyncio.sleep(i * stagger)
        return await _run_session(
            i, engine, system, stimuli, ctx.scenario, opts, None, sleep_until=clock.sleep_until
        )

    probe = LoopLagProbe(interval=o.lag_interval)
    clock.start()
    probe.start()
    cpu0, t0 = cpu_seconds(), now()
    try:
        outcomes = await asyncio.gather(*(call(i) for i in range(n)), return_exceptions=True)
    finally:
        cpu, wall = cpu_seconds() - cpu0, now() - t0
        await probe.stop()
        await clock.stop()
        await engine.aclose()
    runs = [r for r in outcomes if isinstance(r, _SessionRun)]
    failures = [r for r in outcomes if not isinstance(r, _SessionRun)]
    for exc in failures:
        if not isinstance(exc, Exception):
            raise exc  # cancellation, KeyboardInterrupt...
    overhead: list[float] = []
    missed = errors = 0
    push: list[float] = []
    for run in runs:
        analysis = await asyncio.to_thread(_analyze_session, run, ctx.detector, opts)
        push += run.call.push_lag
        for item in analysis.items:
            if item.warmup or not item.expect_reply:
                continue
            missed += item.missed
            errors += len(item.errors)
            if item.v2v_ms is not None:
                overhead.append(item.v2v_ms - budget.total_ms)
    lag_p99 = _ms(_pct(probe.samples, 99))
    p95 = _pct(overhead, 95)
    reasons: list[str] = []
    if failures:
        reasons.append(f"{len(failures)} session(s) failed: {failures[0]!r}")
    if not overhead:
        reasons.append("no measured turns")
    elif p95 is not None and p95 > o.max_overhead_p95_ms:
        reasons.append(f"overhead p95 {p95:.1f} ms > {o.max_overhead_p95_ms:g} ms")
    if lag_p99 is not None and lag_p99 > o.max_lag_p99_ms:
        reasons.append(f"loop lag p99 {lag_p99:.1f} ms > {o.max_lag_p99_ms:g} ms")
    if missed:
        reasons.append(f"{missed} missed turn(s)")
    cpu_pct = 100.0 * cpu / wall if wall > 0 else 0.0
    step = CapacityStepItem(
        sessions=n,
        turns_measured=len(overhead),
        overhead_p50_ms=None if not overhead else round(float(np.median(overhead)), 3),
        overhead_p95_ms=None if p95 is None else round(p95, 3),
        overhead_max_ms=None if not overhead else round(max(overhead), 3),
        loop_lag_p99_ms=lag_p99,
        delivery_lag_p99_ms=_ms(_pct(push, 99)),
        cpu_pct=round(cpu_pct, 3),
        cpu_pct_per_session=round(cpu_pct / n, 4),
        rss_peak_mb=_mb(probe.rss_peak),
        wall_s=round(wall, 3),
        missed=missed,
        errors=errors + len(failures),
        passed=not reasons,
        reasons=reasons,
    )
    return step, probe.rss_peak


def _rss_slope(steps: Sequence[CapacityStepItem]) -> float | None:
    """Least-squares MB per additional session over the steps' peak RSS."""
    pts = [(s.sessions, s.rss_peak_mb) for s in steps if s.rss_peak_mb is not None]
    if len({n for n, _ in pts}) < 2:
        return None
    x = np.asarray([p[0] for p in pts], dtype=np.float64)
    y = np.asarray([p[1] for p in pts], dtype=np.float64)
    slope = float(np.polyfit(x, y, 1)[0])
    return round(max(slope, 0.0), 4)


async def _run_capacity(ctx: _Ctx) -> _Section:
    o = ctx.options
    out = _Section()
    steps: list[CapacityStepItem] = []

    async def step(n: int) -> bool:
        ctx.progress(f"capacity: {n} concurrent session(s)")
        item, rss = await _capacity_step(ctx, n)
        steps.append(item)
        out.rss_peak = _max_rss(out.rss_peak, rss)
        ctx.progress(
            f"capacity: {n} session(s) -> overhead p95 {fmt(item.overhead_p95_ms, 1)} ms, "
            f"loop lag p99 {fmt(item.loop_lag_p99_ms, 1)} ms, CPU {item.cpu_pct:.0f} % "
            f"-> {'pass' if item.passed else 'FAIL: ' + '; '.join(item.reasons)}"
        )
        return item.passed

    n, last_pass, first_fail = 1, 0, None
    while n <= o.capacity_max_sessions:
        if await step(n):
            last_pass = n
            n *= 2
        else:
            first_fail = n
            break
    if first_fail is None and last_pass < o.capacity_max_sessions:
        if await step(o.capacity_max_sessions):
            last_pass = o.capacity_max_sessions
        else:
            first_fail = o.capacity_max_sessions
    lo, hi = last_pass, first_fail
    for _ in range(o.capacity_refine):
        if hi is None or hi - lo <= 1 or lo < 1:
            break
        mid = (lo + hi) // 2
        if await step(mid):
            lo = mid
        else:
            hi = mid
    steps.sort(key=lambda s: s.sessions)
    best, limit_found = sessions_per_core(steps)
    at_best = next((s for s in steps if s.sessions == best), None)
    out.items += [s.model_dump(mode="json") for s in steps]
    out.counts["capacity_steps"] = len(steps)
    out.extra["capacity"] = {
        "condition": o.capacity_condition,
        "sessions_per_core": best,
        "limit_found": limit_found,
        "max_sessions": o.capacity_max_sessions,
        "max_overhead_p95_ms": o.max_overhead_p95_ms,
        "max_lag_p99_ms": o.max_lag_p99_ms,
        "turns_per_session": o.capacity_turns,
        "cpu_pct_per_session": None if at_best is None else at_best.cpu_pct_per_session,
        "rss_per_session_mb": _rss_slope(steps),
        "steps": [s.model_dump(mode="json", exclude={"kind"}) for s in steps],
    }
    if best == 0:
        out.notes.append("Capacity: even a single session exceeded the thresholds.")
    return out


# ------------------------------------------------------------------------- summary

_METRIC_LABELS = {
    "e2e.overhead_ms": "**overhead** `v2v − injected`, all e2e conditions",
    "e2e.frame_jitter_ms": "frame jitter (std of playout gaps per reply)",
    "e2e.frame_gap_max_ms": "largest playout gap per reply",
    "e2e.loop_lag_ms": "event-loop lag (e2e sessions)",
    "e2e.delivery_lag_ms": "caller chunk delivery lag (harness)",
    "flush.flush_ms": "**flush** interrupt → last agent audio",
    "flush.interrupt_call_ms": "`session.interrupt()` call",
}

_METHOD = """\
* **overhead_ms** = `v2v_ms − injected_ms` per user turn. `v2v_ms` is measured on the
  call recording exactly like T1 (annotated end of user speech → first 10 ms frame that
  begins ≥ 100 ms of agent speech, RMS reference VAD, sample-refined). `injected_ms` is
  the critical-path latency the mock configuration prescribes: VAD end-of-speech
  confirmation in whole windows, endpointing, STT latency, LLM TTFT, TTS TTFB and the
  engine's response delay (see the formula per condition). What remains is the framework
  (event loop hops, session and transport) plus asyncio timer lateness.
* **frame_jitter_ms**: standard deviation of `start[i+1] − end[i]` over consecutive agent
  frames of one reply, as played by the loopback transport (0 = seamless).
* **event-loop lag**: a probe sleeps {lag_ms:g} ms and records how late it wakes (other
  callbacks + timer granularity; ~16 ms on Windows with Python < 3.13).
* **flush_ms**: the app calls `session.interrupt()` {interrupt_ms:g} ms into every reply;
  flush = decision → end of the last agent audio played (audio playing is cut by the
  playback clear); frames starting after the clear count as leaked.
* **capacity**: N concurrent sessions in one process (one event loop ≈ one core) on one
  engine, N doubling; a step passes if overhead p95 ≤ {max_overhead:g} ms, loop lag
  p99 ≤ {max_lag:g} ms and no turn is missed. Callers share one timer (≤ 1 ms delivery
  quantization). CPU % = process CPU time / wall time (100 % = one core, harness
  included).
* **micro**: median per-operation cost over {repeats} timed batches (garbage collector
  off); % of real time = cost / audio covered by one operation.
* The first {warmup} turn(s) of every session are excluded. Percentiles are
  linear-interpolated; `[..]` is the 95 % percentile-bootstrap CI of the median.
* Transport: in-process loopback with real-time playout. Smoke numbers are regression
  canaries, not capability scores; compare them with a baseline from the same kind of
  machine only.
"""


def _metric_cell(d: Distribution | None, stat: str = "p50", digits: int = 2) -> str:
    if d is None or not d.n:
        return "–"
    value = getattr(d, stat)
    ci = d.ci95.get(stat)
    text = fmt(value, digits)
    if ci is not None and d.n > 1 and stat == "p50":
        text += f" [{fmt(ci[0], digits)}, {fmt(ci[1], digits)}]"
    return text


def _e2e_table(results: RunResults) -> str | None:
    e2e = results.summary.extra.get("e2e")
    if not e2e:
        return None
    m = results.summary.metrics
    rows = []
    for name, info in e2e["conditions"].items():
        rows.append(
            [
                f"`{name}`",
                info.get("formula", ""),
                fmt(info.get("injected_ms"), 0),
                _metric_cell(m.get(f"e2e.{name}.v2v_ms"), digits=1),
                _metric_cell(m.get(f"e2e.{name}.overhead_ms")),
                _metric_cell(m.get(f"e2e.{name}.overhead_ms"), "p90"),
                _metric_cell(m.get(f"e2e.{name}.overhead_ms"), "p99"),
                _metric_cell(m.get(f"e2e.{name}.residual_ms"), digits=3),
                _metric_cell(m.get(f"e2e.{name}.frame_jitter_ms"), digits=3),
                fmt(info.get("loop_lag_p99_ms"), 2),
                fmt(info.get("cpu_pct"), 1),
                fmt(info.get("turns_measured")),
            ]
        )
    headers = [
        "condition", "injected budget", "injected ms", "v2v p50", "overhead p50 [95% CI]",
        "p90", "p99", "residual p50", "jitter p50", "loop lag p99", "CPU %", "turns",
    ]  # fmt: skip
    return markdown_table(headers, rows, ["l", "l"] + ["r"] * 10)


def _flush_table(results: RunResults) -> str | None:
    flush = results.summary.extra.get("flush")
    if not flush:
        return None
    m = results.summary.metrics
    rows = [
        [
            f"`{name}`",
            fmt(info.get("interrupts")),
            _metric_cell(m.get(f"flush.{name}.flush_ms"), digits=3),
            _metric_cell(m.get(f"flush.{name}.flush_ms"), "p99", digits=3),
            _metric_cell(m.get(f"flush.{name}.flush_ms"), "max", digits=3),
            fmt(info.get("interrupt_call_p50_ms"), 3),
            fmt(info.get("leaked_frames")),
        ]
        for name, info in flush["conditions"].items()
    ]
    headers = ["condition", "interrupts", "flush p50 ms", "p99", "max", "interrupt() p50 ms",
               "leaked frames"]  # fmt: skip
    return markdown_table(headers, rows, ["l"] + ["r"] * 6)


def _capacity_text(results: RunResults) -> str | None:
    cap = results.summary.extra.get("capacity")
    if not cap:
        return None
    best = cap["sessions_per_core"]
    bound = "" if cap["limit_found"] else f" (limit not reached; tested up to {best})"
    lines = [
        f"**sessions per core: {best}**{bound} — condition `{cap['condition']}`, "
        f"{cap['turns_per_session']} turns per session; CPU per session "
        f"{fmt(cap.get('cpu_pct_per_session'), 2)} %; memory per session "
        f"{fmt(cap.get('rss_per_session_mb'), 2)} MB (RSS slope).",
        "",
    ]
    rows = [
        [
            fmt(s["sessions"]),
            fmt(s["turns_measured"]),
            fmt(s.get("overhead_p50_ms"), 1),
            fmt(s.get("overhead_p95_ms"), 1),
            fmt(s.get("loop_lag_p99_ms"), 1),
            fmt(s.get("delivery_lag_p99_ms"), 1),
            fmt(s.get("cpu_pct"), 0),
            fmt(s.get("rss_peak_mb"), 0),
            "pass" if s["passed"] else "fail: " + "; ".join(s["reasons"]),
        ]
        for s in cap["steps"]
    ]
    headers = ["sessions", "turns", "overhead p50", "p95", "loop lag p99", "delivery p99",
               "CPU %", "RSS MB", "result"]  # fmt: skip
    lines.append(markdown_table(headers, rows, ["r"] * 8 + ["l"]))
    return "\n".join(lines)


def _micro_table(results: RunResults, limit: int | None = None) -> str | None:
    micro = results.summary.extra.get("micro")
    if not micro:
        return None
    rows = []
    for b in micro["benchmarks"][:limit]:
        info = b.get("info") or {}
        rows.append(
            [
                f"`{b['name']}`",
                b["description"],
                b["unit"],
                fmt(b["median_us"], 3),
                fmt(b["min_us"], 3),
                fmt(b["iqr_us"], 3),
                "–" if b.get("budget_pct") is None else f"{b['budget_pct']:.3f} %",
                ", ".join(f"{k}={v}" for k, v in info.items()),
            ]
        )
    headers = ["benchmark", "operation", "per", "median µs", "min µs", "IQR µs",
               "% of real time", "notes"]  # fmt: skip
    return markdown_table(headers, rows, ["l", "l", "l", "r", "r", "r", "r", "l"])


def overhead_report_spec(results: RunResults) -> ReportSpec:
    opts = results.manifest.options
    method = _METHOD.format(
        lag_ms=1000.0 * float(opts.get("lag_interval", 0.01)),
        interrupt_ms=1000.0 * float(opts.get("interrupt_after", 0.2)),
        max_overhead=float(opts.get("max_overhead_p95_ms", 50.0)),
        max_lag=float(opts.get("max_lag_p99_ms", 50.0)),
        repeats=opts.get("micro_repeats", 9),
        warmup=opts.get("warmup_turns", 1),
    )
    sections: list[tuple[str, str]] = []
    for heading, body in (
        ("End-to-end overhead (ms)", _e2e_table(results)),
        ("Flush on interrupt", _flush_table(results)),
        ("Capacity: concurrent sessions per core", _capacity_text(results)),
        ("Micro-benchmarks (hot paths)", _micro_table(results)),
    ):
        if body:
            sections.append((heading, body))
    sections.append(("Method", method))
    tier = opts.get("tier", "custom")
    return ReportSpec(
        title=f"Framework overhead (T7) · {tier} tier",
        metric_labels=_METRIC_LABELS,
        rate_labels={
            "e2e_missed_rate": "e2e turns without a reply",
            "flush_leak_rate": "interrupts with frames played after the flush",
        },
        sections=sections,
        digits=2,
    )


def render_overhead_report(results: RunResults) -> str:
    return render_report(results, overhead_report_spec(results))


def overhead_markdown_summary(results: RunResults, *, heading: str = "###") -> str:
    """Compact Markdown of a run for a CI job summary (e2e, flush, capacity, micro)."""
    s = results.summary
    tier = results.manifest.options.get("tier", "custom")
    lines = [f"{heading} Framework overhead (T7) · {tier} tier · `{s.run_id}`", ""]
    headline = s.metrics.get("e2e.overhead_ms")
    if headline is not None and headline.n:
        lines += [
            f"The runtime adds **{fmt(headline.p50, 2)} ms** (p50; p90 {fmt(headline.p90, 2)}, "
            f"p99 {fmt(headline.p99, 2)} ms) on top of the injected component delays "
            f"over {headline.n} turns.",
            "",
        ]
    for title, body in (
        ("End-to-end overhead (ms)", _e2e_table(results)),
        ("Flush on interrupt", _flush_table(results)),
        ("Capacity", _capacity_text(results)),
    ):
        if body:
            lines += [f"**{title}**", "", body, ""]
    micro = _micro_table(results)
    if micro:
        lines += [
            "<details><summary>Micro-benchmarks (µs per operation)</summary>",
            "",
            micro,
            "",
            "</details>",
            "",
        ]
    lines += [f"> {note}" for note in results.manifest.notes]
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- entry


def _merge(sections: Sequence[_Section]) -> _Section:
    out = _Section()
    for s in sections:
        out.items += s.items
        out.metrics.update(s.metrics)
        for k, v in s.counts.items():
            out.counts[k] = out.counts.get(k, 0) + v
        out.rates.update(s.rates)
        out.extra.update(s.extra)
        out.notes += s.notes
        out.headline += s.headline
        out.rss_peak = _max_rss(out.rss_peak, s.rss_peak)
    return out


async def run_overhead_benchmark(
    options: OverheadOptions | None = None,
    *,
    conditions: Mapping[str, OverheadCondition] | None = None,
    scenario: Scenario | None = None,
    out_dir: str | os.PathLike[str] | None = None,
    run_id: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> RunResults:
    """Run the T7 overhead track and (if ``out_dir``) write ``<out_dir>/<run_id>/``.

    Args:
        options: sections, conditions, turns, thresholds... (default: the smoke tier).
        conditions: available conditions by name (default: :data:`BUILTIN_CONDITIONS`).
        scenario: user turns (default: :data:`OVERHEAD_SCENARIO`).
        out_dir: parent of the run directory (``None``: nothing is written).
        run_id: defaults to ``<UTC time>-overhead-<tier>``.
        progress: called with a short line whenever a step starts or ends.
    """
    options = options or OverheadOptions()
    conditions = dict(conditions or BUILTIN_CONDITIONS)
    options.validate(conditions)
    scenario = scenario or Scenario.model_validate(OVERHEAD_SCENARIO)
    final_id = run_id or new_run_id(TRACK, options.tier)
    directory: Path | None = None
    if out_dir is not None:
        directory = _reserve_directory(Path(out_dir), final_id, unique=run_id is None)
        final_id = directory.name
    try:
        return await _run_overhead(
            options, conditions, scenario, directory=directory, run_id=final_id,
            progress=progress or (lambda _line: None),
        )  # fmt: skip
    except BaseException:
        if directory is not None and not any(directory.iterdir()):
            directory.rmdir()
        raise


async def _run_overhead(
    options: OverheadOptions,
    conditions: Mapping[str, OverheadCondition],
    scenario: Scenario,
    *,
    directory: Path | None,
    run_id: str,
    progress: Callable[[str], None],
) -> RunResults:
    created = utc_timestamp()
    t_start = now()
    ctx = _Ctx(options, conditions, scenario, OnsetDetector(), progress)
    runners: dict[str, Callable[[_Ctx], Awaitable[_Section]]] = {
        "micro": _run_micro, "e2e": _run_e2e, "flush": _run_flush, "capacity": _run_capacity,
    }  # fmt: skip
    # fixed order: the micro-benchmarks first, on a quiet process
    merged = _merge([await runners[name](ctx) for name in SECTIONS if name in options.sections])
    config_hash = options.config_sha256(conditions, scenario)
    merged.extra.update(
        {
            "tier": options.tier,
            "sections": list(options.sections),
            "config_sha256": config_hash,
            "rss_kind": rss_kind(),
            "rss_peak_mb": _mb(merged.rss_peak),
        }
    )
    stimuli = next(iter(ctx.stimuli.values()), [])
    unique: dict[str, Stimulus] = {}
    for stim in stimuli:
        unique.setdefault(stim.id, stim)
    manifest = RunManifest(
        run_id=run_id,
        track=TRACK,
        created=created,
        system=json_safe({"conditions": ctx.systems}),
        scenario={
            "name": scenario.name,
            "version": scenario.version,
            "sha256": scenario.definition_sha256(),
            "definition": scenario.model_dump(mode="json"),
            "stimuli": [s.describe() for s in unique.values()],
            "replies": {"short": SHORT_REPLY, "long": LONG_REPLY},
        },
        transport={
            "type": "loopback",
            "realtime_playout": True,
            "chunk_ms": round(scenario.chunk * 1000, 3),
            "delivery": "each chunk when its interval has elapsed (capture-device model)",
        },
        options=json_safe(
            {
                **asdict(options),
                "config_sha256": config_hash,
                "chunk_ms": round(scenario.chunk * 1000, 3),
                "onset": ctx.detector.describe(),
            }
        ),
        environment=await asyncio.to_thread(collect_environment),
        notes=merged.notes,
    )
    labels = ", ".join(dict.fromkeys([*options.conditions, *options.flush_conditions]))
    summary = RunSummary(
        run_id=run_id,
        track=TRACK,
        system=f"mocks ({labels})" if labels else "mocks",
        transport="loopback",
        dataset=_dataset_id(scenario, stimuli) if stimuli else f"{scenario.name}@none",
        n=merged.headline,
        metrics=merged.metrics,
        rates=merged.rates,
        counts=merged.counts,
        extra=json_safe(merged.extra),
        duration_s=round(now() - t_start, 3),
    )
    results = RunResults(manifest, json_safe(merged.items), summary)
    results.report = render_overhead_report(results)
    if directory is not None:
        write_run(directory, results)
    return results
