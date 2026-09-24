"""T1 latency track: user-perceived voice-to-voice latency (research note 06, §8.3).

For every scripted user turn that expects a reply::

    v2v_ms = t_aon - t_uoff

* ``t_uoff`` — the annotated end of user speech (stimulus annotation, placed on the
  recording clock by the caller);
* ``t_aon`` — the agent onset: first 10 ms frame that begins >= 100 ms of speech on the
  agent channel of the recording, according to the reference VAD
  (:mod:`voice_agent_next.bench.onset`).

Reported next to it:

* ``session_v2v_ms`` — the session's own ``TurnMetrics.voice_to_voice`` (engine-reported
  end of speech -> first agent audio handed to the transport) and ``residual_ms`` =
  ``v2v_ms - session_v2v_ms``: what the in-process metric does not see (transport and
  playout buffering, end-of-speech estimation error, leading silence in the reply);
* spans: ``eou_delay_ms`` (end-of-turn delay), ``response_ttfb_ms`` (turn commit -> first
  audio) and component ``stt_latency_ms`` / ``llm_ttft_ms`` / ``tts_ttfb_ms`` /
  ``engine_ttfb_ms`` when the engine reports them;
* cold start: ``first_turn_v2v_ms`` (turn 0 of every session; with the default
  ``warmup_turns=1`` it is excluded from the headline), ``session_ready_ms`` and
  ``greeting_ms``;
* ``dead_air_rate`` — share of turns with ``v2v_ms`` above 2,000 ms (configurable) or
  without any reply; ``missed_rate``, ``premature_rate`` (the agent started before the
  user finished) and ``interrupted_rate``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import os
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from ...audio.frame import AudioFormat
from ...metrics import EngineMetrics, LLMMetrics, Metrics, STTMetrics, TTSMetrics, TurnMetrics
from ...session import AgentSession, AgentState
from ...transports.loopback import LoopbackTransport
from ...utils.clock import now
from ..caller import CallerEmulator, CallResult, TurnTiming
from ..environment import collect_environment
from ..onset import OnsetDetector, first_onset_between
from ..recording import Label, write_labels
from ..report import ReportSpec, render_report
from ..results import (
    ARTIFACTS_DIR,
    Distribution,
    RunManifest,
    RunResults,
    RunSummary,
    new_run_id,
    utc_timestamp,
    write_run,
)
from ..stats import percentile
from ..stimuli import Scenario, Stimulus, render_stimuli
from ..system import BenchSystem

__all__ = [
    "TRACK",
    "LatencyItem",
    "LatencyOptions",
    "latency_report_spec",
    "render_latency_report",
    "run_latency_benchmark",
    "summarize_latency",
]

TRACK = "latency"
T = TypeVar("T")
_AGENT_FORMAT = AudioFormat(24_000, 1)
"""Agent (playout) side of the loopback transport."""


@dataclass
class LatencyOptions:
    """How the latency track runs."""

    turns: int = 20
    """User turns per session (scenario turns are cycled)."""
    sessions: int = 1
    """Separate sessions (connections) on the same engine."""
    warmup_turns: int = 1
    """Leading turns of every session excluded from the headline (reported as cold start)."""
    dead_air_threshold: float = 2.0
    """Seconds of silence after the user stopped that count as dead air."""
    reply_timeout: float | None = None
    """Override the scenario's ``reply_timeout`` (s)."""
    gap_after_reply: float | None = None
    """Override the scenario's ``gap_after_reply`` (s)."""
    save_audio: bool = True
    """Write ``artifacts/session-NNN/stereo.wav`` and ``labels.txt``."""
    warmup_engine: bool = True
    """Call ``engine.warmup()`` (model loading...) before the first session."""
    seed: int = 0
    """Bootstrap seed."""
    bootstrap_resamples: int = 2000

    def timing(self, scenario: Scenario) -> tuple[float, float]:
        """``(reply_timeout, gap_after_reply)``: these options, else the scenario's."""
        reply_timeout = scenario.reply_timeout if self.reply_timeout is None else self.reply_timeout
        gap = scenario.gap_after_reply if self.gap_after_reply is None else self.gap_after_reply
        return reply_timeout, gap

    def validate(self) -> None:
        if self.turns < 1:
            raise ValueError("turns must be >= 1")
        if self.sessions < 1:
            raise ValueError("sessions must be >= 1")
        if self.warmup_turns < 0:
            raise ValueError("warmup_turns must be >= 0")
        if self.dead_air_threshold <= 0:
            raise ValueError("dead_air_threshold must be > 0")


class LatencyItem(BaseModel):
    """One user turn of one session (a line of ``items.jsonl``). Times in the recording
    clock are seconds; durations are milliseconds."""

    model_config = ConfigDict(extra="forbid")

    session: int
    turn: int
    stimulus: str
    text: str | None = None
    warmup: bool = False
    expect_reply: bool = True
    user_speech_start_s: float
    user_speech_end_s: float
    """``t_uoff`` on the recording clock."""
    agent_onset_s: float | None = None
    """``t_aon`` on the recording clock."""
    v2v_ms: float | None = None
    session_v2v_ms: float | None = None
    residual_ms: float | None = None
    eou_delay_ms: float | None = None
    response_ttfb_ms: float | None = None
    stt_latency_ms: float | None = None
    llm_ttft_ms: float | None = None
    tts_ttfb_ms: float | None = None
    engine_ttfb_ms: float | None = None
    agent_speech_ms: float | None = None
    """Agent speech (reference VAD) between this turn and the next."""
    agent_audio: bool = False
    """Any agent audio was played after the user started this turn (transport playout log),
    whether or not the reference VAD found speech in it."""
    missed: bool = False
    premature: bool = False
    dead_air: bool = False
    interrupted: bool = False
    user_transcript: str | None = None
    agent_transcript: str | None = None
    errors: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------- session probe


class _SessionProbe:
    """Records session events with their ``now()`` arrival time."""

    def __init__(self, session: AgentSession) -> None:
        self.session = session
        self.turn_metrics: list[tuple[float, TurnMetrics]] = []
        self.components: list[tuple[float, Metrics]] = []
        self.user_finals: list[tuple[float, str]] = []
        self.agent_text: list[tuple[float, str]] = []
        self.interruptions: list[float] = []
        self.errors: list[tuple[float, str]] = []
        self.timeline: list[tuple[float, str, dict[str, Any]]] = []
        self._handlers: dict[str, Callable[[Any], None]] = {
            "metrics": self._on_metrics,
            "user_transcript": self._on_user_transcript,
            "agent_transcript": self._on_agent_transcript,
            "agent_state_changed": self._on_state("agent_state"),
            "user_state_changed": self._on_state("user_state"),
            "interrupted": self._on_interrupted,
            "error": self._on_error,
        }
        for name, handler in self._handlers.items():
            session.on(name, handler)

    def detach(self) -> None:
        for name, handler in self._handlers.items():
            self.session.off(name, handler)

    def _on_metrics(self, m: Metrics) -> None:
        t = now()
        if isinstance(m, TurnMetrics):
            self.turn_metrics.append((t, m))
        else:
            self.components.append((t, m))
        data = {k: v for k, v in asdict(m).items() if k != "timestamp"}
        self.timeline.append((t, f"metrics.{m.type}", data))

    def _on_user_transcript(self, ev: Any) -> None:
        if ev.is_final:
            t = now()
            self.user_finals.append((t, ev.text))
            self.timeline.append((t, "user_transcript", {"text": ev.text}))

    def _on_agent_transcript(self, ev: Any) -> None:
        t = now()
        self.agent_text.append((t, ev.delta))
        self.timeline.append((t, "agent_transcript", {"delta": ev.delta}))

    def _on_state(self, name: str) -> Callable[[Any], None]:
        def handler(ev: Any) -> None:
            self.timeline.append(
                (now(), name, {"old": str(ev.old_state), "new": str(ev.new_state)})
            )

        return handler

    def _on_interrupted(self, ev: Any) -> None:
        t = now()
        self.interruptions.append(t)
        self.timeline.append((t, "interrupted", {"played_s": ev.played}))

    def _on_error(self, ev: Any) -> None:
        t = now()
        self.errors.append((t, repr(ev.error)))
        self.timeline.append((t, "error", {"error": repr(ev.error), "recoverable": ev.recoverable}))


@dataclass
class _SessionRun:
    index: int
    origin: float
    ready: float
    call: CallResult
    probe: _SessionProbe
    transport: LoopbackTransport | None = None
    """The session's transport (playout log, playback clears), for other tracks."""


@dataclass
class _SessionAnalysis:
    items: list[LatencyItem]
    labels: list[Label]
    info: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------------ running


def _between(entries: Iterable[tuple[float, T]], start: float, end: float) -> list[T]:
    return [value for t, value in entries if start <= t < end]


def _first(values: Iterable[float | None]) -> float | None:
    return next((v for v in values if v is not None), None)


def _ms(seconds: float | None) -> float | None:
    return None if seconds is None else round(seconds * 1000.0, 3)


async def _run_session(
    index: int,
    engine: Any,
    system: BenchSystem,
    stimuli: Sequence[Stimulus],
    scenario: Scenario,
    options: LatencyOptions,
    on_turn: Callable[[TurnTiming], None] | None,
    *,
    on_start: Callable[[AgentSession, LoopbackTransport], None] | None = None,
    sleep_until: Callable[[float], Awaitable[None]] | None = None,
) -> _SessionRun:
    """One simulated call (also used by the overhead track: ``on_start`` sees the started
    session before the caller speaks; ``sleep_until`` paces the caller)."""
    session = AgentSession(engine, options=system.session_options())
    probe = _SessionProbe(session)
    transport = LoopbackTransport(
        input_format=AudioFormat(scenario.sample_rate, 1),
        output_format=_AGENT_FORMAT,
        realtime_playout=True,
    )
    origin = now()
    try:
        await session.start(system.build_agent(), transport)
    except BaseException:
        probe.detach()
        with contextlib.suppress(Exception):
            await session.aclose()
        raise
    ready = now()
    caller = CallerEmulator(
        transport,
        chunk=scenario.chunk,
        origin=origin,
        agent_idle=lambda: session.agent_state in (AgentState.LISTENING, AgentState.CLOSED),
        should_stop=lambda: session.closed,
        on_turn=on_turn,
        sleep_until=sleep_until,
    )
    reply_timeout, gap = options.timing(scenario)
    try:
        if on_start is not None:
            on_start(session, transport)
        call = await caller.run(
            stimuli,
            lead_in=scenario.lead_in,
            reply_timeout=reply_timeout,
            gap_after_reply=gap,
            max_reply=scenario.max_reply,
        )
    finally:
        await session.aclose()
        probe.detach()
    return _SessionRun(index, origin, ready, call, probe, transport)


def _analyze_session(
    run: _SessionRun, detector: OnsetDetector, options: LatencyOptions
) -> _SessionAnalysis:
    rec = run.call.recording
    agent = rec.agent_audio()
    mask = detector.speech_mask(agent)
    onsets = detector.onsets(agent, mask)
    segments = detector.segments(agent, mask)
    probe = run.probe
    turns = run.call.turns
    ready_s = rec.to_offset(run.ready)
    labels = [Label(ready_s, ready_s, "session ready")]
    labels += [Label(s, e, "agent speech") for s, e in segments]
    items: list[LatencyItem] = []
    for k, turn in enumerate(turns):
        stim = turn.stimulus
        w0 = turn.speech_start
        w1 = turns[k + 1].speech_start if k + 1 < len(turns) else math.inf
        uon, uoff = rec.to_offset(w0), rec.to_offset(turn.speech_end)
        window_end = None if math.isinf(w1) else rec.to_offset(w1)
        aon = first_onset_between(onsets, uon, window_end)
        v2v = None if aon is None else (aon - uoff) * 1000.0

        tms = _between(probe.turn_metrics, w0, w1)
        tm = next((m for m in tms if m.voice_to_voice is not None), tms[0] if tms else None)
        comps = _between(probe.components, w0, w1)
        session_v2v = _ms(tm.voice_to_voice) if tm is not None else None
        finals = _between(probe.user_finals, w0, w1)
        agent_text = "".join(_between(probe.agent_text, w0, w1)).strip()
        errors = _between(probe.errors, w0, w1)
        speech_end = rec.duration if window_end is None else window_end
        agent_speech = sum(min(e, speech_end) - s for s, e in segments if uon <= s < speech_end)
        missed = stim.expect_reply and aon is None
        items.append(
            LatencyItem(
                session=run.index,
                turn=k,
                stimulus=stim.id,
                text=stim.text,
                warmup=k < options.warmup_turns,
                expect_reply=stim.expect_reply,
                user_speech_start_s=round(uon, 6),
                user_speech_end_s=round(uoff, 6),
                agent_onset_s=None if aon is None else round(aon, 6),
                v2v_ms=None if v2v is None else round(v2v, 3),
                session_v2v_ms=session_v2v,
                residual_ms=(
                    None if v2v is None or session_v2v is None else round(v2v - session_v2v, 3)
                ),
                eou_delay_ms=_ms(tm.end_of_turn_delay) if tm is not None else None,
                response_ttfb_ms=_ms(tm.response_ttfb) if tm is not None else None,
                stt_latency_ms=_ms(_first(m.latency for m in comps if isinstance(m, STTMetrics))),
                llm_ttft_ms=_ms(_first(m.ttft for m in comps if isinstance(m, LLMMetrics))),
                tts_ttfb_ms=_ms(_first(m.ttfb for m in comps if isinstance(m, TTSMetrics))),
                engine_ttfb_ms=_ms(_first(m.ttfb for m in comps if isinstance(m, EngineMetrics))),
                agent_speech_ms=_ms(agent_speech),
                agent_audio=turn.reply_start is not None,
                missed=missed,
                premature=v2v is not None and v2v < 0,
                dead_air=stim.expect_reply
                and (missed or (v2v is not None and v2v > options.dead_air_threshold * 1000)),
                interrupted=bool(tm is not None and tm.interrupted)
                or bool(_between(((t, t) for t in probe.interruptions), w0, w1)),
                user_transcript=finals[-1] if finals else None,
                agent_transcript=agent_text or None,
                errors=errors,
            )
        )
        text = f"user {stim.id}: {stim.text}" if stim.text else f"user {stim.id}"
        labels.append(Label(uon, uoff, text))
        if aon is not None and v2v is not None:
            labels.append(Label(aon, aon, f"agent onset {stim.id}: v2v {v2v:.0f} ms"))
        elif missed:
            labels.append(Label(uoff, uoff, f"missed reply {stim.id}"))

    first_uon = rec.to_offset(turns[0].speech_start) if turns else None
    greeting = first_onset_between(onsets, 0.0, first_uon)
    lags = run.call.push_lag
    info = {
        "session": run.index,
        "session_ready_ms": _ms(run.ready - run.origin),
        "greeting_ms": None if greeting is None else round((greeting - ready_s) * 1000, 3),
        "turns": len(turns),
        "aborted": run.call.aborted,
        "duration_s": round(rec.duration, 3),
        "agent_frames": run.call.agent_frames,
        "playback_clears": len(run.call.clear_times),
        "push_lag_p99_ms": _ms(percentile(lags, 99)) if lags else None,
        "push_lag_max_ms": _ms(run.call.max_push_lag),
        "errors": [e for _, e in probe.errors],
    }
    return _SessionAnalysis(items, labels, info)


# ----------------------------------------------------------------------- summary


def summarize_latency(
    items: Sequence[LatencyItem],
    sessions: Sequence[dict[str, Any]] = (),
    *,
    dead_air_threshold: float = 2.0,
    seed: int = 0,
    n_resamples: int = 2000,
) -> tuple[dict[str, Distribution], dict[str, float | None], dict[str, int], dict[str, Any]]:
    """Distributions, rates, counts and extras of a latency run."""

    def dist(values: Iterable[float | None]) -> Distribution:
        return Distribution.of(values, seed=seed, n_resamples=n_resamples)

    main = [it for it in items if it.expect_reply and not it.warmup]
    first = [it for it in items if it.expect_reply and it.turn == 0]
    metrics: dict[str, Distribution] = {
        "v2v_ms": dist(it.v2v_ms for it in main),
        "first_turn_v2v_ms": dist(it.v2v_ms for it in first),
    }
    for key in (
        "session_v2v_ms", "residual_ms", "eou_delay_ms", "response_ttfb_ms",
        "stt_latency_ms", "llm_ttft_ms", "tts_ttfb_ms", "engine_ttfb_ms", "agent_speech_ms",
    ):  # fmt: skip
        d = dist(getattr(it, key) for it in main)
        if d.n:
            metrics[key] = d
    for key in ("session_ready_ms", "greeting_ms"):
        d = dist(s.get(key) for s in sessions)
        if d.n:
            metrics[key] = d

    def rate(flag: str) -> float | None:
        return round(sum(bool(getattr(it, flag)) for it in main) / len(main), 6) if main else None

    rates = {
        "dead_air_rate": rate("dead_air"),
        "missed_rate": rate("missed"),
        "premature_rate": rate("premature"),
        "interrupted_rate": rate("interrupted"),
    }
    counts = {
        "sessions": len(sessions) or len({it.session for it in items}),
        "turns": len(items),
        "turns_measured": len(main),
        "warmup_turns": sum(it.warmup for it in items),
        "replies": sum(it.v2v_ms is not None for it in main),
        "missed": sum(it.missed for it in main),
        "missed_with_audio": sum(it.missed and it.agent_audio for it in main),
        "dead_air": sum(it.dead_air for it in main),
        "premature": sum(it.premature for it in main),
        "errors": sum(len(it.errors) for it in items),
    }
    spans = {
        key.removesuffix("_ms"): metrics[key].p50
        for key in ("eou_delay_ms", "response_ttfb_ms", "stt_latency_ms", "llm_ttft_ms",
                    "tts_ttfb_ms", "engine_ttfb_ms", "residual_ms")
        if key in metrics
    }  # fmt: skip
    extra: dict[str, Any] = {
        "dead_air_threshold_ms": dead_air_threshold * 1000.0,
        "spans_p50_ms": spans,
        "sessions": list(sessions),
    }
    return metrics, rates, counts, extra


# ------------------------------------------------------------------------ report

_METRIC_LABELS = {
    "v2v_ms": "**voice-to-voice** `v2v_ms` (recording)",
    "first_turn_v2v_ms": "first turn / cold start",
    "session_v2v_ms": "session `TurnMetrics.voice_to_voice`",
    "residual_ms": "residual (recording − session)",
    "eou_delay_ms": "end-of-turn delay",
    "response_ttfb_ms": "response TTFB (commit → audio)",
    "stt_latency_ms": "STT final-transcript latency",
    "llm_ttft_ms": "LLM time to first token",
    "tts_ttfb_ms": "TTS time to first byte",
    "engine_ttfb_ms": "engine TTFB",
    "agent_speech_ms": "agent speech per turn",
    "session_ready_ms": "session ready (connect)",
    "greeting_ms": "greeting onset after ready",
}

_ITEM_COLUMNS = (
    ("session", "session"),
    ("turn", "turn"),
    ("stimulus", "stimulus"),
    ("v2v_ms", "v2v ms"),
    ("session_v2v_ms", "session ms"),
    ("residual_ms", "residual ms"),
    ("eou_delay_ms", "EOU ms"),
    ("response_ttfb_ms", "TTFB ms"),
    ("agent_speech_ms", "reply ms"),
    ("missed", "missed"),
    ("warmup", "warm-up"),
)

_METHOD = """\
* `v2v_ms` = `t_aon − t_uoff` per user turn that expects a reply, measured on the
  stereo recording (user left, agent right, one clock).
* `t_uoff`: annotated end of user speech of the pre-rendered stimulus (annotated once on
  the clean audio). The caller streams stimuli in real time in {chunk_ms:g} ms chunks,
  each delivered when its interval has elapsed, like a capture device.
* `t_aon`: first {frame_ms:g} ms frame that begins at least {min_speech_ms:g} ms of speech
  on the agent channel according to the reference VAD ({vad}); clicks and comfort noise
  do not count. {refine}
* `session_v2v_ms`: the session's own metric (engine end-of-speech → first agent audio
  handed to the transport); `residual_ms` = `v2v_ms − session_v2v_ms`.
* Headline distributions exclude the first {warmup} turn(s) of every session (reported as
  cold start). Dead air = no reply, or `v2v_ms` above {dead_air_ms:g} ms.
* Percentiles are linear-interpolated; `[..]` is the 95% percentile-bootstrap CI of the
  median ({resamples} resamples, seed {seed}).
* Transport: in-process loopback with real-time playout — no network, codec or device
  latency is included. Smoke-tier numbers are regression canaries, not capability scores.
"""


def latency_report_spec(results: RunResults) -> ReportSpec:
    opts = results.manifest.options
    onset = opts.get("onset", {})
    vad = onset.get("reference_vad", {})
    vad_desc = ", ".join(f"{k}={v}" for k, v in vad.items()) or "default"
    refine = (
        "The onset is then refined to the first sample of that frame reaching "
        f"{onset['refine_threshold_db']:g} dBFS (inside a 1 ms block at that level)."
        if onset.get("refine") and onset.get("refine_threshold_db") is not None
        else ""
    )
    method = _METHOD.format(
        refine=refine,
        chunk_ms=opts.get("chunk_ms", 20),
        frame_ms=onset.get("frame_ms", 10),
        min_speech_ms=onset.get("min_speech_ms", 100),
        vad=vad_desc,
        warmup=opts.get("warmup_turns", 1),
        dead_air_ms=results.summary.extra.get("dead_air_threshold_ms", 2000.0),
        resamples=opts.get("bootstrap_resamples", 2000),
        seed=opts.get("seed", 0),
    )
    threshold = results.summary.extra.get("dead_air_threshold_ms", 2000.0)
    return ReportSpec(
        title=f"Latency (T1) · {results.summary.system}",
        metric_labels=_METRIC_LABELS,
        rate_labels={
            "dead_air_rate": f"dead air (> {threshold:g} ms or no reply)",
            "missed_rate": "missed (no reply)",
            "premature_rate": "premature (agent started before the user finished)",
            "interrupted_rate": "interrupted replies",
        },
        item_columns=_ITEM_COLUMNS,
        sections=[("Method", method)],
    )


def render_latency_report(results: RunResults) -> str:
    return render_report(results, latency_report_spec(results))


# ------------------------------------------------------------------------- entry


def _dataset_id(scenario: Scenario, stimuli: Sequence[Stimulus]) -> str:
    payload = json.dumps(
        {"definition": scenario.definition_sha256(), "stimuli": [s.sha256 for s in stimuli]},
        sort_keys=True,
    )
    return f"{scenario.name}@sha256:{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def _write_artifacts(
    directory: Path, runs: Sequence[_SessionRun], analyses: Sequence[_SessionAnalysis]
) -> None:
    for run, analysis in zip(runs, analyses, strict=True):
        folder = directory / ARTIFACTS_DIR / f"session-{run.index:03d}"
        run.call.recording.write_wav(folder / "stereo.wav")
        write_labels(folder / "labels.txt", analysis.labels)
        with (folder / "timeline.jsonl").open("w", encoding="utf-8") as f:
            for t, kind, data in run.probe.timeline:
                row = {"t": round(t - run.origin, 6), "event": kind, **data}
                f.write(json.dumps(row, default=str) + "\n")


def _reserve_directory(parent: Path, run_id: str, *, unique: bool) -> Path:
    """Create the run directory; generated ids get a ``-2``, ``-3``... suffix on collision."""
    directory = parent / run_id
    suffix = 2
    while unique:
        try:
            directory.mkdir(parents=True)
            return directory
        except FileExistsError:
            directory = parent / f"{run_id}-{suffix}"
            suffix += 1
    directory.mkdir(parents=True, exist_ok=True)
    return directory


async def run_latency_benchmark(
    system: BenchSystem,
    scenario: Scenario,
    options: LatencyOptions | None = None,
    *,
    out_dir: str | os.PathLike[str] | None = None,
    run_id: str | None = None,
    detector: OnsetDetector | None = None,
    on_turn: Callable[[int, TurnTiming], None] | None = None,
) -> RunResults:
    """Run the T1 latency track and (if ``out_dir``) write ``<out_dir>/<run_id>/``.

    Args:
        system: engine or cascade under test.
        scenario: scripted user turns (see :mod:`voice_agent_next.bench.stimuli`).
        options: turns, sessions, warm-up, dead-air threshold...
        out_dir: parent directory of the run directory (``None``: nothing is written).
        run_id: defaults to ``<UTC time>-latency-<system label>``.
        detector: agent onset detector (default: RMS reference VAD, 10 ms / 100 ms).
        on_turn: progress callback ``(session index, turn timing)``.
    """
    options = options or LatencyOptions()
    options.validate()
    final_id = run_id or new_run_id(TRACK, system.label)
    directory: Path | None = None
    if out_dir is not None:
        directory = _reserve_directory(Path(out_dir), final_id, unique=run_id is None)
        final_id = directory.name
    try:
        return await _run_latency(
            system, scenario, options, directory=directory, run_id=final_id,
            detector=detector or OnsetDetector(), on_turn=on_turn,
        )  # fmt: skip
    except BaseException:
        if directory is not None and not any(directory.iterdir()):
            directory.rmdir()  # do not leave empty run directories behind
        raise


async def _run_latency(
    system: BenchSystem,
    scenario: Scenario,
    options: LatencyOptions,
    *,
    directory: Path | None,
    run_id: str,
    detector: OnsetDetector,
    on_turn: Callable[[int, TurnTiming], None] | None,
) -> RunResults:
    created = utc_timestamp()
    t_start = now()
    stimuli = await render_stimuli(scenario, turns=options.turns)

    t0 = now()
    engine = system.build_engine()
    engine_init_ms = (now() - t0) * 1000.0
    engine_warmup_ms: float | None = None
    runs: list[_SessionRun] = []
    try:
        if options.warmup_engine:
            t0 = now()
            await engine.warmup()
            engine_warmup_ms = (now() - t0) * 1000.0
        for index in range(options.sessions):

            def turn_done(turn: TurnTiming, index: int = index) -> None:
                if on_turn is not None:
                    on_turn(index, turn)

            runs.append(
                await _run_session(index, engine, system, stimuli, scenario, options, turn_done)
            )
    finally:
        await engine.aclose()

    analyses = [await asyncio.to_thread(_analyze_session, r, detector, options) for r in runs]
    items = [it for a in analyses for it in a.items]
    sessions = [a.info for a in analyses]
    metrics, rates, counts, extra = summarize_latency(
        items,
        sessions,
        dead_air_threshold=options.dead_air_threshold,
        seed=options.seed,
        n_resamples=options.bootstrap_resamples,
    )
    extra["engine_init_ms"] = round(engine_init_ms, 3)
    extra["engine_warmup_ms"] = None if engine_warmup_ms is None else round(engine_warmup_ms, 3)

    notes: list[str] = []
    worst_lag = max((s.get("push_lag_max_ms") or 0.0 for s in sessions), default=0.0)
    if worst_lag > 50.0:
        notes.append(
            f"The caller delivered audio up to {worst_lag:.0f} ms late (event-loop stalls): "
            "latencies may be inflated."
        )
    if any(s.get("aborted") for s in sessions):
        notes.append("At least one session closed early (see errors in summary.json).")
    if counts["missed_with_audio"]:
        notes.append(
            f"{counts['missed_with_audio']} missed turn(s) had agent audio in which the reference "
            "VAD found no speech onset: check --reference-vad (speech models such as Silero do "
            "not treat the mock engine's synthetic tone as speech)."
        )
    if counts["turns_measured"] == 0:
        notes.append("No turns in the headline population: increase --turns or lower warm-up.")

    unique: dict[str, Stimulus] = {}
    for stim in stimuli:
        unique.setdefault(stim.id, stim)
    reply_timeout, gap = options.timing(scenario)
    manifest = RunManifest(
        run_id=run_id,
        track=TRACK,
        created=created,
        system=system.describe(engine),
        scenario={
            "name": scenario.name,
            "version": scenario.version,
            "sha256": scenario.definition_sha256(),
            "definition": scenario.model_dump(mode="json"),
            "stimuli": [s.describe() for s in unique.values()],
            "sequence": [s.id for s in stimuli],
        },
        transport={
            "type": "loopback",
            "realtime_playout": True,
            "input_format": str(AudioFormat(scenario.sample_rate, 1)),
            "output_format": str(_AGENT_FORMAT),
            "chunk_ms": round(scenario.chunk * 1000, 3),
            "delivery": "each chunk when its interval has elapsed (capture-device model)",
        },
        options={
            **asdict(options),
            "chunk_ms": round(scenario.chunk * 1000, 3),
            "reply_timeout_s": reply_timeout,
            "gap_after_reply_s": gap,
            "lead_in_s": scenario.lead_in,
            "onset": detector.describe(),
        },
        environment=await asyncio.to_thread(collect_environment),
        notes=notes,
    )
    summary = RunSummary(
        run_id=run_id,
        track=TRACK,
        system=system.label,
        transport="loopback",
        dataset=_dataset_id(scenario, stimuli),
        n=metrics["v2v_ms"].n,
        metrics=metrics,
        rates=rates,
        counts=counts,
        extra=extra,
        duration_s=round(now() - t_start, 3),
    )
    results = RunResults(manifest, [it.model_dump(mode="json") for it in items], summary)
    results.report = render_latency_report(results)
    if directory is not None:
        write_run(directory, results)
        if options.save_audio:
            await asyncio.to_thread(_write_artifacts, directory, runs, analyses)
    return results
