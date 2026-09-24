"""T4 turn-taking battery: does the whole system talk and stop at the right time?

``van bench turn-taking`` runs a scripted conversation over the T1 harness (the real-time
:class:`~voice_agent_next.bench.caller.CallerEmulator`, the loopback transport and the
stereo recording) against **any** system — cascade, native speech-to-speech or
full-duplex — and measures turn-taking on the recording (research note 06, §8.3 T4):

**Mid-turn pauses** (turns with ``parts``): the user pauses 0.4–1.0 s inside one turn
("Where is my order? · I placed it last week.").

* ``premature_rate`` — share of those turns in which the agent started speaking before
  the user finished (the agent onset, reference VAD on the agent channel, lies before the
  end of the user's last part). The same rule on plain questions is
  ``premature_rate_questions``.

**Barge-in** (turns with ``barge_in``: spoken that many seconds after the agent's reply
started, over it). ``overlapped`` turns are those where the agent was still speaking when
the user started; the others are reported but not scored.

* ``interruption`` turns (a real request): ``barge_in_stop_ms`` = agent audio stops (start
  of the first ≥ 300 ms silence on the agent channel) − user speech onset, and
  ``stop_within_500ms_rate``; ``interrupted_rate`` (the session cut the reply for good);
  ``post_interrupt_response_ms`` = next agent onset − end of the interruption;
  ``talk_over_ms`` = user and agent speaking at the same time.
* ``backchannel`` ("uh-huh") and ``noise`` (a cough) turns, which must **not** stop the
  agent: ``backchannel_yield_rate`` / ``noise_yield_rate`` — the agent went silent for
  ≥ 300 ms right after (a pause-and-resume policy yields briefly by design);
  ``false_barge_in_rate`` — the agent's reply was abandoned (the session interrupted it,
  or it stopped and never resumed before the next user turn); ``resume_rate`` — share of
  the yields after which the same reply continued; ``yield_ms`` and ``resume_gap_ms``.

**Missed turns and dead air** — as in T1, over every turn that expects a reply:
``missed_rate`` (no agent speech within ``reply_timeout``) and ``dead_air_rate`` (no reply,
or ``v2v_ms`` above 2 s), plus ``v2v_ms`` of plain questions and pause turns.
"""

from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...audio.frame import AudioFormat
from ...session import AgentSession
from ...transports.loopback import LoopbackTransport
from ...utils.clock import now
from ..caller import TurnTiming
from ..environment import collect_environment
from ..onset import OnsetDetector
from ..report import ReportSpec, fmt, markdown_table, render_report
from ..results import (
    Distribution,
    RunManifest,
    RunResults,
    RunSummary,
    new_run_id,
    utc_timestamp,
    write_run,
)
from ..stimuli import Scenario, Stimulus, render_stimuli
from ..system import BenchSystem
from .latency import (
    _AGENT_FORMAT,
    LatencyItem,
    LatencyOptions,
    _analyze_session,
    _dataset_id,
    _reserve_directory,
    _run_session,
    _SessionRun,
    _write_artifacts,
)

__all__ = [
    "DEFAULT_MOCK_ENGINE",
    "TRACK",
    "TurnTakingItem",
    "TurnTakingOptions",
    "category_of",
    "render_turn_taking_report",
    "run_turn_taking_benchmark",
    "summarize_turn_taking",
]

TRACK = "turn-taking"
DEFAULT_MOCK_ENGINE: dict[str, Any] = {
    "provider": "mock",
    "default_response": "Sure. Our store is open every day from nine to six, weekends too.",
}
"""The battery's default system: the mock engine with ~4 s replies (room for barge-ins)."""
BARGE_IN_CATEGORIES = ("interruption", "backchannel", "noise")


@dataclass
class TurnTakingOptions:
    turns: int | None = None
    """Turns per session (default: one pass over the scenario)."""
    sessions: int = 1
    warmup_turns: int = 1
    dead_air_threshold: float = 2.0
    reply_timeout: float | None = None
    gap_after_reply: float | None = None
    yield_gap: float = 0.3
    """Agent silence that counts as stopping / yielding (s)."""
    yield_window: float = 1.0
    """A yield must start before the end of the user's overlap + this (s)."""
    save_audio: bool = True
    warmup_engine: bool = True
    seed: int = 0
    bootstrap_resamples: int = 2000

    def latency_options(self, scenario: Scenario) -> LatencyOptions:
        return LatencyOptions(
            turns=self.turns or len(scenario.turns),
            sessions=self.sessions,
            warmup_turns=self.warmup_turns,
            dead_air_threshold=self.dead_air_threshold,
            reply_timeout=self.reply_timeout,
            gap_after_reply=self.gap_after_reply,
            save_audio=self.save_audio,
            warmup_engine=self.warmup_engine,
            seed=self.seed,
            bootstrap_resamples=self.bootstrap_resamples,
        )

    def validate(self) -> None:
        if self.turns is not None and self.turns < 1:
            raise ValueError("turns must be >= 1")
        if self.sessions < 1:
            raise ValueError("sessions must be >= 1")
        if self.yield_gap <= 0:
            raise ValueError("yield_gap must be > 0")


class TurnTakingItem(LatencyItem):
    """One user turn with the battery's measurements (a line of ``items.jsonl``)."""

    category: str = "question"
    pauses_s: list[list[float]] = []
    """Mid-turn pauses on the recording clock."""
    barge_in_s: float | None = None
    overlapped: bool | None = None
    """Barge-in turns: the agent was speaking when the user started."""
    agent_stop_s: float | None = None
    stop_ms: float | None = None
    """Agent stopped (≥ ``yield_gap`` of silence) − user onset."""
    yielded: bool | None = None
    session_interrupted: bool | None = None
    resumed: bool | None = None
    abandoned: bool | None = None
    resume_gap_ms: float | None = None
    post_interrupt_response_ms: float | None = None
    talk_over_ms: float | None = None
    false_interruption_events: int = 0


def category_of(stim: Stimulus) -> str:
    """The stimulus' ``category``, else inferred: pause / interruption / backchannel /
    question."""
    if stim.category:
        return stim.category
    if stim.pauses:
        return "pause"
    if stim.barge_in is not None:
        if stim.source == "noise":
            return "noise"
        return "interruption" if stim.expect_reply else "backchannel"
    return "question"


# ---------------------------------------------------------------------- analysis


def _first_silence(mask: np.ndarray, start: int, end: int, length: int) -> int | None:
    """First frame in ``[start, end)`` that begins ``length`` silent frames (the end of the
    recording counts as silence)."""
    speech = np.concatenate([mask.astype(np.int64), np.zeros(length, dtype=np.int64)])
    csum = np.concatenate([[0], np.cumsum(speech)])
    for f in range(max(0, start), max(0, end)):
        if csum[f + length] - csum[f] == 0:
            return f
    return None


def _analyze_battery(
    run: _SessionRun,
    items: Sequence[LatencyItem],
    detector: OnsetDetector,
    options: TurnTakingOptions,
    false_interruptions: Sequence[float],
) -> list[TurnTakingItem]:
    rec = run.call.recording
    agent = rec.agent_audio()
    mask = detector.speech_mask(agent)
    fd = detector.frame_duration
    onsets = detector.onsets(agent, mask)
    interruptions = [rec.to_offset(t) for t in run.probe.interruptions]
    false_ints = [rec.to_offset(t) for t in false_interruptions]
    gap_frames = max(1, round(options.yield_gap / fd))
    turns: list[TurnTiming] = run.call.turns
    out: list[TurnTakingItem] = []
    for k, (turn, item) in enumerate(zip(turns, items, strict=True)):
        stim = turn.stimulus
        cat = category_of(stim)
        start = rec.to_offset(turn.start)
        uon, uoff = item.user_speech_start_s, item.user_speech_end_s
        nxt = rec.to_offset(turns[k + 1].speech_start) if k + 1 < len(turns) else rec.duration
        data = item.model_dump()
        data.update(
            category=cat,
            pauses_s=[[round(start + a, 6), round(start + b, 6)] for a, b in stim.pauses],
            barge_in_s=stim.barge_in,
        )
        if stim.barge_in is not None:
            f_on = min(len(mask) - 1, max(0, int(uon / fd)))
            lo, hi = max(0, f_on - 5), min(len(mask), f_on + 6)
            overlapped = bool(mask[lo:hi].any()) if len(mask) else False
            stop = _first_silence(mask, f_on, min(len(mask), int(math.ceil(nxt / fd))), gap_frames)
            stop = None if stop is None else stop * fd
            if not overlapped:
                stop = None
            session_int = any(uon - 0.05 <= t < nxt for t in interruptions)
            after = [t for t in onsets if stop is not None and stop < t < nxt]
            yielded = stop is not None and stop <= uoff + options.yield_window
            resumed = bool(yielded and not session_int and after)
            talk = float(
                mask[int(uon / fd) : int(math.ceil(uoff / fd))].sum() * fd * 1000.0
            )
            data.update(
                overlapped=overlapped,
                agent_stop_s=None if stop is None else round(stop, 6),
                stop_ms=None if stop is None else round((stop - uon) * 1000.0, 3),
                yielded=yielded if overlapped else None,
                session_interrupted=session_int,
                resumed=resumed if overlapped else None,
                abandoned=(session_int or (yielded and not after)) if overlapped else None,
                resume_gap_ms=(
                    round((after[0] - stop) * 1000.0, 3) if resumed and stop is not None else None
                ),
                post_interrupt_response_ms=(
                    round((after[0] - uoff) * 1000.0, 3)
                    if cat == "interruption" and after
                    else None
                ),
                talk_over_ms=round(talk, 3) if overlapped else None,
                false_interruption_events=sum(uon - 0.05 <= t < nxt for t in false_ints),
            )
        out.append(TurnTakingItem.model_validate(data))
    return out


# ----------------------------------------------------------------------- summary


def summarize_turn_taking(
    items: Sequence[TurnTakingItem],
    *,
    dead_air_threshold: float = 2.0,
    seed: int = 0,
    n_resamples: int = 2000,
) -> tuple[dict[str, Distribution], dict[str, float | None], dict[str, int], dict[str, Any]]:
    def dist(values: Any) -> Distribution:
        return Distribution.of(values, seed=seed, n_resamples=n_resamples)

    main = [it for it in items if not it.warmup]
    replies = [it for it in main if it.expect_reply and it.barge_in_s is None]
    pause = [it for it in replies if it.category == "pause"]
    questions = [it for it in replies if it.category != "pause"]
    over = [it for it in main if it.barge_in_s is not None and it.overlapped]
    inter = [it for it in over if it.category == "interruption"]
    bc = [it for it in over if it.category == "backchannel"]
    noise = [it for it in over if it.category == "noise"]
    false_pop = bc + noise
    yielded = [it for it in false_pop if it.yielded]

    def rate(pop: Sequence[TurnTakingItem], pred: Callable[[TurnTakingItem], bool]) -> float | None:
        return round(sum(bool(pred(it)) for it in pop) / len(pop), 6) if pop else None

    metrics = {
        "v2v_ms": dist(it.v2v_ms for it in replies if not it.premature),
        "first_turn_v2v_ms": dist(it.v2v_ms for it in items if it.turn == 0),
        "barge_in_stop_ms": dist(it.stop_ms for it in inter),
        "post_interrupt_response_ms": dist(it.post_interrupt_response_ms for it in inter),
        "talk_over_ms": dist(it.talk_over_ms for it in inter),
        "yield_ms": dist(it.stop_ms for it in yielded),
        "resume_gap_ms": dist(it.resume_gap_ms for it in yielded),
    }
    metrics = {k: v for k, v in metrics.items() if v.n or k == "v2v_ms"}
    expect = [it for it in main if it.expect_reply]
    rates = {
        "premature_rate": rate(pause, lambda it: it.premature),
        "premature_rate_questions": rate(questions, lambda it: it.premature),
        "missed_rate": rate(expect, lambda it: it.missed),
        "dead_air_rate": rate(replies, lambda it: it.dead_air),
        "stop_within_500ms_rate": rate(
            inter, lambda it: it.stop_ms is not None and it.stop_ms <= 500.0
        ),
        "interrupted_rate": rate(inter, lambda it: it.session_interrupted),
        "backchannel_yield_rate": rate(bc, lambda it: it.yielded),
        "noise_yield_rate": rate(noise, lambda it: it.yielded),
        "false_barge_in_rate": rate(false_pop, lambda it: it.abandoned),
        "false_barge_in_rate_backchannel": rate(bc, lambda it: it.abandoned),
        "false_barge_in_rate_noise": rate(noise, lambda it: it.abandoned),
        "resume_rate": rate(yielded, lambda it: it.resumed),
    }
    counts = {
        "turns": len(items),
        "turns_measured": len(main),
        "questions": len(questions),
        "pause_turns": len(pause),
        "premature": sum(it.premature for it in pause),
        "barge_in_turns": sum(it.barge_in_s is not None for it in main),
        "not_overlapped": sum(it.barge_in_s is not None and not it.overlapped for it in main),
        "interruptions": len(inter),
        "backchannels": len(bc),
        "noises": len(noise),
        "missed": sum(it.missed for it in expect),
        "errors": sum(len(it.errors) for it in items),
    }
    extra: dict[str, Any] = {"dead_air_threshold_ms": dead_air_threshold * 1000.0}
    return metrics, rates, counts, extra


# ------------------------------------------------------------------------ report

_RATE_LABELS = {
    "premature_rate": "**premature replies** in mid-turn pauses",
    "premature_rate_questions": "premature replies, plain questions",
    "missed_rate": "missed turns (no reply)",
    "dead_air_rate": "dead air (no reply or > 2 s)",
    "stop_within_500ms_rate": "interruptions: agent stopped within 500 ms",
    "interrupted_rate": "interruptions: reply cut by the session",
    "backchannel_yield_rate": "backchannels: agent yielded (≥ 300 ms silence)",
    "noise_yield_rate": "coughs / noise: agent yielded",
    "false_barge_in_rate": "**false barge-ins** (reply abandoned after a backchannel / noise)",
    "false_barge_in_rate_backchannel": "false barge-ins, backchannels",
    "false_barge_in_rate_noise": "false barge-ins, noise",
    "resume_rate": "resumed after a yield",
}
_METRIC_LABELS = {
    "v2v_ms": "voice-to-voice (questions + pause turns, not premature)",
    "first_turn_v2v_ms": "first turn / cold start",
    "barge_in_stop_ms": "**barge-in stop time** (interruptions)",
    "post_interrupt_response_ms": "answer after an interruption",
    "talk_over_ms": "talk-over during an interruption",
    "yield_ms": "yield time (backchannel / noise)",
    "resume_gap_ms": "silence before resuming",
}
_ITEM_COLUMNS = (
    ("session", "session"),
    ("turn", "turn"),
    ("stimulus", "stimulus"),
    ("category", "category"),
    ("v2v_ms", "v2v ms"),
    ("premature", "premature"),
    ("missed", "missed"),
    ("overlapped", "overlap"),
    ("stop_ms", "stop ms"),
    ("yielded", "yielded"),
    ("resumed", "resumed"),
    ("abandoned", "abandoned"),
)
_METHOD = """\
* The caller speaks the scenario in real time over the loopback transport; the stereo
  recording is analysed with the reference VAD ({vad}); agent onsets as in T1.
* Premature: the first agent onset after the user started lies before the end of the
  user's speech. For pause turns the end is the end of the last part.
* Barge-in turns start {barge:s} after the agent's reply started. Stop = start of the
  first ≥ {gap:g} ms silence on the agent channel after the user's onset; a yield must start
  within {window:g} s of the end of the user's overlap. Abandoned = the session interrupted
  the reply, or the agent stopped and did not speak again before the next user turn.
* Headline populations exclude the first {warmup} turn(s) of every session.
"""


def turn_taking_markdown_table(results: RunResults) -> str:
    s = results.summary
    rows = [[label.replace("**", ""), "–" if s.rates.get(k) is None else
             f"{100 * float(s.rates[k] or 0):.0f}%"]
            for k, label in _RATE_LABELS.items() if k in s.rates]  # fmt: skip
    for key in ("v2v_ms", "barge_in_stop_ms", "post_interrupt_response_ms", "yield_ms"):
        d = s.metrics.get(key)
        if d is not None and d.n:
            rows.append([_METRIC_LABELS[key].replace("**", "") + " p50",
                         f"{fmt(d.p50, 0)} ms (n={d.n})"])  # fmt: skip
    return markdown_table([s.system, "value"], rows, ["l", "r"])


def render_turn_taking_report(results: RunResults) -> str:
    opts = results.manifest.options
    vad = (opts.get("onset") or {}).get("reference_vad", {})
    method = _METHOD.format(
        vad=", ".join(f"{k}={v}" for k, v in vad.items()) or "rms",
        barge="the scenario's `barge_in` seconds",
        gap=1000 * opts.get("yield_gap", 0.3),
        window=opts.get("yield_window", 1.0),
        warmup=opts.get("warmup_turns", 1),
    )
    spec = ReportSpec(
        title=f"Turn-taking battery (T4) · {results.summary.system}",
        metric_labels=_METRIC_LABELS,
        rate_labels=_RATE_LABELS,
        item_columns=_ITEM_COLUMNS,
        sections=[("Method", method)],
    )
    return render_report(results, spec)


# ------------------------------------------------------------------------- entry


async def run_turn_taking_benchmark(
    system: BenchSystem,
    scenario: Scenario,
    options: TurnTakingOptions | None = None,
    *,
    out_dir: str | os.PathLike[str] | None = None,
    run_id: str | None = None,
    detector: OnsetDetector | None = None,
    on_turn: Callable[[int, TurnTiming], None] | None = None,
) -> RunResults:
    """Run the T4 turn-taking battery and (if ``out_dir``) write ``<out_dir>/<run_id>/``."""
    options = options or TurnTakingOptions()
    options.validate()
    detector = detector or OnsetDetector()
    lat = options.latency_options(scenario)
    final_id = run_id or new_run_id(TRACK, system.label)
    directory: Path | None = None
    if out_dir is not None:
        directory = _reserve_directory(Path(out_dir), final_id, unique=run_id is None)
        final_id = directory.name
    created = utc_timestamp()
    t_start = now()
    try:
        stimuli = await render_stimuli(scenario, turns=lat.turns)
        engine = system.build_engine()
        runs: list[_SessionRun] = []
        false_ints: list[list[float]] = []
        try:
            if options.warmup_engine:
                await engine.warmup()
            for index in range(options.sessions):
                events: list[float] = []
                false_ints.append(events)

                def hook(session: AgentSession, _t: LoopbackTransport, ev: list[float] = events) -> None:
                    session.on("agent_false_interruption", lambda _e: ev.append(now()))

                def turn_done(turn: TurnTiming, index: int = index) -> None:
                    if on_turn is not None:
                        on_turn(index, turn)

                runs.append(
                    await _run_session(
                        index, engine, system, stimuli, scenario, lat, turn_done, on_start=hook
                    )
                )
        finally:
            await engine.aclose()
    except BaseException:
        if directory is not None and not any(directory.iterdir()):
            directory.rmdir()
        raise

    analyses = [await asyncio.to_thread(_analyze_session, r, detector, lat) for r in runs]
    items: list[TurnTakingItem] = []
    for run, analysis, fi in zip(runs, analyses, false_ints, strict=True):
        items += await asyncio.to_thread(
            _analyze_battery, run, analysis.items, detector, options, fi
        )
    metrics, rates, counts, extra = summarize_turn_taking(
        items, dead_air_threshold=options.dead_air_threshold, seed=options.seed,
        n_resamples=options.bootstrap_resamples,
    )  # fmt: skip
    extra["sessions"] = [a.info for a in analyses]
    notes: list[str] = []
    if counts["not_overlapped"]:
        notes.append(
            f"{counts['not_overlapped']} barge-in turn(s) did not overlap agent speech (the reply "
            "had ended or never started): they are not scored. Use longer replies."
        )
    if any(a.info.get("aborted") for a in analyses):
        notes.append("At least one session closed early (see errors).")
    unique: dict[str, Stimulus] = {}
    for stim in stimuli:
        unique.setdefault(stim.id, stim)
    reply_timeout, gap = lat.timing(scenario)
    manifest = RunManifest(
        run_id=final_id,
        track=TRACK,
        created=created,
        system=system.describe(),
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
        },
        options={
            **asdict(options),
            "turns": lat.turns,
            "reply_timeout_s": reply_timeout,
            "gap_after_reply_s": gap,
            "onset": detector.describe(),
        },
        environment=await asyncio.to_thread(collect_environment),
        notes=notes,
    )
    summary = RunSummary(
        run_id=final_id,
        track=TRACK,
        system=system.label,
        transport="loopback",
        dataset=_dataset_id(scenario, stimuli),
        n=len([it for it in items if not it.warmup]),
        metrics=metrics,
        rates=rates,
        counts=counts,
        extra=extra,
        duration_s=round(now() - t_start, 3),
    )
    results = RunResults(manifest, [it.model_dump(mode="json") for it in items], summary)
    results.report = render_turn_taking_report(results)
    if directory is not None:
        write_run(directory, results)
        if options.save_audio:
            await asyncio.to_thread(_write_artifacts, directory, runs, analyses)
    return results
