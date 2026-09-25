"""T6 tool-use track: scripted spoken calls against deterministic mock tools.

``van bench tools`` plays every scenario of a pinned suite
(:mod:`voice_agent_next.bench.tool_env`, default ``smoke``: 11 calls, 3–4 turns each) as
a real-time spoken call over the T1 harness — the
:class:`~voice_agent_next.bench.caller.CallerEmulator`, the loopback transport and the
stereo recording — against **any** system that supports tools. The agent gets the
scenario's instructions and tools; the tools read and write a per-call mock database.
Scoring (:mod:`voice_agent_next.bench.tool_scoring`, research note 06 §8.3 T6):

* **Task success** — ``pass_at_1``: the final database equals the expected state and the
  agent said the expected facts; ``pass_hat_k``: every one of ``k`` trials passed
  (``--trials k``, τ-bench's pass^k).
* **Tool calls** — ``tool_precision`` / ``tool_recall`` / ``tool_f1``, ``arg_acc``,
  ``entity_capture_acc``, ``unnecessary_call_rate`` and unexpected writes.
* **Faithfulness** — ``say_do_violation_rate`` (claimed an action no tool performed) and
  ``hallucination_rate`` (agent turns stating numbers or order facts no tool returned).
* **Voice** — ``tool_round_latency_ms`` = end of the user's speech → first agent speech
  after the turn's last tool result, on the recording; split into ``pre_tool_ms`` (speech
  end → first call), ``tool_exec_ms`` (the mock's fixed delay) and ``post_tool_ms``
  (result → speech); ``first_response_ms`` (any agent speech, e.g. "let me check");
  ``filler_rate`` (the session's watchdog filler fired), ``spoke_before_result_rate``,
  ``tool_dead_air_rate``; ``turns_to_completion``.

The caller is scripted by default: it says the next line whatever the agent answered,
so the scripts give information in a natural order. ``caller="llm:<spec>"`` (``van bench
tools --caller llm:<spec>``) lets an LLM play the caller instead
(:class:`~voice_agent_next.bench.llm_caller.LLMCaller`, τ-bench style): persona and goal
from the scenario, the scripted lines as the details it knows, temperature 0 and a fixed
seed; it reacts to what the agent said and hangs up when done. The scoring is the same.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ...audio.frame import AudioFormat
from ...engine import S2SEngine
from ...llm import LLM
from ...registry import create
from ...session import Agent, AgentSession
from ...transports.loopback import LoopbackTransport
from ...tts import TTS
from ...utils.clock import now
from ..caller import NextStimulus, TurnTiming
from ..environment import collect_environment
from ..llm_caller import LLMCaller, LLMCallerOptions, describe_caller, parse_caller
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
from ..stimuli import Stimulus, render_stimuli
from ..system import BenchSystem, parse_component_spec
from ..tool_env import CallRecord, ToolScenario, ToolSuite, build_tools, state_hash
from ..tool_scoring import TurnObservation, score_scenario
from .latency import (
    _AGENT_FORMAT,
    LatencyItem,
    LatencyOptions,
    _analyze_session,
    _reserve_directory,
    _run_session,
    _SessionRun,
    _write_artifacts,
)

__all__ = [
    "TRACK",
    "EngineFactory",
    "ToolScenarioItem",
    "ToolsOptions",
    "render_tools_report",
    "run_tools_benchmark",
    "summarize_tools",
    "tools_markdown_table",
]

TRACK = "tools"
EngineFactory = Callable[[ToolSuite, ToolScenario], S2SEngine]
"""Builds a fresh engine for every call (e.g. the scripted reference engine)."""


@dataclass
class ToolsOptions:
    trials: int = 1
    """Calls per scenario (``k`` of pass^k)."""
    scenarios: tuple[str, ...] = ()
    """Run only these scenario ids (default: all)."""
    tool_delay_scale: float = 1.0
    """Multiplies every mock tool's delay (0 = instant tools)."""
    dead_air_threshold: float = 2.0
    reply_timeout: float | None = None
    gap_after_reply: float | None = None
    save_audio: bool = True
    warmup_engine: bool = True
    seed: int = 0
    bootstrap_resamples: int = 2000
    caller: str = "scripted"
    """``scripted`` (default) or ``llm:<llm spec>``: an LLM plays the caller."""
    caller_max_turns: int | None = None
    """LLM caller: most lines per call (default: the scripted turns + 3)."""
    caller_temperature: float = 0.0
    caller_seed: int | None = 0

    def validate(self) -> None:
        if self.trials < 1:
            raise ValueError("trials must be >= 1")
        parse_caller(self.caller)
        self.caller_options().validate()
        if self.tool_delay_scale < 0:
            raise ValueError("tool_delay_scale must be >= 0")
        if self.dead_air_threshold <= 0:
            raise ValueError("dead_air_threshold must be > 0")

    def caller_options(self) -> LLMCallerOptions:
        return LLMCallerOptions(
            max_turns=self.caller_max_turns,
            temperature=self.caller_temperature,
            seed=self.caller_seed,
        )


class ToolScenarioItem(BaseModel):
    """One call of one scenario (a line of ``items.jsonl``); ``turn_details`` and
    ``calls_detail`` hold the per-turn timing and every tool call."""

    model_config = ConfigDict(extra="forbid")

    scenario: str
    trial: int
    session: int
    tags: list[str] = Field(default_factory=list)
    passed: bool
    state_ok: bool
    outputs_ok: bool
    missing_outputs: list[str] = Field(default_factory=list)
    state_diff: list[str] = Field(default_factory=list)
    required_calls: int = 0
    matched_required: int = 0
    matched_calls: int = 0
    calls: int = 0
    tool_precision: float | None = None
    tool_recall: float | None = None
    args_correct: int = 0
    args_total: int = 0
    arg_acc: float | None = None
    entities_correct: int = 0
    entities_total: int = 0
    unnecessary_calls: int = 0
    unexpected_writes: int = 0
    tool_errors: int = 0
    wrong_args: list[str] = Field(default_factory=list)
    missing_calls: list[str] = Field(default_factory=list)
    say_do_violations: list[str] = Field(default_factory=list)
    say_do_count: int = 0
    hallucinations: list[str] = Field(default_factory=list)
    hallucinated_turns: int = 0
    agent_turns: int = 0
    """Turns in which the agent said anything."""
    turns_to_completion: int | None = None
    turns: int = 0
    missed_turns: int = 0
    tool_rounds: int = 0
    fillers: int = 0
    tool_round_ms: float | None = None
    """Mean ``tool_round_latency_ms`` of this call's tool rounds."""
    turn_details: list[dict[str, Any]] = Field(default_factory=list)
    calls_detail: list[dict[str, Any]] = Field(default_factory=list)
    caller_lines: list[dict[str, str | None]] = Field(default_factory=list)
    """LLM caller: every exchange (``agent`` said, ``caller`` answered; ``None`` = hung
    up). Empty for the scripted caller."""
    errors: list[str] = Field(default_factory=list)


@dataclass
class _ScenarioSystem(BenchSystem):
    """The system under test with the scenario's agent (instructions + mock tools)."""

    agent: Agent | None = None

    def build_agent(self) -> Agent:
        assert self.agent is not None
        return self.agent


@dataclass
class _Call:
    scenario: ToolScenario
    trial: int
    run: _SessionRun
    calls: list[CallRecord]
    fillers: list[float]
    snapshots: dict[int, str]
    final_db: dict[str, Any]
    caller: LLMCaller | None = None
    analysis: Any = None


# ---------------------------------------------------------------------- analysis


def _ms(seconds: float | None) -> float | None:
    return None if seconds is None else round(seconds * 1000.0, 3)


def _analyze_call(
    suite: ToolSuite,
    call: _Call,
    items: Sequence[LatencyItem],
    detector: OnsetDetector,
    options: ToolsOptions,
) -> ToolScenarioItem:
    rec = call.run.call.recording
    agent = rec.agent_audio()
    mask = detector.speech_mask(agent)
    onsets = detector.onsets(agent, mask)
    turns: list[TurnTiming] = call.run.call.turns
    origin = rec.origin
    observations: list[TurnObservation] = []
    details: list[dict[str, Any]] = []
    threshold_ms = options.dead_air_threshold * 1000.0
    for k, (turn, item) in enumerate(zip(turns, items, strict=True)):
        w0 = turn.speech_start
        w1 = turns[k + 1].speech_start if k + 1 < len(turns) else math.inf
        window_end = rec.duration if math.isinf(w1) else rec.to_offset(w1)
        uoff = item.user_speech_end_s
        in_turn = [c for c in call.calls if w0 <= c.started < w1]
        fillers = [t for t in call.fillers if w0 <= t < w1]
        agent_text = item.agent_transcript or ""
        user_text = turn.stimulus.text or ""  # the scripted line, or what the LLM caller said
        observations.append(TurnObservation(k, user_text, agent_text, w1, call.snapshots.get(k)))
        first = next((t for t in onsets if uoff <= t < window_end), None)
        detail: dict[str, Any] = {
            "turn": k,
            "stimulus": item.stimulus,
            "text": item.text,
            "user_transcript": item.user_transcript,
            "agent_transcript": item.agent_transcript,
            "calls": [c.name for c in in_turn],
            "missed": item.missed,
            "first_response_ms": None if first is None else round((first - uoff) * 1000.0, 3),
            "filler": bool(fillers),
        }
        if in_turn:
            t_call = rec.to_offset(min(c.started for c in in_turn))
            ends = [c.ended for c in in_turn if c.ended is not None]
            t_result = rec.to_offset(max(ends)) if ends else None
            after: float | None = None
            glued = False
            if t_result is not None:
                after = next((t for t in onsets if t_result <= t < window_end), None)
                f = int(t_result / detector.frame_duration)
                if after is None and 0 <= f < len(mask) and bool(mask[max(0, f - 2) : f + 1].any()):
                    # still talking (a preamble / filler) when the result came, straight into
                    # the answer: no silent wait; the result time is a lower bound
                    after, glued = t_result, True
            detail.update(
                tool_round=True,
                pre_tool_ms=round((t_call - uoff) * 1000.0, 3),
                tool_exec_ms=[_ms(c.ended - c.started) for c in in_turn if c.ended is not None],
                tool_round_latency_ms=None if after is None else round((after - uoff) * 1000, 3),
                post_tool_ms=(
                    None
                    if after is None or t_result is None
                    else round((after - t_result) * 1000.0, 3)
                ),
                answer_glued=glued,
                spoke_before_result=any(
                    uoff <= t < (t_result if t_result is not None else window_end) for t in onsets
                ),
                tool_dead_air=first is None or (first - uoff) * 1000.0 > threshold_ms,
            )
        else:
            detail.update(tool_round=False, v2v_ms=item.v2v_ms)
        details.append(detail)

    score = score_scenario(suite, call.scenario, call.calls, observations, call.final_db)
    for k, flagged in score.ungrounded.items():
        details[k]["hallucinations"] = flagged
    for k in score.say_do_turns:
        details[k]["say_do_violation"] = True
    rounds = [d for d in details if d.get("tool_round")]
    latencies = [
        d["tool_round_latency_ms"] for d in rounds if d["tool_round_latency_ms"] is not None
    ]
    data = score.summary()
    data.update(
        scenario=call.scenario.id,
        trial=call.trial,
        session=call.run.index,
        tags=list(call.scenario.tags),
        say_do_count=len(score.say_do),
        hallucinated_turns=len(score.ungrounded),
        agent_turns=sum(bool(o.agent_text.strip()) for o in observations),
        turns=len(turns),
        missed_turns=sum(it.missed for it in items),
        tool_rounds=len(rounds),
        fillers=sum(d["filler"] for d in details),
        tool_round_ms=round(sum(latencies) / len(latencies), 3) if latencies else None,
        turn_details=details,
        calls_detail=[c.describe(origin) for c in call.calls],
        caller_lines=call.caller.transcript() if call.caller is not None else [],
        errors=[e for it in items for e in it.errors]
        + (call.caller.errors if call.caller is not None else []),
    )
    return ToolScenarioItem.model_validate(data)


# ----------------------------------------------------------------------- summary


def summarize_tools(
    items: Sequence[ToolScenarioItem],
    *,
    trials: int = 1,
    seed: int = 0,
    n_resamples: int = 2000,
) -> tuple[dict[str, Distribution], dict[str, float | None], dict[str, int], dict[str, Any]]:
    """Distributions, rates, counts and extras of a T6 run."""

    def dist(values: Any) -> Distribution:
        return Distribution.of(values, seed=seed, n_resamples=n_resamples)

    def ratio(num: float, den: float) -> float | None:
        return round(num / den, 6) if den else None

    def share(pred: Callable[[ToolScenarioItem], bool]) -> float | None:
        return ratio(sum(bool(pred(it)) for it in items), len(items))

    turns = [d for it in items for d in it.turn_details]
    rounds = [d for d in turns if d.get("tool_round")]
    plain = [d for d in turns if not d.get("tool_round")]
    by_scenario: dict[str, list[bool]] = {}
    for it in items:
        by_scenario.setdefault(it.scenario, []).append(it.passed)
    calls = sum(it.calls for it in items)
    matched = sum(it.matched_calls for it in items)
    required = sum(it.required_calls for it in items)
    precision = ratio(matched, calls)
    recall = ratio(sum(it.matched_required for it in items), required)
    f1 = (
        round(2 * precision * recall / (precision + recall), 6)
        if precision is not None and recall is not None and precision + recall > 0
        else (0.0 if precision is not None and recall is not None else None)
    )
    rates: dict[str, float | None] = {
        "pass_at_1": share(lambda it: it.passed),
        "pass_hat_k": ratio(sum(all(v) for v in by_scenario.values()), len(by_scenario)),
        "state_ok_rate": share(lambda it: it.state_ok),
        "outputs_ok_rate": share(lambda it: it.outputs_ok),
        "tool_precision": precision,
        "tool_recall": recall,
        "tool_f1": f1,
        "arg_acc": ratio(sum(it.args_correct for it in items), sum(it.args_total for it in items)),
        "entity_capture_acc": ratio(
            sum(it.entities_correct for it in items), sum(it.entities_total for it in items)
        ),
        "unnecessary_call_rate": ratio(sum(it.unnecessary_calls for it in items), calls),
        "say_do_violation_rate": share(lambda it: it.say_do_count > 0),
        "hallucination_rate": ratio(
            sum(it.hallucinated_turns for it in items), sum(it.agent_turns for it in items)
        ),
        "filler_rate": ratio(sum(bool(d.get("filler")) for d in rounds), len(rounds)),
        "spoke_before_result_rate": ratio(
            sum(bool(d.get("spoke_before_result")) for d in rounds), len(rounds)
        ),
        "tool_dead_air_rate": ratio(sum(bool(d.get("tool_dead_air")) for d in rounds), len(rounds)),
        "missed_rate": ratio(sum(bool(d.get("missed")) for d in turns), len(turns)),
    }
    metrics = {
        "tool_round_latency_ms": dist(d.get("tool_round_latency_ms") for d in rounds),
        "first_response_ms": dist(d.get("first_response_ms") for d in rounds),
        "pre_tool_ms": dist(d.get("pre_tool_ms") for d in rounds),
        "tool_exec_ms": dist(v for d in rounds for v in d.get("tool_exec_ms") or []),
        "post_tool_ms": dist(d.get("post_tool_ms") for d in rounds),
        "v2v_ms": dist(d.get("v2v_ms") for d in plain),
        "turns_to_completion": dist(it.turns_to_completion for it in items),
        "calls_per_scenario": dist(it.calls for it in items),
        "unnecessary_calls": dist(it.unnecessary_calls for it in items),
    }
    metrics = {k: v for k, v in metrics.items() if v.n or k == "tool_round_latency_ms"}
    counts = {
        "scenarios": len(by_scenario),
        "trials": trials,
        "calls_run": len(items),
        "passed": sum(it.passed for it in items),
        "required_calls": required,
        "tool_calls": calls,
        "matched_calls": matched,
        "unnecessary_calls": sum(it.unnecessary_calls for it in items),
        "unexpected_writes": sum(it.unexpected_writes for it in items),
        "tool_errors": sum(it.tool_errors for it in items),
        "tool_rounds": len(rounds),
        "fillers": sum(bool(d.get("filler")) for d in rounds),
        "say_do_violations": sum(it.say_do_count for it in items),
        "hallucinated_turns": sum(it.hallucinated_turns for it in items),
        "turns": len(turns),
        "missed_turns": sum(bool(d.get("missed")) for d in turns),
        "errors": sum(len(it.errors) for it in items),
    }
    extra: dict[str, Any] = {
        "k": trials,
        "per_scenario": {
            sid: {"passed": v, "pass_rate": round(sum(v) / len(v), 6)}
            for sid, v in by_scenario.items()
        },
    }
    return metrics, rates, counts, extra


# ------------------------------------------------------------------------ report

_RATE_LABELS = {
    "pass_at_1": "**pass@1** (final state correct and facts said)",
    "pass_hat_k": "pass^k (every trial of a scenario passed)",
    "state_ok_rate": "final database state correct",
    "outputs_ok_rate": "expected facts said",
    "tool_precision": "tool-call precision",
    "tool_recall": "tool-call recall",
    "tool_f1": "**tool F1**",
    "arg_acc": "argument accuracy (matched calls)",
    "entity_capture_acc": "entity capture (names, emails, IDs)",
    "unnecessary_call_rate": "unnecessary calls (share of calls)",
    "say_do_violation_rate": "say-do violations (calls with one)",
    "hallucination_rate": "hallucinated tool results (agent turns)",
    "filler_rate": "tool rounds with a watchdog filler",
    "spoke_before_result_rate": "tool rounds with speech before the result",
    "tool_dead_air_rate": "tool rounds with dead air (no speech within the threshold)",
    "missed_rate": "missed turns (no reply)",
}
_METRIC_LABELS = {
    "tool_round_latency_ms": "**tool round** (speech end → first speech after the tool)",
    "first_response_ms": "first speech in a tool round (preamble / filler)",
    "pre_tool_ms": "speech end → first tool call",
    "tool_exec_ms": "tool execution (mock delay)",
    "post_tool_ms": "tool result → speech",
    "v2v_ms": "voice-to-voice, turns without tools",
    "turns_to_completion": "turns to completion",
    "calls_per_scenario": "tool calls per scenario",
    "unnecessary_calls": "unnecessary calls per scenario",
}
_ITEM_COLUMNS = (
    ("scenario", "scenario"),
    ("trial", "trial"),
    ("passed", "pass"),
    ("state_ok", "state"),
    ("outputs_ok", "said"),
    ("calls", "calls"),
    ("tool_precision", "precision"),
    ("tool_recall", "recall"),
    ("arg_acc", "args"),
    ("unnecessary_calls", "unneeded"),
    ("say_do_count", "say-do"),
    ("hallucinated_turns", "halluc."),
    ("turns_to_completion", "turns"),
    ("tool_round_ms", "tool round ms"),
)
_METHOD = """\
* Every scenario is one real-time call over the loopback transport (T1 caller, stereo
  recording, reference VAD {vad}). {caller}
* The agent gets the scenario's instructions and tools. Tools are deterministic mocks over
  a per-call database with a fixed delay ({delay:g} s; the refund tool 2 s, × {scale:g}).
* pass = the final database equals the expected state (initial state + the expected write
  calls, replayed with the same tools) **and** the agent said every expected fact.
  pass^k = every one of k={k} trials of a scenario passed.
* Calls are matched to expected calls by name (most correct arguments first); arguments
  are compared after normalization by kind (dates, times, emails, IDs, numbers, text).
  Optional expected calls (look-ups) count as matched but are not required.
* Say-do violation: a sentence of the agent (not a question, not negated) claims a write
  tool's action but no successful call of that tool happened by the end of the turn.
  Hallucination: a number or order status in the agent's text that no tool returned and
  the caller or instructions did not mention. Both use the agent's text transcript.
* Tool round latency: end of the user's speech → first agent speech onset (recording)
  after the last tool result of the turn. When the agent is still talking at the result (a
  preamble or filler running straight into the answer) the result time is used: a lower
  bound, flagged `answer_glued` in the turn details.
"""


def tools_markdown_table(results: RunResults) -> str:
    s = results.summary
    rows = [
        [
            label.replace("**", ""),
            "–" if s.rates.get(k) is None else f"{100 * float(s.rates[k] or 0):.0f}%",
        ]
        for k, label in _RATE_LABELS.items()
        if k in s.rates
    ]
    for key in ("tool_round_latency_ms", "post_tool_ms", "v2v_ms"):
        d = s.metrics.get(key)
        if d is not None and d.n:
            rows.append(
                [_METRIC_LABELS[key].replace("**", "") + " p50", f"{fmt(d.p50, 0)} ms (n={d.n})"]
            )
    return markdown_table([s.system, "value"], rows, ["l", "r"])


def _failures(results: RunResults) -> str:
    lines: list[str] = []
    for item in results.items:
        reasons: list[str] = []
        if not item.get("state_ok"):
            reasons.append("state: " + "; ".join(item.get("state_diff") or []))
        if item.get("missing_outputs"):
            reasons.append("not said: " + ", ".join(item["missing_outputs"]))
        if item.get("missing_calls"):
            reasons.append("missing calls: " + ", ".join(item["missing_calls"]))
        if item.get("wrong_args"):
            reasons.append("wrong arguments: " + ", ".join(item["wrong_args"]))
        if item.get("say_do_violations"):
            reasons.append("say-do: " + " / ".join(item["say_do_violations"]))
        if item.get("hallucinations"):
            reasons.append("ungrounded: " + ", ".join(item["hallucinations"]))
        if reasons:
            lines.append(f"* `{item['scenario']}` (trial {item['trial']}): " + " · ".join(reasons))
    return "\n".join(lines) or "None."


def render_tools_report(results: RunResults) -> str:
    opts = results.manifest.options
    vad = (opts.get("onset") or {}).get("reference_vad", {})
    policy = opts.get("caller_policy") or {"type": "scripted"}
    if policy.get("type") == "llm":
        llm = policy.get("llm") or {}
        caller = (
            f"An LLM plays the caller (`{llm.get('provider')}/{llm.get('model')}`, temperature "
            f"{policy.get('temperature')}, seed {policy.get('seed')}): persona and goal from the "
            "scenario, the scripted lines as the details it knows. It speaks after the agent "
            f"has answered and been quiet for {opts.get('gap_after_reply_s', 1.0):g} s, reacts "
            "to what the agent said and hangs up when done (its lines are in `caller_lines`)."
        )
    else:
        caller = (
            "The caller is scripted: it says the next line after the agent has answered and "
            f"been quiet for {opts.get('gap_after_reply_s', 1.0):g} s, whatever the agent said."
        )
    method = _METHOD.format(
        caller=caller,
        vad=", ".join(f"{k}={v}" for k, v in vad.items()) or "rms",
        delay=opts.get("tool_delay_s", 0.3),
        scale=opts.get("tool_delay_scale", 1.0),
        k=opts.get("trials", 1),
    )
    spec = ReportSpec(
        title=f"Tool use (T6) · {results.summary.system}",
        metric_labels=_METRIC_LABELS,
        rate_labels=_RATE_LABELS,
        item_columns=_ITEM_COLUMNS,
        sections=[("Findings", _failures(results)), ("Method", method)],
    )
    return render_report(results, spec)


# ------------------------------------------------------------------------- entry


def _dataset_id(suite: ToolSuite, stimuli: Sequence[Stimulus]) -> str:
    payload = json.dumps(
        {"definition": suite.definition_sha256(), "stimuli": [s.sha256 for s in stimuli]},
        sort_keys=True,
    )
    return f"{suite.name}@sha256:{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


async def run_tools_benchmark(
    system: BenchSystem,
    suite: ToolSuite,
    options: ToolsOptions | None = None,
    *,
    out_dir: str | os.PathLike[str] | None = None,
    run_id: str | None = None,
    detector: OnsetDetector | None = None,
    engine_factory: EngineFactory | None = None,
    on_turn: Callable[[str, int, TurnTiming], None] | None = None,
    on_scenario: Callable[[ToolScenarioItem], None] | None = None,
    caller_llm: LLM | None = None,
) -> RunResults:
    """Run the T6 tool-use track and (if ``out_dir``) write ``<out_dir>/<run_id>/``.

    Args:
        system: the system under test (its agent instructions and tools are replaced by
            each scenario's; voice, language and session options are kept).
        suite: the scenarios (:func:`~voice_agent_next.bench.tool_env.load_tool_suite`).
        engine_factory: build a fresh engine per call instead of sharing
            ``system.build_engine()`` (the scripted reference engine needs one per call).
        on_turn: progress ``(scenario id, trial, turn timing)``.
        on_scenario: called with each scored call.
        caller_llm: the LLM that plays the caller (default: created from
            ``options.caller`` when it is ``llm:<spec>``; not closed when passed in).
    """
    options = options or ToolsOptions()
    options.validate()
    suite = suite.select(options.scenarios)
    detector = detector or OnsetDetector()
    final_id = run_id or new_run_id(TRACK, system.label)
    directory: Path | None = None
    if out_dir is not None:
        directory = _reserve_directory(Path(out_dir), final_id, unique=run_id is None)
        final_id = directory.name
    created = utc_timestamp()
    t_start = now()
    base_agent = system.build_agent()
    caller_spec = parse_caller(options.caller)
    own_llm = caller_llm is None and caller_spec is not None
    if own_llm:
        caller_llm = create("llm", parse_component_spec(caller_spec))
    caller_tts: TTS | None = None
    try:
        if caller_llm is not None and suite.tts is not None:
            caller_tts = create("tts", suite.tts)  # one voice for every generated line
        stimuli = await render_stimuli(suite.stimulus_scenario(), tts=caller_tts)
        per_scenario: dict[str, list[Stimulus]] = {}
        for stim in stimuli:
            per_scenario.setdefault(stim.id.split("/", 1)[0], []).append(stim)
        shared = None if engine_factory is not None else system.build_engine()
        calls: list[_Call] = []
        items: list[ToolScenarioItem] = []
        try:
            if shared is not None and options.warmup_engine:
                await shared.warmup()
            index = 0
            for scenario in suite.scenarios:
                stim_scenario = suite.stimulus_scenario([scenario])
                lat = LatencyOptions(
                    turns=len(scenario.turns),
                    sessions=1,
                    warmup_turns=0,
                    dead_air_threshold=options.dead_air_threshold,
                    reply_timeout=options.reply_timeout,
                    gap_after_reply=options.gap_after_reply,
                    save_audio=options.save_audio,
                    warmup_engine=False,
                    seed=options.seed,
                    bootstrap_resamples=options.bootstrap_resamples,
                )
                for trial in range(options.trials):
                    call = await _run_call(
                        index,
                        suite,
                        scenario,
                        trial,
                        system,
                        base_agent,
                        shared,
                        engine_factory,
                        per_scenario[scenario.id],
                        stim_scenario,
                        lat,
                        options,
                        on_turn,
                        caller_llm,
                        caller_tts,
                    )
                    index += 1
                    analysis = await asyncio.to_thread(_analyze_session, call.run, detector, lat)
                    item = await asyncio.to_thread(
                        _analyze_call, suite, call, analysis.items, detector, options
                    )
                    call.analysis = analysis
                    calls.append(call)
                    items.append(item)
                    if on_scenario is not None:
                        on_scenario(item)
        finally:
            if shared is not None:
                await shared.aclose()
    except BaseException:
        if directory is not None and not any(directory.iterdir()):
            directory.rmdir()
        raise
    finally:
        if caller_tts is not None:
            await caller_tts.aclose()
        if own_llm and caller_llm is not None:
            await caller_llm.aclose()

    metrics, rates, counts, extra = summarize_tools(
        items, trials=options.trials, seed=options.seed, n_resamples=options.bootstrap_resamples
    )
    analyses = [c.analysis for c in calls]
    extra["sessions"] = [
        {**a.info, "scenario": c.scenario.id, "trial": c.trial}
        for a, c in zip(analyses, calls, strict=True)
    ]
    notes: list[str] = []
    if engine_factory is not None:
        notes.append("Scripted reference engine: a harness check, not a capability score.")
    llm_calls = [c.caller for c in calls if c.caller is not None]
    if llm_calls:
        notes.append(
            "LLM-driven caller: the conversations differ from the script (see `caller_lines`); "
            "compare with scripted runs only as a robustness check."
        )
        if suite.tts is None:
            notes.append(
                "The LLM caller speaks synthetic speech: only systems that do not transcribe "
                "the audio (e.g. the reference engine) can follow it; use --caller-tts."
            )
        hung_up = sum(c.ended for c in llm_calls)
        if hung_up < len(llm_calls):
            notes.append(
                f"{len(llm_calls) - hung_up} LLM-caller call(s) ended at the turn limit or on "
                "an error instead of the caller hanging up."
            )
    if counts["missed_turns"]:
        notes.append(f"{counts['missed_turns']} turn(s) got no reply within the reply timeout.")
    if any(a.info.get("aborted") for a in analyses):
        notes.append("At least one call closed early (see errors).")
    manifest = RunManifest(
        run_id=final_id,
        track=TRACK,
        created=created,
        system=system.describe(),
        scenario={
            "name": suite.name,
            "version": suite.version,
            "sha256": suite.definition_sha256(),
            "definition": suite.model_dump(mode="json"),
            "scenarios": [
                {
                    "id": s.id,
                    "sha256": suite.scenario_sha256(s),
                    "tools": s.tools,
                    "turns": len(s.turns),
                    "tags": s.tags,
                }
                for s in suite.scenarios
            ],
            "stimuli": [s.describe() for s in stimuli],
        },
        transport={
            "type": "loopback",
            "realtime_playout": True,
            "input_format": str(AudioFormat(suite.sample_rate, 1)),
            "output_format": str(_AGENT_FORMAT),
            "chunk_ms": round(suite.chunk * 1000, 3),
        },
        options={
            **asdict(options),
            "scenarios": [s.id for s in suite.scenarios],
            "caller": suite.stimuli if suite.tts is None else {"tts": suite.tts},
            "caller_policy": (
                {"type": "scripted"}
                if caller_llm is None
                else {"type": "llm", **describe_caller(caller_llm, options.caller_options())}
            ),
            "tool_delay_s": suite.tool_delay,
            "reply_timeout_s": options.reply_timeout or suite.reply_timeout,
            "gap_after_reply_s": (
                suite.gap_after_reply
                if options.gap_after_reply is None
                else options.gap_after_reply
            ),
            "onset": detector.describe(),
            "reference_engine": engine_factory is not None,
        },
        environment=await asyncio.to_thread(collect_environment),
        notes=notes,
    )
    summary = RunSummary(
        run_id=final_id,
        track=TRACK,
        system=system.label,
        transport="loopback",
        dataset=_dataset_id(suite, stimuli),
        n=len(items),
        metrics=metrics,
        rates=rates,
        counts=counts,
        extra=extra,
        duration_s=round(now() - t_start, 3),
    )
    results = RunResults(manifest, [it.model_dump(mode="json") for it in items], summary)
    results.report = render_tools_report(results)
    if directory is not None:
        write_run(directory, results)
        if options.save_audio:
            await asyncio.to_thread(_write_artifacts, directory, [c.run for c in calls], analyses)
    return results


async def _run_call(
    index: int,
    suite: ToolSuite,
    scenario: ToolScenario,
    trial: int,
    system: BenchSystem,
    base_agent: Agent,
    shared: S2SEngine | None,
    engine_factory: EngineFactory | None,
    stimuli: Sequence[Stimulus],
    stim_scenario: Any,
    lat: LatencyOptions,
    options: ToolsOptions,
    on_turn: Callable[[str, int, TurnTiming], None] | None,
    caller_llm: LLM | None = None,
    caller_tts: TTS | None = None,
) -> _Call:
    db = suite.initial_db(scenario)
    log: list[CallRecord] = []
    tools = build_tools(suite, scenario, db, log, delay_scale=options.tool_delay_scale)
    agent = Agent(
        suite.instructions_for(scenario),
        tools=tools,
        voice=base_agent.voice,
        language=base_agent.language,
    )
    proxy = _ScenarioSystem(system.config, system.label, agent=agent)
    fillers: list[float] = []
    snapshots: dict[int, str] = {}
    heard: list[str] = []  # the agent's transcript since the caller last spoke
    caller = (
        None
        if caller_llm is None
        else LLMCaller(caller_llm, suite, scenario, options.caller_options(), tts=caller_tts)
    )
    source: Sequence[Stimulus] | NextStimulus = stimuli
    if caller is not None:
        llm_caller = caller

        async def next_stimulus(index: int, _turns: Sequence[TurnTiming]) -> Stimulus | None:
            reply = "".join(heard).strip()
            heard.clear()
            line = await llm_caller.next_line(reply)
            return None if line is None else await llm_caller.render(index, line)

        source = next_stimulus

    def hook(session: AgentSession, _t: LoopbackTransport) -> None:
        session.on("tool_filler", lambda _e: fillers.append(now()))
        session.on("agent_transcript", lambda ev: heard.append(ev.delta))

    def turn_done(turn: TurnTiming) -> None:
        snapshots[turn.index] = state_hash(db)
        if on_turn is not None:
            on_turn(scenario.id, trial, turn)

    engine = engine_factory(suite, scenario) if engine_factory is not None else shared
    assert engine is not None
    try:
        run = await _run_session(
            index, engine, proxy, source, stim_scenario, lat, turn_done, on_start=hook
        )
    finally:
        if engine_factory is not None:
            await engine.aclose()
    return _Call(scenario, trial, run, log, fillers, snapshots, db, caller)
