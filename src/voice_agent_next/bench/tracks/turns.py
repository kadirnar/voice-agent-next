"""T4 end-of-turn track: how well a turn detector tells finished turns from pauses.

``van bench turns --detector smart_turn`` scores every silence span of every turn of an
end-of-turn dataset (default: LiveKit's eot-bench, English; see
:mod:`voice_agent_next.bench.eot_datasets`) and reports (research note 06, §8.3 T4):

* **policy metrics** (eot-bench, :mod:`voice_agent_next.bench.eot_metrics`): the lowest
  false-cutoff rate within a 300 / 600 ms latency budget (``false_cutoff_at_300ms`` ...)
  and the lowest mean end-of-turn latency within a 5 / 10 % false-cutoff budget
  (``latency_at_5pct_ms`` ...), for the detector and for the silence-only VAD baseline,
  plus the Pareto frontier;
* the **configured policy**: what the cascade does with the detector by default (fires at
  ``p >= threshold`` after ``min_endpointing_delay`` = 0.4 s, else waits
  ``max_endpointing_delay`` = 2.5 s): its false-cutoff rate and mean latency;
* **classification** at the detector's threshold, complete (eot) vs incomplete (hold)
  turns: accuracy, precision, recall, F1, false-positive rate, ROC-AUC;
* **decision latency**: the detector's inference time per prediction (``inference_ms``).

The detector sees what it would see live: the turn's audio from its start up to
``score_point`` seconds into the silence, and the conversation so far plus the user's
words that ended at least ``transcript_lag`` seconds earlier (an STT's lag) — the same
causal inputs as eot-bench's adapters.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from ...chat import ChatContext
from ...registry import create
from ...turn import TurnDetector
from ...utils.clock import now
from ..environment import collect_environment
from ..eot_datasets import EotDataset, EotTurn
from ..eot_metrics import (
    DEFAULT_ACTION_DELAYS,
    DEFAULT_THRESHOLDS,
    DEFAULT_TIMEOUTS,
    PolicyPoint,
    ScoredSpan,
    classification_metrics,
    evaluate_policy,
    filter_spans,
    min_cutoff_under_latency,
    min_latency_under_cutoff,
    pareto_front,
    sweep_policies,
)
from ..report import ReportSpec, fmt, markdown_table, render_report
from ..results import Distribution, RunManifest, RunResults, RunSummary, new_run_id, utc_timestamp
from ..results import write_run as _write_run
from .latency import _reserve_directory

__all__ = [
    "TRACK",
    "TurnsItem",
    "TurnsOptions",
    "render_turns_report",
    "run_turns_benchmark",
    "summarize_turns",
    "turns_markdown_table",
]

TRACK = "turns"
LATENCY_BUDGETS_MS = (300, 600)
CUTOFF_BUDGETS_PCT = (5, 10)


@dataclass
class TurnsOptions:
    score_point: float = 0.2
    """Seconds into a silence at which the detector is asked (eot-bench scores Smart Turn
    at 0.2 s; a cascade asks when its VAD reports the pause)."""
    transcript_lag: float = 0.5
    """Words reach text detectors this long after they were spoken."""
    min_endpointing_delay: float = 0.4
    """The configured policy's action delay (the cascade's default with a detector)."""
    max_endpointing_delay: float = 2.5
    """The configured policy's timeout."""
    threshold: float | None = None
    """Decision threshold (default: the detector's own)."""
    limit: int | None = None
    """Only the first N turns of each dataset."""
    warmup: bool = True
    seed: int = 0

    def validate(self) -> None:
        if self.score_point <= 0:
            raise ValueError("score_point must be > 0")
        if self.transcript_lag < 0:
            raise ValueError("transcript_lag must be >= 0")
        if self.max_endpointing_delay < self.min_endpointing_delay:
            raise ValueError("max_endpointing_delay must be >= min_endpointing_delay")
        if self.threshold is not None and not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")


class TurnsItem(BaseModel):
    """One silence span (a line of ``items.jsonl``)."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    turn: str
    language: str | None = None
    span: int
    label: str
    start_s: float
    duration_s: float
    scored: bool
    """The span lasted until the score point (shorter hold spans are never scored)."""
    counted: bool
    """Part of the eot-bench denominator (hold spans of 0.2–5 s, every eot span)."""
    p_eot: float | None = None
    inference_ms: float | None = None
    error: str | None = None


# ------------------------------------------------------------------------ inputs


def _chat_context(turn: EotTurn, timestamp: float, lag: float) -> ChatContext:
    ctx = ChatContext()
    for role, content in turn.messages:
        if role in ("user", "assistant", "system") and content.strip():
            ctx.add_message(role, content)  # type: ignore[arg-type]
    visible = [w.strip() for s, e, w in turn.words if e <= timestamp - lag + 1e-6 and w.strip()]
    if visible:
        ctx.add_message("user", " ".join(visible))
    return ctx


async def _score_turn(
    detector: TurnDetector, dataset: str, turn: EotTurn, options: TurnsOptions
) -> list[TurnsItem]:
    audio = turn.load_audio()
    items: list[TurnsItem] = []
    for span in turn.spans:
        scored = span.duration >= options.score_point - 1e-6
        counted = span.label == "eot" or 0.2 - 1e-9 <= span.duration <= 5.0 + 1e-9
        item = TurnsItem(
            dataset=dataset, turn=turn.id, language=turn.language, span=span.index,
            label=span.label, start_s=round(span.start, 6), duration_s=round(span.duration, 6),
            scored=scored, counted=counted,
        )  # fmt: skip
        if scored:
            t = span.start + options.score_point
            clip = audio.slice(0.0, min(t, audio.duration))
            ctx = _chat_context(turn, t, options.transcript_lag)
            t0 = time.perf_counter()
            try:
                p = await detector.predict_end_of_turn(audio=clip, chat_ctx=ctx)
            except Exception as exc:  # recorded per item; the run goes on
                item.error = f"{type(exc).__name__}: {exc}"
            else:
                item.p_eot = round(float(p), 6)
                item.inference_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        items.append(item)
    return items


# ----------------------------------------------------------------------- summary


def _point(p: PolicyPoint | None) -> dict[str, Any] | None:
    if p is None:
        return None
    d = p.as_dict()
    d["mean_latency_ms"] = round(d.pop("mean_latency") * 1000.0, 1)
    d["cutoff_rate"] = round(d["cutoff_rate"], 6)
    d["timeout_rate"] = round(d["timeout_rate"], 6)
    d["f1"] = round(d["f1"], 6)
    return d


def _scored_spans(items: Sequence[TurnsItem], score_point: float) -> list[ScoredSpan]:
    return filter_spans(
        [
            ScoredSpan(it.label, it.duration_s, it.p_eot, score_point)  # type: ignore[arg-type]
            for it in items
            if it.error is None
        ]
    )


def _operating_points(points: Sequence[PolicyPoint], policy: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ms in LATENCY_BUDGETS_MS:
        out[f"false_cutoff_at_{ms}ms"] = _point(min_cutoff_under_latency(points, ms / 1000, policy))
    for pct in CUTOFF_BUDGETS_PCT:
        out[f"latency_at_{pct}pct"] = _point(min_latency_under_cutoff(points, pct / 100, policy))
    return out


def _headline(ops: dict[str, Any]) -> dict[str, float | None]:
    h: dict[str, float | None] = {}
    for ms in LATENCY_BUDGETS_MS:
        p = ops[f"false_cutoff_at_{ms}ms"]
        h[f"false_cutoff_at_{ms}ms"] = None if p is None else p["cutoff_rate"]
    for pct in CUTOFF_BUDGETS_PCT:
        p = ops[f"latency_at_{pct}pct"]
        h[f"latency_at_{pct}pct_ms"] = None if p is None else p["mean_latency_ms"]
    return h


def _evaluate(
    items: Sequence[TurnsItem], options: TurnsOptions, threshold: float
) -> dict[str, Any]:
    spans = _scored_spans(items, options.score_point)
    labels = {s.label for s in spans}
    if labels != {"hold", "eot"}:
        return {"n_hold": sum(s.label == "hold" for s in spans),
                "n_eot": sum(s.label == "eot" for s in spans)}  # fmt: skip
    points = sweep_policies(
        spans, thresholds=DEFAULT_THRESHOLDS, action_delays=DEFAULT_ACTION_DELAYS,
        timeouts=DEFAULT_TIMEOUTS, score_point=options.score_point,
    )  # fmt: skip
    model_ops = _operating_points(points, "model")
    vad_ops = _operating_points(points, "vad")
    configured = evaluate_policy(
        spans,
        threshold=threshold,
        action_delay=max(options.min_endpointing_delay, options.score_point),
        timeout=options.max_endpointing_delay,
    )
    front = pareto_front(points, "model")
    return {
        "n_hold": sum(s.label == "hold" for s in spans),
        "n_eot": sum(s.label == "eot" for s in spans),
        "detector": _headline(model_ops),
        "vad_baseline": _headline(vad_ops),
        "operating_points": {"detector": model_ops, "vad_baseline": vad_ops},
        "configured_policy": _point(configured),
        "classification": classification_metrics(spans, threshold),
        "pareto_front": [
            [
                round(p.cutoff_rate, 4),
                round(p.mean_latency * 1000, 1),
                p.threshold,
                p.action_delay,
                p.timeout,
            ]
            for p in front
            if p.cutoff_rate <= 0.3
        ],
    }


def summarize_turns(
    items: Sequence[TurnsItem],
    options: TurnsOptions,
    threshold: float,
    *,
    n_resamples: int = 2000,
) -> tuple[dict[str, Distribution], dict[str, float | None], dict[str, int], dict[str, Any]]:
    overall = _evaluate(items, options, threshold)
    metrics = {
        "inference_ms": Distribution.of(
            (it.inference_ms for it in items), seed=options.seed, n_resamples=n_resamples
        ),
        "p_eot_hold": Distribution.of(
            (it.p_eot for it in items if it.label == "hold" and it.counted), seed=options.seed,
            n_resamples=n_resamples,
        ),
        "p_eot_eot": Distribution.of(
            (it.p_eot for it in items if it.label == "eot"), seed=options.seed,
            n_resamples=n_resamples,
        ),
    }  # fmt: skip
    rates: dict[str, float | None] = {}
    cls = overall.get("classification") or {}
    for key in ("accuracy", "precision", "recall", "f1", "false_positive_rate", "roc_auc"):
        rates[key] = cls.get(key)
    head = overall.get("detector") or {}
    for key, value in head.items():
        if key.startswith("false_cutoff"):
            rates[key] = value
    conf = overall.get("configured_policy") or {}
    rates["configured_cutoff_rate"] = conf.get("cutoff_rate")
    counts = {
        "turns": len({(it.dataset, it.turn) for it in items}),
        "spans": len(items),
        "hold_spans": overall.get("n_hold", 0),
        "eot_spans": overall.get("n_eot", 0),
        "scored": sum(it.p_eot is not None for it in items),
        "errors": sum(it.error is not None for it in items),
    }
    by_dataset: dict[str, Any] = {}
    datasets = sorted({it.dataset for it in items})
    if len(datasets) > 1:
        for name in datasets:
            by_dataset[name] = _evaluate([it for it in items if it.dataset == name], options,
                                         threshold)  # fmt: skip
    extra = {"threshold": threshold, **overall, "datasets": by_dataset}
    return metrics, rates, counts, extra


# ------------------------------------------------------------------------ report


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def _ms(x: float | None) -> str:
    return "–" if x is None else f"{x:,.0f} ms"


def turns_markdown_table(results: RunResults) -> str:
    """eot-bench operating points: the detector and the VAD baseline (for PRs)."""
    s = results.summary
    x = s.extra
    rows = []
    for name, key in ((s.system, "detector"), ("VAD baseline (silence only)", "vad_baseline")):
        h = x.get(key) or {}
        rows.append(
            [
                name,
                _pct(h.get("false_cutoff_at_300ms")),
                _pct(h.get("false_cutoff_at_600ms")),
                _ms(h.get("latency_at_5pct_ms")),
                _ms(h.get("latency_at_10pct_ms")),
            ]
        )
    table = markdown_table(
        ["system", "false cutoffs @ 300 ms", "false cutoffs @ 600 ms", "latency @ 5% cutoffs",
         "latency @ 10% cutoffs"],
        rows, ["l", "r", "r", "r", "r"],
    )  # fmt: skip
    conf = x.get("configured_policy")
    cls = x.get("classification") or {}
    inf = s.metrics.get("inference_ms")
    lines = [table, ""]
    if conf:
        lines.append(
            f"Configured policy (p ≥ {conf['threshold']:g}, action delay "
            f"{conf['action_delay']:g} s, timeout {conf['timeout']:g} s): false cutoffs "
            f"{_pct(conf['cutoff_rate'])}, mean end-of-turn latency {_ms(conf['mean_latency_ms'])}."
        )
    if cls.get("n"):
        lines.append(
            f"At threshold {x.get('threshold'):g}: accuracy {_pct(cls.get('accuracy'))}, "
            f"precision {_pct(cls.get('precision'))}, recall {_pct(cls.get('recall'))}, "
            f"F1 {_pct(cls.get('f1'))}, ROC-AUC {fmt(cls.get('roc_auc'), 3)}; "
            f"inference p50 {fmt(inf.p50 if inf else None, 1)} ms."
        )
    return "\n".join(lines) + "\n"


_METHOD = """\
* Dataset: every silence span ≥ 100 ms of every turn; the last span of a turn is its end
  (`eot`), earlier spans are mid-turn pauses (`hold`). Hold spans of 0.2–5 s and every eot
  span count (eot-bench's denominator).
* The detector is asked once per span, {score_point:g} s into the silence, with the turn's
  audio up to that moment and the words that ended ≥ {lag:g} s before it (plus the prior
  conversation). Hold spans shorter than that are never scored (the model cannot fire).
* A policy fires when `p > threshold`, at `max(action_delay, {score_point:g} s)`, or ends the
  turn at `timeout`. False cutoff: a hold span the policy ends. Latency: mean over eot spans
  of the fire time (or the timeout). Grid: thresholds 0–1 (0.01), action delays 0.2–1.0 s
  (0.1 s), timeouts 1.0–3.5 s (0.5 s); the VAD baseline answers after a fixed silence.
  `false cutoffs @ 300 ms` = the lowest false-cutoff rate of any policy with mean latency
  ≤ 300 ms; `latency @ 5%` = the lowest mean latency with ≤ 5 % false cutoffs (LiveKit
  eot-bench methodology, numpy port).
* Configured policy: `p ≥ threshold` → answer after {min_ep:g} s, else after {max_ep:g} s (the
  cascade's defaults with a turn detector).
"""


def turns_report_spec(results: RunResults) -> ReportSpec:
    opts = results.manifest.options
    method = _METHOD.format(
        score_point=opts.get("score_point", 0.2),
        lag=opts.get("transcript_lag", 0.5),
        min_ep=opts.get("min_endpointing_delay", 0.4),
        max_ep=opts.get("max_endpointing_delay", 2.5),
    )
    sections = [("Operating points", turns_markdown_table(results))]
    by_dataset = results.summary.extra.get("datasets") or {}
    if by_dataset:
        rows = [
            [name, _pct((d.get("detector") or {}).get("false_cutoff_at_300ms")),
             _pct((d.get("detector") or {}).get("false_cutoff_at_600ms")),
             _ms((d.get("detector") or {}).get("latency_at_5pct_ms")),
             _ms((d.get("detector") or {}).get("latency_at_10pct_ms"))]
            for name, d in by_dataset.items()
        ]  # fmt: skip
        sections.append(
            ("By dataset", markdown_table(["dataset", "fc @ 300 ms", "fc @ 600 ms", "lat @ 5%",
                                           "lat @ 10%"], rows, ["l", "r", "r", "r", "r"]))
        )  # fmt: skip
    sections.append(("Method", method))
    return ReportSpec(
        title=f"End-of-turn detection (T4) · {results.summary.system}",
        metric_labels={
            "inference_ms": "inference time per prediction (ms)",
            "p_eot_hold": "score at mid-turn pauses (hold)",
            "p_eot_eot": "score at turn ends (eot)",
        },
        rate_labels={
            "false_cutoff_at_300ms": "false cutoffs @ 300 ms latency",
            "false_cutoff_at_600ms": "false cutoffs @ 600 ms latency",
            "configured_cutoff_rate": "false cutoffs, configured policy",
            "accuracy": "accuracy (complete vs incomplete)",
            "precision": "precision (eot)",
            "recall": "recall (eot)",
            "f1": "F1 (eot)",
            "false_positive_rate": "false-positive rate (hold scored as eot)",
            "roc_auc": "ROC-AUC",
        },
        sections=sections,
        digits=1,
    )


def render_turns_report(results: RunResults) -> str:
    return render_report(results, turns_report_spec(results))


# ------------------------------------------------------------------------- entry


def _describe(detector: TurnDetector, spec: Any) -> dict[str, Any]:
    return {
        "spec": spec if isinstance(spec, (str, dict)) else None,
        "class": f"{type(detector).__module__}.{type(detector).__qualname__}",
        "provider": detector.provider,
        "model": detector.model,
        "modality": detector.modality,
        "threshold": detector.threshold,
    }


async def run_turns_benchmark(
    detector: TurnDetector | Any,
    datasets: EotDataset | Sequence[EotDataset],
    options: TurnsOptions | None = None,
    *,
    out_dir: str | os.PathLike[str] | None = None,
    run_id: str | None = None,
    label: str | None = None,
    on_turn: Callable[[EotTurn, list[TurnsItem]], None] | None = None,
) -> RunResults:
    """Run the T4 end-of-turn track and (if ``out_dir``) write ``<out_dir>/<run_id>/``.

    Args:
        detector: a :class:`~voice_agent_next.turn.TurnDetector` or a registry spec
            (``"smart_turn"``, ``{"provider": "smart_turn", "model": "v3.2-gpu"}``).
        datasets: end-of-turn datasets (:mod:`voice_agent_next.bench.eot_datasets`).
    """
    options = options or TurnsOptions()
    options.validate()
    sets = [datasets] if isinstance(datasets, EotDataset) else list(datasets)
    if not sets:
        raise ValueError("no dataset")
    sets = [d.limit(options.limit) for d in sets]
    own = not isinstance(detector, TurnDetector)
    instance: TurnDetector = create("turn", detector) if own else detector
    system_label = label or f"{instance.provider}/{instance.model}"
    final_id = run_id or new_run_id(TRACK, system_label)
    directory: Path | None = None
    if out_dir is not None:
        directory = _reserve_directory(Path(out_dir), final_id, unique=run_id is None)
        final_id = directory.name
    created = utc_timestamp()
    t_start = now()
    try:
        warmup_ms = None
        if options.warmup:
            t0 = now()
            await instance.warmup()
            warmup_ms = round((now() - t0) * 1000.0, 3)
        items: list[TurnsItem] = []
        for data in sets:
            for turn in data.turns:
                scored = await _score_turn(instance, data.name, turn, options)
                items += scored
                if on_turn is not None:
                    on_turn(turn, scored)
    except BaseException:
        if directory is not None and not any(directory.iterdir()):
            directory.rmdir()
        raise
    finally:
        if own:
            await instance.aclose()
    threshold = options.threshold if options.threshold is not None else instance.threshold
    metrics, rates, counts, extra = summarize_turns(items, options, threshold)
    extra["warmup_ms"] = warmup_ms
    notes: list[str] = []
    if counts["errors"]:
        notes.append(f"{counts['errors']} prediction(s) failed (see `error` in items.jsonl).")
    if not extra.get("detector"):
        notes.append("Policy metrics need both mid-turn pauses and turn ends.")
    manifest = RunManifest(
        run_id=final_id,
        track=TRACK,
        created=created,
        system={"label": system_label, "detector": _describe(instance, detector)},
        scenario={"datasets": [d.describe() for d in sets]},
        transport={"type": "offline"},
        options={**asdict(options), "threshold_used": threshold},
        environment=await asyncio.to_thread(collect_environment),
        notes=notes,
    )
    summary = RunSummary(
        run_id=final_id,
        track=TRACK,
        system=system_label,
        transport="offline",
        dataset=", ".join(d.id for d in sets),
        n=counts["hold_spans"] + counts["eot_spans"],
        metrics=metrics,
        rates=rates,
        counts=counts,
        extra=extra,
        duration_s=round(now() - t_start, 3),
    )
    results = RunResults(manifest, [it.model_dump(mode="json") for it in items], summary)
    results.report = render_turns_report(results)
    if directory is not None:
        _write_run(directory, results)
    return results
