"""End-of-turn policy metrics, following LiveKit's eot-bench (research note 06, §4.4).

Every silence span of a user turn is a decision point: the last one is the true end of
the turn (``eot``), the others are mid-turn pauses (``hold``). A detector scores each span
once, ``score_point`` seconds into the silence (the moment a cascade's VAD reports the
pause). An endpointing **policy** turns that score into a decision with three knobs:

* ``threshold`` — the score must be **above** it for the model to fire;
* ``action_delay`` — minimum silence before the agent may answer (the model fires at
  ``max(action_delay, score_time)``);
* ``timeout`` — silence after which the turn ends even if the model did not fire.

For a policy:

* a **false cutoff** is a hold span that the policy ends: the model fired and the pause
  lasted longer than the fire time, or the pause outlasted ``timeout``.
  ``cutoff_rate`` = false cutoffs / hold spans;
* ``mean_latency`` = mean over eot spans of the fire time (or ``timeout`` when the model
  did not fire in time) — dead air after the user finished, not inference time.

:func:`sweep_policies` evaluates the whole grid (eot-bench's defaults: thresholds 0..1 in
0.01 steps, action delays 0.2..1.0 s in 0.1 s steps, timeouts 1.0..3.5 s in 0.5 s steps)
plus the silence-only **VAD baseline** (answer after a fixed silence). The operating points
are the lowest false-cutoff rate within a latency budget (300 / 600 ms) and the lowest
latency within a false-cutoff budget (5 / 10 %). This is a numpy port of
``eot_harness/metrics.py`` (Apache-2.0) in its score-point mode.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

__all__ = [
    "DEFAULT_ACTION_DELAYS",
    "DEFAULT_THRESHOLDS",
    "DEFAULT_TIMEOUTS",
    "PolicyPoint",
    "ScoredSpan",
    "classification_metrics",
    "evaluate_policy",
    "filter_spans",
    "min_cutoff_under_latency",
    "min_latency_under_cutoff",
    "pareto_front",
    "roc_auc",
    "sweep_policies",
]

DEFAULT_THRESHOLDS: tuple[float, ...] = tuple(round(0.01 * i, 2) for i in range(101))
DEFAULT_ACTION_DELAYS: tuple[float, ...] = tuple(round(0.2 + 0.1 * i, 2) for i in range(9))
DEFAULT_TIMEOUTS: tuple[float, ...] = tuple(round(1.0 + 0.5 * i, 2) for i in range(6))
MIN_HOLD_SPAN = 0.2
MAX_HOLD_SPAN = 5.0
EPS = 1e-9

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ScoredSpan:
    """A decision point: label, silence duration and the detector's score (``None``: the
    span ended before the score point, so the model never saw it)."""

    label: Literal["hold", "eot"]
    duration: float
    p_eot: float | None
    score_time: float


@dataclass(frozen=True, slots=True)
class PolicyPoint:
    policy: Literal["model", "vad"]
    threshold: float | None
    action_delay: float
    timeout: float
    cutoff_rate: float
    mean_latency: float
    """Seconds."""
    timeout_rate: float
    """Share of eot spans that ended by timeout (the model did not fire in time)."""
    f1: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def filter_spans(
    spans: Sequence[ScoredSpan],
    *,
    min_hold: float = MIN_HOLD_SPAN,
    max_hold: float = MAX_HOLD_SPAN,
) -> list[ScoredSpan]:
    """eot-bench's denominator: hold spans between ``min_hold`` and ``max_hold`` seconds,
    every eot span."""
    return [
        s for s in spans if s.label == "eot" or (min_hold - EPS <= s.duration <= max_hold + EPS)
    ]


def _arrays(spans: Sequence[ScoredSpan], label: str) -> tuple[FloatArray, FloatArray, FloatArray]:
    sel = [s for s in spans if s.label == label]
    dur = np.array([s.duration for s in sel], dtype=np.float64)
    p = np.array([np.nan if s.p_eot is None else s.p_eot for s in sel], dtype=np.float64)
    st = np.array([s.score_time for s in sel], dtype=np.float64)
    return dur, p, st


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def sweep_policies(
    spans: Sequence[ScoredSpan],
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    action_delays: Sequence[float] = DEFAULT_ACTION_DELAYS,
    timeouts: Sequence[float] = DEFAULT_TIMEOUTS,
    score_point: float = 0.2,
    include_model: bool = True,
    include_vad: bool = True,
) -> list[PolicyPoint]:
    """Every (threshold, action delay, timeout) policy plus the VAD baseline."""
    hold_dur, hold_p, hold_st = _arrays(spans, "hold")
    eot_dur, eot_p, eot_st = _arrays(spans, "eot")
    if not len(hold_dur) or not len(eot_dur):
        raise ValueError("need both hold and eot spans")
    n_hold, n_eot = len(hold_dur), len(eot_dur)
    points: list[PolicyPoint] = []
    model_delays = [d for d in action_delays if d >= score_point - EPS]
    if include_model:
        for thr in thresholds:
            hold_fire = np.where(hold_p > thr, hold_st, np.inf)  # NaN > thr is False
            eot_fire = np.where(eot_p > thr, eot_st, np.inf)
            for ad in model_delays:
                hold_t = np.maximum(ad, hold_fire)
                model_cut = np.isfinite(hold_fire) & (hold_dur > hold_t + EPS)
                eot_t = np.maximum(ad, eot_fire)
                for to in timeouts:
                    if to + EPS < ad:
                        continue
                    cut = model_cut | (hold_dur > to + EPS)
                    detect = np.isfinite(eot_fire) & (eot_t <= to + EPS)
                    latency = np.where(detect, eot_t, to)
                    tp = int(detect.sum())
                    fp = int(cut.sum())
                    points.append(
                        PolicyPoint(
                            policy="model",
                            threshold=float(thr),
                            action_delay=float(ad),
                            timeout=float(to),
                            cutoff_rate=fp / n_hold,
                            mean_latency=float(latency.mean()),
                            timeout_rate=1.0 - tp / n_eot,
                            f1=_f1(tp, fp, n_eot - tp),
                        )
                    )
    if include_vad:
        for d in sorted(set(action_delays) | set(timeouts)):
            fp = int((hold_dur > d + EPS).sum())
            points.append(
                PolicyPoint(
                    policy="vad", threshold=None, action_delay=float(d), timeout=float(d),
                    cutoff_rate=fp / n_hold, mean_latency=float(d), timeout_rate=0.0,
                    f1=_f1(n_eot, fp, 0),
                )
            )  # fmt: skip
    return points


def evaluate_policy(
    spans: Sequence[ScoredSpan],
    *,
    threshold: float,
    action_delay: float,
    timeout: float,
    inclusive: bool = True,
) -> PolicyPoint:
    """One policy; ``inclusive``: the model fires at ``p >= threshold`` (the cascade's rule)
    instead of eot-bench's ``p > threshold``."""
    thr = math.nextafter(threshold, -math.inf) if inclusive else threshold
    (point,) = sweep_policies(
        spans, thresholds=[thr], action_delays=[action_delay], timeouts=[timeout],
        score_point=0.0,
        include_vad=False,
    )  # fmt: skip
    return PolicyPoint(
        "model", threshold, point.action_delay, point.timeout, point.cutoff_rate,
        point.mean_latency, point.timeout_rate, point.f1,
    )  # fmt: skip


def min_cutoff_under_latency(
    points: Sequence[PolicyPoint], budget: float, policy: str = "model"
) -> PolicyPoint | None:
    """Lowest false-cutoff rate with ``mean_latency <= budget`` (seconds)."""
    feasible = [p for p in points if p.policy == policy and p.mean_latency <= budget + EPS]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda p: (p.cutoff_rate, p.mean_latency, -(p.threshold or 0.0), p.action_delay,
                       p.timeout),
    )  # fmt: skip


def min_latency_under_cutoff(
    points: Sequence[PolicyPoint], budget: float, policy: str = "model"
) -> PolicyPoint | None:
    """Lowest mean latency with ``cutoff_rate <= budget`` (0..1)."""
    feasible = [p for p in points if p.policy == policy and p.cutoff_rate <= budget + EPS]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda p: (p.mean_latency, p.cutoff_rate, p.timeout_rate, p.timeout, p.action_delay,
                       p.threshold or 0.0),
    )  # fmt: skip


def pareto_front(points: Sequence[PolicyPoint], policy: str = "model") -> list[PolicyPoint]:
    """Policies not dominated in (cutoff rate, mean latency), by increasing cutoff rate."""
    pts = sorted(
        (p for p in points if p.policy == policy), key=lambda p: (p.cutoff_rate, p.mean_latency)
    )
    front: list[PolicyPoint] = []
    best = math.inf
    for p in pts:
        if p.mean_latency < best - EPS:
            front.append(p)
            best = p.mean_latency
    return front


def roc_auc(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    """Area under the ROC curve (Mann-Whitney U with ties counted as 1/2)."""
    pos = np.asarray(positives, dtype=np.float64)
    neg = np.asarray(negatives, dtype=np.float64)
    if not pos.size or not neg.size:
        return None
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):  # average ranks of ties
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def classification_metrics(
    spans: Sequence[ScoredSpan], threshold: float
) -> dict[str, float | int | None]:
    """Complete (eot, positive) vs incomplete (hold) at the detector's threshold
    (``p >= threshold``), over the spans that were scored."""
    scored = [s for s in spans if s.p_eot is not None]
    pos = [s.p_eot for s in scored if s.label == "eot" and s.p_eot is not None]
    neg = [s.p_eot for s in scored if s.label == "hold" and s.p_eot is not None]
    tp = sum(p >= threshold for p in pos)
    fn = len(pos) - tp
    fp = sum(p >= threshold for p in neg)
    tn = len(neg) - fp
    n = tp + fn + fp + tn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": (tp + tn) / n if n else None,
        "precision": precision, "recall": recall, "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "roc_auc": roc_auc(pos, neg),
    }  # fmt: skip
