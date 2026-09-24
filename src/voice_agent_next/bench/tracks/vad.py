"""T4 VAD track: frame accuracy and speed of voice activity detectors.

``van bench vad --vad energy --vad silero --vad sherpa-onnx/ten-vad`` runs every VAD over the
same deterministic, frame-labelled corpus (:mod:`voice_agent_next.bench.vad_corpus`) in
every noise condition, through the public streaming API (``VAD.stream()``, 20 ms chunks),
and reports (research note 06, §8.3 T4):

* **frame metrics** on 10 ms frames, a frame being speech when the VAD's (smoothed)
  probability of the window containing it reaches the VAD's ``activation_threshold``:
  precision, recall, F1, accuracy, false-alarm rate (non-speech frames called speech),
  miss rate, and ROC-AUC over the probabilities;
* **segment metrics** from the VAD's own start/end-of-speech events (its hysteresis and
  minimum durations included — what a cascade reacts to):

  * ``onset_lag_ms`` — ``START_OF_SPEECH`` fired − labelled start of the utterance;
  * ``offset_lag_ms`` — the utterance's last ``END_OF_SPEECH`` fired − labelled end
    (includes ``min_silence_duration``: the time a cascade waits before it even considers
    ending the turn);
  * ``missed_utterances`` — utterances without any detected speech;
  * ``false_alarms_per_min`` — ``START_OF_SPEECH`` events outside every utterance
    (± 100 ms), per minute of non-speech audio;

* **speed**: real-time factor (inference time / audio time) of the whole stream.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from ...audio.frame import AudioFrame
from ...registry import create
from ...utils.clock import now
from ...vad import VAD, VADEventType
from ..environment import collect_environment
from ..eot_metrics import roc_auc
from ..report import ReportSpec, fmt, markdown_table, render_report
from ..results import Distribution, RunManifest, RunResults, RunSummary, new_run_id, utc_timestamp
from ..results import write_run as _write_run
from ..vad_corpus import FRAME, VadClip, VadCorpus
from .latency import _reserve_directory

__all__ = [
    "TRACK",
    "VadClipResult",
    "VadOptions",
    "evaluate_clip",
    "render_vad_report",
    "run_vad_benchmark",
    "run_vad_on_clip",
    "vad_markdown_table",
]

TRACK = "vad"
_ONSET_TOLERANCE = 0.1


@dataclass
class VadOptions:
    chunk_ms: float = 20.0
    """Audio pushed per ``push_audio`` call."""
    warmup: bool = True
    seed: int = 0

    def validate(self) -> None:
        if not 1.0 <= self.chunk_ms <= 1000.0:
            raise ValueError("chunk_ms must be in [1, 1000]")


@dataclass
class VadStreamOutput:
    """What one VAD produced on one clip."""

    frame_probs: np.ndarray
    starts: list[float]
    ends: list[float]
    inference_s: float
    wall_s: float


class VadClipResult(BaseModel):
    """One VAD on one condition (a ``kind: "clip"`` line of ``items.jsonl``)."""

    model_config = ConfigDict(extra="forbid")

    kind: str = "clip"
    vad: str
    condition: str
    duration_s: float
    threshold: float
    precision: float | None
    recall: float | None
    f1: float | None
    accuracy: float
    false_alarm_rate: float | None
    miss_rate: float | None
    roc_auc: float | None
    utterances: int
    missed_utterances: int
    merged_utterances: int
    """Utterances whose end the VAD never reported before the next one began."""
    false_alarms: int
    nonspeech_min: float
    false_alarms_per_min: float | None
    onset_lag_ms: list[float] = []
    offset_lag_ms: list[float] = []
    rtf: float | None = None


def run_vad_on_clip(vad: VAD, audio: AudioFrame, chunk_ms: float = 20.0) -> VadStreamOutput:
    """Stream ``audio`` through ``vad``; per-10 ms-frame probability and event times."""
    stream = vad.stream(emit_inference_events=True)
    step = max(1, round(audio.sample_rate * chunk_ms / 1000.0))
    probs: list[float] = []
    starts: list[float] = []
    ends: list[float] = []
    inference = 0.0
    t0 = time.perf_counter()
    try:
        x = audio.data
        for k in range(0, len(x), step * 2):
            frame = AudioFrame(x[k : k + step * 2], audio.sample_rate, 1)
            for ev in stream.push_audio(frame):
                if ev.type == VADEventType.INFERENCE_DONE:
                    probs.append(ev.probability)
                    inference += ev.inference_duration
                elif ev.type == VADEventType.START_OF_SPEECH:
                    starts.append(ev.audio_time)
                else:
                    ends.append(ev.audio_time)
    finally:
        stream.close()
    wall = time.perf_counter() - t0
    n_frames = round(audio.duration / FRAME)
    window = vad.window_duration
    if probs:
        mids = (np.arange(n_frames) + 0.5) * FRAME
        idx = np.minimum((mids / window).astype(np.int64), len(probs) - 1)
        frame_probs = np.asarray(probs, dtype=np.float64)[idx]
    else:
        frame_probs = np.zeros(n_frames)
    return VadStreamOutput(frame_probs, starts, ends, inference, wall)


def evaluate_clip(
    name: str, clip: VadClip, out: VadStreamOutput, threshold: float
) -> VadClipResult:
    labels = clip.labels
    n = min(len(labels), len(out.frame_probs))
    lab, prob = labels[:n], out.frame_probs[:n]
    pred = prob >= threshold
    tp = int(np.sum(pred & lab))
    fp = int(np.sum(pred & ~lab))
    fn = int(np.sum(~pred & lab))
    tn = int(np.sum(~pred & ~lab))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    # segment level
    utts = clip.utterances
    speaking: list[tuple[float, float]] = []  # detected speech intervals (fire times)
    ends = sorted(out.ends)
    for s in sorted(out.starts):
        e = next((t for t in ends if t > s), clip.duration)
        speaking.append((s, e))
    onset: list[float] = []
    offset: list[float] = []
    missed = merged = 0
    for k, u in enumerate(utts):
        nxt = utts[k + 1].start if k + 1 < len(utts) else clip.duration
        detected = any(a < u.end + 0.5 and b > u.start for a, b in speaking)
        if not detected:
            missed += 1
            continue
        first = next((t for t in sorted(out.starts)
                      if u.start - _ONSET_TOLERANCE <= t <= u.end), None)  # fmt: skip
        # an interval already open before the utterance (a false alarm or the previous
        # utterance's hangover) has no onset of its own
        open_before = any(a < u.start - _ONSET_TOLERANCE and b > u.start for a, b in speaking)
        if first is not None and not open_before:
            onset.append(round((first - u.start) * 1000.0, 3))
        last_end = [t for t in ends if u.start <= t < nxt]
        if last_end:
            offset.append(round((last_end[-1] - u.end) * 1000.0, 3))
        else:
            merged += 1
    windows = [(u.start - _ONSET_TOLERANCE, u.end + _ONSET_TOLERANCE) for u in utts]
    false_alarms = sum(1 for s in out.starts if not any(a <= s <= b for a, b in windows))
    speech_time = sum(b - a for a, b in windows)
    nonspeech_min = max(0.0, clip.duration - speech_time) / 60.0
    return VadClipResult(
        vad=name,
        condition=clip.condition.name,
        duration_s=round(clip.duration, 3),
        threshold=threshold,
        precision=None if precision is None else round(precision, 6),
        recall=None if recall is None else round(recall, 6),
        f1=None if f1 is None else round(f1, 6),
        accuracy=round((tp + tn) / n, 6) if n else 0.0,
        false_alarm_rate=round(fp / (fp + tn), 6) if fp + tn else None,
        miss_rate=round(fn / (fn + tp), 6) if fn + tp else None,
        roc_auc=_auc(prob, lab),
        utterances=len(utts),
        missed_utterances=missed,
        merged_utterances=merged,
        false_alarms=false_alarms,
        nonspeech_min=round(nonspeech_min, 4),
        false_alarms_per_min=round(false_alarms / nonspeech_min, 4) if nonspeech_min else None,
        onset_lag_ms=onset,
        offset_lag_ms=offset,
        rtf=round(out.inference_s / clip.duration, 6) if clip.duration else None,
    )


def _auc(prob: np.ndarray, lab: np.ndarray) -> float | None:
    value = roc_auc(prob[lab].tolist(), prob[~lab].tolist())
    return None if value is None else round(value, 6)


# ----------------------------------------------------------------------- summary


def _vad_name(vad: VAD, spec: Any) -> str:
    if isinstance(spec, str):
        return spec
    return (
        f"{vad.provider}/{vad.model}" if vad.model and vad.model != vad.provider else vad.provider
    )


def summarize_vad(
    results: Sequence[VadClipResult], *, seed: int = 0
) -> tuple[dict[str, Distribution], dict[str, float | None], dict[str, int], dict[str, Any]]:
    metrics: dict[str, Distribution] = {}
    rates: dict[str, float | None] = {}
    table: list[dict[str, Any]] = []
    for name in dict.fromkeys(r.vad for r in results):
        mine = [r for r in results if r.vad == name]
        on = [x for r in mine for x in r.onset_lag_ms]
        off = [x for r in mine for x in r.offset_lag_ms]
        metrics[f"{name}.onset_lag_ms"] = Distribution.of(on, seed=seed)
        metrics[f"{name}.offset_lag_ms"] = Distribution.of(off, seed=seed)
        f1s = [r.f1 for r in mine if r.f1 is not None]
        rates[f"{name}.f1"] = round(float(np.mean(f1s)), 6) if f1s else None
        aucs = [r.roc_auc for r in mine if r.roc_auc is not None]
        rates[f"{name}.roc_auc"] = round(float(np.mean(aucs)), 6) if aucs else None
        for r in mine:
            table.append(
                {
                    "vad": name, "condition": r.condition, "precision": r.precision,
                    "recall": r.recall, "f1": r.f1, "roc_auc": r.roc_auc,
                    "false_alarm_rate": r.false_alarm_rate, "miss_rate": r.miss_rate,
                    "onset_p50_ms": _p50(r.onset_lag_ms), "offset_p50_ms": _p50(r.offset_lag_ms),
                    "missed_utterances": r.missed_utterances,
                    "false_alarms_per_min": r.false_alarms_per_min, "rtf": r.rtf,
                }
            )  # fmt: skip
    counts = {
        "vads": len({r.vad for r in results}),
        "conditions": len({r.condition for r in results}),
        "utterances": sum(r.utterances for r in results),
        "missed_utterances": sum(r.missed_utterances for r in results),
        "false_alarms": sum(r.false_alarms for r in results),
    }
    return metrics, rates, counts, {"table": table}


def _p50(xs: Sequence[float]) -> float | None:
    return round(float(np.percentile(xs, 50)), 3) if xs else None


# ------------------------------------------------------------------------ report


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def vad_markdown_table(results: RunResults) -> str:
    rows = [
        [
            r["vad"], r["condition"], _pct(r["precision"]), _pct(r["recall"]), _pct(r["f1"]),
            fmt(r["roc_auc"], 3), _pct(r["false_alarm_rate"]), fmt(r["onset_p50_ms"], 0),
            fmt(r["offset_p50_ms"], 0), fmt(r["missed_utterances"]),
            fmt(r["false_alarms_per_min"], 1), fmt(r["rtf"], 4),
        ]
        for r in results.summary.extra.get("table", [])
    ]  # fmt: skip
    return markdown_table(
        ["VAD", "condition", "precision", "recall", "F1", "AUC", "false alarm", "onset p50 ms",
         "offset p50 ms", "missed", "FA/min", "RTF"],
        rows, ["l", "l"] + ["r"] * 10,
    )  # fmt: skip


_METHOD = """\
* Corpus: {n_utt} utterances laid out with {gap_lo:g}–{gap_hi:g} s gaps (seed {seed}),
  {lead:g} s of noise before and {tail:g} s after, speech at -20 dBFS; one clip per condition,
  same layout. Corpus SHA-256 `{sha}` (manifest: per-clip hashes and sources).
* Labels: 10 ms frames of each clean utterance within 40 dB of its loudest frame and 10 dB
  above its noise floor; dips < 150 ms bridged. Utterance span = first to last speech frame.
* Frame metrics at each VAD's activation threshold (probability of the window containing
  the frame's midpoint). AUC over the probabilities.
* Onset lag = `START_OF_SPEECH` fired − labelled start; offset lag = the utterance's last
  `END_OF_SPEECH` fired − labelled end (includes `min_silence_duration`). False alarms:
  `START_OF_SPEECH` outside every utterance (± 100 ms), per minute of the rest.
* Audio is streamed in {chunk:g} ms chunks through `VAD.stream()`, faster than real time.
  RTF = inference time / audio duration.
"""


def vad_report_spec(results: RunResults) -> ReportSpec:
    params = (results.manifest.scenario.get("corpus") or {}).get("params", {})
    method = _METHOD.format(
        n_utt=len(params.get("sources", [])), gap_lo=params.get("gap_s", [0, 0])[0],
        gap_hi=params.get("gap_s", [0, 0])[1], seed=params.get("seed", 0),
        lead=params.get("lead_s", 0), tail=params.get("tail_s", 0),
        sha=str((results.manifest.scenario.get("corpus") or {}).get("sha256", ""))[:16],
        chunk=results.manifest.options.get("chunk_ms", 20),
    )  # fmt: skip
    labels = {k: k.replace(".", " · ") for k in results.summary.metrics}
    return ReportSpec(
        title=f"VAD (T4) · {results.summary.system}",
        metric_labels=labels,
        rate_labels={
            k: k.replace(".", " · ") + " (mean over conditions)" for k in results.summary.rates
        },
        sections=[("Per condition", vad_markdown_table(results)), ("Method", method)],
    )


def render_vad_report(results: RunResults) -> str:
    return render_report(results, vad_report_spec(results))


# ------------------------------------------------------------------------- entry


async def run_vad_benchmark(
    vads: Sequence[VAD | Any],
    corpus: VadCorpus,
    options: VadOptions | None = None,
    *,
    dataset: str = "custom",
    out_dir: str | os.PathLike[str] | None = None,
    run_id: str | None = None,
    label: str | None = None,
    on_clip: Callable[[VadClipResult], None] | None = None,
) -> RunResults:
    """Run the T4 VAD track and (if ``out_dir``) write ``<out_dir>/<run_id>/``.

    Args:
        vads: VAD instances or registry specs (``"energy"``, ``"silero"``...).
        corpus: the labelled corpus (:func:`~voice_agent_next.bench.vad_corpus.build_vad_corpus`).
        dataset: name of the source utterances (for the summary).
    """
    options = options or VadOptions()
    options.validate()
    if not vads:
        raise ValueError("no VAD")
    instances: list[tuple[str, VAD, bool]] = []
    for spec in vads:
        own = not isinstance(spec, VAD)
        vad: VAD = create("vad", spec) if own else spec
        instances.append((_vad_name(vad, spec), vad, own))
    names = [n for n, _, _ in instances]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate VADs: {names}")
    system_label = label or " vs ".join(names)
    final_id = run_id or new_run_id(TRACK, system_label)
    directory: Path | None = None
    if out_dir is not None:
        directory = _reserve_directory(Path(out_dir), final_id, unique=run_id is None)
        final_id = directory.name
    created = utc_timestamp()
    t_start = now()
    items: list[VadClipResult] = []
    described: list[dict[str, Any]] = []
    try:
        for name, vad, _ in instances:
            if options.warmup:
                await vad.warmup()
            described.append(
                {"name": name, "class": f"{type(vad).__module__}.{type(vad).__qualname__}",
                 "provider": vad.provider, "model": vad.model, "sample_rate": vad.sample_rate,
                 "window_ms": round(vad.window_duration * 1000, 3),
                 "options": asdict(vad.options)}
            )  # fmt: skip
            for clip in corpus.clips:
                out = await asyncio.to_thread(run_vad_on_clip, vad, clip.audio, options.chunk_ms)
                result = evaluate_clip(name, clip, out, vad.options.activation_threshold)
                items.append(result)
                if on_clip is not None:
                    on_clip(result)
    except BaseException:
        if directory is not None and not any(directory.iterdir()):
            directory.rmdir()
        raise
    finally:
        for _, vad, own in instances:
            if own:
                await vad.aclose()
    metrics, rates, counts, extra = summarize_vad(items, seed=options.seed)
    manifest = RunManifest(
        run_id=final_id,
        track=TRACK,
        created=created,
        system={"label": system_label, "vads": described},
        scenario={"dataset": dataset, "corpus": corpus.describe()},
        transport={"type": "offline", "chunk_ms": options.chunk_ms},
        options=asdict(options),
        environment=await asyncio.to_thread(collect_environment),
    )
    summary = RunSummary(
        run_id=final_id,
        track=TRACK,
        system=system_label,
        transport="offline",
        dataset=f"{dataset}@sha256:{corpus.sha256[:12]}",
        n=len(items),
        metrics=metrics,
        rates=rates,
        counts=counts,
        extra=extra,
        duration_s=round(now() - t_start, 3),
    )
    results = RunResults(manifest, [it.model_dump(mode="json") for it in items], summary)
    results.report = render_vad_report(results)
    if directory is not None:
        _write_run(directory, results)
    return results
