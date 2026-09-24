"""T2 ASR track: accuracy and speed of any STT provider (research note 06, §4.1 and §8.3).

Every utterance of the dataset is transcribed by the STT under test, in one of two modes:

* ``batch`` — :meth:`STT.transcribe` on the whole utterance (what a VAD-segmented cascade
  does with a non-streaming recognizer);
* ``streaming`` — :meth:`STT.stream`: the audio is pushed in ``chunk_ms`` chunks, paced like
  a capture device (chunk *i* is delivered when its interval has elapsed) at
  ``realtime_factor`` × real time (``1``: real time, the default; ``2``: twice as fast;
  ``0``: as fast as possible). At the end of the audio the harness calls
  :meth:`STTStream.end_input` (``flush()`` + end of input), exactly as a cascade's VAD /
  turn detector does when the user stops, and waits for the final transcript.

Metrics (per utterance in ``items.jsonl``, aggregated in ``summary.json``):

* ``wer`` / ``cer`` — **corpus** error rates Σ(S+D+I) / ΣN after normalization (see
  :mod:`voice_agent_next.bench.text_norm`), per dataset; languages written without spaces
  (zh, ja, ko, th...) are scored by CER. Also the per-utterance distribution
  (``wer_pct``/``cer_pct``) and the share of perfect utterances.
* ``rtfx`` — Σ audio duration / Σ processing time (batch: ``transcribe()`` duration;
  streaming: first chunk -> final transcript). Only meaningful in batch mode or
  unpaced streaming (``realtime_factor=0``); real-time pacing caps it near 1.
* ``ttfs_ms`` — **final latency**: end of audio -> final transcript. Streaming: the
  ``end_input()``/``flush()`` call -> the last ``FINAL_TRANSCRIPT`` (Pipecat's TTFS with the
  harness as the VAD: the VAD stop is the end of the file); batch: the ``transcribe()``
  duration (a batch recognizer starts when the audio ends). This is the STT's share of a
  voice agent's response time.
* ``first_partial_ms`` — streaming: audio start (first chunk's capture start) -> first
  non-empty interim transcript.
* ``interim_revision_rate`` — streaming stability: share of interim updates that rewrite
  already-shown words (the previous interim is not a prefix of the next, after
  normalization) instead of only appending. ``0`` = interims only grow.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ...audio.frame import AudioFrame
from ...audio.resample import StreamResampler
from ...config import ComponentSpec
from ...registry import create
from ...stt import STT, StreamAdapter, STTEventType
from ...utils.clock import now
from ..asr_datasets import AsrDataset, AsrUtterance, load_audio
from ..environment import collect_environment
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
from ..system import redact
from ..text_norm import cer_text, get_normalizer, normalizer_for, uses_cer
from ..wer import EditCounts, char_counts, word_counts

__all__ = [
    "TRACK",
    "AsrItem",
    "AsrOptions",
    "asr_markdown_table",
    "asr_report_spec",
    "render_asr_report",
    "run_asr_benchmark",
    "summarize_asr",
]

TRACK = "asr"
Mode = Literal["batch", "streaming"]


@dataclass
class AsrOptions:
    """How the ASR track runs."""

    mode: Mode = "batch"
    chunk_ms: float = 20.0
    """Streaming: chunk size pushed to the recognizer."""
    realtime_factor: float = 1.0
    """Streaming pacing: 1 = real time, 2 = twice as fast, 0 = as fast as possible."""
    normalizer: str = "auto"
    """``auto`` (Whisper English for en, basic otherwise), ``whisper-english``,
    ``whisper-basic`` or ``none``."""
    language: str | None = None
    """Override the dataset language passed to the recognizer (and used for scoring)."""
    limit: int | None = None
    """Only the first N utterances of each dataset."""
    warmup: bool = True
    """``stt.warmup()`` and one unmeasured utterance before the first measured one."""
    final_timeout: float = 30.0
    """Streaming: seconds to wait for the final transcript after the end of the audio."""
    seed: int = 0
    """Bootstrap seed."""
    bootstrap_resamples: int = 2000

    def validate(self) -> None:
        if self.mode not in ("batch", "streaming"):
            raise ValueError(f"mode must be 'batch' or 'streaming', not {self.mode!r}")
        if not 1.0 <= self.chunk_ms <= 1000.0:
            raise ValueError("chunk_ms must be between 1 and 1000")
        if self.realtime_factor < 0 or not math.isfinite(self.realtime_factor):
            raise ValueError("realtime_factor must be >= 0 (0 = as fast as possible)")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be >= 1")
        if self.final_timeout <= 0:
            raise ValueError("final_timeout must be > 0")
        get_normalizer(normalizer_for("en", self.normalizer))  # validates the name


class AsrItem(BaseModel):
    """One utterance (a line of ``items.jsonl``). Durations in ms, rates as fractions."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    id: str
    language: str | None = None
    duration_s: float
    reference: str
    hypothesis: str = ""
    reference_norm: str = ""
    hypothesis_norm: str = ""
    metric: Literal["wer", "cer"] = "wer"
    """Headline metric of the utterance's language."""
    words: int = 0
    """Reference words after normalization (``N`` of WER)."""
    word_errors: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    wer: float | None = None
    chars: int = 0
    char_errors: int = 0
    cer: float | None = None
    processing_ms: float | None = None
    rtf: float | None = None
    """``processing_ms / duration``."""
    ttfs_ms: float | None = None
    first_partial_ms: float | None = None
    interims: int = 0
    interim_updates: int = 0
    interim_revisions: int = 0
    finals: int = 0
    timed_out: bool = False
    error: str | None = None


# ------------------------------------------------------------------ recognition


@dataclass
class _Recognition:
    text: str = ""
    processing_s: float | None = None
    ttfs_s: float | None = None
    first_partial_s: float | None = None
    interims: list[str] = field(default_factory=list)
    interim_segments: list[str | None] = field(default_factory=list)
    finals: int = 0
    timed_out: bool = False


async def _recognize_batch(stt: STT, audio: AudioFrame, language: str | None) -> _Recognition:
    t0 = now()
    transcript = await stt.transcribe(audio, language=language)
    elapsed = now() - t0
    return _Recognition(
        text=transcript.text.strip(), processing_s=elapsed, ttfs_s=elapsed, finals=1
    )


def _chunks(audio: AudioFrame, sample_rate: int, chunk_ms: float) -> list[AudioFrame]:
    """Resample once to the recognizer rate (so the stream does no work) and cut chunks."""
    rs = StreamResampler(sample_rate, 1)
    frame = AudioFrame.concat([rs.push(audio), rs.flush()])
    step = max(1, round(sample_rate * chunk_ms / 1000.0)) * 2
    data = frame.data
    return [AudioFrame(data[i : i + step], sample_rate, 1) for i in range(0, len(data), step)]


async def _recognize_streaming(
    stt: STT, audio: AudioFrame, language: str | None, options: AsrOptions
) -> _Recognition:
    chunks = _chunks(audio, stt.sample_rate, options.chunk_ms)
    stream = stt.stream(language=language)
    finals: list[tuple[float, str]] = []
    rec = _Recognition()
    t_start = now()

    async def consume() -> None:
        async for ev in stream:
            if ev.type == STTEventType.INTERIM_TRANSCRIPT and ev.text.strip():
                if rec.first_partial_s is None:
                    rec.first_partial_s = ev.timestamp - t_start
                rec.interims.append(ev.text.strip())
                rec.interim_segments.append(ev.segment_id)
            elif ev.type == STTEventType.FINAL_TRANSCRIPT:
                finals.append((ev.timestamp, ev.text.strip()))

    consumer = asyncio.create_task(consume(), name="asr-bench-consumer")
    try:
        t = 0.0
        for chunk in chunks:
            t += chunk.duration
            if options.realtime_factor > 0:
                # deliver chunk i when its capture interval has elapsed (capture-device model)
                delay = t_start + t / options.realtime_factor - now()
                await asyncio.sleep(max(0.0, delay))
            else:
                await asyncio.sleep(0)
            stream.push_audio(chunk)
        t_flush = now()
        stream.end_input()
        try:
            await asyncio.wait_for(asyncio.shield(consumer), options.final_timeout)
        except TimeoutError:
            rec.timed_out = True
    finally:
        if not consumer.done():
            consumer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer
        await stream.aclose()
    rec.finals = len(finals)
    rec.text = " ".join(text for _, text in finals if text)
    if finals:
        t_final = finals[-1][0]
        rec.ttfs_s = max(0.0, t_final - t_flush)
        rec.processing_s = t_final - t_start
    return rec


def _revisions(interims: Sequence[str], segments: Sequence[str | None], norm: Any) -> int:
    """Interim updates that are not pure extensions of the previous interim (same segment)."""
    count = 0
    for i in range(1, len(interims)):
        if segments[i] != segments[i - 1]:
            continue
        prev, cur = norm(interims[i - 1]).split(), norm(interims[i]).split()
        # the last word of the previous interim may legitimately grow ("hel" -> "hello")
        stable = prev[:-1] == cur[: len(prev) - 1] if prev else True
        if not stable:
            count += 1
    return count


def _updates(segments: Sequence[str | None]) -> int:
    return sum(1 for i in range(1, len(segments)) if segments[i] == segments[i - 1])


def _score(
    dataset: str,
    utt: AsrUtterance,
    duration: float,
    rec: _Recognition | None,
    *,
    language: str | None,
    normalizer: str,
    error: str | None = None,
) -> AsrItem:
    norm = get_normalizer(normalizer_for(language, normalizer))
    item = AsrItem(
        dataset=dataset,
        id=utt.id,
        language=language,
        duration_s=round(duration, 3),
        reference=utt.text,
        reference_norm=norm(utt.text),
        metric="cer" if uses_cer(language) else "wer",
        error=error,
    )
    if rec is None:
        return item
    item.hypothesis = rec.text
    item.hypothesis_norm = norm(rec.text)
    w = word_counts(item.reference_norm, item.hypothesis_norm)
    c = char_counts(
        cer_text(item.reference_norm, language), cer_text(item.hypothesis_norm, language)
    )
    item.words, item.word_errors = w.ref_len, w.errors
    item.substitutions, item.deletions, item.insertions = w.substitutions, w.deletions, w.insertions
    item.wer = None if w.rate is None else round(w.rate, 6)
    item.chars, item.char_errors = c.ref_len, c.errors
    item.cer = None if c.rate is None else round(c.rate, 6)
    ms = lambda s: None if s is None else round(s * 1000.0, 3)  # noqa: E731
    item.processing_ms = ms(rec.processing_s)
    if rec.processing_s is not None and duration > 0:
        item.rtf = round(rec.processing_s / duration, 6)
    item.ttfs_ms = ms(rec.ttfs_s)
    item.first_partial_ms = ms(rec.first_partial_s)
    item.interims = len(rec.interims)
    item.interim_updates = _updates(rec.interim_segments)
    item.interim_revisions = _revisions(rec.interims, rec.interim_segments, norm)
    item.finals = rec.finals
    item.timed_out = rec.timed_out
    return item


# ---------------------------------------------------------------------- summary


def _pct(x: float | None) -> float | None:
    return None if x is None else x * 100.0


def _group_stats(items: Sequence[AsrItem]) -> dict[str, Any]:
    scored = [it for it in items if it.error is None]
    words = EditCounts(
        hits=sum(it.words - it.substitutions - it.deletions for it in scored),
        substitutions=sum(it.substitutions for it in scored),
        deletions=sum(it.deletions for it in scored),
        insertions=sum(it.insertions for it in scored),
    )
    n_chars = sum(it.chars for it in scored)
    char_errors = sum(it.char_errors for it in scored)
    audio = sum(it.duration_s for it in scored)
    processing = [it.processing_ms for it in scored if it.processing_ms is not None]
    wer = words.rate
    cer = char_errors / n_chars if n_chars else None
    metric = "cer" if scored and all(it.metric == "cer" for it in scored) else "wer"
    headline = cer if metric == "cer" else wer
    updates = sum(it.interim_updates for it in scored)
    ttfs = Distribution.of(it.ttfs_ms for it in scored)
    partial = Distribution.of(it.first_partial_ms for it in scored)
    return {
        "n": len(items),
        "scored": len(scored),
        "audio_s": round(audio, 3),
        "language": items[0].language if items else None,
        "metric": metric,
        "wer": None if wer is None else round(wer, 6),
        "cer": None if cer is None else round(cer, 6),
        "headline": None if headline is None else round(headline, 6),
        "words": words.ref_len,
        "word_errors": words.errors,
        "substitutions": sum(it.substitutions for it in scored),
        "deletions": sum(it.deletions for it in scored),
        "insertions": sum(it.insertions for it in scored),
        "chars": n_chars,
        "char_errors": char_errors,
        "perfect_rate": round(
            sum((it.cer if it.metric == "cer" else it.wer) == 0 for it in scored) / len(scored), 6
        )
        if scored
        else None,
        "rtfx": round(audio / (sum(processing) / 1000.0), 3)
        if processing and len(processing) == len(scored) and sum(processing) > 0
        else None,
        "ttfs_p50_ms": ttfs.p50,
        "ttfs_p90_ms": ttfs.p90,
        "first_partial_p50_ms": partial.p50,
        "interim_revision_rate": round(sum(it.interim_revisions for it in scored) / updates, 6)
        if updates
        else None,
        "errors": sum(it.error is not None for it in items),
        "timeouts": sum(it.timed_out for it in items),
    }


def summarize_asr(
    items: Sequence[AsrItem], *, seed: int = 0, n_resamples: int = 2000
) -> tuple[dict[str, Distribution], dict[str, float | None], dict[str, int], dict[str, Any]]:
    """Distributions, rates, counts and extras (incl. the per-dataset breakdown)."""

    def dist(values: Iterable[float | None]) -> Distribution:
        return Distribution.of(values, seed=seed, n_resamples=n_resamples)

    scored = [it for it in items if it.error is None]
    metrics: dict[str, Distribution] = {
        "wer_pct": dist(_pct(it.wer) for it in scored),
        "cer_pct": dist(_pct(it.cer) for it in scored),
    }
    for key in ("ttfs_ms", "first_partial_ms", "processing_ms", "rtf"):
        d = dist(getattr(it, key) for it in scored)
        if d.n:
            metrics[key] = d
    pooled = _group_stats(items)
    datasets: dict[str, dict[str, Any]] = {}
    for it in items:
        datasets.setdefault(it.dataset, {})
    for name in datasets:
        datasets[name] = _group_stats([it for it in items if it.dataset == name])
    rates: dict[str, float | None] = {
        "wer": pooled["wer"],
        "cer": pooled["cer"],
        "perfect_rate": pooled["perfect_rate"],
    }
    if pooled["interim_revision_rate"] is not None:
        rates["interim_revision_rate"] = pooled["interim_revision_rate"]
    counts = {
        "utterances": len(items),
        "scored": pooled["scored"],
        "datasets": len(datasets),
        "words": pooled["words"],
        "word_errors": pooled["word_errors"],
        "substitutions": pooled["substitutions"],
        "deletions": pooled["deletions"],
        "insertions": pooled["insertions"],
        "chars": pooled["chars"],
        "char_errors": pooled["char_errors"],
        "errors": pooled["errors"],
        "timeouts": pooled["timeouts"],
    }
    extra: dict[str, Any] = {
        "rtfx": pooled["rtfx"],
        "audio_s": pooled["audio_s"],
        "datasets": datasets,
    }
    return metrics, rates, counts, extra


# ----------------------------------------------------------------------- report

_METRIC_LABELS = {
    "wer_pct": "WER per utterance (%)",
    "cer_pct": "CER per utterance (%)",
    "ttfs_ms": "**final latency** `ttfs_ms` (end of audio → final)",
    "first_partial_ms": "first partial (audio start → first interim)",
    "processing_ms": "processing time per utterance",
    "rtf": "RTF per utterance (processing / audio)",
}

_ITEM_COLUMNS = (
    ("dataset", "dataset"),
    ("id", "utterance"),
    ("duration_s", "audio s"),
    ("words", "words"),
    ("word_errors", "word errors"),
    ("char_errors", "char errors"),
    ("ttfs_ms", "TTFS ms"),
    ("first_partial_ms", "1st partial ms"),
    ("processing_ms", "processing ms"),
)


def _rate(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.2f}%"


def asr_markdown_table(results: RunResults) -> str:
    """One row per dataset: the table to paste into a PR or results page."""
    s = results.summary
    opts = results.manifest.options
    datasets: dict[str, dict[str, Any]] = s.extra.get("datasets", {})
    headers = ["system", "dataset", "mode", "n", "WER", "CER", "perfect", "RTFx",
               "TTFS p50", "TTFS p90", "1st partial p50"]  # fmt: skip
    rows = []
    for name, d in datasets.items():
        wer, cer = _rate(d.get("wer")), _rate(d.get("cer"))
        if d.get("metric") == "cer":
            cer = f"**{cer}**"
        else:
            wer = f"**{wer}**"
        rows.append([
            s.system, name, str(opts.get("mode", "?")), fmt(d.get("scored")), wer, cer,
            _rate(d.get("perfect_rate")), fmt(d.get("rtfx"), 1),
            fmt(d.get("ttfs_p50_ms"), 0, unit=" ms"), fmt(d.get("ttfs_p90_ms"), 0, unit=" ms"),
            fmt(d.get("first_partial_p50_ms"), 0, unit=" ms"),
        ])  # fmt: skip
    return markdown_table(headers, rows, ["l", "l", "l"] + ["r"] * 8)


_METHOD = """\
* **Error rates** are corpus rates Σ(S+D+I) / ΣN over each dataset, after normalizing
  reference and hypothesis with the `{normalizer}` normalizer (`auto`: Whisper's
  `EnglishTextNormalizer` for English, `BasicTextNormalizer` keeping combining marks
  otherwise). Bold = headline metric of the language (CER for zh/ja/ko/th and other
  languages written without spaces; their CER ignores whitespace). "Perfect" = share of
  utterances without a single error.
* **Mode `{mode}`.** {mode_text}
* **RTFx** = Σ audio / Σ processing time. **TTFS** (`ttfs_ms`) = end of audio → final
  transcript: the STT's share of a voice agent's response time.
* Percentiles are linear-interpolated; `[..]` is the 95% percentile-bootstrap CI of the
  median ({resamples} resamples, seed {seed}). Smoke subsets are regression canaries, not
  capability scores: 50 utterances give a WER CI of roughly ±1–2 points.
"""

_BATCH_TEXT = (
    "`STT.transcribe()` on each whole utterance; TTFS = processing time (a batch "
    "recognizer starts when the audio ends)."
)
_STREAM_TEXT = (
    "`STT.stream()`; audio pushed in {chunk_ms:g} ms chunks, each delivered when its "
    "interval has elapsed{pace}. At the end of the audio the harness calls `end_input()` "
    "(= `flush()` + end of input) — what a cascade does when its VAD ends the turn — and "
    "TTFS runs from that call to the last final transcript. First partial = first chunk's "
    "capture start → first non-empty interim. Interim revision rate = share of interim "
    "updates that rewrite earlier words instead of only appending."
)


def _errors(item: dict[str, Any]) -> int:
    key = "char_errors" if item.get("metric") == "cer" else "word_errors"
    return int(item.get(key) or 0)


def asr_report_spec(results: RunResults) -> ReportSpec:
    opts = results.manifest.options
    mode = str(opts.get("mode", "batch"))
    if mode == "streaming":
        rtf = float(opts.get("realtime_factor", 1.0))
        pace = (
            " (real time)" if rtf == 1.0
            else " (as fast as possible: latencies are not real-time numbers)" if rtf == 0
            else f" ({rtf:g}× real time)"
        )  # fmt: skip
        mode_text = _STREAM_TEXT.format(chunk_ms=float(opts.get("chunk_ms", 20.0)), pace=pace)
    else:
        mode_text = _BATCH_TEXT
    method = _METHOD.format(
        normalizer=opts.get("normalizer", "auto"),
        mode=mode,
        mode_text=mode_text,
        resamples=opts.get("bootstrap_resamples", 2000),
        seed=opts.get("seed", 0),
    )
    extra = results.summary.extra
    overview = asr_markdown_table(results)
    if extra.get("rtfx") is not None:
        overview += (
            f"\n\nOverall RTFx: {fmt(extra['rtfx'], 1)} over {fmt(extra.get('audio_s'), 1)} s "
            "of audio."
        )
    scored = [it for it in results.items if it.get("error") is None and _errors(it)]
    worst_rows = [
        [it["dataset"], it["id"], str(_errors(it)), it.get("reference_norm", ""),
         it.get("hypothesis_norm", "")]
        for it in sorted(scored, key=lambda it: -_errors(it))[:5]
    ]  # fmt: skip
    sections = [("Results by dataset", overview)]
    if worst_rows:
        sections.append((
            "Most errors (normalized text)",
            markdown_table(["dataset", "utterance", "errors", "reference", "hypothesis"],
                           worst_rows, ["l", "l", "r", "l", "l"]),
        ))  # fmt: skip
    sections.append(("Method", method))
    return ReportSpec(
        title=f"ASR (T2) · {results.summary.system}",
        metric_labels=_METRIC_LABELS,
        rate_labels={
            "wer": "WER (corpus, all datasets pooled)",
            "cer": "CER (corpus, all datasets pooled)",
            "perfect_rate": "perfect utterances",
            "interim_revision_rate": "interim revision rate",
        },
        item_columns=_ITEM_COLUMNS,
        sections=sections,
        digits=1,
    )


def render_asr_report(results: RunResults) -> str:
    return render_report(results, asr_report_spec(results))


# ------------------------------------------------------------------------ entry


def _describe_stt(stt: STT, spec: Any) -> dict[str, Any]:
    inner = stt.wrapped if isinstance(stt, StreamAdapter) else stt
    return {
        "label": _stt_label(spec),
        "kind": "stt",
        "config": redact(spec) if isinstance(spec, (dict, list, str)) else repr(spec),
        "stt": {
            "class": f"{type(inner).__module__}.{type(inner).__qualname__}",
            "provider": inner.provider,
            "model": inner.model,
            "sample_rate": inner.sample_rate,
            "language": inner.language,
            "capabilities": {
                k: getattr(inner.capabilities, k) for k in inner.capabilities.__dataclass_fields__
            },
            "stream_adapter": isinstance(stt, StreamAdapter),
        },
    }


def _stt_label(spec: Any) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, list):
        return "|".join(_stt_label(s) for s in spec)
    if isinstance(spec, dict):
        target = str(spec.get("provider") or spec.get("use") or "?")
        model = spec.get("model")
        return f"{target}/{model}" if model and "/" not in target else target
    return type(spec).__name__


def _reserve_directory(parent: Path, run_id: str, *, unique: bool) -> Path:
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


async def run_asr_benchmark(
    stt: STT | ComponentSpec,
    datasets: AsrDataset | Sequence[AsrDataset],
    options: AsrOptions | None = None,
    *,
    vad: Any = None,
    out_dir: str | os.PathLike[str] | None = None,
    run_id: str | None = None,
    label: str | None = None,
    on_item: Callable[[AsrItem], None] | None = None,
) -> RunResults:
    """Run the T2 ASR track and (if ``out_dir``) write ``<out_dir>/<run_id>/``.

    Args:
        stt: an STT instance or a registry spec (``"faster-whisper/base"``, a mapping).
        datasets: one or more datasets (:mod:`voice_agent_next.bench.asr_datasets`).
        options: mode, chunking, pacing, normalizer...
        vad: VAD (instance or spec) used to stream a batch-only recognizer
            (:class:`~voice_agent_next.stt.StreamAdapter`) in streaming mode.
        out_dir: parent directory of the run directory (``None``: nothing is written).
        run_id: defaults to ``<UTC time>-asr-<label>``.
        label: system label for reports (default: the STT spec).
        on_item: progress callback, called after every utterance.
    """
    options = options or AsrOptions()
    options.validate()
    sets = [datasets] if isinstance(datasets, AsrDataset) else list(datasets)
    if not sets:
        raise ValueError("no dataset")
    sets = [d.limit(options.limit) for d in sets]
    names = [d.name for d in sets]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate dataset names: {names}")
    spec: Any = stt
    instance: STT = stt if isinstance(stt, STT) else create("stt", stt)
    if options.mode == "streaming" and not instance.capabilities.streaming:
        if vad is None:
            raise ValueError(
                f"{type(instance).__name__} does not stream: use --mode batch, or pass a VAD "
                "(--vad) to stream it through StreamAdapter"
            )
        instance = StreamAdapter(instance, create("vad", vad))
    system_label = label or _stt_label(spec)
    final_id = run_id or new_run_id(TRACK, system_label)
    directory: Path | None = None
    if out_dir is not None:
        directory = _reserve_directory(Path(out_dir), final_id, unique=run_id is None)
        final_id = directory.name
    try:
        return await _run_asr(
            instance, spec, sets, options, directory=directory, run_id=final_id,
            label=system_label, on_item=on_item,
        )  # fmt: skip
    except BaseException:
        if directory is not None and not any(directory.iterdir()):
            directory.rmdir()
        raise
    finally:
        if not isinstance(stt, STT):
            await instance.aclose()


async def _recognize(
    stt: STT, audio: AudioFrame, language: str | None, options: AsrOptions
) -> _Recognition:
    if options.mode == "streaming":
        return await _recognize_streaming(stt, audio, language, options)
    return await _recognize_batch(stt, audio, language)


async def _run_asr(
    stt: STT,
    spec: Any,
    datasets: Sequence[AsrDataset],
    options: AsrOptions,
    *,
    directory: Path | None,
    run_id: str,
    label: str,
    on_item: Callable[[AsrItem], None] | None,
) -> RunResults:
    created = utc_timestamp()
    t_start = now()
    warmup_ms: float | None = None
    first_ms: float | None = None
    notes: list[str] = []
    if options.warmup:
        t0 = now()
        await stt.warmup()
        warmup_ms = (now() - t0) * 1000.0
        first = datasets[0].utterances[0]
        audio = await asyncio.to_thread(load_audio, first.audio)
        t0 = now()
        try:
            await _recognize(stt, audio, options.language or first.language, options)
        except Exception as exc:
            notes.append(f"The warm-up utterance failed: {exc!r}")
        first_ms = (now() - t0) * 1000.0

    items: list[AsrItem] = []
    for dataset in datasets:
        for utt in dataset.utterances:
            language = options.language or utt.language or dataset.language
            audio = await asyncio.to_thread(load_audio, utt.audio)
            try:
                rec = await _recognize(stt, audio, language, options)
                item = _score(
                    dataset.name, utt, audio.duration, rec,
                    language=language, normalizer=options.normalizer,
                )  # fmt: skip
            except Exception as exc:
                item = _score(
                    dataset.name, utt, audio.duration, None,
                    language=language, normalizer=options.normalizer, error=repr(exc),
                )  # fmt: skip
            items.append(item)
            if on_item is not None:
                on_item(item)

    metrics, rates, counts, extra = summarize_asr(
        items, seed=options.seed, n_resamples=options.bootstrap_resamples
    )
    extra["warmup_ms"] = None if warmup_ms is None else round(warmup_ms, 3)
    extra["first_utterance_ms"] = None if first_ms is None else round(first_ms, 3)
    if counts["errors"]:
        notes.append(f"{counts['errors']} utterance(s) failed and are not scored (see items).")
    if counts["timeouts"]:
        notes.append(
            f"{counts['timeouts']} utterance(s) got no final transcript within "
            f"{options.final_timeout:g} s of the end of the audio."
        )
    if (
        len({d.get("metric") for d in extra["datasets"].values()}) > 1
        or len({d.get("language") for d in extra["datasets"].values()}) > 1
    ):
        notes.append(
            "Datasets in several languages: the pooled WER/CER mix languages; read the "
            "per-dataset table."
        )
    if options.mode == "streaming" and options.realtime_factor == 1.0:
        notes.append("Real-time pacing: RTFx is capped near 1 (use --mode batch for throughput).")

    manifest = RunManifest(
        run_id=run_id,
        track=TRACK,
        created=created,
        system=_describe_stt(stt, spec),
        scenario={
            "name": "+".join(d.name for d in datasets),
            "datasets": [d.describe() for d in datasets],
        },
        transport={"type": "in-process", "mode": options.mode},
        options={
            **asdict(options),
            "normalizers": {
                d.name: normalizer_for(options.language or d.language, options.normalizer)
                for d in datasets
            },
        },
        environment=await asyncio.to_thread(collect_environment),
        notes=notes,
    )
    summary = RunSummary(
        run_id=run_id,
        track=TRACK,
        system=label,
        transport=f"in-process ({options.mode})",
        dataset=", ".join(d.id for d in datasets),
        n=counts["scored"],
        metrics=metrics,
        rates=rates,
        counts=counts,
        extra=extra,
        duration_s=round(now() - t_start, 3),
    )
    results = RunResults(manifest, [it.model_dump(mode="json") for it in items], summary)
    results.report = render_asr_report(results)
    if directory is not None:
        write_run(directory, results)
    return results
