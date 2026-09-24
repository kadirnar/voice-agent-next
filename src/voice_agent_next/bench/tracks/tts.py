"""T3 TTS track: time to first audio, real-time factor, stalls, round-trip WER and MOS.

Research note 06, §5 and §8.3 (T3). Every text of a pinned set is synthesized in one or
both modes:

* ``batch`` — ``tts.synthesize(text)``: the whole text is known up front;
* ``streaming`` — the voice-agent case: ``tts.stream()`` receives the text word by word
  at an LLM-like pace (``words_per_second``; sentence segmentation, prefetch and
  silence trimming happen exactly as in a live cascade).

The clock starts at the request (batch) or at the first pushed word (streaming). Every
audio chunk is time-stamped when the harness receives it, and playback is simulated as a
real-time player that starts with the first chunk and never waits otherwise:

* ``ttfb_ms`` — first chunk received (what providers usually report);
* ``ttfa_ms`` — first *audible* sample played: the playout time of the first speech
  onset (reference VAD: 10 ms frames ≥ -40 dBFS, ≥ 100 ms of speech, refined to the
  sample), so leading silence counts;
* ``leading_silence_ms`` / ``trailing_silence_ms`` — silence before the first and after
  the last frame at the threshold level;
* ``rtf`` — synthesis wall time ÷ audio duration (< 1 is faster than real time);
* ``underruns`` / ``stall_ms`` — chunks that arrived after the player had run dry (by
  more than ``underrun_threshold``), and the silence that inserted; ``underruns_per_min``
  is per minute of audio;
* ``chunk_gap_max_ms`` / ``chunk_jitter_ms`` — largest inter-arrival gap and the
  standard deviation of the gaps (a stall detector that does not depend on playout).

With an STT (``stt=``) every clip is transcribed after the timed phase and scored
against its text: ``rt_wer`` / ``rt_cer`` (corpus rates Σedits / Σreference length after
normalization, :mod:`voice_agent_next.bench.roundtrip`) and ``hardtext_acc`` (share of
entities — numbers, dates, amounts, e-mails, URLs, abbreviations — read correctly). An
optional MOS predictor (:mod:`voice_agent_next.bench.mos`) adds ``dnsmos_sig/bak/ovrl``.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from ...audio.frame import AudioFrame
from ...audio.wav import write_wav
from ...config import ComponentSpec
from ...registry import create
from ...stt import STT
from ...tts import TTS
from ...utils.aio import cancel_and_wait
from ...utils.clock import now
from ..caller import _sleep_until
from ..environment import collect_environment
from ..mos import MOSPredictor, make_mos_predictor
from ..onset import OnsetDetector, RMSReferenceVAD, frame_levels_db
from ..report import ReportSpec, fmt, markdown_table, render_report
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
from ..roundtrip import resolve_normalizer, score_round_trip
from ..system import redact
from ..text_norm import NORMALIZERS
from .latency import _reserve_directory

__all__ = [
    "MODES",
    "TRACK",
    "Capture",
    "Playout",
    "TTSItem",
    "TTSOptions",
    "TTSText",
    "TextSet",
    "chunk_gaps",
    "load_texts",
    "measure_capture",
    "render_tts_report",
    "run_tts_benchmark",
    "silence_bounds",
    "simulate_playout",
    "summarize_tts",
]

TRACK = "tts"
MODES: tuple[str, ...] = ("batch", "streaming")
BUILTIN_TEXTS = {"smoke": "tts_smoke.json"}
_WARMUP_TEXT = "Hello, this sentence warms the voice up."


# ----------------------------------------------------------------------------- text sets


@dataclass(frozen=True)
class TTSText:
    """One text to synthesize; ``entities`` lists accepted spoken forms per entity."""

    id: str
    text: str
    category: str = ""
    entities: tuple[tuple[str, ...], ...] = ()

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "text": self.text, "category": self.category}
        if self.entities:
            out["entities"] = [list(e) for e in self.entities]
        return out


@dataclass(frozen=True)
class TextSet:
    name: str
    texts: tuple[TTSText, ...]
    version: int = 1
    language: str = "en"
    source: str = ""
    license: str | None = None

    def sha256(self) -> str:
        """Hash of the texts and language (what the results depend on)."""
        payload = json.dumps(
            {"language": self.language, "texts": [t.to_json() for t in self.texts]},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def dataset_id(self) -> str:
        return f"{self.name}@sha256:{self.sha256()[:12]}"

    def limited(self, limit: int | None) -> TextSet:
        if limit is None or limit >= len(self.texts):
            return self
        return TextSet(
            self.name, self.texts[:limit], self.version, self.language, self.source, self.license
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "language": self.language,
            "source": self.source,
            "license": self.license,
            "sha256": self.sha256(),
            "count": len(self.texts),
            "texts": [t.to_json() for t in self.texts],
        }


def _parse_text(raw: Any, index: int) -> TTSText:
    if isinstance(raw, str):
        return TTSText(id=f"t{index + 1:03d}", text=raw.strip())
    if not isinstance(raw, Mapping) or not str(raw.get("text", "")).strip():
        raise ValueError(f"text #{index + 1}: expected a string or a mapping with 'text'")
    entities = tuple(
        tuple(str(a) for a in (e if isinstance(e, (list, tuple)) else [e]))
        for e in raw.get("entities") or ()
    )
    return TTSText(
        id=str(raw.get("id") or f"t{index + 1:03d}"),
        text=str(raw["text"]).strip(),
        category=str(raw.get("category") or ""),
        entities=entities,
    )


def _text_set(data: Any, *, name: str, source: str) -> TextSet:
    if isinstance(data, list):
        data = {"texts": data}
    if not isinstance(data, Mapping) or not isinstance(data.get("texts"), list):
        raise ValueError(f"{source}: expected a list of texts or a mapping with 'texts'")
    texts = tuple(_parse_text(raw, i) for i, raw in enumerate(data["texts"]))
    if not texts:
        raise ValueError(f"{source}: no texts")
    ids = [t.id for t in texts]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{source}: duplicate text ids")
    return TextSet(
        name=str(data.get("name") or name),
        texts=texts,
        version=int(data.get("version") or 1),
        language=str(data.get("language") or "en"),
        source=source,
        license=data.get("license"),
    )


def load_texts(spec: str | os.PathLike[str] = "smoke") -> TextSet:
    """A built-in set (``smoke``) or a file: ``.txt`` (one text per line, ``#`` comments),
    ``.json`` / ``.yaml`` (a list of texts, or ``{name, language, texts: [...]}`` where a
    text is a string or ``{id, text, category, entities}``)."""
    key = str(spec)
    if key in BUILTIN_TEXTS:
        raw = resources.files("voice_agent_next.bench").joinpath("data", BUILTIN_TEXTS[key])
        return _text_set(json.loads(raw.read_text(encoding="utf-8")), name=key, source=key)
    path = Path(spec)
    if not path.is_file():
        raise ValueError(f"unknown text set {key!r}: use {', '.join(BUILTIN_TEXTS)} or a file")
    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".txt":
        lines = [ln.strip() for ln in content.splitlines()]
        data: Any = [ln for ln in lines if ln and not ln.startswith("#")]
    elif path.suffix.lower() == ".json":
        data = json.loads(content)
    else:
        data = yaml.safe_load(content)
    return _text_set(data, name=path.stem, source=str(path))


# ----------------------------------------------------------------------------- options


@dataclass
class TTSOptions:
    """Knobs of a T3 run (recorded in the manifest)."""

    modes: tuple[str, ...] = MODES
    repeats: int = 1
    """Times every text is synthesized per mode."""
    warmup_requests: int = 1
    """Untimed requests per mode after ``tts.warmup()`` (the first one is the cold start)."""
    words_per_second: float = 15.0
    """Streaming mode: LLM-like text pace (~20 tokens/s); 0 pushes the text at once."""
    underrun_threshold: float = 0.010
    """A chunk later than the end of the audio already played by more than this (s) is an
    underrun."""
    silence_threshold_db: float = -40.0
    timeout: float = 120.0
    """Seconds per request before it is abandoned (recorded as an error)."""
    normalizer: str = "auto"
    language: str | None = None
    """STT language for the round trip (default: the text set's language)."""
    save_audio: bool = True
    seed: int = 0
    bootstrap_resamples: int = 2000

    def validate(self) -> None:
        if not self.modes or any(m not in MODES for m in self.modes):
            raise ValueError(f"modes must be a non-empty subset of {MODES}")
        if self.repeats < 1:
            raise ValueError("repeats must be >= 1")
        if self.warmup_requests < 0:
            raise ValueError("warmup_requests must be >= 0")
        if self.words_per_second < 0:
            raise ValueError("words_per_second must be >= 0")
        if self.timeout <= 0:
            raise ValueError("timeout must be > 0")
        if self.normalizer not in NORMALIZERS:
            raise ValueError(
                f"unknown normalizer {self.normalizer!r}; use one of {', '.join(NORMALIZERS)}"
            )


# ------------------------------------------------------------------------- metrics math


@dataclass
class Capture:
    """Audio of one request as received: ``arrivals`` are ``(seconds since the request,
    chunk duration)`` pairs, in order."""

    frames: list[AudioFrame] = field(default_factory=list)
    arrivals: list[tuple[float, float]] = field(default_factory=list)
    end: float | None = None
    """Seconds from the request until the stream finished."""
    error: str | None = None

    def add(self, t: float, frame: AudioFrame) -> None:
        self.frames.append(frame)
        self.arrivals.append((t, frame.duration))

    def audio(self, sample_rate: int) -> AudioFrame:
        if not self.frames:
            return AudioFrame.empty(sample_rate)
        return AudioFrame.concat(self.frames).to_mono()


@dataclass(frozen=True)
class Playout:
    """A real-time player fed with the chunks as they arrive."""

    starts: tuple[float, ...]
    """When every chunk starts playing (s since the request)."""
    underruns: int
    stall: float
    """Total silence inserted by underruns (s)."""

    def time_of(self, arrivals: Sequence[tuple[float, float]], offset: float) -> float | None:
        """Wall time (s since the request) at which audio position ``offset`` plays."""
        pos = 0.0
        for start, (_, duration) in zip(self.starts, arrivals, strict=True):
            if offset < pos + duration - 1e-12:
                return start + max(0.0, offset - pos)
            pos += duration
        return None


def simulate_playout(
    arrivals: Sequence[tuple[float, float]], *, underrun_threshold: float = 0.010
) -> Playout:
    """Play chunks back to back from the first arrival; a chunk that arrives after the
    player ran dry starts late (an underrun when later than ``underrun_threshold``)."""
    starts: list[float] = []
    underruns = 0
    stall = 0.0
    cursor: float | None = None
    for t, duration in arrivals:
        if cursor is None:
            start = t
        else:
            late = t - cursor
            if late > 0:
                stall += late
                if late > underrun_threshold:
                    underruns += 1
            start = max(t, cursor)
        starts.append(start)
        cursor = start + duration
    return Playout(tuple(starts), underruns, stall)


def chunk_gaps(arrivals: Sequence[tuple[float, float]]) -> list[float]:
    """Inter-arrival times of consecutive chunks (s)."""
    return [b[0] - a[0] for a, b in itertools.pairwise(arrivals)]


def silence_bounds(
    audio: AudioFrame, *, threshold_db: float = -40.0
) -> tuple[float | None, float | None]:
    """``(leading, trailing)`` silence in seconds; ``(None, None)`` without speech.

    Leading: the reference-VAD speech onset (≥ 100 ms of 10 ms frames at the threshold,
    refined to the sample). Trailing: after the last 10 ms frame at the threshold."""
    if not audio:
        return None, None
    detector = OnsetDetector(RMSReferenceVAD(threshold_db))
    onsets = detector.onsets(audio)
    if not onsets:
        return None, None
    levels = frame_levels_db(audio, detector.frame_duration)
    loud = np.flatnonzero(levels >= threshold_db)
    end = min(audio.duration, (int(loud[-1]) + 1) * detector.frame_duration)
    return onsets[0], max(0.0, audio.duration - end)


def _ms(seconds: float | None) -> float | None:
    return None if seconds is None else seconds * 1000.0


def measure_capture(
    capture: Capture,
    sample_rate: int,
    *,
    underrun_threshold: float = 0.010,
    silence_threshold_db: float = -40.0,
) -> dict[str, Any]:
    """Timing and silence metrics of one request (see the module docstring)."""
    audio = capture.audio(sample_rate)
    duration = audio.duration
    lead, trail = silence_bounds(audio, threshold_db=silence_threshold_db)
    play = simulate_playout(capture.arrivals, underrun_threshold=underrun_threshold)
    ttfa = play.time_of(capture.arrivals, lead) if lead is not None else None
    gaps = chunk_gaps(capture.arrivals)
    return {
        "ttfb_ms": _ms(capture.arrivals[0][0]) if capture.arrivals else None,
        "ttfa_ms": _ms(ttfa),
        "leading_silence_ms": _ms(lead),
        "trailing_silence_ms": _ms(trail),
        "audio_ms": duration * 1000.0,
        "synth_ms": _ms(capture.end),
        "rtf": capture.end / duration if capture.end is not None and duration > 0 else None,
        "chunks": len(capture.arrivals),
        "chunk_gap_max_ms": _ms(max(gaps)) if gaps else None,
        "chunk_jitter_ms": _ms(float(np.std(gaps, ddof=1))) if len(gaps) > 1 else None,
        "underruns": play.underruns,
        "stall_ms": play.stall * 1000.0,
        "no_speech": lead is None and duration > 0,
    }


# ----------------------------------------------------------------------------- requests


async def _collect(stream: Any, capture: Capture, t0: float) -> None:
    async for chunk in stream:
        if chunk.frame:
            capture.add(now() - t0, chunk.frame)


async def synthesize_batch(tts: TTS, text: str, *, timeout: float) -> Capture:
    """One ``tts.synthesize(text)`` request, time-stamped."""
    capture = Capture()
    t0 = now()
    stream = tts.synthesize(text)
    try:
        async with asyncio.timeout(timeout):
            await _collect(stream, capture, t0)
        capture.end = now() - t0
    except TimeoutError:
        capture.error = f"timeout after {timeout:g} s"
    except Exception as exc:
        capture.error = repr(exc)
    finally:
        await stream.aclose()
    return capture


def _paced_tokens(text: str) -> list[str]:
    return re.findall(r"\S+\s*", text)


async def synthesize_streaming(
    tts: TTS, text: str, *, words_per_second: float, timeout: float
) -> Capture:
    """``tts.stream()`` fed word by word at ``words_per_second`` (0: all at once); the
    clock starts at the first word."""
    capture = Capture()
    stream = tts.stream()
    tokens = _paced_tokens(text)
    t0 = now()

    async def feed() -> None:
        for i, token in enumerate(tokens):
            if words_per_second > 0 and i:
                await _sleep_until(t0 + i / words_per_second)
            stream.push_text(token)
        stream.end_input()

    feeder = asyncio.create_task(feed(), name="tts-bench-feed")
    try:
        async with asyncio.timeout(timeout):
            await _collect(stream, capture, t0)
        capture.end = now() - t0
        if feeder.done() and feeder.exception() is not None:
            capture.error = repr(feeder.exception())
    except TimeoutError:
        capture.error = f"timeout after {timeout:g} s"
    except Exception as exc:
        capture.error = repr(exc)
    finally:
        await cancel_and_wait(feeder)
        await stream.aclose()
    return capture


# --------------------------------------------------------------------------------- items


class TTSItem(BaseModel):
    """One synthesized text in one mode (``items.jsonl``)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text_id: str
    category: str = ""
    mode: str
    repeat: int = 0
    text: str
    characters: int
    ttfb_ms: float | None = None
    ttfa_ms: float | None = None
    leading_silence_ms: float | None = None
    trailing_silence_ms: float | None = None
    audio_ms: float | None = None
    synth_ms: float | None = None
    rtf: float | None = None
    chunks: int = 0
    chunk_gap_max_ms: float | None = None
    chunk_jitter_ms: float | None = None
    underruns: int = 0
    stall_ms: float | None = None
    no_speech: bool = False
    transcript: str | None = None
    reference_norm: str | None = None
    transcript_norm: str | None = None
    word_errors: int | None = None
    ref_words: int | None = None
    char_errors: int | None = None
    ref_chars: int | None = None
    rt_wer: float | None = None
    rt_cer: float | None = None
    rt_metric: str | None = None
    """Headline round-trip metric: ``wer``, or ``cer`` for languages without spaces."""
    entities_total: int = 0
    entities_ok: int | None = None
    entities_missed: list[str] = Field(default_factory=list)
    mos: dict[str, float] = Field(default_factory=dict)
    audio_file: str | None = None
    error: str | None = None


def score_transcript(
    item: TTSItem, text: TTSText, *, language: str | None, normalize: Callable[[str], str]
) -> None:
    """Fill the round-trip fields of ``item`` from ``item.transcript``."""
    score = score_round_trip(
        text.text, item.transcript or "", language=language, normalize=normalize,
        entities=text.entities,
    )  # fmt: skip
    item.reference_norm, item.transcript_norm = score.reference_norm, score.transcript_norm
    item.word_errors, item.ref_words = score.words.errors, score.words.ref_len
    item.char_errors, item.ref_chars = score.chars.errors, score.chars.ref_len
    item.rt_wer, item.rt_cer, item.rt_metric = score.words.rate, score.chars.rate, score.headline
    if text.entities:
        item.entities_ok = len(text.entities) - len(score.entities_missed)
        item.entities_missed = list(score.entities_missed)


# ------------------------------------------------------------------------------- summary

_TIME_METRICS = (
    "ttfa_ms",
    "ttfb_ms",
    "leading_silence_ms",
    "trailing_silence_ms",
    "chunk_gap_max_ms",
    "chunk_jitter_ms",
    "stall_ms",
)
_RATIO_METRICS = ("rtf",)


def _corpus(pairs: Iterable[tuple[int | None, int | None]]) -> float | None:
    """Σ errors / Σ reference length over items (the corpus rate of :mod:`..wer`)."""
    errors = ref = 0
    for e, n in pairs:
        errors += e or 0
        ref += n or 0
    return errors / ref if ref else None


def summarize_tts(
    items: Sequence[TTSItem], modes: Sequence[str], *, seed: int = 0, n_resamples: int = 2000
) -> tuple[dict[str, Distribution], dict[str, float | None], dict[str, int], dict[str, Any]]:
    """Per-mode distributions (``<mode>.<metric>``), rates, counts and extra aggregates."""
    metrics: dict[str, Distribution] = {}
    rates: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    per_mode: dict[str, Any] = {}

    def dist(values: Iterable[float | None]) -> Distribution:
        return Distribution.of(values, seed=seed, n_resamples=n_resamples, digits=4)

    for mode in modes:
        rows = [it for it in items if it.mode == mode]
        ok = [it for it in rows if it.error is None]
        for key in (*_TIME_METRICS, *_RATIO_METRICS):
            metrics[f"{mode}.{key}"] = dist(getattr(it, key) for it in ok)
        mos_keys = sorted({k for it in ok for k in it.mos})
        for key in mos_keys:
            metrics[f"{mode}.{key}"] = dist(it.mos.get(key) for it in ok)
        scored = [it for it in ok if it.word_errors is not None]
        wer = _corpus((it.word_errors, it.ref_words) for it in scored)
        cer = _corpus((it.char_errors, it.ref_chars) for it in scored)
        ent_total = sum(it.entities_total for it in scored)
        ent_ok = sum(it.entities_ok or 0 for it in scored)
        audio_s = sum((it.audio_ms or 0.0) for it in ok) / 1000.0
        synth_s = sum((it.synth_ms or 0.0) for it in ok) / 1000.0
        underruns = sum(it.underruns for it in ok)
        rates[f"{mode}.rt_wer"] = wer if scored else None
        rates[f"{mode}.rt_cer"] = cer if scored else None
        rates[f"{mode}.hardtext_acc"] = ent_ok / ent_total if ent_total else None
        rates[f"{mode}.perfect_rate"] = (
            sum(1 for it in scored if it.word_errors == 0) / len(scored) if scored else None
        )
        rates[f"{mode}.underrun_rate"] = (
            sum(1 for it in ok if it.underruns) / len(ok) if ok else None
        )
        rates[f"{mode}.error_rate"] = (len(rows) - len(ok)) / len(rows) if rows else None
        counts[f"{mode}.items"] = len(rows)
        counts[f"{mode}.errors"] = len(rows) - len(ok)
        counts[f"{mode}.no_speech"] = sum(1 for it in ok if it.no_speech)
        counts[f"{mode}.underruns"] = underruns
        counts[f"{mode}.entities"] = ent_total
        by_category: dict[str, dict[str, Any]] = {}
        for cat in sorted({it.category for it in scored}):
            cat_items = [it for it in scored if it.category == cat]
            by_category[cat or "-"] = {
                "n": len(cat_items),
                "rt_wer": _corpus((it.word_errors, it.ref_words) for it in cat_items),
            }
        per_mode[mode] = {
            "audio_s": round(audio_s, 3),
            "synth_s": round(synth_s, 3),
            "rtf_total": round(synth_s / audio_s, 4) if audio_s > 0 else None,
            "underruns_per_min": round(underruns / (audio_s / 60.0), 3) if audio_s > 0 else None,
            "wer_by_category": by_category,
            "mos_mean": {k: metrics[f"{mode}.{k}"].mean for k in mos_keys},
        }
    return metrics, rates, counts, {"modes": per_mode}


# -------------------------------------------------------------------------------- report

_METRIC_LABELS = {
    "ttfa_ms": "TTFA (first audible sample)",
    "ttfb_ms": "TTFB (first chunk)",
    "leading_silence_ms": "leading silence",
    "trailing_silence_ms": "trailing silence",
    "chunk_gap_max_ms": "largest chunk gap",
    "chunk_jitter_ms": "chunk-gap jitter (std)",
    "stall_ms": "stall (underrun silence)",
}

_METHOD = """\
* **batch**: `tts.synthesize(text)`; **streaming**: `tts.stream()` fed word by word at
  {wps} words/s (LLM-like pacing; sentences are segmented as in a live cascade). The
  clock starts at the request / first word.
* `ttfa_ms`: playout time of the first speech onset (reference VAD: 10 ms frames ≥
  {thr:g} dBFS, ≥ 100 ms of speech, refined to the sample) for a real-time player that
  starts with the first chunk, so leading silence counts. `ttfb_ms`: first chunk received.
* `rtf` = synthesis wall time ÷ audio duration (streaming: includes waiting for the paced
  text, so it is bounded below by the text pace). An underrun is a chunk arriving more than
  {under:g} ms after the player ran dry; `stall_ms` is the silence it inserted.
* Round trip: every clip is transcribed after the timed phase by `{stt}`; `rt_wer` /
  `rt_cer` are corpus rates (Σ edits ÷ Σ reference length) after the `{norm}`
  normalizer. `hardtext_acc`: share of entities (numbers, dates, amounts, e-mails, URLs,
  abbreviations) found in the transcript in one of their accepted spoken forms.
* MOS predictor: {mos}. MOS predictors are regression signals, not rankings.
* {repeats} repetition(s) per text and mode after {warmup} untimed warm-up request(s) per
  mode. Percentiles are linear-interpolated; `[..]` is the 95% bootstrap CI of the median.
"""


def _pct(value: Any) -> str:
    return "–" if value is None else f"{100 * float(value):.1f}%"


def tts_report_spec(results: RunResults) -> ReportSpec:
    s, m = results.summary, results.manifest
    opts = m.options
    modes = [mode for mode in MODES if f"{mode}.ttfa_ms" in s.metrics]
    labels = {
        f"{mode}.{key}": f"{mode} · {label} (ms)"
        for mode in modes
        for key, label in _METRIC_LABELS.items()
    }
    quality_rows = []
    for mode in modes:
        info = s.extra.get("modes", {}).get(mode, {})
        rtf = s.metrics.get(f"{mode}.rtf")
        mos = info.get("mos_mean") or {}
        quality_rows.append(
            [
                mode,
                fmt(rtf.p50 if rtf else None, 3),
                fmt(rtf.p95 if rtf else None, 3),
                fmt(info.get("rtf_total"), 3),
                fmt(info.get("underruns_per_min"), 2),
                _pct(s.rates.get(f"{mode}.rt_wer")),
                _pct(s.rates.get(f"{mode}.rt_cer")),
                _pct(s.rates.get(f"{mode}.hardtext_acc")),
                " / ".join(fmt(mos.get(k), 2) for k in ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl"))
                if mos
                else "–",
            ]
        )
    quality = markdown_table(
        ["mode", "RTF p50", "RTF p95", "RTF total", "underruns/min", "rt WER", "rt CER",
         "hard text", "DNSMOS sig/bak/ovrl"],
        quality_rows,
        ["l"] + ["r"] * 8,
    )  # fmt: skip
    item_rows = [
        [
            str(it.get("text_id")),
            str(it.get("mode")),
            fmt(it.get("ttfa_ms")),
            fmt(it.get("ttfb_ms")),
            fmt(it.get("leading_silence_ms")),
            fmt(it.get("rtf"), 3),
            fmt(it.get("underruns")),
            _pct(it.get("rt_wer")),
            ", ".join(it.get("entities_missed") or []) or "–",
            it.get("error") or (it.get("transcript") or "–"),
        ]
        for it in results.items[:200]
    ]
    per_item = markdown_table(
        ["text", "mode", "TTFA", "TTFB", "lead sil.", "RTF", "underruns", "rt WER",
         "missed entities", "transcript / error"],
        item_rows,
        ["l", "l", "r", "r", "r", "r", "r", "r", "l", "l"],
    )  # fmt: skip
    method = _METHOD.format(
        wps=fmt(opts.get("words_per_second"), 1),
        thr=opts.get("silence_threshold_db", -40.0),
        under=1000 * float(opts.get("underrun_threshold", 0.01)),
        stt=(m.system.get("stt") or {}).get("label") or "no STT (round trip skipped)",
        norm=opts.get("normalizer_resolved", opts.get("normalizer", "auto")),
        mos=(opts.get("mos") or {}).get("name") or "none",
        repeats=opts.get("repeats", 1),
        warmup=opts.get("warmup_requests", 1),
    )
    return ReportSpec(
        title=f"TTS (T3) · {s.system}",
        metric_labels=labels,
        rate_labels={},
        sections=[
            ("Speed and quality", quality),
            ("Per text", per_item),
            ("Method", method),
        ],
    )


def render_tts_report(results: RunResults) -> str:
    spec = tts_report_spec(results)
    # rates and counts are summarized in "Speed and quality"; keep the generic tables short
    trimmed = RunResults(
        results.manifest,
        results.items,
        results.summary.model_copy(update={"rates": {}, "counts": {}}),
        None,
        results.directory,
    )
    return render_report(trimmed, spec)


# --------------------------------------------------------------------------------- entry


def _component_info(component: TTS | STT, spec: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "spec": redact(spec),
        "provider": component.provider,
        "model": component.model,
        "class": type(component).__name__,
    }
    if isinstance(component, TTS):
        info.update(
            voice=component.voice,
            sample_rate=component.sample_rate,
            streaming=component.capabilities.streaming,
            trim_silence=component.trim_silence,
        )
    else:
        info.update(language=component.language, streaming=component.capabilities.streaming)
    info["label"] = f"{component.provider}/{component.model}"
    return info


async def run_tts_benchmark(
    tts: TTS | ComponentSpec,
    texts: TextSet | str | None = None,
    options: TTSOptions | None = None,
    *,
    stt: STT | ComponentSpec | None = None,
    mos: MOSPredictor | str | None = None,
    out_dir: str | os.PathLike[str] | None = None,
    run_id: str | None = None,
    label: str | None = None,
    on_item: Callable[[TTSItem], None] | None = None,
) -> RunResults:
    """Run the T3 TTS track and (if ``out_dir``) write ``<out_dir>/<run_id>/``.

    Args:
        tts: the TTS under test (instance or registry spec, e.g. ``"kokoro"``).
        texts: a :class:`TextSet` or a name/path for :func:`load_texts` (default ``smoke``).
        options: modes, repetitions, pacing, thresholds...
        stt: ASR for the round trip (instance or spec); ``None`` skips WER.
        mos: MOS predictor (``"dnsmos"`` or an instance); skipped with a note when it
            cannot be loaded.
        out_dir: parent of the run directory (``None``: nothing is written).
        run_id: defaults to ``<UTC time>-tts-<label>``.
        label: system label (default ``<provider>/<model>``).
        on_item: progress callback after every timed request.
    """
    options = options or TTSOptions()
    options.validate()
    text_set = texts if isinstance(texts, TextSet) else load_texts(texts or "smoke")
    tts_spec = tts if not isinstance(tts, TTS) else None
    tts_obj: TTS = create("tts", tts) if not isinstance(tts, TTS) else tts
    stt_obj: STT | None = None
    if stt is not None:
        stt_obj = create("stt", stt) if not isinstance(stt, STT) else stt
    system_label = label or f"{tts_obj.provider}/{tts_obj.model}"
    final_id = run_id or new_run_id(TRACK, system_label)
    directory: Path | None = None
    if out_dir is not None:
        directory = _reserve_directory(Path(out_dir), final_id, unique=run_id is None)
        final_id = directory.name
    try:
        return await _run_tts(
            tts_obj, tts_spec, text_set, options, stt_obj,
            stt if not isinstance(stt, STT) else None, make_mos_predictor(mos),
            directory=directory, run_id=final_id, label=system_label, on_item=on_item,
        )  # fmt: skip
    except BaseException:
        if directory is not None and not any(directory.iterdir()):
            directory.rmdir()
        raise
    finally:
        if not isinstance(tts, TTS):
            await tts_obj.aclose()
        if stt_obj is not None and not isinstance(stt, STT):
            await stt_obj.aclose()


async def _request(tts: TTS, text: str, mode: str, options: TTSOptions) -> Capture:
    if mode == "batch":
        return await synthesize_batch(tts, text, timeout=options.timeout)
    return await synthesize_streaming(
        tts, text, words_per_second=options.words_per_second, timeout=options.timeout
    )


async def _run_tts(
    tts: TTS,
    tts_spec: Any,
    text_set: TextSet,
    options: TTSOptions,
    stt: STT | None,
    stt_spec: Any,
    mos: MOSPredictor | None,
    *,
    directory: Path | None,
    run_id: str,
    label: str,
    on_item: Callable[[TTSItem], None] | None,
) -> RunResults:
    created = utc_timestamp()
    t_start = now()
    notes: list[str] = []
    language = options.language or text_set.language
    norm_name, normalize = resolve_normalizer(language, options.normalizer)

    # load everything up front: a broken STT should fail before minutes of synthesis
    t0 = now()
    await tts.warmup()
    warmup_ms = (now() - t0) * 1000.0
    if stt is not None:
        await stt.warmup()
    mos_info: dict[str, Any] | None = None
    if mos is not None:
        try:
            await mos.load()
            mos_info = mos.describe()
        except Exception as exc:
            notes.append(f"MOS predictor {mos.name!r} unavailable, skipped: {exc}")
            mos = None

    cold: dict[str, Any] = {}
    for mode in options.modes:
        for i in range(options.warmup_requests):
            cap = await _request(tts, _WARMUP_TEXT, mode, options)
            if i == 0:
                first = measure_capture(cap, tts.sample_rate)
                cold[mode] = {
                    "ttfa_ms": first["ttfa_ms"],
                    "ttfb_ms": first["ttfb_ms"],
                    "error": cap.error,
                }

    items: list[TTSItem] = []
    audio: list[AudioFrame] = []
    for mode in options.modes:
        for repeat in range(options.repeats):
            for text in text_set.texts:
                cap = await _request(tts, text.text, mode, options)
                metrics = measure_capture(
                    cap,
                    tts.sample_rate,
                    underrun_threshold=options.underrun_threshold,
                    silence_threshold_db=options.silence_threshold_db,
                )
                suffix = f"-r{repeat}" if options.repeats > 1 else ""
                item = TTSItem(
                    id=f"{mode}/{text.id}{suffix}",
                    text_id=text.id,
                    category=text.category,
                    mode=mode,
                    repeat=repeat,
                    text=text.text,
                    characters=len(text.text),
                    entities_total=len(text.entities),
                    error=cap.error or (None if cap.frames else "no audio"),
                    **metrics,
                )
                items.append(item)
                audio.append(cap.audio(tts.sample_rate))
                if on_item is not None:
                    on_item(item)

    # untimed phase: round trip and MOS
    texts_by_id = {t.id: t for t in text_set.texts}
    for item, clip in zip(items, audio, strict=True):
        if not clip:
            continue
        if stt is not None:
            try:
                transcript = await stt.transcribe(clip, language=language)
                item.transcript = transcript.text.strip()
                score_transcript(
                    item, texts_by_id[item.text_id], language=language, normalize=normalize
                )
            except Exception as exc:  # the clip's timing stays valid; it is just not scored
                notes.append(f"Round-trip STT failed for {item.id}: {exc!r}")
        if mos is not None:
            try:
                item.mos = {k: round(v, 4) for k, v in (await mos.score(clip)).items()}
            except Exception as exc:
                notes.append(f"MOS scoring failed for {item.id}: {exc!r}")

    if stt is None:
        notes.append("No --stt given: round-trip WER/CER and hard-text accuracy were skipped.")
    no_speech = sum(1 for it in items if it.no_speech)
    if no_speech:
        notes.append(
            f"{no_speech} clip(s) had no speech above {options.silence_threshold_db:g} dBFS: "
            "TTFA is undefined for them (quiet voice? lower the silence threshold)."
        )
    errors = [it for it in items if it.error]
    if errors:
        notes.append(f"{len(errors)} request(s) failed; see `error` in items.jsonl.")

    metrics, rates, counts, extra = summarize_tts(
        items, options.modes, seed=options.seed, n_resamples=options.bootstrap_resamples
    )
    extra.update(
        tts_warmup_ms=round(warmup_ms, 3),
        cold_start=cold,
        normalizer=norm_name,
        stt=None if stt is None else f"{stt.provider}/{stt.model}",
        mos=None if mos is None else mos.name,
    )
    system: dict[str, Any] = {"label": label, "kind": "tts", "tts": _component_info(tts, tts_spec)}
    if stt is not None:
        system["stt"] = _component_info(stt, stt_spec)
    manifest = RunManifest(
        run_id=run_id,
        track=TRACK,
        created=created,
        system=system,
        scenario=text_set.describe(),
        transport={"type": "none", "note": "in-process component calls"},
        options={
            **asdict(options),
            "normalizer_resolved": norm_name,
            "stt_language": language,
            "mos": mos_info,
        },
        environment=await asyncio.to_thread(collect_environment),
        notes=notes,
    )
    headline = f"{options.modes[-1]}.ttfa_ms"
    summary = RunSummary(
        run_id=run_id,
        track=TRACK,
        system=label,
        transport="none",
        dataset=text_set.dataset_id(),
        n=metrics[headline].n,
        metrics=metrics,
        rates=rates,
        counts=counts,
        extra=extra,
        duration_s=round(now() - t_start, 3),
    )
    if directory is not None and options.save_audio:
        await asyncio.to_thread(_write_audio, directory, items, audio)
    results = RunResults(manifest, [it.model_dump(mode="json") for it in items], summary)
    results.report = render_tts_report(results)
    if directory is not None:
        write_run(directory, results)
    return results


def _write_audio(directory: Path, items: Sequence[TTSItem], audio: Sequence[AudioFrame]) -> None:
    for item, clip in zip(items, audio, strict=True):
        if not clip:
            continue
        rel = Path(ARTIFACTS_DIR) / item.mode / f"{item.id.split('/', 1)[1]}.wav"
        (directory / rel).parent.mkdir(parents=True, exist_ok=True)
        write_wav(directory / rel, clip)
        item.audio_file = rel.as_posix()
