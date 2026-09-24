"""Micro-benchmarks of the per-frame hot paths (research note 06, §8.3, T7).

Each benchmark times one *operation* of a hot path and reports microseconds per
operation: one 20 ms audio frame through a resampler, :class:`~voice_agent_next.audio.
AudioFrame` helpers, the energy VAD, the TTS :class:`~voice_agent_next.audio.silence.
SilenceTrimmer` or the G.711 codecs; one LLM token through the sentence segmenter; one
sentence through the pre-TTS text filter; one event through an ``EventEmitter`` or a
``Chan``.

Method: the operation runs in batches calibrated so that one batch takes at least
``min_time`` (like :mod:`timeit`'s autorange); ``repeats`` batches are timed with the
garbage collector disabled and the per-operation cost of each batch is kept. The median
over batches is the headline number (robust to a noisy neighbour on a shared CI runner);
the minimum and the inter-quartile range are reported next to it. Absolute numbers
depend on the CPU: compare them with a baseline from the same kind of machine only.
"""

from __future__ import annotations

import gc
import itertools
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..audio.codecs import alaw_decode, alaw_encode, mulaw_decode, mulaw_encode
from ..audio.frame import AudioFrame
from ..audio.resample import Backend, Resampler
from ..audio.silence import SilenceTrimmer
from ..providers.energy import EnergyVAD
from ..providers.mock import synth_speech
from ..text.filters import tts_clean
from ..text.sentences import SentenceSegmenter
from ..utils.aio import Chan
from ..utils.emitter import EventEmitter

__all__ = [
    "MicroBenchmark",
    "MicroResult",
    "default_micro_benchmarks",
    "run_micro_benchmarks",
]

Op = Callable[[], object]

_FRAME = 0.02  # seconds of audio per operation for the audio benchmarks
_REPLY = (
    "Sure, I can help with that. Your order shipped on Monday and should arrive by "
    "Thursday, Dr. Smith. Would you like me to text you the tracking number, or is there "
    "anything else I can do for you today? Just let me know."
)


@dataclass(frozen=True, slots=True)
class MicroBenchmark:
    """One hot-path operation to time.

    ``make`` builds fresh state and returns the operation (a zero-argument callable);
    it may return ``(op, info)`` to record details such as the resampler backend.
    """

    name: str
    description: str
    make: Callable[[], Op | tuple[Op, dict[str, Any]]]
    unit: str = "frame"
    """What one operation processes: ``frame`` (``frame_ms`` of audio), ``token``..."""
    frame_ms: float | None = _FRAME * 1000
    """Audio covered by one operation; used to express the cost as a share of real time."""


@dataclass(slots=True)
class MicroResult:
    """Per-operation cost of one :class:`MicroBenchmark` (microseconds)."""

    name: str
    description: str
    unit: str
    frame_ms: float | None
    batch: int
    """Operations per timed batch."""
    per_op_us: list[float]
    """Cost per operation of every timed batch (µs)."""
    info: dict[str, Any] = field(default_factory=dict)

    def _q(self, q: float) -> float:
        return float(np.percentile(np.asarray(self.per_op_us, dtype=np.float64), q))

    @property
    def median_us(self) -> float:
        return self._q(50)

    @property
    def min_us(self) -> float:
        return min(self.per_op_us)

    @property
    def iqr_us(self) -> float:
        return self._q(75) - self._q(25)

    @property
    def budget_pct(self) -> float | None:
        """Median cost as a percentage of the audio duration one operation covers."""
        if not self.frame_ms:
            return None
        return 100.0 * self.median_us / (self.frame_ms * 1000.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "micro",
            "name": self.name,
            "description": self.description,
            "unit": self.unit,
            "frame_ms": self.frame_ms,
            "batch": self.batch,
            "median_us": round(self.median_us, 4),
            "min_us": round(self.min_us, 4),
            "iqr_us": round(self.iqr_us, 4),
            "budget_pct": None if self.budget_pct is None else round(self.budget_pct, 5),
            "per_op_us": [round(v, 4) for v in self.per_op_us],
            **({"info": self.info} if self.info else {}),
        }


# --------------------------------------------------------------------------- inputs


def _speech_frames(rate: int, seconds: float, frame: float = _FRAME) -> list[AudioFrame]:
    audio = synth_speech(seconds, rate, amplitude=0.1)
    n = round(frame * rate) * 2
    return [AudioFrame(audio.data[i : i + n], rate) for i in range(0, len(audio.data), n)]


def _cycle(items: Sequence[AudioFrame]) -> Iterator[AudioFrame]:
    return itertools.cycle(items)


# ----------------------------------------------------------------------- benchmarks


def _resampler(src: int, dst: int, backend: Backend) -> Callable[[], tuple[Op, dict[str, Any]]]:
    def make() -> tuple[Op, dict[str, Any]]:
        rs = Resampler(src, dst, backend=backend)
        frames = _cycle(_speech_frames(src, 1.0))
        return (lambda: rs.push(next(frames))), {"backend": rs.backend}

    return make


def _frame_rms() -> Op:
    frame = _speech_frames(16_000, _FRAME)[0]
    return frame.rms


def _frame_concat() -> Op:
    a, b = _speech_frames(16_000, 2 * _FRAME)[:2]
    pair = [a, b]
    return lambda: AudioFrame.concat(pair)


def _frame_slice() -> Op:
    frame = _speech_frames(24_000, _FRAME)[0]
    return lambda: frame.slice(0.005, 0.015)


def _energy_vad() -> Op:
    stream = EnergyVAD(sample_rate=16_000).stream()
    # utterances and pauses, so segments start and end like in a conversation
    frames = _speech_frames(16_000, 0.6) + [AudioFrame.silence(_FRAME, 16_000)] * 25
    it = _cycle(frames)
    return lambda: stream.push_audio(next(it))


def _silence_trimmer() -> Op:
    frames = _speech_frames(24_000, 1.0)  # one 1 s "sentence" per trimmer, like the TTS path
    trimmer = SilenceTrimmer(24_000)
    i = 0

    def op() -> object:
        nonlocal trimmer, i
        out = trimmer.push(frames[i])
        i += 1
        if i == len(frames):
            trimmer.flush()
            trimmer, i = SilenceTrimmer(24_000), 0
        return out

    return op


def _codec(
    fn: Callable[[bytes], bytes], *, source: Callable[[bytes], bytes] | None = None
) -> Callable[[], Op]:
    """``fn`` applied to 20 ms of 8 kHz audio (``source`` encodes it first, for decoders)."""

    def make() -> Op:
        pcm = _speech_frames(8_000, _FRAME)[0].data
        data = source(pcm) if source is not None else pcm
        return lambda: fn(data)

    return make


def _sentence_segmenter() -> Op:
    tokens = [t + " " for t in _REPLY.split()]
    seg = SentenceSegmenter(min_chars=10, first_segment_min_chars=4, first_segment_max_chars=40)
    i = 0

    def op() -> object:
        nonlocal i
        out = seg.push(tokens[i])
        i += 1
        if i == len(tokens):
            seg.flush()
            seg.reset()
            i = 0
        return out

    return op


def _text_filter() -> Op:
    sentence = "Your **order** shipped on Monday, and it should arrive by `Thursday` 🙂."
    return lambda: tts_clean(sentence)


def _event_emit() -> Op:
    emitter = EventEmitter()
    counter = itertools.count()
    emitter.on("metrics", lambda payload: next(counter))
    emitter.on("metrics", lambda payload: None)
    payload = object()
    return lambda: emitter.emit("metrics", payload)


def _chan_roundtrip() -> Op:
    chan: Chan[int] = Chan()

    def op() -> object:
        chan.send_nowait(1)
        return chan.recv_nowait()

    return op


def default_micro_benchmarks() -> list[MicroBenchmark]:
    """The hot paths timed by ``van bench overhead`` (section ``micro``)."""
    return [
        MicroBenchmark(
            "resample_16k_to_24k", "Resampler 16 → 24 kHz (default backend), 20 ms frame",
            _resampler(16_000, 24_000, "auto"),
        ),
        MicroBenchmark(
            "resample_16k_to_24k_numpy", "Resampler 16 → 24 kHz, numpy fallback backend",
            _resampler(16_000, 24_000, "numpy"),
        ),
        MicroBenchmark(
            "resample_48k_to_16k", "Resampler 48 → 16 kHz (default backend), 20 ms frame",
            _resampler(48_000, 16_000, "auto"),
        ),
        MicroBenchmark("frame_rms", "AudioFrame.rms(), 20 ms @ 16 kHz", _frame_rms),
        MicroBenchmark("frame_concat", "AudioFrame.concat() of two 20 ms frames", _frame_concat),
        MicroBenchmark(
            "frame_slice", "AudioFrame.slice() of 10 ms out of 20 ms @ 24 kHz", _frame_slice,
            frame_ms=10.0,
        ),
        MicroBenchmark("energy_vad", "EnergyVAD stream, 20 ms @ 16 kHz", _energy_vad),
        MicroBenchmark(
            "silence_trimmer", "SilenceTrimmer (TTS output path), 20 ms @ 24 kHz",
            _silence_trimmer,
        ),
        MicroBenchmark(
            "g711_ulaw_encode", "G.711 μ-law encode, 20 ms @ 8 kHz", _codec(mulaw_encode)
        ),
        MicroBenchmark(
            "g711_ulaw_decode", "G.711 μ-law decode, 20 ms @ 8 kHz",
            _codec(mulaw_decode, source=mulaw_encode),
        ),
        MicroBenchmark(
            "g711_alaw_encode", "G.711 A-law encode, 20 ms @ 8 kHz", _codec(alaw_encode)
        ),
        MicroBenchmark(
            "g711_alaw_decode", "G.711 A-law decode, 20 ms @ 8 kHz",
            _codec(alaw_decode, source=alaw_encode),
        ),
        MicroBenchmark(
            "sentence_segmenter", "SentenceSegmenter.push() per LLM token (flush per reply)",
            _sentence_segmenter, unit="token", frame_ms=None,
        ),
        MicroBenchmark(
            "tts_text_filter", "tts_clean() per sentence (markdown/emoji removal)",
            _text_filter, unit="sentence", frame_ms=None,
        ),
        MicroBenchmark(
            "event_emit", "EventEmitter.emit() to two handlers", _event_emit, unit="event",
            frame_ms=None,
        ),
        MicroBenchmark(
            "chan_roundtrip", "Chan.send_nowait() + recv_nowait() of one item",
            _chan_roundtrip, unit="item", frame_ms=None,
        ),
    ]  # fmt: skip


# -------------------------------------------------------------------------- running


def _time_batch(op: Op, number: int) -> float:
    """Seconds to run ``op`` ``number`` times (garbage collector disabled)."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        t0 = time.perf_counter()
        for _ in range(number):
            op()
        return time.perf_counter() - t0
    finally:
        if enabled:
            gc.enable()


def _calibrate(op: Op, min_time: float) -> int:
    """Smallest batch size of the 1-2-5 series whose batch takes ``>= min_time``."""
    for base in itertools.count():
        for factor in (1, 2, 5):
            number = factor * 10**base
            if _time_batch(op, number) >= min_time or number >= 10_000_000:
                return number
    raise AssertionError("unreachable")


def run_micro_benchmarks(
    benchmarks: Iterable[MicroBenchmark] | None = None,
    *,
    repeats: int = 15,
    min_time: float = 0.02,
    only: Iterable[str] | None = None,
) -> list[MicroResult]:
    """Time every benchmark (see the module docstring); blocking, CPU-bound."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if min_time <= 0:
        raise ValueError("min_time must be > 0")
    selected = list(benchmarks if benchmarks is not None else default_micro_benchmarks())
    if only is not None:
        names = set(only)
        unknown = names - {b.name for b in selected}
        if unknown:
            raise ValueError(f"unknown micro-benchmark(s): {', '.join(sorted(unknown))}")
        selected = [b for b in selected if b.name in names]
    results: list[MicroResult] = []
    for bench in selected:
        made = bench.make()
        op, info = made if isinstance(made, tuple) else (made, {})
        number = _calibrate(op, min_time)  # also warms caches and lazy state up
        per_op = [_time_batch(op, number) / number * 1e6 for _ in range(repeats)]
        results.append(
            MicroResult(bench.name, bench.description, bench.unit, bench.frame_ms, number,
                        per_op, dict(info))
        )  # fmt: skip
    return results
