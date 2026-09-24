"""Speech onset detection with a *reference* VAD (research note 06, §8.3).

The agent onset ``t_aon`` is the start of the first 10 ms frame that begins at least
100 ms of speech on the agent channel of the recording, according to a reference VAD.
Clicks (too short) and comfort noise (too quiet) therefore do not count.

The reference VAD is pluggable because it slightly biases every latency number:

* :class:`RMSReferenceVAD` (default, ``"rms"``): a frame counts as speech when its RMS
  level reaches an absolute threshold (-40 dBFS). numpy-only, deterministic and
  independent of any engine under test.
* :class:`ProviderReferenceVAD`: wraps any registered VAD provider (``"silero"``,
  ``"energy"``...) through its public streaming API and thresholds its probabilities.

Frames are laid out on exact 10 ms boundaries for any sample rate (frame ``i`` starts at
``round(i * 0.01 * rate)``), so onset times never drift over long recordings. By default
the onset is refined inside its frame to the first 1 ms block above the level threshold
(see :class:`OnsetDetector`), which removes up to one frame of quantization bias.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ..audio.frame import AudioFrame
from ..audio.resample import resample
from ..vad import VAD, VADEventType

__all__ = [
    "DEFAULT_FRAME_DURATION",
    "DEFAULT_MIN_SPEECH",
    "OnsetDetector",
    "ProviderReferenceVAD",
    "RMSReferenceVAD",
    "ReferenceVAD",
    "find_onsets",
    "first_onset_between",
    "frame_bounds",
    "frame_levels_db",
    "make_reference_vad",
    "speech_runs",
]

DEFAULT_FRAME_DURATION = 0.010
"""Analysis frame (seconds)."""
DEFAULT_MIN_SPEECH = 0.100
"""Speech needed after a frame for it to count as an onset (seconds)."""

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]


@runtime_checkable
class ReferenceVAD(Protocol):
    """Frame-level speech classifier used to annotate recordings."""

    name: str
    threshold: float
    """Probability at/above which a frame is speech."""

    def frame_probabilities(self, audio: AudioFrame, frame_duration: float) -> FloatArray:
        """Speech probability (0..1) of every ``frame_duration`` frame of ``audio``."""
        ...

    def describe(self) -> dict[str, Any]:
        """JSON-able description for the run manifest."""
        ...


def frame_bounds(
    num_samples: int, sample_rate: int, frame_duration: float
) -> npt.NDArray[np.int64]:
    """Sample index where every frame starts, plus the end of the last (partial) frame."""
    n_frames = math.ceil(num_samples / (frame_duration * sample_rate) - 1e-9) if num_samples else 0
    bounds = np.round(np.arange(n_frames + 1) * frame_duration * sample_rate).astype(np.int64)
    if n_frames:
        bounds[-1] = num_samples
    return bounds


def frame_levels_db(
    audio: AudioFrame, frame_duration: float = DEFAULT_FRAME_DURATION
) -> FloatArray:
    """RMS level (dBFS) of every frame; digital silence is ``-inf``."""
    x = audio.to_mono().to_numpy().astype(np.float64) / 32768.0
    bounds = frame_bounds(len(x), audio.sample_rate, frame_duration)
    if len(bounds) < 2:
        return np.zeros(0, dtype=np.float64)
    starts = bounds[:-1]
    sums = np.add.reduceat(np.square(x), starts) if len(x) else np.zeros(len(starts))
    counts = np.diff(bounds).astype(np.float64)
    mean_sq = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    with np.errstate(divide="ignore"):
        levels: FloatArray = 10.0 * np.log10(mean_sq)
    return levels


@dataclass(slots=True)
class RMSReferenceVAD:
    """Absolute RMS-level detector: a frame is speech when its level >= ``threshold_db``."""

    threshold_db: float = -40.0
    name: str = "rms"
    threshold: float = 0.5

    def frame_probabilities(self, audio: AudioFrame, frame_duration: float) -> FloatArray:
        return (frame_levels_db(audio, frame_duration) >= self.threshold_db).astype(np.float64)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "threshold_db": self.threshold_db}


class ProviderReferenceVAD:
    """Uses a registered VAD provider (instance or spec, e.g. ``"silero"``) as reference.

    The whole track is resampled once (delay-compensated) to the VAD's rate and fed
    through ``VAD.stream(emit_inference_events=True)``; every 10 ms frame takes the
    probability of the VAD window containing its midpoint.
    """

    def __init__(self, vad: VAD | str | Mapping[str, Any], *, threshold: float | None = None):
        from ..registry import create

        self.vad: VAD = create("vad", vad)
        self.name = f"vad:{self.vad.provider}"
        self.threshold = (
            threshold if threshold is not None else self.vad.options.activation_threshold
        )

    def frame_probabilities(self, audio: AudioFrame, frame_duration: float) -> FloatArray:
        mono = resample(audio.to_mono(), self.vad.sample_rate)
        n_frames = len(frame_bounds(mono.samples_per_channel, mono.sample_rate, frame_duration)) - 1
        if n_frames <= 0:
            return np.zeros(0, dtype=np.float64)
        stream = self.vad.stream(emit_inference_events=True)
        try:
            events = stream.push_audio(mono) if mono else []
        finally:
            stream.close()
        window = self.vad.window_duration
        probs = [e.probability for e in events if e.type == VADEventType.INFERENCE_DONE]
        if not probs:
            return np.zeros(n_frames, dtype=np.float64)
        mids = (np.arange(n_frames) + 0.5) * frame_duration
        idx = np.minimum((mids / window).astype(np.int64), len(probs) - 1)
        out: FloatArray = np.asarray(probs, dtype=np.float64)[idx]
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.vad.provider,
            "model": self.vad.model,
            "threshold": self.threshold,
            "window_ms": round(self.vad.window_duration * 1000, 3),
        }


def make_reference_vad(spec: str | ReferenceVAD | None = None) -> ReferenceVAD:
    """``"rms"`` / ``"rms:-45"`` (threshold in dBFS) or a VAD registry spec (``"silero"``)."""
    if spec is None:
        return RMSReferenceVAD()
    if not isinstance(spec, str):
        return spec
    name, _, arg = spec.partition(":")
    if name.strip().lower() == "rms":
        return RMSReferenceVAD(float(arg)) if arg else RMSReferenceVAD()
    return ProviderReferenceVAD(spec)


def speech_runs(mask: BoolArray, max_gap_frames: int = 0) -> list[tuple[int, int]]:
    """``(start, end)`` frame ranges (end exclusive) of speech; gaps <= ``max_gap_frames``
    between two runs are bridged."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    padded = np.concatenate([[False], m, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    runs = [(int(s), int(e)) for s, e in zip(edges[::2], edges[1::2], strict=True)]
    if max_gap_frames <= 0:
        return runs
    merged = [runs[0]]
    for start, end in runs[1:]:
        if start - merged[-1][1] <= max_gap_frames:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def find_onsets(
    mask: BoolArray,
    *,
    frame_duration: float = DEFAULT_FRAME_DURATION,
    min_speech: float = DEFAULT_MIN_SPEECH,
    max_gap: float = 0.0,
) -> list[float]:
    """Start times (s) of every speech run lasting at least ``min_speech``."""
    min_frames = max(1, round(min_speech / frame_duration))
    gap_frames = round(max_gap / frame_duration)
    return [
        start * frame_duration
        for start, end in speech_runs(mask, gap_frames)
        if end - start >= min_frames
    ]


def first_onset_between(
    onsets: list[float], start: float, end: float | None = None
) -> float | None:
    """First onset in ``[start, end)`` (``end=None``: unbounded)."""
    for t in onsets:
        if t >= start - 1e-9 and (end is None or t < end):
            return t
    return None


@dataclass(slots=True)
class OnsetDetector:
    """Finds speech onsets/segments in one channel with a :class:`ReferenceVAD`.

    The frame rule decides *which* speech counts (a run of >= ``min_speech``); with
    ``refine`` (default) the onset is then moved from the start of its 10 ms frame to the
    first 1 ms block inside that frame whose RMS level reaches ``refine_threshold_db``
    (default: the RMS reference threshold, -40 dBFS). This removes the up-to-one-frame
    quantization bias without changing which frame is chosen.
    """

    vad: ReferenceVAD = field(default_factory=RMSReferenceVAD)
    frame_duration: float = DEFAULT_FRAME_DURATION
    min_speech: float = DEFAULT_MIN_SPEECH
    max_gap: float = 0.0
    """Dips shorter than this inside speech are bridged (0 = the strict definition)."""
    refine: bool = True
    refine_threshold_db: float | None = None

    def speech_mask(self, audio: AudioFrame) -> BoolArray:
        probs = self.vad.frame_probabilities(audio, self.frame_duration)
        return np.asarray(probs >= self.vad.threshold, dtype=bool)

    def onsets(self, audio: AudioFrame, mask: BoolArray | None = None) -> list[float]:
        """Onset times (s from the start of ``audio``)."""
        m = self.speech_mask(audio) if mask is None else mask
        coarse = find_onsets(
            m, frame_duration=self.frame_duration, min_speech=self.min_speech, max_gap=self.max_gap
        )
        if not self.refine or not coarse:
            return coarse
        x = audio.to_mono().to_numpy().astype(np.float64) / 32768.0
        return [self._refine(x, audio.sample_rate, t) for t in coarse]

    def _refine_threshold(self) -> float:
        if self.refine_threshold_db is not None:
            return self.refine_threshold_db
        return float(getattr(self.vad, "threshold_db", -40.0))

    def _refine(self, x: FloatArray, rate: int, onset: float) -> float:
        start = round(onset * rate)
        end = min(round((onset + self.frame_duration) * rate), len(x))
        block = max(1, round(0.001 * rate))
        power = 10.0 ** (self._refine_threshold() / 10.0)
        for i in range(start, end, block):
            seg = x[i : min(i + block, end)]
            if seg.size and float(np.mean(np.square(seg))) >= power:
                return i / rate
        return onset

    def segments(
        self, audio: AudioFrame, mask: BoolArray | None = None
    ) -> list[tuple[float, float]]:
        """``(start, end)`` of every speech run lasting at least ``min_speech``."""
        m = self.speech_mask(audio) if mask is None else mask
        min_frames = max(1, round(self.min_speech / self.frame_duration))
        gap_frames = round(self.max_gap / self.frame_duration)
        return [
            (s * self.frame_duration, e * self.frame_duration)
            for s, e in speech_runs(m, gap_frames)
            if e - s >= min_frames
        ]

    def describe(self) -> dict[str, Any]:
        return {
            "reference_vad": self.vad.describe(),
            "frame_ms": round(self.frame_duration * 1000, 3),
            "min_speech_ms": round(self.min_speech * 1000, 3),
            "max_gap_ms": round(self.max_gap * 1000, 3),
            "refine": self.refine,
            "refine_threshold_db": self._refine_threshold() if self.refine else None,
        }
