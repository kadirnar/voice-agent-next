"""A deterministic, frame-labelled VAD test corpus built from speech utterances.

Real speech (by default the pinned LibriSpeech smoke subset, see
:mod:`voice_agent_next.bench.asr_datasets`) is laid out on a timeline with silence gaps of
known length and mixed with noise at known levels. Because every utterance is placed at a
known sample offset, the reference labels are exact up to how the clean utterance itself
is labelled:

* **labels** — 10 ms frames of each *clean* utterance whose level is within 40 dB of its
  loudest frame and at least 10 dB above its noise floor (10th percentile of the frame
  levels; this floor rule never asks for more than 20 dB below the peak); dips shorter
  than 150 ms inside speech (stop closures, short breaths) are bridged and runs shorter
  than 30 ms dropped. An utterance's span (for onset / offset
  latency) is its first to last speech frame;
* **layout** — ``lead`` seconds of noise, then every utterance followed by a gap drawn
  uniformly from ``gap`` (seeded, rounded to 10 ms), then ``tail`` seconds of noise only;
  utterances are normalized to -20 dBFS speech RMS. The layout is identical in every
  condition, only the noise changes;
* **conditions** — the noise: ``clean`` (white noise at -70 dBFS: a quiet line, not
  digital silence), ``white`` / ``pink`` / ``brown`` noise at an SNR (noise RMS relative to
  the -20 dBFS speech level), and ``transient`` (keyboard-like clicks at up to -12 dBFS
  over a -60 dBFS pink floor, 4 per second on average).

Everything is generated from ``seed`` with numpy's PCG64, so the corpus (and its SHA-256,
recorded in the run manifest) is identical on every machine.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from ..audio.frame import AudioFrame
from ..audio.resample import resample
from .onset import frame_levels_db, speech_runs

__all__ = [
    "DEFAULT_CONDITIONS",
    "FRAME",
    "SAMPLE_RATE",
    "VadClip",
    "VadCondition",
    "VadCorpus",
    "VadUtterance",
    "build_vad_corpus",
    "make_noise",
    "parse_condition",
    "speech_labels",
]

SAMPLE_RATE = 16_000
FRAME = 0.010
_FRAME_SAMPLES = round(SAMPLE_RATE * FRAME)
SPEECH_DBFS = -20.0

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class VadCondition:
    """A noise condition: ``kind`` (clean, white, pink, brown, transient) and SNR (dB)."""

    kind: str
    snr_db: float | None = None

    @property
    def name(self) -> str:
        return self.kind if self.snr_db is None else f"{self.kind}@{self.snr_db:g}dB"


DEFAULT_CONDITIONS: tuple[VadCondition, ...] = (
    VadCondition("clean"),
    VadCondition("pink", 20.0),
    VadCondition("pink", 10.0),
    VadCondition("pink", 5.0),
    VadCondition("white", 10.0),
    VadCondition("transient"),
)
_KINDS = ("clean", "white", "pink", "brown", "transient")


def parse_condition(text: str) -> VadCondition:
    """``clean``, ``transient``, ``pink@10`` / ``pink@10dB`` / ``white:5``."""
    kind, sep, snr = text.replace(":", "@").partition("@")
    kind = kind.strip().lower()
    if kind not in _KINDS:
        raise ValueError(f"unknown noise {kind!r}; use one of {', '.join(_KINDS)}")
    if kind in ("clean", "transient"):
        if sep:
            raise ValueError(f"{kind} takes no SNR")
        return VadCondition(kind)
    if not sep:
        raise ValueError(f"{kind} needs an SNR, e.g. {kind}@10")
    return VadCondition(kind, float(snr.strip().lower().removesuffix("db")))


@dataclass(frozen=True, slots=True)
class VadUtterance:
    id: str
    start: float
    """Labelled speech start on the clip timeline (s)."""
    end: float
    """Labelled speech end (s)."""
    placed_at: float
    """Where the utterance's audio begins (s)."""


@dataclass
class VadClip:
    """One condition: audio, frame labels and utterance spans."""

    condition: VadCondition
    audio: AudioFrame
    labels: BoolArray
    utterances: list[VadUtterance]

    @property
    def duration(self) -> float:
        return self.audio.duration

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.audio.data).hexdigest()


@dataclass
class VadCorpus:
    clips: list[VadClip]
    sources: list[dict[str, Any]] = field(default_factory=list)
    """Source utterances (id, SHA-256 of the samples, speech seconds)."""
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        """Hash over the parameters, the labels and the audio of every clip."""
        h = hashlib.sha256()
        h.update(json.dumps(self.params, sort_keys=True).encode())
        for clip in self.clips:
            h.update(clip.condition.name.encode())
            h.update(clip.audio.data)
            h.update(np.packbits(clip.labels).tobytes())
        return h.hexdigest()

    def describe(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "params": self.params,
            "sources": self.sources,
            "clips": [
                {
                    "condition": c.condition.name,
                    "duration_s": round(c.duration, 3),
                    "speech_s": round(float(c.labels.sum()) * FRAME, 3),
                    "utterances": len(c.utterances),
                    "sha256": c.sha256,
                }
                for c in self.clips
            ],
        }


# ------------------------------------------------------------------------ labels


def speech_labels(
    audio: AudioFrame,
    *,
    range_db: float = 40.0,
    floor_margin_db: float = 10.0,
    bridge: float = 0.15,
    min_run: float = 0.03,
) -> BoolArray:
    """10 ms speech labels of a clean utterance (see the module docstring)."""
    levels = frame_levels_db(audio, FRAME)
    finite = levels[np.isfinite(levels)]
    if finite.size == 0:
        return np.zeros(len(levels), dtype=bool)
    peak = float(finite.max())
    floor = float(np.percentile(np.where(np.isfinite(levels), levels, -150.0), 10))
    thr = max(peak - range_db, min(floor + floor_margin_db, peak - 20.0))
    mask = np.asarray(levels >= thr, dtype=bool)
    out = np.zeros_like(mask)
    min_frames = max(1, round(min_run / FRAME))
    for s, e in speech_runs(mask, round(bridge / FRAME)):
        if e - s >= min_frames:
            out[s:e] = True
    return out


# ------------------------------------------------------------------------- noise


def make_noise(kind: str, n: int, rng: np.random.Generator) -> FloatArray:
    """Unit-RMS noise of ``kind`` (white, pink, brown), ``n`` samples."""
    white = rng.standard_normal(n)
    if kind == "white":
        x = white
    elif kind in ("pink", "brown"):
        spec = np.fft.rfft(white)
        f = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
        f[0] = f[1] if len(f) > 1 else 1.0
        spec = spec / (np.sqrt(f) if kind == "pink" else f)
        x = np.fft.irfft(spec, n)
    else:
        raise ValueError(f"unknown noise {kind!r}")
    rms = float(np.sqrt(np.mean(x**2))) or 1.0
    return np.asarray(x / rms, dtype=np.float64)


def _clicks(n: int, rng: np.random.Generator, rate_hz: float = 4.0) -> FloatArray:
    """Keyboard-like clicks: 5 ms decaying broadband bursts at random times and levels."""
    out = np.zeros(n)
    count = rng.poisson(rate_hz * n / SAMPLE_RATE)
    length = round(0.005 * SAMPLE_RATE)
    env = np.exp(-np.arange(length) / (0.0012 * SAMPLE_RATE))
    for pos in np.sort(rng.integers(0, max(1, n - length), size=count)):
        peak = 10 ** (rng.uniform(-30.0, -12.0) / 20.0)
        burst = rng.standard_normal(length) * env
        burst *= peak / (np.max(np.abs(burst)) or 1.0)
        out[pos : pos + length] += burst
    return out


# ------------------------------------------------------------------------ corpus


def _prepare(audio: AudioFrame) -> tuple[FloatArray, BoolArray]:
    mono = audio.to_mono()
    if mono.sample_rate != SAMPLE_RATE:
        mono = resample(mono, SAMPLE_RATE)
    labels = speech_labels(mono)
    x = mono.to_float32().astype(np.float64)
    speech = np.concatenate(
        [x[i * _FRAME_SAMPLES : (i + 1) * _FRAME_SAMPLES] for i in np.flatnonzero(labels)]
    ) if labels.any() else x  # fmt: skip
    rms = float(np.sqrt(np.mean(speech**2))) if speech.size else 0.0
    if rms > 0:
        x = x * (10 ** (SPEECH_DBFS / 20.0) / rms)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0.999:
        x = x * (0.999 / peak)
    # whole frames only, so labels and samples stay aligned
    frames = len(labels)
    x = np.pad(x, (0, max(0, frames * _FRAME_SAMPLES - len(x))))[: frames * _FRAME_SAMPLES]
    return x, labels


def build_vad_corpus(
    sources: Sequence[tuple[str, AudioFrame]],
    *,
    conditions: Sequence[VadCondition] = DEFAULT_CONDITIONS,
    seed: int = 0,
    gap: tuple[float, float] = (0.8, 2.5),
    lead: float = 2.0,
    tail: float = 20.0,
) -> VadCorpus:
    """Lay ``sources`` (id, clean utterance) out with gaps and mix every condition."""
    if not sources:
        raise ValueError("no source utterances")
    if not conditions:
        raise ValueError("no conditions")
    rng = np.random.default_rng(seed)
    prepared = [(uid, *_prepare(audio)) for uid, audio in sources]
    gaps = [round(float(rng.uniform(*gap)) / FRAME) for _ in prepared]
    lead_f, tail_f = round(lead / FRAME), round(tail / FRAME)
    total_f = lead_f + sum(len(lab) + g for (_, _, lab), g in zip(prepared, gaps, strict=True))
    total_f += tail_f
    n = total_f * _FRAME_SAMPLES
    speech = np.zeros(n)
    labels = np.zeros(total_f, dtype=bool)
    utterances: list[VadUtterance] = []
    source_info: list[dict[str, Any]] = []
    f = lead_f
    for (uid, x, lab), g in zip(prepared, gaps, strict=True):
        speech[f * _FRAME_SAMPLES : f * _FRAME_SAMPLES + len(x)] = x
        labels[f : f + len(lab)] = lab
        idx = np.flatnonzero(lab)
        if idx.size:
            utterances.append(
                VadUtterance(uid, (f + idx[0]) * FRAME, (f + idx[-1] + 1) * FRAME, f * FRAME)
            )
        source_info.append(
            {"id": uid, "sha256": hashlib.sha256(x.astype(np.float32).tobytes()).hexdigest(),
             "speech_s": round(float(lab.sum()) * FRAME, 3)}
        )  # fmt: skip
        f += len(lab) + g
    speech_level = 10 ** (SPEECH_DBFS / 20.0)
    clips: list[VadClip] = []
    for k, cond in enumerate(conditions):
        crng = np.random.default_rng([seed, k + 1])
        if cond.kind == "clean":
            noise = make_noise("white", n, crng) * 10 ** (-70.0 / 20.0)
        elif cond.kind == "transient":
            noise = make_noise("pink", n, crng) * 10 ** (-60.0 / 20.0) + _clicks(n, crng)
        else:
            snr = 10.0 if cond.snr_db is None else cond.snr_db
            noise = make_noise(cond.kind, n, crng) * speech_level * 10 ** (-snr / 20.0)
        mix = np.clip(speech + noise, -1.0, 32767 / 32768)
        audio = AudioFrame.from_numpy(mix.astype(np.float32), SAMPLE_RATE)
        clips.append(VadClip(cond, audio, labels.copy(), list(utterances)))
    params = {
        "seed": seed,
        "gap_s": list(gap),
        "lead_s": lead,
        "tail_s": tail,
        "speech_dbfs": SPEECH_DBFS,
        "conditions": [c.name for c in conditions],
        "sources": [s["sha256"] for s in source_info],
    }
    return VadCorpus(clips, source_info, params)
