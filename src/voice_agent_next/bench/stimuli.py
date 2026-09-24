"""Scenario manifests and pre-rendered user stimuli.

A *scenario* is a scripted conversation: user utterances, the pauses between them and
which turns expect a reply. It is written in YAML (see ``benchmarks/scenarios/``)::

    name: latency-smoke
    version: 1
    sample_rate: 16000
    loudness_dbfs: -20        # every stimulus is normalized to this RMS level
    lead_in: 0.5              # silence before the first utterance (s)
    stimuli: synthetic        # default source: synthetic | tts | wav
    reply_timeout: 8.0        # no agent speech this long after the user stopped -> missed
    gap_after_reply: 0.3      # the caller waits this long after the agent went quiet
    turns:
      - {id: time, text: What time is it?, duration: 0.6}
      - {id: book, text: Book a table for two., duration: 0.8, pause: 0.2}
      - {id: file, wav: data/hello.wav, speech: [0.12, 0.98]}
      - {id: tts, text: Read me the news., source: tts}

Stimuli are rendered **once** before a run (fixed seeds, fixed resampler, loudness
normalized) and each gets its speech boundaries annotated on the clean audio, which is
what ``t_uoff`` (annotated end of user speech) refers to:

* ``synthetic`` — :func:`~voice_agent_next.providers.mock.synth_speech` (a speech-like
  tone); the speech span is the whole clip. Offline and deterministic: the smoke tier.
* ``tts`` — synthesized from ``text`` with any registered TTS (``tts:`` spec);
* ``wav`` — a WAV file (path relative to the scenario file).

For ``tts``/``wav`` stimuli the span is ``speech: [start, end]`` when given, otherwise
annotated automatically (10 ms frames within 35 dB of the clip's loudest frame).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from ..audio.frame import AudioFrame
from ..audio.resample import resample
from ..audio.wav import read_wav
from ..errors import ConfigurationError
from .onset import frame_levels_db

__all__ = [
    "BUILTIN_SCENARIOS",
    "Scenario",
    "Stimulus",
    "StimulusSource",
    "TurnSpec",
    "annotate_speech",
    "builtin_scenario",
    "load_scenario",
    "normalize_loudness",
    "render_stimuli",
]

StimulusSource = Literal["synthetic", "tts", "wav"]

_CHARS_PER_SECOND = 14.0  # default speaking rate for synthetic stimuli without a duration
_AUTO_ANNOTATION_RANGE_DB = 35.0


class TurnSpec(BaseModel):
    """One scripted user utterance."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    text: str | None = None
    """What the user says (TTS input; transcript for reports)."""
    source: StimulusSource | None = None
    """Overrides the scenario's default ``stimuli`` source."""
    duration: float | None = Field(default=None, gt=0)
    """Synthetic speech duration (s). Default: from the text length (14 chars/s)."""
    frequency: float = Field(default=200.0, gt=0)
    """Synthetic speech pitch (Hz)."""
    wav: str | None = None
    """WAV file (``wav`` source), relative to the scenario file."""
    speech: tuple[float, float] | None = None
    """Annotated speech span within the clip (s); automatic when omitted."""
    expect_reply: bool = True
    """The agent is expected to answer this turn."""
    pause: float = Field(default=0.0, ge=0)
    """Minimum silence after the utterance before the caller may speak again (s)."""

    @field_validator("speech")
    @classmethod
    def _check_span(cls, v: tuple[float, float] | None) -> tuple[float, float] | None:
        if v is not None and not 0.0 <= v[0] < v[1]:
            raise ValueError("speech must be [start, end] with 0 <= start < end")
        return v


class Scenario(BaseModel):
    """A scripted conversation (see the module docstring for the YAML format)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: int = 1
    description: str = ""
    sample_rate: int = Field(default=16_000, gt=0)
    """Rate of the user audio stream (the transport's input format)."""
    chunk: float = Field(default=0.02, gt=0, le=0.2)
    """Real-time streaming chunk (s)."""
    loudness_dbfs: float | None = -20.0
    """RMS level of the speech span of every stimulus (``null`` keeps the source level)."""
    lead_in: float = Field(default=0.5, ge=0)
    """Silence streamed before the first utterance (s)."""
    stimuli: StimulusSource = "synthetic"
    tts: str | dict[str, Any] | None = None
    """TTS spec used to render ``tts`` stimuli (e.g. ``kokoro`` or ``{provider: ...}``)."""
    reply_timeout: float = Field(default=8.0, gt=0)
    """A turn without agent speech this long after the end of user speech is *missed*."""
    gap_after_reply: float = Field(default=0.3, ge=0)
    """The caller speaks again once the agent has been quiet (and idle) this long."""
    max_reply: float = Field(default=60.0, gt=0)
    """Upper bound for waiting until a reply finishes (s)."""
    turns: list[TurnSpec] = Field(min_length=1)

    _base_dir: Path | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _check_turns(self) -> Scenario:
        for i, turn in enumerate(self.turns):
            source = turn.source or self.stimuli
            if source == "wav" and not turn.wav:
                raise ValueError(f"turn {i}: 'wav' stimuli need a `wav:` path")
            if source == "tts" and not turn.text:
                raise ValueError(f"turn {i}: 'tts' stimuli need `text:`")
            if source == "tts" and self.tts is None:
                raise ValueError(f"turn {i}: 'tts' stimuli need a scenario-level `tts:` spec")
            if source == "synthetic" and turn.duration is None and not turn.text:
                raise ValueError(f"turn {i}: synthetic stimuli need `duration:` or `text:`")
        return self

    @property
    def base_dir(self) -> Path | None:
        """Directory relative ``wav:`` paths are resolved against."""
        return self._base_dir

    def with_base_dir(self, base_dir: str | os.PathLike[str] | None) -> Scenario:
        self._base_dir = Path(base_dir) if base_dir is not None else None
        return self

    def turn_id(self, index: int) -> str:
        spec = self.turns[index % len(self.turns)]
        return spec.id or f"turn{index % len(self.turns):02d}"

    def definition_sha256(self) -> str:
        """Hash of the scenario definition (canonical JSON)."""
        data = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(data.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Stimulus:
    """A pre-rendered user utterance with annotated speech boundaries.

    ``audio`` is mono s16le at the scenario rate, padded with silence to whole chunks;
    ``speech_start``/``speech_end`` are seconds from the start of the clip.
    """

    id: str
    text: str | None
    audio: AudioFrame
    speech_start: float
    speech_end: float
    source: StimulusSource
    expect_reply: bool = True
    pause: float = 0.0
    gain_db: float = 0.0

    @property
    def duration(self) -> float:
        return self.audio.duration

    @property
    def speech_duration(self) -> float:
        return self.speech_end - self.speech_start

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.audio.data).hexdigest()

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "duration_s": round(self.duration, 6),
            "speech_start_s": round(self.speech_start, 6),
            "speech_end_s": round(self.speech_end, 6),
            "expect_reply": self.expect_reply,
            "pause_s": self.pause,
            "gain_db": round(self.gain_db, 3),
            "sha256": self.sha256,
        }


# ------------------------------------------------------------------------ loading

BUILTIN_SCENARIOS: dict[str, dict[str, Any]] = {
    "latency-smoke": {
        "name": "latency-smoke",
        "version": 1,
        "description": (
            "T1 smoke tier: short scripted questions rendered as deterministic synthetic "
            "speech (no models, no network). Cycled to the requested number of turns."
        ),
        "sample_rate": 16_000,
        "chunk": 0.02,
        "loudness_dbfs": -20.0,
        "lead_in": 0.5,
        "stimuli": "synthetic",
        "reply_timeout": 8.0,
        "gap_after_reply": 0.25,
        "turns": [
            {"id": "time", "text": "What time is it?", "duration": 0.6},
            {"id": "weather", "text": "How is the weather today?", "duration": 0.8},
            {"id": "table", "text": "Book a table for two.", "duration": 0.7},
            {"id": "yes", "text": "Yes, please.", "duration": 0.4},
            {"id": "order", "text": "Where is my order?", "duration": 0.6},
            {"id": "open", "text": "Are you open on Sunday?", "duration": 0.7},
            {"id": "card", "text": "I lost my card.", "duration": 0.5},
            {"id": "flight", "text": "Change my flight to Friday.", "duration": 0.9},
            {"id": "repeat", "text": "Can you repeat that?", "duration": 0.6},
            {"id": "thanks", "text": "Thanks, that is all.", "duration": 0.6},
        ],
    }
}


def builtin_scenario(name: str) -> Scenario:
    """A scenario shipped with the library (``latency-smoke``)."""
    try:
        data = BUILTIN_SCENARIOS[name]
    except KeyError:
        raise ConfigurationError(
            f"unknown built-in scenario {name!r}; built-in: {', '.join(sorted(BUILTIN_SCENARIOS))}"
        ) from None
    return Scenario.model_validate(data)


def load_scenario(source: str | os.PathLike[str] | dict[str, Any] | Scenario) -> Scenario:
    """Load a scenario from a YAML/JSON file, a built-in name or a mapping."""
    if isinstance(source, Scenario):
        return source
    if isinstance(source, dict):
        return Scenario.model_validate(source)
    path = Path(source)
    if not path.exists():
        if isinstance(source, str) and source in BUILTIN_SCENARIOS:
            return builtin_scenario(source)
        raise ConfigurationError(
            f"scenario not found: {source} (built-in: {', '.join(sorted(BUILTIN_SCENARIOS))})"
        )
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ConfigurationError(f"{path}: scenario root must be a mapping")
    try:
        scenario = Scenario.model_validate(data)
    except ValueError as exc:
        raise ConfigurationError(f"{path}: invalid scenario: {exc}") from exc
    return scenario.with_base_dir(path.parent)


# ---------------------------------------------------------------------- rendering


def annotate_speech(
    audio: AudioFrame, *, range_db: float = _AUTO_ANNOTATION_RANGE_DB, frame_duration: float = 0.01
) -> tuple[float, float]:
    """Speech span of a clean clip: first to last 10 ms frame within ``range_db`` of the
    loudest frame. Raises for silent clips."""
    levels = frame_levels_db(audio, frame_duration)
    finite = levels[np.isfinite(levels)]
    if finite.size == 0:
        raise ValueError("cannot annotate a silent clip")
    speech = np.flatnonzero(levels >= finite.max() - range_db)
    start = speech[0] * frame_duration
    end = min((speech[-1] + 1) * frame_duration, audio.duration)
    return float(start), float(end)


def normalize_loudness(
    audio: AudioFrame, target_dbfs: float, span: tuple[float, float] | None = None
) -> tuple[AudioFrame, float]:
    """Scale ``audio`` so the RMS of ``span`` (default: all) is ``target_dbfs``.

    The gain is limited so the peak stays below full scale. Returns ``(audio, gain_db)``.
    """
    x = audio.to_float32()
    ref = audio.slice(*span).to_float32() if span else x
    rms = float(np.sqrt(np.mean(np.square(ref, dtype=np.float64)))) if ref.size else 0.0
    if rms <= 0:
        return audio, 0.0
    gain = 10.0 ** ((target_dbfs - 20.0 * math.log10(rms)) / 20.0)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0:
        gain = min(gain, 0.999 / peak)
    scaled = AudioFrame.from_numpy(x * gain, audio.sample_rate, channels=audio.channels)
    return scaled, 20.0 * math.log10(gain)


def _pad_to_chunks(audio: AudioFrame, chunk: float) -> AudioFrame:
    n = max(1, round(chunk * audio.sample_rate))
    rem = audio.samples_per_channel % n
    if rem == 0 and audio.samples_per_channel:
        return audio
    pad = AudioFrame.silence((n - rem) / audio.sample_rate, audio.sample_rate, audio.channels)
    return AudioFrame.concat([audio, pad]) if audio else pad


async def _render_tts(spec: str | dict[str, Any], texts: Sequence[str]) -> list[AudioFrame]:
    from ..registry import create

    tts = create("tts", spec)
    try:
        return [(await tts.synthesize(text).collect()).to_mono() for text in texts]
    finally:
        await tts.aclose()


async def render_stimuli(scenario: Scenario, *, turns: int | None = None) -> list[Stimulus]:
    """Render ``turns`` stimuli (default: one per scenario turn, cycling when more).

    Each distinct scenario turn is rendered once; repetitions share the audio.
    """
    from ..providers.mock import synth_speech

    count = len(scenario.turns) if turns is None else turns
    if count < 1:
        raise ValueError("turns must be >= 1")
    rate = scenario.sample_rate
    used = sorted({i % len(scenario.turns) for i in range(count)})
    sources = {i: scenario.turns[i].source or scenario.stimuli for i in used}
    tts_idx = [i for i in used if sources[i] == "tts"]
    tts_audio: dict[int, AudioFrame] = {}
    if tts_idx:
        assert scenario.tts is not None  # checked by the model validator
        rendered = await _render_tts(scenario.tts, [scenario.turns[i].text or "" for i in tts_idx])
        tts_audio = dict(zip(tts_idx, rendered, strict=True))

    cache: dict[int, Stimulus] = {}
    for i in used:
        spec = scenario.turns[i]
        source = sources[i]
        span = spec.speech
        if source == "synthetic":
            duration = spec.duration or max(0.4, len(spec.text or "") / _CHARS_PER_SECOND)
            audio = synth_speech(round(duration, 3), rate, frequency=spec.frequency)
            span = span or (0.0, audio.duration)
        elif source == "tts":
            audio = resample(tts_audio[i], rate)
        else:
            assert spec.wav is not None
            path = Path(spec.wav)
            if not path.is_absolute() and scenario.base_dir is not None:
                path = scenario.base_dir / path
            if not path.exists():
                raise ConfigurationError(f"stimulus WAV not found: {path}")
            audio = resample(read_wav(path).to_mono(), rate)
        if not audio:
            raise ConfigurationError(f"turn {scenario.turn_id(i)!r} rendered no audio")
        if span is None:
            span = annotate_speech(audio)
        if span[1] > audio.duration + 1e-6:
            raise ConfigurationError(
                f"turn {scenario.turn_id(i)!r}: speech span ends after the clip "
                f"({audio.duration:.3f} s)"
            )
        gain_db = 0.0
        if scenario.loudness_dbfs is not None:
            audio, gain_db = normalize_loudness(audio, scenario.loudness_dbfs, span)
        cache[i] = Stimulus(
            id=scenario.turn_id(i),
            text=spec.text,
            audio=_pad_to_chunks(audio, scenario.chunk),
            speech_start=span[0],
            speech_end=span[1],
            source=source,
            expect_reply=spec.expect_reply,
            pause=spec.pause,
            gain_db=gain_db,
        )
    return [cache[i % len(scenario.turns)] for i in range(count)]
