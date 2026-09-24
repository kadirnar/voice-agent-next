"""Voice activity detection (VAD) interface and the generic streaming state machine.

A VAD provider only implements :meth:`VAD._new_inference`, returning a per-stream
callable that maps one window of float32 samples (exactly ``window_samples`` long,
at ``sample_rate``) to a speech probability. :class:`VADStream` handles resampling,
windowing, hysteresis, minimum durations and prefix padding.

``VADStream.push_audio`` is synchronous and returns the events produced by that
frame, which keeps turn-taking logic deterministic and easy to test.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar, Protocol

import numpy as np
import numpy.typing as npt

from .audio.buffer import AudioBuffer, FrameChunker
from .audio.frame import AudioFrame
from .audio.resample import StreamResampler
from .metrics import VADMetrics
from .utils.clock import now
from .utils.emitter import EventEmitter

__all__ = ["VAD", "VADEvent", "VADEventType", "VADInference", "VADOptions", "VADStream"]


class VADEventType(StrEnum):
    START_OF_SPEECH = "start_of_speech"
    INFERENCE_DONE = "inference_done"
    END_OF_SPEECH = "end_of_speech"


@dataclass(slots=True)
class VADEvent:
    type: VADEventType
    speaking: bool
    probability: float = 0.0
    speech_duration: float = 0.0
    """Duration of detected speech in the current segment (seconds)."""
    silence_duration: float = 0.0
    """Trailing silence observed so far (seconds)."""
    audio_time: float = 0.0
    """Position in the input stream (seconds of audio pushed so far) when the event fired."""
    frames: list[AudioFrame] = field(default_factory=list)
    """START: prefix padding + speech so far. END: the full utterance (incl. padding)."""
    inference_duration: float = 0.0
    timestamp: float = field(default_factory=now)


@dataclass(slots=True)
class VADOptions:
    activation_threshold: float = 0.5
    """Probability at/above which a window counts as speech."""
    deactivation_threshold: float | None = None
    """Probability below which a window counts as silence (default: activation - 0.15)."""
    min_speech_duration: float = 0.1
    """Speech needed before START_OF_SPEECH fires (seconds)."""
    min_silence_duration: float = 0.25
    """Silence needed before END_OF_SPEECH fires (seconds). Short on purpose: it is a
    *candidate* pause; engines decide the end of turn (turn detector / endpointing)."""
    prefix_padding_duration: float = 0.5
    """Audio kept before speech start and included in events (seconds)."""
    max_buffered_speech: float = 60.0
    """Cap on the audio kept for one utterance (seconds)."""
    smoothing: float = 0.0
    """Exponential smoothing of probabilities, 0 (off) .. 0.9 (heavy)."""

    def __post_init__(self) -> None:
        if not 0.0 < self.activation_threshold <= 1.0:
            raise ValueError("activation_threshold must be in (0, 1]")
        if not 0.0 <= self.smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")

    @property
    def effective_deactivation(self) -> float:
        if self.deactivation_threshold is not None:
            return self.deactivation_threshold
        return max(self.activation_threshold - 0.15, 0.01)


class VADInference(Protocol):
    """Per-stream inference state (e.g. RNN state for Silero)."""

    def __call__(self, window: npt.NDArray[np.float32]) -> float: ...

    def reset(self) -> None: ...


class VAD(ABC, EventEmitter):
    """Base class for VAD providers. Emits ``"metrics"`` (:class:`VADMetrics`)."""

    provider: ClassVar[str] = "unknown"

    def __init__(
        self,
        *,
        sample_rate: int,
        window_samples: int,
        options: VADOptions | None = None,
        model: str = "",
    ) -> None:
        EventEmitter.__init__(self)
        self.sample_rate = sample_rate
        self.window_samples = window_samples
        self.options = options or VADOptions()
        self.model = model

    @property
    def window_duration(self) -> float:
        return self.window_samples / self.sample_rate

    @abstractmethod
    def _new_inference(self) -> VADInference:
        """Create per-stream inference state."""

    def stream(self, *, emit_inference_events: bool = False) -> VADStream:
        return VADStream(self, emit_inference_events=emit_inference_events)

    async def warmup(self) -> None:
        inf = self._new_inference()
        inf(np.zeros(self.window_samples, dtype=np.float32))

    async def aclose(self) -> None:
        """Release resources."""


class VADStream:
    """Stateful speech segmentation over a stream of frames (any rate / channels)."""

    _METRICS_INTERVAL = 5.0  # seconds of audio between VADMetrics emissions

    def __init__(self, vad: VAD, *, emit_inference_events: bool = False) -> None:
        self._vad = vad
        self._opts = vad.options
        self._emit_inference = emit_inference_events
        self._win = vad.window_samples / vad.sample_rate
        self._resampler = StreamResampler(vad.sample_rate, 1)
        self._chunker = FrameChunker(vad.sample_rate, samples_per_frame=vad.window_samples)
        self._infer = vad._new_inference()
        self._prefix = AudioBuffer(vad.sample_rate, max_duration=self._opts.prefix_padding_duration)
        self._reset_state()
        self._samples_total = 0
        self._inference_count = 0
        self._inference_time = 0.0
        self._metrics_audio = 0.0

    def _reset_state(self) -> None:
        self._speaking = False
        self._pending: list[AudioFrame] = []
        self._speech: list[AudioFrame] = []
        self._speech_acc = 0.0
        self._silence_acc = 0.0
        self._speech_duration = 0.0
        self._prob = 0.0
        self._prefix.clear()

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def probability(self) -> float:
        """Latest (smoothed) speech probability."""
        return self._prob

    @property
    def audio_time(self) -> float:
        return self._samples_total / self._vad.sample_rate

    def speech_frames(self) -> list[AudioFrame]:
        """Frames of the utterance in progress (empty when not speaking)."""
        return list(self._speech) if self._speaking else []

    def reset(self) -> None:
        """Forget the current segment and model state (keeps metrics counters)."""
        self._reset_state()
        self._chunker.reset()
        self._infer.reset()

    def push_audio(self, frame: AudioFrame) -> list[VADEvent]:
        out = self._resampler.push(frame)
        events: list[VADEvent] = []
        if not out:
            return events
        for window in self._chunker.push(out):
            events.extend(self._process(window))
        return events

    def close(self) -> None:
        self._emit_metrics()

    # ----------------------------------------------------------------- internals
    def _emit_metrics(self) -> None:
        if not self._inference_count:
            return
        self._vad.emit(
            "metrics",
            VADMetrics(
                provider=self._vad.provider,
                inference_count=self._inference_count,
                inference_duration_total=self._inference_time,
                audio_duration=self._metrics_audio,
            ),
        )
        self._inference_count = 0
        self._inference_time = 0.0
        self._metrics_audio = 0.0

    def _process(self, window: AudioFrame) -> list[VADEvent]:
        opts = self._opts
        x = window.to_float32()
        t0 = now()
        raw = float(self._infer(x))
        dt = now() - t0
        self._inference_count += 1
        self._inference_time += dt
        self._samples_total += window.samples_per_channel
        self._metrics_audio += self._win
        if self._metrics_audio >= self._METRICS_INTERVAL:
            self._emit_metrics()

        a = opts.smoothing
        self._prob = raw if a == 0 else a * self._prob + (1 - a) * raw
        p = self._prob
        events: list[VADEvent] = []

        if not self._speaking:
            if p >= opts.activation_threshold:
                self._pending.append(window)
                self._speech_acc += self._win
                if self._speech_acc >= opts.min_speech_duration:
                    self._speaking = True
                    prefix = [self._prefix.to_frame()] if self._prefix else []
                    self._speech = prefix + self._pending
                    self._speech_duration = self._speech_acc
                    self._silence_acc = 0.0
                    self._pending = []
                    self._prefix.clear()
                    events.append(
                        VADEvent(
                            VADEventType.START_OF_SPEECH,
                            speaking=True,
                            probability=p,
                            speech_duration=self._speech_duration,
                            audio_time=self.audio_time,
                            frames=list(self._speech),
                            inference_duration=dt,
                        )
                    )
            else:
                for f in self._pending:
                    self._prefix.append(f)
                self._pending = []
                self._speech_acc = 0.0
                self._prefix.append(window)
        else:
            self._speech.append(window)
            max_frames = max(1, round(opts.max_buffered_speech / self._win))
            if len(self._speech) > max_frames:
                del self._speech[: len(self._speech) - max_frames]
            if p < opts.effective_deactivation:
                self._silence_acc += self._win
                if self._silence_acc >= opts.min_silence_duration:
                    events.append(
                        VADEvent(
                            VADEventType.END_OF_SPEECH,
                            speaking=False,
                            probability=p,
                            speech_duration=self._speech_duration,
                            silence_duration=self._silence_acc,
                            audio_time=self.audio_time,
                            frames=self._speech,
                            inference_duration=dt,
                        )
                    )
                    self._speaking = False
                    self._speech = []
                    self._speech_acc = 0.0
                    self._silence_acc = 0.0
                    self._speech_duration = 0.0
            else:
                self._silence_acc = 0.0
                self._speech_duration += self._win

        if self._emit_inference:
            events.append(
                VADEvent(
                    VADEventType.INFERENCE_DONE,
                    speaking=self._speaking,
                    probability=p,
                    speech_duration=self._speech_duration,
                    silence_duration=self._silence_acc,
                    audio_time=self.audio_time,
                    inference_duration=dt,
                )
            )
        return events
