"""Streaming trimming of leading/trailing silence in synthesized speech.

Many TTS models pad every utterance with silence (Kokoro: ~40–120 ms before and ~0.5 s
after each sentence). In a streaming cascade that synthesizes sentence by sentence this
delays the first audible sample and inserts long, unnatural pauses between sentences.
:class:`SilenceTrimmer` removes that padding while keeping pauses *inside* the utterance.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .buffer import FrameChunker
from .frame import AudioFrame

__all__ = ["SilenceTrimmer"]


class SilenceTrimmer:
    """Trims one utterance, streaming: push frames, then :meth:`flush` at its end.

    Args:
        sample_rate: rate of the pushed audio.
        channels: channel count of the pushed audio.
        threshold_db: windows below this RMS level (dBFS) count as silence.
        keep_leading: silence kept before the first speech (seconds).
        keep_trailing: silence kept after the last speech (seconds).
        window: analysis window (seconds).
    """

    def __init__(
        self,
        sample_rate: int,
        channels: int = 1,
        *,
        threshold_db: float = -45.0,
        keep_leading: float = 0.02,
        keep_trailing: float = 0.1,
        window: float = 0.01,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.threshold = 10.0 ** (threshold_db / 20.0)
        self._chunker = FrameChunker(sample_rate, channels, frame_duration=window)
        self._window = self._chunker.samples_per_frame / sample_rate
        self._keep_leading = max(0, round(keep_leading / self._window))
        self._keep_trailing = max(0, round(keep_trailing / self._window))
        self._lead: deque[AudioFrame] = deque()
        self._held: list[AudioFrame] = []
        self.started = False
        """True once the first non-silent window was seen."""
        self.dropped_leading = 0.0
        """Seconds of leading silence removed (to shift word timestamps)."""
        self.dropped_trailing = 0.0

    def _silent(self, frame: AudioFrame) -> bool:
        x = frame.to_float32()
        return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) < self.threshold

    def push(self, frame: AudioFrame) -> AudioFrame:
        """Feed audio; returns the audio that can be emitted now (possibly empty)."""
        out: list[AudioFrame] = []
        for w in self._chunker.push(frame):
            self._process(w, out)
        return self._join(out)

    def flush(self) -> AudioFrame:
        """End of the utterance: returns the remaining audio (trailing silence capped)."""
        out: list[AudioFrame] = []
        for w in self._chunker.flush():
            self._process(w, out)
        if self.started:
            kept = self._held[: self._keep_trailing]
            out.extend(kept)
            self.dropped_trailing += sum(f.duration for f in self._held[len(kept) :])
        self._held = []
        self._lead.clear()
        return self._join(out)

    def _process(self, w: AudioFrame, out: list[AudioFrame]) -> None:
        silent = self._silent(w)
        if not self.started:
            if silent:
                self._lead.append(w)
                while len(self._lead) > self._keep_leading:
                    self.dropped_leading += self._lead.popleft().duration
                return
            self.started = True
            out.extend(self._lead)
            self._lead.clear()
            out.append(w)
            return
        if silent:
            self._held.append(w)
            return
        out.extend(self._held)  # a pause inside the utterance: keep it
        self._held = []
        out.append(w)

    def _join(self, frames: list[AudioFrame]) -> AudioFrame:
        if not frames:
            return AudioFrame.empty(self.sample_rate, self.channels)
        return AudioFrame.concat(frames)
