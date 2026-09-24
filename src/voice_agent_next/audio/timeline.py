"""Mono audio assembled from samples placed at absolute positions on a timeline.

:class:`TimelineTrack` is the building block of two-channel call recordings: the
benchmark's in-memory :class:`~voice_agent_next.bench.recording.DuplexRecording` and the
session's streaming :class:`~voice_agent_next.session.recording.SessionRecorder`. Audio
is written where it was spoken or played (later writes overwrite earlier ones); gaps are
silence. The head of the track can be popped once it is final, so a recorder streaming to
disk only keeps the last few seconds in memory.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["TimelineTrack"]


class TimelineTrack:
    """Mono int16 samples on a sample-index timeline (index 0 = t = 0).

    Positions before :attr:`start` (already popped, or negative) are dropped on write.

    Args:
        sample_rate: samples per second (only used by :meth:`position`).
    """

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._buf: npt.NDArray[np.int16] = np.zeros(0, dtype=np.int16)
        self._head = 0  # index in _buf of sample `start`
        self._start = 0
        self._end = 0

    @property
    def start(self) -> int:
        """First sample still held (everything before it was popped)."""
        return self._start

    @property
    def end(self) -> int:
        """One past the last written sample (at least :attr:`start`)."""
        return self._end

    def position(self, seconds: float) -> int:
        """Sample index of a time offset."""
        return round(seconds * self.sample_rate)

    # ----------------------------------------------------------------- writing
    def write(self, pos: int, samples: npt.NDArray[np.int16]) -> None:
        """Place ``samples`` starting at sample ``pos`` (overwrites what is there)."""
        if pos < self._start:
            samples = samples[self._start - pos :]
            pos = self._start
        n = len(samples)
        if n == 0:
            return
        i = self._reserve(pos + n)
        self._buf[i + pos - self._start : i + pos - self._start + n] = samples
        self._end = max(self._end, pos + n)

    def insert_silence(self, pos: int, n: int) -> None:
        """Shift everything at or after ``pos`` later by ``n`` samples (e.g. a pause)."""
        pos = max(pos, self._start)
        if n <= 0 or pos >= self._end:
            return
        tail = self.read_range(pos, self._end).copy()
        self.truncate(pos)
        self.write(pos + n, tail)

    def truncate(self, pos: int) -> None:
        """Drop everything at or after ``pos``."""
        pos = max(pos, self._start)
        if pos >= self._end:
            return
        i = self._head + pos - self._start
        self._buf[i : self._head + self._end - self._start] = 0
        self._end = pos

    # ----------------------------------------------------------------- reading
    def read_range(self, start: int, stop: int) -> npt.NDArray[np.int16]:
        """Samples ``[start, stop)`` (silence where nothing was written; ``start`` is
        clamped to :attr:`start`)."""
        start = max(start, self._start)
        out = np.zeros(max(0, stop - start), dtype=np.int16)
        hi = min(stop, self._end)
        if hi > start:
            i = self._head + start - self._start
            out[: hi - start] = self._buf[i : i + hi - start]
        return out

    def read(self, n: int | None = None) -> npt.NDArray[np.int16]:
        """``n`` samples from :attr:`start` (default: up to :attr:`end`), without popping."""
        stop = self._end if n is None else self._start + n
        return self.read_range(self._start, stop)

    def pop(self, n: int) -> npt.NDArray[np.int16]:
        """Remove and return the first ``n`` samples (silence past :attr:`end`)."""
        if n <= 0:
            return np.zeros(0, dtype=np.int16)
        out = self.read(n)
        live = self._end - self._start
        if n >= live:  # nothing left: the whole buffer is free again
            self._buf[self._head : self._head + live] = 0
            self._head = 0
            self._start += n
            self._end = self._start
            return out
        self._buf[self._head : self._head + n] = 0  # keep free slots silent
        self._head += n
        self._start += n
        if self._head > len(self._buf) // 2:
            self._compact()
        return out

    # --------------------------------------------------------------- internals
    def _reserve(self, stop: int) -> int:
        """Make room for samples up to ``stop``; return the buffer index of :attr:`start`."""
        need = self._head + stop - self._start
        if need > len(self._buf):
            if self._head:
                self._compact()
                need = stop - self._start
            if need > len(self._buf):
                grown = np.zeros(max(need, 2 * len(self._buf), 4096), dtype=np.int16)
                grown[: len(self._buf)] = self._buf
                self._buf = grown
        return self._head

    def _compact(self) -> None:
        live = self._end - self._start
        self._buf[:live] = self._buf[self._head : self._head + live].copy()
        self._buf[live:] = 0
        self._head = 0
