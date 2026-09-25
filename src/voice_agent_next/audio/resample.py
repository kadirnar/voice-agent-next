"""Streaming sample-rate conversion.

Uses `python-soxr <https://github.com/dofuuz/python-soxr>`_ when installed (``pip install
'voice-agent-next[resample]'``) and otherwise falls back to a pure-numpy polyphase
windowed-sinc resampler, so the core library works everywhere numpy does.

Resamplers are *stateful*: feed consecutive chunks of one stream through the same
instance to avoid clicks at chunk boundaries.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from ..utils.deps import is_installed
from .frame import AudioFrame

__all__ = ["Resampler", "StreamResampler", "resample"]

Quality = Literal["low", "medium", "high"]
Backend = Literal["auto", "soxr", "numpy"]

_SOXR_QUALITY = {"low": "LQ", "medium": "MQ", "high": "HQ"}
_TAPS_PER_PHASE = {"low": 8, "medium": 16, "high": 24}
_SOXR_HISTORY_S = 0.02  # input history re-fed to soxr after a mid-stream drain


def _to_int16(y: npt.NDArray[np.float64]) -> npt.NDArray[np.int16]:
    return np.clip(np.round(y), -32768, 32767).astype(np.int16)


class _PolyphaseResampler:
    """Rational-ratio polyphase FIR resampler (Kaiser-windowed sinc), streaming."""

    def __init__(self, in_rate: int, out_rate: int, channels: int, quality: Quality) -> None:
        g = math.gcd(in_rate, out_rate)
        self.L = out_rate // g  # up-sampling factor
        self.M = in_rate // g  # down-sampling factor
        self.channels = channels
        base = _TAPS_PER_PHASE[quality]
        self.T = T = max(4, math.ceil(base * max(1.0, self.M / self.L)))
        n_taps = T * self.L
        cutoff = 0.5 / max(self.L, self.M) * 0.94  # cycles/sample at the up-sampled rate
        n = np.arange(n_taps) - (n_taps - 1) / 2.0
        h = 2.0 * cutoff * np.sinc(2.0 * cutoff * n) * np.kaiser(n_taps, 8.6)
        h *= self.L / h.sum()  # unity DC gain after zero-stuffing
        # poly[p, j] = h[p + j*L]
        self._poly = h.reshape(T, self.L).T.copy()
        self.group_delay_out = ((n_taps - 1) / 2.0) / self.M  # in output samples
        self._taps = np.arange(T)
        self.reset()

    def reset(self) -> None:
        self._buf = np.zeros((self.T - 1, self.channels), dtype=np.float64)
        self._base = -(self.T - 1)  # global input index of _buf[0]
        self._k = 0  # next output index
        self._pending = False  # input pushed since the last drain

    def process(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        buf = np.concatenate([self._buf, x], axis=0) if x.size else self._buf
        self._pending = self._pending or bool(x.size)
        # ceil(avail_end * L / M); avail_end may be <= 0 right after a start or drain
        k_end = ((self._base + len(buf)) * self.L - 1) // self.M + 1
        return self._emit(buf, k_end)

    def drain(self) -> npt.NDArray[np.float64]:
        """Emit the filter tail of everything pushed so far, *keeping* the history.

        The outputs still owed for the pushed input (the group delay) are computed as
        if the future were silent, but no zeros enter the history: audio pushed next
        continues the same timeline, so there is no inserted gap, no click, and the
        output sample count never drifts. Idempotent until more input arrives.
        """
        if not self._pending:
            return np.zeros((0, self.channels))
        self._pending = False
        # indices are rebased relative to _k, so avail_end may even be negative here
        avail_end = self._base + len(self._buf)
        k_end = (avail_end * self.L - 1) // self.M + 1 + math.ceil(self.group_delay_out)
        return self._emit(self._buf, k_end)

    def _emit(self, buf: npt.NDArray[np.float64], k_end: int) -> npt.NDArray[np.float64]:
        """Compute outputs ``[_k, k_end)`` from ``buf`` (zero-extended when needed)."""
        L, M, T = self.L, self.M, self.T
        out: npt.NDArray[np.float64]
        if k_end > self._k:
            ks = np.arange(self._k, k_end, dtype=np.int64)
            m = ks * M
            n = m // L
            p = m % L
            need = int(n[-1]) - self._base + 1
            src = buf
            if need > len(buf):  # drain: look-ahead past the pushed input reads silence
                src = np.concatenate([buf, np.zeros((need - len(buf), self.channels))], axis=0)
            idx = (n - self._base)[:, None] - self._taps[None, :]
            out = np.einsum("kt,ktc->kc", self._poly[p], src[idx])
            self._k = k_end
        else:
            out = np.zeros((0, self.channels))
        # keep only the (real) history needed for the next output sample
        n_next = (self._k * M) // L
        keep_from = min(max(0, n_next - (T - 1) - self._base), len(buf))
        if keep_from:
            buf = buf[keep_from:]
            self._base += keep_from
        # rebase indices so they never grow unbounded in long sessions
        if self._k >= L:
            q = self._k // L
            self._k -= q * L
            self._base -= q * M
        self._buf = buf
        return out


class Resampler:
    """Stateful resampler for one stream of a fixed input/output format.

    Feed consecutive chunks with :meth:`push`. Use :meth:`drain` to get every output
    sample owed for the audio pushed so far *mid-stream* (e.g. when an STT turn is
    force-finalized) while keeping the filter history, and :meth:`flush` only at the
    true end of a stream (it also resets the state).

    Args:
        input_rate: sample rate of frames passed to :meth:`push`.
        output_rate: desired sample rate.
        channels: channel count (unchanged by resampling).
        quality: ``"low" | "medium" | "high"``.
        backend: ``"auto"`` (soxr if installed), ``"soxr"`` or ``"numpy"``.
    """

    def __init__(
        self,
        input_rate: int,
        output_rate: int,
        channels: int = 1,
        *,
        quality: Quality = "high",
        backend: Backend = "auto",
    ) -> None:
        if input_rate <= 0 or output_rate <= 0:
            raise ValueError("sample rates must be > 0")
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.channels = channels
        self.quality: Quality = quality
        if backend == "auto":
            backend = "soxr" if is_installed("soxr") else "numpy"
        self.backend: Backend = backend
        self._passthrough = input_rate == output_rate
        self._soxr = None
        self._poly: _PolyphaseResampler | None = None
        if not self._passthrough:
            if backend == "soxr":
                self._soxr = self._new_soxr()
                g = math.gcd(input_rate, output_rate)
                self._L, self._M = output_rate // g, input_rate // g
                # soxr has no "drain but keep history": after a drain the stream is
                # re-primed with this much real input history, and the outputs that
                # belong to it are discarded. A whole number of M-sample periods maps
                # to an exact number of output samples.
                self._hist_len = self._M * math.ceil(input_rate * _SOXR_HISTORY_S / self._M)
                self._hist = np.zeros((0, channels), dtype=np.int16)
                self._discard = 0  # outputs still to drop (they belong to the history)
                self._n_in = 0  # input samples since the last flush()
                self._n_out = 0  # output samples since the last flush()
            else:
                self._poly = _PolyphaseResampler(input_rate, output_rate, channels, quality)

    def _new_soxr(self) -> Any:  # soxr.ResampleStream (optional dependency)
        import soxr

        return soxr.ResampleStream(
            self.input_rate,
            self.output_rate,
            self.channels,
            dtype="int16",
            quality=_SOXR_QUALITY[self.quality],
        )

    def _frame(self, y: npt.NDArray[np.int16], timestamp: float | None = None) -> AudioFrame:
        return AudioFrame(y.tobytes(), self.output_rate, self.channels, timestamp)

    def _soxr_chunk(self, x: npt.NDArray[np.int16], *, last: bool) -> npt.NDArray[np.int16]:
        assert self._soxr is not None
        y = self._soxr.resample_chunk(x if self.channels > 1 else x[:, 0], last=last)
        y = np.asarray(y, dtype=np.int16).reshape(-1, self.channels)
        if self._discard:
            drop = min(self._discard, len(y))
            self._discard -= drop
            y = y[drop:]
        return y

    def push(self, frame: AudioFrame) -> AudioFrame:
        """Resample one chunk. The result may be shorter/longer (or empty) due to filter delay."""
        if frame.sample_rate != self.input_rate or frame.channels != self.channels:
            raise ValueError(
                f"expected {self.input_rate}Hz/{self.channels}ch, got {frame.sample_rate}Hz/"
                f"{frame.channels}ch"
            )
        if self._passthrough:
            return frame
        x = frame.to_numpy().reshape(-1, self.channels)
        if self._soxr is not None:
            if len(x):
                self._hist = np.concatenate([self._hist, x])[-(self._hist_len + self._M) :]
                self._n_in += len(x)
            y = self._soxr_chunk(x, last=False)
            self._n_out += len(y)
            return self._frame(y, frame.timestamp)
        assert self._poly is not None
        y64 = self._poly.process(x.astype(np.float64))
        return self._frame(_to_int16(y64), frame.timestamp)

    def drain(self) -> AudioFrame:
        """Emit all output still owed for the audio pushed so far, keeping the stream going.

        Unlike :meth:`flush`, the filter history is preserved, so audio pushed after a
        drain continues seamlessly: no silence is inserted (no click at the boundary)
        and the running output sample count stays exact across any number of drains.
        Calling it again without new input returns an empty frame.
        """
        if self._passthrough:
            return AudioFrame.empty(self.output_rate, self.channels)
        if self._soxr is not None:
            return self._frame(self._soxr_drain(reset=False))
        assert self._poly is not None
        return self._frame(_to_int16(self._poly.drain()))

    def flush(self) -> AudioFrame:
        """Drain the filter tail at the end of a stream and reset internal state."""
        if self._passthrough:
            return AudioFrame.empty(self.output_rate, self.channels)
        if self._soxr is not None:
            return self._frame(self._soxr_drain(reset=True))
        assert self._poly is not None
        y64 = self._poly.drain()
        self._poly.reset()
        return self._frame(_to_int16(y64))

    def _soxr_drain(self, *, reset: bool) -> npt.NDArray[np.int16]:
        assert self._soxr is not None
        # exact output count owed for all input so far: round(n_in * L / M)
        owed = (2 * self._n_in * self._L + self._M) // (2 * self._M) - self._n_out
        if owed <= 0 and not reset:
            return np.zeros((0, self.channels), dtype=np.int16)
        y = self._soxr_chunk(np.zeros((0, self.channels), dtype=np.int16), last=True)
        self._soxr = self._new_soxr()  # a fresh stream: bit-identical to a new Resampler
        self._discard = 0
        # re-segmenting rounds each segment separately; keep the global count exact
        if len(y) > owed:
            y = y[: max(owed, 0)]
        elif len(y) < owed:
            edge = y[-1:] if len(y) else np.zeros((1, self.channels), dtype=np.int16)
            y = np.concatenate([y, np.repeat(edge, owed - len(y), axis=0)])
        if reset:
            self._hist = self._hist[:0]
            self._n_in = self._n_out = 0
            return y
        self._n_out += len(y)
        # Re-prime with real history so the next samples are filtered with true context.
        # The new soxr stream's output grid starts at its first input sample, so that
        # sample must sit on the global grid (an index that is a multiple of M).
        h = min(len(self._hist), self._hist_len + self._n_in % self._M)
        h -= (h - self._n_in) % self._M
        if h > 0:
            start = self._n_in - h  # global input index (multiple of M) of the history
            self._discard = self._n_out - start * self._L // self._M
            self._soxr_chunk(self._hist[len(self._hist) - h :], last=False)  # discarded
        # rebase the counters (whole M periods) so they never grow unbounded
        q = self._n_in // self._M
        self._n_in -= q * self._M
        self._n_out -= q * self._L
        return y


class StreamResampler:
    """Converts a stream of frames of *any* format to a fixed output format.

    Handles channel conversion and lazily (re)creates the inner :class:`Resampler`
    when the input sample rate changes; the previous rate's filter tail is emitted
    first (prepended to the returned frame), so audio never comes out of order.
    Handy at component boundaries.
    """

    def __init__(self, output_rate: int, output_channels: int = 1, *, quality: Quality = "high"):
        self.output_rate = output_rate
        self.output_channels = output_channels
        self.quality: Quality = quality
        self._rs: Resampler | None = None

    def push(self, frame: AudioFrame) -> AudioFrame:
        frame = frame.to_channels(self.output_channels)
        tail = b""
        if self._rs is not None and self._rs.input_rate != frame.sample_rate:
            # rate change: finish the old stream *before* the new audio, then drop it
            tail = self._rs.flush().data
            self._rs = None
        if frame.sample_rate == self.output_rate:
            out = frame
        else:
            if self._rs is None:
                self._rs = Resampler(
                    frame.sample_rate, self.output_rate, self.output_channels, quality=self.quality
                )
            out = self._rs.push(frame)
        if not tail:
            return out
        return AudioFrame(tail + out.data, self.output_rate, self.output_channels, frame.timestamp)

    def drain(self) -> AudioFrame:
        """Mid-stream: emit the pending tail, keep the history (see :meth:`Resampler.drain`)."""
        if self._rs is None:
            return AudioFrame.empty(self.output_rate, self.output_channels)
        return self._rs.drain()

    def flush(self) -> AudioFrame:
        """End of stream: emit the filter tail and reset."""
        if self._rs is None:
            return AudioFrame.empty(self.output_rate, self.output_channels)
        return self._rs.flush()


def resample(
    frame: AudioFrame, output_rate: int, *, quality: Quality = "high", backend: Backend = "auto"
) -> AudioFrame:
    """One-shot resampling of a complete clip, compensating for filter delay."""
    if frame.sample_rate == output_rate:
        return frame
    if not frame.data:
        return AudioFrame.empty(output_rate, frame.channels)
    if backend == "auto":
        backend = "soxr" if is_installed("soxr") else "numpy"
    expected = math.ceil(frame.samples_per_channel * output_rate / frame.sample_rate)
    x = frame.to_numpy().reshape(-1, frame.channels)
    if backend == "soxr":
        import soxr

        y = soxr.resample(
            x if frame.channels > 1 else x[:, 0],
            frame.sample_rate,
            output_rate,
            quality=_SOXR_QUALITY[quality],
        )
        return AudioFrame(
            np.asarray(y, dtype=np.int16).tobytes(), output_rate, frame.channels, frame.timestamp
        )
    poly = _PolyphaseResampler(frame.sample_rate, output_rate, frame.channels, quality)
    delay = round(poly.group_delay_out)
    pad = math.ceil((delay + 2) * poly.M / poly.L) + poly.T
    y64 = np.concatenate(
        [poly.process(x.astype(np.float64)), poly.process(np.zeros((pad, frame.channels)))]
    )
    y64 = y64[delay : delay + expected]
    return AudioFrame(_to_int16(y64).tobytes(), output_rate, frame.channels, frame.timestamp)
