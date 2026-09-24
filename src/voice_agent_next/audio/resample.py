"""Streaming sample-rate conversion.

Uses `python-soxr <https://github.com/dofuuz/python-soxr>`_ when installed (``pip install
'voice-agent-next[resample]'``) and otherwise falls back to a pure-numpy polyphase
windowed-sinc resampler, so the core library works everywhere numpy does.

Resamplers are *stateful*: feed consecutive chunks of one stream through the same
instance to avoid clicks at chunk boundaries.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import numpy.typing as npt

from ..utils.deps import is_installed
from .frame import AudioFrame

__all__ = ["Resampler", "StreamResampler", "resample"]

Quality = Literal["low", "medium", "high"]
Backend = Literal["auto", "soxr", "numpy"]

_SOXR_QUALITY = {"low": "LQ", "medium": "MQ", "high": "HQ"}
_TAPS_PER_PHASE = {"low": 8, "medium": 16, "high": 24}


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

    def process(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        L, M, T = self.L, self.M, self.T
        buf = np.concatenate([self._buf, x], axis=0) if x.size else self._buf
        avail_end = self._base + len(buf)
        out: npt.NDArray[np.float64]
        if avail_end <= 0:
            out = np.zeros((0, self.channels))
        else:
            k_end = (avail_end * L - 1) // M + 1
            if k_end > self._k:
                ks = np.arange(self._k, k_end, dtype=np.int64)
                m = ks * M
                n = m // L
                p = m % L
                idx = (n - self._base)[:, None] - self._taps[None, :]
                out = np.einsum("kt,ktc->kc", self._poly[p], buf[idx])
                self._k = k_end
            else:
                out = np.zeros((0, self.channels))
        # keep only the history needed for the next output sample
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
                import soxr

                self._soxr = soxr.ResampleStream(
                    input_rate, output_rate, channels, dtype="int16", quality=_SOXR_QUALITY[quality]
                )
            else:
                self._poly = _PolyphaseResampler(input_rate, output_rate, channels, quality)

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
            y = self._soxr.resample_chunk(x if self.channels > 1 else x[:, 0], last=False)
            return AudioFrame(
                np.asarray(y, dtype=np.int16).tobytes(),
                self.output_rate,
                self.channels,
                frame.timestamp,
            )
        assert self._poly is not None
        y64 = self._poly.process(x.astype(np.float64))
        return AudioFrame(
            _to_int16(y64).tobytes(), self.output_rate, self.channels, frame.timestamp
        )

    def flush(self) -> AudioFrame:
        """Drain the filter tail at the end of a stream and reset internal state."""
        if self._passthrough:
            return AudioFrame.empty(self.output_rate, self.channels)
        if self._soxr is not None:
            empty = np.zeros((0, self.channels) if self.channels > 1 else (0,), dtype=np.int16)
            y = self._soxr.resample_chunk(empty, last=True)
            self._soxr.clear()
            return AudioFrame(
                np.asarray(y, dtype=np.int16).tobytes(), self.output_rate, self.channels
            )
        assert self._poly is not None
        pad = (
            math.ceil((self._poly.group_delay_out + 1) * self._poly.M / self._poly.L) + self._poly.T
        )
        y64 = self._poly.process(np.zeros((pad, self.channels)))
        y64 = y64[: math.ceil(self._poly.group_delay_out)]
        self._poly.reset()
        return AudioFrame(_to_int16(y64).tobytes(), self.output_rate, self.channels)


class StreamResampler:
    """Converts a stream of frames of *any* format to a fixed output format.

    Handles channel conversion and lazily (re)creates the inner :class:`Resampler`
    when the input sample rate changes. Handy at component boundaries.
    """

    def __init__(self, output_rate: int, output_channels: int = 1, *, quality: Quality = "high"):
        self.output_rate = output_rate
        self.output_channels = output_channels
        self.quality: Quality = quality
        self._rs: Resampler | None = None

    def push(self, frame: AudioFrame) -> AudioFrame:
        frame = frame.to_channels(self.output_channels)
        if frame.sample_rate == self.output_rate:
            return frame
        if self._rs is None or self._rs.input_rate != frame.sample_rate:
            self._rs = Resampler(
                frame.sample_rate, self.output_rate, self.output_channels, quality=self.quality
            )
        return self._rs.push(frame)

    def flush(self) -> AudioFrame:
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
