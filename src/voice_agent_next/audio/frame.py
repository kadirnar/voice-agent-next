"""Audio primitives: :class:`AudioFormat` and :class:`AudioFrame`.

All audio inside voice-agent-next is **16-bit signed little-endian PCM** ("s16le"),
interleaved when multi-channel. Providers and transports convert at the edges
(μ-law for telephony, float32 for ML models, Opus for WebRTC...).
"""

from __future__ import annotations

import base64
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["SAMPLE_WIDTH", "AudioFormat", "AudioFrame"]

SAMPLE_WIDTH = 2  # bytes per sample per channel (int16)
_I16 = np.dtype("<i2")


@dataclass(frozen=True, slots=True)
class AudioFormat:
    """Sample rate + channel count of an s16le PCM stream."""

    sample_rate: int = 16_000
    channels: int = 1

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {self.sample_rate}")
        if self.channels <= 0:
            raise ValueError(f"channels must be > 0, got {self.channels}")

    @property
    def bytes_per_sample(self) -> int:
        """Bytes for one sample across all channels."""
        return SAMPLE_WIDTH * self.channels

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * self.bytes_per_sample

    def samples(self, duration: float) -> int:
        """Number of samples (per channel) in ``duration`` seconds."""
        return round(duration * self.sample_rate)

    def num_bytes(self, duration: float) -> int:
        """Number of bytes in ``duration`` seconds (always frame-aligned)."""
        return self.samples(duration) * self.bytes_per_sample

    def duration(self, num_bytes: int) -> float:
        return num_bytes / self.bytes_per_second

    def __str__(self) -> str:
        return f"{self.sample_rate}Hz/{self.channels}ch/s16le"


@dataclass(slots=True)
class AudioFrame:
    """A chunk of s16le PCM audio.

    Attributes:
        data: interleaved little-endian int16 samples.
        sample_rate: samples per second (per channel).
        channels: number of interleaved channels.
        timestamp: optional :func:`voice_agent_next.utils.now` time at which the first
            sample was captured/produced. Used for latency metrics.
    """

    data: bytes
    sample_rate: int
    channels: int = 1
    timestamp: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes):
            self.data = bytes(self.data)
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {self.sample_rate}")
        if self.channels <= 0:
            raise ValueError(f"channels must be > 0, got {self.channels}")
        if len(self.data) % (SAMPLE_WIDTH * self.channels):
            raise ValueError(
                f"data length {len(self.data)} is not a multiple of "
                f"{SAMPLE_WIDTH * self.channels} (int16 x {self.channels} channels)"
            )

    # ------------------------------------------------------------------ properties
    @property
    def format(self) -> AudioFormat:
        return AudioFormat(self.sample_rate, self.channels)

    @property
    def samples_per_channel(self) -> int:
        return len(self.data) // (SAMPLE_WIDTH * self.channels)

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.samples_per_channel / self.sample_rate

    @property
    def duration_ms(self) -> float:
        return self.duration * 1000.0

    def __bool__(self) -> bool:  # an empty frame is falsy
        return bool(self.data)

    # ----------------------------------------------------------------- conversions
    def to_numpy(self) -> npt.NDArray[np.int16]:
        """int16 samples: shape ``(n,)`` for mono, ``(n, channels)`` otherwise (read-only view)."""
        arr: npt.NDArray[np.int16] = np.frombuffer(self.data, dtype=_I16)
        if self.channels > 1:
            arr = arr.reshape(-1, self.channels)
        return arr.astype(np.int16, copy=False)

    def to_float32(self) -> npt.NDArray[np.float32]:
        """float32 samples in ``[-1, 1)``, same shape convention as :meth:`to_numpy`."""
        return self.to_numpy().astype(np.float32) / 32768.0

    @classmethod
    def from_numpy(
        cls,
        array: npt.ArrayLike,
        sample_rate: int,
        *,
        channels: int | None = None,
        timestamp: float | None = None,
    ) -> AudioFrame:
        """Build a frame from a numpy array.

        Float arrays are interpreted as ``[-1, 1]`` and clipped; integer arrays are
        clipped to the int16 range. 2-D arrays must be ``(samples, channels)``.
        """
        arr = np.asarray(array)
        if arr.ndim == 2:
            ch = arr.shape[1]
        elif arr.ndim == 1:
            ch = 1
        else:
            raise ValueError(f"expected a 1-D or 2-D array, got shape {arr.shape}")
        if channels is not None and channels != ch:
            if arr.ndim == 1 and arr.size % channels == 0:
                ch = channels  # already interleaved
            else:
                raise ValueError(f"array has {ch} channel(s) but channels={channels}")
        if np.issubdtype(arr.dtype, np.floating):
            pcm = np.clip(np.round(arr * 32767.0), -32768, 32767).astype(_I16)
        elif arr.dtype == np.int16:
            pcm = arr.astype(_I16, copy=False)
        elif np.issubdtype(arr.dtype, np.integer):
            pcm = np.clip(arr, -32768, 32767).astype(_I16)
        else:
            raise TypeError(f"unsupported dtype {arr.dtype}")
        return cls(np.ascontiguousarray(pcm).tobytes(), sample_rate, ch, timestamp)

    @classmethod
    def silence(cls, duration: float, sample_rate: int, channels: int = 1) -> AudioFrame:
        n = round(duration * sample_rate)
        return cls(bytes(n * SAMPLE_WIDTH * channels), sample_rate, channels)

    @classmethod
    def empty(cls, sample_rate: int, channels: int = 1) -> AudioFrame:
        return cls(b"", sample_rate, channels)

    @classmethod
    def concat(cls, frames: Iterable[AudioFrame]) -> AudioFrame:
        """Concatenate frames that share the same format."""
        frames = list(frames)
        if not frames:
            raise ValueError("cannot concatenate an empty list of frames")
        first = frames[0]
        for f in frames[1:]:
            if f.sample_rate != first.sample_rate or f.channels != first.channels:
                raise ValueError(f"format mismatch: {f.format} vs {first.format}")
        return cls(
            b"".join(f.data for f in frames), first.sample_rate, first.channels, first.timestamp
        )

    def to_mono(self) -> AudioFrame:
        """Down-mix to mono by averaging channels."""
        if self.channels == 1:
            return self
        mixed = self.to_numpy().astype(np.int32).mean(axis=1)
        return AudioFrame.from_numpy(
            mixed.astype(np.int16), self.sample_rate, timestamp=self.timestamp
        )

    def to_channels(self, channels: int) -> AudioFrame:
        """Convert to ``channels`` channels (mono is duplicated; others are down-mixed first)."""
        if channels == self.channels:
            return self
        mono = self.to_mono().to_numpy()
        if channels == 1:
            return AudioFrame(mono.tobytes(), self.sample_rate, 1, self.timestamp)
        return AudioFrame.from_numpy(
            np.repeat(mono[:, None], channels, axis=1), self.sample_rate, timestamp=self.timestamp
        )

    def slice(self, start: float = 0.0, end: float | None = None) -> AudioFrame:
        """Sub-frame between ``start`` and ``end`` seconds."""
        bps = SAMPLE_WIDTH * self.channels
        s = max(0, round(start * self.sample_rate)) * bps
        e = len(self.data) if end is None else max(0, round(end * self.sample_rate)) * bps
        ts = None if self.timestamp is None else self.timestamp + start
        return AudioFrame(self.data[s:e], self.sample_rate, self.channels, ts)

    def to_base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    @classmethod
    def from_base64(
        cls, data: str, sample_rate: int, channels: int = 1, *, timestamp: float | None = None
    ) -> AudioFrame:
        return cls(base64.b64decode(data), sample_rate, channels, timestamp)

    # ---------------------------------------------------------------- measurements
    def rms(self) -> float:
        """Root-mean-square level normalized to full scale (0.0 – 1.0)."""
        if not self.data:
            return 0.0
        x = self.to_numpy().astype(np.float64) / 32768.0
        return float(np.sqrt(np.mean(np.square(x))))

    def dbfs(self) -> float:
        """Level in dBFS (``-inf`` for digital silence)."""
        r = self.rms()
        return float("-inf") if r <= 0 else 20.0 * float(np.log10(r))

    def __repr__(self) -> str:
        return (
            f"AudioFrame({self.duration * 1000:.1f}ms, {self.sample_rate}Hz, "
            f"{self.channels}ch, {len(self.data)}B)"
        )
