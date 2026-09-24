"""Audio accumulation and re-chunking helpers."""

from __future__ import annotations

from collections.abc import Iterable

from .frame import SAMPLE_WIDTH, AudioFormat, AudioFrame

__all__ = ["AudioBuffer", "FrameChunker"]


class AudioBuffer:
    """Accumulates PCM audio of a single format, optionally bounded to ``max_duration``.

    When bounded, the *oldest* audio is dropped (ring-buffer semantics), which is
    what turn detectors need ("the last 8 seconds of user speech").
    """

    def __init__(self, sample_rate: int, channels: int = 1, *, max_duration: float | None = None):
        self.format = AudioFormat(sample_rate, channels)
        self._data = bytearray()
        self._max_bytes = None if max_duration is None else self.format.num_bytes(max_duration)
        self._start_timestamp: float | None = None

    @property
    def sample_rate(self) -> int:
        return self.format.sample_rate

    @property
    def channels(self) -> int:
        return self.format.channels

    @property
    def duration(self) -> float:
        return self.format.duration(len(self._data))

    def __len__(self) -> int:
        return len(self._data)

    def __bool__(self) -> bool:
        return bool(self._data)

    def append(self, frame: AudioFrame) -> None:
        if frame.sample_rate != self.sample_rate or frame.channels != self.channels:
            raise ValueError(f"frame format {frame.format} does not match buffer {self.format}")
        if not self._data:
            self._start_timestamp = frame.timestamp
        self._data += frame.data
        if self._max_bytes is not None and len(self._data) > self._max_bytes:
            drop = len(self._data) - self._max_bytes
            del self._data[:drop]
            if self._start_timestamp is not None:
                self._start_timestamp += self.format.duration(drop)

    def extend(self, frames: Iterable[AudioFrame]) -> None:
        for f in frames:
            self.append(f)

    def to_frame(self) -> AudioFrame:
        return AudioFrame(bytes(self._data), self.sample_rate, self.channels, self._start_timestamp)

    def pop_all(self) -> AudioFrame:
        frame = self.to_frame()
        self.clear()
        return frame

    def clear(self) -> None:
        self._data.clear()
        self._start_timestamp = None

    def keep_last(self, duration: float) -> None:
        """Drop everything except the most recent ``duration`` seconds."""
        n = self.format.num_bytes(duration)
        if len(self._data) > n:
            drop = len(self._data) - n
            del self._data[:drop]
            if self._start_timestamp is not None:
                self._start_timestamp += self.format.duration(drop)


class FrameChunker:
    """Re-chunks arbitrary-size frames into fixed-size frames.

    VAD models need exact window sizes (e.g. 512 samples @16 kHz for Silero), and
    telephony transports want 20 ms frames. Timestamps are propagated.

    Example:
        >>> chunker = FrameChunker(16000, samples_per_frame=512)
        >>> for frame in chunker.push(incoming):  # 0..n full frames
        ...     process(frame)
    """

    def __init__(
        self,
        sample_rate: int,
        channels: int = 1,
        *,
        samples_per_frame: int | None = None,
        frame_duration: float | None = None,
    ) -> None:
        if (samples_per_frame is None) == (frame_duration is None):
            raise ValueError("pass exactly one of samples_per_frame / frame_duration")
        if samples_per_frame is None:
            assert frame_duration is not None
            samples_per_frame = round(frame_duration * sample_rate)
        if samples_per_frame <= 0:
            raise ValueError("frame size must be > 0")
        self.sample_rate = sample_rate
        self.channels = channels
        self.samples_per_frame = samples_per_frame
        self._frame_bytes = samples_per_frame * SAMPLE_WIDTH * channels
        self._buf = bytearray()
        self._ts: float | None = None

    @property
    def buffered_samples(self) -> int:
        return len(self._buf) // (SAMPLE_WIDTH * self.channels)

    def push(self, frame: AudioFrame) -> list[AudioFrame]:
        if frame.sample_rate != self.sample_rate or frame.channels != self.channels:
            raise ValueError(
                f"frame format {frame.format} does not match chunker "
                f"{AudioFormat(self.sample_rate, self.channels)}"
            )
        if not self._buf:
            self._ts = frame.timestamp
        self._buf += frame.data
        out: list[AudioFrame] = []
        while len(self._buf) >= self._frame_bytes:
            chunk = bytes(self._buf[: self._frame_bytes])
            del self._buf[: self._frame_bytes]
            out.append(AudioFrame(chunk, self.sample_rate, self.channels, self._ts))
            if self._ts is not None:
                self._ts += self.samples_per_frame / self.sample_rate
        return out

    def flush(self, *, pad: bool = False) -> list[AudioFrame]:
        """Return the remaining partial frame (zero-padded to full size if ``pad``)."""
        if not self._buf:
            return []
        data = bytes(self._buf)
        if pad:
            data += bytes(self._frame_bytes - len(data))
        self._buf.clear()
        frame = AudioFrame(data, self.sample_rate, self.channels, self._ts)
        self._ts = None
        return [frame]

    def reset(self) -> None:
        self._buf.clear()
        self._ts = None
