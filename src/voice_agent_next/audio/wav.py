"""WAV file I/O based on the standard-library :mod:`wave` module (PCM only)."""

from __future__ import annotations

import io
import os
import wave
from collections.abc import Iterable
from typing import BinaryIO

import numpy as np

from .frame import AudioFrame

__all__ = ["WavWriter", "read_wav", "wav_bytes", "write_wav"]

PathOrFile = str | os.PathLike[str] | BinaryIO


def _target(target: PathOrFile) -> str | BinaryIO:
    """``wave.open`` accepts only str paths or file objects (not ``PathLike``)."""
    return os.fspath(target) if isinstance(target, os.PathLike) else target


def read_wav(source: PathOrFile | bytes) -> AudioFrame:
    """Read an integer-PCM WAV file (8/16/24/32-bit) and return s16le audio."""
    src = io.BytesIO(source) if isinstance(source, bytes) else _target(source)
    with wave.open(src, "rb") as w:
        channels = w.getnchannels()
        rate = w.getframerate()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 2:
        data = raw
    elif width == 1:  # unsigned 8-bit
        data = ((np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128) << 8).tobytes()
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        i32 = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        i32 = np.where(i32 & 0x800000, i32 - 0x1000000, i32)
        data = (i32 >> 8).astype("<i2").tobytes()
    elif width == 4:
        data = (np.frombuffer(raw, dtype="<i4") >> 16).astype("<i2").tobytes()
    else:
        raise ValueError(f"unsupported WAV sample width: {width} bytes")
    return AudioFrame(data, rate, channels)


def write_wav(target: PathOrFile, audio: AudioFrame | Iterable[AudioFrame]) -> None:
    """Write s16le audio (one frame or an iterable of same-format frames) to a WAV file."""
    frames = [audio] if isinstance(audio, AudioFrame) else list(audio)
    if not frames:
        raise ValueError("no audio to write")
    first = frames[0]
    with wave.open(_target(target), "wb") as w:
        w.setnchannels(first.channels)
        w.setsampwidth(2)
        w.setframerate(first.sample_rate)
        for f in frames:
            if f.sample_rate != first.sample_rate or f.channels != first.channels:
                raise ValueError("all frames must share the same format")
            w.writeframes(f.data)


def wav_bytes(audio: AudioFrame | Iterable[AudioFrame]) -> bytes:
    """Encode audio as an in-memory WAV file."""
    buf = io.BytesIO()
    write_wav(buf, audio)
    return buf.getvalue()


class WavWriter:
    """Incremental WAV writer (e.g. for session recordings)."""

    def __init__(self, target: PathOrFile, sample_rate: int, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._w = wave.open(_target(target), "wb")  # noqa: SIM115
        self._w.setnchannels(channels)
        self._w.setsampwidth(2)
        self._w.setframerate(sample_rate)
        self._closed = False

    def write(self, frame: AudioFrame) -> None:
        if frame.sample_rate != self.sample_rate or frame.channels != self.channels:
            raise ValueError(f"expected {self.sample_rate}Hz/{self.channels}ch, got {frame.format}")
        self._w.writeframes(frame.data)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._w.close()

    def __enter__(self) -> WavWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
