"""Shared helpers for the benchmark tests."""

from __future__ import annotations

import numpy as np

from voice_agent_next.audio import AudioFrame


def tone(duration: float, rate: int, *, amplitude: float = 0.3, freq: float = 300.0) -> AudioFrame:
    t = np.arange(round(duration * rate)) / rate
    return AudioFrame.from_numpy(
        (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32), rate
    )


def silence(duration: float, rate: int) -> AudioFrame:
    return AudioFrame.silence(duration, rate)


def concat(*frames: AudioFrame) -> AudioFrame:
    return AudioFrame.concat(list(frames))


def mix(a: AudioFrame, b: AudioFrame) -> AudioFrame:
    """Sum two mono frames of the same rate (truncated to the shorter one)."""
    n = min(a.samples_per_channel, b.samples_per_channel)
    x = a.to_numpy()[:n].astype(np.int32) + b.to_numpy()[:n].astype(np.int32)
    return AudioFrame.from_numpy(np.clip(x, -32768, 32767).astype(np.int16), a.sample_rate)
