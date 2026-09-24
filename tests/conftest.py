from __future__ import annotations

import numpy as np
import pytest

from voice_agent_next.audio import AudioFrame


def tone(freq: float, duration: float, sample_rate: int, amplitude: float = 0.5) -> AudioFrame:
    t = np.arange(round(duration * sample_rate)) / sample_rate
    return AudioFrame.from_numpy(
        (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32), sample_rate
    )


def dominant_frequency(frame: AudioFrame) -> float:
    x = frame.to_float32()
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1.0 / frame.sample_rate)
    return float(freqs[int(np.argmax(spectrum))])


@pytest.fixture
def speech_16k() -> AudioFrame:
    from voice_agent_next.providers.mock import synth_speech

    return synth_speech(1.0, 16_000)
