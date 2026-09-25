"""Helpers shared by the Whisper provider tests (faster-whisper, mlx-whisper): a scripted
VAD and a real-time audio pusher for :class:`~voice_agent_next.stt.StreamAdapter` streams."""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any

import numpy as np

from voice_agent_next import AudioFrame, VADOptions
from voice_agent_next.vad import VAD


def chunks(frame: AudioFrame, step: float = 0.02) -> list[AudioFrame]:
    n = math.ceil(frame.duration / step - 1e-9)
    return [frame.slice(i * step, (i + 1) * step) for i in range(n)]


class LevelVAD(VAD):
    """Scripted VAD: ``probability`` on any 20 ms window louder than -40 dBFS, else 0."""

    provider = "level"

    def __init__(
        self, probability: float = 0.95, *, min_silence: float = 0.25, activation: float = 0.5
    ) -> None:
        super().__init__(
            sample_rate=16_000,
            window_samples=320,
            options=VADOptions(
                activation_threshold=activation,
                min_speech_duration=0.06,
                min_silence_duration=min_silence,
            ),
        )
        self.probability = probability

    def _new_inference(self) -> Any:
        probability = self.probability

        class Inference:
            def __call__(self, window: Any) -> float:
                return probability if float(np.sqrt(np.mean(np.square(window)))) > 0.01 else 0.0

            def reset(self) -> None:
                pass

        return Inference()


async def stream_paced(
    stream: Any, audio: AudioFrame, *, speed: float = 2.0, step: float = 0.02
) -> list[tuple[float, Any]]:
    """Push ``audio`` in ``step`` chunks paced at ``speed`` x real time, end the input and
    return ``(audio time, event)`` pairs: the audio pushed when each event arrived."""
    events: list[tuple[float, Any]] = []
    pushed = 0.0

    async def consume() -> None:
        async for ev in stream:
            events.append((pushed, ev))

    consumer = asyncio.create_task(consume())
    t0 = time.perf_counter()
    for i, frame in enumerate(chunks(audio, step)):
        delay = t0 + i * step / speed - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        stream.push_audio(frame)
        pushed += frame.duration
    stream.end_input()
    await consumer
    return events


async def push_settled(stream: Any, audio: AudioFrame, *, timeout: float = 10.0) -> list[Any]:
    """Push ``audio`` in 20 ms frames, letting the stream process each frame and finish
    any interim decode it started before the next one (interim decodes then start at
    deterministic audio times), end the input and return the events."""
    events: list[Any] = []

    async def consume() -> None:
        async for ev in stream:
            events.append(ev)

    consumer = asyncio.create_task(consume())
    deadline = time.monotonic() + timeout
    for frame in chunks(audio):
        stream.push_audio(frame)
        while True:
            await asyncio.sleep(0.001)
            step = stream._step
            if stream._input.empty() and (step is None or step.done()):
                break
            assert time.monotonic() < deadline, "the stream did not settle"
    stream.end_input()
    await consumer
    return events
