"""In-memory transport for tests, simulations and benchmarks.

The *agent side* is a normal :class:`~voice_agent_next.transports.base.Transport`;
the *user side* is driven by the test/simulator through ``push_user_audio`` /
``agent_audio``. With ``realtime_playout=True`` a virtual speaker consumes agent
audio at real-time speed, so ``buffered_duration`` and interruption truncation
behave like a real device.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ..audio.frame import AudioFormat, AudioFrame
from ..utils.aio import Chan, cancel_and_wait
from ..utils.clock import now
from .base import Transport, TransportCapabilities

__all__ = ["LoopbackTransport", "PlayedAudio"]


@dataclass(slots=True)
class PlayedAudio:
    """A chunk of agent audio as heard by the simulated user."""

    frame: AudioFrame
    start_time: float
    """:func:`~voice_agent_next.utils.now` when playback of this chunk started."""


class LoopbackTransport(Transport):
    capabilities = TransportCapabilities(playback_position=True, messages=True)

    def __init__(
        self,
        *,
        input_format: AudioFormat | None = None,
        output_format: AudioFormat | None = None,
        realtime_playout: bool = False,
    ) -> None:
        super().__init__(
            input_format=input_format or AudioFormat(16_000, 1),
            output_format=output_format or AudioFormat(24_000, 1),
        )
        self.realtime_playout = realtime_playout
        self._user_audio: Chan[AudioFrame] = Chan()
        self._played: Chan[PlayedAudio] = Chan()
        self._queue: deque[AudioFrame] = deque()
        self._queued_duration = 0.0
        self._current_end: float | None = None
        self._wakeup = asyncio.Event()
        self._player: asyncio.Task[None] | None = None
        self._cleared = 0
        self.messages: list[dict[str, Any]] = []
        self.played_log: list[PlayedAudio] = []
        self.clear_times: list[float] = []

    # ------------------------------------------------------------ agent side API
    async def start(self) -> None:
        if self.realtime_playout and self._player is None:
            self._player = asyncio.create_task(self._play_loop(), name="loopback-player")
        self.emit("connected")

    async def aclose(self) -> None:
        self._user_audio.close()
        self._played.close()
        await cancel_and_wait(self._player)
        self._player = None
        self.emit("disconnected")

    def audio_input(self) -> AsyncIterator[AudioFrame]:
        return self._user_audio.__aiter__()

    async def write_audio(self, frame: AudioFrame) -> None:
        if frame.sample_rate != self.output_format.sample_rate:
            raise ValueError(f"expected {self.output_format}, got {frame.format}")
        if not frame:
            return
        if not self.realtime_playout:
            self._deliver(frame, now())
            return
        self._queue.append(frame)
        self._queued_duration += frame.duration
        self._wakeup.set()

    async def clear_audio(self) -> None:
        self._queue.clear()
        self._queued_duration = 0.0
        self._current_end = None
        self._cleared += 1
        self.clear_times.append(now())
        self._wakeup.set()

    def buffered_duration(self) -> float:
        current = 0.0
        if self._current_end is not None:
            current = max(0.0, self._current_end - now())
        return self._queued_duration + current

    async def send_message(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    # ------------------------------------------------------------- user side API
    def push_user_audio(self, frame: AudioFrame) -> None:
        """Deliver user audio to the agent (the caller controls pacing)."""
        self._user_audio.send_nowait(frame)

    async def play_user_audio(
        self, frame: AudioFrame, *, chunk: float = 0.02, realtime: bool = True
    ) -> None:
        """Push ``frame`` in ``chunk``-second pieces, paced at real time if ``realtime``."""
        n = max(1, round(chunk * frame.sample_rate))
        step = n * frame.channels * 2
        start = now()
        sent = 0.0
        for i in range(0, len(frame.data), step):
            piece = AudioFrame(frame.data[i : i + step], frame.sample_rate, frame.channels, now())
            self.push_user_audio(piece)
            sent += piece.duration
            if realtime:
                delay = start + sent - now()
                if delay > 0:
                    await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0)

    def end_user_audio(self) -> None:
        """The user hung up: ``audio_input()`` ends."""
        self._user_audio.close()

    def agent_audio(self) -> AsyncIterator[PlayedAudio]:
        """Agent audio as it is played (in real time when ``realtime_playout``)."""
        return self._played.__aiter__()

    # ------------------------------------------------------------------ internals
    def _deliver(self, frame: AudioFrame, start: float) -> None:
        played = PlayedAudio(frame, start)
        self.played_log.append(played)
        if not self._played.closed:
            self._played.send_nowait(played)

    async def _play_loop(self) -> None:
        while True:
            if not self._queue:
                self._wakeup.clear()
                await self._wakeup.wait()
                continue
            frame = self._queue.popleft()
            self._queued_duration = max(0.0, self._queued_duration - frame.duration)
            cleared_at_start = self._cleared
            start = now()
            self._current_end = start + frame.duration
            self._deliver(frame, start)
            # sleep for the frame duration unless a clear() interrupts playback
            while self._cleared == cleared_at_start:
                remaining = (self._current_end or 0) - now()
                if remaining <= 0:
                    break
                self._wakeup.clear()
                try:
                    await asyncio.wait_for(self._wakeup.wait(), remaining)
                except TimeoutError:
                    break
            self._current_end = None
