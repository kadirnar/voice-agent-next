"""WAV-file transport: play a recorded user turn into the agent, record the reply."""

from __future__ import annotations

import asyncio
import math
import os
from collections.abc import AsyncIterator

from ..audio.frame import AudioFormat, AudioFrame
from ..audio.wav import WavWriter, read_wav
from ..utils.clock import now
from .base import Transport

__all__ = ["FileTransport"]


class FileTransport(Transport):
    """Streams ``input_path`` as user audio and writes agent audio to ``output_path``.

    Args:
        input_path: WAV file with the user's speech.
        output_path: where to write the agent's audio (optional).
        realtime: pace the input at real-time speed (recommended for cloud engines).
        frame_duration: size of input frames in seconds.
        trailing_silence: seconds of silence appended after the file so VAD-based
            turn detection can fire; the input then ends after ``hold`` more seconds.
        hold: after the input, keep the connection open (sending silence) until the agent
            has been quiet for this many seconds, so its reply is recorded completely.
        max_wait: upper bound for that waiting phase.
        output_sample_rate: sample rate of the recorded agent audio.
    """

    def __init__(
        self,
        input_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str] | None = None,
        *,
        realtime: bool = True,
        frame_duration: float = 0.02,
        trailing_silence: float = 1.5,
        hold: float = 1.5,
        max_wait: float = 60.0,
        output_sample_rate: int = 24_000,
    ) -> None:
        self._audio = read_wav(input_path).to_mono()
        super().__init__(
            input_format=AudioFormat(self._audio.sample_rate, 1),
            output_format=AudioFormat(output_sample_rate, 1),
        )
        self.output_path = output_path
        self.realtime = realtime
        self.frame_duration = frame_duration
        self.trailing_silence = trailing_silence
        self.hold = hold
        self.max_wait = max_wait
        self._last_write = 0.0
        self._writer: WavWriter | None = None
        self._closed = asyncio.Event()

    async def start(self) -> None:
        if self.output_path is not None and self._writer is None:
            self._writer = WavWriter(self.output_path, self.output_format.sample_rate, 1)
        self.emit("connected")

    async def aclose(self) -> None:
        self._closed.set()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self.emit("disconnected")

    async def audio_input(self) -> AsyncIterator[AudioFrame]:
        audio = self._audio
        rate = audio.sample_rate
        step = max(1, round(self.frame_duration * rate))  # samples per frame
        bps = audio.format.bytes_per_sample
        start = now()
        sent = 0  # samples yielded so far (integer positions: no float drift)

        async def pace(realtime: bool) -> None:
            delay = start + sent / rate - now()
            if realtime and delay > 0:
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0)

        def silence() -> AudioFrame:
            return AudioFrame(bytes(step * bps), rate, 1, now())

        # 1) the recorded user speech, then 2) trailing silence so VAD can end the turn
        total = audio.samples_per_channel
        for pos in range(0, total, step):
            if self._closed.is_set():
                return
            data = audio.data[pos * bps : min(pos + step, total) * bps]
            frame = AudioFrame(data, rate, 1, now())
            yield frame
            sent += frame.samples_per_channel
            await pace(self.realtime)
        for _ in range(math.ceil(self.trailing_silence * rate / step)):
            if self._closed.is_set():
                return
            yield silence()
            sent += step
            await pace(self.realtime)
        # 3) keep sending silence (always in real time) until the agent has been quiet
        #    for `hold` seconds, so its reply is fully written to the output file
        hold_start = now()
        start = hold_start - sent / rate  # re-anchor pacing to the wall clock
        while not self._closed.is_set():
            quiet_since = max(hold_start, self._last_write)
            if now() - quiet_since >= self.hold or now() - hold_start >= self.max_wait:
                break
            yield silence()
            sent += step
            await pace(True)

    async def write_audio(self, frame: AudioFrame) -> None:
        self._last_write = now()
        if self._writer is not None:
            self._writer.write(frame)

    async def clear_audio(self) -> None:
        """Nothing is buffered: audio is written to disk immediately."""
