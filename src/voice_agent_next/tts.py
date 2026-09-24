"""Text-to-speech (TTS) component interface.

Implementing a provider:

* every TTS implements :meth:`TTS._synthesize`, returning a :class:`ChunkedStream`
  subclass whose :meth:`ChunkedStream._run` calls ``self._push_audio(...)``;
* TTS engines that accept *incremental text over one connection* (e.g. WebSocket
  input streaming) also set ``capabilities.streaming=True`` and override
  :meth:`TTS._create_stream` with a :class:`SynthesizeStream` subclass.

Everything else gets :meth:`TTS.stream` for free through :class:`SentenceStreamAdapter`,
which splits streamed text into sentences and synthesizes them one by one.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import ClassVar

from .audio.frame import SAMPLE_WIDTH, AudioFrame
from .audio.silence import SilenceTrimmer
from .metrics import TTSMetrics
from .stt import WordTiming
from .text.filters import tts_clean
from .text.sentences import SentenceSegmenter
from .utils.aio import Chan, ChanClosed, cancel_and_wait
from .utils.clock import now
from .utils.emitter import EventEmitter
from .utils.ids import new_id
from .utils.log import logger

__all__ = [
    "TTS",
    "ChunkedStream",
    "SentenceStreamAdapter",
    "SynthesizeStream",
    "SynthesizedAudio",
    "TTSCapabilities",
]


@dataclass(frozen=True, slots=True)
class TTSCapabilities:
    streaming: bool = False
    """Accepts incremental text input over one connection (native :meth:`TTS.stream`)."""
    word_timestamps: bool = False


@dataclass(slots=True)
class SynthesizedAudio:
    frame: AudioFrame
    request_id: str
    segment_id: str = ""
    is_final: bool = False
    """Last chunk of the segment (the frame may be empty)."""
    text: str | None = None
    """Text of the segment this audio belongs to (set on the segment's first chunk)."""
    words: list[WordTiming] | None = None
    """Word timings in seconds relative to the start of this stream's audio (items may
    carry words without audio)."""
    timestamp: float = field(default_factory=now)


class _Flush:
    __slots__ = ()


_FLUSH = _Flush()


class TTS(ABC, EventEmitter):
    """Base class for speech synthesizers. Emits ``"metrics"`` (:class:`TTSMetrics`)."""

    provider: ClassVar[str] = "unknown"

    def __init__(
        self,
        *,
        model: str,
        sample_rate: int,
        channels: int = 1,
        capabilities: TTSCapabilities | None = None,
        voice: str | None = None,
        clean_text: bool = True,
        trim_silence: bool = True,
    ) -> None:
        EventEmitter.__init__(self)
        self.model = model
        self.sample_rate = sample_rate
        self.channels = channels
        self.capabilities = capabilities or TTSCapabilities()
        self.voice = voice
        self.clean_text = clean_text
        self.trim_silence = trim_silence
        """Trim per-sentence leading/trailing silence when streaming via
        :class:`SentenceStreamAdapter` (many models pad every utterance)."""

    def synthesize(self, text: str, *, voice: str | None = None) -> ChunkedStream:
        """Synthesize a complete text. Iterate the result for streamed audio chunks."""
        if self.clean_text:
            text = tts_clean(text)
        return self._synthesize(text, voice=voice or self.voice)

    @abstractmethod
    def _synthesize(self, text: str, *, voice: str | None) -> ChunkedStream:
        """Return a :class:`ChunkedStream` producing audio for ``text``."""

    def stream(self, *, voice: str | None = None) -> SynthesizeStream:
        """Open an incremental-text synthesis stream."""
        if self.capabilities.streaming:
            return self._create_stream(voice=voice or self.voice)
        return SentenceStreamAdapter(self, voice=voice or self.voice)

    def _create_stream(self, *, voice: str | None) -> SynthesizeStream:
        raise NotImplementedError

    async def warmup(self) -> None:
        """Load models / open connections ahead of the first request (optional)."""

    async def aclose(self) -> None:
        """Release resources."""

    async def __aenter__(self) -> TTS:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


class _AudioEmitter:
    """Shared helpers: bytes -> frames, TTFB bookkeeping, metrics."""

    def __init__(self, tts: TTS, streamed: bool) -> None:
        self._tts = tts
        self._streamed = streamed
        self._events: Chan[SynthesizedAudio] = Chan()
        self._request_id = new_id("tts_")
        self._segment_id = new_id("seg_")
        self._segment_text: str | None = None
        self._first_text_time: float | None = None
        self._first_audio_time: float | None = None
        self._start_time = now()
        self._audio_duration = 0.0
        self._characters = 0
        self._remainder = b""
        self._error: BaseException | None = None
        self._cancelled = False

    def _push_audio(
        self, data: bytes | AudioFrame, *, words: list[WordTiming] | None = None
    ) -> None:
        """Emit audio. Raw ``bytes`` must be s16le at ``tts.sample_rate`` / ``tts.channels``."""
        if isinstance(data, AudioFrame):
            if data.sample_rate != self._tts.sample_rate or data.channels != self._tts.channels:
                raise ValueError(
                    f"{type(self._tts).__name__} produced {data.format}, declared "
                    f"{self._tts.sample_rate}Hz/{self._tts.channels}ch"
                )
            payload = data.data
        else:
            payload = data
        payload = self._remainder + payload
        align = SAMPLE_WIDTH * self._tts.channels
        cut = len(payload) - (len(payload) % align)
        payload, self._remainder = payload[:cut], payload[cut:]
        if not payload:
            return
        t = now()
        if self._first_audio_time is None:
            self._first_audio_time = t
        frame = AudioFrame(payload, self._tts.sample_rate, self._tts.channels, t)
        self._audio_duration += frame.duration
        text, self._segment_text = self._segment_text, None
        self._send(SynthesizedAudio(frame, self._request_id, self._segment_id, False, text, words))

    def _end_segment(self) -> None:
        """Mark the end of the current segment (emits an empty ``is_final`` chunk)."""
        empty = AudioFrame.empty(self._tts.sample_rate, self._tts.channels)
        self._send(
            SynthesizedAudio(empty, self._request_id, self._segment_id, True, self._segment_text)
        )
        self._segment_id = new_id("seg_")
        self._segment_text = None

    def _send(self, item: SynthesizedAudio) -> None:
        if not self._events.closed:
            self._events.send_nowait(item)

    def _emit_metrics(self) -> None:
        ttfb = None
        if self._first_audio_time is not None:
            ttfb = self._first_audio_time - (self._first_text_time or self._start_time)
        self._tts.emit(
            "metrics",
            TTSMetrics(
                provider=self._tts.provider,
                model=self._tts.model,
                request_id=self._request_id,
                ttfb=ttfb,
                duration=now() - self._start_time,
                audio_duration=self._audio_duration,
                characters=self._characters,
                streamed=self._streamed,
                cancelled=self._cancelled,
                error=None if self._error is None else repr(self._error),
            ),
        )

    async def _recv(self) -> SynthesizedAudio:
        try:
            return await self._events.recv()
        except ChanClosed:
            if self._error is not None:
                err, self._error = self._error, None
                raise err from None
            raise StopAsyncIteration from None


class ChunkedStream(_AudioEmitter, ABC):
    """Synthesis of one complete text. Async-iterate to receive :class:`SynthesizedAudio`."""

    def __init__(self, tts: TTS, text: str, *, voice: str | None) -> None:
        super().__init__(tts, streamed=False)
        self.text = text
        self.voice = voice
        self._characters = len(text)
        self._segment_text = text
        self._task = asyncio.create_task(self._main(), name=f"{type(self).__name__}._main")

    @abstractmethod
    async def _run(self) -> None:
        """Produce audio for ``self.text`` via ``self._push_audio(...)``."""

    async def _main(self) -> None:
        try:
            await self._run()
            if self._remainder:
                self._remainder = b""
            self._end_segment()
        except asyncio.CancelledError:
            self._cancelled = True
            raise
        except Exception as exc:
            logger.exception("%s failed", type(self).__name__)
            self._error = exc
        finally:
            self._events.close()
            self._emit_metrics()

    async def collect(self) -> AudioFrame:
        """Wait for the whole synthesis and return it as one frame."""
        frames = [a.frame async for a in self if a.frame]
        if not frames:
            return AudioFrame.empty(self._tts.sample_rate, self._tts.channels)
        return AudioFrame.concat(frames)

    async def aclose(self) -> None:
        await cancel_and_wait(self._task)
        self._events.close()

    def __aiter__(self) -> AsyncIterator[SynthesizedAudio]:
        return self

    async def __anext__(self) -> SynthesizedAudio:
        return await self._recv()


class SynthesizeStream(_AudioEmitter, ABC):
    """Incremental synthesis: :meth:`push_text` deltas, :meth:`flush` segments.

    Subclasses implement :meth:`_run`, consuming ``self._input`` (``str`` deltas and
    flush markers — test with :meth:`is_flush`) and calling ``self._push_audio`` and
    ``self._end_segment`` (once per flushed segment).
    """

    def __init__(self, tts: TTS, *, voice: str | None) -> None:
        super().__init__(tts, streamed=True)
        self.voice = voice
        self._input: Chan[str | _Flush] = Chan()
        self._task = asyncio.create_task(self._main(), name=f"{type(self).__name__}._main")

    @staticmethod
    def is_flush(item: object) -> bool:
        return item is _FLUSH

    def push_text(self, text: str) -> None:
        if self._input.closed:
            raise RuntimeError("push_text() after end_input()/aclose()")
        if not text:
            return
        if self._first_text_time is None:
            self._first_text_time = now()
        self._characters += len(text)
        self._input.send_nowait(text)

    def flush(self) -> None:
        """End the current segment: synthesize everything pushed so far without waiting."""
        if not self._input.closed:
            self._input.send_nowait(_FLUSH)

    def end_input(self) -> None:
        self.flush()
        self._input.close()

    @abstractmethod
    async def _run(self) -> None:
        """Consume ``self._input`` until closed."""

    async def _main(self) -> None:
        try:
            await self._run()
        except asyncio.CancelledError:
            self._cancelled = True
            raise
        except Exception as exc:
            logger.exception("%s failed", type(self).__name__)
            self._error = exc
        finally:
            self._events.close()
            self._emit_metrics()

    async def aclose(self) -> None:
        self._input.close()
        await cancel_and_wait(self._task)
        self._events.close()

    def __aiter__(self) -> AsyncIterator[SynthesizedAudio]:
        return self

    async def __anext__(self) -> SynthesizedAudio:
        return await self._recv()


class SentenceStreamAdapter(SynthesizeStream):
    """Gives any :class:`TTS` a streaming interface by synthesizing sentence by sentence.

    The next sentence is synthesized while the current one is still being received
    (one-ahead prefetch), so there is no gap between sentences.
    """

    def __init__(
        self, tts: TTS, *, voice: str | None, segmenter: SentenceSegmenter | None = None
    ) -> None:
        self._segmenter = segmenter or SentenceSegmenter(min_chars=10, first_segment_min_chars=4)
        super().__init__(tts, voice=voice)

    def _start(self, sentence: str) -> ChunkedStream | None:
        return self._tts.synthesize(sentence, voice=self.voice) if sentence else None

    async def _play_sentence(self, stream: ChunkedStream, on_chunk: Callable[[], None]) -> None:
        """Forward one sentence's audio (silence-trimmed) and words onto this stream."""
        tts = self._tts
        trimmer = SilenceTrimmer(tts.sample_rate, tts.channels) if tts.trim_silence else None
        base = self._audio_duration  # where this sentence starts on the stream
        pending: list[WordTiming] = []

        def emit(frame: AudioFrame, *, final: bool = False) -> None:
            nonlocal pending
            words = None
            # words wait until the trimmer knows how much leading silence it dropped
            if pending and (trimmer is None or trimmer.started or final):
                dropped = trimmer.dropped_leading if trimmer is not None else 0.0
                words, pending = _shift_words(pending, base - dropped), []
            if frame:
                self._push_audio(frame, words=words)
            elif words:
                self._send(SynthesizedAudio(frame, self._request_id, self._segment_id, words=words))

        async for chunk in stream:
            if chunk.words:
                pending.extend(chunk.words)
            frame = chunk.frame
            if trimmer is not None and frame:
                frame = trimmer.push(frame)
            emit(frame)
            on_chunk()
        emit(
            trimmer.flush() if trimmer is not None else AudioFrame.empty(tts.sample_rate),
            final=True,
        )

    async def _run(self) -> None:
        jobs: Chan[tuple[str, bool]] = Chan()  # (sentence, ends_segment)

        async def playback() -> None:
            nxt: tuple[str, bool, ChunkedStream | None] | None = None

            def prefetch() -> None:
                nonlocal nxt
                if nxt is None and not jobs.empty():
                    s2, e2 = jobs.recv_nowait()
                    nxt = (s2, e2, self._start(s2))

            try:
                while True:
                    if nxt is None:
                        try:
                            sentence, ends = await jobs.recv()
                        except ChanClosed:
                            break
                        cur = (sentence, ends, self._start(sentence))
                    else:
                        cur, nxt = nxt, None
                    sentence, ends, stream = cur
                    if stream is not None:
                        self._segment_text = sentence
                        try:
                            await self._play_sentence(stream, prefetch)
                        finally:
                            await stream.aclose()
                    if ends:
                        self._end_segment()
                    prefetch()
            finally:
                if nxt is not None and nxt[2] is not None:
                    await nxt[2].aclose()

        player = asyncio.create_task(playback())
        try:
            async for item in self._input:
                if self.is_flush(item):
                    parts = self._segmenter.flush()
                    self._segmenter.reset()
                    if not parts:
                        jobs.send_nowait(("", True))
                    for i, part in enumerate(parts):
                        jobs.send_nowait((part, i == len(parts) - 1))
                else:
                    assert isinstance(item, str)
                    for sentence in self._segmenter.push(item):
                        jobs.send_nowait((sentence, False))
            jobs.close()
            await player
        finally:
            jobs.close()
            await cancel_and_wait(player)


def _shift_words(words: list[WordTiming] | None, offset: float) -> list[WordTiming] | None:
    """Word timings moved from a sentence's own timeline onto the stream's timeline."""
    if not words:
        return None
    return [WordTiming(w.word, w.start + offset, w.end + offset, w.confidence) for w in words]
