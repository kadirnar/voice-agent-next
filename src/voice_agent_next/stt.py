"""Speech-to-text (STT) component interface.

Implementing a provider:

* batch recognizers override :meth:`STT._recognize`;
* streaming recognizers set ``capabilities.streaming=True`` and override
  :meth:`STT._create_stream`, returning an :class:`STTStream` subclass whose
  :meth:`STTStream._run` consumes ``self._input`` and calls ``self._emit(...)``.

Non-streaming recognizers can be used in real time through
:class:`StreamAdapter`, which segments audio with a VAD.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

from .audio.frame import AudioFrame
from .audio.resample import StreamResampler
from .metrics import STTMetrics
from .utils.aio import Chan, ChanClosed, cancel_and_wait
from .utils.clock import now
from .utils.emitter import EventEmitter
from .utils.ids import new_id
from .utils.log import logger

if TYPE_CHECKING:
    from .vad import VAD

__all__ = [
    "STT",
    "STTCapabilities",
    "STTEvent",
    "STTEventType",
    "STTStream",
    "StreamAdapter",
    "Transcript",
    "WordTiming",
]


class STTEventType(StrEnum):
    START_OF_SPEECH = "start_of_speech"
    INTERIM_TRANSCRIPT = "interim_transcript"
    FINAL_TRANSCRIPT = "final_transcript"
    END_OF_SPEECH = "end_of_speech"
    END_OF_TURN = "end_of_turn"
    """Provider-side semantic end-of-turn (Deepgram Flux, AssemblyAI, Cartesia Ink, ...)."""
    EAGER_END_OF_TURN = "eager_end_of_turn"
    """Provider thinks the turn *probably* ended (start speculative generation)."""
    TURN_RESUMED = "turn_resumed"
    """The user kept talking after an eager end-of-turn (cancel speculation)."""


@dataclass(slots=True)
class WordTiming:
    word: str
    start: float
    end: float
    confidence: float | None = None


@dataclass(slots=True)
class Transcript:
    text: str
    language: str | None = None
    confidence: float | None = None
    start_time: float | None = None
    end_time: float | None = None
    words: list[WordTiming] | None = None


@dataclass(slots=True)
class STTEvent:
    type: STTEventType
    transcript: Transcript | None = None
    segment_id: str | None = None
    timestamp: float = field(default_factory=now)

    @property
    def text(self) -> str:
        return self.transcript.text if self.transcript else ""


@dataclass(frozen=True, slots=True)
class STTCapabilities:
    streaming: bool = False
    interim_results: bool = False
    word_timestamps: bool = False
    end_of_turn: bool = False
    language_detection: bool = False


class _Flush:
    __slots__ = ()


_FLUSH = _Flush()


class STT(ABC, EventEmitter):
    """Base class for speech recognizers. Emits ``"metrics"`` (:class:`STTMetrics`)."""

    provider: ClassVar[str] = "unknown"

    def __init__(
        self,
        *,
        model: str,
        capabilities: STTCapabilities,
        sample_rate: int = 16_000,
        language: str | None = None,
    ) -> None:
        EventEmitter.__init__(self)
        self.model = model
        self.capabilities = capabilities
        self.sample_rate = sample_rate
        self.language = language

    async def transcribe(
        self, audio: AudioFrame | Sequence[AudioFrame], *, language: str | None = None
    ) -> Transcript:
        """Recognize a complete utterance (any sample rate / channel count)."""
        frame = audio if isinstance(audio, AudioFrame) else AudioFrame.concat(audio)
        rs = StreamResampler(self.sample_rate, 1)
        frame = AudioFrame.concat([rs.push(frame), rs.flush()])
        t0 = now()
        error: str | None = None
        try:
            return await self._recognize(frame, language=language or self.language)
        except Exception as exc:
            error = repr(exc)
            raise
        finally:
            self.emit(
                "metrics",
                STTMetrics(
                    provider=self.provider,
                    model=self.model,
                    request_id=new_id("stt_"),
                    audio_duration=frame.duration,
                    duration=now() - t0,
                    streamed=False,
                    error=error,
                ),
            )

    async def _recognize(self, audio: AudioFrame, *, language: str | None) -> Transcript:
        """Batch recognition. Default: run a stream over the audio and join the finals."""
        if not self.capabilities.streaming:
            raise NotImplementedError(f"{type(self).__name__} must implement _recognize()")
        stream = self._create_stream(language=language)
        texts: list[str] = []
        detected: str | None = None
        try:
            stream.push_audio(audio)
            stream.end_input()
            async for ev in stream:
                if ev.type == STTEventType.FINAL_TRANSCRIPT and ev.transcript:
                    texts.append(ev.transcript.text.strip())
                    detected = detected or ev.transcript.language
        finally:
            await stream.aclose()
        return Transcript(text=" ".join(t for t in texts if t), language=detected or language)

    def stream(self, *, language: str | None = None) -> STTStream:
        """Open a streaming recognition session (requires ``capabilities.streaming``)."""
        if not self.capabilities.streaming:
            raise NotImplementedError(
                f"{type(self).__name__} does not support streaming; wrap it with "
                "voice_agent_next.stt.StreamAdapter(stt, vad)"
            )
        return self._create_stream(language=language or self.language)

    def _create_stream(self, *, language: str | None) -> STTStream:
        raise NotImplementedError

    def _create_adapter_stream(
        self, adapter: StreamAdapter, *, language: str | None
    ) -> STTStream | None:
        """Hook for batch recognizers: the stream :class:`StreamAdapter` runs for them.

        ``None`` (the default) uses the adapter's own stream: one final transcript per
        VAD-cut utterance. A recognizer that can do more on the VAD's segments (interim
        transcripts, VAD-aware filtering) returns its own :class:`STTStream` and sets
        ``capabilities.interim_results`` accordingly.
        """
        return None

    async def warmup(self) -> None:
        """Load models / open connections ahead of the first request (optional)."""

    async def aclose(self) -> None:
        """Release resources."""

    async def __aenter__(self) -> STT:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


class STTStream(ABC):
    """A streaming recognition session: push audio in, iterate :class:`STTEvent` out.

    Subclasses implement :meth:`_run`, reading ``self._input`` (a :class:`Chan` of
    :class:`AudioFrame` already resampled to ``stt.sample_rate`` mono, interleaved
    with flush markers — test them with :meth:`is_flush`) and calling :meth:`_emit`.
    """

    def __init__(self, stt: STT, *, language: str | None) -> None:
        self._stt = stt
        self._language = language
        self._input: Chan[AudioFrame | _Flush] = Chan()
        self._events: Chan[STTEvent] = Chan()
        self._resampler = StreamResampler(stt.sample_rate, 1)
        self._audio_duration = 0.0
        self._flush_time: float | None = None
        self._error: BaseException | None = None
        self._request_id = new_id("stt_")
        self._task = asyncio.create_task(self._main(), name=f"{type(self).__name__}._main")

    # ------------------------------------------------------------------ producer API
    @property
    def sample_rate(self) -> int:
        return self._stt.sample_rate

    def push_audio(self, frame: AudioFrame) -> None:
        if self._input.closed:
            raise RuntimeError("push_audio() after end_input()/aclose()")
        out = self._resampler.push(frame)
        if out:
            self._audio_duration += out.duration
            self._input.send_nowait(out)

    def flush(self) -> None:
        """Force-finalize: the provider should emit the final transcript ASAP.

        Maps to AssemblyAI ``ForceEndpoint``, Soniox ``finalize``, OpenAI
        ``input_audio_buffer.commit``, ElevenLabs manual ``commit``, Deepgram
        ``Finalize``... Forced finalization is how "external endpointing" reaches
        final transcripts in tens of milliseconds.
        """
        if self._input.closed:
            return
        tail = self._resampler.flush()
        if tail:
            self._input.send_nowait(tail)
        self._flush_time = now()
        self._input.send_nowait(_FLUSH)

    def end_input(self) -> None:
        """No more audio will be pushed. Pending audio is finalized, then the stream ends."""
        self.flush()
        self._input.close()

    async def aclose(self) -> None:
        self._input.close()
        await cancel_and_wait(self._task)
        self._events.close()

    # ------------------------------------------------------------ implementation API
    @staticmethod
    def is_flush(item: object) -> bool:
        return item is _FLUSH

    @abstractmethod
    async def _run(self) -> None:
        """Consume ``self._input`` and emit events with :meth:`_emit` until input ends."""

    def _emit(self, event: STTEvent) -> None:
        if event.type == STTEventType.FINAL_TRANSCRIPT and self._flush_time is not None:
            self._stt.emit(
                "metrics",
                STTMetrics(
                    provider=self._stt.provider,
                    model=self._stt.model,
                    request_id=self._request_id,
                    audio_duration=self._audio_duration,
                    latency=now() - self._flush_time,
                    streamed=True,
                ),
            )
            self._flush_time = None
            self._audio_duration = 0.0
        if not self._events.closed:
            self._events.send_nowait(event)

    async def _main(self) -> None:
        try:
            await self._run()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("%s failed", type(self).__name__)
            self._error = exc
        finally:
            self._events.close()

    # ------------------------------------------------------------------ consumer API
    def __aiter__(self) -> AsyncIterator[STTEvent]:
        return self

    async def __anext__(self) -> STTEvent:
        try:
            return await self._events.recv()
        except ChanClosed:
            if self._error is not None:
                err, self._error = self._error, None
                raise err from None
            raise StopAsyncIteration from None


class StreamAdapter(STT):
    """Makes a batch-only :class:`STT` streamable by segmenting audio with a :class:`VAD`.

    Emits START_OF_SPEECH / END_OF_SPEECH from the VAD and one FINAL_TRANSCRIPT per
    detected utterance (or per :meth:`STTStream.flush`). A recognizer may run its own
    stream over the VAD's segments instead (:meth:`STT._create_adapter_stream`), e.g. to
    add interim transcripts; then ``capabilities.interim_results`` is the wrapped one's.
    """

    def __init__(self, stt: STT, vad: VAD) -> None:
        # interim transcripts only come from a recognizer that runs its own adapter stream
        own_stream = type(stt)._create_adapter_stream is not STT._create_adapter_stream
        super().__init__(
            model=stt.model,
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=own_stream and stt.capabilities.interim_results,
                word_timestamps=stt.capabilities.word_timestamps,
                language_detection=stt.capabilities.language_detection,
            ),
            sample_rate=stt.sample_rate,
            language=stt.language,
        )
        self.provider = stt.provider  # type: ignore[misc]
        self.wrapped = stt
        self.vad = vad
        stt.on("metrics", lambda m: self.emit("metrics", m))

    async def _recognize(self, audio: AudioFrame, *, language: str | None) -> Transcript:
        return await self.wrapped._recognize(audio, language=language)

    def _create_stream(self, *, language: str | None) -> STTStream:
        stream = self.wrapped._create_adapter_stream(self, language=language)
        return stream if stream is not None else _AdapterStream(self, language=language)

    async def warmup(self) -> None:
        await self.wrapped.warmup()

    async def aclose(self) -> None:
        await self.wrapped.aclose()


class _AdapterStream(STTStream):
    def __init__(self, adapter: StreamAdapter, *, language: str | None) -> None:
        self._adapter = adapter
        super().__init__(adapter, language=language)

    async def _transcribe(self, frames: list[AudioFrame], segment_id: str) -> None:
        if not frames:
            return
        transcript = await self._adapter.wrapped.transcribe(frames, language=self._language)
        if transcript.text.strip():
            self._emit(STTEvent(STTEventType.FINAL_TRANSCRIPT, transcript, segment_id))

    async def _run(self) -> None:
        from .vad import VADEventType

        vad_stream = self._adapter.vad.stream()
        segment_id = new_id("seg_")

        async def finish_current() -> None:
            nonlocal segment_id
            await self._transcribe(vad_stream.speech_frames(), segment_id)
            self._emit(STTEvent(STTEventType.END_OF_SPEECH, segment_id=segment_id))
            segment_id = new_id("seg_")
            vad_stream.reset()

        async for item in self._input:
            if self.is_flush(item):
                if vad_stream.speaking:
                    await finish_current()
                continue
            assert isinstance(item, AudioFrame)
            for ev in vad_stream.push_audio(item):
                if ev.type == VADEventType.START_OF_SPEECH:
                    self._emit(STTEvent(STTEventType.START_OF_SPEECH, segment_id=segment_id))
                elif ev.type == VADEventType.END_OF_SPEECH:
                    await self._transcribe(list(ev.frames), segment_id)
                    self._emit(STTEvent(STTEventType.END_OF_SPEECH, segment_id=segment_id))
                    segment_id = new_id("seg_")
        if vad_stream.speaking:
            await finish_current()
