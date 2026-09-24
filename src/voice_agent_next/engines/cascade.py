"""Cascaded speech-to-speech engine: VAD -> STT -> (turn detector) -> LLM -> TTS.

Exposes the same :class:`~voice_agent_next.engine.EngineConnection` interface as native
speech-to-speech models, so the session treats both identically.

Pipeline per user turn:

1. user audio is streamed continuously into the STT stream and the VAD;
2. VAD ``START_OF_SPEECH`` -> ``InputSpeechStarted`` (the session may barge in);
3. VAD ``END_OF_SPEECH`` -> the STT is flushed, the (optional) turn detector scores the
   utterance, and the turn is committed after ``min_endpointing_delay`` (confident) or
   ``max_endpointing_delay`` (user probably not done) unless speech resumes;
4. the LLM streams text; complete sentences are cleaned and pushed into a TTS stream,
   whose audio is emitted as ``ResponseAudio`` with text aligned per sentence.

Half-cascade: with ``stt=None`` and an LLM whose ``capabilities.audio_input`` is True
(Ultravox, Qwen-Omni, Gemini, gpt-4o-audio...), the user's audio is passed to the LLM
directly as :class:`~voice_agent_next.chat.AudioContent`.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from ..audio.buffer import AudioBuffer
from ..audio.frame import AudioFrame
from ..chat import AudioContent, ChatContext, ChatMessage, FunctionCall, FunctionCallOutput
from ..engine import EngineCapabilities, EngineConnection, EngineOptions, S2SEngine
from ..errors import ConfigurationError
from ..events import (
    EngineErrorEvent,
    InputCommitted,
    InputSpeechStarted,
    InputSpeechStopped,
    InputTranscript,
    ResponseAudio,
    ResponseDone,
    ResponseStarted,
    ResponseStatus,
    ResponseText,
    ResponseToolCall,
)
from ..llm import LLM
from ..metrics import EngineMetrics, Metrics
from ..registry import create
from ..stt import STT, StreamAdapter, STTEventType, WordTiming
from ..text.filters import tts_clean
from ..text.sentences import SentenceSegmenter
from ..tools import FunctionTool
from ..tts import TTS
from ..turn import TurnDetector
from ..utils.aio import BackgroundTasks, cancel_and_wait
from ..utils.clock import now
from ..utils.ids import new_id
from ..utils.log import logger
from ..vad import VAD, VADEventType

__all__ = ["CascadeConnection", "CascadeEngine", "CascadeOptions"]


@dataclass(slots=True)
class CascadeOptions:
    min_endpointing_delay: float | None = None
    """Silence (from the end of speech) before committing when the user seems done.
    ``None`` = 0.4 s with a turn detector, 0.6 s with VAD only."""
    max_endpointing_delay: float = 2.5
    """Silence (from the end of speech) before committing when the turn detector says the
    user is probably not done."""
    final_transcript_timeout: float = 1.0
    """Max wait for the STT's final transcript after flushing."""
    text_filter: Callable[[str], str] | None = tts_clean
    """Applied to each sentence before TTS (markdown/emoji removal by default)."""
    first_sentence_min_chars: int = 4
    """Minimum length of the first spoken chunk (smaller = faster first audio)."""
    first_sentence_max_chars: int | None = 40
    """Split a longer first sentence at its first clause boundary: TTS engines that render
    a whole sentence before emitting audio (Kokoro, most local models) start sooner."""
    max_history_items: int | None = None
    """Truncate the LLM context to this many items (system prompt always kept)."""
    turn_audio_prefix: float = 0.5
    """Seconds of audio kept before speech start for the turn detector / audio LLM."""


@dataclass
class _Spoken:
    segments: list[tuple[str, float]] = field(default_factory=list)
    """(text, start offset in seconds within the item's audio)."""
    words: list[WordTiming] = field(default_factory=list)
    """Word timings reported by the TTS (relative to the start of the item's audio)."""
    audio_duration: float = 0.0
    text: list[str] = field(default_factory=list)


class CascadeEngine(S2SEngine):
    """Speech-to-speech engine assembled from STT, LLM, TTS, VAD and turn detection.

    Components can be instances or registry specs (``"deepgram/nova-3"``).
    """

    provider = "cascade"

    def __init__(
        self,
        *,
        stt: Any = None,
        llm: Any,
        tts: Any,
        vad: Any = None,
        turn_detector: Any = None,
        options: CascadeOptions | None = None,
    ) -> None:
        self.llm: LLM = create("llm", llm)
        self.tts: TTS = create("tts", tts)
        self.vad: VAD | None = create("vad", vad) if vad is not None else None
        self.turn_detector: TurnDetector | None = (
            create("turn", turn_detector) if turn_detector is not None else None
        )
        stt_obj: STT | None = create("stt", stt) if stt is not None else None
        if stt_obj is None and not self.llm.capabilities.audio_input:
            raise ConfigurationError("cascade needs stt=... unless the LLM accepts audio input")
        if stt_obj is None and self.vad is None:
            raise ConfigurationError("an audio-input LLM cascade needs vad=... to segment turns")
        if stt_obj is not None and not stt_obj.capabilities.streaming:
            if self.vad is None:
                raise ConfigurationError(
                    f"{type(stt_obj).__name__} is batch-only; add vad=... so it can be streamed"
                )
            stt_obj = StreamAdapter(stt_obj, self.vad)
        self.stt: STT | None = stt_obj
        self.options = options or CascadeOptions()
        input_rate = (
            stt_obj.sample_rate if stt_obj else (self.vad.sample_rate if self.vad else 16_000)
        )
        parts = [c.model for c in (stt_obj, self.llm, self.tts) if c is not None]
        super().__init__(
            model="+".join(parts),
            capabilities=EngineCapabilities(
                native_audio=False,
                server_turn_detection=True,
                tool_calling=self.llm.capabilities.tool_calling,
                input_transcription=stt_obj is not None,
                output_transcription=True,
                truncation=True,
                full_duplex=False,
                text_input=True,
            ),
            input_sample_rate=input_rate,
            output_sample_rate=self.tts.sample_rate,
        )
        for comp in self.components:
            comp.on("metrics", self._forward_metrics)

    @property
    def components(self) -> list[Any]:
        return [
            c for c in (self.vad, self.stt, self.turn_detector, self.llm, self.tts) if c is not None
        ]

    def _forward_metrics(self, m: Metrics) -> None:
        self.emit("metrics", m)

    async def connect(self, options: EngineOptions) -> EngineConnection:
        return CascadeConnection(self, options)

    async def warmup(self) -> None:
        await asyncio.gather(*(c.warmup() for c in self.components))

    async def aclose(self) -> None:
        await asyncio.gather(*(c.aclose() for c in self.components), return_exceptions=True)


class CascadeConnection(EngineConnection):
    def __init__(self, engine: CascadeEngine, options: EngineOptions) -> None:
        super().__init__(engine, options)
        self._e = engine
        self._opts = engine.options
        self.chat_ctx = ChatContext(options.chat_ctx.items if options.chat_ctx else [])
        self.instructions = options.instructions
        self.tools: list[FunctionTool] = list(options.tools)
        self._tasks = BackgroundTasks("cascade")
        rate = engine.input_sample_rate
        self._vad = engine.vad.stream() if engine.vad else None
        self._stt = engine.stt.stream(language=options.language) if engine.stt else None
        self._stt_task = asyncio.create_task(self._stt_loop()) if self._stt else None
        self._recent: deque[AudioFrame] = deque()
        self._recent_duration = 0.0
        self._turn_audio = AudioBuffer(rate, max_duration=120.0)
        self._user_speaking = False
        self._reset_turn()
        self._final_event = asyncio.Event()
        self._endpoint_task: asyncio.Task[None] | None = None
        self._speech_end_wall: float | None = None
        # STTs with built-in turn detection (Flux, Ink, AssemblyAI...) own the end of turn:
        # the VAD then only drives speech start/stop (barge-in), never commits.
        self._stt_turns = bool(engine.stt is not None and engine.stt.capabilities.end_of_turn)
        self._last_final_end: float | None = None
        self._response_task: asyncio.Task[None] | None = None
        self._spoken: dict[str, _Spoken] = {}

    # ---------------------------------------------------------------- user turn
    def _reset_turn(self) -> None:
        self._turn_item_id = new_id("item_")
        self._turn_finals: list[str] = []
        self._turn_interim = ""
        self._turn_audio.clear()
        self._turn_has_speech = False

    def _turn_text(self) -> str:
        return " ".join(t.strip() for t in self._turn_finals if t.strip())

    async def _send_audio(self, frame: AudioFrame) -> None:
        if self._stt is not None:
            self._stt.push_audio(frame)
        self._recent.append(frame)
        self._recent_duration += frame.duration
        while (
            self._recent
            and self._recent_duration - self._recent[0].duration > self._opts.turn_audio_prefix
        ):
            self._recent_duration -= self._recent.popleft().duration
        if self._turn_has_speech:  # keep pauses and resumed onsets: audio detectors need them
            self._turn_audio.append(frame)
        if self._vad is None:
            return
        for ev in self._vad.push_audio(frame):
            if ev.type == VADEventType.START_OF_SPEECH:
                self._on_speech_started(self.input_audio_time - ev.speech_duration)
            elif ev.type == VADEventType.END_OF_SPEECH:
                self._on_speech_stopped(self.input_audio_time - ev.silence_duration)

    def _on_speech_started(self, audio_time: float | None) -> None:
        if self._endpoint_task is not None and not self._endpoint_task.done():
            self._endpoint_task.cancel()  # the user kept talking: same turn continues
        if not self._user_speaking:
            self._user_speaking = True
            if not self._turn_has_speech:
                for f in self._recent:
                    self._turn_audio.append(f)
            self._turn_has_speech = True
        self._emit(InputSpeechStarted(audio_time=audio_time))

    def _on_speech_stopped(self, audio_time: float | None) -> None:
        self._user_speaking = False
        wall = self.audio_time_to_wall(audio_time) if audio_time is not None else None
        self._speech_end_wall = wall if wall is not None else now()
        self._emit(InputSpeechStopped(audio_time=audio_time))
        if self.options.turn_detection and not self._stt_turns:
            self._schedule_endpoint()

    def _schedule_endpoint(self) -> None:
        if self._endpoint_task is not None and not self._endpoint_task.done():
            self._endpoint_task.cancel()
        self._endpoint_task = self._tasks.spawn(self._endpoint(), name="cascade-endpoint")

    async def _wait_final_transcript(self) -> None:
        if self._stt is None:
            return
        self._final_event.clear()
        self._stt.flush()
        try:
            await asyncio.wait_for(self._final_event.wait(), self._opts.final_transcript_timeout)
        except TimeoutError:
            if self._turn_interim:
                self._turn_finals.append(self._turn_interim)  # fall back to the interim text
                self._turn_interim = ""

    def _min_delay(self) -> float:
        if self._opts.min_endpointing_delay is not None:
            return self._opts.min_endpointing_delay
        return 0.4 if self._e.turn_detector is not None else 0.6

    async def _endpoint(self) -> None:
        t_end = self._speech_end_wall if self._speech_end_wall is not None else now()
        detector = self._e.turn_detector
        early: asyncio.Future[float] | None = None
        if detector is not None and detector.modality == "audio" and self._turn_audio:
            # audio-only detectors don't need the transcript: overlap them with the STT flush
            early = asyncio.ensure_future(
                detector.predict_end_of_turn(audio=self._turn_audio.to_frame())
            )
        try:
            await self._wait_final_transcript()
            delay = self._min_delay()
            if detector is not None:
                if early is not None:
                    prob = await early
                elif self._turn_text() or self._turn_audio:
                    ctx = self._llm_context(None)
                    ctx.add_message("user", self._turn_text())
                    prob = await detector.predict_end_of_turn(
                        audio=self._turn_audio.to_frame() if self._turn_audio else None,
                        chat_ctx=ctx,
                    )
                else:
                    prob = 1.0
                if prob < detector.threshold:
                    delay = self._opts.max_endpointing_delay
            remaining = delay - (now() - t_end)
            if remaining > 0:
                await asyncio.sleep(remaining)
        finally:
            if early is not None and not early.done():
                early.cancel()
        await self._commit_turn()

    async def _stt_loop(self) -> None:
        assert self._stt is not None
        try:
            async for ev in self._stt:
                if ev.type == STTEventType.INTERIM_TRANSCRIPT:
                    self._turn_interim = ev.text
                    partial = " ".join(p for p in (self._turn_text(), ev.text) if p)
                    self._emit(
                        InputTranscript(item_id=self._turn_item_id, text=partial, is_final=False)
                    )
                elif ev.type == STTEventType.FINAL_TRANSCRIPT:
                    if ev.transcript is not None and ev.transcript.end_time is not None:
                        self._last_final_end = ev.transcript.end_time
                    if ev.text.strip():
                        self._turn_finals.append(ev.text)
                        self._emit(
                            InputTranscript(
                                item_id=self._turn_item_id,
                                text=self._turn_text(),
                                is_final=False,
                                language=ev.transcript.language if ev.transcript else None,
                            )
                        )
                    self._turn_interim = ""
                    self._final_event.set()
                elif ev.type == STTEventType.START_OF_SPEECH and self._vad is None:
                    self._on_speech_started(None)
                elif ev.type == STTEventType.END_OF_SPEECH and self._vad is None:
                    # STT audio time = our input stream time (all audio goes to the STT)
                    end = ev.transcript.end_time if ev.transcript else None
                    self._on_speech_stopped(end if end is not None else self._last_final_end)
                elif ev.type == STTEventType.END_OF_TURN and self.options.turn_detection:
                    if self._endpoint_task is not None and not self._endpoint_task.done():
                        self._endpoint_task.cancel()
                    self._tasks.spawn(self._commit_turn())
        except Exception as exc:
            logger.exception("STT stream failed")
            self._emit(EngineErrorEvent(error=exc, recoverable=False))

    async def _commit_turn(self) -> None:
        text = self._turn_text()
        audio = self._turn_audio.to_frame() if self._turn_audio else None
        use_audio = self._e.stt is None and audio is not None
        if not text and not use_audio:
            self._reset_turn()
            return
        item_id = self._turn_item_id
        content: str | AudioContent = AudioContent(audio, None) if use_audio and audio else text
        self.chat_ctx.add_message("user", content, id=item_id)
        self._emit(InputCommitted(item_id=item_id))
        self._emit(InputTranscript(item_id=item_id, text=text, is_final=True))
        self._reset_turn()
        await self._start_response()

    async def commit_input(self) -> None:
        if self._endpoint_task is not None and not self._endpoint_task.done():
            self._endpoint_task.cancel()
        self._user_speaking = False
        if self._vad is not None:
            self._vad.reset()
        await self._wait_final_transcript()
        await self._commit_turn()

    async def clear_input(self) -> None:
        if self._endpoint_task is not None and not self._endpoint_task.done():
            self._endpoint_task.cancel()
        self._user_speaking = False
        if self._vad is not None:
            self._vad.reset()
        self._reset_turn()

    # ------------------------------------------------------------------ control
    async def send_text(self, text: str, *, respond: bool = True) -> None:
        self.chat_ctx.add_message("user", text)
        if respond:
            await self._start_response()

    async def create_response(self, *, instructions: str | None = None) -> None:
        await self._start_response(instructions=instructions)

    async def say(self, text: str) -> None:
        await self._start_response(verbatim=text)

    async def cancel_response(self) -> None:
        task = self._response_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def truncate(self, item_id: str, audio_end_ms: int) -> str | None:
        msg = self.chat_ctx.get(item_id)
        spoken = self._spoken.get(item_id)
        if not isinstance(msg, ChatMessage) or msg.role != "assistant":
            return None
        end = audio_end_ms / 1000.0
        if spoken is not None and spoken.words:  # word-exact: TTS reported word timings
            heard = " ".join(w.word for w in spoken.words if w.start < end)
        elif spoken is not None and spoken.segments:
            heard = " ".join(t for t, start in spoken.segments if start < end)
        elif spoken is not None and spoken.audio_duration > 0:
            full = "".join(spoken.text)
            heard = full[: round(len(full) * min(1.0, end / spoken.audio_duration))]
        else:
            heard = ""
        msg.content = [heard.strip()]
        msg.interrupted = True
        return heard.strip()

    async def send_tool_output(self, output: FunctionCallOutput, *, respond: bool = True) -> None:
        self.chat_ctx.append(output)
        if respond:
            await self._start_response()

    async def update(
        self, *, instructions: str | None = None, tools: list[FunctionTool] | None = None
    ) -> None:
        if instructions is not None:
            self.instructions = instructions
        if tools is not None:
            self.tools = list(tools)

    async def aclose(self) -> None:
        await self.cancel_response()
        await self._tasks.cancel_all()
        if self._stt is not None:
            await self._stt.aclose()
        await cancel_and_wait(self._stt_task)
        if self._vad is not None:
            self._vad.close()
        await super().aclose()

    # ----------------------------------------------------------------- response
    def _llm_context(self, extra_instructions: str | None) -> ChatContext:
        ctx = ChatContext()
        if self.instructions:
            ctx.add_message("system", self.instructions)
        items = ChatContext(self.chat_ctx.items)
        if self._opts.max_history_items is not None:
            items.truncate(self._opts.max_history_items)
        ctx.items.extend(items.items)
        if extra_instructions:
            ctx.add_message("system", extra_instructions)
        return ctx

    async def _start_response(
        self, *, instructions: str | None = None, verbatim: str | None = None
    ) -> None:
        await self.cancel_response()
        rid = new_id("resp_")
        self._response_task = asyncio.create_task(
            self._respond(rid, instructions, verbatim), name=f"cascade-{rid}"
        )

    async def _respond(self, rid: str, instructions: str | None, verbatim: str | None) -> None:
        engine = self._e
        t0 = now()
        status: ResponseStatus = "completed"
        error: str | None = None
        first_audio: list[float] = []
        self._emit(ResponseStarted(response_id=rid))
        try:
            item_id = new_id("item_")
            msg = self.chat_ctx.add_message("assistant", "", id=item_id)
            if verbatim is not None:
                msg.content = [verbatim]
                await self._speak(rid, item_id, _once(verbatim), first_audio)
            else:
                ctx = self._llm_context(instructions)
                tools = self.tools if engine.llm.capabilities.tool_calling else []
                stream = engine.llm.chat(ctx, tools=tools)
                calls: list[FunctionCall] = []
                text_parts: list[str] = []

                async def text_source() -> AsyncIterator[str]:
                    try:
                        async for chunk in stream:
                            if chunk.delta:
                                text_parts.append(chunk.delta)
                                msg.content = ["".join(text_parts)]
                                yield chunk.delta
                            calls.extend(chunk.tool_calls)
                    finally:
                        await stream.aclose()

                await self._speak(rid, item_id, text_source(), first_audio)
                if not "".join(text_parts).strip():
                    self.chat_ctx.remove(item_id)
                for call in calls:
                    self.chat_ctx.append(call)
                    self._emit(ResponseToolCall(response_id=rid, call=call))
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except Exception as exc:
            logger.exception("cascade response failed")
            status, error = "failed", repr(exc)
            self._emit(EngineErrorEvent(error=exc, recoverable=True))
        finally:
            self._emit(ResponseDone(response_id=rid, status=status, error=error))
            engine.emit(
                "metrics",
                EngineMetrics(
                    provider=engine.provider,
                    model=engine.model,
                    response_id=rid,
                    ttfb=(first_audio[0] - t0) if first_audio else None,
                    duration=now() - t0,
                    cancelled=status == "cancelled",
                ),
            )

    async def _speak(
        self, rid: str, item_id: str, text: AsyncIterator[str], first_audio: list[float]
    ) -> None:
        """Stream ``text`` through the TTS, emitting aligned text and audio events."""
        engine = self._e
        tts_stream = engine.tts.stream(voice=self.options.voice)
        aligned = not engine.tts.capabilities.streaming  # sentence adapter reports segment text
        segmenter = SentenceSegmenter(
            min_chars=10,
            first_segment_min_chars=self._opts.first_sentence_min_chars,
            first_segment_max_chars=self._opts.first_sentence_max_chars,
        )
        spoken = self._spoken[item_id] = _Spoken()
        text_filter = self._opts.text_filter

        def push(sentence: str) -> None:
            cleaned = text_filter(sentence) if text_filter else sentence
            if not cleaned.strip():
                return
            spoken.text.append(cleaned + " ")
            if not aligned:
                self._emit(ResponseText(response_id=rid, item_id=item_id, delta=cleaned + " "))
            tts_stream.push_text(cleaned + " ")
            if aligned:
                # sentence-at-a-time TTS: synthesize exactly this segment now (the adapter's
                # own segmenter would otherwise hold a first *clause* until the sentence ends)
                tts_stream.flush()

        async def feed() -> None:
            try:
                async for delta in text:
                    for sentence in segmenter.push(delta):
                        push(sentence)
                for sentence in segmenter.flush():
                    push(sentence)
            finally:
                tts_stream.end_input()

        feeder = asyncio.create_task(feed(), name=f"cascade-feed-{rid}")
        try:
            async for audio in tts_stream:
                if audio.words:
                    spoken.words.extend(audio.words)
                if aligned and audio.text:
                    spoken.segments.append((audio.text, spoken.audio_duration))
                    self._emit(
                        ResponseText(response_id=rid, item_id=item_id, delta=audio.text + " ")
                    )
                if audio.frame:
                    if not first_audio:
                        first_audio.append(now())
                    spoken.audio_duration += audio.frame.duration
                    self._emit(ResponseAudio(response_id=rid, item_id=item_id, frame=audio.frame))
            await feeder  # surface LLM errors
        finally:
            await cancel_and_wait(feeder)
            await tts_stream.aclose()


async def _once(text: str) -> AsyncIterator[str]:
    yield text
