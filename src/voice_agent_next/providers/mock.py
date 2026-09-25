"""Deterministic mock providers for tests, demos and framework-overhead benchmarks.

Everything here is offline, dependency-free and scriptable:

* :class:`MockSTT` returns scripted transcripts;
* :class:`MockLLM` returns scripted replies or tool calls (default: echo);
* :class:`MockTTS` produces a speech-like tone whose length depends on the text;
* :class:`MockTurnDetector` returns a fixed or punctuation-based probability;
* :class:`MockEngine` is a scripted *native* speech-to-speech engine with server-side
  (energy) VAD, tool calls, cancellation and truncation.

Latencies (``ttft``, ``ttfb``, ``response_delay``...) are configurable so the
benchmark can measure pure framework overhead against known component latencies.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np

from ..audio.frame import AudioFrame
from ..chat import ChatContext, ChatMessage, FunctionCall, FunctionCallOutput
from ..engine import EngineCapabilities, EngineConnection, EngineOptions, S2SEngine
from ..events import (
    EngineUsage,
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
from ..llm import LLM, ChatChunk, CompletionUsage, LLMCapabilities, LLMStream, ToolChoice
from ..metrics import EngineMetrics
from ..registry import register_provider
from ..stt import STT, STTCapabilities, STTEvent, STTEventType, STTStream, Transcript
from ..tools import FunctionTool
from ..tts import TTS, ChunkedStream, SynthesizeStream, TTSCapabilities
from ..turn import TurnDetector
from ..utils.clock import now
from ..utils.ids import new_id
from ..vad import VADEventType, VADOptions
from .energy import EnergyVAD

__all__ = [
    "MockEngine",
    "MockEngineConnection",
    "MockLLM",
    "MockSTT",
    "MockTTS",
    "MockToolCall",
    "MockTurnDetector",
    "synth_speech",
]


def synth_speech(
    duration: float,
    sample_rate: int,
    *,
    frequency: float = 220.0,
    amplitude: float = 0.3,
    offset: int = 0,
) -> AudioFrame:
    """A speech-like test signal: a tone with 4 Hz (syllable-rate) amplitude modulation."""
    n = max(0, round(duration * sample_rate))
    t = (np.arange(n) + offset) / sample_rate
    envelope = 0.65 + 0.35 * np.sin(2 * np.pi * 4.0 * t)
    x = amplitude * envelope * np.sin(2 * np.pi * frequency * t)
    return AudioFrame.from_numpy(x.astype(np.float32), sample_rate)


@dataclass(slots=True)
class MockToolCall:
    """A scripted tool call for :class:`MockLLM` / :class:`MockEngine`."""

    name: str
    arguments: dict[str, Any] | str = field(default_factory=dict)

    def arguments_json(self) -> str:
        return self.arguments if isinstance(self.arguments, str) else json.dumps(self.arguments)


MockResponse: TypeAlias = str | MockToolCall | list[MockToolCall]
ResponseScript: TypeAlias = Sequence[MockResponse] | Callable[[ChatContext], MockResponse] | None


def _default_reply(ctx: ChatContext) -> str:
    last = ctx.last_message("user")
    if last is None or not last.text.strip():
        return "Hello! How can I help you today?"
    return f"You said: {last.text.strip()}"


class _Script:
    def __init__(self, responses: ResponseScript, default: str | None = None) -> None:
        self._responses = responses
        self._default = default
        self._i = 0

    def next(self, ctx: ChatContext) -> MockResponse:
        if callable(self._responses):
            return self._responses(ctx)
        if self._responses and self._i < len(self._responses):
            r = self._responses[self._i]
            self._i += 1
            return r
        return self._default if self._default is not None else _default_reply(ctx)


# ------------------------------------------------------------------------------ STT
@register_provider(
    "stt",
    "mock",
    description="Scripted fake STT (tests/benchmarks)",
    local=True,
    default_model="mock-stt",
)
class MockSTT(STT):
    """Returns scripted transcripts, one per utterance (then ``default_text``)."""

    provider = "mock"

    def __init__(
        self,
        *,
        model: str = "mock-stt",
        transcripts: Sequence[str] | Callable[[AudioFrame], str] | None = None,
        default_text: str = "hello",
        latency: float = 0.0,
        interim_results: bool = True,
        streaming: bool = True,
        sample_rate: int = 16_000,
        language: str | None = "en",
        speech_threshold: float = 0.01,
    ) -> None:
        super().__init__(
            model=model,
            capabilities=STTCapabilities(streaming=streaming, interim_results=interim_results),
            sample_rate=sample_rate,
            language=language,
        )
        self._transcripts = transcripts
        self._i = 0
        self.default_text = default_text
        self.latency = latency
        self.speech_threshold = speech_threshold

    def peek_transcript(self) -> str:
        if isinstance(self._transcripts, Sequence) and self._i < len(self._transcripts):
            return self._transcripts[self._i]
        return self.default_text

    def next_transcript(self, audio: AudioFrame) -> str:
        if callable(self._transcripts):
            return self._transcripts(audio)
        text = self.peek_transcript()
        if isinstance(self._transcripts, Sequence) and self._i < len(self._transcripts):
            self._i += 1
        return text

    async def _recognize(self, audio: AudioFrame, *, language: str | None) -> Transcript:
        if self.latency:
            await asyncio.sleep(self.latency)
        return Transcript(text=self.next_transcript(audio), language=language, confidence=1.0)

    def _create_stream(self, *, language: str | None) -> STTStream:
        return _MockSTTStream(self, language=language)


class _MockSTTStream(STTStream):
    async def _run(self) -> None:
        stt: MockSTT = self._stt  # type: ignore[assignment]
        frames: list[AudioFrame] = []
        speaking = False
        segment = new_id("seg_")
        speech_time = 0.0
        next_interim = 0.5
        async for item in self._input:
            if self.is_flush(item):
                if frames:
                    if stt.latency:
                        await asyncio.sleep(stt.latency)
                    text = stt.next_transcript(AudioFrame.concat(frames))
                    self._emit(
                        STTEvent(
                            STTEventType.FINAL_TRANSCRIPT,
                            Transcript(text, self._language, 1.0),
                            segment,
                        )
                    )
                    if speaking:
                        self._emit(STTEvent(STTEventType.END_OF_SPEECH, segment_id=segment))
                frames, speaking, speech_time, next_interim = [], False, 0.0, 0.5
                segment = new_id("seg_")
                continue
            assert isinstance(item, AudioFrame)
            voiced = item.rms() >= stt.speech_threshold
            if voiced and not speaking:
                speaking = True
                self._emit(STTEvent(STTEventType.START_OF_SPEECH, segment_id=segment))
            if speaking:
                frames.append(item)
                speech_time += item.duration
                if stt.capabilities.interim_results and speech_time >= next_interim:
                    words = stt.peek_transcript().split()
                    k = max(1, min(len(words), int(speech_time / 0.4)))
                    self._emit(
                        STTEvent(
                            STTEventType.INTERIM_TRANSCRIPT,
                            Transcript(" ".join(words[:k]), self._language),
                            segment,
                        )
                    )
                    next_interim += 0.5


# ------------------------------------------------------------------------------ LLM
@register_provider(
    "llm",
    "mock",
    description="Scripted fake LLM (tests/benchmarks)",
    local=True,
    default_model="mock-llm",
)
class MockLLM(LLM):
    """Streams scripted replies word by word (default: echoes the last user message).

    ``default_response`` replaces the echo once the script is exhausted (e.g. a long reply
    for barge-in benchmarks).
    """

    provider = "mock"

    def __init__(
        self,
        *,
        model: str = "mock-llm",
        responses: ResponseScript = None,
        default_response: str | None = None,
        ttft: float = 0.0,
        token_delay: float = 0.0,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(
            model=model,
            capabilities=LLMCapabilities(tool_calling=True),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.script = _Script(responses, default_response)
        self.ttft = ttft
        self.token_delay = token_delay
        self.requests: list[ChatContext] = []

    def _chat(
        self,
        ctx: ChatContext,
        *,
        tools: list[FunctionTool],
        tool_choice: ToolChoice | None,
        temperature: float | None,
        max_tokens: int | None,
        extra: dict[str, Any],
    ) -> LLMStream:
        self.requests.append(ctx.copy())
        return _MockLLMStream(
            self,
            ctx,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
        )


class _MockLLMStream(LLMStream):
    async def _run(self) -> None:
        llm: MockLLM = self._llm  # type: ignore[assignment]
        response = llm.script.next(self.ctx)
        if llm.ttft:
            await asyncio.sleep(llm.ttft)
        prompt_tokens = sum(len(m.text.split()) for m in self.ctx.messages())
        if isinstance(response, (MockToolCall, list)):
            calls = [response] if isinstance(response, MockToolCall) else response
            fcs = [FunctionCall(name=c.name, arguments=c.arguments_json()) for c in calls]
            self._push(ChatChunk(self.request_id, tool_calls=fcs))
            completion, finish = 10 * len(fcs), "tool_calls"
        else:
            words = re.findall(r"\S+\s*", response)
            for i, word in enumerate(words):
                if i and llm.token_delay:
                    await asyncio.sleep(llm.token_delay)
                self._push(ChatChunk(self.request_id, delta=word))
            completion, finish = len(words), "stop"
        self._push(
            ChatChunk(
                self.request_id,
                usage=CompletionUsage(prompt_tokens=prompt_tokens, completion_tokens=completion),
                finish_reason=finish,
            )
        )


# ------------------------------------------------------------------------------ TTS
@register_provider(
    "tts",
    "mock",
    description="Synthetic-tone fake TTS (tests/benchmarks)",
    local=True,
    default_model="mock-tts",
)
class MockTTS(TTS):
    """Produces a speech-like tone lasting ``len(text) / chars_per_second`` seconds.

    Args:
        ttfb: delay before the first chunk.
        realtime_factor: 0 = produce audio instantly; 1.0 = at real-time speed.
        streaming: expose a native input-streaming interface (synthesizes per flush).
    """

    provider = "mock"

    def __init__(
        self,
        *,
        model: str = "mock-tts",
        voice: str | None = None,
        sample_rate: int = 24_000,
        ttfb: float = 0.0,
        chars_per_second: float = 15.0,
        chunk_duration: float = 0.04,
        realtime_factor: float = 0.0,
        streaming: bool = False,
        frequency: float = 220.0,
        amplitude: float = 0.3,
    ) -> None:
        super().__init__(
            model=model,
            sample_rate=sample_rate,
            capabilities=TTSCapabilities(streaming=streaming),
            voice=voice,
        )
        self.ttfb = ttfb
        self.chars_per_second = chars_per_second
        self.chunk_duration = chunk_duration
        self.realtime_factor = realtime_factor
        self.frequency = frequency
        self.amplitude = amplitude
        self.requests: list[str] = []

    def audio_duration_for(self, text: str) -> float:
        text = text.strip()
        return max(0.2, len(text) / self.chars_per_second) if text else 0.0

    async def generate(self, text: str, push: Callable[[AudioFrame], None]) -> None:
        self.requests.append(text)
        duration = self.audio_duration_for(text)
        if duration <= 0:
            return
        if self.ttfb:
            await asyncio.sleep(self.ttfb)
        audio = synth_speech(
            duration, self.sample_rate, frequency=self.frequency, amplitude=self.amplitude
        )
        # pace against the start time, not chunk by chunk: a late wake-up (a 15.6 ms timer
        # tick on Windows, a loaded runner) must not add up into a slower-than-real-time
        # stream, which would make the audio underrun with gaps
        started = now()
        t = 0.0
        while t < duration - 1e-9:
            chunk = audio.slice(t, min(duration, t + self.chunk_duration))
            push(chunk)
            t = min(duration, t + self.chunk_duration)
            delay = started + t * self.realtime_factor - now() if self.realtime_factor else 0
            await asyncio.sleep(max(0.0, delay))

    def _synthesize(self, text: str, *, voice: str | None) -> ChunkedStream:
        return _MockChunkedStream(self, text, voice=voice)

    def _create_stream(self, *, voice: str | None) -> SynthesizeStream:
        return _MockSynthesizeStream(self, voice=voice)


class _MockChunkedStream(ChunkedStream):
    async def _run(self) -> None:
        tts: MockTTS = self._tts  # type: ignore[assignment]
        await tts.generate(self.text, self._push_audio)


class _MockSynthesizeStream(SynthesizeStream):
    async def _run(self) -> None:
        tts: MockTTS = self._tts  # type: ignore[assignment]
        buf: list[str] = []
        async for item in self._input:
            if self.is_flush(item):
                text = "".join(buf).strip()
                buf = []
                self._segment_text = text or None
                await tts.generate(text, self._push_audio)
                self._end_segment()
            else:
                assert isinstance(item, str)
                buf.append(item)


# ------------------------------------------------------------------------ turn detection
@register_provider(
    "turn", "mock", description="Fixed/punctuation-based fake turn detector", local=True
)
class MockTurnDetector(TurnDetector):
    provider = "mock"
    modality = "text"

    def __init__(
        self,
        *,
        model: str = "mock-turn",
        probability: float | None = None,
        threshold: float = 0.5,
        delay: float = 0.0,
    ) -> None:
        super().__init__(model=model, threshold=threshold)
        self.probability = probability
        self.delay = delay
        self.calls = 0

    async def _predict(self, *, audio: AudioFrame | None, chat_ctx: ChatContext | None) -> float:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.probability is not None:
            return self.probability
        last = chat_ctx.last_message("user") if chat_ctx else None
        if last is None or not last.text.strip():
            return 0.5
        return 0.9 if last.text.rstrip()[-1] in ".?!。？！" else 0.3


# ----------------------------------------------------------------------------- engine
@register_provider(
    "engine",
    "mock",
    description="Scripted native speech-to-speech engine (tests/benchmarks)",
    local=True,
    default_model="mock-s2s",
)
class MockEngine(S2SEngine):
    """A scripted *native* speech-to-speech engine.

    Detects user speech with an energy VAD (server-side turn detection), answers with
    scripted text (see :class:`MockLLM`) spoken as a synthetic tone, supports tool
    calls, cancellation and truncation.

    Args:
        responses: reply script (strings / :class:`MockToolCall` / lists / callable).
        transcripts: user transcript script (see :class:`MockSTT`).
        default_response: reply once ``responses`` is exhausted (default: echo the user).
        response_delay: delay between turn commit and the first response audio.
        vad_options: server-side VAD settings (a :class:`VADOptions` or a mapping of its
            fields, e.g. ``{min_silence_duration: 0.25}`` to end turns sooner).
        realtime_factor: 0 = audio produced instantly; 1.0 = at real-time speed.
        voice_updates / chat_ctx_updates: whether connections accept ``update_voice`` /
            ``update_chat_ctx`` (``False`` simulates a native model that cannot).
    """

    provider = "mock"

    def __init__(
        self,
        *,
        model: str = "mock-s2s",
        responses: ResponseScript = None,
        transcripts: Sequence[str] | None = None,
        default_response: str | None = None,
        response_delay: float = 0.0,
        token_delay: float = 0.0,
        input_sample_rate: int = 16_000,
        output_sample_rate: int = 24_000,
        vad_options: VADOptions | Mapping[str, Any] | None = None,
        chars_per_second: float = 15.0,
        chunk_duration: float = 0.04,
        realtime_factor: float = 0.0,
        voice_updates: bool = True,
        chat_ctx_updates: bool = True,
    ) -> None:
        super().__init__(
            model=model,
            capabilities=EngineCapabilities(
                native_audio=True,
                server_turn_detection=True,
                tool_calling=True,
                input_transcription=True,
                output_transcription=True,
                truncation=True,
                full_duplex=False,
                text_input=True,
            ),
            input_sample_rate=input_sample_rate,
            output_sample_rate=output_sample_rate,
        )
        self.llm = MockLLM(
            responses=responses, default_response=default_response, token_delay=token_delay
        )
        self.stt = MockSTT(transcripts=transcripts)
        self.tts = MockTTS(
            sample_rate=output_sample_rate,
            chars_per_second=chars_per_second,
            chunk_duration=chunk_duration,
            realtime_factor=realtime_factor,
        )
        self.response_delay = response_delay
        if isinstance(vad_options, Mapping):
            vad_options = VADOptions(
                **{"min_speech_duration": 0.1, "min_silence_duration": 0.4, **vad_options}
            )
        self.vad_options = vad_options or VADOptions(
            min_speech_duration=0.1, min_silence_duration=0.4
        )
        self.voice_updates = voice_updates
        self.chat_ctx_updates = chat_ctx_updates
        self.connections: list[MockEngineConnection] = []

    async def connect(self, options: EngineOptions) -> EngineConnection:
        conn = MockEngineConnection(self, options)
        self.connections.append(conn)
        return conn


class MockEngineConnection(EngineConnection):
    def __init__(self, engine: MockEngine, options: EngineOptions) -> None:
        super().__init__(engine, options)
        self._engine = engine
        self._vad = EnergyVAD(
            sample_rate=engine.input_sample_rate, options=engine.vad_options
        ).stream()
        self.chat_ctx = ChatContext(options.chat_ctx.items if options.chat_ctx else [])
        self.instructions = options.instructions
        self.tools: list[FunctionTool] = list(options.tools)
        self._pending: list[AudioFrame] = []
        self._response_task: asyncio.Task[None] | None = None
        self.truncations: list[tuple[str, int]] = []
        self.tool_outputs: list[FunctionCallOutput] = []
        self.received_audio = 0.0
        self.responses_started = 0

    # ------------------------------------------------------------------ input
    async def _send_audio(self, frame: AudioFrame) -> None:
        self.received_audio += frame.duration
        for ev in self._vad.push_audio(frame):
            if ev.type == VADEventType.START_OF_SPEECH:
                self._emit(InputSpeechStarted(audio_time=ev.audio_time - ev.speech_duration))
            elif ev.type == VADEventType.END_OF_SPEECH:
                # report where speech actually ended, not where silence was confirmed
                self._emit(InputSpeechStopped(audio_time=ev.audio_time - ev.silence_duration))
                self._pending = list(ev.frames)
                if self.options.turn_detection:
                    await self._commit()

    async def commit_input(self) -> None:
        if self._vad.speaking:
            self._pending = self._vad.speech_frames()
            self._vad.reset()
        await self._commit()

    async def clear_input(self) -> None:
        self._pending = []
        self._vad.reset()

    async def _commit(self) -> None:
        frames, self._pending = self._pending, []
        audio = AudioFrame.concat(frames) if frames else AudioFrame.empty(self.input_sample_rate)
        text = self._engine.stt.next_transcript(audio)
        item_id = new_id("item_")
        self._emit(InputCommitted(item_id=item_id))
        self.chat_ctx.add_message("user", text, id=item_id)
        self._emit(InputTranscript(item_id=item_id, text=text, is_final=True, language="en"))
        await self._start_response()

    # ---------------------------------------------------------------- control
    async def send_text(self, text: str, *, respond: bool = True) -> None:
        self.chat_ctx.add_message("user", text)
        if respond:
            await self._start_response()

    async def create_response(self, *, instructions: str | None = None) -> None:
        await self._start_response()

    async def say(self, text: str) -> None:
        await self._start_response(verbatim=text)

    async def cancel_response(self) -> None:
        task = self._response_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def truncate(self, item_id: str, audio_end_ms: int) -> str | None:
        self.truncations.append((item_id, audio_end_ms))
        item = self.chat_ctx.get(item_id)
        if isinstance(item, ChatMessage) and item.role == "assistant":
            chars = int(audio_end_ms / 1000.0 * self._engine.tts.chars_per_second)
            item.content = [item.text[:chars]]
            item.interrupted = True
            return item.text
        return None

    async def send_tool_output(self, output: FunctionCallOutput, *, respond: bool = True) -> None:
        self.tool_outputs.append(output)
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

    async def update_voice(self, voice: str) -> bool:
        if not self._engine.voice_updates:
            return False
        self.options.voice = voice
        return True

    async def update_chat_ctx(self, chat_ctx: ChatContext) -> bool:
        if not self._engine.chat_ctx_updates:
            return False
        self.chat_ctx = ChatContext(chat_ctx.items)
        return True

    async def aclose(self) -> None:
        await self.cancel_response()
        await super().aclose()

    # --------------------------------------------------------------- response
    async def _start_response(self, *, verbatim: str | None = None) -> None:
        await self.cancel_response()
        self.responses_started += 1
        rid = new_id("resp_")
        self._response_task = asyncio.create_task(self._respond(rid, verbatim))

    async def _respond(self, rid: str, verbatim: str | None) -> None:
        engine = self._engine
        t0 = now()
        first_audio: float | None = None
        status: ResponseStatus = "completed"
        output_tokens = 0
        self._emit(ResponseStarted(response_id=rid))
        try:
            if engine.response_delay:
                await asyncio.sleep(engine.response_delay)
            if verbatim is not None:
                text, calls = verbatim, []
            else:
                ctx = ChatContext()
                if self.instructions:
                    ctx.add_message("system", self.instructions)
                ctx.items.extend(self.chat_ctx.items)
                result = await engine.llm.chat(ctx, tools=self.tools).collect()
                text, calls = result.text, result.tool_calls
            for call in calls:
                self.chat_ctx.append(call)
                self._emit(ResponseToolCall(response_id=rid, call=call))
            if text.strip():
                item_id = new_id("item_")
                self.chat_ctx.add_message("assistant", text, id=item_id)
                self._emit(ResponseText(response_id=rid, item_id=item_id, delta=text))
                output_tokens = len(text.split())

                def push(frame: AudioFrame, item_id: str = item_id) -> None:
                    nonlocal first_audio
                    if first_audio is None:
                        first_audio = now()
                    self._emit(ResponseAudio(response_id=rid, item_id=item_id, frame=frame))

                await engine.tts.generate(text, push)
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        finally:
            self._emit(
                ResponseDone(
                    response_id=rid,
                    status=status,
                    usage=EngineUsage(output_text_tokens=output_tokens),
                )
            )
            engine.emit(
                "metrics",
                EngineMetrics(
                    provider=engine.provider,
                    model=engine.model,
                    response_id=rid,
                    ttfb=None if first_audio is None else first_audio - t0,
                    duration=now() - t0,
                    output_text_tokens=output_tokens,
                    cancelled=status == "cancelled",
                ),
            )
