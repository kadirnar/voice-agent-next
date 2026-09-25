"""Omni (audio-output) LLMs in the cascade: the LLM's own speech replaces the TTS.

A fake omni model streams text deltas and 24 kHz audio deltas; by default its text runs
ahead of its audio (as LFM2.5-Audio's does), optionally it reports where each word is
spoken (``ChatChunk.audio_offset``).
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import pytest

from tests.test_session import Recorder, speak, wait_for
from voice_agent_next import (
    Agent,
    AgentSession,
    AgentState,
    CascadeOptions,
    ChatContext,
    ChatMessage,
    LLMCapabilities,
)
from voice_agent_next.chat import AudioContent
from voice_agent_next.engine import EngineOptions
from voice_agent_next.engines.cascade import CascadeEngine, _cut_after_word, _Spoken
from voice_agent_next.errors import ConfigurationError
from voice_agent_next.fallback import FallbackLLM
from voice_agent_next.llm import LLM, ChatChunk, CompletionUsage, LLMStream, ToolChoice
from voice_agent_next.metrics import EngineMetrics, LLMMetrics, SpeculationMetrics
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import (
    MockLLM,
    MockSTT,
    MockTTS,
    synth_speech,
)
from voice_agent_next.tools import FunctionTool
from voice_agent_next.transports import LoopbackTransport

RATE = 24_000
CPS = 15.0  # the fake model speaks 15 characters per second


class OmniLLM(LLM):
    """A fake audio-output LLM.

    Every reply word becomes a text delta plus ``len(word) / CPS`` seconds of audio, sent
    in 40 ms chunks. ``text_ahead``: all text first, then the audio (LFM-like), else word
    by word. ``realtime``: audio is produced at ``realtime`` x real time (0 = instantly).
    ``offsets``: report where each word is spoken (``ChatChunk.audio_offset``).
    """

    provider = "fake-omni"

    def __init__(
        self,
        replies: list[str] | None = None,
        *,
        ttfb: float = 0.0,
        realtime: float = 0.0,
        text_ahead: bool = True,
        offsets: bool = False,
    ) -> None:
        super().__init__(
            model="omni-test",
            capabilities=LLMCapabilities(
                tool_calling=False, audio_input=True, audio_output=True, audio_sample_rate=RATE
            ),
        )
        self.replies = list(replies or [])
        self.ttfb = ttfb
        self.realtime = realtime
        self.text_ahead = text_ahead
        self.offsets = offsets
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
        return _OmniStream(
            self,
            ctx,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
        )


class _OmniStream(LLMStream):
    async def _run(self) -> None:
        llm: OmniLLM = self._llm  # type: ignore[assignment]
        reply = llm.replies.pop(0) if llm.replies else "Hello there."
        if llm.ttfb:
            await asyncio.sleep(llm.ttfb)
        words = re.findall(r"\S+\s*", reply)
        offset = 0.0
        audio: list[float] = []
        for word in words:
            self._push(
                ChatChunk(self.request_id, delta=word, audio_offset=offset if llm.offsets else None)
            )
            duration = len(word) / CPS
            offset += duration
            if llm.text_ahead:
                audio.append(duration)
            else:
                await self._audio(duration, llm.realtime)
        for duration in audio:
            await self._audio(duration, llm.realtime)
        self._push(ChatChunk(self.request_id, usage=CompletionUsage(completion_tokens=len(words))))

    async def _audio(self, duration: float, realtime: float) -> None:
        signal = synth_speech(duration, RATE)
        t = 0.0
        while t < duration - 1e-9:
            chunk = signal.slice(t, min(duration, t + 0.04))
            self._push(ChatChunk(self.request_id, audio=chunk))
            t += 0.04
            await asyncio.sleep(chunk.duration * realtime)


def omni_session(llm: LLM, **kw: Any) -> AgentSession:
    return AgentSession(
        llm=llm,
        vad=EnergyVAD(),
        cascade_options=kw.pop("cascade_options", CascadeOptions(min_endpointing_delay=0.0)),
        **kw,
    )


def played(transport: LoopbackTransport) -> float:
    return sum(p.frame.duration for p in transport.played_log)


def assistant_messages(items: list[Any]) -> list[ChatMessage]:
    return [i for i in items if isinstance(i, ChatMessage) and i.role == "assistant"]


# ------------------------------------------------------------------------ configuration


def test_an_audio_llm_needs_no_tts_and_a_text_llm_does() -> None:
    engine = CascadeEngine(llm=OmniLLM(), vad=EnergyVAD())
    assert engine.llm_audio and engine.tts is None
    assert engine.output_sample_rate == RATE
    assert engine.stt is None  # half-cascade: the model hears the audio itself
    with pytest.raises(ConfigurationError, match="tts"):
        CascadeEngine(stt=MockSTT(), llm=MockLLM(), vad=EnergyVAD())
    with pytest.raises(ConfigurationError, match="use_llm_audio"):
        CascadeEngine(
            stt=MockSTT(),
            llm=MockLLM(),
            tts=MockTTS(),
            options=CascadeOptions(use_llm_audio=True),
        )
    # with a TTS, the TTS speaks unless asked otherwise
    tts = MockTTS(sample_rate=16_000)
    assert not CascadeEngine(llm=OmniLLM(), tts=tts, vad=EnergyVAD()).llm_audio
    forced = CascadeEngine(
        llm=OmniLLM(), tts=tts, vad=EnergyVAD(), options=CascadeOptions(use_llm_audio=True)
    )
    assert forced.llm_audio and forced.output_sample_rate == RATE
    # sessions and configs accept a cascade without tts=... (the engine checks it)
    AgentSession(llm=OmniLLM(), vad="energy")
    with pytest.raises(ConfigurationError):
        AgentSession(stt="mock", llm="mock", vad="energy")


def test_fallback_llm_keeps_audio_capabilities() -> None:
    both = FallbackLLM([OmniLLM(), OmniLLM()])
    assert both.capabilities.audio_output
    assert both.capabilities.audio_sample_rate == RATE
    mixed = FallbackLLM([OmniLLM(), MockLLM()])
    assert not mixed.capabilities.audio_output


# ---------------------------------------------------------------------------- replies


async def test_llm_speech_goes_to_the_speaker_and_its_text_to_the_transcript() -> None:
    reply = "Sure, the store opens at nine tomorrow."
    llm = OmniLLM([reply], ttfb=0.05)
    session = omni_session(llm)
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("You are helpful."), transport)
    await speak(transport, 0.8, 0.6)
    await wait_for(lambda: bool(rec.turn_metrics()), 5)
    await wait_for(lambda: session.agent_state == AgentState.LISTENING, 5)
    await session.aclose()

    # the user's audio reached the model (no STT)
    user = llm.requests[0].last_message("user")
    assert user is not None and isinstance(user.content[0], AudioContent)
    # every audio delta was played, and the text is the transcript
    expected = sum(len(w) / CPS for w in re.findall(r"\S+\s*", reply))
    assert played(transport) == pytest.approx(expected, abs=0.1)
    assert "".join(e.delta for e in rec.of("agent_transcript")) == reply
    [answer] = assistant_messages(session.history.items)
    assert answer.text == reply and not answer.interrupted
    # metrics: the turn and the engine measure the first *audio*
    [turn] = rec.turn_metrics()
    assert turn.voice_to_voice is not None and turn.response_ttfb is not None
    assert turn.response_ttfb >= 0.03  # the fake 50 ms TTFB; Windows timers are coarse (15.6 ms)
    engine_m = [m for m in rec.of("metrics") if isinstance(m, EngineMetrics)]
    assert engine_m and engine_m[0].ttfb is not None and engine_m[0].ttfb >= 0.03
    llm_m = [m for m in rec.of("metrics") if isinstance(m, LLMMetrics)]
    assert llm_m and llm_m[0].ttfb is not None and llm_m[0].ttft is not None
    assert llm_m[0].ttft <= llm_m[0].ttfb  # the text comes first


async def test_barge_in_keeps_the_heard_part_of_a_reply_whose_text_runs_ahead() -> None:
    reply = " ".join(f"word{i}" for i in range(60)) + "."  # ~24 s of speech
    llm = OmniLLM([reply, "Okay."], realtime=0.5)  # audio at 2x real time, text first
    session = omni_session(llm)
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await session.start(Agent("x"), transport)
    await speak(transport, 0.6, 0.5)
    await wait_for(lambda: session.agent_state == AgentState.SPEAKING)
    await asyncio.sleep(1.5)  # ~1.5 s heard; the whole text exists, the audio does not
    await speak(transport, 0.5, 0.6)  # barge in
    await wait_for(lambda: bool(rec.of("interrupted")))
    await wait_for(lambda: len(llm.requests) == 2, 10)  # the barge-in turn is answered
    await session.aclose()

    ev = rec.of("interrupted")[0]
    answer = assistant_messages(session.history.items)[0]
    assert answer.interrupted
    heard = answer.text
    assert reply.startswith(heard) and heard.endswith(heard.split()[-1])
    assert re.fullmatch(r"(word\d+ )*word\d+", heard)  # cut after a whole word
    # the speaking rate (15 chars/s here, 14 assumed) places the cut, not the text's lead
    assert len(heard) == pytest.approx(ev.played * CPS, abs=20)
    assert len(heard) < len(reply) / 3
    # the engine's history (what the model sees next) holds the heard part too
    conn: Any = session.connection
    assert conn is None or conn.chat_ctx.get(answer.id).text == heard
    second = llm.requests[1]
    [first_answer] = assistant_messages(second.items)[:1]
    assert first_answer.text == heard and first_answer.interrupted


async def test_truncation_estimates() -> None:
    engine = CascadeEngine(llm=OmniLLM(), vad=EnergyVAD())
    conn: Any = await engine.connect(EngineOptions())
    try:
        text = "one two three four five six seven eight nine ten"  # 49 chars
        msg = conn.chat_ctx.add_message("assistant", text)

        async def heard(spoken: _Spoken, ms: int) -> str:
            msg.content, msg.interrupted = [text], False
            conn._spoken[msg.id] = spoken
            return str(await conn.truncate(msg.id, ms))

        words = [w + " " for w in text.split()]
        # complete: the text's share of the audio
        done = _Spoken(llm_audio=True, complete=True, audio_duration=10.0, text=list(words))
        assert await heard(done, 5000) == "one two three four five"
        assert await heard(done, 20_000) == text
        assert await heard(done, 0) == ""
        # still generating (only 2 s of audio yet): the speaking rate sizes the reply
        conn._speech_rate = 10.0  # 49 chars -> ~4.9 s
        partial = _Spoken(llm_audio=True, complete=False, audio_duration=2.0, text=list(words))
        assert await heard(partial, 1000) == "one two three"
        # reported interleaving is exact
        offsets = [(w, i * 0.5) for i, w in enumerate(words)]
        exact = _Spoken(llm_audio=True, complete=False, audio_duration=2.0, segments=offsets)
        assert await heard(exact, 1200) == "one two three"
        assert msg.interrupted and msg.text == "one two three"
    finally:
        await conn.aclose()


def test_cut_after_word() -> None:
    assert _cut_after_word("hello big world", 7) == "hello big"
    assert _cut_after_word("hello big world", 6) == "hello "
    assert _cut_after_word("hello big world", 5) == "hello"
    assert _cut_after_word("hello", 99) == "hello"
    assert _cut_after_word("hello", 0) == ""


async def test_reported_offsets_are_passed_through() -> None:
    llm = OmniLLM(["alpha beta gamma."], offsets=True)
    engine = CascadeEngine(llm=llm, vad=EnergyVAD())
    conn: Any = await engine.connect(EngineOptions())
    try:
        await conn.send_text("hi")
        await wait_for(lambda: conn._response_task is not None and conn._response_task.done())
        [answer] = assistant_messages(conn.chat_ctx.items)
        spoken = conn._spoken[answer.id]
        assert [s for _, s in spoken.segments] == pytest.approx([0.0, 6 / CPS, 11 / CPS])
        assert spoken.complete and spoken.audio_duration > 1.0
    finally:
        await conn.aclose()


async def test_a_tts_still_speaks_verbatim_text_and_without_one_say_is_text_only() -> None:
    tts = MockTTS()
    engine = CascadeEngine(
        llm=OmniLLM(["From the model."]),
        tts=tts,
        vad=EnergyVAD(),
        options=CascadeOptions(use_llm_audio=True),
    )
    session = AgentSession(engine)
    rec = Recorder(session)
    await session.start(Agent("x"), LoopbackTransport())
    await session.say("Hello from the TTS.")
    await wait_for(lambda: len(assistant_messages(session.history.items)) == 1, 5)
    await wait_for(lambda: session.agent_state == AgentState.LISTENING, 5)
    await session.generate_reply(user_input="question")
    await wait_for(lambda: len(assistant_messages(session.history.items)) == 2, 5)
    await session.aclose()
    assert tts.requests == ["Hello from the TTS."]  # the model's reply never reached it
    transcript = "".join(e.delta for e in rec.of("agent_transcript"))
    assert "From the model." in transcript

    bare = omni_session(OmniLLM())
    rec = Recorder(bare)
    transport = LoopbackTransport()
    await bare.start(Agent("x"), transport)
    await bare.say("No voice for this.")
    await wait_for(lambda: bool(assistant_messages(bare.history.items)), 5)
    await bare.aclose()
    assert "".join(e.delta for e in rec.of("agent_transcript")) == "No voice for this."
    assert played(transport) == 0


# ---------------------------------------------------------------- preemptive generation


@pytest.mark.parametrize("preemptive_tts", [False, True])
async def test_preemptive_generation_needs_preemptive_tts_with_llm_audio(
    preemptive_tts: bool,
) -> None:
    llm = OmniLLM(["Sure thing."] * 3, ttfb=0.05)
    session = AgentSession(
        stt=MockSTT(transcripts=["book a table"], latency=0.02),
        llm=llm,
        vad=EnergyVAD(),
        cascade_options=CascadeOptions(
            min_endpointing_delay=0.8,
            preemptive_generation=True,
            preemptive_tts=preemptive_tts,
        ),
    )
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport, 0.5, 0.4)
    await wait_for(lambda: bool(rec.turn_metrics()), 5)
    await wait_for(lambda: session.agent_state == AgentState.LISTENING, 5)
    await session.aclose()
    specs = [m for m in rec.of("metrics") if isinstance(m, SpeculationMetrics)]
    if preemptive_tts:  # speculative speech allowed: the held reply is released
        assert [(m.hit, m.reason) for m in specs] == [(True, None)]
    else:  # the model's reply is speech: no speculation at all
        assert specs == []
    assert len(llm.requests) == 1
    assert played(transport) > 0.5
    [answer] = assistant_messages(session.history.items)
    assert answer.text == "Sure thing."
