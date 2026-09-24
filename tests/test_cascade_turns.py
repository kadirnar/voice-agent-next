"""Cascade turn-taking with STT-provided end of turn, speech-end timing and word truncation."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.test_session import Recorder, mock_cascade, speak, wait_for
from voice_agent_next import Agent, AudioFrame, ChatMessage
from voice_agent_next.events import InputCommitted, InputSpeechStopped
from voice_agent_next.metrics import TurnMetrics
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import MockLLM, MockTTS, synth_speech
from voice_agent_next.stt import (
    STT,
    STTCapabilities,
    STTEvent,
    STTEventType,
    STTStream,
    Transcript,
    WordTiming,
)
from voice_agent_next.transports import LoopbackTransport
from voice_agent_next.tts import SynthesizeStream


class TurnSTT(STT):
    """Streaming STT that decides the end of turn itself (like Deepgram Flux / Cartesia Ink)."""

    provider = "test"

    def __init__(self, text: str, *, silence: float = 0.8) -> None:
        super().__init__(
            model="turn-stt",
            capabilities=STTCapabilities(streaming=True, end_of_turn=True),
            sample_rate=16_000,
        )
        self.text = text
        self.silence = silence
        self.flushes = 0

    def _create_stream(self, *, language: str | None) -> STTStream:
        return _TurnStream(self, language=language)


class _TurnStream(STTStream):
    async def _run(self) -> None:
        stt: TurnSTT = self._stt  # type: ignore[assignment]
        speaking, quiet, t = False, 0.0, 0.0
        async for item in self._input:
            if self.is_flush(item):
                stt.flushes += 1  # the provider owns the turn: flushes change nothing
                continue
            assert isinstance(item, AudioFrame)
            t += item.duration
            if item.rms() > 0.01:
                if not speaking:
                    speaking = True
                    self._emit(STTEvent(STTEventType.START_OF_SPEECH))
                quiet = 0.0
            elif speaking:
                quiet += item.duration
                if quiet >= stt.silence:
                    speaking = False
                    end = t - quiet
                    self._emit(
                        STTEvent(STTEventType.FINAL_TRANSCRIPT, Transcript(stt.text, end_time=end))
                    )
                    self._emit(STTEvent(STTEventType.END_OF_SPEECH, Transcript("", end_time=end)))
                    self._emit(STTEvent(STTEventType.END_OF_TURN))


def collect(session: Any, kind: type) -> list[Any]:
    """Record engine events of ``kind`` as the session handles them."""
    got: list[Any] = []
    original = session._handle

    async def spy(ev: Any) -> None:
        if isinstance(ev, kind):
            got.append(ev)
        await original(ev)

    session._handle = spy  # looked up per event by the session's event loop
    return got


async def test_stt_end_of_turn_owns_commits_even_with_a_vad() -> None:
    stt = TurnSTT("book a table please", silence=0.8)
    session = mock_cascade(stt=stt, vad=EnergyVAD(), responses=["Sure."])
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    commits = collect(session, InputCommitted)
    # 0.6 s of silence: the VAD (0.25 s) ends speech, but the STT (0.8 s) has not ended the turn
    await speak(transport, 0.6, 0.6)
    await asyncio.sleep(0.3)
    assert commits == []
    assert stt.flushes == 0  # no VAD-driven endpointing flush either
    await transport.play_user_audio(AudioFrame.silence(0.4, 16_000), realtime=False)
    await wait_for(lambda: len(rec.turn_metrics()) == 1, 3)
    await asyncio.sleep(0.2)
    await session.aclose()
    assert len(commits) == 1
    finals = [e.text for e in rec.of("user_transcript") if e.is_final]
    assert finals == ["book a table please"]


async def test_speech_end_time_comes_from_the_stt_without_a_vad() -> None:
    stt = TurnSTT("hello there", silence=0.8)
    session = mock_cascade(stt=stt, vad=None, llm=MockLLM(responses=["Hi."]))
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await session.start(Agent("x"), transport)
    stops = collect(session, InputSpeechStopped)
    await transport.play_user_audio(synth_speech(0.5, 16_000))  # real time
    await transport.play_user_audio(AudioFrame.silence(1.2, 16_000))
    await wait_for(lambda: bool(rec.turn_metrics()), 5)
    await session.aclose()
    assert stops[0].audio_time == pytest.approx(0.5, abs=0.05)  # the STT's end_time
    m: TurnMetrics = rec.turn_metrics()[0]
    # the turn ended 0.8 s after speech: endpointing delay is measured from the real speech end
    assert m.end_of_turn_delay == pytest.approx(0.8, abs=0.15)


class WordTTS(MockTTS):
    """Native streaming TTS that reports word timestamps (like Cartesia)."""

    def __init__(self) -> None:
        super().__init__(streaming=True, chars_per_second=20, realtime_factor=1.0)

    def _create_stream(self, *, voice: str | None) -> SynthesizeStream:
        return _WordStream(self, voice=voice)


class _WordStream(SynthesizeStream):
    async def _run(self) -> None:
        tts: WordTTS = self._tts  # type: ignore[assignment]
        buf: list[str] = []
        offset = 0.0
        async for item in self._input:
            if not self.is_flush(item):
                buf.append(str(item))
                continue
            text = "".join(buf).strip()
            buf = []
            if text:
                duration = tts.audio_duration_for(text)
                words = text.split()
                per = duration / len(words)
                timings = [
                    WordTiming(w, offset + i * per, offset + (i + 1) * per)
                    for i, w in enumerate(words)
                ]
                first = [True]

                def push(
                    frame: AudioFrame,
                    timings: list[WordTiming] = timings,
                    first: list[bool] = first,
                ) -> None:
                    self._push_audio(frame, words=timings if first[0] else None)
                    first[0] = False

                await tts.generate(text, push)
                offset += duration
            self._end_segment()


async def test_truncation_is_word_exact_with_tts_word_timestamps() -> None:
    answer = " ".join(f"word{i}" for i in range(40))
    tts = WordTTS()
    session = mock_cascade(transcripts=["talk"], responses=[answer], tts=tts)
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await session.start(Agent("x"), transport)
    await speak(transport, 0.5, 0.7)
    await wait_for(lambda: session.agent_state.value == "speaking", 5)
    await asyncio.sleep(1.0)
    await transport.play_user_audio(synth_speech(0.4, 16_000), realtime=False)  # barge in
    await wait_for(lambda: bool(rec.of("interrupted")), 5)
    await session.aclose()

    played = rec.of("interrupted")[0].played
    words = answer.split()
    per = tts.audio_duration_for(answer) / len(words)
    end = round(played * 1000) / 1000.0
    expected = " ".join(w for i, w in enumerate(words) if i * per < end)
    msg = session.connection.chat_ctx.get(rec.of("interrupted")[0].item_id)  # type: ignore[attr-defined]
    assert isinstance(msg, ChatMessage)
    assert msg.interrupted
    assert msg.text == expected
    history = [
        i for i in session.history.items if isinstance(i, ChatMessage) and i.role == "assistant"
    ]
    assert history[-1].text == expected
