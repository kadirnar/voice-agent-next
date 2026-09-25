"""TTS usage is reported exactly once per request, whatever path the text takes (#137)."""

from __future__ import annotations

import asyncio

import pytest

from tests.test_session import mock_cascade, wait_for
from voice_agent_next import Agent
from voice_agent_next.fallback import FallbackTTS
from voice_agent_next.metrics import TTSMetrics, UsageSummary
from voice_agent_next.providers.mock import MockTTS
from voice_agent_next.transports import LoopbackTransport
from voice_agent_next.tts import TTS, SentenceStreamAdapter

TEXT = "Hello there. This is a second sentence. And a third one."


def _record(tts: TTS) -> tuple[list[TTSMetrics], UsageSummary]:
    got: list[TTSMetrics] = []
    usage = UsageSummary()

    def on(m: TTSMetrics) -> None:
        got.append(m)
        usage.add(m)

    tts.on("metrics", on)
    return got, usage


async def _stream(tts: TTS, text: str) -> float:
    stream = tts.stream()
    for word in text.split(" "):
        stream.push_text(word + " ")
        await asyncio.sleep(0)
    stream.end_input()
    duration = sum([c.frame.duration async for c in stream])
    await stream.aclose()
    await asyncio.sleep(0)  # let the stream's task report its metrics
    return duration


def _pushed(text: str) -> int:
    return sum(len(w) + 1 for w in text.split(" "))


async def test_chunked_synthesis_reports_usage_once() -> None:
    tts = MockTTS(chars_per_second=200)
    got, usage = _record(tts)
    audio = await tts.synthesize(TEXT).collect()
    assert len(got) == 1
    assert usage.tts_characters == len(TEXT)
    assert usage.tts_audio_seconds == pytest.approx(audio.duration)


async def test_native_stream_reports_usage_once() -> None:
    tts = MockTTS(streaming=True, chars_per_second=200)
    got, usage = _record(tts)
    duration = await _stream(tts, TEXT)
    assert len(got) == 1 and got[0].streamed
    assert usage.tts_characters == _pushed(TEXT)
    assert usage.tts_audio_seconds == pytest.approx(duration)


async def test_sentence_adapter_reports_usage_once() -> None:
    """The per-sentence ``synthesize()`` streams stay quiet: only the adapter reports."""
    tts = MockTTS(chars_per_second=200)
    assert isinstance(tts.stream(), SentenceStreamAdapter)
    got, usage = _record(tts)
    duration = await _stream(tts, TEXT)
    assert len(tts.requests) == 3  # three sentences were synthesized...
    assert len(got) == 1 and got[0].streamed  # ...and reported as one request
    assert usage.tts_characters == _pushed(TEXT)  # not ~2x
    assert usage.tts_audio_seconds == pytest.approx(duration, abs=0.01)


async def test_fallback_synthesis_reports_the_serving_provider_once() -> None:
    a, b = MockTTS(model="a", chars_per_second=200), MockTTS(model="b", chars_per_second=200)
    fb = FallbackTTS([a, b])
    got, usage = _record(fb)
    await fb.synthesize(TEXT).collect()
    assert [m.model for m in got] == ["a"]  # the provider's own metrics are forwarded
    assert usage.tts_characters == len(TEXT)


async def test_fallback_sentence_adapter_reports_usage_once() -> None:
    fb = FallbackTTS([MockTTS(model="a", chars_per_second=200), MockTTS(model="b")])
    assert isinstance(fb.stream(), SentenceStreamAdapter)
    got, usage = _record(fb)
    await _stream(fb, TEXT)
    assert len(got) == 1
    assert usage.tts_characters == _pushed(TEXT)


async def test_fallback_native_stream_reports_usage_once() -> None:
    fb = FallbackTTS([MockTTS(model="a", streaming=True, chars_per_second=200)])
    got, usage = _record(fb)
    await _stream(fb, TEXT)
    assert len(got) == 1
    assert usage.tts_characters == _pushed(TEXT)


@pytest.mark.parametrize("streaming", [False, True])
async def test_cascade_session_counts_tts_characters_once(streaming: bool) -> None:
    session = mock_cascade(tts=MockTTS(streaming=streaming, chars_per_second=200))
    got: list[TTSMetrics] = []
    session.on("metrics", lambda m: got.append(m) if isinstance(m, TTSMetrics) else None)
    await session.start(Agent("x"), LoopbackTransport())
    await session.say(TEXT)
    await wait_for(lambda: bool(got))
    await asyncio.sleep(0.05)  # a double report would arrive right after the first
    await session.aclose()
    assert len(got) == 1
    assert session.usage.tts_characters == got[0].characters
    assert len(TEXT) <= session.usage.tts_characters <= len(TEXT) + 2
