"""Component base classes exercised through the mock providers and EnergyVAD."""

from __future__ import annotations

import asyncio

import pytest

from voice_agent_next import AudioFrame, ChatContext, VADOptions
from voice_agent_next.metrics import LLMMetrics, STTMetrics, TTSMetrics, VADMetrics
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import (
    MockLLM,
    MockSTT,
    MockToolCall,
    MockTTS,
    MockTurnDetector,
    synth_speech,
)
from voice_agent_next.stt import StreamAdapter, STTEventType
from voice_agent_next.tts import SentenceStreamAdapter
from voice_agent_next.vad import VADEventType


def silence(d: float, sr: int = 16_000) -> AudioFrame:
    return AudioFrame.silence(d, sr)


def chunks(frame: AudioFrame, step: float = 0.02) -> list[AudioFrame]:
    out = []
    t = 0.0
    while t < frame.duration - 1e-9:
        out.append(frame.slice(t, t + step))
        t += step
    return out


# ------------------------------------------------------------------------------- VAD


def run_vad(vad: EnergyVAD, audio: list[AudioFrame]) -> list:  # type: ignore[type-arg]
    stream = vad.stream()
    events = []
    for f in audio:
        events += stream.push_audio(f)
    return events


def test_energy_vad_detects_utterances() -> None:
    vad = EnergyVAD(
        options=VADOptions(
            min_speech_duration=0.1, min_silence_duration=0.3, prefix_padding_duration=0.3
        )
    )
    audio = chunks(silence(0.5)) + chunks(synth_speech(1.0, 16_000)) + chunks(silence(0.6))
    audio += chunks(synth_speech(0.5, 16_000)) + chunks(silence(0.6))
    events = run_vad(vad, audio)
    kinds = [e.type for e in events]
    assert kinds == [VADEventType.START_OF_SPEECH, VADEventType.END_OF_SPEECH] * 2
    start, end = events[0], events[1]
    assert start.audio_time == pytest.approx(0.6, abs=0.05)  # 0.5 s silence + 0.1 s min speech
    assert end.speech_duration == pytest.approx(1.0, abs=0.1)
    assert end.silence_duration == pytest.approx(0.3, abs=0.03)
    # END frames = prefix padding (0.3 s) + speech + trailing silence
    total = sum(f.duration for f in end.frames)
    assert total == pytest.approx(0.3 + 1.0 + 0.3, abs=0.1)


def test_vad_ignores_short_blips_and_resamples_input() -> None:
    vad = EnergyVAD(options=VADOptions(min_speech_duration=0.2, min_silence_duration=0.3))
    blip = synth_speech(0.08, 48_000)
    events = run_vad(vad, [AudioFrame.silence(0.3, 48_000), blip, AudioFrame.silence(0.5, 48_000)])
    assert events == []


def test_vad_inference_events_and_reset() -> None:
    vad = EnergyVAD()
    stream = vad.stream(emit_inference_events=True)
    evs = stream.push_audio(synth_speech(0.2, 16_000))
    assert all(e.type in (VADEventType.INFERENCE_DONE, VADEventType.START_OF_SPEECH) for e in evs)
    assert stream.speaking and stream.speech_frames()
    stream.reset()
    assert not stream.speaking and stream.speech_frames() == []


def test_vad_emits_metrics() -> None:
    vad = EnergyVAD()
    got: list[VADMetrics] = []
    vad.on("metrics", got.append)
    stream = vad.stream()
    stream.push_audio(silence(6.0))
    stream.close()
    assert got and got[0].inference_count > 0 and got[0].audio_duration >= 5.0


# ------------------------------------------------------------------------------- STT


async def test_mock_stt_streaming_events_and_metrics() -> None:
    stt = MockSTT(transcripts=["hello world how are you"])
    metrics: list[STTMetrics] = []
    stt.on("metrics", metrics.append)
    stream = stt.stream()
    for f in chunks(synth_speech(1.2, 16_000)):
        stream.push_audio(f)
    stream.end_input()
    events = [e async for e in stream]
    kinds = [e.type for e in events]
    assert kinds[0] == STTEventType.START_OF_SPEECH
    assert STTEventType.INTERIM_TRANSCRIPT in kinds
    finals = [e for e in events if e.type == STTEventType.FINAL_TRANSCRIPT]
    assert [e.text for e in finals] == ["hello world how are you"]
    assert kinds[-1] == STTEventType.END_OF_SPEECH
    assert metrics and metrics[0].streamed and metrics[0].latency is not None
    await stream.aclose()


async def test_stt_transcribe_batch_and_via_stream() -> None:
    batch = MockSTT(transcripts=["one"], streaming=False)
    assert (await batch.transcribe(synth_speech(0.5, 48_000))).text == "one"
    streaming = MockSTT(transcripts=["two"])
    assert (await streaming.transcribe([synth_speech(0.5, 16_000)])).text == "two"
    with pytest.raises(NotImplementedError):
        batch.stream()


async def test_stream_adapter_makes_batch_stt_streamable() -> None:
    batch = MockSTT(transcripts=["first", "second"], streaming=False)
    adapter = StreamAdapter(batch, EnergyVAD(options=VADOptions(min_silence_duration=0.3)))
    stream = adapter.stream()
    audio = (
        chunks(synth_speech(0.6, 16_000)) + chunks(silence(0.5)) + chunks(synth_speech(0.6, 16_000))
    )
    for f in audio:
        stream.push_audio(f)
    stream.end_input()  # second utterance is finalized by end_input (no trailing silence)
    events = [e async for e in stream]
    finals = [e.text for e in events if e.type == STTEventType.FINAL_TRANSCRIPT]
    assert finals == ["first", "second"]
    assert [e.type for e in events].count(STTEventType.START_OF_SPEECH) == 2


# ------------------------------------------------------------------------------- LLM


async def test_mock_llm_streams_text_tool_calls_and_metrics() -> None:
    llm = MockLLM(responses=["Hello there friend.", MockToolCall("lookup", {"q": "x"})], ttft=0.01)
    metrics: list[LLMMetrics] = []
    llm.on("metrics", metrics.append)
    ctx = ChatContext()
    ctx.add_message("user", "hi")
    chunks_ = [c async for c in llm.chat(ctx)]
    assert "".join(c.delta for c in chunks_) == "Hello there friend."
    assert chunks_[-1].usage is not None and chunks_[-1].finish_reason == "stop"
    result = await llm.chat(ctx).collect()
    assert result.text == "" and result.tool_calls[0].name == "lookup"
    assert result.tool_calls[0].parsed_arguments() == {"q": "x"}
    # script exhausted -> echo
    assert (await llm.chat(ctx).collect()).text == "You said: hi"
    assert metrics[0].ttft is not None and metrics[0].ttft >= 0.01
    assert metrics[0].completion_tokens == 3


async def test_llm_stream_cancellation_reports_cancelled() -> None:
    llm = MockLLM(responses=["a b c d e f g"], token_delay=0.05)
    metrics: list[LLMMetrics] = []
    llm.on("metrics", metrics.append)
    stream = llm.chat(ChatContext())
    first = await stream.__anext__()
    assert first.delta
    await stream.aclose()
    assert metrics and metrics[0].cancelled


# ------------------------------------------------------------------------------- TTS


async def test_mock_tts_chunked_synthesis_and_metrics() -> None:
    tts = MockTTS(chars_per_second=20, ttfb=0.01)
    metrics: list[TTSMetrics] = []
    tts.on("metrics", metrics.append)
    audio = await tts.synthesize("**Twenty characters!**").collect()  # markdown stripped
    assert tts.requests == ["Twenty characters!"]
    assert audio.sample_rate == 24_000
    assert audio.duration == pytest.approx(18 / 20, abs=0.05)
    # the injected 10 ms delay is observed (Windows' loop clock can wake ~0.3 ms early)
    assert metrics[0].ttfb is not None and metrics[0].ttfb >= 0.008
    assert metrics[0].audio_duration == pytest.approx(audio.duration)


async def test_sentence_stream_adapter_segments_and_alignment() -> None:
    tts = MockTTS(chars_per_second=100)
    stream = tts.stream()
    assert isinstance(stream, SentenceStreamAdapter)
    for word in [
        "Hello",
        "there.",
        "This",
        "is",
        "the",
        "second",
        "sentence.",
        "And",
        "a",
        "third",
        "one",
    ]:
        stream.push_text(word + " ")
        await asyncio.sleep(0)
    stream.end_input()
    texts, finals, total = [], 0, 0.0
    async for chunk in stream:
        if chunk.text:
            texts.append(chunk.text)
        finals += chunk.is_final
        total += chunk.frame.duration
    assert texts == ["Hello there.", "This is the second sentence.", "And a third one"]
    assert finals == 1  # one flushed segment (end_input)
    assert total > 0.5
    await stream.aclose()


async def test_native_streaming_tts_mock_segments_per_flush() -> None:
    tts = MockTTS(streaming=True, chars_per_second=100)
    stream = tts.stream()
    stream.push_text("First part. ")
    stream.flush()
    stream.push_text("Second part.")
    stream.end_input()
    finals = [c for c in [c async for c in stream] if c.is_final]
    assert len(finals) == 2


# --------------------------------------------------------------------- turn detector


async def test_mock_turn_detector() -> None:
    td = MockTurnDetector()
    ctx = ChatContext()
    ctx.add_message("user", "I want to book a")
    assert await td.predict_end_of_turn(chat_ctx=ctx) < 0.5
    ctx.add_message("user", "I want to book a table.")
    assert await td.predict_end_of_turn(chat_ctx=ctx) > 0.5
    fixed = MockTurnDetector(probability=0.2)
    got = []
    fixed.on("metrics", got.append)
    assert await fixed.predict_end_of_turn() == 0.2
    assert got and not got[0].end_of_turn
