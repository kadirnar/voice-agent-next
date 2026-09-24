"""GeminiTTS against a fake Gemini API (MockTransport replaying streamed audio responses).

Streamed TTS responses carry headerless 24 kHz s16le PCM in ``inlineData`` with
``mimeType: audio/l16;codec=pcm;rate=24000``. Needs the SDK (skipped otherwise); the
real-API test needs ``GOOGLE_API_KEY``/``GEMINI_API_KEY`` and ``-m integration``.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import wave
from typing import Any

import numpy as np
import pytest

from voice_agent_next import create
from voice_agent_next.errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderError,
    RateLimitError,
)
from voice_agent_next.metrics import TTSMetrics
from voice_agent_next.providers.google.tts import SAMPLE_RATE, GeminiTTS
from voice_agent_next.registry import get_provider

from .gemini_fake import (
    API_KEY,
    FakeGeminiAPI,
    audio,
    chunk,
    error_response,
    model_info,
    stream_response,
    text,
    usage,
)

genai = pytest.importorskip("google.genai")

MODEL = "gemini-3.8-flash-tts"
REAL_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def api() -> FakeGeminiAPI:
    return FakeGeminiAPI()


def make_tts(api: FakeGeminiAPI, **kwargs: Any) -> GeminiTTS:
    return GeminiTTS(api_key=API_KEY, http_client=api.http_client(), max_retries=0, **kwargs)


def pcm(seconds: float, rate: int = SAMPLE_RATE, value: int = 1000) -> bytes:
    return np.full(int(seconds * rate), value, dtype="<i2").tobytes()


def audio_stream(*pieces: bytes, mime: str = "audio/l16;codec=pcm;rate=24000") -> Any:
    events = [chunk(audio(p, mime), model=MODEL) for p in pieces]
    events.append(chunk(finish="STOP", usage=usage(9, 25), model=MODEL))
    return stream_response(events)


async def test_synthesize_streams_pcm_and_sends_the_request(api: FakeGeminiAPI) -> None:
    # odd-sized pieces: samples are split across network chunks
    body = pcm(0.3)
    api.reply(audio_stream(body[:4801], body[4801:9601], body[9601:]))
    tts = make_tts(api, voice="puck", style="warm and friendly", language="en-US")
    metrics: list[TTSMetrics] = []
    tts.on("metrics", metrics.append)

    items = [item async for item in tts.synthesize("Hello there! How can I help?")]

    frames = [i.frame for i in items if i.frame]
    assert len(frames) >= 2  # audio arrives as it is generated, not at the end
    assert all(f.sample_rate == SAMPLE_RATE and f.channels == 1 for f in frames)
    assert b"".join(f.data for f in frames) == body
    assert items[-1].is_final and items[0].text == "Hello there! How can I help?"

    request = api.requests[0]
    assert request.url.path == f"/v1beta/models/{MODEL}:streamGenerateContent"
    assert request.headers["x-goog-api-key"] == API_KEY
    sent = api.body()
    assert sent["contents"] == [
        {
            "role": "user",
            "parts": [
                {
                    "text": "Hello there! How can I help?",
                    "speechMetadata": {"style": "warm and friendly"},
                }
            ],
        }
    ]
    assert sent["generationConfig"]["responseModalities"] == ["AUDIO"]
    speech = sent["generationConfig"]["speechConfig"]
    assert speech["voice_config"] == {"prebuilt_voice_config": {"voice_name": "Puck"}}
    assert speech["language_code"] == "en-US"

    (m,) = metrics
    assert (m.provider, m.model, m.error) == ("google", MODEL, None)
    assert m.audio_duration == pytest.approx(0.3)
    assert m.ttfb is not None


async def test_custom_voice_ids_and_per_call_voice(api: FakeGeminiAPI) -> None:
    api.reply(audio_stream(pcm(0.05)), audio_stream(pcm(0.05)))
    tts = make_tts(api, voice="voice_abc123")
    await tts.synthesize("Hi.").collect()
    assert api.body()["generationConfig"]["speechConfig"]["voice_config"] == {
        "voice": "voice_abc123"
    }
    await tts.synthesize("Hi.", voice="Charon").collect()
    speech = api.body()["generationConfig"]["speechConfig"]
    assert speech["voice_config"] == {"prebuilt_voice_config": {"voice_name": "Charon"}}
    assert "language_code" not in speech  # auto-detected by default


async def test_legacy_models_get_the_style_in_the_prompt(api: FakeGeminiAPI) -> None:
    api.reply(audio_stream(pcm(0.05)))
    tts = make_tts(api, model="gemini-2.5-flash-preview-tts", style="cheerfully")
    await tts.synthesize("Have a nice day.").collect()
    assert api.body()["contents"][0]["parts"] == [{"text": "Say it cheerfully: Have a nice day."}]


async def test_other_rates_and_wav_payloads_are_converted(api: FakeGeminiAPI) -> None:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm(0.1))
    api.reply(
        audio_stream(pcm(0.2, rate=16_000), mime="audio/l16;codec=pcm;rate=16000"),
        audio_stream(buf.getvalue(), mime="audio/wav"),
    )
    tts = make_tts(api)
    resampled = await tts.synthesize("One.").collect()
    assert resampled.sample_rate == SAMPLE_RATE
    assert resampled.duration == pytest.approx(0.2, abs=0.01)
    unwrapped = await tts.synthesize("Two.").collect()
    assert unwrapped.data == pcm(0.1)  # the RIFF header is not played as audio


async def test_empty_audio_is_retried_then_fails(api: FakeGeminiAPI) -> None:
    no_audio = [chunk(text("I cannot do that."), finish="OTHER", model=MODEL)]
    api.reply(stream_response(no_audio), audio_stream(pcm(0.05)))
    tts = make_tts(api)
    frame = await tts.synthesize("Hello.").collect()
    assert frame.duration == pytest.approx(0.05)
    assert len(api.requests) == 2

    api.reply(stream_response(no_audio), stream_response(no_audio))
    with pytest.raises(ProviderError, match="no audio") as info:
        await tts.synthesize("Hello.").collect()
    assert info.value.retryable

    silent = make_tts(api, empty_retries=0)
    api.reply(stream_response(no_audio))
    with pytest.raises(ProviderError):
        await silent.synthesize("Hello.").collect()


async def test_blank_text_makes_no_request(api: FakeGeminiAPI) -> None:
    frame = await make_tts(api).synthesize("   ").collect()
    assert not frame and api.requests == []


async def test_stream_synthesizes_sentence_by_sentence(api: FakeGeminiAPI) -> None:
    api.reply(audio_stream(pcm(0.2)), audio_stream(pcm(0.3)))
    tts = make_tts(api, trim_silence=False)
    stream = tts.stream()
    stream.push_text("Sure, I can help with that. ")
    stream.push_text("What city are you in?")
    stream.end_input()
    items = [item async for item in stream]
    await stream.aclose()
    total = sum(i.frame.duration for i in items)
    assert total == pytest.approx(0.5)
    sent = [api.body(i)["contents"][0]["parts"][0]["text"] for i in range(2)]
    assert sent == ["Sure, I can help with that.", "What city are you in?"]


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            error_response(400, "INVALID_ARGUMENT", "API key not valid.", reason="API_KEY_INVALID"),
            AuthenticationError,
        ),
        (error_response(429, "RESOURCE_EXHAUSTED", "Quota exceeded"), RateLimitError),
        (error_response(500, "INTERNAL", "An internal error has occurred."), ProviderError),
    ],
)
async def test_errors_are_mapped(
    api: FakeGeminiAPI, response: Any, expected: type[ProviderError]
) -> None:
    api.reply(response)
    with pytest.raises(expected) as info:
        await make_tts(api).synthesize("Hello.").collect()
    assert info.value.provider == "google" and API_KEY not in str(info.value)


async def test_dropped_connection_is_mapped(api: FakeGeminiAPI) -> None:
    import httpx

    async def drop() -> None:
        raise httpx.RemoteProtocolError("peer closed connection")

    api.reply(stream_response([chunk(audio(pcm(0.05)), model=MODEL)], tail=drop))
    with pytest.raises(ProviderConnectionError):
        await make_tts(api).synthesize("Hello.").collect()


async def test_cancellation_closes_the_http_stream(api: FakeGeminiAPI) -> None:
    closed = asyncio.Event()

    async def hang() -> None:
        await asyncio.sleep(30)

    api.reply(stream_response([chunk(audio(pcm(0.05)), model=MODEL)], tail=hang, closed=closed))
    stream = make_tts(api).synthesize("A long story.")
    first = await stream.__anext__()
    assert first.frame
    await stream.aclose()
    await asyncio.wait_for(closed.wait(), 5)


async def test_extra_config_and_temperature(api: FakeGeminiAPI) -> None:
    api.reply(audio_stream(pcm(0.05)))
    speakers = {
        "multi_speaker_voice_config": {
            "speaker_voice_configs": [
                {
                    "speaker": "Joe",
                    "voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}},
                },
                {
                    "speaker": "Jane",
                    "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}},
                },
            ]
        }
    }
    tts = make_tts(api, temperature=0.7, extra_config={"speech_config": speakers, "seed": 3})
    await tts.synthesize("Joe: Hi Jane. Jane: Hi Joe.").collect()
    config = api.body()["generationConfig"]
    assert config["temperature"] == 0.7 and config["seed"] == 3
    assert "voice_config" not in config["speechConfig"]
    assert (
        config["speechConfig"]["multi_speaker_voice_config"]["speaker_voice_configs"][0]["speaker"]
        == "Joe"
    )


async def test_warmup_and_lifecycle(api: FakeGeminiAPI, caplog: pytest.LogCaptureFixture) -> None:
    api.reply(model_info(MODEL), error_response(403, "PERMISSION_DENIED", "denied"))
    tts = make_tts(api)
    await tts.warmup()
    assert (api.requests[0].method, api.requests[0].url.path) == ("GET", f"/v1beta/models/{MODEL}")
    with caplog.at_level(logging.WARNING):
        await tts.warmup()
    assert "Gemini TTS warmup failed" in caplog.text
    await tts.aclose()
    await tts.aclose()


def test_registry_defaults_and_validation() -> None:
    spec = get_provider("tts", "google")
    assert spec.default_model == MODEL and spec.extra == "google"
    tts = create("tts", "google", api_key=API_KEY)
    assert isinstance(tts, GeminiTTS)
    assert (tts.model, tts.voice, tts.sample_rate, tts.channels) == (MODEL, "Kore", 24_000, 1)
    assert not tts.capabilities.streaming
    lite = create("tts", "gemini/gemini-3.8-flash-lite-tts", api_key=API_KEY)
    assert lite.model == "gemini-3.8-flash-lite-tts"
    with pytest.raises(AuthenticationError):
        GeminiTTS()
    with pytest.raises(ConfigurationError):
        GeminiTTS(api_key=API_KEY, empty_retries=-1)


@pytest.mark.integration
@pytest.mark.skipif(not REAL_KEY, reason="needs GOOGLE_API_KEY or GEMINI_API_KEY")
async def test_real_api_streams_speech() -> None:
    tts = GeminiTTS(model=os.environ.get("GEMINI_TTS_TEST_MODEL", MODEL), api_key=REAL_KEY)
    try:
        frame = await tts.synthesize("Hello! This is a short test.").collect()
    finally:
        await tts.aclose()
    assert frame.sample_rate == SAMPLE_RATE
    assert 0.5 < frame.duration < 10
