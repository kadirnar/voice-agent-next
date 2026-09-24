"""OpenAI TTS (``/audio/speech``) and the OpenAI-compatible speech servers, against
``httpx.MockTransport``.

No network, no API key — except the ``integration`` test at the end, which needs
``OPENAI_API_KEY``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import numpy as np
import pytest

from voice_agent_next import AudioFrame
from voice_agent_next.engine import EngineOptions
from voice_agent_next.engines.cascade import CascadeEngine
from voice_agent_next.errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderError,
    RateLimitError,
)
from voice_agent_next.events import ResponseAudio, ResponseDone, ResponseText
from voice_agent_next.metrics import TTSMetrics
from voice_agent_next.providers.azure_openai import AzureOpenAITTS
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.kokoro_fastapi import KokoroFastAPITTS
from voice_agent_next.providers.localai import LocalAITTS
from voice_agent_next.providers.mock import MockLLM, MockSTT
from voice_agent_next.providers.openai.tts import OpenAITTS, split_text
from voice_agent_next.providers.speaches import SpeachesTTS
from voice_agent_next.registry import create

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def _no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_BASE_URL", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_AD_TOKEN",
                 "AZURE_OPENAI_ENDPOINT", "SPEACHES_BASE_URL", "SPEACHES_API_KEY",
                 "LOCALAI_BASE_URL", "LOCALAI_API_KEY", "KOKORO_FASTAPI_BASE_URL",
                 "KOKORO_FASTAPI_API_KEY"):  # fmt: skip
        monkeypatch.delenv(name, raising=False)


def pcm(seconds: float, rate: int = 24_000, level: int = 3000) -> bytes:
    """A deterministic, non-silent s16le ramp (never trimmed as silence)."""
    n = round(seconds * rate)
    return ((np.arange(n) % 200) * 20 + level).astype("<i2").tobytes()


async def chunks(*parts: bytes) -> AsyncIterator[bytes]:
    for part in parts:
        yield part


def split(data: bytes, *sizes: int) -> list[bytes]:
    """``data`` cut after each of ``sizes`` bytes (odd sizes split samples)."""
    out, pos = [], 0
    for size in sizes:
        out.append(data[pos : pos + size])
        pos += size
    out.append(data[pos:])
    return [p for p in out if p]


def wav_header(rate: int, channels: int = 1, *, data_size: int = 0xFFFFFFFF) -> bytes:
    """A streaming-style WAV header (placeholder sizes) with a LIST chunk before ``data``."""
    fmt = struct.pack("<HHIIHH", 1, channels, rate, rate * 2 * channels, 2 * channels, 16)
    info = b"INFO\x00"  # odd size: padded to a word boundary
    return (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
        + b"fmt " + struct.pack("<I", len(fmt)) + fmt
        + b"LIST" + struct.pack("<I", len(info)) + info + b"\x00"
        + b"data" + struct.pack("<I", data_size)
    )  # fmt: skip


class SpeechServer:
    """Records ``/audio/speech`` requests; answers with ``respond(request, index)``."""

    def __init__(self, respond: Callable[[httpx.Request, int], httpx.Response]) -> None:
        self.respond = respond
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        return self.respond(request, len(self.requests) - 1)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if r.method == "POST"]


def audio_response(*parts: bytes, status: int = 200) -> httpx.Response:
    return httpx.Response(status, content=chunks(*parts), headers={"content-type": "audio/pcm"})


# ------------------------------------------------------------------------- requests
async def test_streams_raw_pcm_from_the_speech_endpoint() -> None:
    audio = pcm(1.0)
    server = SpeechServer(lambda r, i: audio_response(*split(audio, 1001, 4000, 999)))
    tts = OpenAITTS(api_key="sk-test", instructions="Speak warmly.", speed=1.1,
                    http_client=server.client())  # fmt: skip
    metrics: list[TTSMetrics] = []
    tts.on("metrics", metrics.append)
    frame = await tts.synthesize("Hello **there**!").collect()
    await tts.aclose()

    assert frame.sample_rate == 24_000 and frame.channels == 1
    assert frame.data == audio  # odd chunk boundaries are re-aligned, nothing lost
    request = server.requests[0]
    assert str(request.url) == "https://api.openai.com/v1/audio/speech"
    assert request.headers["authorization"] == "Bearer sk-test"
    assert server.bodies()[0] == {
        "model": "gpt-4o-mini-tts",
        "input": "Hello there!",  # markdown cleaned before synthesis
        "response_format": "pcm",
        "voice": "marin",
        "instructions": "Speak warmly.",
        "speed": 1.1,
    }
    assert len(metrics) == 1 and metrics[0].ttfb is not None
    assert metrics[0].audio_duration == pytest.approx(1.0)
    assert metrics[0].characters == len("Hello there!") and metrics[0].error is None


async def test_voices_and_instructions_per_model() -> None:
    server = SpeechServer(lambda r, i: audio_response(pcm(0.1)))
    tts1 = OpenAITTS(api_key="k", model="tts-1", instructions="Whisper.",
                     http_client=server.client())  # fmt: skip
    assert tts1.voice == "alloy"  # the tts-1 models have no marin
    await tts1.synthesize("One.").collect()
    custom = OpenAITTS(api_key="k", voice="voice_1234", extra={"stream_format": "audio"},
                       http_client=server.client())  # fmt: skip
    await custom.synthesize("Two.").collect()
    await custom.synthesize("Three.", voice="cedar").collect()
    bodies = server.bodies()
    assert "instructions" not in bodies[0] and bodies[0]["voice"] == "alloy"
    assert bodies[1]["voice"] == {"id": "voice_1234"}
    assert bodies[1]["stream_format"] == "audio"
    assert bodies[2]["voice"] == "cedar"


async def test_wav_responses_are_unwrapped() -> None:
    audio = pcm(0.5)
    body = wav_header(24_000) + audio
    server = SpeechServer(lambda r, i: audio_response(*split(body, 7, 30, 20, 1)))
    tts = OpenAITTS(api_key="k", response_format="wav", http_client=server.client())
    frame = await tts.synthesize("Hi.").collect()
    assert frame.data == audio  # header (fmt, LIST, data) stripped, samples intact
    assert server.bodies()[0]["response_format"] == "wav"


async def test_wav_with_an_exact_data_size_ignores_trailing_chunks() -> None:
    audio = pcm(0.2)
    body = wav_header(24_000, data_size=len(audio)) + audio + b"LIST\x04\x00\x00\x00abcd"
    server = SpeechServer(lambda r, i: audio_response(body))
    tts = OpenAITTS(api_key="k", http_client=server.client())
    assert (await tts.synthesize("Hi.").collect()).data == audio


async def test_wav_at_another_rate_is_resampled() -> None:
    audio22 = pcm(1.0, rate=22_050)
    stereo = np.repeat(np.frombuffer(pcm(0.5, rate=16_000), "<i2"), 2).astype("<i2").tobytes()
    server = SpeechServer(
        lambda r, i: (
            audio_response(wav_header(22_050) + audio22)
            if i == 0
            else audio_response(wav_header(16_000, channels=2) + stereo)
        )
    )
    tts = LocalAITTS(http_client=server.client())  # LocalAI answers WAV at the model's rate
    frame = await tts.synthesize("Piper voice.").collect()
    assert frame.sample_rate == 24_000
    assert frame.duration == pytest.approx(1.0, abs=0.01)
    frame2 = await tts.synthesize("Stereo.").collect()
    assert frame2.channels == 1 and frame2.duration == pytest.approx(0.5, abs=0.01)
    body = server.bodies()[0]
    assert body == {"model": "tts-1", "input": "Piper voice.", "response_format": "wav",
                    "sample_rate": 24_000}  # fmt: skip
    assert str(server.requests[0].url) == "http://localhost:8080/v1/audio/speech"


async def test_unsupported_wav_audio_is_an_error() -> None:
    header = bytearray(wav_header(24_000))
    header[34:36] = struct.pack("<H", 8)  # 8-bit samples
    server = SpeechServer(lambda r, i: audio_response(bytes(header) + b"\x80" * 100))
    tts = OpenAITTS(api_key="k", http_client=server.client())
    with pytest.raises(ProviderError, match="16-bit"):
        await tts.synthesize("Hi.").collect()


async def test_a_lower_output_rate_is_resampled_client_side() -> None:
    server = SpeechServer(lambda r, i: audio_response(pcm(1.0)))
    tts = OpenAITTS(api_key="k", sample_rate=16_000, http_client=server.client())
    frame = await tts.synthesize("Telephony.").collect()
    assert frame.sample_rate == 16_000 and frame.duration == pytest.approx(1.0, abs=0.01)
    assert "sample_rate" not in server.bodies()[0]  # OpenAI's PCM is always 24 kHz


# ------------------------------------------------------------------------ failures
async def test_retries_before_the_first_audio_byte() -> None:
    audio = pcm(0.3)
    server = SpeechServer(
        lambda r, i: (
            httpx.Response(503, json={"error": {"message": "Overloaded"}})
            if i == 0
            else audio_response(audio)
        )
    )
    tts = OpenAITTS(api_key="k", http_client=server.client())
    frame = await tts.synthesize("Retry me.").collect()
    assert frame.data == audio and len(server.requests) == 2


async def test_no_retry_once_audio_was_played() -> None:
    async def broken() -> AsyncIterator[bytes]:
        yield pcm(0.1)
        raise httpx.ReadError("connection reset by peer")

    server = SpeechServer(lambda r, i: httpx.Response(200, content=broken()))
    tts = OpenAITTS(api_key="k", http_client=server.client())
    chunks_seen: list[AudioFrame] = []

    async def play() -> None:
        async for item in tts.synthesize("Cut short."):
            chunks_seen.append(item.frame)

    with pytest.raises(ProviderConnectionError, match="connection reset"):
        await play()
    assert len(server.requests) == 1
    assert sum(f.duration for f in chunks_seen) == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("status", "body", "error", "retryable"),
    [
        (401, {"error": {"message": "Incorrect API key", "code": "invalid_api_key"}},
         AuthenticationError, False),
        (429, {"error": {"message": "Rate limit", "code": "rate_limit_exceeded"}},
         RateLimitError, True),
        (429, {"error": {"message": "Quota", "code": "insufficient_quota"}}, RateLimitError,
         False),
        (400, {"error": {"message": "Invalid voice", "param": "voice"}}, ProviderError, False),
        (404, {"detail": "Model not found"}, ProviderError, False),
    ],
)  # fmt: skip
async def test_http_errors_are_mapped(
    status: int, body: dict[str, Any], error: type[ProviderError], retryable: bool
) -> None:
    server = SpeechServer(lambda r, i: httpx.Response(status, json=body))
    tts = OpenAITTS(api_key="k", max_retries=0, http_client=server.client())
    with pytest.raises(error) as info:
        await tts.synthesize("Hello.").collect()
    assert info.value.status_code == status and info.value.retryable is retryable
    assert info.value.provider == "openai"
    message = next(iter(body.values()))
    detail = message["message"] if isinstance(message, dict) else message
    assert detail in str(info.value)
    if status == 404:
        assert "check the model id" in str(info.value)


async def test_network_errors_are_connection_errors() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    tts = OpenAITTS(api_key="k", max_retries=1,
                    http_client=httpx.AsyncClient(transport=httpx.MockTransport(fail)))  # fmt: skip
    with pytest.raises(ProviderConnectionError, match="cannot reach") as info:
        await tts.synthesize("Hello.").collect()
    assert info.value.retryable


# ------------------------------------------------------------------ text handling
def test_split_text_keeps_requests_under_the_limit() -> None:
    assert split_text("  Short one.  ", 4096) == ["Short one."]
    assert split_text("   ", 10) == []
    text = "First sentence here. Second one is longer. Third!"
    parts = split_text(text, 25)
    assert parts == ["First sentence here.", "Second one is longer.", "Third!"]
    assert split_text("x" * 30, 12) == ["x" * 12, "x" * 12, "x" * 6]
    words = split_text("aaaa bbbb cccc dddd", 10)
    assert all(len(p) <= 10 for p in words) and " ".join(words) == "aaaa bbbb cccc dddd"


async def test_long_texts_are_synthesized_in_several_requests() -> None:
    server = SpeechServer(lambda r, i: audio_response(pcm(0.1)))
    tts = OpenAITTS(api_key="k", http_client=server.client())
    sentence = "This sentence is exactly fifty characters long ok. "
    text = (sentence * 100).strip()  # 5 000 characters
    frame = await tts.synthesize(text).collect()
    inputs = [b["input"] for b in server.bodies()]
    assert len(inputs) == 2 and all(len(i) <= 4096 for i in inputs)
    assert " ".join(inputs) == text
    assert frame.duration == pytest.approx(0.2)


async def test_empty_text_makes_no_request() -> None:
    server = SpeechServer(lambda r, i: audio_response(pcm(0.1)))
    tts = OpenAITTS(api_key="k", http_client=server.client())
    frame = await tts.synthesize(" ").collect()
    assert not frame and server.requests == []


async def test_streaming_synthesizes_sentence_by_sentence() -> None:
    def respond(request: httpx.Request, index: int) -> httpx.Response:
        text = json.loads(request.content)["input"]
        return audio_response(pcm(len(text) * 0.01, level=1000 + 100 * index))

    server = SpeechServer(respond)
    tts = OpenAITTS(api_key="k", http_client=server.client())
    assert not tts.capabilities.streaming
    stream = tts.stream()
    stream.push_text("Hello there. How are ")
    stream.push_text("you doing today?")
    stream.end_input()
    items = [item async for item in stream]
    await stream.aclose()
    assert [b["input"] for b in server.bodies()] == ["Hello there.", "How are you doing today?"]
    texts = [i.text for i in items if i.text]
    assert texts == ["Hello there.", "How are you doing today?"]
    assert [i.is_final for i in items].count(True) == 1
    total = sum(i.frame.duration for i in items)
    assert total == pytest.approx((12 + 23) * 0.01, abs=0.01)


async def test_cascade_speaks_through_the_speech_endpoint() -> None:
    server = SpeechServer(
        lambda r, i: audio_response(pcm(len(json.loads(r.content)["input"]) * 0.01))
    )
    tts = OpenAITTS(api_key="k", http_client=server.client())
    engine = CascadeEngine(stt=MockSTT(), llm=MockLLM(), tts=tts, vad=EnergyVAD())
    await engine.warmup()  # GET /models opens the connection
    conn = await engine.connect(EngineOptions())
    await conn.say("Hello there. How are you today?")
    events: list[object] = []

    async def drain() -> None:
        async for ev in conn.events():
            events.append(ev)
            if isinstance(ev, ResponseDone):
                return

    await asyncio.wait_for(drain(), 5)
    await conn.aclose()
    await engine.aclose()
    done = events[-1]
    assert isinstance(done, ResponseDone) and done.status == "completed"
    frames = [e.frame for e in events if isinstance(e, ResponseAudio)]
    assert all(f.sample_rate == 24_000 for f in frames)
    assert AudioFrame.concat(frames).duration == pytest.approx(0.12 + 0.18, abs=0.01)
    spoken = "".join(e.delta for e in events if isinstance(e, ResponseText))
    assert spoken.split() == ["Hello", "there.", "How", "are", "you", "today?"]
    assert [r.method for r in server.requests][:1] == ["GET"]
    assert [b["input"] for b in server.bodies()] == ["Hello there.", "How are you today?"]


# -------------------------------------------------------------- setup and registry
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"speed": 5.0}, "speed"),
        ({"response_format": "mp3"}, "response_format"),
        ({"sample_rate": 0}, "sample_rate"),
    ],
)
def test_invalid_options(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        OpenAITTS(api_key="k", **kwargs)


def test_registry_and_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        create("tts", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    tts = create("tts", "openai")
    assert isinstance(tts, OpenAITTS)
    assert (tts.model, tts.voice, tts.sample_rate) == ("gpt-4o-mini-tts", "marin", 24_000)
    assert tts.endpoint.request_headers() == {"Authorization": "Bearer sk-env"}
    hd = create("tts", {"provider": "openai/tts-1-hd", "voice": "nova", "speed": 1.2})
    assert (hd.model, hd.voice, hd.speed) == ("tts-1-hd", "nova", 1.2)
    other = OpenAITTS(base_url="http://127.0.0.1:9/v1")  # the OpenAI key stays with OpenAI
    assert other.endpoint.request_headers() == {}


async def test_compatible_tts_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")  # never sent to these hosts
    server = SpeechServer(lambda r, i: audio_response(pcm(0.1)))

    speaches = create("tts", "speaches", http_client=server.client())
    assert isinstance(speaches, SpeachesTTS)
    await speaches.synthesize("Local speech.").collect()
    kokoro = create("tts", "kokoro-fastapi/kokoro", voice="af_bella(2)+af_sky(1)",
                    http_client=server.client())  # fmt: skip
    assert isinstance(kokoro, KokoroFastAPITTS)
    await kokoro.synthesize("Kokoro speech.").collect()
    monkeypatch.setenv("KOKORO_FASTAPI_BASE_URL", "http://tts-box:8880/v1")
    monkeypatch.setenv("KOKORO_FASTAPI_API_KEY", "kf-key")
    remote = create("tts", "kokoro_fastapi", voice="voice_x", http_client=server.client())
    await remote.synthesize("Remote.").collect()

    urls = [str(r.url) for r in server.requests]
    assert urls == [
        "http://localhost:8000/v1/audio/speech",
        "http://localhost:8880/v1/audio/speech",
        "http://tts-box:8880/v1/audio/speech",
    ]
    assert [r.headers.get("authorization") for r in server.requests] == [
        None,
        None,
        "Bearer kf-key",
    ]
    assert server.bodies() == [
        {"model": "speaches-ai/Kokoro-82M-v1.0-ONNX", "input": "Local speech.",
         "response_format": "pcm", "voice": "af_heart", "sample_rate": 24_000},
        {"model": "kokoro", "input": "Kokoro speech.", "response_format": "pcm",
         "voice": "af_bella(2)+af_sky(1)"},
        {"model": "kokoro", "input": "Remote.", "response_format": "pcm", "voice": "voice_x"},
    ]  # fmt: skip


def test_speaches_base_url_may_be_the_engine_websocket_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPEACHES_BASE_URL", "ws://localhost:8000/v1")  # shared with the engine
    assert SpeachesTTS().base_url == "http://localhost:8000/v1"
    assert LocalAITTS(base_url="wss://ai.example.com/v1").base_url == "https://ai.example.com/v1"


async def test_azure_tts_endpoint_and_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com/")
    with pytest.raises(ConfigurationError, match="AZURE_OPENAI_API_KEY"):
        AzureOpenAITTS()
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
    server = SpeechServer(lambda r, i: audio_response(pcm(0.1)))
    tts = create("tts", "azure_openai/my-tts", http_client=server.client())
    assert isinstance(tts, AzureOpenAITTS) and tts.voice == "alloy"
    await tts.synthesize("From Azure.").collect()
    request = server.requests[0]
    assert str(request.url) == "https://res.openai.azure.com/openai/v1/audio/speech"
    assert request.headers["api-key"] == "az-key" and "authorization" not in request.headers
    assert server.bodies()[0]["model"] == "my-tts"
    token = AzureOpenAITTS(azure_ad_token="entra")
    assert token.endpoint.request_headers() == {"Authorization": "Bearer entra"}


async def test_warmup_opens_the_connection_or_loads_the_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def respond(request: httpx.Request, index: int) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"object": "list", "data": []})
        return audio_response(pcm(0.1))

    server = SpeechServer(respond)
    await OpenAITTS(api_key="k", http_client=server.client()).warmup()
    await SpeachesTTS(http_client=server.client()).warmup()  # loads the model on demand
    assert [(r.method, r.url.path) for r in server.requests] == [
        ("GET", "/v1/models"),
        ("POST", "/v1/audio/speech"),
    ]
    assert server.bodies()[0]["input"] == "Hi."

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    broken = OpenAITTS(api_key="k", http_client=httpx.AsyncClient(
        transport=httpx.MockTransport(fail)))  # fmt: skip
    with caplog.at_level(logging.WARNING, logger="voice_agent_next"):
        await broken.warmup()  # logged, not raised
    assert "warmup failed" in caplog.text


# ------------------------------------------------------------------- real API (opt-in)
needs_key = pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")


@pytest.mark.integration
@needs_key
async def test_integration_speech_and_sentence_streaming() -> None:
    tts = OpenAITTS(instructions="Speak in a calm, friendly tone.")
    audio = await tts.synthesize("Hello from voice agent next.").collect()
    assert audio.sample_rate == 24_000 and audio.duration > 0.8
    stream = tts.stream()
    stream.push_text("Hello there. ")
    stream.push_text("How are you today?")
    stream.end_input()
    items = [item async for item in stream]
    await stream.aclose()
    await tts.aclose()
    assert sum(i.frame.duration for i in items) > 1.0
    assert [i.is_final for i in items].count(True) == 1
