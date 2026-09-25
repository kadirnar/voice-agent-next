"""``nemo-speech-cpp`` provider against a fake ``nemo-speech serve`` (NeMo-Speech.cpp v0.1.0).

The fake replays the protocol of the real server, as recorded on an RTX 5070 Ti with
``nemotron-speech-streaming-en-0.6b.q8_0.gguf`` (``server/http/http_server.cpp`` upstream):

* ``GET /v1/audio/transcriptions/realtime`` upgrades to a WebSocket and sends
  ``session.created`` (``{"input_audio_format": "pcm16", "model": ..., "sample_rate": 16000}``);
  ``session.update`` is echoed back as ``session.updated``;
* every binary PCM16 frame is answered with one
  ``conversation.item.input_audio_transcription.delta`` whose ``delta`` is the new text
  suffix, mostly ``""``, plus ``audio_processed`` (seconds of the current stream);
* ``input_audio_buffer.commit`` -> ``...completed`` (``transcript``; ``words`` with
  ``word_timestamps``; ``""`` when nothing was said) then ``input_audio_buffer.committed``;
  the recognition stream restarts, so word times start from 0 again;
* with server endpointing, ``...completed`` also arrives mid-stream after a pause;
* unknown events -> ``error`` (``invalid_request_error``); the socket stays open;
* a wrong ``--api-key`` -> the socket is accepted, then closed with 1008.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.http11 import Request, Response

from voice_agent_next.audio.frame import AudioFrame
from voice_agent_next.audio.wav import read_wav, wav_bytes
from voice_agent_next.errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderError,
)
from voice_agent_next.metrics import STTMetrics
from voice_agent_next.models import get_model
from voice_agent_next.providers import nemo_speech_cpp
from voice_agent_next.providers.nemo_speech_cpp import (
    NeMoSpeechCppServer,
    NeMoSpeechCppSTT,
    NeMoSpeechCppTTS,
    find_executable,
    right_context,
)
from voice_agent_next.registry import create
from voice_agent_next.stt import STTEvent, STTEventType

RATE = 16_000
MODEL_ID = ".nemotron-speech-streaming-en-0.6b.q8_0.gguf"
WORD_EVERY = 0.16
"""The fake emits the next word every 160 ms of voiced audio (one cache-aware chunk)."""

# A real session (trimmed: the server sends one empty delta per 20 ms frame in between).
REAL_SESSION = [
    {"audio_processed": 0.98000001907348633, "delta": "In other", "type": "conversation.item.input_audio_transcription.delta"},
    {"audio_processed": 1, "delta": "", "type": "conversation.item.input_audio_transcription.delta"},
    {"audio_processed": 1.1399999856948853, "delta": " words", "type": "conversation.item.input_audio_transcription.delta"},
    {"audio_processed": 1.6200000047683716, "delta": ",", "type": "conversation.item.input_audio_transcription.delta"},
    {"audio_processed": 1.7799999713897705, "delta": " while", "type": "conversation.item.input_audio_transcription.delta"},
    {"audio_processed": 1.940000057220459, "delta": " he had", "type": "conversation.item.input_audio_transcription.delta"},
    {"audio_processed": 2.4200000762939453, "delta": " implicit", "type": "conversation.item.input_audio_transcription.delta"},
    {"audio_processed": 2.7400000095367432, "delta": " faith", "type": "conversation.item.input_audio_transcription.delta"},
    {"audio_processed": 3.0, "event_id": "event_639", "transcript": "In other words, while he had implicit faith.", "type": "conversation.item.input_audio_transcription.completed",
     "words": [{"confidence": 1, "end": 0.88, "start": 0.80000000000000004, "word": "In"}, {"confidence": 1, "end": 0.95999999999999996, "start": 0.88, "word": "other"},
               {"confidence": 1, "end": 1.52, "start": 0.95999999999999996, "word": "words,"}, {"confidence": 1, "end": 1.6799999999999999, "start": 1.6000000000000001, "word": "while"},
               {"confidence": 1, "end": 1.8400000000000001, "start": 1.76, "word": "he"}, {"confidence": 1, "end": 1.9199999999999999, "start": 1.8400000000000001, "word": "had"},
               {"confidence": 1, "end": 2.3999999999999999, "start": 2.2400000000000002, "word": "implicit"}, {"confidence": 0.5, "end": 2.6400000000000001, "start": 2.5600000000000001, "word": "faith."}]},
    {"type": "input_audio_buffer.committed"},
]  # fmt: skip


def voiced(duration: float, amplitude: float = 0.3) -> AudioFrame:
    t = np.arange(round(duration * RATE)) / RATE
    return AudioFrame.from_numpy((amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32), RATE)


def silence(duration: float) -> AudioFrame:
    return AudioFrame.silence(duration, RATE)


class FakeNemoServer:
    """A scripted ``nemo-speech serve`` realtime endpoint (see the module docstring)."""

    def __init__(
        self,
        words: list[str] | None = None,
        *,
        api_key: str | None = None,
        endpointing_ms: float | None = None,
        replay: list[dict[str, Any]] | None = None,
    ) -> None:
        self.words = list(words or ["hello", "world", "again", "and", "again"])
        self.api_key = api_key
        self.endpointing_ms = endpointing_ms
        self.replay = replay
        self.sessions: list[dict[str, Any]] = []
        self.received: list[str] = []
        self.headers: list[dict[str, str]] = []
        self.frames = 0
        self._server: Server | None = None
        self.port = 0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    async def __aenter__(self) -> FakeNemoServer:
        self._server = await serve(
            self._handle, "127.0.0.1", 0, process_request=self._route, compression=None
        )
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    @staticmethod
    def _route(conn: ServerConnection, request: Request) -> Response | None:
        if request.path.split("?")[0] != "/v1/audio/transcriptions/realtime":
            return conn.respond(404, "Not Found\n")
        return None

    async def _handle(self, ws: ServerConnection) -> None:
        assert ws.request is not None
        self.headers.append(dict(ws.request.headers.raw_items()))
        if self.api_key and ws.request.headers.get("Authorization") != f"Bearer {self.api_key}":
            await ws.close(1008, "invalid bearer token")
            return
        seq = iter(range(1, 1_000_000))

        async def send(event: dict[str, Any]) -> None:
            event.setdefault("event_id", f"event_{next(seq)}")
            await ws.send(json.dumps(event))

        await send(
            {
                "type": "session.created",
                "session": {"input_audio_format": "pcm16", "model": MODEL_ID, "sample_rate": RATE},
            }
        )
        said: list[str] = []
        words_iter = iter(self.words)
        processed = 0.0
        voiced_s = 0.0
        quiet_s = 0.0
        word_times: list[tuple[str, float, float]] = []
        async for message in ws:
            if isinstance(message, bytes):
                self.frames += 1
                samples = np.frombuffer(message, dtype=np.int16)
                duration = len(samples) / RATE
                loud = bool(samples.size) and float(np.abs(samples).max()) > 1000
                processed += duration
                delta = ""
                if loud:
                    quiet_s = 0.0
                    voiced_s += duration
                    if voiced_s >= WORD_EVERY - 1e-9:
                        voiced_s = 0.0
                        word = next(words_iter, None)
                        if word is not None:
                            delta = word if not said else f" {word}"
                            said.append(word)
                            word_times.append((word, processed - WORD_EVERY, processed))
                else:
                    quiet_s += duration
                if self.replay is None:
                    await send(
                        {
                            "type": "conversation.item.input_audio_transcription.delta",
                            "delta": delta,
                            "audio_processed": processed,
                        }
                    )
                ms = self.endpointing_ms
                if ms is not None and said and quiet_s * 1000 >= ms:
                    await send(self._completed(said, word_times, processed))
                    said, word_times, quiet_s = [], [], 0.0
                continue
            event = json.loads(message)
            self.received.append(event.get("type", ""))
            kind = event.get("type")
            if kind == "session.update":
                self.sessions.append(event["session"])
                await send({"type": "session.updated", "session": event["session"]})
            elif kind == "input_audio_buffer.commit":
                if self.replay is not None:
                    for replayed in self.replay:
                        await send(dict(replayed))
                    continue
                await send(self._completed(said, word_times, processed))
                await send({"type": "input_audio_buffer.committed"})
                said, word_times, processed, voiced_s = [], [], 0.0, 0.0
                words_iter = iter(self.words)
            elif kind in ("input_audio_buffer.clear", "response.cancel"):
                said, word_times, processed = [], [], 0.0
                await send({"type": "input_audio_buffer.cleared"})
            else:
                await send(
                    {
                        "type": "error",
                        "error": {
                            "message": f"unsupported realtime event type: {kind}",
                            "type": "invalid_request_error",
                        },
                    }
                )

    def _completed(
        self, said: list[str], word_times: list[tuple[str, float, float]], processed: float
    ) -> dict[str, Any]:
        text = (" ".join(said).capitalize() + ".") if said else ""
        event: dict[str, Any] = {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": text,
            "audio_processed": processed,
        }
        if self.sessions and self.sessions[-1].get("word_timestamps"):
            event["words"] = [
                {"word": w, "start": s, "end": e, "confidence": 1} for w, s, e in word_times
            ]
        return event


@pytest.fixture
async def server() -> AsyncIterator[FakeNemoServer]:
    async with FakeNemoServer() as srv:
        yield srv


async def push(stream: Any, frame: AudioFrame, chunk: float = 0.02) -> None:
    step = round(chunk * frame.sample_rate)
    data = frame.data
    for i in range(0, len(data), step * 2):
        stream.push_audio(AudioFrame(data[i : i + step * 2], frame.sample_rate, 1))
        await asyncio.sleep(0)


async def collect(stream: Any, until: int = 1, timeout: float = 10.0) -> list[STTEvent]:
    """Events until ``until`` finals arrived."""
    events: list[STTEvent] = []

    async def run() -> None:
        async for ev in stream:
            events.append(ev)
            if sum(e.type == STTEventType.FINAL_TRANSCRIPT for e in events) >= until:
                return

    await asyncio.wait_for(run(), timeout)
    return events


def kinds(events: list[STTEvent]) -> list[STTEventType]:
    return [e.type for e in events]


# --------------------------------------------------------------------------- streaming
async def test_stream_interims_then_a_final_on_flush(server: FakeNemoServer) -> None:
    stt = NeMoSpeechCppSTT(base_url=server.base_url, language="en-US")
    metrics: list[STTMetrics] = []
    stt.on("metrics", metrics.append)
    stream = stt.stream()
    await push(stream, voiced(0.5))
    stream.flush()
    events = await collect(stream)
    await stream.aclose()
    await stt.aclose()

    assert kinds(events) == [
        STTEventType.START_OF_SPEECH,
        STTEventType.INTERIM_TRANSCRIPT,
        STTEventType.INTERIM_TRANSCRIPT,
        STTEventType.INTERIM_TRANSCRIPT,
        STTEventType.FINAL_TRANSCRIPT,
    ]
    assert [e.text for e in events[1:4]] == ["hello", "hello world", "hello world again"]
    assert events[-1].text == "Hello world again."
    assert events[-1].transcript is not None and events[-1].transcript.language == "en-US"
    assert len({e.segment_id for e in events}) == 1
    assert server.sessions == [
        {"sample_rate": 16000, "language": "en-US", "automatic_punctuation": True}
    ]
    assert server.received[:2] == ["session.update", "input_audio_buffer.commit"]
    streamed = [m for m in metrics if m.streamed]
    assert streamed and streamed[0].latency is not None and streamed[0].audio_duration > 0.4
    assert server.frames == 25  # 20 ms frames by default


async def test_end_of_speech_follows_the_final(server: FakeNemoServer) -> None:
    stt = NeMoSpeechCppSTT(base_url=server.base_url)
    stream = stt.stream()
    await push(stream, voiced(0.2))
    stream.end_input()
    events = [ev async for ev in stream]
    await stream.aclose()
    assert kinds(events)[-2:] == [STTEventType.FINAL_TRANSCRIPT, STTEventType.END_OF_SPEECH]
    assert events[-1].segment_id == events[-2].segment_id


async def test_a_flush_without_speech_is_answered_with_an_empty_final(
    server: FakeNemoServer,
) -> None:
    stt = NeMoSpeechCppSTT(base_url=server.base_url)
    stream = stt.stream()
    await push(stream, silence(0.1))
    stream.flush()
    events = await collect(stream)
    await stream.aclose()
    assert kinds(events) == [STTEventType.FINAL_TRANSCRIPT]
    assert events[0].text == ""


async def test_commits_restart_recognition_and_word_times_stay_on_the_stream_clock(
    server: FakeNemoServer,
) -> None:
    stt = NeMoSpeechCppSTT(base_url=server.base_url, word_timestamps=True)
    assert stt.capabilities.word_timestamps
    stream = stt.stream()
    await push(stream, voiced(0.32))
    stream.flush()
    first = await collect(stream)
    await push(stream, silence(0.2))
    await push(stream, voiced(0.32))
    stream.flush()
    second = await collect(stream)
    await stream.aclose()
    assert server.sessions[0]["word_timestamps"] is True
    f1, f2 = first[-1].transcript, second[-1].transcript
    assert f1 is not None and f2 is not None
    assert f1.text == "Hello world." and f2.text == "Hello world."  # a fresh stream
    assert f1.words is not None and f2.words is not None
    assert f1.words[0].start == pytest.approx(0.0, abs=1e-6)
    # the second recognition began 0.32 s into the stream; its words are shifted by it
    assert f2.words[0].start == pytest.approx(0.32 + 0.2, abs=1e-6)
    assert f2.start_time == f2.words[0].start and f2.end_time == f2.words[-1].end
    assert f2.confidence == 1


async def test_server_endpointing_finals_arrive_mid_stream() -> None:
    async with FakeNemoServer(endpointing_ms=300) as srv:
        stt = NeMoSpeechCppSTT(base_url=srv.base_url, endpointing_ms=300)
        stream = stt.stream()
        await push(stream, voiced(0.32))
        await push(stream, silence(0.4))
        events = await collect(stream)  # no flush: the server's endpoint ended the utterance
        assert events[-1].text == "Hello world."
        assert srv.sessions[0]["endpointing_ms"] == 300
        more: list[STTEvent] = []
        stream.flush()  # nothing left: answered with an empty final
        more = await collect(stream)
        await stream.aclose()
    assert kinds(more) == [STTEventType.END_OF_SPEECH, STTEventType.FINAL_TRANSCRIPT]
    assert more[-1].text == ""


async def test_a_real_server_session_replays() -> None:
    async with FakeNemoServer(replay=REAL_SESSION) as srv:
        stt = NeMoSpeechCppSTT(base_url=srv.base_url, word_timestamps=True)
        stream = stt.stream()
        await push(stream, voiced(0.1))
        stream.flush()
        events = await collect(stream)
        await stream.aclose()
    interims = [e.text for e in events if e.type == STTEventType.INTERIM_TRANSCRIPT]
    assert interims == [
        "In other",
        "In other words",
        "In other words,",
        "In other words, while",
        "In other words, while he had",
        "In other words, while he had implicit",
        "In other words, while he had implicit faith",
    ]
    final = events[-1].transcript
    assert final is not None and final.text == "In other words, while he had implicit faith."
    assert final.words is not None and [w.word for w in final.words][:3] == [
        "In",
        "other",
        "words,",
    ]
    assert final.start_time == pytest.approx(0.8) and final.confidence == pytest.approx(0.9375)


async def test_session_options_are_sent() -> None:
    async with FakeNemoServer() as srv:
        stt = NeMoSpeechCppSTT(
            base_url=srv.base_url,
            language="auto",
            automatic_punctuation=False,
            verbatim=True,
            profanity_filter=True,
            speech_contexts=[{"phrases": ["Nemotron"], "boost": 3.0}],
            prompt="Kowalczyk",
            chunk_ms=160,
        )
        assert stt.capabilities.language_detection
        stream = stt.stream()
        await push(stream, voiced(0.32))
        stream.flush()
        await collect(stream)
        await stream.aclose()
    assert srv.sessions == [
        {
            "sample_rate": 16000,
            "language": "auto",
            "automatic_punctuation": False,
            "verbatim": True,
            "profanity_filter": True,
            "speech_contexts": [{"phrases": ["Nemotron"], "boost": 3.0}],
            "prompt": "Kowalczyk",
        }
    ]
    assert srv.frames == 2  # 160 ms frames


async def test_an_error_event_ends_the_stream(
    server: FakeNemoServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    stt = NeMoSpeechCppSTT(base_url=server.base_url)
    ws_send = nemo_speech_cpp.NeMoSpeechCppStream._send_loop

    async def bad_send(self: Any, ws: Any) -> None:
        await ws.send(json.dumps({"type": "bogus"}))
        await ws_send(self, ws)

    monkeypatch.setattr(nemo_speech_cpp.NeMoSpeechCppStream, "_send_loop", bad_send)
    stream = stt.stream()
    with pytest.raises(ProviderError, match="unsupported realtime event type: bogus"):
        async for _ in stream:
            pass
    await stream.aclose()


async def test_the_api_key_is_sent_and_a_wrong_one_is_rejected() -> None:
    async with FakeNemoServer(api_key="s3cret") as srv:
        ok = NeMoSpeechCppSTT(base_url=srv.base_url, api_key="s3cret").stream()
        await push(ok, voiced(0.2))
        ok.flush()
        await collect(ok)
        await ok.aclose()
        assert srv.headers[0]["Authorization"] == "Bearer s3cret"

        bad = NeMoSpeechCppSTT(base_url=srv.base_url, api_key="wrong").stream()
        await push(bad, voiced(0.1))
        with pytest.raises(AuthenticationError, match="invalid bearer token"):
            await collect(bad)
        await bad.aclose()


async def test_the_api_key_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEMO_SPEECH_API_KEY", "from-env")
    monkeypatch.setenv("NEMO_SPEECH_BASE_URL", "http://127.0.0.1:9/v1")
    stt = NeMoSpeechCppSTT()
    assert stt.endpoint.request_headers() == {"Authorization": "Bearer from-env"}
    assert stt.realtime_url() == "ws://127.0.0.1:9/v1/audio/transcriptions/realtime"
    https = NeMoSpeechCppSTT(base_url="https://asr.example/v1", api_key="")
    assert https.realtime_url() == "wss://asr.example/v1/audio/transcriptions/realtime"
    assert https.endpoint.request_headers() == {}


async def test_no_server_is_a_connection_error() -> None:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    stt = NeMoSpeechCppSTT(base_url=f"http://127.0.0.1:{port}/v1", connect_timeout=5.0)
    stream = stt.stream()
    with pytest.raises((ProviderConnectionError, ProviderError)):
        await collect(stream, timeout=30.0)
    await stream.aclose()


async def test_a_server_without_asr_answers_404(server: FakeNemoServer) -> None:
    stt = NeMoSpeechCppSTT(base_url=server.base_url.replace("/v1", "/v2"))
    stream = stt.stream()
    with pytest.raises(ProviderError, match="404"):
        await collect(stream)
    await stream.aclose()


# ------------------------------------------------------------------------------- batch
def transcription_handler(
    seen: list[httpx.Request], body: dict[str, Any] | None = None, status: int = 200
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if status != 200:
            return httpx.Response(
                status,
                json={"error": {"message": "ASR is not loaded", "type": "server_error"}},
            )
        return httpx.Response(200, json=body or {"text": "Hello world."})

    return httpx.MockTransport(handler)


def form_fields(request: httpx.Request) -> dict[str, list[str]]:
    content = request.read().decode("latin-1")
    boundary = request.headers["content-type"].split("boundary=")[1]
    fields: dict[str, list[str]] = {}
    for part in content.split(f"--{boundary}"):
        if 'name="' not in part or 'name="file"' in part:
            continue
        head, _, value = part.partition("\r\n\r\n")
        name = head.split('name="')[1].split('"')[0]
        fields.setdefault(name, []).append(value.rstrip("\r\n"))
    return fields


async def test_transcribe_posts_the_openai_compatible_form() -> None:
    seen: list[httpx.Request] = []
    body = {
        "duration": 2.64,
        "language": "en",
        "task": "transcribe",
        "text": "And what through the left hand window.",
        "words": [
            {"confidence": 1, "end": 0.24, "start": 0.16, "word": "And"},
            {"confidence": 1, "end": 1.2, "start": 1.12, "word": "what"},
        ],
    }
    client = httpx.AsyncClient(transport=transcription_handler(seen, body))
    stt = NeMoSpeechCppSTT(
        base_url="http://127.0.0.1:8080/v1",
        word_timestamps=True,
        automatic_punctuation=False,
        speech_contexts=[{"phrases": ["Kowalczyk"], "boost": 3.0}],
        language="en-US",
        http_client=client,
    )
    result = await stt.transcribe(voiced(0.5))
    await client.aclose()
    assert result.text == "And what through the left hand window."
    assert result.words is not None and result.words[1].start == 1.12
    request = seen[0]
    assert str(request.url) == "http://127.0.0.1:8080/v1/audio/transcriptions"
    fields = form_fields(request)
    assert fields["response_format"] == ["verbose_json"]
    assert fields["language"] == ["en-US"]
    assert fields["model"] == ["nemotron-en"]
    assert fields["automatic_punctuation"] == ["false"]
    assert json.loads(fields["speech_contexts"][0]) == [{"phrases": ["Kowalczyk"], "boost": 3.0}]


async def test_batch_errors_are_mapped() -> None:
    seen: list[httpx.Request] = []
    client = httpx.AsyncClient(transport=transcription_handler(seen, status=500))
    stt = NeMoSpeechCppSTT(base_url="http://127.0.0.1:8080/v1", http_client=client)
    with pytest.raises(ProviderError, match="ASR is not loaded"):
        await stt.transcribe(voiced(0.2))
    await client.aclose()


def test_offline_only_models_are_batch() -> None:
    stt = NeMoSpeechCppSTT(model="parakeet-tdt")
    assert not stt.capabilities.streaming and not stt.streaming
    with pytest.raises(ConfigurationError, match="offline-only"):
        NeMoSpeechCppSTT(model="parakeet-tdt", streaming=True)
    assert NeMoSpeechCppSTT(model="nemotron-3.5").capabilities.streaming
    assert not NeMoSpeechCppSTT(streaming=False).capabilities.interim_results
    with pytest.raises(ConfigurationError):
        NeMoSpeechCppSTT(chunk_ms=0)


def test_registry_and_model_catalog() -> None:
    stt = create("stt", "nemo-speech-cpp/nemotron-3.5")
    assert isinstance(stt, NeMoSpeechCppSTT) and stt.model == "nemotron-3.5"
    assert stt.sample_rate == 16_000
    tts = create("tts", "nemo-speech-cpp")
    assert isinstance(tts, NeMoSpeechCppTTS) and tts.model == "magpie"
    info = get_model("nemo-speech-cpp/nemotron-en")
    assert info.files[0].location == "nvidia/nemotron-speech-streaming-en-0.6b"
    assert info.files[0].sha256 is not None and info.size == 699_872_960


# --------------------------------------------------------------------------------- TTS
def pcm_wav(duration: float, rate: int) -> bytes:
    t = np.arange(round(duration * rate)) / rate
    frame = AudioFrame.from_numpy((0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), rate)
    return wav_bytes(frame)


async def test_tts_posts_json_and_resamples_the_wav() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/audio/speech"
        seen.append(json.loads(request.content))
        return httpx.Response(
            200, content=pcm_wav(0.5, 22_050), headers={"content-type": "audio/wav"}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tts = NeMoSpeechCppTTS(
        base_url="http://127.0.0.1:8080/v1",
        voice="John",
        language="en-US",
        sample_rate=24_000,
        http_client=client,
        trim_silence=False,
    )
    frames = [a.frame async for a in tts.synthesize("Hello there.") if a.frame.duration > 0]
    await client.aclose()
    audio = AudioFrame.concat(frames)
    assert audio.sample_rate == 24_000
    assert audio.duration == pytest.approx(0.5, abs=0.02)
    assert seen == [
        {
            "model": "magpie",
            "input": "Hello there.",
            "response_format": "wav",
            "voice": "John",
            "language": "en-US",
        }
    ]


async def test_tts_defaults_to_22khz_and_the_server_voice() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=pcm_wav(0.2, 22_050))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tts = NeMoSpeechCppTTS(http_client=client, trim_silence=False)
    frames = [a.frame async for a in tts.synthesize("Hi.") if a.frame.duration > 0]
    await client.aclose()
    assert tts.sample_rate == 22_050 and frames[0].sample_rate == 22_050
    assert "voice" not in seen[0] and "speed" not in seen[0] and "sample_rate" not in seen[0]
    assert read_wav(pcm_wav(0.2, 22_050)).sample_rate == 22_050


async def test_tts_errors() -> None:
    with pytest.raises(ConfigurationError, match="speed"):
        NeMoSpeechCppTTS(speed=1.5)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500, json={"error": {"message": "TTS is not loaded", "type": "server_error"}}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tts = NeMoSpeechCppTTS(http_client=client, max_retries=0)
    with pytest.raises(ProviderError, match="TTS is not loaded"):
        async for _ in tts.synthesize("Hi."):
            pass
    await client.aclose()


# ------------------------------------------------------------------------------ server
def test_server_command_line(tmp_path: Path) -> None:
    srv = NeMoSpeechCppServer(
        asr_model="nemotron-en",
        tts_model="magpie",
        device="cuda:0",
        chunk_ms=560,
        endpointing=True,
        endpointing_ms=500,
        port=8123,
        args=["--threads", "8"],
    )
    assert srv.command("nemo-speech", "/m/asr.gguf") == [
        "nemo-speech", "serve", "--host", "127.0.0.1", "--port", "8123", "--no-ui",
        "--asr-model", "/m/asr.gguf",
        "--asr.streaming.rnnt_right_context=6",
        "--asr.endpointing.enable=true",
        "--asr.endpointing.stop_history_eou_ms=500",
        "--tts-model", "magpie", "--device", "cuda:0", "--threads", "8",
    ]  # fmt: skip
    assert srv.base_url == "http://127.0.0.1:8123/v1"
    assert [right_context(c) for c in (80, 160, 560, 1120)] == [0, 1, 6, 13]
    with pytest.raises(ConfigurationError, match="chunk_ms"):
        NeMoSpeechCppServer(asr_model="nemotron-en", chunk_ms=100)
    with pytest.raises(ConfigurationError):
        NeMoSpeechCppServer()
    # local paths and names the server resolves itself are passed through, not downloaded
    gguf = tmp_path / "custom.gguf"
    gguf.write_bytes(b"GGUF")
    assert NeMoSpeechCppServer(asr_model=gguf).resolve_asr_model() == str(gguf)
    assert NeMoSpeechCppServer(asr_model="nvidia/other").resolve_asr_model() == "nvidia/other"
    assert NeMoSpeechCppServer(tts_model="magpie").resolve_asr_model() is None


def test_find_executable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    exe = tmp_path / "nemo-speech"
    exe.write_text("")
    monkeypatch.setenv("NEMO_SPEECH_BIN", str(exe))
    assert find_executable() == exe
    assert find_executable(exe) == exe
    with pytest.raises(ConfigurationError, match="does not exist"):
        find_executable(tmp_path / "missing")
    monkeypatch.delenv("NEMO_SPEECH_BIN")
    monkeypatch.setattr(nemo_speech_cpp.shutil, "which", lambda name: None)
    monkeypatch.setattr(nemo_speech_cpp.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    with pytest.raises(ConfigurationError, match="install"):
        find_executable()


FAKE_SERVER = """
import http.server, sys, time
port = int(sys.argv[sys.argv.index("--port") + 1])
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"capabilities":["asr"],"device":"auto","ready":true}'
        self.send_response(200 if self.path == "/ready" else 404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a):
        pass
time.sleep(0.3)  # "loading the model"
print("key=" + str(__import__("os").environ.get("NEMO_SPEECH_HTTP_API_KEY")), flush=True)
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
"""


async def test_managed_server_starts_and_stops(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER)
    srv = NeMoSpeechCppServer(
        asr_model="nvidia/other", executable=sys.executable, api_key="k1", startup_timeout=30
    )
    original = srv.command

    def command(exe: Any, asr_model: str | None = None) -> list[str]:
        cmd = original(exe, asr_model)
        assert "k1" not in cmd  # the key goes through the environment
        return [sys.executable, str(script), *cmd[2:]]

    monkeypatch.setattr(srv, "command", command)
    async with srv:
        assert srv.running
        assert await srv.start() == srv.base_url  # already running: no-op
        assert "key=k1" in srv.log_tail()
    assert not srv.running


async def test_managed_server_reports_a_failed_start(monkeypatch: pytest.MonkeyPatch) -> None:
    srv = NeMoSpeechCppServer(asr_model="x", executable=sys.executable, startup_timeout=30)
    monkeypatch.setattr(
        srv,
        "command",
        lambda exe, asr_model=None: [
            sys.executable,
            "-c",
            "print('error: unknown model x'); raise SystemExit(2)",
        ],
    )
    with pytest.raises(ProviderError, match="unknown model x"):
        await srv.start()
    assert not srv.running


async def test_serve_true_owns_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def start(self: NeMoSpeechCppServer) -> str:
        calls.append(f"start {self.asr_model} {self.tts_model} {self.chunk_ms}")
        return self.base_url

    async def stop(self: NeMoSpeechCppServer) -> None:
        calls.append("stop")

    monkeypatch.setattr(NeMoSpeechCppServer, "start", start)
    monkeypatch.setattr(NeMoSpeechCppServer, "stop", stop)
    stt = NeMoSpeechCppSTT(model="nemotron-3.5", serve=True, server_options={"chunk_ms": 80})
    assert stt.server is not None and stt.base_url == stt.server.base_url
    await stt._ensure_server()
    await stt.aclose()
    tts = NeMoSpeechCppTTS(serve=True)
    await tts._ensure_server()
    await tts.aclose()
    shared = NeMoSpeechCppServer(asr_model="nemotron-en", tts_model="magpie", api_key="k")
    stt2 = NeMoSpeechCppSTT(server=shared)
    assert stt2.endpoint.request_headers() == {"Authorization": "Bearer k"}
    await stt2._ensure_server()
    await stt2.aclose()  # not owned: not stopped
    assert calls == [
        "start nemotron-3.5 None 80",
        "stop",
        "start None magpie None",
        "stop",
        "start nemotron-en magpie None",
    ]
    with pytest.raises(ConfigurationError):
        NeMoSpeechCppSTT(serve=True, server=shared)
    with pytest.raises(ConfigurationError, match="base_url"):
        NeMoSpeechCppTTS(serve=True, base_url="http://x/v1")


# ------------------------------------------------------------------------- real server
@pytest.mark.integration
async def test_real_server_streaming() -> None:
    """Against a running ``nemo-speech serve`` (``NEMO_SPEECH_TEST_URL``, an ASR model)."""
    import os

    url = os.environ.get("NEMO_SPEECH_TEST_URL")
    if not url:
        pytest.skip("set NEMO_SPEECH_TEST_URL to a running nemo-speech serve (/v1)")
    from voice_agent_next.providers.mock import synth_speech

    stt = NeMoSpeechCppSTT(base_url=url)
    stream = stt.stream()
    await push(stream, synth_speech(1.0, RATE))
    stream.flush()
    events = await collect(stream, timeout=30)
    await stream.aclose()
    assert events[-1].type == STTEventType.FINAL_TRANSCRIPT
