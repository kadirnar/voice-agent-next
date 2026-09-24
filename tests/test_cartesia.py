"""Cartesia Sonic TTS and Ink STT against fake servers replaying Cartesia's protocol.

The fakes speak the real message shapes (API version 2026-08-14): TTS ``chunk`` /
``timestamps`` / ``done`` / ``error`` messages keyed by ``context_id`` with ``cancel``
requests; STT ``connected`` / ``turn.*`` events on ``/stt/turns/websocket`` and
``transcript`` / ``flush_done`` / ``done`` on ``/stt/websocket``.
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import numpy as np
import pytest
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from voice_agent_next import AudioFrame, create
from voice_agent_next.engine import EngineOptions
from voice_agent_next.engines.cascade import CascadeEngine
from voice_agent_next.errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from voice_agent_next.events import (
    InputCommitted,
    InputSpeechStarted,
    InputSpeechStopped,
    InputTranscript,
    ResponseAudio,
    ResponseDone,
    ResponseStarted,
    ResponseText,
)
from voice_agent_next.metrics import STTMetrics, TTSMetrics
from voice_agent_next.providers.cartesia import (
    API_VERSION,
    DEFAULT_VOICE,
    CartesiaSTT,
    CartesiaTTS,
)
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import MockLLM, MockSTT, MockTTS, synth_speech
from voice_agent_next.registry import get_provider
from voice_agent_next.stt import STTEvent, STTEventType, STTStream
from voice_agent_next.tts import SynthesizedAudio, SynthesizeStream

KEY = "sk_car_test"
WORD = 0.1
"""Seconds of audio the fake TTS generates per word."""

_PROXY_VARS = ("http_proxy", "https_proxy", "all_proxy", "ws_proxy", "wss_proxy")


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """websockets honours proxy variables; the fakes live on 127.0.0.1."""
    for var in _PROXY_VARS:
        monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv(var.upper(), raising=False)


def pcm(value: int, seconds: float, rate: int) -> bytes:
    return np.full(round(seconds * rate), value, dtype="<i2").tobytes()


async def wait_until(predicate: Callable[[], object], timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


async def collect_tts(stream: SynthesizeStream, timeout: float = 5.0) -> list[SynthesizedAudio]:
    async def run() -> list[SynthesizedAudio]:
        return [item async for item in stream]

    return await asyncio.wait_for(run(), timeout)


def audio_of(items: list[SynthesizedAudio]) -> AudioFrame:
    return AudioFrame.concat([i.frame for i in items if i.frame])


def query(request: Request) -> dict[str, list[str]]:
    return parse_qs(urlsplit(request.path).query)


def path(request: Request) -> str:
    return urlsplit(request.path).path


# ------------------------------------------------------------------------ fake TTS
@dataclass
class FakeTTSServer:
    """Replays Cartesia's TTS WebSocket: every word of an input becomes one ``chunk`` of
    ``WORD`` seconds (sample value ``1000 * (context index + 1)``), followed by a
    ``timestamps`` message when ``add_timestamps`` is set; ``continue: false`` ends the
    context with ``done``."""

    delays: dict[int, float] = field(default_factory=dict)
    """Context index -> seconds to wait before generating its first input."""
    expire_after_input: bool = False
    """Send ``done`` after every input, as if the context had expired."""
    never_done: bool = False
    error: dict[str, Any] | None = None
    close_after_done: bool = False
    reject_status: int | None = None
    url: str = ""
    handshakes: int = 0
    requests: list[Request] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    done_sent: list[str] = field(default_factory=list)

    def inputs(self) -> list[dict[str, Any]]:
        return [m for m in self.messages if "transcript" in m]

    def process_request(self, conn: ServerConnection, request: Request) -> Response | None:
        self.handshakes += 1
        self.requests.append(request)
        if self.reject_status is not None:
            body = json.dumps({"title": "Rejected", "message": "invalid API key"})
            return conn.respond(HTTPStatus(self.reject_status), body)
        return None

    async def handler(self, ws: ServerConnection) -> None:
        queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        tasks: list[asyncio.Task[None]] = []
        try:
            async for raw in ws:
                msg = json.loads(raw)
                self.messages.append(msg)
                cid = msg["context_id"]
                if msg.get("cancel"):
                    self.cancelled.append(cid)
                    continue
                if cid not in queues:
                    queues[cid] = asyncio.Queue()
                    self.contexts.append(cid)
                    index = len(self.contexts) - 1
                    tasks.append(asyncio.create_task(self._context(ws, cid, index, queues[cid])))
                queues[cid].put_nowait(msg)
        except ConnectionClosed:
            pass
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _context(
        self, ws: ServerConnection, cid: str, index: int, queue: asyncio.Queue[dict[str, Any]]
    ) -> None:
        elapsed = 0.0
        first = True
        while True:
            msg = await queue.get()
            if self.error is not None:
                await ws.send(json.dumps({"type": "error", "done": True, "context_id": cid,
                                          **self.error}))  # fmt: skip
                return
            if first and self.delays.get(index):
                await asyncio.sleep(self.delays[index])
            first = False
            rate = msg["output_format"]["sample_rate"]
            words = msg["transcript"].split()
            for _ in words:
                data = base64.b64encode(pcm(1000 * (index + 1), WORD, rate)).decode()
                await ws.send(json.dumps({"type": "chunk", "data": data, "done": False,
                                          "status_code": 206, "step_time": 1.5,
                                          "context_id": cid}))  # fmt: skip
            if words and msg.get("add_timestamps"):
                starts = [elapsed + i * WORD for i in range(len(words))]
                timestamps = {"words": words, "start": starts, "end": [s + WORD for s in starts]}
                await ws.send(json.dumps({"type": "timestamps", "done": False, "status_code": 206,
                                          "context_id": cid, "word_timestamps": timestamps}))  # fmt: skip
            elapsed += len(words) * WORD
            ends = not msg["continue"] and not self.never_done
            if ends or self.expire_after_input:
                await ws.send(json.dumps({"type": "done", "done": True, "status_code": 206,
                                          "context_id": cid}))  # fmt: skip
                self.done_sent.append(cid)
                if self.close_after_done:
                    await ws.close()
                return


@pytest.fixture
async def tts_server() -> AsyncIterator[FakeTTSServer]:
    fake = FakeTTSServer()
    async with serve(fake.handler, "127.0.0.1", 0, process_request=fake.process_request) as srv:
        port = next(iter(srv.sockets)).getsockname()[1]
        fake.url = f"http://127.0.0.1:{port}"
        yield fake


def make_tts(server: FakeTTSServer, **kw: Any) -> CartesiaTTS:
    return CartesiaTTS(api_key=KEY, base_url=server.url, **kw)


# ------------------------------------------------------------------------ TTS tests
async def test_stream_sends_continuations_on_one_context(tts_server: FakeTTSServer) -> None:
    tts = make_tts(tts_server)
    metrics: list[TTSMetrics] = []
    tts.on("metrics", metrics.append)
    stream = tts.stream()
    stream.push_text("Hello there. ")
    stream.push_text("How are you? ")
    stream.end_input()
    items = await collect_tts(stream)
    await tts.aclose()

    inputs = tts_server.inputs()
    assert [m["transcript"] for m in inputs] == ["Hello there. ", "How are you? ", ""]
    assert [m["continue"] for m in inputs] == [True, True, False]
    assert len({m["context_id"] for m in inputs}) == 1
    # every input of a context repeats the same fields (Cartesia requires it)
    shared = [{k: v for k, v in m.items() if k not in ("transcript", "continue")} for m in inputs]
    assert all(s == shared[0] for s in shared)
    assert shared[0]["model_id"] == "sonic-3.6"
    assert shared[0]["voice"] == {"mode": "id", "id": DEFAULT_VOICE}
    assert shared[0]["output_format"] == {
        "container": "raw",
        "encoding": "pcm_s16le",
        "sample_rate": 24_000,
    }
    assert shared[0]["max_buffer_delay_ms"] == 0
    assert shared[0]["add_timestamps"] is True

    request = tts_server.requests[0]
    assert path(request) == "/tts/websocket"
    assert query(request) == {"cartesia_version": [API_VERSION]}
    assert request.headers["X-API-Key"] == KEY
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert request.headers["Cartesia-Version"] == API_VERSION

    audio = audio_of(items)
    assert audio.sample_rate == 24_000
    assert audio.duration == pytest.approx(5 * WORD)
    assert items[-1].is_final and not items[-1].frame
    assert items[-1].text == "Hello there. How are you?"
    assert not any(i.is_final for i in items[:-1])
    assert metrics and metrics[-1].streamed and not metrics[-1].cancelled
    assert metrics[-1].ttfb is not None and metrics[-1].audio_duration == pytest.approx(0.5)


async def test_flush_opens_a_new_context_and_segments_play_in_order(
    tts_server: FakeTTSServer,
) -> None:
    tts_server.delays = {0: 0.3}  # the second context's audio arrives before the first one's
    tts = make_tts(tts_server)
    stream = tts.stream()
    stream.push_text("One two. ")
    stream.flush()
    stream.push_text("Three. ")
    stream.end_input()
    items = await collect_tts(stream)
    await tts.aclose()

    first, second = tts_server.contexts
    assert [(m["context_id"], m["transcript"], m["continue"]) for m in tts_server.inputs()] == [
        (first, "One two. ", True),
        (first, "", False),
        (second, "Three. ", True),
        (second, "", False),
    ]
    finals = [i for i in items if i.is_final]
    assert [f.text for f in finals] == ["One two.", "Three."]
    assert finals[0].segment_id != finals[1].segment_id
    # audio of the first segment is emitted completely before the second one's
    samples = np.concatenate([i.frame.to_numpy() for i in items if i.frame])
    assert samples.tolist() == [1000] * round(2 * WORD * 24_000) + [2000] * round(WORD * 24_000)
    assert items.index(finals[0]) < min(
        n for n, i in enumerate(items) if i.frame and i.segment_id == finals[1].segment_id
    )


async def test_word_timestamps_are_offset_to_the_stream_timeline(
    tts_server: FakeTTSServer,
) -> None:
    tts = make_tts(tts_server)
    stream = tts.stream()
    stream.push_text("One two. ")
    stream.flush()
    stream.push_text("Three four. ")
    stream.push_text("Five. ")
    stream.end_input()
    items = await collect_tts(stream)
    await tts.aclose()

    word_items = [i for i in items if i.words]
    assert all(not i.frame and not i.is_final for i in word_items)
    words = [w for i in word_items for w in i.words or []]
    assert [w.word for w in words] == ["One", "two.", "Three", "four.", "Five."]
    assert [w.start for w in words] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    assert [w.end for w in words] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
    assert words[-1].end == pytest.approx(audio_of(items).duration)
    finals = [i for i in items if i.is_final]
    assert {i.segment_id for i in word_items} == {f.segment_id for f in finals}


async def test_stream_options_are_sent_with_every_input(tts_server: FakeTTSServer) -> None:
    tts = make_tts(
        tts_server,
        model="sonic-3.6-2026-08-27",
        sample_rate=16_000,
        language="fr",
        speed=0.9,
        volume=1.5,
        emotion="calm",
        max_buffer_delay_ms=None,
        word_timestamps=False,
        pronunciation_dict_id="dict_1",
    )
    assert not tts.capabilities.word_timestamps
    stream = tts.stream(voice="voice_custom")
    stream.push_text("Bonjour tout le monde. ")
    stream.end_input()
    items = await collect_tts(stream)
    await tts.aclose()

    for msg in tts_server.inputs():
        assert msg["model_id"] == "sonic-3.6-2026-08-27"
        assert msg["voice"] == {"mode": "id", "id": "voice_custom"}
        assert msg["output_format"]["sample_rate"] == 16_000
        assert msg["language"] == "fr"
        assert msg["generation_config"] == {"speed": 0.9, "volume": 1.5, "emotion": "calm"}
        assert msg["pronunciation_dict_id"] == "dict_1"
        assert "max_buffer_delay_ms" not in msg and "add_timestamps" not in msg
    assert not any(i.words for i in items)
    audio = audio_of(items)
    assert audio.sample_rate == 16_000 and audio.duration == pytest.approx(4 * WORD)


async def test_aclose_cancels_the_open_context(tts_server: FakeTTSServer) -> None:
    tts = make_tts(tts_server)
    metrics: list[TTSMetrics] = []
    tts.on("metrics", metrics.append)
    stream = tts.stream()
    stream.push_text("Once upon a time ")  # no flush: the context stays open
    first = await asyncio.wait_for(anext(stream), 5)
    assert first.frame
    await stream.aclose()
    await wait_until(lambda: tts_server.cancelled)
    ctx = tts_server.contexts[0]
    assert tts_server.cancelled == [ctx]
    # an aborted context is cancelled, not finished with `continue: false`
    assert [m["continue"] for m in tts_server.inputs()] == [True]
    assert metrics[-1].cancelled

    # the shared socket stays usable after the interruption
    stream = tts.stream()
    stream.push_text("Hi again. ")
    stream.end_input()
    assert audio_of(await collect_tts(stream)).duration == pytest.approx(2 * WORD)
    assert tts_server.handshakes == 1
    await tts.aclose()


async def test_interrupted_cascade_response_cancels_the_context(tts_server: FakeTTSServer) -> None:
    tts_server.delays = {0: 5.0}  # generation never finishes during the test
    tts = make_tts(tts_server)
    engine = CascadeEngine(stt=MockSTT(), llm=MockLLM(), tts=tts, vad=EnergyVAD())
    conn = await engine.connect(EngineOptions())
    await conn.say("This reply will be interrupted. It keeps going.")
    await wait_until(lambda: len(tts_server.inputs()) >= 1)
    await conn.cancel_response()
    await wait_until(lambda: tts_server.cancelled)
    assert tts_server.cancelled == tts_server.contexts[:1]
    await conn.aclose()
    await engine.aclose()


async def test_cascade_engine_speaks_through_the_stream(tts_server: FakeTTSServer) -> None:
    tts = make_tts(tts_server)
    engine = CascadeEngine(stt=MockSTT(), llm=MockLLM(), tts=tts, vad=EnergyVAD())
    await engine.warmup()
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
    assert AudioFrame.concat(frames).duration == pytest.approx(6 * WORD)
    spoken = "".join(e.delta for e in events if isinstance(e, ResponseText))
    assert spoken.split() == ["Hello", "there.", "How", "are", "you", "today?"]
    inputs = tts_server.inputs()
    assert [m["transcript"] for m in inputs] == ["Hello there. ", "How are you today? ", ""]
    assert len({m["context_id"] for m in inputs}) == 1
    assert tts_server.handshakes == 1  # warmup() opened the socket the response used


@pytest.mark.parametrize(
    ("status", "error_type", "retryable"),
    [(400, ProviderError, False), (401, AuthenticationError, False), (429, RateLimitError, True)],
)
async def test_error_messages_are_mapped(
    tts_server: FakeTTSServer, status: int, error_type: type[ProviderError], retryable: bool
) -> None:
    tts_server.error = {"status_code": status, "title": "Bad request", "message": "nope",
                        "error_code": "invalid_voice"}  # fmt: skip
    tts = make_tts(tts_server)
    stream = tts.stream()
    stream.push_text("Hi. ")
    stream.end_input()
    with pytest.raises(error_type) as info:
        await collect_tts(stream)
    assert type(info.value) is error_type
    assert info.value.status_code == status and info.value.retryable is retryable
    assert info.value.provider == "cartesia" and "invalid_voice" in str(info.value)
    await stream.aclose()
    await tts.aclose()


async def test_rejected_handshake_is_an_authentication_error(tts_server: FakeTTSServer) -> None:
    tts_server.reject_status = 401
    tts = make_tts(tts_server)
    with pytest.raises(AuthenticationError):
        await tts.warmup()
    stream = tts.stream()
    stream.push_text("Hi. ")
    stream.end_input()
    with pytest.raises(AuthenticationError, match="invalid API key"):
        await collect_tts(stream)
    await tts.aclose()


async def test_connection_is_shared_and_reopened_after_the_server_closes_it(
    tts_server: FakeTTSServer,
) -> None:
    tts = make_tts(tts_server)
    for text in ("First. ", "Second. "):
        stream = tts.stream()
        stream.push_text(text)
        stream.end_input()
        assert audio_of(await collect_tts(stream)).duration == pytest.approx(WORD)
    assert tts_server.handshakes == 1

    tts_server.close_after_done = True  # e.g. Cartesia's idle timeout
    for text in ("Third. ", "Fourth. "):
        stream = tts.stream()
        stream.push_text(text)
        stream.end_input()
        assert audio_of(await collect_tts(stream)).duration == pytest.approx(WORD)
        await wait_until(lambda: tts._conn is None or tts._conn.closed)
    assert tts_server.handshakes == 2
    await tts.aclose()


async def test_context_ended_by_the_server_continues_on_a_new_context(
    tts_server: FakeTTSServer,
) -> None:
    tts_server.expire_after_input = True
    tts = make_tts(tts_server)
    stream = tts.stream()
    stream.push_text("Hello there. ")
    await wait_until(lambda: tts_server.done_sent)
    await wait_until(lambda: stream._contexts and stream._contexts[0].ended)
    stream.push_text("Still here. ")
    stream.end_input()
    items = await collect_tts(stream)
    await tts.aclose()

    texts = [(m["context_id"], m["transcript"]) for m in tts_server.inputs() if m["transcript"]]
    assert [t for _, t in texts] == ["Hello there. ", "Still here. "]
    assert texts[0][0] != texts[1][0]
    finals = [i for i in items if i.is_final]
    assert [f.text for f in finals] == ["Hello there. Still here."]
    assert audio_of(items).duration == pytest.approx(4 * WORD)
    words = [w for i in items for w in i.words or []]
    assert [w.start for w in words] == pytest.approx([0.0, 0.1, 0.2, 0.3])


async def test_watchdog_fails_when_audio_never_arrives(tts_server: FakeTTSServer) -> None:
    tts_server.never_done = True
    tts = make_tts(tts_server, receive_timeout=0.3)
    stream = tts.stream()
    stream.push_text("Hello. ")
    stream.end_input()
    with pytest.raises(ProviderTimeoutError):
        await collect_tts(stream)
    await stream.aclose()
    await tts.aclose()


async def test_empty_flush_ends_an_empty_segment(tts_server: FakeTTSServer) -> None:
    tts = make_tts(tts_server)
    stream = tts.stream()
    stream.flush()
    stream.push_text("Hi. ")
    stream.end_input()
    items = await collect_tts(stream)
    await tts.aclose()
    finals = [i for i in items if i.is_final]
    assert [f.text for f in finals] == [None, "Hi."]
    assert len(tts_server.contexts) == 1


# -------------------------------------------------------------------- bytes endpoint
async def test_synthesize_uses_the_bytes_endpoint() -> None:
    seen: list[httpx.Request] = []
    body = pcm(1234, 0.25, 24_000)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=body, headers={"content-type": "audio/pcm"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tts = CartesiaTTS(api_key=KEY, http_client=client, voice="voice_1", language="en", speed=1.2)
    items = [item async for item in tts.synthesize("Hello **world**!")]
    await tts.aclose()
    assert not client.is_closed  # injected clients belong to the caller
    await client.aclose()

    assert audio_of(items).duration == pytest.approx(0.25)
    assert items[0].text == "Hello world!" and items[-1].is_final
    request = seen[0]
    assert request.method == "POST" and str(request.url) == "https://api.cartesia.ai/tts/bytes"
    assert request.headers["X-API-Key"] == KEY
    assert request.headers["Cartesia-Version"] == API_VERSION
    assert json.loads(request.content) == {
        "model_id": "sonic-3.6",
        "transcript": "Hello world!",
        "voice": {"mode": "id", "id": "voice_1"},
        "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 24_000},
        "language": "en",
        "generation_config": {"speed": 1.2},
    }


@pytest.mark.parametrize(
    ("status", "error_type", "retryable"),
    [
        (401, AuthenticationError, False),
        (429, RateLimitError, True),
        (500, ProviderError, True),
        (400, ProviderError, False),
    ],
)
async def test_synthesize_maps_http_errors(
    status: int, error_type: type[ProviderError], retryable: bool
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"title": "Failed", "message": "details"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tts = CartesiaTTS(api_key=KEY, http_client=client)
    with pytest.raises(error_type) as info:
        await tts.synthesize("Hello.").collect()
    assert type(info.value) is error_type
    assert info.value.status_code == status and info.value.retryable is retryable
    assert "details" in str(info.value)
    await client.aclose()


async def test_synthesize_maps_network_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tts = CartesiaTTS(api_key=KEY, http_client=client)
    with pytest.raises(ProviderConnectionError):
        await tts.synthesize("Hello.").collect()
    empty = await tts.synthesize("   ").collect()  # nothing to say: no request at all
    assert not empty
    await tts.aclose()
    await client.aclose()


# ------------------------------------------------------------------------ registry
def test_registry_and_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARTESIA_API_KEY", "env-key")
    tts = create("tts", "cartesia")
    assert isinstance(tts, CartesiaTTS)
    assert (tts.model, tts.sample_rate, tts.voice) == ("sonic-3.6", 24_000, DEFAULT_VOICE)
    assert tts.capabilities.streaming and tts.capabilities.word_timestamps
    assert tts._headers()["X-API-Key"] == "env-key"
    assert create("tts", "cartesia/sonic-3.6-2026-08-27").model == "sonic-3.6-2026-08-27"
    assert create("tts", {"provider": "cartesia", "api_key": "k"})._headers()["X-API-Key"] == "k"

    stt = create("stt", "cartesia")
    assert isinstance(stt, CartesiaSTT) and stt.model == "ink-2" and stt.turn_detection
    assert stt.capabilities.streaming and stt.capabilities.end_of_turn
    whisper = create("stt", "cartesia/ink-whisper")
    assert not whisper.turn_detection and whisper.capabilities.word_timestamps
    manual = create("stt", "cartesia", turn_detection=False)
    assert not manual.capabilities.end_of_turn and manual.capabilities.language_detection

    for kind in ("tts", "stt"):
        spec = get_provider(kind, "cartesia")
        assert spec.env == ("CARTESIA_API_KEY",) and not spec.local and spec.extra is None
        assert spec.available

    monkeypatch.delenv("CARTESIA_API_KEY")
    with pytest.raises(ConfigurationError, match="CARTESIA_API_KEY"):
        CartesiaTTS()
    with pytest.raises(ConfigurationError, match="CARTESIA_API_KEY"):
        CartesiaSTT()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 12_345},
        {"max_buffer_delay_ms": 6000},
        {"speed": 3.0},
        {"volume": 0.1},
    ],
)
def test_invalid_tts_options(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        CartesiaTTS(api_key=KEY, **kwargs)


def test_invalid_stt_options() -> None:
    with pytest.raises(ConfigurationError):
        CartesiaSTT(api_key=KEY, model="ink-whisper", turn_detection=True)
    with pytest.raises(ConfigurationError):
        CartesiaSTT(api_key=KEY, keyterms="Cartesia")


# ------------------------------------------------------------------------ fake STT
Script = Callable[["FakeSTTServer", ServerConnection], Awaitable[None]]


@dataclass
class FakeSTTServer:
    script: Script
    url: str = ""
    requests: list[Request] = field(default_factory=list)
    received: list[str | bytes] = field(default_factory=list)
    reject_status: int | None = None

    def process_request(self, conn: ServerConnection, request: Request) -> Response | None:
        self.requests.append(request)
        if self.reject_status is not None:
            return conn.respond(HTTPStatus(self.reject_status), "slow down")
        return None

    async def handler(self, ws: ServerConnection) -> None:
        try:
            await self.script(self, ws)
        except ConnectionClosed:
            pass

    def texts(self) -> list[str]:
        return [m for m in self.received if isinstance(m, str)]

    def audio_bytes(self) -> int:
        return sum(len(m) for m in self.received if isinstance(m, bytes))


StartSTTServer = Callable[[Script], Awaitable[FakeSTTServer]]


@pytest.fixture
async def stt_server() -> AsyncIterator[StartSTTServer]:
    servers: list[Any] = []

    async def start(script: Script) -> FakeSTTServer:
        fake = FakeSTTServer(script)
        srv = await serve(fake.handler, "127.0.0.1", 0, process_request=fake.process_request)
        servers.append(srv)
        fake.url = f"http://127.0.0.1:{next(iter(srv.sockets)).getsockname()[1]}"
        return fake

    yield start
    for srv in servers:
        srv.close()
        await srv.wait_closed()


async def collect_stt(
    stream: STTStream, until: STTEventType | None = None, timeout: float = 5.0
) -> list[STTEvent]:
    events: list[STTEvent] = []

    async def run() -> None:
        async for ev in stream:
            events.append(ev)
            if ev.type == until:
                return

    await asyncio.wait_for(run(), timeout)
    return events


def push_speech(stream: STTStream, seconds: float, rate: int = 16_000) -> None:
    audio = synth_speech(seconds, rate)
    step = 0.02
    t = 0.0
    while t < seconds - 1e-9:
        stream.push_audio(audio.slice(t, t + step))
        t += step


TURN_EVENTS: list[dict[str, Any]] = [
    {"type": "turn.start"},
    {"type": "turn.update", "transcript": "Hey can"},
    {"type": "turn.update", "transcript": "Hey can you help"},
    {"type": "turn.eager_end", "transcript": "Hey can you help"},
    {"type": "turn.resume"},
    {"type": "turn.update", "transcript": "Hey can you help me?"},
    {"type": "turn.end", "transcript": "Hey can you help me?"},
    {"type": "turn.start"},
    {"type": "turn.update", "transcript": ""},
    {"type": "turn.end", "transcript": " Thanks. "},
]


async def turns_script(fake: FakeSTTServer, ws: ServerConnection) -> None:
    await ws.send(json.dumps({"type": "connected", "request_id": "req_1"}))
    started = False
    async for msg in ws:
        fake.received.append(msg)
        if isinstance(msg, bytes):
            if not started and fake.audio_bytes() >= 0.2 * 16_000 * 2:
                started = True
                for ev in TURN_EVENTS:
                    await ws.send(json.dumps({**ev, "request_id": "req_1"}))
        elif json.loads(msg) == {"type": "close"}:
            await ws.close()
            return


async def test_stt_turn_events_map_to_stt_events(stt_server: StartSTTServer) -> None:
    fake = await stt_server(turns_script)
    stt = CartesiaSTT(
        api_key=KEY,
        base_url=fake.url,
        turn_eager_end_threshold=0.5,
        turn_end_timeout_ms=2000,
        keyterms=["Cartesia", "Sonic 3"],
    )
    stream = stt.stream()
    push_speech(stream, 0.4)
    stream.end_input()  # sends the buffered audio, then {"type": "close"}
    events = await collect_stt(stream)  # the server closes after `close`
    await stream.aclose()

    T = STTEventType
    assert [e.type for e in events] == [
        T.START_OF_SPEECH, T.INTERIM_TRANSCRIPT, T.INTERIM_TRANSCRIPT, T.EAGER_END_OF_TURN,
        T.TURN_RESUMED, T.INTERIM_TRANSCRIPT, T.FINAL_TRANSCRIPT, T.END_OF_SPEECH, T.END_OF_TURN,
        T.START_OF_SPEECH, T.FINAL_TRANSCRIPT, T.END_OF_SPEECH, T.END_OF_TURN,
    ]  # fmt: skip
    assert [e.text for e in events[:9]] == [
        "", "Hey can", "Hey can you help", "Hey can you help", "",
        "Hey can you help me?", "Hey can you help me?", "", "Hey can you help me?",
    ]  # fmt: skip
    assert events[10].text == "Thanks."
    assert len({e.segment_id for e in events[:9]}) == 1
    assert events[9].segment_id != events[0].segment_id

    request = fake.requests[0]
    assert path(request) == "/stt/turns/websocket"
    assert query(request) == {
        "model": ["ink-2"],
        "encoding": ["pcm_s16le"],
        "sample_rate": ["16000"],
        "cartesia_version": [API_VERSION],
        "turn_eager_end_threshold": ["0.5"],
        "turn_end_timeout_ms": ["2000"],
        "keyterm": ["Cartesia", "Sonic 3"],
    }
    assert request.headers["X-API-Key"] == KEY
    assert request.headers["Cartesia-Version"] == API_VERSION
    chunks = [m for m in fake.received if isinstance(m, bytes)]
    assert fake.audio_bytes() == round(0.4 * 16_000) * 2
    assert all(len(c) >= 0.05 * 16_000 * 2 for c in chunks[:-1])  # ~50 ms chunks
    assert fake.texts() == [json.dumps({"type": "close"})]


async def test_stt_turn_mode_works_as_cascade_turn_source(stt_server: StartSTTServer) -> None:
    fake = await stt_server(turns_script)
    stt = CartesiaSTT(api_key=KEY, base_url=fake.url)
    stream = stt.stream()
    push_speech(stream, 0.4)
    events = await collect_stt(stream, until=STTEventType.END_OF_TURN)
    finals = [e for e in events if e.type == STTEventType.FINAL_TRANSCRIPT]
    # the final transcript precedes END_OF_TURN, which commits the turn in the cascade
    assert events[-2].type == STTEventType.END_OF_SPEECH and finals[0].text
    stream.flush()  # no finalize command on the turns endpoint: only buffered audio is sent
    stream.end_input()
    await collect_stt(stream)
    await stream.aclose()
    assert fake.texts() == [json.dumps({"type": "close"})]


async def test_ink_turn_events_drive_the_cascade_without_a_vad(
    stt_server: StartSTTServer,
) -> None:
    async def one_turn(fake: FakeSTTServer, ws: ServerConnection) -> None:
        started = False
        async for msg in ws:
            fake.received.append(msg)
            if isinstance(msg, str):
                break  # {"type": "close"}
            if not started and fake.audio_bytes() >= 0.2 * 16_000 * 2:
                started = True
                for ev in TURN_EVENTS[:7]:  # one complete turn
                    await ws.send(json.dumps({**ev, "request_id": "req_1"}))

    fake = await stt_server(one_turn)
    llm = MockLLM(responses=["Sure, what do you need?"])
    stt = CartesiaSTT(api_key=KEY, base_url=fake.url)
    engine = CascadeEngine(stt=stt, llm=llm, tts=MockTTS())  # no VAD: Ink drives the turns
    conn = await engine.connect(EngineOptions())
    events: list[object] = []

    async def drain() -> None:
        async for ev in conn.events():
            events.append(ev)
            if isinstance(ev, ResponseDone):
                return

    drainer = asyncio.create_task(drain())
    audio = synth_speech(0.4, 16_000)
    for n in range(20):
        await conn.send_audio(audio.slice(n * 0.02, (n + 1) * 0.02))
    await asyncio.wait_for(drainer, 5)
    await conn.aclose()
    await engine.aclose()

    def first(kind: type) -> int:
        return next(n for n, e in enumerate(events) if isinstance(e, kind))

    assert (
        first(InputSpeechStarted)
        < first(InputSpeechStopped)
        < first(InputCommitted)
        < first(ResponseStarted)
    )
    finals = [e.text for e in events if isinstance(e, InputTranscript) and e.is_final]
    assert finals == ["Hey can you help me?"]
    last_user = llm.requests[0].last_message("user")
    assert last_user is not None and last_user.text == "Hey can you help me?"


async def manual_script(fake: FakeSTTServer, ws: ServerConnection) -> None:
    finalizes = 0
    interim_sent = False
    async for msg in ws:
        fake.received.append(msg)
        if isinstance(msg, bytes):
            if not interim_sent:
                interim_sent = True
                await ws.send(json.dumps({"type": "transcript", "is_final": False,
                                          "text": " Hello", "request_id": "r"}))  # fmt: skip
        elif msg == "finalize":
            finalizes += 1
            if finalizes == 1:
                words = [{"word": "Hello", "start": 0.1, "end": 0.4},
                         {"word": "world", "start": 0.5, "end": 0.9}]  # fmt: skip
                await ws.send(json.dumps({"type": "transcript", "is_final": True,
                                          "text": " Hello world", "duration": 1.0,
                                          "language": "en", "words": words,
                                          "request_id": "r"}))  # fmt: skip
            await ws.send(json.dumps({"type": "flush_done", "request_id": "r"}))
        elif msg == "close":
            await ws.send(json.dumps({"type": "done", "request_id": "r"}))
            await ws.close()
            return


async def test_stt_manual_finalize_mode(stt_server: StartSTTServer) -> None:
    fake = await stt_server(manual_script)
    stt = CartesiaSTT(api_key=KEY, base_url=fake.url, turn_detection=False, keyterms=["Ink"])
    metrics: list[STTMetrics] = []
    stt.on("metrics", metrics.append)
    stream = stt.stream()
    push_speech(stream, 0.3)
    stream.flush()
    events = await collect_stt(stream, until=STTEventType.FINAL_TRANSCRIPT)
    stream.flush()  # nothing new to finalize: acknowledged with an empty final
    events += await collect_stt(stream, until=STTEventType.FINAL_TRANSCRIPT)
    stream.end_input()
    events += await collect_stt(stream)
    await stream.aclose()

    T = STTEventType
    assert [(e.type, e.text) for e in events] == [
        (T.INTERIM_TRANSCRIPT, "Hello"),
        (T.FINAL_TRANSCRIPT, "Hello world"),
        (T.FINAL_TRANSCRIPT, ""),
        (T.FINAL_TRANSCRIPT, ""),
    ]
    final = events[1].transcript
    assert final is not None and final.language == "en"
    assert [(w.word, w.start, w.end) for w in final.words or []] == [
        ("Hello", 0.1, 0.4),
        ("world", 0.5, 0.9),
    ]
    assert (final.start_time, final.end_time) == (0.1, 0.9)
    assert events[0].segment_id == events[1].segment_id != events[2].segment_id
    assert fake.texts() == ["finalize", "finalize", "finalize", "close"]
    assert fake.audio_bytes() == round(0.3 * 16_000) * 2
    request = fake.requests[0]
    assert path(request) == "/stt/websocket"
    assert query(request) == {
        "model": ["ink-2"],
        "encoding": ["pcm_s16le"],
        "sample_rate": ["16000"],
        "cartesia_version": [API_VERSION],
        "keyterm": ["Ink"],
    }
    assert metrics and all(m.streamed and m.latency is not None for m in metrics)


async def test_stt_whisper_options_in_query(stt_server: StartSTTServer) -> None:
    fake = await stt_server(manual_script)
    stt = CartesiaSTT(
        api_key=KEY,
        base_url=fake.url,
        model="ink-whisper",
        language="de",
        min_volume=0.1,
        max_silence_duration_secs=0.4,
        sample_rate=24_000,
    )
    stream = stt.stream()
    stream.end_input()
    await collect_stt(stream)
    await stream.aclose()
    assert query(fake.requests[0]) == {
        "model": ["ink-whisper"],
        "encoding": ["pcm_s16le"],
        "sample_rate": ["24000"],
        "cartesia_version": [API_VERSION],
        "language": ["de"],
        "min_volume": ["0.1"],
        "max_silence_duration_secs": ["0.4"],
    }


async def test_transcribe_uses_the_finalize_endpoint(stt_server: StartSTTServer) -> None:
    fake = await stt_server(manual_script)
    stt = CartesiaSTT(api_key=KEY, base_url=fake.url)  # turn detection on for streams
    result = await asyncio.wait_for(stt.transcribe(synth_speech(0.5, 24_000)), 5)
    assert result.text == "Hello world" and result.language == "en"
    assert [w.word for w in result.words or []] == ["Hello", "world"]
    assert path(fake.requests[0]) == "/stt/websocket"
    assert fake.texts() == ["finalize", "close"]
    assert abs(fake.audio_bytes() - round(0.5 * 16_000) * 2) <= 64  # resampled to 16 kHz


async def test_stt_error_message_is_raised(stt_server: StartSTTServer) -> None:
    async def script(fake: FakeSTTServer, ws: ServerConnection) -> None:
        await ws.send(json.dumps({"type": "error", "status_code": 401, "title": "Unauthorized",
                                  "message": "invalid key", "request_id": "r"}))  # fmt: skip

    fake = await stt_server(script)
    stream = CartesiaSTT(api_key=KEY, base_url=fake.url).stream()
    push_speech(stream, 0.1)
    with pytest.raises(AuthenticationError, match="invalid key"):
        await collect_stt(stream)
    await stream.aclose()


async def test_stt_unexpected_close_is_an_error(stt_server: StartSTTServer) -> None:
    async def script(fake: FakeSTTServer, ws: ServerConnection) -> None:
        await ws.close()

    fake = await stt_server(script)
    stream = CartesiaSTT(api_key=KEY, base_url=fake.url).stream()
    push_speech(stream, 0.1)
    with pytest.raises(ProviderConnectionError):
        await collect_stt(stream)
    await stream.aclose()


async def test_stt_rejected_handshake_is_mapped(stt_server: StartSTTServer) -> None:
    fake = await stt_server(manual_script)
    fake.reject_status = 429
    stream = CartesiaSTT(api_key=KEY, base_url=fake.url).stream()
    with pytest.raises(RateLimitError):
        await collect_stt(stream)
    await stream.aclose()


# --------------------------------------------------------------------- integration
needs_key = pytest.mark.skipif(
    not os.environ.get("CARTESIA_API_KEY"), reason="needs CARTESIA_API_KEY"
)


@pytest.mark.integration
@needs_key
async def test_integration_tts_stream_with_word_timestamps() -> None:
    tts = CartesiaTTS()
    try:
        await tts.warmup()
        stream = tts.stream()
        stream.push_text("Hello from voice agent next. ")
        stream.push_text("This sentence is streamed as a continuation. ")
        stream.end_input()
        items = await collect_tts(stream, timeout=30)
    finally:
        await tts.aclose()
    audio = audio_of(items)
    assert audio.sample_rate == 24_000 and audio.duration > 1.5
    words = [w for i in items for w in i.words or []]
    assert len(words) >= 8
    assert all(a.start <= b.start for a, b in itertools.pairwise(words))
    assert words[-1].end <= audio.duration + 0.25
    assert items[-1].is_final


@pytest.mark.integration
@needs_key
async def test_integration_tts_cancel_keeps_the_socket_usable() -> None:
    tts = CartesiaTTS()
    try:
        stream = tts.stream()
        stream.push_text("This is a long answer that will be interrupted right after it starts. ")
        assert (await asyncio.wait_for(anext(stream), 30)).frame is not None
        await stream.aclose()
        stream = tts.stream()
        stream.push_text("Still working. ")
        stream.end_input()
        assert audio_of(await collect_tts(stream, timeout=30)).duration > 0.3
    finally:
        await tts.aclose()


@pytest.mark.integration
@needs_key
async def test_integration_bytes_tts_and_stt_round_trip() -> None:
    tts = CartesiaTTS(sample_rate=16_000)
    stt = CartesiaSTT()
    try:
        audio = await tts.synthesize("The quick brown fox jumps over the lazy dog.").collect()
        assert audio.duration > 1.0
        result = await asyncio.wait_for(stt.transcribe(audio), 30)
    finally:
        await tts.aclose()
    assert "fox" in result.text.lower()


@pytest.mark.integration
@needs_key
async def test_integration_stt_turn_events() -> None:
    tts = CartesiaTTS(sample_rate=16_000)
    try:
        audio = await tts.synthesize("Can you tell me the weather in Paris today?").collect()
    finally:
        await tts.aclose()
    audio = AudioFrame.concat([audio, AudioFrame.silence(3.0, 16_000)])
    stream = CartesiaSTT().stream()

    async def feed() -> None:
        t = 0.0
        while t < audio.duration:
            stream.push_audio(audio.slice(t, t + 0.05))
            t += 0.05
            await asyncio.sleep(0.05)  # real time

    feeder = asyncio.create_task(feed())
    try:
        events = await collect_stt(stream, until=STTEventType.END_OF_TURN, timeout=30)
    finally:
        feeder.cancel()
        await asyncio.gather(feeder, return_exceptions=True)
        await stream.aclose()
    types = [e.type for e in events]
    assert STTEventType.START_OF_SPEECH in types
    final = next(e for e in events if e.type == STTEventType.FINAL_TRANSCRIPT)
    assert "weather" in final.text.lower()
