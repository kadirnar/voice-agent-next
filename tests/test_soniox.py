"""Soniox real-time STT against a local fake server.

The fake replays Soniox's documented WebSocket messages (a JSON config first, binary
audio, ``{"type": "finalize"}`` / ``{"type": "keepalive"}``, an empty frame to end;
responses with ``tokens`` whose ``is_final`` flips, ``<end>`` / ``<fin>`` tokens,
``finished: true`` and ``error_code`` / ``error_message``; see
https://soniox.com/docs/stt/api-reference/websocket-api). No network, no API key — except
the ``integration`` tests at the end, which need ``SONIOX_API_KEY``.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from voice_agent_next import Agent, AgentSession, AudioFrame, CascadeOptions, ChatMessage
from voice_agent_next.errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from voice_agent_next.metrics import STTMetrics, TurnMetrics
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import synth_speech
from voice_agent_next.providers.soniox import SonioxStream, SonioxSTT
from voice_agent_next.registry import create
from voice_agent_next.stt import STTEvent, STTEventType
from voice_agent_next.transports import LoopbackTransport
from voice_agent_next.utils import now

BYTES_PER_SECOND = 16_000 * 2
E = STTEventType


# ------------------------------------------------------------------------ fake server
@dataclass
class FakeConn:
    ws: ServerConnection
    path: str
    config: dict[str, Any] = field(default_factory=dict)
    received: list[str | bytes] = field(default_factory=list)

    def controls(self) -> list[str]:
        """``type`` of the control messages received after the config."""
        return [json.loads(m)["type"] for m in self.received if isinstance(m, str) and m]

    async def send(self, message: dict[str, Any]) -> None:
        await self.ws.send(json.dumps(message))


Handler = Callable[[FakeConn], Awaitable[None]]


class FakeSoniox:
    """A WebSocket server on 127.0.0.1 speaking Soniox's real-time protocol."""

    def __init__(self, handler: Handler, *, reject: tuple[int, str] | None = None) -> None:
        self._handler = handler
        self._reject = reject
        self._server: Server | None = None
        self.connections: list[FakeConn] = []

    async def __aenter__(self) -> FakeSoniox:
        self._server = await serve(
            self._handle, "127.0.0.1", 0, process_request=self._process_request
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    @property
    def url(self) -> str:
        assert self._server is not None
        port = next(iter(self._server.sockets)).getsockname()[1]
        return f"ws://127.0.0.1:{port}"

    def _process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        if self._reject is None:
            return None
        status, message = self._reject
        return connection.respond(status, json.dumps({"error_type": "x", "message": message}))

    async def _handle(self, ws: ServerConnection) -> None:
        conn = FakeConn(ws, ws.request.path if ws.request else "/")
        self.connections.append(conn)
        try:
            first = await ws.recv()
            conn.config = json.loads(first)
            await self._handler(conn)
        except ConnectionClosed:
            pass


def tok(
    text: str,
    start_ms: int,
    end_ms: int,
    *,
    final: bool = True,
    confidence: float = 0.95,
    language: str | None = None,
    speaker: str | None = None,
) -> dict[str, Any]:
    token: dict[str, Any] = {
        "text": text,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "confidence": confidence,
        "is_final": final,
    }
    if language:
        token["language"] = language
    if speaker:
        token["speaker"] = speaker
    return token


def special(text: str) -> dict[str, Any]:
    """``<end>`` / ``<fin>``: final tokens with no timing of their own."""
    return {"text": text, "is_final": True}


def resp(*tokens: dict[str, Any], final_ms: int = 0, total_ms: int = 0) -> dict[str, Any]:
    return {
        "tokens": list(tokens),
        "final_audio_proc_ms": final_ms,
        "total_audio_proc_ms": total_ms,
    }


@dataclass
class ScriptedServer:
    """Sends ``script`` responses once that much audio (seconds) arrived. ``finalize``
    turns the last non-final tokens final and adds ``<fin>``; the empty frame does the same,
    then sends ``finished`` and closes, like Soniox."""

    script: list[tuple[float, dict[str, Any]]]
    honor_finalize: bool = True
    sent: list[tuple[float, dict[str, Any]]] = field(default_factory=list)

    async def __call__(self, conn: FakeConn) -> None:
        audio = 0
        pending = list(self.script)
        nonfinal: list[dict[str, Any]] = []

        async def send(msg: dict[str, Any]) -> None:
            nonlocal nonfinal
            nonfinal = [t for t in msg["tokens"] if not t["is_final"]]
            self.sent.append((now(), msg))
            await conn.send(msg)

        def finalized() -> list[dict[str, Any]]:
            return [{**t, "is_final": True} for t in nonfinal]

        async for message in conn.ws:
            conn.received.append(message)
            if message in (b"", ""):  # end of stream
                for _, msg in pending:
                    await send(msg)
                await send(resp(*finalized()))
                await send({**resp(), "finished": True})
                await conn.ws.close(1000)
                return
            if isinstance(message, bytes):
                audio += len(message)
                while pending and audio >= pending[0][0] * BYTES_PER_SECOND:
                    await send(pending.pop(0)[1])
                continue
            kind = json.loads(message)["type"]
            if kind == "finalize" and self.honor_finalize:
                await send(resp(*finalized(), special("<fin>")))


def chunks(frame: AudioFrame, step: float = 0.02) -> list[AudioFrame]:
    n = round(step * frame.sample_rate) * 2
    return [
        AudioFrame(frame.data[i : i + n], frame.sample_rate) for i in range(0, len(frame.data), n)
    ]


async def collect(stream: AsyncIterator[STTEvent]) -> list[STTEvent]:
    return [ev async for ev in stream]


def kinds(events: Sequence[STTEvent]) -> list[tuple[STTEventType, str]]:
    return [(ev.type, ev.text) for ev in events]


async def _until(predicate: Callable[[], bool]) -> None:
    while not predicate():
        await asyncio.sleep(0.01)


# ------------------------------------------------------------------ STT-owned turns
ENDPOINT_SCRIPT = [
    (0.3, resp(tok("Hel", 220, 300, final=False))),
    (0.5, resp(tok("Hel", 220, 300), tok("lo", 300, 420), tok(" world", 480, 700, final=False))),
    (0.8, resp(tok(" world", 480, 700), tok(".", 700, 720), special("<end>"))),
    (1.2, resp(tok(" How", 1100, 1200, final=False))),
    (1.5, resp(tok(" How", 1100, 1200), tok(" are", 1220, 1300), tok(" you?", 1320, 1450),
               special("<end>"))),
]  # fmt: skip


async def test_endpoint_detection_emits_interims_finals_and_end_of_turn() -> None:
    async with FakeSoniox(ScriptedServer(list(ENDPOINT_SCRIPT))) as server:
        stt = SonioxSTT(
            api_key="test-key",
            base_url=server.url,
            language="en-US",
            terms=["Soniox", "Keanu Reeves"],
            context="A support call about an order.",
            context_general={"domain": "Retail"},
            max_endpoint_delay_ms=1500,
            endpoint_sensitivity=0.3,
            client_reference_id="call-42",
        )
        assert stt.capabilities.end_of_turn and stt.capabilities.word_timestamps
        metrics: list[STTMetrics] = []
        stt.on("metrics", metrics.append)
        stream = stt.stream()
        for frame in chunks(synth_speech(1.7, 16_000)):
            stream.push_audio(frame)
        events: list[STTEvent] = []
        turns = 0
        async for ev in stream:
            events.append(ev)
            if ev.type == E.END_OF_TURN:
                turns += 1
                if turns == 2:
                    stream.end_input()  # finalize (answered by an empty <fin>), then end
        await stream.aclose()

    first, second = "Hello world.", "How are you?"
    assert kinds(events) == [
        (E.START_OF_SPEECH, ""),
        (E.INTERIM_TRANSCRIPT, "Hel"),
        (E.INTERIM_TRANSCRIPT, "Hello world"),
        (E.FINAL_TRANSCRIPT, first),
        (E.END_OF_SPEECH, first),
        (E.END_OF_TURN, first),
        (E.START_OF_SPEECH, ""),
        (E.INTERIM_TRANSCRIPT, "How"),
        (E.FINAL_TRANSCRIPT, second),
        (E.END_OF_SPEECH, second),
        (E.END_OF_TURN, second),
        (E.FINAL_TRANSCRIPT, ""),  # end_input's finalize: nothing left to finalize
    ]
    start = events[0].transcript
    assert start is not None and start.start_time == pytest.approx(0.22)
    final = events[3].transcript
    assert final is not None and final.words is not None
    assert [(w.word, w.start, w.end) for w in final.words] == [
        ("Hello", pytest.approx(0.22), pytest.approx(0.42)),
        ("world.", pytest.approx(0.48), pytest.approx(0.72)),
    ]
    assert final.end_time == pytest.approx(0.72) and final.confidence == pytest.approx(0.95)
    assert final.language == "en"
    assert len({ev.segment_id for ev in events[:6]}) == 1
    assert events[6].segment_id != events[0].segment_id
    assert isinstance(stream, SonioxStream)

    conn = server.connections[0]
    assert conn.path == "/transcribe-websocket"
    assert conn.config == {
        "api_key": "test-key",
        "model": "stt-rt-v5",
        "audio_format": "pcm_s16le",
        "sample_rate": 16000,
        "num_channels": 1,
        "language_hints": ["en"],
        "context": {
            "text": "A support call about an order.",
            "terms": ["Soniox", "Keanu Reeves"],
            "general": [{"key": "domain", "value": "Retail"}],
        },
        "enable_endpoint_detection": True,
        "max_endpoint_delay_ms": 1500,
        "endpoint_sensitivity": 0.3,
        "client_reference_id": "call-42",
    }
    assert conn.controls() == ["finalize"]
    assert conn.received[-1] == b""  # end of stream
    assert sum(m.audio_duration for m in metrics) == pytest.approx(1.7, abs=0.01)


# --------------------------------------------------------------- cascade-owned turns
async def test_flush_sends_finalize_and_owns_the_turn() -> None:
    script = [
        (0.3, resp(tok("My", 100, 250, final=False))),
        (0.6, resp(tok("My", 100, 250), tok(" name", 260, 450, final=False),
                   tok(" is", 460, 550, final=False))),
    ]  # fmt: skip
    async with FakeSoniox(ScriptedServer(script)) as server:
        stt = SonioxSTT(api_key="k", base_url=server.url, end_of_turn=False)
        assert not stt.capabilities.end_of_turn
        assert stt.config()["enable_endpoint_detection"] is False
        metrics: list[STTMetrics] = []
        stt.on("metrics", metrics.append)
        stream = stt.stream()
        for frame in chunks(synth_speech(0.7, 16_000)):
            stream.push_audio(frame)
        stream.flush()  # e.g. the cascade's VAD saw the end of speech
        events: list[STTEvent] = []
        async for ev in stream:
            events.append(ev)
            if ev.type == E.END_OF_SPEECH:
                flushed_at = now()
                stream.flush()  # no audio since: answered at once, no finalize sent
            elif ev.type == E.FINAL_TRANSCRIPT and not ev.text:
                acked_at = now()
                break
        stream.end_input()
        events += await collect(stream)
        await stream.aclose()

    assert kinds(events) == [
        (E.START_OF_SPEECH, ""),
        (E.INTERIM_TRANSCRIPT, "My"),
        (E.INTERIM_TRANSCRIPT, "My name is"),
        (E.FINAL_TRANSCRIPT, "My name is"),
        (E.END_OF_SPEECH, "My name is"),
        (E.FINAL_TRANSCRIPT, ""),  # the second flush
        (E.FINAL_TRANSCRIPT, ""),  # end_input's flush
    ]  # no END_OF_TURN: the cascade owns the turn
    assert acked_at - flushed_at < 0.5
    assert server.connections[0].controls() == ["finalize"]
    assert metrics and all(m.latency is not None for m in metrics)


async def test_endpoint_while_a_flush_is_pending_has_no_end_of_turn() -> None:
    """Endpoint detection on, but the flush came first: the flush owns the turn."""
    script = [(0.3, resp(tok("Yes", 100, 300, final=False)))]

    async def handler(conn: FakeConn) -> None:
        async for message in conn.ws:
            conn.received.append(message)
            if isinstance(message, bytes) and len(conn.received) == 10:
                await conn.send(script[0][1])
            if message == json.dumps({"type": "finalize"}):
                await conn.send(resp(tok("Yes", 100, 300), special("<end>")))
                await conn.send(resp(special("<fin>")))
            if message == b"":
                await conn.send({**resp(), "finished": True})
                await conn.ws.close(1000)
                return

    async with FakeSoniox(handler) as server:
        stream = SonioxSTT(api_key="k", base_url=server.url).stream()
        for frame in chunks(synth_speech(0.4, 16_000)):
            stream.push_audio(frame)
        await asyncio.sleep(0.2)
        stream.end_input()
        events = await collect(stream)
        await stream.aclose()
    assert kinds(events) == [
        (E.START_OF_SPEECH, ""),
        (E.INTERIM_TRANSCRIPT, "Yes"),
        (E.FINAL_TRANSCRIPT, "Yes"),
        (E.END_OF_SPEECH, "Yes"),
        (E.FINAL_TRANSCRIPT, ""),  # the <fin>
    ]


async def test_language_identification_and_speakers() -> None:
    script = [
        (0.3, resp(tok("Hola", 100, 300, language="es", speaker="1"),
                   tok(" amigo", 320, 600, language="es", speaker="1"), special("<end>"))),
    ]  # fmt: skip
    async with FakeSoniox(ScriptedServer(script)) as server:
        stt = SonioxSTT(
            api_key="k",
            base_url=server.url,
            language_hints=["en", "es"],
            language_hints_strict=True,
            language_identification=True,
            speaker_diarization=True,
        )
        assert stt.capabilities.language_detection
        stream = stt.stream()
        for frame in chunks(synth_speech(0.5, 16_000)):
            stream.push_audio(frame)
        final = None
        async for ev in stream:
            if ev.type == E.FINAL_TRANSCRIPT:
                final = ev.transcript
                break
        assert final is not None and final.text == "Hola amigo" and final.language == "es"
        assert {t["speaker"] for t in stream.final_tokens} == {"1"}
        stream.end_input()
        await collect(stream)
        await stream.aclose()
    config = server.connections[0].config
    assert config["language_hints"] == ["en", "es"] and config["language_hints_strict"] is True
    assert config["enable_language_identification"] is True
    assert config["enable_speaker_diarization"] is True


async def test_end_of_stream_finalizes_pending_tokens() -> None:
    script = [(0.3, resp(tok("Good", 100, 300, final=False), tok("bye", 300, 450, final=False)))]
    async with FakeSoniox(ScriptedServer(script, honor_finalize=False)) as server:
        stream = SonioxSTT(api_key="k", base_url=server.url).stream()
        for frame in chunks(synth_speech(0.5, 16_000)):
            stream.push_audio(frame)
        stream.end_input()
        events = await collect(stream)
        await stream.aclose()
    assert kinds(events)[-2:] == [(E.FINAL_TRANSCRIPT, "Goodbye"), (E.END_OF_SPEECH, "Goodbye")]


async def test_keepalive_is_sent_while_idle() -> None:
    async with FakeSoniox(ScriptedServer([])) as server:
        stream = SonioxSTT(api_key="k", base_url=server.url, keepalive_interval=0.1).stream()
        stream.push_audio(AudioFrame.silence(0.1, 16_000))

        def two_keepalives() -> bool:
            controls = server.connections[0].controls() if server.connections else []
            return controls[:2] == ["keepalive", "keepalive"]

        await asyncio.wait_for(_until(two_keepalives), 5)
        stream.end_input()
        await collect(stream)
        await stream.aclose()


async def test_transcribe_streams_the_audio() -> None:
    script = [(0.2, resp(tok("Batch", 50, 200, final=False)))]
    async with FakeSoniox(ScriptedServer(script)) as server:
        stt = SonioxSTT(api_key="k", base_url=server.url)
        result = await stt.transcribe(synth_speech(0.5, 24_000))
        await stt.aclose()
    assert result.text == "Batch"
    assert server.connections[0].controls() == ["finalize"]


# ------------------------------------------------------------------------ errors
@pytest.mark.parametrize(
    ("code", "error_type", "error", "retryable"),
    [
        (401, "unauthenticated", AuthenticationError, False),
        (403, "temp_api_key_session_expired", AuthenticationError, False),
        (402, "organization_balance_exhausted", ProviderError, False),
        (400, "invalid_request", ProviderError, False),
        (408, "request_timeout", ProviderTimeoutError, True),
        (413, "max_duration_reached", ProviderConnectionError, True),
        (429, "limit_exceeded", RateLimitError, True),
        (503, "service_unavailable", ProviderError, True),
    ],
)
async def test_error_responses_are_mapped(
    code: int, error_type: str, error: type[ProviderError], retryable: bool
) -> None:
    async def handler(conn: FakeConn) -> None:
        await conn.send(
            {
                "tokens": [],
                "error_code": code,
                "error_type": error_type,
                "error_message": "Something went wrong.",
                "request_id": "3d37a3bd",
            }
        )
        await conn.ws.close(1000)

    async with FakeSoniox(handler) as server:
        stream = SonioxSTT(api_key="k", base_url=server.url).stream()
        stream.push_audio(AudioFrame.silence(0.1, 16_000))
        with pytest.raises(error) as info:
            await collect(stream)
        await stream.aclose()
    assert type(info.value) is error and info.value.retryable is retryable
    assert info.value.status_code == code and error_type in str(info.value)


async def test_server_closing_mid_stream_is_a_connection_error() -> None:
    async def handler(conn: FakeConn) -> None:
        await conn.ws.recv()
        await conn.ws.close(1011, "internal error")

    async with FakeSoniox(handler) as server:
        stream = SonioxSTT(api_key="k", base_url=server.url).stream()
        stream.push_audio(AudioFrame.silence(0.1, 16_000))
        with pytest.raises(ProviderConnectionError):
            await collect(stream)
        await stream.aclose()


@pytest.mark.parametrize(
    ("status", "error"), [(401, AuthenticationError), (429, RateLimitError), (500, ProviderError)]
)
async def test_handshake_errors_are_mapped(status: int, error: type[ProviderError]) -> None:
    async with FakeSoniox(ScriptedServer([]), reject=(status, "nope")) as server:
        stream = SonioxSTT(api_key="k", base_url=server.url).stream()
        with pytest.raises(error):
            await collect(stream)
        await stream.aclose()


def test_option_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SONIOX_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="API key"):
        SonioxSTT()
    for bad in (
        {"max_endpoint_delay_ms": 100},
        {"endpoint_sensitivity": 2.0},
        {"endpoint_latency_adjustment_level": 4},
        {"terms": "Soniox"},
        {"region": "mars"},
        {"keepalive_interval": 30},
        {"client_reference_id": "x" * 300},
    ):
        with pytest.raises(ConfigurationError):
            SonioxSTT(api_key="k", **bad)
    monkeypatch.setenv("SONIOX_API_KEY", "env-key")
    stt = SonioxSTT(region="eu", end_of_turn=False, enable_endpoint_detection=True)
    assert stt.url == "wss://stt-rt.eu.soniox.com/transcribe-websocket"
    assert stt.api_url == "https://api.eu.soniox.com"
    assert stt.config()["enable_endpoint_detection"] is True
    assert "api_key" not in stt.config()


def test_registry_specs_create_soniox_stt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SONIOX_API_KEY", "k")
    stt = create("stt", "soniox")
    assert isinstance(stt, SonioxSTT) and stt.model == "stt-rt-v5"
    stt = create("stt", "soniox/stt-rt-v4", language="de", end_of_turn=False)
    assert isinstance(stt, SonioxSTT) and stt.model == "stt-rt-v4"
    assert stt.config()["language_hints"] == ["de"]


async def test_create_temporary_api_key() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.headers["Authorization"] != "Bearer key":
            return httpx.Response(401, json={"error_type": "unauthenticated", "message": "no"})
        return httpx.Response(
            201, json={"api_key": "snx_temp_abc", "expires_at": "2026-09-25T10:00:00Z"}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    stt = SonioxSTT(api_key="key", http_client=client)
    assert await stt.create_temporary_api_key(expires_in_seconds=60, single_use=True) == (
        "snx_temp_abc"
    )
    assert str(seen[0].url) == "https://api.soniox.com/v1/auth/temporary-api-key"
    assert json.loads(seen[0].content) == {
        "usage_type": "transcribe_websocket",
        "expires_in_seconds": 60,
        "single_use": True,
    }
    with pytest.raises(AuthenticationError):
        await SonioxSTT(api_key="bad", http_client=client).create_temporary_api_key()
    await client.aclose()


# ----------------------------------------------------------------- cascade integration
async def test_cascade_commits_on_soniox_endpoint_without_endpointing_delay() -> None:
    server_script = ScriptedServer(
        [
            (0.3, resp(tok("Book", 200, 400, final=False))),
            (0.6, resp(tok("Book", 200, 400), tok(" a", 420, 480, final=False))),
            (0.9, resp(tok(" a", 420, 480), tok(" table.", 500, 880), special("<end>"))),
        ]
    )
    async with FakeSoniox(server_script) as server:
        session = AgentSession(
            stt=SonioxSTT(api_key="k", base_url=server.url),
            llm="mock",
            tts="mock",  # no vad=...: Soniox does the turn-taking
            cascade_options=CascadeOptions(min_endpointing_delay=3.0, max_endpointing_delay=3.0),
        )
        committed: list[float] = []
        finals: list[str] = []
        metrics: list[TurnMetrics] = []

        def on_user(ev: Any) -> None:
            if ev.is_final:
                committed.append(now())
                finals.append(ev.text)

        session.on("user_transcript", on_user)
        session.on("metrics", lambda m: metrics.append(m) if isinstance(m, TurnMetrics) else None)
        transport = LoopbackTransport()
        await session.start(Agent("You are helpful."), transport)
        await transport.play_user_audio(synth_speech(1.0, 16_000), realtime=False)
        await transport.play_user_audio(AudioFrame.silence(0.8, 16_000), realtime=False)
        await asyncio.wait_for(_until(lambda: bool(finals)), 2.5)  # < min_endpointing_delay
        await asyncio.wait_for(_until(lambda: bool(metrics)), 15)
        await session.aclose()

    end_sent = next(t for t, m in server_script.sent if special("<end>") in m["tokens"])
    assert finals == ["Book a table."]
    assert committed[0] - end_sent < 1.0
    history = [(i.role, i.text) for i in session.history.items if isinstance(i, ChatMessage)]
    assert history == [("user", "Book a table."), ("assistant", "You said: Book a table.")]


async def test_cascade_with_vad_owns_turns_via_finalize() -> None:
    server_script = ScriptedServer(
        [
            (0.3, resp(tok("What", 100, 300, final=False))),
            (
                0.7,
                resp(
                    tok("What", 100, 300),
                    tok(" time", 320, 500, final=False),
                    tok(" is", 520, 600, final=False),
                    tok(" it", 610, 680, final=False),
                ),
            ),
        ]
    )
    async with FakeSoniox(server_script) as server:
        session = AgentSession(
            stt=SonioxSTT(api_key="k", base_url=server.url, end_of_turn=False),
            llm="mock",
            tts="mock",
            vad=EnergyVAD(),
            cascade_options=CascadeOptions(min_endpointing_delay=0.2, max_endpointing_delay=0.2),
        )
        finals: list[str] = []
        session.on("user_transcript", lambda ev: finals.append(ev.text) if ev.is_final else None)
        transport = LoopbackTransport()
        await session.start(Agent("You are helpful."), transport)
        await transport.play_user_audio(synth_speech(0.9, 16_000), realtime=False)
        await transport.play_user_audio(AudioFrame.silence(1.0, 16_000), realtime=False)
        await asyncio.wait_for(_until(lambda: bool(finals)), 5)
        await session.aclose()
    assert finals == ["What time is it"]
    assert "finalize" in server.connections[0].controls()


# ------------------------------------------------------------------- real API (opt-in)
needs_key = pytest.mark.skipif(not os.environ.get("SONIOX_API_KEY"), reason="needs SONIOX_API_KEY")


async def _speech() -> AudioFrame:
    """Real speech from a TTS whose key is available, else skip."""
    if os.environ.get("DEEPGRAM_API_KEY"):
        from voice_agent_next.providers.deepgram import DeepgramTTS

        tts = DeepgramTTS(sample_rate=16_000)
        try:
            return await tts.synthesize("The quick brown fox jumps over the lazy dog.").collect()
        finally:
            await tts.aclose()
    pytest.skip("needs DEEPGRAM_API_KEY to synthesize test speech")


@pytest.mark.integration
@needs_key
async def test_integration_session_protocol_on_silence() -> None:
    stream = SonioxSTT().stream()
    for frame in chunks(AudioFrame.silence(1.0, 16_000), 0.05):
        stream.push_audio(frame)
        await asyncio.sleep(frame.duration)
    stream.end_input()
    events = await collect(stream)
    await stream.aclose()
    assert [ev.text for ev in events if ev.type == E.FINAL_TRANSCRIPT] == [""]


@pytest.mark.integration
@needs_key
@pytest.mark.parametrize("end_of_turn", [True, False])
async def test_integration_stt_round_trip(end_of_turn: bool) -> None:
    speech = await _speech()
    stream = SonioxSTT(end_of_turn=end_of_turn, terms=["fox"]).stream()
    for frame in chunks(AudioFrame.concat([speech, AudioFrame.silence(2.5, 16_000)]), 0.05):
        stream.push_audio(frame)
        await asyncio.sleep(frame.duration)
    stream.end_input()
    events = await collect(stream)
    await stream.aclose()
    text = " ".join(ev.text for ev in events if ev.type == E.FINAL_TRANSCRIPT).lower()
    assert "fox" in text and "lazy dog" in text
    assert (E.END_OF_TURN in [ev.type for ev in events]) is end_of_turn


@pytest.mark.integration
@needs_key
async def test_integration_temporary_api_key() -> None:
    stt = SonioxSTT()
    try:
        key = await stt.create_temporary_api_key(expires_in_seconds=60)
        stream = SonioxSTT(api_key=key).stream()
        stream.push_audio(AudioFrame.silence(0.5, 16_000))
        stream.end_input()
        await collect(stream)
        await stream.aclose()
    finally:
        await stt.aclose()
