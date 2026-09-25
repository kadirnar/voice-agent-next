"""AssemblyAI Universal-3.5 Pro / Universal-Streaming STT against a local fake server.

The fake replays AssemblyAI's documented v3 streaming messages (``Begin``,
``SpeechStarted``, ``Turn``, ``Termination``, ``Error``; see
https://www.assemblyai.com/docs/streaming/message-sequence) and the Sync STT API's
responses. No network, no API key — except the ``integration`` tests at the end, which
need ``ASSEMBLYAI_API_KEY``.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.datastructures import Headers
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
from voice_agent_next.providers.assemblyai import AssemblyAIStream, AssemblyAISTT
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import synth_speech
from voice_agent_next.registry import create
from voice_agent_next.stt import STTEvent, STTEventType
from voice_agent_next.transports import LoopbackTransport
from voice_agent_next.utils import now

SESSION_ID = "3207b601-2054-48df-ba77-8784dfcf9fb8"
BYTES_PER_SECOND = 16_000 * 2  # pcm_s16le mono @ 16 kHz, what the STT sends
MIN_CHUNK_BYTES = 1600  # 50 ms
E = STTEventType


# ------------------------------------------------------------------------ fake server
@dataclass
class FakeConn:
    """One client connection accepted by the fake server."""

    ws: ServerConnection
    path: str
    query: dict[str, list[str]]
    headers: Headers
    received: list[str | bytes] = field(default_factory=list)

    def messages(self) -> list[dict[str, Any]]:
        return [json.loads(m) for m in self.received if isinstance(m, str)]

    def types(self) -> list[str]:
        return [m["type"] for m in self.messages()]

    def audio_chunks(self) -> list[int]:
        return [len(m) for m in self.received if isinstance(m, bytes)]

    def params(self) -> dict[str, str]:
        return {k: v[0] for k, v in self.query.items()}

    async def send(self, message: dict[str, Any]) -> None:
        await self.ws.send(json.dumps(message))


Handler = Callable[[FakeConn], Awaitable[None]]


class FakeAssemblyAI:
    """A WebSocket server on 127.0.0.1 speaking AssemblyAI's v3 protocol."""

    def __init__(self, handler: Handler, *, reject: tuple[int, str] | None = None) -> None:
        self._handler = handler
        self._reject = reject
        self._server: Server | None = None
        self.connections: list[FakeConn] = []

    async def __aenter__(self) -> FakeAssemblyAI:
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
        return connection.respond(status, json.dumps({"error": message}))

    async def _handle(self, ws: ServerConnection) -> None:
        split = urlsplit(ws.request.path if ws.request else "/")
        headers = ws.request.headers if ws.request else Headers()
        conn = FakeConn(ws, split.path, parse_qs(split.query), headers)
        self.connections.append(conn)
        try:
            await self._handler(conn)
        except ConnectionClosed:
            pass


def words_for(text: str, start_ms: int, end_ms: int, *, final: bool) -> list[dict[str, Any]]:
    tokens = text.split()
    step = (end_ms - start_ms) / max(1, len(tokens))
    return [
        {
            "start": round(start_ms + i * step),
            "end": round(start_ms + (i + 0.8) * step),
            "text": token,
            "confidence": 0.95,
            "word_is_final": final,
        }
        for i, token in enumerate(tokens)
    ]


def turn(
    text: str,
    *,
    order: int = 0,
    end: bool = False,
    formatted: bool | None = None,
    start_ms: int = 200,
    end_ms: int = 1000,
    words: list[dict[str, Any]] | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    """A ``Turn`` message shaped like AssemblyAI's (``utterance`` is set on finals)."""
    msg: dict[str, Any] = {
        "turn_order": order,
        "turn_is_formatted": end if formatted is None else formatted,
        "end_of_turn": end,
        "transcript": text,
        "end_of_turn_confidence": 0.93 if end else 0.0,
        "words": words if words is not None else words_for(text, start_ms, end_ms, final=end),
        "utterance": text if end else "",
        "type": "Turn",
    }
    if language is not None:
        msg["language_code"] = language
        msg["language_confidence"] = 0.97
    return msg


def speech_started(timestamp_ms: int) -> dict[str, Any]:
    return {"type": "SpeechStarted", "timestamp": timestamp_ms, "confidence": 0.98}


def begin(model: str = "universal-3-5-pro") -> dict[str, Any]:
    return {
        "type": "Begin",
        "id": SESSION_ID,
        "expires_at": 1772570132,
        "configuration": {"model": model, "mode": "balanced", "api_version": "2025-05-12"},
    }


@dataclass
class ScriptedServer:
    """Sends ``script`` messages once that much audio (seconds) arrived; tracks turn state
    to answer ``ForceEndpoint`` (the open turn's formatted final) and ``Terminate`` (the
    open turn's final, ``Termination``, close 1000) like AssemblyAI does.

    ``on_force_without_turn`` scripts the answer to a ``ForceEndpoint`` with no open turn
    (by default: none, as nothing is documented for that case).
    """

    script: list[tuple[float, dict[str, Any]]]
    model: str = "universal-3-5-pro"
    honor_force_endpoint: bool = True
    finalize_on_terminate: bool = True
    on_force_without_turn: Callable[[FakeConn, int], Awaitable[None]] | None = None
    sent: list[tuple[float, dict[str, Any]]] = field(default_factory=list)

    async def __call__(self, conn: FakeConn) -> None:
        audio, order, last = 0, 0, ""
        pending = list(self.script)

        async def send(msg: dict[str, Any]) -> None:
            nonlocal order, last
            if msg["type"] == "Turn":
                if msg["end_of_turn"] and (msg["turn_is_formatted"] or "universal-3" in self.model):
                    order, last = msg["turn_order"] + 1, ""
                elif msg["transcript"]:
                    last = msg["transcript"]
            self.sent.append((now(), msg))
            await conn.send(msg)

        async def finalize() -> None:
            await send(turn(last.capitalize() + ".", order=order, end=True, end_ms=audio_ms()))

        def audio_ms() -> int:
            return round(audio / BYTES_PER_SECOND * 1000)

        await send(begin(self.model))
        async for message in conn.ws:
            conn.received.append(message)
            if isinstance(message, bytes):
                if len(message) < MIN_CHUNK_BYTES:
                    ms = len(message) * 1000 // BYTES_PER_SECOND
                    reason = f"Input duration violation: {ms} ms. Expected between 50 and 1000 ms"
                    await send({"type": "Error", "error_code": 3007, "error": reason})
                    await conn.ws.close(3007, reason)
                    return
                audio += len(message)
                while pending and audio >= pending[0][0] * BYTES_PER_SECOND:
                    await send(pending.pop(0)[1])
                continue
            kind = json.loads(message)["type"]
            if kind == "ForceEndpoint" and self.honor_force_endpoint:
                if last:
                    await finalize()
                elif self.on_force_without_turn is not None:
                    await self.on_force_without_turn(conn, order)
            elif kind == "Terminate":
                for _, msg in pending:  # messages for audio still in flight
                    await send(msg)
                if last and self.finalize_on_terminate:
                    await finalize()
                await send(
                    {
                        "type": "Termination",
                        "audio_duration_seconds": round(audio / BYTES_PER_SECOND),
                        "session_duration_seconds": round(audio / BYTES_PER_SECOND) + 1,
                    }
                )
                await conn.ws.close(1000)
                return


def chunks(frame: AudioFrame, step: float = 0.02) -> list[AudioFrame]:
    n = round(step * frame.sample_rate) * 2
    return [
        AudioFrame(frame.data[i : i + n], frame.sample_rate) for i in range(0, len(frame.data), n)
    ]


async def collect(stream: AsyncIterator[STTEvent]) -> list[STTEvent]:
    return [ev async for ev in stream]


def kinds(events: Sequence[STTEvent]) -> list[tuple[STTEventType, str]]:
    return [(ev.type, ev.text) for ev in events]


async def end_after_turn(
    stream: AsyncIterator[STTEvent], end_input: Callable[[], None], turns: int = 1
) -> list[STTEvent]:
    """Collect events; end the input after ``turns`` END_OF_TURN events (so the flush of
    ``end_input()`` cannot be mistaken for the one that ended those turns)."""
    events: list[STTEvent] = []
    async for ev in stream:
        events.append(ev)
        if ev.type == E.END_OF_TURN:
            turns -= 1
            if turns == 0:
                end_input()
    return events


async def _until(predicate: Callable[[], bool]) -> None:
    while not predicate():
        await asyncio.sleep(0.01)


# ------------------------------------------------------------ Universal-3.5 Pro turns
U35_TURNS = [
    (0.3, speech_started(220)),
    (0.5, turn("hi i", start_ms=220, end_ms=480)),
    (0.8, turn("hi i need to", start_ms=220, end_ms=780)),
    (0.9, turn("hi i need to", start_ms=220, end_ms=780)),  # unchanged: no event
    (1.2, turn("hi i need to cancel my order", start_ms=220, end_ms=1150)),
    (1.5, turn("Hi, I need to cancel my order.", end=True, start_ms=220, end_ms=1150,
               language="en")),
    (1.9, speech_started(1800)),
    (2.1, turn("thanks", order=1, start_ms=1800, end_ms=2050)),
    (2.4, turn("Thanks.", order=1, end=True, start_ms=1800, end_ms=2050)),
]  # fmt: skip


async def test_u35_turns_emit_interims_finals_and_end_of_turn() -> None:
    async with FakeAssemblyAI(ScriptedServer(list(U35_TURNS))) as server:
        stt = AssemblyAISTT(
            api_key="test-key",
            base_url=server.url,
            keyterms=["AssemblyAI", "Keanu Reeves"],
            mode="balanced",
            min_turn_silence=160,
            max_turn_silence=1500,
            prompt="Customer support call about an order.",
            language_detection=True,
            language="en-US",
        )
        assert stt.capabilities.end_of_turn and stt.capabilities.streaming
        metrics: list[STTMetrics] = []
        stt.on("metrics", metrics.append)
        stream = stt.stream()
        for frame in chunks(synth_speech(2.6, 16_000)):
            stream.push_audio(frame)
        # then ForceEndpoint (no open turn) and Terminate
        events = await end_after_turn(stream, stream.end_input, turns=2)
        await stream.aclose()

    first, second = "Hi, I need to cancel my order.", "Thanks."
    assert kinds(events) == [
        (E.START_OF_SPEECH, ""),
        (E.INTERIM_TRANSCRIPT, "hi i"),
        (E.INTERIM_TRANSCRIPT, "hi i need to"),
        (E.INTERIM_TRANSCRIPT, "hi i need to cancel my order"),
        (E.FINAL_TRANSCRIPT, first),
        (E.END_OF_SPEECH, first),
        (E.END_OF_TURN, first),
        (E.START_OF_SPEECH, ""),
        (E.INTERIM_TRANSCRIPT, "thanks"),
        (E.FINAL_TRANSCRIPT, second),
        (E.END_OF_SPEECH, second),
        (E.END_OF_TURN, second),
        (E.FINAL_TRANSCRIPT, ""),  # end_input's ForceEndpoint found no open turn: ack
    ]
    start = events[0].transcript
    assert start is not None and start.start_time == pytest.approx(0.22)
    final = events[4].transcript
    assert final is not None and final.words is not None and final.words[0].word == "Hi,"
    assert final.start_time == pytest.approx(0.22)
    assert final.end_time == pytest.approx(final.words[-1].end)
    assert final.end_time is not None and 0.9 < final.end_time < 1.15  # stream seconds
    assert final.confidence == pytest.approx(0.95) and final.language == "en"
    assert len({ev.segment_id for ev in events[:7]}) == 1
    assert events[7].segment_id != events[0].segment_id
    assert isinstance(stream, AssemblyAIStream)
    assert stream.session_id == SESSION_ID
    assert stream.configuration["model"] == "universal-3-5-pro"

    conn = server.connections[0]
    assert conn.path == "/v3/ws" and conn.headers["Authorization"] == "test-key"
    assert conn.params() == {
        "speech_model": "universal-3-5-pro",
        "encoding": "pcm_s16le",
        "sample_rate": "16000",
        "mode": "balanced",
        "language_codes": '["en"]',
        "language_detection": "true",
        "min_turn_silence": "160",
        "max_turn_silence": "1500",
        "keyterms_prompt": '["AssemblyAI","Keanu Reeves"]',
        "prompt": "Customer support call about an order.",
    }
    assert set(conn.audio_chunks()) == {MIN_CHUNK_BYTES}  # 50 ms chunks
    assert conn.types() == ["ForceEndpoint", "Terminate"]
    assert sum(m.audio_duration for m in metrics) == pytest.approx(2.6, abs=0.01)


async def test_flush_sends_force_endpoint_and_owns_the_turn() -> None:
    """Cascade-owned turns (``end_of_turn=False``): the flush ends the open turn; a flush
    with no open turn is acknowledged with an empty final after the grace period."""
    script = [
        (0.3, speech_started(150)),
        (0.5, turn("my account number is", start_ms=150, end_ms=480)),
        (0.8, turn("my account number is four two", start_ms=150, end_ms=780)),
    ]
    async with FakeAssemblyAI(ScriptedServer(script)) as server:
        stt = AssemblyAISTT(
            api_key="k", base_url=server.url, end_of_turn=False, force_endpoint_grace=0.2
        )
        assert not stt.capabilities.end_of_turn
        metrics: list[STTMetrics] = []
        stt.on("metrics", metrics.append)
        stream = stt.stream()
        for frame in chunks(synth_speech(0.9, 16_000)):
            stream.push_audio(frame)
        stream.flush()  # e.g. the cascade's VAD saw the end of speech
        events: list[STTEvent] = []
        acked_at = 0.0
        async for ev in stream:
            events.append(ev)
            if ev.type == E.END_OF_SPEECH:
                flushed_at = now()
                stream.flush()  # no open turn: acknowledged after force_endpoint_grace
            elif ev.type == E.FINAL_TRANSCRIPT and not ev.text:
                acked_at = now()
                break
        await stream.aclose()

    text = "My account number is four two."
    assert kinds(events) == [
        (E.START_OF_SPEECH, ""),
        (E.INTERIM_TRANSCRIPT, "my account number is"),
        (E.INTERIM_TRANSCRIPT, "my account number is four two"),
        (E.FINAL_TRANSCRIPT, text),
        (E.END_OF_SPEECH, text),
        (E.FINAL_TRANSCRIPT, ""),
    ]  # no END_OF_TURN: the cascade owns the turn
    assert 0.15 <= acked_at - flushed_at < 1.5
    assert server.connections[0].types()[:2] == ["ForceEndpoint", "ForceEndpoint"]
    assert len(metrics) == 2 and all(m.latency is not None for m in metrics)  # flush answered


async def test_flushed_turn_gets_no_end_of_turn_in_stt_turn_mode() -> None:
    script = [(0.3, speech_started(100)), (0.5, turn("wait", start_ms=100, end_ms=400))]
    async with FakeAssemblyAI(ScriptedServer(script)) as server:
        stream = AssemblyAISTT(api_key="k", base_url=server.url).stream()
        for frame in chunks(synth_speech(0.6, 16_000)):
            stream.push_audio(frame)
        stream.flush()  # push-to-talk release: whoever flushed owns the decision
        events = [ev async for ev in _until_final(stream)]
        await stream.aclose()
    assert kinds(events) == [
        (E.START_OF_SPEECH, ""),
        (E.INTERIM_TRANSCRIPT, "wait"),
        (E.FINAL_TRANSCRIPT, "Wait."),
        (E.END_OF_SPEECH, "Wait."),
    ]


async def _until_final(stream: AsyncIterator[STTEvent]) -> AsyncIterator[STTEvent]:
    async for ev in stream:
        yield ev
        if ev.type == E.END_OF_SPEECH:
            return


async def test_flush_before_the_first_partial_waits_for_the_turn() -> None:
    """A short utterance can be flushed before AssemblyAI emitted anything for it (U3.5's
    ``interruption_delay``): a turn starting within the grace period answers the flush,
    not an empty final."""

    async def late_turn(conn: FakeConn, order: int) -> None:
        await asyncio.sleep(0.1)
        await conn.send(speech_started(300))
        await conn.send(turn("Yes.", order=order, end=True, start_ms=300, end_ms=500))

    server_script = ScriptedServer([], on_force_without_turn=late_turn)
    async with FakeAssemblyAI(server_script) as server:
        stt = AssemblyAISTT(api_key="k", base_url=server.url, force_endpoint_grace=1.0)
        stream = stt.stream()
        for frame in chunks(synth_speech(0.6, 16_000)):
            stream.push_audio(frame)
        stream.flush()
        events = [ev async for ev in _until_final(stream)]
        await stream.aclose()
    assert kinds(events) == [
        (E.START_OF_SPEECH, ""),
        (E.FINAL_TRANSCRIPT, "Yes."),
        (E.END_OF_SPEECH, "Yes."),
    ]


async def test_flush_holds_back_a_tail_under_50_ms_and_end_input_pads_it() -> None:
    async with FakeAssemblyAI(ScriptedServer([])) as server:
        stt = AssemblyAISTT(api_key="k", base_url=server.url, force_endpoint_grace=0)
        stream = stt.stream()
        stream.push_audio(AudioFrame.silence(0.13, 16_000))  # 2 chunks + 30 ms
        stream.flush()
        first = await stream.__anext__()  # acknowledged at once (grace 0)
        stream.push_audio(AudioFrame.silence(0.01, 16_000))
        stream.end_input()
        events = [first, *await collect(stream)]
        await stream.aclose()
    assert kinds(events) == [(E.FINAL_TRANSCRIPT, ""), (E.FINAL_TRANSCRIPT, "")]
    conn = server.connections[0]
    assert conn.audio_chunks() == [1600, 1600, 1600]  # 30 + 10 ms padded to 50 ms
    assert conn.types() == ["ForceEndpoint", "ForceEndpoint", "Terminate"]


async def test_terminate_turns_the_last_partial_into_a_final() -> None:
    script = [(0.3, speech_started(100)), (0.5, turn("so the", start_ms=100, end_ms=450))]
    server_script = ScriptedServer(script, honor_force_endpoint=False, finalize_on_terminate=False)
    async with FakeAssemblyAI(server_script) as server:
        stream = AssemblyAISTT(api_key="k", base_url=server.url).stream()
        for frame in chunks(synth_speech(0.6, 16_000)):
            stream.push_audio(frame)
        stream.end_input()
        events = await collect(stream)
    assert kinds(events)[-2:] == [(E.FINAL_TRANSCRIPT, "so the"), (E.END_OF_SPEECH, "so the")]
    assert E.END_OF_TURN not in [ev.type for ev in events]  # our flush ended it


# ------------------------------------------------------------ Universal-Streaming
async def test_universal_streaming_format_turns_emits_only_the_formatted_final() -> None:
    """Universal-Streaming has no ``SpeechStarted``, keeps the word being decoded out of
    partial ``transcript`` and, with ``format_turns``, sends each final twice."""
    partial_words = words_for("book a tab", 300, 900, final=True)
    partial_words[-1]["word_is_final"] = False
    script = [
        (0.5, turn("book a", start_ms=300, end_ms=900, words=partial_words, formatted=False)),
        (0.9, turn("book a table", end=True, formatted=False, start_ms=300, end_ms=1000)),
        (0.9, turn("Book a table.", end=True, formatted=True, start_ms=300, end_ms=1000)),
    ]
    model = "universal-streaming-english"
    async with FakeAssemblyAI(ScriptedServer(script, model=model)) as server:
        stt = AssemblyAISTT(
            model=model,
            api_key="k",
            base_url=server.url,
            format_turns=True,
            end_of_turn_confidence_threshold=0.7,
            min_turn_silence=400,
            language="en",  # the English model takes no language codes
        )
        stream = stt.stream()
        for frame in chunks(synth_speech(1.1, 16_000)):
            stream.push_audio(frame)
        events = await end_after_turn(stream, stream.end_input)
    assert kinds(events) == [
        (E.START_OF_SPEECH, ""),
        (E.INTERIM_TRANSCRIPT, "book a tab"),
        (E.INTERIM_TRANSCRIPT, "book a table"),  # the unformatted copy of the final
        (E.FINAL_TRANSCRIPT, "Book a table."),
        (E.END_OF_SPEECH, "Book a table."),
        (E.END_OF_TURN, "Book a table."),
        (E.FINAL_TRANSCRIPT, ""),
    ]
    assert server.connections[0].params() == {
        "speech_model": model,
        "encoding": "pcm_s16le",
        "sample_rate": "16000",
        "min_turn_silence": "400",
        "end_of_turn_confidence_threshold": "0.7",
        "format_turns": "true",
    }


async def test_universal_streaming_unformatted_final_without_format_turns() -> None:
    script = [
        (0.4, turn("hello", start_ms=100, end_ms=400, formatted=False)),
        (0.7, turn("hello there", end=True, formatted=False, start_ms=100, end_ms=650)),
    ]
    model = "universal-streaming-multilingual"
    async with FakeAssemblyAI(ScriptedServer(script, model=model)) as server:
        stream = AssemblyAISTT(model=model, api_key="k", base_url=server.url).stream()
        for frame in chunks(synth_speech(0.9, 16_000)):
            stream.push_audio(frame)
        events = await end_after_turn(stream, stream.end_input)
    assert [k for k in kinds(events) if k[0] != E.INTERIM_TRANSCRIPT][:4] == [
        (E.START_OF_SPEECH, ""),
        (E.FINAL_TRANSCRIPT, "hello there"),
        (E.END_OF_SPEECH, "hello there"),
        (E.END_OF_TURN, "hello there"),
    ]


# ---------------------------------------------------------------- session control
async def test_update_configuration_and_token_auth() -> None:
    async with FakeAssemblyAI(ScriptedServer([])) as server:
        stt = AssemblyAISTT(token="temp-token", base_url=server.url)
        stream = stt.stream()
        await stream.update_configuration(
            min_turn_silence=1000, keyterms_prompt=("Kelly Byrne-Donoghue",), mode="balanced"
        )
        with pytest.raises(ConfigurationError, match="format_turns"):
            await stream.update_configuration(format_turns=True)
        stream.end_input()
        await collect(stream)
        with pytest.raises(RuntimeError):
            await stream.update_configuration(mode="balanced")
    conn = server.connections[0]
    assert conn.query["token"] == ["temp-token"] and "Authorization" not in conn.headers
    assert conn.messages()[0] == {
        "type": "UpdateConfiguration",
        "min_turn_silence": 1000,
        "keyterms_prompt": ["Kelly Byrne-Donoghue"],
        "mode": "balanced",
    }
    with pytest.raises(ConfigurationError, match="API key"):
        await stt.transcribe(AudioFrame.silence(0.5, 16_000))  # the Sync API needs a key


async def test_keepalive_is_sent_while_idle_with_an_inactivity_timeout() -> None:
    async with FakeAssemblyAI(ScriptedServer([])) as server:
        stt = AssemblyAISTT(api_key="k", base_url=server.url, inactivity_timeout=5)
        assert server.connections == []
        stt.inactivity_timeout = 0.2  # keepalive every 0.1 s, to keep the test fast
        stream = stt.stream()
        await asyncio.wait_for(
            _until(
                lambda: bool(server.connections) and "KeepAlive" in server.connections[0].types()
            ),
            5,
        )
        stream.end_input()
        await collect(stream)
    assert server.connections[0].params()["inactivity_timeout"] == "0.2"


# ------------------------------------------------------------------------ failures
@pytest.mark.parametrize(
    ("code", "reason", "error", "retryable"),
    [
        (1008, "Unauthorized Connection: Missing Authorization header", AuthenticationError, False),
        (3009, "Unauthorized Connection: Too many concurrent sessions", RateLimitError, True),
        (3008, "Session Expired: Maximum session duration exceeded", ProviderConnectionError, True),
        (3007, "Audio Transmission Rate Exceeded: too much audio buffered", ProviderError, False),
        (3006, "Invalid Message Type: foo", ProviderError, False),
        (3005, "Session Cancelled: An error occurred", ProviderError, True),
    ],
)
async def test_error_messages_are_mapped(
    code: int, reason: str, error: type[ProviderError], retryable: bool
) -> None:
    async def failing(conn: FakeConn) -> None:
        await conn.send(begin())
        await conn.ws.recv()
        await conn.send({"type": "Error", "error_code": code, "error": reason})
        await conn.ws.close(code, reason[:120])

    async with FakeAssemblyAI(failing) as server:
        stream = AssemblyAISTT(api_key="k", base_url=server.url).stream()
        stream.push_audio(AudioFrame.silence(0.2, 16_000))
        with pytest.raises(error, match=reason.split(":")[0]) as info:
            await collect(stream)
        await stream.aclose()
    assert type(info.value) is error
    assert info.value.status_code == code and info.value.retryable is retryable
    assert info.value.provider == "assemblyai"


async def test_close_without_error_message_is_mapped() -> None:
    async def closing(conn: FakeConn) -> None:
        await conn.ws.close(1008, "Unauthorized connection: Too many concurrent sessions")

    async with FakeAssemblyAI(closing) as server:
        stream = AssemblyAISTT(api_key="k", base_url=server.url).stream()
        stream.push_audio(AudioFrame.silence(0.2, 16_000))
        with pytest.raises(RateLimitError, match="concurrent") as info:
            await collect(stream)
        await stream.aclose()
    assert info.value.status_code == 1008 and info.value.retryable


async def test_server_closing_mid_stream_is_a_connection_error() -> None:
    async def dropping(conn: FakeConn) -> None:
        await conn.send(begin())
        await conn.ws.recv()
        await conn.ws.close(1011, "Internal error")

    async with FakeAssemblyAI(dropping) as server:
        stream = AssemblyAISTT(api_key="k", base_url=server.url).stream()
        for frame in chunks(synth_speech(0.4, 16_000)):
            stream.push_audio(frame)
        with pytest.raises(ProviderConnectionError, match="1011") as info:
            await collect(stream)
        await stream.aclose()
    assert info.value.retryable


@pytest.mark.parametrize(
    ("status", "error"),
    [(401, AuthenticationError), (429, RateLimitError), (400, ProviderError),
     (503, ProviderError)],
)  # fmt: skip
async def test_handshake_errors_are_mapped(status: int, error: type[ProviderError]) -> None:
    async with FakeAssemblyAI(ScriptedServer([]), reject=(status, "Invalid API key")) as server:
        stream = AssemblyAISTT(api_key="bad", base_url=server.url).stream()
        stream.push_audio(AudioFrame.silence(0.1, 16_000))
        with pytest.raises(error, match="Invalid API key") as info:
            await collect(stream)
        await stream.aclose()
    assert info.value.status_code == status and info.value.retryable == (status in (429, 503))


async def test_connection_refused_is_a_connection_error() -> None:
    async with FakeAssemblyAI(ScriptedServer([])) as server:
        url = server.url
    stream = AssemblyAISTT(api_key="k", base_url=url, connect_timeout=5).stream()  # gone
    # refused at once on Linux/macOS; Windows retries the SYN and may hit the timeout first
    with pytest.raises((ProviderConnectionError, ProviderTimeoutError)) as info:
        await collect(stream)
    assert info.value.retryable
    await stream.aclose()


def test_option_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="ASSEMBLYAI_API_KEY"):
        AssemblyAISTT()
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "from-env")
    assert AssemblyAISTT()._api_key == "from-env"
    with pytest.raises(ConfigurationError, match="Universal-Streaming only"):
        AssemblyAISTT(format_turns=True)
    with pytest.raises(ConfigurationError, match=r"Universal-3\.5 Pro only"):
        AssemblyAISTT(model="universal-streaming-english", mode="min_latency")
    with pytest.raises(ConfigurationError, match="100 keyterms"):
        AssemblyAISTT(keyterms=[f"term{i}" for i in range(101)])
    with pytest.raises(ConfigurationError, match="50 characters"):
        AssemblyAISTT(keyterms=["x" * 51])
    with pytest.raises(ConfigurationError, match="sequence of strings"):
        AssemblyAISTT(keyterms="AssemblyAI")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="sample_rate"):
        AssemblyAISTT(sample_rate=4_000)
    with pytest.raises(ConfigurationError, match="vad_threshold"):
        AssemblyAISTT(vad_threshold=1.5)
    with pytest.raises(ConfigurationError, match="chunk_ms"):
        AssemblyAISTT(chunk_ms=20)
    with pytest.raises(ConfigurationError, match="mode"):
        AssemblyAISTT(mode="fast")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="region"):
        AssemblyAISTT(region="ap")  # type: ignore[arg-type]
    assert AssemblyAISTT(region="eu").url().startswith("wss://streaming.eu.assemblyai.com/v3/ws?")
    assert AssemblyAISTT().url().startswith("wss://streaming.assemblyai.com/v3/ws?")


def test_registry_specs_create_assemblyai_stt() -> None:
    stt = create("stt", "assemblyai", api_key="k")
    assert isinstance(stt, AssemblyAISTT) and stt.model == "universal-3-5-pro" and stt.is_pro
    english = create("stt", "assemblyai/universal-streaming-english", api_key="k")
    assert isinstance(english, AssemblyAISTT) and not english.is_pro
    assert english.capabilities.end_of_turn


# ------------------------------------------------------------------ Sync STT (batch)
def sync_app(
    seen: list[httpx.Request], *, status: int = 200, body: dict[str, Any] | None = None
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/v3/token":
            return httpx.Response(200, json={"token": "temp-123", "expires_in_seconds": 60})
        return httpx.Response(
            status,
            json=body
            or {
                "text": "Hi, I'm calling about my order.",
                "words": [
                    {"text": "Hi,", "start": 0, "end": 200, "confidence": 0.91},
                    {"text": "I'm", "start": 220, "end": 320, "confidence": 0.89},
                ],
                "confidence": 0.9,
                "audio_duration_ms": 1000,
                "session_id": "eb92c4ff-4bbb-429f-9b99-7279d7fe738f",
                "request_time_ms": 243.7,
            },
        )

    return httpx.MockTransport(handler)


async def test_transcribe_uses_the_sync_api() -> None:
    seen: list[httpx.Request] = []
    async with httpx.AsyncClient(transport=sync_app(seen)) as client:
        stt = AssemblyAISTT(api_key="key", http_client=client, keyterms=["Best Buy"])
        result = await stt.transcribe(synth_speech(1.0, 24_000), language="en")
        silence = await stt.transcribe(AudioFrame.silence(0.05, 16_000))  # < 80 ms
    assert result.text == "Hi, I'm calling about my order."
    assert result.words is not None and result.words[1].start == pytest.approx(0.22)
    assert result.confidence == pytest.approx(0.9) and result.language == "en"
    assert silence.text == "" and len(seen) == 1
    request = seen[0]
    assert str(request.url) == "https://sync.assemblyai.com/v1/transcribe"
    assert request.headers["Authorization"] == "key"
    assert request.headers["X-AAI-Model"] == "universal-3-5-pro"
    body = request.content
    assert b'name="audio"; filename="audio.pcm"' in body and b"Content-Type: audio/pcm" in body
    config = json.loads(body.split(b'name="config"')[1].split(b"\r\n\r\n", 1)[1].split(b"\r\n")[0])
    assert config == {
        "sample_rate": 16_000,  # resampled to the STT's rate
        "channels": 1,
        "timestamps": True,
        "language_codes": ["en"],
        "keyterms_prompt": ["Best Buy"],
    }


@pytest.mark.parametrize(
    ("status", "body", "error", "retryable"),
    [
        (401, {"detail": "Invalid API key"}, AuthenticationError, False),
        (429, {"detail": "Too many requests"}, RateLimitError, True),
        (503, {"error_code": "capacity_exceeded", "message": "at capacity"}, ProviderConnectionError,
         True),
        (413, {"error_code": "audio_too_large", "message": "too long"}, ProviderError, False),
        (504, {"error_code": "inference_timeout", "message": "deadline"}, ProviderTimeoutError,
         True),
    ],
)  # fmt: skip
async def test_sync_api_errors_are_mapped(
    status: int, body: dict[str, Any], error: type[ProviderError], retryable: bool
) -> None:
    seen: list[httpx.Request] = []
    async with httpx.AsyncClient(transport=sync_app(seen, status=status, body=body)) as client:
        stt = AssemblyAISTT(api_key="key", http_client=client)
        with pytest.raises(error) as info:
            await stt.transcribe(AudioFrame.silence(0.5, 16_000))
    assert type(info.value) is error
    assert info.value.status_code == status and info.value.retryable is retryable
    detail = body.get("detail") or body["message"]
    assert detail in str(info.value)


async def test_create_temporary_token() -> None:
    seen: list[httpx.Request] = []
    async with httpx.AsyncClient(transport=sync_app(seen)) as client:
        stt = AssemblyAISTT(api_key="key", http_client=client, region="eu")
        token = await stt.create_temporary_token(
            expires_in_seconds=60, max_session_duration_seconds=600
        )
    assert token == "temp-123"
    assert str(seen[0].url) == (
        "https://streaming.eu.assemblyai.com/v3/token"
        "?expires_in_seconds=60&max_session_duration_seconds=600"
    )
    assert seen[0].headers["Authorization"] == "key"


# ----------------------------------------------------------------- cascade integration
async def test_cascade_commits_on_assemblyai_end_of_turn_without_endpointing_delay() -> None:
    """AssemblyAI's end-of-turn commits the user turn at once, with no VAD in the cascade:
    the endpointing delay (3 s here) would otherwise hold the commit back."""
    server_script = ScriptedServer(
        [
            (0.3, speech_started(200)),
            (0.6, turn("book a", start_ms=200, end_ms=550)),
            (0.9, turn("book a table for two", start_ms=200, end_ms=880)),
            (1.1, turn("Book a table for two.", end=True, start_ms=200, end_ms=900)),
        ]
    )
    async with FakeAssemblyAI(server_script) as server:
        stt = AssemblyAISTT(api_key="k", base_url=server.url)
        session = AgentSession(
            stt=stt,
            llm="mock",
            tts="mock",  # no vad=...: AssemblyAI does the turn-taking
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
        await transport.play_user_audio(synth_speech(1.2, 16_000), realtime=False)
        await transport.play_user_audio(AudioFrame.silence(0.8, 16_000), realtime=False)
        await asyncio.wait_for(_until(lambda: bool(finals)), 2.5)  # < min_endpointing_delay
        await asyncio.wait_for(_until(lambda: bool(metrics)), 15)  # after the reply played out
        await session.aclose()

    eot_sent = next(t for t, m in server_script.sent if m.get("end_of_turn"))
    assert finals == ["Book a table for two."]
    assert committed[0] - eot_sent < 1.0  # committed right away, not after 3 s
    history = [(i.role, i.text) for i in session.history.items if isinstance(i, ChatMessage)]
    assert history == [
        ("user", "Book a table for two."),
        ("assistant", "You said: Book a table for two."),
    ]
    assert metrics[0].end_of_turn_delay is not None and metrics[0].end_of_turn_delay < 0.5


async def test_cascade_with_vad_owns_turns_via_force_endpoint() -> None:
    """``end_of_turn=False`` + a local VAD: the cascade's flush ends the turn with
    ``ForceEndpoint`` and commits the forced final."""
    server_script = ScriptedServer(
        [
            (0.3, speech_started(100)),
            (0.7, turn("what time is it", start_ms=100, end_ms=650)),
        ]
    )
    async with FakeAssemblyAI(server_script) as server:
        stt = AssemblyAISTT(api_key="k", base_url=server.url, end_of_turn=False)
        session = AgentSession(
            stt=stt,
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
    assert finals == ["What time is it."]
    assert "ForceEndpoint" in server.connections[0].types()


# ------------------------------------------------------------------- real API (opt-in)
needs_key = pytest.mark.skipif(
    not os.environ.get("ASSEMBLYAI_API_KEY"), reason="needs ASSEMBLYAI_API_KEY"
)


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
@pytest.mark.parametrize("model", ["universal-3-5-pro", "universal-streaming-english"])
async def test_integration_session_protocol_on_silence(model: str) -> None:
    stream = AssemblyAISTT(model=model).stream()
    for frame in chunks(AudioFrame.silence(1.0, 16_000), 0.05):
        stream.push_audio(frame)
        await asyncio.sleep(frame.duration)  # real time
    await stream.update_configuration(max_turn_silence=2000)
    stream.end_input()
    events = await collect(stream)
    await stream.aclose()
    assert stream.session_id and stream.configuration.get("model") == model
    assert [ev.text for ev in events if ev.type == E.FINAL_TRANSCRIPT] == [""]


@pytest.mark.integration
@needs_key
@pytest.mark.parametrize("model", ["universal-3-5-pro", "universal-streaming-english"])
async def test_integration_stt_round_trip(model: str) -> None:
    speech = await _speech()
    stream = AssemblyAISTT(model=model).stream()
    for frame in chunks(AudioFrame.concat([speech, AudioFrame.silence(2.0, 16_000)]), 0.05):
        stream.push_audio(frame)
        await asyncio.sleep(frame.duration)  # real time
    stream.end_input()
    events = await collect(stream)
    await stream.aclose()
    text = " ".join(ev.text for ev in events if ev.type == E.FINAL_TRANSCRIPT).lower()
    assert "fox" in text and "lazy dog" in text
    assert events[0].type == E.START_OF_SPEECH
    assert E.END_OF_TURN in [ev.type for ev in events]


@pytest.mark.integration
@needs_key
async def test_integration_sync_api_and_temporary_token() -> None:
    stt = AssemblyAISTT()
    try:
        token = await stt.create_temporary_token(expires_in_seconds=60)
        stream = AssemblyAISTT(api_key="", token=token).stream()
        stream.push_audio(AudioFrame.silence(0.5, 16_000))
        stream.end_input()
        await collect(stream)
        await stream.aclose()
        if os.environ.get("DEEPGRAM_API_KEY"):
            result = await stt.transcribe(await _speech())
            assert "fox" in result.text.lower()
            assert result.words and result.words[-1].end > result.words[0].start
    finally:
        await stt.aclose()
