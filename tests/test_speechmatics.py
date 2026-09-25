"""Speechmatics Agent STT (Linden) and Realtime STT against a local fake server.

The fake replays Speechmatics' documented messages: ``StartRecognition`` ->
``RecognitionStarted``, binary audio -> ``AudioAdded``, ``ForceEndOfUtterance``,
``EndOfStream`` -> ``EndOfTranscript``; Agent STT ``AddPartialSegment`` / ``AddSegment`` /
``StartOfTurn`` / ``EndOfTurn`` / ``SpeechStarted`` / ``SpeechEnded``
(https://docs.speechmatics.com/api-ref/agent-stt-websocket) and Realtime
``AddPartialTranscript`` / ``AddTranscript`` / ``EndOfUtterance``
(https://docs.speechmatics.com/rt-api-ref); ``Error`` / ``Warning`` / ``Info``. No network,
no API key — except the ``integration`` tests at the end (``SPEECHMATICS_API_KEY``).
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
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import synth_speech
from voice_agent_next.providers.speechmatics import SpeechmaticsStream, SpeechmaticsSTT
from voice_agent_next.registry import create
from voice_agent_next.stt import STTEvent, STTEventType
from voice_agent_next.transports import LoopbackTransport
from voice_agent_next.utils import now

SESSION_ID = "4ab09a7a-4d1b-4f79-9f3c-0d2b5c1f9c11"
BYTES_PER_SECOND = 16_000 * 2
E = STTEventType


# ------------------------------------------------------------------------ fake server
@dataclass
class FakeConn:
    ws: ServerConnection
    path: str
    query: dict[str, list[str]]
    headers: Headers
    start: dict[str, Any] = field(default_factory=dict)
    received: list[str | bytes] = field(default_factory=list)

    def messages(self) -> list[dict[str, Any]]:
        return [json.loads(m) for m in self.received if isinstance(m, str)]

    def types(self) -> list[str]:
        return [m["message"] for m in self.messages()]

    async def send(self, message: dict[str, Any]) -> None:
        await self.ws.send(json.dumps(message))


Handler = Callable[[FakeConn], Awaitable[None]]


def recognition_started() -> dict[str, Any]:
    return {
        "message": "RecognitionStarted",
        "id": SESSION_ID,
        "orchestrator_version": "2026.09.1",
        "language_pack_info": {
            "adapted": False,
            "itn": True,
            "language_description": "English",
            "word_delimiter": " ",
            "writing_direction": "left-to-right",
        },
    }


class FakeSpeechmatics:
    """A WebSocket server on 127.0.0.1: answers ``StartRecognition``, then the handler."""

    def __init__(
        self,
        handler: Handler,
        *,
        reject: tuple[int, str] | None = None,
        start_reply: dict[str, Any] | None = None,
    ) -> None:
        self._handler = handler
        self._reject = reject
        self._start_reply = start_reply or recognition_started()
        self._server: Server | None = None
        self.connections: list[FakeConn] = []

    async def __aenter__(self) -> FakeSpeechmatics:
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
        return connection.respond(status, message)

    async def _handle(self, ws: ServerConnection) -> None:
        split = urlsplit(ws.request.path if ws.request else "/")
        headers = ws.request.headers if ws.request else Headers()
        conn = FakeConn(ws, split.path, parse_qs(split.query), headers)
        self.connections.append(conn)
        try:
            conn.start = json.loads(await ws.recv())
            await conn.send(self._start_reply)
            if self._start_reply["message"] != "RecognitionStarted":
                await ws.close(1000)
                return
            await self._handler(conn)
        except ConnectionClosed:
            pass


@dataclass
class ScriptedServer:
    """Sends ``script`` messages once that much audio (seconds) arrived and acknowledges
    audio with ``AudioAdded``. ``on_force`` answers ``ForceEndOfUtterance``;
    ``EndOfStream`` flushes the rest of the script, then ``on_end`` (the provider's last
    results), then ``EndOfTranscript`` and a normal close."""

    script: list[tuple[float, dict[str, Any]]]
    on_force: Callable[[float], list[dict[str, Any]]] | None = None
    on_end: list[dict[str, Any]] = field(default_factory=list)
    sent: list[tuple[float, dict[str, Any]]] = field(default_factory=list)

    async def __call__(self, conn: FakeConn) -> None:
        audio, seq = 0, 0
        pending = list(self.script)

        async def send(msg: dict[str, Any]) -> None:
            self.sent.append((now(), msg))
            await conn.send(msg)

        async for message in conn.ws:
            conn.received.append(message)
            if isinstance(message, bytes):
                audio += len(message)
                seq += 1
                await conn.send({"message": "AudioAdded", "seq_no": seq})
                while pending and audio >= pending[0][0] * BYTES_PER_SECOND:
                    await send(pending.pop(0)[1])
                continue
            msg = json.loads(message)
            if msg["message"] == "ForceEndOfUtterance" and self.on_force is not None:
                for reply in self.on_force(msg["timestamp"]):
                    await send(reply)
            elif msg["message"] == "EndOfStream":
                assert msg["last_seq_no"] == seq
                for _, reply in pending:
                    await send(reply)
                for reply in self.on_end:
                    await send(reply)
                await send({"message": "EndOfTranscript"})
                await conn.ws.close(1000)
                return


# Agent STT messages ---------------------------------------------------------------
def segment(text: str, start: float, end: float, *, final: bool = True, speaker: str = "S1"):
    return {
        "message": "AddSegment" if final else "AddPartialSegment",
        "metadata": {"start_time": start, "end_time": end},
        "segment": {"transcript": text, "speaker": speaker},
    }


def timed(kind: str, t: float) -> dict[str, Any]:
    key = "start_time" if kind in ("StartOfTurn", "SpeechStarted") else "end_time"
    return {"message": kind, "metadata": {key: t}}


# Realtime messages ----------------------------------------------------------------
def word(content: str, start: float, end: float, confidence: float = 0.96) -> dict[str, Any]:
    return {
        "type": "word",
        "start_time": start,
        "end_time": end,
        "alternatives": [{"content": content, "confidence": confidence, "language": "en"}],
        "is_eos": False,
    }


def punct(content: str, t: float) -> dict[str, Any]:
    return {
        "type": "punctuation",
        "start_time": t,
        "end_time": t,
        "attaches_to": "previous",
        "alternatives": [{"content": content, "confidence": 1.0}],
        "is_eos": content in ".?!",
    }


def transcript(*results: dict[str, Any], final: bool = True) -> dict[str, Any]:
    text = " ".join(r["alternatives"][0]["content"] for r in results)
    times = [r["start_time"] for r in results] or [0.0]
    ends = [r["end_time"] for r in results] or [0.0]
    return {
        "message": "AddTranscript" if final else "AddPartialTranscript",
        "format": "2.9",
        "metadata": {"start_time": min(times), "end_time": max(ends), "transcript": text},
        "results": list(results),
    }


def end_of_utterance(t: float, *, forced: bool = False) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "message": "EndOfUtterance",
        "metadata": {"start_time": t, "end_time": t},
        "channel": "0",
    }
    if forced:
        msg["forced"] = True
    return msg


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


# ------------------------------------------------------------ Agent STT, service turns
AGENT_TURNS = [
    (0.3, timed("SpeechStarted", 0.2)),
    (0.4, timed("StartOfTurn", 0.24)),
    (0.5, segment("Hi, I need", 0.24, 0.5, final=False)),
    (0.8, segment("Hi, I need to cancel", 0.24, 0.78, final=False)),
    (0.9, segment("Hi, I need to cancel", 0.24, 0.78, final=False)),  # unchanged: no event
    (1.1, segment("Hi, I need to cancel my order.", 0.24, 1.02)),
    (1.3, timed("SpeechEnded", 1.05)),
    (1.4, timed("EndOfTurn", 1.3)),
    (1.8, timed("StartOfTurn", 1.7)),
    (2.0, segment("Thanks.", 1.7, 1.95)),
    (2.1, segment("Bye.", 2.0, 2.05)),
    (2.3, timed("EndOfTurn", 2.25)),
]


async def test_agent_stt_turns_emit_segments_and_end_of_turn() -> None:
    async with FakeSpeechmatics(ScriptedServer(list(AGENT_TURNS))) as server:
        stt = SpeechmaticsSTT(
            api_key="test-key",
            base_url=server.url,
            language="en-US",
            additional_vocab=["Speechmatics", {"content": "gnocchi", "sounds_like": ["nyohki"]}],
            diarization=True,
            emit_sentences=True,
            force_end_grace=0.1,
        )
        assert stt.is_agent and stt.capabilities.end_of_turn
        assert not stt.capabilities.word_timestamps
        metrics: list[STTMetrics] = []
        stt.on("metrics", metrics.append)
        stream = stt.stream()
        for frame in chunks(synth_speech(2.5, 16_000)):
            stream.push_audio(frame)
        events: list[STTEvent] = []
        turns = 0
        async for ev in stream:
            events.append(ev)
            if ev.type == E.END_OF_TURN:
                turns += 1
                if turns == 2:
                    stream.end_input()  # no open turn: acknowledged after the grace
        await stream.aclose()

    first = "Hi, I need to cancel my order."
    assert kinds(events) == [
        (E.START_OF_SPEECH, ""),
        (E.INTERIM_TRANSCRIPT, "Hi, I need"),
        (E.INTERIM_TRANSCRIPT, "Hi, I need to cancel"),
        (E.FINAL_TRANSCRIPT, first),
        (E.END_OF_SPEECH, first),
        (E.END_OF_TURN, first),
        (E.START_OF_SPEECH, ""),
        (E.FINAL_TRANSCRIPT, "Thanks."),
        (E.FINAL_TRANSCRIPT, "Bye."),
        (E.END_OF_SPEECH, "Thanks. Bye."),
        (E.END_OF_TURN, "Thanks. Bye."),
        (E.FINAL_TRANSCRIPT, ""),  # end_input's flush
    ]
    start = events[0].transcript
    assert start is not None and start.start_time == pytest.approx(0.24)
    final = events[3].transcript
    assert final is not None and final.start_time == pytest.approx(0.24)
    assert final.end_time == pytest.approx(1.02) and final.language == "en"
    end = events[10].transcript
    assert end is not None and end.end_time == pytest.approx(2.05)
    assert len({ev.segment_id for ev in events[:6]}) == 1
    assert events[6].segment_id != events[0].segment_id
    assert isinstance(stream, SpeechmaticsStream)
    assert stream.session_id == SESSION_ID and stream.speaker == "S1"

    conn = server.connections[0]
    assert conn.path == "/v2/agent" and conn.headers["Authorization"] == "Bearer test-key"
    assert conn.start == {
        "message": "StartRecognition",
        "audio_format": {"type": "raw", "encoding": "pcm_s16le", "sample_rate": 16000},
        "transcription_config": {
            "language": "en",
            "model": "linden-1",
            "enable_partials": True,
            "additional_vocab": [
                "Speechmatics",
                {"content": "gnocchi", "sounds_like": ["nyohki"]},
            ],
            "diarization": "speaker",
            "emit_sentences": True,
        },
        "turn_config": {"turn_detection_mode": "vad"},
    }
    assert conn.types() == ["EndOfStream"]  # vad mode: no ForceEndOfUtterance
    assert sum(m.audio_duration for m in metrics) == pytest.approx(2.5, abs=0.01)


# ------------------------------------------------------- Agent STT, external turns
def agent_force(open_turn: list[bool]) -> Callable[[float], list[dict[str, Any]]]:
    """Answer ``ForceEndOfUtterance`` like the service: flush the open segment, end the
    turn (nothing when no turn is open)."""

    def answer(timestamp: float) -> list[dict[str, Any]]:
        if not open_turn[0]:
            return []
        open_turn[0] = False
        return [
            segment("My account number is four two.", 0.15, 0.8),
            timed("EndOfTurn", timestamp),
        ]

    return answer


async def test_agent_stt_external_turns_via_force_end_of_utterance() -> None:
    script = [
        (0.3, timed("StartOfTurn", 0.15)),
        (0.5, segment("My account number", 0.15, 0.45, final=False)),
        (0.8, segment("My account number is four two", 0.15, 0.78, final=False)),
    ]
    server_script = ScriptedServer(script, on_force=agent_force([True]))
    async with FakeSpeechmatics(server_script) as server:
        stt = SpeechmaticsSTT(
            api_key="k", base_url=server.url, end_of_turn=False, force_end_grace=0.2
        )
        assert not stt.capabilities.end_of_turn and stt.can_force_end
        metrics: list[STTMetrics] = []
        stt.on("metrics", metrics.append)
        stream = stt.stream()
        for frame in chunks(synth_speech(0.9, 16_000)):
            stream.push_audio(frame)
        stream.flush()  # e.g. the cascade's VAD saw the end of speech
        events: list[STTEvent] = []
        async for ev in stream:
            events.append(ev)
            if ev.type == E.END_OF_SPEECH:
                flushed_at = now()
                stream.flush()  # no open turn: acknowledged after force_end_grace
            elif ev.type == E.FINAL_TRANSCRIPT and not ev.text:
                acked_at = now()
                break
        stream.end_input()
        events += await collect(stream)
        await stream.aclose()

    text = "My account number is four two."
    assert kinds(events) == [
        (E.START_OF_SPEECH, ""),
        (E.INTERIM_TRANSCRIPT, "My account number"),
        (E.INTERIM_TRANSCRIPT, "My account number is four two"),
        (E.FINAL_TRANSCRIPT, text),
        (E.END_OF_SPEECH, text),
        (E.FINAL_TRANSCRIPT, ""),  # the second flush
        (E.FINAL_TRANSCRIPT, ""),  # end_input's flush
    ]  # no END_OF_TURN: the cascade owns the turn
    assert 0.15 <= acked_at - flushed_at < 1.5
    conn = server.connections[0]
    assert conn.start["turn_config"] == {"turn_detection_mode": "external"}
    forces = [m for m in conn.messages() if m["message"] == "ForceEndOfUtterance"]
    assert len(forces) == 3
    assert forces[0]["timestamp"] == pytest.approx(0.9, abs=0.001)  # audio sent so far
    assert metrics and all(m.latency is not None for m in metrics)


# ------------------------------------------------------------------------ Realtime
RT_UTTERANCES = [
    (0.3, transcript(word("hello", 0.2, 0.45), final=False)),
    (0.6, transcript(word("Hello", 0.2, 0.45))),
    (0.7, transcript(word("world", 0.5, 0.7), final=False)),
    (0.9, transcript(word("world", 0.5, 0.72), punct(".", 0.72))),
    (1.4, end_of_utterance(1.22)),
    (1.7, transcript(word("How", 1.5, 1.6), word("are", 1.6, 1.7), word("you", 1.7, 1.8),
                     punct("?", 1.8))),
    (2.3, end_of_utterance(2.3)),
]  # fmt: skip


async def test_realtime_utterances_emit_interims_finals_and_end_of_turn() -> None:
    server_script = ScriptedServer(list(RT_UTTERANCES), on_force=lambda t: [])
    async with FakeSpeechmatics(server_script) as server:
        stt = SpeechmaticsSTT(
            model="enhanced",
            api_key="test-key",
            base_url=server.url,
            language="en",
            max_delay=1.0,
            end_of_utterance_silence_trigger=0.5,
            additional_vocab=["Speechmatics"],
            domain="finance",
            output_locale="en-US",
            force_end_grace=0.05,
        )
        assert not stt.is_agent and stt.capabilities.word_timestamps
        stream = stt.stream()
        for frame in chunks(synth_speech(2.5, 16_000)):
            stream.push_audio(frame)
        events: list[STTEvent] = []
        turns = 0
        async for ev in stream:
            events.append(ev)
            if ev.type == E.END_OF_TURN:
                turns += 1
                if turns == 2:
                    stream.end_input()
        await stream.aclose()

    first, second = "Hello world.", "How are you?"
    assert kinds(events) == [
        (E.START_OF_SPEECH, ""),
        (E.INTERIM_TRANSCRIPT, "hello"),
        (E.INTERIM_TRANSCRIPT, "Hello"),
        (E.INTERIM_TRANSCRIPT, "Hello world"),
        (E.INTERIM_TRANSCRIPT, first),
        (E.FINAL_TRANSCRIPT, first),
        (E.END_OF_SPEECH, first),
        (E.END_OF_TURN, first),
        (E.START_OF_SPEECH, ""),
        (E.INTERIM_TRANSCRIPT, second),
        (E.FINAL_TRANSCRIPT, second),
        (E.END_OF_SPEECH, second),
        (E.END_OF_TURN, second),
        (E.FINAL_TRANSCRIPT, ""),  # end_input's flush, answered by no EndOfUtterance
    ]
    final = events[5].transcript
    assert final is not None and final.words is not None
    assert [(w.word, w.start, w.end) for w in final.words] == [
        ("Hello", 0.2, 0.45),
        ("world.", 0.5, 0.72),
    ]
    assert final.end_time == pytest.approx(0.72) and final.confidence == pytest.approx(0.96)
    conn = server.connections[0]
    assert conn.path == "/v2"
    assert conn.start["transcription_config"] == {
        "language": "en",
        "model": "enhanced",
        "enable_partials": True,
        "domain": "finance",
        "output_locale": "en-US",
        "additional_vocab": ["Speechmatics"],
        "max_delay": 1.0,
        "conversation_config": {"end_of_utterance_silence_trigger": 0.5},
    }
    assert "turn_config" not in conn.start
    assert conn.types() == ["ForceEndOfUtterance", "EndOfStream"]


async def test_realtime_force_end_of_utterance_owns_the_turn() -> None:
    def answer(timestamp: float) -> list[dict[str, Any]]:
        return [
            transcript(word("four", 0.6, 0.8), word("two", 0.85, 1.0), punct(".", 1.0)),
            end_of_utterance(timestamp, forced=True),
        ]

    script = [
        (0.3, transcript(word("My", 0.1, 0.3), word("number", 0.3, 0.55))),
        (0.8, transcript(word("four", 0.6, 0.8), final=False)),
    ]
    async with FakeSpeechmatics(ScriptedServer(script, on_force=answer)) as server:
        stt = SpeechmaticsSTT(model="standard", api_key="k", base_url=server.url, end_of_turn=False)
        stream = stt.stream()
        for frame in chunks(synth_speech(1.0, 16_000)):
            stream.push_audio(frame)
        stream.flush()
        events: list[STTEvent] = []
        async for ev in stream:
            events.append(ev)
            if ev.type == E.END_OF_SPEECH:
                break
        stream.end_input()
        await collect(stream)
        await stream.aclose()
    text = "My number four two."
    assert kinds(events) == [
        (E.START_OF_SPEECH, ""),
        (E.INTERIM_TRANSCRIPT, "My number"),
        (E.INTERIM_TRANSCRIPT, "My number four"),
        (E.INTERIM_TRANSCRIPT, text),
        (E.FINAL_TRANSCRIPT, text),
        (E.END_OF_SPEECH, text),
    ]


async def test_realtime_end_of_stream_closes_the_open_utterance() -> None:
    script = [(0.3, transcript(word("Goodbye", 0.1, 0.4), final=False))]
    server_script = ScriptedServer(script, on_end=[transcript(word("Goodbye", 0.1, 0.4))])
    async with FakeSpeechmatics(server_script) as server:
        stream = SpeechmaticsSTT(model="enhanced", api_key="k", base_url=server.url).stream()
        for frame in chunks(synth_speech(0.5, 16_000)):
            stream.push_audio(frame)
        await asyncio.wait_for(_until(lambda: bool(server_script.sent)), 5)
        stream.end_input()
        events = await collect(stream)
        await stream.aclose()
    assert kinds(events)[-2:] == [(E.FINAL_TRANSCRIPT, "Goodbye"), (E.END_OF_SPEECH, "Goodbye")]


async def test_set_recognition_config_and_jwt_auth() -> None:
    async with FakeSpeechmatics(ScriptedServer([])) as server:
        stt = SpeechmaticsSTT(model="enhanced", api_key="", jwt="temp-jwt", base_url=server.url)
        stream = stt.stream()
        await stream.set_recognition_config(max_delay=2.0)
        stream.end_input()
        await collect(stream)
        await stream.aclose()
    conn = server.connections[0]
    assert conn.query["jwt"] == ["temp-jwt"] and "Authorization" not in conn.headers
    assert conn.messages()[0] == {
        "message": "SetRecognitionConfig",
        "transcription_config": {"language": "en", "max_delay": 2.0},
    }

    async with FakeSpeechmatics(ScriptedServer([])) as server:
        agent_stream = SpeechmaticsSTT(api_key="k", base_url=server.url).stream()
        with pytest.raises(ConfigurationError):
            await agent_stream.set_recognition_config(max_delay=2.0)
        agent_stream.end_input()
        await collect(agent_stream)
        await agent_stream.aclose()


async def test_transcribe_streams_the_audio() -> None:
    def answer(timestamp: float) -> list[dict[str, Any]]:
        return [
            segment("Batch audio.", 0.05, 0.4),
            timed("EndOfTurn", timestamp),
        ]

    script = [(0.2, timed("StartOfTurn", 0.05))]
    async with FakeSpeechmatics(ScriptedServer(script, on_force=answer)) as server:
        stt = SpeechmaticsSTT(api_key="k", base_url=server.url, end_of_turn=False)
        result = await stt.transcribe(synth_speech(0.5, 24_000))
        await stt.aclose()
    assert result.text == "Batch audio."


# ------------------------------------------------------------------------ errors
@pytest.mark.parametrize(
    ("kind", "error", "retryable"),
    [
        ("not_authorised", AuthenticationError, False),
        ("not_allowed", AuthenticationError, False),
        ("quota_exceeded", RateLimitError, True),
        ("invalid_config", ProviderError, False),
        ("invalid_language", ProviderError, False),
        ("timelimit_exceeded", ProviderError, False),
        ("job_error", ProviderError, True),
        ("idle_timeout", ProviderConnectionError, True),
        ("start_recognition_timeout", ProviderTimeoutError, True),
    ],
)
async def test_error_messages_are_mapped(
    kind: str, error: type[ProviderError], retryable: bool
) -> None:
    reply = {"message": "Error", "type": kind, "reason": "Something went wrong"}
    async with FakeSpeechmatics(ScriptedServer([]), start_reply=reply) as server:
        stream = SpeechmaticsSTT(api_key="k", base_url=server.url).stream()
        stream.push_audio(AudioFrame.silence(0.1, 16_000))
        with pytest.raises(error) as info:
            await collect(stream)
        await stream.aclose()
    assert type(info.value) is error and info.value.retryable is retryable
    assert kind in str(info.value)


async def test_error_mid_stream_is_raised() -> None:
    async def handler(conn: FakeConn) -> None:
        await conn.ws.recv()
        await conn.send({"message": "Warning", "type": "duration_limit_exceeded", "reason": "x"})
        await conn.send({"message": "Error", "type": "protocol_error", "reason": "bad order"})
        await conn.ws.close(1003, "protocol_error")

    async with FakeSpeechmatics(handler) as server:
        stream = SpeechmaticsSTT(api_key="k", base_url=server.url).stream()
        stream.push_audio(AudioFrame.silence(0.1, 16_000))
        with pytest.raises(ProviderError, match="protocol_error"):
            await collect(stream)
        await stream.aclose()


@pytest.mark.parametrize(
    ("code", "error"),
    [
        (4001, AuthenticationError),
        (4005, RateLimitError),
        (4013, ProviderError),
        (1011, ProviderError),
        (4999, ProviderConnectionError),
    ],
)
async def test_close_codes_are_mapped(code: int, error: type[ProviderError]) -> None:
    async def handler(conn: FakeConn) -> None:
        await conn.ws.recv()
        await conn.ws.close(code, "closing")

    async with FakeSpeechmatics(handler) as server:
        stream = SpeechmaticsSTT(api_key="k", base_url=server.url).stream()
        stream.push_audio(AudioFrame.silence(0.1, 16_000))
        with pytest.raises(error) as info:
            await collect(stream)
        await stream.aclose()
    assert info.value.status_code == code


@pytest.mark.parametrize(
    ("status", "error"), [(401, AuthenticationError), (429, RateLimitError), (400, ProviderError)]
)
async def test_handshake_errors_are_mapped(status: int, error: type[ProviderError]) -> None:
    async with FakeSpeechmatics(ScriptedServer([]), reject=(status, "nope")) as server:
        stream = SpeechmaticsSTT(api_key="k", base_url=server.url).stream()
        with pytest.raises(error):
            await collect(stream)
        await stream.aclose()


async def test_no_recognition_started_times_out() -> None:
    async def never(ws: ServerConnection) -> None:  # reads StartRecognition, never answers
        try:
            while True:
                await ws.recv()
        except ConnectionClosed:
            pass

    server = await serve(never, "127.0.0.1", 0)
    port = next(iter(server.sockets)).getsockname()[1]
    try:
        stt = SpeechmaticsSTT(api_key="k", base_url=f"ws://127.0.0.1:{port}", start_timeout=0.2)
        stream = stt.stream()
        with pytest.raises(ProviderTimeoutError):
            await collect(stream)
        await stream.aclose()
    finally:
        server.close()
        await server.wait_closed()


def test_option_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPEECHMATICS_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="API key"):
        SpeechmaticsSTT()
    for bad in (
        {"max_delay": 1.0},  # Realtime only
        {"sample_rate": 8000},  # Agent STT is 16 kHz only
        {"model": "enhanced", "emit_sentences": True},
        {"model": "enhanced", "max_delay": 5.0},
        {"model": "enhanced", "end_of_utterance_silence_trigger": 3.0},
        {"additional_vocab": "word"},
        {"additional_vocab": [{"sounds_like": ["x"]}]},
        {"region": "mars"},
        {"force_end_grace": -1},
    ):
        with pytest.raises(ConfigurationError):
            SpeechmaticsSTT(api_key="k", **bad)
    monkeypatch.setenv("SPEECHMATICS_API_KEY", "env-key")
    stt = SpeechmaticsSTT(region="eu")
    assert stt.url == "wss://eu.rt.speechmatics.com/v2/agent"
    rt = SpeechmaticsSTT(model="standard", region="us", end_of_utterance_silence_trigger=None)
    assert rt.url == "wss://us.rt.speechmatics.com/v2"
    assert "conversation_config" not in rt.transcription_config()
    assert SpeechmaticsSTT().url == "wss://global.rt.speechmatics.com/v2/agent"
    assert not SpeechmaticsSTT().can_force_end  # Agent STT vad mode


def test_registry_specs_create_speechmatics_stt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPEECHMATICS_API_KEY", "k")
    stt = create("stt", "speechmatics")
    assert isinstance(stt, SpeechmaticsSTT) and stt.model == "linden-1" and stt.is_agent
    stt = create("stt", "speechmatics/enhanced", language="de", end_of_turn=False)
    assert isinstance(stt, SpeechmaticsSTT) and not stt.is_agent
    assert stt.transcription_config()["language"] == "de"


async def test_create_temporary_key() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.headers["Authorization"] != "Bearer key":
            return httpx.Response(401, json={"detail": "Unauthorized"})
        return httpx.Response(201, json={"apikey_id": "abc", "key_value": "temp-jwt"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    stt = SpeechmaticsSTT(api_key="key", http_client=client)
    assert await stt.create_temporary_key(ttl=120, client_ref="user-1") == "temp-jwt"
    assert str(seen[0].url) == "https://mp.speechmatics.com/v1/api_keys?type=rt"
    assert json.loads(seen[0].content) == {"ttl": 120, "client_ref": "user-1"}
    with pytest.raises(AuthenticationError):
        await SpeechmaticsSTT(api_key="bad", http_client=client).create_temporary_key()
    await client.aclose()


# ----------------------------------------------------------------- cascade integration
async def test_cascade_commits_on_agent_stt_end_of_turn() -> None:
    """Agent STT's EndOfTurn commits the user turn at once, with no VAD in the cascade."""
    server_script = ScriptedServer(
        [
            (0.3, timed("StartOfTurn", 0.2)),
            (0.6, segment("Book a table", 0.2, 0.55, final=False)),
            (0.9, segment("Book a table for two.", 0.2, 0.88)),
            (1.1, timed("EndOfTurn", 1.05)),
        ]
    )
    async with FakeSpeechmatics(server_script) as server:
        session = AgentSession(
            stt=SpeechmaticsSTT(api_key="k", base_url=server.url),
            llm="mock",
            tts="mock",
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
        await asyncio.wait_for(_until(lambda: bool(metrics)), 15)
        await session.aclose()

    eot_sent = next(t for t, m in server_script.sent if m["message"] == "EndOfTurn")
    assert finals == ["Book a table for two."]
    assert committed[0] - eot_sent < 1.0
    history = [(i.role, i.text) for i in session.history.items if isinstance(i, ChatMessage)]
    assert history == [
        ("user", "Book a table for two."),
        ("assistant", "You said: Book a table for two."),
    ]


@pytest.mark.parametrize("model", ["linden-1", "enhanced"])
async def test_cascade_with_vad_owns_turns_via_force_end_of_utterance(model: str) -> None:
    if model == "linden-1":
        script = [
            (0.3, timed("StartOfTurn", 0.1)),
            (0.7, segment("what time is it", 0.1, 0.65, final=False)),
        ]
        on_force = agent_force([True])
        expected = "My account number is four two."
    else:
        script = [(0.7, transcript(word("what", 0.1, 0.3), word("time", 0.3, 0.5), final=False))]

        def on_force(timestamp: float) -> list[dict[str, Any]]:
            return [
                transcript(word("What", 0.1, 0.3), word("time", 0.3, 0.5), punct("?", 0.5)),
                end_of_utterance(timestamp, forced=True),
            ]

        expected = "What time?"
    async with FakeSpeechmatics(ScriptedServer(script, on_force=on_force)) as server:
        session = AgentSession(
            stt=SpeechmaticsSTT(model=model, api_key="k", base_url=server.url, end_of_turn=False),
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
    assert finals == [expected]
    assert "ForceEndOfUtterance" in server.connections[0].types()


# ------------------------------------------------------------------- real API (opt-in)
needs_key = pytest.mark.skipif(
    not os.environ.get("SPEECHMATICS_API_KEY"), reason="needs SPEECHMATICS_API_KEY"
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
@pytest.mark.parametrize("model", ["linden-1", "enhanced"])
async def test_integration_session_protocol_on_silence(model: str) -> None:
    stream = SpeechmaticsSTT(model=model).stream()
    for frame in chunks(AudioFrame.silence(1.0, 16_000), 0.05):
        stream.push_audio(frame)
        await asyncio.sleep(frame.duration)
    stream.end_input()
    events = await collect(stream)
    await stream.aclose()
    assert stream.session_id
    assert [ev.text for ev in events if ev.type == E.FINAL_TRANSCRIPT] == [""]


@pytest.mark.integration
@needs_key
@pytest.mark.parametrize("model", ["linden-1", "enhanced"])
@pytest.mark.parametrize("end_of_turn", [True, False])
async def test_integration_stt_round_trip(model: str, end_of_turn: bool) -> None:
    speech = await _speech()
    stream = SpeechmaticsSTT(model=model, end_of_turn=end_of_turn).stream()
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
async def test_integration_temporary_key() -> None:
    stt = SpeechmaticsSTT()
    try:
        key = await stt.create_temporary_key(ttl=60)
        stream = SpeechmaticsSTT(api_key="", jwt=key).stream()
        stream.push_audio(AudioFrame.silence(0.5, 16_000))
        stream.end_input()
        await collect(stream)
        await stream.aclose()
    finally:
        await stt.aclose()
