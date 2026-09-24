"""Deepgram providers (Nova-3 / Flux STT, Aura-2 TTS) against local fake servers.

The fakes replay Deepgram's documented protocol messages (Listen v1 ``Results`` /
``SpeechStarted`` / ``UtteranceEnd`` / ``Metadata``, Flux v2 ``Connected`` / ``TurnInfo`` /
``Warning`` / ``Error``, Speak v1 audio / ``Flushed`` / ``Cleared``). No network, no API
key — except the ``integration`` tests at the end, which need ``DEEPGRAM_API_KEY``.
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
import numpy as np
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
from voice_agent_next.metrics import STTMetrics, TTSMetrics, TurnMetrics
from voice_agent_next.providers.deepgram import DeepgramSTT, DeepgramTTS
from voice_agent_next.providers.mock import synth_speech
from voice_agent_next.registry import create
from voice_agent_next.stt import STTEvent, STTEventType
from voice_agent_next.transports import LoopbackTransport
from voice_agent_next.utils import now

REQUEST_ID = "5d1a6f0c-8b1e-4c62-9a2d-3f1e0c9b7a11"
BYTES_PER_SECOND = 16_000 * 2  # linear16 mono @ 16 kHz, what the STT sends
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

    async def send(self, message: dict[str, Any]) -> None:
        await self.ws.send(json.dumps(message))


Handler = Callable[[FakeConn], Awaitable[None]]


class FakeDeepgram:
    """A WebSocket server on 127.0.0.1 speaking Deepgram's protocol, scripted per test."""

    def __init__(self, handler: Handler, *, reject: tuple[int, str] | None = None) -> None:
        self._handler = handler
        self._reject = reject
        self._server: Server | None = None
        self.connections: list[FakeConn] = []

    async def __aenter__(self) -> FakeDeepgram:
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
        return f"http://127.0.0.1:{port}"

    def _process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        if self._reject is None:
            return None
        status, message = self._reject
        body = json.dumps({"err_code": "REJECTED", "err_msg": message, "request_id": REQUEST_ID})
        response = connection.respond(status, body)
        response.headers["dg-error"] = message
        response.headers["dg-request-id"] = REQUEST_ID
        return response

    async def _handle(self, ws: ServerConnection) -> None:
        split = urlsplit(ws.request.path if ws.request else "/")
        headers = ws.request.headers if ws.request else Headers()
        conn = FakeConn(ws, split.path, parse_qs(split.query), headers)
        self.connections.append(conn)
        try:
            await self._handler(conn)
        except ConnectionClosed:
            pass


def words_for(text: str, start: float, end: float) -> list[dict[str, Any]]:
    tokens = text.split()
    step = (end - start) / max(1, len(tokens))
    return [
        {
            "word": token.lower().strip(".,?!"),
            "start": round(start + i * step, 3),
            "end": round(start + (i + 0.8) * step, 3),
            "confidence": 0.97,
            "punctuated_word": token,
        }
        for i, token in enumerate(tokens)
    ]


# ------------------------------------------------------------------- Nova-3 (Listen v1)
def results(
    text: str,
    *,
    start: float,
    duration: float,
    is_final: bool,
    speech_final: bool = False,
    from_finalize: bool = False,
) -> dict[str, Any]:
    """A Listen v1 ``Results`` message as Deepgram sends it."""
    return {
        "type": "Results",
        "channel_index": [0, 1],
        "duration": duration,
        "start": start,
        "is_final": is_final,
        "speech_final": speech_final,
        "from_finalize": from_finalize,
        "channel": {
            "alternatives": [
                {
                    "transcript": text,
                    "confidence": 0.98 if text else 0.0,
                    "words": words_for(text, start, start + duration),
                }
            ]
        },
        "metadata": {
            "request_id": REQUEST_ID,
            "model_info": {
                "name": "general-nova-3",
                "version": "2025-04-17.21547",
                "arch": "nova-3",
            },
            "model_uuid": "40bd3654-e622-47c4-a111-63a61b23bd5f",
        },
    }


def speech_started(timestamp: float) -> dict[str, Any]:
    return {"type": "SpeechStarted", "channel": [0, 1], "timestamp": timestamp}


def utterance_end(last_word_end: float) -> dict[str, Any]:
    return {"type": "UtteranceEnd", "channel": [0, 1], "last_word_end": last_word_end}


def nova_server(
    script: Sequence[tuple[float, dict[str, Any]]],
    finalize: Sequence[dict[str, Any]] = (),
) -> Handler:
    """Sends ``script`` messages once that much audio (seconds) arrived; answers
    ``Finalize`` with ``finalize`` and ``CloseStream`` with ``Metadata`` + close."""

    async def handler(conn: FakeConn) -> None:
        pending, audio = list(script), 0
        async for message in conn.ws:
            conn.received.append(message)
            if isinstance(message, bytes):
                audio += len(message)
                while pending and audio >= pending[0][0] * BYTES_PER_SECOND:
                    await conn.send(pending.pop(0)[1])
                continue
            kind = json.loads(message)["type"]
            if kind == "Finalize":
                for reply in finalize:
                    await conn.send(reply)
            elif kind == "CloseStream":
                for _, msg in pending:  # results for audio still in flight
                    await conn.send(msg)
                await conn.send(
                    {
                        "type": "Metadata",
                        "transaction_key": "deprecated",
                        "request_id": REQUEST_ID,
                        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                        "created": "2026-09-24T12:00:00.000Z",
                        "duration": audio / BYTES_PER_SECOND,
                        "channels": 1,
                    }
                )
                await conn.ws.close()
                return

    return handler


def chunks(frame: AudioFrame, step: float = 0.02) -> list[AudioFrame]:
    n = round(step * frame.sample_rate) * 2
    return [
        AudioFrame(frame.data[i : i + n], frame.sample_rate) for i in range(0, len(frame.data), n)
    ]


async def collect(stream: AsyncIterator[STTEvent]) -> list[STTEvent]:
    return [ev async for ev in stream]


def kinds(events: Sequence[STTEvent]) -> list[tuple[STTEventType, str]]:
    return [(ev.type, ev.text) for ev in events]


async def test_nova3_interim_final_speech_final_and_finalize() -> None:
    script = [
        (0.25, speech_started(0.22)),
        (0.5, results("hello", start=0.0, duration=0.5, is_final=False)),
        (1.0, results("Hello world.", start=0.0, duration=1.0, is_final=True)),
        (1.3, results("how are", start=1.0, duration=0.3, is_final=False)),
    ]
    finalize = [results("How are you?", start=1.0, duration=0.6, is_final=True, from_finalize=True)]
    async with FakeDeepgram(nova_server(script, finalize)) as server:
        stt = DeepgramSTT(api_key="test-key", base_url=server.url, keyterms=["Deepgram", "Flux"])
        metrics: list[STTMetrics] = []
        stt.on("metrics", metrics.append)
        stream = stt.stream(language="en")
        for frame in chunks(synth_speech(1.6, 16_000)):
            stream.push_audio(frame)
        stream.end_input()  # flush (Finalize), then CloseStream
        events = await collect(stream)
        await stream.aclose()

    assert kinds(events) == [
        (E.START_OF_SPEECH, ""),
        (E.INTERIM_TRANSCRIPT, "hello"),
        (E.FINAL_TRANSCRIPT, "Hello world."),
        (E.INTERIM_TRANSCRIPT, "how are"),
        (E.FINAL_TRANSCRIPT, "How are you?"),
    ]
    final = events[2].transcript
    assert final is not None and final.language == "en" and final.confidence == pytest.approx(0.98)
    assert final.start_time == 0.0 and final.end_time == pytest.approx(1.0)
    assert final.words is not None and [w.word for w in final.words] == ["Hello", "world."]
    assert len({ev.segment_id for ev in events}) == 1  # one utterance
    conn = server.connections[0]
    assert conn.path == "/v1/listen"
    assert conn.headers["Authorization"] == "Token test-key"
    assert {k: v[0] for k, v in conn.query.items() if k != "keyterm"} == {
        "model": "nova-3",
        "encoding": "linear16",
        "sample_rate": "16000",
        "channels": "1",
        "language": "en",
        "interim_results": "true",
        "smart_format": "true",
        "endpointing": "300",
        "utterance_end_ms": "1000",
        "vad_events": "true",
    }
    assert conn.query["keyterm"] == ["Deepgram", "Flux"]
    assert conn.types() == ["Finalize", "CloseStream"]
    sizes = conn.audio_chunks()
    assert set(sizes) == {1600}  # 50 ms chunks
    assert sum(sizes) == round(1.6 * 16_000) * 2
    assert metrics and metrics[0].streamed and metrics[0].latency is not None
    assert metrics[0].audio_duration == pytest.approx(1.6, abs=0.01)


async def test_nova3_speech_final_and_utterance_end_end_speech() -> None:
    script = [
        (0.2, results("yes", start=0.0, duration=0.2, is_final=False)),  # no SpeechStarted
        (0.6, results("Yes.", start=0.0, duration=0.6, is_final=True, speech_final=True)),
        (0.9, speech_started(0.85)),
        (1.4, results("Thanks.", start=0.6, duration=0.8, is_final=True)),
        (1.6, utterance_end(1.28)),
        (1.7, results("", start=1.4, duration=0.3, is_final=True)),  # silence: not emitted
    ]
    async with FakeDeepgram(nova_server(script)) as server:
        stt = DeepgramSTT(api_key="k", base_url=server.url)
        stream = stt.stream()
        for frame in chunks(synth_speech(1.8, 16_000)):
            stream.push_audio(frame)
        events = []
        async for ev in stream:
            events.append(ev)
            if ev.type == E.END_OF_SPEECH and len(events) > 4:
                stream.end_input()  # Deepgram already finalized: the flush is acked at once

    assert [k for k, _ in kinds(events)] == [
        E.START_OF_SPEECH,
        E.INTERIM_TRANSCRIPT,
        E.FINAL_TRANSCRIPT,
        E.END_OF_SPEECH,
        E.START_OF_SPEECH,
        E.FINAL_TRANSCRIPT,
        E.END_OF_SPEECH,
        E.FINAL_TRANSCRIPT,
    ]
    assert events[-1].text == "" and server.connections[0].types() == ["Finalize", "CloseStream"]
    first_end, second_end = events[3], events[6]
    assert first_end.transcript is not None and first_end.transcript.end_time == pytest.approx(0.48)
    assert second_end.transcript is not None and second_end.transcript.end_time == 1.28
    assert events[0].segment_id == events[3].segment_id != events[4].segment_id


async def test_nova3_flush_after_speech_final_is_acknowledged_immediately() -> None:
    """Deepgram may not answer a Finalize with nothing buffered; the stream acks it."""
    script = [
        (0.1, speech_started(0.05)),
        (0.5, results("Stop.", start=0.0, duration=0.5, is_final=True, speech_final=True)),
    ]
    async with FakeDeepgram(nova_server(script)) as server:  # never answers Finalize
        stream = DeepgramSTT(api_key="k", base_url=server.url).stream()
        for frame in chunks(synth_speech(0.6, 16_000)):
            stream.push_audio(frame)
        got: list[STTEvent] = []
        async for ev in stream:
            got.append(ev)
            if ev.type == E.END_OF_SPEECH:
                stream.flush()
            elif ev.type == E.FINAL_TRANSCRIPT and not ev.text:
                break
        await stream.aclose()
    assert [k for k, _ in kinds(got)] == [
        E.START_OF_SPEECH,
        E.FINAL_TRANSCRIPT,
        E.END_OF_SPEECH,
        E.FINAL_TRANSCRIPT,
    ]
    assert got[-1].text == ""


async def test_nova3_batch_transcribe_joins_finals() -> None:
    script = [
        (0.6, results("Hello world.", start=0.0, duration=0.6, is_final=True)),
        (0.8, results("how", start=0.6, duration=0.2, is_final=False)),
    ]
    finalize = [results("How are you?", start=0.6, duration=0.6, is_final=True, from_finalize=True)]
    async with FakeDeepgram(nova_server(script, finalize)) as server:
        stt = DeepgramSTT(api_key="k", base_url=server.url)
        transcript = await stt.transcribe(synth_speech(1.2, 48_000))  # resampled to 16 kHz
    assert transcript.text == "Hello world. How are you?"
    assert sum(server.connections[0].audio_chunks()) == round(1.2 * 16_000) * 2


async def test_nova3_sends_keepalive_while_idle() -> None:
    async with FakeDeepgram(nova_server([])) as server:
        stream = DeepgramSTT(api_key="k", base_url=server.url, keepalive_interval=0.05).stream()
        stream.push_audio(AudioFrame.silence(0.1, 16_000))
        await asyncio.sleep(0.4)  # no audio for a while
        stream.end_input()
        assert await collect(stream) == []
    types = server.connections[0].types()
    assert types.count("KeepAlive") >= 3
    assert types[-2:] == ["Finalize", "CloseStream"]


async def test_nova3_options_and_disabled_endpointing() -> None:
    stt = DeepgramSTT(
        api_key="k",
        model="nova-3-medical",
        language="multi",
        sample_rate=8000,
        interim_results=False,
        smart_format=False,
        punctuate=True,
        endpointing_ms=False,
        vad_events=False,
        numerals=True,
        tags=["demo"],
        mip_opt_out=True,
        extra_params={"diarize": True},
    )
    query = parse_qs(urlsplit(stt.url("multi")).query)
    assert query["endpointing"] == ["false"] and query["interim_results"] == ["false"]
    assert "utterance_end_ms" not in query  # requires interim results
    assert query["sample_rate"] == ["8000"] and query["punctuate"] == ["true"]
    assert query["tag"] == ["demo"] and query["diarize"] == ["true"]
    assert query["mip_opt_out"] == ["true"] and query["numerals"] == ["true"]
    assert stt.capabilities.language_detection and not stt.capabilities.end_of_turn
    assert not stt.capabilities.interim_results


# ---------------------------------------------------------------------- Flux (Listen v2)
def turn_info(
    event: str,
    transcript: str,
    *,
    window_end: float,
    turn_index: int = 0,
    eot: float = 0.1,
    trigger: str | None = None,
    languages: list[str] | None = None,
) -> dict[str, Any]:
    """A Flux ``TurnInfo`` message (``sequence_id`` is set when sending)."""
    msg: dict[str, Any] = {
        "type": "TurnInfo",
        "request_id": REQUEST_ID,
        "event": event,
        "turn_index": turn_index,
        "audio_window_start": 0.0,
        "audio_window_end": window_end,
        "transcript": transcript,
        "words": words_for(transcript, 0.2, window_end) if transcript else [],
        "end_of_turn_confidence": eot,
    }
    for w in msg["words"]:  # Flux words: punctuated and cased, no punctuated_word field
        w["word"] = w.pop("punctuated_word")
    if trigger is not None:
        msg["trigger"] = trigger
    if languages is not None:
        msg["languages"] = languages
    return msg


@dataclass
class FluxServer:
    """Scripted Flux server; tracks turn state to answer ``ForceEndTurn`` like Flux does."""

    script: list[tuple[float, dict[str, Any]]]
    sent: list[tuple[float, dict[str, Any]]] = field(default_factory=list)
    honor_force_end_turn: bool = True

    async def __call__(self, conn: FakeConn) -> None:
        seq, audio, active, turn, last = 0, 0, False, 0, ""
        pending = list(self.script)

        async def send(msg: dict[str, Any]) -> None:
            nonlocal seq, active, turn, last
            msg = {**msg, "sequence_id": seq}
            seq += 1
            if msg["type"] == "TurnInfo":
                last = msg["transcript"] or last
                if msg["event"] == "StartOfTurn":
                    active = True
                elif msg["event"] == "EndOfTurn":
                    active, turn, last = False, turn + 1, ""
            self.sent.append((now(), msg))
            await conn.send(msg)

        await send({"type": "Connected", "request_id": REQUEST_ID})
        async for message in conn.ws:
            conn.received.append(message)
            if isinstance(message, bytes):
                audio += len(message)
                while pending and audio >= pending[0][0] * BYTES_PER_SECOND:
                    await send(pending.pop(0)[1])
                continue
            kind = json.loads(message)["type"]
            if kind == "ForceEndTurn" and self.honor_force_end_turn:
                if active:
                    end = audio / BYTES_PER_SECOND
                    await send(
                        turn_info("EndOfTurn", last, window_end=end, turn_index=turn,
                                  eot=0.35, trigger="manual")
                    )  # fmt: skip
                else:
                    await send(
                        {
                            "type": "Warning",
                            "code": "FORCE_END_TURN_NO_ACTIVE_TURN",
                            "description": "ForceEndTurn received with no active turn.",
                        }
                    )
            elif kind == "CloseStream":
                await conn.ws.close()
                return


FLUX_TURN = [
    (0.2, turn_info("Update", "", window_end=0.2)),
    (0.4, turn_info("StartOfTurn", "Hi I", window_end=0.4)),
    (0.6, turn_info("Update", "Hi I need to", window_end=0.6)),
    (0.7, turn_info("Update", "Hi I need to", window_end=0.7)),  # unchanged: no event
    (0.9, turn_info("Update", "Hi I need to cancel my subscription.", window_end=0.9, eot=0.3)),
    (1.0, turn_info("EagerEndOfTurn", "Hi I need to cancel my subscription.", window_end=1.0, eot=0.5)),
    (1.1, turn_info("TurnResumed", "Hi I need to cancel my subscription please", window_end=1.1)),
    (1.3, turn_info("EagerEndOfTurn", "Hi I need to cancel my subscription please.", window_end=1.3, eot=0.55)),
    (1.5, turn_info("EndOfTurn", "Hi I need to cancel my subscription please.", window_end=1.5, eot=0.82, trigger="model")),
    (1.6, turn_info("Update", "", window_end=1.6, turn_index=1)),
]  # fmt: skip


async def test_flux_turn_events() -> None:
    async with FakeDeepgram(FluxServer(list(FLUX_TURN))) as server:
        stt = DeepgramSTT(
            model="flux-general-en",
            api_key="test-key",
            base_url=server.url,
            eot_threshold=0.8,
            eager_eot_threshold=0.5,
            eot_timeout_ms=3000,
            keyterms="Deepgram",
            language="en",  # ignored: language hints are for flux-general-multi only
        )
        assert stt.capabilities.end_of_turn and stt.capabilities.streaming
        metrics: list[STTMetrics] = []
        stt.on("metrics", metrics.append)
        stream = stt.stream()
        for frame in chunks(synth_speech(1.7, 16_000)):
            stream.push_audio(frame)
        stream.end_input()
        events = await collect(stream)
        await stream.aclose()

    full = "Hi I need to cancel my subscription please."
    assert kinds(events) == [
        (E.START_OF_SPEECH, ""),
        (E.INTERIM_TRANSCRIPT, "Hi I"),
        (E.INTERIM_TRANSCRIPT, "Hi I need to"),
        (E.INTERIM_TRANSCRIPT, "Hi I need to cancel my subscription."),
        (E.EAGER_END_OF_TURN, "Hi I need to cancel my subscription."),
        (E.TURN_RESUMED, "Hi I need to cancel my subscription please"),
        (E.INTERIM_TRANSCRIPT, "Hi I need to cancel my subscription please"),
        (E.INTERIM_TRANSCRIPT, full),
        (E.EAGER_END_OF_TURN, full),
        (E.FINAL_TRANSCRIPT, full),
        (E.END_OF_SPEECH, full),
        (E.END_OF_TURN, full),
        (E.FINAL_TRANSCRIPT, ""),  # end_input's ForceEndTurn found no active turn: ack
    ]
    final = events[9].transcript
    assert final is not None and final.words is not None and final.words[0].word == "Hi"
    assert final.end_time == pytest.approx(final.words[-1].end)
    assert final.confidence == pytest.approx(0.97)
    assert len({ev.segment_id for ev in events}) == 1
    conn = server.connections[0]
    assert conn.path == "/v2/listen" and conn.headers["Authorization"] == "Token test-key"
    assert {k: v[0] for k, v in conn.query.items()} == {
        "model": "flux-general-en",
        "encoding": "linear16",
        "sample_rate": "16000",
        "eot_threshold": "0.8",
        "eager_eot_threshold": "0.5",
        "eot_timeout_ms": "3000",
        "keyterm": "Deepgram",
    }
    assert set(conn.audio_chunks()[:-1]) == {2560}  # 80 ms chunks, as Flux recommends
    assert conn.types() == ["ForceEndTurn", "CloseStream"]  # end_input()
    assert sum(m.audio_duration for m in metrics) == pytest.approx(1.7, abs=0.01)


async def test_flux_flush_forces_end_of_turn_without_end_of_turn_event() -> None:
    script = [
        (0.3, turn_info("StartOfTurn", "My account number is", window_end=0.3)),
        (0.6, turn_info("Update", "My account number is 4 2", window_end=0.6)),
    ]
    async with FakeDeepgram(FluxServer(script)) as server:
        stt = DeepgramSTT(model="flux-general-multi", api_key="k", base_url=server.url,
                          language="es")  # fmt: skip
        stream = stt.stream()
        for frame in chunks(synth_speech(0.8, 16_000)):
            stream.push_audio(frame)
        stream.flush()  # e.g. push-to-talk release -> ForceEndTurn
        events: list[STTEvent] = []
        async for ev in stream:
            events.append(ev)
            if ev.type == E.END_OF_SPEECH:
                stream.flush()  # outside a turn: Flux warns, the flush is still acknowledged
            elif ev.type == E.FINAL_TRANSCRIPT and not ev.text:
                break
        await stream.aclose()

    assert kinds(events) == [
        (E.START_OF_SPEECH, ""),
        (E.INTERIM_TRANSCRIPT, "My account number is"),
        (E.INTERIM_TRANSCRIPT, "My account number is 4 2"),
        (E.FINAL_TRANSCRIPT, "My account number is 4 2"),
        (E.END_OF_SPEECH, "My account number is 4 2"),
        (E.FINAL_TRANSCRIPT, ""),
    ]  # no END_OF_TURN: the turn was ended by our own flush (trigger="manual")
    conn = server.connections[0]
    assert conn.types()[:2] == ["ForceEndTurn", "ForceEndTurn"]
    assert conn.query["language_hint"] == ["es"] and conn.query["model"] == ["flux-general-multi"]
    assert stt.capabilities.language_detection


async def test_flux_close_stream_turns_last_update_into_final() -> None:
    """``CloseStream`` does not finalize the active turn: the last Update is the final."""
    script = [(0.3, turn_info("StartOfTurn", "Wait", window_end=0.3))]
    async with FakeDeepgram(FluxServer(script, honor_force_end_turn=False)) as server:
        stream = DeepgramSTT(model="flux-general-en", api_key="k", base_url=server.url).stream()
        for frame in chunks(synth_speech(0.5, 16_000)):
            stream.push_audio(frame)
        stream.end_input()
        events = await collect(stream)
    assert kinds(events)[-2:] == [(E.FINAL_TRANSCRIPT, "Wait"), (E.END_OF_SPEECH, "Wait")]


async def test_flux_error_message_raises() -> None:
    async def failing(conn: FakeConn) -> None:
        await conn.send({"type": "Connected", "request_id": REQUEST_ID, "sequence_id": 0})
        await conn.ws.recv()
        await conn.send(
            {"type": "Error", "sequence_id": 1, "code": "INTERNAL_SERVER_ERROR",
             "description": "Something went wrong."}
        )  # fmt: skip
        await conn.ws.wait_closed()

    async with FakeDeepgram(failing) as server:
        stream = DeepgramSTT(model="flux-general-en", api_key="k", base_url=server.url).stream()
        stream.push_audio(AudioFrame.silence(0.2, 16_000))
        with pytest.raises(ProviderError, match="INTERNAL_SERVER_ERROR") as info:
            await collect(stream)
        await stream.aclose()
    assert info.value.retryable


def test_flux_option_validation() -> None:
    with pytest.raises(ConfigurationError, match="eot_threshold"):
        DeepgramSTT(model="flux-general-en", api_key="k", eot_threshold=0.3)
    with pytest.raises(ConfigurationError, match="eager_eot_threshold"):
        DeepgramSTT(model="flux-general-en", api_key="k", eager_eot_threshold=0.95)
    with pytest.raises(ConfigurationError, match="must not exceed"):
        DeepgramSTT(
            model="flux-general-en", api_key="k", eot_threshold=0.6, eager_eot_threshold=0.7
        )
    with pytest.raises(ConfigurationError, match="eot_timeout_ms"):
        DeepgramSTT(model="flux-general-en", api_key="k", eot_timeout_ms=100)
    with pytest.raises(ConfigurationError, match="sample rates"):
        DeepgramSTT(model="flux-general-en", api_key="k", sample_rate=22_050)


# ------------------------------------------------------------------------ STT failures
def test_missing_api_key_is_a_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="DEEPGRAM_API_KEY"):
        DeepgramSTT()
    with pytest.raises(ConfigurationError, match="DEEPGRAM_API_KEY"):
        DeepgramTTS()
    monkeypatch.setenv("DEEPGRAM_API_KEY", "from-env")
    assert DeepgramSTT()._api_key == "from-env"


@pytest.mark.parametrize(
    ("status", "error"),
    [(401, AuthenticationError), (403, AuthenticationError), (429, RateLimitError),
     (400, ProviderError), (503, ProviderError)],
)  # fmt: skip
async def test_stt_handshake_errors_are_mapped(status: int, error: type[ProviderError]) -> None:
    async with FakeDeepgram(nova_server([]), reject=(status, "Invalid credentials.")) as server:
        stream = DeepgramSTT(api_key="bad", base_url=server.url).stream()
        stream.push_audio(AudioFrame.silence(0.1, 16_000))
        with pytest.raises(error, match="Invalid credentials") as info:
            await collect(stream)
        await stream.aclose()
    assert info.value.status_code == status and info.value.provider == "deepgram"
    assert info.value.retryable == (status in (429, 503))


@pytest.mark.parametrize(
    ("code", "reason", "error", "retryable"),
    [(1008, "DATA-0000", ProviderError, False), (1011, "NET-0001", ProviderConnectionError, True)],
)
async def test_stt_close_codes_are_mapped(
    code: int, reason: str, error: type[ProviderError], retryable: bool
) -> None:
    async def closing(conn: FakeConn) -> None:
        await conn.ws.recv()
        await conn.ws.close(code, reason)

    async with FakeDeepgram(closing) as server:
        stream = DeepgramSTT(api_key="k", base_url=server.url).stream()
        stream.push_audio(AudioFrame.silence(0.2, 16_000))
        with pytest.raises(error, match=reason) as info:
            await collect(stream)
        await stream.aclose()
    assert type(info.value) is error
    assert info.value.status_code == code and info.value.retryable is retryable


async def test_stt_connection_refused_is_a_connection_error() -> None:
    async with FakeDeepgram(nova_server([])) as server:
        url = server.url
    stream = DeepgramSTT(api_key="k", base_url=url, connect_timeout=5).stream()  # server gone
    # refused at once on Linux/macOS; Windows retries the SYN and may hit the timeout first
    with pytest.raises((ProviderConnectionError, ProviderTimeoutError)) as info:
        await collect(stream)
    assert info.value.retryable
    await stream.aclose()


# ------------------------------------------------------------- cascade with Flux turns
async def test_cascade_commits_on_flux_end_of_turn_without_endpointing_delay() -> None:
    """Flux ``EndOfTurn`` commits the user turn at once, with no VAD in the cascade: the
    endpointing delay (3 s here) would otherwise hold the commit back."""
    flux = FluxServer(
        [
            (0.3, turn_info("StartOfTurn", "Book a", window_end=0.3)),
            (0.6, turn_info("Update", "Book a table", window_end=0.6)),
            (0.9, turn_info("Update", "Book a table for two.", window_end=0.9, eot=0.6)),
            (1.1, turn_info("EndOfTurn", "Book a table for two.", window_end=1.1, eot=0.85,
                            trigger="model")),
        ]
    )  # fmt: skip
    async with FakeDeepgram(flux) as server:
        stt = DeepgramSTT(model="flux-general-en", api_key="k", base_url=server.url)
        session = AgentSession(
            stt=stt,
            llm="mock",
            tts="mock",  # no vad=...: Flux does the turn-taking
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

    eot_sent = next(t for t, m in flux.sent if m.get("event") == "EndOfTurn")
    assert finals == ["Book a table for two."]
    assert committed[0] - eot_sent < 1.0  # committed right away, not after 3 s
    history = [(i.role, i.text) for i in session.history.items if isinstance(i, ChatMessage)]
    assert history == [
        ("user", "Book a table for two."),
        ("assistant", "You said: Book a table for two."),
    ]
    assert metrics[0].end_of_turn_delay is not None and metrics[0].end_of_turn_delay < 0.5
    assert metrics[0].voice_to_voice is not None


async def _until(predicate: Callable[[], bool]) -> None:
    while not predicate():
        await asyncio.sleep(0.01)


# ------------------------------------------------------------------ Aura-2 (Speak v1)
def aura_level(text: str) -> int:
    """The constant sample value the fake Aura server synthesizes for ``text``."""
    return 1000 + sum(map(ord, text)) % 20_000


def aura_audio(text: str) -> bytes:
    return np.full(len(text) * 240, aura_level(text), dtype="<i2").tobytes()  # 10 ms/char @24k


@dataclass
class AuraServer:
    """Speak v1 fake: buffers ``Speak`` text, synthesizes on ``Flush`` (then ``Flushed``),
    stops and answers ``Cleared`` on ``Clear``, closes on ``Close``."""

    chunk_bytes: int = 4800  # 100 ms @ 24 kHz
    chunk_delay: float = 0.0
    warn_on_flush: bool = False

    async def __call__(self, conn: FakeConn) -> None:
        await conn.send(
            {"type": "Metadata", "request_id": REQUEST_ID, "model_name": "aura-2-thalia-en",
             "model_version": "2025-04-07.0", "model_uuid": "ecb76e9d-f2db-4127-8060-79b05590d22f"}
        )  # fmt: skip
        jobs: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
        buffer: list[str] = []
        flushes = 0

        async def generate() -> None:
            while True:
                text, seq = await jobs.get()
                audio = aura_audio(text)
                for i in range(0, len(audio), self.chunk_bytes):
                    await conn.ws.send(audio[i : i + self.chunk_bytes])
                    await asyncio.sleep(self.chunk_delay)
                await conn.send({"type": "Flushed", "sequence_id": seq})

        generator = asyncio.create_task(generate())
        try:
            async for message in conn.ws:
                conn.received.append(message)
                data = json.loads(message)
                if data["type"] == "Speak":
                    buffer.append(data["text"])
                elif data["type"] == "Flush":
                    if self.warn_on_flush:
                        await conn.send(
                            {"type": "Warning", "code": "FLUSH_LIMIT_EXCEEDED",
                             "description": "Cannot process more Flush messages for 60s."}
                        )  # fmt: skip
                    else:
                        jobs.put_nowait(("".join(buffer), flushes))
                    buffer.clear()
                    flushes += 1
                elif data["type"] == "Clear":
                    generator.cancel()
                    await asyncio.gather(generator, return_exceptions=True)
                    while not jobs.empty():
                        jobs.get_nowait()
                    buffer.clear()
                    await conn.send({"type": "Cleared", "sequence_id": flushes})
                    generator = asyncio.create_task(generate())
                elif data["type"] == "Close":
                    await conn.ws.close()
        finally:
            generator.cancel()
            await asyncio.gather(generator, return_exceptions=True)


async def test_aura_streams_text_and_ends_a_segment_per_flush() -> None:
    async with FakeDeepgram(AuraServer()) as server:
        tts = DeepgramTTS(api_key="test-key", base_url=server.url)
        assert tts.capabilities.streaming and tts.sample_rate == 24_000
        metrics: list[TTSMetrics] = []
        tts.on("metrics", metrics.append)
        stream = tts.stream()
        stream.push_text("Hello there. ")
        stream.push_text("How are you? ")
        stream.flush()
        stream.push_text("Goodbye.")
        stream.end_input()
        out = [chunk async for chunk in stream]
        await stream.aclose()
        await tts.aclose()

    assert [c.text for c in out if c.text] == ["Hello there. How are you?", "Goodbye."]
    assert [c.is_final for c in out].count(True) == 2
    first = b"".join(c.frame.data for c in out[: next(i for i, c in enumerate(out) if c.is_final)])
    assert first == aura_audio("Hello there. How are you? ")
    total = sum(c.frame.duration for c in out)
    assert total == pytest.approx(len("Hello there. How are you? Goodbye.") * 0.01)
    assert all(c.frame.sample_rate == 24_000 for c in out)
    conn = server.connections[0]
    assert conn.path == "/v1/speak" and conn.headers["Authorization"] == "Token test-key"
    assert {k: v[0] for k, v in conn.query.items()} == {
        "model": "aura-2-thalia-en",
        "encoding": "linear16",
        "sample_rate": "24000",
    }
    assert conn.messages() == [
        {"type": "Speak", "text": "Hello there. "},
        {"type": "Speak", "text": "How are you? "},
        {"type": "Flush"},
        {"type": "Speak", "text": "Goodbye."},
        {"type": "Flush"},
    ]
    assert len(metrics) == 1 and metrics[0].streamed and metrics[0].ttfb is not None
    assert metrics[0].characters == len("Hello there. How are you? Goodbye.")
    assert metrics[0].audio_duration == pytest.approx(total)


async def test_aura_reuses_one_websocket_per_conversation() -> None:
    async with FakeDeepgram(AuraServer()) as server:
        tts = DeepgramTTS(api_key="k", base_url=server.url, voice="apollo")
        await tts.warmup()
        for text in ("First answer.", "Second answer."):
            stream = tts.stream()
            stream.push_text(text)
            stream.flush()  # an empty trailing segment (end_input) needs no Flush
            stream.end_input()
            out = [c async for c in stream]
            await stream.aclose()
            assert b"".join(c.frame.data for c in out) == aura_audio(text)
            assert [c.is_final for c in out].count(True) == 2
        await tts.aclose()
    assert len(server.connections) == 1
    assert server.connections[0].query["model"] == ["aura-2-apollo-en"]
    assert server.connections[0].types() == ["Speak", "Flush", "Speak", "Flush"]


async def test_aura_interruption_clears_and_reuses_the_connection() -> None:
    async with FakeDeepgram(AuraServer(chunk_bytes=960, chunk_delay=0.01)) as server:
        tts = DeepgramTTS(api_key="k", base_url=server.url)
        long_text = "This is a long answer that the user is about to interrupt. " * 3
        stream = tts.stream()
        stream.push_text(long_text)
        stream.flush()
        first = await stream.__anext__()
        assert first.frame and first.text == long_text.strip()
        await stream.aclose()  # barge-in: Clear, then the connection goes back to the pool
        await asyncio.wait_for(_until(lambda: bool(tts._pool and any(tts._pool.values()))), 2)
        conn = server.connections[0]
        assert conn.types() == ["Speak", "Flush", "Clear"]

        stream = tts.stream()
        stream.push_text("Sure, go ahead.")
        stream.end_input()
        out = [c async for c in stream]
        await stream.aclose()
        await tts.aclose()
    audio = np.frombuffer(b"".join(c.frame.data for c in out), dtype="<i2")
    assert len(audio) == len("Sure, go ahead.") * 240
    assert set(audio.tolist()) == {aura_level("Sure, go ahead.")}  # no stale audio
    assert len(server.connections) == 1


async def test_aura_connection_dropped_by_server_is_replaced() -> None:
    async with FakeDeepgram(AuraServer()) as server:
        tts = DeepgramTTS(api_key="k", base_url=server.url)
        await tts.warmup()
        await server.connections[0].ws.close()  # e.g. an idle timeout on Deepgram's side
        await asyncio.sleep(0.05)
        audio = b""
        stream = tts.stream()
        stream.push_text("Still here.")
        stream.end_input()
        async for chunk in stream:
            audio += chunk.frame.data
        await stream.aclose()
        await tts.aclose()
    assert audio == aura_audio("Still here.")
    assert len(server.connections) == 2


async def test_aura_flush_limit_warning_fails_instead_of_hanging() -> None:
    async with FakeDeepgram(AuraServer(warn_on_flush=True)) as server:
        tts = DeepgramTTS(api_key="k", base_url=server.url)
        stream = tts.stream()
        stream.push_text("One too many.")
        stream.end_input()
        with pytest.raises(RateLimitError, match="flush limit"):
            [c async for c in stream]
        await stream.aclose()
        await tts.aclose()


async def test_aura_rest_synthesize_streams_raw_pcm() -> None:
    requests: list[httpx.Request] = []
    pcm = np.full(4800, 1234, dtype="<i2").tobytes()  # 0.2 s @ 24 kHz

    async def body() -> AsyncIterator[bytes]:
        for part in (pcm[:1001], pcm[1001:5000], pcm[5000:]):  # odd chunk boundaries
            yield part

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=body(), headers={"content-type": "audio/l16;rate=24000"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tts = DeepgramTTS(api_key="test-key", http_client=client, speed=1.1, streaming=False)
    audio = await tts.synthesize("**Hello** world!").collect()  # markdown stripped
    assert audio.data == pcm and audio.sample_rate == 24_000
    assert await tts.synthesize("   ").collect() == AudioFrame.empty(24_000)  # no request
    await tts.aclose()
    await client.aclose()

    assert len(requests) == 1
    req = requests[0]
    assert req.method == "POST" and req.url.path == "/v1/speak"
    assert req.url.host == "api.deepgram.com" and req.url.scheme == "https"
    assert dict(req.url.params) == {
        "model": "aura-2-thalia-en",
        "encoding": "linear16",
        "sample_rate": "24000",
        "speed": "1.1",
        "container": "none",
    }
    assert req.headers["Authorization"] == "Token test-key"
    assert json.loads(req.content) == {"text": "Hello world!"}


@pytest.mark.parametrize(
    ("status", "body", "error", "match"),
    [
        (401, {"err_code": "INVALID_AUTH", "err_msg": "Invalid credentials."}, AuthenticationError, "Invalid credentials"),
        (429, {"err_code": "Too Many Requests", "err_msg": "Please try again later."}, RateLimitError, "try again"),
        (400, {"err_code": "Bad Request", "err_msg": "No such model/language/tier combination found."}, ProviderError, "No such model"),
        (422, {"category": "UNPROCESSABLE_ENTITY", "message": "Failed to handle request."}, ProviderError, "Failed to handle"),
    ],
)  # fmt: skip
async def test_aura_rest_errors_are_mapped(
    status: int, body: dict[str, str], error: type[ProviderError], match: str
) -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(status, json=body))
    )
    tts = DeepgramTTS(api_key="k", http_client=client)
    with pytest.raises(error, match=match) as info:
        await tts.synthesize("Hello.").collect()
    assert info.value.status_code == status
    await tts.aclose()
    await client.aclose()


async def test_aura_rest_network_failure_is_a_connection_error() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(fail))
    tts = DeepgramTTS(api_key="k", http_client=client)
    with pytest.raises(ProviderConnectionError):
        await tts.synthesize("Hello.").collect()
    await tts.aclose()
    await client.aclose()


async def test_aura_websocket_auth_error() -> None:
    async with FakeDeepgram(AuraServer(), reject=(401, "Invalid credentials.")) as server:
        tts = DeepgramTTS(api_key="bad", base_url=server.url)
        stream = tts.stream()
        stream.push_text("Hi.")
        stream.end_input()
        with pytest.raises(AuthenticationError, match="Invalid credentials"):
            [c async for c in stream]
        await stream.aclose()
        await tts.aclose()


def test_aura_voice_and_option_resolution() -> None:
    tts = DeepgramTTS(api_key="k", model="aura-2-agustina-es")
    assert tts.model_for(None) == "aura-2-agustina-es"
    assert tts.model_for("celeste") == "aura-2-celeste-es"
    assert tts.model_for("aura-2-zeus-en") == "aura-2-zeus-en"
    assert DeepgramTTS(api_key="k", model="custom").model_for("luna") == "aura-2-luna-en"
    with pytest.raises(ConfigurationError, match="sample rates"):
        DeepgramTTS(api_key="k", sample_rate=22_050)
    with pytest.raises(ConfigurationError, match="speed"):
        DeepgramTTS(api_key="k", speed=2.0)


def test_registry_specs_create_deepgram_components() -> None:
    stt = create("stt", "deepgram/flux-general-en", api_key="k")
    assert isinstance(stt, DeepgramSTT) and stt.is_flux and stt.capabilities.end_of_turn
    nova = create("stt", {"provider": "deepgram", "language": "en"}, api_key="k")
    assert isinstance(nova, DeepgramSTT) and nova.model == "nova-3" and not nova.is_flux
    tts = create("tts", "deepgram/aura-2-andromeda-en", api_key="k")
    assert isinstance(tts, DeepgramTTS) and tts.model == "aura-2-andromeda-en"
    assert create("tts", "deepgram", api_key="k").model == "aura-2-thalia-en"


# ------------------------------------------------------------------- real API (opt-in)
needs_key = pytest.mark.skipif(
    not os.environ.get("DEEPGRAM_API_KEY"), reason="needs DEEPGRAM_API_KEY"
)


@pytest.mark.integration
@needs_key
async def test_integration_aura_rest_and_websocket() -> None:
    tts = DeepgramTTS()
    audio = await tts.synthesize("Hello from voice agent next.").collect()
    assert audio.duration > 0.5
    stream = tts.stream()
    stream.push_text("Hello there. ")
    stream.push_text("How are you today?")
    stream.end_input()
    out = [c async for c in stream]
    await stream.aclose()
    await tts.aclose()
    assert sum(c.frame.duration for c in out) > 0.8
    assert [c.is_final for c in out].count(True) == 1


@pytest.mark.integration
@needs_key
@pytest.mark.parametrize("model", ["nova-3", "flux-general-en"])
async def test_integration_stt_round_trip(model: str) -> None:
    tts = DeepgramTTS(sample_rate=16_000)
    speech = await tts.synthesize("The quick brown fox jumps over the lazy dog.").collect()
    await tts.aclose()
    stream = DeepgramSTT(model=model).stream()
    for frame in chunks(AudioFrame.concat([speech, AudioFrame.silence(2.0, 16_000)]), 0.08):
        stream.push_audio(frame)
        await asyncio.sleep(frame.duration)  # real time
    stream.end_input()
    events = await collect(stream)
    await stream.aclose()
    text = " ".join(ev.text for ev in events if ev.type == E.FINAL_TRANSCRIPT).lower()
    assert "fox" in text and "lazy dog" in text
    assert events[0].type == E.START_OF_SPEECH
    if model.startswith("flux"):
        assert E.END_OF_TURN in [ev.type for ev in events]
