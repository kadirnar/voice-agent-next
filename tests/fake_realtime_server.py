"""A scripted, OpenAI-Realtime-compatible WebSocket server for offline tests.

:class:`FakeRealtimeServer` listens on ``127.0.0.1:0`` and replays realistic event
sequences in the GA dialect (or the beta / xAI dialects):

* ``session.created`` (with ``expires_at``); ``session.update`` -> ``session.updated``;
* server VAD (energy based) on the appended audio: ``input_audio_buffer.speech_started``
  (cancelling the active response when ``interrupt_response``), ``speech_stopped``,
  ``committed``, ``conversation.item.added`` and input transcription events (GA deltas,
  xAI cumulative ``.updated`` or Qwen ``text``/``stash``) followed by ``.completed``;
* responses: ``response.created``, ``output_item.added``, ``content_part.added``,
  transcript deltas interleaved with 24 kHz audio deltas, the ``*.done`` events and
  ``response.done`` with usage; function calls with streamed arguments;
* ``response.cancel``, ``conversation.item.truncate`` validated like the real API, and the
  real error codes (``response_cancel_not_active``,
  ``conversation_already_has_active_response``, ``unsupported_content_type``...).

Replies are scripted like :class:`~voice_agent_next.providers.mock.MockLLM` (strings or
:class:`~voice_agent_next.providers.mock.MockToolCall`); ``say()``-style verbatim
instructions are spoken as requested.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, Literal, TypeAlias
from urllib.parse import parse_qsl, urlsplit

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from voice_agent_next.audio import AudioFrame
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import MockToolCall, synth_speech
from voice_agent_next.utils.ids import new_id
from voice_agent_next.vad import VADEventType, VADOptions, VADStream

Dialect: TypeAlias = Literal["ga", "beta", "xai"]
Reply: TypeAlias = str | MockToolCall | list[MockToolCall]

GA_TO_BETA = {
    "response.output_audio.delta": "response.audio.delta",
    "response.output_audio.done": "response.audio.done",
    "response.output_audio_transcript.delta": "response.audio_transcript.delta",
    "response.output_audio_transcript.done": "response.audio_transcript.done",
    "response.output_text.delta": "response.text.delta",
    "conversation.item.added": "conversation.item.created",
}
VERBATIM = re.compile(r'Say exactly the following, verbatim, and nothing else: "(.*)"\s*$', re.S)
PREFIX_PADDING = 0.3


@dataclass
class Handshake:
    path: str
    query: dict[str, str]
    headers: dict[str, str]


@dataclass
class ResponseRecord:
    """One response the server generated (for assertions)."""

    response_id: str
    body: dict[str, Any]
    text: str | None
    calls: list[MockToolCall]
    status: str = "in_progress"
    reason: str | None = None


@dataclass
class _Item:
    item_id: str
    role: str
    kind: str = "message"
    audio_ms: float = 0.0


@dataclass
class _Active:
    record: ResponseRecord
    task: asyncio.Task[None] | None = None
    output: list[dict[str, Any]] = field(default_factory=list)
    audio_ms: float = 0.0
    finished: bool = False


def _merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


class FakeRealtimeServer:
    """Scripted Realtime server. Use as ``async with FakeRealtimeServer(...) as server``.

    Args:
        replies: responses to generate, in order (then ``default_reply``).
        transcripts: user transcripts, one per committed turn (then ``"hello"``).
        dialect: ``ga`` (OpenAI), ``beta`` (Qwen/Speaches event names) or ``xai``.
        speech_end: what ``speech_stopped.audio_end_ms`` reports: ``includes_silence``
            (the documented OpenAI semantics) or ``speech_end``.
        transcript_mode: ``delta`` / ``cumulative`` (xAI) / ``text_stash`` (Qwen);
            default depends on the dialect.
        transcription_delay: send transcription events this long after the commit
            (0 = before the response starts).
        realtime_factor: audio generation pace (0 = instant, 0.5 = twice real time).
        expires_in: seconds until ``expires_at`` in ``session.created``.
        api_key: reject handshakes without this key (401).
        reject_status: reject every handshake with this HTTP status.
        reject_session_update: error object sent instead of ``session.updated``.
    """

    def __init__(
        self,
        *,
        replies: Sequence[Reply] = (),
        transcripts: Sequence[str] = (),
        default_reply: str = "OK.",
        dialect: Dialect = "ga",
        speech_end: Literal["includes_silence", "speech_end"] = "includes_silence",
        transcript_mode: Literal["delta", "cumulative", "text_stash"] | None = None,
        transcription_delay: float = 0.0,
        chars_per_second: float = 15.0,
        chunk_ms: int = 100,
        realtime_factor: float = 0.0,
        supports_cancel: bool = True,
        supports_truncate: bool = True,
        expires_in: float = 3600.0,
        api_key: str | None = None,
        reject_status: int | None = None,
        reject_session_update: dict[str, Any] | None = None,
    ) -> None:
        self.replies: list[Reply] = list(replies)
        self.transcripts: list[str] = list(transcripts)
        self.default_reply = default_reply
        self.dialect: Dialect = dialect
        self.speech_end = speech_end
        self.transcript_mode = (
            transcript_mode
            or {
                "ga": "delta",
                "beta": "text_stash",
                "xai": "cumulative",
            }[dialect]
        )
        self.transcription_delay = transcription_delay
        self.chars_per_second = chars_per_second
        self.chunk_ms = chunk_ms
        self.realtime_factor = realtime_factor
        self.supports_cancel = supports_cancel
        self.supports_truncate = supports_truncate
        self.expires_in = expires_in
        self.api_key = api_key
        self.reject_status = reject_status
        self.reject_session_update = reject_session_update
        self.handshakes: list[Handshake] = []
        self.received: list[dict[str, Any]] = []
        """Client events (``input_audio_buffer.append`` payloads replaced by byte counts)."""
        self.sent: list[dict[str, Any]] = []
        self.responses: list[ResponseRecord] = []
        self.truncations: list[tuple[str, int, int]] = []
        self.connections: list[_Connection] = []
        self.port = 0
        self._server: Server | None = None

    # ------------------------------------------------------------------ lifecycle
    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/v1"

    async def start(self) -> FakeRealtimeServer:
        self._server = await serve(
            self._handle,
            "127.0.0.1",
            0,
            process_request=self._process_request,
            compression=None,
            max_size=None,
        )
        self.port = next(iter(self._server.sockets)).getsockname()[1]
        return self

    async def aclose(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def __aenter__(self) -> FakeRealtimeServer:
        return await self.start()

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------- helpers
    def events(self, etype: str) -> list[dict[str, Any]]:
        """Client events of ``etype`` received so far."""
        return [e for e in self.received if e.get("type") == etype]

    def sent_events(self, etype: str) -> list[dict[str, Any]]:
        return [e for e in self.sent if e.get("type") == etype]

    async def push(self, event: dict[str, Any]) -> None:
        """Send a raw server event to every open connection."""
        for conn in self.connections:
            await conn.send_raw(event)

    async def drop(self) -> None:
        """Abort every connection without a close frame (simulated network failure)."""
        for conn in self.connections:
            conn.ws.transport.abort()

    def next_reply(self) -> Reply:
        return self.replies.pop(0) if self.replies else self.default_reply

    def next_transcript(self) -> str:
        return self.transcripts.pop(0) if self.transcripts else "hello"

    # ----------------------------------------------------------------- internals
    def _process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        parts = urlsplit(request.path)
        headers = {k.lower(): v for k, v in request.headers.raw_items()}
        self.handshakes.append(Handshake(parts.path, dict(parse_qsl(parts.query)), headers))
        if self.reject_status is not None:
            return connection.respond(HTTPStatus(self.reject_status), "rejected\n")
        if self.api_key is not None:
            auth = headers.get("authorization", "")
            key = headers.get("api-key") or (auth[7:] if auth.lower().startswith("bearer ") else "")
            if key != self.api_key:
                return connection.respond(HTTPStatus.UNAUTHORIZED, "invalid api key\n")
        return None

    async def _handle(self, ws: ServerConnection) -> None:
        conn = _Connection(self, ws)
        self.connections.append(conn)
        await conn.run()


class _Connection:
    """Server-side state of one Realtime session."""

    def __init__(self, server: FakeRealtimeServer, ws: ServerConnection) -> None:
        self.server = server
        self.ws = ws
        self.dialect = server.dialect
        self.session: dict[str, Any] = self._default_session()
        self.position = 0.0  # seconds of input audio received
        self.buffered = 0.0  # seconds of uncommitted input audio
        self.vad: VADStream | None = None
        self.vad_origin = 0.0
        self.vad_silence: float | None = None
        self.speech_item: str | None = None
        self.items: dict[str, _Item] = {}
        self.order: list[str] = []
        self.active: _Active | None = None
        self.tasks: set[asyncio.Task[None]] = set()
        self._configure_vad()

    def _default_session(self) -> dict[str, Any]:
        td = {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 500,
            "create_response": True,
            "interrupt_response": True,
        }
        if self.dialect == "beta":
            return {
                "modalities": ["text", "audio"],
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": td,
            }
        audio = {
            "input": {"format": {"type": "audio/pcm", "rate": 24_000}},
            "output": {"format": {"type": "audio/pcm", "rate": 24_000}},
        }
        if self.dialect == "xai":
            return {"voice": "eve", "turn_detection": {"type": "server_vad"}, "audio": audio}
        audio["input"]["turn_detection"] = td
        return {"type": "realtime", "output_modalities": ["audio"], "audio": audio}

    # ------------------------------------------------------------------ config
    def turn_detection(self) -> dict[str, Any] | None:
        if self.dialect == "ga":
            td = self.session.get("audio", {}).get("input", {}).get("turn_detection")
        else:
            td = self.session.get("turn_detection")
        return td if isinstance(td, dict) else None

    def input_rate(self) -> int:
        if self.dialect == "beta":
            return 16_000 if self.session.get("input_audio_format") == "pcm" else 24_000
        return int(self.session["audio"]["input"]["format"].get("rate", 24_000))

    def output_rate(self) -> int:
        if self.dialect == "beta":
            return 24_000
        return int(self.session["audio"]["output"]["format"].get("rate", 24_000))

    def _configure_vad(self) -> None:
        td = self.turn_detection()
        if td is None:
            self.vad, self.vad_silence = None, None
            return
        silence = float(td.get("silence_duration_ms", 500)) / 1000.0
        if self.vad is not None and silence == self.vad_silence:
            return
        options = VADOptions(
            min_speech_duration=0.1,
            min_silence_duration=silence,
            prefix_padding_duration=PREFIX_PADDING,
        )
        self.vad = EnergyVAD(sample_rate=16_000, options=options).stream()
        self.vad_origin, self.vad_silence = self.position, silence

    # ----------------------------------------------------------------- sending
    async def send(self, etype: str, **fields: Any) -> None:
        name = GA_TO_BETA.get(etype, etype) if self.dialect == "beta" else etype
        await self.send_raw({"event_id": new_id("event_"), "type": name, **fields})

    async def send_raw(self, event: dict[str, Any]) -> None:
        payload = json.dumps(event)
        self.server.sent.append(json.loads(payload))
        with contextlib.suppress(ConnectionClosed):
            await self.ws.send(payload)

    async def error(self, err: dict[str, Any], source: dict[str, Any]) -> None:
        error = {"param": None, **err, "event_id": source.get("event_id")}
        await self.send("error", error=error)

    def spawn(self, coro: Any) -> None:
        task: asyncio.Task[None] = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    # -------------------------------------------------------------------- loop
    async def run(self) -> None:
        created = {
            **self.session,
            "id": new_id("sess_"),
            "object": "realtime.session",
            "expires_at": int(time.time() + self.server.expires_in),
        }
        await self.send("session.created", session=created)
        if self.dialect == "xai":
            conversation = {"id": new_id("conv_"), "object": "realtime.conversation"}
            await self.send("conversation.created", conversation=conversation)
        try:
            async for raw in self.ws:
                event = json.loads(raw)
                if event.get("type") == "input_audio_buffer.append":
                    self.server.received.append({**event, "audio": len(event.get("audio", ""))})
                else:
                    self.server.received.append(event)
                await self.handle(event)
        except ConnectionClosed:
            pass
        finally:
            if self.active is not None and self.active.task is not None:
                self.active.task.cancel()
            for task in list(self.tasks):
                task.cancel()

    async def handle(self, ev: dict[str, Any]) -> None:
        etype = ev.get("type")
        if etype == "session.update":
            if self.server.reject_session_update is not None:
                await self.error(self.server.reject_session_update, ev)
                return
            _merge(self.session, ev.get("session") or {})
            self._configure_vad()
            await self.send("session.updated", session=self.session)
        elif etype == "input_audio_buffer.append":
            await self.on_audio(ev.get("audio", ""))
        elif etype == "input_audio_buffer.commit":
            if self.buffered <= 0:
                await self.error(
                    {
                        "type": "invalid_request_error",
                        "code": "input_audio_buffer_commit_empty",
                        "message": "buffer too small",
                    },
                    ev,
                )
                return
            await self.commit(self.speech_item or new_id("item_"), respond=False)
        elif etype == "input_audio_buffer.clear":
            self.buffered, self.speech_item = 0.0, None
            if self.vad is not None:
                self.vad.reset()
            await self.send("input_audio_buffer.cleared")
        elif etype == "conversation.item.create":
            await self.on_item_create(ev)
        elif etype == "response.create":
            await self.on_response_create(ev)
        elif etype == "response.cancel":
            if not self.server.supports_cancel:
                await self.error(
                    {
                        "type": "invalid_request_error",
                        "code": "unsupported_event",
                        "message": "response.cancel is not supported",
                    },
                    ev,
                )
            elif not await self.cancel_active("client_cancelled"):
                await self.error(
                    {
                        "type": "invalid_request_error",
                        "code": "response_cancel_not_active",
                        "message": "Cancellation failed: no active response found",
                    },
                    ev,
                )
        elif etype == "conversation.item.truncate":
            await self.on_truncate(ev)
        else:
            await self.error(
                {
                    "type": "invalid_request_error",
                    "code": "invalid_event",
                    "message": f"unknown event type {etype!r}",
                },
                ev,
            )

    # ------------------------------------------------------------------- input
    async def on_audio(self, payload: str) -> None:
        frame = AudioFrame(base64.b64decode(payload), self.input_rate())
        self.position += frame.duration
        self.buffered += frame.duration
        if self.vad is None:
            return
        td = self.turn_detection() or {}
        for vev in self.vad.push_audio(frame):
            if vev.type == VADEventType.START_OF_SPEECH:
                start = self.vad_origin + vev.audio_time - vev.speech_duration - PREFIX_PADDING
                self.speech_item = new_id("item_")
                await self.send(
                    "input_audio_buffer.speech_started",
                    audio_start_ms=round(max(0.0, start) * 1000),
                    item_id=self.speech_item,
                )
                if td.get("interrupt_response", True):
                    await self.cancel_active("turn_detected")
            elif vev.type == VADEventType.END_OF_SPEECH:
                end = self.vad_origin + vev.audio_time
                if self.server.speech_end == "speech_end":
                    end -= vev.silence_duration
                item_id = self.speech_item or new_id("item_")
                await self.send(
                    "input_audio_buffer.speech_stopped",
                    audio_end_ms=round(end * 1000),
                    item_id=item_id,
                )
                await self.commit(item_id, respond=td.get("create_response", True))

    async def commit(self, item_id: str, *, respond: bool) -> None:
        previous = self.order[-1] if self.order else None
        self.buffered, self.speech_item = 0.0, None
        await self.send("input_audio_buffer.committed", previous_item_id=previous, item_id=item_id)
        self.add_item(_Item(item_id, "user"))
        item = {
            "id": item_id,
            "object": "realtime.item",
            "type": "message",
            "status": "completed",
            "role": "user",
            "content": [{"type": "input_audio", "transcript": None}],
        }
        await self.send("conversation.item.added", previous_item_id=previous, item=item)
        transcript = self.server.next_transcript()
        if self.server.transcription_delay > 0:
            self.spawn(self.transcribe(item_id, transcript, self.server.transcription_delay))
        else:
            await self.transcribe(item_id, transcript, 0.0)
        if respond:
            await self.start_response(self.server.next_reply(), {})

    async def transcribe(self, item_id: str, transcript: str, delay: float) -> None:
        if delay:
            await asyncio.sleep(delay)
        accumulated = ""
        for word in re.findall(r"\S+\s*", transcript):
            accumulated += word
            if self.server.transcript_mode == "delta":
                await self.send(
                    "conversation.item.input_audio_transcription.delta",
                    item_id=item_id,
                    content_index=0,
                    delta=word,
                )
            elif self.server.transcript_mode == "cumulative":
                await self.send(
                    "conversation.item.input_audio_transcription.updated",
                    item_id=item_id,
                    transcript=accumulated,
                )
            else:  # Qwen: confirmed text + tentative stash
                await self.send(
                    "conversation.item.input_audio_transcription.delta",
                    item_id=item_id,
                    content_index=0,
                    text=accumulated[: -len(word)],
                    stash=word,
                )
        await self.send(
            "conversation.item.input_audio_transcription.completed",
            item_id=item_id,
            content_index=0,
            transcript=transcript,
            usage={"type": "duration", "seconds": 1.0},
        )

    def add_item(self, item: _Item) -> None:
        self.items[item.item_id] = item
        self.order.append(item.item_id)

    async def on_item_create(self, ev: dict[str, Any]) -> None:
        item = dict(ev.get("item") or {})
        if item.get("type") == "force_message":
            if self.dialect != "xai":
                await self.error(
                    {
                        "type": "invalid_request_error",
                        "code": "invalid_value",
                        "message": "unknown item type force_message",
                    },
                    ev,
                )
                return
            text = "".join(part.get("text", "") for part in item.get("content", []))
            await self.start_response(text, {"force_message": True})
            return
        item_id = item.get("id") or new_id("item_")
        previous = self.order[-1] if self.order else None
        self.add_item(_Item(item_id, item.get("role", "user"), item.get("type", "message")))
        added = {**item, "id": item_id, "object": "realtime.item", "status": "completed"}
        await self.send("conversation.item.added", previous_item_id=previous, item=added)

    # --------------------------------------------------------------- responses
    async def on_response_create(self, ev: dict[str, Any]) -> None:
        if self.active is not None:
            await self.error(
                {
                    "type": "invalid_request_error",
                    "code": "conversation_already_has_active_response",
                    "message": "Conversation already has an active response",
                },
                ev,
            )
            return
        body = ev.get("response") or {}
        instructions = body.get("instructions")
        source = instructions if instructions is not None else self.session.get("instructions")
        match = VERBATIM.search(source or "")
        reply: Reply = match.group(1) if match else self.server.next_reply()
        await self.start_response(reply, body)

    async def start_response(self, reply: Reply, body: dict[str, Any]) -> None:
        if isinstance(reply, MockToolCall):
            calls, text = [reply], None
        elif isinstance(reply, list):
            calls, text = list(reply), None
        else:
            calls, text = [], reply
        record = ResponseRecord(new_id("resp_"), body, text, calls)
        self.server.responses.append(record)
        active = self.active = _Active(record)
        await self.send("response.created", response=self.response_object(active, "in_progress"))
        active.task = asyncio.create_task(self.generate(active))

    async def generate(self, active: _Active) -> None:
        if active.record.calls:
            await self.generate_calls(active)
        else:
            await self.generate_speech(active, active.record.text or "")
        await self.finish(active, "completed")

    async def generate_speech(self, active: _Active, text: str) -> None:
        rid = active.record.response_id
        item_id = new_id("item_")
        previous = self.order[-1] if self.order else None
        item = _Item(item_id, "assistant")
        self.add_item(item)
        message: dict[str, Any] = {
            "id": item_id,
            "object": "realtime.item",
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        active.output.append(message)
        where = {"response_id": rid, "item_id": item_id, "output_index": 0, "content_index": 0}
        await self.send("response.output_item.added", response_id=rid, output_index=0, item=message)
        await self.send("conversation.item.added", previous_item_id=previous, item=message)
        await self.send(
            "response.content_part.added", **where, part={"type": "audio", "transcript": ""}
        )
        duration = max(0.2, len(text.strip()) / self.server.chars_per_second) if text.strip() else 0
        audio = synth_speech(duration, self.output_rate())
        chunk = self.server.chunk_ms / 1000.0
        chunks = math.ceil(duration / chunk) if duration else 0
        words = re.findall(r"\S+\s*", text)
        spoken = 0
        for i in range(chunks):
            upto = math.ceil(len(words) * (i + 1) / chunks)
            if upto > spoken:  # the transcript runs slightly ahead of the audio
                delta = "".join(words[spoken:upto])
                await self.send("response.output_audio_transcript.delta", **where, delta=delta)
                spoken = upto
            piece = audio.slice(i * chunk, min(duration, (i + 1) * chunk))
            await self.send("response.output_audio.delta", **where, delta=piece.to_base64())
            item.audio_ms += piece.duration_ms
            active.audio_ms += piece.duration_ms
            await asyncio.sleep(piece.duration * self.server.realtime_factor)
        await self.send("response.output_audio.done", **where)
        await self.send("response.output_audio_transcript.done", **where, transcript=text)
        await self.send(
            "response.content_part.done", **where, part={"type": "audio", "transcript": text}
        )
        kind = "audio" if self.dialect == "beta" else "output_audio"
        message.update(status="completed", content=[{"type": kind, "transcript": text}])
        await self.send("response.output_item.done", response_id=rid, output_index=0, item=message)
        if self.dialect == "ga":
            await self.send("conversation.item.done", previous_item_id=previous, item=message)

    async def generate_calls(self, active: _Active) -> None:
        rid = active.record.response_id
        for index, call in enumerate(active.record.calls):
            item_id, call_id = new_id("item_"), new_id("call_")
            arguments = call.arguments_json()
            self.add_item(_Item(item_id, "assistant", "function_call"))
            item: dict[str, Any] = {
                "id": item_id,
                "object": "realtime.item",
                "type": "function_call",
                "status": "in_progress",
                "name": call.name,
                "call_id": call_id,
                "arguments": "",
            }
            where = {
                "response_id": rid,
                "item_id": item_id,
                "output_index": index,
                "call_id": call_id,
            }
            await self.send(
                "response.output_item.added", response_id=rid, output_index=index, item=item
            )
            half = len(arguments) // 2
            for part in (arguments[:half], arguments[half:]):
                await self.send("response.function_call_arguments.delta", **where, delta=part)
            await self.send(
                "response.function_call_arguments.done",
                **where,
                name=call.name,
                arguments=arguments,
            )
            item = {**item, "status": "completed", "arguments": arguments}
            active.output.append(item)
            await self.send(
                "response.output_item.done", response_id=rid, output_index=index, item=item
            )

    def response_object(
        self, active: _Active, status: str, details: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        obj: dict[str, Any] = {
            "object": "realtime.response",
            "id": active.record.response_id,
            "status": status,
            "status_details": details,
            "output": active.output,
            "conversation_id": "conv_fake",
            "usage": None,
        }
        obj["modalities" if self.dialect == "beta" else "output_modalities"] = ["audio"]
        if status != "in_progress":
            text_tokens = len((active.record.text or "").split()) + 5 * len(active.record.calls)
            audio_tokens = round(active.audio_ms / 50)  # 1 token per 50 ms of output audio
            suffix = "s_details" if self.dialect == "beta" else "_details"
            obj["usage"] = {
                "total_tokens": 150 + text_tokens + audio_tokens,
                "input_tokens": 150,
                "output_tokens": text_tokens + audio_tokens,
                f"input_token{suffix}": {
                    "text_tokens": 120,
                    "audio_tokens": 30,
                    "cached_tokens": 64,
                },
                f"output_token{suffix}": {"text_tokens": text_tokens, "audio_tokens": audio_tokens},
            }
        return obj

    async def finish(self, active: _Active, status: str, reason: str | None = None) -> None:
        if active.finished:
            return
        active.finished = True
        if self.active is active:
            self.active = None
        active.record.status, active.record.reason = status, reason
        details = None if status == "completed" else {"type": status, "reason": reason}
        await self.send("response.done", response=self.response_object(active, status, details))

    async def cancel_active(self, reason: str) -> bool:
        active = self.active
        if active is None or active.finished:
            return False
        if active.task is not None and not active.task.done():
            active.task.cancel()
            await asyncio.gather(active.task, return_exceptions=True)
        await self.finish(active, "cancelled", reason)
        return True

    async def on_truncate(self, ev: dict[str, Any]) -> None:
        if not self.server.supports_truncate:
            await self.error(
                {
                    "type": "invalid_request_error",
                    "code": "unsupported_event",
                    "message": "conversation.item.truncate is not supported",
                },
                ev,
            )
            return
        item_id, end = ev.get("item_id", ""), ev.get("audio_end_ms")
        content_index = ev.get("content_index")
        item = self.items.get(item_id)
        if item is None:
            await self.error(
                {
                    "type": "invalid_request_error",
                    "code": "item_not_found",
                    "message": f"Item {item_id} not found",
                },
                ev,
            )
        elif item.role != "assistant" or item.kind != "message" or item.audio_ms <= 0:
            await self.error(
                {
                    "type": "invalid_request_error",
                    "code": "unsupported_content_type",
                    "message": "Only model output audio messages can be truncated",
                },
                ev,
            )
        elif content_index != 0 or not isinstance(end, int) or not 0 <= end <= item.audio_ms:
            await self.error(
                {
                    "type": "invalid_request_error",
                    "code": "invalid_value",
                    "message": "audio_end_ms is greater than the audio duration",
                    "param": "audio_end_ms",
                },
                ev,
            )
        else:
            self.server.truncations.append((item_id, content_index, end))
            await self.send(
                "conversation.item.truncated",
                item_id=item_id,
                content_index=content_index,
                audio_end_ms=end,
            )
