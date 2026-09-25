"""One OpenAI Realtime connection: the protocol state machine between a WebSocket client
and an :class:`~voice_agent_next.engine.EngineConnection`.

Everything that changes state runs in one task (:meth:`RealtimeSession._main_loop`), which
takes engine events before client events, so a client request is always decided on
everything the engine has already reported. A reader task queues client messages (and
stops reading when too much is queued: TCP backpressure), a writer task sends the server
events queued by the main loop, so a slow client never delays the processing of its own
audio (barge-in keeps working while a burst of agent audio is being sent).

Engines answer a committed user turn by themselves, while the Realtime protocol lets the
client decide (``create_response: false``, manual ``input_audio_buffer.commit``). Such a
response is *held*: generated but not shown. A plain ``response.create`` adopts it (no
second generation, no added latency); anything that changes the conversation first
discards it.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import logging
import math
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal

from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from ..audio.frame import AudioFrame
from ..audio.resample import StreamResampler
from ..chat import ChatContext, FunctionCall, FunctionCallOutput
from ..engine import EngineConnection, EngineOptions, S2SEngine
from ..events import (
    EngineErrorEvent,
    EngineStatus,
    EngineUsage,
    InputCommitted,
    InputSpeechStarted,
    InputSpeechStopped,
    InputTranscript,
    ResponseAudio,
    ResponseDone,
    ResponseStarted,
    ResponseStatus,
    ResponseText,
    ResponseToolCall,
    ToolCallCancelled,
)
from ..utils.aio import Chan, ChanClosed, cancel_and_wait
from ..utils.clock import now
from ..utils.ids import new_id
from ..utils.log import logger
from ._protocol import (
    ClientError,
    Dialect,
    Item,
    SessionConfig,
    WireFormat,
    apply_session_update,
    event_name,
    parse_item,
    parse_response_modalities,
)
from .security import INBOX_HIGH, INBOX_LOW, OUTBOX_HIGH, OUTBOX_LOW, report_error

if TYPE_CHECKING:
    from .realtime import RealtimeModel, RealtimeServer

__all__ = ["RealtimeSession"]

_MIN_COMMIT: Final = 0.1
"""Seconds of buffered audio ``input_audio_buffer.commit`` needs (like OpenAI)."""
_AUDIO_CHUNK: Final = 0.1
"""Maximum seconds of audio per ``response.output_audio.delta``."""
_MAX_HELD_AUDIO: Final = 300.0
"""A held response that grows beyond this much audio is discarded."""
_REQUEST_TTL: Final = 30.0
"""A requested response that has not started after this long is forgotten."""
_INBOX_HIGH, _INBOX_LOW = INBOX_HIGH, INBOX_LOW
"""Queued client bytes at which the reader pauses / resumes (TCP backpressure)."""
_OUTBOX_HIGH, _OUTBOX_LOW = OUTBOX_HIGH, OUTBOX_LOW
"""Bytes queued for the client at which engine events stop / resume being taken."""
_FLUSH_TIMEOUT: Final = 2.0
# ``EngineConnection.say`` and ``OpenAIRealtimeEngine.say`` ask for verbatim speech with
# this sentence; engines with direct TTS access (the cascade) then speak it exactly.
_VERBATIM: Final = re.compile(
    r'Say exactly the following, verbatim, and nothing else: "(.*)"\s*$', re.S
)

_CLOSE_NORMAL, _CLOSE_POLICY, _CLOSE_INTERNAL = 1000, 1008, 1011


class _Stop(Exception):
    """Ends the session (after the error, if any, was queued for the client)."""

    def __init__(self, reason: str, code: int = _CLOSE_NORMAL) -> None:
        super().__init__(reason)
        self.reason = reason
        self.code = code


@dataclass(slots=True)
class _Sentinel:
    name: str


_DISCONNECTED: Final = _Sentinel("disconnected")
_EXPIRED: Final = _Sentinel("expired")
_IDLE: Final = _Sentinel("idle")
_ENGINE_CLOSED: Final = _Sentinel("engine closed")


@dataclass(slots=True)
class _Request:
    """A client ``response.create``."""

    event_id: str | None
    plain: bool
    """No per-response overrides: an already generated (held) response satisfies it."""
    modalities: list[str] | None = None
    metadata: Any = None
    instructions: str | None = None
    """Extra instructions for this response (session instructions stripped)."""
    verbatim: str | None = None
    """Text to speak verbatim (``say()``)."""
    created: float = field(default_factory=now)


@dataclass(slots=True)
class _Marker:
    """Placed into the engine's event stream right after a control call, so its effect is
    known exactly where the engine's own events for that call end."""

    kind: Literal["commit_start", "commit_end", "request"]
    item_id: str | None = None
    request: _Request | None = None


@dataclass
class _Output:
    """An output item of a response."""

    item: Item
    index: int
    engine_id: str | None = None
    format: WireFormat | None = None
    resampler: StreamResampler | None = None
    parts: list[str] = field(default_factory=list)


@dataclass
class _Response:
    engine_id: str
    id: str
    state: Literal["held", "forwarded", "discarded"]
    version: int
    modalities: list[str]
    request: _Request | None = None
    metadata: Any = None
    buffered: list[Any] = field(default_factory=list)
    buffered_audio: float = 0.0
    engine_items: list[str] = field(default_factory=list)
    engine_done: ResponseDone | None = None
    finished: bool = False
    cancel_reason: str | None = None
    purged_ms: float = 0.0
    outputs: list[_Output] = field(default_factory=list)
    message: _Output | None = None

    @property
    def text_only(self) -> bool:
        return "audio" not in self.modalities


@dataclass(slots=True)
class _Out:
    payload: str
    response: _Response | None = None
    """Set for audio deltas, which are dropped when their response is cancelled."""
    item: Item | None = None
    ms: float = 0.0


class RealtimeSession:
    """Server side of one Realtime WebSocket connection (created by the server)."""

    def __init__(
        self,
        server: RealtimeServer,
        websocket: ServerConnection,
        model: RealtimeModel,
        *,
        dialect: Dialect = "ga",
    ) -> None:
        self.server = server
        self.websocket = websocket
        self.model = model
        self.dialect: Dialect = dialect
        self.id = new_id("sess_")
        self.config: SessionConfig = model.session_config()
        self.started_at = now()
        limit = server.max_session_duration
        self.expires_at = int(time.time() + limit) if limit else 0
        self.stats: dict[str, float] = {
            "audio_in": 0.0,
            "audio_out": 0.0,
            "responses": 0,
            "errors": 0,
        }
        self._conversation_id = new_id("conv_")
        # client side
        self._client: Chan[str | bytes | _Sentinel] = Chan()
        self._client_bytes = 0
        self._last_client = now()
        """When the last client message arrived (idle timeout)."""
        self._client_space = asyncio.Event()
        self._client_space.set()
        self._disconnected = False
        self._outbox: deque[_Out] = deque()
        self._outbox_bytes = 0
        self._outbox_ready = asyncio.Event()
        self._drained = asyncio.Event()
        self._writer_stop = False
        self._handlers: dict[str, Callable[[dict[str, Any]], Awaitable[None]]] = {
            "session.update": self._on_session_update,
            "input_audio_buffer.append": self._on_append,
            "input_audio_buffer.commit": self._on_commit,
            "input_audio_buffer.clear": self._on_clear,
            "conversation.item.create": self._on_item_create,
            "conversation.item.delete": self._on_item_delete,
            "conversation.item.retrieve": self._on_item_retrieve,
            "conversation.item.truncate": self._on_truncate,
            "response.create": self._on_response_create,
            "response.cancel": self._on_response_cancel,
            "output_audio_buffer.clear": self._on_unsupported,
            "transcription_session.update": self._on_unsupported,
        }
        # engine side
        self._engine: S2SEngine | None = None
        self._owns_engine = False
        self._conn: EngineConnection | None = None
        self._events: Chan[Any] | None = None
        self._pump: asyncio.Task[None] | None = None
        self._settings: tuple[Any, ...] | None = None
        self._dirty = False
        """The engine's context no longer matches the conversation: rebuild it."""
        self._audio_offset = 0.0
        # input audio
        self._rest = b""
        self._buffered = 0.0
        self._pending_user: str | None = None
        self._announced: set[str] = set()
        self._partials: dict[str, str] = {}
        self._commit_binding: str | None = None
        # conversation
        self._items: dict[str, Item] = {}
        self._order: list[str] = []
        self._engine_items: dict[str, str] = {}
        self._version = 0
        self._assistant_audio = False
        # responses
        self._responses: dict[str, _Response] = {}
        self._active: _Response | None = None
        self._held: _Response | None = None
        self._request: _Request | None = None
        self._requesting: _Request | None = None
        self._auto: Literal["forward", "hold", "discard"] | _Request | None = None
        self._cancel_requested = False

    # ------------------------------------------------------------------ lifecycle
    async def run(self) -> None:
        """Serve the connection until either side closes it."""
        reader = asyncio.create_task(self._read_loop(), name=f"realtime-read-{self.id}")
        writer = asyncio.create_task(self._write_loop(), name=f"realtime-write-{self.id}")
        expiry: asyncio.Task[None] | None = None
        idle: asyncio.Task[None] | None = None
        if self.server.max_session_duration:
            expiry = asyncio.create_task(self._expire(self.server.max_session_duration))
        idle_timeout = getattr(self.server, "idle_timeout", None)
        if idle_timeout:
            idle = asyncio.create_task(self._watch_idle(idle_timeout))
        reason, code = "client disconnected", _CLOSE_NORMAL
        remote = self.websocket.remote_address
        logger.info(
            "realtime session %s started: model=%s dialect=%s peer=%s",
            self.id, self.model.name, self.dialect, _peer(remote),
            extra={"session_id": self.id, "model": self.model.name, "dialect": self.dialect},
        )  # fmt: skip
        try:
            self._send("session.created", session=self._render_session())
            await self._main_loop()
        except _Stop as stop:
            reason, code = stop.reason, stop.code
        except Exception as exc:
            error_id, message = report_error(exc, "The session failed", session_id=self.id)
            self._send_error({"type": "server_error", "code": "internal_error", "message": message})
            reason, code = f"internal error ({error_id})", _CLOSE_INTERNAL
        finally:
            await cancel_and_wait(reader, expiry, idle)
            await self._close_engine()
            if self._owns_engine and self._engine is not None:
                with contextlib.suppress(Exception):
                    await self._engine.aclose()
            await self._finish_writer(writer, code, reason)
            logger.info(
                "realtime session %s closed: reason=%s duration=%.1fs responses=%d "
                "audio_in=%.1fs audio_out=%.1fs errors=%d",
                self.id, reason, now() - self.started_at, self.stats["responses"],
                self.stats["audio_in"], self.stats["audio_out"], self.stats["errors"],
                extra={"session_id": self.id, "model": self.model.name, "reason": reason},
            )  # fmt: skip

    async def _main_loop(self) -> None:
        while True:
            for source, item in await self._next():
                if self._disconnected:
                    raise _Stop("client disconnected")
                if source == "engine":
                    await self._on_engine_event(item)
                elif isinstance(item, _Sentinel):
                    if item is _EXPIRED:
                        minutes = (self.server.max_session_duration or 0) / 60
                        self._send_error(
                            {
                                "type": "invalid_request_error",
                                "code": "session_expired",
                                "message": "Your session hit the maximum duration of "
                                f"{minutes:g} minutes.",
                            }
                        )
                        raise _Stop("session expired")
                    if item is _IDLE:
                        idle = getattr(self.server, "idle_timeout", None) or 0
                        self._send_error(
                            {
                                "type": "invalid_request_error",
                                "code": "session_idle",
                                "message": "The session was closed after "
                                f"{idle:g} seconds without client events.",
                            }
                        )
                        raise _Stop("session idle")
                    raise _Stop("client disconnected")
                else:
                    await self._on_client_message(item)

    async def _next(self) -> list[tuple[str, Any]]:
        """The next engine event(s) or client message(s).

        Engine events go first, except while the client is behind on reading what was
        already sent (outbox above the high-water mark): then only client events are taken
        until the writer catches up, so input audio, cancels and barge-in keep flowing
        while the engine's output waits in its own queue.
        """
        while True:
            events = self._events
            take_engine = events is not None and self._outbox_bytes <= _OUTBOX_HIGH
            if take_engine:
                assert events is not None
                try:
                    return [("engine", events.recv_nowait())]
                except asyncio.QueueEmpty:
                    pass
                except ChanClosed:
                    self._events = None
                    return [("engine", _ENGINE_CLOSED)]
            try:
                return [("client", self._client.recv_nowait())]
            except asyncio.QueueEmpty:
                pass
            except ChanClosed:
                return [("client", _DISCONNECTED)]
            waiters: dict[asyncio.Future[Any], str] = {
                asyncio.ensure_future(self._client.recv()): "client"
            }
            if events is not None:
                if take_engine:
                    waiters[asyncio.ensure_future(events.recv())] = "engine"
                else:
                    self._drained.clear()
                    waiters[asyncio.ensure_future(self._drained.wait())] = "drained"
            try:
                await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for fut in waiters:
                    fut.cancel()  # no-op for finished ones; a woken getter passes its item on
                await asyncio.gather(*waiters, return_exceptions=True)
            got: list[tuple[str, Any]] = []
            for fut, source in sorted(waiters.items(), key=lambda kv: kv[1] != "engine"):
                if fut.cancelled() or source == "drained":
                    continue
                exc = fut.exception()
                if isinstance(exc, ChanClosed):
                    if source == "engine":
                        if self._events is events:
                            self._events = None
                        got.append(("engine", _ENGINE_CLOSED))
                    else:
                        got.append(("client", _DISCONNECTED))
                elif exc is not None:
                    raise exc
                elif source != "engine" or self._events is events:
                    got.append((source, fut.result()))
            if got:
                return got

    async def _read_loop(self) -> None:
        try:
            async for message in self.websocket:
                self._last_client = now()
                self._client_bytes += len(message)
                self._client.send_nowait(message)
                if self._client_bytes > _INBOX_HIGH:
                    self._client_space.clear()
                    await self._client_space.wait()
        except ConnectionClosed:
            pass
        except Exception:
            logger.exception("realtime session %s: reader failed", self.id)
        finally:
            self._disconnected = True
            if not self._client.closed:
                self._client.send_nowait(_DISCONNECTED)
                self._client.close()

    async def _write_loop(self) -> None:
        try:
            while True:
                while not self._outbox:
                    if self._writer_stop:
                        return
                    self._outbox_ready.clear()
                    await self._outbox_ready.wait()
                out = self._outbox.popleft()
                self._outbox_bytes -= len(out.payload)
                await self.websocket.send(out.payload)
                if self._outbox_bytes <= _OUTBOX_LOW:
                    self._drained.set()
        except ConnectionClosed:
            pass

    async def _finish_writer(self, writer: asyncio.Task[None], code: int, reason: str) -> None:
        """Send what is queued (errors, final events), then close the connection."""
        self._writer_stop = True
        self._outbox_ready.set()
        if not self._disconnected:
            await asyncio.wait({writer}, timeout=_FLUSH_TIMEOUT)
        await cancel_and_wait(writer)
        with contextlib.suppress(Exception):
            await self.websocket.close(code, reason.encode()[:120].decode(errors="ignore"))

    async def _expire(self, seconds: float) -> None:
        await asyncio.sleep(seconds)
        if not self._client.closed:
            self._client.send_nowait(_EXPIRED)

    async def _watch_idle(self, seconds: float) -> None:
        while True:
            if self._client_bytes > 0:  # messages are waiting: the client is not idle
                self._last_client = now()
            remaining = self._last_client + seconds - now()
            if remaining <= 0:
                break
            await asyncio.sleep(min(remaining, 1.0) + 0.001)
        if not self._client.closed:
            self._client.send_nowait(_IDLE)

    # --------------------------------------------------------------------- sending
    def _send(self, etype: str, **fields: Any) -> None:
        name = event_name(etype, self.dialect)
        if name is None or self._writer_stop:
            return
        self._enqueue(_Out(_dumps({"event_id": new_id("event_"), "type": name, **fields})))

    def _enqueue(self, out: _Out) -> None:
        self._outbox.append(out)
        self._outbox_bytes += len(out.payload)
        self._outbox_ready.set()
        if self._outbox_bytes > self.server.max_send_buffer:
            self._outbox.clear()
            self._outbox_bytes = 0
            raise _Stop("client does not read fast enough (send buffer full)", _CLOSE_POLICY)

    def _send_error(self, error: dict[str, Any]) -> None:
        self.stats["errors"] += 1
        self._send("error", error={"param": None, "event_id": None, **error})

    def _purge(self, resp: _Response) -> None:
        """Drop queued (unsent) audio deltas of ``resp``."""
        kept: deque[_Out] = deque()
        for out in self._outbox:
            if out.response is resp:
                self._outbox_bytes -= len(out.payload)
                if out.item is not None:
                    out.item.audio_ms -= out.ms
                resp.purged_ms += out.ms
            else:
                kept.append(out)
        self._outbox = kept
        if self._outbox_bytes <= _OUTBOX_LOW:
            self._drained.set()

    # ------------------------------------------------------------------ rendering
    def _render_session(self) -> dict[str, Any]:
        return self.config.render(
            self.dialect, session_id=self.id, model=self.model.name, expires_at=self.expires_at
        )

    def _render_response(
        self,
        resp: _Response,
        status: str,
        details: dict[str, Any] | None = None,
        usage: EngineUsage | None = None,
    ) -> dict[str, Any]:
        cfg = self.config
        out: dict[str, Any] = {
            "object": "realtime.response",
            "id": resp.id,
            "status": status,
            "status_details": details,
            "output": [o.item.render(self.dialect) for o in resp.outputs],
            "conversation_id": self._conversation_id,
            "usage": None if status == "in_progress" else _usage(usage),
            "metadata": resp.metadata,
        }
        if self.dialect == "beta":
            out["modalities"] = ["text"] if resp.text_only else ["text", "audio"]
            out["voice"] = cfg.voice
            out["output_audio_format"] = cfg.output_format.beta_name()
            out["max_output_tokens"] = cfg.max_output_tokens
        else:
            out["output_modalities"] = list(resp.modalities)
            out["max_output_tokens"] = cfg.max_output_tokens
            out["audio"] = {"output": {"format": cfg.output_format.to_json(), "voice": cfg.voice}}
        return out

    # -------------------------------------------------------------- client events
    async def _on_client_message(self, raw: str | bytes) -> None:
        self._client_bytes -= len(raw)
        if self._client_bytes <= _INBOX_LOW:
            self._client_space.set()
        if isinstance(raw, bytes):
            self._send_error(
                {
                    "type": "invalid_request_error",
                    "code": "invalid_event",
                    "message": "Binary frames are not supported: send JSON events as text.",
                }
            )
            return
        try:
            event = json.loads(raw)
        except ValueError:
            self._send_error(
                {
                    "type": "invalid_request_error",
                    "code": "invalid_json",
                    "message": "The server could not parse the JSON body of your event.",
                }
            )
            return
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            self._send_error(
                {
                    "type": "invalid_request_error",
                    "code": "invalid_event",
                    "message": "Events must be JSON objects with a string 'type'.",
                }
            )
            return
        event_id = event.get("event_id") if isinstance(event.get("event_id"), str) else None
        handler = self._handlers.get(event["type"])
        try:
            if handler is None:
                raise ClientError(
                    f"Invalid value: {event['type']!r}. Unknown client event type.",
                    code="invalid_event",
                    param="type",
                )
            await handler(event)
        except ClientError as exc:
            logger.debug("realtime session %s: %s rejected: %s", self.id, event["type"], exc)
            self._send_error(exc.to_json(event_id))
        except (_Stop, asyncio.CancelledError):
            raise
        except Exception as exc:  # an engine call failed
            _, message = report_error(
                exc, f"Handling {event['type']} failed", session_id=self.id,
                level=logging.WARNING, traceback=False,
            )  # fmt: skip
            self._send_error(
                {
                    "type": "server_error",
                    "code": "engine_error",
                    "message": message,
                    "event_id": event_id,
                }
            )

    async def _on_unsupported(self, ev: dict[str, Any]) -> None:
        raise ClientError(
            f"{ev['type']} is not supported by this server (WebSocket, realtime sessions).",
            code="unsupported_event",
            param="type",
        )

    async def _on_session_update(self, ev: dict[str, Any]) -> None:
        old = self.config
        new, dialect = apply_session_update(old, ev.get("session"), self.dialect)
        if new.voice != old.voice and self._assistant_audio:
            raise ClientError(
                "Cannot update a conversation's voice if assistant audio is present.",
                code="cannot_update_voice",
                param="session.audio.output.voice",
            )
        if dialect != self.dialect:
            logger.debug("realtime session %s: switching to the %s dialect", self.id, dialect)
            self.dialect = dialect
        if new.input_format != old.input_format:
            self._rest = b""
        self.config = new
        changes: dict[str, Any] = {}
        if new.instructions != old.instructions:
            changes["instructions"] = new.instructions
        if new.tools != old.tools:
            changes["tools"] = new.function_tools()
        if changes:
            await self._conversation_changed()
            if self._conn is not None and not self._dirty:
                await self._conn.update(**changes)
        self._send("session.updated", session=self._render_session())

    async def _on_append(self, ev: dict[str, Any]) -> None:
        audio = ev.get("audio")
        if not isinstance(audio, str):
            raise ClientError(
                "Missing required parameter: 'audio'.",
                code="missing_required_parameter",
                param="audio",
            )
        try:
            data = base64.b64decode(audio, validate=True)
        except (binascii.Error, ValueError):
            raise ClientError("Invalid 'audio': expected base64-encoded audio.",
                              param="audio") from None  # fmt: skip
        fmt = self.config.input_format
        if fmt.bytes_per_sample == 2:
            data = self._rest + data if self._rest else data
            cut = len(data) - len(data) % 2
            self._rest = data[cut:]
            data = data[:cut]
        if not data:
            return
        frame = AudioFrame(fmt.decode(data), fmt.rate)
        frame.timestamp = now() - frame.duration  # fully captured by the time it arrived
        self._buffered += frame.duration
        self.stats["audio_in"] += frame.duration
        conn = await self._ensure_engine()
        await conn.send_audio(frame)

    async def _on_commit(self, ev: dict[str, Any]) -> None:
        if self._buffered < _MIN_COMMIT - 1e-6:
            raise ClientError(
                "Error committing input audio buffer: buffer too small. Expected at least "
                f"100ms of audio, but buffer only has {self._buffered * 1000:.2f}ms of audio.",
                code="input_audio_buffer_commit_empty",
            )
        conn = await self._ensure_engine()
        await self._conversation_changed()
        item_id = self._pending_user or new_id("item_")
        self._pending_user = None
        item = Item(item_id, "message", role="user", content="input_audio",
                    input_seconds=self._buffered)  # fmt: skip
        self._buffered = 0.0
        self._commit_user_item(item)
        self._mark(_Marker("commit_start", item_id=item_id))
        try:
            await conn.commit_input()
        finally:
            self._mark(_Marker("commit_end", item_id=item_id))

    async def _on_clear(self, ev: dict[str, Any]) -> None:
        self._buffered = 0.0
        self._rest = b""
        self._pending_user = None
        if self._conn is not None:
            await self._conn.clear_input()
        self._send("input_audio_buffer.cleared")

    async def _on_item_create(self, ev: dict[str, Any]) -> None:
        item = parse_item(ev.get("item"))
        if item.id and item.id in self._items:
            raise ClientError(f"Item with id {item.id!r} already exists.",
                              code="item_already_exists", param="item.id")  # fmt: skip
        item.id = item.id or new_id("item_")
        if item.type == "function_call" and not item.call_id:
            item.call_id = new_id("call_")
        previous = ev.get("previous_item_id")
        position = len(self._order)
        if previous == "root":
            position = 0
        elif previous is not None:
            if previous not in self._items:
                raise ClientError(f"Item with id {previous!r} not found.",
                                  code="item_not_found", param="previous_item_id")  # fmt: skip
            position = self._order.index(previous) + 1
        at_end = position == len(self._order)
        await self._conversation_changed()
        previous_id = self._add_item(item, position)
        conn = self._conn
        if conn is not None and not self._dirty:
            if at_end and item.type == "message" and item.role == "user":
                if (item.text or "").strip():
                    await conn.send_text(item.text or "", respond=False)
            elif at_end and item.type == "function_call_output":
                output = FunctionCallOutput(call_id=item.call_id or "", output=item.output)
                await conn.send_tool_output(output, respond=False)
            else:  # the engine API cannot insert this item: rebuild its context lazily
                self._dirty = True
        self._send("conversation.item.added", previous_item_id=previous_id,
                   item=item.render(self.dialect))  # fmt: skip
        self._send("conversation.item.done", previous_item_id=previous_id,
                   item=item.render(self.dialect))  # fmt: skip

    async def _on_item_delete(self, ev: dict[str, Any]) -> None:
        item_id = _required_str(ev, "item_id")
        item = self._items.get(item_id)
        if item is None:
            raise ClientError(
                f"Error deleting item: the item with id '{item_id}' does not exist.",
                code="item_delete_invalid_item_id",
                param="item_id",
            )
        await self._conversation_changed()
        del self._items[item_id]
        self._order.remove(item_id)
        if item.engine_id is not None:
            self._engine_items.pop(item.engine_id, None)
        if self._conn is not None:
            self._dirty = True  # engines cannot delete items: rebuild the context lazily
        self._send("conversation.item.deleted", item_id=item_id)

    async def _on_item_retrieve(self, ev: dict[str, Any]) -> None:
        item_id = _required_str(ev, "item_id")
        item = self._items.get(item_id)
        if item is None:
            raise ClientError(
                f"Error retrieving item: the item with id '{item_id}' does not exist.",
                code="item_retrieve_invalid_item_id",
                param="item_id",
            )
        self._send("conversation.item.retrieved", item=item.render(self.dialect))

    async def _on_truncate(self, ev: dict[str, Any]) -> None:
        item_id = _required_str(ev, "item_id")
        content_index = ev.get("content_index", 0)
        end = ev.get("audio_end_ms")
        item = self._items.get(item_id)
        if item is None:
            raise ClientError(
                f"Error truncating item: the item with id '{item_id}' does not exist.",
                code="item_truncate_invalid_item_id",
                param="item_id",
            )
        if item.type != "message" or item.role != "assistant" or item.content != "output_audio":
            raise ClientError("Only assistant messages with audio can be truncated.",
                              code="unsupported_content_type", param="item_id")  # fmt: skip
        if content_index != 0:
            raise ClientError("Invalid content_index: the item has one content part (0).",
                              param="content_index")  # fmt: skip
        if isinstance(end, bool) or not isinstance(end, int) or end < 0:
            raise ClientError("audio_end_ms must be a non-negative integer", param="audio_end_ms")
        total = math.ceil(item.audio_ms - 1e-6)
        if end > total:
            raise ClientError(
                f"audio_end_ms ({end}) is greater than the audio duration ({total}ms).",
                param="audio_end_ms",
            )
        await self._conversation_changed()
        heard: str | None = None
        conn = self._conn
        if conn is not None and item.engine_id is not None and not self._dirty:
            heard = await conn.truncate(item.engine_id, end)
        elif conn is not None:
            self._dirty = True
        text = item.text or ""
        if heard is None:  # estimate: text is roughly proportional to audio
            share = min(1.0, end / item.audio_ms) if item.audio_ms > 0 else 0.0
            heard = text[: round(len(text) * share)]
        item.text = heard.strip()
        self._send(
            "conversation.item.truncated", item_id=item_id, content_index=0, audio_end_ms=end
        )

    async def _on_response_create(self, ev: dict[str, Any]) -> None:
        request = self._parse_request(ev.get("response"), ev.get("event_id"))
        busy = self._active if self._active is not None and not self._active.finished else None
        pending = self._requesting
        if pending is not None and now() - pending.created > _REQUEST_TTL:
            pending = self._requesting = None
        if busy is not None or pending is not None:
            rid = busy.id if busy is not None else "(starting)"
            raise ClientError(
                f"Conversation already has an active response in progress: {rid}. Wait until "
                "the response is finished before creating a new one.",
                code="conversation_already_has_active_response",
            )
        held = self._held
        if held is not None:
            if request.plain and self._adoptable(held):
                self._adopt(held, request)
                return
            await self._discard_held()
        if self._auto in ("hold", "forward") and request.plain:
            # the engine is about to answer the committed turn: that answer is this response
            self._auto = request
            self._requesting = request
            return
        conn = await self._ensure_engine()
        self._requesting = request
        try:
            if request.verbatim is not None:
                await conn.say(request.verbatim)
            else:
                await conn.create_response(instructions=request.instructions)
        except BaseException:
            self._requesting = None
            raise
        self._mark(_Marker("request", request=request))

    async def _on_response_cancel(self, ev: dict[str, Any]) -> None:
        rid = ev.get("response_id")
        active = self._active
        if active is not None and not active.finished and (not rid or rid == active.id):
            await self._cancel_active("client_cancelled")
            return
        if self._requesting is not None and not rid:
            self._cancel_requested = True  # created and cancelled as soon as it starts
            return
        await self._discard_held()
        raise ClientError(
            "Cancellation failed: no active response found", code="response_cancel_not_active"
        )

    def _parse_request(self, body: Any, event_id: Any) -> _Request:
        if body is None:
            body = {}
        if not isinstance(body, Mapping):
            raise ClientError("response must be an object", param="response")
        conversation = body.get("conversation", "auto")
        if conversation not in (None, "auto"):
            raise ClientError(
                "Out-of-band responses (conversation: 'none') are not supported by this server.",
                code="unsupported_value",
                param="response.conversation",
            )
        instructions = body.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise ClientError("instructions must be a string", param="response.instructions")
        modalities = parse_response_modalities(body)
        verbatim: str | None = None
        extra: str | None = None
        if instructions:
            match = _VERBATIM.search(instructions)
            if match is not None:
                verbatim = match.group(1)
            else:
                base, text = self.config.instructions.strip(), instructions.strip()
                if base and text.startswith(base):  # "<session instructions>\n\n<extra>"
                    text = text[len(base) :].strip()
                extra = text or None
        plain = (
            extra is None
            and verbatim is None
            and body.get("input") is None
            and body.get("tools") is None
            and body.get("tool_choice") is None
            and (modalities is None or modalities == self.config.output_modalities)
        )
        return _Request(
            event_id=event_id if isinstance(event_id, str) else None,
            plain=plain,
            modalities=modalities,
            metadata=body.get("metadata"),
            instructions=extra,
            verbatim=verbatim,
        )

    # -------------------------------------------------------------- engine events
    async def _on_engine_event(self, ev: Any) -> None:
        if ev is _ENGINE_CLOSED:
            if self._conn is not None:
                self._fail_active("the engine connection closed")
                self._send_error(
                    {
                        "type": "server_error",
                        "code": "engine_closed",
                        "message": "The engine connection closed.",
                    }
                )
                raise _Stop("engine connection closed", _CLOSE_INTERNAL)
        elif isinstance(ev, _Marker):
            self._on_marker(ev)
        elif isinstance(ev, InputSpeechStarted):
            await self._on_speech_started(ev)
        elif isinstance(ev, InputSpeechStopped):
            self._on_speech_stopped(ev)
        elif isinstance(ev, InputTranscript):
            self._on_transcript(ev)
        elif isinstance(ev, InputCommitted):
            await self._on_committed(ev)
        elif isinstance(ev, ResponseStarted):
            await self._on_response_started(ev)
        elif isinstance(ev, (ResponseText, ResponseAudio, ResponseToolCall, ResponseDone)):
            await self._on_response_event(ev)
        elif isinstance(ev, EngineErrorEvent):
            kind = "recoverable" if ev.recoverable else "fatal"
            error_id, message = report_error(
                ev.error, f"The engine reported a {kind} error", session_id=self.id,
                level=logging.WARNING, traceback=False,
            )  # fmt: skip
            self._send_error({"type": "server_error", "code": "engine_error", "message": message})
            if not ev.recoverable:
                self._fail_active(message)
                raise _Stop(f"engine error ({error_id})", _CLOSE_INTERNAL)
        elif isinstance(ev, EngineStatus):
            logger.info("realtime session %s: engine %s (%s)", self.id, ev.status, ev.detail)
        elif isinstance(ev, ToolCallCancelled):
            logger.debug("realtime session %s: engine withdrew calls %s", self.id, ev.call_ids)

    def _on_marker(self, marker: _Marker) -> None:
        if marker.kind == "commit_start":
            self._commit_binding = marker.item_id
        elif marker.kind == "commit_end":
            if marker.item_id is not None and self._commit_binding == marker.item_id:
                self._commit_binding = None  # the engine found nothing to commit
                if self.config.transcribe:
                    self._send_transcript_completed(marker.item_id, "")
        else:
            self._request = marker.request
            if not isinstance(self._auto, _Request):
                self._auto = None  # create_response() superseded the engine's own answer

    async def _on_speech_started(self, ev: InputSpeechStarted) -> None:
        if not self.config.vad:
            return  # manual turns: no VAD events, like OpenAI with turn_detection null
        item_id = self._pending_user or new_id("item_")
        self._pending_user = item_id
        self._announced.add(item_id)
        self._send(
            "input_audio_buffer.speech_started",
            audio_start_ms=self._audio_ms(ev.audio_time),
            item_id=item_id,
        )
        if self.config.interrupt_response:
            await self._cancel_active("turn_detected")

    def _on_speech_stopped(self, ev: InputSpeechStopped) -> None:
        if not self.config.vad:
            return
        item_id = self._pending_user or new_id("item_")
        self._pending_user = item_id
        self._announced.add(item_id)
        self._send(
            "input_audio_buffer.speech_stopped",
            audio_end_ms=self._audio_ms(ev.audio_time),
            item_id=item_id,
        )

    def _audio_ms(self, audio_time: float | None) -> int:
        conn = self._conn
        position = audio_time
        if position is None:
            position = conn.input_audio_time if conn is not None else 0.0
        return max(0, round((self._audio_offset + position) * 1000))

    def _on_transcript(self, ev: InputTranscript) -> None:
        item_id = self._engine_items.get(ev.item_id)
        if item_id is None:
            if ev.is_final:
                return  # not a turn this session announced
            item_id = self._pending_user or new_id("item_")
            self._pending_user = item_id
            self._engine_items[ev.item_id] = item_id
        item = self._items.get(item_id)
        if not ev.is_final:
            sent = self._partials.get(item_id, "")
            if item_id in self._announced and ev.text.startswith(sent) and len(ev.text) > len(sent):
                if self.config.transcribe:
                    self._send(
                        "conversation.item.input_audio_transcription.delta",
                        item_id=item_id,
                        content_index=0,
                        delta=ev.text[len(sent) :],
                    )
                self._partials[item_id] = ev.text
            return
        self._partials.pop(item_id, None)
        text = ev.text.strip()
        if item is not None:
            item.text = text
        if self.config.transcribe:
            self._send_transcript_completed(item_id, text, ev.language)

    def _send_transcript_completed(
        self, item_id: str, text: str, language: str | None = None
    ) -> None:
        item = self._items.get(item_id)
        seconds = item.input_seconds if item is not None else 0.0
        fields: dict[str, Any] = {
            "item_id": item_id,
            "content_index": 0,
            "transcript": text,
            "usage": {"type": "duration", "seconds": round(seconds, 3)},
        }
        if language:
            fields["languages"] = [{"code": language}]
        self._send("conversation.item.input_audio_transcription.completed", **fields)

    async def _on_committed(self, ev: InputCommitted) -> None:
        binding = self._commit_binding
        if binding is not None and binding in self._items:
            # our input_audio_buffer.commit: the item was announced already
            self._commit_binding = None
            self._items[binding].engine_id = ev.item_id
            self._engine_items[ev.item_id] = binding
            self._auto = "hold"
            return
        await self._discard_held(cancel=False)  # the engine answers the new turn instead
        item_id = self._engine_items.get(ev.item_id) or self._pending_user or new_id("item_")
        self._pending_user = None
        self._engine_items[ev.item_id] = item_id
        item = Item(item_id, "message", role="user", content="input_audio",
                    engine_id=ev.item_id, input_seconds=self._buffered)  # fmt: skip
        self._buffered = 0.0
        self._commit_user_item(item)
        self._auto = "forward" if self.config.vad and self.config.create_response else "hold"

    def _commit_user_item(self, item: Item) -> None:
        previous = self._add_item(item)
        self._announced.add(item.id)
        self._version += 1
        self._send("input_audio_buffer.committed", previous_item_id=previous, item_id=item.id)
        self._send("conversation.item.added", previous_item_id=previous,
                   item=item.render(self.dialect))  # fmt: skip
        self._send("conversation.item.done", previous_item_id=previous,
                   item=item.render(self.dialect))  # fmt: skip

    # ------------------------------------------------------------------ responses
    async def _on_response_started(self, ev: ResponseStarted) -> None:
        request: _Request | None = None
        state: Literal["held", "forwarded", "discarded"] = "forwarded"
        if self._request is not None:
            request, self._request = self._request, None
        elif isinstance(self._auto, _Request):
            request = self._auto
        elif self._auto == "hold":
            state = "held"
        elif self._auto == "discard":
            state = "discarded"
        self._auto = None
        if request is not None and self._requesting is request:
            self._requesting = None
        await self._discard_held(cancel=False)  # replaced (engines run one response at a time)
        rid = ev.response_id if ev.response_id and ev.response_id not in self._responses else ""
        modalities = request.modalities if request is not None and request.modalities else None
        resp = _Response(
            engine_id=ev.response_id,
            id=rid or new_id("resp_"),
            state=state,
            version=self._version,
            modalities=modalities or list(self.config.output_modalities),
            request=request,
            metadata=request.metadata if request is not None else None,
        )
        self._responses[ev.response_id] = resp
        if state == "held":
            self._held = resp
        elif state == "discarded":
            if self._conn is not None:
                await self._conn.cancel_response()
        else:
            self._open_response(resp)
            if self._cancel_requested and request is not None:
                self._cancel_requested = False
                await self._cancel_active("client_cancelled")

    def _open_response(self, resp: _Response) -> None:
        self.stats["responses"] += 1
        self._active = resp
        self._send("response.created", response=self._render_response(resp, "in_progress"))

    def _adoptable(self, resp: _Response) -> bool:
        done = resp.engine_done
        return resp.version == self._version and (done is None or done.status == "completed")

    def _adopt(self, resp: _Response, request: _Request) -> None:
        self._held = None
        resp.state = "forwarded"
        resp.request = request
        resp.metadata = request.metadata
        if request.modalities:
            resp.modalities = request.modalities
        self._open_response(resp)
        events, resp.buffered = resp.buffered, []
        for ev in events:
            self._forward(resp, ev)

    async def _discard_held(self, *, cancel: bool = True) -> None:
        """Drop the held response; the engine keeps it in its context, cut to nothing."""
        resp = self._held
        if resp is None:
            return
        self._held = None
        resp.state = "discarded"
        resp.buffered = []
        conn = self._conn
        if resp.engine_done is not None:
            self._responses.pop(resp.engine_id, None)
        elif cancel and conn is not None:
            await conn.cancel_response()
        if conn is not None and conn.capabilities.truncation:
            for engine_item in resp.engine_items:
                with contextlib.suppress(Exception):
                    await conn.truncate(engine_item, 0)

    async def _conversation_changed(self) -> None:
        """The client changed the conversation: an unshown answer no longer fits it."""
        self._version += 1
        await self._discard_held()
        if self._auto == "hold":
            self._auto = "discard"

    async def _on_response_event(
        self, ev: ResponseText | ResponseAudio | ResponseToolCall | ResponseDone
    ) -> None:
        resp = self._responses.get(ev.response_id)
        if resp is None:
            return  # late event of a response we no longer track
        if isinstance(ev, (ResponseText, ResponseAudio)) and ev.item_id not in resp.engine_items:
            resp.engine_items.append(ev.item_id)
        if resp.state == "forwarded":
            self._forward(resp, ev)
        elif resp.state == "held":
            resp.buffered.append(ev)
            if isinstance(ev, ResponseAudio):
                resp.buffered_audio += ev.frame.duration
                if resp.buffered_audio > _MAX_HELD_AUDIO:
                    await self._discard_held()
            elif isinstance(ev, ResponseDone):
                resp.engine_done = ev
                if ev.status != "completed" and self._held is resp:
                    await self._discard_held(cancel=False)
        elif isinstance(ev, ResponseDone):
            self._responses.pop(ev.response_id, None)

    async def _cancel_active(self, reason: str) -> None:
        resp = self._active
        if resp is None or resp.finished or resp.cancel_reason is not None:
            return
        resp.cancel_reason = reason
        self._purge(resp)
        if self._conn is not None:
            await self._conn.cancel_response()

    def _fail_active(self, message: str) -> None:
        """End the response in progress (the engine went away)."""
        resp = self._active
        if resp is not None and not resp.finished:
            self._finish(resp, ResponseDone(response_id=resp.engine_id, status="failed",
                                            error=message))  # fmt: skip

    # ------------------------------------------------------------- forwarding
    def _forward(
        self, resp: _Response, ev: ResponseText | ResponseAudio | ResponseToolCall | ResponseDone
    ) -> None:
        if resp.finished:
            return
        if resp.cancel_reason is not None and not isinstance(ev, ResponseDone):
            if isinstance(ev, ResponseAudio):  # generated before the cancel took effect
                resp.purged_ms += ev.frame.duration_ms
            return
        if isinstance(ev, ResponseText):
            if not ev.delta:
                return
            out = self._message(resp, ev.item_id)
            out.parts.append(ev.delta)
            out.item.text = "".join(out.parts)
            etype = (
                "response.output_text.delta"
                if resp.text_only
                else "response.output_audio_transcript.delta"
            )
            self._send(etype, **self._where(resp, out), delta=ev.delta)
        elif isinstance(ev, ResponseAudio):
            if resp.text_only or not ev.frame:
                return
            out = self._message(resp, ev.item_id)
            assert out.resampler is not None
            self._send_audio(resp, out, out.resampler.push(ev.frame))
        elif isinstance(ev, ResponseToolCall):
            self._close_message(resp, "completed")
            self._function_call(resp, ev.call)
        else:
            self._finish(resp, ev)

    def _where(self, resp: _Response, out: _Output) -> dict[str, Any]:
        return {
            "response_id": resp.id,
            "item_id": out.item.id,
            "output_index": out.index,
            "content_index": 0,
        }

    def _message(self, resp: _Response, engine_item_id: str) -> _Output:
        out = resp.message
        if out is not None and out.engine_id == engine_item_id:
            return out
        self._close_message(resp, "completed")
        item_id = new_id("item_")
        item = Item(
            item_id, "message", role="assistant",
            content="output_text" if resp.text_only else "output_audio",
            text="", status="in_progress", engine_id=engine_item_id,
        )  # fmt: skip
        self._engine_items[engine_item_id] = item_id
        fmt = self.config.output_format
        out = _Output(item, len(resp.outputs), engine_item_id, fmt, StreamResampler(fmt.rate, 1))
        resp.outputs.append(out)
        resp.message = out
        previous = self._add_item(item)
        rendered = item.render(self.dialect)
        self._send("response.output_item.added", response_id=resp.id, output_index=out.index,
                   item=rendered)  # fmt: skip
        self._send("conversation.item.added", previous_item_id=previous, item=rendered)
        part: dict[str, Any] = (
            {"type": "text", "text": ""} if resp.text_only else {"type": "audio", "transcript": ""}
        )
        self._send("response.content_part.added", **self._where(resp, out), part=part)
        return out

    def _send_audio(self, resp: _Response, out: _Output, frame: AudioFrame) -> None:
        if not frame or out.format is None:
            return
        self._assistant_audio = True
        step = max(1, round(_AUDIO_CHUNK * frame.sample_rate)) * 2
        data = frame.data
        where = self._where(resp, out)
        name = event_name("response.output_audio.delta", self.dialect)
        for i in range(0, len(data), step):
            piece = data[i : i + step]
            ms = len(piece) / 2 / frame.sample_rate * 1000.0
            delta = base64.b64encode(out.format.encode(piece)).decode("ascii")
            payload = _dumps({"event_id": new_id("event_"), "type": name, **where, "delta": delta})
            out.item.audio_ms += ms
            self.stats["audio_out"] += ms / 1000.0
            if not self._writer_stop:
                self._enqueue(_Out(payload, resp, out.item, ms))

    def _close_message(self, resp: _Response, status: str) -> None:
        out = resp.message
        if out is None:
            return
        resp.message = None
        if out.resampler is not None and status == "completed" and resp.cancel_reason is None:
            self._send_audio(resp, out, out.resampler.flush())
        text = "".join(out.parts)
        where = self._where(resp, out)
        if resp.text_only:
            self._send("response.output_text.done", **where, text=text)
            self._send("response.content_part.done", **where, part={"type": "text", "text": text})
        else:
            self._send("response.output_audio.done", **where)
            self._send("response.output_audio_transcript.done", **where, transcript=text)
            self._send("response.content_part.done", **where,
                       part={"type": "audio", "transcript": text})  # fmt: skip
        out.item.status = status
        rendered = out.item.render(self.dialect)
        self._send("response.output_item.done", response_id=resp.id, output_index=out.index,
                   item=rendered)  # fmt: skip
        self._send("conversation.item.done", item=rendered)

    def _function_call(self, resp: _Response, call: FunctionCall) -> None:
        item_id = new_id("item_")
        item = Item(item_id, "function_call", call_id=call.call_id, name=call.name,
                    status="in_progress", engine_id=call.id)  # fmt: skip
        self._engine_items[call.id] = item_id
        out = _Output(item, len(resp.outputs), call.id)
        resp.outputs.append(out)
        previous = self._add_item(item)
        rendered = item.render(self.dialect)
        self._send("response.output_item.added", response_id=resp.id, output_index=out.index,
                   item=rendered)  # fmt: skip
        self._send("conversation.item.added", previous_item_id=previous, item=rendered)
        arguments = call.arguments if call.arguments and call.arguments.strip() else "{}"
        where = {
            "response_id": resp.id,
            "item_id": item_id,
            "output_index": out.index,
            "call_id": call.call_id,
        }
        self._send("response.function_call_arguments.delta", **where, delta=arguments)
        self._send("response.function_call_arguments.done", **where, name=call.name,
                   arguments=arguments)  # fmt: skip
        item.arguments = arguments
        item.status = "completed"
        rendered = item.render(self.dialect)
        self._send("response.output_item.done", response_id=resp.id, output_index=out.index,
                   item=rendered)  # fmt: skip
        self._send("conversation.item.done", item=rendered)

    def _finish(self, resp: _Response, done: ResponseDone) -> None:
        self._responses.pop(resp.engine_id, None)
        status: ResponseStatus = done.status
        if status == "completed" and resp.purged_ms > 0:
            status = "cancelled"  # the client never got the end of it
        self._close_message(resp, "completed" if status == "completed" else "incomplete")
        details: dict[str, Any] | None = None
        if status == "cancelled":
            details = {"type": "cancelled", "reason": resp.cancel_reason or "turn_detected"}
        elif status == "failed":
            message = done.error or "the engine failed to generate the response"
            details = {
                "type": "failed",
                "error": {"type": "server_error", "code": "engine_error", "message": message},
            }
        elif status == "incomplete":
            details = {"type": "incomplete", "reason": done.error or "max_output_tokens"}
        resp.finished = True
        if self._active is resp:
            self._active = None
        self._send("response.done", response=self._render_response(resp, status, details,
                                                                   done.usage))  # fmt: skip

    # --------------------------------------------------------------- conversation
    def _add_item(self, item: Item, position: int | None = None) -> str | None:
        """Insert ``item``; returns the id of the item before it."""
        if position is None:
            position = len(self._order)
        previous = self._order[position - 1] if position > 0 else None
        self._order.insert(position, item.id)
        self._items[item.id] = item
        return previous

    def _chat_context(self) -> tuple[ChatContext | None, list[Item]]:
        """The conversation as engine history (for a new engine connection)."""
        ctx = ChatContext()
        seeded: list[Item] = []
        for item_id in self._order:
            item = self._items[item_id]
            if item.type == "message":
                text = (item.text or "").strip()
                if not text or item.role is None:
                    continue
                ctx.add_message(item.role, text, id=item.id)
            elif item.type == "function_call":
                ctx.append(FunctionCall(name=item.name or "", arguments=item.arguments or "{}",
                                        call_id=item.call_id or item.id, id=item.id))  # fmt: skip
            else:
                ctx.append(FunctionCallOutput(call_id=item.call_id or "", output=item.output,
                                              id=item.id))  # fmt: skip
            seeded.append(item)
        return (ctx if ctx.items else None), seeded

    # --------------------------------------------------------------------- engine
    def _engine_settings(self) -> tuple[Any, ...]:
        """What only a new engine connection can change."""
        cfg = self.config
        return (cfg.vad, cfg.voice, cfg.language or self.model.language)

    async def _ensure_engine(self) -> EngineConnection:
        """The engine connection, (re)opened lazily with the current configuration."""
        conn = self._conn
        if conn is not None:
            stale = self._dirty or self._settings != self._engine_settings()
            busy = (self._active is not None and not self._active.finished) or (
                self._requesting is not None
            )
            if not stale or busy:
                return conn
            logger.debug("realtime session %s: reconnecting the engine", self.id)
            await self._close_engine()
        return await self._connect_engine()

    async def _connect_engine(self) -> EngineConnection:
        if self._engine is None:
            self._engine, self._owns_engine = await self.model.acquire()
        engine = self._engine
        ctx, seeded = self._chat_context()
        cfg = self.config
        options = EngineOptions(
            instructions=cfg.instructions,
            tools=cfg.function_tools(),
            chat_ctx=ctx,
            voice=cfg.voice,
            language=cfg.language or self.model.language,
            turn_detection=cfg.vad,
        )
        try:
            conn = await asyncio.wait_for(
                engine.connect(options), self.server.engine_connect_timeout
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _, message = report_error(
                exc, "The engine is unavailable", session_id=self.id, traceback=False
            )
            self._send_error(
                {"type": "server_error", "code": "engine_unavailable", "message": message}
            )
            raise _Stop("engine unavailable", _CLOSE_INTERNAL) from exc
        self._conn = conn
        self._settings = self._engine_settings()
        self._dirty = False
        for item in seeded:
            item.engine_id = item.id
            self._engine_items[item.id] = item.id
        events = conn.events()
        if isinstance(events, Chan):
            self._events = events
        else:  # a custom event iterator: pump it (markers then land in our own channel)
            chan: Chan[Any] = Chan()
            self._events = chan
            self._pump = asyncio.create_task(_pump(events, chan))
        return conn

    async def _close_engine(self) -> None:
        conn, self._conn = self._conn, None
        self._events = None
        await cancel_and_wait(self._pump)
        self._pump = None
        if conn is None:
            return
        self._audio_offset += conn.input_audio_time
        # the connection's responses and pending requests end with it
        self._responses.clear()
        self._held = None
        self._auto = None
        self._request = None
        self._requesting = None
        self._commit_binding = None
        for item in self._items.values():
            item.engine_id = None
        self._engine_items.clear()
        with contextlib.suppress(Exception):
            await conn.aclose()

    def _mark(self, marker: _Marker) -> None:
        events = self._events
        if events is not None and not events.closed:
            events.send_nowait(marker)


# ----------------------------------------------------------------------------- helpers
async def _pump(source: Any, chan: Chan[Any]) -> None:
    try:
        async for ev in source:
            chan.send_nowait(ev)
    finally:
        chan.close()


def _dumps(event: dict[str, Any]) -> str:
    return json.dumps(event, separators=(",", ":"), ensure_ascii=False)


def _usage(usage: EngineUsage | None) -> dict[str, Any]:
    u = usage or EngineUsage()
    inp = u.input_text_tokens + u.input_audio_tokens
    out = u.output_text_tokens + u.output_audio_tokens
    return {
        "total_tokens": inp + out,
        "input_tokens": inp,
        "output_tokens": out,
        "input_token_details": {
            "text_tokens": u.input_text_tokens,
            "audio_tokens": u.input_audio_tokens,
            "image_tokens": 0,
            "cached_tokens": u.cached_tokens,
            "cached_tokens_details": {"text_tokens": 0, "audio_tokens": 0, "image_tokens": 0},
        },
        "output_token_details": {
            "text_tokens": u.output_text_tokens,
            "audio_tokens": u.output_audio_tokens,
        },
    }


def _required_str(ev: Mapping[str, Any], key: str) -> str:
    value = ev.get(key)
    if not isinstance(value, str) or not value:
        raise ClientError(
            f"Missing required parameter: '{key}'.", code="missing_required_parameter", param=key
        )
    return value


def _peer(remote: Any) -> str:
    if isinstance(remote, tuple) and len(remote) >= 2:
        return f"{remote[0]}:{remote[1]}"
    return str(remote)
