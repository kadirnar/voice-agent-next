"""WebSocket server transport (protocol ``van-ws/1``) and a one-session-per-connection server.

Serve voice agents to browsers and backends over a plain WebSocket::

    from voice_agent_next import Agent, AgentSession
    from voice_agent_next.transports.websocket import serve_websocket

    async def main() -> None:
        server = await serve_websocket(
            lambda: AgentSession("openai/gpt-realtime"),
            lambda: Agent("You are a helpful assistant."),
            host="127.0.0.1",
            port=8765,
        )
        await server.serve_forever()

Protocol summary (full specification: ``docs/transports/websocket.md``):

* **binary frames** carry raw PCM s16le audio — client -> server at the rate announced in
  ``hello``, server -> client at ``ready.output_sample_rate``;
* **JSON text frames** carry control messages — ``hello`` -> ``ready``; server -> client
  ``clear`` (barge-in), ``transcript``, ``state``, ``metrics``, ``error``; client -> server
  ``playback`` (playout position), ``text`` (typed input).

WebSocket runs over TCP, so a lost packet stalls the stream (head-of-line blocking). It is
a good fit for backends, LANs and prototypes; prefer WebRTC for clients on lossy networks.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import inspect
import json
import math
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from ..audio.frame import AudioFormat, AudioFrame
from ..audio.resample import StreamResampler
from ..errors import TransportError
from ..metrics import metrics_to_dict
from ..session.events import AgentState
from ..utils.aio import BackgroundTasks, Chan, cancel_and_wait, wait_first
from ..utils.clock import now
from ..utils.ids import new_id
from ..utils.log import logger
from .base import Transport, TransportCapabilities

if TYPE_CHECKING:
    from ..metrics import Metrics
    from ..session.agent import Agent
    from ..session.events import (
        AgentStateChanged,
        AgentTranscript,
        Interrupted,
        SessionError,
        UserStateChanged,
        UserTranscript,
    )
    from ..session.session import AgentSession

__all__ = [
    "PROTOCOL",
    "SessionBridge",
    "WebSocketAgentServer",
    "WebSocketServerTransport",
    "serve_websocket",
]

PROTOCOL = "van-ws/1"
"""Protocol identifier exchanged in ``hello`` / ``ready``."""

CODEC = "pcm_s16le"
_CODECS = frozenset({"pcm_s16le", "pcm16", "s16le"})
_FRAMINGS = frozenset({"binary", "base64"})
_MIN_RATE, _MAX_RATE = 8_000, 48_000

# WebSocket close codes (RFC 6455 §7.4.1)
CLOSE_NORMAL = 1000
CLOSE_PROTOCOL_ERROR = 1002
CLOSE_INTERNAL_ERROR = 1011
CLOSE_TRY_AGAIN_LATER = 1013

SessionFactory = Callable[..., "AgentSession | Awaitable[AgentSession]"]
"""``() -> AgentSession`` or ``(transport) -> AgentSession`` (sync or async)."""
AgentFactory = Callable[..., "Agent | Awaitable[Agent]"]
"""``() -> Agent`` or ``(transport) -> Agent`` (sync or async)."""


@dataclass(slots=True)
class _Outgoing:
    payload: str | bytes
    samples: int = 0
    """Agent audio samples carried by this message (0 for control messages)."""


class _HandshakeError(TransportError):
    def __init__(self, code: str, message: str, close_code: int = CLOSE_PROTOCOL_ERROR) -> None:
        super().__init__(message)
        self.code = code
        self.close_code = close_code


class WebSocketServerTransport(Transport):
    """Server side of one ``van-ws/1`` WebSocket connection.

    Two ways to use it:

    * **per connection** — wrap a connection accepted by a ``websockets`` server; this is
      what :func:`serve_websocket` does for every client (one session per connection);
    * **standalone** — ``create_transport({"type": "websocket", "host": ..., "port": ...})``:
      :meth:`start` listens on ``host:port`` and serves the *first* client that completes
      the handshake; other clients are refused (close code 1013) while it is connected, and
      the transport (hence the session) ends when that client leaves. Attach a
      :class:`SessionBridge` to also send transcripts, state changes and metrics.

    :meth:`start` performs the handshake: it waits for the client's ``hello``, adopts the
    announced input format and preferred output sample rate, and answers ``ready``.
    :meth:`buffered_duration` follows the client's ``playback`` reports when it sends them
    and falls back to a real-time (wall-clock) estimate otherwise.

    Args:
        websocket: an accepted connection (per-connection mode) or ``None`` (standalone).
        host: interface to listen on in standalone mode.
        port: port to listen on in standalone mode (``0`` picks a free one; see :attr:`port`).
        input_sample_rate: user audio rate assumed when ``hello`` does not announce one.
        output_sample_rate: agent audio rate unless ``hello`` asks for another one.
        frame_duration: maximum duration of each agent audio message, in seconds.
        hello_timeout: seconds to wait for ``hello`` before closing the connection.
        flush_timeout: on close, seconds to wait for queued control messages to be sent.
        serve_options: extra arguments for ``websockets.asyncio.server.serve`` (standalone
            mode), e.g. ``ssl``, ``origins`` or ``process_request``.
    """

    capabilities = TransportCapabilities(playback_position=True, messages=True)

    def __init__(
        self,
        websocket: ServerConnection | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        input_sample_rate: int = 16_000,
        output_sample_rate: int = 24_000,
        frame_duration: float = 0.02,
        hello_timeout: float = 10.0,
        flush_timeout: float = 1.0,
        serve_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            input_format=AudioFormat(input_sample_rate, 1),
            output_format=AudioFormat(output_sample_rate, 1),
        )
        if frame_duration <= 0:
            raise ValueError(f"frame_duration must be > 0, got {frame_duration}")
        self.websocket = websocket
        self.host = host
        self.port = port
        self.frame_duration = frame_duration
        self.hello_timeout = hello_timeout
        self.flush_timeout = flush_timeout
        self.serve_options = dict(serve_options or {})
        self.session_id = new_id("ws_")
        self.hello: dict[str, Any] = {}
        """The client's ``hello`` message (``hello.get("metadata")`` holds app data)."""
        self.framing = "binary"
        """``"binary"`` or ``"base64"`` (agent audio as JSON ``audio`` messages)."""
        self.close_code = CLOSE_NORMAL
        self.close_reason = "session closed"
        self._server: Server | None = None
        self._pending: asyncio.Queue[ServerConnection | None] | None = None
        self._start_lock = asyncio.Lock()
        self._ready = False
        self._closed = False
        self._disconnected = asyncio.Event()
        self._input: Chan[AudioFrame] = Chan()
        self._outbox: Chan[_Outgoing] = Chan()
        self._reader: asyncio.Task[None] | None = None
        self._writer: asyncio.Task[None] | None = None
        self._tasks = BackgroundTasks("ws-transport")
        self._in_rest = b""
        self._out_resampler = StreamResampler(output_sample_rate, 1)
        # playout accounting, in samples at the output rate
        self._out_samples = 0
        """Agent audio accepted for the client (audio dropped by ``clear`` excluded)."""
        self._clear_floor = 0
        """``_out_samples`` at the last ``clear``: older playback reports are stale."""
        self._play_end = 0.0
        """Estimated :func:`now` at which the client finishes playing what it was sent."""

    # ------------------------------------------------------------------ properties
    @property
    def url(self) -> str:
        """``ws://host:port/`` of the listening socket (standalone mode)."""
        scheme = "wss" if self.serve_options.get("ssl") is not None else "ws"
        return f"{scheme}://{_url_host(self.host)}:{self.port}/"

    @property
    def path(self) -> str | None:
        """Request path (with query string) of the client connection."""
        request = getattr(self.websocket, "request", None)
        return getattr(request, "path", None)

    @property
    def connected(self) -> bool:
        """The handshake completed and the client has not disconnected yet."""
        return self._ready and not self._disconnected.is_set()

    # ------------------------------------------------------------------- lifecycle
    async def listen(self) -> None:
        """Standalone mode: start listening (idempotent). :attr:`port` is then the bound port."""
        if self._server is not None:
            return
        if self.websocket is not None:
            raise RuntimeError("listen() is only available in standalone mode")
        self._pending = asyncio.Queue()
        options: dict[str, Any] = {"compression": None, **self.serve_options}
        self._server = await serve(self._accept, self.host, self.port, **options)
        self.port = _bound_port(self._server, self.port)
        logger.info("WebSocket transport waiting for a client on %s", self.url)

    async def start(self) -> None:
        """Perform the ``hello``/``ready`` handshake (standalone: wait for a client first).

        Idempotent. Raises :class:`~voice_agent_next.errors.TransportError` when the client
        of a per-connection transport fails the handshake (it was told why and disconnected).
        """
        async with self._start_lock:
            if self._ready:
                return
            if self._closed:
                raise TransportError("transport is closed")
            if self.websocket is not None:
                await self._handshake(self.websocket)
            else:
                await self._accept_first_client()
            self._reader = asyncio.create_task(self._read_loop(), name="ws-reader")
            self._writer = asyncio.create_task(self._write_loop(), name="ws-writer")
        remote = getattr(self.websocket, "remote_address", None)
        logger.info(
            "WebSocket client %s connected (%s, input %s, output %s, %s)",
            remote, self.session_id, self.input_format, self.output_format, self.framing,
        )  # fmt: skip
        self.emit("connected")

    async def wait_disconnected(self) -> None:
        """Wait until the client disconnects (or the transport is closed)."""
        await self._disconnected.wait()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        # audio is moot now, but flush control messages (errors, final transcripts)
        self._drop_queued_audio()
        self._outbox.close()
        if self._writer is not None and not self._disconnected.is_set():
            await asyncio.wait({self._writer}, timeout=self.flush_timeout)
        await cancel_and_wait(self._writer)
        if self.websocket is not None:
            with contextlib.suppress(Exception):
                await self.websocket.close(self.close_code, _truncate_reason(self.close_reason))
        await cancel_and_wait(self._reader)
        await self._tasks.cancel_all()
        if self._pending is not None:
            self._pending.put_nowait(None)  # wake a start() still waiting for a client
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
        self._on_disconnected()

    # ------------------------------------------------------------------ audio API
    def audio_input(self) -> AsyncIterator[AudioFrame]:
        return self._input.__aiter__()

    async def write_audio(self, frame: AudioFrame) -> None:
        """Queue agent audio for the client, split into ``frame_duration`` messages.

        Never blocks: a writer task sends queued messages in order, so :meth:`clear_audio`
        can still drop audio that has not reached the socket yet.
        """
        if not self.connected or self._closed:
            return
        out = self._out_resampler.push(frame)  # no-op when already in output_format
        if not out:
            return
        bps = out.format.bytes_per_sample
        step = max(1, round(self.frame_duration * out.sample_rate)) * bps
        data = out.data
        for i in range(0, len(data), step):
            piece = data[i : i + step]
            payload: str | bytes = piece
            if self.framing == "base64":
                payload = _dumps({"type": "audio", "data": base64.b64encode(piece).decode()})
            self._outbox.send_nowait(_Outgoing(payload, len(piece) // bps))
        samples = len(data) // bps
        self._out_samples += samples
        t = now()
        self._play_end = max(self._play_end, t) + samples / out.sample_rate

    async def clear_audio(self) -> None:
        """Drop queued agent audio and tell the client to drop what it buffered (``clear``)."""
        if not self._ready or self._closed:
            return
        self._drop_queued_audio()
        self._clear_floor = self._out_samples
        self._play_end = now()
        self._out_resampler = StreamResampler(self.output_format.sample_rate, 1)
        self._send_json({"type": "clear"})

    def buffered_duration(self) -> float:
        """Seconds of agent audio sent/queued but not yet played by the client.

        Re-anchored on every ``playback`` report from the client, extrapolated in real
        time in between (and purely wall-clock based for clients that never report).
        """
        if not self._ready:
            return 0.0
        return max(0.0, self._play_end - now())

    # --------------------------------------------------------------- messages API
    async def send_message(self, message: dict[str, Any]) -> None:
        self.send_message_nowait(message)

    def send_message_nowait(self, message: dict[str, Any]) -> None:
        """Queue a JSON message for the client (dropped before ``ready`` / after close)."""
        if not isinstance(message, dict) or not isinstance(message.get("type"), str):
            raise ValueError("messages must be dicts with a string 'type'")
        if not self._ready:
            logger.debug("dropping %r message sent before the handshake", message["type"])
            return
        self._send_json(message)

    # -------------------------------------------------------------------- internals
    def _send_json(self, message: dict[str, Any]) -> None:
        if self._outbox.closed or self._disconnected.is_set():
            return
        self._outbox.send_nowait(_Outgoing(_dumps(message)))

    def _drop_queued_audio(self) -> None:
        dropped = 0
        kept: list[_Outgoing] = []
        for item in self._outbox.clear():
            if item.samples:
                dropped += item.samples
            else:
                kept.append(item)
        for item in kept:
            self._outbox.send_nowait(item)
        self._out_samples -= dropped

    def _on_disconnected(self) -> None:
        if self._disconnected.is_set():
            return
        self._disconnected.set()
        self._input.close()
        self._outbox.clear()
        self.emit("disconnected")

    async def _accept(self, websocket: ServerConnection) -> None:
        """Standalone-mode connection handler: queue the client, keep it open while in use."""
        if self._closed or self._ready or self._pending is None:
            await _refuse(
                websocket, CLOSE_TRY_AGAIN_LATER, "server_busy", "another client is connected"
            )
            return
        self._pending.put_nowait(websocket)
        await websocket.wait_closed()

    async def _accept_first_client(self) -> None:
        await self.listen()
        assert self._pending is not None
        while True:
            websocket = await self._pending.get()
            if websocket is None:
                raise TransportError("transport closed while waiting for a client")
            try:
                await self._handshake(websocket)
            except TransportError as exc:
                logger.info("WebSocket client rejected: %s", exc)
                continue
            break
        while not self._pending.empty():  # clients that queued up during the handshake
            other = self._pending.get_nowait()
            if other is not None:
                self._tasks.spawn(
                    _refuse(
                        other, CLOSE_TRY_AGAIN_LATER, "server_busy", "another client is connected"
                    )
                )

    async def _handshake(self, websocket: ServerConnection) -> None:
        error: _HandshakeError | None = None
        try:
            async with asyncio.timeout(self.hello_timeout):
                message = await websocket.recv()
        except TimeoutError:
            error = _HandshakeError("bad_hello", f"no `hello` within {self.hello_timeout:g} s")
        except ConnectionClosed:
            raise TransportError("client disconnected before `hello`") from None
        else:
            try:
                self._apply_hello(message)
            except _HandshakeError as exc:
                error = exc
        if error is not None:
            await _refuse(websocket, error.close_code, error.code, str(error))
            raise error
        self.websocket = websocket
        self._out_resampler = StreamResampler(self.output_format.sample_rate, 1)
        self._ready = True
        self._send_json(
            {
                "type": "ready",
                "protocol": PROTOCOL,
                "session_id": self.session_id,
                "codec": CODEC,
                "sample_rate": self.input_format.sample_rate,
                "channels": self.input_format.channels,
                "output_sample_rate": self.output_format.sample_rate,
                "output_channels": self.output_format.channels,
                "framing": self.framing,
                "frame_ms": round(self.frame_duration * 1000),
            }
        )

    def _apply_hello(self, message: str | bytes) -> None:
        if not isinstance(message, str):
            raise _HandshakeError("bad_hello", "the first message must be a JSON `hello`")
        try:
            hello = json.loads(message)
        except ValueError:
            raise _HandshakeError("bad_hello", "`hello` is not valid JSON") from None
        if not isinstance(hello, dict) or hello.get("type") != "hello":
            raise _HandshakeError("bad_hello", 'the first message must be {"type": "hello", ...}')
        protocol = hello.get("protocol", PROTOCOL)
        if protocol != PROTOCOL:
            raise _HandshakeError(
                "unsupported_protocol", f"unsupported protocol {protocol!r}; use {PROTOCOL!r}"
            )
        codec = hello.get("codec", CODEC)
        if not isinstance(codec, str) or codec.lower() not in _CODECS:
            raise _HandshakeError(
                "unsupported_codec", f"unsupported codec {codec!r}; use {CODEC!r}"
            )
        framing = hello.get("framing", "binary")
        if not isinstance(framing, str) or framing not in _FRAMINGS:
            raise _HandshakeError("bad_hello", f"`framing` must be one of {sorted(_FRAMINGS)}")
        rate = _int_field(hello, "sample_rate", self.input_format.sample_rate, _MIN_RATE, _MAX_RATE)
        channels = _int_field(hello, "channels", 1, 1, 2)
        out_rate = _int_field(
            hello, "output_sample_rate", self.output_format.sample_rate, _MIN_RATE, _MAX_RATE
        )
        self.hello = hello
        self.framing = str(framing)
        self.input_format = AudioFormat(rate, channels)
        self.output_format = AudioFormat(out_rate, 1)

    async def _read_loop(self) -> None:
        websocket = self.websocket
        assert websocket is not None
        try:
            async for message in websocket:
                if isinstance(message, str):
                    self._on_text(message)
                else:
                    self._on_audio(bytes(message))
        except ConnectionClosed:
            pass
        except Exception:
            logger.exception("WebSocket reader failed (%s)", self.session_id)
        finally:
            logger.info(
                "WebSocket client disconnected (%s, code %s)", self.session_id, websocket.close_code
            )
            self._on_disconnected()

    async def _write_loop(self) -> None:
        websocket = self.websocket
        assert websocket is not None
        try:
            async for item in self._outbox:
                await websocket.send(item.payload)
        except ConnectionClosed:
            pass
        except Exception:
            logger.exception("WebSocket writer failed (%s)", self.session_id)

    def _on_audio(self, pcm: bytes) -> None:
        data = self._in_rest + pcm if self._in_rest else pcm
        cut = len(data) - len(data) % self.input_format.bytes_per_sample
        self._in_rest = data[cut:]
        if not cut or self._input.closed:
            return
        fmt = self.input_format
        frame = AudioFrame(data[:cut], fmt.sample_rate, fmt.channels)
        frame.timestamp = now() - frame.duration  # fully captured by the time it arrived
        self._input.send_nowait(frame)

    def _on_text(self, text: str) -> None:
        try:
            message = json.loads(text)
        except ValueError:
            self._invalid("messages must be valid JSON")
            return
        if not isinstance(message, dict) or not isinstance(message.get("type"), str):
            self._invalid("messages must be JSON objects with a string `type`")
            return
        kind = message["type"]
        if kind == "audio":
            data = message.get("data")
            try:
                pcm = base64.b64decode(data, validate=True) if isinstance(data, str) else None
            except (binascii.Error, ValueError):
                pcm = None
            if pcm is None:
                self._invalid("`audio.data` must be base64-encoded PCM")
                return
            self._on_audio(pcm)
        elif kind in ("playback", "mark"):
            self._on_playback(message)
        elif kind == "hello":
            self._invalid("duplicate `hello`")
        else:
            self.emit("message", message)

    def _on_playback(self, message: dict[str, Any]) -> None:
        value = message.get("position_ms", message.get("played_ms"))
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or value < 0
        ):
            self._invalid("`playback.position_ms` must be a non-negative number")
            return
        rate = self.output_format.sample_rate
        position = round(value * rate / 1000)
        if position < self._clear_floor:
            return  # sent before the client processed our last `clear`
        self._play_end = now() + max(0, self._out_samples - position) / rate

    def _invalid(self, reason: str) -> None:
        logger.debug("invalid message from WebSocket client (%s): %s", self.session_id, reason)
        self._send_json({"type": "error", "code": "invalid_message", "message": reason})


# ------------------------------------------------------------------------------ bridge
@dataclass(slots=True)
class _AgentItem:
    response_id: str
    text: str = ""


class SessionBridge:
    """Connects an :class:`~voice_agent_next.session.AgentSession` to a client's data channel.

    Session events become ``van-ws/1`` messages:

    * ``user_transcript`` -> ``transcript`` (``role: "user"``, interim and final);
    * ``agent_transcript`` -> ``transcript`` (``role: "assistant"``, streamed with ``delta``;
      a ``final`` message follows once the item is complete, or ``interrupted: true`` with
      only the text the user actually heard after a barge-in);
    * ``agent_state_changed`` / ``user_state_changed`` -> ``state``;
    * ``metrics`` -> ``metrics``; ``error`` -> ``error``;

    and the client's ``text`` messages are answered with
    :meth:`~voice_agent_next.session.AgentSession.generate_reply`.

    :func:`serve_websocket` attaches one per connection. With a standalone transport::

        transport = create_transport({"type": "websocket", "port": 8765})
        bridge = SessionBridge(session, transport)
        try:
            await session.run(agent, transport)
        finally:
            await bridge.aclose()
    """

    def __init__(
        self,
        session: AgentSession,
        transport: Transport,
        *,
        metrics: bool = True,
        errors: bool = True,
        text_input: bool = True,
    ) -> None:
        self.session = session
        self.transport = transport
        self.text_input = text_input
        self._items: dict[str, _AgentItem] = {}
        self._tasks = BackgroundTasks("ws-bridge")
        self._started = asyncio.Event()
        handlers: list[tuple[Any, str, Callable[..., Any]]] = [
            (session, "user_transcript", self._on_user_transcript),
            (session, "agent_transcript", self._on_agent_transcript),
            (session, "interrupted", self._on_interrupted),
            (session, "agent_state_changed", self._on_state),
            (session, "user_state_changed", self._on_state),
            (transport, "message", self._on_client_message),
        ]
        if metrics:
            handlers.append((session, "metrics", self._on_metrics))
        if errors:
            handlers.append((session, "error", self._on_error))
        for emitter, event, handler in handlers:
            emitter.on(event, handler)
        self._handlers = handlers
        if session.agent_state != AgentState.INITIALIZING:
            self._started.set()
            self._send_state()

    async def aclose(self) -> None:
        """Detach from the session and transport."""
        for emitter, event, handler in self._handlers:
            emitter.off(event, handler)
        self._handlers = []
        await self._tasks.cancel_all()

    # ------------------------------------------------------------------ outgoing
    def _send(self, message: dict[str, Any]) -> None:
        send_nowait = getattr(self.transport, "send_message_nowait", None)
        if send_nowait is not None:
            send_nowait(message)
        else:  # tasks start in FIFO order, so messages keep their order
            self._tasks.spawn(self.transport.send_message(message), name="bridge-send")

    def _send_state(self) -> None:
        self._send(
            {
                "type": "state",
                "agent": self.session.agent_state.value,
                "user": self.session.user_state.value,
            }
        )

    def _on_user_transcript(self, ev: UserTranscript) -> None:
        message: dict[str, Any] = {
            "type": "transcript",
            "role": "user",
            "item_id": ev.item_id,
            "text": ev.text,
            "final": ev.is_final,
        }
        if ev.language:
            message["language"] = ev.language
        self._send(message)

    def _on_agent_transcript(self, ev: AgentTranscript) -> None:
        for item_id in [i for i in self._items if i != ev.item_id]:
            self._finalize(item_id)  # a new item: the previous ones are complete
        item = self._items.setdefault(ev.item_id, _AgentItem(ev.response_id))
        item.text += ev.delta
        self._send(
            {
                "type": "transcript",
                "role": "assistant",
                "item_id": ev.item_id,
                "response_id": ev.response_id,
                "delta": ev.delta,
                "text": item.text,
                "final": False,
            }
        )

    def _finalize(self, item_id: str, interrupted: Interrupted | None = None) -> None:
        item = self._items.pop(item_id, None)
        if item is None:
            return
        text = item.text.strip()
        heard = self.session.history.get(item_id)
        if heard is not None and getattr(heard, "role", None) == "assistant":
            text = getattr(heard, "text", text)  # truncated to what was heard on barge-in
        message: dict[str, Any] = {
            "type": "transcript",
            "role": "assistant",
            "item_id": item_id,
            "response_id": item.response_id,
            "text": text,
            "final": True,
        }
        if interrupted is not None:
            message["interrupted"] = True
            message["played_ms"] = round(interrupted.played * 1000)
        self._send(message)

    def _on_interrupted(self, ev: Interrupted) -> None:
        if ev.item_id is not None:
            self._finalize(ev.item_id, interrupted=ev)

    def _on_state(self, ev: AgentStateChanged | UserStateChanged) -> None:
        if self.session.agent_state != AgentState.INITIALIZING:
            self._started.set()
        if self.session.agent_state in (AgentState.LISTENING, AgentState.CLOSED):
            for item_id in list(self._items):
                self._finalize(item_id)
        self._send_state()

    def _on_metrics(self, m: Metrics) -> None:
        data = metrics_to_dict(m)
        kind = data.pop("type", "unknown")
        self._send({"type": "metrics", "kind": kind, "data": data})

    def _on_error(self, ev: SessionError) -> None:
        self._send(
            {
                "type": "error",
                "code": "session_error",
                "message": _describe(ev.error),
                "fatal": not ev.recoverable,
            }
        )

    # ------------------------------------------------------------------ incoming
    def _on_client_message(self, message: dict[str, Any]) -> None:
        if message.get("type") != "text" or not self.text_input:
            return
        text = message.get("text")
        if not isinstance(text, str) or not text.strip():
            self._send(
                {"type": "error", "code": "invalid_message", "message": "`text.text` is empty"}
            )
            return
        self._tasks.spawn(self._reply(text.strip()), name="bridge-text")

    async def _reply(self, text: str) -> None:
        await self._started.wait()  # the session may still be connecting its engine
        if self.session.closed:
            return
        try:
            await self.session.generate_reply(user_input=text)
        except Exception as exc:
            logger.warning("typed input failed: %s", exc)
            self._send({"type": "error", "code": "text_failed", "message": _describe(exc)})


# ------------------------------------------------------------------------------ server
class WebSocketAgentServer:
    """A WebSocket server that runs one :class:`AgentSession` per connection.

    For every client it performs the ``van-ws/1`` handshake, builds a session and an agent
    with the factories, attaches a :class:`SessionBridge` (unless ``forward_events=False``)
    and runs the session until either side hangs up. Closing the connection closes the
    session, and closing the session closes the connection.

    Factories are called once per connection, either without arguments or — when they
    take a required positional argument — with the connection's
    :class:`WebSocketServerTransport` (see its ``hello``, ``path`` and ``session_id``).
    They may be coroutine functions.

    Args:
        session_factory: builds the :class:`AgentSession` for a connection.
        agent_factory: builds the :class:`Agent` for a connection.
        host / port: listening address (``port=0`` picks a free port; see :attr:`port`).
        max_sessions: refuse clients (close code 1013) beyond this many live sessions.
        forward_events: send transcripts, state changes, metrics and errors to clients.
        input_sample_rate / output_sample_rate / frame_duration / hello_timeout: per
            connection transport options (see :class:`WebSocketServerTransport`).
        serve_options: extra ``websockets.asyncio.server.serve`` arguments, e.g. ``ssl``
            (TLS), ``origins`` (browser origin allow-list) or ``process_request`` (HTTP
            routes, authentication).
    """

    def __init__(
        self,
        session_factory: SessionFactory,
        agent_factory: AgentFactory,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        input_sample_rate: int = 16_000,
        output_sample_rate: int = 24_000,
        frame_duration: float = 0.02,
        hello_timeout: float = 10.0,
        max_sessions: int | None = None,
        forward_events: bool = True,
        **serve_options: Any,
    ) -> None:
        self.session_factory = session_factory
        self.agent_factory = agent_factory
        self.host = host
        self.port = port
        self.max_sessions = max_sessions
        self.forward_events = forward_events
        self.serve_options = serve_options
        self._transport_options: dict[str, Any] = {
            "input_sample_rate": input_sample_rate,
            "output_sample_rate": output_sample_rate,
            "frame_duration": frame_duration,
            "hello_timeout": hello_timeout,
        }
        self._server: Server | None = None
        self._sessions: set[AgentSession] = set()
        self._active = 0
        self._closed = asyncio.Event()

    @property
    def url(self) -> str:
        scheme = "wss" if self.serve_options.get("ssl") is not None else "ws"
        return f"{scheme}://{_url_host(self.host)}:{self.port}/"

    @property
    def sessions(self) -> list[AgentSession]:
        """Sessions currently running."""
        return list(self._sessions)

    async def start(self) -> None:
        """Start listening (idempotent). :attr:`port` is then the bound port."""
        if self._server is not None:
            return
        options: dict[str, Any] = {"compression": None, **self.serve_options}
        self._server = await serve(self._handle, self.host, self.port, **options)
        self.port = _bound_port(self._server, self.port)
        logger.info("serving voice agents on %s (%s)", self.url, PROTOCOL)

    async def serve_forever(self) -> None:
        """Serve until :meth:`aclose` is called or this coroutine is cancelled."""
        await self.start()
        try:
            await self._closed.wait()
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        """Stop accepting clients, close every connection (1001) and wait for their sessions."""
        server = self._server
        if server is not None:
            server.close()
            await server.wait_closed()
        self._closed.set()

    async def __aenter__(self) -> WebSocketAgentServer:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _handle(self, websocket: ServerConnection) -> None:
        if self.max_sessions is not None and self._active >= self.max_sessions:
            await _refuse(websocket, CLOSE_TRY_AGAIN_LATER, "server_busy", "too many sessions")
            return
        self._active += 1
        try:
            await self._run(websocket)
        finally:
            self._active -= 1

    async def _run(self, websocket: ServerConnection) -> None:
        transport = WebSocketServerTransport(websocket, **self._transport_options)
        try:
            await transport.start()
        except TransportError as exc:
            logger.info("WebSocket handshake with %s failed: %s", websocket.remote_address, exc)
            return
        session: AgentSession | None = None
        bridge: SessionBridge | None = None
        reason = "user_disconnected"
        try:
            session = await _call_factory(self.session_factory, transport)
            agent = await _call_factory(self.agent_factory, transport)
            self._sessions.add(session)
            if self.forward_events:
                bridge = SessionBridge(session, transport)
            await session.start(agent, transport)
            await wait_first(session.wait_closed(), transport.wait_disconnected())
        except Exception as exc:
            logger.exception("WebSocket session %s failed", transport.session_id)
            reason = "error"
            transport.send_message_nowait(
                {
                    "type": "error",
                    "code": "internal_error",
                    "message": _describe(exc),
                    "fatal": True,
                }
            )
            transport.close_code, transport.close_reason = CLOSE_INTERNAL_ERROR, "internal error"
        finally:
            if session is not None:
                with contextlib.suppress(Exception):
                    await session.aclose(reason)  # no-op if the session already closed
                self._sessions.discard(session)
            if bridge is not None:
                await bridge.aclose()
            await transport.aclose()


async def serve_websocket(
    session_factory: SessionFactory,
    agent_factory: AgentFactory,
    host: str = "127.0.0.1",
    port: int = 8765,
    **options: Any,
) -> WebSocketAgentServer:
    """Start a :class:`WebSocketAgentServer` (one session per connection) and return it.

    ``options`` are the keyword arguments of :class:`WebSocketAgentServer` (``max_sessions``,
    ``output_sample_rate``, ``ssl``, ``origins``, ``process_request``...)::

        server = await serve_websocket(
            lambda: AgentSession("openai/gpt-realtime"),
            lambda: Agent("You are a helpful assistant."),
            host="0.0.0.0",
            port=8765,
        )
        await server.serve_forever()  # Ctrl-C closes every session cleanly
    """
    server = WebSocketAgentServer(session_factory, agent_factory, host=host, port=port, **options)
    await server.start()
    return server


# ----------------------------------------------------------------------------- helpers
def _dumps(message: dict[str, Any]) -> str:
    try:
        return json.dumps(message, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):  # non-JSON values: stringify them, NaN/inf -> null
        return json.dumps(_json_safe(message), separators=(",", ":"), ensure_ascii=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, str | int | bool):
        return value
    return str(value)


def _int_field(message: dict[str, Any], key: str, default: int, low: int, high: int) -> int:
    value = message.get(key, default)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise _HandshakeError("bad_hello", f"`{key}` must be an integer in [{low}, {high}]")
    return value


def _describe(exc: BaseException) -> str:
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _truncate_reason(reason: str) -> str:
    """Close reasons are limited to 123 bytes of UTF-8."""
    return reason.encode()[:123].decode(errors="ignore")


def _url_host(host: str | None) -> str:
    if not host:
        return "localhost"
    return f"[{host}]" if ":" in host else host


def _bound_port(server: Server, default: int) -> int:
    for sock in server.sockets:
        return int(sock.getsockname()[1])
    return default


async def _refuse(websocket: ServerConnection, close_code: int, code: str, message: str) -> None:
    """Tell the client why (``error``) and close the connection."""
    with contextlib.suppress(Exception):
        await websocket.send(
            _dumps({"type": "error", "code": code, "message": message, "fatal": True})
        )
        await websocket.close(close_code, _truncate_reason(message))


def _wants_transport(factory: Callable[..., Any]) -> bool:
    try:
        params = inspect.signature(factory).parameters.values()
    except (TypeError, ValueError):
        return False
    positional = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    return any(p.kind in positional and p.default is inspect.Parameter.empty for p in params)


async def _call_factory(factory: Callable[..., Any], transport: WebSocketServerTransport) -> Any:
    result = factory(transport) if _wants_transport(factory) else factory()
    if inspect.isawaitable(result):
        result = await result
    return result
