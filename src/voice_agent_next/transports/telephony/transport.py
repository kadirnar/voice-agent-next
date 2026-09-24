"""Telephony media-stream transports: a WebSocket server speaking a provider's dialect."""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from dataclasses import dataclass
from typing import Any, ClassVar

import httpx
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from ...audio.frame import AudioFormat, AudioFrame
from ...audio.resample import StreamResampler
from ...errors import TransportError
from ...utils.clock import now
from ...utils.ids import new_id
from ...utils.log import logger
from ..base import TransportCapabilities
from ..websocket import (
    CLOSE_PROTOCOL_ERROR,
    AgentFactory,
    SessionFactory,
    WebSocketAgentServer,
    WebSocketServerTransport,
    _dumps,
    _Outgoing,
    _truncate_reason,
)
from .serializers import (
    AudioCleared,
    AudioReceived,
    CallInfo,
    DtmfReceived,
    MarkReached,
    ProviderError,
    StreamStarted,
    StreamStopped,
    TelephonyEvent,
    TelephonyProtocolError,
    TelephonySerializer,
    create_serializer,
)

__all__ = [
    "PlivoTransport",
    "TelephonyServer",
    "TelephonyTransport",
    "TelnyxTransport",
    "TwilioTransport",
    "VonageTransport",
    "serve_telephony",
]


@dataclass(slots=True)
class _MarkOut(_Outgoing):
    """A queued mark/checkpoint message (dropped together with the audio on ``clear``)."""


@dataclass(slots=True)
class _PendingMark:
    name: str
    position: int
    """``_out_samples`` when the mark was sent: playback reaches it at this sample."""
    due: float
    """Wall-clock estimate of when playback reaches it."""


class TelephonyTransport(WebSocketServerTransport):
    """One phone call's media stream (Twilio, Telnyx, Vonage or Plivo) over a WebSocket.

    The provider connects to us: point its stream URL (TwiML ``<Connect><Stream>``, Telnyx
    ``stream_url``, a Vonage NCCO ``websocket`` endpoint, Plivo ``<Stream>``) at this
    server. Like :class:`~voice_agent_next.transports.websocket.WebSocketServerTransport`
    it works per connection (see :class:`TelephonyServer`, one session per call) or
    standalone (``create_transport({"type": "twilio", "port": 8765})`` serves one call).

    * :meth:`start` waits for the provider's start message; the call metadata is then in
      :attr:`call` and the audio formats follow the stream (μ-law/A-law 8 kHz or L16);
    * caller audio is decoded to s16le frames; agent audio is encoded and sent in
      ``frame_duration`` messages, each batch followed by a *mark* (Twilio/Telnyx ``mark``,
      Vonage ``notify``, Plivo ``checkpoint``) at most every ``mark_interval`` seconds.
      When the provider echoes a mark, playback has reached it: :meth:`buffered_duration`
      is anchored on those echoes and extrapolated in real time in between, never past
      the next unacknowledged mark, so barge-in truncation matches what the caller heard;
    * :meth:`clear_audio` sends the provider's clear message;
    * keypad presses are emitted as ``"dtmf"`` events with the digit (``"0"``-``"9"``,
      ``"*"``, ``"#"``, ``"A"``-``"D"``); ``"call_started"`` carries :class:`CallInfo`;
    * the provider's stop message (the caller hung up) ends :meth:`audio_input`;
    * closing the transport closes the stream and, with ``hangup_on_close`` and REST
      credentials (Twilio, Telnyx, Plivo), hangs up the call.

    Args:
        websocket: an accepted provider connection (per-connection mode) or ``None``.
        provider: ``"twilio"``, ``"telnyx"``, ``"vonage"``, ``"plivo"`` or a
            :class:`~.serializers.TelephonySerializer` instance.
        host / port / serve_options: listening address in standalone mode.
        frame_duration: duration of each outbound media message (seconds).
        mark_interval: minimum audio between two marks (seconds).
        mark_grace: stop waiting for a mark this long after it was due (a provider that
            never echoes marks degrades to the wall-clock estimate).
        start_timeout: seconds to wait for the provider's start message.
        hangup_on_close: hang up the call via the provider's REST API when the transport
            closes before the caller hung up (needs credentials; see the serializers).
        http_client: client for the REST hang-up (default: a short-lived one).
        **serializer_options: credentials and codec options of the provider's serializer.
    """

    capabilities = TransportCapabilities(playback_position=True, dtmf=True)
    default_provider: ClassVar[str] = "twilio"

    def __init__(
        self,
        websocket: ServerConnection | None = None,
        *,
        provider: str | TelephonySerializer | None = None,
        host: str = "127.0.0.1",
        port: int = 8765,
        frame_duration: float = 0.02,
        mark_interval: float = 0.1,
        mark_grace: float = 1.0,
        start_timeout: float = 10.0,
        flush_timeout: float = 1.0,
        hangup_on_close: bool = True,
        http_client: httpx.AsyncClient | None = None,
        serve_options: dict[str, Any] | None = None,
        **serializer_options: Any,
    ) -> None:
        provider = provider if provider is not None else self.default_provider
        if isinstance(provider, TelephonySerializer):
            if serializer_options:
                raise TypeError("serializer options need a provider name, not an instance")
            serializer = provider
        else:
            serializer = create_serializer(provider, **serializer_options)
        rate = serializer.input_codec.sample_rate
        super().__init__(
            websocket,
            host=host,
            port=port,
            input_sample_rate=rate,
            output_sample_rate=serializer.output_codec.sample_rate,
            frame_duration=frame_duration,
            hello_timeout=start_timeout,
            flush_timeout=flush_timeout,
            serve_options=serve_options,
        )
        if mark_interval < 0 or mark_grace < 0:
            raise ValueError("mark_interval and mark_grace must be >= 0")
        self.serializer = serializer
        self.provider = serializer.provider
        self.mark_interval = mark_interval
        self.mark_grace = mark_grace
        self.hangup_on_close = hangup_on_close
        self.http_client = http_client
        self.session_id = new_id("call_")
        self.framing = "json" if serializer.json_audio else "binary"
        self.stopped = False
        """The provider ended the stream (caller hung up / call redirected)."""
        self._hung_up = False
        self._pending_marks: deque[_PendingMark] = deque()
        self._mark_seq = 0
        self._unmarked = 0
        self._anchor_pos = 0
        self._anchor_t = 0.0

    # ------------------------------------------------------------------ properties
    @property
    def call(self) -> CallInfo | None:
        """Call metadata from the provider's start message (``None`` before :meth:`start`)."""
        return self.serializer.call

    # ------------------------------------------------------------------- lifecycle
    async def hangup(self) -> bool:
        """Hang up the call via the provider's REST API; ``False`` if it is not possible
        (no credentials, not supported, already ended or the request failed)."""
        if self._hung_up or self.stopped:
            return False
        request = self.serializer.hangup_request()
        if request is None:
            return False
        self._hung_up = True
        try:
            if self.http_client is not None:
                response = await self.http_client.send(request)
            else:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.send(request)
        except httpx.HTTPError as exc:
            logger.warning("%s hang-up failed: %s", self.provider, exc)
            return False
        if response.status_code >= 400 and response.status_code != 404:  # 404: already over
            logger.warning(
                "%s hang-up failed: HTTP %s %s",
                self.provider, response.status_code, response.text[:200],
            )  # fmt: skip
            return False
        logger.info("%s call %s hung up", self.provider, self.call.call_id if self.call else "?")
        return True

    async def aclose(self) -> None:
        if self._closed:
            return
        if self.hangup_on_close and self._ready and not self.stopped:
            with contextlib.suppress(Exception):
                await self.hangup()
        await super().aclose()

    async def _handshake(self, websocket: ServerConnection) -> None:
        early: list[TelephonyEvent] = []
        try:
            async with asyncio.timeout(self.hello_timeout):
                while True:
                    message = await websocket.recv()
                    events = self.serializer.parse(message)
                    started = [e for e in events if isinstance(e, StreamStarted)]
                    if started:
                        early.extend(events[events.index(started[0]) + 1 :])
                        break
                    if any(isinstance(e, StreamStopped) for e in events):
                        raise TelephonyProtocolError("stream stopped before it started")
        except TimeoutError:
            await _close(websocket, CLOSE_PROTOCOL_ERROR, "no start message")
            raise TransportError(
                f"{self.provider}: no start message within {self.hello_timeout:g} s"
            ) from None
        except ConnectionClosed:
            raise TransportError(
                f"{self.provider} disconnected before the stream started"
            ) from None
        except TelephonyProtocolError as exc:
            await _close(websocket, CLOSE_PROTOCOL_ERROR, str(exc))
            raise TransportError(f"{self.provider}: {exc}") from None
        self.websocket = websocket
        codec_in, codec_out = self.serializer.input_codec, self.serializer.output_codec
        self.input_format = AudioFormat(codec_in.sample_rate, 1)
        self.output_format = AudioFormat(codec_out.sample_rate, 1)
        self._out_resampler = StreamResampler(codec_out.sample_rate, 1)
        self._anchor_t = now()
        self._ready = True
        self.emit("call_started", self.call)
        for ev in early:
            self._handle(ev)

    # ------------------------------------------------------------------ audio API
    async def write_audio(self, frame: AudioFrame) -> None:
        """Encode agent audio and queue it in ``frame_duration`` messages, plus a mark."""
        if not self.connected or self._closed:
            return
        out = self._out_resampler.push(frame)
        if not out:
            return
        rate = out.sample_rate
        step = max(1, round(self.frame_duration * rate)) * 2
        data = out.data
        if self.serializer.fixed_frames and len(data) % step:
            data += bytes(step - len(data) % step)  # whole frames only: pad with silence
        t = now()
        if self._played(t) >= self._out_samples:  # idle: playback (re)starts now
            self._anchor_pos, self._anchor_t = self._out_samples, t
        for i in range(0, len(data), step):
            piece = data[i : i + step]
            self._outbox.send_nowait(
                _Outgoing(self.serializer.encode_audio(piece), len(piece) // 2)
            )
        samples = len(data) // 2
        self._out_samples += samples
        self._unmarked += samples
        self._play_end = max(self._play_end, t) + samples / rate
        if self._unmarked >= self.mark_interval * rate:
            self._send_mark()

    async def clear_audio(self) -> None:
        """Drop queued agent audio and send the provider's clear message."""
        if not self._ready or self._closed:
            return
        self._drop_queued_audio()
        # whatever the provider still buffered is discarded too: nothing is left to play
        # (the marks it flushes back carry names we no longer wait for)
        self._pending_marks.clear()
        self._unmarked = 0
        self._anchor_pos, self._anchor_t = self._out_samples, now()
        self._clear_floor = self._out_samples
        self._play_end = now()
        self._out_resampler = StreamResampler(self.output_format.sample_rate, 1)
        self._send_raw(self.serializer.encode_clear())

    def buffered_duration(self) -> float:
        """Seconds of agent audio sent but not yet played to the caller (marks-based)."""
        if not self._ready:
            return 0.0
        return max(0.0, self._out_samples - self._played(now())) / self.output_format.sample_rate

    # --------------------------------------------------------------- messages API
    def send_message_nowait(self, message: dict[str, Any]) -> None:
        """Phone calls have no data channel: app messages are dropped (see :meth:`send_raw`)."""
        logger.debug("%s transport: dropping %r message", self.provider, message.get("type"))

    async def send_raw(self, message: dict[str, Any] | str | bytes) -> None:
        """Send a provider-specific message verbatim (a dict is JSON-encoded), e.g. Plivo's
        ``{"event": "sendDTMF", "dtmf": "1234#"}``."""
        if isinstance(message, dict):
            message = _dumps(message)
        if self._ready:
            self._send_raw(message)

    # -------------------------------------------------------------------- internals
    def _send_raw(self, payload: str | bytes) -> None:
        if self._outbox.closed or self._disconnected.is_set():
            return
        self._outbox.send_nowait(_Outgoing(payload))

    def _send_mark(self) -> None:
        self._mark_seq += 1
        name = f"van-{self._mark_seq}"
        self._outbox.send_nowait(_MarkOut(self.serializer.encode_mark(name)))
        self._pending_marks.append(_PendingMark(name, self._out_samples, self._play_end))
        self._unmarked = 0

    def _played(self, t: float) -> float:
        """Samples the caller has heard by time ``t`` (anchored on marks, capped by them)."""
        rate = self.output_format.sample_rate
        position = self._anchor_pos + max(0.0, t - self._anchor_t) * rate
        pending = self._pending_marks
        while pending and t > pending[0].due + self.mark_grace:
            pending.popleft()  # never echoed: stop trusting it (wall clock takes over)
        if pending:
            position = min(position, pending[0].position)
        return min(position, self._out_samples)

    def _drop_queued_audio(self) -> None:
        dropped = 0
        kept: list[_Outgoing] = []
        for item in self._outbox.clear():
            if item.samples:
                dropped += item.samples
            elif not isinstance(item, _MarkOut):
                kept.append(item)
        for item in kept:
            self._outbox.send_nowait(item)
        self._out_samples -= dropped

    def _on_text(self, text: str) -> None:
        self._on_wire(text)

    def _on_audio(self, pcm: bytes) -> None:
        self._on_wire(pcm)

    def _on_wire(self, message: str | bytes) -> None:
        try:
            events = self.serializer.parse(message)
        except TelephonyProtocolError as exc:
            logger.debug("ignoring %s message: %s", self.provider, exc)
            return
        for ev in events:
            self._handle(ev)

    def _handle(self, ev: TelephonyEvent) -> None:
        if isinstance(ev, AudioReceived):
            super()._on_audio(ev.pcm)
        elif isinstance(ev, MarkReached):
            self._on_mark(ev.name)
        elif isinstance(ev, DtmfReceived):
            logger.debug("%s DTMF %r", self.provider, ev.digit)
            self.emit("dtmf", ev.digit)
        elif isinstance(ev, StreamStopped):
            logger.info("%s stream stopped (%s)", self.provider, self.session_id)
            self.stopped = True
            self._input.close()  # the caller is gone: the session ends
        elif isinstance(ev, ProviderError):
            logger.warning("%s error: %s", self.provider, ev.message)
        elif isinstance(ev, AudioCleared):
            logger.debug("%s cleared its audio buffer", self.provider)

    def _on_mark(self, name: str) -> None:
        pending = self._pending_marks
        mark = next((m for m in pending if m.name == name), None)
        if mark is None:
            return  # flushed by a clear, given up on, or not ours
        while pending.popleft() is not mark:
            pass  # marks come back in order: the earlier ones were reached too
        t = now()
        self._anchor_pos, self._anchor_t = mark.position, t
        rate = self.output_format.sample_rate
        self._play_end = t + max(0, self._out_samples - mark.position) / rate


class TwilioTransport(TelephonyTransport):
    """Twilio Media Streams (see :class:`TelephonyTransport`)."""

    default_provider = "twilio"


class TelnyxTransport(TelephonyTransport):
    """Telnyx media streaming (see :class:`TelephonyTransport`)."""

    default_provider = "telnyx"


class VonageTransport(TelephonyTransport):
    """Vonage Voice API WebSockets (see :class:`TelephonyTransport`)."""

    default_provider = "vonage"


class PlivoTransport(TelephonyTransport):
    """Plivo audio streams (see :class:`TelephonyTransport`)."""

    default_provider = "plivo"


# ------------------------------------------------------------------------------ server
class TelephonyServer(WebSocketAgentServer):
    """A WebSocket server that answers each phone call with its own :class:`AgentSession`.

    Point the provider's media stream at :attr:`url` (behind TLS in production). Factories
    get the call's :class:`TelephonyTransport` when they take an argument, so they can
    read ``transport.call`` (caller number, custom parameters...).

    Args:
        session_factory / agent_factory: see
            :class:`~voice_agent_next.transports.websocket.WebSocketAgentServer`.
        provider: ``"twilio"``, ``"telnyx"``, ``"vonage"`` or ``"plivo"``.
        serializer_options: provider options (credentials, codecs) for every call.
        transport_options: other :class:`TelephonyTransport` options (``mark_interval``,
            ``hangup_on_close``, ``http_client``...).
        max_sessions: refuse calls beyond this many live sessions.
        serve_options: ``ssl``, ``process_request``...
    """

    def __init__(
        self,
        session_factory: SessionFactory,
        agent_factory: AgentFactory,
        *,
        provider: str = "twilio",
        host: str = "127.0.0.1",
        port: int = 8765,
        frame_duration: float = 0.02,
        start_timeout: float = 10.0,
        max_sessions: int | None = None,
        serializer_options: dict[str, Any] | None = None,
        transport_options: dict[str, Any] | None = None,
        **serve_options: Any,
    ) -> None:
        super().__init__(
            session_factory,
            agent_factory,
            host=host,
            port=port,
            frame_duration=frame_duration,
            hello_timeout=start_timeout,
            max_sessions=max_sessions,
            forward_events=False,  # no data channel on a phone call
            **serve_options,
        )
        create_serializer(provider, **(serializer_options or {}))  # validate early
        self.provider = provider
        self.protocol = f"{provider} media streams"
        self._telephony_options: dict[str, Any] = {
            "frame_duration": frame_duration,
            "start_timeout": start_timeout,
            **(transport_options or {}),
            **(serializer_options or {}),
        }

    def _create_transport(self, websocket: ServerConnection) -> TelephonyTransport:
        return TelephonyTransport(websocket, provider=self.provider, **self._telephony_options)


async def serve_telephony(
    session_factory: SessionFactory,
    agent_factory: AgentFactory,
    provider: str = "twilio",
    host: str = "127.0.0.1",
    port: int = 8765,
    **options: Any,
) -> TelephonyServer:
    """Start a :class:`TelephonyServer` (one session per call) and return it."""
    server = TelephonyServer(
        session_factory, agent_factory, provider=provider, host=host, port=port, **options
    )
    await server.start()
    return server


async def _close(websocket: ServerConnection, code: int, reason: str) -> None:
    with contextlib.suppress(Exception):
        await websocket.close(code, _truncate_reason(reason))
