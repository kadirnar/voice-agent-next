"""WebRTC transport (``aiortc``): peer-to-peer Opus audio plus a JSON data channel.

Serve voice agents to browsers and mobile apps over WebRTC — UDP media with packet-loss
concealment, an adaptive jitter buffer on the client, and the browser's own echo
cancellation, noise suppression and gain control::

    from voice_agent_next import Agent, AgentSession
    from voice_agent_next.transports.webrtc import serve_webrtc

    async def main() -> None:
        server = await serve_webrtc(
            lambda: AgentSession("openai/gpt-realtime"),
            lambda: Agent("You are a helpful assistant."),
            host="127.0.0.1",
            port=8080,
            ice_servers=["stun:stun.l.google.com:19302"],
        )
        await server.serve_forever()

Requires the ``webrtc`` extra (``pip install 'voice-agent-next[webrtc]'``). Protocol summary
(full specification: ``docs/transports/webrtc.md``):

* **signalling** — the client ``POST``s its SDP offer (ICE gathering complete) as JSON
  ``{"type": "offer", "sdp": ...}`` to ``/offer`` and receives ``{"type": "answer", "sdp": ...,
  "session_id": ...}``; ``GET /config`` returns the ICE configuration for the client;
* **audio** — one bidirectional Opus track at 48 kHz, resampled to the session's formats;
* **data channel** — a negotiated channel (label ``"van"``, id ``0``) carrying the same JSON
  messages as the WebSocket transport (``ready``, ``transcript``, ``state``, ``metrics``,
  ``error``, ``clear`` from the server; ``text``, ``playout`` and app messages from the client).
"""

from __future__ import annotations

import asyncio
import contextlib
import fractions
import inspect
import json
import math
import re
import ssl as ssl_module
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..audio.frame import AudioFormat, AudioFrame
from ..audio.resample import StreamResampler, resample
from ..errors import TransportError
from ..utils.aio import BackgroundTasks, Chan, cancel_and_wait, wait_first
from ..utils.clock import now
from ..utils.deps import require
from ..utils.ids import new_id
from ..utils.log import logger
from .base import Transport, TransportCapabilities
from .websocket import SessionBridge, _describe, _dumps, _url_host, _wants_transport

if TYPE_CHECKING:
    from ..session.agent import Agent
    from ..session.session import AgentSession

__all__ = [
    "DATA_CHANNEL_ID",
    "DATA_CHANNEL_LABEL",
    "PROTOCOL",
    "AudioPlayout",
    "WebRTCAgentServer",
    "WebRTCTransport",
    "av_frame_to_audio",
    "normalize_ice_servers",
    "paced_audio_track",
    "serve_webrtc",
]

PROTOCOL = "van-webrtc/1"
"""Protocol identifier sent in the data channel's ``ready`` message."""
DATA_CHANNEL_LABEL = "van"
DATA_CHANNEL_ID = 0
"""The data channel is pre-negotiated (``negotiated: true, id: 0``) on both peers."""

OPUS_RATE = 48_000
_FRAME_SAMPLES = 960  # 20 ms at 48 kHz: one Opus packet
_RESYNC_AFTER = 0.2  # seconds behind schedule before the outbound clock restarts
_MAX_PENDING_MESSAGES = 256
_MAX_PLAYOUT_DELAY = 2.0
_MAX_BODY = 256 * 1024
_MAX_HEADER = 16 * 1024

_END_OF_CANDIDATES = re.compile(r"^a=end-of-candidates\r?\n?", re.MULTILINE)

IceTransportPolicy = Literal["all", "relay"]
SessionFactory = Callable[..., "AgentSession | Awaitable[AgentSession]"]
"""``() -> AgentSession`` or ``(transport) -> AgentSession`` (sync or async)."""
AgentFactory = Callable[..., "Agent | Awaitable[Agent]"]
"""``() -> Agent`` or ``(transport) -> Agent`` (sync or async)."""


def _aiortc() -> Any:
    return require("aiortc", extra="webrtc")


# --------------------------------------------------------------------------- ICE config
def normalize_ice_servers(
    servers: Sequence[str | Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """ICE servers in the browser's ``RTCIceServer`` shape: ``[{"urls": [...], ...}]``.

    Accepts URL strings (``"stun:host:3478"``) and mappings with ``urls`` (or ``url``) plus
    optional ``username`` / ``credential`` for TURN.
    """
    out: list[dict[str, Any]] = []
    for server in servers or ():
        if isinstance(server, str):
            entry: dict[str, Any] = {"urls": [server]}
        elif isinstance(server, Mapping):
            urls = server.get("urls", server.get("url"))
            if isinstance(urls, str):
                urls = [urls]
            if not urls or not all(isinstance(u, str) for u in urls):
                raise ValueError(f"ICE server needs `urls` (a string or a list): {server!r}")
            entry = {"urls": list(urls)}
            for key in ("username", "credential"):
                if server.get(key) is not None:
                    entry[key] = str(server[key])
        else:
            raise TypeError(f"ICE servers are URL strings or mappings, got {server!r}")
        for url in entry["urls"]:
            if url.split(":", 1)[0].lower() not in ("stun", "stuns", "turn", "turns"):
                raise ValueError(f"unsupported ICE server URL {url!r} (stun:, turn:, turns:)")
        out.append(entry)
    return out


def _has_turn(servers: list[dict[str, Any]]) -> bool:
    return any(url.lower().startswith(("turn:", "turns:")) for s in servers for url in s["urls"])


def _check_policy(policy: str, servers: list[dict[str, Any]]) -> IceTransportPolicy:
    if policy not in ("all", "relay"):
        raise ValueError(f"ice_transport_policy must be 'all' or 'relay', got {policy!r}")
    if policy == "relay" and not _has_turn(servers):
        raise ValueError("ice_transport_policy='relay' needs a TURN server in ice_servers")
    return "relay" if policy == "relay" else "all"


# ------------------------------------------------------------------------ audio helpers
def av_frame_to_audio(frame: Any) -> AudioFrame:
    """Convert a PyAV ``AudioFrame`` (any sample format / layout) to mono s16le."""
    arr = np.asarray(frame.to_ndarray())
    channels = max(1, len(frame.layout.channels))
    planar = frame.format.is_planar
    samples = arr.reshape(channels, -1).T if planar else arr.reshape(-1, channels)
    if samples.dtype.kind == "f":
        mono = np.clip(samples.mean(axis=1) * 32768.0, -32768, 32767)
    elif samples.dtype == np.int32:
        mono = samples.mean(axis=1) / 65536.0
    else:
        mono = samples.mean(axis=1)
    pcm = np.round(mono).astype("<i2").tobytes()
    return AudioFrame(pcm, int(frame.sample_rate), 1)


class AudioPlayout:
    """Audio (48 kHz mono s16le) queued for an outbound track, and its sent timeline.

    The outbound track pulls one 20 ms frame at a time, in real time; :meth:`pull` pads
    with silence when the queue runs dry (or while paused), so the RTP clock never stalls.
    """

    def __init__(self) -> None:
        self.queue = bytearray()
        self.paused = False
        self.sent_samples = 0
        """Agent audio samples handed to the encoder (silence padding excluded)."""
        self.sent_end = 0.0
        """:func:`now` at which the last agent sample handed to the encoder ends."""

    @property
    def queued(self) -> float:
        return len(self.queue) / 2 / OPUS_RATE

    def push(self, pcm: bytes | AudioFrame) -> None:
        """Queue 48 kHz mono s16le bytes (or any frame: it is converted)."""
        if isinstance(pcm, AudioFrame):
            if pcm.sample_rate != OPUS_RATE or pcm.channels != 1:
                pcm = resample(pcm.to_mono(), OPUS_RATE)
            pcm = pcm.data
        self.queue += pcm

    def clear(self) -> int:
        dropped = len(self.queue) // 2
        self.queue.clear()
        return dropped

    def pull(self, samples: int) -> bytes:
        size = samples * 2
        if self.paused or not self.queue:
            return bytes(size)
        chunk = bytes(self.queue[:size])
        del self.queue[:size]
        real = len(chunk) // 2
        self.sent_samples += real
        self.sent_end = now() + real / OPUS_RATE
        return chunk + bytes(size - len(chunk))

    def pending_until(self, t: float) -> float:
        """When (on the :func:`now` clock) the encoder will have sent everything queued."""
        if not self.queue:
            return self.sent_end
        return max(self.sent_end, t) + self.queued


_TRACK_CLASS: Any = None


def paced_audio_track(playout: AudioPlayout) -> Any:
    """An ``aiortc`` audio track that sends ``playout``'s audio in real time (20 ms frames).

    Silence is sent while the queue is empty or paused. The server uses one for the agent's
    voice; Python clients can use one as a "microphone" (see ``examples/webrtc``).
    """
    return _outbound_track_class()(playout)


def _outbound_track_class() -> Any:
    # defined lazily: it subclasses ``aiortc.MediaStreamTrack`` (an optional dependency)
    global _TRACK_CLASS
    if _TRACK_CLASS is not None:
        return _TRACK_CLASS
    aiortc = _aiortc()
    av = require("av", extra="webrtc")
    from aiortc.mediastreams import MediaStreamError

    time_base = fractions.Fraction(1, OPUS_RATE)

    base: Any = aiortc.MediaStreamTrack

    class AgentAudioTrack(base):
        kind = "audio"

        def __init__(self, playout: AudioPlayout) -> None:
            super().__init__()
            self._playout = playout
            self._start: float | None = None
            self._pts = 0

        async def recv(self) -> Any:
            if self.readyState != "live":
                raise MediaStreamError
            if self._start is None:
                self._start = now()
            else:
                self._pts += _FRAME_SAMPLES
                wait = self._start + self._pts / OPUS_RATE - now()
                if wait > 0:
                    await asyncio.sleep(wait)
                elif wait < -_RESYNC_AFTER:  # the loop stalled: restart the clock, no burst
                    self._start = now() - self._pts / OPUS_RATE
            if self.readyState != "live":
                raise MediaStreamError
            pcm = np.frombuffer(self._playout.pull(_FRAME_SAMPLES), dtype="<i2").reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
            frame.sample_rate = OPUS_RATE
            frame.pts = self._pts
            frame.time_base = time_base
            return frame

    _TRACK_CLASS = AgentAudioTrack
    return AgentAudioTrack


# -------------------------------------------------------------------------- transport
class WebRTCTransport(Transport):
    """Server side of one WebRTC peer connection (``aiortc``).

    Two ways to use it:

    * **per peer** — :class:`WebRTCAgentServer` creates one per offer, calls
      :meth:`accept_offer` and runs one session on it;
    * **standalone** — ``create_transport({"type": "webrtc", "port": 8080})``: :meth:`start`
      serves the HTTP signalling endpoint and waits for the *first* peer; later offers get
      ``503`` while it is connected, and the transport (hence the session) ends when that
      peer leaves. Attach a :class:`~voice_agent_next.transports.websocket.SessionBridge` to
      send transcripts, state changes and metrics over the data channel.

    User audio arrives as Opus (48 kHz) and is resampled to ``input_format``; agent audio
    (``output_format``) is resampled to 48 kHz and paced in real time by the outbound track,
    so :meth:`clear_audio` drops everything not sent yet. :meth:`buffered_duration` is the
    audio still queued plus the client's playout delay (network, jitter buffer, device),
    which the client refines with ``playout`` reports on the data channel.

    Args:
        host / port: signalling address in standalone mode (``port=0`` picks a free one).
        ice_servers: STUN/TURN servers for the server's peer connection (URL strings or
            ``{"urls", "username", "credential"}`` mappings). Default: none (host candidates
            only — fine on a LAN or when the server has a public address).
        ice_transport_policy: ``"relay"`` only uses TURN relay candidates (needs a TURN server).
        input_sample_rate: user audio rate delivered to the session.
        output_sample_rate: agent audio rate :meth:`write_audio` expects.
        playout_delay: initial estimate of the client's playout delay (seconds) — one-way
            network delay plus jitter buffer plus output device — until the client reports it.
        connect_timeout: seconds :meth:`start` waits for ICE + DTLS to connect.
        flush_timeout: on close, seconds to wait for queued data-channel messages.
        signaling_options: standalone mode: extra :class:`WebRTCAgentServer`-style HTTP options
            (``index_html``, ``ssl``, ``cors_origins``, ``client_ice_servers``).
    """

    capabilities = TransportCapabilities(pause=True, playback_position=True, messages=True)

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
        ice_servers: Sequence[str | Mapping[str, Any]] | None = None,
        ice_transport_policy: str = "all",
        input_sample_rate: int = 16_000,
        output_sample_rate: int = OPUS_RATE,
        playout_delay: float = 0.06,
        connect_timeout: float = 20.0,
        flush_timeout: float = 0.5,
        signaling_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            input_format=AudioFormat(input_sample_rate, 1),
            output_format=AudioFormat(output_sample_rate, 1),
        )
        if not 0 <= playout_delay <= _MAX_PLAYOUT_DELAY:
            raise ValueError(f"playout_delay must be in [0, {_MAX_PLAYOUT_DELAY}] s")
        self.host = host
        self.port = port
        self.ice_servers = normalize_ice_servers(ice_servers)
        self.ice_transport_policy = _check_policy(ice_transport_policy, self.ice_servers)
        self.playout_delay = playout_delay
        """Client playout delay in seconds (updated by the client's ``playout`` reports)."""
        self.connect_timeout = connect_timeout
        self.flush_timeout = flush_timeout
        self.signaling_options = dict(signaling_options or {})
        self.session_id = new_id("rtc_")
        self.offer: dict[str, Any] = {}
        """The client's offer request (``offer.get("metadata")`` holds app data)."""
        self.pc: Any = None
        """The ``aiortc.RTCPeerConnection`` (``None`` until an offer was accepted)."""
        self._channel: Any = None
        self._track: Any = None
        self._playout = AudioPlayout()
        self._pending_messages: list[str] = []
        self._offer_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._connected = asyncio.Event()
        self._disconnected = asyncio.Event()
        self._peer_offered = asyncio.Event()
        self._started = False
        self._closed = False
        self._input: Chan[AudioFrame] = Chan()
        self._reader: asyncio.Task[None] | None = None
        self._tasks = BackgroundTasks("webrtc-transport")
        self._in_resampler = StreamResampler(input_sample_rate, 1)
        self._out_resampler = StreamResampler(OPUS_RATE, 1)
        self._signaling: _SignalingServer | None = None

    # ------------------------------------------------------------------ properties
    @property
    def url(self) -> str:
        """``http://host:port/`` of the signalling endpoint (standalone mode)."""
        scheme = "https" if self.signaling_options.get("ssl") is not None else "http"
        return f"{scheme}://{_url_host(self.host)}:{self.port}/"

    @property
    def connected(self) -> bool:
        """ICE and DTLS are connected and the peer has not left."""
        return self._connected.is_set() and not self._disconnected.is_set()

    @property
    def sent_duration(self) -> float:
        """Seconds of agent audio handed to the Opus encoder so far."""
        return self._playout.sent_samples / OPUS_RATE

    # --------------------------------------------------------------------- signalling
    async def accept_offer(self, sdp: str, type: str = "offer") -> dict[str, Any]:
        """Answer the client's SDP offer; returns ``{"type": "answer", "sdp", "session_id"}``.

        Creates the peer connection, the outbound audio track and (if the offer has an
        ``m=application`` section) the negotiated data channel, then waits for ICE gathering
        (aiortc answers with every candidate: no trickle ICE). One offer per transport.
        """
        aiortc = _aiortc()
        async with self._offer_lock:
            if self._closed:
                raise TransportError("transport is closed")
            if self.pc is not None:
                raise TransportError("this transport already has a peer (no renegotiation)")
            if type != "offer" or not isinstance(sdp, str) or "m=audio" not in sdp:
                raise ValueError("expected an SDP offer with an audio section")
            servers = [
                aiortc.RTCIceServer(
                    urls=s["urls"], username=s.get("username"), credential=s.get("credential")
                )
                for s in self.ice_servers
            ]
            pc = aiortc.RTCPeerConnection(aiortc.RTCConfiguration(iceServers=servers))
            self.pc = pc
            pc.on("connectionstatechange", self._on_connection_state)
            pc.on("track", self._on_track)
            try:
                offer = aiortc.RTCSessionDescription(sdp=_open_candidates(sdp), type="offer")
                await pc.setRemoteDescription(offer)
                self._track = paced_audio_track(self._playout)
                pc.addTrack(self._track)
                if "m=application" in sdp:
                    self._attach_channel(
                        pc.createDataChannel(
                            DATA_CHANNEL_LABEL, negotiated=True, id=DATA_CHANNEL_ID, ordered=True
                        )
                    )
                if self.ice_transport_policy == "relay":
                    _force_relay(pc)
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
            except Exception:
                await self._close_pc()
                raise
        self._peer_offered.set()
        local = pc.localDescription
        return {"type": local.type, "sdp": local.sdp, "session_id": self.session_id}

    # ------------------------------------------------------------------- lifecycle
    async def listen(self) -> None:
        """Standalone mode: start the signalling server (idempotent); :attr:`port` is bound."""
        if self._signaling is not None:
            return
        options = dict(self.signaling_options)
        client_servers = options.pop("client_ice_servers", None)
        self._signaling = _SignalingServer(
            self._on_http_offer,
            host=self.host,
            port=self.port,
            ice_config=_client_config(
                self.ice_servers if client_servers is None else client_servers,
                self.ice_transport_policy,
            ),
            **options,
        )
        await self._signaling.start()
        self.port = self._signaling.port
        logger.info("WebRTC transport waiting for a peer on %s", self.url)

    async def start(self) -> None:
        """Wait until the peer is connected (standalone: serve signalling and wait for it).

        Idempotent. Raises :class:`~voice_agent_next.errors.TransportError` if ICE/DTLS do
        not connect within ``connect_timeout`` or the peer leaves first.
        """
        async with self._start_lock:
            if self._started:
                return
            if self._closed:
                raise TransportError("transport is closed")
            if self.pc is None:
                await self.listen()
                await wait_first(self._peer_offered.wait(), self._disconnected.wait())
            try:
                async with asyncio.timeout(self.connect_timeout):
                    await wait_first(self._connected.wait(), self._disconnected.wait())
            except TimeoutError:
                raise TransportError(
                    f"WebRTC peer did not connect within {self.connect_timeout:g} s"
                ) from None
            if not self.connected:
                raise TransportError("WebRTC peer disconnected before the session started")
            self._started = True
        logger.info(
            "WebRTC peer connected (%s, input %s, output %s)",
            self.session_id, self.input_format, self.output_format,
        )  # fmt: skip
        self.emit("connected")

    async def wait_disconnected(self) -> None:
        """Wait until the peer disconnects (or the transport is closed)."""
        await self._disconnected.wait()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._playout.clear()
        channel = self._channel
        if channel is not None and not self._disconnected.is_set():
            # flush control messages (errors, final transcripts) before hanging up
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(self.flush_timeout):
                    while channel.readyState == "open" and channel.bufferedAmount > 0:  # noqa: ASYNC110 - no event for it
                        await asyncio.sleep(0.01)
        if self._track is not None:
            self._track.stop()
        await self._close_pc()
        await cancel_and_wait(self._reader)
        await self._tasks.cancel_all()
        if self._signaling is not None:
            await self._signaling.aclose()
        self._on_disconnected()

    # ------------------------------------------------------------------ audio API
    def audio_input(self) -> AsyncIterator[AudioFrame]:
        return self._input.__aiter__()

    async def write_audio(self, frame: AudioFrame) -> None:
        """Queue agent audio for the outbound track (never blocks; paced by the track)."""
        if self._closed or self._disconnected.is_set():
            return
        out = self._out_resampler.push(frame)
        if out:
            self._playout.push(out.data)

    async def clear_audio(self) -> None:
        """Drop queued agent audio at once and tell the client (``clear``)."""
        if self._closed:
            return
        self._playout.clear()
        self._out_resampler = StreamResampler(OPUS_RATE, 1)
        self._send_json({"type": "clear"})

    async def pause_audio(self) -> None:
        """Send silence instead of agent audio, keeping the queue (false-interruption)."""
        self._playout.paused = True

    async def resume_audio(self) -> None:
        self._playout.paused = False

    def buffered_duration(self) -> float:
        """Seconds of agent audio accepted but not heard yet by the client.

        The audio still queued for the encoder, plus :attr:`playout_delay` for what was sent
        but is still in flight, in the client's jitter buffer or in its output device.
        """
        t = now()
        end = self._playout.pending_until(t)
        if end <= 0:
            return 0.0
        return max(0.0, end + self.playout_delay - t)

    # --------------------------------------------------------------- messages API
    async def send_message(self, message: dict[str, Any]) -> None:
        self.send_message_nowait(message)

    def send_message_nowait(self, message: dict[str, Any]) -> None:
        """Send a JSON message on the data channel (queued until it opens)."""
        if not isinstance(message, dict) or not isinstance(message.get("type"), str):
            raise ValueError("messages must be dicts with a string 'type'")
        self._send_json(message)

    # -------------------------------------------------------------------- internals
    def _send_json(self, message: dict[str, Any]) -> None:
        if self._disconnected.is_set():
            return
        payload = _dumps(message)
        channel = self._channel
        if channel is not None and channel.readyState == "open":
            try:
                channel.send(payload)
            except Exception as exc:  # the channel closed under us
                logger.debug("data channel send failed (%s): %s", self.session_id, exc)
            return
        if channel is None and self.pc is not None:
            return  # the peer did not negotiate a data channel
        if len(self._pending_messages) >= _MAX_PENDING_MESSAGES:
            del self._pending_messages[0]
        self._pending_messages.append(payload)

    def _attach_channel(self, channel: Any) -> None:
        self._channel = channel

        @channel.on("open")
        def _on_open() -> None:
            channel.send(
                _dumps(
                    {
                        "type": "ready",
                        "protocol": PROTOCOL,
                        "session_id": self.session_id,
                        "codec": "opus",
                        "sample_rate": self.input_format.sample_rate,
                        "output_sample_rate": self.output_format.sample_rate,
                    }
                )
            )
            pending, self._pending_messages = self._pending_messages, []
            for payload in pending:
                channel.send(payload)

        @channel.on("message")
        def _on_message(data: str | bytes) -> None:
            if isinstance(data, str):
                self._on_text(data)

        @channel.on("close")
        def _on_close() -> None:
            if self._connected.is_set():
                logger.debug("data channel closed (%s)", self.session_id)
                self._on_disconnected()

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
        if kind == "playout":
            value = message.get("delay_ms")
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value < 0
            ):
                self._invalid("`playout.delay_ms` must be a non-negative number")
                return
            self.playout_delay = min(float(value) / 1000, _MAX_PLAYOUT_DELAY)
        elif kind == "bye":
            self._on_disconnected()
        else:
            self.emit("message", message)

    def _invalid(self, reason: str) -> None:
        logger.debug("invalid data-channel message (%s): %s", self.session_id, reason)
        self._send_json({"type": "error", "code": "invalid_message", "message": reason})

    def _on_connection_state(self) -> None:
        pc = self.pc
        if pc is None:
            return
        state = pc.connectionState
        logger.debug("WebRTC %s connection state: %s", self.session_id, state)
        if state == "connected":
            self._connected.set()
        elif state in ("failed", "closed"):
            self._on_disconnected()

    def _on_track(self, track: Any) -> None:
        if track.kind != "audio" or self._reader is not None:
            return
        self._reader = asyncio.create_task(self._read_loop(track), name="webrtc-reader")

    async def _read_loop(self, track: Any) -> None:
        from aiortc.mediastreams import MediaStreamError

        try:
            while True:
                frame = await track.recv()
                if self._input.closed:
                    continue
                audio = self._in_resampler.push(av_frame_to_audio(frame))
                if audio:
                    audio.timestamp = now() - audio.duration
                    self._input.send_nowait(audio)
        except MediaStreamError:
            pass
        except Exception:
            logger.exception("WebRTC reader failed (%s)", self.session_id)
        finally:
            self._on_disconnected()

    async def _on_http_offer(self, request: dict[str, Any]) -> dict[str, Any]:
        """Standalone mode: the first peer wins; others are refused while it is served."""
        if self.pc is not None or self._closed:
            raise _HttpError(HTTPStatus.SERVICE_UNAVAILABLE, "another peer is connected")
        sdp, kind = _parse_offer(request)
        self.offer = request
        return await self.accept_offer(sdp, kind)

    def _on_disconnected(self) -> None:
        if self._disconnected.is_set():
            return
        self._disconnected.set()
        self._input.close()
        self._playout.clear()
        self.emit("disconnected")
        if not self._closed and self.pc is not None:
            logger.info("WebRTC peer disconnected (%s)", self.session_id)

    async def _close_pc(self) -> None:
        pc = self.pc
        if pc is not None:
            with contextlib.suppress(Exception):
                await pc.close()


def _open_candidates(sdp: str) -> str:
    """Drop ``a=end-of-candidates`` so ICE also accepts peer-reflexive candidates.

    Browsers hide host addresses behind mDNS names (``<uuid>.local``), which a server often
    cannot resolve (containers, no multicast). With end-of-candidates and no usable remote
    candidate, aioice prunes the component and gathers nothing, so the answer has no
    candidates and the call fails. Without it, the browser's connectivity checks reach our
    candidates and ICE learns the browser's address from them (RFC 8445 §7.3.1.3).
    """
    return _END_OF_CANDIDATES.sub("", sdp)


def _force_relay(pc: Any) -> None:
    """Gather only TURN relay candidates (aiortc has no ``iceTransportPolicy``)."""
    try:
        from aioice import TransportPolicy

        for transceiver in pc.getTransceivers():
            gatherers = {transceiver.receiver.transport.transport.iceGatherer}
            sctp = pc.sctp
            if sctp is not None:
                gatherers.add(sctp.transport.transport.iceGatherer)
            for gatherer in gatherers:
                gatherer._connection._transport_policy = TransportPolicy.RELAY
    except Exception as exc:  # private aiortc/aioice internals changed
        raise TransportError(f"cannot enforce a relay-only ICE policy: {exc}") from exc


# ---------------------------------------------------------------------------- server
class WebRTCAgentServer:
    """An HTTP signalling server that runs one :class:`AgentSession` per WebRTC peer.

    For every ``POST /offer`` it creates a :class:`WebRTCTransport`, answers the offer,
    and — once ICE/DTLS connect — builds a session and an agent with the factories,
    attaches a :class:`~voice_agent_next.transports.websocket.SessionBridge` (unless
    ``forward_events=False``) and runs the session until either side hangs up.

    The signalling server is a deliberately tiny HTTP/1.1 server on ``asyncio`` streams
    (one request per connection, bounded sizes and timeouts): WebRTC needs exactly one
    request per peer, and this avoids an ``aiohttp`` dependency. To mount signalling in
    your own web app instead (FastAPI, aiohttp, Starlette...), use ``serve_http=False`` and
    call :meth:`handle_offer` from your route, returning its result as JSON.

    Routes: ``POST /offer`` (the SDP exchange), ``GET /config`` (ICE configuration for the
    client: ``{"iceServers": [...], "iceTransportPolicy": ...}``), ``GET /`` (``index_html``,
    when given) and ``OPTIONS`` (CORS preflight, when ``cors_origins`` is set).

    Factories are called once per peer, either without arguments or — when they take a
    required positional argument — with the peer's :class:`WebRTCTransport` (see its
    ``offer`` and ``session_id``). They may be coroutine functions.

    Args:
        session_factory / agent_factory: build the session and the agent of a peer.
        host / port: signalling address (``port=0`` picks a free port; see :attr:`port`).
        ice_servers: STUN/TURN servers of the server's peer connections.
        client_ice_servers: ICE servers advertised to clients by ``GET /config`` (default:
            ``ice_servers``) — e.g. short-lived TURN credentials.
        ice_transport_policy: ``"relay"`` forces TURN relaying on both peers.
        max_sessions: answer ``503`` beyond this many live peers.
        forward_events: send transcripts, state changes, metrics and errors to clients.
        index_html: page served at ``GET /`` (e.g. the browser demo).
        cors_origins: origins allowed to call the endpoints from another page (``"*"`` = any).
        ssl: an ``ssl.SSLContext`` to serve HTTPS (browsers only grant microphone access on
            HTTPS pages and on ``http://localhost``).
        serve_http: ``False`` to not listen at all (mount :meth:`handle_offer` yourself).
        **transport_options: :class:`WebRTCTransport` options (``input_sample_rate``,
            ``output_sample_rate``, ``playout_delay``, ``connect_timeout``...).
    """

    def __init__(
        self,
        session_factory: SessionFactory,
        agent_factory: AgentFactory,
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
        ice_servers: Sequence[str | Mapping[str, Any]] | None = None,
        client_ice_servers: Sequence[str | Mapping[str, Any]] | None = None,
        ice_transport_policy: str = "all",
        max_sessions: int | None = None,
        forward_events: bool = True,
        index_html: str | None = None,
        cors_origins: Sequence[str] = (),
        ssl: ssl_module.SSLContext | None = None,
        serve_http: bool = True,
        **transport_options: Any,
    ) -> None:
        self.session_factory = session_factory
        self.agent_factory = agent_factory
        self.host = host
        self.port = port
        self.max_sessions = max_sessions
        self.forward_events = forward_events
        self.serve_http = serve_http
        servers = normalize_ice_servers(ice_servers)
        policy = _check_policy(ice_transport_policy, servers)
        clients = (
            servers if client_ice_servers is None else normalize_ice_servers(client_ice_servers)
        )
        self.ice_config = _client_config(clients, policy)
        """What ``GET /config`` returns (browser ``RTCConfiguration`` fields)."""
        self._transport_options: dict[str, Any] = {
            **transport_options,
            "ice_servers": servers,
            "ice_transport_policy": policy,
        }
        WebRTCTransport(**self._transport_options)  # validate options early
        self._http = (
            _SignalingServer(
                self.handle_offer,
                host=host,
                port=port,
                ice_config=self.ice_config,
                index_html=index_html,
                cors_origins=cors_origins,
                ssl=ssl,
            )
            if serve_http
            else None
        )
        self._transports: set[WebRTCTransport] = set()
        self._sessions: set[AgentSession] = set()
        self._tasks = BackgroundTasks("webrtc-server")
        self._runs: set[asyncio.Task[None]] = set()
        self._closing = False
        self._closed = asyncio.Event()

    @property
    def url(self) -> str:
        scheme = "https" if self._http is not None and self._http.ssl is not None else "http"
        return f"{scheme}://{_url_host(self.host)}:{self.port}/"

    @property
    def sessions(self) -> list[AgentSession]:
        """Sessions currently running."""
        return list(self._sessions)

    @property
    def transports(self) -> list[WebRTCTransport]:
        """Peers currently connecting or connected."""
        return list(self._transports)

    async def start(self) -> None:
        """Start listening (idempotent). :attr:`port` is then the bound port."""
        if self._http is not None and self._http.server is None:
            await self._http.start()
            self.port = self._http.port
            logger.info("serving voice agents over WebRTC; signalling on %s", self.url)

    async def serve_forever(self) -> None:
        """Serve until :meth:`aclose` is called or this coroutine is cancelled."""
        await self.start()
        try:
            await self._closed.wait()
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        """Stop accepting peers, hang up every peer and wait for their sessions."""
        if self._closing:
            await self._closed.wait()
            return
        self._closing = True
        if self._http is not None:
            await self._http.aclose()
        for transport in list(self._transports):
            with contextlib.suppress(Exception):
                await transport.aclose()
        runs = list(self._runs)
        if runs:
            await asyncio.wait(runs, timeout=5.0)
        await cancel_and_wait(*runs)
        self._closed.set()

    async def __aenter__(self) -> WebRTCAgentServer:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def handle_offer(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Answer an offer ``{"type": "offer", "sdp": ..., "metadata"?: ...}``; start its session.

        Returns ``{"type": "answer", "sdp": ..., "session_id": ...}``. Raises ``ValueError``
        for a malformed offer and :class:`~voice_agent_next.errors.TransportError` when the
        server is closing or full (map them to HTTP 400 / 503 in your framework).
        """
        sdp, kind = _parse_offer(request)
        if self._closing:
            raise _HttpError(HTTPStatus.SERVICE_UNAVAILABLE, "server is shutting down")
        if self.max_sessions is not None and len(self._transports) >= self.max_sessions:
            raise _HttpError(HTTPStatus.SERVICE_UNAVAILABLE, "too many sessions")
        transport = WebRTCTransport(**self._transport_options)
        transport.offer = dict(request)
        self._transports.add(transport)
        try:
            answer = await transport.accept_offer(sdp, kind)
        except BaseException:
            self._transports.discard(transport)
            await transport.aclose()
            raise
        run = self._tasks.spawn(self._run(transport), name=f"webrtc-{transport.session_id}")
        self._runs.add(run)
        run.add_done_callback(self._runs.discard)
        return answer

    async def _run(self, transport: WebRTCTransport) -> None:
        session: AgentSession | None = None
        bridge: SessionBridge | None = None
        reason = "user_disconnected"
        try:
            try:
                await transport.start()
            except TransportError as exc:
                logger.info("WebRTC peer %s did not connect: %s", transport.session_id, exc)
                return
            session = await _call_factory(self.session_factory, transport)
            agent = await _call_factory(self.agent_factory, transport)
            self._sessions.add(session)
            if self.forward_events:
                bridge = SessionBridge(session, transport)
            await session.start(agent, transport)
            await wait_first(session.wait_closed(), transport.wait_disconnected())
        except Exception as exc:
            logger.exception("WebRTC session %s failed", transport.session_id)
            reason = "error"
            transport.send_message_nowait(
                {
                    "type": "error",
                    "code": "internal_error",
                    "message": _describe(exc),
                    "fatal": True,
                }
            )
        finally:
            if session is not None:
                with contextlib.suppress(Exception):
                    await session.aclose(reason)  # no-op if the session already closed
                self._sessions.discard(session)
            if bridge is not None:
                await bridge.aclose()
            await transport.aclose()
            self._transports.discard(transport)


async def serve_webrtc(
    session_factory: SessionFactory,
    agent_factory: AgentFactory,
    host: str = "127.0.0.1",
    port: int = 8080,
    **options: Any,
) -> WebRTCAgentServer:
    """Start a :class:`WebRTCAgentServer` (one session per peer) and return it.

    ``options`` are the keyword arguments of :class:`WebRTCAgentServer` (``ice_servers``,
    ``max_sessions``, ``index_html``, ``ssl``, ``output_sample_rate``...).
    """
    server = WebRTCAgentServer(session_factory, agent_factory, host=host, port=port, **options)
    await server.start()
    return server


# ------------------------------------------------------------------ HTTP signalling
class _HttpError(TransportError):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(slots=True)
class _Response:
    status: HTTPStatus
    body: bytes = b""
    content_type: str = "application/json"


def _client_config(servers: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    return {"iceServers": normalize_ice_servers(servers), "iceTransportPolicy": policy}


def _parse_offer(request: Mapping[str, Any]) -> tuple[str, str]:
    if not isinstance(request, Mapping):
        raise ValueError("the offer must be a JSON object")
    sdp, kind = request.get("sdp"), request.get("type", "offer")
    if not isinstance(sdp, str) or not sdp.strip():
        raise ValueError("`sdp` must be a non-empty string")
    if kind != "offer":
        raise ValueError('`type` must be "offer"')
    if "m=audio" not in sdp:
        raise ValueError("the offer has no audio section")
    return sdp, kind


class _SignalingServer:
    """Minimal HTTP/1.1 server for the SDP exchange (one request per connection)."""

    def __init__(
        self,
        on_offer: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
        *,
        host: str,
        port: int,
        ice_config: dict[str, Any],
        index_html: str | None = None,
        cors_origins: Sequence[str] = (),
        ssl: ssl_module.SSLContext | None = None,
        offer_path: str = "/offer",
        config_path: str = "/config",
        request_timeout: float = 10.0,
    ) -> None:
        self.on_offer = on_offer
        self.host = host
        self.port = port
        self.ice_config = ice_config
        self.index_html = index_html
        self.cors_origins = tuple(cors_origins)
        self.ssl = ssl
        self.offer_path = offer_path
        self.config_path = config_path
        self.request_timeout = request_timeout
        self.server: asyncio.Server | None = None
        self._handlers: set[asyncio.Task[Any]] = set()

    async def start(self) -> None:
        if self.server is not None:
            return
        self.server = await asyncio.start_server(
            self._serve, self.host, self.port, ssl=self.ssl, limit=_MAX_HEADER
        )
        for sock in self.server.sockets:
            self.port = int(sock.getsockname()[1])
            break

    async def aclose(self) -> None:
        server, self.server = self.server, None
        if server is not None:
            server.close()
            await cancel_and_wait(*self._handlers)
            with contextlib.suppress(Exception):
                await server.wait_closed()

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._handlers.add(task)  # cancelled by aclose()
        origin: str | None = None
        try:
            try:
                async with asyncio.timeout(self.request_timeout):
                    method, path, headers, body = await _read_request(reader)
                origin = headers.get("origin")
                response = await self._route(method, path, headers, body)
            except _HttpError as exc:
                response = _json_response(exc.status, {"error": str(exc)})
            except TimeoutError:
                response = _json_response(HTTPStatus.REQUEST_TIMEOUT, {"error": "timeout"})
            except (asyncio.IncompleteReadError, ConnectionError):
                return
            writer.write(self._encode(response, origin))
            await writer.drain()
        except ConnectionError:
            pass
        except Exception:
            logger.exception("WebRTC signalling request failed")
        finally:
            if task is not None:
                self._handlers.discard(task)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _route(
        self, method: str, path: str, headers: dict[str, str], body: bytes
    ) -> _Response:
        path = path.split("?", 1)[0]
        if method == "OPTIONS":
            return _Response(HTTPStatus.NO_CONTENT)
        if path == self.offer_path:
            if method != "POST":
                raise _HttpError(HTTPStatus.METHOD_NOT_ALLOWED, "use POST")
            try:
                request = json.loads(body)
            except ValueError:
                raise _HttpError(HTTPStatus.BAD_REQUEST, "the body must be JSON") from None
            try:
                answer = await self.on_offer(request)
            except ValueError as exc:
                raise _HttpError(HTTPStatus.BAD_REQUEST, str(exc)) from None
            except _HttpError:
                raise
            except TransportError as exc:
                raise _HttpError(HTTPStatus.SERVICE_UNAVAILABLE, str(exc)) from None
            except Exception as exc:
                logger.exception("WebRTC offer failed")
                raise _HttpError(HTTPStatus.INTERNAL_SERVER_ERROR, _describe(exc)) from None
            return _json_response(HTTPStatus.OK, answer)
        if method != "GET":
            raise _HttpError(HTTPStatus.METHOD_NOT_ALLOWED, f"{method} not allowed")
        if path == self.config_path:
            return _json_response(HTTPStatus.OK, self.ice_config)
        if self.index_html is not None and path in ("/", "/index.html"):
            return _Response(HTTPStatus.OK, self.index_html.encode(), "text/html; charset=utf-8")
        raise _HttpError(HTTPStatus.NOT_FOUND, "not found")

    def _encode(self, response: _Response, origin: str | None) -> bytes:
        lines = [
            f"HTTP/1.1 {response.status.value} {response.status.phrase}",
            f"Content-Length: {len(response.body)}",
            "Connection: close",
            "Cache-Control: no-store",
        ]
        if response.body:
            lines.append(f"Content-Type: {response.content_type}")
        allowed = self._allowed_origin(origin)
        if allowed is not None:
            lines += [
                f"Access-Control-Allow-Origin: {allowed}",
                "Access-Control-Allow-Methods: GET, POST, OPTIONS",
                "Access-Control-Allow-Headers: Content-Type",
                "Vary: Origin",
            ]
        return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + response.body

    def _allowed_origin(self, origin: str | None) -> str | None:
        if not self.cors_origins:
            return None
        if "*" in self.cors_origins:
            return "*"
        return origin if origin in self.cors_origins else None


def _json_response(status: HTTPStatus, payload: dict[str, Any]) -> _Response:
    return _Response(status, _dumps(payload).encode())


async def _read_request(reader: asyncio.StreamReader) -> tuple[str, str, dict[str, str], bytes]:
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except asyncio.LimitOverrunError:
        raise _HttpError(HTTPStatus.REQUEST_HEADER_FIELDS_TOO_LARGE, "headers too large") from None
    lines = head.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3 or not parts[2].startswith("HTTP/1."):
        raise _HttpError(HTTPStatus.BAD_REQUEST, "malformed request line")
    method, path = parts[0].upper(), parts[1]
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line:
            continue
        name, sep, value = line.partition(":")
        if not sep:
            raise _HttpError(HTTPStatus.BAD_REQUEST, "malformed header")
        headers[name.strip().lower()] = value.strip()
    if "chunked" in headers.get("transfer-encoding", "").lower():
        raise _HttpError(HTTPStatus.LENGTH_REQUIRED, "chunked bodies are not supported")
    try:
        length = int(headers.get("content-length", "0"))
    except ValueError:
        raise _HttpError(HTTPStatus.BAD_REQUEST, "bad Content-Length") from None
    if length < 0:
        raise _HttpError(HTTPStatus.BAD_REQUEST, "bad Content-Length")
    if length > _MAX_BODY:
        raise _HttpError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "body too large")
    body = await reader.readexactly(length) if length else b""
    return method, path, headers, body


async def _call_factory(factory: Callable[..., Any], transport: WebRTCTransport) -> Any:
    result = factory(transport) if _wants_transport(factory) else factory()
    if inspect.isawaitable(result):
        result = await result
    return result
