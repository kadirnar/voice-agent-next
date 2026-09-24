"""Kyutai Moshi full-duplex speech-to-speech engine (the ``moshi-server`` WebSocket protocol).

``AgentSession("moshi")`` connects to a running Moshi server — Kyutai's PyTorch
``python -m moshi.server``, the Rust ``moshi-backend`` or ``python -m moshi_mlx.local_web``
on Apple Silicon — and streams the conversation both ways, continuously. NVIDIA PersonaPlex
speaks the same protocol (see :mod:`voice_agent_next.providers.personaplex`).

**Wire protocol** (``GET /api/chat``, WebSocket, binary messages; the first byte is the
kind): ``0x00`` handshake (the server is ready; the Rust server appends two little-endian
``u32``: protocol and model version), ``0x01`` audio (Ogg/Opus pages, 24 kHz mono, in
both directions), ``0x02`` text (a UTF-8 token of the model's inner monologue, the
transcript of what it says), ``0x03`` control, ``0x04`` metadata (JSON), ``0x05`` error
(UTF-8), ``0x06`` ping and ``0x07`` coloured text. The model runs one step per 80 ms of
*received* audio (1920 samples of the Mimi codec at 12.5 Hz): its output is clocked by
the client's input, so the client must keep streaming — silence included — for the model
to keep talking. The engine does that by itself (``keepalive``) when the transport stops
delivering audio.

**Full duplex.** Moshi listens and speaks at the same time: it produces an output stream
all the time (silence when it is quiet), backchannels, yields when the user talks over
it, and decides by itself when to speak. There is no "request a response", no text
input, no tools, and nothing to cancel on the server. The engine maps the stream onto
the turn-based event protocol like this:

* a **response** is a stretch of agent speech: it starts at the first text token or
  output chunk above ``speech_threshold_db`` and ends after ``response_gap`` seconds
  with neither (the pause belongs to it; the silence between responses is not
  forwarded). Text tokens -> ``ResponseText``;
* **user speech** comes from a local energy VAD on the sent audio. ``InputSpeechStarted``
  / ``InputSpeechStopped`` are reported while the agent is quiet; speech *over* the agent
  is left to the model (Moshi resolves overlaps itself), so the session's barge-in
  policy never pauses, truncates or cancels Moshi. If the agent falls silent while the
  user keeps talking, ``InputSpeechStarted`` is reported then. ``report_overlap=True``
  reports overlapping speech too (the session then applies its interruption policy);
* a **turn**: the first agent response after user speech is preceded by
  ``InputCommitted`` (there is no user transcript: Moshi does not transcribe the user),
  so the session measures voice-to-voice latency from the local end of speech;
* **interrupt** (``session.interrupt()`` or a confirmed barge-in with
  ``report_overlap``): the model cannot be stopped, so the current response ends as
  ``cancelled`` and the agent's audio is muted until its next pause;
* ``say()`` / ``create_response()`` / ``send_text()`` cannot steer Moshi and are ignored
  with a warning. PersonaPlex takes a text role prompt (``instructions``) and a voice
  prompt at connect time.

**Reconnects.** When the connection drops (server restart, the Rust server's step limit
of 4500 steps = 6 minutes) the engine reconnects with backoff and reports
``EngineStatus("reconnecting" / "reconnected")``. The new server session starts from
scratch: Moshi has no way to restore a conversation.

**Why sphn:** it is Kyutai's own Ogg/Opus stream codec (Rust with a bundled libopus,
wheels for Linux, macOS and Windows, numpy only) that ``moshi.server`` itself uses, so
the framing is byte-compatible with every server; ``opuslib``/``pyogg`` need a system
libopus and hand-written Ogg paging, and ``av`` is a 30+ MB FFmpeg build.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import json
import ssl
from collections import deque
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode, urlsplit, urlunsplit

import numpy as np

from ..audio.frame import AudioFrame
from ..chat import FunctionCallOutput
from ..engine import EngineCapabilities, EngineConnection, EngineOptions, S2SEngine
from ..errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from ..events import (
    EngineErrorEvent,
    EngineStatus,
    InputCommitted,
    InputSpeechStarted,
    InputSpeechStopped,
    ResponseAudio,
    ResponseDone,
    ResponseStarted,
    ResponseStatus,
    ResponseText,
)
from ..metrics import EngineMetrics
from ..registry import register_provider
from ..tools import FunctionTool
from ..utils.aio import BackgroundTasks, Chan, cancel_and_wait
from ..utils.clock import now
from ..utils.deps import require
from ..utils.ids import new_id
from ..utils.log import logger
from ..vad import VADEventType, VADOptions, VADStream
from .energy import EnergyVAD

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_URL",
    "FRAME_SAMPLES",
    "SAMPLE_RATE",
    "MoshiConnection",
    "MoshiEngine",
    "MsgKind",
    "OpusDecoder",
    "OpusEncoder",
]

DEFAULT_MODEL = "moshiko"
DEFAULT_URL = "ws://localhost:8998"
KNOWN_MODELS = ("moshiko", "moshika", "moshiko-q8", "moshika-q8", "moshika-rl-seamless")
SAMPLE_RATE = 24_000
FRAME_RATE = 12.5
"""Mimi frames (model steps) per second."""
FRAME_SAMPLES = 1920
"""Samples per Mimi frame (80 ms at 24 kHz): the unit the server processes."""
CHAT_PATH = "/api/chat"

_MAX_MESSAGE = 16 * 2**20
_PUMP_INTERVAL = 0.04
_KEEPALIVE_IDLE = 0.1
"""Input silence (s) after which the engine streams silence itself (see ``keepalive``)."""
_KEEPALIVE_CHUNK = 0.08


class MsgKind:
    """First byte of every ``/api/chat`` message."""

    HANDSHAKE = 0x00
    AUDIO = 0x01
    TEXT = 0x02
    CONTROL = 0x03
    METADATA = 0x04
    ERROR = 0x05
    PING = 0x06
    COLORED_TEXT = 0x07


# ------------------------------------------------------------------------------- codec
class OpusEncoder:
    """PCM16 -> Ogg/Opus pages (``sphn``), in whole Opus frames of ``frame_samples``.

    The first page carries the ``OpusHead``/``OpusTags`` headers, so every encoder starts
    a new Ogg stream (one per server connection).
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, frame_samples: int = FRAME_SAMPLES):
        sphn = require("sphn", extra="moshi")
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self._writer = sphn.OpusStreamWriter(sample_rate)
        self._pending = np.zeros(0, dtype=np.float32)

    @property
    def buffered(self) -> float:
        """Seconds of audio waiting for a complete Opus frame."""
        return len(self._pending) / self.sample_rate

    def encode(self, pcm: bytes | AudioFrame) -> bytes:
        """Buffer ``pcm`` (s16le mono) and return the Ogg pages of every complete frame."""
        data = pcm.data if isinstance(pcm, AudioFrame) else pcm
        x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        buf = np.concatenate((self._pending, x)) if len(self._pending) else x
        out = bytearray()
        n = self.frame_samples
        i = 0
        while len(buf) - i >= n:
            out += self._writer.append_pcm(np.ascontiguousarray(buf[i : i + n]))
            i += n
        self._pending = buf[i:].copy()
        return bytes(out)


class OpusDecoder:
    """Ogg/Opus pages -> PCM16 (``sphn``)."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        sphn = require("sphn", extra="moshi")
        self.sample_rate = sample_rate
        self._reader = sphn.OpusStreamReader(sample_rate)

    def decode(self, pages: bytes) -> AudioFrame:
        pcm = self._reader.append_bytes(pages)
        if pcm is None or len(pcm) == 0:
            return AudioFrame.empty(self.sample_rate)
        return AudioFrame.from_numpy(np.asarray(pcm, dtype=np.float32), self.sample_rate)


# ----------------------------------------------------------------------------- helpers
def _http_error(status: int, body: str, provider: str) -> ProviderError:
    msg = f"{provider} server rejected the connection (HTTP {status}): {body.strip()[:300]}"
    if status in (401, 403):
        return AuthenticationError(msg, provider=provider, status_code=status)
    if status == 429:
        return RateLimitError(msg, provider=provider, status_code=status)
    if status >= 500:
        return ProviderConnectionError(msg, provider=provider, status_code=status)
    return ProviderError(msg, provider=provider, status_code=status)


async def _close_quietly(ws: ClientConnection, timeout: float = 2.0) -> None:
    with contextlib.suppress(Exception):
        await asyncio.wait_for(ws.close(), timeout)


@functools.cache
def _connect_accepts_proxy() -> bool:
    from websockets.asyncio.client import connect

    return "proxy" in inspect.signature(connect.__init__).parameters


def _dbfs(frame: AudioFrame) -> float:
    rms = frame.rms()
    return -120.0 if rms <= 1e-6 else 20.0 * float(np.log10(rms))


# ------------------------------------------------------------------------------- engine
@register_provider(
    "engine",
    "moshi",
    description="Kyutai Moshi full-duplex speech-to-speech (local moshi-server WebSocket)",
    default_model=DEFAULT_MODEL,
    models=KNOWN_MODELS,
    extra="moshi",
    requires=("sphn", "websockets"),
    local=True,
)
class MoshiEngine(S2SEngine):
    """Full-duplex engine for a running Moshi server (see the module docs).

    Args:
        model: label of the model the server runs (``moshiko``, ``moshika``...). The
            server loads one model at start-up; this is informational (metrics).
        url: server origin, e.g. ``"ws://localhost:8998"`` (``http(s)://`` works too); the
            ``/api/chat`` path is added unless the URL has a path.
        ssl_verify: verify the TLS certificate of a ``wss://`` server. Default: not for
            ``localhost`` (the Rust backend and PersonaPlex serve self-signed certificates),
            yes for any other host.
        text_temperature / text_topk / audio_temperature / audio_topk / pad_mult /
        repetition_penalty / repetition_penalty_context / seed: sampling parameters
            sent as query parameters (the Rust server applies them; ``moshi.server``
            ignores them).
        query: extra query parameters.
        frame_duration: Opus frame length sent to the server (0.02-0.08 s; the server
            processes 80 ms steps).
        speech_threshold_db: output level (dBFS) above which agent audio counts as speech.
        response_gap: agent silence (s, no text either) that ends a response.
        preroll: seconds of audio before a detected speech onset that are included in the
            response (keeps the first syllable's attack).
        report_overlap: also report user speech while the agent speaks, letting the
            session's interruption policy mute Moshi (default: the model owns the floor).
        user_vad_threshold_db: level of the local energy VAD on the user audio.
        user_min_silence: silence (s) that ends a user utterance for that VAD.
        keepalive: stream silence to the server when the transport delivers no audio
            (the model only advances with input audio).
        connect_timeout: WebSocket + handshake timeout (a busy ``moshi.server`` serves one
            client at a time and sends the handshake only when free).
        reconnect: reconnect when the connection drops.
        max_reconnect_attempts: consecutive failed reconnects before giving up.
    """

    provider = "moshi"
    handshake_timeout_hint = "is the Moshi server running? `python -m moshi.server`"

    def __init__(
        self,
        *,
        model: str | None = None,
        url: str | None = None,
        ssl_verify: bool | None = None,
        text_temperature: float | None = None,
        text_topk: int | None = None,
        audio_temperature: float | None = None,
        audio_topk: int | None = None,
        pad_mult: float | None = None,
        repetition_penalty: float | None = None,
        repetition_penalty_context: int | None = None,
        seed: int | None = None,
        query: Mapping[str, Any] | None = None,
        frame_duration: float = 0.08,
        speech_threshold_db: float = -40.0,
        response_gap: float = 0.64,
        preroll: float = 0.08,
        report_overlap: bool = False,
        user_vad_threshold_db: float = -40.0,
        user_min_silence: float = 0.3,
        keepalive: bool = True,
        connect_timeout: float = 30.0,
        reconnect: bool = True,
        max_reconnect_attempts: int = 5,
    ) -> None:
        frame_samples = round(frame_duration * SAMPLE_RATE)
        if frame_samples not in (480, 960, 1440, 1920):
            raise ConfigurationError("frame_duration must be 0.02, 0.04, 0.06 or 0.08 seconds")
        if response_gap <= 0:
            raise ConfigurationError("response_gap must be > 0")
        if max_reconnect_attempts < 1:
            raise ConfigurationError("max_reconnect_attempts must be >= 1")
        super().__init__(
            model=model or DEFAULT_MODEL,
            capabilities=EngineCapabilities(
                native_audio=True,
                server_turn_detection=True,  # the model decides when to speak
                tool_calling=False,
                input_transcription=False,  # Moshi does not transcribe the user
                output_transcription=True,  # inner-monologue text tokens
                truncation=False,
                full_duplex=True,
                text_input=False,
                max_session_duration=None,
            ),
            input_sample_rate=SAMPLE_RATE,
            output_sample_rate=SAMPLE_RATE,
        )
        self.url = url or DEFAULT_URL
        self.ssl_verify = ssl_verify
        params: dict[str, Any] = {
            "text_temperature": text_temperature,
            "text_topk": text_topk,
            "audio_temperature": audio_temperature,
            "audio_topk": audio_topk,
            "pad_mult": pad_mult,
            "repetition_penalty": repetition_penalty,
            "repetition_penalty_context": repetition_penalty_context,
        }
        if seed is not None:
            params.update(text_seed=seed, audio_seed=seed, seed=seed)
        self.query: dict[str, Any] = {k: v for k, v in params.items() if v is not None}
        self.query.update(query or {})
        self.frame_samples = frame_samples
        self.speech_threshold_db = speech_threshold_db
        self.response_gap = response_gap
        self.preroll = preroll
        self.report_overlap = report_overlap
        self.user_vad_threshold_db = user_vad_threshold_db
        self.user_min_silence = user_min_silence
        self.keepalive = keepalive
        self.connect_timeout = connect_timeout
        self.reconnect = reconnect
        self.max_reconnect_attempts = max_reconnect_attempts

    # -------------------------------------------------------------------- endpoint
    def _query(self, options: EngineOptions) -> dict[str, Any]:
        """Query parameters of the ``/api/chat`` request (subclasses add prompts)."""
        return dict(self.query)

    def endpoint(self, options: EngineOptions) -> str:
        """The WebSocket URL for a connection with ``options``."""
        parts = urlsplit(self.url if "://" in self.url else f"ws://{self.url}")
        scheme = {"http": "ws", "https": "wss"}.get(parts.scheme, parts.scheme)
        if scheme not in ("ws", "wss"):
            raise ConfigurationError(f"unsupported Moshi server URL {self.url!r}")
        path = parts.path.rstrip("/") or CHAT_PATH
        query = self._query(options)
        extra = parts.query
        encoded = urlencode({k: _query_value(v) for k, v in query.items()})
        full_query = "&".join(q for q in (extra, encoded) if q)
        return urlunsplit((scheme, parts.netloc, path, full_query, ""))

    def _connect_kwargs(self, url: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        parts = urlsplit(url)
        local = parts.hostname in ("127.0.0.1", "localhost", "::1")
        verify = self.ssl_verify if self.ssl_verify is not None else not local
        if parts.scheme == "wss" and not verify:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            kwargs["ssl"] = ctx
        if local and _connect_accepts_proxy():
            kwargs["proxy"] = None  # never route a local server through a system proxy
        return kwargs

    def _check_options(self, options: EngineOptions) -> None:
        if options.instructions.strip():
            logger.info(
                "%s: Moshi takes no instructions; the agent's instructions are ignored "
                "(PersonaPlex takes a role prompt: use the `personaplex` engine)",
                self.provider,
            )
        if options.tools:
            logger.warning(
                "%s: the model cannot call tools; %d tools ignored",
                self.provider,
                len(options.tools),
            )

    async def connect(self, options: EngineOptions) -> EngineConnection:
        self._check_options(options)
        conn = MoshiConnection(self, options)
        try:
            await conn._start()
        except BaseException:
            await conn.aclose()
            raise
        return conn


def _query_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


# --------------------------------------------------------------------------- connection
class _AgentResponse:
    __slots__ = (
        "ended_at",
        "first_audio_at",
        "item_id",
        "last_activity",
        "response_id",
        "samples",
        "started_at",
        "text_tokens",
        "trigger_at",
    )

    def __init__(self, position: float, trigger_at: float | None) -> None:
        self.response_id = new_id("resp_")
        self.item_id = new_id("item_")
        self.started_at = now()
        self.trigger_at = trigger_at
        self.first_audio_at: float | None = None
        self.ended_at: float | None = None
        self.last_activity = position
        """Output position (s) of the latest speech chunk or text token."""
        self.samples = 0
        self.text_tokens = 0


class MoshiConnection(EngineConnection):
    """A live full-duplex conversation with a Moshi server (possibly over reconnects)."""

    def __init__(self, engine: MoshiEngine, options: EngineOptions) -> None:
        super().__init__(engine, options)
        self._e = engine
        self._tasks = BackgroundTasks("moshi")
        self._ws: ClientConnection | None = None
        self._epoch = 0
        self._encoder: OpusEncoder | None = None
        self._decoder: OpusDecoder | None = None
        self._outbox: Chan[bytes] = Chan()
        self._sender_task: asyncio.Task[None] | None = None
        self._pump_task: asyncio.Task[None] | None = None
        self._reconnect_task: asyncio.Task[None] | None = None
        self._reconnecting = False
        self.connections = 0
        """Number of WebSocket connections opened so far."""
        self.server_metadata: dict[str, Any] | None = None
        """The ``0x04`` metadata of the current server (Rust backend), if it sent any."""
        self.server_version: tuple[int, int] | None = None
        """(protocol, model) version from the Rust server's handshake, if sent."""
        self.errors: list[str] = []
        """``0x05`` error messages received from the server."""
        # ---- clocks
        self._started_at = now()
        self._last_input_at: float | None = None
        self._conn_input = 0.0
        """Seconds of audio handed to the encoder on the current connection."""
        self._out_pos = 0.0
        """Seconds of agent audio received on the current connection."""
        self.lag = 0.0
        """Latest model lag: input sent minus output received on this connection (s),
        i.e. how far the server's output trails the audio it was given."""
        self.max_lag = 0.0
        # ---- agent speech segmentation
        self._resp: _AgentResponse | None = None
        self._preroll: deque[AudioFrame] = deque()
        self._muted = False
        self._quiet_since = 0.0
        # ---- user speech (local VAD)
        opts = VADOptions(min_speech_duration=0.1, min_silence_duration=engine.user_min_silence)
        self._vad: VADStream = EnergyVAD(
            sample_rate=SAMPLE_RATE, threshold_db=engine.user_vad_threshold_db, options=opts
        ).stream()
        self._user_speaking = False
        self._user_reported = False
        self._user_pending = False
        """The user spoke since the agent's last response (the next one commits a turn)."""
        self._user_speech_end: float | None = None
        self._input_since_response = 0.0
        self._warned: set[str] = set()

    # ------------------------------------------------------------------- lifecycle
    async def _start(self) -> None:
        ws = await self._open()
        self._install(ws)
        self._started_at = now()
        self._sender_task = asyncio.create_task(self._sender(), name="moshi-send")
        self._pump_task = asyncio.create_task(self._pump(), name="moshi-pump")

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        current = asyncio.current_task()
        tasks = [self._pump_task, self._sender_task, self._reconnect_task]
        await cancel_and_wait(*[t for t in tasks if t is not None and t is not current])
        await self._tasks.cancel_all()
        self._outbox.close()
        ws, self._ws = self._ws, None
        if ws is not None:
            await _close_quietly(ws)
        self._vad.close()
        self._end_response("incomplete")
        await super().aclose()

    async def _open(self) -> ClientConnection:
        """Connect and wait for the server's handshake."""
        from websockets.asyncio.client import connect
        from websockets.exceptions import (
            ConnectionClosed,
            InvalidStatus,
            InvalidURI,
            WebSocketException,
        )

        e = self._e
        url = e.endpoint(self.options)
        try:
            ws = await connect(
                url,
                open_timeout=e.connect_timeout,
                max_size=_MAX_MESSAGE,
                compression=None,
                **e._connect_kwargs(url),
            )
        except InvalidStatus as exc:
            body = exc.response.body.decode("utf-8", "replace") if exc.response.body else ""
            raise _http_error(exc.response.status_code, body, e.provider) from exc
        except InvalidURI as exc:
            raise ConfigurationError(f"invalid {e.provider} server URL: {exc}") from exc
        except (OSError, WebSocketException, TimeoutError) as exc:
            msg = (
                f"cannot connect to the {e.provider} server at {e.url}: "
                f"{type(exc).__name__}: {exc} ({e.handshake_timeout_hint})"
            )
            raise ProviderConnectionError(msg, provider=e.provider) from exc
        self.connections += 1
        try:
            async with asyncio.timeout(e.connect_timeout):
                while True:
                    raw = await ws.recv()
                    if not isinstance(raw, bytes) or not raw:
                        continue
                    if raw[0] == MsgKind.HANDSHAKE:
                        payload = raw[1:]
                        if len(payload) >= 8:
                            proto = int.from_bytes(payload[:4], "little")
                            model = int.from_bytes(payload[4:8], "little")
                            self.server_version = (proto, model)
                        break
                    if raw[0] == MsgKind.ERROR:
                        text = raw[1:].decode("utf-8", "replace")
                        raise ProviderError(
                            f"{e.provider} server error: {text}", provider=e.provider
                        )
                    self._dispatch(raw)  # metadata before the handshake (Rust backend)
        except TimeoutError as exc:
            await _close_quietly(ws)
            msg = (
                f"no handshake from the {e.provider} server within {e.connect_timeout:.0f}s "
                "(moshi.server serves one conversation at a time: is another client connected?)"
            )
            raise ProviderTimeoutError(msg, provider=e.provider) from exc
        except ConnectionClosed as exc:
            msg = f"the {e.provider} server closed the connection during the handshake: {exc}"
            raise ProviderConnectionError(msg, provider=e.provider) from exc
        except BaseException:
            await _close_quietly(ws)
            raise
        return ws

    def _install(self, ws: ClientConnection) -> None:
        self._epoch += 1
        self._ws = ws
        self._encoder = OpusEncoder(SAMPLE_RATE, self._e.frame_samples)
        self._decoder = OpusDecoder(SAMPLE_RATE)
        self._conn_input = 0.0
        self._out_pos = 0.0
        self._quiet_since = 0.0
        self._preroll.clear()
        self._tasks.spawn(self._recv_loop(ws, self._epoch), name=f"moshi-recv-{self._epoch}")

    async def _recv_loop(self, ws: ClientConnection, epoch: int) -> None:
        from websockets.exceptions import ConnectionClosed

        reason = "the server closed the connection"
        try:
            async for raw in ws:
                if self._closed or epoch != self._epoch:
                    return
                if isinstance(raw, bytes) and raw:
                    self._dispatch(raw)
        except ConnectionClosed as exc:
            reason = f"connection lost: {exc}"
        except Exception as exc:
            logger.exception("moshi: receiving failed")
            reason = f"receive failed: {exc}"
            await _close_quietly(ws)
        if self._closed or epoch != self._epoch:
            return
        self._on_disconnect(reason)

    def _on_disconnect(self, reason: str) -> None:
        self._ws = None
        self._end_response("incomplete")
        error = ProviderConnectionError(f"{self._e.provider}: {reason}", provider=self._e.provider)
        if not self._e.reconnect:
            self._fail(error)
            return
        logger.warning("moshi: %s; reconnecting", reason)
        self._reconnecting = True
        self._emit(EngineStatus(status="reconnecting", detail=reason))
        self._reconnect_task = asyncio.create_task(self._reconnect(error), name="moshi-reconnect")

    async def _reconnect(self, error: Exception) -> None:
        delay = 0.25
        for attempt in range(1, self._e.max_reconnect_attempts + 1):
            await asyncio.sleep(delay)
            delay = min(delay * 2, 5.0)
            try:
                ws = await self._open()
            except (ProviderError, ConfigurationError) as exc:
                logger.warning("moshi: reconnect attempt %d failed: %s", attempt, exc)
                error = exc
                if isinstance(exc, (AuthenticationError, ConfigurationError)):
                    break
                continue
            if self._closed:
                await _close_quietly(ws)
                return
            self._install(ws)
            self._reconnecting = False
            self._emit(
                EngineStatus(
                    status="reconnected", detail="new server session: the conversation starts over"
                )
            )
            return
        self._fail(error)

    def _fail(self, error: Exception) -> None:
        if self._closed:
            return
        self._emit(EngineErrorEvent(error=error, recoverable=False))
        self._tasks.spawn(self.aclose(), name="moshi-close")

    # -------------------------------------------------------------------- sending
    async def _send_audio(self, frame: AudioFrame) -> None:
        self._last_input_at = now()
        self._push(frame)

    def _push(self, frame: AudioFrame) -> None:
        """Feed audio at 24 kHz mono (real or keep-alive) to the VAD and the server."""
        self._track_user(frame)
        self._input_since_response += frame.duration
        encoder = self._encoder
        if encoder is None or self._ws is None or self._reconnecting:
            return  # reconnecting: stale audio is useless to a real-time model
        self._conn_input += frame.duration
        pages = encoder.encode(frame)
        if pages:
            self._outbox.send_nowait(bytes([MsgKind.AUDIO]) + pages)

    async def _sender(self) -> None:
        """Send queued messages in order (encoding happens synchronously in :meth:`_push`)."""
        from websockets.exceptions import ConnectionClosed

        async for message in self._outbox:
            ws = self._ws
            if ws is None:
                continue
            try:
                await ws.send(message)
            except ConnectionClosed:
                continue  # the receive loop notices and reconnects
            except Exception as exc:
                logger.warning("moshi: send failed: %s", exc)

    async def _pump(self) -> None:
        """Keep the model's clock running: stream silence while the transport is idle."""
        while not self._closed:
            await asyncio.sleep(_PUMP_INTERVAL)
            if not self._e.keepalive or self._ws is None or self._reconnecting:
                continue
            t = now()
            if self._last_input_at is not None and t - self._last_input_at < _KEEPALIVE_IDLE:
                continue
            behind = (t - self._started_at) - self.input_audio_time
            while behind >= _KEEPALIVE_CHUNK / 2:
                chunk = min(behind, _KEEPALIVE_CHUNK)
                await self.send_audio(AudioFrame.silence(chunk, SAMPLE_RATE))
                self._last_input_at = None  # keep-alive audio is not user input
                behind -= chunk

    # ------------------------------------------------------------------ user speech
    def _track_user(self, frame: AudioFrame) -> None:
        for ev in self._vad.push_audio(frame):
            if ev.type == VADEventType.START_OF_SPEECH:
                self._user_speaking = True
                self._user_pending = True
                start = max(0.0, ev.audio_time - ev.speech_duration)
                if self._resp is None or self._e.report_overlap:
                    self._report_user_start(start)
            elif ev.type == VADEventType.END_OF_SPEECH:
                self._user_speaking = False
                end = max(0.0, ev.audio_time - ev.silence_duration)
                self._user_speech_end = end
                if self._user_reported:
                    self._user_reported = False
                    self._emit(InputSpeechStopped(audio_time=end))

    def _report_user_start(self, audio_time: float | None) -> None:
        if not self._user_reported:
            self._user_reported = True
            self._emit(InputSpeechStarted(audio_time=audio_time))

    # ------------------------------------------------------------------- receiving
    def _dispatch(self, raw: bytes) -> None:
        try:
            self._on_message(raw[0], raw[1:])
        except Exception:
            logger.exception("moshi: failed to handle a server message")

    def _on_message(self, kind: int, payload: bytes) -> None:
        if kind == MsgKind.AUDIO:
            if self._decoder is not None:
                self._on_audio(self._decoder.decode(payload))
        elif kind == MsgKind.TEXT:
            self._on_text(payload.decode("utf-8", "replace"))
        elif kind == MsgKind.COLORED_TEXT:
            self._on_text(payload[1:].decode("utf-8", "replace"))
        elif kind == MsgKind.METADATA:
            with contextlib.suppress(ValueError):
                value = json.loads(payload.decode("utf-8", "replace"))
                if isinstance(value, dict):
                    self.server_metadata = value
        elif kind == MsgKind.ERROR:
            text = payload.decode("utf-8", "replace")
            self.errors.append(text)
            logger.warning("moshi: server error: %s", text)
            error = ProviderError(
                f"{self._e.provider} server error: {text}", provider=self._e.provider
            )
            self._emit(EngineErrorEvent(error=error, recoverable=True))
        # handshake (a repeated one), control and ping carry nothing for the client

    def _on_audio(self, frame: AudioFrame) -> None:
        if not frame:
            return
        self._out_pos += frame.duration
        self.lag = max(0.0, self._conn_input - self._out_pos)
        self.max_lag = max(self.max_lag, self.lag)
        loud = _dbfs(frame) >= self._e.speech_threshold_db
        if self._muted:
            if loud:
                self._quiet_since = self._out_pos
            elif self._out_pos - self._quiet_since >= self._e.response_gap:
                self._muted = False  # the model paused: its next utterance is heard again
            return
        resp = self._resp
        if resp is None:
            if not loud:
                self._keep_preroll(frame)
                return
            resp = self._begin_response()
            preroll, self._preroll = list(self._preroll), deque()
            for pending in preroll:
                self._forward(resp, pending)
        self._forward(resp, frame)
        if loud:
            resp.last_activity = self._out_pos
        elif self._out_pos - resp.last_activity >= self._e.response_gap:
            self._end_response("completed")

    def _keep_preroll(self, frame: AudioFrame) -> None:
        self._preroll.append(frame)
        total = sum(f.duration for f in self._preroll)
        while self._preroll and total - self._preroll[0].duration >= self._e.preroll - 1e-9:
            total -= self._preroll.popleft().duration

    def _on_text(self, text: str) -> None:
        if not text or self._muted:
            return
        resp = self._resp or self._begin_response()
        resp.text_tokens += 1
        resp.last_activity = self._out_pos
        self._emit(ResponseText(response_id=resp.response_id, item_id=resp.item_id, delta=text))

    def _forward(self, resp: _AgentResponse, frame: AudioFrame) -> None:
        if resp.first_audio_at is None:
            resp.first_audio_at = now()
        resp.samples += frame.samples_per_channel
        self._emit(ResponseAudio(response_id=resp.response_id, item_id=resp.item_id, frame=frame))

    # ------------------------------------------------------------------- responses
    def _begin_response(self) -> _AgentResponse:
        trigger: float | None = None
        if self._user_pending:
            self._user_pending = False
            if not self._user_speaking and self._user_speech_end is not None:
                trigger = self.audio_time_to_wall(self._user_speech_end)
            self._emit(InputCommitted(item_id=new_id("item_")))
        resp = _AgentResponse(self._out_pos, trigger)
        self._resp = resp
        self._emit(ResponseStarted(response_id=resp.response_id))
        return resp

    def _end_response(self, status: ResponseStatus) -> None:
        resp, self._resp = self._resp, None
        if resp is None:
            return
        resp.ended_at = now()
        self._emit(ResponseDone(response_id=resp.response_id, status=status))
        self._emit_metrics(resp, cancelled=status == "cancelled")
        self._input_since_response = 0.0
        self._preroll.clear()
        if self._user_speaking and not self._user_reported and not self._closed:
            # the agent yielded while the user is still talking: the floor is theirs now
            self._report_user_start(self.input_audio_time)

    def _emit_metrics(self, resp: _AgentResponse, *, cancelled: bool) -> None:
        ttfb = None
        if resp.trigger_at is not None and resp.first_audio_at is not None:
            ttfb = max(0.0, resp.first_audio_at - resp.trigger_at)
        duration = resp.samples / SAMPLE_RATE
        self._e.emit(
            "metrics",
            EngineMetrics(
                provider=self._e.provider,
                model=self._e.model,
                response_id=resp.response_id,
                ttfb=ttfb,
                duration=(resp.ended_at or now()) - resp.started_at,
                input_audio_tokens=round(self._input_since_response * FRAME_RATE),
                output_text_tokens=resp.text_tokens,
                output_audio_tokens=round(duration * FRAME_RATE),
                cancelled=cancelled,
            ),
        )

    # --------------------------------------------------------------------- control
    def _warn_once(self, what: str, message: str) -> None:
        if what not in self._warned:
            self._warned.add(what)
            logger.warning("%s: %s", self._e.provider, message)

    async def commit_input(self) -> None:
        """No-op: the model takes turns by itself."""

    async def clear_input(self) -> None:
        """No-op: audio already streamed has been heard by the model."""

    async def send_text(self, text: str, *, respond: bool = True) -> None:
        self._warn_once("text", "the model takes no text input; send_text() is ignored")

    async def create_response(self, *, instructions: str | None = None) -> None:
        self._warn_once(
            "respond",
            "the model decides by itself when to speak; create_response()/say() are ignored",
        )

    async def cancel_response(self) -> None:
        """End the current response as cancelled and mute the agent until its next pause
        (a full-duplex model cannot be stopped; it usually yields to the user by itself)."""
        if self._resp is None:
            return
        self._muted = True
        self._quiet_since = self._out_pos
        self._end_response("cancelled")

    async def send_tool_output(self, output: FunctionCallOutput, *, respond: bool = True) -> None:
        self._warn_once("tools", "the model cannot call tools; tool output ignored")

    async def update(
        self, *, instructions: str | None = None, tools: list[FunctionTool] | None = None
    ) -> None:
        if instructions is not None:
            self.options.instructions = instructions
            self._warn_once(
                "update",
                "prompts are fixed when the server session starts; new instructions apply "
                "only after a reconnect",
            )
