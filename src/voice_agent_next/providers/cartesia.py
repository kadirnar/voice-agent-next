"""Cartesia providers: Sonic streaming TTS and Ink streaming STT.

* ``tts="cartesia/sonic-3.6"`` — :class:`CartesiaTTS`. Native text streaming over one
  persistent WebSocket (``/tts/websocket``): one ``context_id`` per segment, text deltas
  sent with ``continue: true``, a low ``max_buffer_delay_ms`` (Cartesia buffers text for
  up to 3 s by default), word timestamps (``add_timestamps``) surfaced as
  :attr:`~voice_agent_next.tts.SynthesizedAudio.words`, and open contexts cancelled on
  close/interrupt. :meth:`~voice_agent_next.tts.TTS.synthesize` uses the HTTP bytes
  endpoint (``/tts/bytes``).
* ``stt="cartesia/ink-2"`` — :class:`CartesiaSTT`. Realtime transcription with semantic
  turn detection (``/stt/turns/websocket``: ``turn.start`` / ``turn.update`` /
  ``turn.eager_end`` / ``turn.resume`` / ``turn.end`` become STT turn events) or, with
  ``turn_detection=False``, externally endpointed transcription (``/stt/websocket``),
  where :meth:`~voice_agent_next.stt.STTStream.flush` sends ``finalize``.

Only core dependencies are used (``websockets`` and ``httpx``, imported lazily to keep
``van providers`` fast). Credentials come from ``api_key=`` or ``CARTESIA_API_KEY``; the
API version is pinned with the ``Cartesia-Version`` header (:data:`API_VERSION`).
See ``docs/providers/cartesia.md``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode, urlsplit, urlunsplit

from ..audio.frame import AudioFrame
from ..errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from ..registry import register_provider
from ..stt import STT, STTCapabilities, STTEvent, STTEventType, STTStream, Transcript, WordTiming
from ..tts import TTS, ChunkedStream, SynthesizedAudio, SynthesizeStream, TTSCapabilities
from ..utils.aio import Chan, cancel_and_wait
from ..utils.clock import now
from ..utils.ids import new_id
from ..utils.log import logger

if TYPE_CHECKING:
    import httpx
    from websockets.asyncio.client import ClientConnection

__all__ = [
    "API_VERSION",
    "DEFAULT_BASE_URL",
    "DEFAULT_VOICE",
    "TTS_SAMPLE_RATES",
    "CartesiaSTT",
    "CartesiaTTS",
]

API_VERSION = "2026-08-14"
"""``Cartesia-Version`` sent with every request (override with ``api_version=``)."""
DEFAULT_BASE_URL = "https://api.cartesia.ai"
DEFAULT_VOICE = "f786b574-daa5-4673-aa0c-cbe3e8534c02"
"""The voice used in Cartesia's quickstart; any voice id from the Cartesia playground works."""
TTS_SAMPLE_RATES = (8000, 16000, 22050, 24000, 44100, 48000)
"""Output sample rates accepted by Sonic for raw PCM."""

_PROVIDER = "cartesia"
_MAX_MESSAGE_BYTES = 16 * 2**20


# ----------------------------------------------------------------------------- helpers
def _resolve_api_key(api_key: str | None) -> str:
    key = api_key or os.environ.get("CARTESIA_API_KEY")
    if not key:
        raise ConfigurationError(
            "Cartesia needs an API key: pass api_key=... or set CARTESIA_API_KEY"
        )
    return key


def _headers(api_key: str, api_version: str) -> dict[str, str]:
    # The official SDK authenticates with a bearer token, the WebSocket reference documents
    # X-API-Key; both carry the same key and are accepted.
    return {
        "Authorization": f"Bearer {api_key}",
        "X-API-Key": api_key,
        "Cartesia-Version": api_version,
    }


def _ws_base(base_url: str) -> str:
    parts = urlsplit(base_url)
    scheme = {"https": "wss", "http": "ws"}.get(parts.scheme, parts.scheme)
    return urlunsplit((scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def _param(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _error_for_status(status: int | None, message: str) -> ProviderError:
    if status in (401, 403):
        return AuthenticationError(message, provider=_PROVIDER, status_code=status)
    if status == 429:
        return RateLimitError(message, provider=_PROVIDER, status_code=status)
    if status in (408, 504):
        return ProviderTimeoutError(message, provider=_PROVIDER, status_code=status)
    retryable = status is not None and status >= 500
    return ProviderError(message, provider=_PROVIDER, retryable=retryable, status_code=status)


def _api_error(msg: Mapping[str, Any]) -> ProviderError:
    """Map a Cartesia ``{"type": "error", ...}`` payload to a library exception."""
    raw_status = msg.get("status_code")
    status = raw_status if isinstance(raw_status, int) else None
    title = msg.get("title") or "error"
    detail = msg.get("message") or msg.get("error") or ""
    text = f"Cartesia {title}"
    if detail:
        text += f": {detail}"
    if msg.get("error_code"):
        text += f" [{msg['error_code']}]"
    if status is not None:
        text += f" (status {status})"
    return _error_for_status(status, text)


def _http_error(status: int, body: str) -> ProviderError:
    with contextlib.suppress(ValueError):
        data = json.loads(body)
        if isinstance(data, dict):
            return _api_error({**data, "status_code": status})
    return _error_for_status(status, f"Cartesia HTTP {status}: {body.strip()[:500]}")


async def _ws_connect(
    url: str, headers: Mapping[str, str], *, open_timeout: float
) -> ClientConnection:
    from websockets.asyncio.client import connect
    from websockets.exceptions import InvalidHandshake, InvalidStatus, InvalidURI

    try:
        return await connect(
            url,
            additional_headers=dict(headers),
            open_timeout=open_timeout,
            max_size=_MAX_MESSAGE_BYTES,
        )
    except InvalidStatus as exc:
        body = exc.response.body or b""
        raise _http_error(exc.response.status_code, body.decode("utf-8", "replace")) from exc
    except TimeoutError as exc:
        raise ProviderTimeoutError(
            f"timed out connecting to Cartesia ({url})", provider=_PROVIDER
        ) from exc
    except (OSError, InvalidHandshake, InvalidURI) as exc:
        raise ProviderConnectionError(
            f"cannot connect to Cartesia ({url}): {exc}", provider=_PROVIDER
        ) from exc


async def _close_ws(ws: ClientConnection) -> None:
    with contextlib.suppress(Exception):
        async with asyncio.timeout(2.0):
            await ws.close()


async def _raise_task_error(
    reader: asyncio.Task[Any], writer: asyncio.Task[Any], *, grace: float = 0.5
) -> None:
    """Raise the error of ``reader`` or else ``writer`` (both are retrieved).

    When the writer fails because the server hung up, the reader usually receives the
    server's explanation (an ``error`` message) right after: give it ``grace`` seconds so
    users see "invalid model" rather than "connection closed".
    """
    writer_failed = writer.done() and not writer.cancelled() and writer.exception() is not None
    if writer_failed and not reader.done():
        await asyncio.wait((reader,), timeout=grace)
    errors = [t.exception() for t in (reader, writer) if t.done() and not t.cancelled()]
    for error in errors:
        if error is not None:
            raise error


def _closed_error(what: str, exc: BaseException | None = None) -> ProviderConnectionError:
    detail = f": {exc}" if exc is not None else ""
    return ProviderConnectionError(f"Cartesia {what} WebSocket closed{detail}", provider=_PROVIDER)


def _parse(raw: str | bytes, what: str) -> dict[str, Any] | None:
    if isinstance(raw, bytes):
        logger.debug("Cartesia %s: ignoring a binary message", what)
        return None
    try:
        msg = json.loads(raw)
    except ValueError:
        logger.warning("Cartesia %s: ignoring invalid JSON: %.200s", what, raw)
        return None
    return msg if isinstance(msg, dict) else None


def _tts_words(data: Any, offset: float) -> list[WordTiming]:
    """``{"words": [...], "start": [...], "end": [...]}`` (seconds) -> offset word timings."""
    if not isinstance(data, Mapping):
        return []
    words, starts, ends = data.get("words") or [], data.get("start") or [], data.get("end") or []
    return [
        WordTiming(str(w), float(s) + offset, float(e) + offset)
        for w, s, e in zip(words, starts, ends, strict=False)
    ]


def _stt_words(data: Any) -> list[WordTiming]:
    """``[{"word": ..., "start": ..., "end": ...}]`` -> word timings."""
    if not isinstance(data, list):
        return []
    out: list[WordTiming] = []
    for w in data:
        if isinstance(w, Mapping) and "word" in w:
            out.append(
                WordTiming(str(w["word"]), float(w.get("start", 0.0)), float(w.get("end", 0.0)))
            )
    return out


# --------------------------------------------------------------------------------- TTS
@register_provider(
    "tts",
    "cartesia",
    description="Cartesia Sonic: WebSocket text streaming (continuations) + word timestamps",
    default_model="sonic-3.6",
    models=("sonic-3.6", "sonic-3.6-2026-08-27", "sonic-3.5", "sonic-3", "sonic-latest"),
    env=("CARTESIA_API_KEY",),
    extra=None,
    requires=("websockets", "httpx"),
    local=False,
)
class CartesiaTTS(TTS):
    """Cartesia Sonic text-to-speech.

    :meth:`stream` pushes text into Cartesia *contexts* over one WebSocket that is shared
    by all streams of this instance (opened lazily or by :meth:`warmup`, reopened after
    Cartesia closes an idle socket). Every segment (text between :meth:`flush` calls) is
    one ``context_id``; each text delta is sent with ``continue: true`` so prosody carries
    across sentences, and the flush sends an empty ``continue: false`` input. Segments are
    played in order even when their generation overlaps. Closing a stream (interruption)
    cancels its unfinished contexts.

    With ``word_timestamps=True`` the stream also yields items with an empty ``frame``
    whose ``words`` carry :class:`~voice_agent_next.stt.WordTiming` in seconds **from the
    start of the stream's audio** (across segments), which gives exact truncation on
    barge-in: the words heard are those that end before the played position.

    :meth:`synthesize` (one complete text) uses the HTTP ``/tts/bytes`` endpoint, which
    streams raw PCM but carries no timestamps.

    Args:
        model: ``sonic-3.6`` (latest stable), a dated snapshot such as
            ``sonic-3.6-2026-08-27``, ``sonic-3.5``, ``sonic-3``, ``sonic-latest``...
        voice: Cartesia voice id (default :data:`DEFAULT_VOICE`).
        api_key: defaults to ``$CARTESIA_API_KEY``.
        sample_rate: raw ``pcm_s16le`` output rate, one of :data:`TTS_SAMPLE_RATES`.
        language: language code of the text (``"en"``, ``"fr"``...); ``None`` = API default.
        speed: ``generation_config.speed`` (0.6-1.5).
        volume: ``generation_config.volume`` (0.5-2.0).
        emotion: ``generation_config.emotion`` (``"calm"``, ``"excited"``...).
        max_buffer_delay_ms: how long Cartesia may wait for more text before generating
            (0-5000). The default ``0`` generates as soon as text arrives, which is right
            when text is pushed in sentences (as the cascade does); raise it (e.g. 300-1000)
            when pushing raw LLM tokens. ``None`` omits the field (API default: 3000 ms).
        word_timestamps: request word timestamps (``add_timestamps``) on streams.
        pronunciation_dict_id: optional pronunciation dictionary id.
        base_url: API base URL (``wss://`` is derived from it for WebSockets).
        api_version: ``Cartesia-Version`` to pin.
        http_client: optional ``httpx.AsyncClient`` for :meth:`synthesize` (not closed by
            :meth:`aclose`).
        connect_timeout: connection / handshake timeout in seconds.
        receive_timeout: after the end of a segment's input, fail with
            :class:`~voice_agent_next.errors.ProviderTimeoutError` if Cartesia stays silent
            this long (a watchdog against audio that never arrives); also the HTTP read
            timeout of :meth:`synthesize`.
    """

    provider = "cartesia"

    def __init__(
        self,
        *,
        model: str = "sonic-3.6",
        voice: str | None = None,
        api_key: str | None = None,
        sample_rate: int = 24_000,
        language: str | None = None,
        speed: float | None = None,
        volume: float | None = None,
        emotion: str | None = None,
        max_buffer_delay_ms: int | None = 0,
        word_timestamps: bool = True,
        pronunciation_dict_id: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        api_version: str = API_VERSION,
        http_client: httpx.AsyncClient | None = None,
        connect_timeout: float = 10.0,
        receive_timeout: float = 10.0,
        clean_text: bool = True,
    ) -> None:
        if sample_rate not in TTS_SAMPLE_RATES:
            raise ConfigurationError(
                f"Cartesia TTS sample_rate must be one of {TTS_SAMPLE_RATES}, got {sample_rate}"
            )
        if max_buffer_delay_ms is not None and not 0 <= max_buffer_delay_ms <= 5000:
            raise ConfigurationError(
                f"max_buffer_delay_ms must be within [0, 5000], got {max_buffer_delay_ms}"
            )
        if speed is not None and not 0.6 <= speed <= 1.5:
            raise ConfigurationError(f"speed must be within [0.6, 1.5], got {speed}")
        if volume is not None and not 0.5 <= volume <= 2.0:
            raise ConfigurationError(f"volume must be within [0.5, 2.0], got {volume}")
        super().__init__(
            model=model,
            sample_rate=sample_rate,
            channels=1,
            capabilities=TTSCapabilities(streaming=True, word_timestamps=word_timestamps),
            voice=voice or DEFAULT_VOICE,
            clean_text=clean_text,
        )
        self._api_key = _resolve_api_key(api_key)
        self.language = language
        self.speed = speed
        self.volume = volume
        self.emotion = emotion
        self.max_buffer_delay_ms = max_buffer_delay_ms
        self.pronunciation_dict_id = pronunciation_dict_id
        self.base_url = base_url.rstrip("/")
        self.api_version = api_version
        self.connect_timeout = connect_timeout
        self.receive_timeout = receive_timeout
        self._http = http_client
        self._owns_http = http_client is None
        self._conn: _TTSConnection | None = None
        self._conn_lock: asyncio.Lock | None = None
        self._conn_lock_loop: asyncio.AbstractEventLoop | None = None

    # ---------------------------------------------------------------- requests
    def _headers(self) -> dict[str, str]:
        return _headers(self._api_key, self.api_version)

    def _ws_url(self) -> str:
        query = urlencode({"cartesia_version": self.api_version})
        return f"{_ws_base(self.base_url)}/tts/websocket?{query}"

    def _base_request(self, voice: str | None) -> dict[str, Any]:
        """Fields shared by ``/tts/bytes`` and every WebSocket input."""
        request: dict[str, Any] = {
            "model_id": self.model,
            "voice": {"id": voice or self.voice},
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self.sample_rate,
            },
        }
        if self.language:
            request["language"] = self.language
        generation = {
            k: v
            for k, v in (("speed", self.speed), ("volume", self.volume), ("emotion", self.emotion))
            if v is not None
        }
        if generation:
            request["generation_config"] = generation
        if self.pronunciation_dict_id:
            request["pronunciation_dict_id"] = self.pronunciation_dict_id
        return request

    def _stream_request(self, voice: str | None) -> dict[str, Any]:
        """WebSocket input fields; identical for every input of a context (API rule)."""
        request = self._base_request(voice)
        if self.max_buffer_delay_ms is not None:
            request["max_buffer_delay_ms"] = self.max_buffer_delay_ms
        if self.capabilities.word_timestamps:
            request["add_timestamps"] = True
        return request

    # --------------------------------------------------------------- transport
    async def _connection(self) -> _TTSConnection:
        """The shared WebSocket, (re)connecting when needed."""
        loop = asyncio.get_running_loop()
        if self._conn_lock is None or self._conn_lock_loop is not loop:
            self._conn_lock, self._conn_lock_loop = asyncio.Lock(), loop
        async with self._conn_lock:
            conn = self._conn
            if conn is not None and (conn.closed or conn.loop is not loop):
                if conn.loop is loop:
                    await conn.aclose()
                conn = self._conn = None
            if conn is None:
                ws = await _ws_connect(
                    self._ws_url(), self._headers(), open_timeout=self.connect_timeout
                )
                conn = self._conn = _TTSConnection(ws)
            return conn

    def _http_client(self) -> httpx.AsyncClient:
        import httpx

        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(self.receive_timeout, connect=self.connect_timeout)
            )
            self._owns_http = True
        return self._http

    # --------------------------------------------------------------------- API
    def _synthesize(self, text: str, *, voice: str | None) -> ChunkedStream:
        return _CartesiaChunkedStream(self, text, voice=voice)

    def _create_stream(self, *, voice: str | None) -> SynthesizeStream:
        return _CartesiaSynthesizeStream(self, voice=voice)

    async def warmup(self) -> None:
        """Open the WebSocket ahead of the first request (saves the TLS + WS handshake)."""
        await self._connection()

    async def aclose(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None and conn.loop is asyncio.get_running_loop():
            await conn.aclose()
        if self._http is not None and self._owns_http:
            http, self._http = self._http, None
            await http.aclose()


class _TTSContext:
    """One Cartesia generation context (``context_id``) on a shared connection."""

    def __init__(self, conn: _TTSConnection) -> None:
        self.conn = conn
        self.id = uuid.uuid4().hex
        self.messages: Chan[dict[str, Any] | ProviderError] = Chan()
        self.last_activity = now()
        self.closed_at: float | None = None
        """When ``continue: false`` was sent; the receive watchdog only runs after it."""
        self.ended = False
        """The server finished the context (``done``/``error``) or the connection failed."""
        self.finished = False
        """The owning stream consumed the context (or gave up on it)."""

    def deliver(self, item: dict[str, Any] | ProviderError) -> None:
        self.last_activity = now()
        if isinstance(item, ProviderError) or item.get("type") in ("done", "error"):
            self.ended = True
        if not self.messages.closed:
            self.messages.send_nowait(item)


class _TTSConnection:
    """A Cartesia TTS WebSocket multiplexing the contexts of every stream of one TTS.

    A reader task routes messages to their context by ``context_id`` and drops late
    messages of cancelled or finished contexts. When the socket closes, every open context
    receives a :class:`~voice_agent_next.errors.ProviderConnectionError`.
    """

    def __init__(self, ws: ClientConnection) -> None:
        self.ws = ws
        self.loop = asyncio.get_running_loop()
        self.closed = False
        self._contexts: dict[str, _TTSContext] = {}
        self._reader = asyncio.create_task(self._read_loop(), name="cartesia-tts-reader")

    def new_context(self) -> _TTSContext:
        ctx = _TTSContext(self)
        self._contexts[ctx.id] = ctx
        return ctx

    def release(self, ctx: _TTSContext) -> None:
        self._contexts.pop(ctx.id, None)

    async def send(self, msg: Mapping[str, Any]) -> None:
        from websockets.exceptions import ConnectionClosed

        if self.closed:
            raise _closed_error("TTS")
        try:
            await self.ws.send(json.dumps(msg))
        except ConnectionClosed as exc:
            self.closed = True
            raise _closed_error("TTS", exc) from exc

    async def _read_loop(self) -> None:
        from websockets.exceptions import ConnectionClosed

        error = _closed_error("TTS")
        try:
            async for raw in self.ws:
                msg = _parse(raw, "TTS")
                if msg is not None:
                    self._dispatch(msg)
        except ConnectionClosed as exc:
            error = _closed_error("TTS", exc)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Cartesia TTS reader failed")
            error = _closed_error("TTS", exc)
        finally:
            self.closed = True
            contexts, self._contexts = list(self._contexts.values()), {}
            for ctx in contexts:
                ctx.deliver(error)

    def _dispatch(self, msg: dict[str, Any]) -> None:
        ctx_id = msg.get("context_id")
        ctx = self._contexts.get(ctx_id) if isinstance(ctx_id, str) else None
        if ctx is not None:
            ctx.deliver(msg)
        elif msg.get("type") == "error" and not ctx_id:
            error = _api_error(msg)  # not tied to a context: fail every open one
            for c in list(self._contexts.values()):
                c.deliver(error)
        # else: a late message for a cancelled or finished context

    async def aclose(self) -> None:
        self.closed = True
        await _close_ws(self.ws)
        await cancel_and_wait(self._reader)


class _CartesiaChunkedStream(ChunkedStream):
    """``POST /tts/bytes`` with raw PCM output, streamed as it arrives."""

    async def _run(self) -> None:
        import httpx

        tts: CartesiaTTS = self._tts  # type: ignore[assignment]
        if not self.text.strip():
            return
        body = {**tts._base_request(self.voice), "transcript": self.text}
        try:
            async with tts._http_client().stream(
                "POST", f"{tts.base_url}/tts/bytes", json=body, headers=tts._headers()
            ) as response:
                if response.status_code >= 400:
                    raw = await response.aread()
                    raise _http_error(response.status_code, raw.decode("utf-8", "replace"))
                async for chunk in response.aiter_bytes():
                    self._push_audio(chunk)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Cartesia /tts/bytes timed out: {exc!r}", provider=_PROVIDER
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderConnectionError(
                f"Cartesia /tts/bytes failed: {exc!r}", provider=_PROVIDER
            ) from exc


@dataclass(eq=False)
class _Segment:
    """Text between two flushes: normally one context, more if Cartesia ended one early."""

    contexts: Chan[_TTSContext] = field(default_factory=Chan)
    text: list[str] = field(default_factory=list)
    current: _TTSContext | None = None


class _CartesiaSynthesizeStream(SynthesizeStream):
    """Native text streaming: a feeder sends inputs, a player emits audio in segment order."""

    def __init__(self, tts: CartesiaTTS, *, voice: str | None) -> None:
        self._request = tts._stream_request(voice)
        self._contexts: list[_TTSContext] = []
        super().__init__(tts, voice=voice)

    async def _run(self) -> None:
        segments: Chan[_Segment] = Chan()
        feeder = asyncio.create_task(self._feed(segments), name="cartesia-tts-feed")
        player = asyncio.create_task(self._play(segments), name="cartesia-tts-play")
        try:
            await asyncio.wait((feeder, player), return_when=asyncio.FIRST_EXCEPTION)
            await _raise_task_error(player, feeder)
        finally:
            segments.close()
            await cancel_and_wait(feeder, player)
            await self._cancel_contexts()

    # ------------------------------------------------------------------ feeder
    async def _feed(self, segments: Chan[_Segment]) -> None:
        segment: _Segment | None = None
        async for item in self._input:
            if segment is None:
                segment = _Segment()
                segments.send_nowait(segment)
            if self.is_flush(item):
                await self._close_segment(segment)
                segment = None
            else:
                assert isinstance(item, str)
                await self._push_segment_text(segment, item)
        # end_input() flushes before closing, so an open segment here means aclose():
        # don't finish its context gracefully, _cancel_contexts() cancels it
        segments.close()

    async def _push_segment_text(self, segment: _Segment, text: str) -> None:
        segment.text.append(text)
        ctx = segment.current
        if ctx is not None and not ctx.ended:
            await self._send_input(ctx, text, more=True)
            return
        # first text of the segment, or Cartesia already ended the context (e.g. it
        # expired while the LLM was slow): continue the segment on a fresh context
        try:
            ctx = await self._open_context(text)
        except ProviderConnectionError:
            logger.info("Cartesia TTS WebSocket was closed; reconnecting")
            ctx = await self._open_context(text)
        segment.current = ctx
        segment.contexts.send_nowait(ctx)

    async def _open_context(self, text: str) -> _TTSContext:
        tts: CartesiaTTS = self._tts  # type: ignore[assignment]
        conn = await tts._connection()
        ctx = conn.new_context()
        try:
            await self._send_input(ctx, text, more=True)
        except BaseException:
            conn.release(ctx)
            raise
        self._contexts.append(ctx)
        return ctx

    async def _close_segment(self, segment: _Segment) -> None:
        ctx = segment.current
        if ctx is not None and not ctx.ended:
            await self._send_input(ctx, "", more=False)
        segment.contexts.close()

    async def _send_input(self, ctx: _TTSContext, text: str, *, more: bool) -> None:
        await ctx.conn.send({**self._request, "context_id": ctx.id, "transcript": text,
                             "continue": more})  # fmt: skip
        if not more:
            ctx.closed_at = now()

    # ------------------------------------------------------------------ player
    async def _play(self, segments: Chan[_Segment]) -> None:
        async for segment in segments:
            async for ctx in segment.contexts:
                await self._play_context(ctx)
            self._segment_text = "".join(segment.text).strip() or None
            self._end_segment()

    async def _play_context(self, ctx: _TTSContext) -> None:
        """Emit a context's audio and words until ``done``.

        Only a completed context is released here: on errors, timeouts and cancellation
        :meth:`_cancel_contexts` releases it and cancels it if it may still be generating.
        """
        tts: CartesiaTTS = self._tts  # type: ignore[assignment]
        offset = self._audio_duration  # Cartesia's timestamps restart at 0 in every context
        while True:
            msg = await self._next_message(ctx, tts.receive_timeout)
            if isinstance(msg, ProviderError):
                raise msg
            kind = msg.get("type")
            if kind == "chunk":
                data = msg.get("data")
                if data:
                    self._push_audio(base64.b64decode(data))
                if msg.get("done") is True:
                    break
            elif kind == "timestamps":
                words = _tts_words(msg.get("word_timestamps"), offset)
                if words:
                    empty = AudioFrame.empty(tts.sample_rate, tts.channels)
                    self._send(
                        SynthesizedAudio(empty, self._request_id, self._segment_id, words=words)
                    )
            elif kind == "done":
                break
            elif kind == "error":
                raise _api_error(msg)
            # flush_done / phoneme_timestamps / anything new: nothing to do
        ctx.finished = True
        ctx.conn.release(ctx)

    @staticmethod
    async def _next_message(ctx: _TTSContext, timeout: float) -> dict[str, Any] | ProviderError:
        # asyncio.timeout rather than wait_for: on Python 3.11, wait_for can swallow the
        # cancellation of an interrupted stream when a message arrives at the same time
        while True:
            wait = timeout
            if ctx.closed_at is not None:
                wait = timeout - (now() - max(ctx.closed_at, ctx.last_activity))
                if wait <= 0:
                    raise ProviderTimeoutError(
                        f"Cartesia sent nothing for {timeout:.1f}s after the end of the input "
                        f"(context {ctx.id})",
                        provider=_PROVIDER,
                    )
            try:
                async with asyncio.timeout(wait):
                    return await ctx.messages.recv()
            except TimeoutError:
                continue  # re-check: the input may have ended in the meantime

    # ----------------------------------------------------------- cancellation
    async def _cancel_contexts(self) -> None:
        """Cancel contexts that are still generating (interruption, error or close)."""
        for ctx in self._contexts:
            if ctx.finished:
                continue
            ctx.finished = True
            ctx.conn.release(ctx)
            if ctx.ended or ctx.conn.closed:
                continue
            try:
                async with asyncio.timeout(1.0):
                    await ctx.conn.send({"context_id": ctx.id, "cancel": True})
            except Exception as exc:  # best effort: the socket may already be gone
                logger.debug("Cartesia TTS: cancelling context %s failed: %r", ctx.id, exc)


# --------------------------------------------------------------------------------- STT
@register_provider(
    "stt",
    "cartesia",
    description="Cartesia Ink: streaming STT with semantic turn events (turn.start/eager_end/end)",
    default_model="ink-2",
    models=("ink-2", "ink-preview", "ink-whisper"),
    env=("CARTESIA_API_KEY",),
    extra=None,
    requires=("websockets",),
    local=False,
)
class CartesiaSTT(STT):
    """Cartesia Ink streaming speech-to-text.

    With ``turn_detection=True`` (default for ``ink-2`` / ``ink-preview``) the stream uses
    ``/stt/turns/websocket``, where the model decides when the user's turn starts and
    ends. Events: ``turn.start`` -> ``START_OF_SPEECH``; ``turn.update`` ->
    ``INTERIM_TRANSCRIPT`` (cumulative text of the turn); ``turn.eager_end`` ->
    ``EAGER_END_OF_TURN``; ``turn.resume`` -> ``TURN_RESUMED``; ``turn.end`` ->
    ``FINAL_TRANSCRIPT`` + ``END_OF_SPEECH`` + ``END_OF_TURN``. That endpoint has no
    finalize command, so :meth:`STTStream.flush` only sends the buffered audio; use it in a
    cascade *without* a VAD and let Ink drive the turns.

    With ``turn_detection=False`` (always for ``ink-whisper``) the stream uses
    ``/stt/websocket`` for external endpointing: interim/final transcripts with word
    timings and detected language, and :meth:`STTStream.flush` sends ``finalize`` (fast
    finals when your VAD / turn detector ends the turn). If a flush had nothing to
    finalize, an empty ``FINAL_TRANSCRIPT`` acknowledges it. :meth:`transcribe` always uses
    this mode.

    Args:
        model: ``ink-2`` (default), ``ink-preview``, ``ink-whisper``.
        api_key: defaults to ``$CARTESIA_API_KEY``.
        language: language hint, only sent to ``ink-whisper`` (Ink-2 detects it).
        sample_rate: rate of the PCM sent to Cartesia (input is resampled to it).
        turn_detection: ``None`` = auto (on for Ink-2 models, off for ``ink-whisper``).
        turn_start_threshold / turn_eager_end_threshold / turn_end_threshold /
            turn_end_timeout_ms: turn detection tuning (see the Cartesia docs); ``None``
            keeps the API defaults.
        keyterms: up to 100 terms (1200 characters) to boost.
        min_volume / max_silence_duration_secs: ``ink-whisper`` endpointing options.
        chunk_duration: audio is sent in chunks of about this many seconds.
        base_url: API base URL (``wss://`` is derived from it).
        api_version: ``Cartesia-Version`` to pin.
        connect_timeout: connection / handshake timeout in seconds.
        close_timeout: how long to wait for the last results after the input ends.
    """

    provider = "cartesia"

    def __init__(
        self,
        *,
        model: str = "ink-2",
        api_key: str | None = None,
        language: str | None = None,
        sample_rate: int = 16_000,
        turn_detection: bool | None = None,
        turn_start_threshold: float | None = None,
        turn_eager_end_threshold: float | None = None,
        turn_end_threshold: float | None = None,
        turn_end_timeout_ms: float | None = None,
        keyterms: Sequence[str] | None = None,
        min_volume: float | None = None,
        max_silence_duration_secs: float | None = None,
        chunk_duration: float = 0.05,
        base_url: str = DEFAULT_BASE_URL,
        api_version: str = API_VERSION,
        connect_timeout: float = 10.0,
        close_timeout: float = 5.0,
    ) -> None:
        whisper = model.startswith("ink-whisper")
        turns = not whisper if turn_detection is None else turn_detection
        if turns and whisper:
            raise ConfigurationError(
                f"{model} has no turn detection; use turn_detection=False or model='ink-2'"
            )
        if isinstance(keyterms, str):
            raise ConfigurationError("keyterms must be a sequence of strings, not a string")
        super().__init__(
            model=model,
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=True,
                word_timestamps=not turns,
                end_of_turn=turns,
                language_detection=not turns,
            ),
            sample_rate=sample_rate,
            language=language,
        )
        self._api_key = _resolve_api_key(api_key)
        self.turn_detection = turns
        self.turn_options = {
            "turn_start_threshold": turn_start_threshold,
            "turn_eager_end_threshold": turn_eager_end_threshold,
            "turn_end_threshold": turn_end_threshold,
            "turn_end_timeout_ms": turn_end_timeout_ms,
        }
        self.keyterms = list(keyterms or ())
        self.min_volume = min_volume
        self.max_silence_duration_secs = max_silence_duration_secs
        self.chunk_duration = chunk_duration
        self.base_url = base_url.rstrip("/")
        self.api_version = api_version
        self.connect_timeout = connect_timeout
        self.close_timeout = close_timeout

    def _headers(self) -> dict[str, str]:
        return _headers(self._api_key, self.api_version)

    def _ws_url(self, *, turns: bool, language: str | None) -> str:
        params: list[tuple[str, str]] = [
            ("model", self.model),
            ("encoding", "pcm_s16le"),
            ("sample_rate", str(self.sample_rate)),
            ("cartesia_version", self.api_version),
        ]
        if turns:
            params += [(k, _param(v)) for k, v in self.turn_options.items() if v is not None]
            path = "/stt/turns/websocket"
        else:
            if language and self.model.startswith("ink-whisper"):
                params.append(("language", language))
            if self.min_volume is not None:
                params.append(("min_volume", _param(self.min_volume)))
            if self.max_silence_duration_secs is not None:
                params.append(("max_silence_duration_secs", _param(self.max_silence_duration_secs)))
            path = "/stt/websocket"
        params += [("keyterm", term) for term in self.keyterms]
        return f"{_ws_base(self.base_url)}{path}?{urlencode(params)}"

    def _create_stream(self, *, language: str | None) -> STTStream:
        return _CartesiaSTTStream(self, language=language, turns=self.turn_detection)

    async def _recognize(self, audio: AudioFrame, *, language: str | None) -> Transcript:
        """Batch recognition over the finalize endpoint (``finalize`` + ``close``)."""
        stream = _CartesiaSTTStream(self, language=language, turns=False)
        texts: list[str] = []
        words: list[WordTiming] = []
        detected: str | None = None
        try:
            stream.push_audio(audio)
            stream.end_input()
            async for ev in stream:
                if ev.type == STTEventType.FINAL_TRANSCRIPT and ev.transcript:
                    if ev.transcript.text:
                        texts.append(ev.transcript.text)
                    words += ev.transcript.words or []
                    detected = detected or ev.transcript.language
        finally:
            await stream.aclose()
        return Transcript(text=" ".join(texts), language=detected or language, words=words or None)


class _CartesiaSTTStream(STTStream):
    def __init__(self, stt: CartesiaSTT, *, language: str | None, turns: bool) -> None:
        self._turns = turns
        self._closing = False
        self._segment = new_id("turn_" if turns else "seg_")
        self._finalize_pending = 0
        self._finals_since_finalize = 0
        super().__init__(stt, language=language)

    async def _run(self) -> None:
        stt: CartesiaSTT = self._stt  # type: ignore[assignment]
        url = stt._ws_url(turns=self._turns, language=self._language)
        ws = await _ws_connect(url, stt._headers(), open_timeout=stt.connect_timeout)
        sender = asyncio.create_task(self._send_loop(ws), name="cartesia-stt-send")
        receiver = asyncio.create_task(self._recv_loop(ws), name="cartesia-stt-recv")
        try:
            await asyncio.wait((sender, receiver), return_when=asyncio.FIRST_COMPLETED)
            await _raise_task_error(receiver, sender)
            if not sender.done():
                raise _closed_error("STT")  # the server ended the stream while audio was flowing
            try:  # all audio sent and `close` requested: collect the last results
                async with asyncio.timeout(stt.close_timeout):
                    await receiver
            except TimeoutError:
                logger.warning(
                    "Cartesia STT: no end of stream %.1fs after close", stt.close_timeout
                )
        finally:
            await cancel_and_wait(sender, receiver)
            await _close_ws(ws)

    async def _send_loop(self, ws: ClientConnection) -> None:
        from websockets.exceptions import ConnectionClosed

        stt: CartesiaSTT = self._stt  # type: ignore[assignment]
        min_bytes = max(2, round(stt.sample_rate * stt.chunk_duration) * 2)
        buf = bytearray()
        try:
            async for item in self._input:
                if self.is_flush(item):
                    if buf:
                        await ws.send(bytes(buf))
                        buf.clear()
                    if not self._turns:
                        self._finalize_pending += 1
                        await ws.send("finalize")
                    continue
                assert isinstance(item, AudioFrame)
                buf += item.data
                if len(buf) >= min_bytes:
                    await ws.send(bytes(buf))
                    buf.clear()
            if buf:
                await ws.send(bytes(buf))
            self._closing = True
            await ws.send(json.dumps({"type": "close"}) if self._turns else "close")
        except ConnectionClosed as exc:
            raise _closed_error("STT", exc) from exc

    async def _recv_loop(self, ws: ClientConnection) -> None:
        from websockets.exceptions import ConnectionClosed

        try:
            async for raw in ws:
                msg = _parse(raw, "STT")
                if msg is None:
                    continue
                if msg.get("type") == "error":
                    raise _api_error(msg)
                if self._turns:
                    self._on_turn_event(msg)
                elif self._on_transcript_event(msg):
                    return  # "done": every result has been delivered
        except ConnectionClosed as exc:
            if not self._closing:
                raise _closed_error("STT", exc) from exc
            return
        if not self._closing:
            raise _closed_error("STT")

    # ------------------------------------------------------------ /stt/turns
    def _on_turn_event(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        text = str(msg.get("transcript") or "").strip()
        seg = self._segment
        if kind == "turn.start":
            self._emit(STTEvent(STTEventType.START_OF_SPEECH, segment_id=seg))
        elif kind == "turn.update":
            if text:
                transcript = Transcript(text, self._language)
                self._emit(STTEvent(STTEventType.INTERIM_TRANSCRIPT, transcript, seg))
        elif kind == "turn.eager_end":
            transcript = Transcript(text, self._language)
            self._emit(STTEvent(STTEventType.EAGER_END_OF_TURN, transcript, seg))
        elif kind == "turn.resume":
            self._emit(STTEvent(STTEventType.TURN_RESUMED, segment_id=seg))
        elif kind == "turn.end":
            transcript = Transcript(text, self._language)
            self._emit(STTEvent(STTEventType.FINAL_TRANSCRIPT, transcript, seg))
            self._emit(STTEvent(STTEventType.END_OF_SPEECH, segment_id=seg))
            self._emit(STTEvent(STTEventType.END_OF_TURN, transcript, seg))
            self._segment = new_id("turn_")
        elif kind == "connected":
            logger.debug("Cartesia STT connected (request %s)", msg.get("request_id"))

    # ------------------------------------------------------------ /stt/websocket
    def _on_transcript_event(self, msg: dict[str, Any]) -> bool:
        kind = msg.get("type")
        if kind == "transcript":
            words = _stt_words(msg.get("words"))
            language = msg.get("language")
            transcript = Transcript(
                text=str(msg.get("text") or "").strip(),
                language=language if isinstance(language, str) else self._language,
                start_time=words[0].start if words else None,
                end_time=words[-1].end if words else None,
                words=words or None,
            )
            if msg.get("is_final"):
                if self._finalize_pending:
                    self._finals_since_finalize += 1
                self._emit(STTEvent(STTEventType.FINAL_TRANSCRIPT, transcript, self._segment))
                self._segment = new_id("seg_")
            elif transcript.text:
                self._emit(STTEvent(STTEventType.INTERIM_TRANSCRIPT, transcript, self._segment))
        elif kind == "flush_done":
            if self._finalize_pending:
                self._finalize_pending -= 1
                if not self._finals_since_finalize:  # nothing to finalize: still acknowledge
                    empty = Transcript("", self._language)
                    self._emit(STTEvent(STTEventType.FINAL_TRANSCRIPT, empty, self._segment))
                self._finals_since_finalize = 0
        elif kind == "done":
            return True
        return False
