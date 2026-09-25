"""Soniox: real-time speech-to-text (``stt-rt-v5``) over a raw WebSocket.

Registered component: ``("stt", "soniox")`` — :class:`SonioxSTT`
(``stt="soniox/stt-rt-v5"``, the default model).

Protocol (https://soniox.com/docs/stt/api-reference/websocket-api):

* connect to ``wss://stt-rt.soniox.com/transcribe-websocket`` (or a data-residency host:
  ``stt-rt.{eu,jp,in}.soniox.com``) and send one JSON config message that carries the
  API key, ``model``, ``audio_format`` (``pcm_s16le`` here), ``sample_rate``,
  ``language_hints``, ``context`` (custom terms), ``enable_endpoint_detection``...;
* stream binary audio; ``{"type": "finalize"}`` finalizes everything sent so far (answered
  by a ``<fin>`` token), ``{"type": "keepalive"}`` keeps an idle session open (Soniox
  closes it after 20 s without audio), and an empty frame ends the stream;
* responses carry ``tokens`` (sub-word pieces with ``start_ms`` / ``end_ms``,
  ``confidence``, ``is_final``, ``language``, ``speaker``). Final tokens are sent once;
  non-final tokens are re-sent (and may change) with every response. Endpoint detection
  adds an ``<end>`` token when the speaker finished; errors arrive as ``error_code`` /
  ``error_message``; the last response has ``finished: true``.

Only core dependencies are used (``websockets`` and ``httpx``, no SDK).
See ``docs/providers/soniox.md``.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import httpx
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed

from ..audio.frame import AudioFrame
from ..errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
    for_status,
)
from ..metrics import STTMetrics
from ..registry import register_provider
from ..stt import STT, STTCapabilities, STTEvent, STTEventType, STTStream, Transcript, WordTiming
from ..utils.aio import ChanClosed, cancel_and_wait
from ..utils.ids import new_id
from ..utils.log import logger
from ._ws import close_ws, raise_task_error, ws_connect

__all__ = ["SonioxSTT", "SonioxStream"]

PROVIDER = "soniox"
API_KEY_ENV = "SONIOX_API_KEY"
DEFAULT_MODEL = "stt-rt-v5"
MODELS = ("stt-rt-v5", "stt-rt-v4")  # v4 is an alias of v5 since 2026-06-30
WS_PATH = "/transcribe-websocket"
REGION_URLS = {
    "us": "wss://stt-rt.soniox.com",
    "eu": "wss://stt-rt.eu.soniox.com",
    "jp": "wss://stt-rt.jp.soniox.com",
    "in": "wss://stt-rt.in.soniox.com",
}
API_URLS = {
    "us": "https://api.soniox.com",
    "eu": "https://api.eu.soniox.com",
    "jp": "https://api.jp.soniox.com",
    "in": "https://api.in.soniox.com",
}
END_TOKEN = "<end>"
FIN_TOKEN = "<fin>"

_MSG_FINALIZE = json.dumps({"type": "finalize"})
_MSG_KEEPALIVE = json.dumps({"type": "keepalive"})
_NORMAL_CLOSE_CODES = frozenset({1000, 1001, 1005})


# ---------------------------------------------------------------------------- helpers
def _check_range(name: str, value: float | None, low: float, high: float) -> None:
    if value is not None and not low <= value <= high:
        raise ConfigurationError(f"{name} must be within [{low}, {high}], got {value}")


def _language_code(language: str | None) -> str | None:
    """``en-US`` -> ``en`` (Soniox language hints are ISO 639-1 codes)."""
    if not language:
        return None
    return language.replace("_", "-").split("-")[0].lower() or None


def _float(value: Any) -> float | None:
    try:
        return None if value is None or isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        return None


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _parse(message: str) -> dict[str, Any]:
    try:
        data = json.loads(message)
    except ValueError:
        logger.warning("Soniox: ignoring a non-JSON message: %.200s", message)
        return {}
    return data if isinstance(data, dict) else {}


def _text(tokens: Sequence[Mapping[str, Any]]) -> str:
    """Tokens are sub-word pieces that carry their own leading spaces: concatenate them."""
    return "".join(str(t.get("text") or "") for t in tokens).strip()


def _words(tokens: Sequence[Mapping[str, Any]]) -> list[WordTiming] | None:
    """Merge sub-word tokens into words (a token with a leading space starts a new word).

    Times are milliseconds from the stream start -> seconds; the word's confidence is its
    least confident token's. ``None`` when the tokens carry no timestamps.
    """
    words: list[WordTiming] = []
    for token in tokens:
        text = str(token.get("text") or "")
        start, end = _float(token.get("start_ms")), _float(token.get("end_ms"))
        if start is None or end is None:
            return None
        conf = _float(token.get("confidence"))
        if words and not text[:1].isspace():
            word = words[-1]
            word.word += text
            word.end = max(word.end, end / 1000.0)
            if conf is not None:
                word.confidence = conf if word.confidence is None else min(word.confidence, conf)
            continue
        if text.strip():
            words.append(WordTiming(text.strip(), start / 1000.0, end / 1000.0, conf))
    return words or None


def _mean_confidence(tokens: Sequence[Mapping[str, Any]]) -> float | None:
    values = [c for c in (_float(t.get("confidence")) for t in tokens) if c is not None]
    return sum(values) / len(values) if values else None


def _language(tokens: Sequence[Mapping[str, Any]]) -> str | None:
    """The most common token language (set with language identification)."""
    counts = Counter(t["language"] for t in tokens if isinstance(t.get("language"), str))
    return counts.most_common(1)[0][0] if counts else None


def _code_error(code: int | None, kind: str, reason: str) -> ProviderError:
    """Map a Soniox ``error_code`` (HTTP-like) / ``error_type`` to a library error."""
    parts = [p for p in (kind, reason) if p]
    detail = f"Soniox error {code}" if code is not None else "Soniox error"
    message = f"{detail}: {': '.join(parts)}" if parts else detail
    if code in (401, 403):  # unauthenticated, permission_denied, temp_api_key_session_expired
        return AuthenticationError(message, provider=PROVIDER, status_code=code)
    if code == 429:
        return RateLimitError(message, provider=PROVIDER, status_code=code)
    if code == 408:
        return ProviderTimeoutError(message, provider=PROVIDER, status_code=code)
    if code == 413:  # max_duration_reached: a new session works
        return ProviderConnectionError(message, provider=PROVIDER, status_code=code)
    if code in (500, 502, 503, 504):
        return ProviderError(message, provider=PROVIDER, status_code=code, retryable=True)
    if code is None:
        return ProviderConnectionError(message, provider=PROVIDER)
    # 400 invalid_request / model_not_available, 402 balance / budget exhausted
    return ProviderError(message, provider=PROVIDER, status_code=code)


def _http_error(status: int, detail: str) -> ProviderError:
    message = f"Soniox returned HTTP {status}" + (f": {detail}" if detail else "")
    return for_status(status, message, provider=PROVIDER)


def _error_detail(body: bytes | bytearray | str | None) -> str:
    if not body:
        return ""
    text = body if isinstance(body, str) else bytes(body).decode("utf-8", "replace")
    try:
        data = json.loads(text)
    except ValueError:
        return text.strip()[:500]
    if not isinstance(data, dict):
        return text.strip()[:500]
    parts = [str(data[k]) for k in ("error_type", "message", "error_message") if data.get(k)]
    return ": ".join(parts) or text.strip()[:500]


# -------------------------------------------------------------------------------- STT
@register_provider(
    "stt",
    "soniox",
    description="Soniox real-time STT (stt-rt-v5): 60+ languages, endpoint detection, finalize",
    default_model=DEFAULT_MODEL,
    models=MODELS,
    env=(API_KEY_ENV,),
    requires=("websockets", "httpx"),
    local=False,
)
class SonioxSTT(STT):
    """Soniox real-time speech-to-text (WebSocket API).

    Events (:class:`~voice_agent_next.stt.STTEventType`), per utterance:

    * the first token (final or not) -> ``START_OF_SPEECH`` (``transcript.start_time`` =
      the token's start, in stream seconds);
    * each response whose text changed -> ``INTERIM_TRANSCRIPT`` with the utterance so far
      (its final tokens + the current non-final ones);
    * an ``<end>`` token (endpoint detection) -> ``FINAL_TRANSCRIPT`` + ``END_OF_SPEECH``
      (``end_time`` = end of the last token) + ``END_OF_TURN`` (with ``end_of_turn=True``);
    * a ``<fin>`` token (answer to :meth:`~voice_agent_next.stt.STTStream.flush`, which
      sends ``finalize``) -> ``FINAL_TRANSCRIPT`` (possibly empty) + ``END_OF_SPEECH``, but
      never ``END_OF_TURN``: whoever flushed owns that decision.

    Two ways to run it in a cascade:

    * ``end_of_turn=True`` (default): Soniox's semantic endpoint detection ends the user's
      turn and the cascade commits it at once (no VAD needed; tune it with
      ``max_endpoint_delay_ms``, ``endpoint_sensitivity``...).
    * ``end_of_turn=False``: endpoint detection is off (unless ``enable_endpoint_detection``
      says otherwise) and the cascade's VAD / turn detector ends turns; its flush sends
      ``finalize`` and Soniox answers with the final tokens and ``<fin>`` within tens of ms.

    Args:
        model: ``stt-rt-v5`` (default; ``stt-rt-v4`` is an alias).
        api_key: Soniox API key or temporary key (default: ``SONIOX_API_KEY``).
        language: a language hint (``en-US`` -> ``en``).
        language_hints: languages expected in the audio (overrides ``language``).
        language_hints_strict: restrict recognition to ``language_hints``.
        language_identification: tag tokens with their language (``Transcript.language``).
        speaker_diarization: tag tokens with speakers (see :attr:`SonioxStream.final_tokens`).
        terms: custom vocabulary (``context.terms``): names, jargon, product words.
        context: free text about the audio (``context.text``) or a full ``context`` object.
        context_general: key/value facts (``{"domain": "Healthcare"}``, ``context.general``).
        end_of_turn: emit ``END_OF_TURN`` on ``<end>`` (see above).
        enable_endpoint_detection: override the default (``end_of_turn``).
        max_endpoint_delay_ms: hard ceiling (500-3000 ms, default 2000) before an endpoint.
        endpoint_sensitivity: -1..1 (default 0): higher ends turns sooner.
        endpoint_latency_adjustment_level: 0-3 (default 0): higher is faster, less accurate.
        client_reference_id: tag for the session (at most 256 characters).
        sample_rate: rate of the PCM sent (input is resampled to it).
        chunk_ms: audio chunk size sent upstream.
        keepalive_interval: seconds without audio before a ``keepalive`` is sent (Soniox
            closes sessions idle for 20 s).
        region: data-residency region ``us`` (default), ``eu``, ``jp`` or ``in``; keys are
            per region.
        base_url: WebSocket origin override (``wss://...``; also for tests).
        api_url: REST origin override (temporary keys).
        http_client: optional ``httpx.AsyncClient`` (not closed by :meth:`aclose`).
        connect_timeout: connection timeout in seconds.
        close_timeout: how long to wait for ``finished`` after the input ends.
        extra_config: more config fields (``translation``...), sent verbatim.
    """

    provider = PROVIDER

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        language: str | None = None,
        language_hints: Sequence[str] = (),
        language_hints_strict: bool | None = None,
        language_identification: bool | None = None,
        speaker_diarization: bool | None = None,
        terms: Sequence[str] = (),
        context: str | Mapping[str, Any] | None = None,
        context_general: Mapping[str, str] | None = None,
        end_of_turn: bool = True,
        enable_endpoint_detection: bool | None = None,
        max_endpoint_delay_ms: int | None = None,
        endpoint_sensitivity: float | None = None,
        endpoint_latency_adjustment_level: int | None = None,
        client_reference_id: str | None = None,
        sample_rate: int = 16_000,
        chunk_ms: int = 40,
        keepalive_interval: float = 5.0,
        region: Literal["us", "eu", "jp", "in"] | None = None,
        base_url: str | None = None,
        api_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        connect_timeout: float = 10.0,
        close_timeout: float = 5.0,
        extra_config: Mapping[str, Any] | None = None,
    ) -> None:
        for name, seq in (("language_hints", language_hints), ("terms", terms)):
            if isinstance(seq, str):
                raise ConfigurationError(f"{name} must be a sequence of strings, not a string")
        _check_range("max_endpoint_delay_ms", max_endpoint_delay_ms, 500, 3000)
        _check_range("endpoint_sensitivity", endpoint_sensitivity, -1.0, 1.0)
        _check_range("endpoint_latency_adjustment_level", endpoint_latency_adjustment_level, 0, 3)
        _check_range("chunk_ms", chunk_ms, 10, 1000)
        if not 8_000 <= sample_rate <= 96_000:
            raise ConfigurationError(f"sample_rate must be 8000-96000 Hz, got {sample_rate}")
        if keepalive_interval <= 0 or keepalive_interval >= 20:
            raise ConfigurationError("keepalive_interval must be within (0, 20) seconds")
        if client_reference_id is not None and len(client_reference_id) > 256:
            raise ConfigurationError("client_reference_id must be at most 256 characters")
        if region is not None and region not in REGION_URLS:
            raise ConfigurationError(f"region must be one of {sorted(REGION_URLS)}, got {region!r}")
        super().__init__(
            model=model or DEFAULT_MODEL,
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=True,
                word_timestamps=True,
                end_of_turn=end_of_turn,
                language_detection=bool(language_identification),
            ),
            sample_rate=sample_rate,
            language=language,
        )
        self._api_key = (api_key or os.environ.get(API_KEY_ENV) or "").strip()
        if not self._api_key:
            raise ConfigurationError(
                f"Soniox needs an API key: pass api_key=... or set {API_KEY_ENV}"
            )
        self.language_hints = list(language_hints)
        self.language_hints_strict = language_hints_strict
        self.language_identification = language_identification
        self.speaker_diarization = speaker_diarization
        self.terms = list(terms)
        self.context = context
        self.context_general = dict(context_general or {})
        self.enable_endpoint_detection = (
            end_of_turn if enable_endpoint_detection is None else enable_endpoint_detection
        )
        self.max_endpoint_delay_ms = max_endpoint_delay_ms
        self.endpoint_sensitivity = endpoint_sensitivity
        self.endpoint_latency_adjustment_level = endpoint_latency_adjustment_level
        self.client_reference_id = client_reference_id
        self.chunk_ms = chunk_ms
        self.keepalive_interval = keepalive_interval
        self.base_url = (base_url or REGION_URLS[region or "us"]).rstrip("/")
        self.api_url = (api_url or API_URLS[region or "us"]).rstrip("/")
        self.connect_timeout = connect_timeout
        self.close_timeout = close_timeout
        self.extra_config = dict(extra_config or {})
        self._http = http_client
        self._owns_http = http_client is None

    # ---------------------------------------------------------------- requests
    @property
    def url(self) -> str:
        """The WebSocket URL (credentials go in the config message, not the URL)."""
        return f"{self.base_url}{WS_PATH}"

    def _context(self) -> dict[str, Any] | None:
        context: dict[str, Any] = {}
        if isinstance(self.context, Mapping):
            context |= dict(self.context)
        elif self.context:
            context["text"] = self.context
        if self.terms:
            context["terms"] = [*context.get("terms", ()), *self.terms]
        if self.context_general:
            general = [{"key": k, "value": v} for k, v in self.context_general.items()]
            context["general"] = [*context.get("general", ()), *general]
        return context or None

    def config(self, language: str | None = None) -> dict[str, Any]:
        """The config message, without the API key (``None`` values are not sent)."""
        hints = list(self.language_hints)
        if not hints:
            code = _language_code(language or self.language)
            hints = [code] if code else []
        config: dict[str, Any] = {
            "model": self.model,
            "audio_format": "pcm_s16le",
            "sample_rate": self.sample_rate,
            "num_channels": 1,
            "language_hints": hints or None,
            "language_hints_strict": self.language_hints_strict,
            "enable_language_identification": self.language_identification,
            "enable_speaker_diarization": self.speaker_diarization,
            "context": self._context(),
            "enable_endpoint_detection": self.enable_endpoint_detection,
            "max_endpoint_delay_ms": self.max_endpoint_delay_ms,
            "endpoint_sensitivity": self.endpoint_sensitivity,
            "endpoint_latency_adjustment_level": self.endpoint_latency_adjustment_level,
            "client_reference_id": self.client_reference_id,
        }
        config |= self.extra_config
        return {k: v for k, v in config.items() if v is not None}

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            timeout = httpx.Timeout(30.0, connect=self.connect_timeout)
            self._http = httpx.AsyncClient(timeout=timeout)
            self._owns_http = True
        return self._http

    async def create_temporary_api_key(
        self,
        *,
        expires_in_seconds: int = 60,
        single_use: bool | None = None,
        max_session_duration_seconds: int | None = None,
        client_reference_id: str | None = None,
    ) -> str:
        """A temporary key for real-time transcription (``POST /v1/auth/temporary-api-key``).

        Hand it to a client that must not see your key: ``SonioxSTT(api_key=temp_key)``.
        """
        _check_range("expires_in_seconds", expires_in_seconds, 1, 3600)
        _check_range("max_session_duration_seconds", max_session_duration_seconds, 1, 18_000)
        body: dict[str, Any] = {
            "usage_type": "transcribe_websocket",
            "expires_in_seconds": expires_in_seconds,
            "single_use": single_use,
            "max_session_duration_seconds": max_session_duration_seconds,
            "client_reference_id": client_reference_id,
        }
        try:
            response = await self._client().post(
                f"{self.api_url}/v1/auth/temporary-api-key",
                json={k: v for k, v in body.items() if v is not None},
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Soniox request timed out: {exc!r}", provider=PROVIDER
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderConnectionError(
                f"Soniox request failed: {exc!r}", provider=PROVIDER
            ) from exc
        if response.status_code >= 400:
            raise _http_error(response.status_code, _error_detail(response.content))
        try:
            key = response.json().get("api_key")
        except (ValueError, AttributeError):
            key = None
        if not isinstance(key, str) or not key:
            raise ProviderError("Soniox returned no temporary API key", provider=PROVIDER)
        return key

    def _create_stream(self, *, language: str | None) -> STTStream:
        return SonioxStream(self, language=language)

    def stream(self, *, language: str | None = None) -> SonioxStream:
        """Open a streaming session (see :class:`SonioxStream`)."""
        stream = super().stream(language=language)
        assert isinstance(stream, SonioxStream)
        return stream

    async def aclose(self) -> None:
        if self._http is not None and self._owns_http:
            http, self._http = self._http, None
            await http.aclose()


async def _connect(stt: SonioxSTT) -> ClientConnection:
    return await ws_connect(
        stt.url,
        provider=PROVIDER,
        target="the Soniox real-time API",
        name="Soniox",
        http_error=lambda r: _http_error(r.status_code, _error_detail(r.body)),
        open_timeout=stt.connect_timeout,
        compression=None,  # PCM audio does not compress; save the CPU
    )


class SonioxStream(STTStream):
    """One Soniox real-time session (see :class:`SonioxSTT` for the events).

    :attr:`final_tokens` holds the raw tokens (``speaker``, ``language``, timings...) of the
    last ``FINAL_TRANSCRIPT``; :attr:`final_audio_proc_ms` / :attr:`total_audio_proc_ms`
    are Soniox's progress counters.
    """

    def __init__(self, stt: SonioxSTT, *, language: str | None) -> None:
        self._snx = stt
        self._chunk_bytes = round(stt.sample_rate * stt.chunk_ms / 1000) * 2
        self._buf = bytearray()
        self._closing = False
        self._finished = False
        self._server_error: ProviderError | None = None
        self._segment_id = new_id("seg_")
        self._turn_active = False
        self._final: list[dict[str, Any]] = []  # final tokens of the open utterance
        self._partial: list[dict[str, Any]] = []  # current non-final tokens
        self._last_text = ""
        self._inflight = 0  # finalize messages not answered by <fin> yet
        self._audio_since_finalize = False
        self.final_tokens: list[dict[str, Any]] = []
        self.final_audio_proc_ms = 0
        self.total_audio_proc_ms = 0
        super().__init__(stt, language=language)

    # ------------------------------------------------------------------- plumbing
    async def _run(self) -> None:
        stt = self._snx
        ws = await _connect(stt)
        receiver: asyncio.Task[None] | None = None
        sender: asyncio.Task[None] | None = None
        try:
            try:
                await ws.send(_json({"api_key": stt._api_key, **stt.config(self._language)}))
            except ConnectionClosed as exc:
                raise self._close_error(exc) from exc
            receiver = asyncio.create_task(self._recv_loop(ws), name="soniox-stt-recv")
            sender = asyncio.create_task(self._send_loop(ws), name="soniox-stt-send")
            await asyncio.wait({receiver, sender}, return_when=asyncio.FIRST_COMPLETED)
            raise_task_error(receiver, sender)
            if not sender.done():  # the server ended the session while audio was flowing
                raise self._server_error or ProviderConnectionError(
                    "Soniox closed the session unexpectedly", provider=PROVIDER
                )
            # end of stream sent: Soniox finalizes the rest, then sends finished: true
            await asyncio.wait({receiver}, timeout=stt.close_timeout)
            raise_task_error(receiver, sender)
            if not receiver.done():
                logger.warning("Soniox: no 'finished' %.1fs after the audio", stt.close_timeout)
            self._finish_session()
        finally:
            self._closing = True
            await cancel_and_wait(*(t for t in (sender, receiver) if t is not None))
            await close_ws(ws)

    async def _send_loop(self, ws: ClientConnection) -> None:
        try:
            while True:
                try:
                    async with asyncio.timeout(self._snx.keepalive_interval):
                        item = await self._input.recv()
                except TimeoutError:
                    await ws.send(_MSG_KEEPALIVE)
                    continue
                except ChanClosed:
                    break
                if self.is_flush(item):
                    if self._buf:
                        await ws.send(bytes(self._buf))
                        self._buf.clear()
                    if self._audio_since_finalize:
                        self._audio_since_finalize = False
                        self._inflight += 1
                        await ws.send(_MSG_FINALIZE)
                    elif not self._inflight:
                        # nothing sent since the last <fin>: everything is final already
                        self._answer_flush()
                    # else: the <fin> in flight comes after this flush and answers it
                    continue
                assert isinstance(item, AudioFrame)
                self._buf += item.data
                self._audio_since_finalize = True
                while len(self._buf) >= self._chunk_bytes:
                    await ws.send(bytes(self._buf[: self._chunk_bytes]))
                    del self._buf[: self._chunk_bytes]
            if self._buf:
                await ws.send(bytes(self._buf))
                self._buf.clear()
            self._closing = True
            await ws.send(b"")  # end of stream
        except ConnectionClosed as exc:
            raise self._server_error or self._close_error(exc) from exc

    async def _recv_loop(self, ws: ClientConnection) -> None:
        try:
            async for message in ws:
                if isinstance(message, str):
                    self._on_message(_parse(message))
                    if self._finished:
                        return
        except ConnectionClosed as exc:
            code = exc.rcvd.code if exc.rcvd is not None else None
            if self._server_error is not None:
                raise self._server_error from exc
            if not (self._closing and (code is None or code in _NORMAL_CLOSE_CODES)):
                raise self._close_error(exc) from exc
            return
        if self._server_error is not None:
            raise self._server_error
        if not self._closing:
            raise ProviderConnectionError(
                "Soniox closed the session unexpectedly", provider=PROVIDER
            )

    @staticmethod
    def _close_error(exc: ConnectionClosed) -> ProviderError:
        frame = exc.rcvd
        if frame is None:
            error: ProviderError = ProviderConnectionError(
                "Soniox connection lost (no close frame)", provider=PROVIDER
            )
        else:
            error = ProviderConnectionError(
                f"Soniox closed the connection ({frame.code}): {frame.reason}",
                provider=PROVIDER,
                status_code=frame.code,
            )
        error.__cause__ = exc
        return error

    # -------------------------------------------------------------------- events
    def _on_message(self, message: dict[str, Any]) -> None:
        if message.get("error_code") is not None or message.get("error_message"):
            code = message.get("error_code")
            self._server_error = _code_error(
                int(code) if isinstance(code, (int, float)) else None,
                str(message.get("error_type") or ""),
                str(message.get("error_message") or ""),
            )
            raise self._server_error
        for key in ("final_audio_proc_ms", "total_audio_proc_ms"):
            value = message.get(key)
            if isinstance(value, (int, float)):
                setattr(self, key, int(value))
        tokens = message.get("tokens")
        partial: list[dict[str, Any]] = []
        for token in tokens if isinstance(tokens, list) else ():
            if not isinstance(token, dict):
                continue
            text = token.get("text")
            if text == END_TOKEN:
                self._end_utterance(endpoint=True)
            elif text == FIN_TOKEN:
                self._inflight = max(0, self._inflight - 1)
                self._end_utterance(endpoint=False)
            elif token.get("is_final"):
                self._final.append(token)
            else:
                partial.append(token)
        self._partial = partial
        self._update_interim()
        if message.get("finished"):
            self._finished = True

    def _update_interim(self) -> None:
        tokens = [*self._final, *self._partial]
        text = _text(tokens)
        if not text or text == self._last_text:
            return
        self._begin_turn(_float(tokens[0].get("start_ms")))
        self._last_text = text
        self._event(STTEventType.INTERIM_TRANSCRIPT, self._transcript(tokens))

    def _begin_turn(self, start_ms: float | None) -> None:
        if not self._turn_active:
            self._turn_active = True
            self._segment_id = new_id("seg_")
            start = start_ms / 1000.0 if start_ms is not None else None
            self._event(STTEventType.START_OF_SPEECH, Transcript("", start_time=start))

    def _transcript(self, tokens: Sequence[Mapping[str, Any]]) -> Transcript:
        words = _words(tokens)
        return Transcript(
            text=_text(tokens),
            language=_language(tokens) or _language_code(self._language),
            confidence=_mean_confidence(tokens),
            start_time=words[0].start if words else None,
            end_time=words[-1].end if words else None,
            words=words,
        )

    def _end_utterance(self, *, endpoint: bool) -> None:
        """``<end>`` (``endpoint``) or ``<fin>``: the final tokens so far form an utterance."""
        tokens, self._final = self._final, []
        if not tokens and not self._turn_active and endpoint:
            return  # an endpoint with nothing to report
        transcript = self._transcript(tokens)
        if transcript.text:
            self._begin_turn(_float(tokens[0].get("start_ms")))
        self.final_tokens = tokens
        self._event(STTEventType.FINAL_TRANSCRIPT, transcript)
        if self._turn_active:
            self._event(STTEventType.END_OF_SPEECH, transcript)
        flushed = not endpoint or self._inflight > 0
        if self._snx.capabilities.end_of_turn and endpoint and transcript.text and not flushed:
            self._event(STTEventType.END_OF_TURN, transcript)
        self._turn_active = False
        self._last_text = ""
        if endpoint:
            self._report_usage()

    def _answer_flush(self) -> None:
        """A flush with nothing left to finalize: acknowledge it with an empty final."""
        self._event(STTEventType.FINAL_TRANSCRIPT, Transcript("", _language_code(self._language)))

    def _finish_session(self) -> None:
        # anything still pending when the session ended (all tokens are final by then)
        if self._final or self._partial or self._turn_active:
            self._final, self._partial = [*self._final, *self._partial], []
            self._end_utterance(endpoint=False)
        elif self._inflight:
            self._answer_flush()
        self._inflight = 0

    def _event(self, kind: STTEventType, transcript: Transcript | None = None) -> None:
        self._emit(STTEvent(kind, transcript, self._segment_id))

    def _report_usage(self) -> None:
        # The base class reports usage when a flush is answered; utterances Soniox ends
        # on its own are never flushed, so report their audio here.
        if self._audio_duration > 0:
            self._stt.emit(
                "metrics",
                STTMetrics(
                    provider=self._stt.provider,
                    model=self._stt.model,
                    request_id=self._request_id,
                    audio_duration=self._audio_duration,
                    streamed=True,
                ),
            )
            self._audio_duration = 0.0
