"""Deepgram: Nova-3 / Flux streaming STT and Aura-2 TTS over raw WebSockets.

Registered components:

* ``("stt", "deepgram")`` — :class:`DeepgramSTT`. The model picks the protocol:

  * ``nova-3`` (default) and the other ``/v1/listen`` models (``nova-3-medical``,
    ``nova-2``...): interim and final results, ``speech_final`` endpointing,
    ``SpeechStarted`` / ``UtteranceEnd`` events; :meth:`STTStream.flush` sends
    ``Finalize``; ``KeepAlive`` is sent while no audio flows.
  * ``flux-general-en`` / ``flux-general-multi`` (``/v2/listen``): conversational STT
    with model-integrated end-of-turn detection (``StartOfTurn``, ``Update``,
    ``EagerEndOfTurn``, ``TurnResumed``, ``EndOfTurn``); :meth:`STTStream.flush` sends
    ``ForceEndTurn``.

* ``("tts", "deepgram")`` — :class:`DeepgramTTS`: Aura-2 voices (``aura-2-thalia-en``...)
  over the ``/v1/speak`` WebSocket (native text-in streaming with ``Speak`` / ``Flush`` /
  ``Clear``) and the REST endpoint for :meth:`TTS.synthesize`.

Only core dependencies are used (``websockets`` and ``httpx``, no SDK). Requests are
authenticated with ``Authorization: Token <key>``, the key coming from ``api_key=...``
or ``DEEPGRAM_API_KEY``. See ``docs/providers/deepgram.md``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlencode

import httpx
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
)
from websockets.protocol import State

from ..audio.buffer import FrameChunker
from ..audio.frame import AudioFrame
from ..errors import (
    AuthenticationError,
    ConfigurationError,
    MissingAPIKeyError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
    for_status,
)
from ..metrics import STTMetrics
from ..registry import register_provider
from ..stt import STT, STTCapabilities, STTEvent, STTEventType, STTStream, Transcript, WordTiming
from ..tts import TTS, ChunkedStream, NormalizeOption, SynthesizeStream, TTSCapabilities
from ..utils.aio import BackgroundTasks, ChanClosed, cancel_and_wait
from ..utils.clock import now
from ..utils.ids import new_id
from ..utils.log import logger
from ._options import deprecated
from ._ws import close_ws, ws_connect

__all__ = ["DeepgramSTT", "DeepgramTTS"]

PROVIDER = "deepgram"
API_KEY_ENV = "DEEPGRAM_API_KEY"
DEFAULT_BASE_URL = "https://api.deepgram.com"

_MSG_KEEPALIVE = json.dumps({"type": "KeepAlive"})
_MSG_FINALIZE = json.dumps({"type": "Finalize"})
_MSG_CLOSE_STREAM = json.dumps({"type": "CloseStream"})
_MSG_FORCE_END_TURN = json.dumps({"type": "ForceEndTurn"})
_MSG_FLUSH = json.dumps({"type": "Flush"})
_MSG_CLEAR = json.dumps({"type": "Clear"})

_FLUX_SAMPLE_RATES = (8_000, 16_000, 24_000, 44_100, 48_000)
_AURA_SAMPLE_RATES = (8_000, 16_000, 24_000, 32_000, 48_000)
_AURA_MODEL = re.compile(r"(?P<family>aura(?:-\d+)?)-(?P<voice>[a-z]+)-(?P<lang>[a-z]{2})")
# Close codes that end a stream normally once we asked the server to close it (Flux closes
# without a status code, i.e. 1005 / 1006 on the client side).
_NORMAL_CLOSE_CODES = frozenset({1000, 1001, 1005, 1006})
# Aura WebSocket limits: 20 Flush messages per 60 s, 60 min per connection.
_FLUSH_LIMIT = 20
_FLUSH_WINDOW = 60.0
_MAX_CONNECTION_AGE = 55 * 60.0


# ---------------------------------------------------------------------------- helpers
def _resolve_api_key(api_key: str | None) -> str:
    key = (api_key or os.environ.get(API_KEY_ENV) or "").strip()
    if not key:
        raise MissingAPIKeyError(
            f"Deepgram needs an API key: pass api_key=... or set {API_KEY_ENV}"
        )
    return key


def _with_scheme(base_url: str, *, websocket: bool) -> str:
    """``https://host`` <-> ``wss://host`` (``http`` <-> ``ws`` for local/self-hosted servers)."""
    scheme, sep, rest = base_url.strip().rstrip("/").partition("://")
    if not sep:
        scheme, rest = "https", scheme
    secure = scheme.lower() in ("https", "wss")
    if websocket:
        return f"{'wss' if secure else 'ws'}://{rest}"
    return f"{'https' if secure else 'http'}://{rest}"


def _query_items(params: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Query parameters: ``None`` dropped, booleans lowercased, sequences repeated."""
    items: list[tuple[str, str]] = []
    for key, value in params.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        for v in values:
            if v is None:
                continue
            items.append((key, ("true" if v else "false") if isinstance(v, bool) else str(v)))
    return items


def _as_tuple(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return (value,) if isinstance(value, str) else tuple(value)


def _float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _words(items: Any) -> list[WordTiming] | None:
    """Word timings from a Deepgram ``words`` array (``None`` if absent or untimed)."""
    if not isinstance(items, list) or not items:
        return None
    words: list[WordTiming] = []
    for item in items:
        if not isinstance(item, dict):
            return None
        start, end = _float(item.get("start")), _float(item.get("end"))
        if start is None or end is None:
            return None
        text = item.get("punctuated_word") or item.get("word") or ""
        words.append(WordTiming(str(text), start, end, _float(item.get("confidence"))))
    return words


def _parse(message: str) -> dict[str, Any]:
    try:
        data = json.loads(message)
    except ValueError:
        logger.warning("Deepgram: ignoring a non-JSON message: %.200s", message)
        return {}
    return data if isinstance(data, dict) else {}


def _detail(data: Mapping[str, Any]) -> str:
    """The message of a Deepgram error object (legacy ``err_msg`` or modern ``message``)."""
    for key in ("err_msg", "message", "description", "details", "reason"):
        if data.get(key):
            return str(data[key])
    return json.dumps(data)[:500]


def _error_detail(
    body: bytes | bytearray | str | None, headers: Mapping[str, str] | None = None
) -> str:
    """Human-readable error from a ``dg-error`` header or an error JSON body."""
    if headers is not None and headers.get("dg-error"):
        return str(headers["dg-error"])
    if not body:
        return ""
    text = body if isinstance(body, str) else bytes(body).decode("utf-8", "replace")
    try:
        data = json.loads(text)
    except ValueError:
        return text.strip()[:500]
    return _detail(data) if isinstance(data, dict) else text.strip()[:500]


def _http_error(status: int, detail: str) -> ProviderError:
    message = f"Deepgram returned HTTP {status}" + (f": {detail}" if detail else "")
    return for_status(status, message, provider=PROVIDER)


def _close_error(exc: ConnectionClosed, what: str) -> ProviderError:
    """Map a WebSocket close (``1008 DATA-0000``, ``1011 NET-0001``...) to a library error."""
    frame = exc.rcvd
    if frame is None:
        detail, code = "no close frame", None
    else:
        code = int(frame.code)
        detail = f"code {code}" + (f" {frame.reason}" if frame.reason else "")
    if code in (1003, 1008, 1009):  # unsupported / undecodable / oversized input
        error: ProviderError = ProviderError(
            f"Deepgram {what} rejected the input ({detail})",
            provider=PROVIDER,
            status_code=code,
        )
    else:
        error = ProviderConnectionError(
            f"Deepgram {what} connection closed unexpectedly ({detail})",
            provider=PROVIDER,
            status_code=code,
        )
    error.__cause__ = exc
    return error


def _task_error(tasks: Sequence[asyncio.Task[None]], what: str) -> BaseException | None:
    """The first failure among finished ``tasks`` (connection closures mapped)."""
    for task in tasks:
        if task.done() and not task.cancelled():
            exc = task.exception()
            if isinstance(exc, ConnectionClosed):
                return _close_error(exc, what)
            if exc is not None:
                return exc
    return None


async def _open_websocket(url: str, api_key: str, *, timeout: float, what: str) -> ClientConnection:
    ws = await ws_connect(
        url,
        provider=PROVIDER,
        target=f"the Deepgram {what} API",
        name="Deepgram",
        http_error=lambda r: _http_error(r.status_code, _error_detail(r.body, r.headers)),
        headers={"Authorization": f"Token {api_key}"},
        open_timeout=timeout,
        compression=None,  # PCM audio does not compress; save the CPU
    )
    request_id = ws.response.headers.get("dg-request-id") if ws.response else None
    logger.debug("Deepgram %s connected (request_id=%s)", what, request_id)
    return ws


# -------------------------------------------------------------------------------- STT
@register_provider(
    "stt",
    "deepgram",
    description="Deepgram streaming STT: Nova-3, and Flux with built-in end-of-turn",
    default_model="nova-3",
    models=("nova-3", "nova-3-medical", "nova-2", "flux-general-en", "flux-general-multi"),
    env=(API_KEY_ENV,),
    requires=("websockets",),
    local=False,
)
class DeepgramSTT(STT):
    """Deepgram streaming speech-to-text (Nova-3 on ``/v1/listen``, Flux on ``/v2/listen``).

    Nova-3 events (:class:`~voice_agent_next.stt.STTEventType`):

    * ``SpeechStarted`` (or the first words) -> ``START_OF_SPEECH``
    * ``Results`` with ``is_final=false`` -> ``INTERIM_TRANSCRIPT``
    * ``Results`` with ``is_final=true`` -> ``FINAL_TRANSCRIPT`` (empty only when it
      answers a ``Finalize`` with nothing left to transcribe)
    * ``speech_final=true`` or ``UtteranceEnd`` -> ``END_OF_SPEECH``

    Flux events (``capabilities.end_of_turn=True``):

    * ``StartOfTurn`` -> ``START_OF_SPEECH`` + ``INTERIM_TRANSCRIPT``
    * ``Update`` -> ``INTERIM_TRANSCRIPT`` (only when the transcript changed)
    * ``EagerEndOfTurn`` -> ``EAGER_END_OF_TURN``; ``TurnResumed`` -> ``TURN_RESUMED``
    * ``EndOfTurn`` -> ``FINAL_TRANSCRIPT`` + ``END_OF_SPEECH`` + ``END_OF_TURN``, except
      that turns ended by our own :meth:`~voice_agent_next.stt.STTStream.flush`
      (``ForceEndTurn``, ``trigger="manual"``) get no ``END_OF_TURN``: the caller that
      flushed owns that decision.

    With Flux, run the cascade **without** a VAD: ``StartOfTurn`` drives barge-in and
    ``EndOfTurn`` commits the user turn immediately.

    Args:
        model: ``nova-3`` (default), another ``/v1/listen`` model, ``flux-general-en`` or
            ``flux-general-multi``.
        api_key: Deepgram API key (default: ``DEEPGRAM_API_KEY``).
        language: Nova ``language`` (``en``, ``en-US``, ``multi``...); for
            ``flux-general-multi`` a ``language_hint``.
        sample_rate: rate of the linear16 audio sent to Deepgram (input is resampled).
        base_url: API origin (``https://api.deepgram.com``; EU or self-hosted servers).
        interim_results, smart_format, punctuate, endpointing_ms, utterance_end_ms,
            vad_events: Nova options (``endpointing_ms=False`` disables endpointing,
            ``None`` keeps Deepgram's default of 10 ms).
        keyterms: keyterm prompting (Nova-3 and Flux).
        numerals, profanity_filter: text formatting options.
        eot_threshold, eager_eot_threshold, eot_timeout_ms: Flux end-of-turn tuning
            (0.5-1.0, 0.3-0.9, 500-60000 ms). ``eager_eot_threshold`` enables
            ``EagerEndOfTurn`` / ``TurnResumed``.
        chunk_ms: audio chunk size sent to Deepgram (default 80 ms for Flux, which
            Deepgram strongly recommends, and 50 ms for Nova).
        keepalive_interval: Nova only: send ``KeepAlive`` after this many seconds without
            audio (Deepgram closes idle streams after 10 s); ``None`` disables it.
        connect_timeout: WebSocket handshake timeout in seconds.
        mip_opt_out, tags: Deepgram model-improvement opt-out and request tags.
        extra_params: additional query parameters, passed through verbatim.
    """

    provider = PROVIDER

    def __init__(
        self,
        *,
        model: str = "nova-3",
        api_key: str | None = None,
        language: str | None = None,
        sample_rate: int = 16_000,
        base_url: str = DEFAULT_BASE_URL,
        interim_results: bool = True,
        smart_format: bool = True,
        punctuate: bool | None = None,
        endpointing_ms: int | Literal[False] | None = 300,
        utterance_end_ms: int | None = 1000,
        vad_events: bool = True,
        keyterms: str | Sequence[str] = (),
        numerals: bool | None = None,
        profanity_filter: bool | None = None,
        eot_threshold: float | None = None,
        eager_eot_threshold: float | None = None,
        eot_timeout_ms: int | None = None,
        chunk_ms: int | None = None,
        keepalive_interval: float | None = 5.0,
        connect_timeout: float = 10.0,
        mip_opt_out: bool | None = None,
        tags: str | Sequence[str] = (),
        extra_params: Mapping[str, Any] | None = None,
    ) -> None:
        flux = model.startswith("flux")
        if flux:
            _check_range("eot_threshold", eot_threshold, 0.5, 1.0)
            _check_range("eager_eot_threshold", eager_eot_threshold, 0.3, 0.9)
            _check_range("eot_timeout_ms", eot_timeout_ms, 500, 60_000)
            if (
                eot_threshold is not None
                and eager_eot_threshold is not None
                and eager_eot_threshold > eot_threshold
            ):
                raise ConfigurationError("eager_eot_threshold must not exceed eot_threshold")
            if sample_rate not in _FLUX_SAMPLE_RATES:
                raise ConfigurationError(
                    f"Flux supports sample rates {_FLUX_SAMPLE_RATES}, got {sample_rate}"
                )
        if chunk_ms is not None and chunk_ms <= 0:
            raise ConfigurationError("chunk_ms must be > 0")
        super().__init__(
            model=model,
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=True if flux else interim_results,
                word_timestamps=True,
                end_of_turn=flux,
                language_detection=model == "flux-general-multi" if flux else language == "multi",
                reconnect=True,  # each stream opens its own WebSocket
            ),
            sample_rate=sample_rate,
            language=language,
        )
        self._api_key = _resolve_api_key(api_key)
        self.is_flux = flux
        self.base_url = base_url
        self.interim_results = interim_results
        self.smart_format = smart_format
        self.punctuate = punctuate
        self.endpointing_ms = endpointing_ms
        self.utterance_end_ms = utterance_end_ms
        self.vad_events = vad_events
        self.keyterms = _as_tuple(keyterms)
        self.numerals = numerals
        self.profanity_filter = profanity_filter
        self.eot_threshold = eot_threshold
        self.eager_eot_threshold = eager_eot_threshold
        self.eot_timeout_ms = eot_timeout_ms
        self.chunk_ms = chunk_ms or (80 if flux else 50)
        self.keepalive_interval = keepalive_interval
        self.connect_timeout = connect_timeout
        self.mip_opt_out = mip_opt_out
        self.tags = _as_tuple(tags)
        self.extra_params = dict(extra_params or {})

    def url(self, language: str | None = None) -> str:
        """The streaming WebSocket URL (query string included, no credentials)."""
        params: dict[str, Any] = {
            "model": self.model,
            "encoding": "linear16",
            "sample_rate": self.sample_rate,
        }
        if self.is_flux:
            path = "/v2/listen"
            params |= {
                "eot_threshold": self.eot_threshold,
                "eager_eot_threshold": self.eager_eot_threshold,
                "eot_timeout_ms": self.eot_timeout_ms,
                # language hints only exist for the multilingual model
                "language_hint": language if self.model != "flux-general-en" else None,
            }
        else:
            path = "/v1/listen"
            endpointing: Any = "false" if self.endpointing_ms is False else self.endpointing_ms
            params |= {
                "channels": 1,
                "language": language,
                "interim_results": self.interim_results,
                "smart_format": self.smart_format,
                "punctuate": self.punctuate,
                "endpointing": endpointing,
                # UtteranceEnd is computed from interim results, which it therefore requires
                "utterance_end_ms": self.utterance_end_ms if self.interim_results else None,
                "vad_events": self.vad_events,
            }
        params |= {
            "keyterm": list(self.keyterms),
            "numerals": self.numerals,
            "profanity_filter": self.profanity_filter,
            "mip_opt_out": self.mip_opt_out,
            "tag": list(self.tags),
        }
        params |= self.extra_params
        base = _with_scheme(self.base_url, websocket=True)
        return f"{base}{path}?{urlencode(_query_items(params))}"

    def _create_stream(self, *, language: str | None) -> STTStream:
        if self.is_flux:
            return _FluxStream(self, language=language)
        return _NovaStream(self, language=language)


def _check_range(name: str, value: float | None, low: float, high: float) -> None:
    if value is not None and not low <= value <= high:
        raise ConfigurationError(f"{name} must be within [{low}, {high}], got {value}")


class _DeepgramStream(STTStream):
    """WebSocket plumbing shared by Nova and Flux: connect, chunk and send audio, receive."""

    def __init__(self, stt: DeepgramSTT, *, language: str | None) -> None:
        self._dg = stt
        self._chunker = FrameChunker(stt.sample_rate, frame_duration=stt.chunk_ms / 1000.0)
        self._closing = False
        self._segment_id = new_id("seg_")
        self.request_id: str | None = None
        super().__init__(stt, language=language)

    # -------------------------------------------------------------- subclass hooks
    def _keepalive_interval(self) -> float | None:
        return None

    async def _send_flush(self, ws: ClientConnection) -> None:
        raise NotImplementedError

    def _on_message(self, message: dict[str, Any]) -> None:
        raise NotImplementedError

    def _on_closed(self) -> None:
        """The server closed the stream after ``CloseStream``."""

    # ------------------------------------------------------------------- plumbing
    async def _run(self) -> None:
        stt = self._dg
        ws = await _open_websocket(
            stt.url(self._language), stt._api_key, timeout=stt.connect_timeout, what="STT"
        )
        if ws.response is not None:
            self.request_id = ws.response.headers.get("dg-request-id")
        receiver = asyncio.create_task(self._recv_loop(ws), name="deepgram-stt-recv")
        sender = asyncio.create_task(self._send_loop(ws), name="deepgram-stt-send")
        try:
            await asyncio.wait({receiver, sender}, return_when=asyncio.FIRST_EXCEPTION)
            error = _task_error([receiver, sender], "STT")
            if error is not None:
                raise error
        finally:
            await cancel_and_wait(sender, receiver)
            await close_ws(ws)

    async def _send_loop(self, ws: ClientConnection) -> None:
        interval = self._keepalive_interval()
        while True:
            try:
                if interval:
                    item = await asyncio.wait_for(self._input.recv(), interval)
                else:
                    item = await self._input.recv()
            except TimeoutError:
                await ws.send(_MSG_KEEPALIVE)  # no audio for a while: keep the stream open
                continue
            except ChanClosed:
                break
            if self.is_flush(item):
                for chunk in self._chunker.flush():  # the partial chunk, so the flush covers it
                    await ws.send(chunk.data)
                await self._send_flush(ws)
            else:
                assert isinstance(item, AudioFrame)
                for chunk in self._chunker.push(item):
                    await ws.send(chunk.data)
        for chunk in self._chunker.flush():
            await ws.send(chunk.data)
        self._closing = True
        await ws.send(_MSG_CLOSE_STREAM)

    async def _recv_loop(self, ws: ClientConnection) -> None:
        try:
            async for message in ws:
                if isinstance(message, str):
                    self._on_message(_parse(message))
        except ConnectionClosedError as exc:
            code = exc.rcvd.code if exc.rcvd is not None else None
            if not (self._closing and (code is None or code in _NORMAL_CLOSE_CODES)):
                raise
        if not self._closing:
            raise ProviderConnectionError(
                "Deepgram closed the STT connection unexpectedly", provider=PROVIDER
            )
        self._on_closed()

    # -------------------------------------------------------------------- events
    def _event(self, kind: STTEventType, transcript: Transcript | None = None) -> None:
        self._emit(STTEvent(kind, transcript, self._segment_id))

    def _empty_final(self) -> None:
        self._event(STTEventType.FINAL_TRANSCRIPT, Transcript("", self._language))


class _NovaStream(_DeepgramStream):
    """``/v1/listen``: Nova-3 (and older Nova/Enhanced/Base models)."""

    def __init__(self, stt: DeepgramSTT, *, language: str | None) -> None:
        self._in_speech = False
        self._finalized = False  # Deepgram ended the utterance itself, no speech since
        super().__init__(stt, language=language)

    def _keepalive_interval(self) -> float | None:
        return self._dg.keepalive_interval or None

    async def _send_flush(self, ws: ClientConnection) -> None:
        await ws.send(_MSG_FINALIZE)
        if self._dg.vad_events and self._finalized:
            # Deepgram already ended the utterance (speech_final / UtteranceEnd) and nothing
            # was heard since, so everything is final; Deepgram "may not" answer a Finalize
            # with nothing buffered: acknowledge the flush now so waiters do not stall.
            self._empty_final()

    def _on_message(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "Results":
            self._on_results(message)
        elif kind == "SpeechStarted":
            self._speech_started()
        elif kind == "UtteranceEnd":
            self._speech_ended(_float(message.get("last_word_end")))
        elif kind == "Metadata":
            self.request_id = self.request_id or message.get("request_id")
        elif kind == "Error" or "err_code" in message:
            raise ProviderError(f"Deepgram STT error: {_detail(message)}", provider=PROVIDER)
        else:
            logger.debug("Deepgram STT: ignoring %s message", kind)

    def _on_results(self, message: dict[str, Any]) -> None:
        alternatives = (message.get("channel") or {}).get("alternatives") or [{}]
        alt = alternatives[0] if isinstance(alternatives[0], dict) else {}
        text = str(alt.get("transcript") or "").strip()
        start, duration = _float(message.get("start")), _float(message.get("duration"))
        words = _words(alt.get("words"))
        languages = alt.get("languages") or []
        transcript = Transcript(
            text=text,
            language=str(languages[0]) if languages else self._language,
            confidence=_float(alt.get("confidence")),
            start_time=start,
            end_time=start + duration if start is not None and duration is not None else None,
            words=words,
        )
        if text:
            self._speech_started()
        if message.get("is_final"):
            # an empty final only matters as the answer to our Finalize (flush)
            if text or message.get("from_finalize"):
                self._event(STTEventType.FINAL_TRANSCRIPT, transcript)
            if message.get("speech_final"):
                self._speech_ended(words[-1].end if words else transcript.end_time)
        elif text:
            self._event(STTEventType.INTERIM_TRANSCRIPT, transcript)

    def _speech_started(self) -> None:
        if not self._in_speech:
            self._in_speech, self._finalized = True, False
            self._segment_id = new_id("seg_")
            self._event(STTEventType.START_OF_SPEECH)

    def _speech_ended(self, end_time: float | None) -> None:
        if self._in_speech:
            self._in_speech, self._finalized = False, True
            self._event(
                STTEventType.END_OF_SPEECH, Transcript("", self._language, end_time=end_time)
            )


class _FluxStream(_DeepgramStream):
    """``/v2/listen``: Flux conversational STT with model-integrated end of turn."""

    def __init__(self, stt: DeepgramSTT, *, language: str | None) -> None:
        self._turn_active = False
        self._last_text = ""
        self._last_transcript: Transcript | None = None
        super().__init__(stt, language=language)

    async def _send_flush(self, ws: ClientConnection) -> None:
        # Ends the active turn now; Flux answers with EndOfTurn(trigger="manual") or, when
        # no turn is active, with a FORCE_END_TURN_NO_ACTIVE_TURN warning.
        await ws.send(_MSG_FORCE_END_TURN)

    def _on_message(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "TurnInfo":
            self._on_turn_info(message)
        elif kind == "Connected":
            self.request_id = message.get("request_id") or self.request_id
        elif kind == "Warning":
            code = str(message.get("code") or "")
            if code == "FORCE_END_TURN_NO_ACTIVE_TURN":
                self._empty_final()  # flush outside a turn: nothing to finalize
            else:
                logger.warning("Deepgram Flux warning %s: %s", code, message.get("description", ""))
        elif kind == "Error":
            raise _flux_error(message)
        elif kind == "ConfigureFailure":
            logger.warning("Deepgram Flux rejected a Configure message: %s", message)
        else:
            logger.debug("Deepgram Flux: ignoring %s message", kind)

    def _transcript(self, message: dict[str, Any], text: str) -> Transcript:
        words = _words(message.get("words"))
        confidences = [w.confidence for w in words or () if w.confidence is not None]
        languages = message.get("languages") or []
        return Transcript(
            text=text,
            language=str(languages[0]) if languages else self._language,
            confidence=sum(confidences) / len(confidences) if confidences else None,
            start_time=words[0].start if words else _float(message.get("audio_window_start")),
            end_time=words[-1].end if words else _float(message.get("audio_window_end")),
            words=words,
        )

    def _on_turn_info(self, message: dict[str, Any]) -> None:
        event = message.get("event")
        text = str(message.get("transcript") or "").strip()
        transcript = self._transcript(message, text)
        if event == "EndOfTurn":
            self._end_turn(transcript, str(message.get("trigger") or "model"))
            return
        if event not in ("StartOfTurn", "Update", "EagerEndOfTurn", "TurnResumed"):
            logger.debug("Deepgram Flux: ignoring TurnInfo event %s", event)
            return
        if not text:  # Updates before StartOfTurn carry no words
            return
        self._begin_turn()
        if event == "TurnResumed":
            self._event(STTEventType.TURN_RESUMED, transcript)
        if text != self._last_text:
            self._last_text, self._last_transcript = text, transcript
            self._event(STTEventType.INTERIM_TRANSCRIPT, transcript)
        if event == "EagerEndOfTurn":
            self._event(STTEventType.EAGER_END_OF_TURN, transcript)

    def _begin_turn(self) -> None:
        if not self._turn_active:
            self._turn_active = True
            self._last_text, self._last_transcript = "", None
            self._segment_id = new_id("seg_")
            self._event(STTEventType.START_OF_SPEECH)

    def _end_turn(self, transcript: Transcript, trigger: str) -> None:
        if transcript.text:
            self._begin_turn()  # a turn with words always has a start
        self._event(STTEventType.FINAL_TRANSCRIPT, transcript)
        if self._turn_active:
            self._event(STTEventType.END_OF_SPEECH, transcript)
        if trigger != "manual":  # "manual" = ended by our own flush (ForceEndTurn)
            self._event(STTEventType.END_OF_TURN, transcript)
        self._turn_active = False
        self._last_text, self._last_transcript = "", None
        self._report_usage()

    def _report_usage(self) -> None:
        # The base class reports STT usage when a flush is answered; turns that Flux ends on
        # its own are never flushed, so report their audio here (no flush latency to add).
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

    def _on_closed(self) -> None:
        # CloseStream never finalizes the active turn: the last Update is its transcript.
        if self._turn_active and self._last_transcript is not None:
            self._event(STTEventType.FINAL_TRANSCRIPT, self._last_transcript)
            self._event(STTEventType.END_OF_SPEECH, self._last_transcript)
            self._turn_active = False


def _flux_error(message: dict[str, Any]) -> ProviderError:
    code = str(message.get("code") or "UNKNOWN")
    text = f"Deepgram Flux error {code}: {message.get('description') or ''}".rstrip(": ")
    upper = code.upper()
    if "AUTH" in upper or "PERMISSION" in upper:
        return AuthenticationError(text, provider=PROVIDER)
    if "RATE" in upper or "TOO_MANY" in upper or "CONCURRENCY" in upper:
        return RateLimitError(text, provider=PROVIDER)
    retryable = "INTERNAL" in upper or "TIMEOUT" in upper or "UNAVAILABLE" in upper
    return ProviderError(text, provider=PROVIDER, retryable=retryable)


# -------------------------------------------------------------------------------- TTS
@dataclass(eq=False)
class _SpeakConnection:
    """A pooled ``/v1/speak`` WebSocket (one per conversation, as Deepgram recommends)."""

    ws: ClientConnection
    url: str
    created: float = field(default_factory=now)
    flushes: deque[float] = field(default_factory=deque)
    expiry: asyncio.Task[None] | None = None

    def recent_flushes(self) -> int:
        cutoff = now() - _FLUSH_WINDOW
        while self.flushes and self.flushes[0] < cutoff:
            self.flushes.popleft()
        return len(self.flushes)

    def reusable(self) -> bool:
        return (
            self.ws.state is State.OPEN
            and now() - self.created < _MAX_CONNECTION_AGE
            and self.recent_flushes() < _FLUSH_LIMIT - 2
        )

    async def aclose(self) -> None:
        await close_ws(self.ws)


@register_provider(
    "tts",
    "deepgram",
    description="Deepgram Aura-2 TTS (WebSocket text streaming + REST)",
    default_model="aura-2-thalia-en",
    models=(
        "aura-2-thalia-en",
        "aura-2-andromeda-en",
        "aura-2-helena-en",
        "aura-2-apollo-en",
        "aura-2-arcas-en",
        "aura-2-aries-en",
    ),
    env=(API_KEY_ENV,),
    requires=("websockets", "httpx"),
    local=False,
)
class DeepgramTTS(TTS):
    """Deepgram Aura-2 text-to-speech (linear16 PCM, 24 kHz by default).

    * :meth:`stream` (``capabilities.streaming=True``) uses the ``/v1/speak`` WebSocket:
      every pushed text delta is sent as ``Speak``; :meth:`SynthesizeStream.flush` sends
      ``Flush`` and the segment ends when Deepgram answers ``Flushed``. Connections are
      kept open between streams (Deepgram asks for one WebSocket per conversation); a
      stream closed mid-synthesis (barge-in) sends ``Clear`` and the connection is reused
      once Deepgram confirms with ``Cleared``.
    * :meth:`synthesize` uses the REST endpoint (``container=none``), streaming the body.

    Args:
        model: Aura voice model, ``aura-2-<voice>-<language>`` (default
            ``aura-2-thalia-en``).
        voice: optional voice override: a full model id (``aura-2-apollo-en``) or a voice
            name (``apollo``) combined with the model's family and language.
        sample_rate: 8000, 16000, 24000 (default), 32000 or 48000 Hz.
        speed: speaking-rate multiplier (0.7-1.5; not supported by every language).
        streaming: ``False`` synthesizes sentence by sentence over REST instead (exact
            text/audio alignment for truncation, at the cost of one request per sentence).
        base_url: API origin (``https://api.deepgram.com``).
        http_client: an ``httpx.AsyncClient`` for REST requests (created if omitted).
        timeout: HTTP timeout of REST requests, in seconds. (``request_timeout`` is a
            deprecated alias.)
        idle_timeout: close pooled WebSocket connections unused for this many seconds.
        clear_timeout: max wait for ``Cleared`` before a connection is dropped instead.
        mip_opt_out, tags, extra_params: passed through as query parameters.
        normalize: spoken-form text normalization (see :class:`~voice_agent_next.tts.TTS`),
            off by default: the service normalizes text itself.
    """

    provider = PROVIDER

    @property
    def request_timeout(self) -> float:
        """Deprecated alias of :attr:`timeout`."""
        return self.timeout

    def __init__(
        self,
        *,
        model: str = "aura-2-thalia-en",
        api_key: str | None = None,
        voice: str | None = None,
        sample_rate: int = 24_000,
        speed: float | None = None,
        streaming: bool = True,
        base_url: str = DEFAULT_BASE_URL,
        http_client: httpx.AsyncClient | None = None,
        connect_timeout: float = 10.0,
        timeout: float = 30.0,
        idle_timeout: float = 60.0,
        clear_timeout: float = 2.0,
        mip_opt_out: bool | None = None,
        tags: str | Sequence[str] = (),
        extra_params: Mapping[str, Any] | None = None,
        clean_text: bool = True,
        normalize: NormalizeOption = None,
        request_timeout: float | None = None,
    ) -> None:
        if request_timeout is not None:
            timeout = deprecated("DeepgramTTS", "timeout", "request_timeout", request_timeout)
        if sample_rate not in _AURA_SAMPLE_RATES:
            raise ConfigurationError(
                f"Aura supports sample rates {_AURA_SAMPLE_RATES} for linear16, got {sample_rate}"
            )
        _check_range("speed", speed, 0.7, 1.5)
        super().__init__(
            model=model,
            sample_rate=sample_rate,
            capabilities=TTSCapabilities(streaming=streaming),
            voice=voice,
            clean_text=clean_text,
            normalize=normalize,
        )
        self._api_key = _resolve_api_key(api_key)
        self.speed = speed
        self.base_url = base_url
        self.connect_timeout = connect_timeout
        self.timeout = timeout
        self.idle_timeout = idle_timeout
        self.clear_timeout = clear_timeout
        self.mip_opt_out = mip_opt_out
        self.tags = _as_tuple(tags)
        self.extra_params = dict(extra_params or {})
        self._http = http_client
        self._owns_http = http_client is None
        self._pool: dict[str, list[_SpeakConnection]] = {}
        self._tasks = BackgroundTasks("deepgram-tts")  # expiry timers, Clear handshakes
        self._closers = BackgroundTasks("deepgram-tts-close")  # awaited, never cancelled

    # ---------------------------------------------------------------- requests
    def model_for(self, voice: str | None) -> str:
        """The Aura model to use for ``voice`` (``None`` = this TTS's model)."""
        if not voice or voice == self.model:
            return self.model
        if voice.startswith("aura"):
            return voice
        match = _AURA_MODEL.fullmatch(self.model)
        if match is not None:
            return f"{match['family']}-{voice}-{match['lang']}"
        return f"aura-2-{voice}-en"

    def _params(self, model: str) -> dict[str, Any]:
        return {
            "model": model,
            "encoding": "linear16",
            "sample_rate": self.sample_rate,
            "speed": self.speed,
            "mip_opt_out": self.mip_opt_out,
            "tag": list(self.tags),
            **self.extra_params,
        }

    def url(self, model: str | None = None) -> str:
        """The ``/v1/speak`` WebSocket URL for ``model`` (no credentials)."""
        query = urlencode(_query_items(self._params(model or self.model)))
        return f"{_with_scheme(self.base_url, websocket=True)}/v1/speak?{query}"

    def _synthesize(self, text: str, *, voice: str | None) -> ChunkedStream:
        return _AuraChunkedStream(self, text, voice=voice)

    def _create_stream(self, *, voice: str | None) -> SynthesizeStream:
        return _AuraStream(self, voice=voice)

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=self.connect_timeout)
            )
        return self._http

    async def _synthesize_rest(self, text: str, model: str, push: Callable[[bytes], None]) -> None:
        url = f"{_with_scheme(self.base_url, websocket=False)}/v1/speak"
        params = tuple(_query_items({**self._params(model), "container": "none"}))
        headers = {"Authorization": f"Token {self._api_key}"}
        try:
            async with self._client().stream(
                "POST", url, params=params, json={"text": text}, headers=headers
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise _http_error(response.status_code, _error_detail(body, response.headers))
                async for chunk in response.aiter_bytes():
                    if chunk:
                        push(chunk)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError("Deepgram TTS request timed out", provider=PROVIDER) from exc
        except httpx.TransportError as exc:
            raise ProviderConnectionError(
                f"Deepgram TTS request failed: {exc}", provider=PROVIDER
            ) from exc

    # ------------------------------------------------------------ connection pool
    async def _acquire(self, url: str) -> _SpeakConnection:
        idle = self._pool.get(url)
        while idle:
            conn = idle.pop()
            if conn.expiry is not None:
                conn.expiry.cancel()
                conn.expiry = None
            if conn.reusable():
                return conn
            self._close_later(conn)
        ws = await _open_websocket(url, self._api_key, timeout=self.connect_timeout, what="TTS")
        return _SpeakConnection(ws, url)

    def _put(self, conn: _SpeakConnection) -> None:
        idle = self._pool.setdefault(conn.url, [])
        if idle or not conn.reusable():  # keep one idle connection per voice
            self._close_later(conn)
            return
        idle.append(conn)
        conn.expiry = self._tasks.spawn(self._expire(conn), name="deepgram-tts-expire")

    def _close_later(self, conn: _SpeakConnection) -> None:
        self._closers.spawn(conn.aclose(), name="deepgram-tts-close")

    async def _expire(self, conn: _SpeakConnection) -> None:
        await asyncio.sleep(self.idle_timeout)
        idle = self._pool.get(conn.url, [])
        if conn in idle:
            idle.remove(conn)
            conn.expiry = None
            await conn.aclose()

    def _release(self, conn: _SpeakConnection, *, broken: bool, dirty: bool) -> None:
        if broken:
            self._close_later(conn)
        elif dirty:
            self._tasks.spawn(self._clear_and_put(conn), name="deepgram-tts-clear")
        else:
            self._put(conn)

    async def _clear_and_put(self, conn: _SpeakConnection) -> None:
        """Drop the audio of an interrupted stream so the connection can be reused."""
        try:
            await conn.ws.send(_MSG_CLEAR)
            async with asyncio.timeout(self.clear_timeout):
                while True:
                    message = await conn.ws.recv()
                    if isinstance(message, str) and _parse(message).get("type") == "Cleared":
                        break
        except asyncio.CancelledError:
            await conn.aclose()
            raise
        except Exception:
            logger.debug("Deepgram TTS: Clear failed, dropping the connection", exc_info=True)
            await conn.aclose()
            return
        self._put(conn)

    async def warmup(self) -> None:
        """Open the streaming WebSocket ahead of the first response."""
        if not self.capabilities.streaming:
            return
        url = self.url(self.model_for(self.voice))
        if not self._pool.get(url):
            self._put(await self._acquire(url))

    async def aclose(self) -> None:
        await self._tasks.cancel_all()
        conns = [conn for idle in self._pool.values() for conn in idle]
        self._pool.clear()
        await asyncio.gather(*(conn.aclose() for conn in conns))
        await self._closers.wait_all()
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None


class _AuraChunkedStream(ChunkedStream):
    async def _run(self) -> None:
        tts: DeepgramTTS = self._tts  # type: ignore[assignment]
        if self.text.strip():  # Deepgram rejects empty input (400)
            await tts._synthesize_rest(self.text, tts.model_for(self.voice), self._push_audio)


class _AuraStream(SynthesizeStream):
    """Native text streaming over a pooled ``/v1/speak`` WebSocket."""

    def __init__(self, tts: DeepgramTTS, *, voice: str | None) -> None:
        self._dg = tts
        self._url = tts.url(tts.model_for(voice))
        self._pending: deque[str | None] = deque()  # flushed segments awaiting "Flushed"
        self._buf: list[str] = []  # text of the segment being pushed (not flushed yet)
        self._unacked = False  # sent text whose audio Deepgram has not finished
        self._input_done = False
        self._segment_audio = False  # audio already emitted for the current segment
        self._finished = asyncio.Event()
        super().__init__(tts, voice=voice)

    async def _run(self) -> None:
        conn = await self._dg._acquire(self._url)
        receiver = asyncio.create_task(self._recv_loop(conn), name="deepgram-tts-recv")
        sender = asyncio.create_task(self._send_loop(conn), name="deepgram-tts-send")
        finished = asyncio.create_task(self._finished.wait(), name="deepgram-tts-finished")
        broken = True
        try:
            waiting: set[asyncio.Task[Any]] = {receiver, sender, finished}
            while not finished.done():
                done, waiting = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                error = _task_error([receiver, sender], "TTS")
                if error is not None:
                    raise error
                if receiver in done and not finished.done():
                    raise ProviderConnectionError(
                        "Deepgram closed the TTS connection unexpectedly", provider=PROVIDER
                    )
            broken = False
        except asyncio.CancelledError:
            broken = False  # interrupted, not failed: the connection is cleared and reused
            raise
        finally:
            await cancel_and_wait(receiver, sender, finished)
            self._dg._release(conn, broken=broken, dirty=self._unacked)

    async def _send_loop(self, conn: _SpeakConnection) -> None:
        async for item in self._input:
            if self.is_flush(item):
                text = "".join(self._buf)
                self._buf.clear()
                if text.strip():
                    self._pending.append(text)
                    conn.flushes.append(now())
                    await conn.ws.send(_MSG_FLUSH)
                else:
                    self._pending.append(None)
                    self._end_empty_segments()
            else:
                assert isinstance(item, str)
                self._buf.append(item)
                if item.strip():
                    self._unacked = True
                    await conn.ws.send(json.dumps({"type": "Speak", "text": item}))
        self._input_done = True
        self._check_finished()

    async def _recv_loop(self, conn: _SpeakConnection) -> None:
        async for message in conn.ws:
            if isinstance(message, bytes):
                if message:
                    self._on_audio(message)
                continue
            data = _parse(message)
            kind = data.get("type")
            if kind == "Flushed":
                self._on_flushed()
            elif kind == "Warning":
                self._on_warning(data)
            elif kind == "Error" or "err_code" in data:
                raise ProviderError(f"Deepgram TTS error: {_detail(data)}", provider=PROVIDER)
            else:  # Metadata, a late Cleared...
                logger.debug("Deepgram TTS: %s", data)

    def _on_audio(self, data: bytes) -> None:
        if not self._segment_audio:
            self._segment_audio = True
            if self._segment_text is None:
                head = self._pending[0] if self._pending else "".join(self._buf)
                self._segment_text = (head or "").strip() or None
        self._push_audio(data)

    def _on_flushed(self) -> None:
        if not self._pending:
            logger.debug("Deepgram TTS: ignoring an unexpected Flushed message")
            return
        text = self._pending.popleft()
        if not self._segment_audio and self._segment_text is None and text:
            self._segment_text = text.strip()
        self._end_segment()
        self._segment_audio = False
        self._end_empty_segments()
        if not self._pending and not "".join(self._buf).strip():
            self._unacked = False
        self._check_finished()

    def _on_warning(self, data: dict[str, Any]) -> None:
        code = str(data.get("code") or data.get("warn_code") or "")
        description = str(data.get("description") or data.get("warn_msg") or "")
        if "flush" in f"{code} {description}".lower() and self._pending:
            # the Flush was dropped (20 per minute): its "Flushed" will never come
            raise RateLimitError(
                f"Deepgram TTS flush limit reached: {description or code}", provider=PROVIDER
            )
        logger.warning("Deepgram TTS warning %s: %s", code, description)

    def _end_empty_segments(self) -> None:
        while self._pending and self._pending[0] is None:
            self._pending.popleft()
            self._end_segment()

    def _check_finished(self) -> None:
        if self._input_done and not self._pending:
            self._finished.set()
