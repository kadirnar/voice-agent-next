"""ElevenLabs providers: Flash v2.5 / Multilingual v2 / v3 streaming TTS and Scribe v2 STT.

* ``tts="elevenlabs/eleven_flash_v2_5"`` — :class:`ElevenLabsTTS`. Native text streaming
  over one persistent **multi-context** WebSocket
  (``/v1/text-to-speech/{voice_id}/multi-stream-input``): one context per segment, a
  ``flush`` at every sentence end, ``close_context`` at the end of a segment and on
  barge-in. Character alignment (``sync_alignment``) becomes word timings on
  :attr:`~voice_agent_next.tts.SynthesizedAudio.words`, so the cascade truncates barge-ins
  word-exactly. Eleven v3 models (``eleven_v3``, ``eleven_v3_conversational``) are not
  served by that endpoint; they use the Text to Dialogue multi-context WebSocket
  (``/v1/text-to-dialogue/multi-stream-input``), which has the same context model with a
  different message framing. :meth:`~voice_agent_next.tts.TTS.synthesize` streams raw PCM
  over HTTP (``POST /v1/text-to-speech/{voice_id}/stream``, or
  ``/v1/text-to-dialogue/stream`` for v3).
* ``stt="elevenlabs/scribe_v2_realtime"`` — :class:`ElevenLabsSTT`. Scribe v2 Realtime
  over ``/v1/speech-to-text/realtime``: base64 ``input_audio_chunk`` messages,
  ``partial_transcript`` -> interim, ``committed_transcript`` -> final. By default the
  cascade owns endpointing: :meth:`~voice_agent_next.stt.STTStream.flush` sends a manual
  ``commit``; ``commit_strategy="vad"`` lets Scribe's server VAD commit instead.
  :meth:`~voice_agent_next.stt.STT.transcribe` uses the batch ``POST /v1/speech-to-text``
  API (Scribe v2).

Only core dependencies are used (``websockets`` and ``httpx``, imported lazily). The API
key comes from ``api_key=`` or ``ELEVEN_API_KEY`` / ``ELEVENLABS_API_KEY`` and is sent in
the ``xi-api-key`` header, never in URLs. ``region=`` / ``base_url=`` select the data
residency or US-only servers. See ``docs/providers/elevenlabs.md`` for the protocol notes
and the documentation pages (checked 2026-09-24) this module follows.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import math
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

from ..audio.frame import AudioFrame
from ..audio.wav import wav_bytes
from ..errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from ..metrics import STTMetrics
from ..registry import register_provider
from ..stt import STT, STTCapabilities, STTEvent, STTEventType, STTStream, Transcript, WordTiming
from ..tts import TTS, ChunkedStream, SynthesizedAudio, SynthesizeStream, TTSCapabilities
from ..utils.aio import Chan, ChanClosed, cancel_and_wait
from ..utils.clock import now
from ..utils.ids import new_id
from ..utils.log import logger

if TYPE_CHECKING:
    import httpx
    from websockets.asyncio.client import ClientConnection
    from websockets.exceptions import ConnectionClosed

__all__ = [
    "API_KEY_ENV",
    "DEFAULT_BASE_URL",
    "DEFAULT_VOICE",
    "MAX_CONTEXTS",
    "REGIONS",
    "STT_SAMPLE_RATES",
    "TTS_SAMPLE_RATES",
    "ElevenLabsSTT",
    "ElevenLabsTTS",
]

API_KEY_ENV = ("ELEVEN_API_KEY", "ELEVENLABS_API_KEY")
"""Environment variables holding the API key, in lookup order."""
DEFAULT_BASE_URL = "https://api.elevenlabs.io"
REGIONS: dict[str, str] = {
    "us": "https://api.us.elevenlabs.io",
    "eu": "https://api.eu.residency.elevenlabs.io",
    "in": "https://api.in.residency.elevenlabs.io",
    "sg": "https://api.sg.residency.elevenlabs.io",
}
"""``region=`` shortcuts: US-only servers and the data-residency environments."""
DEFAULT_VOICE = "JBFqnCBsd6RMkjVDRZzb"
""""George", the voice used throughout the ElevenLabs API reference."""
TTS_SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)
"""Raw PCM output rates (``output_format=pcm_<rate>``; 44.1 kHz needs a Pro plan)."""
STT_SAMPLE_RATES = (8000, 16000, 22050, 24000, 44100, 48000)
"""PCM input rates accepted by Scribe v2 Realtime (``audio_format=pcm_<rate>``)."""
MAX_CONTEXTS = 5
"""Concurrent contexts ElevenLabs allows on one multi-context WebSocket."""

_PROVIDER = "elevenlabs"
_MAX_MESSAGE_BYTES = 16 * 2**20
_LANGUAGE_MODELS = ("eleven_flash_v2_5", "eleven_turbo_v2_5")
"""Non-v3 models that accept ``language_code`` (Multilingual v2 rejects it)."""
_ABANDON_GRACE = 5.0
"""Seconds a cancelled context keeps its slot while its ``isFinal`` is awaited."""
_SENTENCE_END = re.compile(r"(?:[.!?…]+[\"'”’)\]]*|[。！？｡][」』”’）]*)$")
_CLAUSE_END = re.compile(r"[,;:—–]$")
_TIME_BASE_DECIDE_AFTER = 0.5
"""Context audio (s) after which an alignment's time base is decided for good."""


# ----------------------------------------------------------------------------- helpers
def _resolve_api_key(api_key: str | None) -> str:
    for candidate in (api_key, *(os.environ.get(name) for name in API_KEY_ENV)):
        if candidate and candidate.strip():
            return candidate.strip()
    raise ConfigurationError(
        "ElevenLabs needs an API key: pass api_key=... or set ELEVEN_API_KEY "
        "(or ELEVENLABS_API_KEY)"
    )


def _resolve_base_url(base_url: str | None, region: str | None) -> str:
    if region is not None:
        if base_url is not None:
            raise ConfigurationError("pass either base_url=... or region=..., not both")
        try:
            return REGIONS[region.strip().lower()]
        except KeyError:
            raise ConfigurationError(
                f"unknown ElevenLabs region {region!r}; expected one of {sorted(REGIONS)}"
            ) from None
    return (base_url or DEFAULT_BASE_URL).strip().rstrip("/")


def _ws_base(base_url: str) -> str:
    parts = urlsplit(base_url)
    scheme = {"https": "wss", "http": "ws"}.get(parts.scheme, parts.scheme)
    return urlunsplit((scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def _language_code(language: str | None) -> str | None:
    """``"en-US"`` -> ``"en"``: ElevenLabs takes bare ISO 639 codes."""
    if not language or not language.strip():
        return None
    return re.split(r"[-_]", language.strip(), maxsplit=1)[0].lower() or None


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _is_dialogue_model(model: str) -> bool:
    return model.startswith("eleven_v3")


def _check_range(name: str, value: float | None, low: float, high: float) -> None:
    if value is not None and not low <= value <= high:
        raise ConfigurationError(f"{name} must be within [{low}, {high}], got {value}")


def _locators(
    dictionaries: Sequence[tuple[str, str] | Mapping[str, str]] | None,
) -> list[dict[str, str]]:
    """Pronunciation dictionary locators from ``(id, version_id)`` pairs or mappings."""
    out: list[dict[str, str]] = []
    for item in dictionaries or ():
        if isinstance(item, Mapping):
            locator = {str(k): str(v) for k, v in item.items() if v is not None}
        else:
            dict_id, version = item
            locator = {"pronunciation_dictionary_id": str(dict_id), "version_id": str(version)}
        if "pronunciation_dictionary_id" not in locator:
            raise ConfigurationError(f"pronunciation dictionary locator without an id: {item!r}")
        out.append(locator)
    return out


# ------------------------------------------------------------------------------ errors
_AUTH_HINTS = ("api key", "api_key", "unauthori", "authenticat", "auth_error", "permission",
               "forbidden", "unaccepted_terms")  # fmt: skip
_LIMIT_HINTS = ("quota", "rate limit", "rate_limit", "too many", "too_many", "concurrent",
                "busy", "credits", "throttl", "queue_overflow", "resource_exhausted")  # fmt: skip


def _hint(text: str) -> type[ProviderError] | None:
    lowered = text.lower()
    if any(h in lowered for h in _AUTH_HINTS):
        return AuthenticationError
    if any(h in lowered for h in _LIMIT_HINTS):
        return RateLimitError
    return None


def _error_for_status(status: int | None, message: str, *, code: str = "") -> ProviderError:
    if code.lower() in ("quota_exceeded", "insufficient_credits"):
        # not a transient rate limit: retrying does not help until the quota is raised
        return RateLimitError(message, provider=_PROVIDER, status_code=status, retryable=False)
    if status in (401, 403):
        return AuthenticationError(message, provider=_PROVIDER, status_code=status)
    if status == 429:
        return RateLimitError(message, provider=_PROVIDER, status_code=status)
    if status in (408, 504):
        return ProviderTimeoutError(message, provider=_PROVIDER, status_code=status)
    retryable = status is not None and status >= 500
    return ProviderError(message, provider=_PROVIDER, retryable=retryable, status_code=status)


def _describe(data: Any) -> tuple[str, str]:
    """``(message, code)`` of an ElevenLabs error body (``{"detail": {...} | [...] | str}``)."""
    detail = data.get("detail", data) if isinstance(data, Mapping) else data
    if isinstance(detail, Mapping):
        code = next(
            (str(detail[k]) for k in ("code", "status", "type", "error") if detail.get(k)), ""
        )
        message = str(detail.get("message") or detail.get("msg") or "")
        if detail.get("param"):
            message += f" (param: {detail['param']})"
        if detail.get("request_id"):
            message += f" [request_id {detail['request_id']}]"
        return message, code
    if isinstance(detail, list):
        parts = []
        for item in detail:
            if isinstance(item, Mapping):
                loc = ".".join(str(p) for p in item.get("loc") or ())
                parts.append(f"{loc}: {item.get('msg', '')}" if loc else str(item.get("msg", "")))
            else:
                parts.append(str(item))
        return "; ".join(parts), "validation_error"
    return (str(detail) if detail is not None else ""), ""


def _http_error(status: int, body: str) -> ProviderError:
    try:
        message, code = _describe(json.loads(body))
    except ValueError:
        message, code = body.strip()[:500], ""
    text = f"ElevenLabs HTTP {status}"
    if code:
        text += f" {code}"
    if message:
        text += f": {message}"
    return _error_for_status(status, text, code=code)


def _ws_error(msg: Mapping[str, Any], what: str) -> ProviderError:
    """An in-band WebSocket error (``{"error": ..., "message": ..., "code": ...}``)."""
    error = msg.get("error")
    message = msg.get("message")
    if isinstance(error, Mapping):
        message = message or error.get("message")
        name = str(error.get("code") or error.get("type") or "error")
    else:
        name = str(error or "error")
    raw_code = msg.get("code")
    code = raw_code if isinstance(raw_code, int) and not isinstance(raw_code, bool) else None
    text = f"ElevenLabs {what} error {name}"
    if message and str(message) != name:
        text += f": {message}"
    if msg.get("param"):
        text += f" (param: {msg['param']})"
    if code is not None:
        text += f" (code {code})"
    if code is not None and 400 <= code < 600:  # an HTTP status
        return _error_for_status(code, text, code=name)
    kind = _hint(f"{name} {message or ''}")
    if kind is AuthenticationError:
        return AuthenticationError(text, provider=_PROVIDER, status_code=code)
    if kind is RateLimitError:
        return RateLimitError(text, provider=_PROVIDER, status_code=code)
    return ProviderError(text, provider=_PROVIDER, status_code=code)


def _close_error(code: int | None, reason: str, what: str) -> ProviderError:
    """Map a WebSocket close (``1008`` policy violation with a reason, ``1011``...)."""
    detail = f"code {code}" if code is not None else "no close frame"
    if reason:
        detail += f": {reason}"
    text = f"ElevenLabs {what} WebSocket closed ({detail})"
    kind = _hint(reason)
    if kind is AuthenticationError:
        return AuthenticationError(text, provider=_PROVIDER, status_code=code)
    if kind is RateLimitError:
        return RateLimitError(text, provider=_PROVIDER, status_code=code)
    if code in (1003, 1007, 1008, 1009):  # rejected input / policy violation / too big
        return ProviderError(text, provider=_PROVIDER, status_code=code)
    return ProviderConnectionError(text, provider=_PROVIDER, status_code=code)


def _closed(exc: ConnectionClosed, what: str) -> ProviderError:
    frame = exc.rcvd
    error = _close_error(
        int(frame.code) if frame is not None else None, frame.reason if frame else "", what
    )
    error.__cause__ = exc
    return error


async def _ws_connect(
    url: str, headers: Mapping[str, str], *, open_timeout: float, what: str
) -> ClientConnection:
    from websockets.asyncio.client import connect
    from websockets.exceptions import InvalidHandshake, InvalidStatus, InvalidURI

    try:
        return await connect(
            url,
            additional_headers=dict(headers),
            open_timeout=open_timeout,
            max_size=_MAX_MESSAGE_BYTES,
            close_timeout=2.0,
        )
    except InvalidStatus as exc:
        body = exc.response.body or b""
        raise _http_error(exc.response.status_code, body.decode("utf-8", "replace")) from exc
    except InvalidURI as exc:
        raise ConfigurationError(f"invalid ElevenLabs URL: {exc}") from exc
    except TimeoutError as exc:
        raise ProviderTimeoutError(
            f"timed out connecting to the ElevenLabs {what} API", provider=_PROVIDER
        ) from exc
    except (OSError, InvalidHandshake) as exc:
        raise ProviderConnectionError(
            f"cannot connect to the ElevenLabs {what} API: {exc}", provider=_PROVIDER
        ) from exc


async def _close_ws(ws: ClientConnection) -> None:
    with contextlib.suppress(Exception):
        async with asyncio.timeout(2.0):
            await ws.close()


def _parse(raw: str | bytes, what: str) -> dict[str, Any] | None:
    if isinstance(raw, bytes):
        logger.debug("ElevenLabs %s: ignoring a binary message", what)
        return None
    try:
        msg = json.loads(raw)
    except ValueError:
        logger.warning("ElevenLabs %s: ignoring invalid JSON: %.200s", what, raw)
        return None
    return msg if isinstance(msg, dict) else None


def _task_error(task: asyncio.Task[Any]) -> BaseException | None:
    return task.exception() if task.done() and not task.cancelled() else None


async def _raise_task_error(
    reader: asyncio.Task[Any], writer: asyncio.Task[Any], *, grace: float = 0.5
) -> None:
    """Raise the error of ``reader`` or else ``writer``, preferring the server's reason.

    A writer failing because the server hung up is usually followed by the server's
    explanation on the reader side: wait ``grace`` seconds for it.
    """
    if _task_error(writer) is not None and not reader.done():
        await asyncio.wait((reader,), timeout=grace)
    for task in (reader, writer):
        error = _task_error(task)
        if error is not None:
            raise error


# ------------------------------------------------------------------ TTS: server messages
def _context_id(msg: Mapping[str, Any]) -> str | None:
    value = msg.get("contextId") or msg.get("context_id")
    return value if isinstance(value, str) else None


def _is_final(msg: Mapping[str, Any]) -> bool:
    return msg.get("isFinal") is True or msg.get("is_final") is True


@dataclass(slots=True)
class _Alignment:
    kind: str
    """``"original"`` (``alignment``) or ``"normalized"`` (``normalizedAlignment``)."""
    chars: list[str]
    starts: list[float]
    """Seconds, in the server's time base."""
    ends: list[float]


_ALIGNMENT_KEYS = {
    "original": ("alignment",),
    "normalized": ("normalizedAlignment", "normalized_alignment"),
}
_STARTS_KEYS = ("charStartTimesMs", "char_start_times_ms", "charsStartTimesMs")
_DURATIONS_KEYS = ("charDurationsMs", "char_durations_ms", "charsDurationsMs")


def _alignment(msg: Mapping[str, Any], prefer: str) -> _Alignment | None:
    """The message's character alignment (camelCase TTS or snake_case dialogue fields)."""
    other = "normalized" if prefer == "original" else "original"
    for kind in (prefer, other):
        for key in _ALIGNMENT_KEYS[kind]:
            data = msg.get(key)
            if not isinstance(data, Mapping):
                continue
            chars = data.get("chars")
            starts = next((data[k] for k in _STARTS_KEYS if isinstance(data.get(k), list)), None)
            durations = next(
                (data[k] for k in _DURATIONS_KEYS if isinstance(data.get(k), list)), None
            )
            if not isinstance(chars, list) or starts is None or durations is None:
                continue
            n = min(len(chars), len(starts), len(durations))
            if not n:
                continue
            try:
                s = [max(0.0, float(v) / 1000.0) for v in starts[:n]]
                e = [s[i] + max(0.0, float(durations[i]) / 1000.0) for i in range(n)]
            except (TypeError, ValueError):
                continue
            return _Alignment(kind, [str(c) for c in chars[:n]], s, e)
    return None


class _WordAssembler:
    """Characters with stream-timeline times -> complete :class:`WordTiming` s."""

    def __init__(self) -> None:
        self._chars: list[str] = []
        self._start = 0.0
        self._end = 0.0
        self._done: list[WordTiming] = []

    def add(self, char: str, start: float, end: float) -> None:
        for c in char:
            if c.isspace():
                self._close()
                continue
            if not self._chars:
                self._start = start
                self._end = end
            self._chars.append(c)
            self._end = max(self._end, end)

    def _close(self) -> None:
        if self._chars:
            self._done.append(WordTiming("".join(self._chars), self._start, self._end))
            self._chars = []

    def pop(self, *, final: bool = False) -> list[WordTiming]:
        """Completed words; with ``final``, the word still being spelled too."""
        if final:
            self._close()
        words, self._done = self._done, []
        return words


# ------------------------------------------------------------------ TTS: wire protocols
class _Protocol:
    """``/v1/text-to-speech/{voice_id}/multi-stream-input`` (Flash, Turbo, Multilingual)."""

    name = "tts"

    def __init__(self, tts: ElevenLabsTTS) -> None:
        self.tts = tts

    def url(self, voice: str) -> str:
        tts = self.tts
        params: list[tuple[str, str]] = [
            ("model_id", tts.model),
            ("output_format", f"pcm_{tts.sample_rate}"),
            ("inactivity_timeout", str(tts.inactivity_timeout)),
        ]
        if tts.auto_mode:
            params.append(("auto_mode", "true"))
        params += self._common_params()
        if tts.enable_ssml_parsing:
            params.append(("enable_ssml_parsing", "true"))
        path = f"/v1/text-to-speech/{quote(voice, safe='')}/multi-stream-input"
        return f"{_ws_base(tts.base_url)}{path}?{urlencode(params)}"

    def _common_params(self) -> list[tuple[str, str]]:
        tts = self.tts
        params: list[tuple[str, str]] = []
        if tts.capabilities.word_timestamps:
            params.append(("sync_alignment", "true"))
        language = tts._language_code()
        if language:
            params.append(("language_code", language))
        if tts.apply_text_normalization is not None:
            params.append(("apply_text_normalization", tts.apply_text_normalization))
        if tts.seed is not None:
            params.append(("seed", str(tts.seed)))
        if not tts.enable_logging:
            params.append(("enable_logging", "false"))
        return params

    def init_message(self, ctx_id: str, voice: str) -> dict[str, Any]:
        tts = self.tts
        msg: dict[str, Any] = {"text": " ", "context_id": ctx_id}
        if tts.voice_settings:
            msg["voice_settings"] = dict(tts.voice_settings)
        if tts.chunk_length_schedule:
            msg["generation_config"] = {"chunk_length_schedule": list(tts.chunk_length_schedule)}
        if tts.pronunciation_dictionaries:
            msg["pronunciation_dictionary_locators"] = list(tts.pronunciation_dictionaries)
        return msg

    def text_message(self, ctx_id: str, text: str, voice: str, *, flush: bool) -> dict[str, Any]:
        msg: dict[str, Any] = {"text": text, "context_id": ctx_id}
        if flush:
            msg["flush"] = True
        return msg

    def flush_message(self, ctx_id: str) -> dict[str, Any]:
        return {"context_id": ctx_id, "flush": True}

    def close_message(self, ctx_id: str) -> dict[str, Any]:
        return {"context_id": ctx_id, "close_context": True}

    def keepalive_message(self, ctx_id: str) -> dict[str, Any]:
        return {"context_id": ctx_id, "text": ""}

    def keepalive_interval(self) -> float:
        if self.tts.keepalive_interval is not None:
            return self.tts.keepalive_interval
        return max(1.0, self.tts.inactivity_timeout / 2)


class _DialogueProtocol(_Protocol):
    """``/v1/text-to-dialogue/multi-stream-input``: Eleven v3 models (snake_case fields)."""

    name = "dialogue"

    def url(self, voice: str) -> str:
        tts = self.tts
        params: list[tuple[str, str]] = [
            ("model_id", tts.model),
            ("output_format", f"pcm_{tts.sample_rate}"),
            *self._common_params(),
        ]
        path = "/v1/text-to-dialogue/multi-stream-input"
        return f"{_ws_base(tts.base_url)}{path}?{urlencode(params)}"

    def init_message(self, ctx_id: str, voice: str) -> dict[str, Any]:
        tts = self.tts
        msg: dict[str, Any] = {"context_id": ctx_id, "voices": [voice]}
        stability = tts.voice_settings.get("stability")
        if stability is not None:  # the only voice setting dialogue models take
            msg["voice_settings"] = {"stability": stability}
        if tts.pronunciation_dictionaries:
            msg["pronunciation_dictionary_locators"] = list(tts.pronunciation_dictionaries)
        return msg

    def text_message(self, ctx_id: str, text: str, voice: str, *, flush: bool) -> dict[str, Any]:
        msg: dict[str, Any] = {"context_id": ctx_id, "inputs": [{"text": text, "voice_id": voice}]}
        if flush:
            msg["flush"] = True
        return msg

    def keepalive_message(self, ctx_id: str) -> dict[str, Any]:
        return {"context_id": ctx_id, "keep_alive": True}

    def keepalive_interval(self) -> float:
        if self.tts.keepalive_interval is not None:
            return self.tts.keepalive_interval
        return 10.0  # dialogue contexts and connections time out after a fixed 20 s


# ------------------------------------------------------------------------------- TTS
@register_provider(
    "tts",
    "elevenlabs",
    description="ElevenLabs Flash v2.5 / v3: multi-context WebSocket streaming + alignment",
    default_model="eleven_flash_v2_5",
    models=(
        "eleven_flash_v2_5",
        "eleven_turbo_v2_5",
        "eleven_multilingual_v2",
        "eleven_flash_v2",
        "eleven_v3_conversational",
        "eleven_v3",
    ),
    env=API_KEY_ENV,
    extra=None,
    requires=("websockets", "httpx"),
    local=False,
)
class ElevenLabsTTS(TTS):
    """ElevenLabs text-to-speech.

    :meth:`stream` pushes text into ElevenLabs *contexts* over one multi-context WebSocket
    per voice, shared by all streams of this instance (opened lazily or by
    :meth:`warmup`, reopened when ElevenLabs closes it). Every segment (text between
    :meth:`~voice_agent_next.tts.SynthesizeStream.flush` calls; a whole reply in the
    cascade) is one context: it opens with the voice settings, receives the text as it is
    pushed and ends with ``flush`` + ``close_context``; its ``isFinal`` ends the segment.
    Segments play in order even when their generation overlaps. Closing a stream
    (barge-in) closes its unfinished contexts and drops their late audio; the socket
    stays open for the next reply.

    Text is sent at word boundaries only (a partial word waits for the next push). A push
    that ends a sentence — or the first clause of a segment — is sent with ``flush: true``
    so ElevenLabs generates it at once instead of waiting for more text (the cascade
    pushes whole sentences and a short first clause); ``auto_mode`` handles the rest.

    With ``word_timestamps=True`` (default) the socket asks for ``sync_alignment`` and the
    character alignment becomes items with an empty ``frame`` and ``words``
    (:class:`~voice_agent_next.stt.WordTiming` in seconds **from the start of the
    stream's audio**, across segments), which the cascade uses to truncate barge-ins at
    the exact word heard. ElevenLabs documents alignment times as relative to each
    returned chunk, while some clients observe context-absolute times (reported for
    ``normalizedAlignment``); the provider tells the two apart from the audio positions
    and maps both onto the stream timeline.

    Eleven v3 models (``eleven_v3``, ``eleven_v3_conversational``) use the Text to Dialogue
    multi-context WebSocket instead (``voices`` registration, ``inputs`` messages, only the
    ``stability`` voice setting, contexts kept alive with ``keep_alive``).

    :meth:`synthesize` (one complete text) streams raw PCM from the HTTP streaming
    endpoint; it returns no word timings.

    Args:
        model: ``eleven_flash_v2_5`` (default, ~75 ms model latency),
            ``eleven_turbo_v2_5``, ``eleven_multilingual_v2``, ``eleven_flash_v2``,
            ``eleven_v3_conversational``, ``eleven_v3``...
        voice: ElevenLabs voice id (default :data:`DEFAULT_VOICE`).
        api_key: defaults to ``$ELEVEN_API_KEY``, then ``$ELEVENLABS_API_KEY``.
        sample_rate: raw PCM output rate, one of :data:`TTS_SAMPLE_RATES`.
        language: ISO 639-1 code (``"de"``, ``"pt-BR"`` -> ``"pt"``) enforcing the language
            and its text normalization; sent to the v2.5 and v3 models only (Multilingual
            v2 rejects it and detects the language itself).
        stability, similarity_boost, style: voice settings (0-1); ``None`` keeps the
            voice's stored settings. Dialogue (v3) models only take ``stability``.
        use_speaker_boost: voice setting.
        speed: speaking rate, 0.7-1.2.
        auto_mode: let ElevenLabs trigger generations itself (lower latency with
            sentence-wise input). ``None`` = on unless ``chunk_length_schedule`` is set.
        chunk_length_schedule: characters buffered before each generation (50-500 each),
            e.g. ``[50, 120, 160, 290]``; used when ``auto_mode`` is off.
        apply_text_normalization: ``"auto"``, ``"on"`` or ``"off"`` (numbers, dates...);
            ``None`` keeps the API default.
        flush_sentences: send ``flush: true`` with pushes that end a sentence (see above).
        word_timestamps: request character alignment and report word timings.
        alignment: which alignment becomes words: ``"original"`` (the text as sent) or
            ``"normalized"`` (what was spoken, e.g. numbers spelled out). ``None`` =
            original, or normalized when pronunciation dictionaries are used (the
            original-text alignment has been seen to restart mid-sentence then).
        pronunciation_dictionaries: ``(dictionary_id, version_id)`` pairs (or locator
            mappings), at most 3.
        seed: best-effort deterministic sampling.
        enable_ssml_parsing: parse SSML (``<break>``, ``<phoneme>``) in the text.
        enable_logging: ``False`` requests zero-retention mode (enterprise plans).
        inactivity_timeout: seconds before ElevenLabs closes an idle context (max 180).
        keepalive_interval: keep-alive period of contexts waiting for text; ``None`` = half
            the ``inactivity_timeout`` (10 s for the dialogue API's fixed 20 s timeout).
        streaming: ``False`` synthesizes sentence by sentence over HTTP instead of the
            WebSocket (segment-level alignment only).
        base_url: API origin (``wss://`` is derived from it for WebSockets).
        region: ``"us"`` (US-only servers), ``"eu"``, ``"in"`` or ``"sg"`` (data
            residency); an alternative to ``base_url``.
        http_client: optional ``httpx.AsyncClient`` for :meth:`synthesize` (not closed by
            :meth:`aclose`).
        connect_timeout: connection / handshake timeout in seconds.
        receive_timeout: after the end of a segment's input, fail with
            :class:`~voice_agent_next.errors.ProviderTimeoutError` if ElevenLabs stays
            silent this long; also the HTTP read timeout of :meth:`synthesize`.
    """

    provider = "elevenlabs"

    def __init__(
        self,
        *,
        model: str = "eleven_flash_v2_5",
        voice: str | None = None,
        api_key: str | None = None,
        sample_rate: int = 24_000,
        language: str | None = None,
        stability: float | None = None,
        similarity_boost: float | None = None,
        style: float | None = None,
        use_speaker_boost: bool | None = None,
        speed: float | None = None,
        auto_mode: bool | None = None,
        chunk_length_schedule: Sequence[int] | None = None,
        apply_text_normalization: Literal["auto", "on", "off"] | None = None,
        flush_sentences: bool = True,
        word_timestamps: bool = True,
        alignment: Literal["original", "normalized"] | None = None,
        pronunciation_dictionaries: Sequence[tuple[str, str] | Mapping[str, str]] | None = None,
        seed: int | None = None,
        enable_ssml_parsing: bool = False,
        enable_logging: bool = True,
        inactivity_timeout: int = 180,
        keepalive_interval: float | None = None,
        streaming: bool = True,
        base_url: str | None = None,
        region: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        connect_timeout: float = 10.0,
        receive_timeout: float = 10.0,
        clean_text: bool = True,
    ) -> None:
        if sample_rate not in TTS_SAMPLE_RATES:
            raise ConfigurationError(
                f"ElevenLabs TTS sample_rate must be one of {TTS_SAMPLE_RATES}, got {sample_rate}"
            )
        for name, value in (
            ("stability", stability),
            ("similarity_boost", similarity_boost),
            ("style", style),
        ):
            _check_range(name, value, 0.0, 1.0)
        _check_range("speed", speed, 0.7, 1.2)
        _check_range("seed", seed, 0, 4_294_967_295)
        _check_range("inactivity_timeout", inactivity_timeout, 1, 180)
        if keepalive_interval is not None and keepalive_interval <= 0:
            raise ConfigurationError(f"keepalive_interval must be > 0, got {keepalive_interval}")
        schedule = list(chunk_length_schedule or ())
        if chunk_length_schedule is not None and (
            not schedule or any(not 50 <= v <= 500 for v in schedule)
        ):
            raise ConfigurationError(
                f"chunk_length_schedule needs values within [50, 500], got {schedule}"
            )
        if apply_text_normalization not in (None, "auto", "on", "off"):
            raise ConfigurationError(
                "apply_text_normalization must be 'auto', 'on' or 'off', "
                f"got {apply_text_normalization!r}"
            )
        if alignment not in (None, "original", "normalized"):
            raise ConfigurationError(
                f"alignment must be 'original' or 'normalized', got {alignment!r}"
            )
        locators = _locators(pronunciation_dictionaries)
        if len(locators) > 3:
            raise ConfigurationError("ElevenLabs accepts at most 3 pronunciation dictionaries")
        super().__init__(
            model=model,
            sample_rate=sample_rate,
            channels=1,
            capabilities=TTSCapabilities(streaming=streaming, word_timestamps=word_timestamps),
            voice=voice or DEFAULT_VOICE,
            clean_text=clean_text,
        )
        self._api_key = _resolve_api_key(api_key)
        self.base_url = _resolve_base_url(base_url, region)
        self.language = language
        self.voice_settings: dict[str, Any] = {
            k: v
            for k, v in (
                ("stability", stability),
                ("similarity_boost", similarity_boost),
                ("style", style),
                ("use_speaker_boost", use_speaker_boost),
                ("speed", speed),
            )
            if v is not None
        }
        self.chunk_length_schedule = schedule
        self.auto_mode = not schedule if auto_mode is None else auto_mode
        self.apply_text_normalization = apply_text_normalization
        self.flush_sentences = flush_sentences
        self.alignment: str = alignment or ("normalized" if locators else "original")
        self.pronunciation_dictionaries = locators
        self.seed = seed
        self.enable_ssml_parsing = enable_ssml_parsing
        self.enable_logging = enable_logging
        self.inactivity_timeout = inactivity_timeout
        self.keepalive_interval = keepalive_interval
        self.connect_timeout = connect_timeout
        self.receive_timeout = receive_timeout
        self.dialogue = _is_dialogue_model(model)
        """Eleven v3 models stream through the Text to Dialogue API."""
        self._protocol = _DialogueProtocol(self) if self.dialogue else _Protocol(self)
        self._http = http_client
        self._owns_http = http_client is None
        self._conns: dict[str, _Connection] = {}
        self._conn_lock: asyncio.Lock | None = None
        self._conn_lock_loop: asyncio.AbstractEventLoop | None = None
        self._time_bases: dict[str, str] = {}
        """Detected alignment time base per protocol and alignment kind."""
        self._warn_ignored_options()

    def _warn_ignored_options(self) -> None:
        if self.language and not self._language_code():
            logger.warning(
                "ElevenLabs %s takes no language code: language=%r is not sent "
                "(the model detects the language itself)",
                self.model,
                self.language,
            )
        if not self.dialogue:
            return
        ignored = [k for k in self.voice_settings if k != "stability"]
        if self.chunk_length_schedule:
            ignored.append("chunk_length_schedule")
        if self.enable_ssml_parsing:
            ignored.append("enable_ssml_parsing")
        if ignored:
            logger.warning(
                "ElevenLabs %s streams through the Text to Dialogue API, which ignores: %s",
                self.model,
                ", ".join(ignored),
            )

    # ---------------------------------------------------------------- requests
    def _headers(self) -> dict[str, str]:
        return {"xi-api-key": self._api_key}

    def _language_code(self) -> str | None:
        if self.model in _LANGUAGE_MODELS or self.dialogue:
            return _language_code(self.language)
        return None

    def ws_url(self, voice: str | None = None) -> str:
        """The streaming WebSocket URL for ``voice`` (query string included, no key)."""
        return self._protocol.url(voice or self.voice or DEFAULT_VOICE)

    def _http_request(self, text: str, voice: str) -> tuple[str, dict[str, str], dict[str, Any]]:
        """``(url, query, JSON body)`` of the HTTP streaming endpoint."""
        params = {"output_format": f"pcm_{self.sample_rate}"}
        if not self.enable_logging:
            params["enable_logging"] = "false"
        body: dict[str, Any]
        if self.dialogue:
            url = f"{self.base_url}/v1/text-to-dialogue/stream"
            body = {"inputs": [{"text": text, "voice_id": voice}], "model_id": self.model}
            if "stability" in self.voice_settings:
                body["settings"] = {"stability": self.voice_settings["stability"]}
        else:
            url = f"{self.base_url}/v1/text-to-speech/{quote(voice, safe='')}/stream"
            body = {"text": text, "model_id": self.model}
            if self.voice_settings:
                body["voice_settings"] = dict(self.voice_settings)
        language = self._language_code()
        if language:
            body["language_code"] = language
        if self.apply_text_normalization is not None:
            body["apply_text_normalization"] = self.apply_text_normalization
        if self.seed is not None:
            body["seed"] = self.seed
        if self.pronunciation_dictionaries:
            body["pronunciation_dictionary_locators"] = list(self.pronunciation_dictionaries)
        return url, params, body

    def _time_base(self, kind: str, offset: float, first_start: float) -> str:
        """Whether alignment times are relative to their ``chunk`` or to the ``context``.

        ``offset`` is the audio the context produced before the aligned chunk and
        ``first_start`` the chunk's first character time: chunk-relative times restart
        near 0, context-absolute ones continue from ``offset``. The guess is kept for
        good once the context has produced enough audio to tell them apart reliably.
        """
        key = f"{self._protocol.name}/{kind}"
        known = self._time_bases.get(key)
        if known is not None:
            return known
        guess = "context" if offset > 0 and first_start >= offset / 2 else "chunk"
        if offset >= _TIME_BASE_DECIDE_AFTER:
            self._time_bases[key] = guess
            logger.debug("ElevenLabs %s alignment times are %s-relative", key, guess)
        return guess

    # --------------------------------------------------------------- transport
    async def _connection(self, voice: str) -> _Connection:
        """The shared WebSocket for ``voice``, (re)connecting when needed."""
        loop = asyncio.get_running_loop()
        if self._conn_lock is None or self._conn_lock_loop is not loop:
            self._conn_lock, self._conn_lock_loop = asyncio.Lock(), loop
        url = self._protocol.url(voice)
        async with self._conn_lock:
            conn = self._conns.get(url)
            if conn is not None and (conn.closed or conn.loop is not loop):
                if conn.loop is loop:
                    await conn.aclose()
                del self._conns[url]
                conn = None
            if conn is None:
                ws = await _ws_connect(
                    url, self._headers(), open_timeout=self.connect_timeout, what="TTS"
                )
                conn = self._conns[url] = _Connection(ws, self._protocol)
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
        return _ElevenLabsChunkedStream(self, text, voice=voice)

    def _create_stream(self, *, voice: str | None) -> SynthesizeStream:
        return _ElevenLabsSynthesizeStream(self, voice=voice)

    async def warmup(self) -> None:
        """Open the streaming WebSocket ahead of the first reply (saves TLS + WS setup)."""
        if self.capabilities.streaming:
            await self._connection(self.voice or DEFAULT_VOICE)

    async def aclose(self) -> None:
        conns, self._conns = list(self._conns.values()), {}
        loop = asyncio.get_running_loop()
        await asyncio.gather(*(c.aclose() for c in conns if c.loop is loop))
        if self._http is not None and self._owns_http:
            http, self._http = self._http, None
            await http.aclose()


class _Context:
    """One ElevenLabs context (``context_id``) on a shared connection."""

    def __init__(self, conn: _Connection) -> None:
        self.conn = conn
        self.id = uuid.uuid4().hex
        self.messages: Chan[dict[str, Any] | ProviderError] = Chan()
        self.last_activity = now()
        """Last message from the server (the receive watchdog's reference)."""
        self.last_send = now()
        self.unflushed = False
        """Text was sent since the last ``flush``."""
        self.closed_at: float | None = None
        """When ``close_context`` was sent; the receive watchdog only runs after it."""
        self.ended = False
        """The server finished the context (``isFinal``/error) or the connection failed."""
        self.finished = False
        """The owning stream consumed the context (or gave up on it)."""
        self.abandoned_at: float | None = None

    def deliver(self, item: dict[str, Any] | ProviderError) -> None:
        self.last_activity = now()
        if isinstance(item, ProviderError) or _is_final(item):
            self.ended = True
        if not self.messages.closed:
            self.messages.send_nowait(item)


class _Connection:
    """A multi-context WebSocket shared by the streams of one :class:`ElevenLabsTTS`.

    A reader task routes messages to their context by ``contextId`` and drops the late
    audio of cancelled contexts; a keeper task sends keep-alives for contexts that wait
    for text. When the socket closes, every open context receives an error.
    """

    def __init__(self, ws: ClientConnection, protocol: _Protocol) -> None:
        self.ws = ws
        self.protocol = protocol
        self.loop = asyncio.get_running_loop()
        self.closed = False
        self._contexts: dict[str, _Context] = {}
        self._failure: ProviderError | None = None
        """An error not tied to a context: the server usually closes right after it."""
        self._slot_freed = asyncio.Event()
        self._reader = asyncio.create_task(self._read_loop(), name="elevenlabs-tts-reader")
        self._keeper = asyncio.create_task(
            self._keepalive_loop(protocol.keepalive_interval()), name="elevenlabs-tts-keepalive"
        )

    # ----------------------------------------------------------------- contexts
    def new_context(self) -> _Context:
        ctx = _Context(self)
        self._contexts[ctx.id] = ctx
        return ctx

    def release(self, ctx: _Context) -> None:
        """Stop routing ``ctx`` (it finished, or will never be used again)."""
        if self._contexts.pop(ctx.id, None) is not None:
            self._slot_freed.set()

    def abandon(self, ctx: _Context) -> None:
        """``ctx`` was cancelled: drop its late messages, keep its slot until ``isFinal``."""
        ctx.abandoned_at = now()

    def active(self) -> int:
        t = now()
        for ctx in list(self._contexts.values()):
            if ctx.abandoned_at is not None and t - ctx.abandoned_at > _ABANDON_GRACE:
                self.release(ctx)
        return len(self._contexts)

    async def wait_for_slot(self, timeout: float) -> None:
        """Wait until fewer than :data:`MAX_CONTEXTS` contexts are open."""
        deadline = now() + timeout
        while not self.closed and self.active() >= MAX_CONTEXTS:
            remaining = deadline - now()
            if remaining <= 0:
                logger.warning("ElevenLabs TTS: %d contexts still open", self.active())
                return
            self._slot_freed.clear()
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(min(remaining, _ABANDON_GRACE)):
                    await self._slot_freed.wait()

    # ------------------------------------------------------------------ sending
    async def send(self, msg: Mapping[str, Any]) -> None:
        from websockets.exceptions import ConnectionClosed

        if self.closed:
            raise ProviderConnectionError("ElevenLabs TTS WebSocket closed", provider=_PROVIDER)
        try:
            await self.ws.send(json.dumps(msg))
        except ConnectionClosed as exc:
            self.closed = True
            # the reader reports the server's reason; a sender only needs to reconnect
            raise ProviderConnectionError(
                f"ElevenLabs TTS WebSocket closed: {exc}", provider=_PROVIDER
            ) from exc

    async def _keepalive_loop(self, interval: float) -> None:
        """Keep contexts that wait for more text (a slow LLM, a long tool call) alive."""
        while not self.closed:
            await asyncio.sleep(min(interval / 4, 5.0))
            t = now()
            for ctx in list(self._contexts.values()):
                if (
                    ctx.ended
                    or ctx.closed_at is not None
                    or ctx.abandoned_at is not None
                    or t - ctx.last_send < interval
                ):
                    continue
                ctx.last_send = t
                try:
                    await self.send(self.protocol.keepalive_message(ctx.id))
                except ProviderError:
                    return

    # ---------------------------------------------------------------- receiving
    async def _read_loop(self) -> None:
        from websockets.exceptions import ConnectionClosed

        error: ProviderError | None = None
        try:
            async for raw in self.ws:
                msg = _parse(raw, "TTS")
                if msg is not None:
                    self._dispatch(msg)
        except ConnectionClosed as exc:
            error = _closed(exc, "TTS")
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("ElevenLabs TTS reader failed")
            error = ProviderConnectionError(
                f"ElevenLabs TTS reader failed: {exc!r}", provider=_PROVIDER
            )
        finally:
            self.closed = True
            if self._failure is not None:  # the server explained itself before closing
                error = self._failure
            elif error is None:  # a clean close
                error = _close_error(self.ws.close_code, self.ws.close_reason or "", "TTS")
            contexts, self._contexts = list(self._contexts.values()), {}
            for ctx in contexts:
                ctx.deliver(error)
            self._slot_freed.set()

    def _dispatch(self, msg: dict[str, Any]) -> None:
        ctx_id = _context_id(msg)
        ctx = self._contexts.get(ctx_id) if ctx_id is not None else None
        if msg.get("error"):
            error = _ws_error(msg, "TTS")
            if ctx is not None:
                ctx.deliver(error)
            elif ctx_id is None:  # not tied to a context: fail every open one
                logger.warning("%s", error)
                self._failure = error
                for c in list(self._contexts.values()):
                    c.deliver(error)
            return
        if ctx is None:
            return  # a late message of a released context
        if ctx.abandoned_at is None:
            ctx.deliver(msg)
        if _is_final(msg):
            self.release(ctx)

    async def aclose(self) -> None:
        self.closed = True
        await _close_ws(self.ws)
        await cancel_and_wait(self._reader, self._keeper)


class _ElevenLabsChunkedStream(ChunkedStream):
    """HTTP streaming endpoint with raw PCM output, pushed as it arrives."""

    async def _run(self) -> None:
        import httpx

        tts: ElevenLabsTTS = self._tts  # type: ignore[assignment]
        if not self.text.strip():
            return
        url, params, body = tts._http_request(self.text, self.voice or DEFAULT_VOICE)
        try:
            async with tts._http_client().stream(
                "POST", url, params=params, json=body, headers=tts._headers()
            ) as response:
                if response.status_code >= 400:
                    raw = await response.aread()
                    raise _http_error(response.status_code, raw.decode("utf-8", "replace"))
                async for chunk in response.aiter_bytes():
                    if chunk:
                        self._push_audio(chunk)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"ElevenLabs TTS request timed out: {exc!r}", provider=_PROVIDER
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderConnectionError(
                f"ElevenLabs TTS request failed: {exc!r}", provider=_PROVIDER
            ) from exc


@dataclass(eq=False)
class _Segment:
    """Text between two flushes: one context, more if ElevenLabs ended one early."""

    contexts: Chan[_Context] = field(default_factory=Chan)
    text: list[str] = field(default_factory=list)
    current: _Context | None = None
    pending: str = ""
    """Pushed text not sent yet (a word still being spelled)."""
    flushes: int = 0


def _phrase_end(text: str, *, first: bool) -> bool:
    """Whether ``text`` ends a sentence (or, for the first phrase, a clause)."""
    stripped = text.rstrip()
    if not stripped:
        return False
    return bool(_SENTENCE_END.search(stripped)) or (first and bool(_CLAUSE_END.search(stripped)))


class _ElevenLabsSynthesizeStream(SynthesizeStream):
    """Native text streaming: a feeder sends text, a player emits audio in segment order."""

    def __init__(self, tts: ElevenLabsTTS, *, voice: str | None) -> None:
        self._voice = voice or tts.voice or DEFAULT_VOICE
        self._contexts: list[_Context] = []
        super().__init__(tts, voice=voice)

    async def _run(self) -> None:
        segments: Chan[_Segment] = Chan()
        feeder = asyncio.create_task(self._feed(segments), name="elevenlabs-tts-feed")
        player = asyncio.create_task(self._play(segments), name="elevenlabs-tts-play")
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
        # its context is cancelled by _cancel_contexts(), not finished gracefully
        segments.close()

    async def _push_segment_text(self, segment: _Segment, text: str) -> None:
        tts: ElevenLabsTTS = self._tts  # type: ignore[assignment]
        segment.text.append(text)
        segment.pending += text
        # send whole words only: a partial word waits for the rest of it
        cut = max(segment.pending.rfind(c) for c in (" ", "\n", "\t", "\r")) + 1
        if cut <= 0:
            return
        chunk, segment.pending = segment.pending[:cut], segment.pending[cut:]
        if not chunk.strip():
            return
        flush = (
            tts.flush_sentences
            and not segment.pending  # the push itself ended at a word boundary
            and _phrase_end(chunk, first=segment.flushes == 0)
        )
        await self._send_chunk(segment, chunk, flush=flush)

    async def _send_chunk(self, segment: _Segment, chunk: str, *, flush: bool) -> None:
        if not chunk.endswith(" "):
            chunk += " "  # ElevenLabs expects every text input to end with a space
        ctx = segment.current
        if ctx is None or ctx.ended:
            # first text of the segment, or ElevenLabs ended the context (inactivity):
            # continue the segment on a fresh context
            try:
                ctx = await self._open_context()
                await self._send_text(ctx, chunk, flush=flush)
            except ProviderConnectionError:
                logger.info("ElevenLabs TTS WebSocket was closed; reconnecting")
                ctx = await self._open_context()
                await self._send_text(ctx, chunk, flush=flush)
            segment.current = ctx
            segment.contexts.send_nowait(ctx)
        else:
            await self._send_text(ctx, chunk, flush=flush)
        if flush:
            segment.flushes += 1

    async def _open_context(self) -> _Context:
        tts: ElevenLabsTTS = self._tts  # type: ignore[assignment]
        conn = await tts._connection(self._voice)
        await conn.wait_for_slot(tts.receive_timeout)
        ctx = conn.new_context()
        self._contexts.append(ctx)
        try:
            await conn.send(conn.protocol.init_message(ctx.id, self._voice))
        except BaseException:
            ctx.finished = True
            conn.release(ctx)
            raise
        return ctx

    async def _send_text(self, ctx: _Context, text: str, *, flush: bool) -> None:
        await ctx.conn.send(ctx.conn.protocol.text_message(ctx.id, text, self._voice, flush=flush))
        ctx.last_send = now()
        ctx.unflushed = not flush

    async def _close_segment(self, segment: _Segment) -> None:
        rest, segment.pending = segment.pending, ""
        if rest.strip():
            await self._send_chunk(segment, rest, flush=True)
        ctx = segment.current
        if ctx is not None and not ctx.ended:
            protocol = ctx.conn.protocol
            if ctx.unflushed:  # generate what ElevenLabs still buffers...
                await ctx.conn.send(protocol.flush_message(ctx.id))
                ctx.unflushed = False
            await ctx.conn.send(protocol.close_message(ctx.id))  # ...then end with isFinal
            ctx.closed_at = ctx.last_send = now()
        segment.contexts.close()

    # ------------------------------------------------------------------ player
    async def _play(self, segments: Chan[_Segment]) -> None:
        async for segment in segments:
            async for ctx in segment.contexts:
                await self._play_context(ctx)
            self._segment_text = "".join(segment.text).strip() or None
            self._end_segment()

    async def _play_context(self, ctx: _Context) -> None:
        """Emit a context's audio and words until ``isFinal``.

        Only a completed context is released here: on errors, timeouts and cancellation
        :meth:`_cancel_contexts` closes it if it may still be generating.
        """
        tts: ElevenLabsTTS = self._tts  # type: ignore[assignment]
        base = self._audio_duration  # where this context's audio starts on the stream
        produced = 0.0  # seconds of audio the context produced before the current message
        bytes_per_second = 2 * tts.sample_rate * tts.channels
        words = _WordAssembler()
        empty = AudioFrame.empty(tts.sample_rate, tts.channels)
        while True:
            msg = await self._next_message(ctx, tts.receive_timeout)
            if isinstance(msg, ProviderError):
                raise msg
            final = _is_final(msg)
            data = msg.get("audio")
            audio = base64.b64decode(data) if isinstance(data, str) and data else b""
            if tts.capabilities.word_timestamps:
                aligned = _alignment(msg, tts.alignment)
                if aligned is not None:
                    mode = tts._time_base(aligned.kind, produced, aligned.starts[0])
                    offset = base + (produced if mode == "chunk" else 0.0)
                    for char, start, end in zip(
                        aligned.chars, aligned.starts, aligned.ends, strict=True
                    ):
                        words.add(char, offset + start, offset + end)
                done = words.pop(final=final)
                if done:  # words go out before (or with) the audio they start in
                    self._send(SynthesizedAudio(empty, self._request_id, self._segment_id,
                                                words=done))  # fmt: skip
            if audio:
                self._push_audio(audio)
                produced += len(audio) / bytes_per_second
            if final:
                break
        ctx.finished = True
        ctx.conn.release(ctx)

    @staticmethod
    async def _next_message(ctx: _Context, timeout: float) -> dict[str, Any] | ProviderError:
        # asyncio.timeout rather than wait_for: on Python 3.11, wait_for can swallow the
        # cancellation of an interrupted stream when a message arrives at the same time
        while True:
            wait = timeout
            if ctx.closed_at is not None:
                wait = timeout - (now() - max(ctx.closed_at, ctx.last_activity))
                if wait <= 0:
                    raise ProviderTimeoutError(
                        f"ElevenLabs sent nothing for {timeout:.1f}s after the end of the "
                        f"input (context {ctx.id})",
                        provider=_PROVIDER,
                    )
            try:
                async with asyncio.timeout(wait):
                    return await ctx.messages.recv()
            except TimeoutError:
                continue  # re-check: the input may have ended in the meantime

    # ----------------------------------------------------------- cancellation
    async def _cancel_contexts(self) -> None:
        """Close contexts that may still be generating (interruption, error or close)."""
        for ctx in self._contexts:
            if ctx.finished:
                continue
            ctx.finished = True
            if ctx.ended or ctx.conn.closed:
                ctx.conn.release(ctx)
                continue
            ctx.conn.abandon(ctx)
            try:
                async with asyncio.timeout(1.0):
                    await ctx.conn.send(ctx.conn.protocol.close_message(ctx.id))
            except Exception as exc:  # best effort: the socket may already be gone
                logger.debug("ElevenLabs TTS: closing context %s failed: %r", ctx.id, exc)
                ctx.conn.release(ctx)


# ------------------------------------------------------------------------------- STT
_SCRIBE_ERRORS: dict[str, type[ProviderError]] = {
    "auth_error": AuthenticationError,
    "unaccepted_terms": AuthenticationError,
    "quota_exceeded": RateLimitError,
    "rate_limited": RateLimitError,
    "queue_overflow": RateLimitError,
    "resource_exhausted": RateLimitError,
    "commit_throttled": RateLimitError,
    "insufficient_audio_activity": ProviderTimeoutError,
    "session_time_limit_exceeded": ProviderError,
    "input_error": ProviderError,
    "invalid_request": ProviderError,
    "chunk_size_exceeded": ProviderError,
    "transcriber_error": ProviderError,
    "error": ProviderError,
}
"""Scribe realtime error ``message_type`` s (Scribe closes the socket after them)."""
_SCRIBE_RETRYABLE = {"transcriber_error", "error", "session_time_limit_exceeded"}


def _scribe_error(kind: str, msg: Mapping[str, Any]) -> ProviderError:
    detail = msg.get("error") or msg.get("message") or ""
    text = f"ElevenLabs Scribe {kind}" + (f": {detail}" if detail else "")
    cls = _SCRIBE_ERRORS.get(kind, ProviderError)
    if cls is RateLimitError:
        return RateLimitError(text, provider=_PROVIDER, retryable=kind != "quota_exceeded")
    if cls is ProviderError:
        return ProviderError(text, provider=_PROVIDER, retryable=kind in _SCRIBE_RETRYABLE)
    return cls(text, provider=_PROVIDER)


def _scribe_words(items: Any) -> list[WordTiming]:
    """Scribe ``words`` (``type`` ``word`` / ``spacing`` / ``audio_event``) -> word timings."""
    if not isinstance(items, list):
        return []
    words: list[WordTiming] = []
    for item in items:
        if not isinstance(item, Mapping) or item.get("type", "word") != "word":
            continue
        text = str(item.get("text") or "").strip()
        start, end = item.get("start"), item.get("end")
        if not text or not isinstance(start, int | float) or not isinstance(end, int | float):
            continue
        logprob = item.get("logprob")
        confidence = (
            math.exp(min(0.0, float(logprob))) if isinstance(logprob, int | float) else None
        )
        words.append(WordTiming(text, float(start), float(end), confidence))
    return words


def _mean_confidence(words: Sequence[WordTiming]) -> float | None:
    values = [w.confidence for w in words if w.confidence is not None]
    return sum(values) / len(values) if values else None


@register_provider(
    "stt",
    "elevenlabs",
    description="ElevenLabs Scribe v2 Realtime: streaming STT (manual commit or server VAD)",
    default_model="scribe_v2_realtime",
    models=("scribe_v2_realtime", "scribe_v2", "scribe_v2_medical", "scribe_v1"),
    env=API_KEY_ENV,
    extra=None,
    requires=("websockets", "httpx"),
    local=False,
)
class ElevenLabsSTT(STT):
    """ElevenLabs Scribe speech-to-text.

    With ``scribe_v2_realtime`` (default) :meth:`stream` opens a Scribe v2 Realtime
    WebSocket. Audio is sent as base64 ``input_audio_chunk`` messages of about
    ``chunk_duration`` seconds. Events:

    * ``partial_transcript`` -> ``INTERIM_TRANSCRIPT`` (the segment in progress);
    * ``committed_transcript`` -> ``FINAL_TRANSCRIPT``. With ``include_timestamps`` or
      ``include_language_detection``, Scribe sends every commit twice and only the second
      copy (``committed_transcript_with_timestamps``) carries ``words`` and
      ``language_code``: that copy becomes the final instead (slightly later).

    ``commit_strategy="manual"`` (default): your VAD / turn detector owns endpointing and
    :meth:`~voice_agent_next.stt.STTStream.flush` sends ``commit`` (the cascade flushes at
    every candidate end of speech). A flush with no audio since the last answered commit
    is acknowledged at once with an empty final instead of a commit (Scribe throttles
    rapid commits). ``commit_strategy="vad"``: Scribe's own VAD commits after
    ``vad_silence_threshold_secs`` of silence; the stream then also emits
    ``START_OF_SPEECH`` (first partial) and ``END_OF_SPEECH`` (after each commit), which
    lets a cascade run without a local VAD.

    :meth:`transcribe` uses the batch ``POST /v1/speech-to-text`` API with ``batch_model``
    (Scribe v2, more accurate than the realtime model) and returns word timings. Batch
    models (``scribe_v2``, ``scribe_v2_medical``, ``scribe_v1``) are batch-only: use them
    in a cascade with a VAD (:class:`~voice_agent_next.stt.StreamAdapter`).

    Args:
        model: ``scribe_v2_realtime`` (streaming) or a batch model.
        api_key: defaults to ``$ELEVEN_API_KEY``, then ``$ELEVENLABS_API_KEY``.
        language: ISO 639-1/3 code; ``None`` lets Scribe detect the language.
        secondary_languages: other languages the speaker may switch to (realtime).
        sample_rate: rate of the PCM sent to Scribe (input is resampled to it).
        commit_strategy: ``"manual"`` (flush commits) or ``"vad"`` (server VAD).
        vad_silence_threshold_secs, vad_threshold, min_speech_duration_ms,
            min_silence_duration_ms: server VAD tuning (``commit_strategy="vad"``).
        include_timestamps: word timings on finals (realtime).
        include_language_detection: detected language on finals (realtime).
        keyterms: terms to bias recognition towards (realtime: up to 50 terms of at most
            20 characters; batch: up to 1000). Adds a surcharge.
        no_verbatim: drop filler words and false starts.
        previous_text: text context (e.g. the agent's last reply, ideally under 50
            characters) sent with the first audio chunk.
        filter_background_audio: suppress background speech and noise (realtime).
        tag_audio_events: batch only: tag ``(laughter)``... in transcripts.
        batch_model: model of :meth:`transcribe` when ``model`` is the realtime one.
        enable_logging: ``False`` requests zero-retention mode (enterprise plans).
        chunk_duration: seconds of audio per ``input_audio_chunk`` (ElevenLabs suggests
            0.1-1 s; smaller chunks lower the latency).
        keepalive_interval: after this many seconds without audio, send one chunk of
            silence so Scribe keeps the session (it closes sessions without audio
            activity); ``None`` disables it.
        base_url / region: API origin (see :class:`ElevenLabsTTS`).
        http_client: optional ``httpx.AsyncClient`` for :meth:`transcribe` (not closed by
            :meth:`aclose`).
        connect_timeout: connection / handshake timeout in seconds.
        close_timeout: how long to wait for the last commit's transcript after the input
            ends.
        request_timeout: HTTP timeout of :meth:`transcribe`.
    """

    provider = "elevenlabs"

    def __init__(
        self,
        *,
        model: str = "scribe_v2_realtime",
        api_key: str | None = None,
        language: str | None = None,
        secondary_languages: Sequence[str] = (),
        sample_rate: int = 16_000,
        commit_strategy: Literal["manual", "vad"] = "manual",
        vad_silence_threshold_secs: float | None = None,
        vad_threshold: float | None = None,
        min_speech_duration_ms: int | None = None,
        min_silence_duration_ms: int | None = None,
        include_timestamps: bool = False,
        include_language_detection: bool = False,
        keyterms: Sequence[str] = (),
        no_verbatim: bool | None = None,
        previous_text: str | None = None,
        filter_background_audio: bool | None = None,
        tag_audio_events: bool = False,
        batch_model: str = "scribe_v2",
        enable_logging: bool = True,
        chunk_duration: float = 0.1,
        keepalive_interval: float | None = 5.0,
        base_url: str | None = None,
        region: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        connect_timeout: float = 10.0,
        close_timeout: float = 5.0,
        request_timeout: float = 60.0,
    ) -> None:
        realtime = "realtime" in model
        if realtime and sample_rate not in STT_SAMPLE_RATES:
            raise ConfigurationError(
                f"Scribe realtime sample_rate must be one of {STT_SAMPLE_RATES}, got {sample_rate}"
            )
        if commit_strategy not in ("manual", "vad"):
            raise ConfigurationError(
                f"commit_strategy must be 'manual' or 'vad', got {commit_strategy!r}"
            )
        for name, value in (("keyterms", keyterms), ("secondary_languages", secondary_languages)):
            if isinstance(value, str):
                raise ConfigurationError(f"{name} must be a sequence of strings, not a string")
        _check_range("chunk_duration", chunk_duration, 0.02, 1.0)
        _check_range("vad_silence_threshold_secs", vad_silence_threshold_secs, 0.3, 3.0)
        _check_range("vad_threshold", vad_threshold, 0.0, 1.0)
        if keepalive_interval is not None and keepalive_interval <= 0:
            raise ConfigurationError(f"keepalive_interval must be > 0, got {keepalive_interval}")
        super().__init__(
            model=model,
            capabilities=STTCapabilities(
                streaming=realtime,
                interim_results=realtime,
                word_timestamps=include_timestamps or not realtime,
                end_of_turn=False,
                language_detection=(
                    (include_language_detection or include_timestamps) if realtime else True
                ),
            ),
            sample_rate=sample_rate,
            language=language,
        )
        self._api_key = _resolve_api_key(api_key)
        self.base_url = _resolve_base_url(base_url, region)
        self.secondary_languages = list(secondary_languages)
        self.commit_strategy = commit_strategy
        self.vad_options: dict[str, float | None] = {
            "vad_silence_threshold_secs": vad_silence_threshold_secs,
            "vad_threshold": vad_threshold,
            "min_speech_duration_ms": min_speech_duration_ms,
            "min_silence_duration_ms": min_silence_duration_ms,
        }
        self.include_timestamps = include_timestamps
        self.include_language_detection = include_language_detection
        self.keyterms = list(keyterms)
        self.no_verbatim = no_verbatim
        self.previous_text = previous_text
        self.filter_background_audio = filter_background_audio
        self.tag_audio_events = tag_audio_events
        self.batch_model = batch_model if realtime else model
        self.enable_logging = enable_logging
        self.chunk_duration = chunk_duration
        self.keepalive_interval = keepalive_interval
        self.connect_timeout = connect_timeout
        self.close_timeout = close_timeout
        self.request_timeout = request_timeout
        self._http = http_client
        self._owns_http = http_client is None

    def _headers(self) -> dict[str, str]:
        return {"xi-api-key": self._api_key}

    def ws_url(self, language: str | None = None) -> str:
        """The realtime WebSocket URL (query string included, no key)."""
        params: list[tuple[str, str]] = [
            ("model_id", self.model),
            ("audio_format", f"pcm_{self.sample_rate}"),
            ("commit_strategy", self.commit_strategy),
        ]
        code = _language_code(language or self.language)
        if code:
            params.append(("language_code", code))
        for other in self.secondary_languages:
            other_code = _language_code(other)
            if other_code:
                params.append(("secondary_languages", other_code))
        if self.include_timestamps:
            params.append(("include_timestamps", "true"))
        if self.include_language_detection:
            params.append(("include_language_detection", "true"))
        if self.commit_strategy == "vad":
            params += [(k, _number(v)) for k, v in self.vad_options.items() if v is not None]
        params += [("keyterms", term) for term in self.keyterms]
        if self.no_verbatim is not None:
            params.append(("no_verbatim", _bool(self.no_verbatim)))
        if self.filter_background_audio is not None:
            params.append(("filter_background_audio", _bool(self.filter_background_audio)))
        if not self.enable_logging:
            params.append(("enable_logging", "false"))
        return f"{_ws_base(self.base_url)}/v1/speech-to-text/realtime?{urlencode(params)}"

    def _create_stream(self, *, language: str | None) -> STTStream:
        return _ScribeStream(self, language=language)

    def _http_client(self) -> httpx.AsyncClient:
        import httpx

        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(self.request_timeout, connect=self.connect_timeout)
            )
            self._owns_http = True
        return self._http

    async def _recognize(self, audio: AudioFrame, *, language: str | None) -> Transcript:
        """Batch recognition: ``POST /v1/speech-to-text`` (multipart form)."""
        import httpx

        form: dict[str, str | list[str]] = {
            "model_id": self.batch_model,
            "timestamps_granularity": "word",
            "tag_audio_events": _bool(self.tag_audio_events),
        }
        code = _language_code(language or self.language)
        if code:
            form["language_code"] = code
        if self.no_verbatim is not None:
            form["no_verbatim"] = _bool(self.no_verbatim)
        if self.keyterms:
            form["keyterms"] = list(self.keyterms)
        if audio.sample_rate == 16_000 and audio.channels == 1:
            # raw 16 kHz s16le mono skips the server-side decoding (lower latency)
            form["file_format"] = "pcm_s16le_16"
            file = ("audio.pcm", audio.data, "application/octet-stream")
        else:
            file = ("audio.wav", wav_bytes(audio), "audio/wav")
        params = {} if self.enable_logging else {"enable_logging": "false"}
        try:
            response = await self._http_client().post(
                f"{self.base_url}/v1/speech-to-text",
                params=params,
                data=form,
                files={"file": file},
                headers=self._headers(),
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"ElevenLabs speech-to-text timed out: {exc!r}", provider=_PROVIDER
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderConnectionError(
                f"ElevenLabs speech-to-text failed: {exc!r}", provider=_PROVIDER
            ) from exc
        if response.status_code >= 400:
            raise _http_error(response.status_code, response.text)
        try:
            result = response.json()
        except ValueError as exc:
            raise ProviderError(
                f"ElevenLabs speech-to-text returned invalid JSON: {response.text[:200]}",
                provider=_PROVIDER,
            ) from exc
        if isinstance(result, Mapping) and isinstance(result.get("transcripts"), list):
            result = (result["transcripts"] or [{}])[0]  # multichannel shape: first channel
        if not isinstance(result, Mapping):
            raise ProviderError(
                "ElevenLabs speech-to-text returned an unexpected response", provider=_PROVIDER
            )
        words = _scribe_words(result.get("words"))
        detected = result.get("language_code")
        return Transcript(
            text=str(result.get("text") or "").strip(),
            language=detected if isinstance(detected, str) and detected else code,
            confidence=_mean_confidence(words),
            start_time=words[0].start if words else None,
            end_time=words[-1].end if words else None,
            words=words or None,
        )

    async def aclose(self) -> None:
        if self._http is not None and self._owns_http:
            http, self._http = self._http, None
            await http.aclose()


class _ScribeStream(STTStream):
    def __init__(self, stt: ElevenLabsSTT, *, language: str | None) -> None:
        self._scribe = stt
        self._segment = new_id("seg_")
        self._closing = False
        self._pending_commits = 0
        """Commits sent and not answered yet."""
        self._audio_since_commit = False
        self._first_chunk = True
        self._partial = ""
        self._speaking = False
        self._held: Transcript | None = None
        """First copy of a commit, held until its timestamped copy arrives."""
        self._settled = asyncio.Event()
        self._settled.set()
        self._server_error: ProviderError | None = None
        """The last error message (Scribe sends one right before closing the socket)."""
        super().__init__(stt, language=language)

    @property
    def _timestamped(self) -> bool:
        return self._scribe.include_timestamps or self._scribe.include_language_detection

    async def _run(self) -> None:
        stt = self._scribe
        ws = await _ws_connect(
            stt.ws_url(self._language), stt._headers(), open_timeout=stt.connect_timeout,
            what="STT",
        )  # fmt: skip
        sender = asyncio.create_task(self._send_loop(ws), name="elevenlabs-stt-send")
        receiver = asyncio.create_task(self._recv_loop(ws), name="elevenlabs-stt-recv")
        try:
            await asyncio.wait((sender, receiver), return_when=asyncio.FIRST_COMPLETED)
            await _raise_task_error(receiver, sender)
            if not sender.done():  # the server ended the session while audio was flowing
                raise self._server_error or ProviderConnectionError(
                    "ElevenLabs STT WebSocket closed unexpectedly", provider=_PROVIDER
                )
            # all audio sent and committed: collect the last transcript(s), then hang up
            self._closing = True
            settled = asyncio.ensure_future(self._settled.wait())
            try:
                await asyncio.wait((settled, receiver), timeout=stt.close_timeout,
                                   return_when=asyncio.FIRST_COMPLETED)  # fmt: skip
            finally:
                settled.cancel()
            await _raise_task_error(receiver, sender)
            if self._pending_commits:
                logger.warning(
                    "ElevenLabs STT: no transcript for the last commit %.1fs after the end "
                    "of the input",
                    stt.close_timeout,
                )
            if self._held is not None:  # its timestamped copy never came
                held, self._held = self._held, None
                self._final(held)
        finally:
            self._closing = True
            await cancel_and_wait(sender, receiver)
            await _close_ws(ws)

    # ---------------------------------------------------------------- sending
    def _chunk(self, pcm: bytes, *, commit: bool) -> str:
        msg: dict[str, Any] = {
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(pcm).decode("ascii"),
            "commit": commit,
            "sample_rate": self._scribe.sample_rate,
        }
        if self._first_chunk and self._scribe.previous_text:
            msg["previous_text"] = self._scribe.previous_text  # only allowed on the first one
        self._first_chunk = False
        return json.dumps(msg)

    async def _send_loop(self, ws: ClientConnection) -> None:
        from websockets.exceptions import ConnectionClosed

        stt = self._scribe
        chunk_bytes = max(2, round(stt.sample_rate * stt.chunk_duration) * 2)
        buf = bytearray()
        interval = stt.keepalive_interval
        try:
            while True:
                try:
                    async with asyncio.timeout(interval):
                        item = await self._input.recv()
                except TimeoutError:
                    # no audio for a while: Scribe drops sessions without audio activity
                    await ws.send(self._chunk(bytes(chunk_bytes), commit=False))
                    self._audio_since_commit = True
                    continue
                except ChanClosed:
                    break
                if self.is_flush(item):
                    await self._commit(ws, bytes(buf))
                    buf.clear()
                    continue
                assert isinstance(item, AudioFrame)
                buf += item.data
                while len(buf) >= chunk_bytes:
                    await ws.send(self._chunk(bytes(buf[:chunk_bytes]), commit=False))
                    del buf[:chunk_bytes]
                    self._audio_since_commit = True
            if buf:  # end_input() flushes first, so this only follows aclose()
                await ws.send(self._chunk(bytes(buf), commit=False))
        except ConnectionClosed as exc:
            raise self._server_error or _closed(exc, "STT") from exc

    async def _commit(self, ws: ClientConnection, tail: bytes) -> None:
        if tail or self._audio_since_commit:
            await ws.send(self._chunk(tail, commit=True))
            self._audio_since_commit = False
            self._pending_commits += 1
            self._settled.clear()
        elif not self._pending_commits and self._held is None:
            # nothing new since the last committed transcript: acknowledge the flush now
            empty = Transcript("", self._language)
            self._emit(STTEvent(STTEventType.FINAL_TRANSCRIPT, empty, self._segment))
        # else a commit is in flight: its transcript answers this flush too

    # -------------------------------------------------------------- receiving
    async def _recv_loop(self, ws: ClientConnection) -> None:
        from websockets.exceptions import ConnectionClosed

        try:
            async for raw in ws:
                msg = _parse(raw, "STT")
                if msg is not None:
                    self._on_message(msg)
        except ConnectionClosed as exc:
            if not self._closing:
                raise self._server_error or _closed(exc, "STT") from exc

    def _on_message(self, msg: dict[str, Any]) -> None:
        kind = str(msg.get("message_type") or "")
        if kind == "partial_transcript":
            self._on_partial(str(msg.get("text") or "").strip())
        elif kind == "committed_transcript":
            self._on_committed(msg, timestamped=False)
        elif kind == "committed_transcript_with_timestamps":
            self._on_committed(msg, timestamped=True)
        elif kind == "session_started":
            logger.debug("ElevenLabs Scribe session %s started", msg.get("session_id"))
        elif kind == "commit_throttled":
            # the commit was dropped and its audio stays uncommitted: no transcript will
            # answer it, so the next flush has to commit again
            logger.warning("ElevenLabs Scribe throttled a commit: %s", msg.get("error", ""))
            self._server_error = _scribe_error(kind, msg)
            self._pending_commits = max(0, self._pending_commits - 1)
            self._audio_since_commit = True
            self._check_settled()
        elif kind in _SCRIBE_ERRORS:
            self._server_error = _scribe_error(kind, msg)
            raise self._server_error
        elif kind == "warning" or (not kind and "warning" in msg):
            logger.warning("ElevenLabs Scribe warning: %s", msg.get("warning", msg))
        else:
            logger.debug("ElevenLabs Scribe: ignoring a %s message", kind or "untyped")

    def _on_partial(self, text: str) -> None:
        if not text or text == self._partial:
            return
        self._partial = text
        if self._scribe.commit_strategy == "vad" and not self._speaking:
            self._speaking = True
            self._emit(STTEvent(STTEventType.START_OF_SPEECH, segment_id=self._segment))
        transcript = Transcript(text, self._language)
        self._emit(STTEvent(STTEventType.INTERIM_TRANSCRIPT, transcript, self._segment))

    def _on_committed(self, msg: dict[str, Any], *, timestamped: bool) -> None:
        text = str(msg.get("text") or "").strip()
        if not self._timestamped:
            if not timestamped:
                self._final(Transcript(text, self._language))
            return
        if not timestamped:
            if self._held is not None:  # the previous commit's timestamped copy never came
                held, self._held = self._held, None
                self._final(held)
            self._held = Transcript(text, self._language)
            return
        self._held = None
        words = _scribe_words(msg.get("words"))
        language = msg.get("language_code")
        self._final(
            Transcript(
                text=text,
                language=language if isinstance(language, str) and language else self._language,
                confidence=_mean_confidence(words),
                start_time=words[0].start if words else None,
                end_time=words[-1].end if words else None,
                words=(words or None) if self._scribe.include_timestamps else None,
            )
        )

    def _final(self, transcript: Transcript) -> None:
        flushed = self._flush_time is not None
        self._pending_commits = max(0, self._pending_commits - 1)
        self._partial = ""
        self._emit(STTEvent(STTEventType.FINAL_TRANSCRIPT, transcript, self._segment))
        if not flushed:
            self._report_usage()  # a commit of Scribe's own (server VAD, ~36 s limit)
        if self._scribe.commit_strategy == "vad" and self._speaking:
            self._speaking = False
            end = Transcript("", transcript.language, end_time=transcript.end_time)
            self._emit(STTEvent(STTEventType.END_OF_SPEECH, end, self._segment))
        self._segment = new_id("seg_")
        self._check_settled()

    def _check_settled(self) -> None:
        if not self._pending_commits and self._held is None:
            self._settled.set()

    def _report_usage(self) -> None:
        # the base class reports usage when a flush is answered; commits Scribe makes on
        # its own are never flushed, so report their audio here (no flush latency to add)
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
