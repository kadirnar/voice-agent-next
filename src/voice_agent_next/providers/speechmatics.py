"""Speechmatics: Agent STT (Linden) and Realtime STT over a raw WebSocket.

Registered component: ``("stt", "speechmatics")`` — :class:`SpeechmaticsSTT`
(``stt="speechmatics/linden-1"``, the default model; ``"speechmatics/enhanced"`` or
``"speechmatics/standard"`` for the Realtime API).

Two APIs, one protocol family (https://docs.speechmatics.com/rt-api-ref and
https://docs.speechmatics.com/api-ref/agent-stt-websocket):

* **Agent STT** (``linden-1``): ``wss://{region}.rt.speechmatics.com/v2/agent``. The
  service works in segments: ``AddPartialSegment`` / ``AddSegment`` (punctuated text,
  speaker, ``metadata.start_time`` / ``end_time``), ``StartOfTurn`` / ``EndOfTurn`` and
  ``SpeechStarted`` / ``SpeechEnded``. ``turn_config.turn_detection_mode`` is ``vad`` (the
  service ends turns) or ``external`` (the client ends each turn with
  ``ForceEndOfUtterance``). 16 kHz PCM only.
* **Realtime** (``enhanced`` / ``standard``): ``wss://{region}.rt.speechmatics.com/v2``.
  Words stream in ``AddPartialTranscript`` / ``AddTranscript`` (``max_delay`` bounds the
  latency of finals); ``conversation_config.end_of_utterance_silence_trigger`` makes the
  server send ``EndOfUtterance`` after that much silence, and ``ForceEndOfUtterance``
  forces one (``forced: true``).

Both: ``StartRecognition`` (answered by ``RecognitionStarted``), binary ``AddAudio``
frames (acknowledged by ``AudioAdded``), ``EndOfStream`` (``last_seq_no``) answered by the
last results and ``EndOfTranscript``; ``Error`` / ``Warning`` / ``Info`` messages. The API
key goes in an ``Authorization: Bearer`` header; a temporary key goes in ``?jwt=``.

Only core dependencies are used (``websockets`` and ``httpx``, no SDK).
See ``docs/providers/speechmatics.md``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any, Literal
from urllib.parse import urlencode

import httpx
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus, InvalidURI

from ..audio.frame import AudioFrame
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
from ..utils.aio import ChanClosed, cancel_and_wait
from ..utils.ids import new_id
from ..utils.log import logger

__all__ = ["SpeechmaticsSTT", "SpeechmaticsStream"]

PROVIDER = "speechmatics"
API_KEY_ENV = "SPEECHMATICS_API_KEY"
DEFAULT_MODEL = "linden-1"
AGENT_MODELS = ("linden-1",)
REALTIME_MODELS = ("enhanced", "standard")
MODELS = AGENT_MODELS + REALTIME_MODELS
REGION_URLS = {
    "global": "wss://global.rt.speechmatics.com",
    "eu": "wss://eu.rt.speechmatics.com",
    "us": "wss://us.rt.speechmatics.com",
    "au": "wss://au.rt.speechmatics.com",
}
DEFAULT_MANAGEMENT_URL = "https://mp.speechmatics.com"
DEFAULT_EOU_SILENCE = 0.5  # Speechmatics recommends 0.5-0.8 s for voice agents

_NORMAL_CLOSE_CODES = frozenset({1000, 1001, 1005})
_MAX_VOCAB = 20_000


# ---------------------------------------------------------------------------- helpers
def _check_range(name: str, value: float | None, low: float, high: float) -> None:
    if value is not None and not low <= value <= high:
        raise ConfigurationError(f"{name} must be within [{low}, {high}], got {value}")


def _language_code(language: str | None) -> str:
    """``en-US`` -> ``en``; Speechmatics codes such as ``cmn_en`` pass through."""
    if not language:
        return "en"
    return language.split("-")[0].lower() or "en"


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
        logger.warning("Speechmatics: ignoring a non-JSON message: %.200s", message)
        return {}
    return data if isinstance(data, dict) else {}


def _metadata_time(message: Mapping[str, Any], key: str) -> float | None:
    metadata = message.get("metadata")
    return _float(metadata.get(key)) if isinstance(metadata, dict) else None


def _vocab_entry(entry: str | Mapping[str, Any]) -> str | dict[str, Any]:
    if isinstance(entry, str):
        return entry
    if not isinstance(entry.get("content"), str) or not entry["content"]:
        raise ConfigurationError(f"additional_vocab entries need a 'content' string: {entry!r}")
    return dict(entry)


def _alternative(result: Mapping[str, Any]) -> dict[str, Any]:
    alternatives = result.get("alternatives")
    if isinstance(alternatives, list) and alternatives and isinstance(alternatives[0], dict):
        return alternatives[0]
    return {}


def _render(results: Sequence[Mapping[str, Any]], delimiter: str) -> str:
    """Join Realtime results: punctuation attaches per its ``attaches_to``."""
    out = ""
    glue_next = False
    for result in results:
        content = str(_alternative(result).get("content") or "")
        if not content:
            continue
        punctuation = result.get("type") == "punctuation"
        attaches = result.get("attaches_to")
        if not out or glue_next or (punctuation and attaches in ("previous", "both")):
            out += content
        else:
            out += delimiter + content
        glue_next = punctuation and attaches in ("next", "both")
    return out.strip()


def _words(results: Sequence[Mapping[str, Any]]) -> list[WordTiming] | None:
    """Word timings (seconds from the stream start); punctuation joins the previous word."""
    words: list[WordTiming] = []
    for result in results:
        alt = _alternative(result)
        content = str(alt.get("content") or "")
        start, end = _float(result.get("start_time")), _float(result.get("end_time"))
        if not content or start is None or end is None:
            continue
        if result.get("type") == "punctuation":
            if words and result.get("attaches_to") in ("previous", "both", None):
                words[-1].word += content
            continue
        words.append(WordTiming(content, start, end, _float(alt.get("confidence"))))
    return words or None


def _mean_confidence(words: Sequence[WordTiming] | None) -> float | None:
    values = [w.confidence for w in words or () if w.confidence is not None]
    return sum(values) / len(values) if values else None


_AUTH_ERRORS = frozenset({"not_authorised", "not_allowed"})
_CLOSE_TYPES = {
    4001: "not_authorised",
    4003: "not_allowed",
    4004: "invalid_model",
    4005: "quota_exceeded",
    4006: "timelimit_exceeded",
    4013: "job_error",
    1011: "internal_error",
    1003: "protocol_error",
    1008: "policy_violation",
}


def _type_error(kind: str, reason: str, code: int | None = None) -> ProviderError:
    """Map a Speechmatics ``Error`` type (or close code) to a library error.

    ``retryable`` tells whether opening a new session may succeed.
    """
    detail = f"Speechmatics error {kind}" if kind else "Speechmatics error"
    message = f"{detail}: {reason}" if reason else detail
    if kind in _AUTH_ERRORS:
        return AuthenticationError(message, provider=PROVIDER, status_code=code)
    if kind == "quota_exceeded":  # concurrent session limit: retry in 5-10 s
        return RateLimitError(message, provider=PROVIDER, status_code=code)
    if kind == "start_recognition_timeout":
        return ProviderTimeoutError(message, provider=PROVIDER, status_code=code)
    if kind in ("idle_timeout", "session_timeout", "policy_violation"):
        return ProviderConnectionError(message, provider=PROVIDER, status_code=code)
    if kind in ("job_error", "internal_error", "unknown_error"):
        return ProviderError(message, provider=PROVIDER, status_code=code, retryable=True)
    # invalid_* (message, model, language, config, audio_type, output_format),
    # protocol_error, timelimit_exceeded (usage quota exhausted), data_error, buffer_error
    return ProviderError(message, provider=PROVIDER, status_code=code)


def _http_error(status: int, detail: str) -> ProviderError:
    message = f"Speechmatics returned HTTP {status}" + (f": {detail}" if detail else "")
    if status in (401, 403):
        return AuthenticationError(message, provider=PROVIDER, status_code=status)
    if status == 429:
        return RateLimitError(message, provider=PROVIDER, status_code=status)
    if status in (408, 504):
        return ProviderTimeoutError(message, provider=PROVIDER, status_code=status)
    return ProviderError(message, provider=PROVIDER, status_code=status, retryable=status >= 500)


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
    parts = [str(data[k]) for k in ("type", "reason", "detail", "message", "error") if data.get(k)]
    return ": ".join(parts) or text.strip()[:500]


# -------------------------------------------------------------------------------- STT
@register_provider(
    "stt",
    "speechmatics",
    description="Speechmatics Agent STT (Linden) / Realtime STT with end of turn",
    default_model=DEFAULT_MODEL,
    models=MODELS,
    env=(API_KEY_ENV,),
    requires=("websockets", "httpx"),
    local=False,
)
class SpeechmaticsSTT(STT):
    """Speechmatics streaming speech-to-text: Agent STT (``linden-1``) or Realtime.

    Events (:class:`~voice_agent_next.stt.STTEventType`):

    * **Agent STT**: ``StartOfTurn`` or the turn's first segment -> ``START_OF_SPEECH``;
      ``AddPartialSegment`` -> ``INTERIM_TRANSCRIPT`` (the segment being built);
      ``AddSegment`` -> ``FINAL_TRANSCRIPT`` (a turn may have several);
      ``EndOfTurn`` -> ``END_OF_SPEECH`` + ``END_OF_TURN`` (``end_of_turn=True``).
    * **Realtime**: the first words of an utterance -> ``START_OF_SPEECH``; partials and
      finals -> ``INTERIM_TRANSCRIPT`` (the utterance so far); ``EndOfUtterance`` ->
      ``FINAL_TRANSCRIPT`` (the whole utterance, with word timings) + ``END_OF_SPEECH`` +
      ``END_OF_TURN`` (``end_of_turn=True``, not for ``forced`` utterances).

    Turns ended by our own :meth:`~voice_agent_next.stt.STTStream.flush` never get
    ``END_OF_TURN``: whoever flushed owns that decision.

    Two ways to run it in a cascade:

    * ``end_of_turn=True`` (default): Speechmatics ends the user's turn (Agent STT
      ``turn_detection_mode="vad"``; Realtime ``EndOfUtterance`` after
      ``end_of_utterance_silence_trigger`` seconds of silence) and the cascade commits it.
    * ``end_of_turn=False``: the cascade's VAD / turn detector ends turns; its flush sends
      ``ForceEndOfUtterance`` (Agent STT runs with ``turn_detection_mode="external"``).
      A flush while no turn is open is acknowledged with an empty final after
      ``force_end_grace`` seconds, unless a turn starts meanwhile.

    Args:
        model: ``linden-1`` (Agent STT, default), ``enhanced`` or ``standard`` (Realtime).
        api_key: Speechmatics API key (default: ``SPEECHMATICS_API_KEY``).
        jwt: a temporary key (:meth:`create_temporary_key`) used instead of the API key.
        language: language code (default ``en``; ``en-US`` -> ``en``).
        end_of_turn: emit provider ``END_OF_TURN`` events (see above).
        enable_partials: interim results (``AddPartialSegment`` / ``AddPartialTranscript``).
        additional_vocab: custom dictionary: words, or ``{"content": ...,
            "sounds_like": [...]}`` entries (at most 20,000).
        domain: domain language pack (``"finance"``, ``"medical"``...).
        output_locale: output spelling locale (``"en-US"``, ``"en-GB"``...).
        diarization: label speakers (``diarization="speaker"``).
        speaker_diarization_config: ``max_speakers``, ``speaker_sensitivity``...
        transcript_filtering_config: e.g. ``{"remove_disfluencies": True}``.
        emit_sentences: Agent STT: one segment per sentence.
        max_delay: Realtime: latency bound of finals (0.7-4 s; default 4 on the server).
        max_delay_mode: Realtime: ``flexible`` (default) or ``fixed``.
        end_of_utterance_silence_trigger: Realtime: silence (0-2 s, default 0.5 here)
            before ``EndOfUtterance``; ``0`` disables it.
        punctuation_overrides, audio_filtering_config, enable_entities: Realtime, passed
            through.
        sample_rate: rate of the PCM sent (Agent STT requires 16 kHz).
        force_end_grace: see ``end_of_turn=False`` above (``0`` acknowledges at once).
        chunk_ms: audio chunk size sent upstream.
        region: ``global`` (default, nearest region), ``eu``, ``us`` or ``au``.
        base_url: WebSocket origin override (``wss://...``; also for tests).
        management_url: origin of the temporary-key API.
        http_client: optional ``httpx.AsyncClient`` (not closed by :meth:`aclose`).
        connect_timeout: connection timeout in seconds.
        start_timeout: wait for ``RecognitionStarted`` (a large vocabulary takes longer).
        close_timeout: wait for ``EndOfTranscript`` after the input ends.
        extra_config: more ``transcription_config`` fields, sent verbatim.
    """

    provider = PROVIDER

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        jwt: str | None = None,
        language: str | None = None,
        end_of_turn: bool = True,
        enable_partials: bool = True,
        additional_vocab: Sequence[str | Mapping[str, Any]] = (),
        domain: str | None = None,
        output_locale: str | None = None,
        diarization: bool = False,
        speaker_diarization_config: Mapping[str, Any] | None = None,
        transcript_filtering_config: Mapping[str, Any] | None = None,
        emit_sentences: bool | None = None,
        max_delay: float | None = None,
        max_delay_mode: Literal["flexible", "fixed"] | None = None,
        end_of_utterance_silence_trigger: float | None = DEFAULT_EOU_SILENCE,
        punctuation_overrides: Mapping[str, Any] | None = None,
        audio_filtering_config: Mapping[str, Any] | None = None,
        enable_entities: bool | None = None,
        sample_rate: int = 16_000,
        force_end_grace: float = 0.3,
        chunk_ms: int = 40,
        region: Literal["global", "eu", "us", "au"] | None = None,
        base_url: str | None = None,
        management_url: str = DEFAULT_MANAGEMENT_URL,
        http_client: httpx.AsyncClient | None = None,
        connect_timeout: float = 10.0,
        start_timeout: float = 20.0,
        close_timeout: float = 5.0,
        extra_config: Mapping[str, Any] | None = None,
    ) -> None:
        model = model or DEFAULT_MODEL
        agent = model in AGENT_MODELS or model.startswith("linden")
        if isinstance(additional_vocab, str):
            raise ConfigurationError("additional_vocab must be a sequence, not a string")
        if len(additional_vocab) > _MAX_VOCAB:
            raise ConfigurationError(f"at most {_MAX_VOCAB} additional_vocab entries")
        vocab = [_vocab_entry(e) for e in additional_vocab]
        _check_range("max_delay", max_delay, 0.7, 4.0)
        _check_range("end_of_utterance_silence_trigger", end_of_utterance_silence_trigger, 0, 2)
        _check_range("chunk_ms", chunk_ms, 10, 1000)
        if max_delay_mode is not None and max_delay_mode not in ("flexible", "fixed"):
            raise ConfigurationError(f"max_delay_mode must be flexible or fixed: {max_delay_mode}")
        if force_end_grace < 0:
            raise ConfigurationError("force_end_grace must be >= 0")
        if region is not None and region not in REGION_URLS:
            raise ConfigurationError(f"region must be one of {sorted(REGION_URLS)}, got {region!r}")
        if agent:
            realtime_only = {
                "max_delay": max_delay,
                "max_delay_mode": max_delay_mode,
                "punctuation_overrides": punctuation_overrides,
                "audio_filtering_config": audio_filtering_config,
                "enable_entities": enable_entities,
            }
            wrong = [k for k, v in realtime_only.items() if v is not None]
            if wrong:
                raise ConfigurationError(f"{', '.join(wrong)}: Realtime models only, not {model}")
            if sample_rate != 16_000:
                raise ConfigurationError("Speechmatics Agent STT takes 16 kHz audio only")
        elif emit_sentences is not None:
            raise ConfigurationError(f"emit_sentences: Agent STT (linden) only, not {model}")
        elif not 8_000 <= sample_rate <= 48_000:
            raise ConfigurationError(f"sample_rate must be 8000-48000 Hz, got {sample_rate}")
        super().__init__(
            model=model,
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=enable_partials,
                word_timestamps=not agent,
                end_of_turn=end_of_turn,
            ),
            sample_rate=sample_rate,
            language=language,
        )
        self.jwt = jwt
        self._api_key = (api_key or os.environ.get(API_KEY_ENV) or "").strip()
        if not self._api_key and not jwt:
            raise ConfigurationError(
                f"Speechmatics needs an API key: pass api_key=... (or jwt=...) or set {API_KEY_ENV}"
            )
        self.is_agent = agent
        self.enable_partials = enable_partials
        self.additional_vocab = vocab
        self.domain = domain
        self.output_locale = output_locale
        self.diarization = diarization
        self.speaker_diarization_config = dict(speaker_diarization_config or {})
        self.transcript_filtering_config = dict(transcript_filtering_config or {})
        self.emit_sentences = emit_sentences
        self.max_delay = max_delay
        self.max_delay_mode = max_delay_mode
        self.end_of_utterance_silence_trigger = end_of_utterance_silence_trigger
        self.punctuation_overrides = dict(punctuation_overrides or {})
        self.audio_filtering_config = dict(audio_filtering_config or {})
        self.enable_entities = enable_entities
        self.force_end_grace = force_end_grace
        self.chunk_ms = chunk_ms
        self.base_url = (base_url or REGION_URLS[region or "global"]).rstrip("/")
        self.management_url = management_url.rstrip("/")
        self.connect_timeout = connect_timeout
        self.start_timeout = start_timeout
        self.close_timeout = close_timeout
        self.extra_config = dict(extra_config or {})
        self._http = http_client
        self._owns_http = http_client is None

    # ---------------------------------------------------------------- requests
    @property
    def url(self) -> str:
        """The WebSocket URL (no credentials)."""
        return f"{self.base_url}/v2/agent" if self.is_agent else f"{self.base_url}/v2"

    def transcription_config(self, language: str | None = None) -> dict[str, Any]:
        """``StartRecognition.transcription_config`` (``None`` values are not sent)."""
        config: dict[str, Any] = {
            "language": _language_code(language or self.language),
            "model": self.model,
            "enable_partials": self.enable_partials,
            "domain": self.domain,
            "output_locale": self.output_locale,
            "additional_vocab": self.additional_vocab or None,
            "diarization": "speaker" if self.diarization else None,
            "speaker_diarization_config": self.speaker_diarization_config or None,
            "transcript_filtering_config": self.transcript_filtering_config or None,
        }
        if self.is_agent:
            config["emit_sentences"] = self.emit_sentences
        else:
            eou = self.end_of_utterance_silence_trigger
            config |= {
                "max_delay": self.max_delay,
                "max_delay_mode": self.max_delay_mode,
                "punctuation_overrides": self.punctuation_overrides or None,
                "audio_filtering_config": self.audio_filtering_config or None,
                "enable_entities": self.enable_entities,
                "conversation_config": (
                    {"end_of_utterance_silence_trigger": eou} if eou is not None else None
                ),
            }
        config |= self.extra_config
        return {k: v for k, v in config.items() if v is not None}

    def start_message(self, language: str | None = None) -> dict[str, Any]:
        """The ``StartRecognition`` message."""
        message: dict[str, Any] = {
            "message": "StartRecognition",
            "audio_format": {
                "type": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self.sample_rate,
            },
            "transcription_config": self.transcription_config(language),
        }
        if self.is_agent:
            mode = "vad" if self.capabilities.end_of_turn else "external"
            message["turn_config"] = {"turn_detection_mode": mode}
        return message

    @property
    def can_force_end(self) -> bool:
        """Whether a flush sends ``ForceEndOfUtterance`` (Agent STT: external turns only)."""
        return not self.is_agent or not self.capabilities.end_of_turn

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            timeout = httpx.Timeout(30.0, connect=self.connect_timeout)
            self._http = httpx.AsyncClient(timeout=timeout)
            self._owns_http = True
        return self._http

    async def create_temporary_key(self, *, ttl: int = 60, client_ref: str | None = None) -> str:
        """A temporary Realtime key (``POST /v1/api_keys?type=rt``) for a client that must
        not see your API key: ``SpeechmaticsSTT(jwt=key)``. ``ttl`` is 60-86400 s."""
        _check_range("ttl", ttl, 60, 86_400)
        if not self._api_key:
            raise ConfigurationError(f"creating a temporary key needs an API key ({API_KEY_ENV})")
        body: dict[str, Any] = {"ttl": ttl}
        if client_ref:
            body["client_ref"] = client_ref
        try:
            response = await self._client().post(
                f"{self.management_url}/v1/api_keys",
                params={"type": "rt"},
                json=body,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Speechmatics request timed out: {exc!r}", provider=PROVIDER
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderConnectionError(
                f"Speechmatics request failed: {exc!r}", provider=PROVIDER
            ) from exc
        if response.status_code >= 400:
            raise _http_error(response.status_code, _error_detail(response.content))
        try:
            key = response.json().get("key_value")
        except (ValueError, AttributeError):
            key = None
        if not isinstance(key, str) or not key:
            raise ProviderError("Speechmatics returned no temporary key", provider=PROVIDER)
        return key

    def _create_stream(self, *, language: str | None) -> STTStream:
        return SpeechmaticsStream(self, language=language)

    def stream(self, *, language: str | None = None) -> SpeechmaticsStream:
        """Open a streaming session (see :class:`SpeechmaticsStream`)."""
        stream = super().stream(language=language)
        assert isinstance(stream, SpeechmaticsStream)
        return stream

    async def aclose(self) -> None:
        if self._http is not None and self._owns_http:
            http, self._http = self._http, None
            await http.aclose()


async def _connect(stt: SpeechmaticsSTT) -> ClientConnection:
    url, headers = stt.url, {}
    if stt.jwt:
        url = f"{url}?{urlencode({'jwt': stt.jwt})}"
    else:
        headers = {"Authorization": f"Bearer {stt._api_key}"}
    try:
        return await connect(
            url,
            additional_headers=headers,
            open_timeout=stt.connect_timeout,
            close_timeout=2.0,
            compression=None,  # PCM audio does not compress; save the CPU
        )
    except InvalidStatus as exc:
        response = exc.response
        raise _http_error(response.status_code, _error_detail(response.body)) from exc
    except InvalidURI as exc:
        raise ConfigurationError(f"invalid Speechmatics URL: {exc}") from exc
    except TimeoutError as exc:
        raise ProviderTimeoutError(
            "timed out connecting to the Speechmatics real-time API", provider=PROVIDER
        ) from exc
    except (OSError, InvalidHandshake) as exc:
        raise ProviderConnectionError(
            f"could not connect to the Speechmatics real-time API: {exc}", provider=PROVIDER
        ) from exc


class SpeechmaticsStream(STTStream):
    """One Speechmatics session (see :class:`SpeechmaticsSTT` for the events).

    Set once the session started: :attr:`session_id` and :attr:`language_pack_info`.
    :attr:`speaker` is the speaker label of the last Agent STT segment (with diarization).
    """

    def __init__(self, stt: SpeechmaticsSTT, *, language: str | None) -> None:
        self._sm = stt
        self._chunk_bytes = round(stt.sample_rate * stt.chunk_ms / 1000) * 2
        self._buf = bytearray()
        self._bytes_sent = 0
        self._seq_no = 0
        self._ws: ClientConnection | None = None
        self._started = asyncio.Event()
        self._closing = False
        self._done = False
        self._server_error: ProviderError | None = None
        self._segment_id = new_id("seg_")
        self._turn_active = False
        self._turn_flushed = False  # a flush arrived while the open turn was active
        self._turn_start: float | None = None
        self._turn_finals: list[Transcript] = []  # Agent STT: segments of the open turn
        self._results: list[dict[str, Any]] = []  # Realtime: final results of the utterance
        self._last_text = ""
        self._pending_flushes = 0
        self._grace: asyncio.Task[None] | None = None
        self._delimiter = " "
        self.session_id: str | None = None
        self.language_pack_info: dict[str, Any] = {}
        self.speaker: str | None = None
        super().__init__(stt, language=language)

    # ------------------------------------------------------------------ public API
    async def set_recognition_config(self, **transcription_config: Any) -> None:
        """Send ``SetRecognitionConfig`` (Realtime models): change ``max_delay``,
        ``enable_partials``, ``conversation_config``... mid-session (not the language)."""
        if self._sm.is_agent:
            raise ConfigurationError("SetRecognitionConfig is not part of the Agent STT API")
        if not self._started.is_set():
            started = asyncio.ensure_future(self._started.wait())
            try:
                await asyncio.wait({started, self._task}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                started.cancel()
        ws = self._ws
        if ws is None or self._closing or self._task.done():
            raise RuntimeError("the Speechmatics stream is not open")
        config = {"language": _language_code(self._language or self._sm.language)}
        await ws.send(
            _json(
                {
                    "message": "SetRecognitionConfig",
                    "transcription_config": config | transcription_config,
                }
            )
        )

    # ------------------------------------------------------------------- plumbing
    async def _run(self) -> None:
        stt = self._sm
        ws = await _connect(stt)
        self._ws = ws
        receiver: asyncio.Task[None] | None = None
        sender: asyncio.Task[None] | None = None
        try:
            try:
                await ws.send(_json(stt.start_message(self._language)))
            except ConnectionClosed as exc:
                raise self._close_error(exc) from exc
            receiver = asyncio.create_task(self._recv_loop(ws), name="speechmatics-stt-recv")
            started = asyncio.ensure_future(self._started.wait())
            try:
                await asyncio.wait(
                    {started, receiver},
                    timeout=stt.start_timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                started.cancel()
            self._raise_task_error(receiver)
            if not self._started.is_set():
                raise (
                    ProviderConnectionError(
                        "Speechmatics closed the session before it started", provider=PROVIDER
                    )
                    if receiver.done()
                    else ProviderTimeoutError(
                        f"no RecognitionStarted from Speechmatics in {stt.start_timeout}s",
                        provider=PROVIDER,
                    )
                )
            sender = asyncio.create_task(self._send_loop(ws), name="speechmatics-stt-send")
            await asyncio.wait({receiver, sender}, return_when=asyncio.FIRST_COMPLETED)
            self._raise_task_error(receiver, sender)
            if not sender.done():
                raise self._server_error or ProviderConnectionError(
                    "Speechmatics closed the session unexpectedly", provider=PROVIDER
                )
            # EndOfStream sent: the last results come, then EndOfTranscript
            await asyncio.wait({receiver}, timeout=stt.close_timeout)
            self._raise_task_error(receiver, sender)
            if not receiver.done():
                logger.warning("Speechmatics: no EndOfTranscript after %.1fs", stt.close_timeout)
            self._finish_session()
        finally:
            self._closing = True
            if self._grace is not None:
                await cancel_and_wait(self._grace)
            await cancel_and_wait(*(t for t in (sender, receiver) if t is not None))
            with contextlib.suppress(Exception):
                await ws.close()

    def _raise_task_error(self, *tasks: asyncio.Task[None]) -> None:
        for task in tasks:
            if task.done() and not task.cancelled():
                exc = task.exception()
                if exc is not None:
                    raise exc

    async def _send_chunk(self, ws: ClientConnection, data: bytes) -> None:
        await ws.send(data)
        self._seq_no += 1
        self._bytes_sent += len(data)

    async def _send_loop(self, ws: ClientConnection) -> None:
        try:
            while True:
                try:
                    item = await self._input.recv()
                except ChanClosed:
                    break
                if self.is_flush(item):
                    if self._buf:
                        await self._send_chunk(ws, bytes(self._buf))
                        self._buf.clear()
                    if self._sm.can_force_end:
                        timestamp = round(self._bytes_sent / (2 * self._sm.sample_rate), 3)
                        await ws.send(
                            _json({"message": "ForceEndOfUtterance", "timestamp": timestamp})
                        )
                    self._on_flush()
                    continue
                assert isinstance(item, AudioFrame)
                self._buf += item.data
                while len(self._buf) >= self._chunk_bytes:
                    await self._send_chunk(ws, bytes(self._buf[: self._chunk_bytes]))
                    del self._buf[: self._chunk_bytes]
            if self._buf:
                await self._send_chunk(ws, bytes(self._buf))
                self._buf.clear()
            self._closing = True
            await ws.send(_json({"message": "EndOfStream", "last_seq_no": self._seq_no}))
        except ConnectionClosed as exc:
            raise self._server_error or self._close_error(exc) from exc

    async def _recv_loop(self, ws: ClientConnection) -> None:
        try:
            async for message in ws:
                if isinstance(message, str):
                    self._on_message(_parse(message))
                    if self._done:
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
                "Speechmatics closed the session unexpectedly", provider=PROVIDER
            )

    @staticmethod
    def _close_error(exc: ConnectionClosed) -> ProviderError:
        frame = exc.rcvd
        if frame is None:
            error: ProviderError = ProviderConnectionError(
                "Speechmatics connection lost (no close frame)", provider=PROVIDER
            )
        else:
            code = int(frame.code)
            error = _type_error(_CLOSE_TYPES.get(code, ""), frame.reason, code)
            if code not in _CLOSE_TYPES:
                error = ProviderConnectionError(str(error), provider=PROVIDER, status_code=code)
        error.__cause__ = exc
        return error

    # -------------------------------------------------------------------- events
    def _on_message(self, message: dict[str, Any]) -> None:
        kind = message.get("message")
        if kind == "AddPartialSegment":
            self._on_segment(message, final=False)
        elif kind == "AddSegment":
            self._on_segment(message, final=True)
        elif kind == "AddPartialTranscript":
            self._on_transcript(message, final=False)
        elif kind == "AddTranscript":
            self._on_transcript(message, final=True)
        elif kind == "StartOfTurn":
            self._begin_turn(_metadata_time(message, "start_time"))
        elif kind == "EndOfTurn":
            self._end_agent_turn(_metadata_time(message, "end_time"))
        elif kind == "EndOfUtterance":
            self._end_utterance(forced=bool(message.get("forced")))
        elif kind == "RecognitionStarted":
            self.session_id = message.get("id")
            info = message.get("language_pack_info")
            self.language_pack_info = dict(info) if isinstance(info, dict) else {}
            delimiter = self.language_pack_info.get("word_delimiter")
            if isinstance(delimiter, str):
                self._delimiter = delimiter
            self._started.set()
            logger.debug("Speechmatics session %s started", self.session_id)
        elif kind == "EndOfTranscript":
            self._done = True
        elif kind == "Error":
            self._server_error = _type_error(
                str(message.get("type") or ""), str(message.get("reason") or "")
            )
            raise self._server_error
        elif kind == "Warning":
            logger.warning(
                "Speechmatics warning %s: %s", message.get("type"), message.get("reason")
            )
        else:  # AudioAdded, SpeechStarted, SpeechEnded, Info, translations...
            logger.debug("Speechmatics: ignoring %s message", kind)

    # Agent STT --------------------------------------------------------------------
    def _on_segment(self, message: dict[str, Any], *, final: bool) -> None:
        segment = message.get("segment")
        segment = segment if isinstance(segment, dict) else {}
        text = str(segment.get("transcript") or "").strip()
        start = _metadata_time(message, "start_time")
        transcript = Transcript(
            text=text,
            language=_language_code(self._language or self._sm.language),
            start_time=start,
            end_time=_metadata_time(message, "end_time"),
        )
        if not text:
            return
        speaker = segment.get("speaker")
        if final and isinstance(speaker, str):
            self.speaker = speaker
        self._begin_turn(start)
        if final:
            self._turn_finals.append(transcript)
            self._last_text = ""
            self._final(transcript)
        elif text != self._last_text:
            self._last_text = text
            self._event(STTEventType.INTERIM_TRANSCRIPT, transcript)

    def _end_agent_turn(self, end_time: float | None) -> None:
        finals, self._turn_finals = self._turn_finals, []
        was_active = self._turn_active
        flushed = self._pending_flushes > 0 or self._turn_flushed
        if self._pending_flushes:
            self._final(Transcript("", _language_code(self._language or self._sm.language)))
        if not was_active:
            return
        texts = [t.text for t in finals if t.text]
        transcript = Transcript(
            text=self._delimiter.join(texts),
            language=_language_code(self._language or self._sm.language),
            start_time=self._turn_start,
            end_time=finals[-1].end_time if finals and finals[-1].end_time else end_time,
        )
        self._event(STTEventType.END_OF_SPEECH, transcript)
        if self._sm.capabilities.end_of_turn and texts and not flushed:
            self._event(STTEventType.END_OF_TURN, transcript)
        self._close_turn(report=not flushed)

    # Realtime ---------------------------------------------------------------------
    def _results_of(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        results = message.get("results")
        return [r for r in results if isinstance(r, dict)] if isinstance(results, list) else []

    def _rt_transcript(self, results: Sequence[Mapping[str, Any]]) -> Transcript:
        words = _words(results)
        languages = [_alternative(r).get("language") for r in results]
        language = next((lang for lang in languages if isinstance(lang, str) and lang), None)
        return Transcript(
            text=_render(results, self._delimiter),
            language=language or _language_code(self._language or self._sm.language),
            confidence=_mean_confidence(words),
            start_time=words[0].start if words else None,
            end_time=words[-1].end if words else None,
            words=words,
        )

    def _on_transcript(self, message: dict[str, Any], *, final: bool) -> None:
        results = self._results_of(message)
        if final:
            self._results.extend(results)
            shown = self._results
        else:
            shown = [*self._results, *results]
        transcript = self._rt_transcript(shown)
        if not transcript.text or transcript.text == self._last_text:
            return
        self._begin_turn(transcript.start_time)
        self._last_text = transcript.text
        self._event(STTEventType.INTERIM_TRANSCRIPT, transcript)

    def _end_utterance(self, *, forced: bool) -> None:
        results, self._results = self._results, []
        transcript = self._rt_transcript(results)
        flushed = self._pending_flushes > 0 or self._turn_flushed
        if not transcript.text and not self._turn_active and not flushed:
            return
        if transcript.text:
            self._begin_turn(transcript.start_time)
        was_active = self._turn_active
        self._final(transcript)
        if was_active:
            self._event(STTEventType.END_OF_SPEECH, transcript)
        eot = self._sm.capabilities.end_of_turn and transcript.text and not (forced or flushed)
        if eot:
            self._event(STTEventType.END_OF_TURN, transcript)
        self._close_turn(report=not (forced or flushed))

    # shared -----------------------------------------------------------------------
    def _begin_turn(self, start_time: float | None) -> None:
        if self._grace is not None:  # a turn is open: its end answers the flush
            self._grace.cancel()
            self._grace = None
        if not self._turn_active:
            self._turn_active = True
            self._turn_start = start_time
            self._last_text = ""
            self._segment_id = new_id("seg_")
            self._event(STTEventType.START_OF_SPEECH, Transcript("", start_time=start_time))

    def _close_turn(self, *, report: bool) -> None:
        self._turn_active = False
        self._turn_flushed = False
        self._turn_start = None
        self._last_text = ""
        if report:
            self._report_usage()

    def _final(self, transcript: Transcript) -> None:
        """Emit a final; any final after a flush answers it."""
        self._pending_flushes = 0
        if self._grace is not None:
            self._grace.cancel()
            self._grace = None
        self._event(STTEventType.FINAL_TRANSCRIPT, transcript)

    def _on_flush(self) -> None:
        self._pending_flushes += 1
        if self._turn_active:
            self._turn_flushed = True
        if self._turn_active or self._grace is not None:
            return  # the open turn's final answers it
        if self._sm.force_end_grace <= 0:
            self._ack_flushes()
        else:
            self._grace = asyncio.create_task(self._grace_timer(), name="speechmatics-flush-ack")

    async def _grace_timer(self) -> None:
        await asyncio.sleep(self._sm.force_end_grace)
        self._grace = None
        if not self._turn_active:
            self._ack_flushes()

    def _ack_flushes(self) -> None:
        """No turn is open: nothing to finalize, acknowledge pending flushes with ``""``."""
        if self._pending_flushes:
            self._final(Transcript("", _language_code(self._language or self._sm.language)))

    def _finish_session(self) -> None:
        # A turn still open when the session ended: close it with what we have.
        if self._sm.is_agent:
            if self._turn_active:
                self._end_agent_turn(None)
        elif self._results or self._turn_active:
            self._end_utterance(forced=True)
        self._ack_flushes()

    def _event(self, kind: STTEventType, transcript: Transcript | None = None) -> None:
        self._emit(STTEvent(kind, transcript, self._segment_id))

    def _report_usage(self) -> None:
        # The base class reports usage when a flush is answered; turns Speechmatics ends
        # on its own are never flushed, so report their audio here.
        if self._audio_duration > 0:
            self._stt.emit(
                "metrics",
                STTMetrics(
                    provider=self._stt.provider,
                    model=self._stt.model,
                    request_id=self.session_id or self._request_id,
                    audio_duration=self._audio_duration,
                    streamed=True,
                ),
            )
            self._audio_duration = 0.0
