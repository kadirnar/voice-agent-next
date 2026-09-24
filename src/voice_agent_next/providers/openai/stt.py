"""OpenAI speech-to-text: realtime transcription sessions and ``/audio/transcriptions``.

``stt="openai/gpt-live-transcribe"`` streams the user's audio into a Realtime
*transcription session* (``wss://api.openai.com/v1/realtime?intent=transcription``,
``session.update`` with ``type: "transcription"``). Audio is resampled to 24 kHz PCM16 and
sent with ``input_audio_buffer.append``. Server-side turn detection is **off** by default:
the cascade's VAD decides when the user paused and :meth:`STTStream.flush` commits the
buffer (``input_audio_buffer.commit``), which makes the model finalize that audio.

Server events -> :class:`~voice_agent_next.stt.STTEvent`:

=========================================================  ==================================
``conversation.item.input_audio_transcription.delta``      ``INTERIM_TRANSCRIPT`` (text of the
                                                           item so far)
``conversation.item.input_audio_transcription.completed``  ``FINAL_TRANSCRIPT``, in commit order
``conversation.item.input_audio_transcription.failed``     ``FINAL_TRANSCRIPT`` with the partial
                                                           text (the failure is logged)
``input_audio_buffer.speech_started`` / ``speech_stopped``  ``START_OF_SPEECH`` /
                                                           ``END_OF_SPEECH`` (server VAD only)
``error``                                                  see *Errors* below
=========================================================  ==================================

Every flush is answered with at least one ``FINAL_TRANSCRIPT``, which is what the cascade's
endpointing waits for: the transcript of the committed audio, or an empty final when there
was nothing to commit (the API rejects commits of less than 100 ms of audio) and no earlier
commit is still being transcribed. Completions of different commits may arrive out of order;
finals are emitted in commit order.

With ``turn_detection="server_vad"`` / ``"semantic_vad"`` (models that support it, not
``gpt-live-transcribe``/``gpt-realtime-whisper``) the server segments the audio itself and
flushes do not commit; ``semantic_vad`` also emits ``END_OF_TURN`` after each final.

Errors: authentication failures (and an exhausted quota) are fatal; a rejected session
configuration fails the stream at start; a dropped connection or an expired session is
reconnected transparently (``max_reconnect_attempts``): the new session is configured
again, commits still waiting for their transcript are re-sent with their audio, and the
uncommitted audio (up to ``replay_buffer`` seconds) is appended again, so no utterance is
lost. Other server errors are logged.

Batch recognition (:meth:`STT.transcribe`) posts a WAV file to ``/audio/transcriptions``
(``gpt-transcribe``, ``gpt-4o-transcribe``, ``gpt-4o-mini-transcribe``, ``whisper-1``), with
``stream=true`` (server-sent events) where the model supports it; realtime-only models
(``gpt-live-transcribe``, ``gpt-realtime-whisper``) transcribe through a short realtime
session instead.

Only core dependencies are used (``websockets`` and ``httpx``; no ``openai`` SDK).
:class:`OpenAISTT` is also the client for OpenAI-compatible servers: ``speaches`` and
``localai`` (``/audio/transcriptions`` only) and ``azure_openai`` subclass it. See
``docs/providers/openai.md``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final, Literal, TypeAlias

import httpx
from websockets.asyncio.client import ClientConnection
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from ...audio.buffer import FrameChunker
from ...audio.frame import AudioFrame
from ...audio.wav import wav_bytes
from ...errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderError,
    RateLimitError,
)
from ...registry import register_provider
from ...stt import STT, STTCapabilities, STTEvent, STTEventType, STTStream, Transcript, WordTiming
from ...utils.aio import cancel_and_wait
from ...utils.clock import now
from ...utils.log import logger
from ._http import (
    OPENAI_BASE_URL,
    APIEndpoint,
    http_error,
    is_openai_host,
    iter_sse,
    new_http_client,
    transport_error,
)
from .realtime import (
    _CONNECT_ACCEPTS_PROXY,
    _MAX_MESSAGE_SIZE,
    _close_reason,
    _deep_merge,
    _handshake_error,
    _is_loopback,
    _server_error,
    realtime_url,
)

__all__ = [
    "CONTEXT_HINT_MODELS",
    "FILE_STREAMING_MODELS",
    "REALTIME_ONLY_MODELS",
    "OpenAICompatibleSTT",
    "OpenAISTT",
    "TranscriptionTurnDetection",
]

TranscriptionTurnDetection: TypeAlias = (
    Literal["server_vad", "semantic_vad"] | Mapping[str, Any] | None
)
"""``None`` (default: commits come from :meth:`STTStream.flush`), ``"server_vad"``,
``"semantic_vad"`` or a raw ``turn_detection`` object."""

REALTIME_ONLY_MODELS: Final = ("gpt-live-transcribe", "gpt-realtime-whisper")
"""Served only by realtime transcription sessions, without server-side turn detection."""
CONTEXT_HINT_MODELS: Final = ("gpt-live-transcribe", "gpt-transcribe")
"""Take ``languages`` (a list) and ``keywords`` instead of the single ``language``."""
FILE_STREAMING_MODELS: Final = ("gpt-transcribe", "gpt-4o-transcribe", "gpt-4o-mini-transcribe")
"""Support ``stream=true`` on ``/audio/transcriptions`` (``whisper-1`` does not)."""

REALTIME_SAMPLE_RATE: Final = 24_000
_MIN_COMMIT: Final = 0.1
"""The API rejects commits of less than 100 ms of audio (``input_audio_buffer_commit_empty``)."""
_DELAYS: Final = frozenset({"minimal", "low", "medium", "high", "xhigh"})
_NOISE_REDUCTION: Final = frozenset({"near_field", "far_field"})
_REGIONAL_ZH: Final = frozenset({"zh-cn", "zh-tw", "zh-hk"})
_EXPIRY_CODES: Final = frozenset({"session_expired", "max_duration"})
_COMMIT_EMPTY: Final = "input_audio_buffer_commit_empty"
_REPLAY_CHUNK: Final = 1.0
_RECONNECT_WINDOW: Final = 60.0
_MAX_TRACKED: Final = 256


def _startswith(model: str, prefixes: Iterable[str]) -> bool:
    return any(model.startswith(p) for p in prefixes)


def _language_code(language: str, *, plural: bool) -> str:
    """``en-US`` -> ``en``; the regional ``zh-cn``/``zh-tw``/``zh-hk`` codes are kept for the
    ``languages`` list, and ISO 639-3 codes (``yue``, ``cmn``) pass through."""
    code = language.strip().replace("_", "-").lower()
    if plural and code in _REGIONAL_ZH:
        return code
    return code.split("-")[0]


def _check_keyword(keyword: str) -> str:
    if not keyword.strip() or any(c in keyword for c in "<>\r\n"):
        raise ConfigurationError(
            f"invalid keyword {keyword!r}: keywords are single-line literals without '<' or '>'"
        )
    return keyword.strip()


def _confidence(logprobs: Any) -> float | None:
    """Geometric-mean token probability from ``[{"token", "logprob", ...}, ...]``."""
    if not isinstance(logprobs, list):
        return None
    values = [
        float(lp["logprob"])
        for lp in logprobs
        if isinstance(lp, Mapping) and isinstance(lp.get("logprob"), (int, float))
    ]
    if not values:
        return None
    return math.exp(sum(values) / len(values))


def _detected_language(data: Mapping[str, Any]) -> str | None:
    """``languages: [{"code": "fr"}]`` (gpt-transcribe) or ``language`` (verbose JSON)."""
    languages = data.get("languages")
    if isinstance(languages, list) and languages:
        first = languages[0]
        code = first.get("code") if isinstance(first, Mapping) else first
        if isinstance(code, str) and code:
            return code
    language = data.get("language")
    return language if isinstance(language, str) and language else None


def _words(items: Any) -> list[WordTiming] | None:
    if not isinstance(items, list) or not items:
        return None
    words: list[WordTiming] = []
    for item in items:
        if not isinstance(item, Mapping):
            return None
        start, end = item.get("start"), item.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            return None
        words.append(WordTiming(str(item.get("word") or "").strip(), float(start), float(end)))
    return words


# ------------------------------------------------------------------------------- STT
@register_provider(
    "stt",
    "openai",
    description="OpenAI realtime transcription (gpt-live-transcribe...) + /audio/transcriptions",
    default_model="gpt-live-transcribe",
    models=(
        "gpt-live-transcribe",
        "gpt-transcribe",
        "gpt-realtime-whisper",
        "gpt-4o-transcribe",
        "gpt-4o-mini-transcribe",
        "whisper-1",
    ),
    env=("OPENAI_API_KEY",),
    requires=("websockets", "httpx"),
)
class OpenAISTT(STT):
    """OpenAI speech-to-text: realtime transcription sessions and file transcription.

    Args:
        model: ``gpt-live-transcribe`` (default; streaming deltas, lowest latency),
            ``gpt-transcribe`` (transcribes each committed turn, detects the language),
            ``gpt-realtime-whisper``, ``gpt-4o-transcribe``, ``gpt-4o-mini-transcribe`` or
            ``whisper-1``.
        api_key: API key (default: ``OPENAI_API_KEY``, never sent to a non-OpenAI
            ``base_url``).
        base_url: API root including ``/v1`` (default: ``OPENAI_BASE_URL``, then
            ``https://api.openai.com/v1``); ``wss://`` is derived for the realtime session.
        language: expected language (``en``, ``en-US``...); ``Agent(language=...)`` wins.
        languages: expected languages for code-switched audio (``gpt-live-transcribe``,
            ``gpt-transcribe``).
        prompt: free-form context about the audio ("A support call about billing.").
        keywords: literal terms that may be spoken (``gpt-live-transcribe``,
            ``gpt-transcribe``).
        delay: ``minimal``/``low``/``medium``/``high``/``xhigh``: how long the streaming
            models wait for more audio before emitting text (latency vs accuracy).
        noise_reduction: ``near_field`` (headsets) or ``far_field`` (laptop/room mics).
        turn_detection: ``None`` (default: the cascade's VAD flushes, a flush commits),
            ``"server_vad"``, ``"semantic_vad"`` or a raw ``turn_detection`` object; not
            supported by the realtime-only models.
        logprobs: request token log probabilities (``gpt-4o-transcribe`` models); they
            become :attr:`Transcript.confidence`.
        realtime: stream over the realtime API (default ``True``; ``False`` makes this a
            batch STT that the cascade segments with its VAD).
        http_streaming: use ``stream=true`` on ``/audio/transcriptions`` (default: for the
            models that support it on OpenAI).
        word_timestamps: batch only: request word timings (``verbose_json``, ``whisper-1``
            and compatible servers).
        temperature: batch only: sampling temperature.
        sample_rate: batch only: rate of the uploaded WAV (the realtime API takes 24 kHz).
        headers: extra HTTP/WebSocket headers.
        query: extra query parameters of the realtime URL.
        session: extra session fields, deep-merged into ``session.update``.
        extra: extra form fields for ``/audio/transcriptions`` (lists become ``name[]``).
        chunk_ms: audio is appended in chunks of this many milliseconds.
        connect_timeout: WebSocket/HTTP connection and session setup timeout (seconds).
        timeout: HTTP read timeout (seconds).
        close_timeout: after :meth:`STTStream.end_input`, how long to wait for the last
            transcripts.
        max_reconnect_attempts: connection attempts after a transient failure (0 = never
            reconnect); also the maximum number of reconnects per minute.
        reconnect_backoff: delay before the first attempt (doubles per attempt, max 10 s).
        replay_buffer: seconds of uncommitted audio kept to re-send after a reconnect.
        http_client: an ``httpx.AsyncClient`` for the HTTP endpoint (not closed by
            :meth:`aclose`).
    """

    provider = "openai"

    DEFAULT_MODEL: ClassVar[str] = "gpt-live-transcribe"
    DEFAULT_BASE_URL: ClassVar[str] = OPENAI_BASE_URL
    BASE_URL_ENV: ClassVar[tuple[str, ...]] = ("OPENAI_BASE_URL",)
    API_KEY_ENV: ClassVar[tuple[str, ...]] = ("OPENAI_API_KEY",)
    API_KEY_REQUIRED: ClassVar[bool] = True
    GUARD_OPENAI_KEY: ClassVar[bool] = True
    """Never send the ``API_KEY_ENV`` key to an explicit ``base_url`` of another host."""
    AUTH_HEADER: ClassVar[str] = "Authorization"
    REALTIME: ClassVar[bool] = True
    """Default of ``realtime``: stream over a realtime transcription session."""
    MODEL_IN_REALTIME_URL: ClassVar[bool | None] = None
    """Send ``?model=`` on the realtime URL. ``None``: for every host but OpenAI's, whose
    realtime endpoint would open a conversation session instead (gateways and Azure route
    on it)."""
    HTTP_STREAMING: ClassVar[bool | None] = None
    """Default of ``http_streaming`` (``None``: the models that support it, on OpenAI)."""
    DEFAULT_SAMPLE_RATE: ClassVar[int] = REALTIME_SAMPLE_RATE
    PRELOAD_ON_WARMUP: ClassVar[bool] = False
    """:meth:`warmup` transcribes a short silence (servers that load models on demand)."""
    NOT_FOUND_HINT: ClassVar[str] = "check the model id and base_url"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        language: str | None = None,
        languages: Sequence[str] | None = None,
        prompt: str | None = None,
        keywords: Sequence[str] | None = None,
        delay: Literal["minimal", "low", "medium", "high", "xhigh"] | None = None,
        noise_reduction: Literal["near_field", "far_field"] | None = None,
        turn_detection: TranscriptionTurnDetection = None,
        logprobs: bool = False,
        realtime: bool | None = None,
        http_streaming: bool | None = None,
        word_timestamps: bool = False,
        temperature: float | None = None,
        sample_rate: int | None = None,
        headers: Mapping[str, str] | None = None,
        query: Mapping[str, str] | None = None,
        session: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
        chunk_ms: int = 50,
        connect_timeout: float = 10.0,
        timeout: float = 30.0,
        close_timeout: float = 5.0,
        max_reconnect_attempts: int = 3,
        reconnect_backoff: float = 0.5,
        replay_buffer: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved_model = model or self.DEFAULT_MODEL
        use_realtime = self.REALTIME if realtime is None else realtime
        realtime_only = _startswith(resolved_model, REALTIME_ONLY_MODELS)
        if realtime_only and not use_realtime:
            raise ConfigurationError(
                f"{resolved_model} is only served by realtime transcription sessions; "
                "use realtime=True or a file model (gpt-transcribe)"
            )
        td = self._turn_detection(turn_detection)
        if td is not None and realtime_only:
            raise ConfigurationError(
                f"{resolved_model} has no server-side turn detection: leave turn_detection=None "
                "and give the cascade a VAD (its end of speech commits the audio)"
            )
        if td is not None and not use_realtime:
            raise ConfigurationError("turn_detection needs realtime=True")
        rate = sample_rate or (REALTIME_SAMPLE_RATE if use_realtime else self.DEFAULT_SAMPLE_RATE)
        if use_realtime and rate != REALTIME_SAMPLE_RATE:
            raise ConfigurationError(
                f"realtime transcription takes 24 kHz audio (input is resampled), got {rate}"
            )
        if delay is not None and delay not in _DELAYS:
            raise ConfigurationError(f"delay must be one of {sorted(_DELAYS)}, got {delay!r}")
        if noise_reduction is not None and noise_reduction not in _NOISE_REDUCTION:
            raise ConfigurationError(
                f"noise_reduction must be 'near_field' or 'far_field', got {noise_reduction!r}"
            )
        if isinstance(keywords, str) or isinstance(languages, str):
            raise ConfigurationError("keywords and languages take a list of strings")
        context_hints = _startswith(resolved_model, CONTEXT_HINT_MODELS)
        if keywords and not context_hints:
            raise ConfigurationError(
                f"keywords are supported by {' and '.join(CONTEXT_HINT_MODELS)}, "
                f"not {resolved_model}"
            )
        if languages and len(languages) > 1 and not context_hints:
            raise ConfigurationError(
                f"{resolved_model} takes a single language; "
                f"only {' and '.join(CONTEXT_HINT_MODELS)} take a list"
            )
        if chunk_ms <= 0:
            raise ConfigurationError("chunk_ms must be > 0")
        semantic = td is not None and td.get("type") == "semantic_vad"
        super().__init__(
            model=resolved_model,
            capabilities=STTCapabilities(
                streaming=use_realtime,
                interim_results=use_realtime,
                word_timestamps=word_timestamps and not use_realtime,
                end_of_turn=semantic,
                language_detection=resolved_model.startswith("gpt-transcribe"),
            ),
            sample_rate=rate,
            language=language,
        )
        self.endpoint = APIEndpoint.resolve(
            type(self).__name__,
            base_url=base_url,
            api_key=api_key,
            headers=headers,
            default_base_url=self.DEFAULT_BASE_URL,
            base_url_env=self.BASE_URL_ENV,
            api_key_env=self.API_KEY_ENV,
            api_key_required=self.API_KEY_REQUIRED,
            auth_header=self.AUTH_HEADER,
            guard_openai_key=self.GUARD_OPENAI_KEY,
        )
        self.realtime = use_realtime
        self.languages = [v for v in (languages or ()) if v.strip()]
        self.prompt = prompt
        self.keywords = [_check_keyword(k) for k in keywords or ()]
        self.delay = delay
        self.noise_reduction = noise_reduction
        self.turn_detection: dict[str, Any] | None = td
        self.semantic_vad = semantic
        self.logprobs = logprobs
        self.word_timestamps = word_timestamps
        self.temperature = temperature
        self.query: dict[str, str] = dict(query or {})
        self.session_overrides: dict[str, Any] = dict(session or {})
        self.extra: dict[str, Any] = dict(extra or {})
        self.chunk_ms = chunk_ms
        self.connect_timeout = connect_timeout
        self.timeout = timeout
        self.close_timeout = close_timeout
        self.max_reconnect_attempts = max(0, max_reconnect_attempts)
        self.reconnect_backoff = reconnect_backoff
        self.replay_buffer = max(0.0, replay_buffer)
        if http_streaming is None:
            http_streaming = self.HTTP_STREAMING
        if http_streaming is None:
            http_streaming = is_openai_host(self.base_url) and _startswith(
                resolved_model, FILE_STREAMING_MODELS
            )
        self.http_streaming = http_streaming
        self._context_hints = context_hints
        self._http = http_client
        self._owns_http = http_client is None

    @staticmethod
    def _turn_detection(td: TranscriptionTurnDetection) -> dict[str, Any] | None:
        if td is None:
            return None
        cfg: dict[str, Any] = {"type": td} if isinstance(td, str) else dict(td)
        if not isinstance(cfg.get("type"), str):
            raise ConfigurationError(f"turn_detection needs a 'type': {cfg!r}")
        return cfg

    @property
    def base_url(self) -> str:
        return self.endpoint.base_url

    @property
    def http_transcription(self) -> bool:
        """Whether :meth:`transcribe` uses ``/audio/transcriptions`` (not a realtime session)."""
        return not _startswith(self.model, REALTIME_ONLY_MODELS)

    # ---------------------------------------------------------------- configuration
    def languages_for(self, language: str | None) -> list[str]:
        """Normalized language hints for a stream/request in ``language``.

        ``languages`` applies unless another language is asked for (``Agent(language=...)``
        or ``stream(language=...)``), which then wins.
        """
        language = language or self.language
        if self.languages and (language is None or language == self.language):
            values: Sequence[str] = self.languages
        elif language:
            values = [language]
        else:
            values = []
        codes = (_language_code(v, plural=self._context_hints) for v in values if v.strip())
        return list(dict.fromkeys(codes))

    def transcription_config(self, language: str | None = None) -> dict[str, Any]:
        """The ``audio.input.transcription`` object of the realtime session."""
        cfg: dict[str, Any] = {"model": self.model}
        languages = self.languages_for(language)
        if self._context_hints:
            if languages:
                cfg["languages"] = languages
            if self.keywords:
                cfg["keywords"] = list(self.keywords)
        elif languages:
            cfg["language"] = languages[0]
        if self.prompt:
            cfg["prompt"] = self.prompt
        if self.delay:
            cfg["delay"] = self.delay
        return cfg

    def session_config(self, language: str | None = None) -> dict[str, Any]:
        """The ``session`` of the ``session.update`` sent when a stream connects."""
        audio_in: dict[str, Any] = {
            "format": {"type": "audio/pcm", "rate": REALTIME_SAMPLE_RATE},
            "transcription": self.transcription_config(language),
            "turn_detection": self.turn_detection,
        }
        if self.noise_reduction:
            audio_in["noise_reduction"] = {"type": self.noise_reduction}
        session: dict[str, Any] = {"type": "transcription", "audio": {"input": audio_in}}
        if self.logprobs:
            session["include"] = ["item.input_audio_transcription.logprobs"]
        return _deep_merge(session, self.session_overrides)

    def realtime_url(self) -> str:
        """The transcription WebSocket URL (``.../realtime?intent=transcription``)."""
        with_model = self.MODEL_IN_REALTIME_URL
        if with_model is None:
            with_model = not is_openai_host(self.base_url)
        query = {"intent": "transcription", **self.query}
        return realtime_url(self.base_url, model=self.model if with_model else None, query=query)

    def transcription_form(
        self, language: str | None = None, *, stream: bool = False
    ) -> dict[str, str | list[str]]:
        """The multipart form fields (besides ``file``) posted to ``/audio/transcriptions``."""
        form: dict[str, str | list[str]] = {"model": self.model}
        languages = self.languages_for(language)
        if self._context_hints:
            if languages:
                form["languages[]"] = languages
            if self.keywords:
                form["keywords[]"] = list(self.keywords)
        elif languages:
            form["language"] = languages[0]
        if self.prompt:
            form["prompt"] = self.prompt
        if self.word_timestamps:
            form["response_format"] = "verbose_json"
            form["timestamp_granularities[]"] = ["word"]
        else:
            form["response_format"] = "json"
        if self.temperature is not None:
            form["temperature"] = str(self.temperature)
        if self.logprobs:
            form["include[]"] = ["logprobs"]
        if stream:
            form["stream"] = "true"
        for key, value in self.extra.items():
            if value is None:
                form.pop(key, None)
            elif isinstance(value, (list, tuple)):
                name = key if key.endswith("[]") else f"{key}[]"
                form[name] = [_form_value(v) for v in value]
            else:
                form[key] = _form_value(value)
        return form

    # -------------------------------------------------------------------- streaming
    def _create_stream(self, *, language: str | None) -> STTStream:
        return _TranscriptionStream(self, language=language)

    async def _open_socket(self) -> ClientConnection:
        url = self.realtime_url()
        kwargs: dict[str, Any] = {}
        if _CONNECT_ACCEPTS_PROXY and _is_loopback(url):
            kwargs["proxy"] = None  # never route local servers through a system proxy
        try:
            return await ws_connect(
                url,
                additional_headers=self.endpoint.request_headers(),
                open_timeout=self.connect_timeout,
                max_size=_MAX_MESSAGE_SIZE,
                compression=None,  # base64 audio does not compress; avoid the CPU cost
                close_timeout=2.0,
                **kwargs,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise _handshake_error(exc, self.provider, url) from exc

    # ------------------------------------------------------------------------ batch
    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = new_http_client(
                self.base_url,
                timeout=self.timeout,
                connect_timeout=self.connect_timeout,
                keepalive_expiry=120.0,
            )
            self._owns_http = True
        return self._http

    async def _recognize(self, audio: AudioFrame, *, language: str | None) -> Transcript:
        if not self.http_transcription:  # realtime-only model: a short realtime session
            return await super()._recognize(audio, language=language)
        url = self.endpoint.url("audio/transcriptions")
        stream = self.http_streaming
        form = self.transcription_form(language, stream=stream)
        files = {"file": ("audio.wav", wav_bytes(audio), "audio/wav")}
        headers = self.endpoint.request_headers()
        hint = self.NOT_FOUND_HINT.format(model=self.model)
        try:
            async with self._client().stream(
                "POST", url, data=form, files=files, headers=headers
            ) as response:
                if response.status_code >= 400:
                    raw = await response.aread()
                    raise http_error(self.provider, response.status_code, raw, hint=hint)
                content_type = response.headers.get("content-type", "")
                if "text/event-stream" in content_type:
                    result = await _collect_sse(response)
                else:
                    raw = await response.aread()
                    result = _parse_transcription(raw)
        except httpx.HTTPError as exc:
            raise transport_error(self.provider, exc, url) from exc
        languages = self.languages_for(language)
        return Transcript(
            text=str(result.get("text") or "").strip(),
            language=_detected_language(result) or (languages[0] if languages else None),
            confidence=_confidence(result.get("logprobs")),
            words=_words(result.get("words")),
        )

    # --------------------------------------------------------------------- lifecycle
    async def warmup(self) -> None:
        """Batch mode: open the HTTP connection (``GET /models``), or load the model on
        servers that load models on demand (``PRELOAD_ON_WARMUP``). Failures are logged.
        Streams connect when they start."""
        if self.realtime:
            return
        try:
            if self.PRELOAD_ON_WARMUP:
                silence = AudioFrame.silence(0.5, self.sample_rate)
                await self._recognize(silence, language=self.language)
                return
            url = self.endpoint.url("models")
            response = await self._client().get(url, headers=self.endpoint.request_headers())
            if response.status_code in (401, 403):
                raise http_error(self.provider, response.status_code, response.content)
        except Exception as exc:
            logger.warning("%s STT warmup failed: %s", self.provider, exc)

    async def aclose(self) -> None:
        if self._http is not None and self._owns_http:
            http, self._http = self._http, None
            await http.aclose()


class OpenAICompatibleSTT(OpenAISTT):
    """Base class for preconfigured OpenAI-compatible transcription servers.

    Subclasses set ``provider`` and the class attributes of :class:`OpenAISTT`, and register
    themselves with ``@register_provider("stt", "<name>", ...)``. The API key is always read
    from ``API_KEY_ENV`` and is optional unless ``API_KEY_REQUIRED``. By default the server
    is used through ``/audio/transcriptions`` (``REALTIME = False``): the cascade segments the
    audio with its VAD.
    """

    provider = "openai_compatible"
    BASE_URL_ENV: ClassVar[tuple[str, ...]] = ()
    API_KEY_ENV: ClassVar[tuple[str, ...]] = ()
    API_KEY_REQUIRED: ClassVar[bool] = False
    GUARD_OPENAI_KEY: ClassVar[bool] = False
    REALTIME: ClassVar[bool] = False
    HTTP_STREAMING: ClassVar[bool | None] = False
    DEFAULT_SAMPLE_RATE: ClassVar[int] = 16_000


def _form_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def _parse_transcription(raw: bytes) -> dict[str, Any]:
    """A ``/audio/transcriptions`` response body (JSON, or plain text from some servers)."""
    text = raw.decode("utf-8", "replace")
    try:
        data = json.loads(text)
    except ValueError:
        return {"text": text}
    return data if isinstance(data, dict) else {"text": str(data)}


async def _collect_sse(response: httpx.Response) -> dict[str, Any]:
    """Join ``transcript.text.delta`` events; ``transcript.text.done`` has the final text."""
    deltas: list[str] = []
    logprobs: list[Any] = []
    async for event in iter_sse(response):
        kind = event.get("type")
        if kind == "transcript.text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                deltas.append(delta)
            if isinstance(event.get("logprobs"), list):
                logprobs.extend(event["logprobs"])
        elif kind == "transcript.text.done":
            return event
        elif kind == "error" or "error" in event:
            err = event.get("error", event)
            raise ProviderError(f"transcription stream error: {err}")
    return {"text": "".join(deltas), "logprobs": logprobs or None}


# ------------------------------------------------------------------------------ stream
@dataclass(eq=False)
class _Segment:
    """Audio between two commits, and what the server said about it."""

    start: float
    """Stream position (seconds) of the first sample."""
    end: float | None = None
    duration: float = 0.0
    chunks: deque[bytes] = field(default_factory=deque)
    """The audio, kept to re-send it after a reconnect (bounded by ``replay_buffer``)."""
    kept: int = 0
    manual: bool = True
    """Committed by a flush (``False``: by the server VAD)."""
    event_id: str | None = None
    item_id: str | None = None
    partial: str = ""
    transcript: str | None = None
    language: str | None = None
    logprobs: Any = None
    done: bool = False

    def add(self, data: bytes, limit: int, rate: int) -> None:
        self.chunks.append(data)
        self.kept += len(data)
        self.duration += len(data) / (2 * rate)
        while self.chunks and self.kept - len(self.chunks[0]) >= limit:
            self.kept -= len(self.chunks.popleft())


class _SessionExpired(Exception):
    """The provider session ended (``session_expired``): reconnect."""


class _TranscriptionStream(STTStream):
    """One realtime transcription session, reconnected after transient failures."""

    def __init__(self, stt: OpenAISTT, *, language: str | None) -> None:
        self._oa = stt
        self._chunker = FrameChunker(stt.sample_rate, frame_duration=stt.chunk_ms / 1000.0)
        self._keep_bytes = round(stt.replay_buffer * stt.sample_rate) * 2
        self._ws: ClientConnection | None = None
        self._open = _Segment(start=0.0)
        self._pending: deque[_Segment] = deque()
        self._partials: dict[str, str] = {}
        self._delta_logprobs: dict[str, list[Any]] = {}
        self._finished: deque[str] = deque(maxlen=64)
        self._position = 0.0
        self._origin = 0.0
        """Stream position of the server's audio time 0 (server VAD timestamps)."""
        self._in_speech = False
        """Server VAD: between ``speech_started`` and ``speech_stopped``."""
        self._awaiting_commit = False
        """Server VAD: speech stopped, its ``committed`` event has not arrived yet."""
        self._speech_end: float | None = None
        self._flush_requested = False
        self._needs_replay = False
        self._drained = asyncio.Event()
        self._drained.set()
        self._ready = asyncio.Event()
        self._startup_error: Exception | None = None
        self._update_id: str | None = None
        self._event_seq = 0
        self._reconnects: deque[float] = deque()
        self.session_id: str | None = None
        super().__init__(stt, language=language)
        hints = stt.languages_for(language)
        self._hint: str | None = hints[0] if hints else None

    # ----------------------------------------------------------------- connection
    async def _run(self) -> None:
        self._ws = await self._oa._open_socket()
        try:
            while True:
                reason = await self._serve()
                if reason is None:
                    return  # input ended and every transcript was delivered
                await self._reconnect(reason)
        finally:
            await self._close_ws()

    async def _serve(self) -> str | None:
        """Run the current connection until the input is done (``None``) or it is lost."""
        ws = self._ws
        assert ws is not None
        reader = asyncio.create_task(self._read(ws), name="openai-stt-recv")
        writer = asyncio.create_task(self._write(ws), name="openai-stt-send")
        try:
            done, _ = await asyncio.wait({reader, writer}, return_when=asyncio.FIRST_COMPLETED)
            if writer in done:
                lost = writer.result()
                if lost is None:
                    return None
                # the send failed: give the reader a moment to report why (an error event)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(reader), 0.5)
                if reader.done():
                    return reader.result() or lost
                return lost
            reason = reader.result()
            return reason or "closed by the server"
        finally:
            await cancel_and_wait(reader, writer)

    async def _reconnect(self, reason: str) -> None:
        stt, provider = self._oa, self._oa.provider
        await self._close_ws()
        t = now()
        while self._reconnects and self._reconnects[0] < t - _RECONNECT_WINDOW:
            self._reconnects.popleft()
        attempts = stt.max_reconnect_attempts
        if attempts <= 0 or len(self._reconnects) >= attempts:
            raise ProviderConnectionError(
                f"{provider}: transcription connection {reason}", provider=provider
            )
        logger.warning("%s: transcription connection %s; reconnecting", provider, reason)
        last: ProviderError | None = None
        for attempt in range(attempts):
            await asyncio.sleep(min(stt.reconnect_backoff * 2**attempt, 10.0))
            try:
                self._ws = await stt._open_socket()
            except (AuthenticationError, ConfigurationError):
                raise
            except ProviderError as exc:
                last = exc
                logger.warning("%s: reconnect attempt %d failed: %s", provider, attempt + 1, exc)
                continue
            self._reconnects.append(now())
            self._needs_replay = True
            return
        raise last or ProviderConnectionError(
            f"{provider}: transcription connection {reason}", provider=provider
        )

    async def _close_ws(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()

    def _next_event_id(self) -> str:
        self._event_seq += 1
        return f"evt_stt_{self._event_seq:06d}"

    @staticmethod
    async def _send(ws: ClientConnection, event: dict[str, Any]) -> None:
        await ws.send(json.dumps(event, separators=(",", ":"), ensure_ascii=False))

    # --------------------------------------------------------------------- sending
    async def _write(self, ws: ClientConnection) -> str | None:
        """Configure the session, then forward audio and flushes. ``None`` when done."""
        try:
            await self._configure(ws)
            if self._needs_replay:
                await self._replay(ws)
            if self._flush_requested:  # a flush interrupted by the reconnect
                await self._flush(ws)
            async for item in self._input:
                if self.is_flush(item):
                    self._flush_requested = True
                    await self._flush(ws)
                else:
                    assert isinstance(item, AudioFrame)
                    await self._append(ws, [c.data for c in self._chunker.push(item)])
            await self._finish()
            return None
        except ConnectionClosed as exc:
            return f"lost while sending ({_close_reason(exc)})"

    async def _configure(self, ws: ClientConnection) -> None:
        stt = self._oa
        self._ready.clear()
        self._startup_error = None
        self._update_id = self._next_event_id()
        session = stt.session_config(self._language)
        await self._send(ws, {"event_id": self._update_id, "type": "session.update",
                              "session": session})  # fmt: skip
        try:
            await asyncio.wait_for(self._ready.wait(), stt.connect_timeout)
        except TimeoutError:
            logger.warning(
                "%s: no session.updated within %.0fs; continuing", stt.provider, stt.connect_timeout
            )
        if self._startup_error is not None:
            raise self._startup_error

    async def _append(self, ws: ClientConnection, chunks: list[bytes]) -> None:
        rate = self._oa.sample_rate
        for data in chunks:  # recorded first: whatever a lost connection drops is re-sent
            self._open.add(data, self._keep_bytes, rate)
            self._position += len(data) / (2 * rate)
        for data in chunks:
            await self._send_audio(ws, data)

    async def _send_audio(self, ws: ClientConnection, data: bytes) -> None:
        payload = base64.b64encode(data).decode("ascii")
        await self._send(ws, {"type": "input_audio_buffer.append", "audio": payload})

    async def _flush(self, ws: ClientConnection) -> None:
        """Commit the buffered audio (or answer the flush when there is nothing to commit).

        With server turn detection the server commits by itself; only the last flush
        (:meth:`end_input`) commits speech that is still in progress.
        """
        await self._append(ws, [c.data for c in self._chunker.flush()])
        final = self._input.closed and self._input.empty()  # end_input(): the last flush
        duration = self._open.duration
        if self._oa.turn_detection is None:
            commit = duration >= _MIN_COMMIT or (final and duration > 0)
            busy = bool(self._pending)
        else:
            commit = final and self._in_speech and duration > 0
            busy = bool(self._pending) or self._in_speech or self._awaiting_commit
        if commit:
            if duration < _MIN_COMMIT:  # the last few ms: pad to what the API accepts
                pad = bytes(round((_MIN_COMMIT - duration) * self._oa.sample_rate) * 2)
                await self._append(ws, [pad])
            await self._commit(ws)
        elif not busy:
            self._ack()  # nothing to finalize: answer the flush right away
        # otherwise the final of a pending commit answers the flush
        self._flush_requested = False

    async def _commit(self, ws: ClientConnection) -> None:
        seg = self._open
        seg.end = self._position
        seg.event_id = self._next_event_id()
        self._open = _Segment(start=self._position)
        self._pending.append(seg)
        self._drained.clear()
        await self._send(ws, {"event_id": seg.event_id, "type": "input_audio_buffer.commit"})

    async def _replay(self, ws: ClientConnection) -> None:
        """After a reconnect: re-commit what the old session did not transcribe, then re-send
        the uncommitted audio. The server clock restarts with the replayed audio."""
        self._needs_replay = False
        self._in_speech = self._awaiting_commit = False
        pending, self._pending = list(self._pending), deque()
        resend = [s for s in pending if s.manual and s.kept and not s.done]
        replayed = sum(s.kept for s in [*resend, self._open]) / (2 * self._oa.sample_rate)
        self._origin = self._position - replayed
        for seg in pending:
            if seg.item_id is not None:
                seg.partial = self._partials.pop(seg.item_id, seg.partial)
            if seg not in resend:
                seg.done = True  # keep its place; its partial text is the best we have
                self._pending.append(seg)
                continue
            await self._resend(ws, seg)
            seg.item_id = None
            seg.event_id = self._next_event_id()
            self._pending.append(seg)
            await self._send(ws, {"event_id": seg.event_id, "type": "input_audio_buffer.commit"})
        self._drain()
        await self._resend(ws, self._open)

    async def _resend(self, ws: ClientConnection, seg: _Segment) -> None:
        data = b"".join(seg.chunks)
        step = round(_REPLAY_CHUNK * self._oa.sample_rate) * 2
        for i in range(0, len(data), step):
            await self._send_audio(ws, data[i : i + step])

    async def _finish(self) -> None:
        """After :meth:`end_input`: wait for the last transcripts."""
        try:
            await asyncio.wait_for(self._drained.wait(), self._oa.close_timeout)
        except TimeoutError:
            logger.warning(
                "%s: no transcript for %d commit(s) within %.1fs after the end of input",
                self._oa.provider,
                len(self._pending),
                self._oa.close_timeout,
            )
            for seg in self._pending:
                seg.done = True
            self._drain()

    # ------------------------------------------------------------------- receiving
    async def _read(self, ws: ClientConnection) -> str | None:
        """Dispatch server events; returns why the connection ended."""
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                try:
                    event = json.loads(raw)
                except ValueError:
                    logger.warning("%s: ignoring a non-JSON message", self._oa.provider)
                    continue
                if isinstance(event, dict):
                    self._dispatch(event)
        except ConnectionClosed as exc:
            return _close_reason(exc)
        except _SessionExpired:
            return "session expired"
        return "closed by the server"

    def _dispatch(self, event: dict[str, Any]) -> None:
        handler = _HANDLERS.get(str(event.get("type") or ""))
        if handler is None:
            return
        try:
            handler(self, event)
        except (ProviderError, _SessionExpired):
            raise
        except Exception:
            logger.exception("%s: failed to handle %s", self._oa.provider, event.get("type"))

    def _on_session_created(self, ev: dict[str, Any]) -> None:
        session = ev.get("session")
        if isinstance(session, Mapping):
            self.session_id = str(session.get("id") or "") or None

    def _on_session_updated(self, ev: dict[str, Any]) -> None:
        self._ready.set()

    def _on_error(self, ev: dict[str, Any]) -> None:
        stt, provider = self._oa, self._oa.provider
        err = ev.get("error")
        if not isinstance(err, Mapping):
            err = {"message": str(err or ev)}
        code, etype = err.get("code"), err.get("type")
        event_id = err.get("event_id")
        if event_id and event_id == self._update_id and not self._ready.is_set():
            self._startup_error = _server_error(err, provider, "session.update")
            self._ready.set()
            return
        seg = next((s for s in self._pending if event_id and s.event_id == event_id), None)
        if seg is not None:
            self._commit_rejected(seg, err)
            return
        if code in _EXPIRY_CODES or etype in _EXPIRY_CODES:
            logger.info("%s: transcription session expired; reconnecting", provider)
            raise _SessionExpired
        exc = _server_error(err, provider, None)
        if isinstance(exc, AuthenticationError) or (
            isinstance(exc, RateLimitError) and not exc.retryable
        ):
            raise exc
        if code == _COMMIT_EMPTY:  # a commit without event id (server VAD mode)
            logger.debug("%s: %s", provider, exc)
            return
        logger.warning("%s: %s", stt.provider, exc)

    def _commit_rejected(self, seg: _Segment, err: Mapping[str, Any]) -> None:
        """The server refused a commit: its audio is still in the server's buffer."""
        if seg.item_id is not None:
            # the server VAD committed the same audio just before: ours got its item
            seg.event_id = None
            return
        self._pending.remove(seg)
        self._open.chunks.extendleft(reversed(seg.chunks))
        self._open.kept += seg.kept
        self._open.duration += seg.duration
        self._open.start = seg.start
        message = str(err.get("message") or "")
        if err.get("code") == _COMMIT_EMPTY or "too small" in message or "empty" in message:
            logger.debug("%s: commit rejected: %s", self._oa.provider, message)
        else:
            logger.warning("%s: commit rejected: %s", self._oa.provider, message)
        if self._pending:
            self._drain()
        else:
            self._drained.set()
            self._ack()

    def _on_committed(self, ev: dict[str, Any]) -> None:
        item_id = str(ev.get("item_id") or "")
        if not item_id or self._find(item_id) is not None or item_id in self._finished:
            return
        seg = next((s for s in self._pending if s.manual and s.item_id is None), None)
        if seg is None:  # committed by the server VAD
            seg = self._server_segment()
        seg.item_id = item_id
        self._awaiting_commit = False

    def _server_segment(self) -> _Segment:
        seg, self._open = self._open, _Segment(start=self._position)
        seg.manual = False
        seg.end = self._speech_end if self._speech_end is not None else self._position
        self._speech_end = None
        self._pending.append(seg)
        self._drained.clear()
        return seg

    def _on_speech_started(self, ev: dict[str, Any]) -> None:
        item_id = str(ev.get("item_id") or "")
        self._in_speech = True
        ms = ev.get("audio_start_ms")
        start = self._origin + ms / 1000.0 if isinstance(ms, (int, float)) else None
        transcript = Transcript("", self._hint, start_time=start)
        self._emit(STTEvent(STTEventType.START_OF_SPEECH, transcript, item_id or None))

    def _on_speech_stopped(self, ev: dict[str, Any]) -> None:
        item_id = str(ev.get("item_id") or "")
        self._in_speech = False
        self._awaiting_commit = True  # until the server's commit of this speech arrives
        ms = ev.get("audio_end_ms")
        end = self._origin + ms / 1000.0 if isinstance(ms, (int, float)) else self._position
        self._speech_end = end
        transcript = Transcript("", self._hint, end_time=end)
        self._emit(STTEvent(STTEventType.END_OF_SPEECH, transcript, item_id or None))

    def _on_delta(self, ev: dict[str, Any]) -> None:
        item_id = str(ev.get("item_id") or "")
        delta = ev.get("delta")
        if item_id in self._finished or not isinstance(delta, str):
            return
        if isinstance(ev.get("logprobs"), list):
            self._delta_logprobs.setdefault(item_id, []).extend(ev["logprobs"])
            _bound(self._delta_logprobs)
        if not delta:
            return
        text = self._partials.get(item_id, "") + delta
        self._partials[item_id] = text
        _bound(self._partials)
        if text.strip():
            transcript = Transcript(text.strip(), self._hint)
            self._emit(STTEvent(STTEventType.INTERIM_TRANSCRIPT, transcript, item_id or None))

    def _on_completed(self, ev: dict[str, Any]) -> None:
        item_id = str(ev.get("item_id") or "")
        if item_id in self._finished:
            return
        seg = self._find(item_id) or self._claim(item_id)
        seg.transcript = str(ev.get("transcript") or "")
        seg.language = _detected_language(ev)
        seg.logprobs = ev.get("logprobs") or self._delta_logprobs.pop(item_id, None)
        seg.done = True
        self._drain()

    def _on_failed(self, ev: dict[str, Any]) -> None:
        item_id = str(ev.get("item_id") or "")
        if item_id in self._finished:
            return
        err = ev.get("error")
        message = err.get("message") if isinstance(err, Mapping) else err
        logger.warning(
            "%s: transcription of %s failed: %s", self._oa.provider, item_id, message or err
        )
        seg = self._find(item_id) or self._claim(item_id)
        seg.done = True  # the final carries the partial text
        self._drain()

    def _find(self, item_id: str) -> _Segment | None:
        return next((s for s in self._pending if item_id and s.item_id == item_id), None)

    def _claim(self, item_id: str) -> _Segment:
        """The segment a result without a prior ``committed`` event belongs to."""
        seg = next((s for s in self._pending if s.manual and s.item_id is None), None)
        if seg is None:
            seg = self._server_segment()
        seg.item_id = item_id
        return seg

    # -------------------------------------------------------------------- emitting
    def _drain(self) -> None:
        """Emit the finals that are ready, in commit order."""
        while self._pending and self._pending[0].done:
            self._final(self._pending.popleft())
        if not self._pending:
            self._drained.set()

    def _final(self, seg: _Segment) -> None:
        item_id = seg.item_id
        partial = self._partials.pop(item_id, "") if item_id else ""
        self._delta_logprobs.pop(item_id or "", None)
        if item_id:
            self._finished.append(item_id)
        text = seg.transcript if seg.transcript is not None else (partial or seg.partial)
        transcript = Transcript(
            text.strip(),
            language=seg.language or self._hint,
            confidence=_confidence(seg.logprobs),
            start_time=seg.start,
            end_time=seg.end,
        )
        self._emit(STTEvent(STTEventType.FINAL_TRANSCRIPT, transcript, item_id))
        if not seg.manual and self._oa.semantic_vad:
            self._emit(STTEvent(STTEventType.END_OF_TURN, transcript, item_id))

    def _ack(self) -> None:
        self._emit(STTEvent(STTEventType.FINAL_TRANSCRIPT, Transcript("", self._hint)))


def _bound(items: dict[str, Any], limit: int = _MAX_TRACKED) -> None:
    while len(items) > limit:
        del items[next(iter(items))]


_HANDLERS: Final[Mapping[str, Any]] = {
    "session.created": _TranscriptionStream._on_session_created,
    "transcription_session.created": _TranscriptionStream._on_session_created,
    "session.updated": _TranscriptionStream._on_session_updated,
    "transcription_session.updated": _TranscriptionStream._on_session_updated,
    "error": _TranscriptionStream._on_error,
    "input_audio_buffer.committed": _TranscriptionStream._on_committed,
    "input_audio_buffer.speech_started": _TranscriptionStream._on_speech_started,
    "input_audio_buffer.speech_stopped": _TranscriptionStream._on_speech_stopped,
    "conversation.item.input_audio_transcription.delta": _TranscriptionStream._on_delta,
    "conversation.item.input_audio_transcription.completed": _TranscriptionStream._on_completed,
    "conversation.item.input_audio_transcription.failed": _TranscriptionStream._on_failed,
}
