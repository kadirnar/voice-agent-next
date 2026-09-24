"""AssemblyAI: Universal-3.5 Pro / Universal-Streaming v3 STT over a raw WebSocket.

Registered component: ``("stt", "assemblyai")`` — :class:`AssemblyAISTT`
(``stt="assemblyai/universal-3-5-pro"``, the default model).

* **Streaming** (:meth:`STT.stream`): ``wss://streaming.assemblyai.com/v3/ws`` (or the
  US / EU data-zone hosts). Binary ``pcm_s16le`` chunks of 50 ms go up; ``Begin``,
  ``SpeechStarted``, ``Turn`` (partials, then ``end_of_turn: true``) and ``Termination``
  come back. The model's neural end-of-turn detection ends each turn; with
  ``end_of_turn=True`` (default) the stream emits ``END_OF_TURN`` and owns the user's
  turn in a cascade. :meth:`~voice_agent_next.stt.STTStream.flush` sends
  ``ForceEndpoint`` either way, so a VAD / turn detector can end turns instead
  (``end_of_turn=False``). :meth:`AssemblyAIStream.update_configuration` sends
  ``UpdateConfiguration`` (turn silences, keyterms, prompt...) mid-session.
* **Batch** (:meth:`STT.transcribe`): one ``POST https://sync.assemblyai.com/v1/transcribe``
  (the Sync STT API, audio of 80 ms to 120 s); longer audio is streamed instead.

Only core dependencies are used (``websockets`` and ``httpx``, no SDK). Requests carry the
API key in the ``Authorization`` header (no ``Bearer`` prefix); a temporary streaming
token (:meth:`AssemblyAISTT.create_temporary_token`) can be passed as ``token=`` instead.
See ``docs/providers/assemblyai.md``.
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

__all__ = ["AssemblyAISTT", "AssemblyAIStream"]

PROVIDER = "assemblyai"
API_KEY_ENV = "ASSEMBLYAI_API_KEY"
DEFAULT_MODEL = "universal-3-5-pro"
MODELS = ("universal-3-5-pro", "universal-streaming-english", "universal-streaming-multilingual")
DEFAULT_STREAMING_URL = "wss://streaming.assemblyai.com"
REGION_URLS = {
    "us": "wss://streaming.us.assemblyai.com",
    "eu": "wss://streaming.eu.assemblyai.com",
}
DEFAULT_SYNC_URL = "https://sync.assemblyai.com"
SYNC_MIN_DURATION = 0.08
SYNC_MAX_DURATION = 120.0

_MSG_FORCE_ENDPOINT = json.dumps({"type": "ForceEndpoint"})
_MSG_TERMINATE = json.dumps({"type": "Terminate"})
_MSG_KEEPALIVE = json.dumps({"type": "KeepAlive"})
_MIN_CHUNK = 0.05  # AssemblyAI rejects chunks shorter than 50 ms (error 3007)
_MAX_CHUNK = 1.0
_MAX_KEYTERMS = 100
_MAX_KEYTERM_CHARS = 50
_MAX_PROMPT_CHARS = 1750
_MODES = ("min_latency", "balanced", "max_accuracy")
_NORMAL_CLOSE_CODES = frozenset({1000, 1001, 1005})
_UPDATE_FIELDS = frozenset(
    {
        "prompt",
        "keyterms_prompt",
        "min_turn_silence",
        "max_turn_silence",
        "continuous_partials",
        "vad_threshold",
        "interruption_delay",
        "agent_context",
        "mode",
        "end_of_turn_confidence_threshold",
        "language_codes",
        "session_heartbeat",
    }
)


# ---------------------------------------------------------------------------- helpers
def _is_pro(model: str) -> bool:
    """Universal-3 Pro family (``universal-3-5-pro``...) vs Universal-Streaming."""
    return model.startswith("universal-3")


def _check_range(name: str, value: float | None, low: float, high: float) -> None:
    if value is not None and not low <= value <= high:
        raise ConfigurationError(f"{name} must be within [{low}, {high}], got {value}")


def _language_code(language: str | None) -> str | None:
    """``en-US`` -> ``en`` (AssemblyAI takes ISO 639-1 codes)."""
    if not language:
        return None
    return language.replace("_", "-").split("-")[0].lower() or None


def _float(value: Any) -> float | None:
    try:
        return None if value is None or isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        return None


def _words(items: Any) -> list[WordTiming] | None:
    """Word timings (milliseconds from the stream start -> seconds); ``None`` if absent."""
    if not isinstance(items, list) or not items:
        return None
    words: list[WordTiming] = []
    for item in items:
        if not isinstance(item, dict):
            return None
        start, end = _float(item.get("start")), _float(item.get("end"))
        if start is None or end is None:
            return None
        words.append(
            WordTiming(
                str(item.get("text") or ""),
                start / 1000.0,
                end / 1000.0,
                _float(item.get("confidence")),
            )
        )
    return words


def _mean_confidence(words: Sequence[WordTiming] | None) -> float | None:
    values = [w.confidence for w in words or () if w.confidence is not None]
    return sum(values) / len(values) if values else None


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _query_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return _json(list(value))
    return str(value)


def _parse(message: str) -> dict[str, Any]:
    try:
        data = json.loads(message)
    except ValueError:
        logger.warning("AssemblyAI: ignoring a non-JSON message: %.200s", message)
        return {}
    return data if isinstance(data, dict) else {}


def _code_error(code: int | None, reason: str) -> ProviderError:
    """Map an AssemblyAI ``Error`` / close code to a library error.

    ``retryable`` tells whether opening a new session may succeed (reconnect-safe).
    """
    detail = f"AssemblyAI streaming error {code}" if code is not None else "AssemblyAI error"
    message = f"{detail}: {reason}" if reason else detail
    lowered = reason.lower()
    if code == 3009 or "too many concurrent" in lowered or "rate limit" in lowered:
        return RateLimitError(message, provider=PROVIDER, status_code=code)
    if code == 1008:
        return AuthenticationError(message, provider=PROVIDER, status_code=code)
    if code == 3008 or (code == 3006 and "inactivity" in lowered):
        # session expired / idle: a new session works
        return ProviderConnectionError(message, provider=PROVIDER, status_code=code)
    if code in (3006, 3007, 410):  # invalid message, bad chunk size / audio rate, v2 endpoint
        return ProviderError(message, provider=PROVIDER, status_code=code)
    if code == 3005:  # unknown server error
        return ProviderError(message, provider=PROVIDER, status_code=code, retryable=True)
    return ProviderConnectionError(message, provider=PROVIDER, status_code=code)


def _http_error(status: int, detail: str) -> ProviderError:
    message = f"AssemblyAI returned HTTP {status}" + (f": {detail}" if detail else "")
    if status in (401, 403):
        return AuthenticationError(message, provider=PROVIDER, status_code=status)
    if status == 429:
        return RateLimitError(message, provider=PROVIDER, status_code=status)
    if status == 504:
        return ProviderTimeoutError(message, provider=PROVIDER, status_code=status)
    retryable = status >= 500 or status == 408
    return ProviderError(message, provider=PROVIDER, status_code=status, retryable=retryable)


def _error_detail(body: bytes | bytearray | str | None) -> str:
    """The message of an AssemblyAI error body (``error`` / ``message`` / ``detail``)."""
    if not body:
        return ""
    text = body if isinstance(body, str) else bytes(body).decode("utf-8", "replace")
    try:
        data = json.loads(text)
    except ValueError:
        return text.strip()[:500]
    if not isinstance(data, dict):
        return text.strip()[:500]
    parts = [str(data[k]) for k in ("error_code", "code") if data.get(k)]
    for key in ("message", "error", "detail"):
        if data.get(key):
            parts.append(str(data[key]))
            break
    return ": ".join(parts) or text.strip()[:500]


# -------------------------------------------------------------------------------- STT
@register_provider(
    "stt",
    "assemblyai",
    description="AssemblyAI Universal-3.5 Pro / Universal-Streaming STT with neural end-of-turn",
    default_model=DEFAULT_MODEL,
    models=MODELS,
    env=(API_KEY_ENV,),
    requires=("websockets", "httpx"),
    local=False,
)
class AssemblyAISTT(STT):
    """AssemblyAI streaming speech-to-text (Streaming API v3).

    Events (:class:`~voice_agent_next.stt.STTEventType`):

    * ``SpeechStarted`` (Universal-3.5 Pro), or the first partial with words
      (Universal-Streaming, which has no ``SpeechStarted``) -> ``START_OF_SPEECH``
      (``transcript.start_time`` = start of the turn, in stream seconds);
    * ``Turn`` with ``end_of_turn: false`` -> ``INTERIM_TRANSCRIPT`` when the text changed
      (every partial re-transcribes the whole turn: it replaces, not extends, the last);
    * the turn's final ``Turn`` (``end_of_turn: true``, formatted when ``format_turns``
      asks for it) -> ``FINAL_TRANSCRIPT`` + ``END_OF_SPEECH`` (``end_time`` = end of the
      last word) + ``END_OF_TURN``. Turns ended by our own
      :meth:`~voice_agent_next.stt.STTStream.flush` (``ForceEndpoint``) get no
      ``END_OF_TURN``: whoever flushed owns that decision. ``end_of_turn=False`` never
      emits ``END_OF_TURN``.

    Two ways to run it in a cascade:

    * ``end_of_turn=True`` (default, ``capabilities.end_of_turn``): AssemblyAI's neural
      turn detection ends the user's turn and the cascade commits it at once. Run the
      cascade without a VAD (``SpeechStarted`` drives barge-in) or with one (it then only
      drives barge-in). Tune with ``mode``, ``min_turn_silence``, ``max_turn_silence``...
    * ``end_of_turn=False``: the cascade's VAD / turn detector owns endpointing and its
      flush sends ``ForceEndpoint``; turns AssemblyAI ends on its own become plain finals.
      A flush while no turn is open is answered by an empty final after
      ``force_endpoint_grace`` seconds, unless a turn starts meanwhile.

    Args:
        model: ``universal-3-5-pro`` (default), ``universal-streaming-english`` or
            ``universal-streaming-multilingual``.
        api_key: AssemblyAI API key (default: ``ASSEMBLYAI_API_KEY``).
        token: a temporary streaming token (:meth:`create_temporary_token`) used instead of
            the API key for streaming (one session per token).
        language: steers Universal-3.5 Pro (``language_codes=[language]``); ignored by the
            Universal-Streaming models (English-only / auto-detecting).
        language_codes: languages to steer Universal-3.5 Pro towards (overrides
            ``language``).
        language_detection: report ``language_code`` on turns (U3.5 Pro, multilingual).
        sample_rate: rate of the PCM sent (8-96 kHz; input is resampled to it).
        end_of_turn: emit provider ``END_OF_TURN`` events (see above).
        mode: U3.5 Pro preset, ``min_latency`` / ``balanced`` / ``max_accuracy``; sets the
            turn-detection defaults.
        min_turn_silence, max_turn_silence: silence (ms) before an end-of-turn check /
            before a turn is forced to end.
        end_of_turn_confidence_threshold: Universal-Streaming end-of-turn confidence (0-1).
        vad_threshold: speech / silence classification threshold (0-1).
        interruption_delay: U3.5 Pro delay (0-1000 ms) of the first partial of a turn.
        continuous_partials: U3.5 Pro partials every ~3 s during long speech.
        format_turns: Universal-Streaming: finals with punctuation and casing (U3.5 Pro
            finals are always formatted).
        keyterms: terms to boost (at most 100, each at most 50 characters).
        prompt: U3.5 Pro context about the audio (at most 1750 characters).
        agent_context: U3.5 Pro: the agent's last reply, as context for the next turn.
        filter_profanity, voice_focus, domain, speaker_labels: passed through.
        inactivity_timeout: seconds (5-3600) after which AssemblyAI ends an idle session;
            when set, ``KeepAlive`` is sent while no audio flows.
        force_endpoint_grace: see ``end_of_turn=False`` above (``0`` acknowledges at once).
        chunk_ms: audio chunk size (50-1000 ms; AssemblyAI recommends ~50 ms).
        region: ``"us"`` / ``"eu"`` data-zone endpoints (default: edge routing).
        base_url: streaming origin override (``wss://...``; also for tests).
        sync_url: origin of the Sync STT API used by :meth:`transcribe`
            (``None`` streams batch audio instead).
        sync_model: model of the Sync STT API.
        http_client: optional ``httpx.AsyncClient`` (not closed by :meth:`aclose`).
        connect_timeout: WebSocket / HTTP connection timeout in seconds.
        close_timeout: how long to wait for the last ``Turn`` / ``Termination`` after the
            input ends.
        request_timeout: HTTP timeout of :meth:`transcribe`.
        extra_params: additional connection parameters, passed through verbatim.
    """

    provider = PROVIDER

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        token: str | None = None,
        language: str | None = None,
        language_codes: Sequence[str] = (),
        language_detection: bool | None = None,
        sample_rate: int = 16_000,
        end_of_turn: bool = True,
        mode: Literal["min_latency", "balanced", "max_accuracy"] | None = None,
        min_turn_silence: int | None = None,
        max_turn_silence: int | None = None,
        end_of_turn_confidence_threshold: float | None = None,
        vad_threshold: float | None = None,
        interruption_delay: int | None = None,
        continuous_partials: bool | None = None,
        format_turns: bool | None = None,
        keyterms: Sequence[str] = (),
        prompt: str | None = None,
        agent_context: str | None = None,
        filter_profanity: bool | None = None,
        voice_focus: Literal["near-field", "far-field"] | None = None,
        domain: str | None = None,
        speaker_labels: bool | None = None,
        inactivity_timeout: int | None = None,
        force_endpoint_grace: float = 0.3,
        chunk_ms: int = 50,
        region: Literal["us", "eu"] | None = None,
        base_url: str | None = None,
        sync_url: str | None = DEFAULT_SYNC_URL,
        sync_model: str = DEFAULT_MODEL,
        http_client: httpx.AsyncClient | None = None,
        connect_timeout: float = 10.0,
        close_timeout: float = 5.0,
        request_timeout: float = 35.0,
        extra_params: Mapping[str, Any] | None = None,
    ) -> None:
        model = model or DEFAULT_MODEL
        pro = _is_pro(model)
        for name, seq in (("keyterms", keyterms), ("language_codes", language_codes)):
            if isinstance(seq, str):
                raise ConfigurationError(f"{name} must be a sequence of strings, not a string")
        if len(keyterms) > _MAX_KEYTERMS:
            raise ConfigurationError(f"at most {_MAX_KEYTERMS} keyterms, got {len(keyterms)}")
        long_terms = [t for t in keyterms if len(t) > _MAX_KEYTERM_CHARS]
        if long_terms:
            raise ConfigurationError(
                f"keyterms must be at most {_MAX_KEYTERM_CHARS} characters: {long_terms[0]!r}"
            )
        for name, text in (("prompt", prompt), ("agent_context", agent_context)):
            if text is not None and len(text) > _MAX_PROMPT_CHARS:
                raise ConfigurationError(f"{name} must be at most {_MAX_PROMPT_CHARS} characters")
        if not 8_000 <= sample_rate <= 96_000:
            raise ConfigurationError(f"sample_rate must be 8000-96000 Hz, got {sample_rate}")
        if mode is not None and mode not in _MODES:
            raise ConfigurationError(f"mode must be one of {_MODES}, got {mode!r}")
        _check_range("min_turn_silence", min_turn_silence, 0, 10_000)
        _check_range("max_turn_silence", max_turn_silence, 0, 60_000)
        _check_range("end_of_turn_confidence_threshold", end_of_turn_confidence_threshold, 0, 1)
        _check_range("vad_threshold", vad_threshold, 0, 1)
        _check_range("interruption_delay", interruption_delay, 0, 1000)
        _check_range("inactivity_timeout", inactivity_timeout, 5, 3600)
        _check_range("chunk_ms", chunk_ms, _MIN_CHUNK * 1000, _MAX_CHUNK * 1000)
        if force_endpoint_grace < 0:
            raise ConfigurationError("force_endpoint_grace must be >= 0")
        pro_only = {
            "mode": mode,
            "interruption_delay": interruption_delay,
            "continuous_partials": continuous_partials,
            "prompt": prompt,
            "agent_context": agent_context,
            "language_codes": list(language_codes) or None,
        }
        streaming_only = {
            "end_of_turn_confidence_threshold": end_of_turn_confidence_threshold,
            "format_turns": format_turns,
        }
        wrong = [k for k, v in (streaming_only if pro else pro_only).items() if v is not None]
        if wrong:
            family = "Universal-Streaming" if pro else "Universal-3.5 Pro"
            raise ConfigurationError(f"{', '.join(wrong)}: {family} only, not {model}")
        if region is not None and region not in REGION_URLS:
            raise ConfigurationError(f"region must be one of {sorted(REGION_URLS)}, got {region!r}")
        super().__init__(
            model=model,
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=True,
                word_timestamps=True,
                end_of_turn=end_of_turn,
                language_detection=bool(language_detection),
            ),
            sample_rate=sample_rate,
            language=language,
        )
        self.token = token
        self._api_key = (api_key or os.environ.get(API_KEY_ENV) or "").strip()
        if not self._api_key and not token:
            raise ConfigurationError(
                f"AssemblyAI needs an API key: pass api_key=... (or token=...) or set {API_KEY_ENV}"
            )
        self.is_pro = pro
        self.language_codes = list(language_codes)
        self.language_detection = language_detection
        self.mode = mode
        self.min_turn_silence = min_turn_silence
        self.max_turn_silence = max_turn_silence
        self.end_of_turn_confidence_threshold = end_of_turn_confidence_threshold
        self.vad_threshold = vad_threshold
        self.interruption_delay = interruption_delay
        self.continuous_partials = continuous_partials
        self.format_turns = format_turns
        self.keyterms = list(keyterms)
        self.prompt = prompt
        self.agent_context = agent_context
        self.filter_profanity = filter_profanity
        self.voice_focus = voice_focus
        self.domain = domain
        self.speaker_labels = speaker_labels
        self.inactivity_timeout = inactivity_timeout
        self.force_endpoint_grace = force_endpoint_grace
        self.chunk_ms = chunk_ms
        self.base_url = (base_url or REGION_URLS.get(region or "", DEFAULT_STREAMING_URL)).rstrip(
            "/"
        )
        self.sync_url = sync_url.rstrip("/") if sync_url else None
        self.sync_model = sync_model
        self.connect_timeout = connect_timeout
        self.close_timeout = close_timeout
        self.request_timeout = request_timeout
        self.extra_params = dict(extra_params or {})
        self._http = http_client
        self._owns_http = http_client is None

    # ---------------------------------------------------------------- requests
    @property
    def expects_formatted_finals(self) -> bool:
        """Universal-Streaming with ``format_turns`` sends every final twice (unformatted
        first): only the formatted copy is final."""
        return not self.is_pro and bool(self.format_turns)

    def _streaming_language_codes(self, language: str | None) -> list[str]:
        if not self.is_pro:
            return []
        if self.language_codes:
            return list(self.language_codes)
        code = _language_code(language)
        return [code] if code else []

    def params(self, language: str | None = None) -> dict[str, Any]:
        """The connection parameters (``None`` values are not sent)."""
        params: dict[str, Any] = {
            "speech_model": self.model,
            "encoding": "pcm_s16le",
            "sample_rate": self.sample_rate,
            "mode": self.mode,
            "language_codes": self._streaming_language_codes(language or self.language) or None,
            "language_detection": self.language_detection,
            "min_turn_silence": self.min_turn_silence,
            "max_turn_silence": self.max_turn_silence,
            "end_of_turn_confidence_threshold": self.end_of_turn_confidence_threshold,
            "vad_threshold": self.vad_threshold,
            "interruption_delay": self.interruption_delay,
            "continuous_partials": self.continuous_partials,
            "format_turns": self.format_turns,
            "keyterms_prompt": self.keyterms or None,
            "prompt": self.prompt,
            "agent_context": self.agent_context,
            "filter_profanity": self.filter_profanity,
            "voice_focus": self.voice_focus,
            "domain": self.domain,
            "speaker_labels": self.speaker_labels,
            "inactivity_timeout": self.inactivity_timeout,
        }
        params |= self.extra_params
        return {k: v for k, v in params.items() if v is not None}

    def url(self, language: str | None = None) -> str:
        """The streaming WebSocket URL (query string included, no credentials)."""
        query = urlencode([(k, _query_value(v)) for k, v in self.params(language).items()])
        return f"{self.base_url}/v3/ws?{query}"

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise ConfigurationError(
                f"this AssemblyAI request needs an API key (api_key=... or {API_KEY_ENV})"
            )
        return {"Authorization": self._api_key}

    def _http_origin(self) -> str:
        scheme, _, rest = self.base_url.partition("://")
        return f"{'http' if scheme in ('ws', 'http') else 'https'}://{rest}"

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(self.request_timeout, connect=self.connect_timeout)
            )
            self._owns_http = True
        return self._http

    async def create_temporary_token(
        self, *, expires_in_seconds: int = 60, max_session_duration_seconds: int | None = None
    ) -> str:
        """A temporary streaming token (``GET /v3/token``) for a browser or other client.

        ``expires_in_seconds`` (1-600) is the window to open the WebSocket with it;
        ``max_session_duration_seconds`` (60-10800) caps the session it opens.
        """
        _check_range("expires_in_seconds", expires_in_seconds, 1, 600)
        _check_range("max_session_duration_seconds", max_session_duration_seconds, 60, 10_800)
        params: dict[str, int] = {"expires_in_seconds": expires_in_seconds}
        if max_session_duration_seconds is not None:
            params["max_session_duration_seconds"] = max_session_duration_seconds
        data = await self._request("GET", f"{self._http_origin()}/v3/token", params=params)
        token = data.get("token")
        if not isinstance(token, str) or not token:
            raise ProviderError("AssemblyAI returned no token", provider=PROVIDER)
        return token

    async def _request(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        headers = {**kwargs.pop("headers", {}), **self._headers()}
        try:
            response = await self._client().request(method, url, headers=headers, **kwargs)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"AssemblyAI request timed out: {exc!r}", provider=PROVIDER
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderConnectionError(
                f"AssemblyAI request failed: {exc!r}", provider=PROVIDER
            ) from exc
        if response.status_code >= 400:
            raise _http_error(response.status_code, _error_detail(response.content))
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(
                f"AssemblyAI returned invalid JSON: {response.text[:200]}", provider=PROVIDER
            ) from exc
        if not isinstance(data, dict):
            raise ProviderError("AssemblyAI returned an unexpected response", provider=PROVIDER)
        return data

    async def _recognize(self, audio: AudioFrame, *, language: str | None) -> Transcript:
        """Batch recognition with the Sync STT API (80 ms - 120 s), else by streaming."""
        if self.sync_url is None or audio.duration > SYNC_MAX_DURATION:
            return await super()._recognize(audio, language=language)
        if audio.duration < SYNC_MIN_DURATION:
            return Transcript("", language)  # Sync rejects it (audio_too_short); no speech
        config: dict[str, Any] = {
            "sample_rate": audio.sample_rate,
            "channels": audio.channels,
            "timestamps": True,
        }
        codes = list(self.language_codes) or [c for c in [_language_code(language)] if c]
        if codes:
            config["language_codes"] = codes
        if self.keyterms:
            config["keyterms_prompt"] = list(self.keyterms)
        if self.prompt:
            config["prompt"] = self.prompt
        data = await self._request(
            "POST",
            f"{self.sync_url}/v1/transcribe",
            headers={"X-AAI-Model": self.sync_model},
            files={
                "audio": ("audio.pcm", audio.data, "audio/pcm"),
                "config": (None, _json(config), "application/json"),
            },
        )
        words = _words(data.get("words"))
        detected = data.get("language_code")
        return Transcript(
            text=str(data.get("text") or "").strip(),
            language=detected if isinstance(detected, str) and detected else language,
            confidence=_float(data.get("confidence")),
            start_time=words[0].start if words else None,
            end_time=words[-1].end if words else None,
            words=words,
        )

    def _create_stream(self, *, language: str | None) -> STTStream:
        return AssemblyAIStream(self, language=language)

    def stream(self, *, language: str | None = None) -> AssemblyAIStream:
        """Open a streaming session (see :class:`AssemblyAIStream`)."""
        stream = super().stream(language=language)
        assert isinstance(stream, AssemblyAIStream)
        return stream

    async def aclose(self) -> None:
        if self._http is not None and self._owns_http:
            http, self._http = self._http, None
            await http.aclose()


async def _connect(stt: AssemblyAISTT, url: str) -> ClientConnection:
    headers: dict[str, str] = {}
    if stt.token:
        url = f"{url}&{urlencode({'token': stt.token})}"
    else:
        headers = stt._headers()
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
        raise ConfigurationError(f"invalid AssemblyAI URL: {exc}") from exc
    except TimeoutError as exc:
        raise ProviderTimeoutError(
            "timed out connecting to the AssemblyAI streaming API", provider=PROVIDER
        ) from exc
    except (OSError, InvalidHandshake) as exc:
        raise ProviderConnectionError(
            f"could not connect to the AssemblyAI streaming API: {exc}", provider=PROVIDER
        ) from exc


class AssemblyAIStream(STTStream):
    """One AssemblyAI streaming session (see :class:`AssemblyAISTT` for the events).

    Attributes set once the session began: :attr:`session_id`, :attr:`expires_at` (Unix
    seconds) and :attr:`configuration` (what the server applied, e.g. its ``model``).
    """

    def __init__(self, stt: AssemblyAISTT, *, language: str | None) -> None:
        self._aai = stt
        self._chunk_bytes = round(stt.sample_rate * stt.chunk_ms / 1000) * 2
        self._min_bytes = round(stt.sample_rate * _MIN_CHUNK) * 2
        self._buf = bytearray()
        self._ws: ClientConnection | None = None
        self._connected = asyncio.Event()
        self._closing = False
        self._terminated = False
        self._server_error: ProviderError | None = None
        self._segment_id = new_id("seg_")
        self._turn_active = False
        self._last_final_order = -1
        self._last_text = ""
        self._last_transcript: Transcript | None = None
        self._pending_flushes = 0
        self._grace: asyncio.Task[None] | None = None
        self.session_id: str | None = None
        self.expires_at: int | None = None
        self.configuration: dict[str, Any] = {}
        super().__init__(stt, language=language)

    # ------------------------------------------------------------------ public API
    async def update_configuration(self, **fields: Any) -> None:
        """Send ``UpdateConfiguration`` (a delta; applies to audio processed afterwards).

        Fields: ``min_turn_silence``, ``max_turn_silence``, ``vad_threshold``,
        ``keyterms_prompt``, ``prompt``, ``agent_context``, ``mode``,
        ``interruption_delay``, ``continuous_partials``, ``language_codes``,
        ``end_of_turn_confidence_threshold``, ``session_heartbeat``. For example raise
        ``min_turn_silence`` while a caller dictates a phone number, then restore it
        (``mode="balanced"`` restores the preset's defaults on U3.5 Pro).
        """
        unknown = sorted(set(fields) - _UPDATE_FIELDS)
        if unknown:
            raise ConfigurationError(f"not updatable mid-stream: {', '.join(unknown)}")
        keyterms = fields.get("keyterms_prompt")
        if isinstance(keyterms, str):
            raise ConfigurationError("keyterms_prompt must be a sequence of strings")
        message: dict[str, Any] = {"type": "UpdateConfiguration"}
        message |= {k: list(v) if isinstance(v, tuple) else v for k, v in fields.items()}
        if not self._connected.is_set():
            connected = asyncio.ensure_future(self._connected.wait())
            try:
                await asyncio.wait({connected, self._task}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                connected.cancel()
        ws = self._ws
        if ws is None or self._closing or self._task.done():
            raise RuntimeError("the AssemblyAI stream is not open")
        await ws.send(_json(message))

    # ------------------------------------------------------------------- plumbing
    async def _run(self) -> None:
        stt = self._aai
        ws = await _connect(stt, stt.url(self._language))
        self._ws = ws
        self._connected.set()
        receiver = asyncio.create_task(self._recv_loop(ws), name="assemblyai-stt-recv")
        sender = asyncio.create_task(self._send_loop(ws), name="assemblyai-stt-send")
        try:
            await asyncio.wait({receiver, sender}, return_when=asyncio.FIRST_COMPLETED)
            self._raise_task_error(receiver, sender)
            if not sender.done():  # the server ended the session while audio was flowing
                raise self._server_error or ProviderConnectionError(
                    "AssemblyAI closed the streaming session unexpectedly", provider=PROVIDER
                )
            # Terminate sent: AssemblyAI flushes the last Turn(s), then sends Termination
            await asyncio.wait({receiver}, timeout=stt.close_timeout)
            self._raise_task_error(receiver, sender)
            if not receiver.done():
                logger.warning(
                    "AssemblyAI: no Termination %.1fs after Terminate", stt.close_timeout
                )
            self._finish_session()
        finally:
            self._closing = True
            if self._grace is not None:
                await cancel_and_wait(self._grace)
            await cancel_and_wait(sender, receiver)
            with contextlib.suppress(Exception):
                await ws.close()

    def _raise_task_error(self, *tasks: asyncio.Task[None]) -> None:
        for task in tasks:
            if task.done() and not task.cancelled():
                exc = task.exception()
                if exc is not None:
                    raise exc

    async def _send_loop(self, ws: ClientConnection) -> None:
        # With inactivity_timeout set, AssemblyAI ends sessions that receive nothing.
        idle = self._aai.inactivity_timeout
        interval = idle / 2 if idle else None
        try:
            while True:
                try:
                    async with asyncio.timeout(interval):
                        item = await self._input.recv()
                except TimeoutError:
                    await ws.send(_MSG_KEEPALIVE)
                    continue
                except ChanClosed:
                    break
                if self.is_flush(item):
                    # A tail under 50 ms would be rejected (3007): it waits for the next
                    # chunk. It is the end of the silence a VAD waited for, so the endpoint
                    # does not need it, and holding it keeps word times on our timeline.
                    if len(self._buf) >= self._min_bytes:
                        await ws.send(bytes(self._buf))
                        self._buf.clear()
                    await ws.send(_MSG_FORCE_ENDPOINT)
                    self._on_force_endpoint()
                    continue
                assert isinstance(item, AudioFrame)
                self._buf += item.data
                while len(self._buf) >= self._chunk_bytes:
                    await ws.send(bytes(self._buf[: self._chunk_bytes]))
                    del self._buf[: self._chunk_bytes]
            if self._buf:  # the session ends: pad the tail to the minimum chunk size
                tail = bytes(self._buf).ljust(self._min_bytes, b"\x00")
                self._buf.clear()
                await ws.send(tail)
            self._closing = True
            await ws.send(_MSG_TERMINATE)
        except ConnectionClosed as exc:
            raise self._server_error or self._close_error(exc) from exc

    async def _recv_loop(self, ws: ClientConnection) -> None:
        try:
            async for message in ws:
                if isinstance(message, str):
                    self._on_message(_parse(message))
                    if self._terminated:
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
                "AssemblyAI closed the streaming session unexpectedly", provider=PROVIDER
            )

    @staticmethod
    def _close_error(exc: ConnectionClosed) -> ProviderError:
        frame = exc.rcvd
        if frame is None:
            error: ProviderError = ProviderConnectionError(
                "AssemblyAI streaming connection lost (no close frame)", provider=PROVIDER
            )
        else:
            error = _code_error(int(frame.code), frame.reason)
        error.__cause__ = exc
        return error

    # -------------------------------------------------------------------- events
    def _on_message(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "Turn":
            self._on_turn(message)
        elif kind == "SpeechStarted":
            timestamp = _float(message.get("timestamp"))
            self._begin_turn(timestamp / 1000.0 if timestamp is not None else None)
        elif kind == "Begin":
            self.session_id = message.get("id")
            expires = message.get("expires_at")
            self.expires_at = int(expires) if isinstance(expires, (int, float)) else None
            config = message.get("configuration")
            self.configuration = dict(config) if isinstance(config, dict) else {}
            applied = self.configuration.get("model")
            if applied and applied != self._aai.model:
                logger.warning(
                    "AssemblyAI applied model %s, not the requested %s", applied, self._aai.model
                )
            logger.debug("AssemblyAI session %s began", self.session_id)
        elif kind == "Termination":
            self._terminated = True
            logger.debug(
                "AssemblyAI session %s terminated (%ss of audio, %ss billed)",
                self.session_id,
                message.get("audio_duration_seconds"),
                message.get("session_duration_seconds"),
            )
        elif kind == "Error" or (kind is None and "error" in message):
            code = message.get("error_code")
            self._server_error = _code_error(
                int(code) if isinstance(code, (int, float)) else None,
                str(message.get("error") or ""),
            )
            raise self._server_error
        else:  # Heartbeat, SpeakerRevision...
            logger.debug("AssemblyAI: ignoring %s message", kind)

    def _transcript(self, message: dict[str, Any], *, final: bool) -> Transcript:
        words = _words(message.get("words"))
        text = str(message.get("transcript") or "").strip()
        if not final and words:
            # Universal-Streaming partials keep the word still being decoded out of
            # ``transcript``; show it in the interim text too.
            joined = " ".join(w.word for w in words if w.word).strip()
            if len(joined) > len(text):
                text = joined
        language = message.get("language_code")
        return Transcript(
            text=text,
            language=language if isinstance(language, str) and language else self._language,
            confidence=_mean_confidence(words),
            start_time=words[0].start if words else None,
            end_time=words[-1].end if words else None,
            words=words,
        )

    def _on_turn(self, message: dict[str, Any]) -> None:
        order_value = message.get("turn_order")
        order = int(order_value) if isinstance(order_value, (int, float)) else None
        if order is not None and order <= self._last_final_order:
            return  # the formatted copy of a final we already emitted, or a stale message
        end = bool(message.get("end_of_turn"))
        formatted = bool(message.get("turn_is_formatted"))
        if end and (formatted or not self._aai.expects_formatted_finals):
            self._end_turn(order, self._transcript(message, final=True))
            return
        transcript = self._transcript(message, final=False)
        if not transcript.text:
            return
        self._begin_turn(transcript.start_time)
        if transcript.text != self._last_text:
            self._last_text, self._last_transcript = transcript.text, transcript
            self._event(STTEventType.INTERIM_TRANSCRIPT, transcript)

    def _begin_turn(self, start_time: float | None) -> None:
        if self._grace is not None:  # a turn is open: its final answers the flush
            self._grace.cancel()
            self._grace = None
        if not self._turn_active:
            self._turn_active = True
            self._last_text, self._last_transcript = "", None
            self._segment_id = new_id("seg_")
            self._event(STTEventType.START_OF_SPEECH, Transcript("", start_time=start_time))

    def _end_turn(self, order: int | None, transcript: Transcript) -> None:
        if order is not None:
            self._last_final_order = order
        flushed = self._pending_flushes > 0
        if not transcript.text and not self._turn_active and not flushed:
            return  # e.g. the empty closing Turn of Universal-Streaming: nothing to report
        if transcript.text:
            self._begin_turn(transcript.start_time)
        self._pending_flushes = 0
        if self._grace is not None:
            self._grace.cancel()
            self._grace = None
        self._event(STTEventType.FINAL_TRANSCRIPT, transcript)
        if self._turn_active:
            self._event(STTEventType.END_OF_SPEECH, transcript)
        if self._aai.capabilities.end_of_turn and transcript.text and not flushed:
            self._event(STTEventType.END_OF_TURN, transcript)
        self._turn_active = False
        self._last_text, self._last_transcript = "", None
        if not flushed:
            self._report_usage()

    def _on_force_endpoint(self) -> None:
        self._pending_flushes += 1
        if self._turn_active or self._grace is not None:
            return  # the open turn's final answers it
        if self._aai.force_endpoint_grace <= 0:
            self._ack_flushes()
        else:
            self._grace = asyncio.create_task(self._grace_timer(), name="assemblyai-flush-ack")

    async def _grace_timer(self) -> None:
        await asyncio.sleep(self._aai.force_endpoint_grace)
        self._grace = None
        if not self._turn_active:
            self._ack_flushes()

    def _ack_flushes(self) -> None:
        """No turn is open: nothing to finalize, acknowledge pending flushes with ``""``."""
        if self._pending_flushes:
            self._pending_flushes = 0
            self._event(STTEventType.FINAL_TRANSCRIPT, Transcript("", self._language))

    def _finish_session(self) -> None:
        # A turn still open when the session ended: its last partial is all we have.
        if self._turn_active and self._last_transcript is not None:
            self._end_turn(None, self._last_transcript)
        self._ack_flushes()

    def _event(self, kind: STTEventType, transcript: Transcript | None = None) -> None:
        self._emit(STTEvent(kind, transcript, self._segment_id))

    def _report_usage(self) -> None:
        # The base class reports usage when a flush is answered; turns AssemblyAI ends on
        # its own are never flushed, so report their audio here (no flush latency to add).
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
