"""OpenAI Realtime speech-to-speech engine (WebSocket, GA protocol) and compatible backends.

``AgentSession("openai/gpt-realtime-2.1")`` runs an agent on OpenAI's native
speech-to-speech models. The same client drives every OpenAI-Realtime-compatible
server: a :class:`RealtimeProfile` captures what differs between them — URL and
authentication, the ``session.update`` dialect (GA, beta or xAI), beta-era event names,
audio sample rates and missing features (cancel, truncation, text input, per-response
instructions). Thin provider modules select a profile: ``azure_openai``, ``xai``,
``qwen_omni``, ``vllm_realtime``, ``speaches`` and ``localai``.

Server events are mapped onto :mod:`voice_agent_next.events`:

=====================================================  ===================================
``input_audio_buffer.speech_started``                  ``InputSpeechStarted``
``input_audio_buffer.speech_stopped``                  ``InputSpeechStopped`` (speech end)
``input_audio_buffer.committed``                       ``InputCommitted``
``conversation.item.input_audio_transcription.*``      ``InputTranscript`` (by ``item_id``)
``response.created`` / ``response.done``               ``ResponseStarted`` / ``ResponseDone``
``response.output_audio.delta``                        ``ResponseAudio``
``response.output_audio_transcript.delta``             ``ResponseText``
function-call items (``response.output_item.done``)    ``ResponseToolCall``
``error``                                              ``EngineErrorEvent``
=====================================================  ===================================

Every response also emits :class:`~voice_agent_next.metrics.EngineMetrics` (time from the
response trigger to the first audio, token usage) through ``engine.emit("metrics")``.

Speech end: OpenAI documents ``speech_stopped.audio_end_ms`` as the end of the audio
committed to the model, i.e. *including* the trailing silence (``silence_duration_ms`` for
``server_vad``, a variable hold for ``semantic_vad``), while other servers report the
speech end itself. The connection keeps a short history of the input levels it sent and
reports the last voiced position before ``audio_end_ms`` (falling back to ``audio_end_ms``
when the levels are inconclusive), so voice-to-voice latency is measured from where the
user really stopped talking with every backend.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import inspect
import json
import math
import os
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal, TypeAlias
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
from websockets.asyncio.client import ClientConnection
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed, InvalidStatus, InvalidURI

from ...audio.frame import AudioFrame
from ...chat import ChatContext, ChatMessage, FunctionCall, FunctionCallOutput
from ...engine import EngineCapabilities, EngineConnection, EngineOptions, S2SEngine
from ...errors import (
    AuthenticationError,
    ConfigurationError,
    EngineError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from ...events import (
    EngineErrorEvent,
    EngineStatus,
    EngineUsage,
    InputCommitted,
    InputSpeechStarted,
    InputSpeechStopped,
    InputTranscript,
    ResponseAudio,
    ResponseDone,
    ResponseStarted,
    ResponseStatus,
    ResponseText,
    ResponseToolCall,
)
from ...metrics import EngineMetrics
from ...registry import register_provider
from ...tools import FunctionTool
from ...utils.aio import BackgroundTasks, cancel_and_wait
from ...utils.clock import now
from ...utils.ids import new_id
from ...utils.log import logger

__all__ = [
    "BETA_EVENT_ALIASES",
    "PROFILES",
    "OpenAIRealtimeConnection",
    "OpenAIRealtimeEngine",
    "RealtimeConnectTimeoutError",
    "RealtimeProfile",
    "TurnDetection",
    "get_profile",
    "realtime_url",
]

TurnDetection: TypeAlias = Literal["server_vad", "semantic_vad"] | Mapping[str, Any] | None
"""``"server_vad"``, ``"semantic_vad"``, a raw ``turn_detection`` object, or ``None``
(manual turns: the application calls ``commit_input()``)."""

SessionDialect: TypeAlias = Literal["ga", "beta", "xai"]

#: Beta-era server event names (Qwen-Omni-Realtime, Speaches, older self-hosted servers)
#: mapped to their GA equivalents. Harmless for GA servers, so every profile accepts them.
BETA_EVENT_ALIASES: Final[Mapping[str, str]] = {
    "response.audio.delta": "response.output_audio.delta",
    "response.audio.done": "response.output_audio.done",
    "response.audio_transcript.delta": "response.output_audio_transcript.delta",
    "response.audio_transcript.done": "response.output_audio_transcript.done",
    "response.text.delta": "response.output_text.delta",
    "response.text.done": "response.output_text.done",
    "conversation.item.created": "conversation.item.added",
}

VERBATIM_INSTRUCTIONS: Final = 'Say exactly the following, verbatim, and nothing else: "{text}"'

_IGNORED_ERROR_CODES: Final = frozenset({"response_cancel_not_active"})
"""Errors caused by benign races (e.g. cancelling a response the server VAD already
cancelled); logged, never surfaced."""
_AUTH_ERROR_CODES: Final = frozenset({"invalid_api_key", "invalid_authentication", "unauthorized"})
_RATE_LIMIT_CODES: Final = frozenset({"rate_limit_exceeded", "insufficient_quota"})
_EXPIRY_CODES: Final = frozenset({"session_expired", "max_duration"})
_STATUSES: Final[Mapping[str, ResponseStatus]] = {
    "completed": "completed",
    "cancelled": "cancelled",
    "incomplete": "incomplete",
    "failed": "failed",
    "in_progress": "completed",
}
_MAX_MESSAGE_SIZE: Final = 32 * 2**20
_MAX_TRACKED_ITEMS: Final = 256
_RECONNECT_WINDOW: Final = 60.0
_CANCEL_TIMEOUT: Final = 2.0
_FINISH_TIMEOUT: Final = 10.0
_TRIGGER_TTL: Final = 30.0
"""A response request older than this that never started a response is forgotten."""
_UNSET: Final[Any] = object()


# ------------------------------------------------------------------------------ profiles
@dataclass(frozen=True)
class RealtimeProfile:
    """Everything that differs between OpenAI-Realtime-compatible backends.

    The defaults describe OpenAI's GA Realtime API; :data:`PROFILES` holds the built-in
    compatibility profiles. Pass ``profile=RealtimeProfile(...)`` (or
    ``dataclasses.replace(PROFILES["openai"], ...)``) to target another server.
    """

    name: str
    base_url: str | None = "wss://api.openai.com/v1"
    """Base URL (``.../v1``); ``/realtime`` is appended. ``http(s)://`` becomes ``ws(s)://``."""
    base_url_env: tuple[str, ...] = ()
    """Environment variables that may hold the base URL (first one set wins)."""
    api_key_env: tuple[str, ...] = ()
    api_key_required: bool = True
    auth_header: str = "Authorization"
    """``Authorization`` (``Bearer <key>``) or a raw key header such as Azure's ``api-key``."""
    default_model: str | None = None
    dialect: SessionDialect = "ga"
    """``session.update`` shape: GA (``session.audio.*``), beta (flat
    ``input_audio_format``/``voice``/``turn_detection``) or xAI (GA audio formats with
    top-level ``voice``/``turn_detection``)."""
    beta_audio_format: str = "pcm16"
    """Beta dialect only: value of ``input_audio_format``/``output_audio_format``."""
    input_sample_rate: int = 24_000
    output_sample_rate: int = 24_000
    default_voice: str | None = None
    default_turn_detection: TurnDetection = "server_vad"
    semantic_vad: bool = True
    turn_detection_flags: bool = True
    """Accepts ``create_response``/``interrupt_response`` in ``turn_detection``."""
    default_transcription: Mapping[str, Any] | None = None
    """Input transcription config sent by default (``None``: leave the server default)."""
    builtin_transcription: bool = False
    """The server transcribes user audio even without a transcription config."""
    transcription_language: bool = True
    """Accepts a language hint in the input transcription config."""
    event_aliases: Mapping[str, str] = field(default_factory=lambda: dict(BETA_EVENT_ALIASES))
    supports_cancel: bool = True
    supports_truncate: bool = True
    text_input: bool = True
    """Accepts user/assistant text items (``conversation.item.create`` messages)."""
    response_instructions: bool = True
    """Accepts per-response ``instructions`` in ``response.create``; otherwise the session
    instructions are patched for one response and restored when it starts."""
    isolated_say: bool = False
    """``say()`` sends ``response.input = []`` (no context, best verbatim compliance)."""
    say_mode: Literal["instructions", "force_message"] = "instructions"
    """``force_message``: xAI's scripted TTS item (truly verbatim)."""
    tool_format: Literal["flat", "nested"] = "flat"
    """``flat``: Realtime ``{"type", "name", ...}``; ``nested``: Chat-Completions style."""
    temperature: bool = False
    """Accepts ``temperature`` in the session (GA removed it)."""
    max_output_tokens: bool = True
    max_session_duration: float | None = None


_OPENAI = RealtimeProfile(
    name="openai",
    api_key_env=("OPENAI_API_KEY",),
    default_model="gpt-realtime-2.1",
    default_voice="marin",
    default_turn_detection="semantic_vad",
    default_transcription={"model": "gpt-4o-mini-transcribe"},
    isolated_say=True,
    max_session_duration=3600.0,
)

#: Built-in compatibility profiles (see ``docs/providers/openai-realtime.md``).
PROFILES: dict[str, RealtimeProfile] = {
    "openai": _OPENAI,
    "azure_openai": replace(
        _OPENAI,
        name="azure_openai",
        base_url=None,  # https://<resource>.openai.azure.com -> .../openai/v1
        api_key_env=("AZURE_OPENAI_API_KEY",),
        auth_header="api-key",
        default_transcription={"model": "whisper-1"},
    ),
    "xai": RealtimeProfile(
        name="xai",
        base_url="wss://api.x.ai/v1",
        api_key_env=("XAI_API_KEY",),
        default_model="grok-voice-latest",
        dialect="xai",
        default_voice="eve",
        semantic_vad=False,
        turn_detection_flags=False,
        default_transcription={"model": "grok-transcribe"},
        builtin_transcription=True,
        say_mode="force_message",
        max_output_tokens=False,
    ),
    "qwen_omni": RealtimeProfile(
        name="qwen_omni",
        base_url=None,  # wss://{WorkspaceId}.{region}.maas.aliyuncs.com/api-ws/v1
        api_key_env=("DASHSCOPE_API_KEY",),
        default_model="qwen3.8-omni-flash-realtime",
        dialect="beta",
        beta_audio_format="pcm",
        input_sample_rate=16_000,
        default_transcription={"model": "qwen3-asr-flash-realtime"},
        builtin_transcription=True,
        transcription_language=False,
        supports_truncate=False,
        text_input=False,
        response_instructions=False,
        tool_format="nested",
        temperature=True,
        max_output_tokens=False,
        max_session_duration=7200.0,
    ),
    "vllm_realtime": RealtimeProfile(
        name="vllm_realtime",
        base_url="ws://localhost:8000/v1",
        base_url_env=("VLLM_BASE_URL",),
        api_key_env=("VLLM_API_KEY",),
        api_key_required=False,
    ),
    "speaches": RealtimeProfile(
        name="speaches",
        base_url="ws://localhost:8000/v1",
        base_url_env=("SPEACHES_BASE_URL",),
        api_key_env=("SPEACHES_API_KEY",),
        api_key_required=False,
        dialect="beta",
        semantic_vad=False,
        builtin_transcription=True,
        supports_cancel=False,
        supports_truncate=False,
        temperature=True,
        max_output_tokens=False,
    ),
    "localai": RealtimeProfile(
        name="localai",
        base_url="ws://localhost:8080/v1",
        base_url_env=("LOCALAI_BASE_URL",),
        api_key_env=("LOCALAI_API_KEY",),
        api_key_required=False,
        default_model="gpt-realtime",
        builtin_transcription=True,
    ),
}


def get_profile(profile: str | RealtimeProfile) -> RealtimeProfile:
    """Look up a built-in profile by name (profiles pass through unchanged)."""
    if isinstance(profile, RealtimeProfile):
        return profile
    key = profile.strip().lower().replace("-", "_")
    try:
        return PROFILES[key]
    except KeyError:
        raise ConfigurationError(
            f"unknown realtime profile {profile!r}; expected one of {sorted(PROFILES)}"
        ) from None


def realtime_url(
    base_url: str, *, model: str | None = None, query: Mapping[str, str] | None = None
) -> str:
    """``https://host/v1`` -> ``wss://host/v1/realtime?model=...`` (existing query kept)."""
    parts = urlsplit(base_url.strip())
    scheme = {"http": "ws", "https": "wss"}.get(parts.scheme, parts.scheme)
    if scheme not in ("ws", "wss") or not parts.netloc:
        raise ConfigurationError(f"invalid realtime base URL {base_url!r} (expected ws[s]://...)")
    path = parts.path.rstrip("/")
    if not path.endswith("/realtime"):
        path += "/realtime"
    params = dict(parse_qsl(parts.query, keep_blank_values=True))
    if model:
        params["model"] = model
    params.update(query or {})
    return urlunsplit((scheme, parts.netloc, path, urlencode(params), ""))


# ------------------------------------------------------------------------------- helpers
def _deep_merge(base: dict[str, Any], *overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    for override in overrides:
        for key, value in (override or {}).items():
            current = base.get(key)
            if isinstance(value, Mapping) and isinstance(current, dict):
                base[key] = _deep_merge(dict(current), value)
            else:
                base[key] = value
    return base


def _tool_payload(tool: FunctionTool, fmt: Literal["flat", "nested"]) -> dict[str, Any]:
    fn = {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
    if fmt == "nested":
        return {"type": "function", "function": fn}
    return {"type": "function", **fn}


def _turn_detection_payload(td: TurnDetection, profile: RealtimeProfile) -> dict[str, Any] | None:
    if td is None:
        return None
    cfg: dict[str, Any] = {"type": td} if isinstance(td, str) else dict(td)
    kind = cfg.get("type")
    if not isinstance(kind, str):
        raise ConfigurationError(f"turn_detection needs a 'type': {cfg!r}")
    if kind == "semantic_vad" and not profile.semantic_vad:
        raise ConfigurationError(
            f"the {profile.name} realtime profile does not support semantic_vad"
        )
    if profile.turn_detection_flags:
        cfg.setdefault("create_response", True)
        cfg.setdefault("interrupt_response", True)
    return cfg


def _parse_usage(raw: Any) -> EngineUsage | None:
    if not isinstance(raw, Mapping):
        return None
    # GA: input_token_details / output_token_details; Qwen: input_tokens_details / ...
    inp = raw.get("input_token_details") or raw.get("input_tokens_details") or {}
    out = raw.get("output_token_details") or raw.get("output_tokens_details") or {}

    def count(details: Any, key: str) -> int:
        value = details.get(key) if isinstance(details, Mapping) else None
        return int(value) if isinstance(value, (int, float)) else 0

    return EngineUsage(
        input_text_tokens=count(inp, "text_tokens"),
        input_audio_tokens=count(inp, "audio_tokens"),
        output_text_tokens=count(out, "text_tokens"),
        output_audio_tokens=count(out, "audio_tokens"),
        cached_tokens=count(inp, "cached_tokens"),
    )


class RealtimeConnectTimeoutError(ProviderTimeoutError, ProviderConnectionError):
    """Opening the realtime connection (TCP, TLS or WebSocket handshake) timed out.

    Both a :class:`~voice_agent_next.errors.ProviderTimeoutError` and a retryable
    :class:`~voice_agent_next.errors.ProviderConnectionError`: whether a connection attempt
    is refused at once or runs into ``connect_timeout`` depends on the network and the OS —
    Windows only reports a refused connection after retrying the SYN for about two seconds —
    so both outcomes must be handled as the same connection failure.
    """


def _handshake_error(exc: BaseException, provider: str, url: str) -> Exception:
    if isinstance(exc, InvalidStatus):
        status = exc.response.status_code
        body = (
            exc.response.body.decode("utf-8", "replace").strip()[:300] if exc.response.body else ""
        )
        msg = f"{provider}: realtime handshake rejected with HTTP {status}" + (
            f": {body}" if body else ""
        )
        if status in (401, 403):
            return AuthenticationError(msg, provider=provider, status_code=status)
        if status == 429:
            return RateLimitError(msg, provider=provider, status_code=status)
        if status >= 500:
            return ProviderConnectionError(msg, provider=provider, status_code=status)
        return ProviderError(msg, provider=provider, status_code=status)
    if isinstance(exc, InvalidURI):
        return ConfigurationError(f"{provider}: invalid realtime URL {url!r}: {exc}")
    if isinstance(exc, TimeoutError):
        return RealtimeConnectTimeoutError(
            f"{provider}: timed out connecting to {url}", provider=provider
        )
    return ProviderConnectionError(
        f"{provider}: cannot connect to {url}: {type(exc).__name__}: {exc}", provider=provider
    )


def _server_error(err: Mapping[str, Any], provider: str, context: str | None) -> ProviderError:
    etype = str(err.get("type") or "error")
    code = err.get("code")
    label = etype if not code or code == etype else f"{etype}/{code}"
    msg = f"{provider} realtime error ({label}): {err.get('message') or 'no message'}"
    if err.get("param"):
        msg += f" [param: {err['param']}]"
    if context:
        msg += f" [in reply to {context}]"
    if etype == "authentication_error" or code in _AUTH_ERROR_CODES:
        return AuthenticationError(msg, provider=provider)
    if etype == "rate_limit_error" or code in _RATE_LIMIT_CODES:
        return RateLimitError(msg, provider=provider, retryable=code != "insufficient_quota")
    if etype in ("server_error", "internal_error"):
        return ProviderError(msg, provider=provider, retryable=True)
    return ProviderError(msg, provider=provider)


def _bound(items: dict[str, Any], limit: int = _MAX_TRACKED_ITEMS) -> None:
    """Drop the oldest entries (dicts keep insertion order) of a per-item map."""
    while len(items) > limit:
        del items[next(iter(items))]


def _close_reason(exc: ConnectionClosed) -> str:
    frame = exc.rcvd or exc.sent
    if frame is None:
        return "connection lost (no close frame)"
    return f"closed with code {frame.code}" + (f": {frame.reason}" if frame.reason else "")


def _is_loopback(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1") or host.startswith("127.")


_CONNECT_ACCEPTS_PROXY: Final = "proxy" in inspect.signature(ws_connect.__init__).parameters


class _LevelTracker:
    """Recent input levels, used to find where speech really ended before a VAD stop.

    Keeps ``(start, end, dBFS)`` per sent frame for ``horizon`` seconds. Speech is any
    frame clearly above the recent noise floor; when the history has no contrast (noise as
    loud as speech, or too short) the server's position is kept unchanged.
    """

    def __init__(self, horizon: float = 12.0) -> None:
        self._horizon = horizon
        self._levels: deque[tuple[float, float, float]] = deque()

    def reset(self) -> None:
        self._levels.clear()

    def push(self, end: float, frame: AudioFrame) -> None:
        db = frame.dbfs()
        self._levels.append((end - frame.duration, end, db if math.isfinite(db) else -120.0))
        while self._levels and self._levels[0][1] < end - self._horizon:
            self._levels.popleft()

    def speech_end(self, reported: float, lookback: float) -> float:
        levels = [lv for lv in self._levels if lv[0] < reported]
        if len(levels) < 5:
            return reported
        dbs = np.array([db for _, _, db in levels])
        floor, peak = float(np.percentile(dbs, 10)), float(np.percentile(dbs, 95))
        if peak - floor < 10.0:
            return reported
        threshold = max(-50.0, min(floor + 12.0, peak - 12.0))
        for _, end, db in reversed(levels):
            if end < reported - lookback:
                break
            if db >= threshold:
                return min(end, reported)
        return reported


# -------------------------------------------------------------------------------- engine
@register_provider(
    "engine",
    "openai",
    description="OpenAI Realtime speech-to-speech (gpt-realtime, WebSocket GA protocol)",
    default_model="gpt-realtime-2.1",
    models=(
        "gpt-realtime-2.1",
        "gpt-realtime-2.1-mini",
        "gpt-realtime-2",
        "gpt-realtime-1.5",
        "gpt-realtime",
        "gpt-realtime-mini",
    ),
    env=("OPENAI_API_KEY",),
)
class OpenAIRealtimeEngine(S2SEngine):
    """OpenAI Realtime (and compatible servers) over WebSocket.

    Args:
        model: model id (Azure: deployment name). Default: the profile's default model.
        api_key: API key (default: the profile's environment variable).
        base_url: server base URL, e.g. ``wss://api.openai.com/v1`` (``/realtime`` is
            appended; ``http(s)`` is accepted).
        profile: compatibility profile name or :class:`RealtimeProfile` (default ``openai``).
        voice: output voice (default: the profile's; ``Agent(voice=...)`` wins).
        turn_detection: ``"semantic_vad"``/``"server_vad"``, a raw ``turn_detection`` object
            (e.g. ``{"type": "server_vad", "silence_duration_ms": 400}``), or ``None`` for
            manual turns. ``create_response``/``interrupt_response`` default to ``True``.
        input_transcription: transcription model name or config object (``None`` disables
            user transcripts). Default: the profile's.
        noise_reduction: ``"near_field"`` / ``"far_field"`` (GA servers).
        reasoning_effort: ``minimal``/``low``/``medium``/``high``/``xhigh`` (2.x models;
            xAI: ``high``/``none``).
        speed: output speech speed multiplier.
        max_output_tokens: per-response output token cap.
        temperature: sampling temperature (only profiles that accept it).
        session: extra session fields, deep-merged into every full ``session.update``.
        headers: extra handshake headers.
        query: extra URL query parameters (e.g. vLLM-Omni ``{"duplex": "1"}``).
        input_sample_rate / output_sample_rate: override the profile's PCM rates.
        connect_timeout: handshake + session configuration timeout (seconds).
        max_reconnect_attempts: connection attempts after a transient failure (0 = never
            reconnect); also the maximum number of reconnects per minute.
        reconnect_backoff: delay before the first attempt (doubles per attempt, max 10 s).
        expiry_warning: emit ``EngineStatus("expiring")`` this many seconds before the
            provider session limit.
        refine_speech_end: locate the real speech end from the sent audio levels (see the
            module docs); ``False`` reports ``audio_end_ms`` unchanged.
    """

    provider = "openai"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        profile: str | RealtimeProfile = "openai",
        voice: str | None = None,
        turn_detection: TurnDetection = _UNSET,
        input_transcription: str | Mapping[str, Any] | None = _UNSET,
        noise_reduction: Literal["near_field", "far_field"] | None = None,
        reasoning_effort: str | None = None,
        speed: float | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        session: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        query: Mapping[str, str] | None = None,
        input_sample_rate: int | None = None,
        output_sample_rate: int | None = None,
        connect_timeout: float = 10.0,
        max_reconnect_attempts: int = 3,
        reconnect_backoff: float = 0.5,
        expiry_warning: float = 60.0,
        refine_speech_end: bool = True,
    ) -> None:
        prof = get_profile(profile)
        td = prof.default_turn_detection if turn_detection is _UNSET else turn_detection
        transcription: Mapping[str, Any] | None
        if input_transcription is _UNSET:
            transcription = prof.default_transcription
        elif isinstance(input_transcription, str):
            transcription = {"model": input_transcription}
        else:
            transcription = input_transcription or None
        self.profile = prof
        self.turn_detection = _turn_detection_payload(td, prof)
        self.transcription: dict[str, Any] | None = (
            dict(transcription) if transcription is not None else None
        )
        super().__init__(
            model=model or prof.default_model or "",
            capabilities=EngineCapabilities(
                native_audio=True,
                server_turn_detection=self.turn_detection is not None,
                tool_calling=True,
                input_transcription=self.transcription is not None or prof.builtin_transcription,
                output_transcription=True,
                truncation=prof.supports_truncate,
                full_duplex=False,
                text_input=prof.text_input,
                tool_mode="blocking",
                max_session_duration=prof.max_session_duration,
            ),
            input_sample_rate=input_sample_rate or prof.input_sample_rate,
            output_sample_rate=output_sample_rate or prof.output_sample_rate,
        )
        # api_key="" means "no key" (e.g. Azure Entra tokens): no environment fallback
        self.api_key = (
            api_key
            if api_key is not None
            else next((v for v in (os.environ.get(e) for e in prof.api_key_env) if v), None)
        )
        self.headers: dict[str, str] = dict(headers or {})
        auth_names = {"authorization", prof.auth_header.lower()}
        has_auth = any(h.lower() in auth_names for h in self.headers)
        if prof.api_key_required and not self.api_key and not has_auth:
            hint = " or ".join(prof.api_key_env) or "api_key=..."
            raise ConfigurationError(f"{self.provider}: no API key; pass api_key=... or set {hint}")
        resolved_base = (
            base_url
            or next((v for v in (os.environ.get(e) for e in prof.base_url_env) if v), None)
            or prof.base_url
        )
        if not resolved_base:
            raise ConfigurationError(f"{self.provider}: base_url=... is required")
        self.url = realtime_url(resolved_base, model=self.model or None, query=query)
        self.voice = voice or prof.default_voice
        self.noise_reduction = noise_reduction
        self.reasoning_effort = reasoning_effort
        self.speed = speed
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.session_overrides: dict[str, Any] = dict(session or {})
        self.connect_timeout = connect_timeout
        self.max_reconnect_attempts = max(0, max_reconnect_attempts)
        self.reconnect_backoff = reconnect_backoff
        self.expiry_warning = expiry_warning
        self.refine_speech_end = refine_speech_end
        self._connections: set[OpenAIRealtimeConnection] = set()

    def request_headers(self) -> dict[str, str]:
        """Handshake headers (authentication + ``headers``). No ``OpenAI-Beta`` header: GA."""
        headers = dict(self.headers)
        if self.api_key:
            if self.profile.auth_header.lower() == "authorization":
                headers.setdefault("Authorization", f"Bearer {self.api_key}")
            else:
                headers.setdefault(self.profile.auth_header, self.api_key)
        return headers

    async def connect(self, options: EngineOptions) -> EngineConnection:
        conn = OpenAIRealtimeConnection(self, options)
        self._connections.add(conn)
        try:
            await conn.start()
        except BaseException:
            self._connections.discard(conn)
            raise
        return conn

    async def aclose(self) -> None:
        await asyncio.gather(*(c.aclose() for c in list(self._connections)), return_exceptions=True)


# ---------------------------------------------------------------------------- connection
@dataclass
class _Response:
    response_id: str
    trigger: float
    """``now()`` of the turn commit / ``response.create`` that triggered the response."""
    first_audio: float | None = None
    calls: set[str] = field(default_factory=set)
    done: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class _AudioItem:
    response_id: str
    content_index: int = 0
    audio_ms: float = 0.0


class OpenAIRealtimeConnection(EngineConnection):
    """A live Realtime session on one WebSocket, reconnected after transient failures.

    A reconnect starts a *new* provider session (the conversation context is not carried
    over yet); it is announced with ``EngineStatus("reconnecting"/"reconnected")`` and any
    response in flight ends with ``ResponseDone(status="failed")``.
    """

    engine: OpenAIRealtimeEngine

    def __init__(self, engine: OpenAIRealtimeEngine, options: EngineOptions) -> None:
        super().__init__(engine, options)
        self.engine = engine
        self._profile = engine.profile
        self.instructions = options.instructions
        self.tools: list[FunctionTool] = list(options.tools)
        self.voice = options.voice or engine.voice
        self.turn_detection = engine.turn_detection if options.turn_detection else None
        self.session_id: str | None = None
        self._aliases = self._profile.event_aliases
        self._ws: ClientConnection | None = None
        self._supervisor: asyncio.Task[None] | None = None
        self._tasks = BackgroundTasks(f"{engine.provider}-realtime")
        self._expiry_task: asyncio.Task[None] | None = None
        self._closing = False
        self._started = False
        self._reconnecting = False
        self._ready = asyncio.Event()
        self._startup_error: Exception | None = None
        self._fatal: Exception | None = None
        self._reconnect_times: deque[float] = deque()
        self._audio_offset = 0.0
        self._levels = _LevelTracker()
        self._responses: dict[str, _Response] = {}
        self._finished: deque[str] = deque(maxlen=64)
        self._active: str | None = None
        self._pending_trigger: float | None = None
        """When the next ``response.created`` was requested (commit / ``response.create``)."""
        self._request_event: str | None = None
        self._restore_instructions = False
        self._audio_items: OrderedDict[str, _AudioItem] = OrderedDict()
        self._transcripts: dict[str, str] = {}
        self._call_names: dict[str, str] = {}
        self._event_seq = 0
        self._sent_types: OrderedDict[str, str] = OrderedDict()
        self._handlers: dict[str, Callable[[dict[str, Any]], None]] = {
            "session.created": self._on_session_created,
            "session.updated": self._on_session_updated,
            "error": self._on_error,
            "input_audio_buffer.speech_started": self._on_speech_started,
            "input_audio_buffer.speech_stopped": self._on_speech_stopped,
            "input_audio_buffer.committed": self._on_committed,
            "conversation.item.input_audio_transcription.delta": self._on_transcript_delta,
            "conversation.item.input_audio_transcription.updated": self._on_transcript_update,
            "conversation.item.input_audio_transcription.completed": self._on_transcript_done,
            "conversation.item.input_audio_transcription.failed": self._on_transcript_failed,
            "response.created": self._on_response_created,
            "response.output_item.added": self._on_output_item_added,
            "response.output_audio.delta": self._on_audio_delta,
            "response.output_audio_transcript.delta": self._on_text_delta,
            "response.output_text.delta": self._on_text_delta,
            "response.function_call_arguments.done": self._on_function_call_done,
            "response.output_item.done": self._on_output_item_done,
            "response.done": self._on_response_done,
        }

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        """Open the socket, configure the session and wait for ``session.updated``."""
        try:
            self._ws = await self._open_socket()
            self._supervisor = asyncio.create_task(
                self._supervise(), name=f"{self.engine.provider}-realtime"
            )
            await self._configure(seed=True)
            try:
                await asyncio.wait_for(self._ready.wait(), self.engine.connect_timeout)
            except TimeoutError:
                logger.warning(
                    "%s: no session.updated within %.0fs; continuing",
                    self.engine.provider,
                    self.engine.connect_timeout,
                )
            if self._startup_error is not None:
                raise self._startup_error
            self._started = True
        except BaseException:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        if self.closed:
            return
        self._closing = True
        # stop the supervisor first: it can neither reconnect behind us nor dispatch events
        # that spawn new background tasks after they were cancelled
        if self._supervisor is not None and self._supervisor is not asyncio.current_task():
            await cancel_and_wait(self._supervisor)
        await self._tasks.cancel_all()
        ws, self._ws = self._ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()
        self.engine._connections.discard(self)
        await super().aclose()

    async def _open_socket(self) -> ClientConnection:
        engine = self.engine
        kwargs: dict[str, Any] = {}
        if _CONNECT_ACCEPTS_PROXY and _is_loopback(engine.url):
            kwargs["proxy"] = None  # never route local servers through a system proxy
        try:
            ws = await ws_connect(
                engine.url,
                additional_headers=engine.request_headers(),
                open_timeout=engine.connect_timeout,
                max_size=_MAX_MESSAGE_SIZE,
                compression=None,  # base64 audio does not compress; avoid the CPU cost
                close_timeout=2.0,
                **kwargs,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise _handshake_error(exc, engine.provider, engine.url) from exc
        # The server's audio clock (audio_start_ms/audio_end_ms) restarts with each session.
        self._audio_offset = self.input_audio_time
        self._levels.reset()
        return ws

    async def _configure(self, *, seed: bool) -> None:
        await self._send({"type": "session.update", "session": self._session_payload(full=True)})
        if seed and self.options.chat_ctx is not None:
            await self._seed_history(self.options.chat_ctx)

    async def _supervise(self) -> None:
        while True:
            ws = self._ws
            if ws is None:
                return
            reason = await self._read(ws)
            if self._closing:
                return
            if not self._started:
                if self._startup_error is None:
                    self._startup_error = ProviderConnectionError(
                        f"{self.engine.provider}: connection {reason} during setup",
                        provider=self.engine.provider,
                    )
                self._ready.set()
                return
            if not await self._reconnect(reason):
                return

    async def _read(self, ws: ClientConnection) -> str:
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue  # binary audio transport is never requested
                try:
                    event = json.loads(raw)
                except ValueError:
                    logger.warning("%s: ignoring a non-JSON message", self.engine.provider)
                    continue
                if isinstance(event, dict):
                    self._dispatch(event)
        except ConnectionClosed as exc:
            return _close_reason(exc)
        except Exception as exc:
            logger.warning("%s: realtime socket failed: %r", self.engine.provider, exc)
            return f"failed: {exc!r}"
        return "closed by the server"

    async def _reconnect(self, reason: str) -> bool:
        engine = self.engine
        self._reconnecting = True
        self._ws = None
        self._fail_responses(f"connection lost: {reason}")
        self._reset_server_state()
        last: Exception | None = self._fatal
        t = now()
        while self._reconnect_times and self._reconnect_times[0] < t - _RECONNECT_WINDOW:
            self._reconnect_times.popleft()
        attempts = engine.max_reconnect_attempts
        if last is None and attempts > 0 and len(self._reconnect_times) >= attempts:
            last = ProviderConnectionError(  # a server that keeps dropping us: stop looping
                f"{engine.provider}: realtime connection {reason} "
                f"({len(self._reconnect_times)} reconnects in {_RECONNECT_WINDOW:.0f}s)",
                provider=engine.provider,
            )
        if last is None and attempts > 0:
            logger.warning("%s: realtime connection %s; reconnecting", engine.provider, reason)
            self._emit(EngineStatus(status="reconnecting", detail=reason))
            for attempt in range(attempts):
                await asyncio.sleep(min(engine.reconnect_backoff * 2**attempt, 10.0))
                try:
                    ws = await self._open_socket()
                except (AuthenticationError, ConfigurationError) as exc:
                    last = exc
                    break
                except ProviderError as exc:
                    last = exc
                    logger.warning(
                        "%s: reconnect attempt %d failed: %s", engine.provider, attempt + 1, exc
                    )
                    continue
                if self._closing:
                    with contextlib.suppress(Exception):
                        await ws.close()
                    return False
                self._ws = ws
                self._reconnecting = False
                self._reconnect_times.append(now())
                await self._configure(seed=False)
                self._emit(
                    EngineStatus(status="reconnected", detail=f"after {attempt + 1} attempt(s)")
                )
                return True
        self._reconnecting = False
        error = last or ProviderConnectionError(
            f"{engine.provider}: realtime connection {reason}", provider=engine.provider
        )
        self._emit(EngineErrorEvent(error=error, recoverable=False))
        await self.aclose()
        return False

    def _reset_server_state(self) -> None:
        self._audio_items.clear()
        self._transcripts.clear()
        self._call_names.clear()
        self._restore_instructions = False
        self._pending_trigger = self._request_event = None
        if self._expiry_task is not None:
            self._expiry_task.cancel()
            self._expiry_task = None

    # --------------------------------------------------------------------- sending
    def _next_event_id(self, etype: str) -> str:
        self._event_seq += 1
        event_id = f"evt_{self._event_seq:06d}"
        self._sent_types[event_id] = etype  # to name the request an error refers to
        while len(self._sent_types) > _MAX_TRACKED_ITEMS:
            self._sent_types.popitem(last=False)
        return event_id

    async def _send(self, event: dict[str, Any]) -> bool:
        ws = self._ws
        if ws is None or self._reconnecting or self._closing:
            return False
        etype = event["type"]
        if etype != "input_audio_buffer.append" and "event_id" not in event:
            event = {"event_id": self._next_event_id(etype), **event}
        try:
            await ws.send(json.dumps(event, separators=(",", ":"), ensure_ascii=False))
        except ConnectionClosed:
            return False  # the supervisor reconnects
        return True

    def _session_payload(
        self,
        *,
        full: bool,
        instructions: str | None = None,
        tools: Sequence[FunctionTool] | None = None,
    ) -> dict[str, Any]:
        engine, prof = self.engine, self._profile
        if full:
            instructions, tools = self.instructions, self.tools
        session: dict[str, Any] = {"type": "realtime"} if prof.dialect == "ga" else {}
        if instructions is not None:
            session["instructions"] = instructions
        if tools is not None:
            session["tools"] = [_tool_payload(t, prof.tool_format) for t in tools]
        if not full:
            return session
        td = self.turn_detection
        transcription = self._transcription_payload()
        if prof.dialect == "beta":
            session["modalities"] = ["text", "audio"]
            if self.voice:
                session["voice"] = self.voice
            session["input_audio_format"] = prof.beta_audio_format
            session["output_audio_format"] = prof.beta_audio_format
            session["turn_detection"] = td
            if transcription is not None:
                session["input_audio_transcription"] = transcription
        else:
            audio_in: dict[str, Any] = {
                "format": {"type": "audio/pcm", "rate": self.input_sample_rate}
            }
            audio_out: dict[str, Any] = {
                "format": {"type": "audio/pcm", "rate": self.output_sample_rate}
            }
            if transcription is not None:
                audio_in["transcription"] = transcription
            if engine.speed is not None:
                audio_out["speed"] = engine.speed
            if prof.dialect == "ga":
                session["output_modalities"] = ["audio"]
                audio_in["turn_detection"] = td
                if engine.noise_reduction:
                    audio_in["noise_reduction"] = {"type": engine.noise_reduction}
                if self.voice:
                    audio_out["voice"] = self.voice
            else:  # xAI: GA audio formats, top-level voice and turn_detection
                session["turn_detection"] = td
                if self.voice:
                    session["voice"] = self.voice
            session["audio"] = {"input": audio_in, "output": audio_out}
        if engine.reasoning_effort:
            session["reasoning"] = {"effort": engine.reasoning_effort}
        if engine.max_output_tokens is not None and prof.max_output_tokens:
            session["max_output_tokens"] = engine.max_output_tokens
        temperature = (
            self.options.temperature if self.options.temperature is not None else engine.temperature
        )
        if temperature is not None and prof.temperature:
            session["temperature"] = temperature
        return _deep_merge(session, engine.session_overrides, self.options.extra)

    def _transcription_payload(self) -> dict[str, Any] | None:
        base = self.engine.transcription
        if base is None:
            return None
        cfg = dict(base)
        language = (self.options.language or "").strip().replace("_", "-")
        if language and self._profile.transcription_language:
            if self._profile.dialect == "xai":  # BCP-47 hint, regional variants matter
                cfg.setdefault("language_hint", language)
            else:  # ISO-639-1 ("en-US" -> "en")
                cfg.setdefault("language", language.split("-")[0].lower())
        return cfg

    async def _seed_history(self, ctx: ChatContext) -> None:
        if not self._profile.text_input:
            if ctx.items:
                logger.warning("%s: cannot seed chat history (no text items)", self.engine.provider)
            return
        for item in ctx.items:
            payload: dict[str, Any]
            if isinstance(item, ChatMessage):
                text = item.text
                if not text.strip():
                    continue
                role = "system" if item.role in ("system", "developer") else item.role
                if role == "assistant":
                    part = "text" if self._profile.dialect == "beta" else "output_text"
                else:
                    part = "input_text"
                payload = {
                    "type": "message",
                    "role": role,
                    "content": [{"type": part, "text": text}],
                }
            elif isinstance(item, FunctionCall):
                payload = {
                    "type": "function_call",
                    "call_id": item.call_id,
                    "name": item.name,
                    "arguments": item.arguments,
                }
            else:
                payload = {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": item.output,
                }
            await self._send({"type": "conversation.item.create", "item": payload})

    # -------------------------------------------------------------------- audio in
    async def _send_audio(self, frame: AudioFrame) -> None:
        if self.engine.refine_speech_end and self.turn_detection is not None:
            self._levels.push(self.input_audio_time, frame)
        await self._send({"type": "input_audio_buffer.append", "audio": frame.to_base64()})

    async def commit_input(self) -> None:
        await self._send({"type": "input_audio_buffer.commit"})
        await self._create_response()

    async def clear_input(self) -> None:
        await self._send({"type": "input_audio_buffer.clear"})

    # --------------------------------------------------------------------- control
    async def send_text(self, text: str, *, respond: bool = True) -> None:
        if not self._profile.text_input:
            raise EngineError(f"the {self._profile.name} realtime backend accepts no text messages")
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
        if respond:
            await self._create_response()

    async def create_response(self, *, instructions: str | None = None) -> None:
        if instructions and self.instructions:
            instructions = f"{self.instructions}\n\n{instructions}"
        await self._create_response(instructions)

    async def say(self, text: str) -> None:
        if self._profile.say_mode == "force_message":
            trigger = now()
            await self._cancel_active_and_wait()
            item = {
                "type": "force_message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
            event = {"type": "conversation.item.create", "item": item}
            await self._request_response(event, trigger)
            return
        await self._create_response(
            VERBATIM_INSTRUCTIONS.format(text=text), isolated=self._profile.isolated_say
        )

    async def _create_response(
        self, instructions: str | None = None, *, isolated: bool = False
    ) -> None:
        trigger = now()  # the caller's view: waiting for a cancellation counts toward the TTFB
        await self._cancel_active_and_wait()
        body: dict[str, Any] = {}
        if instructions:
            if self._profile.response_instructions:
                body["instructions"] = instructions
            else:  # patch the session prompt for one response; restored at response.created
                patch = self._session_payload(full=False, instructions=instructions)
                await self._send({"type": "session.update", "session": patch})
                self._restore_instructions = True
        if isolated:
            body["input"] = []
        event: dict[str, Any] = {"type": "response.create"}
        if body:
            event["response"] = body
        await self._request_response(event, trigger)

    async def _request_response(self, event: dict[str, Any], trigger: float) -> None:
        """Send an event that makes the server start a response, timed from ``trigger``."""
        event_id = self._next_event_id(event["type"])
        self._request_event = event_id
        self._pending_trigger = trigger
        if not await self._send({"event_id": event_id, **event}):
            self._request_event = self._pending_trigger = None
            self._restore_instructions = False  # a reconnect re-sends the full configuration

    async def cancel_response(self) -> None:
        if self._active is not None and self._profile.supports_cancel:
            await self._send_cancel(self._active)

    async def _send_cancel(self, response_id: str) -> None:
        event: dict[str, Any] = {"type": "response.cancel"}
        if self._profile.dialect != "beta":
            event["response_id"] = response_id
        await self._send(event)

    async def _cancel_active_and_wait(self) -> None:
        """Servers reject ``response.create`` while a response is in progress."""
        state = self._responses.get(self._active) if self._active is not None else None
        if state is None:
            return
        if self._profile.supports_cancel:
            await self._send_cancel(state.response_id)
            timeout = _CANCEL_TIMEOUT
        else:
            timeout = _FINISH_TIMEOUT  # cannot be cancelled: let it finish
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(state.done.wait(), timeout)

    async def truncate(self, item_id: str, audio_end_ms: int) -> str | None:
        """Truncate the assistant audio to what was played (``content_index`` tracked).

        ``audio_end_ms`` counts from the start of ``item_id``; when a response has several
        audio items (e.g. a preamble and the answer) the cut lands in the right one and
        later items are truncated to zero. The value is clamped to the audio received, so
        the server never rejects it. Returns ``None``: the server does not report the
        heard transcript (the session estimates it).
        """
        if not self._profile.supports_truncate:
            return None
        for target, content_index, end_ms in self._truncation_plan(item_id, audio_end_ms):
            await self._send(
                {
                    "type": "conversation.item.truncate",
                    "item_id": target,
                    "content_index": content_index,
                    "audio_end_ms": end_ms,
                }
            )
        return None

    def _truncation_plan(self, item_id: str, audio_end_ms: int) -> list[tuple[str, int, int]]:
        first = self._audio_items.get(item_id)
        if first is None:  # no audio received: nothing the server could truncate
            return []
        items = [
            iid for iid, it in self._audio_items.items() if it.response_id == first.response_id
        ]
        remaining = max(0, int(audio_end_ms))
        plan: list[tuple[str, int, int]] = []
        for iid in items[items.index(item_id) :]:
            info = self._audio_items[iid]
            total = math.floor(info.audio_ms)
            heard = min(remaining, total)
            remaining -= heard
            if heard < total:
                plan.append((iid, info.content_index, heard))
        return plan

    async def send_tool_output(self, output: FunctionCallOutput, *, respond: bool = True) -> None:
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": output.call_id,
                    "output": output.output,
                },
            }
        )
        if respond:
            await self._create_response()

    async def update(
        self, *, instructions: str | None = None, tools: list[FunctionTool] | None = None
    ) -> None:
        if instructions is not None:
            self.instructions = instructions
        if tools is not None:
            self.tools = list(tools)
        if instructions is None and tools is None:
            return
        patch = self._session_payload(full=False, instructions=instructions, tools=tools)
        await self._send({"type": "session.update", "session": patch})

    # ---------------------------------------------------------------------- events
    def _dispatch(self, event: dict[str, Any]) -> None:
        etype = event.get("type")
        if not isinstance(etype, str):
            return
        handler = self._handlers.get(self._aliases.get(etype, etype))
        if handler is None:
            return
        try:
            handler(event)
        except Exception:
            logger.exception("%s: failed to handle %s", self.engine.provider, etype)

    def _on_session_created(self, ev: dict[str, Any]) -> None:
        session = ev.get("session") or {}
        self.session_id = session.get("id")
        expires_at = session.get("expires_at")
        time_left: float | None = self._profile.max_session_duration
        if isinstance(expires_at, (int, float)) and expires_at > 0:
            # expires_at is a Unix timestamp, so the wall clock is the right reference here
            time_left = max(0.0, float(expires_at) - time.time())
        if self._expiry_task is not None:
            self._expiry_task.cancel()
            self._expiry_task = None
        if time_left is not None:
            self._expiry_task = self._tasks.spawn(self._expiry_notice(time_left))

    async def _expiry_notice(self, time_left: float) -> None:
        deadline = now() + time_left
        await asyncio.sleep(max(0.0, time_left - self.engine.expiry_warning))
        self._emit(
            EngineStatus(
                status="expiring",
                detail="provider session duration limit",
                time_left=max(0.0, deadline - now()),
            )
        )

    def _on_session_updated(self, ev: dict[str, Any]) -> None:
        self._ready.set()

    def _on_error(self, ev: dict[str, Any]) -> None:
        err = ev.get("error")
        if not isinstance(err, Mapping):
            err = {"message": str(err or ev)}
        code = err.get("code")
        source = self._sent_types.get(str(err.get("event_id") or ""))
        provider = self.engine.provider
        if err.get("event_id") and err.get("event_id") == self._request_event:
            # our response request was rejected: nothing will consume its trigger or patch
            self._request_event = None
            self._pending_trigger = None
            if self._restore_instructions:
                self._restore_session_instructions()
        if code in _IGNORED_ERROR_CODES:
            logger.debug("%s: ignoring %s (%s)", provider, code, err.get("message"))
            return
        if code == "conversation_already_has_active_response":
            logger.warning("%s: %s; keeping the response in progress", provider, code)
            return
        if code in _EXPIRY_CODES or err.get("type") in _EXPIRY_CODES:
            logger.info("%s: provider session expired; reconnecting", provider)
            ws = self._ws
            if ws is not None:
                self._tasks.spawn(ws.close())
            return
        exc = _server_error(err, provider, source)
        fatal = isinstance(exc, AuthenticationError)
        if fatal:
            self._fatal = exc
        if not self._started and not self._ready.is_set():  # rejected session configuration
            self._startup_error = exc
            self._ready.set()
            return
        self._emit(EngineErrorEvent(error=exc, recoverable=not fatal))

    def _on_speech_started(self, ev: dict[str, Any]) -> None:
        ms = ev.get("audio_start_ms")
        audio_time = self._audio_offset + ms / 1000.0 if isinstance(ms, (int, float)) else None
        self._emit(InputSpeechStarted(audio_time=audio_time))

    def _on_speech_stopped(self, ev: dict[str, Any]) -> None:
        ms = ev.get("audio_end_ms")
        audio_time: float | None = None
        if isinstance(ms, (int, float)):
            audio_time = self._audio_offset + ms / 1000.0
            if self.engine.refine_speech_end:
                audio_time = self._levels.speech_end(audio_time, self._speech_hold())
        self._emit(InputSpeechStopped(audio_time=audio_time))

    def _speech_hold(self) -> float:
        """Upper bound on the silence the server waits before ``speech_stopped``."""
        td = self.turn_detection or {}
        if td.get("type") == "semantic_vad":
            return 8.3  # low eagerness waits up to 8 s
        silence = td.get("silence_duration_ms")
        return (silence if isinstance(silence, (int, float)) else 500) / 1000.0 + 0.3

    def _on_committed(self, ev: dict[str, Any]) -> None:
        item_id = ev.get("item_id") or new_id("item_")
        td = self.turn_detection
        if not self._trigger_pending() and td is not None and td.get("create_response", True):
            self._pending_trigger = now()  # the server responds to this commit by itself
        self._emit(InputCommitted(item_id=item_id))

    def _on_transcript_delta(self, ev: dict[str, Any]) -> None:
        item_id = str(ev.get("item_id") or "")
        delta = ev.get("delta")
        if isinstance(delta, str):
            text = self._transcripts.get(item_id, "") + delta
        else:  # Qwen: cumulative confirmed ``text`` + tentative ``stash``
            text = str(ev.get("text") or "") + str(ev.get("stash") or "")
        self._partial_transcript(item_id, text, ev.get("language"))

    def _on_transcript_update(self, ev: dict[str, Any]) -> None:
        # xAI: cumulative transcript that may revise earlier updates
        self._partial_transcript(
            str(ev.get("item_id") or ""), str(ev.get("transcript") or ""), None
        )

    def _partial_transcript(self, item_id: str, text: str, language: Any) -> None:
        self._transcripts[item_id] = text
        _bound(self._transcripts)
        if text.strip():
            self._emit(
                InputTranscript(
                    item_id=item_id,
                    text=text,
                    is_final=False,
                    language=language if isinstance(language, str) else None,
                )
            )

    def _on_transcript_done(self, ev: dict[str, Any]) -> None:
        item_id = str(ev.get("item_id") or "")
        self._transcripts.pop(item_id, None)
        language = ev.get("language")
        languages = ev.get("languages")
        if not isinstance(language, str) and isinstance(languages, list) and languages:
            first = languages[0]
            language = first.get("code") if isinstance(first, Mapping) else None
        self._emit(
            InputTranscript(
                item_id=item_id,
                text=str(ev.get("transcript") or "").strip(),
                is_final=True,
                language=language if isinstance(language, str) else None,
            )
        )

    def _on_transcript_failed(self, ev: dict[str, Any]) -> None:
        item_id = str(ev.get("item_id") or "")
        self._transcripts.pop(item_id, None)
        err = ev.get("error") or {}
        message = err.get("message") if isinstance(err, Mapping) else str(err)
        provider = self.engine.provider
        self._emit(
            EngineErrorEvent(
                error=ProviderError(
                    f"{provider}: input transcription failed for {item_id}: {message}",
                    provider=provider,
                ),
                recoverable=True,
            )
        )

    def _on_response_created(self, ev: dict[str, Any]) -> None:
        response = ev.get("response") or {}
        self._response(str(response.get("id") or ev.get("response_id") or new_id("resp_")))
        if self._restore_instructions:
            self._restore_session_instructions()

    def _restore_session_instructions(self) -> None:
        self._restore_instructions = False
        patch = self._session_payload(full=False, instructions=self.instructions)
        self._tasks.spawn(self._send({"type": "session.update", "session": patch}))

    def _trigger_pending(self) -> bool:
        trigger = self._pending_trigger
        return trigger is not None and now() - trigger < _TRIGGER_TTL

    def _response(self, response_id: str) -> _Response:
        """State of ``response_id``, starting it if the server skipped ``response.created``."""
        state = self._responses.get(response_id)
        if state is None:
            trigger = self._pending_trigger if self._trigger_pending() else None
            self._pending_trigger = self._request_event = None
            state = _Response(response_id, now() if trigger is None else trigger)
            self._responses[response_id] = state
            self._active = response_id
            self._emit(ResponseStarted(response_id=response_id))
        return state

    def _response_for(self, ev: dict[str, Any]) -> _Response | None:
        rid = ev.get("response_id") or self._active
        if not isinstance(rid, str) or rid in self._finished:
            return None  # late event of a finished response
        return self._response(rid)

    def _on_output_item_added(self, ev: dict[str, Any]) -> None:
        item = ev.get("item") or {}
        if item.get("type") == "function_call" and item.get("id") and item.get("name"):
            self._call_names[str(item["id"])] = str(item["name"])
            _bound(self._call_names)

    def _on_audio_delta(self, ev: dict[str, Any]) -> None:
        payload = ev.get("delta")
        state = self._response_for(ev)
        if not isinstance(payload, str) or not payload or state is None:
            return
        data = base64.b64decode(payload)
        if len(data) % 2:
            data = data[:-1]
        if not data:
            return
        frame = AudioFrame(data, self.output_sample_rate)
        if state.first_audio is None:
            state.first_audio = now()
        item_id = str(ev.get("item_id") or state.response_id)
        info = self._audio_items.get(item_id)
        if info is None:
            content_index = ev.get("content_index")
            info = self._audio_items[item_id] = _AudioItem(
                state.response_id, content_index if isinstance(content_index, int) else 0
            )
            while len(self._audio_items) > _MAX_TRACKED_ITEMS:
                self._audio_items.popitem(last=False)
        info.audio_ms += frame.duration * 1000.0
        self._emit(ResponseAudio(response_id=state.response_id, item_id=item_id, frame=frame))

    def _on_text_delta(self, ev: dict[str, Any]) -> None:
        delta = ev.get("delta")
        state = self._response_for(ev)
        if not isinstance(delta, str) or not delta or state is None:
            return
        item_id = str(ev.get("item_id") or state.response_id)
        self._emit(ResponseText(response_id=state.response_id, item_id=item_id, delta=delta))

    def _on_function_call_done(self, ev: dict[str, Any]) -> None:
        item_id = str(ev.get("item_id") or "")
        name = ev.get("name") or self._call_names.get(item_id)
        state = self._response_for(ev)
        if name and state is not None:  # otherwise response.output_item.done completes it
            item = {"id": item_id, "call_id": ev.get("call_id"), "name": name}
            self._emit_tool_call(state, {**item, "arguments": ev.get("arguments")})

    def _on_output_item_done(self, ev: dict[str, Any]) -> None:
        item = ev.get("item") or {}
        state = self._response_for(ev)
        if item.get("type") == "function_call" and state is not None:
            self._emit_tool_call(state, item)

    def _emit_tool_call(self, state: _Response, item: Mapping[str, Any]) -> None:
        call_id, name = item.get("call_id"), item.get("name")
        if not isinstance(call_id, str) or not isinstance(name, str) or call_id in state.calls:
            return
        state.calls.add(call_id)
        self._call_names.pop(str(item.get("id") or ""), None)
        arguments = item.get("arguments")
        call = FunctionCall(
            name=name,
            arguments=arguments if isinstance(arguments, str) and arguments.strip() else "{}",
            call_id=call_id,
        )
        if item.get("id"):
            call.id = str(item["id"])
        self._emit(ResponseToolCall(response_id=state.response_id, call=call))

    def _on_response_done(self, ev: dict[str, Any]) -> None:
        response = ev.get("response") or {}
        rid = response.get("id") or ev.get("response_id") or self._active
        if not isinstance(rid, str) or rid in self._finished:
            return
        state = self._response(rid)
        for item in response.get("output") or ():
            if isinstance(item, Mapping) and item.get("type") == "function_call":
                self._emit_tool_call(state, item)  # servers that skip output_item.done
        status = _STATUSES.get(str(response.get("status") or "completed"), "failed")
        error: str | None = None
        if status in ("failed", "incomplete"):
            details = response.get("status_details") or {}
            detail_error = details.get("error") if isinstance(details, Mapping) else None
            if isinstance(detail_error, Mapping):
                error = str(detail_error.get("message") or detail_error.get("code") or status)
            elif isinstance(details, Mapping) and details.get("reason"):
                error = str(details["reason"])
        if status == "failed":
            provider = self.engine.provider
            message = f"{provider}: response {rid} failed: {error or 'no details'}"
            self._emit(
                EngineErrorEvent(error=ProviderError(message, provider=provider), recoverable=True)
            )
        self._finish_response(state, status, _parse_usage(response.get("usage")), error)

    def _finish_response(
        self, state: _Response, status: ResponseStatus, usage: EngineUsage | None, error: str | None
    ) -> None:
        self._responses.pop(state.response_id, None)
        self._finished.append(state.response_id)
        if self._active == state.response_id:
            self._active = None
        self._emit(
            ResponseDone(response_id=state.response_id, status=status, usage=usage, error=error)
        )
        engine, u, t = self.engine, usage or EngineUsage(), now()
        engine.emit(
            "metrics",
            EngineMetrics(
                provider=engine.provider,
                model=engine.model,
                response_id=state.response_id,
                ttfb=None if state.first_audio is None else state.first_audio - state.trigger,
                duration=t - state.trigger,
                input_text_tokens=u.input_text_tokens,
                input_audio_tokens=u.input_audio_tokens,
                output_text_tokens=u.output_text_tokens,
                output_audio_tokens=u.output_audio_tokens,
                cached_tokens=u.cached_tokens,
                cancelled=status == "cancelled",
            ),
        )
        state.done.set()

    def _fail_responses(self, reason: str) -> None:
        for state in list(self._responses.values()):
            self._finish_response(state, "failed", None, reason)
