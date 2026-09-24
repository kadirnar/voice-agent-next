"""Gemini Live speech-to-speech engine (``BidiGenerateContent`` over a raw WebSocket).

``AgentSession("google/gemini-3.8-live")`` (alias ``"gemini/..."``) streams 16 kHz PCM to
the `Gemini Live API <https://ai.google.dev/api/live>`_ and plays its native 24 kHz audio.

Server messages map onto :mod:`voice_agent_next.events` as follows:

* ``voiceActivity`` start/end -> ``InputSpeechStarted`` / ``InputSpeechStopped``;
* ``serverContent.(interim)InputTranscription`` -> ``InputTranscript`` (partial, then final);
* the first model output of a user turn -> ``InputCommitted`` + ``ResponseStarted`` (Gemini
  commits user turns implicitly, with its own server-side VAD);
* ``serverContent.modelTurn`` audio -> ``ResponseAudio``; ``outputTranscription`` ->
  ``ResponseText``;
* ``serverContent.interrupted`` -> server-side barge-in: ``InputSpeechStarted`` +
  ``ResponseDone(status="cancelled")``;
* ``generationComplete`` / ``turnComplete`` -> ``ResponseDone``;
* ``toolCall`` -> ``ResponseToolCall`` (answered with ``toolResponse``);
  ``toolCallCancellation`` -> ``ToolCallCancelled``;
* ``goAway`` and reconnects -> ``EngineStatus`` (``expiring`` / ``reconnecting`` /
  ``resumed`` / ``reconnected``).

**Session rotation.** A Live API connection lasts about ten minutes. The engine keeps the
latest ``sessionResumptionUpdate`` handle and moves the conversation to a new connection
when the server sends ``goAway``, proactively after ``rotate_after`` seconds, after
:meth:`GeminiLiveConnection.update`, or when the connection drops. Planned rotations wait
for an idle moment (no generation, no pending tool call, user silent, resumable handle)
and are forced shortly before the ``goAway`` deadline. The switch is make-before-break:
user audio is buffered while the new connection is set up, and the audio sent since the
last handle (plus a small safety margin) is replayed, so no user audio is lost.

**Why raw websockets instead of google-genai:** the engine needs full control over the
connection lifecycle (make-before-break rotation, replaying buffered audio into the new
connection, the raw ``voiceActivity`` / ``interimInputTranscription`` fields) and must not
add a dependency to the core install; ``websockets`` is already a core dependency.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import functools
import inspect
import json
import os
import re
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias
from urllib.parse import urlsplit

from ...audio.frame import AudioFrame
from ...audio.resample import StreamResampler
from ...chat import ChatContext, ChatMessage, FunctionCall, FunctionCallOutput
from ...engine import EngineCapabilities, EngineConnection, EngineOptions, S2SEngine
from ...errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from ...events import (
    EngineErrorEvent,
    EngineStatus,
    EngineStatusKind,
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
    ToolCallCancelled,
)
from ...metrics import EngineMetrics
from ...registry import register_provider
from ...tools import FunctionTool
from ...utils.aio import BackgroundTasks, cancel_and_wait
from ...utils.clock import now
from ...utils.ids import new_id
from ...utils.log import logger
from ...vad import VADEventType, VADOptions, VADStream
from ..energy import EnergyVAD

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

__all__ = [
    "CONNECTION_LIMIT",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "GeminiLiveConnection",
    "GeminiLiveEngine",
]

DEFAULT_MODEL = "gemini-3.8-live"
DEFAULT_BASE_URL = "wss://generativelanguage.googleapis.com"
KNOWN_MODELS = (
    "gemini-3.8-live",
    "gemini-3.8-live-extended-thinking",
    "gemini-3.1-flash-live-preview",
    "gemini-2.5-flash-native-audio-preview-12-2025",
)
API_KEY_ENV = ("GOOGLE_API_KEY", "GEMINI_API_KEY")
INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
CONNECTION_LIMIT = 600.0
"""Approximate lifetime of one Live API WebSocket connection (seconds)."""

ToolBehavior: TypeAlias = Literal["non_blocking", "blocking"]
ActivityHandling: TypeAlias = Literal["start_of_activity_interrupts", "no_interruption"]

_MAX_MESSAGE = 16 * 2**20
_MONITOR_INTERVAL = 0.1
_REQUEST_TTL = 30.0
"""A client-requested response must start within this time to be recognized as such."""
_TEXT_HOLD = 1.0
"""Max time agent transcript deltas wait for the user's (late) transcript at turn start."""
_LATE_TRANSCRIPT_GRACE = 1.0
"""Input transcription arriving this soon after the answer still belongs to its turn."""
_MAX_SEED_TURNS = 100


# ------------------------------------------------------------------------------ helpers
def _parse_duration(value: Any) -> float | None:
    """protobuf ``Duration`` JSON (``"1.5s"`` or ``{"seconds": 1, "nanos": 5e8}``) -> seconds."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Mapping):
        return float(value.get("seconds") or 0) + float(value.get("nanos") or 0) / 1e9
    text = str(value).strip().removesuffix("s")
    try:
        return float(text)
    except ValueError:
        return None


def _audio_rate(mime: str, default: int) -> int:
    match = re.search(r"rate=(\d+)", mime)
    return int(match.group(1)) if match else default


def _tool_result(text: str) -> Any:
    stripped = text.strip()
    if stripped[:1] in ("{", "["):
        with contextlib.suppress(ValueError):
            return json.loads(stripped)
    return text


def _camelize(value: Any) -> Any:
    """``{"silence_duration_ms": 500}`` -> ``{"silenceDurationMs": 500}`` (recursively)."""
    if isinstance(value, Mapping):
        return {
            re.sub(r"_([a-z0-9])", lambda m: m.group(1).upper(), str(k)): _camelize(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_camelize(v) for v in value]
    return value


def _deep_merge(base: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        current = out.get(key)
        if isinstance(value, Mapping) and isinstance(current, dict):
            out[key] = _deep_merge(current, value)
        else:
            out[key] = value
    return out


def _decode(raw: str | bytes) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except ValueError:
        logger.warning("gemini-live: ignoring a non-JSON server message")
        return None
    return value if isinstance(value, dict) else None


def _usage(u: Mapping[str, Any]) -> EngineUsage:
    """``usageMetadata`` -> :class:`EngineUsage` (non-audio modalities count as text)."""

    def by_modality(key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for detail in u.get(key) or ():
            if isinstance(detail, Mapping):
                modality = str(detail.get("modality") or "TEXT").upper()
                out[modality] = out.get(modality, 0) + int(detail.get("tokenCount") or 0)
        return out

    usage = EngineUsage(cached_tokens=int(u.get("cachedContentTokenCount") or 0))
    prompt = by_modality("promptTokensDetails")
    if prompt:
        usage.input_audio_tokens = prompt.pop("AUDIO", 0)
        usage.input_text_tokens = sum(prompt.values())
    else:
        usage.input_text_tokens = int(u.get("promptTokenCount") or 0)
    response = by_modality("responseTokensDetails")
    if response:
        usage.output_audio_tokens = response.pop("AUDIO", 0)
        usage.output_text_tokens = sum(response.values())
    else:
        usage.output_text_tokens = int(u.get("responseTokenCount") or 0)
    return usage


def _add_usage(a: EngineUsage, b: EngineUsage) -> EngineUsage:
    return EngineUsage(
        input_text_tokens=a.input_text_tokens + b.input_text_tokens,
        input_audio_tokens=a.input_audio_tokens + b.input_audio_tokens,
        output_text_tokens=a.output_text_tokens + b.output_text_tokens,
        output_audio_tokens=a.output_audio_tokens + b.output_audio_tokens,
        cached_tokens=a.cached_tokens + b.cached_tokens,
    )


_AUTH_HINTS = ("api key", "api_key", "permission", "unauthenticated", "unauthorized", "credential")
_RATE_HINTS = ("quota", "resource_exhausted", "resource exhausted", "rate limit", "too many")


def _close_error(code: int | None, reason: str) -> ProviderError:
    """Map a server-initiated WebSocket close to a library error."""
    text = reason.lower()
    msg = f"Gemini Live closed the connection ({code}): {reason or 'no reason given'}"
    if any(h in text for h in _AUTH_HINTS):
        return AuthenticationError(msg, provider="google", status_code=code)
    if code == 1013 or any(h in text for h in _RATE_HINTS):
        return RateLimitError(msg, provider="google", status_code=code)
    if code in (1002, 1003, 1007, 1008, 1009, 1010):
        return ProviderError(msg, provider="google", status_code=code)
    return ProviderConnectionError(msg, provider="google", status_code=code)


def _http_error(status: int, body: str) -> ProviderError:
    """Map a rejected WebSocket handshake to a library error."""
    msg = f"Gemini Live rejected the connection (HTTP {status}): {body.strip()[:300]}"
    if status in (401, 403):
        return AuthenticationError(msg, provider="google", status_code=status)
    if status == 429:
        return RateLimitError(msg, provider="google", status_code=status)
    if status >= 500:
        return ProviderConnectionError(msg, provider="google", status_code=status)
    return ProviderError(msg, provider="google", status_code=status)


def _close_info(exc: Exception) -> tuple[int | None, str]:
    rcvd = getattr(exc, "rcvd", None)
    if rcvd is None:
        return None, ""
    return int(rcvd.code), str(rcvd.reason or "")


async def _close_quietly(ws: ClientConnection, timeout: float = 2.0) -> None:
    with contextlib.suppress(Exception):
        await asyncio.wait_for(ws.close(), timeout)


@functools.cache
def _connect_accepts_proxy() -> bool:
    from websockets.asyncio.client import connect

    return "proxy" in inspect.signature(connect.__init__).parameters


# ------------------------------------------------------------------------------- engine
@register_provider(
    "engine",
    "google",
    description="Gemini Live API native speech-to-speech (BidiGenerateContent WebSocket)",
    default_model=DEFAULT_MODEL,
    models=KNOWN_MODELS,
    env=API_KEY_ENV,
    extra=None,  # raw WebSocket protocol: `websockets` is a core dependency
    requires=("websockets",),
    local=False,
    aliases=("gemini",),
)
class GeminiLiveEngine(S2SEngine):
    """Gemini Live API engine (``gemini-3.8-live`` by default).

    Args:
        model: Live model id, e.g. ``"gemini-3.8-live"``.
        api_key: Gemini API key (default: ``GOOGLE_API_KEY`` then ``GEMINI_API_KEY``).
            Ephemeral tokens (``"auth_tokens/..."``) use the constrained endpoint.
        voice: prebuilt voice name (``"Kore"``, ``"Puck"``, ...); ``EngineOptions.voice`` wins.
        temperature: sampling temperature.
        base_url: WebSocket origin (proxies, tests). A URL containing ``/ws/`` is used as is.
        api_version: API version of the endpoint path.
        input_transcription / output_transcription: request transcripts of the user's and
            the model's audio (``InputTranscript`` / ``ResponseText``).
        vad: ``realtimeInputConfig.automaticActivityDetection`` settings, e.g.
            ``{"silence_duration_ms": 500, "end_of_speech_sensitivity": "END_SENSITIVITY_LOW"}``.
        activity_handling: ``"start_of_activity_interrupts"`` (server default) or
            ``"no_interruption"`` (then also use ``SessionOptions(allow_interruptions=False)``).
        turn_coverage: ``realtimeInputConfig.turnCoverage`` enum value.
        tool_behavior: ``"non_blocking"`` (model keeps talking while tools run) or
            ``"blocking"``. Default: blocking for ``gemini-3.1-flash-live*`` (no async
            function calling), non-blocking otherwise.
        thinking_level: ``generationConfig.thinkingConfig.thinkingLevel`` (extended-thinking
            models only).
        session_resumption: keep resumption handles and resume on rotation/reconnect.
        context_window_compression: ``True`` for a default sliding window, a mapping for
            explicit ``contextWindowCompression`` settings, ``False`` to disable.
        rotate_after: proactively move to a new connection at the first idle moment after
            this many seconds (``None`` = only on ``goAway``/errors).
        go_away_margin: force the rotation this long before the ``goAway`` deadline.
        resume_replay: seconds of audio sent *before* the latest resumption handle that are
            replayed into a resumed connection (covers audio in flight).
        max_buffered_audio: cap on user audio buffered during a switch (seconds).
        connect_timeout: WebSocket handshake + setup timeout.
        max_reconnect_attempts: consecutive failed reconnects before giving up.
        local_vad: run a cheap energy VAD on the sent audio to estimate where speech ended
            (voice-to-voice metrics) and to rotate only while the user is silent.
        extra_setup: extra ``setup`` fields (API camelCase), deep-merged last.
    """

    provider = "google"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        voice: str | None = None,
        temperature: float | None = None,
        base_url: str | None = None,
        api_version: str = "v1beta",
        input_transcription: bool = True,
        output_transcription: bool = True,
        vad: Mapping[str, Any] | None = None,
        activity_handling: ActivityHandling | None = None,
        turn_coverage: str | None = None,
        tool_behavior: ToolBehavior | None = None,
        thinking_level: str | None = None,
        session_resumption: bool = True,
        context_window_compression: bool | Mapping[str, Any] = True,
        rotate_after: float | None = 540.0,
        go_away_margin: float = 2.0,
        resume_replay: float = 0.5,
        max_buffered_audio: float = 30.0,
        connect_timeout: float = 15.0,
        max_reconnect_attempts: int = 5,
        local_vad: bool = True,
        extra_setup: Mapping[str, Any] | None = None,
    ) -> None:
        model = model or DEFAULT_MODEL
        if tool_behavior is None:
            tool_behavior = "blocking" if "3.1-flash-live" in model else "non_blocking"
        if tool_behavior not in ("non_blocking", "blocking"):
            raise ConfigurationError(f"unknown tool_behavior {tool_behavior!r}")
        if activity_handling not in (None, "start_of_activity_interrupts", "no_interruption"):
            raise ConfigurationError(f"unknown activity_handling {activity_handling!r}")
        if rotate_after is not None and rotate_after <= 0:
            raise ConfigurationError("rotate_after must be > 0 (or None)")
        if max_reconnect_attempts < 1:
            raise ConfigurationError("max_reconnect_attempts must be >= 1")
        super().__init__(
            model=model,
            capabilities=EngineCapabilities(
                native_audio=True,
                server_turn_detection=True,
                tool_calling=True,
                input_transcription=input_transcription,
                output_transcription=output_transcription,
                truncation=False,  # the Live API cannot truncate what the model said
                full_duplex=False,
                text_input=True,
                tool_mode=tool_behavior,
                max_session_duration=CONNECTION_LIMIT,
            ),
            input_sample_rate=INPUT_SAMPLE_RATE,
            output_sample_rate=OUTPUT_SAMPLE_RATE,
        )
        self._api_key = api_key
        self.voice = voice
        self.temperature = temperature
        self.base_url = base_url or DEFAULT_BASE_URL
        self.api_version = api_version
        self.input_transcription = input_transcription
        self.output_transcription = output_transcription
        self.vad = dict(vad or {})
        self.activity_handling = activity_handling
        self.turn_coverage = turn_coverage
        self.tool_behavior: ToolBehavior = tool_behavior
        self.thinking_level = thinking_level
        self.session_resumption = session_resumption
        self.context_window_compression = context_window_compression
        self.rotate_after = rotate_after
        self.go_away_margin = go_away_margin
        self.resume_replay = resume_replay
        self.max_buffered_audio = max_buffered_audio
        self.connect_timeout = connect_timeout
        self.max_reconnect_attempts = max_reconnect_attempts
        self.local_vad = local_vad
        self.extra_setup = dict(extra_setup or {})

    def _resolve_api_key(self) -> str:
        key = self._api_key or next((os.environ[e] for e in API_KEY_ENV if os.environ.get(e)), "")
        if not key:
            raise ConfigurationError(
                "Gemini Live needs an API key: pass api_key=... or set GOOGLE_API_KEY "
                "(or GEMINI_API_KEY)"
            )
        return key

    def _endpoint(self) -> tuple[str, dict[str, str]]:
        """WebSocket URL and auth headers (the key travels in a header, never in the URL)."""
        key = self._resolve_api_key()
        ephemeral = key.startswith("auth_tokens/")
        method = "BidiGenerateContentConstrained" if ephemeral else "BidiGenerateContent"
        base = self.base_url.rstrip("/")
        service = f"google.ai.generativelanguage.{self.api_version}.GenerativeService"
        url = base if "/ws/" in base else f"{base}/ws/{service}.{method}"
        headers = {"Authorization": f"Token {key}"} if ephemeral else {"x-goog-api-key": key}
        return url, headers

    def _connect_kwargs(self, url: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        host = urlsplit(url).hostname or ""
        if host in ("127.0.0.1", "localhost", "::1") and _connect_accepts_proxy():
            kwargs["proxy"] = None  # never route local (test) servers through a system proxy
        return kwargs

    async def connect(self, options: EngineOptions) -> EngineConnection:
        conn = GeminiLiveConnection(self, options)
        try:
            await conn._start()
        except BaseException:
            await conn.aclose()
            raise
        return conn


# --------------------------------------------------------------------------- connection
@dataclass(slots=True)
class _Outgoing:
    """A serialized client message."""

    data: str
    replayable: bool
    """Re-sent to a resumed connection if it was sent after the latest handle."""
    audio_start: float | None = None
    """Input-stream position of the first sample (audio messages only)."""
    audio_duration: float = 0.0
    sent_at: float = 0.0


@dataclass
class _UserItem:
    """One user utterance as the engine sees it."""

    item_id: str = field(default_factory=lambda: new_id("item_"))
    final: str = ""
    interim: str = ""
    language: str | None = None
    evidence: bool = False
    """The server signalled user activity for it (VAD, barge-in or transcription)."""
    committed: bool = False
    stopped: bool = False
    speech_end: float | None = None
    emitted: str | None = None
    """Last text emitted as the final transcript."""
    closed_at: float | None = None
    """When the server turn answering it completed."""

    @property
    def text(self) -> str:
        return (self.final + self.interim).strip()


@dataclass
class _Generation:
    """One engine response (``response_id``) within a server turn."""

    response_id: str
    item_id: str
    started_at: float
    trigger_at: float
    first_audio_at: float | None = None
    text: list[str] = field(default_factory=list)
    calls: list[FunctionCall] = field(default_factory=list)
    status: ResponseStatus | None = None
    server_complete: bool = False
    ended_at: float | None = None
    metrics_sent: bool = False


class GeminiLiveConnection(EngineConnection):
    """A live Gemini Live conversation, possibly spanning several WebSocket connections."""

    def __init__(self, engine: GeminiLiveEngine, options: EngineOptions) -> None:
        super().__init__(engine, options)
        self._e = engine
        self.chat_ctx = ChatContext(options.chat_ctx.items if options.chat_ctx else [])
        """Local transcript used to re-seed a fresh session when resumption is impossible."""
        self.instructions = options.instructions
        self.tools: list[FunctionTool] = list(options.tools)
        self._manual = not options.turn_detection
        self._tasks = BackgroundTasks("gemini-live")
        # ---- transport
        self._ws: ClientConnection | None = None
        self._retired: set[ClientConnection] = set()
        self._epoch = 0
        self._recv_task: asyncio.Task[None] | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._rotation_task: asyncio.Task[None] | None = None
        self._connected_at = now()
        self._switching = False
        self._outbox: deque[_Outgoing] = deque()
        self._outbox_audio = 0.0
        self._replay: deque[_Outgoing] = deque()
        self._replay_audio = 0.0
        self._stream_base: float | None = None
        self.connections = 0
        """Number of WebSocket connections opened so far."""
        self.resumptions = 0
        """Number of connection switches that resumed the server-side session."""
        # ---- resumption / rotation
        self._handle: str | None = None
        self._resumable = False
        self._pending_rotation: str | None = None
        self._rotation_deadline: float | None = None
        self._rotation_retry_at: float | None = None
        # ---- server turn state
        self._server_turn_open = False
        self._gen: _Generation | None = None
        self._turn_gens: list[_Generation] = []
        self._last_gen: _Generation | None = None
        self._turn_trigger = now()
        self._turn_usage: EngineUsage | None = None
        self._dropping = False
        self._self_interrupt = False
        self._continuation_expected = False
        self._requested_at: float | None = None
        self._held: list[ResponseText] = []
        self._hold_until: float | None = None
        self._held_user: _UserItem | None = None
        # ---- user state
        self._user: _UserItem | None = None
        self._user_speaking = False
        self._activity_open = False
        self._last_input_transcript_at = 0.0
        self._last_commit_pos = 0.0
        self._answered_until = 0.0
        """Input position up to which user speech has been answered (never replayed)."""
        # ---- tools
        self._pending_calls: dict[str, str] = {}
        self._stale_calls: set[str] = set()
        # ---- local speech tracking (metrics and idle detection only; never turn-taking)
        self._vad: VADStream | None = None
        if engine.local_vad:
            opts = VADOptions(min_speech_duration=0.1, min_silence_duration=0.3)
            self._vad = EnergyVAD(sample_rate=engine.input_sample_rate, options=opts).stream()
        self._local_speaking = False
        self._local_speech_end: float | None = None
        self._out_resampler = StreamResampler(engine.output_sample_rate, 1)

    # ------------------------------------------------------------------ properties
    @property
    def resumption_handle(self) -> str | None:
        """Latest session resumption handle (valid for about two hours)."""
        return self._handle

    # ------------------------------------------------------------------ lifecycle
    async def _start(self) -> None:
        ws, early = await self._open(None)
        self._install(ws, early)
        self._monitor_task = asyncio.create_task(self._monitor(), name="gemini-live-monitor")

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        current = asyncio.current_task()
        tasks = [self._monitor_task, self._rotation_task, self._recv_task]
        await cancel_and_wait(*[t for t in tasks if t is not None and t is not current])
        await self._tasks.cancel_all()
        ws, self._ws = self._ws, None
        for sock in [ws, *self._retired]:
            if sock is not None:
                await _close_quietly(sock)
        self._retired.clear()
        if self._vad is not None:
            self._vad.close()
        if self._server_turn_open or self._gen is not None:
            if self._gen is not None:
                self._finish_generation(self._gen, "incomplete")
            self._end_server_turn()
        await super().aclose()

    async def _open(self, handle: str | None) -> tuple[ClientConnection, list[dict[str, Any]]]:
        """Connect, send ``setup`` and wait for ``setupComplete``."""
        from websockets.asyncio.client import connect
        from websockets.exceptions import (
            ConnectionClosed,
            InvalidStatus,
            InvalidURI,
            WebSocketException,
        )

        url, headers = self._e._endpoint()
        turns = self._history_turns() if handle is None else []
        setup = self._build_setup(handle, seed_history=bool(turns))
        try:
            ws = await connect(
                url,
                additional_headers=headers,
                open_timeout=self._e.connect_timeout,
                max_size=_MAX_MESSAGE,
                **self._e._connect_kwargs(url),
            )
        except InvalidStatus as exc:
            body = exc.response.body.decode("utf-8", "replace") if exc.response.body else ""
            raise _http_error(exc.response.status_code, body) from exc
        except InvalidURI as exc:
            raise ConfigurationError(f"invalid Gemini Live base_url: {exc}") from exc
        except (OSError, WebSocketException, TimeoutError) as exc:
            msg = f"cannot connect to Gemini Live: {type(exc).__name__}: {exc}"
            raise ProviderConnectionError(msg, provider="google") from exc
        self.connections += 1
        early: list[dict[str, Any]] = []
        try:
            await ws.send(json.dumps(setup))
            async with asyncio.timeout(self._e.connect_timeout):
                while True:
                    reply = _decode(await ws.recv())
                    if reply is None:
                        continue
                    if "setupComplete" in reply:
                        break
                    early.append(reply)
            if turns:  # historyConfig.initialHistoryInClientContent: no model call
                await ws.send(json.dumps({"clientContent": {"turns": turns, "turnComplete": True}}))
        except ConnectionClosed as exc:
            await _close_quietly(ws)
            raise _close_error(*_close_info(exc)) from exc
        except TimeoutError as exc:
            await _close_quietly(ws)
            msg = "Gemini Live did not acknowledge the session setup in time"
            raise ProviderTimeoutError(msg, provider="google") from exc
        except BaseException:
            await _close_quietly(ws)
            raise
        return ws, early

    def _build_setup(self, handle: str | None, *, seed_history: bool) -> dict[str, Any]:
        e, opts = self._e, self.options
        generation: dict[str, Any] = {"responseModalities": ["AUDIO"]}
        voice = opts.voice or e.voice
        if voice:
            prebuilt = {"prebuiltVoiceConfig": {"voiceName": voice}}
            generation["speechConfig"] = {"voiceConfig": prebuilt}
        temperature = opts.temperature if opts.temperature is not None else e.temperature
        if temperature is not None:
            generation["temperature"] = temperature
        if e.thinking_level:
            generation["thinkingConfig"] = {"thinkingLevel": e.thinking_level}
        model = e.model if e.model.startswith(("models/", "tunedModels/")) else f"models/{e.model}"
        setup: dict[str, Any] = {"model": model, "generationConfig": generation}
        if self.instructions:
            setup["systemInstruction"] = {"parts": [{"text": self.instructions}]}
        if self.tools:
            behavior = "NON_BLOCKING" if e.tool_behavior == "non_blocking" else None
            declarations = [_declaration(t, behavior) for t in self.tools]
            setup["tools"] = [{"functionDeclarations": declarations}]
        realtime: dict[str, Any] = {}
        detection = _camelize(e.vad)
        if self._manual:
            detection["disabled"] = True
        if detection:
            realtime["automaticActivityDetection"] = detection
        if e.activity_handling:
            realtime["activityHandling"] = e.activity_handling.upper()
        if e.turn_coverage:
            realtime["turnCoverage"] = e.turn_coverage
        if realtime:
            setup["realtimeInputConfig"] = realtime
        if e.input_transcription:  # the language only hints the transcriber (audio is auto)
            hints = {"languageCodes": [opts.language]} if opts.language else {}
            setup["inputAudioTranscription"] = hints
        if e.output_transcription:
            setup["outputAudioTranscription"] = {}
        if e.session_resumption:
            setup["sessionResumption"] = {"handle": handle} if handle else {}
        cwc = e.context_window_compression
        if cwc:
            default_cwc: dict[str, Any] = {"slidingWindow": {}}
            setup["contextWindowCompression"] = (
                _camelize(cwc) if isinstance(cwc, Mapping) else default_cwc
            )
        if seed_history:
            setup["historyConfig"] = {"initialHistoryInClientContent": True}
        setup = _deep_merge(setup, e.extra_setup)
        setup = _deep_merge(setup, opts.extra)
        return {"setup": setup}

    def _history_turns(self) -> list[dict[str, Any]]:
        """User/assistant text history as ``Content`` turns (tool items are not re-seeded)."""
        turns: list[dict[str, Any]] = []
        for item in self.chat_ctx.items:
            if not isinstance(item, ChatMessage) or item.role not in ("user", "assistant"):
                continue
            text = item.text.strip()
            if not text:
                continue
            role = "user" if item.role == "user" else "model"
            if turns and turns[-1]["role"] == role:
                turns[-1]["parts"].append({"text": text})
            else:
                turns.append({"role": role, "parts": [{"text": text}]})
        return turns[-_MAX_SEED_TURNS:]

    def _install(self, ws: ClientConnection, early: list[dict[str, Any]]) -> None:
        self._epoch += 1
        self._ws = ws
        self._connected_at = now()
        self._stream_base = None
        self._resumable = False
        self._recv_task = self._tasks.spawn(
            self._recv_loop(ws, self._epoch, early), name=f"gemini-live-recv-{self._epoch}"
        )

    async def _recv_loop(
        self, ws: ClientConnection, epoch: int, early: list[dict[str, Any]]
    ) -> None:
        from websockets.exceptions import ConnectionClosed

        for first in early:
            self._dispatch(first)
        code: int | None
        try:
            async for raw in ws:
                if self._closed or epoch != self._epoch:
                    return
                message = _decode(raw)
                if message is not None:
                    self._dispatch(message)
            code, reason = ws.close_code, ws.close_reason or ""
        except ConnectionClosed as exc:
            code, reason = _close_info(exc)
        except Exception:
            logger.exception("gemini-live: receiving failed")
            code, reason = None, "receive failed"
            await _close_quietly(ws)
        if self._closed or epoch != self._epoch:
            return
        error = _close_error(code, reason)
        if isinstance(error, AuthenticationError):
            self._fail(error)
            return
        logger.warning("gemini-live: connection closed unexpectedly (%s %s)", code, reason)
        self._start_rotation(f"connection closed ({code} {reason})".strip())

    def _dispatch(self, msg: dict[str, Any]) -> None:
        try:
            self._on_message(msg)
        except Exception:
            logger.exception("gemini-live: failed to handle a server message")

    def _fail(self, error: Exception) -> None:
        if self._closed:
            return
        self._emit(EngineErrorEvent(error=error, recoverable=False))
        self._tasks.spawn(self.aclose(), name="gemini-live-close")

    # ------------------------------------------------------------------ sending
    async def _send(
        self,
        payload: dict[str, Any],
        *,
        replayable: bool,
        audio: tuple[float, float] | None = None,
    ) -> None:
        if self._closed:
            return
        start, duration = audio if audio is not None else (None, 0.0)
        entry = _Outgoing(json.dumps(payload), replayable, start, duration)
        ws = self._ws
        if self._switching or ws is None:
            self._queue(entry)
        elif not await self._deliver(ws, entry):
            self._queue(entry, front=True)
            self._start_rotation("connection lost while sending")

    async def _deliver(self, ws: ClientConnection, entry: _Outgoing) -> bool:
        from websockets.exceptions import ConnectionClosed

        if entry.audio_start is not None and self._stream_base is None:
            self._stream_base = entry.audio_start
        try:
            await ws.send(entry.data)
        except ConnectionClosed:
            return False
        entry.sent_at = now()
        if entry.replayable:
            self._replay.append(entry)
            self._replay_audio += entry.audio_duration
            while self._replay_audio > self._e.max_buffered_audio and self._replay:
                self._replay_audio -= self._replay.popleft().audio_duration
        return True

    def _queue(self, entry: _Outgoing, *, front: bool = False) -> None:
        if front:
            self._outbox.appendleft(entry)
        else:
            self._outbox.append(entry)
        self._outbox_audio += entry.audio_duration
        dropped = 0.0
        while self._outbox_audio > self._e.max_buffered_audio:
            victim = next((x for x in self._outbox if x.audio_start is not None), None)
            if victim is None:
                break
            self._outbox.remove(victim)
            self._outbox_audio -= victim.audio_duration
            dropped += victim.audio_duration
        if dropped:
            logger.warning("gemini-live: reconnect buffer full, dropped %.2fs of audio", dropped)

    def _trim_replay(self) -> None:
        """A new handle arrived: keep only what the resumed state may be missing -- the
        audio sent in the last ``resume_replay`` seconds, minus speech already answered."""
        cutoff = now() - self._e.resume_replay
        while self._replay:
            head = self._replay[0]
            answered = (
                head.audio_start is not None
                and head.audio_start + head.audio_duration <= self._answered_until
            )
            if head.sent_at >= cutoff and not answered:
                break
            self._replay_audio -= self._replay.popleft().audio_duration

    # ----------------------------------------------------------------- rotation
    def _request_rotation(self, reason: str, *, deadline: float | None = None) -> None:
        if self._pending_rotation is None or reason == "go_away":
            self._pending_rotation = reason
        if deadline is not None:
            current = self._rotation_deadline
            self._rotation_deadline = deadline if current is None else min(current, deadline)
        self._check_rotation()

    def _check_rotation(self) -> None:
        if self._closed or self._switching:
            return
        t = now()
        rotate_after = self._e.rotate_after
        if (
            self._pending_rotation is None
            and rotate_after is not None
            and t - self._connected_at >= rotate_after
        ):
            self._pending_rotation = "max_connection_age"
        reason = self._pending_rotation
        if reason is None:
            return
        if self._rotation_deadline is not None and t >= self._rotation_deadline:
            self._start_rotation(f"{reason} (deadline)")
        elif (self._rotation_retry_at is None or t >= self._rotation_retry_at) and self._is_idle():
            self._start_rotation(reason)

    def _is_idle(self) -> bool:
        if self._gen is not None or self._server_turn_open or self._pending_calls:
            return False
        if self._user_speaking or self._local_speaking or self._activity_open:
            return False
        if now() - self._last_input_transcript_at < 0.5:
            return False
        return not self._e.session_resumption or (self._resumable and self._handle is not None)

    def _start_rotation(self, reason: str) -> None:
        if self._closed:
            return
        if self._rotation_task is not None and not self._rotation_task.done():
            return  # the running rotation re-checks the new connection before finishing
        self._switching = True
        self._pending_rotation = None
        self._rotation_deadline = None
        self._rotation_retry_at = None
        self._rotation_task = asyncio.create_task(self._rotate(reason), name="gemini-live-rotate")

    async def _rotate(self, reason: str) -> None:
        from websockets.protocol import State

        self._emit(EngineStatus(status="reconnecting", detail=reason))
        failures = 0
        delay = 0.5
        while not self._closed:
            handle = self._handle if self._e.session_resumption else None
            try:
                ws, early = await self._open(handle)
            except AuthenticationError as exc:
                self._fail(exc)
                return
            except Exception as exc:
                failures += 1
                rejected = isinstance(exc, ProviderError) and not exc.retryable
                current = self._ws
                if (  # a planned rotation failed but the current connection still works
                    current is not None
                    and current.state is State.OPEN
                    and await self._keep_current(current, reason, exc, retry_in=delay)
                ):
                    return
                if rejected and handle is not None:
                    # e.g. an expired handle: continue in a fresh session (history re-seeded)
                    logger.warning("gemini-live: cannot resume (%s); starting a new session", exc)
                    self._handle = None
                    continue
                if rejected or failures >= self._e.max_reconnect_attempts:
                    msg = f"Gemini Live reconnection failed after {failures} attempts: {exc}"
                    self._fail(exc if rejected else ProviderConnectionError(msg, provider="google"))
                    return
                logger.warning("gemini-live: reconnect attempt %d failed: %s", failures, exc)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 8.0)
                continue
            old = self._ws
            self._switch(ws, early, resumed=handle is not None)
            if old is not None and old is not ws:
                self._retire(old)
            if await self._flush(ws) and ws.state is State.OPEN:
                self._switching = False  # outbox empty; no await since the check
                if handle is not None:
                    self.resumptions += 1
                status: EngineStatusKind = "resumed" if handle is not None else "reconnected"
                self._emit(EngineStatus(status=status, detail=reason))
                return
            failures += 1
            reason = "connection lost during reconnection"
            if failures >= self._e.max_reconnect_attempts:
                msg = "Gemini Live connection keeps dropping"
                self._fail(ProviderConnectionError(msg, provider="google"))
                return
            await asyncio.sleep(delay)
            delay = min(delay * 2, 8.0)

    def _switch(self, ws: ClientConnection, early: list[dict[str, Any]], *, resumed: bool) -> None:
        """Retire the previous connection's server-side state and install ``ws``."""
        if self._gen is not None:
            self._finish_generation(self._gen, "incomplete")
        if self._server_turn_open:
            self._end_server_turn()
        if self._pending_calls:  # unknown to the resumed/new session: withdraw them
            stale = list(self._pending_calls)
            self._pending_calls.clear()
            self._stale_calls.update(stale)
            self._emit(ToolCallCancelled(call_ids=stale))
        self._dropping = False
        self._self_interrupt = False
        self._continuation_expected = False
        if not resumed:
            self._handle = None
        self._install(ws, early)

    def _retire(self, ws: ClientConnection) -> None:
        """Close a replaced connection in the background (``aclose`` finishes the job)."""
        self._retired.add(ws)

        async def close() -> None:
            await _close_quietly(ws)
            self._retired.discard(ws)

        self._tasks.spawn(close(), name="gemini-live-close-old")

    async def _keep_current(
        self, ws: ClientConnection, reason: str, error: Exception, *, retry_in: float
    ) -> bool:
        """Abort a planned rotation and carry on with ``ws`` (retried later if sensible)."""
        if not await self._flush_outbox(ws):
            return False
        self._switching = False  # outbox empty; no await since the check
        logger.warning("gemini-live: could not rotate (%s); keeping the current connection", error)
        if not (isinstance(error, ProviderError) and not error.retryable):
            self._pending_rotation = reason.removesuffix(" (deadline)")
            self._rotation_retry_at = now() + retry_in
        self._emit(EngineErrorEvent(error=error, recoverable=True))
        self._emit(EngineStatus(status="resumed", detail=f"kept the current connection ({reason})"))
        return True

    async def _flush(self, ws: ClientConnection) -> bool:
        """Replay audio sent since the handle, then the messages queued during the switch."""
        replay = list(self._replay)
        self._replay.clear()
        self._replay_audio = 0.0
        for i, entry in enumerate(replay):
            if not await self._deliver(ws, entry):
                for rest in replay[i:]:
                    self._replay.append(rest)
                    self._replay_audio += rest.audio_duration
                return False
        return await self._flush_outbox(ws)

    async def _flush_outbox(self, ws: ClientConnection) -> bool:
        while self._outbox:
            entry = self._outbox.popleft()
            self._outbox_audio -= entry.audio_duration
            if not await self._deliver(ws, entry):
                self._outbox.appendleft(entry)
                self._outbox_audio += entry.audio_duration
                return False
        return True

    async def _monitor(self) -> None:
        while not self._closed:
            await asyncio.sleep(_MONITOR_INTERVAL)
            if self._hold_until is not None and now() >= self._hold_until:
                self._end_hold()
            self._check_rotation()

    # --------------------------------------------------------------- audio input
    async def _send_audio(self, frame: AudioFrame) -> None:
        end = self.input_audio_time
        start = end - frame.duration
        if self._vad is not None:
            for ev in self._vad.push_audio(frame):
                if ev.type == VADEventType.START_OF_SPEECH:
                    self._local_speaking = True
                elif ev.type == VADEventType.END_OF_SPEECH:
                    self._local_speaking = False
                    self._local_speech_end = ev.audio_time - ev.silence_duration
        if self._manual and not self._activity_open:
            self._activity_open = True
            await self._send({"realtimeInput": {"activityStart": {}}}, replayable=True)
            self._on_user_activity_start(start)
        audio = {"data": frame.to_base64(), "mimeType": f"audio/pcm;rate={frame.sample_rate}"}
        await self._send(
            {"realtimeInput": {"audio": audio}}, replayable=True, audio=(start, frame.duration)
        )

    async def commit_input(self) -> None:
        """Manual turns: send ``activityEnd``. Automatic VAD: ``audioStreamEnd`` flushes the
        server's end-of-speech wait (hybrid VAD)."""
        if not self._manual:
            await self._send({"realtimeInput": {"audioStreamEnd": True}}, replayable=False)
            return
        user = self._user if self._user is not None and not self._user.committed else None
        if not self._activity_open and user is None:
            return  # nothing was said since the last commit
        if self._activity_open:
            self._activity_open = False
            await self._send({"realtimeInput": {"activityEnd": {}}}, replayable=True)
        if user is not None and user.speech_end is None:
            user.speech_end = self.input_audio_time
        self._commit_user(user)
        self._requested_at = now()

    async def clear_input(self) -> None:
        """No-op: the Live API cannot discard audio it already received."""

    # ------------------------------------------------------------------- control
    async def send_text(self, text: str, *, respond: bool = True) -> None:
        self.chat_ctx.add_message("user", text)
        if respond:
            await self._prepare_request()
        elif self._server_turn_open:
            self._self_interrupt = True  # if the server stops talking, it was not the user
        content = {"turns": [{"role": "user", "parts": [{"text": text}]}], "turnComplete": respond}
        await self._send({"clientContent": content}, replayable=True)

    async def create_response(self, *, instructions: str | None = None) -> None:
        turns = [{"role": "user", "parts": [{"text": instructions}]}] if instructions else []
        await self._prepare_request()
        await self._send({"clientContent": {"turns": turns, "turnComplete": True}}, replayable=True)

    async def _prepare_request(self) -> None:
        # a clientContent turn interrupts any ongoing generation: not a user barge-in
        if self._server_turn_open or self._gen is not None:
            self._self_interrupt = True
            await self.cancel_response()
        self._requested_at = now()

    async def cancel_response(self) -> None:
        """Stop the current response locally (the rest of the server turn is dropped).

        The Live API has no explicit cancel message; the server stops generating by itself
        when it detects user speech (``activityHandling``) or receives new client content.
        """
        gen = self._gen
        if gen is None:
            return
        self._finish_generation(gen, "cancelled")
        self._dropping = True

    async def interrupt(
        self, item_id: str | None = None, played_ms: int | None = None
    ) -> str | None:
        gen = self._gen
        if gen is not None and (item_id is None or gen.item_id == item_id):
            await self.cancel_response()
        return None  # no truncation: the session estimates what was heard

    async def send_tool_output(self, output: FunctionCallOutput, *, respond: bool = True) -> None:
        self.chat_ctx.append(output)
        if output.call_id in self._stale_calls:
            logger.warning(
                "gemini-live: dropping the output of tool call %s (issued before a reconnect)",
                output.call_id,
            )
            return
        name = self._pending_calls.pop(output.call_id, None) or output.name
        key = "error" if output.is_error else "result"
        response: dict[str, Any] = {
            "id": output.call_id,
            "name": name,
            "response": {key: _tool_result(output.output)},
        }
        if self._e.tool_behavior == "non_blocking":
            response["scheduling"] = "WHEN_IDLE" if respond else "SILENT"
        if respond:
            self._requested_at = now()
        await self._send({"toolResponse": {"functionResponses": [response]}}, replayable=False)

    async def update(
        self, *, instructions: str | None = None, tools: list[FunctionTool] | None = None
    ) -> None:
        """Change instructions/tools. The Live API only accepts a new ``setup`` on a new
        connection, so the engine rotates (resuming the session) at the next idle moment."""
        changed = False
        if instructions is not None and instructions != self.instructions:
            self.instructions = instructions
            changed = True
        if tools is not None:
            self.tools = list(tools)
            changed = True
        if changed:
            self._request_rotation("update")

    # ----------------------------------------------------------- server messages
    def _on_message(self, msg: dict[str, Any]) -> None:
        if self._hold_until is not None and now() >= self._hold_until:
            self._end_hold()
        usage = msg.get("usageMetadata")
        if isinstance(usage, Mapping):
            self._on_usage(usage)
        content = msg.get("serverContent")
        if isinstance(content, Mapping):
            self._on_server_content(content)
        tool_call = msg.get("toolCall")
        if isinstance(tool_call, Mapping):
            self._on_tool_call(tool_call)
        cancellation = msg.get("toolCallCancellation")
        if isinstance(cancellation, Mapping):
            self._on_tool_cancellation(cancellation)
        activity = msg.get("voiceActivity")
        if isinstance(activity, Mapping):
            self._on_voice_activity(activity)
        update = msg.get("sessionResumptionUpdate")
        if isinstance(update, Mapping):
            self._on_resumption_update(update)
        go_away = msg.get("goAway")
        if isinstance(go_away, Mapping):
            self._on_go_away(go_away)

    def _on_server_content(self, sc: Mapping[str, Any]) -> None:
        interim = sc.get("interimInputTranscription")
        if isinstance(interim, Mapping):
            self._on_input_transcription(interim, interim=True)
        final = sc.get("inputTranscription")
        if isinstance(final, Mapping):
            self._on_input_transcription(final, interim=False)
        turn = sc.get("modelTurn")
        if isinstance(turn, Mapping):
            for part in turn.get("parts") or ():
                if isinstance(part, Mapping):
                    self._on_part(part)
        out = sc.get("outputTranscription")
        if isinstance(out, Mapping) and out.get("text"):
            self._on_output_text(str(out["text"]))
        if sc.get("interrupted"):
            self._on_interrupted()
        if sc.get("generationComplete"):
            gen = self._gen
            if gen is not None:
                gen.server_complete = True
                self._finish_generation(gen, "completed")
        if sc.get("turnComplete"):
            self._on_turn_complete(sc)

    def _on_part(self, part: Mapping[str, Any]) -> None:
        if part.get("thought"):
            return  # reasoning summaries are not spoken
        inline = part.get("inlineData")
        if isinstance(inline, Mapping):
            mime = str(inline.get("mimeType") or "")
            data = inline.get("data")
            if not data or (mime and not mime.startswith("audio/")):
                return
            gen = self._open_generation()
            if gen is None:
                return
            pcm = base64.b64decode(data)
            if len(pcm) % 2:
                pcm = pcm[:-1]
            rate = _audio_rate(mime, self.output_sample_rate)
            frame = self._out_resampler.push(AudioFrame(pcm, rate, 1))
            if not frame:
                return
            if gen.first_audio_at is None:
                gen.first_audio_at = now()
            self._emit(ResponseAudio(response_id=gen.response_id, item_id=gen.item_id, frame=frame))
            return
        text = part.get("text")
        if text and not self._e.output_transcription:
            self._on_output_text(str(text))

    def _on_output_text(self, text: str) -> None:
        gen = self._gen
        if gen is None:
            last = self._last_gen
            if (
                last is not None
                and last.server_complete
                and now() - (last.ended_at or 0.0) < _LATE_TRANSCRIPT_GRACE
            ):
                gen = last  # a transcript straggler of a finished generation
            else:
                gen = self._open_generation()
        if gen is None:
            return
        gen.text.append(text)
        ev = ResponseText(response_id=gen.response_id, item_id=gen.item_id, delta=text)
        if self._hold_until is not None:
            self._held.append(ev)
        else:
            self._emit(ev)
        if gen.status is not None:
            self._sync_assistant(gen)

    def _on_input_transcription(self, t: Mapping[str, Any], *, interim: bool) -> None:
        text = str(t.get("text") or "")
        if not text:
            if t.get("finished") and self._held_user is not None and not interim:
                self._end_hold()  # the (late) transcript is complete
            return
        t_now = now()
        self._last_input_transcript_at = t_now
        user = self._user
        if user is None or (
            user.committed
            and user.closed_at is not None
            and t_now - user.closed_at > _LATE_TRANSCRIPT_GRACE
        ):
            user = self._user = _UserItem()
        if not user.committed:
            user.evidence = True
        if t.get("languageCode"):
            user.language = str(t["languageCode"])
        if interim:
            if user.committed:
                return
            user.interim = text
        else:
            user.final += text
            user.interim = ""
        if user.committed:
            if self._held_user is not user:
                self._emit_user_final(user)  # a late correction: update the final transcript
            elif t.get("finished"):
                self._end_hold()
        else:
            self._emit(
                InputTranscript(
                    item_id=user.item_id, text=user.text, is_final=False, language=user.language
                )
            )

    def _on_voice_activity(self, va: Mapping[str, Any]) -> None:
        kind = str(va.get("type") or va.get("voiceActivityType") or "").upper()
        offset = _parse_duration(va.get("audioOffset"))
        audio_time = None
        if offset is not None and self._stream_base is not None:
            audio_time = self._stream_base + offset
        if kind.endswith("START"):
            self._on_user_activity_start(audio_time)
        elif kind.endswith("END"):
            if audio_time is None:
                audio_time = self._local_end_estimate()
            user = self._user
            if user is not None and not user.committed:
                user.speech_end = audio_time
                user.stopped = True
            if self._user_speaking:
                self._user_speaking = False
                self._emit(InputSpeechStopped(audio_time=audio_time))

    def _on_user_activity_start(self, audio_time: float | None) -> None:
        user = self._user
        if user is None or user.committed:
            user = self._user = _UserItem()
        user.evidence = True
        if not self._user_speaking:
            self._user_speaking = True
            self._emit(InputSpeechStarted(audio_time=audio_time))

    def _on_interrupted(self) -> None:
        if self._self_interrupt:
            self._self_interrupt = False  # our own clientContent stopped the generation
        else:
            self._on_user_activity_start(None)  # server-side barge-in
        gen = self._gen
        if gen is not None:
            self._finish_generation(gen, "cancelled")
        self._dropping = False

    def _on_turn_complete(self, sc: Mapping[str, Any]) -> None:
        gen = self._gen
        if gen is not None:
            self._finish_generation(gen, "completed")
        had_output = bool(self._turn_gens)
        status = str(sc.get("interactionStatus") or "").upper()
        reason = str(sc.get("turnCompleteReason") or "").upper()
        waiting = bool(sc.get("waitingForInput")) or reason == "NEED_MORE_INPUT"
        self._end_server_turn()
        self._continuation_expected = status == "IN_PROGRESS"
        user = self._user
        if user is not None:
            if user.committed:
                if user.closed_at is None:
                    user.closed_at = now()
                if user.speech_end is not None:
                    self._answered_until = max(self._answered_until, user.speech_end)
            elif not had_output and not waiting and user.text:
                self._emit_user_final(user)  # the model chose not to answer
                self._user = None
        self._check_rotation()

    def _on_tool_call(self, tc: Mapping[str, Any]) -> None:
        calls = [c for c in tc.get("functionCalls") or () if isinstance(c, Mapping)]
        if not calls:
            return
        gen = self._open_generation(force=True)
        if gen is None:  # pragma: no cover - force=True always opens one
            return
        for c in calls:
            call_id = str(c.get("id") or new_id("call_"))
            name = str(c.get("name") or "")
            args = c.get("args")
            arguments = json.dumps(args if args is not None else {})
            call = FunctionCall(name=name, arguments=arguments, call_id=call_id)
            self._pending_calls[call_id] = name
            gen.calls.append(call)
            self.chat_ctx.append(call)
            self._emit(ResponseToolCall(response_id=gen.response_id, call=call))
        # End the response here so the session runs the tools right away; anything the
        # model says afterwards (non-blocking) or after the results (blocking) is a new one.
        self._finish_generation(gen, "completed")

    def _on_tool_cancellation(self, tcc: Mapping[str, Any]) -> None:
        ids = [str(i) for i in tcc.get("ids") or ()]
        if not ids:
            return
        for call_id in ids:
            self._pending_calls.pop(call_id, None)
        self._emit(ToolCallCancelled(call_ids=ids))

    def _on_resumption_update(self, update: Mapping[str, Any]) -> None:
        handle = update.get("newHandle")
        if update.get("resumable") and handle:
            self._handle = str(handle)
            self._resumable = True
            self._trim_replay()
        else:
            self._resumable = False
        self._check_rotation()

    def _on_go_away(self, go_away: Mapping[str, Any]) -> None:
        time_left = _parse_duration(go_away.get("timeLeft"))
        self._emit(EngineStatus(status="expiring", detail="go_away", time_left=time_left))
        budget = max(0.0, (time_left or 0.0) - self._e.go_away_margin)
        self._request_rotation("go_away", deadline=now() + budget)

    def _on_usage(self, u: Mapping[str, Any]) -> None:
        usage = _usage(u)
        if self._server_turn_open or self._gen is not None:
            previous = self._turn_usage
            self._turn_usage = usage if previous is None else _add_usage(previous, usage)
            return
        # reported outside of a server turn: account for it right away
        rid = self._last_gen.response_id if self._last_gen is not None else new_id("resp_")
        self._emit_metrics(rid, usage=usage)

    # ------------------------------------------------------- turns and responses
    def _open_generation(self, *, force: bool = False) -> _Generation | None:
        if self._gen is not None:
            return self._gen
        if self._dropping and not force:
            return None
        t = now()
        trigger = t
        if not self._server_turn_open:
            self._server_turn_open = True
            self._begin_server_turn()
            trigger = self._turn_trigger
        elif self._requested_at is not None and self._e.tool_behavior == "blocking":
            # the model resumes the paused turn with the tool results we sent
            trigger, self._requested_at = self._requested_at, None
        gen = _Generation(
            response_id=new_id("resp_"), item_id=new_id("item_"), started_at=t, trigger_at=trigger
        )
        self._gen = gen
        self._turn_gens.append(gen)
        self._last_gen = gen
        self._emit(ResponseStarted(response_id=gen.response_id))
        return gen

    def _begin_server_turn(self) -> None:
        """Decide whether this server turn answers user speech (-> commit the user turn)."""
        t = now()
        requested, self._requested_at = self._requested_at, None
        fresh_request = requested is not None and t - requested < _REQUEST_TTL
        continuation, self._continuation_expected = self._continuation_expected, False
        user = self._user
        evidence = user is not None and not user.committed and user.evidence
        self._turn_trigger = t
        if evidence or not (fresh_request or continuation):
            self._commit_user(user)
        elif fresh_request and requested is not None:
            self._turn_trigger = requested

    def _commit_user(self, user: _UserItem | None) -> None:
        if user is None or user.committed:
            user = self._user = _UserItem()
        if not user.stopped:
            end = user.speech_end if user.speech_end is not None else self._local_end_estimate()
            if self._user_speaking or end is not None:
                self._user_speaking = False
                self._emit(InputSpeechStopped(audio_time=end))
            user.stopped = True
            user.speech_end = end
        user.committed = True
        self._last_commit_pos = self.input_audio_time
        self._turn_trigger = now()
        self._emit(InputCommitted(item_id=user.item_id))
        if self.chat_ctx.get(user.item_id) is None:  # keep the history in conversation order
            self.chat_ctx.add_message("user", user.text, id=user.item_id)
        if user.text:
            self._emit_user_final(user)
        elif self._e.input_transcription:
            # the transcript lags behind: hold the answer's text back briefly so that the
            # user's final transcript (accumulated meanwhile) comes first
            self._hold_until = now() + _TEXT_HOLD
            self._held_user = user

    def _emit_user_final(self, user: _UserItem) -> None:
        text = user.text
        if not text or text == user.emitted:
            return
        user.emitted = text
        self._emit(
            InputTranscript(item_id=user.item_id, text=text, is_final=True, language=user.language)
        )
        msg = self.chat_ctx.get(user.item_id)
        if isinstance(msg, ChatMessage):
            msg.content = [text]
        else:
            self.chat_ctx.add_message("user", text, id=user.item_id)

    def _local_end_estimate(self) -> float | None:
        end = self._local_speech_end
        if end is None or self._local_speaking or end <= self._last_commit_pos:
            return None
        return end

    def _end_hold(self) -> None:
        """Emit the held user transcript, then the agent text deltas held behind it."""
        self._hold_until = None
        user, self._held_user = self._held_user, None
        if user is not None:
            self._emit_user_final(user)
        held, self._held = self._held, []
        for ev in held:
            self._emit(ev)

    def _finish_generation(self, gen: _Generation, status: ResponseStatus) -> None:
        if gen.status is not None:
            return
        gen.status = status
        gen.ended_at = now()
        if self._gen is gen:
            self._gen = None
        self._end_hold()  # transcript deltas always precede ResponseDone
        self._sync_assistant(gen)
        usage = self._turn_usage
        self._emit(ResponseDone(response_id=gen.response_id, status=status, usage=usage))

    def _sync_assistant(self, gen: _Generation) -> None:
        text = "".join(gen.text).strip()
        if not text:
            return
        msg = self.chat_ctx.get(gen.item_id)
        if isinstance(msg, ChatMessage):
            msg.content = [text]
        else:
            msg = self.chat_ctx.add_message("assistant", text, id=gen.item_id)
        msg.interrupted = gen.status == "cancelled"

    def _end_server_turn(self) -> None:
        self._end_hold()
        gens = [g for g in self._turn_gens if not g.metrics_sent]
        for i, gen in enumerate(gens):
            last = i == len(gens) - 1
            self._emit_metrics(gen.response_id, gen=gen, usage=self._turn_usage if last else None)
        if not gens and self._turn_usage is not None and self._last_gen is not None:
            self._emit_metrics(self._last_gen.response_id, usage=self._turn_usage)
        self._server_turn_open = False
        self._dropping = False
        self._self_interrupt = False
        self._turn_gens = []
        self._turn_usage = None

    def _emit_metrics(
        self, response_id: str, *, gen: _Generation | None = None, usage: EngineUsage | None
    ) -> None:
        u = usage or EngineUsage()
        ttfb = duration = None
        if gen is not None:
            gen.metrics_sent = True
            if gen.first_audio_at is not None:
                ttfb = gen.first_audio_at - gen.trigger_at
            duration = (gen.ended_at or now()) - gen.started_at
        self._e.emit(
            "metrics",
            EngineMetrics(
                provider=self._e.provider,
                model=self._e.model,
                response_id=response_id,
                ttfb=ttfb,
                duration=duration or 0.0,
                input_text_tokens=u.input_text_tokens,
                input_audio_tokens=u.input_audio_tokens,
                output_text_tokens=u.output_text_tokens,
                output_audio_tokens=u.output_audio_tokens,
                cached_tokens=u.cached_tokens,
                cancelled=gen is not None and gen.status == "cancelled",
            ),
        )


def _declaration(tool: FunctionTool, behavior: str | None) -> dict[str, Any]:
    decl: dict[str, Any] = {"name": tool.name, "description": tool.description or tool.name}
    if tool.parameters.get("properties"):
        decl["parametersJsonSchema"] = tool.parameters
    if behavior:
        decl["behavior"] = behavior
    return decl
