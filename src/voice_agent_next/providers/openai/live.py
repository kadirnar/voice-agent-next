"""OpenAI GPT-Live full-duplex speech-to-speech engine (the Live protocol over WebSocket).

``AgentSession("openai-live/gpt-live-1")`` runs an agent on GPT-Live, OpenAI's full-duplex
voice model: it listens while it speaks, decides by itself when to talk, and *delegates*
reasoning and tool use to a backend. The protocol is not the Realtime API: one WebSocket
to ``wss://api.openai.com/v1/live/sessions`` carries ``session.*`` events (docs checked on
2026-09-25: https://developers.openai.com/api/reference/resources/live/primary-websocket).

**Wire protocol.** The client sends ``session.start`` (model, instructions, audio format,
voice, delegation, seed history) and waits for ``session.started`` (session id and
``expires_at``); then ``session.input_audio.append`` (base64 PCM16, 16 or 24 kHz — the
same format both ways). The server streams ``session.output_audio.delta`` (no timing, no
"done" event), ``session.input_transcript.delta`` / ``session.output_transcript.delta``
(text fragments with ``start_ms``/``end_ms`` on the session timeline, no item ids, no
turn boundaries), ``session.usage.updated`` (cumulative *seconds*), ``error`` and, last,
``session.closed`` (reason ``close_requested``, ``expired``, ``content``,
``remote_hangup`` or ``connection_lost``, plus the final usage).

**Full duplex** follows the Moshi precedent (:mod:`voice_agent_next.providers.moshi`):

* a **response** is a stretch of agent speech. It starts at the first output chunk above
  ``speech_threshold_db`` (or the first output transcript fragment) and ends once the
  received speech has played out and ``response_gap`` seconds passed (``yield_gap``
  while the user talks) with neither speech nor transcript. Output transcript fragments
  -> ``ResponseText``;
* **user speech** comes from a local energy VAD on the sent audio. ``InputSpeechStarted``
  / ``InputSpeechStopped`` are reported while the agent is quiet; speech *over* the agent
  is left to the model (GPT-Live resolves overlaps itself), unless ``report_overlap``;
* a **turn**: the first response after user speech is preceded by ``InputCommitted`` and
  the user's transcript so far (``InputTranscript(is_final=True)``); transcript fragments
  from before the commit that arrive later update that item;
* **interrupt** (``session.interrupt()``): the model cannot be stopped from the client,
  so the current response ends as ``cancelled`` and the agent is muted locally until its
  next pause;
* ``say()`` / ``create_response()`` append *instructions* (``session.instructions.append``,
  which can redirect speech in progress); ``send_text()`` appends *commentary* (said
  aloud, paraphrased) or, with ``respond=False``, *thinking* (context, not said).

**Delegation** (``EngineCapabilities.tool_mode == "delegation"``): the voice model hands
work to a backend and keeps talking; interrupting speech never cancels backend work.

* ``delegation="responses"`` (default): GPT-Live runs a Responses model
  (``responses_model``) with the agent's tools registered as function tools. A backend
  function call (a ``response.output_item.done`` inside a ``response.event`` envelope)
  becomes a ``ResponseToolCall``; the session runs the tool; its output goes back as
  ``response.item.create`` (``function_call_output``) and, once every call of that
  backend response has an output, ``response.create`` continues the backend.
* ``delegation="client"``: GPT-Live asks *the application* (``session.delegation.created``,
  metadata only). The engine turns it into a call of the agent's ``delegation_tool``
  (default ``"delegate"``) with ``{"request": <what the user said since the previous
  delegation>}`` and ``call_id`` = the delegation id; the tool's output goes back as
  ``session.commentary.append`` (spoken) or, for ``scheduling="silent"`` tools,
  ``session.thinking.append``, with that ``delegation_id``.

**Mute.** :meth:`OpenAILiveConnection.mute_input` / :meth:`~OpenAILiveConnection.unmute_input`
send ``session.input_audio.mute``/``unmute`` and wait for the acknowledgement; the state
survives session rotation. Muting input stops neither the agent's speech nor backend work.

**Rotation.** Sessions expire (``expires_at``); the engine announces
``EngineStatus("expiring")`` ``expiry_warning`` seconds ahead and
:class:`~voice_agent_next.engines.rotation.RotatingConnection` moves the conversation to a
fresh session at a quiet moment, seeded with the carried-over text history
(``session.input``, at most 128 messages / ~8k tokens). A drop, or a ``session.closed``
with reason ``expired``/``connection_lost``, reconnects the same way. ``aclose()`` sends
``session.close`` and waits for ``session.closed`` so the final usage is confirmed.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal, TypeAlias
from urllib.parse import urlsplit, urlunsplit

from websockets.asyncio.client import ClientConnection
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from ...audio.frame import AudioFrame
from ...audio.pcm import PCM16Reassembler
from ...chat import ChatContext, ChatMessage, FunctionCall, FunctionCallOutput
from ...engine import EngineCapabilities, EngineConnection, EngineOptions, S2SEngine
from ...engines.rotation import RotatingConnection, RotatingEngine, RotationPolicy, _Link
from ...errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderError,
    RateLimitError,
)
from ...events import (
    EngineErrorEvent,
    EngineStatus,
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
from ...metrics import EngineMetrics, LLMMetrics
from ...registry import register_provider
from ...tools import FunctionTool, ToolScheduling
from ...utils.aio import BackgroundTasks, cancel_and_wait
from ...utils.clock import now
from ...utils.ids import new_id
from ...utils.log import logger
from ...vad import VADEventType, VADOptions, VADStream
from ..energy import EnergyVAD
from .realtime import (
    _AUTH_ERROR_CODES,
    _CONNECT_ACCEPTS_PROXY,
    _RATE_LIMIT_CODES,
    RealtimeConnectTimeoutError,
    _close_reason,
    _deep_merge,
    _handshake_error,
    _is_loopback,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_URL",
    "VOICES",
    "LiveSessionConnection",
    "OpenAILiveConnection",
    "OpenAILiveEngine",
    "OpenAILiveSessionEngine",
    "live_url",
    "seed_items",
]

DEFAULT_MODEL: Final = "gpt-live-1"
DEFAULT_URL: Final = "wss://api.openai.com/v1"
DEFAULT_VOICE: Final = "marin"
DEFAULT_RESPONSES_MODEL: Final = "gpt-5.6-terra"
VOICES: Final = (
    "alloy", "ash", "ballad", "beacon", "bossa", "cedar", "cinder", "coral", "delta",
    "echo", "gleam", "marin", "meridian", "quartz", "ripple", "sage", "shimmer", "stone",
    "tempo", "verse", "vesper", "willow",
)  # fmt: skip
"""Built-in voices of ``session.audio.output.voice`` (custom voices: ``{"id": ...}``)."""
SAMPLE_RATES: Final = (16_000, 24_000)

Delegation: TypeAlias = Literal["responses", "client"]
CloseReason: TypeAlias = Literal[
    "close_requested", "expired", "content", "remote_hangup", "connection_lost"
]

SAY_INSTRUCTIONS: Final = (
    'Immediately say exactly the following, verbatim and in full, then pause and listen: "{text}"'
)
RESPOND_INSTRUCTIONS: Final = "Respond to the user now."
UPDATE_INSTRUCTIONS: Final = "Updated instructions (they replace the earlier ones): {text}"

_MAX_MESSAGE_SIZE: Final = 32 * 2**20
_MAX_APPEND_CHARS: Final = 1800
"""Appends take at most 500 tokens: longer content is split into several appends."""
_MAX_SEED_MESSAGES: Final = 128
_MAX_SEED_CHARS: Final = 28_000
"""``session.input`` takes at most 8,192 tokens (about 4 characters each)."""
_TICK: Final = 0.04
_KEEPALIVE_IDLE: Final = 0.1
_KEEPALIVE_CHUNK: Final = 0.1
_MAX_TRACKED: Final = 256
_RETRYABLE_CLOSE: Final = frozenset({"expired", "connection_lost"})


# ------------------------------------------------------------------------------ helpers
def live_url(base_url: str) -> str:
    """``https://host/v1`` -> ``wss://host/v1/live/sessions`` (a full ``.../live/sessions``
    URL is kept)."""
    parts = urlsplit(base_url.strip())
    scheme = {"http": "ws", "https": "wss"}.get(parts.scheme, parts.scheme)
    if scheme not in ("ws", "wss") or not parts.netloc:
        raise ConfigurationError(f"invalid GPT-Live base URL {base_url!r} (expected ws[s]://...)")
    path = parts.path.rstrip("/")
    if not path.endswith("/live/sessions"):
        path += "/live/sessions"
    return urlunsplit((scheme, parts.netloc, path, parts.query, ""))


def _live_error(err: Mapping[str, Any], provider: str, context: str | None) -> ProviderError:
    etype = str(err.get("type") or "error")
    code = err.get("code")
    label = etype if not code or code == etype else f"{etype}/{code}"
    msg = f"{provider} error ({label}): {err.get('message') or 'no message'}"
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


def _chunks(text: str, limit: int = _MAX_APPEND_CHARS) -> list[str]:
    """Split ``text`` into pieces of at most ``limit`` characters, at whitespace."""
    text = text.strip()
    out: list[str] = []
    while len(text) > limit:
        cut = text.rfind(" ", 0, limit)
        cut = cut if cut > limit // 2 else limit
        out.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        out.append(text)
    return out


def seed_items(ctx: ChatContext | None) -> list[dict[str, Any]]:
    """The text history of ``ctx`` as ``session.input`` items (the newest 128 messages that
    fit about 8k tokens). Tool results become developer messages; calls are dropped."""
    items: list[dict[str, Any]] = []
    for item in ctx.items if ctx is not None else []:
        if isinstance(item, ChatMessage):
            text = item.text.strip()
            if not text:
                continue
            if item.role == "assistant":
                items.append(_message("assistant", "output_text", text))
            elif item.role in ("system", "developer"):
                items.append(_message("developer", "input_text", text))
            else:
                items.append(_message("user", "input_text", text))
        elif isinstance(item, FunctionCallOutput) and item.output.strip():
            name = item.name or "tool"
            text = f"Result of the {name} call: {item.output.strip()}"
            items.append(_message("developer", "input_text", text))
    items = items[-_MAX_SEED_MESSAGES:]
    budget = _MAX_SEED_CHARS
    kept: list[dict[str, Any]] = []
    for msg in reversed(items):
        size = len(msg["content"][0]["text"])
        if kept and size > budget:
            break
        kept.append(msg)
        budget -= size
    kept.reverse()
    return kept


def _message(role: str, part: str, text: str) -> dict[str, Any]:
    return {"type": "message", "role": role, "content": [{"type": part, "text": text}]}


def _tool_payload(tool: FunctionTool) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }


# -------------------------------------------------------------------------------- engine
class OpenAILiveSessionEngine(S2SEngine):
    """GPT-Live on one Live session per connection (no rotation; see
    :class:`OpenAILiveEngine`, which wraps this engine and is what ``openai-live`` creates).

    Args:
        model: Live model (default ``gpt-live-1``).
        api_key: API key (default: ``OPENAI_API_KEY``).
        base_url: ``wss://api.openai.com/v1`` (``/live/sessions`` is appended;
            ``http(s)`` is accepted). Default: ``OPENAI_LIVE_BASE_URL`` or OpenAI.
        voice: output voice (``marin`` by default; see :data:`VOICES`) or a custom voice
            ``{"id": ...}``. Fixed for the session: ``Agent(voice=...)`` wins.
        sample_rate: PCM16 rate of both directions, 24000 (default) or 16000.
        delegation: ``"responses"`` (GPT-Live runs a Responses backend with the agent's
            tools) or ``"client"`` (the application answers delegations through
            ``delegation_tool``). Fixed for the session.
        responses_model: backend model of Responses delegation.
        responses_instructions: backend prompt (task rules, tool workflows); the agent's
            instructions go to the voice model. ``EngineOptions.extra
            ["responses_instructions"]`` overrides it per connection.
        web_search: also give the Responses backend the hosted ``web_search`` tool.
        responses: extra ``delegation.responses`` fields (``reasoning``, ``service_tier``,
            ``tool_choice``, ``parallel_tool_calls``, ``max_output_tokens``, ``text``).
        delegation_tool: client delegation: name of the agent tool that answers
            delegations (called with ``request``: the user's words since the previous
            delegation).
        delegation_wait: client delegation: how long to wait for the user's transcript to
            catch up with the delegation before calling the tool (seconds).
        store: ``session.store`` (keep the session for forking / recording download).
        session: extra ``session.start`` fields, deep-merged last.
        headers: extra handshake headers (e.g. ``OpenAI-Safety-Identifier``).
        speech_threshold_db: output level (dBFS) above which agent audio counts as speech.
        response_gap: agent silence (s, no transcript either) that ends a response.
        yield_gap: shorter silence that ends it while the user talks.
        transcript_grace: a response does not end until its transcript has been quiet
            this long (transcripts trail the audio).
        handover_delay: user speech that continues after a response ended is reported
            this long after the end (earlier it would count as a barge-in on its tail).
        preroll: seconds of quiet audio before a speech onset kept in the response.
        report_overlap: also report user speech while the agent speaks, letting the
            session's interruption policy mute the agent (default: the model owns the floor).
        user_vad_threshold_db / user_min_silence: local energy VAD on the user audio.
        keepalive: stream silence while the transport delivers no audio (the session
            timeline runs on input audio: a greeting or an acknowledgement waits for it).
        connect_timeout: WebSocket handshake + ``session.started`` timeout (seconds).
        close_timeout: how long ``aclose()`` waits for ``session.closed`` (final usage).
        expiry_warning: emit ``EngineStatus("expiring")`` this long before ``expires_at``.
    """

    provider = "openai_live"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        voice: str | Mapping[str, Any] | None = None,
        sample_rate: int = 24_000,
        delegation: Delegation = "responses",
        responses_model: str = DEFAULT_RESPONSES_MODEL,
        responses_instructions: str | None = None,
        web_search: bool = False,
        responses: Mapping[str, Any] | None = None,
        delegation_tool: str = "delegate",
        delegation_wait: float = 0.6,
        store: bool | None = None,
        session: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        speech_threshold_db: float = -40.0,
        response_gap: float = 0.64,
        yield_gap: float = 0.24,
        transcript_grace: float = 0.3,
        handover_delay: float = 0.3,
        preroll: float = 0.08,
        report_overlap: bool = False,
        user_vad_threshold_db: float = -40.0,
        user_min_silence: float = 0.3,
        keepalive: bool = True,
        connect_timeout: float = 10.0,
        close_timeout: float = 5.0,
        expiry_warning: float = 120.0,
    ) -> None:
        if sample_rate not in SAMPLE_RATES:
            raise ConfigurationError(f"sample_rate must be one of {SAMPLE_RATES}")
        if delegation not in ("responses", "client"):
            raise ConfigurationError("delegation must be 'responses' or 'client'")
        if response_gap <= 0:
            raise ConfigurationError("response_gap must be > 0")
        super().__init__(
            model=model or DEFAULT_MODEL,
            capabilities=EngineCapabilities(
                native_audio=True,
                server_turn_detection=True,  # the model decides when to speak
                tool_calling=True,
                input_transcription=True,
                output_transcription=True,
                truncation=False,
                full_duplex=True,
                text_input=True,
                tool_mode="delegation",
                max_session_duration=None,  # per session: ``expires_at``
            ),
            input_sample_rate=sample_rate,
            output_sample_rate=sample_rate,
        )
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.headers: dict[str, str] = dict(headers or {})
        if not self.api_key and not any(h.lower() == "authorization" for h in self.headers):
            raise ConfigurationError(
                f"{self.provider}: no API key; pass api_key=... or set OPENAI_API_KEY"
            )
        self.url = live_url(base_url or os.environ.get("OPENAI_LIVE_BASE_URL") or DEFAULT_URL)
        self.voice = voice
        self.delegation: Delegation = delegation
        self.responses_model = responses_model
        self.responses_instructions = responses_instructions
        self.web_search = web_search
        self.responses_overrides: dict[str, Any] = dict(responses or {})
        self.delegation_tool = delegation_tool
        self.delegation_wait = delegation_wait
        self.store = store
        self.session_overrides: dict[str, Any] = dict(session or {})
        self.speech_threshold_db = speech_threshold_db
        self.response_gap = response_gap
        self.yield_gap = min(yield_gap, response_gap)
        self.transcript_grace = transcript_grace
        self.handover_delay = handover_delay
        self.preroll = preroll
        self.report_overlap = report_overlap
        self.user_vad_threshold_db = user_vad_threshold_db
        self.user_min_silence = user_min_silence
        self.keepalive = keepalive
        self.connect_timeout = connect_timeout
        self.close_timeout = close_timeout
        self.expiry_warning = expiry_warning

    def request_headers(self) -> dict[str, str]:
        headers = dict(self.headers)
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")
        return headers

    async def connect(self, options: EngineOptions) -> EngineConnection:
        conn = LiveSessionConnection(self, options)
        try:
            await conn.start()
        except BaseException:
            await conn.aclose()
            raise
        return conn


@register_provider(
    "engine",
    "openai_live",
    description="OpenAI GPT-Live full-duplex speech-to-speech with delegation (Live protocol)",
    default_model=DEFAULT_MODEL,
    models=(DEFAULT_MODEL,),
    env=("OPENAI_API_KEY",),
    aliases=("gpt_live",),
)
class OpenAILiveEngine(RotatingEngine):
    """GPT-Live with transparent session rotation (see the module docs).

    Takes every argument of :class:`OpenAILiveSessionEngine`, plus ``rotation`` (a
    :class:`~voice_agent_next.engines.rotation.RotationPolicy`: when to move to a fresh
    session and how much history to carry over).
    """

    provider = "openai_live"

    def __init__(
        self, *, model: str | None = None, rotation: RotationPolicy | None = None, **kwargs: Any
    ) -> None:
        super().__init__(OpenAILiveSessionEngine(model=model, **kwargs), policy=rotation)

    @property
    def session_engine(self) -> OpenAILiveSessionEngine:
        """The wrapped single-session engine (its configuration)."""
        inner = self.inner
        assert isinstance(inner, OpenAILiveSessionEngine)
        return inner

    async def connect(self, options: EngineOptions) -> EngineConnection:
        conn = OpenAILiveConnection(self, options)
        try:
            await conn.start()
        except BaseException:
            await conn.aclose()
            raise
        return conn


# ---------------------------------------------------------------- rotating connection
class OpenAILiveConnection(RotatingConnection):
    """A GPT-Live conversation over a sequence of Live sessions (rotation, reconnects).

    Adds the Live-specific controls to :class:`RotatingConnection`: input mute, context
    appends and the usage of every session so far.
    """

    def __init__(self, engine: OpenAILiveEngine, options: EngineOptions) -> None:
        super().__init__(engine, options)
        self._input_muted = False
        self._closed_usage = 0.0

    @property
    def session(self) -> LiveSessionConnection | None:
        """The Live session currently carrying the conversation."""
        inner = self.inner
        return inner if isinstance(inner, LiveSessionConnection) else None

    @property
    def session_id(self) -> str | None:
        session = self.session
        return session.session_id if session is not None else None

    @property
    def input_muted(self) -> bool:
        return self._input_muted

    @property
    def usage_seconds(self) -> float:
        """Billed voice duration of every session so far (the latest cumulative
        ``usage.seconds`` of each)."""
        session = self.session
        return self._closed_usage + (session.usage_seconds if session is not None else 0.0)

    async def mute_input(self) -> None:
        """Stop the model from hearing the user (``session.input_audio.mute``); waits for
        the acknowledgement. The agent keeps talking and backend work goes on."""
        self._input_muted = True
        await self._call(lambda c: _as_live(c).mute_input())
        await self._sync_standby()

    async def unmute_input(self) -> None:
        """Let the model hear the user again (``session.input_audio.unmute``)."""
        self._input_muted = False
        await self._call(lambda c: _as_live(c).unmute_input())
        await self._sync_standby()

    async def _sync_standby(self) -> None:
        """A session prepared for the next rotation takes over the current mute state."""
        standby = self._standby
        if standby is None:
            return
        live = _as_live(standby.conn)
        if live.input_muted != self._input_muted:
            with contextlib.suppress(Exception):  # a broken standby is replaced anyway
                await (live.mute_input() if self._input_muted else live.unmute_input())

    async def append_instructions(self, text: str, *, delegation_id: str | None = None) -> None:
        """``session.instructions.append``: trusted instructions (may redirect speech)."""
        await self._call(lambda c: _as_live(c).append("instructions", text, delegation_id))

    async def append_thinking(self, text: str, *, delegation_id: str | None = None) -> None:
        """``session.thinking.append``: facts the model may use, not said right away."""
        await self._call(lambda c: _as_live(c).append("thinking", text, delegation_id))

    async def append_commentary(self, text: str, *, delegation_id: str | None = None) -> None:
        """``session.commentary.append``: content the model says aloud (paraphrased)."""
        await self._call(lambda c: _as_live(c).append("commentary", text, delegation_id))

    async def _open(self, seed: ChatContext, version: int) -> _Link:
        link = await super()._open(seed, version)
        live = _as_live(link.conn)
        live.on_finished = self._on_session_finished
        if self._input_muted:
            await live.mute_input()
        return link

    def _on_session_finished(self, session: LiveSessionConnection) -> None:
        self._closed_usage += session.usage_seconds


def _as_live(conn: EngineConnection) -> LiveSessionConnection:
    assert isinstance(conn, LiveSessionConnection)
    return conn


# ------------------------------------------------------------------------ one session
class _AgentSpeech:
    """One response: a stretch of agent speech."""

    __slots__ = (
        "ended_at",
        "first_audio_at",
        "item_id",
        "last_text",
        "response_id",
        "samples",
        "speech_end",
        "started_at",
        "text_chars",
        "trigger_at",
    )

    def __init__(self, trigger_at: float | None) -> None:
        self.response_id = new_id("resp_")
        self.item_id = new_id("item_")
        self.started_at = now()
        self.trigger_at = trigger_at
        self.first_audio_at: float | None = None
        self.ended_at: float | None = None
        self.speech_end = self.started_at
        """Estimated wall-clock time the response's latest speech finishes playing."""
        self.last_text = 0.0
        self.samples = 0
        self.text_chars = 0


@dataclass
class _Backend:
    """One Responses backend response."""

    response_id: str
    delegation_id: str | None
    started_at: float = field(default_factory=now)
    first_output_at: float | None = None
    pending: set[str] = field(default_factory=set)
    """Function calls still waiting for their output."""
    answered: int = 0
    done: bool = False
    continued: bool = False


@dataclass
class _ClientDelegation:
    delegation_id: str
    offset_ms: float
    created_at: float = field(default_factory=now)


class LiveSessionConnection(EngineConnection):
    """One GPT-Live session on one WebSocket (see the module docs)."""

    engine: OpenAILiveSessionEngine

    def __init__(self, engine: OpenAILiveSessionEngine, options: EngineOptions) -> None:
        super().__init__(engine, options)
        self.engine = engine
        self._e = engine
        self.session_id: str | None = None
        self.expires_at: float | None = None
        """Unix time at which the session expires (from ``session.started``)."""
        self.usage_seconds = 0.0
        """Latest cumulative billed voice duration (``usage.seconds``)."""
        self.context_usage: float | None = None
        """Latest ``context_window.usage_ratio``."""
        self.close_reason: str | None = None
        """``session.closed`` reason, once the session finished."""
        self.session_config: dict[str, Any] = {}
        """The resolved configuration from ``session.started``/``session.updated``."""
        self.input_muted = False
        self.on_finished: Any = None
        """Called with the connection once it closed (usage accounting)."""
        self._tasks = BackgroundTasks("openai-live")
        self._ws: ClientConnection | None = None
        self._reader: asyncio.Task[None] | None = None
        self._ticker: asyncio.Task[None] | None = None
        self._started = asyncio.Event()
        self._finished = asyncio.Event()
        self._startup_error: Exception | None = None
        self._closing = False
        self._event_seq = 0
        self._sent_types: dict[str, str] = {}
        self._acks: dict[str, asyncio.Future[None]] = {}
        # ---- clocks
        self._t0 = now()
        self._stream_pos = 0.0
        """Seconds of audio sent to the session (user audio and keep-alive silence)."""
        self._last_input_at: float | None = None
        self._playout_end = 0.0
        # ---- agent speech
        self._resp: _AgentSpeech | None = None
        self._preroll: deque[AudioFrame] = deque()
        self._muted_until_pause = False
        self._mute_speech_end = 0.0
        self._response_ended_at: float | None = None
        self._input_since_response = 0.0
        self._pcm = PCM16Reassembler()
        # ---- user speech (local VAD) and transcript
        opts = VADOptions(min_speech_duration=0.1, min_silence_duration=engine.user_min_silence)
        self._vad: VADStream = EnergyVAD(
            sample_rate=engine.input_sample_rate,
            threshold_db=engine.user_vad_threshold_db,
            options=opts,
        ).stream()
        self._user_speaking = False
        self._user_reported = False
        self._user_pending = False
        self._user_speech_start: float | None = None
        self._user_speech_end: float | None = None
        self._user_item: str | None = None
        self._user_text: list[str] = []
        self._committed: tuple[str, list[str], float] | None = None
        """(item id, transcript parts, session-timeline ms of the commit)."""
        self._transcript_end_ms = 0.0
        # ---- delegation
        self._request_text: list[str] = []
        """User transcript since the previous client delegation."""
        self._waiting: list[_ClientDelegation] = []
        self._client_calls: set[str] = set()
        self._backends: dict[str, _Backend] = {}
        self._latest_backend: dict[str | None, str] = {}
        self._call_backend: dict[str, str] = {}
        self._warned: set[str] = set()

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        """Open the socket, send ``session.start`` and wait for ``session.started``."""
        e = self._e
        self._ws = await self._open_socket()
        self._reader = asyncio.create_task(self._read(self._ws), name="openai-live-read")
        await self._send({"type": "session.start", "session": self._start_payload()})
        try:
            await asyncio.wait_for(self._started.wait(), e.connect_timeout)
        except TimeoutError:
            raise RealtimeConnectTimeoutError(
                f"{e.provider}: no session.started within {e.connect_timeout:.0f}s",
                provider=e.provider,
            ) from None
        if self._startup_error is not None:
            raise self._startup_error
        self._t0 = now()
        self._ticker = asyncio.create_task(self._tick(), name="openai-live-tick")
        if not self.options.turn_detection:
            self._warn_once(
                "turns", "GPT-Live takes turns by itself; turn_detection=False is ignored"
            )

    async def aclose(self) -> None:
        if self.closed:
            return
        self._closing = True
        ws = self._ws
        if (
            ws is not None
            and self._started.is_set()
            and self._startup_error is None
            and not self._finished.is_set()
        ):
            # graceful close: the final usage arrives with session.closed
            with contextlib.suppress(Exception):
                await ws.send(json.dumps({"type": "session.close", "event_id": self._next_id()}))
                await asyncio.wait_for(self._finished.wait(), self._e.close_timeout)
        current = asyncio.current_task()
        await cancel_and_wait(*[t for t in (self._reader, self._ticker) if t and t is not current])
        await self._tasks.cancel_all()
        self._ws = None
        if ws is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(ws.close(), 2.0)
        for fut in self._acks.values():
            if not fut.done():
                fut.cancel()
        self._vad.close()
        self._end_response("incomplete")
        if self.on_finished is not None:
            with contextlib.suppress(Exception):
                self.on_finished(self)
        await super().aclose()

    async def _open_socket(self) -> ClientConnection:
        e = self._e
        kwargs: dict[str, Any] = {}
        if _CONNECT_ACCEPTS_PROXY and _is_loopback(e.url):
            kwargs["proxy"] = None  # never route a local server through a system proxy
        try:
            return await ws_connect(
                e.url,
                additional_headers=e.request_headers(),
                open_timeout=e.connect_timeout,
                max_size=_MAX_MESSAGE_SIZE,
                compression=None,
                close_timeout=2.0,
                **kwargs,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise _handshake_error(exc, e.provider, e.url) from exc

    def _start_payload(self) -> dict[str, Any]:
        e, opts = self._e, self.options
        voice = opts.voice or e.voice or DEFAULT_VOICE
        session: dict[str, Any] = {
            "model": e.model,
            "audio": {
                "format": {"type": "audio/pcm", "rate": e.input_sample_rate},
                "output": {"voice": voice},
            },
            "delegation": self._delegation_payload(opts.tools),
        }
        if opts.instructions.strip():
            session["instructions"] = opts.instructions
        items = seed_items(opts.chat_ctx)
        if items:
            session["input"] = items
        if e.store is not None:
            session["store"] = e.store
        extra = opts.extra.get("session")
        return _deep_merge(
            session, e.session_overrides, extra if isinstance(extra, Mapping) else None
        )

    def _delegation_payload(self, tools: list[FunctionTool]) -> dict[str, Any]:
        e = self._e
        if e.delegation == "client":
            names = {t.name for t in tools}
            if e.delegation_tool not in names:
                self._warn_once(
                    "delegate",
                    f"client delegation: the agent has no {e.delegation_tool!r} tool, so "
                    "delegated requests fail (add one or use delegation='responses')",
                )
            return {"type": "client"}
        responses: dict[str, Any] = {"model": e.responses_model}
        instructions = self.options.extra.get("responses_instructions", e.responses_instructions)
        if instructions:
            responses["instructions"] = instructions
        backend_tools = [_tool_payload(t) for t in tools]
        if e.web_search:
            backend_tools.append({"type": "web_search"})
        if backend_tools:
            responses["tools"] = backend_tools
        return {"type": "responses", "responses": _deep_merge(responses, e.responses_overrides)}

    # --------------------------------------------------------------------- sending
    def _next_id(self, etype: str = "") -> str:
        self._event_seq += 1
        event_id = f"evt_{self._event_seq:06d}"
        if etype:
            self._sent_types[event_id] = etype
            while len(self._sent_types) > _MAX_TRACKED:
                del self._sent_types[next(iter(self._sent_types))]
        return event_id

    async def _send(self, event: dict[str, Any]) -> str | None:
        """Send a client event (an ``event_id`` is added); ``None`` if the socket is gone."""
        ws = self._ws
        if ws is None or self._finished.is_set():
            return None
        event_id = event.setdefault("event_id", self._next_id(str(event.get("type", ""))))
        try:
            await ws.send(json.dumps(event))
        except ConnectionClosed:
            return None  # the reader notices and reports the drop
        return str(event_id)

    async def _send_audio(self, frame: AudioFrame) -> None:
        self._last_input_at = now()
        self._input_since_response += frame.duration
        # muted: the model hears nothing, and neither does the VAD (its clock keeps going)
        self._track_user(
            AudioFrame(bytes(len(frame.data)), frame.sample_rate) if self.input_muted else frame
        )
        await self._push_audio(frame)

    async def _push_audio(self, frame: AudioFrame) -> None:
        if self._ws is None or not self._started.is_set() or self._finished.is_set():
            return
        self._stream_pos += frame.duration
        with contextlib.suppress(ConnectionClosed):
            await self._ws.send(
                json.dumps({"type": "session.input_audio.append", "audio": frame.to_base64()})
            )

    async def append(
        self,
        kind: Literal["instructions", "thinking", "commentary"],
        text: str,
        delegation_id: str | None = None,
    ) -> None:
        """``session.<kind>.append`` (split into several appends above ~500 tokens)."""
        for piece in _chunks(text):
            await self._send(
                {
                    "type": f"session.{kind}.append",
                    "delegation_id": delegation_id,
                    "content": piece,
                }
            )

    async def _acknowledged(self, etype: str) -> None:
        event_id = self._next_id(etype)
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._acks[event_id] = fut
        try:
            if await self._send({"type": etype, "event_id": event_id}) is None:
                raise ProviderConnectionError(
                    f"{self._e.provider}: cannot send {etype}: the session is closed",
                    provider=self._e.provider,
                )
            await asyncio.wait_for(fut, self._e.connect_timeout)
        finally:
            self._acks.pop(event_id, None)

    async def mute_input(self) -> None:
        """``session.input_audio.mute``; returns once the server acknowledged it."""
        self.input_muted = True
        self._end_user_speech()
        await self._acknowledged("session.input_audio.mute")

    async def unmute_input(self) -> None:
        """``session.input_audio.unmute``; returns once the server acknowledged it."""
        await self._acknowledged("session.input_audio.unmute")
        self.input_muted = False

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
        """Commentary (said aloud, paraphrased) or, without ``respond``, thinking."""
        await self.append("commentary" if respond else "thinking", text)

    async def create_response(self, *, instructions: str | None = None) -> None:
        await self.append("instructions", instructions or RESPOND_INSTRUCTIONS)

    async def say(self, text: str) -> None:
        await self.append("instructions", SAY_INSTRUCTIONS.format(text=text))

    async def cancel_response(self) -> None:
        """End the current response as cancelled and mute the agent until its next pause
        (the Live protocol has no cancel: the model yields to the user by itself)."""
        if self._resp is None:
            return
        self._muted_until_pause = True
        self._mute_speech_end = max(now(), self._resp.speech_end)
        self._end_response("cancelled")

    async def send_tool_output(self, output: FunctionCallOutput, *, respond: bool = True) -> None:
        await self.send_async_tool_output(output, scheduling="when_idle" if respond else "silent")

    async def send_async_tool_output(
        self, output: FunctionCallOutput, *, scheduling: ToolScheduling = "when_idle"
    ) -> None:
        """Return a delegated result: a Responses function output (the backend continues
        once every call of its response has one), or, for a client delegation, commentary
        (thinking when ``scheduling="silent"``) tagged with the delegation id."""
        call_id = output.call_id
        backend_id = self._call_backend.pop(call_id, None)
        if backend_id is not None:
            await self._send(
                {
                    "type": "response.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": output.output,
                    },
                }
            )
            backend = self._backends.get(backend_id)
            if backend is not None:
                backend.pending.discard(call_id)
                backend.answered += 1
                await self._continue_backend(backend)
            return
        text = output.output.strip() or "Done."
        if output.is_error:
            text = f"The task failed: {text}"
        kind: Literal["thinking", "commentary"] = (
            "thinking" if scheduling == "silent" else "commentary"
        )
        if call_id in self._client_calls:
            self._client_calls.discard(call_id)
            await self.append(kind, text, call_id)
            return
        # a result this session did not ask for (e.g. from before a rotation)
        name = output.name or "a background task"
        await self.append(kind, f"Result of {name}: {text}")

    async def _continue_backend(self, backend: _Backend) -> None:
        if backend.done and not backend.pending and backend.answered and not backend.continued:
            backend.continued = True
            await self._send({"type": "response.create"})

    async def update(
        self, *, instructions: str | None = None, tools: list[FunctionTool] | None = None
    ) -> None:
        """Instructions are appended (the startup prompt is fixed); tools update the
        Responses backend (``session.update``)."""
        if instructions is not None:
            self.options.instructions = instructions
            await self.append("instructions", UPDATE_INSTRUCTIONS.format(text=instructions))
        if tools is not None:
            self.options.tools = list(tools)
            if self._e.delegation == "responses":
                delegation = self._delegation_payload(list(tools))
                await self._send({"type": "session.update", "session": {"delegation": delegation}})

    # ------------------------------------------------------------------ receiving
    async def _read(self, ws: ClientConnection) -> None:
        reason = "closed by the server"
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                try:
                    event = json.loads(raw)
                except ValueError:
                    logger.warning("%s: ignoring a non-JSON message", self._e.provider)
                    continue
                if isinstance(event, dict):
                    try:
                        self._dispatch(event)
                    except Exception:
                        logger.exception(
                            "%s: failed to handle %s", self._e.provider, event.get("type")
                        )
        except ConnectionClosed as exc:
            reason = _close_reason(exc)
        except Exception as exc:
            reason = f"failed: {exc!r}"
        self._ws = None
        if not self._started.is_set():
            if self._startup_error is None:
                self._startup_error = ProviderConnectionError(
                    f"{self._e.provider}: connection {reason} during setup",
                    provider=self._e.provider,
                )
            self._started.set()
            return
        if self._closing or self._finished.is_set():
            return
        error = ProviderConnectionError(
            f"{self._e.provider}: connection {reason} (no session.closed)",
            provider=self._e.provider,
        )
        self._fail(error)

    def _fail(self, error: Exception) -> None:
        if self.closed:
            return
        self._emit(EngineErrorEvent(error=error, recoverable=False))
        self._tasks.spawn(self.aclose(), name="openai-live-close")

    def _dispatch(self, ev: dict[str, Any]) -> None:
        etype = ev.get("type")
        handler = self._handlers.get(str(etype))
        if handler is not None:
            handler(self, ev)
        elif etype == "info":
            logger.info("%s: %s", self._e.provider, ev.get("message"))
        else:
            logger.debug("%s: unhandled event %s", self._e.provider, etype)

    def _on_started(self, ev: dict[str, Any]) -> None:
        session = ev.get("session") if isinstance(ev.get("session"), dict) else {}
        assert isinstance(session, dict)
        self.session_config = session
        self.session_id = session.get("id")
        expires = session.get("expires_at")
        if isinstance(expires, (int, float)):
            self.expires_at = float(expires)
            self._tasks.spawn(self._expiry_notice(float(expires)), name="openai-live-expiry")
        self._started.set()

    async def _expiry_notice(self, expires_at: float) -> None:
        await asyncio.sleep(max(0.0, expires_at - time.time() - self._e.expiry_warning))
        left = max(0.0, expires_at - time.time())
        self._emit(EngineStatus(status="expiring", detail="session expires", time_left=left))

    def _on_updated(self, ev: dict[str, Any]) -> None:
        session = ev.get("session")
        if isinstance(session, dict):
            self.session_config = session

    def _on_ack(self, ev: dict[str, Any]) -> None:
        fut = self._acks.get(str(ev.get("client_event_id")))
        if fut is not None and not fut.done():
            fut.set_result(None)

    def _on_error(self, ev: dict[str, Any]) -> None:
        err = ev.get("error")
        err = err if isinstance(err, Mapping) else {"message": str(err)}
        client_id = err.get("client_event_id") or ev.get("client_event_id")
        context = self._sent_types.get(str(client_id)) if client_id else None
        error = _live_error(err, self._e.provider, context)
        fut = self._acks.get(str(client_id)) if client_id else None
        if fut is not None and not fut.done():
            fut.set_exception(error)
            return
        if not self._started.is_set():
            self._startup_error = error
            self._started.set()
            return
        logger.warning("%s", error)
        self._emit(EngineErrorEvent(error=error, recoverable=True))

    def _on_usage(self, ev: dict[str, Any]) -> None:
        usage = ev.get("usage")
        if isinstance(usage, Mapping) and isinstance(usage.get("seconds"), (int, float)):
            self.usage_seconds = float(usage["seconds"])
        window = ev.get("context_window")
        if isinstance(window, Mapping) and isinstance(window.get("usage_ratio"), (int, float)):
            self.context_usage = float(window["usage_ratio"])

    def _on_closed(self, ev: dict[str, Any]) -> None:
        self._on_usage(ev)
        reason = str(ev.get("reason") or "unknown")
        self.close_reason = reason
        self._finished.set()
        self._end_response("incomplete")
        if self._closing:
            return
        provider = self._e.provider
        if reason in _RETRYABLE_CLOSE:  # the conversation continues on a new session
            error: ProviderError = ProviderConnectionError(
                f"{provider}: the session closed ({reason})", provider=provider
            )
        elif reason == "content":
            error = ProviderError(
                f"{provider}: the session was ended by a safety filter (content)",
                provider=provider,
            )
        else:
            error = ProviderError(f"{provider}: the session ended ({reason})", provider=provider)
        self._fail(error)

    # ------------------------------------------------------------------ agent speech
    def _on_audio_delta(self, ev: dict[str, Any]) -> None:
        delta = ev.get("delta")
        if not isinstance(delta, str) or not delta:
            return
        data = self._pcm.push(base64.b64decode(delta))
        if not data:
            return
        frame = AudioFrame(data, self._e.output_sample_rate)
        t = now()
        self._playout_end = max(self._playout_end, t) + frame.duration
        loud = frame.dbfs() >= self._e.speech_threshold_db
        if self._muted_until_pause:
            if loud:
                self._mute_speech_end = self._playout_end
            elif self._playout_end - self._mute_speech_end >= self._e.response_gap:
                self._muted_until_pause = False  # the model paused: it is heard again
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
            resp.speech_end = self._playout_end
        self._check_end()

    def _keep_preroll(self, frame: AudioFrame) -> None:
        self._preroll.append(frame)
        total = sum(f.duration for f in self._preroll)
        while self._preroll and total - self._preroll[0].duration >= self._e.preroll - 1e-9:
            total -= self._preroll.popleft().duration

    def _forward(self, resp: _AgentSpeech, frame: AudioFrame) -> None:
        if resp.first_audio_at is None:
            resp.first_audio_at = now()
        resp.samples += frame.samples_per_channel
        self._emit(ResponseAudio(response_id=resp.response_id, item_id=resp.item_id, frame=frame))

    def _on_output_transcript(self, ev: dict[str, Any]) -> None:
        delta = ev.get("delta")
        if not isinstance(delta, str) or not delta or self._muted_until_pause:
            return
        resp = self._resp or self._begin_response()
        t = now()
        resp.last_text = t
        resp.speech_end = max(resp.speech_end, t)
        resp.text_chars += len(delta)
        self._emit(ResponseText(response_id=resp.response_id, item_id=resp.item_id, delta=delta))

    def _check_end(self) -> None:
        resp = self._resp
        if resp is None:
            return
        t = now()
        gap = self._e.yield_gap if self._user_speaking else self._e.response_gap
        if t >= resp.speech_end + gap and t - resp.last_text >= self._e.transcript_grace:
            self._end_response("completed")

    def _begin_response(self) -> _AgentSpeech:
        trigger: float | None = None
        if self._user_pending:
            self._user_pending = False
            if not self._user_speaking and self._user_speech_end is not None:
                trigger = self.audio_time_to_wall(self._user_speech_end)
            if self._user_reported:
                # the agent takes the floor while the user still talks: from here on the
                # user's speech is an overlap (and a later stop must not end *this* turn)
                self._user_reported = False
                self._emit(InputSpeechStopped(audio_time=self.input_audio_time))
            self._commit_user_turn()
        resp = _AgentSpeech(trigger)
        self._resp = resp
        self._emit(ResponseStarted(response_id=resp.response_id))
        return resp

    def _end_response(self, status: ResponseStatus) -> None:
        resp, self._resp = self._resp, None
        if resp is None:
            return
        resp.ended_at = now()
        self._emit(ResponseDone(response_id=resp.response_id, status=status))
        ttfb = None
        if resp.trigger_at is not None and resp.first_audio_at is not None:
            ttfb = max(0.0, resp.first_audio_at - resp.trigger_at)
        self._e.emit(
            "metrics",
            EngineMetrics(
                provider=self._e.provider,
                model=self._e.model,
                response_id=resp.response_id,
                ttfb=ttfb,
                duration=resp.ended_at - resp.started_at,
                cancelled=status == "cancelled",
            ),
        )
        self._input_since_response = 0.0
        self._preroll.clear()
        self._response_ended_at = now()

    # ------------------------------------------------------------------- user side
    def _track_user(self, frame: AudioFrame) -> None:
        for ev in self._vad.push_audio(frame):
            if ev.type == VADEventType.START_OF_SPEECH:
                self._user_speaking = True
                self._user_pending = True
                self._user_speech_start = max(0.0, ev.audio_time - ev.speech_duration)
            elif ev.type == VADEventType.END_OF_SPEECH:
                self._user_speaking = False
                end = max(0.0, ev.audio_time - ev.silence_duration)
                self._user_speech_end = end
                if self._user_reported:
                    self._user_reported = False
                    self._emit(InputSpeechStopped(audio_time=end))
        if self._user_speaking and not self._user_reported and self._floor_free():
            self._user_reported = True
            self._emit(InputSpeechStarted(audio_time=self._user_speech_start))

    def _end_user_speech(self) -> None:
        """Muted: the model no longer hears the user (an open utterance ends here)."""
        self._user_speaking = False
        if self._user_reported:
            self._user_reported = False
            self._emit(InputSpeechStopped(audio_time=self.input_audio_time))

    def _floor_free(self) -> bool:
        if self._e.report_overlap:
            return True
        if self._resp is not None:
            return False
        ended = self._response_ended_at
        return ended is None or now() - ended >= self._e.handover_delay

    def _commit_user_turn(self) -> None:
        item_id = self._user_item or new_id("item_")
        parts, self._user_text, self._user_item = self._user_text, [], None
        self._emit(InputCommitted(item_id=item_id))
        text = "".join(parts).strip()
        if text:
            self._emit(InputTranscript(item_id=item_id, text=text, is_final=True))
        self._committed = (item_id, parts, self._timeline_ms())

    def _timeline_ms(self) -> float:
        return self._stream_pos * 1000.0

    def _on_input_transcript(self, ev: dict[str, Any]) -> None:
        delta = ev.get("delta")
        if not isinstance(delta, str) or not delta:
            return
        start_ms = ev.get("start_ms")
        end_ms = ev.get("end_ms")
        if isinstance(end_ms, (int, float)):
            self._transcript_end_ms = max(self._transcript_end_ms, float(end_ms))
        self._request_text.append(delta)
        committed = self._committed
        if committed is not None and isinstance(start_ms, (int, float)) and start_ms < committed[2]:
            # speech from before the agent took the floor: completes the committed turn
            item_id, parts, _ = committed
            parts.append(delta)
            self._emit(InputTranscript(item_id=item_id, text="".join(parts).strip(), is_final=True))
        else:
            if self._user_item is None:
                self._user_item = new_id("item_")
            self._user_text.append(delta)
            self._user_pending = True
            text = "".join(self._user_text).strip()
            if text:
                self._emit(InputTranscript(item_id=self._user_item, text=text, is_final=False))
        self._flush_delegations()

    # ------------------------------------------------------------------ delegation
    def _on_delegation(self, ev: dict[str, Any]) -> None:
        delegation = ev.get("delegation")
        if not isinstance(delegation, Mapping) or not delegation.get("id"):
            return
        delegation_id = str(delegation["id"])
        if delegation.get("target") == "responses":
            response_id = delegation.get("response_id")
            if isinstance(response_id, str):
                self._latest_backend[delegation_id] = response_id
                self._backends.setdefault(response_id, _Backend(response_id, delegation_id))
            return
        offset = ev.get("offset_ms")
        self._waiting.append(
            _ClientDelegation(
                delegation_id, float(offset) if isinstance(offset, (int, float)) else 0.0
            )
        )
        self._flush_delegations()

    def _flush_delegations(self) -> None:
        """Call the delegation tool once the user's transcript has caught up with the
        delegation (or ``delegation_wait`` passed)."""
        t = now()
        while self._waiting:
            first = self._waiting[0]
            caught_up = self._transcript_end_ms >= first.offset_ms
            if not caught_up and t - first.created_at < self._e.delegation_wait:
                return
            self._waiting.pop(0)
            request = "".join(self._request_text).strip()
            self._request_text.clear()
            call = FunctionCall(
                name=self._e.delegation_tool,
                arguments=json.dumps({"request": request}),
                call_id=first.delegation_id,
            )
            self._client_calls.add(call.call_id)
            response_id = self._resp.response_id if self._resp is not None else first.delegation_id
            self._emit(ResponseToolCall(response_id=response_id, call=call))

    def _on_response_event(self, ev: dict[str, Any]) -> None:
        inner = ev.get("event")
        if not isinstance(inner, Mapping):
            return
        delegation_id = ev.get("delegation_id")
        delegation_id = delegation_id if isinstance(delegation_id, str) else None
        etype = str(inner.get("type"))
        response = inner.get("response")
        response = response if isinstance(response, Mapping) else {}
        if etype == "response.created":
            rid = str(response.get("id") or new_id("resp_"))
            self._backends.setdefault(rid, _Backend(rid, delegation_id))
            self._latest_backend[delegation_id] = rid
            self._bound_backends()
        elif etype in ("response.output_text.delta", "response.output_item.added"):
            backend = self._backend_for(delegation_id, response)
            if backend is not None and backend.first_output_at is None:
                backend.first_output_at = now()
        elif etype == "response.output_item.done":
            item = inner.get("item")
            if isinstance(item, Mapping) and item.get("type") == "function_call":
                self._on_backend_call(delegation_id, item)
        elif etype in (
            "response.completed",
            "response.done",
            "response.failed",
            "response.incomplete",
            "response.cancelled",
        ):
            self._on_backend_done(delegation_id, etype, response)
        elif etype == "error":
            err = inner.get("error") if isinstance(inner.get("error"), Mapping) else inner
            assert isinstance(err, Mapping)
            error = _live_error(err, self._e.provider, "a delegated response")
            self._emit(EngineErrorEvent(error=error, recoverable=True))

    def _backend_for(
        self, delegation_id: str | None, response: Mapping[str, Any]
    ) -> _Backend | None:
        rid = response.get("id") or self._latest_backend.get(delegation_id)
        if rid is None and self._latest_backend:
            rid = next(reversed(self._latest_backend.values()))
        return self._backends.get(str(rid)) if rid is not None else None

    def _bound_backends(self) -> None:
        while len(self._backends) > _MAX_TRACKED:
            del self._backends[next(iter(self._backends))]
        while len(self._latest_backend) > _MAX_TRACKED:
            del self._latest_backend[next(iter(self._latest_backend))]

    def _on_backend_call(self, delegation_id: str | None, item: Mapping[str, Any]) -> None:
        backend = self._backend_for(delegation_id, {})
        if backend is None:  # no response.created seen: track the call anyway
            rid = new_id("resp_")
            backend = self._backends[rid] = _Backend(rid, delegation_id)
            self._latest_backend[delegation_id] = rid
        call_id = str(item.get("call_id") or new_id("call_"))
        if call_id in self._call_backend or call_id in backend.pending:
            return
        call = FunctionCall(
            name=str(item.get("name") or ""),
            arguments=str(item.get("arguments") or "{}"),
            call_id=call_id,
        )
        backend.pending.add(call_id)
        self._call_backend[call_id] = backend.response_id
        self._emit(ResponseToolCall(response_id=backend.response_id, call=call))

    def _on_backend_done(
        self, delegation_id: str | None, etype: str, response: Mapping[str, Any]
    ) -> None:
        backend = self._backend_for(delegation_id, response)
        if backend is None:
            return
        backend.done = True
        status = str(response.get("status") or etype.removeprefix("response."))
        usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
        assert isinstance(usage, Mapping)
        details = usage.get("input_tokens_details")
        cached = details.get("cached_tokens", 0) if isinstance(details, Mapping) else 0
        self._e.emit(
            "metrics",
            LLMMetrics(
                provider=self._e.provider,
                model=str(response.get("model") or self._e.responses_model),
                request_id=backend.response_id,
                ttft=(
                    backend.first_output_at - backend.started_at
                    if backend.first_output_at is not None
                    else None
                ),
                duration=now() - backend.started_at,
                prompt_tokens=int(usage.get("input_tokens") or 0),
                completion_tokens=int(usage.get("output_tokens") or 0),
                cached_tokens=int(cached or 0),
                error=None if status in ("completed", "done") else status,
            ),
        )
        if status in ("failed", "incomplete", "cancelled"):
            err = response.get("error")
            detail = err.get("message") if isinstance(err, Mapping) else None
            msg = f"{self._e.provider}: a delegated response ended {status}"
            error = ProviderError(
                msg + (f": {detail}" if detail else ""), provider=self._e.provider
            )
            self._emit(EngineErrorEvent(error=error, recoverable=True))
        self._tasks.spawn(self._continue_backend(backend), name="openai-live-continue")

    # ----------------------------------------------------------------------- ticker
    async def _tick(self) -> None:
        """Response ends and unmutes without audio, delegation deadlines, keep-alive."""
        silence = AudioFrame.silence(_KEEPALIVE_CHUNK, self._e.input_sample_rate)
        while not self.closed:
            await asyncio.sleep(_TICK)
            t = now()
            self._check_end()
            if self._muted_until_pause and t >= self._mute_speech_end + self._e.response_gap:
                self._muted_until_pause = False
            if self._waiting:
                self._flush_delegations()
            if not self._e.keepalive or self._finished.is_set():
                continue
            if self._last_input_at is not None and t - self._last_input_at < _KEEPALIVE_IDLE:
                continue
            behind = (t - self._t0) - self._stream_pos
            while behind >= _KEEPALIVE_CHUNK / 2:
                await self._push_audio(silence)
                behind -= _KEEPALIVE_CHUNK

    _handlers: dict[str, Any] = {
        "session.started": _on_started,
        "session.updated": _on_updated,
        "session.closed": _on_closed,
        "session.usage.updated": _on_usage,
        "session.input_audio.muted": _on_ack,
        "session.input_audio.unmuted": _on_ack,
        "session.output_audio.delta": _on_audio_delta,
        "session.output_transcript.delta": _on_output_transcript,
        "session.input_transcript.delta": _on_input_transcript,
        "session.delegation.created": _on_delegation,
        "response.event": _on_response_event,
        "error": _on_error,
    }
