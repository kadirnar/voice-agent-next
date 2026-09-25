"""Engine session rotation and reconnection with context carry-over.

Provider sessions are finite (OpenAI Realtime: 60 min, Gemini Live: ~10 min connections,
Nova Sonic: 8 min) and connections drop. This module holds the engine-agnostic pieces
that make both invisible to the user (see ``docs/concepts/session-rotation.md``):

* :class:`RotationPolicy` — when to rotate proactively, how long to wait for a quiet
  moment, backoff, audio buffering and the history carry-over strategy;
* :class:`HistoryCarryOver` strategies — :class:`TruncateHistory` (default) and
  :class:`SummarizeHistory` — that fit the conversation into a fresh session;
* :class:`ConversationRecorder` — the conversation as the user *heard* it, rebuilt from
  the engine's own events (transcripts, truncation, tool calls), so a fresh session can
  be re-seeded without the session's help;
* :class:`QuietTracker` — "nobody is talking, nothing is playing, no tool is running";
* :class:`AudioReplay` / :class:`AudioBuffer` — recent user audio that the new session
  may lack, and audio held back while the connection is switched;
* :class:`RotatingEngine` — wraps *any* :class:`~voice_agent_next.engine.S2SEngine` without
  native rotation (make-before-break: the next connection is opened, seeded and ready
  before the switch, which happens at a quiet moment).

Native engines use the same pieces inside their connections: OpenAI Realtime rotates its
WebSocket session (``providers/openai/realtime.py``) and Gemini Live complements its
resumption handles with the carry-over policy for fresh sessions
(``providers/google/live.py``). All of them report the same ``EngineStatus`` sequence
(``expiring`` -> ``reconnecting`` -> ``reconnected``/``resumed``) and one
:class:`~voice_agent_next.metrics.RotationMetrics` per switch.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

from ..audio.frame import AudioFrame
from ..chat import (
    ChatContext,
    ChatItem,
    ChatMessage,
    ChatRole,
    FunctionCall,
    FunctionCallOutput,
)
from ..engine import EngineConnection, EngineOptions, S2SEngine
from ..errors import AuthenticationError, ConfigurationError, ProviderConnectionError, ProviderError
from ..events import (
    EngineErrorEvent,
    EngineEvent,
    EngineStatus,
    InputCommitted,
    InputSpeechStarted,
    InputSpeechStopped,
    InputTranscript,
    ResponseAudio,
    ResponseDone,
    ResponseStarted,
    ResponseText,
    ResponseToolCall,
    ToolCallCancelled,
)
from ..llm import LLM
from ..metrics import RotationMetrics
from ..tools import FunctionTool, ToolScheduling
from ..utils.aio import BackgroundTasks, cancel_and_wait
from ..utils.clock import now
from ..utils.log import logger

__all__ = [
    "AudioBuffer",
    "AudioReplay",
    "ConversationRecorder",
    "HistoryCarryOver",
    "QuietTracker",
    "RotatingConnection",
    "RotatingEngine",
    "RotationPolicy",
    "SummarizeHistory",
    "TruncateHistory",
    "backoff_delays",
]

_REQUEST_TTL = 30.0
"""A requested response that never started stops blocking a rotation after this long."""
_MONITOR_INTERVAL = 0.05


# ------------------------------------------------------------------------ carry-over
@runtime_checkable
class HistoryCarryOver(Protocol):
    """Fits the conversation into a fresh provider session.

    Called with the recorded conversation (user and agent text as *heard*, tool calls and
    results; instructions and tools are carried separately) and returns the items to seed
    the new session with. Any ``async def f(history) -> ChatContext`` works.
    """

    def __call__(self, history: ChatContext) -> Awaitable[ChatContext]: ...


def _has_content(item: ChatItem) -> bool:
    return not isinstance(item, ChatMessage) or bool(item.text.strip())


def _item_chars(item: ChatItem) -> int:
    if isinstance(item, ChatMessage):
        return len(item.text)
    if isinstance(item, FunctionCall):
        return len(item.name) + len(item.arguments)
    if isinstance(item, FunctionCallOutput):
        return len(item.output)
    return 0


@dataclass(frozen=True)
class TruncateHistory:
    """Keep the most recent items that fit ``max_items`` and ``max_chars`` (default).

    Leading system/developer messages are always kept; empty messages are dropped; a tool
    result is never kept without its call (and ``include_tools=False`` drops tool items).
    """

    max_items: int = 100
    max_chars: int = 24_000
    """Rough text budget (about 6k tokens) — far below every provider's context window, so
    re-seeding stays fast and cheap."""
    include_tools: bool = True

    async def __call__(self, history: ChatContext) -> ChatContext:
        return self.select(history)

    def select(self, history: ChatContext) -> ChatContext:
        """Synchronous version of the strategy."""
        items = [i for i in history.items if _has_content(i)]
        if not self.include_tools:
            items = [i for i in items if isinstance(i, ChatMessage)]
        head = [
            i for i in items if isinstance(i, ChatMessage) and i.role in ("system", "developer")
        ]
        rest = [i for i in items if not any(i is h for h in head)]
        budget = self.max_chars - sum(_item_chars(i) for i in head)
        tail: list[ChatItem] = []
        for item in reversed(rest):
            size = _item_chars(item)
            if len(tail) >= self.max_items or (tail and size > budget):
                break
            tail.append(item)
            budget -= size
        tail.reverse()
        kept_calls = {i.call_id for i in tail if isinstance(i, FunctionCall)}
        tail = [i for i in tail if not isinstance(i, FunctionCallOutput) or i.call_id in kept_calls]
        return ChatContext([*head, *tail])


_SUMMARY_PROMPT = (
    "Summarize the earlier part of this voice conversation for the assistant that continues "
    "it. Keep names, numbers, decisions, open questions and anything the user asked to "
    "remember. Write at most {words} words of plain prose, no preamble."
)


@dataclass
class SummarizeHistory:
    """Summarize older turns with an LLM; keep the most recent ones verbatim.

    The summary is seeded as a system message (``metadata["carry_over"] == "summary"``)
    followed by the last ``keep_last`` items. Summaries are cached and extended
    incrementally, so repeated rotations only summarize what is new. When the LLM fails or
    times out the strategy falls back to ``fallback`` (plain truncation).
    """

    llm: LLM
    keep_last: int = 12
    max_words: int = 200
    min_items: int = 20
    """Only summarize once the history has more items than this."""
    timeout: float = 10.0
    prompt: str = _SUMMARY_PROMPT
    fallback: TruncateHistory = field(default_factory=TruncateHistory)
    _summary: str = field(default="", init=False, repr=False)
    _summarized: int = field(default=0, init=False, repr=False)
    """Number of leading (non-system) items covered by ``_summary``."""

    async def __call__(self, history: ChatContext) -> ChatContext:
        items = [i for i in history.items if _has_content(i)]
        head = [
            i for i in items if isinstance(i, ChatMessage) and i.role in ("system", "developer")
        ]
        rest = [i for i in items if not any(i is h for h in head)]
        if len(rest) <= self.min_items:
            return self.fallback.select(history)
        split = max(0, len(rest) - self.keep_last)
        while split > 0 and isinstance(rest[split], FunctionCallOutput):
            split -= 1  # never separate a tool result from its call
        if split < self._summarized:  # the history shrank (a new conversation): start over
            self._summary, self._summarized = "", 0
        try:
            if split > self._summarized:
                summary = await asyncio.wait_for(
                    self._summarize(rest[self._summarized : split]), self.timeout
                )
                self._summary, self._summarized = summary, split
        except Exception as exc:
            logger.warning("history summary failed (%s); truncating instead", exc)
            return self.fallback.select(history)
        out = ChatContext(head)
        if self._summary:
            out.add_message(
                "system",
                f"Summary of the conversation so far: {self._summary}",
                metadata={"carry_over": "summary"},
            )
        out.items.extend(rest[split:])
        return out

    async def _summarize(self, items: list[ChatItem]) -> str:
        lines: list[str] = []
        if self._summary:
            lines.append(f"Earlier summary: {self._summary}")
        for item in items:
            if isinstance(item, ChatMessage):
                lines.append(f"{item.role}: {item.text}")
            elif isinstance(item, FunctionCall):
                lines.append(f"tool call {item.name}({item.arguments})")
            elif isinstance(item, FunctionCallOutput):
                lines.append(f"tool result: {item.output}")
        ctx = ChatContext()
        ctx.add_message("system", self.prompt.format(words=self.max_words))
        ctx.add_message("user", "\n".join(lines))
        result = await self.llm.chat(ctx).collect()
        text = result.text.strip()
        if not text:
            raise ProviderError("the summarizer returned no text")
        return text


# ------------------------------------------------------------------------- policy
@dataclass(frozen=True)
class RotationPolicy:
    """When and how an engine connection moves to a new provider connection.

    Proactive rotation starts ``lead`` seconds before the provider's session limit
    (``EngineCapabilities.max_session_duration``, an ``expires_at``, or a ``goAway``) — or
    after ``rotate_after`` seconds when set. The next connection is prepared right away
    and the switch waits for a quiet moment (``quiet_period`` without speech, playback,
    pending responses or running tools); it is forced ``force_margin`` seconds before the
    limit.
    """

    rotate_after: float | None = None
    """Rotate after this connection age (seconds) — overrides the limit-based schedule."""
    lead: float = 300.0
    """Start looking for a quiet moment this long before the limit (at most half of it)."""
    force_margin: float = 10.0
    """Switch even if the conversation is busy this long before the limit."""
    quiet_period: float = 0.5
    """How long the conversation must have been quiet before a planned switch."""
    replay: float = 1.0
    """Seconds of recent user audio (not yet committed as a turn) re-sent to the new
    connection, so speech the old one had only started to hear is not lost."""
    max_buffered_audio: float = 30.0
    """Cap on user audio held back during a switch (older audio is dropped beyond it)."""
    max_reconnect_attempts: int = 3
    """Connection attempts after a drop (0 = never reconnect)."""
    backoff: float = 0.5
    """Delay before the first reconnect attempt (doubles per attempt)."""
    max_backoff: float = 10.0
    carry_over: HistoryCarryOver = field(default_factory=TruncateHistory)
    proactive: bool = True
    """``False``: only reconnect after failures, never rotate ahead of the limit."""

    def schedule(self, limit: float | None) -> tuple[float | None, float | None]:
        """``(rotate_at, force_at)`` as connection ages (seconds) for a session ``limit``."""
        if not self.proactive:
            return None, None
        force_at = None if limit is None else max(0.0, limit - self.force_margin)
        if self.rotate_after is not None:
            return self.rotate_after, force_at
        if limit is None:
            return None, None
        return max(limit / 2, limit - self.lead), force_at


def backoff_delays(base: float, maximum: float, attempts: int) -> Iterator[float]:
    """``base, 2*base, 4*base, ...`` capped at ``maximum``, ``attempts`` values."""
    for attempt in range(max(0, attempts)):
        yield min(base * 2**attempt, maximum)


# --------------------------------------------------------------------- recorder
class ConversationRecorder:
    """The conversation as the user heard it, rebuilt from engine events.

    Feed every emitted event to :meth:`observe` and report client-side actions
    (:meth:`add_user_text`, :meth:`add_tool_output`, :meth:`truncate`). Assistant text is
    trimmed to the audio the user heard when a response is truncated; a response cut off by
    a failure keeps its text, flagged ``interrupted``.
    """

    def __init__(self, seed: ChatContext | None = None) -> None:
        self.history = ChatContext(seed.items if seed is not None else [])
        self.version = 0
        """Bumped on every change (a prepared seed is stale once it differs)."""
        self._text: dict[str, list[str]] = {}
        self._audio_ms: dict[str, float] = {}
        self._response_items: dict[str, list[str]] = {}

    def _changed(self) -> None:
        self.version += 1

    def observe(self, ev: EngineEvent) -> None:
        if isinstance(ev, InputCommitted):
            if self.history.get(ev.item_id) is None:  # keeps the user turn before the reply
                self.history.add_message("user", "", id=ev.item_id)
                self._changed()
        elif isinstance(ev, InputTranscript):
            if ev.is_final:
                self._set_text(ev.item_id, "user", ev.text.strip())
        elif isinstance(ev, ResponseText):
            parts = self._text.setdefault(ev.item_id, [])
            parts.append(ev.delta)
            self._track(ev.response_id, ev.item_id)
            self._set_text(ev.item_id, "assistant", "".join(parts).strip())
        elif isinstance(ev, ResponseAudio):
            self._audio_ms[ev.item_id] = self._audio_ms.get(ev.item_id, 0.0) + ev.frame.duration_ms
            self._track(ev.response_id, ev.item_id)
        elif isinstance(ev, ResponseToolCall):
            if not any(
                isinstance(i, FunctionCall) and i.call_id == ev.call.call_id
                for i in self.history.items
            ):
                self.history.append(ev.call)
                self._changed()
        elif isinstance(ev, ResponseDone):
            items = self._response_items.pop(ev.response_id, [])
            if ev.status in ("cancelled", "failed", "incomplete"):
                for item_id in items:
                    msg = self.history.get(item_id)
                    if isinstance(msg, ChatMessage) and not msg.interrupted:
                        msg.interrupted = True
                        self._changed()
            for item_id in items:
                self._text.pop(item_id, None)
            while len(self._audio_ms) > 256:
                del self._audio_ms[next(iter(self._audio_ms))]

    def _track(self, response_id: str, item_id: str) -> None:
        items = self._response_items.setdefault(response_id, [])
        if item_id not in items:
            items.append(item_id)

    def _set_text(self, item_id: str, role: ChatRole, text: str) -> None:
        msg = self.history.get(item_id)
        if isinstance(msg, ChatMessage):
            if msg.text == text:
                return
            msg.content = [text]
        else:
            self.history.add_message(role, text, id=item_id)
        self._changed()

    def add_user_text(self, text: str) -> None:
        self.history.add_message("user", text)
        self._changed()

    def add_tool_output(self, output: FunctionCallOutput) -> None:
        if any(
            isinstance(i, FunctionCallOutput) and i.call_id == output.call_id
            for i in self.history.items
        ):
            return
        self.history.append(output)
        self._changed()

    def truncate(self, item_id: str, audio_end_ms: float, *, heard: str | None = None) -> None:
        """The user heard ``audio_end_ms`` of ``item_id``: keep only that much text."""
        msg = self.history.get(item_id)
        if not isinstance(msg, ChatMessage) or msg.role != "assistant":
            return
        full = msg.text
        if heard is None:
            total = self._audio_ms.get(item_id, 0.0)
            if total <= 0 or audio_end_ms >= total:
                heard = full
            else:
                heard = full[: round(len(full) * max(0.0, audio_end_ms) / total)]
        msg.content = [heard.strip()]
        msg.interrupted = True
        self._text[item_id] = [heard]
        self._changed()


# -------------------------------------------------------------------- quietness
class QuietTracker:
    """Tracks whether the conversation is quiet enough to switch connections.

    Quiet = the user is not speaking, no response is being generated or requested, no
    tool call is waiting for its result, the agent's audio has (approximately) finished
    playing, and nothing happened for the policy's ``quiet_period``.
    """

    def __init__(self) -> None:
        self.user_speaking = False
        self.active: set[str] = set()
        self.pending_calls: set[str] = set()
        self.requested_at: float | None = None
        self.last_activity = now()
        self.playout_end = 0.0
        """Estimated wall-clock time the received agent audio finishes playing."""

    def observe(self, ev: EngineEvent) -> None:
        t = now()
        if isinstance(ev, InputSpeechStarted):
            self.user_speaking = True
        elif isinstance(ev, InputSpeechStopped):
            self.user_speaking = False
        elif isinstance(ev, InputCommitted):
            self.user_speaking = False
            self.requested_at = t  # a response normally follows
        elif isinstance(ev, ResponseStarted):
            self.active.add(ev.response_id)
            self.requested_at = None
        elif isinstance(ev, ResponseAudio):
            self.playout_end = max(self.playout_end, t) + ev.frame.duration
        elif isinstance(ev, ResponseToolCall):
            self.pending_calls.add(ev.call.call_id)
        elif isinstance(ev, ToolCallCancelled):
            self.pending_calls.difference_update(ev.call_ids)
        elif isinstance(ev, ResponseDone):
            self.active.discard(ev.response_id)
        elif isinstance(ev, (EngineStatus, EngineErrorEvent)):
            return
        self.last_activity = t

    def tool_output(self, call_id: str, *, respond: bool) -> None:
        self.pending_calls.discard(call_id)
        if respond:
            self.request()
        self.last_activity = now()

    def request(self) -> None:
        """A response was requested (it has not started yet)."""
        self.requested_at = now()
        self.last_activity = self.requested_at

    def reset_server_state(self) -> None:
        """The provider session is gone: its responses and requests will never finish."""
        self.active.clear()
        self.requested_at = None
        self.user_speaking = False

    def is_quiet(self, quiet_period: float) -> bool:
        t = now()
        if self.user_speaking or self.active or self.pending_calls:
            return False
        if self.requested_at is not None and t - self.requested_at < _REQUEST_TTL:
            return False
        return t >= self.playout_end and t - self.last_activity >= quiet_period


# -------------------------------------------------------------------------- audio
class AudioReplay:
    """Recently sent user audio that a new connection may lack.

    Keeps the frames sent since the last committed user turn (at most ``window`` seconds):
    a switch re-sends them so speech the old connection had only started to hear reaches
    the new one. Audio of committed turns is never replayed (it would be answered twice).
    """

    def __init__(self, window: float) -> None:
        self.window = window
        self._frames: deque[tuple[float, AudioFrame]] = deque()
        self._floor = 0.0

    def record(self, start: float, frame: AudioFrame) -> None:
        if self.window <= 0:
            return
        self._frames.append((start, frame))
        end = start + frame.duration
        while self._frames and self._frames[0][0] + self._frames[0][1].duration < end - self.window:
            self._frames.popleft()

    def mark_committed(self, position: float) -> None:
        """Audio before ``position`` belongs to a committed turn."""
        self._floor = max(self._floor, position)

    def frames(self) -> list[tuple[float, AudioFrame]]:
        return [(s, f) for s, f in self._frames if s + f.duration > self._floor]

    def clear(self) -> None:
        self._frames.clear()


class AudioBuffer:
    """User audio held back while a connection is being switched (bounded)."""

    def __init__(self, max_duration: float) -> None:
        self.max_duration = max_duration
        self._frames: deque[tuple[float, AudioFrame]] = deque()
        self.duration = 0.0
        self.dropped = 0.0
        self.total = 0.0

    def push(self, start: float, frame: AudioFrame) -> None:
        self._frames.append((start, frame))
        self.duration += frame.duration
        self.total += frame.duration
        while self.duration > self.max_duration and self._frames:
            _, old = self._frames.popleft()
            self.duration -= old.duration
            self.dropped += old.duration

    def drain(self) -> list[tuple[float, AudioFrame]]:
        frames = list(self._frames)
        self._frames.clear()
        self.duration = 0.0
        return frames

    def reset_stats(self) -> None:
        self.dropped = self.total = 0.0


# ------------------------------------------------------------------ generic wrapper
class RotatingEngine(S2SEngine):
    """Adds session rotation and reconnection to any engine.

    Each :class:`RotatingConnection` runs the conversation on a sequence of connections
    of the wrapped ``engine``: ahead of ``max_session_duration`` (or on ``EngineStatus
    ("expiring")`` / :meth:`RotatingConnection.rotate`) the next one is opened with the
    carried-over history, instructions, tools and voice, and takes over at a quiet
    moment; after a drop the conversation continues on a new connection, with backoff.
    Use it for engines without native rotation (OpenAI Realtime and Gemini Live rotate
    by themselves).
    """

    provider = "rotating"

    def __init__(self, engine: S2SEngine, *, policy: RotationPolicy | None = None) -> None:
        super().__init__(
            model=engine.model,
            capabilities=engine.capabilities,
            input_sample_rate=engine.input_sample_rate,
            output_sample_rate=engine.output_sample_rate,
        )
        self.inner = engine
        self.policy = policy or RotationPolicy()
        engine.on("metrics", lambda m: self.emit("metrics", m))

    async def connect(self, options: EngineOptions) -> EngineConnection:
        conn = RotatingConnection(self, options)
        try:
            await conn.start()
        except BaseException:
            await conn.aclose()
            raise
        return conn

    async def warmup(self) -> None:
        await self.inner.warmup()

    async def aclose(self) -> None:
        await self.inner.aclose()


@dataclass
class _Link:
    conn: EngineConnection
    epoch: int
    opened_at: float = field(default_factory=now)
    offset: float = 0.0
    """Position of the wrapper's input stream where this connection's stream starts."""
    version: int = -1
    """Recorder version the connection was seeded with."""
    carried: int = 0
    """Number of items it was seeded with."""
    limit: float | None = None
    pump: asyncio.Task[None] | None = None


def _retryable(error: Exception) -> bool:
    if isinstance(error, (AuthenticationError, ConfigurationError)):
        return False
    if isinstance(error, ProviderError):
        return error.retryable or isinstance(error, ProviderConnectionError)
    return isinstance(error, (ConnectionError, OSError, TimeoutError))


class RotatingConnection(EngineConnection):
    """A conversation that outlives the connections of the wrapped engine."""

    engine: RotatingEngine

    def __init__(self, engine: RotatingEngine, options: EngineOptions) -> None:
        super().__init__(engine, options)
        self.engine = engine
        self.policy = engine.policy
        self.instructions = options.instructions
        self.tools: list[FunctionTool] = list(options.tools)
        self.recorder = ConversationRecorder(options.chat_ctx)
        self.tracker = QuietTracker()
        self.rotations = 0
        """Number of connection switches so far."""
        self._replay = AudioReplay(self.policy.replay)
        self._buffer = AudioBuffer(self.policy.max_buffered_audio)
        self._tasks = BackgroundTasks("rotating-engine")
        self._link: _Link | None = None
        self._standby: _Link | None = None
        self._epoch = 0
        self._switching = False
        self._ready = asyncio.Event()
        """Set while no switch is in progress (control calls wait for it)."""
        self._pending: str | None = None
        self._deadline: float | None = None
        self._monitor: asyncio.Task[None] | None = None
        self._preparing: asyncio.Task[None] | None = None
        self._reconnecting: asyncio.Task[None] | None = None

    @property
    def inner(self) -> EngineConnection | None:
        """The wrapped engine's connection currently carrying the conversation."""
        return self._link.conn if self._link is not None else None

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        link = await self._open(self.options.chat_ctx or ChatContext(), self.recorder.version)
        self._link = link
        link.pump = self._tasks.spawn(self._pump(link), name=f"rotating-pump-{link.epoch}")
        self._ready.set()
        self._monitor = asyncio.create_task(self._monitor_loop(), name="rotating-monitor")

    async def aclose(self) -> None:
        if self.closed:
            return
        self._closed = True
        self._ready.set()  # release waiting control calls (they see the connection closed)
        current = asyncio.current_task()
        tasks = [self._monitor, self._preparing, self._reconnecting]
        await cancel_and_wait(*[t for t in tasks if t is not None and t is not current])
        links = [link for link in (self._link, self._standby) if link is not None]
        self._link = self._standby = None
        for link in links:
            with contextlib.suppress(Exception):
                await link.conn.aclose()
        await self._tasks.cancel_all()
        await super().aclose()

    async def _open(self, seed: ChatContext, version: int) -> _Link:
        options = replace(
            self.options, instructions=self.instructions, tools=list(self.tools), chat_ctx=seed
        )
        conn = await self.engine.inner.connect(options)
        self._epoch += 1
        return _Link(
            conn,
            self._epoch,
            version=version,
            carried=len(seed),
            limit=conn.capabilities.max_session_duration,
        )

    async def _open_seeded(self) -> _Link:
        version = self.recorder.version
        seed = await self.policy.carry_over(self.recorder.history)
        return await self._open(seed, version)

    async def _close_link(self, link: _Link) -> None:
        with contextlib.suppress(Exception):
            await link.conn.aclose()
        if link.pump is not None and link.pump is not asyncio.current_task():
            await cancel_and_wait(link.pump)

    # --------------------------------------------------------------------- events
    def _emit(self, event: EngineEvent) -> None:
        self.recorder.observe(event)
        self.tracker.observe(event)
        super()._emit(event)

    async def _pump(self, link: _Link) -> None:
        async for ev in link.conn.events():
            if self._link is not link or self.closed:
                return  # a replaced connection: its late events are not ours any more
            if isinstance(ev, EngineErrorEvent) and not ev.recoverable:
                if _retryable(ev.error) and self.policy.max_reconnect_attempts > 0:
                    self._start_reconnect(link, f"connection failed: {ev.error}")
                    return
                self._emit(ev)
                self._tasks.spawn(self.aclose())
                return
            if isinstance(ev, EngineStatus) and ev.status == "expiring":
                deadline = None
                if ev.time_left is not None:
                    deadline = now() + max(0.0, ev.time_left - self.policy.force_margin)
                self._emit(ev)
                self.rotate("expiring", deadline=deadline)
                continue
            if isinstance(ev, (InputSpeechStarted, InputSpeechStopped)):
                if ev.audio_time is not None:
                    ev.audio_time += link.offset
            elif isinstance(ev, InputCommitted):
                self._replay.mark_committed(self.input_audio_time)
            self._emit(ev)
        if self._link is link and not self.closed:
            self._start_reconnect(link, "connection closed")

    # ---------------------------------------------------------------- rotation
    def rotate(self, reason: str = "requested", *, deadline: float | None = None) -> None:
        """Move to a new connection at the next quiet moment (at the latest at
        ``deadline``, a :func:`~voice_agent_next.utils.now` time)."""
        if self._pending is None:
            self._pending = reason
        if deadline is not None:
            self._deadline = deadline if self._deadline is None else min(self._deadline, deadline)

    async def _monitor_loop(self) -> None:
        while not self.closed:
            await asyncio.sleep(_MONITOR_INTERVAL)
            try:
                await self._check()
            except Exception:
                logger.exception("rotation check failed")

    async def _check(self) -> None:
        link = self._link
        if link is None or self._switching:
            return
        t = now()
        rotate_at, force_at = self.policy.schedule(link.limit)
        if self._pending is None and rotate_at is not None and t - link.opened_at >= rotate_at:
            self._pending = "max_session_duration"
            if link.limit is not None:
                left = max(0.0, link.opened_at + link.limit - t)
                self._emit(EngineStatus(status="expiring", detail=self._pending, time_left=left))
        if self._pending is None:
            return
        if force_at is not None:
            forced = link.opened_at + force_at
            self._deadline = forced if self._deadline is None else min(self._deadline, forced)
        forced_now = self._deadline is not None and t >= self._deadline
        standby = self._standby
        if forced_now:
            await self._switch(self._pending, forced=True)
        elif standby is None:
            if self._preparing is None or self._preparing.done():
                self._preparing = asyncio.create_task(self._prepare(), name="rotating-prepare")
        elif self.tracker.is_quiet(self.policy.quiet_period):
            if standby.version == self.recorder.version:
                await self._switch(self._pending, forced=False)
            else:  # the conversation moved on since it was seeded: prepare it again
                self._standby = None
                await self._close_link(standby)

    async def _prepare(self) -> None:
        """Make-before-break: open and seed the next connection ahead of the switch."""
        try:
            standby = await self._open_seeded()
        except Exception as exc:
            logger.warning("could not prepare the next connection: %s", exc)
            self._emit(EngineErrorEvent(error=exc, recoverable=True))
            await asyncio.sleep(self.policy.backoff)
            return
        if self.closed or self._switching or self._standby is not None:
            await self._close_link(standby)
            return
        self._standby = standby

    async def _switch(self, reason: str, *, forced: bool) -> None:
        old = self._link
        if old is None or self.closed:
            return
        if self._preparing is not None and self._preparing is not asyncio.current_task():
            await cancel_and_wait(self._preparing)
        started = now()
        self._begin_switch()
        self._emit(EngineStatus(status="reconnecting", detail=reason))
        standby, self._standby = self._standby, None
        try:
            if standby is None or standby.version != self.recorder.version:
                if standby is not None:
                    await self._close_link(standby)
                standby = await self._open_seeded()
        except Exception as exc:
            logger.warning("planned rotation failed (%s); keeping the connection", exc)
            await self._flush(old)
            self._end_switch()
            self._emit(EngineErrorEvent(error=exc, recoverable=True))
            self._emit(EngineStatus(status="resumed", detail=f"kept the connection ({reason})"))
            await asyncio.sleep(self.policy.backoff)  # the next check retries (with a deadline)
            return
        self._link = standby  # from now on the old connection's events are ignored
        self._pending = self._deadline = None
        failed = self._fail_inflight("session rotated") if forced else 0
        replayed = await self._deliver(standby)
        self._finish_switch(reason, planned=True, started=started, attempts=1,
                            replayed=replayed, link=standby, failed=failed)  # fmt: skip
        self._end_switch()
        await self._close_link(old)

    def _begin_switch(self) -> None:
        self._switching = True
        self._ready.clear()
        self._buffer.reset_stats()

    def _end_switch(self) -> None:
        self._switching = False
        self._ready.set()

    def _fail_inflight(self, reason: str) -> int:
        active = list(self.tracker.active)
        for rid in active:
            self._emit(ResponseDone(response_id=rid, status="failed", error=reason))
        self.tracker.reset_server_state()
        return len(active)

    async def _deliver(self, link: _Link) -> float:
        """Start ``link``: re-send recent uncommitted audio, then the buffered audio."""
        buffered = self._buffer.drain()
        held = {id(f) for _, f in buffered}
        replay = [(s, f) for s, f in self._replay.frames() if id(f) not in held]
        frames = replay + buffered
        link.offset = frames[0][0] if frames else self.input_audio_time
        link.opened_at = now()
        link.pump = self._tasks.spawn(self._pump(link), name=f"rotating-pump-{link.epoch}")
        for _, frame in frames:  # audio arriving meanwhile is buffered behind these frames
            await link.conn.send_audio(frame)
        await self._flush(link)
        return sum(f.duration for _, f in replay)

    async def _flush(self, link: _Link) -> None:
        while frames := self._buffer.drain():
            for _, frame in frames:
                await link.conn.send_audio(frame)

    def _finish_switch(
        self,
        reason: str,
        *,
        planned: bool,
        started: float,
        attempts: int,
        replayed: float,
        link: _Link,
        failed: int,
    ) -> None:
        self.rotations += 1
        lost = self._buffer.dropped
        if lost > 0:  # never a silent loss: the application must know
            msg = f"{lost:.2f}s of user audio lost while switching connections ({reason})"
            self._emit(EngineErrorEvent(error=ProviderConnectionError(msg), recoverable=True))
        self._emit(EngineStatus(status="reconnected", detail=reason))
        self.engine.emit(
            "metrics",
            RotationMetrics(
                provider=self.engine.inner.provider,
                model=self.engine.model,
                reason=reason,
                planned=planned,
                rotation=self.rotations,
                gap=now() - started,
                attempts=attempts,
                buffered_audio=self._buffer.total,
                replayed_audio=replayed,
                lost_audio=lost,
                carried_items=link.carried,
                failed_responses=failed,
            ),
        )

    # ---------------------------------------------------------------- reconnect
    def _start_reconnect(self, link: _Link, reason: str) -> None:
        if self.closed or (self._reconnecting is not None and not self._reconnecting.done()):
            return
        self._begin_switch()
        self._reconnecting = self._tasks.spawn(self._reconnect(link, reason), name="reconnect")

    async def _reconnect(self, dead: _Link, reason: str) -> None:
        started = now()
        failed = self._fail_inflight(f"connection lost: {reason}")
        self._emit(EngineStatus(status="reconnecting", detail=reason))
        self._tasks.spawn(dead.conn.aclose())
        if self._preparing is not None:
            await cancel_and_wait(self._preparing)
        standby, self._standby = self._standby, None
        if standby is not None:
            await self._close_link(standby)
        last: Exception | None = None
        policy = self.policy
        delays = backoff_delays(policy.backoff, policy.max_backoff, policy.max_reconnect_attempts)
        for attempt, delay in enumerate(delays, 1):
            await asyncio.sleep(delay)
            try:
                link = await self._open_seeded()
            except Exception as exc:
                last = exc
                if not _retryable(exc):
                    break
                logger.warning("reconnect attempt %d failed: %s", attempt, exc)
                continue
            self._link = link
            self._pending = self._deadline = None
            replayed = await self._deliver(link)
            self._finish_switch(reason, planned=False, started=started, attempts=attempt,
                                replayed=replayed, link=link, failed=failed)  # fmt: skip
            self._end_switch()
            return
        error = last or ProviderConnectionError(f"engine connection {reason}")
        self._emit(EngineErrorEvent(error=error, recoverable=False))
        await self.aclose()

    # ---------------------------------------------------------------- audio in
    async def _send_audio(self, frame: AudioFrame) -> None:
        start = self.input_audio_time - frame.duration
        self._replay.record(start, frame)
        link = self._link
        if link is not None and link.conn.closed and not self._switching:
            self._start_reconnect(link, "connection closed")  # before its pump noticed
        if self._switching or link is None:
            self._buffer.push(start, frame)
            return
        await link.conn.send_audio(frame)

    async def commit_input(self) -> None:
        self._replay.mark_committed(self.input_audio_time)
        await self._call(lambda c: c.commit_input())

    async def clear_input(self) -> None:
        await self._call(lambda c: c.clear_input())

    # ------------------------------------------------------------------ control
    async def _call(self, fn: Callable[[EngineConnection], Awaitable[None]]) -> bool:
        """Run ``fn`` on the current connection, after a switch in progress."""
        await self._ready.wait()
        link = self._link
        if link is None or self.closed:
            return False
        await fn(link.conn)
        return True

    async def send_text(self, text: str, *, respond: bool = True) -> None:
        if respond:
            self.tracker.request()
        # recorded once delivered: a connection seeded meanwhile must not get it twice
        if await self._call(lambda c: c.send_text(text, respond=respond)):
            self.recorder.add_user_text(text)

    async def create_response(self, *, instructions: str | None = None) -> None:
        self.tracker.request()
        await self._call(lambda c: c.create_response(instructions=instructions))

    async def say(self, text: str) -> None:
        self.tracker.request()
        await self._call(lambda c: c.say(text))

    async def cancel_response(self) -> None:
        if self._link is not None and not self._switching:
            await self._link.conn.cancel_response()

    async def truncate(self, item_id: str, audio_end_ms: int) -> str | None:
        link = self._link
        heard = None
        if link is not None and not self._switching:
            heard = await link.conn.truncate(item_id, audio_end_ms)
        self.recorder.truncate(item_id, audio_end_ms, heard=heard)
        return heard

    async def interrupt(
        self, item_id: str | None = None, played_ms: int | None = None
    ) -> str | None:
        link = self._link
        heard = None
        if link is not None and not self._switching:
            heard = await link.conn.interrupt(item_id, played_ms)
        if item_id is not None and played_ms is not None:
            self.recorder.truncate(item_id, played_ms, heard=heard)
        return heard

    async def send_tool_output(self, output: FunctionCallOutput, *, respond: bool = True) -> None:
        self.tracker.tool_output(output.call_id, respond=respond)
        if await self._call(lambda c: c.send_tool_output(output, respond=respond)):
            self.recorder.add_tool_output(output)

    async def send_async_tool_output(
        self, output: FunctionCallOutput, *, scheduling: ToolScheduling = "when_idle"
    ) -> None:
        self.tracker.tool_output(output.call_id, respond=scheduling != "silent")
        if await self._call(lambda c: c.send_async_tool_output(output, scheduling=scheduling)):
            self.recorder.add_tool_output(output)

    async def update(
        self, *, instructions: str | None = None, tools: list[FunctionTool] | None = None
    ) -> None:
        if instructions is not None:
            self.instructions = instructions
        if tools is not None:
            self.tools = list(tools)
        await self._call(lambda c: c.update(instructions=instructions, tools=tools))
