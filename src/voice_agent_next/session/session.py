"""The :class:`AgentSession` runtime: transport <-> engine orchestration.

Responsibilities (identical for native speech-to-speech and cascaded engines):

* stream user audio from the transport into the engine (through optional audio
  processors such as echo cancellation);
* play engine audio through the transport, paced in real time with a small
  look-ahead so the session always knows what the user has actually *heard*;
* barge-in: when the user starts speaking over the agent, pause playback and let the
  :mod:`~voice_agent_next.session.interruptions` policy decide — a real interruption
  cancels the response and truncates the agent's turn to what was heard, a false one
  (cough, noise, "uh-huh") resumes the agent where it stopped;
* execute tool calls and feed the results back to the engine: a filler when a round is
  slow (watchdog), non-blocking tools whose results arrive later, progress updates and
  background work (:meth:`AgentSession.delegate`);
* keep the conversation history and emit transcripts, state changes and metrics;
* hand the conversation over to another :class:`Agent` when a tool asks for it
  (:mod:`~voice_agent_next.session.handoff`).
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import functools
import inspect
import os
import random
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, cast

from ..audio.frame import AudioFrame
from ..audio.processing import AudioProcessor, ProcessorChain
from ..audio.resample import StreamResampler
from ..chat import ChatContext, ChatMessage, FunctionCall, FunctionCallOutput
from ..engine import EngineConnection, EngineOptions, S2SEngine
from ..errors import ConfigurationError
from ..events import (
    EngineErrorEvent,
    EngineEvent,
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
from ..metrics import Metrics, TurnMetrics, UsageSummary
from ..registry import create
from ..tools import (
    DEFAULT_TOOL_ACK,
    FunctionTool,
    ToolContext,
    ToolScheduling,
    UserdataT,
    _stringify,
    execute_function_call,
    find_tool,
)
from ..transports.base import Transport
from ..utils.aio import BackgroundTasks, Chan, cancel_and_wait
from ..utils.clock import now
from ..utils.emitter import EventEmitter
from ..utils.ids import new_id
from ..utils.log import logger
from .agent import Agent
from .events import (
    AgentFalseInterruption,
    AgentHandoff,
    AgentState,
    AgentStateChanged,
    AgentTranscript,
    ConversationItemAdded,
    Interrupted,
    SessionClosed,
    SessionError,
    ToolCalled,
    ToolCancelled,
    ToolFiller,
    ToolProgress,
    ToolResult,
    UserState,
    UserStateChanged,
    UserTranscript,
)
from .handoff import Handoff, HistoryMode, as_handoff, carry_history, history_mode_name
from .interruptions import InterruptionPolicy, Overlap, Verdict
from .taps import SessionTap

if TYPE_CHECKING:
    from ..engines.cascade import CascadeOptions
    from ..llm import LLM
    from .recording import SessionRecorder
    from .tracing import SessionTracer

__all__ = ["DEFAULT_TOOL_FILLERS", "AgentSession", "SessionOptions"]

_MAX_ONSET_LAG = 0.5
"""Upper bound (s) on how long after its onset an engine reports user speech."""
_SAY_REQUEST_TTL = 10.0
"""A ``say()`` whose response has not started within this many seconds is forgotten."""
_RECHECK_DELAY = 0.01
"""Re-check an overlap this soon when engine events are still waiting to be handled."""
_MAX_PLAYBACK_LAG = 2.0
"""Longest wait (s) for a transport to finish playing a response after the virtual clock."""
_ASIDE_TIMEOUT = 10.0
"""Longest wait (s) for a filler/progress utterance to be generated before tool outputs
(whose follow-up response would cut it off) are sent."""
_IDLE_POLL = 0.05
"""Poll interval (s) of the watchdog and of result delivery waiting for a quiet moment."""
_IDLE_SETTLE = 0.15
"""The conversation must stay idle this long before a background result is delivered."""

_OWNER_TASK: contextvars.ContextVar[asyncio.Task[Any] | None] = contextvars.ContextVar(
    "voice_agent_next_session_owner_task", default=None
)
"""The session's background task running a tool or delegated work. Code it runs may sit in
an inner task (``asyncio.wait_for`` creates one on Python 3.11); ``aclose()`` called from
there must not cancel the owner it is awaited by."""

DEFAULT_TOOL_FILLERS: tuple[str, ...] = (
    "One moment, let me check that.",
    "Just a second.",
    "Let me look that up.",
    "Give me a moment.",
    "Hang on, I'm checking.",
)
"""Default fillers of ``SessionOptions.tool_fillers``."""


@dataclass(slots=True)
class SessionOptions:
    allow_interruptions: bool = True
    """Let the user barge in while the agent is speaking/thinking."""
    output_lookahead: float = 0.15
    """Seconds of agent audio handed to the transport ahead of real time."""
    tool_timeout: float | None = 30.0
    """Per-call timeout for tool execution (``None`` = no timeout)."""
    max_tool_steps: int = 5
    """Maximum consecutive tool-call rounds per request chain: a user turn (spoken or
    typed with ``generate_reply(user_input=)``), a ``generate_reply()`` or a background
    result that asks for a response, and the tool rounds that follow it."""
    close_on_disconnect: bool = True
    """Close the session when the transport's audio input ends (user hung up)."""
    warmup: bool = True
    """Pre-warm the engine before the conversation starts (load models, open connections):
    local model loads and cold connections otherwise land on the first turn."""
    min_interruption_duration: float = 0.5
    """Seconds of user speech needed to confirm a barge-in; until then the agent is only
    paused. ``0`` together with ``min_interruption_words=0`` interrupts at the first sign
    of speech, without pausing (the behaviour before the interruption policy)."""
    min_interruption_words: int = 0
    """Non-backchannel words needed as well, counted in the engine's interim transcripts
    (``0`` = duration only). Use 1+ with a streaming STT to ignore noise and coughs."""
    backchannel_words: Sequence[str] | None = None
    """Words and phrases that never count as an interruption ("uh-huh", "okay"...).
    ``None`` = the built-in list for the agent's language (see
    :func:`~voice_agent_next.session.interruptions.backchannel_words_for`)."""
    max_backchannel_duration: float | None = None
    """The short-utterance rule: user speech over the agent shorter than this (seconds,
    counted until the VAD reports its end), with at most two words and no
    ``interruption_words``, is a backchannel whatever the STT made of it — small ASR models
    turn "uh-huh" into words like "but high". Such an utterance never confirms a barge-in
    by duration and is dropped instead of answered. ``None`` = off. The local presets use
    1.0 s (see ``docs/concepts/interruptions.md``)."""
    interruption_words: Sequence[str] | None = None
    """Words that make even a short utterance a real barge-in ("stop", "wait", "no",
    question words...). ``None`` = the built-in list for the agent's language (see
    :func:`~voice_agent_next.session.interruptions.interruption_words_for`)."""
    false_interruption_timeout: float | None = 2.0
    """Seconds the user must stay quiet, without meaningful words, before paused speech
    resumes. ``None`` disables pause-and-resume."""
    resume_false_interruption: bool = True
    """Pause the agent while a barge-in is unconfirmed and resume it after a false
    interruption. ``False``: the agent keeps talking until the barge-in is confirmed."""
    discard_audio_if_uninterruptible: bool = True
    """While uninterruptible speech plays, the engine receives silence instead of the
    user's audio (so it can neither barge in nor queue a turn)."""
    record: str | os.PathLike[str] | None = None
    """Record the call: a directory (a new ``<time>-<id>.wav`` + ``.jsonl`` pair per
    session) or a ``.wav`` path. See :class:`~voice_agent_next.session.SessionRecorder`."""
    trace: bool = False
    """Export OpenTelemetry spans (needs the ``otel`` extra and an SDK configured by the
    app). See :class:`~voice_agent_next.session.SessionTracer`."""
    tool_filler_delay: float | None = 1.5
    """Watchdog: seconds a blocking tool round may keep the conversation silent before the
    session says a filler ("One moment, let me check that."), once per round. ``None``
    disables fillers. Not used with engines whose model keeps talking while tools run
    (``tool_mode`` other than ``"blocking"``)."""
    tool_fillers: Sequence[str] = DEFAULT_TOOL_FILLERS
    """Fillers of tools without their own (``function_tool(filler=...)``), picked without
    repeating until all were used."""
    tool_filler_interruptible: bool = True
    """``False``: fillers and spoken progress updates cannot be interrupted."""

    def __post_init__(self) -> None:
        if self.min_interruption_duration < 0:
            raise ValueError("min_interruption_duration must be >= 0")
        if self.min_interruption_words < 0:
            raise ValueError("min_interruption_words must be >= 0")
        if self.false_interruption_timeout is not None and self.false_interruption_timeout < 0:
            raise ValueError("false_interruption_timeout must be >= 0 or None")
        if isinstance(self.backchannel_words, str):
            raise TypeError("backchannel_words must be a list of words, not a string")
        if self.max_backchannel_duration is not None and self.max_backchannel_duration < 0:
            raise ValueError("max_backchannel_duration must be >= 0 or None")
        if isinstance(self.interruption_words, str):
            raise TypeError("interruption_words must be a list of words, not a string")
        if self.tool_filler_delay is not None and self.tool_filler_delay < 0:
            raise ValueError("tool_filler_delay must be >= 0 or None")
        if isinstance(self.tool_fillers, str):
            raise TypeError("tool_fillers must be a list of phrases, not a string")


@dataclass
class _Turn:
    """One user turn and every agent response it triggered (tool rounds included)."""

    turn_id: str
    speech_end: float | None
    committed_at: float
    first_audio_at: float | None = None
    tool_calls: int = 0
    agent_speech: float = 0.0
    interrupted: bool = False
    closed: bool = False


@dataclass(eq=False)
class _ToolRun:
    """One execution of a tool call (or the immediate acknowledgement of a non-blocking
    call, ``ack=True``)."""

    call: FunctionCall
    tool: FunctionTool | None
    task: asyncio.Future[FunctionCallOutput]
    started: float = field(default_factory=now)
    ack: bool = False
    cancelled: bool = False

    def outcome(self) -> FunctionCallOutput | None:
        """The output of the finished run; ``None`` if it was cancelled."""
        if self.cancelled or self.task.cancelled():
            return None
        return self.task.result()


class _PhrasePicker:
    """Picks phrases from a list without repeating one before all were used."""

    def __init__(self) -> None:
        self._rng = random.Random()
        self._used: dict[tuple[str, ...], set[str]] = {}
        self._last: str | None = None

    def pick(self, phrases: Sequence[str]) -> str | None:
        options = tuple(dict.fromkeys(p for p in phrases if p.strip()))
        if not options:
            return None
        used = self._used.setdefault(options, set())
        fresh = [p for p in options if p not in used]
        if not fresh:
            used.clear()
            fresh = [p for p in options if p != self._last] or list(options)
        choice = self._rng.choice(fresh)
        used.add(choice)
        self._last = choice
        return choice


@dataclass(eq=False)
class _Response:
    response_id: str
    started_at: float
    turn: _Turn | None
    item_id: str | None = None
    message: ChatMessage | None = None
    text: list[str] = field(default_factory=list)
    tool_calls: list[FunctionCall] = field(default_factory=list)
    """Calls of this response's tool round (the model waits for their outputs)."""
    tool_runs: list[_ToolRun] = field(default_factory=list)
    watchdog: asyncio.Task[None] | None = None
    """Says a filler if the tool round is slow."""
    filler_said: bool = False
    """The round already had its filler (or a spoken progress update)."""
    aside: bool = False
    """A filler or progress utterance: it neither belongs to nor ends the user's turn."""
    say: _SayRequest | None = None
    done: bool = False
    status: ResponseStatus | None = None
    interrupted: bool = False
    finished: bool = False
    first_audio_at: float | None = None
    received: float = 0.0
    """Seconds of audio received from the engine."""
    segments: list[tuple[float, float]] = field(default_factory=list)
    """(playback start time, duration) of every chunk sent to the transport."""
    allow_interruptions: bool | None = None
    """Overrides ``SessionOptions.allow_interruptions`` (``say(..., allow_interruptions=)``)."""

    def played(self, t: float) -> float:
        return sum(min(max(t - start, 0.0), dur) for start, dur in self.segments)

    @property
    def sent(self) -> float:
        return sum(d for _, d in self.segments)

    @property
    def end(self) -> float:
        """Playback time at which the audio sent so far ends."""
        return max((start + dur for start, dur in self.segments), default=0.0)


@dataclass(slots=True)
class _EndOfResponse:
    response_id: str


@dataclass(eq=False)
class _BargeIn:
    """User speech over ``response`` awaiting a verdict (see :class:`Overlap`)."""

    overlap: Overlap
    response: _Response
    ignore_items: frozenset[str]
    """Input items committed before the overlap: their late transcripts are not evidence."""
    paused_at: float | None = None
    """When this overlap paused playback (``None`` = not paused)."""
    paused_total: float = 0.0
    """Seconds of pauses that already ended."""
    input_cleared: bool = False
    """The engine's pending input was cleared during the current silence."""


@dataclass(eq=False, slots=True)
class _SayRequest:
    allow_interruptions: bool | None
    aside: bool = False
    created: float = field(default_factory=now)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    """Set once the engine finished generating the utterance."""


class AgentSession(EventEmitter, Generic[UserdataT]):
    """Runs an :class:`Agent` over a :class:`Transport` with a speech-to-speech engine.

    ``userdata`` is application state shared by every agent and tool of the session
    (``ToolContext.userdata``); ``AgentSession[MyData](..., userdata=MyData())`` types it.

    Either pass a native/prebuilt ``engine`` (instance or spec such as
    ``"openai/gpt-realtime"``) **or** the cascade components ``stt``/``llm``/``tts``
    (plus optional ``vad`` and ``turn_detector``), which are assembled into a
    :class:`~voice_agent_next.engines.cascade.CascadeEngine`.

    Example:
        >>> session = AgentSession("mock")                    # native S2S (mock)
        >>> session = AgentSession(stt="deepgram/nova-3", llm="openai/gpt-4.1-mini",
        ...                        tts="cartesia/sonic-2", vad="silero")  # cascade
        >>> await session.run(Agent("You are helpful."), transport)
    """

    def __init__(
        self,
        engine: S2SEngine | str | Mapping[str, Any] | None = None,
        *,
        stt: Any = None,
        llm: Any = None,
        tts: Any = None,
        vad: Any = None,
        turn_detector: Any = None,
        cascade_options: CascadeOptions | None = None,
        options: SessionOptions | None = None,
        processors: Sequence[AudioProcessor] = (),
        userdata: UserdataT | None = None,
        record: str | os.PathLike[str] | SessionRecorder | None = None,
        trace: bool | SessionTracer | None = None,
    ) -> None:
        """
        Args:
            userdata: application state shared by the agents and tools (``None`` if unset).
            record: record the call to a stereo WAV (user left, agent right) and a JSONL
                event timeline: a directory, a ``.wav`` path or a
                :class:`~voice_agent_next.session.SessionRecorder`. Overrides
                ``SessionOptions.record``.
            trace: export OpenTelemetry spans (``True`` or a
                :class:`~voice_agent_next.session.SessionTracer`). Overrides
                ``SessionOptions.trace``.
        """
        super().__init__()
        if engine is not None:
            if any(c is not None for c in (stt, llm, tts, turn_detector)):
                raise ConfigurationError("pass either engine=... or cascade components, not both")
            self.engine: S2SEngine = create("engine", engine)
        else:
            if llm is None:  # (the cascade checks for tts=... unless the LLM speaks)
                raise ConfigurationError(
                    "AgentSession needs engine=... (native speech-to-speech) or at least "
                    "llm=... and tts=... (cascade; add stt=... unless the LLM takes audio, "
                    "tts=... is optional when it outputs audio)"
                )
            from ..engines.cascade import CascadeEngine

            self.engine = CascadeEngine(
                stt=stt, llm=llm, tts=tts, vad=vad, turn_detector=turn_detector,
                options=cascade_options,
            )  # fmt: skip
        self.options = options or SessionOptions()
        self.userdata: UserdataT = cast(UserdataT, userdata)
        """Application state shared by every agent and tool (survives handoffs)."""
        self.history = ChatContext()
        self.usage = UsageSummary()
        self.agent_state = AgentState.INITIALIZING
        self.user_state = UserState.LISTENING
        self._processors = ProcessorChain(processors) if processors else None
        self._agent: Agent | None = None
        self._transport: Transport | None = None
        self._conn: EngineConnection | None = None
        self._tasks = BackgroundTasks("session")
        self._loops: list[asyncio.Task[Any]] = []
        self._closed = asyncio.Event()
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None
        self._close_callers: set[asyncio.Task[Any]] = set()  # tasks waiting in aclose()
        self._out: Chan[ResponseAudio | _EndOfResponse] = Chan()
        self._virtual_end = 0.0
        self._responses: dict[str, _Response] = {}
        self._current: _Response | None = None
        self._user_items: dict[str, ChatMessage] = {}
        self._turn: _Turn | None = None
        self._user_speech_end: float | None = None
        self._tool_steps = 0  # tool rounds of the current request chain
        self._tool_runs: dict[str, _ToolRun] = {}  # running executions by call id
        self._pending_rounds: set[_Response] = set()  # tool rounds the model waits for
        self._fillers = _PhrasePicker()
        self._handoffs: dict[str, Handoff] = {}  # handoffs requested by finished tool calls
        self._handoff_lock = asyncio.Lock()
        self._voice: str | None = None  # the voice the engine speaks with
        self._reply_requests = 0  # say()/generate_reply() calls (did on_enter speak?)
        # interruption policy: the overlap awaiting a verdict and the playback pause state
        self._barge: _BargeIn | None = None
        self._barge_timer: asyncio.TimerHandle | None = None
        self._commits_deferred = False  # the engine holds its commits (see defer_commit)
        self._committed_items: deque[str] = deque(maxlen=8)
        self._say_requests: deque[_SayRequest] = deque()
        self._send_gate = asyncio.Event()  # cleared while playback is paused
        self._send_gate.set()
        self._clock_paused_at: float | None = None  # the transport itself is paused
        self._clock_running = asyncio.Event()
        self._clock_running.set()
        self._transport_paused = False
        self._pause_seq = 0  # bumped whenever the playback timeline pauses or moves
        self._engine_events: AsyncIterator[EngineEvent] | None = None
        self._taps: list[SessionTap] = []
        self.recorder: SessionRecorder | None = None
        """The call recorder (``record=...``), if any."""
        self.engine.on("metrics", self._on_metrics)
        self._setup_observability(
            record if record is not None else self.options.record,
            trace if trace is not None else self.options.trace,
        )

    def _setup_observability(
        self, record: str | os.PathLike[str] | SessionRecorder | None, trace: Any
    ) -> None:
        if record is not None:
            from .recording import SessionRecorder

            recorder = record if isinstance(record, SessionRecorder) else SessionRecorder(record)
            recorder.attach(self)
            self.recorder = recorder
        if trace:
            from .tracing import SessionTracer

            if isinstance(trace, SessionTracer):
                trace.attach(self)
            elif SessionTracer.available():
                SessionTracer().attach(self)
            else:
                logger.warning(
                    "tracing requested but opentelemetry-api is not installed "
                    "(pip install 'voice-agent-next[otel]'): spans are not exported"
                )

    def add_tap(self, tap: SessionTap) -> None:
        """Attach a low-level observer (see :mod:`voice_agent_next.session.taps`) before
        the session starts."""
        if self._agent is not None:
            raise RuntimeError("attach taps before the session starts")
        self._taps.append(tap)

    def _notify(self, hook: str, *args: Any) -> None:
        for tap in self._taps:
            try:
                getattr(tap, hook)(*args)
            except Exception:
                logger.exception("session tap %s.%s failed", type(tap).__name__, hook)

    # ---------------------------------------------------------------- properties
    @property
    def agent(self) -> Agent:
        if self._agent is None:
            raise RuntimeError("session not started")
        return self._agent

    @property
    def transport(self) -> Transport:
        if self._transport is None:
            raise RuntimeError("session not started")
        return self._transport

    @property
    def connection(self) -> EngineConnection:
        if self._conn is None:
            raise RuntimeError("session not started")
        return self._conn

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    # ----------------------------------------------------------------- lifecycle
    async def start(self, agent: Agent, transport: Transport) -> None:
        """Connect the engine and transport and start processing audio."""
        if self._agent is not None:
            raise RuntimeError("session already started")
        self._agent = agent
        self._voice = agent.voice
        self._transport = transport
        try:
            await self._start(agent, transport)
        except BaseException:
            # release whatever was opened (transport, recorder file, engine, loops)
            with contextlib.suppress(Exception):
                await self.aclose("start_failed")
            raise

    async def _start(self, agent: Agent, transport: Transport) -> None:
        if self.options.warmup:
            try:  # before the transport opens, so no user audio queues up meanwhile
                await self.engine.warmup()
            except Exception as exc:
                logger.warning("engine warmup failed (continuing without it): %s", exc)
        await transport.start()
        if self._taps:
            self._notify("session_started", self, now())
        self._conn = await self.engine.connect(
            EngineOptions(
                instructions=agent.instructions,
                tools=list(agent.tools),
                chat_ctx=agent.chat_ctx,
                voice=agent.voice,
                language=agent.language,
            )
        )
        if agent.chat_ctx is not None:
            self.history.items.extend(agent.chat_ctx.items)
        self._loops = [
            asyncio.create_task(self._input_loop(), name="session-input"),
            asyncio.create_task(self._event_loop(), name="session-events"),
            asyncio.create_task(self._playout_loop(), name="session-playout"),
        ]
        for task in self._loops:
            task.add_done_callback(self._on_loop_done)
        self._set_agent_state(AgentState.LISTENING)
        await agent.on_enter(self)
        if agent.greeting:
            await self.say(agent.greeting)

    async def run(self, agent: Agent, transport: Transport) -> None:
        """Start the session and wait until it closes."""
        await self.start(agent, transport)
        await self.wait_closed()

    async def wait_closed(self) -> None:
        await self._closed.wait()

    def _schedule_close(self, reason: str) -> None:
        self._begin_close(reason)

    def _begin_close(self, reason: str) -> asyncio.Task[None]:
        """Start closing the session (once); returns the task doing it."""
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._close(reason), name="session-close")
        return self._close_task

    async def aclose(self, reason: str = "closed") -> None:
        """Stop the session and release the engine connection and transport.

        Idempotent and safe to call concurrently: every call waits for the same close.
        Cancelling a caller does not abort the cleanup, which runs to the end in its own
        task. The calling task (e.g. a tool ending the call) is not cancelled by it.
        """
        task = self._begin_close(reason)
        current = asyncio.current_task()
        if task is current:
            return  # called from a hook during the close (e.g. ``Agent.on_exit``)
        callers = {t for t in (current, _OWNER_TASK.get()) if t is not None}
        if not task.done():
            self._close_callers.update(callers)
        try:
            await asyncio.shield(task)
        finally:
            self._close_callers.difference_update(callers)

    async def _close(self, reason: str) -> None:
        try:
            await self._release(reason)
        finally:
            self._closed.set()
            self.emit("close", SessionClosed(reason))

    async def _release(self, reason: str) -> None:
        self._cancel_barge_in_timer()
        exempt = self._close_callers
        await cancel_and_wait(*[t for t in self._loops if t not in exempt])
        await self._tasks.cancel_all(exclude=exempt)
        self._out.close()
        steps: list[tuple[str, Callable[[], Any]]] = []
        if self._conn is not None:
            steps.append(("engine connection close", self._conn.aclose))
        if self._transport is not None:
            steps.append(("transport close", self._transport.aclose))
        if self._processors is not None:
            steps.append(("audio processor close", self._processors.close))
        steps.append(("state change", lambda: self._set_agent_state(AgentState.CLOSED)))
        if self._agent is not None:
            steps.append(("on_exit", functools.partial(self._agent.on_exit, self)))
            if self._taps:
                steps.append(("taps", lambda: self._notify("session_closing", self, reason, now())))
        for what, step in steps:
            try:
                result = step()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("session close: %s failed", what)

    def _on_loop_done(self, task: asyncio.Task[Any]) -> None:
        """A session loop failed: report it as fatal and close the session."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        if self._closing:  # a consequence of the close (e.g. a socket torn down under it)
            logger.warning("session loop %s failed while closing: %r", task.get_name(), exc)
            return
        logger.error("session loop %s failed", task.get_name(), exc_info=exc)
        error = exc if isinstance(exc, Exception) else RuntimeError(repr(exc))
        self.emit("error", SessionError(error, recoverable=False))
        reason = "engine_error" if task.get_name() == "session-events" else "transport_error"
        self._schedule_close(reason)

    async def __aenter__(self) -> AgentSession:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # --------------------------------------------------------------- public API
    async def say(self, text: str, *, allow_interruptions: bool | None = None) -> None:
        """Speak ``text`` (verbatim when the engine supports it).

        Args:
            allow_interruptions: ``False`` makes this utterance uninterruptible (the user's
                audio is discarded while it plays, see
                ``SessionOptions.discard_audio_if_uninterruptible``); ``None`` follows
                ``SessionOptions.allow_interruptions``.
        """
        await self._say(text, allow_interruptions)

    async def _say(
        self, text: str, allow_interruptions: bool | None, *, aside: bool = False
    ) -> None:
        # the engine gives no handle for the response it starts: the next one is ours
        self._reply_requests += 1
        request = _SayRequest(allow_interruptions, aside)
        self._say_requests.append(request)
        try:
            await self.connection.say(text)
        except BaseException:
            with contextlib.suppress(ValueError):
                self._say_requests.remove(request)
            request.done.set()
            raise

    async def generate_reply(
        self, *, instructions: str | None = None, user_input: str | None = None
    ) -> None:
        """Make the agent respond now, optionally to a typed ``user_input``."""
        self._reply_requests += 1
        self._tool_steps = 0  # a new request chain
        if user_input is not None:
            msg = self.history.add_message("user", user_input)
            self.emit("conversation_item", ConversationItemAdded(msg))
            await self.connection.send_text(user_input, respond=True)
        else:
            await self.connection.create_response(instructions=instructions)

    async def interrupt(self) -> None:
        """Stop the agent's current response immediately."""
        if self._barge is not None:
            self._end_barge_in(self._barge)  # the app decided: no verdict needed
        await self._interrupt()
        await self._ensure_playback()

    async def update_instructions(self, instructions: str) -> None:
        self.agent.instructions = instructions
        await self.connection.update(instructions=instructions)

    def update_endpointing(
        self, *, mode: Literal["fixed", "dynamic"] | None = None, dictation: bool | None = None
    ) -> None:
        """Switch the cascade's endpointing policy and/or dictation mode from the next user
        pause on (e.g. from a tool before the user reads out a phone number). See
        ``docs/concepts/endpointing.md``. Engines that own turn detection themselves
        (native speech-to-speech models) raise :class:`ConfigurationError`."""
        update = getattr(self.connection, "update_endpointing", None)
        if update is None:
            raise ConfigurationError(
                f"{type(self.connection).__name__} does not support endpointing updates"
            )
        update(mode=mode, dictation=dictation)

    async def handoff(
        self,
        agent: Agent,
        *,
        history: HistoryMode = "full",
        respond: bool = True,
        summary_llm: LLM | None = None,
    ) -> None:
        """Hand the conversation to ``agent`` now (from application code; tools return the
        agent instead, see :mod:`~voice_agent_next.session.handoff`).

        Args:
            history: what ``agent`` sees of the conversation (see
                :data:`~voice_agent_next.session.handoff.HistoryMode`).
            respond: let ``agent`` speak right away (its greeting, or a generated reply)
                unless the user or the agent is talking.
            summary_llm: the LLM for ``history="summary"`` (default: the cascade's LLM).
        """
        request = Handoff(agent, history=history, respond=respond, summary_llm=summary_llm)
        await self._apply_handoff(request, call=None, respond=not self._agent_or_user_talking())

    async def _apply_handoff(
        self, handoff: Handoff, *, call: FunctionCall | None, respond: bool
    ) -> bool:
        """Switch to ``handoff.agent``; returns whether a reply was started (by the new
        agent's ``on_enter``, its greeting or a generated reply)."""
        async with self._handoff_lock:
            if self._closing:
                return False
            started = now()
            old, new = self.agent, handoff.agent
            try:
                await old.on_exit(self)
            except Exception:
                logger.exception("on_exit of agent %s failed", old.name)
            ctx = await self._carried_history(handoff)
            self._agent = new
            conn = self.connection
            unsupported: list[str] = []
            try:
                await conn.update(instructions=new.instructions, tools=list(new.tools))
            except Exception as exc:
                logger.warning("engine update failed during the handoff to %s: %s", new.name, exc)
                self.emit("error", SessionError(exc, recoverable=True))
            if ctx is not None and not await self._try_engine(conn.update_chat_ctx, ctx):
                unsupported.append("chat_ctx")
            voice_changed = False
            if new.voice is not None and new.voice != self._voice:
                voice_changed = await self._try_engine(conn.update_voice, new.voice)
                if voice_changed:
                    self._voice = new.voice
                else:
                    unsupported.append("voice")
            if unsupported:
                logger.info(
                    "handoff to %s: the engine cannot change %s mid-session; keeping the "
                    "current one", new.name, " or ".join(unsupported),
                )  # fmt: skip
            self.emit(
                "agent_handoff",
                AgentHandoff(
                    from_agent=old.name,
                    to_agent=new.name,
                    history=history_mode_name(handoff.history),
                    call=call,
                    voice_changed=voice_changed,
                    unsupported=unsupported,
                    duration=now() - started,
                ),
            )
            requests = self._reply_requests
            try:
                await new.on_enter(self)
            except Exception as exc:
                logger.exception("on_enter of agent %s failed", new.name)
                self.emit("error", SessionError(exc, recoverable=True))
            if self._reply_requests != requests:
                return True  # on_enter spoke
            if not (respond and handoff.respond) or self._closing:
                return False
            if new.greeting:
                await self.say(new.greeting)
            else:
                await self.generate_reply()
            return True

    async def _carried_history(self, handoff: Handoff) -> ChatContext | None:
        """The context the new agent starts with (``None``: keep the model's context)."""
        llm = handoff.summary_llm
        if llm is None and handoff.history == "summary":
            from ..engines.cascade import CascadeEngine

            if isinstance(self.engine, CascadeEngine):
                llm = self.engine.llm
        try:
            ctx = await carry_history(self.history, handoff.history, llm=llm)
        except Exception as exc:
            logger.warning("history carry-over failed (%s); keeping the full history", exc)
            ctx = None
        own = handoff.agent.chat_ctx
        if own is not None and own.items:  # the new agent's own seed goes first
            base = ctx if ctx is not None else self.history
            ctx = ChatContext([*own.items, *base.items])
        return ctx

    @staticmethod
    async def _try_engine(method: Callable[[Any], Awaitable[bool]], arg: Any) -> bool:
        try:
            return bool(await method(arg))
        except Exception as exc:
            logger.warning("engine %s failed: %s", getattr(method, "__name__", method), exc)
            return False

    def cancel_tool_call(self, call_id: str) -> bool:
        """Cancel a running tool call (e.g. a non-blocking one the user no longer needs).

        A cancelled call emits ``tool_cancelled`` and sends no output. Returns ``False`` if
        no such call is running.
        """
        return self._cancel_run(call_id)

    def delegate(
        self,
        work: Awaitable[Any] | Callable[[], Awaitable[Any]],
        *,
        name: str = "task",
        scheduling: ToolScheduling = "when_idle",
        timeout: float | None = None,
    ) -> asyncio.Task[Any]:
        """Run ``work`` in the background, independently of turns and interruptions
        (talker/thinker): the conversation goes on, and when it finishes its result (or
        error) is added to the conversation according to ``scheduling``.

        Returns the task running ``work``; cancelling it withdraws the work (the model is
        told silently).

        Example:
            >>> session.delegate(research(topic), name="research")
        """
        if scheduling not in ("interrupt", "when_idle", "silent"):
            raise ValueError(f"unknown scheduling {scheduling!r}")
        coro = work() if callable(work) else work

        async def run() -> Any:
            _OWNER_TASK.set(asyncio.current_task())
            return await (asyncio.wait_for(coro, timeout) if timeout is not None else coro)

        task: asyncio.Task[Any] = self._tasks.spawn(run(), name=f"delegate-{name}")
        self._tasks.spawn(self._deliver_delegated(task, name, scheduling), name="delegate")
        return task

    async def _deliver_delegated(
        self, task: asyncio.Task[Any], name: str, scheduling: ToolScheduling
    ) -> None:
        await asyncio.wait([task])
        meta = {"delegated": name}
        if task.cancelled():
            await self._inject(f"The background task {name} was cancelled.", "silent", meta)
            return
        exc = task.exception()
        if isinstance(exc, TimeoutError):
            text = f"Result of the background task {name}: it timed out."
        elif exc is not None:
            text = f"Result of the background task {name}: it failed: {type(exc).__name__}: {exc}"
        else:
            text = f"Result of the background task {name}: {_stringify(task.result())}"
        await self._inject(text, scheduling, meta)

    async def report_tool_progress(
        self, call: FunctionCall, message: str, *, speak: bool = True, to_model: bool = False
    ) -> bool:
        """Progress of a running tool call (usually via ``ToolContext.report_progress``).

        Emits ``tool_progress``; with ``speak`` the agent says ``message`` unless someone is
        talking (spoken progress replaces the round's filler); with ``to_model`` it is
        added to the model's context without a response. Returns whether it is spoken.
        """
        spoken = speak and self._conn is not None and not self._agent_or_user_talking()
        if spoken:
            for resp in self._pending_rounds:
                if any(r.call.call_id == call.call_id for r in resp.tool_runs):
                    resp.filler_said = True  # the user heard something: no filler after it
        self.emit("tool_progress", ToolProgress(call, message, spoken))
        if spoken:
            await self._say(message, self.options.tool_filler_interruptible, aside=True)
        if to_model:
            text = f"Progress of {call.name} (call {call.call_id}): {message}"
            meta = {"tool_progress": True, "tool_call_id": call.call_id}
            msg = self.history.add_message("user", text, metadata=meta)
            self.emit("conversation_item", ConversationItemAdded(msg))
            await self.connection.send_text(text, respond=False)
        return spoken

    # ------------------------------------------------------------------- loops
    async def _input_loop(self) -> None:
        transport, conn = self.transport, self.connection
        discarding = False
        # a failure here is fatal: see _on_loop_done
        async for frame in transport.audio_input():
            if self._taps:
                self._notify("user_audio", frame, now())
            if self._processors is not None:
                frame = self._processors.process_capture(frame)
            discard = self._discard_input()
            if discard and not discarding and self.user_state == UserState.SPEAKING:
                # the user is mid-utterance: silence would end it and the engine would
                # answer the fragment (cancelling the uninterruptible speech): drop it
                try:
                    await conn.clear_input()
                except Exception as exc:
                    logger.warning("engine clear_input failed: %s", exc)
                self._set_user_state(UserState.LISTENING)
            discarding = discard
            if discard:
                # uninterruptible speech is playing: the engine must not hear the user
                frame = AudioFrame(
                    bytes(len(frame.data)), frame.sample_rate, frame.channels, frame.timestamp
                )
            await conn.send_audio(frame)
        if self.options.close_on_disconnect and not self._closing:
            self._schedule_close("user_disconnected")

    async def _event_loop(self) -> None:
        self._engine_events = events = self.connection.events()
        async for ev in events:
            if self._taps:
                self._notify("engine_event", ev)
            try:
                await self._handle(ev)
            except Exception as exc:
                logger.exception("error handling engine event %s", type(ev).__name__)
                self.emit("error", SessionError(exc, recoverable=True))
        if not self._closing:
            self._schedule_close("engine_closed")

    async def _playout_loop(self) -> None:
        fmt = self.transport.output_format
        resampler = StreamResampler(fmt.sample_rate, fmt.channels)
        held: str | None = None  # the response whose audio the resampler holds back
        async for item in self._out:
            if isinstance(item, _EndOfResponse):
                resp = self._responses.get(item.response_id)
                if resp is not None and not resp.interrupted:
                    if held == item.response_id:
                        # the resampler's filter delay holds back the end of the response
                        # (e.g. 44 ms at 24 -> 8 kHz with soxr): play it with the response
                        held = None
                        await self._play_frame(resp, resampler.flush())
                    delay = max(0.0, self._virtual_end - now())
                    self._tasks.spawn(self._finish_after(resp, delay))
                continue
            resp = self._responses.get(item.response_id)
            if resp is None or resp.interrupted:
                continue
            if held is not None and held != item.response_id:
                # the tail of an interrupted response must not open the next one
                resampler = StreamResampler(fmt.sample_rate, fmt.channels)
            held = item.response_id
            await self._play_frame(resp, resampler.push(item.frame))

    async def _play_frame(self, resp: _Response, frame: AudioFrame) -> None:
        """Send one (resampled) frame of ``resp`` on the virtual playback clock."""
        if not frame:
            return
        await self._wait_for_playout_slot()
        if resp.interrupted:
            return
        t = now()
        start = max(t, self._virtual_end)
        self._virtual_end = start + frame.duration
        resp.segments.append((start, frame.duration))
        if resp.first_audio_at is None:
            resp.first_audio_at = t
            turn = resp.turn
            if turn is not None and not turn.closed and turn.first_audio_at is None:
                turn.first_audio_at = t
            self._set_agent_state(AgentState.SPEAKING)
        if self._processors is not None:
            self._processors.process_render(frame)
        if self._taps:
            self._notify("agent_audio", frame, start)
        await self.transport.write_audio(frame)

    # ------------------------------------------------------------ event handling
    async def _handle(self, ev: EngineEvent) -> None:
        if isinstance(ev, InputSpeechStarted):
            self._set_user_state(UserState.SPEAKING)
            await self._on_user_speech_started(ev)
        elif isinstance(ev, InputSpeechStopped):
            self._set_user_state(UserState.LISTENING)
            self._user_speech_end = self._wall_time(ev.audio_time, ev.timestamp)
            await self._on_user_speech_stopped(ev, self._user_speech_end)
        elif isinstance(ev, InputTranscript):
            self._on_input_transcript(ev)
            await self._on_barge_in_transcript(ev)
        elif isinstance(ev, InputCommitted):
            self._committed_items.append(ev.item_id)
            if ev.item_id not in self._user_items:
                # keep the user turn before the reply in the history even when the engine
                # delivers its transcript after the response started (e.g. OpenAI Realtime)
                placeholder = self.history.add_message(
                    "user", "", id=ev.item_id, metadata={"transcript_pending": True}
                )
                self._user_items[ev.item_id] = placeholder
            await self._on_barge_in_committed()
            if self._turn is not None:
                self._close_turn(self._turn)
            self._turn = _Turn(new_id("turn_"), self._user_speech_end, ev.timestamp)
            self._user_speech_end = None
            self._tool_steps = 0
            if self.agent_state == AgentState.LISTENING:
                self._set_agent_state(AgentState.THINKING)
        elif isinstance(ev, ResponseStarted):
            await self._on_barge_in_superseded()
            turn = self._turn if self._turn is not None and not self._turn.closed else None
            request = self._say_request_for(ev.timestamp)
            if request is not None and request.aside:
                turn = None  # a filler: the turn goes on with the tool round's answer
            started = _Response(ev.response_id, ev.timestamp, turn)
            if request is not None:
                started.allow_interruptions = request.allow_interruptions
                started.aside = request.aside
                started.say = request
            self._responses[ev.response_id] = started
            self._current = started
            if self.agent_state != AgentState.SPEAKING:
                self._set_agent_state(AgentState.THINKING)
        elif isinstance(ev, ResponseAudio):
            resp = self._responses.get(ev.response_id)
            if resp is not None and not resp.interrupted:
                resp.item_id = resp.item_id or ev.item_id
                resp.received += ev.frame.duration
                self._out.send_nowait(ev)
        elif isinstance(ev, ResponseText):
            self._on_response_text(ev)
        elif isinstance(ev, ResponseToolCall):
            self._on_tool_call(ev)
        elif isinstance(ev, ResponseDone):
            await self._on_barge_in_response_done(ev)
            self._on_response_done(ev)
        elif isinstance(ev, ToolCallCancelled):
            for call_id in ev.call_ids:
                self._cancel_run(call_id)
        elif isinstance(ev, EngineErrorEvent):
            self.emit("error", SessionError(ev.error, ev.recoverable))
            if not ev.recoverable:
                self._schedule_close("engine_error")

    def _say_request_for(self, started_at: float) -> _SayRequest | None:
        """The ``say()`` that started a response at ``started_at`` (if any)."""
        requests = self._say_requests
        while requests and started_at - requests[0].created > _SAY_REQUEST_TTL:
            requests.popleft().done.set()  # the engine never started a response for it
        if requests and started_at >= requests[0].created:
            return requests.popleft()
        return None

    def _on_input_transcript(self, ev: InputTranscript) -> None:
        self.emit("user_transcript", UserTranscript(ev.text, ev.is_final, ev.item_id, ev.language))
        if not ev.is_final:
            return
        msg = self._user_items.get(ev.item_id)
        if msg is None:
            msg = self.history.add_message("user", ev.text, id=ev.item_id)
            self._user_items[ev.item_id] = msg
            self.emit("conversation_item", ConversationItemAdded(msg))
        else:
            msg.content = [ev.text]
            if msg.metadata.pop("transcript_pending", False):
                self.emit("conversation_item", ConversationItemAdded(msg))
        if self._agent is not None:
            self._tasks.spawn(self._agent.on_user_turn_completed(self, msg))

    def _on_response_text(self, ev: ResponseText) -> None:
        resp = self._responses.get(ev.response_id)
        if resp is None or resp.interrupted:
            return
        resp.item_id = resp.item_id or ev.item_id
        resp.text.append(ev.delta)
        if resp.message is None:
            resp.message = self.history.add_message("assistant", "", id=ev.item_id)
        resp.message.content = ["".join(resp.text).strip()]
        self.emit("agent_transcript", AgentTranscript(ev.delta, ev.item_id, ev.response_id))

    def _on_tool_call(self, ev: ResponseToolCall) -> None:
        resp = self._responses.get(ev.response_id)
        call = ev.call
        self.history.append(call)
        self.emit("conversation_item", ConversationItemAdded(call))
        self.emit("tool_call", ToolCalled(call))
        if resp is not None and resp.turn is not None and not resp.turn.closed:
            resp.turn.tool_calls += 1  # now: the turn may close before the round completes
        tool = find_tool(self.agent.tools, call.name)
        task = self._tasks.spawn(self._run_tool(call), name=f"tool-{call.name}")
        run = _ToolRun(call, tool, task)
        self._tool_runs[call.call_id] = run
        task.add_done_callback(functools.partial(self._forget_run, run))
        # a delegating engine (GPT-Live) keeps the conversation going while its backend
        # waits: every result is delivered when ready, whatever the tool's ``blocking``
        delegated = self.connection.capabilities.tool_mode == "delegation"
        if (tool is not None and not tool.blocking) or delegated:
            native = self.connection.capabilities.tool_mode != "blocking"
            self._tasks.spawn(self._deliver_later(run, native=native), name=f"tool-{call.name}")
            if native:  # the model does not wait: nothing to answer now
                return
            # the model waits for an output: acknowledge now, deliver the result later
            ack: asyncio.Future[FunctionCallOutput] = asyncio.get_running_loop().create_future()
            text = tool.ack if tool is not None and tool.ack is not None else DEFAULT_TOOL_ACK
            ack.set_result(FunctionCallOutput(call_id=call.call_id, name=call.name, output=text))
            run = _ToolRun(call, tool, ack, ack=True)
        if resp is not None:
            resp.tool_calls.append(call)
            resp.tool_runs.append(run)
            self._pending_rounds.add(resp)
            if not run.ack:
                self._start_watchdog(resp)
        elif run.ack:  # no round to answer with: acknowledge on its own
            output = run.task.result()
            self._tasks.spawn(self.connection.send_tool_output(output, respond=False))

    def _forget_run(self, run: _ToolRun, _: object = None) -> None:
        if self._tool_runs.get(run.call.call_id) is run:
            del self._tool_runs[run.call.call_id]

    def _cancel_run(self, call_id: str) -> bool:
        run = self._tool_runs.get(call_id)
        if run is None or run.task.done():
            return False
        run.cancelled = True
        run.task.cancel()
        return True

    async def _run_tool(self, call: FunctionCall) -> FunctionCallOutput:
        _OWNER_TASK.set(asyncio.current_task())  # (this task's own context)
        ctx: ToolContext[UserdataT] = ToolContext(call=call, session=self, userdata=self.userdata)
        tools = self.agent.tools
        tool = find_tool(tools, call.name)
        if tool is not None and tool.fn is not None:
            tools = [self._capture_handoff(tool, ctx)]
        output = await execute_function_call(
            call, tools, ctx=ctx, timeout=self.options.tool_timeout
        )
        handoff = ctx.pending_handoff
        if handoff is not None and not output.is_error:
            self._handoffs[call.call_id] = handoff
            if not output.output:
                output.output = handoff.output
        return output

    @staticmethod
    def _capture_handoff(tool: FunctionTool, ctx: ToolContext[Any]) -> FunctionTool:
        """``tool`` whose returned :class:`Agent`/:class:`Handoff` is recorded on ``ctx``
        (the model gets the handoff's message as the output)."""
        fn = tool.fn
        assert fn is not None
        is_async = inspect.iscoroutinefunction(fn)

        async def run(**kwargs: Any) -> Any:
            result = await fn(**kwargs) if is_async else await asyncio.to_thread(fn, **kwargs)
            handoff, rest = as_handoff(result)
            if handoff is not None:
                ctx.pending_handoff = handoff
            return rest

        return dataclasses.replace(tool, fn=run)

    def _take_handoff(self, runs: Sequence[_ToolRun]) -> tuple[FunctionCall, Handoff] | None:
        """The handoff requested by a finished (blocking) call of a tool round."""
        found: tuple[FunctionCall, Handoff] | None = None
        for run in runs:
            if run.ack:
                continue  # the real run is delivered later (see _deliver_later)
            handoff = self._handoffs.pop(run.call.call_id, None)
            if handoff is None or run.outcome() is None:
                continue
            if found is not None:
                logger.warning(
                    "several handoffs in one tool round: %s wins over %s",
                    handoff.agent.name, found[1].agent.name,
                )  # fmt: skip
            found = (run.call, handoff)
        return found

    def _on_response_done(self, ev: ResponseDone) -> None:
        resp = self._responses.get(ev.response_id)
        if resp is None:
            return
        resp.done = True
        resp.status = ev.status
        if resp.say is not None:
            resp.say.done.set()
        if resp.tool_runs:
            self._tasks.spawn(self._complete_tools(resp))
        if resp.interrupted:
            return
        if resp.first_audio_at is None and not resp.segments and self._out.empty():
            # nothing was (or will be) played for this response
            self._tasks.spawn(self._finish_after(resp, 0.0))
        else:
            self._out.send_nowait(_EndOfResponse(ev.response_id))

    async def _complete_tools(self, resp: _Response) -> None:
        await asyncio.wait([run.task for run in resp.tool_runs])
        self._pending_rounds.discard(resp)
        if resp.watchdog is not None and not resp.filler_said:
            resp.watchdog.cancel()  # the round finished first: no filler
        outputs: list[FunctionCallOutput] = []
        for run in resp.tool_runs:
            output = run.outcome()
            if output is None:  # withdrawn by the engine: it expects no output
                self.emit("tool_cancelled", ToolCancelled(run.call, now() - run.started))
                continue
            outputs.append(output)
            self.history.append(output)
            self.emit("conversation_item", ConversationItemAdded(output))
            if not run.ack:
                self.emit("tool_result", ToolResult(run.call, output, now() - run.started))
        self._tool_steps += 1
        respond = (
            bool(outputs)
            and not resp.interrupted
            and resp.status == "completed"
            and self.user_state != UserState.SPEAKING
            and self._tool_steps <= self.options.max_tool_steps
        )
        if self._tool_steps > self.options.max_tool_steps:
            logger.warning(
                "max_tool_steps (%d) reached; not responding", self.options.max_tool_steps
            )
        handoff = self._take_handoff(resp.tool_runs)
        if respond:
            await self._wait_asides()  # the follow-up response would cut a filler off
        for i, output in enumerate(outputs):
            await self.connection.send_tool_output(
                output, respond=respond and handoff is None and i == len(outputs) - 1
            )
        if handoff is not None:  # the new agent answers (the old one is done)
            call, request = handoff
            respond = await self._apply_handoff(request, call=call, respond=respond)
        if not respond and not resp.interrupted:
            if resp.turn is not None:
                self._close_turn(resp.turn)
            if self._current is resp or self._current is None:
                self._current = None
                self._set_agent_state(AgentState.LISTENING)

    async def _wait_asides(self) -> None:
        """Wait until pending fillers/progress utterances have been generated."""
        pending = [r.done for r in self._say_requests if r.aside and not r.done.is_set()]
        pending += [
            r.say.done
            for r in self._responses.values()
            if r.aside and r.say is not None and not r.say.done.is_set()
        ]
        if pending:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    asyncio.gather(*(done.wait() for done in pending)), _ASIDE_TIMEOUT
                )

    # ------------------------------------------------------------ tool fillers
    def _start_watchdog(self, resp: _Response) -> None:
        delay = self.options.tool_filler_delay
        if (
            resp.watchdog is not None
            or delay is None
            or self.connection.capabilities.tool_mode != "blocking"
        ):
            return
        resp.watchdog = self._tasks.spawn(self._tool_watchdog(resp, delay), name="tool-watchdog")

    async def _tool_watchdog(self, resp: _Response, delay: float) -> None:
        """Say a filler once the round has kept the conversation silent for ``delay`` s."""
        started = quiet_since = now()
        while True:
            wait = quiet_since + delay - now()
            if wait > 0:
                await asyncio.sleep(wait)
            if resp.filler_said or resp not in self._pending_rounds or self._closing:
                return
            if resp.interrupted or (resp.turn is not None and resp.turn is not self._turn):
                return  # the user moved on
            if self._agent_or_user_talking():
                await asyncio.sleep(_IDLE_POLL)
                quiet_since = now()  # a filler right after speech would be odd: start over
                continue
            running = [r for r in resp.tool_runs if not r.task.done()]
            text = self._filler_for(running)
            resp.filler_said = True
            if text is None:
                return
            calls = [r.call for r in running]
            self.emit("tool_filler", ToolFiller(text, calls, now() - started))
            await self._say(text, self.options.tool_filler_interruptible, aside=True)
            return

    def _filler_for(self, runs: Sequence[_ToolRun]) -> str | None:
        """The filler of the first running call whose tool has one."""
        for run in runs:
            spec = run.tool.filler if run.tool is not None else None
            if spec is False:
                continue
            if spec is None or spec is True:
                phrases: Sequence[str] = self.options.tool_fillers
            elif isinstance(spec, str):
                phrases = (spec,)
            elif callable(spec):
                try:
                    text = spec(run.call)
                except Exception:
                    logger.exception("filler callable of tool %s failed", run.call.name)
                    continue
                if text and text.strip():
                    return text
                continue
            else:
                phrases = spec
            choice = self._fillers.pick(phrases)
            if choice is not None:
                return choice
        return None

    def _agent_or_user_talking(self) -> bool:
        return (
            self.user_state == UserState.SPEAKING
            or self._barge is not None
            or self._is_responding()
            or any(self._aside_pending(r) for r in self._say_requests)
        )

    @staticmethod
    def _aside_pending(request: _SayRequest) -> bool:
        """A filler/progress utterance was requested but its response has not started."""
        return (
            request.aside
            and not request.done.is_set()
            and now() - request.created < _SAY_REQUEST_TTL
        )

    # ------------------------------------------------- non-blocking tools/delegation
    async def _deliver_later(self, run: _ToolRun, *, native: bool) -> None:
        """Deliver the result of a non-blocking call once it is ready."""
        await asyncio.wait([run.task])
        tool = run.tool
        scheduling: ToolScheduling = tool.scheduling if tool is not None else "when_idle"
        output = run.outcome()
        call = run.call
        handoff = self._handoffs.pop(call.call_id, None)
        if handoff is not None and output is not None:
            await self._deliver_later_handoff(run, output, handoff, native=native)
            return
        if output is None:
            self.emit("tool_cancelled", ToolCancelled(call, now() - run.started))
            if not native:  # the model was told a result would follow
                note = f"The background task {call.name} (call {call.call_id}) was cancelled."
                await self._inject(note, "silent", {"tool_call_id": call.call_id})
            return
        self.emit("tool_result", ToolResult(call, output, now() - run.started, blocking=False))
        if native:
            self.history.append(output)
            self.emit("conversation_item", ConversationItemAdded(output))
            await self.connection.send_async_tool_output(output, scheduling=scheduling)
            return
        what = f"the background task {call.name} (call {call.call_id})"
        if output.is_error:
            text = f"Result of {what}: it failed: {output.output}"
        else:
            text = f"Result of {what}: {output.output}"
        await self._inject(text, scheduling, {"tool_call_id": call.call_id})

    async def _deliver_later_handoff(
        self, run: _ToolRun, output: FunctionCallOutput, handoff: Handoff, *, native: bool
    ) -> None:
        """A non-blocking call asked for a handoff: tell the model silently, then switch at
        the next quiet moment."""
        call = run.call
        self.emit("tool_result", ToolResult(call, output, now() - run.started, blocking=False))
        if native:
            self.history.append(output)
            self.emit("conversation_item", ConversationItemAdded(output))
            await self.connection.send_async_tool_output(output, scheduling="silent")
            await self._wait_for_quiet(interrupt=False)
        else:
            text = f"Result of the background task {call.name} (call {call.call_id}): "
            await self._inject(text + output.output, "silent", {"tool_call_id": call.call_id})
        if not self._closing:
            await self._apply_handoff(handoff, call=call, respond=True)

    async def _inject(self, text: str, scheduling: ToolScheduling, meta: dict[str, Any]) -> None:
        """Add a background result to the conversation at a moment that fits ``scheduling``."""
        await self._wait_for_quiet(interrupt=scheduling == "interrupt")
        if self._closing:
            return
        if scheduling == "interrupt" and self._is_responding():
            if self._barge is not None:
                self._end_barge_in(self._barge)
            await self._interrupt()
            await self._ensure_playback()
        metadata = {"background_result": True, **meta}
        msg = self.history.add_message("user", text, metadata=metadata)
        self.emit("conversation_item", ConversationItemAdded(msg))
        if scheduling != "silent":
            self._tool_steps = 0  # the response to a background result is a new chain
        await self.connection.send_text(text, respond=scheduling != "silent")

    async def _wait_for_quiet(self, *, interrupt: bool) -> None:
        """Wait until nobody talks, no tool round is pending and no reply is on its way
        (``interrupt``: the agent may be talking; only the user and tool rounds count)."""

        def quiet() -> bool:
            if self._pending_rounds or self.user_state == UserState.SPEAKING:
                return False
            if interrupt:
                return True
            return self.agent_state == AgentState.LISTENING and not self._agent_or_user_talking()

        while not self._closing:
            if quiet():
                await asyncio.sleep(_IDLE_SETTLE)  # e.g. a reply about to start
                if quiet():
                    return
            await asyncio.sleep(_IDLE_POLL)

    async def _finish_after(self, resp: _Response, delay: float) -> None:
        pauses = self._pause_seq
        if delay > 0:
            await asyncio.sleep(delay)
        if pauses != self._pause_seq or self._clock_paused_at is not None:
            await self._wait_playout_end(resp)  # paused meanwhile: its audio ends later
        await self._wait_until_heard(resp)
        if resp.finished or resp.interrupted:
            return
        resp.finished = True
        await self._release_barge_in(resp)
        if resp.turn is not None and not resp.turn.closed:  # its metrics are already out
            resp.turn.agent_speech += resp.played(now())
        if resp.tool_calls:
            return  # the turn continues once the tool results are sent back
        if resp.turn is not None:
            self._close_turn(resp.turn)
        if self._current is resp and self.agent_state != AgentState.CLOSED:
            self._current = None
            # after a filler, the agent is still busy with the tool round
            pending = any(not r.interrupted for r in self._pending_rounds)
            self._set_agent_state(AgentState.THINKING if pending else AgentState.LISTENING)

    async def _wait_for_playout_slot(self) -> None:
        """Pace against the virtual playback clock; hold audio back while paused."""
        while True:
            await self._send_gate.wait()
            ahead = self._virtual_end - now()
            if ahead > self.options.output_lookahead:
                pauses = self._pause_seq
                await asyncio.sleep(ahead - self.options.output_lookahead)
                if pauses != self._pause_seq:
                    continue  # paused meanwhile: the timeline may have moved
            if self._send_gate.is_set():
                return

    def _listener_time(self, t: float) -> float:
        """Where the listener is on the playback timeline at time ``t``.

        The virtual clock assumes audio is heard the moment it is due. A transport with
        ``capabilities.playback_position`` knows better (device output latency, a
        client-side jitter buffer, playback marks): the part of its backlog that the
        virtual clock does not account for is how far the listener lags behind.
        """
        transport = self._transport
        if transport is None or not transport.capabilities.playback_position:
            return t
        try:
            buffered = transport.buffered_duration()
        except Exception:
            return t
        return t - (buffered - max(0.0, self._virtual_end - t))

    async def _wait_until_heard(self, resp: _Response) -> None:
        """Wait until the listener has heard ``resp`` to the end (transport lag included)."""
        deadline = now() + _MAX_PLAYBACK_LAG
        while not (resp.finished or resp.interrupted):
            if self._clock_paused_at is not None:
                await self._clock_running.wait()
                deadline = now() + _MAX_PLAYBACK_LAG
                continue
            remaining = resp.end - self._listener_time(now())
            if remaining <= 0.005 or now() >= deadline:
                return
            await asyncio.sleep(min(remaining, 0.25))  # re-check: the estimate moves

    async def _wait_playout_end(self, resp: _Response) -> None:
        """Wait until the audio sent for ``resp`` has been played, pauses included."""
        while resp.segments and not (resp.finished or resp.interrupted):
            paused_at = self._clock_paused_at
            if paused_at is not None:
                if resp.end <= paused_at:
                    return
                await self._clock_running.wait()
                continue
            remaining = resp.end - now()
            if remaining <= 0:
                return
            await asyncio.sleep(remaining)

    # ------------------------------------------------------------- interruption
    def _can_interrupt(self, resp: _Response) -> bool:
        if resp.allow_interruptions is not None:
            return resp.allow_interruptions
        return self.options.allow_interruptions

    def _is_responding(self) -> bool:
        resp = self._current
        if resp is None or resp.interrupted or resp.finished:
            return False
        if self._clock_paused_at is not None:
            return True  # paused mid-speech
        t = now()
        if not resp.done or self._virtual_end > t or not self._out.empty():
            return True
        return self._listener_time(t) < resp.end  # the listener still hears the tail

    def _discard_input(self) -> bool:
        """True while uninterruptible speech plays (and discarding is enabled)."""
        if not self.options.discard_audio_if_uninterruptible:
            return False
        resp = self._current
        return resp is not None and not self._can_interrupt(resp) and self._is_responding()

    def _wall_time(self, audio_time: float | None, fallback: float) -> float:
        """``now()`` time at which an input-stream position was captured."""
        if audio_time is not None and self._conn is not None:
            wall = self._conn.audio_time_to_wall(audio_time)
            if wall is not None:
                return wall
        return fallback

    async def _on_user_speech_started(self, ev: InputSpeechStarted) -> None:
        # where speech began; an onset mapped earlier than the VAD could plausibly report it
        # comes from a gap in the input stream, not from speech
        start = self._wall_time(ev.audio_time, ev.timestamp)
        start = min(max(start, ev.timestamp - _MAX_ONSET_LAG), now())
        barge = self._barge
        if barge is not None:
            barge.overlap.speech_started(start)  # the same overlap goes on
            barge.input_cleared = False
            await self._evaluate_barge_in()
            return
        resp = self._current
        if resp is None or not self._can_interrupt(resp) or not self._is_responding():
            return
        opts = self.options
        if opts.min_interruption_duration <= 0 and opts.min_interruption_words <= 0:
            await self._interrupt()  # no policy: interrupt at the first sign of speech
            return
        language = self._agent.language if self._agent is not None else None
        policy = InterruptionPolicy.from_options(opts, language=language)
        overlap = Overlap.begin(policy, start, held=policy.pauses)
        barge = _BargeIn(overlap, resp, frozenset(self._committed_items))
        self._barge = barge
        self._defer_engine_commits(True)
        if policy.pauses:
            await self._pause_playback(barge)
        await self._evaluate_barge_in()

    async def _on_user_speech_stopped(self, ev: InputSpeechStopped, speech_end: float) -> None:
        if self._barge is not None:
            self._barge.overlap.speech_stopped(speech_end, ev.timestamp)
            await self._evaluate_barge_in()

    async def _on_barge_in_transcript(self, ev: InputTranscript) -> None:
        barge = self._barge
        if barge is None or ev.item_id in barge.ignore_items:
            return
        final = ev.is_final or ev.segment_final
        barge.overlap.add_transcript(ev.item_id, ev.text, final=final)
        await self._evaluate_barge_in()

    async def _on_barge_in_committed(self) -> None:
        if self._barge is not None:
            # the engine took the user's turn (and answers it): the agent's speech is over
            await self._close_barge_in(self._barge, interrupt=True)

    async def _on_barge_in_superseded(self) -> None:
        barge = self._barge
        if barge is not None:
            # another response starts: paused speech can no longer be resumed
            await self._close_barge_in(barge, interrupt=barge.paused_at is not None)

    async def _on_barge_in_response_done(self, ev: ResponseDone) -> None:
        barge = self._barge
        if (
            barge is None
            or barge.overlap.confirmed
            or ev.status != "cancelled"
            or barge.response.response_id != ev.response_id
        ):
            return
        # the engine cancelled the response itself (e.g. server-side VAD): resuming is
        # impossible, so treat it as a real interruption and keep watching for a false one
        barge.overlap.confirmed = True
        self._defer_engine_commits(False)
        await self._interrupt(barge.response, cancel=False)
        self._watch_aftermath(barge)

    async def _evaluate_barge_in(self) -> None:
        barge = self._barge
        if barge is None:
            return
        if self._engine_events_pending():
            # decide on everything the engine already reported: e.g. a queued InputCommitted
            # means it took the user's turn, and a cancel would hit the response it started
            self._schedule_barge_in_check(barge, delay=_RECHECK_DELAY)
            return
        overlap = barge.overlap
        if overlap.not_a_turn() and not barge.input_cleared:
            # keep the engine from committing it (and answering it, which would cancel the
            # paused speech)
            barge.input_cleared = True
            try:
                await self.connection.clear_input()
            except Exception as exc:
                logger.warning("engine clear_input failed: %s", exc)
            if self._barge is not barge:
                return
        elif overlap.is_turn():
            self._defer_engine_commits(False)  # a real utterance: the engine may commit it
        verdict = overlap.verdict(now())
        if verdict is None:
            self._schedule_barge_in_check(barge)
        elif verdict == Verdict.INTERRUPT:
            overlap.confirmed = True
            self._defer_engine_commits(False)
            # only a response still being generated needs cancelling; a complete one is just
            # truncated (a cancel might hit a response the engine started meanwhile)
            await self._interrupt(barge.response, cancel=not barge.response.done)
            self._watch_aftermath(barge)
        elif verdict == Verdict.UNPAUSE:
            overlap.held = False
            if barge.paused_at is not None:
                barge.paused_total += now() - barge.paused_at
                barge.paused_at = None
                await self._resume_playback()
            if self._barge is barge:
                self._schedule_barge_in_check(barge)
        elif verdict == Verdict.RESUME:
            await self._resume_after_false_interruption(barge)
        else:
            self._end_barge_in(barge)
            if verdict == Verdict.FALSE_INTERRUPTION:
                self._emit_false_interruption(barge, resumed=False, paused=0.0)
        await self._ensure_playback()

    def _engine_events_pending(self) -> bool:
        """Engine events are queued but not handled yet (known for the base :class:`Chan`)."""
        events = self._engine_events
        return isinstance(events, Chan) and not events.empty()

    async def _ensure_playback(self) -> None:
        """Invariant: audio is only held back while an overlap has playback paused."""
        barge = self._barge
        if not self._send_gate.is_set() and (barge is None or barge.paused_at is None):
            await self._resume_playback()

    def _watch_aftermath(self, barge: _BargeIn) -> None:
        """After a confirmed interruption, watch whether the user really takes the turn."""
        if self._barge is not barge:
            return
        if barge.overlap.policy.false_interruption_timeout is None:
            self._end_barge_in(barge)
        else:
            self._schedule_barge_in_check(barge)

    async def _resume_after_false_interruption(self, barge: _BargeIn) -> None:
        self._end_barge_in(barge)
        if barge.paused_at is not None:
            barge.paused_total += now() - barge.paused_at
            barge.paused_at = None
            await self._resume_playback()
        if barge.paused_total > 0:  # else the agent never stopped talking: nothing to report
            self._emit_false_interruption(barge, resumed=True, paused=barge.paused_total)

    async def _release_barge_in(self, resp: _Response) -> None:
        """``resp`` finished playing while its overlap was pending: nothing to decide."""
        barge = self._barge
        if barge is not None and barge.response is resp and not barge.overlap.confirmed:
            await self._close_barge_in(barge, interrupt=False)

    async def _close_barge_in(self, barge: _BargeIn, *, interrupt: bool) -> None:
        """Stop tracking ``barge``: interrupt its response (truncate only: the engine has
        moved on) or, if it is still paused, let playback continue."""
        self._end_barge_in(barge)
        paused = barge.paused_at is not None
        barge.paused_at = None
        resp = barge.response
        if interrupt and not barge.overlap.confirmed and not (resp.interrupted or resp.finished):
            barge.overlap.confirmed = True
            await self._interrupt(resp, cancel=False)  # also restarts playback
        elif paused:
            await self._resume_playback()

    def _end_barge_in(self, barge: _BargeIn) -> None:
        if self._barge is barge:
            self._barge = None
            self._cancel_barge_in_timer()
            self._defer_engine_commits(False)

    def _defer_engine_commits(self, deferred: bool) -> None:
        """Ask the engine (a cascade) to hold its automatic commits while an overlap awaits
        its verdict, so a backchannel can be dropped before it is answered."""
        if deferred == self._commits_deferred or self._conn is None:
            return
        defer = getattr(self._conn, "defer_commit", None)
        if defer is None:
            return
        self._commits_deferred = deferred
        defer(deferred)

    def _schedule_barge_in_check(self, barge: _BargeIn, *, delay: float | None = None) -> None:
        """Re-evaluate ``barge`` at its next deadline (or after ``delay`` seconds)."""
        self._cancel_barge_in_timer()
        if self._closing:
            return
        if delay is None:
            deadline = barge.overlap.deadline()
            if deadline is None:
                return
            delay = max(0.0, deadline - now())
        self._barge_timer = asyncio.get_running_loop().call_later(
            delay, self._on_barge_in_timer, barge
        )

    def _on_barge_in_timer(self, barge: _BargeIn) -> None:
        self._barge_timer = None
        if self._barge is barge and not self._closing:
            self._tasks.spawn(self._evaluate_barge_in(), name="session-barge-in")

    def _cancel_barge_in_timer(self) -> None:
        if self._barge_timer is not None:
            self._barge_timer.cancel()
            self._barge_timer = None

    def _emit_false_interruption(self, barge: _BargeIn, *, resumed: bool, paused: float) -> None:
        overlap = barge.overlap
        self.emit(
            "agent_false_interruption",
            AgentFalseInterruption(
                resumed=resumed,
                reason=overlap.reason(),
                response_id=barge.response.response_id,
                item_id=barge.response.item_id,
                transcript=overlap.transcript,
                speech_duration=overlap.speech_duration(now()),
                paused=paused,
            ),
        )

    # ------------------------------------------------------------ pause/resume
    async def _pause_playback(self, barge: _BargeIn) -> None:
        """Hold the agent's audio, without dropping it, until the overlap's verdict."""
        t = now()
        barge.paused_at = t
        self._pause_seq += 1
        self._send_gate.clear()
        transport = self.transport
        if transport.capabilities.pause:
            self._transport_paused = True
            self._freeze_clock(t)  # audio already queued in the transport stops too
            try:
                await transport.pause_audio()
            except Exception as exc:
                logger.warning("transport pause failed; holding back new audio only: %s", exc)
                if self._clock_paused_at == t:
                    self._transport_paused = False
                    self._unfreeze_clock()
        # else: the short look-ahead already handed to the transport plays out.
        # The agent state is left as is: LISTENING means the agent's turn is over (clients
        # finalize its transcript), which is only true once the interruption is confirmed.

    async def _resume_playback(self) -> None:
        paused_at = self._clock_paused_at
        if paused_at is not None:
            self._shift_timeline(paused_at, now() - paused_at)
            self._unfreeze_clock()
            self._pause_seq += 1  # the timeline moved: pending waits must re-check it
        transport_paused, self._transport_paused = self._transport_paused, False
        self._send_gate.set()
        if transport_paused:
            try:
                await self.transport.resume_audio()
            except Exception as exc:
                logger.warning("transport resume failed: %s", exc)

    def _freeze_clock(self, t: float) -> None:
        self._clock_paused_at = t
        self._clock_running.clear()
        if self._taps:
            self._notify("playback_paused", t)

    def _unfreeze_clock(self) -> None:
        was_paused = self._clock_paused_at is not None
        self._clock_paused_at = None
        self._clock_running.set()
        if self._taps and was_paused:
            self._notify("playback_resumed")

    def _shift_timeline(self, paused_at: float, delta: float) -> None:
        """Playback stood still from ``paused_at`` for ``delta`` s: unheard audio moves later."""
        if delta <= 0:
            return
        if self._taps:
            self._notify("playback_shifted", paused_at, delta)
        for resp in self._responses.values():
            if resp.finished or resp.interrupted or resp.end <= paused_at:
                continue
            shifted: list[tuple[float, float]] = []
            for start, dur in resp.segments:
                if start >= paused_at:
                    shifted.append((start + delta, dur))
                elif start + dur > paused_at:
                    shifted.append((start, paused_at - start))
                    shifted.append((paused_at + delta, start + dur - paused_at))
                else:
                    shifted.append((start, dur))
            resp.segments = shifted
        if self._virtual_end > paused_at:
            self._virtual_end += delta

    async def _interrupt(self, resp: _Response | None = None, *, cancel: bool = True) -> None:
        """Stop ``resp`` (default: the current response) for good; truncate it to what was
        heard. ``cancel=False`` when the engine already moved on (it committed the user's
        turn or cancelled the response itself): only tell it how much was heard."""
        if resp is None:
            resp = self._current
        if resp is None or resp.interrupted or resp.finished:
            return
        resp.interrupted = True
        barge = self._barge
        if barge is not None and barge.response is resp:
            barge.overlap.confirmed = True
            barge.paused_at = None
        paused_at = self._clock_paused_at
        played = resp.played(self._listener_time(paused_at if paused_at is not None else now()))
        if self._taps:  # audio after the pause (or after now) is never heard
            self._notify("playback_cleared", paused_at if paused_at is not None else now())
        self._out.clear()
        self._virtual_end = now()
        self._unfreeze_clock()  # paused audio is dropped, not resumed
        transport_paused, self._transport_paused = self._transport_paused, False
        with contextlib.suppress(Exception):
            await self.transport.clear_audio()
        if transport_paused:
            with contextlib.suppress(Exception):
                await self.transport.resume_audio()
        self._send_gate.set()
        heard: str | None = None
        played_ms = round(played * 1000)
        try:
            if cancel:
                heard = await self.connection.interrupt(resp.item_id, played_ms)
            elif resp.item_id is not None and self.connection.capabilities.truncation:
                heard = await self.connection.truncate(resp.item_id, played_ms)
        except Exception as exc:
            logger.warning("engine interrupt failed: %s", exc)
        if resp.message is not None:
            if heard is None:  # estimate: text is roughly proportional to audio
                full = "".join(resp.text).strip()
                total = max(resp.received, resp.sent)
                heard = full[: round(len(full) * min(1.0, played / total))] if total > 0 else ""
            resp.message.content = [heard.strip()]
            resp.message.interrupted = True
        self.emit("interrupted", Interrupted(resp.response_id, resp.item_id, played))
        if resp.turn is not None and not resp.turn.closed:
            resp.turn.interrupted = True
            resp.turn.agent_speech += played
            self._close_turn(resp.turn)
        resp.finished = True
        if self._current is resp:
            self._current = None
            self._set_agent_state(AgentState.LISTENING)

    # ------------------------------------------------------------------ metrics
    def _on_metrics(self, m: Metrics) -> None:
        self.usage.add(m)
        self.emit("metrics", m)

    def _close_turn(self, turn: _Turn) -> None:
        """Emit :class:`TurnMetrics` once per user turn."""
        if turn.closed:
            return
        turn.closed = True
        v2v = eot = ttfb = None
        if turn.first_audio_at is not None and turn.speech_end is not None:
            v2v = turn.first_audio_at - turn.speech_end
        if turn.speech_end is not None:
            eot = turn.committed_at - turn.speech_end
        if turn.first_audio_at is not None:
            ttfb = turn.first_audio_at - turn.committed_at
        self.emit(
            "metrics",
            TurnMetrics(
                turn_id=turn.turn_id,
                voice_to_voice=v2v,
                end_of_turn_delay=eot,
                response_ttfb=ttfb,
                agent_speech_duration=turn.agent_speech,
                interrupted=turn.interrupted,
                tool_calls=turn.tool_calls,
                agent=self._agent.name if self._agent is not None else None,
            ),
        )

    # ------------------------------------------------------------------- state
    def _set_agent_state(self, state: AgentState) -> None:
        if state == self.agent_state:
            return
        old, self.agent_state = self.agent_state, state
        self.emit("agent_state_changed", AgentStateChanged(old, state))

    def _set_user_state(self, state: UserState) -> None:
        if state == self.user_state:
            return
        old, self.user_state = self.user_state, state
        self.emit("user_state_changed", UserStateChanged(old, state))
