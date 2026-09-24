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
* execute tool calls and feed the results back to the engine;
* keep the conversation history and emit transcripts, state changes and metrics.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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
from ..tools import ToolContext, execute_function_call
from ..transports.base import Transport
from ..utils.aio import BackgroundTasks, Chan, cancel_and_wait
from ..utils.clock import now
from ..utils.emitter import EventEmitter
from ..utils.ids import new_id
from ..utils.log import logger
from .agent import Agent
from .events import (
    AgentFalseInterruption,
    AgentState,
    AgentStateChanged,
    AgentTranscript,
    ConversationItemAdded,
    Interrupted,
    SessionClosed,
    SessionError,
    ToolCalled,
    ToolResult,
    UserState,
    UserStateChanged,
    UserTranscript,
)
from .interruptions import InterruptionPolicy, Overlap, Verdict

if TYPE_CHECKING:
    from ..engines.cascade import CascadeOptions

__all__ = ["AgentSession", "SessionOptions"]

_MAX_ONSET_LAG = 0.5
"""Upper bound (s) on how long after its onset an engine reports user speech."""
_SAY_REQUEST_TTL = 10.0
"""A ``say()`` whose response has not started within this many seconds is forgotten."""


@dataclass(slots=True)
class SessionOptions:
    allow_interruptions: bool = True
    """Let the user barge in while the agent is speaking/thinking."""
    output_lookahead: float = 0.15
    """Seconds of agent audio handed to the transport ahead of real time."""
    tool_timeout: float | None = 30.0
    """Per-call timeout for tool execution (``None`` = no timeout)."""
    max_tool_steps: int = 5
    """Maximum consecutive tool-call rounds per user turn."""
    close_on_disconnect: bool = True
    """Close the session when the transport's audio input ends (user hung up)."""
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
    false_interruption_timeout: float | None = 2.0
    """Seconds the user must stay quiet, without meaningful words, before paused speech
    resumes. ``None`` disables pause-and-resume."""
    resume_false_interruption: bool = True
    """Pause the agent while a barge-in is unconfirmed and resume it after a false
    interruption. ``False``: the agent keeps talking until the barge-in is confirmed."""
    discard_audio_if_uninterruptible: bool = True
    """While uninterruptible speech plays, the engine receives silence instead of the
    user's audio (so it can neither barge in nor queue a turn)."""

    def __post_init__(self) -> None:
        if self.min_interruption_duration < 0:
            raise ValueError("min_interruption_duration must be >= 0")
        if self.min_interruption_words < 0:
            raise ValueError("min_interruption_words must be >= 0")
        if self.false_interruption_timeout is not None and self.false_interruption_timeout < 0:
            raise ValueError("false_interruption_timeout must be >= 0 or None")
        if isinstance(self.backchannel_words, str):
            raise TypeError("backchannel_words must be a list of words, not a string")


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


@dataclass
class _Response:
    response_id: str
    started_at: float
    turn: _Turn | None
    item_id: str | None = None
    message: ChatMessage | None = None
    text: list[str] = field(default_factory=list)
    tool_calls: list[FunctionCall] = field(default_factory=list)
    tool_tasks: list[asyncio.Task[tuple[FunctionCallOutput, float]]] = field(default_factory=list)
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
    input_cleared: bool = False
    """The engine's pending input was cleared during the current silence."""


@dataclass(eq=False, slots=True)
class _SayRequest:
    allow_interruptions: bool | None
    created: float = field(default_factory=now)


class AgentSession(EventEmitter):
    """Runs an :class:`Agent` over a :class:`Transport` with a speech-to-speech engine.

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
        userdata: Any = None,
    ) -> None:
        super().__init__()
        if engine is not None:
            if any(c is not None for c in (stt, llm, tts, turn_detector)):
                raise ConfigurationError("pass either engine=... or cascade components, not both")
            self.engine: S2SEngine = create("engine", engine)
        else:
            if llm is None or tts is None:
                raise ConfigurationError(
                    "AgentSession needs engine=... (native speech-to-speech) or at least "
                    "llm=... and tts=... (cascade; add stt=... unless the LLM takes audio)"
                )
            from ..engines.cascade import CascadeEngine

            self.engine = CascadeEngine(
                stt=stt, llm=llm, tts=tts, vad=vad, turn_detector=turn_detector,
                options=cascade_options,
            )  # fmt: skip
        self.options = options or SessionOptions()
        self.userdata = userdata
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
        self._out: Chan[ResponseAudio | _EndOfResponse] = Chan()
        self._virtual_end = 0.0
        self._responses: dict[str, _Response] = {}
        self._current: _Response | None = None
        self._user_items: dict[str, ChatMessage] = {}
        self._turn: _Turn | None = None
        self._user_speech_end: float | None = None
        self._tool_steps = 0
        # interruption policy: the overlap awaiting a verdict and the playback pause state
        self._barge: _BargeIn | None = None
        self._barge_timer: asyncio.TimerHandle | None = None
        self._committed_items: deque[str] = deque(maxlen=8)
        self._say_requests: deque[_SayRequest] = deque()
        self._send_gate = asyncio.Event()  # cleared while playback is paused
        self._send_gate.set()
        self._clock_paused_at: float | None = None  # the transport itself is paused
        self._clock_running = asyncio.Event()
        self._clock_running.set()
        self._transport_paused = False
        self._pause_seq = 0
        self.engine.on("metrics", self._on_metrics)

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
        self._transport = transport
        await transport.start()
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
        if self._closing or self._close_task is not None:
            return
        self._close_task = asyncio.create_task(self.aclose(reason), name="session-close")

    async def aclose(self, reason: str = "closed") -> None:
        """Stop the session and release the engine connection and transport."""
        if self._closing:
            await self._closed.wait()
            return
        self._closing = True
        self._cancel_barge_in_timer()
        current = asyncio.current_task()
        await cancel_and_wait(*[t for t in self._loops if t is not current])
        await self._tasks.cancel_all()
        self._out.close()
        if self._conn is not None:
            with contextlib.suppress(Exception):
                await self._conn.aclose()
        if self._transport is not None:
            with contextlib.suppress(Exception):
                await self._transport.aclose()
        if self._processors is not None:
            self._processors.close()
        self._set_agent_state(AgentState.CLOSED)
        if self._agent is not None:
            with contextlib.suppress(Exception):
                await self._agent.on_exit(self)
        self._closed.set()
        self.emit("close", SessionClosed(reason))

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
        # the engine gives no handle for the response it starts: the next one is ours
        request = _SayRequest(allow_interruptions)
        self._say_requests.append(request)
        try:
            await self.connection.say(text)
        except BaseException:
            with contextlib.suppress(ValueError):
                self._say_requests.remove(request)
            raise

    async def generate_reply(
        self, *, instructions: str | None = None, user_input: str | None = None
    ) -> None:
        """Make the agent respond now, optionally to a typed ``user_input``."""
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

    # ------------------------------------------------------------------- loops
    async def _input_loop(self) -> None:
        transport, conn = self.transport, self.connection
        try:
            async for frame in transport.audio_input():
                if self._processors is not None:
                    frame = self._processors.process_capture(frame)
                if self._discard_input():
                    # uninterruptible speech is playing: the engine must not hear the user
                    frame = AudioFrame(
                        bytes(len(frame.data)), frame.sample_rate, frame.channels, frame.timestamp
                    )
                await conn.send_audio(frame)
        except Exception as exc:
            logger.exception("audio input failed")
            self.emit("error", SessionError(exc, recoverable=False))
        if self.options.close_on_disconnect and not self._closing:
            self._schedule_close("user_disconnected")

    async def _event_loop(self) -> None:
        async for ev in self.connection.events():
            try:
                await self._handle(ev)
            except Exception as exc:
                logger.exception("error handling engine event %s", type(ev).__name__)
                self.emit("error", SessionError(exc, recoverable=True))
        if not self._closing:
            self._schedule_close("engine_closed")

    async def _playout_loop(self) -> None:
        transport = self.transport
        fmt = transport.output_format
        resampler = StreamResampler(fmt.sample_rate, fmt.channels)
        async for item in self._out:
            if isinstance(item, _EndOfResponse):
                resp = self._responses.get(item.response_id)
                if resp is not None and not resp.interrupted:
                    delay = max(0.0, self._virtual_end - now())
                    self._tasks.spawn(self._finish_after(resp, delay))
                continue
            resp = self._responses.get(item.response_id)
            if resp is None or resp.interrupted:
                continue
            frame = resampler.push(item.frame)
            if not frame:
                continue
            await self._wait_for_playout_slot()
            if resp.interrupted:
                continue
            t = now()
            start = max(t, self._virtual_end)
            self._virtual_end = start + frame.duration
            resp.segments.append((start, frame.duration))
            if resp.first_audio_at is None:
                resp.first_audio_at = t
                if resp.turn is not None and resp.turn.first_audio_at is None:
                    resp.turn.first_audio_at = t
                self._set_agent_state(AgentState.SPEAKING)
            if self._processors is not None:
                self._processors.process_render(frame)
            await transport.write_audio(frame)

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
            started = _Response(ev.response_id, ev.timestamp, turn)
            started.allow_interruptions = self._say_request_for(ev.timestamp)
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
            for task_owner in self._responses.values():
                for call, task in zip(task_owner.tool_calls, task_owner.tool_tasks, strict=False):
                    if call.call_id in ev.call_ids and not task.done():
                        task.cancel()
        elif isinstance(ev, EngineErrorEvent):
            self.emit("error", SessionError(ev.error, ev.recoverable))
            if not ev.recoverable:
                self._schedule_close("engine_error")

    def _say_request_for(self, started_at: float) -> bool | None:
        """``allow_interruptions`` of the ``say()`` that started a response at ``started_at``."""
        requests = self._say_requests
        while requests and started_at - requests[0].created > _SAY_REQUEST_TTL:
            requests.popleft()  # the engine never started a response for it
        if requests and started_at >= requests[0].created:
            return requests.popleft().allow_interruptions
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
        self.history.append(ev.call)
        self.emit("conversation_item", ConversationItemAdded(ev.call))
        self.emit("tool_call", ToolCalled(ev.call))
        task = self._tasks.spawn(self._run_tool(ev.call), name=f"tool-{ev.call.name}")
        if resp is not None:
            resp.tool_calls.append(ev.call)
            resp.tool_tasks.append(task)

    async def _run_tool(self, call: FunctionCall) -> tuple[FunctionCallOutput, float]:
        t0 = now()
        output = await execute_function_call(
            call,
            self.agent.tools,
            ctx=ToolContext(call=call, session=self, userdata=self.userdata),
            timeout=self.options.tool_timeout,
        )
        return output, now() - t0

    def _on_response_done(self, ev: ResponseDone) -> None:
        resp = self._responses.get(ev.response_id)
        if resp is None:
            return
        resp.done = True
        resp.status = ev.status
        if resp.tool_tasks:
            self._tasks.spawn(self._complete_tools(resp))
        if resp.interrupted:
            return
        if resp.first_audio_at is None and not resp.segments and self._out.empty():
            # nothing was (or will be) played for this response
            self._tasks.spawn(self._finish_after(resp, 0.0))
        else:
            self._out.send_nowait(_EndOfResponse(ev.response_id))

    async def _complete_tools(self, resp: _Response) -> None:
        results = await asyncio.gather(*resp.tool_tasks)
        for call, (output, duration) in zip(resp.tool_calls, results, strict=True):
            self.history.append(output)
            self.emit("conversation_item", ConversationItemAdded(output))
            self.emit("tool_result", ToolResult(call, output, duration))
        self._tool_steps += 1
        respond = (
            not resp.interrupted
            and resp.status == "completed"
            and self.user_state != UserState.SPEAKING
            and self._tool_steps <= self.options.max_tool_steps
        )
        if self._tool_steps > self.options.max_tool_steps:
            logger.warning(
                "max_tool_steps (%d) reached; not responding", self.options.max_tool_steps
            )
        if resp.turn is not None:
            resp.turn.tool_calls += len(results)
        for i, (output, _) in enumerate(results):
            await self.connection.send_tool_output(
                output, respond=respond and i == len(results) - 1
            )
        if not respond and not resp.interrupted:
            if resp.turn is not None:
                self._close_turn(resp.turn)
            if self._current is resp or self._current is None:
                self._current = None
                self._set_agent_state(AgentState.LISTENING)

    async def _finish_after(self, resp: _Response, delay: float) -> None:
        pauses = self._pause_seq
        if delay > 0:
            await asyncio.sleep(delay)
        if pauses != self._pause_seq or self._clock_paused_at is not None:
            await self._wait_playout_end(resp)  # paused meanwhile: its audio ends later
        if resp.finished or resp.interrupted:
            return
        resp.finished = True
        await self._release_barge_in(resp)
        if resp.turn is not None:
            resp.turn.agent_speech += resp.played(now())
        if resp.tool_calls:
            return  # the turn continues once the tool results are sent back
        if resp.turn is not None:
            self._close_turn(resp.turn)
        if self._current is resp and self.agent_state != AgentState.CLOSED:
            self._current = None
            self._set_agent_state(AgentState.LISTENING)

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
        return not resp.done or self._virtual_end > now() or not self._out.empty()

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
        barge = _BargeIn(Overlap.begin(policy, start), resp, frozenset(self._committed_items))
        self._barge = barge
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
        overlap = barge.overlap
        overlap.add_transcript(ev.item_id, ev.text)
        if (
            not overlap.confirmed
            and not overlap.speaking
            and not barge.input_cleared
            and overlap.transcript
            and len(overlap.meaningful_words()) < overlap.policy.words_needed
        ):
            # not a turn: keep the engine from committing it (and answering it, which
            # would cancel the paused speech)
            barge.input_cleared = True
            try:
                await self.connection.clear_input()
            except Exception as exc:
                logger.warning("engine clear_input failed: %s", exc)
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
        await self._interrupt(barge.response, cancel=False)
        self._watch_aftermath(barge)

    async def _evaluate_barge_in(self) -> None:
        barge = self._barge
        if barge is None:
            return
        verdict = barge.overlap.verdict(now())
        if verdict is None:
            self._schedule_barge_in_check(barge)
        elif verdict == Verdict.INTERRUPT:
            barge.overlap.confirmed = True
            await self._interrupt(barge.response)
            self._watch_aftermath(barge)
        elif verdict == Verdict.RESUME:
            await self._resume_after_false_interruption(barge)
        else:
            self._end_barge_in(barge)
            if verdict == Verdict.FALSE_INTERRUPTION:
                self._emit_false_interruption(barge, resumed=False, paused=0.0)
        await self._ensure_playback()

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
        if barge.paused_at is None:
            return  # the agent never stopped talking: nothing to resume or report
        paused = now() - barge.paused_at
        barge.paused_at = None
        await self._resume_playback()
        resp = barge.response
        if self._current is resp and not (resp.finished or resp.interrupted):
            speaking = resp.first_audio_at is not None
            self._set_agent_state(AgentState.SPEAKING if speaking else AgentState.THINKING)
        self._emit_false_interruption(barge, resumed=True, paused=paused)

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

    def _schedule_barge_in_check(self, barge: _BargeIn) -> None:
        self._cancel_barge_in_timer()
        deadline = barge.overlap.deadline()
        if deadline is None or self._closing:
            return
        self._barge_timer = asyncio.get_running_loop().call_later(
            max(0.0, deadline - now()), self._on_barge_in_timer, barge
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
        # else: the short look-ahead already handed to the transport plays out
        if self.agent_state in (AgentState.SPEAKING, AgentState.THINKING):
            self._set_agent_state(AgentState.LISTENING)

    async def _resume_playback(self) -> None:
        paused_at = self._clock_paused_at
        if paused_at is not None:
            self._shift_timeline(paused_at, now() - paused_at)
            self._unfreeze_clock()
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

    def _unfreeze_clock(self) -> None:
        self._clock_paused_at = None
        self._clock_running.set()

    def _shift_timeline(self, paused_at: float, delta: float) -> None:
        """Playback stood still from ``paused_at`` for ``delta`` s: unheard audio moves later."""
        if delta <= 0:
            return
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
        played = resp.played(paused_at if paused_at is not None else now())
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
        if resp.turn is not None:
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
