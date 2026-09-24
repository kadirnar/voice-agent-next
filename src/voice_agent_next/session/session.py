"""The :class:`AgentSession` runtime: transport <-> engine orchestration.

Responsibilities (identical for native speech-to-speech and cascaded engines):

* stream user audio from the transport into the engine (through optional audio
  processors such as echo cancellation);
* play engine audio through the transport, paced in real time with a small
  look-ahead so the session always knows what the user has actually *heard*;
* barge-in: when the user starts speaking over the agent, stop playback, cancel the
  response and truncate the agent's turn to what was heard;
* execute tool calls and feed the results back to the engine;
* keep the conversation history and emit transcripts, state changes and metrics.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from ..engines.cascade import CascadeOptions

__all__ = ["AgentSession", "SessionOptions"]


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

    def played(self, t: float) -> float:
        return sum(min(max(t - start, 0.0), dur) for start, dur in self.segments)

    @property
    def sent(self) -> float:
        return sum(d for _, d in self.segments)


@dataclass(slots=True)
class _EndOfResponse:
    response_id: str


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
    async def say(self, text: str) -> None:
        """Speak ``text`` (verbatim when the engine supports it)."""
        await self.connection.say(text)

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
        await self._interrupt()

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
            ahead = self._virtual_end - now()
            if ahead > self.options.output_lookahead:
                await asyncio.sleep(ahead - self.options.output_lookahead)
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
            if self.options.allow_interruptions and self._is_responding():
                await self._interrupt()
        elif isinstance(ev, InputSpeechStopped):
            self._set_user_state(UserState.LISTENING)
            wall = None
            if ev.audio_time is not None:
                wall = self.connection.audio_time_to_wall(ev.audio_time)
            self._user_speech_end = wall if wall is not None else ev.timestamp
        elif isinstance(ev, InputTranscript):
            self._on_input_transcript(ev)
        elif isinstance(ev, InputCommitted):
            if self._turn is not None:
                self._close_turn(self._turn)
            self._turn = _Turn(new_id("turn_"), self._user_speech_end, ev.timestamp)
            self._user_speech_end = None
            self._tool_steps = 0
            if self.agent_state == AgentState.LISTENING:
                self._set_agent_state(AgentState.THINKING)
        elif isinstance(ev, ResponseStarted):
            turn = self._turn if self._turn is not None and not self._turn.closed else None
            started = _Response(ev.response_id, ev.timestamp, turn)
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
        if delay > 0:
            await asyncio.sleep(delay)
        if resp.finished or resp.interrupted:
            return
        resp.finished = True
        if resp.turn is not None:
            resp.turn.agent_speech += resp.played(now())
        if resp.tool_calls:
            return  # the turn continues once the tool results are sent back
        if resp.turn is not None:
            self._close_turn(resp.turn)
        if self._current is resp and self.agent_state != AgentState.CLOSED:
            self._current = None
            self._set_agent_state(AgentState.LISTENING)

    # ------------------------------------------------------------- interruption
    def _is_responding(self) -> bool:
        resp = self._current
        if resp is None or resp.interrupted or resp.finished:
            return False
        return not resp.done or self._virtual_end > now() or not self._out.empty()

    async def _interrupt(self) -> None:
        resp = self._current
        if resp is None or resp.interrupted or resp.finished:
            return
        resp.interrupted = True
        t = now()
        played = resp.played(t)
        self._out.clear()
        self._virtual_end = t
        with contextlib.suppress(Exception):
            await self.transport.clear_audio()
        heard: str | None = None
        try:
            heard = await self.connection.interrupt(resp.item_id, round(played * 1000))
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
