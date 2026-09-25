"""Failure-safe lifecycle of AgentSession: loop failures, start/close errors, tool budget."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from tests.test_session import Recorder, speak, wait_for
from voice_agent_next import (
    Agent,
    AgentSession,
    AgentState,
    AudioFrame,
    SessionOptions,
    function_tool,
)
from voice_agent_next.audio.processing import AudioProcessor
from voice_agent_next.engine import EngineConnection, EngineOptions
from voice_agent_next.events import EngineEvent
from voice_agent_next.providers.mock import MockEngine, MockEngineConnection, MockToolCall
from voice_agent_next.session.events import SessionError
from voice_agent_next.transports import LoopbackTransport

TIMEOUT = 5.0


class BrokenEventsConnection(MockEngineConnection):
    """The engine's event stream fails (e.g. the provider's socket dies mid-read)."""

    def events(self) -> AsyncIterator[EngineEvent]:
        async def broken() -> AsyncIterator[EngineEvent]:
            await asyncio.sleep(0.01)
            raise RuntimeError("event stream broke")
            yield  # pragma: no cover

        return broken()


class SlowCloseConnection(MockEngineConnection):
    """Closing the connection takes a while (a provider flushing its socket)."""

    close_delay = 0.3

    async def aclose(self) -> None:
        await asyncio.sleep(self.close_delay)
        await super().aclose()


class ScriptedEngine(MockEngine):
    def __init__(
        self, conn_cls: type[MockEngineConnection] = MockEngineConnection, **kw: Any
    ) -> None:
        super().__init__(**kw)
        self.conn_cls = conn_cls
        self.fail_connect = False

    async def connect(self, options: EngineOptions) -> EngineConnection:
        if self.fail_connect:
            raise ConnectionError("engine unreachable")
        conn = self.conn_cls(self, options)
        self.connections.append(conn)
        return conn


class TrackingTransport(LoopbackTransport):
    def __init__(self, *, fail_write: bool = False, fail_start: bool = False) -> None:
        super().__init__()
        self.fail_write = fail_write
        self.fail_start = fail_start
        self.closed_calls = 0

    async def start(self) -> None:
        if self.fail_start:
            raise OSError("audio device busy")
        await super().start()

    async def write_audio(self, frame: AudioFrame) -> None:
        if self.fail_write:
            raise BrokenPipeError("client went away")
        await super().write_audio(frame)

    async def aclose(self) -> None:
        self.closed_calls += 1
        await super().aclose()


def closes(session: AgentSession) -> list[str]:
    reasons: list[str] = []
    session.on("close", lambda ev: reasons.append(ev.reason))
    return reasons


def fatal_errors(rec: Recorder) -> list[SessionError]:
    return [e for e in rec.of("error") if not e.recoverable]


# -------------------------------------------------------------------- loop failures


async def test_a_failing_engine_event_stream_closes_the_session() -> None:
    session = AgentSession(ScriptedEngine(BrokenEventsConnection))
    rec = Recorder(session)
    reasons = closes(session)
    transport = TrackingTransport()
    await asyncio.wait_for(session.run(Agent("x"), transport), TIMEOUT)  # returns, no hang
    assert session.closed and session.agent_state == AgentState.CLOSED
    [err] = fatal_errors(rec)
    assert isinstance(err.error, RuntimeError) and "event stream broke" in str(err.error)
    assert reasons == ["engine_error"]
    assert transport.closed_calls == 1
    assert all(t.done() for t in session._loops)


async def test_a_failing_transport_write_closes_the_session() -> None:
    session = AgentSession(ScriptedEngine())
    rec = Recorder(session)
    reasons = closes(session)
    transport = TrackingTransport(fail_write=True)
    agent = Agent("x", greeting="Hello there, how can I help you today?")
    await asyncio.wait_for(session.run(agent, transport), TIMEOUT)
    assert session.closed
    [err] = fatal_errors(rec)
    assert isinstance(err.error, BrokenPipeError)
    assert reasons == ["transport_error"]
    assert transport.closed_calls == 1


class BrokenInputTransport(TrackingTransport):
    def audio_input(self) -> AsyncIterator[AudioFrame]:
        async def broken() -> AsyncIterator[AudioFrame]:
            await asyncio.sleep(0.01)
            raise OSError("microphone unplugged")
            yield  # pragma: no cover

        return broken()


async def test_a_failing_audio_input_closes_the_session_even_without_close_on_disconnect() -> None:
    session = AgentSession(ScriptedEngine(), options=SessionOptions(close_on_disconnect=False))
    rec = Recorder(session)
    reasons = closes(session)
    await asyncio.wait_for(session.run(Agent("x"), BrokenInputTransport()), TIMEOUT)
    [err] = fatal_errors(rec)
    assert isinstance(err.error, OSError)
    assert reasons == ["transport_error"]


# -------------------------------------------------------------------------- start()


async def test_a_failed_connect_cleans_up_and_reraises(tmp_path: Path) -> None:
    engine = ScriptedEngine()
    engine.fail_connect = True
    session = AgentSession(engine, record=tmp_path / "call.wav")
    transport = TrackingTransport()
    with pytest.raises(ConnectionError, match="engine unreachable"):
        await asyncio.wait_for(session.start(Agent("x"), transport), TIMEOUT)
    assert session.closed and session.agent_state == AgentState.CLOSED
    assert transport.closed_calls == 1
    recorder = session.recorder
    assert recorder is not None and recorder._closed  # the recording file is finalized
    assert recorder._timeline is not None and recorder._timeline.closed
    # run() re-raises too, and waiting on the session does not hang
    await asyncio.wait_for(session.wait_closed(), 1.0)


async def test_a_failed_transport_start_cleans_up_and_reraises() -> None:
    session = AgentSession(ScriptedEngine())
    transport = TrackingTransport(fail_start=True)
    with pytest.raises(OSError, match="audio device busy"):
        await asyncio.wait_for(session.run(Agent("x"), transport), TIMEOUT)
    assert session.closed and transport.closed_calls == 1


class FailingEnterAgent(Agent):
    async def on_enter(self, session: AgentSession) -> None:
        raise ValueError("on_enter failed")


async def test_a_failing_on_enter_stops_the_loops_and_reraises() -> None:
    engine = ScriptedEngine()
    session = AgentSession(engine)
    transport = TrackingTransport()
    with pytest.raises(ValueError, match="on_enter failed"):
        await asyncio.wait_for(session.start(FailingEnterAgent("x"), transport), TIMEOUT)
    assert session.closed and transport.closed_calls == 1
    assert session._loops and all(t.done() for t in session._loops)
    assert engine.connections[0]._closed


# -------------------------------------------------------------------------- aclose()


async def test_aclose_is_cancel_safe_and_a_second_aclose_does_not_hang() -> None:
    engine = ScriptedEngine(SlowCloseConnection)
    session = AgentSession(engine)
    transport = TrackingTransport()
    await session.start(Agent("x"), transport)
    first = asyncio.create_task(session.aclose())
    await asyncio.sleep(0.05)  # inside the connection's slow aclose
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    await asyncio.wait_for(session.aclose(), TIMEOUT)
    assert session.closed
    assert engine.connections[0]._closed  # the cleanup still ran to the end
    assert transport.closed_calls == 1


async def test_concurrent_aclose_calls_close_once() -> None:
    session = AgentSession(ScriptedEngine(SlowCloseConnection))
    reasons = closes(session)
    transport = TrackingTransport()
    await session.start(Agent("x"), transport)
    await asyncio.wait_for(asyncio.gather(session.aclose("a"), session.aclose("b")), TIMEOUT)
    assert reasons == ["a"] and transport.closed_calls == 1


class ExplodingProcessor(AudioProcessor):
    def process_capture(self, frame: AudioFrame) -> AudioFrame:
        return frame

    def close(self) -> None:
        raise RuntimeError("native handle already freed")


async def test_aclose_survives_a_failing_cleanup_step() -> None:
    session = AgentSession(ScriptedEngine(), processors=[ExplodingProcessor()])
    reasons = closes(session)
    await session.start(Agent("x"), TrackingTransport())
    await asyncio.wait_for(session.aclose(), TIMEOUT)
    assert session.closed and reasons == ["closed"]
    await asyncio.wait_for(session.aclose(), 1.0)  # idempotent


class ClosingOnExitAgent(Agent):
    async def on_exit(self, session: AgentSession) -> None:
        await session.aclose("again")  # must not deadlock


async def test_aclose_from_on_exit_does_not_deadlock() -> None:
    session = AgentSession(ScriptedEngine())
    reasons = closes(session)
    await session.start(ClosingOnExitAgent("x"), TrackingTransport())
    await asyncio.wait_for(session.aclose(), TIMEOUT)
    assert reasons == ["closed"]


async def test_a_tool_can_end_the_call() -> None:
    finished: list[bool] = []

    @function_tool
    async def hang_up() -> str:
        """End the call."""
        await session.aclose("tool")
        finished.append(True)
        return "bye"

    session = AgentSession(ScriptedEngine(responses=[MockToolCall("hang_up")]))
    reasons = closes(session)
    transport = LoopbackTransport()
    await session.start(Agent("x", tools=[hang_up]), transport)
    await session.generate_reply(user_input="bye")
    await asyncio.wait_for(session.wait_closed(), TIMEOUT)
    await wait_for(lambda: bool(finished))  # the tool itself is not cancelled by the close
    assert reasons == ["tool"]


# ------------------------------------------------------------------- tool-step budget


async def test_max_tool_steps_is_per_request_chain_for_typed_input() -> None:
    @function_tool
    async def lookup() -> str:
        """Look something up."""
        return "found it"

    script: list[Any] = []
    for i in range(4):
        script += [MockToolCall("lookup"), f"Answer {i}."]
    session = AgentSession(ScriptedEngine(responses=script))
    session.options.max_tool_steps = 2
    rec = Recorder(session)
    await session.start(Agent("x", tools=[lookup]), LoopbackTransport())
    for i in range(4):  # 4 tool rounds in total > max_tool_steps, one per request
        await session.generate_reply(user_input=f"question {i}")
        await wait_for(lambda i=i: any(f"Answer {i}." in e.delta for e in rec.of("agent_transcript")))
        await wait_for(lambda: session.agent_state == AgentState.LISTENING)
    await session.aclose()
    assert len(rec.of("tool_result")) == 4


async def test_max_tool_steps_still_stops_a_typed_tool_loop() -> None:
    @function_tool
    async def again() -> str:
        """Loop forever."""
        return "call me again"

    session = AgentSession(ScriptedEngine(responses=[MockToolCall("again")] * 10))
    session.options.max_tool_steps = 2
    rec = Recorder(session)
    await session.start(Agent("x", tools=[again]), LoopbackTransport())
    await session.generate_reply(user_input="go")
    await wait_for(lambda: len(rec.of("tool_result")) == 3)
    await wait_for(lambda: session.agent_state == AgentState.LISTENING)
    await asyncio.sleep(0.1)
    await session.aclose()
    assert len(rec.of("tool_call")) == 3  # 2 allowed rounds + the one that hit the limit


# -------------------------------------------------------------------- turn metrics


async def test_tool_calls_count_in_a_turn_closed_before_the_tool_finished() -> None:
    gate = asyncio.Event()

    @function_tool
    async def slow() -> str:
        """A slow lookup."""
        await gate.wait()
        return "done"

    session = AgentSession(
        ScriptedEngine(transcripts=["first", "second"], responses=[MockToolCall("slow"), "Ok."]),
        options=SessionOptions(tool_filler_delay=None),
    )
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x", tools=[slow]), transport)
    await speak(transport)
    await wait_for(lambda: bool(rec.of("tool_call")))
    await speak(transport)  # the user moves on while the tool runs: the first turn closes
    await wait_for(lambda: len(rec.turn_metrics()) >= 1)
    gate.set()
    await wait_for(lambda: bool(rec.of("tool_result")))
    await asyncio.sleep(0.1)
    await session.aclose()
    first = rec.turn_metrics()[0]
    assert first.tool_calls == 1
