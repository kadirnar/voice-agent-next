"""Engine-agnostic session rotation (``engines/rotation.py``), tested with the mock engine."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any, TypeVar

import pytest

from voice_agent_next import Agent, AgentSession, AudioFrame, ChatContext, ChatMessage
from voice_agent_next.chat import FunctionCall, FunctionCallOutput
from voice_agent_next.engine import EngineConnection, EngineOptions
from voice_agent_next.engines.rotation import (
    AudioBuffer,
    AudioReplay,
    ConversationRecorder,
    QuietTracker,
    RotatingConnection,
    RotatingEngine,
    RotationPolicy,
    SummarizeHistory,
    TruncateHistory,
    backoff_delays,
)
from voice_agent_next.errors import ProviderConnectionError
from voice_agent_next.events import (
    EngineErrorEvent,
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
)
from voice_agent_next.metrics import RotationMetrics, UsageSummary
from voice_agent_next.providers.mock import (
    MockEngine,
    MockEngineConnection,
    MockLLM,
    MockToolCall,
    synth_speech,
)
from voice_agent_next.transports import LoopbackTransport
from voice_agent_next.utils import now

T = TypeVar("T")
FAST = RotationPolicy(quiet_period=0.1, backoff=0.01, replay=1.0)


async def wait_for(predicate: Callable[[], Any], timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), timeout)


class Collector:
    def __init__(self, conn: EngineConnection) -> None:
        self.events: list[Any] = []
        self.task = asyncio.create_task(self._run(conn))

    async def _run(self, conn: EngineConnection) -> None:
        async for ev in conn.events():
            self.events.append(ev)

    def of(self, cls: type[T]) -> list[T]:
        return [e for e in self.events if isinstance(e, cls)]

    def statuses(self) -> list[str]:
        return [e.status for e in self.of(EngineStatus)]


async def feed(conn: EngineConnection, frame: AudioFrame, chunk: float = 0.02) -> None:
    step = round(chunk * frame.sample_rate) * 2
    for i in range(0, len(frame.data), step):
        await conn.send_audio(AudioFrame(frame.data[i : i + step], frame.sample_rate, 1, now()))
    await asyncio.sleep(0)


async def user_turn(conn: EngineConnection, speech: float = 0.6, silence: float = 0.6) -> None:
    await feed(conn, AudioFrame.silence(0.2, 16_000))
    await feed(conn, synth_speech(speech, 16_000))
    await feed(conn, AudioFrame.silence(silence, 16_000))


class SlowMockEngine(MockEngine):
    """A mock engine whose connections take ``connect_delay`` to open (and can fail)."""

    def __init__(self, *, connect_delay: float = 0.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.connect_delay = connect_delay
        self.failures = 0
        self.options: list[EngineOptions] = []

    async def connect(self, options: EngineOptions) -> EngineConnection:
        self.options.append(options)
        if self.connect_delay:
            await asyncio.sleep(self.connect_delay)
        if self.failures:
            self.failures -= 1
            raise ProviderConnectionError("refused", provider="mock")
        return await super().connect(options)


async def open_rotating(
    inner: MockEngine, policy: RotationPolicy = FAST, options: EngineOptions | None = None
) -> tuple[RotatingEngine, RotatingConnection, Collector, list[RotationMetrics]]:
    engine = RotatingEngine(inner, policy=policy)
    metrics: list[RotationMetrics] = []
    engine.on("metrics", lambda m: metrics.append(m) if isinstance(m, RotationMetrics) else None)
    conn = await engine.connect(options or EngineOptions(instructions="Be brief."))
    assert isinstance(conn, RotatingConnection)
    return engine, conn, Collector(conn), metrics


@pytest.fixture
async def closing() -> AsyncIterator[list[EngineConnection]]:
    conns: list[EngineConnection] = []
    yield conns
    for conn in conns:
        await conn.aclose()


def texts(ctx: ChatContext) -> list[tuple[str, str]]:
    return [(m.role, m.text) for m in ctx.messages()]


# --------------------------------------------------------------------------- helpers
def test_policy_schedule() -> None:
    assert RotationPolicy().schedule(3600.0) == (3300.0, 3590.0)
    assert RotationPolicy().schedule(480.0) == (240.0, 470.0)  # Nova Sonic: at most half
    assert RotationPolicy().schedule(None) == (None, None)
    assert RotationPolicy(rotate_after=10.0).schedule(None) == (10.0, None)
    assert RotationPolicy(proactive=False).schedule(3600.0) == (None, None)
    assert list(backoff_delays(0.5, 3.0, 5)) == [0.5, 1.0, 2.0, 3.0, 3.0]


async def test_truncate_history_keeps_recent_items_and_tool_pairs() -> None:
    ctx = ChatContext()
    ctx.add_message("system", "notes")
    for i in range(10):
        ctx.add_message("user", f"question {i}")
        ctx.add_message("assistant", f"answer {i}")
    call = FunctionCall(name="lookup", arguments='{"q": 1}')
    ctx.append(call)
    ctx.append(FunctionCallOutput(call_id=call.call_id, output="found"))
    ctx.add_message("assistant", "")  # empty placeholders are never carried
    kept = await TruncateHistory(max_items=4)(ctx)
    assert [getattr(i, "role", i.type) for i in kept.items] == [
        "system", "user", "assistant", "function_call", "function_call_output",
    ]  # fmt: skip
    # the tool result is dropped when its call does not fit
    kept = TruncateHistory(max_items=1).select(ctx)
    assert [i.type for i in kept.items] == ["message"]
    kept = TruncateHistory(max_chars=30, include_tools=False).select(ctx)
    assert texts(kept) == [("system", "notes"), ("user", "question 9"), ("assistant", "answer 9")]


async def test_summarize_history_summarizes_older_turns_incrementally() -> None:
    llm = MockLLM(responses=["They talked about Paris.", "Paris, then Rome."])
    strategy = SummarizeHistory(llm, keep_last=2, min_items=4)
    ctx = ChatContext()
    for i in range(3):
        ctx.add_message("user", f"q{i}")
        ctx.add_message("assistant", f"a{i}")
    assert texts(await strategy(ctx))[0] == ("system", "Summary of the conversation so far: "
                                             "They talked about Paris.")  # fmt: skip
    assert texts(await strategy(ctx))[1:] == [("user", "q2"), ("assistant", "a2")]
    assert len(llm.requests) == 1  # cached
    ctx.add_message("user", "q3")
    out = await strategy(ctx)
    assert out.messages()[0].metadata == {"carry_over": "summary"}
    assert "Paris, then Rome." in out.messages()[0].text
    prompt = llm.requests[1].messages()[-1].text
    assert "Earlier summary: They talked about Paris." in prompt and "q0" not in prompt
    # a failing summarizer falls back to truncation
    broken = SummarizeHistory(MockLLM(responses=[""]), keep_last=2, min_items=1)
    assert texts(await broken(ctx)) == texts(TruncateHistory().select(ctx))


def test_recorder_keeps_what_was_heard() -> None:
    rec = ConversationRecorder()
    rec.observe(InputCommitted(item_id="u1"))
    rec.observe(ResponseStarted(response_id="r1"))
    rec.observe(ResponseText(response_id="r1", item_id="a1", delta="Hello there, "))
    rec.observe(ResponseText(response_id="r1", item_id="a1", delta="how are you?"))
    rec.observe(
        ResponseAudio(response_id="r1", item_id="a1", frame=AudioFrame.silence(2.0, 24_000))
    )
    rec.observe(InputTranscript(item_id="u1", text="hi", is_final=True))  # late transcript
    assert texts(rec.history) == [("user", "hi"), ("assistant", "Hello there, how are you?")]
    version = rec.version
    rec.truncate("a1", 1000)  # heard half of the audio
    msg = rec.history.messages()[1]
    assert msg.text == "Hello there," and msg.interrupted and rec.version > version
    rec.observe(ResponseDone(response_id="r1", status="cancelled"))
    call = FunctionCall(name="f")
    rec.observe(ResponseToolCall(response_id="r2", call=call))
    rec.observe(ResponseToolCall(response_id="r2", call=call))
    rec.add_tool_output(FunctionCallOutput(call_id=call.call_id, output="x"))
    rec.add_tool_output(FunctionCallOutput(call_id=call.call_id, output="x"))
    assert [i.type for i in rec.history.items][2:] == ["function_call", "function_call_output"]


def test_quiet_tracker() -> None:
    q = QuietTracker()
    q.last_activity -= 1
    assert q.is_quiet(0.5)
    q.observe(InputSpeechStarted())
    assert not q.is_quiet(0.0)
    q.observe(InputSpeechStopped())
    q.observe(ResponseStarted(response_id="r"))
    assert not q.is_quiet(0.0)
    q.observe(ResponseAudio(response_id="r", item_id="i", frame=AudioFrame.silence(0.3, 24_000)))
    q.observe(ResponseDone(response_id="r"))
    assert not q.is_quiet(0.0)  # still playing
    q.playout_end = now() - 1
    assert q.is_quiet(0.0) and not q.is_quiet(0.5)
    q.observe(ResponseToolCall(response_id="r2", call=FunctionCall(name="f", call_id="c")))
    assert not q.is_quiet(0.0)
    q.tool_output("c", respond=True)
    assert not q.is_quiet(0.0)  # the answer to the result has not started
    q.reset_server_state()
    assert q.is_quiet(0.0)


def test_audio_replay_and_buffer() -> None:
    replay = AudioReplay(0.5)
    for i in range(10):
        replay.record(i * 0.1, AudioFrame.silence(0.1, 16_000))
    assert [round(s, 1) for s, _ in replay.frames()] == [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    replay.mark_committed(0.75)
    assert [round(s, 1) for s, _ in replay.frames()] == [0.7, 0.8, 0.9]
    buf = AudioBuffer(0.25)
    for i in range(4):
        buf.push(i * 0.1, AudioFrame.silence(0.1, 16_000))
    assert buf.dropped == pytest.approx(0.2) and buf.total == pytest.approx(0.4)
    assert [round(s, 1) for s, _ in buf.drain()] == [0.2, 0.3] and buf.duration == 0.0


def test_usage_summary_counts_rotations() -> None:
    usage = UsageSummary()
    usage.add(RotationMetrics(provider="p", model="m", reason="r", planned=True, gap=0.25))
    usage.add(RotationMetrics(provider="p", model="m", reason="r", planned=False, lost_audio=1.0))
    assert (usage.engine_rotations, usage.engine_rotation_gap, usage.engine_lost_audio) == (
        2, 0.25, 1.0,
    )  # fmt: skip


# ------------------------------------------------------------------ RotatingEngine
async def test_forced_rotation_carries_the_conversation(closing: list[EngineConnection]) -> None:
    inner = SlowMockEngine(transcripts=["my name is Ada", "what is my name"],
                           responses=["Nice to meet you, Ada.", "You are Ada."])  # fmt: skip
    _, conn, rec, metrics = await open_rotating(inner)
    closing.append(conn)
    first = conn.inner
    assert isinstance(first, MockEngineConnection)
    await user_turn(conn)
    await wait_for(lambda: rec.of(ResponseDone))
    conn.rotate("test")
    await wait_for(lambda: rec.statuses()[-1:] == ["reconnected"])
    assert rec.statuses() == ["reconnecting", "reconnected"]
    second = conn.inner
    assert isinstance(second, MockEngineConnection) and second is not first and first.closed
    # make-before-break: the new connection was opened with the heard history
    seeded = inner.options[-1]
    assert seeded.instructions == "Be brief."
    assert texts(seeded.chat_ctx or ChatContext()) == [
        ("user", "my name is Ada"), ("assistant", "Nice to meet you, Ada."),
    ]  # fmt: skip
    (m,) = metrics
    assert m.planned and m.rotation == 1 and m.carried_items == 2 and m.lost_audio == 0
    assert m.failed_responses == 0 and m.gap < 1.0
    # the conversation continues on the new connection only: no double responses
    started = first.responses_started
    await user_turn(conn)
    await wait_for(lambda: len(rec.of(ResponseDone)) == 2)
    assert first.responses_started == started and second.responses_started == 1
    assert "".join(t.delta for t in rec.of(ResponseText)[1:]) == "You are Ada."
    stopped = rec.of(InputSpeechStopped)[1].audio_time
    assert stopped is not None and stopped == pytest.approx(1.4 + 0.2 + 0.6, abs=0.1)
    assert not rec.of(EngineErrorEvent)


async def test_rotation_waits_while_the_agent_speaks(closing: list[EngineConnection]) -> None:
    inner = MockEngine(responses=["A long answer that takes a while to speak out loud."],
                       realtime_factor=1.0, chars_per_second=40.0)  # fmt: skip
    _, conn, rec, _ = await open_rotating(inner)
    closing.append(conn)
    await conn.create_response()
    await wait_for(lambda: rec.of(ResponseAudio))
    conn.rotate("test")
    await asyncio.sleep(0.5)
    assert not rec.of(EngineStatus)  # still speaking: the switch waits
    await wait_for(lambda: rec.of(ResponseDone), timeout=10)
    done_at = now()
    await wait_for(lambda: rec.statuses()[-1:] == ["reconnected"], timeout=10)
    assert rec.of(ResponseDone)[0].status == "completed"  # never cut off
    assert rec.of(EngineStatus)[0].timestamp >= done_at + 0.05  # quiet period respected


async def test_rotation_waits_while_the_user_speaks(closing: list[EngineConnection]) -> None:
    _, conn, rec, _ = await open_rotating(MockEngine())
    closing.append(conn)
    await feed(conn, synth_speech(0.5, 16_000))
    await wait_for(lambda: rec.of(InputSpeechStarted))
    conn.rotate("test")
    await asyncio.sleep(0.4)
    assert not rec.of(EngineStatus)
    await feed(conn, AudioFrame.silence(0.6, 16_000))
    await wait_for(lambda: rec.statuses()[-1:] == ["reconnected"], timeout=10)
    # the reply to the user's turn started before the switch and was not duplicated
    assert len(rec.of(ResponseStarted)) == 1


async def test_deadline_forces_the_switch_and_buffers_audio(
    closing: list[EngineConnection],
) -> None:
    inner = SlowMockEngine(responses=["Speaking for quite a long time, sorry about that."],
                           realtime_factor=1.0)  # fmt: skip
    _, conn, rec, metrics = await open_rotating(inner)
    closing.append(conn)
    await conn.create_response()
    await wait_for(lambda: rec.of(ResponseAudio))
    inner.connect_delay = 0.3  # the next connection is slow to open
    conn.rotate("test", deadline=now())
    sent = 0.0
    while rec.statuses()[-1:] != ["reconnected"]:
        await feed(conn, AudioFrame.silence(0.02, 16_000))
        sent += 0.02
        await asyncio.sleep(0.01)
    second = conn.inner
    assert isinstance(second, MockEngineConnection)
    (m,) = metrics
    assert m.failed_responses == 1 and m.gap >= 0.25 and m.buffered_audio > 0.1
    assert m.lost_audio == 0
    # every frame sent during the switch reached the new connection (plus the replay)
    assert second.received_audio == pytest.approx(m.buffered_audio + m.replayed_audio, abs=0.03)
    failed = [d for d in rec.of(ResponseDone) if d.status == "failed"]
    assert len(failed) == 1 and failed[0].error == "session rotated"


async def test_rotation_before_the_session_limit(closing: list[EngineConnection]) -> None:
    policy = RotationPolicy(rotate_after=0.3, quiet_period=0.05)
    _, conn, rec, metrics = await open_rotating(MockEngine(), policy)
    closing.append(conn)
    await wait_for(lambda: len(metrics) >= 2, timeout=5)  # rotates again on the new one
    assert rec.statuses()[:2] == ["reconnecting", "reconnected"]
    assert metrics[0].reason == "max_session_duration" and conn.rotations >= 2


async def test_drop_reconnects_with_backoff_and_history(closing: list[EngineConnection]) -> None:
    inner = SlowMockEngine(transcripts=["remember blue"],
                           responses=["Blue it is.", "It was blue."])  # fmt: skip
    _, conn, rec, metrics = await open_rotating(inner)
    closing.append(conn)
    await user_turn(conn)
    await wait_for(lambda: rec.of(ResponseDone))
    inner.failures = 2  # the first two reconnect attempts are refused
    first = conn.inner
    assert first is not None
    await first.aclose()  # the connection drops
    await feed(conn, AudioFrame.silence(0.3, 16_000))  # buffered meanwhile
    await wait_for(lambda: rec.statuses()[-1:] == ["reconnected"])
    assert rec.statuses() == ["reconnecting", "reconnected"]
    (m,) = metrics
    assert not m.planned and m.attempts == 3 and m.lost_audio == 0
    assert m.buffered_audio == pytest.approx(0.3, abs=0.03)
    assert texts(inner.options[-1].chat_ctx or ChatContext()) == [
        ("user", "remember blue"), ("assistant", "Blue it is."),
    ]  # fmt: skip
    await conn.send_text("which color?")
    await wait_for(lambda: len(rec.of(ResponseDone)) == 2)
    assert rec.of(ResponseDone)[1].status == "completed"


async def test_reconnect_gives_up_and_reports_a_fatal_error(
    closing: list[EngineConnection],
) -> None:
    inner = SlowMockEngine()
    policy = RotationPolicy(backoff=0.01, max_reconnect_attempts=2)
    _, conn, rec, _ = await open_rotating(inner, policy)
    inner.failures = 5
    assert conn.inner is not None
    await conn.inner.aclose()
    await asyncio.wait_for(rec.task, 5)
    assert rec.statuses() == ["reconnecting"] and conn.closed
    error = rec.of(EngineErrorEvent)[-1]
    assert not error.recoverable and isinstance(error.error, ProviderConnectionError)


async def test_buffer_overflow_is_never_silent(closing: list[EngineConnection]) -> None:
    inner = SlowMockEngine()
    policy = RotationPolicy(backoff=0.2, max_buffered_audio=0.1)
    _, conn, rec, metrics = await open_rotating(inner, policy)
    closing.append(conn)
    assert conn.inner is not None
    await conn.inner.aclose()
    await wait_for(lambda: rec.statuses() == ["reconnecting"])
    await feed(conn, AudioFrame.silence(0.5, 16_000))
    await wait_for(lambda: metrics)
    assert metrics[0].lost_audio == pytest.approx(0.4, abs=0.03)
    errors = rec.of(EngineErrorEvent)
    assert len(errors) == 1 and errors[0].recoverable and "lost" in str(errors[0].error)


async def test_tools_in_flight_across_rotation_and_reconnect(
    closing: list[EngineConnection],
) -> None:
    inner = SlowMockEngine(responses=[MockToolCall("lookup", {"q": "x"}), "Found it."])
    _, conn, rec, metrics = await open_rotating(inner)
    closing.append(conn)
    await conn.send_text("look it up")
    await wait_for(lambda: rec.of(ResponseToolCall))
    call = rec.of(ResponseToolCall)[0].call
    conn.rotate("test")
    await asyncio.sleep(0.4)
    assert not metrics  # a tool is running: no planned switch
    # the connection drops while the tool runs: the call is carried to the new one
    assert conn.inner is not None
    await conn.inner.aclose()
    await wait_for(lambda: metrics)
    carried = inner.options[-1].chat_ctx or ChatContext()
    assert [i.type for i in carried.items] == ["message", "function_call"]
    await conn.send_tool_output(FunctionCallOutput(call_id=call.call_id, output="x=1"))
    second = conn.inner
    assert isinstance(second, MockEngineConnection)
    assert [o.output for o in second.tool_outputs] == ["x=1"]
    await wait_for(lambda: any(t.delta == "Found it." for t in rec.of(ResponseText)))
    # the pending planned rotation was satisfied by the reconnect
    await asyncio.sleep(0.3)
    assert len(metrics) == 1


async def test_control_calls_wait_for_a_switch(closing: list[EngineConnection]) -> None:
    inner = SlowMockEngine(responses=["ok"])
    _, conn, rec, _ = await open_rotating(inner)
    closing.append(conn)
    inner.connect_delay = 0.3
    conn.rotate("test", deadline=now())
    await wait_for(lambda: rec.statuses() == ["reconnecting"])
    await conn.send_text("during the switch")  # waits, then goes to the new connection
    second = conn.inner
    assert isinstance(second, MockEngineConnection) and rec.statuses()[-1] == "reconnected"
    assert second.chat_ctx.messages()[-1].text == "during the switch"
    assert inner.options[-1].chat_ctx is not None
    assert not inner.options[-1].chat_ctx.messages()  # not seeded twice


async def test_agent_session_survives_rotations() -> None:
    inner = MockEngine(transcripts=["I like tea", "what do I like"],
                       responses=["Tea is great.", "You like tea."])  # fmt: skip
    engine = RotatingEngine(inner, policy=FAST)
    session = AgentSession(engine)
    metrics: list[Any] = []
    session.on("metrics", metrics.append)
    transport = LoopbackTransport()
    await session.start(Agent("Be brief."), transport)
    try:
        await transport.play_user_audio(synth_speech(0.6, 16_000), realtime=False)
        await transport.play_user_audio(AudioFrame.silence(0.6, 16_000), realtime=False)
        await wait_for(lambda: len(session.history.messages()) >= 2)
        conn = session.connection
        assert isinstance(conn, RotatingConnection)
        conn.rotate("test")
        await wait_for(lambda: any(isinstance(m, RotationMetrics) for m in metrics), timeout=10)
        await transport.play_user_audio(synth_speech(0.6, 16_000), realtime=False)
        await transport.play_user_audio(AudioFrame.silence(0.6, 16_000), realtime=False)
        await wait_for(lambda: len(session.history.messages()) >= 4, timeout=10)
        await wait_for(lambda: session.history.messages()[-1].text == "You like tea.")
    finally:
        await session.aclose()
    assert [m.text for m in session.history.messages()] == [
        "I like tea", "Tea is great.", "what do I like", "You like tea.",
    ]  # fmt: skip
    assert session.usage.engine_rotations == 1
    assert isinstance(session.history.messages()[1], ChatMessage)
