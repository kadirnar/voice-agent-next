"""Tool watchdog fillers, non-blocking tools, progress updates and delegation (issue #28).

Everything runs offline against the scripted mock engine (a native, blocking-tools
engine), the mock cascade and the fake Gemini Live server (native non-blocking tools).
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable
from typing import Any

import pytest

from voice_agent_next import (
    Agent,
    AgentSession,
    AgentState,
    AudioFrame,
    CascadeOptions,
    ChatMessage,
    SessionOptions,
    ToolContext,
    function_tool,
)
from voice_agent_next.chat import FunctionCall, FunctionCallOutput
from voice_agent_next.events import ToolCallCancelled
from voice_agent_next.metrics import TurnMetrics
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import (
    MockEngine,
    MockEngineConnection,
    MockLLM,
    MockSTT,
    MockToolCall,
    MockTTS,
    synth_speech,
)
from voice_agent_next.session import DEFAULT_TOOL_FILLERS
from voice_agent_next.tools import DEFAULT_TOOL_ACK
from voice_agent_next.transports import LoopbackTransport

EVENTS = ("agent_transcript", "tool_call", "tool_result", "tool_filler", "tool_progress",
          "tool_cancelled", "interrupted", "metrics", "agent_state_changed", "error",
          "conversation_item")  # fmt: skip


class Recorder:
    def __init__(self, session: AgentSession) -> None:
        self.events: list[tuple[str, Any]] = []
        for name in EVENTS:
            session.on(name, self._make(name))

    def _make(self, name: str) -> Callable[[Any], None]:
        return lambda ev: self.events.append((name, ev))

    def of(self, name: str) -> list[Any]:
        return [ev for n, ev in self.events if n == name]

    def said(self) -> list[str]:
        """What the agent said, one entry per response."""
        texts: dict[str, str] = {}
        for e in self.of("agent_transcript"):
            texts[e.response_id] = texts.get(e.response_id, "") + e.delta
        return [t.strip() for t in texts.values()]

    def turn_metrics(self) -> list[TurnMetrics]:
        return [m for m in self.of("metrics") if isinstance(m, TurnMetrics)]

    def states(self) -> list[AgentState]:
        return [ev.new_state for ev in self.of("agent_state_changed")]


async def speak(transport: LoopbackTransport) -> None:
    await transport.play_user_audio(synth_speech(0.8, 16_000), realtime=False)
    await transport.play_user_audio(AudioFrame.silence(0.6, 16_000), realtime=False)


async def wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


ENGINES = ["native", "cascade"]
CPS = 45.0
"""Mock speech rate (characters/s): 3x the default, to keep the tests short."""


def make_session(
    kind: str, responses: list[Any], *, realtime_factor: float = 0.0, **options: Any
) -> AgentSession:
    options.setdefault("tool_filler_delay", 0.2)
    opts = SessionOptions(**options)
    if kind == "native":
        engine = MockEngine(transcripts=["go"], responses=responses, chars_per_second=CPS,
                            realtime_factor=realtime_factor)  # fmt: skip
        return AgentSession(engine, options=opts)
    return AgentSession(
        stt=MockSTT(transcripts=["go"]),
        llm=MockLLM(responses=responses),
        tts=MockTTS(realtime_factor=realtime_factor, chars_per_second=CPS),
        vad=EnergyVAD(),
        cascade_options=CascadeOptions(min_endpointing_delay=0.0),
        options=opts,
    )


def engine_conn(session: AgentSession) -> Any:
    """The mock engine's connection (native) or the cascade connection."""
    return session.connection


def history_text(session: AgentSession) -> list[tuple[str, str]]:
    return [(i.role, i.text) for i in session.history.items if isinstance(i, ChatMessage)]


# ----------------------------------------------------------------------- tool options
def test_function_tool_options() -> None:
    @function_tool(filler=["Hold on."], blocking=False, scheduling="silent", ack="Queued.")
    async def book(city: str) -> str:
        """Book a trip."""
        return city

    assert book.filler == ["Hold on."] and not book.blocking
    assert book.scheduling == "silent" and book.ack == "Queued."
    assert book.schema()["name"] == "book"

    @function_tool
    def plain() -> str:
        """Plain."""
        return ""

    assert plain.blocking and plain.filler is None and plain.scheduling == "when_idle"
    with pytest.raises(ValueError):
        function_tool(scheduling="later")(plain.fn)  # type: ignore[call-overload]
    with pytest.raises(ValueError):
        SessionOptions(tool_filler_delay=-1)
    with pytest.raises(TypeError):
        SessionOptions(tool_fillers="One moment")


async def test_report_progress_without_a_session_is_a_noop() -> None:
    ctx = ToolContext(call=FunctionCall(name="x", arguments="{}"))
    assert await ctx.report_progress("halfway") is False


# ---------------------------------------------------------------------------- fillers
@pytest.mark.parametrize("kind", ENGINES)
async def test_slow_tool_gets_one_filler(kind: str) -> None:
    @function_tool
    async def lookup(q: str) -> str:
        """Slow search."""
        await asyncio.sleep(0.8)
        return f"found {q}"

    session = make_session(kind, [MockToolCall("lookup", {"q": "x"}), "Here it is."])
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x", tools=[lookup]), transport)
    await speak(transport)
    await wait_for(lambda: len(rec.turn_metrics()) == 1)
    await session.aclose()

    (filler,) = rec.of("tool_filler")  # 0.8 s tool, 0.2 s delay: still only one filler
    assert filler.text in DEFAULT_TOOL_FILLERS
    assert [c.name for c in filler.calls] == ["lookup"]
    assert 0.15 <= filler.waited < 0.7
    assert rec.said() == [filler.text, "Here it is."]
    assert rec.of("tool_result")[0].output.output == "found x"
    # the filler is spoken before the answer and is not part of the tool round's history
    kinds = [getattr(i, "role", i.type) for i in session.history.items]
    assert kinds == ["user", "function_call", "assistant", "function_call_output", "assistant"]
    m = rec.turn_metrics()[0]
    assert m.tool_calls == 1 and not m.interrupted
    # the agent stays busy (never LISTENING) from the user's turn to the answer
    states = rec.states()
    thinking = states.index(AgentState.THINKING)
    last_speaking = len(states) - 1 - states[::-1].index(AgentState.SPEAKING)
    assert AgentState.LISTENING not in states[thinking:last_speaking]


@pytest.mark.parametrize("kind", ENGINES)
async def test_fast_tool_gets_no_filler(kind: str) -> None:
    @function_tool
    async def lookup() -> str:
        """Fast."""
        return "ok"

    session = make_session(kind, [MockToolCall("lookup"), "Done."], tool_filler_delay=0.3)
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x", tools=[lookup]), transport)
    await speak(transport)
    await wait_for(lambda: len(rec.turn_metrics()) == 1)
    await asyncio.sleep(0.4)  # past the filler delay
    await session.aclose()
    assert rec.of("tool_filler") == []
    assert rec.said() == ["Done."]


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("Checking the calendar.", {"Checking the calendar."}),
        (["A.", "B."], {"A.", "B."}),
        (lambda call: f"Looking up {call.name}.", {"Looking up lookup."}),
    ],
)
async def test_per_tool_filler(spec: Any, expected: set[str]) -> None:
    @function_tool(filler=spec)
    async def lookup() -> str:
        """Slow."""
        await asyncio.sleep(0.5)
        return "ok"

    session = make_session("native", [MockToolCall("lookup"), "Done."])
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x", tools=[lookup]), transport)
    await speak(transport)
    await wait_for(lambda: len(rec.turn_metrics()) == 1)
    await session.aclose()
    (filler,) = rec.of("tool_filler")
    assert filler.text in expected


async def test_filler_disabled_per_tool_and_per_session() -> None:
    @function_tool(filler=False)
    async def quiet() -> str:
        """Slow."""
        await asyncio.sleep(0.5)
        return "ok"

    @function_tool
    async def loud() -> str:
        """Slow."""
        await asyncio.sleep(0.5)
        return "ok"

    for tool, delay in ((quiet, 0.2), (loud, None)):
        session = make_session("native", [MockToolCall(tool.name), "Done."],
                               tool_filler_delay=delay)  # fmt: skip
        rec = Recorder(session)
        transport = LoopbackTransport()
        await session.start(Agent("x", tools=[tool]), transport)
        await speak(transport)
        await wait_for(lambda rec=rec: len(rec.turn_metrics()) == 1)  # type: ignore[misc]
        await session.aclose()
        assert rec.of("tool_filler") == [] and rec.said() == ["Done."]


async def test_fillers_do_not_repeat_across_rounds() -> None:
    @function_tool
    async def step() -> str:
        """Slow step."""
        await asyncio.sleep(0.8)  # > delay + the previous filler's audio
        return "next"

    fillers = ["One.", "Two.", "Three."]
    session = make_session("native", [MockToolCall("step")] * 3 + ["Done."], tool_fillers=fillers)
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x", tools=[step]), transport)
    await speak(transport)
    await wait_for(lambda: len(rec.turn_metrics()) == 1, timeout=10)
    await session.aclose()
    said = [f.text for f in rec.of("tool_filler")]
    assert sorted(said) == sorted(fillers)  # one per round, all different


async def test_uninterruptible_filler() -> None:
    @function_tool
    async def lookup() -> str:
        """Slow."""
        await asyncio.sleep(0.5)
        return "ok"

    session = make_session("native", [MockToolCall("lookup"), "Done."],
                           tool_filler_interruptible=False)  # fmt: skip
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x", tools=[lookup]), transport)
    await speak(transport)
    await wait_for(lambda: len(rec.turn_metrics()) == 1)
    asides = [r for r in session._responses.values() if r.aside]
    await session.aclose()
    assert len(asides) == 1 and asides[0].allow_interruptions is False


async def test_no_filler_when_the_engine_keeps_talking() -> None:
    """Engines with non-blocking tools (the model is not waiting) get no filler."""

    @function_tool
    async def lookup() -> str:
        """Slow."""
        await asyncio.sleep(0.5)
        return "ok"

    session = make_session("native", [MockToolCall("lookup"), "Done."])
    engine = session.engine
    engine.capabilities = dataclasses.replace(engine.capabilities, tool_mode="non_blocking")
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x", tools=[lookup]), transport)
    await speak(transport)
    await wait_for(lambda: len(rec.turn_metrics()) == 1)
    await session.aclose()
    assert rec.of("tool_filler") == []


# ------------------------------------------------------------------- progress updates
@pytest.mark.parametrize("kind", ENGINES)
async def test_progress_is_spoken_and_replaces_the_filler(kind: str) -> None:
    @function_tool
    async def search(ctx: ToolContext) -> str:
        """Search flights."""
        await asyncio.sleep(0.05)
        assert await ctx.report_progress("Found three flights, comparing prices.")
        await ctx.report_progress("Cheapest is 90 euros.", speak=False, to_model=True)
        await asyncio.sleep(0.6)
        return "flight A"

    session = make_session(kind, [MockToolCall("search"), "Flight A it is."])
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x", tools=[search]), transport)
    await speak(transport)
    await wait_for(lambda: len(rec.turn_metrics()) == 1)
    await session.aclose()

    progress = rec.of("tool_progress")
    assert [(p.message, p.spoken) for p in progress] == [
        ("Found three flights, comparing prices.", True),
        ("Cheapest is 90 euros.", False),
    ]
    assert rec.of("tool_filler") == []  # the user already heard progress
    assert rec.said() == ["Found three flights, comparing prices.", "Flight A it is."]
    conn = engine_conn(session)
    injected = [i.text for i in conn.chat_ctx.items if isinstance(i, ChatMessage)]
    assert any("Cheapest is 90 euros." in t for t in injected)


# ------------------------------------------------------------------ non-blocking tools
@pytest.mark.parametrize("kind", ENGINES)
async def test_non_blocking_tool_result_is_delivered_when_idle(kind: str) -> None:
    release = asyncio.Event()

    @function_tool(blocking=False)
    async def book(city: str) -> str:
        """Book a hotel."""
        await release.wait()
        return f"booked in {city}"

    session = make_session(kind, [MockToolCall("book", {"city": "Rome"}),
                                  "I am booking it, anything else?", "Your hotel is booked."])  # fmt: skip
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x", tools=[book]), transport)
    await speak(transport)
    # the model got an immediate acknowledgement and the conversation went on
    await wait_for(lambda: rec.said() == ["I am booking it, anything else?"])
    await wait_for(lambda: session.agent_state == AgentState.LISTENING)
    assert rec.of("tool_result") == [] and rec.of("tool_filler") == []
    outputs = [i for i in session.history.items if isinstance(i, FunctionCallOutput)]
    assert [o.output for o in outputs] == [DEFAULT_TOOL_ACK]
    release.set()
    await wait_for(lambda: len(rec.said()) == 2)
    await wait_for(lambda: session.agent_state == AgentState.LISTENING)
    await session.aclose()

    assert rec.said()[-1] == "Your hotel is booked."
    (result,) = rec.of("tool_result")
    assert result.output.output == "booked in Rome" and result.blocking is False
    injected = [i for i in session.history.items
                if isinstance(i, ChatMessage) and i.metadata.get("background_result")]  # fmt: skip
    assert len(injected) == 1 and "booked in Rome" in injected[0].text
    assert injected[0].metadata["tool_call_id"] == result.call.call_id
    assert rec.turn_metrics()[0].tool_calls == 1


async def test_non_blocking_result_waits_while_the_agent_speaks() -> None:
    @function_tool(blocking=False)
    async def book() -> str:
        """Book."""
        await asyncio.sleep(0.2)
        return "booked"

    long_reply = "Sure, I started the booking, meanwhile let me tell you about the city."
    session = make_session("native", [MockToolCall("book"), long_reply, "It is booked."])
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await session.start(Agent("x", tools=[book]), transport)
    await speak(transport)
    await wait_for(lambda: bool(rec.of("tool_result")))  # arrived while speaking
    assert session.agent_state == AgentState.SPEAKING
    await wait_for(lambda: len(rec.said()) == 2, timeout=10)
    await session.aclose()
    assert rec.of("interrupted") == []  # when_idle: the long reply was not cut off
    assert rec.said() == [long_reply, "It is booked."]


async def test_non_blocking_interrupt_scheduling_cuts_the_agent_off() -> None:
    @function_tool(blocking=False, scheduling="interrupt")
    async def alarm() -> str:
        """Urgent."""
        await asyncio.sleep(0.3)
        return "fire alarm"

    long_reply = "Let me tell you a long story about the history of this lovely old city."
    session = make_session("native", [MockToolCall("alarm"), long_reply, "Alarm!"])
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await session.start(Agent("x", tools=[alarm]), transport)
    await speak(transport)
    await wait_for(lambda: rec.said()[-1:] == ["Alarm!"], timeout=10)
    await session.aclose()
    assert len(rec.of("interrupted")) == 1
    said = [i for i in session.history.items
            if isinstance(i, ChatMessage) and i.role == "assistant"]  # fmt: skip
    assert said[0].interrupted and len(said[0].text) < len(long_reply)


async def test_non_blocking_silent_scheduling_adds_context_only() -> None:
    @function_tool(blocking=False, scheduling="silent")
    async def note() -> str:
        """Note."""
        await asyncio.sleep(0.1)
        return "noted 42"

    session = make_session("native", [MockToolCall("note"), "Okay.", "Unexpected."])
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x", tools=[note]), transport)
    await speak(transport)
    await wait_for(lambda: bool(rec.of("tool_result")))
    conn: MockEngineConnection = engine_conn(session)
    await wait_for(lambda: any("noted 42" in getattr(i, "text", "") for i in conn.chat_ctx.items))
    started = conn.responses_started
    await asyncio.sleep(0.3)
    await session.aclose()
    assert rec.said() == ["Okay."] and conn.responses_started == started


async def test_native_non_blocking_tools_use_engine_scheduling(monkeypatch: Any) -> None:
    @function_tool(blocking=False, scheduling="interrupt")
    async def fetch() -> str:
        """Fetch."""
        await asyncio.sleep(0.2)
        return "fetched"

    session = make_session("native", [MockToolCall("fetch"), "Fetching now.", "Fetched."])
    engine = session.engine
    engine.capabilities = dataclasses.replace(engine.capabilities, tool_mode="non_blocking")
    sent: list[tuple[str, str]] = []
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x", tools=[fetch]), transport)
    conn = session.connection
    original = conn.send_async_tool_output

    async def spy(output: FunctionCallOutput, *, scheduling: Any = "when_idle") -> None:
        sent.append((output.output, scheduling))
        await original(output, scheduling=scheduling)

    monkeypatch.setattr(conn, "send_async_tool_output", spy)
    await speak(transport)
    await wait_for(lambda: bool(sent))
    await wait_for(lambda: len(rec.turn_metrics()) >= 1)
    await session.aclose()
    assert sent == [("fetched", "interrupt")]
    assert conn.tool_outputs[0].output == "fetched"  # type: ignore[attr-defined]
    # no acknowledgement: the engine does not wait for an output
    assert DEFAULT_TOOL_ACK not in [o.output for o in conn.tool_outputs]  # type: ignore[attr-defined]
    assert rec.of("tool_result")[0].blocking is False


@pytest.mark.parametrize(
    ("body", "expected"),
    [("raise", "it failed: Tool book failed: RuntimeError: no rooms"), ("hang", "timed out")],
)
async def test_non_blocking_errors_and_timeouts_are_reported(body: str, expected: str) -> None:
    @function_tool(blocking=False, timeout=0.2)
    async def book() -> str:
        """Book."""
        if body == "raise":
            raise RuntimeError("no rooms")
        await asyncio.sleep(10)
        return "never"

    session = make_session("native", [MockToolCall("book"), "On it.", "Sorry."])
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x", tools=[book]), transport)
    await speak(transport)
    await wait_for(lambda: rec.said() == ["On it.", "Sorry."])
    await session.aclose()
    (result,) = rec.of("tool_result")
    assert result.output.is_error
    injected = [i.text for i in session.history.items
                if isinstance(i, ChatMessage) and i.metadata.get("background_result")]  # fmt: skip
    assert len(injected) == 1 and expected in injected[0]


# ---------------------------------------------------------------------- cancellation
@pytest.mark.parametrize("kind", ENGINES)
async def test_engine_cancels_a_blocking_round(kind: str) -> None:
    cancelled = asyncio.Event()

    @function_tool
    async def slow() -> str:
        """Slow."""
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "never"

    session = make_session(kind, [MockToolCall("slow"), "Never said."], tool_filler_delay=None)
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x", tools=[slow]), transport)
    await speak(transport)
    await wait_for(lambda: bool(rec.of("tool_call")))
    call = rec.of("tool_call")[0].call
    session.connection._emit(ToolCallCancelled(call_ids=[call.call_id]))  # the user moved on
    await asyncio.wait_for(cancelled.wait(), 2)
    await wait_for(lambda: len(rec.turn_metrics()) == 1)
    await wait_for(lambda: session.agent_state == AgentState.LISTENING)
    await session.aclose()
    assert rec.of("tool_result") == [] and rec.said() == []
    (ev,) = rec.of("tool_cancelled")
    assert ev.call.call_id == call.call_id
    assert not any(isinstance(i, FunctionCallOutput) for i in session.history.items)


async def test_app_cancels_a_non_blocking_call() -> None:
    @function_tool(blocking=False)
    async def slow() -> str:
        """Slow."""
        await asyncio.sleep(10)
        return "never"

    session = make_session("native", [MockToolCall("slow"), "Started.", "Unexpected."])
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x", tools=[slow]), transport)
    await speak(transport)
    await wait_for(lambda: rec.said() == ["Started."])
    call = rec.of("tool_call")[0].call
    assert session.cancel_tool_call(call.call_id)
    assert not session.cancel_tool_call("unknown")
    await wait_for(lambda: bool(rec.of("tool_cancelled")))
    conn: MockEngineConnection = engine_conn(session)
    await wait_for(
        lambda: any("was cancelled" in getattr(i, "text", "") for i in conn.chat_ctx.items)
    )
    await asyncio.sleep(0.2)
    await session.aclose()
    assert rec.said() == ["Started."]  # the note is silent
    assert rec.of("tool_result") == []


# ------------------------------------------------------------------------ delegation
async def test_delegate_runs_in_the_background_and_reports_back() -> None:
    session = make_session("native", ["Hello.", "The report is ready."])
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport)
    await wait_for(lambda: rec.said() == ["Hello."])

    async def research() -> dict[str, int]:
        await asyncio.sleep(0.1)
        return {"sources": 3}

    task = session.delegate(research, name="research")
    assert await asyncio.wait_for(task, 2) == {"sources": 3}
    await wait_for(lambda: rec.said() == ["Hello.", "The report is ready."])
    await session.aclose()
    (msg,) = [i for i in session.history.items
              if isinstance(i, ChatMessage) and i.metadata.get("delegated")]  # fmt: skip
    assert msg.text == 'Result of the background task research: {"sources": 3}'
    with pytest.raises(ValueError):
        session.delegate(research, scheduling="later")  # type: ignore[arg-type]


# --------------------------------------------------------------- Gemini Live (native)
async def test_gemini_non_blocking_tool_scheduling() -> None:
    from voice_agent_next.providers.google.live import GeminiLiveEngine
    from voice_agent_next.testing.gemini_live import FakeGeminiLiveServer, FakeToolCall

    @function_tool(blocking=False, scheduling="silent")
    async def remember(fact: str) -> str:
        """Remember a fact."""
        await asyncio.sleep(0.1)
        return f"remembered {fact}"

    server = FakeGeminiLiveServer(
        replies=[FakeToolCall("remember", {"fact": "cats"}), "Noted."], transcripts=["remember"]
    )
    await server.start()
    try:
        engine = GeminiLiveEngine(api_key="fake-gemini-key", base_url=server.url, rotate_after=None)
        session = AgentSession(engine, options=SessionOptions(tool_filler_delay=0.05))
        rec = Recorder(session)
        await session.start(Agent("x", tools=[remember]), LoopbackTransport())
        await speak(session.transport)  # type: ignore[arg-type]
        await wait_for(lambda: bool(server.connection.tool_responses))
        await session.aclose()
    finally:
        await server.aclose()
    assert server.errors == []
    (response,) = server.connection.tool_responses
    assert response["scheduling"] == "SILENT"
    assert response["response"] == {"result": "remembered cats"}
    assert rec.of("tool_filler") == []  # the model keeps talking: no filler
    assert rec.of("tool_result")[0].blocking is False
