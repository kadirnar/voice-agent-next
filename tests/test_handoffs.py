"""Multi-agent handoffs, shared userdata and conversation flows (issue #32).

Everything runs offline against the scripted native mock engine and the mock cascade.
The scripted models answer according to the active agent's instructions (see
:class:`Router`), so each test checks which agent produced which reply.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from voice_agent_next import (
    Agent,
    AgentSession,
    AgentState,
    AudioFrame,
    CascadeOptions,
    ChatContext,
    ChatMessage,
    Flow,
    FlowNode,
    Handoff,
    SessionOptions,
    ToolContext,
    Transition,
    function_tool,
)
from voice_agent_next.chat import FunctionCall, FunctionCallOutput
from voice_agent_next.errors import ToolError
from voice_agent_next.llm import ChatChunk, CompletionUsage
from voice_agent_next.metrics import TurnMetrics
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import (
    MockEngine,
    MockLLM,
    MockResponse,
    MockSTT,
    MockToolCall,
    MockTTS,
    _MockLLMStream,
    synth_speech,
)
from voice_agent_next.session import AgentHandoff, without_tool_items
from voice_agent_next.session.handoff import as_handoff, carry_history
from voice_agent_next.transports import LoopbackTransport

EVENTS = ("agent_transcript", "tool_call", "tool_result", "agent_handoff", "interrupted",
          "metrics", "error", "agent_state_changed")  # fmt: skip
ENGINES = ["native", "cascade"]
CPS = 60.0
"""Mock speech rate (characters/s): fast, to keep the tests short."""


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
        texts: dict[str, str] = {}
        for e in self.of("agent_transcript"):
            texts[e.response_id] = texts.get(e.response_id, "") + e.delta
        return [t.strip() for t in texts.values()]

    def handoffs(self) -> list[AgentHandoff]:
        return self.of("agent_handoff")


@dataclass
class Router:
    """A scripted model that answers by role: the first ``ROLE:<name>`` in the system
    prompt picks the script. Records what each role saw."""

    scripts: dict[str, list[MockResponse]]
    seen: dict[str, list[ChatContext]] = field(default_factory=dict)

    def __call__(self, ctx: ChatContext) -> MockResponse:
        system = ctx.items[0].text if ctx.items and isinstance(ctx.items[0], ChatMessage) else ""
        if system.startswith("Summarize"):
            return "SUMMARY: the caller asked about an invoice."
        match = re.search(r"ROLE:(\w+)", system)
        role = match.group(1) if match else "?"
        self.seen.setdefault(role, []).append(ctx.copy())
        script = self.scripts.get(role, [])
        return script.pop(0) if script else f"{role} has nothing more to say."


def make_session(
    kind: str, router: Router, transcripts: list[str], *, realtime_factor: float = 0.0,
    options: SessionOptions | None = None, userdata: Any = None, **engine: Any,
) -> AgentSession:  # fmt: skip
    if kind == "native":
        mock = MockEngine(transcripts=transcripts, responses=router, chars_per_second=CPS,
                          realtime_factor=realtime_factor, **engine)  # fmt: skip
        return AgentSession(mock, options=options, userdata=userdata)
    return AgentSession(
        stt=MockSTT(transcripts=transcripts),
        llm=MockLLM(responses=router),
        tts=MockTTS(realtime_factor=realtime_factor, chars_per_second=CPS),
        vad=EnergyVAD(),
        cascade_options=CascadeOptions(min_endpointing_delay=0.0),
        options=options,
        userdata=userdata,
    )


async def speak(transport: LoopbackTransport, seconds: float = 0.8) -> None:
    await transport.play_user_audio(synth_speech(seconds, 16_000), realtime=False)
    await transport.play_user_audio(AudioFrame.silence(0.6, 16_000), realtime=False)


async def wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


async def turn(
    session: AgentSession, transport: LoopbackTransport, rec: Recorder, text: str
) -> None:
    """The user says something; wait until the agent said ``text`` and went quiet."""
    await speak(transport)
    await wait_for(lambda: any(text in s for s in rec.said()))
    await wait_for(lambda: session.agent_state == AgentState.LISTENING)


def texts(ctx: ChatContext) -> list[str]:
    return [i.text for i in ctx.items if isinstance(i, ChatMessage)]


def body(seen: ChatContext) -> list[Any]:
    """What a model saw after its system prompt, without the reply being generated (the
    cascade adds the reply's message to its context before calling the LLM; the router
    records it by reference, so it shows its final text)."""
    return [i for i in seen.items[1:] if not (isinstance(i, ChatMessage)
            and i.role == "assistant" and i is seen.items[-1])]  # fmt: skip


# --------------------------------------------------------------------------- agents
class Tracked(Agent):
    """An agent that logs its lifecycle hooks into a shared list."""

    def __init__(self, name: str, log: list[str], **kw: Any) -> None:
        super().__init__(f"ROLE:{name} You are the {name} agent.", name=name, **kw)
        self.log = log

    async def on_enter(self, session: AgentSession) -> None:
        self.log.append(f"enter:{self.name}")

    async def on_exit(self, session: AgentSession) -> None:
        self.log.append(f"exit:{self.name}")


class Billing(Tracked):
    def __init__(self, log: list[str], **kw: Any) -> None:
        super().__init__("billing", log, **kw)

    @function_tool
    async def refund(self, amount: int) -> str:
        """Refund an amount to the caller."""
        return f"refunded {amount}"


class Front(Tracked):
    def __init__(self, log: list[str], target: Callable[[], Agent | Handoff]) -> None:
        super().__init__("front", log)
        self._target = target

    @function_tool
    async def transfer_to_billing(self) -> Agent | Handoff:
        """Transfer the caller to billing."""
        return self._target()


# ------------------------------------------------------------------------ handoffs
@pytest.mark.parametrize("kind", ENGINES)
async def test_tool_returning_an_agent_hands_the_conversation_over(kind: str) -> None:
    log: list[str] = []
    router = Router({"front": [MockToolCall("transfer_to_billing")],
                     "billing": ["Billing here, how can I help?"]})  # fmt: skip
    session = make_session(kind, router, ["I have a question about my invoice"])
    rec = Recorder(session)
    transport = LoopbackTransport()
    front = Front(log, lambda: Billing(log))
    await session.start(front, transport)
    await turn(session, transport, rec, "Billing here")
    await session.aclose()

    (ev,) = rec.handoffs()
    assert (ev.from_agent, ev.to_agent, ev.history) == ("front", "billing", "full")
    assert ev.call is not None and ev.call.name == "transfer_to_billing"
    assert ev.unsupported == [] and ev.duration >= 0
    assert session.agent.name == "billing"
    # hooks: front entered at start, exited before billing entered; billing exits on close
    assert log == ["enter:front", "exit:front", "enter:billing", "exit:billing"]
    # the engine switched instructions and tools
    conn: Any = session.connection
    assert conn.instructions.startswith("ROLE:billing")
    assert [t.name for t in conn.tools] == ["refund"]
    # the model got the default handoff message as the tool output, and only billing spoke
    outputs = [i for i in session.history.items if isinstance(i, FunctionCallOutput)]
    assert [o.output for o in outputs] == ["Transferred to billing."]
    assert rec.said() == ["Billing here, how can I help?"]
    # billing saw the full conversation (the user's question and the transfer)
    (seen,) = router.seen["billing"]
    assert "I have a question about my invoice" in texts(seen)
    assert any(isinstance(i, FunctionCall) for i in seen.items)
    turns = [m for m in rec.of("metrics") if isinstance(m, TurnMetrics)]
    assert turns and turns[-1].agent == "billing" and turns[-1].voice_to_voice is not None


@pytest.mark.parametrize("kind", ENGINES)
@pytest.mark.parametrize("mode", ["full", "summary", "none", "custom"])
async def test_history_carry_over_modes(kind: str, mode: str) -> None:
    log: list[str] = []
    router = Router({"front": ["Front answer one.", MockToolCall("transfer_to_billing")],
                     "billing": ["Billing answer."]})  # fmt: skip
    session = make_session(kind, router, ["first question", "second question"])
    rec = Recorder(session)
    transport = LoopbackTransport()
    history: Any = without_tool_items if mode == "custom" else mode
    front = Front(log, lambda: Handoff(Billing(log), history=history))
    await session.start(front, transport)
    await turn(session, transport, rec, "Front answer one.")
    await turn(session, transport, rec, "Billing answer.")
    await session.aclose()

    assert rec.handoffs()[0].history == mode
    (seen,) = router.seen["billing"]
    items = body(seen)
    said = texts(ChatContext(items))
    if mode == "full":
        assert said == ["first question", "Front answer one.", "second question"]
        assert any(isinstance(i, FunctionCall) for i in items)
    elif mode == "none":
        assert items == []
    elif mode == "custom":  # messages only
        assert said == ["first question", "Front answer one.", "second question"]
        assert not any(isinstance(i, FunctionCall | FunctionCallOutput) for i in items)
    else:
        summary = [i for i in items if isinstance(i, ChatMessage) and i.role == "system"]
        assert len(summary) == 1 and summary[0].metadata.get("carry_over") == "summary"
        if kind == "cascade":  # the cascade's LLM wrote it; recent items stay verbatim
            assert "SUMMARY: the caller asked about an invoice." in summary[0].text
            assert "second question" in said
        else:  # no LLM: a compact transcript of the latest messages
            assert "user: first question" in summary[0].text
            assert "assistant: Front answer one." in summary[0].text
    # the session's own history is the whole call, whatever the model sees
    all_text = texts(session.history)
    assert all_text[:3] == ["first question", "Front answer one.", "second question"]
    assert all_text[-1] == "Billing answer."


async def test_engine_without_mid_session_updates_falls_back_gracefully() -> None:
    log: list[str] = []
    router = Router({"front": [MockToolCall("transfer_to_billing")], "billing": ["Hello."]})
    session = make_session("native", router, ["hi"], voice_updates=False, chat_ctx_updates=False)
    rec = Recorder(session)
    transport = LoopbackTransport()
    front = Front(log, lambda: Handoff(Billing(log, voice="alloy"), history="none"))
    await session.start(front, transport)
    await turn(session, transport, rec, "Hello.")
    await session.aclose()

    (ev,) = rec.handoffs()
    assert ev.unsupported == ["chat_ctx", "voice"] and not ev.voice_changed
    assert rec.of("error") == []
    # the model kept its full context and the old voice; instructions/tools still switched
    assert "hi" in texts(router.seen["billing"][0])
    conn: Any = session.connection
    assert conn.options.voice is None and conn.instructions.startswith("ROLE:billing")


@pytest.mark.parametrize("kind", ENGINES)
async def test_voice_switches_when_the_engine_supports_it(kind: str) -> None:
    log: list[str] = []
    router = Router({"front": [MockToolCall("transfer_to_billing")], "billing": ["Hello."]})
    session = make_session(kind, router, ["hi"])
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Front(log, lambda: Billing(log, voice="alloy")), transport)
    await turn(session, transport, rec, "Hello.")
    await session.aclose()
    (ev,) = rec.handoffs()
    assert ev.voice_changed and ev.unsupported == []
    assert session.connection.options.voice == "alloy"


@pytest.mark.parametrize("kind", ENGINES)
async def test_tools_switch_with_the_agent_and_back(kind: str) -> None:
    log: list[str] = []
    front = Front(log, lambda: billing)

    class BillingWithBack(Billing):
        @function_tool
        async def back_to_front(self) -> tuple[Agent, str]:
            """Return the caller to the front desk."""
            return front, "Back at the front desk."

    billing = BillingWithBack(log)
    router = Router({
        "front": [MockToolCall("transfer_to_billing"), "Front again."],
        "billing": ["Billing.", MockToolCall("refund", {"amount": 5}), "Refunded five.",
                    MockToolCall("back_to_front")],
    })  # fmt: skip
    session = make_session(kind, router, ["bill", "refund me", "thanks"])
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(front, transport)
    await turn(session, transport, rec, "Billing.")
    await turn(session, transport, rec, "Refunded five.")
    await turn(session, transport, rec, "Front again.")
    await session.aclose()

    results = {e.call.name: e.output.output for e in rec.of("tool_result")}
    assert results["refund"] == "refunded 5"
    assert results["back_to_front"] == "Back at the front desk."  # the tuple's message
    assert [(h.from_agent, h.to_agent) for h in rec.handoffs()] == [
        ("front", "billing"),
        ("billing", "front"),
    ]
    assert session.agent is front
    assert [t.name for t in session.connection.tools] == ["transfer_to_billing"]  # type: ignore[attr-defined]


@pytest.mark.parametrize("kind", ENGINES)
async def test_greeting_on_enter_and_respond_false(kind: str) -> None:
    log: list[str] = []

    class Speaks(Tracked):
        async def on_enter(self, session: AgentSession) -> None:
            await super().on_enter(session)
            await session.say("Custom hello from on_enter.")

    targets: list[Agent | Handoff] = [
        Billing(log, greeting="Hi, billing speaking."),  # greeting: said verbatim
        Speaks("speaks", log),  # on_enter speaks: no generated reply on top
        Handoff(Tracked("silent", log), respond=False),  # waits for the user
    ]
    router = Router({"front": [MockToolCall("transfer_to_billing")],
                     "billing": [MockToolCall("transfer_to_billing")],
                     "speaks": [MockToolCall("transfer_to_billing")],
                     "silent": ["Silent answers when asked."]})  # fmt: skip
    session = make_session(kind, router, ["a", "b", "c", "d"])
    rec = Recorder(session)
    transport = LoopbackTransport()
    front = Front(log, lambda: targets.pop(0))
    for agent in targets[:2]:  # every agent can transfer on to the next one
        agent.tools.append(front.tools[0])
    await session.start(front, transport)
    await turn(session, transport, rec, "Hi, billing speaking.")
    await turn(session, transport, rec, "Custom hello from on_enter.")
    await speak(transport)
    await wait_for(lambda: len(rec.handoffs()) == 3)
    await asyncio.sleep(0.3)
    assert session.agent.name == "silent" and session.agent_state == AgentState.LISTENING
    assert "silent" not in router.seen  # it did not speak on its own
    await turn(session, transport, rec, "Silent answers when asked.")
    await session.aclose()

    assert rec.said() == [
        "Hi, billing speaking.",
        "Custom hello from on_enter.",
        "Silent answers when asked.",
    ]
    # billing and speaks only ran the model to transfer on (turns b and c)
    assert len(router.seen["billing"]) == 1 and len(router.seen["speaks"]) == 1
    assert log[:6] == ["enter:front", "exit:front", "enter:billing", "exit:billing",
                       "enter:speaks", "exit:speaks"]  # fmt: skip


# ------------------------------------------------------------------------ userdata
@dataclass
class Caller:
    name: str | None = None
    refunds: list[int] = field(default_factory=list)


@pytest.mark.parametrize("kind", ENGINES)
async def test_tool_context_handoff_and_typed_userdata(kind: str) -> None:
    log: list[str] = []

    @function_tool
    async def refund(ctx: ToolContext[Caller], amount: int) -> str:
        """Refund money."""
        ctx.userdata.refunds.append(amount)
        return f"refunded {amount} to {ctx.userdata.name}"

    billing = Tracked("billing", log, tools=[refund])

    @function_tool
    async def identify(ctx: ToolContext[Caller], name: str) -> str:
        """Identify the caller and move them to billing."""
        ctx.userdata.name = name
        ctx.handoff(billing, history="none", message="ignored: the tool returned text")
        return f"Identified {name}, transferring."

    router = Router({
        "front": [MockToolCall("identify", {"name": "Ada"})],
        "billing": ["Billing.", MockToolCall("refund", {"amount": 7}), "Done."],
    })  # fmt: skip
    session: AgentSession[Caller] = make_session(kind, router, ["I'm Ada", "refund 7"],
                                                 userdata=Caller())  # fmt: skip
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Tracked("front", log, tools=[identify]), transport)
    await turn(session, transport, rec, "Billing.")
    await turn(session, transport, rec, "Done.")
    await session.aclose()

    assert session.userdata == Caller("Ada", [7])
    results = [e.output.output for e in rec.of("tool_result")]
    assert results == ["Identified Ada, transferring.", "refunded 7 to Ada"]
    assert rec.handoffs()[0].history == "none"
    assert body(router.seen["billing"][0]) == []  # history="none"


# -------------------------------------------------------------------- interruption
class SpeakThenCallLLM(MockLLM):
    """A model that says something *and* calls a tool in the same response
    (``(text, MockToolCall)`` script entries)."""

    def _chat(self, ctx: ChatContext, **kw: Any) -> Any:
        self.requests.append(ctx.copy())
        return _SpeakThenCallStream(self, ctx, **kw)


class _SpeakThenCallStream(_MockLLMStream):
    async def _run(self) -> None:
        llm: MockLLM = self._llm  # type: ignore[assignment]
        response: Any = llm.script.next(self.ctx)
        text, call = response if isinstance(response, tuple) else (response, None)
        if isinstance(text, MockToolCall):
            text, call = "", text
        if text:
            self._push(ChatChunk(self.request_id, delta=text))
        if call is not None:
            fc = FunctionCall(name=call.name, arguments=call.arguments_json())
            self._push(ChatChunk(self.request_id, tool_calls=[fc]))
        self._push(ChatChunk(self.request_id, usage=CompletionUsage(prompt_tokens=1,
                             completion_tokens=1), finish_reason="stop"))  # fmt: skip


@pytest.mark.parametrize("kind", ENGINES)
async def test_handoff_when_the_transfer_announcement_is_interrupted(kind: str) -> None:
    """The model says "let me transfer you" and calls the transfer tool; the user talks
    over it.

    Native engines report the call while the announcement still plays: the handoff still
    happens, but the new agent does not speak on its own, it answers the user's new turn.
    The cascade reports a response's tool calls once it has been spoken, so an interrupted
    announcement drops its call: no handoff, the current agent answers."""
    log: list[str] = []
    announcement = "Let me transfer you to our billing team, please stay on the line for a moment."
    router = Router({"front": [(announcement, MockToolCall("transfer_to_billing"))],  # type: ignore[list-item]
                     "billing": ["Billing: your balance is zero."]})  # fmt: skip
    transcripts = ["transfer me", "actually what is my balance"]
    llm = SpeakThenCallLLM(responses=router)
    if kind == "native":
        engine = MockEngine(transcripts=transcripts, chars_per_second=30.0, realtime_factor=1.0)
        engine.llm = llm
        session = AgentSession(engine)
    else:
        session = AgentSession(
            stt=MockSTT(transcripts=transcripts), llm=llm,
            tts=MockTTS(realtime_factor=1.0, chars_per_second=30.0), vad=EnergyVAD(),
            cascade_options=CascadeOptions(min_endpointing_delay=0.0),
        )  # fmt: skip
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await session.start(Front(log, lambda: Billing(log)), transport)
    await speak(transport)
    await wait_for(lambda: session.agent_state == AgentState.SPEAKING)
    await asyncio.sleep(0.6)
    # barge in over the announcement, in real time: the interruption is confirmed while the
    # user still talks, so the transfer completes before the engine answers the new turn
    await transport.play_user_audio(synth_speech(1.0, 16_000), realtime=True)
    await transport.play_user_audio(AudioFrame.silence(0.6, 16_000), realtime=False)
    await wait_for(lambda: bool(rec.of("interrupted")))
    reply = "balance is zero" if kind == "native" else "front has nothing more to say"
    await wait_for(lambda: any(reply in s for s in rec.said()), timeout=8.0)
    await session.aclose()

    assistant = [i for i in session.history.items
                 if isinstance(i, ChatMessage) and i.role == "assistant"]  # fmt: skip
    assert assistant[0].interrupted and assistant[0].text != announcement
    if kind == "cascade":
        assert rec.handoffs() == [] and rec.of("tool_call") == []
        assert session.agent.name == "front" and "billing" not in router.seen
        return
    (ev,) = rec.handoffs()
    assert (ev.from_agent, ev.to_agent) == ("front", "billing")
    # billing answered exactly once: the user's new turn, not an automatic reply
    assert len(router.seen["billing"]) == 1
    assert "actually what is my balance" in texts(router.seen["billing"][0])
    assert assistant[-1].text == "Billing: your balance is zero."


# ------------------------------------------------------------------------ app API
async def test_session_handoff_from_application_code() -> None:
    log: list[str] = []
    router = Router({"billing": ["Billing speaking."]})
    session = make_session("native", router, [])
    rec = Recorder(session)
    await session.start(Tracked("front", log), LoopbackTransport())
    await session.handoff(Billing(log), history="none")
    await wait_for(lambda: "Billing speaking." in rec.said())
    await session.aclose()
    (ev,) = rec.handoffs()
    assert ev.call is None and ev.history == "none"
    assert log == ["enter:front", "exit:front", "enter:billing", "exit:billing"]


# --------------------------------------------------------------------------- flows
@pytest.mark.parametrize("kind", ENGINES)
async def test_three_node_flow(kind: str) -> None:
    @dataclass
    class Booking:
        people: int | None = None

    entered: list[str] = []

    async def set_party_size(ctx: ToolContext[Booking], people: int) -> str:
        """Record how many people are coming."""
        if people > 10:
            raise ToolError("We only take tables of up to 10 people.")
        ctx.userdata.people = people
        return f"Party of {people} recorded."

    flow = Flow(
        [
            FlowNode("greet", "ROLE:greet Greet the caller.", greeting="Welcome to Roma!",
                     transitions=[Transition("collect", "The caller wants to book.")]),
            FlowNode("collect", "ROLE:collect Ask how many people are coming.",
                     transitions=[Transition("confirm", handler=set_party_size)],
                     on_enter=lambda session: entered.append("collect")),
            FlowNode("confirm", "ROLE:confirm Confirm the booking and say goodbye.",
                     history="none"),
        ],
        role="You work at Trattoria Roma.",
    )  # fmt: skip
    router = Router({
        "greet": [MockToolCall("go_to_collect")],
        "collect": ["How many people?", MockToolCall("set_party_size", {"people": 12}),
                    "Sorry, at most ten. How many?",
                    MockToolCall("set_party_size", {"people": 4})],
        "confirm": ["A table for four. Goodbye!"],
    })  # fmt: skip
    booking = Booking()
    session = make_session(kind, router, ["a table please", "twelve", "four then"],
                           userdata=booking)  # fmt: skip
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(flow.agent(), transport)
    await wait_for(lambda: "Welcome to Roma!" in rec.said())
    await turn(session, transport, rec, "How many people?")
    await turn(session, transport, rec, "Sorry, at most ten.")
    await turn(session, transport, rec, "Goodbye!")
    await session.aclose()

    assert flow.path == ["greet", "collect", "confirm"] and flow.current == "confirm"
    assert entered == ["collect"]
    assert booking.people == 4
    assert [(h.from_agent, h.to_agent) for h in rec.handoffs()] == [
        ("greet", "collect"),
        ("collect", "confirm"),
    ]
    results = [(e.call.name, e.output.is_error) for e in rec.of("tool_result")]
    assert results == [("go_to_collect", False), ("set_party_size", True),
                       ("set_party_size", False)]  # fmt: skip
    assert rec.said() == ["Welcome to Roma!", "How many people?",
                          "Sorry, at most ten. How many?", "A table for four. Goodbye!"]  # fmt: skip
    # every node's instructions start with the shared role; confirm started fresh
    (confirm_ctx,) = router.seen["confirm"]
    assert texts(confirm_ctx)[0].startswith("You work at Trattoria Roma.\n\nROLE:confirm")
    assert body(confirm_ctx) == []
    assert session.connection.tools == []  # type: ignore[attr-defined]


def test_flow_validation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Flow([])
    with pytest.raises(ValueError, match="duplicate flow node"):
        Flow([FlowNode("a", "x"), FlowNode("a", "y")])
    with pytest.raises(ValueError, match="unknown node 'b'"):
        Flow([FlowNode("a", "x", transitions=[Transition("b")])])
    with pytest.raises(ValueError, match="unknown initial"):
        Flow([FlowNode("a", "x")], initial="b")

    def go_to_b() -> None: ...

    with pytest.raises(ValueError, match="duplicate tool names"):
        Flow([FlowNode("a", "x", tools=[go_to_b], transitions=[Transition("b")]),
              FlowNode("b", "y")])  # fmt: skip
    flow = Flow([FlowNode("a", "x", transitions=[Transition("b", name="next")]),
                 FlowNode("b", "y")], initial="b")  # fmt: skip
    assert flow.agent().name == "b" and flow.current is None
    assert [t.name for t in flow.agent("a").tools] == ["next"]
    with pytest.raises(KeyError):
        flow.agent("zzz")


# ------------------------------------------------------------------------- helpers
async def test_carry_history_helpers() -> None:
    ctx = ChatContext()
    ctx.add_message("user", "hello")
    ctx.add_function_call("f", "{}", call_id="c1")
    ctx.add_message("assistant", "hi")
    assert await carry_history(ctx, "full") is None
    assert (await carry_history(ctx, "none")).items == []  # type: ignore[union-attr]
    kept = await carry_history(ctx, without_tool_items)
    assert kept is not None and texts(kept) == ["hello", "hi"] and len(kept.items) == 2

    async def last_only(history: ChatContext) -> ChatContext:
        return ChatContext(history.items[-1:])

    kept = await carry_history(ctx, last_only)
    assert kept is not None and texts(kept) == ["hi"]
    with pytest.raises(TypeError):
        await carry_history(ctx, lambda h: "nope")  # type: ignore[arg-type,return-value]
    with pytest.raises(ValueError):
        Handoff(Agent(), history="most")  # type: ignore[arg-type]

    agent = Agent(name="x")
    assert as_handoff(agent)[0] is not None and as_handoff("text") == (None, "text")
    handoff, rest = as_handoff((agent, "msg"))
    assert handoff is not None and handoff.output == "msg" and rest is None
    assert Handoff(agent).output == "Transferred to x."


# ------------------------------------------------------------ recording and tracing
async def test_handoff_is_recorded_and_traced(tmp_path: Path) -> None:
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from voice_agent_next.session import SessionTracer

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    log: list[str] = []
    router = Router({"front": [MockToolCall("transfer_to_billing")], "billing": ["Hello."]})
    engine = MockEngine(transcripts=["hi"], responses=router, chars_per_second=CPS)
    session = AgentSession(engine, record=tmp_path / "call.wav",
                           trace=SessionTracer(provider))  # fmt: skip
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Front(log, lambda: Billing(log)), transport)
    await turn(session, transport, rec, "Hello.")
    await session.aclose()

    text = (tmp_path / "call.jsonl").read_text(encoding="utf-8")
    lines = [json.loads(line) for line in text.splitlines()]
    (entry,) = [ln for ln in lines if ln["event"] == "agent_handoff"]
    data = entry["data"]
    assert (data["from_agent"], data["to_agent"], data["history"]) == ("front", "billing", "full")
    assert data["call"]["name"] == "transfer_to_billing"

    spans = exporter.get_finished_spans()
    (span,) = [s for s in spans if s.name == "agent_handoff"]
    attrs = dict(span.attributes or {})
    assert attrs["voice_agent.handoff.from"] == "front"
    assert attrs["gen_ai.agent.name"] == "billing"
    root = next(s for s in spans if s.name == "session")
    assert dict(root.attributes or {})["gen_ai.agent.name"] == "front"
    turns = [m for m in rec.of("metrics") if isinstance(m, TurnMetrics)]
    assert turns[-1].agent == "billing"
