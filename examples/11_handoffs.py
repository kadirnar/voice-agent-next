"""Multi-agent handoffs, shared userdata and a conversation flow.

Two scenarios (docs/concepts/handoffs.md):

* **desk** — a front desk agent hands the caller to a billing agent. The transfer tool
  returns the next agent. The billing agent gets its own instructions, tools and voice,
  plus a summary of the call so far. Both agents share typed ``session.userdata``.
* **flow** — a restaurant booking as a :class:`Flow` of three steps (greet -> party ->
  confirm). The model moves between them by calling transition tools, and a handler
  validates and stores the party size.

Run::

    python examples/11_handoffs.py --mock                                 # offline, both scenarios
    python examples/11_handoffs.py --scenario desk --engine openai/gpt-realtime-2.1
    python examples/11_handoffs.py --scenario flow --stt deepgram --llm openai --tts cartesia
    python examples/11_handoffs.py --scenario desk --engine google --wav question.wav

A cascade (``--stt/--llm/--tts``) switches the voice and the model's context on a
handoff. Native engines switch instructions and tools, and keep the voice and the full
context they already have.

Without ``--mock``: the engine's API key (``OPENAI_API_KEY``, ``GOOGLE_API_KEY``), or for
the cascade the extras ``deepgram,openai,cartesia,silero`` and their keys; extra ``audio``
for the microphone.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from _common import log_conversation, simulated_caller

from voice_agent_next import (
    Agent,
    AgentSession,
    AgentState,
    Flow,
    FlowNode,
    Handoff,
    ToolContext,
    Transition,
    function_tool,
)
from voice_agent_next.chat import ChatContext
from voice_agent_next.errors import ToolError
from voice_agent_next.providers.mock import MockEngine, MockResponse, MockToolCall
from voice_agent_next.transports import FileTransport, LoopbackTransport, create_transport


# ------------------------------------------------------------------- shared state
@dataclass
class CallState:
    """What every agent of the call can read and write (``session.userdata``)."""

    issue: str | None = None
    refunds: list[float] = field(default_factory=list)
    party_size: int | None = None


# ------------------------------------------------------------------ desk scenario
class Billing(Agent):
    def __init__(self) -> None:
        super().__init__(
            "You are the billing specialist of ACME. Solve payment problems, refund when "
            "the customer was charged by mistake. Keep replies short.",
            name="billing",
            voice="sage",  # a different voice, where the engine can switch mid-call
        )

    @function_tool
    async def refund(self, ctx: ToolContext[CallState], amount: float) -> str:
        """Refund an amount (in euros) to the customer."""
        ctx.userdata.refunds.append(amount)
        return f"Refunded {amount:g} EUR."


class FrontDesk(Agent):
    def __init__(self) -> None:
        super().__init__(
            "You are the front desk of ACME. Find out what the caller needs and transfer "
            "payment questions to billing.",
            name="front",
        )

    @function_tool
    async def transfer_to_billing(self, ctx: ToolContext[CallState], issue: str) -> Handoff:
        """Transfer the caller to billing.

        Args:
            issue: the billing problem in a few words.
        """
        ctx.userdata.issue = issue
        return Handoff(Billing(), history="summary", message="Transferring to billing.")


# ------------------------------------------------------------------ flow scenario
async def set_party_size(ctx: ToolContext[CallState], people: int) -> str:
    """Record how many people are coming."""
    if not 1 <= people <= 10:
        raise ToolError("We take tables for 1 to 10 people.")  # stay on this step
    ctx.userdata.party_size = people
    return f"Table for {people} noted."


def booking_flow() -> Flow:
    return Flow(
        [
            FlowNode(
                "greet",
                "Greet the caller and ask whether they want to book a table.",
                greeting="Welcome to Trattoria Roma! Would you like to book a table?",
                transitions=[Transition("party", "The caller wants to book a table.")],
            ),
            FlowNode(
                "party",
                "Ask how many people are coming.",
                transitions=[Transition("confirm", handler=set_party_size)],
            ),
            FlowNode("confirm", "Confirm the booking in one sentence and say goodbye."),
        ],
        role="You are the booking assistant of Trattoria Roma. Keep replies short.",
    )


# ---------------------------------------------------------------------- mock model
def scripted(session: Callable[[], AgentSession], scripts: dict[str, list[MockResponse]]) -> Any:
    """A scripted model that answers as whichever agent is active."""

    def respond(ctx: ChatContext) -> MockResponse:
        script = scripts[session().agent.name]
        return script.pop(0) if script else "Anything else?"

    return respond


MOCK: dict[str, tuple[list[str], dict[str, list[MockResponse]]]] = {
    "desk": (
        ["Hi, I was charged twice for my order.", "Yes please."],
        {
            "front": [MockToolCall("transfer_to_billing", {"issue": "double charge"})],
            "billing": [
                "I see the double charge on order 1042. Shall I refund 25 euros?",
                MockToolCall("refund", {"amount": 25}),
                "Done: 25 euros are on their way back to you.",
            ],
        },
    ),
    "flow": (
        ["Yes, for tonight.", "We are twelve.", "Four then."],
        {
            "greet": [MockToolCall("go_to_party")],
            "party": [
                "How many people are coming?",
                MockToolCall("set_party_size", {"people": 12}),
                "Sorry, we take at most ten. How many people?",
                MockToolCall("set_party_size", {"people": 4}),
            ],
            "confirm": ["A table for four tonight. See you soon!"],
        },
    ),
}


# ---------------------------------------------------------------------------- run
def build_session(args: argparse.Namespace, scenario: str) -> AgentSession[CallState]:
    state = CallState()
    if args.mock:
        transcripts, scripts = MOCK[scenario]
        holder: list[AgentSession[CallState]] = []
        engine = MockEngine(transcripts=transcripts, chars_per_second=60,
                            responses=scripted(lambda: holder[0], scripts))  # fmt: skip
        holder.append(AgentSession(engine, userdata=state))
        return holder[0]
    if args.llm:  # a cascade: voice and context switch on handoffs
        return AgentSession(stt=args.stt, llm=args.llm, tts=args.tts, vad="silero",
                            userdata=state)  # fmt: skip
    return AgentSession(args.engine, userdata=state)


def first_reply_done(session: AgentSession[Any]) -> asyncio.Event:
    """Set once the agent has spoken and listens again (e.g. after its greeting)."""
    done = asyncio.Event()
    spoke = False

    def on_state(ev: Any) -> None:
        nonlocal spoke
        spoke = spoke or ev.new_state == AgentState.SPEAKING
        if spoke and ev.new_state == AgentState.LISTENING:
            done.set()

    session.on("agent_state_changed", on_state)
    return done


async def run(args: argparse.Namespace, scenario: str) -> None:
    print(f"--- scenario: {scenario}")
    session = build_session(args, scenario)
    log_conversation(session)
    session.on(
        "agent_handoff",
        lambda ev: print(
            f"  == handoff {ev.from_agent} -> {ev.to_agent} (history={ev.history}"
            + (f", engine kept: {', '.join(ev.unsupported)}" if ev.unsupported else "")
            + ")"
        ),
    )
    flow = booking_flow() if scenario == "flow" else None
    agent: Agent = flow.agent() if flow is not None else FrontDesk()
    if args.mock:
        transport = LoopbackTransport()
        greeted = first_reply_done(session)
        await session.start(agent, transport)
        if agent.greeting:  # a polite caller lets the agent finish its greeting
            await asyncio.wait_for(greeted.wait(), timeout=10)
        turns = len(MOCK[scenario][0])
        await simulated_caller(session, transport, turns=turns, quiet=0.6)
        await session.wait_closed()
    elif args.wav:
        await session.run(agent, FileTransport(args.wav, f"reply-{scenario}.wav", hold=3.0))
    else:
        await session.run(agent, create_transport("local"))
    print(f"userdata: {session.userdata}")
    if flow is not None:
        print(f"flow path: {' -> '.join(flow.path)}")


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mock", action="store_true", help="offline, scripted model")
    parser.add_argument("--scenario", choices=["desk", "flow"], help="default: both with --mock")
    parser.add_argument("--engine", default="openai/gpt-realtime-2.1", help="native engine spec")
    parser.add_argument("--stt", default="deepgram", help="cascade STT (with --llm)")
    parser.add_argument("--llm", help="cascade LLM, e.g. openai (selects the cascade)")
    parser.add_argument("--tts", default="cartesia", help="cascade TTS (with --llm)")
    parser.add_argument("--wav", type=Path, help="talk from a WAV file instead of the mic")
    args = parser.parse_args(argv)
    scenarios = [args.scenario] if args.scenario else ["desk", "flow"] if args.mock else ["desk"]
    for scenario in scenarios:
        await run(args, scenario)
    return 0


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        sys.exit(asyncio.run(main()))
