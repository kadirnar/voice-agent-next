"""Tools that take time: fillers, progress updates and non-blocking tools.

Three ways to keep a voice conversation natural while a tool runs (docs/concepts/tools.md):

* **Filler.** A *blocking* tool that runs longer than ``tool_filler_delay`` makes the agent
  say a short phrase ("One moment...") so the caller does not hear dead air.
* **Progress.** A tool that declares a ``ToolContext`` parameter can report progress. The
  agent speaks it, adds it to the model's context, or both.
* **Non-blocking.** ``@function_tool(blocking=False)`` answers the model at once, so the
  conversation goes on. The result is added when it arrives (``scheduling``:
  ``when_idle``, ``interrupt`` or ``silent``). Gemini Live does this natively. Other
  engines get an immediate acknowledgement as the tool output.

Run::

    python examples/05_tools.py --mock                         # offline, scripted model
    python examples/05_tools.py --engine openai/gpt-realtime-2.1   # a real model, mic + speakers
    python examples/05_tools.py --engine google --wav two_questions.wav

In ``--mock`` mode a simulated caller asks two questions over an in-memory transport.
The scripted model calls ``search_flights``, which is slow and reports progress. Then it
calls ``email_itinerary``, which is non-blocking: the agent keeps talking, and the
result arrives later.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from pathlib import Path
from typing import Any

from _common import log_conversation, simulated_caller

from voice_agent_next import (
    Agent,
    AgentSession,
    SessionOptions,
    ToolContext,
    function_tool,
)
from voice_agent_next.providers.mock import MockEngine, MockToolCall
from voice_agent_next.transports import FileTransport, LoopbackTransport, create_transport

FAST = 0.12  # --mock shortens the tools' waits by this factor


# ------------------------------------------------------------------------------ tools
@function_tool(filler=["Let me look that up.", "Searching flights now."])
async def search_flights(ctx: ToolContext, destination: str) -> str:
    """Search flights to a destination and return the cheapest one."""
    scale = ctx.userdata["time_scale"]
    await asyncio.sleep(8 * scale)  # a slow API: the filler plays after tool_filler_delay
    # Spoken (the round's filler is then skipped) and not added to the model's context:
    await ctx.report_progress(f"Found 3 flights to {destination}, comparing prices.")
    # Given to the model silently, so it can use it in its answer:
    await ctx.report_progress("Cheapest so far: 90 euros.", speak=False, to_model=True)
    await asyncio.sleep(3 * scale)
    return f"Cheapest flight to {destination}: 90 EUR, departs 09:10."


@function_tool(blocking=False, scheduling="when_idle")
async def email_itinerary(ctx: ToolContext) -> str:
    """Email the itinerary to the user (runs in the background)."""
    await asyncio.sleep(5 * ctx.userdata["time_scale"])
    return "Itinerary emailed to the user."


# ---------------------------------------------------------------------------- session
def mock_engine() -> MockEngine:
    """A scripted model: one entry per model response, in order."""
    return MockEngine(
        transcripts=["Find me a flight to Rome.", "Great, email it to me."],
        responses=[
            MockToolCall("search_flights", {"destination": "Rome"}),
            "The cheapest is 90 euros at 9:10.",
            MockToolCall("email_itinerary"),
            "I'm sending it now. Anything else?",  # said while the email is being sent
            "Done: the itinerary is in your inbox.",  # the background result arrived
        ],
        chars_per_second=60,  # fast mock speech: keeps the smoke test short
    )


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mock", action="store_true", help="offline, scripted model")
    parser.add_argument("--engine", default="openai/gpt-realtime-2.1", help="engine spec")
    parser.add_argument("--wav", type=Path, help="talk from a WAV file instead of the mic")
    args = parser.parse_args(argv)

    options = SessionOptions(
        tool_filler_delay=0.2 if args.mock else 1.5,  # say a filler after this many seconds
        tool_filler_interruptible=True,
    )
    userdata = {"time_scale": FAST if args.mock else 1.0}
    session = AgentSession(mock_engine() if args.mock else args.engine, options=options,
                           userdata=userdata)  # fmt: skip
    log_conversation(session)
    session.on("tool_filler", lambda ev: print(f"  (filler after {ev.waited:.1f} s: {ev.text!r})"))
    session.on(
        "tool_progress", lambda ev: print(f"  (progress, spoken={ev.spoken}: {ev.message!r})")
    )

    def on_result(ev: Any) -> None:
        if not ev.blocking:
            print(f"  (background result after {ev.duration:.1f} s)")

    session.on("tool_result", on_result)

    agent = Agent(
        "You are a travel agent. Use your tools.", tools=[search_flights, email_itinerary]
    )
    if args.mock:
        # an in-memory call: a simulated caller asks two questions, waiting for each answer
        loopback = LoopbackTransport()
        await session.start(agent, loopback)
        await simulated_caller(session, loopback, turns=2, quiet=0.6)
        await session.wait_closed()
    elif args.wav:
        # hold: stay on the line until the agent has been quiet for 3 s (background results)
        await session.run(agent, FileTransport(args.wav, "reply.wav", hold=3.0))
        print("reply written to reply.wav")
    else:
        await session.run(agent, create_transport("local"))
    return 0


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        sys.exit(asyncio.run(main()))
