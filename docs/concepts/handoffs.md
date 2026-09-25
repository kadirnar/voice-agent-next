# Handoffs and conversation flows

One call often needs more than one agent. A front desk transfers the caller to billing,
or a booking assistant goes through fixed steps: greet, collect details, confirm.
`AgentSession` can switch the active `Agent` in the middle of a call. The new agent's
instructions, tools and, when the engine can, voice take over. The engine connection,
the transport, the history and the metrics carry on, so the caller hears no gap and
there is no reconnect.

| Feature | API |
|---|---|
| Hand over from a tool | return an `Agent`, a `Handoff(...)` or `(agent, "message")`, or call `ctx.handoff(agent, ...)` |
| Hand over from app code | `await session.handoff(agent, history=..., respond=...)` |
| What the new agent sees | `history="full"` / `"summary"` / `"none"` / a callable |
| Shared state | `AgentSession(userdata=...)`, `ToolContext[MyData].userdata` |
| Step-by-step flows | `Flow`, `FlowNode`, `Transition` |
| Observability | `agent_handoff` event, `TurnMetrics.agent`, the recording timeline, an `agent_handoff` span |

## Handing over from a tool

A tool hands the conversation over by returning the next agent:

```python
from voice_agent_next import Agent, function_tool


class Billing(Agent):
    def __init__(self) -> None:
        super().__init__(
            "You handle billing questions: invoices, refunds, payment methods.",
            name="billing",
            voice="sage",
        )

    @function_tool
    async def refund(self, amount: float) -> str:
        """Refund an amount to the caller."""
        return f"Refunded {amount} EUR."


class FrontDesk(Agent):
    def __init__(self) -> None:
        super().__init__("You are the front desk of ACME.", name="front")

    @function_tool
    async def transfer_to_billing(self) -> Agent:
        """Transfer the caller to the billing department."""
        return Billing()
```

When the model calls `transfer_to_billing`, the session does this once the tool round is
over:

1. The model gets a tool output, `"Transferred to billing."` by default. It is sent without
   asking the old agent for a reply.
2. The old agent's `on_exit` runs.
3. The history for the new agent is prepared (see [below](#history-carry-over)).
4. The engine switches to the new instructions and tools (`EngineConnection.update`), then
   to the new context and voice if they changed and the engine supports it (see
   [engine support](#engine-support)).
5. The session emits `agent_handoff`, and `session.agent` is now the new agent.
6. The new agent's `on_enter` runs.
7. The new agent speaks: its `greeting` verbatim if it has one, otherwise a generated
   reply. It stays quiet if `on_enter` already spoke (`session.say()` or
   `session.generate_reply()`), or if the handoff was made with `respond=False`.

For more control, return a `Handoff`, or call `ctx.handoff(...)` from a tool that takes a
`ToolContext`:

```python
from voice_agent_next import Handoff, ToolContext


@function_tool
async def transfer_to_billing() -> Handoff:
    """Transfer the caller to billing."""
    return Handoff(Billing(), history="summary", message="Transferring you to billing.")


@function_tool
async def identify(ctx: ToolContext, account: str) -> str:
    """Identify the caller by account number and move them to billing."""
    ctx.userdata.account = account
    ctx.handoff(Billing(), history="none")
    return f"Caller identified as {account}."  # this is the tool output
```

`message` is the tool output that the model gets (a returned `(agent, "message")` tuple
sets it as well). It stays in the history, so in `"full"` mode the new agent sees it too.
If a tool raises an exception, times out, or is cancelled, the handoff does not happen.
If several calls in one round ask for a handoff, the last one wins.

Application code (a supervisor, a dashboard button, an escalation rule) can switch agents
too:

```python
await session.handoff(Billing(), history="full")
```

## History carry-over

`history` decides what the new agent sees of the conversation:

| Mode | The new agent sees |
|---|---|
| `"full"` (default) | everything so far: the engine keeps its context |
| `"summary"` | a summary of the earlier turns plus the last 4 items verbatim. The cascade's LLM (or `summary_llm=...`) writes it with `SummarizeHistory`. Without an LLM, it is a compact transcript of the latest messages |
| `"none"` | a fresh start: only the new agent's own `chat_ctx` |
| a callable | whatever it returns: `(ChatContext) -> ChatContext`, sync or async |

```python
from voice_agent_next.engines.rotation import TruncateHistory
from voice_agent_next.session import without_tool_items

Handoff(Billing(), history=without_tool_items)  # messages only, no tool calls
Handoff(Billing(), history=TruncateHistory(max_items=10))  # the last 10 items
```

If the new agent has its own `chat_ctx`, its items come before the carried-over history.
`session.history` itself always keeps the whole call, whatever each agent sees. Recording,
transcripts and analytics do not depend on the mode.

## Shared state: `userdata`

State that must outlive a single agent, such as the caller's name, the order being built,
or authentication, goes in `session.userdata`. Every tool gets it as `ctx.userdata`.
Declare its type once and both are typed:

```python
from dataclasses import dataclass, field


@dataclass
class CallState:
    account: str | None = None
    refunds: list[float] = field(default_factory=list)


session: AgentSession[CallState] = AgentSession("openai/gpt-realtime", userdata=CallState())


@function_tool
async def refund(ctx: ToolContext[CallState], amount: float) -> str:
    """Refund an amount."""
    ctx.userdata.refunds.append(amount)  # typed as CallState
    return "done"
```

## Conversation flows

For conversations that follow fixed steps, `Flow` is a small state machine on top of
handoffs, a minimal take on Pipecat Flows. Each `FlowNode` has instructions, tools and
`Transition`s. A transition is a tool the model calls to move to another node. Its
optional `handler` validates and stores what the step collected:

```python
from voice_agent_next import Flow, FlowNode, ToolContext, Transition
from voice_agent_next.errors import ToolError


async def set_party_size(ctx: ToolContext[Booking], people: int) -> str:
    """Record how many people are coming."""
    if people > 10:
        raise ToolError("We only take tables of up to 10 people.")  # stay on this node
    ctx.userdata.people = people
    return f"Party of {people} recorded."  # the tool output


flow = Flow(
    [
        FlowNode(
            "greet",
            "Greet the caller and ask what they need.",
            greeting="Welcome to Trattoria Roma!",
            transitions=[Transition("collect", "The caller wants to book a table.")],
        ),
        FlowNode(
            "collect",
            "Ask how many people are coming.",
            transitions=[Transition("confirm", handler=set_party_size)],
        ),
        FlowNode("confirm", "Read the booking back and say goodbye.", history="summary"),
    ],
    role="You are the booking assistant of Trattoria Roma. Keep replies short.",
)
await session.run(flow.agent(), transport)
print(flow.path)  # ['greet', 'collect', 'confirm']
```

* Each node runs as its own agent (`FlowAgent`, named after the node), so events, metrics
  and traces show which step the call is in. `flow.current` and `flow.path` track it.
* `role` is put before every node's instructions (persona and style). The node's
  instructions describe the task of that step.
* A transition without a handler is a `go_to_<node>` tool whose `description` tells the
  model when to take it. With a handler, the handler's name, parameters and docstring
  define the tool. Raising `ToolError` keeps the conversation on the current node, and
  the model gets the error. A returned string becomes the tool output.
* Per node you can set `tools`, `greeting`, `voice`, `history` (the carry-over when the
  node is entered), `respond` and `on_enter(session)`.
* A node without transitions is final. To hang up, add a tool that calls
  `session.aclose()`.

## Engine support

Instructions and tools can change mid-session on every engine (`EngineConnection.update`).
Voice and context changes depend on the engine. When the engine cannot apply one, the
handoff still happens, the missing part is listed in `AgentHandoff.unsupported`, and the
session logs it:

| Engine | Instructions and tools | Voice (`update_voice`) | Context (`update_chat_ctx`) |
|---|---|---|---|
| Cascade | yes, from the next reply | yes, from the next reply | yes |
| OpenAI Realtime | yes (`session.update`) | no: the voice is fixed once the model has spoken, so the old voice goes on | no: the model keeps the full context |
| Gemini Live | yes (the session rotates at the next quiet moment) | no | no |
| Mock engine | yes | yes (`MockEngine(voice_updates=False)` simulates no) | yes (`chat_ctx_updates=False`) |

On native engines that keep their context, `"summary"`, `"none"` and custom filters fall
back to `"full"`. Your own engine opts in by overriding
`EngineConnection.update_voice(voice) -> bool` and `update_chat_ctx(ctx) -> bool`. The
default implementations return `False`.

## Timing and interruptions

The handoff happens when the tool round ends: all calls of the response have finished,
and the old agent has finished generating what it was saying, such as "Let me transfer
you". The new agent never talks over the old one.

If the user interrupts the old agent's last response, the handoff still happens (the
tool has run), but the new agent does not speak on its own. It answers the user's next
turn. Two engine differences:

* Native engines report a tool call as soon as it is generated, usually while the
  announcement is still playing, so the transfer is applied before the user finishes
  talking. If the engine answers the user's new turn before the tool finishes (a very
  slow transfer tool), that answer still comes from the previous agent.
* The cascade reports a response's tool calls only after the response has been spoken.
  If the user interrupts "Let me transfer you" before the tool call is reported, the call
  is dropped with the rest of the response. There is no handoff, and the current agent
  answers.

Handoffs from non-blocking tools (`function_tool(blocking=False)`) wait for a quiet
moment. The result is added silently, and then the new agent takes over.

## Observability

* The `agent_handoff` event (`AgentHandoff`) has `from_agent`, `to_agent`, `history`, the
  requesting `call` (`None` for `session.handoff()`), `voice_changed`, `unsupported` and
  `duration`.
* `TurnMetrics.agent` is the name of the agent that answered the turn.
* The recording timeline (`record=...`) logs `agent_handoff`.
* Tracing (`trace=...`) adds an `agent_handoff` span with `voice_agent.handoff.*`
  attributes and `gen_ai.agent.name`. The root span carries the first agent's name.

`examples/11_handoffs.py` in the [example gallery](../examples.md) runs a front desk that
transfers the caller to billing, and a booking flow. Both run offline with `--mock`.
