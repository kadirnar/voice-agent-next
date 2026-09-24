"""Conversation flows: a small declarative state machine built on agent handoffs.

A :class:`Flow` is a graph of :class:`FlowNode` steps. Each node has its own
instructions and tools, and :class:`Transition` edges that the model takes by calling a
tool (``go_to_<node>`` or a named handler that validates and stores what was collected).
Taking a transition hands the conversation to the next node's agent — instructions, tools,
voice and history carry-over switch exactly as with any handoff.

Example::

    flow = Flow(
        [
            FlowNode("greet", "Greet the caller and ask how many people are coming.",
                     transitions=[Transition("time", handler=set_party_size)]),
            FlowNode("time", "Ask for the time of the reservation.",
                     transitions=[Transition("confirm", handler=set_time)]),
            FlowNode("confirm", "Read the reservation back and say goodbye."),
        ],
        role="You are the booking assistant of Trattoria Roma. Keep replies short.",
    )
    await session.run(flow.agent(), transport)

Minimal on purpose (compare Pipecat Flows): no actions or context strategies beyond
:data:`~voice_agent_next.session.handoff.HistoryMode`; use ``on_enter`` and tool handlers
for side effects, and ``session.userdata`` for the collected state.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..tools import FunctionTool, function_tool
from .agent import Agent
from .handoff import Handoff, HistoryMode

if TYPE_CHECKING:
    from .session import AgentSession

__all__ = ["Flow", "FlowAgent", "FlowNode", "Transition"]


@dataclass
class Transition:
    """An edge to the node ``to``, taken when the model calls its tool.

    Attributes:
        to: name of the target node.
        description: when to take it (the tool description the model sees). Default:
            the handler's docstring, else "Move the conversation to the <to> step."
        name: tool name (default: the handler's name, else ``go_to_<to>``).
        handler: optional function (sync or async, or a ``@function_tool``) run before
            moving on. Its parameters
            are the tool's arguments (collect data with them; declare a ``ToolContext``
            parameter for ``userdata``). Raise :class:`~voice_agent_next.errors.ToolError`
            to stay on the current node (the model gets the error). A returned string
            becomes the tool output.
        message: tool output when the handler returns none (default "Transferred to
            <to>.").
    """

    to: str
    description: str = ""
    name: str | None = None
    handler: Callable[..., Any] | None = None
    message: str | None = None

    @property
    def tool_name(self) -> str:
        if self.name:
            return self.name
        if isinstance(self.handler, FunctionTool):
            return self.handler.name
        if self.handler is not None:
            return str(getattr(self.handler, "__name__", f"go_to_{self.to}"))
        return f"go_to_{self.to}"


@dataclass
class FlowNode:
    """One step of a :class:`Flow`.

    Attributes:
        name: unique node name (also the agent name in events and metrics).
        instructions: the task of this step (``Flow.role`` is put before it).
        tools: extra tools available in this step.
        transitions: where the conversation can go from here; none = a final node.
        greeting: said verbatim when the node is entered.
        voice / language: forwarded to the engine (voice changes need engine support).
        history: what this node sees of the conversation when entered.
        respond: speak right away when entered (greeting or generated reply).
        on_enter: called with the session when the node is entered (sync or async).
    """

    name: str
    instructions: str
    tools: Sequence[FunctionTool | Callable[..., Any]] = ()
    transitions: Sequence[Transition] = ()
    greeting: str | None = None
    voice: str | None = None
    language: str | None = None
    history: HistoryMode = "full"
    respond: bool = True
    on_enter: Callable[[AgentSession], Any] | None = None


class FlowAgent(Agent):
    """The agent that runs one :class:`FlowNode` (created by :class:`Flow`)."""

    def __init__(self, flow: Flow, node: FlowNode) -> None:
        instructions = f"{flow.role}\n\n{node.instructions}" if flow.role else node.instructions
        super().__init__(
            instructions,
            tools=[*node.tools, *(flow._transition_tool(t) for t in node.transitions)],
            greeting=node.greeting,
            name=node.name,
            voice=node.voice,
            language=node.language,
        )
        self.flow = flow
        self.node = node

    async def on_enter(self, session: AgentSession) -> None:
        self.flow._entered(self.node)
        if self.node.on_enter is not None:
            result = self.node.on_enter(session)
            if inspect.isawaitable(result):
                await result


@dataclass
class Flow:
    """A conversation as a graph of nodes (see the module docs).

    Attributes:
        nodes: the steps; names must be unique and every transition must point to one.
        initial: the first node (default: the first of ``nodes``).
        role: instructions shared by every node (persona, style), put before each node's.
        path: names of the nodes entered so far, in order (read-only).
    """

    nodes: Sequence[FlowNode]
    initial: str | None = None
    role: str = ""
    path: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("a flow needs at least one node")
        self._nodes: dict[str, FlowNode] = {}
        for node in self.nodes:
            if node.name in self._nodes:
                raise ValueError(f"duplicate flow node {node.name!r}")
            self._nodes[node.name] = node
        if self.initial is None:
            self.initial = self.nodes[0].name
        elif self.initial not in self._nodes:
            raise ValueError(f"unknown initial node {self.initial!r}")
        for node in self.nodes:
            names = [getattr(t, "name", None) or getattr(t, "__name__", "") for t in node.tools]
            for t in node.transitions:
                if t.to not in self._nodes:
                    raise ValueError(f"node {node.name!r}: transition to unknown node {t.to!r}")
                names.append(t.tool_name)
            dupes = sorted({n for n in names if n and names.count(n) > 1})
            if dupes:
                raise ValueError(f"node {node.name!r}: duplicate tool names {dupes}")

    @property
    def current(self) -> str | None:
        """The node the conversation is in (``None`` before the flow starts)."""
        return self.path[-1] if self.path else None

    def node(self, name: str) -> FlowNode:
        try:
            return self._nodes[name]
        except KeyError:
            raise KeyError(f"unknown flow node {name!r}") from None

    def agent(self, name: str | None = None) -> FlowAgent:
        """A fresh agent for node ``name`` (default: the initial node): pass it to
        ``AgentSession.start``/``run`` to start the flow."""
        target = name if name is not None else self.initial
        assert target is not None
        return FlowAgent(self, self.node(target))

    # ----------------------------------------------------------------- internals
    def _entered(self, node: FlowNode) -> None:
        self.path.append(node.name)

    def _transition_tool(self, transition: Transition) -> FunctionTool:
        description = transition.description or None
        if transition.handler is None:

            def go() -> None:
                return None

            default = f"Move the conversation to the {transition.to} step."
            base = function_tool(name=transition.tool_name, description=description or default)(go)
            handler: Callable[..., Any] | None = None
        elif isinstance(transition.handler, FunctionTool):  # already a tool: keep its schema
            base = dataclasses.replace(
                transition.handler,
                name=transition.tool_name,
                description=description or transition.handler.description,
            )
            handler = transition.handler.fn
        else:
            base = function_tool(name=transition.tool_name, description=description)(
                transition.handler
            )
            handler = transition.handler

        async def run(**kwargs: Any) -> Handoff:
            result: Any = None
            if handler is not None:
                result = handler(**kwargs)
                if inspect.isawaitable(result):
                    result = await result
            target = self.node(transition.to)
            message = result if isinstance(result, str) and result else transition.message
            return Handoff(
                FlowAgent(self, target),
                history=target.history,
                message=message,
                respond=target.respond,
            )

        return dataclasses.replace(base, fn=run)
