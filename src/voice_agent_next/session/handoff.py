"""Multi-agent handoffs: switch the active :class:`~voice_agent_next.session.Agent`
mid-session, with rules for what the new agent sees of the conversation.

A tool hands the conversation over by returning another agent (or a :class:`Handoff`, or
calling :meth:`ToolContext.handoff <voice_agent_next.tools.ToolContext.handoff>`)::

    class Billing(Agent): ...

    @function_tool
    async def transfer_to_billing() -> Agent:
        \"\"\"Transfer the caller to the billing department.\"\"\"
        return Billing()

The session then switches instructions, tools and (when the engine can) voice, carries the
history over according to :data:`HistoryMode`, runs the old agent's ``on_exit`` and the new
one's ``on_enter``, and lets the new agent speak. See ``docs/concepts/handoffs.md``.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from ..chat import ChatContext, ChatItem, ChatMessage, FunctionCall, FunctionCallOutput

if TYPE_CHECKING:
    from ..llm import LLM
    from .agent import Agent

__all__ = [
    "Handoff",
    "HistoryFilter",
    "HistoryMode",
    "carry_history",
    "history_mode_name",
    "without_tool_items",
]

HistoryFilter: TypeAlias = Callable[[ChatContext], "ChatContext | Awaitable[ChatContext]"]
"""A custom carry-over: gets the conversation so far, returns what the new agent sees
(sync or async). :class:`~voice_agent_next.engines.rotation.TruncateHistory` and
:class:`~voice_agent_next.engines.rotation.SummarizeHistory` instances work too."""

HistoryMode: TypeAlias = Literal["full", "summary", "none"] | HistoryFilter
"""What the new agent sees of the conversation:

* ``"full"`` (default) — everything so far (the engine keeps its context);
* ``"summary"`` — an LLM summary of the earlier turns plus the last few items verbatim
  (without an LLM: a compact transcript of the last messages);
* ``"none"`` — a fresh start (only the new agent's own ``chat_ctx``);
* a callable — a custom filter (:data:`HistoryFilter`).
"""

_SUMMARY_KEEP_LAST = 4
"""Items kept verbatim after the summary (the request that led to the handoff)."""
_DIGEST_MESSAGES = 12
"""Messages in the no-LLM fallback summary."""
_DIGEST_CHARS = 300
"""Longest message in the no-LLM fallback summary."""


@dataclass(eq=False)
class Handoff:
    """A request to hand the conversation to ``agent``.

    Return it from a tool (a bare :class:`Agent` means ``Handoff(agent)``), create it with
    ``ToolContext.handoff(...)`` or pass its fields to ``AgentSession.handoff(...)``.

    Attributes:
        agent: the agent that takes over.
        history: what ``agent`` sees of the conversation (see :data:`HistoryMode`).
        message: the tool output the model gets for the call that requested the handoff
            (default ``"Transferred to <name>."``); it stays in the history, so the new agent
            sees it too.
        respond: the new agent speaks right away (its ``greeting`` if it has one, else a
            generated reply). ``False``: it waits for the user. Ignored when ``on_enter``
            already made it speak.
        summary_llm: the LLM that writes the ``"summary"`` (default: the cascade's LLM).
    """

    agent: Agent
    history: HistoryMode = "full"
    message: str | None = None
    respond: bool = True
    summary_llm: LLM | None = None

    def __post_init__(self) -> None:
        if isinstance(self.history, str) and self.history not in ("full", "summary", "none"):
            raise ValueError(f"unknown history mode {self.history!r}")

    @property
    def output(self) -> str:
        """The tool output for the call that requested this handoff."""
        if self.message is not None:
            return self.message
        return f"Transferred to {self.agent.name}."


def history_mode_name(mode: HistoryMode) -> str:
    """``"full"``, ``"summary"``, ``"none"`` or ``"custom"`` (for events and traces)."""
    return mode if isinstance(mode, str) else "custom"


async def carry_history(
    history: ChatContext, mode: HistoryMode, *, llm: LLM | None = None
) -> ChatContext | None:
    """The context the next agent starts with; ``None`` for ``"full"`` (nothing to change).

    Args:
        history: the conversation so far (the session's history).
        mode: see :data:`HistoryMode`.
        llm: the summarizer for ``"summary"`` (``None``: a compact transcript instead).
    """
    if mode == "full":
        return None
    if mode == "none":
        return ChatContext()
    if mode == "summary":
        return await _summarize(history, llm)
    if not callable(mode):
        raise ValueError(f"unknown history mode {mode!r}")
    result = mode(history.copy())
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, ChatContext):
        raise TypeError(f"the history filter returned {type(result).__name__}, not ChatContext")
    return result


async def _summarize(history: ChatContext, llm: LLM | None) -> ChatContext:
    if llm is not None:
        from ..engines.rotation import SummarizeHistory

        strategy = SummarizeHistory(llm, keep_last=_SUMMARY_KEEP_LAST, min_items=0)
        return await strategy(history)
    return _digest(history)


def _digest(history: ChatContext) -> ChatContext:
    """No LLM at hand: the last messages as one compact system message."""
    lines = []
    for item in history.items:
        if isinstance(item, ChatMessage) and item.role in ("user", "assistant"):
            text = " ".join(item.text.split())
            if text:
                cut = text if len(text) <= _DIGEST_CHARS else text[: _DIGEST_CHARS - 3] + "..."
                lines.append(f"{item.role}: {cut}")
    out = ChatContext()
    if lines:
        transcript = "\n".join(lines[-_DIGEST_MESSAGES:])
        out.add_message(
            "system",
            f"Summary of the conversation so far (latest messages):\n{transcript}",
            metadata={"carry_over": "summary"},
        )
    return out


def without_tool_items(history: ChatContext) -> ChatContext:
    """A :data:`HistoryFilter` that drops tool calls and results (messages only)."""
    items: list[ChatItem] = [
        i for i in history.items if not isinstance(i, FunctionCall | FunctionCallOutput)
    ]
    return ChatContext(items)


def as_handoff(result: Any) -> tuple[Handoff | None, Any]:
    """Split a tool's return value into (handoff, remaining result).

    Accepts an :class:`Agent`, a :class:`Handoff`, or a ``(agent_or_handoff, message)``
    tuple (the message becomes the tool output).
    """
    from .agent import Agent

    if isinstance(result, Agent):
        return Handoff(result), None
    if isinstance(result, Handoff):
        return result, None
    if (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[0], Agent | Handoff)
        and isinstance(result[1], str)
    ):
        target, message = result
        handoff = Handoff(target) if isinstance(target, Agent) else target
        if handoff.message is None:
            handoff.message = message
        return handoff, None
    return None, result
