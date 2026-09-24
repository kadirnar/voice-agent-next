"""The :class:`Agent`: instructions, tools and lifecycle hooks of a voice agent."""

from __future__ import annotations

import dataclasses
import inspect
import types
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from ..chat import ChatContext, ChatMessage
from ..tools import FunctionTool, function_tool

if TYPE_CHECKING:
    from .session import AgentSession

__all__ = ["DEFAULT_INSTRUCTIONS", "Agent"]

DEFAULT_INSTRUCTIONS = (
    "You are a helpful, friendly voice assistant. Your replies are spoken aloud, so keep "
    "them short and conversational, avoid lists, markdown, emojis and special characters, "
    "and ask a clarifying question when the request is ambiguous."
)


class Agent:
    """Describes *what* the voice agent does (the engine/session decide *how*).

    Tools can be passed explicitly or declared as ``@function_tool`` methods on a
    subclass::

        class Concierge(Agent):
            def __init__(self) -> None:
                super().__init__(instructions="You are a hotel concierge.")

            @function_tool
            async def book_table(self, restaurant: str, people: int) -> str:
                \"\"\"Book a restaurant table.\"\"\"
                return "booked"

    Args:
        instructions: system prompt.
        tools: function tools (plain callables are wrapped with ``function_tool``).
        greeting: text spoken verbatim when the session starts (optional).
        voice / language: forwarded to the engine.
        chat_ctx: initial conversation history.
    """

    def __init__(
        self,
        instructions: str = DEFAULT_INSTRUCTIONS,
        *,
        tools: Sequence[FunctionTool | Callable[..., Any]] = (),
        greeting: str | None = None,
        name: str = "assistant",
        voice: str | None = None,
        language: str | None = None,
        chat_ctx: ChatContext | None = None,
    ) -> None:
        self.instructions = instructions
        self.greeting = greeting
        self.name = name
        self.voice = voice
        self.language = language
        self.chat_ctx = chat_ctx
        explicit = [t if isinstance(t, FunctionTool) else function_tool(t) for t in tools]
        self.tools: list[FunctionTool] = explicit + self._method_tools()

    def _method_tools(self) -> list[FunctionTool]:
        tools: list[FunctionTool] = []
        seen: set[str] = set()
        for klass in type(self).__mro__:
            for attr_name, attr in vars(klass).items():
                if attr_name in seen or not isinstance(attr, FunctionTool) or attr.fn is None:
                    continue
                seen.add(attr_name)
                params = list(inspect.signature(attr.fn).parameters)
                fn = attr.fn
                if params and params[0] == "self":
                    fn = types.MethodType(attr.fn, self)
                tools.append(dataclasses.replace(attr, fn=fn))
        return tools

    # ------------------------------------------------------------------- hooks
    async def on_enter(self, session: AgentSession) -> None:
        """Called once the session is connected (before the greeting)."""

    async def on_exit(self, session: AgentSession) -> None:
        """Called when the session closes."""

    async def on_user_turn_completed(self, session: AgentSession, message: ChatMessage) -> None:
        """Called with the final transcript of each user turn."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, tools={[t.name for t in self.tools]})"
