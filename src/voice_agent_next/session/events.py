"""Typed payloads of the events emitted by :class:`~voice_agent_next.session.AgentSession`.

=========================  ===============================================
event name                 payload
=========================  ===============================================
``agent_state_changed``    :class:`AgentStateChanged`
``user_state_changed``     :class:`UserStateChanged`
``user_transcript``        :class:`UserTranscript`
``agent_transcript``       :class:`AgentTranscript`
``conversation_item``      :class:`ConversationItemAdded`
``tool_call``              :class:`ToolCalled`
``tool_result``            :class:`ToolResult`
``interrupted``            :class:`Interrupted`
``metrics``                any :data:`~voice_agent_next.metrics.Metrics`
``error``                  :class:`SessionError`
``close``                  :class:`SessionClosed`
=========================  ===============================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ..chat import ChatItem, FunctionCall, FunctionCallOutput
from ..utils.clock import now

__all__ = [
    "AgentState",
    "AgentStateChanged",
    "AgentTranscript",
    "ConversationItemAdded",
    "Interrupted",
    "SessionClosed",
    "SessionError",
    "ToolCalled",
    "ToolResult",
    "UserState",
    "UserStateChanged",
    "UserTranscript",
]


class AgentState(StrEnum):
    INITIALIZING = "initializing"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    CLOSED = "closed"


class UserState(StrEnum):
    LISTENING = "listening"
    SPEAKING = "speaking"


@dataclass(slots=True)
class AgentStateChanged:
    old_state: AgentState
    new_state: AgentState
    timestamp: float = field(default_factory=now)


@dataclass(slots=True)
class UserStateChanged:
    old_state: UserState
    new_state: UserState
    timestamp: float = field(default_factory=now)


@dataclass(slots=True)
class UserTranscript:
    text: str
    is_final: bool
    item_id: str
    language: str | None = None
    timestamp: float = field(default_factory=now)


@dataclass(slots=True)
class AgentTranscript:
    delta: str
    item_id: str
    response_id: str
    timestamp: float = field(default_factory=now)


@dataclass(slots=True)
class ConversationItemAdded:
    item: ChatItem
    timestamp: float = field(default_factory=now)


@dataclass(slots=True)
class ToolCalled:
    call: FunctionCall
    timestamp: float = field(default_factory=now)


@dataclass(slots=True)
class ToolResult:
    call: FunctionCall
    output: FunctionCallOutput
    duration: float
    timestamp: float = field(default_factory=now)


@dataclass(slots=True)
class Interrupted:
    response_id: str
    item_id: str | None
    played: float
    """Seconds of the agent's audio the user actually heard."""
    timestamp: float = field(default_factory=now)


@dataclass(slots=True)
class SessionError:
    error: Exception
    recoverable: bool
    timestamp: float = field(default_factory=now)


@dataclass(slots=True)
class SessionClosed:
    reason: str
    timestamp: float = field(default_factory=now)
