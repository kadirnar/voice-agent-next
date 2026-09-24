"""Typed payloads of the events emitted by :class:`~voice_agent_next.session.AgentSession`.

============================  ============================================
event name                    payload
============================  ============================================
``agent_state_changed``       :class:`AgentStateChanged`
``user_state_changed``        :class:`UserStateChanged`
``user_transcript``           :class:`UserTranscript`
``agent_transcript``          :class:`AgentTranscript`
``conversation_item``         :class:`ConversationItemAdded`
``tool_call``                 :class:`ToolCalled`
``tool_result``               :class:`ToolResult`
``interrupted``               :class:`Interrupted`
``agent_false_interruption``  :class:`AgentFalseInterruption`
``metrics``                   any :data:`~voice_agent_next.metrics.Metrics`
``error``                     :class:`SessionError`
``close``                     :class:`SessionClosed`
============================  ============================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, TypeAlias

from ..chat import ChatItem, FunctionCall, FunctionCallOutput
from ..utils.clock import now

__all__ = [
    "AgentFalseInterruption",
    "AgentState",
    "AgentStateChanged",
    "AgentTranscript",
    "ConversationItemAdded",
    "FalseInterruptionReason",
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


FalseInterruptionReason: TypeAlias = Literal["noise", "backchannel", "too_few_words"]
"""``noise``: nothing was transcribed (cough, noise, echo); ``backchannel``: only words such
as "uh-huh" or "okay"; ``too_few_words``: fewer than ``min_interruption_words`` words."""


@dataclass(slots=True)
class AgentFalseInterruption:
    """The user spoke over the agent, but it was not a real interruption.

    Emitted when the paused agent speech resumes (``resumed=True``), or when an
    interruption that already stopped the agent — on duration, or cancelled by the engine
    itself — turns out to have been noise or a backchannel (``resumed=False``: nothing
    could be resumed; the app may continue with ``generate_reply()``).
    """

    resumed: bool
    reason: FalseInterruptionReason
    response_id: str
    item_id: str | None
    transcript: str
    """What was transcribed while the user spoke ("" if nothing)."""
    speech_duration: float
    """Seconds of user speech."""
    paused: float
    """Seconds the agent's audio was paused before it resumed (0.0 if it did not resume)."""
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
