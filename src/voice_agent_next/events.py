"""Events emitted by speech-to-speech engines (:class:`~voice_agent_next.engine.EngineConnection`).

The same event vocabulary is produced by native speech-to-speech models (OpenAI
Realtime, Gemini Live, Moshi, ...) and by the cascaded pipeline
(:class:`~voice_agent_next.engines.cascade.CascadeEngine`), so the session and
applications never need to know which kind of engine is running.

Lifecycle of one exchange::

    InputSpeechStarted -> InputTranscript(partial)* -> InputSpeechStopped
      -> InputCommitted -> InputTranscript(final)
      -> ResponseStarted -> (ResponseAudio | ResponseText | ResponseToolCall)* -> ResponseDone
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from .audio.frame import AudioFrame
from .chat import FunctionCall
from .utils.clock import now

__all__ = [
    "EngineErrorEvent",
    "EngineEvent",
    "EngineStatus",
    "EngineStatusKind",
    "EngineUsage",
    "InputCommitted",
    "InputSpeechStarted",
    "InputSpeechStopped",
    "InputTranscript",
    "ResponseAudio",
    "ResponseDone",
    "ResponseStarted",
    "ResponseStatus",
    "ResponseText",
    "ResponseToolCall",
    "ToolCallCancelled",
]


@dataclass(slots=True, kw_only=True)
class InputSpeechStarted:
    """The engine detected that the user started speaking (barge-in trigger)."""

    audio_time: float | None = None
    """Position in the input audio stream (seconds), when known."""
    timestamp: float = field(default_factory=now)
    type: Literal["input_speech_started"] = "input_speech_started"


@dataclass(slots=True, kw_only=True)
class InputSpeechStopped:
    """The engine detected that the user stopped speaking (not necessarily end of turn)."""

    audio_time: float | None = None
    """Position in the input stream (seconds) where speech *ended* (not where the
    trailing silence was confirmed) — used for voice-to-voice latency."""
    timestamp: float = field(default_factory=now)
    type: Literal["input_speech_stopped"] = "input_speech_stopped"


@dataclass(slots=True, kw_only=True)
class InputTranscript:
    """Transcript of user speech. ``is_final=False`` for interim/partial results."""

    item_id: str
    text: str
    is_final: bool
    language: str | None = None
    timestamp: float = field(default_factory=now)
    type: Literal["input_transcript"] = "input_transcript"


@dataclass(slots=True, kw_only=True)
class InputCommitted:
    """The user's turn was committed (end of turn decided); a response normally follows."""

    item_id: str
    timestamp: float = field(default_factory=now)
    type: Literal["input_committed"] = "input_committed"


@dataclass(slots=True, kw_only=True)
class ResponseStarted:
    response_id: str
    timestamp: float = field(default_factory=now)
    type: Literal["response_started"] = "response_started"


@dataclass(slots=True, kw_only=True)
class ResponseAudio:
    """A chunk of agent audio at the engine's ``output_sample_rate``."""

    response_id: str
    item_id: str
    frame: AudioFrame
    timestamp: float = field(default_factory=now)
    type: Literal["response_audio"] = "response_audio"


@dataclass(slots=True, kw_only=True)
class ResponseText:
    """Text delta of the agent's reply (the transcript of what is being spoken)."""

    response_id: str
    item_id: str
    delta: str
    timestamp: float = field(default_factory=now)
    type: Literal["response_text"] = "response_text"


@dataclass(slots=True, kw_only=True)
class ResponseToolCall:
    """The model requests a tool call. Answer with ``EngineConnection.send_tool_output``."""

    response_id: str
    call: FunctionCall
    timestamp: float = field(default_factory=now)
    type: Literal["response_tool_call"] = "response_tool_call"


@dataclass(slots=True)
class EngineUsage:
    input_text_tokens: int = 0
    input_audio_tokens: int = 0
    output_text_tokens: int = 0
    output_audio_tokens: int = 0
    cached_tokens: int = 0


ResponseStatus: TypeAlias = Literal["completed", "cancelled", "incomplete", "failed"]


@dataclass(slots=True, kw_only=True)
class ResponseDone:
    response_id: str
    status: ResponseStatus = "completed"
    usage: EngineUsage | None = None
    error: str | None = None
    timestamp: float = field(default_factory=now)
    type: Literal["response_done"] = "response_done"


@dataclass(slots=True, kw_only=True)
class ToolCallCancelled:
    """The engine withdrew tool calls it requested earlier (e.g. Gemini Live after barge-in)."""

    call_ids: list[str]
    timestamp: float = field(default_factory=now)
    type: Literal["tool_call_cancelled"] = "tool_call_cancelled"


EngineStatusKind: TypeAlias = Literal["reconnecting", "reconnected", "expiring", "resumed"]


@dataclass(slots=True, kw_only=True)
class EngineStatus:
    """Connection lifecycle notices (session rotation, reconnects, provider GoAway)."""

    status: EngineStatusKind
    detail: str | None = None
    time_left: float | None = None
    timestamp: float = field(default_factory=now)
    type: Literal["engine_status"] = "engine_status"


@dataclass(slots=True, kw_only=True)
class EngineErrorEvent:
    """A provider/engine error. If not ``recoverable`` the connection is unusable."""

    error: Exception
    recoverable: bool = True
    timestamp: float = field(default_factory=now)
    type: Literal["error"] = "error"


EngineEvent: TypeAlias = (
    InputSpeechStarted
    | InputSpeechStopped
    | InputTranscript
    | InputCommitted
    | ResponseStarted
    | ResponseAudio
    | ResponseText
    | ResponseToolCall
    | ResponseDone
    | ToolCallCancelled
    | EngineStatus
    | EngineErrorEvent
)
