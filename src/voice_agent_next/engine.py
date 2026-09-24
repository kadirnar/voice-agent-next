"""Speech-to-speech engine interface.

An :class:`S2SEngine` is a (reusable) factory; :meth:`S2SEngine.connect` opens a live
:class:`EngineConnection` that consumes user audio and produces agent audio plus the
events in :mod:`voice_agent_next.events`.

Two families implement it:

* **native** speech-to-speech models (OpenAI Realtime, Gemini Live, Nova Sonic, Moshi,
  Qwen-Omni, ...) — ``capabilities.native_audio = True``;
* the **cascade** (VAD -> STT -> LLM -> TTS, optionally with an audio-input LLM) in
  :mod:`voice_agent_next.engines.cascade` — ``native_audio = False``.

The :class:`~voice_agent_next.session.AgentSession` only talks to this interface, so
interruption handling, tool execution, transcripts and metrics work identically for
every engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from .audio.frame import AudioFrame
from .audio.resample import StreamResampler
from .chat import ChatContext, FunctionCallOutput
from .events import EngineEvent
from .tools import FunctionTool
from .utils.aio import Chan
from .utils.clock import now
from .utils.emitter import EventEmitter

__all__ = ["EngineCapabilities", "EngineConnection", "EngineOptions", "S2SEngine"]


@dataclass(frozen=True, slots=True)
class EngineCapabilities:
    native_audio: bool = True
    """End-to-end audio model (False for cascades)."""
    server_turn_detection: bool = True
    """The engine detects user turns itself (VAD / semantic turn detection)."""
    tool_calling: bool = True
    input_transcription: bool = True
    output_transcription: bool = True
    truncation: bool = False
    """Supports truncating the assistant item to what the user actually heard."""
    full_duplex: bool = False
    """Listens and speaks simultaneously (backchannels, overlaps) — e.g. Moshi."""
    text_input: bool = True
    """Accepts injected text messages (:meth:`EngineConnection.send_text`)."""
    tool_mode: Literal["blocking", "non_blocking", "delegation"] = "blocking"
    """How tool calls interact with speech: the model waits for results (blocking), keeps
    talking (non_blocking, e.g. Gemini Live), or delegates to a backend (GPT-Live)."""
    max_session_duration: float | None = None
    """Hard provider session/connection limit in seconds (engines rotate transparently)."""


@dataclass(slots=True)
class EngineOptions:
    """Per-connection options passed to :meth:`S2SEngine.connect`."""

    instructions: str = ""
    tools: list[FunctionTool] = field(default_factory=list)
    chat_ctx: ChatContext | None = None
    """Initial conversation history to seed the engine with."""
    voice: str | None = None
    language: str | None = None
    temperature: float | None = None
    turn_detection: bool = True
    """If False the engine never ends user turns by itself; call ``commit_input()``."""
    extra: dict[str, Any] = field(default_factory=dict)
    """Provider-specific options."""


class S2SEngine(ABC, EventEmitter):
    """Factory for engine connections. Emits ``"metrics"`` from its connections."""

    provider: ClassVar[str] = "unknown"

    def __init__(
        self,
        *,
        model: str,
        capabilities: EngineCapabilities,
        input_sample_rate: int,
        output_sample_rate: int,
    ) -> None:
        EventEmitter.__init__(self)
        self.model = model
        self.capabilities = capabilities
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate

    @abstractmethod
    async def connect(self, options: EngineOptions) -> EngineConnection:
        """Open a live connection/session."""

    async def warmup(self) -> None:
        """Pre-load models / pre-open connections (optional)."""

    async def aclose(self) -> None:
        """Release shared resources."""

    async def __aenter__(self) -> S2SEngine:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


class EngineConnection(ABC):
    """A live speech-to-speech session.

    Implementations push events with :meth:`_emit` and implement the abstract
    control methods. :meth:`send_audio` resamples input to ``input_sample_rate``
    mono before calling :meth:`_send_audio`.
    """

    def __init__(self, engine: S2SEngine, options: EngineOptions) -> None:
        self.engine = engine
        self.options = options
        self._events: Chan[EngineEvent] = Chan()
        self._in_resampler = StreamResampler(engine.input_sample_rate, 1)
        self._closed = False
        self._audio_sent = 0.0
        # (input audio position at end of frame, capture wall-clock time at end of frame)
        self._audio_clock: deque[tuple[float, float]] = deque(maxlen=6000)

    @property
    def capabilities(self) -> EngineCapabilities:
        return self.engine.capabilities

    @property
    def input_sample_rate(self) -> int:
        return self.engine.input_sample_rate

    @property
    def output_sample_rate(self) -> int:
        return self.engine.output_sample_rate

    @property
    def closed(self) -> bool:
        return self._closed

    # ------------------------------------------------------------------ audio in
    async def send_audio(self, frame: AudioFrame) -> None:
        """Send user audio (any sample rate / channel count)."""
        if self._closed:
            return
        out = self._in_resampler.push(frame)
        if out:
            self._audio_sent += out.duration
            captured = frame.timestamp + frame.duration if frame.timestamp is not None else now()
            self._audio_clock.append((self._audio_sent, captured))
            await self._send_audio(out)

    @property
    def input_audio_time(self) -> float:
        """Seconds of user audio sent so far (the engine's input stream position)."""
        return self._audio_sent

    def audio_time_to_wall(self, audio_time: float) -> float | None:
        """Map an input-stream position (seconds) to the ``now()`` time it was captured.

        Engines report speech boundaries as stream positions (e.g. OpenAI's
        ``audio_end_ms``); this converts them to wall-clock for latency metrics.
        """
        if not self._audio_clock:
            return None
        for audio_end, wall_end in reversed(self._audio_clock):
            if audio_end <= audio_time:
                return wall_end + (audio_time - audio_end)
        audio_end, wall_end = self._audio_clock[0]
        return wall_end - (audio_end - audio_time)

    @abstractmethod
    async def _send_audio(self, frame: AudioFrame) -> None:
        """Send audio already at ``input_sample_rate`` mono."""

    @abstractmethod
    async def commit_input(self) -> None:
        """Manually end the user's turn (when ``turn_detection`` is off)."""

    @abstractmethod
    async def clear_input(self) -> None:
        """Discard buffered, uncommitted user audio."""

    # ------------------------------------------------------------------ control
    @abstractmethod
    async def send_text(self, text: str, *, respond: bool = True) -> None:
        """Add a user text message; trigger a response if ``respond``."""

    @abstractmethod
    async def create_response(self, *, instructions: str | None = None) -> None:
        """Ask the model to respond now (e.g. greeting, or after tool outputs)."""

    @abstractmethod
    async def cancel_response(self) -> None:
        """Stop generating the current response (no-op if none)."""

    async def truncate(self, item_id: str, audio_end_ms: int) -> str | None:
        """Tell the engine the user only heard ``audio_end_ms`` of ``item_id``.

        Returns the transcript of what was heard when the engine knows it (the session
        otherwise estimates it). Default: no-op (engines without ``truncation``).
        """
        return None

    async def interrupt(
        self, item_id: str | None = None, played_ms: int | None = None
    ) -> str | None:
        """Barge-in: cancel the response and truncate the context to what was played.

        Returns the heard transcript if known (see :meth:`truncate`).
        """
        await self.cancel_response()
        if self.capabilities.truncation and item_id is not None and played_ms is not None:
            return await self.truncate(item_id, played_ms)
        return None

    @abstractmethod
    async def send_tool_output(self, output: FunctionCallOutput, *, respond: bool = True) -> None:
        """Return a tool result; trigger a response if ``respond``."""

    @abstractmethod
    async def update(
        self, *, instructions: str | None = None, tools: list[FunctionTool] | None = None
    ) -> None:
        """Update instructions and/or tools mid-session."""

    async def say(self, text: str) -> None:
        """Speak ``text``. Engines with direct TTS access override this to be verbatim."""
        await self.create_response(
            instructions=f'Say exactly the following, verbatim, and nothing else: "{text}"'
        )

    # ------------------------------------------------------------------- events
    def _emit(self, event: EngineEvent) -> None:
        if not self._events.closed:
            self._events.send_nowait(event)

    def events(self) -> AsyncIterator[EngineEvent]:
        """Async iterator over engine events; ends when the connection closes."""
        return self._events.__aiter__()

    async def aclose(self) -> None:
        """Close the connection. Subclasses should call ``await super().aclose()``."""
        self._closed = True
        self._events.close()

    async def __aenter__(self) -> EngineConnection:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
