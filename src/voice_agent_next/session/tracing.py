"""OpenTelemetry tracing of an :class:`~voice_agent_next.session.AgentSession`.

Span tree (one trace per session)::

    session
    ├── turn                       one user turn: speech start -> the agent's reply ended
    │   ├── end_of_turn            user stopped speaking -> turn committed (endpointing)
    │   ├── turn_detection         semantic end-of-turn inference (cascade)
    │   ├── stt                    speech recognition (cascade)
    │   ├── chat {model}           LLM request (cascade): ttft, tokens (GenAI conventions)
    │   ├── tts                    synthesis (cascade): ttfb, characters
    │   ├── response               one engine response: status, usage, ttfb (native S2S)
    │   ├── execute_tool {name}    tool call (GenAI conventions)
    │   └── agent_handoff          the conversation moved to another agent
    └── response                   responses outside a user turn (greeting, say())

Spans are built from the session's own events and metrics, with their real start and end
times. Attributes follow the OpenTelemetry GenAI semantic conventions where they apply
(``gen_ai.operation.name``, ``gen_ai.provider.name``, ``gen_ai.request.model``,
``gen_ai.usage.*``, ``gen_ai.tool.*``, ``gen_ai.conversation.id``); voice-specific values
use the ``voice_agent.`` prefix (``voice_agent.turn.voice_to_voice`` ...). Durations are
in seconds. Transcripts and tool arguments/results are recorded only with
``capture_content=True``.

Only ``opentelemetry-api`` is needed (extra ``otel``); the application configures the SDK
and exporter (OTLP, Jaeger, Langfuse...). Without a configured SDK the API is a no-op. A
session without ``trace=...`` creates no tracer at all.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..events import (
    InputCommitted,
    InputSpeechStarted,
    InputSpeechStopped,
    ResponseDone,
    ResponseStarted,
)
from ..metrics import (
    EngineMetrics,
    EOTMetrics,
    LLMMetrics,
    Metrics,
    STTMetrics,
    TTSMetrics,
    TurnMetrics,
)
from ..utils.clock import now
from ..utils.deps import is_installed, require
from ..utils.ids import new_id
from .taps import SessionTap

if TYPE_CHECKING:
    from ..events import EngineEvent
    from .events import (
        AgentFalseInterruption,
        AgentHandoff,
        AgentTranscript,
        Interrupted,
        SessionError,
        ToolCalled,
        ToolResult,
        UserTranscript,
    )
    from .session import AgentSession

__all__ = ["SessionTracer"]

_SEGMENT_GAP = 2.0
"""User speech starting this long after the last speech stopped begins a new turn."""


@dataclass
class _Turn:
    span: Any
    ctx: Any
    number: int
    user_text: list[str] = field(default_factory=list)


@dataclass
class _Child:
    """A span measured before its turn existed (e.g. STT before the commit)."""

    name: str
    start: float
    end: float
    attributes: dict[str, Any]
    kind: str = "INTERNAL"
    error: str | None = None


class SessionTracer(SessionTap):
    """Exports a session as OpenTelemetry spans (see the module docs).

    Args:
        tracer_provider: the provider to get the tracer from (default: the global one set
            with ``opentelemetry.trace.set_tracer_provider``).
        tracer: use this tracer instead.
        capture_content: record transcripts and tool arguments/results as attributes
            (off by default: they may contain personal data).

    Raises:
        MissingDependencyError: ``opentelemetry-api`` is not installed.
    """

    def __init__(
        self,
        tracer_provider: Any = None,
        *,
        tracer: Any = None,
        capture_content: bool = False,
    ) -> None:
        self._otel = require("opentelemetry.trace", extra="otel", package="opentelemetry-api")
        if tracer is None:
            from .. import __version__

            tracer = self._otel.get_tracer(
                "voice_agent_next", __version__, tracer_provider=tracer_provider
            )
        self._tracer = tracer
        self.capture_content = capture_content
        self.conversation_id = new_id("conv_")
        self._session: AgentSession | None = None
        self._wall_offset = 0.0
        self._root: Any = None
        self._root_ctx: Any = None
        self._turns: deque[_Turn] = deque()
        self._turn_count = 0
        self._pending: list[_Child] = []
        self._responses: dict[str, tuple[Any, list[str]]] = {}
        self._tools: dict[str, tuple[Any, float]] = {}
        self._speech_start: float | None = None
        self._speech_stop: float | None = None
        self._closed = False

    @staticmethod
    def available() -> bool:
        """True if ``opentelemetry-api`` is installed."""
        return is_installed("opentelemetry.trace")

    def attach(self, session: AgentSession) -> SessionTracer:
        """Trace ``session`` (call before it starts)."""
        if self._session is not None:
            raise RuntimeError("a SessionTracer traces one session")
        self._session = session
        session.add_tap(self)
        session.on("metrics", self._on_metrics)
        session.on("tool_call", self._on_tool_call)
        session.on("tool_result", self._on_tool_result)
        session.on("interrupted", self._on_interrupted)
        session.on("agent_false_interruption", self._on_false_interruption)
        session.on("error", self._on_error)
        session.on("agent_handoff", self._on_agent_handoff)
        if self.capture_content:
            session.on("user_transcript", self._on_user_transcript)
            session.on("agent_transcript", self._on_agent_transcript)
        return self

    # -------------------------------------------------------------- span helpers
    def _ns(self, t: float) -> int:
        return int((t + self._wall_offset) * 1e9)

    def _parent(self) -> Any:
        return self._turns[-1].ctx if self._turns else self._root_ctx

    def _start(
        self, name: str, start: float, attributes: dict[str, Any], *, parent: Any = None,
        kind: str = "INTERNAL",
    ) -> Any:  # fmt: skip
        return self._tracer.start_span(
            name,
            context=parent if parent is not None else self._parent(),
            kind=getattr(self._otel.SpanKind, kind),
            attributes=_clean(attributes),
            start_time=self._ns(start),
        )

    def _span(self, child: _Child, *, parent: Any = None) -> None:
        span = self._start(
            child.name, child.start, child.attributes, parent=parent, kind=child.kind
        )
        if child.error:
            self._fail(span, child.error)
        span.end(end_time=self._ns(max(child.end, child.start)))

    def _child(self, child: _Child, *, needs_turn: bool = False) -> None:
        if self._closed:
            return
        if needs_turn and not self._turns:
            self._pending.append(child)  # parented once the turn is committed
        else:
            self._span(child)

    def _fail(self, span: Any, error: str) -> None:
        span.set_status(self._otel.Status(self._otel.StatusCode.ERROR, error))
        span.set_attribute("error.type", error.split(":", 1)[0][:64] or "error")

    # --------------------------------------------------------------------- hooks
    def session_started(self, session: AgentSession, t: float) -> None:
        self._wall_offset = time.time() - now()
        engine = session.engine
        self._root = self._tracer.start_span(
            "session",
            kind=self._otel.SpanKind.INTERNAL,
            attributes=_clean(
                {
                    "gen_ai.conversation.id": self.conversation_id,
                    "gen_ai.provider.name": engine.provider,
                    "gen_ai.request.model": engine.model,
                    "voice_agent.engine": type(engine).__name__,
                    "voice_agent.transport": type(session.transport).__name__,
                    "gen_ai.agent.name": session.agent.name,
                }
            ),
            start_time=self._ns(t),
        )
        self._root_ctx = self._otel.set_span_in_context(self._root)

    def session_closing(self, session: AgentSession, reason: str, t: float) -> None:
        if self._closed or self._root is None:
            return
        for child in self._pending:
            self._span(child, parent=self._root_ctx)
        self._pending.clear()
        end = self._ns(t)
        for span, _ in self._responses.values():
            span.set_attribute("voice_agent.response.status", "closed")
            span.end(end_time=end)
        self._responses.clear()
        while self._turns:
            self._turns.popleft().span.end(end_time=end)
        usage = session.usage
        self._root.set_attributes(
            {
                "voice_agent.session.close_reason": reason,
                "voice_agent.session.turns": self._turn_count,
                "gen_ai.usage.input_tokens": usage.llm_prompt_tokens
                + usage.engine_input_text_tokens
                + usage.engine_input_audio_tokens,
                "gen_ai.usage.output_tokens": usage.llm_completion_tokens
                + usage.engine_output_text_tokens
                + usage.engine_output_audio_tokens,
                "voice_agent.usage.stt_audio_seconds": usage.stt_audio_seconds,
                "voice_agent.usage.tts_characters": usage.tts_characters,
            }
        )
        self._root.end(end_time=end)
        self._closed = True

    def engine_event(self, event: EngineEvent) -> None:
        if self._closed or self._root is None:
            return
        if isinstance(event, InputSpeechStarted):
            stop = self._speech_stop
            if self._speech_start is None or (
                stop is not None and event.timestamp - stop > _SEGMENT_GAP
            ):
                self._speech_start = event.timestamp
            self._speech_stop = None
        elif isinstance(event, InputSpeechStopped):
            self._speech_stop = event.timestamp
        elif isinstance(event, InputCommitted):
            self._open_turn(event)
        elif isinstance(event, ResponseStarted):
            span = self._start(
                "response", event.timestamp, {"gen_ai.response.id": event.response_id}
            )
            self._responses[event.response_id] = (span, [])
        elif isinstance(event, ResponseDone):
            entry = self._responses.pop(event.response_id, None)
            if entry is None:
                return
            span, text = entry
            span.set_attribute("voice_agent.response.status", event.status)
            span.set_attribute("gen_ai.response.finish_reasons", [event.status])
            if event.usage is not None:
                u = event.usage
                span.set_attributes(
                    {
                        "gen_ai.usage.input_tokens": u.input_text_tokens + u.input_audio_tokens,
                        "gen_ai.usage.output_tokens": u.output_text_tokens + u.output_audio_tokens,
                        "voice_agent.usage.input_audio_tokens": u.input_audio_tokens,
                        "voice_agent.usage.output_audio_tokens": u.output_audio_tokens,
                        "gen_ai.usage.cache_read.input_tokens": u.cached_tokens,
                    }
                )
            if text:
                span.set_attribute("voice_agent.agent.transcript", "".join(text).strip())
            if event.status == "failed" or event.error:
                self._fail(span, event.error or event.status)
            span.end(end_time=self._ns(event.timestamp))

    def _open_turn(self, event: InputCommitted) -> None:
        start = self._speech_start if self._speech_start is not None else event.timestamp
        start = min(start, event.timestamp)
        self._turn_count += 1
        span = self._start(
            "turn",
            start,
            {"voice_agent.turn.number": self._turn_count, "voice_agent.item_id": event.item_id},
            parent=self._root_ctx,
        )
        turn = _Turn(span, self._otel.set_span_in_context(span, self._root_ctx), self._turn_count)
        self._turns.append(turn)
        if self._speech_stop is not None and self._speech_stop <= event.timestamp:
            self._span(
                _Child(
                    "end_of_turn",
                    self._speech_stop,
                    event.timestamp,
                    {"voice_agent.end_of_turn_delay": event.timestamp - self._speech_stop},
                )
            )
        for child in self._pending:
            self._span(child)
        self._pending.clear()
        self._speech_start = self._speech_stop = None

    # ------------------------------------------------------------ session events
    def _on_metrics(self, m: Metrics) -> None:
        if self._closed or self._root is None:
            return
        t = now()
        if isinstance(m, TurnMetrics):
            self._close_turn(m, t)
        elif isinstance(m, LLMMetrics):
            self._child(
                _Child(
                    f"chat {m.model}",
                    t - m.duration,
                    t,
                    {
                        "gen_ai.operation.name": "chat",
                        "gen_ai.provider.name": m.provider,
                        "gen_ai.request.model": m.model,
                        "gen_ai.response.id": m.request_id,
                        "gen_ai.usage.input_tokens": m.prompt_tokens,
                        "gen_ai.usage.output_tokens": m.completion_tokens,
                        "gen_ai.usage.cache_read.input_tokens": m.cached_tokens,
                        "gen_ai.usage.cache_creation.input_tokens": m.cache_creation_tokens,
                        "voice_agent.llm.ttft": m.ttft,
                        "voice_agent.llm.tokens_per_second": m.tokens_per_second,
                        "voice_agent.cancelled": m.cancelled,
                    },
                    kind="CLIENT",
                    error=m.error,
                )
            )
        elif isinstance(m, TTSMetrics):
            self._child(
                _Child(
                    "tts",
                    t - m.duration,
                    t,
                    {
                        "gen_ai.provider.name": m.provider,
                        "gen_ai.request.model": m.model,
                        "voice_agent.request_id": m.request_id,
                        "voice_agent.tts.ttfb": m.ttfb,
                        "voice_agent.tts.characters": m.characters,
                        "voice_agent.tts.audio_duration": m.audio_duration,
                        "voice_agent.streamed": m.streamed,
                        "voice_agent.cancelled": m.cancelled,
                    },
                    kind="CLIENT",
                    error=m.error,
                )
            )
        elif isinstance(m, STTMetrics):
            took = m.latency if m.latency is not None else m.duration
            self._child(
                _Child(
                    "stt",
                    t - took,
                    t,
                    {
                        "gen_ai.provider.name": m.provider,
                        "gen_ai.request.model": m.model,
                        "voice_agent.request_id": m.request_id,
                        "voice_agent.stt.latency": m.latency,
                        "voice_agent.stt.audio_duration": m.audio_duration,
                        "voice_agent.streamed": m.streamed,
                    },
                    kind="CLIENT",
                    error=m.error,
                ),
                needs_turn=True,
            )
        elif isinstance(m, EOTMetrics):
            self._child(
                _Child(
                    "turn_detection",
                    t - m.inference_duration,
                    t,
                    {
                        "gen_ai.provider.name": m.provider,
                        "gen_ai.request.model": m.model,
                        "voice_agent.eot.probability": m.probability,
                        "voice_agent.eot.threshold": m.threshold,
                        "voice_agent.eot.end_of_turn": m.end_of_turn,
                    },
                ),
                needs_turn=True,
            )
        elif isinstance(m, EngineMetrics):
            attrs = {
                "gen_ai.provider.name": m.provider,
                "gen_ai.request.model": m.model,
                "gen_ai.usage.input_tokens": m.input_text_tokens + m.input_audio_tokens,
                "gen_ai.usage.output_tokens": m.output_text_tokens + m.output_audio_tokens,
                "gen_ai.usage.cache_read.input_tokens": m.cached_tokens,
                "voice_agent.usage.input_audio_tokens": m.input_audio_tokens,
                "voice_agent.usage.output_audio_tokens": m.output_audio_tokens,
                "voice_agent.response.ttfb": m.ttfb,
                "voice_agent.cancelled": m.cancelled,
            }
            entry = self._responses.get(m.response_id)
            if entry is not None:  # usually: metrics come before the session sees the done
                entry[0].set_attributes(_clean(attrs))
            else:
                attrs["gen_ai.response.id"] = m.response_id
                self._child(_Child(f"response {m.model}", t - m.duration, t, attrs))

    def _close_turn(self, m: TurnMetrics, t: float) -> None:
        if not self._turns:
            return
        turn = self._turns.popleft()  # the session closes turns in commit order
        turn.span.set_attributes(
            _clean(
                {
                    "voice_agent.turn.id": m.turn_id,
                    "voice_agent.turn.voice_to_voice": m.voice_to_voice,
                    "voice_agent.turn.end_of_turn_delay": m.end_of_turn_delay,
                    "voice_agent.turn.response_ttfb": m.response_ttfb,
                    "voice_agent.turn.agent_speech_duration": m.agent_speech_duration,
                    "voice_agent.turn.was_interrupted": m.interrupted,
                    "voice_agent.turn.tool_calls": m.tool_calls,
                }
            )
        )
        if turn.user_text:
            turn.span.set_attribute("voice_agent.user.transcript", " ".join(turn.user_text))
        turn.span.end(end_time=self._ns(t))

    def _on_tool_call(self, ev: ToolCalled) -> None:
        if not self._closed:
            self._tools[ev.call.call_id] = (self._parent(), ev.timestamp)

    def _on_tool_result(self, ev: ToolResult) -> None:
        if self._closed or self._root is None:
            return
        parent, start = self._tools.pop(ev.call.call_id, (None, ev.timestamp - ev.duration))
        attrs: dict[str, Any] = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": ev.call.name,
            "gen_ai.tool.call.id": ev.call.call_id,
            "gen_ai.tool.type": "function",
        }
        if self.capture_content:
            attrs["gen_ai.tool.call.arguments"] = ev.call.arguments
            attrs["gen_ai.tool.call.result"] = ev.output.output
        error = ev.output.output if ev.output.is_error else None
        self._span(
            _Child(f"execute_tool {ev.call.name}", start, ev.timestamp, attrs, error=error),
            parent=parent,
        )

    def _on_interrupted(self, ev: Interrupted) -> None:
        span = self._current_span()
        if span is not None:
            span.add_event(
                "interrupted",
                _clean({"voice_agent.played": ev.played, "gen_ai.response.id": ev.response_id}),
                timestamp=self._ns(ev.timestamp),
            )

    def _on_false_interruption(self, ev: AgentFalseInterruption) -> None:
        span = self._current_span()
        if span is not None:
            span.add_event(
                "false_interruption",
                {
                    "voice_agent.resumed": ev.resumed,
                    "voice_agent.reason": ev.reason,
                    "voice_agent.paused": ev.paused,
                    "voice_agent.speech_duration": ev.speech_duration,
                },
                timestamp=self._ns(ev.timestamp),
            )

    def _on_agent_handoff(self, ev: AgentHandoff) -> None:
        attrs: dict[str, Any] = {
            "gen_ai.agent.name": ev.to_agent,
            "voice_agent.handoff.from": ev.from_agent,
            "voice_agent.handoff.to": ev.to_agent,
            "voice_agent.handoff.history": ev.history,
            "voice_agent.handoff.voice_changed": ev.voice_changed,
            "voice_agent.handoff.unsupported": list(ev.unsupported),
        }
        if ev.call is not None:
            attrs["gen_ai.tool.call.id"] = ev.call.call_id
        self._child(_Child("agent_handoff", ev.timestamp - ev.duration, ev.timestamp, attrs))

    def _on_error(self, ev: SessionError) -> None:
        if self._closed or self._root is None:
            return
        self._root.record_exception(
            ev.error,
            attributes={"voice_agent.recoverable": ev.recoverable},
            timestamp=self._ns(ev.timestamp),
        )
        if not ev.recoverable:
            self._fail(self._root, f"{type(ev.error).__name__}: {ev.error}")

    def _on_user_transcript(self, ev: UserTranscript) -> None:
        if ev.is_final and self._turns and not self._closed:
            self._turns[-1].user_text.append(ev.text)

    def _on_agent_transcript(self, ev: AgentTranscript) -> None:
        entry = self._responses.get(ev.response_id)
        if entry is not None:
            entry[1].append(ev.delta)

    def _current_span(self) -> Any:
        if self._closed or self._root is None:
            return None
        # the oldest open turn: a barge-in that commits a new turn interrupts the previous one
        return self._turns[0].span if self._turns else self._root


def _clean(attributes: dict[str, Any]) -> dict[str, Any]:
    """OpenTelemetry attributes cannot be ``None``."""
    return {k: v for k, v in attributes.items() if v is not None}
