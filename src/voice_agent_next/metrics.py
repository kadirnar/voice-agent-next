"""Metrics emitted by components and the session.

Every component (STT, LLM, TTS, VAD, turn detector, engine) is an
:class:`~voice_agent_next.utils.EventEmitter` and emits ``"metrics"`` events with one
of the dataclasses below. :class:`~voice_agent_next.session.AgentSession` re-emits
them and adds per-turn :class:`TurnMetrics` (voice-to-voice latency etc.).

All durations are in **seconds**; ``timestamp`` is wall-clock ``time.time()``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypeAlias

__all__ = [
    "EOTMetrics",
    "EndpointingMetrics",
    "EndpointingPolicy",
    "EngineMetrics",
    "LLMMetrics",
    "Metrics",
    "RotationMetrics",
    "STTMetrics",
    "SpeculationMetrics",
    "SpeculationReason",
    "TTSMetrics",
    "TurnMetrics",
    "UsageSummary",
    "VADMetrics",
    "metrics_to_dict",
    "percentile",
    "summarize",
]


def _ts() -> float:
    return time.time()


@dataclass(slots=True, kw_only=True)
class STTMetrics:
    provider: str
    model: str
    request_id: str
    audio_duration: float = 0.0
    """Seconds of audio sent to the recognizer."""
    duration: float = 0.0
    """Processing time for batch recognition (0 for streaming)."""
    latency: float | None = None
    """Streaming: time from end-of-input/flush to the final transcript."""
    streamed: bool = False
    error: str | None = None
    timestamp: float = field(default_factory=_ts)
    type: Literal["stt"] = "stt"


@dataclass(slots=True, kw_only=True)
class LLMMetrics:
    provider: str
    model: str
    request_id: str
    ttft: float | None = None
    """Time to first token (text delta, tool call or audio)."""
    ttfb: float | None = None
    """Time to the first audio chunk (audio-output models only)."""
    duration: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    """Prompt tokens read from the prompt cache (included in ``prompt_tokens``)."""
    cache_creation_tokens: int = 0
    """Prompt tokens written to the prompt cache (included in ``prompt_tokens``)."""
    tokens_per_second: float = 0.0
    cancelled: bool = False
    error: str | None = None
    timestamp: float = field(default_factory=_ts)
    type: Literal["llm"] = "llm"


@dataclass(slots=True, kw_only=True)
class TTSMetrics:
    provider: str
    model: str
    request_id: str
    ttfb: float | None = None
    """Time from the first text input to the first audio byte."""
    duration: float = 0.0
    audio_duration: float = 0.0
    characters: int = 0
    streamed: bool = False
    cancelled: bool = False
    error: str | None = None
    timestamp: float = field(default_factory=_ts)
    type: Literal["tts"] = "tts"


@dataclass(slots=True, kw_only=True)
class VADMetrics:
    provider: str
    inference_count: int = 0
    inference_duration_total: float = 0.0
    audio_duration: float = 0.0
    timestamp: float = field(default_factory=_ts)
    type: Literal["vad"] = "vad"


@dataclass(slots=True, kw_only=True)
class EOTMetrics:
    """End-of-turn (semantic turn detection) inference."""

    provider: str
    model: str
    probability: float
    threshold: float
    inference_duration: float
    end_of_turn: bool
    timestamp: float = field(default_factory=_ts)
    type: Literal["eot"] = "eot"


@dataclass(slots=True, kw_only=True)
class EngineMetrics:
    """One response of a speech-to-speech engine."""

    provider: str
    model: str
    response_id: str
    ttfb: float | None = None
    """Time from the response trigger (turn commit / create_response) to the first audio."""
    duration: float = 0.0
    input_text_tokens: int = 0
    input_audio_tokens: int = 0
    output_text_tokens: int = 0
    output_audio_tokens: int = 0
    cached_tokens: int = 0
    cancelled: bool = False
    timestamp: float = field(default_factory=_ts)
    type: Literal["engine"] = "engine"


@dataclass(slots=True, kw_only=True)
class RotationMetrics:
    """One switch of an engine connection to a new provider connection/session: a planned
    rotation (session limit, ``goAway``, configuration change) or a reconnect after a drop.
    See ``docs/concepts/session-rotation.md``."""

    provider: str
    model: str
    reason: str
    planned: bool
    """Proactive rotation (``True``) or reactive reconnect after a failure (``False``)."""
    resumed: bool = False
    """The provider resumed its server-side session (e.g. a Gemini resumption handle); else
    a fresh session was seeded with the carried-over conversation."""
    rotation: int = 1
    """Number of switches on this engine connection so far (this one included)."""
    gap: float = 0.0
    """Seconds during which user audio was held back (buffered) by the switch."""
    attempts: int = 1
    """Connection attempts this switch needed."""
    buffered_audio: float = 0.0
    """Seconds of user audio buffered during the switch and delivered afterwards."""
    replayed_audio: float = 0.0
    """Seconds of audio sent to the old connection that were re-sent to the new one."""
    lost_audio: float = 0.0
    """Seconds of user audio that could not be delivered (buffer overflow)."""
    carried_items: int = 0
    """Conversation items re-seeded into a fresh session."""
    failed_responses: int = 0
    """Responses in flight that the switch cut off (``ResponseDone(status="failed")``)."""
    timestamp: float = field(default_factory=_ts)
    type: Literal["rotation"] = "rotation"


SpeculationReason: TypeAlias = Literal[
    "resumed", "transcript", "context", "cancelled", "cleared", "failed", "closed"
]
"""Why a speculative reply was discarded: the user kept talking (``resumed``), the
committed transcript differs (``transcript``), the conversation, instructions or tools
changed (``context``), a response was cancelled or another one started (``cancelled``),
the pending input was cleared (``cleared``), the generation failed before the commit
(``failed``) or the connection closed (``closed``)."""


@dataclass(slots=True, kw_only=True)
class SpeculationMetrics:
    """One speculative (preemptive) reply of the cascade, generated before the user's turn
    was committed (``CascadeOptions.preemptive_generation``): kept (``hit``) or discarded."""

    provider: str
    model: str
    request_id: str
    """The speculative LLM request (matches its :class:`LLMMetrics`)."""
    hit: bool
    """The committed turn matched: the reply was released at the commit."""
    reason: SpeculationReason | None = None
    """Why it was discarded (``None`` for a hit)."""
    response_id: str | None = None
    """The response it became (hits only)."""
    lead: float = 0.0
    """Seconds from the speculative start to the commit (hit) or to the discard."""
    output_tokens: int = 0
    """Output tokens generated before the commit or discard (the LLM's usage when it
    reported it by then, otherwise the number of streamed chunks)."""
    timestamp: float = field(default_factory=_ts)
    type: Literal["speculation"] = "speculation"


EndpointingPolicy: TypeAlias = Literal["fixed", "dynamic", "dictation"]
"""How the cascade chose an endpointing delay (see ``docs/concepts/endpointing.md``)."""


@dataclass(slots=True, kw_only=True)
class EndpointingMetrics:
    """One endpointing decision of the cascade: a pause after user speech, and the delay
    it waited before committing the turn (``CascadeOptions.endpointing``).

    Emitted when the outcome is known: at once when the user resumed before the commit
    (``committed=False``), otherwise ``false_commit_window`` after the commit (or earlier,
    when the user started speaking again within it: ``false_commit=True``).
    """

    provider: str
    model: str
    item_id: str
    """The user turn the pause belongs to."""
    policy: EndpointingPolicy
    delay: float
    """Chosen silence (seconds from the end of speech) before the commit."""
    probability: float | None = None
    """The turn detector's end-of-turn probability (``None``: no detector)."""
    threshold: float | None = None
    hold: float | None = None
    """Dynamic policy: the delay at the detector threshold (learned from the user's
    mid-turn pauses)."""
    committed: bool = True
    """``False``: the user resumed before the delay elapsed (the turn continued)."""
    pause: float | None = None
    """Seconds from the end of speech until the user spoke again (resumed pauses and
    false commits)."""
    false_commit: bool = False
    """The turn was committed, but the user started speaking again within
    ``false_commit_window``: probably cut off mid-thought."""
    audio_probability: float | None = None
    """Fused turn detector: the audio half's end-of-turn probability."""
    text_probability: float | None = None
    """Fused turn detector: the text half's end-of-turn probability (``None``: no
    transcript, or over its latency budget)."""
    detector_error: str | None = None
    """The turn detector failed at this pause (``repr`` of its error): the delay was
    chosen as if there were no detector."""
    timestamp: float = field(default_factory=_ts)
    type: Literal["endpointing"] = "endpointing"


@dataclass(slots=True, kw_only=True)
class TurnMetrics:
    """Session-level metrics for one agent turn (user speech -> agent reply)."""

    turn_id: str
    voice_to_voice: float | None = None
    """User stopped speaking -> first agent audio handed to the transport."""
    end_of_turn_delay: float | None = None
    """User stopped speaking -> turn committed (endpointing delay)."""
    response_ttfb: float | None = None
    """Turn committed -> first agent audio."""
    agent_speech_duration: float = 0.0
    interrupted: bool = False
    tool_calls: int = 0
    agent: str | None = None
    """Name of the agent that answered the turn (see agent handoffs)."""
    timestamp: float = field(default_factory=_ts)
    type: Literal["turn"] = "turn"


Metrics: TypeAlias = (
    STTMetrics
    | LLMMetrics
    | TTSMetrics
    | VADMetrics
    | EOTMetrics
    | EngineMetrics
    | SpeculationMetrics
    | EndpointingMetrics
    | TurnMetrics
    | RotationMetrics
)


def metrics_to_dict(m: Metrics) -> dict[str, Any]:
    return asdict(m)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in 0..100). NaN for empty input."""
    xs = sorted(v for v in values if v is not None and not math.isnan(v))
    if not xs:
        return math.nan
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (q / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(values: Iterable[float | None]) -> dict[str, float]:
    """count/mean/min/p50/p90/p95/p99/max of the non-None values."""
    xs = [v for v in values if v is not None and not math.isnan(v)]
    if not xs:
        return {"count": 0}
    return {
        "count": len(xs),
        "mean": sum(xs) / len(xs),
        "min": min(xs),
        "p50": percentile(xs, 50),
        "p90": percentile(xs, 90),
        "p95": percentile(xs, 95),
        "p99": percentile(xs, 99),
        "max": max(xs),
    }


@dataclass
class UsageSummary:
    """Aggregates usage across a session for cost estimation."""

    stt_audio_seconds: float = 0.0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_cached_tokens: int = 0
    llm_cache_creation_tokens: int = 0
    tts_characters: int = 0
    tts_audio_seconds: float = 0.0
    engine_input_audio_tokens: int = 0
    engine_output_audio_tokens: int = 0
    engine_input_text_tokens: int = 0
    engine_output_text_tokens: int = 0
    speculation_hits: int = 0
    """Speculative (preemptive) replies that were kept."""
    speculation_waste_calls: int = 0
    """Speculative LLM calls that were discarded."""
    speculation_waste_tokens: int = 0
    """Output tokens generated by discarded speculative calls."""
    engine_rotations: int = 0
    """Engine connection switches (planned rotations and reconnects)."""
    engine_rotation_gap: float = 0.0
    """Total seconds user audio was held back by those switches."""
    engine_lost_audio: float = 0.0
    """Total seconds of user audio lost by those switches."""

    def add(self, m: Metrics) -> None:
        if isinstance(m, STTMetrics):
            self.stt_audio_seconds += m.audio_duration
        elif isinstance(m, LLMMetrics):
            self.llm_prompt_tokens += m.prompt_tokens
            self.llm_completion_tokens += m.completion_tokens
            self.llm_cached_tokens += m.cached_tokens
            self.llm_cache_creation_tokens += m.cache_creation_tokens
        elif isinstance(m, TTSMetrics):
            self.tts_characters += m.characters
            self.tts_audio_seconds += m.audio_duration
        elif isinstance(m, EngineMetrics):
            self.engine_input_audio_tokens += m.input_audio_tokens
            self.engine_output_audio_tokens += m.output_audio_tokens
            self.engine_input_text_tokens += m.input_text_tokens
            self.engine_output_text_tokens += m.output_text_tokens
        elif isinstance(m, SpeculationMetrics):
            if m.hit:
                self.speculation_hits += 1
            else:
                self.speculation_waste_calls += 1
                self.speculation_waste_tokens += m.output_tokens
        elif isinstance(m, RotationMetrics):
            self.engine_rotations += 1
            self.engine_rotation_gap += m.gap
            self.engine_lost_audio += m.lost_audio
