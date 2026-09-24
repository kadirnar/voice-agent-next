"""LocalAI Realtime API — the OpenAI Realtime engine with the ``localai`` profile.

``AgentSession("localai/<pipeline>")`` connects to a local LocalAI server
(``ws://localhost:8080/v1`` or ``LOCALAI_BASE_URL``; ``LOCALAI_API_KEY`` if set). The
model is the name of a LocalAI *pipeline* model (VAD + STT + LLM + TTS), ``gpt-realtime``
by default. LocalAI speaks the GA dialect. See ``docs/providers/openai-realtime.md``.
"""

from __future__ import annotations

from typing import Any

from ..registry import register_provider
from .openai.realtime import OpenAIRealtimeEngine

__all__ = ["LocalAIRealtimeEngine"]


@register_provider(
    "engine",
    "localai",
    description="LocalAI OpenAI-Realtime-compatible voice pipeline (local server)",
    default_model="gpt-realtime",
    local=True,
)
class LocalAIRealtimeEngine(OpenAIRealtimeEngine):
    """A LocalAI server's Realtime endpoint."""

    provider = "localai"

    def __init__(self, *, model: str | None = None, **kwargs: Any) -> None:
        super().__init__(model=model, profile="localai", **kwargs)
