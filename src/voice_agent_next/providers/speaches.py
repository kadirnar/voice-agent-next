"""Speaches Realtime API — the OpenAI Realtime engine with the ``speaches`` profile.

``AgentSession("speaches/<llm-model>")`` connects to a local Speaches server
(``ws://localhost:8000/v1`` or ``SPEACHES_BASE_URL``; ``SPEACHES_API_KEY`` if set).
Speaches runs a VAD -> STT -> LLM -> TTS pipeline behind the beta Realtime dialect and
supports neither ``response.cancel`` nor ``conversation.item.truncate``, so the engine
declares ``truncation=False`` and barge-in only stops local playback.
See ``docs/providers/openai-realtime.md``.
"""

from __future__ import annotations

from typing import Any

from ..registry import register_provider
from .openai.realtime import OpenAIRealtimeEngine

__all__ = ["SpeachesRealtimeEngine"]


@register_provider(
    "engine",
    "speaches",
    description="Speaches OpenAI-Realtime-compatible voice pipeline (local server)",
    local=True,
)
class SpeachesRealtimeEngine(OpenAIRealtimeEngine):
    """A Speaches server's Realtime endpoint (the model is its conversation LLM)."""

    provider = "speaches"

    def __init__(self, *, model: str | None = None, **kwargs: Any) -> None:
        super().__init__(model=model, profile="speaches", **kwargs)
