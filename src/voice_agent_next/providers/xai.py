"""xAI Grok Voice Agent API — the OpenAI Realtime engine with the ``xai`` profile.

``AgentSession("xai/grok-voice-latest")`` connects to ``wss://api.x.ai/v1/realtime``
(``XAI_API_KEY``). Differences handled by the profile: top-level ``voice`` /
``turn_detection`` in ``session.update`` (``server_vad`` or manual only), cumulative
``conversation.item.input_audio_transcription.updated`` transcripts, and verbatim
``say()`` through xAI's ``force_message`` item. See ``docs/providers/openai-realtime.md``.
"""

from __future__ import annotations

from typing import Any

from ..registry import register_provider
from .openai.realtime import OpenAIRealtimeEngine

__all__ = ["XAIRealtimeEngine"]


@register_provider(
    "engine",
    "xai",
    description="xAI Grok Voice Agent speech-to-speech (OpenAI-Realtime-compatible)",
    default_model="grok-voice-latest",
    models=("grok-voice-latest", "grok-voice-think-fast-2.0"),
    env=("XAI_API_KEY",),
)
class XAIRealtimeEngine(OpenAIRealtimeEngine):
    """Grok voice models over xAI's Realtime-compatible WebSocket API.

    Accepts every :class:`~voice_agent_next.providers.openai.realtime.OpenAIRealtimeEngine`
    option (``voice="eve"``, ``reasoning_effort="none"``, ...).
    """

    provider = "xai"

    def __init__(self, *, model: str | None = None, **kwargs: Any) -> None:
        super().__init__(model=model, profile="xai", **kwargs)
