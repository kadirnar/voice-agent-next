"""vLLM-Omni ``/v1/realtime`` — the OpenAI Realtime engine with the ``vllm_realtime`` profile.

``AgentSession("vllm_realtime/<served-model>")`` connects to a local vLLM-Omni server
(``ws://localhost:8000/v1`` or ``VLLM_BASE_URL``; ``VLLM_API_KEY`` if the server needs
one). The profile speaks the GA dialect and also accepts beta event names. Upstream
vLLM's transcription-only ``/v1/realtime`` (``transcription.delta`` events) is a
speech-to-text endpoint, not a speech-to-speech engine, and is not supported here.
Experimental: see ``docs/providers/openai-realtime.md``.
"""

from __future__ import annotations

from typing import Any

from ..registry import register_provider
from .openai.realtime import OpenAIRealtimeEngine

__all__ = ["VLLMRealtimeEngine"]


@register_provider(
    "engine",
    "vllm_realtime",
    description="vLLM-Omni OpenAI-Realtime-compatible speech-to-speech (local server)",
    models=("Qwen/Qwen3-Omni-30B-A3B-Instruct", "openbmb/MiniCPM-o-4_5"),
    local=True,
)
class VLLMRealtimeEngine(OpenAIRealtimeEngine):
    """A vLLM-Omni server's Realtime endpoint.

    The model is the served model name (omitted from the URL when not given). For
    MiniCPM-o full-duplex sessions pass ``query={"duplex": "1"}, turn_detection=None``
    (the model owns turn-taking).
    """

    provider = "vllm_realtime"

    def __init__(self, *, model: str | None = None, **kwargs: Any) -> None:
        super().__init__(model=model, profile="vllm_realtime", **kwargs)
