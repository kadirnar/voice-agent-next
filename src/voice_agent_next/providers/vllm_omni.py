"""vLLM-Omni: omni models (Qwen2.5/3-Omni, MiniCPM-o...) on ``/v1/chat/completions``.

``vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091`` serves the whole
Thinker -> Talker -> Code2Wav pipeline. ``llm="vllm_omni"`` sends the user's audio as
``input_audio`` (WAV, 16 kHz) — a half-cascade without an STT — and asks for text only
(``modalities: ["text"]``): the cascade's TTS speaks the reply. Pass ``voice=`` (or
``extra={"modalities": ["text", "audio"]}``) to have the model speak for itself instead;
its streamed audio is read as pcm16 at 24 kHz (experimental: check that your vLLM-Omni
version streams ``delta.audio``). Server address: ``base_url=``, else
``VLLM_OMNI_BASE_URL``, else ``http://127.0.0.1:8091/v1``; ``VLLM_API_KEY`` when the
server runs with ``--api-key``. The Realtime WebSocket of the same server is the
``vllm_realtime`` engine.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from ..registry import register_provider
from .vllm import VllmLLM

__all__ = ["VllmOmniLLM"]


@register_provider(
    "llm",
    "vllm_omni",
    description="vLLM-Omni server (Qwen-Omni audio in; OpenAI-compatible chat)",
    models=(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "Qwen/Qwen2.5-Omni-7B",
        "Qwen/Qwen2.5-Omni-3B",
    ),
    extra="openai",
    requires=("openai",),
    local=True,
    aliases=("vllm-omni",),
)
class VllmOmniLLM(VllmLLM):
    """A vLLM-Omni server (default port 8091). Audio input is on by default."""

    provider = "vllm_omni"
    DEFAULT_BASE_URL = "http://127.0.0.1:8091/v1"
    BASE_URL_ENV = ("VLLM_OMNI_BASE_URL",)
    AUDIO_INPUT: ClassVar[bool | None] = True
    DEFAULT_EXTRA: ClassVar[Mapping[str, Any]] = {"modalities": ["text"]}
