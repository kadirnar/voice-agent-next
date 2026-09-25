"""vLLM: the ``vllm serve`` OpenAI-compatible server (self-hosted, GPU).

``llm="vllm"`` uses the model the server is serving (``GET /v1/models``); pass the
served model name to pick one explicitly (``llm="vllm/Qwen/Qwen3-8B"``). Tool calling
needs ``--enable-auto-tool-choice --tool-call-parser <parser>`` on the server. Server
address: ``base_url=``, else ``VLLM_BASE_URL``, else ``http://127.0.0.1:8000/v1``.
When the server runs with ``--api-key``, pass ``api_key=`` or set ``VLLM_API_KEY`` (the
server's own variable).

Audio input (half-cascade): vLLM serves audio-in / text-out models (Qwen2-Audio,
Qwen2.5-Omni and Qwen3-Omni's thinker, Ultravox, Voxtral, Gemma 3n, Phi-4-multimodal...)
and takes ``input_audio`` parts in WAV. Known model ids turn ``audio_input`` on; with a
discovered model pass ``audio_input=True``. For Qwen-Omni's own voice use vLLM-Omni
(``llm="vllm_omni"``).
"""

from __future__ import annotations

from typing import ClassVar

from ..registry import register_provider
from .openai._format import AudioInputFormat
from .openai.llm import OpenAICompatibleLLM

__all__ = ["VllmLLM"]


@register_provider(
    "llm",
    "vllm",
    description="vLLM server (OpenAI-compatible API)",
    extra="openai",
    requires=("openai",),
    local=True,
)
class VllmLLM(OpenAICompatibleLLM):
    """vLLM ``vllm serve`` (default port 8000)."""

    provider = "vllm"
    SYSTEM_MESSAGE_POLICY = "merge"  # Jinja chat templates: one leading system message
    DEFAULT_MODEL = None
    DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
    BASE_URL_ENV = ("VLLM_BASE_URL",)
    API_KEY_ENV = ("VLLM_API_KEY",)
    API_KEY_REQUIRED = False
    # vLLM resamples to the processor's rate; the audio encoders it serves run at 16 kHz
    AUDIO_INPUT_FORMAT: ClassVar[AudioInputFormat] = AudioInputFormat("wav", 16_000)
