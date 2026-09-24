"""vLLM: the ``vllm serve`` OpenAI-compatible server (self-hosted, GPU).

``llm="vllm"`` uses the model the server is serving (``GET /v1/models``); pass the
served model name to pick one explicitly (``llm="vllm/Qwen/Qwen3-8B"``). Tool calling
needs ``--enable-auto-tool-choice --tool-call-parser <parser>`` on the server. Server
address: ``base_url=``, else ``VLLM_BASE_URL``, else ``http://127.0.0.1:8000/v1``.
When the server runs with ``--api-key``, pass ``api_key=`` or set ``VLLM_API_KEY`` (the
server's own variable).
"""

from __future__ import annotations

from ..registry import register_provider
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
