"""llama.cpp ``llama-server``: local GGUF models through its OpenAI-compatible API.

Start the server with ``--jinja`` for tool calling, e.g.
``llama-server -hf ggml-org/gpt-oss-20b-GGUF --jinja``. ``llm="llamacpp"`` uses the
model the server has loaded (``GET /v1/models``). Server address: ``base_url=``, else
``LLAMACPP_BASE_URL``, else ``http://127.0.0.1:8080/v1``. When the server runs with
``--api-key``, pass ``api_key=`` or set ``LLAMA_API_KEY`` (the server's own variable).
"""

from __future__ import annotations

from ..registry import register_provider
from .openai.llm import OpenAICompatibleLLM

__all__ = ["LlamaCppLLM"]


@register_provider(
    "llm",
    "llamacpp",
    description="llama.cpp llama-server (OpenAI-compatible API)",
    extra="openai",
    requires=("openai",),
    local=True,
)
class LlamaCppLLM(OpenAICompatibleLLM):
    """llama.cpp ``llama-server`` (default port 8080)."""

    provider = "llamacpp"
    SYSTEM_MESSAGE_POLICY = "merge"  # Jinja chat templates: one leading system message
    DEFAULT_MODEL = None
    DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
    BASE_URL_ENV = ("LLAMACPP_BASE_URL",)
    API_KEY_ENV = ("LLAMA_API_KEY",)
    API_KEY_REQUIRED = False
