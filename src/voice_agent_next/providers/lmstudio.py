"""LM Studio: the desktop app's local OpenAI-compatible server.

Enable the server in LM Studio (Developer tab) or run ``lms server start``.
``llm="lmstudio/<model-key>"`` picks a model (loaded on demand); without a model the
first LLM the server lists is used. Server address: ``base_url=``, else
``LMSTUDIO_BASE_URL``, else ``http://127.0.0.1:1234/v1``. With authentication enabled
in the server settings, pass ``api_key=`` or set ``LM_API_TOKEN``.
"""

from __future__ import annotations

from ..registry import register_provider
from .openai.llm import OpenAICompatibleLLM

__all__ = ["LMStudioLLM"]


@register_provider(
    "llm",
    "lmstudio",
    description="LM Studio local server (OpenAI-compatible API)",
    extra="openai",
    requires=("openai",),
    local=True,
)
class LMStudioLLM(OpenAICompatibleLLM):
    """LM Studio local server (default port 1234). Models load on demand; call
    :meth:`warmup` to load the model before the first turn."""

    provider = "lmstudio"
    SYSTEM_MESSAGE_POLICY = "merge"  # Jinja chat templates: one leading system message
    DEFAULT_MODEL = None
    DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
    BASE_URL_ENV = ("LMSTUDIO_BASE_URL",)
    API_KEY_ENV = ("LM_API_TOKEN",)
    API_KEY_REQUIRED = False
    PRELOAD_ON_WARMUP = True
