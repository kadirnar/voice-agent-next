"""SambaNova Cloud: fast hosted inference through SambaNova's OpenAI-compatible API.

``llm="sambanova/Meta-Llama-3.3-70B-Instruct"``. Needs ``SAMBANOVA_API_KEY`` (or
``api_key=``). Base URL ``https://api.sambanova.ai/v1`` (override: ``base_url=`` or
``SAMBANOVA_BASE_URL``).
"""

from __future__ import annotations

from ..registry import register_provider
from .openai.llm import OpenAICompatibleLLM

__all__ = ["SambaNovaLLM"]


@register_provider(
    "llm",
    "sambanova",
    description="SambaNova Cloud hosted LLMs (OpenAI-compatible API)",
    default_model="Meta-Llama-3.3-70B-Instruct",
    models=("Meta-Llama-3.3-70B-Instruct", "gpt-oss-120b"),
    env=("SAMBANOVA_API_KEY",),
    extra="openai",
    requires=("openai",),
)
class SambaNovaLLM(OpenAICompatibleLLM):
    """SambaNova Cloud (``https://api.sambanova.ai/v1``)."""

    provider = "sambanova"
    DEFAULT_MODEL = "Meta-Llama-3.3-70B-Instruct"
    DEFAULT_BASE_URL = "https://api.sambanova.ai/v1"
    BASE_URL_ENV = ("SAMBANOVA_BASE_URL",)
    API_KEY_ENV = ("SAMBANOVA_API_KEY",)
