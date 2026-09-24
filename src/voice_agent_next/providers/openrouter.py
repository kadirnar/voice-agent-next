"""OpenRouter: one API for models from many providers (OpenAI-compatible).

``llm="openrouter/openai/gpt-4.1-mini"``. Needs ``OPENROUTER_API_KEY`` (or ``api_key=``).
Base URL ``https://openrouter.ai/api/v1`` (override: ``base_url=`` or
``OPENROUTER_BASE_URL``). Optional app attribution:
``headers={"HTTP-Referer": "https://your.app", "X-Title": "Your app"}``; routing
preferences go through ``extra``, e.g. ``extra={"provider": {"sort": "latency"}}``.
"""

from __future__ import annotations

from ..registry import register_provider
from .openai.llm import OpenAICompatibleLLM

__all__ = ["OpenRouterLLM"]


@register_provider(
    "llm",
    "openrouter",
    description="OpenRouter multi-provider LLM gateway (OpenAI-compatible API)",
    default_model="openai/gpt-4.1-mini",
    env=("OPENROUTER_API_KEY",),
    extra="openai",
    requires=("openai",),
)
class OpenRouterLLM(OpenAICompatibleLLM):
    """OpenRouter (``https://openrouter.ai/api/v1``)."""

    provider = "openrouter"
    DEFAULT_MODEL = "openai/gpt-4.1-mini"
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    BASE_URL_ENV = ("OPENROUTER_BASE_URL",)
    API_KEY_ENV = ("OPENROUTER_API_KEY",)
