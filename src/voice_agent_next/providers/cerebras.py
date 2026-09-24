"""Cerebras: very fast hosted inference through Cerebras' OpenAI-compatible API.

``llm="cerebras/gpt-oss-120b"``. Needs ``CEREBRAS_API_KEY`` (or ``api_key=``).
Base URL ``https://api.cerebras.ai/v1`` (override: ``base_url=`` or ``CEREBRAS_BASE_URL``).
``gpt-oss-120b`` is a reasoning model; ``reasoning_effort="low"`` keeps the first token fast.
"""

from __future__ import annotations

from ..registry import register_provider
from .openai.llm import OpenAICompatibleLLM

__all__ = ["CerebrasLLM"]


@register_provider(
    "llm",
    "cerebras",
    description="Cerebras hosted LLMs (OpenAI-compatible API)",
    default_model="gpt-oss-120b",
    models=("gpt-oss-120b", "qwen-3.8-27b"),
    env=("CEREBRAS_API_KEY",),
    extra="openai",
    requires=("openai",),
)
class CerebrasLLM(OpenAICompatibleLLM):
    """Cerebras (``https://api.cerebras.ai/v1``)."""

    provider = "cerebras"
    DEFAULT_MODEL = "gpt-oss-120b"
    DEFAULT_BASE_URL = "https://api.cerebras.ai/v1"
    BASE_URL_ENV = ("CEREBRAS_BASE_URL",)
    API_KEY_ENV = ("CEREBRAS_API_KEY",)
