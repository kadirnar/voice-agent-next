"""Fireworks AI: hosted open models through Fireworks' OpenAI-compatible API.

``llm="fireworks/accounts/fireworks/models/llama-v3p3-70b-instruct"``. Needs
``FIREWORKS_API_KEY`` (or ``api_key=``). Base URL ``https://api.fireworks.ai/inference/v1``
(override: ``base_url=`` or ``FIREWORKS_BASE_URL``).
"""

from __future__ import annotations

from ..registry import register_provider
from .openai.llm import OpenAICompatibleLLM

__all__ = ["FireworksLLM"]


@register_provider(
    "llm",
    "fireworks",
    description="Fireworks AI hosted LLMs (OpenAI-compatible API)",
    default_model="accounts/fireworks/models/llama-v3p3-70b-instruct",
    models=(
        "accounts/fireworks/models/llama-v3p3-70b-instruct",
        "accounts/fireworks/models/gpt-oss-120b",
        "accounts/fireworks/models/gpt-oss-20b",
        "accounts/fireworks/models/qwen3p5-4b",
    ),
    env=("FIREWORKS_API_KEY",),
    extra="openai",
    requires=("openai",),
)
class FireworksLLM(OpenAICompatibleLLM):
    """Fireworks AI (``https://api.fireworks.ai/inference/v1``)."""

    provider = "fireworks"
    DEFAULT_MODEL = "accounts/fireworks/models/llama-v3p3-70b-instruct"
    DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"
    BASE_URL_ENV = ("FIREWORKS_BASE_URL",)
    API_KEY_ENV = ("FIREWORKS_API_KEY",)
