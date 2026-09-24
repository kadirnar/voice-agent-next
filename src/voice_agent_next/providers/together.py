"""Together AI: hosted open models through Together's OpenAI-compatible API.

``llm="together/meta-llama/Llama-3.3-70B-Instruct-Turbo"``. Needs ``TOGETHER_API_KEY``
(or ``api_key=``). Base URL ``https://api.together.ai/v1`` (override: ``base_url=`` or
``TOGETHER_BASE_URL``).
"""

from __future__ import annotations

from ..registry import register_provider
from .openai.llm import OpenAICompatibleLLM

__all__ = ["TogetherLLM"]


@register_provider(
    "llm",
    "together",
    description="Together AI hosted LLMs (OpenAI-compatible API)",
    default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    models=("meta-llama/Llama-3.3-70B-Instruct-Turbo", "openai/gpt-oss-120b"),
    env=("TOGETHER_API_KEY",),
    extra="openai",
    requires=("openai",),
)
class TogetherLLM(OpenAICompatibleLLM):
    """Together AI (``https://api.together.ai/v1``)."""

    provider = "together"
    DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    DEFAULT_BASE_URL = "https://api.together.ai/v1"
    BASE_URL_ENV = ("TOGETHER_BASE_URL",)
    API_KEY_ENV = ("TOGETHER_API_KEY",)
