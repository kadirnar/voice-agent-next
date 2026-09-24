"""DeepSeek: DeepSeek's hosted models through their OpenAI-compatible API.

``llm="deepseek/deepseek-flash"``. Needs ``DEEPSEEK_API_KEY`` (or ``api_key=``).
Base URL ``https://api.deepseek.com`` (override: ``base_url=`` or ``DEEPSEEK_BASE_URL``).

The API enables thinking by default, which delays the first spoken word by seconds;
this provider turns it off. Re-enable it with
``extra={"thinking": {"type": "enabled"}}``.
"""

from __future__ import annotations

from ..registry import register_provider
from .openai.llm import OpenAICompatibleLLM

__all__ = ["DeepSeekLLM"]


@register_provider(
    "llm",
    "deepseek",
    description="DeepSeek hosted LLMs (OpenAI-compatible API)",
    default_model="deepseek-flash",
    models=("deepseek-flash", "deepseek-v4-pro"),
    env=("DEEPSEEK_API_KEY",),
    extra="openai",
    requires=("openai",),
)
class DeepSeekLLM(OpenAICompatibleLLM):
    """DeepSeek (``https://api.deepseek.com``), thinking disabled by default."""

    provider = "deepseek"
    DEFAULT_MODEL = "deepseek-flash"
    DEFAULT_BASE_URL = "https://api.deepseek.com"
    BASE_URL_ENV = ("DEEPSEEK_BASE_URL",)
    API_KEY_ENV = ("DEEPSEEK_API_KEY",)
    DEFAULT_EXTRA = {"thinking": {"type": "disabled"}}
