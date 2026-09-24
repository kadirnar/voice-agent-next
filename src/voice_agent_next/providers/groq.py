"""Groq: fast hosted inference through Groq's OpenAI-compatible API.

``llm="groq/llama-3.3-70b-versatile"``. Needs ``GROQ_API_KEY`` (or ``api_key=``).
Base URL ``https://api.groq.com/openai/v1`` (override: ``base_url=`` or ``GROQ_BASE_URL``).
"""

from __future__ import annotations

from ..registry import register_provider
from .openai.llm import OpenAICompatibleLLM

__all__ = ["GroqLLM"]


@register_provider(
    "llm",
    "groq",
    description="Groq hosted LLMs (OpenAI-compatible API)",
    default_model="llama-3.3-70b-versatile",
    models=(
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
    ),
    env=("GROQ_API_KEY",),
    extra="openai",
    requires=("openai",),
)
class GroqLLM(OpenAICompatibleLLM):
    """Groq (``https://api.groq.com/openai/v1``)."""

    provider = "groq"
    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
    BASE_URL_ENV = ("GROQ_BASE_URL",)
    API_KEY_ENV = ("GROQ_API_KEY",)
