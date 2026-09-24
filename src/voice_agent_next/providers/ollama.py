"""Ollama: local models through Ollama's OpenAI-compatible API (``/v1``).

``llm="ollama/qwen3.5:4b"``. Server address: ``base_url=``, else ``OLLAMA_BASE_URL``
(a full URL), else Ollama's own ``OLLAMA_HOST`` (``host:port`` or URL), else
``http://127.0.0.1:11434/v1``. No API key is needed locally; ``OLLAMA_API_KEY`` is sent
when set (Ollama's cloud endpoint, ``https://ollama.com/v1``).
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

from ..registry import register_provider
from .openai.llm import OpenAICompatibleLLM

__all__ = ["OllamaLLM", "ollama_base_url"]


def ollama_base_url(host: str) -> str:
    """OpenAI base URL for an ``OLLAMA_HOST`` value, following Ollama's own parsing.

    ``"0.0.0.0:11434"`` -> ``"http://127.0.0.1:11434/v1"``; without a scheme the port
    defaults to 11434, with ``http://``/``https://`` to 80/443. Wildcard bind addresses
    are replaced by loopback, which every OS can connect to.
    """
    value = host.strip()
    scheme, sep, rest = value.partition("://")
    if sep:
        scheme = scheme.lower()
        default_port = 443 if scheme == "https" else 80
    else:
        scheme, rest, default_port = "http", value, 11434
    parts = urlsplit(f"{scheme}://{rest}")
    hostname = parts.hostname or "127.0.0.1"
    hostname = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(hostname, hostname)
    try:
        port = parts.port or default_port
    except ValueError:  # invalid port: Ollama falls back to the default too
        port = default_port
    netloc = f"[{hostname}]" if ":" in hostname else hostname
    if port != (443 if scheme == "https" else 80):
        netloc += f":{port}"
    return f"{scheme}://{netloc}{parts.path.rstrip('/')}/v1"


@register_provider(
    "llm",
    "ollama",
    description="Ollama local models (OpenAI-compatible API)",
    default_model="qwen3.5:4b",
    models=("qwen3.5:4b", "qwen3.5:9b", "qwen3.5:2b", "qwen3.5:27b"),
    extra="openai",
    requires=("openai",),
    local=True,
)
class OllamaLLM(OpenAICompatibleLLM):
    """Ollama (``ollama serve``, default port 11434). Models load on first use; call
    :meth:`warmup` to load the model before the first turn."""

    provider = "ollama"
    DEFAULT_MODEL = "qwen3.5:4b"
    DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
    BASE_URL_ENV = ("OLLAMA_BASE_URL",)
    API_KEY_ENV = ("OLLAMA_API_KEY",)
    API_KEY_REQUIRED = False
    PRELOAD_ON_WARMUP = True
    NOT_FOUND_HINT = "pull it first: `ollama pull {model}`"

    def _base_url_from_env(self) -> str | None:
        url = super()._base_url_from_env()
        if url:
            return url
        host = os.environ.get("OLLAMA_HOST", "").strip()
        return ollama_base_url(host) if host else None
