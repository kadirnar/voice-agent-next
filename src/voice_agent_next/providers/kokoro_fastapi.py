"""Kokoro-FastAPI — Kokoro-82M behind an OpenAI-compatible ``/v1/audio/speech`` server.

``tts="kokoro_fastapi"`` (or ``"kokoro-fastapi/kokoro"``) talks to a
`Kokoro-FastAPI <https://github.com/remsky/Kokoro-FastAPI>`_ server
(``http://localhost:8880/v1`` or ``KOKORO_FASTAPI_BASE_URL``; ``KOKORO_FASTAPI_API_KEY`` if
the server requires one), e.g. started with
``docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:latest`` (or the ``-gpu``
image). Audio is requested as raw 24 kHz PCM and streamed as it is generated.

Voices are Kokoro voice names (``af_heart`` by default, ``af_bella``, ``bm_george``...) or
mixes (``af_bella(2)+af_sky(1)``). Unlike the in-process ``kokoro`` provider
(``kokoro-onnx``), the model runs in the server, so this provider has no local dependency.
See ``docs/providers/openai.md``.
"""

from __future__ import annotations

from typing import ClassVar

from ..registry import register_provider
from .openai.tts import OpenAICompatibleTTS

__all__ = ["KokoroFastAPITTS"]


@register_provider(
    "tts",
    "kokoro_fastapi",
    description="Kokoro-FastAPI server (OpenAI-compatible /v1/audio/speech, 24 kHz PCM)",
    default_model="kokoro",
    requires=("httpx",),
    local=True,
)
class KokoroFastAPITTS(OpenAICompatibleTTS):
    """A Kokoro-FastAPI server (any :class:`~voice_agent_next.providers.openai.tts.OpenAITTS`
    option; ``speed`` is 0.25-4.0)."""

    provider = "kokoro_fastapi"
    DEFAULT_MODEL: ClassVar[str] = "kokoro"
    DEFAULT_VOICE: ClassVar[str | None] = "af_heart"
    DEFAULT_BASE_URL: ClassVar[str] = "http://localhost:8880/v1"
    BASE_URL_ENV: ClassVar[tuple[str, ...]] = ("KOKORO_FASTAPI_BASE_URL",)
    API_KEY_ENV: ClassVar[tuple[str, ...]] = ("KOKORO_FASTAPI_API_KEY",)
