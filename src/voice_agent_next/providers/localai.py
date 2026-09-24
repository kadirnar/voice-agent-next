"""LocalAI — a local OpenAI-compatible server (realtime pipelines, whisper.cpp, TTS backends).

* ``AgentSession("localai/<pipeline>")`` — :class:`LocalAIRealtimeEngine`, the OpenAI
  Realtime engine with the ``localai`` profile (``ws://localhost:8080/v1`` or
  ``LOCALAI_BASE_URL``; ``LOCALAI_API_KEY`` if set). The model is the name of a LocalAI
  *pipeline* model (VAD + STT + LLM + TTS), ``gpt-realtime`` by default. LocalAI speaks the
  GA dialect. See ``docs/providers/openai-realtime.md``.
* ``stt="localai/<model>"`` — :class:`LocalAISTT`: ``/v1/audio/transcriptions`` (default
  ``whisper-1``, the model name used by LocalAI's examples and all-in-one images).
  Batch-only: the cascade segments the audio with its VAD.
* ``tts="localai/<model>"`` — :class:`LocalAITTS`: ``/v1/audio/speech`` (default
  ``tts-1``). LocalAI answers with WAV whatever the requested format; the header is parsed
  and the audio resampled to ``sample_rate`` when the backend (e.g. Piper at 22.05 kHz)
  uses another rate.

LocalAI loads models on first use, so ``warmup()`` of the STT/TTS runs one short request.
See ``docs/providers/openai.md``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..registry import register_provider
from .openai.realtime import OpenAIRealtimeEngine
from .openai.stt import OpenAICompatibleSTT
from .openai.tts import OpenAICompatibleTTS

__all__ = ["LocalAIRealtimeEngine", "LocalAISTT", "LocalAITTS"]

DEFAULT_BASE_URL = "http://localhost:8080/v1"


@register_provider(
    "engine",
    "localai",
    description="LocalAI OpenAI-Realtime-compatible voice pipeline (local server)",
    default_model="gpt-realtime",
    local=True,
)
class LocalAIRealtimeEngine(OpenAIRealtimeEngine):
    """A LocalAI server's Realtime endpoint."""

    provider = "localai"

    def __init__(self, *, model: str | None = None, **kwargs: Any) -> None:
        super().__init__(model=model, profile="localai", **kwargs)


@register_provider(
    "stt",
    "localai",
    description="LocalAI /v1/audio/transcriptions (whisper.cpp, faster-whisper...; local)",
    default_model="whisper-1",
    requires=("httpx",),
    local=True,
)
class LocalAISTT(OpenAICompatibleSTT):
    """LocalAI file transcription (any :class:`~voice_agent_next.providers.openai.stt.
    OpenAISTT` option); the model is the name of a transcription model configured in
    LocalAI."""

    provider = "localai"
    DEFAULT_MODEL: ClassVar[str] = "whisper-1"
    DEFAULT_BASE_URL: ClassVar[str] = DEFAULT_BASE_URL
    BASE_URL_ENV: ClassVar[tuple[str, ...]] = ("LOCALAI_BASE_URL",)
    API_KEY_ENV: ClassVar[tuple[str, ...]] = ("LOCALAI_API_KEY",)
    PRELOAD_ON_WARMUP: ClassVar[bool] = True


@register_provider(
    "tts",
    "localai",
    description="LocalAI /v1/audio/speech (Piper, Kokoro, ... backends; local server)",
    default_model="tts-1",
    requires=("httpx",),
    local=True,
)
class LocalAITTS(OpenAICompatibleTTS):
    """LocalAI speech synthesis (any :class:`~voice_agent_next.providers.openai.tts.
    OpenAITTS` option); the model is the name of a TTS model configured in LocalAI, the
    voice (optional) a voice of that backend."""

    provider = "localai"
    DEFAULT_MODEL: ClassVar[str] = "tts-1"
    DEFAULT_BASE_URL: ClassVar[str] = DEFAULT_BASE_URL
    BASE_URL_ENV: ClassVar[tuple[str, ...]] = ("LOCALAI_BASE_URL",)
    API_KEY_ENV: ClassVar[tuple[str, ...]] = ("LOCALAI_API_KEY",)
    RESPONSE_FORMAT: ClassVar[str] = "wav"
    SAMPLE_RATE_PARAM: ClassVar[bool] = True
    PRELOAD_ON_WARMUP: ClassVar[bool] = True
