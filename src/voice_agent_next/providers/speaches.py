"""Speaches — a local OpenAI-compatible speech server (faster-whisper, Kokoro, Piper).

* ``AgentSession("speaches/<llm-model>")`` — :class:`SpeachesRealtimeEngine`, the OpenAI
  Realtime engine with the ``speaches`` profile (``ws://localhost:8000/v1`` or
  ``SPEACHES_BASE_URL``; ``SPEACHES_API_KEY`` if set). Speaches runs a VAD -> STT -> LLM ->
  TTS pipeline behind the beta Realtime dialect and supports neither ``response.cancel``
  nor ``conversation.item.truncate``, so the engine declares ``truncation=False`` and
  barge-in only stops local playback. See ``docs/providers/openai-realtime.md``.
* ``stt="speaches/<whisper-model>"`` — :class:`SpeachesSTT`: ``/v1/audio/transcriptions``
  (default ``Systran/faster-distil-whisper-small.en``). Batch-only: the cascade segments the
  audio with its VAD.
* ``tts="speaches/<tts-model>"`` — :class:`SpeachesTTS`: ``/v1/audio/speech`` with raw PCM
  at the requested ``sample_rate`` (default ``speaches-ai/Kokoro-82M-v1.0-ONNX``, voice
  ``af_heart``).

Speaches loads models on first use, so ``warmup()`` of the STT/TTS runs one short request.
See ``docs/providers/openai.md``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..registry import register_provider
from .openai.realtime import OpenAIRealtimeEngine
from .openai.stt import OpenAICompatibleSTT
from .openai.tts import OpenAICompatibleTTS

__all__ = ["SpeachesRealtimeEngine", "SpeachesSTT", "SpeachesTTS"]

DEFAULT_BASE_URL = "http://localhost:8000/v1"


@register_provider(
    "engine",
    "speaches",
    description="Speaches OpenAI-Realtime-compatible voice pipeline (local server)",
    local=True,
)
class SpeachesRealtimeEngine(OpenAIRealtimeEngine):
    """A Speaches server's Realtime endpoint (the model is its conversation LLM)."""

    provider = "speaches"

    def __init__(self, *, model: str | None = None, **kwargs: Any) -> None:
        super().__init__(model=model, profile="speaches", **kwargs)


@register_provider(
    "stt",
    "speaches",
    description="Speaches /v1/audio/transcriptions (faster-whisper; local server)",
    default_model="Systran/faster-distil-whisper-small.en",
    models=(
        "Systran/faster-distil-whisper-small.en",
        "Systran/faster-whisper-small",
        "Systran/faster-whisper-large-v3",
        "deepdml/faster-whisper-large-v3-turbo-ct2",
    ),
    requires=("httpx",),
    local=True,
)
class SpeachesSTT(OpenAICompatibleSTT):
    """Speaches file transcription (any :class:`~voice_agent_next.providers.openai.stt.
    OpenAISTT` option). English-only by default: use ``Systran/faster-whisper-small`` or a
    larger multilingual model for other languages."""

    provider = "speaches"
    DEFAULT_MODEL: ClassVar[str] = "Systran/faster-distil-whisper-small.en"
    DEFAULT_BASE_URL: ClassVar[str] = DEFAULT_BASE_URL
    BASE_URL_ENV: ClassVar[tuple[str, ...]] = ("SPEACHES_BASE_URL",)
    API_KEY_ENV: ClassVar[tuple[str, ...]] = ("SPEACHES_API_KEY",)
    PRELOAD_ON_WARMUP: ClassVar[bool] = True


@register_provider(
    "tts",
    "speaches",
    description="Speaches /v1/audio/speech (Kokoro, Piper; local server)",
    default_model="speaches-ai/Kokoro-82M-v1.0-ONNX",
    models=("speaches-ai/Kokoro-82M-v1.0-ONNX", "speaches-ai/piper-en_US-amy-medium"),
    requires=("httpx",),
    local=True,
)
class SpeachesTTS(OpenAICompatibleTTS):
    """Speaches speech synthesis (any :class:`~voice_agent_next.providers.openai.tts.
    OpenAITTS` option). The server resamples to ``sample_rate`` (default 24 kHz), so Piper
    voices work at any rate."""

    provider = "speaches"
    DEFAULT_MODEL: ClassVar[str] = "speaches-ai/Kokoro-82M-v1.0-ONNX"
    DEFAULT_VOICE: ClassVar[str | None] = "af_heart"
    DEFAULT_BASE_URL: ClassVar[str] = DEFAULT_BASE_URL
    BASE_URL_ENV: ClassVar[tuple[str, ...]] = ("SPEACHES_BASE_URL",)
    API_KEY_ENV: ClassVar[tuple[str, ...]] = ("SPEACHES_API_KEY",)
    SAMPLE_RATE_PARAM: ClassVar[bool] = True
    PRELOAD_ON_WARMUP: ClassVar[bool] = True
