"""Semantic end-of-turn detection.

A :class:`TurnDetector` estimates the probability that the user has *finished* their
turn, given the recent user audio (audio models such as Smart Turn) and/or the
conversation text (text models). Engines combine it with VAD silence to decide when
to respond: respond quickly when the user is clearly done, wait longer when the
user paused mid-sentence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Literal

from .audio.frame import AudioFrame
from .chat import ChatContext
from .metrics import EOTMetrics
from .utils.clock import now
from .utils.emitter import EventEmitter

__all__ = ["TurnDetector", "TurnModality"]

TurnModality = Literal["audio", "text", "audio_text"]


class TurnDetector(ABC, EventEmitter):
    """Base class for end-of-turn models. Emits ``"metrics"`` (:class:`EOTMetrics`)."""

    provider: ClassVar[str] = "unknown"
    modality: ClassVar[TurnModality] = "text"

    def __init__(
        self,
        *,
        model: str = "",
        threshold: float = 0.5,
        sample_rate: int = 16_000,
        max_audio_duration: float = 8.0,
    ) -> None:
        EventEmitter.__init__(self)
        self.model = model
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.max_audio_duration = max_audio_duration

    async def predict_end_of_turn(
        self, *, audio: AudioFrame | None = None, chat_ctx: ChatContext | None = None
    ) -> float:
        """Probability in ``[0, 1]`` that the user's turn is complete.

        Args:
            audio: the user's current utterance (the most recent ``max_audio_duration``
                seconds are used), for audio-based detectors.
            chat_ctx: conversation so far, ending with the (partial) user message,
                for text-based detectors.
        """
        t0 = now()
        p = float(await self._predict(audio=audio, chat_ctx=chat_ctx))
        p = min(1.0, max(0.0, p))
        self.emit(
            "metrics",
            EOTMetrics(
                provider=self.provider,
                model=self.model,
                probability=p,
                threshold=self.threshold,
                inference_duration=now() - t0,
                end_of_turn=p >= self.threshold,
            ),
        )
        return p

    @abstractmethod
    async def _predict(self, *, audio: AudioFrame | None, chat_ctx: ChatContext | None) -> float:
        """Return the end-of-turn probability."""

    def supports_language(self, language: str | None) -> bool:
        return True

    async def warmup(self) -> None:
        """Load the model ahead of the first prediction (optional)."""

    async def aclose(self) -> None:
        """Release resources."""
