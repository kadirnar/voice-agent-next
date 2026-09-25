"""Semantic end-of-turn detection.

A :class:`TurnDetector` estimates the probability that the user has *finished* their
turn, given the recent user audio (audio models such as Smart Turn) and/or the
conversation text (text models). Engines combine it with VAD silence to decide when
to respond: respond quickly when the user is clearly done, wait longer when the
user paused mid-sentence.

:class:`FusedTurnDetector` combines an audio and a text detector (:func:`fuse_end_of_turn`):
the audio model hears a finished sentence, the text model reads whether the words and the
conversation call for more. The cascade runs its audio half concurrently with the STT
flush and its text half on the transcript (see ``docs/concepts/turn-taking.md``).
"""

from __future__ import annotations

import asyncio
import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

from .audio.frame import AudioFrame
from .chat import ChatContext
from .metrics import EOTMetrics
from .utils.clock import now
from .utils.emitter import EventEmitter
from .utils.log import logger

__all__ = [
    "FusedTurnDetector",
    "FusionMethod",
    "TurnDetector",
    "TurnModality",
    "fuse_end_of_turn",
    "turn_text",
]

TurnModality = Literal["audio", "text", "audio_text"]
FusionMethod = Literal["logit", "product", "min", "mean"]
_EPS = 1e-6


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
        return self._report(p, now() - t0)

    def _report(self, probability: float, duration: float) -> float:
        """Clamp ``probability`` to ``[0, 1]`` and emit its :class:`EOTMetrics`."""
        p = min(1.0, max(0.0, probability))
        self.emit(
            "metrics",
            EOTMetrics(
                provider=self.provider,
                model=self.model,
                probability=p,
                threshold=self.threshold,
                inference_duration=duration,
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


def turn_text(chat_ctx: ChatContext | None) -> tuple[str, str]:
    """``(agent, user)``: what a text detector judges — the user's current (last) message,
    and the last agent message before it (``""`` when missing)."""
    if chat_ctx is None:
        return "", ""
    items = chat_ctx.messages()
    user_at = next((i for i in range(len(items) - 1, -1, -1) if items[i].role == "user"), None)
    if user_at is None:
        return "", ""
    agent = next(
        (m.text for m in reversed(items[:user_at]) if m.role == "assistant" and m.text.strip()),
        "",
    )
    return agent.strip(), items[user_at].text.strip()


def _logit(p: float) -> float:
    p = min(1.0 - _EPS, max(_EPS, p))
    return math.log(p / (1.0 - p))


def fuse_end_of_turn(
    audio: float | None,
    text: float | None,
    *,
    method: FusionMethod = "logit",
    audio_weight: float = 1.0,
    text_weight: float = 1.0,
    bias: float = 0.0,
) -> float | None:
    """Combine an audio and a text end-of-turn probability.

    Either may be ``None`` (no audio, no transcript yet, or the text detector missed its
    latency budget): the other one is returned as is (``None`` when both are).

    ``method``:

    * ``"logit"``: ``sigmoid(audio_weight * logit(audio) + text_weight * logit(text) +
      bias)``, a logistic regression over the two scores. With weights 1 and bias 0 it is
      the naive-Bayes combination of two independent, calibrated opinions.
    * ``"product"``: ``audio ** audio_weight * text ** text_weight`` (both must agree
      that the user is done).
    * ``"min"``: the less confident of the two (weights ignored).
    * ``"mean"``: the weighted mean.
    """
    if audio is None or text is None:
        return text if audio is None else audio
    a, t = min(1.0, max(0.0, audio)), min(1.0, max(0.0, text))
    if method == "logit":
        x = audio_weight * _logit(a) + text_weight * _logit(t) + bias
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))
    if method == "product":
        return float(a**audio_weight * t**text_weight)
    if method == "min":
        return min(a, t)
    if method == "mean":
        total = audio_weight + text_weight
        if total <= 0:
            raise ValueError("mean fusion needs a positive weight")
        return (audio_weight * a + text_weight * t) / total
    raise ValueError(f"unknown fusion method {method!r}")


class FusedTurnDetector(TurnDetector):
    """An audio and a text end-of-turn detector, fused (:func:`fuse_end_of_turn`).

    Smart Turn (audio) is confident at a pause after any complete-sounding sentence; a
    text detector reads the words ("I would like to book a table," is not done) in the
    context of the agent's last turn. In the cascade the audio half runs concurrently
    with the STT flush and the text half starts on the interim transcript, so fusing
    adds little or no latency.

    Args:
        audio: the audio detector (spec or instance), e.g. ``"smart_turn"``.
        text: the text detector (spec or instance), e.g. ``"lm_turn"`` or
            ``{"provider": "llm_turn", "model": "ollama/qwen3.5:4b"}``.
        method, audio_weight, text_weight, bias: see :func:`fuse_end_of_turn`. The
            defaults were fitted on eot-bench English (Smart Turn v3.2 int8 and
            ``lm_turn``; see ``docs/providers/lm-turn.md``).
        threshold: fused probability at/above which the turn counts as complete.
        text_timeout: latency budget of the text detector, in seconds from the start of
            its prediction. Past it, the audio verdict is used alone. ``None``: no budget.
    """

    provider = "fused"
    modality = "audio_text"

    def __init__(
        self,
        *,
        audio: Any = "smart_turn",
        text: Any = "lm_turn",
        model: str | None = None,
        method: FusionMethod = "logit",
        audio_weight: float = 0.4,
        text_weight: float = 0.9,
        bias: float = 0.1,
        threshold: float = 0.5,
        text_timeout: float | None = 0.5,
    ) -> None:
        from .registry import create

        if method not in ("logit", "product", "min", "mean"):
            raise ValueError(f"unknown fusion method {method!r}")
        if audio_weight < 0 or text_weight < 0:
            raise ValueError("fusion weights must be >= 0")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.audio: TurnDetector = create("turn", audio)
        self.text: TurnDetector = create("turn", text)
        if self.audio.modality != "audio":
            raise ValueError(f"audio= needs an audio turn detector, got {self.audio.provider}")
        if self.text.modality != "text":
            raise ValueError(f"text= needs a text turn detector, got {self.text.provider}")
        super().__init__(
            model=model or f"{self.audio.model}+{self.text.model}",
            threshold=threshold,
            sample_rate=self.audio.sample_rate,
            max_audio_duration=self.audio.max_audio_duration,
        )
        self.method: FusionMethod = method
        self.audio_weight = audio_weight
        self.text_weight = text_weight
        self.bias = bias
        self.text_timeout = text_timeout
        for part in (self.audio, self.text):
            part.on("metrics", self._forward_metrics)

    def _forward_metrics(self, m: EOTMetrics) -> None:
        self.emit("metrics", m)

    async def predict_audio(self, audio: AudioFrame | None) -> float | None:
        """The audio detector's probability (``None`` without audio)."""
        if audio is None or not audio:
            return None
        return await self.audio.predict_end_of_turn(audio=audio)

    async def predict_text(self, chat_ctx: ChatContext | None) -> float | None:
        """The text detector's probability within ``text_timeout``: ``None`` without a user
        transcript, past the budget, or when the text detector fails (logged)."""
        if not turn_text(chat_ctx)[1]:
            return None
        try:
            return await asyncio.wait_for(
                self.text.predict_end_of_turn(chat_ctx=chat_ctx), self.text_timeout
            )
        except TimeoutError:
            logger.debug("%s: text end of turn over its %ss budget", self.model, self.text_timeout)
        except Exception:
            logger.warning("text end-of-turn detector %s failed", self.text.provider, exc_info=True)
        return None

    def fuse(self, audio: float | None, text: float | None) -> float:
        """The fused probability; ``1.0`` when neither half had anything to judge (silence
        decides, as with any detector that has no input)."""
        p = fuse_end_of_turn(
            audio,
            text,
            method=self.method,
            audio_weight=self.audio_weight,
            text_weight=self.text_weight,
            bias=self.bias,
        )
        return 1.0 if p is None else p

    def report(self, probability: float, duration: float) -> float:
        """Emit the metrics of a fused verdict whose halves the caller ran itself."""
        return self._report(probability, duration)

    async def _predict(self, *, audio: AudioFrame | None, chat_ctx: ChatContext | None) -> float:
        pa, pt = await asyncio.gather(self.predict_audio(audio), self.predict_text(chat_ctx))
        return self.fuse(pa, pt)

    def supports_language(self, language: str | None) -> bool:
        return self.audio.supports_language(language) and self.text.supports_language(language)

    async def warmup(self) -> None:
        await asyncio.gather(self.audio.warmup(), self.text.warmup())

    async def aclose(self) -> None:
        await asyncio.gather(self.audio.aclose(), self.text.aclose(), return_exceptions=True)
