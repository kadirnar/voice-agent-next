"""Hallucination guard for Whisper-family recognizers.

Whisper was trained on subtitles: on noise, breathing or silence that a VAD let through it
tends to produce text anyway ("Thank you.", "Thanks for watching!", "you", a phrase
repeated until the window ends). :class:`HallucinationGuard` drops such decoded segments
using Whisper's own per-segment statistics (``no_speech_prob``, ``avg_logprob``,
``compression_ratio``), phrase lists of known artifacts, repeated n-grams and, when the
recognizer runs behind a VAD, how confident the VAD was that the utterance is speech.

It works on any object with faster-whisper's segment attributes (``text``,
``avg_logprob``, ``no_speech_prob``, ``compression_ratio``; missing ones are ignored).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ARTIFACT_PHRASES", "SUSPECT_PHRASES", "GuardVerdict", "HallucinationGuard"]

ARTIFACT_PHRASES: tuple[str, ...] = (
    # subtitle and video-outro lines Whisper learned from its training data
    "thank you for watching",
    "thanks for watching",
    "thank you so much for watching",
    "thank you very much for watching",
    "thank you for watching and see you next time",
    "thank you for watching please subscribe",
    "please subscribe",
    "please subscribe to my channel",
    "subscribe to my channel",
    "like and subscribe",
    "please like and subscribe",
    "don't forget to like and subscribe",
    "see you in the next video",
    "see you next time",
    "subtitles by the amara.org community",
    "transcription by castingwords",
    "transcription by esoteric",
    "ご視聴ありがとうございました",
    "untertitel im auftrag des zdf für funk 2017",
    "untertitel der amara.org community",
    "sous-titrage société radio-canada",
    "sous-titres réalisés par la communauté d'amara.org",
    "продолжение следует",
    "субтитры сделал dimatorzok",
    "字幕由amara.org社区提供",
    "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目",
)
"""Whole-segment texts that are dropped whenever they are decoded (compared after
:func:`normalize`: lower case, punctuation removed)."""

SUSPECT_PHRASES: tuple[str, ...] = (
    "you",
    "thank you",
    "thank you very much",
    "thank you so much",
    "thanks",
    "bye",
    "bye bye",
    "the end",
    "so",
    "oh",
    "hmm",
    "uh",
    "um",
)
"""Whole-segment texts that are real words people say but also Whisper's most common
noise outputs: dropped only when there is other evidence of non-speech (see
:attr:`HallucinationGuard.suspect_no_speech_threshold`)."""


@dataclass(frozen=True, slots=True)
class GuardVerdict:
    """What the guard kept and why it dropped the rest."""

    kept: list[Any]
    dropped: list[tuple[str, str]] = field(default_factory=list)
    """``(segment text, reason)`` for every dropped segment."""


@dataclass(frozen=True, slots=True)
class HallucinationGuard:
    """Drops Whisper segments that are probably not speech. Every rule can be disabled
    with ``None`` (or an empty tuple for the phrase lists).

    A segment is dropped when:

    1. it has no word characters (``"..."``, ``"♪"``), unless :attr:`drop_empty` is off;
    2. its normalized text is in :attr:`artifacts`;
    3. its normalized text is in :attr:`suspects` and there is other evidence of
       non-speech: ``no_speech_prob >= suspect_no_speech_threshold``, ``avg_logprob <
       suspect_log_prob_threshold`` or a weak VAD;
    4. ``no_speech_prob >= no_speech_threshold`` and (``avg_logprob <
       log_prob_threshold`` or a weak VAD) — Whisper's own rule, extended with the VAD;
    5. ``compression_ratio > compression_ratio_threshold`` (a repetition loop that
       Whisper's temperature fallback did not fix).

    Repetition loops inside a kept segment (an n-gram of up to :attr:`max_ngram` words
    repeated more than :attr:`max_ngram_repeats` times in a row) are cut back to one
    occurrence.

    "Weak VAD" means the recognizer runs behind a VAD
    (:class:`~voice_agent_next.stt.StreamAdapter`) and the mean speech probability of the
    utterance's speech windows is below :attr:`vad_threshold`: the VAD itself was unsure.
    """

    no_speech_threshold: float | None = 0.6
    log_prob_threshold: float | None = -1.0
    compression_ratio_threshold: float | None = 2.4
    suspect_no_speech_threshold: float | None = 0.2
    suspect_log_prob_threshold: float | None = -0.8
    vad_threshold: float | None = 0.5
    max_ngram: int = 4
    max_ngram_repeats: int | None = 4
    artifacts: tuple[str, ...] = ARTIFACT_PHRASES
    suspects: tuple[str, ...] = SUSPECT_PHRASES
    drop_empty: bool = True

    def __post_init__(self) -> None:
        if self.max_ngram < 1:
            raise ValueError("max_ngram must be >= 1")
        if self.max_ngram_repeats is not None and self.max_ngram_repeats < 1:
            raise ValueError("max_ngram_repeats must be >= 1")
        # stored normalized (frozen dataclass: bypass __setattr__)
        object.__setattr__(self, "artifacts", tuple(normalize(p) for p in self.artifacts))
        object.__setattr__(self, "suspects", tuple(normalize(p) for p in self.suspects))

    @classmethod
    def disabled(cls) -> HallucinationGuard:
        """A guard that keeps everything."""
        return cls(
            no_speech_threshold=None,
            log_prob_threshold=None,
            compression_ratio_threshold=None,
            suspect_no_speech_threshold=None,
            suspect_log_prob_threshold=None,
            vad_threshold=None,
            max_ngram_repeats=None,
            artifacts=(),
            suspects=(),
            drop_empty=False,
        )

    def weak_vad(self, vad_confidence: float | None) -> bool:
        return (
            self.vad_threshold is not None
            and vad_confidence is not None
            and vad_confidence < self.vad_threshold
        )

    def reason(self, segment: Any, *, vad_confidence: float | None = None) -> str | None:
        """Why ``segment`` should be dropped, or ``None`` to keep it."""
        text = normalize(str(getattr(segment, "text", "")))
        no_speech = _number(segment, "no_speech_prob")
        logprob = _number(segment, "avg_logprob")
        ratio = _number(segment, "compression_ratio")
        weak = self.weak_vad(vad_confidence)
        if not text and self.drop_empty:
            return "no words"
        if text in self.artifacts:
            return "known artifact"
        if text in self.suspects and (
            weak
            or _at_least(no_speech, self.suspect_no_speech_threshold)
            or _below(logprob, self.suspect_log_prob_threshold)
        ):
            return "suspect phrase on weak evidence"
        if _at_least(no_speech, self.no_speech_threshold) and (
            weak or _below(logprob, self.log_prob_threshold)
        ):
            return "no speech"
        threshold = self.compression_ratio_threshold
        if ratio is not None and threshold is not None and ratio > threshold:
            return "repetitive"
        return None

    def filter(
        self, segments: Sequence[Any], *, vad_confidence: float | None = None
    ) -> GuardVerdict:
        """Split ``segments`` into kept and dropped ones."""
        kept: list[Any] = []
        dropped: list[tuple[str, str]] = []
        for segment in segments:
            why = self.reason(segment, vad_confidence=vad_confidence)
            if why is None:
                kept.append(segment)
            else:
                dropped.append((str(getattr(segment, "text", "")).strip(), why))
        return GuardVerdict(kept, dropped)

    def clean_text(self, text: str) -> str:
        """``text`` with repetition loops cut back to one occurrence."""
        if self.max_ngram_repeats is None:
            return text
        return collapse_repeats(text, max_n=self.max_ngram, max_repeats=self.max_ngram_repeats)


_PUNCT = re.compile(r"[^\w\s']+")
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lower case, punctuation (except apostrophes) removed, whitespace collapsed."""
    return _SPACE.sub(" ", _PUNCT.sub(" ", text.lower().replace("_", " "))).strip()


def collapse_repeats(text: str, *, max_n: int = 4, max_repeats: int = 4) -> str:
    """Cut runs of an n-gram (``n <= max_n`` words) repeated more than ``max_repeats``
    times in a row back to a single occurrence. Words are compared normalized; the
    original spelling of the kept words is preserved."""
    words = text.split()
    keys = [normalize(w) for w in words]
    i = 0
    out: list[str] = []
    while i < len(words):
        best_n, best_count = 0, 0
        for n in range(1, max_n + 1):
            if i + n > len(words) or not all(keys[i : i + n]):
                break
            gram = keys[i : i + n]
            count = 1
            while keys[i + count * n : i + (count + 1) * n] == gram:
                count += 1
            if count > max_repeats and count * n > best_count * best_n:
                best_n, best_count = n, count
        if best_n:
            out.extend(words[i : i + best_n])
            i += best_n * best_count
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


def _number(segment: Any, name: str) -> float | None:
    value = getattr(segment, name, None)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _at_least(value: float | None, threshold: float | None) -> bool:
    return value is not None and threshold is not None and value >= threshold


def _below(value: float | None, threshold: float | None) -> bool:
    return value is not None and threshold is not None and value < threshold
