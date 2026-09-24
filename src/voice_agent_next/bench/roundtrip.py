"""Round-trip scoring for the TTS track: WER/CER of the ASR transcript and hard-text checks.

A TTS output is transcribed by a fixed ASR and compared with the input text
(Seed-TTS-eval convention, research note 06, §5 and §8.3). Normalization and error
counting are the T2 ASR track's (:mod:`voice_agent_next.bench.text_norm`,
:mod:`voice_agent_next.bench.wer`), so a round-trip WER and an ASR WER mean the same:
Whisper's English normalizer for English (numbers to digits, spelling and contractions
unified, punctuation removed), the basic normalizer elsewhere, CER as headline for
languages written without spaces.

*Hard-text accuracy* (``hardtext_acc``) asks whether the entities of a text — numbers,
dates, times, amounts, e-mail addresses, URLs, abbreviations — survived the round trip.
Every entity lists acceptable spoken forms; one of them must appear, as whole words, in
the normalized transcript. Digit groups are compared without separators, so ``4:30``,
``4 30`` and ``four thirty`` all match.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .text_norm import cer_text, get_normalizer, normalizer_for, uses_cer
from .wer import EditCounts, char_counts, word_counts

__all__ = ["RoundTripScore", "entity_matches", "resolve_normalizer", "score_round_trip", "squash"]


def resolve_normalizer(
    language: str | None, name: str = "auto"
) -> tuple[str, Callable[[str], str]]:
    """``(resolved name, normalizer)``; ``auto`` picks Whisper English for English."""
    resolved = normalizer_for(language, name)
    return resolved, get_normalizer(resolved)


_DIGIT_GAP = re.compile(r"(?<=\d)[^a-z0-9]+(?=\d)")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def squash(normalized: str) -> str:
    """Comparison form for entities: separators between digits removed (``4:30``,
    ``4 30`` -> ``430``), anything but letters and digits turned into single spaces."""
    s = _DIGIT_GAP.sub("", normalized.lower())
    return " ".join(_NON_ALNUM.sub(" ", s).split())


def entity_matches(
    alternatives: Sequence[str], transcript: str, normalize: Callable[[str], str]
) -> bool:
    """True when one spoken form of an entity appears (as whole words) in the transcript."""
    hyp = f" {squash(normalize(transcript))} "
    for alt in alternatives:
        needle = squash(normalize(alt))
        if needle and f" {needle} " in hyp:
            return True
    return False


@dataclass(frozen=True)
class RoundTripScore:
    reference_norm: str
    transcript_norm: str
    words: EditCounts
    chars: EditCounts
    """Character counts; whitespace removed for CER languages (:func:`uses_cer`)."""
    headline: str
    """``"wer"`` or ``"cer"``."""
    entities_missed: tuple[str, ...] = ()
    """First listed form of every entity that was not found."""


def score_round_trip(
    reference: str,
    transcript: str,
    *,
    language: str | None,
    normalize: Callable[[str], str],
    entities: Sequence[Sequence[str]] = (),
) -> RoundTripScore:
    """Score one transcript of synthesized ``reference`` text."""
    ref_n, hyp_n = normalize(reference), normalize(transcript)
    return RoundTripScore(
        reference_norm=ref_n,
        transcript_norm=hyp_n,
        words=word_counts(ref_n, hyp_n),
        chars=char_counts(cer_text(ref_n, language), cer_text(hyp_n, language)),
        headline="cer" if uses_cer(language) else "wer",
        entities_missed=tuple(
            e[0] for e in entities if e and not entity_matches(e, transcript, normalize)
        ),
    )
