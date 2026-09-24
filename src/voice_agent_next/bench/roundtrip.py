"""Round-trip scoring for the TTS track: normalization, WER/CER and hard-text checks.

A TTS output is transcribed by a fixed ASR and compared with the input text
(Seed-TTS-eval convention, research note 06, §5 and §8.3). Both sides must be
normalized the same way first, or number formatting alone ("42" vs "forty two")
dominates the error rate.

The normalizer is chosen through one small interface (:class:`TextNormalizer`):

* ``whisper-english`` — Whisper's ``EnglishTextNormalizer`` from the T2 ASR track
  (:mod:`voice_agent_next.bench.text_norm`), used automatically when that module is
  available;
* ``basic-english`` — the built-in fallback (:class:`BasicEnglishNormalizer`): lower case,
  spelled-out cardinals to digits (``forty-two`` -> ``42``, ``two thousand twenty five``
  -> ``2025``), thousands separators dropped, ``%``/``&``/``@`` spelled out, punctuation
  removed. It does not expand abbreviations or read currencies, so it is stricter than
  the Whisper normalizer; the manifest records which one ran.

*Hard-text accuracy* (``hardtext_acc``) asks whether the entities of a text — numbers,
dates, times, amounts, e-mail addresses, URLs, abbreviations — survived the round trip.
Every entity lists acceptable spoken forms; one of them must appear in the normalized
transcript (digit groups are compared without separators, so ``4:30``, ``4 30`` and
``four thirty`` all match).
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Protocol, runtime_checkable

__all__ = [
    "NORMALIZERS",
    "BasicEnglishNormalizer",
    "EditCounts",
    "TextNormalizer",
    "char_counts",
    "edit_distance",
    "entity_matches",
    "get_normalizer",
    "squash",
    "word_counts",
]

NORMALIZERS: tuple[str, ...] = ("auto", "whisper-english", "basic-english", "none")


@runtime_checkable
class TextNormalizer(Protocol):
    """Maps a reference text or a transcript to the canonical form that is scored."""

    name: str

    def __call__(self, text: str) -> str: ...


# ---------------------------------------------------------------- basic English normalizer

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}  # fmt: skip
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}  # fmt: skip
_SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_NUMBER_WORDS = {*_UNITS, *_TENS, "hundred", *_SCALES}


def _words_to_numbers(tokens: Sequence[str]) -> list[str]:
    """Replace runs of spelled-out cardinals with digits.

    A new number starts when a word cannot continue the current one, so digit-by-digit
    readings stay separate (``five five five`` -> ``5 5 5``) and so do year-style pairs
    (``twenty twenty five`` -> ``20 25``). ``and`` is absorbed inside a number
    (``one hundred and five`` -> ``105``).
    """
    out: list[str] = []
    total = current = 0
    active = False
    last = ""  # "unit" | "teen" | "tens" | "hundred" | "scale"
    last_scale = 0
    pending_and = False

    def close() -> None:
        nonlocal total, current, active, last, last_scale, pending_and
        if active:
            out.append(str(total + current))
            if pending_and:
                out.append("and")
        total = current = last_scale = 0
        active = False
        last = ""
        pending_and = False

    for tok in tokens:
        if tok == "and" and active and last in ("hundred", "scale"):
            pending_and = True
            continue
        if tok not in _NUMBER_WORDS:
            close()
            out.append(tok)
            continue
        if tok in _UNITS:
            value = _UNITS[tok]
            kind = "unit" if value < 10 else "teen"
            # continues after "twenty" (unit only), "hundred" or a scale word
            if active and not (
                (last == "tens" and kind == "unit") or last in ("hundred", "scale")
            ):
                close()
            current += value
            last = kind
        elif tok in _TENS:
            if active and last not in ("hundred", "scale"):
                close()
            current += _TENS[tok]
            last = "tens"
        elif tok == "hundred":
            if not active or last not in ("unit", "teen") or current >= 100:
                close()
                current = 1
            current *= 100
            last = "hundred"
        else:  # thousand / million / billion
            scale = _SCALES[tok]
            # "thousand thousand" or "two million three million": start a new number
            if active and (current == 0 or (last_scale and scale >= last_scale)):
                close()
            if not active:
                current = 1
            total += current * scale
            current = 0
            last = "scale"
            last_scale = scale
        active = True
        pending_and = False
    close()
    return out


_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_KEEP = re.compile(r"[^a-z0-9.:' ]+")
_INNER_POINT = re.compile(r"(?<=\d)[.:](?=\d)")


class BasicEnglishNormalizer:
    """Dependency-free English normalizer (fallback when the Whisper one is missing)."""

    name = "basic-english"

    def __call__(self, text: str) -> str:
        s = text.lower().replace("’", "'").replace("‘", "'")
        s = _THOUSANDS.sub("", s)
        s = s.replace("%", " percent ").replace("&", " and ").replace("@", " at ")
        s = s.replace("-", " ").replace("/", " ")
        s = _KEEP.sub(" ", s)
        # keep "." and ":" only between digits (2.5, 4:30); apostrophes are dropped
        s = _INNER_POINT.sub(lambda m: "\x00" if m.group(0) == "." else "\x01", s)
        s = s.replace(".", " ").replace(":", " ").replace("'", "")
        s = s.replace("\x00", ".").replace("\x01", ":")
        return " ".join(_words_to_numbers(s.split()))


def _identity(text: str) -> str:
    return " ".join(text.split())


@dataclass(frozen=True)
class _Named:
    name: str
    fn: Callable[[str], str]

    def __call__(self, text: str) -> str:
        return self.fn(text)


def _whisper_english() -> TextNormalizer | None:
    """The T2 track's Whisper English normalizer, when this installation has it."""
    try:
        module = importlib.import_module("voice_agent_next.bench.text_norm")
    except ImportError:
        return None
    getter = getattr(module, "get_normalizer", None)
    if getter is None:
        return None
    try:
        return _Named("whisper-english", getter("whisper-english"))
    except Exception:
        return None


@cache
def get_normalizer(name: str = "auto") -> TextNormalizer:
    """``auto`` (Whisper English if available, else basic), ``whisper-english``,
    ``basic-english`` or ``none`` (whitespace only)."""
    if name == "auto":
        return _whisper_english() or BasicEnglishNormalizer()
    if name == "whisper-english":
        whisper = _whisper_english()
        if whisper is None:
            raise ValueError(
                "the whisper-english normalizer is not available in this installation "
                "(it ships with the T2 ASR track); use 'basic-english'"
            )
        return whisper
    if name == "basic-english":
        return BasicEnglishNormalizer()
    if name == "none":
        return _Named("none", _identity)
    raise ValueError(f"unknown normalizer {name!r}; use one of {', '.join(NORMALIZERS)}")


# --------------------------------------------------------------------------- error rates


@dataclass(frozen=True, slots=True)
class EditCounts:
    """Minimum edits turning a reference into a hypothesis."""

    errors: int = 0
    """Substitutions + deletions + insertions."""
    ref_len: int = 0

    @property
    def rate(self) -> float | None:
        """``errors / ref_len``; ``None`` for an empty reference."""
        return self.errors / self.ref_len if self.ref_len else None

    def __add__(self, other: EditCounts) -> EditCounts:
        return EditCounts(self.errors + other.errors, self.ref_len + other.ref_len)


def edit_distance(ref: Sequence[object], hyp: Sequence[object]) -> int:
    """Levenshtein distance (unit costs) between two token sequences."""
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        row = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            row[j] = min(prev[j - 1] + (r != h), prev[j] + 1, row[j - 1] + 1)
        prev = row
    return prev[-1]


def word_counts(ref: str, hyp: str) -> EditCounts:
    """Word-level edits between two normalized strings."""
    r, h = ref.split(), hyp.split()
    return EditCounts(edit_distance(r, h), len(r))


def char_counts(ref: str, hyp: str) -> EditCounts:
    """Character-level edits between two normalized strings (whitespace collapsed)."""
    r, h = " ".join(ref.split()), " ".join(hyp.split())
    return EditCounts(edit_distance(r, h), len(r))


# ------------------------------------------------------------------------ hard-text check

_DIGIT_GAP = re.compile(r"(?<=\d)[^a-z0-9]+(?=\d)")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def squash(normalized: str) -> str:
    """Comparison form for entities: separators between digits removed (``4:30``,
    ``4 30`` -> ``430``), everything else except letters and digits turned into single
    spaces."""
    s = _DIGIT_GAP.sub("", normalized.lower())
    return " ".join(_NON_ALNUM.sub(" ", s).split())


def entity_matches(
    alternatives: Sequence[str], transcript: str, normalizer: TextNormalizer
) -> bool:
    """True when one spoken form of an entity appears (as whole words) in the transcript."""
    hyp = f" {squash(normalizer(transcript))} "
    for alt in alternatives:
        needle = squash(normalizer(alt))
        if needle and f" {needle} " in hyp:
            return True
    return False
