"""Transcript normalization for the ASR track (research note 06, §4.1 and §8.3).

WER is only comparable after both the reference and the hypothesis are normalized the
same way. The track follows the Open ASR Leaderboard convention:

* **English** — Whisper's ``EnglishTextNormalizer``: lower case, bracketed/parenthesized
  spans and fillers (``uh``, ``um``, ``hmm``...) removed, contractions and titles
  expanded (``won't`` -> ``will not``, ``Mr`` -> ``mister``), spelled-out numbers
  converted to digits (``twenty one`` -> ``21``, ``$20 million`` -> ``$20000000``),
  British spellings mapped to American ones, punctuation and diacritics removed.
* **Other languages** — Whisper's ``BasicTextNormalizer`` with combining marks preserved
  (``preserve_marks``, so Indic/Thai vowel signs are not turned into spaces): lower case,
  bracketed spans removed, punctuation and symbols replaced by spaces. Numbers are *not*
  normalized, so "1940" vs "nineteen forty" counts as an error outside English.
* **Languages written without spaces** (Chinese, Japanese, Korean, Thai, Lao, Burmese,
  Khmer...) — the basic normalizer, then CER on the text with all whitespace removed. CER
  is their headline metric.

The Whisper normalizers are vendored in :mod:`._whisper_normalizer` (MIT), so no extra
dependency is needed.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Literal

from ._whisper_normalizer import BasicTextNormalizer, EnglishTextNormalizer

__all__ = [
    "CER_LANGUAGES",
    "NORMALIZERS",
    "NormalizerName",
    "base_language",
    "cer_text",
    "get_normalizer",
    "normalizer_for",
    "uses_cer",
]

NormalizerName = Literal["auto", "whisper-english", "whisper-basic", "none"]
NORMALIZERS: tuple[str, ...] = ("auto", "whisper-english", "whisper-basic", "none")

CER_LANGUAGES: frozenset[str] = frozenset(
    {"zh", "ja", "ko", "th", "yue", "lo", "my", "km", "bo", "dz", "wuu", "nan"}
)
"""Languages whose headline metric is CER (written without spaces between words, or —
Korean — conventionally scored by character)."""

_ALIASES = {"cmn": "zh", "jpn": "ja", "kor": "ko", "tha": "th", "eng": "en", "zho": "zh"}


def base_language(language: str | None) -> str | None:
    """``"en-US"`` / ``"en_us"`` / ``"cmn_hans_cn"`` -> ``"en"`` / ``"en"`` / ``"zh"``."""
    if not language:
        return None
    base = language.strip().lower().replace("_", "-").split("-")[0]
    return _ALIASES.get(base, base) or None


def uses_cer(language: str | None) -> bool:
    """True when CER (not WER) is the headline metric for ``language``."""
    return base_language(language) in CER_LANGUAGES


def _identity(text: str) -> str:
    return " ".join(text.split())


@cache
def get_normalizer(name: str) -> Callable[[str], str]:
    """A normalizer by name: ``whisper-english``, ``whisper-basic`` or ``none``."""
    if name == "whisper-english":
        english = EnglishTextNormalizer()
        return lambda text: english(text).strip()
    if name == "whisper-basic":
        basic = BasicTextNormalizer(preserve_marks=True)
        return lambda text: basic(text).strip()
    if name == "none":
        return _identity
    raise ValueError(f"unknown normalizer {name!r}; use one of {', '.join(NORMALIZERS)}")


def normalizer_for(language: str | None, name: str = "auto") -> str:
    """Resolve ``auto``: Whisper English for ``en`` (and unknown languages), basic otherwise."""
    if name != "auto":
        get_normalizer(name)  # validate
        return name
    base = base_language(language)
    return "whisper-english" if base in (None, "en") else "whisper-basic"


def cer_text(text: str, language: str | None) -> str:
    """Text scored by CER: whitespace removed for :data:`CER_LANGUAGES`, collapsed otherwise."""
    if uses_cer(language):
        return "".join(text.split())
    return " ".join(text.split())
