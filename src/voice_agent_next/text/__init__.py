"""Text utilities for speech: streaming sentence segmentation and pre-TTS filters."""

from __future__ import annotations

from .filters import normalize_whitespace, strip_emoji, strip_markdown, tts_clean
from .normalize import (
    EnglishNormalizer,
    NormalizedText,
    TextNormalizer,
    get_normalizer,
    normalize_text,
    register_normalizer,
)
from .sentences import SentenceSegmenter, split_sentences

__all__ = [
    "EnglishNormalizer",
    "NormalizedText",
    "SentenceSegmenter",
    "TextNormalizer",
    "get_normalizer",
    "normalize_text",
    "normalize_whitespace",
    "register_normalizer",
    "split_sentences",
    "strip_emoji",
    "strip_markdown",
    "tts_clean",
]
