"""Text clean-up before speech synthesis (markdown, emoji, whitespace)."""

from __future__ import annotations

import re

__all__ = ["normalize_whitespace", "strip_emoji", "strip_markdown", "tts_clean"]

_CODE_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\n?(.*?)```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_BOLD = re.compile(r"(\*\*|__)(.+?)\1")
_ITALIC = re.compile(r"(?<![\w*])([*_])(?!\s)(.+?)(?<!\s)\1(?![\w*])")
_STRIKE = re.compile(r"~~(.+?)~~")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_BULLET = re.compile(r"^\s*[-*+•]\s+", re.MULTILINE)
_NUMBERED = re.compile(r"^\s*(\d+)[.)]\s+", re.MULTILINE)
_HR = re.compile(r"^\s*([-*_]\s*){3,}$", re.MULTILINE)
_TABLE_SEP = re.compile(r"^\s*\|?(\s*:?-+:?\s*\|)+\s*:?-*:?\s*$", re.MULTILINE)
_EMOJI = re.compile(
    "["
    "\U0001f300-\U0001faff"  # symbols & pictographs, emoticons, transport, supplemental
    "\U00002600-\U000027bf"  # misc symbols, dingbats
    "\U0001f000-\U0001f2ff"  # mahjong, playing cards, enclosed
    "\U0000fe0f\U0000200d"  # variation selector, zero-width joiner
    "]+"
)
_WS = re.compile(r"[ \t\f\v]+")


def strip_markdown(text: str) -> str:
    """Remove markdown syntax while keeping the readable text."""
    text = _CODE_FENCE.sub(lambda m: m.group(1), text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _IMAGE.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _BOLD.sub(r"\2", text)
    text = _STRIKE.sub(r"\1", text)
    text = _ITALIC.sub(r"\2", text)
    text = _HR.sub("", text)
    text = _TABLE_SEP.sub("", text)
    text = _HEADING.sub("", text)
    text = _BLOCKQUOTE.sub("", text)
    text = _BULLET.sub("", text)
    text = _NUMBERED.sub(r"\1. ", text)
    return text.replace("|", " ")


def strip_emoji(text: str) -> str:
    return _EMOJI.sub("", text)


def normalize_whitespace(text: str) -> str:
    return _WS.sub(" ", text).strip()


def tts_clean(text: str) -> str:
    """Default pre-TTS filter: markdown + emoji removal and whitespace normalization."""
    return normalize_whitespace(strip_emoji(strip_markdown(text)))
