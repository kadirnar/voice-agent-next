"""Incremental sentence/clause segmentation of streamed LLM text for TTS.

TTS engines sound best with complete sentences, but waiting for the whole LLM reply
adds latency. :class:`SentenceSegmenter` releases each sentence as soon as it is
complete, merges very short fragments, and force-splits over-long run-on text at
clause boundaries so the first audio starts quickly.
"""

from __future__ import annotations

import re

__all__ = ["SentenceSegmenter", "split_sentences"]

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "u.s",
    "u.k", "a.m", "p.m", "no", "inc", "ltd", "co", "corp", "fig", "approx", "dept", "est",
    "mt", "ave", "blvd", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
}  # fmt: skip

# terminal punctuation (+ optional closing quotes/brackets) followed by whitespace
_LATIN_END = re.compile(r"([.!?…]+)([\"'”’)\]]*)(\s+)")
# CJK / full-width terminal punctuation needs no trailing whitespace
_CJK_END = re.compile(r"([。！？｡]+)([」』”’）]*)")
_PARAGRAPH = re.compile(r"\n\s*\n|\n(?=\s*(?:[-*•]|\d+[.)])\s)")
_CLAUSE = re.compile(r"[,;:—–]\s+|\s+-\s+")


def _is_abbreviation(text: str, dot_index: int) -> bool:
    """True if the '.' at ``dot_index`` ends an abbreviation or an initial."""
    j = dot_index - 1
    while j >= 0 and (text[j].isalpha() or text[j] == "."):
        j -= 1
    word = text[j + 1 : dot_index].lower()
    if not word:
        return False
    if word in _ABBREVIATIONS:
        return True
    # single letter initials: "J. K. Rowling"
    return len(word) == 1 and text[dot_index - 1].isupper()


class SentenceSegmenter:
    """Feed text deltas with :meth:`push`; get complete segments back.

    Args:
        min_chars: segments shorter than this are merged with the following text
            (except at :meth:`flush`).
        max_chars: when the buffer grows beyond this without a sentence end, split at
            the last clause boundary (or whitespace).
        first_segment_min_chars: optional smaller minimum for the very first segment,
            to start speaking sooner.
    """

    def __init__(
        self,
        *,
        min_chars: int = 10,
        max_chars: int = 220,
        first_segment_min_chars: int | None = None,
    ) -> None:
        if max_chars <= min_chars:
            raise ValueError("max_chars must be greater than min_chars")
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.first_segment_min_chars = first_segment_min_chars
        self._buf = ""
        self._emitted = 0

    def _min(self) -> int:
        if self._emitted == 0 and self.first_segment_min_chars is not None:
            return self.first_segment_min_chars
        return self.min_chars

    def push(self, text: str) -> list[str]:
        self._buf += text
        out: list[str] = []
        while True:
            seg = self._next_segment()
            if seg is None:
                break
            out.append(seg)
        return out

    def flush(self) -> list[str]:
        rest = self._buf.strip()
        self._buf = ""
        if rest:
            self._emitted += 1
            return [rest]
        return []

    def reset(self) -> None:
        self._buf = ""
        self._emitted = 0

    # ----------------------------------------------------------------- internals
    def _boundaries(self) -> list[int]:
        """Candidate cut positions (end index of a complete sentence) in the buffer."""
        buf = self._buf
        cuts: list[int] = []
        for m in _LATIN_END.finditer(buf):
            punct_start = m.start(1)
            if (
                (m.group(1) == "." and _is_abbreviation(buf, punct_start))
                or (
                    m.group(1) == "."
                    and punct_start > 0
                    and buf[punct_start - 1].isdigit()
                    and m.end(3) < len(buf)
                    and buf[m.end(3)].islower()
                )  # "version 2. then" -> probably not a sentence end
            ):
                continue
            cuts.append(m.end(2))
        cuts.extend(m.end() for m in _CJK_END.finditer(buf))
        cuts.extend(m.start() for m in _PARAGRAPH.finditer(buf) if m.start() > 0)
        return sorted(set(cuts))

    def _next_segment(self) -> str | None:
        buf = self._buf
        min_len = self._min()
        for cut in self._boundaries():
            seg = buf[:cut].strip()
            if len(seg) >= min_len:
                self._buf = buf[cut:].lstrip()
                self._emitted += 1
                return seg
        if len(buf) > self.max_chars:
            window = buf[: self.max_chars]
            clauses = list(_CLAUSE.finditer(window))
            if clauses and clauses[-1].end() >= min_len:
                cut = clauses[-1].end()
            else:
                cut = window.rfind(" ")
                if cut < min_len:
                    cut = self.max_chars
            seg = buf[:cut].strip()
            self._buf = buf[cut:].lstrip()
            if seg:
                self._emitted += 1
                return seg
        return None


def split_sentences(text: str, *, min_chars: int = 1, max_chars: int = 400) -> list[str]:
    """Split complete text into sentences (convenience wrapper over the segmenter)."""
    seg = SentenceSegmenter(min_chars=min_chars, max_chars=max_chars)
    return seg.push(text) + seg.flush()
