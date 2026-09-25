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


def _clause_ends(text: str) -> list[int]:
    """Clause boundaries in ``text``, except between two numbers ("July 14, 2025") or
    after a number whose continuation is not known yet."""
    return [
        m.end()
        for m in _CLAUSE.finditer(text)
        if not (
            m.start() > 0
            and text[m.start() - 1].isdigit()
            and (m.end() == len(text) or text[m.end()].isdigit())
        )
    ]


def _last_space(text: str) -> int:
    """Last whitespace in ``text`` that is not next to a number ("4:30 PM", "5 kg",
    "July 14, 2025"); any last whitespace if there is none."""
    for i in range(len(text) - 1, 1, -1):
        if not text[i].isspace():
            continue
        before = text[i - 1] if text[i - 1] not in ",:" else text[i - 2]
        after = text[i + 1] if i + 1 < len(text) else ""
        if not before.isdigit() and not after.isdigit():
            return i
    return text.rfind(" ")


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
        first_segment_max_chars: if the first sentence is longer than this, release its
            first clause (up to the first comma/semicolon/colon/dash) on its own, so a
            sentence-at-a-time TTS can start speaking sooner.
    """

    def __init__(
        self,
        *,
        min_chars: int = 10,
        max_chars: int = 220,
        first_segment_min_chars: int | None = None,
        first_segment_max_chars: int | None = None,
    ) -> None:
        if max_chars <= min_chars:
            raise ValueError("max_chars must be greater than min_chars")
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.first_segment_min_chars = first_segment_min_chars
        self.first_segment_max_chars = first_segment_max_chars
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

    def _first_clause(self, min_len: int) -> str | None:
        """Cut a long *first* sentence at its first clause boundary (faster first audio)."""
        limit = self.first_segment_max_chars
        if limit is None or self._emitted:
            return None
        buf = self._buf
        sentence_end = next(
            (c for c in self._boundaries() if len(buf[:c].strip()) >= min_len), None
        )
        span = buf[:sentence_end] if sentence_end is not None else buf
        if len(span.strip()) <= limit:
            return None
        cut = next((end for end in _clause_ends(span) if end >= min_len), None)
        if cut is None and len(span) > 2 * limit:  # no clause in sight: word boundary
            space = _last_space(span[: 2 * limit])
            cut = space if space >= min_len else None
        if cut is None or (sentence_end is not None and cut >= sentence_end):
            return None
        seg = buf[:cut].strip()
        self._buf = buf[cut:].lstrip()
        self._emitted += 1
        return seg

    def _next_segment(self) -> str | None:
        buf = self._buf
        min_len = self._min()
        first = self._first_clause(min_len)
        if first:
            return first
        for cut in self._boundaries():
            seg = buf[:cut].strip()
            if len(seg) >= min_len:
                self._buf = buf[cut:].lstrip()
                self._emitted += 1
                return seg
        if len(buf) > self.max_chars:
            window = buf[: self.max_chars]
            clauses = _clause_ends(window)
            if clauses and clauses[-1] >= min_len:
                cut = clauses[-1]
            else:
                cut = _last_space(window)
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
