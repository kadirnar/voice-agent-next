"""Word and character error counts (Levenshtein alignment) for the ASR track.

``WER = (S + D + I) / N`` where ``N`` is the number of reference tokens. The *corpus* rate
sums errors and reference lengths over all utterances (Σ(S+D+I) / ΣN, research note 06,
§8.3) — it is not the mean of per-utterance rates, which over-weights short utterances.

The alignment is a plain dynamic program (no dependency). Among alignments with the
minimum number of edits, the backtrace prefers hits/substitutions, then deletions, then
insertions, so S/D/I splits are deterministic. The total ``S + D + I`` equals
``jiwer``'s (the tests check it when ``jiwer`` is installed).
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass

__all__ = ["EditCounts", "char_counts", "corpus_rate", "edit_counts", "word_counts"]


@dataclass(frozen=True, slots=True)
class EditCounts:
    """Result of aligning a hypothesis to a reference."""

    hits: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    @property
    def ref_len(self) -> int:
        """``N``: reference tokens (hits + substitutions + deletions)."""
        return self.hits + self.substitutions + self.deletions

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float | None:
        """``errors / N``; ``None`` for an empty reference (undefined)."""
        return self.errors / self.ref_len if self.ref_len else None

    def __add__(self, other: EditCounts) -> EditCounts:
        return EditCounts(
            self.hits + other.hits,
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "ref_len": self.ref_len,
            "errors": self.errors,
        }


def edit_counts(ref: Sequence[Hashable], hyp: Sequence[Hashable]) -> EditCounts:
    """Minimum-edit alignment of ``hyp`` to ``ref`` (any hashable tokens)."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return EditCounts(insertions=m)
    if m == 0:
        return EditCounts(deletions=n)
    # dist[i][j]: edits to turn ref[:i] into hyp[:j]
    dist = [list(range(m + 1))]
    for i in range(1, n + 1):
        prev = dist[-1]
        row = [i] + [0] * m
        r = ref[i - 1]
        for j in range(1, m + 1):
            diag = prev[j - 1] + (0 if r == hyp[j - 1] else 1)
            up = prev[j] + 1  # deletion
            left = row[j - 1] + 1  # insertion
            row[j] = min(diag, up, left)
        dist.append(row)
    hits = subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            same = ref[i - 1] == hyp[j - 1]
            if dist[i][j] == dist[i - 1][j - 1] + (0 if same else 1):
                if same:
                    hits += 1
                else:
                    subs += 1
                i, j = i - 1, j - 1
                continue
        if i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return EditCounts(hits, subs, dels, ins)


def word_counts(ref: str, hyp: str) -> EditCounts:
    """Word-level counts of two (already normalized) strings, split on whitespace."""
    return edit_counts(ref.split(), hyp.split())


def char_counts(ref: str, hyp: str) -> EditCounts:
    """Character-level counts (spaces count as characters unless removed beforehand)."""
    return edit_counts(ref, hyp)


def corpus_rate(counts: Iterable[EditCounts]) -> float | None:
    """Σ(S+D+I) / ΣN over utterances; ``None`` if the references are all empty."""
    total = EditCounts()
    for c in counts:
        total = total + c
    return total.rate
