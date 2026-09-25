"""Endpointing policies of the cascade: how long to wait after the user stops speaking.

The cascade commits a user turn once the silence after the end of speech reaches the
*endpointing delay*. Three policies choose that delay at every pause (see
``docs/concepts/endpointing.md``):

* ``fixed`` — ``min_endpointing_delay`` when the turn detector says the user is done (or
  without a detector), ``max_endpointing_delay`` when it says they are not;
* ``dynamic`` — the delay follows the turn detector's confidence and the user's own
  mid-turn pauses, learned during the session (:class:`PauseTracker`), within
  ``[min_endpointing_delay, max_endpointing_delay]``;
* dictation — long pauses expected (numbers, e-mail addresses, notes): no early commits;
  only a confident turn detector ends the turn before ``dictation_max_delay``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ..metrics import EndpointingPolicy

if TYPE_CHECKING:
    from .cascade import CascadeOptions

__all__ = ["Endpointer", "EndpointingDecision", "EndpointingMode", "PauseTracker"]

EndpointingMode = Literal["fixed", "dynamic"]

_FIXED_MIN = (0.4, 0.6)
"""Default ``min_endpointing_delay`` of the fixed policy: (with a turn detector, VAD only)."""
_DYNAMIC_MIN = (0.25, 0.3)
"""Default lower bound of the dynamic policy: (with a turn detector, VAD only)."""


class PauseTracker:
    """Running estimate of how long the user pauses *inside* a turn.

    Smoothed mean and mean absolute deviation, updated like TCP's retransmission timer
    (RFC 6298): ``mean += alpha * (x - mean)``, ``dev += beta * (|x - mean| - dev)``; the
    first sample sets ``mean = x`` and ``dev = x / 2``. :meth:`bound` = ``mean + k * dev``
    is a pause length the user rarely exceeds mid-turn.
    """

    def __init__(self, *, alpha: float = 0.25, beta: float = 0.25) -> None:
        if not (0.0 < alpha <= 1.0 and 0.0 < beta <= 1.0):
            raise ValueError("alpha and beta must be in (0, 1]")
        self.alpha = alpha
        self.beta = beta
        self.count = 0
        self.mean = 0.0
        self.deviation = 0.0

    def add(self, pause: float) -> None:
        if pause <= 0:
            return
        if self.count == 0:
            self.mean, self.deviation = pause, pause / 2
        else:
            self.deviation += self.beta * (abs(pause - self.mean) - self.deviation)
            self.mean += self.alpha * (pause - self.mean)
        self.count += 1

    def bound(self, k: float) -> float | None:
        """``mean + k * deviation`` (``None`` before the first pause)."""
        return self.mean + k * self.deviation if self.count else None


@dataclass(frozen=True, slots=True)
class EndpointingDecision:
    """The endpointing delay chosen at one pause."""

    delay: float
    """Silence (seconds from the end of speech) before the turn is committed."""
    policy: EndpointingPolicy
    probability: float | None = None
    """The turn detector's end-of-turn probability (``None``: no detector)."""
    threshold: float | None = None
    """The probability at/above which the detector's verdict counts as "done"."""
    hold: float | None = None
    """Dynamic policy: the delay at ``probability == threshold`` (learned from the user's
    pauses, else the fixed policy's minimum)."""
    audio_probability: float | None = None
    """Fused detector (:class:`~voice_agent_next.turn.FusedTurnDetector`): the audio
    half's probability (``None``: no audio)."""
    text_probability: float | None = None
    """Fused detector: the text half's probability (``None``: no transcript, or over its
    latency budget)."""
    detector_error: str | None = None
    """The turn detector failed at this pause (``repr`` of its error): the delay was
    chosen without its verdict."""


class Endpointer:
    """Chooses the endpointing delay of each pause of one cascade connection and learns
    the user's pauses (dynamic policy). Created per connection from
    :class:`~voice_agent_next.engines.cascade.CascadeOptions`; ``mode`` and ``dictation``
    can be changed at runtime."""

    def __init__(self, options: CascadeOptions, *, has_detector: bool) -> None:
        if options.endpointing not in ("fixed", "dynamic"):
            raise ValueError(f"unknown endpointing mode {options.endpointing!r}")
        self.options = options
        self.has_detector = has_detector
        self.mode: EndpointingMode = options.endpointing
        self.dictation = options.dictation
        self.pauses = PauseTracker(alpha=options.pause_alpha, beta=options.pause_alpha)
        self.guard: float | None = None
        """Dynamic policy: the least delay of a *confident* pause while the detector is not
        trusted — it said "done" and the user went on (:meth:`observe_cutoff`). Decays back
        to the floor with every commit the user leaves alone (:meth:`observe_commit`)."""

    @property
    def policy(self) -> EndpointingPolicy:
        return "dictation" if self.dictation else self.mode

    # ---------------------------------------------------------------- bounds
    def fixed_min(self) -> float:
        """The fixed policy's delay when the user seems done."""
        o = self.options
        if o.min_endpointing_delay is not None:
            return o.min_endpointing_delay
        return _FIXED_MIN[0] if self.has_detector else _FIXED_MIN[1]

    def bounds(self) -> tuple[float, float]:
        """``(lowest, highest)`` delay of the current policy."""
        o = self.options
        if self.dictation:
            return o.dictation_min_delay, max(o.dictation_min_delay, o.dictation_max_delay)
        if self.mode == "fixed":
            lo = self.fixed_min()
        elif o.min_endpointing_delay is not None:
            lo = o.min_endpointing_delay
        else:
            lo = _DYNAMIC_MIN[0] if self.has_detector else _DYNAMIC_MIN[1]
        return lo, max(lo, o.max_endpointing_delay)

    def hold(self) -> float:
        """Dynamic policy: the delay for an undecided pause — the user's learned mid-turn
        pause bound, or (before any pause was seen) the fixed policy's minimum."""
        lo, hi = self.bounds()
        learned = self.pauses.bound(self.options.pause_deviations)
        prior = max(lo, self.fixed_min())
        return min(hi, max(lo, prior if learned is None else learned))

    # -------------------------------------------------------------- decisions
    def decide(self, probability: float | None, threshold: float | None) -> EndpointingDecision:
        """The delay for a pause, given the detector's verdict (``None``: no detector)."""
        o = self.options
        lo, hi = self.bounds()
        policy = self.policy
        if probability is None or threshold is None:
            probability = threshold = None
        if self.dictation:
            if threshold is not None and o.dictation_threshold is not None:
                threshold = o.dictation_threshold
            done = probability is not None and threshold is not None and probability >= threshold
            return EndpointingDecision(lo if done else hi, policy, probability, threshold)
        if self.mode == "fixed":
            unlikely = probability is not None and threshold is not None and probability < threshold
            return EndpointingDecision(hi if unlikely else lo, policy, probability, threshold)
        hold = self.hold()
        if probability is None or threshold is None:
            delay = hold
        elif probability >= threshold:
            # likely done: from the hold (at the threshold) down to the floor (at 1.0) —
            # unless the detector cut this user off recently
            w = 1.0 if threshold >= 1.0 else (probability - threshold) / (1.0 - threshold)
            delay = max(hold + (lo - hold) * w, self.guard or lo)
        else:
            # likely mid-thought: from the hold (at the threshold) up to the ceiling (at 0)
            w = (threshold - probability) / threshold
            delay = hold + (hi - hold) * w
        return EndpointingDecision(min(hi, max(lo, delay)), policy, probability, threshold, hold)

    def observe_pause(self, pause: float) -> None:
        """The user paused ``pause`` seconds and then continued the same thought: a pause
        the user resumed from before the commit, or right after a (false) commit."""
        if not self.dictation:  # dictation pauses (digit groups...) are not typical pauses
            self.pauses.add(pause)

    def observe_cutoff(self, pause: float) -> None:
        """The turn detector said the user was done, but they went on after ``pause``
        seconds (a false commit, or a confident pause they resumed from in time).

        The pause is learned, and (dynamic policy) confident pauses wait at least the
        learned hold delay from now on: an audio detector that is sure a complete sentence
        ends the turn is wrong for a user who pauses between sentences, and the confidence
        alone would keep cutting them off."""
        if self.dictation:
            return
        self.pauses.add(pause)
        if self.mode == "dynamic":
            self.guard = max(self.guard or 0.0, self.hold())

    def observe_commit(self) -> None:
        """A commit the user did not contest: the detector regains trust, the guard decays
        towards the floor (by ``pause_alpha`` of its excess)."""
        if self.guard is None:
            return
        lo, _ = self.bounds()
        excess = (self.guard - lo) * (1.0 - self.options.pause_alpha)
        self.guard = lo + excess if excess > 0.01 else None
