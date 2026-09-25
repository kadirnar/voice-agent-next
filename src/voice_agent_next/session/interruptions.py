"""Interruption policy: tell real barge-ins from backchannels, coughs and noise.

Voice activity alone is a poor barge-in signal: most VAD triggers while the agent talks
are coughs, noise, echo or backchannels ("uh-huh", "right") — Krisp measured 66% false
positives for VAD-only barge-in. :class:`~voice_agent_next.session.AgentSession` therefore
*pauses* the agent when the user starts speaking over it and collects evidence in an
:class:`Overlap` until :meth:`Overlap.verdict` decides:

``INTERRUPT``
    a real barge-in — the user spoke for ``min_interruption_duration`` seconds and said at
    least ``min_interruption_words`` non-backchannel words: stop for good, cancel the
    response and truncate the agent's turn to what the user heard;
``RESUME``
    a false interruption — the user went quiet for ``false_interruption_timeout`` seconds
    without saying anything meaningful, or the final transcript of what they said is only
    backchannels: resume the paused speech where it stopped.

Small streaming ASR models are trained on transcripts without fillers: they often turn a
backchannel into a wrong word ("uh-huh" -> "but high", "mm-hmm" -> "m"), which no list can
match. With ``max_backchannel_duration`` set, a *short* utterance over the agent (less
speech than that, at most two words) is therefore a backchannel unless one of its words
is an **interruption word** ("stop", "wait", "no", question words...): it never confirms
a barge-in by duration alone, and its transcript is dropped instead of answered.

If the user keeps making sound without words (noise, a TV) the agent is held at most
``false_interruption_timeout`` seconds (``UNPAUSE``) while the overlap keeps watching.
After a confirmed interruption it keeps watching as well: if the user then stays quiet
without meaningful words, the verdict is ``FALSE_INTERRUPTION`` (reported to the app,
which may continue the conversation); otherwise ``SETTLED``.

This module is pure logic — explicit timestamps, no asyncio — so it is easy to test and
to reuse in simulators and benchmarks. See ``docs/concepts/interruptions.md``.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING

from .events import FalseInterruptionReason

if TYPE_CHECKING:
    from .session import SessionOptions

__all__ = [
    "BACKCHANNEL_WORDS",
    "INTERRUPTION_WORDS",
    "BackchannelFilter",
    "InterruptionPolicy",
    "Overlap",
    "Verdict",
    "backchannel_words_for",
    "interruption_words_for",
    "split_words",
]

# ------------------------------------------------------------------------ backchannels

# Short reactions that mean "go on", never "stop". Kept conservative: a word that is also a
# common answer or correction ("no", "wait", "but") must stay out. Fillers such as "uh"/"um"
# are included because STT engines transcribe hesitations and coughs as them.
_EN = (
    "uh-huh", "uh huh", "uhhuh", "mm-hmm", "mm hmm", "mmhmm", "mhm", "m", "mm", "hm", "hmm",
    "uh", "um", "er", "erm", "ah", "oh", "eh", "yeah", "yea", "yep", "yup", "yes", "ok",
    "okay", "k", "right", "sure", "alright", "all right", "i see", "got it", "cool", "nice",
    "great", "wow", "true", "exactly", "indeed", "totally", "absolutely", "of course",
    "makes sense", "fair enough", "go on", "go ahead",
)  # fmt: skip
# English reactions heard in every language (added to the other languages' lists).
_EN_CORE = (
    "uh-huh", "uh huh", "mm-hmm", "mm hmm", "mhm", "m", "mm", "hm", "hmm", "ok", "okay",
    "yeah", "yes",
)  # fmt: skip

BACKCHANNEL_WORDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "en": _EN,
        "de": (
            "ja", "jaja", "jap", "jo", "genau", "aha", "achso", "ach so", "stimmt", "richtig",
            "klar", "alles klar", "gut", "super", "ah", "oh", "äh", "ähm",
        ),
        "es": (
            "sí", "si", "ajá", "aja", "vale", "claro", "ya", "bueno", "okey", "de acuerdo",
            "entiendo", "exacto", "cierto", "eh", "ah",
        ),
        "fr": (
            "oui", "ouais", "d'accord", "ah", "oh", "euh", "hum", "voilà", "exactement",
            "bien sûr", "je vois", "c'est ça", "en effet", "tout à fait", "super",
        ),
        "it": (
            "sì", "si", "certo", "esatto", "giusto", "va bene", "capisco", "ah", "eh", "ehm",
            "mh", "già", "vero", "perfetto",
        ),
        "pt": (
            "sim", "é", "tá", "ta", "certo", "claro", "entendi", "uhum", "aham", "hum",
            "beleza", "exato", "isso", "ah",
        ),
        "tr": (
            "evet", "tamam", "hı hı", "hıhı", "hı", "he", "peki", "anladım", "aynen", "doğru",
            "tabii", "tabi", "olur", "iyi", "ha", "haa",
        ),
        "ja": ("はい", "うん", "ええ", "そう", "そうですね", "なるほど", "へえ", "ふーん", "ほう"),
        "zh": ("嗯", "对", "好", "好的", "是", "是的", "哦", "啊", "明白", "没错", "行"),
    }
)  # fmt: skip
"""Built-in backchannel lists per language (primary subtag). Use
:func:`backchannel_words_for` to get the list for an agent's language."""


def backchannel_words_for(language: str | None) -> tuple[str, ...]:
    """The built-in backchannel list for ``language`` (``"tr"``, ``"pt-BR"``, ``None``...).

    English (or no/unknown language) gets the full English list; other languages get
    their own list plus English reactions that are heard everywhere ("okay", "mm-hmm").
    """
    primary = (language or "en").replace("_", "-").split("-")[0].casefold()
    if primary == "en":
        return _EN
    return BACKCHANNEL_WORDS.get(primary, ()) + _EN_CORE


# Words that make even a very short utterance a real interruption (commands, refusals,
# repair requests and question words). Used by the short-utterance rule only.
_EN_INTERRUPT = (
    "stop", "wait", "no", "nope", "not", "don't", "hold on", "hang on", "one moment",
    "one second", "sorry", "excuse me", "pardon", "cancel", "enough", "quiet", "shut up",
    "pause", "hey", "hello", "listen", "actually", "repeat", "again", "what", "why", "how",
    "who", "where", "when", "which",
)  # fmt: skip
INTERRUPTION_WORDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "en": _EN_INTERRUPT,
        "de": (
            "stopp", "halt", "warte", "warten", "nein", "moment", "entschuldigung", "sorry",
            "was", "wie", "warum", "wer", "wo", "wann", "welche", "nochmal",
        ),
        "es": (
            "para", "espera", "no", "perdón", "perdona", "disculpa", "momento", "qué",
            "que", "cómo", "como", "por qué", "quién", "dónde", "cuándo", "cuál",
        ),
        "fr": (
            "stop", "arrête", "attends", "non", "pardon", "excusez-moi", "moment", "quoi",
            "comment", "pourquoi", "qui", "où", "quand", "quel", "quelle",
        ),
        "it": (
            "stop", "basta", "aspetta", "no", "scusa", "scusi", "momento", "cosa", "come",
            "perché", "chi", "dove", "quando", "quale",
        ),
        "pt": (
            "para", "pare", "espera", "espere", "não", "desculpa", "momento", "o quê", "que",
            "como", "por que", "quem", "onde", "quando", "qual",
        ),
        "tr": (
            "dur", "bekle", "hayır", "yok", "pardon", "affedersin", "ne", "nasıl", "neden",
            "niye", "kim", "nerede", "ne zaman", "hangi", "bir dakika",
        ),
        "ja": ("待って", "ちょっと", "いいえ", "違う", "すみません", "何", "なに", "どう", "なぜ"),
        "zh": ("等", "停", "不", "不是", "对不起", "什么", "为什么", "怎么", "哪"),
    }
)  # fmt: skip
"""Built-in interruption words per language (primary subtag): words that make even a very
short utterance over the agent a real barge-in (see ``max_backchannel_duration``)."""
_EN_INTERRUPT_CORE = ("stop", "wait", "no", "sorry", "hey", "hello", "what")
_MAX_SHORT_WORDS = 2
"""A short utterance with more words than this is a real turn."""


def interruption_words_for(language: str | None) -> tuple[str, ...]:
    """The built-in interruption words for ``language`` (``"de"``, ``"pt-BR"``, ``None``...):
    English for English or no/unknown language; other languages get their own list plus a
    few English words heard everywhere ("stop", "sorry")."""
    primary = (language or "en").replace("_", "-").split("-")[0].casefold()
    if primary == "en":
        return _EN_INTERRUPT
    return INTERRUPTION_WORDS.get(primary, ()) + _EN_INTERRUPT_CORE


# kana + CJK ideographs: languages written without spaces count characters as words
_CJK = "぀-ヿ㐀-䶿一-鿿豈-﫿"
_CJK_SPLIT = re.compile(rf"[{_CJK}]|[^{_CJK}]+")
_SEPARATORS = re.compile(r"[^\w']+|_")
_ELONGATED = re.compile(r"(\w)\1{2,}")


def split_words(text: str) -> list[str]:
    """Normalized words of ``text``: case-folded, punctuation and hyphens removed,
    elongations collapsed ("Mmmm" -> "m", "yeahhh" -> "yeah"), CJK split per character."""
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    words: list[str] = []
    for token in _SEPARATORS.sub(" ", text).split():
        token = _ELONGATED.sub(r"\1", token.strip("'"))
        if token:
            words.extend(_CJK_SPLIT.findall(token))
    return words


class BackchannelFilter:
    """Separates backchannels ("uh-huh", "okay", "mm-hmm") from meaningful words.

    Entries may be phrases ("all right", "je vois"); matching is on normalized words
    (see :func:`split_words`), longest phrase first.
    """

    def __init__(self, phrases: Iterable[str]) -> None:
        self._phrases = frozenset(p for p in (tuple(split_words(s)) for s in phrases) if p)
        self._longest = max((len(p) for p in self._phrases), default=0)

    def meaningful_words(self, text: str) -> list[str]:
        """The words of ``text`` that are not part of a backchannel."""
        words = split_words(text)
        meaningful: list[str] = []
        i = 0
        while i < len(words):
            for n in range(min(self._longest, len(words) - i), 0, -1):
                if tuple(words[i : i + n]) in self._phrases:
                    i += n
                    break
            else:
                meaningful.append(words[i])
                i += 1
        return meaningful

    def is_backchannel(self, text: str) -> bool:
        """True if ``text`` has words and all of them are backchannels."""
        return bool(split_words(text)) and not self.meaningful_words(text)

    def contains(self, words: list[str]) -> bool:
        """True if a phrase of the list occurs in the normalized ``words``."""
        for i in range(len(words)):
            for n in range(min(self._longest, len(words) - i), 0, -1):
                if tuple(words[i : i + n]) in self._phrases:
                    return True
        return False


@functools.lru_cache(maxsize=32)
def _shared_filter(phrases: tuple[str, ...]) -> BackchannelFilter:
    return BackchannelFilter(phrases)


# ----------------------------------------------------------------------------- policy
@dataclass(frozen=True, slots=True)
class InterruptionPolicy:
    """When user speech over the agent is a real barge-in (see :class:`SessionOptions`).

    Attributes:
        min_duration: seconds of user speech needed to confirm a barge-in.
        min_words: non-backchannel words needed as well (``0`` = duration only).
        false_interruption_timeout: seconds of user silence, without meaningful words,
            after which an unconfirmed barge-in is declared false (``None`` = never).
        resume: pause (instead of keep playing) while the verdict is pending and resume
            after a false interruption.
        backchannels: words that never count as an interruption.
        max_backchannel_duration: the short-utterance rule (``None`` = off): user speech
            over the agent shorter than this, with at most two words and no interruption
            word, is a backchannel whatever the STT made of it. Include the VAD's
            end-of-speech hangover: speech counts until the VAD reports its end.
        interruptions: words that make even a short utterance a real barge-in.
    """

    min_duration: float = 0.5
    min_words: int = 0
    false_interruption_timeout: float | None = 2.0
    resume: bool = True
    backchannels: BackchannelFilter = field(
        default_factory=lambda: _shared_filter(backchannel_words_for(None))
    )
    max_backchannel_duration: float | None = None
    interruptions: BackchannelFilter = field(
        default_factory=lambda: _shared_filter(interruption_words_for(None))
    )

    def __post_init__(self) -> None:
        if self.min_duration < 0:
            raise ValueError("min_duration must be >= 0")
        if self.min_words < 0:
            raise ValueError("min_words must be >= 0")
        if self.false_interruption_timeout is not None and self.false_interruption_timeout < 0:
            raise ValueError("false_interruption_timeout must be >= 0 or None")
        if self.max_backchannel_duration is not None and self.max_backchannel_duration < 0:
            raise ValueError("max_backchannel_duration must be >= 0 or None")

    @classmethod
    def from_options(
        cls, options: SessionOptions, *, language: str | None = None
    ) -> InterruptionPolicy:
        words = options.backchannel_words
        phrases = tuple(words) if words is not None else backchannel_words_for(language)
        stops = options.interruption_words
        interruptions = tuple(stops) if stops is not None else interruption_words_for(language)
        return cls(
            min_duration=options.min_interruption_duration,
            min_words=options.min_interruption_words,
            false_interruption_timeout=options.false_interruption_timeout,
            resume=options.resume_false_interruption,
            backchannels=_shared_filter(phrases),
            max_backchannel_duration=options.max_backchannel_duration,
            interruptions=_shared_filter(interruptions),
        )

    @property
    def immediate(self) -> bool:
        """Interrupt at the first sign of speech, without pausing (no policy)."""
        return self.min_duration <= 0 and self.min_words <= 0

    @property
    def pauses(self) -> bool:
        """Pause playback while the verdict is pending (else the agent keeps talking)."""
        return self.resume and self.false_interruption_timeout is not None

    @property
    def words_needed(self) -> int:
        """Meaningful words that make an utterance a real turn (at least one)."""
        return max(1, self.min_words)


class Verdict(StrEnum):
    INTERRUPT = "interrupt"
    """A real barge-in: stop, cancel and truncate."""
    RESUME = "resume"
    """A false interruption: resume the paused speech."""
    UNPAUSE = "unpause"
    """The user keeps making sound without words: stop holding the agent, keep watching."""
    FALSE_INTERRUPTION = "false_interruption"
    """After a confirmed interruption the user said nothing meaningful (nothing to resume)."""
    SETTLED = "settled"
    """After a confirmed interruption the user took the turn: stop watching."""


# Deadlines are computed from the same timestamps the verdict compares: this slack keeps a
# timer that fires a hair early (coarse timers, float rounding) from re-arming forever.
_EPS = 1e-3


@dataclass(eq=False)
class Overlap:
    """Evidence about one stretch of user speech over the agent.

    Every time is a :func:`~voice_agent_next.utils.now` timestamp. The duration rule is
    per speech segment (a pause starts a new one), measured from where speech started to
    where it ended; while the engine still reports speech the ongoing segment counts up to
    ``t``, so it includes the VAD's trailing-silence hangover (as in LiveKit).
    ``quiet_since`` is when the engine reported the user quiet.
    """

    policy: InterruptionPolicy
    started_at: float
    speech: float = 0.0
    """Seconds of speech in finished segments (all of them)."""
    longest: float = 0.0
    """The longest finished segment."""
    speaking_since: float | None = None
    """Start of the ongoing speech segment (``None`` while the user is quiet)."""
    quiet_since: float | None = None
    confirmed: bool = False
    """The interruption was confirmed (the agent's speech is over)."""
    held: bool = False
    """The agent is paused while the verdict is pending."""
    final_after_quiet: bool = False
    """A final transcript arrived after the user went quiet: it covers the whole utterance."""
    texts: dict[str, str] = field(default_factory=dict)
    """Latest transcript per input item."""
    _words: tuple[str, tuple[str, ...]] = field(default=("", ()), init=False, repr=False)

    @classmethod
    def begin(cls, policy: InterruptionPolicy, start: float, *, held: bool = False) -> Overlap:
        """User speech started at ``start`` while the agent was talking."""
        return cls(policy, started_at=start, speaking_since=start, held=held)

    # -------------------------------------------------------------------- evidence
    def speech_started(self, start: float) -> None:
        if self.speaking_since is None:
            self.speaking_since = start
        self.quiet_since = None
        self.final_after_quiet = False

    def speech_stopped(self, speech_end: float, t: float) -> None:
        """Speech ended at ``speech_end``; the engine reported it at ``t``."""
        if self.speaking_since is None:
            return
        segment = max(0.0, speech_end - self.speaking_since)
        self.speech += segment
        self.longest = max(self.longest, segment)
        self.speaking_since = None
        self.quiet_since = t

    def add_transcript(self, item_id: str, text: str, *, final: bool = False) -> None:
        """Interim (``final=False``) or final transcript of what the user said."""
        self.texts[item_id] = text
        if final and self.speaking_since is None:
            self.final_after_quiet = True

    # --------------------------------------------------------------------- queries
    @property
    def speaking(self) -> bool:
        return self.speaking_since is not None

    @property
    def transcript(self) -> str:
        return " ".join(t.strip() for t in self.texts.values() if t.strip())

    def meaningful_words(self) -> list[str]:
        """Non-backchannel words transcribed during the overlap."""
        text = self.transcript
        if text != self._words[0]:
            self._words = (text, tuple(self.policy.backchannels.meaningful_words(text)))
        return list(self._words[1])

    def speech_duration(self, t: float) -> float:
        """Seconds of speech in the whole overlap (all segments)."""
        ongoing = 0.0 if self.speaking_since is None else max(0.0, t - self.speaking_since)
        return self.speech + ongoing

    def segment_duration(self, t: float) -> float:
        """The longest speech segment so far (the duration rule is per segment)."""
        ongoing = 0.0 if self.speaking_since is None else max(0.0, t - self.speaking_since)
        return max(self.longest, ongoing)

    def urgent(self) -> bool:
        """An interruption word ("stop", "wait", "what"...) was transcribed."""
        return self.policy.interruptions.contains(self.meaningful_words())

    def short_backchannel(self, t: float) -> bool:
        """The short-utterance rule (``max_backchannel_duration``): so far less speech than
        that, at most two meaningful words and no interruption word — a backchannel,
        however the STT transcribed it."""
        limit = self.policy.max_backchannel_duration
        return (
            limit is not None
            and self.speech_duration(t) < limit - _EPS
            and len(self.meaningful_words()) <= _MAX_SHORT_WORDS
            and not self.urgent()
        )

    def not_a_turn(self) -> bool:
        """The user went quiet and the final transcript of what they said is not a turn
        (only backchannels, fewer words than needed, or a short utterance without an
        interruption word): the engine should drop it."""
        return (
            not self.confirmed
            and self.speaking_since is None
            and self.final_after_quiet
            and bool(self.transcript)
            and (
                len(self.meaningful_words()) < self.policy.words_needed
                or self.short_backchannel(self.quiet_since or self.started_at)
            )
        )

    def is_turn(self) -> bool:
        """The user went quiet after saying enough meaningful words, and not a short
        backchannel: whatever the verdict, the engine may commit (and answer) it."""
        return (
            not self.confirmed
            and self.speaking_since is None
            and len(self.meaningful_words()) >= self.policy.words_needed
            and not self.short_backchannel(self.quiet_since or self.started_at)
        )

    def reason(self) -> FalseInterruptionReason:
        """Why the overlap was not an interruption."""
        if not self.transcript:
            return "noise"
        if not self.meaningful_words():
            return "backchannel"
        if self.speaking_since is None and self.short_backchannel(self.started_at):
            return "backchannel"  # a short reaction the STT turned into words
        return "too_few_words"

    def _quiet_for(self, t: float) -> float | None:
        if self.speaking_since is not None or self.quiet_since is None:
            return None
        return t - self.quiet_since

    def verdict(self, t: float) -> Verdict | None:
        """The decision at time ``t`` (``None`` = wait for more evidence)."""
        p = self.policy
        timeout = p.false_interruption_timeout
        words = len(self.meaningful_words())
        quiet_for = self._quiet_for(t)
        timed_out = quiet_for is not None and timeout is not None and quiet_for >= timeout - _EPS
        if self.confirmed:
            if timed_out:
                return Verdict.SETTLED if words else Verdict.FALSE_INTERRUPTION
            return None
        short = self.short_backchannel(t)
        long_enough = self.segment_duration(t) >= p.min_duration - _EPS
        if long_enough and words >= p.min_words and not short:
            return Verdict.INTERRUPT
        if (
            quiet_for is not None
            and self.final_after_quiet
            and self.transcript
            and (not words or short)
        ):
            return Verdict.RESUME  # only backchannels: no reason to keep the agent waiting
        if timed_out:
            return Verdict.INTERRUPT if words >= p.words_needed and not short else Verdict.RESUME
        if (
            self.held
            and self.speaking_since is not None
            and timeout is not None
            and t - self.started_at >= timeout - _EPS
        ):
            return Verdict.UNPAUSE  # sound without words for too long (noise, a TV...)
        return None

    def deadline(self) -> float | None:
        """When :meth:`verdict` may change without new evidence (``None`` = only on events)."""
        p = self.policy
        timeout = p.false_interruption_timeout
        times: list[float] = []
        if not self.confirmed and self.speaking_since is not None:
            if len(self.meaningful_words()) >= p.min_words:
                # the duration rule: the segment and, for a possible backchannel, the speech
                # of the whole overlap must be long enough
                due = []
                if self.longest < p.min_duration:
                    due.append(self.speaking_since + p.min_duration)
                limit = p.max_backchannel_duration
                if limit is not None and self.short_backchannel(self.speaking_since):
                    due.append(self.speaking_since + limit - self.speech)
                if due:
                    times.append(max(due))
            if self.held and timeout is not None:
                times.append(self.started_at + timeout)
        if timeout is not None and self.quiet_since is not None and self.speaking_since is None:
            times.append(self.quiet_since + timeout)
        return min(times) if times else None
