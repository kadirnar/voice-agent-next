"""Hallucination guard for Whisper-family recognizers.

Whisper was trained on subtitles: on noise, breathing or silence that a VAD let through it
tends to produce text anyway ("Thank you.", "Thanks for watching!", "Untertitel im Auftrag
des ZDF", "ご視聴ありがとうございました", a phrase repeated until the window ends).
:class:`HallucinationGuard` drops such decoded segments using Whisper's own per-segment
statistics (``no_speech_prob``, ``avg_logprob``, ``compression_ratio``), lists of known
artifacts in many languages, repeated n-grams and, when the recognizer runs behind a VAD,
how confident the VAD was that the utterance is speech.

It works on faster-whisper's segment objects and on the segment dictionaries of
openai-whisper / mlx-whisper (``text``, ``avg_logprob``, ``no_speech_prob``,
``compression_ratio``; missing ones are ignored). The faster-whisper and mlx-whisper
providers apply it by default (``hallucination_guard=True``).

The phrase lists come from public reports of Whisper output on audio without speech:

* ``openai/whisper`` discussions `#928 <https://github.com/openai/whisper/discussions/928>`_
  (subtitle credits: Amara.org in several languages, SousTitreur, ST'501, QTSS, ZDF/WDR),
  `#1873 <https://github.com/openai/whisper/discussions/1873>`_ ("Share your
  hallucinations here": DimaTorzok, 明镜, ご視聴ありがとうございました, CastingWords),
  `#2412 <https://github.com/openai/whisper/discussions/2412>`_ (Turkish "Altyazı M.K."),
  `#2608 <https://github.com/openai/whisper/discussions/2608>`_ (Arabic "ترجمة نانسي
  قنقر", German "Untertitelung des ZDF für funk", Norwegian "Tekstet av Nicolai Winther");
* ``m-bain/whisperX`` issue `#230 <https://github.com/m-bain/whisperX/issues/230>`_ (German
  ZDF / Amara.org credits);
* the `sachaarbonel/whisper-hallucinations
  <https://huggingface.co/datasets/sachaarbonel/whisper-hallucinations>`_ dataset (MIT):
  every non-empty output of Whisper on a noise-only corpus, per language (the "thanks for
  watching" / "subscribe" outros and short thank-yous of every list below);
* Barański et al., *Investigation of Whisper ASR Hallucinations Induced by Non-Speech
  Audio*, ICASSP 2025 (`arXiv:2501.11378 <https://arxiv.org/abs/2501.11378>`_), whose
  "bag of hallucinations" is dominated by the same English outros.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .errors import ConfigurationError

__all__ = [
    "ARTIFACTS_BY_LANGUAGE",
    "ARTIFACT_PATTERNS",
    "ARTIFACT_PHRASES",
    "SUSPECTS_BY_LANGUAGE",
    "SUSPECT_PHRASES",
    "GuardVerdict",
    "HallucinationGuard",
    "artifact_phrases",
    "coerce_guard",
    "collapse_repeats",
    "normalize",
    "suspect_phrases",
]

ARTIFACTS_BY_LANGUAGE: dict[str, tuple[str, ...]] = {
    "en": (
        "thank you for watching",
        "thanks for watching",
        "thank you so much for watching",
        "thank you very much for watching",
        "thank you for watching and see you next time",
        "thank you for watching please subscribe",
        "thanks for watching please subscribe",
        "please subscribe",
        "please subscribe to my channel",
        "subscribe to my channel",
        "like and subscribe",
        "please like and subscribe",
        "don't forget to like and subscribe",
        "this is the end of the video remember to like and subscribe",
        "see you in the next video",
        "see you next time",
        "subtitles by the amara.org community",
        "transcription by castingwords",
        "transcription by esoteric",
        "transcribed by https://otter.ai",
        "satsang with mooji",
        "www.mooji.org",
    ),
    "de": (
        "vielen dank fürs zuschauen",
        "vielen dank fürs ansehen",
        "danke fürs zuschauen",
        "untertitel im auftrag des zdf für funk 2017",
        "untertitelung des zdf für funk 2017",
        "untertitel der amara.org community",
        "untertitel von stephanie geiges",
        "copyright wdr 2021",
        "swr 2021",
    ),
    "es": (
        "gracias por ver",
        "gracias por ver el video",
        "subtítulos realizados por la comunidad de amara.org",
        "subtítulos por la comunidad de amara.org",
        "suscríbete",
        "suscríbete al canal",
        "suscríbete a mi canal",
        "suscríbete gracias",
        "no olvides suscribirte",
        "no olviden suscribirse",
    ),
    "fr": (
        "merci d'avoir regardé",
        "merci d'avoir regardé cette vidéo",
        "merci d'avoir regardé la vidéo",
        "j'espère que vous avez apprécié la vidéo",
        "je vous remercie de vous abonner",
        "n'oubliez pas de vous abonner",
        "sous-titres réalisés par la communauté d'amara.org",
        "sous-titres réalisés para la communauté d'amara.org",
        "sous-titrage société radio-canada",
        "sous-titrage st' 501",
        "par soustitreur.com",
    ),
    "it": (
        "grazie per la visione",
        "sottotitoli creati dalla comunità amara.org",
        "sottotitoli e revisione a cura di qtss",
        "non dimenticare di iscriverti",
    ),
    "pt": (
        "obrigado por assistir",
        "obrigada por assistir",
        "legendas pela comunidade amara.org",
        "transcrição e legendas pela comunidade amara.org",
    ),
    "nl": (
        "bedankt voor het kijken",
        "ondertiteld door de amara.org gemeenschap",
        "ondertitels ingediend door de amara.org gemeenschap",
    ),
    "pl": (
        "dziękuję za oglądanie",
        "napisy stworzone przez społeczność amara.org",
    ),
    "tr": (
        "altyazı m.k.",
        "çeviri ve altyazı m.k.",
        "izlediğiniz için teşekkürler",
        "abone ol",
        "abone olmayı unutmayın",
        "kanalıma abone olmayı ve videoyu beğenmeyi unutmayın",
    ),
    "ru": (
        "продолжение следует",
        "спасибо за просмотр",
        "подписывайтесь на мой канал",
        "подписывайтесь на наш канал",
        "ставьте лайки и подписывайтесь",
        "субтитры сделал dimatorzok",
        "субтитры создавал dimatorzok",
    ),
    "zh": (
        "字幕由amara.org社区提供",
        "由 amara.org 社群提供的字幕",
        "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目",
        "谢谢观看",
        "謝謝觀看",
        "謝謝觀看 下次見",
        "记得订阅",
    ),
    "ja": (
        "ご視聴ありがとうございました",
        "ご視聴ありがとうございます",
        "見てくれてありがとう",
        "チャンネル登録お願いします",
        "チャンネル登録をお願いします",
    ),
    "ko": (
        "시청해주셔서 감사합니다",
        "시청해 주셔서 감사합니다",
        "구독해주세요",
        "구독과 좋아요 부탁드립니다",
        "채널 구독 부탁드립니다",
    ),
    "ar": (
        "ترجمة نانسي قنقر",
        "شكرا على المشاهدة",
        "شكرا للمشاهدة",
        "اشتركوا في القناة",
    ),
    "el": ("ευχαριστώ που παρακολουθήσατε",),
    "no": ("tekstet av nicolai winther",),
}
"""Known Whisper artifacts per language (ISO 639-1 code): subtitle credits and video
outros that Whisper produces on non-speech. A segment whose whole text is one of them is
dropped whatever the evidence. See the module docstring for the sources."""

SUSPECTS_BY_LANGUAGE: dict[str, tuple[str, ...]] = {
    "en": (
        "you",
        "thank you",
        "thank you very much",
        "thank you so much",
        "thanks",
        "bye",
        "bye bye",
        "the end",
        "so",
        "oh",
        "hmm",
        "uh",
        "um",
    ),
    "de": ("danke", "danke schön", "vielen dank", "tschüss", "tschüß"),
    "es": ("gracias", "muchas gracias", "adiós"),
    "fr": ("merci", "merci beaucoup", "merci à tous", "au revoir"),
    "it": ("grazie", "grazie mille", "ciao"),
    "pt": ("obrigado", "obrigada", "tchau", "fim"),
    "nl": ("bedankt", "dank je", "dank u wel"),
    "pl": ("dziękuję", "koniec"),
    "tr": ("teşekkürler", "teşekkür ederim"),
    "ru": ("спасибо", "спасибо за внимание"),
    "zh": ("谢谢", "謝謝", "好"),
    "ja": ("ありがとうございました", "ありがとう"),
    "ko": ("감사합니다",),
    "ar": ("شكرا",),
}
"""Short phrases per language that people really say but that are also Whisper's most
common outputs on noise: dropped only when there is other evidence of non-speech (see
:attr:`HallucinationGuard.suspect_no_speech_threshold`)."""

ARTIFACT_PATTERNS: tuple[str, ...] = (
    r".*\bamara org\b.*",  # "Subtitles by the Amara.org community", in any language
    r".*\bdimatorzok\b.*",  # Russian "Субтитры сделал/создавал/добавил DimaTorzok"
    r"untertitel(ung)? (im auftrag )?(des|der) (zdf|ard|wdr|swr|ndr)\b.*",  # any year
    r"(copyright )?(zdf|ard|wdr|swr|ndr|mdr) \d{4}",
    r"(ceviri ve )?altyazı m k",  # Turkish "Altyazı M.K.", accents stripped
    r"sous titres? (realises )?par .*",  # French credits: "Sous-titres par <name>"
    r"sous titrage (st ?501|societe radio canada)",
    r"napisy (stworzone przez|by) .*",  # Polish subtitle credits
    r"字幕由.*提供",  # Chinese "字幕由Amara.org社区提供" and other credits
    r"中文字幕志愿者.*",
    r".*打赏支持明镜.*",
)
"""Regular expressions for artifacts whose wording varies (years, names): a segment is
dropped when one :func:`re.fullmatch` es its :func:`normalize` d text, so patterns are
written in normalized form (lower case, no accents, no punctuation)."""


def artifact_phrases(*languages: str) -> tuple[str, ...]:
    """The artifact phrases of ``languages`` (all languages when none is given)."""
    return _union(ARTIFACTS_BY_LANGUAGE, languages)


def suspect_phrases(*languages: str) -> tuple[str, ...]:
    """The suspect phrases of ``languages`` (all languages when none is given)."""
    return _union(SUSPECTS_BY_LANGUAGE, languages)


def _union(table: Mapping[str, tuple[str, ...]], languages: Iterable[str]) -> tuple[str, ...]:
    keys = [code.strip().replace("_", "-").split("-")[0].lower() for code in languages]
    unknown = [k for k in keys if k not in table]
    if unknown:
        raise ValueError(f"no phrase list for {unknown}; known: {sorted(table)}")
    out: dict[str, None] = {}
    for key in keys or table:
        out.update(dict.fromkeys(table[key]))
    return tuple(out)


ARTIFACT_PHRASES: tuple[str, ...] = artifact_phrases()
"""Every language's artifacts (the default of :attr:`HallucinationGuard.artifacts`)."""

SUSPECT_PHRASES: tuple[str, ...] = suspect_phrases()
"""Every language's suspect phrases (the default of :attr:`HallucinationGuard.suspects`)."""


@dataclass(frozen=True, slots=True)
class GuardVerdict:
    """What the guard kept and why it dropped the rest."""

    kept: list[Any]
    dropped: list[tuple[str, str]] = field(default_factory=list)
    """``(segment text, reason)`` for every dropped segment."""


@dataclass(frozen=True, slots=True)
class HallucinationGuard:
    """Drops Whisper segments that are probably not speech. Every rule can be disabled
    with ``None`` (or an empty tuple for the phrase lists).

    A segment is dropped when:

    1. it has no word characters (``"..."``, ``"♪"``), unless :attr:`drop_empty` is off;
    2. its normalized text is in :attr:`artifacts` or matches one of :attr:`patterns`;
    3. its normalized text is in :attr:`suspects` and there is other evidence of
       non-speech: ``no_speech_prob >= suspect_no_speech_threshold``, ``avg_logprob <
       suspect_log_prob_threshold`` or a weak VAD;
    4. ``no_speech_prob >= no_speech_threshold`` and (``avg_logprob <
       log_prob_threshold`` or a weak VAD) — Whisper's own rule, extended with the VAD;
    5. ``compression_ratio > compression_ratio_threshold`` (a repetition loop that
       Whisper's temperature fallback did not fix).

    Repetition loops inside a kept segment (an n-gram of up to :attr:`max_ngram` words
    repeated more than :attr:`max_ngram_repeats` times in a row) are cut back to one
    occurrence.

    "Weak VAD" means the recognizer runs behind a VAD
    (:class:`~voice_agent_next.stt.StreamAdapter`) and the mean speech probability of the
    utterance's speech windows is below :attr:`vad_threshold`: the VAD itself was unsure.

    The phrase lists default to every language of :data:`ARTIFACTS_BY_LANGUAGE` and
    :data:`SUSPECTS_BY_LANGUAGE` (Whisper detects the language per utterance, and an
    artifact can come out in another language than the user's). Restrict them with
    :func:`artifact_phrases` / :func:`suspect_phrases`, e.g.
    ``HallucinationGuard(suspects=suspect_phrases("en", "de"))``.
    """

    no_speech_threshold: float | None = 0.6
    log_prob_threshold: float | None = -1.0
    compression_ratio_threshold: float | None = 2.4
    suspect_no_speech_threshold: float | None = 0.2
    suspect_log_prob_threshold: float | None = -0.8
    vad_threshold: float | None = 0.5
    max_ngram: int = 4
    max_ngram_repeats: int | None = 4
    artifacts: tuple[str, ...] = ARTIFACT_PHRASES
    suspects: tuple[str, ...] = SUSPECT_PHRASES
    patterns: tuple[str, ...] = ARTIFACT_PATTERNS
    drop_empty: bool = True
    _compiled: tuple[re.Pattern[str], ...] = field(
        default=(), init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.max_ngram < 1:
            raise ValueError("max_ngram must be >= 1")
        if self.max_ngram_repeats is not None and self.max_ngram_repeats < 1:
            raise ValueError("max_ngram_repeats must be >= 1")
        try:
            compiled = tuple(re.compile(p) for p in self.patterns)
        except re.error as exc:
            raise ValueError(f"invalid pattern: {exc}") from exc
        # stored normalized (frozen dataclass: bypass __setattr__)
        object.__setattr__(self, "artifacts", _normalized_set(self.artifacts))
        object.__setattr__(self, "suspects", _normalized_set(self.suspects))
        object.__setattr__(self, "patterns", tuple(self.patterns))
        object.__setattr__(self, "_compiled", compiled)

    @classmethod
    def disabled(cls) -> HallucinationGuard:
        """A guard that keeps everything."""
        return cls(
            no_speech_threshold=None,
            log_prob_threshold=None,
            compression_ratio_threshold=None,
            suspect_no_speech_threshold=None,
            suspect_log_prob_threshold=None,
            vad_threshold=None,
            max_ngram_repeats=None,
            artifacts=(),
            suspects=(),
            patterns=(),
            drop_empty=False,
        )

    def weak_vad(self, vad_confidence: float | None) -> bool:
        return (
            self.vad_threshold is not None
            and vad_confidence is not None
            and vad_confidence < self.vad_threshold
        )

    def is_artifact(self, text: str) -> bool:
        """``text`` is a known artifact (a phrase of :attr:`artifacts` or a pattern)."""
        return self._artifact(normalize(text))

    def _artifact(self, key: str) -> bool:
        if not key:
            return False
        return key in self.artifacts or any(p.fullmatch(key) for p in self._compiled)

    def reason(self, segment: Any, *, vad_confidence: float | None = None) -> str | None:
        """Why ``segment`` should be dropped, or ``None`` to keep it."""
        text = normalize(str(_field(segment, "text") or ""))
        no_speech = _number(segment, "no_speech_prob")
        logprob = _number(segment, "avg_logprob")
        ratio = _number(segment, "compression_ratio")
        weak = self.weak_vad(vad_confidence)
        if not text and self.drop_empty:
            return "no words"
        if self._artifact(text):
            return "known artifact"
        if text in self.suspects and (
            weak
            or _at_least(no_speech, self.suspect_no_speech_threshold)
            or _below(logprob, self.suspect_log_prob_threshold)
        ):
            return "suspect phrase on weak evidence"
        if _at_least(no_speech, self.no_speech_threshold) and (
            weak or _below(logprob, self.log_prob_threshold)
        ):
            return "no speech"
        threshold = self.compression_ratio_threshold
        if ratio is not None and threshold is not None and ratio > threshold:
            return "repetitive"
        return None

    def filter(
        self, segments: Sequence[Any], *, vad_confidence: float | None = None
    ) -> GuardVerdict:
        """Split ``segments`` into kept and dropped ones."""
        kept: list[Any] = []
        dropped: list[tuple[str, str]] = []
        for segment in segments:
            why = self.reason(segment, vad_confidence=vad_confidence)
            if why is None:
                kept.append(segment)
            else:
                dropped.append((str(_field(segment, "text") or "").strip(), why))
        return GuardVerdict(kept, dropped)

    def clean_text(self, text: str) -> str:
        """``text`` with repetition loops cut back to one occurrence."""
        if self.max_ngram_repeats is None:
            return text
        return collapse_repeats(text, max_n=self.max_ngram, max_repeats=self.max_ngram_repeats)


def coerce_guard(
    value: bool | HallucinationGuard | Mapping[str, Any] | None, *, provider: str
) -> HallucinationGuard:
    """A provider's ``hallucination_guard`` option as a :class:`HallucinationGuard`:
    ``True`` is the default guard, ``False`` / ``None`` a disabled one, a mapping holds
    its fields (lists become tuples, as in a YAML config)."""
    if isinstance(value, HallucinationGuard):
        return value
    if value is None or value is False:
        return HallucinationGuard.disabled()
    if value is True:
        return HallucinationGuard()
    if isinstance(value, Mapping):
        fields: dict[str, Any] = {
            k: tuple(v) if isinstance(v, list) else v for k, v in value.items()
        }
        try:
            return HallucinationGuard(**fields)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"{provider}: hallucination_guard: {exc}") from exc
    raise ConfigurationError(
        f"{provider}: hallucination_guard must be a bool, a HallucinationGuard or a "
        f"mapping, got {type(value).__name__}"
    )


_PUNCT = re.compile(r"[^\w\s']+")
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Case-folded, accents removed, punctuation (except apostrophes) replaced by spaces,
    whitespace collapsed: ``"Sous-titres réalisés"`` -> ``"sous titres realises"``."""
    folded = unicodedata.normalize("NFKD", text.casefold().replace("’", "'"))
    bare = unicodedata.normalize("NFC", "".join(c for c in folded if not unicodedata.combining(c)))
    return _SPACE.sub(" ", _PUNCT.sub(" ", bare.replace("_", " "))).strip()


def collapse_repeats(text: str, *, max_n: int = 4, max_repeats: int = 4) -> str:
    """Cut runs of an n-gram (``n <= max_n`` words) repeated more than ``max_repeats``
    times in a row back to a single occurrence. Words are compared normalized; the
    original spelling of the kept words is preserved."""
    words = text.split()
    keys = [normalize(w) for w in words]
    i = 0
    out: list[str] = []
    while i < len(words):
        best_n, best_count = 0, 0
        for n in range(1, max_n + 1):
            if i + n > len(words) or not all(keys[i : i + n]):
                break
            gram = keys[i : i + n]
            count = 1
            while keys[i + count * n : i + (count + 1) * n] == gram:
                count += 1
            if count > max_repeats and count * n > best_count * best_n:
                best_n, best_count = n, count
        if best_n:
            out.extend(words[i : i + best_n])
            i += best_n * best_count
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


def _normalized_set(phrases: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(k for k in (normalize(p) for p in phrases) if k))


def _field(segment: Any, name: str) -> Any:
    if isinstance(segment, Mapping):  # openai-whisper / mlx-whisper segments are dicts
        return segment.get(name)
    return getattr(segment, name, None)


def _number(segment: Any, name: str) -> float | None:
    value = _field(segment, name)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _at_least(value: float | None, threshold: float | None) -> bool:
    return value is not None and threshold is not None and value >= threshold


def _below(value: float | None, threshold: float | None) -> bool:
    return value is not None and threshold is not None and value < threshold
