"""Spoken-form text normalization for TTS input.

LLM replies are full of digits, prices, times, dates, e-mail addresses and URLs. Many TTS
models read them badly (Pocket TTS reads "58213" as "five tu o thiurt"). A normalizer
rewrites them as words before synthesis::

    >>> from voice_agent_next.text.normalize import normalize_text
    >>> normalize_text("The total is $42.50, due March 3rd at 4:30 PM.")
    'The total is forty two dollars and fifty cents, due March third at four thirty pee em.'

Every normalizer returns a :class:`NormalizedText` that remembers which span of the
original text each rewritten span came from, so word timings reported by a TTS on the
normalized text can be mapped back to the words the LLM wrote (:class:`WordAligner`):
barge-in truncation keeps "$42.50" in the chat history, not "forty two dollars and".

Languages are pluggable (:func:`register_normalizer`): English has a dependency-free
rule set (:class:`EnglishNormalizer`); other languages get numbers and percentages via
the optional ``num2words`` package (:class:`Num2WordsNormalizer`) when it is installed.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Protocol, runtime_checkable

from ..stt import WordTiming
from ..utils.deps import is_installed
from .sentences import _ABBREVIATIONS as _SENTENCE_ABBREVIATIONS

__all__ = [
    "EnglishNormalizer",
    "NormalizedText",
    "Num2WordsNormalizer",
    "Replacement",
    "RuleNormalizer",
    "StreamNormalizer",
    "TextNormalizer",
    "WordAligner",
    "cardinal",
    "get_normalizer",
    "language_code",
    "normalize_text",
    "ordinal",
    "register_normalizer",
    "spell_digits",
    "year",
]


# ----------------------------------------------------------------------- offset map
@dataclass(frozen=True, slots=True)
class Replacement:
    """``original[start:end]`` is spoken as ``text``."""

    start: int
    end: int
    text: str


class NormalizedText:
    """Normalized text plus the map from its character offsets back to the original.

    Text outside the replaced spans is copied verbatim, so its offsets shift by a
    constant; a position inside a replacement maps to the whole replaced span.
    """

    __slots__ = ("_n_starts", "_spans", "original", "text")

    def __init__(self, original: str, replacements: Iterable[Replacement] = ()) -> None:
        parts: list[str] = []
        spans: list[tuple[int, int, int, int]] = []  # (norm start, norm end, orig start, end)
        pos = n = 0
        for r in sorted(replacements, key=lambda r: (r.start, r.end)):
            if r.start < pos or r.end < r.start or r.end > len(original):
                raise ValueError(f"invalid or overlapping replacement {r!r}")
            parts.append(original[pos : r.start])
            n += r.start - pos
            parts.append(r.text)
            spans.append((n, n + len(r.text), r.start, r.end))
            n += len(r.text)
            pos = r.end
        parts.append(original[pos:])
        self.original = original
        self.text = "".join(parts)
        self._spans = tuple(spans)
        self._n_starts = [s[0] for s in spans]

    @property
    def changed(self) -> bool:
        return self.text != self.original

    @property
    def replacements(self) -> tuple[Replacement, ...]:
        return tuple(Replacement(o0, o1, self.text[n0:n1]) for n0, n1, o0, o1 in self._spans)

    def _map(self, pos: int, *, end: bool) -> int:
        # the last span starting before (end) / at or before (start) ``pos``
        i = bisect.bisect_right(self._n_starts, pos - 1 if end else pos) - 1
        if i < 0:
            return pos
        _, n1, o0, o1 = self._spans[i]
        if (pos <= n1) if end else (pos < n1):
            return o1 if end else o0
        return o1 + (pos - n1)

    def to_original(self, start: int, end: int) -> tuple[int, int]:
        """The original span that ``text[start:end]`` was produced from."""
        s = self._map(start, end=False)
        return s, max(s, self._map(end, end=True))

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"NormalizedText({self.original!r} -> {self.text!r})"


# ------------------------------------------------------------------ word alignment
_TOKEN = re.compile(r"\S+")


def _key(word: str) -> str:
    return "".join(ch for ch in word.lower() if ch.isalnum())


class WordAligner:
    """Maps word timings reported on normalized text back to the original words.

    Feed the normalized texts in the order they are synthesized (:meth:`add`), then the
    TTS's word timings in order (:meth:`map`). Consecutive spoken words that come from one
    original word ("forty two dollars and fifty cents" <- "$42.50") are merged into one
    timing that carries the original word. A group is released as soon as its last spoken
    word is seen; one whose words are split across calls to :meth:`map` is held back until
    then, so call :meth:`finish` at the end of a segment to release it.
    """

    def __init__(self) -> None:
        self._keys: list[str] = []  # per normalized token
        self._group: list[int] = []  # token -> index in self._words
        self._words: list[str] = []  # original text of every group
        self._last: list[int] = []  # group -> index of its last token
        self._cursor = 0
        self._rest = ""  # unmatched remainder of the current token's key (split words)
        self._pending: tuple[int, float, float, float | None] | None = None

    def add(self, normalized: NormalizedText) -> None:
        text, original = normalized.text, normalized.original
        group_start = group_end = -1
        for m in _TOKEN.finditer(text):
            o0, o1 = normalized.to_original(m.start(), m.end())
            if group_end > o0:  # overlaps the previous token's original span: same word
                group_start, group_end = min(group_start, o0), max(group_end, o1)
                self._words[-1] = original[group_start:group_end].strip()
            else:
                group_start, group_end = o0, o1
                self._words.append(original[o0:o1].strip())
                self._last.append(-1)
            self._keys.append(_key(m.group()))
            self._group.append(len(self._words) - 1)
            self._last[-1] = len(self._keys) - 1

    def _locate(self, word: str) -> int | None:
        """Index of the normalized token ``word`` belongs to (advances the cursor)."""
        n = len(self._keys)
        if not n:
            return None
        key = _key(word)
        if self._rest and key and self._rest.startswith(key):  # a piece of a split token
            self._rest = self._rest[len(key) :]
            return self._cursor - 1
        self._rest = ""
        if key:
            for j in range(self._cursor, min(n, self._cursor + 8)):
                if self._keys[j] == key:
                    self._cursor = j + 1
                    return j
                if self._keys[j].startswith(key):
                    self._cursor = j + 1
                    self._rest = self._keys[j][len(key) :]
                    return j
        if self._cursor < n:
            self._cursor += 1
            return self._cursor - 1
        return n - 1

    def map(self, words: Sequence[WordTiming]) -> list[WordTiming]:
        """Timings for the original words; the last group may be held back."""
        out: list[WordTiming] = []
        for w in words:
            i = self._locate(w.word)
            if i is None:
                out.append(w)
                continue
            g = self._group[i]
            p = self._pending
            if p is not None and p[0] == g:
                self._pending = (g, p[1], max(p[2], w.end), p[3])
            else:
                if p is not None:
                    out.append(self._release(p))
                self._pending = (g, w.start, w.end, w.confidence)
            if i >= self._last[g] and not self._rest:  # the group's last word: complete
                out.append(self._release(self._pending))
                self._pending = None
        return out

    def _release(self, p: tuple[int, float, float, float | None]) -> WordTiming:
        return WordTiming(self._words[p[0]], p[1], p[2], p[3])

    def finish(self) -> list[WordTiming]:
        """Release the held-back group (end of a segment)."""
        p, self._pending = self._pending, None
        return [self._release(p)] if p is not None else []


# --------------------------------------------------------------------- normalizers
@runtime_checkable
class TextNormalizer(Protocol):
    """Rewrites text into its spoken form. ``context`` is text that precedes ``text``
    (already normalized and synthesized): rules may look at it, never rewrite it."""

    language: str

    def normalize(self, text: str, *, context: str = "") -> NormalizedText: ...


Handler = Callable[["re.Match[str]", str], "str | None"]


class RuleNormalizer:
    """A normalizer made of ``(pattern, handler)`` rules applied in one left-to-right pass.

    At each position the earliest match wins; on a tie, the rule listed first. A handler
    gets the match and the whole text and returns the spoken form, or ``None`` to decline
    (the other rules are then tried at that position). Spoken forms are padded with a
    space where they would otherwise run into a neighbouring letter or digit.
    """

    language = "und"

    def __init__(self, rules: Sequence[tuple[str | re.Pattern[str], Handler]]) -> None:
        self._rules = [(re.compile(p) if isinstance(p, str) else p, h) for p, h in rules]

    def normalize(self, text: str, *, context: str = "") -> NormalizedText:
        full = context + text
        base = len(context)
        out: list[Replacement] = []
        for r in self._scan(full, base):
            out.append(Replacement(r.start - base, r.end - base, r.text))
        return NormalizedText(text, out)

    def _scan(self, text: str, pos: int) -> list[Replacement]:
        out: list[Replacement] = []
        nxt: list[re.Match[str] | None] = [p.search(text, pos) for p, _ in self._rules]
        while True:
            candidates = [(m.start(), i) for i, m in enumerate(nxt) if m is not None]
            if not candidates:
                return out
            best = min(candidates)[1]
            m = nxt[best]
            assert m is not None
            spoken = self._rules[best][1](m, text) if m.end() > m.start() else None
            if spoken is None:
                nxt[best] = self._rules[best][0].search(text, m.start() + 1)
                continue
            if spoken != m.group():
                out.append(Replacement(m.start(), m.end(), _pad(spoken, text, m.start(), m.end())))
            pos = m.end()
            for i, other in enumerate(nxt):
                if other is not None and other.start() < pos:
                    nxt[i] = self._rules[i][0].search(text, pos)


def _pad(spoken: str, text: str, start: int, end: int) -> str:
    if spoken and start > 0 and text[start - 1].isalnum() and spoken[0].isalnum():
        spoken = " " + spoken
    if spoken and end < len(text) and text[end].isalnum() and spoken[-1].isalnum():
        spoken += " "
    return spoken


# --------------------------------------------------------------- English number words
_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen",
)  # fmt: skip
_TENS = (
    "_", "_", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
)  # fmt: skip
_SCALES = (
    "_", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion",
)  # fmt: skip
_ORDINAL_IRREGULAR = {
    "one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
    "nine": "ninth", "twelve": "twelfth",
}  # fmt: skip


def _under_thousand(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    words = [f"{_ONES[hundreds]} hundred"] if hundreds else []
    if rest >= 20:
        tens, ones = divmod(rest, 10)
        # "forty two", not "forty-two": espeak-based models (Piper) pause at the hyphen
        words.append(_TENS[tens] + (f" {_ONES[ones]}" if ones else ""))
    elif rest:
        words.append(_ONES[rest])
    return " ".join(words)


def cardinal(n: int) -> str:
    """``1234`` -> ``"one thousand two hundred thirty four"`` (American, no "and"; no hyphens)."""
    if n < 0:
        return "minus " + cardinal(-n)
    if n < 1000:
        return _under_thousand(n) or "zero"
    if n >= 1000 ** len(_SCALES):
        return spell_digits(str(n))
    groups: list[str] = []
    scale = 0
    while n:
        n, chunk = divmod(n, 1000)
        if chunk:
            groups.append(_under_thousand(chunk) + (f" {_SCALES[scale]}" if scale else ""))
        scale += 1
    return " ".join(reversed(groups))


def ordinal(n: int) -> str:
    """``21`` -> ``"twenty first"``."""
    words = cardinal(n)
    head, sep, last = words.rpartition(" ")
    stem, hyphen, unit = last.rpartition("-")
    if unit in _ORDINAL_IRREGULAR:
        unit = _ORDINAL_IRREGULAR[unit]
    elif unit.endswith("y"):
        unit = unit[:-1] + "ieth"
    else:
        unit += "th"
    return head + sep + stem + hyphen + unit


def spell_digits(digits: str) -> str:
    """``"0142"`` -> ``"zero one four two"`` (non-digits are skipped)."""
    return " ".join(_ONES[int(d)] for d in digits if d.isdigit())


def year(y: int) -> str:
    """``1999`` -> ``"nineteen ninety nine"``, ``2005`` -> ``"two thousand five"``,
    ``1905`` -> ``"nineteen oh five"``, ``2025`` -> ``"twenty twenty five"``."""
    if y < 1000 or y >= 10000 or (2000 <= y < 2010) or y % 1000 == 0:
        return cardinal(y)
    hi, lo = divmod(y, 100)
    if lo == 0:
        return f"{cardinal(hi)} hundred"
    if lo < 10:
        return f"{cardinal(hi)} oh {cardinal(lo)}"
    return f"{cardinal(hi)} {cardinal(lo)}"


def _int(s: str) -> int:
    return int(s.replace(",", ""))


def _decimal(int_part: str, frac: str | None) -> str:
    words = cardinal(_int(int_part)) if int_part else ""
    if frac:
        words = (words + " " if words else "") + "point " + spell_digits(frac)
    return words


def _number(s: str) -> str:
    """Cardinal or decimal for ``"1,234.5"``."""
    int_part, _, frac = s.partition(".")
    return _decimal(int_part, frac or None)


def _plural(value: str, singular: str, plural: str) -> str:
    return singular if value in ("1", "-1") else plural


_LETTER_NAMES = {
    "A": "ay", "B": "bee", "C": "see", "D": "dee", "E": "ee", "F": "eff", "G": "jee",
    "H": "aitch", "I": "eye", "J": "jay", "K": "kay", "L": "el", "M": "em", "N": "en",
    "O": "oh", "P": "pee", "Q": "cue", "R": "ar", "S": "ess", "T": "tee", "U": "you",
    "V": "vee", "W": "double-you", "X": "ex", "Y": "why", "Z": "zee",
}  # fmt: skip


def _letters(word: str) -> str:
    """``"NYC"`` -> ``"en why see"``: letter names as words, which every model tested
    (Pocket TTS, Kokoro, Piper) reads as letters; bare capitals are often read as a word
    ("Nysi")."""
    return " ".join(_LETTER_NAMES.get(ch, ch) for ch in word.upper())


# ------------------------------------------------------------------- English rules
_MONTHS = {
    "jan": "January", "feb": "February", "mar": "March", "apr": "April", "may": "May",
    "jun": "June", "jul": "July", "aug": "August", "sep": "September", "sept": "September",
    "oct": "October", "nov": "November", "dec": "December",
}  # fmt: skip
_MONTH_LIST = [
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December",
]  # fmt: skip
_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|"
    r"December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)"
)
_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
_CURRENCIES = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "€": ("euro", "euros", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
    "¥": ("yen", "yen", "", ""),
    "₹": ("rupee", "rupees", "paisa", "paise"),
    "USD": ("US dollar", "US dollars", "cent", "cents"),
    "EUR": ("euro", "euros", "cent", "cents"),
    "GBP": ("pound", "pounds", "penny", "pence"),
    "JPY": ("yen", "yen", "", ""),
    "INR": ("rupee", "rupees", "paisa", "paise"),
    "CAD": ("Canadian dollar", "Canadian dollars", "cent", "cents"),
    "AUD": ("Australian dollar", "Australian dollars", "cent", "cents"),
}
_MAGNITUDES = {
    "k": "thousand", "K": "thousand", "thousand": "thousand", "m": "million", "M": "million",
    "mn": "million", "million": "million", "b": "billion", "B": "billion", "bn": "billion",
    "billion": "billion", "t": "trillion", "T": "trillion", "trillion": "trillion",
}  # fmt: skip
# unit -> (singular, plural); ambiguous ones are only read when glued to the number
_UNITS: dict[str, tuple[str, str]] = {
    "km": ("kilometer", "kilometers"), "cm": ("centimeter", "centimeters"),
    "mm": ("millimeter", "millimeters"), "kg": ("kilogram", "kilograms"),
    "mg": ("milligram", "milligrams"), "lb": ("pound", "pounds"), "lbs": ("pounds", "pounds"),
    "oz": ("ounce", "ounces"), "mi": ("mile", "miles"), "mph": ("mile per hour", "miles per hour"),
    "km/h": ("kilometer per hour", "kilometers per hour"),
    "kph": ("kilometer per hour", "kilometers per hour"), "ft": ("foot", "feet"),
    "yd": ("yard", "yards"), "ml": ("milliliter", "milliliters"),
    "mL": ("milliliter", "milliliters"), "KB": ("kilobyte", "kilobytes"),
    "MB": ("megabyte", "megabytes"), "GB": ("gigabyte", "gigabytes"),
    "TB": ("terabyte", "terabytes"), "Hz": ("hertz", "hertz"), "kHz": ("kilohertz", "kilohertz"),
    "MHz": ("megahertz", "megahertz"), "GHz": ("gigahertz", "gigahertz"),
    "kW": ("kilowatt", "kilowatts"), "kWh": ("kilowatt hour", "kilowatt hours"),
    "ms": ("millisecond", "milliseconds"), "hrs": ("hours", "hours"), "hr": ("hour", "hours"),
    "mins": ("minutes", "minutes"), "min": ("minute", "minutes"),
    "secs": ("seconds", "seconds"), "sec": ("second", "seconds"),
    "°C": ("degree Celsius", "degrees Celsius"), "°F": ("degree Fahrenheit", "degrees Fahrenheit"),
    "°": ("degree", "degrees"),
    # glued only
    "m": ("meter", "meters"), "g": ("gram", "grams"), "s": ("second", "seconds"),
    "h": ("hour", "hours"), "L": ("liter", "liters"), "l": ("liter", "liters"),
    "W": ("watt", "watts"), "V": ("volt", "volts"), "in": ("inch", "inches"),
    "x": ("times", "times"),
}  # fmt: skip
_GLUED_ONLY = {"m", "g", "s", "h", "L", "l", "W", "V", "in", "x"}
_UNIT_RE = "|".join(re.escape(u) for u in sorted(_UNITS, key=len, reverse=True))
_TLDS = (
    "com|org|net|edu|gov|io|ai|dev|app|co|uk|us|ca|de|fr|es|it|nl|eu|info|biz|me|tv|ly|"
    "gg|xyz|tech|cloud|shop|store|site|online|ch|at|be|se|no|dk|fi|pl|jp|cn|in|au|nz|br|"
    "mx|ru|tr|kr|sg|hk|ie|pt|gr|cz|il|za|ar|cl|us"
)
_ID_CONTEXT = re.compile(
    r"(?:\b(?:number|no|code|zip|postcode|postal|id|pin|account|acct|order|confirmation|"
    r"reference|ref|ticket|flight|room|extension|ext|serial|tracking|invoice|booking|"
    r"policy|case|member|membership|card|ending|voucher|coupon|otp|passcode|unit|"
    r"model|part|sku|isbn)|#)\W*(?:(?:is|was|will|be|number|code|no|id|in|of|reads?)\W+){0,2}$",
    re.IGNORECASE,
)
_YEAR_CONTEXT = re.compile(
    r"\b(?:in|since|by|from|until|till|year|circa|around|before|after|during|early|"
    r"late|mid|back|spring|summer|fall|autumn|winter|class|vintage|established|founded|"
    r"born|est)\W*$",
    re.IGNORECASE,
)
_SPELL_ACRONYMS = frozenset({
    "USA", "UK", "EU", "UN", "UAE", "AI", "API", "ID", "CEO", "CFO", "CTO", "COO", "FAQ",
    "URL", "PDF", "SMS", "TV", "PC", "GPS", "ATM", "DIY", "FBI", "CIA", "BBC", "CNN", "HR",
    "IP", "VIP", "USB", "CPU", "GPU", "LLM", "TTS", "STT", "SQL", "HTML", "CSS", "NYC",
    "LA", "DC", "NY", "SF", "PS", "IBM", "HP", "AWS", "SUV", "RV", "ETA", "EST", "PST",
    "CST", "MST", "GMT", "UTC", "BTW", "FYI", "IOU", "DNA", "RNA", "ICU", "ER", "MD",
    "MBA", "BA", "BS", "ISP", "VPN", "LAN", "PR", "QA", "UX", "UI", "OS", "IQ", "KPI",
    "ROI",
})  # fmt: skip
_ROMAN = re.compile(r"^[IVXLCDM]+$")
_ABBREVIATIONS = {
    # title before a name -> word
    "Mr": "Mister", "Mrs": "Missus", "Ms": "Miz", "Prof": "Professor", "Gen": "General",
    "Capt": "Captain", "Lt": "Lieutenant", "Sgt": "Sergeant", "Col": "Colonel",
    "Gov": "Governor", "Sen": "Senator", "Rep": "Representative", "Rev": "Reverend",
    "Hon": "Honorable", "Mt": "Mount", "Ft": "Fort",
    # after a name
    "Jr": "Junior", "Sr": "Senior", "Ave": "Avenue", "Blvd": "Boulevard", "Rd": "Road",
    "Ln": "Lane", "Hwy": "Highway", "Inc": "Incorporated", "Ltd": "Limited",
    "Corp": "Corporation", "Co": "Company", "Bros": "Brothers", "Dept": "Department",
    "Univ": "University", "Assn": "Association",
    # anywhere
    "approx": "approximately", "etc": "et cetera", "vs": "versus", "est": "established",
    "min": "minimum", "max": "maximum", "tel": "telephone", "Tel": "telephone",
}  # fmt: skip
_TITLES = frozenset({
    "Mr", "Mrs", "Ms", "Prof", "Gen", "Capt", "Lt", "Sgt", "Col", "Gov", "Sen", "Rep",
    "Rev", "Hon", "Mt", "Ft",
})  # fmt: skip
_URL_SYMBOLS = {
    ".": "dot", "/": "slash", "-": "dash", "_": "underscore", "?": "question mark",
    "=": "equals", "&": "and", "#": "hash", "~": "tilde", "+": "plus", ":": "colon",
    "%": "percent", "@": "at",
}  # fmt: skip


def _say_chunk(chunk: str) -> str:
    """A piece of an e-mail/URL: words as they are, digit runs digit by digit."""
    parts = re.findall(r"[A-Za-z]+|\d+", chunk)
    out: list[str] = []
    for p in parts:
        if p.isdigit():
            out.append(spell_digits(p) if len(p) > 2 else cardinal(int(p)))
        elif p.lower() == "www":
            out.append(_letters("WWW"))
        elif len(p) == 1 or (p.isupper() and len(p) <= 3 and p not in ("COM", "ORG", "NET")):
            out.append(_letters(p))
        else:
            out.append(p.lower() if p.isupper() else p)
    return " ".join(out)


def _say_address(text: str) -> str:
    """``anna.lee@example.com`` -> ``anna dot lee at example dot com``."""
    tokens = re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9]", text)
    out = [_URL_SYMBOLS.get(t, "") if not t.isalnum() else _say_chunk(t) for t in tokens]
    return " ".join(w for w in out if w)


def _sentence_end(m: re.Match[str], text: str) -> bool:
    """True when the ``.`` the match consumed also ends the sentence."""
    if not m.group().endswith("."):
        return False
    rest = text[m.end() :]
    return not rest.strip() or bool(re.match(r"\s+[\"'”’)\]]*[A-Z]", rest))


def _prev_words(text: str, start: int, n: int = 40) -> str:
    return text[max(0, start - n) : start]


class EnglishNormalizer(RuleNormalizer):
    """Dependency-free English spoken-form normalizer.

    Covers cardinals, decimals, ordinals, negative numbers, currency, percentages, common
    units, times, dates, years and decades, fractions, ranges, phone numbers and IDs
    (digit by digit), alphanumeric codes, e-mail addresses, URLs, common abbreviations and
    spelled-out initialisms. Text it does not recognize is left untouched.

    Args:
        spell_acronyms: spell initialisms such as "NYC" and "API" as letter names.
        hyphenate_digit_groups: read phone number groups as "five-five-five" instead of
            "five five five" (Pocket TTS runs digit groups together otherwise; Piper
            reads the plain form better).
    """

    language = "en"

    def __init__(
        self, *, spell_acronyms: bool = True, hyphenate_digit_groups: bool = False
    ) -> None:
        self._group_joiner = "-" if hyphenate_digit_groups else " "
        rules: list[tuple[str, Handler]] = [
            (r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b",
             self._email),
            (rf"(?<![\w@.-])(?:https?://)?(?:www\.)?[A-Za-z0-9][A-Za-z0-9-]*(?:\.[A-Za-z0-9-]+)*"
             rf"\.(?:{_TLDS})\b(?![.-]?\w)(?:/[\w\-./?=&%#~+:]*[\w/])?", self._url),
            (r"(?<![\w+])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?![\w-])"
             r"|(?<![\w.,+-])\d{3}-\d{4}(?![\w-]|[.,]\d)|(?<![\w+])\+\d{8,15}\b", self._phone),
            (r"(?<![\w-])(\d{4})-(\d{2})-(\d{2})(?![\w-])", self._iso_date),
            (r"(?<![\w.,-])\d+(?:-\d+){2,}(?![\w-])", self._digit_groups),
            (r"(?<![\w.,/-])v?\d+\.\d+\.\d+(?:\.\d+)*(?!\w|\.\d)", self._version),
            (r"(?<![\w/.])(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})(?![\w/])", self._numeric_date),
            (r"(?<![\w:.,])(\d{1,2}):([0-5]\d)(?::([0-5]\d))?(?![\d:])"
             r"(?:\s?([AaPp])\.?\s?([Mm])\b\.?)?", self._time),
            (r"(?<![\w:.,])(\d{1,2})\s?([AaPp])\.?\s?([Mm])\b\.?", self._time_ampm),
            (rf"\b({_MONTH})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?!:\d|[.,]\d)"
             r"(?:,?\s+(\d{4})\b(?![.,]\d))?", self._month_day),
            (rf"(?<![\w.,])(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH})\b\.?"
             r"(?:,?\s+(\d{4})\b(?![.,]\d))?", self._day_month),
            (rf"\b({_MONTH})\.?,?\s+(\d{{4}})\b(?![.,]\d)", self._month_year),
            (r"(?<![\w.])([-−])?([$€£¥₹])\s?(\d{1,3}(?:,\d{3})+|\d+)?(?:\.(\d+))?"
             r"(?:(k|K|mn|m|M|bn|b|B|t|T)\b|\s?(thousand|million|billion|trillion)\b)?",
             self._currency),
            (rf"(?<![\w.,])([-−])?({_NUM})(?:\.(\d+))?\s?(USD|EUR|GBP|JPY|INR|CAD|AUD)\b",
             self._currency_code),
            (rf"(?<![\w.,])([-−])?({_NUM}|(?=\.\d))(?:\.(\d+))?\s?%", self._percent),
            (r"(?<![\w'’])(?:(\d\d)(\d)0|['’](\d)0|([1-9])0)s\b", self._decade),
            (rf"(?<![\w.,])([-−])?({_NUM})(?:\.(\d+))?(\s?)({_UNIT_RE})(?![\w/])", self._unit),
            (rf"(?<![\w.,])({_NUM})(st|nd|rd|th)\b", self._ordinal),
            (r"(?<![\w.,/-])(\d+(?:\.\d+)?)\s?[-–—]\s?(\d+(?:\.\d+)?)(?![\w-]|[.,]\d)(\s?%)?",
             self._range),
            (r"(?<![\w/.,])(\d{1,2})/(\d{1,2})(?![\w/]|[.,]\d)", self._fraction),
            (r"(?<![\w@./-])(?=[A-Za-z0-9-]*\d)(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*"
             r"(?![\w@]|\.\w)", self._alnum),
            (r"(?<![\w.,])#(\d+)\b", self._hash_number),
            (r"(?:(?<=[\s(])|^)([-−])?\.(\d+)(?![\w.]|,\d)"
             r"|(?<![\w.,])([-−](?=\d))?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?![\w]|[.,]\d)",
             self._plain_number),
            (r"(?<![\w.])(e\.g|i\.e|a\.k\.a|U\.S\.A|U\.S|U\.K|a\.m|p\.m|A\.M|P\.M)\.?(?!\w)",
             self._dotted),
            (r"\b(Dr|St|No|Nos|" + "|".join(_ABBREVIATIONS) + r")\.(?!\w)", self._abbreviation),
            (r"\b(vs|w/o|w/)(?!\w)", self._bare_abbreviation),
            (r"(?<![\w'’.-])[A-Z][A-Za-z]{1,4}(?![\w'’-])", self._acronym if spell_acronyms
             else lambda m, t: None),
            (r"(?<=\s)[&+=~@](?=\s)|(?<=\w)&(?=\w)", self._symbol),
        ]  # fmt: skip
        super().__init__(rules)

    # --------------------------------------------------------------- handlers
    @staticmethod
    def _email(m: re.Match[str], text: str) -> str:
        return _say_address(m.group())

    @staticmethod
    def _url(m: re.Match[str], text: str) -> str:
        return _say_address(re.sub(r"^https?://", "", m.group(), flags=re.IGNORECASE))

    def _group(self, digits: str) -> str:
        return spell_digits(digits).replace(" ", self._group_joiner)

    def _phone(self, m: re.Match[str], text: str) -> str:
        s = m.group()
        plus = s.startswith("+")
        groups = re.findall(r"\d+", s)
        if plus and len(groups) == 1:  # compact international number
            return "plus " + spell_digits(groups[0])
        words = [self._group(g) for g in groups]
        if plus:
            words[0] = "plus " + words[0]
        return ", ".join(words)

    def _digit_groups(self, m: re.Match[str], text: str) -> str:
        return ", ".join(self._group(g) for g in m.group().split("-"))

    @staticmethod
    def _version(m: re.Match[str], text: str) -> str:
        s = m.group()
        prefix = "version " if s.startswith("v") else ""
        return prefix + " point ".join(cardinal(int(p)) for p in s.lstrip("v").split("."))

    @staticmethod
    def _date(month: int, day: int, y: str | None) -> str | None:
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return None
        spoken = f"{_MONTH_LIST[month - 1]} {ordinal(day)}"
        if y:
            spoken += f" {year(int(y))}"  # no comma: Piper garbles the sentence with one
        return spoken

    def _iso_date(self, m: re.Match[str], text: str) -> str | None:
        return self._date(int(m.group(2)), int(m.group(3)), m.group(1))

    def _numeric_date(self, m: re.Match[str], text: str) -> str | None:
        a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if len(y) == 2:
            y = str(2000 + int(y)) if int(y) < 50 else str(1900 + int(y))
        if a > 12 and b <= 12:  # day/month
            a, b = b, a
        return self._date(a, b, y)

    @staticmethod
    def _clock(h: int, mins: int, ampm: str | None) -> str | None:
        if h > 24 or (ampm and h > 12):
            return None
        spoken = cardinal(h)
        if mins == 0:
            spoken += "" if ampm else (" o'clock" if h <= 12 else " hundred")
        elif mins < 10:
            spoken += f" oh {cardinal(mins)}"
        else:
            spoken += f" {cardinal(mins)}"
        if ampm:
            spoken += f" {ampm}"
        return spoken

    def _time(self, m: re.Match[str], text: str) -> str | None:
        ampm = _letters(m.group(4) + m.group(5)) if m.group(4) else None
        spoken = self._clock(int(m.group(1)), int(m.group(2)), ampm)
        if spoken is None:
            return None
        if m.group(3) and int(m.group(3)):
            spoken += f" and {cardinal(int(m.group(3)))} seconds"
        return spoken + ("." if _sentence_end(m, text) else "")

    def _time_ampm(self, m: re.Match[str], text: str) -> str | None:
        ampm = _letters(m.group(2) + m.group(3))
        spoken = self._clock(int(m.group(1)), 0, ampm)
        if spoken is None:
            return None
        return spoken + ("." if _sentence_end(m, text) else "")

    def _month_day(self, m: re.Match[str], text: str) -> str | None:
        month = _MONTHS[m.group(1)[:3].lower()]
        return self._date(_MONTH_LIST.index(month) + 1, int(m.group(2)), m.group(3))

    def _day_month(self, m: re.Match[str], text: str) -> str | None:
        day = int(m.group(1))
        month = _MONTHS[m.group(2)[:3].lower()]
        if not 1 <= day <= 31:
            return None
        spoken = f"{ordinal(day)} of {month}"
        if m.group(3):
            spoken += f" {year(int(m.group(3)))}"
        elif _sentence_end(m, text):
            spoken += "."
        return spoken

    @staticmethod
    def _month_year(m: re.Match[str], text: str) -> str | None:
        month = _MONTHS[m.group(1)[:3].lower()]
        return f"{month} {year(int(m.group(2)))}"

    @staticmethod
    def _money(
        sign: str | None, unit: tuple[str, str, str, str], whole: str | None,
        frac: str | None, magnitude: str | None,
    ) -> str | None:  # fmt: skip
        one, many, cent, cents = unit
        minus = "minus " if sign else ""
        if whole is None and frac is None:
            return None
        if magnitude:
            amount = _decimal(whole or "0", frac)
            return f"{minus}{amount} {_MAGNITUDES[magnitude]} {many}"
        value = _int(whole) if whole else 0
        if frac and (len(frac) != 2 or not cent):
            return f"{minus}{_decimal(whole or '0', frac)} {many}"
        sub = int(frac) if frac else 0
        parts = []
        if value or not sub:
            parts.append(f"{cardinal(value)} {one if value == 1 else many}")
        if sub:
            parts.append(f"{cardinal(sub)} {cent if sub == 1 else cents}")
        return minus + " and ".join(parts)

    def _currency(self, m: re.Match[str], text: str) -> str | None:
        magnitude = m.group(5) or m.group(6)
        return self._money(m.group(1), _CURRENCIES[m.group(2)], m.group(3), m.group(4), magnitude)

    def _currency_code(self, m: re.Match[str], text: str) -> str | None:
        return self._money(m.group(1), _CURRENCIES[m.group(4)], m.group(2), m.group(3), None)

    @staticmethod
    def _percent(m: re.Match[str], text: str) -> str:
        minus = "minus " if m.group(1) else ""
        return f"{minus}{_decimal(m.group(2), m.group(3))} percent"

    @staticmethod
    def _unit(m: re.Match[str], text: str) -> str | None:
        unit = m.group(5)
        if unit in _GLUED_ONLY and m.group(4):
            return None
        if unit == "x" and m.group(3):
            return None
        minus = "minus " if m.group(1) else ""
        value = m.group(2) + (f".{m.group(3)}" if m.group(3) else "")
        one, many = _UNITS[unit]
        return f"{minus}{_decimal(m.group(2), m.group(3))} {_plural(value, one, many)}"

    @staticmethod
    def _decade(m: re.Match[str], text: str) -> str | None:
        century, decade, short, bare = m.groups()
        if bare and not re.search(
            r"\b(?:the|in|early|late|mid|her|his|their|my|your|our|its)\W*$",
            _prev_words(text, m.start()),
            re.IGNORECASE,
        ):
            return None  # "30s" without a decade context: thirty seconds
        d = int(decade or short or bare)
        word = {0: "", 1: "tens"}.get(d) or _TENS[d][:-1] + "ies"
        if century is None:
            return word
        c = int(century)
        if d == 0:
            return f"{cardinal(c * 100) if c % 10 == 0 else cardinal(c) + ' hundred'}s"
        return f"{cardinal(c)} {word}"

    @staticmethod
    def _ordinal(m: re.Match[str], text: str) -> str:
        return ordinal(_int(m.group(1)))

    @staticmethod
    def _range(m: re.Match[str], text: str) -> str | None:
        a, b = m.group(1), m.group(2)
        if "." not in a + b and len(a) == 4 and len(b) == 4 and 1000 <= int(a) < 2100:
            return f"{year(int(a))} to {year(int(b))}"
        return f"{_number(a)} to {_number(b)}" + (" percent" if m.group(3) else "")

    @staticmethod
    def _fraction(m: re.Match[str], text: str) -> str | None:
        a, b = int(m.group(1)), int(m.group(2))
        if (a, b) in ((24, 7), (50, 50)):
            return f"{cardinal(a)} {cardinal(b)}"
        if b == 0:
            return None
        if b == 2:
            unit = "half" if a == 1 else "halves"
        elif b == 4:
            unit = "quarter" if a == 1 else "quarters"
        elif 3 <= b <= 10 and a < b:
            unit = ordinal(b) + ("" if a == 1 else "s")
        else:
            return f"{cardinal(a)} over {cardinal(b)}"
        return f"{cardinal(a)} {unit}"

    @staticmethod
    def _alnum(m: re.Match[str], text: str) -> str | None:
        words: list[str] = []
        for part in m.group().split("-"):
            for p in re.findall(r"[A-Za-z]+|\d+", part):
                if p.isdigit():
                    words.append(
                        cardinal(int(p)) if len(p) <= 2 and p[0] != "0" else spell_digits(p)
                    )
                elif p.isupper() and len(p) <= 3:
                    words.append(_letters(p))
                else:
                    words.append(p)
        return " ".join(words)

    @staticmethod
    def _hash_number(m: re.Match[str], text: str) -> str:
        n = m.group(1)
        return "number " + (cardinal(int(n)) if len(n) <= 4 and n[0] != "0" else spell_digits(n))

    @staticmethod
    def _plain_number(m: re.Match[str], text: str) -> str:
        if m.group(2) is not None:  # ".5" / "-.5"
            sign, whole, frac = m.group(1), None, m.group(2)
        else:
            sign, whole, frac = m.group(3), m.group(4), m.group(5)
        minus = "minus " if sign else ""
        if frac is not None:
            return minus + _decimal(whole or "", frac)
        assert whole is not None
        if "," in whole:
            return minus + cardinal(_int(whole))
        before = _prev_words(text, m.start())
        if (
            not sign
            and len(whole) > 1
            and (
                whole[0] == "0"
                or len(whole) >= 7
                or (len(whole) >= 3 and _ID_CONTEXT.search(before))
            )
        ):
            return spell_digits(whole)
        if (
            not sign
            and len(whole) == 4
            and 1000 <= int(whole) < 2100
            and (_YEAR_CONTEXT.search(before) or (int(whole) % 100 == 0 and int(whole) < 2000))
        ):
            return year(int(whole))
        return minus + cardinal(int(whole))

    @staticmethod
    def _dotted(m: re.Match[str], text: str) -> str:
        key = m.group(1).lower()
        spoken = {
            "e.g": "for example", "i.e": "that is", "a.k.a": "also known as",
            "u.s.a": _letters("USA"), "u.s": _letters("US"), "u.k": _letters("UK"),
            "a.m": _letters("AM"), "p.m": _letters("PM"),
        }[key]  # fmt: skip
        return spoken + ("." if _sentence_end(m, text) and key not in ("e.g", "i.e") else "")

    @staticmethod
    def _abbreviation(m: re.Match[str], text: str) -> str | None:
        word = m.group(1)
        rest = text[m.end() :]
        next_is_name = bool(re.match(r"\s+[A-Z]", rest))
        if word in ("No", "Nos"):
            return "number" if re.match(r"\s?\d", rest) else None
        if word == "Dr":
            spoken = "Doctor" if next_is_name else "Drive"
        elif word == "St":
            # "Main St." (after a name that does not start the sentence) vs "St. Louis"
            after_name = re.search(r"[^.!?\s]\s+[A-Z][a-z]*\s*$", text[: m.start()])
            spoken = "Saint" if next_is_name and not after_name else "Street"
        else:
            spoken = _ABBREVIATIONS[word]
        if word not in _TITLES and word != "Dr" and _sentence_end(m, text):
            if spoken == "Saint":
                return spoken
            spoken += "."
        return spoken

    @staticmethod
    def _bare_abbreviation(m: re.Match[str], text: str) -> str:
        return {"vs": "versus", "w/": "with", "w/o": "without"}[m.group(1)]

    @staticmethod
    def _acronym(m: re.Match[str], text: str) -> str | None:
        word = m.group()
        if word == "OK":
            return "okay"
        if not word.isupper() and word != "PhD":
            return None
        if word in _SPELL_ACRONYMS or word == "PhD":
            pass
        elif _ROMAN.match(word) or any(v in word for v in "AEIOU"):
            return None
        else:
            # shouting: neighbouring words in capitals too
            before = re.search(r"([A-Za-z]+)\W*$", text[: m.start()])
            after = re.match(r"\W*([A-Za-z]+)", text[m.end() :])
            if any(n is not None and n.group(1).isupper() and len(n.group(1)) > 1
                   for n in (before, after)):  # fmt: skip
                return None
        return _letters(word.replace("h", "H"))

    @staticmethod
    def _symbol(m: re.Match[str], text: str) -> str | None:
        return {"&": "and", "+": "plus", "=": "equals", "~": "about", "@": "at"}.get(m.group())


# --------------------------------------------------------------- other languages
_PERCENT_WORDS = {
    "de": "Prozent", "fr": "pour cent", "es": "por ciento", "it": "per cento",
    "pt": "por cento", "nl": "procent", "pl": "procent", "tr": "yüzde", "ru": "процентов",
    "sv": "procent", "da": "procent", "no": "prosent", "fi": "prosenttia", "cs": "procent",
}  # fmt: skip
_COMMA_DECIMAL = frozenset({
    "de", "fr", "es", "it", "pt", "nl", "pl", "tr", "ru", "sv", "da", "no", "fi", "cs",
    "ro", "hu", "uk", "id",
})  # fmt: skip


class Num2WordsNormalizer(RuleNormalizer):
    """Numbers (cardinals, decimals, percentages) for any language ``num2words`` supports.

    Needs the optional ``num2words`` package. Languages that write decimals with a comma
    (``3,5``) and group thousands with a dot or a space are handled accordingly.
    """

    def __init__(self, language: str) -> None:
        from num2words import num2words  # type: ignore[import-not-found,unused-ignore]

        self.language = language
        self._words: Callable[..., str] = num2words
        num2words(1, lang=language)  # raises NotImplementedError for unknown languages
        comma = language in _COMMA_DECIMAL
        dec, grp = (",", r"[.   ]") if comma else (r"\.", ",")
        number = rf"(?<![\w{dec}])([-−])?(\d{{1,3}}(?:{grp}\d{{3}})+|\d+)(?:{dec}(\d+))?"
        rules: list[tuple[str, Handler]] = [
            (number + r"\s?%", self._percent),
            (number + r"(?![\w]|[.,]\d)", self._number),
        ]
        super().__init__(rules)

    def _value(self, m: re.Match[str]) -> str:
        whole = re.sub(r"\D", "", m.group(2))
        value: float | int = float(f"{whole}.{m.group(3)}") if m.group(3) else int(whole)
        if m.group(1):
            value = -value
        return str(self._words(value, lang=self.language))

    def _number(self, m: re.Match[str], text: str) -> str:
        return self._value(m)

    def _percent(self, m: re.Match[str], text: str) -> str:
        word = _PERCENT_WORDS.get(self.language, "%")
        return f"{self._value(m)} {word}"


# ----------------------------------------------------------------------- registry
_LANGUAGE_NAMES = {
    "english": "en", "eng": "en", "french": "fr", "german": "de", "spanish": "es",
    "italian": "it", "portuguese": "pt", "dutch": "nl", "polish": "pl", "turkish": "tr",
    "russian": "ru", "japanese": "ja", "chinese": "zh", "cmn": "zh", "korean": "ko",
    "hindi": "hi", "arabic": "ar",
}  # fmt: skip
_FACTORIES: dict[str, Callable[[], TextNormalizer]] = {"en": EnglishNormalizer}


def language_code(language: str | None) -> str | None:
    """``"en-US"``, ``"en_us"``, ``"english"``, ``"english_2026-04"`` -> ``"en"``."""
    if not language:
        return None
    base = re.split(r"[-_\s]", language.strip().lower())[0]
    return _LANGUAGE_NAMES.get(base, base) or None


def register_normalizer(language: str, factory: Callable[[], TextNormalizer]) -> None:
    """Use ``factory()`` for ``language`` (an ISO 639-1 code), replacing any default."""
    code = language_code(language) or language
    _FACTORIES[code] = factory
    get_normalizer.cache_clear()


@cache
def get_normalizer(language: str | None = "en") -> TextNormalizer | None:
    """The normalizer for ``language``, or ``None`` when none is available (a registered
    one first, then :class:`Num2WordsNormalizer` if ``num2words`` is installed)."""
    code = language_code(language)
    if code is None:
        return None
    factory = _FACTORIES.get(code)
    if factory is not None:
        return factory()
    if is_installed("num2words"):
        try:
            return Num2WordsNormalizer(code)
        except (NotImplementedError, ImportError):
            return None
    return None


def normalize_text(text: str, language: str | None = "en") -> str:
    """``text`` in spoken form (unchanged when no normalizer exists for ``language``).

    Usable as (part of) a cascade ``text_filter``; note that the transcript then shows the
    spoken form too, whereas ``TTS(normalize=True)`` keeps the original text in it.
    """
    normalizer = get_normalizer(language)
    return normalizer.normalize(text).text if normalizer is not None else text


# ---------------------------------------------------------------------- streaming
_PLAIN = re.compile(r"[\"'“‘(]*[A-Za-z][a-z'’]*[\"'”’),;:!?]*")
_SENTENCE_END = re.compile(r"[^.][.!?]+$")
_PLAIN_END = re.compile(r"[A-Za-z][a-z'’]*[,;:!?.]+[\"'”’)]*")


class StreamNormalizer:
    """Normalizes text that arrives in arbitrary pieces without splitting a number, date
    or address across two normalization calls.

    :meth:`push` returns the part of the buffered text that can safely be normalized now:
    everything up to the last whitespace between two plain words (or the end, after a
    word that closes a clause or sentence); :meth:`flush` returns the rest. The last few
    words already released are passed on as ``context``.
    """

    def __init__(self, normalizer: TextNormalizer, *, context_chars: int = 48) -> None:
        self.normalizer = normalizer
        self._buf = ""
        self._context = ""
        self._context_chars = context_chars

    def _safe_cut(self) -> int:
        buf = self._buf
        tokens = list(_TOKEN.finditer(buf))
        if not tokens:
            return 0
        last = tokens[-1]
        if last.end() < len(buf):
            core = last.group().rstrip("\"'”’)")
            if _PLAIN_END.fullmatch(last.group()) or _SENTENCE_END.search(core):
                word = core.rstrip(",;:!?.").lower().lstrip("\"'“‘(")
                if not core.endswith(".") or (
                    len(word) > 1 and word not in _SENTENCE_ABBREVIATIONS
                ):
                    return len(buf)  # after a word that closes a clause or a sentence
        for prev, nxt in zip(reversed(tokens[:-1]), reversed(tokens[1:]), strict=True):
            if _PLAIN.fullmatch(prev.group()) and _PLAIN.fullmatch(nxt.group()):
                return nxt.start()
        return 0

    def _release(self, text: str) -> NormalizedText | None:
        if not text:
            return None
        result = self.normalizer.normalize(text, context=self._context)
        self._context = (self._context + text)[-self._context_chars :]
        return result

    def push(self, text: str) -> NormalizedText | None:
        self._buf += text
        cut = self._safe_cut()
        ready, self._buf = self._buf[:cut], self._buf[cut:]
        return self._release(ready)

    def flush(self) -> NormalizedText | None:
        ready, self._buf = self._buf, ""
        return self._release(ready)
