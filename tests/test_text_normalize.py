"""Spoken-form text normalization: rules, offset map, word alignment, streaming, TTS."""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator

import pytest

from voice_agent_next.providers.mock import MockTTS
from voice_agent_next.stt import WordTiming
from voice_agent_next.text.normalize import (
    EnglishNormalizer,
    NormalizedText,
    Num2WordsNormalizer,
    Replacement,
    StreamNormalizer,
    WordAligner,
    cardinal,
    get_normalizer,
    language_code,
    normalize_text,
    ordinal,
    register_normalizer,
    spell_digits,
    year,
)
from voice_agent_next.text.sentences import SentenceSegmenter, split_sentences
from voice_agent_next.tts import ChunkedStream

# (written by an LLM, spoken form)
CASES: list[tuple[str, str]] = [
    # cardinals, decimals, negatives
    ("I have 3 cats.", "I have three cats."),
    ("It seats 12 people", "It seats twelve people"),
    ("Only 0 left", "Only zero left"),
    ("About 1,234 people", "About one thousand two hundred thirty four people"),
    ("We sold 1,000,000 units", "We sold one million units"),
    ("It has 58213 rows", "It has fifty eight thousand two hundred thirteen rows"),
    ("Pi is 3.14", "Pi is three point one four"),
    ("Add .5 cup", "Add point five cup"),
    ("It fell to -7 today", "It fell to minus seven today"),
    ("Agent 007 here", "Agent zero zero seven here"),
    ("It is 2.0 now", "It is two point zero now"),
    # ordinals
    ("She came 1st", "She came first"),
    ("the 2nd and 3rd floors", "the second and third floors"),
    ("our 21st year", "our twenty first year"),
    ("the 112th street", "the one hundred twelfth street"),
    # currency
    ("The total is $42.50.", "The total is forty two dollars and fifty cents."),
    ("It costs $1.", "It costs one dollar."),
    ("Only $0.99", "Only ninety nine cents"),
    ("Pay $12.00 today", "Pay twelve dollars today"),
    ("That is $1,250 in total", "That is one thousand two hundred fifty dollars in total"),
    ("A $1.5 million deal", "A one point five million dollars deal"),
    ("Raised $5k", "Raised five thousand dollars"),
    ("It's $3B", "It's three billion dollars"),
    ("It's €12.50", "It's twelve euros and fifty cents"),
    ("Just £3", "Just three pounds"),
    ("It's £1.01", "It's one pound and one penny"),
    ("Only ¥500", "Only five hundred yen"),
    ("Send 20 USD", "Send twenty US dollars"),
    ("a -$5 balance", "a minus five dollars balance"),
    # percentages and units
    ("Up by 12% this year", "Up by twelve percent this year"),
    ("A 2.5% fee", "A two point five percent fee"),
    ("Save 10-20% now", "Save ten to twenty percent now"),
    ("Run 5km today", "Run five kilometers today"),
    ("It weighs 2.5 kg", "It weighs two point five kilograms"),
    ("It weighs 1 kg", "It weighs one kilogram"),
    ("Drive at 60 mph", "Drive at sixty miles per hour"),
    ("A 5m pole", "A five meters pole"),
    ("Put 5 m apart", "Put five m apart"),
    ("It is -5°C outside", "It is minus five degrees Celsius outside"),
    ("Set it to 72°F", "Set it to seventy two degrees Fahrenheit"),
    ("A 512 GB disk", "A five hundred twelve gigabytes disk"),
    ("It takes 25 min", "It takes twenty five minutes"),
    ("It is 2x faster", "It is two times faster"),
    # times
    ("See you at 4:30 PM.", "See you at four thirty pee em."),
    ("See you at 4:30 PM today", "See you at four thirty pee em today"),
    ("Open at 9:05 am", "Open at nine oh five ay em"),
    ("Leaves at 7 AM.", "Leaves at seven ay em."),
    ("Call at 5pm", "Call at five pee em"),
    ("Meet at 5 p.m. tomorrow", "Meet at five pee em tomorrow"),
    ("Meet at 5 p.m. Then leave", "Meet at five pee em. Then leave"),
    ("Come at 12:00", "Come at twelve o'clock"),
    ("Lands at 16:45", "Lands at sixteen forty five"),
    ("Lands at 18:00", "Lands at eighteen hundred"),
    # dates and years
    ("on March 3rd", "on March third"),
    ("on July 14, 2025.", "on July fourteenth, twenty twenty five."),
    ("by Sept 5", "by September fifth"),
    ("on the 3rd of March", "on the third of March"),
    ("on 3 March 2024", "on third of March, twenty twenty four"),
    ("due 2025-03-04", "due March fourth, twenty twenty five"),
    ("due 3/4/2025", "due March fourth, twenty twenty five"),
    ("due 25/12/2024", "due December twenty fifth, twenty twenty four"),
    ("in March 2025", "in March twenty twenty five"),
    ("in 1999", "in nineteen ninety nine"),
    ("since 2005", "since two thousand five"),
    ("by 2030", "by twenty thirty"),
    ("founded in 1905", "founded in nineteen oh five"),
    ("in 1900", "in nineteen hundred"),
    ("from 2023-2024", "from twenty twenty three to twenty twenty four"),
    ("the 1990s", "the nineteen nineties"),
    ("the '80s", "the eighties"),
    ("in the 90s", "in the nineties"),
    ("the 2000s", "the two thousands"),
    ("wait 30s", "wait thirty seconds"),
    # ranges and fractions
    ("in 3-5 days", "in three to five days"),
    ("pages 10–12", "pages ten to twelve"),
    ("add 1/2 cup", "add one half cup"),
    ("3/4 full", "three quarters full"),
    ("2/3 done", "two thirds done"),
    ("open 24/7", "open twenty four seven"),
    # phone numbers and IDs, digit by digit
    ("Call 555-0142.", "Call five-five-five, zero-one-four-two."),
    (
        "Call (415) 555-0142",
        "Call four-one-five, five-five-five, zero-one-four-two",
    ),
    (
        "Call +1 415-555-0142",
        "Call plus one, four-one-five, five-five-five, zero-one-four-two",
    ),
    (
        "Dial +442071838750",
        "Dial plus four four two zero seven one eight three eight seven five zero",
    ),
    (
        "Call 1-800-555-1234",
        "Call one, eight-zero-zero, five-five-five, one-two-three-four",
    ),
    ("Your order number is 58213.", "Your order number is five eight two one three."),
    ("The zip code was 94107?", "The zip code was nine four one zero seven?"),
    ("Your PIN is 1234", "Your PIN is one two three four"),
    ("Order #4521 shipped", "Order number four thousand five hundred twenty one shipped"),
    ("Ref 1234567", "Ref one two three four five six seven"),
    ("Code 1234567 works", "Code one two three four five six seven works"),
    ("gate B12", "gate bee twelve"),
    ("an A320 jet", "an ay three two zero jet"),
    ("COVID-19 news", "COVID nineteen news"),
    ("Use v1.2.3", "Use version one point two point three"),
    ("No. 5 wins", "number five wins"),
    # e-mails and URLs
    ("Mail anna.lee@example.com.", "Mail anna dot lee at example dot com."),
    ("Mail support@acme.io", "Mail support at acme dot io"),
    ("Visit docs.example.org/setup.", "Visit docs dot example dot org slash setup."),
    ("See https://www.example.com", "See double-you double-you double-you dot example dot com"),
    ("Go to example.com/a-b", "Go to example dot com slash ay dash bee"),
    # abbreviations and initialisms
    ("Dr. Smith is in", "Doctor Smith is in"),
    ("Mrs. Jones and Mr. Lee", "Missus Jones and Mister Lee"),
    ("Visit St. Louis", "Visit Saint Louis"),
    ("I live on Main St. It is nice", "I live on Main Street. It is nice"),
    ("Take a fruit, e.g. an apple", "Take a fruit, for example an apple"),
    ("Pens, pencils, etc.", "Pens, pencils, et cetera."),
    ("Acme Inc. makes it", "Acme Incorporated makes it"),
    ("cats vs dogs", "cats versus dogs"),
    ("It's approx. 5", "It's approximately five"),
    ("the U.S. market", "the you ess market"),
    ("Fly to NYC", "Fly to en why see"),
    ("Ask the AI", "Ask the ay eye"),
    ("It is OK", "It is okay"),
    ("NASA said", "NASA said"),
    ("Chapter IV", "Chapter IV"),
    ("R&D team", "R and D team"),
    ("Tom & Jerry", "Tom and Jerry"),
    # left alone
    ("Hello there!", "Hello there!"),
    ("", ""),
    ("No, thanks.", "No, thanks."),
    ("Mr. Brown", "Mister Brown"),
]


@pytest.mark.parametrize(("text", "spoken"), CASES)
def test_english_spoken_forms(text: str, spoken: str) -> None:
    assert normalize_text(text) == spoken


def test_number_words() -> None:
    assert cardinal(0) == "zero"
    assert cardinal(105) == "one hundred five"
    assert cardinal(1_000_001) == "one million one"
    assert cardinal(-42) == "minus forty two"
    assert cardinal(10**21) == "one " + " ".join(["zero"] * 21)  # beyond the scales
    assert [ordinal(n) for n in (1, 2, 3, 4, 5, 8, 9, 12, 20, 21, 100, 1000)] == [
        "first", "second", "third", "fourth", "fifth", "eighth", "ninth", "twelfth",
        "twentieth", "twenty first", "one hundredth", "one thousandth",
    ]  # fmt: skip
    assert spell_digits("0142") == "zero one four two"
    assert [year(y) for y in (1999, 2000, 2005, 2010, 1905, 1800, 999)] == [
        "nineteen ninety nine", "two thousand", "two thousand five", "twenty ten",
        "nineteen oh five", "eighteen hundred", "nine hundred ninety nine",
    ]  # fmt: skip


def test_bench_smoke_hard_texts_are_spoken() -> None:
    from voice_agent_next.bench.tracks.tts import load_texts

    for item in load_texts("smoke").texts:
        spoken = normalize_text(item.text)
        assert not any(ch.isdigit() for ch in spoken), spoken
        assert "@" not in spoken and "$" not in spoken and "%" not in spoken


# ---------------------------------------------------------------------- offset map
def test_offset_map_points_back_to_the_original() -> None:
    n = EnglishNormalizer().normalize("Pay $12.50 by 5pm, ok")
    assert n.text == "Pay twelve dollars and fifty cents by five pee em, ok"
    assert n.changed
    i = n.text.index("dollars")
    assert n.to_original(i, i + len("dollars")) == (4, 10)  # "$12.50"
    assert n.original[slice(*n.to_original(0, 3))] == "Pay"
    j = n.text.index("by")
    assert n.original[slice(*n.to_original(j, j + 2))] == "by"
    k = n.text.index("five")
    assert n.original[slice(*n.to_original(k, len(n.text)))] == "5pm, ok"
    assert [r.text for r in n.replacements] == ["twelve dollars and fifty cents", "five pee em"]


def test_offset_map_rejects_overlaps_and_pads_glued_words() -> None:
    with pytest.raises(ValueError):
        NormalizedText("abcdef", [Replacement(0, 3, "x"), Replacement(2, 4, "y")])
    unchanged = NormalizedText("same text")
    assert not unchanged.changed and unchanged.to_original(2, 4) == (2, 4)
    assert normalize_text("5%off") == "five percent off"


def test_context_is_seen_but_not_rewritten() -> None:
    norm = EnglishNormalizer()
    assert norm.normalize("58213", context="order number is ").text == "five eight two one three"
    assert norm.normalize("58213").text == "fifty eight thousand two hundred thirteen"


# ---------------------------------------------------------------- word alignment
def _timings(text: str) -> list[WordTiming]:
    return [WordTiming(w, float(i), i + 0.5) for i, w in enumerate(text.split())]


def test_word_aligner_merges_spoken_words_into_the_original_word() -> None:
    n = EnglishNormalizer().normalize("Total $42.50, due at 4:30 PM today.")
    aligner = WordAligner()
    aligner.add(n)
    words = aligner.map(_timings(n.text)) + aligner.finish()
    assert [w.word for w in words] == ["Total", "$42.50,", "due", "at", "4:30 PM", "today."]
    money = words[1]
    assert (money.start, money.end) == (1.0, 6.5)  # "forty" .. "cents,"


def test_word_aligner_holds_a_group_until_it_is_complete() -> None:
    n = EnglishNormalizer().normalize("Pay $3 now")
    aligner = WordAligner()
    aligner.add(n)
    timings = _timings(n.text)  # Pay three dollars now
    assert [w.word for w in aligner.map(timings[:2])] == ["Pay"]  # "three" of "$3" waits
    assert [w.word for w in aligner.map(timings[2:])] == ["$3", "now"]
    assert aligner.finish() == []


def test_word_aligner_survives_split_and_missing_words() -> None:
    n = EnglishNormalizer().normalize("It is 21 now")  # It is twenty one now
    aligner = WordAligner()
    aligner.add(n)
    pieces = [WordTiming(w, i, i + 1) for i, w in enumerate(["It", "twenty", "one", "now"])]
    words = aligner.map(pieces) + aligner.finish()
    assert [w.word for w in words] == ["It", "21", "now"]


def test_word_aligner_across_several_texts() -> None:
    norm = EnglishNormalizer()
    aligner = WordAligner()
    first, second = norm.normalize("Call 555-0142."), norm.normalize("Thanks $5")
    aligner.add(first)
    aligner.add(second)
    words = aligner.map(_timings(first.text + " " + second.text)) + aligner.finish()
    assert [w.word for w in words] == ["Call", "555-0142.", "Thanks", "$5"]


# --------------------------------------------------------------------- streaming
STREAM_TEXTS = [
    "Your order number is 58213, and it should arrive within 3 to 5 business days.",
    "The total comes to $42.50, including tax. Your appointment is on Tuesday, "
    "March 3rd at 4:30 PM. Send it to anna.lee@example.com or call 555-0142.",
    "Dr. Smith said the contract was signed on July 14, 2025 by Mrs. Jones etc. Thanks!",
]


def _pieces(text: str, step: int) -> Iterator[str]:
    for i in range(0, len(text), step):
        yield text[i : i + step]


@pytest.mark.parametrize("text", STREAM_TEXTS)
@pytest.mark.parametrize("step", [1, 2, 5, 13])
def test_stream_normalizer_matches_whole_text_normalization(text: str, step: int) -> None:
    stream = StreamNormalizer(EnglishNormalizer())
    parts = [stream.push(p) for p in _pieces(text, step)] + [stream.flush()]
    released = [p for p in parts if p is not None]
    assert "".join(p.original for p in released) == text
    assert "".join(p.text for p in released) == normalize_text(text)
    assert len(released) > 1  # it does release text before the end


def test_stream_normalizer_releases_at_a_sentence_end_without_waiting() -> None:
    stream = StreamNormalizer(EnglishNormalizer())
    first = stream.push("It costs $5")
    assert first is not None and first.text == "It "  # "costs $5" may continue
    out = stream.push(". ")
    assert out is not None and out.text.endswith("five dollars. ")
    assert stream.push("Meet Dr. ") is None  # "Dr." waits for the name
    rest = stream.push("Smith. ")
    assert rest is not None and rest.text == "Meet Doctor Smith. "


@pytest.mark.parametrize(
    "text",
    ["The contract was signed on July 14, 2025, by both of us and our lawyers.",
     "It leaves at 4:30 PM and arrives at 9:15 PM, so plan for a long trip there."],
)  # fmt: skip
def test_segmenter_never_splits_a_number_group(text: str) -> None:
    seg = SentenceSegmenter(min_chars=4, first_segment_max_chars=20, max_chars=40)
    parts = [s for p in _pieces(text, 3) for s in seg.push(p)] + seg.flush()
    joined = " | ".join(parts)
    assert "14, | 2025" not in joined and "14 | 2025" not in joined
    assert "4:30 | PM" not in joined and "9:15 | PM" not in joined
    assert split_sentences("Call 555-0142. Pay $42.50 now.") == [
        "Call 555-0142.",
        "Pay $42.50 now.",
    ]


# ------------------------------------------------------------------- languages
def test_language_codes_and_registry() -> None:
    assert language_code("en-US") == language_code("english_2026-04") == "en"
    assert language_code("French") == "fr"
    assert language_code(None) is None
    assert isinstance(get_normalizer("en-gb"), EnglishNormalizer)
    assert get_normalizer(None) is None


def test_register_a_custom_language(monkeypatch: pytest.MonkeyPatch) -> None:
    from voice_agent_next.text import normalize as mod

    class Shout:
        language = "xx"

        def normalize(self, text: str, *, context: str = "") -> NormalizedText:
            return NormalizedText(text, [Replacement(0, len(text), text.upper())])

    monkeypatch.setattr(mod, "_FACTORIES", dict(mod._FACTORIES))
    register_normalizer("xx", Shout)
    try:
        assert normalize_text("hi", "xx") == "HI"
    finally:
        mod.get_normalizer.cache_clear()


@pytest.fixture
def fake_num2words(monkeypatch: pytest.MonkeyPatch) -> None:
    words = {1: "un", 3: "trois", 1500: "mille cinq cents", 3.5: "trois virgule cinq"}

    def num2words(value: float, lang: str = "en") -> str:
        if lang not in ("fr", "de"):
            raise NotImplementedError(lang)
        return ("moins " if value < 0 else "") + words[abs(value)]

    module = types.ModuleType("num2words")
    module.num2words = num2words  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "num2words", module)
    from voice_agent_next.text import normalize as mod

    monkeypatch.setattr(mod, "is_installed", lambda name: True)
    mod.get_normalizer.cache_clear()
    yield
    mod.get_normalizer.cache_clear()


@pytest.mark.usefixtures("fake_num2words")
def test_num2words_for_other_languages() -> None:
    norm = get_normalizer("fr")
    assert isinstance(norm, Num2WordsNormalizer)
    assert norm.normalize("J'ai 3 chats et 1 500 livres, 3,5 % de plus.").text == (
        "J'ai trois chats et mille cinq cents livres, trois virgule cinq pour cent de plus."
    )
    assert norm.normalize("Il fait -3").text == "Il fait moins trois"
    assert get_normalizer("xx") is None  # num2words does not know it


def test_without_num2words_other_languages_are_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voice_agent_next.text import normalize as mod

    monkeypatch.setattr(mod, "is_installed", lambda name: False)
    mod.get_normalizer.cache_clear()
    try:
        assert get_normalizer("fr") is None
        assert normalize_text("J'ai 3 chats", "fr") == "J'ai 3 chats"
    finally:
        mod.get_normalizer.cache_clear()


# ------------------------------------------------------------------------ TTS
class WordMockTTS(MockTTS):
    """Mock TTS that reports one word timing per word of the text it synthesizes: the
    first ``early_words`` with the first audio chunk, the others after the audio."""

    early_words = 3

    def _synthesize(self, text: str, *, voice: str | None) -> ChunkedStream:
        return _WordChunkedStream(self, text, voice=voice)


class _WordChunkedStream(ChunkedStream):
    async def _run(self) -> None:
        tts: WordMockTTS = self._tts  # type: ignore[assignment]
        words = self.text.split()
        per = tts.audio_duration_for(self.text) / max(1, len(words))
        timings = [WordTiming(w, i * per, (i + 1) * per) for i, w in enumerate(words)]
        first = [True]
        early = tts.early_words

        def push(frame: object) -> None:
            self._push_audio(frame, words=timings[:early] if first[0] else None)  # type: ignore[arg-type]
            first[0] = False

        await tts.generate(self.text, push)
        if len(timings) > early:  # the rest after the audio, like Pocket TTS
            from voice_agent_next.audio.frame import AudioFrame
            from voice_agent_next.tts import SynthesizedAudio

            empty = AudioFrame.empty(tts.sample_rate)
            rest = timings[early:]
            self._send(SynthesizedAudio(empty, self._request_id, self._segment_id, words=rest))


def test_normalize_option_and_provider_defaults() -> None:
    tts = MockTTS()
    assert tts.normalize is None and tts.normalizer_for() is None  # mock: off by default
    tts.normalize = True
    assert isinstance(tts.normalizer_for(), EnglishNormalizer)
    tts.normalize = "fr"
    assert tts.normalizer_for() is get_normalizer("fr")
    custom = EnglishNormalizer(spell_acronyms=False)
    tts.normalize = custom
    assert tts.normalizer_for() is custom

    from voice_agent_next.providers.kokoro import KokoroTTS
    from voice_agent_next.providers.pocket_tts import PocketTTS
    from voice_agent_next.providers.sherpa_onnx import SherpaOnnxTTS

    assert PocketTTS.normalize_by_default and KokoroTTS.normalize_by_default
    assert SherpaOnnxTTS.normalize_by_default

    from voice_agent_next.providers.cartesia import CartesiaTTS
    from voice_agent_next.providers.elevenlabs import ElevenLabsTTS
    from voice_agent_next.providers.openai.tts import OpenAITTS

    assert not any(c.normalize_by_default for c in (CartesiaTTS, ElevenLabsTTS, OpenAITTS))


async def test_synthesize_speaks_the_normalized_text_and_reports_the_original() -> None:
    tts = WordMockTTS(normalize=True)
    text = "Your total is $42.50, due March 3rd."
    items = [a async for a in tts.synthesize(text)]
    assert tts.requests == [normalize_text(text)]
    assert items[0].text == text  # the segment text is what the LLM wrote
    words = [w.word for a in items for w in a.words or []]
    assert words == ["Your", "total", "is", "$42.50,", "due", "March 3rd."]
    starts = [w.start for a in items for w in a.words or []]
    assert starts == sorted(starts)


async def test_synthesize_without_normalization_is_untouched() -> None:
    tts = WordMockTTS()
    items = [a async for a in tts.synthesize("It is $5.")]
    assert tts.requests == ["It is $5."]
    assert [w.word for a in items for w in a.words or []] == ["It", "is", "$5."]


async def test_sentence_adapter_streams_normalized_sentences() -> None:
    tts = WordMockTTS(normalize=True)
    text = "Your order number is 58213. The total is $42.50, thanks!"
    stream = tts.stream()
    for piece in _pieces(text, 2):
        stream.push_text(piece)
    stream.end_input()
    items = [a async for a in stream]
    await stream.aclose()
    assert " ".join(tts.requests) == normalize_text(text)
    assert [a.text for a in items if a.text] == [
        "Your order number is 58213.",
        "The total is $42.50, thanks!",
    ]
    words = [w.word for a in items for w in a.words or []]
    assert words == text.split()


async def test_native_stream_normalizes_pushed_text() -> None:
    tts = MockTTS(streaming=True, normalize=True)
    text = "It costs $42.50 and ships on July 14, 2025. Call 555-0142."
    stream = tts.stream()
    for piece in _pieces(text, 3):
        stream.push_text(piece)
    stream.end_input()
    _ = [a async for a in stream]
    await stream.aclose()
    assert tts.requests == [normalize_text(text)]


# -------------------------------------------------------------- cascade barge-in
async def test_barge_in_truncation_keeps_the_numbers_the_llm_wrote() -> None:
    import asyncio

    from tests.test_session import Recorder, mock_cascade, speak, wait_for
    from voice_agent_next import Agent, ChatMessage
    from voice_agent_next.providers.mock import synth_speech
    from voice_agent_next.transports import LoopbackTransport

    sentences = [
        "Your order number is 58213.",
        "The total is $42.50, and it arrives on July 14, 2025 at 4:30 PM.",
        "Thanks for waiting so patiently today, and have a good evening.",
    ]
    answer = " ".join(sentences)
    tts = WordMockTTS(normalize=True, chars_per_second=20, realtime_factor=1.0)
    tts.early_words = 1000  # all word timings with the first audio (Cartesia, ElevenLabs)
    session = mock_cascade(transcripts=["where is my order"], responses=[answer], tts=tts)
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await session.start(Agent("x"), transport)
    await speak(transport, 0.5, 0.7)
    await wait_for(lambda: session.agent_state.value == "speaking", 5)
    await asyncio.sleep(3.2)  # into the second sentence
    await transport.play_user_audio(synth_speech(0.4, 16_000), realtime=False)  # barge in
    await wait_for(lambda: bool(rec.of("interrupted")), 5)
    await session.aclose()

    assert tts.requests[:2] == [normalize_text(s) for s in sentences[:2]]
    # expected: original words whose (first) spoken word started before the cut
    end = round(rec.of("interrupted")[0].played * 1000) / 1000.0
    heard: list[str] = []
    offset = 0.0
    for sentence in sentences:
        normalized = EnglishNormalizer().normalize(sentence)
        duration = tts.audio_duration_for(normalized.text)
        aligner = WordAligner()
        aligner.add(normalized)
        words = aligner.map(_timings_over(normalized.text, offset, duration)) + aligner.finish()
        heard += [w.word for w in words if w.start < end]
        offset += duration
    assert "58213." in heard and 0 < len(heard) < len(answer.split())
    msg = session.connection.chat_ctx.get(rec.of("interrupted")[0].item_id)  # type: ignore[attr-defined]
    assert isinstance(msg, ChatMessage) and msg.interrupted
    assert msg.text == " ".join(heard)
    assert answer.startswith(msg.text)  # the LLM's own words, never "five eight two..."


def _timings_over(text: str, offset: float, duration: float) -> list[WordTiming]:
    words = text.split()
    per = duration / len(words)
    return [WordTiming(w, offset + i * per, offset + (i + 1) * per) for i, w in enumerate(words)]
