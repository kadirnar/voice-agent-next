from __future__ import annotations

import pytest

from voice_agent_next.text import SentenceSegmenter, split_sentences, strip_markdown, tts_clean


def stream(text: str, seg: SentenceSegmenter, step: int = 3) -> list[str]:
    out: list[str] = []
    for i in range(0, len(text), step):
        out += seg.push(text[i : i + step])
    return out + seg.flush()


def test_basic_sentences_streamed_char_by_char() -> None:
    text = "Hello there, how are you today? I am fine. Thanks for asking!"
    assert stream(text, SentenceSegmenter(min_chars=5), step=1) == [
        "Hello there, how are you today?",
        "I am fine.",
        "Thanks for asking!",
    ]


def test_sentence_released_only_after_following_whitespace() -> None:
    seg = SentenceSegmenter(min_chars=1)
    assert seg.push("It costs 3.") == []  # could be "3.14"
    assert seg.push("14 dollars. Next") == ["It costs 3.14 dollars."]
    assert seg.flush() == ["Next"]


@pytest.mark.parametrize(
    "text",
    ["Dr. Smith will see you now.", "Talk to Mr. and Mrs. Jones today.", "J. K. Rowling wrote it.",
     "Use e.g. a spoon."],
)  # fmt: skip
def test_abbreviations_do_not_split(text: str) -> None:
    assert split_sentences(text) == [text]


def test_short_fragments_are_merged() -> None:
    seg = SentenceSegmenter(min_chars=15)
    assert stream("Hi. Yes. This is a longer sentence. Ok.", seg) == [
        "Hi. Yes. This is a longer sentence.",
        "Ok.",
    ]


def test_first_segment_can_be_shorter() -> None:
    seg = SentenceSegmenter(min_chars=20, first_segment_min_chars=3)
    assert seg.push("Sure! Let me check that for you right now. ") == [
        "Sure!",
        "Let me check that for you right now.",
    ]


def test_long_run_on_text_is_split_at_clauses() -> None:
    seg = SentenceSegmenter(min_chars=5, max_chars=60)
    text = (
        "one two three four five, six seven eight nine ten eleven twelve thirteen fourteen fifteen"
    )
    parts = seg.push(text) + seg.flush()
    assert len(parts) >= 2
    assert all(len(p) <= 60 for p in parts)
    assert " ".join(parts) == text


def test_cjk_punctuation() -> None:
    assert split_sentences("你好。今天天气很好！你呢？") == ["你好。", "今天天气很好！", "你呢？"]


def test_paragraph_and_list_breaks() -> None:
    parts = split_sentences("Here are options:\n- first option\n- second option")
    assert parts[0] == "Here are options:"
    assert len(parts) == 3


def test_segmenter_validation_and_reset() -> None:
    with pytest.raises(ValueError):
        SentenceSegmenter(min_chars=10, max_chars=5)
    seg = SentenceSegmenter()
    seg.push("partial text")
    seg.reset()
    assert seg.flush() == []


def test_strip_markdown() -> None:
    md = (
        "# Title\n**Bold** and *italic* with `code` and [a link](http://x.y).\n- item one\n1. first"
    )
    out = strip_markdown(md)
    for token in ("#", "**", "`", "](", "- item"):
        assert token not in out
    assert "Bold and italic with code and a link." in out
    assert "1. first" in out


def test_tts_clean_removes_emoji_and_normalizes_space() -> None:
    assert tts_clean("Great   job! 🎉👍  **Really**") == "Great job! Really"


def test_first_long_sentence_is_split_at_its_first_clause() -> None:
    seg = SentenceSegmenter(min_chars=10, first_segment_min_chars=4, first_segment_max_chars=40)
    text = "Sure, the store opens at nine tomorrow and closes at six in the evening. Anything else?"
    parts = seg.push(text) + seg.flush()
    assert parts == [
        "Sure,",
        "the store opens at nine tomorrow and closes at six in the evening.",
        "Anything else?",
    ]


def test_first_clause_split_only_applies_to_long_first_sentences() -> None:
    seg = SentenceSegmenter(min_chars=10, first_segment_min_chars=4, first_segment_max_chars=40)
    assert seg.push("Yes, we are open. ") == ["Yes, we are open."]  # short: kept whole
    later = "Later sentences, even long ones with commas, are never split early. "
    assert seg.push(later) == [later.strip()]


def test_first_clause_split_while_streaming_and_word_fallback() -> None:
    seg = SentenceSegmenter(min_chars=10, first_segment_min_chars=4, first_segment_max_chars=20)
    out: list[str] = []
    for word in ["Well,", "I", "think", "the", "answer", "depends", "on"]:
        out += seg.push(word + " ")
    assert out == ["Well,"]  # released before the sentence is complete
    seg2 = SentenceSegmenter(min_chars=5, first_segment_min_chars=4, first_segment_max_chars=10)
    parts = seg2.push("averyveryvery longrunonsentence withoutanyclause boundaries at all here ")
    assert parts and len(parts[0]) <= 20
