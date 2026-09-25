"""The shared Whisper hallucination guard (``voice_agent_next.stt_guard``): multilingual
artifact and suspect lists, patterns, normalization, dict segments and configuration.
Provider integration is tested in ``tests/providers/test_faster_whisper.py`` and
``tests/providers/test_mlx.py``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from voice_agent_next.errors import ConfigurationError
from voice_agent_next.stt_guard import (
    ARTIFACT_PATTERNS,
    ARTIFACT_PHRASES,
    ARTIFACTS_BY_LANGUAGE,
    SUSPECT_PHRASES,
    SUSPECTS_BY_LANGUAGE,
    HallucinationGuard,
    artifact_phrases,
    coerce_guard,
    normalize,
    suspect_phrases,
)


def seg(text: str, **stats: Any) -> SimpleNamespace:
    """A faster-whisper-like segment: confident speech unless ``stats`` say otherwise."""
    fields = {"avg_logprob": -0.2, "no_speech_prob": 0.01, "compression_ratio": 1.4}
    return SimpleNamespace(text=text, **{**fields, **stats})


GUARD = HallucinationGuard()


def test_normalize_folds_case_accents_and_punctuation() -> None:
    assert normalize("  Sous-titres réalisés par la communauté d'Amara.org ") == (
        "sous titres realises par la communaute d'amara org"
    )
    assert normalize("İzlediğiniz için TEŞEKKÜRLER!") == "izlediginiz icin tesekkurler"
    assert normalize("Straße") == "strasse"
    assert normalize("It’s") == "it's"  # typographic apostrophe
    assert normalize("谢谢观看！") == "谢谢观看"
    assert normalize("시청해주셔서 감사합니다.") == "시청해주셔서 감사합니다"  # Hangul recomposed
    assert normalize("...♪") == ""


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("en", " Thanks for watching!"),
        ("en", " Transcribed by https://otter.ai"),
        ("de", " Untertitel im Auftrag des ZDF für funk, 2017"),
        ("de", " Untertitel im Auftrag des ZDF, 2020"),  # another year: a pattern
        ("de", " Vielen Dank fürs Zuschauen!"),
        ("de", " Copyright WDR 2021"),
        ("de", " Untertitel der Amara.org-Community"),
        ("es", " Subtítulos realizados por la comunidad de Amara.org"),
        ("es", " ¡Gracias por ver el video!"),
        ("es", " Suscríbete al canal"),
        ("fr", " Sous-titres réalisés para la communauté d'Amara.org"),
        ("fr", " Merci d'avoir regardé cette vidéo !"),
        ("fr", " Sous-titres par David Liou"),  # a credit with any name: a pattern
        ("fr", " — Sous-titrage ST'501 —"),
        ("it", " Sottotitoli e revisione a cura di QTSS"),
        ("it", " Grazie per la visione"),
        ("pt", " Obrigado por assistir."),
        ("pt", " Legendas pela comunidade Amara.org"),
        ("nl", " Ondertiteld door de Amara.org gemeenschap"),
        ("pl", " Napisy stworzone przez społeczność Amara.org"),
        ("tr", " Altyazı M.K."),
        ("tr", " altyazi mk"),  # without the dotless i and the dots
        ("tr", " İzlediğiniz için teşekkürler."),
        ("tr", " Abone olmayı unutmayın!"),
        ("ru", " Продолжение следует..."),
        ("ru", " Субтитры добавил DimaTorzok"),  # any verb: a pattern
        ("ru", " Спасибо за просмотр!"),
        ("zh", " 字幕由Amara.org社区提供"),
        ("zh", " 請不吝點贊 訂閱 轉發 打賞支持明鏡與點點欄目"),
        ("zh", " 谢谢观看!"),
        ("ja", " ご視聴ありがとうございました"),
        ("ja", " チャンネル登録をお願いします!"),
        ("ko", " 시청해주셔서 감사합니다."),
        ("ar", " ترجمة نانسي قنقر"),
        ("ar", " شكرا على المشاهدة"),
    ],
)
def test_known_artifacts_in_many_languages(language: str, text: str) -> None:
    assert GUARD.reason(seg(text), vad_confidence=0.99) == "known artifact", (language, text)
    assert GUARD.is_artifact(text)
    assert HallucinationGuard.disabled().reason(seg(text)) is None


@pytest.mark.parametrize(
    "text",
    [
        " Gracias por ver el documento conmigo.",  # starts like an outro, is not one
        " Danke für die schnelle Hilfe.",
        " Ich schaue gerne das ZDF.",
        " Merci, j'ai une question sur ma facture.",
        " Altyazıları açabilir misin?",
        " 字幕太小了。",  # "the subtitles are too small"
        " Спасибо, а сколько это стоит?",
        " How do I subscribe to the newsletter?",
    ],
)
def test_real_sentences_are_kept(text: str) -> None:
    assert GUARD.reason(seg(text), vad_confidence=0.3) is None  # even with a weak VAD


@pytest.mark.parametrize(
    "text",
    [" Danke.", " Gracias.", " Merci.", " Grazie.", " Obrigado.", " Teşekkürler.", " Спасибо.",
     " 谢谢。", " ありがとう。", " 감사합니다.", " شكرا", " Thank you."],
)  # fmt: skip
def test_multilingual_suspects_need_other_evidence(text: str) -> None:
    assert GUARD.reason(seg(text)) is None  # a clear "thank you" is an answer
    assert GUARD.reason(seg(text), vad_confidence=0.95) is None
    assert GUARD.reason(seg(text), vad_confidence=0.4) == "suspect phrase on weak evidence"
    assert GUARD.reason(seg(text, no_speech_prob=0.3)) == "suspect phrase on weak evidence"
    assert GUARD.reason(seg(text, avg_logprob=-0.9)) == "suspect phrase on weak evidence"


def test_default_lists_cover_every_language() -> None:
    assert {"en", "de", "es", "fr", "it", "pt", "nl", "tr", "ru", "zh", "ja", "ko"} <= set(
        ARTIFACTS_BY_LANGUAGE
    ) & set(SUSPECTS_BY_LANGUAGE)
    guard = HallucinationGuard()
    for phrases in ARTIFACTS_BY_LANGUAGE.values():
        for phrase in phrases:
            assert normalize(phrase), phrase
            assert guard.reason(seg(f" {phrase.upper()}."), vad_confidence=1.0) == (
                "known artifact"
            ), phrase
    # no suspect is an artifact (suspects are real answers, dropped on weak evidence only)
    for phrases in SUSPECTS_BY_LANGUAGE.values():
        for phrase in phrases:
            assert not guard.is_artifact(phrase), phrase
    assert set(ARTIFACT_PHRASES) == {p for ps in ARTIFACTS_BY_LANGUAGE.values() for p in ps}
    assert set(SUSPECT_PHRASES) == {p for ps in SUSPECTS_BY_LANGUAGE.values() for p in ps}
    assert guard.patterns == ARTIFACT_PATTERNS


def test_phrase_lists_by_language() -> None:
    assert artifact_phrases() == ARTIFACT_PHRASES
    german = artifact_phrases("de")
    assert "vielen dank fürs zuschauen" in german and "thanks for watching" not in german
    both = suspect_phrases("en", "de-DE", "pt_BR")
    assert {"thank you", "danke", "obrigado"} <= set(both) and "merci" not in both
    assert len(both) == len(set(both))
    with pytest.raises(ValueError, match="no phrase list"):
        artifact_phrases("xx")

    english_only = HallucinationGuard(suspects=suspect_phrases("en"))
    assert english_only.reason(seg(" Danke."), vad_confidence=0.1) is None
    assert english_only.reason(seg(" Thank you."), vad_confidence=0.1) is not None


def test_dict_segments_like_mlx_whisper() -> None:
    segments = [
        {"text": " Hallo, wie geht's?", "avg_logprob": -0.3, "no_speech_prob": 0.02},
        {"text": " Untertitel im Auftrag des ZDF, 2018", "avg_logprob": -0.4},
        {"text": " Danke.", "avg_logprob": -0.2, "no_speech_prob": 0.5},
        {"text": " ok ok", "compression_ratio": 3.0},
        {"text": " Tschüss!", "avg_logprob": "not a number"},  # ignored, not an error
    ]
    verdict = GUARD.filter(segments)
    assert [s["text"] for s in verdict.kept] == [" Hallo, wie geht's?", " Tschüss!"]
    assert verdict.dropped == [
        ("Untertitel im Auftrag des ZDF, 2018", "known artifact"),
        ("Danke.", "suspect phrase on weak evidence"),
        ("ok ok", "repetitive"),
    ]


def test_patterns_are_configurable() -> None:
    guard = HallucinationGuard(patterns=(r"untertitel von .*",))
    assert guard.reason(seg(" Untertitel von Max Mustermann")) == "known artifact"
    assert guard.reason(seg(" Untertitel im Auftrag des ZDF, 2019")) is None  # pattern replaced
    assert guard.reason(seg(" Thanks for watching!")) == "known artifact"  # phrases kept
    no_patterns = HallucinationGuard(patterns=())
    assert no_patterns.reason(seg(" Субтитры добавил DimaTorzok")) is None
    with pytest.raises(ValueError, match="invalid pattern"):
        HallucinationGuard(patterns=("(",))


def test_coerce_guard() -> None:
    guard = HallucinationGuard(vad_threshold=0.7)
    assert coerce_guard(guard, provider="p") is guard
    assert coerce_guard(True, provider="p") == HallucinationGuard()
    assert coerce_guard(False, provider="p") == HallucinationGuard.disabled()
    assert coerce_guard(None, provider="p") == HallucinationGuard.disabled()
    from_yaml = coerce_guard(
        {"suspects": ["Danke"], "patterns": [r"foo \d+"], "vad_threshold": None}, provider="p"
    )
    assert from_yaml.suspects == ("danke",) and from_yaml.patterns == (r"foo \d+",)
    assert from_yaml.reason(seg(" Foo 12")) == "known artifact"
    for bad in ({"nope": 1}, {"max_ngram": 0}, {"patterns": ["("]}):
        with pytest.raises(ConfigurationError, match="p: hallucination_guard"):
            coerce_guard(bad, provider="p")
    with pytest.raises(ConfigurationError, match="must be a bool"):
        coerce_guard("yes", provider="p")  # type: ignore[arg-type]
