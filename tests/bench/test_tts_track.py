"""T3 TTS track: metrics math, round-trip scoring, MOS plumbing and end-to-end runs."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from voice_agent_next.audio import AudioFrame
from voice_agent_next.bench import load_run
from voice_agent_next.bench.mos import (
    DNSMOS,
    dnsmos_calibrate,
    dnsmos_windows,
    make_mos_predictor,
)
from voice_agent_next.bench.results import ITEMS_FILE, MANIFEST_FILE, REPORT_FILE, SUMMARY_FILE
from voice_agent_next.bench.roundtrip import (
    entity_matches,
    resolve_normalizer,
    score_round_trip,
    squash,
)
from voice_agent_next.bench.tracks.tts import (
    Capture,
    TextSet,
    TTSOptions,
    TTSText,
    chunk_gaps,
    load_texts,
    measure_capture,
    render_tts_report,
    run_tts_benchmark,
    silence_bounds,
    simulate_playout,
)
from voice_agent_next.cli.main import app
from voice_agent_next.providers.mock import MockSTT, MockTTS
from voice_agent_next.tts import TTS, ChunkedStream

from .helpers import concat, silence, tone

RATE = 24_000

# ---------------------------------------------------------------------- round-trip scoring

_, ENGLISH = resolve_normalizer("en")


def test_resolve_normalizer_reuses_the_asr_track() -> None:
    assert resolve_normalizer("en")[0] == "whisper-english"
    assert resolve_normalizer("de")[0] == "whisper-basic"
    assert resolve_normalizer("en", "none")[0] == "none"
    with pytest.raises(ValueError, match="unknown normalizer"):
        resolve_normalizer("en", "nope")


def test_squash() -> None:
    assert squash("4:30 pm") == "430 pm"
    assert squash("5 5 5 0 1 4 2") == "5550142"
    assert squash("$42.50, ok") == "4250 ok"
    assert squash("july 14 2025", join_digits=False) == "july 14 2025"


@pytest.mark.parametrize(
    ("alternatives", "transcript", "found"),
    [
        (["4:30"], "See you at four thirty.", True),
        (["58213"], "fifty eight two thirteen", True),
        (["58213"], "five eight two one three", True),
        (["58213"], "fifty eight two fourteen", False),
        (["$42.50"], "forty two dollars and fifty cents", True),
        (["12%"], "twelve percent", True),
        (["555-0142", "5550142"], "five five five zero one four two", True),
        (["nyc", "new york city"], "Your flight to New York City", True),
        (["Dr. Smith"], "Doctor Smith will see you", True),
        (["12"], "one hundred twenty", False),  # whole words only
        (["July 14"], "The contract was signed on July 14, 2025.", True),
        (["2025"], "signed on July fourteenth, twenty twenty-five", True),
    ],
)
def test_entity_matching(alternatives: list[str], transcript: str, found: bool) -> None:
    assert entity_matches(alternatives, transcript, ENGLISH) is found


def test_score_round_trip() -> None:
    score = score_round_trip(
        "Your total is $42.50.", "your total is forty two dollars and fifty cents",
        language="en", normalize=ENGLISH, entities=[["$42.50"], ["ok"]],
    )  # fmt: skip
    assert score.words.errors == 0 and score.words.ref_len == 4 and score.headline == "wer"
    assert score.entities_missed == ("ok",)
    zh = score_round_trip("你好世界", "你好", language="zh", normalize=lambda t: t)
    assert zh.headline == "cer" and zh.chars.errors == 2 and zh.chars.ref_len == 4


# ------------------------------------------------------------------------- metrics math


def test_simulated_playout_counts_underruns_and_stall() -> None:
    arrivals = [(0.1, 0.2), (0.25, 0.2), (0.6, 0.2), (0.805, 0.1)]
    play = simulate_playout(arrivals, underrun_threshold=0.010)
    assert play.starts == pytest.approx((0.1, 0.3, 0.6, 0.805))
    # chunk 3 arrived 100 ms after the player ran dry; chunk 4 only 5 ms (below threshold)
    assert play.underruns == 1
    assert play.stall == pytest.approx(0.105)
    assert play.time_of(arrivals, 0.0) == pytest.approx(0.1)
    assert play.time_of(arrivals, 0.25) == pytest.approx(0.35)
    assert play.time_of(arrivals, 0.45) == pytest.approx(0.65)
    assert play.time_of(arrivals, 10.0) is None
    assert chunk_gaps(arrivals) == pytest.approx([0.15, 0.35, 0.205])
    assert simulate_playout([]).starts == ()


def test_silence_bounds() -> None:
    audio = concat(silence(0.3, RATE), tone(0.5, RATE), silence(0.2, RATE))
    lead, trail = silence_bounds(audio)
    assert lead == pytest.approx(0.3, abs=0.002)
    assert trail == pytest.approx(0.2, abs=0.011)
    assert silence_bounds(silence(0.5, RATE)) == (None, None)
    assert silence_bounds(AudioFrame.empty(RATE)) == (None, None)


def test_measure_capture_counts_leading_silence_in_ttfa() -> None:
    cap = Capture()
    cap.add(0.100, silence(0.2, RATE))  # first chunk: silence only
    cap.add(0.150, tone(0.4, RATE))
    cap.add(0.900, tone(0.2, RATE))  # arrives 0.2 s after the player ran dry at 0.7
    cap.end = 0.95
    m = measure_capture(cap, RATE)
    assert m["ttfb_ms"] == pytest.approx(100.0)
    assert m["leading_silence_ms"] == pytest.approx(200.0, abs=2)
    assert m["ttfa_ms"] == pytest.approx(300.0, abs=2)  # first chunk + 200 ms of silence
    assert m["audio_ms"] == pytest.approx(800.0)
    assert m["rtf"] == pytest.approx(0.95 / 0.8)
    assert m["underruns"] == 1 and m["stall_ms"] == pytest.approx(200.0)
    assert m["chunks"] == 3 and m["chunk_gap_max_ms"] == pytest.approx(750.0)
    assert m["no_speech"] is False
    empty = measure_capture(Capture(), RATE)
    assert empty["ttfa_ms"] is None and empty["rtf"] is None and empty["chunks"] == 0


# ---------------------------------------------------------------------------- text sets


SMOKE_SHA256 = "4dc845a9a76a61d6ab262f149bcc7a25f2e39d7b59b78b5f02cb6225fceb6f1d"


def test_smoke_text_set_is_pinned() -> None:
    ts = load_texts("smoke")
    assert ts.name == "tts-smoke" and ts.language == "en" and len(ts.texts) == 20
    categories = {t.category for t in ts.texts}
    assert {"numbers", "dates", "email", "url", "abbreviations", "question", "long"} <= categories
    assert sum(len(t.entities) for t in ts.texts) >= 15
    # changing the smoke texts changes every result: bump "version" and this hash together
    assert ts.sha256() == SMOKE_SHA256
    assert ts.dataset_id() == f"tts-smoke@sha256:{SMOKE_SHA256[:12]}"
    assert len(ts.limited(3).texts) == 3 and ts.limited(3).sha256() != ts.sha256()


def test_load_texts_from_files(tmp_path: Path) -> None:
    txt = tmp_path / "mine.txt"
    txt.write_text("# comment\nHello there.\n\nSecond line?\n", encoding="utf-8")
    ts = load_texts(txt)
    assert ts.name == "mine" and [t.text for t in ts.texts] == ["Hello there.", "Second line?"]
    assert [t.id for t in ts.texts] == ["t001", "t002"]
    js = tmp_path / "set.json"
    js.write_text(
        json.dumps({"name": "x", "texts": [{"id": "a", "text": "Call 911.", "entities": ["911"]}]}),
        encoding="utf-8",
    )
    assert load_texts(js).texts[0].entities == (("911",),)
    yml = tmp_path / "set.yaml"
    yml.write_text("- One.\n- {id: b, text: Two.}\n", encoding="utf-8")
    assert [t.id for t in load_texts(yml).texts] == ["t001", "b"]
    dup = tmp_path / "dup.yaml"
    dup.write_text("- {id: a, text: One.}\n- {id: a, text: Two.}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_texts(dup)
    with pytest.raises(ValueError, match="unknown text set"):
        load_texts("nope")


# ------------------------------------------------------------------------------ DNSMOS


def test_dnsmos_windows_and_calibration() -> None:
    assert dnsmos_windows(np.zeros(0, dtype=np.float32)) == []
    short = dnsmos_windows(np.ones(16_000, dtype=np.float32))  # tiled to >= 9.01 s
    assert len(short) == 7 and all(w.size == 144_160 for w in short)
    long = dnsmos_windows(np.ones(16_000 * 12, dtype=np.float32))
    assert len(long) == 3
    sig, bak, ovr = dnsmos_calibrate(3.0, 3.0, 3.0)
    assert sig == pytest.approx(-0.08397278 * 9 + 1.22083953 * 3 + 0.0052439)
    assert bak == pytest.approx(-0.13166888 * 9 + 1.60915514 * 3 - 0.39604546)
    assert ovr == pytest.approx(-0.06766283 * 9 + 1.11546468 * 3 + 0.04602535)


class _FakeSession:
    def __init__(self) -> None:
        self.calls = 0

    def get_inputs(self) -> list[Any]:
        return [type("I", (), {"name": "input_1"})()]

    def run(self, _outputs: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        (x,) = feeds.values()
        assert x.shape == (1, 144_160) and x.dtype == np.float32
        self.calls += 1
        return [np.array([[3.0, 3.5, 2.5]], dtype=np.float32)]


async def test_dnsmos_scores_with_an_injected_session() -> None:
    mos = DNSMOS()
    mos._session = _FakeSession()
    scores = await mos.score(tone(2.0, RATE))
    expected = dnsmos_calibrate(3.0, 3.5, 2.5)
    assert scores == pytest.approx(
        {"dnsmos_sig": expected[0], "dnsmos_bak": expected[1], "dnsmos_ovrl": expected[2]}
    )
    assert mos._session.calls == 7
    assert mos.describe()["license"] == "CC-BY-4.0"


def test_make_mos_predictor() -> None:
    assert make_mos_predictor(None) is None and make_mos_predictor("none") is None
    assert isinstance(make_mos_predictor("dnsmos"), DNSMOS)
    custom = make_mos_predictor("dnsmos:/tmp/model.onnx")
    assert isinstance(custom, DNSMOS) and custom.model_path is not None
    with pytest.raises(ValueError, match="unknown MOS"):
        make_mos_predictor("utmos9")


# ----------------------------------------------------------------------- end-to-end runs


class _PaddedTTS(TTS):
    """Emits ``lead`` s of silence, then a tone, in two chunks after ``delay`` s each."""

    provider = "padded"

    def __init__(self, *, lead: float = 0.2, delay: float = 0.02) -> None:
        super().__init__(model="padded-1", sample_rate=RATE)
        self.lead = lead
        self.delay = delay

    def _synthesize(self, text: str, *, voice: str | None) -> ChunkedStream:
        return _PaddedStream(self, text, voice=voice)


class _PaddedStream(ChunkedStream):
    async def _run(self) -> None:
        tts: _PaddedTTS = self._tts  # type: ignore[assignment]
        await asyncio.sleep(tts.delay)
        self._push_audio(silence(tts.lead, RATE))
        await asyncio.sleep(tts.delay)
        self._push_audio(tone(max(0.3, len(self.text) / 30), RATE))


class _HangingTTS(_PaddedTTS):
    def _synthesize(self, text: str, *, voice: str | None) -> ChunkedStream:
        return _HangingStream(self, text, voice=voice)


class _HangingStream(ChunkedStream):
    async def _run(self) -> None:
        await asyncio.sleep(3600)


TEXTS = TextSet(
    name="tiny",
    texts=(
        TTSText("a", "Your code is 427.", "numbers", (("427",),)),
        TTSText("b", "Is that right?", "question"),
    ),
)


async def test_track_run_writes_schema_and_scores_round_trip(tmp_path: Path) -> None:
    heard = iter(
        [
            "your code is four hundred twenty seven",
            "is that light",
            "your code is 421",
            "is that right",
        ]
    )
    stt = MockSTT(transcripts=lambda _audio: next(heard))
    results = await run_tts_benchmark(
        _PaddedTTS(lead=0.2),
        TEXTS,
        TTSOptions(words_per_second=0, warmup_requests=1),
        stt=stt,
        out_dir=tmp_path,
        run_id="run1",
    )
    s = results.summary
    assert s.track == "tts" and s.system == "padded/padded-1" and s.n == 2
    assert s.dataset == TEXTS.dataset_id()
    for mode in ("batch", "streaming"):
        ttfa = s.metrics[f"{mode}.ttfa_ms"]
        lead = s.metrics[f"{mode}.leading_silence_ms"]
        assert ttfa.n == 2
        assert s.counts[f"{mode}.items"] == 2 and s.counts[f"{mode}.errors"] == 0
    # batch keeps the provider's 200 ms of leading silence and TTFA includes it
    assert s.metrics["batch.leading_silence_ms"].p50 == pytest.approx(200, abs=3)
    ttfb, ttfa = s.metrics["batch.ttfb_ms"].p50, s.metrics["batch.ttfa_ms"].p50
    assert ttfa is not None and ttfb is not None and ttfa - ttfb == pytest.approx(200, abs=5)
    # streaming goes through SentenceStreamAdapter, which trims leading silence
    lead = s.metrics["streaming.leading_silence_ms"].p50
    assert lead is not None and lead < 50
    # batch: 1 error in 4 + 3 words ("light"); streaming: 1 ("421" vs "427")
    assert s.rates["batch.rt_wer"] == pytest.approx(1 / 7)
    assert s.rates["streaming.rt_wer"] == pytest.approx(1 / 7)
    assert s.rates["batch.hardtext_acc"] == 1.0 and s.rates["streaming.hardtext_acc"] == 0.0
    assert s.rates["batch.perfect_rate"] == 0.5
    assert s.extra["normalizer"] == "whisper-english" and s.extra["stt"] == "mock/mock-stt"
    assert set(s.extra["cold_start"]) == {"batch", "streaming"}

    run_dir = tmp_path / "run1"
    for name in (MANIFEST_FILE, ITEMS_FILE, SUMMARY_FILE, REPORT_FILE):
        assert (run_dir / name).is_file()
    loaded = load_run(run_dir)
    assert loaded.summary == s
    assert loaded.manifest.scenario["sha256"] == TEXTS.sha256()
    assert loaded.manifest.system["tts"]["provider"] == "padded"
    assert loaded.manifest.system["stt"]["provider"] == "mock"
    assert loaded.manifest.options["normalizer_resolved"] == "whisper-english"
    items = {it["id"]: it for it in loaded.items}
    assert set(items) == {"batch/a", "batch/b", "streaming/a", "streaming/b"}
    assert items["streaming/a"]["entities_missed"] == ["427"]
    assert items["batch/b"]["transcript"] == "is that light"
    assert (run_dir / items["batch/a"]["audio_file"]).is_file()
    report = render_tts_report(loaded)
    assert "TTS (T3)" in report and "Speed and quality" in report and "is that light" in report


async def test_streaming_ttfa_follows_the_text_pace() -> None:
    texts = TextSet("one", (TTSText("s", "one two three four five six seven."),))
    results = await run_tts_benchmark(
        MockTTS(), texts, TTSOptions(modes=("streaming",), words_per_second=20,
                                     warmup_requests=0, save_audio=False),
    )  # fmt: skip
    ttfa = results.summary.metrics["streaming.ttfa_ms"].p50
    # the sentence is complete after the 7th word, pushed 6 / 20 s = 300 ms in
    assert ttfa is not None and 280 < ttfa < 1000
    assert results.summary.rates["streaming.rt_wer"] is None  # no STT
    assert any("round-trip" in n for n in results.manifest.notes)


async def test_timeouts_and_mos_failures_are_recorded_not_raised() -> None:
    class Broken:
        name = "broken"

        async def load(self) -> None:
            raise RuntimeError("no model here")

        async def score(self, audio: AudioFrame) -> dict[str, float]:
            raise AssertionError

        def describe(self) -> dict[str, Any]:
            return {}

    results = await run_tts_benchmark(
        _HangingTTS(), TEXTS.limited(1),
        TTSOptions(modes=("batch",), timeout=0.2, warmup_requests=0, save_audio=False),
        mos=Broken(),  # type: ignore[arg-type]
    )  # fmt: skip
    (item,) = results.items
    assert item["error"] == "timeout after 0.2 s"
    assert results.summary.counts["batch.errors"] == 1
    assert results.summary.rates["batch.error_rate"] == 1.0
    assert any("MOS predictor 'broken' unavailable" in n for n in results.manifest.notes)


async def test_mos_scores_become_metrics() -> None:
    class Constant:
        name = "const"

        async def load(self) -> None:
            pass

        async def score(self, audio: AudioFrame) -> dict[str, float]:
            return {"dnsmos_ovrl": 3.25}

        def describe(self) -> dict[str, Any]:
            return {"name": "const"}

    results = await run_tts_benchmark(
        MockTTS(), TEXTS, TTSOptions(modes=("batch",), warmup_requests=0, save_audio=False),
        mos=Constant(),  # type: ignore[arg-type]
    )  # fmt: skip
    assert results.summary.metrics["batch.dnsmos_ovrl"].mean == 3.25
    assert results.summary.extra["modes"]["batch"]["mos_mean"] == {"dnsmos_ovrl": 3.25}
    assert results.manifest.options["mos"] == {"name": "const"}


def test_options_validation() -> None:
    with pytest.raises(ValueError, match="modes"):
        TTSOptions(modes=("live",)).validate()
    with pytest.raises(ValueError, match="repeats"):
        TTSOptions(repeats=0).validate()
    with pytest.raises(ValueError, match="normalizer"):
        TTSOptions(normalizer="x").validate()


# ----------------------------------------------------------------------------------- CLI


def test_cli_tts_command(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["bench", "tts", "--tts", "{provider: mock, ttfb: 0.01}", "--stt", "mock",
         "--limit", "2", "--mode", "batch", "--out", str(tmp_path), "--run-id", "cli",
         "--no-audio", "--json"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["track"] == "tts" and summary["metrics"]["batch.ttfa_ms"]["n"] == 2
    assert not (tmp_path / "cli" / "artifacts").exists()
    rerender = CliRunner().invoke(app, ["bench", "report", str(tmp_path / "cli"), "--no-write"])
    assert rerender.exit_code == 0 and "TTS (T3)" in rerender.stdout


def test_cli_tts_rejects_bad_options() -> None:
    result = CliRunner().invoke(app, ["bench", "tts", "--tts", "mock", "--mode", "live"])
    assert result.exit_code == 2
    result = CliRunner().invoke(app, ["bench", "tts", "--tts", "mock", "--texts", "nope"])
    assert result.exit_code == 2


# ------------------------------------------------------------------------ real DNSMOS


@pytest.mark.model
async def test_real_dnsmos_scores_speechlike_audio() -> None:
    pytest.importorskip("onnxruntime")
    mos = DNSMOS()
    await mos.load()
    scores = await mos.score(tone(3.0, 16_000))
    assert set(scores) == {"dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl"}
    assert all(0.5 < v < 5.5 for v in scores.values())
