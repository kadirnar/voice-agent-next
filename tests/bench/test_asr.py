"""T2 ASR track: normalization, error-rate math, datasets and the track end to end."""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
import tarfile
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from voice_agent_next.audio import AudioFrame
from voice_agent_next.audio.wav import write_wav
from voice_agent_next.bench import asr_datasets
from voice_agent_next.bench.asr_datasets import (
    AsrDataset,
    builtin_datasets,
    fetch_archive_members,
    load_asr_dataset,
    load_builtin_dataset,
    load_manifest,
)
from voice_agent_next.bench.results import SUMMARY_FILE, load_run
from voice_agent_next.bench.text_norm import (
    base_language,
    cer_text,
    get_normalizer,
    normalizer_for,
    uses_cer,
)
from voice_agent_next.bench.tracks.asr import (
    AsrItem,
    AsrOptions,
    _revisions,
    asr_markdown_table,
    render_asr_report,
    run_asr_benchmark,
)
from voice_agent_next.bench.wer import EditCounts, corpus_rate, edit_counts, word_counts
from voice_agent_next.cli.main import app
from voice_agent_next.providers.mock import MockSTT
from voice_agent_next.utils.download import DownloadError

from .helpers import concat, silence, tone

runner = CliRunner()
ANSI = re.compile(r"\x1b\[[0-9;]*m")


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VAN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("VAN_OFFLINE", raising=False)


# ------------------------------------------------------------------ normalization


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # values checked against openai-whisper's EnglishTextNormalizer (release 20250625)
        (
            "Mr. Smith paid $20 million in 1960s, twenty-one point five percent!",
            "mister smith paid $20000000 in 1960s 21.5%",
        ),
        (
            "It's the colour of the harbour; I won't go. Let's see: one oh one, double seven.",
            "it is the color of the harbor i will not go let us see 10177",
        ),
        (
            "He was born on the twenty third of March, nineteen ninety-nine at 3:45 p.m.",
            "he was born on the 23rd of march 1999 at 3 45 p m",
        ),
        ("£5 and 50 cents [noise] (laughs) um uh hmm the Dr. Who", "£5.50 the doctor who"),
        ("THE ENGLISH FORWARDED TO THE FRENCH", "the english forwarded to the french"),
        ("Café naïve résumé", "cafe naive resume"),
        ("", ""),
    ],
)
def test_whisper_english_normalizer(text: str, expected: str) -> None:
    assert get_normalizer("whisper-english")(text) == expected


def test_basic_normalizer_keeps_letters_and_marks() -> None:
    basic = get_normalizer("whisper-basic")
    assert basic("¿Dónde está, señor?") == "dónde está señor"
    assert basic("Merhaba, İstanbul'a (gürültü) gidiyorum!") == "merhaba i̇stanbul a gidiyorum"
    # Devanagari vowel signs are combining marks: kept (Whisper's basic normalizer would
    # turn them into spaces and split every word)
    assert basic("नमस्ते दुनिया।") == "नमस्ते दुनिया"
    assert get_normalizer("none")("  A,  b ") == "A, b"


def test_language_helpers() -> None:
    assert base_language("en-US") == base_language("en_us") == "en"
    assert base_language("cmn_hans_cn") == "zh"
    assert base_language(None) is None
    assert uses_cer("zh") and uses_cer("ja-JP") and uses_cer("th") and not uses_cer("tr")
    assert normalizer_for("en") == "whisper-english"
    assert normalizer_for(None) == "whisper-english"
    assert normalizer_for("de") == "whisper-basic"
    assert normalizer_for("de", "none") == "none"
    assert cer_text("这 是 测试", "zh") == "这是测试"
    assert cer_text("a  b", "en") == "a b"
    with pytest.raises(ValueError, match="unknown normalizer"):
        normalizer_for("en", "fancy")


# ---------------------------------------------------------------------- WER math


def test_edit_counts_classifies_errors() -> None:
    c = word_counts("the cat sat on the mat", "the cat sit on mat")
    assert (c.hits, c.substitutions, c.deletions, c.insertions) == (4, 1, 1, 0)
    assert c.ref_len == 6 and c.errors == 2
    assert c.rate == pytest.approx(1 / 3)
    c = word_counts("turn the lights on", "turn the the lights on now")
    assert (c.hits, c.substitutions, c.deletions, c.insertions) == (4, 0, 0, 2)
    assert word_counts("", "") == EditCounts()
    assert word_counts("", "extra words").insertions == 2
    assert word_counts("", "x").rate is None
    assert word_counts("a b", "").deletions == 2
    assert edit_counts("kitten", "sitting").errors == 3  # classic Levenshtein example


def test_corpus_rate_weights_by_reference_length() -> None:
    short = word_counts("hello", "yellow")  # 1/1
    long = word_counts("a b c d e f g h i j", "a b c d e f g h i j")  # 0/10
    assert corpus_rate([short, long]) == pytest.approx(1 / 11)
    assert corpus_rate([]) is None
    assert (short + long).as_dict()["ref_len"] == 11


def test_edit_totals_match_jiwer() -> None:
    jiwer = pytest.importorskip("jiwer")
    rng = random.Random(7)
    vocab = ["a", "b", "c", "d", "e"]
    for _ in range(200):
        ref = " ".join(rng.choice(vocab) for _ in range(rng.randint(1, 12)))
        hyp = " ".join(rng.choice(vocab) for _ in range(rng.randint(0, 12)))
        out = jiwer.process_words(ref, hyp) if hyp else None
        mine = word_counts(ref, hyp)
        if out is None:
            assert mine.errors == len(ref.split())
            continue
        assert mine.errors == out.substitutions + out.deletions + out.insertions
        assert mine.rate == pytest.approx(out.wer)


def test_interim_revisions() -> None:
    norm = get_normalizer("whisper-english")
    seg = ["s1"] * 4
    # growing interims (the last word may still change) are stable
    assert _revisions(["hel", "hello wor", "hello world", "hello world again"], seg, norm) == 0
    # "hello world" -> "yellow world" rewrites an earlier word
    assert _revisions(["hello world", "yellow world now"], ["s1", "s1"], norm) == 1
    # a new segment starts afresh
    assert _revisions(["hello world", "other"], ["s1", "s2"], norm) == 0


# ------------------------------------------------------------------------ datasets


def test_builtin_catalog_is_pinned() -> None:
    catalog = builtin_datasets()
    assert len(catalog["librispeech-test-clean-smoke"]["items"]) == 50
    for lang in ("en", "es", "de", "tr", "zh"):
        spec = catalog[f"fleurs-{lang}-smoke"]
        assert spec["language"] == lang and len(spec["items"]) == 10
        assert spec["archive"]["revision"] in spec["archive"]["urls"][0]  # pinned HF commit
    for name, spec in catalog.items():
        items = spec["items"]
        assert spec["license"] == "CC-BY-4.0" and spec["attribution"], name
        assert len({i["id"] for i in items}) == len(items), name
        assert len({i["file"] for i in items}) == len(items), name
        for item in items:
            assert re.fullmatch(r"[0-9a-f]{64}", item["sha256"]), (name, item["id"])
            assert item["text"].strip() and 1.0 <= item["duration"] <= 20.0
            assert "/" not in item["file"] and ".." not in item["file"]
        assert spec["archive"]["streamed_mb"] <= 300


def _wav_bytes(duration: float) -> bytes:
    buf = io.BytesIO()
    write_wav(buf, concat(silence(0.1, 16_000), tone(duration, 16_000)))
    return buf.getvalue()


def _tar_gz(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _serve(routes: dict[str, bytes]) -> tuple[httpx.Client, list[str]]:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        body = routes.get(str(request.url))
        return httpx.Response(404) if body is None else httpx.Response(200, content=body)

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def _fake_catalog(monkeypatch: pytest.MonkeyPatch, files: dict[str, bytes]) -> dict[str, Any]:
    items = [
        {
            "id": f"utt{i}",
            "member": name,
            "file": Path(name).name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "duration": 0.5,
            "text": f"reference {i}",
        }
        for i, (name, data) in enumerate((n, d) for n, d in files.items() if n.endswith(".wav"))
    ]
    spec = {
        "language": "en",
        "license": "CC-BY-4.0",
        "attribution": "test",
        "archive": {
            "urls": ["https://mirror-a.test/set.tar.gz", "https://mirror-b.test/set.tar.gz"]
        },
        "items": items,
    }
    monkeypatch.setattr(asr_datasets, "builtin_datasets", lambda: {"tiny-smoke": spec})
    return spec


def test_builtin_dataset_is_streamed_verified_and_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {
        "data/README": b"not audio",
        "data/a.wav": _wav_bytes(0.3),
        "data/b.wav": _wav_bytes(0.4),
        "data/unused.wav": _wav_bytes(0.2),
    }
    spec = _fake_catalog(monkeypatch, files)
    spec["items"] = spec["items"][:2]  # a.wav and b.wav
    # the first mirror is down: the second one is used
    client, calls = _serve({"https://mirror-b.test/set.tar.gz": _tar_gz(files)})
    ds = load_builtin_dataset("tiny-smoke", client=client)
    assert [u.id for u in ds.utterances] == ["utt0", "utt1"]
    assert all(u.audio.exists() for u in ds.utterances)
    assert not (ds.utterances[0].audio.parent / "unused.wav").exists()
    assert ds.language == "en" and ds.builtin and ds.id.startswith("tiny-smoke@sha256:")
    assert len(calls) == 2
    again = load_asr_dataset("tiny-smoke", client=client)
    assert len(calls) == 2  # cache hit: no network
    assert again.sha256 == ds.sha256
    desc = ds.describe()
    assert desc["n"] == 2 and desc["items"][0]["sha256"] == spec["items"][0]["sha256"]


def test_builtin_dataset_rejects_bad_checksums_and_missing_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {"a.wav": _wav_bytes(0.3)}
    spec = _fake_catalog(monkeypatch, files)
    tampered = _tar_gz({"a.wav": _wav_bytes(0.35)})
    client, _ = _serve(dict.fromkeys(spec["archive"]["urls"], tampered))
    with pytest.raises(DownloadError, match="checksum mismatch"):
        load_builtin_dataset("tiny-smoke", client=client)
    empty = _tar_gz({"other.wav": b"x"})
    client, _ = _serve(dict.fromkeys(spec["archive"]["urls"], empty))
    with pytest.raises(DownloadError, match="not found"):
        load_builtin_dataset("tiny-smoke", client=client)
    with pytest.raises(ValueError, match="unknown dataset"):
        load_builtin_dataset("nope")


def test_offline_mode_refuses_to_download(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_catalog(monkeypatch, {"a.wav": _wav_bytes(0.3)})
    monkeypatch.setenv("VAN_OFFLINE", "1")
    client, calls = _serve({})
    with pytest.raises(DownloadError, match="VAN_OFFLINE"):
        load_builtin_dataset("tiny-smoke", client=client)
    assert calls == []


def test_fetch_stops_after_the_last_wanted_member(tmp_path: Path) -> None:
    good = _wav_bytes(0.1)
    archive = _tar_gz({"a.wav": good, "zzz.bin": random.Random(0).randbytes(2_000_000)})
    client, _ = _serve({"https://x.test/a.tar.gz": archive})
    fetched = fetch_archive_members(
        ["https://x.test/a.tar.gz"],
        {"a.wav": ("a.wav", hashlib.sha256(good).hexdigest())},
        tmp_path,
        client=client,
    )
    assert (tmp_path / "a.wav").read_bytes() == good
    assert fetched < len(archive)


def _write_manifest(tmp_path: Path, texts: list[str], durations: list[float]) -> Path:
    audio = tmp_path / "audio"
    audio.mkdir()
    lines = []
    for i, (text, duration) in enumerate(zip(texts, durations, strict=True)):
        write_wav(audio / f"u{i}.wav", concat(silence(0.1, 16_000), tone(duration, 16_000)))
        lines.append(json.dumps({"audio": f"audio/u{i}.wav", "text": text, "id": f"u{i}"}))
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_manifest_formats(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, ["hello world", "good morning"], [0.3, 0.4])
    ds = load_manifest(path, language="en")
    assert [u.id for u in ds.utterances] == ["u0", "u1"]
    assert ds.language == "en" and not ds.builtin
    assert all(u.sha256 and u.audio.is_file() for u in ds.utterances)
    tsv = tmp_path / "set.tsv"
    tsv.write_text("audio_filepath\ttranscript\n" + "audio/u0.wav\thello world\n", "utf-8")
    other = load_manifest(tsv)
    assert other.utterances[0].text == "hello world" and other.utterances[0].id == "u0"
    assert other.sha256 != ds.sha256
    changed = tmp_path / "changed.jsonl"
    changed.write_text(path.read_text("utf-8").replace("hello world", "hello"), "utf-8")
    assert load_manifest(changed, language="en").sha256 != ds.sha256
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"audio": "audio/missing.wav", "text": "x"}\n', "utf-8")
    with pytest.raises(ValueError, match="not found"):
        load_manifest(bad)
    with pytest.raises(ValueError, match="unknown dataset"):
        load_asr_dataset("no-such-dataset")


# --------------------------------------------------------------------- the track

_TEXTS = ["hello world", "good morning everyone", "the quick brown fox"]
_DURATIONS = [0.3, 0.7, 1.1]


def _stt_by_duration(hyps: list[str], **kwargs: Any) -> MockSTT:
    """MockSTT whose transcript depends on the (unique) utterance duration."""

    def transcript(audio: AudioFrame) -> str:
        i = min(range(len(_DURATIONS)), key=lambda k: abs(_DURATIONS[k] - audio.duration))
        return hyps[i]

    return MockSTT(transcripts=transcript, speech_threshold=0.001, **kwargs)


def _dataset(tmp_path: Path) -> AsrDataset:
    return load_manifest(_write_manifest(tmp_path, _TEXTS, _DURATIONS), language="en")


async def test_batch_track_end_to_end(tmp_path: Path) -> None:
    hyps = ["Hello, world!", "good morning every one", "the quick brown fox"]
    stt = _stt_by_duration(hyps, latency=0.01)
    dataset = _dataset(tmp_path)
    ds_hash = dataset.sha256
    results = await run_asr_benchmark(
        stt, dataset, AsrOptions(mode="batch", warmup=True),
        out_dir=tmp_path / "out", run_id="asr-run", label="mock-stt",
    )  # fmt: skip
    s = results.summary
    assert s.track == "asr" and s.system == "mock-stt" and s.n == 3
    # "everyone" -> "every one": 1 substitution + 1 insertion over 9 reference words
    assert s.rates["wer"] == pytest.approx(2 / 9, abs=1e-6)
    assert s.counts["word_errors"] == 2 and s.counts["words"] == 9
    assert s.rates["perfect_rate"] == pytest.approx(2 / 3, abs=1e-6)
    items = [AsrItem.model_validate(it) for it in results.items]
    assert items[0].hypothesis_norm == "hello world" and items[0].wer == 0
    assert all(it.ttfs_ms is not None and it.ttfs_ms >= 5 for it in items)
    assert s.extra["rtfx"] > 1 and s.metrics["ttfs_ms"].n == 3
    ds = s.extra["datasets"]["manifest"]
    assert ds["metric"] == "wer" and ds["wer"] == pytest.approx(2 / 9, abs=1e-6)
    assert s.extra["warmup_ms"] is not None
    # results round-trip through the run directory; the report re-renders identically
    loaded = load_run(tmp_path / "out" / "asr-run")
    assert loaded.manifest.track == "asr"
    assert loaded.manifest.scenario["datasets"][0]["sha256"] == ds_hash
    assert loaded.manifest.options["normalizers"] == {"manifest": "whisper-english"}
    assert render_asr_report(loaded) == results.report
    assert "**22.22%**" in asr_markdown_table(loaded)
    assert "every one" in (results.report or "")


async def test_streaming_track_measures_latency_and_partials(tmp_path: Path) -> None:
    stt = _stt_by_duration(list(_TEXTS), latency=0.05, interim_results=True)
    results = await run_asr_benchmark(
        stt, _dataset(tmp_path),
        AsrOptions(mode="streaming", chunk_ms=20, realtime_factor=4.0, warmup=False),
    )  # fmt: skip
    s = results.summary
    assert s.rates["wer"] == 0 and s.counts["timeouts"] == 0
    items = [AsrItem.model_validate(it) for it in results.items]
    for it in items:
        assert it.finals == 1 and it.ttfs_ms is not None
        assert 40 <= it.ttfs_ms <= 1000  # the mock's 50 ms finalization delay
    # MockSTT emits interims after 0.5 s of speech: only the longer utterances have them
    assert items[0].first_partial_ms is None
    assert items[2].first_partial_ms is not None
    # 4x real time: the 1st partial comes ~0.6 s of audio (0.15 s wall) after the start
    assert 50 <= items[2].first_partial_ms <= 2000
    assert s.rates["interim_revision_rate"] == 0
    assert results.manifest.options["mode"] == "streaming"


async def test_streaming_needs_a_vad_for_batch_only_recognizers(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    batch_only = _stt_by_duration(list(_TEXTS), streaming=False)
    with pytest.raises(ValueError, match="does not stream"):
        await run_asr_benchmark(batch_only, ds, AsrOptions(mode="streaming", warmup=False))
    results = await run_asr_benchmark(
        batch_only, ds, AsrOptions(mode="streaming", realtime_factor=0, warmup=False),
        vad="energy",
    )  # fmt: skip
    assert results.manifest.system["stt"]["stream_adapter"] is True
    assert results.summary.rates["wer"] == 0


async def test_failures_are_reported_not_scored(tmp_path: Path) -> None:
    class Flaky(MockSTT):
        async def _recognize(self, audio: AudioFrame, *, language: str | None) -> Any:
            if audio.duration > 1.0:
                raise RuntimeError("boom")
            return await super()._recognize(audio, language=language)

    stt = Flaky(transcripts=lambda a: "hello world" if a.duration < 0.5 else "x")
    results = await run_asr_benchmark(stt, _dataset(tmp_path), AsrOptions(warmup=False))
    s = results.summary
    assert s.counts["errors"] == 1 and s.n == 2
    assert any("failed" in note for note in results.manifest.notes)


def test_options_validation() -> None:
    with pytest.raises(ValueError, match="mode"):
        AsrOptions(mode="live").validate()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="realtime_factor"):
        AsrOptions(realtime_factor=-1).validate()
    with pytest.raises(ValueError, match="normalizer"):
        AsrOptions(normalizer="x").validate()


def test_cli_asr_runs_and_report_rerenders(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, _TEXTS, _DURATIONS)
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "bench", "asr", "--stt", "{provider: mock, default_text: hello world}",
            "-d", str(manifest), "--language", "en", "--no-warmup",
            "--out", str(out), "--run-id", "cli-asr", "--json",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    summary = json.loads((out / "cli-asr" / SUMMARY_FILE).read_text(encoding="utf-8"))
    assert summary["track"] == "asr" and summary["n"] == 3
    assert summary["rates"]["perfect_rate"] == pytest.approx(1 / 3, abs=1e-6)
    rerender = runner.invoke(app, ["bench", "report", str(out / "cli-asr")])
    assert rerender.exit_code == 0, rerender.output
    assert "ASR (T2)" in rerender.output
    md = runner.invoke(
        app,
        ["bench", "asr", "--stt", "mock", "-d", str(manifest), "--no-warmup", "--markdown",
         "--mode", "streaming", "--realtime-factor", "0", "--out", str(out)],
    )  # fmt: skip
    assert md.exit_code == 0, md.output
    assert "| system | dataset | mode |" in md.output


def test_cli_asr_rejects_bad_input(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, _TEXTS, _DURATIONS)
    bad_mode = runner.invoke(app, ["bench", "asr", "--stt", "mock", "-d", str(manifest),
                                   "--mode", "live"])  # fmt: skip
    assert bad_mode.exit_code == 2
    assert "mode" in ANSI.sub("", bad_mode.output)
    no_vad = runner.invoke(
        app,
        ["bench", "asr", "--stt", "{provider: mock, streaming: false}", "-d", str(manifest),
         "--mode", "streaming", "--no-warmup", "--out", str(tmp_path / "o")],
    )  # fmt: skip
    assert no_vad.exit_code == 2
    assert "does not stream" in ANSI.sub("", no_vad.output)
    missing = runner.invoke(app, ["bench", "asr", "--stt", "mock", "-d", "nope"])
    assert missing.exit_code == 2


def test_load_audio_reads_float_and_flac_files(tmp_path: Path) -> None:
    sf = pytest.importorskip("soundfile")
    import numpy as np

    from voice_agent_next.bench.asr_datasets import load_audio

    t = np.arange(8000) / 16_000
    wave = (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    for name, subtype in (("f.wav", "FLOAT"), ("f.flac", "PCM_16")):
        sf.write(str(tmp_path / name), wave, 16_000, subtype=subtype)
        audio = load_audio(tmp_path / name)
        assert audio.sample_rate == 16_000 and audio.channels == 1
        assert audio.duration == pytest.approx(0.5)
        assert audio.rms() == pytest.approx(0.25 / np.sqrt(2), rel=0.02)
