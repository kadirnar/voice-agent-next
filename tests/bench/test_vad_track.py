"""T4 VAD track: corpus synthesis and frame / segment metrics."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from voice_agent_next.bench.results import load_run
from voice_agent_next.bench.tracks.vad import (
    VadStreamOutput,
    evaluate_clip,
    render_vad_report,
    run_vad_benchmark,
    vad_markdown_table,
)
from voice_agent_next.bench.vad_corpus import (
    FRAME,
    VadCondition,
    build_vad_corpus,
    make_noise,
    parse_condition,
    speech_labels,
)
from voice_agent_next.providers.mock import synth_speech

from .helpers import concat, silence


def sources() -> list[tuple[str, object]]:
    return [
        ("a", concat(silence(0.2, 16000), synth_speech(1.0, 16000), silence(0.3, 16000))),
        ("b", concat(synth_speech(0.6, 16000), silence(0.4, 16000), synth_speech(0.5, 16000))),
        ("c", synth_speech(0.8, 16000, frequency=180.0)),
    ]


def small_corpus(**kw: object):  # type: ignore[no-untyped-def]
    return build_vad_corpus(
        sources(),  # type: ignore[arg-type]
        conditions=kw.pop("conditions", (VadCondition("clean"), VadCondition("pink", 10.0))),  # type: ignore[arg-type]
        lead=1.0, tail=3.0, **kw,  # type: ignore[arg-type]
    )  # fmt: skip


def test_speech_labels_follow_the_clean_signal() -> None:
    audio = concat(silence(0.2, 16000), synth_speech(0.5, 16000), silence(0.1, 16000),
                   synth_speech(0.5, 16000), silence(0.4, 16000), synth_speech(0.3, 16000))  # fmt: skip
    lab = speech_labels(audio)
    assert not lab[:20].any() and lab[20:70].all()
    assert lab[70:80].all()  # 100 ms dip bridged
    assert not lab[130:170].any()  # 400 ms pause kept
    assert lab[170:200].all()


def test_corpus_is_deterministic_and_aligned() -> None:
    a, b = small_corpus(), small_corpus()
    assert a.sha256 == b.sha256
    assert small_corpus(seed=1).sha256 != a.sha256
    clean, noisy = a.clips
    assert clean.condition.name == "clean" and noisy.condition.name == "pink@10dB"
    assert len(clean.labels) == round(clean.duration / FRAME)
    assert clean.utterances == noisy.utterances and (clean.labels == noisy.labels).all()
    # utterance "a": 0.2 s leading silence inside the source, placed after 1 s of lead-in
    u = clean.utterances[0]
    assert u.placed_at == pytest.approx(1.0) and u.start == pytest.approx(1.2)
    assert u.end == pytest.approx(2.2)
    # the gap between utterances comes from the seeded layout: 0.8 - 2.5 s + source padding
    assert 0.8 <= clean.utterances[1].placed_at - (u.placed_at + 1.5) <= 2.5 + 1e-9
    # speech level -20 dBFS; pink noise 10 dB below it
    x = clean.audio.to_float32()
    lab = np.repeat(clean.labels, 160)[: len(x)]
    rms = 20 * np.log10(np.sqrt(np.mean(x[lab] ** 2)))
    assert rms == pytest.approx(-20.0, abs=0.5)
    noise = noisy.audio.to_float32()[~lab]
    assert 20 * np.log10(np.sqrt(np.mean(noise**2))) == pytest.approx(-30.0, abs=0.5)
    desc = a.describe()
    assert desc["sha256"] == a.sha256 and len(desc["clips"]) == 2


def test_noise_and_conditions() -> None:
    rng = np.random.default_rng(0)
    for kind in ("white", "pink", "brown"):
        x = make_noise(kind, 16000, rng)
        assert np.sqrt(np.mean(x**2)) == pytest.approx(1.0)
    assert parse_condition("pink@10dB") == VadCondition("pink", 10.0)
    assert parse_condition("white:5").name == "white@5dB"
    assert parse_condition("transient").snr_db is None
    for bad in ("pink", "clean@5", "rain@3"):
        with pytest.raises(ValueError):
            parse_condition(bad)


def test_evaluate_clip_frame_and_segment_metrics() -> None:
    clip = small_corpus(conditions=(VadCondition("clean"),)).clips[0]
    n = len(clip.labels)
    # an oracle that is 50 ms late at every onset and 300 ms late at every end
    starts = [u.start + 0.05 for u in clip.utterances] + [0.3]  # + one false alarm
    ends = [u.end + 0.3 for u in clip.utterances] + [0.4]
    out = VadStreamOutput(clip.labels.astype(float), sorted(starts), sorted(ends), 0.01, 0.02)
    r = evaluate_clip("oracle", clip, out, 0.5)
    assert r.f1 == 1.0 and r.roc_auc == 1.0 and r.false_alarm_rate == 0.0
    assert r.onset_lag_ms == pytest.approx([50.0] * 3)
    assert r.offset_lag_ms == pytest.approx([300.0] * 3)
    assert r.missed_utterances == 0 and r.false_alarms == 1
    assert r.false_alarms_per_min is not None and r.false_alarms_per_min > 0
    assert r.rtf == pytest.approx(0.01 / clip.duration, rel=1e-3)
    # a deaf VAD misses everything
    deaf = evaluate_clip("deaf", clip, VadStreamOutput(np.zeros(n), [], [], 0.0, 0.0), 0.5)
    assert deaf.recall == 0.0 and deaf.missed_utterances == 3 and deaf.onset_lag_ms == []


def test_vad_run_with_energy(tmp_path) -> None:  # type: ignore[no-untyped-def]
    corpus = small_corpus()
    results = asyncio.run(
        run_vad_benchmark(["energy"], corpus, dataset="synthetic", out_dir=tmp_path)
    )
    s = results.summary
    assert s.counts["vads"] == 1 and s.counts["conditions"] == 2
    row = s.extra["table"][0]
    assert row["condition"] == "clean" and row["f1"] > 0.9 and row["missed_utterances"] == 0
    assert s.metrics["energy.onset_lag_ms"].n >= 3
    assert "energy" in vad_markdown_table(results)
    assert s.dataset.startswith("synthetic@sha256:")
    assert "VAD (T4)" in render_vad_report(load_run(results.directory))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        asyncio.run(run_vad_benchmark(["energy", "energy"], corpus))
