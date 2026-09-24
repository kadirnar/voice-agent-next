"""T4 end-of-turn track: policy metrics, datasets and the offline run."""

from __future__ import annotations

import asyncio
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from voice_agent_next.audio import AudioFrame
from voice_agent_next.audio.wav import write_wav
from voice_agent_next.bench import eot_datasets
from voice_agent_next.bench.eot_datasets import (
    EOT_BENCH_FILES,
    builtin_eot_datasets,
    load_eot_dataset,
    load_eot_manifest,
)
from voice_agent_next.bench.eot_metrics import (
    PolicyPoint,
    ScoredSpan,
    classification_metrics,
    evaluate_policy,
    filter_spans,
    min_cutoff_under_latency,
    min_latency_under_cutoff,
    pareto_front,
    roc_auc,
    sweep_policies,
)
from voice_agent_next.bench.results import load_run
from voice_agent_next.bench.tracks.turns import (
    TurnsOptions,
    render_turns_report,
    run_turns_benchmark,
    turns_markdown_table,
)
from voice_agent_next.chat import ChatContext
from voice_agent_next.turn import TurnDetector

from .helpers import concat, silence, tone

# ----------------------------------------------------------------------- metrics


def spans() -> list[ScoredSpan]:
    return [
        ScoredSpan("hold", 0.5, 0.1, 0.2),
        ScoredSpan("hold", 1.5, 0.7, 0.2),  # a confident but wrong score
        ScoredSpan("hold", 0.15, None, 0.2),  # ended before the score point
        ScoredSpan("eot", 3.0, 0.9, 0.2),
        ScoredSpan("eot", 3.0, 0.4, 0.2),
    ]


def test_policy_counts_cutoffs_and_latency_like_eot_bench() -> None:
    s = filter_spans(spans())
    assert [x.duration for x in s] == [0.5, 1.5, 3.0, 3.0]  # 0.15 s hold left out
    (p,) = sweep_policies(
        s, thresholds=[0.5], action_delays=[0.4], timeouts=[2.0], include_vad=False
    )
    # hold 1.5 s: fires at max(0.4, 0.2) = 0.4 < 1.5 -> cut off; hold 0.5 s: p=0.1 no fire
    assert p.cutoff_rate == pytest.approx(0.5)
    # eot: 0.9 fires at 0.4 s; 0.4 does not -> timeout 2.0 s
    assert p.mean_latency == pytest.approx((0.4 + 2.0) / 2)
    assert p.timeout_rate == pytest.approx(0.5)
    # the timeout also cuts off pauses longer than it
    (q,) = sweep_policies(
        s, thresholds=[0.95], action_delays=[0.4], timeouts=[1.0], include_vad=False
    )
    assert q.cutoff_rate == pytest.approx(0.5) and q.mean_latency == pytest.approx(1.0)


def test_vad_baseline_and_operating_points() -> None:
    s = filter_spans(spans())
    points = sweep_policies(s, thresholds=[0.0, 0.5, 0.95], action_delays=[0.2, 0.4],
                            timeouts=[1.0, 2.0])  # fmt: skip
    vad = [p for p in points if p.policy == "vad"]
    assert [p.action_delay for p in vad] == [0.2, 0.4, 1.0, 2.0]
    by_delay = {p.action_delay: p for p in vad}
    assert by_delay[0.4].cutoff_rate == pytest.approx(1.0)  # both holds > 0.4 s
    assert by_delay[1.0].cutoff_rate == pytest.approx(0.5)
    assert by_delay[2.0].cutoff_rate == 0.0 and by_delay[2.0].mean_latency == 2.0
    best = min_latency_under_cutoff(points, 0.0, "vad")
    assert best is not None and best.mean_latency == 2.0
    assert min_cutoff_under_latency(points, 0.1, "vad") is None  # no delay that short
    model = min_cutoff_under_latency(points, 1.2)
    assert model is not None and model.mean_latency <= 1.2
    front = pareto_front(points)
    assert front and all(a.cutoff_rate <= b.cutoff_rate for a, b in itertools.pairwise(front))
    assert all(a.mean_latency > b.mean_latency for a, b in itertools.pairwise(front))


def test_configured_policy_is_inclusive() -> None:
    s = [ScoredSpan("hold", 1.0, 0.5, 0.2), ScoredSpan("eot", 2.0, 0.5, 0.2)]
    inclusive = evaluate_policy(s, threshold=0.5, action_delay=0.4, timeout=2.5)
    assert inclusive.cutoff_rate == 1.0 and inclusive.mean_latency == pytest.approx(0.4)
    strict = evaluate_policy(s, threshold=0.5, action_delay=0.4, timeout=2.5, inclusive=False)
    assert strict.cutoff_rate == 0.0 and strict.mean_latency == pytest.approx(2.5)
    assert isinstance(strict, PolicyPoint) and strict.threshold == 0.5


def test_roc_auc_and_classification() -> None:
    assert roc_auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert roc_auc([0.1], [0.9]) == 0.0
    assert roc_auc([0.5], [0.5]) == 0.5
    assert roc_auc([], [0.1]) is None
    m = classification_metrics(filter_spans(spans()), 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["accuracy"] == 0.5 and m["f1"] == pytest.approx(0.5)
    assert m["roc_auc"] == pytest.approx(0.75)


def test_sweep_needs_both_labels() -> None:
    with pytest.raises(ValueError):
        sweep_policies([ScoredSpan("eot", 1.0, 0.9, 0.2)])


# ---------------------------------------------------------------------- datasets


def make_manifest(tmp_path: Path, turns: int = 4) -> Path:
    """Synthetic turns: speech 0.6 s, pause 0.5 s, speech 0.6 s, final silence 1.5 s."""
    rows = []
    for i in range(turns):
        audio = concat(tone(0.6, 16000), silence(0.5, 16000), tone(0.6, 16000),
                       silence(1.5, 16000))  # fmt: skip
        write_wav(tmp_path / f"t{i}.wav", audio)
        rows.append(
            {"id": f"t{i}", "audio": f"t{i}.wav", "language": "en",
             "silence_spans": [{"start": 0.6, "end": 1.1}, {"start": 1.7, "end": 3.2}],
             "words": [{"start": 0.0, "end": 0.6, "word": "hello"},
                       {"start": 1.1, "end": 1.7, "word": "there."}],
             "messages": [{"role": "assistant", "content": "How can I help?"}]}
        )  # fmt: skip
    path = tmp_path / "turns.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_manifest_loader_labels_the_last_span_as_end_of_turn(tmp_path: Path) -> None:
    data = load_eot_manifest(make_manifest(tmp_path))
    assert len(data.turns) == 4
    t = data.turns[0]
    assert [s.label for s in t.spans] == ["hold", "eot"]
    assert t.words[1] == (1.1, 1.7, "there.") and t.messages == (("assistant", "How can I help?"),)
    assert data.sha256 == load_eot_manifest(tmp_path / "turns.jsonl").sha256  # deterministic
    assert data.limit(2).turns == data.turns[:2]
    assert load_eot_dataset(str(tmp_path / "turns.jsonl")).name == "turns"


def test_builtin_catalog_is_pinned() -> None:
    assert "eot-bench-en" in builtin_eot_datasets() and len(EOT_BENCH_FILES) == 14
    assert all(len(sha) == 64 and size < 300e6 for size, sha in EOT_BENCH_FILES.values())
    assert len(eot_datasets.EOT_BENCH_REVISION) == 40
    with pytest.raises(ValueError):
        load_eot_dataset("eot-bench-xx")


def test_offline_builtin_without_cache_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voice_agent_next.utils.download import DownloadError

    monkeypatch.setenv("VAN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("VAN_OFFLINE", "1")
    with pytest.raises(DownloadError):
        load_eot_dataset("eot-bench-en")


def test_unpack_parquet_writes_wavs_and_manifest(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from voice_agent_next.audio.wav import wav_bytes

    audio = concat(tone(0.5, 16000), silence(0.4, 16000))
    table = pa.table(
        {
            "id": ["en__1"], "audio": [{"bytes": wav_bytes(audio), "path": "x.wav"}],
            "language": ["en"], "duration": [0.9],
            "silence_spans": [[{"start": 0.5, "end": 0.9}]],
            "words": [[{"start": 0.0, "end": 0.5, "word": "hi"}]],
            "messages": [[{"role": "assistant", "content": "Hello"}]],
        }
    )  # fmt: skip
    pq.write_table(table, tmp_path / "x.parquet")
    out = tmp_path / "ds"
    assert eot_datasets._unpack_parquet(tmp_path / "x.parquet", out) == 1
    data = load_eot_manifest(out / "turns.jsonl")
    assert data.turns[0].id == "en__1" and data.turns[0].spans[0].label == "eot"
    assert data.turns[0].load_audio().duration == pytest.approx(0.9)


# --------------------------------------------------------------------------- run


class ContextDetector(TurnDetector):
    """Complete when the visible words end with a period; records its inputs."""

    provider = "test"
    modality = "audio_text"

    def __init__(self) -> None:
        super().__init__(model="ctx", threshold=0.5)
        self.inputs: list[tuple[float, str]] = []

    async def _predict(self, *, audio: AudioFrame | None, chat_ctx: ChatContext | None) -> float:
        assert audio is not None and chat_ctx is not None
        last = chat_ctx.last_message("user")
        text = last.text if last is not None else ""
        self.inputs.append((audio.duration, text))
        return 0.9 if text.endswith(".") else 0.1


def test_turns_run_is_causal_and_scores_a_perfect_detector(tmp_path: Path) -> None:
    data = load_eot_manifest(make_manifest(tmp_path))
    det = ContextDetector()
    results = asyncio.run(
        run_turns_benchmark(det, data, TurnsOptions(transcript_lag=0.2), out_dir=tmp_path / "o")
    )
    # hold scored at 0.6 + 0.2 s: audio up to 0.8 s, "hello" only (0.6 <= 0.8 - 0.2)
    assert det.inputs[0] == (pytest.approx(0.8), "hello")
    # eot at 1.9 s: "there." ended at 1.7 <= 1.9 - 0.2
    assert det.inputs[1] == (pytest.approx(1.9), "hello there.")
    s = results.summary
    assert s.counts["hold_spans"] == 4 and s.counts["eot_spans"] == 4
    assert s.rates["accuracy"] == 1.0 and s.rates["roc_auc"] == 1.0
    assert s.rates["false_cutoff_at_300ms"] == 0.0
    assert s.extra["detector"]["latency_at_5pct_ms"] == pytest.approx(200.0)
    assert s.extra["vad_baseline"]["latency_at_5pct_ms"] == pytest.approx(500.0)  # holds: 0.5 s
    conf = s.extra["configured_policy"]
    assert conf["cutoff_rate"] == 0.0 and conf["mean_latency_ms"] == pytest.approx(400.0)
    assert s.metrics["inference_ms"].n == 8
    table = turns_markdown_table(results)
    assert "false cutoffs @ 300 ms" in table and "VAD baseline" in table
    loaded = load_run(results.directory)  # type: ignore[arg-type]
    assert "End-of-turn detection" in render_turns_report(loaded)


def test_turns_run_with_a_registry_spec(tmp_path: Path) -> None:
    data = load_eot_manifest(make_manifest(tmp_path, turns=2))
    results = asyncio.run(run_turns_benchmark({"provider": "mock", "probability": 0.7}, data))
    s = results.summary
    # always "complete": every pause is cut off once the action delay has passed
    assert s.extra["configured_policy"]["cutoff_rate"] == 1.0
    assert s.rates["recall"] == 1.0 and s.rates["precision"] == 0.5
    assert not math.isnan(s.metrics["inference_ms"].p50 or 0.0)
    assert np.isclose(s.extra["threshold"], 0.5)
