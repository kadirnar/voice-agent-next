"""Result schema round trip (manifest.json / items.jsonl / summary.json) and reports."""

from __future__ import annotations

import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path

import pydantic
import pytest

from voice_agent_next.bench.environment import collect_environment
from voice_agent_next.bench.report import ReportSpec, fmt, markdown_table, render_report
from voice_agent_next.bench.results import (
    ITEMS_FILE,
    MANIFEST_FILE,
    REPORT_FILE,
    SCHEMA_VERSION,
    SUMMARY_FILE,
    Distribution,
    RunManifest,
    RunResults,
    RunSummary,
    json_safe,
    load_run,
    new_run_id,
    slugify,
    write_run,
)
from voice_agent_next.bench.tracks.latency import (
    LatencyItem,
    render_latency_report,
    summarize_latency,
)


def make_items() -> list[LatencyItem]:
    rows = [(0, 0, 612.5, True), (0, 1, 480.25, False), (0, 2, None, False), (1, 0, 700.0, True),
            (1, 1, 2450.0, False), (1, 2, 455.5, False)]  # fmt: skip
    items = []
    for session, turn, v2v, warmup in rows:
        items.append(
            LatencyItem(
                session=session,
                turn=turn,
                stimulus=f"s{turn}",
                text="What time is it?",
                warmup=warmup,
                user_speech_start_s=1.0 + 3 * turn,
                user_speech_end_s=1.6 + 3 * turn,
                agent_onset_s=None if v2v is None else 1.6 + 3 * turn + v2v / 1000,
                v2v_ms=v2v,
                session_v2v_ms=None if v2v is None else v2v - 1.5,
                residual_ms=None if v2v is None else 1.5,
                eou_delay_ms=400.0,
                missed=v2v is None,
                dead_air=v2v is None or v2v > 2000,
            )
        )
    return items


def make_results() -> RunResults:
    items = make_items()
    sessions = [{"session": 0, "session_ready_ms": 12.0}, {"session": 1, "session_ready_ms": 9.0}]
    metrics, rates, counts, extra = summarize_latency(items, sessions, n_resamples=200)
    manifest = RunManifest(
        run_id="20260924T000000Z-latency-mock",
        track="latency",
        system={"label": "mock", "config": {"engine": "mock"}},
        scenario={"name": "unit", "stimuli": [{"id": "s0", "sha256": "0" * 64}]},
        transport={"type": "loopback"},
        options={"turns": 3, "sessions": 2, "warmup_turns": 1, "onset": {"frame_ms": 10}},
        environment=collect_environment(),
        notes=["a note"],
    )
    summary = RunSummary(
        run_id=manifest.run_id,
        track="latency",
        system="mock",
        transport="loopback",
        dataset="unit@sha256:0123456789ab",
        n=metrics["v2v_ms"].n,
        metrics=metrics,
        rates=rates,
        counts=counts,
        extra=extra,
        duration_s=12.5,
    )
    return RunResults(manifest, [it.model_dump(mode="json") for it in items], summary)


def test_summary_of_latency_items() -> None:
    items = make_items()
    metrics, rates, counts, extra = summarize_latency(items, [{"session_ready_ms": 5.0}])
    # headline: non-warm-up turns only; the missed turn has no v2v
    assert metrics["v2v_ms"].n == 3
    assert metrics["v2v_ms"].p50 == pytest.approx(480.25)
    assert metrics["first_turn_v2v_ms"].n == 2
    assert metrics["first_turn_v2v_ms"].mean == pytest.approx(656.25)
    assert metrics["session_ready_ms"].n == 1
    assert "llm_ttft_ms" not in metrics  # no data -> omitted
    assert rates["missed_rate"] == pytest.approx(1 / 4)
    assert rates["dead_air_rate"] == pytest.approx(2 / 4)  # the missed turn + 2450 ms
    assert counts["turns"] == 6 and counts["turns_measured"] == 4 and counts["warmup_turns"] == 2
    assert extra["spans_p50_ms"]["eou_delay"] == pytest.approx(400.0)


def test_run_directory_round_trip(tmp_path: Path) -> None:
    results = make_results()
    results.report = render_latency_report(results)
    run_dir = write_run(tmp_path / "run", results)
    for name in (MANIFEST_FILE, ITEMS_FILE, SUMMARY_FILE, REPORT_FILE):
        assert (run_dir / name).is_file()

    def strict(name: str) -> None:
        raise ValueError(f"non-standard JSON constant {name}")

    for name in (MANIFEST_FILE, SUMMARY_FILE):
        json.loads((run_dir / name).read_text(encoding="utf-8"), parse_constant=strict)
    lines = (run_dir / ITEMS_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(results.items)

    loaded = load_run(run_dir)
    assert loaded.manifest == results.manifest
    assert loaded.summary == results.summary
    assert loaded.items == results.items
    assert [LatencyItem.model_validate(i) for i in loaded.items] == make_items()
    assert loaded.report == results.report
    assert loaded.summary.metrics["v2v_ms"].ci95["p50"][0] <= loaded.summary.metrics["v2v_ms"].p50
    assert loaded.manifest.schema_version == SCHEMA_VERSION


def test_non_finite_values_become_null(tmp_path: Path) -> None:
    results = make_results()
    results.items.append({"v2v_ms": float("nan"), "nested": {"x": float("inf"), "t": (1.0, 2.0)}})
    run_dir = write_run(tmp_path / "run", results)
    last = json.loads((run_dir / ITEMS_FILE).read_text(encoding="utf-8").splitlines()[-1])
    assert last == {"v2v_ms": None, "nested": {"x": None, "t": [1.0, 2.0]}}
    assert json_safe({"a": [float("-inf"), 1]}) == {"a": [None, 1]}


def test_schema_is_versioned_and_strict(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path / "run", make_results())
    manifest = json.loads((run_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    manifest["schema_version"] = SCHEMA_VERSION + 1
    (run_dir / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_run(run_dir)
    summary = make_results().summary.model_dump()
    with pytest.raises(pydantic.ValidationError):
        RunSummary.model_validate({**summary, "unexpected": 1})
    with pytest.raises(pydantic.ValidationError):
        Distribution.model_validate({"n": 1, "median": 3.0})


def test_run_ids_and_slugs() -> None:
    when = datetime(2026, 9, 24, 17, 15, 0, tzinfo=UTC)
    assert new_run_id("latency", "cascade:mock+mock+mock", when) == (
        "20260924T171500Z-latency-cascade-mock-mock-mock"
    )
    assert slugify("openai/gpt-realtime") == "openai-gpt-realtime"
    assert slugify("{provider: mock}") == "provider-mock"
    assert slugify("///") == "run"


def test_environment_manifest_is_complete_and_json_safe() -> None:
    env = collect_environment()
    json.dumps(env)
    assert env["python"]["version"] == platform.python_version()
    assert env["cpu"]["logical_cores"] == os.cpu_count()
    assert env["os"]["system"] == platform.system()
    assert "voice-agent-next" in env["packages"] and "numpy" in env["packages"]
    assert env["memory_bytes"] is None or env["memory_bytes"] > 0
    if env["git"] is not None:  # running from a checkout
        assert len(env["git"]["sha"]) == 40


def test_report_contents() -> None:
    results = make_results()
    text = render_latency_report(results)
    assert text.startswith("# Latency (T1) · mock")
    assert "`20260924T000000Z-latency-mock`" in text
    assert "**voice-to-voice** `v2v_ms` (recording)" in text
    assert "| 480 [" in text  # p50 with its bootstrap CI
    assert "dead air (> 2000 ms or no reply) | 50.0%" in text
    assert "## Method" in text and "## Items" in text
    assert "> a note" in text
    spec = ReportSpec(title="T", item_columns=[("turn", "turn")], max_items=2)
    short = render_report(results, spec)
    assert "_4 more items in `items.jsonl`._" in short


def test_markdown_helpers() -> None:
    table = markdown_table(["a", "b"], [["x|y", "1"]], ["l", "r"])
    assert table.splitlines() == ["| a | b |", "| :--- | ---: |", "| x\\|y | 1 |"]
    assert fmt(None) == "–" and fmt(float("nan")) == "–"
    assert fmt(-0.2) == "0" and fmt(1234.5678, 1) == "1,234.6"
    assert fmt(True) == "yes" and fmt(3) == "3" and fmt("x") == "x"
