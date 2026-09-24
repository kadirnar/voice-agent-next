"""Regression gate: rules, baseline file, comparison of runs, retries and Markdown."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from voice_agent_next.bench.gate import (
    DEFAULT_RULES,
    Baseline,
    BaselineEntry,
    GatedMetric,
    compare_metric,
    compare_to_baseline,
    gate_spec_for,
    gated_metrics,
    load_baseline,
    platform_key,
    update_baseline,
    write_baseline,
)
from voice_agent_next.bench.results import Distribution, RunManifest, RunResults, RunSummary

LATENCY = DEFAULT_RULES["latency"]
MICRO = DEFAULT_RULES["micro"]


def metric(
    value: float,
    ci: tuple[float, float] | None = None,
    *,
    rule: str = "latency",
    unit: str = "ms",
    stat: str = "p50",
) -> GatedMetric:
    return GatedMetric(value=value, stat=stat, unit=unit, rule=rule, ci95=ci, n=10)


def run_with(
    metrics: Mapping[str, Sequence[float]],
    *,
    sections: Sequence[str] = ("e2e", "flush", "micro"),
    system: str = "Linux",
    run_id: str = "run",
    tier: str = "smoke",
) -> RunResults:
    manifest = RunManifest(
        run_id=run_id,
        track="overhead",
        options={"tier": tier, "sections": list(sections), "config_sha256": "abc"},
        environment={
            "os": {"system": system, "release": "6.8"},
            "cpu": {"model": "Test CPU", "logical_cores": 4},
            "python": {"version": "3.12.0"},
            "git": {"sha": "0123456789abcdef"},
        },
    )
    summary = RunSummary(
        run_id=run_id,
        track="overhead",
        system="mocks",
        transport="loopback",
        dataset="overhead-smoke@sha256:x",
        n=0,
        metrics={k: Distribution.of(v, n_resamples=200) for k, v in metrics.items()},
    )
    return RunResults(manifest, [], summary)


def baseline_with(entries: Mapping[str, Mapping[str, GatedMetric]]) -> Baseline:
    return Baseline(
        entries={
            key: BaselineEntry(tier="smoke", run_id=f"base-{key}", created="2026-09-24",
                               config_sha256="abc", metrics=dict(metrics))
            for key, metrics in entries.items()
        }
    )  # fmt: skip


def test_gate_spec_selects_the_gated_metrics() -> None:
    assert gate_spec_for("e2e.overhead_ms") == ("p50", "latency")
    assert gate_spec_for("e2e.cascade-delay.overhead_ms") == ("p50", "latency")
    assert gate_spec_for("e2e.engine.v2v_ms") == ("p50", "latency")
    assert gate_spec_for("e2e.loop_lag_ms") == ("p99", "latency")
    assert gate_spec_for("flush.flush_ms") == ("p50", "latency")
    assert gate_spec_for("micro.energy_vad_us") == ("p50", "micro")
    for reported_only in ("e2e.engine.cpu_pct", "e2e.delivery_lag_ms", "flush.engine.flush_ms",
                          "e2e.engine.residual_ms", "capacity.sessions_per_core"):  # fmt: skip
        assert gate_spec_for(reported_only) is None


def test_gated_metrics_takes_the_rule_statistic_and_its_ci() -> None:
    results = run_with(
        {
            "e2e.overhead_ms": [1.0, 2.0, 3.0, 4.0],
            "e2e.loop_lag_ms": [float(i) for i in range(100)],
            "micro.frame_rms_us": [4.0, 5.0, 6.0],
            "e2e.engine.cpu_pct": [3.0],  # not gated
            "e2e.cascade.overhead_ms": [],  # no values: skipped
        }
    )
    gated = gated_metrics(results.summary)
    assert set(gated) == {"e2e.overhead_ms", "e2e.loop_lag_ms", "micro.frame_rms_us"}
    assert gated["e2e.overhead_ms"].value == pytest.approx(2.5)
    assert gated["e2e.overhead_ms"].ci95 is not None and gated["e2e.overhead_ms"].n == 4
    assert gated["e2e.loop_lag_ms"].stat == "p99"
    assert gated["e2e.loop_lag_ms"].value == pytest.approx(98.01)
    assert gated["micro.frame_rms_us"].unit == "µs" and gated["micro.frame_rms_us"].rule == "micro"


@pytest.mark.parametrize(
    ("base", "current", "status"),
    [
        (2.0, 2.5, "ok"),
        (2.0, 31.0, "ok"),  # +1,450 % but only +29 ms: below the absolute floor
        (2.0, 33.0, "regressed"),  # > 30 ms and > 10 %
        (400.0, 435.0, "ok"),  # +35 ms but only +8.75 %
        (400.0, 445.0, "regressed"),  # +45 ms and +11.25 %
        (400.0, 340.0, "improved"),  # -60 ms and -15 %
        (-1.0, 40.0, "regressed"),  # a negative baseline: relative to |baseline|
    ],
)
def test_latency_rule_needs_both_thresholds(base: float, current: float, status: str) -> None:
    result = compare_metric("m", metric(base), metric(current), LATENCY)
    assert result.status == status
    assert result.delta == pytest.approx(current - base)
    assert result.limit == "+10 % and +30 ms, CIs apart"


def test_latency_rule_requires_separated_confidence_intervals() -> None:
    base = metric(400.0, (390.0, 450.0))
    assert compare_metric("m", base, metric(445.0, (430.0, 460.0)), LATENCY).status == "ok"
    assert compare_metric("m", base, metric(460.0, (451.0, 470.0)), LATENCY).status == "regressed"
    # without a CI on either side, the two thresholds decide alone
    assert compare_metric("m", metric(400.0), metric(445.0), LATENCY).status == "regressed"


def test_micro_rule_only_fails_large_regressions() -> None:
    def judge(base: float, current: float) -> str:
        b, c = metric(base, rule="micro", unit="µs"), metric(current, rule="micro", unit="µs")
        return compare_metric("micro.x_us", b, c, MICRO).status

    assert judge(10.0, 29.0) == "ok"  # +190 %
    assert judge(10.0, 31.0) == "regressed"  # +210 % and +21 µs
    assert judge(1.0, 4.5) == "ok"  # +350 % but only +3.5 µs
    assert judge(30.0, 8.0) == "improved"


def test_missing_metric_fails() -> None:
    result = compare_metric("m", metric(2.0), None, LATENCY)
    assert result.status == "missing" and result.failed and result.current is None


def test_compare_to_baseline_judges_only_sections_that_ran() -> None:
    baseline = baseline_with(
        {
            "linux": {
                "e2e.overhead_ms": metric(2.0),
                "e2e.engine.v2v_ms": metric(400.0),
                "flush.flush_ms": metric(0.05),
                "micro.frame_rms_us": metric(5.0, rule="micro", unit="µs"),
            }
        }
    )
    results = run_with(
        {
            "e2e.overhead_ms": [2.1, 2.2, 2.3],
            "e2e.engine.v2v_ms": [480.0, 481.0, 482.0],  # +81 ms, +20 %: regressed
            "e2e.frame_jitter_ms": [0.1, 0.2],  # not in the baseline: new
            # flush ran but produced nothing -> missing; micro did not run -> skipped
        },
        sections=("e2e", "flush"),
    )
    report = compare_to_baseline(baseline, results, baseline_path="base.json")
    status = {c.key: c.status for c in report.comparisons}
    assert status == {
        "e2e.overhead_ms": "ok",
        "e2e.engine.v2v_ms": "regressed",
        "flush.flush_ms": "missing",
        "e2e.frame_jitter_ms": "new",
    }
    assert report.platform == "linux" and report.gated and not report.passed
    assert report.failing_sections() == ["e2e", "flush"]
    assert [c.key for c, _ in report.rows()][:2] == ["e2e.engine.v2v_ms", "flush.flush_ms"]
    md = report.to_markdown()
    assert "Regression gate (linux): **FAIL — 2 regression(s)**" in md
    assert "| `e2e.engine.v2v_ms` | p50 | 400.0 ms | 481.0 ms | +81 ms (+20 %) |" in md
    assert "base-linux" in md and "`base.json`" in md
    data = report.to_dict()
    assert data["passed"] is False and data["baseline_run_id"] == "base-linux"
    json.dumps(data, allow_nan=False)


def test_runs_are_compared_with_the_entry_of_their_os() -> None:
    baseline = baseline_with({"windows": {"e2e.overhead_ms": metric(20.0)}})
    linux = run_with({"e2e.overhead_ms": [900.0]})
    report = compare_to_baseline(baseline, linux)
    assert not report.gated and report.passed  # no linux entry: report only
    assert [c.status for c in report.comparisons] == ["new"]
    assert "No baseline entry for `linux`" in report.notes[0]
    assert "REPORT ONLY" in report.to_markdown()
    windows = run_with({"e2e.overhead_ms": [21.0]}, system="Windows")
    assert compare_to_baseline(baseline, windows).passed
    assert compare_to_baseline(baseline, linux, key="windows").failures  # explicit entry
    assert platform_key({"os": {"system": "Darwin"}}) == "darwin"


def test_different_settings_are_noted_but_still_compared() -> None:
    baseline = baseline_with({"linux": {"e2e.overhead_ms": metric(2.0)}})
    report = compare_to_baseline(baseline, run_with({"e2e.overhead_ms": [2.0]}, tier="full"))
    assert report.passed and any("different settings" in n for n in report.notes)


def test_confirm_keeps_only_regressions_that_repeat() -> None:
    baseline = baseline_with(
        {"linux": {"e2e.overhead_ms": metric(2.0), "flush.flush_ms": metric(0.05)}}
    )
    first = compare_to_baseline(
        baseline, run_with({"e2e.overhead_ms": [90.0], "flush.flush_ms": [60.0]})
    )
    assert {c.key for c in first.failures} == {"e2e.overhead_ms", "flush.flush_ms"}
    retry = compare_to_baseline(
        baseline,
        run_with({"e2e.overhead_ms": [2.5], "flush.flush_ms": [70.0]}, run_id="run-retry1"),
    )
    merged = first.confirm(retry)
    status = {c.key: (c.status, c.attempts) for c in merged.comparisons}
    assert status["e2e.overhead_ms"] == ("ok", [2.5])  # transient: passed on the re-run
    assert status["flush.flush_ms"] == ("regressed", [70.0])  # confirmed
    assert not merged.passed and "run-retry1" in merged.notes[-1]
    assert "(re-run: 70.0 ms)" in merged.to_markdown()


def test_baseline_file_round_trip_and_update(tmp_path: Path) -> None:
    path = tmp_path / "baselines" / "overhead-ci.json"
    linux = run_with({"e2e.overhead_ms": [1.0, 2.0, 3.0]}, run_id="linux-run")
    baseline, key = update_baseline(path, linux)
    assert key == "linux" and path.is_file()
    entry = baseline.entries["linux"]
    assert entry.run_id == "linux-run" and entry.tier == "smoke" and entry.config_sha256 == "abc"
    assert entry.git_sha == "0123456789abcdef" and "Test CPU" in entry.machine
    assert entry.metrics["e2e.overhead_ms"].value == pytest.approx(2.0)
    assert load_baseline(path) == baseline

    # hand-tuned rules survive updates; other runners' entries are kept
    data = json.loads(path.read_text(encoding="utf-8"))
    data["rules"]["latency"]["max_increase_abs"] = 50.0
    path.write_text(json.dumps(data), encoding="utf-8")
    windows = run_with({"e2e.overhead_ms": [9.0]}, system="Windows", run_id="win-run")
    updated, key = update_baseline(path, windows)
    assert key == "windows" and set(updated.entries) == {"linux", "windows"}
    assert updated.rules["latency"].max_increase_abs == 50.0
    assert load_baseline(path).entries["linux"].run_id == "linux-run"

    data["schema_version"] = 99
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_baseline(path)
    with pytest.raises(FileNotFoundError):
        load_baseline(tmp_path / "missing.json")
    assert write_baseline(tmp_path / "empty.json", Baseline()).is_file()
    assert load_baseline(tmp_path / "empty.json").entries == {}
