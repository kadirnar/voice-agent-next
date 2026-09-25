"""Regression gate: rules, baseline file, comparison of runs, retries and Markdown."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from voice_agent_next.bench.gate import (
    DEFAULT_ENTRY_RULES,
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
    assert gate_spec_for("e2e.overhead_ms") == ("p50", "overhead")
    assert gate_spec_for("e2e.cascade-delay.overhead_ms") == ("p50", "overhead")
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


# ------------------------------------------------------------- overhead: absolute floor

OVERHEAD = DEFAULT_RULES["overhead"]
WINDOWS_OVERHEAD = DEFAULT_ENTRY_RULES["windows"]["overhead"]


@pytest.mark.parametrize(
    ("base", "current", "status"),
    [
        (2.7, 32.0, "regressed"),  # 12x: the latency rule's 30 ms floor let this through
        (2.7, 8.0, "regressed"),  # +5.3 ms
        (2.7, 7.5, "ok"),  # +4.8 ms: below the 5 ms floor
        (2.7, 3.4, "ok"),  # CI noise
        (5.1, 6.3, "ok"),  # the noisiest Linux condition, its worst run of 45 on CI
        (20.0, 29.0, "ok"),  # +9 ms but only +45 %
        (20.0, 31.0, "regressed"),  # +11 ms and +55 %
        (-14.0, 4.0, "ok"),  # a negative baseline is read as 0: +4 ms
        (-14.0, 6.0, "regressed"),  # ... +6 ms (not +20 ms relative to -14)
        (-3.0, -12.0, "ok"),  # both read as 0
    ],
)
def test_overhead_rule_has_an_absolute_floor(base: float, current: float, status: str) -> None:
    result = compare_metric("e2e.overhead_ms", metric(base), metric(current), OVERHEAD)
    assert result.status == status
    assert result.limit == "+50 % and +5 ms, CIs apart"
    assert result.baseline is not None and result.baseline >= 0
    assert result.current is not None and result.current >= 0


def test_the_latency_rule_alone_misses_a_10x_overhead_regression() -> None:
    assert compare_metric("m", metric(2.7), metric(32.0), LATENCY).status == "ok"


def test_overhead_rule_still_requires_separated_cis() -> None:
    base = metric(2.7, (2.4, 3.2))
    noisy = metric(9.0, (2.9, 16.0))  # one slow session: the CIs overlap
    assert compare_metric("m", base, noisy, OVERHEAD).status == "ok"
    assert compare_metric("m", base, metric(32.0, (30.5, 33.0)), OVERHEAD).status == "regressed"
    below = metric(-1.0, (-9.0, 4.0))  # a CI reaching below 0 is clamped too
    assert compare_metric("m", below, below, OVERHEAD).baseline_ci == (0.0, 4.0)


def test_windows_overhead_floor_is_wider() -> None:
    def judge(base: float, current: float) -> str:
        return compare_metric("m", metric(base), metric(current), WINDOWS_OVERHEAD).status

    assert judge(0.0, 11.0) == "ok"  # the worst Windows run of 45 (16 ms timer ticks)
    assert judge(2.7, 32.0) == "regressed"
    assert judge(-15.0, 21.0) == "regressed"


def test_overhead_metrics_are_gated_by_the_overhead_rule_and_clamped() -> None:
    results = run_with({"e2e.overhead_ms": [-9.0, -5.0, -4.0, 1.0], "e2e.engine.v2v_ms": [-1.0]})
    gated = gated_metrics(results.summary)
    assert gated["e2e.overhead_ms"].rule == "overhead"
    assert gated["e2e.overhead_ms"].value == 0.0  # p50 -4.5 ms is timer granularity
    ci = gated["e2e.overhead_ms"].ci95
    assert ci is not None and min(ci) >= 0.0
    assert gated["e2e.engine.v2v_ms"].value == -1.0  # other rules do not clamp


def run_around(
    values: Mapping[str, float], spread: float, *, system: str, sections: Sequence[str]
) -> RunResults:
    """A run whose per-turn samples scatter by ``±spread`` around ``values``."""
    offsets = [-spread, -spread / 2, -spread / 4, 0.0, spread / 4, spread / 2, spread]
    samples = {k: [v + o for o in offsets] for k, v in values.items()}
    return run_with(samples, system=system, sections=sections)


@pytest.mark.parametrize("os_name", ["Linux", "Windows"])
def test_committed_baseline_passes_noise_and_fails_a_10x_overhead(os_name: str) -> None:
    path = Path(__file__).parents[2] / "benchmarks" / "baselines" / "overhead-ci.json"
    baseline = load_baseline(path)
    entry = baseline.entries[os_name.lower()]
    overhead = {k: m for k, m in entry.metrics.items() if k.endswith("overhead_ms")}
    assert overhead and all(m.rule == "overhead" for m in overhead.values())
    assert all(m.value >= 0 and (m.ci95 is None or m.ci95[0] >= 0) for m in overhead.values())

    same = {k: m.value for k, m in overhead.items()}
    spread = 1.0 if os_name == "Linux" else 12.0  # ~ the CI runners' turn-to-turn noise

    def failures(values: Mapping[str, float]) -> set[str]:
        run = run_around(values, spread, system=os_name, sections=("e2e",))
        report = compare_to_baseline(baseline, run)
        assert report.gated
        return {c.key for c in report.failures if c.key in overhead}  # (v2v etc. not run)

    assert failures(same) == set()
    tenfold = {k: 30.0 + 10 * v for k, v in same.items()}  # 2.7 -> 57 ms, 0 -> 30 ms
    assert failures(tenfold) == set(overhead)


def test_baseline_update_adds_the_new_rules_and_keeps_entry_overrides(tmp_path: Path) -> None:
    path = tmp_path / "overhead-ci.json"
    old = Baseline(
        rules={"latency": LATENCY, "micro": MICRO},  # a file from before the overhead rule
        entries={"windows": BaselineEntry(tier="smoke", run_id="old", created="x",
                                          metrics={"e2e.overhead_ms": metric(-12.0)})},
    )  # fmt: skip
    write_baseline(path, old)
    # an entry recorded before keeps the rule it was recorded with (latency)
    windows = run_with({"e2e.overhead_ms": [15.0]}, system="Windows")
    assert compare_to_baseline(load_baseline(path), windows).passed

    run = run_with({"e2e.overhead_ms": [-9.0, -8.0, -7.0]}, system="Windows")
    baseline, key = update_baseline(path, run)
    assert key == "windows" and baseline.rules["overhead"] == OVERHEAD
    entry = load_baseline(path).entries["windows"]
    assert entry.rules == {"overhead": WINDOWS_OVERHEAD}
    assert entry.metrics["e2e.overhead_ms"].value == 0.0
    assert entry.metrics["e2e.overhead_ms"].rule == "overhead"

    # hand-tuned entry overrides survive the next update
    data = json.loads(path.read_text(encoding="utf-8"))
    data["entries"]["windows"]["rules"]["overhead"]["max_increase_abs"] = 25.0
    path.write_text(json.dumps(data), encoding="utf-8")
    update_baseline(path, run_with({"e2e.overhead_ms": [1.0]}, system="Windows"))
    assert load_baseline(path).entries["windows"].rules["overhead"].max_increase_abs == 25.0
    assert compare_to_baseline(load_baseline(path), windows).passed  # +14 ms < 25 ms
    assert not compare_to_baseline(
        load_baseline(path), run_with({"e2e.overhead_ms": [30.0]}, system="Windows")
    ).passed
    linux_update, _ = update_baseline(path, run_with({"e2e.overhead_ms": [1.0]}))
    assert linux_update.entries["linux"].rules == {}  # Linux uses the file's 5 ms floor
