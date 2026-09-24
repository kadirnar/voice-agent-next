"""Percentile and bootstrap confidence-interval math."""

from __future__ import annotations

import math
import statistics

import numpy as np
import pytest

from voice_agent_next import metrics
from voice_agent_next.bench.results import Distribution
from voice_agent_next.bench.stats import bootstrap_ci, clean, describe, percentile


@pytest.mark.parametrize("q", [0, 10, 50, 90, 95, 99, 100])
def test_percentile_is_linear_interpolation(q: float) -> None:
    xs = [5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 11.0]
    expected = float(np.percentile(xs, q))
    assert percentile(xs, q) == pytest.approx(expected)
    assert metrics.percentile(xs, q) == pytest.approx(expected)  # same definition


def test_percentile_by_hand() -> None:
    xs = [10.0, 20.0, 30.0, 40.0]
    assert percentile(xs, 50) == pytest.approx(25.0)  # k = 1.5
    assert percentile(xs, 90) == pytest.approx(37.0)  # k = 2.7
    assert math.isnan(percentile([], 50))


def test_describe_ignores_missing_values() -> None:
    xs = [4.0, None, 1.0, float("nan"), 3.0, 2.0, float("inf")]
    assert clean(xs) == [4.0, 1.0, 3.0, 2.0]
    d = describe(xs)
    assert d["n"] == 4
    assert d["mean"] == pytest.approx(2.5)
    assert d["std"] == pytest.approx(statistics.stdev([1, 2, 3, 4]))
    assert (d["min"], d["max"]) == (1.0, 4.0)
    assert d["p50"] == pytest.approx(2.5)
    assert d["p99"] == pytest.approx(3.97)
    assert describe([]) == {"n": 0}
    assert describe([7.0])["std"] == 0.0


def test_bootstrap_ci_covers_the_statistic_and_is_reproducible() -> None:
    xs = np.random.default_rng(1).normal(500.0, 50.0, 200)
    lo, hi = bootstrap_ci(xs, "p50", seed=0)
    assert lo < float(np.median(xs)) < hi
    assert 5.0 < hi - lo < 30.0  # SE(median) ~ 1.25 * 50 / sqrt(200) ~ 4.4 ms
    assert bootstrap_ci(xs, "p50", seed=0) == (lo, hi)
    assert bootstrap_ci(xs, "p50", seed=1) != (lo, hi)
    m_lo, m_hi = bootstrap_ci(xs, "mean", seed=0)
    half = 1.96 * float(np.std(xs, ddof=1)) / math.sqrt(len(xs))
    assert m_lo == pytest.approx(float(np.mean(xs)) - half, abs=2.5)
    assert m_hi == pytest.approx(float(np.mean(xs)) + half, abs=2.5)
    p90 = bootstrap_ci(xs, "p90", seed=0)
    assert p90 is not None and p90[0] < float(np.percentile(xs, 90)) < p90[1]


def test_bootstrap_ci_narrows_with_more_samples() -> None:
    rng = np.random.default_rng(2)
    small = rng.normal(0.0, 1.0, 20)
    large = rng.normal(0.0, 1.0, 2000)
    s_lo, s_hi = bootstrap_ci(small, "p50")
    l_lo, l_hi = bootstrap_ci(large, "p50")
    assert l_hi - l_lo < (s_hi - s_lo) / 3


def test_bootstrap_ci_edge_cases() -> None:
    assert bootstrap_ci([]) is None
    assert bootstrap_ci([None, float("nan")]) is None
    assert bootstrap_ci([42.0]) == (42.0, 42.0)
    assert bootstrap_ci([3.0, 3.0, 3.0]) == (3.0, 3.0)
    assert bootstrap_ci([1.0, 2.0], "median") is not None
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], "mode")
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], "p101")
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], confidence=1.0)
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], n_resamples=0)


def test_distribution_summary() -> None:
    values = [float(v) for v in range(100, 200)]
    d = Distribution.of([*values, None])
    assert d.n == 100
    assert d.mean == pytest.approx(149.5)
    assert d.p50 == pytest.approx(149.5)
    assert d.p90 == pytest.approx(float(np.percentile(values, 90)))
    assert (d.min, d.max) == (100.0, 199.0)
    assert set(d.ci95) == {"mean", "p50", "p90", "p95", "p99"}
    for name, (lo, hi) in d.ci95.items():
        assert lo <= getattr(d, name) <= hi
    assert Distribution.of(values, ci=()).ci95 == {}
    empty = Distribution.of([None])
    assert empty.n == 0 and empty.p50 is None and empty.ci95 == {}
