"""Distribution statistics for benchmark results: percentiles and bootstrap CIs.

Percentiles use linear interpolation between order statistics (numpy's default and
:func:`voice_agent_next.metrics.percentile`). Confidence intervals are percentile
bootstrap intervals with a fixed seed, so re-summarizing the same items is reproducible.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import numpy.typing as npt

__all__ = [
    "CI_STATISTICS",
    "PERCENTILES",
    "bootstrap_ci",
    "clean",
    "describe",
    "percentile",
]

PERCENTILES: tuple[int, ...] = (50, 90, 95, 99)
CI_STATISTICS: tuple[str, ...] = ("mean", "p50", "p90", "p95", "p99")
_CHUNK = 256  # bootstrap resamples evaluated per vectorized batch (bounds memory)


def clean(values: Iterable[float | None]) -> list[float]:
    """Drop ``None`` and non-finite values."""
    return [float(v) for v in values if v is not None and math.isfinite(v)]


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in 0..100); NaN for empty input."""
    xs = clean(values)
    if not xs:
        return math.nan
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


def describe(values: Iterable[float | None]) -> dict[str, float | int]:
    """``n``, ``mean``, ``std`` (sample), ``min``, ``p50/p90/p95/p99`` and ``max``."""
    xs = clean(values)
    if not xs:
        return {"n": 0}
    x = np.asarray(xs, dtype=np.float64)
    out: dict[str, float | int] = {
        "n": len(xs),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(xs) > 1 else 0.0,
        "min": float(x.min()),
    }
    for q, v in zip(PERCENTILES, np.percentile(x, PERCENTILES), strict=True):
        out[f"p{q}"] = float(v)
    out["max"] = float(x.max())
    return out


def _statistic(name: str) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """Vectorized statistic over axis 1 of a (resamples, n) matrix."""
    if name == "mean":
        return lambda m: np.mean(m, axis=1)
    if name == "median":
        name = "p50"
    if name.startswith("p") and name[1:].replace(".", "", 1).isdigit():
        q = float(name[1:])
        if not 0.0 <= q <= 100.0:
            raise ValueError(f"percentile out of range: {name}")
        return lambda m: np.percentile(m, q, axis=1)
    raise ValueError(f"unknown statistic {name!r}; use 'mean' or 'p<q>' (e.g. 'p50')")


def bootstrap_ci(
    values: Iterable[float | None],
    statistic: str = "p50",
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Percentile-bootstrap confidence interval of ``statistic`` (``"mean"`` or ``"p<q>"``).

    Returns ``None`` for empty input and a zero-width interval for a single value.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    fn = _statistic(statistic)
    xs = clean(values)
    if not xs:
        return None
    x = np.asarray(xs, dtype=np.float64)
    if len(x) == 1 or bool(np.all(x == x[0])):
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, _CHUNK):
        size = min(_CHUNK, n_resamples - start)
        idx = rng.integers(0, len(x), size=(size, len(x)))
        stats[start : start + size] = fn(x[idx])
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.percentile(stats, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return (float(lo), float(hi))
