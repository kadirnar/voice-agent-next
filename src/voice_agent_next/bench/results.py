"""Benchmark result schema and run directories (research note 06, §8.5).

A run directory ``<out>/<run_id>/`` contains:

* ``manifest.json`` — what was measured and on what: engine config, scenario and
  stimulus hashes, options, versions, git SHA, hardware, OS, Python, timestamp;
* ``items.jsonl`` — one JSON object per item (e.g. user turn) and trial/session;
* ``summary.json`` — per-metric distributions (n, mean, p50/p90/p95/p99, max and
  bootstrap 95% confidence intervals), rates and counts;
* ``report.md`` — a human-readable summary (:mod:`voice_agent_next.bench.report`);
* ``artifacts/`` — recordings (stereo WAV: user left, agent right) and labels.

The schema is versioned (:data:`SCHEMA_VERSION`); readers must check it.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .stats import CI_STATISTICS, bootstrap_ci, describe

__all__ = [
    "ARTIFACTS_DIR",
    "ITEMS_FILE",
    "MANIFEST_FILE",
    "REPORT_FILE",
    "SCHEMA_VERSION",
    "SUITE",
    "SUITE_VERSION",
    "SUMMARY_FILE",
    "Distribution",
    "RunManifest",
    "RunResults",
    "RunSummary",
    "json_safe",
    "load_run",
    "new_run_id",
    "slugify",
    "utc_timestamp",
    "write_run",
]

SUITE = "van-bench"
SUITE_VERSION = "0.1.0"
"""Bump the major version whenever a methodology change makes results incomparable."""
SCHEMA_VERSION = 1

MANIFEST_FILE = "manifest.json"
ITEMS_FILE = "items.jsonl"
SUMMARY_FILE = "summary.json"
REPORT_FILE = "report.md"
ARTIFACTS_DIR = "artifacts"


class Distribution(BaseModel):
    """Summary statistics of one metric (values in the metric's unit, e.g. ms)."""

    model_config = ConfigDict(extra="forbid")

    n: int = 0
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    p50: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    max: float | None = None
    ci95: dict[str, tuple[float, float]] = Field(default_factory=dict)
    """95% bootstrap confidence intervals, keyed by statistic (``mean``, ``p50``...)."""

    @classmethod
    def of(
        cls,
        values: Iterable[float | None],
        *,
        ci: Iterable[str] = CI_STATISTICS,
        n_resamples: int = 2000,
        seed: int = 0,
        digits: int = 3,
    ) -> Distribution:
        """Describe ``values`` (``None``/NaN are ignored) with bootstrap CIs."""
        xs = [v for v in values if v is not None and math.isfinite(v)]
        stats = describe(xs)
        if not xs:
            return cls(n=0)
        intervals: dict[str, tuple[float, float]] = {}
        for name in ci:
            interval = bootstrap_ci(xs, name, n_resamples=n_resamples, seed=seed)
            if interval is not None:
                intervals[name] = (round(interval[0], digits), round(interval[1], digits))
        fields = {k: round(float(v), digits) for k, v in stats.items() if k != "n"}
        return cls(n=int(stats["n"]), ci95=intervals, **fields)


class RunManifest(BaseModel):
    """Everything needed to interpret and reproduce a run (research note 06, §8.6)."""

    model_config = ConfigDict(extra="forbid")

    suite: str = SUITE
    suite_version: str = SUITE_VERSION
    schema_version: int = SCHEMA_VERSION
    run_id: str
    track: str
    created: str = Field(default_factory=lambda: utc_timestamp())
    """UTC ISO-8601 start time of the run."""
    system: dict[str, Any] = Field(default_factory=dict)
    """System under test: engine / cascade components as configured and as resolved."""
    scenario: dict[str, Any] = Field(default_factory=dict)
    """Scenario definition, its hash and one entry (with SHA-256) per stimulus."""
    transport: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    """Versions, git SHA, lockfile hash, hardware (CPU/GPU/RAM), OS, Python."""
    notes: list[str] = Field(default_factory=list)


class RunSummary(BaseModel):
    """Aggregated results of a run (``summary.json``)."""

    model_config = ConfigDict(extra="forbid")

    suite: str = SUITE
    suite_version: str = SUITE_VERSION
    schema_version: int = SCHEMA_VERSION
    run_id: str
    track: str
    system: str
    """Short label of the system under test, e.g. ``mock`` or ``cascade:mock+mock+mock``."""
    transport: str
    dataset: str
    """Scenario name and hash, e.g. ``latency-smoke@sha256:0123abcd4567``."""
    n: int
    """Number of items in the headline population."""
    metrics: dict[str, Distribution] = Field(default_factory=dict)
    rates: dict[str, float | None] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)
    duration_s: float | None = None
    """Wall-clock duration of the run."""


@dataclass
class RunResults:
    """A complete run: manifest, per-item records, summary and optional report."""

    manifest: RunManifest
    items: list[dict[str, Any]]
    summary: RunSummary
    report: str | None = None
    directory: Path | None = None


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with ``None`` and tuples with lists."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def _dump(data: Any) -> str:
    return json.dumps(json_safe(data), indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def write_run(directory: str | Path, results: RunResults) -> Path:
    """Write ``manifest.json``, ``items.jsonl``, ``summary.json`` (+ ``report.md``)."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / MANIFEST_FILE).write_text(
        _dump(results.manifest.model_dump(mode="json")), encoding="utf-8"
    )
    with (path / ITEMS_FILE).open("w", encoding="utf-8") as f:
        for item in results.items:
            f.write(json.dumps(json_safe(item), ensure_ascii=False, allow_nan=False) + "\n")
    (path / SUMMARY_FILE).write_text(
        _dump(results.summary.model_dump(mode="json")), encoding="utf-8"
    )
    if results.report is not None:
        (path / REPORT_FILE).write_text(results.report, encoding="utf-8")
    results.directory = path
    return path


def load_run(directory: str | Path) -> RunResults:
    """Read a run directory written by :func:`write_run`."""
    path = Path(directory)
    manifest_data = json.loads((path / MANIFEST_FILE).read_text(encoding="utf-8"))
    version = manifest_data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{path / MANIFEST_FILE}: unsupported schema_version {version!r} "
            f"(this version reads {SCHEMA_VERSION})"
        )
    manifest = RunManifest.model_validate(manifest_data)
    items: list[dict[str, Any]] = []
    items_path = path / ITEMS_FILE
    if items_path.exists():
        with items_path.open(encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
    summary = RunSummary.model_validate_json((path / SUMMARY_FILE).read_text(encoding="utf-8"))
    report_path = path / REPORT_FILE
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else None
    return RunResults(manifest, items, summary, report, path)


def utc_timestamp(when: datetime | None = None) -> str:
    """ISO-8601 UTC timestamp with second resolution."""
    return (when or datetime.now(UTC)).astimezone(UTC).isoformat(timespec="seconds")


def slugify(text: str, *, max_len: int = 48) -> str:
    """File-system friendly identifier: ``"cascade:mock+mock"`` -> ``"cascade-mock-mock"``."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._").lower()
    return slug[:max_len].rstrip("-._") or "run"


def new_run_id(track: str, label: str, when: datetime | None = None) -> str:
    """``<UTC timestamp>-<track>-<label>``, e.g. ``20260924T171500Z-latency-mock``."""
    stamp = (when or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{slugify(track)}-{slugify(label)}"
