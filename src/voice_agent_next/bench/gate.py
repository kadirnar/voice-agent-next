"""Benchmark regression gate: compare a run with a committed baseline.

The baseline file (``benchmarks/baselines/overhead-ci.json``) holds one *entry* per kind
of machine, keyed by operating system (``linux``, ``windows``, ``darwin``): timer
granularity and CPU speed differ between CI runners, so a run is only ever compared with
a baseline recorded on the same kind of runner (research note 06, §8.4). An entry keeps
the gated statistic of every metric (``p50`` of a latency, ``p99`` of the event-loop
lag, the median of a micro-benchmark), its 95 % bootstrap CI and the rule that judges it.

A metric **regresses** only when it got worse by more than *both* thresholds of its rule:

* ``overhead`` (``overhead_ms``, milliseconds): more than 50 % **and** more than an
  absolute floor of 5 ms (20 ms on Windows, whose asyncio timers are ~16 ms coarse), and
  the 95 % CIs do not overlap. The overhead is a few milliseconds, so the ``latency``
  rule's 30 ms would let a 10x regression through. Overhead is ``>= 0`` by definition: a
  negative value (a timer that fired a tick early) is read as 0 (``min_value``);
* ``latency`` (milliseconds): more than 10 % **and** more than 30 ms, and the 95 % CIs of
  baseline and run do not overlap (research note 06, §8.5);
* ``micro`` (microseconds per operation): more than 200 % (3x) **and** more than 5 µs —
  runner hardware varies, so only large regressions of a hot path fail.

Rules live in the baseline file and can be tuned there, per runner in an entry's
``rules`` (overrides of the file's rules, which override :data:`DEFAULT_RULES`). A gated
metric that the run should have produced but did not (e.g. every turn of a condition was
missed) fails as ``missing``; metrics without a baseline are listed as ``new``. The gate
uses medians of many turns and batches, so a single slow turn on a shared runner cannot
fail it; the CLI can additionally re-run failing sections and fail only on confirmed
regressions.
"""

from __future__ import annotations

import fnmatch
import json
import math
import os
import platform
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .report import fmt, markdown_table
from .results import SUITE, RunResults, RunSummary, json_safe

__all__ = [
    "BASELINE_SCHEMA_VERSION",
    "DEFAULT_BASELINE_PATH",
    "DEFAULT_ENTRY_RULES",
    "DEFAULT_RULES",
    "GATE_SPEC",
    "ROW_HEADERS",
    "Baseline",
    "BaselineEntry",
    "GateReport",
    "GateRule",
    "GatedMetric",
    "MetricComparison",
    "compare_metric",
    "compare_to_baseline",
    "entry_from_run",
    "gate_spec_for",
    "gated_metrics",
    "load_baseline",
    "platform_key",
    "rule_for",
    "update_baseline",
    "write_baseline",
]

BASELINE_SCHEMA_VERSION = 1
DEFAULT_BASELINE_PATH = Path("benchmarks/baselines/overhead-ci.json")

Status = Literal["ok", "improved", "regressed", "missing", "new"]
FAILING: frozenset[str] = frozenset({"regressed", "missing"})


class GateRule(BaseModel):
    """A metric fails when it got worse by more than ``max_increase_pct`` percent **and**
    by more than ``max_increase_abs`` (in the metric's unit) — and, with
    ``require_ci_separation``, when the 95 % CIs of baseline and run do not overlap.

    ``max_increase_abs`` is the absolute floor: for a metric of a few milliseconds the
    relative threshold is tiny and the floor alone decides. ``min_value`` (optional) is
    the smallest meaningful value: lower values and CI bounds are read as it (recording
    and judging), e.g. an overhead measured below 0 because a timer fired early."""

    model_config = ConfigDict(extra="forbid")

    max_increase_pct: float = Field(ge=0)
    max_increase_abs: float = Field(ge=0)
    require_ci_separation: bool = False
    min_value: float | None = None

    def describe(self, unit: str) -> str:
        text = f"+{self.max_increase_pct:g} % and +{self.max_increase_abs:g} {unit}"
        return text + (", CIs apart" if self.require_ci_separation else "")

    def clamp(self, value: float) -> float:
        return value if self.min_value is None else max(self.min_value, value)


DEFAULT_RULES: dict[str, GateRule] = {
    "overhead": GateRule(
        max_increase_pct=50.0, max_increase_abs=5.0, require_ci_separation=True, min_value=0.0
    ),
    "latency": GateRule(max_increase_pct=10.0, max_increase_abs=30.0, require_ci_separation=True),
    "micro": GateRule(max_increase_pct=200.0, max_increase_abs=5.0),
}

GATE_SPEC: tuple[tuple[str, str, str], ...] = (
    # (metric key pattern, gated statistic, rule); first match wins
    ("e2e.overhead_ms", "p50", "overhead"),
    ("e2e.*.overhead_ms", "p50", "overhead"),
    ("e2e.*.v2v_ms", "p50", "latency"),
    ("e2e.frame_jitter_ms", "p50", "latency"),
    ("e2e.loop_lag_ms", "p99", "latency"),
    ("flush.flush_ms", "p50", "latency"),
    ("micro.*", "p50", "micro"),
)
"""Which summary metrics are gated. Everything else is reported only."""

DEFAULT_ENTRY_RULES: dict[str, dict[str, GateRule]] = {
    # asyncio timers are ~16 ms coarse on Windows (Python < 3.13): the p50 overhead of a
    # smoke run swings between about -23 and +11 ms from run to run on the CI runners
    "windows": {
        "overhead": GateRule(
            max_increase_pct=50.0,
            max_increase_abs=20.0,
            require_ci_separation=True,
            min_value=0.0,
        ),
    },
}
"""Rule overrides a new baseline entry of that runner kind starts with."""


class GatedMetric(BaseModel):
    """The gated statistic of one metric (baseline or run)."""

    model_config = ConfigDict(extra="forbid")

    value: float
    stat: str = "p50"
    unit: str = "ms"
    rule: str = "latency"
    ci95: tuple[float, float] | None = None
    n: int = 0


class BaselineEntry(BaseModel):
    """Gated metrics of one reference run on one kind of machine."""

    model_config = ConfigDict(extra="forbid")

    tier: str
    run_id: str
    created: str
    config_sha256: str | None = None
    """Hash of the benchmark settings; a different one means a different workload."""
    git_sha: str | None = None
    machine: str = ""
    rules: dict[str, GateRule] = Field(default_factory=dict)
    """Overrides of the file's rules for this runner kind (kept by baseline updates)."""
    metrics: dict[str, GatedMetric] = Field(default_factory=dict)


class Baseline(BaseModel):
    """The committed baseline file."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = BASELINE_SCHEMA_VERSION
    suite: str = SUITE
    track: str = "overhead"
    description: str = (
        "Regression baseline of `van bench overhead --tier smoke` (T7), one entry per CI "
        "runner OS. Refresh with `van bench overhead --update-baseline` on the runner kind "
        "in question, or from a CI artifact with `--from-run <dir> --update-baseline` "
        "(see benchmarks/README.md)."
    )
    rules: dict[str, GateRule] = Field(default_factory=lambda: dict(DEFAULT_RULES))
    entries: dict[str, BaselineEntry] = Field(default_factory=dict)


# ------------------------------------------------------------------- extraction


def gate_spec_for(key: str) -> tuple[str, str] | None:
    """``(statistic, rule)`` for a summary metric key, or ``None`` if it is not gated."""
    for pattern, stat, rule in GATE_SPEC:
        if fnmatch.fnmatchcase(key, pattern):
            return stat, rule
    return None


def _unit(key: str) -> str:
    return "µs" if key.endswith("_us") else "ms"


def section_of(key: str) -> str:
    return key.split(".", 1)[0]


def rule_for(
    name: str, baseline: Baseline | None = None, entry: BaselineEntry | None = None
) -> GateRule | None:
    """The rule ``name``: the entry's override, else the file's rule, else the default."""
    for rules in (
        entry.rules if entry is not None else {},
        baseline.rules if baseline is not None else {},
        DEFAULT_RULES,
    ):
        if name in rules:
            return rules[name]
    return None


def _clamped(m: GatedMetric, rule: GateRule | None) -> GatedMetric:
    if rule is None or rule.min_value is None:
        return m
    ci = None if m.ci95 is None else (rule.clamp(m.ci95[0]), rule.clamp(m.ci95[1]))
    return m.model_copy(update={"value": rule.clamp(m.value), "ci95": ci})


def gated_metrics(
    summary: RunSummary, rules: Mapping[str, GateRule] | None = None
) -> dict[str, GatedMetric]:
    """The gated statistic of every gated metric of a run summary (clamped to the
    ``min_value`` of its rule, looked up in ``rules`` then :data:`DEFAULT_RULES`)."""
    out: dict[str, GatedMetric] = {}
    for key, dist in summary.metrics.items():
        spec = gate_spec_for(key)
        if spec is None or not dist.n:
            continue
        stat, rule = spec
        value = getattr(dist, stat, None)
        if value is None or not math.isfinite(value):
            continue
        ci = dist.ci95.get(stat)
        m = GatedMetric(value=value, stat=stat, unit=_unit(key), rule=rule, ci95=ci, n=dist.n)
        out[key] = _clamped(m, (rules or {}).get(rule) or DEFAULT_RULES.get(rule))
    return out


def platform_key(environment: Mapping[str, Any] | None = None) -> str:
    """Baseline entry key: the OS of ``environment`` (a run manifest's) or of this machine."""
    system = ((environment or {}).get("os") or {}).get("system") or platform.system()
    return str(system).strip().lower() or "unknown"


def _machine(env: Mapping[str, Any]) -> str:
    cpu = env.get("cpu") or {}
    os_info = env.get("os") or {}
    py = env.get("python") or {}
    parts = [
        str(cpu.get("model") or ""),
        f"{cpu['logical_cores']} threads" if cpu.get("logical_cores") else "",
        f"{os_info.get('system', '')} {os_info.get('release', '')}".strip(),
        f"Python {py['version']}" if py.get("version") else "",
    ]
    return " · ".join(p for p in parts if p)


def entry_from_run(
    results: RunResults,
    *,
    rules: Mapping[str, GateRule] | None = None,
    file_rules: Mapping[str, GateRule] | None = None,
) -> BaselineEntry:
    """A baseline entry holding the gated metrics of ``results``. ``rules`` are the
    entry's overrides of ``file_rules`` (the baseline file's); both clamp the values."""
    manifest, summary = results.manifest, results.summary
    env = manifest.environment
    return BaselineEntry(
        tier=str(manifest.options.get("tier", "custom")),
        run_id=manifest.run_id,
        created=manifest.created,
        config_sha256=manifest.options.get("config_sha256"),
        git_sha=(env.get("git") or {}).get("sha"),
        machine=_machine(env),
        rules=dict(rules or {}),
        metrics=gated_metrics(summary, {**(file_rules or {}), **(rules or {})}),
    )


# ------------------------------------------------------------------------ file I/O


def load_baseline(path: str | os.PathLike[str]) -> Baseline:
    """Read a baseline file (``FileNotFoundError`` if it does not exist)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    version = data.get("schema_version") if isinstance(data, dict) else None
    if version != BASELINE_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported baseline schema_version {version!r} "
            f"(this version reads {BASELINE_SCHEMA_VERSION})"
        )
    return Baseline.model_validate(data)


def write_baseline(path: str | os.PathLike[str], baseline: Baseline) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = json_safe(baseline.model_dump(mode="json"))
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def update_baseline(
    path: str | os.PathLike[str], results: RunResults, *, key: str | None = None
) -> tuple[Baseline, str]:
    """Set the entry of ``key`` (default: the run's OS) to ``results``; other entries and
    the rules (the file's and the entry's overrides) are kept, and default rules the file
    lacks are added (for the entry: :data:`DEFAULT_ENTRY_RULES`). Returns the new baseline
    and the key."""
    target = Path(path)
    baseline = load_baseline(target) if target.exists() else Baseline()
    for name, rule in DEFAULT_RULES.items():
        baseline.rules.setdefault(name, rule)
    key = key or platform_key(results.manifest.environment)
    old = baseline.entries.get(key)
    overrides = {**DEFAULT_ENTRY_RULES.get(key, {}), **(old.rules if old is not None else {})}
    baseline.entries[key] = entry_from_run(results, rules=overrides, file_rules=baseline.rules)
    write_baseline(target, baseline)
    return baseline, key


# ---------------------------------------------------------------------- comparison


@dataclass(slots=True)
class MetricComparison:
    """One metric of the run against its baseline."""

    key: str
    status: Status
    stat: str
    unit: str
    rule: str
    baseline: float | None = None
    current: float | None = None
    baseline_ci: tuple[float, float] | None = None
    current_ci: tuple[float, float] | None = None
    limit: str = ""
    attempts: list[float | None] = field(default_factory=list)
    """Values of re-runs (``--retries``) after a failure."""

    @property
    def section(self) -> str:
        return section_of(self.key)

    @property
    def failed(self) -> bool:
        return self.status in FAILING

    @property
    def delta(self) -> float | None:
        if self.baseline is None or self.current is None:
            return None
        return self.current - self.baseline

    @property
    def delta_pct(self) -> float | None:
        delta = self.delta
        if delta is None or self.baseline is None or self.baseline == 0:
            return None
        return 100.0 * delta / abs(self.baseline)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "status": self.status,
            "stat": self.stat,
            "unit": self.unit,
            "rule": self.rule,
            "baseline": self.baseline,
            "current": self.current,
            "baseline_ci95": self.baseline_ci,
            "current_ci95": self.current_ci,
            "delta": self.delta,
            "delta_pct": self.delta_pct,
            "limit": self.limit,
            "attempts": self.attempts,
        }


def compare_metric(
    key: str, base: GatedMetric, current: GatedMetric | None, rule: GateRule
) -> MetricComparison:
    """Judge one metric (see the module docstring)."""
    base = _clamped(base, rule)
    current = None if current is None else _clamped(current, rule)
    out = MetricComparison(
        key, "ok", base.stat, base.unit, base.rule, baseline=base.value,
        baseline_ci=base.ci95, limit=rule.describe(base.unit),
    )  # fmt: skip
    if current is None:
        out.status = "missing"
        return out
    out.current, out.current_ci = current.value, current.ci95
    delta = current.value - base.value
    scale = abs(base.value)
    ratio = 1.0 + rule.max_increase_pct / 100.0
    if delta > rule.max_increase_abs and delta > scale * (ratio - 1.0):
        separated = True
        if rule.require_ci_separation and base.ci95 is not None and current.ci95 is not None:
            separated = current.ci95[0] > base.ci95[1]
        out.status = "regressed" if separated else "ok"
    elif -delta > rule.max_increase_abs and -delta > scale * (1.0 - 1.0 / ratio):
        out.status = "improved"  # the mirror image (x3 worse <-> /3 better): refresh baseline
    return out


@dataclass
class GateReport:
    """The comparison of one run with one baseline entry."""

    platform: str
    comparisons: list[MetricComparison]
    baseline_path: str | None = None
    entry: BaselineEntry | None = None
    run_id: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(c.failed for c in self.comparisons)

    @property
    def failures(self) -> list[MetricComparison]:
        return [c for c in self.comparisons if c.failed]

    @property
    def gated(self) -> bool:
        """False when there was no baseline entry to compare with (report only)."""
        return self.entry is not None

    def failing_sections(self) -> list[str]:
        return sorted({c.section for c in self.failures})

    def confirm(self, retry: GateReport) -> GateReport:
        """Combine with a re-run of the failing sections: a metric keeps failing only if it
        fails again (a transient slowdown of a shared runner does not)."""
        again = {c.key: c for c in retry.comparisons}
        merged: list[MetricComparison] = []
        for c in self.comparisons:
            if not c.failed:
                merged.append(c)
                continue
            r = again.get(c.key)
            attempts = [*c.attempts, None if r is None else r.current]
            if r is not None and not r.failed:
                merged.append(replace(c, status=r.status, attempts=attempts))
            else:
                merged.append(replace(c, attempts=attempts))
        notes = [*self.notes]
        if retry.run_id:
            notes.append(f"Failing sections re-run as `{retry.run_id}`.")
        return replace(self, comparisons=merged, notes=notes)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "passed": self.passed,
                "gated": self.gated,
                "platform": self.platform,
                "baseline_path": self.baseline_path,
                "baseline_run_id": self.entry.run_id if self.entry else None,
                "baseline_created": self.entry.created if self.entry else None,
                "baseline_git_sha": self.entry.git_sha if self.entry else None,
                "run_id": self.run_id,
                "notes": self.notes,
                "comparisons": [c.to_dict() for c in self.comparisons],
            }
        )

    @property
    def verdict(self) -> str:
        if not self.gated:
            return "REPORT ONLY (no baseline for this runner)"
        if self.passed:
            return "PASS"
        return f"FAIL — {len(self.failures)} regression(s)"

    def rows(self) -> list[tuple[MetricComparison, list[str]]]:
        """Comparisons (failures first) with their table cells, see :data:`ROW_HEADERS`."""
        return [(c, _comparison_row(c)) for c in _ordered(self.comparisons)]

    def to_markdown(self, *, heading: str = "###") -> str:
        """Markdown for a CI job summary."""
        lines = [f"{heading} Regression gate ({self.platform}): **{self.verdict}**", ""]
        if self.entry is not None:
            sha = f" · git `{self.entry.git_sha[:10]}`" if self.entry.git_sha else ""
            lines += [
                f"Baseline `{self.baseline_path}` → `{self.platform}`: run "
                f"`{self.entry.run_id}` ({self.entry.created}{sha}; {self.entry.machine}).",
                "",
            ]
        rows = [cells for _, cells in self.rows()]
        if rows:
            lines += [markdown_table(ROW_HEADERS, rows, ["l", "l", "r", "r", "r", "l", "l"]), ""]
        lines += [f"> {note}" for note in self.notes]
        return "\n".join(lines).rstrip() + "\n"


ROW_HEADERS = ("metric", "stat", "baseline", "run", "change", "fails above", "status")


def _ordered(comparisons: Iterable[MetricComparison]) -> list[MetricComparison]:
    rank = {"regressed": 0, "missing": 1, "improved": 2, "new": 3, "ok": 4}
    return sorted(comparisons, key=lambda c: (rank.get(c.status, 9), c.section != "e2e", c.key))


def _value(value: float | None, unit: str) -> str:
    if value is None:
        return "–"
    digits = 3 if unit == "µs" or abs(value) < 10 else 1
    return f"{fmt(value, digits)} {unit}"


def _comparison_row(c: MetricComparison) -> list[str]:
    change = "–"
    if c.delta is not None:
        pct = "" if c.delta_pct is None else f" ({c.delta_pct:+.0f} %)"
        change = f"{c.delta:+.3g} {c.unit}{pct}"
    status: str = c.status
    if c.attempts:
        retried = ", ".join(_value(v, c.unit) for v in c.attempts)
        status += f" (re-run: {retried})"
    return [f"`{c.key}`", c.stat, _value(c.baseline, c.unit), _value(c.current, c.unit), change,
            c.limit or "–", status]  # fmt: skip


def compare_to_baseline(
    baseline: Baseline,
    results: RunResults,
    *,
    key: str | None = None,
    baseline_path: str | os.PathLike[str] | None = None,
) -> GateReport:
    """Compare ``results`` with the baseline entry of ``key`` (default: the run's OS).

    Only sections that ran (``manifest.options["sections"]``) are judged. Without an entry
    for ``key`` the report lists the run's metrics as new and does not gate.
    """
    manifest = results.manifest
    key = key or platform_key(manifest.environment)
    ran = set(manifest.options.get("sections") or ())
    entry = baseline.entries.get(key)
    current = gated_metrics(
        results.summary, {**baseline.rules, **(entry.rules if entry is not None else {})}
    )
    path = None if baseline_path is None else str(baseline_path)
    notes: list[str] = []
    if entry is None:
        notes.append(
            f"No baseline entry for `{key}`: nothing is gated. Record one with "
            "`van bench overhead --update-baseline` on this kind of runner."
        )
        listed = [
            MetricComparison(k, "new", m.stat, m.unit, m.rule, current=m.value,
                             current_ci=m.ci95)
            for k, m in current.items()
        ]  # fmt: skip
        return GateReport(key, listed, path, None, manifest.run_id, notes)
    tier = manifest.options.get("tier")
    if entry.tier != tier or (
        entry.config_sha256 and entry.config_sha256 != manifest.options.get("config_sha256")
    ):
        notes.append(
            f"The baseline was recorded with different settings (tier `{entry.tier}`, this run "
            f"`{tier}`): numbers may not be comparable."
        )
    comparisons: list[MetricComparison] = []
    for name, base in entry.metrics.items():
        if ran and section_of(name) not in ran:
            continue  # section not run: nothing to judge
        rule = rule_for(base.rule, baseline, entry)
        if rule is None:
            notes.append(f"`{name}`: unknown rule `{base.rule}`, not gated.")
            continue
        comparisons.append(compare_metric(name, base, current.get(name), rule))
    for name, m in current.items():
        if name not in entry.metrics:
            comparisons.append(
                MetricComparison(name, "new", m.stat, m.unit, m.rule, current=m.value,
                                 current_ci=m.ci95)
            )  # fmt: skip
    return GateReport(key, comparisons, path, entry, manifest.run_id, notes)
