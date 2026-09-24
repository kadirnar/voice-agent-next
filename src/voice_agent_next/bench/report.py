"""Markdown reports for benchmark runs (``report.md``).

The report is generated from the same objects that are written to ``manifest.json``,
``items.jsonl`` and ``summary.json``, so it can always be re-rendered from a run
directory (``van bench report <run-dir>``). Tracks supply labels, per-item columns and
extra sections (e.g. metric definitions).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .results import Distribution, RunResults

__all__ = ["ReportSpec", "fmt", "markdown_table", "render_report"]


def fmt(value: Any, digits: int = 0, *, unit: str = "") -> str:
    """Human-friendly cell: ``None``/NaN -> ``–``; floats rounded to ``digits``."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:,.{digits}f}{unit}"
    if isinstance(value, int):
        return f"{value:,}{unit}"
    return f"{value}{unit}"


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def markdown_table(
    headers: Sequence[str], rows: Sequence[Sequence[str]], align: Sequence[str] | None = None
) -> str:
    """A GitHub-flavoured Markdown table; ``align`` items are ``"l"``, ``"r"`` or ``"c"``."""
    marks = {"l": ":---", "r": "---:", "c": ":---:"}
    align = align or ["l"] * len(headers)
    lines = [
        "| " + " | ".join(_cell(h) for h in headers) + " |",
        "| " + " | ".join(marks.get(a, "---") for a in align) + " |",
    ]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


@dataclass
class ReportSpec:
    """Track-specific presentation of a run."""

    title: str
    metric_labels: Mapping[str, str] = field(default_factory=dict)
    """Metrics shown in the results table, in order (key -> label). Empty: all metrics."""
    rate_labels: Mapping[str, str] = field(default_factory=dict)
    item_columns: Sequence[tuple[str, str]] = ()
    """``(item key, header)`` columns of the per-item table."""
    sections: Sequence[tuple[str, str]] = ()
    """Extra ``(heading, markdown)`` sections appended after the results."""
    max_items: int = 200
    digits: int = 0


def _dist_row(label: str, d: Distribution, digits: int) -> list[str]:
    ci = d.ci95.get("p50")
    p50 = fmt(d.p50, digits)
    if ci is not None and d.n > 1:
        p50 += f" [{fmt(ci[0], digits)}, {fmt(ci[1], digits)}]"
    return [
        label,
        fmt(d.n),
        fmt(d.mean, digits),
        p50,
        fmt(d.p90, digits),
        fmt(d.p95, digits),
        fmt(d.p99, digits),
        fmt(d.max, digits),
    ]


def _environment_line(env: Mapping[str, Any]) -> str:
    cpu = env.get("cpu") or {}
    parts = []
    if cpu.get("model"):
        parts.append(str(cpu["model"]))
    if cpu.get("logical_cores"):
        parts.append(f"{cpu['logical_cores']} threads")
    if env.get("memory_bytes"):
        parts.append(f"{env['memory_bytes'] / 2**30:.1f} GiB RAM")
    gpus = env.get("gpus") or []
    if gpus:
        parts.append(", ".join(str(g.get("name")) for g in gpus))
    os_info = env.get("os") or {}
    if os_info:
        parts.append(f"{os_info.get('system', '')} {os_info.get('release', '')}".strip())
    py = env.get("python") or {}
    if py.get("version"):
        parts.append(f"Python {py['version']}")
    return " · ".join(parts) or "–"


def _version_line(env: Mapping[str, Any]) -> str:
    version = (env.get("packages") or {}).get("voice-agent-next", "?")
    git = env.get("git") or {}
    line = f"voice-agent-next {version}"
    if git.get("sha"):
        line += f" · git `{str(git['sha'])[:10]}`" + (" (dirty)" if git.get("dirty") else "")
    return line


def render_report(results: RunResults, spec: ReportSpec) -> str:
    """Render a run as Markdown."""
    m, s = results.manifest, results.summary
    out: list[str] = [f"# {spec.title}", ""]
    info = [
        ["run", f"`{m.run_id}`"],
        ["track", m.track],
        ["system", s.system],
        ["dataset", f"`{s.dataset}`"],
        ["transport", s.transport],
        ["started", m.created],
        ["duration", fmt(s.duration_s, 1, unit=" s")],
        ["software", _version_line(m.environment)],
        ["machine", _environment_line(m.environment)],
    ]
    out += [markdown_table(["", ""], info), ""]

    out += ["## Results", ""]
    labels = dict(spec.metric_labels) or {k: k for k in s.metrics}
    rows = [_dist_row(label, s.metrics[key], spec.digits) for key, label in labels.items()
            if key in s.metrics]  # fmt: skip
    if rows:
        headers = ["metric", "n", "mean", "p50 [95% CI]", "p90", "p95", "p99", "max"]
        out += [markdown_table(headers, rows, ["l"] + ["r"] * 7), ""]
    rate_labels = dict(spec.rate_labels) or {k: k for k in s.rates}
    rate_rows = [
        [label, "–" if s.rates.get(key) is None else f"{100 * float(s.rates[key] or 0):.1f}%"]
        for key, label in rate_labels.items()
        if key in s.rates
    ]
    if rate_rows:
        out += [markdown_table(["rate", "value"], rate_rows, ["l", "r"]), ""]
    if s.counts:
        count_rows = [[k, fmt(v)] for k, v in s.counts.items()]
        out += [markdown_table(["count", "value"], count_rows, ["l", "r"]), ""]

    for heading, body in spec.sections:
        out += [f"## {heading}", "", body.strip(), ""]

    if spec.item_columns and results.items:
        shown = results.items[: spec.max_items]
        out += ["## Items", ""]
        headers = [h for _, h in spec.item_columns]
        item_rows = [
            [fmt(item.get(key), spec.digits) for key, _ in spec.item_columns] for item in shown
        ]
        out += [markdown_table(headers, item_rows), ""]
        if len(results.items) > len(shown):
            out += [f"_{len(results.items) - len(shown)} more items in `items.jsonl`._", ""]
    for note in m.notes:
        out += [f"> {note}", ""]
    return "\n".join(out).rstrip() + "\n"
