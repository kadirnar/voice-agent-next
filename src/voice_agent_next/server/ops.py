"""Operations for served agents: health, readiness, Prometheus metrics and session logs.

:class:`ServeState` is shared by every protocol of ``van serve``. It answers:

* ``GET /health`` — liveness: 200 while the process serves (also while draining);
* ``GET /ready`` — readiness: 200 when a new call would be served warm right now (the
  engines are warm, a prewarmed engine is available, the session limit is not reached and
  the process is not draining), 503 otherwise, with the reasons;
* ``GET /metrics`` — Prometheus text format (active sessions, pool size, sessions total,
  refusals, errors, voice-to-voice and engine-connect histograms).

Session logs carry the session id: :func:`session_context` sets it for the current task
(and the tasks it starts), :class:`SessionLogFilter` adds it to every record and
:class:`JsonLogFormatter` writes one JSON object per line.
"""

from __future__ import annotations

import contextvars
import json
import logging
import math
import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http import HTTPStatus
from typing import Any

from ..utils.clock import now
from .pool import EnginePool

__all__ = [
    "V2V_BUCKETS",
    "Histogram",
    "JsonLogFormatter",
    "ServeState",
    "SessionLogFilter",
    "configure_logging",
    "current_session_id",
    "session_context",
]

V2V_BUCKETS: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
"""Voice-to-voice latency buckets, seconds."""
CONNECT_BUCKETS: tuple[float, ...] = (
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
)  # fmt: skip
"""Engine connection (session setup) buckets, seconds."""

PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "van_session_id", default=None
)


# ------------------------------------------------------------------------ logging
def current_session_id() -> str | None:
    """The session id of the running task (``None`` outside a session)."""
    return _session_id.get()


def set_session_id(session_id: str | None) -> None:
    """Tag the current task (and the tasks it creates from now on) with ``session_id``."""
    _session_id.set(session_id)


@contextmanager
def session_context(session_id: str | None) -> Iterator[None]:
    """Log records emitted inside the block carry ``session_id``."""
    token = _session_id.set(session_id)
    try:
        yield
    finally:
        _session_id.reset(token)


class SessionLogFilter(logging.Filter):
    """Adds ``session_id`` (from the task context) and ``worker`` to log records."""

    def __init__(self, worker: int | None = None) -> None:
        super().__init__()
        self.worker = worker

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "session_id", None) is None:
            record.session_id = _session_id.get()
        if self.worker is not None and getattr(record, "worker", None) is None:
            record.worker = self.worker
        return True


_STANDARD_ATTRS = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonLogFormatter(logging.Formatter):
    """One JSON object per line: ``ts``, ``level``, ``logger``, ``msg``, ``session_id``,
    ``worker``, ``pid`` and the record's ``extra`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
            "pid": record.process,
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and not key.startswith("_") and value is not None:
                data[key] = value if isinstance(value, str | int | float | bool) else str(value)
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        sid = getattr(record, "session_id", None)
        worker = getattr(record, "worker", None)
        prefix = "" if worker is None else f"[w{worker}] "
        return f"{prefix}{text}" + (f" [session={sid}]" if sid else "")


def configure_logging(
    *, level: int = logging.INFO, fmt: str = "text", worker: int | None = None
) -> None:
    """Configure the root logger for ``van serve`` (``fmt``: ``"text"`` or ``"json"``)."""
    handler = logging.StreamHandler()
    handler.addFilter(SessionLogFilter(worker))
    if fmt == "json":
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(_TextFormatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    for old in list(root.handlers):
        root.removeHandler(old)
    root.addHandler(handler)
    root.setLevel(level)


# ------------------------------------------------------------------------ metrics
@dataclass
class Histogram:
    """A cumulative Prometheus histogram."""

    buckets: Sequence[float]
    counts: list[int] = field(default_factory=list)
    total: float = 0.0
    count: int = 0

    def __post_init__(self) -> None:
        self.buckets = tuple(sorted(self.buckets))
        self.counts = [0] * len(self.buckets)

    def observe(self, value: float) -> None:
        if value is None or math.isnan(value):
            return
        self.count += 1
        self.total += value
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[i] += 1

    def render(self, name: str, labels: str) -> list[str]:
        sep = "," if labels else ""
        lines = [
            f'{name}_bucket{{{labels}{sep}le="{_num(b)}"}} {c}'
            for b, c in zip(self.buckets, self.counts, strict=True)
        ]
        lines.append(f'{name}_bucket{{{labels}{sep}le="+Inf"}} {self.count}')
        lines.append(f"{name}_sum{{{labels}}} {_num(self.total)}")
        lines.append(f"{name}_count{{{labels}}} {self.count}")
        return lines


def _num(value: float) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return repr(float(value))


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


class ServeState:
    """Health, readiness, admission and metrics of one serving process.

    Args:
        protocol: served protocol (a metrics label).
        pools: the engine pools of this process.
        max_sessions: per-process session limit (``None``: unlimited).
        worker: worker index (a metrics label and a log field).
        active: returns the number of live sessions (default: the sessions counted with
            :meth:`session_started` / :meth:`session_ended`).
    """

    def __init__(
        self,
        protocol: str,
        *,
        pools: Sequence[EnginePool] = (),
        max_sessions: int | None = None,
        worker: int | None = None,
        active: Callable[[], int] | None = None,
    ) -> None:
        self.protocol = protocol
        self.pools = list(pools)
        self.max_sessions = max_sessions
        self.worker = worker
        self._active_fn = active
        self._active = 0
        self.started = False
        self.draining = False
        self.started_at = now()
        self.sessions_total = 0
        self.errors_total = 0
        self.rejected: dict[str, int] = {}
        self.turns_total = 0
        self.v2v = Histogram(V2V_BUCKETS)
        self.connect: dict[bool, Histogram] = {
            True: Histogram(CONNECT_BUCKETS),
            False: Histogram(CONNECT_BUCKETS),
        }

    # ----------------------------------------------------------------- admission
    @property
    def active(self) -> int:
        """Live sessions."""
        return self._active_fn() if self._active_fn is not None else self._active

    def refusal(self) -> tuple[HTTPStatus, str, str] | None:
        """Why a new session would be refused now (``None``: accept it)."""
        if self.draining:
            return HTTPStatus.SERVICE_UNAVAILABLE, "draining", "The server is shutting down."
        if self.max_sessions is not None and self.active >= self.max_sessions:
            return (
                HTTPStatus.SERVICE_UNAVAILABLE,
                "busy",
                f"The server already runs {self.max_sessions} sessions; try again later.",
            )
        return None

    def reject(self, reason: str) -> None:
        """Count a refused session (``reason``: ``busy``, ``draining``...)."""
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def session_started(self) -> None:
        self.sessions_total += 1
        self._active += 1

    def session_ended(self, *, error: bool = False) -> None:
        self._active = max(0, self._active - 1)
        if error:
            self.errors_total += 1

    def observe_turn(self, metrics: Any) -> None:
        """Record a :class:`~voice_agent_next.metrics.TurnMetrics`."""
        self.turns_total += 1
        v2v = getattr(metrics, "voice_to_voice", None)
        if v2v is not None:
            self.v2v.observe(v2v)

    def observe_connect(self, seconds: float, prewarmed: bool) -> None:
        """Record a session's first engine connection (``prewarmed``: from the pool)."""
        self.connect[prewarmed].observe(seconds)

    # ------------------------------------------------------------- health checks
    @property
    def ready(self) -> bool:
        return not self.readiness_problems()

    def readiness_problems(self) -> list[str]:
        problems: list[str] = []
        if not self.started:
            problems.append("starting")
        if self.draining:
            problems.append("draining")
        for pool in self.pools:
            if not pool.warm:
                problems.append(f"pool {pool.name}: warming up")
            elif not pool.ready:
                problems.append(f"pool {pool.name}: no prewarmed engine available")
        if self.max_sessions is not None and self.active >= self.max_sessions:
            problems.append("session limit reached")
        return problems

    def health(self) -> dict[str, Any]:
        """The ``GET /health`` document (liveness)."""
        doc: dict[str, Any] = {
            "status": "draining" if self.draining else "ok",
            "protocol": self.protocol,
            "pid": os.getpid(),
            "sessions": self.active,
            "max_sessions": self.max_sessions,
            "uptime": round(now() - self.started_at, 3),
        }
        if self.worker is not None:
            doc["worker"] = self.worker
        return doc

    def readiness(self) -> dict[str, Any]:
        """The ``GET /ready`` document."""
        problems = self.readiness_problems()
        return {
            "ready": not problems,
            "reasons": problems,
            "sessions": self.active,
            "max_sessions": self.max_sessions,
            "pools": {
                p.name: {"size": p.size, "idle": p.idle, "leased": p.leased, "warm": p.warm}
                for p in self.pools
            },
        }

    def http(self, path: str) -> tuple[HTTPStatus, str, bytes] | None:
        """``(status, content type, body)`` for the ops routes; ``None`` for other paths."""
        path = path.split("?", 1)[0].rstrip("/") or "/"
        if path in ("/health", "/healthz", "/livez"):
            return HTTPStatus.OK, "application/json", _json(self.health())
        if path in ("/ready", "/readyz"):
            doc = self.readiness()
            status = HTTPStatus.OK if doc["ready"] else HTTPStatus.SERVICE_UNAVAILABLE
            return status, "application/json", _json(doc)
        if path == "/metrics":
            return (
                HTTPStatus.OK,
                "text/plain; version=0.0.4; charset=utf-8",
                self.metrics().encode(),
            )
        return None

    # ------------------------------------------------------------------ metrics
    def metrics(self) -> str:
        """Prometheus text exposition format (version 0.0.4)."""
        base = [f'protocol="{_escape(self.protocol)}"']
        if self.worker is not None:
            base.append(f'worker="{self.worker}"')
        labels = ",".join(base)
        out: list[str] = []

        def metric(name: str, kind: str, help_: str, samples: list[tuple[str, float]]) -> None:
            out.append(f"# HELP {name} {help_}")
            out.append(f"# TYPE {name} {kind}")
            for extra, value in samples:
                lbl = ",".join(x for x in (labels, extra) if x)
                out.append(f"{name}{{{lbl}}} {_num(value)}")

        metric("van_up", "gauge", "1 while the process serves.", [("", 1)])
        metric("van_ready", "gauge", "1 when a new session would be served warm.",
               [("", int(self.ready))])  # fmt: skip
        metric("van_draining", "gauge", "1 while draining for shutdown.",
               [("", int(self.draining))])  # fmt: skip
        metric("van_uptime_seconds", "gauge", "Seconds since the process started serving.",
               [("", round(now() - self.started_at, 3))])  # fmt: skip
        metric("van_sessions_active", "gauge", "Live sessions.", [("", self.active)])
        metric("van_sessions_max", "gauge", "Session limit of the process (0: unlimited).",
               [("", self.max_sessions or 0)])  # fmt: skip
        metric("van_sessions_total", "counter", "Sessions started.", [("", self.sessions_total)])
        metric("van_sessions_rejected_total", "counter", "Sessions refused, by reason.",
               [(f'reason="{_escape(r)}"', n) for r, n in sorted(self.rejected.items())]
               or [('reason="busy"', 0)])  # fmt: skip
        metric("van_session_errors_total", "counter", "Sessions that ended with an error.",
               [("", self.errors_total)])  # fmt: skip
        metric("van_turns_total", "counter", "Agent turns completed.", [("", self.turns_total)])
        pools = self.pools
        pool_samples = [(f'pool="{_escape(p.name)}"', p) for p in pools]
        metric("van_pool_size", "gauge", "Target number of prewarmed engines.",
               [(lbl, p.size) for lbl, p in pool_samples])  # fmt: skip
        metric("van_pool_idle", "gauge", "Prewarmed engines ready for the next session.",
               [(lbl, p.idle) for lbl, p in pool_samples])  # fmt: skip
        metric("van_pool_leased", "gauge", "Engines used by live sessions.",
               [(lbl, p.leased) for lbl, p in pool_samples])  # fmt: skip
        stats = [(lbl, p.stats()) for lbl, p in pool_samples]
        metric("van_pool_hits_total", "counter", "Sessions served from a prewarmed engine.",
               [(lbl, s.hits) for lbl, s in stats])  # fmt: skip
        metric("van_pool_misses_total", "counter", "Sessions that started cold.",
               [(lbl, s.misses) for lbl, s in stats])  # fmt: skip
        metric("van_pool_errors_total", "counter", "Failed prewarm attempts.",
               [(lbl, s.errors) for lbl, s in stats])  # fmt: skip
        metric("van_pool_warmup_seconds", "gauge", "Duration of the initial warm-up.",
               [(lbl, round(s.warmup_seconds, 6)) for lbl, s in stats
                if s.warmup_seconds is not None])  # fmt: skip
        name = "van_voice_to_voice_seconds"
        out.append(f"# HELP {name} User stopped speaking -> first agent audio (TurnMetrics).")
        out.append(f"# TYPE {name} histogram")
        out.extend(self.v2v.render(name, labels))
        name = "van_engine_connect_seconds"
        out.append(f"# HELP {name} A session's first engine connection, by prewarmed.")
        out.append(f"# TYPE {name} histogram")
        for prewarmed, hist in self.connect.items():
            out.extend(hist.render(name, f'{labels},prewarmed="{str(prewarmed).lower()}"'))
        return "\n".join(out) + "\n"


def _json(doc: Any) -> bytes:
    return (json.dumps(doc) + "\n").encode()
