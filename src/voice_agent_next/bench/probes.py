"""Runtime probes for the framework-overhead track: event-loop lag, CPU time and memory.

Everything here is standard library only (no ``psutil``) and works on Linux, macOS and
Windows:

* :class:`LoopLagProbe` — a task that sleeps for a fixed interval and records how late
  it wakes up (*event-loop lag*: time the loop spent on other callbacks, plus the
  platform's timer granularity — up to ~16 ms on Windows with Python < 3.13). It also
  samples the resident set size, so a run knows its memory high-water mark;
* :func:`cpu_seconds` — CPU time of the whole process (all threads, user + system);
* :func:`rss_bytes` — resident set size: the *current* value on Linux
  (``/proc/self/statm``) and Windows (``GetProcessMemoryInfo``), the *peak*
  (``getrusage``) elsewhere; :func:`rss_kind` says which one.
"""

from __future__ import annotations

import asyncio
import ctypes
import os
import sys
import time
from dataclasses import dataclass, field
from types import TracebackType
from typing import Literal

from ..utils.aio import cancel_and_wait
from ..utils.clock import now

__all__ = ["LoopLagProbe", "cpu_seconds", "rss_bytes", "rss_kind"]

RSSKind = Literal["current", "peak", "unavailable"]


def cpu_seconds() -> float:
    """CPU time (user + system) consumed by this process so far, all threads included."""
    return time.process_time()


def _rss_linux() -> int | None:
    try:
        with open("/proc/self/statm", encoding="ascii") as f:
            resident_pages = int(f.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def _rss_windows() -> int | None:
    if sys.platform != "win32":  # keeps mypy's platform narrowing happy
        return None
    from ctypes import wintypes

    class _MemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_process = kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE  # a pseudo-handle: pointer-sized
        get_info = kernel32.K32GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(_MemoryCounters), wintypes.DWORD]
        get_info.restype = wintypes.BOOL
        counters = _MemoryCounters()
        counters.cb = ctypes.sizeof(_MemoryCounters)
        if get_info(get_process(), ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize)
    except (AttributeError, OSError):
        return None
    return None


def _rss_peak() -> int | None:
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024  # bytes vs KiB


def rss_kind() -> RSSKind:
    """What :func:`rss_bytes` returns on this platform."""
    if sys.platform.startswith("linux") or sys.platform == "win32":
        return "current"
    return "peak" if _rss_peak() is not None else "unavailable"


def rss_bytes() -> int | None:
    """Resident set size of this process in bytes (see :func:`rss_kind`); ``None`` if
    the platform offers no way to read it."""
    if sys.platform.startswith("linux"):
        value = _rss_linux()
        return value if value is not None else _rss_peak()
    if sys.platform == "win32":
        return _rss_windows()
    return _rss_peak()


@dataclass
class LoopLagProbe:
    """Measures event-loop lag while it runs (``async with LoopLagProbe() as probe:``).

    Every ``interval`` seconds the probe task wakes from ``asyncio.sleep`` and records
    ``max(0, woke - due)``: how long the loop was busy with other callbacks when the timer
    became due, plus timer granularity. Every ``rss_every`` wake-ups it samples
    :func:`rss_bytes` to track the memory high-water mark (``rss_peak``).
    """

    interval: float = 0.01
    rss_every: int = 5
    samples: list[float] = field(default_factory=list)
    """Lag of every wake-up (seconds)."""
    rss_peak: int | None = None
    """Highest RSS sampled while running (bytes)."""
    _task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ValueError("interval must be > 0")
        if self.rss_every < 1:
            raise ValueError("rss_every must be >= 1")

    def _sample_rss(self) -> None:
        rss = rss_bytes()
        if rss is not None and (self.rss_peak is None or rss > self.rss_peak):
            self.rss_peak = rss

    async def _run(self) -> None:
        wakeups = 0
        while True:
            due = now() + self.interval
            await asyncio.sleep(self.interval)
            self.samples.append(max(0.0, now() - due))
            wakeups += 1
            if wakeups % self.rss_every == 0:
                self._sample_rss()

    def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("probe already started")
        self._sample_rss()
        self._task = asyncio.create_task(self._run(), name="bench-loop-lag-probe")

    async def stop(self) -> None:
        await cancel_and_wait(self._task)
        self._sample_rss()

    def mark(self) -> int:
        """Index of the next sample: ``samples[mark:]`` are the samples taken after it."""
        return len(self.samples)

    async def __aenter__(self) -> LoopLagProbe:
        self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.stop()
