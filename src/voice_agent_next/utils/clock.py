"""Monotonic high-resolution clock used for every timestamp in the library.

``time.monotonic()`` has ~15.6 ms resolution on Windows, which is too coarse for
latency metrics, so all timestamps use :func:`time.perf_counter` instead.
"""

from __future__ import annotations

import time

__all__ = ["now"]


def now() -> float:
    """Seconds from an arbitrary, monotonic, high-resolution origin (``perf_counter``)."""
    return time.perf_counter()
