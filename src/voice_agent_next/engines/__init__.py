"""Engines built into the library (the cascade, the rotation wrapper). Native S2S engines
live in ``providers``."""

from __future__ import annotations

from .cascade import CascadeConnection, CascadeEngine, CascadeOptions
from .rotation import (
    HistoryCarryOver,
    RotatingConnection,
    RotatingEngine,
    RotationPolicy,
    SummarizeHistory,
    TruncateHistory,
)

__all__ = [
    "CascadeConnection",
    "CascadeEngine",
    "CascadeOptions",
    "HistoryCarryOver",
    "RotatingConnection",
    "RotatingEngine",
    "RotationPolicy",
    "SummarizeHistory",
    "TruncateHistory",
]
