"""Internal utilities shared across the library."""

from __future__ import annotations

from .aio import BackgroundTasks, Chan, ChanClosed, cancel_and_wait, merge_async_iterators
from .clock import now
from .deps import is_installed, require
from .emitter import EventEmitter
from .ids import new_id
from .log import logger

__all__ = [
    "BackgroundTasks",
    "Chan",
    "ChanClosed",
    "EventEmitter",
    "cancel_and_wait",
    "is_installed",
    "logger",
    "merge_async_iterators",
    "new_id",
    "now",
    "require",
]
