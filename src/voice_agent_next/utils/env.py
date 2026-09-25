"""Environment switches read in more than one place."""

from __future__ import annotations

import os

__all__ = ["OFFLINE_VARS", "env_flag", "is_offline"]

OFFLINE_VARS = ("VAN_OFFLINE", "HF_HUB_OFFLINE")
"""Either one set to a true value forbids downloads (models, datasets)."""

_TRUE = frozenset({"1", "true", "yes", "on"})


def env_flag(name: str) -> bool:
    """Whether the environment variable ``name`` is set to 1/true/yes/on (any case)."""
    return os.environ.get(name, "").strip().lower() in _TRUE


def is_offline() -> bool:
    """Offline mode: ``VAN_OFFLINE`` or ``HF_HUB_OFFLINE`` is set. Nothing may be
    downloaded; cached files are used, and a missing one is an error."""
    return any(env_flag(name) for name in OFFLINE_VARS)
