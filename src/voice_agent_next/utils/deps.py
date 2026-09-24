"""Helpers for optional dependencies.

Provider modules must be importable without their optional dependencies installed
(so ``van providers`` can list everything). Heavy/optional packages are imported lazily
with :func:`require`, which raises a helpful :class:`MissingDependencyError`.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

from ..errors import MissingDependencyError

__all__ = ["is_installed", "require"]


def is_installed(module: str) -> bool:
    """Return True if ``module`` can be imported (without importing it)."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def require(module: str, *, extra: str | None = None, package: str | None = None) -> ModuleType:
    """Import and return ``module`` or raise :class:`MissingDependencyError`.

    Args:
        module: dotted module name to import, e.g. ``"onnxruntime"``.
        extra: the voice-agent-next extra that provides it, e.g. ``"silero"``.
        package: the pip distribution name if it differs from the module name.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        pkg = package or module.split(".")[0]
        if extra:
            hint = f"pip install 'voice-agent-next[{extra}]'  (or: pip install {pkg})"
        else:
            hint = f"pip install {pkg}"
        raise MissingDependencyError(
            f"Optional dependency '{pkg}' is required for this feature but is not installed. "
            f"Install it with: {hint}"
        ) from exc
