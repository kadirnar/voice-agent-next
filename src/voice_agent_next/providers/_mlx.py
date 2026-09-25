"""Plumbing shared by the MLX providers (``mlx``, ``mlx_whisper``, ``mlx_audio``).

MLX runs on the GPU of Apple silicon Macs through Metal. Two things shape these providers:

* **One thread.** MLX keeps its default GPU stream per thread, arrays and compiled graphs
  are not meant to be shared across threads, and the recognizer, the synthesizer and a
  concurrent session all compete for the same GPU anyway. Every MLX call of every MLX
  provider in the process therefore runs on :data:`WORKER`, a single worker thread, in
  small steps (one audio chunk, one synthesized segment), so that speech recognition and
  synthesis interleave instead of waiting for each other's whole request.
* **Hugging Face repositories.** Models are MLX conversions on the Hub
  (``mlx-community/...``). :func:`snapshot` fetches exactly the files the model manager
  (``van models``) registers, and honours ``VAN_OFFLINE``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import platform
import sys
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any, TypeVar

import httpx

from ..errors import (
    AuthenticationError,
    ConfigurationError,
    MissingDependencyError,
    ProviderConnectionError,
    ProviderError,
    RateLimitError,
    VoiceAgentError,
)
from ..utils.deps import require
from ..utils.env import is_offline

__all__ = [
    "PLATFORMS",
    "WORKER",
    "MLXWorker",
    "ensure_available",
    "import_mlx",
    "is_apple_silicon",
    "map_error",
    "offline",
    "snapshot",
]

T = TypeVar("T")

PLATFORMS = ("darwin",)
"""``platforms`` of every MLX provider: MLX's Metal backend exists on macOS only."""


def is_apple_silicon() -> bool:
    """macOS on arm64 (a Python running under Rosetta 2 reports x86_64)."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


class MLXWorker:
    """A lazily started single worker thread that runs every MLX call of the process."""

    def __init__(self, name: str = "mlx") -> None:
        self._name = name
        self._executor: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()

    def _get(self) -> ThreadPoolExecutor:
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=self._name)
            return self._executor

    async def run(self, fn: Callable[..., T], *args: Any) -> T:
        """Run ``fn(*args)`` on the worker thread and await its result."""
        return await asyncio.get_running_loop().run_in_executor(self._get(), fn, *args)

    def submit(self, fn: Callable[..., Any], *args: Any) -> None:
        """Queue ``fn(*args)`` without waiting (cleanup from a cancelled task)."""
        self._get().submit(fn, *args)

    def close(self) -> None:
        """Stop the thread once queued work is done (a later call starts a new one)."""
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False)


WORKER = MLXWorker()
"""The process-wide MLX thread."""


def _importable(module: str) -> bool:
    if module in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def ensure_available(module: str, *, extra: str, package: str, provider: str) -> None:
    """Fail fast (in the constructor, without importing MLX) when ``module`` is missing."""
    if _importable(module):
        return
    if not is_apple_silicon():
        where = "under Rosetta 2" if sys.platform == "darwin" else f"on {sys.platform}"
        raise MissingDependencyError(
            f"{provider}: MLX runs on Apple silicon Macs only (macOS arm64); this Python runs "
            f"{where}. Pick another provider (`van providers`), or on a Mac with an arm64 "
            f"Python: pip install 'voice-agent-next[{extra}]'"
        )
    require(module, extra=extra, package=package)


def offline(local_files_only: bool = False) -> bool:
    return local_files_only or is_offline()


def snapshot(
    repo_or_path: str,
    *,
    patterns: Sequence[str] | None,
    local_files_only: bool,
    provider: str,
    revision: str | None = None,
) -> str:
    """Local directory of a model: ``repo_or_path`` itself if it is a directory, else the
    Hugging Face snapshot with ``patterns`` (downloaded on first use)."""
    if os.path.isdir(repo_or_path):
        return repo_or_path
    hub = require("huggingface_hub", extra="mlx", package="huggingface-hub")
    try:
        return str(
            Path(
                hub.snapshot_download(
                    repo_or_path,
                    revision=revision,
                    allow_patterns=list(patterns) if patterns else None,
                    local_files_only=offline(local_files_only),
                )
            )
        )
    except Exception as exc:
        raise map_error(exc, provider, f"downloading model {repo_or_path!r}") from exc


def map_error(exc: BaseException, provider: str, action: str) -> VoiceAgentError:
    """Translate MLX / Hugging Face Hub failures to library errors.

    Hub exceptions are matched by class name so that neither this module nor its tests
    import ``huggingface_hub``.
    """
    if isinstance(exc, VoiceAgentError):
        return exc
    message = f"{provider} {action} failed: {exc}"
    names = {cls.__name__ for cls in type(exc).__mro__}
    status = getattr(getattr(exc, "response", None), "status_code", None)
    status = status if isinstance(status, int) else None
    if "GatedRepoError" in names:
        return AuthenticationError(message, provider=provider, status_code=status)
    if names & {"RepositoryNotFoundError", "RevisionNotFoundError", "EntryNotFoundError"}:
        return ConfigurationError(message)  # unknown model id or revision
    if status in (401, 403):
        return AuthenticationError(message, provider=provider, status_code=status)
    if status == 429:
        return RateLimitError(message, provider=provider, status_code=status)
    if names & {"HfHubHTTPError", "LocalEntryNotFoundError", "OfflineModeIsEnabled"} or isinstance(
        exc, (ConnectionError, TimeoutError, httpx.TransportError)
    ):
        return ProviderConnectionError(message, provider=provider, status_code=status)
    if isinstance(exc, (ValueError, TypeError, FileNotFoundError)):
        return ConfigurationError(message)  # unsupported model type, bad option or voice
    return ProviderError(message, provider=provider, status_code=status)


def import_mlx(provider: str) -> ModuleType:
    """``mlx.core`` (raises :class:`MissingDependencyError` with the install hint)."""
    ensure_available("mlx", extra="mlx", package="mlx", provider=provider)
    return require("mlx.core", extra="mlx", package="mlx")
