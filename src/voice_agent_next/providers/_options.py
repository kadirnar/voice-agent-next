"""Shared handling of constructor options whose names are standardized across providers.

Standard names: ``model``, ``language``, ``device``, ``base_url``, ``timeout`` (one HTTP
request), ``connect_timeout``, ``extra`` (extra request-body fields), ``extra_params``
(extra URL query parameters), ``extra_config`` (extra session-configuration fields).
Older provider-specific names keep working through :func:`renamed` with a
:class:`DeprecationWarning`.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, TypeVar

from ..errors import ConfigurationError

__all__ = ["deprecated", "onnx_providers", "renamed"]

T = TypeVar("T")

_ONNX_DEVICES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "gpu": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "mps": "CoreMLExecutionProvider",
    "directml": "DmlExecutionProvider",
    "dml": "DmlExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
}


def deprecated(owner: str, new: str, old: str, value: T, *, stacklevel: int = 3) -> T:
    """Warn that option ``old`` of ``owner`` is deprecated in favour of ``new``; returns
    ``value``."""
    warnings.warn(
        f"{owner}({old}=...) is deprecated, use {new}=...",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
    return value


def renamed(
    owner: str,
    new: str,
    new_value: T | None,
    old: str,
    old_value: T | None,
    *,
    stacklevel: int = 3,
) -> T | None:
    """The value of option ``new``, also accepting its deprecated name ``old``.

    Warns (:class:`DeprecationWarning`) when ``old`` is used, and raises
    :class:`ConfigurationError` when both are given different values.
    """
    if old_value is None:
        return new_value
    deprecated(owner, new, old, old_value, stacklevel=stacklevel + 1)
    if new_value is not None and new_value != old_value:
        raise ConfigurationError(f"{owner}: pass {new}=... or the deprecated {old}=..., not both")
    return old_value


def onnx_providers(device: str | Sequence[Any] | None) -> list[Any] | None:
    """ONNX Runtime execution providers for a ``device`` option.

    ``None`` / ``"auto"``: ``None`` (the provider's own choice). A device name (``"cpu"``,
    ``"cuda"``, ``"coreml"``, ``"directml"``...) or an execution provider name
    (``"CUDAExecutionProvider"``) gives that provider; a sequence is taken as the list of
    execution providers (entries may be ``(name, options)`` tuples).
    """
    if device is None:
        return None
    if isinstance(device, str):
        name = device.strip()
        if not name or name.lower() == "auto":
            return None
        if name.endswith("ExecutionProvider"):
            return [name]
        key = name.lower().split(":")[0]  # "cuda:0" -> cuda
        if key not in _ONNX_DEVICES:
            raise ConfigurationError(
                f"unknown device {device!r}; use 'auto', one of {sorted(_ONNX_DEVICES)} or "
                "an ONNX Runtime execution provider name"
            )
        return [_ONNX_DEVICES[key]]
    return list(device)
