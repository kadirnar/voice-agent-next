"""Silero VAD v6 on ONNX Runtime: the default production VAD, without torch.

The official Silero VAD ONNX model (MIT, 2.3 MB) is downloaded once into the shared
model cache (:mod:`voice_agent_next.utils.download`) from a pinned URL and checked
against a pinned SHA-256. After that it loads offline (``VAN_OFFLINE=1``). The
``silero-vad`` pip package is not used because it pulls in torch.

The model is fed exactly like ``OnnxWrapper`` in the reference implementation
(snakers4/silero-vad, ``src/silero_vad/utils_vad.py``). Each 32 ms window (512 samples
at 16 kHz, 256 at 8 kHz) is prefixed with the last 64 (32) samples of the previous
window. A ``(2, 1, 128)`` recurrent state carries over from one window to the next.
Both live in the per-stream inference object. All streams of one :class:`SileroVAD`
share a single ONNX Runtime session.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ..errors import ConfigurationError, ProviderError
from ..registry import register_provider
from ..utils.clock import now
from ..utils.deps import require
from ..utils.download import download
from ..utils.log import logger
from ..vad import VAD, VADOptions

__all__ = ["SileroVAD"]


@dataclass(frozen=True, slots=True)
class _ModelFile:
    url: str
    sha256: str
    filename: str
    """Name in the model cache; versioned so that model versions never overwrite each other."""


_MODELS: dict[str, _ModelFile] = {
    # silero_vad.onnx (opset 16, 8 and 16 kHz) of release v6.2, unchanged up to v6.2.3.
    "v6.2": _ModelFile(
        url="https://raw.githubusercontent.com/snakers4/silero-vad/v6.2/src/silero_vad/data/silero_vad.onnx",
        sha256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
        filename="silero_vad_v6.2.onnx",
    ),
}
_DEFAULT_MODEL = "v6.2"

_WINDOW_SAMPLES = {16_000: 512, 8_000: 256}
"""Samples per inference window (32 ms): the only window sizes the model accepts."""
_CONTEXT_SAMPLES = {16_000: 64, 8_000: 32}
"""Samples of the previous window prepended to each window, as in ``OnnxWrapper``."""
_STATE_SHAPE = (2, 1, 128)
_INPUT_NAMES = frozenset({"input", "state", "sr"})


class _SileroInference:
    """Per-stream model state: the recurrent state and the previous window's context."""

    def __init__(self, session: Any, sample_rate: int) -> None:
        self._session = session
        self._window = _WINDOW_SAMPLES[sample_rate]
        self._context = _CONTEXT_SAMPLES[sample_rate]
        self._sr = np.array(sample_rate, dtype=np.int64)
        # The model input is [context | window]; the buffer is reused for every window.
        self._input = np.zeros((1, self._context + self._window), dtype=np.float32)
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)

    def reset(self) -> None:
        self._input.fill(0.0)
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)

    def __call__(self, window: npt.NDArray[np.float32]) -> float:
        if window.shape != (self._window,):
            raise ValueError(
                f"Silero VAD expects windows of {self._window} samples, got shape {window.shape}"
            )
        x = self._input
        x[0, self._context :] = window
        feeds = {"input": x, "state": self._state, "sr": self._sr}
        try:
            prob, state = self._session.run(None, feeds)
        except Exception as exc:  # onnxruntime errors derive from Exception only
            raise ProviderError(f"Silero VAD inference failed: {exc}", provider="silero") from exc
        self._state = state
        x[0, : self._context] = x[0, -self._context :]  # context for the next window
        return float(np.ravel(prob)[0])


def _merge_options(options: VADOptions | None, overrides: dict[str, float]) -> VADOptions:
    opts = options or VADOptions()
    if not overrides:
        return opts
    known = {f.name for f in dataclasses.fields(VADOptions)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ConfigurationError(
            f"unknown Silero VAD option(s) {unknown}; valid options: {sorted(known)}"
        )
    return dataclasses.replace(opts, **overrides)


@register_provider(
    "vad",
    "silero",
    description="Silero VAD v6 on ONNX Runtime (8/16 kHz, CPU, no torch) — default local VAD",
    default_model=_DEFAULT_MODEL,
    models=tuple(_MODELS),
    extra="silero",
    requires=("onnxruntime",),
    local=True,
)
class SileroVAD(VAD):
    """Silero VAD v6 running on ONNX Runtime.

    The model is downloaded on first use (2.3 MB, pinned URL and SHA-256) and loaded
    lazily: :meth:`warmup` does both in a worker thread, otherwise the first
    :meth:`stream` does them synchronously. One ONNX Runtime session is shared by all
    streams of this instance (1 intra-op and 1 inter-op thread, spin-waiting off); each
    stream keeps its own recurrent state and audio context. Inference takes about
    0.1 ms per 32 ms window on one CPU core.

    Args:
        model: model id. Only ``"v6.2"`` (the default) is known today.
        sample_rate: ``16000`` (512-sample windows) or ``8000`` (256-sample windows).
            Input audio at any rate or channel count is converted by the stream.
        model_path: local Silero v5/v6 ONNX file to use instead of downloading the
            pinned model. ``model`` is then only a label.
        force_cpu: use ONNX Runtime's CPU execution provider even when an accelerator
            is available. Keep it on: the model is tiny and runs with batch size 1.
        options: thresholds and durations (:class:`~voice_agent_next.vad.VADOptions`).
        **option_overrides: individual ``VADOptions`` fields, applied on top of
            ``options``, e.g. ``min_silence_duration=0.3``.

    Raises:
        ConfigurationError: unsupported ``sample_rate``, unknown ``model`` or option,
            missing ``model_path`` file, or a file that is not a Silero VAD model.
        MissingDependencyError: ``onnxruntime`` is not installed (extra ``silero``).
        DownloadError: the model is not cached and cannot be downloaded (for example
            with ``VAN_OFFLINE=1``); raised when the model is first loaded.
        ProviderError: ONNX Runtime fails to load the model or to run it.
    """

    provider = "silero"

    def __init__(
        self,
        *,
        model: str | None = None,
        sample_rate: int = 16_000,
        model_path: str | os.PathLike[str] | None = None,
        force_cpu: bool = True,
        options: VADOptions | None = None,
        **option_overrides: float,
    ) -> None:
        if sample_rate not in _WINDOW_SAMPLES:
            raise ConfigurationError(
                f"Silero VAD runs at 8000 or 16000 Hz, got sample_rate={sample_rate} "
                "(input audio at any rate is resampled automatically)"
            )
        model = model or _DEFAULT_MODEL
        if model_path is None and model not in _MODELS:
            raise ConfigurationError(
                f"unknown Silero VAD model {model!r}; known models: {', '.join(_MODELS)} "
                "(or pass model_path=... to use a local ONNX file)"
            )
        super().__init__(
            sample_rate=sample_rate,
            window_samples=_WINDOW_SAMPLES[sample_rate],
            options=_merge_options(options, option_overrides),
            model=model,
        )
        self._ort = require("onnxruntime", extra="silero")
        self.model_path = Path(model_path) if model_path is not None else None
        self.force_cpu = force_cpu
        self._session: Any = None
        self._session_lock = threading.Lock()

    def _new_inference(self) -> _SileroInference:
        return _SileroInference(self._get_session(), self.sample_rate)

    async def warmup(self) -> None:
        """Download (first run only) and load the model off the event loop, then run it once."""
        await asyncio.to_thread(self._warmup)

    async def aclose(self) -> None:
        """Drop the shared session; streams that are still open keep working."""
        self._session = None

    # ----------------------------------------------------------------- internals
    def _warmup(self) -> None:
        self._new_inference()(np.zeros(self.window_samples, dtype=np.float32))

    def _get_session(self) -> Any:
        """The shared ONNX Runtime session, created on first use (may download: blocking)."""
        with self._session_lock:
            if self._session is None:
                self._session = self._load_session(self._resolve_model_path())
            return self._session

    def _resolve_model_path(self) -> Path:
        if self.model_path is not None:
            if not self.model_path.is_file():
                raise ConfigurationError(f"Silero VAD model file not found: {self.model_path}")
            return self.model_path
        spec = _MODELS[self.model]
        return download(spec.url, filename=spec.filename, subdir="silero", sha256=spec.sha256)

    def _load_session(self, path: Path) -> Any:
        ort = self._ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        # Idle ONNX Runtime threads spin-wait for work by default, burning CPU between windows.
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        available = list(ort.get_available_providers())
        if self.force_cpu and "CPUExecutionProvider" in available:
            providers = ["CPUExecutionProvider"]
        else:
            providers = available  # older onnxruntime requires an explicit list
        t0 = now()
        try:
            session = ort.InferenceSession(os.fspath(path), sess_options=opts, providers=providers)
        except Exception as exc:  # onnxruntime errors derive from Exception only
            raise ProviderError(
                f"failed to load Silero VAD model {path}: {exc}", provider=self.provider
            ) from exc
        names = {i.name for i in session.get_inputs()}
        if _INPUT_NAMES - names:
            raise ConfigurationError(
                f"{path} is not a Silero VAD v5/v6 ONNX model: expected inputs "
                f"{sorted(_INPUT_NAMES)}, got {sorted(names)}"
            )
        logger.debug(
            "loaded Silero VAD model %s in %.1f ms (providers: %s)",
            path,
            (now() - t0) * 1e3,
            providers,
        )
        return session
