"""Smart Turn v3 end-of-turn detector: local ONNX model, audio only, 23 languages.

`Smart Turn <https://github.com/pipecat-ai/smart-turn>`_ (Daily / Pipecat, BSD-2-Clause) is a
Whisper-Tiny encoder with a linear classification head (~8M parameters). It listens to the
user's current turn and returns the probability that the turn is *complete*, from prosody
and acoustic cues rather than from a transcript. It is meant to run when the VAD reports a
pause; the cascade does exactly that while endpointing.

Model contract (reference: ``inference.py`` and ``audio_utils.py`` in the Smart Turn repo):

* 16 kHz mono audio, exactly 8 s: longer turns keep their **last** 8 s, shorter ones are
  zero-padded at the **start** so that the speech ends the window;
* Whisper log-mel features as computed by ``WhisperFeatureExtractor(chunk_length=8)`` with
  ``do_normalize=True``: the waveform is normalized to zero mean / unit variance, then an
  80-bin Slaney mel spectrogram (400-sample periodic Hann window, 160-sample hop, centered
  with reflect padding, last frame dropped) is converted to log10, floored 8 decades below
  its maximum and scaled with ``(x + 4) / 4``;
* ONNX input ``input_features`` of shape ``(batch, 80, 800)`` (float32); the single output
  (``(batch, 1)``) is already a sigmoid probability.

:func:`log_mel_features` reimplements that feature extractor with numpy only (no torch or
transformers at runtime) and matches ``transformers`` to about 1e-6.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from ..audio.frame import AudioFrame
from ..audio.resample import resample
from ..chat import ChatContext
from ..errors import ConfigurationError, ProviderError
from ..registry import register_provider
from ..turn import TurnDetector
from ..utils.clock import now
from ..utils.deps import require
from ..utils.download import hf_file
from ..utils.log import logger

__all__ = [
    "DEFAULT_MODEL",
    "HF_REPO",
    "HF_REVISION",
    "LANGUAGES",
    "MODEL_SHA256",
    "SmartTurnDetector",
    "log_mel_features",
    "prepare_audio",
]

HF_REPO = "pipecat-ai/smart-turn-v3"
HF_REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
"""Pinned commit of :data:`HF_REPO` (2026-01-07, the Smart Turn v3.2 release)."""
DEFAULT_MODEL = "smart-turn-v3.2-cpu"
MODEL_SHA256: dict[str, str] = {
    "smart-turn-v3.2-cpu.onnx": "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f",
    "smart-turn-v3.2-gpu.onnx": "ab8dc64b88713f90b571c15b714bd1330e6c883cad8763dacf65c9376dc539be",
    "smart-turn-v3.1-cpu.onnx": "fb68d55c2d542ce79e44b12013bfd571e90df8594ab096d757198e851b0c6594",
    "smart-turn-v3.1-gpu.onnx": "a32f7445d5076029472b6c9f7a71005df576ea19d5f929021200f535b962af84",
    "smart-turn-v3.0.onnx": "07a133aba31e2d0b523f17f8c2e4e65efe6d8f685efd12ca4fe21ebf4e798991",
}
"""SHA-256 of the ONNX files at :data:`HF_REVISION`."""

SAMPLE_RATE = 16_000
WINDOW_SECONDS = 8
N_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS  # 128 000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 800

# ISO 639-1 code -> other spellings seen in the wild (ISO 639-2/3 codes, English names).
_LANGUAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "ar": ("ara", "arabic"),
    "bn": ("ben", "bengali", "bangla"),
    "da": ("dan", "danish"),
    "de": ("deu", "ger", "german"),
    "en": ("eng", "english"),
    "es": ("spa", "spanish"),
    "fi": ("fin", "finnish"),
    "fr": ("fra", "fre", "french"),
    "hi": ("hin", "hindi"),
    "id": ("ind", "indonesian"),
    "it": ("ita", "italian"),
    "ja": ("jpn", "japanese"),
    "ko": ("kor", "korean"),
    "mr": ("mar", "marathi"),
    "nl": ("nld", "dut", "dutch"),
    "no": ("nor", "nb", "nob", "nn", "nno", "norwegian"),
    "pl": ("pol", "polish"),
    "pt": ("por", "portuguese"),
    "ru": ("rus", "russian"),
    "tr": ("tur", "turkish"),
    "uk": ("ukr", "ukrainian"),
    "vi": ("vie", "vietnamese"),
    "zh": ("zho", "chi", "cmn", "chinese", "mandarin"),
}
LANGUAGES: tuple[str, ...] = tuple(_LANGUAGE_ALIASES)
"""The 23 languages Smart Turn v3.2 was trained and evaluated on (ISO 639-1)."""
_LANGUAGE_LOOKUP = {
    alias: code for code, aliases in _LANGUAGE_ALIASES.items() for alias in (code, *aliases)
}


def _language_code(language: str) -> str | None:
    """``"en-US"`` / ``"eng"`` / ``"English"`` -> ``"en"``; ``None`` if not supported."""
    tag = language.strip().lower().replace("_", "-")
    return _LANGUAGE_LOOKUP.get(tag) or _LANGUAGE_LOOKUP.get(tag.split("-")[0])


# ------------------------------------------------------------------------------ features
def _hz_to_mel(freq: float) -> float:
    """Slaney mel scale (linear below 1 kHz, logarithmic above)."""
    if freq >= 1000.0:
        return 15.0 + float(np.log(freq / 1000.0)) * (27.0 / float(np.log(6.4)))
    return 3.0 * freq / 200.0


def _mel_to_hz(mels: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    freqs = 200.0 * mels / 3.0
    log_region = mels >= 15.0
    freqs[log_region] = 1000.0 * np.exp((np.log(6.4) / 27.0) * (mels[log_region] - 15.0))
    return freqs


@cache
def _mel_filters() -> npt.NDArray[np.float64]:
    """Slaney-normalized triangular mel filters, shape ``(N_FFT // 2 + 1, N_MELS)``."""
    hz = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(SAMPLE_RATE / 2), N_MELS + 2))
    fft_freqs = np.linspace(0, SAMPLE_RATE // 2, N_FFT // 2 + 1)
    slopes = hz[np.newaxis, :] - fft_freqs[:, np.newaxis]
    widths = np.diff(hz)
    down = -slopes[:, :-2] / widths[:-1]
    up = slopes[:, 2:] / widths[1:]
    filters = np.maximum(0.0, np.minimum(down, up))
    filters *= 2.0 / (hz[2:] - hz[:-2])  # constant energy per band
    return filters


@cache
def _hann_window() -> npt.NDArray[np.float64]:
    return np.hanning(N_FFT + 1)[:-1]  # periodic, as torch.hann_window


def prepare_audio(samples: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Fit 16 kHz mono samples to the model window (exactly 8 s).

    Keeps the **last** 8 s of longer clips and zero-pads shorter ones at the **start**, so
    the end of the user's speech is always at the end of the window.
    """
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size >= N_SAMPLES:
        return np.ascontiguousarray(x[x.size - N_SAMPLES :])
    out = np.zeros(N_SAMPLES, dtype=np.float32)
    out[N_SAMPLES - x.size :] = x
    return out


def log_mel_features(window: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Whisper log-mel features of one model window, shape ``(80, 800)``, float32.

    Numpy port of ``WhisperFeatureExtractor(chunk_length=8)(window, sampling_rate=16000,
    padding="max_length", max_length=128000, truncation=True, do_normalize=True)``, which is
    how Smart Turn was trained and is run upstream.

    Args:
        window: exactly 128 000 samples (8 s at 16 kHz) in ``[-1, 1]``; see
            :func:`prepare_audio`.
    """
    x = np.asarray(window, dtype=np.float32).reshape(-1)
    if x.size != N_SAMPLES:
        raise ValueError(f"expected {N_SAMPLES} samples (8 s at 16 kHz), got {x.size}")
    # zero-mean / unit-variance waveform normalization, in float32 like the reference
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    padded = np.pad(x.astype(np.float64), N_FFT // 2, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP_LENGTH]
    spectrum = np.fft.rfft(frames * _hann_window(), axis=-1)  # (801, 201)
    power = spectrum.real**2 + spectrum.imag**2
    mel = np.maximum(power @ _mel_filters(), 1e-10)  # (801, 80)
    log_spec = np.log10(mel).T[:, :-1].astype(np.float32)  # drop the last frame -> (80, 800)
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


def _model_samples(audio: AudioFrame) -> npt.NDArray[np.float32]:
    """The last 8 s of ``audio`` as 16 kHz mono float32 (not yet padded)."""
    # a little extra context keeps the resampler's edge effects out of the window
    keep = WINDOW_SECONDS + (0.0 if audio.sample_rate == SAMPLE_RATE else 0.05)
    if audio.duration > keep:
        audio = audio.slice(audio.duration - keep)
    audio = audio.to_mono()
    if audio.sample_rate != SAMPLE_RATE:
        audio = resample(audio, SAMPLE_RATE)
    return audio.to_float32()


# ------------------------------------------------------------------------------ detector
@register_provider(
    "turn",
    "smart_turn",
    description="Smart Turn v3.2 audio end-of-turn model (local ONNX, 23 languages, BSD-2)",
    default_model=DEFAULT_MODEL,
    models=tuple(name.removesuffix(".onnx") for name in MODEL_SHA256),
    env=(),
    extra="smart-turn",
    requires=("onnxruntime",),
    local=True,
)
class SmartTurnDetector(TurnDetector):
    """Smart Turn v3 audio end-of-turn detector (ONNX Runtime, CPU by default).

    The model file is downloaded from Hugging Face on first use (``pipecat-ai/smart-turn-v3``
    at a pinned revision, checksum-verified) into the shared model cache; call
    :meth:`warmup` to load it before the first turn.

    Args:
        model: ``"smart-turn-v3.2-cpu"`` (default: int8, ~8 MB) or ``"smart-turn-v3.2-gpu"``
            (fp32, ~32 MB, about 1 point more accurate); older ``smart-turn-v3.1-*`` /
            ``smart-turn-v3.0`` files work too. The ``smart-turn-`` prefix may be omitted.
        model_path: local ``.onnx`` file (e.g. a fine-tuned model) used instead of
            downloading ``model``; the file name then serves as the model name.
        threshold: probability at/above which the turn counts as complete.
        revision: Hugging Face revision of :data:`HF_REPO` to download from.
        providers: ONNX Runtime execution providers, e.g.
            ``["CUDAExecutionProvider", "CPUExecutionProvider"]`` (needs onnxruntime-gpu).
        num_threads: intra-op threads. The default (1, no spinning) keeps a voice agent's
            CPU usage predictable; raise it for the fp32 model on large machines.
    """

    provider = "smart_turn"
    modality = "audio"
    languages: ClassVar[tuple[str, ...]] = LANGUAGES

    def __init__(
        self,
        *,
        model: str | None = None,
        model_path: str | os.PathLike[str] | None = None,
        threshold: float = 0.5,
        revision: str = HF_REVISION,
        providers: Sequence[str] | None = None,
        num_threads: int = 1,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ConfigurationError(f"threshold must be in [0, 1], got {threshold}")
        if num_threads < 1:
            raise ConfigurationError(f"num_threads must be >= 1, got {num_threads}")
        if model_path is not None:
            name = Path(model_path).stem
        else:
            name = (model or DEFAULT_MODEL).strip().removesuffix(".onnx")
            if not name.startswith("smart-turn-"):
                name = f"smart-turn-{name}"
        super().__init__(
            model=name,
            threshold=threshold,
            sample_rate=SAMPLE_RATE,
            max_audio_duration=float(WINDOW_SECONDS),
        )
        self.model_path = Path(model_path) if model_path is not None else None
        self.revision = revision
        self.providers: tuple[str, ...] = tuple(providers or ("CPUExecutionProvider",))
        self.num_threads = num_threads
        self._session: Any = None
        self._input_name = "input_features"
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ model
    def _model_file(self) -> Path:
        if self.model_path is not None:
            if not self.model_path.is_file():
                raise ConfigurationError(f"Smart Turn model file not found: {self.model_path}")
            return self.model_path
        filename = f"{self.model}.onnx"
        sha256 = MODEL_SHA256.get(filename) if self.revision == HF_REVISION else None
        return hf_file(HF_REPO, filename, revision=self.revision, sha256=sha256)

    def _load_session(self) -> Any:
        ort = require("onnxruntime", extra="smart-turn")
        path = self._model_file()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.num_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # idle worker threads must not busy-wait between the (rare) predictions
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        t0 = now()
        try:
            session = ort.InferenceSession(
                os.fspath(path), sess_options=opts, providers=list(self.providers)
            )
        except Exception as exc:  # onnxruntime errors derive from Exception only
            raise ProviderError(
                f"failed to load Smart Turn model {path}: {exc}", provider=self.provider
            ) from exc
        self._input_name = session.get_inputs()[0].name
        logger.debug(
            "loaded Smart Turn model %s in %.1f ms (providers: %s)",
            path,
            (now() - t0) * 1e3,
            self.providers,
        )
        return session

    def _get_session(self) -> Any:
        session = self._session
        if session is None:
            with self._lock:
                if self._session is None:
                    self._session = self._load_session()
                session = self._session
        return session

    # -------------------------------------------------------------- inference
    def _infer(self, audio: AudioFrame) -> float:
        """Blocking inference (worker thread): features + ONNX Runtime."""
        session = self._get_session()
        features = log_mel_features(prepare_audio(_model_samples(audio)))
        try:
            outputs = session.run(None, {self._input_name: features[np.newaxis]})
        except Exception as exc:
            raise ProviderError(
                f"Smart Turn inference failed: {exc}", provider=self.provider
            ) from exc
        return float(np.asarray(outputs[0], dtype=np.float32).reshape(-1)[0])

    async def _predict(self, *, audio: AudioFrame | None, chat_ctx: ChatContext | None) -> float:
        if audio is None or not audio:
            # nothing to listen to: report "complete" so silence-based endpointing decides
            return 1.0
        return await asyncio.to_thread(self._infer, audio)

    def supports_language(self, language: str | None) -> bool:
        """True for the 23 training languages (ISO 639-1/639-3 codes, BCP-47 tags or names)."""
        return not language or _language_code(language) is not None

    async def warmup(self) -> None:
        """Download (first run only) and load the model, then run one inference."""
        await asyncio.to_thread(self._infer, AudioFrame.silence(1.0, SAMPLE_RATE))

    async def aclose(self) -> None:
        self._session = None
