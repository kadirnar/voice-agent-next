"""faster-whisper speech-to-text: Whisper on CTranslate2, running locally on CPU or CUDA.

``create("stt", "faster_whisper/large-v3-turbo")`` (alias: ``"whisper/small"``).

Whisper is a batch recognizer (``capabilities.streaming=False``). The cascade makes it
real-time by wrapping it in :class:`~voice_agent_next.stt.StreamAdapter`, which cuts the
input into utterances with the configured VAD and transcribes each one.

The model is downloaded into the Hugging Face cache on first use. It is loaded once,
lazily, in a worker thread (:meth:`FasterWhisperSTT.warmup` loads it ahead of time), and
every transcription runs in ``asyncio.to_thread``. See ``docs/providers/faster_whisper.md``
for devices, compute types and latency numbers.
"""

from __future__ import annotations

import asyncio
import math
import os
import threading
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
import numpy as np
import numpy.typing as npt

from ..audio.frame import AudioFrame
from ..errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderError,
    RateLimitError,
    VoiceAgentError,
)
from ..registry import register_provider
from ..stt import STT, STTCapabilities, Transcript, WordTiming
from ..utils.clock import now
from ..utils.deps import is_installed, require
from ..utils.log import logger

__all__ = ["FasterWhisperSTT"]

_PROVIDER = "faster_whisper"
_EXTRA = "faster-whisper"
_SAMPLE_RATE = 16_000  # Whisper's native input rate
_DEVICES = ("auto", "cpu", "cuda")
# compute_type="auto": the first type CTranslate2 supports on the device wins
_COMPUTE_PREFERENCE = {"cuda": ("float16", "int8", "float32"), "cpu": ("int8", "float32")}
_MODELS = (
    "large-v3-turbo",
    "large-v3",
    "distil-large-v3.5",
    "medium",
    "small",
    "small.en",
    "base",
    "base.en",
    "tiny",
    "tiny.en",
)


@register_provider(
    "stt",
    _PROVIDER,
    description="faster-whisper (Whisper on CTranslate2), local CPU int8 / CUDA float16",
    default_model="large-v3-turbo",
    models=_MODELS,
    env=(),
    extra=_EXTRA,
    requires=("faster_whisper", "ctranslate2"),
    local=True,
    aliases=("whisper",),
)
class FasterWhisperSTT(STT):
    """Local Whisper recognizer powered by faster-whisper / CTranslate2.

    Args:
        model: a model size (``tiny``, ``base``, ``small``, ``medium``, ``large-v3``,
            ``large-v3-turbo``, ``distil-large-v3.5``, ``*.en``...), a CTranslate2 Whisper
            repository on the Hugging Face Hub, or a local model directory.
        language: language code such as ``"en"`` or ``"de"`` (region suffixes like
            ``"en-US"`` are dropped); ``None`` detects the language of every utterance.
        device: ``"auto"`` uses CUDA when CTranslate2 sees a GPU *and* a warm-up inference
            succeeds on it (otherwise CPU); ``"cpu"`` or ``"cuda"`` force a device.
        device_index: CUDA device id (or ids, to spread concurrent requests over GPUs).
        compute_type: ``"auto"`` picks ``float16`` on CUDA and ``int8`` on CPU (falling
            back to what the device supports); any CTranslate2 compute type
            (``int8_float16``, ``float32``...) is used as given.
        beam_size: 1 is greedy decoding (lowest latency); Whisper's own default is 5.
        word_timestamps: fill :attr:`Transcript.words` (extra alignment pass).
        vad_filter: run faster-whisper's Silero VAD over each utterance first. Off by
            default: the pipeline has already segmented the speech.
        initial_prompt: text that primes the decoder (spelling, style, vocabulary).
        hotwords: hint phrases such as names or product terms.
        cpu_threads: CTranslate2 threads on CPU (0 = its default of 4, or
            ``OMP_NUM_THREADS``).
        num_workers: model replicas that can transcribe concurrently (for several
            sessions sharing one instance).
        download_root: model cache directory (default: the Hugging Face cache).
        local_files_only: never download; also implied by ``VAN_OFFLINE=1``.
        transcribe_options: extra keyword arguments for ``WhisperModel.transcribe()``
            (e.g. ``{"temperature": 0.0, "no_speech_threshold": 0.5}``); they override
            the options above.
    """

    provider = _PROVIDER

    def __init__(
        self,
        *,
        model: str | None = None,
        language: str | None = None,
        device: str = "auto",
        device_index: int | Sequence[int] = 0,
        compute_type: str = "auto",
        beam_size: int = 1,
        word_timestamps: bool = False,
        vad_filter: bool = False,
        initial_prompt: str | None = None,
        hotwords: str | None = None,
        cpu_threads: int = 0,
        num_workers: int = 1,
        download_root: str | os.PathLike[str] | None = None,
        local_files_only: bool = False,
        transcribe_options: Mapping[str, Any] | None = None,
    ) -> None:
        if not is_installed("faster_whisper"):
            # Fail fast with an install hint; the import itself (~0.3 s) happens in the
            # loader thread so constructing the provider never blocks an event loop.
            require("faster_whisper", extra=_EXTRA, package="faster-whisper")
        device = device.strip().lower()
        if device not in _DEVICES:
            raise ConfigurationError(
                f"faster_whisper: device must be one of {_DEVICES}, got {device!r}"
            )
        if beam_size < 1:
            raise ConfigurationError(f"faster_whisper: beam_size must be >= 1, got {beam_size}")
        name = model or "large-v3-turbo"
        super().__init__(
            model=name,
            capabilities=STTCapabilities(
                streaming=False,
                interim_results=False,
                word_timestamps=word_timestamps,
                language_detection=not _english_only(name),
            ),
            sample_rate=_SAMPLE_RATE,
            language=_whisper_language(language),
        )
        self.device = device
        self.device_index = device_index if isinstance(device_index, int) else list(device_index)
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.word_timestamps = word_timestamps
        self.vad_filter = vad_filter
        self.initial_prompt = initial_prompt
        self.hotwords = hotwords
        self.cpu_threads = cpu_threads
        self.num_workers = num_workers
        self.download_root = os.fspath(download_root) if download_root is not None else None
        self.local_files_only = local_files_only
        self.transcribe_options = dict(transcribe_options or {})
        self.resolved_device: str | None = None
        """Device the model runs on (``None`` until it is loaded)."""
        self.resolved_compute_type: str | None = None
        """CTranslate2 compute type the model was loaded with (``None`` until loaded)."""
        self._model: Any = None
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle
    async def warmup(self) -> None:
        """Download (if needed) and load the model, then run a short warm-up inference."""
        await asyncio.to_thread(self._ensure_model)

    async def aclose(self) -> None:
        # CTranslate2 frees host/GPU memory when the last reference goes away; an
        # in-flight transcription keeps its own reference until it finishes.
        self._model = None

    def _ensure_model(self) -> Any:
        model = self._model
        if model is None:
            with self._load_lock:
                model = self._model
                if model is None:
                    model = self._model = self._load()
        return model

    def _load(self) -> Any:
        fw = require("faster_whisper", extra=_EXTRA, package="faster-whisper")
        ct2 = require("ctranslate2", extra=_EXTRA)
        t0 = now()
        path = self._model_path(fw)
        if self.device == "auto":
            if _cuda_device_count(ct2) > 0:
                try:
                    return self._load_on(fw, ct2, path, "cuda", t0)
                except Exception as exc:
                    # Typical cause: the GPU is visible but CTranslate2 cannot run on it
                    # (cuBLAS 12 / cuDNN 9 missing, unsupported GPU, out of memory).
                    logger.warning(
                        "faster-whisper: CUDA is not usable (%s); falling back to CPU. "
                        "See docs/providers/faster_whisper.md to enable the GPU.",
                        exc.__cause__ or exc,
                    )
            return self._load_on(fw, ct2, path, "cpu", t0)
        return self._load_on(fw, ct2, path, self.device, t0)

    def _model_path(self, fw: Any) -> str:
        if os.path.isdir(self.model):
            return self.model
        offline = os.environ.get("VAN_OFFLINE", "").lower() in ("1", "true", "yes")
        try:
            return str(
                fw.download_model(
                    self.model,
                    local_files_only=self.local_files_only or offline,
                    cache_dir=self.download_root,
                )
            )
        except Exception as exc:
            raise _map_error(exc, f"downloading model {self.model!r}") from exc

    def _load_on(self, fw: Any, ct2: Any, path: str, device: str, t0: float) -> Any:
        compute_type = _resolve_compute_type(ct2, device, self.compute_type)
        try:
            model = fw.WhisperModel(
                path,
                device=device,
                device_index=self.device_index,
                compute_type=compute_type,
                cpu_threads=self.cpu_threads,
                num_workers=self.num_workers,
            )
            _warm_up(model)
        except Exception as exc:
            raise _map_error(exc, f"loading model {self.model!r} on {device}") from exc
        self.resolved_device, self.resolved_compute_type = device, compute_type
        logger.info(
            "faster-whisper %s loaded on %s (%s) in %.2fs",
            self.model,
            device,
            compute_type,
            now() - t0,
        )
        return model

    # ---------------------------------------------------------------- recognition
    async def _recognize(self, audio: AudioFrame, *, language: str | None) -> Transcript:
        # STT.transcribe() has already resampled the audio to 16 kHz mono.
        samples = audio.to_float32()
        return await asyncio.to_thread(self._transcribe_sync, samples, _whisper_language(language))

    def _transcribe_sync(
        self, samples: npt.NDArray[np.float32], language: str | None
    ) -> Transcript:
        if samples.size == 0:
            return Transcript(text="", language=language)
        model = self._ensure_model()
        options: dict[str, Any] = {
            "language": language,
            "beam_size": self.beam_size,
            "word_timestamps": self.word_timestamps,
            "vad_filter": self.vad_filter,
            "initial_prompt": self.initial_prompt,
            "hotwords": self.hotwords,
            **self.transcribe_options,
        }
        try:
            segments, info = model.transcribe(samples, **options)
            segments = list(segments)  # lazy generator: decoding happens while iterating
        except Exception as exc:
            raise _map_error(exc, "transcription") from exc
        return _to_transcript(segments, info, language, words=bool(options["word_timestamps"]))


# ---------------------------------------------------------------------- helpers
def _whisper_language(language: str | None) -> str | None:
    """``"en-US"``/``"pt_BR"`` -> ``"en"``/``"pt"``; empty, ``"auto"``, ``"multi"`` -> detect."""
    if not language:
        return None
    code = language.strip().replace("_", "-").split("-")[0].lower()
    return None if code in ("", "auto", "multi") else code


def _english_only(model: str) -> bool:
    name = os.path.basename(model.rstrip("/\\")).lower()
    return name.endswith(".en") or "distil" in name


def _cuda_device_count(ct2: Any) -> int:
    try:
        return int(ct2.get_cuda_device_count())
    except Exception:  # CPU-only builds (macOS wheels), broken drivers
        return 0


def _resolve_compute_type(ct2: Any, device: str, requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        supported = set(ct2.get_supported_compute_types(device))
    except Exception:
        supported = set()
    return next((ct for ct in _COMPUTE_PREFERENCE[device] if ct in supported), "default")


def _warm_up(model: Any) -> None:
    """Half a second of silence: initializes the kernels (and on CUDA proves they run)."""
    segments, _ = model.transcribe(
        np.zeros(_SAMPLE_RATE // 2, dtype=np.float32),
        language="en",
        beam_size=1,
        temperature=0.0,
        without_timestamps=True,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    for _ in segments:
        pass


def _to_transcript(
    segments: list[Any], info: Any, language: str | None, *, words: bool
) -> Transcript:
    text = "".join(s.text for s in segments).strip()
    confidence: float | None = None
    if segments:
        # exp(mean token log-probability), weighted by segment length in tokens
        weights = [max(1, len(s.tokens)) for s in segments]
        mean_logprob = sum(s.avg_logprob * w for s, w in zip(segments, weights, strict=True))
        confidence = min(1.0, math.exp(mean_logprob / sum(weights)))
    return Transcript(
        text=text,
        language=getattr(info, "language", None) or language,
        confidence=confidence,
        start_time=float(segments[0].start) if segments else None,
        end_time=float(segments[-1].end) if segments else None,
        words=(
            [
                WordTiming(
                    word=w.word.strip(),
                    start=float(w.start),
                    end=float(w.end),
                    confidence=float(w.probability),
                )
                for s in segments
                for w in (s.words or ())
            ]
            if words
            else None
        ),
    )


def _map_error(exc: Exception, action: str) -> VoiceAgentError:
    """Translate faster-whisper / CTranslate2 / Hugging Face Hub failures to library errors.

    Hub exceptions are matched by class name so that this module (and its tests) never
    has to import ``huggingface_hub``.
    """
    if isinstance(exc, VoiceAgentError):
        return exc
    message = f"faster-whisper {action} failed: {exc}"
    names = {cls.__name__ for cls in type(exc).__mro__}
    status = getattr(getattr(exc, "response", None), "status_code", None)
    status = status if isinstance(status, int) else None
    if "GatedRepoError" in names:
        return AuthenticationError(message, provider=_PROVIDER, status_code=status)
    if names & {"RepositoryNotFoundError", "RevisionNotFoundError"} or isinstance(
        exc, (ValueError, TypeError)
    ):
        return ConfigurationError(message)  # unknown model/revision, bad option or language
    if status in (401, 403):
        return AuthenticationError(message, provider=_PROVIDER, status_code=status)
    if status == 429:
        return RateLimitError(message, provider=_PROVIDER, status_code=status)
    if names & {"HfHubHTTPError", "LocalEntryNotFoundError", "OfflineModeIsEnabled"} or isinstance(
        exc, (ConnectionError, TimeoutError, httpx.TransportError)
    ):
        return ProviderConnectionError(message, provider=_PROVIDER, status_code=status)
    return ProviderError(message, provider=_PROVIDER, status_code=status)
