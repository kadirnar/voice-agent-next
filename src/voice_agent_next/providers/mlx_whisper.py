"""Whisper on MLX (mlx-whisper): 99-language speech recognition on Apple silicon GPUs.

``create("stt", "mlx_whisper/large-v3-turbo")`` (the default). Model names follow
faster-whisper's (``tiny``, ``base.en``, ``small``, ``medium``, ``large-v3``,
``large-v3-turbo``, ``distil-large-v3``); a Hugging Face repository id of an MLX Whisper
conversion or a local directory works too.

Whisper is a batch recognizer (``capabilities.streaming=False``): the cascade wraps it in
a :class:`~voice_agent_next.stt.StreamAdapter`, which cuts the input into utterances with
the VAD and transcribes each one. With ``interim_results=True`` it also re-decodes the
growing utterance while the user speaks, like faster-whisper (the same adapter stream).
For streaming English and European languages, prefer Parakeet
(``mlx/parakeet-tdt-0.6b-v3``).

The :class:`~voice_agent_next.stt_guard.HallucinationGuard` shared with faster-whisper (on
by default) drops what Whisper "hears" in noise: no-speech segments, known subtitle
artifacts in many languages, repetition loops and stock phrases the VAD was unsure about.

Install with ``pip install 'voice-agent-next[mlx-whisper]'`` (macOS on Apple silicon).
Every MLX call runs on one shared worker thread (see :mod:`._mlx`). See
``docs/providers/mlx.md``.
"""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ..audio.frame import AudioFrame
from ..errors import ConfigurationError
from ..models import ModelFile, register_model
from ..registry import register_provider
from ..stt import STT, STTCapabilities, STTStream, Transcript, WordTiming
from ..stt_guard import HallucinationGuard, coerce_guard
from ..utils.clock import now
from ..utils.deps import require
from ..utils.log import logger
from . import _mlx
from ._whisper_stream import WhisperAdapterStream, frames_to_samples
from ._whisper_stream import whisper_language as _whisper_language

if TYPE_CHECKING:
    from ..stt import StreamAdapter

__all__ = ["DEFAULT_MODEL", "MODELS", "MLXWhisperSTT"]

_PROVIDER = "mlx_whisper"
_EXTRA = "mlx-whisper"
_SAMPLE_RATE = 16_000
_FILES = ("config.json", "weights.npz", "weights.safetensors")

DEFAULT_MODEL = "large-v3-turbo"
_INTERIM_INTERVAL = 0.25
"""Default seconds of new speech between two interim decodes (a GPU, like CUDA)."""

MODELS: dict[str, tuple[str, int]] = {
    "large-v3-turbo": ("mlx-community/whisper-large-v3-turbo", 1_613_977_880),
    "large-v3": ("mlx-community/whisper-large-v3-mlx", 3_083_520_685),
    "distil-large-v3": ("mlx-community/distil-whisper-large-v3", 1_509_130_380),
    "medium": ("mlx-community/whisper-medium-mlx", 1_524_925_180),
    "medium.en": ("mlx-community/whisper-medium.en-mlx", 1_524_923_324),
    "small": ("mlx-community/whisper-small-mlx", 481_307_858),
    "small.en": ("mlx-community/whisper-small.en-mlx", 481_306_466),
    "base": ("mlx-community/whisper-base-mlx", 143_724_466),
    "base.en": ("mlx-community/whisper-base.en-mlx", 143_723_394),
    "tiny": ("mlx-community/whisper-tiny", 74_418_444),
    "tiny.en": ("mlx-community/whisper-tiny.en-mlx", 74_418_066),
}
"""Known models: name -> (MLX conversion on the Hugging Face Hub, download size)."""


def _resolve(model: str) -> str:
    key = model.strip()
    short = key.removeprefix("whisper-")
    if short in MODELS:
        return MODELS[short][0]
    return key


def _english_only(model: str) -> bool:
    name = os.path.basename(model.rstrip("/\\")).lower()
    return ".en" in name or "distil" in name


@register_provider(
    "stt",
    _PROVIDER,
    description="Whisper on MLX (mlx-whisper), 99 languages, Apple silicon GPU",
    default_model=DEFAULT_MODEL,
    models=tuple(MODELS),
    env=(),
    extra=_EXTRA,
    requires=("mlx_whisper", "mlx"),
    local=True,
    platforms=_mlx.PLATFORMS,
)
class MLXWhisperSTT(STT):
    """Whisper recognition on the Apple silicon GPU with mlx-whisper.

    Args:
        model: a name from :data:`MODELS`, a Hugging Face repository id of an MLX Whisper
            conversion, or a local directory.
        language: language code such as ``"en"`` (region suffixes are dropped); ``None``
            detects the language of every utterance.
        fp16: run in float16 (default) rather than float32.
        word_timestamps: fill :attr:`Transcript.words` (extra alignment pass).
        initial_prompt: text that primes the decoder (spelling, vocabulary).
        local_files_only: never download; also implied by ``VAN_OFFLINE=1``.
        transcribe_options: extra keyword arguments for ``mlx_whisper.transcribe()``
            (e.g. ``{"no_speech_threshold": 0.5}``); they override the options above.
        interim_results: behind a VAD (:class:`~voice_agent_next.stt.StreamAdapter`, as in
            the cascade), re-decode the utterance while the user speaks and emit
            ``INTERIM_TRANSCRIPT`` events. Interim decodes run without timestamps or word
            alignment; the final transcript is decoded as usual.
        interim_interval: seconds of new speech between two interim decodes (at least: a
            decode that takes longer than half of it spaces the next one out); 0.25 s by
            default.
        hallucination_guard: drop segments that are probably not speech (see
            :class:`~voice_agent_next.stt_guard.HallucinationGuard`): ``True`` (default),
            ``False``, a guard or a mapping of its fields.
        final_from_interim: behind a VAD with ``interim_results``, use the latest interim
            decode as the final transcript when it started after the last voiced VAD window
            (no second decode when the input ends right after the speech). Ignored with
            ``word_timestamps``.
    """

    provider = _PROVIDER

    def __init__(
        self,
        *,
        model: str | None = None,
        language: str | None = None,
        fp16: bool = True,
        word_timestamps: bool = False,
        initial_prompt: str | None = None,
        local_files_only: bool = False,
        transcribe_options: Mapping[str, Any] | None = None,
        interim_results: bool = False,
        interim_interval: float | None = None,
        hallucination_guard: bool | HallucinationGuard | Mapping[str, Any] = True,
        final_from_interim: bool = False,
    ) -> None:
        _mlx.ensure_available(
            "mlx_whisper", extra=_EXTRA, package="mlx-whisper", provider=_PROVIDER
        )
        name = (model or DEFAULT_MODEL).strip()
        if "parakeet" in name.lower():
            raise ConfigurationError(f"mlx_whisper: {name!r} is a Parakeet model; use mlx/{name}")
        if interim_interval is not None and interim_interval <= 0:
            raise ConfigurationError(
                f"mlx_whisper: interim_interval must be > 0, got {interim_interval}"
            )
        guard = coerce_guard(hallucination_guard, provider=_PROVIDER)
        super().__init__(
            model=name,
            capabilities=STTCapabilities(
                streaming=False,
                interim_results=interim_results,
                word_timestamps=word_timestamps,
                language_detection=not _english_only(_resolve(name)),
            ),
            sample_rate=_SAMPLE_RATE,
            language=_whisper_language(language),
        )
        self.fp16 = fp16
        self.word_timestamps = word_timestamps
        self.initial_prompt = initial_prompt
        self.local_files_only = local_files_only
        self.transcribe_options = dict(transcribe_options or {})
        self.interim_interval = interim_interval
        self.guard = guard
        """The :class:`~voice_agent_next.stt_guard.HallucinationGuard` applied to every
        transcript."""
        self.final_from_interim = final_from_interim
        self._model: Any = None
        self._path: str | None = None
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle
    async def warmup(self) -> None:
        """Download (if needed) and load the model, then run a short warm-up inference."""
        await _mlx.WORKER.run(self._ensure_model)

    async def aclose(self) -> None:
        self._model = None

    def _ensure_model(self) -> Any:
        model = self._model
        if model is None:
            with self._load_lock:
                model = self._model
                if model is None:
                    model = self._load()
        return model

    def _load(self) -> Any:
        mx = _mlx.import_mlx(_PROVIDER)
        load_models = require("mlx_whisper.load_models", extra=_EXTRA, package="mlx-whisper")
        t0 = now()
        path = _mlx.snapshot(
            _resolve(self.model),
            patterns=_FILES,
            local_files_only=self.local_files_only,
            provider=_PROVIDER,
        )
        try:
            model = load_models.load_model(path, dtype=mx.float16 if self.fp16 else mx.float32)
        except Exception as exc:
            raise _mlx.map_error(exc, _PROVIDER, f"loading model {self.model!r}") from exc
        self._model, self._path = model, path
        try:  # compiles the Metal kernels ahead of the first utterance
            self._run(np.zeros(_SAMPLE_RATE // 2, dtype=np.float32), "en", {})
        except Exception as exc:
            self._model = None
            raise _mlx.map_error(exc, _PROVIDER, f"warming up model {self.model!r}") from exc
        logger.info("mlx-whisper %s loaded in %.2fs", self.model, now() - t0)
        return model

    # ---------------------------------------------------------------- recognition
    async def _recognize(self, audio: AudioFrame, *, language: str | None) -> Transcript:
        samples = audio.to_float32()
        return await _mlx.WORKER.run(self._transcribe_sync, samples, _whisper_language(language))

    def _create_adapter_stream(
        self, adapter: StreamAdapter, *, language: str | None
    ) -> STTStream | None:
        return WhisperAdapterStream(adapter, self, language=language)

    @property
    def resolved_interim_interval(self) -> float:
        """Seconds of new speech between interim decodes."""
        return self.interim_interval if self.interim_interval is not None else _INTERIM_INTERVAL

    @property
    def parallel_final(self) -> bool:
        """Never: every MLX call runs on the one MLX worker thread."""
        return False

    async def decode_frames(
        self,
        frames: Sequence[AudioFrame],
        language: str | None,
        *,
        interim: bool,
        vad_confidence: float | None,
    ) -> Transcript:
        """Transcribe VAD frames (any rate) on the MLX thread (for the adapter stream)."""
        return await _mlx.WORKER.run(
            self._transcribe_frames, frames, language, interim, vad_confidence
        )

    def _transcribe_frames(
        self,
        frames: Sequence[AudioFrame],
        language: str | None,
        interim: bool,
        vad_confidence: float | None,
    ) -> Transcript:
        return self._transcribe_sync(
            frames_to_samples(frames), language, interim=interim, vad_confidence=vad_confidence
        )

    def _transcribe_sync(
        self,
        samples: npt.NDArray[np.float32],
        language: str | None,
        *,
        interim: bool = False,
        vad_confidence: float | None = None,
    ) -> Transcript:
        if samples.size == 0:
            return Transcript(text="", language=language)
        self._ensure_model()
        options: dict[str, Any] = {
            "word_timestamps": self.word_timestamps,
            "initial_prompt": self.initial_prompt,
            **self.transcribe_options,
        }
        if interim:  # fast and cheap: the final transcript is decoded properly anyway
            options.update(word_timestamps=False, temperature=0.0, without_timestamps=True)
        try:
            result = self._run(samples, language, options)
        except Exception as exc:
            raise _mlx.map_error(exc, _PROVIDER, "transcription") from exc
        segments = list(result.get("segments") or [])
        verdict = self.guard.filter(segments, vad_confidence=vad_confidence)
        if verdict.dropped:
            logger.debug(
                "mlx-whisper: dropped %s (%s)",
                "; ".join(f"{t!r}: {why}" for t, why in verdict.dropped),
                "interim" if interim else "final",
            )
        return _to_transcript(
            verdict.kept,
            result.get("language") or language,
            words=bool(options["word_timestamps"]),
            clean=self.guard.clean_text,
            text=None if len(verdict.kept) != len(segments) else result.get("text"),
        )

    def _run(
        self, samples: npt.NDArray[np.float32], language: str | None, options: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        mw = require("mlx_whisper", extra=_EXTRA, package="mlx-whisper")
        transcribe_mod = require("mlx_whisper.transcribe", extra=_EXTRA, package="mlx-whisper")
        # mlx_whisper.transcribe() keeps one model in ModelHolder, keyed by path. Every
        # call runs on the MLX worker thread, so pointing the holder at this instance's
        # model first means no reload, also with several models in one process.
        holder = transcribe_mod.ModelHolder
        holder.model, holder.model_path = self._model, self._path
        opts: dict[str, Any] = {
            "path_or_hf_repo": self._path,
            "language": language,
            "fp16": self.fp16,
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "verbose": None,
            **options,
        }
        result: Mapping[str, Any] = mw.transcribe(samples, **opts)
        return result


def _to_transcript(
    segments: Sequence[Mapping[str, Any]],
    language: str | None,
    *,
    words: bool,
    clean: Callable[[str], str] | None = None,
    text: str | None = None,
) -> Transcript:
    """``text`` is mlx-whisper's own text of all the segments (``None``: join them)."""
    if text is None:
        text = "".join(str(s.get("text") or "") for s in segments)
    text = text.strip()
    if clean is not None:
        text = clean(text)
    confidence: float | None = None
    if segments:
        weights = [max(1, len(s.get("tokens") or ())) for s in segments]
        logprobs = [float(s.get("avg_logprob", 0.0)) for s in segments]
        total = sum(lp * w for lp, w in zip(logprobs, weights, strict=True))
        confidence = min(1.0, math.exp(total / sum(weights)))
    return Transcript(
        text=text,
        language=language,
        confidence=confidence,
        start_time=float(segments[0]["start"]) if segments else None,
        end_time=float(segments[-1]["end"]) if segments else None,
        words=(
            [
                WordTiming(
                    word=str(w["word"]).strip(),
                    start=float(w["start"]),
                    end=float(w["end"]),
                    confidence=float(w.get("probability", 0.0)),
                )
                for s in segments
                for w in (s.get("words") or ())
            ]
            if words
            else None
        ),
    )


for _name, (_repo, _size) in MODELS.items():
    register_model(
        _PROVIDER,
        _name,
        kind="stt",
        files=[
            ModelFile.from_hf_repo(_repo, patterns=_FILES, required=("config.json",), size=_size)
        ],
        license="MIT",
        languages="en" if _english_only(_repo) else "99 languages",
        description=f"Whisper {_name} (MLX) from {_repo}",
        aliases=("turbo",) if _name == "large-v3-turbo" else (),
    )
