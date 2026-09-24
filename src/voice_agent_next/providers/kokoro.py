"""Kokoro-82M local text-to-speech through ``kokoro-onnx``.

Kokoro is an 82M-parameter, Apache-2.0 speech synthesis model with 24 kHz mono output:
54 voices in 8 languages for v1.0, plus a Chinese/English v1.1 variant. It runs on ONNX
Runtime: CUDA, CoreML or DirectML when available, CPU otherwise.

Usage::

    from voice_agent_next import create

    tts = create("tts", "kokoro")  # v1.0 fp32, voice "af_heart"
    tts = create("tts", "kokoro/v1.0-int8", voice="bf_emma", speed=1.1)
    await tts.warmup()  # download + load ahead of the first request

The model and voice files come from the kokoro-onnx ``model-files-v1.1`` GitHub release
(pinned URLs and sha256 digests). They are downloaded into the shared model cache on
first use, unless ``model_path`` / ``voices_path`` point at local files.

Kokoro takes complete text (no incremental text input), so :meth:`TTS.stream` uses the
base :class:`~voice_agent_next.tts.SentenceStreamAdapter`. Every request is split into
sentences that are synthesized one at a time on a dedicated worker thread; the audio of
each sentence is emitted in ``chunk_duration`` chunks as soon as it is ready.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, TypeVar

import numpy as np

from ..audio.frame import SAMPLE_WIDTH, AudioFrame
from ..errors import ConfigurationError, ProviderError
from ..registry import register_provider
from ..text.sentences import SentenceSegmenter
from ..tts import TTS, ChunkedStream
from ..utils.clock import now
from ..utils.deps import is_installed, require
from ..utils.download import download
from ..utils.log import logger

__all__ = [
    "DEFAULT_MODEL",
    "KOKORO_MODELS",
    "SAMPLE_RATE",
    "ExecutionProvider",
    "KokoroAsset",
    "KokoroModel",
    "KokoroTTS",
    "lang_for_voice",
    "select_execution_providers",
]

T = TypeVar("T")

ExecutionProvider: TypeAlias = str | tuple[str, dict[str, Any]]
"""An ONNX Runtime execution provider name, optionally with its provider options."""

SAMPLE_RATE = 24_000
DEFAULT_MODEL = "v1.0"
_EXTRA = "kokoro"
_RELEASE = "model-files-v1.1"
_RELEASE_URL = f"https://github.com/thewh1teagle/kokoro-onnx/releases/download/{_RELEASE}"
_CPU = "CPUExecutionProvider"
_ACCELERATED = ("CUDAExecutionProvider", "CoreMLExecutionProvider", "DmlExecutionProvider")
"""Preference order; the first one ONNX Runtime reports as available is used."""


@dataclass(frozen=True, slots=True)
class KokoroAsset:
    """One file of the pinned kokoro-onnx ``model-files-v1.1`` release."""

    filename: str
    sha256: str
    size: int
    """Size in bytes (informational)."""

    @property
    def url(self) -> str:
        return f"{_RELEASE_URL}/{self.filename}"


@dataclass(frozen=True, slots=True)
class KokoroModel:
    """A Kokoro model variant: the ONNX graph and the voice pack that goes with it."""

    onnx: KokoroAsset
    voices: KokoroAsset
    default_voice: str


# sha256 digests as published by GitHub for the release assets; voices-v1.1-zh.bin has
# no published digest, so its digest was computed from the release asset (2026-09-24).
_VOICES_V1_0 = KokoroAsset(
    "voices-v1.0.bin",
    "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
    28_214_398,
)
_VOICES_V1_1_ZH = KokoroAsset(
    "voices-v1.1-zh.bin",
    "14cb6186c99e4f6016871405f62046c5df863ae27465cbdc4ee08be7dd703acd",
    53_815_880,
)

KOKORO_MODELS: dict[str, KokoroModel] = {
    "v1.0": KokoroModel(
        KokoroAsset(
            "kokoro-v1.0.onnx",
            "beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a",
            325_505_369,
        ),
        _VOICES_V1_0,
        "af_heart",
    ),
    "v1.0-fp16": KokoroModel(
        KokoroAsset(
            "kokoro-v1.0.fp16.onnx",
            "f3a290d384fbb27966d462905c71a46cef9e5fd00516b40df32a0b4afe77ac96",
            163_527_961,
        ),
        _VOICES_V1_0,
        "af_heart",
    ),
    "v1.0-int8": KokoroModel(
        KokoroAsset(
            "kokoro-v1.0.int8.onnx",
            "ae315a79b623f244700e4afb9246c46a26066782e049ba174bf3ba433970ee9c",
            114_119_327,
        ),
        _VOICES_V1_0,
        "af_heart",
    ),
    "v1.1-zh": KokoroModel(
        KokoroAsset(
            "kokoro-v1.1-zh.onnx",
            "859f9ded9f53be16c24857cdab3254a45da53c3afd5ba6ef134c7de3f822e326",
            325_506_167,
        ),
        _VOICES_V1_1_ZH,
        "zf_001",
    ),
    "v1.1-zh-fp16": KokoroModel(
        KokoroAsset(
            "kokoro-v1.1-zh.fp16.onnx",
            "a628ea5d6fbde96d1a85f691a6a00847829937f9e488021ba2c5359bc6ea08b5",
            163_528_759,
        ),
        _VOICES_V1_1_ZH,
        "zf_001",
    ),
    "v1.1-zh-int8": KokoroModel(
        KokoroAsset(
            "kokoro-v1.1-zh.int8.onnx",
            "11751c087b4bbeed031e2b687b11dda698bd27ba0509472f960b57f835a999f7",
            114_120_125,
        ),
        _VOICES_V1_1_ZH,
        "zf_001",
    ),
}
"""Known model ids. Aliases such as ``"int8"`` or ``"kokoro-v1.0.int8.onnx"`` also work."""

# The first letter of a Kokoro voice name is its language; values are espeak-ng codes.
_LANG_BY_PREFIX = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt-br",
    "z": "cmn",
}

_CLOSERS = r"\"'”’)\]」』）"
_SENTENCE_END = re.compile(rf"[.!?…。！？｡][{_CLOSERS}]*$")
_CLAUSE_END = re.compile(rf"[,;:，；：、][{_CLOSERS}]*$")


def _normalize_model(name: str) -> str:
    """``"kokoro-v1.0.int8.onnx"`` / ``"V1.0_INT8"`` / ``"int8"`` -> ``"v1.0-int8"``."""
    key = name.strip().lower().removesuffix(".onnx").removeprefix("kokoro-")
    if key in ("fp32", "fp16", "int8"):
        key = f"{DEFAULT_MODEL}-{key}"
    key = re.sub(r"[._-](fp32|fp16|int8)$", r"-\1", key)
    return key.removesuffix("-fp32")


def lang_for_voice(voice: str) -> str:
    """espeak-ng language code for a Kokoro voice (``"bf_emma"`` -> ``"en-gb"``)."""
    return _LANG_BY_PREFIX.get(voice[:1].lower(), "en-us")


def select_execution_providers(available: Sequence[str]) -> list[str]:
    """Pick ONNX Runtime execution providers: the best accelerator available, then CPU.

    CUDA is preferred, then CoreML (macOS), then DirectML (Windows); CPU is always last,
    both as the fallback for operators the accelerator does not support and as the only
    provider when no accelerator is available.
    """
    for name in _ACCELERATED:
        if name in available:
            return [name, _CPU]
    return [_CPU]


def _pause_after(text: str, sentence_pause: float, clause_pause: float) -> float:
    """Seconds of silence to append after a segment, from its final punctuation."""
    text = text.rstrip()
    if _SENTENCE_END.search(text):
        return sentence_pause
    if _CLAUSE_END.search(text):
        return clause_pause
    return 0.0


class _DropWordsMismatch(logging.Filter):
    """phonemizer warns on every sentence where espeak merges words ("of the" -> "ʌvðə")."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not str(record.msg).startswith("words count mismatch")


def _quiet_phonemizer() -> None:
    log = logging.getLogger("phonemizer")
    if not any(isinstance(f, _DropWordsMismatch) for f in log.filters):
        log.addFilter(_DropWordsMismatch())


def _check_installed() -> None:
    """Fail fast when the extra is missing, without importing the heavy packages."""
    for module, package in (("kokoro_onnx", "kokoro-onnx"), ("onnxruntime", "onnxruntime")):
        if not is_installed(module):
            require(module, extra=_EXTRA, package=package)


def _local_file(override: Path | None, asset: KokoroAsset) -> Path:
    if override is None:
        return download(asset.url, subdir=f"kokoro/{_RELEASE}", sha256=asset.sha256)
    if not override.is_file():
        raise ConfigurationError(f"Kokoro file not found: {override}")
    return override


@register_provider(
    "tts",
    "kokoro",
    description="Kokoro-82M local TTS via kokoro-onnx (24 kHz, 54 voices, 8 languages)",
    default_model=DEFAULT_MODEL,
    models=tuple(KOKORO_MODELS),
    env=(),
    extra=_EXTRA,
    requires=("kokoro_onnx", "onnxruntime"),
    local=True,
)
class KokoroTTS(TTS):
    """Kokoro-82M text-to-speech running locally on ONNX Runtime (24 kHz mono).

    Args:
        model: model id: ``"v1.0"`` (fp32, default), ``"v1.0-fp16"``, ``"v1.0-int8"``
            (smallest download), ``"v1.1-zh"``, ``"v1.1-zh-fp16"`` or ``"v1.1-zh-int8"``.
            With ``model_path`` any label is accepted.
        voice: default voice, e.g. ``"af_heart"`` (the default for v1.0) or ``"bm_george"``.
            The first letter selects the language (see ``lang``).
        speed: speaking rate, 0.5 to 2.0.
        lang: espeak-ng language code for phonemization (``"en-us"``, ``"en-gb"``, ``"es"``,
            ``"fr-fr"``, ``"hi"``, ``"it"``, ``"ja"``, ``"pt-br"``, ``"cmn"``...). Default:
            derived from the voice name of each request.
        model_path: local ``.onnx`` file instead of the downloaded one.
        voices_path: local voice pack (``voices-*.bin``) instead of the downloaded one.
        providers: ONNX Runtime execution providers, e.g. ``["CPUExecutionProvider"]``.
            Default: CUDA, CoreML or DirectML when available, CPU otherwise (and CPU when
            the accelerated session cannot be created).
        num_threads: ONNX Runtime intra-op threads (default: one per physical core).
        chunk_duration: duration of the emitted audio chunks, in seconds.
        split_sentences: synthesize long texts sentence by sentence so the first audio
            is ready sooner (``False``: one inference pass per request, split by
            kokoro-onnx only when longer than the model context).
        sentence_pause: silence after sentence-final punctuation, in seconds (inside the
            text and at the end of each sentence, since ``trim`` removes the model's own).
        clause_pause: silence after ``,`` ``;`` ``:``, in seconds.
        trim: trim the silence the model generates around each segment.
        g2p: optional ``(text, lang) -> phonemes`` function used instead of the built-in
            espeak-ng phonemizer (e.g. misaki for better Japanese/Chinese).
        clean_text: strip markdown/emoji before synthesis.
    """

    provider = "kokoro"

    def __init__(
        self,
        *,
        model: str | None = None,
        voice: str | None = None,
        speed: float = 1.0,
        lang: str | None = None,
        model_path: str | os.PathLike[str] | None = None,
        voices_path: str | os.PathLike[str] | None = None,
        providers: Sequence[ExecutionProvider] | str | None = None,
        num_threads: int | None = None,
        chunk_duration: float = 0.05,
        split_sentences: bool = True,
        sentence_pause: float = 0.25,
        clause_pause: float = 0.1,
        trim: bool = True,
        g2p: Callable[[str, str], str] | None = None,
        clean_text: bool = True,
    ) -> None:
        model_id = _normalize_model(model or DEFAULT_MODEL)
        variant = KOKORO_MODELS.get(model_id)
        if variant is None:
            if model_path is None:
                raise ConfigurationError(
                    f"unknown Kokoro model {model!r}; known models: {', '.join(KOKORO_MODELS)} "
                    "(or pass model_path= for a custom export)"
                )
            variant = KOKORO_MODELS[DEFAULT_MODEL]  # custom exports use the v1.0 voice pack
            model_id = (model or model_id).strip()
        if not 0.5 <= speed <= 2.0:
            raise ConfigurationError(f"Kokoro speed must be between 0.5 and 2.0, got {speed}")
        if chunk_duration <= 0:
            raise ConfigurationError(f"chunk_duration must be > 0, got {chunk_duration}")
        if sentence_pause < 0 or clause_pause < 0:
            raise ConfigurationError("sentence_pause and clause_pause must be >= 0")
        if num_threads is not None and num_threads < 1:
            raise ConfigurationError(f"num_threads must be >= 1, got {num_threads}")
        _check_installed()
        super().__init__(
            model=model_id,
            sample_rate=SAMPLE_RATE,
            voice=voice or variant.default_voice,
            clean_text=clean_text,
        )
        self.speed = float(speed)
        self.lang = lang
        self.num_threads = num_threads
        self.chunk_duration = chunk_duration
        self.split_sentences = split_sentences
        self.sentence_pause = sentence_pause
        self.clause_pause = clause_pause
        self.trim = trim
        self.g2p = g2p
        self._variant = variant
        self._model_path = Path(model_path).expanduser() if model_path is not None else None
        self._voices_path = Path(voices_path).expanduser() if voices_path is not None else None
        if isinstance(providers, str):
            providers = [providers]
        self._providers: list[ExecutionProvider] | None = (
            list(providers) if providers is not None else None
        )
        self._chunk_bytes = max(1, round(chunk_duration * SAMPLE_RATE)) * SAMPLE_WIDTH
        self._engine: Any = None  # kokoro_onnx.Kokoro, created on the worker thread
        self._executor: ThreadPoolExecutor | None = None

    # ------------------------------------------------------------------ public API
    def _synthesize(self, text: str, *, voice: str | None) -> ChunkedStream:
        return _KokoroChunkedStream(self, text, voice=voice)

    async def warmup(self) -> None:
        """Download (if needed) and load the model, then run one short synthesis."""
        await self._submit(self._warmup_sync)

    async def list_voices(self) -> list[str]:
        """Voice names of the loaded voice pack (loads the model if needed)."""
        engine = await self._submit(self._get_engine)
        return list(engine.get_voices())

    async def aclose(self) -> None:
        """Stop the worker thread (a synthesis already running finishes in background)."""
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        self._engine = None

    # ------------------------------------------------------------------ internals
    async def _submit(self, fn: Callable[..., T], *args: Any) -> T:
        """Run ``fn`` on the worker thread: one at a time, in submission order."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kokoro")
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    def _segments(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if self.split_sentences:
            segmenter = SentenceSegmenter(min_chars=10, first_segment_min_chars=4)
            parts = segmenter.push(text) + segmenter.flush()
        else:
            parts = [text]
        return [p for p in parts if any(ch.isalnum() for ch in p)]

    def _get_engine(self) -> Any:
        """The loaded ``kokoro_onnx.Kokoro`` instance (worker thread only)."""
        if self._engine is None:
            self._engine = self._load()
        return self._engine

    def _load(self) -> Any:
        kokoro_onnx = require("kokoro_onnx", extra=_EXTRA, package="kokoro-onnx")
        ort = require("onnxruntime", extra=_EXTRA)
        _quiet_phonemizer()
        t0 = now()
        model_path = _local_file(self._model_path, self._variant.onnx)
        voices_path = _local_file(self._voices_path, self._variant.voices)
        options = ort.SessionOptions()
        if self.num_threads is not None:
            options.intra_op_num_threads = self.num_threads
        session = self._create_session(ort, model_path, options)
        try:
            engine = kokoro_onnx.Kokoro.from_session(session, str(voices_path))
        except Exception as exc:
            raise ProviderError(
                f"failed to initialize Kokoro with {voices_path}: {exc}", provider=self.provider
            ) from exc
        logger.info(
            "kokoro: loaded %s on %s in %.2f s",
            model_path.name,
            ", ".join(session.get_providers()),
            now() - t0,
        )
        return engine

    def _create_session(self, ort: Any, path: Path, options: Any) -> Any:
        explicit = self._providers is not None
        providers = (
            list(self._providers)
            if self._providers is not None
            else select_execution_providers(ort.get_available_providers())
        )
        try:
            return ort.InferenceSession(str(path), sess_options=options, providers=providers)
        except Exception as exc:
            if explicit or providers == [_CPU]:
                raise ProviderError(
                    f"failed to load Kokoro model {path}: {exc}", provider=self.provider
                ) from exc
            logger.warning("kokoro: cannot use %s (%s); falling back to CPU", providers[0], exc)
        try:
            return ort.InferenceSession(str(path), sess_options=options, providers=[_CPU])
        except Exception as exc:
            raise ProviderError(
                f"failed to load Kokoro model {path}: {exc}", provider=self.provider
            ) from exc

    def _warmup_sync(self) -> None:
        t0 = now()
        self._synthesize_segment("Hello.", self.voice or KOKORO_MODELS[DEFAULT_MODEL].default_voice)
        logger.debug("kokoro: warm-up took %.2f s", now() - t0)

    def _synthesize_segment(self, text: str, voice: str) -> bytes:
        """Synthesize one segment to s16le PCM, followed by its pause (worker thread)."""
        engine = self._get_engine()
        if voice not in engine.voices:
            raise ConfigurationError(
                f"unknown Kokoro voice {voice!r} for model {self.model!r}; available: "
                f"{', '.join(engine.get_voices())}"
            )
        lang = self.lang or lang_for_voice(voice)
        options: dict[str, Any] = {
            "voice": voice,
            "speed": self.speed,
            "lang": lang,
            "trim": self.trim,
            "sentence_pause": self.sentence_pause,
            "clause_pause": self.clause_pause,
        }
        try:
            if self.g2p is None:
                samples, sample_rate = engine.create(text, **options)
            else:
                phonemes = self.g2p(text, lang)
                samples, sample_rate = engine.create(phonemes, is_phonemes=True, **options)
        except ValueError as exc:
            # kokoro-onnx raises ValueError when nothing in the text is pronounceable
            if "phoneme" not in str(exc):
                raise ProviderError(
                    f"Kokoro synthesis failed: {exc}", provider=self.provider
                ) from exc
            logger.warning("kokoro: nothing to synthesize in %r (%s)", text, exc)
            return b""
        except Exception as exc:
            raise ProviderError(f"Kokoro synthesis failed: {exc}", provider=self.provider) from exc
        if sample_rate != SAMPLE_RATE:
            raise ProviderError(
                f"Kokoro returned {sample_rate} Hz audio, expected {SAMPLE_RATE} Hz",
                provider=self.provider,
            )
        pcm = AudioFrame.from_numpy(np.asarray(samples, dtype=np.float32).reshape(-1), SAMPLE_RATE)
        pause = _pause_after(text, self.sentence_pause, self.clause_pause)
        return pcm.data + bytes(round(pause * SAMPLE_RATE) * SAMPLE_WIDTH)


class _KokoroChunkedStream(ChunkedStream):
    async def _run(self) -> None:
        tts = self._tts
        assert isinstance(tts, KokoroTTS)
        voice = self.voice or tts.voice or KOKORO_MODELS[DEFAULT_MODEL].default_voice
        step = tts._chunk_bytes
        for segment in tts._segments(self.text):
            pcm = await tts._submit(tts._synthesize_segment, segment, voice)
            for start in range(0, len(pcm), step):
                self._push_audio(pcm[start : start + step])
