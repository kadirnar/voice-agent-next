"""mlx-audio text-to-speech on Apple silicon GPUs: Kokoro, Pocket TTS and more.

``create("tts", "mlx_audio/kokoro")`` (the default model) or
``create("tts", "mlx_audio/pocket-tts")``. Any other mlx-audio TTS model loads from its
Hugging Face repository id (``mlx_audio/mlx-community/<model>``) or a local directory;
``sample_rate=`` then declares the output rate (mlx-audio's output is resampled to it).

* **Kokoro-82M** (Apache-2.0, 24 kHz, 54 voices in 8 languages): one chunk per text
  segment; the cascade feeds it sentence by sentence. The voice's first letter picks the
  language (``af_heart`` American English, ``bf_emma`` British, ``ef_dora`` Spanish,
  ``ff_siwis`` French, ``hf_alpha`` Hindi, ``if_sara`` Italian, ``jf_alpha`` Japanese,
  ``pf_dora`` Portuguese, ``zf_xiaobei`` Mandarin). English needs misaki's G2P (in the
  ``mlx`` extra on Python < 3.13) and spaCy's ``en_core_web_sm``, which misaki installs
  with pip on first use (see ``docs/providers/mlx.md`` for environments without pip).
* **Pocket TTS** (Kyutai, CC-BY-4.0, 24 kHz, English): audio streaming — chunks of
  ``streaming_interval`` seconds arrive while the rest of the sentence is generated.
  Voices: ``alba`` (default), ``marius``, ``javert``, ``jean``, ``fantine``, ``cosette``,
  ``eponine``, ``azelma``, or a WAV file to clone.

Install with ``pip install 'voice-agent-next[mlx]'`` (macOS on Apple silicon). Every MLX
call runs on one shared worker thread, one generation step at a time, so a concurrent
speech recognizer is not blocked for a whole sentence (see :mod:`._mlx`). See
``docs/providers/mlx.md``.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Generator, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..audio.frame import AudioFrame
from ..audio.resample import StreamResampler
from ..errors import ConfigurationError, MissingDependencyError
from ..models import ModelFile, register_model
from ..registry import register_provider
from ..tts import TTS, ChunkedStream
from ..utils.clock import now
from ..utils.deps import is_installed, require
from ..utils.log import logger
from . import _mlx
from ._options import renamed

__all__ = ["DEFAULT_MODEL", "MODELS", "MLXAudioTTS", "kokoro_lang_code"]

_PROVIDER = "mlx_audio"
_EXTRA = "mlx"
_DEFAULT_RATE = 24_000
# mlx-audio's own download patterns (mlx_audio.utils.DEFAULT_ALLOW_PATTERNS), so that
# `van models download` fetches exactly what the provider loads
_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.py",
    "*.model",
    "*.tiktoken",
    "*.txt",
    "*.jinja",
    "*.jsonl",
    "*.yaml",
    "*.npz",
    "*.pth",
)
_SPACY_MODEL = "en_core_web_sm"
_SPACY_WHEEL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)

DEFAULT_MODEL = "kokoro"


@dataclass(frozen=True)
class _Model:
    repo: str
    sample_rate: int
    voice: str
    size: int
    license: str
    languages: str
    description: str


MODELS: dict[str, _Model] = {
    "kokoro": _Model(
        "mlx-community/Kokoro-82M-bf16",
        24_000,
        "af_heart",
        355_322_783,
        "Apache-2.0",
        "en, es, fr, hi, it, ja, pt, zh",
        "Kokoro-82M v1.0 (bfloat16), 54 voices",
    ),
    "pocket-tts": _Model(
        "mlx-community/pocket-tts",
        24_000,
        "alba",
        240_320_741,
        "CC-BY-4.0",
        "en",
        "Kyutai Pocket TTS 100M, audio streaming",
    ),
}
"""Known models: spec name -> MLX conversion on the Hugging Face Hub."""

_KOKORO_LANGS = frozenset("abefhijpz")


def kokoro_lang_code(voice: str) -> str:
    """Kokoro's language code from a voice name: ``"bf_emma"`` -> ``"b"`` (British)."""
    code = voice.strip()[:1].lower()
    return code if code in _KOKORO_LANGS else "a"


_KOKORO_LANG_BY_TAG = {
    "en": "a",
    "en-us": "a",
    "en-gb": "b",
    "es": "e",
    "fr": "f",
    "hi": "h",
    "it": "i",
    "ja": "j",
    "pt": "p",
    "pt-br": "p",
    "zh": "z",
    "cmn": "z",
}


def kokoro_lang_for(language: str) -> str:
    """Kokoro's language code for a language tag (``"en-GB"`` -> ``"b"``); Kokoro codes
    (``"a"``, ``"b"``...) and unknown tags pass through."""
    tag = language.strip().lower().replace("_", "-")
    return _KOKORO_LANG_BY_TAG.get(tag) or _KOKORO_LANG_BY_TAG.get(tag.split("-")[0]) or language


def _lookup(model: str) -> _Model | None:
    key = model.strip().lower().replace("_", "-")
    return MODELS.get(key) or MODELS.get(key.removesuffix("-82m"))


@register_provider(
    "tts",
    _PROVIDER,
    description="mlx-audio TTS (Kokoro, Pocket TTS...) on the Apple silicon GPU",
    default_model=DEFAULT_MODEL,
    models=tuple(MODELS),
    env=(),
    extra=_EXTRA,
    requires=("mlx_audio", "mlx"),
    local=True,
    platforms=_mlx.PLATFORMS,
)
class MLXAudioTTS(TTS):
    """Local speech synthesis on the Apple silicon GPU with mlx-audio.

    Args:
        model: ``kokoro``, ``pocket-tts``, an mlx-audio TTS repository id on the Hugging
            Face Hub, or a local model directory.
        voice: model voice (Kokoro: ``af_heart``...; Pocket TTS: ``alba``..., or a WAV
            file to clone). Default: the model's own default voice.
        speed: speaking rate multiplier (models that support it, e.g. Kokoro).
        language: the model's language: for Kokoro a language tag (``"en-us"``, ``"en-gb"``,
            ``"es"``, ``"fr"``, ``"ja"``...) or Kokoro code (``"a"``, ``"b"``...); default:
            from the voice's first letter. (``lang_code`` is a deprecated alias.)
        sample_rate: output rate for models not in :data:`MODELS` (default 24 kHz).
        streaming_interval: seconds of audio per chunk for audio-streaming models (Pocket
            TTS); smaller chunks arrive sooner.
        generate_options: extra keyword arguments for the model's ``generate()`` (e.g.
            ``{"temperature": 0.7}``).
        local_files_only: never download; also implied by ``VAN_OFFLINE=1``.
    """

    provider = _PROVIDER

    def __init__(
        self,
        *,
        model: str | None = None,
        voice: str | None = None,
        speed: float = 1.0,
        language: str | None = None,
        sample_rate: int | None = None,
        streaming_interval: float = 0.4,
        generate_options: Mapping[str, Any] | None = None,
        local_files_only: bool = False,
        lang_code: str | None = None,
    ) -> None:
        language = renamed("MLXAudioTTS", "language", language, "lang_code", lang_code)
        _mlx.ensure_available("mlx_audio", extra=_EXTRA, package="mlx-audio", provider=_PROVIDER)
        name = (model or DEFAULT_MODEL).strip()
        if speed <= 0:
            raise ConfigurationError(f"mlx_audio: speed must be > 0, got {speed}")
        if streaming_interval <= 0:
            raise ConfigurationError(
                f"mlx_audio: streaming_interval must be > 0, got {streaming_interval}"
            )
        known = _lookup(name)
        rate = known.sample_rate if known else (sample_rate or _DEFAULT_RATE)
        super().__init__(
            model=name,
            sample_rate=sample_rate or rate,
            voice=voice or (known.voice if known else None),
        )
        self.repo = known.repo if known else name
        self.is_kokoro = "kokoro" in self.repo.lower()
        self.speed = speed
        self.language = language
        self.streaming_interval = streaming_interval
        self.generate_options = dict(generate_options or {})
        self.local_files_only = local_files_only
        self.model_sample_rate: int | None = None
        """The loaded model's own output rate (``None`` until loaded)."""
        self._model: Any = None
        self._voice_dir: Path | None = None
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle
    async def warmup(self) -> None:
        """Download (if needed) and load the model, then synthesize a short phrase."""
        await _mlx.WORKER.run(self._warmup_sync)

    async def aclose(self) -> None:
        self._model = None

    def _warmup_sync(self) -> None:
        self._ensure_model()
        for _ in self._generate_sync("Hello.", self.voice):
            pass

    def _ensure_model(self) -> Any:
        model = self._model
        if model is None:
            with self._load_lock:
                model = self._model
                if model is None:
                    model = self._model = self._load()
        return model

    def _load(self) -> Any:
        _mlx.import_mlx(_PROVIDER)
        tts_mod = require("mlx_audio.tts", extra=_EXTRA, package="mlx-audio")
        if self.is_kokoro:
            self._check_kokoro_g2p(self.voice)
        t0 = now()
        path = _mlx.snapshot(
            self.repo,
            patterns=_PATTERNS,
            local_files_only=self.local_files_only,
            provider=_PROVIDER,
        )
        utils = require("mlx_audio.utils", extra=_EXTRA, package="mlx-audio")
        local = os.path.isdir(self.repo)
        try:
            # mlx-audio falls back to the model's name for the architecture when config.json
            # has no model_type (Kokoro): name it after the repository, not the snapshot dir
            parts = utils.get_model_name_parts(Path(self.repo) if local else self.repo)
            model = tts_mod.load(Path(path), model_name_parts=parts)
        except Exception as exc:
            raise _mlx.map_error(exc, _PROVIDER, f"loading model {self.model!r}") from exc
        if not local and getattr(model, "repo_id", False) is None:
            model.repo_id = self.repo  # Kokoro fetches voices from here (else another repo)
        self._voice_dir = Path(path) / "voices"
        rate = getattr(model, "sample_rate", None)
        self.model_sample_rate = int(rate) if rate else self.sample_rate
        logger.info(
            "mlx-audio %s loaded in %.2fs (%d Hz)", self.model, now() - t0, self.model_sample_rate
        )
        return model

    @property
    def lang_code(self) -> str | None:
        """Deprecated alias of :attr:`language`."""
        return self.language

    def _language(self, voice: str | None) -> str | None:
        if self.language:
            return kokoro_lang_for(self.language) if self.is_kokoro else self.language
        if self.is_kokoro:
            return kokoro_lang_code(voice or "af_heart")
        return None

    def _check_kokoro_g2p(self, voice: str | None) -> None:
        """Fail early, with the fix, where mlx-audio's Kokoro pipeline would fail late."""
        if not is_installed("misaki"):
            raise MissingDependencyError(
                "mlx_audio: Kokoro needs misaki (its grapheme-to-phoneme library), which "
                "supports Python < 3.13: pip install 'voice-agent-next[mlx]' on Python 3.11 "
                "or 3.12, or use mlx_audio/pocket-tts"
            )
        if self._language(voice) in ("a", "b") and not is_installed(_SPACY_MODEL):
            if is_installed("pip"):
                logger.info("mlx_audio: misaki will download spaCy's %s now", _SPACY_MODEL)
                return
            raise MissingDependencyError(
                f"mlx_audio: Kokoro's English G2P needs spaCy's {_SPACY_MODEL}, and this "
                f"environment has no pip to let misaki install it. Install it with: "
                f"uv pip install {_SPACY_WHEEL}"
            )

    # -------------------------------------------------------------- synthesis
    def _synthesize(self, text: str, *, voice: str | None) -> ChunkedStream:
        return _MLXAudioStream(self, text, voice=voice)

    def _generate_sync(
        self, text: str, voice: str | None
    ) -> Generator[np.ndarray[Any, Any], None, None]:
        """Float32 mono chunks at the model's rate (a generator: one step per ``next``)."""
        model = self._ensure_model()
        mx = _mlx.import_mlx(_PROVIDER)
        options: dict[str, Any] = {
            "voice": voice,
            "speed": self.speed,
            "stream": True,
            "streaming_interval": self.streaming_interval,
            "verbose": False,
        }
        lang = self._language(voice)
        if lang is not None:
            options["lang_code"] = lang
        if self.is_kokoro and voice and self._voice_dir is not None:
            # a voice of the downloaded snapshot: no Hub lookup (works with VAN_OFFLINE)
            local_voice = self._voice_dir / f"{voice}.safetensors"
            if local_voice.is_file():
                options["voice"] = str(local_voice)
        options.update(self.generate_options)
        try:
            for result in model.generate(text=text, **options):
                audio = result.audio
                if audio is None:
                    continue
                yield np.asarray(audio.astype(mx.float32), dtype=np.float32).reshape(-1)
        except Exception as exc:
            raise _mlx.map_error(exc, _PROVIDER, "synthesis") from exc


_DONE = object()


def _next_chunk(gen: Iterator[np.ndarray[Any, Any]]) -> Any:
    return next(gen, _DONE)


class _MLXAudioStream(ChunkedStream):
    async def _run(self) -> None:
        tts = self._tts
        assert isinstance(tts, MLXAudioTTS)
        text = self.text.strip()
        if not any(ch.isalnum() for ch in text):
            return  # nothing to pronounce
        await _mlx.WORKER.run(tts._ensure_model)
        source_rate = tts.model_sample_rate or tts.sample_rate
        resampler = StreamResampler(tts.sample_rate, 1) if source_rate != tts.sample_rate else None
        gen = tts._generate_sync(text, self.voice)
        try:
            while True:
                chunk = await _mlx.WORKER.run(_next_chunk, gen)
                if chunk is _DONE:
                    break
                frame = AudioFrame.from_numpy(chunk, source_rate)
                if resampler is not None:
                    frame = resampler.push(frame)
                if frame:
                    self._push_audio(frame)
            if resampler is not None:
                tail = resampler.flush()
                if tail:
                    self._push_audio(tail)
        finally:
            # also after a cancellation (barge-in): the generator is closed on the MLX
            # thread, between two generation steps
            _mlx.WORKER.submit(gen.close)


for _name, _info in MODELS.items():
    register_model(
        _PROVIDER,
        _name,
        kind="tts",
        files=[ModelFile.from_hf_repo(_info.repo, patterns=_PATTERNS, size=_info.size)],
        license=_info.license,
        languages=_info.languages,
        description=f"{_info.description} (MLX, {_info.repo})",
    )
