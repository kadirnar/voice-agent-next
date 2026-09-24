"""Resemble AI Chatterbox: local text-to-speech with zero-shot voice cloning (MIT).

Models (24 kHz mono, all with voice cloning from a reference WAV):

* ``turbo`` (default): 350M-parameter English model built for voice agents, with a
  one-step speech-token-to-mel decoder and paralinguistic tags (``[laugh]``,
  ``[chuckle]``, ``[cough]``...). ~3 GB download, ~4 GB of VRAM.
* ``nano``: 110M-parameter version of Turbo, fast enough for a CPU. ~2 GB download.
  Needs a chatterbox-tts release newer than 0.1.7 (see ``docs/providers/chatterbox.md``).
* ``multilingual``: 500M-parameter model for 23 languages (``language="fr"``...), with
  ``exaggeration`` and ``cfg_weight`` controls.

Usage::

    from voice_agent_next import create

    tts = create("tts", "chatterbox")                         # Turbo, built-in voice
    tts = create("tts", "chatterbox/nano", device="cpu")
    tts = create("tts", "chatterbox", voice="me.wav")         # clone (> 5 s of speech)
    tts = create("tts", "chatterbox/multilingual", language="fr")
    await tts.warmup()  # download + load the model ahead of the first request

Chatterbox renders a whole sentence at once (no audio streaming), so :meth:`TTS.stream`
uses the sentence adapter and long texts are synthesized sentence by sentence; word
timings are estimates, spread over each sentence's speech span. Every output carries
Resemble AI's imperceptible Perth watermark, as upstream does.
"""

from __future__ import annotations

import inspect
import os
import threading
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import ConfigurationError, ProviderError
from ..registry import register_provider
from ..utils.clock import now
from ..utils.deps import is_installed, require
from ..utils.log import logger
from ._torch_tts import LocalTorchTTS, stop_on

__all__ = ["LANGUAGES", "MODELS", "REPOS", "SAMPLE_RATE", "ChatterboxTTS"]

SAMPLE_RATE = 24_000
_EXTRA = "chatterbox"
MODELS = ("turbo", "nano", "multilingual")
REPOS = {
    "turbo": "ResembleAI/chatterbox-turbo",
    "nano": "ResembleAI/chatterbox-nano",
    "multilingual": "ResembleAI/chatterbox",
}
"""Hugging Face repositories of the models."""
_T3_WEIGHTS = {"turbo": "t3_turbo_v1.safetensors", "nano": "t3_nano_v1.safetensors"}
_ALIASES = {"chatterbox-turbo": "turbo", "chatterbox-nano": "nano", "mtl": "multilingual"}
LANGUAGES = (
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
    "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
)  # fmt: skip
"""Languages of the multilingual model (ISO 639-1)."""
_VOICE_CACHE_SIZE = 8


def _check_installed() -> None:
    """Fail fast when the extra is missing, without importing torch."""
    for module, package in (("chatterbox", "chatterbox-tts"), ("torch", "torch")):
        if not is_installed(module):
            require(module, extra=_EXTRA, package=package)


def _parse_model(model: str | None) -> str:
    name = (model or "turbo").strip().lower()
    name = _ALIASES.get(name, name)
    if name not in MODELS:
        raise ConfigurationError(f"unknown Chatterbox model {model!r}; known: {', '.join(MODELS)}")
    return name


@register_provider(
    "tts",
    "chatterbox",
    description="Resemble AI Chatterbox Turbo/Nano/Multilingual: local TTS with voice cloning",
    default_model="turbo",
    models=MODELS,
    env=(),
    extra=_EXTRA,
    requires=("chatterbox", "torch"),
    local=True,
)
class ChatterboxTTS(LocalTorchTTS):
    """Chatterbox running locally with PyTorch (CUDA, Apple MPS or CPU; 24 kHz mono).

    Args:
        model: ``"turbo"`` (default), ``"nano"`` or ``"multilingual"``.
        voice: default voice: a reference audio file to clone (WAV/MP3/FLAC..., more than
            5 s of clean speech; 10 s works best). Default: the model's built-in voice.
        language: language of the multilingual model (ISO 639-1, default ``"en"``).
        device: ``"auto"`` (CUDA, then Apple MPS, then CPU; see
            :func:`~voice_agent_next.hardware.select_torch_backend`), ``"cuda"``,
            ``"cuda:<n>"``, ``"mps"`` or ``"cpu"``.
        model_path: load the weights from this directory instead of Hugging Face.
        temperature: sampling temperature (default 0.8).
        top_p: nucleus sampling (Turbo/Nano default 0.95, multilingual 1.0).
        top_k: top-k sampling (Turbo/Nano only, default 1000).
        repetition_penalty: Turbo/Nano default 1.2, multilingual 2.0.
        exaggeration: emotion intensity of the multilingual model (default 0.5).
        cfg_weight: classifier-free guidance of the multilingual model (default 0.5;
            lower for faster speakers or a cross-language reference voice).
        norm_loudness: normalize a cloned reference to -27 LUFS (Turbo/Nano).
        split_sentences: synthesize long texts sentence by sentence (sooner first audio,
            per-sentence word timings).
        word_timings: attach estimated word timings to the audio (for word-exact
            truncation on barge-in).
        clean_text: strip markdown/emoji before synthesis.
    """

    provider = "chatterbox"
    _thread_name = "chatterbox"

    def __init__(
        self,
        *,
        model: str | None = None,
        voice: str | os.PathLike[str] | None = None,
        language: str | None = None,
        device: str = "auto",
        model_path: str | os.PathLike[str] | None = None,
        temperature: float = 0.8,
        top_p: float | None = None,
        top_k: int = 1000,
        repetition_penalty: float | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        norm_loudness: bool = True,
        split_sentences: bool = True,
        word_timings: bool = True,
        clean_text: bool = True,
    ) -> None:
        variant = _parse_model(model)
        if language is not None:
            language = language.strip().lower()
            if variant != "multilingual" and language != "en":
                raise ConfigurationError(
                    f"Chatterbox {variant} is English only; use model='multilingual' "
                    f"for language={language!r}"
                )
            if language not in LANGUAGES:
                raise ConfigurationError(
                    f"unknown Chatterbox language {language!r}; known: {', '.join(LANGUAGES)}"
                )
        if temperature <= 0:
            raise ConfigurationError(f"temperature must be > 0, got {temperature}")
        if top_p is not None and not 0 < top_p <= 1:
            raise ConfigurationError(f"top_p must be in (0, 1], got {top_p}")
        if top_k < 1:
            raise ConfigurationError(f"top_k must be >= 1, got {top_k}")
        _check_installed()
        super().__init__(
            model=variant,
            sample_rate=SAMPLE_RATE,
            voice=str(voice) if voice is not None else None,
            device=device,
            split_sentences=split_sentences,
            word_timings=word_timings,
            clean_text=clean_text,
        )
        self.language = language or "en"
        self.model_path = Path(model_path).expanduser() if model_path is not None else None
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.norm_loudness = norm_loudness
        self._default_conds: Any = None
        self._voices: OrderedDict[str, Any] = OrderedDict()  # voice -> Conditionals

    async def load_voice(self, voice: str | os.PathLike[str]) -> None:
        """Encode a reference voice ahead of its first use."""

        def run() -> None:
            self._conditionals(self._get_model(), str(voice))

        await self._submit(run)

    async def list_voices(self) -> list[str]:
        """Chatterbox has one built-in voice per model; others are cloned from audio."""
        return []

    async def aclose(self) -> None:
        await super().aclose()
        self._voices.clear()
        self._default_conds = None

    # ------------------------------------------------------------------ internals
    def _load(self, device: str) -> Any:
        if self.model == "multilingual":
            module = require("chatterbox.mtl_tts", extra=_EXTRA, package="chatterbox-tts")
            cls = module.ChatterboxMultilingualTTS
            if self.model_path is not None:
                tts = cls.from_local(self.model_path, device)
            else:
                tts = cls.from_pretrained(device=device)
        else:
            module = require("chatterbox.tts_turbo", extra=_EXTRA, package="chatterbox-tts")
            cls = module.ChatterboxTurboTTS
            nano = self.model == "nano"
            if nano and "nano" not in inspect.signature(cls.from_local).parameters:
                raise ConfigurationError(
                    "Chatterbox-Nano needs a chatterbox-tts release newer than 0.1.7: "
                    "pip install -U 'chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox'"
                )
            path = self.model_path or self._download(self.model)
            tts = cls.from_local(path, device, nano=True) if nano else cls.from_local(path, device)
        _quiet_progress_bars()
        _float32_loudness(tts)
        self._default_conds = tts.conds
        self._voices.clear()
        return tts

    def _download(self, variant: str) -> Path:
        """Only the files the model loads: upstream's ``from_pretrained`` also fetches the
        unused 1 GB multi-step decoder."""
        hub = require("huggingface_hub", extra=_EXTRA)
        t0 = now()
        path = hub.snapshot_download(
            repo_id=REPOS[variant],
            allow_patterns=[
                _T3_WEIGHTS[variant],
                "s3gen_meanflow.safetensors",
                "ve.safetensors",
                "conds.pt",
                "*.json",
                "*.txt",
            ],
            token=os.environ.get("HF_TOKEN") or None,
        )
        logger.debug("chatterbox: %s files ready in %.1f s", variant, now() - t0)
        return Path(path)

    def _conditionals(self, model: Any, voice: str | None) -> Any:
        """The conditioning of ``voice`` (worker thread; cloned voices are cached)."""
        if not voice:
            if self._default_conds is None:
                raise ConfigurationError(
                    f"Chatterbox {self.model} has no built-in voice: pass voice=<reference.wav>"
                )
            return self._default_conds
        cached = self._voices.get(voice)
        if cached is not None:
            self._voices.move_to_end(voice)
            return cached
        path = Path(voice).expanduser()
        if not path.is_file():
            raise ConfigurationError(
                f"Chatterbox voice {voice!r} is not an audio file (voices are cloned from a "
                "reference recording)"
            )
        t0 = now()
        try:
            if self.model == "multilingual":
                model.prepare_conditionals(str(path), exaggeration=self.exaggeration)
            else:
                model.prepare_conditionals(str(path), norm_loudness=self.norm_loudness)
        except AssertionError as exc:  # "Audio prompt must be longer than 5 seconds!"
            raise ConfigurationError(f"cannot clone {path.name}: {exc}") from exc
        except Exception as exc:
            raise ProviderError(
                f"cannot encode Chatterbox voice {path.name}: {exc}", provider=self.provider
            ) from exc
        logger.info("chatterbox: voice %s ready in %.2f s", path.name, now() - t0)
        conds = model.conds
        self._voices[voice] = conds
        while len(self._voices) > _VOICE_CACHE_SIZE:
            self._voices.popitem(last=False)
        return conds

    def _render(
        self, model: Any, text: str, voice: str | None, stop: threading.Event
    ) -> Iterable[np.ndarray]:
        model.conds = self._conditionals(model, voice)
        kwargs: dict[str, Any] = {"temperature": self.temperature}
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.repetition_penalty is not None:
            kwargs["repetition_penalty"] = self.repetition_penalty
        if self.model == "multilingual":
            kwargs.update(
                language_id=self.language,
                exaggeration=self.exaggeration,
                cfg_weight=self.cfg_weight,
            )
        else:
            kwargs.update(top_k=self.top_k, norm_loudness=self.norm_loudness)
        with stop_on(model.t3.tfmr, stop):
            wav = model.generate(text, **kwargs)
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        yield np.asarray(wav, dtype=np.float32).reshape(-1)


def _float32_loudness(tts: Any) -> None:
    """Turbo's reference loudness normalization multiplies float32 audio by a NumPy float64
    gain: float64 under NumPy 2 (NEP 50), which the speech tokenizer rejects. Keep it
    float32, as it was under the NumPy 1 that chatterbox-tts pins."""
    normalize = getattr(tts, "norm_loudness", None)
    if not callable(normalize):
        return

    def norm_loudness(wav: Any, sr: int, *args: Any, **kwargs: Any) -> Any:
        return np.asarray(normalize(wav, sr, *args, **kwargs), dtype=np.float32)

    tts.norm_loudness = norm_loudness


def _quiet(*args: Any, **kwargs: Any) -> None:
    return None


def _passthrough(iterable: Any, *args: Any, **kwargs: Any) -> Any:
    return iterable


def _quiet_progress_bars() -> None:
    """chatterbox prints a line and draws tqdm bars for every sentence: no place for them
    in a voice agent's console (module globals shadow the builtins/imports they use)."""
    for name in ("chatterbox.models.t3.t3", "chatterbox.models.s3gen.flow_matching"):
        try:
            module = require(name, extra=_EXTRA, package="chatterbox-tts")
        except Exception:
            continue
        if hasattr(module, "tqdm"):
            setattr(module, "tqdm", _passthrough)  # noqa: B010
        setattr(module, "print", _quiet)  # noqa: B010
