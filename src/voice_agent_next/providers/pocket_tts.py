"""Kyutai Pocket TTS: local, audio-streaming text-to-speech on CPU, with voice cloning.

Pocket TTS is a 100M-parameter model (CC-BY-4.0) with 24 kHz mono output, for English,
French, German, Portuguese, Italian, Spanish and Dutch. It generates 80 ms audio frames
autoregressively and decodes them while it generates the next ones, so the first audio
of a sentence is ready after roughly 100 ms on a desktop CPU, instead of after the whole
clause as with Kokoro.

Usage::

    from voice_agent_next import create

    tts = create("tts", "pocket-tts")                  # English, voice "alba"
    tts = create("tts", "pocket-tts/marius")           # another predefined voice
    tts = create("tts", "pocket-tts/french/estelle")   # language/voice
    tts = create("tts", "pocket-tts", voice="me.wav")  # clone a voice from a WAV prompt
    await tts.warmup()  # download + load the model and the voice ahead of the first request

``voice`` is a predefined voice name (see :data:`VOICES`), an audio file to clone
(WAV/MP3/FLAC...), an exported voice state (``.safetensors``) or an ``hf://`` /
``https://`` URL of either. Encoding a voice prompt takes a few seconds, so cloned voices
are cached in memory and on disk (as ``.safetensors`` under the model cache).

Cloning from audio needs the ``kyutai/pocket-tts`` weights, which are gated on Hugging
Face: accept the terms on the model page and log in (``hf auth login`` or ``HF_TOKEN``).
Without access, pocket-tts falls back to the ungated weights, which only support the
predefined voices and exported voice states.

Pocket TTS has no incremental text input and no text alignment, so :meth:`TTS.stream`
uses the base :class:`~voice_agent_next.tts.SentenceStreamAdapter` and the word timings
are estimates: once a segment is synthesized, its speech span is split over its words in
proportion to their length (see :func:`estimate_word_timings`).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from ..audio.frame import AudioFrame
from ..errors import ConfigurationError, ProviderError
from ..registry import register_provider
from ..stt import WordTiming
from ..text.sentences import SentenceSegmenter
from ..tts import TTS, ChunkedStream, SynthesizedAudio
from ..utils.clock import now
from ..utils.deps import is_installed, require
from ..utils.download import cache_dir
from ..utils.log import logger

__all__ = [
    "DEFAULT_LANGUAGE",
    "DEFAULT_VOICE",
    "LANGUAGES",
    "SAMPLE_RATE",
    "VOICES",
    "PocketTTS",
    "estimate_word_timings",
    "parse_model",
    "speech_bounds",
]

T = TypeVar("T")

SAMPLE_RATE = 24_000
DEFAULT_LANGUAGE = "english"
DEFAULT_VOICE = "alba"
_EXTRA = "pocket-tts"

LANGUAGES = ("english", "french", "german", "portuguese", "italian", "spanish", "dutch")
"""Languages with released weights. ``"<language>_24l"`` (24-layer variants, not for
English) and dated English releases such as ``"english_2026-04"`` are accepted too."""

VOICES = (
    "alba",
    "anna",
    "azelma",
    "bill_boerst",
    "caro_davy",
    "charles",
    "cosette",
    "daan",
    "eponine",
    "estelle",
    "eve",
    "fantine",
    "george",
    "giovanni",
    "jane",
    "javert",
    "jean",
    "juergen",
    "lola",
    "marius",
    "mary",
    "michael",
    "paul",
    "peter_yearsley",
    "rafael",
    "stuart_bell",
    "vera",
)
"""Predefined voices of pocket-tts 3.3 (precomputed states, available for every language)."""

_DEFAULT_VOICE_FOR_LANGUAGE = {
    "french": "estelle",
    "german": "juergen",
    "portuguese": "rafael",
    "italian": "giovanni",
    "spanish": "lola",
    "dutch": "daan",
}
_LANGUAGE_ALIASES = {
    "en": "english",
    "fr": "french",
    "de": "german",
    "pt": "portuguese",
    "it": "italian",
    "es": "spanish",
    "nl": "dutch",
}
_LANGUAGE_RE = re.compile(rf"^({'|'.join(LANGUAGES)})(_2\d{{3}}-\d\d)?(_24l)?$")
_REMOTE = ("hf://", "http://", "https://")
_SENTENCE_END = re.compile(r"[.!?…][\"'”’)\]]*$")
_CLAUSE_END = re.compile(r"[,;:—–-][\"'”’)\]]*$")
_VOICE_CACHE_SIZE = 8
"""Voice states kept in memory (~6-8 MB each)."""


def _normalize_language(name: str) -> str | None:
    key = name.strip().lower().replace("-", "_")
    key = _LANGUAGE_ALIASES.get(key, key)
    if key.startswith("english_2"):  # "english_2026_04" -> "english_2026-04"
        key = re.sub(r"^english_(2\d{3})_(\d\d)", r"english_\1-\2", key)
    return key if _LANGUAGE_RE.match(key) else None


def parse_model(model: str | None) -> tuple[str | None, str | None]:
    """Split a model id into ``(language, voice)``; either may be ``None``.

    ``"marius"`` -> ``(None, "marius")``, ``"french"`` -> ``("french", None)``,
    ``"french/estelle"`` -> ``("french", "estelle")``.
    """
    if not model or not model.strip():
        return None, None
    head, sep, tail = model.strip().partition("/")
    if sep:
        language = _normalize_language(head)
        if language is None:
            raise ConfigurationError(
                f"unknown Pocket TTS language {head!r}; known: {', '.join(LANGUAGES)}"
            )
        return language, tail.strip() or None
    language = _normalize_language(head)
    if language is not None:
        return language, None
    return None, head


def speech_bounds(samples: np.ndarray, *, threshold: float = 0.1) -> tuple[int, int] | None:
    """First and last sample index (exclusive end) louder than ``threshold`` x the peak.

    Returns ``None`` for silence.
    """
    if samples.size == 0:
        return None
    level = np.abs(samples)
    peak = float(level.max())
    if peak < 1e-3:
        return None
    loud = np.flatnonzero(level >= threshold * peak)
    return int(loud[0]), int(loud[-1]) + 1


def estimate_word_timings(text: str, start: float, end: float) -> list[WordTiming]:
    """Spread the words of ``text`` over ``[start, end]`` seconds.

    Each word gets time in proportion to its number of letters and digits plus one (the
    gap to the next word); punctuation after a word adds a pause (more for the end of a
    sentence than for a clause), which is left out of the word's own span.
    """
    words = text.split()
    if not words or end <= start:
        return []
    spans: list[tuple[str, float, float]] = []  # (word, spoken weight, pause weight)
    for word in words:
        spoken = sum(ch.isalnum() for ch in word) + 1.0
        pause = 3.0 if _SENTENCE_END.search(word) else 2.0 if _CLAUSE_END.search(word) else 0.0
        spans.append((word, spoken, pause))
    if spans[-1][2]:  # no pause after the last word: the speech span ends with it
        spans[-1] = (spans[-1][0], spans[-1][1], 0.0)
    scale = (end - start) / sum(s + p for _, s, p in spans)
    out: list[WordTiming] = []
    t = start
    for word, spoken, pause in spans:
        w_end = t + spoken * scale
        out.append(WordTiming(word, round(t, 4), round(min(w_end, end), 4)))
        t = w_end + pause * scale
    return out


def _check_installed() -> None:
    """Fail fast when the extra is missing, without importing torch."""
    for module, package in (("pocket_tts", "pocket-tts"), ("torch", "torch")):
        if not is_installed(module):
            require(module, extra=_EXTRA, package=package)


def _is_remote(voice: str) -> bool:
    return voice.startswith(_REMOTE)


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@register_provider(
    "tts",
    "pocket-tts",
    description="Kyutai Pocket TTS: local audio-streaming TTS on CPU with voice cloning (24 kHz)",
    default_model=DEFAULT_VOICE,
    models=VOICES,
    env=(),
    extra=_EXTRA,
    requires=("pocket_tts", "torch"),
    local=True,
    aliases=("pocket",),
)
class PocketTTS(TTS):
    """Kyutai Pocket TTS running locally with PyTorch on the CPU (24 kHz mono).

    Args:
        model: ``"<voice>"``, ``"<language>"`` or ``"<language>/<voice>"``, e.g.
            ``"marius"``, ``"french"`` or ``"german/juergen"``.
        voice: default voice, overriding the one in ``model``: a predefined voice name, an
            audio file to clone, an exported ``.safetensors`` voice state, or an
            ``hf://`` / ``https://`` URL of either. Default: ``"alba"`` for English, the
            pocket-tts default voice of the language otherwise.
        language: ``"english"`` (default), ``"french"``, ``"german"``, ``"portuguese"``,
            ``"italian"``, ``"spanish"``, ``"dutch"`` (or ISO codes ``"fr"``...), their
            ``"_24l"`` variants, or a dated English release (``"english_2026-04"``).
        config: custom pocket-tts YAML config (local path, URL or ``hf://``) instead of
            ``language``. Predefined voices do not work with custom weights.
        temperature: sampling temperature (default: the model's recommended value, 0.3).
        sampler_decode_steps: flow decoding steps (more: slower, sometimes better).
        eos_threshold: end-of-speech threshold (higher: the model speaks longer).
        frames_after_eos: 80 ms frames generated after the end of speech (default: chosen
            by pocket-tts from the text length).
        quantize: dynamic int8 quantization of the transformer (faster on x86, less RAM).
        num_threads: PyTorch intra-op threads. pocket-tts sets 1 on import (it decodes on
            a second thread); this is process-wide, so it also affects other torch models.
        split_sentences: synthesize long texts sentence by sentence (sooner first audio,
            per-sentence word timings).
        word_timings: attach estimated word timings to the audio (for word-exact
            truncation on barge-in).
        truncate_prompt: use only the first 30 s of a voice prompt.
        voice_cache_dir: where cloned voice states are cached (default: the model cache).
        clean_text: strip markdown/emoji before synthesis.
    """

    provider = "pocket-tts"

    def __init__(
        self,
        *,
        model: str | None = None,
        voice: str | os.PathLike[str] | None = None,
        language: str | None = None,
        config: str | os.PathLike[str] | None = None,
        temperature: float | None = None,
        sampler_decode_steps: int = 1,
        eos_threshold: float = -4.0,
        frames_after_eos: int | None = None,
        quantize: bool = False,
        num_threads: int | None = None,
        split_sentences: bool = True,
        word_timings: bool = True,
        truncate_prompt: bool = True,
        voice_cache_dir: str | os.PathLike[str] | None = None,
        clean_text: bool = True,
    ) -> None:
        model_language, model_voice = parse_model(model)
        if language is not None:
            lang = _normalize_language(language)
            if lang is None:
                raise ConfigurationError(
                    f"unknown Pocket TTS language {language!r}; known: {', '.join(LANGUAGES)}"
                )
            model_language = lang
        if config is not None and model_language is not None:
            raise ConfigurationError("pass either language or config to Pocket TTS, not both")
        if temperature is not None and temperature < 0:
            raise ConfigurationError(f"temperature must be >= 0, got {temperature}")
        if sampler_decode_steps < 1:
            raise ConfigurationError(
                f"sampler_decode_steps must be >= 1, got {sampler_decode_steps}"
            )
        if frames_after_eos is not None and frames_after_eos < 0:
            raise ConfigurationError(f"frames_after_eos must be >= 0, got {frames_after_eos}")
        if num_threads is not None and num_threads < 1:
            raise ConfigurationError(f"num_threads must be >= 1, got {num_threads}")
        _check_installed()
        self.language = None if config is not None else (model_language or DEFAULT_LANGUAGE)
        self.config = str(config) if config is not None else None
        base_language = (self.language or "").removesuffix("_24l").split("_2")[0]
        default_voice = str(voice) if voice is not None else model_voice
        if default_voice is None:
            default_voice = _DEFAULT_VOICE_FOR_LANGUAGE.get(base_language, DEFAULT_VOICE)
        super().__init__(
            model=self.language or Path(self.config or "custom").stem,
            sample_rate=SAMPLE_RATE,
            voice=default_voice,
            clean_text=clean_text,
        )
        self.temperature = temperature
        self.sampler_decode_steps = sampler_decode_steps
        self.eos_threshold = eos_threshold
        self.frames_after_eos = frames_after_eos
        self.quantize = quantize
        self.num_threads = num_threads
        self.split_sentences = split_sentences
        self.word_timings = word_timings
        self.truncate_prompt = truncate_prompt
        self._voice_cache_dir = (
            Path(voice_cache_dir).expanduser() if voice_cache_dir is not None else None
        )
        self._model: Any = None  # pocket_tts.TTSModel, created on the worker thread
        self._voices: OrderedDict[str, Any] = OrderedDict()  # voice -> model state
        self._executor: ThreadPoolExecutor | None = None
        self._active: set[threading.Event] = set()

    # ------------------------------------------------------------------ public API
    def _synthesize(self, text: str, *, voice: str | None) -> ChunkedStream:
        return _PocketChunkedStream(self, text, voice=voice)

    async def warmup(self) -> None:
        """Download (if needed) and load the model and the default voice, then run one
        short synthesis."""
        await self._submit(self._warmup_sync)

    async def load_voice(self, voice: str | os.PathLike[str]) -> None:
        """Encode (or load) a voice ahead of its first use, e.g. a WAV prompt to clone."""
        await self._submit(self._voice_state, str(voice))

    async def export_voice(
        self, voice: str | os.PathLike[str], dest: str | os.PathLike[str]
    ) -> Path:
        """Save the state of ``voice`` as ``.safetensors``: reloading it is much faster
        than encoding the audio prompt again, and needs no gated weights."""
        target = Path(dest).expanduser()

        def run() -> Path:
            state = self._voice_state(str(voice))
            pocket_tts = require("pocket_tts", extra=_EXTRA, package="pocket-tts")
            target.parent.mkdir(parents=True, exist_ok=True)
            pocket_tts.export_model_state(state, str(target))
            return target

        return await self._submit(run)

    async def list_voices(self) -> list[str]:
        """Names of the predefined voices."""
        return list(VOICES)

    async def aclose(self) -> None:
        """Stop the worker thread; a synthesis still running stops at its next frame."""
        for stop in list(self._active):
            stop.set()
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        self._model = None
        self._voices.clear()

    # ------------------------------------------------------------------ internals
    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            # the model is not thread-safe: one synthesis at a time, in submission order
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pocket-tts")
        return self._executor

    async def _submit(self, fn: Callable[..., T], *args: Any) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._get_executor(), fn, *args)

    def _segments(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if self.split_sentences:
            segmenter = SentenceSegmenter(min_chars=20, max_chars=250)
            parts = segmenter.push(text) + segmenter.flush()
        else:
            parts = [text]
        return [p for p in parts if any(ch.isalnum() for ch in p)]

    def _get_model(self) -> Any:
        """The loaded ``pocket_tts.TTSModel`` (worker thread only)."""
        if self._model is None:
            self._model = self._load()
        return self._model

    def _load(self) -> Any:
        pocket_tts = require("pocket_tts", extra=_EXTRA, package="pocket-tts")
        if self.num_threads is not None:
            torch = require("torch", extra=_EXTRA)
            torch.set_num_threads(self.num_threads)
        t0 = now()
        kwargs: dict[str, Any] = {
            "temp": self.temperature,
            "sampler_decode_steps": self.sampler_decode_steps,
            "eos_threshold": self.eos_threshold,
            "quantize": self.quantize,
        }
        if self.config is not None:
            kwargs["config"] = self.config
        else:
            kwargs["language"] = self.language
        try:
            model = pocket_tts.TTSModel.load_model(**kwargs)
        except (FileNotFoundError, ValueError) as exc:
            raise ConfigurationError(f"cannot load Pocket TTS model: {exc}") from exc
        except Exception as exc:
            raise ProviderError(
                f"failed to load Pocket TTS model: {exc}", provider=self.provider
            ) from exc
        logger.info(
            "pocket-tts: loaded %s in %.2f s (voice cloning: %s)",
            self.model,
            now() - t0,
            "yes" if getattr(model, "has_voice_cloning", False) else "no, gated weights",
        )
        return model

    def _voice_cache_path(self, digest: str) -> Path:
        root = self._voice_cache_dir or cache_dir() / "pocket-tts" / "voices"
        key = hashlib.sha256(f"{self.model}\0{self.config}\0{digest}".encode()).hexdigest()
        return root / f"{key[:32]}.safetensors"

    def _voice_state(self, voice: str) -> Any:
        """The model state for ``voice``, cached (worker thread only)."""
        state = self._voices.get(voice)
        if state is not None:
            self._voices.move_to_end(voice)
            return state
        model = self._get_model()
        t0 = now()
        state = self._load_voice(model, voice)
        logger.info("pocket-tts: voice %s ready in %.2f s", voice, now() - t0)
        self._voices[voice] = state
        while len(self._voices) > _VOICE_CACHE_SIZE:
            self._voices.popitem(last=False)
        return state

    def _load_voice(self, model: Any, voice: str) -> Any:
        if voice in VOICES or _is_remote(voice):
            return self._prompt(model, voice)
        path = Path(voice).expanduser()
        if not path.is_file():
            raise ConfigurationError(
                f"unknown Pocket TTS voice {voice!r}: not a predefined voice "
                f"({', '.join(VOICES)}) nor an existing audio or .safetensors file"
            )
        if path.suffix.lower() == ".safetensors":
            return self._prompt(model, path)
        cached = self._voice_cache_path(_file_digest(path))
        if cached.is_file():
            try:
                return self._prompt(model, cached)
            except ProviderError as exc:
                logger.warning("pocket-tts: ignoring unreadable cached voice %s (%s)", cached, exc)
        if not getattr(model, "has_voice_cloning", True):
            raise ConfigurationError(
                f"cannot clone {path.name}: voice cloning needs the gated kyutai/pocket-tts "
                "weights. Accept the terms on https://huggingface.co/kyutai/pocket-tts and "
                "log in (`hf auth login` or HF_TOKEN), or pass an exported .safetensors "
                "voice state"
            )
        state = self._prompt(model, path, truncate=self.truncate_prompt)
        self._save_voice(state, cached)
        return state

    def _prompt(self, model: Any, source: str | Path, truncate: bool = False) -> Any:
        try:
            return model.get_state_for_audio_prompt(source, truncate=truncate)
        except Exception as exc:
            raise ProviderError(
                f"cannot load Pocket TTS voice {source}: {exc}", provider=self.provider
            ) from exc

    def _save_voice(self, state: Any, target: Path) -> None:
        pocket_tts = require("pocket_tts", extra=_EXTRA, package="pocket-tts")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".voice-", suffix=".part")
            os.close(fd)
            try:
                pocket_tts.export_model_state(state, tmp)
                os.replace(tmp, target)  # atomic: concurrent processes never read half a file
            finally:
                Path(tmp).unlink(missing_ok=True)
        except Exception as exc:  # caching is an optimization: never fail the request
            logger.warning("pocket-tts: cannot cache voice state in %s: %s", target, exc)

    def _warmup_sync(self) -> None:
        t0 = now()
        for _ in self._generate("Hello.", self.voice or DEFAULT_VOICE, threading.Event()):
            pass
        logger.debug("pocket-tts: warm-up took %.2f s", now() - t0)

    def _generate(
        self, text: str, voice: str, stop: threading.Event
    ) -> Iterable[bytes | list[WordTiming]]:
        """Synthesize ``text`` (worker thread): yields s16le PCM chunks as they are
        decoded and, after each segment, its estimated word timings (seconds from the
        start of this text's audio)."""
        model = self._get_model()
        state = self._voice_state(voice)
        offset = 0  # samples emitted before the current segment
        for segment in self._segments(text):
            if stop.is_set():
                return
            chunks: list[np.ndarray] = []
            try:
                for chunk in model.generate_audio_stream(
                    state,
                    segment,
                    frames_after_eos=self.frames_after_eos,
                    copy_state=True,
                    stop=stop,
                ):
                    samples = np.asarray(chunk.detach().cpu().numpy(), dtype=np.float32)
                    samples = samples.reshape(-1)
                    if not samples.size:
                        continue
                    chunks.append(samples)
                    yield AudioFrame.from_numpy(samples, SAMPLE_RATE).data
            except Exception as exc:
                raise ProviderError(
                    f"Pocket TTS synthesis failed: {exc}", provider=self.provider
                ) from exc
            audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
            if self.word_timings and not stop.is_set():
                bounds = speech_bounds(audio)
                if bounds is not None:
                    start, end = ((offset + b) / SAMPLE_RATE for b in bounds)
                    words = estimate_word_timings(segment, start, end)
                    if words:
                        yield words
            offset += audio.size


_DONE = object()


class _PocketChunkedStream(ChunkedStream):
    async def _run(self) -> None:
        tts = self._tts
        assert isinstance(tts, PocketTTS)
        voice = self.voice or tts.voice or DEFAULT_VOICE
        loop = asyncio.get_running_loop()
        items: asyncio.Queue[Any] = asyncio.Queue()
        stop = threading.Event()

        def post(item: object) -> None:
            try:
                loop.call_soon_threadsafe(items.put_nowait, item)
            except RuntimeError:  # the event loop is closed: nobody is listening
                stop.set()

        def produce() -> None:
            if stop.is_set():  # cancelled while waiting for the worker
                post(_DONE)
                return
            try:
                for item in tts._generate(self.text, voice, stop):
                    post(item)
                    if stop.is_set():
                        break
            except BaseException as exc:
                post(exc)
            finally:
                post(_DONE)

        def dropped(job: asyncio.Future[None]) -> None:
            if job.cancelled():  # the TTS was closed before this request reached the worker
                items.put_nowait(ProviderError("Pocket TTS was closed", provider=tts.provider))
                items.put_nowait(_DONE)

        tts._active.add(stop)
        loop.run_in_executor(tts._get_executor(), produce).add_done_callback(dropped)
        try:
            while True:
                item = await items.get()
                if item is _DONE:
                    break
                if isinstance(item, BaseException):
                    raise item
                if isinstance(item, bytes):
                    self._push_audio(item)
                else:  # word timings of the segment that just ended
                    empty = AudioFrame.empty(tts.sample_rate, tts.channels)
                    self._send(
                        SynthesizedAudio(empty, self._request_id, self._segment_id, words=item)
                    )
        finally:
            stop.set()  # cancelled: the generation stops at its next frame
            tts._active.discard(stop)
