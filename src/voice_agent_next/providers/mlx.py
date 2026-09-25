"""Parakeet on MLX (parakeet-mlx): streaming speech recognition on Apple silicon GPUs.

``create("stt", "mlx/parakeet-tdt-0.6b-v3")`` (the default model; alias
``"parakeet-mlx"``). NVIDIA's Parakeet TDT 0.6B v3 transcribes 25 European languages
with punctuation and casing; ``parakeet-tdt-0.6b-v2`` is the English-only predecessor.
Any Parakeet checkpoint converted for parakeet-mlx works: a name from :data:`MODELS`, a
Hugging Face repository id or a local directory with ``config.json`` and
``model.safetensors``. Whisper models run through :mod:`~voice_agent_next.providers.mlx_whisper`
(``mlx_whisper/large-v3-turbo``).

Streaming uses parakeet-mlx's cache-aware ``StreamingParakeet`` (local attention with a
rotating key/value cache): audio is fed every ``chunk_duration`` seconds, finalized and
draft tokens come back as interim transcripts, and :meth:`STTStream.flush` (the cascade
calls it at the end of the user's turn) returns the final transcript of the utterance at
once — no second pass over the audio. Each flushed utterance starts a new recognizer
state. Batch calls (:meth:`STT.transcribe`) run the full model over the utterance.

MLX needs macOS on Apple silicon: install with ``pip install 'voice-agent-next[mlx]'``.
Every MLX call runs on one shared worker thread (see :mod:`._mlx`). See
``docs/providers/mlx.md``.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ..audio.frame import AudioFrame
from ..errors import ConfigurationError
from ..models import ModelFile, register_model
from ..registry import register_provider
from ..stt import STT, STTCapabilities, STTEvent, STTEventType, STTStream, Transcript, WordTiming
from ..utils.aio import ChanClosed
from ..utils.clock import now
from ..utils.deps import require
from ..utils.ids import new_id
from ..utils.log import logger
from . import _mlx

__all__ = ["DEFAULT_MODEL", "MODELS", "ParakeetMLXSTT"]

_PROVIDER = "mlx"
_EXTRA = "mlx"
_SAMPLE_RATE = 16_000
_FILES = ("config.json", "model.safetensors")
_DTYPES = ("bfloat16", "float16", "float32")
_MIN_TAIL = _SAMPLE_RATE // 20
_PACING = 2.0
"""Queue at least this many times the last step's duration of audio between steps."""
"""Shorter audio left at a flush is not fed (50 ms: a few mel windows)."""

DEFAULT_MODEL = "parakeet-tdt-0.6b-v3"


@dataclass(frozen=True)
class _Model:
    repo: str
    size: int
    languages: str
    description: str


MODELS: dict[str, _Model] = {
    "parakeet-tdt-0.6b-v3": _Model(
        "mlx-community/parakeet-tdt-0.6b-v3",
        2_508_532_829,
        "25 European languages",
        "Parakeet TDT 0.6B v3: multilingual, punctuation and casing",
    ),
    "parakeet-tdt-0.6b-v2": _Model(
        "mlx-community/parakeet-tdt-0.6b-v2",
        2_471_596_080,
        "en",
        "Parakeet TDT 0.6B v2: English, punctuation and casing",
    ),
    "parakeet-tdt-1.1b": _Model(
        "mlx-community/parakeet-tdt-1.1b", 4_282_297_109, "en", "Parakeet TDT 1.1B (English)"
    ),
    "parakeet-tdt_ctc-1.1b": _Model(
        "mlx-community/parakeet-tdt_ctc-1.1b",
        4_286_517_890,
        "en",
        "Parakeet TDT-CTC 1.1B (English)",
    ),
    "parakeet-tdt_ctc-110m": _Model(
        "mlx-community/parakeet-tdt_ctc-110m",
        458_690_629,
        "en",
        "Parakeet TDT-CTC 110M: the smallest Parakeet (English)",
    ),
    "parakeet-rnnt-0.6b": _Model(
        "mlx-community/parakeet-rnnt-0.6b", 2_467_092_634, "en", "Parakeet RNN-T 0.6B (English)"
    ),
    "parakeet-ctc-0.6b": _Model(
        "mlx-community/parakeet-ctc-0.6b", 2_435_527_077, "en", "Parakeet CTC 0.6B (English)"
    ),
}
"""Known models: spec name -> MLX conversion on the Hugging Face Hub (float32 weights,
cast to ``dtype`` when loading)."""


def _resolve(model: str) -> str:
    """Spec model name -> repository id or local path."""
    key = model.strip()
    if key in MODELS:
        return MODELS[key].repo
    short = key.removeprefix("nvidia/")
    if short in MODELS:
        return MODELS[short].repo
    return key


@register_provider(
    "stt",
    _PROVIDER,
    description="Parakeet on MLX (parakeet-mlx), streaming, Apple silicon GPU",
    default_model=DEFAULT_MODEL,
    models=tuple(MODELS),
    env=(),
    extra=_EXTRA,
    requires=("parakeet_mlx", "mlx"),
    local=True,
    platforms=_mlx.PLATFORMS,
    aliases=("parakeet_mlx",),
)
class ParakeetMLXSTT(STT):
    """Parakeet speech recognition on the Apple silicon GPU with parakeet-mlx.

    Args:
        model: a name from :data:`MODELS` (``parakeet-tdt-0.6b-v3``...), a Hugging Face
            repository id or a local model directory.
        language: reported on transcripts. Parakeet v3 detects the language itself; the
            other models are English-only.
        streaming: ``True`` streams with ``StreamingParakeet`` (interim results, final
            transcript at once on :meth:`~voice_agent_next.stt.STTStream.flush`);
            ``False`` declares a batch recognizer, which the cascade wraps in a
            :class:`~voice_agent_next.stt.StreamAdapter` (one full pass per utterance).
        interim_results: emit ``INTERIM_TRANSCRIPT`` events while streaming.
        chunk_duration: seconds of audio fed to the streaming recognizer at a time. Each
            step re-encodes the attention context, so tiny chunks cost more GPU time.
        context_size: ``(left, right)`` attention context of streaming, in encoder frames
            (80 ms each).
        depth: encoder layers whose cache is carried exactly across chunks (streaming).
        dtype: ``"bfloat16"`` (default), ``"float16"`` or ``"float32"``.
        beam_size: 1 is greedy decoding (lowest latency); more runs beam search.
        local_files_only: never download; also implied by ``VAN_OFFLINE=1``.
    """

    provider = _PROVIDER

    def __init__(
        self,
        *,
        model: str | None = None,
        language: str | None = None,
        streaming: bool = True,
        interim_results: bool = True,
        chunk_duration: float = 0.32,
        context_size: tuple[int, int] | Sequence[int] = (256, 256),
        depth: int = 1,
        dtype: str = "bfloat16",
        beam_size: int = 1,
        local_files_only: bool = False,
    ) -> None:
        _mlx.ensure_available(
            "parakeet_mlx", extra=_EXTRA, package="parakeet-mlx", provider=_PROVIDER
        )
        name = (model or DEFAULT_MODEL).strip()
        if "whisper" in name.lower():
            raise ConfigurationError(
                f"mlx: {name!r} is a Whisper model; use mlx_whisper/{name} "
                "(pip install 'voice-agent-next[mlx-whisper]')"
            )
        if dtype not in _DTYPES:
            raise ConfigurationError(f"mlx: dtype must be one of {_DTYPES}, got {dtype!r}")
        if chunk_duration <= 0:
            raise ConfigurationError(f"mlx: chunk_duration must be > 0, got {chunk_duration}")
        if beam_size < 1:
            raise ConfigurationError(f"mlx: beam_size must be >= 1, got {beam_size}")
        left, right = (int(v) for v in context_size)
        if left < 1 or right < 1 or depth < 1:
            raise ConfigurationError("mlx: context_size and depth must be positive")
        super().__init__(
            model=name,
            capabilities=STTCapabilities(
                streaming=streaming,
                interim_results=streaming and interim_results,
                word_timestamps=True,
                language_detection=False,
            ),
            sample_rate=_SAMPLE_RATE,
            language=language,
        )
        self.chunk_duration = chunk_duration
        self.context_size = (left, right)
        self.depth = depth
        self.dtype = dtype
        self.beam_size = beam_size
        self.local_files_only = local_files_only
        self._model: Any = None
        self._load_lock = threading.Lock()
        self._open_streams = 0
        """Streams using local attention (touched on the MLX worker thread only)."""

    # ------------------------------------------------------------------ lifecycle
    async def warmup(self) -> None:
        """Download (if needed) and load the model, then run a short warm-up inference."""
        await _mlx.WORKER.run(self._ensure_model)

    async def aclose(self) -> None:
        self._model = None  # MLX frees the weights with the last reference

    def _ensure_model(self) -> Any:
        model = self._model
        if model is None:
            with self._load_lock:
                model = self._model
                if model is None:
                    model = self._model = self._load()
        return model

    def _load(self) -> Any:
        mx = _mlx.import_mlx(_PROVIDER)
        pm = require("parakeet_mlx", extra=_EXTRA, package="parakeet-mlx")
        t0 = now()
        path = _mlx.snapshot(
            _resolve(self.model),
            patterns=_FILES,
            local_files_only=self.local_files_only,
            provider=_PROVIDER,
        )
        try:
            model = pm.from_pretrained(path, dtype=getattr(mx, self.dtype))
            self._generate(model, np.zeros(_SAMPLE_RATE // 2, dtype=np.float32))  # warm-up
        except Exception as exc:
            raise _mlx.map_error(exc, _PROVIDER, f"loading model {self.model!r}") from exc
        logger.info("parakeet-mlx %s loaded in %.2fs", self.model, now() - t0)
        return model

    # ------------------------------------------------------------------- helpers
    def _decoding_config(self) -> Any:
        pm = require("parakeet_mlx", extra=_EXTRA, package="parakeet-mlx")
        decoding = pm.Greedy() if self.beam_size == 1 else pm.Beam(beam_size=self.beam_size)
        return pm.DecodingConfig(decoding=decoding)

    def _audio(self, samples: npt.NDArray[np.float32]) -> Any:
        # float32 like parakeet-mlx's own load_audio(): get_logmel() views the STFT's complex
        # output in the input dtype, so bfloat16 audio would double the frequency bins
        mx = _mlx.import_mlx(_PROVIDER)
        return mx.array(np.ascontiguousarray(samples, dtype=np.float32))

    def _generate(self, model: Any, samples: npt.NDArray[np.float32]) -> Any:
        audio_mod = require("parakeet_mlx.audio", extra=_EXTRA, package="parakeet-mlx")
        mel = audio_mod.get_logmel(self._audio(samples), model.preprocessor_config)
        return model.generate(mel, decoding_config=self._decoding_config())[0]

    def _transcript(
        self, result: Any, offset: float = 0.0, language: str | None = None
    ) -> Transcript:
        return _to_transcript(result, offset, language or self.language)

    # ---------------------------------------------------------------- recognition
    async def _recognize(self, audio: AudioFrame, *, language: str | None) -> Transcript:
        # STT.transcribe() has already resampled the audio to 16 kHz mono.
        samples = audio.to_float32()
        return await _mlx.WORKER.run(self._recognize_sync, samples, language)

    def _recognize_sync(self, samples: npt.NDArray[np.float32], language: str | None) -> Transcript:
        if samples.size < _SAMPLE_RATE // 100:  # under one 10 ms hop: nothing to decode
            return Transcript(text="", language=language or self.language)
        model = self._ensure_model()
        try:
            result = self._generate(model, samples)
        except Exception as exc:
            raise _mlx.map_error(exc, _PROVIDER, "transcription") from exc
        return self._transcript(result, 0.0, language)

    def _create_stream(self, *, language: str | None) -> STTStream:
        return _ParakeetStream(self, language=language)

    # ------------------------------------------------- streaming (worker thread)
    def _open_streamer(self) -> Any:
        """A new ``StreamingParakeet``; the first open stream switches to local attention."""
        model = self._ensure_model()
        pm = require("parakeet_mlx", extra=_EXTRA, package="parakeet-mlx")
        if self._open_streams == 0:
            model.encoder.set_attention_model("rel_pos_local_attn", self.context_size)
        self._open_streams += 1
        # keep_original_attention: this provider switches the attention itself, once for
        # all concurrent streams (StreamingParakeet.__exit__ would reset it under them).
        return pm.StreamingParakeet(
            model,
            self.context_size,
            self.depth,
            keep_original_attention=True,
            decoding_config=self._decoding_config(),
        )

    def _close_streamer(self) -> None:
        self._open_streams = max(0, self._open_streams - 1)
        model = self._model
        if self._open_streams == 0 and model is not None:
            model.encoder.set_attention_model("rel_pos")


def _words(tokens: Sequence[Any], offset: float) -> list[WordTiming]:
    """Merge sub-word tokens (a leading space starts a word) into words."""
    words: list[WordTiming] = []
    parts: list[Any] = []

    def close() -> None:
        text = "".join(t.text for t in parts).strip()
        if text:
            conf = [float(t.confidence) for t in parts]
            words.append(
                WordTiming(
                    word=text,
                    start=offset + float(parts[0].start),
                    end=offset + float(parts[-1].start) + float(parts[-1].duration),
                    confidence=float(np.exp(np.mean(np.log(np.asarray(conf) + 1e-10)))),
                )
            )

    for token in tokens:
        if parts and token.text.startswith(" "):
            close()
            parts = []
        parts.append(token)
    if parts:
        close()
    return words


def _to_transcript(result: Any, offset: float, language: str | None) -> Transcript:
    tokens = [t for s in result.sentences for t in s.tokens]
    text = str(result.text).strip()
    if not tokens:
        return Transcript(text=text, language=language)
    words = _words(tokens, offset)
    confidence = float(np.exp(np.mean(np.log([float(t.confidence) + 1e-10 for t in tokens]))))
    return Transcript(
        text=text,
        language=language,
        confidence=min(1.0, confidence),
        start_time=offset + float(tokens[0].start),
        end_time=offset + float(tokens[-1].start) + float(tokens[-1].duration),
        words=words,
    )


class _ParakeetStream(STTStream):
    """One ``StreamingParakeet`` per utterance; a flush finalizes it and starts the next."""

    def __init__(self, stt: ParakeetMLXSTT, *, language: str | None) -> None:
        self._parakeet = stt
        self._streamer: Any = None
        self._pending: list[npt.NDArray[np.float32]] = []
        self._pending_samples = 0
        self._offset = 0
        """Input samples before the current utterance."""
        self._fed = 0
        """Samples fed into the current utterance."""
        self._segment_id = new_id("seg_")
        self._partial = ""
        self._speaking = False
        self._chunk = max(1, round(stt.chunk_duration * _SAMPLE_RATE))
        self._step_time = 0.0
        """Duration of the last streaming step on the MLX thread (seconds)."""
        super().__init__(stt, language=language)

    # -------------------------------------------------------------- event loop
    async def _run(self) -> None:
        stt = self._parakeet
        try:
            await _mlx.WORKER.run(stt._ensure_model)
            while True:
                try:
                    item = await self._input.recv()
                except ChanClosed:
                    break
                flush = self.is_flush(item)
                if not flush:
                    assert isinstance(item, AudioFrame)
                    self._queue(item)
                while not flush:  # batch whatever else is queued
                    try:
                        nxt = self._input.recv_nowait()
                    except (asyncio.QueueEmpty, ChanClosed):
                        break
                    if self.is_flush(nxt):
                        flush = True
                    else:
                        assert isinstance(nxt, AudioFrame)
                        self._queue(nxt)
                if self._pending_samples >= self._threshold or (flush and self._pending):
                    self._update(await _mlx.WORKER.run(self._feed_sync, flush))
                if flush:
                    self._finish(await _mlx.WORKER.run(self._finalize_sync))
        finally:
            if self._streamer is not None:
                self._streamer = None
                await _mlx.WORKER.run(stt._close_streamer)

    @property
    def _threshold(self) -> int:
        """Samples to queue before the next step: ``chunk_duration``, or more when a step
        takes longer than that. Every step re-encodes the right-context window, so this keeps
        the MLX thread at most about half busy: a flush then rarely waits for a step
        in flight, and the final transcript costs one step."""
        return max(self._chunk, round(self._step_time * _PACING * _SAMPLE_RATE))

    def _queue(self, frame: AudioFrame) -> None:
        samples = frame.to_float32()
        self._pending.append(samples)
        self._pending_samples += len(samples)

    def _update(self, result: Any) -> None:
        if result is None:
            return
        text = str(result.text).strip()
        if text and not self._speaking:
            self._speaking = True
            self._emit(STTEvent(STTEventType.START_OF_SPEECH, segment_id=self._segment_id))
        if text and text != self._partial and self._parakeet.capabilities.interim_results:
            self._partial = text
            self._emit(
                STTEvent(
                    STTEventType.INTERIM_TRANSCRIPT,
                    self._parakeet._transcript(result, self._offset_s, self._language),
                    self._segment_id,
                )
            )

    def _finish(self, result: Any) -> None:
        """Final transcript (even an empty one: the cascade waits for it after a flush),
        framed by START/END_OF_SPEECH when the utterance had any text."""
        segment = self._segment_id
        if result is None:
            transcript = Transcript(text="", language=self._language or self._parakeet.language)
        else:
            transcript = self._parakeet._transcript(result, self._offset_s, self._language)
        if transcript.text and not self._speaking:
            self._speaking = True
            self._emit(STTEvent(STTEventType.START_OF_SPEECH, segment_id=segment))
        self._emit(STTEvent(STTEventType.FINAL_TRANSCRIPT, transcript, segment))
        if self._speaking:
            start, stop = transcript.start_time, transcript.end_time
            end = Transcript(text="", start_time=start, end_time=stop)
            self._emit(STTEvent(STTEventType.END_OF_SPEECH, end, segment))
        self._offset += self._fed
        self._fed = 0
        self._segment_id = new_id("seg_")
        self._partial = ""
        self._speaking = False

    @property
    def _offset_s(self) -> float:
        return self._offset / _SAMPLE_RATE

    # ------------------------------------------------------------ worker thread
    def _feed_sync(self, final: bool) -> Any:
        """Feed the queued audio; returns the running result (finalized + draft tokens)."""
        pending, self._pending, self._pending_samples = self._pending, [], 0
        samples = pending[0] if len(pending) == 1 else np.concatenate(pending)
        stt = self._parakeet
        if final and len(samples) < _MIN_TAIL:
            # a few ms left at a flush: too short for a mel frame, and silent anyway
            self._fed += len(samples)
            return None
        try:
            t0 = now()
            if self._streamer is None:
                self._streamer = stt._open_streamer()
            self._streamer.add_audio(stt._audio(samples))
            self._fed += len(samples)
            result = self._streamer.result
            self._step_time = now() - t0
            return result
        except Exception as exc:
            raise _mlx.map_error(exc, _PROVIDER, "streaming recognition") from exc

    def _finalize_sync(self) -> Any:
        """The utterance's result; the next audio starts a new recognizer state."""
        streamer, self._streamer = self._streamer, None
        if streamer is None:
            return None
        try:
            return streamer.result
        finally:
            self._parakeet._close_streamer()


for _name, _info in MODELS.items():
    register_model(
        _PROVIDER,
        _name,
        kind="stt",
        files=[
            ModelFile.from_hf_repo(_info.repo, patterns=_FILES, required=_FILES, size=_info.size)
        ],
        license="CC-BY-4.0",
        languages=_info.languages,
        description=f"{_info.description} (MLX, {_info.repo})",
    )
