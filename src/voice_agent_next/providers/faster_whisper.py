"""faster-whisper speech-to-text: Whisper on CTranslate2, running locally on CPU or CUDA.

``create("stt", "faster_whisper/large-v3-turbo")`` (alias: ``"whisper/small"``).

Whisper is a batch recognizer (``capabilities.streaming=False``). The cascade makes it
real-time by wrapping it in :class:`~voice_agent_next.stt.StreamAdapter`, which cuts the
input into utterances with the configured VAD and transcribes each one. With
``interim_results=True`` the adapter also re-decodes the growing utterance while the user
speaks (interim transcripts), paced by the cost of a decode so that the model is idle
when the utterance ends and the final transcript is not delayed.

A :class:`HallucinationGuard` (on by default) drops what Whisper "hears" in noise:
segments Whisper itself rates as no-speech, known subtitle artifacts ("Thanks for
watching!"), repetition loops, and short stock phrases ("Thank you.") that the VAD was
unsure about.

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
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np
import numpy.typing as npt

from ..audio.frame import AudioFrame
from ..audio.resample import StreamResampler
from ..errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderError,
    RateLimitError,
    VoiceAgentError,
)
from ..hardware import ctranslate2_compute_type, select_ctranslate2_backend
from ..metrics import STTMetrics
from ..models import ModelFile, register_model
from ..registry import register_provider
from ..stt import STT, STTCapabilities, STTEvent, STTEventType, STTStream, Transcript, WordTiming
from ..utils.aio import cancel_and_wait
from ..utils.clock import now
from ..utils.deps import is_installed, require
from ..utils.ids import new_id
from ..utils.log import logger
from ._whisper_guard import HallucinationGuard

if TYPE_CHECKING:
    from ..stt import StreamAdapter

__all__ = ["FasterWhisperSTT", "HallucinationGuard"]

_PROVIDER = "faster_whisper"
_EXTRA = "faster-whisper"
_SAMPLE_RATE = 16_000  # Whisper's native input rate
_DEVICES = ("auto", "cpu", "cuda")
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
_INTERIM_INTERVAL = {"cuda": 0.25, "cpu": 0.5}
"""Default seconds of new speech between two interim decodes, per device."""
_PACING = 2.0
"""Queue at least this many times the last interim decode's duration between decodes."""
_MIN_INTERIM_AUDIO = 0.3
"""Utterances shorter than this (seconds, VAD prefix included) get no interim decode."""


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
        device: ``"auto"`` uses CUDA when CTranslate2 sees a GPU, the CUDA libraries it
            opens load (from the ``cuda`` extra's pip wheels or the system) *and* a warm-up
            inference succeeds on it; otherwise CPU, with a log line naming the fix (see
            ``docs/hardware.md``). ``"cpu"`` or ``"cuda"`` force a device.
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
        interim_results: behind a VAD (:class:`~voice_agent_next.stt.StreamAdapter`, as in
            the cascade), re-decode the utterance while the user speaks and emit
            ``INTERIM_TRANSCRIPT`` events. Interim decodes are greedy, without timestamps
            or temperature fallback; the final transcript is decoded as usual.
        interim_interval: seconds of new speech between two interim decodes (at least: a
            decode that takes longer than half of it spaces the next one out). ``None``
            picks 0.25 s on CUDA and 0.5 s on the CPU.
        hallucination_guard: drop segments that are probably not speech (see
            :class:`HallucinationGuard`): ``True`` (default) uses the default thresholds,
            ``False`` disables it, a :class:`HallucinationGuard` or a mapping of its
            fields configures it.
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
        interim_results: bool = False,
        interim_interval: float | None = None,
        hallucination_guard: bool | HallucinationGuard | Mapping[str, Any] = True,
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
        if interim_interval is not None and interim_interval <= 0:
            raise ConfigurationError(
                f"faster_whisper: interim_interval must be > 0, got {interim_interval}"
            )
        guard = _make_guard(hallucination_guard)
        name = model or "large-v3-turbo"
        super().__init__(
            model=name,
            capabilities=STTCapabilities(
                streaming=False,
                interim_results=interim_results,
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
        self.interim_interval = interim_interval
        self.guard = guard
        """The :class:`HallucinationGuard` applied to every transcript."""
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
        # Also loads cuBLAS from the `cuda` extra's pip wheels, for "cuda" and "auto".
        backend = select_ctranslate2_backend(
            self.device, self.compute_type, ct2=ct2, device_index=self._first_device_index
        )
        if self.device != "auto":
            return self._load_on(fw, path, self.device, backend.compute_type, t0)
        if backend.device == "cuda":
            try:
                return self._load_on(fw, path, "cuda", backend.compute_type, t0)
            except Exception as exc:
                # The GPU is visible and its libraries load, but CTranslate2 cannot run on
                # it (no kernels for this GPU, out of memory, a broken driver...).
                logger.warning(
                    "faster-whisper: CUDA is not usable (%s); falling back to CPU. "
                    "See docs/hardware.md to enable the GPU.",
                    exc.__cause__ or exc,
                )
            cpu_type = ctranslate2_compute_type(ct2, "cpu", self.compute_type)
            return self._load_on(fw, path, "cpu", cpu_type, t0)
        if backend.fix:
            logger.info(
                "faster-whisper: running on CPU: %s. To use the GPU: %s",
                backend.reason,
                backend.fix,
            )
        else:
            logger.debug("faster-whisper: running on CPU: %s", backend.reason)
        return self._load_on(fw, path, "cpu", backend.compute_type, t0)

    @property
    def _first_device_index(self) -> int:
        index = self.device_index
        return index if isinstance(index, int) else (index[0] if index else 0)

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

    def _load_on(self, fw: Any, path: str, device: str, compute: str | None, t0: float) -> Any:
        compute_type = compute or "default"
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

    @property
    def resolved_interim_interval(self) -> float:
        """Seconds of new speech between interim decodes (``interim_interval`` or the
        device default)."""
        if self.interim_interval is not None:
            return self.interim_interval
        return _INTERIM_INTERVAL.get(self.resolved_device or "cpu", _INTERIM_INTERVAL["cpu"])

    # ---------------------------------------------------------------- recognition
    async def _recognize(self, audio: AudioFrame, *, language: str | None) -> Transcript:
        # STT.transcribe() has already resampled the audio to 16 kHz mono.
        samples = audio.to_float32()
        return await asyncio.to_thread(self._transcribe_sync, samples, _whisper_language(language))

    def _create_adapter_stream(
        self, adapter: StreamAdapter, *, language: str | None
    ) -> STTStream | None:
        return _WhisperAdapterStream(adapter, self, language=language)

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
        if interim:  # fast and cheap: the final transcript is decoded properly anyway
            options.update(
                beam_size=1,
                word_timestamps=False,
                temperature=0.0,
                without_timestamps=True,
                condition_on_previous_text=False,
            )
        try:
            segments, info = model.transcribe(samples, **options)
            segments = list(segments)  # lazy generator: decoding happens while iterating
        except Exception as exc:
            raise _map_error(exc, "transcription") from exc
        verdict = self.guard.filter(segments, vad_confidence=vad_confidence)
        if verdict.dropped:
            logger.debug(
                "faster-whisper: dropped %s (%s)",
                "; ".join(f"{t!r}: {why}" for t, why in verdict.dropped),
                "interim" if interim else "final",
            )
        return _to_transcript(
            verdict.kept,
            info,
            language,
            words=bool(options["word_timestamps"]),
            clean=self.guard.clean_text,
        )


class _WhisperAdapterStream(STTStream):
    """What :class:`~voice_agent_next.stt.StreamAdapter` runs for faster-whisper.

    Like the adapter's own stream, the VAD cuts the input into utterances and each one gets
    one final transcript. In addition:

    * **interim transcripts** (``interim_results=True``): while the VAD reports speech, the
      utterance so far is re-decoded in a background thread whenever enough new speech has
      arrived: ``interim_interval``, or twice the duration of the last decode when that is
      longer (back-off on a slow device), so that the model is busy at most about half of
      the time. No decode starts while the latest VAD window is below the activation
      threshold (pauses, the trailing silence before END_OF_SPEECH): a decode that starts
      on the last voiced window has the VAD's ``min_silence_duration`` to finish before
      the final one is needed. A decode still running when the utterance ends is awaited
      and its result discarded (a CTranslate2 call cannot be interrupted), unless
      ``num_workers >= 2``: then the final runs on the second model replica meanwhile.
    * **VAD-aware hallucination guard**: the mean speech probability of the utterance's
      speech windows is passed to :class:`HallucinationGuard`.
    """

    def __init__(
        self, adapter: StreamAdapter, whisper: FasterWhisperSTT, *, language: str | None
    ) -> None:
        self._adapter = adapter
        self._whisper = whisper
        self._segment_id = new_id("seg_")
        self._live: str | None = None
        """Segment whose interim results are still wanted (``None`` once it ends)."""
        self._probs: list[float] = []
        """VAD probabilities of the utterance's speech windows."""
        self._voiced = False
        """The latest VAD window is at or above the activation threshold."""
        self._since_step = 0.0
        """Seconds of input since the last interim decode started (or the utterance)."""
        self._step_time = 0.0
        """Duration of the last interim decode (seconds)."""
        self._step: asyncio.Task[None] | None = None
        self._partial = ""
        self._detected: str | None = None
        """Language detected by an interim decode (reused by the next ones)."""
        self.interim_decodes = 0
        """Interim decodes run by this stream (diagnostics)."""
        self.final_waits: list[float] = []
        """Seconds each final transcript waited for an interim decode in flight."""
        super().__init__(adapter, language=language)

    # -------------------------------------------------------------- event loop
    async def _run(self) -> None:
        from ..vad import VADEventType

        vad = self._adapter.vad
        activation = vad.options.activation_threshold
        deactivation = vad.options.effective_deactivation
        vad_stream = vad.stream(emit_inference_events=True)
        try:
            async for item in self._input:
                if self.is_flush(item):
                    if vad_stream.speaking:
                        await self._finish(vad_stream.speech_frames())
                        vad_stream.reset()
                    continue
                assert isinstance(item, AudioFrame)
                for ev in vad_stream.push_audio(item):
                    if ev.type == VADEventType.START_OF_SPEECH:
                        self._live = self._segment_id
                        self._emit(
                            STTEvent(STTEventType.START_OF_SPEECH, segment_id=self._segment_id)
                        )
                    elif ev.type == VADEventType.END_OF_SPEECH:
                        await self._finish(list(ev.frames))
                    elif ev.type == VADEventType.INFERENCE_DONE:
                        self._voiced = ev.speaking and ev.probability >= activation
                        if ev.speaking and ev.probability >= deactivation:
                            self._probs.append(ev.probability)
                if vad_stream.speaking:
                    self._since_step += item.duration
                    self._maybe_step(vad_stream.speech_frames)
            if vad_stream.speaking:
                await self._finish(vad_stream.speech_frames())
        finally:
            await cancel_and_wait(self._step)
            vad_stream.close()

    @property
    def _threshold(self) -> float:
        """Seconds of new input before the next interim decode."""
        return max(self._whisper.resolved_interim_interval, _PACING * self._step_time)

    @property
    def _vad_confidence(self) -> float | None:
        return sum(self._probs) / len(self._probs) if self._probs else None

    def _maybe_step(self, speech_frames: Callable[[], list[AudioFrame]]) -> None:
        if not self._whisper.capabilities.interim_results or not self._voiced:
            return
        if self._step is not None and not self._step.done():
            return
        if self._live is None or self._since_step < self._threshold:
            return
        frames = speech_frames()
        if sum(f.duration for f in frames) < _MIN_INTERIM_AUDIO:
            return
        self._since_step = 0.0
        self._step = asyncio.create_task(
            self._interim(frames, self._live, self._vad_confidence),
            name="faster-whisper-interim",
        )

    async def _interim(
        self, frames: list[AudioFrame], segment_id: str, vad_confidence: float | None
    ) -> None:
        whisper = self._whisper
        language = _whisper_language(self._language) or self._detected
        t0 = now()
        try:
            transcript = await asyncio.to_thread(
                _run_on_frames,
                whisper,
                frames,
                language,
                interim=True,
                vad_confidence=vad_confidence,
            )
        except Exception as exc:  # interims are best effort; the final reports errors
            logger.warning("faster-whisper: interim decode failed: %s", exc)
            return
        finally:
            self._step_time = now() - t0
            self.interim_decodes += 1
        if segment_id != self._live:
            return  # the utterance ended meanwhile: its final transcript is coming
        self._detected = self._detected or transcript.language
        text = transcript.text
        if text and text != self._partial:
            self._partial = text
            self._emit(STTEvent(STTEventType.INTERIM_TRANSCRIPT, transcript, segment_id))

    async def _finish(self, frames: list[AudioFrame]) -> None:
        segment_id = self._segment_id
        self._live = None  # a decode in flight is now stale: its result is dropped
        step = self._step
        if step is not None and not step.done() and self._whisper.num_workers < 2:
            # one model replica: the final would queue behind the decode anyway
            t0 = now()
            await asyncio.wait({step})
            self.final_waits.append(now() - t0)
        else:  # idle, or a second replica runs the final next to the interim decode
            self.final_waits.append(0.0)
        vad_confidence = self._vad_confidence
        self._segment_id = new_id("seg_")
        self._probs = []
        self._voiced = False
        self._since_step = 0.0
        self._partial = ""
        self._detected = None
        if frames:
            transcript = await self._final(frames, vad_confidence)
            if transcript.text.strip():
                self._emit(STTEvent(STTEventType.FINAL_TRANSCRIPT, transcript, segment_id))
        self._emit(STTEvent(STTEventType.END_OF_SPEECH, segment_id=segment_id))

    async def _final(self, frames: list[AudioFrame], vad_confidence: float | None) -> Transcript:
        """Like :meth:`STT.transcribe` (same metrics), with the VAD's confidence."""
        whisper = self._whisper
        t0 = now()
        error: str | None = None
        duration = sum(f.duration for f in frames)
        try:
            return await asyncio.to_thread(
                _run_on_frames,
                whisper,
                frames,
                _whisper_language(self._language),
                vad_confidence=vad_confidence,
            )
        except Exception as exc:
            error = repr(exc)
            raise
        finally:
            whisper.emit(
                "metrics",
                STTMetrics(
                    provider=whisper.provider,
                    model=whisper.model,
                    request_id=new_id("stt_"),
                    audio_duration=duration,
                    duration=now() - t0,
                    streamed=False,
                    error=error,
                ),
            )


def _run_on_frames(
    whisper: FasterWhisperSTT,
    frames: Sequence[AudioFrame],
    language: str | None,
    *,
    interim: bool = False,
    vad_confidence: float | None = None,
) -> Transcript:
    """Transcribe VAD frames (any rate) on the calling (worker) thread."""
    frame = AudioFrame.concat(list(frames))
    if frame.sample_rate != _SAMPLE_RATE or frame.channels != 1:
        rs = StreamResampler(_SAMPLE_RATE, 1)
        frame = AudioFrame.concat([rs.push(frame), rs.flush()])
    return whisper._transcribe_sync(
        frame.to_float32(), language, interim=interim, vad_confidence=vad_confidence
    )


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


def _make_guard(
    value: bool | HallucinationGuard | Mapping[str, Any] | None,
) -> HallucinationGuard:
    if isinstance(value, HallucinationGuard):
        return value
    if value is None or value is False:
        return HallucinationGuard.disabled()
    if value is True:
        return HallucinationGuard()
    if isinstance(value, Mapping):
        fields: dict[str, Any] = {
            k: tuple(v) if isinstance(v, list) else v for k, v in value.items()
        }
        try:
            return HallucinationGuard(**fields)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"faster_whisper: hallucination_guard: {exc}") from exc
    raise ConfigurationError(
        "faster_whisper: hallucination_guard must be a bool, a HallucinationGuard or a "
        f"mapping, got {type(value).__name__}"
    )


def _to_transcript(
    segments: list[Any],
    info: Any,
    language: str | None,
    *,
    words: bool,
    clean: Callable[[str], str] | None = None,
) -> Transcript:
    text = "".join(s.text for s in segments).strip()
    if clean is not None:
        text = clean(text)
    confidence: float | None = None
    if segments:
        # exp(mean token log-probability), weighted by segment length in tokens
        weights = [max(1, len(s.tokens)) for s in segments]
        total = sum(s.avg_logprob * w for s, w in zip(segments, weights, strict=True))
        confidence = min(1.0, math.exp(total / sum(weights)))
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


# faster_whisper.utils._MODELS: model name -> Hugging Face repository (fetched at "main"
# into the HF cache by faster_whisper.download_model with these allow_patterns).
_HF_REPOS = {
    "large-v3-turbo": ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", 1_621_665_983),
    "large-v3": ("Systran/faster-whisper-large-v3", 3_090_835_702),
    "distil-large-v3.5": ("distil-whisper/distil-large-v3.5-ct2", 1_516_479_656),
    "medium": ("Systran/faster-whisper-medium", 1_530_571_735),
    "small": ("Systran/faster-whisper-small", 486_212_372),
    "small.en": ("Systran/faster-whisper-small.en", 486_098_798),
    "base": ("Systran/faster-whisper-base", 147_882_941),
    "base.en": ("Systran/faster-whisper-base.en", 147_769_510),
    "tiny": ("Systran/faster-whisper-tiny", 78_203_619),
    "tiny.en": ("Systran/faster-whisper-tiny.en", 78_090_594),
}
_ALLOW_PATTERNS = (
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
)
for _name, (_repo, _size) in _HF_REPOS.items():
    register_model(
        _PROVIDER,
        _name,
        kind="stt",
        files=[
            ModelFile.from_hf_repo(
                _repo,
                patterns=_ALLOW_PATTERNS,
                required=("model.bin", "config.json", "tokenizer.json"),
                size=_size,
            )
        ],
        license="MIT",
        languages="en" if _name.endswith(".en") or _name.startswith("distil") else "99 languages",
        description=f"Whisper {_name} (CTranslate2) from {_repo}",
        aliases=("turbo",) if _name == "large-v3-turbo" else (),
    )
