"""Shared machinery of the local PyTorch TTS providers (Chatterbox, Qwen3-TTS).

A :class:`LocalTorchTTS` runs its model on one dedicated worker thread (the models are
not thread-safe, and a GPU runs one synthesis at a time anyway): requests are served in
order, audio is posted back to the event loop chunk by chunk as the model produces it,
and a request that is closed (barge-in) stops the model at its next decoding step.

Subclasses implement :meth:`LocalTorchTTS._load` (called once, on the worker thread) and
:meth:`LocalTorchTTS._render` (one text segment -> float32 audio chunks). The base class
splits texts into sentences, adds estimated word timings per sentence (see
:func:`~voice_agent_next.providers.pocket_tts.estimate_word_timings`) and maps failures to
:mod:`voice_agent_next.errors`.
"""

from __future__ import annotations

import asyncio
import threading
from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, TypeVar

import numpy as np

from ..audio.frame import AudioFrame
from ..errors import ConfigurationError, ProviderError
from ..hardware import select_torch_backend
from ..stt import WordTiming
from ..text.sentences import SentenceSegmenter
from ..tts import TTS, ChunkedStream, SynthesizedAudio
from ..utils.clock import now
from ..utils.log import logger
from .pocket_tts import estimate_word_timings, speech_bounds

__all__ = ["LocalTorchTTS", "Stopped", "stop_on"]

T = TypeVar("T")


class Stopped(Exception):
    """Raised inside the model to abandon a generation (the request was closed)."""


@contextmanager
def stop_on(module: Any, stop: threading.Event) -> Iterator[None]:
    """Make ``module`` raise :class:`Stopped` at its next forward pass once ``stop`` is set.

    Autoregressive models call their transformer once per generated token, so this ends a
    generation within one step instead of after the whole sentence.
    """

    def check(_module: Any, _args: Any) -> None:
        if stop.is_set():
            raise Stopped

    handle = module.register_forward_pre_hook(check)
    try:
        yield
    finally:
        handle.remove()


class LocalTorchTTS(TTS):
    """Base class of TTS models running locally with PyTorch (see the module docstring)."""

    _thread_name = "torch-tts"
    _warmup_text = "Hello, this is a warm-up."

    def __init__(
        self,
        *,
        model: str,
        sample_rate: int,
        voice: str | None,
        device: str = "auto",
        split_sentences: bool = True,
        word_timings: bool = True,
        clean_text: bool = True,
    ) -> None:
        super().__init__(model=model, sample_rate=sample_rate, voice=voice, clean_text=clean_text)
        self.device_request = device
        self.device: str | None = None
        """Where the model runs once loaded (``"cuda"``, ``"mps"``, ``"cpu"``)."""
        self.split_sentences = split_sentences
        self.word_timings = word_timings
        self._model: Any = None
        self._executor: ThreadPoolExecutor | None = None
        self._active: set[threading.Event] = set()

    # ------------------------------------------------------------------ subclass hooks
    @abstractmethod
    def _load(self, device: str) -> Any:
        """Load the model on ``device`` (worker thread)."""

    @abstractmethod
    def _render(
        self, model: Any, text: str, voice: str | None, stop: threading.Event
    ) -> Iterable[np.ndarray]:
        """Synthesize one segment (worker thread): yield mono float32 chunks at
        ``self.sample_rate``. Should return early (or raise :class:`Stopped`) once ``stop``
        is set."""

    def _accelerators(self) -> tuple[str, ...]:
        """Devices ``device="auto"`` may pick, best first."""
        return ("cuda", "mps")

    # ------------------------------------------------------------------ public API
    def _synthesize(self, text: str, *, voice: str | None) -> ChunkedStream:
        return _TorchChunkedStream(self, text, voice=voice)

    async def warmup(self) -> None:
        """Download (if needed) and load the model, then run one short synthesis (the
        first CUDA run compiles kernels and allocates memory)."""
        await self._submit(self._warmup_sync)

    async def aclose(self) -> None:
        """Stop the worker thread; a synthesis still running stops at its next step."""
        for stop in list(self._active):
            stop.set()
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        self._model = None

    # ------------------------------------------------------------------ internals
    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=self._thread_name)
        return self._executor

    async def _submit(self, fn: Callable[..., T], *args: Any) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._get_executor(), fn, *args)

    def _get_model(self) -> Any:
        """The loaded model (worker thread only)."""
        if self._model is None:
            backend = select_torch_backend(self.device_request, accelerators=self._accelerators())
            if backend.fix:
                logger.warning("%s: running on %s; to use the GPU: %s", self.provider, backend, backend.fix)
            t0 = now()
            try:
                self._model = self._load(backend.device)
            except (ConfigurationError, ProviderError):
                raise
            except Exception as exc:
                raise ProviderError(
                    f"failed to load {self.provider} model {self.model}: {exc}",
                    provider=self.provider,
                ) from exc
            self.device = backend.device
            logger.info(
                "%s: loaded %s on %s in %.2f s", self.provider, self.model, backend, now() - t0
            )
        return self._model

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

    def _warmup_sync(self) -> None:
        t0 = now()
        for _ in self._generate(self._warmup_text, self.voice, threading.Event()):
            pass
        logger.debug("%s: warm-up took %.2f s", self.provider, now() - t0)

    def _generate(
        self, text: str, voice: str | None, stop: threading.Event
    ) -> Iterable[bytes | list[WordTiming]]:
        """Synthesize ``text`` (worker thread): yields s16le PCM chunks as they are
        produced and, after each segment, its estimated word timings (seconds from the
        start of this text's audio)."""
        model = self._get_model()
        offset = 0  # samples emitted before the current segment
        for segment in self._segments(text):
            if stop.is_set():
                return
            chunks: list[np.ndarray] = []
            try:
                for chunk in self._render(model, segment, voice, stop):
                    samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
                    if not samples.size:
                        continue
                    chunks.append(samples)
                    yield AudioFrame.from_numpy(samples, self.sample_rate).data
                    if stop.is_set():
                        return
            except Stopped:
                return
            except (ConfigurationError, ProviderError):
                raise
            except Exception as exc:
                raise ProviderError(
                    f"{self.provider} synthesis failed: {exc}", provider=self.provider
                ) from exc
            audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
            if self.word_timings and not stop.is_set():
                bounds = speech_bounds(audio)
                if bounds is not None:
                    start, end = ((offset + b) / self.sample_rate for b in bounds)
                    words = estimate_word_timings(segment, start, end)
                    if words:
                        yield words
            offset += audio.size


_DONE = object()


class _TorchChunkedStream(ChunkedStream):
    async def _run(self) -> None:
        tts = self._tts
        assert isinstance(tts, LocalTorchTTS)
        voice = self.voice or tts.voice
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
                items.put_nowait(ProviderError(f"{tts.provider} was closed", provider=tts.provider))
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
            stop.set()  # cancelled: the generation stops at its next step
            tts._active.discard(stop)
