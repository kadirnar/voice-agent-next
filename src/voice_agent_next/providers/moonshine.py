"""Moonshine Streaming speech-to-text: low-latency local STT for CPUs and edge devices.

``create("stt", "moonshine/small-streaming")`` runs the Moonshine Streaming models through
the native runtime of the ``moonshine-voice`` package (C++ on ONNX Runtime, no torch).

Streaming models (``tiny-streaming``, ``small-streaming``, ``medium-streaming``) encode
audio incrementally while the user speaks, so a :meth:`~voice_agent_next.stt.STTStream.flush`
only has to finish the last few hundred milliseconds and decode the line: the final
transcript arrives in tens of milliseconds instead of re-transcribing the whole utterance
(what batch Whisper does). The runtime also segments speech into lines with its own VAD;
a completed line is emitted as a final transcript, but in the cascade the VAD and turn
detector own endpointing and :meth:`flush` force-completes the current line.

Models come from the package's built-in catalog (size and CRC32C of every file are
compiled into the native library, so a given ``moonshine-voice`` release always fetches
the same files). They are cached in the package's own cache (``~/.cache/moonshine_voice``,
``$MOONSHINE_VOICE_CACHE``). All native calls run in worker threads. See
``docs/providers/moonshine.md`` for models, languages, licences and latency numbers.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ..audio.frame import AudioFrame
from ..errors import (
    ConfigurationError,
    ProviderConnectionError,
    ProviderError,
    VoiceAgentError,
)
from ..registry import register_provider
from ..stt import STT, STTCapabilities, STTEvent, STTEventType, STTStream, Transcript, WordTiming
from ..utils.aio import ChanClosed
from ..utils.clock import now
from ..utils.deps import is_installed, require
from ..utils.log import logger

__all__ = ["MoonshineSTT"]

_PROVIDER = "moonshine"
_EXTRA = "moonshine"
_PACKAGE = "moonshine-voice"
_SAMPLE_RATE = 16_000
_DEFAULT_MODEL = "small-streaming"
_ARCHS: dict[str, int] = {
    # name -> moonshine_voice.ModelArch value (moonshine-c-api.h MOONSHINE_MODEL_ARCH_*)
    "tiny": 0,
    "base": 1,
    "tiny-streaming": 2,
    "base-streaming": 3,
    "small-streaming": 4,
    "medium-streaming": 5,
}
_MODELS = ("tiny-streaming", "small-streaming", "medium-streaming", "tiny", "base")
_NO_UPDATES = 1e9
"""``Stream`` update interval that disables the package's own update cadence (we drive it)."""


@dataclass(slots=True)
class _Line:
    """The fields of a ``moonshine_voice.TranscriptLine`` the stream needs (plain data)."""

    line_id: int
    text: str
    start_time: float
    duration: float
    is_complete: bool
    is_updated: bool
    words: list[WordTiming] | None


@register_provider(
    "stt",
    _PROVIDER,
    description="Moonshine Streaming (moonshine-voice): streaming local STT for CPUs / edge",
    default_model=_DEFAULT_MODEL,
    models=_MODELS,
    env=(),
    extra=_EXTRA,
    requires=("moonshine_voice",),
    local=True,
)
class MoonshineSTT(STT):
    """Local streaming recognizer running Moonshine models on the ``moonshine-voice`` runtime.

    Args:
        model: ``tiny-streaming`` (34M), ``small-streaming`` (123M, default),
            ``medium-streaming`` (245M), or the non-streaming ``tiny`` / ``base``; or a
            local model directory (then ``arch`` names its architecture).
        language: language code (``"en"`` default; region suffixes are dropped). Other
            languages have separate models, see ``docs/providers/moonshine.md``.
        arch: architecture of a local model directory (one of the model names above).
        update_interval: seconds of new audio between incremental transcription passes.
            Each pass encodes the new audio (so a flush has little left to do) and, with
            ``interim_results``, decodes the current line. The runtime skips passes that
            come less than ~0.5 s apart unless ``force_updates`` is set.
        force_updates: run every pass at ``update_interval`` (lower flush latency, about
            twice the CPU).
        interim_results: emit interim transcripts; ``False`` only encodes while the
            user speaks and decodes on completion (``decode_incomplete_lines=false``).
        word_timestamps: fill :attr:`Transcript.words` (downloads an extra decoder).
        keyterms: words or phrases to bias the decoder towards (names, jargon); they
            must not contain commas.
        cache_dir: model cache root (default: the package's cache).
        local_files_only: never download; also implied by ``VAN_OFFLINE=1``.
        options: extra native transcriber options (``moonshine-c-api.h``), e.g.
            ``{"vad_threshold": 0.6}``; they override the options above.
    """

    provider = _PROVIDER

    def __init__(
        self,
        *,
        model: str | None = None,
        language: str | None = None,
        arch: str | None = None,
        update_interval: float = 0.2,
        force_updates: bool = False,
        interim_results: bool = True,
        word_timestamps: bool = False,
        keyterms: Sequence[str] | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        local_files_only: bool = False,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        if not is_installed("moonshine_voice"):
            require("moonshine_voice", extra=_EXTRA, package=_PACKAGE)
        name = model or _DEFAULT_MODEL
        local_dir = os.path.isdir(name)
        arch_name = (arch or ("" if local_dir else name)).strip().lower()
        if arch_name not in _ARCHS:
            if local_dir:
                raise ConfigurationError(
                    f"moonshine: a local model directory needs arch=, one of {tuple(_ARCHS)}"
                )
            raise ConfigurationError(
                f"moonshine: unknown model {name!r}; expected one of {_MODELS} "
                "or a local model directory"
            )
        if update_interval <= 0:
            raise ConfigurationError(
                f"moonshine: update_interval must be > 0, got {update_interval}"
            )
        terms = [t.strip() for t in keyterms or () if t.strip()]
        if any("," in t for t in terms):
            raise ConfigurationError("moonshine: keyterms must not contain commas")
        super().__init__(
            model=name,
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=interim_results,
                word_timestamps=word_timestamps,
            ),
            sample_rate=_SAMPLE_RATE,
            language=_moonshine_language(language),
        )
        self.arch = arch_name
        self.update_interval = update_interval
        self.force_updates = force_updates
        self.interim_results = interim_results
        self.word_timestamps = word_timestamps
        self.keyterms = terms
        self.cache_dir = os.fspath(cache_dir) if cache_dir is not None else None
        self.local_files_only = local_files_only
        self.options = dict(options or {})
        self.model_path: str | None = None
        """Directory the model was loaded from (``None`` until it is loaded)."""
        self._local_dir = local_dir
        self._transcriber: Any = None
        self._load_lock = threading.Lock()
        self.native_lock = threading.Lock()
        """Serializes native calls on the shared transcriber (streams run in threads)."""

    @property
    def streaming_model(self) -> bool:
        return self.arch.endswith("-streaming")

    # ------------------------------------------------------------------ lifecycle
    async def warmup(self) -> None:
        """Download (if needed) and load the model, then run a short warm-up pass."""
        await asyncio.to_thread(self._warm_up)

    async def aclose(self) -> None:
        # The native transcriber is freed when its last reference goes away; open
        # streams keep their own reference until they are closed.
        self._transcriber = None

    def _ensure_transcriber(self) -> Any:
        transcriber = self._transcriber
        if transcriber is None:
            with self._load_lock:
                transcriber = self._transcriber
                if transcriber is None:
                    transcriber = self._transcriber = self._load()
        return transcriber

    def _load(self) -> Any:
        mv = require("moonshine_voice", extra=_EXTRA, package=_PACKAGE)
        t0 = now()
        path = self._model_dir()
        options: dict[str, Any] = {"return_audio_data": "false"}
        if not self.interim_results:
            options["decode_incomplete_lines"] = "false"
        if self.word_timestamps:
            options["word_timestamps"] = "true"
        if self.keyterms:
            options["keyterms"] = ",".join(self.keyterms)
        for key, value in self.options.items():
            options[key] = _option_value(value)
        try:
            transcriber = mv.Transcriber(
                model_path=path,
                model_arch=mv.ModelArch(_ARCHS[self.arch]),
                update_interval=_NO_UPDATES,
                options=options,
            )
        except Exception as exc:
            raise _map_error(exc, f"loading model {self.model!r}") from exc
        self.model_path = path
        logger.info("moonshine %s (%s) loaded in %.2fs", self.model, self.language, now() - t0)
        return transcriber

    def _model_dir(self) -> str:
        if self._local_dir:
            return self.model
        language = self.language or "en"
        mv_download = require("moonshine_voice.download", extra=_EXTRA, package=_PACKAGE)
        arch = require("moonshine_voice", extra=_EXTRA, package=_PACKAGE).ModelArch(
            _ARCHS[self.arch]
        )
        try:
            info = mv_download.find_model_info(language, arch)
        except ValueError as exc:
            raise ConfigurationError(
                f"moonshine: no {self.model!r} model for language {language!r} ({exc})"
            ) from exc
        if info.get("language", language) != "en":
            logger.warning(
                "moonshine: the %s model is released under the non-commercial Moonshine "
                "Community License (https://www.moonshine.ai/license)",
                language,
            )
        cache_root = Path(self.cache_dir) if self.cache_dir is not None else None
        offline = os.environ.get("VAN_OFFLINE", "").lower() in ("1", "true", "yes")
        if self.local_files_only or offline:
            return self._cached_model_dir(info, cache_root)
        try:
            path, _ = mv_download.download_model_from_info(
                info,
                cache_root=cache_root,
                on_progress=_log_progress(self.model),
                include_word_timestamps=self.word_timestamps,
            )
        except Exception as exc:
            raise _map_error(exc, f"downloading model {self.model!r}") from exc
        return str(path)

    def _cached_model_dir(self, info: Mapping[str, Any], cache_root: Path | None) -> str:
        """The model directory when every file of the catalog entry is already cached."""
        mv_api = require("moonshine_voice.moonshine_api", extra=_EXTRA, package=_PACKAGE)
        mv_files = require("moonshine_voice.download_file", extra=_EXTRA, package=_PACKAGE)
        opts: dict[str, Any] = {"model_arch": int(info["model_arch"])}
        if self.word_timestamps:
            opts["word_timestamps"] = True
        manifest = json.loads(mv_api.moonshine_get_stt_dependencies_string(info["language"], opts))
        groups = manifest.get("groups") or []
        if not groups:
            raise ProviderError(f"moonshine: empty download manifest for {self.model!r}")
        group = groups[0]
        root = Path(cache_root or mv_files.get_cache_dir()) / group["base_url"].replace(
            "https://", ""
        )
        missing = [f["name"] for f in group.get("files", []) if not (root / f["name"]).is_file()]
        if missing:
            raise ProviderConnectionError(
                f"moonshine: model {self.model!r} is not cached ({', '.join(missing)} missing "
                f"under {root}) and downloads are disabled",
                provider=_PROVIDER,
            )
        return str(root)

    def _warm_up(self) -> None:
        transcriber = self._ensure_transcriber()
        with self.native_lock:
            stream = transcriber.create_stream(update_interval=_NO_UPDATES)
            try:
                stream.start()
                stream.add_audio(np.zeros(_SAMPLE_RATE // 2, dtype=np.float32).tolist())
                stream.stop()
            except Exception as exc:
                raise _map_error(exc, "warm-up") from exc
            finally:
                stream.close()

    # ---------------------------------------------------------------- recognition
    def _create_stream(self, *, language: str | None) -> STTStream:
        requested = _moonshine_language(language)
        if requested is not None and requested != (self.language or "en"):
            raise ConfigurationError(
                f"moonshine: this instance runs a {self.language or 'en'!r} model; create "
                f"another MoonshineSTT(language={requested!r}) for {requested!r}"
            )
        return _MoonshineStream(self, language=self.language or "en")


class _MoonshineStream(STTStream):
    """One native Moonshine stream; every native call runs in a worker thread.

    Audio is batched while a pass runs, so a slow machine does fewer, larger passes rather
    than falling behind. A flush stops the native stream (the runtime then completes every
    line), emits one final transcript per completed line — an empty one if there was
    nothing to complete, so the caller's wait always ends — and restarts the stream.
    """

    def __init__(self, stt: MoonshineSTT, *, language: str) -> None:
        self._moonshine = stt
        self._native: Any = None
        self._errors: list[BaseException] = []
        self._offset = 0.0  # stream time at which the native stream was last (re)started
        self._pushed = 0.0  # seconds of audio handed to the native stream
        self._since_update = 0.0
        self._seen: set[int] = set()
        self._done: set[int] = set()
        self._interims: dict[int, str] = {}
        self._speaking = False
        super().__init__(stt, language=language)

    # --------------------------------------------------------------- worker thread
    def _open_sync(self) -> None:
        stt = self._moonshine
        transcriber = stt._ensure_transcriber()
        with stt.native_lock:
            try:
                native = transcriber.create_stream(update_interval=_NO_UPDATES)
                native.add_listener(self._on_native_event)
                native.start()
            except Exception as exc:
                raise _map_error(exc, "opening a stream") from exc
        self._native = native

    def _on_native_event(self, event: Any) -> None:
        # Stream.stop() reports a failed final pass as an ``Error`` event, not an exception.
        error = getattr(event, "error", None)
        if isinstance(error, BaseException):
            self._errors.append(error)

    def _process_sync(self, samples: npt.NDArray[np.float32] | None, flush: bool) -> list[_Line]:
        stt = self._moonshine
        native = self._native
        with stt.native_lock:
            try:
                if samples is not None and samples.size:
                    native.add_audio(samples.tolist(), _SAMPLE_RATE)
                    self._pushed += samples.size / _SAMPLE_RATE
                    self._since_update += samples.size / _SAMPLE_RATE
                if flush:
                    self._errors.clear()
                    transcript = native.stop()
                    if transcript is None:
                        raise (
                            self._errors[0]
                            if self._errors
                            else ProviderError(
                                "moonshine: stopping the stream returned no transcript",
                                provider=_PROVIDER,
                            )
                        )
                    native.start()
                elif self._since_update >= stt.update_interval:
                    flags = 1 if stt.force_updates else 0  # MOONSHINE_FLAG_FORCE_UPDATE
                    transcript = native.update_transcription(flags)
                else:
                    return []
            except VoiceAgentError:
                raise
            except Exception as exc:
                raise _map_error(exc, "transcription") from exc
            self._since_update = 0.0
            return [_line(raw) for raw in transcript.lines if raw.is_updated or flush]

    def _close_sync(self) -> None:
        native, self._native = self._native, None
        if native is not None:
            with self._moonshine.native_lock:
                try:
                    native.close()
                except Exception:
                    logger.debug("moonshine: closing a stream failed", exc_info=True)

    # ----------------------------------------------------------------- event loop
    async def _run(self) -> None:
        try:
            await asyncio.to_thread(self._open_sync)
            while True:
                try:
                    items = [await self._input.recv()]
                except ChanClosed:
                    break
                while not self._input.empty():  # batch whatever arrived during the last pass
                    try:
                        items.append(self._input.recv_nowait())
                    except ChanClosed:  # pragma: no cover - closed between the two calls
                        break
                pending: list[npt.NDArray[np.float32]] = []
                for item in items:
                    if self.is_flush(item):
                        await self._process(pending, flush=True)
                        pending = []
                    else:
                        assert isinstance(item, AudioFrame)
                        pending.append(item.to_float32())
                if pending:
                    await self._process(pending, flush=False)
        finally:
            await asyncio.shield(asyncio.to_thread(self._close_sync))

    async def _process(self, chunks: list[npt.NDArray[np.float32]], *, flush: bool) -> None:
        samples = np.concatenate(chunks) if chunks else None
        if flush:
            offset = self._offset
            lines = await asyncio.to_thread(self._process_sync, samples, True)
            self._offset = self._pushed  # the native stream restarted at time 0
            finals = self._handle(lines, offset, flush=True)
            self._seen.clear()
            self._done.clear()
            self._interims.clear()
            self._speaking = False
            if not finals:
                self._emit(
                    STTEvent(
                        STTEventType.FINAL_TRANSCRIPT,
                        Transcript(text="", language=self._language),
                        segment_id=None,
                    )
                )
        elif samples is not None:
            lines = await asyncio.to_thread(self._process_sync, samples, False)
            self._handle(lines, self._offset, flush=False)

    def _handle(self, lines: list[_Line], offset: float, *, flush: bool) -> int:
        """Emit events for updated lines; returns the number of final transcripts emitted."""
        finals = 0
        for line in lines:
            segment_id = f"moonshine_{line.line_id}"
            text = line.text.strip()
            if line.line_id not in self._seen:
                self._seen.add(line.line_id)
                if not self._speaking and not flush:
                    self._speaking = True
                    self._emit(STTEvent(STTEventType.START_OF_SPEECH, segment_id=segment_id))
            if line.line_id in self._done:
                continue
            transcript = Transcript(
                text=text,
                language=self._language,
                start_time=offset + line.start_time,
                end_time=offset + line.start_time + line.duration,
                words=(
                    [
                        WordTiming(w.word, offset + w.start, offset + w.end, w.confidence)
                        for w in line.words
                    ]
                    if line.words is not None
                    else None
                ),
            )
            if line.is_complete:
                self._done.add(line.line_id)
                self._interims.pop(line.line_id, None)
                if not text:
                    continue
                finals += 1
                self._emit(STTEvent(STTEventType.FINAL_TRANSCRIPT, transcript, segment_id))
                if not flush:
                    # The runtime's own VAD ended the line (the cascade ignores this when
                    # it runs a VAD; VAD-less pipelines use it for endpointing).
                    self._speaking = False
                    self._emit(STTEvent(STTEventType.END_OF_SPEECH, transcript, segment_id))
            elif (
                text
                and self._moonshine.interim_results
                and self._interims.get(line.line_id) != text
            ):
                self._interims[line.line_id] = text
                self._emit(STTEvent(STTEventType.INTERIM_TRANSCRIPT, transcript, segment_id))
        return finals


# ---------------------------------------------------------------------- helpers
def _moonshine_language(language: str | None) -> str | None:
    """``"en-US"``/``"pt_BR"`` -> ``"en"``/``"pt"``; empty or ``"auto"`` -> default (English)."""
    if not language:
        return None
    code = language.strip().replace("_", "-").split("-")[0].lower()
    return None if code in ("", "auto", "multi") else code


def _option_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _line(raw: Any) -> _Line:
    words = getattr(raw, "words", None)
    return _Line(
        line_id=int(raw.line_id),
        text=str(raw.text or ""),
        start_time=float(raw.start_time),
        duration=float(raw.duration),
        is_complete=bool(raw.is_complete),
        is_updated=bool(raw.is_updated),
        words=(
            [
                WordTiming(
                    word=str(w.word).strip(),
                    start=float(w.start),
                    end=float(w.end),
                    confidence=float(w.confidence),
                )
                for w in words
            ]
            if words
            else None
        ),
    )


def _log_progress(model: str) -> Any:
    state = {"last": -1}

    def progress(fraction: float, name: str) -> None:
        step = int(fraction * 10)
        if step != state["last"]:
            state["last"] = step
            logger.info("moonshine: downloading %s: %d%% (%s)", model, step * 10, name)

    return progress


def _map_error(exc: BaseException, action: str) -> VoiceAgentError:
    """Translate ``moonshine-voice`` / download failures to library errors.

    Matched by class name so that this module (and its tests) never imports the package or
    ``requests`` at module level.
    """
    if isinstance(exc, VoiceAgentError):
        return exc
    message = f"moonshine {action} failed: {exc}"
    names = {cls.__name__ for cls in type(exc).__mro__}
    if "MoonshineInvalidArgumentError" in names or isinstance(exc, (ValueError, TypeError)):
        return ConfigurationError(message)
    if names & {"RequestException", "ConnectionError", "Timeout"} or isinstance(
        exc, (ConnectionError, TimeoutError)
    ):
        return ProviderConnectionError(message, provider=_PROVIDER)
    return ProviderError(message, provider=_PROVIDER)
