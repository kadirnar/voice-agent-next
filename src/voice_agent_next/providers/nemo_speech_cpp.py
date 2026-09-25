"""NVIDIA NeMo-Speech.cpp: Nemotron streaming STT, Parakeet and Magpie TTS (`nemo-speech serve`).

`NeMo-Speech.cpp <https://github.com/NVIDIA/NeMo-Speech.cpp>`_ is a ggml runtime for
NVIDIA's speech models (CPU, CUDA, Metal, Vulkan; Linux, macOS and Windows). Its
``nemo-speech serve`` command hosts one ASR model and/or one TTS model behind an HTTP API
(``docs/server.md``, ``docs/api.md`` upstream). This module is a client for it:

* ``stt="nemo-speech-cpp/nemotron-en"`` — :class:`NeMoSpeechCppSTT`.

  - **Streaming** (:meth:`STT.stream`) over the project-specific realtime WebSocket
    ``/v1/audio/transcriptions/realtime``: an optional ``session.update`` JSON event, then
    binary little-endian PCM16 frames; ``input_audio_buffer.commit`` finalizes. The server
    answers with ``conversation.item.input_audio_transcription.delta`` (a text *suffix* per
    audio frame, often empty, or the whole partial when it was rewritten),
    ``...completed`` (the final ``transcript``, with ``words`` when asked) and
    ``input_audio_buffer.committed``. With server endpointing on
    (``asr.endpointing.enable``) finals also arrive mid-stream after a pause. A commit
    ends the server's recognition stream: the next audio starts a fresh one.
  - **Batch** (:meth:`STT.transcribe`) through ``POST /v1/audio/transcriptions`` (the
    OpenAI-compatible subset): :class:`NeMoSpeechCppSTT` is an
    :class:`~voice_agent_next.providers.openai.stt.OpenAICompatibleSTT`.

  Cache-aware models (Nemotron Speech Streaming EN, Nemotron 3.5) stream in chunks of
  80, 160, 560 or 1120 ms, set on the *server* (``asr.streaming.rnnt_right_context`` =
  chunk / 80 ms - 1; :class:`NeMoSpeechCppServer` ``chunk_ms``). Parakeet TDT v3 is
  offline-only: it is used through the HTTP endpoint and the cascade segments the audio
  with its VAD.

* ``tts="nemo-speech-cpp/magpie"`` — :class:`NeMoSpeechCppTTS`: Magpie TTS Multilingual
  through ``POST /v1/audio/speech`` (OpenAI-compatible subset; the complete audio is
  returned at once, 22.05 kHz mono).

:class:`NeMoSpeechCppServer` runs ``nemo-speech serve`` as a managed subprocess
(``serve=True`` on the STT/TTS, or shared between both with ``server=``). The binary comes
from NVIDIA's installer or release archives (it is not on PyPI). Only core dependencies are
used (``websockets`` and ``httpx``). See ``docs/providers/nemo-speech-cpp.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlsplit, urlunsplit

import httpx
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus, InvalidURI

from ..audio.frame import AudioFrame
from ..errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    for_status,
)
from ..models import ModelFile, register_model
from ..registry import register_provider
from ..stt import STTCapabilities, STTEvent, STTEventType, STTStream, Transcript, WordTiming
from ..tts import NormalizeOption
from ..utils.aio import ChanClosed, cancel_and_wait
from ..utils.ids import new_id
from ..utils.log import logger
from ._ws import close_ws, raise_task_error
from .openai._http import is_loopback
from .openai.stt import OpenAICompatibleSTT
from .openai.tts import OpenAICompatibleTTS

__all__ = [
    "ASR_MODELS",
    "CHUNK_SIZES_MS",
    "NeMoSpeechCppSTT",
    "NeMoSpeechCppServer",
    "NeMoSpeechCppStream",
    "NeMoSpeechCppTTS",
    "find_executable",
]

PROVIDER = "nemo_speech_cpp"
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
BASE_URL_ENV = ("NEMO_SPEECH_BASE_URL",)
API_KEY_ENV = ("NEMO_SPEECH_API_KEY", "NEMO_SPEECH_HTTP_API_KEY")
EXECUTABLE_ENV = "NEMO_SPEECH_BIN"
REALTIME_PATH = "audio/transcriptions/realtime"
SAMPLE_RATE = 16_000
"""Input rate of the shipped ASR models (the stream is resampled to it)."""
TTS_SAMPLE_RATE = 22_050
"""Output rate of Magpie TTS with the NanoCodec 22 kHz decoder."""
DEFAULT_STT_MODEL = "nemotron-en"
DEFAULT_TTS_MODEL = "magpie"
CHUNK_SIZES_MS = (80, 160, 560, 1120)
"""Chunk sizes the cache-aware Nemotron models were trained for (``att_context_size``
right contexts 0, 1, 6 and 13 encoder frames of 80 ms)."""
_ENCODER_FRAME_MS = 80
_POLICY_VIOLATION = 1008  # the server closes the socket with it on a wrong API key
_NORMAL_CLOSE = (1000, 1001, 1005)


@dataclass(frozen=True)
class _ASRModel:
    name: str
    repo: str
    revision: str
    filename: str
    size: int
    sha256: str
    streaming: bool
    license: str
    languages: str
    description: str
    aliases: tuple[str, ...] = ()


# Pinned like the upstream model index (models/index.json of NeMo-Speech.cpp v0.1.0), so
# the managed server and `nemo-speech pull` load the same, SHA-256-verified GGUFs.
_ASR_CATALOG = (
    _ASRModel(
        "nemotron-en",
        "nvidia/nemotron-speech-streaming-en-0.6b",
        "ebe59e5a817142986528bbbee5dba8db7b38ed50",
        "nemotron-speech-streaming-en-0.6b.q8_0.gguf",
        699_872_960,
        "d9a01898d2a611c8764e23a1c2f45e70bbd5a425dc4de93692ac951dd603812d",
        True,
        "NVIDIA Open Model License",
        "en",
        "Nemotron Speech Streaming EN 0.6B (cache-aware RNNT, 80-1120 ms chunks), Q8 GGUF",
        ("nemotron-speech-streaming-en-0.6b",),
    ),
    _ASRModel(
        "nemotron-3.5",
        "nvidia/nemotron-3.5-asr-streaming-0.6b",
        "1c8deaecc64b91f034d73e08dd8b64625eb3395d",
        "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf",
        741_548_352,
        "a5c435f294eea8f88ce68dd27b8c3bfea7f777cb2fbba04fcd30eaa555f429ae",
        True,
        "NVIDIA Open Model License (OpenMDW 1.1)",
        "multilingual (40+ locales)",
        "Nemotron 3.5 ASR streaming 0.6B (cache-aware, prompt-conditioned), Q8 GGUF",
        ("nemotron-asr", "nemotron-3.5-asr-streaming-0.6b"),
    ),
    _ASRModel(
        "parakeet-tdt",
        "nvidia/parakeet-tdt-0.6b-v3",
        "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
        "parakeet-tdt-0.6b-v3.q8_0.gguf",
        713_975_456,
        "e3880d0aaaaf2c308ea2c35016b2b895c423eb3fda924c1b463d1c19b7f4d32e",
        False,
        "CC-BY-4.0",
        "multilingual (25 European)",
        "Parakeet TDT 0.6B v3 (offline only), Q8 GGUF",
        ("parakeet-tdt-0.6b-v3",),
    ),
    _ASRModel(
        "parakeet-ctc",
        "nvidia/parakeet-ctc-1.1b",
        "20e63a0fed6aedba145b74b826dbd41df0941730",
        "parakeet-ctc-1.1b.q8_0.gguf",
        1_178_100_960,
        "6584fc0fdacf1c220401ea4c3a1d5b44454b655c141cb8672178072c203d92b8",
        True,
        "CC-BY-4.0",
        "en",
        "Parakeet CTC 1.1B (buffered streaming), Q8 GGUF",
        ("parakeet-ctc-1.1b",),
    ),
)
ASR_MODELS: dict[str, _ASRModel] = {m.name: m for m in _ASR_CATALOG}
_BY_ALIAS: dict[str, _ASRModel] = {a: m for m in _ASR_CATALOG for a in (m.name, *m.aliases)}
_BY_ALIAS.update({m.repo: m for m in _ASR_CATALOG})

for _m in _ASR_CATALOG:
    register_model(
        PROVIDER,
        _m.name,
        kind="stt",
        files=[
            ModelFile.from_hf(
                _m.repo, _m.filename, revision=_m.revision, sha256=_m.sha256, size=_m.size
            )
        ],
        license=_m.license,
        languages=_m.languages,
        description=_m.description,
        aliases=_m.aliases,
    )


def _asr_model(name: str | None) -> _ASRModel | None:
    return _BY_ALIAS.get((name or "").strip().lower()) if name else None


def _offline_only(model: str) -> bool:
    info = _asr_model(model)
    return info is not None and not info.streaming


def _float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


# ------------------------------------------------------------------------------ server
def find_executable(executable: str | os.PathLike[str] | None = None) -> Path:
    """The ``nemo-speech`` binary: ``executable``, ``$NEMO_SPEECH_BIN``, ``PATH``, then the
    installers' default locations. Raises :class:`ConfigurationError` when none exists."""
    candidates: list[Path] = []
    explicit = executable if executable is not None else os.environ.get(EXECUTABLE_ENV)
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path
        raise ConfigurationError(f"{PROVIDER}: the nemo-speech binary {str(path)!r} does not exist")
    found = shutil.which("nemo-speech")
    if found:
        return Path(found)
    exe = "nemo-speech.exe" if sys.platform == "win32" else "nemo-speech"
    candidates.append(Path.home() / ".local" / "bin" / exe)
    local_app = os.environ.get("LOCALAPPDATA")
    if local_app:
        candidates.append(Path(local_app) / "Programs" / "NeMoSpeech" / "bin" / exe)
    for path in candidates:
        if path.is_file():
            return path
    raise ConfigurationError(
        f"{PROVIDER}: the nemo-speech binary was not found. Install NeMo-Speech.cpp "
        "(https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/install.md), put it on "
        f"PATH or set {EXECUTABLE_ENV}; see docs/providers/nemo-speech-cpp.md"
    )


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def right_context(chunk_ms: int) -> int:
    """``asr.streaming.rnnt_right_context`` for a cache-aware chunk of ``chunk_ms``."""
    if chunk_ms not in CHUNK_SIZES_MS:
        raise ConfigurationError(
            f"{PROVIDER}: chunk_ms must be one of {list(CHUNK_SIZES_MS)} (the chunk sizes the "
            f"cache-aware models were trained for), got {chunk_ms}"
        )
    return chunk_ms // _ENCODER_FRAME_MS - 1


class NeMoSpeechCppServer:
    """``nemo-speech serve`` as a managed subprocess.

    ``start()`` starts the server (models given by an indexed name are downloaded first,
    by this library for the ASR models of the catalog and by the server otherwise) and
    waits until ``GET /ready`` succeeds; ``stop()`` terminates it. Also an async context
    manager::

        async with NeMoSpeechCppServer(asr_model="nemotron-en", tts_model="magpie") as srv:
            stt = NeMoSpeechCppSTT(server=srv)
            tts = NeMoSpeechCppTTS(server=srv)

    Args:
        asr_model: ASR model: a catalog name (``nemotron-en``, ``nemotron-3.5``,
            ``parakeet-tdt``, ``parakeet-ctc``), another indexed name or repository id the
            server resolves itself, or a local GGUF path. ``None``: no ASR.
        tts_model: TTS model (``magpie`` — the server also fetches the NanoCodec decoder and
            the tokenizer), or a local Magpie GGUF path (then pass ``--codec-model`` and
            ``--tokenizer-dir`` in ``args``). ``None``: no TTS.
        device: ``auto`` (default), ``cpu``, ``cuda[:N]``, ``metal`` or ``vulkan[:N]``.
        chunk_ms: cache-aware streaming chunk (80, 160, 560 or 1120 ms; default: the
            server's, 160 ms). Smaller chunks give earlier partials and finals at more
            compute per second of audio.
        endpointing: server-side end-of-utterance detection: finals are emitted mid-stream
            after ``endpointing_ms`` of trailing silence (off by default: the cascade's VAD
            flushes the stream instead).
        endpointing_ms: trailing silence that ends an utterance (server default 800 ms).
        host: interface to bind (keep the loopback default unless you set ``api_key``).
        port: port to listen on (default: a free port, chosen when the object is created).
        api_key: require ``Authorization: Bearer <key>`` (passed through the environment,
            never on the command line).
        args: extra command-line arguments (``--asr.*``, ``--tts.*``, ``--vad-model``...).
        env: extra environment variables for the server.
        executable: the ``nemo-speech`` binary (default: see :func:`find_executable`).
        startup_timeout: seconds to wait for readiness (model downloads count).
    """

    def __init__(
        self,
        *,
        asr_model: str | os.PathLike[str] | None = None,
        tts_model: str | os.PathLike[str] | None = None,
        device: str | None = None,
        chunk_ms: int | None = None,
        endpointing: bool = False,
        endpointing_ms: float | None = None,
        host: str = "127.0.0.1",
        port: int | None = None,
        api_key: str | None = None,
        args: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        executable: str | os.PathLike[str] | None = None,
        startup_timeout: float = 900.0,
    ) -> None:
        if asr_model is None and tts_model is None:
            raise ConfigurationError(f"{PROVIDER}: the server needs an asr_model or a tts_model")
        if chunk_ms is not None:
            right_context(chunk_ms)  # validate early
        self.asr_model = str(asr_model) if asr_model is not None else None
        self.tts_model = str(tts_model) if tts_model is not None else None
        self.device = device
        self.chunk_ms = chunk_ms
        self.endpointing = endpointing
        self.endpointing_ms = endpointing_ms
        self.host = host
        self.port = port if port is not None else _free_port(host)
        self.api_key = api_key
        self.args = list(args)
        self.env = dict(env or {})
        self.executable = executable
        self.startup_timeout = startup_timeout
        self._proc: subprocess.Popen[bytes] | None = None
        self._log: Any = None
        self._lock = asyncio.Lock()

    @property
    def base_url(self) -> str:
        """The OpenAI-style base URL (``http://host:port/v1``)."""
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}/v1"

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def resolve_asr_model(self) -> str | None:
        """The ``--asr-model`` value: catalog models are downloaded (blocking) to a path."""
        model = self.asr_model
        if model is None or Path(model).expanduser().is_file():
            return model
        info = _asr_model(model)
        if info is None:
            return model  # an indexed name or repository id: the server resolves it
        from ..utils.download import hf_file

        return str(hf_file(info.repo, info.filename, revision=info.revision, sha256=info.sha256))

    def command(self, exe: str | os.PathLike[str], asr_model: str | None = None) -> list[str]:
        """The server's command line (``asr_model``: the resolved ``--asr-model``)."""
        cmd = [str(exe), "serve", "--host", self.host, "--port", str(self.port), "--no-ui"]
        asr = asr_model if asr_model is not None else self.asr_model
        if asr is not None:
            cmd += ["--asr-model", asr]
            if self.chunk_ms is not None:
                cmd.append(f"--asr.streaming.rnnt_right_context={right_context(self.chunk_ms)}")
            if self.endpointing:
                cmd.append("--asr.endpointing.enable=true")
            if self.endpointing_ms is not None:
                cmd.append(f"--asr.endpointing.stop_history_eou_ms={self.endpointing_ms:g}")
        if self.tts_model is not None:
            cmd += ["--tts-model", self.tts_model]
        if self.device:
            cmd += ["--device", self.device]
        return cmd + self.args

    async def start(self) -> str:
        """Start the server (no-op when it is running) and return its base URL."""
        async with self._lock:
            if self.running:
                return self.base_url
            exe = find_executable(self.executable)
            asr = await asyncio.to_thread(self.resolve_asr_model)
            cmd = self.command(exe, asr)
            env = {**os.environ, **self.env}
            if self.api_key:
                env["NEMO_SPEECH_HTTP_API_KEY"] = self.api_key
            self._log = tempfile.TemporaryFile()  # noqa: SIM115 - closed in stop()
            logger.info("%s: starting %s", PROVIDER, " ".join(cmd))
            self._proc = subprocess.Popen(  # noqa: ASYNC220 - returns at once; no pipes to drain
                cmd, stdin=subprocess.DEVNULL, stdout=self._log, stderr=subprocess.STDOUT, env=env
            )
            try:
                await self._wait_ready()
            except BaseException:
                await self._terminate()
                raise
            logger.info("%s: server ready at %s", PROVIDER, self.base_url)
            return self.base_url

    async def _wait_ready(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.startup_timeout
        url = self.base_url.removesuffix("/v1") + "/ready"
        async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
            while True:
                proc = self._proc
                if proc is None or proc.poll() is not None:
                    code = None if proc is None else proc.returncode
                    raise ProviderError(
                        f"{PROVIDER}: the server exited during startup (code {code}):\n"
                        f"{self.log_tail()}",
                        provider=PROVIDER,
                    )
                with contextlib.suppress(httpx.HTTPError):
                    response = await client.get(url)
                    if response.status_code == 200:
                        return
                if loop.time() > deadline:
                    raise ProviderTimeoutError(
                        f"{PROVIDER}: the server was not ready within {self.startup_timeout:g}s:"
                        f"\n{self.log_tail()}",
                        provider=PROVIDER,
                    )
                await asyncio.sleep(0.25)

    def log_tail(self, lines: int = 20) -> str:
        """The last lines of the server's output."""
        if self._log is None:
            return ""
        try:
            self._log.flush()
            self._log.seek(0)
            text = self._log.read().decode("utf-8", "replace")
        except (OSError, ValueError):
            return ""
        return "\n".join(text.splitlines()[-lines:])

    async def stop(self) -> None:
        """Terminate the server (and wait for it to exit)."""
        async with self._lock:
            await self._terminate()

    async def _terminate(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                await asyncio.to_thread(proc.wait, 10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                await asyncio.to_thread(proc.wait)
        if self._log is not None:
            self._log.close()
            self._log = None

    async def __aenter__(self) -> NeMoSpeechCppServer:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()


class _ManagedServer:
    """Shared ``server=`` / ``serve=True`` handling of the STT and TTS clients."""

    server: NeMoSpeechCppServer | None
    _owns_server: bool

    def _setup_server(
        self,
        server: NeMoSpeechCppServer | None,
        serve: bool,
        base_url: str | None,
        make: Any,
    ) -> str | None:
        if server is not None and serve:
            raise ConfigurationError(f"{PROVIDER}: pass either server= or serve=True, not both")
        if (server is not None or serve) and base_url is not None:
            raise ConfigurationError(f"{PROVIDER}: base_url is set by the managed server")
        self._owns_server = serve
        self.server = make() if serve else server
        return self.server.base_url if self.server is not None else base_url

    async def _ensure_server(self) -> None:
        if self.server is not None and not self.server.running:
            await self.server.start()

    async def _stop_server(self) -> None:
        if self.server is not None and self._owns_server:
            await self.server.stop()


# ------------------------------------------------------------------------------ STT
@register_provider(
    "stt",
    PROVIDER,
    description="NeMo-Speech.cpp server: Nemotron cache-aware streaming, Parakeet (local)",
    default_model=DEFAULT_STT_MODEL,
    models=tuple(ASR_MODELS),
    requires=("httpx", "websockets"),
    local=True,
)
class NeMoSpeechCppSTT(_ManagedServer, OpenAICompatibleSTT):
    """Speech recognition with a ``nemo-speech serve`` server.

    Streams (:meth:`STT.stream`) use the realtime WebSocket; :meth:`STT.transcribe` posts
    the audio to ``/v1/audio/transcriptions`` (one request, lowest latency for a complete
    utterance).

    Events of a stream:

    * ``INTERIM_TRANSCRIPT`` — the text of the utterance so far (the server sends only the
      new suffix; it is accumulated here);
    * ``FINAL_TRANSCRIPT`` — the server's ``...completed`` event: after
      :meth:`~voice_agent_next.stt.STTStream.flush` (``input_audio_buffer.commit``; an
      empty final when there was nothing to transcribe) or, with server endpointing on,
      after a pause;
    * ``START_OF_SPEECH`` / ``END_OF_SPEECH`` — around every utterance that produced text.

    Args:
        model: the model the server runs (``nemotron-en`` by default, ``nemotron-3.5``,
            ``parakeet-tdt``, ``parakeet-ctc`` or a GGUF path). The server serves the one
            model it loaded; the name picks the managed server's model and whether streaming
            is possible (``parakeet-tdt`` is offline-only).
        base_url: server root including ``/v1`` (default ``NEMO_SPEECH_BASE_URL``, then
            ``http://127.0.0.1:8080/v1``).
        api_key: the server's ``--api-key`` (default ``NEMO_SPEECH_API_KEY``).
        language: language code (``en-US``, ``es-ES``...; ``auto`` detects it with
            Nemotron 3.5).
        streaming: use the realtime WebSocket for streams (default: unless the model is
            offline-only); ``False`` makes this a batch STT the cascade segments with its VAD.
        word_timestamps: word timings on finals (seconds from the start of the stream).
        automatic_punctuation: punctuation and capitalization (default on).
        verbatim: skip inverse text normalization.
        profanity_filter: mask words of the server's profanity list.
        endpointing_ms: end-of-utterance silence for this session (with server endpointing
            enabled).
        speech_contexts: word boosting, ``[{"phrases": [...], "boost": 3.0}]``.
        prompt: one phrase to boost (OpenAI-compatible ``prompt``; boost 10).
        chunk_ms: audio per WebSocket frame (the server emits one result per frame; the
            model's own chunk size is a server setting, see :class:`NeMoSpeechCppServer`).
        serve: run a managed ``nemo-speech serve`` for this STT (see ``server_options``).
        server: a :class:`NeMoSpeechCppServer` to use (not stopped by :meth:`aclose`).
        server_options: keyword arguments of the managed server (``chunk_ms``,
            ``endpointing``, ``device``, ``executable``...).
        connect_timeout, timeout, close_timeout, headers, extra, http_client: as for
            :class:`~voice_agent_next.providers.openai.stt.OpenAISTT`.
    """

    provider = PROVIDER
    DEFAULT_MODEL: ClassVar[str] = DEFAULT_STT_MODEL
    DEFAULT_BASE_URL: ClassVar[str] = DEFAULT_BASE_URL
    BASE_URL_ENV: ClassVar[tuple[str, ...]] = BASE_URL_ENV
    API_KEY_ENV: ClassVar[tuple[str, ...]] = API_KEY_ENV
    DEFAULT_SAMPLE_RATE: ClassVar[int] = SAMPLE_RATE
    NOT_FOUND_HINT: ClassVar[str] = "is the server running with an ASR model? check base_url"

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        language: str | None = None,
        streaming: bool | None = None,
        word_timestamps: bool = False,
        automatic_punctuation: bool = True,
        verbatim: bool = False,
        profanity_filter: bool = False,
        endpointing_ms: float | None = None,
        speech_contexts: Sequence[Mapping[str, Any]] | None = None,
        prompt: str | None = None,
        chunk_ms: int = 20,
        serve: bool = False,
        server: NeMoSpeechCppServer | None = None,
        server_options: Mapping[str, Any] | None = None,
        connect_timeout: float = 10.0,
        timeout: float = 60.0,
        close_timeout: float = 5.0,
        headers: Mapping[str, str] | None = None,
        extra: Mapping[str, Any] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved = model or self.DEFAULT_MODEL
        offline = _offline_only(resolved)
        if streaming is None:
            streaming = not offline
        elif streaming and offline:
            raise ConfigurationError(
                f"{PROVIDER}: {resolved} is offline-only (no streaming recognition); use "
                "streaming=False or a streaming model (nemotron-en, nemotron-3.5)"
            )
        if chunk_ms <= 0:
            raise ConfigurationError("chunk_ms must be > 0")
        if isinstance(speech_contexts, Mapping):
            raise ConfigurationError("speech_contexts takes a list of {phrases, boost} objects")

        def make() -> NeMoSpeechCppServer:
            return NeMoSpeechCppServer(
                asr_model=resolved, api_key=api_key, **dict(server_options or {})
            )

        url = self._setup_server(server, serve, base_url, make)
        form: dict[str, Any] = {}
        if not automatic_punctuation:
            form["automatic_punctuation"] = False
        if verbatim:
            form["verbatim"] = True
        if profanity_filter:
            form["profanity_filter"] = True
        if speech_contexts:
            form["speech_contexts"] = json.dumps([dict(c) for c in speech_contexts])
        super().__init__(
            model=resolved,
            api_key=api_key if self.server is None else (self.server.api_key or ""),
            base_url=url,
            language=language,
            prompt=prompt,
            realtime=False,
            word_timestamps=word_timestamps,
            sample_rate=SAMPLE_RATE,
            headers=headers,
            extra={**form, **dict(extra or {})},
            chunk_ms=chunk_ms,
            connect_timeout=connect_timeout,
            timeout=timeout,
            close_timeout=close_timeout,
            http_client=http_client,
        )
        self.streaming = streaming
        self.capabilities = STTCapabilities(
            streaming=streaming,
            interim_results=streaming,
            word_timestamps=word_timestamps,
            language_detection=language == "auto",
        )
        self.automatic_punctuation = automatic_punctuation
        self.verbatim = verbatim
        self.profanity_filter = profanity_filter
        self.endpointing_ms = endpointing_ms
        self.speech_contexts = [dict(c) for c in speech_contexts or ()]

    # ---------------------------------------------------------------- configuration
    def session_config(self, language: str | None = None) -> dict[str, Any]:
        """The ``session`` of the ``session.update`` sent when a stream connects."""
        session: dict[str, Any] = {"sample_rate": SAMPLE_RATE}
        lang = language or self.language
        if lang:
            session["language"] = lang
        session["automatic_punctuation"] = self.automatic_punctuation
        if self.verbatim:
            session["verbatim"] = True
        if self.profanity_filter:
            session["profanity_filter"] = True
        if self.word_timestamps:
            session["word_timestamps"] = True
        if self.endpointing_ms is not None:
            session["endpointing_ms"] = self.endpointing_ms
        if self.speech_contexts:
            session["speech_contexts"] = self.speech_contexts
        if self.prompt:
            session["prompt"] = self.prompt
        return session

    def realtime_url(self) -> str:
        """``ws(s)://host:port/v1/audio/transcriptions/realtime``."""
        parts = urlsplit(self.endpoint.url(REALTIME_PATH))
        scheme = "wss" if parts.scheme == "https" else "ws"
        return urlunsplit((scheme, parts.netloc, parts.path, parts.query, ""))

    def languages_for(self, language: str | None) -> list[str]:
        """Language codes are passed through unchanged (``en-US``, ``auto``...)."""
        lang = language or self.language
        return [lang] if lang else []

    # -------------------------------------------------------------------- streaming
    def _create_stream(self, *, language: str | None) -> STTStream:
        return NeMoSpeechCppStream(self, language=language)

    def stream(self, *, language: str | None = None) -> NeMoSpeechCppStream:
        return super().stream(language=language)  # type: ignore[return-value]

    async def _open_realtime(self) -> ClientConnection:
        await self._ensure_server()
        url = self.realtime_url()
        kwargs: dict[str, Any] = {}
        if is_loopback(url):
            kwargs["proxy"] = None  # never route the local server through a system proxy
        try:
            return await connect(
                url,
                additional_headers=self.endpoint.request_headers(),
                open_timeout=self.connect_timeout,
                close_timeout=2.0,
                max_size=None,
                compression=None,  # PCM audio does not compress; save the CPU
                **kwargs,
            )
        except InvalidStatus as exc:
            status = exc.response.status_code
            msg = f"{PROVIDER}: realtime handshake rejected with HTTP {status} ({url})"
            if status == 404:
                msg += ": is an ASR model loaded? (the route exists only with --asr-model)"
            raise for_status(status, msg, provider=PROVIDER) from exc
        except InvalidURI as exc:
            raise ConfigurationError(f"{PROVIDER}: invalid realtime URL {url!r}: {exc}") from exc
        except TimeoutError as exc:
            raise ProviderTimeoutError(
                f"{PROVIDER}: timed out connecting to {url}", provider=PROVIDER
            ) from exc
        except (OSError, InvalidHandshake) as exc:
            raise ProviderConnectionError(
                f"{PROVIDER}: cannot connect to {url}: {exc} (is `nemo-speech serve` running?)",
                provider=PROVIDER,
            ) from exc

    # ------------------------------------------------------------------------ batch
    async def _recognize(self, audio: AudioFrame, *, language: str | None) -> Transcript:
        await self._ensure_server()
        return await super()._recognize(audio, language=language)

    async def warmup(self) -> None:
        """Start the managed server, then open the HTTP connection (``GET /models``)."""
        await self._ensure_server()
        await super().warmup()

    async def aclose(self) -> None:
        await super().aclose()
        await self._stop_server()


class NeMoSpeechCppStream(STTStream):
    """One realtime transcription session (see :class:`NeMoSpeechCppSTT`).

    :attr:`session` holds the server's ``session.created`` fields (``model``,
    ``sample_rate``) once connected.
    """

    def __init__(self, stt: NeMoSpeechCppSTT, *, language: str | None) -> None:
        self._nemo = stt
        self._chunk_bytes = max(2, round(SAMPLE_RATE * stt.chunk_ms / 1000) * 2)
        self._buf = bytearray()
        self._closing = False
        self._server_error: ProviderError | None = None
        self._segment_id = new_id("seg_")
        self._speaking = False
        self._partial = ""
        self._sent_samples = 0
        self._origin = 0.0  # stream time at which the server's current recognition began
        self._pending: deque[float] = deque()  # origins of the commits not yet acknowledged
        self._answered = True  # the oldest pending commit got a final already
        self._drained = asyncio.Event()
        self._drained.set()
        self.session: dict[str, Any] = {}
        super().__init__(stt, language=language)

    # ------------------------------------------------------------------- plumbing
    async def _run(self) -> None:
        ws = await self._nemo._open_realtime()
        receiver = asyncio.create_task(self._recv_loop(ws), name="nemo-speech-stt-recv")
        sender = asyncio.create_task(self._send_loop(ws), name="nemo-speech-stt-send")
        try:
            await asyncio.wait({receiver, sender}, return_when=asyncio.FIRST_COMPLETED)
            raise_task_error(receiver, sender)
            if not sender.done():
                raise self._server_error or ProviderConnectionError(
                    f"{PROVIDER}: the server closed the realtime session", provider=PROVIDER
                )
            # input ended: wait for the answers to the last commits, then hang up
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(self._nemo.close_timeout):
                    waiter = asyncio.ensure_future(self._drained.wait())
                    try:
                        await asyncio.wait({waiter, receiver}, return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        waiter.cancel()
            raise_task_error(receiver)
            if not self._drained.is_set():
                logger.warning(
                    "%s: no answer to the last commit %.1fs after the input ended",
                    PROVIDER,
                    self._nemo.close_timeout,
                )
            self._finish_session()
        finally:
            self._closing = True
            await cancel_and_wait(sender, receiver)
            await close_ws(ws)

    async def _send_loop(self, ws: ClientConnection) -> None:
        try:
            config = self._nemo.session_config(self._language)
            await ws.send(json.dumps({"type": "session.update", "session": config}))
            while True:
                try:
                    item = await self._input.recv()
                except ChanClosed:
                    break
                if self.is_flush(item):
                    if self._buf:
                        await self._send_audio(ws, bytes(self._buf))
                        self._buf.clear()
                    self._pending.append(self._sent_samples / SAMPLE_RATE)
                    self._drained.clear()
                    if len(self._pending) == 1:
                        self._answered = False
                    await ws.send('{"type": "input_audio_buffer.commit"}')
                    continue
                assert isinstance(item, AudioFrame)
                self._buf += item.data
                while len(self._buf) >= self._chunk_bytes:
                    await self._send_audio(ws, bytes(self._buf[: self._chunk_bytes]))
                    del self._buf[: self._chunk_bytes]
            if self._buf:  # end_input() flushes first, so this is only reached on aclose()
                self._buf.clear()
            self._closing = True
        except ConnectionClosed as exc:
            raise self._server_error or self._close_error(exc) from exc

    async def _send_audio(self, ws: ClientConnection, data: bytes) -> None:
        await ws.send(data)
        self._sent_samples += len(data) // 2

    async def _recv_loop(self, ws: ClientConnection) -> None:
        try:
            async for message in ws:
                if isinstance(message, str):
                    self._on_message(message)
        except ConnectionClosed as exc:
            if self._server_error is not None:
                raise self._server_error from exc
            code = exc.rcvd.code if exc.rcvd is not None else None
            if not (self._closing and (code is None or code in _NORMAL_CLOSE)):
                raise self._close_error(exc) from exc
            return
        if self._server_error is not None:
            raise self._server_error
        if not self._closing:
            raise ProviderConnectionError(
                f"{PROVIDER}: the server closed the realtime session", provider=PROVIDER
            )

    @staticmethod
    def _close_error(exc: ConnectionClosed) -> ProviderError:
        frame = exc.rcvd
        if frame is None:
            error: ProviderError = ProviderConnectionError(
                f"{PROVIDER}: realtime connection lost (no close frame)", provider=PROVIDER
            )
        elif frame.code == _POLICY_VIOLATION:
            error = AuthenticationError(
                f"{PROVIDER}: the server rejected the session: {frame.reason or 'policy violation'}"
                " (check api_key / NEMO_SPEECH_API_KEY)",
                provider=PROVIDER,
                status_code=frame.code,
            )
        else:
            error = ProviderConnectionError(
                f"{PROVIDER}: realtime session closed ({frame.code} {frame.reason})".rstrip(),
                provider=PROVIDER,
                status_code=frame.code,
            )
        error.__cause__ = exc
        return error

    # --------------------------------------------------------------------- events
    def _on_message(self, raw: str) -> None:
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("%s: ignoring a non-JSON message", PROVIDER)
            return
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        if kind == "conversation.item.input_audio_transcription.delta":
            self._on_delta(event)
        elif kind == "conversation.item.input_audio_transcription.completed":
            self._on_completed(event)
        elif kind == "input_audio_buffer.committed":
            self._on_committed()
        elif kind == "session.created":
            session = event.get("session")
            self.session = dict(session) if isinstance(session, dict) else {}
            logger.debug("%s: session created (%s)", PROVIDER, self.session.get("model"))
        elif kind == "error":
            err = event.get("error")
            err = err if isinstance(err, dict) else {}
            self._server_error = ProviderError(
                f"{PROVIDER}: realtime error ({err.get('type') or 'error'}): "
                f"{err.get('message') or 'no message'}",
                provider=PROVIDER,
            )
            raise self._server_error
        else:  # session.updated, input_audio_buffer.cleared...
            logger.debug("%s: ignoring %s", PROVIDER, kind)

    def _on_delta(self, event: dict[str, Any]) -> None:
        delta = event.get("delta")
        if not isinstance(delta, str) or not delta:
            return  # one (mostly empty) delta arrives per audio frame
        if _rewrites(self._partial, delta):
            self._partial = delta  # the server rewrote the partial: this is the whole text
        else:
            self._partial += delta
        text = self._partial.strip()
        if not text:
            return
        self._begin()
        self._event(STTEventType.INTERIM_TRANSCRIPT, Transcript(text, language=self._lang()))

    def _on_completed(self, event: dict[str, Any]) -> None:
        text = str(event.get("transcript") or "").strip()
        answers_commit = bool(self._pending) and not self._answered
        if not text and not answers_commit:
            return  # an empty mid-stream final: nothing to report
        if self._pending:
            self._answered = True
        words = self._words(event.get("words"))
        transcript = Transcript(
            text,
            language=self._lang(),
            confidence=_mean([w.confidence for w in words or ()]),
            start_time=words[0].start if words else None,
            end_time=words[-1].end if words else None,
            words=words,
        )
        if text:
            self._begin()
        self._event(STTEventType.FINAL_TRANSCRIPT, transcript)
        if self._speaking:
            self._event(STTEventType.END_OF_SPEECH, transcript)
        self._next_segment()

    def _on_committed(self) -> None:
        if self._pending:
            self._origin = self._pending.popleft()
        self._answered = not self._pending
        if not self._pending:
            self._drained.set()

    def _words(self, items: Any) -> list[WordTiming] | None:
        if not isinstance(items, list):
            return None
        out: list[WordTiming] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            start, end = _float(item.get("start")), _float(item.get("end"))
            word = str(item.get("word") or "")
            if not word or start is None or end is None:
                continue
            out.append(
                WordTiming(
                    word, self._origin + start, self._origin + end, _float(item.get("confidence"))
                )
            )
        return out or None

    def _begin(self) -> None:
        if not self._speaking:
            self._speaking = True
            self._event(STTEventType.START_OF_SPEECH)

    def _next_segment(self) -> None:
        self._segment_id = new_id("seg_")
        self._partial = ""
        self._speaking = False

    def _finish_session(self) -> None:
        # an utterance still open when the session ended: its partial is all we have
        if self._speaking and self._partial.strip():
            self._on_completed({"transcript": self._partial})

    def _lang(self) -> str | None:
        return self._language or self._nemo.language

    def _event(self, kind: STTEventType, transcript: Transcript | None = None) -> None:
        self._emit(STTEvent(kind, transcript, self._segment_id))


def _words_only(text: str) -> str:
    """Lower-case letters/digits and single spaces: what a rewrite keeps of a partial."""
    return " ".join("".join(c if c.isalnum() else " " for c in text.lower()).split())


def _rewrites(partial: str, delta: str) -> bool:
    """Is ``delta`` a whole rewritten partial rather than a suffix of ``partial``?

    The server sends the suffix when the new partial extends the previous one, else the
    full new partial, with nothing telling the two apart. A full partial starts with a
    word (the server's transcripts never start with a space), while a suffix starts with
    a space (a new word), punctuation or the rest of an unfinished word. So ``delta`` is
    taken as a rewrite when it does not start with a space and either repeats the start
    of the previous partial (the usual case: re-capitalized, re-punctuated or a changed
    word further on) or has at least as many words as a partial of two words or more.
    """
    if not partial.strip() or not delta or delta[0].isspace():
        return False
    old, new = _words_only(partial), _words_only(delta)
    if not old or not new:
        return False  # e.g. punctuation only: an extension
    if len(os.path.commonprefix([old, new])) >= min(len(old), 3):
        return True
    old_words, new_words = len(old.split()), len(new.split())
    return old_words >= 2 and new_words >= old_words


def _mean(values: Sequence[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


# ------------------------------------------------------------------------------ TTS
@register_provider(
    "tts",
    PROVIDER,
    description="NeMo-Speech.cpp server: Magpie TTS Multilingual (local)",
    default_model=DEFAULT_TTS_MODEL,
    models=(DEFAULT_TTS_MODEL,),
    requires=("httpx",),
    local=True,
)
class NeMoSpeechCppTTS(_ManagedServer, OpenAICompatibleTTS):
    """Magpie TTS through ``POST /v1/audio/speech`` of a ``nemo-speech serve`` server.

    The server synthesizes the complete text before answering, so the cascade's sentences
    are synthesized one request at a time (see
    :class:`~voice_agent_next.providers.openai.tts.OpenAITTS`). Responses are WAV (the
    header carries the model rate, 22.05 kHz for the NanoCodec decoder) and are resampled
    to ``sample_rate``.

    Args:
        model: ``magpie`` (the model the managed server loads; informational otherwise).
        voice: a local voice name (listed under ``voices`` of ``GET /v1/models``), a
            model-qualified name (``magpietts.John``) or a zero-based speaker index;
            ``None``: the server's default speaker.
        language: language code of the text (``en-US``, ``es-ES``, ``de-DE``...).
        sample_rate: output rate (default 22050, Magpie's rate).
        base_url, api_key, headers, timeout, connect_timeout, max_retries, http_client,
            extra, normalize: as for :class:`~voice_agent_next.providers.openai.tts.OpenAITTS`.
        serve / server / server_options: as for :class:`NeMoSpeechCppSTT`.
    """

    provider = PROVIDER
    DEFAULT_MODEL: ClassVar[str] = DEFAULT_TTS_MODEL
    DEFAULT_BASE_URL: ClassVar[str] = DEFAULT_BASE_URL
    BASE_URL_ENV: ClassVar[tuple[str, ...]] = BASE_URL_ENV
    API_KEY_ENV: ClassVar[tuple[str, ...]] = API_KEY_ENV
    RESPONSE_FORMAT: ClassVar[str] = "wav"
    PCM_SAMPLE_RATE: ClassVar[int] = TTS_SAMPLE_RATE
    MAX_INPUT_CHARS: ClassVar[int] = 1000
    NOT_FOUND_HINT: ClassVar[str] = "is the server running with a TTS model? check base_url"

    def __init__(
        self,
        *,
        model: str | None = None,
        voice: str | None = None,
        language: str | None = None,
        sample_rate: int | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        serve: bool = False,
        server: NeMoSpeechCppServer | None = None,
        server_options: Mapping[str, Any] | None = None,
        speed: float | None = None,
        response_format: str | None = None,
        extra: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 60.0,
        connect_timeout: float = 10.0,
        max_retries: int = 1,
        http_client: httpx.AsyncClient | None = None,
        clean_text: bool = True,
        normalize: NormalizeOption = None,
        trim_silence: bool = True,
    ) -> None:
        if speed is not None and speed != 1.0:
            raise ConfigurationError(f"{PROVIDER}: Magpie TTS does not support speed={speed}")
        resolved = model or self.DEFAULT_MODEL

        def make() -> NeMoSpeechCppServer:
            return NeMoSpeechCppServer(
                tts_model=resolved, api_key=api_key, **dict(server_options or {})
            )

        url = self._setup_server(server, serve, base_url, make)
        body: dict[str, Any] = {}
        if language:
            body["language"] = language
        super().__init__(
            model=resolved,
            voice=voice,
            api_key=api_key if self.server is None else (self.server.api_key or ""),
            base_url=url,
            sample_rate=sample_rate,
            response_format=response_format,
            extra={**body, **dict(extra or {})},
            headers=headers,
            timeout=timeout,
            connect_timeout=connect_timeout,
            max_retries=max_retries,
            http_client=http_client,
            clean_text=clean_text,
            normalize=normalize,
            trim_silence=trim_silence,
        )
        self.language = language

    async def _speak(self, text: str, voice: str | None, push: Any) -> None:
        await self._ensure_server()
        await super()._speak(text, voice, push)

    async def warmup(self) -> None:
        """Start the managed server, then open the HTTP connection (``GET /models``)."""
        await self._ensure_server()
        await super().warmup()

    async def aclose(self) -> None:
        await super().aclose()
        await self._stop_server()


# ------------------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    """``python -m voice_agent_next.providers.nemo_speech_cpp serve|command``."""
    parser = argparse.ArgumentParser(
        prog="python -m voice_agent_next.providers.nemo_speech_cpp",
        description="Run `nemo-speech serve` with the options this provider uses.",
    )
    parser.add_argument("action", choices=("serve", "command"))
    parser.add_argument("--asr-model", default=None)
    parser.add_argument("--tts-model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--chunk-ms", type=int, default=None, choices=CHUNK_SIZES_MS)
    parser.add_argument("--endpointing", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--executable", default=None)
    args = parser.parse_args(argv)
    if args.asr_model is None and args.tts_model is None:
        args.asr_model = DEFAULT_STT_MODEL
    server = NeMoSpeechCppServer(
        asr_model=args.asr_model, tts_model=args.tts_model, device=args.device,
        chunk_ms=args.chunk_ms, endpointing=args.endpointing, host=args.host, port=args.port,
        executable=args.executable,
    )  # fmt: skip
    exe = find_executable(args.executable)
    if args.action == "command":
        print(" ".join(server.command(exe)))
        return 0
    cmd = server.command(exe, server.resolve_asr_model())
    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
