"""LFM2.5-Audio (Liquid AI) through ``llama-liquid-audio-server``: a local omni model.

``AgentSession(llm="liquid-audio/lfm2.5-audio-1.5b", vad="silero", turn_detector=...)``
runs native speech-to-speech on the cascade: the model hears the user's audio and speaks
its reply itself (no STT, no TTS). See ``docs/providers/liquid-audio.md``.

The server (built from llama.cpp PR ggml-org/llama.cpp#18641, shipped as prebuilt runners
in the ``LiquidAI/LFM2.5-Audio-1.5B-GGUF`` repository) speaks a subset of OpenAI Chat
Completions:

* ``POST /v1/chat/completions`` with ``stream: true`` only; ``system`` and ``user`` roles
  only (no ``assistant`` messages, no tools); user content is text or ``input_audio``
  (WAV) parts;
* the system prompt selects the mode and must be one of a fixed set:
  ``Respond with interleaved text and audio.`` is the conversational one;
* streamed deltas carry ``content`` (text) and ``audio_chunk.data`` (base64 float32 PCM,
  24 kHz mono, one 80 ms chunk per audio step); text runs ahead of the audio;
* the context is **stateful**: ``reset_context: true`` (the default) clears it, otherwise
  the messages are appended to what the server holds, which includes its own replies.

The provider therefore tracks what the server holds and sends only the new user turn
while the history matches. When it diverges — a reply cut by a barge-in (the history
keeps only the heard part), a failed or cancelled request, an edited history — it resets
the context and replays the conversation: user turns as they were, the agent's (heard)
replies as short user notes (the server accepts no assistant messages). A cancelled
request must be followed by a reset: the server keeps its stop flag until then.

:class:`LiquidAudioServer` downloads the pinned GGUF set and the runner for this platform
(SHA-256 verified) and runs the server as a managed subprocess (``serve=True``).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import os
import platform
import socket
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import httpx
import numpy as np

from ..audio.frame import AudioFrame
from ..audio.resample import resample
from ..chat import AudioContent, ChatContext, ChatMessage
from ..errors import ConfigurationError, ProviderConnectionError, ProviderError
from ..llm import LLM, ChatChunk, CompletionUsage, LLMCapabilities, LLMStream, ToolChoice
from ..registry import register_provider
from ..tools import FunctionTool
from ..utils.download import cache_dir, download, download_archive
from ..utils.log import logger
from .openai._format import audio_content_part
from .openai._http import first_env, http_error, iter_sse, new_http_client, transport_error

__all__ = [
    "INTERLEAVED_PROMPT",
    "LiquidAudioLLM",
    "LiquidAudioModel",
    "LiquidAudioServer",
    "download_model",
    "download_runner",
    "runner_supported",
]

PROVIDER = "liquid-audio"
INTERLEAVED_PROMPT = "Respond with interleaved text and audio."
"""The system prompt that selects the conversational (interleaved) mode."""
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
OUTPUT_SAMPLE_RATE = 24_000
INPUT_SAMPLE_RATE = 16_000
SERVER_CONTEXT_SIZE = 4096
"""The server's default context length (``-c``)."""
MANAGED_CONTEXT_SIZE = 16_384
"""Context length of a managed server (the model was trained with 128k): a spoken reply
takes ~13 positions per second of audio, and the server exits when its context is full."""

HF_REPO = "LiquidAI/LFM2.5-Audio-1.5B-GGUF"
HF_REVISION = "7d525f883a077e20afb782f2ff618edcae0e39e4"
"""Pinned revision of the GGUF repository (2026-03-30)."""

_ROLES = {"model": "", "mmproj": "mmproj-", "vocoder": "vocoder-", "tokenizer": "tokenizer-"}
"""The four files the server needs (``-m``, ``-mm``, ``-mv``, ``--tts-speaker-file``) and
their file name prefixes."""
_MODEL_SHA256: dict[tuple[str, str], str] = {
    ("Q4_0", "model"): "3583bee853be20331ca342b0593fefd8acc43fb61a41ec6f1a1dc7465823e0d8",
    ("Q4_0", "mmproj"): "6b483682c263b100f8cc8022d61507e446b1d320b9febc328e7960f72d03f7ea",
    ("Q4_0", "vocoder"): "423cfcb054f41b69a5706226c243abc96d2531c3aff1121f7a2ed17149b79c95",
    ("Q4_0", "tokenizer"): "01ec6afe4578bb1e02a4d43d87e7e5827d6b3d94d2d36912ee931b9c3050f1c1",
    ("Q8_0", "model"): "fce1362831bad25de69a9a83d5016adf64e9a83bb5f9f97c092256ba8236f7d0",
    ("Q8_0", "mmproj"): "8f943619aa95cfb6218347ffd97edc4c485e76b6a839c899a4d9617db1939c67",
    ("Q8_0", "vocoder"): "3063b8ebfe08e231353506a23e765a4016db31fbb9cdd2881f3283a390313f23",
    ("Q8_0", "tokenizer"): "74f7f648699dcd819bb1eca8e80dfeed5a83684395c443d876b25bb3bcac15e3",
    ("F16", "model"): "60c8b3c36e52ee75deeccf036297f4a0f2bd156375fe866f1552cca8711ac5fe",
    ("F16", "mmproj"): "71330d7820768417d950f2dce42227896c7f6146917453957a63ba765decf621",
    ("F16", "vocoder"): "203ce137dafd19132aa4f536faaf44494ea93274d0e665a160aa044f76692440",
    ("F16", "tokenizer"): "465a08ae25120a40399156658e50f9d6f6685d62e8a49ef5eb6c178b09a8888e",
}
QUANTS = ("Q4_0", "Q8_0", "F16")

# (sys.platform, machine) -> runner zip in the repository's runners/ folder, and its sha256
_RUNNERS: dict[tuple[str, str], tuple[str, str]] = {
    ("linux", "x86_64"): (
        "llama-liquid-audio-ubuntu-x64.zip",
        "e587627538182b23b7aa3a37fdeea7a858b4edf38aeaa89637cc87300d209d3e",
    ),
    ("linux", "aarch64"): (
        "llama-liquid-audio-ubuntu-arm64.zip",
        "c1d6e5cdb67dcbb3881b274b17c002aac6ffc287e481dc483eb8ba33952f0736",
    ),
    ("darwin", "aarch64"): (
        "llama-liquid-audio-macos-arm64.zip",
        "5c2767d184eedd16007746bcfa43ef8d1574e10197138b131c879cb003dd5393",
    ),
}
_MACHINE_ALIASES = {"amd64": "x86_64", "x64": "x86_64", "arm64": "aarch64"}
_SERVER_EXE = "llama-liquid-audio-server"


def _hf_url(filename: str) -> str:
    return f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/{filename}"


# ------------------------------------------------------------------------ downloads
@dataclass(frozen=True)
class LiquidAudioModel:
    """Local paths of one GGUF set (the server's ``-m``, ``-mm``, ``-mv`` and
    ``--tts-speaker-file`` arguments)."""

    model: Path
    mmproj: Path
    vocoder: Path
    tokenizer: Path

    def server_args(self) -> list[str]:
        return [
            "-m", str(self.model), "-mm", str(self.mmproj), "-mv", str(self.vocoder),
            "--tts-speaker-file", str(self.tokenizer),
        ]  # fmt: skip


def download_model(quant: str = "Q4_0") -> LiquidAudioModel:
    """Download (once) the pinned LFM2.5-Audio-1.5B GGUF set and return its paths.

    ``quant``: ``"Q4_0"`` (~1.1 GB, default), ``"Q8_0"`` (~1.8 GB) or ``"F16"`` (~3.3 GB).
    Files are verified against their SHA-256 and kept in the model cache
    (``$VAN_CACHE_DIR`` or the platform cache directory), under ``liquid-audio/``.
    Blocking: call it with ``asyncio.to_thread`` from async code.
    """
    quant = quant.upper()
    if quant not in QUANTS:
        raise ConfigurationError(
            f"{PROVIDER}: unknown quantization {quant!r}; expected one of {', '.join(QUANTS)}"
        )
    subdir = f"liquid-audio/{HF_REPO.split('/')[-1]}"
    paths: dict[str, Path] = {}
    for role, prefix in _ROLES.items():
        name = f"{prefix}LFM2.5-Audio-1.5B-{quant}.gguf"
        digest = _MODEL_SHA256[quant, role]
        paths[role] = download(
            _hf_url(name), filename=name, subdir=subdir, sha256=digest, timeout=120.0
        )
    return LiquidAudioModel(**paths)


def _runner_key() -> tuple[str, str]:
    machine = platform.machine().lower()
    return sys.platform, _MACHINE_ALIASES.get(machine, machine)


def runner_supported() -> bool:
    """Whether a prebuilt ``llama-liquid-audio-server`` exists for this platform."""
    return _runner_key() in _RUNNERS


def download_runner() -> Path:
    """Download (once) the prebuilt ``llama-liquid-audio-server`` for this platform and
    return the executable's path.

    Runners exist for Linux x86-64 and arm64 and macOS arm64; anything else (Windows,
    Intel Macs) raises :class:`ConfigurationError`: build the server from llama.cpp PR
    ggml-org/llama.cpp#18641, or run it elsewhere and pass ``base_url``.
    """
    system, machine = _runner_key()
    entry = _RUNNERS.get((system, machine))
    if entry is None:
        raise ConfigurationError(
            f"{PROVIDER}: no prebuilt llama-liquid-audio-server for {system}/{machine} (runners "
            "exist for Linux x86-64/arm64 and macOS arm64). Build it from "
            "https://github.com/ggml-org/llama.cpp/pull/18641 and pass executable=..., or run "
            "it on another machine (or in WSL) and pass base_url=..."
        )
    name, digest = entry
    root = download_archive(
        _hf_url(f"runners/{name}"), sha256=digest, subdir="liquid-audio/runners"
    )
    exe = root / _SERVER_EXE
    if not exe.is_file():
        raise ProviderError(f"{PROVIDER}: {_SERVER_EXE} not found in {root}", provider=PROVIDER)
    return exe


# --------------------------------------------------------------------------- server
def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


class LiquidAudioServer:
    """``llama-liquid-audio-server`` as a managed subprocess.

    ``start()`` downloads the model and the runner if needed, starts the server and waits
    until it accepts requests (loading takes a few seconds); ``stop()`` terminates it.
    Also an async context manager::

        async with LiquidAudioServer() as server:
            llm = LiquidAudioLLM(base_url=server.base_url)

    Args:
        quant: GGUF quantization (see :func:`download_model`).
        host: interface to bind (keep the default loopback address: the server has no
            authentication).
        port: port to listen on (default: a free port).
        threads: CPU threads for generation (``-t``; default: the server's choice).
        ctx_size: context length in tokens (``-c``; the server's default is 4096). Replies
            and the user's audio accumulate in the stateful context.
        args: extra command-line arguments for the server.
        executable: a server binary to use instead of the downloaded runner.
        model: a :class:`LiquidAudioModel` to use instead of the downloaded GGUF set.
        startup_timeout: seconds to wait for the model to load.
    """

    def __init__(
        self,
        *,
        quant: str = "Q4_0",
        host: str = "127.0.0.1",
        port: int | None = None,
        threads: int | None = None,
        ctx_size: int | None = None,
        args: Sequence[str] = (),
        executable: str | os.PathLike[str] | None = None,
        model: LiquidAudioModel | None = None,
        startup_timeout: float = 300.0,
    ) -> None:
        self.quant = quant
        self.host = host
        self.port = port
        self.threads = threads
        self.ctx_size = ctx_size
        self.args = list(args)
        self.executable = Path(executable) if executable is not None else None
        self.model = model
        self.startup_timeout = startup_timeout
        self._proc: subprocess.Popen[bytes] | None = None
        self._log: Any = None
        self._lock = asyncio.Lock()

    @property
    def base_url(self) -> str:
        """The OpenAI-style base URL (``http://host:port/v1``) once started."""
        if self.port is None:
            raise ProviderError(f"{PROVIDER}: the server is not started", provider=PROVIDER)
        return f"http://{self.host}:{self.port}/v1"

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def command(self, exe: Path, model: LiquidAudioModel, port: int) -> list[str]:
        cmd = [str(exe), *model.server_args(), "--host", self.host, "--port", str(port)]
        if self.threads is not None:
            cmd += ["-t", str(self.threads)]
        if self.ctx_size is not None:
            cmd += ["-c", str(self.ctx_size)]
        return cmd + self.args

    async def start(self) -> str:
        """Start the server (no-op when it is running) and return its base URL."""
        async with self._lock:
            if self.running:
                return self.base_url
            exe = self.executable
            if exe is None:
                exe = await asyncio.to_thread(download_runner)
            model = self.model
            if model is None:
                model = await asyncio.to_thread(download_model, self.quant)
            port = self.port if self.port is not None else _free_port(self.host)
            cmd = self.command(exe, model, port)
            self._log = tempfile.TemporaryFile()  # noqa: SIM115 - closed in stop()
            logger.info("%s: starting %s", PROVIDER, " ".join(cmd))
            self._proc = subprocess.Popen(  # noqa: ASYNC220 - returns at once; no pipes to drain
                cmd, stdin=subprocess.DEVNULL, stdout=self._log, stderr=subprocess.STDOUT
            )
            self.port = port
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
        while True:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                code = None if proc is None else proc.returncode
                raise ProviderError(
                    f"{PROVIDER}: the server exited during startup (code {code}):\n"
                    f"{self.log_tail()}",
                    provider=PROVIDER,
                )
            try:  # the port only opens once the model is loaded
                _, writer = await asyncio.wait_for(
                    asyncio.open_connection(self.host, self.port), 2.0
                )
            except (OSError, TimeoutError):
                if loop.time() > deadline:
                    raise ProviderError(
                        f"{PROVIDER}: the server did not start within {self.startup_timeout:g}s:"
                        f"\n{self.log_tail()}",
                        provider=PROVIDER,
                    ) from None
                await asyncio.sleep(0.25)
                continue
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
            return

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

    async def __aenter__(self) -> LiquidAudioServer:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()


# ---------------------------------------------------------------------- context sync
@dataclass(frozen=True)
class _Turn:
    """One message of the conversation, as the server would hold it."""

    role: str  # "user" | "assistant"
    key: tuple[Any, ...]
    content: Any  # user: str | list of parts; assistant: its text
    tokens: int = 0
    """Estimated context positions of the message as sent."""


def _audio_digest(frame: AudioFrame) -> str:
    return hashlib.blake2b(frame.data, digest_size=12).hexdigest()


def _normalize(text: str) -> str:
    return " ".join(text.split())


_MESSAGE_TOKENS = 8  # chat-template tokens around each message
_AUDIO_TOKENS_PER_SECOND = 13.0  # the audio encoder's rate (measured: ~41 tokens for 3.2 s)


def _text_tokens(text: str) -> int:
    return len(text) // 3 + 1


def _user_turn(msg: ChatMessage) -> _Turn | None:
    parts: list[dict[str, Any]] = []
    key: list[Any] = ["user", msg.id]
    tokens = _MESSAGE_TOKENS
    for c in msg.content:
        if isinstance(c, str):
            if c.strip():
                parts.append({"type": "text", "text": c})
                key.append(("text", c))
                tokens += _text_tokens(c)
        elif isinstance(c, AudioContent) and c.frame:
            tokens += round(c.frame.duration * _AUDIO_TOKENS_PER_SECOND) + 2
            frame = c.frame
            if frame.sample_rate != INPUT_SAMPLE_RATE:
                frame = resample(frame, INPUT_SAMPLE_RATE)
            parts.append(audio_content_part(frame.to_mono() if frame.channels > 1 else frame))
            key.append(("audio", _audio_digest(c.frame)))
    if not parts:
        return None
    content: Any = parts[0]["text"] if len(parts) == 1 and parts[0]["type"] == "text" else parts
    return _Turn("user", tuple(key), content, tokens)


def _turns(ctx: ChatContext) -> tuple[list[_Turn], list[str]]:
    """The conversation turns of ``ctx`` and the instructions it carries."""
    turns: list[_Turn] = []
    instructions: list[str] = []
    for item in ctx.items:
        if not isinstance(item, ChatMessage):
            continue  # tool calls: not supported by the server
        if item.role in ("system", "developer"):
            if item.text.strip():
                instructions.append(item.text)
        elif item.role == "assistant":
            text = item.text.strip()
            if text:
                turns.append(_Turn("assistant", ("assistant", _normalize(text)), text))
        else:
            turn = _user_turn(item)
            if turn is not None:
                turns.append(turn)
    return turns, instructions


@dataclass(frozen=True)
class _Plan:
    """One request: its messages, whether it resets the server's context, the keys of the
    turns the server holds once it is sent, and their estimated context positions."""

    messages: list[dict[str, Any]]
    reset: bool
    keys: list[tuple[Any, ...]]
    tokens: int


# ------------------------------------------------------------------------------- LLM
@register_provider(
    "llm",
    PROVIDER,
    description="LFM2.5-Audio (Liquid AI) omni model via llama-liquid-audio-server: "
    "audio in, speech + text out (local, CPU)",
    default_model="lfm2.5-audio-1.5b",
    models=("lfm2.5-audio-1.5b",),
    env=(),
    local=True,
)
class LiquidAudioLLM(LLM):
    """LFM2.5-Audio-1.5B, an audio-in / speech-out LLM served by ``llama-liquid-audio-server``.

    Use it without STT and TTS: it hears the user's audio (``AudioContent``) and streams
    its own speech (:attr:`~voice_agent_next.llm.ChatChunk.audio`, 24 kHz) with the text.

    Args:
        model: model name (informational: the server serves one model).
        base_url: server root including ``/v1``. Default: ``LIQUID_AUDIO_BASE_URL``, else
            ``http://127.0.0.1:8080/v1`` — or the managed server's address with ``serve``.
        serve: start (and stop) a managed :class:`LiquidAudioServer` — downloads the
            pinned Q4_0 GGUF set (~1.1 GB) and the runner on first use.
        server: a :class:`LiquidAudioServer` to use (started on warmup; stopped on
            :meth:`aclose` only when ``serve=True`` created it).
        quant: GGUF quantization for ``serve`` (``"Q4_0"``, ``"Q8_0"``, ``"F16"``).
        server_options: more :class:`LiquidAudioServer` arguments for ``serve``
            (``threads``, ``ctx_size``, ``args``, ``port``, ``executable``...).
        max_tokens: generation steps per reply (text tokens + 80 ms audio frames).
        assistant_note: how the agent's earlier replies are replayed after a context reset
            (the server accepts no assistant messages): a user message built from this
            template (``{text}`` = the reply as heard). ``None`` drops them.
        max_replay_turns: most recent messages replayed after a reset (the server's
            context is limited; older turns are forgotten).
        context_size: the server's context length in tokens (its ``-c``; default: the
            managed server's ``ctx_size``, else the server's default of 4096). The server
            *exits* when its context overflows, and every reply (its audio frames
            included) stays in it, so the provider resets the context and replays the
            recent turns before a request could overflow it.
        timeout: HTTP timeout (connect and between streamed chunks).
    """

    provider = PROVIDER
    BASE_URL_ENV: ClassVar[tuple[str, ...]] = ("LIQUID_AUDIO_BASE_URL",)

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        serve: bool = False,
        server: LiquidAudioServer | None = None,
        quant: str = "Q4_0",
        server_options: Mapping[str, Any] | None = None,
        max_tokens: int | None = 1024,
        temperature: float | None = None,
        assistant_note: str | None = "(Earlier you said: {text})",
        max_replay_turns: int | None = 8,
        context_size: int | None = None,
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            model=model or "lfm2.5-audio-1.5b",
            capabilities=LLMCapabilities(
                tool_calling=False,
                parallel_tool_calls=False,
                audio_input=True,
                audio_output=True,
                audio_sample_rate=OUTPUT_SAMPLE_RATE,
            ),
            temperature=temperature,  # (the server samples greedily)
            max_tokens=max_tokens,
        )
        if serve and server is None:
            options = {"ctx_size": MANAGED_CONTEXT_SIZE, **dict(server_options or {})}
            server = LiquidAudioServer(quant=quant, **options)
        self.server = server
        self._owns_server = serve
        self._base_url = (base_url or first_env(self.BASE_URL_ENV) or "").rstrip("/") or None
        if self._base_url is None and server is None:
            self._base_url = DEFAULT_BASE_URL
        self.assistant_note = assistant_note
        self.max_replay_turns = max_replay_turns
        if context_size is None:
            server_ctx = server.ctx_size if server is not None else None
            context_size = server_ctx or SERVER_CONTEXT_SIZE
        self.context_size = context_size
        self.timeout = timeout
        self._client = http_client
        self._owns_client = http_client is None
        self._lock = asyncio.Lock()
        """The server generates one reply at a time and its context is shared: requests
        are serialized."""
        self._held: list[tuple[Any, ...]] | None = None
        """Keys of the turns the server's context holds (``None``: unknown -> reset)."""
        self._held_tokens = 0
        """Estimated context positions in use on the server."""
        self._warned_instructions = False
        self.resets = 0
        """Context resets so far (the first request always resets)."""

    @property
    def base_url(self) -> str:
        if self._base_url is not None:
            return self._base_url
        assert self.server is not None
        return self.server.base_url

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = new_http_client(
                self.base_url, timeout=self.timeout, connect_timeout=10.0, keepalive_expiry=4.0
            )
        return self._client

    @property
    def managed(self) -> bool:
        """The server runs as a subprocess of this provider (``serve`` / ``server``)."""
        return self.server is not None and self._base_url is None

    async def _ensure_server(self) -> None:
        if self.managed and self.server is not None and not self.server.running:
            self._held = None  # a (re)started server has an empty context
            await self.server.start()

    def _chat(
        self,
        ctx: ChatContext,
        *,
        tools: list[FunctionTool],
        tool_choice: ToolChoice | None,
        temperature: float | None,
        max_tokens: int | None,
        extra: dict[str, Any],
    ) -> LLMStream:
        if tools:
            logger.debug("%s: tools are not supported by the model; ignoring them", PROVIDER)
        return _LiquidAudioStream(
            self,
            ctx,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
        )

    # ------------------------------------------------------------ context tracking
    def plan(self, ctx: ChatContext) -> _Plan:
        """The request for ``ctx``.

        Only the new user turns are sent while the server's context is a prefix of the
        history and has room for them and the reply; otherwise the context is reset and
        the most recent turns are replayed (as many as fit).
        """
        turns, instructions = _turns(ctx)
        if instructions and not self._warned_instructions:
            self._warned_instructions = True
            logger.warning(
                "%s: the server only accepts its fixed mode prompts; the agent's "
                "instructions are not sent to the model",
                PROVIDER,
            )
        keys = [t.key for t in turns]
        held = self._held
        budget = self.context_size - (self.max_tokens or 1024) - 64
        if (
            held is not None
            and len(keys) > len(held)
            and keys[: len(held)] == held
            and all(t.role == "user" for t in turns[len(held) :])
        ):
            new = turns[len(held) :]
            tokens = self._held_tokens + sum(t.tokens for t in new)
            if tokens <= budget:
                messages = [{"role": "user", "content": t.content} for t in new]
                return _Plan(messages, False, keys, tokens)
            logger.debug("%s: the context is nearly full: resetting it", PROVIDER)
        messages = [{"role": "system", "content": INTERLEAVED_PROMPT}]
        tokens = _MESSAGE_TOKENS + _text_tokens(INTERLEAVED_PROMPT)
        replay: list[tuple[str, dict[str, Any], int]] = []  # (role, message, tokens)
        limit = len(turns) if self.max_replay_turns is None else max(1, self.max_replay_turns)
        for i, turn in enumerate(reversed(turns)):  # the most recent turns that fit
            if turn.role == "user":
                message = {"role": "user", "content": turn.content}
                cost = turn.tokens
            elif self.assistant_note:
                note = self.assistant_note.format(text=turn.content)
                message = {"role": "user", "content": note}
                cost = _MESSAGE_TOKENS + _text_tokens(note)
            else:
                continue
            if i >= limit or (replay and tokens + cost > budget):
                break
            replay.append((turn.role, message, cost))
            tokens += cost
        replay.reverse()
        while len(replay) > 1 and replay[0][0] != "user":  # not a reply to a forgotten turn
            tokens -= replay.pop(0)[2]
        messages += [message for _, message, _ in replay]
        return _Plan(messages, True, keys, tokens)

    def _sent(self, keys: list[tuple[Any, ...]] | None, tokens: int = 0) -> None:
        self._held = keys
        self._held_tokens = tokens

    # -------------------------------------------------------------------- lifecycle
    async def warmup(self) -> None:
        """Start the managed server (``serve``) and run one tiny request, so the first
        turn does not pay the cold start (~1 s on the first request)."""
        try:
            await self._ensure_server()
            async with self._lock:
                self._held = None  # whatever the warm-up leaves behind is reset next time
                body = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": INTERLEAVED_PROMPT},
                        {"role": "user", "content": "Hi"},
                    ],
                    "stream": True,
                    "max_tokens": 4,
                    "reset_context": True,
                }
                async with self._http().stream(
                    "POST", f"{self.base_url}/chat/completions", json=body
                ) as resp:
                    async for _ in resp.aiter_bytes():
                        pass
        except Exception as exc:
            logger.warning("%s warmup failed: %s", PROVIDER, exc)

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None
        if self.server is not None and self._owns_server:
            await self.server.stop()


# ---------------------------------------------------------------------------- stream
class _LiquidAudioStream(LLMStream):
    async def _run(self) -> None:
        llm: LiquidAudioLLM = self._llm  # type: ignore[assignment]
        await llm._ensure_server()
        async with llm._lock:
            try:
                plan = llm.plan(self.ctx)
                text = await self._request(llm, plan)
                if text is None and not plan.reset:  # failed before any output: start over
                    llm._held = None
                    await self._request(llm, llm.plan(self.ctx), retry=False)
            except ProviderConnectionError:
                server = llm.server
                if not llm.managed or server is None or server.running or self._first_token:
                    raise
                # the managed server died (e.g. out of memory): restart it and retry once
                logger.warning(
                    "%s: the server exited; restarting it:\n%s", PROVIDER, server.log_tail()
                )
                await llm._ensure_server()
                await self._request(llm, llm.plan(self.ctx), retry=False)

    async def _request(self, llm: LiquidAudioLLM, plan: _Plan, *, retry: bool = True) -> str | None:
        """Stream one request; returns the reply text, or ``None`` when the server failed
        before any output (and ``retry`` allows another attempt)."""
        reset = plan.reset
        body: dict[str, Any] = {
            "model": llm.model,
            "messages": plan.messages,
            "stream": True,
            "reset_context": reset,
        }
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        body.update({k: v for k, v in self.extra.items() if v is not None})
        if reset:
            llm.resets += 1
        llm._sent(None)  # until the reply is complete, the server's context is unknown
        url = f"{llm.base_url}/chat/completions"
        text: list[str] = []
        steps = 0
        finished = False
        try:
            async with llm._http().stream("POST", url, json=body) as resp:
                if resp.status_code != 200:
                    raise http_error(PROVIDER, resp.status_code, await resp.aread())
                async for event in iter_sse(resp):
                    error = event.get("error")
                    if error is not None:
                        message = error.get("message") if isinstance(error, Mapping) else error
                        if retry and not reset and steps == 0:
                            logger.info("%s: %s; resetting the context", PROVIDER, message)
                            return None
                        raise ProviderError(f"{PROVIDER}: {message}", provider=PROVIDER)
                    for delta_text, frame, reason in _parse_event(event):
                        if delta_text:
                            text.append(delta_text)
                            steps += 1
                            self._push(ChatChunk(self.request_id, delta=delta_text))
                        if frame is not None:
                            steps += 1
                            self._push(ChatChunk(self.request_id, audio=frame))
                        if reason:
                            finished = True
        except httpx.HTTPError as exc:
            raise transport_error(PROVIDER, exc, url) from exc
        reply = "".join(text)
        if finished:  # the server holds the history + its complete reply
            held = list(plan.keys)
            if reply.strip():
                held.append(("assistant", _normalize(reply)))
            llm._sent(held, plan.tokens + steps + _MESSAGE_TOKENS)
        self._push(
            ChatChunk(
                self.request_id,
                usage=CompletionUsage(completion_tokens=steps),
                finish_reason="stop" if finished else None,
            )
        )
        return reply


def _parse_event(event: Mapping[str, Any]) -> Iterator[tuple[str, AudioFrame | None, str | None]]:
    """``(text, audio, finish_reason)`` for every choice of a streamed chunk.

    ``audio_chunk`` (released runners) is base64 float32 PCM; ``audio`` (the upstream PR's
    format) is base64 int16 PCM. Both are mono at ``sample_rate`` (24 kHz).
    """
    for choice in event.get("choices") or ():
        delta = choice.get("delta") or {}
        content = delta.get("content")
        frame: AudioFrame | None = None
        for key, dtype in (("audio_chunk", "<f4"), ("audio", "<i2")):
            audio = delta.get(key)
            if isinstance(audio, Mapping) and audio.get("data"):
                rate = int(audio.get("sample_rate") or OUTPUT_SAMPLE_RATE)
                raw = base64.b64decode(audio["data"])
                usable = len(raw) - len(raw) % np.dtype(dtype).itemsize
                samples = np.frombuffer(raw[:usable], dtype=dtype)
                frame = AudioFrame.from_numpy(samples, rate) if samples.size else None
                break
        reason = choice.get("finish_reason")
        yield (content if isinstance(content, str) else ""), frame, reason or None


# ------------------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    """``python -m voice_agent_next.providers.liquid_audio download|serve``."""
    parser = argparse.ArgumentParser(
        prog="python -m voice_agent_next.providers.liquid_audio",
        description="Download and run llama-liquid-audio-server (LFM2.5-Audio-1.5B).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="download the GGUF set and the runner")
    dl.add_argument("--quant", default="Q4_0", choices=QUANTS)
    sv = sub.add_parser("serve", help="download if needed and run the server")
    sv.add_argument("--quant", default="Q4_0", choices=QUANTS)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8080)
    sv.add_argument("--threads", type=int, default=None)
    sv.add_argument("--ctx-size", type=int, default=None)
    args = parser.parse_args(argv)
    model = download_model(args.quant)
    exe = download_runner()
    if args.command == "download":
        print(f"runner: {exe}")
        for field in ("model", "mmproj", "vocoder", "tokenizer"):
            print(f"{field}: {getattr(model, field)}")
        print(f"cache: {cache_dir() / 'liquid-audio'}")
        return 0
    server = LiquidAudioServer(
        quant=args.quant, host=args.host, port=args.port, threads=args.threads,
        ctx_size=args.ctx_size,
    )  # fmt: skip
    cmd = server.command(exe, model, args.port)
    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
