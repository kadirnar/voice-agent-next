"""OpenAI text-to-speech: ``gpt-4o-mini-tts`` (and ``tts-1`` / ``tts-1-hd``) over HTTP.

``tts="openai/gpt-4o-mini-tts"`` posts every text to ``/audio/speech`` with
``response_format="pcm"`` and streams the chunked response body (24 kHz s16le mono) as it
arrives. ``instructions`` steers the voice (tone, accent, pacing...) on the GPT-4o models.

The speech endpoint takes complete text (no incremental text input), so :meth:`TTS.stream`
uses :class:`~voice_agent_next.tts.SentenceStreamAdapter`: the cascade's sentences are
synthesized one request at a time, the next one prefetched while the current one plays,
which also gives exact text/audio alignment for truncation on barge-in.

:class:`OpenAITTS` is also the client for OpenAI-compatible speech servers; the
preconfigured hosts (``speaches``, ``localai``, ``azure_openai``, ``kokoro_fastapi``) are
subclasses that change the class attributes below. Responses in WAV (LocalAI returns WAV
whatever the requested format) are detected and unwrapped, and audio at another sample rate
is resampled, so every host yields frames at :attr:`TTS.sample_rate`.

Only core dependencies are used (``httpx``; no ``openai`` SDK). See
``docs/providers/openai.md``.
"""

from __future__ import annotations

import asyncio
import re
import struct
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import httpx

from ...audio.frame import AudioFrame
from ...audio.resample import StreamResampler
from ...errors import ConfigurationError, ProviderError
from ...registry import register_provider
from ...tts import TTS, ChunkedStream, TTSCapabilities
from ...utils.log import logger
from ._http import (
    OPENAI_BASE_URL,
    APIEndpoint,
    http_error,
    new_http_client,
    transport_error,
)

__all__ = ["OPENAI_VOICES", "OpenAICompatibleTTS", "OpenAITTS"]

OPENAI_VOICES = (
    "alloy", "ash", "ballad", "cedar", "coral", "echo", "fable",
    "marin", "nova", "onyx", "sage", "shimmer", "verse",
)  # fmt: skip
"""Built-in voices of ``gpt-4o-mini-tts`` (``tts-1``/``tts-1-hd``: alloy, ash, coral, echo,
fable, nova, onyx, sage, shimmer). Custom voices are passed by id (``voice_...``)."""

_WAV_HEADER_LIMIT = 1 << 20  # give up looking for the "data" chunk after 1 MiB
_SENTENCE_BREAK = re.compile(r"(?<=[.!?…。！？])\s+")


@register_provider(
    "tts",
    "openai",
    description="OpenAI speech (gpt-4o-mini-tts, tts-1): 24 kHz PCM streamed over HTTP",
    default_model="gpt-4o-mini-tts",
    models=("gpt-4o-mini-tts", "gpt-4o-mini-tts-2025-12-15", "tts-1", "tts-1-hd"),
    env=("OPENAI_API_KEY",),
    requires=("httpx",),
)
class OpenAITTS(TTS):
    """Speech synthesis through ``/audio/speech`` (OpenAI and compatible servers).

    Args:
        model: ``gpt-4o-mini-tts`` (default; a dated snapshot such as
            ``gpt-4o-mini-tts-2025-12-15`` works too), ``tts-1`` or ``tts-1-hd``.
        voice: built-in voice (:data:`OPENAI_VOICES`; default ``marin``, ``alloy`` for the
            ``tts-1`` models) or a custom voice id (``voice_...``).
        api_key: API key (default: ``OPENAI_API_KEY``, never sent to a non-OpenAI
            ``base_url``).
        base_url: API root including ``/v1`` (default: ``OPENAI_BASE_URL``, then
            ``https://api.openai.com/v1``).
        instructions: voice steering for the GPT-4o models ("Speak in a calm, warm tone.");
            ignored by ``tts-1`` / ``tts-1-hd``.
        speed: 0.25-4.0 (default 1.0).
        sample_rate: output rate of the frames (default 24000, the rate of OpenAI's raw PCM;
            other rates are resampled).
        response_format: ``pcm`` (default, lowest latency) or ``wav``; both are decoded.
        extra: extra JSON body fields (``None`` removes a default one).
        headers: extra HTTP headers.
        timeout: read timeout in seconds (the longest wait for the next audio chunk).
        connect_timeout: connection timeout in seconds.
        max_retries: retries of a request that failed before any audio arrived (connection
            errors, timeouts, 429 and 5xx).
        keepalive_expiry: seconds an idle connection stays open; long enough to reuse the
            TLS connection from one turn to the next.
        http_client: an ``httpx.AsyncClient`` to use (not closed by :meth:`aclose`).
    """

    provider = "openai"

    DEFAULT_MODEL: ClassVar[str] = "gpt-4o-mini-tts"
    DEFAULT_VOICE: ClassVar[str | None] = "marin"
    DEFAULT_BASE_URL: ClassVar[str] = OPENAI_BASE_URL
    BASE_URL_ENV: ClassVar[tuple[str, ...]] = ("OPENAI_BASE_URL",)
    API_KEY_ENV: ClassVar[tuple[str, ...]] = ("OPENAI_API_KEY",)
    API_KEY_REQUIRED: ClassVar[bool] = True
    GUARD_OPENAI_KEY: ClassVar[bool] = True
    """Never send the ``API_KEY_ENV`` key to an explicit ``base_url`` of another host."""
    AUTH_HEADER: ClassVar[str] = "Authorization"
    RESPONSE_FORMAT: ClassVar[str] = "pcm"
    PCM_SAMPLE_RATE: ClassVar[int] = 24_000
    """Rate of the server's raw ``pcm`` output (when it does not take ``sample_rate``)."""
    SAMPLE_RATE_PARAM: ClassVar[bool] = False
    """The server accepts a ``sample_rate`` body field (Speaches, LocalAI)."""
    CUSTOM_VOICE_IDS: ClassVar[bool] = True
    """Voices named ``voice_...`` are custom voice ids, sent as ``{"id": ...}``."""
    DEFAULT_EXTRA: ClassVar[Mapping[str, Any]] = {}
    PRELOAD_ON_WARMUP: ClassVar[bool] = False
    """:meth:`warmup` synthesizes a short text (servers that load models on demand)."""
    MAX_INPUT_CHARS: ClassVar[int] = 4096
    """Longer texts are split at sentence boundaries into several requests."""
    NOT_FOUND_HINT: ClassVar[str] = "check the model id and base_url"

    def __init__(
        self,
        *,
        model: str | None = None,
        voice: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        instructions: str | None = None,
        speed: float | None = None,
        sample_rate: int | None = None,
        response_format: str | None = None,
        extra: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        connect_timeout: float = 10.0,
        max_retries: int = 1,
        keepalive_expiry: float = 120.0,
        http_client: httpx.AsyncClient | None = None,
        clean_text: bool = True,
        trim_silence: bool = True,
    ) -> None:
        resolved_model = model or self.DEFAULT_MODEL
        if speed is not None and not 0.25 <= speed <= 4.0:
            raise ConfigurationError(f"speed must be within [0.25, 4.0], got {speed}")
        rate = self.PCM_SAMPLE_RATE if sample_rate is None else sample_rate
        if rate <= 0:
            raise ConfigurationError(f"sample_rate must be > 0, got {rate}")
        fmt = (response_format or self.RESPONSE_FORMAT).lower()
        if fmt not in ("pcm", "wav"):
            raise ConfigurationError(
                f"response_format must be 'pcm' or 'wav' (decoded to PCM), got {fmt!r}"
            )
        super().__init__(
            model=resolved_model,
            sample_rate=rate,
            channels=1,
            capabilities=TTSCapabilities(streaming=False),
            voice=voice or self._default_voice(resolved_model),
            clean_text=clean_text,
            trim_silence=trim_silence,
        )
        self.endpoint = APIEndpoint.resolve(
            type(self).__name__,
            base_url=base_url,
            api_key=api_key,
            headers=headers,
            default_base_url=self.DEFAULT_BASE_URL,
            base_url_env=self.BASE_URL_ENV,
            api_key_env=self.API_KEY_ENV,
            api_key_required=self.API_KEY_REQUIRED,
            auth_header=self.AUTH_HEADER,
            guard_openai_key=self.GUARD_OPENAI_KEY,
        )
        self.instructions = instructions
        self.speed = speed
        self.response_format = fmt
        self.extra: dict[str, Any] = {**self.DEFAULT_EXTRA, **(extra or {})}
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.max_retries = max(0, max_retries)
        self.keepalive_expiry = keepalive_expiry
        self._http = http_client
        self._owns_http = http_client is None

    @property
    def base_url(self) -> str:
        return self.endpoint.base_url

    def _default_voice(self, model: str) -> str | None:
        voice = self.DEFAULT_VOICE
        if voice in ("marin", "cedar") and model.startswith("tts-1"):
            return "alloy"  # the tts-1 models have no marin/cedar
        return voice

    # ------------------------------------------------------------------ requests
    def build_request(self, text: str, *, voice: str | None = None) -> dict[str, Any]:
        """The JSON body posted to ``/audio/speech`` for ``text``."""
        body: dict[str, Any] = {
            "model": self.model,
            "input": text,
            "response_format": self.response_format,
        }
        chosen = voice or self.voice
        if chosen:
            custom = self.CUSTOM_VOICE_IDS and chosen.startswith("voice_")
            body["voice"] = {"id": chosen} if custom else chosen
        if self.instructions and not self.model.startswith("tts-1"):
            body["instructions"] = self.instructions
        if self.speed is not None:
            body["speed"] = self.speed
        if self.SAMPLE_RATE_PARAM:
            body["sample_rate"] = self.sample_rate
        for key, value in self.extra.items():
            if value is None:
                body.pop(key, None)
            else:
                body[key] = value
        return body

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = new_http_client(
                self.base_url,
                timeout=self.timeout,
                connect_timeout=self.connect_timeout,
                keepalive_expiry=self.keepalive_expiry,
            )
            self._owns_http = True
        return self._http

    async def _speak(self, text: str, voice: str | None, push: Callable[[bytes], None]) -> None:
        """Synthesize one request's worth of text, pushing s16le at :attr:`sample_rate`."""
        body = self.build_request(text, voice=voice)
        url = self.endpoint.url("audio/speech")
        hint = self.NOT_FOUND_HINT.format(model=self.model)
        for attempt in range(self.max_retries + 1):
            decoder = _PCMDecoder(self._pcm_rate(), self.sample_rate, self.provider)
            received = False
            try:
                async with self._client().stream(
                    "POST", url, json=body, headers=self.endpoint.request_headers()
                ) as response:
                    if response.status_code >= 400:
                        raw = await response.aread()
                        raise http_error(self.provider, response.status_code, raw, hint=hint)
                    async for chunk in response.aiter_bytes():
                        pcm = decoder.feed(chunk)
                        if pcm:
                            received = True
                            push(pcm)
                tail = decoder.flush()
                if tail:
                    push(tail)
                return
            except ProviderError as exc:
                error: ProviderError = exc
            except httpx.HTTPError as exc:
                error = transport_error(self.provider, exc, url)
                error.__cause__ = exc
            if received or not error.retryable or attempt >= self.max_retries:
                raise error
            delay = min(0.25 * 2**attempt, 2.0)
            logger.warning(
                "%s TTS request failed (%s); retrying in %.2fs", self.provider, error, delay
            )
            await asyncio.sleep(delay)

    def _pcm_rate(self) -> int:
        """Rate of the raw PCM the server returns (WAV responses carry their own)."""
        return self.sample_rate if self.SAMPLE_RATE_PARAM else self.PCM_SAMPLE_RATE

    def _synthesize(self, text: str, *, voice: str | None) -> ChunkedStream:
        return _SpeechStream(self, text, voice=voice)

    # ----------------------------------------------------------------- lifecycle
    async def warmup(self) -> None:
        """Open the HTTP connection ahead of the first sentence (``GET /models``).

        Hosts that load models on demand (``PRELOAD_ON_WARMUP``) synthesize a short text
        instead, so the model is in memory for the first turn. Failures are logged.
        """
        try:
            if self.PRELOAD_ON_WARMUP:
                await self._speak("Hi.", None, lambda _pcm: None)
                return
            url = self.endpoint.url("models")
            response = await self._client().get(url, headers=self.endpoint.request_headers())
            if response.status_code in (401, 403):
                raise http_error(self.provider, response.status_code, response.content)
        except Exception as exc:
            logger.warning("%s TTS warmup failed: %s", self.provider, exc)

    async def aclose(self) -> None:
        if self._http is not None and self._owns_http:
            http, self._http = self._http, None
            await http.aclose()


class OpenAICompatibleTTS(OpenAITTS):
    """Base class for preconfigured OpenAI-compatible speech servers.

    Subclasses set ``provider`` and the class attributes of :class:`OpenAITTS`, and
    register themselves with ``@register_provider("tts", "<name>", ...)``. The API key is
    always read from ``API_KEY_ENV`` and is optional unless ``API_KEY_REQUIRED``.
    """

    provider = "openai_compatible"
    DEFAULT_VOICE: ClassVar[str | None] = None
    BASE_URL_ENV: ClassVar[tuple[str, ...]] = ()
    API_KEY_ENV: ClassVar[tuple[str, ...]] = ()
    API_KEY_REQUIRED: ClassVar[bool] = False
    GUARD_OPENAI_KEY: ClassVar[bool] = False
    CUSTOM_VOICE_IDS: ClassVar[bool] = False


class _SpeechStream(ChunkedStream):
    async def _run(self) -> None:
        tts: OpenAITTS = self._tts  # type: ignore[assignment]
        for part in split_text(self.text, tts.MAX_INPUT_CHARS):
            await tts._speak(part, self.voice, self._push_audio)


def split_text(text: str, limit: int) -> list[str]:
    """``text`` in pieces of at most ``limit`` characters, cut at sentence boundaries."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    parts: list[str] = []
    current = ""
    for sentence in _SENTENCE_BREAK.split(text):
        while len(sentence) > limit:  # one overlong sentence: cut at the last space
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if current:
                parts.append(current)
                current = ""
            parts.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if current and len(current) + 1 + len(sentence) > limit:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    return [p for p in parts if p]


class _PCMDecoder:
    """Raw s16le PCM or a (streamed) WAV file -> s16le mono at ``target_rate``.

    Servers that return WAV even when asked for PCM (LocalAI), or audio at another rate
    than requested (Piper voices are 22.05 kHz), are handled here: a ``RIFF``/``WAVE``
    header is parsed incrementally and stripped, and the audio is converted to mono at the
    target rate. Raw PCM at the target rate passes through untouched.
    """

    def __init__(self, rate: int, target_rate: int, provider: str) -> None:
        self._rate = rate
        self._channels = 1
        self._target = target_rate
        self._provider = provider
        self._state = "sniff"  # -> "header" (WAV) -> "pcm"
        self._head = bytearray()
        self._remaining: int | None = None  # bytes left in the WAV "data" chunk
        self._odd = b""
        self._resampler: StreamResampler | None = None

    def feed(self, data: bytes) -> bytes:
        if self._state == "pcm":
            return self._pcm(data)
        self._head += data
        if self._state == "sniff":
            if len(self._head) < 12:
                return b""
            if self._head[:4] == b"RIFF" and self._head[8:12] == b"WAVE":
                self._state = "header"
            else:
                self._state = "pcm"
                data, self._head = bytes(self._head), bytearray()
                return self._pcm(data)
        start = self._parse_header()
        if start is None:
            if len(self._head) > _WAV_HEADER_LIMIT:
                raise ProviderError(f"{self._provider}: no audio data in the WAV response")
            return b""
        self._state = "pcm"
        data, self._head = bytes(self._head[start:]), bytearray()
        return self._pcm(data)

    def flush(self) -> bytes:
        out = b""
        if self._state == "sniff" and self._head:  # a response shorter than a WAV header
            self._state = "pcm"
            data, self._head = bytes(self._head), bytearray()
            out = self._pcm(data)
        if self._resampler is not None:
            out += self._resampler.flush().data
        return out

    def _parse_header(self) -> int | None:
        """Offset of the PCM samples once the ``fmt `` and ``data`` chunks are in."""
        head = self._head
        pos = 12
        fmt_seen = False
        while pos + 8 <= len(head):
            chunk_id = bytes(head[pos : pos + 4])
            (size,) = struct.unpack_from("<I", head, pos + 4)
            body = pos + 8
            if chunk_id == b"data":
                if not fmt_seen:
                    raise ProviderError(f"{self._provider}: WAV response without a fmt chunk")
                # streamed WAVs carry a placeholder size (0 or 0xFFFFFFFF): read to the end
                self._remaining = size if 0 < size < 0xFFFFFFF0 else None
                return body
            if body + size > len(head):
                return None  # the chunk is not complete yet
            if chunk_id == b"fmt ":
                if size < 16:
                    raise ProviderError(f"{self._provider}: malformed WAV fmt chunk")
                tag, channels, rate = struct.unpack_from("<HHI", head, body)
                (bits,) = struct.unpack_from("<H", head, body + 14)
                if tag not in (1, 0xFFFE) or bits != 16:
                    raise ProviderError(
                        f"{self._provider}: unsupported WAV audio (format {tag}, {bits}-bit); "
                        "expected 16-bit PCM"
                    )
                self._channels, self._rate = max(1, channels), rate
                fmt_seen = True
            pos = body + size + (size & 1)  # chunks are word-aligned
        return None

    def _pcm(self, data: bytes) -> bytes:
        if self._remaining is not None:
            data = data[: self._remaining]
            self._remaining -= len(data)
        if self._rate == self._target and self._channels == 1:
            return data  # the base class keeps odd trailing bytes for the next chunk
        data = self._odd + data
        align = 2 * self._channels
        cut = len(data) - len(data) % align
        data, self._odd = data[:cut], data[cut:]
        if not data:
            return b""
        if self._resampler is None:
            self._resampler = StreamResampler(self._target, 1)
        return self._resampler.push(AudioFrame(data, self._rate, self._channels)).data
