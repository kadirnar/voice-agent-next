"""Gemini TTS provider (``streamGenerateContent`` with audio output through ``google-genai``).

``create("tts", "google/gemini-3.8-flash-tts")`` (alias ``"gemini/..."``) returns a
:class:`GeminiTTS`. Every synthesis is one ``generate_content_stream`` request with
``response_modalities=["AUDIO"]``; the API streams headerless 16-bit little-endian PCM
(``audio/l16;codec=pcm;rate=24000``, mono), which is forwarded chunk by chunk as it arrives.

The request text must be complete before synthesis starts (``generateContent`` has no
incremental text input), so :meth:`GeminiTTS.stream` uses the
:class:`~voice_agent_next.tts.SentenceStreamAdapter`: sentences are synthesized one by one
(the next one is prefetched while the current one plays), each with streamed audio output.

Delivery style ("cheerful and friendly", "whispering"...) is sent as
``speech_metadata.style`` next to the verbatim transcript, as Google recommends for the
Gemini 3.8 TTS models; Gemini 2.5 TTS models get it as a natural-language prefix instead.
Inline vocal tags such as ``<sigh>`` or ``<short pause>`` in the text are passed through.
"""

from __future__ import annotations

import io
import re
import wave
from collections.abc import Mapping
from typing import Any

from ...audio.frame import AudioFrame
from ...audio.resample import StreamResampler
from ...errors import ConfigurationError, ProviderError, VoiceAgentError
from ...registry import register_provider
from ...tts import TTS, ChunkedStream, TTSCapabilities
from ...utils.log import logger
from ._common import API_KEY_ENV, PROVIDER, deep_merge, make_genai_client, map_google_error

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_VOICE",
    "KNOWN_MODELS",
    "PREBUILT_VOICES",
    "SAMPLE_RATE",
    "GeminiTTS",
]

DEFAULT_MODEL = "gemini-3.8-flash-tts"
KNOWN_MODELS = (
    "gemini-3.8-flash-tts",
    "gemini-3.8-flash-lite-tts",
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
)
DEFAULT_VOICE = "Kore"
SAMPLE_RATE = 24_000
"""Output rate of the Gemini TTS models (mono, 16-bit)."""
PREBUILT_VOICES = (
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat",
)
"""The 30 prebuilt studio voices (any other name/id is passed through as ``voice``)."""

_RATE = re.compile(r"rate=(\d+)")
_CUSTOM_VOICE_PREFIXES = ("voice_", "voicekey_")


def _legacy_prompting(model: str) -> bool:
    """Gemini 2.5 TTS models take the style as part of the prompt text."""
    return model.strip().rsplit("/", 1)[-1].lower().startswith("gemini-2.")


def _voice_config(voice: str) -> dict[str, Any]:
    """Prebuilt voice names use ``prebuilt_voice_config`` (every TTS model accepts it);
    designed/replicated voice ids (``voice_...``/``voicekey_...``) go in ``voice``."""
    if voice.startswith(_CUSTOM_VOICE_PREFIXES):
        return {"voice": voice}
    name = next((v for v in PREBUILT_VOICES if v.lower() == voice.lower()), voice)
    return {"prebuilt_voice_config": {"voice_name": name}}


@register_provider(
    "tts",
    PROVIDER,
    description="Google Gemini TTS (streamGenerateContent audio: 30 voices, style prompts)",
    default_model=DEFAULT_MODEL,
    models=KNOWN_MODELS,
    env=API_KEY_ENV,
    extra="google",
    requires=("google.genai",),
    local=False,
    aliases=("gemini",),
)
class GeminiTTS(TTS):
    """Gemini text-to-speech over streaming ``generateContent``.

    Args:
        model: ``"gemini-3.8-flash-tts"`` (default), ``"gemini-3.8-flash-lite-tts"``
            (cheaper, 101 languages), or an older preview model.
        voice: a prebuilt voice (default ``"Kore"``, see :data:`PREBUILT_VOICES`) or a
            designed/replicated voice id (``voice_...`` / ``voicekey_...``).
        language: BCP-47/ISO 639-1 code sent as ``speech_config.language_code``; ``None``
            (default) lets the model detect the language from the text.
        style: delivery instruction applied to every utterance, e.g. ``"warm and calm"``,
            ``"fast-paced, excited"``. Sent as ``speech_metadata.style`` (Gemini 2.5 TTS:
            prefixed to the text).
        temperature: sampling temperature (only sent when set).
        extra_config: extra ``GenerateContentConfig`` fields (snake_case) merged into every
            request, e.g. a full ``speech_config`` with ``multi_speaker_voice_config``.
        api_key, vertexai, project, location, credentials: credentials, as for
            :class:`~voice_agent_next.providers.google.llm.GeminiLLM`.
        timeout: seconds per request phase (connect, each streamed read); ``None``
            disables it.
        max_retries: retries for connection errors and 408/429/5xx before audio starts.
        empty_retries: extra attempts when the model finishes without any audio (the TTS
            models occasionally return text or nothing); after them a retryable
            :class:`~voice_agent_next.errors.ProviderError` is raised.
        keepalive_expiry: seconds an idle HTTP connection is kept for reuse.
        base_url, api_version, headers: endpoint overrides and extra HTTP headers.
        client: a pre-built ``google.genai.Client`` (not closed by :meth:`aclose`).
        http_client: an ``httpx.AsyncClient`` for the SDK (not closed by :meth:`aclose`).
        clean_text, trim_silence: see :class:`~voice_agent_next.tts.TTS`.
    """

    provider = PROVIDER

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        voice: str | None = DEFAULT_VOICE,
        language: str | None = None,
        style: str | None = None,
        temperature: float | None = None,
        extra_config: Mapping[str, Any] | None = None,
        api_key: str | None = None,
        vertexai: bool | None = None,
        project: str | None = None,
        location: str | None = None,
        credentials: Any = None,
        timeout: float | None = 30.0,
        max_retries: int = 1,
        empty_retries: int = 1,
        keepalive_expiry: float = 120.0,
        base_url: str | None = None,
        api_version: str | None = None,
        headers: Mapping[str, str] | None = None,
        client: Any = None,
        http_client: Any = None,
        clean_text: bool = True,
        trim_silence: bool = True,
    ) -> None:
        if max_retries < 0 or empty_retries < 0:
            raise ConfigurationError("max_retries and empty_retries must be >= 0")
        super().__init__(
            model=model or DEFAULT_MODEL,
            sample_rate=SAMPLE_RATE,
            channels=1,
            capabilities=TTSCapabilities(streaming=False),
            voice=voice or DEFAULT_VOICE,
            clean_text=clean_text,
            trim_silence=trim_silence,
        )
        self.language = language
        self.style = style.strip() if style and style.strip() else None
        self.temperature = temperature
        self.extra_config: dict[str, Any] = dict(extra_config or {})
        self.empty_retries = empty_retries
        self._closed = False
        self._owns_client = client is None
        if client is not None:
            self._client: Any = client
            return
        self._client = make_genai_client(
            what="Gemini TTS",
            api_key=api_key,
            vertexai=vertexai,
            project=project,
            location=location,
            credentials=credentials,
            base_url=base_url,
            api_version=api_version,
            headers=headers,
            timeout=timeout,
            attempts=max_retries + 1,
            keepalive_expiry=keepalive_expiry,
            http_client=http_client,
        )

    @property
    def client(self) -> Any:
        """The underlying ``google.genai.Client``."""
        return self._client

    def build_request(self, text: str, *, voice: str | None = None) -> dict[str, Any]:
        """Keyword arguments for ``client.aio.models.generate_content_stream``."""
        part: dict[str, Any] = {"text": text}
        if self.style:
            if _legacy_prompting(self.model):
                part["text"] = f"Say it {self.style}: {text}"
            else:
                part["speech_metadata"] = {"style": self.style}
        speech: dict[str, Any] = {"voice_config": _voice_config(voice or self.voice or "")}
        if self.language:
            speech["language_code"] = self.language
        config: dict[str, Any] = {"response_modalities": ["AUDIO"], "speech_config": speech}
        if self.temperature is not None:
            config["temperature"] = self.temperature
        if self.extra_config:
            extra = dict(self.extra_config)
            if "speech_config" in extra:  # a full speech_config replaces the voice settings
                config["speech_config"] = extra.pop("speech_config")
            config = deep_merge(config, extra)
        return {
            "model": self.model,
            "contents": [{"role": "user", "parts": [part]}],
            "config": config,
        }

    def map_error(self, exc: BaseException) -> VoiceAgentError | None:
        """Map an SDK/transport exception to :mod:`voice_agent_next.errors` (or ``None``)."""
        return map_google_error(exc, what="Gemini TTS")

    def _synthesize(self, text: str, *, voice: str | None) -> ChunkedStream:
        return _GeminiChunkedStream(self, text, voice=voice)

    async def warmup(self) -> None:
        """Open the HTTPS connection with a free ``models.get`` (failures are logged)."""
        try:
            await self._client.aio.models.get(model=self.model)
        except Exception as exc:
            logger.warning("Gemini TTS warmup failed: %s", self.map_error(exc) or exc)

    async def aclose(self) -> None:
        if self._owns_client and not self._closed:
            self._closed = True
            await self._client.aio.aclose()
            self._client.close()


class _GeminiChunkedStream(ChunkedStream):
    """One ``generate_content_stream`` request; PCM chunks are pushed as they arrive."""

    async def _run(self) -> None:
        tts: GeminiTTS = self._tts  # type: ignore[assignment]
        if not self.text.strip():
            return
        request = tts.build_request(self.text, voice=self.voice)
        finish: str | None = None
        for attempt in range(tts.empty_retries + 1):
            got_audio, finish = await self._attempt(tts, request)
            if got_audio:
                return
            logger.warning(
                "Gemini TTS returned no audio (finish reason %s, attempt %d/%d)",
                finish,
                attempt + 1,
                tts.empty_retries + 1,
            )
        raise ProviderError(
            f"Gemini TTS returned no audio for {len(self.text)} characters "
            f"(finish reason {finish})",
            provider=PROVIDER,
            retryable=True,
        )

    async def _attempt(self, tts: GeminiTTS, request: dict[str, Any]) -> tuple[bool, str | None]:
        decoder = _AudioDecoder()
        finish: str | None = None
        try:
            stream = await tts.client.aio.models.generate_content_stream(**request)
            async for response in stream:
                for candidate in (getattr(response, "candidates", None) or [])[:1]:
                    content = getattr(candidate, "content", None)
                    for part in getattr(content, "parts", None) or ():
                        blob = getattr(part, "inline_data", None)
                        if blob is not None and blob.data:
                            for frame in decoder.decode(blob.data, blob.mime_type):
                                self._push_audio(frame.data)
                    reason = getattr(candidate, "finish_reason", None)
                    if reason is not None:
                        finish = str(getattr(reason, "value", reason)).rsplit(".", 1)[-1]
        except Exception as exc:
            mapped = tts.map_error(exc)
            if mapped is None:
                raise
            raise mapped from exc
        tail = decoder.flush()
        if tail:
            self._push_audio(tail.data)
        return decoder.got_audio, finish


class _AudioDecoder:
    """Turns ``inline_data`` blobs into 24 kHz mono s16le frames.

    Streamed responses carry headerless PCM (``audio/l16;codec=pcm;rate=24000``); unary
    ones a WAV file. Other rates are resampled, so the TTS always yields its declared rate.
    """

    def __init__(self) -> None:
        self._resampler = StreamResampler(SAMPLE_RATE, 1)
        self.got_audio = False

    def decode(self, data: bytes, mime_type: str | None) -> list[AudioFrame]:
        mime = (mime_type or "").lower()
        if mime.startswith(("audio/wav", "audio/x-wav", "audio/wave")) or data[:4] == b"RIFF":
            frame = _wav_frame(data)
        elif not mime or mime.startswith(("audio/l16", "audio/pcm")):
            match = _RATE.search(mime)
            rate = int(match.group(1)) if match else SAMPLE_RATE
            frame = AudioFrame(data[: len(data) - len(data) % 2], rate, 1)
        else:
            raise ProviderError(
                f"Gemini TTS returned unsupported audio format {mime_type!r}", provider=PROVIDER
            )
        if not frame:
            return []
        self.got_audio = True
        out = self._resampler.push(frame)
        return [out] if out else []

    def flush(self) -> AudioFrame | None:
        tail = self._resampler.flush()
        return tail if tail else None


def _wav_frame(data: bytes) -> AudioFrame:
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            if wav.getsampwidth() != 2:
                raise ProviderError(
                    f"Gemini TTS returned {8 * wav.getsampwidth()}-bit WAV", provider=PROVIDER
                )
            return AudioFrame(
                wav.readframes(wav.getnframes()), wav.getframerate(), wav.getnchannels()
            )
    except (wave.Error, EOFError) as exc:
        raise ProviderError(f"Gemini TTS returned malformed WAV: {exc}", provider=PROVIDER) from exc
