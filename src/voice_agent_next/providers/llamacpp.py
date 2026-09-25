"""llama.cpp ``llama-server``: local GGUF models through its OpenAI-compatible API.

Start the server with ``--jinja`` for tool calling, e.g.
``llama-server -hf ggml-org/gpt-oss-20b-GGUF --jinja``. ``llm="llamacpp"`` uses the
model the server has loaded (``GET /v1/models``). Server address: ``base_url=``, else
``LLAMACPP_BASE_URL``, else ``http://127.0.0.1:8080/v1``. When the server runs with
``--api-key``, pass ``api_key=`` or set ``LLAMA_API_KEY`` (the server's own variable).

Audio input (half-cascade): a server started with an audio model and its projector
(libmtmd: ``llama-server -hf ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF``, Voxtral Mini,
Qwen2.5-Omni, Gemma 4 E2B/E4B, LFM2.5-Audio...) takes ``input_audio`` parts in WAV or
MP3. Known model ids turn ``audio_input`` on; with a discovered model pass
``audio_input=True``. :meth:`LlamaCppLLM.warmup` reads the server's ``/props`` and warns
when the declared ``audio_input`` does not match what the loaded model supports.
"""

from __future__ import annotations

from typing import ClassVar

from ..registry import register_provider
from ..utils.log import logger
from .openai._format import AudioInputFormat
from .openai.llm import OpenAICompatibleLLM

__all__ = ["LlamaCppLLM"]


@register_provider(
    "llm",
    "llamacpp",
    description="llama.cpp llama-server (OpenAI-compatible API; audio models via mtmd)",
    extra="openai",
    requires=("openai",),
    local=True,
)
class LlamaCppLLM(OpenAICompatibleLLM):
    """llama.cpp ``llama-server`` (default port 8080)."""

    provider = "llamacpp"
    SYSTEM_MESSAGE_POLICY = "merge"  # Jinja chat templates: one leading system message
    DEFAULT_MODEL = None
    DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
    BASE_URL_ENV = ("LLAMACPP_BASE_URL",)
    API_KEY_ENV = ("LLAMA_API_KEY",)
    API_KEY_REQUIRED = False
    # mtmd decodes WAV/MP3 and resamples to the audio encoder's rate (16 kHz for the
    # supported models): sending 16 kHz skips that and keeps requests small
    AUDIO_INPUT_FORMAT: ClassVar[AudioInputFormat] = AudioInputFormat("wav", 16_000)

    async def warmup(self) -> None:
        await super().warmup()
        await self.check_audio_support()

    async def check_audio_support(self) -> bool | None:
        """Whether the loaded model accepts audio (``modalities`` in ``GET /props``;
        ``None`` if the server does not say). Logs a mismatch with ``audio_input``."""
        root = self.base_url.removesuffix("/v1")
        try:
            props = await self._client.get(f"{root}/props", cast_to=object)
        except Exception as exc:  # older servers, proxies: nothing to compare
            logger.debug("llamacpp: GET /props failed: %s", exc)
            return None
        modalities = props.get("modalities") if isinstance(props, dict) else None
        if not isinstance(modalities, dict) or "audio" not in modalities:
            return None
        served = bool(modalities["audio"])
        if self.capabilities.audio_input and not served:
            logger.warning(
                "llamacpp: audio_input is on, but the model at %s does not accept audio "
                "(start llama-server with an audio model and its --mmproj)",
                self.base_url,
            )
        elif served and not self.capabilities.audio_input:
            logger.info(
                "llamacpp: the model at %s accepts audio; pass audio_input=True to send "
                "the user's audio to it (half-cascade, no STT)",
                self.base_url,
            )
        return served
