"""Azure OpenAI — the OpenAI Realtime engine, STT and TTS on an Azure OpenAI resource.

* ``AgentSession("azure_openai/<deployment>")`` — :class:`AzureOpenAIRealtimeEngine`
  connects to the GA endpoint
  ``wss://<resource>.openai.azure.com/openai/v1/realtime?model=<deployment>`` (no
  ``api-version``). The model is your *deployment name* (default:
  ``AZURE_OPENAI_DEPLOYMENT_NAME``, then ``gpt-realtime-2.1``).
  See ``docs/providers/openai-realtime.md``.
* ``stt="azure_openai/<deployment>"`` — :class:`AzureOpenAISTT`: realtime transcription
  sessions on ``.../openai/v1/realtime?model=<deployment>&intent=transcription`` and
  ``.../openai/v1/audio/transcriptions`` (default deployment ``gpt-4o-mini-transcribe``).
* ``tts="azure_openai/<deployment>"`` — :class:`AzureOpenAITTS`:
  ``.../openai/v1/audio/speech`` (default deployment ``gpt-4o-mini-tts``).

Authentication: a resource key (``api_key=`` / ``AZURE_OPENAI_API_KEY``, sent as the
``api-key`` header) or a Microsoft Entra ID token (``azure_ad_token=`` /
``AZURE_OPENAI_AD_TOKEN``, sent as ``Authorization: Bearer``). Explicit arguments win over
the environment. See ``docs/providers/openai.md``.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar
from urllib.parse import urlsplit, urlunsplit

from ..errors import ConfigurationError
from ..registry import register_provider
from .openai.realtime import OpenAIRealtimeEngine
from .openai.stt import OpenAISTT
from .openai.tts import OpenAITTS

__all__ = [
    "AzureOpenAIRealtimeEngine",
    "AzureOpenAISTT",
    "AzureOpenAITTS",
    "azure_realtime_base_url",
]

DEFAULT_DEPLOYMENT = "gpt-realtime-2.1"
DEFAULT_STT_DEPLOYMENT = "gpt-4o-mini-transcribe"
DEFAULT_TTS_DEPLOYMENT = "gpt-4o-mini-tts"


def azure_realtime_base_url(endpoint: str) -> str:
    """``https://<resource>.openai.azure.com`` -> ``https://<resource>.openai.azure.com/openai/v1``.

    The scheme is converted to ``wss://`` by
    :func:`~voice_agent_next.providers.openai.realtime.realtime_url`.
    """
    parts = urlsplit(endpoint.strip())
    if not parts.netloc:
        raise ConfigurationError(f"invalid Azure OpenAI endpoint {endpoint!r}")
    path = parts.path.rstrip("/")
    if not path.endswith("/openai/v1"):
        path = path.removesuffix("/openai") + "/openai/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def _azure_connection(
    *,
    endpoint: str | None,
    base_url: str | None,
    api_key: str | None,
    azure_ad_token: str | None,
    headers: dict[str, str] | None,
) -> tuple[str, str | None, dict[str, str]]:
    """``(base_url, api_key, headers)`` for a resource: the ``/openai/v1`` root, and either
    the resource key or an Entra ID bearer token."""
    if base_url is None:
        endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ConfigurationError(
                "azure_openai needs endpoint=... or AZURE_OPENAI_ENDPOINT "
                "(https://<resource>.openai.azure.com)"
            )
        base_url = azure_realtime_base_url(endpoint)
    request_headers = dict(headers or {})
    # precedence: api_key= > azure_ad_token= > AZURE_OPENAI_API_KEY > AZURE_OPENAI_AD_TOKEN
    token = None
    if not api_key:
        token = azure_ad_token or (
            None
            if os.environ.get("AZURE_OPENAI_API_KEY")
            else os.environ.get("AZURE_OPENAI_AD_TOKEN")
        )
    if token:
        request_headers.setdefault("Authorization", f"Bearer {token}")
        api_key = ""  # authenticated by the token: never fall back to the key variable
    return base_url, api_key, request_headers


@register_provider(
    "engine",
    "azure_openai",
    description="Azure OpenAI Realtime speech-to-speech (GA /openai/v1 WebSocket)",
    models=("gpt-realtime-2.1", "gpt-realtime-2.1-mini", "gpt-realtime", "gpt-realtime-mini"),
    env=("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_AD_TOKEN"),
)
class AzureOpenAIRealtimeEngine(OpenAIRealtimeEngine):
    """OpenAI Realtime on Azure OpenAI.

    Args:
        model: deployment name (alias: ``deployment``).
        endpoint: ``https://<resource>.openai.azure.com`` (default: ``AZURE_OPENAI_ENDPOINT``).
        api_key: resource key (default: ``AZURE_OPENAI_API_KEY``).
        azure_ad_token: Microsoft Entra ID bearer token (default: ``AZURE_OPENAI_AD_TOKEN``,
            used when no resource key is configured).
        **kwargs: any :class:`~voice_agent_next.providers.openai.realtime.OpenAIRealtimeEngine`
            option.
    """

    provider = "azure_openai"

    def __init__(
        self,
        *,
        model: str | None = None,
        deployment: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        base_url: str | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        base_url, api_key, request_headers = _azure_connection(
            endpoint=endpoint,
            base_url=base_url,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            headers=headers,
        )
        super().__init__(
            model=deployment
            or model
            or os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
            or DEFAULT_DEPLOYMENT,
            api_key=api_key,
            base_url=base_url,
            headers=request_headers,
            profile="azure_openai",
            **kwargs,
        )


@register_provider(
    "stt",
    "azure_openai",
    description="Azure OpenAI transcription (realtime sessions + /audio/transcriptions)",
    default_model=DEFAULT_STT_DEPLOYMENT,
    models=("gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"),
    env=("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_AD_TOKEN"),
    requires=("websockets", "httpx"),
)
class AzureOpenAISTT(OpenAISTT):
    """OpenAI speech-to-text on Azure OpenAI.

    The model is your *deployment name*; name deployments after the model
    (``gpt-4o-mini-transcribe``...) so that model-specific options (``languages`` and
    ``keywords`` for the GPT transcribe models) are sent correctly.

    Args:
        model: deployment name (alias: ``deployment``).
        endpoint: ``https://<resource>.openai.azure.com`` (default: ``AZURE_OPENAI_ENDPOINT``).
        api_key: resource key (default: ``AZURE_OPENAI_API_KEY``).
        azure_ad_token: Microsoft Entra ID bearer token (default: ``AZURE_OPENAI_AD_TOKEN``).
        **kwargs: any :class:`~voice_agent_next.providers.openai.stt.OpenAISTT` option.
    """

    provider = "azure_openai"
    DEFAULT_MODEL: ClassVar[str] = DEFAULT_STT_DEPLOYMENT
    BASE_URL_ENV: ClassVar[tuple[str, ...]] = ()
    API_KEY_ENV: ClassVar[tuple[str, ...]] = ("AZURE_OPENAI_API_KEY",)
    GUARD_OPENAI_KEY: ClassVar[bool] = False
    AUTH_HEADER: ClassVar[str] = "api-key"
    MODEL_IN_REALTIME_URL: ClassVar[bool | None] = True
    HTTP_STREAMING: ClassVar[bool | None] = False

    def __init__(
        self,
        *,
        model: str | None = None,
        deployment: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        base_url: str | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        base_url, api_key, request_headers = _azure_connection(
            endpoint=endpoint,
            base_url=base_url,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            headers=headers,
        )
        super().__init__(
            model=deployment or model,
            api_key=api_key,
            base_url=base_url,
            headers=request_headers,
            **kwargs,
        )


@register_provider(
    "tts",
    "azure_openai",
    description="Azure OpenAI speech (gpt-4o-mini-tts, tts-1 deployments over HTTP)",
    default_model=DEFAULT_TTS_DEPLOYMENT,
    models=("gpt-4o-mini-tts", "tts-1", "tts-1-hd"),
    env=("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_AD_TOKEN"),
    requires=("httpx",),
)
class AzureOpenAITTS(OpenAITTS):
    """OpenAI text-to-speech on Azure OpenAI (the model is your *deployment name*).

    Args:
        model: deployment name (alias: ``deployment``).
        endpoint: ``https://<resource>.openai.azure.com`` (default: ``AZURE_OPENAI_ENDPOINT``).
        api_key: resource key (default: ``AZURE_OPENAI_API_KEY``).
        azure_ad_token: Microsoft Entra ID bearer token (default: ``AZURE_OPENAI_AD_TOKEN``).
        **kwargs: any :class:`~voice_agent_next.providers.openai.tts.OpenAITTS` option.
    """

    provider = "azure_openai"
    DEFAULT_MODEL: ClassVar[str] = DEFAULT_TTS_DEPLOYMENT
    DEFAULT_VOICE: ClassVar[str | None] = "alloy"
    BASE_URL_ENV: ClassVar[tuple[str, ...]] = ()
    API_KEY_ENV: ClassVar[tuple[str, ...]] = ("AZURE_OPENAI_API_KEY",)
    GUARD_OPENAI_KEY: ClassVar[bool] = False
    AUTH_HEADER: ClassVar[str] = "api-key"
    CUSTOM_VOICE_IDS: ClassVar[bool] = False

    def __init__(
        self,
        *,
        model: str | None = None,
        deployment: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        base_url: str | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        base_url, api_key, request_headers = _azure_connection(
            endpoint=endpoint,
            base_url=base_url,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            headers=headers,
        )
        super().__init__(
            model=deployment or model,
            api_key=api_key,
            base_url=base_url,
            headers=request_headers,
            **kwargs,
        )
