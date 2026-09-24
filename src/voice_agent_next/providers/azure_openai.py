"""Azure OpenAI Realtime — the OpenAI Realtime engine with the ``azure_openai`` profile.

``AgentSession("azure_openai/<deployment>")`` connects to the GA endpoint
``wss://<resource>.openai.azure.com/openai/v1/realtime?model=<deployment>`` (no
``api-version``). The model is your *deployment name* (default:
``AZURE_OPENAI_DEPLOYMENT_NAME``, then ``gpt-realtime-2.1``).

Authentication: a resource key (``api_key=`` / ``AZURE_OPENAI_API_KEY``, sent as the
``api-key`` header) or a Microsoft Entra ID token (``azure_ad_token=`` /
``AZURE_OPENAI_AD_TOKEN``, sent as ``Authorization: Bearer``). Explicit arguments win over
the environment. See ``docs/providers/openai-realtime.md``.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ..errors import ConfigurationError
from ..registry import register_provider
from .openai.realtime import OpenAIRealtimeEngine

__all__ = ["AzureOpenAIRealtimeEngine", "azure_realtime_base_url"]

DEFAULT_DEPLOYMENT = "gpt-realtime-2.1"


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
