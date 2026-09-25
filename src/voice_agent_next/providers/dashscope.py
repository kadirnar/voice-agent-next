"""Alibaba Model Studio (DashScope) OpenAI-compatible mode: Qwen-Omni with audio input.

``llm="dashscope/qwen3.5-omni-flash"`` with ``stt=None`` sends the user's audio to
Qwen-Omni (``input_audio``: base64 WAV as a ``data:;base64,`` URL). The API serves these
models only streamed, which is how every request is sent. By default the model answers
in text and the cascade's TTS speaks it; pass ``voice="Tina"`` (and no TTS) to play
Qwen-Omni's own voice (``modalities: ["text", "audio"]``, pcm16 at 24 kHz). For the
Realtime WebSocket use the ``qwen_omni`` engine instead.

Credentials: ``DASHSCOPE_API_KEY``. Endpoint: ``base_url=``, else ``DASHSCOPE_BASE_URL``,
else the workspace endpoint
``https://{DASHSCOPE_WORKSPACE_ID}.{region}.maas.aliyuncs.com/compatible-mode/v1`` when a
workspace id is set, else the regional endpoint (``DASHSCOPE_REGION``: ``ap-southeast-1``
= Singapore, the default; ``cn-beijing``; ``us-east-1``).
"""

from __future__ import annotations

import os
import re
from typing import Any, ClassVar

from ..errors import ConfigurationError
from ..registry import register_provider
from .openai._format import AudioInputFormat
from .openai.llm import OpenAICompatibleLLM

__all__ = ["DashScopeLLM", "dashscope_base_url"]

_HOST_PART = re.compile(r"^[A-Za-z0-9-]+$")
_REGIONAL = {
    "ap-southeast-1": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "cn-beijing": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "us-east-1": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
}


def dashscope_base_url(workspace_id: str | None = None, region: str | None = None) -> str:
    """The compatible-mode endpoint for a workspace and/or region (see the module doc)."""
    workspace = workspace_id or os.environ.get("DASHSCOPE_WORKSPACE_ID") or None
    region = region or os.environ.get("DASHSCOPE_REGION") or "ap-southeast-1"
    if not _HOST_PART.fullmatch(region) or (workspace and not _HOST_PART.fullmatch(workspace)):
        raise ConfigurationError(f"invalid DashScope workspace/region: {workspace!r}, {region!r}")
    if workspace:
        return f"https://{workspace}.{region}.maas.aliyuncs.com/compatible-mode/v1"
    if region not in _REGIONAL:
        raise ConfigurationError(
            f"dashscope: unknown region {region!r} without a workspace id; expected one of "
            f"{', '.join(_REGIONAL)}, or set DASHSCOPE_WORKSPACE_ID"
        )
    return _REGIONAL[region]


@register_provider(
    "llm",
    "dashscope",
    description="Alibaba Model Studio (DashScope) OpenAI-compatible mode: Qwen-Omni audio in",
    default_model="qwen3.5-omni-flash",
    models=(
        "qwen3.5-omni-flash",
        "qwen3.5-omni-plus",
        "qwen3.8-omni-flash",
        "qwen3-omni-flash",
        "qwen-omni-turbo",
    ),
    env=("DASHSCOPE_API_KEY",),
    extra="openai",
    requires=("openai",),
)
class DashScopeLLM(OpenAICompatibleLLM):
    """Qwen-Omni (and any other Model Studio chat model) over the compatible-mode API.

    Args:
        workspace_id: Model Studio workspace id (default: ``DASHSCOPE_WORKSPACE_ID``).
        region: ``ap-southeast-1`` (default), ``cn-beijing`` or ``us-east-1``
            (default: ``DASHSCOPE_REGION``).
        base_url: the endpoint, instead of ``workspace_id``/``region``.
        **kwargs: any :class:`~voice_agent_next.providers.openai.llm.OpenAILLM` option
            (``voice="Tina"`` for spoken replies, ``audio_history``...).
    """

    provider = "dashscope"
    DEFAULT_MODEL = "qwen3.5-omni-flash"
    DEFAULT_BASE_URL = _REGIONAL["ap-southeast-1"]
    BASE_URL_ENV = ("DASHSCOPE_BASE_URL",)
    API_KEY_ENV = ("DASHSCOPE_API_KEY",)
    # the per-response instructions the cascade adds are merged into the system prompt
    SYSTEM_MESSAGE_POLICY = "merge"
    AUDIO_INPUT_FORMAT: ClassVar[AudioInputFormat] = AudioInputFormat(
        "wav", 16_000, data_url=True
    )
    AUDIO_OUTPUT_FORMAT: ClassVar[str] = "wav"  # the only value accepted; chunks are PCM
    NOT_FOUND_HINT = "check the model id and the region of your API key"

    def __init__(
        self,
        *,
        workspace_id: str | None = None,
        region: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        if base_url is None and not os.environ.get("DASHSCOPE_BASE_URL"):
            base_url = dashscope_base_url(workspace_id, region)
        super().__init__(base_url=base_url, **kwargs)
