"""Alibaba Qwen-Omni-Realtime (Model Studio / DashScope) — OpenAI Realtime engine profile.

``AgentSession("qwen_omni/qwen3.8-omni-flash-realtime")`` connects to the workspace
endpoint ``wss://{WorkspaceId}.{region}.maas.aliyuncs.com/api-ws/v1/realtime``
(``DASHSCOPE_API_KEY``, ``DASHSCOPE_WORKSPACE_ID``, optional ``DASHSCOPE_REGION``, default
``ap-southeast-1``). The server speaks the beta-era dialect: flat ``session.update``
fields, ``response.audio.delta`` event names, 16 kHz input / 24 kHz output, Chat-style
tool definitions, no ``conversation.item.truncate``, no user text items and no
per-response instructions. See ``docs/providers/openai-realtime.md``.
"""

from __future__ import annotations

import os
import re
from typing import Any

from ..errors import ConfigurationError
from ..registry import register_provider
from .openai.realtime import OpenAIRealtimeEngine

__all__ = ["QwenOmniRealtimeEngine"]

_HOST_PART = re.compile(r"^[A-Za-z0-9-]+$")


@register_provider(
    "engine",
    "qwen_omni",
    description="Alibaba Qwen-Omni-Realtime speech-to-speech (DashScope, Realtime-style)",
    default_model="qwen3.8-omni-flash-realtime",
    models=(
        "qwen3.8-omni-flash-realtime",
        "qwen3.5-omni-plus-realtime",
        "qwen3.5-omni-flash-realtime",
        "qwen3-omni-flash-realtime",
    ),
    env=("DASHSCOPE_API_KEY",),
)
class QwenOmniRealtimeEngine(OpenAIRealtimeEngine):
    """Qwen-Omni-Realtime over WebSocket.

    Args:
        workspace_id: Model Studio workspace id (default: ``DASHSCOPE_WORKSPACE_ID``).
        region: ``ap-southeast-1`` (Singapore) or ``cn-beijing`` (default:
            ``DASHSCOPE_REGION`` or Singapore).
        base_url: full base URL instead of ``workspace_id``/``region``.
        **kwargs: any :class:`~voice_agent_next.providers.openai.realtime.OpenAIRealtimeEngine`
            option (``voice="Tina"``, ``turn_detection="semantic_vad"``, ``temperature``...).
    """

    provider = "qwen_omni"

    def __init__(
        self,
        *,
        model: str | None = None,
        workspace_id: str | None = None,
        region: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        if base_url is None:
            workspace = workspace_id or os.environ.get("DASHSCOPE_WORKSPACE_ID")
            region = region or os.environ.get("DASHSCOPE_REGION") or "ap-southeast-1"
            if not workspace:
                raise ConfigurationError(
                    "qwen_omni needs workspace_id=... or DASHSCOPE_WORKSPACE_ID "
                    "(or base_url=wss://{WorkspaceId}.{region}.maas.aliyuncs.com/api-ws/v1)"
                )
            if not _HOST_PART.fullmatch(workspace) or not _HOST_PART.fullmatch(region):
                raise ConfigurationError(
                    f"invalid DashScope workspace/region: {workspace!r}, {region!r}"
                )
            base_url = f"wss://{workspace}.{region}.maas.aliyuncs.com/api-ws/v1"
        super().__init__(model=model, base_url=base_url, profile="qwen_omni", **kwargs)
