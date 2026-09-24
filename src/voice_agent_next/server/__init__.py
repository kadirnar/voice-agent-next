"""Serve engines over the network.

* :mod:`voice_agent_next.server.realtime` — an OpenAI-Realtime-compatible WebSocket server
  (``/v1/realtime``) in front of any engine (``van serve --protocol openai-realtime``).
"""

from __future__ import annotations

from .realtime import (
    EngineFactory,
    EngineSource,
    RealtimeModel,
    RealtimeServer,
    engine_from_config,
    serve_realtime,
)

__all__ = [
    "EngineFactory",
    "EngineSource",
    "RealtimeModel",
    "RealtimeServer",
    "engine_from_config",
    "serve_realtime",
]
