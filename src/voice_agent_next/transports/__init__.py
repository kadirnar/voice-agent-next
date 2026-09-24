"""Audio transports (local devices, network, telephony, files, in-memory)."""

from __future__ import annotations

import importlib
from typing import Any

from ..errors import ConfigurationError
from .base import Transport, TransportCapabilities
from .file import FileTransport
from .loopback import LoopbackTransport, PlayedAudio

__all__ = [
    "FileTransport",
    "LoopbackTransport",
    "PlayedAudio",
    "Transport",
    "TransportCapabilities",
    "create_transport",
]

# transport type -> "module:Class" (imported lazily so optional deps stay optional)
_TRANSPORTS: dict[str, str] = {
    "loopback": "voice_agent_next.transports.loopback:LoopbackTransport",
    "file": "voice_agent_next.transports.file:FileTransport",
    "local": "voice_agent_next.transports.local:LocalAudioTransport",
    "websocket": "voice_agent_next.transports.websocket:WebSocketServerTransport",
    "webrtc": "voice_agent_next.transports.webrtc:WebRTCTransport",
    "telephony": "voice_agent_next.transports.telephony:TelephonyTransport",
    "twilio": "voice_agent_next.transports.telephony:TwilioTransport",
    "telnyx": "voice_agent_next.transports.telephony:TelnyxTransport",
    "vonage": "voice_agent_next.transports.telephony:VonageTransport",
    "plivo": "voice_agent_next.transports.telephony:PlivoTransport",
}


def create_transport(config: dict[str, Any] | str) -> Transport:
    """Build a transport from ``{"type": "local", ...options}`` (or just the type name)."""
    opts = {"type": config} if isinstance(config, str) else dict(config)
    kind = str(opts.pop("type", "local")).lower()
    target = _TRANSPORTS.get(kind)
    if target is None:
        raise ConfigurationError(
            f"unknown transport type {kind!r}; known: {', '.join(sorted(_TRANSPORTS))}"
        )
    module_name, _, cls_name = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise ConfigurationError(f"transport {kind!r} is not implemented yet") from exc
        raise
    transport: Transport = getattr(module, cls_name)(**opts)
    return transport
