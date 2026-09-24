"""Telephony media streams: Twilio, Telnyx, Vonage and Plivo over WebSocket.

The provider calls our WebSocket server for every phone call; a provider-specific
:class:`~.serializers.TelephonySerializer` converts its messages (μ-law/A-law 8 kHz or
L16 audio, marks, clear, DTMF, start/stop) to the library's transport model::

    from voice_agent_next import Agent, AgentSession
    from voice_agent_next.transports.telephony import serve_telephony

    server = await serve_telephony(
        lambda: AgentSession("openai/gpt-realtime"),
        lambda: Agent("You are a helpful phone assistant."),
        provider="twilio",
        host="0.0.0.0",
        port=8765,
    )
    await server.serve_forever()

See ``docs/transports/telephony.md`` for the provider setup (TwiML, NCCO, XML).
"""

from __future__ import annotations

from .markup import plivo_stream_xml, telnyx_stream_texml, twilio_stream_twiml, vonage_ncco
from .serializers import (
    SERIALIZERS,
    AudioCodec,
    CallInfo,
    PlivoSerializer,
    TelephonySerializer,
    TelnyxSerializer,
    TwilioSerializer,
    VonageSerializer,
    create_serializer,
)
from .transport import (
    PlivoTransport,
    TelephonyServer,
    TelephonyTransport,
    TelnyxTransport,
    TwilioTransport,
    VonageTransport,
    serve_telephony,
)

__all__ = [
    "SERIALIZERS",
    "AudioCodec",
    "CallInfo",
    "PlivoSerializer",
    "PlivoTransport",
    "TelephonySerializer",
    "TelephonyServer",
    "TelephonyTransport",
    "TelnyxSerializer",
    "TelnyxTransport",
    "TwilioSerializer",
    "TwilioTransport",
    "VonageSerializer",
    "VonageTransport",
    "create_serializer",
    "plivo_stream_xml",
    "serve_telephony",
    "telnyx_stream_texml",
    "twilio_stream_twiml",
    "vonage_ncco",
]
