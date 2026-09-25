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
        stream_secret=os.environ["VAN_TELEPHONY_SECRET"],  # the webhook uses it too
    )
    await server.serve_forever()

Every stream must carry the stream token of its call (:func:`stream_token`): the markup
helpers add it when given the secret and the call ID, and the server can serve the markup
itself (``answer_path``). Carrier signatures (Twilio, Plivo, Vonage) can be checked on the
WebSocket upgrade too (:class:`CarrierVerifier`). See ``docs/transports/telephony.md``
for the provider setup (TwiML, NCCO, XML).
"""

from __future__ import annotations

from .auth import (
    SECRET_ENV,
    TOKEN_PARAMETER,
    plivo_signature_v3,
    stream_token,
    twilio_signature,
    validate_plivo_signature_v3,
    validate_twilio_signature,
    verify_stream_token,
    verify_vonage_jwt,
)
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
from .webhook import CarrierVerifier, answer_markup

__all__ = [
    "SECRET_ENV",
    "SERIALIZERS",
    "TOKEN_PARAMETER",
    "AudioCodec",
    "CallInfo",
    "CarrierVerifier",
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
    "answer_markup",
    "create_serializer",
    "plivo_signature_v3",
    "plivo_stream_xml",
    "serve_telephony",
    "stream_token",
    "telnyx_stream_texml",
    "twilio_signature",
    "twilio_stream_twiml",
    "validate_plivo_signature_v3",
    "validate_twilio_signature",
    "verify_stream_token",
    "verify_vonage_jwt",
    "vonage_ncco",
]
