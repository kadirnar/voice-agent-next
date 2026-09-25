"""Authentication of telephony media streams and of the provider's webhooks.

Anyone who can reach the media-stream WebSocket can claim to be the carrier, and the
transport acts on what the start message says (it may hang up the call through the
provider's REST API with *your* credentials). Every stream therefore carries a **stream
token**: an HMAC of the call ID keyed with a secret only your webhook and the
:class:`~.transport.TelephonyServer` know. The webhook that answers the call puts it in the
markup (a Twilio/Telnyx ``<Parameter>``, a Vonage NCCO header, a Plivo ``extraHeaders``
entry — the markup helpers do this when given ``secret`` and ``call_id``); the transport
checks it before the call starts. A token only authorizes the call it was issued for.

The webhook itself should check that the request comes from the provider:
:func:`validate_twilio_signature` implements Twilio's ``X-Twilio-Signature``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
from collections.abc import Mapping, Sequence

from ...errors import ConfigurationError

__all__ = [
    "SECRET_ENV",
    "TELNYX_TOKEN_HEADER",
    "TOKEN_PARAMETER",
    "stream_secret_from_env",
    "stream_token",
    "twilio_signature",
    "validate_twilio_signature",
    "verify_stream_token",
]

TOKEN_PARAMETER = "vanToken"
"""Name of the custom stream parameter (header) that carries the stream token (letters
only: Plivo ``extraHeaders`` allow ``[A-Za-z0-9]`` keys and values)."""

TELNYX_TOKEN_HEADER = "x-telnyx-streaming-auth-token"
"""Header of a Telnyx ``stream_auth_token``, also accepted as the stream token."""

SECRET_ENV = "VAN_TELEPHONY_SECRET"
"""Environment variable read when no ``stream_secret`` is passed."""

_CONTEXT = b"voice-agent-next/telephony-stream/v1\x00"


def stream_secret_from_env() -> str | None:
    """The stream secret from ``VAN_TELEPHONY_SECRET`` (``None`` when unset or empty)."""
    return os.environ.get(SECRET_ENV) or None


def stream_token(secret: str, call_id: str) -> str:
    """The token that authorizes the media stream of call ``call_id``.

    ``call_id`` is the provider's call identifier as the stream reports it: Twilio
    ``CallSid``, Telnyx ``call_control_id``, Plivo ``CallUUID``, and for Vonage the
    ``uuid`` header you put in the NCCO (see :func:`~.markup.vonage_ncco`).
    """
    if not secret:
        raise ConfigurationError("the telephony stream secret is empty")
    if not call_id:
        raise ConfigurationError("a stream token is bound to a call ID; got an empty one")
    # hex: the only encoding every provider carries verbatim (Plivo: alphanumeric only)
    return hmac.new(secret.encode(), _CONTEXT + call_id.encode(), hashlib.sha256).hexdigest()


def verify_stream_token(secret: str, call_id: str | None, token: object) -> bool:
    """``token`` was issued with ``secret`` for ``call_id`` (constant-time comparison)."""
    if not secret or not call_id or not isinstance(token, str) or not token:
        return False
    expected = stream_token(secret, call_id)
    return hmac.compare_digest(expected.encode(), token.encode(errors="replace"))


def twilio_signature(
    auth_token: str, url: str, params: Mapping[str, str | Sequence[str]] | None = None
) -> str:
    """Twilio's ``X-Twilio-Signature`` of a request: base64(HMAC-SHA1(auth token, data)).

    ``data`` is the full URL Twilio requested (scheme, host, path and query string exactly
    as configured in the console) followed, for ``POST`` form requests, by every parameter
    name and value sorted by name. Pass no ``params`` for ``GET`` requests and WebSocket
    upgrades.
    """
    data = url
    for key in sorted(params or {}):
        value = (params or {})[key]
        values = [value] if isinstance(value, str) else sorted(value)
        data += "".join(key + v for v in values)
    digest = hmac.new(auth_token.encode(), data.encode(), hashlib.sha1).digest()
    return base64.b64encode(digest).decode()


def validate_twilio_signature(
    auth_token: str,
    url: str,
    params: Mapping[str, str | Sequence[str]] | None,
    signature: str | None,
) -> bool:
    """Check an ``X-Twilio-Signature`` header (constant-time; ``False`` when missing).

    Use it in the webhook that returns the TwiML: behind a proxy, ``url`` must be the
    public URL Twilio called, not the one your framework sees.
    """
    if not auth_token or not signature:
        return False
    expected = twilio_signature(auth_token, url, params)
    return hmac.compare_digest(expected.encode(), signature.encode(errors="replace"))
