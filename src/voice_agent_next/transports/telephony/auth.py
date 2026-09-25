"""Authentication of telephony media streams and of the provider's webhooks.

Anyone who can reach the media-stream WebSocket can claim to be the carrier, and the
transport acts on what the start message says (it may hang up the call through the
provider's REST API with *your* credentials). Every stream therefore carries a **stream
token**: an HMAC of the call ID keyed with a secret only your webhook and the
:class:`~.transport.TelephonyServer` know. The webhook that answers the call puts it in the
markup (a Twilio/Telnyx ``<Parameter>``, a Vonage NCCO header, a Plivo ``extraHeaders``
entry — the markup helpers do this when given ``secret`` and ``call_id``); the transport
checks it before the call starts. A token only authorizes the call it was issued for.

The webhook itself should check that the request comes from the provider, and so can the
media-stream WebSocket upgrade:

* Twilio signs both with ``X-Twilio-Signature`` (:func:`validate_twilio_signature`);
* Plivo signs both with ``X-Plivo-Signature-V3`` and a nonce
  (:func:`validate_plivo_signature_v3`, the algorithm of Plivo's SDKs);
* Vonage sends ``Authorization: Bearer <JWT>``, signed with your account's signature
  secret (HS256), with signed webhooks and on the WebSocket upgrade when the NCCO endpoint
  has ``"authorization": {"type": "vonage"}`` (:func:`verify_vonage_jwt`).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import parse_qs, urlsplit, urlunsplit

from ...errors import ConfigurationError

__all__ = [
    "SECRET_ENV",
    "TELNYX_TOKEN_HEADER",
    "TOKEN_PARAMETER",
    "VONAGE_JWT_MAX_AGE",
    "plivo_signature_v3",
    "stream_secret_from_env",
    "stream_token",
    "twilio_signature",
    "validate_plivo_signature_v3",
    "validate_twilio_signature",
    "verify_stream_token",
    "verify_vonage_jwt",
    "vonage_jwt",
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


# ---------------------------------------------------------------------------- Plivo
def _plivo_url(url: str) -> str:
    """The URL Plivo signs for a ``GET`` request (and a WebSocket upgrade): scheme, host
    and path, then ``?`` and the query parameters sorted by name (and value)."""
    parts = urlsplit(url)
    base = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    params = parse_qs(parts.query, keep_blank_values=True)
    query = "&".join(
        "&".join(f"{key}={value}" for value in sorted(params[key])) for key in sorted(params)
    )
    return f"{base}?{query}" if query else base


def plivo_signature_v3(auth_token: str, url: str, nonce: str) -> str:
    """Plivo's ``X-Plivo-Signature-V3`` of a ``GET`` request or WebSocket upgrade:
    base64(HMAC-SHA256(auth token, url + "." + nonce)), ``url`` normalized as Plivo's
    SDKs do (query parameters sorted)."""
    data = f"{_plivo_url(url)}.{nonce}".encode()
    return base64.b64encode(hmac.new(auth_token.encode(), data, hashlib.sha256).digest()).decode()


def validate_plivo_signature_v3(
    auth_token: str, url: str, nonce: str | None, signature: str | None
) -> bool:
    """Check ``X-Plivo-Signature-V3`` (with ``X-Plivo-Signature-V3-Nonce``) of a ``GET``
    request or WebSocket upgrade to ``url`` (the full URL Plivo requested). The header may
    list several signatures (one per active auth token), separated by commas: one match is
    enough. Constant-time; ``False`` when anything is missing."""
    if not auth_token or not nonce or not signature:
        return False
    expected = plivo_signature_v3(auth_token, url, nonce).encode()
    matched = False
    for given in signature.split(","):
        matched |= hmac.compare_digest(expected, given.strip().encode(errors="replace"))
    return matched


# --------------------------------------------------------------------------- Vonage
VONAGE_JWT_MAX_AGE = 300.0
"""Seconds a Vonage-signed JWT is accepted after its ``iat`` (issued-at) time."""
_JWT_LEEWAY = 60.0


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64url(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def vonage_jwt(secret: str, claims: Mapping[str, Any]) -> str:
    """An HS256 JWT like the ones Vonage signs with your signature secret (for tests and
    local tools; Vonage makes the real ones)."""
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64url(json.dumps(dict(claims), separators=(",", ":")).encode())
    signing_input = f"{header}.{payload}".encode()
    signature = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64url(signature)}"


def verify_vonage_jwt(
    token: str | None,
    secret: str,
    *,
    max_age: float | None = VONAGE_JWT_MAX_AGE,
    now: float | None = None,
) -> dict[str, Any] | None:
    """The claims of a Vonage-signed JWT, or ``None`` when it is not valid.

    ``token`` is the JWT of an ``Authorization: Bearer`` header (the ``Bearer`` prefix is
    accepted). It must be HS256-signed with ``secret`` (your account's signature secret),
    issued (``iat``) at most ``max_age`` seconds ago and not expired (``exp``). The
    ``payload_hash`` claim is not checked: WebSocket upgrades and ``GET`` webhooks have no
    body.
    """
    if not token or not secret:
        return None
    token = token.strip()
    if token[:7].lower() == "bearer ":
        token = token[7:].strip()
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        header = json.loads(_unb64url(parts[0]))
        claims = json.loads(_unb64url(parts[1]))
        signature = _unb64url(parts[2])
    except (ValueError, binascii.Error):
        return None
    if not isinstance(header, dict) or header.get("alg") != "HS256" or not isinstance(claims, dict):
        return None
    expected = hmac.new(secret.encode(), f"{parts[0]}.{parts[1]}".encode(), hashlib.sha256)
    if not hmac.compare_digest(expected.digest(), signature):
        return None
    t = time.time() if now is None else now
    iat, exp = claims.get("iat"), claims.get("exp")
    if max_age is not None:
        if isinstance(iat, bool) or not isinstance(iat, int | float):
            return None
        if iat > t + _JWT_LEEWAY or t - iat > max_age:
            return None
    if exp is not None and (
        isinstance(exp, bool) or not isinstance(exp, int | float) or t > exp + _JWT_LEEWAY
    ):
        return None
    return claims
