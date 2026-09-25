"""The voice webhook (answer URL) and carrier checks of a :class:`~.transport.TelephonyServer`.

* :class:`CarrierVerifier` checks that a request comes from the carrier: Twilio's
  ``X-Twilio-Signature`` and Plivo's ``X-Plivo-Signature-V3`` (both over the public URL
  the carrier requested), or Vonage's ``Authorization: Bearer <JWT>`` (HS256, your
  account's signature secret). The server applies it to every media-stream WebSocket
  upgrade and to the answer webhook.
* :func:`answer_markup` builds the markup that answers a call — TwiML, TeXML, an NCCO or
  Plivo XML — with the media-stream URL and the call's stream token; the server serves it
  on ``GET answer_path`` (``van serve`` uses ``/answer``).

The answer webhook mints stream tokens, so it is always authenticated: by the carrier
signature when the server can check it, else by an API key in the webhook URL
(``https://host/answer?key=<key>``) or its ``Authorization`` header.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import parse_qs, urlsplit

from ...errors import ConfigurationError
from .auth import (
    VONAGE_JWT_MAX_AGE,
    validate_plivo_signature_v3,
    validate_twilio_signature,
    verify_vonage_jwt,
)
from .markup import plivo_stream_xml, telnyx_stream_texml, twilio_stream_twiml, vonage_ncco
from .serializers import TelephonySerializer

__all__ = [
    "CALL_ID_PARAMETERS",
    "CarrierVerifier",
    "answer_markup",
    "call_id_from_query",
    "public_base",
    "stream_url_for",
]

CALL_ID_PARAMETERS: Final[Mapping[str, tuple[str, ...]]] = {
    "twilio": ("CallSid",),
    "telnyx": ("CallControlId", "call_control_id", "CallSid"),
    "vonage": ("uuid",),
    "plivo": ("CallUUID",),
}
"""Query parameters of the answer webhook that carry the call ID the stream reports."""

_HOST = re.compile(r"[A-Za-z0-9.\-]+(:[0-9]{1,5})?|\[[0-9A-Fa-f:.]+\](:[0-9]{1,5})?")
_VONAGE_UUID = re.compile(r"[A-Za-z0-9\-]{1,128}")


def _split_public_url(public_url: str) -> tuple[str, str]:
    """``(host[:port], path prefix)`` of a ``wss://``/``https://`` public URL."""
    parts = urlsplit(public_url.strip())
    if parts.scheme not in ("wss", "https") or not parts.netloc:
        raise ConfigurationError(
            f"public_url must be the public wss:// (or https://) URL of the server: {public_url!r}"
        )
    if parts.query or parts.fragment:
        raise ConfigurationError(f"public_url must not have a query or fragment: {public_url!r}")
    return parts.netloc, parts.path.rstrip("/")


def public_base(public_url: str, scheme: str = "https") -> str:
    """``public_url`` (``wss://`` or ``https://``) with ``scheme``, without a trailing ``/``."""
    netloc, prefix = _split_public_url(public_url)
    return f"{scheme}://{netloc}{prefix}"


@dataclass(frozen=True)
class CarrierVerifier:
    """Checks that a request (WebSocket upgrade or webhook) comes from the carrier.

    * ``twilio`` / ``plivo``: the signature is computed over the URL the carrier
      requested, so the server needs its ``public_url`` (``wss://agent.example.com``, as
      written in the markup, without the path the request carries) and the account's
      auth token;
    * ``vonage``: ``signature_secret`` (your account's signature secret) verifies the
      ``Authorization: Bearer`` JWT; ``max_age`` bounds its ``iat``.
    """

    provider: str
    public_url: str | None = None
    auth_token: str | None = None
    signature_secret: str | None = None
    max_age: float | None = VONAGE_JWT_MAX_AGE

    @classmethod
    def create(
        cls,
        serializer: TelephonySerializer,
        *,
        public_url: str | None,
        signature_secret: str | None,
        max_age: float | None = VONAGE_JWT_MAX_AGE,
    ) -> CarrierVerifier | None:
        """The verifier of a server, ``None`` when nothing is checked. Raises when the
        configuration cannot be checked (e.g. ``public_url`` for Twilio without the auth
        token)."""
        provider = serializer.provider
        if public_url is not None:
            _split_public_url(public_url)
        if signature_secret is not None and provider != "vonage":
            raise ConfigurationError("signature_secret verifies Vonage's JWT: provider='vonage'")
        if provider == "vonage":
            if not signature_secret:
                return None
            return cls(provider, public_url, None, signature_secret, max_age)
        if provider in ("twilio", "plivo") and public_url is not None:
            auth_token = getattr(serializer, "auth_token", None)
            if not auth_token:
                env = "TWILIO_AUTH_TOKEN" if provider == "twilio" else "PLIVO_AUTH_TOKEN"
                raise ConfigurationError(
                    f"public_url checks the {provider} request signature: it needs the "
                    f"{provider} auth_token (serializer_options or {env})"
                )
            return cls(provider, public_url, auth_token)
        return None

    def verify(self, path: str, headers: Any, *, scheme: str = "wss") -> bool:
        """Whether a request for ``path`` (with its query string) with ``headers`` (a
        ``websockets`` ``Headers``) comes from the carrier. ``scheme``: ``wss`` for a
        WebSocket upgrade, ``https`` for a webhook."""
        if self.provider == "vonage":
            assert self.signature_secret is not None
            return any(
                verify_vonage_jwt(value, self.signature_secret, max_age=self.max_age) is not None
                for value in headers.get_all("Authorization")
            )
        assert self.public_url is not None
        assert self.auth_token is not None
        netloc, prefix = _split_public_url(self.public_url)
        base = f"{scheme}://{netloc}{prefix}"
        bare, _, query = path.partition("?")
        suffix = f"?{query}" if query else ""
        # a WebSocket handshake may be signed with or without a trailing "/" (Twilio's docs)
        urls = {base + bare + suffix, base + bare.rstrip("/") + suffix}
        urls.add(base + bare.rstrip("/") + "/" + suffix)
        if self.provider == "twilio":
            signature = headers.get("X-Twilio-Signature")
            return any(validate_twilio_signature(self.auth_token, u, None, signature) for u in urls)
        nonce = headers.get("X-Plivo-Signature-V3-Nonce")
        signature = headers.get("X-Plivo-Signature-V3") or headers.get("X-Plivo-Signature-Ma-V3")
        return any(validate_plivo_signature_v3(self.auth_token, u, nonce, signature) for u in urls)


def call_id_from_query(provider: str, serializer: type[TelephonySerializer], query: str) -> str:
    """The call ID of an answer webhook request (its query string); raises ``ValueError``
    when it is missing or not in the carrier's format."""
    params = parse_qs(query, keep_blank_values=False)
    for name in CALL_ID_PARAMETERS.get(provider, ()):
        values = params.get(name)
        if not values:
            continue
        call_id = values[0]
        if provider == "vonage":
            if _VONAGE_UUID.fullmatch(call_id):
                return call_id
        elif serializer.valid_call_id(call_id):
            return call_id
        raise ValueError(f"invalid {name}")
    names = " or ".join(CALL_ID_PARAMETERS.get(provider, ("a call ID",)))
    raise ValueError(f"the request has no {names} (configure the webhook to use GET)")


def stream_url_for(public_url: str | None, host: str | None, stream_path: str = "/") -> str:
    """The ``wss://`` media-stream URL to put in the markup: from ``public_url`` when
    given, else from the request's ``Host`` header (a TLS proxy or tunnel in front)."""
    if public_url is not None:
        netloc, prefix = _split_public_url(public_url)
    else:
        if not host or not _HOST.fullmatch(host.strip()):
            raise ValueError("the request has no valid Host header; set public_url")
        netloc, prefix = host.strip(), ""
    return f"wss://{netloc}{prefix}{stream_path}"


def answer_markup(
    serializer: TelephonySerializer,
    stream_url: str,
    *,
    secret: str | None,
    call_id: str | None,
    vonage_authorization: bool = False,
) -> tuple[str, str]:
    """``(content type, body)`` that connects the call to ``stream_url`` with the call's
    stream token (when ``secret`` is given). The audio format follows ``serializer``."""
    provider = serializer.provider
    if provider == "twilio":
        return "text/xml", twilio_stream_twiml(stream_url, secret=secret, call_id=call_id)
    if provider == "telnyx":
        encoding = str(getattr(serializer, "outbound_encoding", None) or "PCMU").upper()
        rate = getattr(serializer, "outbound_sample_rate", None) or (
            16_000 if encoding == "L16" else 8_000
        )
        texml = telnyx_stream_texml(
            stream_url, codec=encoding, sample_rate=rate, secret=secret, call_id=call_id
        )
        return "text/xml", texml
    if provider == "vonage":
        ncco = vonage_ncco(
            stream_url,
            sample_rate=serializer.input_codec.sample_rate,
            secret=secret,
            call_id=call_id,
            authorization={"type": "vonage"} if vonage_authorization else None,
        )
        return "application/json", json.dumps(ncco)
    if provider == "plivo":
        return "text/xml", plivo_stream_xml(stream_url, secret=secret, call_id=call_id)
    raise ConfigurationError(f"no answer markup for provider {provider!r}")
