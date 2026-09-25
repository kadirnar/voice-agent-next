"""Call-control snippets that point a provider's media stream at a :class:`TelephonyServer`.

Return them from your voice webhook (answer URL) — any HTTP framework will do. Pass the
stream ``secret`` and the ``call_id`` from the webhook request: the helper adds the stream
token (:func:`~.auth.stream_token`) that the :class:`~.transport.TelephonyServer` requires.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from xml.sax.saxutils import escape, quoteattr

from ...errors import ConfigurationError
from .auth import TOKEN_PARAMETER, stream_token

__all__ = ["plivo_stream_xml", "telnyx_stream_texml", "twilio_stream_twiml", "vonage_ncco"]

_XML = '<?xml version="1.0" encoding="UTF-8"?>'


def _with_token(
    parameters: Mapping[str, Any] | None, secret: str | None, call_id: str | None
) -> dict[str, Any]:
    out = dict(parameters or {})
    if secret is None:
        return out
    if not call_id:
        raise ConfigurationError("a stream secret needs the call_id of the webhook request")
    out[TOKEN_PARAMETER] = stream_token(secret, call_id)
    return out


def _parameters(parameters: Mapping[str, Any] | None) -> str:
    return "".join(
        f"<Parameter name={quoteattr(str(k))} value={quoteattr(str(v))}/>"
        for k, v in (parameters or {}).items()
    )


def twilio_stream_twiml(
    url: str,
    parameters: Mapping[str, Any] | None = None,
    *,
    secret: str | None = None,
    call_id: str | None = None,
) -> str:
    """TwiML that connects the call to a bidirectional Media Stream (``<Connect><Stream>``).

    ``parameters`` arrive in ``transport.call.custom_parameters``. With ``<Connect>`` the
    call stays on the stream until the WebSocket closes; TwiML after it then continues.
    ``secret`` / ``call_id`` (the webhook's ``CallSid``) add the stream token parameter.
    """
    parameters = _with_token(parameters, secret, call_id)
    return (
        f"{_XML}<Response><Connect><Stream url={quoteattr(url)}>"
        f"{_parameters(parameters)}</Stream></Connect></Response>"
    )


def telnyx_stream_texml(
    url: str,
    *,
    codec: str = "PCMU",
    sample_rate: int = 8_000,
    parameters: Mapping[str, Any] | None = None,
    pause: int = 3600,
    secret: str | None = None,
    call_id: str | None = None,
) -> str:
    """TeXML that starts a bidirectional RTP stream and keeps the call up for ``pause`` s.

    ``codec``/``sample_rate`` are the outbound (``bidirectionalCodec`` /
    ``bidirectionalSamplingRate``) format: pass the same ``outbound_encoding`` /
    ``outbound_sample_rate`` to the Telnyx serializer. ``secret`` / ``call_id`` (the
    webhook's ``CallControlId``, as the stream's ``call_control_id``) add the stream token.
    """
    parameters = _with_token(parameters, secret, call_id)
    attrs = (
        f'url={quoteattr(url)} bidirectionalMode="rtp" bidirectionalCodec={quoteattr(codec)}'
        f' bidirectionalSamplingRate="{int(sample_rate)}"'
    )
    return (
        f"{_XML}<Response><Start><Stream {attrs}>{_parameters(parameters)}</Stream></Start>"
        f'<Pause length="{int(pause)}"/></Response>'
    )


def vonage_ncco(
    uri: str,
    *,
    sample_rate: int = 16_000,
    headers: Mapping[str, Any] | None = None,
    secret: str | None = None,
    call_id: str | None = None,
    authorization: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """An NCCO that connects the call to a WebSocket endpoint (``audio/l16``).

    ``headers`` come back in the ``websocket:connected`` message
    (``transport.call.custom_parameters``). ``secret`` / ``call_id`` (the answer webhook's
    ``uuid``) add a ``uuid`` header and the stream token: Vonage does not identify the call
    in ``websocket:connected`` otherwise. ``authorization`` is the endpoint's
    ``authorization`` object: ``{"type": "vonage"}`` makes Vonage sign the WebSocket
    upgrade with a JWT (checked by a server given the signature secret).
    """
    if secret is not None and call_id:
        headers = {**(headers or {}), "uuid": call_id}
    headers = _with_token(headers, secret, call_id)
    endpoint: dict[str, Any] = {
        "type": "websocket",
        "uri": uri,
        "content-type": f"audio/l16;rate={int(sample_rate)}",
    }
    if headers:
        endpoint["headers"] = dict(headers)
    if authorization:
        endpoint["authorization"] = dict(authorization)
    return [{"action": "connect", "endpoint": [endpoint]}]


def plivo_stream_xml(
    url: str,
    *,
    content_type: str = "audio/x-mulaw;rate=8000",
    keep_call_alive: bool = True,
    extra_headers: Mapping[str, Any] | None = None,
    secret: str | None = None,
    call_id: str | None = None,
) -> str:
    """Plivo XML with a bidirectional ``<Stream>``.

    ``content_type`` is ``audio/x-mulaw;rate=8000``, ``audio/x-l16;rate=8000`` or
    ``audio/x-l16;rate=16000``. ``extra_headers`` (letters and digits only, 512 bytes in
    all) arrive in
    ``transport.call.custom_parameters``. ``secret`` / ``call_id`` (the answer URL's
    ``CallUUID``) add the stream token.
    """
    extra_headers = _with_token(extra_headers, secret, call_id)
    attrs = (
        f'bidirectional="true" keepCallAlive="{str(keep_call_alive).lower()}"'
        f" contentType={quoteattr(content_type)}"
    )
    if extra_headers:
        joined = ",".join(f"{k}={v}" for k, v in extra_headers.items())
        attrs += f" extraHeaders={quoteattr(joined)}"
    return f"{_XML}<Response><Stream {attrs}>{escape(url)}</Stream></Response>"
