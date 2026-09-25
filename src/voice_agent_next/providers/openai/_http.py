"""HTTP plumbing shared by the OpenAI audio providers (speech synthesis and transcription).

The audio endpoints (``/audio/speech``, ``/audio/transcriptions``) are called with the core
``httpx`` dependency instead of the ``openai`` SDK: the same code then serves OpenAI, Azure
OpenAI and the OpenAI-compatible local servers (Speaches, LocalAI, Kokoro-FastAPI), and the
providers import without the ``openai`` extra.

* :class:`APIEndpoint` resolves the base URL and the credentials (explicit arguments, then
  environment variables) and builds the authentication header (``Authorization: Bearer``
  or a raw key header such as Azure's ``api-key``);
* :func:`http_error` / :func:`transport_error` map failures to
  :mod:`voice_agent_next.errors` (401/403 -> ``AuthenticationError``, 429 ->
  ``RateLimitError``, network -> ``ProviderConnectionError``, timeouts ->
  ``ProviderTimeoutError``);
* :func:`iter_sse` parses ``text/event-stream`` bodies (``stream=true`` transcriptions).
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx

from ...errors import (
    ConfigurationError,
    MissingAPIKeyError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    for_status,
)
from ...utils.log import logger

__all__ = [
    "OPENAI_BASE_URL",
    "APIEndpoint",
    "first_env",
    "http_error",
    "is_loopback",
    "is_openai_host",
    "iter_sse",
    "new_http_client",
    "transport_error",
]

OPENAI_BASE_URL = "https://api.openai.com/v1"


def first_env(names: Iterable[str]) -> str | None:
    """The first non-empty environment variable among ``names``."""
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def is_openai_host(url: str) -> bool:
    """True for OpenAI's own API (``api.openai.com``, including the regional hosts)."""
    host = _host(url)
    return host == "api.openai.com" or host.endswith(".api.openai.com")


def is_loopback(url: str) -> bool:
    host = _host(url)
    return host in ("localhost", "::1") or host.startswith("127.")


@dataclass(frozen=True)
class APIEndpoint:
    """Where and how to reach an OpenAI-compatible audio API.

    Attributes:
        base_url: API root including the version path (``https://host/v1``), no trailing
            slash.
        api_key: the key sent with every request (``None``: no authentication header).
        auth_header: ``Authorization`` (sent as ``Bearer <key>``) or a raw key header.
        headers: extra headers sent with every request (they win over the key header).
    """

    base_url: str
    api_key: str | None = None
    auth_header: str = "Authorization"
    headers: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def resolve(
        cls,
        owner: str,
        *,
        base_url: str | None,
        api_key: str | None,
        headers: Mapping[str, str] | None = None,
        default_base_url: str,
        base_url_env: Iterable[str] = (),
        api_key_env: Iterable[str] = (),
        api_key_required: bool = True,
        auth_header: str = "Authorization",
        guard_openai_key: bool = False,
    ) -> APIEndpoint:
        """Resolve the endpoint from explicit arguments, then the environment.

        Args:
            owner: provider name used in error messages.
            base_url: explicit base URL (wins over ``base_url_env`` and the default).
            api_key: explicit key; ``""`` means "no key" (no environment fallback).
            guard_openai_key: never send the ``api_key_env`` key to an explicit
                ``base_url`` that is not OpenAI's (the key belongs to OpenAI).
            api_key_required: raise :class:`ConfigurationError` when no key (and no
                authentication header in ``headers``) is configured. For guarded OpenAI
                endpoints, a key is only required by OpenAI's own host.
        """
        resolved = (base_url or first_env(base_url_env) or default_base_url).strip().rstrip("/")
        scheme, sep, rest = resolved.partition("://")
        if sep and scheme.lower() in ("ws", "wss"):  # e.g. a variable shared with the engine
            resolved = f"{'https' if scheme.lower() == 'wss' else 'http'}://{rest}"
        if not urlsplit(resolved).netloc:
            raise ConfigurationError(f"{owner}: invalid base URL {resolved!r}")
        extra = dict(headers or {})
        auth_names = {"authorization", auth_header.lower()}
        has_auth = any(name.lower() in auth_names for name in extra)
        key: str | None
        if api_key is not None:
            key = api_key.strip() or None
        elif guard_openai_key and base_url is not None and not is_openai_host(resolved):
            key = None  # never send the OpenAI key to another server
        else:
            key = first_env(api_key_env)
        required = api_key_required and (not guard_openai_key or is_openai_host(resolved))
        if required and not key and not has_auth:
            names = " or ".join(api_key_env) or "api_key=..."
            raise MissingAPIKeyError(f"{owner} needs an API key: pass api_key=... or set {names}")
        return cls(resolved, key, auth_header, extra)

    def url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def request_headers(self) -> dict[str, str]:
        """Authentication plus the extra headers."""
        out: dict[str, str] = {}
        if self.api_key:
            if self.auth_header.lower() == "authorization":
                out["Authorization"] = f"Bearer {self.api_key}"
            else:
                out[self.auth_header] = self.api_key
        out.update(self.headers)
        return out


def new_http_client(
    base_url: str, *, timeout: float, connect_timeout: float, keepalive_expiry: float
) -> httpx.AsyncClient:
    """An ``httpx.AsyncClient`` that keeps connections alive between turns.

    httpx's default 5 s keep-alive would make most turns pay a new TCP + TLS handshake on
    the first request. Local servers are never reached through a system proxy.
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout, connect=connect_timeout),
        limits=httpx.Limits(
            max_connections=20, max_keepalive_connections=10, keepalive_expiry=keepalive_expiry
        ),
        trust_env=not is_loopback(base_url),
    )


# ----------------------------------------------------------------------------- errors
def _error_fields(body: bytes | str | None) -> tuple[str, str | None]:
    """``(message, code)`` from an OpenAI (``{"error": {...}}``) or FastAPI error body."""
    if not body:
        return "", None
    text = body if isinstance(body, str) else bytes(body).decode("utf-8", "replace")
    try:
        data = json.loads(text)
    except ValueError:
        return text.strip()[:500], None
    if not isinstance(data, Mapping):
        return text.strip()[:500], None
    err = data.get("error", data)
    if isinstance(err, Mapping):
        message = err.get("message") or err.get("detail") or err.get("error")
        code = err.get("code") or err.get("type")
        if isinstance(message, (list, dict)):  # FastAPI validation errors
            message = json.dumps(message)[:500]
        return str(message or json.dumps(err)[:500]), str(code) if code else None
    return str(err)[:500], None


def http_error(
    provider: str, status: int, body: bytes | str | None = None, *, hint: str | None = None
) -> ProviderError:
    """Map an HTTP error response to :mod:`voice_agent_next.errors`."""
    detail, code = _error_fields(body)
    message = f"{provider}: HTTP {status}" + (f": {detail}" if detail else "")
    if status == 429 and code == "insufficient_quota":
        # an exhausted quota does not recover by retrying
        return for_status(status, message, provider=provider, retryable=False)
    if status == 404 and hint:
        message = f"{message} ({hint})"
    return for_status(status, message, provider=provider)


def transport_error(provider: str, exc: httpx.HTTPError, url: str) -> ProviderError:
    """Map an ``httpx`` transport failure (no HTTP response) to a library error."""
    if isinstance(exc, httpx.ConnectTimeout):
        # a refused connection can surface as a connect timeout (Windows retries the SYN)
        return ProviderConnectionError(
            f"{provider}: timed out connecting to {url}", provider=provider
        )
    if isinstance(exc, httpx.TimeoutException):
        return ProviderTimeoutError(f"{provider}: request to {url} timed out", provider=provider)
    return ProviderConnectionError(
        f"{provider}: cannot reach {url}: {type(exc).__name__}: {exc}", provider=provider
    )


# -------------------------------------------------------------------------------- SSE
async def iter_sse(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """JSON objects from the ``data:`` fields of a server-sent event stream.

    Events are separated by blank lines; multi-line ``data`` is joined with newlines.
    Comments, ``event:``/``id:``/``retry:`` fields and non-JSON payloads are skipped, and a
    ``data: [DONE]`` sentinel (sent by some compatible servers) ends the stream.
    """
    data: list[str] = []

    def parse(payload: str) -> dict[str, Any] | None:
        try:
            obj = json.loads(payload)
        except ValueError:
            logger.debug("ignoring a non-JSON server-sent event: %.200s", payload)
            return None
        return obj if isinstance(obj, dict) else None

    async for line in response.aiter_lines():
        if not line:
            if data:
                payload, data = "\n".join(data), []
                if payload.strip() == "[DONE]":
                    return
                event = parse(payload)
                if event is not None:
                    yield event
            continue
        if line.startswith(":"):
            continue
        name, _, value = line.partition(":")
        if name == "data":
            data.append(value[1:] if value.startswith(" ") else value)
    if data:  # a stream that ended without the final blank line
        payload = "\n".join(data)
        if payload.strip() != "[DONE]":
            event = parse(payload)
            if event is not None:
                yield event
