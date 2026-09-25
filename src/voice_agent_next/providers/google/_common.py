"""Shared helpers for the Gemini LLM and TTS providers.

Credentials, the ``google-genai`` client factory and error mapping. Nothing here imports
``google-genai`` at module level, so the provider modules stay importable without the
``google`` extra.

Credentials are resolved like the Gemini Live engine does:

* **Gemini Developer API** (default): ``api_key=``, else ``GOOGLE_API_KEY``, else
  ``GEMINI_API_KEY``. The key travels in the ``x-goog-api-key`` header and is never logged.
* **Vertex AI** (``vertexai=True``, or ``GOOGLE_GENAI_USE_VERTEXAI=true``): ``project`` /
  ``location`` (or ``GOOGLE_CLOUD_PROJECT`` / ``GOOGLE_CLOUD_LOCATION``) with Application
  Default Credentials or explicit ``credentials``; an ``api_key`` selects Vertex AI express
  mode. The SDK applies its own precedence rules between these.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from typing import Any

from ...errors import (
    AuthenticationError,
    ConfigurationError,
    MissingAPIKeyError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    VoiceAgentError,
    for_status,
)
from ...utils.deps import require

__all__ = [
    "API_KEY_ENV",
    "PROVIDER",
    "VERTEX_ENV",
    "deep_merge",
    "env_api_key",
    "error_for_status",
    "make_genai_client",
    "map_google_error",
    "use_vertexai",
]

PROVIDER = "google"
API_KEY_ENV = ("GOOGLE_API_KEY", "GEMINI_API_KEY")
"""Environment variables holding a Gemini API key (the first one set wins)."""
VERTEX_ENV = ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE")
"""Environment variables that switch ``google-genai`` to Vertex AI."""

_TRUE = frozenset({"1", "true"})
_RETRY_INITIAL_DELAY = 0.5
_RETRY_MAX_DELAY = 4.0
_RETRY_JITTER = 0.25
_MAX_MESSAGE = 500


# ------------------------------------------------------------------------ credentials
def env_api_key() -> str | None:
    """The first non-empty key among :data:`API_KEY_ENV`."""
    for name in API_KEY_ENV:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def use_vertexai(
    vertexai: bool | None,
    *,
    project: str | None = None,
    location: str | None = None,
    credentials: Any = None,
) -> bool:
    """Whether to talk to Vertex AI.

    ``vertexai`` if given; otherwise Vertex AI when a project, location or credentials
    object is passed, or when ``GOOGLE_GENAI_USE_VERTEXAI`` (or ``..._ENTERPRISE``) is
    ``true``/``1``, like ``google-genai`` itself.
    """
    if vertexai is not None:
        return bool(vertexai)
    if project is not None or location is not None or credentials is not None:
        return True
    return any(os.environ.get(name, "").strip().lower() in _TRUE for name in VERTEX_ENV)


def missing_key_error(what: str) -> MissingAPIKeyError:
    return MissingAPIKeyError(
        f"{what} needs credentials: pass api_key=... or set GOOGLE_API_KEY (or "
        "GEMINI_API_KEY); for Vertex AI pass vertexai=True (or set "
        "GOOGLE_GENAI_USE_VERTEXAI=true) with a project and Application Default Credentials",
        provider=PROVIDER,
    )


def make_genai_client(
    *,
    what: str,
    api_key: str | None,
    vertexai: bool | None,
    project: str | None,
    location: str | None,
    credentials: Any,
    base_url: str | None,
    api_version: str | None,
    headers: Mapping[str, str] | None,
    timeout: float | None,
    attempts: int,
    keepalive_expiry: float,
    http_client: Any,
) -> Any:
    """Build a ``google.genai.Client`` for the Gemini Developer API or Vertex AI.

    Args:
        what: component name used in error messages.
        attempts: HTTP attempts per request, including the first one (retries cover
            connection errors, 408/429/5xx before the response starts).
        keepalive_expiry: seconds an idle pooled connection is kept (ignored with
            ``http_client``); httpx's default of 5 s would add a TLS handshake to most turns.
        http_client: an ``httpx.AsyncClient`` used for every request (not closed by the SDK).

    Raises:
        AuthenticationError: no API key for the Gemini Developer API, or no usable
            Application Default Credentials for Vertex AI.
        ConfigurationError: inconsistent settings (e.g. a project without Vertex AI).
    """
    genai = require("google.genai", extra="google", package="google-genai")
    types = genai.types
    kwargs: dict[str, Any] = {}
    if use_vertexai(vertexai, project=project, location=location, credentials=credentials):
        kwargs["vertexai"] = True
        # only explicit values: the SDK resolves GOOGLE_CLOUD_PROJECT/LOCATION and the API
        # key env vars itself, with its documented precedence
        for name, value in (
            ("api_key", api_key),
            ("project", project),
            ("location", location),
            ("credentials", credentials),
        ):
            if value is not None:
                kwargs[name] = value
    else:
        if project is not None or location is not None or credentials is not None:
            raise ConfigurationError(
                f"{what}: project/location/credentials are Vertex AI settings, but "
                "vertexai=False selects the Gemini Developer API"
            )
        key = api_key or env_api_key()
        if not key:
            raise missing_key_error(what)
        kwargs["vertexai"] = False
        kwargs["api_key"] = key

    options: dict[str, Any] = {
        "retry_options": types.HttpRetryOptions(
            attempts=max(1, attempts),
            initial_delay=_RETRY_INITIAL_DELAY,
            max_delay=_RETRY_MAX_DELAY,
            jitter=_RETRY_JITTER,
        )
    }
    if timeout is not None:
        options["timeout"] = max(1, round(timeout * 1000))  # milliseconds
    if base_url:
        options["base_url"] = base_url
    if api_version:
        options["api_version"] = api_version
    if headers:
        options["headers"] = dict(headers)
    if http_client is not None:
        options["httpx_async_client"] = http_client
    else:
        import httpx

        options["async_client_args"] = {
            "limits": httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=keepalive_expiry,
            )
        }
    try:
        return genai.Client(http_options=types.HttpOptions(**options), **kwargs)
    except Exception as exc:
        mapped = map_google_error(exc, what=what)
        if mapped is not None:
            raise mapped from exc
        if isinstance(exc, ValueError):
            if "API key" in str(exc):
                raise missing_key_error(what) from exc
            raise ConfigurationError(f"{what}: invalid google-genai settings: {exc}") from exc
        raise


# ---------------------------------------------------------------------------- errors
def _reasons(details: Any) -> set[str]:
    """``google.rpc.ErrorInfo`` reasons found in an error payload (any nesting)."""
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            reason = value.get("reason")
            if isinstance(reason, str):
                found.add(reason.upper())
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(details)
    return found


def error_for_status(
    status: int,
    message: str,
    *,
    status_text: str | None = None,
    details: Any = None,
    what: str = "Gemini API",
) -> ProviderError:
    """Map an HTTP/RPC status to :mod:`voice_agent_next.errors`.

    401/403 and invalid API keys (which the Gemini API reports as 400
    ``API_KEY_INVALID``) -> :class:`AuthenticationError`, 429 -> :class:`RateLimitError`,
    408/504 -> :class:`ProviderTimeoutError`, other 5xx -> :class:`ProviderConnectionError`,
    409 -> retryable :class:`ProviderError` (see :func:`voice_agent_next.errors.for_status`).
    """
    text = f"{what} error {status}"
    if status_text:
        text += f" ({status_text})"
    text += f": {message.strip()[:_MAX_MESSAGE] or 'no details'}"
    lowered = message.lower()
    key_problem = bool(_reasons(details) & {"API_KEY_INVALID", "API_KEY_EXPIRED"}) or (
        "api key" in lowered and ("not valid" in lowered or "expired" in lowered)
    )
    if key_problem:
        return AuthenticationError(text, provider=PROVIDER, status_code=status)
    if status == 404:
        text += " (check the model id)"
    return for_status(status, text, provider=PROVIDER)


def _transport_error_kind(exc: BaseException) -> str | None:
    """``"timeout"`` / ``"connection"`` for raw network errors from httpx(2) or aiohttp."""
    for name in ("httpx", "httpx2"):
        module = sys.modules.get(name)
        if module is None:
            continue
        if isinstance(exc, module.TimeoutException):
            return "timeout"
        if isinstance(exc, module.TransportError):
            return "connection"
    aiohttp = sys.modules.get("aiohttp")
    if aiohttp is not None:
        if isinstance(exc, aiohttp.ServerTimeoutError):
            return "timeout"
        if isinstance(exc, aiohttp.ClientError):
            return "connection"
    if isinstance(exc, TimeoutError):  # aiohttp's total timeout
        return "timeout"
    return None


def map_google_error(exc: BaseException, *, what: str = "Gemini API") -> VoiceAgentError | None:
    """Map a ``google-genai`` / ``google-auth`` / transport exception to library errors.

    Returns ``None`` for exceptions that are not provider failures.
    """
    if isinstance(exc, VoiceAgentError):
        return exc
    sdk_errors = sys.modules.get("google.genai.errors")
    if sdk_errors is not None:
        if isinstance(exc, sdk_errors.APIError):
            code = exc.code if isinstance(exc.code, int) else 500
            status = exc.status if isinstance(exc.status, str) else None
            message = exc.message if isinstance(exc.message, str) else str(exc)
            return error_for_status(
                code, message, status_text=status, details=exc.details, what=what
            )
        if isinstance(exc, sdk_errors.UnknownApiResponseError):
            return ProviderError(f"{what}: malformed response: {exc}", provider=PROVIDER)
    kind = _transport_error_kind(exc)
    if kind == "timeout":
        return ProviderTimeoutError(f"{what} request timed out: {exc!r}", provider=PROVIDER)
    if kind == "connection":
        return ProviderConnectionError(f"cannot reach the {what}: {exc!r}", provider=PROVIDER)
    auth_errors = sys.modules.get("google.auth.exceptions")
    if auth_errors is not None:
        if isinstance(exc, auth_errors.TransportError):
            return ProviderConnectionError(
                f"{what}: fetching Google credentials failed: {exc}", provider=PROVIDER
            )
        if isinstance(exc, auth_errors.GoogleAuthError):
            return AuthenticationError(f"{what}: Google credentials: {exc}", provider=PROVIDER)
    pydantic = sys.modules.get("pydantic")
    if pydantic is not None and isinstance(exc, pydantic.ValidationError):
        return ConfigurationError(f"{what}: invalid request options: {exc}")
    return None


# -------------------------------------------------------------------------- options
def deep_merge(base: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    """``extra`` merged into a copy of ``base`` (nested mappings are merged, not replaced)."""
    out = dict(base)
    for key, value in extra.items():
        current = out.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            out[key] = deep_merge(current, value)
        else:
            out[key] = value
    return out
