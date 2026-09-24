"""Exception hierarchy for voice-agent-next.

Every exception raised deliberately by the library derives from :class:`VoiceAgentError`
so applications can catch library failures with a single ``except`` clause.
"""

from __future__ import annotations

__all__ = [
    "AuthenticationError",
    "ConfigurationError",
    "EngineError",
    "MissingDependencyError",
    "ProviderConnectionError",
    "ProviderError",
    "ProviderNotFoundError",
    "ProviderTimeoutError",
    "RateLimitError",
    "ToolError",
    "TransportError",
    "VoiceAgentError",
]


class VoiceAgentError(Exception):
    """Base class for all voice-agent-next errors."""


class ConfigurationError(VoiceAgentError, ValueError):
    """Invalid configuration (bad option values, unknown keys, missing API key...)."""


class MissingDependencyError(VoiceAgentError, ImportError):
    """An optional dependency required by a provider/feature is not installed."""


class ProviderNotFoundError(VoiceAgentError, LookupError):
    """No provider is registered under the requested name/kind."""


class ProviderError(VoiceAgentError):
    """A provider (cloud API or local model) failed.

    Attributes:
        provider: provider name, e.g. ``"deepgram"``.
        retryable: whether retrying the same request may succeed.
        status_code: HTTP/WebSocket status code, when available.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable
        self.status_code = status_code


class AuthenticationError(ProviderError):
    """Missing or invalid credentials."""


class RateLimitError(ProviderError):
    """The provider rejected the request because of rate limiting / quota."""

    def __init__(self, message: str, **kwargs: object) -> None:
        kwargs.setdefault("retryable", True)
        super().__init__(message, **kwargs)  # type: ignore[arg-type]


class ProviderConnectionError(ProviderError):
    """Network/connection level failure talking to a provider."""

    def __init__(self, message: str, **kwargs: object) -> None:
        kwargs.setdefault("retryable", True)
        super().__init__(message, **kwargs)  # type: ignore[arg-type]


class ProviderTimeoutError(ProviderError):
    """The provider did not answer in time."""

    def __init__(self, message: str, **kwargs: object) -> None:
        kwargs.setdefault("retryable", True)
        super().__init__(message, **kwargs)  # type: ignore[arg-type]


class EngineError(VoiceAgentError):
    """A speech-to-speech engine failed or was used incorrectly."""


class TransportError(VoiceAgentError):
    """An audio transport (local audio, WebSocket, WebRTC, telephony...) failed."""


class ToolError(VoiceAgentError):
    """A function tool failed. The message is reported back to the model."""
