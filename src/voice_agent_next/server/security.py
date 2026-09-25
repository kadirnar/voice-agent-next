"""Secure-by-default serving: Origin allow-lists, exposure checks, limits, safe errors.

Shared by every WebSocket server of the library (:class:`~voice_agent_next.server.RealtimeServer`,
:class:`~voice_agent_next.transports.websocket.WebSocketAgentServer` and the servers built on
it) and by ``van serve``:

* :class:`OriginPolicy` — which browser pages may open a session. By default only pages
  served from the machine itself (``http://localhost:*``, ``http://127.0.0.1:*``,
  ``http://[::1]:*``) and clients that send no ``Origin`` header (native clients, SDKs,
  telephony providers). Any other website is refused with HTTP 403, which stops
  cross-site WebSocket hijacking: a page a user happens to visit cannot drive the agent
  (and spend its engine) from their browser;
* :func:`is_loopback_host` / :func:`exposure_warning` — binding a non-loopback address
  without authentication is flagged (``van serve`` refuses it for OpenAI Realtime unless
  ``--insecure`` is given);
* the default limits (:data:`DEFAULT_MAX_SESSIONS`, :data:`DEFAULT_MAX_SESSION_DURATION`,
  :data:`DEFAULT_IDLE_TIMEOUT`) and the queue high-water marks every WebSocket session uses;
* :func:`report_error` — clients get a generic message with a correlation id; the full
  exception is only logged, under the same id.

See the security section of ``docs/deploy/serving.md``.
"""

from __future__ import annotations

import ipaddress
import logging
from collections.abc import Iterable
from typing import Final
from urllib.parse import urlsplit

from ..utils.ids import new_id
from ..utils.log import logger

__all__ = [
    "DEFAULT_IDLE_TIMEOUT",
    "DEFAULT_MAX_SESSIONS",
    "DEFAULT_MAX_SESSION_DURATION",
    "INBOX_HIGH",
    "INBOX_LOW",
    "MAX_SEND_BUFFER",
    "OUTBOX_HIGH",
    "OUTBOX_LOW",
    "OriginPolicy",
    "exposure_warning",
    "header_origin",
    "is_loopback_host",
    "report_error",
]

DEFAULT_MAX_SESSIONS: Final = 64
"""Concurrent sessions per server (process) unless configured (``None``: no limit)."""
DEFAULT_MAX_SESSION_DURATION: Final = 3600.0
"""Seconds after which a session is closed unless configured (OpenAI's limit, too)."""
DEFAULT_IDLE_TIMEOUT: Final = 300.0
"""Seconds without any client message after which a session is closed."""

INBOX_HIGH: Final = 4 * 2**20
"""Client bytes queued (not yet consumed by the session) at which the server stops
reading the socket — TCP backpressure slows a flooding client down."""
INBOX_LOW: Final = 2**20
"""... and at which it resumes reading."""
OUTBOX_HIGH: Final = 2 * 2**20
"""Bytes queued for the client at which the Realtime server stops taking engine events."""
OUTBOX_LOW: Final = 512 * 2**10
"""... and at which it takes them again."""
MAX_SEND_BUFFER: Final = 32 * 2**20
"""Bytes queued for a client that does not read them before the connection is closed
(1008): a client that never reads cannot grow the server's memory without bound."""

_LOOPBACK_NAMES: Final = frozenset({"localhost", "localhost.localdomain", "ip6-localhost"})
_DEFAULT_PORTS: Final = {"http": 80, "https": 443, "ws": 80, "wss": 443}


def is_loopback_host(host: str | None) -> bool:
    """``host`` (a bind address or a host name) only reaches this machine.

    ``""``/``None``/``0.0.0.0``/``::`` (every interface) and other names are not loopback.
    """
    if not host:
        return False
    name = host.strip().strip("[]").lower().rstrip(".")
    if name in _LOOPBACK_NAMES or name.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(name.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _split_origin(origin: str) -> tuple[str, str, int | None] | None:
    """``(scheme, host, port)`` of an Origin (``None``: not a URL origin, e.g. ``null``)."""
    try:
        parts = urlsplit(origin.strip())
        host, port = parts.hostname, parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if not scheme or not host:
        return None
    return scheme, host.lower().rstrip("."), port or _DEFAULT_PORTS.get(scheme)


class OriginPolicy:
    """Decides which ``Origin`` headers may open a WebSocket session.

    Always allowed: requests without an ``Origin`` header (browsers always send one on a
    WebSocket upgrade, so these are native clients) and pages served from this machine
    (a loopback host name or address, any port). ``allowed`` adds origins:

    * exact origins, ``"https://app.example.com"`` (the port matters when not the
      default one: ``"http://intranet:8080"``);
    * a wildcard subdomain, ``"https://*.example.com"`` (not ``example.com`` itself);
    * ``"null"`` (sandboxed iframes, ``file://`` pages — anyone can produce it);
    * ``"*"``: every origin (only behind other authentication).

    Args:
        allowed: extra origins (see above), or a single string.
        allow_localhost: also allow loopback origins (default ``True``).
    """

    def __init__(self, allowed: str | Iterable[str] | None = (), *, allow_localhost: bool = True):
        values = [allowed] if isinstance(allowed, str) else list(allowed or ())
        self.allow_any = False
        self.allow_null = False
        self.allow_localhost = allow_localhost
        self._exact: set[tuple[str, str, int | None]] = set()
        self._wildcards: set[tuple[str, str, int | None]] = set()
        self.allowed: tuple[str, ...] = tuple(v.strip() for v in values if v and v.strip())
        for value in self.allowed:
            if value == "*":
                self.allow_any = True
                continue
            if value.lower() == "null":
                self.allow_null = True
                continue
            wildcard = "://*." in value
            parsed = _split_origin(value.replace("://*.", "://", 1) if wildcard else value)
            if parsed is None:
                raise ValueError(
                    f"invalid allowed origin {value!r}: use scheme://host[:port], e.g. "
                    "'https://app.example.com', 'https://*.example.com', 'null' or '*'"
                )
            (self._wildcards if wildcard else self._exact).add(parsed)

    def __repr__(self) -> str:
        return f"OriginPolicy({list(self.allowed)!r}, allow_localhost={self.allow_localhost})"

    def allows(self, origin: str | None) -> bool:
        """Whether a request with this ``Origin`` header (``None``: none) may connect."""
        if origin is None or self.allow_any:
            return True
        origin = origin.strip()
        if origin.lower() == "null":
            return self.allow_null
        parsed = _split_origin(origin)
        if parsed is None:
            return False
        scheme, host, port = parsed
        if self.allow_localhost and scheme in ("http", "https") and is_loopback_host(host):
            return True
        if parsed in self._exact:
            return True
        return any(
            scheme == w_scheme and port == w_port and host.endswith("." + w_host)
            for w_scheme, w_host, w_port in self._wildcards
        )


def header_origin(values: list[str]) -> str | None:
    """The ``Origin`` of a request from all its ``Origin`` header values (``None``: none;
    several values yield an invalid origin, which only ``"*"`` allows)."""
    if not values:
        return None
    return values[0] if len(values) == 1 else "invalid:multiple-origins"


def exposure_warning(host: str | None, *, authenticated: bool, what: str) -> str | None:
    """The warning to log when ``what`` listens beyond this machine without authentication
    (``None`` when it binds a loopback address or authenticates its clients)."""
    if authenticated or is_loopback_host(host):
        return None
    shown = host or "all interfaces"
    return (
        f"{what} listens on {shown} without authentication: anyone who can reach this "
        "port can open sessions (and spend the engine). Require an API key, put an "
        "authenticating proxy in front, or bind 127.0.0.1."
    )


def report_error(
    exc: BaseException,
    what: str,
    *,
    session_id: str | None = None,
    level: int = logging.ERROR,
    traceback: bool = True,
) -> tuple[str, str]:
    """Log ``exc`` in full under a new correlation id; return ``(error_id, client_message)``.

    The client message never contains the exception text (paths, hosts, provider
    responses, stack details): only ``what`` failed and the id to find it in the logs.
    ``what`` is a short, client-safe description (e.g. ``"The session failed"``).
    """
    error_id = new_id("err_")
    logger.log(
        level,
        "%s (error id %s%s): %s",
        what,
        error_id,
        f", session {session_id}" if session_id else "",
        describe(exc),
        exc_info=(type(exc), exc, exc.__traceback__) if traceback else None,
        extra={"error_id": error_id},
    )
    return error_id, f"{what} (error id {error_id})."


def describe(exc: BaseException) -> str:
    """``Type: message`` of an exception — for logs only, never for clients."""
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__
