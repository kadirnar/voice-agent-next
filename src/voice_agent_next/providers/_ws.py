"""Helpers shared by the WebSocket providers.

* :func:`ws_connect` opens a client connection and maps handshake failures to
  :mod:`voice_agent_next.errors` (the status code through a provider callback, which
  usually ends in :func:`voice_agent_next.errors.for_status`).
* :func:`close_ws` closes a connection, never raising and never hanging.
* :func:`raise_task_error` / :func:`raise_reader_error` re-raise the failure of a
  reader/writer task pair.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from ..errors import ConfigurationError, ProviderConnectionError, ProviderTimeoutError

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection
    from websockets.http11 import Response

__all__ = [
    "body_text",
    "close_ws",
    "raise_reader_error",
    "raise_task_error",
    "task_error",
    "ws_connect",
]

HttpErrorFactory = Callable[["Response"], Exception]
"""Maps the HTTP response of a rejected handshake to the exception to raise."""


def body_text(response: Response, limit: int | None = None) -> str:
    """The decoded body of a rejected handshake (``""`` when empty)."""
    text = response.body.decode("utf-8", "replace") if response.body else ""
    return text if limit is None else text[:limit]


async def ws_connect(
    url: str,
    *,
    provider: str,
    target: str,
    name: str,
    http_error: HttpErrorFactory,
    headers: Mapping[str, str] | None = None,
    open_timeout: float,
    close_timeout: float | None = 2.0,
    **kwargs: Any,
) -> ClientConnection:
    """Open a WebSocket client connection, mapping failures to library errors.

    Args:
        url: the ``ws://`` / ``wss://`` URL.
        provider: the provider name set on the raised errors.
        target: what is connected to, for messages (``"the Soniox real-time API"``).
        name: the provider's display name, for ``invalid <name> URL`` messages.
        http_error: builds the exception for a rejected handshake (HTTP status != 101).
        headers: extra request headers.
        open_timeout: seconds allowed for the TCP/TLS/HTTP handshake.
        close_timeout: seconds allowed for the closing handshake.
        **kwargs: passed to :func:`websockets.asyncio.client.connect` (``max_size``,
            ``compression``, ``proxy``...).

    Raises:
        ConfigurationError: invalid URL.
        ProviderTimeoutError: the handshake timed out.
        ProviderConnectionError: the connection failed (refused, reset, bad handshake).
        Exception: whatever ``http_error`` returns, for a rejected handshake.
    """
    from websockets.asyncio.client import connect
    from websockets.exceptions import InvalidHandshake, InvalidStatus, InvalidURI

    try:
        return await connect(
            url,
            additional_headers=dict(headers) if headers else None,
            open_timeout=open_timeout,
            close_timeout=close_timeout,
            **kwargs,
        )
    except InvalidStatus as exc:
        raise http_error(exc.response) from exc
    except InvalidURI as exc:
        raise ConfigurationError(f"invalid {name} URL: {exc}") from exc
    except TimeoutError as exc:
        raise ProviderTimeoutError(f"timed out connecting to {target}", provider=provider) from exc
    except (OSError, InvalidHandshake) as exc:
        raise ProviderConnectionError(
            f"could not connect to {target}: {exc}", provider=provider
        ) from exc


async def close_ws(ws: ClientConnection | None, *, timeout: float = 2.0) -> None:
    """Close ``ws`` (if any) within ``timeout`` seconds, ignoring every error."""
    if ws is None:
        return
    with contextlib.suppress(Exception):
        async with asyncio.timeout(timeout):
            await ws.close()


def task_error(task: asyncio.Task[Any]) -> BaseException | None:
    """The exception of a finished, non-cancelled ``task`` (``None`` otherwise)."""
    return task.exception() if task.done() and not task.cancelled() else None


def raise_task_error(*tasks: asyncio.Task[Any]) -> None:
    """Raise the exception of the first of ``tasks`` that failed.

    Every task's exception is retrieved (no "exception was never retrieved" warnings).
    """
    errors = [task_error(task) for task in tasks]
    for error in errors:
        if error is not None:
            raise error


async def raise_reader_error(
    reader: asyncio.Task[Any], writer: asyncio.Task[Any], *, grace: float = 0.5
) -> None:
    """Raise the error of ``reader`` or else ``writer``, preferring the server's reason.

    A writer failing because the server hung up is usually followed by the server's
    explanation (an error message) on the reader side: wait ``grace`` seconds for it so
    users see "invalid model" rather than "connection closed".
    """
    if task_error(writer) is not None and not reader.done():
        await asyncio.wait((reader,), timeout=grace)
    raise_task_error(reader, writer)
