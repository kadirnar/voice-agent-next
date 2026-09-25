"""The shared provider helpers: ``errors.for_status``, ``MissingAPIKeyError`` and
``providers/_ws.py`` (against a local WebSocket server)."""

from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import AsyncIterator
from typing import Any

import pytest
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.http11 import Request, Response

from voice_agent_next.errors import (
    AuthenticationError,
    ConfigurationError,
    MissingAPIKeyError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
    for_status,
)
from voice_agent_next.providers._ws import (
    body_text,
    close_ws,
    raise_reader_error,
    raise_task_error,
    ws_connect,
)
from voice_agent_next.registry import ComponentKind, create, get_provider


@pytest.mark.parametrize(
    ("status", "error", "retryable"),
    [
        (400, ProviderError, False),
        (401, AuthenticationError, False),
        (403, AuthenticationError, False),
        (404, ProviderError, False),
        (408, ProviderTimeoutError, True),
        (409, ProviderError, True),
        (422, ProviderError, False),
        (429, RateLimitError, True),
        (500, ProviderConnectionError, True),
        (502, ProviderConnectionError, True),
        (503, ProviderConnectionError, True),
        (504, ProviderTimeoutError, True),
        (529, ProviderConnectionError, True),
        (None, ProviderError, False),
    ],
)
def test_for_status_mapping(
    status: int | None, error: type[ProviderError], retryable: bool
) -> None:
    err = for_status(status, "boom", provider="acme")
    assert type(err) is error
    assert (str(err), err.provider, err.status_code, err.retryable) == (
        "boom",
        "acme",
        status,
        retryable,
    )


def test_for_status_retryable_override() -> None:
    quota = for_status(429, "no credits", retryable=False)
    assert isinstance(quota, RateLimitError) and quota.retryable is False
    assert for_status(503, "x", retryable=False).retryable is False


def test_missing_api_key_error_is_a_configuration_error() -> None:
    err = MissingAPIKeyError("no key", provider="acme")
    assert isinstance(err, ConfigurationError) and isinstance(err, ValueError)
    assert isinstance(err, AuthenticationError)  # backward compatible
    assert (err.provider, err.retryable, err.status_code) == ("acme", False, None)


# ---------------------------------------------------------------------- ws_connect
@pytest.fixture
async def rejecting_server() -> AsyncIterator[str]:
    def reject(connection: ServerConnection, request: Request) -> Response:
        status = int(request.path.strip("/") or "503")
        return connection.respond(status, f"nope {status}\n")

    async def handler(ws: ServerConnection) -> None:  # pragma: no cover - never reached
        await ws.close()

    server: Server = await serve(handler, "127.0.0.1", 0, process_request=reject)
    port = next(iter(server.sockets)).getsockname()[1]
    try:
        yield f"ws://127.0.0.1:{port}"
    finally:
        server.close()
        await server.wait_closed()


def _http_error(response: Response) -> ProviderError:
    return for_status(response.status_code, f"HTTP {response.status_code}: {body_text(response)}")


async def _connect(url: str, *, open_timeout: float = 10.0) -> None:
    ws = await ws_connect(
        url,
        provider="acme",
        target="the Acme API",
        name="Acme",
        http_error=_http_error,
        headers={"Authorization": "Token k"},
        open_timeout=open_timeout,
    )
    await close_ws(ws)


@pytest.mark.parametrize(
    ("status", "error"),
    [(401, AuthenticationError), (429, RateLimitError), (503, ProviderConnectionError)],
)
async def test_ws_connect_maps_rejected_handshakes(
    rejecting_server: str, status: int, error: type[ProviderError]
) -> None:
    with pytest.raises(error, match=f"HTTP {status}: nope {status}") as info:
        await _connect(f"{rejecting_server}/{status}")
    assert info.value.status_code == status


async def test_ws_connect_maps_invalid_urls_and_refused_connections() -> None:
    with pytest.raises(ConfigurationError, match="invalid Acme URL"):
        await _connect("http://127.0.0.1:1/not-a-websocket-url")
    with socket.socket() as sock:  # a port nobody listens on
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    # refused at once, or (Windows retries the SYN for ~2 s) the handshake times out
    with pytest.raises((ProviderConnectionError, ProviderTimeoutError), match="the Acme API"):
        await _connect(f"ws://127.0.0.1:{port}", open_timeout=5.0)


async def test_ws_connect_hint_and_timeout_error() -> None:
    """``hint`` is appended to the messages; ``timeout_error`` picks the timeout class."""

    async def silent(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.read()  # accept TCP, never answer the HTTP upgrade
        writer.close()

    server = await asyncio.start_server(silent, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    kw: dict[str, Any] = {
        "provider": "acme",
        "target": "the Acme server",
        "name": "Acme",
        "http_error": _http_error,
        "open_timeout": 0.3,
        "hint": "is it running?",
    }
    try:
        with pytest.raises(ProviderTimeoutError, match=r"timed out .*\(is it running\?\)"):
            await ws_connect(f"ws://127.0.0.1:{port}", **kw)
        with pytest.raises(ProviderConnectionError, match=r"\(is it running\?\)") as info:
            await ws_connect(f"ws://127.0.0.1:{port}", timeout_error=ProviderConnectionError, **kw)
        assert type(info.value) is ProviderConnectionError and info.value.retryable
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.parametrize("module", ["moshi", "google.live"])
async def test_engines_connect_through_the_shared_helper(
    module: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Moshi and Gemini Live open their sockets with ``ws_connect`` (#157)."""
    import importlib

    from voice_agent_next.engine import EngineOptions

    mod = importlib.import_module(f"voice_agent_next.providers.{module}")
    calls: list[dict[str, Any]] = []

    async def fake_connect(url: str, **kw: Any) -> Any:
        calls.append({"url": url, **kw})
        raise ProviderConnectionError("stop here", provider=kw["provider"])

    monkeypatch.setattr(mod, "ws_connect", fake_connect)
    if module == "moshi":
        engine: Any = mod.MoshiEngine(base_url="ws://127.0.0.1:9", reconnect=False)
    else:
        engine = mod.GeminiLiveEngine(api_key="k", base_url="ws://127.0.0.1:9")
    with pytest.raises(ProviderConnectionError, match="stop here"):
        await engine.connect(EngineOptions())
    (call,) = calls
    assert call["url"].startswith("ws://127.0.0.1:9")
    assert call["timeout_error"] is ProviderConnectionError


async def test_ws_connect_opens_a_connection() -> None:
    seen: list[str | None] = []

    async def handler(ws: ServerConnection) -> None:
        assert ws.request is not None
        seen.append(ws.request.headers.get("Authorization"))
        await ws.wait_closed()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = next(iter(server.sockets)).getsockname()[1]
        await _connect(f"ws://127.0.0.1:{port}")
    assert seen == ["Token k"]


async def test_close_ws_ignores_none_and_errors() -> None:
    class Broken:
        async def close(self) -> None:
            raise OSError("gone")

    class Hanging:
        async def close(self) -> None:
            await asyncio.sleep(3600)

    await close_ws(None)
    await close_ws(Broken())  # type: ignore[arg-type]
    await close_ws(Hanging(), timeout=0.05)  # type: ignore[arg-type]


# ---------------------------------------------------------------- task errors
async def _fail(exc: BaseException, delay: float = 0.0) -> None:
    await asyncio.sleep(delay)
    raise exc


async def test_raise_task_error_raises_the_first_failure_and_retrieves_all() -> None:
    ok = asyncio.create_task(asyncio.sleep(0))
    first = asyncio.create_task(_fail(ValueError("first")))
    second = asyncio.create_task(_fail(KeyError("second")))
    await asyncio.wait({ok, first, second})
    raise_task_error(ok)  # nothing failed
    with pytest.raises(ValueError, match="first"):
        raise_task_error(ok, first, second)
    assert second._log_traceback is False  # retrieved: no "never retrieved" warning


async def test_raise_reader_error_prefers_the_servers_reason() -> None:
    writer = asyncio.create_task(_fail(ConnectionError("closed")))
    reader = asyncio.create_task(_fail(ProviderError("invalid model"), delay=0.05))
    await asyncio.wait({writer})
    with pytest.raises(ProviderError, match="invalid model"):
        await raise_reader_error(reader, writer, grace=5.0)
    # without the reader's reason in time, the writer's error is raised
    writer = asyncio.create_task(_fail(ConnectionError("closed")))
    reader = asyncio.create_task(asyncio.sleep(3600))
    await asyncio.wait({writer})
    with pytest.raises(ConnectionError, match="closed"):
        await raise_reader_error(reader, writer, grace=0.01)
    reader.cancel()


@pytest.mark.parametrize(
    ("kind", "name"),
    [
        ("stt", "deepgram"), ("tts", "deepgram"), ("stt", "assemblyai"), ("stt", "soniox"),
        ("stt", "speechmatics"), ("stt", "cartesia"), ("tts", "cartesia"),
        ("stt", "elevenlabs"), ("tts", "elevenlabs"), ("llm", "openai"), ("stt", "openai"),
        ("tts", "openai"), ("engine", "openai"), ("llm", "groq"), ("llm", "anthropic"),
        ("llm", "google"), ("tts", "google"),
    ],
)  # fmt: skip
def test_a_missing_key_is_one_error_type(
    monkeypatch: pytest.MonkeyPatch, kind: ComponentKind, name: str
) -> None:
    spec = get_provider(kind, name)
    if spec.missing_dependencies():
        pytest.skip(f"needs {spec.missing_dependencies()}")
    for var in list(os.environ):
        if var.endswith("_API_KEY") or var.startswith(("GOOGLE", "GEMINI", "ELEVEN")):
            monkeypatch.delenv(var)
    with pytest.raises(MissingAPIKeyError) as info:
        create(kind, name)
    assert isinstance(info.value, ConfigurationError)
