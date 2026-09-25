"""Secure-by-default serving (#139): Origin allow-list, exposure checks, default limits,
bounded WebSocket queues and client-safe error messages."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
from collections.abc import Callable
from typing import Any

import pytest
from typer.testing import CliRunner
from websockets.asyncio.client import ClientConnection, connect
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed, InvalidStatus

from voice_agent_next import Agent, AgentSession, AudioFrame
from voice_agent_next.cli.main import app as cli_app
from voice_agent_next.cli.serve import ServeOptions, SourceOptions, build_served, check_exposure
from voice_agent_next.errors import ConfigurationError, SessionRefused
from voice_agent_next.providers.mock import MockEngine
from voice_agent_next.server import RealtimeServer
from voice_agent_next.server.security import (
    DEFAULT_IDLE_TIMEOUT,
    DEFAULT_MAX_SESSION_DURATION,
    DEFAULT_MAX_SESSIONS,
    INBOX_HIGH,
    MAX_SEND_BUFFER,
    OriginPolicy,
    exposure_warning,
    is_loopback_host,
    report_error,
)
from voice_agent_next.session.events import AgentState, UserState
from voice_agent_next.transports.websocket import (
    PROTOCOL,
    SessionBridge,
    WebSocketAgentServer,
    WebSocketServerTransport,
)
from voice_agent_next.utils import EventEmitter, cancel_and_wait

KEY = "sk-local-test"
EVIL = "https://evil.example"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


async def wait_for(predicate: Callable[[], Any], timeout: float = 10.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


def hello() -> str:
    return json.dumps({"type": "hello", "protocol": PROTOCOL, "sample_rate": 16_000})


async def text_messages(ws: ClientConnection) -> list[dict[str, Any]]:
    """Every JSON message until the server closes the connection (any close code)."""
    messages: list[dict[str, Any]] = []
    with contextlib.suppress(ConnectionClosed):
        async for message in ws:
            if isinstance(message, str):
                messages.append(json.loads(message))
    return messages


def agent_server(**kw: Any) -> WebSocketAgentServer:
    return WebSocketAgentServer(
        lambda: AgentSession(MockEngine(responses=["ok"])), lambda: Agent("x"), port=0, **kw
    )


async def refused_status(url: str, origin: str | None, **kw: Any) -> tuple[int, dict[str, Any]]:
    """HTTP status (101 = accepted) and JSON body of a WebSocket upgrade from ``origin``."""
    try:
        async with connect(url, origin=origin, **kw) as ws:  # type: ignore[arg-type]
            await ws.close()
            return 101, {}
    except InvalidStatus as exc:
        body = exc.response.body or b""
        return exc.response.status_code, json.loads(body) if body.startswith(b"{") else {}


# ------------------------------------------------------------------------ units
@pytest.mark.parametrize(
    ("host", "loopback"),
    [
        ("127.0.0.1", True),
        ("127.1.2.3", True),
        ("::1", True),
        ("[::1]", True),
        ("localhost", True),
        ("app.localhost", True),
        ("0.0.0.0", False),
        ("::", False),
        ("", False),
        (None, False),
        ("192.168.1.5", False),
        ("example.com", False),
    ],
)
def test_is_loopback_host(host: str | None, loopback: bool) -> None:
    assert is_loopback_host(host) is loopback


@pytest.mark.parametrize(
    ("origin", "allowed"),
    [
        (None, True),  # native clients send no Origin
        ("http://localhost:3000", True),
        ("http://127.0.0.1:8765", True),
        ("https://[::1]", True),
        ("http://app.localhost:5173", True),
        (EVIL, False),
        ("http://localhost.evil.example", False),
        ("http://127.0.0.1.evil.example", False),
        ("null", False),
        ("not an origin", False),
        ("ftp://localhost", False),
        ("https://app.example.com", True),
        ("https://app.example.com:443", True),
        ("http://app.example.com", False),  # another scheme
        ("https://app.example.com:8443", False),  # another port
        ("https://x.corp.example", True),
        ("https://a.b.corp.example", True),
        ("https://corp.example", False),  # the wildcard is for subdomains
        ("https://evilcorp.example", False),
        ("invalid:multiple-origins", False),
    ],
)
def test_origin_policy(origin: str | None, allowed: bool) -> None:
    policy = OriginPolicy(["https://app.example.com", "https://*.corp.example"])
    assert policy.allows(origin) is allowed


def test_origin_policy_options() -> None:
    assert OriginPolicy("*").allows(EVIL) and OriginPolicy("*").allows("null")
    assert OriginPolicy(["null"]).allows("null") and not OriginPolicy().allows("null")
    assert not OriginPolicy(allow_localhost=False).allows("http://localhost:3000")
    assert OriginPolicy("HTTP://Intranet:8080/").allows("http://intranet:8080")
    with pytest.raises(ValueError, match="invalid allowed origin"):
        OriginPolicy(["app.example.com"])


def test_exposure_warning() -> None:
    assert exposure_warning("127.0.0.1", authenticated=False, what="x") is None
    assert exposure_warning("0.0.0.0", authenticated=True, what="x") is None
    warning = exposure_warning("0.0.0.0", authenticated=False, what="the server")
    assert warning is not None and "without authentication" in warning


def test_report_error_keeps_details_in_the_log(caplog: pytest.LogCaptureFixture) -> None:
    secret = "postgres://admin:hunter2@10.0.0.7/db"
    try:
        raise RuntimeError(secret)
    except RuntimeError as exc:
        error_id, message = report_error(exc, "The session failed", session_id="sess_1")
    assert re.fullmatch(r"err_[0-9a-f]{12}", error_id)
    assert message == f"The session failed (error id {error_id})."
    assert secret not in message and "RuntimeError" not in message
    (record,) = [r for r in caplog.records if error_id in r.getMessage()]
    assert secret in record.getMessage() and "sess_1" in record.getMessage()
    assert record.exc_info is not None and record.levelno == logging.ERROR


def test_secure_defaults() -> None:
    assert (DEFAULT_MAX_SESSIONS, DEFAULT_MAX_SESSION_DURATION, DEFAULT_IDLE_TIMEOUT) == (
        64, 3600.0, 300.0,
    )  # fmt: skip
    realtime = RealtimeServer(MockEngine())
    assert realtime.host == "127.0.0.1"
    assert (realtime.max_sessions, realtime.max_session_duration, realtime.idle_timeout) == (
        64, 3600.0, 300.0,
    )  # fmt: skip
    ws = agent_server()
    assert ws.host == "127.0.0.1"
    assert (ws.max_sessions, ws.max_session_duration, ws.idle_timeout) == (64, 3600.0, 300.0)
    assert WebSocketServerTransport().max_send_buffer == MAX_SEND_BUFFER
    options = ServeOptions()
    assert (options.host, options.max_sessions, options.max_session_duration) == (
        "127.0.0.1", 64, 3600.0,
    )  # fmt: skip
    assert options.idle_timeout == 300.0 and not options.insecure and not options.allowed_origins
    # limits can still be lifted explicitly
    lifted = RealtimeServer(MockEngine(), max_sessions=None, max_session_duration=None,
                            idle_timeout=None)  # fmt: skip
    assert (lifted.max_sessions, lifted.max_session_duration, lifted.idle_timeout) == (
        None, None, None,
    )  # fmt: skip
    with pytest.raises(ConfigurationError):
        RealtimeServer(MockEngine(), idle_timeout=0)
    with pytest.raises(ConfigurationError, match="invalid allowed origin"):
        RealtimeServer(MockEngine(), allowed_origins=["example.com"])
    with pytest.raises(ValueError):
        agent_server(max_session_duration=-1)


# ------------------------------------------------------------------ CLI exposure
def test_cli_refuses_unauthenticated_realtime_beyond_loopback() -> None:
    refusal = check_exposure(ServeOptions(host="0.0.0.0"))
    assert refusal is not None and "--api-key" in refusal and "--insecure" in refusal
    assert check_exposure(ServeOptions(host="0.0.0.0", api_keys=[KEY])) is None
    assert check_exposure(ServeOptions(host="0.0.0.0", insecure=True)) is None
    assert check_exposure(ServeOptions(host="127.0.0.1")) is None

    result = CliRunner().invoke(cli_app, ["serve", "--host", "0.0.0.0", "--port", "0"])
    assert result.exit_code == 2, result.output
    assert "refusing to serve openai-realtime" in ANSI.sub("", result.output)


def test_cli_warns_for_protocols_without_authentication(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert check_exposure(ServeOptions(protocol="websocket", host="0.0.0.0")) is None
    assert "no authentication" in ANSI.sub("", capsys.readouterr().err)
    assert check_exposure(ServeOptions(protocol="websocket", host="0.0.0.0", insecure=True)) is None
    assert "no authentication" not in capsys.readouterr().err


def test_serve_options_reach_the_websocket_server() -> None:
    options = ServeOptions(
        protocol="websocket", port=0, allowed_origins=["https://app.example.com"],
        max_sessions=3, max_session_duration=60.0, idle_timeout=None, warmup=False,
    )  # fmt: skip
    server = build_served(SourceOptions(), options).server
    assert isinstance(server, WebSocketAgentServer)
    assert server.origin_policy.allows("https://app.example.com")
    assert not server.origin_policy.allows(EVIL)
    assert (server.max_sessions, server.max_session_duration, server.idle_timeout) == (
        3, 60.0, None,
    )  # fmt: skip
    realtime = build_served(SourceOptions(), ServeOptions(allowed_origins=["*"], port=0)).server
    assert realtime.origin_policy.allows(EVIL) and realtime.idle_timeout == 300.0


# ---------------------------------------------------------------- cross-Origin
async def test_realtime_server_rejects_cross_origin_pages() -> None:
    async with RealtimeServer(MockEngine(), port=0, api_keys=KEY) as server:
        url = f"{server.url}/realtime"
        auth = {"additional_headers": {"Authorization": f"Bearer {KEY}"}}
        status, body = await refused_status(url, EVIL, **auth)
        assert status == 403 and body["error"]["code"] == "origin_not_allowed"
        # even without a key: the Origin check comes first and says nothing about keys
        status, body = await refused_status(url, EVIL)
        assert status == 403
        for ok in (None, "http://localhost:3000", "http://127.0.0.1:5500"):
            assert (await refused_status(url, ok, **auth))[0] == 101
    async with RealtimeServer(
        MockEngine(), port=0, allowed_origins=["https://app.example.com"]
    ) as server:
        url = f"{server.url}/realtime"
        assert (await refused_status(url, "https://app.example.com"))[0] == 101
        assert (await refused_status(url, EVIL))[0] == 403


async def test_websocket_server_rejects_cross_origin_pages() -> None:
    async with agent_server() as server:
        status, body = await refused_status(server.url, EVIL)
        assert status == 403 and body["code"] == "origin_not_allowed"
        assert (await refused_status(server.url, None))[0] == 101
        assert (await refused_status(server.url, "http://localhost:8080"))[0] == 101
        assert not server.sessions
    async with agent_server(allowed_origins="https://app.example.com") as server:
        assert (await refused_status(server.url, "https://app.example.com"))[0] == 101
        assert (await refused_status(server.url, EVIL))[0] == 403
    # the websockets `origins` option still works (and replaces allowed_origins)
    async with agent_server(origins=[EVIL]) as server:
        assert (await refused_status(server.url, EVIL))[0] == 101
        assert (await refused_status(server.url, "http://localhost:8080"))[0] == 403


async def test_standalone_transport_rejects_cross_origin_pages() -> None:
    transport = WebSocketServerTransport(port=0)
    try:
        await transport.listen()
        assert (await refused_status(transport.url, EVIL))[0] == 403
    finally:
        await transport.aclose()


# ------------------------------------------------------------ bounded queues
class _Harness:
    """A websockets server whose connections get a per-connection transport."""

    def __init__(self, **transport_options: Any) -> None:
        self.options = transport_options
        self.transports: list[WebSocketServerTransport] = []
        self._done = asyncio.Event()

    async def handler(self, websocket: ServerConnection) -> None:
        transport = WebSocketServerTransport(websocket, **self.options)
        self.transports.append(transport)
        await transport.start()
        await self._done.wait()
        await transport.aclose()

    def finish(self) -> None:
        self._done.set()


async def test_flooding_client_is_slowed_down_by_backpressure() -> None:
    """A client sending audio faster than the session consumes it: the server stops
    reading at the high-water mark (TCP backpressure), queued audio stays bounded, and
    nothing is lost once the session catches up."""
    harness = _Harness()
    chunk = bytes(64 * 1024)
    total = 48 * 2**20  # far more than socket buffers hold
    async with serve(harness.handler, "127.0.0.1", 0, max_size=None) as server:
        port = server.sockets[0].getsockname()[1]
        async with connect(f"ws://127.0.0.1:{port}", compression=None) as ws:
            await ws.send(hello())
            assert json.loads(await ws.recv())["type"] == "ready"

            async def flood() -> None:
                for _ in range(total // len(chunk)):
                    await ws.send(chunk)

            sender = asyncio.create_task(flood())
            try:
                await wait_for(lambda: harness.transports and harness.transports[0]._input.full)
                transport = harness.transports[0]
                peak = 0
                for _ in range(30):  # the session never consumes: the inbox stays bounded
                    peak = max(peak, transport._input.bytes)
                    await asyncio.sleep(0.01)
                assert INBOX_HIGH < peak <= INBOX_HIGH + len(chunk)
                assert not sender.done()  # the client is blocked, not buffered by us
                received = 0
                async for frame in transport.audio_input():  # the session catches up
                    received += len(frame.data)
                    assert transport._input.bytes <= INBOX_HIGH + len(chunk)
                    if received >= total:
                        break
                assert received == total
                await asyncio.wait_for(sender, 10)
            finally:
                await cancel_and_wait(sender)
                harness.finish()


async def test_never_reading_client_is_disconnected() -> None:
    """A client that never reads: the queue for it stays bounded, then it is dropped (1008)."""
    limit = 256 * 1024
    harness = _Harness(max_send_buffer=limit, output_sample_rate=24_000)
    async with serve(harness.handler, "127.0.0.1", 0, close_timeout=1) as server:
        port = server.sockets[0].getsockname()[1]
        ws = await connect(f"ws://127.0.0.1:{port}", compression=None, max_queue=1)
        try:
            await ws.send(hello())
            await wait_for(lambda: harness.transports and harness.transports[0].connected)
            transport = harness.transports[0]
            second = AudioFrame(bytes(48_000), 24_000)  # 1 s of agent audio
            peak = 0
            for _ in range(2000):  # up to ~96 MB of audio if nothing stopped it
                await transport.write_audio(second)
                peak = max(peak, transport._outbox.bytes)
                if not transport.connected:
                    break
                await asyncio.sleep(0)
            assert not transport.connected
            assert peak <= limit
            assert transport.close_code == 1008
            for _ in range(10):  # later audio is discarded, not queued
                await transport.write_audio(second)
            assert transport._outbox.bytes == 0
        finally:
            ws.transport.abort()
            harness.finish()


# ----------------------------------------------------------- session limits
async def test_websocket_server_closes_idle_and_expired_sessions() -> None:
    async with agent_server(idle_timeout=0.3, max_session_duration=None) as server:
        async with connect(server.url) as ws:
            await ws.send(hello())
            messages = await text_messages(ws)
            errors = [m for m in messages if m["type"] == "error"]
            assert errors[-1]["code"] == "session_idle" and errors[-1]["fatal"] is True
            assert ws.protocol.close_code == 1000
    async with agent_server(max_session_duration=0.3, idle_timeout=None) as server:
        async with connect(server.url) as ws:
            await ws.send(hello())
            messages = await text_messages(ws)
            assert [m for m in messages if m["type"] == "error"][-1]["code"] == "session_expired"


async def test_realtime_server_closes_idle_sessions() -> None:
    async with RealtimeServer(MockEngine(), port=0, idle_timeout=0.3) as server:
        async with connect(f"{server.url}/realtime") as ws:
            events = await text_messages(ws)
            errors = [e["error"] for e in events if e["type"] == "error"]
            assert errors[-1]["code"] == "session_idle"
            assert ws.protocol.close_code == 1000


async def test_refusals_from_factories_reach_the_client() -> None:
    def refuse() -> AgentSession:
        raise SessionRefused("invalid token", code="unauthorized")

    server = WebSocketAgentServer(refuse, lambda: Agent("x"), port=0)
    async with server, connect(server.url) as ws:
        await ws.send(hello())
        messages = await text_messages(ws)
        error = messages[-1]
        assert (error["code"], error["message"]) == ("unauthorized", "invalid token")
        assert ws.protocol.close_code == 1008


# ------------------------------------------------------------- bridge limits
class _FakeSession(EventEmitter):
    agent_state = AgentState.LISTENING
    user_state = UserState.LISTENING
    closed = False
    history: dict[str, Any] = {}

    async def generate_reply(self, **kw: Any) -> None:
        await asyncio.Event().wait()  # never answers


class _FakeTransport(EventEmitter):
    session_id = "ws_test"

    def __init__(self) -> None:
        super().__init__()
        self.sent: list[dict[str, Any]] = []

    def send_message_nowait(self, message: dict[str, Any]) -> None:
        self.sent.append(message)


async def test_a_flood_of_typed_messages_is_rate_limited() -> None:
    transport = _FakeTransport()
    bridge = SessionBridge(_FakeSession(), transport)  # type: ignore[arg-type]
    try:
        for _ in range(12):
            transport.emit("message", {"type": "text", "text": "hi"})
        await asyncio.sleep(0)
        limited = [m for m in transport.sent if m.get("code") == "rate_limited"]
        assert len(limited) == 4 and bridge._pending_replies == 8
    finally:
        await bridge.aclose()
    assert bridge._pending_replies == 0
