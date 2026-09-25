"""``van serve`` follow-ups (#156): layered flags, API keys for every protocol, the WebRTC
Origin check and limits, and the telephony answer webhook served by ``van serve``."""

from __future__ import annotations

import asyncio
import base64
import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from voice_agent_next import Agent, AgentSession
from voice_agent_next.cli import serve as serve_cli
from voice_agent_next.cli.main import app as cli_app
from voice_agent_next.cli.serve import (
    ServeOptions,
    SourceOptions,
    build_models,
    build_served,
    check_exposure,
    resolve_auth,
)
from voice_agent_next.errors import ConfigurationError
from voice_agent_next.providers.mock import MockEngine
from voice_agent_next.server import security
from voice_agent_next.server.security import (
    KEY_SUBPROTOCOL,
    ApiKeys,
    generate_api_key,
    request_credentials,
)
from voice_agent_next.server.serving import run_served
from voice_agent_next.transports.telephony import SECRET_ENV, TOKEN_PARAMETER, stream_token
from voice_agent_next.transports.websocket import (
    PROTOCOL,
    WebSocketAgentServer,
    _session_watchdog,
)

KEY = "van_test-key"
EVIL = "https://evil.example"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
CALL_SID = "CA5a1ebd8dcc0ff4c2a2ea3fcbbdf3a1c4"


async def wait_for(predicate: Any, timeout: float = 10.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


async def upgrade(url: str, **kw: Any) -> tuple[int, dict[str, Any], str | None]:
    """HTTP status (101 = accepted), JSON body and selected subprotocol of an upgrade."""
    try:
        async with connect(url, proxy=None, **kw) as ws:
            subprotocol = ws.subprotocol
            await ws.send(json.dumps({"type": "hello", "protocol": PROTOCOL}))
            await ws.close()
            return 101, {}, subprotocol
    except InvalidStatus as exc:
        body = exc.response.body or b""
        return exc.response.status_code, json.loads(body) if body.startswith(b"{") else {}, None


def header(values: dict[str, list[str]]) -> Any:
    lower = {k.lower(): v for k, v in values.items()}
    return lambda name: lower.get(name.lower(), [])


# ---------------------------------------------------------------------- API keys
def test_api_keys_accept_every_documented_carrier() -> None:
    keys = ApiKeys([KEY, "second"])
    assert keys and len(keys) == 2 and KEY not in repr(keys)
    basic = base64.b64encode(f"user:{KEY}".encode()).decode()
    for headers in (
        {"Authorization": [f"Bearer {KEY}"]},
        {"authorization": [f"bearer {KEY}"]},
        {"api-key": ["second"]},
        {"Authorization": [f"Basic {basic}"]},
        {"Sec-WebSocket-Protocol": [f"van-ws, {KEY_SUBPROTOCOL}{KEY}"]},
    ):
        assert keys.authorized(header(headers)), headers
    for headers in (
        {},
        {"Authorization": ["Bearer wrong"]},
        {"Authorization": [f"Token {KEY}"]},
        {"Authorization": ["Basic !!!not-base64"]},
        {"Sec-WebSocket-Protocol": [KEY]},
    ):
        assert not keys.authorized(header(headers)), headers
    assert keys.authorized(header({}), query_key=KEY)
    assert not keys.authorized(header({}), query_key="nope")
    assert ApiKeys().authorized(header({}))  # no keys: no authentication
    assert request_credentials(header({"Authorization": ["Bearer  "]})) == []
    with pytest.raises(ValueError):
        ApiKeys([""])
    generated = generate_api_key()
    assert generated.startswith("van_") and len(generated) > 30 and generated != generate_api_key()
    assert re.fullmatch(r"[A-Za-z0-9_\-]+", generated)  # a valid subprotocol token


async def test_websocket_server_requires_the_api_key() -> None:
    server = WebSocketAgentServer(
        lambda: AgentSession(MockEngine()), lambda: Agent("x"), port=0, api_keys=[KEY]
    )
    async with server:
        status, body, _ = await upgrade(server.url)
        assert status == 401 and body["code"] == "invalid_api_key"
        wrong = {"Authorization": "Bearer nope"}
        assert (await upgrade(server.url, additional_headers=wrong))[0] == 401
        good = {"Authorization": f"Bearer {KEY}"}
        assert (await upgrade(server.url, additional_headers=good))[0] == 101
        # browsers: the key as a subprotocol, with or without van-ws; one is selected
        status, _, selected = await upgrade(
            server.url,
            origin="http://localhost:3000",
            subprotocols=["van-ws", KEY_SUBPROTOCOL + KEY],
        )
        assert (status, selected) == (101, "van-ws")
        status, _, selected = await upgrade(server.url, subprotocols=[KEY_SUBPROTOCOL + KEY])
        assert (status, selected) == (101, KEY_SUBPROTOCOL + KEY)
        # the Origin check comes first
        status, body, _ = await upgrade(server.url, origin=EVIL, additional_headers=good)
        assert status == 403 and body["code"] == "origin_not_allowed"


async def test_ops_routes_stay_open_and_sessions_need_the_key() -> None:
    served = build_served(
        SourceOptions(), ServeOptions(protocol="websocket", port=0, api_keys=[KEY], warmup=False)
    )
    task = asyncio.create_task(run_served(served, handle_signals=False))
    await wait_for(lambda: served.server.port)
    await wait_for(lambda: served.state.ready)
    port = served.server.port
    async with httpx.AsyncClient(trust_env=False) as client:
        assert (await client.get(f"http://127.0.0.1:{port}/ready")).status_code == 200
    url = f"ws://127.0.0.1:{port}/"
    assert (await upgrade(url))[0] == 401
    assert (await upgrade(url, additional_headers={"Authorization": f"Bearer {KEY}"}))[0] == 101
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# ----------------------------------------------------------- secure defaults (CLI)
def test_resolve_auth_generates_what_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SECRET_ENV, raising=False)
    local = ServeOptions(protocol="websocket")
    resolve_auth(local)
    assert local.api_keys == [] and local.generated == []  # loopback: no key needed
    for protocol in ("websocket", "webrtc"):
        exposed = ServeOptions(protocol=protocol, host="0.0.0.0")
        resolve_auth(exposed)
        assert len(exposed.api_keys) == 1 and exposed.generated == ["api key"]
        assert check_exposure(exposed) is None
        resolve_auth(exposed)  # idempotent: the workers get the same key
        assert len(exposed.api_keys) == 1
        given = ServeOptions(protocol=protocol, host="0.0.0.0", api_keys=[KEY])
        resolve_auth(given)
        assert given.api_keys == [KEY] and given.generated == []
        insecure = ServeOptions(protocol=protocol, host="0.0.0.0", insecure=True)
        resolve_auth(insecure)
        assert insecure.api_keys == []
    realtime = ServeOptions(host="0.0.0.0")
    resolve_auth(realtime)
    assert realtime.api_keys == [] and check_exposure(realtime) is not None  # still refused

    twilio = ServeOptions(protocol="twilio")
    resolve_auth(twilio)
    assert twilio.stream_secret and twilio.generated == ["stream secret", "api key"]
    monkeypatch.setenv(SECRET_ENV, "from-env")
    signed = ServeOptions(protocol="twilio", public_url="wss://agent.example.com")
    resolve_auth(signed)
    assert signed.stream_secret == "from-env" and signed.api_keys == []  # signature instead
    vonage = ServeOptions(protocol="vonage", signature_secret="s")
    resolve_auth(vonage)
    assert vonage.api_keys == [] and vonage.generated == []
    telnyx = ServeOptions(protocol="telnyx", api_keys=[KEY])
    resolve_auth(telnyx)
    assert telnyx.api_keys == [KEY] and telnyx.generated == []


def test_cli_prints_the_generated_key_and_the_server_requires_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    async def serve_briefly(self: WebSocketAgentServer) -> None:
        url = f"ws://127.0.0.1:{self.port}/"
        seen["anonymous"] = (await upgrade(url))[0]
        seen["keys"] = self.api_keys
        await self.aclose()

    monkeypatch.setattr(WebSocketAgentServer, "serve_forever", serve_briefly)
    # pretend 127.0.0.1 is a public interface (no real non-loopback bind in tests)
    monkeypatch.setattr(security, "is_loopback_host", lambda host: False)
    result = CliRunner().invoke(cli_app, ["serve", "-p", "websocket", "--port", "0"],
                                env={"COLUMNS": "200", "VAN_SERVER_API_KEY": ""})  # fmt: skip
    assert result.exit_code == 0, result.output
    text = ANSI.sub("", result.output)
    match = re.search(r"generated for this run: (van_\S+)", text)
    assert match is not None, text
    assert seen["anonymous"] == 401 and seen["keys"].matches(match.group(1))
    assert "API key" in text


# ------------------------------------------------------------------ layered flags
def test_realtime_models_layer_flags_on_the_preset_and_config(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(
        "stt: mock\nllm: mock\ntts: mock\nagent: {instructions: From file.}\n", encoding="utf-8"
    )
    models = build_models(config=str(path), llm="{provider: mock, ttft: 0.25}")
    assert list(models) == ["agent"]  # one model: the file with the flag on top
    engine = models["agent"].engine
    assert models["agent"].instructions == "From file."
    assert engine.llm.ttft == 0.25  # type: ignore[attr-defined]
    named = build_models(["other=mock"], config=str(path), name="ignored-with-engines")
    assert sorted(named) == ["agent", "other"]
    assert list(build_models(config=str(path), name="n")) == ["n"]
    with pytest.raises(ConfigurationError):
        build_models(config=str(tmp_path / "missing.yaml"))


def test_served_websocket_agent_layers_flags(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text("engine: mock\nagent: {greeting: From file.}\n", encoding="utf-8")
    sources = SourceOptions(config=str(path), engines=["{provider: mock, response_delay: 0.1}"])
    served = build_served(sources, ServeOptions(protocol="websocket", port=0, warmup=False))
    assert served.pools[0].engine.response_delay == 0.1  # type: ignore[attr-defined]


# ---------------------------------------------------------------- session watchdog
class _FakeTransport:
    session_id = "fake"

    def __init__(self) -> None:
        self.idle = 0.0
        self.sent: list[dict[str, Any]] = []

    def idle_time(self) -> float:
        return self.idle

    def send_message_nowait(self, message: dict[str, Any]) -> None:
        self.sent.append(message)


async def test_session_watchdog_reports_expiry_and_idleness() -> None:
    transport = _FakeTransport()
    assert await _session_watchdog(transport, 0.05, None, what="test") == "session_expired"
    assert transport.sent[-1]["code"] == "session_expired" and transport.sent[-1]["fatal"]
    transport.idle = 10.0
    assert await _session_watchdog(transport, None, 5.0, what="test") == "session_idle"
    transport.idle = 0.0
    watchdog = asyncio.ensure_future(_session_watchdog(transport, None, 5.0, what="test"))
    await asyncio.sleep(0.05)
    assert not watchdog.done()  # an active peer is never idle
    watchdog.cancel()
    with pytest.raises(asyncio.CancelledError):
        await watchdog


# -------------------------------------------------------- WebRTC signalling (no aiortc)
async def test_webrtc_signalling_checks_the_origin_and_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voice_agent_next.transports import webrtc

    monkeypatch.setattr(webrtc, "_aiortc", lambda: None)  # no offer is answered here
    options = ServeOptions(protocol="webrtc", port=0, api_keys=[KEY], warmup=False,
                           allowed_origins=["https://app.example.com"],
                           max_session_duration=60.0, idle_timeout=30.0)  # fmt: skip
    served = build_served(SourceOptions(), options)
    server = served.server
    assert (server.max_session_duration, server.idle_timeout) == (60.0, 30.0)
    task = asyncio.create_task(run_served(served, handle_signals=False))
    await wait_for(lambda: server.port)
    await wait_for(lambda: served.state.ready)
    base = f"http://127.0.0.1:{server.port}"
    auth = {"Authorization": f"Bearer {KEY}"}
    async with httpx.AsyncClient(base_url=base, trust_env=False) as client:

        async def offer(headers: dict[str, str]) -> httpx.Response:
            return await client.post("/offer", content=b"not json", headers=headers)

        assert (await client.get("/health")).status_code == 200  # ops routes: open
        assert (await offer({**auth, "Origin": EVIL})).status_code == 403
        assert (await offer({"Origin": "https://app.example.com"})).status_code == 401
        assert (await offer({**auth, "Origin": "https://app.example.com"})).status_code == 400
        assert (await offer({**auth, "Origin": "http://localhost:5173"})).status_code == 400
        assert (await offer(auth)).status_code == 400  # native clients send no Origin
        # the server's own page (index_html): Origin == Host
        same = {**auth, "Origin": "http://agent.lan:8080", "Host": "agent.lan:8080"}
        assert (await offer(same)).status_code == 400
        other = {**auth, "Origin": "http://agent.lan:8080", "Host": "other.lan:8080"}
        assert (await offer(other)).status_code == 403
        assert (await client.get("/config")).status_code == 401
        assert (await client.get("/config", headers=auth)).status_code == 200
        preflight = await client.options("/offer", headers={"Origin": "https://app.example.com"})
        assert preflight.status_code == 204
        assert "Authorization" in preflight.headers["access-control-allow-headers"]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_same_origin_matching() -> None:
    from voice_agent_next.transports.webrtc import _same_origin

    assert _same_origin("http://a.lan:8080", "a.lan:8080")
    assert _same_origin("https://a.example", "a.example:443")
    assert _same_origin("http://a.example:80", "a.example")
    assert not _same_origin("http://a.lan:8080", "a.lan:8081")
    assert not _same_origin("null", "a.lan")
    assert not _same_origin(None, "a.lan") and not _same_origin("http://a.lan", None)


# ------------------------------------------------ telephony answer webhook (van serve)
def test_van_serve_twilio_serves_the_answer_webhook(monkeypatch: pytest.MonkeyPatch) -> None:
    from voice_agent_next.transports.telephony import TelephonyServer

    monkeypatch.delenv(SECRET_ENV, raising=False)
    seen: dict[str, Any] = {}

    async def serve_briefly(self: TelephonyServer) -> None:
        async with httpx.AsyncClient(trust_env=False) as client:
            base = f"http://127.0.0.1:{self.port}/answer?CallSid={CALL_SID}"
            seen["no_key"] = (await client.get(base)).status_code
            seen["answer"] = await client.get(f"{base}&key={KEY}", headers={"Host": "a.example"})
        seen["secret"] = self._secret
        await self.aclose()

    monkeypatch.setattr(TelephonyServer, "serve_forever", serve_briefly)
    args = ["serve", "-p", "twilio", "--port", "0", "--api-key", KEY]
    result = CliRunner().invoke(cli_app, args, env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    text = ANSI.sub("", result.output)
    assert seen["no_key"] == 403
    answer = seen["answer"]
    assert answer.status_code == 200 and answer.headers["content-type"].startswith("text/xml")
    assert 'url="wss://a.example/"' in answer.text
    token = stream_token(seen["secret"], CALL_SID)
    assert f'name="{TOKEN_PARAMETER}" value="{token}"' in answer.text
    assert "answer webhook (HTTP GET)" in text and "/answer?key=<api key>" in text
    assert "stream tokens use a secret generated for this run" in text


def test_van_serve_telephony_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SECRET_ENV, "s")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "")
    options = ServeOptions(protocol="twilio", port=0, public_url="https://agent.example.com")
    with pytest.raises(ConfigurationError, match="TWILIO_AUTH_TOKEN"):
        build_served(SourceOptions(), options)
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "t")
    served = build_served(SourceOptions(), options)
    assert served.server.verifier is not None
    assert served.server.answer_url() == "https://agent.example.com/answer"
    assert serve_cli._answer_url(options) == "https://agent.example.com/answer"
    vonage = build_served(
        SourceOptions(), ServeOptions(protocol="vonage", port=0, signature_secret="sig")
    ).server
    assert vonage.verifier is not None and vonage.verifier.provider == "vonage"
    # the signature secret only means something for Vonage
    plivo = build_served(
        SourceOptions(), ServeOptions(protocol="plivo", port=0, signature_secret="sig")
    ).server
    assert plivo.verifier is None and plivo.answer_keys
    result = CliRunner().invoke(cli_app, ["serve", "--help"], env={"COLUMNS": "200"})
    help_text = ANSI.sub("", result.output)
    for flag in ("--public-url", "--stream-secret", "--vonage-signature-secret", "--api-key"):
        assert flag in help_text, flag
