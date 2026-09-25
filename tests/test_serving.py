"""``van serve`` production serving: prewarm pools, admission, health/ready/metrics, drain,
protocol selection and worker processes (``voice_agent_next.server.serving``)."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from tests.test_websocket_transport import Client, wait_for
from voice_agent_next.cli import serve as serve_cli
from voice_agent_next.cli.main import app as cli_app
from voice_agent_next.cli.serve import (
    ServeOptions,
    SourceOptions,
    build_app_config,
    build_models,
    build_served,
)
from voice_agent_next.config import AppConfig
from voice_agent_next.engine import EngineConnection, EngineOptions
from voice_agent_next.errors import ConfigurationError
from voice_agent_next.metrics import TurnMetrics
from voice_agent_next.providers.mock import MockEngine
from voice_agent_next.server import serving
from voice_agent_next.server.ops import (
    JsonLogFormatter,
    ServeState,
    SessionLogFilter,
    session_context,
)
from voice_agent_next.server.pool import EnginePool, PooledEngine, options_match
from voice_agent_next.server.serving import Served, run_served

ANSI = re.compile(r"\x1b\[[0-9;]*m")
_CONNECT_KW: dict[str, Any] = {"proxy": None}


class SlowEngine(MockEngine):
    """A mock engine whose warm-up (model load) and connect (connection setup) take time."""

    instances: list[SlowEngine] = []

    def __init__(self, *, load: float = 0.0, setup: float = 0.0, **kw: Any) -> None:
        super().__init__(**kw)
        self.load, self.setup = load, setup
        self.warmups = 0
        self.connects = 0
        self.closed = False
        self._warm = False
        SlowEngine.instances.append(self)

    async def warmup(self) -> None:
        self.warmups += 1
        if not self._warm:  # like real engines: models load once
            await asyncio.sleep(self.load)
            self._warm = True

    async def connect(self, options: EngineOptions) -> EngineConnection:
        if not self._warm:
            await self.warmup()
        self.connects += 1
        await asyncio.sleep(self.setup)
        return await super().connect(options)

    async def aclose(self) -> None:
        self.closed = True


def http(port: int, path: str) -> httpx.Response:
    return httpx.get(f"http://127.0.0.1:{port}{path}", timeout=5, trust_env=False)


async def aget(port: int, path: str) -> httpx.Response:
    async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
        return await client.get(f"http://127.0.0.1:{port}{path}")


# ------------------------------------------------------------------------------- pool
async def test_shared_pool_prewarms_once_and_hands_over_connections() -> None:
    engine = SlowEngine(load=0.05)
    options = EngineOptions(instructions="Be brief.")
    pool = EnginePool(engine, size=2, options=options, max_idle=None, name="m")
    assert not pool.warm and not pool.ready
    await pool.start()
    assert pool.warm and pool.ready and pool.idle == 2
    assert engine.warmups == 1 and engine.connects == 2  # models once, two connections
    lease = await pool.lease()
    assert isinstance(lease, PooledEngine) and lease.prewarmed and lease.inner is engine
    assert (lease.model, lease.provider) == (engine.model, engine.provider)
    assert lease.llm is engine.llm  # engine attributes pass through
    conn = await lease.connect(EngineOptions(instructions="Be brief."))
    assert engine.connects == 2  # the prewarmed connection, no new one
    assert not conn.closed
    await wait_for(lambda: pool.idle == 2)  # refilled in the background
    assert engine.connects == 3
    second = await lease.connect(EngineOptions(instructions="Be brief."))  # a reconnect
    assert second is not conn and engine.connects == 4
    await lease.aclose()
    await lease.aclose()  # idempotent
    assert pool.leased == 0 and not engine.closed  # shared: stays open
    with pytest.raises(ConfigurationError):
        await lease.connect(options)
    stats = pool.stats()
    assert (stats.hits, stats.misses, stats.reused, stats.errors) == (1, 0, 1, 0)
    await pool.aclose()
    assert pool.idle == 0 and not engine.closed  # not owned
    with pytest.raises(ConfigurationError):
        await pool.lease()


async def test_prewarmed_connection_is_updated_or_replaced() -> None:
    engine = SlowEngine()
    pool = EnginePool(engine, size=1, options=EngineOptions(instructions="A"), max_idle=None)
    await pool.start()
    [prewarmed] = [i.conn for i in pool._idle]
    lease = await pool.lease()
    # other instructions: MockEngineConnection supports update(), so it is reused
    conn = await lease.connect(EngineOptions(instructions="B"))
    assert conn is prewarmed and conn.options.instructions == "B"
    await lease.aclose()
    await wait_for(lambda: pool.idle == 1)
    [prewarmed] = [i.conn for i in pool._idle]
    lease = await pool.lease()
    # another voice cannot be changed on an open connection: a fresh one, the old closes
    conn = await lease.connect(EngineOptions(instructions="A", voice="cedar"))
    assert conn is not prewarmed and prewarmed is not None and prewarmed.closed
    await lease.aclose()
    await pool.aclose()


def test_options_match() -> None:
    from voice_agent_next.chat import ChatContext
    from voice_agent_next.tools import function_tool

    @function_tool
    def weather(city: str) -> str:
        """Weather in a city."""
        return "sunny"

    base = EngineOptions(instructions="x", tools=[weather], voice="v")
    assert options_match(base, EngineOptions(instructions="x", tools=[weather], voice="v"))
    assert not options_match(base, EngineOptions(instructions="y", tools=[weather], voice="v"))
    assert not options_match(base, EngineOptions(instructions="x", voice="v"))
    history = ChatContext()
    history.add_message("user", "hi")
    assert not options_match(base, EngineOptions(instructions="x", tools=[weather], voice="v",
                                                 chat_ctx=history))  # fmt: skip


async def test_per_session_pool_builds_warms_and_closes_engines() -> None:
    built: list[SlowEngine] = []

    def factory() -> SlowEngine:
        engine = SlowEngine(load=0.01)
        built.append(engine)
        return engine

    pool = EnginePool(factory, size=2, options=EngineOptions(), max_idle=None)
    assert pool.per_session
    await pool.start()
    assert len(built) == 2 and all(e.warmups == 1 and e.connects == 1 for e in built)
    a, b = await pool.lease(), await pool.lease()
    assert a.inner is not b.inner and a.prewarmed and b.prewarmed
    c = await pool.lease()  # the pool is empty (refill runs in the background)
    assert not c.prewarmed or c.inner not in (a.inner, b.inner)
    await a.aclose()
    assert a.inner.closed  # per-session engines are used once
    await wait_for(lambda: pool.idle == 2)
    await b.aclose()
    await c.aclose()
    idle = [i.engine for i in pool._idle]
    await pool.aclose()
    assert all(e.closed for e in idle)  # idle prewarmed engines close with the pool


async def test_pool_recycles_stale_connections_and_survives_failures() -> None:
    engine = SlowEngine()
    pool = EnginePool(engine, size=1, options=EngineOptions(), max_idle=0.2)
    await pool.start()
    [first] = [i.conn for i in pool._idle]
    await wait_for(lambda: first is not None and first.closed, timeout=5)
    await wait_for(lambda: pool.idle == 1)
    assert pool._idle[0].conn is not first
    await pool.aclose()

    calls = 0

    def flaky() -> SlowEngine:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("GPU busy")
        return SlowEngine()

    pool = EnginePool(flaky, size=1, max_idle=None)
    await pool.start()  # the failure is counted, not raised
    assert pool.stats().errors == 1
    await wait_for(lambda: pool.idle == 1, timeout=10)  # retried by the maintainer
    assert pool.ready
    await pool.aclose()


async def test_warm_start_is_faster_than_cold_start() -> None:
    """Time to a connected engine for a new call: prewarmed vs cold (models + setup)."""

    async def first_connect(prewarm: int) -> float:
        engine = SlowEngine(load=0.4, setup=0.2)
        pool = EnginePool(engine, size=prewarm, options=EngineOptions(), warmup=bool(prewarm),
                          max_idle=None)  # fmt: skip
        pool._warm.set() if not prewarm else await pool.start()
        started = time.perf_counter()
        lease = await pool.lease()
        await lease.connect(EngineOptions())
        elapsed = time.perf_counter() - started
        await lease.aclose()
        await pool.aclose()
        return elapsed

    cold, warm = await first_connect(0), await first_connect(1)
    assert cold >= 0.55  # model load + connection setup
    assert warm < 0.15 and warm < cold / 3


# -------------------------------------------------------------------------- ops state
def test_readiness_transitions() -> None:
    pool = EnginePool(SlowEngine(), size=1)
    state = ServeState("websocket", pools=[pool], max_sessions=2)
    assert state.readiness_problems() == ["starting", "pool default: warming up"]
    state.started = True
    pool._warm.set()
    assert "pool default: no prewarmed engine available" in state.readiness_problems()
    pool._idle.append(object())  # type: ignore[arg-type]
    assert state.ready and state.refusal() is None
    state.session_started()
    state.session_started()
    assert state.readiness_problems() == ["session limit reached"]
    refusal = state.refusal()
    assert refusal is not None and refusal[0] == 503 and refusal[1] == "busy"
    state.session_ended()
    assert state.ready
    state.draining = True
    assert not state.ready and state.refusal() is not None
    assert state.refusal()[1] == "draining"  # type: ignore[index]
    assert state.health()["status"] == "draining"
    status, _, body = state.http("/ready")  # type: ignore[misc]
    assert status == 503 and json.loads(body)["reasons"] == ["draining"]
    assert state.http("/health")[0] == 200  # type: ignore[index]  # liveness while draining
    assert state.http("/v1/realtime") is None


_LABEL = r'[a-z_]+="(?:[^"\\]|\\.)*"'
_SAMPLE = re.compile(rf"^([a-z_:][a-z0-9_:]*)(\{{({_LABEL}(?:,{_LABEL})*)?\}})? (\S+)$")


def parse_prometheus(text: str) -> dict[str, dict[str, float]]:
    """Validate the text exposition format; ``{name: {labels: value}}``."""
    assert text.endswith("\n")
    types: dict[str, str] = {}
    samples: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        if line.startswith("# HELP "):
            continue
        if line.startswith("# TYPE "):
            _, _, name, kind = line.split(" ")
            assert kind in ("gauge", "counter", "histogram")
            assert name not in types, f"duplicate TYPE for {name}"
            types[name] = kind
            continue
        match = _SAMPLE.match(line)
        assert match, f"bad sample line: {line!r}"
        name, labels, value = match.group(1), match.group(3) or "", match.group(4)
        family = re.sub(r"_(bucket|sum|count)$", "", name)
        assert name in types or family in types, f"sample without TYPE: {name}"
        if types.get(name) == "counter" or types.get(family) == "counter":
            assert name.endswith("_total")
        samples.setdefault(name, {})[labels] = float(value)
    return samples


def test_metrics_output_format() -> None:
    engine = SlowEngine()
    pool = EnginePool(engine, size=2, name='say "hi"')
    state = ServeState("openai-realtime", pools=[pool], max_sessions=4, worker=3)
    state.started = True
    state.session_started()
    state.reject("busy")
    state.reject("draining")
    state.session_ended(error=True)
    for v2v in (0.25, 0.45, 0.9, 12.0):
        state.observe_turn(TurnMetrics(turn_id="t", voice_to_voice=v2v))
    state.observe_turn(TurnMetrics(turn_id="t"))  # no audio: counted as a turn only
    state.observe_connect(0.002, True)
    state.observe_connect(0.8, False)
    samples = parse_prometheus(state.metrics())
    base = 'protocol="openai-realtime",worker="3"'
    assert samples["van_sessions_total"][base] == 1
    assert samples["van_sessions_active"][base] == 0
    assert samples["van_sessions_max"][base] == 4
    assert samples["van_session_errors_total"][base] == 1
    assert samples["van_sessions_rejected_total"][f'{base},reason="busy"'] == 1
    assert samples["van_sessions_rejected_total"][f'{base},reason="draining"'] == 1
    assert samples["van_pool_size"][f'{base},pool="say \\"hi\\""'] == 2
    assert samples["van_turns_total"][base] == 5
    buckets = samples["van_voice_to_voice_seconds_bucket"]
    assert buckets[f'{base},le="0.3"'] == 1 and buckets[f'{base},le="1"'] == 3
    assert buckets[f'{base},le="+Inf"'] == 4 == samples["van_voice_to_voice_seconds_count"][base]
    values = list(buckets.values())
    assert values == sorted(values)  # cumulative
    assert samples["van_voice_to_voice_seconds_sum"][base] == pytest.approx(13.6)
    connect = samples["van_engine_connect_seconds_count"]
    assert connect[f'{base},prewarmed="true"'] == 1 and connect[f'{base},prewarmed="false"'] == 1


def test_session_logs_carry_the_session_id() -> None:
    record = logging.makeLogRecord({"name": "voice_agent_next", "msg": "hello %s", "args": ("x",),
                                    "levelname": "INFO", "model": "mock"})  # fmt: skip
    with session_context("ws_123"):
        SessionLogFilter(worker=2).filter(record)
    data = json.loads(JsonLogFormatter().format(record))
    assert data["msg"] == "hello x" and data["session_id"] == "ws_123"
    assert data["worker"] == 2 and data["model"] == "mock" and data["level"] == "info"
    outside = logging.makeLogRecord({"msg": "m"})
    SessionLogFilter().filter(outside)
    assert "session_id" not in json.loads(JsonLogFormatter().format(outside))


# ------------------------------------------------------------------ protocol selection
def options(protocol: str, **kw: Any) -> ServeOptions:
    return ServeOptions(protocol=protocol, port=0, **kw)


async def test_openai_realtime_protocol_prewarm_limit_and_drain() -> None:
    engine = SlowEngine(setup=0.05)
    served = serving.build_realtime_served(
        {"local": engine}, port=0, prewarm=1, max_sessions=1, max_idle=None
    )
    task = asyncio.create_task(run_served(served, handle_signals=False, drain_timeout=10))
    port = await wait_port(served)
    await wait_for(lambda: served.state.ready)
    assert (await aget(port, "/health")).json()["models"] == ["local"]
    assert (await aget(port, "/ready")).status_code == 200
    url = f"ws://127.0.0.1:{port}/v1/realtime"
    async with connect(url, **_CONNECT_KW) as ws:
        created = json.loads(await ws.recv())
        assert created["type"] == "session.created"
        # the first audio opens the engine connection: the prewarmed one is handed over
        silence = base64.b64encode(bytes(4800)).decode()
        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": silence}))
        await wait_for(lambda: served.pools[0].stats().reused == 1)
        assert engine.connects == 2  # the prewarmed one, then the pool's refill
        # the session limit: a second client is refused before the upgrade
        with pytest.raises(InvalidStatus) as refused:
            async with connect(url, **_CONNECT_KW):
                pass
        assert refused.value.response.status_code == 503
        assert refused.value.response.headers["Retry-After"] == "1"
        assert json.loads(refused.value.response.body)["error"]["code"] == "session_limit_reached"
        assert (await aget(port, "/ready")).status_code == 503
        metrics = parse_prometheus((await aget(port, "/metrics")).text)
        assert metrics["van_sessions_active"]['protocol="openai-realtime"'] == 1
        # drain: new sessions are refused while the live one may finish
        drain = asyncio.create_task(served.drain(10))
        await wait_for(lambda: served.state.draining)
        ready = await aget(port, "/ready")
        assert ready.status_code == 503 and "draining" in ready.json()["reasons"]
        with pytest.raises(InvalidStatus) as refused:
            async with connect(url, **_CONNECT_KW):
                pass
        assert json.loads(refused.value.response.body)["error"]["code"] == "server_draining"
        assert not drain.done()
    await asyncio.wait_for(drain, 5)  # the client left: the drain ends early
    served.server._closed.set()  # what serve_forever waits for
    await asyncio.wait_for(task, 10)
    metrics = parse_prometheus(served.state.metrics())
    assert metrics["van_sessions_total"]['protocol="openai-realtime"'] == 1
    assert metrics["van_sessions_rejected_total"]['protocol="openai-realtime",reason="busy"'] == 1


async def wait_port(served: Served) -> int:
    http = getattr(served.server, "_http", None)
    if http is not None:
        await wait_for(lambda: http.server is not None)
    else:
        await wait_for(lambda: served.server._server is not None)
    return int(served.server.port)


async def test_websocket_protocol_runs_agent_sessions_from_the_pool() -> None:
    served = build_served(SourceOptions(), options("websocket", prewarm=1, max_sessions=1))
    task = asyncio.create_task(run_served(served, handle_signals=False))
    port = await wait_port(served)
    await wait_for(lambda: served.state.ready)
    pool = served.pools[0]
    async with Client(f"ws://127.0.0.1:{port}/") as client:
        await client.handshake()
        await wait_for(lambda: served.state.active == 1)
        assert pool.stats().hits == 1
        await client.speak()
        await wait_for(lambda: served.state.turns_total >= 1, timeout=10)
        assert pool.stats().reused == 1  # the agent's options matched the prewarmed ones
        assert served.state.v2v.count >= 1
        # busy: the handshake is refused with 503 before the upgrade
        with pytest.raises(InvalidStatus) as refused:
            async with connect(f"ws://127.0.0.1:{port}/", **_CONNECT_KW):
                pass
        assert refused.value.response.status_code == 503
        assert json.loads(refused.value.response.body)["code"] == "server_busy"
    await wait_for(lambda: served.state.active == 0)
    assert pool.leased == 0
    metrics = parse_prometheus((await aget(port, "/metrics")).text)
    assert metrics["van_sessions_total"]['protocol="websocket"'] == 1
    assert metrics["van_voice_to_voice_seconds_count"]['protocol="websocket"'] >= 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert served.state.draining


async def test_drain_timeout_closes_live_sessions() -> None:
    served = build_served(SourceOptions(), options("websocket"))
    stop = asyncio.Event()
    task = asyncio.create_task(run_served(served, handle_signals=False, drain_timeout=0.3))
    port = await wait_port(served)
    await wait_for(lambda: served.state.started)
    async with Client(f"ws://127.0.0.1:{port}/") as client:
        await client.handshake()
        await wait_for(lambda: served.state.active == 1)
        started = time.perf_counter()
        await served.drain(0.3, stop)
        assert time.perf_counter() - started >= 0.25 and served.state.active == 1
        await served.aclose()  # after the timeout: live sessions are closed
        assert await client.wait_closed() == 1001
    await wait_for(lambda: served.state.active == 0)
    await asyncio.wait_for(task, 10)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
async def test_sigterm_drains_then_exits() -> None:
    served = build_served(SourceOptions(), options("websocket"))
    task = asyncio.create_task(run_served(served, drain_timeout=10))
    port = await wait_port(served)
    await wait_for(lambda: served.state.started)
    async with Client(f"ws://127.0.0.1:{port}/") as client:
        await client.handshake()
        await wait_for(lambda: served.state.active == 1)
        os.kill(os.getpid(), signal.SIGTERM)
        await wait_for(lambda: served.state.draining)
        await asyncio.sleep(0.2)
        assert not task.done() and served.state.active == 1  # the call goes on
    await asyncio.wait_for(task, 10)  # it ended: the process stops


async def test_telephony_protocols_serve_ops_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    from voice_agent_next.transports.telephony import SECRET_ENV, TelephonyServer

    monkeypatch.setenv(SECRET_ENV, "stream-secret")  # media streams are authenticated

    for provider in serving.TELEPHONY_PROTOCOLS:
        served = build_served(SourceOptions(), options(provider, max_sessions=3))
        assert isinstance(served.server, TelephonyServer) and served.server.provider == provider
    served = build_served(SourceOptions(), options("twilio"))
    task = asyncio.create_task(run_served(served, handle_signals=False))
    port = await wait_port(served)
    await wait_for(lambda: served.state.ready)
    assert (await aget(port, "/health")).json()["protocol"] == "twilio"
    assert (await aget(port, "/ready")).json()["ready"] is True
    served.state.draining = True
    with pytest.raises(InvalidStatus) as refused:
        async with connect(f"ws://127.0.0.1:{port}/", **_CONNECT_KW):
            pass
    assert refused.value.response.status_code == 503
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_webrtc_protocol_serves_ops_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    from voice_agent_next.transports import webrtc

    monkeypatch.setattr(webrtc, "_aiortc", lambda: None)  # no offer is answered here
    served = build_served(SourceOptions(), options("webrtc", max_sessions=2))
    assert isinstance(served.server, webrtc.WebRTCAgentServer)
    task = asyncio.create_task(run_served(served, handle_signals=False))
    port = await wait_port(served)
    await wait_for(lambda: served.state.ready)
    assert (await aget(port, "/health")).json()["protocol"] == "webrtc"
    metrics = await aget(port, "/metrics")
    assert metrics.headers["content-type"].startswith("text/plain; version=0.0.4")
    parse_prometheus(metrics.text)
    assert (await aget(port, "/config")).status_code == 200  # the server's own routes
    served.state.draining = True
    async with httpx.AsyncClient(trust_env=False) as client:
        offer = await client.post(f"http://127.0.0.1:{port}/offer",
                                  json={"type": "offer", "sdp": "v=0\r\nm=audio"})  # fmt: skip
    assert offer.status_code == 503 and "shutting down" in offer.json()["error"]
    assert served.state.rejected == {"draining": 1}
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_app_config_from_sources(tmp_path: Path) -> None:
    assert build_app_config(SourceOptions()).engine == "mock"
    cfg = build_app_config(SourceOptions(engines=["{provider: mock, response_delay: 0.1}"],
                                         instructions="Hi.", voice="v"))  # fmt: skip
    assert cfg.engine == {"provider": "mock", "response_delay": 0.1}
    assert (cfg.agent.instructions, cfg.agent.voice) == ("Hi.", "v")
    cascade = build_app_config(SourceOptions(stt="mock", llm="mock", tts="mock", vad="energy"))
    assert cascade.is_cascade() and cascade.vad == "energy"
    path = tmp_path / "agent.yaml"
    path.write_text("engine: mock\nagent: {instructions: From file.}\n", encoding="utf-8")
    assert build_app_config(SourceOptions(config=str(path))).agent.instructions == "From file."
    assert build_app_config(SourceOptions(engines=[str(path)])).engine == "mock"
    layered = build_app_config(SourceOptions(config=str(path), llm="mock", tts="mock"))
    assert layered.is_cascade() and layered.llm == "mock"  # file < flags (#156)
    assert layered.agent.instructions == "From file."
    for bad in (SourceOptions(engines=["mock", "mock"]), SourceOptions(engines=["mock"], llm="x"),
                SourceOptions(engines=["a=mock"]), SourceOptions(tts="mock"),  # (TTS optional: omni LLMs)
                SourceOptions(vad="energy"), SourceOptions(config=str(tmp_path / "no.yaml"))):  # fmt: skip
        with pytest.raises(ConfigurationError):
            build_app_config(bad)
    with pytest.raises(ConfigurationError, match="unknown protocol"):
        build_served(SourceOptions(), options("sip"))
    # --api-key applies to every protocol now (#156)
    keyed = build_served(SourceOptions(), options("websocket", api_keys=["k"])).server
    assert keyed.api_keys.matches("k") and not keyed.api_keys.matches("x")
    with pytest.raises(ConfigurationError, match="public-url"):
        build_served(SourceOptions(), options("websocket", public_url="wss://a.example"))
    assert serve_cli.PROTOCOLS == serving.PROTOCOLS


async def test_engine_per_session_realtime_models() -> None:
    models = build_models(["fast={provider: mock, response_delay: 0.1}"], per_session=True)
    served = serving.build_realtime_served(models, port=0, prewarm=1, max_idle=None)
    pool = served.pools[0]
    assert pool.per_session
    await pool.start()
    first = await pool.lease()
    second = await pool.lease()
    assert isinstance(first.inner, MockEngine) and first.inner is not second.inner
    assert first.inner.response_delay == 0.1
    await first.aclose()
    await second.aclose()
    await served.aclose()


def test_served_config_is_an_app_config() -> None:
    served = build_served(SourceOptions(engines=["mock"]), options("websocket", prewarm=2))
    assert served.pools[0].size == 2 and not served.pools[0].per_session
    assert isinstance(build_app_config(SourceOptions()), AppConfig)


# ------------------------------------------------------------------------------- CLI
def test_van_serve_help_lists_the_production_options() -> None:
    result = CliRunner().invoke(cli_app, ["serve", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    text = ANSI.sub("", result.output)
    for flag in ("--protocol", "--preset", "--config", "--prewarm", "--max-sessions",
                 "--workers", "--drain-timeout", "--log-format", "--engine-per-session"):  # fmt: skip
        assert flag in text, flag


def test_van_serve_websocket_from_the_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    from voice_agent_next.transports.websocket import WebSocketAgentServer

    seen: dict[str, Any] = {}

    async def serve_briefly(self: WebSocketAgentServer) -> None:
        async with httpx.AsyncClient(trust_env=False) as client:
            seen["ready"] = (await client.get(f"http://127.0.0.1:{self.port}/ready")).json()
        await self.aclose()

    monkeypatch.setattr(WebSocketAgentServer, "serve_forever", serve_briefly)
    args = ["serve", "-p", "websocket", "--port", "0", "--prewarm", "1", "--max-sessions", "2"]
    result = CliRunner().invoke(cli_app, args)
    assert result.exit_code == 0, result.output
    assert seen["ready"]["ready"] is True and seen["ready"]["max_sessions"] == 2
    assert seen["ready"]["pools"]["agent"]["size"] == 1
    assert "websocket" in ANSI.sub("", result.output)
    bad = CliRunner().invoke(cli_app, ["serve", "-p", "websocket", "-e", "a", "-e", "b"])
    assert bad.exit_code == 2 and "one agent" in ANSI.sub("", bad.output)


def test_workers_fall_back_to_one_process_without_reuse_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voice_agent_next.transports.websocket import WebSocketAgentServer

    async def stop(self: WebSocketAgentServer) -> None:
        await self.aclose()

    monkeypatch.setattr(serving, "reuse_port_supported", lambda: False)
    monkeypatch.setattr(WebSocketAgentServer, "serve_forever", stop)
    monkeypatch.setattr(serving, "run_workers", lambda *a, **k: pytest.fail("no workers"))
    result = CliRunner().invoke(cli_app, ["serve", "-p", "websocket", "--port", "0",
                                          "--workers", "3"])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "running one process" in ANSI.sub("", result.output)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="SO_REUSEPORT balancing: Linux")
def test_worker_processes_share_the_port_and_drain_on_sigterm() -> None:
    port = serving.free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "voice_agent_next.cli.main", "serve", "-p", "websocket",
         "--port", str(port), "--workers", "2", "--drain-timeout", "5"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip
    try:
        pids: set[int] = set()
        deadline = time.monotonic() + 60
        while len(pids) < 2 and time.monotonic() < deadline:
            try:
                pids.add(http(port, "/health").json()["pid"])
            except httpx.HTTPError:
                time.sleep(0.2)
        assert len(pids) == 2, "both workers answer on the shared port"
        assert proc.pid not in pids
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        output = proc.communicate()[0]
    assert "2 workers" in output
