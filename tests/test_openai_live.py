"""GPT-Live engine (``providers/openai/live.py``) against the fake Live server.

Everything runs offline: :class:`FakeLiveServer` speaks the Live WebSocket protocol on
``127.0.0.1``. The real-API test at the end is marked ``integration`` (``OPENAI_API_KEY``).
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable
from typing import Any

import numpy as np
import pytest

from voice_agent_next import Agent, AgentSession, AgentState, ChatMessage, function_tool
from voice_agent_next.audio import AudioFrame
from voice_agent_next.chat import ChatContext, FunctionCall, FunctionCallOutput
from voice_agent_next.engine import EngineOptions
from voice_agent_next.engines.rotation import RotationPolicy
from voice_agent_next.errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderError,
)
from voice_agent_next.events import (
    EngineErrorEvent,
    EngineEvent,
    EngineStatus,
    InputCommitted,
    InputSpeechStarted,
    InputTranscript,
    ResponseAudio,
    ResponseDone,
    ResponseStarted,
    ResponseText,
    ResponseToolCall,
)
from voice_agent_next.metrics import (
    EngineMetrics,
    LLMMetrics,
    RotationMetrics,
    TurnMetrics,
    UsageSummary,
)
from voice_agent_next.providers.mock import synth_speech
from voice_agent_next.providers.openai.live import (
    LiveSessionConnection,
    OpenAILiveConnection,
    OpenAILiveEngine,
    OpenAILiveSessionEngine,
    live_url,
    seed_items,
)
from voice_agent_next.registry import create
from voice_agent_next.testing.openai_live import FakeDelegation, FakeLiveServer, FakeTurn
from voice_agent_next.transports import LoopbackTransport

KEY = "sk-test"


# ----------------------------------------------------------------------------- helpers
@pytest.fixture
async def fake() -> AsyncIterator[Callable[..., Any]]:
    """Factory starting fake servers that are closed (and checked) after the test."""
    servers: list[FakeLiveServer] = []

    async def make(**kw: Any) -> FakeLiveServer:
        server = FakeLiveServer(**kw)
        await server.start()
        servers.append(server)
        return server

    yield make
    for server in servers:
        await server.aclose()
        assert server.errors == [], f"protocol violations: {server.errors}"


def engine_for(server: FakeLiveServer, **kw: Any) -> OpenAILiveEngine:
    return OpenAILiveEngine(api_key=KEY, base_url=server.url, **kw)


class Events:
    """Collects the events of an engine connection in the background."""

    def __init__(self, conn: Any) -> None:
        self.items: list[EngineEvent] = []
        self._task = asyncio.create_task(self._pump(conn))

    async def _pump(self, conn: Any) -> None:
        async for ev in conn.events():
            self.items.append(ev)

    def of(self, kind: type[Any]) -> list[Any]:
        return [e for e in self.items if isinstance(e, kind)]

    def kinds(self) -> list[str]:
        return [e.type for e in self.items if not isinstance(e, ResponseAudio)]

    def text(self) -> str:
        return "".join(e.delta for e in self.of(ResponseText))

    async def close(self) -> None:
        await asyncio.wait_for(self._task, 10)


async def wait_for(cond: Callable[[], bool], timeout: float = 15.0) -> None:
    async with asyncio.timeout(timeout):
        while not cond():
            await asyncio.sleep(0.01)


async def speak(conn: Any, seconds: float) -> None:
    """Real-time user speech (16 kHz, resampled by the engine), in 20 ms frames."""
    for i in range(round(seconds / 0.02)):
        await conn.send_audio(synth_speech(0.02, 16_000, offset=i * 320))
        await asyncio.sleep(0.02)


async def silence(conn: Any, seconds: float) -> None:
    for _ in range(round(seconds / 0.02)):
        await conn.send_audio(AudioFrame.silence(0.02, 16_000))
        await asyncio.sleep(0.02)


# ------------------------------------------------------------------------ configuration
def test_registry_and_url() -> None:
    engine = create("engine", "openai-live/gpt-live-1", api_key=KEY)
    assert isinstance(engine, OpenAILiveEngine)
    assert engine.model == "gpt-live-1"
    caps = engine.capabilities
    assert caps.full_duplex and caps.tool_mode == "delegation" and caps.server_turn_detection
    assert isinstance(create("engine", "gpt-live", api_key=KEY), OpenAILiveEngine)
    assert engine.session_engine.url == "wss://api.openai.com/v1/live/sessions"
    assert live_url("https://example.com/v1/") == "wss://example.com/v1/live/sessions"
    assert live_url("ws://h:1/v1/live/sessions") == "ws://h:1/v1/live/sessions"


def test_configuration_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="API key"):
        OpenAILiveEngine()
    with pytest.raises(ConfigurationError, match="sample_rate"):
        OpenAILiveEngine(api_key=KEY, sample_rate=8000)
    with pytest.raises(ConfigurationError, match="delegation"):
        OpenAILiveEngine(api_key=KEY, delegation="other")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="base URL"):
        OpenAILiveEngine(api_key=KEY, base_url="ftp://x")


def test_seed_items_convert_the_text_history() -> None:
    ctx = ChatContext()
    ctx.add_message("system", "Be brief.")
    ctx.add_message("user", "Weather in Paris?")
    ctx.append(FunctionCall(name="get_weather", arguments="{}", call_id="c1"))
    ctx.append(FunctionCallOutput(call_id="c1", name="get_weather", output="sunny"))
    ctx.add_message("assistant", "It is sunny.")
    ctx.add_message("user", "   ")
    items = seed_items(ctx)
    assert [(i["role"], i["content"][0]["type"]) for i in items] == [
        ("developer", "input_text"),
        ("user", "input_text"),
        ("developer", "input_text"),
        ("assistant", "output_text"),
    ]
    assert items[2]["content"][0]["text"] == "Result of the get_weather call: sunny"
    many = ChatContext()
    for i in range(300):
        many.add_message("user", f"message {i}")
    assert len(seed_items(many)) == 128


async def test_session_start_payload(fake: Callable[..., Any]) -> None:
    server = await fake()

    @function_tool
    async def get_weather(city: str) -> str:
        """Get the weather."""
        return "sunny"

    history = ChatContext()
    history.add_message("user", "Hi")
    history.add_message("assistant", "Hello!")
    engine = engine_for(
        server,
        voice="cedar",
        sample_rate=16_000,
        responses_model="gpt-5.6-luna",
        responses_instructions="Use the tools.",
        web_search=True,
        responses={"reasoning": {"effort": "low"}, "service_tier": "priority"},
        store=True,
        headers={"OpenAI-Safety-Identifier": "user-1"},
    )
    conn = await engine.connect(
        EngineOptions(instructions="Be concise.", tools=[get_weather], chat_ctx=history)
    )
    assert isinstance(conn, OpenAILiveConnection)
    assert conn.session_id == "live_001"
    config = server.session.config
    assert config["model"] == "gpt-live-1"
    assert config["instructions"] == "Be concise."
    assert config["audio"] == {
        "format": {"type": "audio/pcm", "rate": 16_000},
        "output": {"voice": "cedar"},
    }
    assert config["store"] is True
    assert [i["role"] for i in config["input"]] == ["user", "assistant"]
    backend = config["delegation"]["responses"]
    assert config["delegation"]["type"] == "responses"
    assert backend["model"] == "gpt-5.6-luna"
    assert backend["instructions"] == "Use the tools."
    assert backend["reasoning"] == {"effort": "low"} and backend["service_tier"] == "priority"
    assert backend["tools"][0]["name"] == "get_weather"
    assert backend["tools"][0]["type"] == "function"
    assert backend["tools"][1] == {"type": "web_search"}
    assert server.session.headers["OpenAI-Safety-Identifier"] == "user-1"
    assert conn.input_sample_rate == conn.output_sample_rate == 16_000
    await conn.aclose()
    assert server.session.of("session.close")  # graceful close
    assert server.session.close_reason == "close_requested"


async def test_client_delegation_payload_and_tool_update(fake: Callable[..., Any]) -> None:
    server = await fake()
    engine = engine_for(server, delegation="client")
    conn = await engine.connect(EngineOptions())
    assert server.session.config["delegation"] == {"type": "client"}
    assert "instructions" not in server.session.config
    await conn.aclose()
    # Responses delegation: new tools update the backend (session.update)
    server2 = await fake()
    conn = await engine_for(server2).connect(EngineOptions())

    @function_tool
    async def lookup(order: str) -> str:
        """Look up an order."""
        return "shipped"

    await conn.update(tools=[lookup], instructions="Be polite.")
    await wait_for(lambda: bool(server2.session.of("session.update")))
    update = server2.session.of("session.update")[0]["session"]["delegation"]
    assert update["responses"]["tools"][0]["name"] == "lookup"
    appended = server2.session.of("session.instructions.append")
    assert appended and "Be polite." in appended[0]["content"]
    assert appended[0]["delegation_id"] is None
    await conn.aclose()


# ------------------------------------------------------------------------------ errors
async def test_handshake_rejections_map_to_library_errors(fake: Callable[..., Any]) -> None:
    server = await fake()
    with pytest.raises(AuthenticationError):
        await OpenAILiveEngine(api_key="wrong", base_url=server.url).connect(EngineOptions())
    busy = await fake(reject_status=429)
    with pytest.raises(ProviderError, match="429"):
        await engine_for(busy).connect(EngineOptions())
    server.sessions.clear()  # the rejected handshake never reached the handler


async def test_session_start_error_is_raised(fake: Callable[..., Any]) -> None:
    server = await fake(
        start_error={
            "type": "invalid_request_error",
            "code": "invalid_value",
            "message": "Unknown voice.",
            "param": "session.audio.output.voice",
        }
    )
    with pytest.raises(ProviderError, match="Unknown voice") as info:
        await engine_for(server, voice="nobody").connect(EngineOptions())
    assert "session.start" in str(info.value)


async def test_connection_refused_is_a_connection_error() -> None:
    engine = OpenAILiveEngine(api_key=KEY, base_url="ws://127.0.0.1:9/v1", connect_timeout=5)
    with pytest.raises(ProviderConnectionError):
        await engine.connect(EngineOptions())


async def test_runtime_errors_are_recoverable_events(fake: Callable[..., Any]) -> None:
    server = await fake()
    conn = await engine_for(server).connect(EngineOptions())
    events = Events(conn)
    await server.send_error("Audio cut by moderation.", code="moderation")
    await wait_for(lambda: bool(events.of(EngineErrorEvent)))
    err = events.of(EngineErrorEvent)[0]
    assert err.recoverable and "moderation" in str(err.error)
    # a rejected command names the request it answers
    await conn.append_instructions("x", delegation_id="del_unknown")  # type: ignore[union-attr]
    await wait_for(lambda: len(events.of(EngineErrorEvent)) == 2)
    assert "session.instructions.append" in str(events.of(EngineErrorEvent)[1].error)
    server.errors.clear()  # that append was a deliberate protocol violation
    assert not conn.closed
    await conn.aclose()
    await events.close()


async def test_safety_close_ends_the_conversation(fake: Callable[..., Any]) -> None:
    server = await fake()
    conn = await engine_for(server).connect(EngineOptions())
    events = Events(conn)
    await server.close_session("content")
    await events.close()  # the connection closes by itself
    assert conn.closed
    fatal = [e for e in events.of(EngineErrorEvent) if not e.recoverable]
    assert len(fatal) == 1 and "safety" in str(fatal[0].error)
    assert len(server.sessions) == 1  # no reconnect


# --------------------------------------------------------------------- full duplex
async def test_greeting_reply_and_transcripts(fake: Callable[..., Any]) -> None:
    server = await fake(turns=[FakeTurn("Nice to meet you", user="I am Ada")])
    conn = await engine_for(server).connect(EngineOptions())
    events = Events(conn)
    await conn.say("Hello there")
    await silence(conn, 1.0)
    await wait_for(lambda: bool(events.of(ResponseDone)))
    await speak(conn, 0.6)
    await silence(conn, 1.6)
    await wait_for(lambda: len(events.of(ResponseDone)) == 2)
    await conn.aclose()
    await events.close()

    say = server.sessions[0].of("session.instructions.append")[0]
    assert 'verbatim and in full, then pause and listen: "Hello there"' in say["content"]
    kinds = events.kinds()
    assert kinds.count("response_started") == 2
    first_reply = kinds.index("response_started", kinds.index("response_done"))
    assert "input_speech_started" in kinds[:first_reply]
    assert kinds[first_reply - 2 : first_reply] == ["input_committed", "input_transcript"]
    final = [e for e in events.of(InputTranscript) if e.is_final]
    assert final and final[-1].text == "I am Ada"
    assert final[-1].item_id == events.of(InputCommitted)[0].item_id
    texts: dict[str, str] = {}
    for ev in events.of(ResponseText):
        texts[ev.response_id] = texts.get(ev.response_id, "") + ev.delta
    assert list(texts.values()) == ["Hello there", "Nice to meet you"]
    assert all(e.status == "completed" for e in events.of(ResponseDone))
    audio = np.concatenate([e.frame.to_float32() for e in events.of(ResponseAudio)])
    assert np.sqrt(np.mean(audio**2)) > 0.05  # speech, not the quiet noise between replies
    assert conn.usage_seconds >= 1.0  # type: ignore[attr-defined]


async def test_billed_seconds_are_reported_per_response(fake: Callable[..., Any]) -> None:
    """Each response's ``EngineMetrics.billed_seconds`` is the billed voice time since the
    previous response (#157): the per-response values add up to the session's usage."""
    server = await fake(turns=[FakeTurn("Nice to meet you", user="I am Ada")])
    engine = engine_for(server)
    seen: list[tuple[EngineMetrics, float]] = []
    conn = await engine.connect(EngineOptions())
    engine.on(
        "metrics",
        lambda m: seen.append((m, conn.usage_seconds)) if isinstance(m, EngineMetrics) else None,  # type: ignore[attr-defined]
    )
    events = Events(conn)
    await conn.say("Hello there")
    await silence(conn, 1.0)
    await wait_for(lambda: bool(events.of(ResponseDone)))
    await speak(conn, 0.6)
    await silence(conn, 1.6)
    await wait_for(lambda: len(events.of(ResponseDone)) == 2)
    await conn.aclose()
    await events.close()
    (first, usage1), (second, usage2) = seen[:2]
    # usage arrives once per second of session time: a quick first response may see none
    # yet (its time is then attributed to the next one), but the total is never lost
    assert first.billed_seconds + second.billed_seconds > 0
    assert first.billed_seconds == pytest.approx(usage1)
    assert second.billed_seconds == pytest.approx(usage2 - usage1)
    summary = UsageSummary()
    for m, _ in seen:
        summary.add(m)
    assert summary.engine_billed_seconds == pytest.approx(seen[-1][1])


async def test_cancel_mutes_the_agent_until_its_next_pause(fake: Callable[..., Any]) -> None:
    server = await fake()
    conn = await engine_for(server).connect(EngineOptions())
    events = Events(conn)
    await conn.say("one two three four five six seven eight nine ten")
    await wait_for(lambda: bool(events.of(ResponseAudio)))
    await conn.interrupt()
    await silence(conn, 3.0)
    await conn.aclose()
    await events.close()
    done = events.of(ResponseDone)
    assert [d.status for d in done] == ["cancelled"]
    assert len(events.of(ResponseStarted)) == 1  # the rest of the utterance was muted
    assert server.utterances[0].completed  # the model itself kept talking


async def test_user_talking_over_the_agent_is_left_to_the_model(
    fake: Callable[..., Any],
) -> None:
    server = await fake(turns=["one two three four five six seven eight nine ten", "Okay."])
    conn = await engine_for(server).connect(EngineOptions())
    events = Events(conn)
    await speak(conn, 0.5)
    await silence(conn, 0.8)
    await wait_for(lambda: bool(events.of(ResponseAudio)))
    await speak(conn, 0.8)  # the fake yields after 0.4 s of overlap
    await silence(conn, 1.8)
    await wait_for(lambda: len(events.of(ResponseDone)) == 2)
    await conn.aclose()
    await events.close()
    assert not server.utterances[0].completed
    # the overlap is not reported while the agent talks, only once the model yielded
    kinds = events.kinds()
    assert kinds.count("input_speech_started") == 2
    assert kinds.index("response_done") < len(kinds) - kinds[::-1].index("input_speech_started")
    assert kinds.index("response_done") > kinds.index("response_started")
    assert [d.status for d in events.of(ResponseDone)] == ["completed", "completed"]
    assert len(events.of(InputCommitted)) == 2


# ----------------------------------------------------------------------------- mute
async def test_mute_and_unmute(fake: Callable[..., Any]) -> None:
    server = await fake(turns=["First reply"])
    conn = await engine_for(server).connect(EngineOptions())
    assert isinstance(conn, OpenAILiveConnection)
    events = Events(conn)
    await conn.mute_input()
    assert conn.input_muted and server.session.muted
    await speak(conn, 0.6)
    await silence(conn, 1.0)
    assert not events.of(ResponseStarted)  # the model heard nothing
    assert not events.of(InputSpeechStarted)
    await conn.unmute_input()
    assert not conn.input_muted and not server.session.muted
    await speak(conn, 0.6)
    await silence(conn, 1.4)
    await wait_for(lambda: bool(events.of(ResponseDone)))
    await conn.aclose()
    await events.close()
    assert events.text() == "First reply"


# ---------------------------------------------------------------------- delegation
async def test_responses_delegation_round_trip(fake: Callable[..., Any]) -> None:
    server = await fake(
        turns=[
            FakeTurn(
                "Let me check",
                user="Weather in Paris?",
                delegate=FakeDelegation("get_weather", {"city": "Paris"}, "It is {output}"),
            )
        ]
    )
    engine = engine_for(server)
    metrics: list[Any] = []
    engine.on("metrics", metrics.append)
    conn = await engine.connect(EngineOptions())
    events = Events(conn)
    await speak(conn, 0.5)
    await silence(conn, 1.0)
    await wait_for(lambda: bool(events.of(ResponseToolCall)))
    call = events.of(ResponseToolCall)[0].call
    assert call.name == "get_weather" and json.loads(call.arguments) == {"city": "Paris"}
    await silence(conn, 1.0)  # a slow tool: the reply has ended meanwhile
    await conn.send_async_tool_output(
        FunctionCallOutput(call_id=call.call_id, name=call.name, output="sunny")
    )
    await silence(conn, 2.0)
    await wait_for(lambda: len(events.of(ResponseDone)) == 2)
    await conn.aclose()
    await events.close()
    session = server.session
    item = session.of("response.item.create")[0]["item"]
    assert item == {"type": "function_call_output", "call_id": call.call_id, "output": "sunny"}
    assert len(session.of("response.create")) == 1
    types = [e["type"] for e in session.events]
    assert types.index("response.item.create") < types.index("response.create")
    assert "It is sunny" in events.text()
    backend = [m for m in metrics if isinstance(m, LLMMetrics)]
    assert [m.prompt_tokens for m in backend] == [100, 120]
    assert backend[0].model == "gpt-5.6-terra"


async def test_client_delegation_round_trip(fake: Callable[..., Any]) -> None:
    server = await fake(
        turns=[
            FakeTurn(
                "One moment",
                user="Where is my order?",
                delegate=FakeDelegation(answer="{output}"),
            )
        ]
    )
    conn = await engine_for(server, delegation="client").connect(EngineOptions())
    events = Events(conn)
    await speak(conn, 0.5)
    await silence(conn, 1.0)
    await wait_for(lambda: bool(events.of(ResponseToolCall)))
    call = events.of(ResponseToolCall)[0].call
    assert call.name == "delegate"
    assert json.loads(call.arguments) == {"request": "Where is my order?"}
    await silence(conn, 1.0)
    await conn.send_async_tool_output(
        FunctionCallOutput(call_id=call.call_id, name=call.name, output="It shipped today.")
    )
    await silence(conn, 1.6)
    await wait_for(lambda: len(events.of(ResponseDone)) == 2)
    await conn.aclose()
    await events.close()
    append = server.sessions[0].of("session.commentary.append")[0]
    assert append["delegation_id"] == call.call_id
    assert append["content"] == "It shipped today."
    assert "It shipped today." in events.text()


async def test_silent_and_failed_client_results(fake: Callable[..., Any]) -> None:
    server = await fake(turns=[FakeTurn("Sure", user="Remember this", delegate=FakeDelegation())])
    conn = await engine_for(server, delegation="client").connect(EngineOptions())
    events = Events(conn)
    await speak(conn, 0.5)
    await silence(conn, 1.0)
    await wait_for(lambda: bool(events.of(ResponseToolCall)))
    call = events.of(ResponseToolCall)[0].call
    out = FunctionCallOutput(call_id=call.call_id, output="boom", is_error=True)
    await conn.send_async_tool_output(out, scheduling="silent")
    # a result without a delegation of this session (e.g. from before a rotation)
    await conn.send_tool_output(FunctionCallOutput(call_id="old", name="lookup", output="42"))
    await wait_for(lambda: bool(server.session.of("session.commentary.append")))
    await conn.aclose()
    await events.close()
    thinking = server.session.of("session.thinking.append")[0]
    assert thinking["delegation_id"] == call.call_id
    assert thinking["content"] == "The task failed: boom"
    commentary = server.session.of("session.commentary.append")[0]
    assert commentary["delegation_id"] is None
    assert commentary["content"] == "Result of lookup: 42"


async def test_long_appends_are_split(fake: Callable[..., Any]) -> None:
    server = await fake()
    conn = await engine_for(server).connect(EngineOptions())
    await conn.send_text("word " * 1000, respond=False)
    await wait_for(lambda: len(server.session.of("session.thinking.append")) >= 3)
    appends = server.session.of("session.thinking.append")
    assert all(len(a["content"]) <= 1800 for a in appends)
    assert sum(len(a["content"].split()) for a in appends) == 1000
    await conn.aclose()


# ------------------------------------------------------------- expiry and rotation
async def test_expiry_rotates_to_a_seeded_session(fake: Callable[..., Any]) -> None:
    server = await fake(turns=["Hello Ada"], session_duration=5.0)
    policy = RotationPolicy(quiet_period=0.2, force_margin=0.5)
    engine = engine_for(server, expiry_warning=2.5, rotation=policy)
    metrics: list[Any] = []
    engine.on("metrics", metrics.append)
    conn = await engine.connect(EngineOptions(instructions="Be nice."))
    assert isinstance(conn, OpenAILiveConnection)
    events = Events(conn)
    await conn.mute_input()
    await conn.unmute_input()
    await speak(conn, 0.5)
    await silence(conn, 1.0)
    await wait_for(lambda: bool(events.of(ResponseDone)))
    await conn.mute_input()  # re-applied on the next session
    await wait_for(lambda: len(server.sessions) == 2 and conn.session_id == "live_002")
    await wait_for(lambda: any(e.status == "reconnected" for e in events.of(EngineStatus)))
    await silence(conn, 0.3)
    await conn.aclose()
    await events.close()
    statuses = [e.status for e in events.of(EngineStatus)]
    assert statuses[:3] == ["expiring", "reconnecting", "reconnected"]
    first, second = server.sessions
    assert first.close_reason == "close_requested"  # retired gracefully, before expiry
    assert second.config["instructions"] == "Be nice."
    seeded = [(i["role"], i["content"][0]["text"]) for i in second.config["input"]]
    assert ("assistant", "Hello Ada") in seeded
    assert second.muted
    rotations = [m for m in metrics if isinstance(m, RotationMetrics)]
    assert len(rotations) == 1 and rotations[0].planned
    assert conn.usage_seconds > 1.0


async def test_expired_session_reconnects_with_history(fake: Callable[..., Any]) -> None:
    server = await fake(turns=["Hello Ada"])
    engine = engine_for(server, rotation=RotationPolicy(backoff=0.05))
    conn = await engine.connect(EngineOptions())
    events = Events(conn)
    await speak(conn, 0.5)
    await silence(conn, 1.0)
    await wait_for(lambda: bool(events.of(ResponseDone)))
    await server.close_session("expired")
    # the switch is over once reconnected is reported (not when the fake saw session.start)
    await wait_for(lambda: any(e.status == "reconnected" for e in events.of(EngineStatus)))
    await conn.aclose()
    await events.close()
    assert [e.status for e in events.of(EngineStatus)] == ["reconnecting", "reconnected"]
    seeded = [i["content"][0]["text"] for i in server.sessions[1].config["input"]]
    assert "Hello Ada" in seeded
    assert not [e for e in events.of(EngineErrorEvent) if not e.recoverable]


async def test_dropped_connection_reconnects(fake: Callable[..., Any]) -> None:
    server = await fake()
    conn = await engine_for(server, rotation=RotationPolicy(backoff=0.05)).connect(EngineOptions())
    events = Events(conn)
    await server.drop()
    # the switch is over once reconnected is reported (not when the fake saw session.start)
    await wait_for(lambda: any(e.status == "reconnected" for e in events.of(EngineStatus)))
    await silence(conn, 0.2)
    assert not conn.closed
    await conn.aclose()
    await events.close()
    assert [e.status for e in events.of(EngineStatus)] == ["reconnecting", "reconnected"]


async def test_single_session_engine_reports_expiry(fake: Callable[..., Any]) -> None:
    server = await fake(session_duration=2.0)
    engine = OpenAILiveSessionEngine(api_key=KEY, base_url=server.url, expiry_warning=1.5)
    conn = await engine.connect(EngineOptions())
    assert isinstance(conn, LiveSessionConnection)
    events = Events(conn)
    await wait_for(lambda: bool(events.of(EngineStatus)))
    status = events.of(EngineStatus)[0]
    assert status.status == "expiring" and status.time_left is not None
    assert 0.5 < status.time_left <= 1.6
    await silence(conn, 1.2)  # the fake closes the session at expires_at
    await events.close()
    fatal = events.of(EngineErrorEvent)
    assert fatal and isinstance(fatal[0].error, ProviderConnectionError)
    assert conn.close_reason == "expired"


# ----------------------------------------------------------- end-to-end with AgentSession
def history(session: AgentSession) -> list[tuple[str, str]]:
    return [(i.role, i.text) for i in session.history.items if isinstance(i, ChatMessage)]


async def test_session_full_duplex_conversation_with_responses_delegation(
    fake: Callable[..., Any],
) -> None:
    server = await fake(
        turns=[
            FakeTurn("Nice to meet you", user="Hi I am Ada"),
            FakeTurn(
                "Let me check",
                user="Weather in Paris?",
                delegate=FakeDelegation("get_weather", {"city": "Paris"}, "It is {output}"),
            ),
        ]
    )
    calls: list[str] = []

    @function_tool
    async def get_weather(city: str) -> str:
        """Get the weather for a city."""
        calls.append(city)
        await asyncio.sleep(1.0)
        return "sunny"

    session = AgentSession(engine_for(server))
    seen: dict[str, list[Any]] = {
        "interrupted": [],
        "metrics": [],
        "tool_result": [],
        "tool_filler": [],
    }
    for name, items in seen.items():
        session.on(name, items.append)
    transport = LoopbackTransport(realtime_playout=True)
    await session.start(Agent("Be nice.", greeting="Welcome!", tools=[get_weather]), transport)
    await wait_for(lambda: len(history(session)) == 1 and session.agent_state == AgentState.LISTENING)  # fmt: skip

    async def say(seconds: float) -> None:
        pieces = [synth_speech(0.02, 16_000, offset=i * 320) for i in range(round(seconds / 0.02))]
        await transport.play_user_audio(AudioFrame.concat(pieces), realtime=True)

    async def quiet(seconds: float) -> None:
        await transport.play_user_audio(AudioFrame.silence(seconds, 16_000), realtime=True)

    await say(0.6)
    await quiet(1.0)
    await wait_for(lambda: len([m for m in seen["metrics"] if isinstance(m, TurnMetrics)]) == 1)
    await wait_for(lambda: session.agent_state == AgentState.LISTENING)
    await say(0.6)
    await quiet(2.0)
    await wait_for(lambda: len(seen["tool_result"]) == 1)
    await wait_for(lambda: any("It is sunny" in t for _, t in history(session)))
    await wait_for(lambda: session.agent_state == AgentState.LISTENING)
    transport.end_user_audio()
    await asyncio.wait_for(session.wait_closed(), 10)

    assert calls == ["Paris"]
    assert seen["interrupted"] == [] and seen["tool_filler"] == []
    assert history(session) == [
        ("assistant", "Welcome!"),
        ("user", "Hi I am Ada"),
        ("assistant", "Nice to meet you"),
        ("user", "Weather in Paris?"),
        ("assistant", "Let me check"),
        ("assistant", "It is sunny"),
    ]
    outputs = [i for i in session.history.items if isinstance(i, FunctionCallOutput)]
    assert [o.output for o in outputs] == ["sunny"]
    turns = [m for m in seen["metrics"] if isinstance(m, TurnMetrics)]
    assert turns[0].voice_to_voice is not None and 0.3 < turns[0].voice_to_voice < 2.0
    assert any(isinstance(m, LLMMetrics) for m in seen["metrics"])
    assert any(isinstance(m, EngineMetrics) and m.ttfb for m in seen["metrics"])
    played = np.concatenate([p.frame.to_float32() for p in transport.played_log])
    assert np.sqrt(np.mean(played**2)) > 0.01
    assert server.sessions[0].close_reason == "close_requested"


async def test_session_client_delegation_runs_the_delegate_tool(
    fake: Callable[..., Any],
) -> None:
    server = await fake(
        turns=[FakeTurn("One moment", user="Where is my order?", delegate=FakeDelegation())]
    )
    requests: list[str] = []

    @function_tool
    async def delegate(request: str) -> str:
        """Answer a request the voice model delegated."""
        requests.append(request)
        return "Your order shipped today."

    session = AgentSession(engine_for(server, delegation="client"))
    results: list[Any] = []
    session.on("tool_result", results.append)
    transport = LoopbackTransport(realtime_playout=True)
    await session.start(Agent("Be nice.", tools=[delegate]), transport)
    pieces = [synth_speech(0.02, 16_000, offset=i * 320) for i in range(30)]
    await transport.play_user_audio(AudioFrame.concat(pieces), realtime=True)
    await transport.play_user_audio(AudioFrame.silence(2.5, 16_000), realtime=True)
    await wait_for(lambda: any("shipped" in t for _, t in history(session)))
    await wait_for(lambda: session.agent_state == AgentState.LISTENING)
    transport.end_user_audio()
    await asyncio.wait_for(session.wait_closed(), 10)
    assert requests == ["Where is my order?"]
    assert len(results) == 1 and not results[0].blocking
    append = server.sessions[0].of("session.commentary.append")[0]
    assert append["content"] == "Your order shipped today."
    assert append["delegation_id"] == results[0].call.call_id


# ------------------------------------------------------------------- real GPT-Live
@pytest.mark.integration
async def test_real_gpt_live_greeting() -> None:
    """Needs ``OPENAI_API_KEY`` with GPT-Live access: ``pytest -m integration``."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("set OPENAI_API_KEY")
    engine = OpenAILiveEngine(delegation="client")
    conn = await engine.connect(EngineOptions(instructions="You are a friendly assistant."))
    assert isinstance(conn, OpenAILiveConnection)
    events = Events(conn)
    await conn.say("Hello, how can I help you today?")
    await silence(conn, 6.0)
    await conn.mute_input()
    await conn.unmute_input()
    await conn.aclose()
    await events.close()
    assert conn.session_id
    assert events.of(ResponseAudio), "GPT-Live said nothing in 6 s"
    assert events.of(ResponseText)
    assert conn.usage_seconds > 0
