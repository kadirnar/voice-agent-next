"""Amazon Nova 2 Sonic engine (``providers/aws/nova_sonic.py``) on a fake event stream.

Everything runs offline: :class:`FakeNovaSonic` replaces the Bedrock SDK with an
in-process stream that speaks the Nova Sonic event protocol, and the SDK adapter
(:class:`BedrockStream`) is exercised against fake ``aws_sdk_bedrock_runtime`` modules.
The real-API test at the end is marked ``integration`` and needs AWS credentials plus
``VAN_TEST_NOVA_SONIC=1`` (it costs money).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import types
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from voice_agent_next import Agent, AgentSession, AgentState, ChatMessage, function_tool
from voice_agent_next.audio import AudioFrame
from voice_agent_next.chat import ChatContext, FunctionCallOutput
from voice_agent_next.engine import EngineOptions
from voice_agent_next.engines.rotation import RotationPolicy
from voice_agent_next.errors import (
    AuthenticationError,
    ConfigurationError,
    MissingDependencyError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from voice_agent_next.events import (
    EngineErrorEvent,
    EngineEvent,
    EngineStatus,
    InputCommitted,
    InputTranscript,
    ResponseAudio,
    ResponseDone,
    ResponseStarted,
    ResponseText,
    ResponseToolCall,
)
from voice_agent_next.metrics import EngineMetrics, RotationMetrics, TurnMetrics
from voice_agent_next.providers.aws.nova_sonic import (
    DEFAULT_MODEL,
    BedrockStream,
    NovaSonicConnection,
    NovaSonicEngine,
    NovaSonicSessionConnection,
    NovaSonicSessionEngine,
    history_messages,
    map_aws_error,
    resolve_model,
)
from voice_agent_next.providers.mock import synth_speech
from voice_agent_next.registry import create, get_provider
from voice_agent_next.testing.nova_sonic import FakeNovaSonic, FakeNovaTurn, FakeToolUse
from voice_agent_next.tools import FunctionTool
from voice_agent_next.transports import LoopbackTransport


# ----------------------------------------------------------------------------- helpers
@pytest.fixture
async def fake_factory() -> AsyncIterator[Callable[..., FakeNovaSonic]]:
    """Factory of fake streams; checked for protocol violations after the test."""
    made: list[FakeNovaSonic] = []

    def make(*turns: str | FakeNovaTurn, **kw: Any) -> FakeNovaSonic:
        fake = FakeNovaSonic(turns, **kw)
        made.append(fake)
        return fake

    yield make
    for fake in made:
        await fake.aclose()
        assert fake.errors == [], f"protocol violations: {fake.errors}"


def engine_for(fake: FakeNovaSonic, **kw: Any) -> NovaSonicEngine:
    return NovaSonicEngine(stream_factory=fake, region="us-east-1", **kw)


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
    """Real-time user speech (16 kHz), in 20 ms frames."""
    for i in range(round(seconds / 0.02)):
        await conn.send_audio(synth_speech(0.02, 16_000, offset=i * 320))
        await asyncio.sleep(0.02)


async def silence(conn: Any, seconds: float) -> None:
    for _ in range(round(seconds / 0.02)):
        await conn.send_audio(AudioFrame.silence(0.02, 16_000))
        await asyncio.sleep(0.02)


def weather_tool() -> FunctionTool:
    @function_tool
    async def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return "sunny"

    return get_weather


# ------------------------------------------------------------------------ configuration
def test_registry_models_and_capabilities() -> None:
    engine = create("engine", "aws/nova-2-sonic", region="us-west-2")
    assert isinstance(engine, NovaSonicEngine)
    assert engine.model == DEFAULT_MODEL == "amazon.nova-2-sonic-v1:0"
    caps = engine.capabilities
    assert caps.max_session_duration == 480.0 and caps.tool_mode == "non_blocking"
    assert caps.server_turn_detection and not caps.truncation and not caps.full_duplex
    assert (engine.input_sample_rate, engine.output_sample_rate) == (16_000, 24_000)
    assert engine.session_engine.region == "us-west-2"
    assert isinstance(create("engine", "aws"), NovaSonicEngine)  # default model
    assert create("engine", "aws/nova-sonic").model == "amazon.nova-sonic-v1:0"
    arn = "arn:aws:bedrock:us-east-1:123:inference-profile/x"
    assert resolve_model(arn) == arn
    spec = get_provider("engine", "aws")
    assert spec.extra == "aws" and "aws_sdk_bedrock_runtime" in spec.requires
    # the default policy looks for a quiet moment from minute 5 and forces it at 7:50
    assert engine.policy.schedule(480.0) == (300.0, 470.0)


def test_region_resolution_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-north-1")
    assert NovaSonicSessionEngine().region == "eu-north-1"
    monkeypatch.setenv("AWS_REGION", "ap-northeast-1")
    assert NovaSonicSessionEngine().region == "ap-northeast-1"
    assert NovaSonicSessionEngine(region="us-west-2").region == "us-west-2"
    low = NovaSonicSessionEngine(endpointing_sensitivity="low")  # type: ignore[arg-type]
    assert low.endpointing_sensitivity == "LOW"
    with pytest.raises(ConfigurationError):
        NovaSonicSessionEngine(output_sample_rate=44_100)
    with pytest.raises(ConfigurationError):
        NovaSonicSessionEngine(endpointing_sensitivity="FAST")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        NovaSonicSessionEngine(transcript="all")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="both"):
        NovaSonicSessionEngine(aws_access_key_id="AKIA")


def test_history_messages_fit_the_chat_history_rules() -> None:
    ctx = ChatContext()
    ctx.add_message("system", "Summary: the user is Ada.")
    ctx.add_message("user", "Hi")
    ctx.add_message("user", "I need the weather")
    ctx.add_message("assistant", "")
    ctx.add_message("assistant", "Sure.")
    ctx.append(FunctionCallOutput(call_id="c1", name="get_weather", output="sunny"))
    ctx.add_message("user", "Thanks")
    system, messages = history_messages(ctx)
    assert system == "Summary: the user is Ada."
    assert messages == [
        ("USER", "Hi\nI need the weather"),
        ("ASSISTANT", "Sure.\n(Result of get_weather: sunny)"),
        ("USER", "Thanks"),
    ]
    big = ChatContext()
    for i in range(300):
        big.add_message("user" if i % 2 else "assistant", f"{i} " + "x" * 1000)
    _, kept = history_messages(big)
    assert sum(len(t) for _, t in kept) < 200_000
    assert kept[-1][1].startswith("299 ")  # the newest messages are kept


def test_error_mapping_by_sdk_class_name() -> None:
    def err(name: str) -> BaseException:
        return type(name, (Exception,), {})("boom")

    assert isinstance(map_aws_error(err("AccessDeniedException")), AuthenticationError)
    assert isinstance(map_aws_error(err("ThrottlingException")), RateLimitError)
    assert isinstance(map_aws_error(err("ModelTimeoutException")), ProviderTimeoutError)
    validation = map_aws_error(err("ValidationException"))
    assert isinstance(validation, ProviderError) and not validation.retryable
    assert "ValidationException: boom" in str(validation)
    retry = map_aws_error(err("ModelStreamErrorException"))
    assert isinstance(retry, ProviderError) and retry.retryable
    assert isinstance(map_aws_error(ConnectionResetError("x")), ProviderConnectionError)
    # subclasses are matched through the MRO (e.g. the SDK's identity errors)
    base = type("IdentityChainError", (Exception,), {})
    sub = type("NoCredentials", (base,), {})
    assert isinstance(map_aws_error(sub("none")), AuthenticationError)
    same = AuthenticationError("x")
    assert map_aws_error(same) is same


# --------------------------------------------------------------------- wire protocol
async def test_setup_events_follow_the_protocol(fake_factory: Callable[..., Any]) -> None:
    fake = fake_factory()
    engine = NovaSonicSessionEngine(
        stream_factory=fake,
        voice="tiffany",
        endpointing_sensitivity="HIGH",
        max_tokens=2048,
        tool_choice="get_weather",
    )
    seed = ChatContext()
    seed.add_message("user", "My name is Ada.")
    seed.add_message("assistant", "Hi Ada!")
    options = EngineOptions(
        instructions="Be brief.", tools=[weather_tool()], chat_ctx=seed, temperature=0.3
    )
    conn = await engine.connect(options)
    assert isinstance(conn, NovaSonicSessionConnection)
    await silence(conn, 0.1)
    await conn.aclose()
    (stream,) = fake.sessions
    assert stream.session_start == {
        "inferenceConfiguration": {"maxTokens": 2048, "topP": 0.9, "temperature": 0.3},
        "turnDetectionConfiguration": {"endpointingSensitivity": "HIGH"},
    }
    prompt = stream.prompt_start
    audio_out = prompt["audioOutputConfiguration"]
    assert audio_out["voiceId"] == "tiffany" and audio_out["sampleRateHertz"] == 24_000
    tool = prompt["toolConfiguration"]["tools"][0]["toolSpec"]
    assert tool["name"] == "get_weather"
    assert json.loads(tool["inputSchema"]["json"])["properties"]["city"]["type"] == "string"
    assert prompt["toolConfiguration"]["toolChoice"] == {"tool": {"name": "get_weather"}}
    assert stream.system == "Be brief."
    assert stream.history == [("USER", "My name is Ada."), ("ASSISTANT", "Hi Ada!")]
    assert stream.audio_config["sampleRateHertz"] == 16_000
    names = [next(iter(e)) for e in stream.events]
    assert names[:2] == ["sessionStart", "promptStart"]
    assert names[-3:] == ["contentEnd", "promptEnd", "sessionEnd"]
    assert stream.graceful and stream.closed
    assert stream.audio_received >= 0.1


async def test_voice_turn_events_and_usage(fake_factory: Callable[..., Any]) -> None:
    fake = fake_factory(FakeNovaTurn("Nice to meet you Ada", user="hi I am Ada"))
    engine = NovaSonicSessionEngine(stream_factory=fake)
    metrics: list[Any] = []
    engine.on("metrics", metrics.append)
    conn = await engine.connect(EngineOptions())
    events = Events(conn)
    await speak(conn, 0.5)
    await silence(conn, 0.8)
    await wait_for(lambda: bool(events.of(ResponseDone)))
    await conn.aclose()
    await events.close()
    kinds = events.kinds()
    assert kinds.index("input_speech_started") < kinds.index("input_speech_stopped")
    assert kinds.index("input_speech_stopped") < kinds.index("input_committed")
    assert kinds.index("input_committed") < kinds.index("response_started")
    partial = [e for e in events.of(InputTranscript) if not e.is_final]
    final = [e for e in events.of(InputTranscript) if e.is_final]
    assert partial and final[0].text == "hi I am Ada"
    assert final[0].item_id == events.of(InputCommitted)[0].item_id
    assert events.text() == "Nice to meet you Ada"
    audio = events.of(ResponseAudio)
    assert audio and audio[0].frame.sample_rate == 24_000
    assert sum(a.frame.duration for a in audio) == pytest.approx(1.2, abs=0.05)  # 5 words
    (done,) = events.of(ResponseDone)
    assert done.status == "completed"
    assert done.usage is not None and done.usage.output_audio_tokens > 0
    assert conn.usage.input_audio_tokens > 0
    m = [x for x in metrics if isinstance(x, EngineMetrics)]
    assert m and m[0].ttfb is not None and m[0].output_audio_tokens > 0
    assert conn.session_id == "sess_001"


async def test_final_transcript_mode(fake_factory: Callable[..., Any]) -> None:
    fake = fake_factory(FakeNovaTurn("Spoken words here", user="hello"))
    engine = NovaSonicSessionEngine(stream_factory=fake, transcript="final")
    conn = await engine.connect(EngineOptions())
    events = Events(conn)
    await speak(conn, 0.4)
    await silence(conn, 0.8)
    await wait_for(lambda: bool(events.of(ResponseDone)))
    await conn.aclose()
    await events.close()
    assert events.text() == "Spoken words here"
    # the final transcript comes after the audio: the response waits for it
    last_audio = max(i for i, e in enumerate(events.items) if isinstance(e, ResponseAudio))
    first_text = min(i for i, e in enumerate(events.items) if isinstance(e, ResponseText))
    assert first_text > last_audio


async def test_text_input_say_and_echo_guard(fake_factory: Callable[..., Any]) -> None:
    fake = fake_factory("Paris is lovely", echo_text=True)
    engine = NovaSonicSessionEngine(stream_factory=fake)
    conn = await engine.connect(EngineOptions())
    events = Events(conn)
    await conn.say("Welcome!")
    await wait_for(lambda: len(events.of(ResponseDone)) == 1)
    await conn.send_text("Tell me about Paris")
    await wait_for(lambda: len(events.of(ResponseDone)) == 2)
    await conn.aclose()
    await events.close()
    (stream,) = fake.sessions
    assert stream.text_messages[0] == (
        'Say exactly the following, verbatim, and nothing else: "Welcome!"'
    )
    assert stream.text_messages[1] == "Tell me about Paris"
    assert [e.delta for e in events.of(ResponseText)] == ["Welcome!", "Paris is lovely"]
    # the model's echo of our own text is not reported as user speech
    assert not events.of(InputTranscript) and not events.of(InputCommitted)


async def test_cancel_drops_the_rest_of_the_turn(fake_factory: Callable[..., Any]) -> None:
    fake = fake_factory()
    conn = await NovaSonicSessionEngine(stream_factory=fake).connect(EngineOptions())
    events = Events(conn)
    await conn.say("one two three four five six seven eight")
    await wait_for(lambda: bool(events.of(ResponseAudio)))
    await conn.cancel_response()
    received = len(events.of(ResponseAudio))
    await asyncio.sleep(0.5)
    assert len(events.of(ResponseAudio)) == received  # muted until the turn ends
    (done,) = events.of(ResponseDone)
    assert done.status == "cancelled"
    await conn.say("again")  # the next turn is heard
    await wait_for(lambda: len(events.of(ResponseDone)) == 2)
    await conn.aclose()
    await events.close()
    assert events.of(ResponseDone)[1].status == "completed"
    assert "again" in events.text()


@pytest.mark.parametrize("legacy", [False, True])
async def test_model_barge_in_cancels_the_response(
    fake_factory: Callable[..., Any], legacy: bool
) -> None:
    fake = fake_factory(
        FakeNovaTurn("one two three four five six seven eight nine ten", user="tell me"),
        FakeNovaTurn("Sure", user="stop please"),
        legacy_interrupt=legacy,
    )
    conn = await NovaSonicSessionEngine(stream_factory=fake).connect(EngineOptions())
    events = Events(conn)
    await speak(conn, 0.4)
    await silence(conn, 0.6)
    await wait_for(lambda: bool(events.of(ResponseAudio)))
    await speak(conn, 0.5)  # talk over the agent
    await silence(conn, 0.7)
    await wait_for(lambda: len(events.of(ResponseDone)) == 2)
    await conn.aclose()
    await events.close()
    first, second = events.of(ResponseDone)
    assert first.status == "cancelled" and second.status == "completed"
    kinds = events.kinds()
    cancelled_at = kinds.index("response_done")
    assert "input_speech_started" in kinds[kinds.index("response_started") : cancelled_at]
    finals = [e.text for e in events.of(InputTranscript) if e.is_final]
    assert finals == ["tell me", "stop please"]


async def test_tool_round_trip_at_the_engine_level(fake_factory: Callable[..., Any]) -> None:
    fake = fake_factory(
        FakeNovaTurn(
            "Let me check",
            user="weather in Paris?",
            tool=FakeToolUse("get_weather", {"city": "Paris"}, "It is {result}"),
        )
    )
    conn = await NovaSonicSessionEngine(stream_factory=fake).connect(
        EngineOptions(tools=[weather_tool()])
    )
    events = Events(conn)
    await speak(conn, 0.4)
    await silence(conn, 0.6)
    await wait_for(lambda: bool(events.of(ResponseToolCall)))
    call = events.of(ResponseToolCall)[0].call
    assert call.name == "get_weather" and json.loads(call.arguments) == {"city": "Paris"}
    await wait_for(lambda: "Let me check" in events.text())  # said while the tool runs
    await conn.send_async_tool_output(
        FunctionCallOutput(call_id=call.call_id, name=call.name, output="sunny")
    )
    await wait_for(lambda: "It is sunny" in events.text())
    await wait_for(lambda: not any(e.status == "incomplete" for e in events.of(ResponseDone)))
    await conn.aclose()
    await events.close()
    (stream,) = fake.sessions
    assert json.loads(stream.tool_results[call.call_id]) == {"result": "sunny"}


# ------------------------------------------------------------------------ failures
async def test_open_failure_raises(fake_factory: Callable[..., Any]) -> None:
    fake = fake_factory(fail_open=AuthenticationError("aws: AccessDeniedException"))
    with pytest.raises(AuthenticationError):
        await engine_for(fake).connect(EngineOptions())
    throttled = type("ThrottlingException", (Exception,), {})("slow down")
    fake2 = fake_factory(fail_open=throttled)
    with pytest.raises(RateLimitError):
        await NovaSonicSessionEngine(stream_factory=fake2).connect(EngineOptions())


async def test_fatal_stream_error_closes_the_conversation(
    fake_factory: Callable[..., Any],
) -> None:
    fake = fake_factory()
    conn = await engine_for(fake).connect(EngineOptions())
    events = Events(conn)
    fake.sessions[0].fail(map_aws_error(type("ValidationException", (Exception,), {})("bad")))
    await wait_for(lambda: conn.closed)
    await events.close()
    (error,) = events.of(EngineErrorEvent)
    assert not error.recoverable and "ValidationException" in str(error.error)
    assert len(fake.sessions) == 1  # not retried


async def test_dropped_stream_reconnects_with_history(fake_factory: Callable[..., Any]) -> None:
    fake = fake_factory(FakeNovaTurn("Hello Ada", user="I am Ada"))
    engine = engine_for(fake, rotation=RotationPolicy(backoff=0.05))
    conn = await engine.connect(EngineOptions(instructions="Be nice."))
    events = Events(conn)
    await speak(conn, 0.4)
    await silence(conn, 0.6)
    await wait_for(lambda: bool(events.of(ResponseDone)))
    fake.sessions[0].drop()
    await wait_for(lambda: any(e.status == "reconnected" for e in events.of(EngineStatus)))
    await silence(conn, 0.2)
    await conn.aclose()
    await events.close()
    assert [e.status for e in events.of(EngineStatus)] == ["reconnecting", "reconnected"]
    second = fake.sessions[1]
    assert second.system == "Be nice."
    assert second.history == [("USER", "I am Ada"), ("ASSISTANT", "Hello Ada")]
    assert not [e for e in events.of(EngineErrorEvent) if not e.recoverable]


# ------------------------------------------------------------------------ rotation
async def test_connection_limit_rotates_before_it_is_hit(
    fake_factory: Callable[..., Any],
) -> None:
    fake = fake_factory(
        FakeNovaTurn("Hello Ada", user="I am Ada"), FakeNovaTurn("Yes Ada", user="my name?")
    )
    policy = RotationPolicy(lead=2.0, quiet_period=0.2, force_margin=0.5)
    engine = engine_for(fake, session_limit=3.0, rotation=policy)
    metrics: list[Any] = []
    engine.on("metrics", metrics.append)
    conn = await engine.connect(EngineOptions(instructions="Be nice."))
    assert isinstance(conn, NovaSonicConnection)
    events = Events(conn)
    await speak(conn, 0.4)
    await silence(conn, 0.6)
    await wait_for(lambda: bool(events.of(ResponseDone)))
    await wait_for(lambda: any(e.status == "reconnected" for e in events.of(EngineStatus)))
    first, second = fake.sessions
    assert first.graceful  # retired with the closing sequence, before the limit
    assert second.system == "Be nice."
    assert ("ASSISTANT", "Hello Ada") in second.history
    # the conversation goes on on the new connection
    await speak(conn, 0.4)
    await silence(conn, 0.6)
    await wait_for(lambda: len(events.of(ResponseDone)) == 2)
    await conn.aclose()
    await events.close()
    statuses = [e.status for e in events.of(EngineStatus)]
    assert statuses[:3] == ["expiring", "reconnecting", "reconnected"]
    rotations = [m for m in metrics if isinstance(m, RotationMetrics)]
    assert rotations[0].planned and rotations[0].carried_items >= 2
    assert conn.usage.output_audio_tokens > 0
    assert "Yes Ada" in events.text()


async def test_hard_limit_without_rotation_reconnects(fake_factory: Callable[..., Any]) -> None:
    """If the limit is hit anyway (proactive rotation off), the conversation reconnects."""
    fake = fake_factory(session_limit=1.0)
    engine = engine_for(fake, rotation=RotationPolicy(proactive=False, backoff=0.05))
    conn = await engine.connect(EngineOptions())
    events = Events(conn)
    await wait_for(lambda: any(e.status == "reconnected" for e in events.of(EngineStatus)))
    await conn.aclose()
    await events.close()
    assert len(fake.sessions) >= 2


async def test_update_rotates_to_a_connection_with_the_new_prompt(
    fake_factory: Callable[..., Any],
) -> None:
    fake = fake_factory()
    engine = engine_for(fake, rotation=RotationPolicy(quiet_period=0.1))
    conn = await engine.connect(EngineOptions(instructions="Old."))
    events = Events(conn)
    await silence(conn, 0.2)
    await conn.update(instructions="New.", tools=[weather_tool()])
    await wait_for(lambda: any(e.status == "reconnected" for e in events.of(EngineStatus)))
    await conn.aclose()
    await events.close()
    second = fake.sessions[1]
    assert second.system == "New."
    tools = second.prompt_start["toolConfiguration"]["tools"]
    assert tools[0]["toolSpec"]["name"] == "get_weather"


# ----------------------------------------------------------- end-to-end with AgentSession
COUNTING = "One two three. Four five six. Seven eight nine. Ten eleven twelve."


def history(session: AgentSession) -> list[tuple[str, str]]:
    return [(i.role, i.text) for i in session.history.items if isinstance(i, ChatMessage)]


async def test_session_voice_turn_with_tool_and_greeting(
    fake_factory: Callable[..., Any],
) -> None:
    fake = fake_factory(
        FakeNovaTurn("Nice to meet you", user="Hi I am Ada"),
        FakeNovaTurn(
            "",
            user="Weather in Paris?",
            tool=FakeToolUse("get_weather", {"city": "Paris"}, "It is {result} in Paris"),
        ),
    )
    calls: list[str] = []

    @function_tool
    async def get_weather(city: str) -> str:
        """Get the weather for a city."""
        calls.append(city)
        await asyncio.sleep(0.3)
        return "sunny"

    session = AgentSession(engine_for(fake))
    seen: dict[str, list[Any]] = {"metrics": [], "tool_result": [], "interrupted": []}
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

    await say(0.5)
    await quiet(1.0)
    await wait_for(lambda: any(isinstance(m, TurnMetrics) for m in seen["metrics"]))
    await wait_for(lambda: session.agent_state == AgentState.LISTENING)
    await say(0.5)
    await quiet(1.0)
    await wait_for(lambda: len(seen["tool_result"]) == 1)
    await wait_for(lambda: any("It is sunny" in t for _, t in history(session)))
    await wait_for(lambda: session.agent_state == AgentState.LISTENING)
    transport.end_user_audio()
    await asyncio.wait_for(session.wait_closed(), 10)

    assert calls == ["Paris"]
    assert seen["interrupted"] == []
    assert history(session) == [
        ("assistant", "Welcome!"),
        ("user", "Hi I am Ada"),
        ("assistant", "Nice to meet you"),
        ("user", "Weather in Paris?"),
        ("assistant", "It is sunny in Paris"),
    ]
    outputs = [i for i in session.history.items if isinstance(i, FunctionCallOutput)]
    assert [o.output for o in outputs] == ["sunny"]
    turns = [m for m in seen["metrics"] if isinstance(m, TurnMetrics)]
    assert turns[0].voice_to_voice is not None and 0.2 < turns[0].voice_to_voice < 2.5
    played = np.concatenate([p.frame.to_float32() for p in transport.played_log])
    assert np.sqrt(np.mean(played**2)) > 0.01
    assert fake.sessions[0].graceful
    assert fake.sessions[0].system == "Be nice."


async def test_session_barge_in(fake_factory: Callable[..., Any]) -> None:
    fake = fake_factory(
        FakeNovaTurn(COUNTING, user="count"),
        FakeNovaTurn("Okay", user="stop"),
    )
    session = AgentSession(engine_for(fake))
    interrupted: list[Any] = []
    session.on("interrupted", interrupted.append)
    transport = LoopbackTransport(realtime_playout=True)
    await session.start(Agent("Be nice."), transport)

    async def say(seconds: float) -> None:
        pieces = [synth_speech(0.02, 16_000, offset=i * 320) for i in range(round(seconds / 0.02))]
        await transport.play_user_audio(AudioFrame.concat(pieces), realtime=True)

    await say(0.4)
    await transport.play_user_audio(AudioFrame.silence(0.6, 16_000), realtime=True)
    await wait_for(lambda: session.agent_state == AgentState.SPEAKING)
    await asyncio.sleep(0.4)
    await say(0.5)
    await transport.play_user_audio(AudioFrame.silence(0.8, 16_000), realtime=True)
    await wait_for(lambda: any(t == "Okay" for _, t in history(session)))
    await wait_for(lambda: session.agent_state == AgentState.LISTENING)
    transport.end_user_audio()
    await asyncio.wait_for(session.wait_closed(), 10)
    assert len(interrupted) == 1
    counted = next(
        i for i in session.history.items if isinstance(i, ChatMessage) and i.role == "assistant"
    )
    assert counted.interrupted
    assert counted.text.startswith("One two three.") and len(counted.text) < len(COUNTING)
    assert [t for r, t in history(session) if r == "user"] == ["count", "stop"]


# ------------------------------------------------------------------ SDK adapter (faked)
@dataclass
class _Part:
    bytes_: bytes | None = None


@dataclass
class _InChunk:
    value: _Part


@dataclass
class _OutChunk:
    value: _Part


class ModelStreamErrorException(Exception):
    pass


@dataclass
class _OutError:
    value: BaseException


class _FakeSDK:
    """Stand-ins for ``aws_sdk_bedrock_runtime.{client,config,models}``."""

    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = outputs
        self.sent: list[dict[str, Any]] = []
        self.resolved: dict[str, Any] = {}
        self.invoked: list[str] = []
        self.closed: list[str] = []
        sdk = self

        class Config:
            region: str | None = None

            @classmethod
            async def resolve(cls, **kw: Any) -> Config:
                sdk.resolved = kw
                cfg = cls()
                cfg.region = kw.get("region")
                return cfg

        class Input:
            def __init__(self, model_id: str) -> None:
                self.model_id = model_id

        class InputStream:
            async def send(self, chunk: _InChunk) -> None:
                assert chunk.value.bytes_ is not None
                sdk.sent.append(json.loads(chunk.value.bytes_))

            async def close(self) -> None:
                sdk.closed.append("input")

        class OutputStream:
            async def receive(self) -> Any:
                return sdk.outputs.pop(0) if sdk.outputs else None

            async def close(self) -> None:
                sdk.closed.append("output")

        class Duplex:
            input_stream = InputStream()

            async def await_output(self) -> tuple[Any, Any]:
                return object(), OutputStream()

        class Client:
            def __init__(self, config: Config) -> None:
                self.config = config

            async def invoke_model_with_bidirectional_stream(self, inp: Input) -> Duplex:
                sdk.invoked.append(inp.model_id)
                return Duplex()

            async def close(self) -> None:
                sdk.closed.append("client")

        self.client = types.ModuleType("aws_sdk_bedrock_runtime.client")
        self.client.AsyncBedrockRuntimeClient = Client  # type: ignore[attr-defined]
        self.client.InvokeModelWithBidirectionalStreamOperationInput = Input  # type: ignore[attr-defined]
        self.config = types.ModuleType("aws_sdk_bedrock_runtime.config")
        self.config.AsyncBedrockRuntimeConfig = Config  # type: ignore[attr-defined]
        self.models = types.ModuleType("aws_sdk_bedrock_runtime.models")
        self.models.InvokeModelWithBidirectionalStreamInputChunk = _InChunk  # type: ignore[attr-defined]
        self.models.BidirectionalInputPayloadPart = _Part  # type: ignore[attr-defined]

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pkg = types.ModuleType("aws_sdk_bedrock_runtime")
        monkeypatch.setitem(sys.modules, "aws_sdk_bedrock_runtime", pkg)
        monkeypatch.setitem(sys.modules, "aws_sdk_bedrock_runtime.client", self.client)
        monkeypatch.setitem(sys.modules, "aws_sdk_bedrock_runtime.config", self.config)
        monkeypatch.setitem(sys.modules, "aws_sdk_bedrock_runtime.models", self.models)


def _out(name: str, **body: Any) -> _OutChunk:
    return _OutChunk(_Part(json.dumps({"event": {name: body}}).encode()))


async def test_bedrock_stream_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = _FakeSDK(
        [
            _out("completionStart", sessionId="s1"),
            _OutChunk(_Part(None)),  # an empty chunk is skipped
            _out("usageEvent", details={"total": {"output": {"speechTokens": 3}}}),
            _OutError(ModelStreamErrorException("model failed")),
        ]
    )
    sdk.install(monkeypatch)
    engine = NovaSonicSessionEngine(
        region="eu-north-1",
        profile="work",
        aws_access_key_id="AKIA",
        aws_secret_access_key="secret",
        aws_session_token="token",
        endpoint_url="https://bedrock.example",
    )
    stream = await BedrockStream.open(engine)
    assert sdk.invoked == [DEFAULT_MODEL]
    assert sdk.resolved == {
        "profile": "work",
        "region": "eu-north-1",
        "endpoint_uri": "https://bedrock.example",
        "aws_access_key_id": "AKIA",
        "aws_secret_access_key": "secret",
        "aws_session_token": "token",
    }
    await stream.send({"event": {"sessionEnd": {}}})
    assert sdk.sent == [{"event": {"sessionEnd": {}}}]
    first = await stream.receive()
    assert first == {"event": {"completionStart": {"sessionId": "s1"}}}
    second = await stream.receive()
    assert second is not None and "usageEvent" in second["event"]
    with pytest.raises(ProviderError, match="model failed") as info:
        await stream.receive()
    assert info.value.retryable
    await stream.close()
    await engine.aclose()
    assert sdk.closed == ["input", "output", "client"]


async def test_bedrock_client_uses_the_default_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(var, raising=False)
    sdk = _FakeSDK([])
    sdk.install(monkeypatch)
    engine = NovaSonicSessionEngine()
    client = await engine.bedrock_client()
    assert await engine.bedrock_client() is client  # shared
    assert sdk.resolved == {"profile": None}  # credentials/region from the AWS chain
    assert client.config.region == "us-east-1"  # fallback when no region is configured
    await engine.aclose()


async def test_missing_sdk_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("", ".client", ".config", ".models"):
        monkeypatch.setitem(sys.modules, "aws_sdk_bedrock_runtime" + name, None)
    engine = NovaSonicEngine()
    with pytest.raises(MissingDependencyError, match=r"voice-agent-next\[aws\]"):
        await engine.connect(EngineOptions())


# ------------------------------------------------------------------- real Nova Sonic
def _aws_ready() -> str | None:
    if os.environ.get("VAN_TEST_NOVA_SONIC") != "1":
        return "set VAN_TEST_NOVA_SONIC=1 (the test calls Bedrock and costs money)"
    if not (os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE")):
        return "set AWS credentials (AWS_ACCESS_KEY_ID/... or AWS_PROFILE)"
    if importlib.util.find_spec("aws_sdk_bedrock_runtime") is None:
        return "install the aws extra (Python >= 3.12)"
    return None


@pytest.mark.integration
async def test_real_nova_sonic_greeting() -> None:
    """``VAN_TEST_NOVA_SONIC=1 pytest -m integration tests/test_nova_sonic.py``."""
    reason = _aws_ready()
    if reason:
        pytest.skip(reason)
    engine = NovaSonicEngine(endpointing_sensitivity="MEDIUM")
    conn = await engine.connect(EngineOptions(instructions="You are a friendly assistant."))
    events = Events(conn)
    await conn.say("Hello, how can I help you today?")
    await silence(conn, 6.0)
    await conn.aclose()
    await engine.aclose()
    await events.close()
    assert not [e for e in events.of(EngineErrorEvent) if not e.recoverable]
    assert events.of(ResponseStarted) and events.of(ResponseAudio), "Nova said nothing in 6 s"
    assert events.of(ResponseText)
    assert isinstance(conn, NovaSonicConnection) and conn.usage.output_audio_tokens > 0
