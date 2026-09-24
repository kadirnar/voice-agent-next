"""OpenAI Realtime engine + compatibility profiles, tested against a fake Realtime server."""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
from collections.abc import AsyncIterator, Callable
from typing import Any, TypeVar

import pytest

from tests.fake_realtime_server import FakeRealtimeServer
from voice_agent_next import (
    Agent,
    AgentSession,
    AgentState,
    AudioFrame,
    ChatContext,
    ChatMessage,
    FunctionCallOutput,
    create,
    function_tool,
    list_providers,
)
from voice_agent_next.engine import EngineConnection, EngineOptions
from voice_agent_next.errors import (
    AuthenticationError,
    ConfigurationError,
    EngineError,
    ProviderConnectionError,
    ProviderError,
    RateLimitError,
)
from voice_agent_next.events import (
    EngineErrorEvent,
    EngineStatus,
    EngineUsage,
    InputCommitted,
    InputSpeechStarted,
    InputSpeechStopped,
    InputTranscript,
    ResponseAudio,
    ResponseDone,
    ResponseStarted,
    ResponseText,
    ResponseToolCall,
)
from voice_agent_next.metrics import EngineMetrics, TurnMetrics
from voice_agent_next.providers.azure_openai import AzureOpenAIRealtimeEngine
from voice_agent_next.providers.localai import LocalAIRealtimeEngine
from voice_agent_next.providers.mock import MockToolCall, synth_speech
from voice_agent_next.providers.openai.realtime import (
    VERBATIM_INSTRUCTIONS,
    OpenAIRealtimeConnection,
    OpenAIRealtimeEngine,
    _AudioItem,
    _LevelTracker,
    realtime_url,
)
from voice_agent_next.providers.qwen_omni import QwenOmniRealtimeEngine
from voice_agent_next.providers.speaches import SpeachesRealtimeEngine
from voice_agent_next.providers.vllm_realtime import VLLMRealtimeEngine
from voice_agent_next.providers.xai import XAIRealtimeEngine
from voice_agent_next.transports import LoopbackTransport
from voice_agent_next.utils import now

KEY = "sk-test"
T = TypeVar("T")

ENV_VARS = (
    "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_AD_TOKEN", "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME", "XAI_API_KEY", "DASHSCOPE_API_KEY", "DASHSCOPE_WORKSPACE_ID",
    "DASHSCOPE_REGION", "VLLM_BASE_URL", "VLLM_API_KEY", "SPEACHES_BASE_URL", "SPEACHES_API_KEY",
    "LOCALAI_BASE_URL", "LOCALAI_API_KEY",
)  # fmt: skip


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@function_tool
async def get_weather(city: str) -> str:
    """Weather lookup."""
    return f"sunny in {city}"


async def wait_for(predicate: Callable[[], Any], timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), timeout)


class Collector:
    """Collects the events of an engine connection."""

    def __init__(self, conn: EngineConnection) -> None:
        self.events: list[Any] = []
        self.task = asyncio.create_task(self._run(conn))

    async def _run(self, conn: EngineConnection) -> None:
        async for ev in conn.events():
            self.events.append(ev)

    def of(self, cls: type[T]) -> list[T]:
        return [e for e in self.events if isinstance(e, cls)]

    def index(self, cls: type) -> int:
        return next(i for i, e in enumerate(self.events) if isinstance(e, cls))

    async def wait(self, predicate: Callable[[], Any], timeout: float = 5.0) -> None:
        await wait_for(predicate, timeout)


@contextlib.asynccontextmanager
async def connected(
    engine: OpenAIRealtimeEngine, options: EngineOptions | None = None
) -> AsyncIterator[tuple[OpenAIRealtimeConnection, Collector]]:
    conn = await engine.connect(options or EngineOptions())
    assert isinstance(conn, OpenAIRealtimeConnection)
    rec = Collector(conn)
    try:
        yield conn, rec
    finally:
        await conn.aclose()
        await asyncio.wait_for(rec.task, 2)


async def feed(conn: EngineConnection, frame: AudioFrame, chunk: float = 0.02) -> None:
    """Send ``frame`` in ``chunk``-second pieces (faster than real time)."""
    step = round(chunk * frame.sample_rate) * 2
    for i in range(0, len(frame.data), step):
        await conn.send_audio(AudioFrame(frame.data[i : i + step], frame.sample_rate, 1, now()))
    await asyncio.sleep(0)


async def user_turn(conn: EngineConnection, speech: float = 1.0, silence: float = 0.8) -> None:
    await feed(conn, AudioFrame.silence(0.3, 16_000))
    await feed(conn, synth_speech(speech, 16_000))
    await feed(conn, AudioFrame.silence(silence, 16_000))


def spoken(rec: Collector, response_id: str | None = None) -> str:
    return "".join(
        t.delta for t in rec.of(ResponseText) if response_id is None or t.response_id == response_id
    )


# ------------------------------------------------------------------ registry & profiles
def test_registry_resolves_engines_and_profiles() -> None:
    engine = create("engine", "openai/gpt-realtime-2.1", api_key=KEY)
    assert type(engine) is OpenAIRealtimeEngine and engine.model == "gpt-realtime-2.1"
    assert engine.url == "wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1"
    assert engine.request_headers() == {"Authorization": f"Bearer {KEY}"}  # GA: no OpenAI-Beta
    caps = engine.capabilities
    assert caps.native_audio and caps.server_turn_detection and caps.truncation
    assert caps.text_input and caps.input_transcription and caps.max_session_duration == 3600
    assert (engine.input_sample_rate, engine.output_sample_rate) == (24_000, 24_000)
    assert engine.turn_detection == {
        "type": "semantic_vad", "create_response": True, "interrupt_response": True,
    }  # fmt: skip
    manual = create("engine", "openai", api_key=KEY, turn_detection=None)
    assert not manual.capabilities.server_turn_detection

    xai = create("engine", "xai", api_key=KEY)
    assert type(xai) is XAIRealtimeEngine and xai.voice == "eve"
    assert xai.url == "wss://api.x.ai/v1/realtime?model=grok-voice-latest"
    assert xai.turn_detection == {"type": "server_vad"}  # no create/interrupt flags for xAI

    qwen = create("engine", "qwen_omni", api_key=KEY, workspace_id="ws-123")
    assert type(qwen) is QwenOmniRealtimeEngine
    assert qwen.url == (
        "wss://ws-123.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime"
        "?model=qwen3.8-omni-flash-realtime"
    )
    assert qwen.input_sample_rate == 16_000 and qwen.output_sample_rate == 24_000
    assert not qwen.capabilities.truncation and not qwen.capabilities.text_input

    azure = create("engine", "azure_openai/my-dep", endpoint="https://res.openai.azure.com",
                   api_key=KEY)  # fmt: skip
    assert type(azure) is AzureOpenAIRealtimeEngine
    assert azure.url == "wss://res.openai.azure.com/openai/v1/realtime?model=my-dep"
    assert azure.request_headers() == {"api-key": KEY}

    vllm = create("engine", "vllm_realtime", query={"duplex": "1"}, turn_detection=None)
    assert type(vllm) is VLLMRealtimeEngine
    assert vllm.url == "ws://localhost:8000/v1/realtime?duplex=1"  # no model: server default
    speaches = create("engine", "speaches")
    assert type(speaches) is SpeachesRealtimeEngine and not speaches.capabilities.truncation
    localai = create("engine", "localai")
    assert type(localai) is LocalAIRealtimeEngine
    assert localai.url == "ws://localhost:8080/v1/realtime?model=gpt-realtime"

    specs = {s.name: s for s in list_providers("engine")}
    names = {"openai", "azure_openai", "xai", "qwen_omni", "vllm_realtime", "speaches", "localai"}
    assert names <= specs.keys()
    assert specs["openai"].env == ("OPENAI_API_KEY",) and not specs["openai"].local
    assert specs["vllm_realtime"].local and specs["speaches"].local and specs["localai"].local
    assert specs["xai"].default_model == "grok-voice-latest"


def test_configuration_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        OpenAIRealtimeEngine()
    with pytest.raises(ConfigurationError, match="AZURE_OPENAI_ENDPOINT"):
        AzureOpenAIRealtimeEngine(api_key=KEY)
    with pytest.raises(ConfigurationError, match="DASHSCOPE_WORKSPACE_ID"):
        QwenOmniRealtimeEngine(api_key=KEY)
    with pytest.raises(ConfigurationError, match="semantic_vad"):
        XAIRealtimeEngine(api_key=KEY, turn_detection="semantic_vad")
    with pytest.raises(ConfigurationError, match="unknown realtime profile"):
        OpenAIRealtimeEngine(api_key=KEY, profile="nope")
    with pytest.raises(ConfigurationError, match="invalid realtime base URL"):
        OpenAIRealtimeEngine(api_key=KEY, base_url="ftp://example.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert OpenAIRealtimeEngine().api_key == "env-key"
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://r.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "dep")
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", "entra")
    azure = AzureOpenAIRealtimeEngine()
    assert azure.url == "wss://r.openai.azure.com/openai/v1/realtime?model=dep"
    assert azure.request_headers() == {"Authorization": "Bearer entra"}


def test_realtime_url() -> None:
    assert realtime_url("https://api.openai.com/v1", model="m") == (
        "wss://api.openai.com/v1/realtime?model=m"
    )
    assert realtime_url("http://localhost:8000/v1/") == "ws://localhost:8000/v1/realtime"
    assert realtime_url("wss://h/v1/realtime?x=1", model="a/b", query={"duplex": "1"}) == (
        "wss://h/v1/realtime?x=1&model=a%2Fb&duplex=1"
    )


def test_level_tracker_locates_the_real_speech_end() -> None:
    tracker, t = _LevelTracker(), 0.0
    audio = AudioFrame.concat(
        [
            AudioFrame.silence(0.5, 24_000),
            synth_speech(1.0, 24_000),
            AudioFrame.silence(0.7, 24_000),
        ]
    )
    for i in range(0, 2200, 20):
        frame = audio.slice(i / 1000, (i + 20) / 1000)
        t += frame.duration
        tracker.push(t, frame)
    # audio_end_ms including the trailing silence (OpenAI) or at the speech end (others)
    assert tracker.speech_end(2.2, lookback=0.8) == pytest.approx(1.5, abs=0.021)
    assert tracker.speech_end(1.5, lookback=0.8) == pytest.approx(1.5, abs=0.021)
    assert tracker.speech_end(2.2, lookback=0.3) == 2.2  # no speech within the hold: keep it
    noisy, t = _LevelTracker(), 0.0
    for i in range(50):  # noise as loud as speech: inconclusive, keep the server value
        frame = synth_speech(0.02, 24_000, offset=i * 480)
        t += frame.duration
        noisy.push(t, frame)
    assert noisy.speech_end(0.9, lookback=0.8) == 0.9


def test_truncation_plan_spans_multiple_audio_items() -> None:
    conn = OpenAIRealtimeConnection(OpenAIRealtimeEngine(api_key=KEY), EngineOptions())
    conn._audio_items.update(
        {"a": _AudioItem("r1", 0, 800.4), "b": _AudioItem("r1", 1, 1200.0),
         "c": _AudioItem("r2", 0, 500.0)}
    )  # fmt: skip
    assert conn._truncation_plan("a", 300) == [("a", 0, 300), ("b", 1, 0)]
    assert conn._truncation_plan("a", 1000) == [("b", 1, 200)]  # the preamble was fully heard
    assert conn._truncation_plan("a", 5000) == []  # everything was heard
    assert conn._truncation_plan("c", 501) == []  # clamped to the audio received
    assert conn._truncation_plan("unknown", 100) == []  # no audio: nothing to truncate


# -------------------------------------------------------------------- GA protocol
async def test_handshake_and_ga_session_update() -> None:
    history = ChatContext()
    history.add_message("system", "Be kind.")
    history.add_message("user", "Hi there")
    history.add_message("assistant", "Hello!")
    history.add_function_call("get_weather", '{"city": "Rome"}', "call_1")
    history.add_function_output("call_1", "rainy")
    options = EngineOptions(
        instructions="You are a weather bot.",
        tools=[get_weather],
        voice="cedar",
        language="en-US",  # sent as ISO-639-1
        chat_ctx=history,
        extra={"audio": {"output": {"speed": 1.2}}},
    )
    async with FakeRealtimeServer(api_key=KEY) as server:
        engine = OpenAIRealtimeEngine(
            base_url=server.url, api_key=KEY, noise_reduction="near_field",
            reasoning_effort="low", speed=1.1, max_output_tokens=512, session={"tracing": "auto"},
        )  # fmt: skip
        async with connected(engine, options) as (conn, rec):
            handshake = server.handshakes[0]
            assert handshake.path == "/v1/realtime"
            assert handshake.query == {"model": "gpt-realtime-2.1"}
            assert handshake.headers["authorization"] == f"Bearer {KEY}"
            assert "openai-beta" not in handshake.headers
            assert server.events("session.update")[0]["session"] == {
                "type": "realtime",
                "instructions": "You are a weather bot.",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Weather lookup.",
                        "parameters": get_weather.parameters,
                    }
                ],
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24_000},
                        "transcription": {"model": "gpt-4o-mini-transcribe", "language": "en"},
                        "turn_detection": {
                            "type": "semantic_vad",
                            "create_response": True,
                            "interrupt_response": True,
                        },
                        "noise_reduction": {"type": "near_field"},
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": 24_000},
                        "voice": "cedar",
                        "speed": 1.2,
                    },
                },
                "reasoning": {"effort": "low"},
                "max_output_tokens": 512,
                "tracing": "auto",
            }
            assert conn.session_id is not None and conn.session_id.startswith("sess_")
            assert [e["item"] for e in server.events("conversation.item.create")] == [
                {"type": "message", "role": "system",
                 "content": [{"type": "input_text", "text": "Be kind."}]},
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "Hi there"}]},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "Hello!"}]},
                {"type": "function_call", "call_id": "call_1", "name": "get_weather",
                 "arguments": '{"city": "Rome"}'},
                {"type": "function_call_output", "call_id": "call_1", "output": "rainy"},
            ]  # fmt: skip
            assert all(e["event_id"].startswith("evt_") for e in server.received)

            await conn.update(instructions="Now be brief.")
            await wait_for(lambda: len(server.events("session.update")) == 2)
            assert server.events("session.update")[1]["session"] == {
                "type": "realtime",
                "instructions": "Now be brief.",
            }
            await conn.update(tools=[])
            await wait_for(lambda: len(server.events("session.update")) == 3)
            assert server.events("session.update")[2]["session"] == {
                "type": "realtime",
                "tools": [],
            }
        assert not rec.of(EngineErrorEvent)


async def test_say_speaks_verbatim_and_reports_metrics() -> None:
    async with FakeRealtimeServer(replies=["The weather is fine."]) as server:
        engine = OpenAIRealtimeEngine(base_url=server.url, api_key=KEY)
        metrics: list[EngineMetrics] = []
        engine.on("metrics", metrics.append)
        async with connected(engine, EngineOptions(instructions="Be a pirate.")) as (conn, rec):
            await conn.say("Welcome aboard!")
            await rec.wait(lambda: rec.of(ResponseDone))
            assert server.events("response.create")[0]["response"] == {
                "instructions": VERBATIM_INSTRUCTIONS.format(text="Welcome aboard!"),
                "input": [],  # no context: best verbatim compliance
            }
            started, done = rec.of(ResponseStarted)[0], rec.of(ResponseDone)[0]
            assert started.response_id == done.response_id == server.responses[0].response_id
            assert rec.index(ResponseStarted) < rec.index(ResponseText) < rec.index(ResponseDone)
            assert spoken(rec) == "Welcome aboard!"
            frames = [a.frame for a in rec.of(ResponseAudio)]
            assert {f.sample_rate for f in frames} == {24_000}
            assert sum(f.duration for f in frames) == pytest.approx(1.0, abs=0.01)
            assert {a.item_id for a in rec.of(ResponseAudio)} == {rec.of(ResponseText)[0].item_id}
            assert done.status == "completed" and done.error is None
            assert done.usage == EngineUsage(
                input_text_tokens=120, input_audio_tokens=30, output_text_tokens=2,
                output_audio_tokens=20, cached_tokens=64,
            )  # fmt: skip
            m = metrics[0]
            assert (m.provider, m.model, m.response_id) == (
                "openai",
                engine.model,
                done.response_id,
            )
            assert m.ttfb is not None and 0 <= m.ttfb < 1.0 and m.duration >= m.ttfb
            assert m.output_audio_tokens == 20 and m.cached_tokens == 64 and not m.cancelled

            await conn.create_response(instructions="Mention the weather.")
            await rec.wait(lambda: len(rec.of(ResponseDone)) == 2)
            assert server.events("response.create")[1]["response"] == {
                "instructions": "Be a pirate.\n\nMention the weather."
            }
            assert spoken(rec, rec.of(ResponseDone)[1].response_id) == "The weather is fine."
            assert len(metrics) == 2


@pytest.mark.parametrize("speech_end", ["includes_silence", "speech_end"])
async def test_user_turn_with_server_vad(speech_end: str) -> None:
    server = FakeRealtimeServer(
        speech_end=speech_end,
        transcripts=["what time is it"],
        replies=["It is noon."],
    )
    async with server:
        engine = OpenAIRealtimeEngine(base_url=server.url, api_key=KEY, turn_detection="server_vad")
        metrics: list[EngineMetrics] = []
        engine.on("metrics", metrics.append)
        async with connected(engine) as (conn, rec):
            await user_turn(conn, speech=1.0, silence=0.8)
            await rec.wait(lambda: rec.of(ResponseDone))
            order = [InputSpeechStarted, InputSpeechStopped, InputCommitted, ResponseStarted,
                     ResponseDone]  # fmt: skip
            assert [rec.index(cls) for cls in order] == sorted(rec.index(cls) for cls in order)
            started, stopped = rec.of(InputSpeechStarted)[0], rec.of(InputSpeechStopped)[0]
            # speech ran from 0.3 s to 1.3 s of the input stream
            assert started.audio_time is not None and started.audio_time < 0.4
            assert stopped.audio_time == pytest.approx(1.3, abs=0.06)
            committed = rec.of(InputCommitted)[0]
            partials = [t for t in rec.of(InputTranscript) if not t.is_final]
            finals = [t for t in rec.of(InputTranscript) if t.is_final]
            assert [t.text for t in partials] == ["what ", "what time ", "what time is ",
                                                  "what time is it"]  # fmt: skip
            assert [(t.item_id, t.text) for t in finals] == [(committed.item_id, "what time is it")]
            assert all(t.item_id == committed.item_id for t in partials)
            assert spoken(rec) == "It is noon."
            assert metrics[0].ttfb is not None and metrics[0].ttfb < 0.5  # from the commit
            assert conn.audio_time_to_wall(stopped.audio_time) is not None


async def test_speech_end_refinement_can_be_disabled() -> None:
    async with FakeRealtimeServer() as server:
        engine = OpenAIRealtimeEngine(
            base_url=server.url, api_key=KEY, turn_detection="server_vad", refine_speech_end=False
        )
        async with connected(engine) as (conn, rec):
            await user_turn(conn, speech=1.0, silence=0.8)
            await rec.wait(lambda: rec.of(InputSpeechStopped))
            # raw audio_end_ms: speech end + 500 ms of silence confirmation
            assert rec.of(InputSpeechStopped)[0].audio_time == pytest.approx(1.8, abs=0.06)


@pytest.mark.parametrize("where", ["engine", "options"])
async def test_manual_turns_commit_and_respond(where: str) -> None:
    async with FakeRealtimeServer(transcripts=["push to talk"], replies=["Got it."]) as server:
        engine_td: Any = None if where == "engine" else "semantic_vad"
        engine = OpenAIRealtimeEngine(base_url=server.url, api_key=KEY, turn_detection=engine_td)
        options = EngineOptions(turn_detection=where == "engine")
        async with connected(engine, options) as (conn, rec):
            session = server.events("session.update")[0]["session"]
            assert session["audio"]["input"]["turn_detection"] is None
            await feed(conn, synth_speech(0.6, 16_000))
            await conn.commit_input()
            await rec.wait(lambda: rec.of(ResponseDone))
            control = [
                e["type"] for e in server.received if e["type"] != "input_audio_buffer.append"
            ]
            assert control[-2:] == ["input_audio_buffer.commit", "response.create"]
            assert not rec.of(InputSpeechStarted)
            committed = rec.of(InputCommitted)[0]
            final = next(t for t in rec.of(InputTranscript) if t.is_final)
            assert (final.item_id, final.text) == (committed.item_id, "push to talk")
            assert spoken(rec) == "Got it."
            await conn.clear_input()
            await wait_for(lambda: server.sent_events("input_audio_buffer.cleared"))


async def test_tool_call_round_trip() -> None:
    replies = [MockToolCall("get_weather", {"city": "Paris"}), "Sunny in Paris."]
    async with FakeRealtimeServer(replies=replies) as server:
        engine = OpenAIRealtimeEngine(base_url=server.url, api_key=KEY)
        async with connected(engine, EngineOptions(tools=[get_weather])) as (conn, rec):
            await conn.send_text("Weather in Paris?")
            await rec.wait(lambda: rec.of(ResponseDone))
            assert server.events("conversation.item.create")[0]["item"] == {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Weather in Paris?"}],
            }
            calls = rec.of(ResponseToolCall)
            assert len(calls) == 1  # arguments.done + output_item.done + response.done: once
            call, item = calls[0].call, server.sent_events("response.output_item.done")[-1]["item"]
            assert (call.name, call.parsed_arguments()) == ("get_weather", {"city": "Paris"})
            assert (call.call_id, call.id) == (item["call_id"], item["id"])
            assert calls[0].response_id == rec.of(ResponseDone)[0].response_id
            assert rec.index(ResponseToolCall) < rec.index(ResponseDone)

            await conn.send_tool_output(FunctionCallOutput(call_id=call.call_id, output="sunny"))
            await rec.wait(lambda: len(rec.of(ResponseDone)) == 2)
            control = [e for e in server.received if e["type"] != "input_audio_buffer.append"]
            assert control[-2]["item"] == {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": "sunny",
            }
            assert control[-1]["type"] == "response.create"
            assert spoken(rec) == "Sunny in Paris."
        assert not rec.of(EngineErrorEvent)


async def test_tool_calls_from_response_done_and_late_events() -> None:
    """Servers that only list calls in ``response.done``; late events are ignored."""
    async with FakeRealtimeServer() as server:
        engine = OpenAIRealtimeEngine(base_url=server.url, api_key=KEY)
        async with connected(engine) as (_, rec):
            call = {"type": "function_call", "id": "fc_1", "call_id": "call_9", "name": "lookup",
                    "arguments": ""}  # fmt: skip
            await server.push({"type": "response.done",
                               "response": {"id": "resp_x", "status": "completed",
                                            "output": [call]}})  # fmt: skip
            await server.push({"type": "response.output_audio.delta", "response_id": "resp_x",
                               "item_id": "i", "delta": "AAAA"})  # fmt: skip
            await server.push({"type": "rate_limits.updated", "rate_limits": []})
            await rec.wait(lambda: rec.of(ResponseDone))
            await asyncio.sleep(0.05)
            assert [s.response_id for s in rec.of(ResponseStarted)] == ["resp_x"]
            tool = rec.of(ResponseToolCall)
            assert len(tool) == 1 and tool[0].call.arguments == "{}"
            assert (tool[0].call.call_id, tool[0].call.id) == ("call_9", "fc_1")
            assert not rec.of(ResponseAudio)  # the late delta was dropped
            assert rec.of(ResponseDone)[0].usage is None
            failed = {"type": "failed", "error": {"type": "server_error", "code": "overloaded",
                                                  "message": "try again"}}  # fmt: skip
            await server.push({"type": "response.done",
                               "response": {"id": "resp_y", "status": "failed",
                                            "status_details": failed}})  # fmt: skip
            await rec.wait(lambda: len(rec.of(ResponseDone)) == 2)
            done = rec.of(ResponseDone)[1]
            assert (done.status, done.error) == ("failed", "try again")
            error = rec.of(EngineErrorEvent)[0]
            assert error.recoverable and "try again" in str(error.error)


async def test_barge_in_cancels_and_truncates_to_the_played_audio() -> None:
    long_answer = "This answer is long enough to keep the speaker busy for quite a while. " * 2
    async with FakeRealtimeServer(replies=[long_answer], realtime_factor=0.25) as server:
        engine = OpenAIRealtimeEngine(base_url=server.url, api_key=KEY)
        metrics: list[EngineMetrics] = []
        engine.on("metrics", metrics.append)
        async with connected(engine) as (conn, rec):
            await conn.create_response()
            await rec.wait(lambda: sum(a.frame.duration for a in rec.of(ResponseAudio)) >= 1.5)
            item_id = rec.of(ResponseAudio)[0].item_id
            assert await conn.interrupt(item_id, 1234) is None
            await rec.wait(lambda: rec.of(ResponseDone))
            await wait_for(lambda: server.truncations)
            response_id = rec.of(ResponseStarted)[0].response_id
            assert server.events("response.cancel")[0]["response_id"] == response_id
            truncate = server.events("conversation.item.truncate")[0]
            assert (truncate["item_id"], truncate["content_index"], truncate["audio_end_ms"]) == (
                item_id, 0, 1234,
            )  # fmt: skip
            assert server.truncations == [(item_id, 0, 1234)]  # accepted by the server
            done = rec.of(ResponseDone)[0]
            assert done.status == "cancelled"
            assert metrics[0].cancelled and metrics[0].ttfb is not None

            # interrupting after the response finished: no cancel, truncation still exact
            await conn.say("Short one.")
            await rec.wait(lambda: len(rec.of(ResponseDone)) == 2)
            second = rec.of(ResponseDone)[1].response_id
            item2 = next(a.item_id for a in rec.of(ResponseAudio) if a.response_id == second)
            await conn.interrupt(item2, 300)
            await wait_for(lambda: len(server.truncations) == 2)
            assert server.truncations[1] == (item2, 0, 300)
            await conn.truncate(item2, 99_999)  # longer than the audio: fully heard, nothing sent
            await asyncio.sleep(0.05)
            assert len(server.events("response.cancel")) == 1
            assert len(server.events("conversation.item.truncate")) == 2
        assert not rec.of(EngineErrorEvent)


async def test_server_errors_are_mapped_and_benign_races_ignored() -> None:
    async with FakeRealtimeServer() as server:
        engine = OpenAIRealtimeEngine(base_url=server.url, api_key=KEY)
        async with connected(engine) as (conn, rec):
            await conn._send({"type": "response.cancel"})  # nothing to cancel: benign race
            await server.push({"type": "error", "error": {"type": "invalid_request_error",
                               "code": "conversation_already_has_active_response",
                               "message": "busy"}})  # fmt: skip
            await server.push({"type": "error", "error": {"type": "rate_limit_error",
                               "code": "rate_limit_exceeded", "message": "slow down"}})  # fmt: skip
            await server.push({"type": "error", "error": {"type": "invalid_request_error",
                               "code": "invalid_value", "message": "bad voice",
                               "param": "session.audio.output.voice"}})  # fmt: skip
            await rec.wait(lambda: len(rec.of(EngineErrorEvent)) == 2)
            rate, invalid = rec.of(EngineErrorEvent)
            assert isinstance(rate.error, RateLimitError) and rate.recoverable
            assert type(invalid.error) is ProviderError and invalid.recoverable
            assert "bad voice" in str(invalid.error) and "session.audio.output.voice" in str(
                invalid.error
            )
            await server.push({"type": "conversation.item.input_audio_transcription.failed",
                               "item_id": "item_1", "content_index": 0,
                               "error": {"code": "audio_unintelligible",
                                         "message": "could not transcribe"}})  # fmt: skip
            await server.push({"type": "error", "error": {"type": "invalid_request_error",
                               "code": "invalid_api_key", "message": "bad key"}})  # fmt: skip
            await rec.wait(lambda: len(rec.of(EngineErrorEvent)) == 4)
            failed, auth = rec.of(EngineErrorEvent)[2:]
            assert failed.recoverable and "could not transcribe" in str(failed.error)
            assert isinstance(auth.error, AuthenticationError) and not auth.recoverable
            assert all("response_cancel_not_active" not in str(e.error)
                       for e in rec.of(EngineErrorEvent))  # fmt: skip


async def test_handshake_and_startup_errors() -> None:
    async with FakeRealtimeServer(api_key="right") as server:
        engine = OpenAIRealtimeEngine(base_url=server.url, api_key="wrong")
        with pytest.raises(AuthenticationError, match="401"):
            await engine.connect(EngineOptions())
    async with FakeRealtimeServer(reject_status=429) as server:
        engine = OpenAIRealtimeEngine(base_url=server.url, api_key=KEY)
        with pytest.raises(RateLimitError, match="429"):
            await engine.connect(EngineOptions())
    rejection = {"type": "invalid_request_error", "code": "invalid_value",
                 "message": "Invalid voice", "param": "session.audio.output.voice"}  # fmt: skip
    async with FakeRealtimeServer(reject_session_update=rejection) as server:
        engine = OpenAIRealtimeEngine(base_url=server.url, api_key=KEY)
        with pytest.raises(ProviderError, match="Invalid voice"):
            await engine.connect(EngineOptions())
        assert not engine._connections
    with socket.socket() as sock:  # a port nobody listens on
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    engine = OpenAIRealtimeEngine(base_url=f"ws://127.0.0.1:{port}/v1", api_key=KEY,
                                  connect_timeout=2)  # fmt: skip
    with pytest.raises(ProviderConnectionError):
        await engine.connect(EngineOptions())


async def test_reconnects_after_a_dropped_connection() -> None:
    replies = ["A fairly long reply that is still streaming when the network fails.", "Back."]
    async with FakeRealtimeServer(replies=replies, realtime_factor=1.0) as server:
        engine = OpenAIRealtimeEngine(base_url=server.url, api_key=KEY, reconnect_backoff=0.02,
                                      turn_detection="server_vad")  # fmt: skip
        async with connected(engine, EngineOptions(instructions="hi")) as (conn, rec):
            await feed(conn, AudioFrame.silence(0.5, 16_000))
            await conn.create_response()
            await rec.wait(lambda: rec.of(ResponseAudio))
            await server.drop()
            await rec.wait(lambda: [s.status for s in rec.of(EngineStatus)][-1:] == ["reconnected"])
            assert [s.status for s in rec.of(EngineStatus)] == ["reconnecting", "reconnected"]
            failed = rec.of(ResponseDone)[0]
            assert failed.status == "failed" and "connection lost" in (failed.error or "")
            assert len(server.handshakes) == 2
            updates = server.events("session.update")
            assert len(updates) == 2 and updates[1]["session"]["instructions"] == "hi"
            await server.connections[-1].ws.send("not json")  # ignored
            # the new provider session works and its audio clock is mapped onto ours
            await user_turn(conn, speech=1.0, silence=0.8)
            await rec.wait(lambda: len(rec.of(ResponseDone)) == 2, timeout=8)
            assert rec.of(ResponseDone)[1].status == "completed"
            # input stream: 0.5 s before the drop + 0.3 s silence + 1.0 s speech
            assert rec.of(InputSpeechStopped)[0].audio_time == pytest.approx(1.8, abs=0.06)
        assert not rec.of(EngineErrorEvent)


async def test_gives_up_when_the_server_is_gone() -> None:
    server = await FakeRealtimeServer().start()
    engine = OpenAIRealtimeEngine(base_url=server.url, api_key=KEY, max_reconnect_attempts=2,
                                  reconnect_backoff=0.01, connect_timeout=1)  # fmt: skip
    conn = await engine.connect(EngineOptions())
    rec = Collector(conn)
    await server.aclose()  # closes the session and stops listening
    await asyncio.wait_for(rec.task, 5)  # events() ends once the connection gives up
    assert [s.status for s in rec.of(EngineStatus)] == ["reconnecting"]
    error = rec.of(EngineErrorEvent)[-1]
    assert isinstance(error.error, ProviderConnectionError) and not error.recoverable
    assert conn.closed and not engine._connections


async def test_session_expiry_notice() -> None:
    async with FakeRealtimeServer(expires_in=1.5) as server:
        engine = OpenAIRealtimeEngine(base_url=server.url, api_key=KEY, expiry_warning=1.0)
        async with connected(engine) as (_, rec):
            await rec.wait(lambda: rec.of(EngineStatus), timeout=3)
            status = rec.of(EngineStatus)[0]
            assert status.status == "expiring"
            assert status.time_left is not None and 0.2 < status.time_left <= 1.5


# ------------------------------------------------------------------ compatibility profiles
async def test_qwen_profile_speaks_the_beta_dialect() -> None:
    long_answer = "Here is a longer answer that will be interrupted before it is finished. " * 2
    server = FakeRealtimeServer(
        dialect="beta",
        transcripts=["how is the weather"],
        replies=["Sunny and warm.", long_answer],
        realtime_factor=0.25,
    )
    async with server:
        engine = QwenOmniRealtimeEngine(base_url=server.url, api_key=KEY, temperature=0.7)
        options = EngineOptions(instructions="You are helpful.", tools=[get_weather])
        async with connected(engine, options) as (conn, rec):
            assert server.handshakes[0].query == {"model": "qwen3.8-omni-flash-realtime"}
            assert server.events("session.update")[0]["session"] == {
                "instructions": "You are helpful.",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Weather lookup.",
                            "parameters": get_weather.parameters,
                        },
                    }
                ],
                "modalities": ["text", "audio"],
                "input_audio_format": "pcm",
                "output_audio_format": "pcm",
                "turn_detection": {
                    "type": "server_vad",
                    "create_response": True,
                    "interrupt_response": True,
                },
                "input_audio_transcription": {"model": "qwen3-asr-flash-realtime"},
                "temperature": 0.7,
            }
            await user_turn(conn, speech=1.0, silence=0.8)
            await rec.wait(lambda: rec.of(ResponseDone))
            assert server.connections[0].input_rate() == 16_000
            assert server.connections[0].position == pytest.approx(2.1, abs=0.01)  # 16 kHz PCM
            partials = [t.text for t in rec.of(InputTranscript) if not t.is_final]
            assert partials == ["how ", "how is ", "how is the ", "how is the weather"]
            assert [t.text for t in rec.of(InputTranscript) if t.is_final] == ["how is the weather"]
            assert spoken(rec) == "Sunny and warm."  # response.audio_transcript.delta
            assert sum(a.frame.duration for a in rec.of(ResponseAudio)) == pytest.approx(1.0, 0.02)
            usage = rec.of(ResponseDone)[0].usage
            assert usage is not None and usage.output_audio_tokens == 20  # *_tokens_details

            # no per-response instructions: the session prompt is patched, then restored
            await conn.say("Welcome back!")
            await rec.wait(lambda: len(rec.of(ResponseDone)) == 2)
            assert spoken(rec, rec.of(ResponseDone)[1].response_id) == "Welcome back!"
            assert "response" not in server.events("response.create")[-1]
            await wait_for(lambda: len(server.events("session.update")) == 3)
            patched, restored = (e["session"] for e in server.events("session.update")[1:])
            assert patched == {"instructions": VERBATIM_INSTRUCTIONS.format(text="Welcome back!")}
            assert restored == {"instructions": "You are helpful."}

            # barge-in: response.cancel without response_id, never a truncate
            await conn.create_response()
            await rec.wait(lambda: len(rec.of(ResponseStarted)) == 3)
            third = rec.of(ResponseStarted)[2].response_id
            await rec.wait(lambda: any(a.response_id == third for a in rec.of(ResponseAudio)))
            await conn.interrupt(rec.of(ResponseAudio)[-1].item_id, 200)
            await rec.wait(lambda: len(rec.of(ResponseDone)) == 3)
            assert rec.of(ResponseDone)[2].status == "cancelled"
            assert server.events("response.cancel") == [
                {"event_id": server.events("response.cancel")[0]["event_id"],
                 "type": "response.cancel"}
            ]  # fmt: skip
            assert not server.events("conversation.item.truncate")
            with pytest.raises(EngineError, match="no text"):
                await conn.send_text("typed input")
        assert not rec.of(EngineErrorEvent)


async def test_xai_profile_cumulative_transcripts_and_force_message() -> None:
    server = FakeRealtimeServer(dialect="xai", transcripts=["tell me a joke"], replies=["Why not?"])
    async with server:
        engine = XAIRealtimeEngine(base_url=server.url, api_key=KEY)
        options = EngineOptions(instructions="Be funny.", language="es-MX")  # BCP-47 kept
        async with connected(engine, options) as (conn, rec):
            assert server.events("session.update")[0]["session"] == {
                "instructions": "Be funny.",
                "tools": [],
                "voice": "eve",
                "turn_detection": {"type": "server_vad"},
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24_000},
                        "transcription": {"model": "grok-transcribe", "language_hint": "es-MX"},
                    },
                    "output": {"format": {"type": "audio/pcm", "rate": 24_000}},
                },
            }
            await conn.say("Welcome to Grok!")  # xAI force_message: truly verbatim
            await rec.wait(lambda: rec.of(ResponseDone))
            assert server.events("conversation.item.create")[0]["item"] == {
                "type": "force_message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Welcome to Grok!"}],
            }
            assert not server.events("response.create")
            assert spoken(rec) == "Welcome to Grok!"
            await user_turn(conn, speech=1.0, silence=0.8)
            await rec.wait(lambda: len(rec.of(ResponseDone)) == 2)
            partials = [t.text for t in rec.of(InputTranscript) if not t.is_final]
            assert partials == ["tell ", "tell me ", "tell me a ", "tell me a joke"]
            assert spoken(rec, rec.of(ResponseDone)[1].response_id) == "Why not?"
        assert not rec.of(EngineErrorEvent)


async def test_speaches_profile_without_cancel_or_truncate() -> None:
    answer = "A slow local pipeline keeps on talking for a while."  # 3.4 s of audio
    server = FakeRealtimeServer(dialect="beta", supports_cancel=False, supports_truncate=False,
                                replies=[answer, "Done."], realtime_factor=0.25)  # fmt: skip
    async with server:
        engine = SpeachesRealtimeEngine(base_url=server.url)  # no API key needed
        assert not engine.capabilities.truncation
        async with connected(engine) as (conn, rec):
            session = server.events("session.update")[0]["session"]
            assert session["input_audio_format"] == "pcm16" and "voice" not in session
            assert "authorization" not in server.handshakes[0].headers
            await conn.create_response()
            await rec.wait(lambda: rec.of(ResponseAudio))
            await conn.interrupt(rec.of(ResponseAudio)[0].item_id, 500)
            # a new response waits for the uncancellable one instead of being rejected
            await conn.create_response()
            await rec.wait(lambda: len(rec.of(ResponseDone)) == 2)
            assert [d.status for d in rec.of(ResponseDone)] == ["completed", "completed"]
            assert spoken(rec, rec.of(ResponseDone)[1].response_id) == "Done."
            assert not server.events("response.cancel")
            assert not server.events("conversation.item.truncate")
        assert not server.sent_events("error") and not rec.of(EngineErrorEvent)


async def test_azure_openai_url_auth_and_deployment() -> None:
    async with FakeRealtimeServer(api_key="azure-key") as server:
        endpoint = f"http://127.0.0.1:{server.port}"
        engine = AzureOpenAIRealtimeEngine(
            endpoint=endpoint, deployment="my-rt", api_key="azure-key"
        )
        async with connected(engine):
            handshake = server.handshakes[0]
            assert handshake.path == "/openai/v1/realtime" and handshake.query == {"model": "my-rt"}
            assert handshake.headers["api-key"] == "azure-key"
            assert "authorization" not in handshake.headers
            session = server.events("session.update")[0]["session"]
            assert session["audio"]["input"]["transcription"] == {"model": "whisper-1"}
    async with FakeRealtimeServer(api_key="entra-token") as server:
        endpoint = f"http://127.0.0.1:{server.port}"
        engine = AzureOpenAIRealtimeEngine(endpoint=endpoint, azure_ad_token="entra-token")
        async with connected(engine):
            assert server.handshakes[0].headers["authorization"] == "Bearer entra-token"
            assert server.handshakes[0].query == {"model": "gpt-realtime-2.1"}


# ------------------------------------------------------------------ end to end (session)
async def test_agent_session_end_to_end() -> None:
    calls: list[str] = []

    @function_tool
    async def weather(city: str) -> str:
        """Weather lookup."""
        calls.append(city)
        return f"sunny in {city}"

    story = "Once upon a time a voice agent kept talking about the weather all day long. " * 2
    server = FakeRealtimeServer(
        transcripts=["what is the weather in paris", "tell me a story"],
        replies=[MockToolCall("weather", {"city": "Paris"}), "It is sunny in Paris.", story],
        realtime_factor=0.5,
    )
    async with server:
        engine = OpenAIRealtimeEngine(base_url=server.url, api_key=KEY, turn_detection="server_vad")
        session = AgentSession(engine)
        events: dict[str, list[Any]] = {}
        for name in ("user_transcript", "agent_transcript", "tool_result", "interrupted",
                     "metrics", "error"):  # fmt: skip
            session.on(name, lambda ev, name=name: events.setdefault(name, []).append(ev))
        transport = LoopbackTransport(realtime_playout=True)
        agent = Agent("You are a weather bot.", tools=[weather], greeting="Welcome!")
        await session.start(agent, transport)

        def turns() -> list[TurnMetrics]:
            return [m for m in events.get("metrics", []) if isinstance(m, TurnMetrics)]

        # 1. greeting via say()
        await wait_for(
            lambda: any("Welcome!" in e.delta for e in events.get("agent_transcript", []))
        )
        await wait_for(lambda: session.agent_state == AgentState.LISTENING, timeout=5)
        # 2. user turn (server VAD) -> tool call round trip -> spoken answer
        await transport.play_user_audio(synth_speech(0.8, 16_000), realtime=False)
        await transport.play_user_audio(AudioFrame.silence(0.7, 16_000), realtime=False)
        await wait_for(lambda: len(turns()) == 1, timeout=8)
        # 3. a long answer, interrupted after about a second of playback
        await transport.play_user_audio(synth_speech(0.8, 16_000), realtime=False)
        await transport.play_user_audio(AudioFrame.silence(0.7, 16_000), realtime=False)
        await wait_for(lambda: session.agent_state == AgentState.SPEAKING, timeout=5)
        await asyncio.sleep(1.0)
        await transport.play_user_audio(synth_speech(0.4, 16_000), realtime=False)
        await wait_for(lambda: events.get("interrupted"))
        await wait_for(lambda: server.truncations)
        await session.aclose()

    greeting = server.responses[0]
    assert greeting.text == "Welcome!" and greeting.body["input"] == []
    finals = [e.text for e in events["user_transcript"] if e.is_final]
    assert finals[:2] == ["what is the weather in paris", "tell me a story"]
    assert calls == ["Paris"]
    call_item = server.sent_events("response.output_item.done")[1]["item"]  # after the greeting
    outputs = [e["item"] for e in server.events("conversation.item.create")]
    assert outputs == [
        {
            "type": "function_call_output",
            "call_id": call_item["call_id"],
            "output": "sunny in Paris",
        }
    ]
    kinds = [getattr(i, "role", i.type) for i in session.history.items]
    assert kinds[:5] == ["assistant", "user", "function_call", "function_call_output", "assistant"]
    assert session.history.items[4].text == "It is sunny in Paris."  # type: ignore[union-attr]

    # barge-in: the engine truncated the item to exactly what the user heard
    interrupted = events["interrupted"][0]
    item_id, content_index, end_ms = server.truncations[0]
    assert (item_id, content_index) == (interrupted.item_id, 0)
    assert end_ms == pytest.approx(interrupted.played * 1000, abs=1)
    assert 0.5 < interrupted.played < 3.0
    story_msg = next(i for i in session.history.items if getattr(i, "id", None) == item_id)
    assert isinstance(story_msg, ChatMessage) and story_msg.interrupted
    assert 0 < len(story_msg.text) < len(story.strip())

    engine_metrics = [m for m in events["metrics"] if isinstance(m, EngineMetrics)]
    assert engine_metrics and all(m.provider == "openai" for m in engine_metrics)
    assert server.responses[3].status == "cancelled"  # the server VAD / our cancel stopped it
    assert any(m.cancelled for m in engine_metrics)
    assert session.usage.engine_output_audio_tokens > 0
    turn = turns()[0]
    assert turn.tool_calls == 1 and turn.voice_to_voice is not None and not turn.interrupted
    assert not events.get("error")


# ------------------------------------------------------------------------- real API
@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")
async def test_openai_realtime_live_api() -> None:
    engine = OpenAIRealtimeEngine(model=os.environ.get("OPENAI_REALTIME_MODEL") or None)
    async with connected(engine, EngineOptions(instructions="You are terse.")) as (conn, rec):
        await conn.say("Hello from the integration test.")
        await rec.wait(lambda: rec.of(ResponseDone), timeout=30)
        done = rec.of(ResponseDone)[0]
        assert done.status == "completed", done.error
        assert sum(a.frame.duration for a in rec.of(ResponseAudio)) > 0.5
        assert "hello" in spoken(rec).lower()
        assert done.usage is not None and done.usage.output_audio_tokens > 0
        await conn.send_text("Reply with exactly one word: ready")
        await rec.wait(lambda: len(rec.of(ResponseDone)) == 2, timeout=30)
        assert "ready" in spoken(rec, rec.of(ResponseDone)[1].response_id).lower()
    assert not [e for e in rec.of(EngineErrorEvent) if not e.recoverable]
