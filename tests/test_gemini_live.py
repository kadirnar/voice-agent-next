"""Gemini Live engine (``providers/google/live.py``) against the fake BidiGenerateContent server.

Everything runs offline: :class:`FakeGeminiLiveServer` speaks the real JSON protocol on
``127.0.0.1``. The real-API test at the end is marked ``integration``.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from voice_agent_next import Agent, AgentSession, AgentState, ChatMessage, function_tool
from voice_agent_next.audio import AudioFrame
from voice_agent_next.chat import FunctionCallOutput
from voice_agent_next.engine import EngineOptions
from voice_agent_next.engines.rotation import SummarizeHistory
from voice_agent_next.errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderError,
    RateLimitError,
)
from voice_agent_next.events import (
    EngineEvent,
    EngineStatus,
    InputCommitted,
    InputSpeechStarted,
    InputSpeechStopped,
    InputTranscript,
    ResponseAudio,
    ResponseDone,
    ResponseStarted,
    ResponseText,
    ResponseToolCall,
    ToolCallCancelled,
)
from voice_agent_next.metrics import EngineMetrics, RotationMetrics, TurnMetrics
from voice_agent_next.providers.google.live import GeminiLiveConnection, GeminiLiveEngine
from voice_agent_next.providers.mock import MockLLM, synth_speech
from voice_agent_next.testing.gemini_live import FakeGeminiLiveServer, FakeReply, FakeToolCall
from voice_agent_next.transports import LoopbackTransport

KEY = "fake-gemini-key"


# ----------------------------------------------------------------------------- helpers
@pytest.fixture
async def fake() -> AsyncIterator[Callable[..., Any]]:
    """Factory starting fake servers that are closed (and checked) after the test."""
    servers: list[FakeGeminiLiveServer] = []

    async def make(**kw: Any) -> FakeGeminiLiveServer:
        server = FakeGeminiLiveServer(**kw)
        await server.start()
        servers.append(server)
        return server

    yield make
    for server in servers:
        await server.aclose()
        assert server.errors == [], f"protocol violations: {server.errors}"


def engine_for(server: FakeGeminiLiveServer, **kw: Any) -> GeminiLiveEngine:
    kw.setdefault("rotate_after", None)
    return GeminiLiveEngine(api_key=KEY, base_url=server.url, **kw)


class Events:
    """Collects the events of an engine connection in the background."""

    def __init__(self, conn: GeminiLiveConnection) -> None:
        self.items: list[EngineEvent] = []
        self._task = asyncio.create_task(self._pump(conn))

    async def _pump(self, conn: GeminiLiveConnection) -> None:
        async for ev in conn.events():
            self.items.append(ev)

    def of(self, kind: type[Any]) -> list[Any]:
        return [e for e in self.items if isinstance(e, kind)]

    def kinds(self) -> list[str]:
        return [type(e).__name__ for e in self.items if not isinstance(e, ResponseAudio)]

    async def wait(self, predicate: Callable[[], bool], timeout: float = 5.0) -> None:
        await wait_for(predicate, timeout)


async def wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), timeout)


async def connect(
    server: FakeGeminiLiveServer, options: EngineOptions | None = None, **kw: Any
) -> tuple[GeminiLiveConnection, Events]:
    engine = engine_for(server, **kw)
    conn = await engine.connect(options or EngineOptions(instructions="be brief"))
    assert isinstance(conn, GeminiLiveConnection)
    return conn, Events(conn)


async def say(conn: GeminiLiveConnection, speech: float = 0.8, silence: float = 0.6) -> None:
    """The user speaks (0.3 s lead-in silence), pushed in 20 ms frames, faster than real time."""
    audio = AudioFrame.concat(
        [
            AudioFrame.silence(0.3, 16_000),
            synth_speech(speech, 16_000),
            AudioFrame.silence(silence, 16_000),
        ]
    )
    await push(conn, audio)


async def push(conn: GeminiLiveConnection, audio: AudioFrame, chunk: float = 0.02) -> None:
    step = round(chunk * audio.sample_rate) * 2
    for i in range(0, len(audio.data), step):
        await conn.send_audio(AudioFrame(audio.data[i : i + step], audio.sample_rate))
        await asyncio.sleep(0)


# ------------------------------------------------------------------ registry & config
def test_registry_alias_and_capabilities() -> None:
    from voice_agent_next import create
    from voice_agent_next.registry import get_provider

    engine = create("engine", "google/gemini-3.8-live", api_key="k")
    assert isinstance(engine, GeminiLiveEngine) and engine.model == "gemini-3.8-live"
    assert isinstance(create("engine", "gemini", api_key="k"), GeminiLiveEngine)
    caps = engine.capabilities
    assert (engine.input_sample_rate, engine.output_sample_rate) == (16_000, 24_000)
    assert caps.native_audio and caps.server_turn_detection and not caps.truncation
    assert caps.tool_mode == "non_blocking" and caps.max_session_duration == 600.0
    spec = get_provider("engine", "google")
    assert spec.default_model == "gemini-3.8-live" and "GOOGLE_API_KEY" in spec.env
    assert not spec.local and spec.requires == ("websockets",)
    # models without asynchronous function calling default to blocking tools
    legacy = GeminiLiveEngine(model="gemini-3.1-flash-live-preview", api_key="k")
    assert legacy.capabilities.tool_mode == "blocking"
    with pytest.raises(ConfigurationError):
        GeminiLiveEngine(tool_behavior="sometimes")  # type: ignore[arg-type]


def test_gemini_alias_resolves_on_a_cold_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from voice_agent_next import registry

    monkeypatch.delitem(registry._ALIASES, "gemini", raising=False)
    monkeypatch.delitem(sys.modules, "voice_agent_next.providers.gemini", raising=False)
    assert registry.get_provider("engine", "gemini").factory is GeminiLiveEngine


async def test_missing_api_key_is_a_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ConfigurationError, match="GOOGLE_API_KEY"):
        await GeminiLiveEngine().connect(EngineOptions())


async def test_setup_message_and_auth(fake: Callable[..., Any]) -> None:
    @function_tool
    async def get_weather(city: str, unit: str = "celsius") -> str:
        """Look up the weather."""
        return "sunny"

    server = await fake()
    options = EngineOptions(
        instructions="You are terse.", tools=[get_weather], voice="Kore", language="en-US",
        temperature=0.4, extra={"generationConfig": {"maxOutputTokens": 256}},
    )  # fmt: skip
    conn, _ = await connect(server, options, vad={"silence_duration_ms": 500})
    await conn.aclose()

    conn_info = server.connections[0]
    assert conn_info.headers["x-goog-api-key"] == KEY and "key=" not in conn_info.path
    assert conn_info.path.endswith("v1beta.GenerativeService.BidiGenerateContent")
    setup = server.setups[0]
    assert setup["model"] == "models/gemini-3.8-live"
    gen = setup["generationConfig"]
    assert gen["responseModalities"] == ["AUDIO"] and gen["temperature"] == 0.4
    assert gen["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Kore"
    assert gen["maxOutputTokens"] == 256  # EngineOptions.extra is merged into setup
    assert setup["systemInstruction"] == {"parts": [{"text": "You are terse."}]}
    (decl,) = setup["tools"][0]["functionDeclarations"]
    assert decl["name"] == "get_weather" and decl["behavior"] == "NON_BLOCKING"
    assert decl["parametersJsonSchema"]["required"] == ["city"]
    assert setup["realtimeInputConfig"]["automaticActivityDetection"] == {"silenceDurationMs": 500}
    assert setup["inputAudioTranscription"] == {"languageCodes": ["en-US"]}
    assert setup["outputAudioTranscription"] == {}
    assert setup["sessionResumption"] == {}
    assert setup["contextWindowCompression"] == {"slidingWindow": {}}
    assert "historyConfig" not in setup


# --------------------------------------------------------------------------- turns
async def test_audio_turn_event_sequence(fake: Callable[..., Any]) -> None:
    server = await fake(replies=["Hi! Nice to meet you."], transcripts=["hello gemini"])
    metrics: list[EngineMetrics] = []
    conn, events = await connect(server)
    conn.engine.on("metrics", metrics.append)
    await say(conn)
    await events.wait(lambda: bool(events.of(ResponseDone)))
    await wait_for(lambda: bool(metrics))  # emitted when the server turn completes
    await conn.aclose()

    kinds = events.kinds()
    assert kinds[0] == "InputSpeechStarted"
    first = kinds.index
    assert first("InputSpeechStopped") < first("InputCommitted") < first("ResponseStarted")
    assert kinds[-1] == "ResponseDone"
    started, stopped = events.of(InputSpeechStarted)[0], events.of(InputSpeechStopped)[0]
    assert started.audio_time == pytest.approx(0.3, abs=0.1)  # speech onset in the stream
    assert stopped.audio_time == pytest.approx(1.1, abs=0.1)  # where speech ended
    partials = [e.text for e in events.of(InputTranscript) if not e.is_final]
    finals = [e for e in events.of(InputTranscript) if e.is_final]
    assert partials == ["hello", "hello gemini"]
    assert [f.text for f in finals] == ["hello gemini"]
    assert finals[0].item_id == events.of(InputCommitted)[0].item_id
    assert kinds.index("InputTranscript") < first("ResponseStarted")  # final before the reply
    audio = events.of(ResponseAudio)
    assert audio and all(a.frame.sample_rate == 24_000 for a in audio)
    assert sum(a.frame.duration for a in audio) == pytest.approx(len("Hi! Nice to meet you.") / 15, abs=0.05)  # fmt: skip
    assert "".join(e.delta for e in events.of(ResponseText)) == "Hi! Nice to meet you."
    done = events.of(ResponseDone)[0]
    assert done.status == "completed"
    (m,) = metrics
    assert m.response_id == done.response_id and m.ttfb is not None and not m.cancelled
    assert m.output_audio_tokens == round(len("Hi! Nice to meet you.") / 15 * 25)
    assert (m.input_text_tokens, m.input_audio_tokens) == (100, 20)
    assert conn.chat_ctx.messages()[-2].text == "hello gemini"
    assert conn.chat_ctx.messages()[-1].text == "Hi! Nice to meet you."
    # the user audio reached the server untouched (16 kHz PCM)
    assert len(server.connections[0].audio) == round(1.7 * 16_000) * 2


async def test_late_input_transcript_still_precedes_the_answer(fake: Callable[..., Any]) -> None:
    server = await fake(replies=["Sure thing."], transcripts=["book a table"], late_transcription=True)  # fmt: skip
    conn, events = await connect(server)
    await say(conn)
    await events.wait(lambda: bool(events.of(ResponseDone)))
    await conn.aclose()

    kinds = events.kinds()
    finals = [e.text for e in events.of(InputTranscript) if e.is_final]
    assert finals == ["book a table"]  # accumulated while the answer's text was held back
    assert kinds.index("InputCommitted") < kinds.index("ResponseStarted")
    assert kinds.index("InputTranscript") < kinds.index("ResponseText")
    assert "".join(e.delta for e in events.of(ResponseText)) == "Sure thing."


async def test_interim_then_final_input_transcripts(fake: Callable[..., Any]) -> None:
    server = await fake(replies=["Okay."], transcripts=["book a table"], interim_transcription=True)
    conn, events = await connect(server)
    await say(conn)
    await events.wait(lambda: bool(events.of(ResponseDone)))
    await conn.aclose()

    partials = [e.text for e in events.of(InputTranscript) if not e.is_final]
    assert partials[:3] == ["book", "book a", "book a table"]  # interim hypotheses
    assert partials[-1] == "book a table"
    assert [e.text for e in events.of(InputTranscript) if e.is_final] == ["book a table"]
    assert len({e.item_id for e in events.of(InputTranscript)}) == 1


async def test_without_voice_activity_the_speech_end_is_estimated_locally(
    fake: Callable[..., Any],
) -> None:
    server = await fake(replies=["Okay."], transcripts=["hello gemini"], voice_activity=False)
    conn, events = await connect(server)
    await say(conn)
    await events.wait(lambda: bool(events.of(ResponseDone)))
    await conn.aclose()

    assert not events.of(InputSpeechStarted)  # the server never signalled speech start
    kinds = events.kinds()
    assert kinds[:2] == ["InputTranscript", "InputTranscript"]  # partials
    (stopped,) = events.of(InputSpeechStopped)
    assert stopped.audio_time == pytest.approx(1.1, abs=0.1)
    assert kinds.index("InputSpeechStopped") < kinds.index("InputCommitted")


async def test_text_input_and_requested_responses(fake: Callable[..., Any]) -> None:
    server = await fake(replies=["Paris.", "Hello there."])
    conn, events = await connect(server)
    await conn.send_text("What is the capital of France?")
    await events.wait(lambda: len(events.of(ResponseDone)) == 1)
    await conn.say("Welcome aboard.")
    await events.wait(lambda: len(events.of(ResponseDone)) == 2)
    await conn.send_text("Remember my name is Ada.", respond=False)
    await conn.create_response()
    await events.wait(lambda: len(events.of(ResponseDone)) == 3)
    await conn.aclose()

    assert not events.of(InputCommitted)  # client-requested responses are not user turns
    texts = ["".join(e.delta for e in events.of(ResponseText) if e.response_id == d.response_id)
             for d in events.of(ResponseDone)]  # fmt: skip
    assert texts == ["Paris.", "Welcome aboard.", "Hello there."]
    contents = server.connection.client_contents
    assert contents[0] == {
        "turns": [{"role": "user", "parts": [{"text": "What is the capital of France?"}]}],
        "turnComplete": True,
    }
    assert "verbatim" in contents[1]["turns"][0]["parts"][0]["text"]
    assert contents[2]["turnComplete"] is False
    assert contents[3] == {"turns": [], "turnComplete": True}


# --------------------------------------------------------------------------- tools
@function_tool
async def get_weather(city: str) -> str:
    """Look up the weather in a city."""
    return f"sunny in {city}"


@function_tool
async def lookup(q: str) -> str:
    """Search the knowledge base."""
    return "found"


TOOLS = EngineOptions(tools=[get_weather, lookup])


async def test_non_blocking_tool_call_round_trip(fake: Callable[..., Any]) -> None:
    server = await fake(
        replies=[FakeReply("Let me check.", [FakeToolCall("get_weather", {"city": "Paris"})]),
                 "It is sunny in Paris."],
    )  # fmt: skip
    conn, events = await connect(server, TOOLS)
    await say(conn)
    await events.wait(lambda: bool(events.of(ResponseToolCall)))
    (call_ev,) = events.of(ResponseToolCall)
    assert call_ev.call.name == "get_weather" and json.loads(call_ev.call.arguments) == {"city": "Paris"}  # fmt: skip
    # the response ends with the tool call so the tools can run right away
    kinds = events.kinds()
    assert kinds[kinds.index("ResponseToolCall") + 1] == "ResponseDone"
    output = FunctionCallOutput(call_id=call_ev.call.call_id, output='{"sky": "clear"}', name="get_weather")  # fmt: skip
    await conn.send_tool_output(output)
    await events.wait(lambda: any("sunny" in e.delta for e in events.of(ResponseText)))
    await events.wait(lambda: len(events.of(ResponseDone)) == 3)
    await conn.aclose()

    (response,) = server.connection.tool_responses
    assert response == {"id": call_ev.call.call_id, "name": "get_weather",
                        "response": {"result": {"sky": "clear"}}, "scheduling": "WHEN_IDLE"}  # fmt: skip
    assert len(events.of(InputCommitted)) == 1  # the follow-up answer is not a user turn
    spoken = ["".join(e.delta for e in events.of(ResponseText) if e.response_id == d.response_id)
              for d in events.of(ResponseDone)]  # fmt: skip
    assert spoken == ["", "Let me check.", "It is sunny in Paris."]


async def test_silent_tool_output_and_errors(fake: Callable[..., Any]) -> None:
    server = await fake(replies=[FakeToolCall("lookup", {"q": "x"})])
    conn, events = await connect(server, TOOLS)
    await say(conn)
    await events.wait(lambda: bool(events.of(ResponseToolCall)))
    call = events.of(ResponseToolCall)[0].call
    await conn.send_tool_output(FunctionCallOutput(call_id=call.call_id, output="boom", is_error=True), respond=False)  # fmt: skip
    await asyncio.sleep(0.2)
    await conn.aclose()
    (response,) = server.connection.tool_responses
    assert response["response"] == {"error": "boom"} and response["name"] == "lookup"
    assert response["scheduling"] == "SILENT"
    assert len(events.of(ResponseStarted)) == 1  # SILENT: no follow-up response


async def test_blocking_tools(fake: Callable[..., Any]) -> None:
    server = await fake(replies=[FakeToolCall("get_time"), "It is noon."])

    @function_tool
    async def get_time() -> str:
        """Current time."""
        return "12:00"

    conn, events = await connect(server, EngineOptions(tools=[get_time]), tool_behavior="blocking")
    assert conn.capabilities.tool_mode == "blocking"
    await say(conn)
    await events.wait(lambda: bool(events.of(ResponseToolCall)))
    call = events.of(ResponseToolCall)[0].call
    await events.wait(lambda: bool(events.of(ResponseDone)))  # ends with the call (no deadlock)
    await conn.send_tool_output(FunctionCallOutput(call_id=call.call_id, output="12:00"))
    await events.wait(lambda: any("noon" in e.delta for e in events.of(ResponseText)))
    await events.wait(lambda: len(events.of(ResponseDone)) == 2)
    await conn.aclose()

    (decl,) = server.setups[0]["tools"][0]["functionDeclarations"]
    assert "behavior" not in decl and "parametersJsonSchema" not in decl  # no-arg tool
    (response,) = server.connection.tool_responses
    assert "scheduling" not in response  # only meaningful for NON_BLOCKING calls
    assert len(events.of(InputCommitted)) == 1


async def test_barge_in_cancels_pending_tool_calls(fake: Callable[..., Any]) -> None:
    filler = "Let me look that up for you, this could take a little while, please hold on."
    server = await fake(
        replies=[FakeReply(filler, [FakeToolCall("lookup", {"q": "x"})])], realtime_factor=1.0
    )
    conn, events = await connect(server, TOOLS)
    await say(conn)
    await events.wait(lambda: len(events.of(ResponseAudio)) > 5)  # the filler is streaming
    call = events.of(ResponseToolCall)[0].call
    await push(conn, synth_speech(0.4, 16_000))  # the user talks over the model
    await events.wait(lambda: bool(events.of(ToolCallCancelled)))
    await conn.aclose()

    assert events.of(ToolCallCancelled)[0].call_ids == [call.call_id]
    assert server.connection.interruptions == 1
    starts = events.of(InputSpeechStarted)
    assert len(starts) == 2  # the first turn + the barge-in (server VAD)
    assert events.of(ResponseDone)[-1].status == "cancelled"
    assert conn.chat_ctx.messages()[-1].interrupted  # the filler was cut off


async def test_server_side_interruption_without_voice_activity(fake: Callable[..., Any]) -> None:
    server = await fake(replies=["A fairly long answer that goes on and on for a while."],
                        realtime_factor=1.0, voice_activity=False)  # fmt: skip
    conn, events = await connect(server)
    await say(conn)
    await events.wait(lambda: len(events.of(ResponseAudio)) > 5)
    await push(conn, synth_speech(0.4, 16_000))
    await events.wait(lambda: events.of(ResponseDone) != [])
    await conn.aclose()

    (started,) = events.of(InputSpeechStarted)  # only `interrupted` signals the barge-in
    assert started.audio_time is None  # (no voiceActivity offset to report)
    kinds = events.kinds()
    assert kinds.index("InputSpeechStarted") < kinds.index("ResponseDone")
    assert events.of(ResponseDone)[0].status == "cancelled"


# ---------------------------------------------------------------- session rotation
def quiet_noise(seconds: float, seed: int = 1) -> list[AudioFrame]:
    """Low-level random noise (below any VAD threshold) in 20 ms frames: every sample is
    distinctive, so lost or reordered audio would be detected byte for byte."""
    import numpy as np

    rng = np.random.default_rng(seed)
    data = rng.integers(-60, 60, round(seconds * 16_000), dtype=np.int16)
    return [AudioFrame(data[i : i + 320].tobytes(), 16_000) for i in range(0, len(data), 320)]


async def sleep_at_least(seconds: float) -> None:
    """``asyncio.sleep`` that never returns early.

    On Windows with Python < 3.13 the loop clock has a 15.6 ms resolution and a timer due
    within that resolution fires on the next wake-up, so a short sleep ends as soon as any
    socket I/O completes. Loop until a ``perf_counter`` deadline instead.
    """
    from voice_agent_next.utils import now

    deadline = now() + seconds
    await asyncio.sleep(0)
    while (delay := deadline - now()) > 0:
        await asyncio.sleep(delay)


class Feeder:
    """Streams frames into the engine in the background at ``speed`` x real time, paced
    against ``perf_counter`` deadlines (see :func:`sleep_at_least`)."""

    def __init__(
        self, conn: GeminiLiveConnection, frames: list[AudioFrame], speed: float = 5.0
    ) -> None:
        self.sent = bytearray()
        self._task = asyncio.create_task(self._run(conn, frames, speed))

    async def _run(
        self, conn: GeminiLiveConnection, frames: list[AudioFrame], speed: float
    ) -> None:
        from voice_agent_next.utils import now

        start, streamed = now(), 0.0
        for frame in frames:
            await conn.send_audio(frame)
            self.sent += frame.data
            streamed += frame.duration
            await sleep_at_least(start + streamed / speed - now())

    async def done(self) -> bytes:
        await self._task
        return bytes(self.sent)


def assert_no_audio_lost(sent: bytes, first: bytes, second: bytes, handle_pos: int) -> int:
    """``first`` got a prefix of ``sent``; ``second`` got the rest, replayed from at most the
    handle's position (the resumed server state) -- nothing missing, nothing reordered."""
    assert sent.startswith(first)
    assert sent.endswith(second)
    start = len(sent) - len(second)
    assert start <= handle_pos, "audio after the resumption point was not replayed"
    assert start <= len(first), "a gap between the two connections"
    return start


def statuses(events: Events) -> list[tuple[str, str | None]]:
    return [(e.status, e.detail) for e in events.of(EngineStatus)]


def resumed(events: Events) -> bool:
    return any(e.status in ("resumed", "reconnected") for e in events.of(EngineStatus))


async def test_go_away_rotates_seamlessly_without_losing_audio(fake: Callable[..., Any]) -> None:
    server = await fake(replies=["Hello!"])
    conn, events = await connect(server)
    await say(conn)  # one exchange, so the resumption handle points past the start
    await events.wait(lambda: bool(events.of(ResponseDone)))
    first = server.connection
    await wait_for(lambda: len(first.handles) == 2)  # re-issued after the turn completed
    before = bytes(first.audio)  # the spoken turn, fully received
    feeder = Feeder(conn, quiet_noise(3.0))
    await wait_for(lambda: len(feeder.sent) > 32_000)  # 1 s streamed on the first connection
    await server.go_away(time_left=10.0)
    await events.wait(lambda: resumed(events))
    sent = await feeder.done()
    second = server.connections[1]
    await wait_for(lambda: sent.endswith(bytes(second.audio[-640:])))
    await asyncio.sleep(0.1)
    await conn.aclose()

    assert statuses(events) == [
        ("expiring", "go_away"), ("reconnecting", "go_away"), ("resumed", "go_away")
    ]  # fmt: skip
    assert events.of(EngineStatus)[0].time_left == pytest.approx(10.0)
    second = server.connections[1]
    assert second.resumed_from == first.handles[-1] and second.session is first.session
    assert first.session.resumptions == 1 and conn.resumptions == 1
    await wait_for(lambda: first.closed.is_set())
    assert first.close_code == 1000  # the client closed the old connection (make-before-break)
    whole = before + sent  # everything the client streamed
    handle_pos = first.session.handles[second.resumed_from]
    assert_no_audio_lost(whole, bytes(first.audio), bytes(second.audio), handle_pos)
    assert second.setup["sessionResumption"] == {"handle": second.resumed_from}


async def test_rotation_waits_for_the_model_to_finish(fake: Callable[..., Any]) -> None:
    server = await fake(replies=["One moment, please."], realtime_factor=1.0)
    conn, events = await connect(server)
    await say(conn)
    await events.wait(lambda: bool(events.of(ResponseAudio)))
    await server.go_away(time_left=30.0)  # plenty of time: rotate at the next idle moment
    await events.wait(lambda: resumed(events), timeout=10)
    await conn.aclose()

    order = [type(e).__name__ + (f":{e.status}" if isinstance(e, EngineStatus) else "")
             for e in events.items if not isinstance(e, ResponseAudio)]  # fmt: skip
    assert order.index("ResponseDone") < order.index("EngineStatus:reconnecting")
    assert events.of(ResponseDone)[0].status == "completed"


async def test_rotation_is_forced_at_the_go_away_deadline(fake: Callable[..., Any]) -> None:
    server = await fake(replies=["A long answer that is still going when time runs out."],
                        realtime_factor=1.0)  # fmt: skip
    conn, events = await connect(server)
    await say(conn)
    await events.wait(lambda: bool(events.of(ResponseAudio)))
    await server.go_away(time_left=2.2)  # margin 2 s: the deadline is ~0.2 s away
    await events.wait(lambda: resumed(events), timeout=5)
    await conn.aclose()

    assert ("reconnecting", "go_away (deadline)") in statuses(events)
    assert events.of(ResponseDone)[0].status == "incomplete"  # cut off by the switch


async def test_proactive_rotation_before_the_connection_limit(fake: Callable[..., Any]) -> None:
    server = await fake()
    conn, events = await connect(server, rotate_after=0.3)
    await events.wait(lambda: resumed(events))
    await conn.aclose()

    assert statuses(events)[:2] == [
        ("reconnecting", "max_connection_age"), ("resumed", "max_connection_age")
    ]  # fmt: skip
    assert server.connections[1].resumed_from == server.connections[0].handles[-1]


async def test_dropped_connection_is_resumed_and_buffered_audio_delivered(
    fake: Callable[..., Any],
) -> None:
    server = await fake()
    conn, events = await connect(server, resume_replay=0.05)
    first = server.connection
    # The engine replays what was sent since the handle *plus* the audio sent within
    # `resume_replay` seconds (wall clock) before it arrived, which may still be in flight.
    # So the pre-handle audio must be clearly older than that margin when the handle comes;
    # otherwise replaying it is correct (and timing-dependent: coarse timers on Windows).
    before = quiet_noise(0.5, seed=7)
    for frame in before:
        await conn.send_audio(frame)
    before_bytes = b"".join(f.data for f in before)
    await wait_for(lambda: len(first.audio) == len(before_bytes))  # all of it arrived
    await sleep_at_least(0.3)  # >> resume_replay
    handle = await first.issue_handle()  # a fresh resumption point mid-stream
    assert handle is not None
    handle_pos = first.session.handles[handle]
    await wait_for(lambda: conn.resumption_handle == handle)
    feeder = Feeder(conn, quiet_noise(2.0, seed=8))
    await wait_for(lambda: len(feeder.sent) > 32_000)
    await server.drop(1011, "Internal error encountered.")
    await events.wait(lambda: resumed(events))
    after = await feeder.done()
    second = server.connections[1]
    await wait_for(lambda: len(second.audio) >= len(after), timeout=10)
    await conn.aclose()

    (reconnecting, done_status) = events.of(EngineStatus)
    assert reconnecting.status == "reconnecting" and "1011" in (reconnecting.detail or "")
    assert done_status.status == "resumed" and second.resumed_from == handle
    sent = before_bytes + after
    start = assert_no_audio_lost(sent, bytes(first.audio), bytes(second.audio), handle_pos)
    # exactly the audio since the handle was replayed: nothing lost, nothing duplicated
    assert start == handle_pos == len(before_bytes)
    assert bytes(second.audio) == after


async def test_a_drop_noticed_by_a_send_keeps_the_close_code(fake: Callable[..., Any]) -> None:
    """A send can hit the closed socket before the receive loop reads the close frame
    (a loaded runner): the reconnect must still report, and act on, the server's code."""
    server = await fake()
    conn, events = await connect(server)
    recv = conn._recv_task
    assert recv is not None
    recv.cancel()  # the receive loop has not got to the close frame yet
    await asyncio.gather(recv, return_exceptions=True)
    ws = conn._ws
    assert ws is not None
    await server.drop(1011, "Internal error encountered.")
    await wait_for(lambda: ws.close_code is not None)
    await conn.send_audio(quiet_noise(0.02)[0])  # this send notices the drop
    await events.wait(lambda: resumed(events))
    await conn.aclose()
    reconnecting = events.of(EngineStatus)[0]
    assert reconnecting.status == "reconnecting" and "1011" in (reconnecting.detail or "")


async def test_expired_handle_falls_back_to_a_fresh_session_with_history(
    fake: Callable[..., Any],
) -> None:
    server = await fake(replies=["Nice to meet you, Ada."], transcripts=["my name is Ada"])
    conn, events = await connect(server)
    await say(conn)
    await events.wait(lambda: bool(events.of(ResponseDone)))
    await wait_for(lambda: len(server.connection.handles) == 2)
    server._by_handle.clear()  # the service no longer knows the session (e.g. expired)
    await server.drop()
    await events.wait(lambda: resumed(events))
    await conn.aclose()

    assert [s for s, _ in statuses(events)] == ["reconnecting", "reconnected"]
    fresh = server.connections[-1]
    assert fresh.resumed_from is None and fresh.session is not server.connections[0].session
    assert fresh.setup["historyConfig"] == {"initialHistoryInClientContent": True}
    assert fresh.client_contents[0] == {
        "turns": [
            {"role": "user", "parts": [{"text": "my name is Ada"}]},
            {"role": "model", "parts": [{"text": "Nice to meet you, Ada."}]},
        ],
        "turnComplete": True,
    }


async def test_update_applies_the_new_setup_on_a_resumed_connection(
    fake: Callable[..., Any],
) -> None:
    server = await fake()
    conn, events = await connect(server)
    await conn.update(instructions="Speak like a pirate.", tools=[get_weather])
    await events.wait(lambda: resumed(events))
    await conn.aclose()

    second = server.connections[1]
    assert second.resumed_from is not None
    assert second.setup["systemInstruction"] == {"parts": [{"text": "Speak like a pirate."}]}
    assert second.setup["tools"][0]["functionDeclarations"][0]["name"] == "get_weather"
    assert statuses(events) == [("reconnecting", "update"), ("resumed", "update")]


async def test_failed_planned_rotation_keeps_the_current_connection(
    fake: Callable[..., Any],
) -> None:
    server = await fake(replies=["Still here."])
    conn, events = await connect(server)
    server.reject_status = 503  # the service refuses new connections for a moment
    await conn.update(instructions="New instructions.")
    await events.wait(lambda: len(events.of(EngineStatus)) >= 2)
    assert statuses(events) == [
        ("reconnecting", "update"), ("resumed", "kept the current connection (update)")
    ]  # fmt: skip
    errors = [e for e in events.items if type(e).__name__ == "EngineErrorEvent"]
    assert len(errors) == 1 and errors[0].recoverable
    assert isinstance(errors[0].error, ProviderConnectionError)
    await say(conn)  # the conversation carries on over the first connection
    await events.wait(lambda: bool(events.of(ResponseDone)))
    assert len(server.connections) == 1
    server.reject_status = None
    await events.wait(lambda: conn.resumptions == 1)  # retried at a later idle moment
    await conn.aclose()
    assert server.connections[1].setup["systemInstruction"] == {"parts": [{"text": "New instructions."}]}  # fmt: skip


async def test_reconnect_withdraws_pending_tool_calls(fake: Callable[..., Any]) -> None:
    server = await fake(replies=[FakeToolCall("lookup", {"q": "x"})])
    conn, events = await connect(server, TOOLS)
    await say(conn)
    await events.wait(lambda: bool(events.of(ResponseToolCall)))
    call = events.of(ResponseToolCall)[0].call
    await server.drop()  # the resumed session predates the call (resumable=false meanwhile)
    await events.wait(lambda: resumed(events))
    assert events.of(ToolCallCancelled)[-1].call_ids == [call.call_id]
    await conn.send_tool_output(FunctionCallOutput(call_id=call.call_id, output="late"))
    await asyncio.sleep(0.1)
    await conn.aclose()
    assert server.connections[1].tool_responses == []  # not sent to a session without the call


# ------------------------------------------------------------------- manual turns
async def test_manual_turn_detection(fake: Callable[..., Any]) -> None:
    server = await fake(replies=["Got it."], transcripts=["push to talk"])
    conn, events = await connect(server, EngineOptions(turn_detection=False))
    await push(conn, synth_speech(0.6, 16_000))
    await conn.commit_input()
    await events.wait(lambda: bool(events.of(ResponseDone)))
    await conn.aclose()

    assert server.setups[0]["realtimeInputConfig"] == {"automaticActivityDetection": {"disabled": True}}  # fmt: skip
    kinds = [next(iter(m["realtimeInput"])) for m in server.connection.messages if "realtimeInput" in m]  # fmt: skip
    assert kinds[0] == "activityStart" and kinds[-1] == "activityEnd"
    assert set(kinds[1:-1]) == {"audio"}
    started, stopped = events.of(InputSpeechStarted)[0], events.of(InputSpeechStopped)[0]
    assert started.audio_time == pytest.approx(0.0, abs=0.03)
    assert stopped.audio_time == pytest.approx(0.6, abs=0.03)
    order = events.kinds()
    assert order.index("InputCommitted") < order.index("ResponseStarted")
    assert len(events.of(InputCommitted)) == 1
    assert [e.text for e in events.of(InputTranscript) if e.is_final] == ["push to talk"]


async def test_manual_turns_with_a_continuous_microphone(fake: Callable[..., Any]) -> None:
    """Audio keeps flowing after commit_input() (AgentSession forwards every frame): only
    speech opens an activity, so the requested answer is not interrupted."""
    server = await fake(replies=["One moment, please."], realtime_factor=1.0)
    conn, events = await connect(server, EngineOptions(turn_detection=False))
    await push(conn, AudioFrame.silence(0.3, 16_000))  # silence: no activity, nothing sent
    assert not any("realtimeInput" in m for m in server.connections[0].messages)
    await push(conn, synth_speech(0.6, 16_000))
    await conn.commit_input()
    await events.wait(lambda: bool(events.of(ResponseAudio)))
    await push(conn, AudioFrame.silence(1.0, 16_000))  # the mic stays open while it answers
    await events.wait(lambda: bool(events.of(ResponseDone)), timeout=5)
    await conn.aclose()

    kinds = [next(iter(m["realtimeInput"])) for m in server.connection.messages if "realtimeInput" in m]  # fmt: skip
    assert kinds.count("activityStart") == 1 and kinds.count("activityEnd") == 1
    assert kinds[-1] == "activityEnd"  # the trailing silence was not sent
    assert events.of(ResponseDone)[0].status == "completed"
    assert len(events.of(InputSpeechStarted)) == 1
    # the pre-roll carried the audio from before the detected speech start
    assert len(server.connection.audio) == round(0.9 * 16_000) * 2


async def test_fresh_session_replays_only_the_unanswered_audio(fake: Callable[..., Any]) -> None:
    server = await fake(replies=["Hello!", "Again?"])
    conn, events = await connect(server, session_resumption=False)
    await say(conn)
    await events.wait(lambda: bool(events.of(ResponseDone)))
    await wait_for(lambda: bool(events.of(ResponseDone)) and not conn._server_turn_open)
    await conn.send_text("Note: I am in Paris.", respond=False)
    noise = quiet_noise(0.5, seed=3)
    for frame in noise:
        await conn.send_audio(frame)
    await asyncio.sleep(0.05)
    await server.drop()
    await events.wait(lambda: resumed(events))
    await asyncio.sleep(0.3)
    await conn.aclose()

    assert [s for s, _ in statuses(events)] == ["reconnecting", "reconnected"]
    fresh = server.connections[1]
    assert "sessionResumption" not in fresh.setup
    # the answered turn is re-seeded as text, never replayed as audio or text again
    seeded = fresh.client_contents[0]["turns"]
    assert seeded[0] == {"role": "user", "parts": [{"text": "hello"}]}
    assert seeded[-1] == {"role": "user", "parts": [{"text": "Note: I am in Paris."}]}
    assert len(fresh.client_contents) == 1
    new_audio = b"".join(f.data for f in noise)
    replayed = bytes(fresh.audio)
    assert replayed.endswith(new_audio)  # the audio after the answered turn was replayed...
    before = replayed[: len(replayed) - len(new_audio)]
    assert before == bytes(len(before)) and len(before) < 0.6 * 32_000  # ...not its speech
    assert fresh.user_turns == [] and len(events.of(ResponseStarted)) == 1


async def test_go_away_from_the_replaced_connection_is_ignored(fake: Callable[..., Any]) -> None:
    server = await fake()
    conn, events = await connect(server)
    await conn.update(instructions="Rotate now.")  # rotation starts (idle)
    await server.go_away(time_left=2.2, abort=False)  # the old connection says goodbye meanwhile
    await events.wait(lambda: resumed(events))
    await asyncio.sleep(0.8)  # a stale deadline (~0.2 s) would force another rotation
    await conn.aclose()
    assert ("expiring", "go_away") in statuses(events)  # received while switching...
    assert len(server.connections) == 2  # ...and not applied to the new connection


async def test_a_connection_that_keeps_dying_is_fatal(fake: Callable[..., Any]) -> None:
    server = await fake()
    conn, events = await connect(server, max_reconnect_attempts=3)
    server.drop_after_setup = (1007, "Request contains an invalid argument.")
    await server.drop(1007, "Request contains an invalid argument.")
    await events.wait(lambda: conn.closed, timeout=10)
    errors = [e for e in events.items if type(e).__name__ == "EngineErrorEvent"]
    assert errors and not errors[-1].recoverable
    assert 3 <= len(server.connections) <= 6  # bounded, not an endless reconnect loop


async def test_output_of_a_withdrawn_call_is_not_sent(fake: Callable[..., Any]) -> None:
    filler = "Let me look that up for you, this could take a little while, please hold on."
    server = await fake(replies=[FakeReply(filler, [FakeToolCall("lookup", {"q": "x"})])],
                        realtime_factor=1.0)  # fmt: skip
    conn, events = await connect(server, TOOLS)
    await say(conn)
    await events.wait(lambda: len(events.of(ResponseAudio)) > 5)
    call = events.of(ResponseToolCall)[0].call
    await push(conn, synth_speech(0.4, 16_000))  # barge-in withdraws the call
    await events.wait(lambda: bool(events.of(ToolCallCancelled)))
    await conn.send_tool_output(FunctionCallOutput(call_id=call.call_id, output="too late"))
    await asyncio.sleep(0.1)
    await conn.aclose()
    assert server.connection.tool_responses == []


async def test_text_sent_before_a_handle_is_not_replayed(fake: Callable[..., Any]) -> None:
    server = await fake(replies=["Paris."])
    conn, events = await connect(server)
    await conn.send_text("Capital of France?")
    await events.wait(lambda: bool(events.of(ResponseDone)))
    await wait_for(lambda: len(server.connection.handles) == 2)  # issued after the answer
    await server.drop()
    await events.wait(lambda: resumed(events))
    await asyncio.sleep(0.2)
    await conn.aclose()
    assert server.connections[1].client_contents == []  # the resumed state already has it
    assert len(events.of(ResponseStarted)) == 1


async def test_commit_input_flushes_server_vad(fake: Callable[..., Any]) -> None:
    server = await fake()
    conn, _ = await connect(server)
    await conn.commit_input()  # automatic VAD: audioStreamEnd ends the turn immediately
    await wait_for(lambda: server.connection.audio_stream_ends == 1)
    await conn.clear_input()  # no-op (cannot be undone server-side)
    await conn.aclose()


# ------------------------------------------------------------------------- errors
@pytest.mark.parametrize(
    ("server_kw", "error"),
    [
        ({"api_key": "another-key"}, AuthenticationError),
        ({"reject_status": 403}, AuthenticationError),
        ({"reject_status": 429}, RateLimitError),
        ({"reject_status": 503}, ProviderConnectionError),
        ({"close_after_setup": (1011, "You exceeded your current quota.")}, RateLimitError),
        ({"close_after_setup": (1008, "models/x is not found")}, ProviderError),
    ],
)
async def test_connection_errors_are_mapped(
    fake: Callable[..., Any], server_kw: dict[str, Any], error: type[Exception]
) -> None:
    server = await fake(**server_kw)
    with pytest.raises(error) as info:
        await engine_for(server).connect(EngineOptions())
    assert KEY not in str(info.value)  # never leak the key


async def test_unreachable_server_is_a_connection_error() -> None:
    engine = GeminiLiveEngine(api_key=KEY, base_url="ws://127.0.0.1:9", connect_timeout=2)
    with pytest.raises(ProviderConnectionError):
        await engine.connect(EngineOptions())


async def test_auth_failure_during_reconnect_is_fatal(fake: Callable[..., Any]) -> None:
    server = await fake()
    conn, events = await connect(server)
    server.api_key = "rotated-key"  # e.g. the key was revoked
    await server.drop()
    await events.wait(lambda: conn.closed, timeout=5)
    errors = [e for e in events.items if type(e).__name__ == "EngineErrorEvent"]
    assert len(errors) == 1 and not errors[0].recoverable
    assert isinstance(errors[0].error, AuthenticationError)


# ------------------------------------------------------- end-to-end with AgentSession
class Recorder:
    def __init__(self, session: AgentSession) -> None:
        self.events: list[tuple[str, Any]] = []
        for name in ("user_transcript", "agent_transcript", "tool_call", "tool_result",
                     "interrupted", "metrics", "error", "close"):  # fmt: skip
            session.on(name, self._make(name))

    def _make(self, name: str) -> Callable[[Any], None]:
        return lambda ev: self.events.append((name, ev))

    def of(self, name: str) -> list[Any]:
        return [ev for n, ev in self.events if n == name]

    def turn_metrics(self) -> list[TurnMetrics]:
        return [m for m in self.of("metrics") if isinstance(m, TurnMetrics)]


async def speak_to(
    transport: LoopbackTransport, seconds: float = 0.8, silence: float = 0.6
) -> None:
    await transport.play_user_audio(synth_speech(seconds, 16_000), realtime=False)
    await transport.play_user_audio(AudioFrame.silence(silence, 16_000), realtime=False)


def history(session: AgentSession) -> list[tuple[str, str]]:
    return [(i.role, i.text) for i in session.history.items if isinstance(i, ChatMessage)]


async def test_session_turn_with_greeting_and_transcripts(fake: Callable[..., Any]) -> None:
    server = await fake(replies=["Hi there!"], transcripts=["hello gemini"])
    session = AgentSession(engine_for(server))
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("Be nice.", greeting="Welcome."), transport)
    await wait_for(lambda: session.agent_state == AgentState.LISTENING and bool(rec.of("agent_transcript")))  # fmt: skip
    await speak_to(transport)
    await wait_for(lambda: len(rec.turn_metrics()) == 1)
    transport.end_user_audio()  # the user hangs up
    await asyncio.wait_for(session.wait_closed(), 5)

    assert history(session) == [
        ("assistant", "Welcome."), ("user", "hello gemini"), ("assistant", "Hi there!")
    ]  # fmt: skip
    finals = [e.text for e in rec.of("user_transcript") if e.is_final]
    assert finals == ["hello gemini"]
    assert [e.text for e in rec.of("user_transcript") if not e.is_final] == [
        "hello",
        "hello gemini",
    ]
    m = rec.turn_metrics()[0]  # (audio pushed faster than real time: values not asserted)
    assert m.voice_to_voice is not None and m.end_of_turn_delay is not None
    assert not m.interrupted
    played = sum(p.frame.duration for p in transport.played_log)
    assert played == pytest.approx((len("Welcome.") + len("Hi there!")) / 15, abs=0.15)
    assert session.usage.engine_output_audio_tokens > 0
    assert rec.of("close")[0].reason == "user_disconnected"
    assert server.setups[0]["systemInstruction"] == {"parts": [{"text": "Be nice."}]}


async def test_session_tool_call_round_trip(fake: Callable[..., Any]) -> None:
    calls: list[str] = []

    @function_tool
    async def weather(city: str) -> str:
        """Weather lookup."""
        calls.append(city)
        return f"sunny in {city}"

    server = await fake(replies=[FakeToolCall("weather", {"city": "Paris"}), "Sunny."],
                        transcripts=["weather in paris?"])  # fmt: skip
    session = AgentSession(engine_for(server))
    rec = Recorder(session)
    await session.start(Agent("x", tools=[weather]), LoopbackTransport())
    await speak_to(session.transport)  # type: ignore[arg-type]
    await wait_for(lambda: len(rec.turn_metrics()) == 1)
    await session.aclose()

    assert calls == ["Paris"]
    assert rec.of("tool_result")[0].output.output == "sunny in Paris"
    (response,) = server.connection.tool_responses
    assert response["response"] == {"result": "sunny in Paris"}
    assert response["scheduling"] == "WHEN_IDLE"
    kinds = [getattr(i, "role", i.type) for i in session.history.items]
    assert kinds == ["user", "function_call", "function_call_output", "assistant"]
    assert history(session)[-1] == ("assistant", "Sunny.")
    m = rec.turn_metrics()[0]
    assert m.tool_calls == 1 and m.voice_to_voice is not None


async def test_session_barge_in_cancels_the_running_tool(fake: Callable[..., Any]) -> None:
    started, cancelled = asyncio.Event(), asyncio.Event()

    @function_tool
    async def slow_lookup(query: str) -> str:
        """A slow search."""
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "done"

    filler = "Let me look that up for you, this could take a little while, please hold on."
    server = await fake(replies=[FakeReply(filler, [FakeToolCall("slow_lookup", {"query": "flights"})])],
                        realtime_factor=1.0)  # fmt: skip
    session = AgentSession(engine_for(server))
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await session.start(Agent("x", tools=[slow_lookup]), transport)
    await speak_to(transport, 0.6, 0.5)
    await asyncio.wait_for(started.wait(), 5)
    await wait_for(lambda: session.agent_state == AgentState.SPEAKING)
    await asyncio.sleep(0.5)
    await transport.play_user_audio(synth_speech(0.4, 16_000), realtime=False)  # barge in
    await asyncio.wait_for(cancelled.wait(), 5)  # toolCallCancellation reached the tool
    await wait_for(lambda: bool(rec.of("interrupted")))
    await session.aclose()

    assert server.connection.interruptions == 1
    assert not rec.of("tool_result") and not server.connection.tool_responses
    kinds = [getattr(i, "role", i.type) for i in session.history.items]
    assert kinds[:2] == ["user", "function_call"] and "function_call_output" not in kinds
    said = [
        i for i in session.history.items if isinstance(i, ChatMessage) and i.role == "assistant"
    ]
    assert said and said[-1].interrupted and len(said[-1].text) < len(filler)


@pytest.mark.parametrize("voice_activity", [True, False])
async def test_session_server_side_interruption(
    fake: Callable[..., Any], voice_activity: bool
) -> None:
    long_answer = "This is a very long answer that keeps going and going for quite a while. " * 3
    server = await fake(replies=[long_answer], transcripts=["tell me a story"], realtime_factor=1.0,
                        voice_activity=voice_activity)  # fmt: skip
    session = AgentSession(engine_for(server))
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await session.start(Agent("x"), transport)
    await speak_to(transport, 0.6, 0.5)
    await wait_for(lambda: session.agent_state == AgentState.SPEAKING)
    await asyncio.sleep(1.0)  # let ~1 s of the answer play
    await transport.play_user_audio(synth_speech(0.4, 16_000), realtime=False)  # barge in
    await wait_for(lambda: bool(rec.of("interrupted")))
    await session.aclose()

    ev = rec.of("interrupted")[0]
    assert 0.5 < ev.played < 2.5
    assert transport.clear_times, "transport playback must be cleared"
    assert server.connection.interruptions == 1  # the server stopped generating too
    said = [
        i for i in session.history.items if isinstance(i, ChatMessage) and i.role == "assistant"
    ]
    assert said[-1].interrupted and 0 < len(said[-1].text) < len(long_answer.strip())
    m = rec.turn_metrics()[0]
    assert m.interrupted and m.agent_speech_duration == pytest.approx(ev.played, abs=0.3)


@pytest.mark.parametrize("voice_activity", [True, False])
async def test_session_voice_to_voice_matches_external_measurement(
    fake: Callable[..., Any], voice_activity: bool
) -> None:
    from voice_agent_next.utils import now

    server = await fake(replies=["Okay then."], realtime_factor=1.0, voice_activity=voice_activity)
    session = AgentSession(engine_for(server))
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await session.start(Agent("x"), transport)
    await transport.play_user_audio(synth_speech(0.6, 16_000))  # real time
    speech_end = now()
    await transport.play_user_audio(AudioFrame.silence(1.0, 16_000))
    await wait_for(lambda: bool(rec.turn_metrics()), 5)
    await session.aclose()

    m = rec.turn_metrics()[0]
    heard = transport.played_log[0].start_time - speech_end  # what the simulated user measured
    assert m.voice_to_voice == pytest.approx(0.4, abs=0.15)  # the fake server's VAD silence
    assert m.voice_to_voice == pytest.approx(heard, abs=0.08)
    (engine_metrics,) = [x for x in rec.of("metrics") if isinstance(x, EngineMetrics)]
    assert engine_metrics.ttfb == pytest.approx(m.voice_to_voice, abs=0.05)  # from speech end


async def test_session_continues_across_a_go_away_rotation(fake: Callable[..., Any]) -> None:
    server = await fake(replies=["First.", "Second."], transcripts=["one", "two"])
    session = AgentSession(engine_for(server))
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak_to(transport)
    await wait_for(lambda: len(rec.turn_metrics()) == 1)
    await wait_for(lambda: len(server.connection.handles) == 2)
    await server.go_away(time_left=10.0)
    conn = session.connection
    assert isinstance(conn, GeminiLiveConnection)
    await wait_for(lambda: conn.resumptions == 1)
    await speak_to(transport)
    await wait_for(lambda: len(rec.turn_metrics()) == 2)
    await session.aclose()

    assert history(session) == [
        ("user", "one"), ("assistant", "First."), ("user", "two"), ("assistant", "Second.")
    ]  # fmt: skip
    first, second = server.connections
    assert second.session is first.session and second.user_turns == ["two"]
    assert not rec.of("error")


# ------------------------------------------------------------------- real API (opt-in)
@pytest.mark.integration
@pytest.mark.skipif(
    not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")),
    reason="needs GOOGLE_API_KEY or GEMINI_API_KEY",
)
async def test_real_gemini_live_turn() -> None:
    engine = GeminiLiveEngine(model=os.environ.get("GEMINI_LIVE_MODEL"))
    conn = await engine.connect(EngineOptions(instructions="Answer in one short sentence."))
    assert isinstance(conn, GeminiLiveConnection)
    events = Events(conn)
    await conn.send_text("Say hello and tell me the capital of France.")
    await asyncio.wait_for(events.wait(lambda: bool(events.of(ResponseDone)), timeout=60), 65)
    await conn.aclose()

    audio = events.of(ResponseAudio)
    assert audio and all(a.frame.sample_rate == 24_000 for a in audio)
    assert sum(a.frame.duration for a in audio) > 0.3
    assert "paris" in "".join(e.delta for e in events.of(ResponseText)).lower()
    assert events.of(ResponseDone)[0].status == "completed"
    assert conn.resumption_handle is not None


async def test_close_while_reconnecting_does_not_hang(fake: Callable[..., Any]) -> None:
    server = await fake()
    conn, events = await connect(server)
    server.reject_status = 503  # reconnects keep failing (with backoff)
    await server.drop()
    await events.wait(lambda: bool(events.of(EngineStatus)))
    await asyncio.wait_for(conn.aclose(), 3)
    assert conn.closed
    await conn.send_audio(AudioFrame.silence(0.02, 16_000))  # ignored after close
    await conn.send_text("ignored")


# ------------------------------------------------------- rotation metrics & carry-over
async def test_rotations_are_reported_as_metrics(fake: Callable[..., Any]) -> None:
    server = await fake(replies=["Hello!"])
    engine = engine_for(server)
    metrics: list[RotationMetrics] = []
    engine.on("metrics", lambda m: metrics.append(m) if isinstance(m, RotationMetrics) else None)
    conn = await engine.connect(EngineOptions(instructions="be brief"))
    assert isinstance(conn, GeminiLiveConnection)
    events = Events(conn)
    await say(conn)
    await events.wait(lambda: bool(events.of(ResponseDone)))
    await wait_for(lambda: len(server.connection.handles) == 2)
    await server.go_away(time_left=10.0)
    await events.wait(lambda: bool(metrics))
    await server.drop()
    await events.wait(lambda: len(metrics) == 2)
    await conn.aclose()

    planned, dropped = metrics
    assert (planned.reason, planned.planned, planned.resumed) == ("go_away", True, True)
    assert planned.rotation == 1 and planned.attempts == 1 and planned.lost_audio == 0
    assert planned.carried_items == 0 and planned.failed_responses == 0
    assert 0 <= planned.gap < 2.0
    assert not dropped.planned and dropped.resumed and dropped.rotation == 2
    assert conn.rotations == 2


async def test_fresh_session_is_seeded_with_the_carry_over_strategy(
    fake: Callable[..., Any],
) -> None:
    server = await fake(replies=["Nice to meet you, Ada."], transcripts=["my name is Ada"])
    summarizer = SummarizeHistory(MockLLM(responses=["The user is Ada."]), keep_last=1,
                                  min_items=1)  # fmt: skip
    conn, events = await connect(server, carry_over=summarizer)
    metrics: list[RotationMetrics] = []
    conn._e.on("metrics", lambda m: metrics.append(m) if isinstance(m, RotationMetrics) else None)
    await say(conn)
    await events.wait(lambda: bool(events.of(ResponseDone)))
    await wait_for(lambda: len(server.connection.handles) == 2)
    server._by_handle.clear()  # the handle expired: a fresh session must be seeded
    await server.drop()
    await events.wait(lambda: bool(metrics))
    await conn.aclose()

    fresh = server.connections[-1]
    assert fresh.client_contents[0]["turns"] == [
        {"role": "user", "parts": [{"text": "Summary of the conversation so far: The user is Ada."}]},
        {"role": "model", "parts": [{"text": "Nice to meet you, Ada."}]},
    ]  # fmt: skip
    (m,) = metrics
    assert not m.planned and not m.resumed and m.carried_items == 2
