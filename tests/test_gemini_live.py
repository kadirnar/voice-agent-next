"""Gemini Live engine (``providers/google/live.py``) against the fake BidiGenerateContent server.

Everything runs offline: :class:`FakeGeminiLiveServer` speaks the real JSON protocol on
``127.0.0.1``. The real-API test at the end is marked ``integration``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from voice_agent_next import function_tool
from voice_agent_next.audio import AudioFrame
from voice_agent_next.chat import FunctionCallOutput
from voice_agent_next.engine import EngineOptions
from voice_agent_next.errors import (
    ConfigurationError,
)
from voice_agent_next.events import (
    EngineEvent,
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
from voice_agent_next.metrics import EngineMetrics
from voice_agent_next.providers.google.live import GeminiLiveConnection, GeminiLiveEngine
from voice_agent_next.providers.mock import synth_speech
from voice_agent_next.testing.gemini_live import FakeGeminiLiveServer, FakeReply, FakeToolCall

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
    kinds = events.kinds()
    assert kinds.index("InputSpeechStarted") < kinds.index("ResponseDone")
    assert events.of(ResponseDone)[0].status == "cancelled"
