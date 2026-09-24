"""OpenAI-Realtime-compatible server (``voice_agent_next.server``).

Two kinds of tests:

* compatibility round trips: our own ``OpenAIRealtimeEngine`` (the client) talks to the
  server, which runs a ``MockEngine`` or a ``CascadeEngine`` of mocks, driven by an
  ``AgentSession`` over a ``LoopbackTransport`` (voice turns, text, tools, barge-in);
* raw protocol tests with a plain ``websockets`` client (errors, auth, formats, manual
  turns, dialects). When the official ``openai`` SDK is installed, every server event is
  also validated against the SDK's generated (OpenAPI) event models.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import inspect
import json
import re
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from voice_agent_next import (
    Agent,
    AgentSession,
    AgentState,
    AudioFrame,
    ChatMessage,
    function_tool,
)
from voice_agent_next.audio.codecs import mulaw_encode
from voice_agent_next.cli.main import app as cli_app
from voice_agent_next.cli.serve import ServeOptions, build_models, build_server
from voice_agent_next.engine import EngineConnection, EngineOptions, S2SEngine
from voice_agent_next.engines.cascade import CascadeEngine, CascadeOptions
from voice_agent_next.errors import ConfigurationError
from voice_agent_next.events import (
    InputCommitted,
    InputTranscript,
    ResponseAudio,
    ResponseDone,
    ResponseText,
)
from voice_agent_next.metrics import TurnMetrics
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import (
    MockEngine,
    MockEngineConnection,
    MockLLM,
    MockSTT,
    MockToolCall,
    MockTTS,
    synth_speech,
)
from voice_agent_next.providers.openai.realtime import VERBATIM_INSTRUCTIONS, OpenAIRealtimeEngine
from voice_agent_next.server import RealtimeModel, RealtimeServer
from voice_agent_next.server._session import _VERBATIM
from voice_agent_next.transports import LoopbackTransport
from voice_agent_next.utils import now

KEY = "sk-local-test"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
_CONNECT_ACCEPTS_PROXY = "proxy" in inspect.signature(connect.__init__).parameters


async def wait_for(predicate: Callable[[], Any], timeout: float = 8.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), timeout)


# ------------------------------------------------------------------ schema validation
def _sdk_validator() -> Callable[[dict[str, Any]], None] | None:
    """Validate server events against the openai SDK's event models (if installed)."""
    try:
        import typing

        from openai.types.realtime import RealtimeServerEvent
        from pydantic import BaseModel
    except ImportError:
        return None

    models: dict[str, type[BaseModel]] = {}

    def collect(tp: Any) -> None:
        for arg in typing.get_args(tp):
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                literal = arg.model_fields["type"].annotation
                for name in typing.get_args(literal):
                    models[name] = arg
            else:
                collect(arg)

    collect(RealtimeServerEvent)

    def validate(event: dict[str, Any]) -> None:
        model = models.get(event["type"])
        assert model is not None, f"not a GA server event: {event['type']}"
        model.model_validate(event)  # raises on missing fields / wrong types

    return validate


SDK_VALIDATE = _sdk_validator()


# ---------------------------------------------------------------------------- clients
class RawClient:
    """A plain websockets client that records every server event, in order."""

    def __init__(self, url: str, *, headers: dict[str, str] | None = None,
                 subprotocols: list[str] | None = None, validate: bool = True) -> None:  # fmt: skip
        self.url = url
        self.headers = headers if headers is not None else {"Authorization": f"Bearer {KEY}"}
        self.subprotocols = subprotocols
        self.validate = validate and SDK_VALIDATE is not None
        self.events: list[dict[str, Any]] = []
        self.ws: ClientConnection | None = None
        self._reader: asyncio.Task[None] | None = None
        self._seq = 0

    async def __aenter__(self) -> RawClient:
        kwargs: dict[str, Any] = {"additional_headers": self.headers, "max_size": None}
        if self.subprotocols:
            kwargs["subprotocols"] = self.subprotocols
        if _CONNECT_ACCEPTS_PROXY:
            kwargs["proxy"] = None
        self.ws = await connect(self.url, **kwargs)
        self._reader = asyncio.create_task(self._read())
        await self.wait("session.created")
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self.ws is not None
        await self.ws.close()
        if self._reader is not None:
            await asyncio.wait_for(self._reader, 5)

    async def _read(self) -> None:
        assert self.ws is not None
        with contextlib.suppress(ConnectionClosed):
            async for raw in self.ws:
                event = json.loads(raw)
                if self.validate and SDK_VALIDATE is not None:
                    SDK_VALIDATE(event)
                self.events.append(event)

    async def send(self, etype: str, **fields: Any) -> str:
        assert self.ws is not None
        self._seq += 1
        event_id = f"client_evt_{self._seq}"
        await self.ws.send(json.dumps({"type": etype, "event_id": event_id, **fields}))
        return event_id

    async def send_raw(self, data: str | bytes) -> None:
        assert self.ws is not None
        await self.ws.send(data)

    async def append(self, frame: AudioFrame, chunk: float = 0.02, encode: Any = None) -> None:
        step = round(chunk * frame.sample_rate) * 2
        for i in range(0, len(frame.data), step):
            data = frame.data[i : i + step]
            wire = encode(data) if encode is not None else data
            await self.send("input_audio_buffer.append", audio=base64.b64encode(wire).decode())

    def of(self, etype: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == etype]

    def types(self) -> list[str]:
        return [e["type"] for e in self.events]

    async def wait(self, etype: str, count: int = 1, timeout: float = 8.0) -> dict[str, Any]:
        await wait_for(lambda: len(self.of(etype)) >= count, timeout)
        return self.of(etype)[count - 1]

    def errors(self) -> list[dict[str, Any]]:
        return [e["error"] for e in self.of("error")]

    def audio_bytes(self, response_id: str | None = None,
                    etype: str = "response.output_audio.delta") -> bytes:  # fmt: skip
        return b"".join(
            base64.b64decode(e["delta"])
            for e in self.of(etype)
            if response_id is None or e["response_id"] == response_id
        )


def speech(seconds: float = 1.0, rate: int = 24_000, lead: float = 0.3,
           tail: float = 0.8) -> AudioFrame:  # fmt: skip
    return AudioFrame.concat(
        [
            AudioFrame.silence(lead, rate),
            synth_speech(seconds, rate),
            AudioFrame.silence(tail, rate),
        ]
    )


@contextlib.asynccontextmanager
async def serve(engine: Any = None, **kwargs: Any) -> AsyncIterator[RealtimeServer]:
    kwargs.setdefault("api_keys", KEY)
    server = RealtimeServer(engine if engine is not None else MockEngine(), port=0, **kwargs)
    async with server:
        yield server


def realtime_path(server: RealtimeServer, query: str = "") -> str:
    return f"{server.url}/realtime{query}"


def mock_cascade(**kw: Any) -> CascadeEngine:
    return CascadeEngine(
        stt=MockSTT(transcripts=kw.pop("transcripts", None)),
        llm=MockLLM(responses=kw.pop("responses", None)),
        tts=MockTTS(realtime_factor=kw.pop("realtime_factor", 0.0)),
        vad=EnergyVAD(),
        options=CascadeOptions(min_endpointing_delay=0.0),
    )


def record_connections(engine: S2SEngine) -> list[EngineConnection]:
    """Keep the engine connections the server opens (any engine type)."""
    conns: list[EngineConnection] = []
    original = engine.connect

    async def connect_and_record(options: EngineOptions) -> EngineConnection:
        conn = await original(options)
        conns.append(conn)
        return conn

    engine.connect = connect_and_record  # type: ignore[method-assign]
    return conns


def test_verbatim_convention_matches_the_client() -> None:
    """``say()`` requests are recognized so engines with a TTS speak them exactly."""
    match = _VERBATIM.search(VERBATIM_INSTRUCTIONS.format(text='Hi "there"!'))
    assert match is not None and match.group(1) == 'Hi "there"!'


# ------------------------------------------------------- compatibility (AgentSession)
@function_tool
async def weather(city: str) -> str:
    """Weather lookup."""
    return f"sunny in {city}"


@pytest.mark.parametrize("kind", ["native", "cascade"])
async def test_agent_session_round_trip_with_tools_and_barge_in(kind: str) -> None:
    story = "Once upon a time a voice agent kept talking about the weather all day long. " * 2
    script = [MockToolCall("weather", {"city": "Paris"}), "It is sunny in Paris.", story]
    transcripts = ["what is the weather in paris", "tell me a story", "stop"]
    engine: S2SEngine
    if kind == "native":
        engine = MockEngine(responses=script, transcripts=transcripts, realtime_factor=1.0)
    else:
        engine = mock_cascade(responses=script, transcripts=transcripts, realtime_factor=1.0)
    conns = record_connections(engine)
    async with serve(engine, model="local") as server:
        client = OpenAIRealtimeEngine(base_url=server.url, api_key=KEY, model="local",
                                      turn_detection="server_vad")  # fmt: skip
        session = AgentSession(client)
        events: dict[str, list[Any]] = {}
        for name in ("user_transcript", "agent_transcript", "tool_result", "interrupted",
                     "metrics", "error"):  # fmt: skip
            session.on(name, lambda ev, name=name: events.setdefault(name, []).append(ev))
        transport = LoopbackTransport(realtime_playout=True)
        agent = Agent("You are a weather bot.", tools=[weather], greeting="Welcome!")
        await session.start(agent, transport)

        def turns() -> list[TurnMetrics]:
            return [m for m in events.get("metrics", []) if isinstance(m, TurnMetrics)]

        # 1. greeting: say() is spoken verbatim by the engine
        await wait_for(
            lambda: any("Welcome!" in e.delta for e in events.get("agent_transcript", []))
        )
        await wait_for(lambda: session.agent_state == AgentState.LISTENING)
        # 2. a voice turn -> tool call round trip -> spoken answer
        await transport.play_user_audio(synth_speech(0.8, 16_000), realtime=False)
        await transport.play_user_audio(AudioFrame.silence(0.8, 16_000), realtime=False)
        await wait_for(lambda: len(turns()) == 1, timeout=10)
        # 3. a long answer, interrupted after about a second of playback
        await transport.play_user_audio(synth_speech(0.8, 16_000), realtime=False)
        await transport.play_user_audio(AudioFrame.silence(0.8, 16_000), realtime=False)
        await wait_for(lambda: session.agent_state == AgentState.SPEAKING)
        await asyncio.sleep(1.0)
        await transport.play_user_audio(synth_speech(0.8, 16_000), realtime=False)
        await wait_for(lambda: events.get("interrupted"))
        await wait_for(lambda: _truncated(conns[0]) is not None)  # conversation.item.truncate
        await session.aclose()

    assert not events.get("error"), events.get("error")
    finals = [e.text for e in events["user_transcript"] if e.is_final]
    assert finals[:2] == ["what is the weather in paris", "tell me a story"]
    [result] = events["tool_result"]
    assert result.output.output == "sunny in Paris"
    kinds = [getattr(i, "role", i.type) for i in session.history.items]
    assert kinds[:5] == ["assistant", "user", "function_call", "function_call_output", "assistant"]
    assert session.history.items[0].text == "Welcome!"  # type: ignore[union-attr]
    assert session.history.items[4].text == "It is sunny in Paris."  # type: ignore[union-attr]
    turn = turns()[0]
    assert turn.tool_calls == 1 and turn.voice_to_voice is not None and not turn.interrupted
    # the engine received the tool output under the call id it produced
    conn = conns[0]
    call = next(i for i in _history(conn) if getattr(i, "type", "") == "function_call")
    output = next(i for i in _history(conn) if getattr(i, "type", "") == "function_call_output")
    assert output.call_id == call.call_id and output.output == "sunny in Paris"
    # barge-in: the engine truncated its reply to what the user heard
    interrupted = events["interrupted"][0]
    assert 0.5 < interrupted.played < 3.5
    heard = _truncated(conn)
    assert heard is not None and 0 < len(heard) < len(story.strip())
    story_msg = next(
        i for i in session.history.items if isinstance(i, ChatMessage) and i.interrupted
    )
    assert 0 < len(story_msg.text) < len(story.strip())
    if isinstance(conn, MockEngineConnection):
        [(_, played_ms)] = conn.truncations
        assert played_ms == pytest.approx(interrupted.played * 1000, abs=60)


def _history(conn: EngineConnection) -> list[Any]:
    return list(conn.chat_ctx.items)  # type: ignore[attr-defined]


def _truncated(conn: EngineConnection) -> str | None:
    """Text of the engine's interrupted assistant message (``None`` until truncated)."""
    for item in _history(conn):
        if isinstance(item, ChatMessage) and item.role == "assistant" and item.interrupted:
            return item.text
    return None


async def test_text_input_through_agent_session() -> None:
    engine = MockEngine(responses=["Paris is the capital of France."])
    async with serve(engine) as server:
        client = OpenAIRealtimeEngine(base_url=server.url, api_key=KEY)
        session = AgentSession(client)
        transport = LoopbackTransport()
        await session.start(Agent("Be brief."), transport)
        await session.generate_reply(user_input="What is the capital of France?")
        await wait_for(
            lambda: any(
                isinstance(i, ChatMessage) and i.role == "assistant" and i.text
                for i in session.history.items
            )
        )
        await session.aclose()
    user, assistant = [i for i in session.history.items if isinstance(i, ChatMessage)]
    assert (user.role, user.text) == ("user", "What is the capital of France?")
    assert assistant.text == "Paris is the capital of France."
    [conn] = engine.connections
    assert conn.instructions == "Be brief."  # from session.update
    texts = [m.text for m in conn.chat_ctx.messages()]
    assert texts == ["What is the capital of France?", "Paris is the capital of France."]


async def test_manual_turns_adopt_the_engine_answer() -> None:
    """``turn_detection: None``: commit + response.create; the engine answers once."""
    engine = MockEngine(responses=["Got it."], transcripts=["push to talk"])
    async with serve(engine) as server:
        client = OpenAIRealtimeEngine(base_url=server.url, api_key=KEY, turn_detection=None)
        conn = await client.connect(EngineOptions())
        got: list[Any] = []

        async def collect() -> None:
            async for ev in conn.events():
                got.append(ev)

        task = asyncio.create_task(collect())
        frame = speech(0.6, 16_000, lead=0.1, tail=0.2)
        step = 640
        for i in range(0, len(frame.data), step):
            await conn.send_audio(AudioFrame(frame.data[i : i + step], 16_000, 1, now()))
        await conn.commit_input()  # input_audio_buffer.commit + response.create
        await wait_for(lambda: any(isinstance(e, ResponseDone) for e in got))
        await conn.aclose()
        await asyncio.wait_for(task, 5)
    committed = [e for e in got if isinstance(e, InputCommitted)]
    finals = [e for e in got if isinstance(e, InputTranscript) and e.is_final]
    assert len(committed) == 1
    assert [(f.item_id, f.text) for f in finals] == [(committed[0].item_id, "push to talk")]
    assert "".join(e.delta for e in got if isinstance(e, ResponseText)) == "Got it."
    assert sum(e.frame.duration for e in got if isinstance(e, ResponseAudio)) > 0.3
    [engine_conn] = engine.connections
    assert engine_conn.responses_started == 1  # the held answer was adopted, not regenerated


# ------------------------------------------------------------------ raw protocol
async def test_session_created_update_and_client_errors() -> None:
    async with serve(MockEngine(), model="mock") as server, RawClient(realtime_path(server)) as c:
        created = c.of("session.created")[0]["session"]
        assert created["type"] == "realtime" and created["object"] == "realtime.session"
        assert created["model"] == "mock" and created["id"].startswith("sess_")
        assert created["output_modalities"] == ["audio"]
        assert created["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24_000}
        assert created["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24_000}
        td = created["audio"]["input"]["turn_detection"]
        assert (td["type"], td["create_response"], td["interrupt_response"]) == (
            "server_vad",
            True,
            True,
        )
        assert created["audio"]["input"]["transcription"] is None  # off until requested
        tool = {
            "type": "function",
            "name": "lookup",
            "description": "Look up.",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        }
        audio = {
            "input": {
                "format": {"type": "audio/pcmu"},
                "transcription": {"model": "whisper-1", "language": "de"},
                "turn_detection": {"type": "semantic_vad", "eagerness": "low"},
            },
            "output": {"format": {"type": "audio/pcma"}, "voice": "cedar"},
        }
        await c.send(
            "session.update",
            session={
                "type": "realtime",
                "instructions": "Be terse.",
                "tools": [tool],
                "audio": audio,
                "tracing": "auto",
            },
        )
        updated = (await c.wait("session.updated"))["session"]
        assert updated["instructions"] == "Be terse." and updated["tools"] == [tool]
        assert updated["audio"]["input"]["format"] == {"type": "audio/pcmu"}
        assert updated["audio"]["output"] == {
            "format": {"type": "audio/pcma"},
            "voice": "cedar",
            "speed": 1.0,
        }
        assert updated["audio"]["input"]["turn_detection"] == {
            "type": "semantic_vad",
            "eagerness": "low",
            "create_response": True,
            "interrupt_response": True,
        }
        assert updated["audio"]["input"]["transcription"]["language"] == "de"
        assert updated["tracing"] == "auto"
        # rejected events: an error event naming the client's event_id and the parameter
        bad_format = await c.send(
            "session.update",
            session={"type": "realtime", "audio": {"input": {"format": {"type": "audio/opus"}}}},
        )
        bad_tool = await c.send(
            "session.update",
            session={"type": "realtime", "tools": [{"type": "mcp", "server_label": "x"}]},
        )
        unknown = await c.send("response.bogus")
        await c.send_raw("{not json")
        await c.send_raw(b"\x00\x01")
        missing = await c.send("conversation.item.delete")
        await wait_for(lambda: len(c.errors()) == 6)
        errors = c.errors()
        assert (errors[0]["event_id"], errors[0]["param"]) == (
            bad_format,
            "session.audio.input.format.type",
        )
        assert (errors[1]["event_id"], errors[1]["code"]) == (bad_tool, "unsupported_tool_type")
        assert (errors[2]["event_id"], errors[2]["code"]) == (unknown, "invalid_event")
        assert errors[3]["code"] == "invalid_json" and errors[3]["event_id"] is None
        assert errors[4]["code"] == "invalid_event"
        assert (errors[5]["event_id"], errors[5]["code"], errors[5]["param"]) == (
            missing,
            "missing_required_parameter",
            "item_id",
        )
        assert all(e["type"] == "invalid_request_error" for e in errors)
        # rejected updates changed nothing
        await c.send("session.update", session={"type": "realtime"})
        assert (await c.wait("session.updated", 2))["session"] == updated


async def test_voice_turn_events_with_transcription() -> None:
    engine = MockEngine(responses=["It is noon."], transcripts=["what time is it"])
    async with serve(engine) as server, RawClient(realtime_path(server)) as c:
        transcription = {"model": "gpt-4o-mini-transcribe"}
        await c.send(
            "session.update",
            session={"type": "realtime", "audio": {"input": {"transcription": transcription}}},
        )
        await c.wait("session.updated")
        await c.append(speech(1.0))
        done = await c.wait("response.done")
        types = c.types()
        order = [
            "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped",
            "input_audio_buffer.committed",
            "conversation.item.added",
            "conversation.item.done",
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_audio_transcript.delta",
            "response.output_audio.delta",
            "response.output_audio.done",
            "response.output_audio_transcript.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.done",
        ]
        positions = [types.index(t) for t in order]
        assert positions == sorted(positions), types
        started = c.of("input_audio_buffer.speech_started")[0]
        stopped = c.of("input_audio_buffer.speech_stopped")[0]
        committed = c.of("input_audio_buffer.committed")[0]
        assert started["item_id"] == stopped["item_id"] == committed["item_id"]
        assert 150 <= started["audio_start_ms"] <= 400  # speech starts at 0.3 s
        assert 1200 <= stopped["audio_end_ms"] <= 1450  # and ends at 1.3 s
        user_item = c.of("conversation.item.added")[0]["item"]
        assert (user_item["role"], user_item["content"][0]["type"]) == ("user", "input_audio")
        completed = await c.wait("conversation.item.input_audio_transcription.completed")
        assert (completed["item_id"], completed["transcript"]) == (
            committed["item_id"],
            "what time is it",
        )
        response = done["response"]
        assert response["status"] == "completed" and response["status_details"] is None
        [message] = response["output"]
        assert message["role"] == "assistant" and message["status"] == "completed"
        assert message["content"] == [{"type": "output_audio", "transcript": "It is noon."}]
        assert response["usage"]["output_token_details"]["text_tokens"] == 3
        audio = c.audio_bytes(response["id"])
        assert len(audio) / 2 / 24_000 == pytest.approx(len("It is noon.") / 15, abs=0.05)
        deltas = c.of("response.output_audio.delta")
        assert all(d["item_id"] == message["id"] for d in deltas)
        assert max(len(base64.b64decode(d["delta"])) for d in deltas) <= 2 * 2_400  # <= 100 ms
        # the conversation keeps the user transcript
        await c.send("conversation.item.retrieve", item_id=committed["item_id"])
        retrieved = (await c.wait("conversation.item.retrieved"))["item"]
        assert retrieved["content"][0]["transcript"] == "what time is it"


async def test_manual_commit_and_clear_with_raw_client() -> None:
    engine = MockEngine(responses=["Sure."], transcripts=["book a table"])
    async with serve(engine) as server, RawClient(realtime_path(server)) as c:
        await c.send(
            "session.update",
            session={"type": "realtime", "audio": {"input": {"turn_detection": None}}},
        )
        await c.wait("session.updated")
        empty = await c.send("input_audio_buffer.commit")
        await wait_for(lambda: c.errors())
        assert (c.errors()[0]["code"], c.errors()[0]["event_id"]) == (
            "input_audio_buffer_commit_empty",
            empty,
        )
        await c.append(speech(0.8, lead=0.1, tail=0.2))
        await c.send("input_audio_buffer.commit")
        committed = await c.wait("input_audio_buffer.committed")
        await c.wait("conversation.item.done")
        await asyncio.sleep(0.3)  # the engine answers internally, but nothing is shown yet
        assert not c.of("response.created") and not c.of("input_audio_buffer.speech_started")
        await c.send("response.create")
        done = await c.wait("response.done")
        assert done["response"]["output"][0]["content"][0]["transcript"] == "Sure."
        assert engine.connections[0].responses_started == 1
        added = c.of("conversation.item.added")
        assert added[0]["item"]["id"] == committed["item_id"]
        assert added[1]["previous_item_id"] == committed["item_id"]  # the reply follows it
        # clear drops buffered audio
        await c.append(speech(0.3, lead=0.0, tail=0.0))
        await c.send("input_audio_buffer.clear")
        await c.wait("input_audio_buffer.cleared")
        await c.send("input_audio_buffer.commit")
        await wait_for(lambda: len(c.errors()) == 2)
        assert c.errors()[1]["code"] == "input_audio_buffer_commit_empty"


async def test_create_response_false_holds_until_requested() -> None:
    engine = MockEngine(responses=["First answer.", "Held answer.", "New answer."])
    async with serve(engine) as server, RawClient(realtime_path(server)) as c:
        td = {"type": "server_vad", "create_response": False, "interrupt_response": False}
        await c.send(
            "session.update",
            session={"type": "realtime", "audio": {"input": {"turn_detection": td}}},
        )
        await c.wait("session.updated")
        await c.append(speech(0.8))
        await c.wait("input_audio_buffer.committed")
        await asyncio.sleep(0.3)
        assert not c.of("response.created")
        await c.send("response.create", response={"metadata": {"turn": "1"}})
        done = await c.wait("response.done")
        assert done["response"]["metadata"] == {"turn": "1"}
        assert done["response"]["output"][0]["content"][0]["transcript"] == "First answer."
        assert engine.connections[0].responses_started == 1
        # per-response instructions cannot reuse a held answer: it is generated again
        await c.append(speech(0.8))
        await c.wait("input_audio_buffer.committed", 2)
        await c.send("response.create", response={"instructions": "Answer in French."})
        done = await c.wait("response.done", 2)
        assert done["response"]["output"][0]["content"][0]["transcript"] == "New answer."
        assert engine.connections[0].responses_started == 3


async def test_text_items_tool_calls_and_function_call_output() -> None:
    engine = MockEngine(responses=[MockToolCall("lookup", {"q": "pi"}), "Pi is 3.14."])
    async with serve(engine) as server, RawClient(realtime_path(server)) as c:
        tool = {"type": "function", "name": "lookup", "parameters": {"type": "object"}}
        await c.send("session.update", session={"type": "realtime", "tools": [tool]})
        await c.wait("session.updated")
        text = {"type": "input_text", "text": "What is pi?"}
        await c.send(
            "conversation.item.create", item={"type": "message", "role": "user", "content": [text]}
        )
        added = (await c.wait("conversation.item.added"))["item"]
        assert added["content"] == [text]
        await c.send("response.create")
        done = (await c.wait("response.done"))["response"]
        [call] = done["output"]
        assert (call["type"], call["name"], json.loads(call["arguments"])) == (
            "function_call",
            "lookup",
            {"q": "pi"},
        )
        args_done = c.of("response.function_call_arguments.done")[0]
        assert (args_done["call_id"], args_done["item_id"]) == (call["call_id"], call["id"])
        [engine_conn] = engine.connections
        assert engine_conn.tools[0].name == "lookup"
        output = {"type": "function_call_output", "call_id": call["call_id"], "output": "3.14"}
        await c.send("conversation.item.create", item=output)
        await c.send("response.create")
        final = (await c.wait("response.done", 2))["response"]
        assert final["output"][0]["content"][0]["transcript"] == "Pi is 3.14."
        assert [(o.call_id, o.output) for o in engine_conn.tool_outputs] == [
            (call["call_id"], "3.14")
        ]
        assert not c.errors()


async def test_truncate_validation_and_engine_truncation() -> None:
    engine = MockEngine(responses=["This reply lasts a couple of seconds."])
    async with serve(engine) as server, RawClient(realtime_path(server)) as c:
        hi = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi"}]}
        await c.send("conversation.item.create", item=hi)
        user_id = (await c.wait("conversation.item.added"))["item"]["id"]
        await c.send("response.create")
        done = (await c.wait("response.done"))["response"]
        item_id = done["output"][0]["id"]
        audio_ms = len(c.audio_bytes(done["id"])) / 2 / 24
        invalid = [
            {"item_id": "nope", "content_index": 0, "audio_end_ms": 10},
            {"item_id": user_id, "content_index": 0, "audio_end_ms": 10},
            {"item_id": item_id, "content_index": 1, "audio_end_ms": 10},
            {"item_id": item_id, "content_index": 0, "audio_end_ms": round(audio_ms) + 50},
        ]
        for fields in invalid:
            await c.send("conversation.item.truncate", **fields)
        await wait_for(lambda: len(c.errors()) == 4)
        codes = [e["code"] for e in c.errors()]
        assert codes == [
            "item_truncate_invalid_item_id",
            "unsupported_content_type",
            "invalid_value",
            "invalid_value",
        ]
        await c.send(
            "conversation.item.truncate", item_id=item_id, content_index=0, audio_end_ms=1000
        )
        truncated = await c.wait("conversation.item.truncated")
        assert (truncated["item_id"], truncated["audio_end_ms"]) == (item_id, 1000)
        [engine_conn] = engine.connections
        [(engine_item, end)] = engine_conn.truncations
        assert end == 1000 and engine_item != item_id  # mapped to the engine's own item id
        await c.send("conversation.item.retrieve", item_id=item_id)
        retrieved = (await c.wait("conversation.item.retrieved"))["item"]
        assert retrieved["content"][0]["transcript"] == "This reply last"  # 1 s = 15 chars
        assert engine_conn.chat_ctx.messages()[1].interrupted


async def test_response_cancel_and_active_response_errors() -> None:
    engine = MockEngine(
        responses=["A long answer that keeps going for a few seconds."], realtime_factor=1.0
    )
    async with serve(engine) as server, RawClient(realtime_path(server)) as c:
        nothing = await c.send("response.cancel")
        await wait_for(lambda: c.errors())
        assert (c.errors()[0]["code"], c.errors()[0]["event_id"]) == (
            "response_cancel_not_active",
            nothing,
        )
        await c.send("response.create")
        busy = await c.send("response.create")
        await wait_for(lambda: len(c.errors()) == 2)
        assert (c.errors()[1]["code"], c.errors()[1]["event_id"]) == (
            "conversation_already_has_active_response",
            busy,
        )
        created = await c.wait("response.created")
        await c.wait("response.output_audio.delta", 3)
        await c.send("response.cancel", response_id=created["response"]["id"])
        done = (await c.wait("response.done"))["response"]
        assert done["status"] == "cancelled"
        assert done["status_details"] == {"type": "cancelled", "reason": "client_cancelled"}
        assert done["output"][0]["status"] == "incomplete"
        sent = len(c.of("response.output_audio.delta"))
        await asyncio.sleep(0.3)
        assert len(c.of("response.output_audio.delta")) == sent  # nothing after the cancel
        assert len(c.audio_bytes()) / 2 / 24_000 < 2.0


async def test_interrupt_response_on_user_speech() -> None:
    story = "A very long story that the user will interrupt before it is over. " * 3
    engine = MockEngine(responses=[story, "Okay."], realtime_factor=1.0)
    async with serve(engine) as server, RawClient(realtime_path(server)) as c:
        await c.send("response.create")
        await c.wait("response.output_audio.delta", 5)
        await c.append(speech(0.6, lead=0.0, tail=0.6))
        done = (await c.wait("response.done"))["response"]
        assert done["status_details"] == {"type": "cancelled", "reason": "turn_detected"}
        types = c.types()
        assert types.index("input_audio_buffer.speech_started") < types.index("response.done")
        second = (await c.wait("response.done", 2))["response"]  # the user's turn is answered
        assert second["status"] == "completed"
        assert second["output"][0]["content"][0]["transcript"] == "Okay."


async def test_item_delete_rebuilds_the_engine_context() -> None:
    engine = MockEngine(responses=["Noted.", "Done."])
    async with serve(engine) as server, RawClient(realtime_path(server)) as c:

        def user(text: str) -> dict[str, Any]:
            return {"type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": text}]}  # fmt: skip

        await c.send("conversation.item.create", item=user("first"))
        await c.send("response.create")  # opens the engine connection
        await c.wait("response.done")
        ids = []
        for text in ("second", "third"):  # appended to the live engine context
            await c.send("conversation.item.create", item=user(text))
            ids.append((await c.wait("conversation.item.added", len(ids) + 3))["item"]["id"])
        assert [m.text for m in engine.connections[0].chat_ctx.messages()][-2:] == [
            "second",
            "third",
        ]
        system = {
            "id": "sys_1",
            "type": "message",
            "role": "system",
            "content": [{"type": "input_text", "text": "context"}],
        }
        await c.send("conversation.item.create", previous_item_id="root", item=system)
        root = await c.wait("conversation.item.added", 5)
        assert root["previous_item_id"] is None and root["item"]["id"] == "sys_1"
        await c.send("conversation.item.delete", item_id=ids[0])
        assert (await c.wait("conversation.item.deleted"))["item_id"] == ids[0]
        await c.send("conversation.item.retrieve", item_id=ids[0])
        await wait_for(lambda: c.errors())
        assert c.errors()[0]["code"] == "item_retrieve_invalid_item_id"
        await c.send("response.create")
        await c.wait("response.done", 2)
        # engines cannot insert at the start or delete: the context was rebuilt on a new
        # engine connection, in the client's order
        assert len(engine.connections) == 2 and engine.connections[0].closed
        texts = [(m.role, m.text) for m in engine.connections[1].chat_ctx.messages()]
        assert texts == [
            ("system", "context"),
            ("user", "first"),
            ("assistant", "Noted."),
            ("user", "third"),
            ("assistant", "Done."),
        ]


async def test_g711_audio_in_and_out() -> None:
    engine = MockEngine(responses=["Telephone quality."])
    async with serve(engine) as server, RawClient(realtime_path(server)) as c:
        ulaw = {"format": {"type": "audio/pcmu"}}
        await c.send(
            "session.update", session={"type": "realtime", "audio": {"input": ulaw, "output": ulaw}}
        )
        await c.wait("session.updated")
        await c.append(speech(0.8, 8_000), encode=mulaw_encode)
        done = (await c.wait("response.done"))["response"]
        assert done["audio"]["output"]["format"] == {"type": "audio/pcmu"}
        audio = c.audio_bytes(done["id"])  # one byte per sample at 8 kHz
        assert len(audio) / 8_000 == pytest.approx(len("Telephone quality.") / 15, abs=0.05)
        await wait_for(lambda: engine.connections[0].received_audio >= 1.85)
        assert engine.connections[0].received_audio == pytest.approx(1.9, abs=0.05)


async def test_text_only_output() -> None:
    engine = MockEngine(responses=["Just text.", "Spoken."])
    async with serve(engine) as server, RawClient(realtime_path(server)) as c:
        await c.send("session.update", session={"type": "realtime", "output_modalities": ["text"]})
        await c.wait("session.updated")
        await c.send("response.create")
        done = (await c.wait("response.done"))["response"]
        assert done["output_modalities"] == ["text"]
        assert done["output"][0]["content"] == [{"type": "output_text", "text": "Just text."}]
        assert [e["delta"] for e in c.of("response.output_text.delta")] == ["Just text."]
        assert c.of("response.output_text.done")[0]["text"] == "Just text."
        assert not c.of("response.output_audio.delta")
        await c.send("response.create", response={"output_modalities": ["audio"]})
        second = (await c.wait("response.done", 2))["response"]
        assert second["output"][0]["content"][0] == {
            "type": "output_audio",
            "transcript": "Spoken.",
        }
        assert c.audio_bytes(second["id"])


async def test_beta_dialect() -> None:
    engine = MockEngine(responses=["Beta reply."], transcripts=["hello beta"])
    async with serve(engine) as server:
        headers = {"Authorization": f"Bearer {KEY}", "OpenAI-Beta": "realtime=v1"}
        async with RawClient(realtime_path(server), headers=headers, validate=False) as c:
            session = c.of("session.created")[0]["session"]
            assert session["input_audio_format"] == "pcm16" and "audio" not in session
            update = {
                "modalities": ["text", "audio"],
                "input_audio_format": "g711_ulaw",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {"type": "server_vad", "silence_duration_ms": 400},
            }
            await c.send("session.update", session=update)
            updated = (await c.wait("session.updated"))["session"]
            assert updated["input_audio_format"] == "g711_ulaw"
            assert updated["output_audio_format"] == "pcm16"
            assert updated["turn_detection"]["silence_duration_ms"] == 400
            await c.append(speech(0.8, 8_000), encode=mulaw_encode)
            done = (await c.wait("response.done"))["response"]
            assert done["modalities"] == ["text", "audio"]
            assert done["output"][0]["content"] == [{"type": "audio", "transcript": "Beta reply."}]
            types = set(c.types())
            beta = {
                "response.audio.delta",
                "response.audio_transcript.delta",
                "response.audio.done",
                "conversation.item.created",
            }
            assert beta <= types
            ga_only = {
                "response.output_audio.delta",
                "conversation.item.added",
                "conversation.item.done",
            }
            assert not types & ga_only
            await c.wait("conversation.item.input_audio_transcription.completed")


async def test_auth_model_routing_health_and_session_limit() -> None:
    first, second = MockEngine(responses=["one"]), MockEngine(responses=["two"])
    server = RealtimeServer(
        models={"one": first, "two": second}, port=0, api_keys=[KEY, "k2"], max_sessions=1
    )
    async with server:
        url = realtime_path(server)
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic xyz"}):
            with pytest.raises(InvalidStatus) as info:
                async with RawClient(url, headers=headers):
                    pass
            assert info.value.response.status_code == 401
            assert b"invalid_api_key" in (info.value.response.body or b"")
        with pytest.raises(InvalidStatus) as info:  # several models: names are checked
            async with RawClient(realtime_path(server, "?model=three")):
                pass
        assert info.value.response.status_code == 404
        async with RawClient(realtime_path(server, "?model=two"), headers={"api-key": "k2"}) as c:
            assert c.of("session.created")[0]["session"]["model"] == "two"
            with pytest.raises(InvalidStatus) as info:  # max_sessions=1
                async with RawClient(url):
                    pass
            assert info.value.response.status_code == 503
            base = f"http://127.0.0.1:{server.port}"
            async with httpx.AsyncClient(trust_env=False) as http:
                health = (await http.get(f"{base}/health")).json()
                models = (await http.get(f"{base}/v1/models")).json()
                missing = await http.get(f"{base}/v2/other")
            assert health["status"] == "ok" and health["sessions"] == 1
            assert health["models"] == ["one", "two"] and health["default_model"] == "one"
            assert [m["id"] for m in models["data"]] == ["one", "two"]
            assert missing.status_code == 404
            await c.send("response.create")
            done = (await c.wait("response.done"))["response"]
            assert done["output"][0]["content"][0]["transcript"] == "two"
        # browsers authenticate with subprotocols; the server selects "realtime"
        await wait_for(lambda: not server.sessions)
        protocols = ["realtime", f"openai-insecure-api-key.{KEY}"]
        async with RawClient(url, headers={}, subprotocols=protocols) as c:
            assert c.ws is not None and c.ws.subprotocol == "realtime"
            assert c.of("session.created")[0]["session"]["model"] == "one"
    # a single model serves any name (clients with a hard-coded model), unless strict
    lenient = serve(first, model="local", api_keys=None)
    async with lenient as srv, RawClient(realtime_path(srv, "?model=gpt"), headers={}) as c:
        assert c.of("session.created")[0]["session"]["model"] == "local"
    async with serve(first, model="local", api_keys=None, accept_any_model=False) as strict:
        with pytest.raises(InvalidStatus) as info:
            async with RawClient(realtime_path(strict, "?model=gpt-realtime"), headers={}):
                pass
        assert info.value.response.status_code == 404


async def test_concurrent_sessions_and_clean_shutdown() -> None:
    def echo(ctx: Any) -> str:
        return f"echo {ctx.last_message('user').text}"

    engine = MockEngine(responses=echo)
    server = RealtimeServer(engine, port=0)
    await server.start()
    clients = [RawClient(realtime_path(server), headers={}) for _ in range(3)]
    for c in clients:
        await c.__aenter__()
    for i, c in enumerate(clients):
        item = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": f"n{i}"}],
        }
        await c.send("conversation.item.create", item=item)
        await c.send("response.create")
    for i, c in enumerate(clients):
        done = (await c.wait("response.done"))["response"]
        assert done["output"][0]["content"][0]["transcript"] == f"echo n{i}"
    assert len(server.sessions) == 3 and len(engine.connections) == 3
    await server.aclose()
    for c in clients:
        assert c._reader is not None
        await asyncio.wait_for(c._reader, 5)
        assert c.ws is not None and c.ws.close_code == 1001
    assert all(conn.closed for conn in engine.connections)
    assert not server.sessions


async def test_per_session_engine_factory_and_session_defaults() -> None:
    built: list[MockEngine] = []

    def factory() -> MockEngine:
        built.append(MockEngine(responses=[f"engine {len(built) + 1}"]))
        return built[-1]

    model = RealtimeModel(factory, instructions="Default prompt.", voice="alloy")
    async with serve(model, model="per-session") as server:
        for n in (1, 2):
            async with RawClient(realtime_path(server)) as c:
                session = c.of("session.created")[0]["session"]
                assert session["instructions"] == "Default prompt."
                assert session["audio"]["output"]["voice"] == "alloy"
                await c.send("response.create")
                done = (await c.wait("response.done"))["response"]
                assert done["output"][0]["content"][0]["transcript"] == f"engine {n}"
            await wait_for(lambda: not server.sessions)
    assert len(built) == 2 and all(e.connections[0].closed for e in built)
    assert built[0].connections[0].options.voice == "alloy"


async def test_engine_failure_is_reported() -> None:
    class Broken(MockEngine):
        async def connect(self, options: EngineOptions) -> EngineConnection:
            raise ConnectionRefusedError("model server is down")

    async with serve(Broken()) as server, RawClient(realtime_path(server)) as c:
        await c.send("response.create")
        await wait_for(lambda: c.errors())
        error = c.errors()[0]
        assert (error["type"], error["code"]) == ("server_error", "engine_unavailable")
        assert "model server is down" in error["message"]
        assert c._reader is not None
        await asyncio.wait_for(c._reader, 5)
        assert c.ws is not None and c.ws.close_code == 1011


@pytest.mark.skipif(SDK_VALIDATE is None, reason="needs the openai SDK (extra: openai)")
async def test_official_openai_sdk_client() -> None:
    from openai import AsyncOpenAI

    engine = MockEngine(responses=["Hello from a local engine."])
    async with serve(engine, model="local") as server:
        client = AsyncOpenAI(api_key=KEY, websocket_base_url=server.url)
        options: Any = {"proxy": None} if _CONNECT_ACCEPTS_PROXY else {}
        transcript, audio, seen = "", b"", []
        async with client.realtime.connect(
            model="local", websocket_connection_options=options
        ) as conn:
            await conn.session.update(
                session={
                    "type": "realtime",
                    "instructions": "Be kind.",
                    "output_modalities": ["audio"],
                }
            )
            hi = {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hi"}],
            }
            await conn.conversation.item.create(item=hi)  # type: ignore[arg-type]
            await conn.response.create()
            async for event in conn:
                seen.append(event.type)
                if event.type == "response.output_audio_transcript.delta":
                    transcript += event.delta
                elif event.type == "response.output_audio.delta":
                    audio += base64.b64decode(event.delta)
                elif event.type == "response.done":
                    assert event.response.status == "completed"
                    break
        await client.close()
    assert seen[:2] == ["session.created", "session.updated"]
    assert transcript == "Hello from a local engine."
    assert len(audio) / 2 / 24_000 > 1.0
    assert engine.connections[0].instructions == "Be kind."


# ------------------------------------------------------------------------------ CLI
def test_van_serve_is_registered() -> None:
    result = CliRunner().invoke(cli_app, ["serve", "--help"])
    assert result.exit_code == 0, result.output
    text = ANSI.sub("", result.output)
    assert "--engine" in text and "--protocol" in text and "--api-key" in text


def test_build_models_from_cli_options(tmp_path: Path) -> None:
    assert list(build_models(None)) == ["mock"]
    [(name, model)] = build_models(["mock"], name="local", voice="alloy").items()
    assert (name, model.engine, model.voice) == ("local", "mock", "alloy")
    models = build_models(["fast={provider: mock, response_delay: 0.2}", "{provider: mock}"])
    assert list(models) == ["fast", "mock"]
    assert models["fast"].engine == {"provider": "mock", "response_delay": 0.2}
    native = tmp_path / "native.yaml"
    native.write_text(
        "engine: mock\nagent: {instructions: From the file., voice: cedar, language: de}\n",
        encoding="utf-8",
    )
    cascade = tmp_path / "local-cascade.toml"
    cascade.write_text(
        'stt = "mock"\nllm = "mock"\ntts = "mock"\nvad = "energy"\n', encoding="utf-8"
    )
    models = build_models([str(native), str(cascade)], instructions=None)
    assert list(models) == ["native", "local-cascade"]
    assert isinstance(models["native"].engine, MockEngine)
    assert (models["native"].instructions, models["native"].voice) == ("From the file.", "cedar")
    assert models["native"].language == "de" and models["native"].owned
    assert isinstance(models["local-cascade"].engine, CascadeEngine)
    flags = build_models(stt="mock", llm="mock", tts="mock", vad="energy", instructions="Hi.")
    assert list(flags) == ["cascade"] and isinstance(flags["cascade"].engine, CascadeEngine)
    assert flags["cascade"].instructions == "Hi."
    for kwargs in ({"engines": ["mock", "mock"]}, {"vad": "energy"}, {"llm": "mock"},
                   {"engines": [str(tmp_path / "missing.yaml")]}):  # fmt: skip
        with pytest.raises(ConfigurationError):
            build_models(**kwargs)
    with pytest.raises(ConfigurationError, match="unknown protocol"):
        build_server(build_models(None), ServeOptions(protocol="sip"))


def test_van_serve_runs_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def serve_briefly(self: RealtimeServer) -> None:
        async with httpx.AsyncClient(trust_env=False) as http:
            seen["health"] = (await http.get(f"http://127.0.0.1:{self.port}/health")).json()
        seen["url"], seen["keys"] = self.url, len(self._keys)
        await self.aclose()

    monkeypatch.setattr(RealtimeServer, "serve_forever", serve_briefly)
    args = ["serve", "-e", "local=mock", "--port", "0", "--api-key", "k", "--max-sessions", "2"]
    result = CliRunner().invoke(cli_app, args)
    assert result.exit_code == 0, result.output
    assert seen["health"]["models"] == ["local"] and seen["health"]["max_sessions"] == 2
    assert seen["keys"] == 1 and seen["url"].startswith("ws://127.0.0.1:")
    assert "/v1/realtime" in ANSI.sub("", result.output)
    bad = CliRunner().invoke(cli_app, ["serve", "--protocol", "sip", "--port", "0"])
    assert bad.exit_code == 2 and "unknown protocol" in ANSI.sub("", bad.output)
