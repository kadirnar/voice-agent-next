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
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from voice_agent_next import (
    Agent,
    AgentSession,
    AgentState,
    AudioFrame,
    ChatMessage,
    FunctionCallOutput,
    function_tool,
)
from voice_agent_next.audio.codecs import mulaw_encode
from voice_agent_next.engine import EngineConnection, EngineOptions, S2SEngine
from voice_agent_next.engines.cascade import CascadeEngine, CascadeOptions
from voice_agent_next.events import (
    InputCommitted,
    InputTranscript,
    ResponseAudio,
    ResponseDone,
    ResponseText,
    ResponseToolCall,
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
        [AudioFrame.silence(lead, rate), synth_speech(seconds, rate), AudioFrame.silence(tail, rate)]
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
        await wait_for(lambda: any("Welcome!" in e.delta for e in events.get("agent_transcript", [])))
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
        [(item_id, played_ms)] = conn.truncations
        assert played_ms == pytest.approx(interrupted.played * 1000, abs=60)


def _history(conn: EngineConnection) -> list[Any]:
    return list(conn.chat_ctx.items)  # type: ignore[attr-defined]


def _truncated(conn: EngineConnection) -> str | None:
    """Text of the engine's interrupted assistant message (``None`` until truncated)."""
    for item in _history(conn):
        if isinstance(item, ChatMessage) and item.role == "assistant" and item.interrupted:
            return item.text
    return None
