"""WebSocket transport (van-ws/1): a real ``websockets`` client talks to the mock engine."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable
from typing import Any

import pytest
from websockets.asyncio.client import ClientConnection, connect
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from voice_agent_next import Agent, AgentSession, AudioFrame, ChatMessage
from voice_agent_next.metrics import TurnMetrics
from voice_agent_next.providers.mock import MockEngine, synth_speech
from voice_agent_next.session import SessionClosed
from voice_agent_next.transports import create_transport
from voice_agent_next.transports.websocket import (
    PROTOCOL,
    SessionBridge,
    WebSocketAgentServer,
    WebSocketServerTransport,
)
from voice_agent_next.utils import cancel_and_wait


async def wait_for(predicate: Callable[[], Any], timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


class Client:
    """A minimal van-ws/1 client that records everything the server sends, in order."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.log: list[tuple[str, Any]] = []  # ("audio", bytes) | ("json", dict)
        self._ws: ClientConnection | None = None
        self._reader: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Client:
        self._ws = await connect(self.url)
        self._reader = asyncio.create_task(self._read())
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    @property
    def ws(self) -> ClientConnection:
        assert self._ws is not None
        return self._ws

    async def close(self) -> None:
        await self.ws.close()
        await cancel_and_wait(self._reader)

    async def _read(self) -> None:
        try:
            async for message in self.ws:
                if isinstance(message, bytes):
                    self.log.append(("audio", message))
                else:
                    self.log.append(("json", json.loads(message)))
        except ConnectionClosed:
            pass

    async def wait_closed(self, timeout: float = 5.0) -> int | None:
        assert self._reader is not None
        await asyncio.wait_for(asyncio.shield(self._reader), timeout)
        return self.ws.close_code

    async def send(self, message: dict[str, Any] | bytes) -> None:
        await self.ws.send(message if isinstance(message, bytes) else json.dumps(message))

    async def handshake(self, **fields: Any) -> dict[str, Any]:
        hello = {"type": "hello", "protocol": PROTOCOL, "sample_rate": 16_000, "channels": 1}
        await self.send({**hello, "codec": "pcm_s16le", **fields})
        await wait_for(lambda: self.of("ready") or self.of("error"))
        assert self.of("ready"), self.of("error")
        return self.of("ready")[0]

    async def speak(
        self, seconds: float = 0.8, silence: float = 0.6, rate: int = 16_000, base64_: bool = False
    ) -> None:
        """Send speech then silence as 20 ms frames (faster than real time)."""
        audio = AudioFrame.concat([synth_speech(seconds, rate), AudioFrame.silence(silence, rate)])
        step = round(0.02 * rate) * 2
        for i in range(0, len(audio.data), step):
            chunk = audio.data[i : i + step]
            if base64_:
                await self.send({"type": "audio", "data": base64.b64encode(chunk).decode()})
            else:
                await self.send(chunk)
            await asyncio.sleep(0)

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [m for k, m in self.log if k == "json" and m.get("type") == kind]

    def transcripts(self, role: str, *, final: bool | None = None) -> list[dict[str, Any]]:
        return [
            m
            for m in self.of("transcript")
            if m["role"] == role and (final is None or m["final"] == final)
        ]

    def audio_bytes(self) -> bytes:
        return b"".join(m for k, m in self.log if k == "audio")


def mock_server(
    responses: list[str] | None = None, *, greeting: str | None = None, **kw: Any
) -> WebSocketAgentServer:
    transcripts = kw.pop("transcripts", None)
    return WebSocketAgentServer(
        lambda: AgentSession(MockEngine(responses=responses, transcripts=transcripts)),
        lambda: Agent("be brief", greeting=greeting),
        port=0,
        **kw,
    )


# --------------------------------------------------------------------- conversation


async def test_conversation_audio_both_ways_with_transcripts_state_and_metrics() -> None:
    server = mock_server(["Hi! Nice to meet you."], greeting="Welcome.", transcripts=["hello"])
    async with server, Client(server.url) as client:
        ready = await client.handshake(metadata={"user": "ada"})
        assert ready["protocol"] == PROTOCOL
        assert (ready["sample_rate"], ready["channels"], ready["codec"]) == (16_000, 1, "pcm_s16le")
        assert (ready["output_sample_rate"], ready["output_channels"]) == (24_000, 1)
        assert ready["framing"] == "binary" and ready["session_id"].startswith("ws_")

        # greeting: streamed transcript, then audio, then a final transcript
        await wait_for(lambda: client.transcripts("assistant", final=True))
        greeting = client.transcripts("assistant")
        assert greeting[0]["delta"] == "Welcome." and not greeting[0]["final"]
        assert greeting[-1]["text"] == "Welcome."
        greeting_audio = len(client.audio_bytes())
        assert greeting_audio / 48_000 == pytest.approx(len("Welcome.") / 15, abs=0.05)

        # one user turn: 16 kHz speech in, 24 kHz agent speech out
        await client.speak()
        await wait_for(lambda: len(client.transcripts("assistant", final=True)) == 2)
        assert [m["text"] for m in client.transcripts("user", final=True)] == ["hello"]
        answer = client.transcripts("assistant", final=True)[-1]
        assert answer["text"] == "Hi! Nice to meet you." and "interrupted" not in answer
        answer_audio = (len(client.audio_bytes()) - greeting_audio) / 48_000
        assert answer_audio == pytest.approx(len(answer["text"]) / 15, abs=0.05)
        # every agent audio message is at most 20 ms (960 bytes at 24 kHz)
        sizes = {len(m) for k, m in client.log if k == "audio"}
        assert max(sizes) == 960 and all(s % 2 == 0 for s in sizes)

        agent_states = [m["agent"] for m in client.of("state")]
        assert agent_states[0] == "listening" and agent_states[-1] == "listening"
        assert "thinking" in agent_states and "speaking" in agent_states
        assert "speaking" in [m["user"] for m in client.of("state")]
        turns = [m for m in client.of("metrics") if m["kind"] == "turn"]
        assert len(turns) == 1 and turns[0]["data"]["voice_to_voice"] is not None
        assert {m["kind"] for m in client.of("metrics")} >= {"turn", "engine"}

        (session,) = server.sessions
        texts = [(i.role, i.text) for i in session.history.items if isinstance(i, ChatMessage)]
        assert texts == [
            ("assistant", "Welcome."),
            ("user", "hello"),
            ("assistant", answer["text"]),
        ]
        assert not client.of("error")


async def test_barge_in_sends_clear_and_truncated_final_transcript() -> None:
    long_answer = "This answer keeps going and going for quite a while, on and on. " * 3
    server = mock_server([long_answer.strip()], transcripts=["tell me a story"])
    async with server, Client(server.url) as client:
        await client.handshake()
        await client.speak(0.6, 0.5)
        await wait_for(lambda: len(client.audio_bytes()) > 0)
        await asyncio.sleep(0.6)  # let the user hear ~0.6 s of the answer
        await client.speak(0.4, 0.0)  # barge in (and keep "speaking": no new turn yet)
        await wait_for(lambda: client.of("clear"))
        await wait_for(lambda: client.transcripts("assistant", final=True))
        await asyncio.sleep(0.3)

        final = client.transcripts("assistant", final=True)[0]
        assert final["interrupted"] is True
        assert 300 < final["played_ms"] < 2500
        assert 0 < len(final["text"]) < len(long_answer.strip())
        assert long_answer.startswith(final["text"])
        # nothing of the interrupted answer arrives after `clear`
        clear_at = next(
            i for i, (k, m) in enumerate(client.log) if k == "json" and m["type"] == "clear"
        )
        assert [k for k, _ in client.log[clear_at:]].count("audio") == 0
        # the client heard about as much audio as the session thinks was played
        heard = len(client.audio_bytes()) / 48_000
        assert heard == pytest.approx(final["played_ms"] / 1000, abs=0.35)
        assert client.of("state")[-1]["agent"] == "listening"
        (session,) = server.sessions
        assistant = session.history.messages()[-1]
        assert assistant.interrupted and assistant.text == final["text"]


async def test_typed_text_input_and_factories_receive_the_transport() -> None:
    seen: list[tuple[str, Any]] = []

    def session_factory(transport: WebSocketServerTransport) -> AgentSession:
        seen.append(("session", transport.path))
        return AgentSession(MockEngine(responses=["Typed answer.", "Yes."]))

    async def agent_factory(transport: WebSocketServerTransport) -> Agent:
        seen.append(("agent", transport.hello.get("metadata")))
        return Agent("x")

    async with (
        WebSocketAgentServer(session_factory, agent_factory, port=0) as server,
        Client(server.url + "demo?room=1") as client,
    ):
        await client.handshake(metadata={"user": "ada"})
        await client.send({"type": "text", "text": "typed question"})
        await wait_for(lambda: client.transcripts("assistant", final=True))
        assert client.transcripts("assistant", final=True)[0]["text"] == "Typed answer."
        (session,) = server.sessions
        user = session.history.messages()[0]
        assert (user.role, user.text) == ("user", "typed question")
        assert seen == [("session", "/demo?room=1"), ("agent", {"user": "ada"})]

        # malformed messages are reported but keep the connection open
        for bad in ("not json", json.dumps([1]), json.dumps({"type": "text", "text": " "})):
            await client.ws.send(bad)
        await client.send({"type": "playback", "position_ms": -5})
        await client.send({"type": "audio", "data": "***"})
        await wait_for(lambda: len(client.of("error")) == 5)
        assert {e["code"] for e in client.of("error")} == {"invalid_message"}
        await client.send({"type": "text", "text": "still there?"})
        await wait_for(lambda: len(client.transcripts("assistant", final=True)) == 2)


# --------------------------------------------------------------------- lifecycle


async def test_client_disconnect_closes_the_session() -> None:
    sessions: list[AgentSession] = []
    closed: list[SessionClosed] = []

    def session_factory() -> AgentSession:
        session = AgentSession(MockEngine())
        session.on("close", closed.append)
        sessions.append(session)
        return session

    async with WebSocketAgentServer(session_factory, lambda: Agent("x"), port=0) as server:
        async with Client(server.url) as client:
            await client.handshake()
            await wait_for(lambda: server.sessions)
            await client.speak()  # the session is live and answering
            await wait_for(lambda: client.transcripts("assistant"))
        await wait_for(lambda: closed)
        assert closed[0].reason == "user_disconnected"
        await wait_for(lambda: not server.sessions)
        assert sessions[0].closed and sessions[0].connection.closed


async def test_session_close_closes_the_connection() -> None:
    server = mock_server()
    async with server, Client(server.url) as client:
        await client.handshake()
        await wait_for(lambda: server.sessions)
        await server.sessions[0].aclose()  # e.g. the agent hangs up
        assert await client.wait_closed() == 1000
        await wait_for(lambda: not server.sessions)


async def test_server_close_ends_every_session() -> None:
    server = mock_server()
    await server.start()
    try:
        async with Client(server.url) as a, Client(server.url) as b:
            await a.handshake()
            await b.handshake()
            await wait_for(lambda: len(server.sessions) == 2)
            sessions = server.sessions
            await server.aclose()
            assert await a.wait_closed() == 1001 and await b.wait_closed() == 1001
            assert all(s.closed for s in sessions) and not server.sessions
    finally:
        await server.aclose()


async def test_max_sessions_refuses_extra_clients() -> None:
    server = mock_server(max_sessions=1)
    async with server, Client(server.url) as first, Client(server.url) as second:
        await first.handshake()
        assert await second.wait_closed() == 1013
        assert second.of("error")[0]["code"] == "server_busy"


async def test_session_factory_failure_is_reported() -> None:
    def broken() -> AgentSession:
        raise RuntimeError("no engine for you")

    server = WebSocketAgentServer(broken, lambda: Agent("x"), port=0)
    async with server, Client(server.url) as client:
        await client.handshake()
        assert await client.wait_closed() == 1011
        (error,) = client.of("error")
        assert error["code"] == "internal_error" and "no engine for you" in error["message"]


@pytest.mark.parametrize(
    ("first_message", "code"),
    [
        (b"\x00\x00", "bad_hello"),
        ("{not json", "bad_hello"),
        (json.dumps({"type": "text", "text": "hi"}), "bad_hello"),
        (json.dumps({"type": "hello", "protocol": "van-ws/9"}), "unsupported_protocol"),
        (json.dumps({"type": "hello", "codec": "opus"}), "unsupported_codec"),
        (json.dumps({"type": "hello", "sample_rate": 4000}), "bad_hello"),
        (json.dumps({"type": "hello", "channels": 6}), "bad_hello"),
        (json.dumps({"type": "hello", "framing": "xml"}), "bad_hello"),
        (json.dumps({"type": "hello", "framing": ["binary"]}), "bad_hello"),
        (None, "bad_hello"),  # nothing at all: hello timeout
    ],
)
async def test_bad_handshakes_are_rejected(first_message: str | bytes | None, code: str) -> None:
    server = mock_server(hello_timeout=0.3)
    async with server, Client(server.url) as client:
        if first_message is not None:
            await client.ws.send(first_message)
        assert await client.wait_closed() == 1002
        (error,) = client.of("error")
        assert error["code"] == code and error["fatal"] is True
        assert not server.sessions


# ------------------------------------------------------------ framing and playback


async def test_base64_framing_with_negotiated_rates() -> None:
    server = mock_server(["Base64 works."], transcripts=["over json"])
    async with server, Client(server.url) as client:
        ready = await client.handshake(
            sample_rate=8_000, output_sample_rate=16_000, framing="base64"
        )
        assert (ready["sample_rate"], ready["output_sample_rate"]) == (8_000, 16_000)
        assert ready["framing"] == "base64"
        await client.speak(rate=8_000, base64_=True)
        await wait_for(lambda: client.transcripts("assistant", final=True))
        assert [m["text"] for m in client.transcripts("user", final=True)] == ["over json"]
        assert not client.audio_bytes()  # no binary frames in base64 mode
        pcm = b"".join(base64.b64decode(m["data"]) for m in client.of("audio"))
        assert len(pcm) / 32_000 == pytest.approx(len("Base64 works.") / 15, abs=0.05)
        assert max(len(base64.b64decode(m["data"])) for m in client.of("audio")) == 640


def raw_transport_server(
    queue: asyncio.Queue[WebSocketServerTransport], **options: Any
) -> Callable[[ServerConnection], Any]:
    async def handler(websocket: ServerConnection) -> None:
        transport = WebSocketServerTransport(websocket, **options)
        try:
            await transport.start()
        except Exception:
            return
        await queue.put(transport)
        await transport.wait_disconnected()
        await transport.aclose()

    return handler


async def test_playback_reports_drive_buffered_duration() -> None:
    queue: asyncio.Queue[WebSocketServerTransport] = asyncio.Queue()
    async with serve(raw_transport_server(queue), "127.0.0.1", 0) as ws_server:
        port = ws_server.sockets[0].getsockname()[1]
        async with Client(f"ws://127.0.0.1:{port}/") as client:
            await client.handshake()
            transport = await queue.get()
            assert transport.capabilities.playback_position and transport.connected
            assert transport.buffered_duration() == 0.0

            await transport.write_audio(AudioFrame.silence(1.0, 24_000))
            assert transport.buffered_duration() == pytest.approx(1.0, abs=0.1)  # wall clock
            await wait_for(lambda: len(client.audio_bytes()) == 48_000)
            assert [len(m) for k, m in client.log if k == "audio"] == [960] * 50
            await asyncio.sleep(0.3)
            assert transport.buffered_duration() == pytest.approx(0.7, abs=0.15)

            # the client reports it has not played anything yet (e.g. still buffering)
            await client.send({"type": "playback", "position_ms": 0})
            await wait_for(lambda: transport.buffered_duration() > 0.9)
            # ... then that it played everything
            await client.send({"type": "playback", "position_ms": 1000})
            await wait_for(lambda: transport.buffered_duration() == 0.0)
            await asyncio.wait_for(transport.wait_for_playout(), 1)

            # clear: queued audio is dropped and reports older than the clear are ignored
            await transport.write_audio(AudioFrame.silence(0.5, 24_000))
            await transport.clear_audio()
            assert transport.buffered_duration() == 0.0
            await client.send({"type": "playback", "position_ms": 500})  # stale
            await wait_for(lambda: client.of("clear"))
            assert transport.buffered_duration() == 0.0
            await transport.write_audio(AudioFrame.silence(0.4, 24_000))
            assert transport.buffered_duration() == pytest.approx(0.4, abs=0.1)
            await wait_for(lambda: len(client.audio_bytes()) >= 48_000 + 19_200)
            await client.send({"type": "mark", "played_ms": 1400})  # `mark` is an alias
            await wait_for(lambda: transport.buffered_duration() == 0.0)


async def test_partial_samples_and_app_messages() -> None:
    queue: asyncio.Queue[WebSocketServerTransport] = asyncio.Queue()
    async with serve(raw_transport_server(queue), "127.0.0.1", 0) as ws_server:
        port = ws_server.sockets[0].getsockname()[1]
        async with Client(f"ws://127.0.0.1:{port}/") as client:
            await client.handshake(sample_rate=48_000, channels=2)
            transport = await queue.get()
            assert (transport.input_format.sample_rate, transport.input_format.channels) == (
                48_000,
                2,
            )
            messages: list[dict[str, Any]] = []
            transport.on("message", messages.append)
            await client.send(b"\x01\x00\x02\x00\x03")  # 1 stereo sample + 1 byte
            await client.send(b"\x00\x04\x00")  # completes the 2nd sample
            await client.send({"type": "custom", "value": 42})
            frames = transport.audio_input()
            first = await asyncio.wait_for(anext(frames), 1)
            second = await asyncio.wait_for(anext(frames), 1)
            assert first.data + second.data == b"\x01\x00\x02\x00\x03\x00\x04\x00"
            assert (first.sample_rate, first.channels) == (48_000, 2)
            await wait_for(lambda: messages)
            assert messages == [{"type": "custom", "value": 42}]

            await transport.send_message({"type": "custom_reply", "ok": True, "nan": float("nan")})
            await wait_for(lambda: client.of("custom_reply"))
            assert client.of("custom_reply") == [{"type": "custom_reply", "ok": True, "nan": None}]
        # the client left: the input stream ends
        await asyncio.wait_for(transport.wait_disconnected(), 2)
        assert [f async for f in frames] == []


async def test_standalone_transport_from_create_transport() -> None:
    transport = create_transport({"type": "websocket", "host": "127.0.0.1", "port": 0})
    assert isinstance(transport, WebSocketServerTransport)
    await transport.listen()
    assert transport.port != 0 and transport.url == f"ws://127.0.0.1:{transport.port}/"

    session = AgentSession(MockEngine(responses=["Standalone answer."]))
    bridge = SessionBridge(session, transport)
    metrics: list[TurnMetrics] = []
    session.on("metrics", lambda m: metrics.append(m) if isinstance(m, TurnMetrics) else None)
    run = asyncio.create_task(session.run(Agent("x", greeting="Hello there."), transport))
    try:
        async with Client(transport.url) as client:
            await client.handshake()
            await wait_for(lambda: client.transcripts("assistant", final=True))
            assert client.transcripts("assistant", final=True)[0]["text"] == "Hello there."
            # a second client is refused while the first one is connected
            async with Client(transport.url) as intruder:
                assert await intruder.wait_closed() == 1013
                assert intruder.of("error")[0]["code"] == "server_busy"
            await client.send({"type": "text", "text": "hi"})
            await wait_for(lambda: len(client.transcripts("assistant", final=True)) == 2)
            assert len(client.audio_bytes()) > 48_000  # greeting + answer
        # the client left: the session ends
        await asyncio.wait_for(run, 5)
        assert session.closed
    finally:
        await bridge.aclose()
        await cancel_and_wait(run)
        await transport.aclose()


async def test_standalone_transport_closed_while_waiting_for_a_client() -> None:
    transport = WebSocketServerTransport(port=0)
    start = asyncio.create_task(transport.start())
    await wait_for(lambda: transport.port != 0)
    await transport.aclose()
    with pytest.raises(Exception, match="closed"):
        await asyncio.wait_for(start, 2)
