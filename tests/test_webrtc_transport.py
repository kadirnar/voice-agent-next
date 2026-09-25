"""WebRTC transport: two in-process aiortc peers over localhost, talking to the mock engine."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import numpy as np
import pytest

from voice_agent_next import Agent, AgentSession, AudioFrame, ChatMessage
from voice_agent_next.errors import TransportError
from voice_agent_next.providers.mock import MockEngine, synth_speech
from voice_agent_next.transports import create_transport
from voice_agent_next.utils import cancel_and_wait, now

aiortc = pytest.importorskip("aiortc", reason="needs the `webrtc` extra")
av = pytest.importorskip("av")

from voice_agent_next.transports.webrtc import (  # noqa: E402
    DATA_CHANNEL_ID,
    DATA_CHANNEL_LABEL,
    OPUS_RATE,
    PROTOCOL,
    AudioPlayout,
    WebRTCAgentServer,
    WebRTCTransport,
    _force_relay,
    av_frame_to_audio,
    normalize_ice_servers,
    paced_audio_track,
)


@pytest.fixture(autouse=True)
def loopback_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """ICE host candidates on 127.0.0.1 only: deterministic, and no network interfaces needed."""
    import aioice.ice

    monkeypatch.setattr(aioice.ice, "get_host_addresses", lambda *a, **k: ["127.0.0.1"])


async def wait_for(predicate: Callable[[], Any], timeout: float = 10.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


# ------------------------------------------------------------------------ test client
class Peer:
    """A browser stand-in: sends mic audio, records agent audio and data-channel messages."""

    def __init__(self, *, data_channel: bool = True) -> None:
        self.pc = aiortc.RTCPeerConnection(aiortc.RTCConfiguration(iceServers=[]))
        self.mic = AudioPlayout()
        self.pc.addTrack(paced_audio_track(self.mic))
        self.messages: list[dict[str, Any]] = []
        self.audio: list[tuple[float, AudioFrame]] = []  # (arrival time, 48 kHz mono)
        self.channel: Any = None
        if data_channel:
            self.channel = self.pc.createDataChannel(
                DATA_CHANNEL_LABEL, negotiated=True, id=DATA_CHANNEL_ID
            )
            self.channel.on("message", self._on_message)
        self._reader: asyncio.Task[None] | None = None
        self.pc.on("track", self._on_track)

    def _on_message(self, data: str | bytes) -> None:
        assert isinstance(data, str)
        self.messages.append(json.loads(data))

    def _on_track(self, track: Any) -> None:
        self._reader = asyncio.create_task(self._read(track))

    async def _read(self, track: Any) -> None:
        from aiortc.mediastreams import MediaStreamError

        try:
            while True:
                frame = await track.recv()
                self.audio.append((now(), av_frame_to_audio(frame)))
        except MediaStreamError:
            pass

    async def offer(self) -> dict[str, Any]:
        await self.pc.setLocalDescription(await self.pc.createOffer())
        return {"type": "offer", "sdp": self.pc.localDescription.sdp}

    async def answer(self, answer: dict[str, Any]) -> None:
        await self.pc.setRemoteDescription(
            aiortc.RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )

    async def connect(self, server: WebRTCAgentServer) -> dict[str, Any]:
        answer = await server.handle_offer({**await self.offer(), "metadata": {"user": "ada"}})
        await self.answer(answer)
        await wait_for(lambda: self.pc.connectionState == "connected")
        return answer

    async def close(self) -> None:
        await self.pc.close()
        await cancel_and_wait(self._reader)

    def speak(self, seconds: float = 0.8, silence: float = 0.6) -> float:
        """Queue speech (then silence) on the mic track; returns its duration."""
        audio = AudioFrame.concat(
            [synth_speech(seconds, OPUS_RATE), AudioFrame.silence(silence, OPUS_RATE)]
        )
        self.mic.push(audio.data)
        return audio.duration

    def send(self, message: dict[str, Any]) -> None:
        self.channel.send(json.dumps(message))

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [m for m in self.messages if m.get("type") == kind]

    def transcripts(self, role: str, *, final: bool | None = None) -> list[dict[str, Any]]:
        return [
            m
            for m in self.of("transcript")
            if m["role"] == role and (final is None or m["final"] == final)
        ]

    def voiced(self, since: float = 0.0) -> float:
        """Seconds of received agent audio above the noise floor (Opus silence is not 0)."""
        return sum(f.duration for t, f in self.audio if t >= since and f.rms() > 0.01)


def mock_server(
    responses: list[str] | None = None, *, greeting: str | None = None, **kw: Any
) -> WebRTCAgentServer:
    transcripts = kw.pop("transcripts", None)
    return WebRTCAgentServer(
        lambda: AgentSession(MockEngine(responses=responses, transcripts=transcripts)),
        lambda: Agent("be brief", greeting=greeting),
        port=0,
        **kw,
    )


def http(server: WebRTCAgentServer) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=server.url, trust_env=False, timeout=10)


@pytest.fixture
async def peers() -> AsyncIterator[list[Peer]]:
    created: list[Peer] = []
    yield created
    for peer in created:
        await peer.close()


# ----------------------------------------------------------------------- conversation
async def test_conversation_audio_both_ways_over_http_signalling(peers: list[Peer]) -> None:
    server = mock_server(["Hi! Nice to meet you."], greeting="Welcome.", transcripts=["hello"])
    async with server, http(server) as client:
        peer = Peer()
        peers.append(peer)
        config = (await client.get("/config")).json()
        assert config == {"iceServers": [], "iceTransportPolicy": "all"}
        response = await client.post("/offer", json={**await peer.offer(), "metadata": {"a": 1}})
        assert response.status_code == 200, response.text
        answer = response.json()
        assert answer["type"] == "answer" and answer["session_id"].startswith("rtc_")
        assert "a=candidate:" in answer["sdp"]  # non-trickle: candidates are in the answer
        await peer.answer(answer)

        ready = await _wait_message(peer, "ready")
        assert ready["protocol"] == PROTOCOL and ready["session_id"] == answer["session_id"]
        assert (ready["sample_rate"], ready["output_sample_rate"]) == (16_000, 48_000)
        (transport,) = server.transports
        assert transport.offer["metadata"] == {"a": 1}

        # greeting: transcript on the data channel, Opus audio on the track
        await wait_for(lambda: peer.transcripts("assistant", final=True))
        await wait_for(lambda: peer.voiced() > 0.3)
        assert peer.transcripts("assistant", final=True)[0]["text"] == "Welcome."

        # one user turn: 48 kHz Opus in (16 kHz to the engine), agent speech out
        before = peer.voiced()
        peer.speak()
        await wait_for(lambda: len(peer.transcripts("assistant", final=True)) == 2)
        await wait_for(lambda: peer.voiced() - before > 1.0)
        assert [m["text"] for m in peer.transcripts("user", final=True)] == ["hello"]
        answer_text = peer.transcripts("assistant", final=True)[-1]["text"]
        assert answer_text == "Hi! Nice to meet you."
        expected = len(answer_text) / 15  # mock engine: 15 characters per second
        await asyncio.sleep(0.3)
        assert peer.voiced() - before == pytest.approx(expected, abs=0.3)
        assert transport.sent_duration == pytest.approx(len("Welcome.") / 15 + expected, abs=0.1)

        states = [m["agent"] for m in peer.of("state")]
        assert "speaking" in states and states[-1] == "listening"
        assert [m for m in peer.of("metrics") if m["kind"] == "turn"]
        (session,) = server.sessions
        texts = [(i.role, i.text) for i in session.history.items if isinstance(i, ChatMessage)]
        assert texts == [("assistant", "Welcome."), ("user", "hello"), ("assistant", answer_text)]
        assert not peer.of("error")


async def test_barge_in_clears_queued_audio_at_once(peers: list[Peer]) -> None:
    long_answer = "This answer keeps going and going for quite a while, on and on. " * 3
    server = mock_server([long_answer.strip()], transcripts=["tell me a story"])
    async with server:
        peer = Peer()
        peers.append(peer)
        await peer.connect(server)
        (transport,) = server.transports
        peer.speak(0.6, 0.5)
        await wait_for(lambda: peer.voiced() > 0.5)
        peer.speak(0.6, 0.0)  # barge in
        await wait_for(lambda: peer.of("clear"))
        cleared_at = now()
        assert transport.buffered_duration() <= transport.playout_delay + 0.05
        await wait_for(lambda: peer.transcripts("assistant", final=True))
        final = peer.transcripts("assistant", final=True)[0]
        assert final["interrupted"] is True
        assert 0 < len(final["text"]) < len(long_answer.strip())
        # the outbound track stops sending agent audio right away (allow for the jitter buffer)
        await asyncio.sleep(0.5)
        assert peer.voiced(since=cleared_at + 0.25) == 0
        heard = transport.sent_duration
        assert heard < len(long_answer) / 15 / 2
        assert final["played_ms"] / 1000 == pytest.approx(heard, abs=0.4)
        (session,) = server.sessions
        # the long answer (not a reply to the barge-in turn, which may already exist on a
        # slow runner) is the interrupted one
        [story] = [
            m
            for m in session.history.messages()
            if m.role == "assistant" and long_answer.startswith(m.text)
        ]
        assert story.interrupted


async def test_data_channel_text_app_messages_and_playout_reports(peers: list[Peer]) -> None:
    received: list[dict[str, Any]] = []

    def session_factory(transport: WebRTCTransport) -> AgentSession:
        transport.on("message", received.append)
        return AgentSession(MockEngine(responses=["Typed reply."]))

    server = WebRTCAgentServer(session_factory, lambda: Agent("hi"), port=0)
    async with server:
        peer = Peer()
        peers.append(peer)
        await peer.connect(server)
        await _wait_message(peer, "ready")
        (transport,) = server.transports
        await wait_for(lambda: peer.of("state"))

        peer.send({"type": "playout", "delay_ms": 180})
        peer.send({"type": "custom", "value": 1})
        peer.send({"type": "playout", "delay_ms": -1})
        peer.channel.send("not json")
        peer.send({"type": "text", "text": "What can you do?"})
        await wait_for(lambda: transport.playout_delay == pytest.approx(0.18))
        await wait_for(lambda: len(received) == 2)  # app messages (`text` too) are emitted
        assert received == [
            {"type": "custom", "value": 1},
            {"type": "text", "text": "What can you do?"},
        ]
        await wait_for(lambda: len(peer.of("error")) == 2)
        assert {m["code"] for m in peer.of("error")} == {"invalid_message"}
        await wait_for(lambda: peer.transcripts("assistant", final=True))
        assert peer.transcripts("assistant", final=True)[0]["text"] == "Typed reply."
        await transport.send_message({"type": "hello_app", "n": 2})
        await wait_for(lambda: peer.of("hello_app"))


async def test_messages_sent_before_the_channel_opens_are_queued() -> None:
    peer = Peer()
    transport = WebRTCTransport(connect_timeout=5)
    try:
        transport.send_message_nowait({"type": "early", "n": 1})
        answer = await transport.accept_offer(**{"sdp": (await peer.offer())["sdp"]})
        transport.send_message_nowait({"type": "early", "n": 2})
        await peer.answer(answer)
        await transport.start()
        await wait_for(lambda: len(peer.of("early")) == 2)
        assert [m["type"] for m in peer.messages][:3] == ["ready", "early", "early"]
        with pytest.raises(ValueError):
            transport.send_message_nowait({"no": "type"})
        with pytest.raises(TransportError):
            await transport.accept_offer((await peer.offer())["sdp"])  # no renegotiation
    finally:
        await transport.aclose()
        await peer.close()


# -------------------------------------------------------------------------- teardown
async def test_peer_hangup_closes_the_session(peers: list[Peer]) -> None:
    closed: list[str] = []

    def session_factory() -> AgentSession:
        session = AgentSession(MockEngine())
        session.on("close", lambda ev: closed.append(ev.reason))
        return session

    server = WebRTCAgentServer(session_factory, lambda: Agent("hi"), port=0)
    async with server:
        peer = Peer()
        await peer.connect(server)
        await wait_for(lambda: server.sessions)
        (transport,) = server.transports
        disconnected: list[bool] = []
        transport.on("disconnected", lambda: disconnected.append(True))
        await peer.close()
        await wait_for(lambda: not server.sessions and not server.transports)
        assert disconnected == [True] and closed
        assert transport.pc.connectionState == "closed"


async def test_bye_message_and_server_close_end_sessions(peers: list[Peer]) -> None:
    server = mock_server()
    async with server:
        first, second = Peer(), Peer()
        peers += [first, second]
        await first.connect(server)
        await second.connect(server)
        await wait_for(lambda: len(server.sessions) == 2)
        await _wait_message(first, "ready")
        first.send({"type": "bye"})
        await wait_for(lambda: len(server.sessions) == 1)
    assert not server.sessions and not server.transports
    await wait_for(lambda: second.pc.connectionState == "closed")


async def test_session_close_hangs_up(peers: list[Peer]) -> None:
    server = mock_server()
    async with server:
        peer = Peer()
        peers.append(peer)
        await peer.connect(server)
        await wait_for(lambda: server.sessions)
        await server.sessions[0].aclose()
        await wait_for(lambda: peer.pc.connectionState == "closed")
        await wait_for(lambda: not server.transports)


async def test_browser_style_mdns_candidates_connect_peer_reflexive(peers: list[Peer]) -> None:
    """Browsers hide host IPs behind ``<uuid>.local`` names the server may not resolve."""
    server = mock_server(greeting="Hello.")
    async with server:
        peer = Peer()
        peers.append(peer)
        offer = await peer.offer()
        assert "a=end-of-candidates" in offer["sdp"]
        hidden = re.sub(
            r" 127\.0\.0\.1 ", " 2b1d6ab8-6b4e-4d4b-9d6c-3c1e1d7c0f5e.local ", offer["sdp"]
        )
        assert ".local" in hidden and " 127.0.0.1 " not in hidden
        answer = await server.handle_offer({**offer, "sdp": hidden})
        assert "a=candidate:" in answer["sdp"]  # the server still offers its own candidates
        await peer.answer(answer)
        await wait_for(lambda: peer.transcripts("assistant", final=True))
        await wait_for(lambda: peer.voiced() > 0.2)


async def test_peer_that_never_connects_times_out() -> None:
    server = mock_server(connect_timeout=0.3)
    async with server:
        peer = Peer()
        try:
            await server.handle_offer(await peer.offer())  # the answer is never applied
            await wait_for(lambda: not server.transports, timeout=5)
            assert not server.sessions
        finally:
            await peer.close()


# ------------------------------------------------------------------ standalone mode
async def test_standalone_transport_from_create_transport() -> None:
    transport = create_transport({"type": "webrtc", "port": 0, "input_sample_rate": 24_000})
    assert isinstance(transport, WebRTCTransport)
    assert transport.input_format.sample_rate == 24_000
    starting = asyncio.create_task(transport.start())
    await wait_for(lambda: transport._signaling is not None and transport.port != 0)
    peer, other = Peer(), Peer()
    try:
        async with httpx.AsyncClient(base_url=transport.url, trust_env=False) as client:
            response = await client.post("/offer", json=await peer.offer())
            assert response.status_code == 200, response.text
            await peer.answer(response.json())
            await asyncio.wait_for(starting, 10)
            assert transport.connected
            busy = await client.post("/offer", json=await other.offer())
            assert busy.status_code == 503

        # user audio reaches audio_input at the session rate; agent audio reaches the peer
        peer.speak(0.5, 0.2)
        frames: list[AudioFrame] = []

        async def collect() -> None:
            async for frame in transport.audio_input():
                frames.append(frame)

        collector = asyncio.create_task(collect())
        await wait_for(lambda: sum(f.duration for f in frames) > 0.6)
        assert {f.sample_rate for f in frames} == {24_000}
        await transport.write_audio(synth_speech(0.4, 24_000))
        assert transport.buffered_duration() > 0.4
        await wait_for(lambda: peer.voiced() > 0.3)
        await wait_for(lambda: transport.buffered_duration() == 0, timeout=3)

        await peer.close()
        await asyncio.wait_for(collector, 10)  # the input ends when the peer leaves
    finally:
        await transport.aclose()
        await peer.close()
        await other.close()


async def test_standalone_transport_closed_while_waiting() -> None:
    transport = WebRTCTransport(port=0)
    starting = asyncio.create_task(transport.start())
    await wait_for(lambda: transport._signaling is not None)
    await transport.aclose()
    with pytest.raises(TransportError):
        await asyncio.wait_for(starting, 5)


# ------------------------------------------------------------------- HTTP signalling
async def test_http_errors_cors_and_index_page() -> None:
    server = mock_server(
        max_sessions=0,
        index_html="<h1>demo</h1>",
        cors_origins=["https://app.example"],
        ice_servers=["stun:stun.example.org:3478"],
        client_ice_servers=[{"urls": "turn:turn.example.org", "username": "u", "credential": "p"}],
    )
    async with server, http(server) as client:
        page = await client.get("/")
        assert page.status_code == 200 and page.text == "<h1>demo</h1>"
        assert page.headers["content-type"].startswith("text/html")
        config = await client.get("/config", headers={"Origin": "https://app.example"})
        assert config.json()["iceServers"] == [
            {"urls": ["turn:turn.example.org"], "username": "u", "credential": "p"}
        ]
        assert config.headers["access-control-allow-origin"] == "https://app.example"
        other = await client.get("/config", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in other.headers
        preflight = await client.options("/offer", headers={"Origin": "https://app.example"})
        assert preflight.status_code == 204
        assert (await client.get("/nope")).status_code == 404
        assert (await client.get("/offer")).status_code == 405
        bad = await client.post("/offer", content=b"{not json")
        assert bad.status_code == 400
        no_audio = await client.post("/offer", json={"type": "offer", "sdp": "v=0\r\n"})
        assert no_audio.status_code == 400 and "audio" in no_audio.json()["error"]
        wrong = await client.post("/offer", json={"type": "answer", "sdp": "m=audio"})
        assert wrong.status_code == 400
        full = await client.post("/offer", json={"type": "offer", "sdp": "m=audio 9"})
        assert full.status_code == 503 and full.json()["error"] == "too many sessions"
        # raw requests: the server answers from the headers, before reading a body
        assert (await raw_request(server, "POST /offer", "Content-Length: 300000")).startswith(
            b"HTTP/1.1 413 "
        )
        chunked = await raw_request(server, "POST /offer", "Transfer-Encoding: chunked")
        assert chunked.startswith(b"HTTP/1.1 411 ")
        assert (await raw_request(server, "GARBAGE")).startswith(b"HTTP/1.1 400 ")


async def raw_request(server: WebRTCAgentServer, line: str, *headers: str) -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    try:
        head = "\r\n".join([f"{line} HTTP/1.1" if " " in line else line, *headers])
        writer.write(head.encode() + b"\r\n\r\n")
        await writer.drain()
        return await asyncio.wait_for(reader.read(), 10)
    finally:
        writer.close()


async def test_mounting_handle_offer_without_http(peers: list[Peer]) -> None:
    server = mock_server(serve_http=False)
    async with server:
        assert server.port == 0  # never listened
        peer = Peer(data_channel=False)
        peers.append(peer)
        await peer.connect(server)
        await wait_for(lambda: server.sessions)
        (transport,) = server.transports
        transport.send_message_nowait({"type": "dropped"})  # no data channel: a no-op
        assert transport._pending_messages == []


# ------------------------------------------------------------------ units (no media)
def test_ice_server_normalization_and_policy() -> None:
    assert normalize_ice_servers(None) == []
    assert normalize_ice_servers(["stun:a:1", {"url": "turn:b", "username": "u"}]) == [
        {"urls": ["stun:a:1"]},
        {"urls": ["turn:b"], "username": "u"},
    ]
    with pytest.raises(ValueError):
        normalize_ice_servers(["http://nope"])
    with pytest.raises(ValueError):
        normalize_ice_servers([{"username": "u"}])
    with pytest.raises(TypeError):
        normalize_ice_servers([42])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="TURN"):
        WebRTCTransport(ice_servers=["stun:a"], ice_transport_policy="relay")
    with pytest.raises(ValueError):
        WebRTCTransport(ice_transport_policy="sometimes")
    relay = WebRTCTransport(ice_servers=["turn:t:3478"], ice_transport_policy="relay")
    assert relay.ice_transport_policy == "relay"


async def test_relay_policy_is_applied_to_the_ice_gatherers() -> None:
    from aioice import TransportPolicy

    peer = Peer()
    pc = aiortc.RTCPeerConnection(aiortc.RTCConfiguration(iceServers=[]))
    try:
        offer = await peer.offer()
        await pc.setRemoteDescription(aiortc.RTCSessionDescription(**offer))
        _force_relay(pc)
        gatherers = [t.receiver.transport.transport.iceGatherer for t in pc.getTransceivers()]
        assert gatherers
        assert all(g._connection._transport_policy == TransportPolicy.RELAY for g in gatherers)
    finally:
        await pc.close()
        await peer.close()


@pytest.mark.parametrize(
    ("fmt", "layout"), [("s16", "stereo"), ("s16p", "stereo"), ("fltp", "mono"), ("flt", "stereo")]
)
def test_av_frame_conversion(fmt: str, layout: str) -> None:
    ref = synth_speech(0.02, OPUS_RATE).to_float32()
    channels = 2 if layout == "stereo" else 1
    if fmt.startswith("s16"):
        data: np.ndarray[Any, Any] = np.round(ref * 32767).astype(np.int16)
    else:
        data = ref.astype(np.float32)
    stacked = np.stack([data] * channels)  # planar: (channels, samples)
    array = stacked if fmt.endswith("p") else stacked.T.reshape(1, -1)
    frame = av.AudioFrame.from_ndarray(array, format=fmt, layout=layout)
    frame.sample_rate = OPUS_RATE
    out = av_frame_to_audio(frame)
    assert (out.sample_rate, out.channels, out.samples_per_channel) == (OPUS_RATE, 1, 960)
    assert np.abs(out.to_float32() - ref).max() < 1e-3


def test_playout_timeline(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [100.0]
    monkeypatch.setattr("voice_agent_next.transports.webrtc.now", lambda: clock[0])
    transport = WebRTCTransport(playout_delay=0.1)
    playout = transport._playout
    assert transport.buffered_duration() == 0
    playout.push(bytes(2 * OPUS_RATE))  # 1 s of audio
    assert transport.buffered_duration() == pytest.approx(1.1)
    chunk = playout.pull(960)
    assert len(chunk) == 1920 and playout.sent_samples == 960
    assert transport.buffered_duration() == pytest.approx(1.1)  # 20 ms sent, 980 ms queued
    playout.paused = True
    assert playout.pull(960) == bytes(1920) and playout.sent_samples == 960
    playout.paused = False
    clock[0] += 0.02
    assert playout.clear() == OPUS_RATE - 960
    assert transport.buffered_duration() == pytest.approx(0.1)  # the last 20 ms is in flight
    clock[0] += 0.2
    assert transport.buffered_duration() == 0


async def test_write_audio_resamples_and_pause_keeps_the_queue() -> None:
    transport = WebRTCTransport(output_sample_rate=24_000)
    await transport.write_audio(synth_speech(0.5, 24_000))
    assert transport._playout.queued == pytest.approx(0.5, abs=0.03)  # resampler delay
    await transport.pause_audio()
    assert transport._playout.pull(960) == bytes(1920)
    await transport.resume_audio()
    assert transport._playout.pull(960) != bytes(1920)
    await transport.clear_audio()
    assert transport._playout.queued == 0
    await transport.aclose()
    await transport.write_audio(synth_speech(0.1, 24_000))  # after close: dropped
    assert transport._playout.queued == 0


async def _wait_message(peer: Peer, kind: str) -> dict[str, Any]:
    await wait_for(lambda: peer.of(kind))
    return peer.of(kind)[0]


# ---------------------------------------------------------- limits and auth (#156)
async def test_sessions_expire_after_the_maximum_duration(peers: list[Peer]) -> None:
    closed: list[str] = []

    def session_factory() -> AgentSession:
        session = AgentSession(MockEngine())
        session.on("close", lambda ev: closed.append(ev.reason))
        return session

    server = WebRTCAgentServer(session_factory, lambda: Agent("hi"), port=0,
                               max_session_duration=0.5, idle_timeout=None)  # fmt: skip
    async with server:
        peer = Peer()
        peers.append(peer)
        await peer.connect(server)
        error = await _wait_message(peer, "error")
        assert error["code"] == "session_expired" and error["fatal"] is True
        await wait_for(lambda: not server.sessions and not server.transports)
        assert closed == ["session_expired"]


async def test_idle_peers_are_closed_and_audio_counts_as_activity(peers: list[Peer]) -> None:
    server = mock_server(max_session_duration=None, idle_timeout=0.3)
    async with server:
        peer = Peer()
        peers.append(peer)
        await peer.connect(server)
        await wait_for(lambda: server.sessions)
        (transport,) = server.transports
        await asyncio.sleep(0.8)  # the mic track streams (silent) audio: never idle
        assert server.sessions and transport.idle_time() < 0.3
        transport.idle_time = lambda: 60.0  # type: ignore[method-assign]  # the peer went quiet
        error = await _wait_message(peer, "error")
        assert error["code"] == "session_idle"
        await wait_for(lambda: not server.sessions)


async def test_offers_need_an_allowed_origin_and_the_api_key(peers: list[Peer]) -> None:
    server = mock_server(api_keys=["rtc-key"], allowed_origins=["https://app.example.com"])
    async with server, http(server) as client:
        peer = Peer()
        peers.append(peer)
        offer = await peer.offer()
        auth = {"Authorization": "Bearer rtc-key"}
        evil = await client.post("/offer", json=offer, headers={**auth, "Origin": EVIL})
        assert evil.status_code == 403
        assert (await client.post("/offer", json=offer)).status_code == 401
        assert (await client.get("/config")).status_code == 401  # may hold TURN credentials
        assert (await client.get("/config", headers=auth)).status_code == 200
        headers = {**auth, "Origin": "https://app.example.com"}
        response = await client.post("/offer", json=offer, headers=headers)
        assert response.status_code == 200, response.text
        await peer.answer(response.json())
        await wait_for(lambda: server.sessions)


EVIL = "https://evil.example"
