"""Moshi / PersonaPlex engine (``providers/moshi.py``) against the fake ``/api/chat`` server.

Everything runs offline: :class:`FakeMoshiServer` speaks the real binary protocol with
real Ogg/Opus pages on ``127.0.0.1``. The test against a real local server at the end is
marked ``integration`` (set ``MOSHI_URL``).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable
from typing import Any

import numpy as np
import pytest

pytest.importorskip("sphn")

from voice_agent_next import Agent, AgentSession, AgentState, ChatMessage
from voice_agent_next.audio import AudioFrame
from voice_agent_next.engine import EngineOptions
from voice_agent_next.errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderTimeoutError,
)
from voice_agent_next.events import (
    EngineErrorEvent,
    EngineEvent,
    EngineStatus,
    InputCommitted,
    InputSpeechStarted,
    InputSpeechStopped,
    ResponseAudio,
    ResponseDone,
    ResponseStarted,
    ResponseText,
)
from voice_agent_next.metrics import EngineMetrics, TurnMetrics
from voice_agent_next.providers.mock import synth_speech
from voice_agent_next.providers.moshi import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    MoshiConnection,
    MoshiEngine,
    OpusDecoder,
    OpusEncoder,
)
from voice_agent_next.providers.personaplex import PersonaPlexEngine
from voice_agent_next.registry import create
from voice_agent_next.testing.moshi import FakeMoshiServer
from voice_agent_next.transports import LoopbackTransport


# ----------------------------------------------------------------------------- helpers
@pytest.fixture
async def fake() -> AsyncIterator[Callable[..., Any]]:
    """Factory starting fake servers that are closed (and checked) after the test."""
    servers: list[FakeMoshiServer] = []

    async def make(**kw: Any) -> FakeMoshiServer:
        server = FakeMoshiServer(**kw)
        await server.start()
        servers.append(server)
        return server

    yield make
    for server in servers:
        await server.aclose()
        assert server.errors == [], f"protocol violations: {server.errors}"


class Events:
    """Collects the events of an engine connection in the background."""

    def __init__(self, conn: MoshiConnection) -> None:
        self.items: list[EngineEvent] = []
        self._task = asyncio.create_task(self._pump(conn))

    async def _pump(self, conn: MoshiConnection) -> None:
        async for ev in conn.events():
            self.items.append(ev)

    def of(self, kind: type[Any]) -> list[Any]:
        return [e for e in self.items if isinstance(e, kind)]

    def kinds(self) -> list[str]:
        """Event types without the audio chunks."""
        return [e.type for e in self.items if not isinstance(e, ResponseAudio)]

    async def close(self) -> None:
        await asyncio.wait_for(self._task, 5)


async def wait_for(cond: Callable[[], bool], timeout: float = 10.0) -> None:
    async with asyncio.timeout(timeout):
        while not cond():
            await asyncio.sleep(0.01)


async def connect(engine: MoshiEngine, **opts: Any) -> MoshiConnection:
    conn = await engine.connect(EngineOptions(**opts))
    assert isinstance(conn, MoshiConnection)
    return conn


async def speak(conn: MoshiConnection, seconds: float, *, realtime: bool = False) -> None:
    """User speech (16 kHz, resampled by the engine), in 20 ms frames."""
    for i in range(round(seconds / 0.02)):
        await conn.send_audio(synth_speech(0.02, 16_000, offset=i * 320))
        if realtime:
            await asyncio.sleep(0.02)


async def silence(conn: MoshiConnection, seconds: float, *, realtime: bool = False) -> None:
    for _ in range(round(seconds / 0.02)):
        await conn.send_audio(AudioFrame.silence(0.02, 16_000))
        if realtime:
            await asyncio.sleep(0.02)


def text_of(events: Events, response_id: str) -> str:
    return "".join(e.delta for e in events.of(ResponseText) if e.response_id == response_id)


# ------------------------------------------------------------------------------ codec
def test_opus_round_trip_keeps_timing_and_level() -> None:
    enc, dec = OpusEncoder(), OpusDecoder()
    speech = synth_speech(0.96, SAMPLE_RATE)
    pages = b""
    out: list[AudioFrame] = []
    for i in range(48):  # 20 ms frames, as a transport delivers them
        chunk = enc.encode(speech.slice(i * 0.02, (i + 1) * 0.02))
        if chunk:
            if not pages:
                assert chunk.startswith(b"OggS")  # the first page starts the Ogg stream
            pages += chunk
            out.append(dec.decode(chunk))
    assert enc.buffered == 0.0
    decoded = AudioFrame.concat(out)
    assert decoded.samples_per_channel == 12 * FRAME_SAMPLES  # 0.96 s in 80 ms Opus frames
    assert decoded.dbfs() == pytest.approx(speech.dbfs(), abs=1.5)
    assert len(pages) < speech.duration * 24_000  # compressed (~24 kbit/s at most)


def test_opus_encoder_buffers_partial_frames() -> None:
    enc = OpusEncoder(frame_samples=960)
    assert enc.encode(AudioFrame.silence(0.03, SAMPLE_RATE)) == b""
    assert enc.buffered == pytest.approx(0.03)
    assert enc.encode(AudioFrame.silence(0.02, SAMPLE_RATE)).startswith(b"OggS")
    assert enc.buffered == pytest.approx(0.01)


# ------------------------------------------------------------------------ endpoint
def test_endpoint_urls_and_query() -> None:
    opts = EngineOptions()
    assert MoshiEngine().endpoint(opts) == "ws://localhost:8998/api/chat"
    assert MoshiEngine(base_url="https://box:9000").endpoint(opts) == "wss://box:9000/api/chat"
    assert MoshiEngine(base_url="127.0.0.1:8998").endpoint(opts) == "ws://127.0.0.1:8998/api/chat"
    url = MoshiEngine(base_url="ws://h/custom/chat?x=1", text_temperature=0.6, seed=7).endpoint(opts)
    assert url == ("ws://h/custom/chat?x=1&text_temperature=0.6&text_seed=7&audio_seed=7&seed=7")
    with pytest.raises(ConfigurationError):
        MoshiEngine(base_url="ftp://h").endpoint(opts)
    with pytest.raises(ConfigurationError):
        MoshiEngine(frame_duration=0.05)


def test_tls_verification_defaults_to_off_for_localhost_only() -> None:
    local = MoshiEngine()._connect_kwargs("wss://localhost:8998/api/chat")
    assert local["ssl"].check_hostname is False
    assert "ssl" not in MoshiEngine()._connect_kwargs("wss://gpu.example.com/api/chat")
    forced = MoshiEngine(ssl_verify=False)._connect_kwargs("wss://gpu.example.com/api/chat")
    assert forced["ssl"].check_hostname is False


def test_registry_and_capabilities() -> None:
    engine = create("engine", "moshi/moshika")
    assert isinstance(engine, MoshiEngine) and engine.model == "moshika"
    caps = engine.capabilities
    assert caps.full_duplex and caps.server_turn_detection and caps.native_audio
    assert not (caps.tool_calling or caps.text_input or caps.truncation)
    assert not caps.input_transcription and caps.output_transcription
    assert engine.input_sample_rate == engine.output_sample_rate == 24_000
    pp = create("engine", "personaplex")
    assert isinstance(pp, PersonaPlexEngine) and pp.model == "personaplex-7b-v1"
    assert pp.capabilities.full_duplex


def test_personaplex_prompts_in_query() -> None:
    engine = PersonaPlexEngine(base_url="ws://h:1", voice="NATM1")
    url = engine.endpoint(EngineOptions(instructions="You work for Acme & Co."))
    assert url == "ws://h:1/api/chat?voice_prompt=NATM1.pt&text_prompt=You+work+for+Acme+%26+Co."
    url = engine.endpoint(EngineOptions(voice="custom.wav"))
    assert "voice_prompt=custom.wav" in url
    assert "text_prompt=You+enjoy+having+a+good+conversation." in url


# --------------------------------------------------------------------- handshake
@pytest.mark.parametrize("flavor", ["python", "rust"])
async def test_handshake_flavors(fake: Callable[..., Any], flavor: str) -> None:
    server = await fake(flavor=flavor)
    conn = await connect(MoshiEngine(base_url=server.url))
    try:
        if flavor == "rust":
            assert conn.server_version == (0, 0)
            assert conn.server_metadata is not None
            assert conn.server_metadata["instance_name"] == "fake"
        else:
            assert conn.server_version is None and conn.server_metadata is None
        assert conn.connections == 1
    finally:
        await conn.aclose()


async def test_connection_refused_is_a_connection_error() -> None:
    engine = MoshiEngine(base_url="ws://127.0.0.1:9", connect_timeout=5)
    with pytest.raises(ProviderConnectionError, match=r"moshi\.server"):
        await engine.connect(EngineOptions())


async def test_busy_server_times_out_waiting_for_the_handshake(fake: Callable[..., Any]) -> None:
    server = await fake(handshake=False)
    with pytest.raises(ProviderTimeoutError, match="one conversation at a time"):
        await MoshiEngine(base_url=server.url, connect_timeout=0.5).connect(EngineOptions())


async def test_http_rejection_maps_to_library_errors(fake: Callable[..., Any]) -> None:
    server = await fake(reject_status=403)
    with pytest.raises(AuthenticationError):
        await MoshiEngine(base_url=server.url).connect(EngineOptions())


async def test_personaplex_sends_the_prompts(fake: Callable[..., Any]) -> None:
    server = await fake(require_prompts=True, greeting="Hello")
    engine = PersonaPlexEngine(base_url=server.url)
    conn = await connect(engine, instructions="You are Alex, an astronaut.", voice="VARF3")
    await conn.aclose()
    assert server.connection.query == {
        "voice_prompt": "VARF3.pt",
        "text_prompt": "You are Alex, an astronaut.",
    }
    with pytest.raises(ProviderConnectionError):  # the plain Moshi engine sends no prompts
        await MoshiEngine(base_url=server.url).connect(EngineOptions())


# ------------------------------------------------------------------------ events
async def test_greeting_and_reply_events(fake: Callable[..., Any]) -> None:
    server = await fake(greeting="Hi there how are you", replies=["Sure thing my friend"])
    engine = MoshiEngine(base_url=server.url)
    metrics: list[EngineMetrics] = []
    engine.on("metrics", metrics.append)
    conn = await connect(engine)
    events = Events(conn)
    # nothing is sent: the keep-alive streams silence, so the model runs and greets
    await wait_for(lambda: len(events.of(ResponseDone)) == 1)
    greeting = events.of(ResponseStarted)[0].response_id
    assert text_of(events, greeting) == " Hi there how are you"
    assert events.kinds()[:2] == ["response_started", "response_text"]
    assert InputCommitted not in {type(e) for e in events.items}  # nobody spoke yet

    await asyncio.sleep(0.4)  # the greeting's tail has been played (handover_delay)
    await speak(conn, 0.8)
    await silence(conn, 1.2)
    await wait_for(lambda: len(events.of(ResponseDone)) == 2)
    await conn.aclose()
    await events.close()

    reply = events.of(ResponseStarted)[1].response_id
    assert text_of(events, reply) == " Sure thing my friend"
    kinds = events.kinds()
    turn = kinds[kinds.index("input_speech_started") :]
    assert turn[:4] == [
        "input_speech_started", "input_speech_stopped", "input_committed", "response_started"
    ]  # fmt: skip
    stopped = events.of(InputSpeechStopped)[0]
    assert stopped.audio_time is not None
    assert stopped.audio_time == pytest.approx(events.of(InputSpeechStarted)[0].audio_time + 0.8, abs=0.15)  # fmt: skip
    for done in events.of(ResponseDone):
        assert done.status == "completed"
    audio = [e for e in events.items if isinstance(e, ResponseAudio) and e.response_id == reply]
    spoken = sum(e.frame.duration for e in audio)
    # the reply (1.04 s of speech) plus pre-roll and the closing pause
    assert 1.04 <= spoken <= 1.04 + 0.08 + 0.64 + 0.2
    assert all(e.frame.sample_rate == 24_000 for e in audio)
    assert [u.completed for u in server.utterances] == [True, True]
    assert len(metrics) == 2
    assert metrics[0].ttfb is None  # unprompted greeting
    assert metrics[1].ttfb is not None and metrics[1].ttfb >= 0
    assert metrics[1].output_text_tokens == 4
    assert metrics[1].output_audio_tokens >= 13


async def test_user_talking_over_the_agent_is_left_to_the_model(fake: Callable[..., Any]) -> None:
    server = await fake(greeting=" ".join(["word"] * 16), replies=["Go on"], yield_after=0.4)
    conn = await connect(MoshiEngine(base_url=server.url))
    events = Events(conn)
    await wait_for(lambda: len(events.of(ResponseText)) >= 2)
    # the user talks over the greeting: no InputSpeechStarted while the agent speaks
    await speak(conn, 1.0, realtime=True)
    assert not events.of(InputSpeechStarted) or events.of(ResponseDone)
    await silence(conn, 1.5)
    await wait_for(lambda: len(events.of(ResponseDone)) == 2)
    await conn.aclose()
    await events.close()

    greeting = server.utterances[0]
    assert not greeting.completed  # the fake yielded to the user, like Moshi does
    kinds = events.kinds()
    first_done = kinds.index("response_done")
    # the user was reported only once the agent had fallen silent
    assert kinds.index("input_speech_started") > first_done
    assert "input_committed" in kinds[first_done:]
    assert [d.status for d in events.of(ResponseDone)] == ["completed", "completed"]


async def test_report_overlap_reports_speech_over_the_agent(fake: Callable[..., Any]) -> None:
    server = await fake(greeting=" ".join(["word"] * 16), yield_after=5.0)
    conn = await connect(MoshiEngine(base_url=server.url, report_overlap=True))
    events = Events(conn)
    await wait_for(lambda: len(events.of(ResponseText)) >= 2)
    await speak(conn, 0.5, realtime=True)
    await wait_for(lambda: bool(events.of(InputSpeechStarted)))
    assert not events.of(ResponseDone)  # reported while the agent still talks
    await conn.aclose()
    await events.close()


async def test_cancel_mutes_the_agent_until_its_next_pause(fake: Callable[..., Any]) -> None:
    server = await fake(
        greeting=" ".join(["word"] * 12), replies=["Second reply here"], words_per_second=8
    )
    conn = await connect(MoshiEngine(base_url=server.url))
    events = Events(conn)
    await wait_for(lambda: len(events.of(ResponseText)) >= 2)
    await conn.interrupt()
    await wait_for(lambda: bool(events.of(ResponseDone)))
    assert [d.status for d in events.of(ResponseDone)] == ["cancelled"]
    muted_at = len(events.items)
    await wait_for(lambda: server.utterances[0].completed)
    await silence(conn, 1.0)  # the model pauses: the mute ends
    assert not [e for e in events.items[muted_at:] if isinstance(e, (ResponseAudio, ResponseText))]
    await speak(conn, 0.6)
    await silence(conn, 1.5)
    await wait_for(lambda: len(events.of(ResponseDone)) == 2)
    await conn.aclose()
    await events.close()
    second = events.of(ResponseStarted)[1].response_id
    assert text_of(events, second) == " Second reply here"
    assert events.of(ResponseDone)[1].status == "completed"


async def test_text_only_controls_are_ignored(
    fake: Callable[..., Any], caplog: pytest.LogCaptureFixture
) -> None:
    server = await fake()
    conn = await connect(MoshiEngine(base_url=server.url))
    await conn.send_text("hello")
    await conn.create_response()
    await conn.say("verbatim")
    await conn.commit_input()
    await conn.clear_input()
    await conn.update(instructions="new")
    await conn.aclose()
    assert "no text input" in caplog.text and "decides by itself" in caplog.text


async def test_server_error_message_is_reported(fake: Callable[..., Any]) -> None:
    server = await fake()
    conn = await connect(MoshiEngine(base_url=server.url))
    events = Events(conn)
    await server.send_error("model overloaded")
    await wait_for(lambda: bool(events.of(EngineErrorEvent)))
    err = events.of(EngineErrorEvent)[0]
    assert err.recoverable and "model overloaded" in str(err.error)
    assert conn.errors == ["model overloaded"]
    await conn.aclose()
    await events.close()


# ---------------------------------------------------------------------- reconnects
async def test_reconnects_after_a_drop(fake: Callable[..., Any]) -> None:
    server = await fake(greeting=" ".join(["word"] * 20), flavor="rust")
    conn = await connect(MoshiEngine(base_url=server.url))
    events = Events(conn)
    await wait_for(lambda: len(events.of(ResponseText)) >= 2)
    await server.drop()
    await wait_for(lambda: [s.status for s in events.of(EngineStatus)] == ["reconnecting", "reconnected"])  # fmt: skip
    assert events.of(ResponseDone)[0].status == "incomplete"
    # the new server session greets again: the conversation continues on it
    await wait_for(lambda: len(events.of(ResponseStarted)) == 2)
    assert conn.connections == 2 and len(server.connections) == 2
    assert server.connections[1].utterances
    await conn.aclose()
    await events.close()


async def test_step_limit_close_reconnects(fake: Callable[..., Any]) -> None:
    server = await fake(max_steps=10)  # the Rust server closes after max_steps
    conn = await connect(MoshiEngine(base_url=server.url))
    events = Events(conn)
    await silence(conn, 1.0)
    await wait_for(lambda: conn.connections >= 2)
    assert "reconnecting" in [s.status for s in events.of(EngineStatus)]
    await conn.aclose()
    await events.close()


async def test_gives_up_when_reconnecting_is_disabled(fake: Callable[..., Any]) -> None:
    server = await fake()
    conn = await connect(MoshiEngine(base_url=server.url, reconnect=False))
    events = Events(conn)
    await server.drop()
    await events.close()  # the connection closes itself
    errors = events.of(EngineErrorEvent)
    assert len(errors) == 1 and not errors[0].recoverable
    assert conn.closed


async def test_gives_up_after_failed_reconnects(fake: Callable[..., Any]) -> None:
    server = await fake()
    conn = await connect(MoshiEngine(base_url=server.url, max_reconnect_attempts=1, connect_timeout=2))
    events = Events(conn)
    server.reject_status = 503
    await server.drop()
    await events.close()
    assert [s.status for s in events.of(EngineStatus)] == ["reconnecting"]
    assert not events.of(EngineErrorEvent)[-1].recoverable


# ------------------------------------------------------- end-to-end with AgentSession
def history(session: AgentSession) -> list[tuple[str, str]]:
    return [(i.role, i.text) for i in session.history.items if isinstance(i, ChatMessage)]


async def test_session_full_duplex_conversation(fake: Callable[..., Any]) -> None:
    server = await fake(
        greeting="Hello there", replies=["Nice to meet you", "Sure go ahead"], yield_after=0.4
    )
    session = AgentSession(MoshiEngine(base_url=server.url))
    seen: dict[str, list[Any]] = {"interrupted": [], "metrics": [], "agent_transcript": []}
    for name, items in seen.items():
        session.on(name, items.append)
    transport = LoopbackTransport(realtime_playout=True)
    # a greeting request is ignored (Moshi decides when to speak) but must not break anything
    await session.start(Agent("Be nice.", greeting="Welcome!"), transport)
    await wait_for(lambda: len(history(session)) == 1 and session.agent_state == AgentState.LISTENING)  # fmt: skip

    async def say(seconds: float) -> None:  # the caller: real-time microphone audio
        # one paced stream: pacing 20 ms pieces one call at a time adds every late wake-up
        # up, and the engine's keep-alive fills the lag with silence that Moshi steps on
        # (a loaded macOS runner lost the overlap the fake yields after)
        pieces = [synth_speech(0.02, 16_000, offset=i * 320) for i in range(round(seconds / 0.02))]
        await transport.play_user_audio(AudioFrame.concat(pieces), realtime=True)

    async def quiet(seconds: float) -> None:
        await transport.play_user_audio(AudioFrame.silence(seconds, 16_000), realtime=True)

    await say(0.6)
    await quiet(1.0)
    await wait_for(lambda: len([m for m in seen["metrics"] if isinstance(m, TurnMetrics)]) == 1)
    await wait_for(lambda: session.agent_state == AgentState.LISTENING)
    # the user talks over the next reply: the model yields, the session does not fight it
    await say(0.5)
    await quiet(0.6)
    await wait_for(lambda: session.agent_state == AgentState.SPEAKING)
    await say(0.8)  # the fake yields after 0.4 s of overlap, then answers "Okay."
    await quiet(1.5)
    await wait_for(lambda: len(server.utterances) == 4 and session.agent_state == AgentState.LISTENING)  # fmt: skip
    # the last reply reaches the history once its text is complete: on a slow runner that
    # can be after the agent went quiet
    await wait_for(lambda: len([r for r, _ in history(session) if r == "assistant"]) >= 4)
    transport.end_user_audio()
    await asyncio.wait_for(session.wait_closed(), 5)

    assert server.errors == []
    assert seen["interrupted"] == []  # the session never cancelled or truncated Moshi
    agent = [text for role, text in history(session) if role == "assistant"]
    assert agent[:2] == ["Hello there", "Nice to meet you"]
    assert not server.utterances[2].completed  # Moshi yielded to the user
    assert agent[2] in {"Sure", "Sure go", "Sure go ahead"} and agent[3] == "Okay."
    turns = [m for m in seen["metrics"] if isinstance(m, TurnMetrics)]
    assert turns and turns[0].voice_to_voice is not None
    assert 0.3 < turns[0].voice_to_voice < 1.5  # 0.4 s model pause + frame/playout slack
    played = np.concatenate([p.frame.to_float32() for p in transport.played_log])
    assert np.sqrt(np.mean(played**2)) > 0.01  # agent speech was played


# ------------------------------------------------------------- real local server
@pytest.mark.integration
async def test_real_moshi_server() -> None:
    """Needs a running server: ``MOSHI_URL=ws://localhost:8998 pytest -m integration``."""
    url = os.environ.get("MOSHI_URL")
    if not url:
        pytest.skip("set MOSHI_URL to a running moshi server")
    conn = await connect(MoshiEngine(base_url=url))
    events = Events(conn)
    await silence(conn, 3.0, realtime=True)
    await speak(conn, 1.0, realtime=True)
    await silence(conn, 5.0, realtime=True)
    await conn.aclose()
    await events.close()
    assert events.of(ResponseAudio), "the model said nothing in 9 s"
    assert events.of(ResponseText)
    assert conn.max_lag < 1.0
