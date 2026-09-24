"""End-to-end tests of AgentSession with the native mock engine and the cascade."""

from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import Callable
from typing import Any

import pytest

from voice_agent_next import (
    Agent,
    AgentSession,
    AgentState,
    AudioFrame,
    CascadeOptions,
    ChatMessage,
    LLMCapabilities,
    SessionOptions,
    function_tool,
)
from voice_agent_next.audio.frame import AudioFormat
from voice_agent_next.chat import AudioContent
from voice_agent_next.errors import ConfigurationError
from voice_agent_next.metrics import TurnMetrics
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import (
    MockEngine,
    MockLLM,
    MockSTT,
    MockToolCall,
    MockTTS,
    MockTurnDetector,
    synth_speech,
)
from voice_agent_next.transports import FileTransport, LoopbackTransport


class Recorder:
    def __init__(self, session: AgentSession) -> None:
        self.events: list[tuple[str, Any]] = []
        for name in ("user_transcript", "agent_transcript", "tool_call", "tool_result",
                     "interrupted", "metrics", "agent_state_changed", "error", "close",
                     "conversation_item"):  # fmt: skip
            session.on(name, self._make(name))

    def _make(self, name: str) -> Callable[[Any], None]:
        return lambda ev: self.events.append((name, ev))

    def of(self, name: str) -> list[Any]:
        return [ev for n, ev in self.events if n == name]

    def turn_metrics(self) -> list[TurnMetrics]:
        return [m for m in self.of("metrics") if isinstance(m, TurnMetrics)]

    def states(self) -> list[AgentState]:
        return [ev.new_state for ev in self.of("agent_state_changed")]


async def speak(
    transport: LoopbackTransport, seconds: float = 0.8, then_silence: float = 0.6
) -> None:
    """The simulated user says something, then stays quiet (fast, not real time)."""
    await transport.play_user_audio(synth_speech(seconds, 16_000), realtime=False)
    await transport.play_user_audio(AudioFrame.silence(then_silence, 16_000), realtime=False)


async def wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


def mock_cascade(**kw: Any) -> AgentSession:
    return AgentSession(
        stt=kw.pop("stt", MockSTT(transcripts=kw.pop("transcripts", None))),
        llm=kw.pop("llm", MockLLM(responses=kw.pop("responses", None))),
        tts=kw.pop("tts", MockTTS()),
        vad=kw.pop("vad", EnergyVAD()),
        turn_detector=kw.pop("turn_detector", None),
        cascade_options=kw.pop("cascade_options", CascadeOptions(min_endpointing_delay=0.0)),
        **kw,
    )


ENGINES = ["native", "cascade"]


def make_session(kind: str, **kw: Any) -> AgentSession:
    if kind == "native":
        return AgentSession(
            MockEngine(transcripts=kw.pop("transcripts", None), responses=kw.pop("responses", None),
                       realtime_factor=kw.pop("realtime_factor", 0.0)),
            **kw,
        )  # fmt: skip
    rf = kw.pop("realtime_factor", 0.0)
    return mock_cascade(tts=MockTTS(realtime_factor=rf), **kw)


# ----------------------------------------------------------------------------- basics


def test_session_requires_engine_or_components() -> None:
    with pytest.raises(ConfigurationError):
        AgentSession()
    with pytest.raises(ConfigurationError):
        AgentSession("mock", llm="mock")
    with pytest.raises(ConfigurationError):  # cascade without STT needs an audio LLM
        AgentSession(llm="mock", tts="mock", vad="energy")


@pytest.mark.parametrize("kind", ENGINES)
async def test_greeting_then_turn_then_close_on_hangup(kind: str) -> None:
    session = make_session(kind, transcripts=["hello agent"], responses=["Hi! Nice to meet you."])
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("be nice", greeting="Welcome."), transport)
    await wait_for(
        lambda: session.agent_state == AgentState.LISTENING and len(rec.of("agent_transcript")) >= 1
    )
    await speak(transport)
    await wait_for(lambda: any("Nice to meet" in e.delta for e in rec.of("agent_transcript")))
    await wait_for(lambda: len(rec.turn_metrics()) == 1)
    transport.end_user_audio()  # user hangs up
    await asyncio.wait_for(session.wait_closed(), 2)

    finals = [e.text for e in rec.of("user_transcript") if e.is_final]
    assert finals == ["hello agent"]
    roles = [(i.role, i.text) for i in session.history.items if isinstance(i, ChatMessage)]
    assert roles == [
        ("assistant", "Welcome."),
        ("user", "hello agent"),
        ("assistant", "Hi! Nice to meet you."),
    ]
    m = rec.turn_metrics()[0]  # (user audio was pushed faster than real time: values not asserted)
    assert m.voice_to_voice is not None and m.response_ttfb is not None
    assert not m.interrupted
    assert rec.states()[-1] == AgentState.CLOSED
    assert rec.of("close")[0].reason == "user_disconnected"
    played = sum(p.frame.duration for p in transport.played_log)
    assert played > 1.0  # greeting + answer were played


async def test_loopback_playout_keeps_its_sample_clock_through_late_wakeups() -> None:
    """Queued frames play back to back, like a sound card's: a stalled event loop (a coarse
    timer, a loaded machine) must not make the simulated device fall behind."""
    transport = LoopbackTransport(realtime_playout=True)
    await transport.start()
    for _ in range(10):
        await transport.write_audio(AudioFrame.silence(0.02, 24_000))
    await asyncio.sleep(0.03)
    time.sleep(0.06)  # noqa: ASYNC251 - the loop stalls in the middle of playback
    await asyncio.wait_for(transport.wait_for_playout(), 2)
    log = transport.played_log
    assert len(log) == 10
    for prev, cur in itertools.pairwise(log):
        assert cur.start_time == pytest.approx(prev.start_time + prev.frame.duration, abs=1e-9)
    await transport.aclose()


async def test_resampled_responses_are_played_to_the_end() -> None:
    """24 kHz engine audio on an 8 kHz line: the resampler's filter delay must not hold back
    the end of a response (soxr keeps 44 ms) until the next one starts."""
    engine = MockEngine(transcripts=["hello"], responses=["Hi! Nice to meet you."])
    session = AgentSession(engine)
    transport = LoopbackTransport(output_format=AudioFormat(8_000, 1))
    await session.start(Agent("x", greeting="Welcome."), transport)

    def samples() -> int:
        return sum(len(p.frame.data) // 2 for p in transport.played_log)

    await wait_for(lambda: session.agent_state == AgentState.LISTENING and samples() > 0)
    greeting = samples()
    assert greeting == pytest.approx(len("Welcome.") / 15 * 8_000, abs=2)
    await speak(transport)
    await wait_for(lambda: len(session.history.messages()) == 3)
    await wait_for(lambda: session.agent_state == AgentState.LISTENING)
    assert samples() - greeting == pytest.approx(len("Hi! Nice to meet you.") / 15 * 8_000, abs=2)
    await session.aclose()


@pytest.mark.parametrize("kind", ENGINES)
async def test_tool_call_round_trip(kind: str) -> None:
    calls: list[str] = []

    @function_tool
    async def get_weather(city: str) -> str:
        """Weather lookup."""
        calls.append(city)
        return f"sunny in {city}"

    session = make_session(
        kind,
        transcripts=["weather in paris?"],
        responses=[MockToolCall("get_weather", {"city": "Paris"}), "It is sunny in Paris."],
    )
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x", tools=[get_weather]), transport)
    await speak(transport)
    await wait_for(lambda: len(rec.turn_metrics()) == 1)
    await session.aclose()

    assert calls == ["Paris"]
    assert rec.of("tool_result")[0].output.output == "sunny in Paris"
    kinds = [getattr(i, "role", i.type) for i in session.history.items]
    assert kinds == ["user", "function_call", "function_call_output", "assistant"]
    m = rec.turn_metrics()[0]
    assert m.tool_calls == 1 and m.voice_to_voice is not None
    # no LISTENING flicker between the tool round and the spoken answer
    states = rec.states()
    first_speaking = states.index(AgentState.SPEAKING)
    assert AgentState.LISTENING not in states[states.index(AgentState.THINKING) : first_speaking]


@pytest.mark.parametrize("kind", ENGINES)
async def test_max_tool_steps_stops_loop(kind: str) -> None:
    @function_tool
    async def again() -> str:
        """Loop forever."""
        return "call me again"

    session = make_session(kind, transcripts=["go"], responses=[MockToolCall("again")] * 10)
    session.options.max_tool_steps = 2
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x", tools=[again]), transport)
    await speak(transport)
    await wait_for(lambda: len(rec.turn_metrics()) == 1)
    await asyncio.sleep(0.1)
    await session.aclose()
    assert len(rec.of("tool_call")) == 3  # 2 allowed rounds + the one that hit the limit
    assert session.agent_state == AgentState.CLOSED


# ----------------------------------------------------------------------- interruption


@pytest.mark.parametrize("kind", ENGINES)
async def test_barge_in_truncates_to_what_was_heard(kind: str) -> None:
    long_answer = "This is a very long answer that keeps going and going for quite a while. " * 3
    session = make_session(
        kind, transcripts=["tell me a story"], responses=[long_answer], realtime_factor=1.0
    )
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await session.start(Agent("x"), transport)
    await speak(transport, 0.6, 0.5)
    await wait_for(lambda: session.agent_state == AgentState.SPEAKING)
    await asyncio.sleep(1.0)  # let ~1 s of the answer play
    await transport.play_user_audio(synth_speech(0.4, 16_000), realtime=False)  # barge in
    await wait_for(lambda: bool(rec.of("interrupted")))
    await session.aclose()

    ev = rec.of("interrupted")[0]
    assert 0.5 < ev.played < 2.5
    assert transport.clear_times, "transport playback must be cleared"
    assistant = [
        i for i in session.history.items if isinstance(i, ChatMessage) and i.role == "assistant"
    ]
    assert assistant[-1].interrupted
    assert 0 < len(assistant[-1].text) < len(long_answer.strip())
    m = rec.turn_metrics()[0]
    assert m.interrupted and m.agent_speech_duration == pytest.approx(ev.played, abs=0.3)
    # the engine was told how much was heard
    conn = session.connection
    if kind == "native":
        assert conn.truncations and conn.truncations[0][1] == pytest.approx(
            ev.played * 1000, abs=100
        )  # type: ignore[attr-defined]
    engine_msg = conn.chat_ctx.get(assistant[-1].id)  # type: ignore[attr-defined]
    assert engine_msg.interrupted and len(engine_msg.text) < len(long_answer.strip())


async def test_interruptions_can_be_disabled() -> None:
    session = make_session("native", responses=["A long answer that takes a while to say out loud."],
                           realtime_factor=1.0, options=SessionOptions(allow_interruptions=False))  # fmt: skip
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await session.start(Agent("x"), transport)
    await speak(transport, 0.5, 0.5)
    await wait_for(lambda: session.agent_state == AgentState.SPEAKING)
    await transport.play_user_audio(synth_speech(0.3, 16_000), realtime=False)
    await asyncio.sleep(0.3)
    assert rec.of("interrupted") == []
    await session.aclose()


async def test_generate_reply_and_say_api() -> None:
    session = make_session("native", responses=["Typed answer."])
    rec = Recorder(session)
    await session.start(Agent("x"), LoopbackTransport())
    await session.generate_reply(user_input="typed question")
    await wait_for(lambda: any("Typed answer" in e.delta for e in rec.of("agent_transcript")))
    await session.say("Verbatim text.")
    await wait_for(lambda: any("Verbatim text." in e.delta for e in rec.of("agent_transcript")))
    await session.aclose()
    texts = [i.text for i in session.history.items if isinstance(i, ChatMessage)]
    assert texts == ["typed question", "Typed answer.", "Verbatim text."]


# ---------------------------------------------------------------------- cascade only


async def test_cascade_turn_detector_delays_commit_until_user_is_done() -> None:
    td = MockTurnDetector(probability=0.1)  # "user is not done"
    session = mock_cascade(
        transcripts=["I would like to", "book a table"],
        turn_detector=td,
        cascade_options=CascadeOptions(min_endpointing_delay=0.0, max_endpointing_delay=0.6),
    )
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport, 0.6, 0.55)  # pause long enough for VAD end-of-speech (0.5 s)
    await asyncio.sleep(0.2)  # within max_endpointing_delay: not committed yet
    assert not [e for e in rec.of("user_transcript") if e.is_final]
    await speak(transport, 0.6, 0.55)  # user continues -> same turn
    await wait_for(lambda: bool([e for e in rec.of("user_transcript") if e.is_final]), 3)
    await session.aclose()
    finals = [e.text for e in rec.of("user_transcript") if e.is_final]
    assert finals == ["I would like to book a table"]
    assert td.calls >= 2


async def test_cascade_confident_turn_detector_commits_fast() -> None:
    session = mock_cascade(
        transcripts=["book a table."],
        turn_detector=MockTurnDetector(),
        cascade_options=CascadeOptions(min_endpointing_delay=0.0, max_endpointing_delay=5.0),
    )
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport)
    await wait_for(lambda: bool(rec.turn_metrics()), 2)  # far below max_endpointing_delay
    await session.aclose()


async def test_cascade_ignores_noise_without_transcript() -> None:
    session = mock_cascade(stt=MockSTT(default_text=""))
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport)
    await asyncio.sleep(0.3)
    await session.aclose()
    assert not rec.of("agent_transcript")
    assert session.history.items == []


async def test_half_cascade_passes_user_audio_to_audio_llm() -> None:
    class AudioLLM(MockLLM):
        def __init__(self, **kw: Any) -> None:
            super().__init__(**kw)
            self.capabilities = LLMCapabilities(audio_input=True)

    llm = AudioLLM(responses=["I heard you."])
    session = AgentSession(llm=llm, tts=MockTTS(), vad=EnergyVAD(),
                           cascade_options=CascadeOptions(min_endpointing_delay=0.0))  # fmt: skip
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport, 0.7, 0.6)
    await wait_for(lambda: any("I heard you" in e.delta for e in rec.of("agent_transcript")))
    await session.aclose()
    user_msg = llm.requests[0].last_message("user")
    assert user_msg is not None and isinstance(user_msg.content[0], AudioContent)
    assert user_msg.content[0].frame.duration > 0.6


async def test_cascade_streams_sentences_to_tts_and_cleans_markdown() -> None:
    tts = MockTTS()
    session = mock_cascade(
        transcripts=["hi"], responses=["**Hello** there, friend. Here is `code`. Bye!"], tts=tts
    )
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport)
    await wait_for(lambda: bool(rec.turn_metrics()))
    await session.aclose()
    assert tts.requests == ["Hello there, friend.", "Here is code.", "Bye!"]
    spoken = "".join(e.delta for e in rec.of("agent_transcript"))
    assert "**" not in spoken and "`" not in spoken


async def test_file_transport_end_to_end(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from voice_agent_next.audio import read_wav, write_wav

    src = tmp_path / "in.wav"
    out = tmp_path / "out.wav"
    write_wav(src, AudioFrame.concat([AudioFrame.silence(0.3, 16_000), synth_speech(0.8, 16_000)]))
    session = make_session("native", responses=["File based answer."])
    transport = FileTransport(src, out, realtime=False, trailing_silence=0.8, hold=0.5)
    await asyncio.wait_for(session.run(Agent("x"), transport), 10)
    reply = read_wav(out)
    assert reply.sample_rate == 24_000 and reply.duration > 0.5


@pytest.mark.parametrize("kind", ENGINES)
async def test_voice_to_voice_latency_matches_external_measurement(kind: str) -> None:
    from voice_agent_next.utils import now

    if kind == "native":
        session = AgentSession(MockEngine(response_delay=0.3, realtime_factor=1.0))
        expected = 0.4 + 0.3  # engine VAD min_silence_duration + response delay
    else:
        session = mock_cascade(
            llm=MockLLM(ttft=0.3), tts=MockTTS(realtime_factor=1.0),
            vad=EnergyVAD(min_silence_duration=0.4),
        )  # fmt: skip
        expected = 0.4 + 0.3  # VAD silence + LLM time-to-first-token
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await session.start(Agent("x"), transport)
    await transport.play_user_audio(synth_speech(0.6, 16_000))  # real time
    speech_end = now()
    await transport.play_user_audio(AudioFrame.silence(1.5, 16_000))
    await wait_for(lambda: bool(rec.turn_metrics()), 5)
    await session.aclose()
    m = rec.turn_metrics()[0]
    heard = transport.played_log[0].start_time - speech_end  # what the simulated user measured
    assert m.voice_to_voice == pytest.approx(expected, abs=0.15)
    assert m.voice_to_voice == pytest.approx(heard, abs=0.08)
    assert m.end_of_turn_delay is not None and m.end_of_turn_delay == pytest.approx(0.4, abs=0.1)


async def test_cascade_turn_audio_keeps_pauses_and_audio_detector_runs_before_final() -> None:
    """Audio turn detectors see the whole turn (incl. pauses) and don't wait for the STT."""
    from voice_agent_next.turn import TurnDetector
    from voice_agent_next.utils import now

    class AudioDetector(TurnDetector):
        provider = "test"
        modality = "audio"

        def __init__(self) -> None:
            super().__init__(threshold=0.5)
            self.durations: list[float] = []
            self.started: list[float] = []

        async def _predict(self, *, audio, chat_ctx):  # type: ignore[no-untyped-def]
            self.started.append(now())
            self.durations.append(audio.duration if audio is not None else 0.0)
            return 0.1 if len(self.durations) == 1 else 0.9  # first pause: "not done"

    detector = AudioDetector()
    stt = MockSTT(transcripts=["I would like to", "book a table"], latency=0.2)
    session = mock_cascade(
        stt=stt,
        turn_detector=detector,
        cascade_options=CascadeOptions(min_endpointing_delay=0.0, max_endpointing_delay=1.5),
    )
    finals: list[float] = []  # when the STT delivered each final transcript
    stt.on("metrics", lambda m: finals.append(now()))
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport, 0.6, 0.55)
    await wait_for(lambda: bool(finals), 3)  # detector said "not done": the turn stays open
    # the audio detector started before the (slow, 0.2 s) STT delivered the first final
    assert detector.started[0] < finals[0] - 0.1
    await speak(transport, 0.6, 0.55)  # the user resumes the same turn
    await wait_for(lambda: len(detector.durations) >= 2, 3)
    await session.aclose()
    # the second prediction covers speech + pause + resumed speech (>= 0.6 + 0.55 + 0.6 s)
    assert detector.durations[1] >= 1.7


async def test_file_transport_streams_exact_lengths_without_float_drift(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Regression: 11.0 s at 16 kHz in 20 ms frames used to stall on a rounding-empty slice."""
    from voice_agent_next.audio import write_wav

    src = tmp_path / "in.wav"
    write_wav(src, synth_speech(11.0, 16_000))
    transport = FileTransport(src, realtime=False, trailing_silence=0.5, hold=0.0)
    await transport.start()
    frames = [f async for f in transport.audio_input()]
    await transport.aclose()
    assert all(f for f in frames)  # never an empty frame
    assert sum(f.samples_per_channel for f in frames) >= 16_000 * 11.5
    assert sum(f.samples_per_channel for f in frames) < 16_000 * 11.6


async def test_session_prewarms_the_engine_before_opening_the_transport() -> None:
    order: list[str] = []

    class WarmEngine(MockEngine):
        async def warmup(self) -> None:
            order.append("warmup")

    class Tracking(LoopbackTransport):
        async def start(self) -> None:
            order.append("transport")
            await super().start()

    session = AgentSession(WarmEngine())
    await session.start(Agent("x"), Tracking())
    await session.aclose()
    assert order == ["warmup", "transport"]

    order.clear()
    cold = AgentSession(WarmEngine(), options=SessionOptions(warmup=False))
    await cold.start(Agent("x"), Tracking())
    await cold.aclose()
    assert order == ["transport"]


async def test_user_turn_precedes_the_reply_even_with_a_late_transcript() -> None:
    from voice_agent_next.events import InputCommitted, InputTranscript
    from voice_agent_next.providers.mock import MockEngineConnection
    from voice_agent_next.utils import new_id

    class LateTranscript(MockEngineConnection):
        async def _commit(self) -> None:  # the transcript arrives after the answer started
            self._pending = []
            text = self._engine.stt.next_transcript(AudioFrame.empty(16_000))
            item_id = new_id("item_")
            self._emit(InputCommitted(item_id=item_id))
            self.chat_ctx.add_message("user", text, id=item_id)
            await self._start_response()
            await asyncio.sleep(0.2)
            self._emit(InputTranscript(item_id=item_id, text=text, is_final=True))

    class LateEngine(MockEngine):
        async def connect(self, options):  # type: ignore[no-untyped-def]
            conn = LateTranscript(self, options)
            self.connections.append(conn)
            return conn

    session = AgentSession(LateEngine(transcripts=["where is my order"], responses=["On its way."]))
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport)
    await wait_for(lambda: any(e.is_final for e in rec.of("user_transcript")), 3)
    await session.aclose()
    roles = [(i.role, i.text) for i in session.history.items if isinstance(i, ChatMessage)]
    assert roles == [("user", "where is my order"), ("assistant", "On its way.")]
    added = [e.item.role for e in rec.of("conversation_item") if isinstance(e.item, ChatMessage)]
    assert added.count("user") == 1  # announced once, when the transcript arrived


class LaggyTransport(LoopbackTransport):
    """A real-time speaker that reports ``lag`` s of extra output latency (a device buffer
    or a client-side jitter buffer): the listener hears everything ``lag`` s later."""

    def __init__(self, lag: float) -> None:
        super().__init__(realtime_playout=True)
        self.lag = lag

    def buffered_duration(self) -> float:
        return super().buffered_duration() + self.lag


def speaking_since(rec: Recorder) -> float:
    return next(e.timestamp for e in rec.of("agent_state_changed")
                if e.new_state == AgentState.SPEAKING)  # fmt: skip


@pytest.mark.parametrize("lag", [0.0, 0.4])
async def test_truncation_follows_the_transports_playback_position(lag: float) -> None:
    session = make_session(
        "native", transcripts=["tell me a story"], responses=["A long story. " * 20],
        realtime_factor=1.0,
    )  # fmt: skip
    rec = Recorder(session)
    user_speaking: list[float] = []
    session.on("user_state_changed",
               lambda e: user_speaking.append(e.timestamp) if e.new_state == "speaking" else None)  # fmt: skip
    transport = LaggyTransport(lag)
    await session.start(Agent("x"), transport)
    await speak(transport, 0.6, 0.5)
    await wait_for(lambda: session.agent_state == AgentState.SPEAKING)
    await asyncio.sleep(1.0)
    await transport.play_user_audio(synth_speech(0.6, 16_000), realtime=False)  # barge in
    await wait_for(lambda: bool(rec.of("interrupted")))
    await session.aclose()

    # heard = agent audio played until the user started to speak, minus what the
    # transport still had to play out (lag)
    expected = user_speaking[-1] - speaking_since(rec) - lag
    (ev,) = rec.of("interrupted")
    # loaded CI runners jitter ~0.2 s; a missing lag correction would be 0.4 s off
    assert ev.played == pytest.approx(expected, abs=0.25)
    assert session.connection.truncations[0][1] == pytest.approx(ev.played * 1000, abs=1)  # type: ignore[attr-defined]


@pytest.mark.parametrize("lag", [0.0, 0.4])
async def test_agent_keeps_speaking_until_the_listener_heard_the_reply(lag: float) -> None:
    session = make_session("native", responses=["Sure, it is done."], realtime_factor=1.0)
    rec = Recorder(session)
    transport = LaggyTransport(lag)
    await session.start(Agent("x"), transport)
    await speak(transport)
    await wait_for(lambda: AgentState.LISTENING in rec.states()[2:], timeout=8)
    await session.aclose()

    listening = [e.timestamp for e in rec.of("agent_state_changed")
                 if e.new_state == AgentState.LISTENING][-1]  # fmt: skip
    audio = sum(p.frame.duration for p in transport.played_log)
    assert listening - speaking_since(rec) == pytest.approx(audio + lag, abs=0.25)
    (m,) = rec.turn_metrics()
    assert m.agent_speech_duration == pytest.approx(audio, abs=0.05)


def test_loopback_reports_a_playback_position_only_with_real_time_playout() -> None:
    assert LoopbackTransport(realtime_playout=True).capabilities.playback_position
    assert not LoopbackTransport().capabilities.playback_position
    assert not LoopbackTransport(pausable=False).capabilities.pause
