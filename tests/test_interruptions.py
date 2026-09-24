"""Interruption policy: backchannel filter, overlap verdicts, pause & resume in the session.

Session tests drive the loopback transport with the mock engine / cascade. Timings are
asserted with generous tolerances (CI runners are slow, Windows timers are coarse).
"""

from __future__ import annotations

import asyncio
import itertools
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
    SessionOptions,
    UserState,
)
from voice_agent_next.engine import EngineConnection, EngineOptions
from voice_agent_next.events import EngineEvent, InputSpeechStarted
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import (
    MockEngine,
    MockEngineConnection,
    MockLLM,
    MockSTT,
    MockTTS,
    synth_speech,
)
from voice_agent_next.session import AgentFalseInterruption, Interrupted
from voice_agent_next.session.interruptions import (
    BackchannelFilter,
    InterruptionPolicy,
    Overlap,
    Verdict,
    backchannel_words_for,
    split_words,
)
from voice_agent_next.transports import LoopbackTransport
from voice_agent_next.utils import now

ANSWER = "Here is a fairly long answer that the user is going to talk over at some point."
SR = 16_000

# ------------------------------------------------------------------ backchannel filter


def test_split_words_normalizes() -> None:
    assert split_words("Uh-huh... OKAY!") == ["uh", "huh", "okay"]
    assert split_words("Mmmm, yeahhh — right’s") == ["m", "yeah", "right's"]
    assert split_words("好的，我知道") == ["好", "的", "我", "知", "道"]
    assert split_words("  ") == []


def test_backchannel_filter() -> None:
    en = BackchannelFilter(backchannel_words_for("en-US"))
    assert en.meaningful_words("Mm-hmm, okay. Yeah!") == []
    assert en.meaningful_words("Uh huh, all right") == []
    assert en.meaningful_words("yeah but wait") == ["but", "wait"]
    assert en.is_backchannel("Mhm.") and not en.is_backchannel("stop") and not en.is_backchannel("")
    tr = BackchannelFilter(backchannel_words_for("tr"))
    assert tr.is_backchannel("Hı hı, tamam") and tr.is_backchannel("okay")  # + English core
    zh = BackchannelFilter(backchannel_words_for("zh-CN"))
    assert zh.meaningful_words("嗯嗯，好的，等一下") == ["等", "一", "下"]
    # English fillers that are words elsewhere ("er" = "he" in German) stay out
    assert "er" in backchannel_words_for(None) and "er" not in backchannel_words_for("de")


def test_policy_options_and_validation() -> None:
    assert InterruptionPolicy(min_duration=0).immediate
    assert not InterruptionPolicy(min_duration=0, min_words=1).immediate
    assert InterruptionPolicy().pauses
    assert not InterruptionPolicy(resume=False).pauses
    assert not InterruptionPolicy(false_interruption_timeout=None).pauses
    assert (
        InterruptionPolicy(min_words=3).words_needed == 3 and InterruptionPolicy().words_needed == 1
    )
    policy = InterruptionPolicy.from_options(SessionOptions(backchannel_words=["bof"]))
    assert policy.backchannels.is_backchannel("Bof!") and not policy.backchannels.is_backchannel(
        "ok"
    )
    localized = InterruptionPolicy.from_options(SessionOptions(), language="tr")
    assert localized.backchannels.is_backchannel("tamam")
    with pytest.raises(ValueError):
        SessionOptions(min_interruption_duration=-1)
    with pytest.raises(ValueError):
        SessionOptions(false_interruption_timeout=-0.5)
    with pytest.raises(TypeError):
        SessionOptions(backchannel_words="uh-huh")  # type: ignore[arg-type]


# ----------------------------------------------------------------------- overlap verdicts


def test_overlap_cough_resumes_after_timeout() -> None:
    ov = Overlap.begin(InterruptionPolicy(false_interruption_timeout=2.0), 10.0)
    assert ov.verdict(10.3) is None and ov.deadline() == pytest.approx(10.5)
    ov.speech_stopped(10.15, 10.4)  # a 0.15 s cough; the VAD reported the end at 10.4
    assert ov.speech_duration(11.0) == pytest.approx(0.15)
    assert ov.verdict(10.6) is None and ov.deadline() == pytest.approx(12.4)
    assert ov.verdict(12.3) is None
    assert ov.verdict(12.4) == Verdict.RESUME and ov.reason() == "noise"


def test_overlap_long_speech_interrupts() -> None:
    ov = Overlap.begin(InterruptionPolicy(), 0.0)
    assert ov.verdict(0.45) is None
    assert ov.verdict(0.5) == Verdict.INTERRUPT
    # speech is counted over several segments of the same overlap
    ov = Overlap.begin(InterruptionPolicy(), 0.0)
    ov.speech_stopped(0.3, 0.55)
    ov.speech_started(1.0)
    assert ov.verdict(1.1) is None and ov.deadline() == pytest.approx(1.2)
    assert ov.verdict(1.2) == Verdict.INTERRUPT


def test_overlap_backchannel_resumes_once_the_whole_utterance_is_transcribed() -> None:
    ov = Overlap.begin(InterruptionPolicy(), 0.0)
    ov.add_transcript("item", "uh")  # interim while the user speaks: not conclusive
    ov.speech_stopped(0.35, 0.6)
    assert ov.verdict(0.6) is None
    ov.add_transcript("item", "Uh-huh.")  # transcript of the whole utterance
    assert ov.verdict(0.7) == Verdict.RESUME and ov.reason() == "backchannel"


def test_overlap_short_but_meaningful_speech_interrupts_at_timeout() -> None:
    ov = Overlap.begin(InterruptionPolicy(false_interruption_timeout=1.0), 0.0)
    ov.speech_stopped(0.3, 0.55)
    ov.add_transcript("item", "stop")
    assert ov.verdict(1.0) is None
    assert ov.verdict(1.55) == Verdict.INTERRUPT


def test_overlap_min_words() -> None:
    ov = Overlap.begin(InterruptionPolicy(min_words=2), 0.0)
    assert ov.verdict(1.0) is None and ov.deadline() is None  # long, but no words yet
    ov.add_transcript("item", "yeah okay")  # backchannels do not count
    assert ov.verdict(1.2) is None
    ov.add_transcript("item", "yeah okay wait please")
    assert ov.verdict(1.3) == Verdict.INTERRUPT
    few = Overlap.begin(InterruptionPolicy(min_words=2), 0.0)
    few.add_transcript("item", "wait")
    few.speech_stopped(0.8, 1.0)
    assert few.verdict(3.0) == Verdict.RESUME and few.reason() == "too_few_words"


def test_overlap_aftermath_of_a_confirmed_interruption() -> None:
    ov = Overlap.begin(InterruptionPolicy(false_interruption_timeout=1.0), 0.0)
    assert ov.verdict(0.6) == Verdict.INTERRUPT
    ov.confirmed = True
    ov.speech_stopped(0.8, 1.0)
    assert ov.verdict(1.5) is None
    assert ov.verdict(2.0) == Verdict.FALSE_INTERRUPTION
    ov.add_transcript("item", "hold on")
    assert ov.verdict(2.0) == Verdict.SETTLED


# ------------------------------------------------------------------ loopback transport


async def test_loopback_pause_and_resume() -> None:
    transport = LoopbackTransport(realtime_playout=True)
    assert transport.capabilities.pause
    await transport.start()
    for _ in range(10):
        await transport.write_audio(AudioFrame.silence(0.05, 24_000))
    await asyncio.sleep(0.12)
    await transport.pause_audio()
    assert transport.paused and transport.pause_times
    await asyncio.sleep(0.3)
    heard = sum(p.frame.duration for p in transport.played_log)
    assert 0.1 <= heard <= 0.3  # the frame playing when paused finishes, nothing more
    assert transport.buffered_duration() == pytest.approx(0.5 - heard, abs=0.06)
    await transport.resume_audio()
    await asyncio.wait_for(transport.wait_for_playout(), 3)
    assert sum(p.frame.duration for p in transport.played_log) == pytest.approx(0.5)
    assert longest_gap(transport) >= 0.25  # the pause
    await transport.aclose()


async def test_loopback_pause_without_realtime_playout_holds_writes() -> None:
    transport = LoopbackTransport()
    await transport.start()
    await transport.pause_audio()
    await transport.write_audio(AudioFrame.silence(0.1, 24_000))
    assert transport.played_log == [] and transport.buffered_duration() == pytest.approx(0.1)
    await transport.resume_audio()
    assert len(transport.played_log) == 1 and transport.buffered_duration() == 0.0
    await transport.aclose()


async def test_loopback_can_be_made_unpausable() -> None:
    transport = LoopbackTransport(pausable=False)
    assert not transport.capabilities.pause
    assert LoopbackTransport.capabilities.pause  # the class default is untouched
    with pytest.raises(NotImplementedError):
        await transport.pause_audio()


# -------------------------------------------------------------------- session helpers


class Recorder:
    NAMES = ("interrupted", "agent_false_interruption", "agent_state_changed",
             "user_state_changed", "user_transcript", "agent_transcript", "error")  # fmt: skip

    def __init__(self, session: AgentSession) -> None:
        self.events: list[tuple[str, Any]] = []
        for name in self.NAMES:
            session.on(name, self._make(name))

    def _make(self, name: str) -> Callable[[Any], None]:
        return lambda ev: self.events.append((name, ev))

    def of(self, name: str) -> list[Any]:
        return [ev for n, ev in self.events if n == name]

    def interrupted(self) -> list[Interrupted]:
        return self.of("interrupted")

    def false_interruptions(self) -> list[AgentFalseInterruption]:
        return self.of("agent_false_interruption")


async def wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), timeout)


def cascade(transcripts: list[str], *, answer: str = ANSWER, **options: Any) -> AgentSession:
    return AgentSession(
        stt=MockSTT(transcripts=transcripts),
        llm=MockLLM(responses=[answer]),
        tts=MockTTS(chars_per_second=30.0),
        vad=EnergyVAD(),  # 0.1 s to detect speech, 0.25 s of silence to end it
        cascade_options=CascadeOptions(min_endpointing_delay=0.3),
        options=SessionOptions(**options),
    )


def native(**options: Any) -> AgentSession:
    engine = MockEngine(responses=[ANSWER], transcripts=["wait stop"], realtime_factor=1.0)
    return AgentSession(engine, options=SessionOptions(**options))


async def agent_speaking(session: AgentSession, transport: LoopbackTransport) -> None:
    """Start the session and let the agent talk for a moment."""
    await session.start(Agent("x"), transport)
    await session.generate_reply(user_input="tell me something")
    await wait_for(lambda: session.agent_state == AgentState.SPEAKING)
    await asyncio.sleep(0.3)


async def burst(transport: LoopbackTransport, speech: float, silence: float = 0.5) -> float:
    """A short sound + enough silence for the VAD to end it, pushed at once (robust on slow
    runners: the verdict depends on the VAD, not on scheduling). Returns the start time."""
    t0 = now()
    await transport.play_user_audio(synth_speech(speech, SR), realtime=False)
    await transport.play_user_audio(AudioFrame.silence(silence, SR), realtime=False)
    return t0


def played(transport: LoopbackTransport) -> float:
    return sum(p.frame.duration for p in transport.played_log)


def longest_gap(transport: LoopbackTransport) -> float:
    """Longest silence between the ends and starts of consecutive played chunks."""
    log = transport.played_log
    return max(b.start_time - (a.start_time + a.frame.duration) for a, b in itertools.pairwise(log))


def assistant_messages(session: AgentSession) -> list[ChatMessage]:
    return [
        i for i in session.history.items if isinstance(i, ChatMessage) and i.role == "assistant"
    ]


# ------------------------------------------------------------------- session: resume


async def test_cough_pauses_then_resumes() -> None:
    session = cascade([""], false_interruption_timeout=0.8)  # the cough is not transcribed
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await agent_speaking(session, transport)
    t0 = await burst(transport, 0.15)
    await wait_for(lambda: bool(rec.false_interruptions()), 4)
    resumed_at = now()
    await asyncio.sleep(0.3)
    heard_after = played(transport)
    await session.aclose()

    assert transport.pause_times and transport.pause_times[0] - t0 < 0.3  # paused at once
    ev = rec.false_interruptions()[0]
    assert ev.resumed and ev.reason == "noise" and ev.transcript == ""
    assert ev.paused == pytest.approx(0.8, abs=0.3)  # the false-interruption timeout
    assert transport.resume_times and transport.resume_times[0] - t0 == pytest.approx(0.8, abs=0.35)
    assert resumed_at - t0 < 2.0
    assert not rec.interrupted()
    assert not transport.clear_times  # nothing was dropped
    assert heard_after > 0.5  # playback continued after the resume
    [answer] = assistant_messages(session)
    assert answer.text == ANSWER and not answer.interrupted
    states = [e.new_state for e in rec.of("agent_state_changed")]
    i = states.index(AgentState.SPEAKING)
    assert states[i : i + 3] == [AgentState.SPEAKING, AgentState.LISTENING, AgentState.SPEAKING]


async def test_backchannel_resumes_without_a_new_turn() -> None:
    llm = MockLLM(responses=[ANSWER])
    session = AgentSession(
        stt=MockSTT(transcripts=["uh-huh"]), llm=llm, tts=MockTTS(chars_per_second=30.0),
        vad=EnergyVAD(), cascade_options=CascadeOptions(min_endpointing_delay=0.3),
        options=SessionOptions(false_interruption_timeout=2.0),
    )  # fmt: skip
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await agent_speaking(session, transport)
    t0 = await burst(transport, 0.35)
    await wait_for(lambda: bool(rec.false_interruptions()), 3)
    resumed_at = now()
    await asyncio.sleep(0.6)  # longer than the cascade's endpointing delay
    await session.aclose()

    ev = rec.false_interruptions()[0]
    assert ev.resumed and ev.reason == "backchannel" and ev.transcript == "uh-huh"
    assert resumed_at - t0 < 1.0  # no need to wait for the 2 s timeout
    assert not rec.interrupted()
    assert len(llm.requests) == 1  # the backchannel was not answered...
    users = [i.text for i in session.history.items if getattr(i, "role", "") == "user"]
    assert users == ["tell me something"]  # ...nor committed as a user turn
    [answer] = assistant_messages(session)
    assert not answer.interrupted


async def test_long_backchannel_with_min_words_resumes() -> None:
    session = cascade(["yeah yeah okay"], min_interruption_words=1, false_interruption_timeout=2.0)
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await agent_speaking(session, transport)
    t0 = now()
    await transport.play_user_audio(synth_speech(1.0, SR))  # long enough for the duration rule
    await transport.play_user_audio(AudioFrame.silence(0.5, SR), realtime=False)
    await wait_for(lambda: bool(rec.false_interruptions()), 3)
    await session.aclose()
    ev = rec.false_interruptions()[0]
    assert ev.resumed and ev.reason == "backchannel" and ev.speech_duration > 0.8
    assert not rec.interrupted()
    assert now() - t0 < 2.8


async def test_resume_works_without_transport_pause() -> None:
    answer = "A short answer that still gets paused."
    session = cascade([""], answer=answer, false_interruption_timeout=0.6)
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True, pausable=False)
    await agent_speaking(session, transport)
    await burst(transport, 0.15)
    await wait_for(lambda: bool(rec.false_interruptions()), 4)
    await wait_for(lambda: session.agent_state == AgentState.LISTENING, 5)
    await asyncio.wait_for(transport.wait_for_playout(), 3)
    await session.aclose()

    assert not transport.pause_times and not transport.clear_times
    assert rec.false_interruptions()[0].resumed
    # the session stopped sending and continued where it stopped: nothing lost or repeated
    expected = MockTTS(chars_per_second=30.0).audio_duration_for(answer)
    assert played(transport) == pytest.approx(expected, abs=0.05)
    assert longest_gap(transport) > 0.4  # the pause


# ---------------------------------------------------------------- session: interrupt


@pytest.mark.parametrize("kind", ["native", "cascade"])
async def test_real_barge_in_is_confirmed_after_min_duration(kind: str) -> None:
    session = native() if kind == "native" else cascade(["wait stop please"])
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await agent_speaking(session, transport)
    heard_before = played(transport)
    t0 = now()
    user = asyncio.create_task(transport.play_user_audio(synth_speech(1.0, SR)))  # real time
    await wait_for(lambda: bool(rec.interrupted()), 3)
    await user
    await transport.play_user_audio(AudioFrame.silence(0.6, SR), realtime=False)
    await wait_for(lambda: len(assistant_messages(session)) == 2, 4)
    await session.aclose()

    pause = transport.pause_times[0] - t0
    assert 0.0 < pause < 0.35  # paused as soon as the VAD fired (~0.1 s)
    ev = rec.interrupted()[0]
    assert ev.timestamp - t0 == pytest.approx(0.5, abs=0.25)  # min_interruption_duration
    assert ev.played == pytest.approx(heard_before + pause, abs=0.25)  # heard up to the pause
    assert transport.clear_times
    first, second = assistant_messages(session)
    assert first.interrupted and not second.interrupted
    assert "wait stop" in second.text  # the barge-in became the next user turn
    assert not rec.false_interruptions()
    if kind == "native":
        conn: Any = session.connection
        assert conn.truncations[0][1] == pytest.approx(ev.played * 1000, abs=1)
        assert len(first.text) < len(ANSWER)


async def test_min_interruption_words_waits_for_words() -> None:
    session = cascade(["please stop talking now"], min_interruption_words=2)
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await agent_speaking(session, transport)
    t0 = now()
    user = asyncio.create_task(transport.play_user_audio(synth_speech(1.6, SR)))
    await wait_for(lambda: bool(rec.interrupted()), 4)
    await user
    await session.aclose()
    # the mock STT transcribes one word per 0.4 s: "please" at 0.5 s, "please stop" at 1.0 s
    assert rec.interrupted()[0].timestamp - t0 == pytest.approx(1.0, abs=0.3)


async def test_noise_that_interrupted_is_reported_as_false_interruption() -> None:
    session = cascade([""], false_interruption_timeout=0.6)  # noise is not transcribed
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await agent_speaking(session, transport)
    await transport.play_user_audio(synth_speech(0.8, SR))  # long enough to interrupt
    await transport.play_user_audio(AudioFrame.silence(0.5, SR), realtime=False)
    await wait_for(lambda: bool(rec.false_interruptions()), 3)
    await session.aclose()
    assert rec.interrupted()
    ev = rec.false_interruptions()[0]
    assert not ev.resumed and ev.reason == "noise" and ev.paused == 0.0


async def test_engine_side_cancellation_falls_back_to_interrupt() -> None:
    class CancelOnSpeech(MockEngineConnection):
        """Server-side VAD that cancels the response when speech starts (OpenAI-style)."""

        def _emit(self, event: EngineEvent) -> None:
            super()._emit(event)
            if isinstance(event, InputSpeechStarted):
                self.cancelling = asyncio.ensure_future(self.cancel_response())

    class ServerVADEngine(MockEngine):
        async def connect(self, options: EngineOptions) -> EngineConnection:
            conn = CancelOnSpeech(self, options)
            self.connections.append(conn)
            return conn

    engine = ServerVADEngine(responses=[ANSWER, "Sure."], transcripts=["hm"], realtime_factor=1.0)
    session = AgentSession(engine, options=SessionOptions(false_interruption_timeout=0.8))
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await agent_speaking(session, transport)
    t0 = await burst(transport, 0.2)  # too short for the policy: the engine decides
    await wait_for(lambda: len(assistant_messages(session)) == 2, 4)
    await wait_for(lambda: session.agent_state == AgentState.LISTENING, 4)
    await session.aclose()

    ev = rec.interrupted()[0]
    assert ev.timestamp - t0 < 0.4  # right away, not after min_interruption_duration
    assert not rec.false_interruptions()  # resuming was impossible
    conn: Any = session.connection
    assert conn.truncations[0][1] == pytest.approx(ev.played * 1000, abs=1)
    first, second = assistant_messages(session)
    assert first.interrupted and second.text == "Sure." and not second.interrupted


# ------------------------------------------------------------ session: other options


async def test_zero_min_duration_interrupts_immediately() -> None:
    session = native(min_interruption_duration=0.0)  # the behaviour before the policy
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await agent_speaking(session, transport)
    t0 = await burst(transport, 0.3, silence=0.0)
    await wait_for(lambda: bool(rec.interrupted()), 2)
    await session.aclose()
    assert rec.interrupted()[0].timestamp - t0 < 0.3
    assert not transport.pause_times and transport.clear_times


async def test_without_resume_the_agent_keeps_talking_until_confirmed() -> None:
    session = cascade([""], resume_false_interruption=False)
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await agent_speaking(session, transport)
    await burst(transport, 0.15)  # a cough: ignored, the agent never stops
    await transport.play_user_audio(AudioFrame.silence(0.4, SR))  # (the mic keeps streaming)
    assert not transport.pause_times and not rec.interrupted()
    assert session.agent_state == AgentState.SPEAKING
    t0 = now()
    await transport.play_user_audio(synth_speech(0.7, SR))  # a real barge-in
    await wait_for(lambda: bool(rec.interrupted()), 2)
    await session.aclose()
    assert rec.interrupted()[0].timestamp - t0 == pytest.approx(0.5, abs=0.25)
    assert not transport.pause_times and not rec.false_interruptions()


async def test_uninterruptible_say_discards_user_audio() -> None:
    notice = "This call may be recorded for quality purposes."  # ~1.6 s
    engine = MockEngine(responses=["Got it."], realtime_factor=1.0, chars_per_second=30.0)
    session = AgentSession(engine)
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await session.start(Agent("x"), transport)
    await session.say(notice, allow_interruptions=False)
    await wait_for(lambda: session.agent_state == AgentState.SPEAKING)
    await asyncio.sleep(0.2)
    await burst(transport, 0.8, silence=0.4)  # the user talks over it
    await asyncio.sleep(0.2)
    assert not transport.pause_times and not rec.interrupted()
    assert not [e for e in rec.of("user_state_changed") if e.new_state == UserState.SPEAKING]
    conn: Any = session.connection
    assert conn.received_audio >= 1.1  # the engine got (silent) audio, not the speech
    await wait_for(lambda: session.agent_state == AgentState.LISTENING, 4)
    [said] = assistant_messages(session)
    assert said.text == notice and not said.interrupted
    await burst(transport, 0.6, silence=0.6)  # once it is done, the user is heard again
    await wait_for(lambda: len(assistant_messages(session)) == 2, 4)
    await session.aclose()
