"""Dynamic endpointing and dictation mode of the cascade (``CascadeOptions.endpointing``).

Policy tests drive :class:`Endpointer` directly (pure, deterministic); session tests use
the mock STT, a scripted turn detector and the energy VAD (0.25 s of silence ends speech).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.test_session import Recorder, speak, wait_for
from voice_agent_next import Agent, AgentSession, AudioFrame, CascadeOptions
from voice_agent_next.chat import ChatContext
from voice_agent_next.engines.endpointing import Endpointer, PauseTracker
from voice_agent_next.errors import ConfigurationError
from voice_agent_next.metrics import EndpointingMetrics, SpeculationMetrics, TurnMetrics
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import (
    MockEngine,
    MockLLM,
    MockSTT,
    MockTTS,
    MockTurnDetector,
    synth_speech,
)
from voice_agent_next.transports import LoopbackTransport

SR = 16_000


# ------------------------------------------------------------------------ policy


def endpointer(detector: bool = True, **options: Any) -> Endpointer:
    return Endpointer(CascadeOptions(**options), has_detector=detector)


def test_pause_tracker_follows_mean_and_deviation() -> None:
    t = PauseTracker(alpha=0.25, beta=0.25)
    assert t.bound(2.0) is None
    t.add(0.4)
    assert (t.mean, t.deviation) == pytest.approx((0.4, 0.2))
    for _ in range(40):  # a steady pause length: the deviation decays towards zero
        t.add(0.4)
    assert t.mean == pytest.approx(0.4)
    assert t.bound(2.0) == pytest.approx(0.4, abs=0.01)
    t.add(0.0)  # not a pause
    assert t.count == 41
    with pytest.raises(ValueError):
        PauseTracker(alpha=0.0)


def test_fixed_policy_is_the_legacy_min_or_max() -> None:
    e = endpointer()
    assert e.decide(0.9, 0.5).delay == 0.4
    assert e.decide(0.49, 0.5).delay == 2.5
    assert endpointer(detector=False).decide(None, None).delay == 0.6
    assert endpointer(min_endpointing_delay=0.3).decide(0.5, 0.5).delay == 0.3
    e.observe_pause(1.5)  # learning never changes the fixed policy
    assert e.decide(0.9, 0.5).delay == 0.4


def test_dynamic_policy_follows_the_detector_confidence() -> None:
    e = endpointer(endpointing="dynamic")
    certain_done, at_threshold, unsure, certain_not = (
        e.decide(p, 0.5).delay for p in (1.0, 0.5, 0.25, 0.0)
    )
    assert certain_done == pytest.approx(0.25)  # the floor: commit quickly
    assert at_threshold == pytest.approx(0.4)  # no pauses learned yet: the fixed delay
    assert unsure == pytest.approx(0.4 + (2.5 - 0.4) / 2)
    assert certain_not == pytest.approx(2.5)  # the ceiling: likely mid-thought
    assert e.decide(0.75, 0.5).delay == pytest.approx((0.25 + 0.4) / 2)
    d = e.decide(0.6, 0.5)
    assert (d.policy, d.probability, d.threshold, d.hold) == ("dynamic", 0.6, 0.5, 0.4)


def test_dynamic_policy_learns_the_users_pauses_within_bounds() -> None:
    e = endpointer(detector=False, endpointing="dynamic")
    assert e.decide(None, None).delay == pytest.approx(0.6)  # nothing learned: fixed delay
    for _ in range(30):  # a user who pauses ~0.9 s mid-turn
        e.observe_pause(0.9)
    assert e.decide(None, None).delay == pytest.approx(0.9, abs=0.02)
    fast = endpointer(detector=False, endpointing="dynamic")
    for _ in range(30):  # ... and one whose pauses are short
        fast.observe_pause(0.3)
    assert fast.decide(None, None).delay == pytest.approx(0.3, abs=0.02)
    # bounds: never below min_endpointing_delay, never above max_endpointing_delay
    lo = endpointer(detector=False, endpointing="dynamic", min_endpointing_delay=0.45)
    hi = endpointer(detector=False, endpointing="dynamic", max_endpointing_delay=1.0)
    for _ in range(30):
        lo.observe_pause(0.3)
        hi.observe_pause(2.0)
    assert lo.decide(None, None).delay == pytest.approx(0.45)
    assert hi.decide(None, None).delay == pytest.approx(1.0)
    # with a detector, learned pauses move the undecided middle; confidence still wins
    d = endpointer(endpointing="dynamic")
    for _ in range(30):
        d.observe_pause(0.8)
    assert d.decide(0.5, 0.5).delay == pytest.approx(0.8, abs=0.02)
    assert d.decide(1.0, 0.5).delay == pytest.approx(0.25)
    assert d.decide(0.0, 0.5).delay == pytest.approx(2.5)


def test_dictation_waits_for_the_detector_and_never_commits_early() -> None:
    e = endpointer(endpointing="dynamic", dictation=True)
    assert e.policy == "dictation"
    assert e.decide(1.0, 0.5).delay == 1.0  # dictation_min_delay: no fast commits
    assert e.decide(0.4, 0.5).delay == 5.0  # dictation_max_delay
    assert endpointer(detector=False, dictation=True).decide(None, None).delay == 5.0
    strict = endpointer(dictation=True, dictation_threshold=0.8)
    assert strict.decide(0.7, 0.5).delay == 5.0
    assert strict.decide(0.8, 0.5).threshold == 0.8
    e.observe_pause(3.0)  # digit groups are not typical pauses: nothing is learned
    assert e.pauses.count == 0
    e.dictation = False
    assert e.decide(1.0, 0.5).delay == pytest.approx(0.25)


def test_dynamic_policy_stops_trusting_a_detector_that_cut_the_user_off() -> None:
    """Issue #113: Smart Turn is ~0.97 sure that "Where is my order?" ends the turn, so the
    confident branch kept committing at the floor although the user had been cut off."""
    e = endpointer(endpointing="dynamic")
    assert e.decide(0.97, 0.5).delay < 0.3
    e.observe_cutoff(0.8)  # the user went on 0.8 s after a confident commit
    guarded = e.decide(0.97, 0.5).delay
    assert guarded == pytest.approx(e.hold()) and guarded > 0.8  # the learned hold
    assert e.decide(1.0, 0.5).delay == pytest.approx(guarded)
    for _ in range(3):  # commits the user leaves alone: trust comes back
        e.observe_commit()
    assert 0.25 < e.decide(0.97, 0.5).delay < guarded
    for _ in range(30):
        e.observe_commit()
    assert e.guard is None and e.decide(0.97, 0.5).delay < 0.35  # (the learned hold moved)
    fixed = endpointer()
    fixed.observe_cutoff(0.8)  # the fixed policy stays fixed
    assert fixed.guard is None and fixed.decide(0.97, 0.5).delay == 0.4
    dictation = endpointer(endpointing="dynamic", dictation=True)
    dictation.observe_cutoff(3.0)
    assert dictation.guard is None and dictation.pauses.count == 0


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        endpointer(endpointing="eager")


# ------------------------------------------------------------------ in a session


class ScriptedDetector(MockTurnDetector):
    """Returns the scripted probabilities in order (then the last one)."""

    def __init__(self, *probabilities: float) -> None:
        super().__init__(threshold=0.5)
        self.script = list(probabilities)

    async def _predict(self, *, audio: AudioFrame | None, chat_ctx: ChatContext | None) -> float:
        self.calls += 1
        return self.script.pop(0) if len(self.script) > 1 else self.script[0]


def cascade(turn_detector: Any = None, **options: Any) -> AgentSession:
    return AgentSession(
        stt=MockSTT(latency=0.01),
        llm=MockLLM(responses=lambda ctx: "Okay."),
        tts=MockTTS(chars_per_second=200.0),
        vad=EnergyVAD(),
        turn_detector=turn_detector,
        cascade_options=CascadeOptions(**options),
    )


def endpointing(rec: Recorder) -> list[EndpointingMetrics]:
    return [m for m in rec.of("metrics") if isinstance(m, EndpointingMetrics)]


async def pause_then_continue(transport: LoopbackTransport, pause: float) -> None:
    """Speech, a real-time pause of ``pause`` seconds, then more speech and silence."""
    await transport.play_user_audio(synth_speech(0.4, SR), realtime=False)
    await transport.play_user_audio(AudioFrame.silence(pause, SR))  # real time
    await speak(transport, 0.4, 0.3)


async def eot_delays(detector: Any, **options: Any) -> tuple[list[float], Recorder]:
    session = cascade(detector, **options)
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport, 0.5, 0.3)
    await wait_for(lambda: bool(rec.turn_metrics()), 6)
    await session.aclose()
    return [m.end_of_turn_delay or 0.0 for m in rec.turn_metrics()], rec


async def test_confident_detector_commits_at_the_floor() -> None:
    [fixed], _ = await eot_delays(ScriptedDetector(0.99))
    [dynamic], rec = await eot_delays(ScriptedDetector(0.99), endpointing="dynamic")
    assert fixed == pytest.approx(0.4, abs=0.1)
    assert dynamic == pytest.approx(0.25, abs=0.1)  # = the VAD's own 0.25 s pause
    assert dynamic < fixed - 0.05
    [m] = endpointing(rec)
    assert m.policy == "dynamic" and m.committed and not m.false_commit
    assert m.delay == pytest.approx(0.25 + 0.15 * 0.02)
    assert m.probability == 0.99 and m.threshold == 0.5


async def test_unconfident_detector_waits_up_to_the_ceiling() -> None:
    [delay], rec = await eot_delays(
        ScriptedDetector(0.0), endpointing="dynamic", max_endpointing_delay=1.0
    )
    assert delay == pytest.approx(1.0, abs=0.15)
    assert endpointing(rec)[0].delay == pytest.approx(1.0)


async def test_resumed_pauses_are_learned_and_reported() -> None:
    session = cascade(endpointing="dynamic")  # VAD only: 0.6 s until pauses are learned
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    conn: Any = session.connection
    for _ in range(2):
        await pause_then_continue(transport, 0.45)
    await wait_for(lambda: len(rec.turn_metrics()) == 1, 5)  # one turn despite the pause...
    await asyncio.sleep(0.3)
    await session.aclose()
    holds = [m for m in endpointing(rec) if not m.committed]
    assert len(holds) >= 1  # ... reported as a resumed pause (the second may be cut short)
    assert holds[0].pause == pytest.approx(0.45, abs=0.12)
    assert holds[0].delay == pytest.approx(0.6)
    assert conn.endpointer.pauses.count >= 1
    assert conn.endpointer.pauses.mean == pytest.approx(0.45, abs=0.12)


async def test_false_commit_is_detected_and_raises_the_delay() -> None:
    session = cascade(endpointing="dynamic", false_commit_window=1.0)
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    conn: Any = session.connection
    # the pause (0.9 s) is longer than the delay (0.6 s): committed, then the user resumes
    await pause_then_continue(transport, 0.9)
    await wait_for(lambda: len(rec.turn_metrics()) == 2, 6)
    await asyncio.sleep(1.1)  # the second commit's window closes
    await session.aclose()
    first, second = endpointing(rec)[:2]
    assert first.committed and first.false_commit
    assert first.pause == pytest.approx(0.9, abs=0.15)
    assert second.committed and not second.false_commit and second.pause is None
    # the cut-off pause was learned: the next undecided pause waits longer than 0.9 s
    assert second.hold is not None and second.hold > 0.9
    assert conn.endpointer.decide(None, None).delay > 0.9


async def test_dictation_mode_at_runtime() -> None:
    detector = ScriptedDetector(0.3)  # "probably not done" at every pause
    session = cascade(detector, dictation_max_delay=1.2)
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    session.update_endpointing(dictation=True)
    # a 0.9 s pause while reading out a number: fixed endpointing would wait 2.5 s too,
    # but a confident detector would cut it at 0.4 s; dictation never commits before 1 s
    detector.script = [0.9, 0.3]
    await pause_then_continue(transport, 0.8)
    await wait_for(lambda: bool(rec.turn_metrics()), 6)
    [turn] = rec.turn_metrics()
    assert turn.end_of_turn_delay == pytest.approx(1.2, abs=0.2)  # dictation_max_delay
    await wait_for(lambda: len(endpointing(rec)) == 2, 3)  # after false_commit_window
    hold, commit = endpointing(rec)
    assert hold.policy == "dictation" and not hold.committed and hold.delay == 1.0
    assert commit.policy == "dictation" and commit.delay == 1.2
    conn: Any = session.connection
    assert conn.endpointer.pauses.count == 0  # dictation pauses are not learned
    session.update_endpointing(dictation=False, mode="dynamic")
    assert conn.endpointer.policy == "dynamic"
    with pytest.raises(ValueError):
        session.update_endpointing(mode="eager")  # type: ignore[arg-type]
    await session.aclose()


async def test_dictation_option_per_session() -> None:
    [delay], rec = await eot_delays(ScriptedDetector(0.99), dictation=True, dictation_min_delay=0.7)
    assert delay == pytest.approx(0.7, abs=0.15)
    assert endpointing(rec)[0].policy == "dictation"


async def test_native_engines_reject_endpointing_updates() -> None:
    session = AgentSession(MockEngine())
    await session.start(Agent("x"), LoopbackTransport())
    with pytest.raises(ConfigurationError):
        session.update_endpointing(dictation=True)
    await session.aclose()


async def test_preemptive_generation_with_dynamic_endpointing() -> None:
    llm = MockLLM(responses=lambda ctx: "Sure.", ttft=0.3)
    session = AgentSession(
        stt=MockSTT(transcripts=["book a table"], latency=0.02),
        llm=llm,
        tts=MockTTS(chars_per_second=200.0),
        vad=EnergyVAD(),
        turn_detector=ScriptedDetector(0.6),  # likely done, not certain: ~0.37 s
        cascade_options=CascadeOptions(endpointing="dynamic", preemptive_generation=True),
    )
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport, 0.5, 0.4)
    await wait_for(lambda: bool(rec.turn_metrics()), 5)
    await session.aclose()
    [spec] = [m for m in rec.of("metrics") if isinstance(m, SpeculationMetrics)]
    assert spec.hit
    assert len(llm.requests) == 1
    turn: TurnMetrics = rec.turn_metrics()[0]
    assert turn.end_of_turn_delay == pytest.approx(0.37, abs=0.1)


async def test_confident_false_commit_raises_the_confident_delay() -> None:
    session = cascade(ScriptedDetector(0.99), endpointing="dynamic")
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    conn: Any = session.connection
    await pause_then_continue(transport, 0.8)  # cut off at the floor, then goes on
    await wait_for(lambda: len(rec.turn_metrics()) == 2, 6)
    await session.aclose()
    first = endpointing(rec)[0]
    assert first.committed and first.false_commit and first.probability == 0.99
    assert conn.endpointer.guard is not None and conn.endpointer.guard > 0.8
    assert conn.endpointer.decide(0.99, 0.5).delay > 0.8


async def test_no_commit_while_resumed_speech_is_unconfirmed() -> None:
    """The VAD confirms speech only after ``min_speech_duration`` (0.3 s here): a user who
    resumes just before the commit must not be answered in that gap."""
    session = AgentSession(
        stt=MockSTT(latency=0.01),
        llm=MockLLM(responses=lambda ctx: "Okay."),
        tts=MockTTS(chars_per_second=200.0),
        vad=EnergyVAD(min_speech_duration=0.3),
        cascade_options=CascadeOptions(min_endpointing_delay=0.5),  # VAD only
    )
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await transport.play_user_audio(synth_speech(0.4, SR), realtime=False)
    await transport.play_user_audio(AudioFrame.silence(0.35, SR))  # resumes 0.15 s early...
    await transport.play_user_audio(synth_speech(0.5, SR))  # ...confirmed ~0.15 s too late
    await transport.play_user_audio(AudioFrame.silence(0.8, SR), realtime=False)
    await wait_for(lambda: bool(rec.turn_metrics()), 6)
    await asyncio.sleep(0.5)
    await session.aclose()
    assert len(rec.turn_metrics()) == 1  # one turn: the pause was not committed
    [hold] = [m for m in endpointing(rec) if not m.committed]
    assert hold.pause == pytest.approx(0.35, abs=0.1)
