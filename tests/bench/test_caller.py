"""CallerEmulator: real-time pacing, turn-taking and the one-clock recording."""

from __future__ import annotations

import asyncio
import statistics

import pytest

from tests.bench.helpers import concat, silence, tone
from voice_agent_next.audio import AudioFrame
from voice_agent_next.bench.caller import CallerEmulator
from voice_agent_next.bench.onset import OnsetDetector
from voice_agent_next.bench.stimuli import Stimulus
from voice_agent_next.transports import LoopbackTransport
from voice_agent_next.utils import now


def stimulus(speech: float = 0.4, *, lead: float = 0.0, **kw: object) -> Stimulus:
    audio = concat(silence(lead, 16_000), tone(speech, 16_000)) if lead else tone(speech, 16_000)
    return Stimulus(
        id=str(kw.pop("id", "s")),
        text="hi",
        audio=audio,
        speech_start=lead,
        speech_end=lead + speech,
        source="synthetic",
        **kw,  # type: ignore[arg-type]
    )


class FakeAgent:
    """Agent side of the loopback: optional greeting, then replies after user speech."""

    def __init__(
        self, transport: LoopbackTransport, *, delay: float, reply: float, greeting: float = 0.0
    ) -> None:
        self.transport = transport
        self.delay = delay
        self.reply = reply
        self.greeting = greeting
        self.arrivals: list[tuple[float, float, float]] = []
        self._tasks: list[asyncio.Task[None]] = []

    async def _play(self, duration: float, delay: float = 0.0) -> None:
        await asyncio.sleep(delay)
        audio = tone(duration, 24_000, freq=440.0)
        for start in range(0, audio.samples_per_channel, 960):  # 40 ms frames
            await self.transport.write_audio(audio.slice(start / 24_000, (start + 960) / 24_000))

    async def run(self) -> None:
        if self.greeting:
            self._tasks.append(asyncio.create_task(self._play(self.greeting)))
        speaking, quiet = False, 0.0
        async for frame in self.transport.audio_input():
            assert frame.timestamp is not None
            self.arrivals.append((now(), frame.timestamp, frame.duration))
            if frame.rms() > 0.01:
                speaking, quiet = True, 0.0
            elif speaking:
                quiet += frame.duration
                if quiet >= 0.1 - 1e-9:  # 100 ms of silence ends the user's turn
                    speaking = False
                    self._tasks.append(asyncio.create_task(self._play(self.reply, self.delay)))
        for task in self._tasks:
            task.cancel()


async def test_caller_paces_chunks_waits_for_replies_and_records_one_clock() -> None:
    transport = LoopbackTransport(realtime_playout=True)
    await transport.start()
    agent = FakeAgent(transport, delay=0.2, reply=0.2, greeting=0.2)
    agent_task = asyncio.create_task(agent.run())
    caller = CallerEmulator(transport, tail=0.1)
    result = await caller.run(
        [stimulus(id="a"), stimulus(id="b", lead=0.1)],
        lead_in=0.2,
        reply_timeout=2.0,
        gap_after_reply=0.1,
    )
    await transport.aclose()
    await asyncio.wait_for(agent_task, 2)

    # every chunk is time-stamped on one exact 20 ms grid and delivered once it has elapsed
    stamps = [ts for _, ts, _ in agent.arrivals]
    assert len(stamps) == result.chunks_sent
    assert stamps == [pytest.approx(result.stream_start + 0.02 * i, abs=1e-9)
                      for i in range(len(stamps))]  # fmt: skip
    lateness = [arrived - (ts + dur) for arrived, ts, dur in agent.arrivals]
    assert min(lateness) > -0.002  # never before the chunk's interval ended
    assert statistics.median(lateness) < 0.015
    assert result.max_push_lag == pytest.approx(max(result.push_lag))

    first, second = result.turns
    greeting_end = transport.played_log[0].start_time + 0.2
    assert first.start >= greeting_end + 0.1 - 0.021  # waited for the greeting to finish
    for turn in result.turns:
        assert not turn.missed and turn.reply_start is not None and turn.reply_end is not None
        # fake agent: 100 ms silence detection + 200 ms delay
        assert turn.reply_start - turn.speech_end == pytest.approx(0.3, abs=0.06)
        assert turn.reply_end - turn.reply_start == pytest.approx(0.2, abs=0.03)
    assert second.start >= first.reply_end + 0.1 - 0.021
    assert second.speech_start == pytest.approx(second.start + 0.1)

    rec = result.recording
    det = OnsetDetector(min_speech=0.05)
    user = det.segments(rec.user_audio())
    agent_speech = det.segments(rec.agent_audio())
    assert [s for s, _ in user] == [
        pytest.approx(rec.to_offset(t.speech_start), abs=0.011) for t in result.turns
    ]
    assert [e for _, e in user] == [
        pytest.approx(rec.to_offset(t.speech_end), abs=0.011) for t in result.turns
    ]
    assert len(agent_speech) == 3  # greeting + two replies
    for turn, (start, _) in zip(result.turns, agent_speech[1:], strict=True):
        assert start == pytest.approx(rec.to_offset(turn.reply_start or 0.0), abs=0.011)
    assert rec.duration >= rec.to_offset(result.stream_end) - 1e-6


async def test_missed_turn_after_reply_timeout() -> None:
    transport = LoopbackTransport(realtime_playout=True)
    await transport.start()
    seen: list[int] = []
    caller = CallerEmulator(transport, tail=0.0, on_turn=lambda t: seen.append(t.index))
    t0 = now()
    result = await caller.run([stimulus(0.2)], lead_in=0.0, reply_timeout=0.3, gap_after_reply=0.1)
    elapsed = now() - t0
    await transport.aclose()
    (turn,) = result.turns
    assert turn.missed and turn.reply_start is None and not turn.timed_out
    assert elapsed == pytest.approx(0.2 + 0.3, abs=0.1)
    assert seen == [0]
    assert result.agent_frames == 0 and not result.aborted


async def test_turns_without_expected_reply_only_wait_for_the_pause() -> None:
    transport = LoopbackTransport(realtime_playout=True)
    await transport.start()
    caller = CallerEmulator(transport, tail=0.0)
    result = await caller.run(
        [stimulus(0.2, expect_reply=False, pause=0.2), stimulus(0.2)],
        lead_in=0.0,
        reply_timeout=0.2,
    )
    await transport.aclose()
    first, second = result.turns
    assert not first.missed and second.missed
    assert second.start == pytest.approx(first.end + 0.2, abs=0.021)


async def test_should_stop_aborts_the_call() -> None:
    transport = LoopbackTransport(realtime_playout=True)
    await transport.start()
    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 15  # after 15 chunks (300 ms)

    caller = CallerEmulator(transport, should_stop=stop)
    result = await caller.run([stimulus(), stimulus()], lead_in=0.1, reply_timeout=1.0)
    await transport.aclose()
    assert result.aborted and len(result.turns) == 1 and result.chunks_sent == 15


def test_caller_requires_realtime_playout() -> None:
    with pytest.raises(ValueError, match="realtime_playout"):
        CallerEmulator(LoopbackTransport())
    with pytest.raises(ValueError):
        CallerEmulator(LoopbackTransport(realtime_playout=True), chunk=0.0)


async def test_stimuli_at_another_rate_are_resampled() -> None:
    transport = LoopbackTransport(realtime_playout=True)
    await transport.start()
    stim = Stimulus("x", None, tone(0.2, 48_000), 0.0, 0.2, "wav")
    result = await CallerEmulator(transport, tail=0.0).run([stim], lead_in=0.0, reply_timeout=0.1)
    await transport.aclose()
    assert result.chunks_sent == 10 + 5  # 200 ms of speech + 100 ms waiting for a reply
    user = result.recording.user_audio()
    assert user.sample_rate == 16_000
    assert isinstance(user, AudioFrame)
