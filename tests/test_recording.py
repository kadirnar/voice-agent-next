"""Session recording: stereo WAV alignment, barge-in truncation and the JSONL timeline."""

from __future__ import annotations

import asyncio
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from voice_agent_next import Agent, AgentSession, AgentState, AudioFrame, SessionOptions
from voice_agent_next.audio.frame import AudioFormat
from voice_agent_next.audio.timeline import TimelineTrack
from voice_agent_next.audio.wav import read_wav
from voice_agent_next.providers.mock import MockEngine, MockToolCall, synth_speech
from voice_agent_next.session import SessionRecorder
from voice_agent_next.tools import function_tool
from voice_agent_next.transports import LoopbackTransport
from voice_agent_next.transports.loopback import PlayedAudio
from voice_agent_next.utils import now

from .test_session import Recorder, make_session, speak, wait_for

TOL = 0.1  # s: frame size, event-loop and timer jitter (Windows: 15.6 ms ticks)


def channels(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    audio = read_wav(path)
    assert audio.channels == 2
    both = audio.to_numpy().reshape(-1, 2)
    return both[:, 0], both[:, 1], audio.sample_rate


def active(x: np.ndarray, rate: int, threshold: int = 500) -> tuple[float, float]:
    """(first, last) time (s) where the channel is not silent."""
    idx = np.flatnonzero(np.abs(x.astype(np.int32)) > threshold)
    assert len(idx), "channel is silent"
    return idx[0] / rate, (idx[-1] + 1) / rate


def underruns(played: list[PlayedAudio], until: float = float("inf")) -> float:
    """Seconds the simulated speaker ran dry between agent frames (before ``until``): a
    stalled runner delivers a real-time reply late, and the recording keeps those gaps."""
    starts = [p for p in played if p.start_time < until]
    return sum(
        max(0.0, cur.start_time - (prev.start_time + prev.frame.duration))
        for prev, cur in itertools.pairwise(starts)
    )


def timeline(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def events(lines: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [ln for ln in lines if ln["event"] == name]


# ------------------------------------------------------------------- TimelineTrack


def test_timeline_track_write_pop_insert_truncate() -> None:
    tr = TimelineTrack(10)
    ones = np.ones(5, dtype=np.int16)
    tr.write(2, ones)
    assert tr.end == 7
    assert tr.read().tolist() == [0, 0, 1, 1, 1, 1, 1]
    tr.insert_silence(4, 3)  # a pause at sample 4
    assert tr.read().tolist() == [0, 0, 1, 1, 0, 0, 0, 1, 1, 1]
    tr.truncate(8)
    assert tr.end == 8
    assert tr.pop(3).tolist() == [0, 0, 1]
    assert tr.start == 3
    tr.write(1, np.full(4, 2, dtype=np.int16))  # the popped part is dropped
    assert tr.read().tolist() == [2, 2, 0, 0, 1]
    assert tr.pop(10).tolist() == [2, 2, 0, 0, 1, 0, 0, 0, 0, 0]
    assert (tr.start, tr.end) == (13, 13)
    for i in range(2000):  # stays small while streaming
        tr.write(tr.end, np.full(100, i % 7, dtype=np.int16))
        tr.pop(100)
    assert len(tr._buf) < 10_000


# ------------------------------------------------------- recorder hooks (no session)


def fake_session(output_rate: int = 1000, input_rate: int = 1000) -> Any:
    transport = SimpleNamespace(
        output_format=AudioFormat(output_rate, 1), input_format=AudioFormat(input_rate, 1)
    )
    engine = SimpleNamespace(provider="mock", model="m")
    return SimpleNamespace(transport=transport, engine=engine)


def tone(seconds: float, rate: int = 1000, value: int = 1000) -> AudioFrame:
    return AudioFrame(np.full(round(seconds * rate), value, np.int16).tobytes(), rate, 1)


def test_pause_shifts_and_clear_cuts_the_agent_channel(tmp_path: Path) -> None:
    rec = SessionRecorder(tmp_path / "call.wav")
    session = fake_session()
    t0 = now()
    rec.session_started(session, t0)
    rec.agent_audio(tone(0.5), t0 + 0.1)  # plays [0.1, 0.6)
    rec.playback_paused(t0 + 0.3)
    rec.playback_shifted(t0 + 0.3, 0.4)  # resumed 0.4 s later: [0.1, 0.3) + [0.7, 1.0)
    rec.playback_resumed()
    rec.agent_audio(tone(0.5, value=2000), t0 + 1.0)  # [1.0, 1.5) ...
    rec.playback_cleared(t0 + 1.2)  # ... cut by a barge-in at 1.2
    rec.session_closing(session, "done", t0 + 2.0)

    user, agent, rate = channels(tmp_path / "call.wav")
    assert rate == 1000 and len(agent) == 2000
    assert not user.any()
    expected = np.zeros(2000, dtype=np.int16)
    expected[100:300] = 1000
    expected[700:1000] = 1000
    expected[1000:1200] = 2000
    assert np.array_equal(agent, expected)
    lines = timeline(tmp_path / "call.jsonl")
    assert [ln["event"] for ln in lines] == [
        "recording_started",
        "playback_cleared",
        "session_closed",
    ]
    assert lines[1]["t"] == pytest.approx(1.2)
    assert lines[-1]["data"] == {"reason": "done", "duration": 2.0}


def test_user_stream_is_contiguous_with_gaps_as_silence(tmp_path: Path) -> None:
    rec = SessionRecorder(tmp_path / "u.wav")
    session = fake_session(output_rate=2000, input_rate=1000)  # user resampled 1k -> 2k
    t0 = now()
    rec.session_started(session, t0)
    for i in range(10):  # 0.2 s of audio in 20 ms frames arriving from 0.52 s on (+jitter)
        rec.user_audio(tone(0.02, value=3000), t0 + 0.52 + i * 0.02 + (0.005 if i % 2 else 0))
    for i in range(5):  # after a 1 s gap in the input
        rec.user_audio(tone(0.02, value=3000), t0 + 1.72 + i * 0.02)
    rec.session_closing(session, "done", t0 + 2.5)
    user, agent, rate = channels(tmp_path / "u.wav")
    assert rate == 2000 and len(user) == 5000
    assert not agent.any()
    loud = np.abs(user.astype(np.int32)) > 1500
    edges = np.flatnonzero(np.diff(loud.astype(np.int8))) + 1
    starts, stops = edges[::2] / rate, edges[1::2] / rate
    assert len(starts) == 2
    assert starts[0] == pytest.approx(0.5, abs=0.005) and stops[0] == pytest.approx(0.7, abs=0.005)
    assert starts[1] == pytest.approx(1.7, abs=0.005) and stops[1] == pytest.approx(1.8, abs=0.005)


# ------------------------------------------------------------------- live sessions


async def test_session_recording_aligns_user_and_agent_audio(tmp_path: Path) -> None:
    session = make_session(
        "native",
        transcripts=["hello"],
        responses=["Hi there, how can I help you today?"],
        realtime_factor=1.0,
        record=tmp_path,
    )
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await session.start(Agent("x"), transport)
    recorder = session.recorder
    assert recorder is not None and recorder.wav_path is not None
    await asyncio.sleep(0.3)
    spoke_at = now()
    await transport.play_user_audio(synth_speech(0.6, 16_000))  # real time
    await transport.play_user_audio(AudioFrame.silence(0.6, 16_000))
    await wait_for(lambda: session.agent_state == AgentState.SPEAKING)
    await asyncio.sleep(0.8)
    # streamed while the call goes on (the header is kept valid)
    assert read_wav(recorder.wav_path).duration > 0.4
    await wait_for(lambda: len(rec.turn_metrics()) == 1, timeout=10)
    await session.aclose()

    assert recorder.wav_path.parent == tmp_path and recorder.wav_path.suffix == ".wav"
    user, agent, rate = channels(recorder.wav_path)
    assert rate == transport.output_format.sample_rate
    origin = recorder.origin
    u0, u1 = active(user, rate)
    assert u0 == pytest.approx(spoke_at - origin, abs=TOL)
    assert u1 - u0 == pytest.approx(0.6, abs=TOL)
    a0, a1 = active(agent, rate)
    played = transport.played_log
    assert a0 == pytest.approx(played[0].start_time - origin, abs=TOL)
    heard = sum(p.frame.duration for p in played)
    assert a1 - a0 == pytest.approx(heard + underruns(played), abs=TOL)
    assert a0 > u1  # the reply follows the question on the same clock
    # the whole call is there, and no more
    # ends with the call; loaded runners (macOS CI) deliver the last frames ~0.25 s late
    assert len(user) / rate == pytest.approx(rec.of("close")[0].timestamp - origin, abs=0.35)


async def test_barge_in_truncates_the_recorded_agent_audio(tmp_path: Path) -> None:
    long_answer = "This is a very long answer that keeps going and going for quite a while. " * 3
    session = make_session(
        "native",
        transcripts=["tell me a story", "stop"],
        responses=[long_answer, "Okay."],
        realtime_factor=1.0,
        record=tmp_path / "barge.wav",
    )
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await session.start(Agent("x"), transport)
    await speak(transport, 0.6, 0.5)
    await wait_for(lambda: session.agent_state == AgentState.SPEAKING)
    await asyncio.sleep(1.0)
    await transport.play_user_audio(synth_speech(0.8, 16_000))  # barge in (real time)
    await wait_for(lambda: bool(rec.of("interrupted")))
    await asyncio.sleep(0.3)
    await session.aclose()

    recorder = session.recorder
    assert recorder is not None
    _, agent, rate = channels(tmp_path / "barge.wav")
    interrupted = rec.of("interrupted")[0]
    # the agent went quiet where playback was paused (the barge-in), not at the end
    stopped = transport.pause_times[0] if transport.pause_times else transport.clear_times[0]
    first = next(p for p in transport.played_log)
    a0 = first.start_time - recorder.origin
    agent_until_barge = agent[: round((stopped - recorder.origin + TOL) * rate)]
    b0, b1 = active(agent_until_barge, rate)
    assert b0 == pytest.approx(a0, abs=TOL)
    assert b1 == pytest.approx(stopped - recorder.origin, abs=TOL)
    gaps = underruns(transport.played_log, until=stopped)
    assert b1 - b0 == pytest.approx(interrupted.played + gaps, abs=TOL)
    # nothing of the long answer after the cut (only possibly the short "Okay." reply)
    after = agent[round((stopped - recorder.origin + TOL) * rate) :]
    okay = [p for p in transport.played_log if p.start_time > stopped + TOL]
    assert (
        np.count_nonzero(np.abs(after.astype(np.int32)) > 500) / rate
        <= sum(p.frame.duration for p in okay) + TOL
    )

    lines = timeline(tmp_path / "barge.jsonl")
    (cut,) = events(lines, "playback_cleared")
    assert cut["t"] == pytest.approx(stopped - recorder.origin, abs=TOL)
    (intr,) = events(lines, "interrupted")
    assert intr["data"]["played"] == pytest.approx(interrupted.played)
    assert intr["t"] >= cut["t"]


@pytest.mark.parametrize("kind", ["native", "cascade"])
async def test_timeline_has_session_and_engine_events(kind: str, tmp_path: Path) -> None:
    @function_tool
    async def get_weather(city: str) -> str:
        """Weather for a city."""
        return f"Sunny in {city}"

    session = make_session(
        kind,
        transcripts=["weather in paris?"],
        responses=[MockToolCall("get_weather", {"city": "Paris"}), "It is sunny in Paris."],
        options=SessionOptions(record=str(tmp_path)),
    )
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x", tools=[get_weather], greeting="Hello!"), transport)
    await speak(transport)
    await wait_for(lambda: any("sunny" in e.delta for e in rec.of("agent_transcript")))
    await wait_for(lambda: len(rec.turn_metrics()) == 1)
    transport.end_user_audio()
    await asyncio.wait_for(session.wait_closed(), 5)

    recorder = session.recorder
    assert recorder is not None and recorder.timeline_path is not None
    assert recorder.timeline_path.parent == tmp_path
    lines = timeline(recorder.timeline_path)
    first, last = lines[0], lines[-1]
    assert first["event"] == "recording_started" and first["t"] == 0
    assert first["data"]["channels"] == {"left": "user", "right": "agent"}
    assert first["data"]["wav"] == recorder.wav_path.name  # type: ignore[union-attr]
    assert last["event"] == "session_closed"
    assert last["data"]["reason"] == "user_disconnected"
    assert all(0 <= ln["t"] <= last["t"] + TOL for ln in lines)

    names = {(ln["source"], ln["event"]) for ln in lines}
    for expected in [
        ("engine", "input_speech_started"),
        ("engine", "input_committed"),
        ("engine", "response_started"),
        ("engine", "response_tool_call"),
        ("engine", "response_done"),
        ("session", "agent_state_changed"),
        ("session", "user_state_changed"),
        ("session", "conversation_item"),
        ("session", "tool_call"),
        ("session", "tool_result"),
        ("session", "metrics"),
    ]:
        assert expected in names, expected
    assert ("engine", "response_audio") not in names  # off by default
    finals = [
        ln["data"]["text"] for ln in events(lines, "user_transcript") if ln["data"]["is_final"]
    ]
    assert finals == ["weather in paris?"]
    said = "".join(ln["data"]["delta"] for ln in events(lines, "agent_transcript"))
    assert "Hello!" in said and "sunny in Paris" in said
    (result,) = events(lines, "tool_result")
    assert result["data"]["output"]["output"] == "Sunny in Paris"
    assert result["data"]["call"]["name"] == "get_weather"
    metric_types = {ln["data"]["type"] for ln in events(lines, "metrics")}
    assert "turn" in metric_types
    assert metric_types & {"engine", "llm"}
    states = [ln["data"]["new_state"] for ln in events(lines, "agent_state_changed")]
    assert states[-1] == "closed"


async def test_audio_events_option_logs_agent_chunks(tmp_path: Path) -> None:
    session = AgentSession(
        MockEngine(responses=["Hi."]),
        record=SessionRecorder(tmp_path / "a.wav", audio_events=True),
    )
    rec = Recorder(session)
    await session.start(Agent("x", greeting="Hello there."), LoopbackTransport())
    await wait_for(lambda: AgentState.SPEAKING in rec.states(), timeout=5)
    await wait_for(lambda: session.agent_state == AgentState.LISTENING, timeout=5)
    await session.aclose()
    chunks = events(timeline(tmp_path / "a.jsonl"), "response_audio")
    assert chunks and "frame" in chunks[0]["data"]
    assert chunks[0]["data"]["frame"]["audio_duration"] > 0


def test_sessions_without_recording_have_no_taps() -> None:
    session = AgentSession("mock")
    assert session.recorder is None
    assert session._taps == []
