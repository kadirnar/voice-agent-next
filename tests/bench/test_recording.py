"""Stereo call recordings on one clock and label files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.bench.helpers import tone
from voice_agent_next.audio import AudioFrame, read_wav
from voice_agent_next.bench.onset import OnsetDetector
from voice_agent_next.bench.recording import DuplexRecording, Label, read_labels, write_labels


def spans(audio: AudioFrame) -> list[tuple[float, float]]:
    return OnsetDetector(min_speech=0.05).segments(audio)


def test_both_channels_share_one_clock(tmp_path: Path) -> None:
    rec = DuplexRecording(origin=100.0, user_rate=16_000, agent_rate=24_000)
    rec.add_user(tone(0.5, 16_000), 100.5)
    rec.add_agent(tone(0.3, 24_000), 101.2)
    rec.add_agent(tone(0.5, 24_000), 102.0, end_time=102.1)  # playback cleared (barge-in)
    assert rec.duration == pytest.approx(2.1)
    assert rec.to_offset(101.2) == pytest.approx(1.2)

    user, agent = rec.user_audio(), rec.agent_audio()
    assert (user.sample_rate, agent.sample_rate) == (16_000, 24_000)
    assert user.duration == pytest.approx(2.1) and agent.duration == pytest.approx(2.1)
    assert spans(user) == [(pytest.approx(0.5), pytest.approx(1.0))]
    assert spans(agent) == [
        (pytest.approx(1.2), pytest.approx(1.5)),
        (pytest.approx(2.0), pytest.approx(2.1)),
    ]

    wav = read_wav(rec.write_wav(tmp_path / "rec" / "stereo.wav"))
    assert (wav.channels, wav.sample_rate) == (2, 24_000)
    both = wav.to_numpy()
    left = AudioFrame.from_numpy(np.ascontiguousarray(both[:, 0]), 24_000)
    right = AudioFrame.from_numpy(np.ascontiguousarray(both[:, 1]), 24_000)
    # the user channel is resampled with delay compensation: still aligned to the clock
    assert spans(left) == [(pytest.approx(0.5, abs=0.011), pytest.approx(1.0, abs=0.011))]
    assert spans(right) == spans(agent)


def test_audio_before_the_origin_is_dropped() -> None:
    rec = DuplexRecording(origin=10.0, user_rate=16_000, agent_rate=16_000)
    rec.add_user(tone(1.0, 16_000), 9.5)  # the first half precedes t = 0
    assert rec.duration == pytest.approx(0.5)
    with pytest.raises(ValueError):
        rec.add_agent(tone(0.1, 24_000), 10.0)  # wrong rate


def test_empty_recording_still_writes_a_wav(tmp_path: Path) -> None:
    rec = DuplexRecording(origin=0.0, user_rate=16_000, agent_rate=24_000)
    assert rec.duration == 0.0
    assert read_wav(rec.write_wav(tmp_path / "empty.wav")).channels == 2


def test_labels_round_trip(tmp_path: Path) -> None:
    labels = [
        Label(1.2, 1.2, "agent onset: v2v 612 ms"),
        Label(0.5, 1.0, "user t01: What   time\tis it?"),
    ]
    path = write_labels(tmp_path / "labels.txt", labels)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "0.500000\t1.000000\tuser t01: What time is it?"  # sorted, one line each
    assert read_labels(path) == [
        Label(0.5, 1.0, "user t01: What time is it?"),
        Label(1.2, 1.2, "agent onset: v2v 612 ms"),
    ]
    (tmp_path / "audacity.txt").write_text(
        "0.1\t0.2\tx\n\\\t100.0\t200.0\n", encoding="utf-8"
    )  # spectral label lines start with a backslash
    assert read_labels(tmp_path / "audacity.txt") == [Label(0.1, 0.2, "x")]
