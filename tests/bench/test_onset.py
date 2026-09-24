"""Agent onset detection with the reference VAD (10 ms frames, >= 100 ms of speech)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.bench.helpers import concat, mix, silence, tone
from voice_agent_next.audio import AudioFrame
from voice_agent_next.bench.onset import (
    OnsetDetector,
    ProviderReferenceVAD,
    RMSReferenceVAD,
    find_onsets,
    first_onset_between,
    frame_bounds,
    frame_levels_db,
    make_reference_vad,
    speech_runs,
)


@pytest.mark.parametrize("rate", [8_000, 16_000, 22_050, 24_000, 44_100])
def test_frames_sit_on_exact_10ms_boundaries_at_any_rate(rate: int) -> None:
    audio = concat(silence(1.0, rate), tone(0.5, rate), silence(0.5, rate))
    levels = frame_levels_db(audio)
    assert len(levels) == 200
    speech = np.flatnonzero(levels >= -40.0)
    assert speech[0] == 100 and speech[-1] == 149  # no drift even at 22.05 kHz
    assert np.isneginf(levels[0])  # digital silence
    bounds = frame_bounds(audio.samples_per_channel, rate, 0.01)
    assert bounds[0] == 0 and bounds[-1] == audio.samples_per_channel


def test_partial_last_frame_is_kept() -> None:
    bounds = frame_bounds(1_050, 16_000, 0.01)  # 6.56 frames of 160 samples
    assert list(bounds) == [0, 160, 320, 480, 640, 800, 960, 1_050]
    assert len(frame_levels_db(tone(1_050 / 16_000, 16_000))) == 7


def test_onset_ignores_clicks_and_comfort_noise() -> None:
    rate = 24_000
    rng = np.random.default_rng(0)
    noise = AudioFrame.from_numpy(
        (rng.standard_normal(3 * rate) * 10 ** (-60 / 20)).astype(np.float32), rate
    )  # comfort noise at -60 dBFS
    click = AudioFrame.from_numpy(np.full(72, 0.9, dtype=np.float32), rate)  # 3 ms
    speech = tone(0.5, rate)
    signal = concat(
        silence(0.2, rate), click, silence(1.234 - 0.2 - 0.003, rate), speech, silence(1.0, rate)
    )
    audio = mix(signal, noise)

    refined = OnsetDetector().onsets(audio)
    assert refined == [pytest.approx(1.234, abs=0.0015)]
    coarse = OnsetDetector(refine=False).onsets(audio)
    assert coarse == [pytest.approx(1.23, abs=1e-9)]  # start of the 10 ms frame


def test_runs_shorter_than_min_speech_are_not_onsets() -> None:
    rate = 16_000
    audio = concat(
        silence(0.5, rate), tone(0.09, rate),  # 90 ms: too short
        silence(0.41, rate), tone(0.1, rate),  # exactly 100 ms: counts
        silence(0.9, rate), tone(0.3, rate), silence(0.2, rate),
    )  # fmt: skip
    det = OnsetDetector()
    assert det.onsets(audio) == [pytest.approx(1.0, abs=0.002), pytest.approx(2.0, abs=0.002)]
    assert det.segments(audio) == [
        (pytest.approx(1.0), pytest.approx(1.1)),
        (pytest.approx(2.0), pytest.approx(2.3)),
    ]


def test_masks_runs_and_gap_bridging() -> None:
    mask = np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0], dtype=bool)
    assert speech_runs(mask) == [(1, 3), (4, 14), (15, 16)]
    assert speech_runs(mask, max_gap_frames=1) == [(1, 16)]
    assert find_onsets(mask) == [pytest.approx(0.04)]
    assert find_onsets(mask, max_gap=0.01) == [pytest.approx(0.01)]
    assert find_onsets(mask, min_speech=0.2) == []
    assert speech_runs(np.zeros(5, dtype=bool)) == []
    assert speech_runs(np.ones(3, dtype=bool)) == [(0, 3)]


def test_first_onset_between_is_start_inclusive_end_exclusive() -> None:
    onsets = [0.5, 1.0, 2.0]
    assert first_onset_between(onsets, 0.6, 1.5) == 1.0
    assert first_onset_between(onsets, 1.0, 2.0) == 1.0
    assert first_onset_between(onsets, 1.1, 2.0) is None
    assert first_onset_between(onsets, 1.1) == 2.0
    assert first_onset_between([], 0.0) is None


def test_threshold_is_configurable() -> None:
    rate = 16_000
    quiet = tone(0.5, rate, amplitude=0.01)  # ~ -43 dBFS
    audio = concat(silence(0.5, rate), quiet, silence(0.5, rate))
    assert OnsetDetector().onsets(audio) == []
    assert OnsetDetector(RMSReferenceVAD(-50.0)).onsets(audio) == [pytest.approx(0.5, abs=0.002)]
    assert isinstance(make_reference_vad(None), RMSReferenceVAD)
    assert make_reference_vad("rms:-50").threshold_db == -50.0


def test_registry_vad_as_reference() -> None:
    rate = 24_000
    audio = concat(silence(0.5, rate), tone(0.6, rate), silence(0.5, rate))
    vad = make_reference_vad("energy")
    assert isinstance(vad, ProviderReferenceVAD)
    assert vad.describe()["provider"] == "energy"
    probs = vad.frame_probabilities(audio, 0.01)
    assert len(probs) == 160
    det = OnsetDetector(vad)
    onsets = det.onsets(audio)
    assert onsets == [pytest.approx(0.5, abs=0.025)]  # 20 ms VAD windows
    assert OnsetDetector(vad).describe()["reference_vad"]["name"] == "vad:energy"
