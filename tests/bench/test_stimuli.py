"""Scenario manifests and stimulus rendering."""

from __future__ import annotations

from pathlib import Path

import pydantic
import pytest

from tests.bench.helpers import concat, silence, tone
from voice_agent_next.audio import write_wav
from voice_agent_next.bench.onset import frame_levels_db
from voice_agent_next.bench.stimuli import (
    Scenario,
    TurnSpec,
    annotate_speech,
    builtin_scenario,
    load_scenario,
    normalize_loudness,
    render_stimuli,
)
from voice_agent_next.errors import ConfigurationError

REPO = Path(__file__).resolve().parents[2]
SCENARIOS = sorted((REPO / "benchmarks" / "scenarios").glob("*.yaml"))


def test_repository_ships_scenarios() -> None:
    assert {p.stem for p in SCENARIOS} >= {"latency-smoke", "latency-conversational"}


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_repository_scenarios_are_valid(path: Path) -> None:
    scenario = load_scenario(path)
    assert scenario.name == path.stem
    assert scenario.base_dir == path.parent


def test_builtin_smoke_scenario_matches_the_repository_file() -> None:
    from_file = load_scenario(REPO / "benchmarks" / "scenarios" / "latency-smoke.yaml")
    assert from_file.model_dump() == builtin_scenario("latency-smoke").model_dump()
    assert load_scenario("latency-smoke").name == "latency-smoke"  # built-in by name


async def test_synthetic_stimuli_are_cycled_normalized_and_chunk_aligned() -> None:
    scenario = Scenario(
        name="t",
        turns=[TurnSpec(id="a", duration=0.5), TurnSpec(id="b", text="hello there", pause=0.1)],
    )
    stimuli = await render_stimuli(scenario, turns=5)
    assert [s.id for s in stimuli] == ["a", "b", "a", "b", "a"]
    assert stimuli[0] is stimuli[2]  # rendered once, shared
    a, b = stimuli[0], stimuli[1]
    assert (a.speech_start, a.speech_end) == (0.0, pytest.approx(0.5))
    assert b.speech_end == pytest.approx(len("hello there") / 14.0, abs=0.001)  # 14 chars/s
    assert b.pause == 0.1 and b.source == "synthetic" and b.expect_reply
    for stim in (a, b):
        assert stim.audio.sample_rate == 16_000
        assert stim.audio.samples_per_channel % 320 == 0  # whole 20 ms chunks
        level = stim.audio.slice(stim.speech_start, stim.speech_end).dbfs()
        assert level == pytest.approx(-20.0, abs=0.1)
    again = await render_stimuli(scenario, turns=2)
    assert [s.sha256 for s in again] == [a.sha256, b.sha256]  # deterministic
    assert a.describe()["sha256"] == a.sha256


async def test_wav_stimuli_are_resampled_and_annotated(tmp_path: Path) -> None:
    write_wav(
        tmp_path / "clip.wav",
        concat(silence(0.25, 22_050), tone(0.6, 22_050), silence(0.3, 22_050)),
    )
    (tmp_path / "s.yaml").write_text(
        "name: wavs\nstimuli: wav\nloudness_dbfs: null\nturns:\n"
        "  - {id: auto, wav: clip.wav}\n"
        "  - {id: manual, wav: clip.wav, speech: [0.2, 0.9]}\n",
        encoding="utf-8",
    )
    auto, manual = await render_stimuli(load_scenario(tmp_path / "s.yaml"))
    assert auto.audio.sample_rate == 16_000 and auto.source == "wav"
    assert auto.speech_start == pytest.approx(0.25, abs=0.011)
    assert auto.speech_end == pytest.approx(0.85, abs=0.011)
    assert (manual.speech_start, manual.speech_end) == (0.2, 0.9)
    assert auto.gain_db == 0.0  # loudness normalization disabled


async def test_tts_stimuli_use_a_registered_tts() -> None:
    text = "Hello there, how are you?"
    scenario = Scenario(name="t", stimuli="tts", tts="mock", turns=[TurnSpec(text=text)])
    (stim,) = await render_stimuli(scenario)
    assert stim.source == "tts" and stim.id == "turn00"
    assert stim.speech_start == pytest.approx(0.0, abs=0.011)
    assert stim.speech_end == pytest.approx(len(text) / 15.0, abs=0.02)  # MockTTS: 15 chars/s


def test_invalid_scenarios_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(pydantic.ValidationError):
        Scenario(name="x", turns=[])
    with pytest.raises(pydantic.ValidationError):
        Scenario(name="x", stimuli="wav", turns=[TurnSpec(text="hi")])
    with pytest.raises(pydantic.ValidationError):
        Scenario(name="x", stimuli="tts", turns=[TurnSpec(text="hi")])  # no tts spec
    with pytest.raises(pydantic.ValidationError):
        Scenario(name="x", turns=[TurnSpec()])  # no duration or text
    with pytest.raises(pydantic.ValidationError):
        TurnSpec(duration=1.0, speech=(0.5, 0.2))
    with pytest.raises(pydantic.ValidationError):
        Scenario.model_validate({"name": "x", "turns": [{"duration": 1}], "bogus": 1})
    with pytest.raises(ConfigurationError):
        load_scenario(tmp_path / "missing.yaml")
    (tmp_path / "bad.yaml").write_text("name: x\nturns: []\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid scenario"):
        load_scenario(tmp_path / "bad.yaml")
    with pytest.raises(ConfigurationError):
        builtin_scenario("nope")


async def test_missing_wav_is_reported(tmp_path: Path) -> None:
    (tmp_path / "s.yaml").write_text(
        "name: w\nstimuli: wav\nturns: [{wav: nope.wav}]\n", encoding="utf-8"
    )
    with pytest.raises(ConfigurationError, match="not found"):
        await render_stimuli(load_scenario(tmp_path / "s.yaml"))


def test_annotation_and_loudness_helpers() -> None:
    audio = concat(silence(0.3, 16_000), tone(0.5, 16_000, amplitude=0.05), silence(0.2, 16_000))
    start, end = annotate_speech(audio)
    assert (start, end) == (pytest.approx(0.3, abs=0.011), pytest.approx(0.8, abs=0.011))
    loud, gain_db = normalize_loudness(audio, -20.0, (start, end))
    assert loud.slice(start, end).dbfs() == pytest.approx(-20.0, abs=0.1)
    assert gain_db == pytest.approx(-20.0 - audio.slice(start, end).dbfs(), abs=0.1)
    clipped, _ = normalize_loudness(tone(0.2, 16_000, amplitude=0.9), 0.0)
    assert max(abs(int(v)) for v in clipped.to_numpy()) < 32767  # gain limited by the peak
    with pytest.raises(ValueError):
        annotate_speech(silence(0.5, 16_000))
    assert len(frame_levels_db(audio)) == 100
