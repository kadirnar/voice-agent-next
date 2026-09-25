"""T4 turn-taking battery: stimuli with pauses / barge-ins and the battery on mock engines."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from voice_agent_next import CascadeOptions, SessionOptions
from voice_agent_next.bench.results import load_run
from voice_agent_next.bench.stimuli import (
    Scenario,
    builtin_scenario,
    load_scenario,
    noise_burst,
    render_stimuli,
)
from voice_agent_next.bench.system import BenchSystem
from voice_agent_next.bench.tracks.turn_taking import (
    TurnTakingOptions,
    category_of,
    render_turn_taking_report,
    run_turn_taking_benchmark,
    turn_taking_markdown_table,
)
from voice_agent_next.engines.cascade import CascadeEngine
from voice_agent_next.errors import ConfigurationError
from voice_agent_next.presets import LOCAL_TURN_TAKING
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import MockLLM, MockSTT, MockTTS, MockTurnDetector

REPLY = "Sure, we open at nine every day."  # 32 chars: ~2.1 s of mock speech


def test_published_scenario_hashes_do_not_change() -> None:
    # computed with the scenario model before `parts`/`barge_in`/`category` existed
    assert (
        builtin_scenario("latency-smoke").definition_sha256()
        == "4577d9613a5cefc2683b7fe1e3c4847efb38a60a1d78a8cd224d36ac4236cb15"
    )


def test_parts_render_with_exact_pauses() -> None:
    scn = Scenario.model_validate(
        {"name": "p", "turns": [{"id": "x", "parts": [
            {"duration": 0.5, "pause": 0.7}, {"duration": 0.4, "pause": 0.3},
            {"duration": 0.3}]}]}
    )  # fmt: skip
    (stim,) = asyncio.run(render_stimuli(scn))
    assert len(stim.pauses) == 2
    (a0, a1), (b0, b1) = stim.pauses
    assert a1 - a0 == pytest.approx(0.7) and b1 - b0 == pytest.approx(0.3)
    assert a0 == pytest.approx(0.5, abs=0.02)
    assert stim.speech_end == pytest.approx(0.5 + 0.7 + 0.4 + 0.3 + 0.3, abs=0.05)
    assert category_of(stim) == "pause"
    assert stim.describe()["pauses_s"][0][1] == pytest.approx(a1, abs=1e-6)


def test_noise_and_barge_in_stimuli() -> None:
    scn = Scenario.model_validate(
        {"name": "n", "turns": [
            {"id": "q", "duration": 0.5},
            {"id": "cough", "source": "noise", "duration": 0.3, "barge_in": 0.8,
             "expect_reply": False, "seed": 3, "loudness_dbfs": -30},
            {"id": "bc", "duration": 0.3, "barge_in": 1.0, "expect_reply": False},
            {"id": "stop", "duration": 0.8, "barge_in": 1.0}]}
    )  # fmt: skip
    q, cough, bc, stop = asyncio.run(render_stimuli(scn))
    assert cough.source == "noise" and cough.barge_in == 0.8
    assert cough.gain_db != 0.0
    assert [category_of(s) for s in (q, cough, bc, stop)] == [
        "question", "noise", "backchannel", "interruption",
    ]  # fmt: skip
    a = noise_burst(0.3, 16000, seed=3)
    assert a.data == noise_burst(0.3, 16000, seed=3).data != noise_burst(0.3, 16000).data


@pytest.mark.parametrize(
    "turn",
    [
        {"parts": [{"duration": 0.3}], "source": "wav", "wav": "x.wav"},
        {"source": "noise"},
        {"parts": [{"pause": 0.3}]},
    ],
)
def test_invalid_turns(turn: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Scenario.model_validate({"name": "x", "turns": [turn]})


def test_builtin_battery_scenario_loads() -> None:
    scn = load_scenario("turn-taking-smoke")
    stims = asyncio.run(render_stimuli(scn))
    cats = {category_of(s) for s in stims}
    assert cats == {"question", "pause", "backchannel", "noise", "interruption"}
    scn_file = Path(__file__).parents[2] / "benchmarks/scenarios/turn-taking-local.yaml"
    local = load_scenario(scn_file)
    assert {t.category for t in local.turns} >= {"pause", "backchannel", "interruption"}


def test_missing_wav_is_a_configuration_error(tmp_path: Path) -> None:
    scn = load_scenario({"name": "w", "turns": [{"id": "a", "source": "wav", "wav": "no.wav"}]})
    with pytest.raises(ConfigurationError):
        asyncio.run(render_stimuli(scn.with_base_dir(tmp_path)))


BATTERY = {
    "name": "battery-test",
    "lead_in": 0.3,
    "reply_timeout": 4.0,
    "gap_after_reply": 0.3,
    "turns": [
        {"id": "hello", "duration": 0.4},
        {"id": "order", "parts": [{"duration": 0.5, "pause": 0.7}, {"duration": 0.4}]},
        {"id": "policy", "duration": 0.5},
        {"id": "uh-huh", "duration": 0.3, "barge_in": 0.6, "expect_reply": False},
        {"id": "hours", "duration": 0.5},
        {"id": "stop", "duration": 0.9, "barge_in": 0.6},
    ],
}


def engine(min_silence: float) -> dict[str, Any]:
    return {
        "provider": "mock",
        "default_response": REPLY,
        "vad_options": {"min_silence_duration": min_silence},
    }


def test_eager_mock_engine_answers_in_mid_turn_pauses(tmp_path: Path) -> None:
    system = BenchSystem.from_options(engine=engine(0.3), label="eager")
    results = asyncio.run(
        run_turn_taking_benchmark(
            system, load_scenario(BATTERY), TurnTakingOptions(), out_dir=tmp_path
        )
    )
    s = results.summary
    assert s.rates["premature_rate"] == 1.0  # replies in the 0.7 s pause
    assert s.rates["premature_rate_questions"] == 0.0
    assert s.counts["interruptions"] == 1 and s.counts["backchannels"] == 1
    assert s.counts["not_overlapped"] == 0
    # the session pauses the agent at the user's onset: the stop is quick
    stop = s.metrics["barge_in_stop_ms"]
    assert stop.n == 1 and stop.p50 is not None and 0 < stop.p50 < 500
    assert s.rates["stop_within_500ms_rate"] == 1.0 and s.rates["interrupted_rate"] == 1.0
    assert s.metrics["post_interrupt_response_ms"].n == 1
    assert s.rates["backchannel_yield_rate"] == 1.0
    assert s.rates["missed_rate"] == 0.0
    items = {it["stimulus"]: it for it in results.items}
    assert items["order"]["premature"] and items["order"]["category"] == "pause"
    assert items["stop"]["overlapped"] and items["stop"]["talk_over_ms"] >= 0
    assert "premature replies" in turn_taking_markdown_table(results)
    loaded = load_run(results.directory)  # type: ignore[arg-type]
    assert "Turn-taking battery" in render_turn_taking_report(loaded)
    assert (results.directory / "artifacts/session-000/stereo.wav").exists()  # type: ignore[operator]


def test_patient_mock_engine_waits_through_pauses() -> None:
    scn = load_scenario({**BATTERY, "turns": BATTERY["turns"][:2]})
    system = BenchSystem.from_options(engine=engine(1.0), label="patient")
    results = asyncio.run(
        run_turn_taking_benchmark(system, scn, TurnTakingOptions(save_audio=False))
    )
    s = results.summary
    assert s.rates["premature_rate"] == 0.0
    v2v = s.metrics["v2v_ms"]
    assert v2v.n == 1 and v2v.p50 is not None and v2v.p50 == pytest.approx(1000, abs=150)


# ------------------------------------------ smoke battery on the mock cascade (issue #113)

CASCADE_BATTERY = {
    "name": "battery-mock-cascade",
    "lead_in": 0.3,
    "reply_timeout": 4.0,
    "gap_after_reply": 0.3,
    "turns": [
        {"id": "hello", "duration": 0.5},
        {"id": "order", "parts": [{"duration": 0.5, "pause": 0.3}, {"duration": 0.5}]},
        {"id": "policy", "duration": 0.6},
        {"id": "uh-huh", "duration": 0.4, "barge_in": 0.8, "expect_reply": False,
         "category": "backchannel"},
        {"id": "hours", "duration": 0.6},
        {"id": "stop", "duration": 1.2, "barge_in": 0.8, "category": "interruption"},
    ],
}  # fmt: skip


def _voiced_seconds(audio: Any) -> float:
    x = np.abs(audio.to_float32())
    win = audio.sample_rate // 50
    frames = x[: len(x) // win * win].reshape(-1, win)
    return float((frames.max(axis=1) >= 0.01).sum() * 0.02)


def _hear(audio: Any) -> str:
    """The mock STT's transcripts: what small streaming ASR models make of the stimuli —
    a short reaction comes out as a wrong word pair (NeMo: "Uh-huh." -> "but high")."""
    return "but high" if _voiced_seconds(audio) < 0.5 else "Where is my order?"


class _MockCascade(BenchSystem):
    """Mock STT/LLM/TTS + energy VAD + mock turn detector, with the local presets'
    turn-taking settings."""

    def build_engine(self) -> Any:
        return CascadeEngine(
            stt=MockSTT(transcripts=_hear, default_text=""),  # no interim words
            llm=MockLLM(default_response=REPLY),
            tts=MockTTS(),
            vad=EnergyVAD(),
            turn_detector=MockTurnDetector(),
            options=CascadeOptions(**LOCAL_TURN_TAKING["cascade"]),
        )

    def session_options(self) -> SessionOptions:
        return SessionOptions(**LOCAL_TURN_TAKING["session"])


def test_mock_cascade_battery_smoke() -> None:
    """CI smoke version of the T4 battery on a cascade: no premature reply in a short
    pause, no false barge-in on a mis-transcribed backchannel, no dead air."""
    base = BenchSystem.from_options(stt="mock", llm="mock", tts="mock", vad="energy")
    system = _MockCascade(base.config, "mock-cascade")
    results = asyncio.run(
        run_turn_taking_benchmark(
            system, load_scenario(CASCADE_BATTERY), TurnTakingOptions(save_audio=False)
        )
    )
    s = results.summary
    assert s.counts["backchannels"] == 1 and s.counts["interruptions"] == 1
    assert s.rates["premature_rate"] == 0.0
    assert s.rates["false_barge_in_rate_backchannel"] == 0.0
    assert s.rates["backchannel_yield_rate"] == 1.0 and s.rates["resume_rate"] == 1.0
    assert s.rates["interrupted_rate"] == 1.0 and s.rates["stop_within_500ms_rate"] == 1.0
    assert s.rates["dead_air_rate"] == 0.0 and s.rates["missed_rate"] == 0.0
