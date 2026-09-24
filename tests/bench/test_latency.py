"""T1 latency track end to end: the harness is validated against known injected delays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from voice_agent_next.audio import read_wav
from voice_agent_next.bench import (
    BenchSystem,
    LatencyItem,
    LatencyOptions,
    Scenario,
    TurnSpec,
    load_run,
    read_labels,
    run_latency_benchmark,
)
from voice_agent_next.bench.results import ITEMS_FILE, MANIFEST_FILE, REPORT_FILE, SUMMARY_FILE
from voice_agent_next.errors import ConfigurationError

# short turns and replies keep these real-time tests fast
FAST = Scenario(
    name="fast",
    lead_in=0.2,
    gap_after_reply=0.15,
    reply_timeout=2.0,
    turns=[TurnSpec(id="a", duration=0.3), TurnSpec(id="b", duration=0.4)],
)
MOCK_VAD_SILENCE = 0.4  # MockEngine's energy VAD: min_silence_duration


def mock_engine(**kw: Any) -> dict[str, Any]:
    return {"provider": "mock", "responses": ["Ok."] * 4, **kw}  # "Ok." = 0.2 s of audio


def items_of(results: Any) -> list[LatencyItem]:
    return [LatencyItem.model_validate(i) for i in results.items]


@pytest.mark.parametrize("delay", [0.1, 0.4])
async def test_measured_v2v_is_vad_silence_plus_injected_delay(delay: float) -> None:
    system = BenchSystem.from_options(engine=mock_engine(response_delay=delay))
    results = await run_latency_benchmark(
        system, FAST, LatencyOptions(turns=2, warmup_turns=0, save_audio=False)
    )
    items = items_of(results)
    assert len(items) == 2
    expected_ms = (MOCK_VAD_SILENCE + delay) * 1000
    for item in items:
        assert not item.missed and not item.premature and not item.dead_air
        assert item.v2v_ms == pytest.approx(expected_ms, abs=60)
        # the recording and the session's own metric agree on a loopback transport
        assert item.residual_ms == pytest.approx(0.0, abs=25)
        assert item.eou_delay_ms == pytest.approx(MOCK_VAD_SILENCE * 1000, abs=40)
        assert item.engine_ttfb_ms == pytest.approx(delay * 1000, abs=40)
        assert item.agent_speech_ms == pytest.approx(200, abs=30)
        assert item.agent_transcript == "Ok."
    summary = results.summary
    assert summary.n == 2 and summary.metrics["v2v_ms"].n == 2
    assert summary.metrics["v2v_ms"].p50 == pytest.approx(expected_ms, abs=60)
    assert summary.rates["dead_air_rate"] == 0.0 and summary.counts["replies"] == 2
    assert summary.extra["spans_p50_ms"]["eou_delay"] == pytest.approx(400, abs=40)


async def test_cascade_from_registry_specs_writes_all_result_files(tmp_path: Path) -> None:
    system = BenchSystem.from_options(
        stt="mock", llm={"provider": "mock", "responses": ["Ok."] * 4}, tts="mock", vad="energy"
    )
    assert system.label == "cascade:mock+mock+mock" and system.kind == "cascade"
    turns: list[tuple[int, int]] = []
    results = await run_latency_benchmark(
        system,
        FAST,
        LatencyOptions(turns=2),  # default warm-up: turn 0 is the cold-start turn
        out_dir=tmp_path,
        on_turn=lambda session, turn: turns.append((session, turn.index)),
    )
    assert turns == [(0, 0), (0, 1)]
    items = items_of(results)
    # no turn detector: the cascade commits 0.6 s after the end of speech
    for item in items:
        assert item.v2v_ms == pytest.approx(600, abs=60)
        assert item.residual_ms == pytest.approx(0.0, abs=25)
        assert item.llm_ttft_ms is not None and item.tts_ttfb_ms is not None
        assert item.stt_latency_ms is not None
    assert [i.warmup for i in items] == [True, False]
    summary = results.summary
    assert summary.n == 1 and summary.metrics["first_turn_v2v_ms"].n == 1
    assert summary.system == "cascade:mock+mock+mock"
    assert summary.dataset.startswith("fast@sha256:")

    run_dir = results.directory
    assert run_dir is not None and run_dir.parent == tmp_path
    assert run_dir.name == results.manifest.run_id == summary.run_id
    assert "-latency-cascade-mock-mock-mock" in run_dir.name
    for name in (MANIFEST_FILE, ITEMS_FILE, SUMMARY_FILE, REPORT_FILE):
        assert (run_dir / name).is_file()
    manifest = json.loads((run_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["track"] == "latency" and manifest["created"].endswith("+00:00")
    assert manifest["system"]["engine"]["components"]["stt"]["provider"] == "mock"
    assert manifest["system"]["config"]["vad"] == "energy"
    assert [s["id"] for s in manifest["scenario"]["stimuli"]] == ["a", "b"]
    assert all(len(s["sha256"]) == 64 for s in manifest["scenario"]["stimuli"])
    assert manifest["transport"]["chunk_ms"] == 20
    assert manifest["options"]["onset"]["reference_vad"]["name"] == "rms"
    env = manifest["environment"]
    assert {"packages", "git", "python", "os", "cpu", "memory_bytes", "gpus"} <= set(env)
    loaded = load_run(run_dir)
    assert loaded.summary == summary and len(loaded.items) == 2
    report = (run_dir / REPORT_FILE).read_text(encoding="utf-8")
    assert "voice-to-voice" in report and "cascade:mock+mock+mock" in report

    session_dir = run_dir / "artifacts" / "session-000"
    wav = read_wav(session_dir / "stereo.wav")
    assert wav.channels == 2 and wav.duration > 2.0
    labels = [lb.text for lb in read_labels(session_dir / "labels.txt")]
    assert any(t.startswith("agent onset a: v2v") for t in labels)
    assert any(t.startswith("user b") for t in labels)
    timeline = [json.loads(line) for line in (session_dir / "timeline.jsonl").open()]
    assert any(e["event"] == "metrics.turn" for e in timeline)


async def test_missed_replies_dead_air_and_greeting() -> None:
    # a greeting first; turn "a" is answered 0.4 + 0.2 s after the user stopped (dead air
    # above 0.5 s); turn "b" gets an empty response: no audio at all -> missed
    system = BenchSystem.from_options(
        config={
            "engine": {"provider": "mock", "response_delay": 0.2, "responses": ["Ok.", ""]},
            "agent": {"greeting": "Hi!"},
        }
    )
    options = LatencyOptions(
        turns=2, warmup_turns=0, dead_air_threshold=0.5, reply_timeout=0.8, save_audio=False
    )
    results = await run_latency_benchmark(system, FAST, options)
    answered, missed = items_of(results)
    assert answered.dead_air and not answered.missed
    assert answered.v2v_ms == pytest.approx(600, abs=60)  # the reply, not the greeting
    assert missed.missed and missed.dead_air and missed.v2v_ms is None
    assert missed.agent_onset_s is None and not missed.agent_audio
    assert answered.agent_audio
    summary = results.summary
    assert summary.rates["dead_air_rate"] == 1.0 and summary.rates["missed_rate"] == 0.5
    assert summary.counts["missed"] == 1 and summary.counts["missed_with_audio"] == 0
    assert results.manifest.notes == []
    greeting = summary.metrics["greeting_ms"]
    assert greeting.n == 1 and greeting.p50 is not None
    assert greeting.p50 == pytest.approx(200, abs=40)  # response_delay applies to it too
    assert summary.metrics["session_ready_ms"].n == 1


def test_system_specs(tmp_path: Path) -> None:
    native = BenchSystem.from_options(engine="mock")
    assert native.kind == "native" and native.label == "mock"
    assert BenchSystem.from_options().label == "mock"  # default engine
    assert BenchSystem.from_options(engine={"provider": "mock", "model": "x"}).label == "mock/x"
    cascade = BenchSystem.from_options(
        config={"engine": "mock"}, stt="mock", llm="mock", tts="mock"
    )
    assert cascade.kind == "cascade" and cascade.config.engine is None  # flags win
    with pytest.raises(ConfigurationError):
        BenchSystem.from_options(engine="mock", stt="mock")
    with pytest.raises(ConfigurationError):
        BenchSystem.from_options(stt="mock", default_engine=None)  # no llm/tts
    agent_only = tmp_path / "agent.yaml"
    agent_only.write_text("agent: {greeting: Hi}\nsession: {allow_interruptions: false}\n")
    from_file = BenchSystem.from_options(config=agent_only, engine="mock")
    assert from_file.config.agent.greeting == "Hi" and from_file.label == "mock"
    assert not from_file.session_options().allow_interruptions
    assert BenchSystem.from_options(config=agent_only).config.engine == "mock"  # default
    toml = tmp_path / "cascade.toml"
    toml.write_text('stt = "mock"\nllm = "mock"\ntts = "mock"\nvad = "energy"\n')
    assert BenchSystem.from_options(config=toml).label == "cascade:mock+mock+mock"
    assert BenchSystem.from_options(config=from_file.config).config == from_file.config
    with pytest.raises(ConfigurationError, match="not found"):
        BenchSystem.from_options(config=tmp_path / "missing.yaml")
    described = BenchSystem.from_options(
        engine={"provider": "mock", "api_key": "sk-secret"}
    ).describe()
    assert described["config"]["engine"]["api_key"] == "***"
    assert "sk-secret" not in json.dumps(described)
