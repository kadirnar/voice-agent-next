"""``van bench`` command group."""

from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from voice_agent_next.bench.results import ITEMS_FILE, MANIFEST_FILE, REPORT_FILE, SUMMARY_FILE
from voice_agent_next.cli.main import app

runner = CliRunner()
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    """Output without ANSI styling (CI sets FORCE_COLOR)."""
    return ANSI.sub("", text)


def tiny_scenario(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.yaml"
    path.write_text(
        "name: tiny\nlead_in: 0.2\ngap_after_reply: 0.15\nreply_timeout: 2.0\n"
        "turns:\n  - {id: a, duration: 0.3}\n",
        encoding="utf-8",
    )
    return path


def test_bench_group_is_registered() -> None:
    result = runner.invoke(app, ["bench", "--help"])
    assert result.exit_code == 0, result.output
    assert "latency" in plain(result.output) and "report" in plain(result.output)


def test_latency_command_writes_results_and_report_can_be_rerendered(tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "bench", "latency",
            "--engine", "{provider: mock, response_delay: 0.1, responses: [Ok.]}",
            "--scenario", str(tiny_scenario(tmp_path)),
            "--turns", "1", "--warmup-turns", "0",
            "--out", str(out), "--run-id", "cli-run", "--json",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["run_id"] == "cli-run" and summary["system"] == "mock"
    assert summary["metrics"]["v2v_ms"]["n"] == 1
    assert 440 < summary["metrics"]["v2v_ms"]["p50"] < 560  # 0.4 s VAD silence + 0.1 s
    run_dir = out / "cli-run"
    for name in (MANIFEST_FILE, ITEMS_FILE, SUMMARY_FILE, REPORT_FILE):
        assert (run_dir / name).is_file()
    assert (run_dir / "artifacts" / "session-000" / "stereo.wav").is_file()
    assert "turn   1/1" in plain(result.stderr)  # live progress goes to stderr

    (run_dir / REPORT_FILE).unlink()
    rendered = runner.invoke(app, ["bench", "report", str(run_dir)])
    assert rendered.exit_code == 0, rendered.output
    assert plain(rendered.stdout).startswith("# Latency (T1) · mock")
    written = (run_dir / REPORT_FILE).read_text(encoding="utf-8")
    assert written.rstrip("\n") == plain(rendered.stdout).rstrip("\n")


def test_latency_command_human_output(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "bench", "latency", "--stt", "mock", "--llm", "{provider: mock, responses: [Ok.]}",
            "--tts", "mock", "--vad", "energy", "--scenario", str(tiny_scenario(tmp_path)),
            "--turns", "1", "--warmup-turns", "0", "--out", str(tmp_path), "--no-audio",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    out = plain(result.stdout)
    assert "voice-to-voice" in out and "cascade:mock+mock+mock" in out
    (run_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert not (run_dir / "artifacts").exists()


def test_latency_command_rejects_bad_input(tmp_path: Path) -> None:
    base = ["bench", "latency", "--out", str(tmp_path)]
    both = runner.invoke(app, [*base, "--engine", "mock", "--stt", "mock"])
    assert both.exit_code == 2 and "not both" in plain(both.stderr)
    scenario = runner.invoke(app, [*base, "--scenario", "no-such-scenario"])
    assert scenario.exit_code == 2 and "scenario not found" in plain(scenario.stderr)
    inline = runner.invoke(app, [*base, "--engine", "{provider: mock"])
    assert inline.exit_code == 2 and "invalid inline component spec" in plain(inline.stderr)
    missing = runner.invoke(app, ["bench", "report", str(tmp_path / "nope")])
    assert missing.exit_code == 2
    assert list(tmp_path.iterdir()) == []  # nothing was written
