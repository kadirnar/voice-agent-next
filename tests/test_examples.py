"""The example gallery (``examples/``) runs: every script in ``--mock`` mode (issue #47).

Each example runs in its own interpreter, exactly as a user would run it
(``python examples/NN_name.py --mock``). They are offline: mock engines and providers,
local fake servers, WAV-file or in-memory transports. The processes start together in a
module fixture and each test waits for its own, so the whole module takes about as long
as the slowest example.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import typer

from voice_agent_next.cli.main import app as cli_app

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
TIMEOUT = 90.0  # per process; they take 1-8 s, but CI runners can be slow

# example -> lines its --mock output must contain
EXPECTED: dict[str, list[str]] = {
    "01_offline_local_agent.py": [
        "user : What can you do offline?",
        "agent: I can chat, call tools and keep your audio private.",
        "agent audio:",
    ],
    "02_openai_realtime.py": [
        "user : What's the weather in Paris?",
        "-> tool get_weather",
        "<- 'Sunny and 22 degrees in Paris.'",
        "agent: It's sunny in Paris.",
    ],
    "03_gemini_live.py": [
        "user : Is there an Italian restaurant nearby?",
        "-> tool find_restaurant",
        "agent: Trattoria Roma is 300 meters away.",
    ],
    "04_cascade_mix.py": [
        "failover: tts mock/stalled-cloud-tts -> mock/local-tts (timeout)",
        "agent: The next train to Lyon leaves at nine.",
    ],
    "05_tools.py": [
        "-> tool search_flights",
        "(filler after",
        "(progress, spoken=True: 'Found 3 flights to Rome, comparing prices.')",
        "agent: The cheapest is 90 euros at 9:10.",
        "-> tool email_itinerary",
        "(background result after",
        "agent: Done: the itinerary is in your inbox.",
    ],
    "06_telephony_twilio.py": [
        "<Connect><Stream url=",
        "[call CA0] agent: Thanks for calling the bike shop!",
        "[call CA0] user : Hi, is the shop open today?",
        "[call CA0] agent: Yes, until 6 pm.",
        "the caller heard",
    ],
    "07_realtime_server.py": [
        "agent: Hello from a local engine!",
        "user (transcribed by the server): What time is it?",
        "agent: It is noon.",
    ],
    "08_recording_and_tracing.py": [
        "-> tool check_order",
        "(2 channels,",
        "turn: voice-to-voice",
    ],
    "10_custom_provider.py": [
        "registered: llm/faq",
        "agent: We are open from nine to six, Monday to Saturday.",
    ],
    "11_handoffs.py": [
        "-> tool transfer_to_billing",
        "== handoff front -> billing (history=summary)",
        "agent: Done: 25 euros are on their way back to you.",
        "refunds=[25.0]",
        "<- 'We take tables for 1 to 10 people.'",
        "== handoff party -> confirm (history=full)",
        "agent: A table for four tonight. See you soon!",
        "flow path: greet -> party -> confirm",
    ],
}

BENCHMARK_DOC = EXAMPLES / "09_benchmark.md"
SMOKE_BENCH = (
    "van bench latency --engine mock --turns 3 --warmup-turns 0 --no-audio --out bench-results/"
)


def _env(cwd: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", NO_COLOR="1")
    env.update(TMPDIR=str(cwd), TEMP=str(cwd), TMP=str(cwd))  # the examples' scratch files
    return env


def _van(args: list[str]) -> list[str]:
    return [sys.executable, "-m", "voice_agent_next.cli.main", *args]


@dataclass
class Run:
    proc: subprocess.Popen[str]
    cwd: Path


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Run]]:
    """Start every example (and the documented benchmark smoke run) at once."""
    commands = {name: [sys.executable, str(EXAMPLES / name), "--mock"] for name in EXPECTED}
    commands["bench"] = _van(shlex.split(SMOKE_BENCH)[1:])
    started: dict[str, Run] = {}
    for name, argv in commands.items():
        cwd = tmp_path_factory.mktemp(name.removesuffix(".py"))  # nothing lands in the repo
        proc = subprocess.Popen(
            argv, cwd=cwd, env=_env(cwd), text=True, encoding="utf-8",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )  # fmt: skip
        started[name] = Run(proc, cwd)
    yield started
    for run in started.values():
        if run.proc.poll() is None:
            run.proc.kill()
        run.proc.communicate()


def _finish(run: Run) -> str:
    proc = run.proc
    try:
        out, _ = proc.communicate(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        pytest.fail(f"timed out after {TIMEOUT} s:\n{out}")
    assert proc.returncode == 0, out
    return out


def test_every_example_is_tested_and_indexed() -> None:
    scripts = {p.name for p in EXAMPLES.glob("[0-9][0-9]_*.py")}
    assert scripts == set(EXPECTED)
    index = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    for name in [*scripts, BENCHMARK_DOC.name]:
        assert f"]({name})" in index, f"{name} is missing from examples/README.md"
    readme = (EXAMPLES.parent / "README.md").read_text(encoding="utf-8")
    assert "examples/README.md" in readme


@pytest.mark.parametrize("name", list(EXPECTED))
def test_example_runs_in_mock_mode(name: str, runs: dict[str, Run]) -> None:
    out = _finish(runs[name])
    assert "Traceback" not in out, out
    assert not re.search(r"^\S*error: ", out, re.MULTILINE), out  # session errors
    for line in EXPECTED[name]:
        assert line in out, f"{line!r} not in the output of {name}:\n{out}"


# ------------------------------------------------------------------ 09_benchmark.md
def _documented_commands() -> list[list[str]]:
    """Every ``$ van ...`` line in the console blocks of the benchmark guide."""
    text = BENCHMARK_DOC.read_text(encoding="utf-8")
    commands = []
    for line in text.splitlines():
        if line.startswith("$ van "):
            commands.append(shlex.split(line[2:].split("  #")[0]))
    return commands


def test_benchmark_guide_commands_exist() -> None:
    """Every documented subcommand and option exists (the guide cannot rot silently)."""
    commands = _documented_commands()
    assert shlex.split(SMOKE_BENCH) in commands
    root = typer.main.get_command(cli_app)
    for argv in commands:
        cmd: Any = root
        rest = argv[1:]
        while hasattr(cmd, "get_command") and rest and not rest[0].startswith("-"):
            sub = cmd.get_command(None, rest[0])  # a (sub)command group
            assert sub is not None, f"unknown command in {argv}"
            cmd, rest = sub, rest[1:]
        options = {opt for p in cmd.params for opt in (*p.opts, *p.secondary_opts)}
        for arg in rest:
            if arg.startswith("-"):
                assert arg.split("=")[0] in options, f"{arg} is not an option of {argv}"


def test_benchmark_smoke_run(runs: dict[str, Run]) -> None:
    out = _finish(runs["bench"])
    [run_dir] = (runs["bench"].cwd / "bench-results").iterdir()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    v2v = summary["metrics"]["v2v_ms"]
    # the mock engine ends a turn after 400 ms of silence and answers at once (see the guide)
    assert summary["track"] == "latency" and v2v["n"] == 3, out
    assert 300 < v2v["p50"] < 700, v2v
    assert (run_dir / "report.md").exists()
