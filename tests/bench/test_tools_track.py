"""T6 tool-use track: mock tool world, scoring and runs on scripted mock systems."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from voice_agent_next.bench.results import load_run
from voice_agent_next.bench.system import BenchSystem
from voice_agent_next.bench.tool_env import (
    TOOL_LIBRARY,
    CallRecord,
    ExpectedCall,
    ToolScenario,
    ToolSuite,
    build_tools,
    canonical,
    load_tool_suite,
    reference_engine,
    state_hash,
)
from voice_agent_next.bench.tool_scoring import (
    TurnObservation,
    match_calls,
    say_do_violations,
    score_scenario,
    ungrounded_facts,
)
from voice_agent_next.bench.tracks.tools import (
    ToolsOptions,
    render_tools_report,
    run_tools_benchmark,
    tools_markdown_table,
)
from voice_agent_next.chat import FunctionCall
from voice_agent_next.engines.cascade import CascadeEngine
from voice_agent_next.errors import ConfigurationError, ToolError
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import MockEngine, MockLLM, MockSTT, MockToolCall, MockTTS
from voice_agent_next.tools import execute_function_call

SMOKE_SHA256 = "b23497dcf43d72354805410ef827b23f7c5e19c1e1fa92c50ef9985c55dd77ca"


# ----------------------------------------------------------------- tool world


@pytest.mark.parametrize(
    ("value", "kind", "expected"),
    [
        ("October 3rd", "date", "2026-10-03"),
        ("3 October 2027", "date", "2027-10-03"),
        ("2026-10-03", "date", "2026-10-03"),
        ("10/3", "date", "2026-10-03"),
        ("7 pm", "time", "19:00"),
        ("7:30 PM", "time", "19:30"),
        ("seven pm", "time", "19:00"),
        ("19:00", "time", "19:00"),
        ("12 am", "time", "00:00"),
        ("jane dot doe at example dot com", "email", "jane.doe@example.com"),
        ("Jane.Doe@Example.com", "email", "jane.doe@example.com"),
        ("one oh four two", "id", "1042"),
        ("#1042", "id", "1042"),
        ("four four one O", "id", "4410"),
        ("order 1042", "id", "1042"),
        ("four", "int", "4"),
        (4.0, "int", "4"),
        ("twenty-one", "int", "21"),
        ("12 Oak St., Springfield", "text", "12 oak street springfield"),
        ("Maria  Lopez", "name", "maria lopez"),
        ("+1 (555) 010-2030", "phone", "15550102030"),
    ],
)
def test_canonical_arguments(value: Any, kind: Any, expected: str) -> None:
    assert canonical(value, kind) == expected


def test_smoke_suite_is_pinned_and_consistent() -> None:
    suite = load_tool_suite("smoke")
    assert suite.definition_sha256() == SMOKE_SHA256  # bump `version` and this on a change
    assert 10 <= len(suite.scenarios) <= 12
    for scenario in suite.scenarios:
        assert 3 <= len(scenario.turns) <= 6, scenario.id
        assert scenario.expected_calls or scenario.expect_said, scenario.id
        suite.expected_db(scenario)  # every expected write replays cleanly
    writes = {
        s.id
        for s in suite.scenarios
        if state_hash(suite.expected_db(s)) != state_hash(suite.initial_db(s))
    }
    assert {"cancel-order", "book-table", "escalate"} <= writes
    assert "cancel-shipped" not in writes  # the policy scenario must not change anything
    stims = suite.stimulus_scenario()
    assert len(stims.turns) == sum(len(s.turns) for s in suite.scenarios)
    assert stims.turns[0].id == "order-status/t0"


def test_invalid_suites_are_rejected(tmp_path: Path) -> None:
    base = {
        "name": "x",
        "scenarios": [{"id": "a", "tools": ["lookup_order"], "turns": [{"text": "hi"}]}],
    }
    ToolSuite.model_validate(base)
    bad_tool = json.loads(json.dumps(base))
    bad_tool["scenarios"][0]["tools"] = ["launch_rocket"]
    with pytest.raises(ValueError, match="unknown tool"):
        ToolSuite.model_validate(bad_tool)
    not_offered = json.loads(json.dumps(base))
    not_offered["scenarios"][0]["turns"][0]["expect_calls"] = [{"name": "cancel_order"}]
    with pytest.raises(ValueError, match="not offered"):
        ToolSuite.model_validate(not_offered)
    bad_arg = json.loads(json.dumps(base))
    bad_arg["scenarios"][0]["turns"][0]["expect_calls"] = [
        {"name": "lookup_order", "args": {"id": "1"}}
    ]
    with pytest.raises(ValueError, match="no argument"):
        ToolSuite.model_validate(bad_arg)
    with pytest.raises(ConfigurationError):
        load_tool_suite(tmp_path / "missing.yaml")
    failing = {
        **base,
        "scenarios": [
            {
                "id": "a",
                "tools": ["cancel_order"],
                "turns": [
                    {
                        "text": "cancel 1",
                        "expect_calls": [{"name": "cancel_order", "args": {"order_id": "1"}}],
                    }
                ],
            }
        ],
    }
    with pytest.raises(ConfigurationError, match="fails"):
        ToolSuite.model_validate(failing).expected_db(
            ToolSuite.model_validate(failing).scenarios[0]
        )
    with pytest.raises(ConfigurationError, match="unknown scenario"):
        ToolSuite.model_validate(base).select(["nope"])


def test_mock_tools_read_write_and_enforce_policies() -> None:
    suite = load_tool_suite("smoke")
    scenario = next(s for s in suite.scenarios if s.id == "book-table")
    scenario = scenario.model_copy(update={"tools": [*scenario.tools, "cancel_order"]})
    db = suite.initial_db(scenario)
    log: list[CallRecord] = []
    tools = build_tools(suite, scenario, db, log, delay_scale=0.0)

    async def call(name: str, args: dict[str, Any]) -> Any:
        return await execute_function_call(
            FunctionCall(name=name, arguments=json.dumps(args)), tools
        )

    async def scenario_run() -> None:
        out = await call(
            "check_availability", {"date": "October 3", "time": "7 pm", "party_size": "four"}
        )
        assert json.loads(out.output)["available"] is True
        out = await call(
            "book_table",
            {"name": "Maria Lopez", "date": "2026-10-03", "time": "19:00", "party_size": 4},
        )
        booking = json.loads(out.output)
        assert booking["status"] == "confirmed" and not out.is_error
        out = await call("cancel_order", {"order_id": "1042"})  # shipped: refused
        assert out.is_error and "can no longer be cancelled" in out.output
        out = await call(
            "book_table", {"name": "X", "date": "2026-10-04", "time": "20:00", "party_size": 2}
        )  # fully booked
        assert out.is_error

    asyncio.run(scenario_run())
    assert [r.ok for r in log] == [True, True, False, False]
    assert log[1].changed_state and not log[0].changed_state
    assert db["availability"]["2026-10-03 19:00"] == 1
    # a replay of the same booking gives the same id: expected states are reproducible
    assert state_hash(db) == state_hash(suite.expected_db(scenario))
    with pytest.raises(ToolError):
        TOOL_LIBRARY["lookup_order"].run(db, {"order_id": "9999"})


# -------------------------------------------------------------------- scoring


def _rec(
    name: str,
    args: dict[str, Any],
    *,
    t: float = 0.0,
    ok: bool = True,
    output: str = "",
    changed: bool = False,
) -> CallRecord:
    return CallRecord(
        name, args, json.dumps(args), t, t + 0.1, ok, output, TOOL_LIBRARY[name].write, changed
    )


def test_call_matching_precision_recall_and_arguments() -> None:
    expected = [
        ExpectedCall(name="lookup_order", args={"order_id": "2077"}, optional=True),
        ExpectedCall(
            name="book_table",
            args={"name": "Maria Lopez", "date": "2026-10-03", "time": "19:00", "party_size": 4},
        ),
    ]
    actual = [
        _rec(
            "book_table",
            {"name": "maria lopes", "date": "Oct 3", "time": "7 PM", "party_size": "4"},
        ),
        _rec("lookup_order", {"order_id": "2077"}),
        _rec("lookup_order", {"order_id": "2077"}),  # a repeated look-up
    ]
    matches, unmatched = match_calls(expected, actual)
    assert [m.actual for m in matches] == [1, 0]
    assert unmatched == [2]
    book = matches[1]
    assert (book.args_correct, book.args_total) == (3, 4)  # the name was misheard
    assert (book.entities_correct, book.entities_total) == (0, 1)
    assert book.wrong_args and "maria lopes" in book.wrong_args[0]
    # nothing matches a missing call
    matches, unmatched = match_calls(expected[1:], [])
    assert matches[0].actual is None and unmatched == []


def test_say_do_violations_and_hallucinations() -> None:
    tools = ["lookup_order", "cancel_order", "book_table"]
    assert say_do_violations("Done! Your order has been cancelled.", tools, set())
    assert not say_do_violations("Your order has been cancelled.", tools, {"cancel_order"})
    assert not say_do_violations("I'm sorry, it can't be cancelled.", tools, set())
    assert not say_do_violations("Should I have it cancelled?", tools, set())
    assert not say_do_violations("Would you like me to cancel it?", tools, set())
    assert say_do_violations("I've booked a table for four.", tools, set())[0].startswith(
        "book_table"
    )
    output = json.dumps(
        {"order_id": "1042", "status": "shipped", "total": 89.99, "delivery_date": "2026-10-02"}
    )
    user = ["The order number is one oh four two."]
    assert (
        ungrounded_facts(
            "Order 1042 has shipped and arrives on October 2nd, $89.99.", [output, *user]
        )
        == []
    )
    assert ungrounded_facts("Your table is at 7 pm.", ['{"time": "19:00"}']) == []
    assert ungrounded_facts("It was delivered, total $45.", [*user]) == ["45", "delivered"]
    assert ungrounded_facts("Your order 4410 is on its way.", ["four four one oh"]) == []


def test_status_words_need_a_stated_status() -> None:
    assert ungrounded_facts("It has shipped.", []) == ["shipped"]
    assert ungrounded_facts("It is currently in transit.", []) == ["in transit"]
    assert ungrounded_facts("It will be delivered on Friday.", []) == []
    assert ungrounded_facts("It has shipped.", ['{"status": "shipped"}']) == []


def test_caller_voice_override() -> None:
    suite = load_tool_suite("smoke")
    assert suite.stimuli == "synthetic" and suite.with_caller(None) is suite
    voiced = suite.with_caller({"provider": "kokoro"})
    assert voiced.stimuli == "tts" and voiced.tts == {"provider": "kokoro"}
    assert voiced.with_caller("synthetic").tts is None
    assert voiced.stimulus_scenario().tts == {"provider": "kokoro"}


def _turns(texts: list[str], agent: list[str], states: list[str | None]) -> list[TurnObservation]:
    return [
        TurnObservation(i, u, a, float(i + 1), s)
        for i, (u, a, s) in enumerate(zip(texts, agent, states, strict=True))
    ]


def test_score_scenario_pass_and_fail() -> None:
    suite = load_tool_suite("smoke")
    scenario = next(s for s in suite.scenarios if s.id == "cancel-order")
    texts = [t.text for t in scenario.turns]
    done = suite.expected_db(scenario)
    good_calls = [
        _rec("lookup_order", {"order_id": "2077"}, t=1.2, output='{"status": "processing"}'),
        _rec("cancel_order", {"order_id": "2077"}, t=2.2, changed=True),
    ]
    initial = state_hash(suite.initial_db(scenario))
    final = state_hash(done)
    good = score_scenario(
        suite,
        scenario,
        good_calls,
        _turns(
            texts,
            ["Sure.", "It is processing. Cancel it?", "Your order is cancelled.", "Bye."],
            [initial, initial, final, final],
        ),
        done,
    )
    assert good.passed and good.state_ok and good.precision == 1.0 and good.recall == 1.0
    assert good.arg_acc == 1.0 and good.unnecessary_calls == 0 and not good.say_do
    assert good.turns_to_completion == 3

    bad = score_scenario(
        suite,
        scenario,
        [_rec("lookup_order", {"order_id": "2070"}, t=1.2, ok=False)],
        _turns(
            texts,
            ["Sure.", "Found it.", "Your order has been cancelled, refund of $12.", "Bye."],
            [initial] * 4,
        ),
        suite.initial_db(scenario),
    )
    assert not bad.passed and not bad.state_ok
    assert bad.state_diff and "orders.2077.status" in bad.state_diff[0]
    assert bad.recall == 0.0 and bad.arg_acc == 0.0 and bad.tool_errors == 1
    assert bad.say_do and bad.say_do_turns == [2]
    assert bad.ungrounded == {2: ["12"]}
    assert bad.turns_to_completion is None
    assert bad.missing_calls == ["cancel_order({'order_id': '2077'})"]


# ---------------------------------------------------------------------- runs

MINI: dict[str, Any] = {
    "name": "mini",
    "lead_in": 0.2,
    "reply_timeout": 3.0,
    "gap_after_reply": 0.3,
    "tool_delay": 0.05,
    "instructions": "You are a store assistant. Today is {today}.",
    "database": {"orders": {"2077": {"status": "processing", "total": 45.5}}},
    "scenarios": [
        {
            "id": "cancel",
            "tools": ["lookup_order", "cancel_order"],
            "turns": [
                {
                    "text": "Cancel order two oh seven seven.",
                    "duration": 0.5,
                    "expect_calls": [{"name": "cancel_order", "args": {"order_id": "2077"}}],
                },
                {"text": "Thanks.", "duration": 0.4},
            ],
            "expect_said": [["cancel"]],
        }
    ],
}


def _mock_system() -> BenchSystem:
    return BenchSystem.from_options(engine="mock", label="scripted")


def test_reference_engine_passes_every_check(tmp_path: Path) -> None:
    suite = load_tool_suite(MINI)
    results = asyncio.run(
        run_tools_benchmark(
            _mock_system(),
            suite,
            ToolsOptions(trials=2),
            out_dir=tmp_path,
            engine_factory=reference_engine,
        )
    )
    s = results.summary
    assert s.rates["pass_at_1"] == 1.0 and s.rates["pass_hat_k"] == 1.0
    assert s.rates["tool_f1"] == 1.0 and s.rates["arg_acc"] == 1.0
    assert s.rates["say_do_violation_rate"] == 0.0 and s.rates["hallucination_rate"] == 0.0
    assert s.counts["calls_run"] == 2 and s.counts["tool_rounds"] == 2
    lat = s.metrics["tool_round_latency_ms"]
    assert lat.n == 2 and lat.p50 is not None and lat.p50 > 0
    # the tool round = speech end -> call -> 50 ms mock tool -> speech
    item = results.items[0]
    round0 = item["turn_details"][0]
    assert round0["tool_round"] and round0["tool_exec_ms"][0] >= 35  # 40 ms tool; Windows timers
    assert round0["tool_round_latency_ms"] >= round0["pre_tool_ms"]
    assert item["calls_detail"][0]["name"] == "cancel_order"
    assert "tool F1" in tools_markdown_table(results)
    loaded = load_run(results.directory)  # type: ignore[arg-type]
    assert loaded.manifest.scenario["scenarios"][0]["sha256"]
    assert "Tool use (T6)" in render_tools_report(loaded)
    assert (results.directory / "artifacts/session-001/stereo.wav").exists()  # type: ignore[operator]


def test_mock_engine_that_claims_without_acting_fails() -> None:
    suite = load_tool_suite(MINI)

    def liar(_suite: ToolSuite, scenario: ToolScenario) -> MockEngine:
        return MockEngine(
            transcripts=[t.text for t in scenario.turns],
            responses=["Done, your order has been cancelled.", "Bye."],
            chars_per_second=40.0,
            vad_options={"min_silence_duration": 0.3},
        )

    results = asyncio.run(
        run_tools_benchmark(
            _mock_system(), suite, ToolsOptions(save_audio=False), engine_factory=liar
        )
    )
    s = results.summary
    assert s.rates["pass_at_1"] == 0.0 and s.rates["state_ok_rate"] == 0.0
    assert s.rates["tool_recall"] == 0.0 and s.rates["tool_precision"] is None
    assert s.rates["say_do_violation_rate"] == 1.0
    assert s.counts["tool_rounds"] == 0
    assert "Findings" in results.report and "say-do" in results.report


def test_mock_cascade_scripted_to_succeed() -> None:
    suite = load_tool_suite(MINI)

    def cascade(_suite: ToolSuite, scenario: ToolScenario) -> CascadeEngine:
        return CascadeEngine(
            stt=MockSTT(transcripts=[t.text for t in scenario.turns]),
            llm=MockLLM(
                responses=[
                    MockToolCall("lookup_order", {"order_id": "2077"}),
                    MockToolCall("cancel_order", {"order_id": "two oh seven seven"}),
                    "Your order 2077 is cancelled.",
                    "You're welcome.",
                ]
            ),
            tts=MockTTS(chars_per_second=40.0),
            vad=EnergyVAD(),
        )

    results = asyncio.run(
        run_tools_benchmark(
            _mock_system(), suite, ToolsOptions(save_audio=False), engine_factory=cascade
        )
    )
    s = results.summary
    assert s.rates["pass_at_1"] == 1.0, results.items[0]
    # the look-up was not expected (not even optional here): one unnecessary call
    assert s.rates["tool_recall"] == 1.0 and s.rates["tool_precision"] == 0.5
    assert s.counts["unnecessary_calls"] == 1 and s.counts["unexpected_writes"] == 0
    assert s.rates["arg_acc"] == 1.0  # spoken digits are normalized
    assert s.rates["hallucination_rate"] == 0.0
    details = results.items[0]["turn_details"][0]
    assert details["calls"] == ["lookup_order", "cancel_order"]


def test_cli_runs_the_reference_on_a_suite_file(tmp_path: Path) -> None:
    from voice_agent_next.cli.main import app

    path = tmp_path / "mini.yaml"
    path.write_text(yaml.safe_dump(MINI), encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "bench",
            "tools",
            "-s",
            str(path),
            "--no-audio",
            "--json",
            "--out",
            str(tmp_path),
            "--run-id",
            "t6",
        ],
        env={"NO_COLOR": "1", "FORCE_COLOR": ""},
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout[result.stdout.index("{") :])
    assert summary["track"] == "tools" and summary["rates"]["pass_at_1"] == 1.0
    assert (tmp_path / "t6" / "report.md").exists()
    bad = CliRunner().invoke(app, ["bench", "tools", "-s", str(path), "--only", "nope"])
    assert bad.exit_code != 0
    preset = CliRunner().invoke(app, ["bench", "tools", "--preset", "no-such-preset"])
    assert preset.exit_code == 2
    both = CliRunner().invoke(app, ["bench", "tools", "--preset", "local-cpu", "-c", str(path)])
    assert both.exit_code == 2
