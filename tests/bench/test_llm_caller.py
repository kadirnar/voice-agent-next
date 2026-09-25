"""The LLM-driven T6 caller: prompt, line generation, dynamic calls and the CLI."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from voice_agent_next.bench.caller import CallerEmulator, TurnTiming
from voice_agent_next.bench.llm_caller import (
    END_TOKEN,
    LLMCaller,
    LLMCallerOptions,
    caller_prompt,
    parse_caller,
)
from voice_agent_next.bench.stimuli import Stimulus
from voice_agent_next.bench.system import BenchSystem
from voice_agent_next.bench.tool_env import load_tool_suite, reference_engine
from voice_agent_next.bench.tracks.tools import ToolsOptions, run_tools_benchmark
from voice_agent_next.chat import ChatContext
from voice_agent_next.llm import LLM, LLMStream
from voice_agent_next.providers.mock import MockLLM
from voice_agent_next.transports.loopback import LoopbackTransport

MINI: dict[str, Any] = {
    "name": "mini",
    "description": "A small web store.",
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


def test_parse_caller() -> None:
    assert parse_caller(None) is None and parse_caller("scripted") is None
    assert parse_caller("llm:ollama/qwen3.5:4b") == "ollama/qwen3.5:4b"
    assert parse_caller(" llm: mock ") == "mock"
    for bad in ("llm:", "ollama/qwen3", "LLM"):
        with pytest.raises(ValueError, match="--caller"):
            parse_caller(bad)
    with pytest.raises(ValueError, match="--caller"):
        ToolsOptions(caller="gpt").validate()
    with pytest.raises(ValueError, match="max_turns"):
        ToolsOptions(caller="llm:mock", caller_max_turns=0).validate()


def test_prompt_takes_persona_and_goal_from_the_scenario() -> None:
    suite = load_tool_suite(MINI)
    scenario = suite.scenarios[0]
    prompt = caller_prompt(suite, scenario)
    assert "customer" in prompt and "A small web store." in prompt
    assert "- Cancel order two oh seven seven." in prompt and "- Thanks." in prompt
    assert END_TOKEN in prompt and "Your goal" not in prompt  # no goal / description
    own = scenario.model_copy(update={"persona": "You are Sam, in a hurry.", "goal": "Cancel."})
    prompt = caller_prompt(suite, own)
    assert prompt.startswith("You are Sam, in a hurry.") and "Your goal for this call: Cancel." in (
        prompt
    )


def test_unset_persona_and_goal_keep_the_suite_hashes() -> None:
    suite = load_tool_suite(MINI)
    plain = suite.definition_sha256()
    assert "persona" not in suite.scenarios[0].model_dump()
    data = json.loads(json.dumps(MINI))
    data["scenarios"][0]["goal"] = "Cancel the order."
    changed = load_tool_suite(data)
    assert changed.scenarios[0].goal == "Cancel the order."
    assert changed.definition_sha256() != plain


def _caller(responses: Any, **options: Any) -> tuple[LLMCaller, MockLLM]:
    suite = load_tool_suite(MINI)
    llm = MockLLM(responses=responses)
    return LLMCaller(llm, suite, suite.scenarios[0], LLMCallerOptions(**options)), llm


async def test_next_line_reacts_to_the_agent_and_hangs_up() -> None:
    caller, llm = _caller(['Caller: "Hi, cancel order 2077."', "Thanks, bye.", END_TOKEN])
    assert await caller.next_line("") == "Hi, cancel order 2077."  # label and quotes gone
    assert await caller.next_line("Done, it is  cancelled.") == "Thanks, bye."
    assert await caller.next_line("Goodbye!") is None
    assert caller.ended and caller.errors == []
    # the agent's words are the caller model's "user" turns; its own lines the "assistant"
    last = llm.requests[-1].messages()
    assert [m.role for m in last] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert last[3].text == "Done, it is cancelled." and last[-1].text == "Goodbye!"
    assert caller.transcript()[-1] == {"agent": "Goodbye!", "caller": None}
    assert await caller.next_line("Anything else?") is None  # stays hung up


async def test_a_line_with_the_end_token_is_said_then_the_call_ends() -> None:
    caller, _ = _caller([f"Okay, goodbye. {END_TOKEN}"])
    assert await caller.next_line("") == "Okay, goodbye."
    assert caller.ended
    assert await caller.next_line("Bye!") is None


async def test_an_empty_reply_ends_the_call_with_an_error() -> None:
    caller, _ = _caller([""])  # e.g. a reasoning model that spent max_tokens thinking
    assert await caller.next_line("") is None
    assert not caller.ended and "empty reply" in caller.errors[0]


async def test_repeating_the_previous_line_hangs_up() -> None:
    caller, _ = _caller(["That's all, thanks.", "that's all, thanks."])
    assert await caller.next_line("") == "That's all, thanks."
    assert await caller.next_line("Okay.") is None
    assert caller.ended and caller.errors == []


async def test_max_turns_limits_the_call() -> None:
    caller, _ = _caller(None, max_turns=2)  # MockLLM echoes the last message
    assert await caller.next_line("") is not None
    assert await caller.next_line("one") == "You said: one"
    assert await caller.next_line("two") is None and not caller.ended
    assert caller.max_turns == 2
    default, _ = _caller(None)
    assert default.max_turns == 2 + 3  # the script's turns + 3


class _Broken(LLM):
    provider = "broken"

    def __init__(self) -> None:
        super().__init__(model="x")

    def _chat(self, ctx: ChatContext, **_kw: Any) -> LLMStream:
        raise ConnectionError("no server")


async def test_a_failing_caller_llm_ends_the_call_with_an_error() -> None:
    suite = load_tool_suite(MINI)
    caller = LLMCaller(_Broken(), suite, suite.scenarios[0])
    assert await caller.next_line("") is None
    assert caller.errors and "no server" in caller.errors[0] and not caller.ended


async def test_rendered_lines_are_stimuli_of_the_scenario() -> None:
    caller, _ = _caller(None)
    stim = await caller.render(3, "Please cancel order two oh seven seven.")
    assert stim.id == "cancel/llm3" and stim.text == "Please cancel order two oh seven seven."
    assert stim.source == "synthetic" and stim.speech_end > stim.speech_start >= 0


async def test_caller_emulator_takes_a_next_stimulus_function() -> None:
    suite = load_tool_suite(MINI)
    caller, _ = _caller(None)
    stims: list[Stimulus] = [await caller.render(i, "Hello there.") for i in range(2)]
    transport = LoopbackTransport(realtime_playout=True)
    asked: list[int] = []

    async def source(index: int, turns: Any) -> Stimulus | None:
        asked.append(index)
        assert len(turns) == index
        await asyncio.sleep(0.05)  # "thinking": the microphone keeps streaming
        return stims[index] if index < len(stims) else None

    emulator = CallerEmulator(transport, chunk=suite.chunk)
    result = await emulator.run(source, lead_in=0.1, reply_timeout=0.2, gap_after_reply=0.05)
    assert asked == [0, 1, 2]
    assert [t.stimulus.id for t in result.turns] == ["cancel/llm0", "cancel/llm1"]
    assert all(t.missed for t in result.turns)  # nobody answers on this transport
    assert result.turns[1].start - result.turns[0].end >= 0.2


def _reactive_caller(ctx: ChatContext) -> str:
    """A caller model that reacts to what the agent said."""
    heard = (ctx.last_message("user").text if ctx.last_message("user") else "").lower()
    if "cancel" in heard:
        return "Great, thanks for the help."
    if "okay" in heard:
        return END_TOKEN
    return "Hi, please cancel my order two oh seven seven."


def test_the_tools_track_runs_with_an_llm_caller(tmp_path: Path) -> None:
    suite = load_tool_suite(MINI)
    llm = MockLLM(responses=_reactive_caller)
    turns: list[TurnTiming] = []
    results = asyncio.run(
        run_tools_benchmark(
            BenchSystem.from_options(engine="mock", label="reference"),
            suite,
            ToolsOptions(caller="llm:mock", save_audio=False),
            out_dir=tmp_path,
            engine_factory=reference_engine,
            caller_llm=llm,
            on_turn=lambda _s, _t, turn: turns.append(turn),
        )
    )
    s = results.summary
    assert s.rates["pass_at_1"] == 1.0 and s.rates["tool_f1"] == 1.0
    item = results.items[0]
    said = [d["text"] for d in item["turn_details"]]
    assert said == ["Hi, please cancel my order two oh seven seven.", "Great, thanks for the help."]
    lines = item["caller_lines"]
    assert lines[0] == {"agent": "", "caller": said[0]}
    assert "cancel" in (lines[1]["agent"] or "").lower()  # it heard the reference agent
    assert lines[-1]["caller"] is None and len(lines) == 3
    assert [t.stimulus.id for t in turns] == ["cancel/llm0", "cancel/llm1"]
    policy = results.manifest.options["caller_policy"]
    assert policy["type"] == "llm" and policy["llm"]["provider"] == "mock"
    assert policy["temperature"] == 0.0 and policy["seed"] == 0
    assert any("LLM-driven caller" in n for n in results.manifest.notes)
    assert "An LLM plays the caller" in (results.report or "")
    assert not any("turn limit" in n for n in results.manifest.notes)  # it hung up itself


def test_cli_caller_option(tmp_path: Path) -> None:
    from voice_agent_next.cli.main import app

    path = tmp_path / "mini.yaml"
    path.write_text(yaml.safe_dump(MINI), encoding="utf-8")
    env = {"NO_COLOR": "1", "FORCE_COLOR": ""}
    result = CliRunner().invoke(
        app,
        ["bench", "tools", "-s", str(path), "--no-audio", "--json", "--out", str(tmp_path),
         "--run-id", "t6-llm", "--caller", "llm:mock", "--caller-max-turns", "2"],
        env=env,
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout[result.stdout.index("{") :])
    assert summary["track"] == "tools"
    manifest = json.loads((tmp_path / "t6-llm" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["options"]["caller_policy"]["llm"]["provider"] == "mock"
    assert manifest["options"]["caller"] == "synthetic"  # the reference engine: synthetic voice
    bad = CliRunner().invoke(app, ["bench", "tools", "-s", str(path), "--caller", "gpt"], env=env)
    assert bad.exit_code == 2 and "--caller" in bad.output
