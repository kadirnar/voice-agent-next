from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from voice_agent_next import create, list_providers, register_provider
from voice_agent_next.cli.main import app
from voice_agent_next.config import load_config, resolve_callable
from voice_agent_next.errors import ConfigurationError, ProviderNotFoundError
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import MockEngine, MockLLM, MockSTT
from voice_agent_next.registry import get_provider, parse_spec, register_alias

# -------------------------------------------------------------------------- registry


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("deepgram/nova-3", ("deepgram", "nova-3")), ("silero", ("silero", None)),
     ("together/meta-llama/Llama-3.3-70B", ("together", "meta-llama/Llama-3.3-70B")),
     ("ollama/llama3.2:3b", ("ollama", "llama3.2:3b")), ("Faster-Whisper/large-v3", ("faster_whisper", "large-v3"))],
)  # fmt: skip
def test_parse_spec(spec: str, expected: tuple[str, str | None]) -> None:
    assert parse_spec(spec) == expected


def test_parse_spec_rejects_empty() -> None:
    with pytest.raises(ConfigurationError):
        parse_spec("  ")


def test_create_from_string_mapping_and_instance() -> None:
    llm = create("llm", "mock/my-model", ttft=0.5)
    assert isinstance(llm, MockLLM) and llm.model == "my-model" and llm.ttft == 0.5
    stt = create("stt", {"provider": "mock", "model": "m2", "default_text": "yo"})
    assert isinstance(stt, MockSTT) and stt.model == "m2" and stt.default_text == "yo"
    vad = EnergyVAD()
    assert create("vad", vad) is vad
    assert isinstance(create("engine", "mock"), MockEngine)
    with pytest.raises(ConfigurationError):
        create("stt", {"model": "x"})
    with pytest.raises(ConfigurationError):
        create("stt", None)


def test_unknown_provider_error_lists_known() -> None:
    with pytest.raises(ProviderNotFoundError, match="mock"):
        get_provider("stt", "definitely_not_a_provider")


def test_register_custom_provider_and_alias() -> None:
    @register_provider("llm", "unit_test_llm", default_model="tiny", aliases=["utl"])
    class TinyLLM(MockLLM):
        pass

    assert isinstance(create("llm", "unit_test_llm"), TinyLLM)
    assert create("llm", "utl").model == "tiny"
    register_alias("utl2", "unit_test_llm")
    assert isinstance(create("llm", "utl2"), TinyLLM)


def test_list_providers_includes_builtins_and_metadata() -> None:
    specs = list_providers()
    keys = {(s.kind, s.name) for s in specs}
    assert {("engine", "mock"), ("vad", "energy"), ("stt", "mock"), ("tts", "mock")} <= keys
    energy = next(s for s in specs if s.name == "energy")
    assert energy.local and energy.available and energy.missing_env() == []
    assert all(s.supports_platform("linux") for s in specs if s.name in ("mock", "energy"))


# ---------------------------------------------------------------------------- config


def test_load_config_yaml_with_env(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("VAN_TEST_VOICE", "alloy")
    path = tmp_path / "agent.yaml"
    path.write_text(
        "engine: {provider: mock, response_delay: 0.1}\n"
        "agent:\n  instructions: hi\n  voice: ${VAN_TEST_VOICE}\n  greeting: ${MISSING:-hello}\n"
        "transport: {type: loopback}\n"
    )
    cfg = load_config(path)
    assert cfg.agent.voice == "alloy" and cfg.agent.greeting == "hello"
    assert not cfg.is_cascade()


def test_load_config_toml_json_and_validation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    toml = tmp_path / "a.toml"
    toml.write_text('stt = "mock"\nllm = "mock"\ntts = "mock"\nvad = "energy"\n')
    assert load_config(toml).is_cascade()
    js = tmp_path / "a.json"
    js.write_text(json.dumps({"engine": "mock", "llm": "mock"}))
    with pytest.raises(ConfigurationError):
        load_config(js)
    with pytest.raises(ConfigurationError):
        load_config({"stt": "mock"})
    with pytest.raises(ConfigurationError):
        load_config({"engine": "mock", "agent": {"instructions": "${NOT_SET_ANYWHERE_123}"}})
    with pytest.raises(Exception):  # noqa: B017 - pydantic rejects unknown keys
        load_config({"engine": "mock", "bogus": 1})


def test_resolve_callable() -> None:
    assert resolve_callable("json:dumps") is json.dumps
    assert resolve_callable("json.loads") is json.loads
    with pytest.raises(ConfigurationError):
        resolve_callable("json:nope")


# ------------------------------------------------------------------------------- CLI


def test_cli_version_and_providers() -> None:
    runner = CliRunner()
    assert "voice-agent-next" in runner.invoke(app, ["version"]).stdout
    result = runner.invoke(app, ["providers", "--json", "--kind", "engine"])
    assert result.exit_code == 0, result.stdout
    rows = json.loads(result.stdout)
    assert any(r["name"] == "mock" and r["status"] == "ready" for r in rows)
    assert runner.invoke(app, ["providers", "--kind", "bogus"]).exit_code != 0


def test_cli_doctor_runs() -> None:
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "python" in result.stdout
