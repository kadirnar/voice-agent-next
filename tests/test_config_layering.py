"""Config layering: preset < config file < flags, validated once; ``${ENV}`` before
``extends:``; one config-file reader (#141)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.test_presets import cli, fake_env  # noqa: F401 - `cli` is a fixture
from voice_agent_next import config as config_mod
from voice_agent_next.config import layer_config, load_config, read_config_file
from voice_agent_next.errors import ConfigurationError
from voice_agent_next.presets import PRESETS

LOCAL_CPU = PRESETS["local-cpu"].config


# --------------------------------------------------------------- read_config_file
@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("a.yaml", "llm: mock\nagent: {greeting: Hi}\n"),
        ("a.yml", "llm: mock\nagent: {greeting: Hi}\n"),
        ("a.toml", 'llm = "mock"\n[agent]\ngreeting = "Hi"\n'),
        ("a.json", '{"llm": "mock", "agent": {"greeting": "Hi"}}'),
    ],
)
def test_read_config_file_formats(tmp_path: Path, name: str, text: str) -> None:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    assert read_config_file(path) == {"llm": "mock", "agent": {"greeting": "Hi"}}


def test_read_config_file_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        read_config_file(tmp_path / "missing.yaml")
    bad = tmp_path / "a.ini"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unsupported config format"):
        read_config_file(bad)
    listed = tmp_path / "a.yaml"
    listed.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="must be a mapping"):
        read_config_file(listed)
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert read_config_file(empty) == {}


def test_bench_system_uses_the_shared_reader() -> None:
    from voice_agent_next.bench import system

    assert not hasattr(system, "_read_config_file")


# -------------------------------------------------------------- ${ENV} + extends
def test_env_is_expanded_before_extends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VAN_TEST_PRESET", "local-cpu")
    path = tmp_path / "agent.yaml"
    path.write_text("extends: ${VAN_TEST_PRESET}\nagent: {greeting: Hi}\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.extends == "local-cpu" and cfg.llm == LOCAL_CPU["llm"]
    monkeypatch.delenv("VAN_TEST_PRESET")
    defaulted = load_config({"extends": "${VAN_TEST_PRESET:-local-cpu}"})
    assert defaulted.extends == "local-cpu" and defaulted.stt == LOCAL_CPU["stt"]


def test_env_values_are_expanded_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A value that itself looks like ``${...}`` is not expanded a second time."""
    monkeypatch.setenv("VAN_TEST_TEXT", "${NOT_A_VAR}")
    cfg = load_config({"llm": "mock", "agent": {"instructions": "${VAN_TEST_TEXT}"}})
    assert cfg.agent.instructions == "${NOT_A_VAR}"


# ------------------------------------------------------------------- layer_config
def test_layer_config_orders_preset_file_flags(tmp_path: Path) -> None:
    path = tmp_path / "tweaks.yaml"
    path.write_text("stt: {language: en}\nllm: ollama/qwen3.5:4b\n", encoding="utf-8")
    raw = layer_config(preset="local-cpu", file=path)
    assert raw["extends"] == "local-cpu"
    assert raw["stt"] == {"provider": LOCAL_CPU["stt"], "language": "en"}  # a tweak
    assert raw["llm"] == "ollama/qwen3.5:4b" and raw["tts"] == LOCAL_CPU["tts"]
    flags = layer_config(preset="local-cpu", file=path, overrides={"llm": "mock"})
    assert flags["llm"] == "mock"  # flags win over the file
    # the file's own `extends:` is the preset when --preset is not given
    path.write_text("extends: local-cpu\nstt: {language: en}\n", encoding="utf-8")
    assert layer_config(file=path)["stt"] == raw["stt"]
    assert layer_config(preset="local-cpu", file=path)["extends"] == "local-cpu"
    with pytest.raises(ConfigurationError, match="conflicts with `extends: local-cpu`"):
        layer_config(preset="cloud-fast", file=path)
    assert layer_config() == {}


def test_layer_config_is_not_validated_until_the_end(tmp_path: Path) -> None:
    """A tweaks file is not a complete config on its own (the old code rejected it)."""
    path = tmp_path / "tweaks.yaml"
    path.write_text("agent: {greeting: Hi}\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(path)  # no engine or LLM of its own
    raw = layer_config(preset="local-cpu", file=path)
    cfg = config_mod.AppConfig.model_validate(raw)
    cfg.validate_components()
    assert cfg.agent.greeting == "Hi" and cfg.llm == LOCAL_CPU["llm"]


# --------------------------------------------------------------------- van run
def test_cli_run_preset_with_a_tweaks_file(cli: Callable[..., Any], tmp_path: Path) -> None:  # noqa: F811
    tweaks = tmp_path / "tweaks.yaml"
    tweaks.write_text("stt: {language: en}\nagent: {greeting: Hello}\n", encoding="utf-8")
    code, out, ran = cli("run", "--preset", "local-cpu", "-c", str(tweaks))
    assert code == 0, out
    cfg = ran[-1]
    assert cfg.extends == "local-cpu" and cfg.agent.greeting == "Hello"
    assert cfg.stt == {"provider": LOCAL_CPU["stt"], "language": "en"}
    assert cfg.llm == LOCAL_CPU["llm"]
    # flags win over the file
    code, out, ran = cli("run", "--preset", "local-cpu", "-c", str(tweaks), "--tts", "mock")
    assert code == 0, out
    assert ran[-1].tts == "mock" and ran[-1].agent.greeting == "Hello"
    # a bad tweak is reported, not a traceback
    bad = tmp_path / "bad.yaml"
    bad.write_text("nonsense: 1\n", encoding="utf-8")
    code, out, _ = cli("run", "--preset", "local-cpu", "-c", str(bad))
    assert code == 2 and "nonsense" in out


# ------------------------------------------------------------------ van bench
def test_bench_system_layers_a_preset_under_the_file(tmp_path: Path) -> None:
    from voice_agent_next.bench.system import BenchSystem

    tweaks = tmp_path / "tweaks.yaml"
    tweaks.write_text("stt: {language: en}\n", encoding="utf-8")
    system = BenchSystem.from_options(preset="local-cpu", config=tweaks, llm="mock")
    cfg = system.config
    assert cfg.extends == "local-cpu" and cfg.llm == "mock"
    assert cfg.stt == {"provider": LOCAL_CPU["stt"], "language": "en"}
    with pytest.raises(ConfigurationError, match="conflicts"):
        BenchSystem.from_options(preset="cloud-fast", config={"extends": "local-cpu"})


def test_bench_system_expands_env_in_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from voice_agent_next.bench.system import BenchSystem

    monkeypatch.setenv("VAN_TEST_PRESET", "local-cpu")
    path = tmp_path / "bench.yaml"
    path.write_text("extends: ${VAN_TEST_PRESET}\n", encoding="utf-8")
    assert BenchSystem.from_options(config=path).config.extends == "local-cpu"


# ------------------------------------------------------------------ van serve
def test_serve_layers_preset_file_and_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voice_agent_next import presets
    from voice_agent_next.cli.serve import SourceOptions, build_app_config

    machine = fake_env()
    monkeypatch.setattr(presets, "current_environment", lambda: machine)

    tweaks = tmp_path / "tweaks.yaml"
    tweaks.write_text("agent: {greeting: Hello}\nstt: {language: en}\n", encoding="utf-8")
    cfg = build_app_config(SourceOptions(preset="local-cpu", config=str(tweaks)))
    assert cfg.extends == "local-cpu" and cfg.agent.greeting == "Hello"
    assert cfg.llm == LOCAL_CPU["llm"]
    assert cfg.stt == {"provider": LOCAL_CPU["stt"], "language": "en"}
    # flags on top of the preset and the file (#156): preset < file < flags
    flags = build_app_config(SourceOptions(preset="local-cpu", config=str(tweaks), llm="mock"))
    assert flags.extends == "local-cpu" and flags.llm == "mock" and flags.agent.greeting == "Hello"
    assert flags.stt == {"provider": LOCAL_CPU["stt"], "language": "en"}
    engine = build_app_config(SourceOptions(preset="local-cpu", engines=["mock"]))
    assert engine.engine == "mock" and engine.is_cascade() is False and engine.stt is None
    for bad in (
        SourceOptions(engines=["mock", "mock"]),
        SourceOptions(preset="local-cpu", engines=["mock"], llm="mock"),  # engine or cascade
        SourceOptions(config=str(tweaks), engines=[str(tweaks)]),  # two files
    ):
        with pytest.raises(ConfigurationError, match="one agent"):
            build_app_config(bad)
