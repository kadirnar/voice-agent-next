"""Presets: resolution, `extends:` merging, readiness with a faked machine, and the CLI."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from voice_agent_next import presets
from voice_agent_next.cli import main as cli_main
from voice_agent_next.config import AppConfig, load_config, merge_config
from voice_agent_next.errors import ConfigurationError
from voice_agent_next.hardware import AppleSilicon, Backend
from voice_agent_next.presets import (
    AUTO_ORDER,
    PRESETS,
    Environment,
    check_config,
    check_preset,
    get_preset,
    load_preset,
    pick_preset,
)
from voice_agent_next.registry import get_provider

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LOCAL_MODELS = ("qwen3.5:4b", "qwen3.5:9b", "LiquidAI/lfm2.5-1.2b-instruct:latest")


def fake_env(
    *,
    platform: str = "linux",
    environ: dict[str, str] | None = None,
    missing: Iterable[str] = (),
    gpus: Sequence[str] = (),
    apple: AppleSilicon | None = None,
    backend: Backend | None = None,
    ollama: Sequence[str] | None = LOCAL_MODELS,
    mlx_lm: Sequence[str] | None = None,
) -> Environment:
    """A machine with every module installed except ``missing``."""
    absent = set(missing)
    urls: list[str] = []

    def ollama_models(url: str) -> Sequence[str] | None:
        urls.append(url)
        return ollama

    env = Environment(
        platform=platform,
        environ=environ or {},
        installed=lambda module: module not in absent,
        nvidia_gpus=lambda: tuple(gpus),
        apple_silicon=lambda: apple,
        cuda_backend=lambda: backend,
        ollama_models=ollama_models,
        mlx_lm_models=lambda url: mlx_lm,
    )
    env.probed_urls = urls  # type: ignore[attr-defined]
    return env


CLOUD_KEYS = {
    "DEEPGRAM_API_KEY": "k",
    "GROQ_API_KEY": "k",
    "CEREBRAS_API_KEY": "k",
    "CARTESIA_API_KEY": "k",
    "ASSEMBLYAI_API_KEY": "k",
    "ANTHROPIC_API_KEY": "k",
    "OPENAI_API_KEY": "k",
    "ELEVENLABS_API_KEY": "k",
    "GOOGLE_API_KEY": "k",
}


# ------------------------------------------------------------------------- catalog
def test_catalog_is_consistent() -> None:
    assert set(AUTO_ORDER) == set(PRESETS)
    assert [p.name for p in presets.list_presets()] == list(AUTO_ORDER)
    for preset in PRESETS.values():
        cfg = preset.app_config()
        cfg.validate_components()
        assert cfg.extends == preset.name
        assert preset.summary and preset.rationale
        for key, member in preset.components():  # every spec resolves in the registry
            kind = {"turn_detector": "turn"}.get(key, key)
            spec = get_provider(kind, presets._split(member)[0])  # type: ignore[arg-type]
            assert spec.supports_platform(preset.platforms[0])


def test_required_presets_exist() -> None:
    for name in ("local-cpu", "local-gpu", "apple", "cloud-fast", "cloud-quality"):
        assert PRESETS[name].app_config().is_cascade()
    assert PRESETS["openai-realtime"].config == {"engine": "openai/gpt-realtime-2.1"}
    assert PRESETS["gemini-live"].config == {"engine": "gemini/gemini-3.8-live"}
    assert PRESETS["local-gpu"].accelerator == "cuda"
    assert PRESETS["apple"].platforms == ("darwin",)


def test_preset_metadata() -> None:
    local = get_preset("local-cpu")
    assert local.config["stt"] == "sherpa-onnx/zipformer-en-kroko"  # measured best on CPU
    assert set(local.extras) == {"sherpa-onnx", "openai", "kokoro", "silero", "smart-turn"}
    assert local.env_vars == ()
    fast = get_preset("cloud-fast")
    assert ("DEEPGRAM_API_KEY",) in fast.env_vars and ("CARTESIA_API_KEY",) in fast.env_vars
    assert fast.stack() == (
        "deepgram/flux-general-en > groq/openai/gpt-oss-120b (+1 failover) > cartesia/sonic-3.6"
    )


def test_get_preset_normalizes_and_suggests() -> None:
    assert get_preset("Local_CPU").name == "local-cpu"
    with pytest.raises(ConfigurationError, match="Did you mean 'local-cpu'"):
        get_preset("loca-cpu")


# ------------------------------------------------------------------------ merging
def test_merge_config_rules() -> None:
    base = {
        "stt": "sherpa-onnx/zipformer-en-kroko",
        "llm": ["groq/x", "cerebras/y"],
        "tts": {"provider": "kokoro", "voice": "af_heart"},
        "vad": "silero",
        "agent": {"instructions": "hi", "greeting": "hello"},
        "transport": {"type": "local", "input_device": 3},
    }
    out = merge_config(
        base,
        {
            "stt": {"language": "en"},  # options only: tweak the base STT
            "tts": {"voice": "am_adam"},
            "llm": "ollama/qwen3.5:4b",  # replaced
            "vad": None,  # removed
            "agent": {"instructions": "be brief"},  # sections merge
            "transport": {"input_device": 5},
        },
    )
    assert out["stt"] == {"provider": "sherpa-onnx/zipformer-en-kroko", "language": "en"}
    assert out["tts"] == {"provider": "kokoro", "voice": "am_adam"}
    assert out["llm"] == "ollama/qwen3.5:4b"
    assert out["vad"] is None
    assert out["agent"] == {"instructions": "be brief", "greeting": "hello"}
    assert out["transport"] == {"type": "local", "input_device": 5}
    assert base["tts"] == {"provider": "kokoro", "voice": "af_heart"}  # base untouched
    # another transport type replaces the whole section
    assert merge_config(base, {"transport": {"type": "file"}})["transport"] == {"type": "file"}
    # engine and cascade exclude each other
    engine = merge_config(base, {"engine": "mock"})
    assert engine["engine"] == "mock" and "stt" not in engine and "llm" not in engine
    assert "engine" not in merge_config({"engine": "mock"}, {"llm": "x", "tts": "y"})
    with pytest.raises(ConfigurationError, match="failover list"):
        merge_config(base, {"llm": {"temperature": 0}})
    with pytest.raises(ConfigurationError, match="provider"):
        merge_config({}, {"stt": {"language": "en"}})


def test_extends_in_yaml_and_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_PROMPT", "You are a pirate.")
    yaml_file = tmp_path / "agent.yaml"
    yaml_file.write_text(
        "extends: local-cpu\n"
        "llm: ollama/qwen3.5:4b\n"
        "stt: {language: en}\n"
        "agent: {instructions: '${MY_PROMPT}'}\n",
        encoding="utf-8",
    )
    cfg = load_config(yaml_file)
    assert cfg.extends == "local-cpu"
    assert cfg.llm == "ollama/qwen3.5:4b"
    assert cfg.stt == {"provider": "sherpa-onnx/zipformer-en-kroko", "language": "en"}
    assert cfg.tts == "kokoro/v1.0-fp16" and cfg.turn_detector == "smart_turn"
    assert cfg.agent.instructions == "You are a pirate."

    toml_file = tmp_path / "agent.toml"
    toml_file.write_text('extends = "cloud-fast"\n[agent]\ngreeting = "Hi"\n', encoding="utf-8")
    cfg = load_config(toml_file)
    assert cfg.stt == "deepgram/flux-general-en" and cfg.agent.greeting == "Hi"

    native = load_config({"extends": "openai-realtime", "session": {"allow_interruptions": False}})
    assert (
        native.engine == "openai/gpt-realtime-2.1"
        and native.session["allow_interruptions"] is False
    )
    switched = load_config({"extends": "local-cpu", "engine": "mock"})
    assert switched.engine == "mock" and switched.stt is None and switched.llm is None


def test_extends_errors() -> None:
    with pytest.raises(ConfigurationError, match="unknown preset"):
        load_config({"extends": "nope"})
    with pytest.raises(ConfigurationError, match="preset name"):
        load_config({"extends": ["local-cpu"]})
    assert load_config({"engine": "mock"}).extends is None  # plain configs are unchanged


# ---------------------------------------------------------------------- readiness
def test_local_cpu_ready_on_a_complete_machine() -> None:
    env = fake_env()
    result = check_preset("local-cpu", env=env)
    assert result.ready, result.explain()
    assert result.summary() == "ready" and result.fixes() == []
    assert env.probed_urls == ["http://127.0.0.1:11434/v1"]  # type: ignore[attr-defined]
    assert result.config["llm"] == "ollama/LiquidAI/lfm2.5-1.2b-instruct"


def test_missing_extras_become_one_install_command() -> None:
    env = fake_env(missing=("sherpa_onnx", "kokoro_onnx", "sounddevice"))
    result = check_preset("local-cpu", env=env)
    assert not result.ready
    assert result.extras == ("sherpa-onnx", "kokoro", "audio")
    assert result.fixes() == ["pip install 'voice-agent-next[sherpa-onnx,kokoro,audio]'"]
    assert "stt: sherpa-onnx/zipformer-en-kroko needs the 'sherpa-onnx' extra" in result.explain()
    # without the local audio transport the audio extra is not needed
    assert check_preset("local-cpu", env=env, transport=None).extras == ("sherpa-onnx", "kokoro")


def test_ollama_server_and_model_checks() -> None:
    down = check_preset("local-cpu", env=fake_env(ollama=None))
    assert not down.ready
    assert any("ollama serve" in fix for fix in down.fixes())
    assert "ollama pull LiquidAI/lfm2.5-1.2b-instruct" in down.fixes()

    no_model = check_preset("local-gpu", env=fake_env(ollama=["qwen3.5:4b"], gpus=["RTX"]))
    assert [p.message for p in no_model.problems] == ["Ollama has no model qwen3.5:9b"]
    assert no_model.fixes() == ["ollama pull qwen3.5:9b"]

    # ":latest" is implied, names are case-insensitive
    ok = fake_env(ollama=["liquidai/LFM2.5-1.2b-instruct:latest"])
    assert check_preset("local-cpu", env=ok).ready


def test_ollama_url_follows_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    env = fake_env(environ={"OLLAMA_HOST": "0.0.0.0:9999"})
    assert check_preset("local-cpu", env=env).ready
    assert env.probed_urls == ["http://127.0.0.1:9999/v1"]  # type: ignore[attr-defined]
    env = fake_env()
    check_config({"llm": {"provider": "ollama/qwen3.5:4b", "base_url": "http://h:1/v1"}}, env=env)
    assert env.probed_urls == ["http://h:1/v1"]  # type: ignore[attr-defined]


def test_cloud_keys_and_failover_pruning() -> None:
    none = check_preset("cloud-fast", env=fake_env())
    assert not none.ready
    assert "set DEEPGRAM_API_KEY (e.g. `export DEEPGRAM_API_KEY=...`)" in none.fixes()
    assert any("fixing any one of" in n for n in none.notes)

    keys = {"DEEPGRAM_API_KEY": "k", "CEREBRAS_API_KEY": "k", "CARTESIA_API_KEY": "k"}
    result = check_preset("cloud-fast", env=fake_env(environ=keys))
    assert result.ready, result.explain()
    assert result.config["llm"] == "cerebras/gpt-oss-120b"  # groq skipped: no key
    assert any("skipping groq/openai/gpt-oss-120b" in n for n in result.notes)

    both = check_preset("cloud-fast", env=fake_env(environ={**keys, "GROQ_API_KEY": "k"}))
    assert both.config["llm"] == ["groq/openai/gpt-oss-120b", "cerebras/gpt-oss-120b"]

    # the mapping form keeps its chain options
    chain = {"fallback": ["groq/a", "cerebras/b"], "cooldown": 5}
    pruned = check_config({"llm": chain, "tts": "mock"}, env=fake_env(environ=keys))
    assert pruned.config["llm"] == "cerebras/b"
    kept = check_config(
        {"llm": chain, "tts": "mock"}, env=fake_env(environ={**keys, "GROQ_API_KEY": "k"})
    )
    assert kept.config["llm"] == chain


def test_api_key_in_options_counts() -> None:
    result = check_config({"engine": {"provider": "openai", "api_key": "sk"}}, env=fake_env())
    assert result.ready


def test_platform_and_apple_silicon() -> None:
    linux = check_preset("apple", env=fake_env())
    assert [p.message for p in linux.problems] == ["needs macOS (this is linux)"]
    assert linux.fixes() == []

    intel = check_preset("apple", env=fake_env(platform="darwin"))
    assert intel.problems[0].message == "needs an Apple silicon Mac"
    rosetta = check_preset(
        "apple", env=fake_env(platform="darwin", apple=AppleSilicon("Apple M3", rosetta=True))
    )
    assert "Rosetta" in rosetta.problems[0].message and rosetta.problems[0].fix
    mac = check_preset("apple", env=fake_env(platform="darwin", apple=AppleSilicon("Apple M3")))
    assert mac.ready and "Apple silicon: Apple M3" in mac.notes


def test_local_gpu_needs_a_usable_cuda_gpu() -> None:
    no_gpu = check_preset("local-gpu", env=fake_env())
    assert no_gpu.problems[0].message == "no NVIDIA GPU found"
    assert check_preset("local-gpu", env=fake_env(platform="darwin")).problems[0].component == (
        "platform"
    )

    cpu = Backend("cpu", "int8", reason="RTX found but cuBLAS 12 is missing", fix="pip x[cuda]")
    no_libs = check_preset("local-gpu", env=fake_env(gpus=["RTX 5070 Ti"], backend=cpu))
    assert not no_libs.ready
    assert "faster-whisper would run on the CPU: RTX found" in no_libs.problems[0].message
    assert no_libs.fixes() == ["pip x[cuda]"]

    cuda = Backend("cuda", "float16", reason="RTX 5070 Ti")
    ready = check_preset("local-gpu", env=fake_env(gpus=["RTX 5070 Ti"], backend=cuda))
    assert ready.ready and "GPU: RTX 5070 Ti" in ready.notes
    # CTranslate2 not installed: the extra is the fix, not a GPU problem
    missing = check_preset("local-gpu", env=fake_env(gpus=["RTX"], missing=["faster_whisper"]))
    assert missing.fixes() == ["pip install 'voice-agent-next[faster-whisper]'"]


def test_unknown_provider_is_a_problem() -> None:
    result = check_config({"llm": "no-such-provider/x", "tts": "mock"}, env=fake_env())
    assert not result.ready and "no llm provider named" in result.problems[0].message


def test_pick_preset_order() -> None:
    cuda = Backend("cuda", "float16", reason="RTX")
    picked, _ = pick_preset(env=fake_env(gpus=["RTX"], backend=cuda))
    assert picked is not None and picked.name == "local-gpu"
    picked, _ = pick_preset(env=fake_env())
    assert picked is not None and picked.name == "local-cpu"
    mac = fake_env(platform="darwin", apple=AppleSilicon("Apple M4"))
    assert pick_preset(env=mac)[0].name == "apple"  # type: ignore[union-attr]
    # nothing local installed: the first cloud preset whose keys are set
    cloud_only = ("sherpa_onnx", "faster_whisper", "kokoro_onnx", "onnxruntime")
    env = fake_env(missing=cloud_only, environ={"OPENAI_API_KEY": "k"})
    picked, checked = pick_preset(env=env)
    assert picked is not None and picked.name == "openai-realtime"
    assert [r.name for r in checked] == list(AUTO_ORDER[: AUTO_ORDER.index("openai-realtime") + 1])
    assert pick_preset(env=fake_env(missing=cloud_only))[0] is None


def test_load_preset() -> None:
    env = fake_env(environ={"DEEPGRAM_API_KEY": "k", "GROQ_API_KEY": "k", "CARTESIA_API_KEY": "k"})
    cfg = load_preset("cloud-fast", env=env, agent={"instructions": "Be brief."})
    assert isinstance(cfg, AppConfig) and cfg.extends == "cloud-fast"
    assert cfg.llm == "groq/openai/gpt-oss-120b" and cfg.agent.instructions == "Be brief."
    with pytest.raises(ConfigurationError, match=r"to fix:\n  1\. set DEEPGRAM_API_KEY"):
        load_preset("cloud-fast", env=fake_env())
    raw = load_preset("cloud-fast", check=False, llm="mock")
    assert raw.llm == "mock" and raw.stt == "deepgram/flux-general-en"


def test_session_from_preset() -> None:
    session, agent = presets.session_from_preset(
        "openai-realtime", env=fake_env(environ={"OPENAI_API_KEY": "k"}), engine="mock"
    )
    assert agent.instructions
    assert type(session).__name__ == "AgentSession"


# ---------------------------------------------------------------------------- CLI
@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch) -> Callable[..., tuple[int, str, list[AppConfig]]]:
    """Run ``van`` on a fake machine; ``van run`` records its config instead of running."""
    ran: list[AppConfig] = []
    monkeypatch.setattr(cli_main, "_run_session", ran.append)

    def invoke(*args: str, env: Environment | None = None) -> tuple[int, str, list[AppConfig]]:
        machine = env or fake_env()
        monkeypatch.setattr(presets, "current_environment", lambda: machine)
        result = CliRunner().invoke(cli_main.app, list(args), env={"COLUMNS": "250"})
        return result.exit_code, ANSI.sub("", result.output), ran

    return invoke


def test_cli_presets_table_and_json(cli: Callable[..., Any]) -> None:
    code, out, _ = cli("presets", env=fake_env(missing=["sounddevice"]))
    assert code == 0, out
    for name in AUTO_ORDER:
        assert name in out
    assert "needs the 'audio' extra" in out

    code, out, _ = cli("presets")
    assert code == 0 and "`van run` picks local-cpu" in out

    code, out, _ = cli("presets", "--json", env=fake_env(environ={"OPENAI_API_KEY": "k"}))
    rows = {row["name"]: row for row in json.loads(out)}
    assert rows["openai-realtime"]["ready"] is True
    assert rows["cloud-fast"]["ready"] is False
    assert (
        "set DEEPGRAM_API_KEY (e.g. `export DEEPGRAM_API_KEY=...`)" in rows["cloud-fast"]["fixes"]
    )
    assert rows["local-gpu"]["problems"][0]["component"] == "gpu"
    assert rows["local-cpu"]["extras"] == [
        "sherpa-onnx",
        "openai",
        "kokoro",
        "silero",
        "smart-turn",
    ]


def test_cli_presets_detail(cli: Callable[..., Any]) -> None:
    code, out, _ = cli("presets", "cloud-fast")
    assert code == 0, out
    assert "extends: cloud-fast" in out and "stt: deepgram/flux-general-en" in out
    assert "to fix:" in out and "DEEPGRAM_API_KEY" in out
    code, out, _ = cli("presets", "cloud-fast", "--json")
    assert json.loads(out)["name"] == "cloud-fast"
    code, out, _ = cli("presets", "clod-fast")
    assert code == 2 and "Did you mean 'cloud-fast'" in out


def test_cli_run_preset_validates(cli: Callable[..., Any]) -> None:
    code, out, ran = cli("run", "--preset", "cloud-fast")
    assert code == 1 and not ran
    assert "cloud-fast is not ready" in out and "1. set DEEPGRAM_API_KEY" in out
    assert "--skip-checks" in out

    code, out, ran = cli("run", "--preset", "cloud-fast", "--skip-checks")
    assert code == 0, out
    assert ran[-1].llm == ["groq/openai/gpt-oss-120b", "cerebras/gpt-oss-120b"]

    code, out, _ = cli("run", "--preset", "nope")
    assert code == 2 and "unknown preset" in out


def test_cli_run_preset_with_overrides(cli: Callable[..., Any]) -> None:
    code, out, ran = cli(
        "run", "--preset", "local-cpu", "--llm", "ollama/qwen3.5:4b", "--instructions", "Hi."
    )
    assert code == 0, out
    cfg = ran[-1]
    assert cfg.extends == "local-cpu" and cfg.llm == "ollama/qwen3.5:4b"
    assert cfg.stt == "sherpa-onnx/zipformer-en-kroko" and cfg.agent.instructions == "Hi."
    assert cfg.transport == {"type": "local"}

    # an override that is not ready fails the check (qwen3.5:27b is not pulled)
    code, out, _ = cli("run", "--preset", "local-cpu", "--llm", "ollama/qwen3.5:27b")
    assert code == 1 and "ollama pull qwen3.5:27b" in out

    # a native engine preset turned into a cascade by flags
    code, out, ran = cli(
        "run", "--preset", "openai-realtime", "--llm", "mock", "--tts", "mock",
        env=fake_env(environ={"OPENAI_API_KEY": "k"}),
    )  # fmt: skip
    assert code == 0, out
    assert ran[-1].engine is None and ran[-1].llm == "mock"


def test_cli_run_config_extends(cli: Callable[..., Any], tmp_path: Path) -> None:
    config = tmp_path / "agent.yaml"
    config.write_text("extends: cloud-fast\nagent: {greeting: Hello}\n", encoding="utf-8")
    code, out, _ = cli("run", "--config", str(config))
    assert code == 1 and "cloud-fast is not ready" in out
    env = fake_env(environ=CLOUD_KEYS)
    code, out, ran = cli("run", "--config", str(config), env=env)
    assert code == 0, out
    assert ran[-1].agent.greeting == "Hello" and ran[-1].extends == "cloud-fast"
    code, out, _ = cli("run", "--config", str(config), "--preset", "local-cpu", env=env)
    assert code == 2 and "conflicts with `extends: cloud-fast`" in out

    plain = tmp_path / "plain.yaml"  # no preset: no readiness check, as before
    plain.write_text("llm: no-such/x\ntts: mock\n", encoding="utf-8")
    code, out, ran = cli("run", "--config", str(plain), env=fake_env(ollama=None))
    assert code == 0, out
    assert ran[-1].llm == "no-such/x" and ran[-1].extends is None


def test_cli_run_auto_picks_a_ready_preset(cli: Callable[..., Any]) -> None:
    code, out, ran = cli("run")
    assert code == 0, out
    assert "using preset local-cpu" in out and "--preset" in out
    assert ran[-1].extends == "local-cpu" and ran[-1].stt == "sherpa-onnx/zipformer-en-kroko"

    nothing = fake_env(missing=["sherpa_onnx", "faster_whisper", "kokoro_onnx", "onnxruntime"])
    code, out, ran = cli("run", env=nothing)
    assert code == 0, out
    assert "no preset is ready" in out and ran[-1].engine == "mock"

    # explicit components: no auto-pick (the previous behavior)
    code, out, ran = cli("run", "--engine", "mock", env=nothing)
    assert code == 0 and "preset" not in out and ran[-1].engine == "mock"
