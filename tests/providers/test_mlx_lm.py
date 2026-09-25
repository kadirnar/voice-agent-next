"""mlx-lm server LLM (``mlx_lm``): an OpenAI-compatible host profile, tested against a fake
server that replays mlx_lm.server's streaming chunks, and the ``apple`` preset's readiness."""

from __future__ import annotations

import pytest

from tests.test_openai_llm import (
    PARALLEL_TOOL_STREAM,
    TEXT_STREAM,
    FakeServer,
    SSEStream,
    get_weather,
    user_ctx,
)
from tests.test_presets import fake_env
from voice_agent_next import create
from voice_agent_next.hardware import AppleSilicon
from voice_agent_next.presets import check_preset, get_preset
from voice_agent_next.providers.mlx_lm import MLXLMServerLLM
from voice_agent_next.registry import get_provider
from voice_agent_next.utils.deps import is_installed

requires_openai = pytest.mark.skipif(not is_installed("openai"), reason="needs the openai extra")


def test_registry_metadata() -> None:
    spec = get_provider("llm", "mlx-lm")
    assert spec.factory is MLXLMServerLLM and spec.local and spec.env == ()
    assert spec.extra == "openai" and spec.requires == ("openai",)
    assert spec.default_model is None  # the class asks for the server's --model
    assert MLXLMServerLLM.DEFAULT_BASE_URL == "http://127.0.0.1:8080/v1"


@requires_openai
async def test_streams_with_thinking_off_and_the_server_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLX_LM_BASE_URL", raising=False)
    server = FakeServer([SSEStream(TEXT_STREAM)])
    llm = create("llm", "mlx_lm", http_client=server.http_client(), max_retries=0)
    text = "".join([c.delta async for c in llm.chat(user_ctx(), max_tokens=32)])
    assert text == "Hello! How can I help?"
    [body] = server.chat_bodies
    assert body["model"] == "default_model"  # mlx_lm.server: the model given with --model
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["max_tokens"] == 32
    assert server.requests[0].url == "http://127.0.0.1:8080/v1/chat/completions"


@requires_openai
async def test_tool_calls_model_and_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLX_LM_BASE_URL", "http://mac-studio:8081/v1")
    server = FakeServer([SSEStream(PARALLEL_TOOL_STREAM)])
    llm = create(
        "llm",
        "mlx_lm/mlx-community/Qwen3.5-4B-4bit",
        http_client=server.http_client(),
        max_retries=0,
        extra={"chat_template_kwargs": None},
    )
    result = await llm.chat(user_ctx("Weather in Paris and Rome?"), tools=[get_weather]).collect()
    assert [c.name for c in result.tool_calls] == ["get_weather", "get_weather"]
    [body] = server.chat_bodies
    assert body["model"] == "mlx-community/Qwen3.5-4B-4bit"
    assert "chat_template_kwargs" not in body  # None removes the default
    assert body["tools"][0]["function"]["name"] == "get_weather"
    assert str(server.requests[0].url).startswith("http://mac-studio:8081/v1/")


# ---------------------------------------------------------------------- preset
MAC = {"platform": "darwin", "apple": AppleSilicon("Apple M4 Pro")}


def test_apple_preset_is_mlx() -> None:
    preset = get_preset("apple")
    assert preset.config["stt"] == "mlx/parakeet-tdt-0.6b-v3"
    assert preset.config["tts"] == "mlx_audio/kokoro"
    assert preset.config["llm"][0] == "mlx_lm/mlx-community/Qwen3.5-4B-4bit"
    assert set(preset.extras) >= {"mlx", "openai"}


def test_apple_preset_prefers_a_running_mlx_lm_server() -> None:
    ready = check_preset("apple", env=fake_env(**MAC, mlx_lm=["mlx-community/Qwen3.5-4B-4bit"]))
    assert ready.ready and ready.config["llm"] == ["mlx_lm/mlx-community/Qwen3.5-4B-4bit",
                                                   "ollama/qwen3.5:4b"]  # fmt: skip


def test_apple_preset_falls_back_to_ollama_without_the_server() -> None:
    result = check_preset("apple", env=fake_env(**MAC))
    assert result.ready and result.config["llm"] == "ollama/qwen3.5:4b"
    assert any("no mlx-lm server at http://127.0.0.1:8080/v1" in n for n in result.notes)


def test_apple_preset_without_any_llm_server() -> None:
    result = check_preset("apple", env=fake_env(**MAC, ollama=None))
    assert not result.ready
    assert result.problems[0].message == "no mlx-lm server at http://127.0.0.1:8080/v1"
    assert "python -m mlx_lm.server --model mlx-community/Qwen3.5-4B-4bit" in result.fixes()[0]


def test_mlx_lm_server_url_from_options_and_env() -> None:
    probed: list[str] = []
    env = fake_env(**MAC, environ={"MLX_LM_BASE_URL": "http://studio:9000/v1"})
    env.mlx_lm_models = lambda url: probed.append(url) or None  # type: ignore[func-returns-value]
    check_preset("apple", env=env)
    check_preset("apple", env=env, overrides={"llm": {"provider": "mlx_lm",
                                                      "base_url": "http://other:1/v1"}})  # fmt: skip
    assert probed == ["http://studio:9000/v1", "http://other:1/v1"]


def test_apple_preset_needs_the_mlx_extra() -> None:
    result = check_preset("apple", env=fake_env(**MAC, missing=["parakeet_mlx", "mlx_audio"]))
    assert "pip install 'voice-agent-next[mlx]'" in result.fixes()
