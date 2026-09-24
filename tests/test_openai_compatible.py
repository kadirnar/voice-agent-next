"""Preconfigured OpenAI-compatible hosts: registry metadata, base URLs and API keys."""

from __future__ import annotations

import importlib

import pytest

from voice_agent_next.errors import ConfigurationError
from voice_agent_next.providers.ollama import ollama_base_url
from voice_agent_next.registry import create, get_provider
from voice_agent_next.utils.deps import is_installed

requires_openai = pytest.mark.skipif(not is_installed("openai"), reason="needs the openai extra")

# name -> (class, base URL, API key env var, key required, default model, local)
HOSTS: dict[str, tuple[str, str, str, bool, str | None, bool]] = {
    "ollama": ("OllamaLLM", "http://127.0.0.1:11434/v1", "OLLAMA_API_KEY", False, "qwen3.5:4b", True),
    "llamacpp": ("LlamaCppLLM", "http://127.0.0.1:8080/v1", "LLAMA_API_KEY", False, None, True),
    "vllm": ("VllmLLM", "http://127.0.0.1:8000/v1", "VLLM_API_KEY", False, None, True),
    "lmstudio": ("LMStudioLLM", "http://127.0.0.1:1234/v1", "LM_API_TOKEN", False, None, True),
    "groq": ("GroqLLM", "https://api.groq.com/openai/v1", "GROQ_API_KEY", True, "llama-3.3-70b-versatile", False),
    "cerebras": ("CerebrasLLM", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", True, "gpt-oss-120b", False),
    "together": ("TogetherLLM", "https://api.together.ai/v1", "TOGETHER_API_KEY", True, "meta-llama/Llama-3.3-70B-Instruct-Turbo", False),
    "openrouter": ("OpenRouterLLM", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", True, "openai/gpt-4.1-mini", False),
    "deepseek": ("DeepSeekLLM", "https://api.deepseek.com", "DEEPSEEK_API_KEY", True, "deepseek-flash", False),
    "fireworks": ("FireworksLLM", "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY", True, "accounts/fireworks/models/llama-v3p3-70b-instruct", False),
    "sambanova": ("SambaNovaLLM", "https://api.sambanova.ai/v1", "SAMBANOVA_API_KEY", True, "Meta-Llama-3.3-70B-Instruct", False),
}  # fmt: skip


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in HOSTS:
        cls = getattr(importlib.import_module(f"voice_agent_next.providers.{name}"), HOSTS[name][0])
        for var in (*cls.API_KEY_ENV, *cls.BASE_URL_ENV):
            monkeypatch.delenv(var, raising=False)
    for var in ("OLLAMA_HOST", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.mark.parametrize("name", sorted(HOSTS))
def test_registry_metadata(name: str) -> None:
    cls_name, base_url, key_env, key_required, default_model, local = HOSTS[name]
    spec = get_provider("llm", name)
    cls = spec.factory
    assert cls.__name__ == cls_name and cls.provider == name
    assert base_url == cls.DEFAULT_BASE_URL
    assert (key_env,) == cls.API_KEY_ENV and cls.API_KEY_REQUIRED is key_required
    assert spec.default_model == cls.DEFAULT_MODEL == default_model
    assert spec.local is local
    assert spec.extra == "openai" and spec.requires == ("openai",)
    # local servers work without a key, so `van providers` must not ask for one
    assert spec.env == (() if local else (key_env,))


def test_openai_registry_metadata() -> None:
    spec = get_provider("llm", "openai")
    assert spec.default_model == "gpt-4.1-mini" and spec.env == ("OPENAI_API_KEY",)
    assert spec.extra == "openai" and spec.requires == ("openai",) and not spec.local


@pytest.mark.parametrize(
    ("host", "url"),
    [
        ("127.0.0.1", "http://127.0.0.1:11434/v1"),
        ("0.0.0.0:11434", "http://127.0.0.1:11434/v1"),
        ("0.0.0.0", "http://127.0.0.1:11434/v1"),
        (":11500", "http://127.0.0.1:11500/v1"),
        ("gpu-box:8080", "http://gpu-box:8080/v1"),
        ("[::]:11434", "http://[::1]:11434/v1"),
        ("http://ollama.local", "http://ollama.local/v1"),
        ("https://ollama.com", "https://ollama.com/v1"),
        ("https://proxy.example.com:8443/ollama/", "https://proxy.example.com:8443/ollama/v1"),
        ("localhost:notaport", "http://localhost:11434/v1"),
    ],
)
def test_ollama_host_parsing(host: str, url: str) -> None:
    assert ollama_base_url(host) == url


@requires_openai
@pytest.mark.parametrize("name", sorted(HOSTS))
def test_create_defaults_and_keys(name: str, clean_env: pytest.MonkeyPatch) -> None:
    _, base_url, key_env, key_required, default_model, _ = HOSTS[name]
    if key_required:
        with pytest.raises(ConfigurationError, match=key_env):
            create("llm", name)
        clean_env.setenv(key_env, f"key-for-{name}")
    llm = create("llm", name)
    assert llm.base_url == base_url.rstrip("/")
    assert llm.model == (default_model or "auto")
    assert llm._client.api_key == (f"key-for-{name}" if key_required else "no-key")
    assert llm._max_tokens_param == "max_tokens" and llm._developer_role == "system"
    # an explicit model and base_url (e.g. a proxy) still use the host's key
    llm = create("llm", f"{name}/some/model:tag", base_url="https://proxy.example.com/v1/")
    assert llm.model == "some/model:tag" and llm.base_url == "https://proxy.example.com/v1"
    if key_required:
        assert llm._client.api_key == f"key-for-{name}"


@requires_openai
def test_base_url_environment_overrides(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("OLLAMA_HOST", "0.0.0.0:11500")
    assert create("llm", "ollama").base_url == "http://127.0.0.1:11500/v1"
    clean_env.setenv("OLLAMA_BASE_URL", "http://ollama.internal:9000/v1")
    assert create("llm", "ollama").base_url == "http://ollama.internal:9000/v1"
    clean_env.setenv("VLLM_BASE_URL", "http://gpu-node:8001/v1")
    assert create("llm", "vllm").base_url == "http://gpu-node:8001/v1"
    clean_env.setenv("GROQ_API_KEY", "k")
    clean_env.setenv("GROQ_BASE_URL", "https://groq-proxy.example.com/openai/v1")
    llm = create("llm", "groq", base_url="https://explicit.example.com/v1")
    assert llm.base_url == "https://explicit.example.com/v1"  # the argument wins
    assert create("llm", "groq").base_url == "https://groq-proxy.example.com/openai/v1"


@requires_openai
def test_local_server_keys_are_optional_but_sent_when_set(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("VLLM_API_KEY", "vllm-secret")
    clean_env.setenv("LM_API_TOKEN", "lm-token")
    assert create("llm", "vllm")._client.api_key == "vllm-secret"
    assert create("llm", "lmstudio")._client.api_key == "lm-token"
    assert create("llm", "llamacpp", api_key="explicit")._client.api_key == "explicit"


def test_host_modules_import_without_side_effects() -> None:
    # every module must be importable without the openai SDK installed (`van providers`)
    for name in HOSTS:
        module = importlib.import_module(f"voice_agent_next.providers.{name}")
        assert HOSTS[name][0] in module.__all__
