"""``OPENAI_API_KEY`` is never sent to a non-OpenAI host by the Realtime / Live engines."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import pytest

from tests.fake_realtime_server import FakeRealtimeServer
from voice_agent_next.engine import EngineOptions
from voice_agent_next.errors import AuthenticationError, ConfigurationError
from voice_agent_next.providers.azure_openai import AzureOpenAIRealtimeEngine
from voice_agent_next.providers.localai import LocalAIRealtimeEngine
from voice_agent_next.providers.openai.live import OpenAILiveEngine, OpenAILiveSessionEngine
from voice_agent_next.providers.openai.realtime import PROFILES, OpenAIRealtimeEngine
from voice_agent_next.providers.qwen_omni import QwenOmniRealtimeEngine
from voice_agent_next.providers.speaches import SpeachesRealtimeEngine
from voice_agent_next.providers.vllm_realtime import VLLMRealtimeEngine
from voice_agent_next.providers.xai import XAIRealtimeEngine
from voice_agent_next.testing.openai_live import FakeLiveServer

OPENAI_KEY = "sk-openai-secret"

ENV_VARS = (
    "OPENAI_API_KEY", "OPENAI_LIVE_BASE_URL", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_AD_TOKEN",
    "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT_NAME", "XAI_API_KEY", "DASHSCOPE_API_KEY",
    "DASHSCOPE_WORKSPACE_ID", "DASHSCOPE_REGION", "VLLM_BASE_URL", "VLLM_API_KEY",
    "SPEACHES_BASE_URL", "SPEACHES_API_KEY", "LOCALAI_BASE_URL", "LOCALAI_API_KEY",
    "MY_REALTIME_URL",
)  # fmt: skip


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)


def _auth(headers: dict[str, str]) -> str | None:
    return next((v for k, v in headers.items() if k.lower() == "authorization"), None)


# ------------------------------------------------------------------------------ Realtime
@pytest.mark.parametrize(
    "url", ["wss://api.openai.com/v1", "https://api.openai.com/v1", "wss://us.api.openai.com/v1"]
)
def test_realtime_openai_hosts_use_the_env_key(url: str) -> None:
    assert OpenAIRealtimeEngine().api_key == OPENAI_KEY  # default host
    engine = OpenAIRealtimeEngine(base_url=url)
    assert _auth(engine.request_headers()) == f"Bearer {OPENAI_KEY}"


@pytest.mark.parametrize(
    "url", ["wss://example.com/v1", "ws://localhost:8000/v1", "wss://api.openai.com.evil.io/v1"]
)
def test_realtime_custom_base_url_never_gets_the_openai_key(
    url: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="voice_agent_next"):
        engine = OpenAIRealtimeEngine(base_url=url)
    assert engine.api_key is None and _auth(engine.request_headers()) is None
    assert "OPENAI_API_KEY is only sent to api.openai.com" in caplog.text
    assert OPENAI_KEY not in caplog.text
    # an explicit key is always sent, without a warning
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="voice_agent_next"):
        engine = OpenAIRealtimeEngine(base_url=url, api_key="sk-proxy")
    assert _auth(engine.request_headers()) == "Bearer sk-proxy"
    assert not caplog.text
    # so is an explicit Authorization header
    engine = OpenAIRealtimeEngine(base_url=url, headers={"Authorization": "Bearer tok"})
    assert engine.request_headers() == {"Authorization": "Bearer tok"}


def test_realtime_key_still_required_for_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        OpenAIRealtimeEngine()
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        OpenAIRealtimeEngine(base_url="wss://api.openai.com/v1")
    assert OpenAIRealtimeEngine(base_url="wss://example.com/v1").api_key is None
    assert OpenAIRealtimeEngine(api_key="").api_key == ""  # "no key": no env fallback


def test_realtime_custom_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    # a profile pointed at another server is guarded too
    proxy = replace(PROFILES["openai"], name="proxy", base_url="wss://proxy.example/v1")
    assert OpenAIRealtimeEngine(profile=proxy).api_key is None
    # ... unless its URL comes from the environment next to the key (a configured proxy)
    env_proxy = replace(proxy, base_url_env=("MY_REALTIME_URL",))
    monkeypatch.setenv("MY_REALTIME_URL", "wss://proxy.example/v1")
    assert OpenAIRealtimeEngine(profile=env_proxy).api_key == OPENAI_KEY
    assert (
        OpenAIRealtimeEngine(profile=env_proxy, base_url="wss://other.example/v1").api_key is None
    )


async def test_realtime_handshake_to_custom_server_carries_no_openai_key() -> None:
    async with FakeRealtimeServer() as server:
        conn = await OpenAIRealtimeEngine(base_url=server.url).connect(EngineOptions())
        await conn.aclose()
        assert "authorization" not in server.handshakes[0].headers
    async with FakeRealtimeServer(api_key="sk-proxy") as server:
        engine = OpenAIRealtimeEngine(base_url=server.url, api_key="sk-proxy")
        conn = await engine.connect(EngineOptions())
        await conn.aclose()
        assert server.handshakes[0].headers["authorization"] == "Bearer sk-proxy"


async def test_compat_profiles_use_their_own_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "xai-key")
    assert _auth(XAIRealtimeEngine().request_headers()) == "Bearer xai-key"
    monkeypatch.setenv("DASHSCOPE_API_KEY", "ds-key")
    qwen = QwenOmniRealtimeEngine(workspace_id="ws-1")
    assert _auth(qwen.request_headers()) == "Bearer ds-key"
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
    azure = AzureOpenAIRealtimeEngine(endpoint="https://r.openai.azure.com", deployment="d")
    headers = azure.request_headers()
    assert headers["api-key"] == "az-key" and _auth(headers) is None
    # local servers: no key by default, their own key when set; never OpenAI's
    for engine_cls, key_env, url_env in (
        (VLLMRealtimeEngine, "VLLM_API_KEY", "VLLM_BASE_URL"),
        (SpeachesRealtimeEngine, "SPEACHES_API_KEY", "SPEACHES_BASE_URL"),
        (LocalAIRealtimeEngine, "LOCALAI_API_KEY", "LOCALAI_BASE_URL"),
    ):
        assert _auth(engine_cls().request_headers()) is None, engine_cls
        async with FakeRealtimeServer(api_key="local-key") as server:
            monkeypatch.setenv(key_env, "local-key")
            monkeypatch.setenv(url_env, server.url)
            conn = await engine_cls().connect(EngineOptions())
            await conn.aclose()
            assert server.handshakes[0].headers["authorization"] == "Bearer local-key"
        monkeypatch.delenv(key_env)
        monkeypatch.delenv(url_env)


# ---------------------------------------------------------------------------------- Live
def _live(wrapped: bool, **kw: Any) -> OpenAILiveSessionEngine:
    if wrapped:  # what ``openai-live`` creates: a rotating wrapper around a session engine
        return OpenAILiveEngine(**kw).session_engine
    return OpenAILiveSessionEngine(**kw)


@pytest.mark.parametrize("wrapped", [True, False])
def test_live_key_guard(
    wrapped: bool, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    assert _live(wrapped).api_key == OPENAI_KEY  # OpenAI's host
    assert _live(wrapped, base_url="https://api.openai.com/v1").api_key == OPENAI_KEY
    with caplog.at_level(logging.WARNING, logger="voice_agent_next"):
        custom = _live(wrapped, base_url="wss://live.example/v1")
    assert custom.api_key is None and _auth(custom.request_headers()) is None
    assert "only sent to api.openai.com" in caplog.text
    explicit = _live(wrapped, base_url="wss://live.example/v1", api_key="sk-proxy")
    assert _auth(explicit.request_headers()) == "Bearer sk-proxy"
    # OPENAI_LIVE_BASE_URL is configured next to the key: a deliberate proxy
    monkeypatch.setenv("OPENAI_LIVE_BASE_URL", "wss://live.example/v1")
    assert _live(wrapped).api_key == OPENAI_KEY
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        _live(wrapped)
    monkeypatch.delenv("OPENAI_LIVE_BASE_URL")
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        _live(wrapped)


async def test_live_handshake_to_custom_server_carries_no_openai_key() -> None:
    server = FakeLiveServer(api_key=OPENAI_KEY)
    await server.start()
    try:
        # the fake server only accepts the OpenAI key, which must not be sent to it
        with pytest.raises(AuthenticationError):
            await OpenAILiveSessionEngine(base_url=server.url).connect(EngineOptions())
        engine = OpenAILiveSessionEngine(base_url=server.url, api_key=OPENAI_KEY)
        conn = await engine.connect(EngineOptions())
        await conn.aclose()
    finally:
        await server.aclose()
    assert server.errors == []
