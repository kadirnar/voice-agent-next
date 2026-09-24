"""Real-server tests for the OpenAI-compatible LLMs (``-m integration``).

* OpenAI: needs ``OPENAI_API_KEY`` (model: ``VAN_OPENAI_MODEL``, default ``gpt-4.1-mini``).
* Ollama: skipped unless a local Ollama answers (``OLLAMA_HOST``, default
  ``127.0.0.1:11434``) and has the model: ``VAN_OLLAMA_MODEL`` (default
  ``LiquidAI/lfm2.5-1.2b-instruct:latest``); tool calls use ``VAN_OLLAMA_TOOL_MODEL`` or
  the first local model with the ``tools`` capability.

Run: ``uv run pytest -m integration tests/test_openai_integration.py -s``
"""

from __future__ import annotations

import asyncio
import os

import pytest

pytest.importorskip("openai")

import httpx

from voice_agent_next import Agent, AgentSession, AudioFrame, CascadeOptions, ChatContext, create
from voice_agent_next.metrics import LLMMetrics, TurnMetrics
from voice_agent_next.providers.mock import synth_speech
from voice_agent_next.providers.ollama import ollama_base_url
from voice_agent_next.tools import function_tool
from voice_agent_next.transports import LoopbackTransport

pytestmark = pytest.mark.integration

OLLAMA_MODEL = os.environ.get("VAN_OLLAMA_MODEL", "LiquidAI/lfm2.5-1.2b-instruct:latest")


@function_tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny and 21 degrees in {city}"


def voice_ctx(question: str) -> ChatContext:
    ctx = ChatContext()
    ctx.add_message("system", "You are a voice assistant. Answer in one short sentence.")
    ctx.add_message("user", question)
    return ctx


@pytest.fixture(scope="module")
def ollama_models() -> dict[str, list[str]]:
    """Local Ollama models and their capabilities; skips when Ollama is not reachable."""
    root = ollama_base_url(os.environ.get("OLLAMA_HOST") or "127.0.0.1:11434").removesuffix("/v1")
    try:
        response = httpx.get(f"{root}/api/tags", timeout=2.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"Ollama is not reachable at {root}: {exc}")
    return {m["name"]: m.get("capabilities", []) for m in response.json().get("models", [])}


def require_model(models: dict[str, list[str]], name: str) -> str:
    if name not in models:
        pytest.skip(f"Ollama model {name!r} is not pulled (ollama pull {name})")
    return name


# ------------------------------------------------------------------------------ Ollama


async def test_ollama_streams_text_with_usage(ollama_models: dict[str, list[str]]) -> None:
    model = require_model(ollama_models, OLLAMA_MODEL)
    llm = create("llm", f"ollama/{model}", temperature=0.0)
    metrics: list[LLMMetrics] = []
    llm.on("metrics", metrics.append)
    await llm.warmup()  # loads the model: the measured request is a warm one
    try:
        chunks = [c async for c in llm.chat(voice_ctx("What is the capital of France?"))]
    finally:
        await llm.aclose()
    text = "".join(c.delta for c in chunks)
    assert "paris" in text.lower()
    assert chunks[-1].finish_reason == "stop"
    [m] = metrics
    assert m.prompt_tokens > 0 and m.completion_tokens > 0 and m.error is None
    assert m.ttft is not None
    print(f"\nollama {model}: ttft={m.ttft * 1000:.0f} ms, total={m.duration * 1000:.0f} ms, "
          f"{m.completion_tokens} tokens at {m.tokens_per_second:.0f} tok/s: {text!r}")  # fmt: skip


async def test_ollama_tool_calls(ollama_models: dict[str, list[str]]) -> None:
    name = os.environ.get("VAN_OLLAMA_TOOL_MODEL") or next(
        (n for n, caps in ollama_models.items() if "tools" in caps), None
    )
    if name is None:
        pytest.skip("no local Ollama model with the 'tools' capability")
    model = require_model(ollama_models, name)
    llm = create("llm", f"ollama/{model}", temperature=0.0)
    ctx = ChatContext()
    ctx.add_message("user", "What is the weather in Paris? Use the tool.")
    try:
        result = await llm.chat(ctx, tools=[get_weather]).collect()
    finally:
        await llm.aclose()
    assert result.tool_calls, f"{model} answered without calling the tool: {result.text!r}"
    call = result.tool_calls[0]
    assert call.name == "get_weather"
    assert "paris" in call.parsed_arguments()["city"].lower()


async def test_ollama_in_the_cascade(ollama_models: dict[str, list[str]]) -> None:
    model = require_model(ollama_models, OLLAMA_MODEL)
    session = AgentSession(
        stt="mock",  # hears "hello"
        llm=f"ollama/{model}",
        tts="mock",
        vad="energy",
        cascade_options=CascadeOptions(min_endpointing_delay=0.0),
    )
    await session.engine.warmup()
    replies: list[str] = []
    turns: list[TurnMetrics] = []
    session.on("agent_transcript", lambda ev: replies.append(ev.delta))
    session.on("metrics", lambda m: turns.append(m) if isinstance(m, TurnMetrics) else None)
    transport = LoopbackTransport()
    await session.start(Agent("You are a friendly voice assistant. Reply briefly."), transport)
    await transport.play_user_audio(synth_speech(0.8, 16_000), realtime=False)
    await transport.play_user_audio(AudioFrame.silence(0.6, 16_000), realtime=False)
    try:
        async with asyncio.timeout(60):
            while not turns:
                await asyncio.sleep(0.05)
    finally:
        await session.aclose()
    assert "".join(replies).strip()
    assert turns[0].response_ttfb is not None


# ------------------------------------------------------------------------------ OpenAI


@pytest.fixture
def openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY is not set")
    return key


async def test_openai_text_and_tool_calls(openai_key: str) -> None:
    model = os.environ.get("VAN_OPENAI_MODEL", "gpt-4.1-mini")
    llm = create("llm", f"openai/{model}", max_tokens=64)
    metrics: list[LLMMetrics] = []
    llm.on("metrics", metrics.append)
    try:
        text = await llm.chat(voice_ctx("What is the capital of France?")).collect()
        tools = await llm.chat(
            voice_ctx("What's the weather in Paris and in Rome?"),
            tools=[get_weather],
            tool_choice="required",
        ).collect()
    finally:
        await llm.aclose()
    assert "paris" in text.text.lower()
    assert {c.name for c in tools.tool_calls} == {"get_weather"}
    assert metrics[0].prompt_tokens > 0 and metrics[0].ttft is not None
