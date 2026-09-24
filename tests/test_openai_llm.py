"""OpenAILLM end to end through the ``openai`` SDK against an ``httpx.MockTransport``
server that replays Chat Completions SSE streams (no network)."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("openai")

import httpx

from voice_agent_next import Agent, AgentSession, AudioFrame, CascadeOptions, ChatContext
from voice_agent_next.errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from voice_agent_next.metrics import LLMMetrics
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import MockSTT, MockTTS, synth_speech
from voice_agent_next.providers.ollama import OllamaLLM
from voice_agent_next.providers.openai.llm import OpenAILLM
from voice_agent_next.providers.vllm import VllmLLM
from voice_agent_next.tools import function_tool
from voice_agent_next.transports import LoopbackTransport

# ------------------------------------------------------------------------- fake server


def chunk(
    delta: dict[str, Any] | None = None,
    *,
    finish: str | None = None,
    usage: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """A ``chat.completion.chunk`` as OpenAI streams it (``usage`` is null until the end)."""
    return {
        "id": "chatcmpl-B9MHDbslfkBeAs8l4bebGdFOJ6PeG",
        "object": "chat.completion.chunk",
        "created": 1790000000,
        "model": "gpt-4.1-mini-2025-04-14",
        "service_tier": "default",
        "system_fingerprint": "fp_6f2eabb9a5",
        "choices": []
        if delta is None
        else [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}],
        "usage": usage,
        **extra,
    }


def usage_chunk(prompt: int, completion: int, cached: int = 0) -> dict[str, Any]:
    return chunk(
        usage={
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": cached, "audio_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0, "audio_tokens": 0},
        }
    )


TEXT_STREAM = [
    chunk({"role": "assistant", "content": "", "refusal": None}),
    chunk({"content": "Hello"}),
    chunk({"content": "! How"}),
    chunk({"content": " can I help?"}),
    chunk({}, finish="stop"),
    usage_chunk(1450, 7, cached=1280),
]


def tool_fragment(index: int, *, call_id: str | None = None, name: str | None = None,
                  arguments: str = "") -> dict[str, Any]:  # fmt: skip
    fragment: dict[str, Any] = {"index": index, "function": {"arguments": arguments}}
    if call_id is not None:
        fragment.update(id=call_id, type="function")
        fragment["function"]["name"] = name
    return fragment


PARALLEL_TOOL_STREAM = [
    chunk({"role": "assistant", "content": None, "refusal": None,
           "tool_calls": [tool_fragment(0, call_id="call_paris", name="get_weather")]}),
    chunk({"tool_calls": [tool_fragment(0, arguments='{"ci')]}),
    chunk({"tool_calls": [tool_fragment(0, arguments='ty": "Pa')]}),
    chunk({"tool_calls": [tool_fragment(0, arguments='ris"}')]}),
    chunk({"tool_calls": [tool_fragment(1, call_id="call_rome", name="get_weather")]}),
    chunk({"tool_calls": [tool_fragment(1, arguments='{"city": ')]}),
    chunk({"tool_calls": [tool_fragment(1, arguments='"Rome"}')]}),
    chunk({}, finish="tool_calls"),
    usage_chunk(210, 38),
]  # fmt: skip


class SSEStream(httpx.AsyncByteStream):
    """SSE body sent in ``piece``-byte slices, ``delay`` apart; records ``aclose()``."""

    def __init__(self, events: list[Any], *, piece: int = 0, delay: float = 0.0) -> None:
        lines = [f"data: {e if isinstance(e, str) else json.dumps(e)}\n\n" for e in events]
        self.data = ("".join(lines) + "data: [DONE]\n\n").encode()
        self.piece = piece or len(self.data)
        self.delay = delay
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for i in range(0, len(self.data), self.piece):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield self.data[i : i + self.piece]

    async def aclose(self) -> None:
        self.closed = True


Reply = SSEStream | httpx.Response | Exception


@dataclass
class FakeServer:
    """Serves scripted replies to ``POST /chat/completions`` and a model list at ``GET /models``."""

    replies: list[Reply] = field(default_factory=list)
    models: list[str] = field(default_factory=lambda: ["gpt-4.1-mini"])
    models_status: int = 200
    requests: list[httpx.Request] = field(default_factory=list)
    streams: list[SSEStream] = field(default_factory=list)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "GET" and request.url.path.endswith("/models"):
            if self.models_status != 200:
                return error_response(self.models_status, "no such route")
            data = [
                {"id": m, "object": "model", "created": 0, "owned_by": "me"} for m in self.models
            ]
            return httpx.Response(200, json={"object": "list", "data": data})
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, SSEStream):
            self.streams.append(reply)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=reply)
        return reply

    def http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))

    @property
    def chat_bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if r.method == "POST"]


def make_llm(server: FakeServer, cls: type[OpenAILLM] = OpenAILLM, **kw: Any) -> OpenAILLM:
    kw.setdefault("api_key", "sk-test")
    kw.setdefault("max_retries", 0)
    return cls(http_client=server.http_client(), **kw)


def user_ctx(text: str = "Hi there") -> ChatContext:
    ctx = ChatContext()
    ctx.add_message("system", "You are a helpful voice assistant.")
    ctx.add_message("user", text)
    return ctx


def collect_metrics(llm: OpenAILLM) -> list[LLMMetrics]:
    metrics: list[LLMMetrics] = []
    llm.on("metrics", metrics.append)
    return metrics


@function_tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"sunny in {city}"


# ------------------------------------------------------------------------------ streams


async def test_streams_text_and_usage_with_cached_tokens() -> None:
    server = FakeServer([SSEStream(TEXT_STREAM)])
    llm = make_llm(server)
    metrics = collect_metrics(llm)
    deltas = [
        c.delta async for c in llm.chat(user_ctx(), temperature=0.3, max_tokens=64) if c.delta
    ]
    assert deltas == ["Hello", "! How", " can I help?"]

    [body] = server.chat_bodies
    assert body["model"] == "gpt-4.1-mini"
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["messages"] == [
        {"role": "system", "content": "You are a helpful voice assistant."},
        {"role": "user", "content": "Hi there"},
    ]
    assert body["temperature"] == 0.3
    assert body["max_completion_tokens"] == 64 and "max_tokens" not in body  # OpenAI itself
    assert "tools" not in body and "tool_choice" not in body
    request = server.requests[0]
    assert request.url == "https://api.openai.com/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-test"

    [m] = metrics
    assert (m.provider, m.model, m.error, m.cancelled) == ("openai", "gpt-4.1-mini", None, False)
    assert (m.prompt_tokens, m.completion_tokens, m.cached_tokens) == (1450, 7, 1280)
    assert m.ttft is not None and 0 <= m.ttft <= m.duration


async def test_collect_returns_text_usage_and_finish_reason() -> None:
    server = FakeServer([SSEStream(TEXT_STREAM, piece=7)])  # SSE split at odd byte offsets
    llm = make_llm(server)
    stream = llm.chat(user_ctx())
    chunks = [c async for c in stream]
    assert chunks[-1].finish_reason == "stop" and chunks[-1].usage is not None
    assert "".join(c.delta for c in chunks) == "Hello! How can I help?"
    result = await make_llm(FakeServer([SSEStream(TEXT_STREAM)])).chat(user_ctx()).collect()
    assert result.text == "Hello! How can I help?" and result.tool_calls == []
    assert result.usage is not None and result.usage.total_tokens == 1457


@pytest.mark.parametrize("piece", [0, 5])
async def test_parallel_tool_calls_with_fragmented_arguments(piece: int) -> None:
    server = FakeServer([SSEStream(PARALLEL_TOOL_STREAM, piece=piece)])
    llm = make_llm(server, parallel_tool_calls=True)
    metrics = collect_metrics(llm)
    chunks = [c async for c in llm.chat(user_ctx("Weather in Paris and Rome?"), tools=[get_weather],
                                         tool_choice="get_weather")]  # fmt: skip
    calls = [call for c in chunks for call in c.tool_calls]
    assert [(c.call_id, c.name, c.arguments) for c in calls] == [
        ("call_paris", "get_weather", '{"city": "Paris"}'),
        ("call_rome", "get_weather", '{"city": "Rome"}'),
    ]
    assert [c.parsed_arguments() for c in calls] == [{"city": "Paris"}, {"city": "Rome"}]
    assert sum(1 for c in chunks if c.tool_calls) == 1  # emitted once, complete
    assert all(not c.delta for c in chunks)
    assert chunks[-1].finish_reason == "tool_calls"

    [body] = server.chat_bodies
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": get_weather.parameters,
            },
        }
    ]
    assert body["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}
    assert body["parallel_tool_calls"] is True
    [m] = metrics
    assert m.ttft is not None and m.completion_tokens == 38 and m.error is None


async def test_reasoning_is_never_spoken_and_think_blocks_are_stripped() -> None:
    events = [
        chunk({"role": "assistant", "content": "", "reasoning_content": "The user greets me."}),
        chunk({"content": "", "reasoning": "Keep it short."}),
        chunk({"content": "<think>\nOkay, a greeting.\n</th"}),
        chunk({"content": "ink>\n\nHi"}),
        chunk({"content": "! Nice to meet you."}),
        chunk({}, finish="stop"),
    ]
    result = await make_llm(FakeServer([SSEStream(events)])).chat(user_ctx()).collect()
    assert result.text == "Hi! Nice to meet you."
    raw = (
        await make_llm(FakeServer([SSEStream(events)]), strip_thinking=False)
        .chat(user_ctx())
        .collect()
    )
    assert raw.text.startswith("<think>")


async def test_refusal_is_spoken() -> None:
    events = [chunk({"role": "assistant", "content": None, "refusal": "I can't help with that."}),
              chunk({}, finish="stop")]  # fmt: skip
    result = await make_llm(FakeServer([SSEStream(events)])).chat(user_ctx()).collect()
    assert result.text == "I can't help with that."


async def test_cancellation_closes_the_http_stream() -> None:
    events = [chunk({"content": f"word{i} "}) for i in range(50)]
    body = SSEStream(events, piece=64, delay=0.01)
    llm = make_llm(FakeServer([body]))
    metrics = collect_metrics(llm)
    stream = llm.chat(user_ctx())
    first = await stream.__anext__()
    assert first.delta == "word0 "
    await stream.aclose()
    assert body.closed, "the HTTP response must be closed on cancellation"
    [m] = metrics
    assert m.cancelled and m.error is None


# ------------------------------------------------------------------------------ errors


def error_response(status: int, message: str, code: str | None = None) -> httpx.Response:
    error = {"message": message, "type": "invalid_request_error", "param": None, "code": code}
    return httpx.Response(status, json={"error": error})


@pytest.mark.parametrize(
    ("reply", "error_type", "status", "retryable"),
    [
        (error_response(401, "Incorrect API key provided"), AuthenticationError, 401, False),
        (error_response(403, "Project does not have access"), AuthenticationError, 403, False),
        (error_response(404, "The model `gpt-9` does not exist"), ProviderError, 404, False),
        (error_response(429, "Rate limit reached", "rate_limit_exceeded"), RateLimitError, 429, True),
        (error_response(429, "You exceeded your quota", "insufficient_quota"), RateLimitError, 429, False),
        (error_response(400, "Invalid 'messages'"), ProviderError, 400, False),
        (httpx.Response(500, text="upstream crashed"), ProviderError, 500, True),
        (error_response(503, "Service overloaded"), ProviderError, 503, True),
        (httpx.Response(408, text="timeout"), ProviderTimeoutError, 408, True),
    ],
)  # fmt: skip
async def test_http_errors_are_mapped(
    reply: httpx.Response, error_type: type[Exception], status: int, retryable: bool
) -> None:
    llm = make_llm(FakeServer([reply]))
    metrics = collect_metrics(llm)
    with pytest.raises(error_type) as info:
        await llm.chat(user_ctx()).collect()
    err = info.value
    assert isinstance(err, ProviderError)
    assert (err.status_code, err.retryable, err.provider) == (status, retryable, "openai")
    assert f"HTTP {status}" in str(err) and "gpt-4.1-mini" in str(err)
    assert metrics[0].error is not None and metrics[0].ttft is None


async def test_not_found_error_has_provider_hint() -> None:
    reply = httpx.Response(404, json={"error": {"message": "model 'qwen9' not found",
                                                "type": "not_found_error"}})  # fmt: skip
    llm = make_llm(FakeServer([reply]), cls=OllamaLLM, model="qwen9")
    with pytest.raises(ProviderError, match=r"model 'qwen9' not found.*ollama pull qwen9"):
        await llm.chat(user_ctx()).collect()


@pytest.mark.parametrize(
    ("exc", "error_type"),
    [
        (httpx.ConnectError("All connection attempts failed"), ProviderConnectionError),
        (httpx.ReadTimeout("timed out"), ProviderTimeoutError),
    ],
)
async def test_network_errors_are_mapped(exc: Exception, error_type: type[Exception]) -> None:
    llm = make_llm(FakeServer([exc]), cls=OllamaLLM)
    with pytest.raises(error_type) as info:
        await llm.chat(user_ctx()).collect()
    assert info.value.retryable  # type: ignore[attr-defined]
    assert "127.0.0.1:11434" in str(info.value)


async def test_error_event_in_the_middle_of_the_stream() -> None:
    events = [
        chunk({"role": "assistant", "content": "Let me"}),
        {"error": {"message": "The server is overloaded", "type": "server_error", "code": None}},
    ]
    llm = make_llm(FakeServer([SSEStream(events)]))
    received: list[str] = []

    async def consume() -> None:
        async for c in llm.chat(user_ctx()):
            received.append(c.delta)

    with pytest.raises(ProviderError, match="overloaded") as info:
        await consume()
    assert received == ["Let me"]  # text before the error was delivered
    assert info.value.retryable


async def test_sdk_retries_transient_errors_before_streaming() -> None:
    server = FakeServer([error_response(503, "busy"), SSEStream(TEXT_STREAM)])
    llm = make_llm(server, max_retries=1)
    result = await llm.chat(user_ctx()).collect()
    assert result.text == "Hello! How can I help?"
    assert len(server.chat_bodies) == 2


# -------------------------------------------------------------------- request options


async def test_compatible_server_request_shape() -> None:
    server = FakeServer([SSEStream(TEXT_STREAM)])
    llm = make_llm(server, base_url="http://localhost:8080/v1/", model="local-model",
                   api_key=None, max_tokens=100, reasoning_effort="none",
                   extra={"top_k": 20, "seed": 7, "chat_template_kwargs": {"enable_thinking": False}},
                   headers={"X-Title": "voice-agent-next"})  # fmt: skip
    ctx = user_ctx()
    ctx.add_message("developer", "Answer in English.")
    await llm.chat(ctx, extra={"top_k": 40}).collect()
    [body] = server.chat_bodies
    assert body["model"] == "local-model"
    assert body["max_tokens"] == 100 and "max_completion_tokens" not in body
    assert body["messages"][-1] == {"role": "system", "content": "Answer in English."}
    assert body["reasoning_effort"] == "none" and body["seed"] == 7
    assert body["top_k"] == 40  # per-call extra wins; unknown keys go to the JSON body
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    request = server.requests[0]
    assert request.url == "http://localhost:8080/v1/chat/completions"
    assert request.headers["x-title"] == "voice-agent-next"
    assert request.headers["authorization"] == "Bearer no-key"


async def test_tool_only_parameters_are_dropped_without_tools() -> None:
    server = FakeServer([SSEStream(TEXT_STREAM)])
    llm = make_llm(server, parallel_tool_calls=False)
    await llm.chat(user_ctx(), tool_choice="required").collect()
    [body] = server.chat_bodies
    assert "parallel_tool_calls" not in body and "tool_choice" not in body


async def test_none_in_extra_removes_a_default_parameter() -> None:
    from voice_agent_next.providers.deepseek import DeepSeekLLM

    server = FakeServer([SSEStream(TEXT_STREAM)])
    llm = make_llm(server, cls=DeepSeekLLM)
    await llm.chat(user_ctx()).collect()
    await llm.chat(user_ctx(), extra={"thinking": None, "stream_options": None}).collect()
    first, second = server.chat_bodies
    assert first["thinking"] == {"type": "disabled"} and first["model"] == "deepseek-flash"
    assert "thinking" not in second and "stream_options" not in second
    assert server.requests[0].url == "https://api.deepseek.com/chat/completions"


async def test_model_is_discovered_when_the_server_has_no_default() -> None:
    server = FakeServer([SSEStream(TEXT_STREAM)], models=["nomic-embed-text", "Qwen/Qwen3-8B"])
    llm = make_llm(server, cls=VllmLLM, api_key=None)
    assert llm.model == "auto"
    metrics = collect_metrics(llm)
    await llm.chat(user_ctx()).collect()
    await llm.chat(user_ctx()).collect()
    assert llm.model == "Qwen/Qwen3-8B"
    assert [b["model"] for b in server.chat_bodies] == ["Qwen/Qwen3-8B"] * 2
    assert sum(r.method == "GET" for r in server.requests) == 1  # discovered once
    assert metrics[0].model == "Qwen/Qwen3-8B"


async def test_model_discovery_without_models_is_a_configuration_error() -> None:
    llm = make_llm(FakeServer([SSEStream(TEXT_STREAM)], models=[]), cls=VllmLLM)
    with pytest.raises(ConfigurationError, match="lists no models"):
        await llm.chat(user_ctx()).collect()
    server = FakeServer([SSEStream(TEXT_STREAM)], models_status=404)
    llm = make_llm(server, cls=VllmLLM)
    with pytest.raises(ConfigurationError, match=r"cannot list the models.*pass model="):
        await llm.chat(user_ctx()).collect()
    llm = make_llm(FakeServer([SSEStream(TEXT_STREAM)], models_status=401), cls=VllmLLM)
    with pytest.raises(AuthenticationError):
        await llm.chat(user_ctx()).collect()


async def test_warmup_tolerates_servers_without_a_model_list(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = FakeServer([SSEStream(TEXT_STREAM)], models_status=404)
    with caplog.at_level(logging.WARNING, logger="voice_agent_next"):
        await make_llm(server).warmup()
    assert "warmup failed" not in caplog.text
    denied = FakeServer([SSEStream(TEXT_STREAM)], models_status=401)
    with caplog.at_level(logging.WARNING, logger="voice_agent_next"):
        await make_llm(denied).warmup()
    assert "warmup failed" in caplog.text and "HTTP 401" in caplog.text


async def test_warmup_opens_connection_and_preloads_local_models(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = FakeServer([httpx.Response(200, json={
        "id": "x", "object": "chat.completion", "created": 0, "model": "qwen3.5:4b",
        "choices": [{"index": 0, "finish_reason": "length",
                     "message": {"role": "assistant", "content": "Hi"}}],
    })])  # fmt: skip
    llm = make_llm(server, cls=OllamaLLM)
    await llm.warmup()
    assert [(r.method, r.url.path) for r in server.requests] == [
        ("GET", "/v1/models"),
        ("POST", "/v1/chat/completions"),
    ]
    assert server.chat_bodies[0]["max_tokens"] == 1

    down = make_llm(FakeServer([httpx.ConnectError("refused")]), cls=OllamaLLM)
    with caplog.at_level(logging.WARNING, logger="voice_agent_next"):
        await down.warmup()  # never raises
    assert "warmup failed" in caplog.text


async def test_custom_client_is_used_and_not_closed() -> None:
    import openai

    server = FakeServer([SSEStream(TEXT_STREAM)])
    client = openai.AsyncOpenAI(api_key="k", base_url="https://example.test/v1",
                                http_client=server.http_client(), max_retries=0)  # fmt: skip
    llm = OpenAILLM(model="my-model", client=client)
    assert llm.base_url == "https://example.test/v1"
    assert (await llm.chat(user_ctx()).collect()).text == "Hello! How can I help?"
    await llm.aclose()
    assert not client.is_closed()


# -------------------------------------------------------------------- keys and config


def test_openai_requires_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        OpenAILLM()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    llm = OpenAILLM()
    assert llm._client.api_key == "sk-env" and llm.base_url == "https://api.openai.com/v1"
    assert llm.capabilities.image_input and llm.capabilities.tool_calling


def test_openai_key_is_never_sent_to_another_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    llm = OpenAILLM(base_url="https://llm.example.com/v1", model="m")
    assert llm._client.api_key == "no-key"
    # OPENAI_BASE_URL is the SDK's own setting: the key goes with it
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.com/v1")
    llm = OpenAILLM()
    assert llm.base_url == "https://gateway.example.com/v1" and llm._client.api_key == "sk-secret"


# -------------------------------------------------------------------- in the cascade


async def speak(transport: LoopbackTransport) -> None:
    await transport.play_user_audio(synth_speech(0.8, 16_000), realtime=False)
    await transport.play_user_audio(AudioFrame.silence(0.6, 16_000), realtime=False)


async def wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


async def test_cascade_turn_with_tool_round_trip() -> None:
    answer = [chunk({"role": "assistant", "content": "It is sunny"}),
              chunk({"content": " in Paris."}), chunk({}, finish="stop"), usage_chunk(300, 6)]  # fmt: skip
    tool_call = [chunk({"role": "assistant", "tool_calls": [
                     tool_fragment(0, call_id="call_1", name="get_weather", arguments='{"city":')]}),
                 chunk({"tool_calls": [tool_fragment(0, arguments=' "Paris"}')]}),
                 chunk({}, finish="tool_calls")]  # fmt: skip
    server = FakeServer([SSEStream(tool_call), SSEStream(answer)])
    llm = make_llm(server, cls=OllamaLLM)
    session = AgentSession(
        stt=MockSTT(transcripts=["what's the weather in paris"]),
        llm=llm,
        tts=MockTTS(),
        vad=EnergyVAD(),
        cascade_options=CascadeOptions(min_endpointing_delay=0.0),
    )
    transcripts: list[str] = []
    session.on("agent_transcript", lambda ev: transcripts.append(ev.delta))
    transport = LoopbackTransport()
    await session.start(Agent("You are a weather bot.", tools=[get_weather]), transport)
    await speak(transport)
    await wait_for(lambda: "Paris." in "".join(transcripts))
    await session.aclose()

    first, second = server.chat_bodies
    assert first["messages"] == [
        {"role": "system", "content": "You are a weather bot."},
        {"role": "user", "content": "what's the weather in paris"},
    ]
    assert first["model"] == "qwen3.5:4b" and first["tools"][0]["function"]["name"] == "get_weather"
    assert second["messages"][2:] == [
        {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {
            "name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny in Paris"},
    ]  # fmt: skip
    assert "It is sunny in Paris." in "".join(transcripts)
    kinds = [getattr(i, "role", i.type) for i in session.history.items]
    assert kinds == ["user", "function_call", "function_call_output", "assistant"]
