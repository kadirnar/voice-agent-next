"""AnthropicLLM against a fake Messages API (MockTransport replaying real SSE streams).

The SSE payloads mirror the event sequences documented for the streaming Messages API
(``message_start`` -> content blocks -> ``message_delta`` -> ``message_stop``, with
``ping`` and ``error`` events), delivered in small byte chunks so events and JSON
fragments are split across network reads. Needs the SDK (skipped otherwise):
``uv sync --extra anthropic && uv run pytest tests/providers/test_anthropic.py``; the
real-API test additionally needs ``ANTHROPIC_API_KEY`` and ``-m integration``.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from voice_agent_next import (
    Agent,
    AgentSession,
    AgentState,
    AudioFrame,
    CascadeOptions,
    ChatContext,
    create,
    function_tool,
)
from voice_agent_next.chat import AudioContent
from voice_agent_next.errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from voice_agent_next.llm import ChatChunk
from voice_agent_next.metrics import LLMMetrics, TurnMetrics
from voice_agent_next.providers.anthropic import (
    CONTINUE_PLACEHOLDER,
    START_PLACEHOLDER,
    AnthropicLLM,
    AnthropicUsage,
)
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import MockSTT, MockTTS, synth_speech
from voice_agent_next.registry import get_provider
from voice_agent_next.transports import LoopbackTransport

anthropic = pytest.importorskip("anthropic")
# anthropic >= 1.0 is built on httpx2 (the maintained httpx fork); 0.x used httpx.
http = importlib.import_module("httpx" if anthropic.__version__.startswith("0.") else "httpx2")

MODEL = "claude-haiku-4-5"
EPHEMERAL = {"type": "ephemeral"}


# ------------------------------------------------------------------ SSE fixtures


def sse(events: list[dict[str, Any]]) -> bytes:
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)


def message_start(
    input_tokens: int = 25, cache_read: int = 0, cache_write: int = 0
) -> dict[str, Any]:
    return {
        "type": "message_start",
        "message": {
            "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens,
                "cache_creation_input_tokens": cache_write,
                "cache_read_input_tokens": cache_read,
                "output_tokens": 1,
            },
        },
    }


def text_block(index: int, *deltas: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "text", "text": ""},
        },
        {"type": "ping"},
        *(
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": d},
            }
            for d in deltas
        ),
        {"type": "content_block_stop", "index": index},
    ]


def tool_block(index: int, call_id: str, name: str, *json_parts: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "tool_use", "id": call_id, "name": name, "input": {}},
        },
        *(
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "input_json_delta", "partial_json": part},
            }
            for part in json_parts
        ),
        {"type": "content_block_stop", "index": index},
    ]


def thinking_block(index: int, thought: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "thinking_delta", "thinking": thought},
        },
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "signature_delta", "signature": "EqQBCgIYAhIM1gbcDa9GJwZA2b3h"},
        },
        {"type": "content_block_stop", "index": index},
    ]


def message_end(stop_reason: str, output_tokens: int, **usage: int) -> list[dict[str, Any]]:
    return [
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens, **usage},
        },
        {"type": "message_stop"},
    ]


def error_event(error_type: str, message: str) -> dict[str, Any]:
    return {"type": "error", "error": {"type": error_type, "message": message}}


def stream_response(
    events: list[dict[str, Any]],
    *,
    chunk_size: int = 11,
    tail: Callable[[], Any] | None = None,
    closed: asyncio.Event | None = None,
) -> Any:
    """A 200 text/event-stream response whose body arrives in ``chunk_size`` byte pieces."""
    body = sse(events)

    async def chunks() -> AsyncIterator[bytes]:
        try:
            for i in range(0, len(body), chunk_size):
                await asyncio.sleep(0)
                yield body[i : i + chunk_size]
            if tail is not None:
                await tail()
        finally:
            if closed is not None:
                closed.set()

    return http.Response(
        200,
        headers={"content-type": "text/event-stream", "request-id": "req_011CStreamTest"},
        content=chunks(),
    )


def error_response(status: int, error_type: str, message: str) -> Any:
    return http.Response(
        status,
        headers={"request-id": "req_011CErrorTest"},
        json={"type": "error", "error": {"type": error_type, "message": message}},
    )


class FakeAnthropicAPI:
    """MockTransport handler: records requests and replays scripted responses."""

    def __init__(self) -> None:
        self.requests: list[Any] = []
        self.responses: list[Any] = []

    def reply(self, *responses: Any) -> None:
        self.responses.extend(responses)

    def __call__(self, request: Any) -> Any:
        self.requests.append(request)
        response = self.responses.pop(0)
        return response(request) if callable(response) else response

    def body(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index].content)


@pytest.fixture
def api() -> FakeAnthropicAPI:
    return FakeAnthropicAPI()


def make_llm(api: FakeAnthropicAPI, **kwargs: Any) -> AnthropicLLM:
    return AnthropicLLM(
        api_key="sk-ant-test-key",
        http_client=http.AsyncClient(transport=http.MockTransport(api)),
        max_retries=0,
        **kwargs,
    )


@function_tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny in {city}"


def user_ctx(text: str = "What's the weather in Paris?") -> ChatContext:
    ctx = ChatContext()
    ctx.add_message("system", "You are a friendly voice assistant.")
    ctx.add_message("user", text)
    return ctx


async def run(llm: AnthropicLLM, ctx: ChatContext, **kwargs: Any) -> list[ChatChunk]:
    return [chunk async for chunk in llm.chat(ctx, **kwargs)]


# ---------------------------------------------------------------------- streaming


async def test_streams_text_with_usage_and_sends_a_cached_request(api: FakeAnthropicAPI) -> None:
    api.reply(
        stream_response(
            [
                message_start(input_tokens=12, cache_read=4100, cache_write=230),
                *text_block(0, "It's", " sunny", " in Paris today."),
                *message_end("end_turn", output_tokens=9),
            ],
            chunk_size=7,
        )
    )
    llm = make_llm(api)
    metrics: list[LLMMetrics] = []
    llm.on("metrics", metrics.append)

    chunks = await run(llm, user_ctx(), tools=[get_weather])

    assert "".join(c.delta for c in chunks) == "It's sunny in Paris today."
    final = chunks[-1]
    assert final.finish_reason == "stop"
    assert final.usage == AnthropicUsage(
        prompt_tokens=12 + 4100 + 230,
        completion_tokens=9,
        cached_tokens=4100,
        cache_creation_tokens=230,
    )
    assert isinstance(final.usage, AnthropicUsage) and final.usage.uncached_prompt_tokens == 12

    request = api.requests[0]
    assert request.method == "POST" and request.url.path == "/v1/messages"
    assert request.headers["x-api-key"] == "sk-ant-test-key"
    assert request.headers["anthropic-version"]
    assert api.body() == {
        "model": MODEL,
        "max_tokens": 1024,
        "stream": True,
        "system": [
            {
                "type": "text",
                "text": "You are a friendly voice assistant.",
                "cache_control": EPHEMERAL,
            }
        ],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
                "cache_control": EPHEMERAL,
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's the weather in Paris?",
                        "cache_control": EPHEMERAL,
                    }
                ],
            }
        ],
    }

    (m,) = metrics
    assert (m.provider, m.model, m.error, m.cancelled) == ("anthropic", MODEL, None, False)
    assert (m.prompt_tokens, m.completion_tokens, m.cached_tokens) == (4342, 9, 4100)
    assert m.ttft is not None and m.ttft >= 0


async def test_tool_use_with_fragmented_json_yields_one_complete_call(
    api: FakeAnthropicAPI,
) -> None:
    api.reply(
        stream_response(
            [
                message_start(),
                *text_block(0, "Okay, let's check the weather", " for Paris:"),
                *tool_block(
                    1,
                    "toolu_01T1x1fJ34qAmk2tNTrN7Up6",
                    "get_weather",
                    "",
                    '{"ci',
                    'ty": "Pa',
                    'ris"}',
                ),
                *message_end("tool_use", output_tokens=89),
            ],
            chunk_size=5,  # splits SSE lines and the partial JSON strings mid-way
        )
    )
    llm = make_llm(api)

    chunks = await run(llm, user_ctx(), tools=[get_weather])

    calls = [call for chunk in chunks for call in chunk.tool_calls]
    assert len(calls) == 1
    (call,) = calls
    assert (call.name, call.call_id) == ("get_weather", "toolu_01T1x1fJ34qAmk2tNTrN7Up6")
    assert call.arguments == '{"city": "Paris"}'
    assert call.parsed_arguments() == {"city": "Paris"}
    assert "".join(c.delta for c in chunks) == "Okay, let's check the weather for Paris:"
    assert chunks[-1].finish_reason == "tool_calls"
    tool_chunk = next(i for i, c in enumerate(chunks) if c.tool_calls)
    assert tool_chunk < len(chunks) - 1  # emitted at content_block_stop, before the final chunk


async def test_tool_round_trip_sends_tool_use_and_tool_result(api: FakeAnthropicAPI) -> None:
    api.reply(
        stream_response(
            [
                message_start(),
                *tool_block(0, "toolu_A", "get_weather", '{"city": "Paris"}'),
                *message_end("tool_use", output_tokens=20),
            ]
        ),
        stream_response(
            [message_start(), *text_block(0, "It is sunny."), *message_end("end_turn", 5)]
        ),
    )
    llm = make_llm(api)
    ctx = user_ctx()
    ctx.add_message("assistant", "")  # the cascade adds the reply item before streaming
    result = await llm.chat(ctx, tools=[get_weather]).collect()
    ctx.items.pop()  # ...and removes it again when no text was produced
    for call in result.tool_calls:
        ctx.append(call)
        ctx.add_function_output(call.call_id, "Sunny, 21 C", name=call.name)

    assert (await llm.chat(ctx, tools=[get_weather]).collect()).text == "It is sunny."
    messages = api.body()["messages"]
    assert messages[1:] == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_A",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_A",
                    "content": "Sunny, 21 C",
                    "cache_control": EPHEMERAL,
                }
            ],
        },
    ]


async def test_parallel_tool_calls_are_emitted_in_order(api: FakeAnthropicAPI) -> None:
    api.reply(
        stream_response(
            [
                message_start(),
                *text_block(0, "Checking both."),
                *tool_block(1, "toolu_paris", "get_weather", '{"city":', ' "Paris"}'),
                *tool_block(2, "toolu_rome", "get_weather", '{"city": "Ro', 'me"}'),
                *message_end("tool_use", output_tokens=120),
            ]
        )
    )
    llm = make_llm(api)

    result = await llm.chat(user_ctx("Paris and Rome?"), tools=[get_weather]).collect()

    assert [(c.call_id, c.parsed_arguments()) for c in result.tool_calls] == [
        ("toolu_paris", {"city": "Paris"}),
        ("toolu_rome", {"city": "Rome"}),
    ]
    assert result.text == "Checking both."


async def test_disabling_parallel_tool_calls(api: FakeAnthropicAPI) -> None:
    api.reply(stream_response([message_start(), *message_end("end_turn", 1)]))
    llm = make_llm(api, parallel_tool_calls=False)
    assert not llm.capabilities.parallel_tool_calls
    await run(llm, user_ctx(), tools=[get_weather])
    assert api.body()["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}


async def test_tool_without_input_gets_empty_object_arguments(api: FakeAnthropicAPI) -> None:
    api.reply(
        stream_response(
            [
                message_start(),
                *tool_block(0, "toolu_now", "get_time", ""),
                *message_end("tool_use", 8),
            ]
        )
    )
    result = await make_llm(api).chat(user_ctx("What time is it?")).collect()
    assert [(c.name, c.arguments) for c in result.tool_calls] == [("get_time", "{}")]


async def test_tool_call_truncated_by_max_tokens_is_dropped(api: FakeAnthropicAPI) -> None:
    api.reply(
        stream_response(
            [
                message_start(),
                *text_block(0, "Let me look that up."),
                *tool_block(1, "toolu_cut", "get_weather", '{"city": "San Fra'),
                *message_end("max_tokens", output_tokens=1024),
            ]
        )
    )
    chunks = await run(make_llm(api), user_ctx(), tools=[get_weather])
    assert not any(c.tool_calls for c in chunks)
    assert chunks[-1].finish_reason == "length"


async def test_thinking_is_not_part_of_the_reply(api: FakeAnthropicAPI) -> None:
    api.reply(
        stream_response(
            [
                message_start(),
                *thinking_block(0, "The user wants a greeting."),
                *text_block(1, "Hello there!"),
                *message_end("end_turn", output_tokens=40),
            ]
        )
    )
    result = await make_llm(api).chat(user_ctx("Hi")).collect()
    assert result.text == "Hello there!"
    assert result.tool_calls == []


async def test_cumulative_usage_from_message_delta_wins(api: FakeAnthropicAPI) -> None:
    api.reply(
        stream_response(
            [
                message_start(input_tokens=10, cache_read=0, cache_write=0),
                *text_block(0, "Hi."),
                *message_end(
                    "end_turn", output_tokens=3, input_tokens=15, cache_read_input_tokens=7
                ),
            ]
        )
    )
    result = await make_llm(api).chat(user_ctx("Hi")).collect()
    assert result.usage == AnthropicUsage(
        prompt_tokens=22, completion_tokens=3, cached_tokens=7, cache_creation_tokens=0
    )


# -------------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("status", "error_type", "expected", "retryable"),
    [
        (400, "invalid_request_error", ProviderError, False),
        (401, "authentication_error", AuthenticationError, False),
        (403, "permission_error", AuthenticationError, False),
        (404, "not_found_error", ProviderError, False),
        (429, "rate_limit_error", RateLimitError, True),
        (500, "api_error", ProviderError, True),
        (529, "overloaded_error", ProviderError, True),
    ],
)
async def test_http_errors_are_mapped(
    api: FakeAnthropicAPI,
    status: int,
    error_type: str,
    expected: type[ProviderError],
    retryable: bool,
) -> None:
    api.reply(error_response(status, error_type, f"{error_type} happened"))
    llm = make_llm(api)
    metrics: list[LLMMetrics] = []
    llm.on("metrics", metrics.append)

    with pytest.raises(expected) as info:
        await run(llm, user_ctx())

    err = info.value
    assert type(err) is expected
    assert (err.provider, err.status_code, err.retryable) == ("anthropic", status, retryable)
    assert f"{error_type} happened" in str(err) and "req_011CErrorTest" in str(err)
    assert "sk-ant-test-key" not in str(err)
    assert metrics[0].error is not None


async def test_non_json_gateway_errors_are_mapped_and_truncated(api: FakeAnthropicAPI) -> None:
    api.reply(http.Response(502, text="<html>" + "bad gateway " * 200 + "</html>"))
    with pytest.raises(ProviderError) as info:
        await run(make_llm(api), user_ctx())
    assert (info.value.status_code, info.value.retryable) == (502, True)
    assert len(str(info.value)) < 600


async def test_error_event_mid_stream_is_mapped_after_the_text_so_far(
    api: FakeAnthropicAPI,
) -> None:
    api.reply(
        stream_response(
            [
                message_start(),
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hel"},
                },
                error_event("overloaded_error", "Overloaded"),
            ]
        )
    )
    received: list[str] = []

    async def consume() -> None:
        async for chunk in make_llm(api).chat(user_ctx("Hi")):
            received.append(chunk.delta)

    with pytest.raises(ProviderError) as info:
        await consume()

    assert received == ["Hel"]
    assert type(info.value) is ProviderError
    assert (info.value.status_code, info.value.retryable) == (529, True)
    assert "overloaded_error" in str(info.value) and "Overloaded" in str(info.value)


async def test_rate_limit_error_event_mid_stream(api: FakeAnthropicAPI) -> None:
    api.reply(stream_response([message_start(), error_event("rate_limit_error", "Slow down")]))
    with pytest.raises(RateLimitError):
        await run(make_llm(api), user_ctx())


async def test_connection_failures_are_mapped(api: FakeAnthropicAPI) -> None:
    def refuse(request: Any) -> Any:
        raise http.ConnectError("connection refused", request=request)

    def time_out(request: Any) -> Any:
        raise http.ReadTimeout("read timed out", request=request)

    async def drop() -> None:
        raise http.RemoteProtocolError("peer closed connection without sending complete body")

    async def stall() -> None:
        raise http.ReadTimeout("no data")

    api.reply(
        refuse,
        time_out,
        stream_response([message_start(), *text_block(0, "Partial")], tail=drop),
        stream_response([message_start()], tail=stall),
    )
    llm = make_llm(api)

    with pytest.raises(ProviderConnectionError) as refused:
        await run(llm, user_ctx())
    with pytest.raises(ProviderTimeoutError):
        await run(llm, user_ctx())
    with pytest.raises(ProviderConnectionError) as dropped:
        await run(llm, user_ctx())
    with pytest.raises(ProviderTimeoutError):
        await run(llm, user_ctx())
    assert refused.value.retryable and dropped.value.retryable


async def test_cancellation_closes_the_http_stream(api: FakeAnthropicAPI) -> None:
    closed = asyncio.Event()

    async def hang() -> None:
        await asyncio.sleep(30)

    api.reply(
        stream_response(
            [message_start(), *text_block(0, "Once upon a time")], tail=hang, closed=closed
        )
    )
    llm = make_llm(api)
    metrics: list[LLMMetrics] = []
    llm.on("metrics", metrics.append)

    stream = llm.chat(user_ctx("Tell me a story"))
    first = await stream.__anext__()
    assert first.delta == "Once upon a time"
    await stream.aclose()

    await asyncio.wait_for(closed.wait(), 5)
    assert metrics[0].cancelled
    assert metrics[0].prompt_tokens == 25  # billed input is still counted after a barge-in


# ------------------------------------------------------------ caching and options


async def test_prompt_caching_can_be_disabled(api: FakeAnthropicAPI) -> None:
    api.reply(stream_response([message_start(), *message_end("end_turn", 1)]))
    await run(make_llm(api, prompt_caching=False), user_ctx(), tools=[get_weather])
    assert "cache_control" not in json.dumps(api.body())


async def test_one_hour_cache_ttl(api: FakeAnthropicAPI) -> None:
    api.reply(stream_response([message_start(), *message_end("end_turn", 1)]))
    await run(make_llm(api, cache_ttl="1h"), user_ctx())
    assert api.body()["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


async def test_cache_breakpoints_skip_late_instructions_and_placeholders(
    api: FakeAnthropicAPI,
) -> None:
    api.reply(stream_response([message_start(), *message_end("end_turn", 1)]))
    ctx = ChatContext()
    ctx.add_message("system", "Stable instructions.")
    ctx.add_message("user", "Hi")
    ctx.add_message("assistant", "Hello! How can I help?")
    ctx.add_message("system", "Ask whether the user is still there.")  # per-response

    await run(make_llm(api), ctx)

    body = api.body()
    assert body["system"] == [
        {"type": "text", "text": "Stable instructions.", "cache_control": EPHEMERAL},
        {"type": "text", "text": "Ask whether the user is still there."},
    ]
    assert body["messages"][-2] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello! How can I help?", "cache_control": EPHEMERAL}],
    }
    assert body["messages"][-1] == {
        "role": "user",
        "content": [{"type": "text", "text": CONTINUE_PLACEHOLDER}],
    }
    assert json.dumps(body).count("cache_control") <= 4  # the API allows 4 breakpoints


async def test_request_options_and_extra_body(api: FakeAnthropicAPI) -> None:
    api.reply(stream_response([message_start(), *message_end("end_turn", 1)]))
    llm = make_llm(
        api,
        model="claude-sonnet-4-6",
        temperature=0.2,
        extra_params={"metadata": {"user_id": "caller-42"}, "top_k": 5},
        extra_headers={"anthropic-beta": "some-beta-2026-01-01"},
    )

    await run(
        llm,
        user_ctx(),
        tools=[get_weather],
        tool_choice="required",
        max_tokens=256,
        extra={"top_k": 3, "stop_sequences": ["END"]},
    )

    body = api.body()
    assert body["model"] == "claude-sonnet-4-6"
    assert body["max_tokens"] == 256
    assert body["temperature"] == 0.2
    assert body["metadata"] == {"user_id": "caller-42"}
    assert body["top_k"] == 3  # per-call extra overrides extra_params
    assert body["stop_sequences"] == ["END"]
    assert body["tool_choice"] == {"type": "any"}
    assert api.requests[-1].headers["anthropic-beta"] == "some-beta-2026-01-01"


async def test_default_tool_choice_and_named_tool(api: FakeAnthropicAPI) -> None:
    api.reply(
        stream_response([message_start(), *message_end("end_turn", 1)]),
        stream_response([message_start(), *message_end("end_turn", 1)]),
    )
    llm = make_llm(api, tool_choice="get_weather")
    await run(llm, user_ctx(), tools=[get_weather])
    assert api.body()["tool_choice"] == {"type": "tool", "name": "get_weather"}
    await run(llm, user_ctx(), tools=[get_weather], tool_choice="none")
    assert api.body()["tool_choice"] == {"type": "none"}


async def test_tool_history_without_tools_sends_stub_definitions(api: FakeAnthropicAPI) -> None:
    api.reply(stream_response([message_start(), *message_end("end_turn", 1)]))
    ctx = user_ctx()
    ctx.add_function_call("get_weather", '{"city": "Paris"}', call_id="toolu_1")
    ctx.add_function_output("toolu_1", "Sunny")

    await run(make_llm(api, prompt_caching=False), ctx)  # e.g. after a handoff to a tool-less agent

    body = api.body()
    assert [t["name"] for t in body["tools"]] == ["get_weather"]
    assert body["tool_choice"] == {"type": "none"}


async def test_untranscribed_audio_fails_the_stream_with_a_clear_error(
    api: FakeAnthropicAPI,
) -> None:
    ctx = ChatContext()
    ctx.add_message("user", AudioContent(AudioFrame.silence(0.2, 16_000)))
    with pytest.raises(ConfigurationError, match="do not accept audio input"):
        await run(make_llm(api), ctx)
    assert api.requests == []


# -------------------------------------------------------------------------- warmup


async def test_warmup_opens_the_connection_with_a_free_request(api: FakeAnthropicAPI) -> None:
    api.reply(
        http.Response(
            200,
            json={
                "id": MODEL,
                "type": "model",
                "display_name": "Claude Haiku 4.5",
                "created_at": "2025-10-15T00:00:00Z",
            },
        )
    )
    await make_llm(api).warmup()
    (request,) = api.requests
    assert (request.method, request.url.path) == ("GET", f"/v1/models/{MODEL}")


async def test_warmup_prewarms_the_prompt_cache(api: FakeAnthropicAPI) -> None:
    api.reply(
        http.Response(
            200,
            json={
                "id": "msg_warm",
                "type": "message",
                "role": "assistant",
                "model": MODEL,
                "content": [],
                "stop_reason": "max_tokens",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 4,
                    "cache_creation_input_tokens": 4200,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 0,
                },
            },
        )
    )
    ctx = ChatContext()
    ctx.add_message("system", "You are a friendly voice assistant.")

    await make_llm(api).warmup(ctx, tools=[get_weather])

    body = api.body()
    assert body["max_tokens"] == 0
    assert not body.get("stream")  # max_tokens=0 is rejected on streaming requests
    assert body["system"][-1]["cache_control"] == EPHEMERAL
    assert body["tools"][-1]["cache_control"] == EPHEMERAL
    # the placeholder user turn is never a cache breakpoint
    assert body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": START_PLACEHOLDER}]}
    ]


async def test_warmup_failures_are_logged_not_raised(
    api: FakeAnthropicAPI, caplog: pytest.LogCaptureFixture
) -> None:
    api.reply(error_response(401, "authentication_error", "invalid x-api-key"))
    with caplog.at_level(logging.WARNING, logger="voice_agent_next"):
        await make_llm(api).warmup()
    assert "warmup failed" in caplog.text and "invalid x-api-key" in caplog.text


# ------------------------------------------------------------------ construction


def test_registry_default_model_metadata_and_capabilities() -> None:
    llm = create("llm", "anthropic", api_key="sk-ant-test-key")
    assert isinstance(llm, AnthropicLLM)
    assert (llm.model, llm.max_tokens, llm.provider) == (MODEL, 1024, "anthropic")
    caps = llm.capabilities
    assert caps.tool_calling and caps.parallel_tool_calls and caps.image_input
    assert not caps.audio_input
    spec = get_provider("llm", "anthropic")
    assert spec.extra == "anthropic" and spec.requires == ("anthropic",) and not spec.local
    assert "ANTHROPIC_API_KEY" in spec.env and spec.aliases == ("claude",)


def test_claude_alias_resolves_in_a_fresh_process() -> None:
    code = (
        "from voice_agent_next import create\n"
        "llm = create('llm', 'claude/claude-sonnet-5', api_key='sk-ant-test-key')\n"
        "print(type(llm).__name__, llm.model)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True
    )
    assert out.stdout.split() == ["AnthropicLLM", "claude-sonnet-5"]


def test_missing_credentials_raise_authentication_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    for var in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_PROFILE",
        "ANTHROPIC_CONFIG_DIR",
        "ANTHROPIC_FEDERATION_RULE_ID",
        "ANTHROPIC_ORGANIZATION_ID",
        "ANTHROPIC_IDENTITY_TOKEN",
        "ANTHROPIC_IDENTITY_TOKEN_FILE",
    ):
        monkeypatch.delenv(var, raising=False)
    for home in ("HOME", "USERPROFILE", "APPDATA"):  # no on-disk `ant auth login` profile
        monkeypatch.setenv(home, str(tmp_path))
    with pytest.raises(AuthenticationError, match="ANTHROPIC_API_KEY"):
        AnthropicLLM()


def test_api_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
    assert AnthropicLLM().client.api_key == "sk-ant-from-env"


def test_option_validation_and_defaults() -> None:
    with pytest.raises(ConfigurationError):
        AnthropicLLM(api_key="k", max_tokens=0)
    with pytest.raises(ConfigurationError):
        AnthropicLLM(api_key="k", cache_ttl="10m")  # type: ignore[arg-type]
    # e.g. `max_tokens: null` in a config file: the API requires a value
    assert AnthropicLLM(api_key="k", max_tokens=None).max_tokens == 1024


async def test_external_client_is_used_but_not_closed_or_probed(api: FakeAnthropicAPI) -> None:
    external = anthropic.AsyncAnthropic(
        api_key="k", http_client=http.AsyncClient(transport=http.MockTransport(api))
    )
    api.reply(
        stream_response([message_start(), *text_block(0, "Hi!"), *message_end("end_turn", 2)])
    )
    borrowed = AnthropicLLM(client=external)
    assert borrowed.client is external
    assert (await borrowed.chat(user_ctx("Hello")).collect()).text == "Hi!"
    await borrowed.warmup()  # e.g. a Bedrock/Vertex client: no /v1/models probe
    await borrowed.aclose()
    assert len(api.requests) == 1
    assert not external.is_closed()
    await external.close()


async def test_aclose_closes_the_owned_client() -> None:
    owned = AnthropicLLM(api_key="sk-ant-test-key")
    async with owned:
        assert not owned.client.is_closed()
    assert owned.client.is_closed()
    await owned.aclose()  # idempotent


# ------------------------------------------------------------- inside the cascade


async def test_cascade_session_greeting_then_tool_round_trip(api: FakeAnthropicAPI) -> None:
    """A real AgentSession + CascadeEngine turn: greeting, speech, tool call, spoken answer."""
    looked_up: list[str] = []

    @function_tool
    async def lookup_weather(city: str) -> str:
        """Weather lookup."""
        looked_up.append(city)
        return f"sunny in {city}"

    api.reply(
        http.Response(  # the session pre-warms the engine: the LLM opens its connection
            200, json={"id": MODEL, "type": "model", "display_name": "Claude", "created_at": ""}
        ),
        stream_response(
            [
                message_start(),
                *text_block(0, "Let me check."),
                *tool_block(1, "toolu_w1", "lookup_weather", '{"city": ', '"Paris"}'),
                *message_end("tool_use", output_tokens=30),
            ]
        ),
        stream_response(
            [message_start(), *text_block(0, "It is sunny in Paris."), *message_end("end_turn", 8)]
        ),
    )
    session = AgentSession(
        stt=MockSTT(transcripts=["weather in paris?"]),
        llm=make_llm(api),
        tts=MockTTS(),
        vad=EnergyVAD(),
        cascade_options=CascadeOptions(min_endpointing_delay=0.0),
    )
    turns: list[TurnMetrics] = []
    session.on("metrics", lambda m: turns.append(m) if isinstance(m, TurnMetrics) else None)
    transport = LoopbackTransport()

    async def wait_for(predicate: Callable[[], bool]) -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await session.start(
        Agent("You are a weather bot.", tools=[lookup_weather], greeting="Welcome."), transport
    )
    await asyncio.wait_for(
        wait_for(
            lambda: session.agent_state == AgentState.LISTENING and bool(transport.played_log)
        ),
        5,
    )
    await transport.play_user_audio(synth_speech(0.8, 16_000), realtime=False)
    await transport.play_user_audio(AudioFrame.silence(0.6, 16_000), realtime=False)
    await asyncio.wait_for(wait_for(lambda: len(turns) == 1), 5)
    await session.aclose()

    assert looked_up == ["Paris"]
    assert (api.requests[0].method, api.requests[0].url.path) == ("GET", f"/v1/models/{MODEL}")
    first, second = api.body(1), api.body(2)
    assert first["system"][0]["text"] == "You are a weather bot."
    assert [t["name"] for t in first["tools"]] == ["lookup_weather"]
    assert [(m["role"], m["content"][0]["text"]) for m in first["messages"]] == [
        ("user", START_PLACEHOLDER),  # the greeting was spoken before any user input
        ("assistant", "Welcome."),
        ("user", "weather in paris?"),
    ]
    assert second["messages"][-2] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me check."},
            {
                "type": "tool_use",
                "id": "toolu_w1",
                "name": "lookup_weather",
                "input": {"city": "Paris"},
            },
        ],
    }
    assert second["messages"][-1] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_w1",
                "content": "sunny in Paris",
                "cache_control": EPHEMERAL,
            }
        ],
    }
    assert turns[0].tool_calls == 1


# --------------------------------------------------------------------- integration


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY")
async def test_real_api_tool_round_trip() -> None:
    llm = AnthropicLLM(model=os.environ.get("ANTHROPIC_TEST_MODEL", MODEL), max_tokens=256)
    ctx = ChatContext()
    ctx.add_message("system", "You are a voice assistant. Use tools when they help.")
    ctx.add_message("user", "What's the weather in Paris right now?")
    try:
        first = await llm.chat(ctx, tools=[get_weather], tool_choice="required").collect()
        assert first.tool_calls and first.tool_calls[0].name == "get_weather"
        assert "paris" in first.tool_calls[0].parsed_arguments()["city"].lower()
        assert first.usage is not None and first.usage.prompt_tokens > 0
        for call in first.tool_calls:
            ctx.append(call)
            ctx.add_function_output(call.call_id, "Sunny, 21 degrees Celsius", name=call.name)
        second = await llm.chat(ctx, tools=[get_weather]).collect()
        assert "21" in second.text or "sunny" in second.text.lower()
    finally:
        await llm.aclose()
