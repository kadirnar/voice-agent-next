"""GeminiLLM against a fake Gemini API (MockTransport replaying real SSE streams).

Needs the SDK (skipped otherwise): ``uv sync --extra google && uv run pytest
tests/providers/test_google_llm.py``; the real-API test additionally needs
``GOOGLE_API_KEY`` or ``GEMINI_API_KEY`` and ``-m integration``.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import httpx
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
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.google._format import START_PLACEHOLDER
from voice_agent_next.providers.google.llm import GeminiLLM, GeminiUsage
from voice_agent_next.providers.mock import MockSTT, MockTTS, synth_speech
from voice_agent_next.registry import get_provider
from voice_agent_next.transports import LoopbackTransport

from .gemini_fake import (
    API_KEY,
    FakeGeminiAPI,
    call,
    chunk,
    error_response,
    model_info,
    stream_response,
    text,
    usage,
)

genai = pytest.importorskip("google.genai")

MODEL = "gemini-3.8-flash"
SIGNATURE = "CiQBjz1rX2abc_-A"  # base64 of an opaque signature, as the API returns it
STREAM_PATH = f"/v1beta/models/{MODEL}:streamGenerateContent"
REAL_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
"""Read at import: the autouse fixture clears the key variables for the offline tests."""


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_GENAI_USE_ENTERPRISE",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def api() -> FakeGeminiAPI:
    return FakeGeminiAPI()


def make_llm(api: FakeGeminiAPI, **kwargs: Any) -> GeminiLLM:
    return GeminiLLM(api_key=API_KEY, http_client=api.http_client(), max_retries=0, **kwargs)


@function_tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny in {city}"


def user_ctx(value: str = "What's the weather in Paris?") -> ChatContext:
    ctx = ChatContext()
    ctx.add_message("system", "You are a friendly voice assistant.")
    ctx.add_message("user", value)
    return ctx


async def run(llm: GeminiLLM, ctx: ChatContext, **kwargs: Any) -> list[ChatChunk]:
    return [c async for c in llm.chat(ctx, **kwargs)]


def done(value: str = "Done.", **counts: int) -> list[dict[str, Any]]:
    return [chunk(text(value), finish="STOP", usage=usage(10, 2, **counts))]


# ---------------------------------------------------------------------- streaming


async def test_streams_text_with_usage_and_sends_the_request(api: FakeGeminiAPI) -> None:
    api.reply(
        stream_response(
            [
                chunk(text("It's"), usage=usage(4130)),
                chunk(text(" sunny in"), usage=usage(4130)),
                chunk(
                    text(" Paris today.", thoughtSignature=SIGNATURE),
                    finish="STOP",
                    usage=usage(4130, 9, thoughtsTokenCount=21, cachedContentTokenCount=4096),
                ),
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
    assert final.usage == GeminiUsage(
        prompt_tokens=4130, completion_tokens=30, cached_tokens=4096, thoughts_tokens=21
    )
    assert isinstance(final.usage, GeminiUsage) and final.usage.uncached_prompt_tokens == 34

    request = api.requests[0]
    assert request.method == "POST" and request.url.path == STREAM_PATH
    assert request.url.params["alt"] == "sse"
    assert request.headers["x-goog-api-key"] == API_KEY
    body = api.body()
    assert body["contents"] == [
        {"role": "user", "parts": [{"text": "What's the weather in Paris?"}]}
    ]
    assert body["systemInstruction"]["parts"] == [{"text": "You are a friendly voice assistant."}]
    (decl,) = body["tools"][0]["functionDeclarations"]
    assert decl["name"] == "get_weather"
    assert decl["description"] == "Get the current weather for a city."
    assert decl["parameters_json_schema"]["properties"] == {"city": {"type": "string"}}
    # Gemini 3.8 Flash has no "minimal" level: "low" is the fastest setting
    assert body["generationConfig"] == {"thinkingConfig": {"thinking_level": "LOW"}}
    assert "toolConfig" not in body

    (m,) = metrics
    assert (m.provider, m.model, m.error, m.cancelled) == ("google", MODEL, None, False)
    assert (m.prompt_tokens, m.completion_tokens, m.cached_tokens) == (4130, 30, 4096)
    assert m.ttft is not None and m.ttft >= 0


async def test_thoughts_are_never_spoken(api: FakeGeminiAPI) -> None:
    api.reply(
        stream_response(
            [
                chunk(text("**Planning** the user wants weather", thought=True)),
                chunk(text("Sunny."), finish="STOP", usage=usage(5, 2, thoughtsTokenCount=40)),
            ]
        )
    )
    result = await make_llm(api, thinking_level="high").chat(user_ctx()).collect()
    assert result.text == "Sunny."
    assert result.usage is not None and result.usage.completion_tokens == 42
    assert api.body()["generationConfig"]["thinkingConfig"] == {"thinking_level": "HIGH"}


async def test_parallel_tool_calls_get_ids_and_round_trip_with_signatures(
    api: FakeGeminiAPI,
) -> None:
    api.reply(
        stream_response(
            [
                chunk(text("Let me check.")),
                chunk(
                    call("get_weather", {"city": "Paris"}, signature=SIGNATURE),
                    call("get_weather", {"city": "Rome"}),
                    finish="STOP",
                    usage=usage(20, 12),
                ),
            ]
        ),
        stream_response(done("Sunny in both.")),
    )
    llm = make_llm(api)
    ctx = user_ctx("Weather in Paris and Rome?")
    first = await llm.chat(ctx, tools=[get_weather]).collect()

    assert first.text == "Let me check."
    assert [c.parsed_arguments() for c in first.tool_calls] == [
        {"city": "Paris"},
        {"city": "Rome"},
    ]
    ids = [c.call_id for c in first.tool_calls]
    assert len(set(ids)) == 2 and all(i.startswith("call_") for i in ids)

    ctx.add_message("assistant", first.text)
    for c in first.tool_calls:
        ctx.append(c)
    for c in first.tool_calls:
        ctx.add_function_output(c.call_id, f"sunny in {c.parsed_arguments()['city']}", name=c.name)
    second = await llm.chat(ctx, tools=[get_weather]).collect()
    assert second.text == "Sunny in both."

    contents = api.body()["contents"]
    assert contents[1] == {
        "role": "model",
        "parts": [
            {"text": "Let me check."},
            {
                "functionCall": {"name": "get_weather", "args": {"city": "Paris"}},
                "thoughtSignature": SIGNATURE,
            },
            {"functionCall": {"name": "get_weather", "args": {"city": "Rome"}}},
        ],
    }
    assert contents[2] == {
        "role": "user",
        "parts": [
            {
                "functionResponse": {
                    "name": "get_weather",
                    "response": {"output": "sunny in Paris"},
                }
            },
            {
                "functionResponse": {
                    "name": "get_weather",
                    "response": {"output": "sunny in Rome"},
                }
            },
        ],
    }


async def test_api_call_ids_are_kept_and_echoed(api: FakeGeminiAPI) -> None:
    api.reply(
        stream_response([chunk(call("get_weather", {"city": "Oslo"}, id="fc-7"), finish="STOP")]),
        stream_response(done()),
    )
    llm = make_llm(api)
    ctx = user_ctx()
    first = await llm.chat(ctx, tools=[get_weather]).collect()
    assert first.tool_calls[0].call_id == "fc-7"
    ctx.append(first.tool_calls[0])
    ctx.add_function_output("fc-7", "rain", name="get_weather")
    await llm.chat(ctx, tools=[get_weather]).collect()
    model, responses = api.body()["contents"][1:]
    assert model["parts"][0]["functionCall"]["id"] == "fc-7"
    assert responses["parts"][0]["functionResponse"]["id"] == "fc-7"


async def test_foreign_tool_history_gets_the_skip_signature(api: FakeGeminiAPI) -> None:
    api.reply(stream_response(done()), stream_response(done()))
    ctx = user_ctx()
    ctx.add_function_call("get_weather", '{"city": "Paris"}', call_id="toolu_01")
    ctx.add_function_output("toolu_01", "sunny", name="get_weather")
    await run(make_llm(api), ctx, tools=[get_weather])
    part = api.body()["contents"][1]["parts"][0]
    assert part["thoughtSignature"] == "skip_thought_signature_validator"

    await run(make_llm(api, model="gemini-2.5-flash"), ctx, tools=[get_weather])
    assert "thoughtSignature" not in api.body()["contents"][1]["parts"][0]


@pytest.mark.parametrize(
    ("finish", "expected"),
    [
        ("MAX_TOKENS", "length"),
        ("SAFETY", "content_filter"),
        ("RECITATION", "content_filter"),
        ("MALFORMED_FUNCTION_CALL", "malformed_function_call"),
    ],
)
async def test_finish_reasons(api: FakeGeminiAPI, finish: str, expected: str) -> None:
    api.reply(stream_response([chunk(text("Partial"), finish=finish, usage=usage(5, 1))]))
    chunks = await run(make_llm(api), user_ctx())
    assert chunks[-1].finish_reason == expected


async def test_tool_call_finish_reason_and_blocked_prompt(api: FakeGeminiAPI) -> None:
    api.reply(
        stream_response([chunk(call("get_weather", {"city": "Paris"}), finish="STOP")]),
        stream_response(
            [
                {
                    "promptFeedback": {"blockReason": "PROHIBITED_CONTENT"},
                    "usageMetadata": usage(7),
                    "modelVersion": MODEL,
                }
            ]
        ),
    )
    llm = make_llm(api)
    assert (await run(llm, user_ctx(), tools=[get_weather]))[-1].finish_reason == "tool_calls"
    blocked = await run(llm, user_ctx())
    assert blocked[-1].finish_reason == "content_filter"
    assert blocked[-1].usage is not None and blocked[-1].usage.prompt_tokens == 7


# ------------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("status", "rpc", "reason", "expected", "retryable"),
    [
        (400, "INVALID_ARGUMENT", "API_KEY_INVALID", AuthenticationError, False),
        (403, "PERMISSION_DENIED", None, AuthenticationError, False),
        (429, "RESOURCE_EXHAUSTED", None, RateLimitError, True),
        (500, "INTERNAL", None, ProviderError, True),
        (503, "UNAVAILABLE", None, ProviderError, True),
        (504, "DEADLINE_EXCEEDED", None, ProviderTimeoutError, True),
        (404, "NOT_FOUND", None, ProviderError, False),
        (400, "INVALID_ARGUMENT", None, ProviderError, False),
    ],
)
async def test_http_errors_are_mapped(
    api: FakeGeminiAPI,
    status: int,
    rpc: str,
    reason: str | None,
    expected: type[ProviderError],
    retryable: bool,
) -> None:
    message = "API key not valid. Please pass a valid API key." if reason else f"{rpc} happened"
    api.reply(error_response(status, rpc, message, reason=reason))
    llm = make_llm(api)
    metrics: list[LLMMetrics] = []
    llm.on("metrics", metrics.append)

    with pytest.raises(expected) as info:
        await run(llm, user_ctx())

    err = info.value
    assert type(err) is expected
    assert (err.provider, err.status_code, err.retryable) == ("google", status, retryable)
    assert message in str(err) and rpc in str(err)
    assert API_KEY not in str(err)
    if status == 404:
        assert "check the model id" in str(err)
    assert metrics[0].error is not None


async def test_connection_failures_are_mapped(api: FakeGeminiAPI) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    def time_out(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=request)

    async def drop() -> None:
        raise httpx.RemoteProtocolError("peer closed connection without sending complete body")

    async def stall() -> None:
        raise httpx.ReadTimeout("no data")

    api.reply(
        refuse,
        time_out,
        stream_response([chunk(text("Partial"))], tail=drop),
        stream_response([chunk(text("Partial"))], tail=stall),
    )
    llm = make_llm(api)
    received: list[str] = []

    async def consume() -> None:
        async for c in llm.chat(user_ctx()):
            received.append(c.delta)

    with pytest.raises(ProviderConnectionError) as refused:
        await consume()
    with pytest.raises(ProviderTimeoutError):
        await consume()
    with pytest.raises(ProviderConnectionError) as dropped:
        await consume()
    with pytest.raises(ProviderTimeoutError):
        await consume()
    assert refused.value.retryable and dropped.value.retryable
    assert received == ["Partial", "Partial"]


async def test_retries_before_the_stream_starts(api: FakeGeminiAPI) -> None:
    api.reply(error_response(503, "UNAVAILABLE", "overloaded"), stream_response(done("Hi")))
    llm = GeminiLLM(api_key=API_KEY, http_client=api.http_client(), max_retries=1)
    assert (await llm.chat(user_ctx()).collect()).text == "Hi"
    assert len(api.requests) == 2


async def test_cancellation_closes_the_http_stream(api: FakeGeminiAPI) -> None:
    closed = asyncio.Event()

    async def hang() -> None:
        await asyncio.sleep(30)

    api.reply(
        stream_response(
            [chunk(text("Once upon a time"), usage=usage(25))], tail=hang, closed=closed
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


async def test_untranscribed_audio_fails_with_a_clear_error(api: FakeGeminiAPI) -> None:
    ctx = ChatContext()
    ctx.add_message("user", AudioContent(AudioFrame.silence(0.2, 16_000)))
    with pytest.raises(ConfigurationError, match="audio input is disabled"):
        await run(make_llm(api, audio_input=False), ctx)
    assert api.requests == []


async def test_user_audio_is_sent_inline(api: FakeGeminiAPI) -> None:
    api.reply(stream_response(done()))
    ctx = ChatContext()
    ctx.add_message("user", AudioContent(AudioFrame.silence(0.2, 16_000), transcript="hi"))
    await run(make_llm(api), ctx)
    (part,) = api.body()["contents"][0]["parts"]
    blob = part["inlineData"]
    # the SDK passes dict keys through as given; the API accepts either spelling
    assert blob.get("mimeType", blob.get("mime_type")) == "audio/wav"
    assert base64.b64decode(blob["data"])[:4] == b"RIFF"


# ------------------------------------------------------------------------ options


async def test_request_options(api: FakeGeminiAPI) -> None:
    api.reply(stream_response(done()), stream_response(done()), stream_response(done()))
    llm = make_llm(
        api,
        temperature=0.4,
        max_tokens=256,
        tool_choice="required",
        extra_config={
            "top_p": 0.9,
            "thinking_config": {"include_thoughts": False},
            "tools": [{"google_search": {}}],
        },
    )
    await run(llm, user_ctx(), tools=[get_weather], extra={"seed": 7})
    config = api.body()["generationConfig"]
    assert config["temperature"] == 0.4
    assert config["maxOutputTokens"] == 256
    assert config["topP"] == 0.9 and config["seed"] == 7
    assert config["thinkingConfig"] == {"thinking_level": "LOW", "include_thoughts": False}
    body = api.body()
    assert body["toolConfig"] == {"functionCallingConfig": {"mode": "ANY"}}
    assert {"googleSearch": {}} in body["tools"]
    assert any("functionDeclarations" in tool for tool in body["tools"])

    await run(llm, user_ctx(), tools=[get_weather], tool_choice="get_weather", temperature=0.1)
    assert api.body()["toolConfig"]["functionCallingConfig"] == {
        "mode": "ANY",
        "allowedFunctionNames": ["get_weather"],
    }
    assert api.body()["generationConfig"]["temperature"] == 0.1

    cached = make_llm(api, cached_content="cachedContents/abc123")
    await run(cached, user_ctx(), tools=[get_weather])
    body = api.body()
    assert body["cachedContent"] == "cachedContents/abc123"
    assert "systemInstruction" not in body and "tools" not in body


@pytest.mark.parametrize(
    ("model", "kwargs", "expected"),
    [
        ("gemini-3.6-flash", {}, {"thinking_level": "MINIMAL"}),
        ("gemini-2.5-flash", {}, {"thinking_budget": 0}),
        (MODEL, {"thinking_level": None}, None),
        ("gemini-2.5-pro", {"thinking_budget": -1}, {"thinking_budget": -1}),
    ],
)
async def test_thinking_settings(
    api: FakeGeminiAPI, model: str, kwargs: dict[str, Any], expected: dict[str, Any] | None
) -> None:
    api.reply(stream_response(done()))
    await run(make_llm(api, model=model, **kwargs), user_ctx())
    assert api.body().get("generationConfig", {}).get("thinkingConfig") == expected


def test_option_validation() -> None:
    with pytest.raises(ConfigurationError, match="thinking_level"):
        GeminiLLM(api_key=API_KEY, thinking_level="extreme")
    with pytest.raises(ConfigurationError, match="either"):
        GeminiLLM(api_key=API_KEY, thinking_level="low", thinking_budget=100)
    with pytest.raises(ConfigurationError, match="max_tokens"):
        GeminiLLM(api_key=API_KEY, max_tokens=0)
    with pytest.raises(ConfigurationError, match="max_retries"):
        GeminiLLM(api_key=API_KEY, max_retries=-1)
    with pytest.raises(ConfigurationError, match="base64"):
        GeminiLLM(api_key=API_KEY, fallback_thought_signature="not base64!")
    llm = GeminiLLM(api_key=API_KEY)
    assert llm.model == MODEL
    assert llm.capabilities.audio_input and llm.capabilities.image_input
    assert llm.capabilities.tool_calling and llm.capabilities.parallel_tool_calls


# -------------------------------------------------------------------- credentials


def test_missing_credentials_raise_authentication_error() -> None:
    with pytest.raises(AuthenticationError, match="GOOGLE_API_KEY"):
        GeminiLLM()


@pytest.mark.parametrize("env", ["GOOGLE_API_KEY", "GEMINI_API_KEY"])
async def test_api_key_from_environment(
    api: FakeGeminiAPI, monkeypatch: pytest.MonkeyPatch, env: str
) -> None:
    monkeypatch.setenv(env, "env-key")
    api.reply(stream_response(done()))
    llm = GeminiLLM(http_client=api.http_client(), max_retries=0)
    await run(llm, user_ctx())
    assert api.requests[0].headers["x-goog-api-key"] == "env-key"
    assert not llm.vertexai


async def test_vertex_ai_express_mode(api: FakeGeminiAPI) -> None:
    api.reply(stream_response(done()))
    llm = GeminiLLM(vertexai=True, api_key="vertex-key", http_client=api.http_client())
    assert llm.vertexai
    await run(llm, user_ctx())
    url = api.requests[0].url
    assert url.host == "aiplatform.googleapis.com"
    assert url.path.endswith(f"publishers/google/models/{MODEL}:streamGenerateContent")


def test_vertex_settings_select_vertex_ai(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigurationError, match="Vertex AI settings"):
        GeminiLLM(api_key=API_KEY, vertexai=False, project="my-project")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    llm = GeminiLLM(api_key="vertex-key")
    assert llm.vertexai


# ---------------------------------------------------------------------- lifecycle


async def test_warmup_opens_the_connection_with_a_free_request(api: FakeGeminiAPI) -> None:
    api.reply(model_info(MODEL))
    await make_llm(api).warmup()
    request = api.requests[0]
    assert (request.method, request.url.path) == ("GET", f"/v1beta/models/{MODEL}")


async def test_warmup_failures_are_logged_not_raised(
    api: FakeGeminiAPI, caplog: pytest.LogCaptureFixture
) -> None:
    api.reply(
        error_response(400, "INVALID_ARGUMENT", "API key not valid.", reason="API_KEY_INVALID")
    )
    with caplog.at_level(logging.WARNING):
        await make_llm(api).warmup()
    assert "Gemini warmup failed" in caplog.text and "API key not valid" in caplog.text


async def test_external_client_is_used_but_not_closed(api: FakeGeminiAPI) -> None:
    external = genai.Client(
        api_key="k",
        http_options=genai.types.HttpOptions(httpx_async_client=api.http_client()),
    )
    api.reply(stream_response(done("Hi!")))
    borrowed = GeminiLLM(client=external)
    assert borrowed.client is external
    assert (await borrowed.chat(user_ctx("Hello")).collect()).text == "Hi!"
    await borrowed.aclose()
    api.reply(stream_response(done("Still open")))
    assert (await borrowed.chat(user_ctx("Hello")).collect()).text == "Still open"


async def test_aclose_is_idempotent() -> None:
    owned = GeminiLLM(api_key=API_KEY)
    async with owned:
        pass
    await owned.aclose()


def test_registry_metadata_and_gemini_alias() -> None:
    spec = get_provider("llm", "google")
    assert spec.default_model == MODEL
    assert spec.env == ("GOOGLE_API_KEY", "GEMINI_API_KEY")
    assert spec.extra == "google"
    llm = create("llm", "gemini/gemini-3.5-flash-lite", api_key=API_KEY)
    assert isinstance(llm, GeminiLLM) and llm.model == "gemini-3.5-flash-lite"


def test_gemini_alias_resolves_in_a_fresh_process() -> None:
    code = (
        "from voice_agent_next import create\n"
        "llm = create('llm', 'gemini/gemini-3.8-flash', api_key='k')\n"
        "tts = create('tts', 'gemini', api_key='k')\n"
        "print(type(llm).__name__, llm.model, type(tts).__name__, tts.model)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60
    )
    assert out.stdout.split() == ["GeminiLLM", MODEL, "GeminiTTS", "gemini-3.8-flash-tts"]


# ------------------------------------------------------------- inside the cascade


async def test_cascade_session_greeting_then_tool_round_trip(api: FakeGeminiAPI) -> None:
    """A real AgentSession + CascadeEngine turn: greeting, speech, tool call, spoken answer."""
    looked_up: list[str] = []

    @function_tool
    async def lookup_weather(city: str) -> str:
        """Weather lookup."""
        looked_up.append(city)
        return f"sunny in {city}"

    api.reply(
        model_info(MODEL),  # the session pre-warms the engine: the LLM opens its connection
        stream_response(
            [
                chunk(text("Let me check.")),
                chunk(
                    call("lookup_weather", {"city": "Paris"}, signature=SIGNATURE),
                    finish="STOP",
                    usage=usage(30, 12),
                ),
            ]
        ),
        stream_response(done("It is sunny in Paris.")),
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
    assert (api.requests[0].method, api.requests[0].url.path) == (
        "GET",
        f"/v1beta/models/{MODEL}",
    )
    first, second = api.body(1), api.body(2)
    assert first["systemInstruction"]["parts"][0]["text"] == "You are a weather bot."
    assert [(c["role"], c["parts"][0]["text"]) for c in first["contents"]] == [
        ("user", START_PLACEHOLDER),  # the greeting was spoken before any user input
        ("model", "Welcome."),
        ("user", "weather in paris?"),
    ]
    assert second["contents"][-2] == {
        "role": "model",
        "parts": [
            {"text": "Let me check."},
            {
                "functionCall": {"name": "lookup_weather", "args": {"city": "Paris"}},
                "thoughtSignature": SIGNATURE,
            },
        ],
    }
    assert second["contents"][-1] == {
        "role": "user",
        "parts": [
            {
                "functionResponse": {
                    "name": "lookup_weather",
                    "response": {"output": "sunny in Paris"},
                }
            }
        ],
    }
    assert turns[0].tool_calls == 1


# --------------------------------------------------------------------- integration


@pytest.mark.integration
@pytest.mark.skipif(not REAL_KEY, reason="needs GOOGLE_API_KEY or GEMINI_API_KEY")
async def test_real_api_tool_round_trip() -> None:
    llm = GeminiLLM(
        model=os.environ.get("GEMINI_TEST_MODEL", MODEL), api_key=REAL_KEY, max_tokens=512
    )
    ctx = ChatContext()
    ctx.add_message("system", "You are a voice assistant. Use tools when they help.")
    ctx.add_message("user", "What's the weather in Paris right now?")
    try:
        await llm.warmup()
        first = await llm.chat(ctx, tools=[get_weather], tool_choice="required").collect()
        assert first.tool_calls and first.tool_calls[0].name == "get_weather"
        assert "paris" in first.tool_calls[0].parsed_arguments()["city"].lower()
        assert first.usage is not None and first.usage.prompt_tokens > 0
        for c in first.tool_calls:
            ctx.append(c)
            ctx.add_function_output(c.call_id, "Sunny, 21 degrees Celsius", name=c.name)
        second = await llm.chat(ctx, tools=[get_weather]).collect()
        assert "21" in second.text or "sunny" in second.text.lower()
    finally:
        await llm.aclose()
