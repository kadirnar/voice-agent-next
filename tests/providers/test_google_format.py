"""ChatContext -> Gemini ``generateContent`` conversion (pure functions, no SDK needed)."""

from __future__ import annotations

import base64
from typing import Any

import pytest

from voice_agent_next import AudioFrame, ChatContext, function_tool
from voice_agent_next.chat import AudioContent, FunctionCall, ImageContent
from voice_agent_next.errors import ConfigurationError
from voice_agent_next.providers.google._format import (
    CONTINUE_PLACEHOLDER,
    START_PLACEHOLDER,
    CallMeta,
    default_thinking_config,
    gemini_generation,
    to_gemini_contents,
    to_gemini_tool_config,
    to_gemini_tools,
)

SIG = b"\x0a\x24\x01sig"


def text_parts(content: dict[str, Any]) -> list[str]:
    return [p["text"] for p in content["parts"] if "text" in p]


def test_system_roles_and_merging() -> None:
    ctx = ChatContext()
    ctx.add_message("system", "You are terse.")
    ctx.add_message("developer", "Speak English.")
    ctx.add_message("user", "Hi")
    ctx.add_message("user", "there")
    ctx.add_message("assistant", "Hello!")
    ctx.add_message("user", "Bye")
    ctx.add_message("system", "Now say goodbye.")  # a late, per-response instruction
    prompt = to_gemini_contents(ctx)
    assert prompt.system_instruction == "You are terse.\n\nSpeak English.\n\nNow say goodbye."
    assert [c["role"] for c in prompt.contents] == ["user", "model", "user"]
    assert text_parts(prompt.contents[0]) == ["Hi", "there"]
    assert text_parts(prompt.contents[1]) == ["Hello!"]


def test_placeholders_for_model_first_and_model_last_conversations() -> None:
    ctx = ChatContext()
    ctx.add_message("assistant", "Welcome.")
    prompt = to_gemini_contents(ctx)
    assert [(c["role"], text_parts(c)) for c in prompt.contents] == [
        ("user", [START_PLACEHOLDER]),
        ("model", ["Welcome."]),
        ("user", [CONTINUE_PLACEHOLDER]),
    ]
    empty = to_gemini_contents(ChatContext())
    assert empty.contents == [{"role": "user", "parts": [{"text": START_PLACEHOLDER}]}]
    assert empty.system_instruction is None


def test_interrupted_assistant_message_is_sent_as_heard() -> None:
    ctx = ChatContext()
    ctx.add_message("user", "Tell me a story")
    ctx.add_message("assistant", "Once upon a", interrupted=True)
    ctx.add_message("user", "Stop, what time is it?")
    prompt = to_gemini_contents(ctx)
    assert text_parts(prompt.contents[1]) == ["Once upon a"]


def test_function_calls_and_responses_pair_up() -> None:
    ctx = ChatContext()
    ctx.add_message("user", "Weather in Paris and Rome?")
    ctx.add_message("assistant", "Let me check.")
    ctx.add_function_call("get_weather", '{"city": "Paris"}', call_id="c1")
    ctx.add_function_call("get_weather", '{"city": "Rome"}', call_id="c2")
    ctx.add_function_output("c1", '{"temp": 21}', name="get_weather")
    ctx.add_function_output("c2", "boom", name="get_weather", is_error=True)
    ctx.add_function_call("orphan", "{}", call_id="c3")  # no output: dropped
    ctx.add_function_output("c4", "no call", name="x")  # no call: dropped
    calls = {"c1": CallMeta(signature=SIG, api_id=True)}
    prompt = to_gemini_contents(ctx, calls=calls)
    _user, model, responses = prompt.contents
    assert model == {
        "role": "model",
        "parts": [
            {"text": "Let me check."},
            {
                "function_call": {"name": "get_weather", "args": {"city": "Paris"}, "id": "c1"},
                "thought_signature": SIG,
            },
            {"function_call": {"name": "get_weather", "args": {"city": "Rome"}}},
        ],
    }
    assert responses == {
        "role": "user",
        "parts": [
            {
                "function_response": {
                    "name": "get_weather",
                    "response": {"output": {"temp": 21}},
                    "id": "c1",
                }
            },
            {"function_response": {"name": "get_weather", "response": {"error": "boom"}}},
        ],
    }
    assert prompt.tool_names == {"get_weather"}


def test_fallback_signature_only_for_the_current_turn() -> None:
    ctx = ChatContext()
    ctx.add_message("user", "old question")
    ctx.add_function_call("lookup", "{}", call_id="old")
    ctx.add_function_output("old", "x")
    ctx.add_message("assistant", "old answer")
    ctx.add_message("user", "new question")
    ctx.add_function_call("lookup", "{}", call_id="a")
    ctx.add_function_call("lookup", "{}", call_id="b")
    ctx.add_function_output("a", "1")
    ctx.add_function_output("b", "2")
    ctx.add_function_call("lookup", "{}", call_id="c")
    ctx.add_function_output("c", "3")
    prompt = to_gemini_contents(ctx, fallback_signature=b"skip")
    model_steps = [c for c in prompt.contents if c["role"] == "model"]
    old_call, old_answer, step1, step2 = model_steps
    assert all("thought_signature" not in p for p in old_call["parts"] + old_answer["parts"])
    assert step1["parts"][0]["thought_signature"] == b"skip"
    assert "thought_signature" not in step1["parts"][1]  # only the first parallel call
    assert step2["parts"][0]["thought_signature"] == b"skip"


def test_user_audio_and_images() -> None:
    frame = AudioFrame.silence(0.1, 16_000)
    png = base64.b64encode(b"\x89PNG fake").decode()
    ctx = ChatContext()
    ctx.add_message(
        "user",
        [
            "Look:",
            ImageContent(f"data:image/png;base64,{png}"),
            ImageContent("https://example.com/cat.webp"),
            ImageContent("gs://bucket/dog", mime_type="image/jpeg"),
            AudioContent(frame, transcript="what is this"),
        ],
    )
    parts = to_gemini_contents(ctx).contents[0]["parts"]
    assert parts[0] == {"text": "Look:"}
    assert parts[1] == {"inline_data": {"mime_type": "image/png", "data": b"\x89PNG fake"}}
    assert parts[2] == {
        "file_data": {"file_uri": "https://example.com/cat.webp", "mime_type": "image/webp"}
    }
    assert parts[3] == {"file_data": {"file_uri": "gs://bucket/dog", "mime_type": "image/jpeg"}}
    audio = parts[4]["inline_data"]
    assert audio["mime_type"] == "audio/wav" and audio["data"][:4] == b"RIFF"

    transcripts = to_gemini_contents(ctx, audio_input=False).contents[0]["parts"]
    assert transcripts[-1] == {"text": "what is this"}

    bare = ChatContext()
    bare.add_message("user", AudioContent(frame))
    with pytest.raises(ConfigurationError, match="audio without a transcript"):
        to_gemini_contents(bare, audio_input=False)
    bad = ChatContext()
    bad.add_message("user", ImageContent("ftp://x/y.png"))
    with pytest.raises(ConfigurationError, match="unsupported image URL"):
        to_gemini_contents(bad)


def test_invalid_tool_arguments_are_sent_as_an_empty_object() -> None:
    ctx = ChatContext()
    ctx.add_message("user", "go")
    ctx.append(FunctionCall(name="f", arguments="{not json", call_id="x"))
    ctx.add_function_output("x", "ok")
    model = to_gemini_contents(ctx).contents[1]
    assert model["parts"][0]["function_call"]["args"] == {}


def test_tools_and_tool_choice() -> None:
    @function_tool
    async def get_weather(city: str) -> str:
        """Get the weather."""
        return city

    @function_tool
    async def ping() -> str:
        """Ping."""
        return "pong"

    tools = to_gemini_tools([get_weather, ping])
    decls = tools[0]["function_declarations"]
    assert decls[0]["name"] == "get_weather"
    assert decls[0]["description"] == "Get the weather."
    assert decls[0]["parameters_json_schema"]["properties"]["city"]["type"] == "string"
    assert "parameters_json_schema" not in decls[1]  # no parameters at all
    assert to_gemini_tools([]) == []

    assert to_gemini_tool_config(None) is None
    assert to_gemini_tool_config("auto") is None
    assert to_gemini_tool_config("none") == {"function_calling_config": {"mode": "NONE"}}
    assert to_gemini_tool_config("required") == {"function_calling_config": {"mode": "ANY"}}
    assert to_gemini_tool_config("ping", tool_names=["ping"]) == {
        "function_calling_config": {"mode": "ANY", "allowed_function_names": ["ping"]}
    }
    with pytest.raises(ConfigurationError):
        to_gemini_tool_config("nope", tool_names=["ping"])


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gemini-3.8-flash", {"thinking_level": "low"}),
        ("models/gemini-3.7-flash", {"thinking_level": "low"}),
        ("gemini-3.6-flash", {"thinking_level": "minimal"}),
        ("gemini-3.5-flash-lite", {"thinking_level": "minimal"}),
        ("gemini-3.1-flash-lite", {"thinking_level": "minimal"}),
        ("gemini-3-flash-preview", {"thinking_level": "minimal"}),
        ("gemini-3.1-pro-preview", {"thinking_level": "low"}),
        ("gemini-2.5-flash", {"thinking_budget": 0}),
        ("gemini-2.5-flash-lite", {"thinking_budget": 0}),
        ("gemini-2.5-pro", {"thinking_budget": 128}),
        ("gemma-4-27b-it", None),
    ],
)
def test_default_thinking_config(model: str, expected: dict[str, Any] | None) -> None:
    assert default_thinking_config(model) == expected


def test_gemini_generation() -> None:
    assert gemini_generation("gemini-3.8-flash") == (3, 8)
    assert gemini_generation("gemini-3-flash-preview") == (3, 0)
    assert gemini_generation("publishers/google/models/gemini-2.5-pro") == (2, 5)
    assert gemini_generation("gemma-3") is None
