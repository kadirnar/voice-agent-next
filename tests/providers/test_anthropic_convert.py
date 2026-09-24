"""ChatContext -> Anthropic Messages API conversion (no SDK or network needed)."""

from __future__ import annotations

import base64
from typing import Annotated

import pytest

from voice_agent_next import AudioFrame, ChatContext
from voice_agent_next.chat import AudioContent, FunctionCall, FunctionCallOutput, ImageContent
from voice_agent_next.errors import ConfigurationError
from voice_agent_next.providers.anthropic import (
    CONTINUE_PLACEHOLDER,
    START_PLACEHOLDER,
    to_anthropic_messages,
    to_anthropic_tool_choice,
    to_anthropic_tools,
)
from voice_agent_next.tools import FunctionTool, function_tool


def text(t: str) -> dict[str, str]:
    return {"type": "text", "text": t}


def audio(transcript: str | None) -> AudioContent:
    return AudioContent(AudioFrame.silence(0.1, 16_000), transcript)


# ------------------------------------------------------------------ system / roles


def test_system_and_developer_messages_become_system_blocks() -> None:
    ctx = ChatContext()
    ctx.add_message("system", "You are a voice assistant.")
    ctx.add_message("developer", "Keep answers short.")
    ctx.add_message("user", "Hi!")
    ctx.add_message("system", "Answer in French.")  # per-response instructions come last

    prompt = to_anthropic_messages(ctx)

    assert prompt.system == [
        text("You are a voice assistant."),
        text("Keep answers short."),
        text("Answer in French."),
    ]
    assert prompt.stable_system_blocks == 2  # only the leading instructions are cached
    assert prompt.messages == [{"role": "user", "content": [text("Hi!")]}]


def test_consecutive_same_role_messages_are_merged() -> None:
    ctx = ChatContext()
    ctx.add_message("user", "Hello.")
    ctx.add_message("user", ["Are you ", "there?"])
    ctx.add_message("assistant", "Yes.")
    ctx.add_message("assistant", "How can I help?")
    ctx.add_message("user", "Weather?")

    assert to_anthropic_messages(ctx).messages == [
        {"role": "user", "content": [text("Hello."), text("Are you there?")]},
        {"role": "assistant", "content": [text("Yes."), text("How can I help?")]},
        {"role": "user", "content": [text("Weather?")]},
    ]


def test_interrupted_assistant_message_keeps_only_the_heard_text() -> None:
    ctx = ChatContext()
    ctx.add_message("user", "Tell me about Paris.")
    ctx.add_message("assistant", "Paris is the capital of", interrupted=True)
    ctx.add_message("user", "Stop, what about Rome?")
    ctx.add_message("assistant", "", interrupted=True)  # cut off before anything was heard
    ctx.add_message("user", "Hello?")

    assert to_anthropic_messages(ctx).messages == [
        {"role": "user", "content": [text("Tell me about Paris.")]},
        {"role": "assistant", "content": [text("Paris is the capital of")]},
        {"role": "user", "content": [text("Stop, what about Rome?"), text("Hello?")]},
    ]


def test_conversation_starting_with_the_agent_gets_a_placeholder_user_turn() -> None:
    ctx = ChatContext()
    ctx.add_message("system", "Be nice.")
    ctx.add_message("assistant", "Hi! How can I help?")  # greeting spoken first
    ctx.add_message("user", "What time is it?")

    assert to_anthropic_messages(ctx).messages == [
        {"role": "user", "content": [text(START_PLACEHOLDER)]},
        {"role": "assistant", "content": [text("Hi! How can I help?")]},
        {"role": "user", "content": [text("What time is it?")]},
    ]


def test_empty_conversation_gets_a_placeholder_user_turn() -> None:
    ctx = ChatContext()
    ctx.add_message("system", "Greet the caller.")
    prompt = to_anthropic_messages(ctx)
    assert prompt.messages == [{"role": "user", "content": [text(START_PLACEHOLDER)]}]
    assert not prompt.ends_with_placeholder


def test_trailing_assistant_turn_is_followed_by_a_continue_turn() -> None:
    ctx = ChatContext()
    ctx.add_message("user", "Hi")
    ctx.add_message("assistant", "Hello! Anything else?")
    prompt = to_anthropic_messages(ctx)
    assert prompt.messages[-1] == {"role": "user", "content": [text(CONTINUE_PLACEHOLDER)]}
    assert prompt.ends_with_placeholder


def test_empty_assistant_message_being_generated_is_ignored() -> None:
    # The cascade adds an empty assistant item for the reply before calling the LLM.
    ctx = ChatContext()
    ctx.add_message("user", "Hi")
    ctx.add_message("assistant", "")
    prompt = to_anthropic_messages(ctx)
    assert prompt.messages == [{"role": "user", "content": [text("Hi")]}]
    assert not prompt.ends_with_placeholder


# ------------------------------------------------------------------------- tools


def test_tool_call_and_output_round_trip() -> None:
    ctx = ChatContext()
    ctx.add_message("user", "Weather in Paris?")
    ctx.add_message("assistant", "Let me check.")
    ctx.add_function_call("get_weather", '{"city": "Paris"}', call_id="toolu_01")
    ctx.add_function_output("toolu_01", "Sunny, 21 C", name="get_weather")
    ctx.add_message("assistant", "It is sunny.")
    ctx.add_message("user", "Thanks!")

    prompt = to_anthropic_messages(ctx)

    assert prompt.messages == [
        {"role": "user", "content": [text("Weather in Paris?")]},
        {
            "role": "assistant",
            "content": [
                text("Let me check."),
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_01", "content": "Sunny, 21 C"}
            ],
        },
        {"role": "assistant", "content": [text("It is sunny.")]},
        {"role": "user", "content": [text("Thanks!")]},
    ]
    assert prompt.tool_names == {"get_weather"}


def test_parallel_tool_calls_share_one_assistant_and_one_user_turn() -> None:
    ctx = ChatContext()
    ctx.add_message("user", "Weather in Paris and Rome?")
    ctx.add_function_call("get_weather", '{"city": "Paris"}', call_id="toolu_a")
    ctx.add_function_call("get_weather", '{"city": "Rome"}', call_id="toolu_b")
    ctx.add_function_output("toolu_a", "Service down", is_error=True)
    ctx.add_function_output("toolu_b", "Cloudy")

    messages = to_anthropic_messages(ctx).messages

    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert [b["id"] for b in messages[1]["content"]] == ["toolu_a", "toolu_b"]
    assert messages[2]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_a",
            "content": "Service down",
            "is_error": True,
        },
        {"type": "tool_result", "tool_use_id": "toolu_b", "content": "Cloudy"},
    ]


def test_sequential_tool_rounds_stay_separate_turns() -> None:
    ctx = ChatContext()
    ctx.add_message("user", "Book the cheapest flight.")
    ctx.add_function_call("search", "{}", call_id="toolu_1")
    ctx.add_function_output("toolu_1", "[AF12: 99 EUR]")
    ctx.add_function_call("book", '{"flight": "AF12"}', call_id="toolu_2")
    ctx.add_function_output("toolu_2", "booked")

    messages = to_anthropic_messages(ctx).messages

    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    assert messages[1]["content"][0]["name"] == "search"
    assert messages[2]["content"][0]["tool_use_id"] == "toolu_1"
    assert messages[3]["content"][0]["input"] == {"flight": "AF12"}
    assert messages[4]["content"][0]["tool_use_id"] == "toolu_2"


def test_tool_results_come_first_when_the_user_speaks_during_a_tool_call() -> None:
    ctx = ChatContext()
    ctx.add_message("user", "Check my calendar.")
    ctx.add_message("assistant", "One moment.")
    ctx.add_function_call("calendar", "{}", call_id="toolu_1")
    ctx.add_message("user", "Hello? Still there?")  # barge-in while the tool runs
    ctx.add_function_output("toolu_1", "Free all day")

    messages = to_anthropic_messages(ctx).messages

    assert messages[-1] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Free all day"},
            text("Hello? Still there?"),
        ],
    }
    assert messages[-2]["role"] == "assistant"
    assert [b["type"] for b in messages[-2]["content"]] == ["text", "tool_use"]


def test_unmatched_calls_and_outputs_are_dropped() -> None:
    ctx = ChatContext()
    ctx.add_message("user", "Hi")
    ctx.add_function_output("toolu_gone", "orphan result")  # its call was truncated away
    ctx.add_function_call("slow_tool", "{}", call_id="toolu_pending")  # no output (yet)
    ctx.add_message("assistant", "Hello!")
    ctx.add_message("user", "Bye")

    prompt = to_anthropic_messages(ctx)

    assert prompt.messages == [
        {"role": "user", "content": [text("Hi")]},
        {"role": "assistant", "content": [text("Hello!")]},
        {"role": "user", "content": [text("Bye")]},
    ]
    assert prompt.tool_names == set()


def test_duplicate_tool_calls_are_sent_once() -> None:
    ctx = ChatContext()
    ctx.add_message("user", "Hi")
    call = ctx.add_function_call("ping", "{}", call_id="toolu_1")
    ctx.append(call)
    ctx.add_function_output("toolu_1", "pong")
    messages = to_anthropic_messages(ctx).messages
    assert len(messages[1]["content"]) == 1
    assert len(messages[2]["content"]) == 1


def test_tool_result_content_edge_cases() -> None:
    ctx = ChatContext()
    ctx.add_message("user", "Do both")
    ctx.add_function_call("fire_and_forget", "", call_id="toolu_1")
    ctx.add_function_call("broken", "{}", call_id="toolu_2")
    ctx.add_function_output("toolu_1", "")
    ctx.add_function_output("toolu_2", "", is_error=True)

    messages = to_anthropic_messages(ctx).messages

    assert messages[1]["content"][0]["input"] == {}  # empty arguments -> {}
    assert messages[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_1"},
        {
            "type": "tool_result",
            "tool_use_id": "toolu_2",
            "content": "The tool call failed.",
            "is_error": True,
        },
    ]


def test_invalid_json_arguments_are_sent_as_empty_input() -> None:
    ctx = ChatContext()
    ctx.add_message("user", "Hi")
    ctx.add_function_call("lookup", '{"q": ', call_id="toolu_1")
    ctx.add_function_output("toolu_1", "error: bad arguments", is_error=True)
    assert to_anthropic_messages(ctx).messages[1]["content"][0]["input"] == {}


def test_foreign_call_ids_are_sanitized_consistently() -> None:
    ctx = ChatContext()
    ctx.add_message("user", "Hi")
    ctx.add_function_call("lookup", "{}", call_id="functions.lookup:0")
    ctx.add_function_output("functions.lookup:0", "ok")
    messages = to_anthropic_messages(ctx).messages
    assert messages[1]["content"][0]["id"] == "functions_lookup_0"
    assert messages[2]["content"][0]["tool_use_id"] == "functions_lookup_0"


def test_function_calls_created_with_defaults_keep_their_ids() -> None:
    call = FunctionCall(name="lookup")
    output = FunctionCallOutput(call_id=call.call_id, output="ok")
    ctx = ChatContext([call, output])
    messages = to_anthropic_messages(ctx).messages
    assert messages[1]["content"][0]["id"] == call.call_id
    assert messages[2]["content"][0]["tool_use_id"] == call.call_id


# ---------------------------------------------------------------- images / audio


def test_user_images_from_urls_and_data_urls() -> None:
    png = base64.b64encode(b"\x89PNG fake").decode()
    ctx = ChatContext()
    ctx.add_message(
        "user",
        [
            "What is in these pictures?",
            ImageContent("https://example.com/cat.jpg"),
            ImageContent(f"data:image/png;base64,{png[:6]}\n{png[6:]}"),
            ImageContent("data:,%3Csvg%2F%3E", mime_type="image/svg+xml"),
            "Be brief.",
        ],
    )

    blocks = to_anthropic_messages(ctx).messages[0]["content"]

    assert blocks == [
        text("What is in these pictures?"),
        {"type": "image", "source": {"type": "url", "url": "https://example.com/cat.jpg"}},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png}},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/svg+xml",
                "data": base64.b64encode(b"<svg/>").decode(),
            },
        },
        text("Be brief."),
    ]


@pytest.mark.parametrize(
    ("url", "match"),
    [
        ("file:///tmp/cat.png", "unsupported image URL"),
        ("data:;base64,AAAA", "no media type"),
        ("data:image/png;base64", "malformed data: URL"),
    ],
)
def test_unsupported_images_raise_a_clear_error(url: str, match: str) -> None:
    ctx = ChatContext()
    ctx.add_message("user", ImageContent(url))
    with pytest.raises(ConfigurationError, match=match):
        to_anthropic_messages(ctx)


def test_images_outside_user_messages_are_dropped() -> None:
    ctx = ChatContext()
    ctx.add_message("system", ["Rules.", ImageContent("https://example.com/logo.png")])
    ctx.add_message("user", "Hi")
    ctx.add_message("assistant", ["Look:", ImageContent("https://example.com/x.png")])
    ctx.add_message("user", "Nice")
    prompt = to_anthropic_messages(ctx)
    assert prompt.system == [text("Rules.")]
    assert prompt.messages[1] == {"role": "assistant", "content": [text("Look:")]}


def test_audio_is_represented_by_its_transcript() -> None:
    ctx = ChatContext()
    ctx.add_message("user", audio("What's the weather?"))
    ctx.add_message("assistant", audio("Sunny."))
    ctx.add_message("user", ["Typed text wins.", audio(None)])  # audio duplicates the text
    ctx.add_message("assistant", audio(None))  # nothing to replay: dropped

    assert to_anthropic_messages(ctx).messages == [
        {"role": "user", "content": [text("What's the weather?")]},
        {"role": "assistant", "content": [text("Sunny.")]},
        {"role": "user", "content": [text("Typed text wins.")]},
    ]


def test_untranscribed_user_audio_raises_a_clear_error() -> None:
    ctx = ChatContext()
    ctx.add_message("user", audio(None))
    with pytest.raises(ConfigurationError, match="do not accept audio input"):
        to_anthropic_messages(ctx)


# ------------------------------------------------------------ tool definitions


def test_function_tools_become_tool_definitions() -> None:
    @function_tool
    async def get_weather(city: Annotated[str, "City name"], unit: str = "celsius") -> str:
        """Get the current weather."""
        return "sunny"

    raw = FunctionTool(
        name="lookup",
        description="",
        parameters={"properties": {"q": {"type": "string"}}},
        strict=True,
    )

    specs = to_anthropic_tools([get_weather, raw])

    assert specs[0] == {
        "name": "get_weather",
        "description": "Get the current weather.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "default": "celsius"},
            },
            "required": ["city"],
        },
    }
    assert specs[1] == {
        "name": "lookup",
        "input_schema": {"properties": {"q": {"type": "string"}}, "type": "object"},
        "strict": True,
    }
    assert "type" not in raw.parameters  # the tool's own schema is not mutated


@pytest.mark.parametrize(
    ("choice", "parallel", "expected"),
    [
        (None, None, None),
        ("auto", None, None),
        ("auto", False, {"type": "auto", "disable_parallel_tool_use": True}),
        (None, False, {"type": "auto", "disable_parallel_tool_use": True}),
        ("required", None, {"type": "any"}),
        ("any", False, {"type": "any", "disable_parallel_tool_use": True}),
        ("none", False, {"type": "none"}),
        ("get_weather", None, {"type": "tool", "name": "get_weather"}),
    ],
)
def test_tool_choice_mapping(
    choice: str | None, parallel: bool | None, expected: dict[str, object] | None
) -> None:
    result = to_anthropic_tool_choice(
        choice, tool_names=["get_weather"], parallel_tool_calls=parallel
    )
    assert result == expected


def test_tool_choice_for_an_unknown_tool_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="not one of the provided tools"):
        to_anthropic_tool_choice("send_email", tool_names=["get_weather"])
