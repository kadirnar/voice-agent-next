"""ChatContext -> Chat Completions conversion and stream parsing (no ``openai`` SDK needed)."""

from __future__ import annotations

import base64
import io
import wave
from typing import Any

import pytest

from voice_agent_next import AudioFrame, ChatContext, function_tool
from voice_agent_next.chat import AudioContent, FunctionCall, FunctionCallOutput, ImageContent
from voice_agent_next.providers.mock import synth_speech
from voice_agent_next.providers.openai._format import (
    to_chat_messages,
    to_chat_tools,
    to_tool_choice,
)
from voice_agent_next.providers.openai.llm import _StreamParser, _ThinkStripper
from voice_agent_next.tools import FunctionTool


def weather_call(city: str, call_id: str) -> FunctionCall:
    return FunctionCall(name="get_weather", arguments=f'{{"city": "{city}"}}', call_id=call_id)


def tool_msg(call_id: str, output: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": output}


# ---------------------------------------------------------------------------- messages


def test_text_messages_and_developer_role_mapping() -> None:
    ctx = ChatContext()
    ctx.add_message("system", "You are terse.")
    ctx.add_message("developer", "Never use markdown.")
    ctx.add_message("user", ["Hello ", "there"])
    ctx.add_message("assistant", "Hi!")
    assert to_chat_messages(ctx) == [
        {"role": "system", "content": "You are terse."},
        {"role": "developer", "content": "Never use markdown."},
        {"role": "user", "content": "Hello there"},
        {"role": "assistant", "content": "Hi!"},
    ]
    compat = to_chat_messages(ctx, developer_role="system")
    assert compat[1] == {"role": "system", "content": "Never use markdown."}


def test_user_images_and_audio_become_content_parts() -> None:
    audio = synth_speech(0.25, 16_000)
    ctx = ChatContext()
    ctx.add_message(
        "user",
        [
            "What is in this picture?",
            ImageContent("https://example.com/cat.png"),
            AudioContent(audio, transcript="and what did I say?"),
        ],
    )
    [msg] = to_chat_messages(ctx, audio_input=True)
    text, image, sound = msg["content"]
    assert text == {"type": "text", "text": "What is in this picture?"}
    assert image == {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}
    assert sound["type"] == "input_audio" and sound["input_audio"]["format"] == "wav"
    with wave.open(io.BytesIO(base64.b64decode(sound["input_audio"]["data"]))) as wav:
        assert wav.getframerate() == 16_000 and wav.getnchannels() == 1
        assert wav.getnframes() == len(audio.to_numpy())


def test_audio_for_text_only_models_falls_back_to_transcript() -> None:
    ctx = ChatContext()
    ctx.add_message("user", AudioContent(synth_speech(0.2, 16_000), transcript="book a table"))
    ctx.add_message("user", AudioContent(synth_speech(0.2, 16_000)))  # no transcript: dropped
    ctx.add_message("assistant", AudioContent(AudioFrame.empty(24_000), transcript="Done."))
    assert to_chat_messages(ctx) == [
        {"role": "user", "content": "book a table"},
        {"role": "assistant", "content": "Done."},
    ]


def test_tool_calls_grouped_into_assistant_message_followed_by_outputs() -> None:
    ctx = ChatContext()
    ctx.add_message("user", "Weather in Paris and Rome?")
    ctx.add_message("assistant", "Let me check.")
    ctx.append(weather_call("Paris", "call_1"))
    ctx.append(weather_call("Rome", "call_2"))
    ctx.add_function_output("call_1", "sunny", name="get_weather")
    ctx.add_function_output("call_2", "rainy", name="get_weather")
    ctx.add_message("assistant", "Paris is sunny, Rome is rainy.")
    assert to_chat_messages(ctx)[1:] == [
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Rome"}'},
                },
            ],
        },
        tool_msg("call_1", "sunny"),
        tool_msg("call_2", "rainy"),
        {"role": "assistant", "content": "Paris is sunny, Rome is rainy."},
    ]


def test_sequential_tool_rounds_and_text_after_calls() -> None:
    ctx = ChatContext()
    ctx.add_message("user", "Plan my day")
    ctx.append(FunctionCall(name="calendar", arguments="", call_id="c1"))
    ctx.add_message("assistant", "Checking your calendar.")  # spoken while calling
    ctx.add_function_output("c1", "free after 3pm")
    ctx.append(weather_call("Paris", "c2"))
    ctx.add_function_output("c2", "sunny")
    messages = to_chat_messages(ctx)
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant", "tool"]
    assert messages[1]["content"] == "Checking your calendar."
    assert messages[1]["tool_calls"][0]["function"] == {"name": "calendar", "arguments": "{}"}
    assert "content" not in messages[3] and messages[3]["tool_calls"][0]["id"] == "c2"


def test_unanswered_calls_and_orphan_outputs_are_dropped() -> None:
    ctx = ChatContext()
    ctx.add_function_output("lost", "output of a truncated call")
    ctx.add_message("user", "What's the weather?")
    ctx.append(weather_call("Paris", "running"))  # tool still running: no output yet
    ctx.add_message("user", "Actually, never mind.")
    assert to_chat_messages(ctx) == [
        {"role": "user", "content": "What's the weather?"},
        {"role": "user", "content": "Actually, never mind."},
    ]


def test_output_is_moved_right_after_its_call() -> None:
    ctx = ChatContext()
    ctx.add_message("user", "Weather?")
    ctx.append(weather_call("Paris", "c1"))
    ctx.add_message("user", "In Celsius please")  # user spoke while the tool ran
    ctx.add_function_output("c1", "21C")
    roles = [m["role"] for m in to_chat_messages(ctx)]
    assert roles == ["user", "assistant", "tool", "user"]


def test_interrupted_message_kept_as_heard_and_empty_messages_skipped() -> None:
    ctx = ChatContext()
    ctx.add_message("system", "")
    ctx.add_message("user", "Tell me a story")
    ctx.add_message("assistant", "Once upon a", interrupted=True)
    ctx.add_message("assistant", "", interrupted=True)  # cut off before any word was heard
    ctx.add_message("user", "Stop")
    ctx.add_message("assistant", "")  # the cascade's placeholder for the pending reply
    assert to_chat_messages(ctx) == [
        {"role": "user", "content": "Tell me a story"},
        {"role": "assistant", "content": "Once upon a"},
        {"role": "user", "content": "Stop"},
    ]


def test_roundtrip_through_serialized_context() -> None:
    ctx = ChatContext()
    ctx.add_message("system", "Be brief.")
    ctx.add_message("user", "Weather?")
    ctx.append(weather_call("Paris", "c1"))
    ctx.add_function_output("c1", "sunny")
    restored = ChatContext.from_dict(ctx.to_dict())
    assert to_chat_messages(restored) == to_chat_messages(ctx)


# ------------------------------------------------------------------------------- tools


def test_tools_schema() -> None:
    @function_tool
    async def get_weather(city: str, unit: str = "celsius") -> str:
        """Get the weather for a city."""
        return "sunny"

    raw = FunctionTool(
        name="hang_up",
        description="",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        strict=True,
    )
    weather, hang_up = to_chat_tools([get_weather, raw])
    assert weather["type"] == "function"
    assert weather["function"]["name"] == "get_weather"
    assert weather["function"]["description"] == "Get the weather for a city."
    assert weather["function"]["parameters"]["required"] == ["city"]
    assert "strict" not in weather["function"]
    assert hang_up == {
        "type": "function",
        "function": {
            "name": "hang_up",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            "strict": True,
        },
    }


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        (None, None),
        ("auto", "auto"),
        ("required", "required"),
        ("none", "none"),
        ("get_weather", {"type": "function", "function": {"name": "get_weather"}}),
    ],
)
def test_tool_choice(choice: str | None, expected: Any) -> None:
    assert to_tool_choice(choice) == expected


# -------------------------------------------------------------------- stream parsing


def chunk(
    delta: dict[str, Any] | None = None, finish: str | None = None, **kw: Any
) -> dict[str, Any]:
    choices = [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]
    return {"object": "chat.completion.chunk", "choices": choices, **kw}


def test_parser_accumulates_fragmented_tool_calls_by_index() -> None:
    parser = _StreamParser()
    fragments = [
        {"index": 0, "id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": ""}},
        {"index": 0, "function": {"arguments": '{"ci'}},
        {"index": 0, "function": {"arguments": 'ty": "Paris"}'}},
        {"index": 1, "id": "call_b", "type": "function", "function": {"name": "get_time", "arguments": ""}},
    ]  # fmt: skip
    for frag in fragments:
        assert parser.feed(chunk({"tool_calls": [frag]})) == ("", [])
    assert parser.tool_call_started
    _, calls = parser.feed(chunk({}, finish="tool_calls"))
    assert [(c.name, c.arguments, c.call_id) for c in calls] == [
        ("get_weather", '{"city": "Paris"}', "call_a"),
        ("get_time", "{}", "call_b"),  # empty arguments are normalized
    ]
    assert parser.finish_reason == "tool_calls"
    assert parser.finish() == ("", [])


def test_parser_handles_missing_or_reused_indexes() -> None:
    parser = _StreamParser()
    parser.feed(chunk({"tool_calls": [{"id": "a", "function": {"name": "f", "arguments": "{"}}]}))
    parser.feed(chunk({"tool_calls": [{"function": {"arguments": "}"}}]}))  # continues "a"
    parser.feed(chunk({"tool_calls": [{"index": 0, "id": "b", "function": {"name": "g"}}]}))
    parser.feed(chunk({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "h"}}]}))
    parser.feed(chunk({"tool_calls": [{"index": 5, "function": {"arguments": "{}"}}]}))  # no name
    _, calls = parser.finish()  # no finish_reason from this server
    assert [(c.call_id, c.name, c.arguments) for c in calls] == [
        ("a", "f", "{}"),
        ("b", "g", "{}"),
        ("c", "h", "{}"),
    ]
    assert calls[0].call_id == "a"


@pytest.mark.parametrize(
    ("usage_chunk", "expected"),
    [
        (  # OpenAI: separate usage chunk with cached prompt tokens
            chunk(usage={"prompt_tokens": 120, "completion_tokens": 9,
                         "prompt_tokens_details": {"cached_tokens": 64}}),
            (120, 9, 64),
        ),
        (  # DeepSeek: prompt_cache_hit_tokens
            chunk(usage={"prompt_tokens": 50, "completion_tokens": 5, "prompt_cache_hit_tokens": 32,
                         "prompt_cache_miss_tokens": 18}),
            (50, 5, 32),
        ),
        (  # Fireworks: usage on the chunk that carries finish_reason
            chunk({}, finish="stop", usage={"prompt_tokens": 7, "completion_tokens": 3}),
            (7, 3, 0),
        ),
        (  # Groq (legacy): x_groq.usage
            chunk({}, finish="stop", x_groq={"id": "req", "usage": {"prompt_tokens": 11, "completion_tokens": 2}}),
            (11, 2, 0),
        ),
    ],
)  # fmt: skip
def test_parser_usage_variants(usage_chunk: dict[str, Any], expected: tuple[int, int, int]) -> None:
    parser = _StreamParser()
    parser.feed(chunk({"content": "ok"}))
    parser.feed(usage_chunk)
    assert parser.usage is not None
    got = parser.usage
    assert (got.prompt_tokens, got.completion_tokens, got.cached_tokens) == expected


def test_parser_skips_reasoning_and_speaks_refusals() -> None:
    parser = _StreamParser()
    assert parser.feed(chunk({"content": "", "reasoning": "The user wants"})) == ("", [])
    assert parser.feed(chunk({"reasoning_content": "hmm"})) == ("", [])
    assert parser.feed(chunk({"content": None, "refusal": "I can't help with that."})) == (
        "I can't help with that.",
        [],
    )


@pytest.mark.parametrize(
    ("deltas", "spoken"),
    [
        (["<think>", "plan", "</think>", "\n\n", "Hello", " there"], "Hello there"),
        (["<th", "ink>a</th", "ink>Hi"], "Hi"),
        (["  <think>x", "y</think>  Sure."], "Sure."),
        (["Hello <think> is a tag"], "Hello <think> is a tag"),  # only a leading block
        (["<", "b>bold"], "<b>bold"),
        (["<think>never closed"], ""),
        (["\n", "Hi"], "\nHi"),
        (["<thi"], "<thi"),  # stream ended on a partial tag: nothing is lost
    ],
)
def test_think_stripper(deltas: list[str], spoken: str) -> None:
    stripper = _ThinkStripper()
    out = "".join(stripper.push(d) for d in deltas) + stripper.flush()
    assert out == spoken


def test_function_call_output_type_is_preserved_for_errors() -> None:
    ctx = ChatContext()
    ctx.add_message("user", "Book it")
    ctx.append(FunctionCall(name="book", arguments="{}", call_id="c1"))
    ctx.append(FunctionCallOutput(call_id="c1", output="Tool book failed: timeout", is_error=True))
    assert to_chat_messages(ctx)[-1] == tool_msg("c1", "Tool book failed: timeout")


def strict_ctx() -> ChatContext:
    ctx = ChatContext()
    ctx.add_message("system", "You are terse.")
    ctx.add_message("user", "Hi")
    ctx.add_message("assistant", "Hello!")
    ctx.add_message("user", "Bye")
    ctx.add_message("system", "Say goodbye politely.")  # per-response instructions
    return ctx


def test_system_messages_kept_in_place_by_default() -> None:
    assert [m["role"] for m in to_chat_messages(strict_ctx())] == [
        "system", "user", "assistant", "user", "system"]  # fmt: skip


def test_system_messages_merged_into_one_leading_message() -> None:
    assert to_chat_messages(strict_ctx(), system_messages="merge") == [
        {"role": "system", "content": "You are terse.\n\nSay goodbye politely."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Bye"},
    ]
    only_late = ChatContext()
    only_late.add_message("user", "Hi")
    only_late.add_message("developer", "Answer in French.")
    assert to_chat_messages(only_late, developer_role="system", system_messages="merge") == [
        {"role": "system", "content": "Answer in French."},
        {"role": "user", "content": "Hi"},
    ]


def test_late_system_messages_as_user_keep_roles_alternating() -> None:
    assert to_chat_messages(strict_ctx(), system_messages="as_user") == [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Bye\n\nSay goodbye politely."},
    ]
    ctx = ChatContext()
    ctx.add_message("user", ["Look: ", ImageContent("data:image/png;base64,AAAA")])
    ctx.add_message("system", "Describe it briefly.")
    (msg,) = to_chat_messages(ctx, system_messages="as_user")
    assert msg["role"] == "user" and msg["content"][-1] == {
        "type": "text", "text": "Describe it briefly."}  # fmt: skip


def test_hosts_with_strict_templates_merge_system_messages() -> None:
    from voice_agent_next.providers.llamacpp import LlamaCppLLM
    from voice_agent_next.providers.ollama import OllamaLLM
    from voice_agent_next.providers.openai.llm import OpenAILLM

    pytest.importorskip("openai")
    assert LlamaCppLLM().system_message_policy == "merge"
    assert OllamaLLM().system_message_policy == "keep"
    llm = OpenAILLM(api_key="sk-test", system_message_policy="as_user")
    request = llm.build_request(strict_ctx())
    assert [m["role"] for m in request["messages"]] == ["system", "user", "assistant", "user"]
