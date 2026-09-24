from __future__ import annotations

import asyncio
import json
from enum import Enum
from typing import Annotated, Literal

import pydantic
import pytest

from voice_agent_next import Agent, AudioFrame, ChatContext, FunctionCall, function_tool
from voice_agent_next.chat import AudioContent
from voice_agent_next.errors import ToolError
from voice_agent_next.tools import ToolContext, execute_function_call

# ------------------------------------------------------------------------------ chat


def test_chat_context_items_and_queries() -> None:
    ctx = ChatContext()
    ctx.add_message("system", "be nice")
    u = ctx.add_message("user", "hi")
    call = ctx.add_function_call("lookup", '{"q": 1}')
    ctx.add_function_output(call.call_id, "result", name="lookup")
    a = ctx.add_message("assistant", "hello", id="msg_fixed")
    assert len(ctx) == 5
    assert ctx.get("msg_fixed") is a
    assert ctx.last_message("user") is u
    assert ctx.last_message() is a
    assert [m.role for m in ctx.messages()] == ["system", "user", "assistant"]
    assert ctx.remove(u.id) is u and len(ctx) == 4
    assert call.parsed_arguments() == {"q": 1}


def test_chat_context_truncate_keeps_system_and_no_orphan_outputs() -> None:
    ctx = ChatContext()
    ctx.add_message("system", "sys")
    ctx.add_message("user", "u1")
    call = ctx.add_function_call("f", "{}")
    ctx.add_function_output(call.call_id, "out")
    ctx.add_message("assistant", "a1")
    ctx.truncate(2)
    assert [getattr(i, "role", i.type) for i in ctx.items] == ["system", "assistant"]


def test_chat_context_serialization_roundtrip() -> None:
    ctx = ChatContext()
    ctx.add_message("user", [AudioContent(AudioFrame.silence(0.1, 16_000), "hello"), "text"])
    ctx.add_function_call("f", '{"a": 2}', call_id="call_1")
    ctx.add_function_output("call_1", "ok", name="f", is_error=True)
    data = json.loads(json.dumps(ctx.to_dict(include_audio=True)))
    restored = ChatContext.from_dict(data)
    msg = restored.messages()[0]
    assert msg.text == "hellotext"
    assert msg.audio[0].frame.duration == pytest.approx(0.1)
    assert restored.items[1].call_id == "call_1"  # type: ignore[union-attr]
    assert restored.items[2].is_error  # type: ignore[union-attr]
    # without audio, transcripts are kept as text
    light = ChatContext.from_dict(ctx.to_dict())
    assert light.messages()[0].text == "hellotext"


def test_function_call_bad_arguments() -> None:
    assert FunctionCall(name="f", arguments="").parsed_arguments() == {}
    with pytest.raises(ValueError):
        FunctionCall(name="f", arguments="[1, 2]").parsed_arguments()


# ----------------------------------------------------------------------------- tools


class Unit(Enum):
    C = "celsius"
    F = "fahrenheit"


@function_tool
async def get_weather(
    city: Annotated[str, "City name"],
    unit: Literal["celsius", "fahrenheit"] = "celsius",
    days: int | None = None,
) -> str:
    """Get the weather.

    Args:
        city: ignored because Annotated wins.
        days: Forecast length in days.
    """
    return f"{city}:{unit}:{days}"


def test_schema_generation() -> None:
    schema = get_weather.schema()
    assert schema["name"] == "get_weather"
    assert schema["description"] == "Get the weather."
    props = schema["parameters"]["properties"]
    assert props["city"]["description"] == "City name"
    assert props["unit"]["enum"] == ["celsius", "fahrenheit"]
    assert props["days"]["description"] == "Forecast length in days."
    assert schema["parameters"]["required"] == ["city"]
    assert "title" not in json.dumps(schema)


async def test_tool_call_validation_and_coercion() -> None:
    assert await get_weather('{"city": "Paris", "days": "3"}') == "Paris:celsius:3"
    with pytest.raises(ToolError):
        await get_weather('{"unit": "kelvin", "city": "x"}')
    with pytest.raises(ToolError):
        await get_weather("[]")


async def test_sync_tool_runs_in_thread_and_context_injection() -> None:
    seen: dict[str, object] = {}

    @function_tool(name="whoami", description="Return the caller")
    def whoami(ctx: ToolContext, shout: bool = False) -> dict[str, object]:
        seen["ctx"] = ctx
        return {"user": ctx.userdata, "shout": shout}

    assert "ctx" not in whoami.parameters["properties"]
    call = FunctionCall(name="whoami", arguments='{"shout": true}')
    out = await execute_function_call(call, [whoami], ctx=ToolContext(call=call, userdata="bob"))
    assert not out.is_error
    assert json.loads(out.output) == {"user": "bob", "shout": True}
    assert isinstance(seen["ctx"], ToolContext)


async def test_execute_function_call_errors_become_outputs() -> None:
    @function_tool
    async def boom() -> None:
        """Explodes."""
        raise RuntimeError("kaput")

    @function_tool(timeout=0.01)
    async def slow() -> str:
        """Too slow."""
        await asyncio.sleep(1)
        return "never"

    unknown = await execute_function_call(FunctionCall(name="nope"), [boom])
    assert unknown.is_error and "Unknown tool" in unknown.output
    failed = await execute_function_call(FunctionCall(name="boom"), [boom])
    assert failed.is_error and "kaput" in failed.output
    timed_out = await execute_function_call(FunctionCall(name="slow"), [slow])
    assert timed_out.is_error and "timed out" in timed_out.output


async def test_pydantic_and_enum_parameters() -> None:
    class Address(pydantic.BaseModel):
        street: str
        zip_code: str

    @function_tool
    async def ship(address: Address, unit: Unit = Unit.C) -> str:
        """Ship a package."""
        return f"{address.zip_code}-{unit.value}"

    assert "address" in ship.parameters["properties"]
    assert (
        await ship({"address": {"street": "a", "zip_code": "123"}, "unit": "fahrenheit"})
        == "123-fahrenheit"
    )


def test_agent_collects_method_tools_bound_to_instance() -> None:
    class Shop(Agent):
        def __init__(self) -> None:
            self.stock = {"apple": 3}
            super().__init__(instructions="shop", tools=[get_weather])

        @function_tool
        async def check_stock(self, item: str) -> int:
            """Check stock for an item."""
            return self.stock.get(item, 0)

    shop = Shop()
    names = [t.name for t in shop.tools]
    assert names == ["get_weather", "check_stock"]
    tool = shop.tools[1]
    assert "self" not in tool.parameters["properties"]
    assert asyncio.run(tool('{"item": "apple"}')) == 3
