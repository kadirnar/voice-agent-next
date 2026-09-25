"""Session hot paths: settled responses are pruned, reply text is built incrementally (#141)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from voice_agent_next import Agent, AgentSession, AgentState, ChatMessage
from voice_agent_next.bench.microbench import run_micro_benchmarks
from voice_agent_next.providers.mock import MockEngine
from voice_agent_next.session import session as session_mod
from voice_agent_next.transports import LoopbackTransport


async def wait_for(predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


async def test_settled_responses_are_pruned_in_a_long_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keep = 3
    monkeypatch.setattr(session_mod, "_KEEP_SETTLED", keep)
    session = AgentSession(MockEngine(realtime_factor=0.0, chars_per_second=2000.0))
    await session.start(Agent("x"), LoopbackTransport())

    def texts() -> list[str]:
        return [m.text for m in session.history.items if isinstance(m, ChatMessage)]

    n = keep + 5
    for i in range(n):
        await session.say(f"Line {i}.")
        await wait_for(lambda i=i: len(texts()) > i)  # type: ignore[misc]
        await wait_for(lambda: session.agent_state == AgentState.LISTENING)
    await wait_for(lambda: len(session._responses) <= keep + 1)
    await session.aclose()
    assert texts() == [f"Line {i}." for i in range(n)]  # the history keeps every reply


async def test_long_reply_text_is_complete_in_the_history() -> None:
    reply = " ".join(f"word{i}" for i in range(600)) + "."
    session = AgentSession(MockEngine(responses=[reply], realtime_factor=0.0))
    await session.start(Agent("x"), LoopbackTransport())
    await session.generate_reply(user_input="talk")
    await wait_for(
        lambda: any(m.text == reply for m in session.history.items if isinstance(m, ChatMessage))
    )
    await session.aclose()


def test_agent_transcript_micro_benchmark_exists() -> None:
    (result,) = run_micro_benchmarks(repeats=1, min_time=0.001, only=["agent_transcript"])
    assert result.unit == "token" and result.median_us > 0
