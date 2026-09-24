"""OpenAI Realtime session rotation and reconnect with context carry-over (fake server)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

import pytest

from tests.fake_realtime_server import FakeRealtimeServer
from tests.test_openai_realtime import KEY, Collector, feed, spoken, user_turn, wait_for
from voice_agent_next import AudioFrame, ChatContext, FunctionCallOutput, function_tool
from voice_agent_next.engine import EngineOptions
from voice_agent_next.engines.rotation import RotationPolicy
from voice_agent_next.events import (
    EngineErrorEvent,
    EngineStatus,
    ResponseAudio,
    ResponseDone,
    ResponseToolCall,
)
from voice_agent_next.metrics import RotationMetrics
from voice_agent_next.providers.mock import MockToolCall
from voice_agent_next.providers.openai.realtime import (
    OpenAIRealtimeConnection,
    OpenAIRealtimeEngine,
)
from voice_agent_next.utils import now

FAST = RotationPolicy(quiet_period=0.1)


@function_tool
async def lookup(query: str) -> str:
    """Look something up."""
    return "found"


@contextlib.asynccontextmanager
async def rotating(
    server: FakeRealtimeServer,
    *,
    policy: RotationPolicy = FAST,
    options: EngineOptions | None = None,
    **kwargs: Any,
) -> AsyncIterator[tuple[OpenAIRealtimeConnection, Collector, list[RotationMetrics]]]:
    engine = OpenAIRealtimeEngine(base_url=server.url, api_key=KEY, turn_detection="server_vad",
                                  rotation=policy, **kwargs)  # fmt: skip
    metrics: list[RotationMetrics] = []
    engine.on("metrics", lambda m: metrics.append(m) if isinstance(m, RotationMetrics) else None)
    conn = await engine.connect(options or EngineOptions(instructions="Be brief.", voice="cedar"))
    assert isinstance(conn, OpenAIRealtimeConnection)
    rec = Collector(conn)
    try:
        yield conn, rec, metrics
    finally:
        await conn.aclose()
        await asyncio.wait_for(rec.task, 2)


def statuses(rec: Collector) -> list[str]:
    return [s.status for s in rec.of(EngineStatus)]


def created_items(server: FakeRealtimeServer, since: int) -> list[dict[str, Any]]:
    """Items created by the client after the ``since``-th received event."""
    return [
        e["item"] for e in server.received[since:] if e.get("type") == "conversation.item.create"
    ]


async def keep_talking(conn: OpenAIRealtimeConnection, stop: asyncio.Event) -> float:
    """Stream silence (like an open microphone) until ``stop``; returns seconds sent."""
    sent = 0.0
    while not stop.is_set():
        await feed(conn, AudioFrame.silence(0.02, 16_000))
        sent += 0.02
        await asyncio.sleep(0.01)
    return sent


async def test_forced_rotation_carries_the_conversation_to_a_new_session() -> None:
    server = FakeRealtimeServer(transcripts=["my name is Ada", "what is my name"],
                                replies=["Nice to meet you, Ada.", "You are Ada."])  # fmt: skip
    async with server, rotating(server) as (conn, rec, metrics):
        await user_turn(conn)
        await rec.wait(lambda: rec.of(ResponseDone))
        await wait_for(
            lambda: server.sent_events("conversation.item.input_audio_transcription.completed")
        )
        first_session = conn.session_id
        stop = asyncio.Event()
        mic = asyncio.create_task(keep_talking(conn, stop))  # the microphone never stops
        mark = len(server.received)
        conn.rotate("test")
        await rec.wait(lambda: statuses(rec)[-1:] == ["reconnected"])
        await asyncio.sleep(0.2)
        stop.set()
        await mic
        assert statuses(rec) == ["reconnecting", "reconnected"]
        assert len(server.handshakes) == 2 and conn.session_id != first_session
        # the new session got the configuration and the conversation as heard
        update = server.events("session.update")[-1]["session"]
        assert update["instructions"] == "Be brief."
        assert update["audio"]["output"]["voice"] == "cedar"
        assert created_items(server, mark) == [
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "my name is Ada"}]},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "Nice to meet you, Ada."}]},
        ]  # fmt: skip
        (m,) = metrics
        assert m.planned and m.carried_items == 2 and m.failed_responses == 0 and m.lost_audio == 0
        # no lost audio: every frame reached one of the sessions (the replay reached both)
        old, new = server.connections
        await wait_for(lambda: new.position + old.position
                       >= conn.input_audio_time + m.replayed_audio - 0.005)  # fmt: skip
        assert new.position + old.position == pytest.approx(
            conn.input_audio_time + m.replayed_audio, abs=0.005
        )
        # the conversation continues on the new session only: no double responses
        await user_turn(conn)
        await rec.wait(lambda: len(rec.of(ResponseDone)) == 2)
        await asyncio.sleep(0.2)
        assert len(server.responses) == 2 and spoken(rec).endswith("You are Ada.")
        assert old.ws.state.name == "CLOSED"
    assert not rec.of(EngineErrorEvent)


async def test_rotation_waits_until_the_answer_was_played() -> None:
    reply = "A long answer that keeps playing for a little while."
    server = FakeRealtimeServer(replies=[reply], realtime_factor=0.5)
    async with server, rotating(server) as (conn, rec, metrics):
        await conn.create_response()
        await rec.wait(lambda: rec.of(ResponseAudio))
        conn.rotate("test")
        await rec.wait(lambda: rec.of(ResponseDone), timeout=10)
        done_at = now()
        played = sum(a.frame.duration for a in rec.of(ResponseAudio))
        await rec.wait(lambda: metrics, timeout=10)
        switched = rec.of(EngineStatus)[0].timestamp
        # generation ran at 2x real time: the switch waited for the playout to finish
        assert switched - rec.of(ResponseAudio)[0].timestamp >= played - 0.1
        assert switched >= done_at
        assert rec.of(ResponseDone)[0].status == "completed" and metrics[0].failed_responses == 0


async def test_rotation_at_the_deadline_cuts_the_response_short() -> None:
    server = FakeRealtimeServer(replies=["Talking and talking and talking for ages.", "Hi."],
                                realtime_factor=1.0)  # fmt: skip
    async with server, rotating(server) as (conn, rec, metrics):
        await conn.create_response()
        await rec.wait(lambda: rec.of(ResponseAudio))
        conn.rotate("test", deadline=now())
        await rec.wait(lambda: statuses(rec)[-1:] == ["reconnected"])
        done = rec.of(ResponseDone)[0]
        assert done.status == "failed" and "session rotated" in (done.error or "")
        assert metrics[0].failed_responses == 1
        await conn.create_response()
        await rec.wait(lambda: len(rec.of(ResponseDone)) == 2)
        assert rec.of(ResponseDone)[1].status == "completed"


async def test_rotation_before_the_session_limit() -> None:
    policy = RotationPolicy(lead=2.0, force_margin=0.5, quiet_period=0.05)
    async with FakeRealtimeServer(expires_in=3.0) as server:
        start = now()
        async with rotating(server, policy=policy) as (_, rec, metrics):
            await rec.wait(lambda: metrics, timeout=5)
            # expires_at is whole seconds: the limit is 2-3 s, the rotation after half of it
            assert 0.9 < now() - start < 3.0
            assert metrics[0].reason == "session_limit" and metrics[0].planned
            assert statuses(rec)[0] == "expiring"


async def test_drop_reconnects_with_history_and_buffered_audio() -> None:
    server = FakeRealtimeServer(transcripts=["remember blue", "which color"],
                                replies=["Blue it is.", "It was blue.", "Blue!"])  # fmt: skip
    async with server, rotating(server, reconnect_backoff=0.3) as (conn, rec, metrics):
        await user_turn(conn)
        await rec.wait(lambda: rec.of(ResponseDone))
        await wait_for(
            lambda: server.sent_events("conversation.item.input_audio_transcription.completed")
        )
        mark = len(server.received)
        await server.drop()
        await rec.wait(lambda: statuses(rec) == ["reconnecting"])
        await feed(conn, AudioFrame.silence(0.4, 16_000))  # buffered during the outage
        await conn.send_text("still there?")  # carried over; its response is deferred
        await rec.wait(lambda: statuses(rec)[-1:] == ["reconnected"])
        (m,) = metrics
        assert not m.planned and m.attempts == 1 and m.lost_audio == 0
        assert m.buffered_audio == pytest.approx(0.4, abs=0.01) and m.gap >= 0.3
        texts = [item["content"][0]["text"] for item in created_items(server, mark)]
        assert texts == ["remember blue", "Blue it is.", "still there?"]
        new = server.connections[-1]
        await wait_for(lambda: new.position >= m.buffered_audio - 0.001)
        await rec.wait(lambda: len(rec.of(ResponseDone)) == 2)  # the deferred response
        assert spoken(rec).endswith("It was blue.")
        # and voice turns keep working on the new session
        await user_turn(conn)
        await rec.wait(lambda: len(rec.of(ResponseDone)) == 3, timeout=8)
        assert spoken(rec).endswith("Blue!")
    assert not rec.of(EngineErrorEvent)


async def test_tools_in_flight_block_rotation_and_survive_a_drop() -> None:
    server = FakeRealtimeServer(replies=[MockToolCall("lookup", {"query": "x"}), "Found it."])
    options = EngineOptions(instructions="Use tools.", tools=[lookup])
    async with server, rotating(server, options=options, reconnect_backoff=0.02) as (
        conn, rec, metrics,
    ):  # fmt: skip
        await conn.send_text("look it up")
        await rec.wait(lambda: rec.of(ResponseToolCall))
        call = rec.of(ResponseToolCall)[0].call
        conn.rotate("test")
        await asyncio.sleep(0.4)
        assert not metrics  # the tool is still running: no planned switch
        mark = len(server.received)
        await server.drop()
        await rec.wait(lambda: metrics)
        items = created_items(server, mark)
        assert [i["type"] for i in items] == ["message", "function_call"]
        assert items[1]["call_id"] == call.call_id
        await conn.send_tool_output(FunctionCallOutput(call_id=call.call_id, output="found"))
        await rec.wait(lambda: spoken(rec).endswith("Found it."))
        assert server.events("conversation.item.create")[-1]["item"]["type"] == (
            "function_call_output"
        )
        await asyncio.sleep(0.4)  # the pending rotation was satisfied by the reconnect
        assert len(metrics) == 1


async def test_carry_over_strategy_is_pluggable() -> None:
    async def only_last(history: ChatContext) -> ChatContext:
        return ChatContext(history.items[-1:])

    server = FakeRealtimeServer(replies=["One.", "Two."])
    policy = RotationPolicy(quiet_period=0.05, carry_over=only_last)
    async with server, rotating(server, policy=policy) as (conn, rec, metrics):
        await conn.send_text("first")
        await rec.wait(lambda: rec.of(ResponseDone))
        mark = len(server.received)
        conn.rotate()
        await rec.wait(lambda: metrics)
        items = created_items(server, mark)
        assert [i["content"][0]["text"] for i in items] == ["One."]
        assert metrics[0].carried_items == 1


async def test_failed_rotation_keeps_the_current_session() -> None:
    server = FakeRealtimeServer(replies=["ok"])
    async with server, rotating(server) as (conn, rec, metrics):
        server.reject_status = 503
        conn.rotate("test", deadline=now())
        await rec.wait(lambda: "resumed" in statuses(rec))
        assert statuses(rec)[:2] == ["reconnecting", "resumed"]
        assert rec.of(EngineErrorEvent)[0].recoverable and not metrics
        server.reject_status = None
        await rec.wait(lambda: metrics, timeout=5)  # retried
        await conn.create_response()
        await rec.wait(lambda: rec.of(ResponseDone))
        assert spoken(rec) == "ok"
