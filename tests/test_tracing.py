"""OpenTelemetry tracing of sessions (in-memory span exporter) and the disabled path."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from typing import Any

import pytest

from voice_agent_next import Agent, AgentSession, AgentState, SessionOptions
from voice_agent_next.providers.mock import MockToolCall, synth_speech
from voice_agent_next.session import SessionTracer
from voice_agent_next.tools import function_tool
from voice_agent_next.transports import LoopbackTransport

from .test_session import Recorder, make_session, speak, wait_for


# ------------------------------------------------------------------ disabled path
def test_no_tracing_imports_no_opentelemetry() -> None:
    code = (
        "import sys\n"
        "from voice_agent_next import AgentSession\n"
        "s = AgentSession('mock')\n"
        "assert s._taps == [] and s.recorder is None\n"
        "assert not [m for m in sys.modules if m.startswith('opentelemetry')], 'otel imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)


def test_trace_without_the_extra_is_a_logged_no_op(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(SessionTracer, "available", staticmethod(lambda: False))
    session = AgentSession("mock", options=SessionOptions(trace=True))
    assert session._taps == []
    assert "opentelemetry-api is not installed" in caplog.text


async def test_trace_true_with_the_api_only_is_a_no_op_tracer() -> None:
    pytest.importorskip("opentelemetry.trace")
    session = make_session("native", transcripts=["hi"], responses=["Hello."], trace=True)
    assert any(isinstance(t, SessionTracer) for t in session._taps)
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport)
    await wait_for(lambda: len(rec.turn_metrics()) == 1)
    await session.aclose()
    assert rec.of("error") == []


# ------------------------------------------------------------------- exported spans
@pytest.fixture
def exporter() -> Any:
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    exp.provider = provider  # type: ignore[attr-defined]
    return exp


def by_name(spans: list[Any], prefix: str) -> list[Any]:
    return [s for s in spans if s.name == prefix or s.name.startswith(prefix + " ")]


def one(spans: list[Any], prefix: str) -> Any:
    found = by_name(spans, prefix)
    assert len(found) == 1, [s.name for s in spans]
    return found[0]


def parent_of(span: Any, spans: list[Any]) -> Any:
    assert span.parent is not None, f"{span.name} has no parent"
    return next(s for s in spans if s.context.span_id == span.parent.span_id)


@function_tool
async def get_weather(city: str) -> str:
    """Weather for a city."""
    return f"Sunny in {city}"


async def run_tool_turn(kind: str, tracer: SessionTracer) -> Recorder:
    session = make_session(
        kind,
        transcripts=["weather in paris?"],
        responses=[MockToolCall("get_weather", {"city": "Paris"}), "It is sunny in Paris."],
        trace=tracer,
    )
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x", tools=[get_weather], greeting="Hello!"), transport)
    await wait_for(lambda: AgentState.SPEAKING in rec.states())
    await wait_for(lambda: session.agent_state == AgentState.LISTENING)
    await speak(transport)
    await wait_for(lambda: len(rec.turn_metrics()) == 1)
    transport.end_user_audio()
    await asyncio.wait_for(session.wait_closed(), 5)
    return rec


@pytest.mark.parametrize("kind", ["native", "cascade"])
async def test_span_tree_session_turn_components(kind: str, exporter: Any) -> None:
    tracer = SessionTracer(exporter.provider, capture_content=True)
    rec = await run_tool_turn(kind, tracer)
    spans = list(exporter.get_finished_spans())

    root = one(spans, "session")
    assert root.parent is None
    assert root.attributes["gen_ai.conversation.id"] == tracer.conversation_id
    assert root.attributes["voice_agent.session.close_reason"] == "user_disconnected"
    assert root.attributes["voice_agent.session.turns"] == 1
    assert {s.context.trace_id for s in spans} == {root.context.trace_id}

    turn = one(spans, "turn")
    assert parent_of(turn, spans) is root
    m = rec.turn_metrics()[0]
    assert turn.attributes["voice_agent.turn.id"] == m.turn_id
    assert turn.attributes["voice_agent.turn.number"] == 1
    assert turn.attributes["voice_agent.turn.voice_to_voice"] == pytest.approx(m.voice_to_voice)
    assert turn.attributes["voice_agent.turn.was_interrupted"] is False
    assert turn.attributes["voice_agent.turn.tool_calls"] == 1
    assert turn.attributes["voice_agent.user.transcript"] == "weather in paris?"
    assert root.start_time <= turn.start_time <= turn.end_time <= root.end_time

    eot = one(spans, "end_of_turn")
    assert parent_of(eot, spans) is turn
    assert eot.attributes["voice_agent.end_of_turn_delay"] >= 0

    tool = one(spans, "execute_tool")
    assert tool.name == "execute_tool get_weather"
    assert parent_of(tool, spans) is turn
    assert tool.attributes["gen_ai.operation.name"] == "execute_tool"
    assert tool.attributes["gen_ai.tool.name"] == "get_weather"
    assert tool.attributes["gen_ai.tool.call.result"] == "Sunny in Paris"
    assert tool.start_time <= tool.end_time

    responses = by_name(spans, "response")
    greeting = [r for r in responses if parent_of(r, spans) is root]
    in_turn = [r for r in responses if parent_of(r, spans) is turn]
    assert len(greeting) == 1  # say() outside a user turn
    assert len(in_turn) == 2  # the tool call, then the answer
    assert all(r.attributes["voice_agent.response.status"] == "completed" for r in in_turn)
    assert "sunny in Paris" in in_turn[-1].attributes["voice_agent.agent.transcript"]

    if kind == "native":
        assert all(r.attributes["gen_ai.provider.name"] == "mock" for r in in_turn)
        assert "voice_agent.response.ttfb" in in_turn[-1].attributes
    else:
        chats = by_name(spans, "chat")
        assert len(chats) == 2  # the tool call and the answer (say() skips the LLM)
        for chat in chats:
            assert chat.attributes["gen_ai.operation.name"] == "chat"
            assert chat.attributes["gen_ai.provider.name"] == "mock"
            assert "gen_ai.usage.output_tokens" in chat.attributes
        assert any("voice_agent.llm.ttft" in c.attributes for c in chats)
        assert all(parent_of(c, spans) is turn for c in chats)
        tts = by_name(spans, "tts")
        assert tts and any(parent_of(t, spans) is turn for t in tts)
        assert any("voice_agent.tts.ttfb" in t.attributes for t in tts)
        stt = by_name(spans, "stt")
        assert stt and all(parent_of(s, spans) is turn for s in stt)


async def test_content_is_not_captured_by_default(exporter: Any) -> None:
    await run_tool_turn("native", SessionTracer(exporter.provider))
    for span in exporter.get_finished_spans():
        for key in span.attributes:
            assert "transcript" not in key and "arguments" not in key and "result" not in key


async def test_interrupted_turn_is_marked(exporter: Any) -> None:
    long_answer = "This is a very long answer that keeps going and going for quite a while. " * 3
    session = make_session(
        "native",
        transcripts=["tell me a story", "stop"],
        responses=[long_answer, "Okay."],
        realtime_factor=1.0,
        trace=SessionTracer(exporter.provider),
    )
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await session.start(Agent("x"), transport)
    await speak(transport, 0.6, 0.5)
    await wait_for(lambda: session.agent_state == AgentState.SPEAKING)
    await asyncio.sleep(0.8)
    await transport.play_user_audio(synth_speech(0.8, 16_000))  # barge in
    await wait_for(lambda: bool(rec.of("interrupted")))
    await session.aclose()

    spans = list(exporter.get_finished_spans())
    turns = sorted(by_name(spans, "turn"), key=lambda s: s.start_time)
    first = turns[0]
    assert first.attributes["voice_agent.turn.was_interrupted"] is True
    (event,) = [e for e in first.events if e.name == "interrupted"]
    assert event.attributes["voice_agent.played"] == pytest.approx(rec.of("interrupted")[0].played)
    assert first.start_time <= event.timestamp <= first.end_time
