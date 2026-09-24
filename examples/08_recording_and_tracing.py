"""Record calls (stereo WAV + JSONL timeline) and export OpenTelemetry traces.

* ``record=`` writes the call as the caller heard it: user on the left channel, agent on
  the right, barge-in cuts included. A JSONL timeline of every session and engine event
  uses the same clock.
* ``trace=`` exports spans (``session > turn > stt / chat / tts / response /
  execute_tool``) through whatever OpenTelemetry SDK and exporter you configure.

See docs/concepts/observability.md.

Run::

    python examples/08_recording_and_tracing.py --mock                  # offline
    python examples/08_recording_and_tracing.py --engine openai/gpt-realtime-2.1 --wav q.wav
    pip install 'voice-agent-next[otel]' opentelemetry-sdk               # to see the spans

After a run, open the WAV in Audacity. The gap between the end of the left channel and
the start of the right channel is the voice-to-voice latency the caller heard.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from _common import log_conversation, scratch_dir, synthetic_question

from voice_agent_next import Agent, AgentSession, function_tool, read_wav
from voice_agent_next.providers.mock import MockEngine, MockToolCall
from voice_agent_next.session import SessionTracer
from voice_agent_next.transports import FileTransport


@function_tool
async def check_order(order_id: str) -> str:
    """Look up the status of an order."""
    return f"Order {order_id} shipped yesterday."


def make_tracer() -> tuple[SessionTracer | None, Any]:
    """A tracer that keeps spans in memory (swap in an OTLP exporter for Jaeger/Tempo...)."""
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    except ImportError:
        print("(tracing skipped: pip install 'voice-agent-next[otel]' opentelemetry-sdk)")
        return None, None
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # In production: provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    # and `trace=True`, which uses the global provider.
    return SessionTracer(provider, capture_content=True), exporter


def print_spans(exporter: Any) -> None:
    spans = exporter.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}

    def depth(span: Any) -> int:
        return 0 if span.parent is None or span.parent.span_id not in by_id else 1 + depth(by_id[span.parent.span_id])

    print(f"\n{len(spans)} spans:")
    for span in sorted(spans, key=lambda s: s.start_time):
        ms = (span.end_time - span.start_time) / 1e6
        print(f"  {'  ' * depth(span)}{span.name}  {ms:.0f} ms")


def summarize_timeline(path: Path) -> None:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    print(f"\ntimeline {path.name}: {len(events)} events")
    print("  by source:", dict(Counter(e["source"] for e in events)))
    for e in events:
        if e["event"] == "metrics" and e["data"].get("type") == "turn":
            v2v = e["data"]["voice_to_voice"] or 0.0
            print(f"  t={e['t']:.2f} s  turn: voice-to-voice {v2v * 1000:.0f} ms, "
                  f"tool calls {e['data']['tool_calls']}")  # fmt: skip


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mock", action="store_true", help="offline mock engine")
    parser.add_argument("--engine", default="openai/gpt-realtime-2.1")
    parser.add_argument("--wav", type=Path, help="the user's side of the call (a WAV file)")
    parser.add_argument("--record-dir", type=Path, help="where recordings go")
    args = parser.parse_args(argv)
    if not args.mock and not args.wav:
        parser.error("--wav is required without --mock")

    tmp = scratch_dir()
    record_dir = args.record_dir or (tmp / "recordings" if args.mock else Path("recordings"))
    engine: Any = args.engine
    if args.mock:
        engine = MockEngine(transcripts=["Where is my order 42?"], chars_per_second=40,
                            responses=[MockToolCall("check_order", {"order_id": "42"}),
                                       "It shipped yesterday."])  # fmt: skip
    tracer, exporter = make_tracer()

    # record= takes a directory (one WAV + JSONL per session) or a .wav path
    session = AgentSession(engine, record=record_dir, trace=tracer)
    log_conversation(session)
    wav = args.wav or synthetic_question(tmp / "question.wav")
    # real-time pacing, even for the mock: the recording places user audio as it arrives
    transport = FileTransport(wav, trailing_silence=0.8, hold=0.5 if args.mock else 1.5)
    await session.run(Agent("You are a support agent.", tools=[check_order]), transport)

    recorder = session.recorder
    assert recorder is not None and recorder.wav_path and recorder.timeline_path
    audio = read_wav(recorder.wav_path)
    print(f"\nrecording {recorder.wav_path} ({audio.channels} channels, {audio.duration:.1f} s)")
    summarize_timeline(recorder.timeline_path)
    if exporter is not None:
        print_spans(exporter)
    return 0


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        sys.exit(asyncio.run(main()))
