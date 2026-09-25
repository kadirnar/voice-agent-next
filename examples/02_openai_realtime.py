"""OpenAI Realtime (native speech-to-speech) with a tool, over microphone and speakers.

One model hears the audio and speaks the answer: no separate STT/LLM/TTS. The session
still does the work around it: barge-in with truncation (the model forgets what the user
did not hear), tool execution, transcripts and latency metrics (docs/providers/openai-realtime.md).

Run::

    export OPENAI_API_KEY=sk-...
    pip install 'voice-agent-next[audio]'      # raw WebSocket: no OpenAI SDK needed
    python examples/02_openai_realtime.py                          # mic + speakers
    python examples/02_openai_realtime.py --voice marin --model gpt-realtime-2.1-mini
    python examples/02_openai_realtime.py --wav question.wav --output reply.wav
    python examples/02_openai_realtime.py --mock                   # offline smoke test

``--mock`` starts this library's OpenAI-Realtime-compatible server (``van serve``) on
localhost with the scripted mock engine behind it, and points the *real*
``OpenAIRealtimeEngine`` client at it. The whole protocol runs, only the model is fake.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path

from _common import log_conversation, scratch_dir, synthetic_question

from voice_agent_next import Agent, AgentSession, function_tool
from voice_agent_next.providers.mock import MockEngine, MockToolCall
from voice_agent_next.providers.openai.realtime import OpenAIRealtimeEngine
from voice_agent_next.server import RealtimeServer
from voice_agent_next.transports import FileTransport, Transport, create_transport


@function_tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny and 22 degrees in {city}."  # call a real weather API here


@contextlib.asynccontextmanager
async def realtime_endpoint(args: argparse.Namespace) -> AsyncIterator[OpenAIRealtimeEngine]:
    """The engine: OpenAI's API, or (``--mock``) a local server that speaks its protocol."""
    if not args.mock:
        # api_key defaults to $OPENAI_API_KEY, base_url to wss://api.openai.com/v1
        yield OpenAIRealtimeEngine(model=args.model, voice=args.voice)
        return
    fake_model = MockEngine(
        transcripts=["What's the weather in Paris?"],
        responses=[MockToolCall("get_weather", {"city": "Paris"}), "It's sunny in Paris."],
        chars_per_second=40,
    )
    async with RealtimeServer(fake_model, model=args.model, port=0, api_keys="mock-key") as server:
        yield OpenAIRealtimeEngine(base_url=server.url, api_key="mock-key", model=args.model)


def make_transport(args: argparse.Namespace) -> Transport:
    if args.wav or args.mock:
        tmp = scratch_dir() if args.mock else Path()
        wav = args.wav or synthetic_question(tmp / "question.wav")
        args.output = args.output or tmp / "reply.wav"
        # a cloud model expects real-time input: pace the file unless it is the mock
        return FileTransport(wav, args.output, realtime=not args.mock, hold=0.5 if args.mock else 2)
    return create_transport("local")


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mock", action="store_true", help="offline: local fake endpoint")
    parser.add_argument("--model", default="gpt-realtime-2.1")
    parser.add_argument("--voice", default=None, help="e.g. marin, cedar, alloy")
    parser.add_argument("--wav", type=Path, help="talk from a WAV file instead of the mic")
    parser.add_argument("--output", type=Path, help="with --wav: where to write the reply")
    args = parser.parse_args(argv)
    if not args.mock and not os.environ.get("OPENAI_API_KEY"):
        print("set OPENAI_API_KEY (or run with --mock)", file=sys.stderr)
        return 1

    agent = Agent(
        "You are a cheerful weather assistant. Keep answers short.",
        tools=[get_weather],
    )
    async with realtime_endpoint(args) as engine:
        session = AgentSession(engine)
        log_conversation(session)
        await session.run(agent, make_transport(args))
    if args.output:
        print(f"reply written to {args.output}")
    return 0


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        sys.exit(asyncio.run(main()))
