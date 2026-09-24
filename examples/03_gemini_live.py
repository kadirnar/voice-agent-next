"""Gemini Live (native speech-to-speech) with a tool, over microphone and speakers.

The engine speaks the Live API's raw WebSocket protocol (no Google SDK needed). It keeps
long calls alive by itself: session resumption, context-window compression, and a
switch to a fresh connection before the server's ``goAway`` (docs/providers/gemini-live.md).

Run::

    export GOOGLE_API_KEY=...                  # or GEMINI_API_KEY
    pip install 'voice-agent-next[audio]'
    python examples/03_gemini_live.py                              # mic + speakers
    python examples/03_gemini_live.py --voice Puck --model gemini-3.1-flash-live
    python examples/03_gemini_live.py --wav question.wav --output reply.wav
    python examples/03_gemini_live.py --mock                       # offline smoke test

``--mock`` points the real ``GeminiLiveEngine`` at ``FakeGeminiLiveServer``
(``voice_agent_next.testing``), a local server that replays the Live API protocol with
scripted replies. Use it in your own tests too.
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
from voice_agent_next.providers.google.live import GeminiLiveEngine
from voice_agent_next.testing.gemini_live import FakeGeminiLiveServer, FakeToolCall
from voice_agent_next.transports import FileTransport, Transport, create_transport


@function_tool
async def find_restaurant(cuisine: str) -> str:
    """Find a restaurant nearby serving the given cuisine."""
    return f"Trattoria Roma serves {cuisine} food, 300 m away, open until 23:00."


@contextlib.asynccontextmanager
async def live_engine(args: argparse.Namespace) -> AsyncIterator[GeminiLiveEngine]:
    if not args.mock:
        yield GeminiLiveEngine(model=args.model, voice=args.voice)  # key from the environment
        return
    fake = FakeGeminiLiveServer(
        transcripts=["Is there an Italian restaurant nearby?"],
        replies=[
            FakeToolCall("find_restaurant", {"cuisine": "Italian"}),
            "Trattoria Roma is 300 meters away.",
        ],
        chars_per_second=40,
    )
    async with fake:
        yield GeminiLiveEngine(model=args.model, api_key=fake.api_key, base_url=fake.url)


def make_transport(args: argparse.Namespace) -> Transport:
    if args.wav or args.mock:
        tmp = scratch_dir() if args.mock else Path()
        wav = args.wav or synthetic_question(tmp / "question.wav")
        args.output = args.output or tmp / "reply.wav"
        return FileTransport(wav, args.output, realtime=not args.mock, hold=0.5 if args.mock else 2)
    return create_transport("local")


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mock", action="store_true", help="offline: fake Live API server")
    parser.add_argument("--model", default="gemini-3.8-live")
    parser.add_argument("--voice", default="Kore", help="prebuilt voice: Kore, Puck, Charon...")
    parser.add_argument("--wav", type=Path, help="talk from a WAV file instead of the mic")
    parser.add_argument("--output", type=Path, help="with --wav: where to write the reply")
    args = parser.parse_args(argv)
    if not (args.mock or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        print("set GOOGLE_API_KEY or GEMINI_API_KEY (or run with --mock)", file=sys.stderr)
        return 1

    agent = Agent("You are a local guide. Answer in one sentence.", tools=[find_restaurant])
    async with live_engine(args) as engine:
        session = AgentSession(engine)
        log_conversation(session)
        await session.run(agent, make_transport(args))
    if args.output:
        print(f"reply written to {args.output}")
    return 0


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        sys.exit(asyncio.run(main()))
