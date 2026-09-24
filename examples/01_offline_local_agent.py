"""A fully local voice agent: the ``local-cpu`` preset over your microphone and speakers.

Nothing leaves the machine: sherpa-onnx streaming STT, a small Ollama LLM, Kokoro TTS,
Silero VAD and Smart Turn (see docs/presets.md). Talk to it, interrupt it, press Ctrl-C
to stop.

Setup (once)::

    pip install 'voice-agent-next[audio,sherpa-onnx,kokoro,silero,smart-turn]'
    ollama serve & ollama pull LiquidAI/lfm2.5-1.2b-instruct
    van presets local-cpu            # checks everything and prints what is missing

Run::

    python examples/01_offline_local_agent.py                         # mic + speakers
    python examples/01_offline_local_agent.py --wav question.wav --output reply.wav
    python examples/01_offline_local_agent.py --llm ollama/qwen3.5:4b # better tool calling
    python examples/01_offline_local_agent.py --mock                  # offline smoke test

``--mock`` swaps the preset for the scripted mock cascade and the audio devices for a
synthetic WAV file, so it runs anywhere in about a second (this is what the tests run).
The same program is ``van run --preset local-cpu`` on the command line.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from _common import log_conversation, scratch_dir, synthetic_question

from voice_agent_next import Agent, AgentSession, CascadeOptions, ConfigurationError, read_wav
from voice_agent_next.presets import session_from_preset
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import MockLLM, MockSTT, MockTTS
from voice_agent_next.transports import FileTransport, Transport, create_transport

INSTRUCTIONS = "You are a friendly local assistant. Answer in one or two short sentences."
GREETING = "Hi! I run entirely on this computer. What can I do for you?"


def build(args: argparse.Namespace) -> tuple[AgentSession, Agent]:
    if args.mock:
        # The same cascade shape as local-cpu (VAD -> STT -> LLM -> TTS), with scripted
        # components: the STT "hears" a fixed sentence, the LLM answers from a script.
        session = AgentSession(
            stt=MockSTT(transcripts=["What can you do offline?"]),
            llm=MockLLM(responses=["I can chat, call tools and keep your audio private."]),
            tts=MockTTS(chars_per_second=40),  # 40 chars/s: keeps the smoke test short
            vad=EnergyVAD(),
            cascade_options=CascadeOptions(min_endpointing_delay=0.0),
        )
        return session, Agent(INSTRUCTIONS)
    # load_preset / session_from_preset check the machine first (extras, Ollama model...)
    # and raise ConfigurationError with the exact fixes when something is missing.
    overrides: dict[str, object] = {"agent": {"instructions": INSTRUCTIONS, "greeting": GREETING}}
    if args.llm:
        overrides["llm"] = args.llm
    return session_from_preset("local-cpu", **overrides)


def make_transport(args: argparse.Namespace) -> Transport:
    if args.wav or args.mock:
        tmp = scratch_dir() if args.mock else Path()
        wav = args.wav or synthetic_question(tmp / "question.wav")
        output = args.output or tmp / "reply.wav"
        args.output = output
        # realtime=False feeds the file as fast as possible (fine for local/mock engines);
        # hold keeps the call open until the agent has been quiet this long.
        return FileTransport(wav, output, realtime=not args.mock, hold=0.5 if args.mock else 1.5)
    return create_transport("local")  # default microphone and speakers (extra: audio)


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mock", action="store_true", help="offline mock cascade + WAV file")
    parser.add_argument("--wav", type=Path, help="talk from a WAV file instead of the microphone")
    parser.add_argument("--output", type=Path, help="with --wav: where to write the reply")
    parser.add_argument("--llm", help="override the preset's LLM, e.g. ollama/qwen3.5:4b")
    args = parser.parse_args(argv)

    try:
        session, agent = build(args)
    except ConfigurationError as exc:  # the preset cannot run here: print the fixes
        print(exc, file=sys.stderr)
        return 1
    transport = make_transport(args)
    log_conversation(session)
    await session.run(agent, transport)  # returns when the file ends / the call closes

    if args.output:
        reply = read_wav(args.output)
        print(f"agent audio: {reply.duration:.1f} s written to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        pass
