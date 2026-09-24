"""Mix cloud and local components in one cascade, with failover.

    STT  Deepgram Nova-3 (cloud)  -> sherpa-onnx Kroko (local backup)
    LLM  Ollama qwen3.5:4b (local, private)
    TTS  Cartesia Sonic 3.6 (cloud) -> Kokoro (local backup)

A list of providers is a failover chain (docs/concepts/failover.md). A request moves to
the next provider when the current one errors or stalls, but only while nobody can hear
the switch: the TTS switches before the sentence's first audio, the LLM before its first
token. The STT replays the current utterance into the backup. Failed providers cool down
and are tried again later.

Run::

    export DEEPGRAM_API_KEY=... CARTESIA_API_KEY=...   # either may be missing: see below
    pip install 'voice-agent-next[audio,sherpa-onnx,openai,kokoro,silero]'  # openai: Ollama client
    ollama pull qwen3.5:4b
    python examples/04_cascade_mix.py --dry-run     # check this machine, print the plan
    python examples/04_cascade_mix.py               # mic + speakers
    python examples/04_cascade_mix.py --mock        # offline: a stalled "cloud" TTS fails over

The readiness check prunes chains to the members that can run here: without
``CARTESIA_API_KEY`` the agent simply speaks with Kokoro.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from pathlib import Path
from typing import Any

from _common import log_conversation, scratch_dir, synthetic_question

from voice_agent_next import AgentSession, CascadeOptions, FallbackTTS
from voice_agent_next.app import build_agent, build_session
from voice_agent_next.config import AppConfig
from voice_agent_next.presets import check_config
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import MockLLM, MockSTT, MockTTS
from voice_agent_next.session import Agent
from voice_agent_next.transports import FileTransport, create_transport

# The same shape as a YAML config file (`van run --config agent.yaml`).
CONFIG: dict[str, Any] = {
    "stt": ["deepgram/nova-3", "sherpa-onnx/zipformer-en-kroko"],  # a list = failover chain
    "llm": "ollama/qwen3.5:4b",
    "tts": {
        "fallback": ["cartesia/sonic-3.6", "kokoro/v1.0-fp16"],
        "first_audio_timeout": 2.0,  # a TTS that stays silent this long counts as failed
    },
    "vad": "silero",
    "agent": {"instructions": "You are a concise travel assistant."},
}


def mock_session() -> tuple[AgentSession, Agent]:
    """The same failover, offline: the first TTS is 'cloud' and never answers in time."""
    tts = FallbackTTS(
        [
            MockTTS(model="stalled-cloud-tts", ttfb=30.0),  # a provider that hangs
            MockTTS(model="local-tts", chars_per_second=40),
        ],
        first_audio_timeout=0.3,
    )
    tts.on(
        "provider_failover",
        lambda ev: print(
            f"  failover: {ev.kind} {ev.from_provider} -> {ev.to_provider} ({ev.reason})"
        ),
    )
    session = AgentSession(
        stt=MockSTT(transcripts=["Find me a train to Lyon."]),
        llm=MockLLM(responses=["The next train to Lyon leaves at nine."]),
        tts=tts,
        vad=EnergyVAD(),
        cascade_options=CascadeOptions(min_endpointing_delay=0.0),
    )
    return session, Agent(CONFIG["agent"]["instructions"])


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mock", action="store_true", help="offline mock with a TTS failover")
    parser.add_argument("--dry-run", action="store_true", help="check readiness, then exit")
    parser.add_argument("--wav", type=Path, help="talk from a WAV file instead of the mic")
    args = parser.parse_args(argv)

    if args.mock:
        session, agent = mock_session()
        tmp = scratch_dir()
        transport: Any = FileTransport(
            synthetic_question(tmp / "q.wav"), tmp / "reply.wav", realtime=False, hold=0.5
        )
    else:
        readiness = check_config(CONFIG, transport=None if args.wav else "local")
        print(readiness.explain())
        for note in readiness.notes:  # e.g. "tts: skipping cartesia/sonic-3.6 (...)"
            print(f"note: {note}")
        if not readiness.ready or args.dry_run:
            return 0 if readiness.ready else 1
        cfg = AppConfig.model_validate(readiness.config)  # chains pruned to what can run
        session, agent = build_session(cfg), build_agent(cfg)
        transport = FileTransport(args.wav, "reply.wav") if args.wav else create_transport("local")

    log_conversation(session)
    await session.run(agent, transport)
    return 0


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        sys.exit(asyncio.run(main()))
