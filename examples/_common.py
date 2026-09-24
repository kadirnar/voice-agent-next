"""Small helpers shared by the example scripts (not part of the library).

Every example runs from the repository root as ``python examples/NN_name.py`` (Python puts
the script's directory on ``sys.path``, so ``import _common`` works). Copy what you need
from here into your own code: there is nothing magic in it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from voice_agent_next import AgentSession, write_wav
from voice_agent_next.metrics import TurnMetrics
from voice_agent_next.providers.mock import synth_speech


def log_conversation(session: AgentSession, prefix: str = "") -> None:
    """Print the conversation as it happens: transcripts, tool calls and turn latency."""
    agent_text: dict[str, str] = {}

    def on_user(ev: Any) -> None:
        if ev.is_final:
            print(f"{prefix}user : {ev.text}", flush=True)

    def on_agent(ev: Any) -> None:
        # the agent's transcript arrives in pieces: print whole responses at turn end
        agent_text[ev.response_id] = agent_text.get(ev.response_id, "") + ev.delta

    def on_metrics(m: Any) -> None:
        if not isinstance(m, TurnMetrics):
            return  # per-component metrics (STT/LLM/TTS...) are also emitted here
        flush_agent()
        if m.voice_to_voice is not None:
            print(f"{prefix}       (voice-to-voice {m.voice_to_voice * 1000:.0f} ms)", flush=True)

    def flush_agent() -> None:
        for text in agent_text.values():
            if text.strip():
                print(f"{prefix}agent: {text.strip()}", flush=True)
        agent_text.clear()

    session.on("user_transcript", on_user)
    session.on("agent_transcript", on_agent)
    session.on("metrics", on_metrics)
    session.on("tool_call", lambda ev: print(f"{prefix}  -> tool {ev.call.name}({ev.call.arguments})"))
    session.on("tool_result", lambda ev: print(f"{prefix}  <- {ev.output.output!r}"))
    session.on("error", lambda ev: print(f"{prefix}error: {ev.error}", flush=True))
    session.on("close", lambda ev: (flush_agent(), print(f"{prefix}(session closed: {ev.reason})")))


def synthetic_question(path: str | Path | None = None, seconds: float = 1.0) -> Path:
    """Write a speech-like test WAV (a modulated tone) and return its path.

    The mock providers cannot understand words: they only detect that *something* was
    said and return scripted transcripts. Real engines need real speech (record a WAV).
    """
    if path is None:
        path = Path(tempfile.mkdtemp(prefix="van-example-")) / "question.wav"
    path = Path(path)
    write_wav(path, synth_speech(seconds, 16_000))
    return path


def scratch_dir() -> Path:
    """A fresh temporary directory for the files a mock run writes."""
    return Path(tempfile.mkdtemp(prefix="van-example-"))
