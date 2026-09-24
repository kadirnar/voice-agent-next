# voice-agent-next

**Real-time speech-to-speech voice agents in Python — native S2S models and streaming cascades, local and cloud, on Linux, macOS and Windows, with a built-in benchmark suite.**

> Status: **pre-alpha** — the core runtime works end-to-end with mock providers; real providers are landing issue by issue. See the [roadmap](ROADMAP.md) and the [research report](docs/research/REPORT.md).

## Why

Voice agents are built in two ways:

* **Native speech-to-speech** models (OpenAI Realtime, Gemini Live, Nova Sonic, Moshi, Qwen-Omni, …) — lowest latency and most natural prosody;
* **cascades** (VAD → STT → LLM → TTS) — maximum control, any LLM, cheapest, fully local if you want.

voice-agent-next puts both behind **one engine interface**, so the session runtime — turn-taking, barge-in, truncation to what the user actually heard, tool calling, transcripts, metrics — behaves identically whichever engine you pick, and you can benchmark them against each other on the same scenarios.

## Install

```bash
pip install voice-agent-next            # core (numpy, pydantic, httpx, websockets)
pip install "voice-agent-next[audio]"   # + local microphone/speaker (sounddevice, soxr)
```

Provider extras (`[openai]`, `[google]`, `[silero]`, `[faster-whisper]`, `[kokoro]`, …) are listed by `van providers`.

## Quick start

```python
import asyncio
from voice_agent_next import Agent, AgentSession, function_tool
from voice_agent_next.transports import create_transport


@function_tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It is sunny in {city}."


async def main() -> None:
    agent = Agent("You are a friendly assistant.", tools=[get_weather], greeting="Hi!")

    # Native speech-to-speech model...
    session = AgentSession("openai/gpt-realtime")
    # ...or a cascade (any mix of local and cloud components):
    # session = AgentSession(stt="deepgram/nova-3", llm="openai/gpt-4.1-mini",
    #                        tts="cartesia/sonic-2", vad="silero", turn_detector="smart_turn")

    await session.run(agent, create_transport("local"))  # microphone + speakers


asyncio.run(main())
```

Try it offline right now (simulated user + mock engine):

```bash
van demo
van providers      # what's available, what's missing
van doctor         # environment check (audio devices, GPUs, API keys)
```

## Architecture (short)

```
Transport (mic/speaker, WebSocket, WebRTC, telephony, file)
   │  user audio                               ▲ agent audio (paced, truncatable)
   ▼                                           │
AgentSession ── barge-in · truncation · tools · history · metrics
   │                                           ▲
   ▼  EngineConnection (one event protocol)    │
 ┌────────────────────────────┬──────────────────────────────────────────┐
 │ native S2S engines         │ CascadeEngine                            │
 │ OpenAI Realtime, Gemini    │ VAD → STT → turn detector → LLM → TTS    │
 │ Live, Nova Sonic, Moshi…   │ (any provider, local or cloud)           │
 └────────────────────────────┴──────────────────────────────────────────┘
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Development

```bash
uv sync                 # create .venv with dev tools
uv run pytest           # unit tests (no network, no models)
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

Contributions follow [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under [Apache-2.0](LICENSE).
