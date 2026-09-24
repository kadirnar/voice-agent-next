# voice-agent-next

**Real-time speech-to-speech voice agents in Python — native S2S models and streaming cascades, local and cloud, on Linux, macOS and Windows, with a built-in benchmark suite.**

> Status: **alpha** — 52 providers (local and cloud), native speech-to-speech engines and streaming cascades behind one runtime, a benchmark suite; APIs may still change. See the [roadmap](ROADMAP.md) and the [research report](docs/research/REPORT.md).

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

Pick a [preset](docs/presets.md), a tested stack, and talk to it:

```bash
van presets                      # which presets run on this machine, and what the others need
van run --preset local-cpu       # fully offline: sherpa-onnx streaming STT + Ollama + Kokoro
van run --preset openai-realtime # or cloud-fast, cloud-quality, gemini-live, local-gpu, apple, hybrid
van run                          # no preset: the best one that is ready here (it says which)
```

`van run --preset` checks the preset first. When something is missing, it prints the exact
fixes (`pip install 'voice-agent-next[sherpa-onnx,kokoro,...]'`, `ollama pull ...`,
`export DEEPGRAM_API_KEY=...`). Presets are starting points. Override any component with a
flag (`--llm ollama/qwen3.5:4b`), or with a config file:

```yaml
# agent.yaml  ->  van run -c agent.yaml
extends: local-cpu
llm: ollama/qwen3.5:4b
agent:
  instructions: You are a friendly assistant.
  greeting: Hi!
```

In Python:

```python
import asyncio
from voice_agent_next import Agent, function_tool
from voice_agent_next.app import build_session
from voice_agent_next.presets import load_preset
from voice_agent_next.transports import create_transport


@function_tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It is sunny in {city}."


async def main() -> None:
    agent = Agent("You are a friendly assistant.", tools=[get_weather], greeting="Hi!")

    # Checks the preset (raises with the fixes if it cannot run here), then builds it.
    session = build_session(load_preset("local-cpu"))  # or "cloud-fast", "openai-realtime"...
    # Without presets, name any mix of components:
    # session = AgentSession(stt="deepgram/nova-3", llm="anthropic/claude-haiku-4-5",
    #                        tts="cartesia", vad="silero", turn_detector="smart_turn")

    await session.run(agent, create_transport("local"))  # microphone + speakers


asyncio.run(main())
```

Try it offline right now (simulated user + mock engine):

```bash
van demo
van providers      # what's available, what's missing
van doctor         # environment check (audio devices, GPUs, API keys)
```

More runnable scenarios (local agent, OpenAI Realtime, Gemini Live, telephony, tools, benchmarks, each with an offline `--mock` mode): [examples/README.md](examples/README.md).

## Providers

Every component is addressed by a `provider/model` spec and installed through an extra; `van providers` shows what is ready on your machine.

| | Local | Cloud |
|---|---|---|
| **Speech-to-speech engines** | any OpenAI-Realtime-compatible server: Speaches, LocalAI, vLLM-Omni | OpenAI Realtime, Gemini Live, Azure OpenAI Realtime, xAI Grok Voice, Qwen-Omni Realtime |
| **STT** | sherpa-onnx (streaming Zipformer/NeMo, Parakeet, Moonshine, SenseVoice, Whisper), faster-whisper (CPU / CUDA) | Deepgram Nova-3 & Flux, AssemblyAI Universal-Streaming, ElevenLabs Scribe v2, OpenAI transcribe, Cartesia Ink |
| **LLM** | Ollama, llama.cpp, vLLM, LM Studio | OpenAI, Anthropic Claude, Google Gemini, Groq, Cerebras, Together, OpenRouter, DeepSeek, Fireworks, SambaNova |
| **TTS** | Kokoro-82M, sherpa-onnx (Piper/VITS, Kokoro, Matcha), Kokoro-FastAPI | Cartesia Sonic, ElevenLabs Flash/v3, OpenAI gpt-4o-mini-tts, Gemini TTS, Deepgram Aura-2 |
| **VAD & turn-taking** | Silero VAD v6, TEN VAD, energy VAD, Smart Turn v3.2 | STT-native turn events (Deepgram Flux, AssemblyAI, Cartesia Ink) |
| **Transports** | microphone/speakers (with WebRTC echo cancellation), files, loopback | WebSocket + browser client, telephony (Twilio, Telnyx, Vonage, Plivo) |
| **Serving & ops** | `van serve`: any engine behind the OpenAI Realtime protocol | failover chains, call recording (stereo WAV + JSONL), OpenTelemetry tracing, GPU auto-selection |

In progress ([roadmap](ROADMAP.md)): omni models (LFM2.5-Audio), presets, WebRTC, Moonshine, Pocket TTS, async tools, Moshi, MLX on Apple Silicon.

## Benchmarks

`van bench latency` drives a simulated caller through the real runtime and measures voice-to-voice latency **on the call recording** (end of user speech → first agent audio), with a reproducibility manifest for every run. A fully local pipeline (Silero + Smart Turn + faster-whisper `base` + Ollama LFM2.5-1.2B + Kokoro) on a Ryzen 5 5600 CPU: **1.13 s p50 / 1.63 s p90**, and **0.97 s p50** with GPU speech recognition on an RTX 5070 Ti; the runtime itself adds ≈ 2 ms (checked on every PR). Details and methodology: [docs/benchmarks/results.md](docs/benchmarks/results.md), [benchmarks/README.md](benchmarks/README.md).

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
