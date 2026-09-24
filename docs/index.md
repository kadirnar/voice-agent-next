# voice-agent-next

**Real-time speech-to-speech voice agents in Python.** Native speech-to-speech models and
streaming cascades, local and cloud, on Linux, macOS and Windows, with a built-in
benchmark suite.

!!! warning "Alpha"
    52 providers (local and cloud), native speech-to-speech engines and streaming cascades
    behind one runtime, and a benchmark suite. APIs may still change.

Voice agents are built in two ways:

* **Native speech-to-speech** models (OpenAI Realtime, Gemini Live, …): lowest latency and
  the most natural prosody;
* **cascades** (VAD → STT → LLM → TTS): maximum control, any LLM, cheapest, fully local if
  you want.

voice-agent-next puts both behind **one engine interface**. The session runtime (turn-taking,
barge-in, truncation to what the user actually heard, tool calling, transcripts, metrics)
behaves the same whichever engine you pick, and you can benchmark engines against each
other on the same scenarios.

```bash
pip install "voice-agent-next[audio]"
van presets                      # which tested stacks run on this machine
van run --preset local-cpu       # fully offline: sherpa-onnx streaming STT + Ollama + Kokoro
van run --preset openai-realtime # or cloud-fast, cloud-quality, gemini-live, local-gpu, apple, hybrid
```

```python
from voice_agent_next import Agent, AgentSession
from voice_agent_next.transports import create_transport

session = AgentSession(
    stt="deepgram/nova-3",
    llm="anthropic/claude-haiku-4-5",
    tts="cartesia",
    vad="silero",
    turn_detector="smart_turn",
)
await session.run(Agent("You are a friendly assistant."), create_transport("local"))
```

## Where to go next

<div class="grid cards" markdown>

* **[Getting started](getting-started/installation.md)**

    Install per OS, run a preset, build your first agent in Python, the `van` CLI.

* **[Concepts](ARCHITECTURE.md)**

    Engines and events, the cascade, turn-taking, endpointing, interruptions, preemptive
    generation, tools, failover, observability.

* **[Providers](providers/index.md)**

    Every STT, LLM, TTS, VAD, turn detector and speech-to-speech engine, generated from the
    registry.

* **[Transports](transports/index.md) and [deploy](deploy/index.md)**

    Microphone and speakers, WebSocket, WebRTC, telephony, and an OpenAI-Realtime-compatible
    server for any engine.

* **[Benchmarks](benchmarks/methodology.md)**

    Voice-to-voice latency measured on the call recording, ASR accuracy, framework
    overhead, and [results](benchmarks/results.md).

* **[API reference](reference/index.md)**

    The public API, generated from the docstrings.

</div>

## Architecture in short

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
 │ Live, compatible servers   │ (any provider, local or cloud)           │
 └────────────────────────────┴──────────────────────────────────────────┘
```

Details: [architecture](ARCHITECTURE.md). Background: the
[research report](research/REPORT.md). Source, issues and roadmap:
[GitHub](https://github.com/kadirnar/voice-agent-next).
