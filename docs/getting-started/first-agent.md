# Your first agent

## 1. Try it offline

```bash
van demo
```

A simulated user talks to the mock speech-to-speech engine: no microphone, API keys or
downloads. It shows the transcripts, the barge-in handling and the turn metrics the real
runtime produces.

## 2. Pick a preset

A [preset](../presets.md) is a tested stack plus what it needs to run. `van presets` says
which ones are ready on this machine and prints the exact fixes for the others:

```bash
van presets                        # every preset: ready, or what is missing
van run --preset local-cpu         # fully offline: sherpa-onnx STT + Ollama + Kokoro
van run --preset openai-realtime   # native speech-to-speech (OPENAI_API_KEY)
van run                            # no preset: the best one that is ready here (it says which)
```

| Preset | Stack |
|---|---|
| `local-cpu`, `local-gpu`, `apple` | fully local cascades (Ollama for the LLM) |
| `hybrid` | local speech, cloud LLM with a local fallback |
| `cloud-fast`, `cloud-quality` | cloud cascades (Deepgram / AssemblyAI, Groq / Claude, Cartesia / ElevenLabs) |
| `openai-realtime`, `gemini-live` | native speech-to-speech engines |

When something is missing, `van run --preset` prints the fixes
(`pip install 'voice-agent-next[sherpa-onnx,kokoro]'`, `ollama pull ...`,
`export DEEPGRAM_API_KEY=...`) and exits.

## 3. Change what you need

Override any component with a flag, or start a config file from a preset:

```bash
van run --preset local-cpu --llm ollama/qwen3.5:4b
van run --stt deepgram/nova-3 --llm anthropic/claude-haiku-4-5 --tts cartesia --vad silero --turn smart_turn
```

```yaml
# agent.yaml  ->  van run -c agent.yaml
extends: local-cpu
llm: ollama/qwen3.5:4b
agent:
  instructions: You are a friendly assistant.
  greeting: Hi!
cascade:
  min_endpointing_delay: 0.3
```

Components are `provider/model` specs; the [provider index](../providers/index.md) lists
them all. A list of specs is a [failover chain](../concepts/failover.md).

## 4. In Python

```python
import asyncio

from voice_agent_next import Agent, AgentSession, function_tool
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
    session = build_session(load_preset("local-cpu"))
    # Or name any mix of components, or a native engine:
    # session = AgentSession(stt="deepgram/nova-3", llm="anthropic/claude-haiku-4-5",
    #                        tts="cartesia", vad="silero", turn_detector="smart_turn")
    # session = AgentSession("openai/gpt-realtime")

    session.on("user_transcript", lambda ev: ev.is_final and print("user:", ev.text))
    session.on("metrics", print)  # TurnMetrics (voice-to-voice latency...) and component metrics

    await session.run(agent, create_transport("local"))  # microphone + speakers


asyncio.run(main())
```

* `Agent` holds the instructions, the tools and the greeting. `function_tool` turns an
  async function into a tool: the signature becomes the JSON schema and the docstring the
  description ([tools](../concepts/tools.md)).
* `AgentSession` is the runtime: turn-taking, barge-in, truncation to what the user heard,
  tool calls, history and metrics. It behaves the same with every engine
  ([architecture](../ARCHITECTURE.md)).
* The transport moves audio: `local` (microphone and speakers), `file`, `websocket`,
  `webrtc` or a telephony provider ([transports](../transports/index.md)).

## Next steps

* [Concepts](../ARCHITECTURE.md): how engines, the cascade and the session fit together.
* [Turn-taking](../concepts/turn-taking.md) and [endpointing](../concepts/endpointing.md):
  when the agent answers.
* [Deploy](../deploy/index.md): serve agents to browsers and phones.
* [Examples](../examples.md): runnable scripts, most of them testable offline.
* [Benchmarks](../benchmarks/methodology.md): measure voice-to-voice latency of your stack.
