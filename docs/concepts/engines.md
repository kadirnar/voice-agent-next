# Engines and events

Every way of building a voice agent is a **speech-to-speech engine**. A native audio model
(OpenAI Realtime, Gemini Live, Moshi, an OpenAI-Realtime-compatible local server) and a streaming
cascade (VAD → STT → turn detector → LLM → TTS) implement the same interface and emit the
same events. The session runtime, the transports, the metrics and the benchmarks are
written once against that interface.

```python
from voice_agent_next import AgentSession

AgentSession("openai/gpt-realtime")  # a native engine, by spec
AgentSession("google/gemini-3.8-live")  # another one
AgentSession(stt="deepgram/nova-3", llm="groq", tts="cartesia", vad="silero")  # a cascade
```

## The interface

| Class | Role |
|---|---|
| `S2SEngine` | A factory: configuration, `warmup()` (load models, pre-open connections), `connect(EngineOptions)` |
| `EngineConnection` | One live conversation: audio in, events out, control calls |
| `EngineOptions` | Per-connection instructions, tools, initial history, voice, language, temperature, `turn_detection` |
| `EngineCapabilities` | What the engine can do, declared rather than guessed |

`EngineConnection` control calls: `send_audio`, `commit_input`, `clear_input`,
`send_text`, `create_response`, `cancel_response`, `truncate(item_id, audio_end_ms)`,
`interrupt(item_id, played_ms)`, `send_tool_output`, `send_async_tool_output`, `update`,
`say`. See the [API reference](../reference/engine.md).

## The event protocol

`EngineConnection.events()` yields, per user turn:

```
InputSpeechStarted → InputTranscript(partial)* → InputSpeechStopped
  → InputCommitted → InputTranscript(final)
  → ResponseStarted → (ResponseAudio | ResponseText | ResponseToolCall)* → ResponseDone
+ ToolCallCancelled, EngineStatus (rotation/reconnect notices), EngineErrorEvent
```

Rules every engine follows:

* `InputSpeechStopped.audio_time` is where speech **ended** in the input stream, not where
  the silence was confirmed. `audio_time_to_wall()` maps it to capture time, which makes
  voice-to-voice latency comparable across engines.
* `ResponseText` is the transcript of what is *spoken*; `ResponseAudio` carries
  `AudioFrame`s at the engine's output rate.
* `truncate()` / `interrupt()` return the transcript that was actually heard when the
  engine knows it (the cascade does, from per-sentence alignment); otherwise the session
  estimates it from the played duration.

## Capabilities

| Capability | Meaning |
|---|---|
| `native_audio` | End-to-end audio model (`False` for cascades) |
| `server_turn_detection` | The engine decides when the user's turn ends (server VAD, semantic VAD, the cascade's endpointing) |
| `truncation` | The assistant item can be truncated to what the user heard |
| `full_duplex` | Listens while speaking (backchannels, overlaps) |
| `text_input` | Accepts injected text messages |
| `tool_mode` | `blocking` (the model waits for results), `non_blocking` (keeps talking, e.g. Gemini Live) or `delegation` (a backend agent answers) |
| `max_session_duration` | Hard provider session limit; engines rotate transparently |

The session reads these instead of special-casing providers: for example, it only calls
`truncate()` on engines that support it, and schedules non-blocking tool results natively
where the engine can.

## Native engines

| Engine | Page |
|---|---|
| OpenAI Realtime, Azure OpenAI, xAI Grok Voice, Qwen-Omni Realtime, vLLM-Omni, Speaches, LocalAI | [OpenAI Realtime (+ compatible)](../providers/openai-realtime.md) |
| Gemini Live | [Gemini Live](../providers/gemini-live.md) |
| Moshi, PersonaPlex (local, full-duplex) | [Moshi and PersonaPlex](../providers/moshi.md) |
| `mock` | a deterministic engine for tests, `van demo` and benchmarks |

Native engines detect turns on the server by default (`EngineOptions.turn_detection`);
set it to `False` to commit turns yourself with `commit_input()` (push-to-talk).

Any engine can also be served to other clients over the OpenAI Realtime protocol with
[`van serve`](../deploy/realtime-server.md).

## Cascades

`CascadeEngine` builds an engine from components. See [the cascade](cascade.md).

## Writing an engine

Subclass `S2SEngine` and `EngineConnection`, emit `voice_agent_next.events` from
`events()`, declare `EngineCapabilities`, and register the class with
`@register_provider("engine", "<name>")` ([contributing](../contributing.md#adding-a-provider)).
