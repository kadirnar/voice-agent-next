# Architecture

voice-agent-next is organized around one idea: **every way of building a voice agent is a speech-to-speech engine**. Native audio models (OpenAI Realtime, Gemini Live, Moshi...) and streaming cascades (VAD → STT → LLM → TTS) implement the same interface and speak the same event protocol, so everything around them — the session runtime, transports, metrics, benchmarks — is shared.

```
               ┌──────────────────────────── AgentSession ────────────────────────────┐
 user audio    │  input loop ──► [AudioProcessor: AEC/NS] ──► EngineConnection.send_audio │
 ─────────────►│                                                                        │
  Transport    │  event loop ◄── EngineConnection.events() (voice_agent_next.events)    │
 ◄─────────────│  playout loop: real-time pacing, played-position tracking             │
 agent audio   │  barge-in · truncation · tools · history · TurnMetrics · usage        │
               └────────────────────────────────────────────────────────────────────────┘
                                        │ S2SEngine.connect(EngineOptions)
                  ┌─────────────────────┴──────────────────────────┐
         native engines (providers/*)                    CascadeEngine (engines/cascade.py)
   OpenAI Realtime + compat profiles, Gemini Live,   VAD ─► STT ─► turn detector ─► LLM ─► TTS
   Nova Sonic, GPT-Live, Moshi/PersonaPlex, ...      (any registered providers, local or cloud)
```

## Layers

| Layer | Module(s) | Responsibility |
|---|---|---|
| Audio primitives | `audio/` | `AudioFrame` (s16le + sample rate + channels + capture timestamp), buffers, fixed-size chunking, streaming resampling (soxr or numpy polyphase), bit-exact G.711, WAV I/O, `AudioProcessor` hooks |
| Components | `stt.py`, `tts.py`, `llm.py`, `vad.py`, `turn.py` | provider-neutral interfaces + streaming base classes that handle resampling, error propagation and metrics |
| Engine | `engine.py`, `events.py` | `S2SEngine` factory, `EngineConnection` live session, `EngineCapabilities`, the event protocol |
| Engines | `engines/cascade.py`, `providers/*` | the cascade; native engines live with their provider |
| Session | `session/` | `Agent` (instructions, tools, hooks), `AgentSession` runtime |
| Transports | `transports/` | move audio between user and session: loopback, file, local devices, WebSocket, WebRTC, telephony |
| Registry & config | `registry.py`, `config.py`, `app.py` | `provider/model` specs → instances; YAML/TOML/JSON configs; presets |
| Tooling | `cli/`, `bench/`, `testing/` | `van` CLI, benchmark suite, test helpers |

## The event protocol

`EngineConnection.events()` yields (see `events.py`):

```
InputSpeechStarted → InputTranscript(partial)* → InputSpeechStopped
  → InputCommitted → InputTranscript(final)
  → ResponseStarted → (ResponseAudio | ResponseText | ResponseToolCall)* → ResponseDone
+ ToolCallCancelled, EngineStatus (rotation/reconnect notices), EngineErrorEvent
```

Control goes the other way: `send_audio`, `commit_input`, `clear_input`, `send_text`, `create_response`, `cancel_response`, `truncate(item_id, audio_end_ms)`, `interrupt(item_id, played_ms)`, `send_tool_output`, `update`, `say`.

Rules every engine follows:

* `InputSpeechStopped.audio_time` is where speech **ended** in the input stream (not where silence was confirmed); `EngineConnection.audio_time_to_wall()` maps it to capture time, which makes voice-to-voice metrics comparable across engines.
* `ResponseText` is the transcript of what is *spoken*; `ResponseAudio` carries `AudioFrame`s at the engine's output rate.
* `truncate()`/`interrupt()` return the transcript that was actually heard when the engine knows it (the cascade does, from per-sentence alignment); otherwise the session estimates it.
* Capabilities are declared, not guessed: `native_audio`, `server_turn_detection`, `truncation`, `full_duplex`, `tool_mode` (blocking / non-blocking / delegation), `max_session_duration`, ...

## The session runtime

* **Start:** `engine.warmup()` (model loads, connection pre-opening; `SessionOptions.warmup`) runs before the transport opens, then `engine.connect()`. Cold starts otherwise land on the first turn (a local LLM: 582 ms cold vs 10 ms warm TTFT).
* **Input loop:** transport frames → optional processors (echo cancellation needs the played audio as a reference, which the playout loop feeds via `process_render`) → engine.
* **Playout loop:** engine audio is resampled to the transport format and paced against a virtual playback clock with a small look-ahead (`SessionOptions.output_lookahead`, 150 ms). The session therefore always knows how much of each response the user has heard, independent of transport buffering. Transports that know the real playback position (`capabilities.playback_position`: device output latency, a browser client's playback reports) refine it: the part of `buffered_duration()` the virtual clock doesn't account for is the listener's lag, which truncation subtracts and which keeps the agent `speaking` until the reply has actually been heard.
* **Barge-in:** `InputSpeechStarted` while a response is generating or playing → pause playback (transport `pause_audio()`, or hold queued frames) and let the interruption policy (`session/interruptions.py`) decide. A real barge-in (`min_interruption_duration` of speech, `min_interruption_words` non-backchannel words, or the engine committing the user's turn / cancelling the response) → clear transport audio, drop queued frames, `interrupt(item_id, played_ms)`, trim the history message to what was heard and mark it `interrupted`. A false one (cough, noise, "uh-huh") → resume and emit `agent_false_interruption`. See `docs/concepts/interruptions.md`.
* **History order:** the user message is added when the engine commits the turn (`InputCommitted`) and its text is filled in when the final transcript arrives, so it always precedes the reply even when an engine transcribes after it starts answering (OpenAI Realtime). Until then it carries `metadata["transcript_pending"]`; `conversation_item` fires once, with the text.
* **Tools:** `ResponseToolCall` → tools run concurrently (`execute_function_call`, timeouts, errors become tool outputs) → outputs are sent back once the response is done; only the last output triggers the follow-up response; `max_tool_steps` bounds loops. Tool calls withdrawn by the engine (`ToolCallCancelled`) are cancelled.
* **Turn metrics:** one `TurnMetrics` per user turn spanning tool rounds: `voice_to_voice` (speech end → first agent audio handed to the transport), `end_of_turn_delay`, `response_ttfb`, `agent_speech_duration`, `interrupted`, `tool_calls`. Component metrics (`STTMetrics`, `LLMMetrics`, `TTSMetrics`, `VADMetrics`, `EOTMetrics`, `EngineMetrics`) are re-emitted and aggregated into `session.usage`.

## The cascade engine

* Audio streams continuously into the STT stream and the VAD (Silero/energy/...). VAD start → `InputSpeechStarted`; VAD end (a *candidate* pause, 0.25 s by default) → endpointing.
* **Endpointing:** flush (force-finalize) the STT, wait briefly for the final transcript, score the turn with the turn detector (audio and/or text), then commit after `min_endpointing_delay` (0.4 s with a detector, 0.6 s without) or `max_endpointing_delay` (2.5 s) when the user is probably not done — all measured from the end of speech. Speech resuming cancels the pending commit and the turn continues. STT-provided `END_OF_TURN` events commit immediately.
* **Response:** the LLM streams text; a `SentenceSegmenter` releases complete sentences (short first chunk for fast first audio), each is cleaned (`tts_clean`: markdown/emoji) and pushed into a TTS stream. With non-streaming TTS the `SentenceStreamAdapter` synthesizes sentence by sentence with one-ahead prefetch and reports per-sentence text, which gives exact text/audio alignment for truncation.
* **Half-cascade:** without STT, an audio-input LLM receives the user's audio as `AudioContent`.

## Providers and the registry

`create("stt", "deepgram/nova-3")` imports `voice_agent_next.providers.deepgram` (module name = provider name), which registered its classes with `@register_provider`. Provider modules must import without their optional dependencies (heavy packages are imported lazily with `require()`), so `van providers` can list everything and show what is missing. Third-party packages add providers through the `voice_agent_next.providers` entry-point group.

## Concurrency model

Everything is asyncio on one event loop. Blocking model inference runs in threads (`asyncio.to_thread`) when it takes more than a few milliseconds; audio device callbacks hand frames over with `loop.call_soon_threadsafe`. All timestamps use `perf_counter` (`utils.now()`), because `time.monotonic()` has 15.6 ms resolution on Windows. `Chan`, `cancel_and_wait` and `BackgroundTasks` keep task lifetimes explicit: everything a component starts is cancelled and awaited on close.

## Cross-platform rules

Pure-Python core (numpy, pydantic, httpx, websockets); native wheels only in extras; `pathlib`/`platformdirs`; no Unix-only signals; timing tolerant of coarse Windows timers; CI on Linux and Windows on every PR, macOS on `main` and on PRs labelled `ci:full`.

## Extension points

* new provider → `providers/<name>.py` + `@register_provider`
* new engine → subclass `S2SEngine`/`EngineConnection`, emit `events.*`
* new transport → subclass `Transport` and add it to `transports.create_transport`
* audio processing → `AudioProcessor` (capture + render reference)
* CLI command group → `cli/<name>.py` exposing a Typer `app`
