# Observability: call recordings and tracing

Timing metrics and logs don't show what the caller actually heard. The recorded audio is
the ground truth, and traces explain it (research note 05 §7.1). `AgentSession` provides
both, and each is off unless you turn it on:

| | What you get | Turn on with |
|---|---|---|
| **Recording** | stereo WAV (user left, agent right) + JSONL event timeline on the same clock | `AgentSession(record="recordings/")` |
| **Tracing** | OpenTelemetry spans: `session → turn → end_of_turn / stt / chat / tts / response / execute_tool` | `AgentSession(trace=True)` + extra `otel` |

Both work with every engine (native speech-to-speech or cascade) and every transport. They
use the session's own events and audio timeline, so providers need no changes.

## Recording a call

```python
session = AgentSession("openai/gpt-realtime", record="recordings/")
await session.run(agent, transport)
print(session.recorder.wav_path, session.recorder.timeline_path)
```

`record=` accepts:

* a directory: each session writes `<YYYYmmdd-HHMMSS>-<id>.wav` and a `.jsonl` with the same name;
* a `.wav` path: the timeline goes next to it, e.g. `call.wav` + `call.jsonl`;
* a `SessionRecorder(path, audio_events=False, flush_delay=1.0)` for more options.

In config files, set `session: {record: recordings/}` (`SessionOptions.record`).

### The WAV

* **Left: the user.** This is the audio the transport delivered, before echo cancellation or other
  processors. It is placed at the time it arrived. Gaps in the input stay silent.
* **Right: the agent.** Each chunk is placed at the time it was scheduled to play, so the
  recording reproduces the call's timing:
  * while a possible barge-in pauses playback, the agent's channel pauses too, and the
    rest of the audio moves later when playback resumes;
  * when a barge-in is confirmed, the audio is cut where playback stopped. The audio that
    was generated but never played is not in the file;
  * audio still scheduled when the session closes is cut at the close.
* The recording uses the transport's output sample rate. The user channel is resampled
  to it, and the resampler's delay is compensated.
* `t = 0` is the moment the transport opened, both in the WAV and in the timeline.

The recording is **streamed to disk**. Audio older than `flush_delay` (1 s) is written
out, so memory holds only about a second of audio. The WAV header is updated on every
write, so the file stays playable during the call. Everything is flushed and closed when
the session closes, before `session.run()` / `wait_closed()` returns.

For latency analysis, open the WAV in Audacity or any editor with a waveform view. The
gap between the end of the left channel and the start of the right channel is the
voice-to-voice latency as the caller heard it.

### The timeline (JSONL)

One JSON object per line:

```json
{"t": 0.0, "source": "recording", "event": "recording_started", "data": {"wall_time": 1790271420.30, "wav": "20260924-203700-260ee764.wav", "sample_rate": 24000, "channels": {"left": "user", "right": "agent"}, "engine": {"provider": "openai", "model": "gpt-realtime"}, "transport": "WebSocketServerTransport", ...}}
{"t": 0.3819, "source": "engine", "event": "input_speech_started", "data": {"audio_time": 0.0}}
{"t": 1.2210, "source": "session", "event": "user_transcript", "data": {"text": "hello", "is_final": true, "item_id": "item_…", "language": "en"}}
{"t": 1.9034, "source": "session", "event": "metrics", "data": {"turn_id": "turn_…", "voice_to_voice": 0.61, "type": "turn", ...}}
{"t": 7.5120, "source": "recording", "event": "playback_cleared"}
{"t": 9.0000, "source": "recording", "event": "session_closed", "data": {"reason": "user_disconnected", "duration": 9.0}}
```

* `t` is the number of seconds since the recording started, on the same clock as the WAV.
  Session and engine events use the time they were created. Metrics use the time they
  were received.
* `source`:
  * `session`: every public session event (`agent_state_changed`,
    `user_state_changed`, `user_transcript`, `agent_transcript`, `conversation_item`,
    `tool_call`, `tool_result`, `interrupted`, `agent_false_interruption`,
    `agent_handoff`, `metrics`
    for every component and turn, `error`);
  * `engine`: every engine event (`input_speech_started`, `input_committed`,
    `response_started`, `response_done`...). Agent audio chunks (`response_audio`, with
    ids and durations only) are logged only with `SessionRecorder(..., audio_events=True)`;
  * `recording`: the start and end markers, and `playback_cleared` (where a barge-in cut
    the agent's audio).
* `recording_started.data.wall_time` (Unix time) maps `t` to wall-clock time.

Example: time to first agent audio per turn, with `jq`:

```bash
jq -c 'select(.event=="metrics" and .data.type=="turn") | {t, v2v: .data.voice_to_voice}' call.jsonl
```

## OpenTelemetry tracing

```bash
pip install 'voice-agent-next[otel]'           # opentelemetry-api only
pip install opentelemetry-sdk opentelemetry-exporter-otlp   # your choice of SDK/exporter
```

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

session = AgentSession("openai/gpt-realtime", trace=True)  # uses the global provider
# or: AgentSession(..., trace=SessionTracer(provider, capture_content=True))
```

`trace=True` without the `otel` extra logs a warning and traces nothing. Without a
configured SDK, the OpenTelemetry API is a no-op. A session without `trace=` creates no
tracer and never imports OpenTelemetry.

### Spans

```
session                        the whole call                           gen_ai.conversation.id
├── turn                       user speech start -> the reply finished   voice_agent.turn.*
│   ├── end_of_turn            user stopped -> turn committed (endpointing)
│   ├── turn_detection         semantic end-of-turn inference (cascade)
│   ├── stt                    recognition (cascade)
│   ├── chat {model}           LLM request (cascade): ttft, tokens
│   ├── tts                    synthesis (cascade): ttfb, characters
│   ├── response               one engine response: status, usage, ttfb
│   └── execute_tool {name}    a tool call
└── response                   responses outside a user turn (greeting, say())
```

Spans carry their real start and end times, reconstructed from the session's events and
metrics. For example, an STT result measured before the turn was committed still
appears under that turn.

| Span | Attributes |
|---|---|
| `session` | `gen_ai.conversation.id`, `gen_ai.provider.name`, `gen_ai.request.model`, `voice_agent.transport`, `voice_agent.session.turns`, `voice_agent.session.close_reason`, usage totals (`gen_ai.usage.input_tokens`/`output_tokens`, `voice_agent.usage.*`) |
| `turn` | `voice_agent.turn.{id, number, voice_to_voice, end_of_turn_delay, response_ttfb, agent_speech_duration, was_interrupted, tool_calls}`; events `interrupted` (`voice_agent.played`) and `false_interruption` |
| `chat {model}` | `gen_ai.operation.name=chat`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.id`, `gen_ai.usage.{input_tokens, output_tokens, cache_read.input_tokens, cache_creation.input_tokens}`, `voice_agent.llm.{ttft, tokens_per_second}` |
| `tts` / `stt` | `gen_ai.provider.name`, `gen_ai.request.model`, `voice_agent.tts.{ttfb, characters, audio_duration}` / `voice_agent.stt.{latency, audio_duration}` |
| `response` | `gen_ai.response.id`, `gen_ai.response.finish_reasons`, `voice_agent.response.{status, ttfb}`, `gen_ai.usage.*` (from the engine's metrics) |
| `execute_tool {name}` | `gen_ai.operation.name=execute_tool`, `gen_ai.tool.{name, call.id, type}`; with `capture_content`, `gen_ai.tool.call.{arguments, result}` |

Durations are in seconds. Failures set the span status to `ERROR` and add `error.type`.
Errors are recorded as exceptions on the `session` span.

The names follow the OpenTelemetry [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
where they apply. Those conventions are still in development, and the realtime-voice
proposal has not been merged yet. Voice-specific values use the `voice_agent.` prefix.

**Privacy.** Transcripts (`voice_agent.user.transcript`, `voice_agent.agent.transcript`)
and tool arguments/results are only recorded with `SessionTracer(capture_content=True)`.
Recordings and timelines always contain the conversation, so give them the same
retention and redaction policies as call recordings.

## Custom observers

Both features are built on `SessionTap` (`voice_agent_next.session.taps`). A tap
receives the raw audio timeline and every engine event:

* `user_audio(frame, t)`
* `agent_audio(frame, start)`
* `playback_paused` / `playback_shifted` / `playback_resumed` / `playback_cleared`
* `engine_event(ev)`
* `session_started` / `session_closing`

Attach your own tap with `session.add_tap(tap)` before the session starts. Hooks run
synchronously in the session's loops, so keep them cheap. Their exceptions are logged
and never reach the call.

## Limitations

* The agent channel follows the session's playback clock, which is when audio was due to
  play. For transports that report a playback position (a client-side jitter buffer, for
  example), the listener may hear it slightly later. For mouth-to-ear measurements, use a
  client-side recording (see the benchmark's `DuplexRecording`).
* User audio pushed faster than real time (`FileTransport(realtime=False)`,
  `LoopbackTransport.play_user_audio(realtime=False)`) is laid out back to back from
  its arrival. It is not placed where the engine processed it.
* Disk writes are small and synchronous, a few kB every half second. A very slow disk
  would add that time to the event loop.
