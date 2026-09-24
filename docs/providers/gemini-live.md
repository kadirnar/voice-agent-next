# Gemini Live (`google` engine)

Native speech-to-speech with Google's [Gemini Live API](https://ai.google.dev/api/live)
(`BidiGenerateContent` over WebSocket). Registered as `("engine", "google")` with the alias
`gemini`; implemented in `voice_agent_next/providers/google/live.py`.

```python
from voice_agent_next import Agent, AgentSession

session = AgentSession("google/gemini-3.8-live")  # or "gemini/gemini-3.8-live", "google"
await session.run(Agent("You are a friendly concierge."), transport)
```

```yaml
# agent.yaml
engine: {provider: google/gemini-3.8-live, voice: Kore, vad: {silence_duration_ms: 600}}
```

## Setup

* No extra needed: the engine speaks the raw WebSocket protocol with `websockets`, a core
  dependency (`pip install voice-agent-next`).
* API key: `api_key=...`, else `GOOGLE_API_KEY`, else `GEMINI_API_KEY`. The key is sent in the
  `x-goog-api-key` header (never in the URL, never logged). Ephemeral tokens
  (`auth_tokens/...`) use the `BidiGenerateContentConstrained` endpoint automatically.

## Models

| Model | Notes |
|---|---|
| `gemini-3.8-live` (default) | recommended; asynchronous (non-blocking) function calling by default |
| `gemini-3.8-live-extended-thinking` | background reasoning, `thinking_level="low" \| "medium" \| "high"`; non-blocking tools only |
| `gemini-3.1-flash-live-preview` | legacy; sequential tools only (`tool_mode="blocking"` is selected automatically) |
| `gemini-2.5-flash-native-audio-preview-12-2025` | restricted to prior users |

Pricing (3.8 Live, September 2026): audio input about $0.005/min, audio output about
$0.018/min; see the [pricing page](https://ai.google.dev/gemini-api/docs/pricing).

## Capabilities

| | |
|---|---|
| audio | 16 kHz PCM in, 24 kHz PCM out (the session resamples to/from the transport) |
| `server_turn_detection` | yes: Gemini's VAD (`automaticActivityDetection`); manual turns with `EngineOptions(turn_detection=False)` |
| `truncation` | **no**: the Live API cannot truncate what the model said; after a barge-in the session estimates the heard text |
| `tool_mode` | `"non_blocking"` (`"blocking"` for 3.1 Flash Live or `tool_behavior="blocking"`) |
| `max_session_duration` | 600 s: the approximate connection lifetime, rotated transparently (see below) |
| transcripts | input (partial + final) and output transcription, enabled by default |

## Options

| Argument | Default | Meaning |
|---|---|---|
| `model` | `gemini-3.8-live` | Live model id |
| `api_key` | env | see Setup |
| `voice` | server default | prebuilt voice (`Kore`, `Puck`, `Charon`, ...); `Agent(voice=...)` wins |
| `temperature` | `None` | sampling temperature (`EngineOptions.temperature` wins) |
| `vad` | `{}` | `automaticActivityDetection` fields, snake or camel case: `silence_duration_ms`, `prefix_padding_ms`, `start_of_speech_sensitivity`, `end_of_speech_sensitivity` |
| `activity_handling` | server default | `"start_of_activity_interrupts"` or `"no_interruption"` (then also use `SessionOptions(allow_interruptions=False)`) |
| `turn_coverage` | model default | `realtimeInputConfig.turnCoverage` enum value |
| `tool_behavior` | per model | `"non_blocking"` / `"blocking"` |
| `thinking_level` | `None` | extended-thinking models only |
| `input_transcription`, `output_transcription` | `True` | transcripts of the user's / model's audio |
| `session_resumption` | `True` | keep resumption handles; resume on rotation and reconnect |
| `context_window_compression` | `True` | sliding window with server defaults; pass a mapping (e.g. `{"trigger_tokens": 100000, "sliding_window": {"target_tokens": 50000}}`) or `False` |
| `rotate_after` | `540` | proactively move to a new connection at the first idle moment after this many seconds (`None` = only on `goAway`/errors) |
| `go_away_margin` | `2.0` | force the rotation this long before the `goAway` deadline |
| `resume_replay` | `0.5` | audio sent up to this long *before* the latest handle arrived is replayed into a resumed connection (covers audio in flight) |
| `max_buffered_audio` | `30` | cap (seconds) on audio buffered while switching connections |
| `connect_timeout` | `15` | handshake + setup timeout |
| `max_reconnect_attempts` | `5` | consecutive failed reconnects before the connection fails |
| `local_vad` | `True` | cheap energy VAD on the sent audio: speech-end estimates, idle detection and the start of manual turns (never automatic end of turn) |
| `base_url`, `api_version` | Google, `v1beta` | endpoint (proxies, tests) |
| `extra_setup` | `{}` | extra `setup` fields in API camel case, deep-merged last (`EngineOptions.extra` too) |

The language (`Agent(language="de-DE")`) is passed to the input transcriber as a hint; native
audio models choose the spoken language themselves (steer it in the instructions).

## How the protocol maps to engine events

| Live API (server → client) | Engine events |
|---|---|
| `voiceActivity` start / end (`audioOffset`) | `InputSpeechStarted` / `InputSpeechStopped` (stream positions) |
| `serverContent.interimInputTranscription`, `inputTranscription` | `InputTranscript(is_final=False)`, then a final one per turn |
| first model output after user speech | `InputCommitted` + final `InputTranscript` + `ResponseStarted` |
| `serverContent.modelTurn` inline audio | `ResponseAudio` (24 kHz) |
| `serverContent.outputTranscription` | `ResponseText` |
| `serverContent.interrupted` | server-side barge-in: `InputSpeechStarted` + `ResponseDone(status="cancelled")` |
| `generationComplete` / `turnComplete` | `ResponseDone` (usage from `usageMetadata` goes to `EngineMetrics`) |
| `toolCall` | `ResponseToolCall` (the response ends there, so the session runs the tool at once) |
| `toolCallCancellation {ids}` | `ToolCallCancelled` (the session cancels the running tool) |
| `goAway {timeLeft}`, reconnects | `EngineStatus("expiring" / "reconnecting" / "resumed" / "reconnected")` |

Client side: `send_audio` → `realtimeInput.audio`; `send_text` / `create_response` / `say` →
`clientContent` (`say` is instruction-based: Gemini has no verbatim TTS); `send_tool_output` →
`toolResponse`; `commit_input` → `activityEnd` (manual turns) or `audioStreamEnd` (automatic
VAD: finalize the turn now, e.g. from your own client-side VAD); `update()` → a new `setup`
on a resumed connection (see below).

**Manual turns** (`EngineOptions(turn_detection=False)`, e.g. push-to-talk): server-side
activity detection is disabled. When the local VAD hears the user start speaking, the engine
sends `activityStart` followed by up to 0.5 s of pre-roll audio, and `commit_input()` sends
`activityEnd`. Audio outside activities is not sent, so a microphone that keeps streaming
after `commit_input()` does not interrupt the answer. With `local_vad=False`, the activity
opens on the first audio frame.

Gemini commits user turns implicitly, so the engine recognizes a user turn when the model
starts answering without the client having asked for a response. If the user's transcript
arrives after the answer started (the API does not order transcripts), the answer's text
deltas are held back for up to one second so that the history stays in conversation order.

Without `voiceActivity` messages (older models), `InputSpeechStopped.audio_time` comes from
the local energy VAD, so voice-to-voice metrics stay comparable across engines.

## Tools

Function tools are declared with `parametersJsonSchema` (the JSON schema generated from your
Python signature) and, by default, `behavior: NON_BLOCKING`: the model keeps talking while the
tool runs. Results are returned with `scheduling: WHEN_IDLE` when the session wants a
follow-up answer and `SILENT` otherwise (e.g. while the user is talking); errors are sent as
`{"error": ...}`. When the user barges in, Gemini withdraws pending calls with
`toolCallCancellation`, which cancels the running tool task.

## Session management (rotation and resumption)

A Live API connection lasts about ten minutes; audio-only sessions last 15 minutes without
context window compression (enabled by default, so sessions are not limited). The engine keeps
the latest `sessionResumptionUpdate` handle (valid for about two hours) and moves the
conversation to a new connection:

* on `goAway` — at the next idle moment, or forced `go_away_margin` seconds before the deadline;
* proactively after `rotate_after` seconds, at an idle moment;
* after `update(instructions=..., tools=...)` (the API only accepts a new `setup` on a new
  connection);
* when the connection drops (reconnect with backoff).

"Idle" means: no generation, no pending tool call, the user is silent and the server has
issued a resumable handle. The switch is make-before-break: the new connection is set up
while user audio is buffered, then the audio sent since the latest handle (minus speech that
was already answered) is replayed, so no user audio is lost. A planned rotation that fails
keeps the current connection and retries later with backoff. If the handle is rejected
(expired) or resumption is disabled, the engine starts a fresh session, re-seeds it with the
user/assistant text history (`historyConfig.initialHistoryInClientContent`) and replays only
the audio after the last answered turn. Tool calls that the new session cannot know about are
withdrawn with `ToolCallCancelled`. If the connection keeps dying (more than
`max_reconnect_attempts` unexpected closes within a minute), the engine gives up with a
non-recoverable `EngineErrorEvent`.

## Latency notes

* End of turn is decided by the server VAD (about 800 ms of silence by default); 500–800 ms
  (`vad={"silence_duration_ms": 600}`) is a good range. Lower values split utterances.
* For faster endpointing, run a client-side VAD and call `commit_input()` at the end of
  speech (hybrid VAD); the server VAD stays as a fallback.
* `EngineMetrics.ttfb` runs to the first audio received. For voice turns it starts at the end
  of the user's speech, because Gemini commits turns implicitly; for client requests
  (`send_text`, tool results, ...) it starts at the request. The session's
  `TurnMetrics.voice_to_voice` runs from the end of speech to the first audio handed to the
  transport.

## Testing

Unit tests run against `voice_agent_next.testing.gemini_live.FakeGeminiLiveServer`, an
in-process fake of the WebSocket API (server VAD, transcripts, streamed audio, tool calls and
cancellation, barge-in, resumption handles, `goAway`, drops):

```python
from voice_agent_next.testing.gemini_live import FakeGeminiLiveServer, FakeToolCall

async with FakeGeminiLiveServer(
    replies=[FakeToolCall("get_weather", {"city": "Paris"}), "Sunny."]
) as server:
    engine = GeminiLiveEngine(api_key=server.api_key, base_url=server.url)
```

Real API: `GOOGLE_API_KEY=... uv run pytest -m integration tests/test_gemini_live.py`
(`GEMINI_LIVE_MODEL` selects another model).

## Limitations

* No truncation API: after a barge-in the model's context still contains its whole
  answer; the session history holds the estimated heard part.
* `cancel_response()` / `session.interrupt()` stop playback locally only (the API has no
  cancel message; the server stops by itself on user speech or new client content).
* `clear_input()` is a no-op: audio already sent cannot be withdrawn.
* Fresh-session re-seeding carries user/assistant text only (no tool calls/results), and a
  response requested just before the connection was lost is not re-requested.
* A late transcript correction arriving after the final transcript was emitted produces a
  second final `InputTranscript` for the same item (the history message is updated in place).
