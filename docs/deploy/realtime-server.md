# OpenAI-Realtime-compatible server

`voice_agent_next.server` puts **any** engine of this library behind the
[OpenAI Realtime](https://platform.openai.com/docs/guides/realtime) WebSocket protocol at
`/v1/realtime`. That includes a cascade of local models, Gemini Live, another provider, or
the scripted mock. Existing Realtime clients then run on it unchanged: the official `openai`
SDK, the OpenAI Agents SDK, LiveKit/Pipecat OpenAI-realtime plugins, browser code, and this
library's own `OpenAIRealtimeEngine`.

```
Realtime client ──ws /v1/realtime──▶ RealtimeServer ──▶ RealtimeSession ──▶ EngineConnection
 (openai SDK, Agents SDK,             auth, ?model=,      protocol state       (cascade, Gemini,
  OpenAIRealtimeEngine, ...)          session limit       machine, formats     mock, ...)
```

## Quick start

```bash
van serve --engine mock                                   # offline, no downloads
van serve --stt faster_whisper --llm ollama/qwen3 --tts kokoro --vad silero
van serve --engine agent.yaml --host 0.0.0.0 --port 8000 --api-key "$KEY"
van serve -e local=local.yaml -e cloud=openai/gpt-realtime   # two models, ?model= picks
```

Clients connect to `ws://HOST:PORT/v1/realtime?model=NAME`. From Python:

```python
import asyncio

from voice_agent_next.engines.cascade import CascadeEngine
from voice_agent_next.server import RealtimeServer


async def main() -> None:
    engine = CascadeEngine(stt="faster_whisper", llm="ollama/qwen3", tts="kokoro", vad="silero")
    server = RealtimeServer(engine, model="local", host="127.0.0.1", port=8000)
    await server.serve_forever()  # Ctrl-C: clients are closed with 1001, engines closed


asyncio.run(main())
```

### `van serve` options

| Option | Meaning |
| --- | --- |
| `--protocol/-p` | Wire protocol: `openai-realtime` (default, this page), or `websocket`, `webrtc`, `twilio`, `telnyx`, `vonage`, `plivo` (see [Serving in production](serving.md)). |
| `--engine/-e [NAME=]SOURCE` | A registry spec (`mock`, `openai/gpt-realtime`), an inline mapping (`'{provider: mock, response_delay: 0.2}'`) or an agent config file (YAML/TOML/JSON, engine or cascade). Repeat it to serve several models. |
| `--stt --llm --tts --vad --turn` | Build a cascade (named `--name`, default `cascade`). |
| `--name/-n` | Model name of a single engine. |
| `--instructions --voice --language` | Session defaults. A client's `session.update` overrides them. |
| `--api-key` (env `VAN_SERVER_API_KEY`) | Require this bearer token. Repeat it to accept several. Without it there is no authentication. |
| `--max-sessions` | Refuse clients beyond this many live sessions (HTTP 503, `Retry-After: 1`). |
| `--any-model/--strict-model` | Serve the default model for an unknown `?model=` (for clients with a hard-coded `gpt-realtime`). Default: only when a single model is served. |
| `--max-session-duration` | Close sessions after N seconds with a `session_expired` error, like OpenAI. |
| `--warmup/--no-warmup` | Call `engine.warmup()` before accepting clients (default on). |
| `--host --port` | Listening address. `--port 0` picks a free port. |
| `--preset --config` | Serve a preset's or a config file's engine (like `-e agent.yaml`). |
| `--prewarm N`, `--engine-per-session` | Keep N engine connections open ahead of calls; one engine per session. See [prewarm](serving.md#prewarm-no-model-load-or-connection-setup-in-the-call-path). |
| `--workers N`, `--drain-timeout S`, `--log-format json` | Worker processes on one port, graceful drain, structured logs. See [Serving in production](serving.md). |

With a config file, the agent's `instructions`, `voice` and `language` become session
defaults. Agent `tools` in the config are **not** served: Realtime clients declare and run
their own tools.

## Using it from clients

**This library** (e.g. an `AgentSession` with tools, barge-in and metrics):

```python
from voice_agent_next.providers.openai.realtime import OpenAIRealtimeEngine

engine = OpenAIRealtimeEngine(base_url="ws://127.0.0.1:8000/v1", api_key=KEY, model="local")
```

**The official `openai` SDK**:

```python
client = AsyncOpenAI(api_key=KEY, websocket_base_url="ws://127.0.0.1:8000/v1")
async with client.realtime.connect(model="local") as conn:
    await conn.session.update(session={"type": "realtime", "instructions": "Be kind."})
    ...
```

**Browsers** cannot set headers on a WebSocket. Pass the key as the subprotocol
`openai-insecure-api-key.<key>`, as with OpenAI's API. Only do that with a short-lived key
over TLS.

## Protocol mapping

The server speaks the GA protocol by default. Clients that send `OpenAI-Beta: realtime=v1`
(or the `openai-beta.realtime-v1` subprotocol, or a beta-shaped `session.update`) get the
beta dialect: `response.audio.delta` instead of `response.output_audio.delta`,
`conversation.item.created` instead of `conversation.item.added`, beta `session` fields, and
no `conversation.item.done`.

### Client events

| Client event | What the server does |
| --- | --- |
| `session.update` | Validates and applies `instructions`, `tools` (function tools only; MCP is rejected), `tool_choice`, `output_modalities` (`audio` or `text`), `audio.input.format` / `audio.output.format` (`audio/pcm` at 8–48 kHz, `audio/pcmu`, `audio/pcma`), `voice`, `speed`, `turn_detection` (`server_vad`, `semantic_vad` or `null`), `input_audio_transcription` / `transcription`, `max_output_tokens`. Answers with `session.updated`. Instructions and tools reach the running engine connection. Changing voice, language or VAD on/off reopens the engine connection (with the conversation) between responses. |
| `input_audio_buffer.append` | Decodes G.711 or PCM16, resamples to the engine's input rate and streams it. |
| `input_audio_buffer.commit` / `.clear` | Manual turns (with `turn_detection: null`): commit needs at least 100 ms of audio. Answers `input_audio_buffer.committed` / `.cleared`. |
| `conversation.item.create` | User/assistant/system text messages, `function_call` and `function_call_output` items, honouring `previous_item_id`. Answers `conversation.item.added` + `.done`. |
| `conversation.item.truncate` | Barge-in: cuts the assistant message to `audio_end_ms` in the engine (`EngineConnection.truncate`). Answers `conversation.item.truncated`. |
| `conversation.item.delete` / `.retrieve` | Delete rebuilds the engine context lazily. Retrieve returns the item. |
| `response.create` | Starts a response, with optional per-response `instructions` and `output_modalities`. |
| `response.cancel` | Cancels the active response. Queued audio that was not yet sent is dropped. |
| `output_audio_buffer.clear` | Rejected with `unsupported_event`: it is a WebRTC/SIP event, and WebSocket clients play the audio themselves. |

### Server events

| Engine event | Server events |
| --- | --- |
| connection opened | `session.created` (with `expires_at` when a duration limit is set) |
| `InputSpeechStarted` / `InputSpeechStopped` | `input_audio_buffer.speech_started` / `.speech_stopped` (with `audio_start_ms` / `audio_end_ms`). Only with turn detection. |
| `InputCommitted` | `input_audio_buffer.committed`, then `conversation.item.added` / `.done` for the user message |
| `InputTranscript` | `conversation.item.input_audio_transcription.delta` / `.completed` (when transcription is enabled) |
| `ResponseStarted` | `response.created` |
| `ResponseAudio` | `response.output_item.added`, `response.content_part.added`, `response.output_audio.delta` (at most 100 ms per delta) |
| `ResponseText` | `response.output_audio_transcript.delta` (audio responses) or `response.output_text.delta` (text-only) |
| `ResponseToolCall` | `response.output_item.added` (`function_call`), `response.function_call_arguments.delta` / `.done`, `response.output_item.done` |
| `ResponseDone` | the `.done` events of every open part/item, then `response.done` with `status`, `status_details` and `usage` (mapped from `EngineUsage`) |
| `EngineErrorEvent` / engine failure | `error` (`server_error`, `engine_error` / `engine_unavailable`) |
| invalid client event | `error` with `invalid_request_error`, a `code` and a `param` as OpenAI sends them (`invalid_value`, `missing_required_parameter`, `item_not_found`, `conversation_already_has_active_response`, ...) and the client's `event_id` |

### Who decides when to answer

Engines answer a committed user turn by themselves. The Realtime protocol lets the client
decide (`create_response: false`, or manual commits). The server bridges the two. It holds the
engine's answer, and the next plain `response.create` adopts it: no second generation, no added
latency. If the client changes the conversation first, the held answer is discarded.

`interrupt_response` (default `true`): when the user starts speaking during a response, the
server cancels it (`status_details.reason = "turn_detected"`). The client plays the audio, so
only the client knows how much the user heard. It reports that with
`conversation.item.truncate`, exactly as with OpenAI's API over WebSocket.

## Production notes

* **Authentication.** `Authorization: Bearer <key>`, an `api-key` header, or the browser
  subprotocol. Keys are compared in constant time and never logged. A bad key gets HTTP 401.
* **TLS and origins.** Pass `serve_options={"ssl": ctx, "origins": [...]}` to
  `RealtimeServer`, or terminate TLS at a reverse proxy (enable WebSocket upgrades on it).
* **Health.** `GET /health` returns `{"status": "ok", ...}`. `GET /v1/models` lists the
  served models. Under `van serve`, `GET /ready` and `GET /metrics` (Prometheus) are
  served too: see [Serving in production](serving.md#health-readiness-and-metrics).
* **Concurrency.** Each client session gets its own `engine.connect()`. A shared engine loads
  its models once. A factory (`RealtimeModel(lambda: make_engine())`) builds and closes one
  engine per session.
* **Backpressure.** A reader stops reading from the socket when too many client messages
  are queued (TCP backpressure). When too many server events wait to be sent to a slow client,
  the session stops taking engine output, so the engine's own buffers hold it. Client events
  (cancel, barge-in) are still handled. A client that stops reading entirely is closed (1008)
  once `max_send_buffer` is exceeded.
* **Shutdown.** `server.aclose()` closes every session with 1001 and closes the engines
  the server created. Under `van serve`, SIGTERM or Ctrl-C first drains: new sessions are
  refused and live ones get `--drain-timeout` seconds to finish
  ([graceful drain](serving.md#graceful-drain)).

## Limitations

* Audio input inside `conversation.item.create` (`input_audio` content) is rejected.
  Stream audio through `input_audio_buffer.append` instead.
* Out-of-band responses (`response.create` with `conversation: "none"`) are rejected.
* WebRTC and SIP Realtime endpoints are not provided. Only WebSocket is.
* Voice activity detection runs in the engine. `threshold`, `silence_duration_ms` and
  `eagerness` are accepted and echoed back, but each engine applies its own VAD settings.
* `usage` counts are only as good as the engine's `EngineUsage` (a cascade reports LLM
  tokens; audio token counts may be zero).
