# OpenAI Realtime engine (and compatible backends)

Native speech-to-speech over the OpenAI Realtime **WebSocket GA protocol**, plus one client
for every OpenAI-Realtime-compatible server through **compatibility profiles**.

| Spec | Backend | Module |
| --- | --- | --- |
| `openai/gpt-realtime-2.1` | OpenAI Realtime API | `providers/openai/realtime.py` |
| `azure_openai/<deployment>` | Azure OpenAI Realtime (GA `/openai/v1`) | `providers/azure_openai.py` |
| `xai/grok-voice-latest` | xAI Grok Voice Agent API | `providers/xai.py` |
| `qwen_omni/qwen3.8-omni-flash-realtime` | Alibaba Qwen-Omni-Realtime (Model Studio / DashScope) | `providers/qwen_omni.py` |
| `vllm_realtime/<served-model>` | vLLM-Omni `/v1/realtime` (local, experimental) | `providers/vllm_realtime.py` |
| `speaches/<llm>` | Speaches Realtime API (local VAD → STT → LLM → TTS) | `providers/speaches.py` |
| `localai/<pipeline>` | LocalAI Realtime API (local pipeline) | `providers/localai.py` |

No extra dependency: the engine uses the core `websockets` package (no `openai` SDK).

## Quick start

```python
from voice_agent_next import Agent, AgentSession
from voice_agent_next.transports import create_transport

session = AgentSession("openai/gpt-realtime-2.1")  # OPENAI_API_KEY
await session.run(Agent("You are a helpful assistant.", greeting="Hi!"), create_transport("local"))
```

With options (instance or config mapping):

```python
from voice_agent_next.providers.openai.realtime import OpenAIRealtimeEngine

engine = OpenAIRealtimeEngine(
    model="gpt-realtime-2.1",
    voice="marin",
    turn_detection={"type": "server_vad", "silence_duration_ms": 400},
    input_transcription="gpt-4o-mini-transcribe",
    reasoning_effort="low",
)
session = AgentSession(engine)
```

```yaml
# agent.yaml
engine: {provider: xai/grok-voice-latest, voice: eve}
agent: {instructions: You are a friendly assistant., greeting: Hello!}
```

## Setup per backend

| Profile | Credentials / endpoint |
| --- | --- |
| `openai` | `OPENAI_API_KEY` (or `api_key=`); `base_url=` for a proxy |
| `azure_openai` | `AZURE_OPENAI_ENDPOINT` (`https://<resource>.openai.azure.com`), `AZURE_OPENAI_API_KEY` (sent as `api-key`) **or** an Entra ID token (`azure_ad_token=` / `AZURE_OPENAI_AD_TOKEN`, sent as `Authorization: Bearer`); the model is the *deployment name* (`AZURE_OPENAI_DEPLOYMENT_NAME`, default `gpt-realtime-2.1`) |
| `xai` | `XAI_API_KEY` |
| `qwen_omni` | `DASHSCOPE_API_KEY`, `DASHSCOPE_WORKSPACE_ID`, optional `DASHSCOPE_REGION` (`ap-southeast-1` default, or `cn-beijing`); or `base_url=wss://{WorkspaceId}.{region}.maas.aliyuncs.com/api-ws/v1` |
| `vllm_realtime` | `VLLM_BASE_URL` (default `ws://localhost:8000/v1`), optional `VLLM_API_KEY` |
| `speaches` | `SPEACHES_BASE_URL` (default `ws://localhost:8000/v1`), optional `SPEACHES_API_KEY` |
| `localai` | `LOCALAI_BASE_URL` (default `ws://localhost:8080/v1`), optional `LOCALAI_API_KEY`; the model is a LocalAI *pipeline* (default `gpt-realtime`) |

`base_url` accepts `ws(s)://` or `http(s)://` and gets `/realtime` appended; the model goes in
the `model` query parameter (omitted when empty, e.g. a single-model vLLM server).

## Models

* OpenAI: `gpt-realtime-2.1` (default), `gpt-realtime-2.1-mini`, `gpt-realtime-2`,
  `gpt-realtime-1.5`, `gpt-realtime`, `gpt-realtime-mini`. Reasoning models (2.x) accept
  `reasoning_effort` (`minimal` … `xhigh`).
* xAI: `grok-voice-latest` (alias of `grok-voice-think-fast-2.0`); `reasoning_effort` is `high` or `none`.
* Qwen: `qwen3.8-omni-flash-realtime` (default), `qwen3.5-omni-plus-realtime`,
  `qwen3.5-omni-flash-realtime`, `qwen3-omni-flash-realtime`.

## Options

`OpenAIRealtimeEngine(...)` (every profile module accepts the same keyword arguments):

| Option | Default | Notes |
| --- | --- | --- |
| `model` | profile default | Azure: deployment name |
| `api_key`, `base_url`, `headers`, `query` | env / profile | `query={"duplex": "1"}` for vLLM-Omni MiniCPM duplex |
| `voice` | `marin` (OpenAI/Azure), `eve` (xAI), server default otherwise | `Agent(voice=...)` wins; the voice cannot change after the first audio |
| `turn_detection` | `semantic_vad` (OpenAI/Azure), `server_vad` otherwise | `"server_vad"`, `"semantic_vad"`, a raw object, or `None` for manual turns (`commit_input()`); `create_response`/`interrupt_response` default to `true` |
| `input_transcription` | `gpt-4o-mini-transcribe` (OpenAI), `whisper-1` (Azure), `grok-transcribe` (xAI), `qwen3-asr-flash-realtime` (Qwen) | model name, config object, or `None` to disable user transcripts; `Agent(language=...)` becomes the transcription language |
| `noise_reduction` | – | `near_field` / `far_field` (GA) |
| `reasoning_effort`, `speed`, `max_output_tokens`, `temperature` | – | sent only where the profile supports them (GA removed `temperature`) |
| `session` | – | extra session fields, deep-merged into every full `session.update` (also `EngineOptions.extra`) |
| `input_sample_rate`, `output_sample_rate` | profile (24 kHz; Qwen input 16 kHz) | PCM16 mono; the connection resamples input |
| `connect_timeout` | 10 s | handshake + `session.updated` |
| `max_reconnect_attempts`, `reconnect_backoff` | 3, 0.5 s | see *Reconnects* |
| `expiry_warning` | 60 s | `EngineStatus("expiring")` before the session limit |
| `refine_speech_end` | `True` | see *Speech end* |

## Compatibility profiles

A `RealtimeProfile` (see `PROFILES` in `providers/openai/realtime.py`) captures everything that
differs; pass `profile=RealtimeProfile(...)` or `dataclasses.replace(PROFILES["openai"], ...)`
for another server.

| | openai | azure_openai | xai | qwen_omni | vllm_realtime | speaches | localai |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Auth | Bearer | `api-key` / Bearer | Bearer | Bearer | optional | optional | optional |
| `session.update` dialect | GA | GA | xAI¹ | beta² | GA | beta² | GA |
| Server event names | GA | GA | GA | beta³ | GA / beta³ | beta³ | GA |
| Input / output PCM | 24k / 24k | 24k / 24k | 24k / 24k | **16k** / 24k | 24k / 24k | 24k / 24k | 24k / 24k |
| `semantic_vad` | ✓ | ✓ | – | ✓ (3.5+) | server-dependent | – | ✓ |
| `response.cancel` | ✓ | ✓ | ✓ | ✓ (no `response_id`) | ✓ | – | ✓ |
| `conversation.item.truncate` (`capabilities.truncation`) | ✓ | ✓ | ✓ | – | ✓ | – | ✓ |
| User text items (`send_text`) | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ |
| Per-response instructions | ✓ | ✓ | ✓ | – (session patched⁴) | ✓ | ✓ | ✓ |
| `say()` | instructions + `input: []` | instructions + `input: []` | `force_message` (verbatim TTS) | patched instructions⁴ | instructions | instructions | instructions |
| Partial user transcripts | deltas | deltas | cumulative `…transcription.updated` | cumulative `text` + `stash` | deltas | deltas | deltas |
| Tool definitions | flat | flat | flat | Chat-style (`function: {...}`) | flat | flat | flat |
| Session limit (`max_session_duration`) | 60 min | 60 min | – | 120 min | – | – | – |

¹ GA audio formats (`audio.input/output.format`) with top-level `voice` and `turn_detection`
(`server_vad` or `null`, no `create_response`/`interrupt_response`).
² Flat `modalities`, `voice`, `input_audio_format`/`output_audio_format` (`pcm` for Qwen,
`pcm16` for Speaches), `turn_detection`, `input_audio_transcription`.
³ Beta names are accepted by every profile: `response.audio.delta`, `response.audio_transcript.delta`,
`response.text.delta`, `conversation.item.created`, … map to their GA equivalents.
⁴ `session.update` with the one-off instructions, `response.create`, and the original
instructions restored as soon as `response.created` arrives.

Upstream vLLM's own `/v1/realtime` is a *speech-to-text* endpoint (`transcription.delta`
events) and is not supported by this engine; the `vllm_realtime` profile targets vLLM-Omni's
OpenAI-Realtime-shaped endpoint (MiniCPM-o duplex today: `query={"duplex": "1"},
turn_detection=None`; Qwen3-Omni once [vllm-omni#6592](https://github.com/vllm-project/vllm-omni/issues/6592) lands).

## Event mapping

| Server event | Engine event |
| --- | --- |
| `input_audio_buffer.speech_started` | `InputSpeechStarted(audio_time=audio_start_ms)` — the session barges in |
| `input_audio_buffer.speech_stopped` | `InputSpeechStopped(audio_time=<speech end>)` |
| `input_audio_buffer.committed` | `InputCommitted(item_id)` |
| `conversation.item.input_audio_transcription.delta` / `.updated` | `InputTranscript(is_final=False)` accumulated per `item_id` |
| `conversation.item.input_audio_transcription.completed` | `InputTranscript(is_final=True)` (matched by `item_id`, may arrive after the response started) |
| `conversation.item.input_audio_transcription.failed` | recoverable `EngineErrorEvent` |
| `response.created` | `ResponseStarted` |
| `response.output_audio.delta` | `ResponseAudio` (PCM16 at `output_sample_rate`) |
| `response.output_audio_transcript.delta`, `response.output_text.delta` | `ResponseText` |
| `response.output_item.done` / `response.function_call_arguments.done` (function call) | `ResponseToolCall` (once per `call_id`; calls only listed in `response.done` are emitted too) |
| `response.done` | `ResponseDone(status, usage)` + `EngineMetrics` |
| `error` | `EngineErrorEvent` (auth errors are fatal; `response_cancel_not_active` is ignored) |

Control methods:

* `send_audio` → `input_audio_buffer.append`; `clear_input` → `input_audio_buffer.clear`;
* `commit_input` → `input_audio_buffer.commit`, then `response.create` (manual turns);
* `send_text` / `send_tool_output` → `conversation.item.create`, then `response.create` when
  `respond=True`;
* `create_response(instructions=...)` → `response.create` with the session instructions plus
  the extra ones; `say(text)` → see the profile table;
* `update` → partial `session.update`;
* `interrupt(item_id, played_ms)` → `response.cancel`, then `conversation.item.truncate`.

## Barge-in and truncation

Over WebSocket the client plays the audio, so the server cannot know what was heard. On
`speech_started` the session stops playback and calls `interrupt(item_id, played_ms)`: the engine
cancels the response (when still generating) and truncates the assistant item with the tracked
`content_index` at the **played** milliseconds — clamped to the audio received, so the server
never rejects it, and skipped for items without audio. When a response has several audio items
(GPT-Realtime-2 preambles), the cut lands in the right item and later items are truncated to 0.
The server removes the unheard transcript; the session trims its history to its own estimate.

With the default `interrupt_response: true` the server also cancels the response itself as soon
as its VAD hears the user. If you disable interruptions in the session
(`SessionOptions(allow_interruptions=False)`), also pass
`turn_detection={"type": "semantic_vad", "interrupt_response": False}`.

## Speech end and latency metrics

`InputSpeechStopped.audio_time` must be where the user **stopped talking**. OpenAI documents
`speech_stopped.audio_end_ms` as the end of the committed audio, i.e. including the
`silence_duration_ms` hold (or `semantic_vad`'s variable wait), while xAI documents it as the
speech end. The connection keeps a short history of the levels of the audio it sent and reports
the last voiced position before `audio_end_ms` (within the maximum hold), falling back to
`audio_end_ms` when the levels are inconclusive (e.g. loud background noise). Pass
`refine_speech_end=False` to report `audio_end_ms` unchanged. The session maps this position to
the capture time, so `TurnMetrics.voice_to_voice` is comparable with other engines.

`EngineMetrics` per response: `ttfb` = response trigger (turn commit, or the `create_response()`
/ `say()` call, including any wait for the previous response to be cancelled) → first audio delta, `duration`, token usage from `response.done.usage`
(`input_token_details`/`output_token_details`; Qwen's `*_tokens_details`), `cancelled`.

## Sessions, keepalive and reconnects

* WebSocket pings every 20 s keep the connection alive; `aclose()` closes it cleanly (1000).
* `session.created.expires_at` (or the profile's `max_session_duration`) schedules
  `EngineStatus("expiring", time_left=...)` `expiry_warning` seconds before the limit.
* After a transient failure (network drop, server close, `session_expired`) the connection emits
  `EngineStatus("reconnecting")`, fails the response in flight with
  `ResponseDone(status="failed")`, reconnects with exponential backoff, re-sends the session
  configuration and emits `EngineStatus("reconnected")`. The new provider session starts with an
  **empty conversation** (context re-seeding is issue #17); the server's audio clock restarts and
  is re-based onto the input stream. Authentication failures are fatal; a server that keeps
  dropping the connection (more than `max_reconnect_attempts` reconnects per minute) ends the
  connection with a non-recoverable `EngineErrorEvent`.

## Latency notes

* Audio is sent as it arrives (20 ms frames are fine); the engine never buffers user audio.
* Semantic VAD waits longer on trailing-off speech (up to 8 s at `eagerness: low`); use
  `server_vad` with a short `silence_duration_ms` for the snappiest turns.
* Input transcription runs asynchronously; final user transcripts can arrive after the agent
  started answering.

## Testing

* Unit tests run against `tests/fake_realtime_server.py`, a scripted Realtime server
  (`websockets.serve` on `127.0.0.1:0`) that replays GA, beta and xAI event sequences with an
  energy-based server VAD, tool calls, cancellation and truncation validated like the real API.
* `uv run pytest -m integration tests/test_openai_realtime.py` talks to the real API when
  `OPENAI_API_KEY` is set (`OPENAI_REALTIME_MODEL` selects the model).

## Limitations

* WebSocket only (WebRTC/SIP transports, where the server truncates by itself, are not wired).
* No context carry-over across reconnects or session rotation yet (#17).
* xAI reports usage totals only, so `EngineUsage` token details stay 0 for xAI.
* Qwen-Omni accepts no user text items: `send_text()` raises `EngineError` and an initial
  `chat_ctx` cannot be seeded. Without per-response instructions, `say()` and
  `create_response(instructions=...)` patch the session prompt for one response: if the server
  VAD starts a response for the user at that very moment, that response gets the one-off
  instructions (a rejected request rolls the patch back immediately).
* The Azure Voice Live, Kyutai Unmute (Opus audio) and LiteLLM proxy dialects are not profiled
  yet; `base_url=` + a custom `RealtimeProfile` covers servers that follow the GA or beta protocol.
