# AssemblyAI: Universal-3.5 Pro and Universal-Streaming STT

| Component | Spec | Protocol |
|---|---|---|
| Streaming STT with neural end of turn | `stt="assemblyai/universal-3-5-pro"` (default model) | `wss://streaming.assemblyai.com/v3/ws` (Streaming API v3) |
| Universal-Streaming | `stt="assemblyai/universal-streaming-english"` or `"assemblyai/universal-streaming-multilingual"` | same endpoint |
| Batch `transcribe()` | any of the above | `POST https://sync.assemblyai.com/v1/transcribe` (Sync STT, 80 ms – 120 s) |

Implementation: `src/voice_agent_next/providers/assemblyai.py`. It uses raw `websockets` and `httpx`, which are core dependencies, so there is no extra to install and no SDK.

## Setup

```bash
export ASSEMBLYAI_API_KEY=...      # or pass api_key="..." to the constructor
van providers --kind stt           # assemblyai shows "ready" once the key is set
```

* The key goes in the `Authorization` header, with no `Bearer` prefix. It never appears in URLs or logs.
* **Temporary tokens.** `await stt.create_temporary_token(expires_in_seconds=60, max_session_duration_seconds=600)` calls `GET /v3/token`. Hand the token to a client that must not see your key, which then uses `AssemblyAISTT(token=...)`. The token goes in the `token=` query parameter, and each token opens one session.
* **Data zones.** `region="us"` or `region="eu"` selects `wss://streaming.{us,eu}.assemblyai.com`, which keeps audio in that zone. The default is edge routing to the nearest region. `base_url=` overrides the origin completely.

## Recommended cascades

```python
from voice_agent_next import AgentSession

# AssemblyAI decides when the user is done (neural end of turn). Leave out vad=...
session = AgentSession(
    stt="assemblyai/universal-3-5-pro", llm="openai/gpt-4.1-mini", tts="cartesia"
)

# Your VAD / turn detector decides; AssemblyAI only transcribes (flush -> ForceEndpoint)
session = AgentSession(
    stt={"provider": "assemblyai", "end_of_turn": False},
    llm="openai/gpt-4.1-mini",
    tts="cartesia",
    vad="silero",
    turn_detector="smart_turn",
)
```

* **STT-owned turns (`end_of_turn=True`, the default).** `capabilities.end_of_turn` is `True`, so the cascade commits the user turn as soon as AssemblyAI ends it and skips its own endpointing delay.
  * Without a VAD, `SpeechStarted` becomes `InputSpeechStarted` and drives barge-in. AssemblyAI emits `SpeechStarted` only once the model has produced words, so background noise alone does not trigger it.
  * With a VAD, the VAD only drives barge-in and never commits a turn.
  * Tune the turn-taking with `mode`, `min_turn_silence` and `max_turn_silence`. See below.
* **Cascade-owned turns (`end_of_turn=False`).** The cascade's VAD and turn detector decide when a turn ends. Its `flush()` sends `ForceEndpoint`, and AssemblyAI answers at once with the open turn's formatted final.
  * AssemblyAI still ends turns by itself at long pauses. Those turns become plain `FINAL_TRANSCRIPT`s that the cascade adds to the user's turn. Raise `min_turn_silence` or `max_turn_silence` to reduce how often this happens.
  * AssemblyAI does not document an answer to `ForceEndpoint` when no turn is open. The stream then waits `force_endpoint_grace` seconds (0.3 by default). If no turn starts in that window, it acknowledges the flush with an empty final. A short utterance can be flushed before its first partial, because U3.5's `interruption_delay` is 500 ms in `balanced` mode. A turn that starts within the grace period answers the flush with its final. Set `force_endpoint_grace=0` to acknowledge at once.

## STT options (`AssemblyAISTT`)

| Option | Default | Models | Notes |
|---|---|---|---|
| `model` | `universal-3-5-pro` | — | `universal-streaming-english`, `universal-streaming-multilingual` |
| `api_key` / `token` | `$ASSEMBLYAI_API_KEY` / — | all | See Setup |
| `end_of_turn` | `True` | all | Emit `END_OF_TURN` and own the turn. See above |
| `language` / `language_codes` | — | U3.5 Pro | Sent as `language_codes=["en"]`. `en-US` becomes `en`. The Universal-Streaming models ignore `language` |
| `language_detection` | unset | U3.5 Pro, multilingual | Turns then carry `language_code`, which becomes `Transcript.language` |
| `sample_rate` | `16000` | all | 8–96 kHz `pcm_s16le`. Input is resampled to this rate |
| `mode` | unset (`balanced`) | U3.5 Pro | `min_latency`, `balanced` or `max_accuracy`. Sets the turn-detection defaults |
| `min_turn_silence` | from `mode` (128 / 128 / 512 ms); 400 ms on Universal-Streaming | all | Silence before an end-of-turn check |
| `max_turn_silence` | from `mode` (640 / 1280 / 2560 ms); 1280 ms on Universal-Streaming | all | Silence after which the turn ends regardless of content |
| `end_of_turn_confidence_threshold` | unset (0.4) | Universal-Streaming | `1.0` means acoustic-only endpointing and `0` means silence-only |
| `vad_threshold` | unset (0.2 on U3.5, 0.4 on Universal-Streaming) | all | Raise it for noisy audio or false barge-ins |
| `interruption_delay` | from `mode` (0 / 500 / 500 ms) | U3.5 Pro | Delay of a turn's first partial and `SpeechStarted` |
| `continuous_partials` | unset (on) | U3.5 Pro | Partials every ~3 s during long speech |
| `format_turns` | unset (off) | Universal-Streaming | Punctuated and cased finals. U3.5 Pro finals are always formatted |
| `keyterms` | `()` | all | Up to 100 terms of at most 50 characters each, sent as `keyterms_prompt` |
| `prompt` / `agent_context` | — | U3.5 Pro | Context about the audio, and the agent's last reply. At most 1750 characters each |
| `filter_profanity`, `voice_focus`, `domain`, `speaker_labels` | — | see AssemblyAI | Passed through |
| `inactivity_timeout` | unset | all | 5–3600 s. When set, `KeepAlive` is sent every `inactivity_timeout / 2` s while no audio flows |
| `force_endpoint_grace` | `0.3` s | all | See cascade-owned turns above |
| `chunk_ms` | `50` | all | Binary audio chunk size, 50–1000 ms. AssemblyAI recommends about 50 ms |
| `sync_url` / `sync_model` | Sync STT / `universal-3-5-pro` | batch | `sync_url=None` makes `transcribe()` stream instead |
| `close_timeout` | `5.0` s | all | Wait for the last `Turn` and `Termination` after `end_input()` |
| `extra_params` | — | all | Extra connection parameters, passed through verbatim |

Options for the wrong model family raise `ConfigurationError` when the STT is constructed, because AssemblyAI silently ignores unknown parameters. So do out-of-range values and keyterm lists that are too long. After `Begin`, the stream checks that `configuration.model` matches the requested model and logs a warning if it does not. `stream.session_id`, `stream.expires_at` and `stream.configuration` expose the session that `Begin` reported.

### Mid-session updates

`stt.stream()` returns an `AssemblyAIStream`. Its `await stream.update_configuration(...)` sends `UpdateConfiguration`, a delta that applies to audio processed afterwards. For example, while a caller dictates a phone number:

```python
await stream.update_configuration(min_turn_silence=1000, max_turn_silence=3000)
...
await stream.update_configuration(mode="balanced")  # U3.5 Pro: restore the preset
```

Updatable fields are `min_turn_silence`, `max_turn_silence`, `vad_threshold`, `keyterms_prompt`, `prompt`, `agent_context`, `mode`, `interruption_delay`, `continuous_partials`, `language_codes`, `end_of_turn_confidence_threshold` and `session_heartbeat`.

### Event mapping

| AssemblyAI message | `STTEventType` emitted |
|---|---|
| `SpeechStarted` (U3.5 Pro), or the first partial with words (Universal-Streaming, which has no `SpeechStarted`) | `START_OF_SPEECH`. `transcript.start_time` is the turn start in stream seconds |
| `Turn` with `end_of_turn: false` | `INTERIM_TRANSCRIPT`, only when the text changed. Each partial re-transcribes the whole turn and replaces the previous one. On Universal-Streaming, the word still being decoded (`word_is_final: false`) is included |
| The turn's final `Turn` (`end_of_turn: true`) | `FINAL_TRANSCRIPT` + `END_OF_SPEECH` + `END_OF_TURN`. The final carries word timings in stream seconds, with `end_time` at the end of the last word. There is no `END_OF_TURN` when `end_of_turn=False`, or when our own `flush()` (`ForceEndpoint`) ended the turn |
| Universal-Streaming with `format_turns=true` | Each final arrives twice. The unformatted copy becomes an `INTERIM_TRANSCRIPT`, and only the formatted copy becomes the final |
| `Termination` with a turn still open | Its last partial becomes the final |
| `Heartbeat`, `SpeakerRevision` | ignored (debug log) |

`end_input()` sends the audio tail, `ForceEndpoint` and then `Terminate`. The stream keeps reading until `Termination`, as AssemblyAI requires, because closing early drops the last transcript. AssemblyAI rejects audio chunks under 50 ms (error 3007). A tail shorter than that at a flush is held back and sent with the next chunk: it is the end of the silence the VAD waited for, and holding it keeps word timestamps on the input timeline. At `end_input()`, the tail is padded to 50 ms instead.

Metrics: the base class reports `STTMetrics` when a flush is answered. Turns that AssemblyAI ends by itself are never flushed, so their audio is reported at each final, with `latency=None`.

## Batch (`transcribe()`)

`transcribe()` sends one multipart request to the Sync STT API: `audio/pcm` at the STT's sample rate, with a `config` part that carries `timestamps`, `language_codes`, `keyterms_prompt` and `prompt`. It returns the text, word timings and confidence. There is no upload and no polling.

* Audio under 80 ms returns an empty transcript without a request.
* Audio over 120 s is streamed through the WebSocket instead. AssemblyAI throttles that at about 1.25× real time.

## Errors

AssemblyAI sends an `Error` message (`{"type": "Error", "error_code": ..., "error": ...}`) and then closes with the same code. `retryable` says whether opening a **new session** can succeed, so reconnect logic and the failover chain (#30) can use it.

| Situation | Exception | `retryable` |
|---|---|---|
| Missing API key and token | `ConfigurationError` (at construction) | — |
| Handshake HTTP 401 / 403, close `1008` | `AuthenticationError` | no |
| Close `3009`, or `1008` "Too many concurrent sessions", HTTP 429 | `RateLimitError` | yes |
| `3008` session expired (3 h, or the token's limit), `3006` inactivity timeout | `ProviderConnectionError` | yes |
| `1011`, an unexpected close, a network failure | `ProviderConnectionError` | yes |
| `3005` server error | `ProviderError` | yes |
| `3006` invalid message, `3007` chunk size or audio rate, `410` retired v2 endpoint | `ProviderError` | no |
| Handshake or HTTP timeout, Sync `504 inference_timeout` | `ProviderTimeoutError` | yes |
| Sync `400` / `413` / `415` | `ProviderError` | no |
| Sync `503 capacity_exceeded` / `service_unavailable`, `500` | `ProviderError` | yes |

## Latency notes (research note 03 §3.1)

* **U3.5 Pro Realtime.** Artificial Analysis measures 4.02 % WER at 0.19 s in `min_latency` mode. Pipecat measures 1.22 % semantic WER with 282 / 354 ms time to final segment. Pricing is $0.45 per hour of session time. Billing is per session, so always `end_input()` or `aclose()`.
* **STT-owned turns** commit as soon as the final arrives. Voice-to-voice is then roughly AssemblyAI's end-of-turn latency (`min_turn_silence` plus the model check), plus LLM time to first token, plus TTS time to first byte. `mode="min_latency"` drops `interruption_delay` to 0 for faster barge-in.
* **Forced finalization.** `ForceEndpoint` returns the final "right away", which is the "external endpointing" path for a cascade with a fast local turn detector.

## Testing

* **Unit tests** (`tests/test_assemblyai.py`) run against a local fake WebSocket server that replays AssemblyAI's documented `Begin`, `SpeechStarted`, `Turn`, `Termination` and `Error` messages. The fake enforces the 50 ms minimum chunk. Batch and token tests use `httpx.MockTransport`. Two cascade tests run `AgentSession` with the provider: one checks that the end of turn commits without the endpointing delay, the other that VAD-driven `ForceEndpoint` works.
* **Real-API tests** are skipped without a key:

  ```bash
  ASSEMBLYAI_API_KEY=... uv run pytest -m integration tests/test_assemblyai.py
  ```

  The speech round trip also needs `DEEPGRAM_API_KEY`, which synthesizes the test utterance. Without it, only the protocol and silence tests run.

## Protocol references (checked 2026-09-24)

* Streaming API reference: <https://www.assemblyai.com/docs/streaming/api-spec/streaming-websocket>
* Message sequence: <https://www.assemblyai.com/docs/streaming/message-sequence>
* Turn detection: <https://www.assemblyai.com/docs/streaming/turn-detection>
* Updating configuration mid-stream: <https://www.assemblyai.com/docs/streaming/updating-configuration-mid-stream>
* Mode presets and Universal-Streaming tuning: <https://www.assemblyai.com/docs/streaming/getting-started/optimizing-accuracy-and-latency>
* Prompting and keyterms: <https://www.assemblyai.com/docs/streaming/prompting-and-keyterms>
* Errors and close codes: <https://www.assemblyai.com/docs/streaming/common-session-errors-and-closures>
* Endpoints and data zones: <https://www.assemblyai.com/docs/streaming/endpoints-and-data-zones>
* Temporary tokens: <https://www.assemblyai.com/docs/streaming/api-spec/generate-streaming-token>
* Sync STT: <https://www.assemblyai.com/docs/sync-stt/getting-started/transcribe-a-short-audio-file>

## Limitations and follow-ups

* No automatic reconnection when a session drops. The stream fails with a mapped, `retryable` error. Engine rotation and reconnection are tracked in #17, and failover in #30.
* The answer to `ForceEndpoint` with no open turn is undocumented, hence `force_endpoint_grace`. Check it against the live API.
* The query encoding of the list parameters `language_codes` and `keyterms_prompt` is a JSON array, as the documented `keyterms_prompt` example shows. For `language_codes` this is inferred.
* The cascade does not yet push the agent's reply into `agent_context`, U3.5's context carryover. Call `update_configuration(agent_context=...)` yourself.
* Speaker labels (`SpeakerRevision`), PII redaction, `llm_gateway` and Opus/AAC input are not mapped. Pass them through with `extra_params` if you need them.
