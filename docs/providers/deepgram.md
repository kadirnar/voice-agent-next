# Deepgram: Nova-3 and Flux STT, Aura-2 TTS

| Component | Spec | Protocol |
|---|---|---|
| Streaming STT | `stt="deepgram/nova-3"` (default model) | `wss://api.deepgram.com/v1/listen` |
| Conversational STT with end of turn | `stt="deepgram/flux-general-en"` or `"deepgram/flux-general-multi"` | `wss://api.deepgram.com/v2/listen` |
| TTS | `tts="deepgram/aura-2-thalia-en"` (default model) | `wss://api.deepgram.com/v1/speak` for streaming, `POST /v1/speak` for `synthesize()` |

Implementation: `src/voice_agent_next/providers/deepgram.py`. It uses raw `websockets` and `httpx`, which are core dependencies, so there is no extra to install and no SDK.

## Setup

```bash
export DEEPGRAM_API_KEY=...        # or pass api_key="..." to the constructor
van providers --kind stt           # deepgram shows "ready" once the key is set
```

Requests carry the key as `Authorization: Token <key>`. The key never appears in URLs or logs. For EU or self-hosted deployments, point `base_url=` at the API origin, for example `base_url="https://api.eu.deepgram.com"`. Both `http(s)` and `ws(s)` forms work.

## Recommended cascades

```python
from voice_agent_next import AgentSession

# Flux: the STT decides when the user is done. Leave out vad=...
session = AgentSession(
    stt="deepgram/flux-general-en", llm="openai/gpt-4.1-mini", tts="deepgram/aura-2-thalia-en"
)

# Nova-3 with a local VAD and turn detector (the cascade does the endpointing)
session = AgentSession(
    stt="deepgram/nova-3",
    llm="openai/gpt-4.1-mini",
    tts="deepgram/aura-2-thalia-en",
    vad="silero",
    turn_detector="smart_turn",
)
```

* **Flux without a VAD.** `StartOfTurn` becomes `InputSpeechStarted`, which drives barge-in. Deepgram recommends it over an external VAD because every `StartOfTurn` carries words. `EndOfTurn` commits the user turn at once and skips the cascade's endpointing delay. If you also pass a VAD, Flux's `EndOfTurn` still commits right away. The VAD's pauses then also trigger `flush()`, which is `ForceEndTurn`, so the cascade's own endpointing can end Flux turns early.
* **Nova-3 with or without a VAD.** Without a VAD, the cascade uses Deepgram's `SpeechStarted` and `speech_final`/`UtteranceEnd` as speech boundaries. It then flushes (`Finalize`) and commits after `min_endpointing_delay`.

## STT options (`DeepgramSTT`)

| Option | Default | Applies to | Notes |
|---|---|---|---|
| `model` | `nova-3` | both | A model starting with `flux` selects the `/v2/listen` protocol |
| `language` | `None` | both | Nova sends it as `language` (`en`, `en-US`, `multi`…). `flux-general-multi` sends it as `language_hint`. Pass several hints with `extra_params={"language_hint": [...]}` |
| `sample_rate` | `16000` | both | Input is resampled to this rate. Flux accepts 8000, 16000, 24000, 44100 or 48000 |
| `keyterms` | `()` | both | Keyterm prompting, one `keyterm=` parameter per term |
| `interim_results` | `True` | Nova | Needed for `utterance_end_ms` |
| `smart_format` / `punctuate` | `True` / unset | Nova | |
| `endpointing_ms` | `300` | Nova | `False` disables endpointing. `None` keeps Deepgram's default of 10 ms |
| `utterance_end_ms` | `1000` | Nova | Sent only with interim results |
| `vad_events` | `True` | Nova | Enables `SpeechStarted` |
| `keepalive_interval` | `5.0` s | Nova | Sends `KeepAlive` when no audio has been pushed for this long. Deepgram closes a stream after 10 s without audio (`NET-0001`). `None` disables it |
| `eot_threshold` | unset (0.7) | Flux | 0.5–1.0. `1.0` hands turn endings to `flush()` |
| `eager_eot_threshold` | unset | Flux | 0.3–0.9, at most `eot_threshold`. Enables `EagerEndOfTurn` / `TurnResumed` |
| `eot_timeout_ms` | unset (5000) | Flux | 500–60000. Silence after which the turn ends regardless of confidence |
| `chunk_ms` | 80 (Flux) / 50 (Nova) | both | Size of the binary audio messages. Deepgram strongly recommends 80 ms for Flux |
| `numerals`, `profanity_filter`, `mip_opt_out`, `tags`, `extra_params` | — | both | Passed through as query parameters |

Out-of-range Flux thresholds, and Flux sample rates it does not support, raise `ConfigurationError` when the STT is constructed.

### Event mapping

| Deepgram message | `STTEventType` emitted |
|---|---|
| Nova `SpeechStarted` (or the first words of an utterance) | `START_OF_SPEECH` |
| Nova `Results` with `is_final=false` | `INTERIM_TRANSCRIPT` |
| Nova `Results` with `is_final=true` | `FINAL_TRANSCRIPT`. Empty finals are dropped unless they answer a `Finalize` (`from_finalize=true`) |
| Nova `speech_final=true` / `UtteranceEnd` | `END_OF_SPEECH` (`transcript.end_time` = end of the last word, in stream seconds) |
| Flux `StartOfTurn` | `START_OF_SPEECH` + `INTERIM_TRANSCRIPT` |
| Flux `Update` | `INTERIM_TRANSCRIPT`, only when the transcript changed (Updates arrive every ~0.25 s) |
| Flux `EagerEndOfTurn` | `EAGER_END_OF_TURN` (the eventual `EndOfTurn` transcript matches it) |
| Flux `TurnResumed` | `TURN_RESUMED`, then `INTERIM_TRANSCRIPT` |
| Flux `EndOfTurn` (`trigger` `model` / `timeout`) | `FINAL_TRANSCRIPT` + `END_OF_SPEECH` + `END_OF_TURN` |
| Flux `EndOfTurn` (`trigger="manual"`, the answer to our `ForceEndTurn`) | `FINAL_TRANSCRIPT` + `END_OF_SPEECH`, with no `END_OF_TURN` because whoever flushed owns that decision |

`STTStream.flush()`, the cascade's force-finalize, maps to Nova `Finalize` and Flux `ForceEndTurn`. Deepgram does not always answer a `Finalize` when nothing is buffered. After Deepgram has itself ended the utterance (`speech_final`/`UtteranceEnd`) the flush is acknowledged at once with an empty final. Flux answers a `ForceEndTurn` outside a turn with a `FORCE_END_TURN_NO_ACTIVE_TURN` warning, which is also turned into an empty final. Callers waiting for a final transcript never stall.

`end_input()` flushes, then sends `CloseStream`. Flux's `CloseStream` does not finalize an active turn, so the last `Update` is emitted as the final transcript.

Metrics: the base class reports `STTMetrics` when a flush is answered. Flux turns ended by the model are never flushed, so their audio duration is reported at each `EndOfTurn` with `latency=None`.

## TTS (`DeepgramTTS`)

* **Streaming** (`capabilities.streaming=True`, the default):
  * Each `push_text()` delta is sent as `Speak`, and `flush()`/`end_input()` sends `Flush`.
  * A segment ends (`is_final` chunk) when Deepgram answers `Flushed`. In Deepgram's words, `Flush` lets it "generate the audio from its existing text buffer without waiting for additional text".
  * The cascade flushes once per response, which keeps well inside Deepgram's limit of 20 `Flush` per 60 s.
  * Audio is linear16 mono at `sample_rate` (24 kHz by default; 8/16/24/32/48 kHz allowed).
* **One WebSocket per conversation.** Deepgram asks for this.
  * After a stream finishes, its connection stays open for the next response.
  * A stream closed mid-synthesis, typically on barge-in, sends `Clear`. The connection is reused once Deepgram confirms with `Cleared`, so the next response pays no handshake and gets no stale audio.
  * Idle connections close after `idle_timeout` (60 s), and connections are replaced before Deepgram's 60-minute limit.
  * Connections closed by the server are detected and replaced.
  * `await tts.warmup()` opens the connection ahead of the first response.
* **`synthesize()`** uses `POST /v1/speak?container=none` and streams the raw PCM body. Empty text sends no request. REST requests are limited to 2000 characters.
* **`streaming=False`** synthesizes sentence by sentence over REST, through the `SentenceStreamAdapter`. This gives exact per-sentence text/audio alignment for truncation after barge-in, at the cost of one request per sentence. With native streaming the cascade estimates what was heard from the audio duration, because Aura returns no timestamps.
* **Voices.** `model="aura-2-<voice>-<lang>"`. English has 40+ voices; Aura-2 also covers `es`, `de`, `fr`, `nl`, `it` and `ja`. See [Deepgram's voice list](https://developers.deepgram.com/docs/tts-models). `voice=` (or the agent's voice) accepts a full model id or just a name: `voice="apollo"` with `aura-2-thalia-en` becomes `aura-2-apollo-en`. `speed=` accepts 0.7–1.5.

## Errors

| Situation | Exception |
|---|---|
| Missing API key | `ConfigurationError` (at construction) |
| Handshake / HTTP 401, 403 | `AuthenticationError` (message from `dg-error` / `err_msg`) |
| HTTP 429, Aura flush-limit warning | `RateLimitError` (retryable) |
| Other HTTP 4xx | `ProviderError` (5xx and 408 are retryable) |
| Close `1008 DATA-0000`, `1003`, `1009` | `ProviderError` (bad input, not retryable) |
| Close `1011 NET-000x`, unexpected close, network failure | `ProviderConnectionError` (retryable) |
| Handshake timeout / REST timeout | `ProviderTimeoutError` |
| Flux `Error` message | `ProviderError` (`INTERNAL_*` / timeouts retryable) |

## Latency notes

* **Flux**:
  * Deepgram quotes about 260 ms end-of-turn detection, and a median below 300 ms with p95 around 1.5 s (research note 03 §3.1).
  * `EndOfTurn` commits immediately, so voice-to-voice is roughly Flux's end-of-turn latency plus LLM time to first token plus TTS time to first byte.
  * `eager_eot_threshold` emits `EAGER_END_OF_TURN` earlier, for speculative generation (#27).
* **Nova-3**:
  * Artificial Analysis measures about 66 ms from the end of speech to the final transcript (research note 03 §3.1).
  * The cascade's endpointing (`min_endpointing_delay`) dominates the end-of-turn delay.
* **Aura-2**:
  * Reusing the connection removes the TLS/WebSocket handshake from each response's time to first byte. Call `warmup()` before the first one.
  * Time to first byte is reported per stream in `TTSMetrics.ttfb`.

## Testing

* **Unit tests** (`tests/test_deepgram.py`) run against local fake servers that replay Deepgram's documented messages. They need no network and no key.
* **Cascade test.** One test runs `AgentSession(stt=DeepgramSTT(model="flux-general-en", ...), llm="mock", tts="mock")` with a 3 s endpointing delay and checks that Flux's `EndOfTurn` commits the turn immediately.
* **Real-API tests** are skipped without a key:

  ```bash
  DEEPGRAM_API_KEY=... uv run pytest -m integration tests/test_deepgram.py
  ```

## Limitations and follow-ups

* No automatic reconnection when an STT stream drops mid-session; the stream fails with a `ProviderConnectionError`. Engine rotation and reconnection are tracked in #17 and failover in #30.
* The cascade ignores STT event timestamps, so with STT-driven endpointing it places the end of speech at the time the event arrives. `END_OF_SPEECH` events already carry `transcript.end_time`, and passing it through would make voice-to-voice metrics exact.
* Mid-stream `Configure` for Flux, for example to change end-of-turn thresholds while a caller dictates digits, is not exposed yet (#31).
* Flux TTS (`/v2/speak`) and Deepgram's Voice Agent API are not implemented.
