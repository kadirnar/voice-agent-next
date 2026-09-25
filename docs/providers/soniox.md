# Soniox: real-time STT (`stt-rt-v5`)

| Component | Spec | Protocol |
|---|---|---|
| Streaming STT with semantic endpoint detection | `stt="soniox/stt-rt-v5"` (default model) | `wss://stt-rt.soniox.com/transcribe-websocket` |
| Batch `transcribe()` | same | the same WebSocket, audio sent in one go |

Implementation: `src/voice_agent_next/providers/soniox.py`. It uses raw `websockets` and `httpx`, which are core dependencies, so there is no extra to install and no SDK.

Soniox v5 is the cheapest fast cloud STT in the research survey: about $0.12 per hour ($0.002/min), a final transcript 0.054 s after speech ends on Artificial Analysis, 60+ languages in one model, and a `finalize` message for external endpointing (research note 03 §3.1). `stt-rt-v4` is an alias of v5 since 2026-06-30.

## Setup

```bash
export SONIOX_API_KEY=...          # or pass api_key="..." to the constructor
van providers --kind stt           # soniox shows "ready" once the key is set
```

* Soniox takes the key in the first WebSocket message (the config), not in a header or the URL. It never appears in logs.
* **Temporary keys.** `await stt.create_temporary_api_key(expires_in_seconds=60, single_use=True)` calls `POST /v1/auth/temporary-api-key` with `usage_type="transcribe_websocket"`. Hand the key to a client that must not see yours, which then uses `SonioxSTT(api_key=temp_key)`.
* **Data residency.** `region="eu"`, `"jp"` or `"in"` selects `wss://stt-rt.{eu,jp,in}.soniox.com` and the matching REST host. Keys are per region, and regional projects are enabled by Soniox support. `base_url=` / `api_url=` override the origins.

## Recommended cascades

```python
from voice_agent_next import AgentSession

# Soniox decides when the user is done (semantic endpoint detection). Leave out vad=...
session = AgentSession(stt="soniox/stt-rt-v5", llm="openai/gpt-4.1-mini", tts="cartesia")

# Your VAD / turn detector decides; Soniox only transcribes (flush -> finalize)
session = AgentSession(
    stt={"provider": "soniox", "end_of_turn": False},
    llm="openai/gpt-4.1-mini",
    tts="cartesia",
    vad="silero",
    turn_detector="smart_turn",
)
```

* **STT-owned turns (`end_of_turn=True`, the default).** The config sets `enable_endpoint_detection: true`. When the speaker finished, Soniox finalizes the utterance and adds an `<end>` token. The stream emits `FINAL_TRANSCRIPT`, `END_OF_SPEECH` and `END_OF_TURN`, and the cascade commits the turn without its own endpointing delay.
  * Without a VAD, the first recognized token becomes `START_OF_SPEECH` and drives barge-in. It needs a word, so noise alone does not interrupt the agent.
  * Tune the endpoint with `max_endpoint_delay_ms` (500–3000, default 2000), `endpoint_sensitivity` (-1 to 1; higher ends turns sooner) and `endpoint_latency_adjustment_level` (0–3; higher is faster but may split turns).
* **Cascade-owned turns (`end_of_turn=False`).** Endpoint detection is off (set `enable_endpoint_detection=True` to keep it). The cascade's VAD and turn detector end turns. Its `flush()` sends `{"type": "finalize"}`: Soniox finalizes all audio sent so far and answers with the final tokens and a `<fin>` token, which becomes the `FINAL_TRANSCRIPT`.
  * A flush with no audio since the last `finalize` is answered at once with an empty final and sends nothing. Soniox asks clients not to `finalize` too often.
  * Soniox recommends finalizing after about 200 ms of trailing silence. A VAD's end-of-speech already waits for that.

## STT options (`SonioxSTT`)

| Option | Default | Notes |
|---|---|---|
| `model` | `stt-rt-v5` | `stt-rt-v4` is an alias |
| `api_key` | `$SONIOX_API_KEY` | An API key or a temporary key |
| `language` / `language_hints` | — | Sent as `language_hints`. `en-US` becomes `en`. `language_hints` overrides `language` |
| `language_hints_strict` | unset | Restrict recognition to the hinted languages |
| `language_identification` | unset | Tokens carry `language`. The utterance's most common one becomes `Transcript.language` |
| `speaker_diarization` | unset | Tokens carry `speaker`; read them from `stream.final_tokens`. Soniox notes that manual finalization lowers diarization accuracy |
| `terms` | `()` | Custom vocabulary: `context.terms` |
| `context` | — | Free text about the audio (`context.text`), or a full `context` object |
| `context_general` | — | Key/value facts, e.g. `{"domain": "Healthcare"}` (`context.general`) |
| `end_of_turn` | `True` | Emit `END_OF_TURN` and own the turn. See above |
| `enable_endpoint_detection` | `end_of_turn` | Override the default |
| `max_endpoint_delay_ms`, `endpoint_sensitivity`, `endpoint_latency_adjustment_level` | server defaults | Endpoint tuning. See above |
| `client_reference_id` | — | Up to 256 characters, shown in the Soniox console |
| `sample_rate` | `16000` | `pcm_s16le` mono. Input is resampled to this rate |
| `chunk_ms` | `40` | Binary audio chunk size |
| `keepalive_interval` | `5.0` s | A `keepalive` is sent after this long without audio. Soniox closes sessions idle for 20 s |
| `region`, `base_url`, `api_url` | `us` | See Setup |
| `close_timeout` | `5.0` s | Wait for `finished` after `end_input()` |
| `extra_config` | — | More config fields (`translation`...), sent verbatim |

`stt.config()` returns the config message without the API key.

### Event mapping

Soniox streams sub-word **tokens** with `start_ms`, `end_ms`, `confidence` and `is_final`. A final token is sent once. Non-final tokens are sent again with every response and may change. The stream keeps an utterance buffer of final tokens:

| Soniox response | `STTEventType` emitted |
|---|---|
| The first token of an utterance, final or not | `START_OF_SPEECH`. `transcript.start_time` is the token's start in stream seconds |
| Any response that changes the text | `INTERIM_TRANSCRIPT` with the utterance so far: its final tokens plus the current non-final ones |
| `<end>` token (endpoint detection) | `FINAL_TRANSCRIPT` + `END_OF_SPEECH` + `END_OF_TURN`. There is no `END_OF_TURN` when `end_of_turn=False`, or when a flush was pending |
| `<fin>` token (answer to `finalize`) | `FINAL_TRANSCRIPT`, which may be empty and still answers the flush, + `END_OF_SPEECH` if an utterance was open. Never `END_OF_TURN`: whoever flushed owns that decision |
| `finished: true` | Anything still buffered becomes a final. The session ends |

Finals carry word timings in stream seconds. Tokens are merged into words at their leading spaces, a word's confidence is its least confident token's, and `end_time` is the end of the last token. `stream.final_tokens` keeps the raw tokens of the last final, with speaker and language labels.

`end_input()` sends the audio tail, `finalize` (if audio was sent since the last one) and an empty frame. The stream then reads until `finished`, because closing early drops the last tokens. `final_audio_proc_ms` and `total_audio_proc_ms` on the stream mirror Soniox's progress counters.

Metrics: the base class reports `STTMetrics` with the flush latency when a flush is answered. Utterances that Soniox ends by itself are reported at each `<end>`, with `latency=None`.

## Batch (`transcribe()`)

`transcribe()` streams the audio over the same WebSocket without real-time pacing, then finalizes and joins the finals. Soniox also has an async file API (upload, then poll), which is not used here.

## Errors

Soniox sends `{"error_code": ..., "error_type": ..., "error_message": ...}` and closes. `retryable` says whether opening a **new session** can succeed.

| Situation | Exception | `retryable` |
|---|---|---|
| Missing API key | `ConfigurationError` (at construction) | — |
| `401` unauthenticated, `403` permission denied / temporary key expired, handshake 401 / 403 | `AuthenticationError` | no |
| `429` limit exceeded | `RateLimitError` | yes |
| `408` request timeout, connection timeout | `ProviderTimeoutError` | yes |
| `413` maximum session duration (300 min), an unexpected close, a network failure | `ProviderConnectionError` | yes |
| `500` / `503` | `ProviderError` | yes |
| `400` invalid request / model not available, `402` balance or budget exhausted | `ProviderError` | no |

## Latency notes (research note 03 §3.1)

* Artificial Analysis: 4.50 % WER, final transcript **0.054 s** after the end of speech. Pipecat: 1.27 % semantic WER, 260 / 305 ms time to final segment.
* **Forced finalization** is the fast path for a cascade with a local VAD and turn detector: the final arrives tens of milliseconds after `finalize`.
* Soniox bills the whole stream duration, including silence and keepalives. Always `end_input()` or `aclose()`.

## Testing

* **Unit tests** (`tests/test_soniox.py`) run against a local fake WebSocket server that replays Soniox's documented config, token, `<end>`, `<fin>`, `finished` and error messages. The temporary-key test uses `httpx.MockTransport`. Two cascade tests run `AgentSession` with the provider: one checks that `<end>` commits without the endpointing delay, the other that VAD-driven `finalize` works.
* **Real-API tests** are skipped without a key:

  ```bash
  SONIOX_API_KEY=... uv run pytest -m integration tests/test_soniox.py
  ```

  The speech round trip also needs `DEEPGRAM_API_KEY`, which synthesizes the test utterance.

## Protocol references (checked 2026-09-25)

* WebSocket API: <https://soniox.com/docs/stt/api-reference/websocket-api>
* Real-time transcription: <https://soniox.com/docs/stt/rt/real-time-transcription>
* Endpoint detection: <https://soniox.com/docs/stt/rt/endpoint-detection>
* Manual finalization: <https://soniox.com/docs/stt/rt/manual-finalization>
* Connection keepalive: <https://soniox.com/docs/stt/rt/connection-keepalive>
* Models: <https://soniox.com/docs/stt/models>
* Data residency: <https://soniox.com/docs/data-residency>
* Temporary API keys: <https://soniox.com/docs/api-reference/auth/create_temporary_api_key>

## Limitations and follow-ups

* No automatic reconnection when a session drops. The stream fails with a mapped, `retryable` error.
* The documentation does not say whether `finalize` is answered with `<fin>` when nothing is pending. The stream answers such flushes itself when no audio was sent since the last `finalize`; otherwise it relies on `<fin>`, and the cascade's `final_transcript_timeout` covers a missing answer. Check it against the live API.
* Speaker labels are only available as raw tokens (`stream.final_tokens`); `Transcript` has no speaker field.
* Real-time translation (`translation`) passes through `extra_config`, but translated tokens are not separated from the transcript.
* `transcribe()` does not use the async file API.
