# Cartesia: Sonic TTS and Ink STT

| Component | Spec | Class | Endpoint |
| --- | --- | --- | --- |
| TTS | `cartesia/sonic-3.6` (default model) | `voice_agent_next.providers.cartesia.CartesiaTTS` | `wss://api.cartesia.ai/tts/websocket` (streams), `POST /tts/bytes` (`synthesize()`) |
| STT | `cartesia/ink-2` (default model) | `voice_agent_next.providers.cartesia.CartesiaSTT` | `wss://api.cartesia.ai/stt/turns/websocket` (turn detection) or `wss://api.cartesia.ai/stt/websocket` (finalize) |

Sonic 3.6 is ranked first in the Artificial Analysis Speech Arena. It accepts text over a WebSocket as it is generated and returns word timestamps. Ink-2 transcribes streaming audio and has semantic turn detection built in. See `docs/research/03-stt-tts-llm-landscape.md` §3 and §5.

## Setup

No extra is needed. Both components use only the core dependencies (`websockets`, `httpx`), not the Cartesia SDK.

```bash
export CARTESIA_API_KEY=sk_car_...     # or pass api_key="..." to the constructor
uv run van providers --kind tts        # "cartesia ... ready"
```

Every request pins the API version. It sends the `Cartesia-Version: 2026-08-14` header; WebSocket URLs also carry the `cartesia_version` query parameter. Override the version with `api_version="..."`. The API key goes in the `Authorization: Bearer` and `X-API-Key` headers and never in URLs.

```python
from voice_agent_next import Agent, AgentSession

session = AgentSession(
    stt="cartesia/ink-2",  # Ink decides when the user's turn ends: no VAD needed
    llm="openai/gpt-4.1-mini",
    tts={"provider": "cartesia/sonic-3.6", "voice": "<voice id>"},
)
```

```yaml
# van run --config agent.yaml
stt: {provider: cartesia/ink-2}
llm: openai/gpt-4.1-mini
tts: {provider: cartesia/sonic-3.6, voice: "<voice id>", speed: 1.1}
```

## TTS: `CartesiaTTS`

### How streaming works

`capabilities.streaming` is `True`, so the cascade sends its sentences directly to `tts.stream()`. The `SentenceStreamAdapter` is not used.

- **One persistent WebSocket per `CartesiaTTS`.** All streams of the instance share it, because Cartesia multiplexes contexts by `context_id`. The socket opens on first use or in `warmup()` (`CascadeEngine.warmup()` calls it), which removes a TLS and WebSocket handshake from the first reply. Cartesia closes WebSockets after about 5 idle minutes. The next stream then reconnects, and if a send hits the closed socket it retries once.
- **One `context_id` per segment.** A segment is the text between two `flush()` calls. The whole reply is one segment in the cascade. Each `push_text()` delta is sent with `continue: true`, so prosody carries across sentences. `flush()` / `end_input()` sends an empty `transcript` with `continue: false`. Every input repeats the same `model_id`, `voice`, `output_format` and options, as the API requires.
- **Segments play in order.** Segment N+1 can start generating while segment N is still arriving. Its audio is buffered until N's `done`. If Cartesia ends a context early (for example if a context expires while the LLM is slow), the rest of the segment continues on a new context.
- **Cancellation.** Closing a stream sends `{"context_id": ..., "cancel": true}` for each context that has not finished and drops any late messages for it. The cascade closes the stream on barge-in (`cancel_response()`). Cartesia only halts generations that have not started yet, so a few chunks may still arrive; they are discarded. The socket stays open for the next reply.
- **Watchdog.** After a segment's input ends, if Cartesia sends nothing for `receive_timeout` seconds (default 10), the stream fails with `ProviderTimeoutError`. Without it, a response could stay silent indefinitely.
- **`synthesize(text)`** (one complete text) uses `POST /tts/bytes` and streams the raw PCM as it arrives. It returns no timestamps.

### Word timestamps

With `word_timestamps=True` (the default), requests set `add_timestamps: true`. Each Cartesia `timestamps` message becomes a `SynthesizedAudio` item with an empty `frame` and `words: list[WordTiming]`. Times are in seconds from the start of the stream's audio. That timeline is the concatenation of every frame the stream has yielded, across segments; Cartesia's own per-context offsets are already added. On barge-in, the words the user heard are the words with `end <= played_seconds`. This lets truncation cut at exact words instead of estimating from audio duration. The cascade engine currently truncates per sentence and does not read `words` yet (see the follow-ups).

### `max_buffer_delay_ms`

By default, Cartesia buffers streamed text for up to 3000 ms while it waits for "enough context". That delay adds to time-to-first-audio when text arrives in pieces. This provider sends `max_buffer_delay_ms: 0` by default: generation starts as soon as text arrives. That is the right setting when text is already pushed in sentences, as the cascade's `SentenceSegmenter` does. If you push raw LLM tokens into `tts.stream()` yourself, set it to 300-1000 so Cartesia can group tokens into natural phrases. `None` omits the field.

### Options

| Option | Default | Notes |
| --- | --- | --- |
| `model` | `sonic-3.6` | alias of the latest stable snapshot; pin a snapshot such as `sonic-3.6-2026-08-27` for reproducible output; `sonic-3.5`, `sonic-3`, `sonic-latest` also work |
| `voice` | `f786b574-daa5-4673-aa0c-cbe3e8534c02` | Cartesia voice id (the one Cartesia's quickstart uses); `stream(voice=...)` overrides it per stream |
| `sample_rate` | `24000` | raw `pcm_s16le`; one of 8000, 16000, 22050, 24000, 44100, 48000 |
| `language` | `None` | e.g. `"en"`, `"fr"`; `None` uses the API default; Sonic 3.6 supports 44 languages |
| `speed` / `volume` / `emotion` | `None` | `generation_config` (speed 0.6-1.5, volume 0.5-2.0, emotion such as `"calm"`) |
| `max_buffer_delay_ms` | `0` | see above (0-5000) |
| `word_timestamps` | `True` | `add_timestamps` on streams |
| `pronunciation_dict_id` | `None` | custom pronunciations |
| `receive_timeout` | `10.0` | watchdog after end of input; also the HTTP read timeout of `synthesize()` |
| `connect_timeout` | `10.0` | WebSocket/HTTP connect timeout |
| `http_client` | `None` | your own `httpx.AsyncClient` for `synthesize()` (not closed by `aclose()`) |
| `base_url`, `api_version`, `api_key` | Cartesia defaults | `wss://` is derived from `base_url` |

## STT: `CartesiaSTT`

Ink-2 transcribes English, French, Hindi, Japanese and Spanish and detects the language itself. The class has two modes.

### Turn detection (default for `ink-2`, `ink-preview`)

The stream uses `/stt/turns/websocket`, where the model decides when the user's turn starts and ends:

| Cartesia event | STT events |
| --- | --- |
| `turn.start` | `START_OF_SPEECH` |
| `turn.update` (cumulative text of the turn) | `INTERIM_TRANSCRIPT` |
| `turn.eager_end` | `EAGER_END_OF_TURN` (with the text so far) |
| `turn.resume` | `TURN_RESUMED` |
| `turn.end` (definitive text) | `FINAL_TRANSCRIPT`, `END_OF_SPEECH`, `END_OF_TURN` |

Use this mode in a cascade **without a VAD** (`vad=None`). `turn.start` triggers barge-in, and `END_OF_TURN` commits the turn with the final text immediately. This endpoint has no finalize command, so `flush()` only sends the buffered audio. `end_input()` sends `{"type": "close"}`, and Cartesia then processes the remaining audio before it closes the socket. You can tune the detector with `turn_start_threshold`, `turn_eager_end_threshold`, `turn_end_threshold` and `turn_end_timeout_ms`; `None` keeps the API defaults. Do not combine this mode with a cascade VAD. The VAD endpointing and Ink's `turn.end` would then both commit turns.

### External endpointing (`turn_detection=False`, always for `ink-whisper`)

The stream uses `/stt/websocket`. Your VAD or turn detector decides when the turn ends. `flush()` sends `finalize`, and the final transcript arrives quickly: Artificial Analysis measured 0.067 s for external endpoints versus 0.43 s for semantic endpoints (note 03 §3.1). Transcripts carry `words` (word timings), `language`, `start_time` and `end_time`. When a flush has nothing to finalize, the stream still emits an empty `FINAL_TRANSCRIPT` at `flush_done`, so the cascade does not wait for its final-transcript timeout. `end_input()` sends `finalize` and then `close`, and waits for `done`. `transcribe()` (batch) always uses this mode.

```python
from voice_agent_next import AgentSession

session = AgentSession(
    stt={"provider": "cartesia/ink-2", "turn_detection": False},
    vad="silero",
    turn_detector="smart_turn",  # your endpointing
    llm="...",
    tts="cartesia/sonic-3.6",
)
```

### Options

| Option | Default | Notes |
| --- | --- | --- |
| `model` | `ink-2` | `ink-preview`, `ink-whisper` (finalize mode only) |
| `turn_detection` | auto | `True` for Ink-2 models, `False` for `ink-whisper` |
| `sample_rate` | `16000` | input is resampled to it and sent as `pcm_s16le` |
| `keyterms` | `()` | up to 100 terms / 1200 characters to boost (`keyterm` query parameters) |
| `language` | `None` | only sent to `ink-whisper` (Ink-2 detects the language) |
| `min_volume`, `max_silence_duration_secs` | `None` | `ink-whisper` endpointing options |
| `chunk_duration` | `0.05` | audio is sent in about 50 ms binary frames (Cartesia suggests about 100 ms; smaller chunks detect turns sooner) |
| `close_timeout` | `5.0` | how long to wait for the last results after `end_input()` |

## Errors

| Situation | Exception |
| --- | --- |
| no API key | `ConfigurationError` (at construction) |
| HTTP/handshake 401/403 or an `error` message with that status | `AuthenticationError` |
| 429 | `RateLimitError` (`retryable=True`) |
| 408 / 504, no answer after end of input, connect timeout | `ProviderTimeoutError` |
| network failure, socket closed mid-stream | `ProviderConnectionError` (`retryable=True`) |
| other API errors (`status_code`, `title`, `message`, `error_code`) | `ProviderError` (`retryable` for 5xx) |

A reconnect that loses in-flight audio is reported as an error, not retried silently (see the Pipecat #5305 case study in note 05 §7.3).

## Latency notes

- Call `await engine.warmup()` (or `await tts.warmup()`) before the first turn so the first reply does not pay for the WebSocket handshake. Later replies reuse the socket.
- Keep `max_buffer_delay_ms` low (the default is 0) when the text is already segmented. Cartesia's 3 s default buffer is the most common cause of slow first audio (note 05 §2.5).
- The output is 24 kHz `pcm_s16le`, so 24 kHz transports need no resampling. For 8 kHz telephony, set `sample_rate=8000`; Cartesia also offers μ-law, but this provider always requests s16le.
- TTFB is measured per stream (per reply), from the first text to the first audio byte (`TTSMetrics.ttfb`). The timer does not start at connection setup.
- In a cascade, Ink turn detection replaces the VAD and the turn detector. Choose external endpointing when you want the fastest finals and already run a good endpointer.

## Testing

Unit tests (`tests/test_cartesia.py`) run fake servers on `127.0.0.1` with `websockets.serve` and `httpx.MockTransport`. The fakes replay Cartesia's messages: continuations, segments and flushes, timestamps, cancellation, errors, reconnects, both STT endpoints, and full cascade round trips. Tests against the real API are marked `integration` and skipped without a key:

```bash
CARTESIA_API_KEY=sk_car_... uv run pytest -m integration tests/test_cartesia.py
```

## Limitations and follow-ups

- The cascade does not yet use `SynthesizedAudio.words` for truncation. It still estimates the heard text proportionally for streaming TTS, which is follow-up work in the cascade engine.
- Phoneme timestamps (`add_phoneme_timestamps`), Cartesia `flush` / `flush_id` within a context, and the SSE endpoint are not used.
- The STT WebSocket is not reconnected automatically mid-session. A dropped connection fails the stream with `ProviderConnectionError`.
- Cartesia documents that contexts expire about 1 s after their last audio. The provider continues on a new context when Cartesia ends one early. If an input is silently dropped after expiry, the stream reports it through the watchdog (`ProviderTimeoutError`) and does not recover it.
