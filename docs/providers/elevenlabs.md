# ElevenLabs: Flash v2.5 / v3 TTS and Scribe v2 STT

| Component | Spec | Class | Endpoint |
| --- | --- | --- | --- |
| TTS | `elevenlabs/eleven_flash_v2_5` (default model) | `voice_agent_next.providers.elevenlabs.ElevenLabsTTS` | `wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/multi-stream-input` (streams), `POST /v1/text-to-speech/{voice_id}/stream` (`synthesize()`) |
| TTS, Eleven v3 | `elevenlabs/eleven_v3_conversational`, `elevenlabs/eleven_v3` | same class | `wss://api.elevenlabs.io/v1/text-to-dialogue/multi-stream-input`, `POST /v1/text-to-dialogue/stream` |
| STT | `elevenlabs/scribe_v2_realtime` (default model) | `voice_agent_next.providers.elevenlabs.ElevenLabsSTT` | `wss://api.elevenlabs.io/v1/speech-to-text/realtime` (streams), `POST /v1/speech-to-text` (`transcribe()`, Scribe v2) |

Flash v2.5 is the latency reference among cloud TTS models (about 75 ms of model time) and returns character alignment, which this provider turns into word timings for word-exact barge-in truncation. Scribe v2 Realtime is among the most accurate streaming recognizers (3.59 % AA-WER, final transcript about 0.14 s after the end of speech) and supports manual commits, so the cascade's VAD and turn detector keep control of endpointing. See `docs/research/03-stt-tts-llm-landscape.md` §3 and §5.

## Setup

No extra is needed. Both components use only the core dependencies (`websockets`, `httpx`), not the ElevenLabs SDK.

```bash
export ELEVEN_API_KEY=sk_...        # or ELEVENLABS_API_KEY, or pass api_key="..."
uv run van providers --kind tts     # "elevenlabs ... ready"
```

The key is read from `api_key=`, then `ELEVEN_API_KEY`, then `ELEVENLABS_API_KEY`. It is sent in the `xi-api-key` header and never appears in URLs or logs.

`region="us"` pins the US-only servers (`api.us.elevenlabs.io`); `"eu"`, `"in"` and `"sg"` select the data-residency environments (enterprise). `base_url=` points anywhere else; `wss://` URLs are derived from it.

```python
from voice_agent_next import AgentSession

session = AgentSession(
    stt="elevenlabs/scribe_v2_realtime",
    vad="silero",  # manual commits: your VAD / turn detector end turns
    turn_detector="smart_turn",
    llm="openai/gpt-4.1-mini",
    tts={"provider": "elevenlabs/eleven_flash_v2_5", "voice": "<voice id>"},
)
```

```yaml
# van run --config agent.yaml
stt: {provider: elevenlabs/scribe_v2_realtime, language: en}
vad: silero
llm: openai/gpt-4.1-mini
tts: {provider: elevenlabs/eleven_flash_v2_5, voice: "<voice id>", speed: 1.05}
```

## TTS: `ElevenLabsTTS`

### How streaming works

`capabilities.streaming` is `True`, so the cascade sends its sentences straight to `tts.stream()`.

- **One multi-context WebSocket per voice**, shared by all streams of the instance. It opens on first use or in `warmup()` (`CascadeEngine.warmup()` calls it), which removes the TLS and WebSocket handshake from the first reply. `inactivity_timeout=180` (the maximum) keeps it open between replies; if ElevenLabs closes it anyway, the next stream reconnects, and a send that hits the closed socket retries once. The multi-context API was built for this: only generation time counts against the plan's concurrency limit, not the open socket.
- **One context per segment.** A segment is the text between two `flush()` calls; in the cascade that is a whole reply. The context opens with `{"text": " ", "context_id", "voice_settings", "generation_config", "pronunciation_dictionary_locators"}` (settings are only accepted on a context's first message), receives the text, and ends with `flush` (if text is still buffered) and `close_context`. ElevenLabs answers with `isFinal`, which ends the segment. Segments play in order even when their generation overlaps; at most 5 contexts may be open per socket, so a burst of flushes waits for a free slot.
- **Per-sentence flush.** Text is sent at word boundaries only: a partial word (raw LLM tokens) waits for the rest of it, and every input ends with a space, as ElevenLabs requires. A push that ends a sentence (`. ! ? …` and CJK terminators), or the first clause of a segment (`, ; : —`), is sent with `flush: true`, so ElevenLabs generates it at once instead of buffering for more text. The cascade pushes whole sentences and a short first clause, which is exactly what ElevenLabs recommends for agents ("use the `flush: true` flag at the end of complete sentences"). `auto_mode=true` lets ElevenLabs trigger the rest; passing `chunk_length_schedule` turns it off (the schedule then decides). `flush_sentences=False` disables the flushes.
- **Barge-in.** Closing a stream sends `close_context` for every context that is still open and drops its late audio; a context whose input had already ended is not messaged again (on the dialogue API that is a protocol error that closes the socket). ElevenLabs keeps generating text that was already flushed, so a few late chunks may still arrive; they are discarded. The socket stays open for the next reply.
- **Keep-alive.** A context waiting for more text (slow LLM, long tool call) gets a keep-alive (`{"context_id", "text": ""}`) every `keepalive_interval` seconds (default: half the inactivity timeout).
- **Watchdog.** After a segment's input ends, if ElevenLabs sends nothing for `receive_timeout` seconds (default 10), the stream fails with `ProviderTimeoutError`.
- **`synthesize(text)`** uses the HTTP streaming endpoint (`output_format=pcm_<rate>`) and yields the raw PCM as it arrives. It returns no word timings. `streaming=False` makes the whole TTS work sentence by sentence over HTTP (the `SentenceStreamAdapter`), with segment-level alignment only.

### Word timings and alignment

With `word_timestamps=True` (the default) the socket URL carries `sync_alignment=true` and every audio chunk comes with its character alignment. The provider assembles characters into words (whitespace-separated, punctuation attached) and yields items with an empty `frame` and `words: list[WordTiming]`, in seconds **from the start of the stream's audio** across segments. A word is reported once it is complete, before or with the audio it starts in. On barge-in the cascade keeps the words that started before the played position, so the history holds exactly what the user heard.

- `alignment="original"` (default) reports the text as sent; `"normalized"` reports what was spoken (numbers and dates spelled out, pronunciation dictionaries applied). With pronunciation dictionaries the default switches to `"normalized"`, because the original-text alignment has been observed to restart mid-sentence then.
- **Time base.** ElevenLabs documents alignment times as "relative to the returned chunk from the model", and several clients (Pipecat, a Rust client) accumulate chunk offsets accordingly; others (LiveKit, the Fish Audio compatibility layer) treat the times, notably `normalizedAlignment`, as absolute from the start of the context. The provider does not have to guess: it compares each chunk's first character time with how much audio the context has already produced (chunk-relative times restart near 0, context-absolute ones continue), keeps the result once it is unambiguous (after 0.5 s of audio), and maps both onto the stream timeline.

### Eleven v3 (`eleven_v3_conversational`, `eleven_v3`)

The text-to-speech WebSockets do not serve Eleven v3. Models whose id starts with `eleven_v3` stream through the **Text to Dialogue** multi-context WebSocket instead, with the same context lifecycle and a different framing: the first message registers the voice (`{"context_id", "voices": [voice_id], "voice_settings": {"stability"}}`), text goes in `{"context_id", "inputs": [{"text", "voice_id"}], "flush"}`, keep-alives are `{"context_id", "keep_alive": true}` (every 10 s: dialogue contexts time out after a fixed 20 s), and responses use snake_case (`context_id`, `is_final`, `alignment.char_start_times_ms`). Only `stability` is supported as a voice setting; other voice settings, `chunk_length_schedule` and SSML parsing are ignored with a warning. `synthesize()` uses `POST /v1/text-to-dialogue/stream`. Each open dialogue socket holds one dialogue session for its lifetime; an idle one closes after 20 s and the next reply reconnects.

### Options

| Option | Default | Notes |
| --- | --- | --- |
| `model` | `eleven_flash_v2_5` | `eleven_turbo_v2_5`, `eleven_multilingual_v2`, `eleven_flash_v2` (English), `eleven_v3_conversational`, `eleven_v3` |
| `voice` | `JBFqnCBsd6RMkjVDRZzb` | ElevenLabs voice id ("George", the API reference's example voice); `stream(voice=...)` overrides it per stream (one socket per voice) |
| `sample_rate` | `24000` | raw PCM (`pcm_<rate>`): 8000, 16000, 22050, 24000, 32000, 44100 (Pro plan), 48000 |
| `language` | `None` | ISO 639-1 (`"pt-BR"` is sent as `pt`); only for Flash/Turbo v2.5 and v3, since Multilingual v2 rejects `language_code` (a warning is logged) |
| `stability`, `similarity_boost`, `style` | `None` | 0-1; `None` keeps the voice's stored settings |
| `use_speaker_boost` | `None` | voice setting |
| `speed` | `None` | 0.7-1.2 |
| `auto_mode` | on unless `chunk_length_schedule` is set | server-side generation triggers |
| `chunk_length_schedule` | `None` | e.g. `[50, 120, 160, 290]` (50-500 each) |
| `apply_text_normalization` | `None` | `"auto"`, `"on"`, `"off"` (for v2.5 models `"on"` is an enterprise feature) |
| `flush_sentences` | `True` | flush at sentence ends and the first clause |
| `word_timestamps` / `alignment` | `True` / auto | see above |
| `pronunciation_dictionaries` | `None` | up to 3 `(dictionary_id, version_id)` pairs |
| `seed`, `enable_ssml_parsing` | `None`, `False` | passed through |
| `enable_logging` | `True` | `False` = zero retention mode (enterprise) |
| `inactivity_timeout` / `keepalive_interval` | `180` / half of it | seconds |
| `receive_timeout`, `connect_timeout` | `10.0`, `10.0` | watchdog after end of input (also the HTTP read timeout); connect timeout |
| `http_client` | `None` | your own `httpx.AsyncClient` for `synthesize()` (not closed by `aclose()`) |
| `base_url`, `region`, `api_key` | ElevenLabs defaults | see Setup |

## STT: `ElevenLabsSTT`

### Manual commits (default)

The stream sends 16 kHz PCM (`sample_rate`: 8000-48000) as base64 `input_audio_chunk` JSON messages of `chunk_duration` seconds (100 ms by default; ElevenLabs suggests 0.1-1 s). `partial_transcript` becomes `INTERIM_TRANSCRIPT`, `committed_transcript` becomes `FINAL_TRANSCRIPT`.

`flush()` commits: the buffered tail is sent with `commit: true`. The cascade flushes at every candidate end of speech and waits for the final, so every flush is answered:

- a commit with audio is answered by Scribe's committed transcript (possibly empty);
- a flush with no audio since the last answered commit is acknowledged at once with an empty final and sends nothing, because Scribe throttles rapid commits (`commit_throttled`) and warns that committing several times in a short sequence degrades accuracy;
- a flush while a commit is in flight is answered by that commit's transcript.

A throttled commit is logged, not fatal: its audio stays uncommitted and the next flush commits it again. `end_input()` commits, waits up to `close_timeout` for the transcript, then closes the socket. Scribe also commits on its own after about 36 s of uncommitted audio; those finals are reported too.

Use this mode with a VAD in the cascade (`vad="silero"` or `"energy"`), optionally with a turn detector.

### Server VAD (`commit_strategy="vad"`)

Scribe's VAD commits after `vad_silence_threshold_secs` (0.3-3.0) of silence (`vad_threshold`, `min_speech_duration_ms`, `min_silence_duration_ms` tune it). The stream then also emits `START_OF_SPEECH` with the first partial of a segment and `END_OF_SPEECH` after its committed transcript, so a cascade can run without a local VAD. The cascade still flushes when speech ends; that commit only covers the silence since Scribe's own commit and is answered with an (empty) final.

### Timestamps and language

With `include_timestamps=True` or `include_language_detection=True`, Scribe sends every commit twice: `committed_transcript`, then `committed_transcript_with_timestamps` with `words` (type `word` / `spacing` / `audio_event`) and `language_code`. The final is emitted on the second copy, with `words` (spacing and audio events dropped, `logprob` turned into a confidence), `start_time` / `end_time` and the detected language. This costs the delay between the two copies, so both options are off by default. Word times are passed through as Scribe reports them.

### Other options

| Option | Default | Notes |
| --- | --- | --- |
| `model` | `scribe_v2_realtime` | batch models (`scribe_v2`, `scribe_v2_medical`, `scribe_v1`) are batch-only: the cascade wraps them in a `StreamAdapter` with its VAD |
| `language` | `None` | ISO 639-1/3 (`"en-US"` is sent as `en`); `None` = automatic detection |
| `secondary_languages` | `()` | languages the speaker may switch to |
| `keyterms` | `()` | biasing terms (realtime: up to 50 terms of at most 20 characters; surcharge) |
| `no_verbatim` | `None` | drop filler words and false starts |
| `previous_text` | `None` | context (e.g. the agent's last question, ideally under 50 characters) sent with the first audio chunk only |
| `filter_background_audio` | `None` | realtime background filtering |
| `keepalive_interval` | `5.0` | after this many seconds without audio, one chunk of silence is sent (Scribe closes sessions without audio activity); `None` disables it |
| `enable_logging` | `True` | `False` = zero retention mode (enterprise) |
| `batch_model`, `tag_audio_events`, `timeout` (formerly `request_timeout`) | `scribe_v2`, `False`, `60.0` | `transcribe()` |
| `connect_timeout`, `close_timeout` | `10.0`, `5.0` | seconds |

### Batch: `transcribe()`

`transcribe()` posts the audio to `POST /v1/speech-to-text` (multipart) with `batch_model` (Scribe v2, more accurate than the realtime model), `timestamps_granularity=word`, and the language, keyterms and `no_verbatim` options. 16 kHz mono audio is sent as raw PCM with `file_format=pcm_s16le_16`, which skips server-side decoding; other rates are sent as WAV. The transcript carries word timings, a confidence (mean word probability) and the detected language.

## Errors

| Situation | Exception |
| --- | --- |
| no API key | `ConfigurationError` (at construction) |
| HTTP/handshake 401/403, `auth_error`, `unaccepted_terms`, a close reason about the API key | `AuthenticationError` |
| 429, `rate_limited`, `queue_overflow`, `resource_exhausted`, concurrency limits | `RateLimitError` (`retryable=True`) |
| `quota_exceeded`, `insufficient_credits` | `RateLimitError` (`retryable=False`) |
| 408 / 504, connect timeout, no answer after the end of the input, `insufficient_audio_activity` | `ProviderTimeoutError` |
| network failure, socket closed mid-stream (1006, 1011) | `ProviderConnectionError` (`retryable=True`) |
| other API errors (`detail.code` / `message` / `param` / `request_id`, 422 validation lists, 1008 policy closes) | `ProviderError` (`retryable` for 5xx and transcriber errors) |

An error for one context fails only the stream that owns it; an error without a context fails every stream on the socket and retires the socket, so the next reply reconnects.

## Latency notes

- Call `await engine.warmup()` (or `await tts.warmup()`) before the first turn so the first reply does not pay for the WebSocket handshake.
- Keep `flush_sentences` and `auto_mode` on (the defaults) when the text is already segmented. A `chunk_length_schedule` without flushes delays the first audio until enough characters arrive (120 by default).
- `word_timestamps=True` requests `sync_alignment`. If you do not need word-exact truncation, `word_timestamps=False` drops it; the cascade then estimates the heard text from the audio duration.
- The output is raw 24 kHz PCM by default: no MP3 decoding. For 8 kHz telephony set `sample_rate=8000`.
- Flash v2.5 does not normalize numbers by default (phone numbers, dates); let the LLM write them out or set `apply_text_normalization="on"` (enterprise for v2.5 models).
- Scribe: smaller `chunk_duration` lowers latency at the cost of more messages. Enable timestamps or language detection only if you need them (the final then waits for the second copy of each commit).

## Testing

Unit tests (`tests/test_elevenlabs.py`) run fake servers on `127.0.0.1` (`websockets.serve`) and `httpx.MockTransport`. The fakes replay the documented messages: context init/flush/close and keep-alives, chunk-relative and context-absolute alignment (split words, normalized text with a leading space), `isFinal`, error payloads with and without a context, 1008 policy closes, the context limit, reconnects, the dialogue framing, Scribe's partial/committed/timestamped messages, error types, commit throttling, keep-alive silence, the batch multipart API, and full cascade round trips (word-exact truncation, Scribe commits driving endpointing). Tests against the real API are marked `integration` and skipped without a key:

```bash
ELEVENLABS_API_KEY=sk_... uv run pytest -m integration tests/test_elevenlabs.py
```

## Protocol references

Checked on 2026-09-24 (the `.md` versions of these pages carry the AsyncAPI/OpenAPI definitions):

- TTS WebSocket: https://elevenlabs.io/docs/api-reference/text-to-speech/v-1-text-to-speech-voice-id-stream-input
- TTS multi-context WebSocket: https://elevenlabs.io/docs/api-reference/text-to-speech/v-1-text-to-speech-voice-id-multi-stream-input and the guide https://elevenlabs.io/docs/eleven-api/guides/how-to/websockets/multi-context-web-socket
- Realtime TTS guide (buffering, `flush`, `chunk_length_schedule`): https://elevenlabs.io/docs/eleven-api/guides/how-to/websockets/realtime-tts
- Text to Dialogue multi-context WebSocket (Eleven v3): https://elevenlabs.io/docs/api-reference/text-to-dialogue/ttd-multi-websocket and https://elevenlabs.io/docs/eleven-api/guides/how-to/websockets/tts-vs-ttd-websockets
- HTTP streaming: https://elevenlabs.io/docs/api-reference/text-to-speech/stream and https://elevenlabs.io/docs/api-reference/text-to-dialogue/stream
- Scribe v2 Realtime: https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime, https://elevenlabs.io/docs/eleven-api/guides/how-to/speech-to-text/realtime/transcripts-and-commit-strategies and https://elevenlabs.io/docs/eleven-api/guides/how-to/speech-to-text/realtime/event-reference
- Batch speech to text: https://elevenlabs.io/docs/api-reference/speech-to-text/convert
- Models, languages, concurrency: https://elevenlabs.io/docs/overview/models; latency: https://elevenlabs.io/docs/eleven-api/guides/how-to/best-practices/latency-optimization; errors: https://elevenlabs.io/docs/eleven-api/resources/errors

## Limitations and follow-ups

- The unit tests replay documented shapes; the parts the documentation leaves open (alignment time base, `isFinal` vs `is_final`, error payloads on the text-to-speech socket) are handled defensively and should be confirmed with the integration tests on a real key.
- `synthesize()` returns no word timings; `/v1/text-to-speech/{voice_id}/stream/with-timestamps` could provide them.
- Scribe word times are passed through unchanged (their origin, session or segment, is not documented); the cascade only uses them for metrics when it has no VAD.
- The STT WebSocket is not reconnected mid-session (`session_time_limit_exceeded` or a dropped connection fail the stream).
- μ-law / A-law output, the single-context `stream-input` endpoint, Scribe entity detection and multichannel batch transcription are not used.
