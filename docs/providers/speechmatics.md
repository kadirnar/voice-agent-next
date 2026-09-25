# Speechmatics: Agent STT (Linden) and Realtime STT

| Component | Spec | Protocol |
|---|---|---|
| Agent STT: segments and turns | `stt="speechmatics/linden-1"` (default model) | `wss://global.rt.speechmatics.com/v2/agent` |
| Realtime STT: words and `EndOfUtterance` | `stt="speechmatics/enhanced"` or `"speechmatics/standard"` | `wss://global.rt.speechmatics.com/v2` |
| Batch `transcribe()` | any of the above | the same WebSocket, audio sent in one go |

Implementation: `src/voice_agent_next/providers/speechmatics.py`. It uses raw `websockets` and `httpx`, which are core dependencies, so there is no extra to install and no SDK.

**Agent STT** launched on 2026-09-17 with the Linden 1 model. It returns punctuated, speaker-attributed **segments** and **turn** messages instead of a running word stream, and has the best accuracy per unit of latency on Pipecat's benchmark (1.05 % semantic WER, 369 / 438 ms; research note 03 §3.1). The **Realtime API** (`enhanced`, `standard`) streams words with timings and ends utterances with `EndOfUtterance`.

## Setup

```bash
export SPEECHMATICS_API_KEY=...    # or pass api_key="..." to the constructor
van providers --kind stt           # speechmatics shows "ready" once the key is set
```

* The key goes in an `Authorization: Bearer` header. It never appears in URLs or logs.
* **Temporary keys.** `await stt.create_temporary_key(ttl=60)` calls `POST https://mp.speechmatics.com/v1/api_keys?type=rt`. Hand the key to a client that must not see yours, which then uses `SpeechmaticsSTT(jwt=key)`. It goes in the `jwt=` query parameter and works with any region.
* **Regions.** `region="global"` (the default) routes to the nearest region. `"eu"`, `"us"` and `"au"` pin the session. `base_url=` overrides the origin.

## Recommended cascades

```python
from voice_agent_next import AgentSession

# Speechmatics decides when the user is done. Leave out vad=...
session = AgentSession(stt="speechmatics", llm="openai/gpt-4.1-mini", tts="cartesia")

# Your VAD / turn detector decides; Speechmatics only transcribes
# (flush -> ForceEndOfUtterance). A VAD is required in this mode.
session = AgentSession(
    stt={"provider": "speechmatics", "end_of_turn": False},
    llm="openai/gpt-4.1-mini",
    tts="cartesia",
    vad="silero",
    turn_detector="smart_turn",
)
```

* **STT-owned turns (`end_of_turn=True`, the default).** `capabilities.end_of_turn` is `True`, so the cascade commits the user turn as soon as Speechmatics ends it.
  * Agent STT runs with `turn_config.turn_detection_mode: "vad"`. The service's VAD ends each turn, and `EndOfTurn` becomes `END_OF_TURN`.
  * Realtime models get `conversation_config.end_of_utterance_silence_trigger` (0.5 s by default here; Speechmatics recommends 0.5–0.8 s and keeping it below `max_delay`). `EndOfUtterance` becomes `END_OF_TURN`.
  * Without a VAD, the turn's first words (`StartOfTurn`, or the first segment or transcript) become `START_OF_SPEECH` and drive barge-in.
* **Cascade-owned turns (`end_of_turn=False`).** The cascade's VAD and turn detector end turns. Its `flush()` sends `ForceEndOfUtterance` with the audio time sent so far as `timestamp`.
  * Agent STT runs with `turn_detection_mode: "external"`: the service never ends a turn by itself. Each `ForceEndOfUtterance` flushes the open segment as a normal `AddSegment`, and `EndOfTurn` follows.
  * Realtime models still send `EndOfUtterance` after the silence trigger. Those utterances become plain finals that the cascade adds to the user's turn. Raise the trigger, or set it to `0` to disable it.
  * Neither API documents an answer to `ForceEndOfUtterance` when nothing was said. The stream then waits `force_end_grace` seconds (0.3 by default). If no turn starts in that window, it acknowledges the flush with an empty final. Set `force_end_grace=0` to acknowledge at once.

## STT options (`SpeechmaticsSTT`)

| Option | Default | Models | Notes |
|---|---|---|---|
| `model` | `linden-1` | — | `enhanced`, `standard` (Realtime) |
| `api_key` / `jwt` | `$SPEECHMATICS_API_KEY` / — | all | See Setup |
| `language` | `en` | all | `en-US` becomes `en`. Codes such as `cmn_en` pass through |
| `end_of_turn` | `True` | all | Emit `END_OF_TURN` and own the turn. See above |
| `enable_partials` | `True` | all | `AddPartialSegment` / `AddPartialTranscript` become interim results |
| `additional_vocab` | `()` | all | Custom dictionary: words, or `{"content": "gnocchi", "sounds_like": ["nyohki"]}`. At most 20,000 entries; a large list delays the session start |
| `domain`, `output_locale` | — | all | E.g. `domain="finance"`, `output_locale="en-US"` |
| `diarization` | `False` | all | `diarization: "speaker"`. Agent STT segments then carry `speaker` (`stream.speaker`) |
| `speaker_diarization_config`, `transcript_filtering_config` | — | all | Passed through, e.g. `{"remove_disfluencies": True}` |
| `emit_sentences` | unset | Agent STT | One segment per sentence |
| `max_delay` | unset (4 s on the server) | Realtime | 0.7–4 s latency bound of finals |
| `max_delay_mode` | unset (`flexible`) | Realtime | `flexible` or `fixed` |
| `end_of_utterance_silence_trigger` | `0.5` | Realtime | 0–2 s; `0` disables `EndOfUtterance`, `None` sends no `conversation_config` |
| `punctuation_overrides`, `audio_filtering_config`, `enable_entities` | — | Realtime | Passed through |
| `sample_rate` | `16000` | all | `pcm_s16le`. Agent STT takes 16 kHz only |
| `force_end_grace` | `0.3` s | all | See cascade-owned turns above |
| `chunk_ms` | `40` | all | Binary `AddAudio` chunk size |
| `start_timeout` | `20.0` s | all | Wait for `RecognitionStarted`. A large vocabulary takes up to ~15 s |
| `close_timeout` | `5.0` s | all | Wait for `EndOfTranscript` after `end_input()` |
| `extra_config` | — | all | More `transcription_config` fields, sent verbatim |

Options for the wrong API raise `ConfigurationError` when the STT is constructed, as do out-of-range values. `stt.start_message()` returns the `StartRecognition` message. On a Realtime stream, `await stream.set_recognition_config(max_delay=2.0, ...)` sends `SetRecognitionConfig` mid-session (the language cannot change).

### Event mapping

**Agent STT**

| Message | `STTEventType` emitted |
|---|---|
| `StartOfTurn`, or the turn's first segment | `START_OF_SPEECH`. `transcript.start_time` is the turn start in stream seconds |
| `AddPartialSegment` | `INTERIM_TRANSCRIPT` (the segment being built), only when the text changed |
| `AddSegment` | `FINAL_TRANSCRIPT` with the segment's `start_time` / `end_time`. A turn can have several |
| `EndOfTurn` | `END_OF_SPEECH` + `END_OF_TURN` with the turn's joined text. There is no `END_OF_TURN` when `end_of_turn=False` or when our own flush ended the turn |
| `SpeechStarted`, `SpeechEnded`, `AudioAdded`, `Info` | ignored (debug log). `Warning` is logged |

Agent STT has no word timings: `capabilities.word_timestamps` is `False`.

**Realtime**

| Message | `STTEventType` emitted |
|---|---|
| The first words of an utterance | `START_OF_SPEECH` |
| `AddPartialTranscript`, `AddTranscript` | `INTERIM_TRANSCRIPT` with the utterance so far: its finals plus the current partial |
| `EndOfUtterance` | `FINAL_TRANSCRIPT` (the whole utterance with word timings; punctuation joins the previous word) + `END_OF_SPEECH` + `END_OF_TURN`. There is no `END_OF_TURN` for `forced` utterances, or when `end_of_turn=False` |

Both: `end_input()` sends the audio tail, `ForceEndOfUtterance` where allowed, then `EndOfStream` with `last_seq_no`. The stream reads until `EndOfTranscript`. A turn still open at that point is closed with what arrived.

Metrics: the base class reports `STTMetrics` with the flush latency when a flush is answered. Turns that Speechmatics ends by itself are reported at their end, with `latency=None`.

## Batch (`transcribe()`)

`transcribe()` streams the audio over the same WebSocket without real-time pacing, then joins the finals. The Speechmatics batch jobs API (submit, then poll) is not used.

## Errors

Speechmatics sends `{"message": "Error", "type": ..., "reason": ...}` and closes. `retryable` says whether opening a **new session** can succeed.

| Situation | Exception | `retryable` |
|---|---|---|
| Missing API key and `jwt` | `ConfigurationError` (at construction) | — |
| `not_authorised`, `not_allowed`, handshake 401 / 403, close `4001` / `4003` | `AuthenticationError` | no |
| `quota_exceeded` (concurrent sessions), close `4005`, HTTP 429 | `RateLimitError` | yes |
| `start_recognition_timeout`, no `RecognitionStarted` within `start_timeout`, connection timeout | `ProviderTimeoutError` | yes |
| `idle_timeout`, `session_timeout`, an unexpected close, a network failure | `ProviderConnectionError` | yes |
| `job_error`, `unknown_error`, close `4013` / `1011` | `ProviderError` | yes |
| `invalid_*`, `protocol_error`, `timelimit_exceeded` (usage exhausted), close `4004` / `4006` | `ProviderError` | no |

## Latency notes (research note 03 §3.1)

* **Agent STT (Linden 1).** Artificial Analysis: 4.42 % WER at 0.16 s. Pipecat: 1.05 % semantic WER, 369 / 438 ms time to final segment. From $0.30 per hour, down to $0.16 at volume.
* **Realtime.** Finals arrive within `max_delay` (default 4 s). For a voice agent, lower it (e.g. 0.7–1.0 s) or rely on `EndOfUtterance` / `ForceEndOfUtterance`, which finalize at once.

## Testing

* **Unit tests** (`tests/test_speechmatics.py`) run against a local fake WebSocket server that replays the documented messages of both APIs: `RecognitionStarted`, `AudioAdded`, segments and turns, transcripts and `EndOfUtterance`, `EndOfTranscript`, `Error`, `Warning` and close codes. The temporary-key test uses `httpx.MockTransport`. Cascade tests run `AgentSession` with Agent STT turns (no VAD, no endpointing delay) and with VAD-driven `ForceEndOfUtterance` on both APIs.
* **Real-API tests** are skipped without a key:

  ```bash
  SPEECHMATICS_API_KEY=... uv run pytest -m integration tests/test_speechmatics.py
  ```

  The speech round trip also needs `DEEPGRAM_API_KEY`, which synthesizes the test utterance.

## Protocol references (checked 2026-09-25)

* Realtime API reference: <https://docs.speechmatics.com/rt-api-ref>
* Agent STT API reference: <https://docs.speechmatics.com/api-ref/agent-stt-websocket>
* Agent STT turn detection: <https://docs.speechmatics.com/speech-to-text/agent-stt/turn-detection>
* Agent STT segmentation: <https://docs.speechmatics.com/speech-to-text/agent-stt/segmentation>
* Realtime turn detection (`EndOfUtterance`): <https://docs.speechmatics.com/speech-to-text/realtime/turn-detection>
* Models: <https://docs.speechmatics.com/speech-to-text/models>
* Authentication, regions and temporary keys: <https://docs.speechmatics.com/get-started/authentication>
* Official SDK used as a cross-check of message shapes: <https://github.com/speechmatics/speechmatics-python-sdk/tree/main/sdk/agent_stt>

## Limitations and follow-ups

* No automatic reconnection when a session drops. The stream fails with a mapped, `retryable` error.
* The answer to `ForceEndOfUtterance` when nothing was said is undocumented, hence `force_end_grace`. Check it against the live API.
* Agent STT's documented turn detection has no eager end of turn yet, so there is no `EAGER_END_OF_TURN`.
* Speaker labels are only exposed as `stream.speaker` (the last Agent STT segment's); `Transcript` has no speaker field. `GetSpeakers`, translation and audio events are not mapped.
* `transcribe()` does not use the batch jobs API.
