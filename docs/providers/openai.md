# OpenAI (LLM, STT, TTS)

| Component | Spec | Class | Transport |
|---|---|---|---|
| LLM | `llm="openai/gpt-4.1-mini"` | `OpenAILLM` | Chat Completions through the `openai` SDK |
| STT | `stt="openai/gpt-live-transcribe"` | `OpenAISTT` | Realtime transcription session (WebSocket); `/audio/transcriptions` for batch |
| TTS | `tts="openai/gpt-4o-mini-tts"` | `OpenAITTS` | `/audio/speech`, 24 kHz PCM streamed over HTTP |

The speech-to-speech engine (`AgentSession("openai/gpt-realtime-2.1")`) is documented in
[OpenAI Realtime](openai-realtime.md). The STT and TTS use only core dependencies
(`websockets`, `httpx`); the LLM needs the `openai` extra. The same STT and TTS classes also
drive OpenAI-compatible speech servers (Speaches, LocalAI, Azure OpenAI, Kokoro-FastAPI):
see [Compatible speech servers](#compatible-speech-servers).

```python
from voice_agent_next import AgentSession

session = AgentSession(
    stt="openai/gpt-live-transcribe",
    llm="openai/gpt-4.1-mini",
    tts="openai/gpt-4o-mini-tts",
    vad="silero",  # the VAD decides when the user paused; the STT then finalizes
)
```

## LLM

`llm="openai/gpt-4.1-mini"` streams OpenAI's Chat Completions API through the official
[`openai`](https://pypi.org/project/openai/) SDK. It forwards text deltas as they arrive,
emits complete tool calls and reports token usage, including cached prompt tokens.

The same class, `OpenAILLM`, talks to any server that speaks the Chat Completions
protocol. Preconfigured providers exist for Ollama, llama.cpp, vLLM, LM Studio, Groq,
Cerebras, Together, OpenRouter, DeepSeek, Fireworks and SambaNova: see
[OpenAI-compatible servers](openai-compatible.md).

### Setup

```bash
pip install 'voice-agent-next[openai]'   # or: uv sync --extra openai
export OPENAI_API_KEY=sk-...
```

Credentials come from `api_key=`, then `OPENAI_API_KEY`. Without a key the constructor
raises `ConfigurationError`. `OPENAI_BASE_URL`, the SDK's own variable, points the
provider at a gateway, and the OpenAI key is sent there too. An explicit `base_url=`
means *another server*: `OPENAI_API_KEY` is then never sent, so pass `api_key=` if that
server needs one.

### Usage

```python
from voice_agent_next import AgentSession, ChatContext, create

# in a cascade
session = AgentSession(
    stt="deepgram/nova-3", llm="openai/gpt-4.1-mini", tts="cartesia", vad="silero"
)

# or directly
llm = create("llm", "openai/gpt-4.1", temperature=0.6, max_tokens=300)
ctx = ChatContext()
ctx.add_message("system", "You are a helpful voice assistant.")
ctx.add_message("user", "What's the weather in Paris?")
async for chunk in llm.chat(ctx, tools=[get_weather]):
    print(chunk.delta, chunk.tool_calls)
```

```yaml
# config file
llm:
  provider: openai/gpt-4.1-mini
  max_tokens: 300
  extra: {prompt_cache_key: support-bot}
```

### Models

Figures from the Pipecat voice benchmark (30-turn tool-use conversations, strict per-turn
pass rate, p50 time to first audio token) and list prices per 1M input/output tokens, as
collected in [research note 03 §7](../research/03-stt-tts-llm-landscape.md):

| Model | Notes for voice |
|---|---|
| `gpt-4.1-mini` (default) | 85.3 % at 851 ms. $0.40 / $1.60. Accepts `temperature`. |
| `gpt-4.1` | 96.3 % at 536 ms, the best OpenAI result. $2 / $8. |
| `gpt-5.6-luna` | 88.3 % at 671 ms with reasoning `none`. $0.20 / $1.20. Reasoning model (see below). |
| `gpt-5.4-mini`, `gpt-5.4-nano` | $0.75 / $4.50 and $0.20 / $1.25. Reasoning models. |
| `gpt-6-luna` | Newest (Sep 2026), $0.10 / $0.50; not in voice benchmarks yet. |

Any model id the API accepts works. **Reasoning models** reject `temperature`, need
`max_completion_tokens` (sent automatically for OpenAI) and think before they answer,
which delays the first spoken word: set `reasoning_effort="none"` (or `"minimal"` /
`"low"`, depending on the model).

### Options

| Option | Default | Meaning |
|---|---|---|
| `model` | `gpt-4.1-mini` | Model id. |
| `api_key`, `base_url` | env | See Setup. `base_url` includes the version path (`https://host/v1`). |
| `temperature`, `max_tokens` | not sent | Defaults for every request, overridable per `chat()`. `max_tokens` is sent as `max_completion_tokens` to OpenAI and as `max_tokens` to other servers. |
| `reasoning_effort` | not sent | `"none"`, `"minimal"`, `"low"`, ... for reasoning models. |
| `parallel_tool_calls` | not sent | `False` limits the model to one tool call per response. Only sent with tools. |
| `extra` | `{}` | Extra request parameters for every request; the per-call `extra=` wins. Parameters the SDK knows (`seed`, `stop`, `top_p`, `service_tier`, `prompt_cache_key`, `verbosity`...) are passed as such; anything else goes into the JSON body (`extra_body`). A `None` value removes a default. |
| `headers` | — | Extra HTTP headers on every request. |
| `capabilities` | tools, parallel tools, image input | Override the declared `LLMCapabilities`. |
| `audio_input` | from the model id | The model hears `AudioContent` (half-cascade). Unset: the host's default, else the known-model table. |
| `audio_format` | the host's | `AudioInputFormat` (or a mapping): `format` (`"wav"` or `"pcm16"`), `sample_rate`, `data_url`. |
| `audio_history` | all | Send only the last N user audio clips as audio, older ones as their transcripts. |
| `voice` | — | Voice of an audio-output model; also turns audio output on (`modalities: ["text", "audio"]`). |
| `max_tokens_param` | auto | `"max_completion_tokens"` for OpenAI and Azure OpenAI, `"max_tokens"` elsewhere. |
| `developer_role` | auto | Role used for `developer` messages: `"developer"` for OpenAI, `"system"` elsewhere. |
| `include_usage` | `True` | Sends `stream_options={"include_usage": true}` to get token usage. Turn it off for servers that reject it. |
| `strip_thinking` | `True` | Drops a leading `<think>…</think>` block from the text (see below). |
| `timeout` | `60.0` | Seconds for connecting and for each read, i.e. the longest wait for the next streamed chunk. |
| `max_retries` | `1` | SDK retries for connection errors and 408/409/429/5xx, before the stream starts. The SDK obeys `retry-after`, which can be long: use `0` behind a failover chain. |
| `keepalive_expiry` | `120.0` | Seconds an idle connection stays pooled. httpx's default of 5 s would make most turns pay a new TLS handshake. |
| `client` | — | A pre-built `openai.AsyncOpenAI`-compatible client, e.g. `openai.AsyncAzureOpenAI`. The connection options above are then ignored and the client is not closed by `aclose()`. |
| `http_client` | — | An `httpx.AsyncClient` (or `httpx2.AsyncClient` with openai 3.x) for proxies or custom transports. Not closed by `aclose()`. |

### How the conversation is sent

`ChatContext` items map to Chat Completions messages as follows
(`providers/openai/_format.py`, `to_chat_messages()`):

* **System and developer messages** keep their position and role (`developer` becomes
  `system` for servers other than OpenAI).
* **User messages** are sent as a string, or as content parts when they carry media:
  `image_url` for `ImageContent` (`https://` or `data:` URLs) and `input_audio` (base64
  WAV by default, see `audio_format`) for `AudioContent`. Audio is sent only when
  `capabilities.audio_input` is set; otherwise its transcript is used, and audio without
  a transcript is dropped. With `audio_history=N`, clips older than the last N are sent
  as their transcripts when they have one.
* **Tool calls**: consecutive assistant text and `FunctionCall` items become one assistant
  message with `tool_calls`, and each `FunctionCallOutput` is placed right after it as a
  `tool` message, even if the user spoke while the tool was running. Calls without an
  output and outputs without a call are dropped, because the API rejects the request
  otherwise.
* **Interrupted assistant messages** are sent as heard: the session has already truncated
  them. Empty messages are skipped.

Tools are sent as `{"type": "function", "function": {name, description, parameters}}`,
with `"strict": true` for `FunctionTool(strict=True)`. `tool_choice` accepts `"auto"`,
`"required"`, `"none"` or a tool name. `tool_choice` and `parallel_tool_calls` are
dropped from requests without tools, which the API would reject.

### What the stream yields

* **Text** deltas as they arrive. A refusal (`delta.refusal`) is spoken like text.
  Reasoning deltas (`reasoning` / `reasoning_content`, from reasoning models on
  compatible servers) are never spoken. A leading `<think>…</think>` block in the text,
  produced by reasoning models served without a reasoning parser, is removed.
* **Tool calls**: argument fragments are accumulated per `index`, and complete
  `FunctionCall`s (the server's `call_id`, JSON `arguments`, `"{}"` for tools without
  arguments) are emitted once, when the choice finishes.
* **Usage and finish reason** in the last chunk. `LLMMetrics` reports `prompt_tokens`,
  `completion_tokens` and `cached_tokens` (from `prompt_tokens_details.cached_tokens`,
  or DeepSeek's `prompt_cache_hit_tokens`). `ttft` is the time to the first text delta,
  or to the first tool-call fragment for tool-only responses.

### Errors

| Failure | Raised |
|---|---|
| 401, 403 | `AuthenticationError` |
| 429 | `RateLimitError` (`retryable=False` for `insufficient_quota`) |
| 404 | `ProviderError(status_code=404)` with a hint (e.g. `ollama pull <model>`) |
| 408, timeouts | `ProviderTimeoutError` |
| connection failures | `ProviderConnectionError` |
| 5xx, 409 | `ProviderError(retryable=True)` |
| other 4xx | `ProviderError(retryable=False)` |
| error event inside the stream | `ProviderError` (`RateLimitError` for rate limits); text received before it is delivered first |

Messages include the provider, model, HTTP status and the server's error message. The
cascade reports a failed response as an `EngineErrorEvent` and keeps the session running.

### Latency notes

* **`warmup()`** opens the HTTP connection (`GET /models`) so the first turn skips the
  TCP and TLS handshakes; `CascadeEngine.warmup()` calls it. Idle connections are kept
  for `keepalive_expiry` (120 s) so the next turns reuse them. Failures are logged, not
  raised.
* **Prompt caching** is automatic on OpenAI for prompts of 1,024 tokens or more. The
  instructions and tools come first in every request, so keep them stable. `extra={"prompt_cache_key": ...}`
  improves hit rates across sessions. Cache hits appear in `LLMMetrics.cached_tokens`.
* **Reasoning** delays the first token: prefer non-reasoning models, or
  `reasoning_effort="none"`, for conversational turns.

### Azure OpenAI and custom clients

Pass a configured SDK client; the model is your deployment name:

```python
import os

import openai
from voice_agent_next.providers.openai.llm import OpenAILLM

llm = OpenAILLM(
    model="my-gpt-4-1-mini",
    client=openai.AsyncAzureOpenAI(
        azure_endpoint="https://my-resource.openai.azure.com",
        api_version="2025-04-01-preview",
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    ),
)
```

### Audio input

Chat models that accept audio (`gpt-audio*`, `gpt-4o-audio*`, and on other servers
Qwen-Omni, Ultravox, Voxtral, Gemma audio...) receive the user's audio directly: user
`AudioContent` is sent as `input_audio` parts. `audio_input` is set from a table of known
model ids, or explicitly with `audio_input=True`. With `stt=None` the cascade then runs as
a half-cascade. `await llm.transcribe(frame)` asks the model for a verbatim transcript
(text only, not in the metrics), which `CascadeOptions(input_transcriber="llm")` uses for
the history. Hosts, formats and options: [audio-input half-cascades](openai-compatible.md#audio-input-half-cascades).

### Audio output (`gpt-audio`)

Audio-output models (`gpt-audio*`, `gpt-4o-audio*`) speak for themselves. The LLM then
requests `modalities: ["text", "audio"]` and `audio: {"voice": ..., "format": "pcm16"}`
(`voice=` on the LLM, default `alloy`), and declares
`LLMCapabilities(audio_input=True, audio_output=True)`. The streamed `delta.audio.data`
(pcm16, 24 kHz) arrives as `ChatChunk.audio`, and `delta.audio.transcript` as the text.
Without a TTS, the cascade plays that voice directly: no STT and no TTS
(see [omni models](../concepts/omni-models.md)).

```python
session = AgentSession(
    llm={"provider": "openai", "model": "gpt-audio", "voice": "marin"},
    vad="silero",
    turn_detector="smart_turn",
)
```

Any OpenAI-compatible server that streams audio the same way (for example vLLM-Omni
serving Qwen-Omni) works with `extra={"modalities": ["text", "audio"]}`.

## Speech-to-text

`stt="openai/gpt-live-transcribe"` streams the user's audio into a Realtime
**transcription session**: a WebSocket to `wss://api.openai.com/v1/realtime?intent=transcription`
configured with `session.update` (`type: "transcription"`). Audio is resampled to 24 kHz
PCM16 and appended in 50 ms chunks (`input_audio_buffer.append`).

Server-side turn detection is **off** by default. The cascade's VAD decides when the user
paused, and the cascade's `flush()` commits the buffered audio (`input_audio_buffer.commit`),
which makes the model finalize it. This is the flow OpenAI documents for
`gpt-live-transcribe`, which has no server VAD. A cascade using this STT therefore needs
`vad=...`.

```python
from voice_agent_next import create

stt = create("stt", "openai/gpt-live-transcribe", language="en", delay="low")
stt = create("stt", "openai/gpt-transcribe", languages=["en", "fr"], keywords=["AC-42"])
```

### Models

| Model | Streaming (realtime) | File (`transcribe()`) | Notes |
|---|---|---|---|
| `gpt-live-transcribe` (default) | ✓ deltas while the user speaks | via a short realtime session | $0.017/min of streamed audio. `languages`, `keywords`, `prompt`, `delay`. No server VAD, no timestamps or confidence. |
| `gpt-transcribe` | ✓ transcribes each committed turn | ✓ (`stream=true`) | $0.0045/min. Detects the language (`languages: [{"code": ...}]`), uses earlier turns as context. `languages`, `keywords`, `prompt`. |
| `gpt-realtime-whisper` | ✓ deltas | via a short realtime session | $0.017/min. `delay`; no `prompt`, no server VAD. |
| `gpt-4o-transcribe`, `gpt-4o-mini-transcribe` | ✓ | ✓ (`stream=true`) | Single `language`, `prompt`, `logprobs`, server VAD. **Removal announced for 2027-02-26.** |
| `whisper-1` | ✓ | ✓ | Single `language`; word timestamps with `word_timestamps=True`. **Removal announced for 2027-02-26.** |

Prices and deprecations are OpenAI's as of 2026-09-24. `gpt-live-transcribe` and
`gpt-realtime-whisper` are served only by realtime sessions: `realtime=False` is rejected
for them.

### Options

| Option | Default | Meaning |
|---|---|---|
| `model` | `gpt-live-transcribe` | See above. |
| `api_key`, `base_url` | env | As for the LLM: `OPENAI_API_KEY`, `OPENAI_BASE_URL`; the OpenAI key is never sent to another `base_url`. `wss://` is derived for the realtime session; `?model=` is added to the URL for hosts other than OpenAI (gateways route on it). |
| `language` | — | Expected language (`en`, `en-US` is sent as `en`). `Agent(language=...)` wins. |
| `languages` | — | Expected languages for code-switched audio (`gpt-live-transcribe`, `gpt-transcribe`): ISO 639-1, some ISO 639-3 (`yue`, `cmn`) and `zh-cn`/`zh-tw`/`zh-hk`. Other models accept one language. |
| `prompt` | — | Free-form context ("A support call about billing."). |
| `keywords` | — | Literal terms that may be spoken (`gpt-live-transcribe`, `gpt-transcribe`); single-line, no `<`/`>`. |
| `delay` | server default | `minimal`, `low`, `medium`, `high`, `xhigh`: how long the streaming models wait for more audio before emitting text. Lower = earlier partials, higher = better accuracy. |
| `noise_reduction` | off | `near_field` (headsets) or `far_field` (laptop and room microphones). |
| `turn_detection` | `None` | `None`: flushes commit. `"server_vad"`, `"semantic_vad"` or a raw object (e.g. `{"type": "server_vad", "silence_duration_ms": 300}`) let the server segment the audio (not with `gpt-live-transcribe`/`gpt-realtime-whisper`). |
| `logprobs` | `False` | Token log probabilities (`gpt-4o-*-transcribe`); they become `Transcript.confidence` (geometric-mean token probability). |
| `realtime` | `True` | `False` makes the STT batch-only (`/audio/transcriptions`): the cascade then segments the audio with its VAD (`StreamAdapter`). |
| `http_streaming` | auto | `stream=true` on `/audio/transcriptions` (default for the models that support it, on OpenAI). |
| `word_timestamps` | `False` | Batch only: `verbose_json` with word timings (`whisper-1` and compatible servers). |
| `temperature` | — | Batch only. |
| `session`, `extra`, `query`, `headers` | — | Extra `session.update` fields (deep-merged), extra form fields for `/audio/transcriptions`, extra realtime URL parameters, extra headers. |
| `chunk_ms` | `50` | Size of the appended audio chunks. |
| `connect_timeout`, `timeout` | 10 s, 30 s | Connection and session setup; HTTP read timeout. |
| `close_timeout` | 5 s | After `end_input()`, how long to wait for the last transcripts. |
| `max_reconnect_attempts`, `reconnect_backoff` | 3, 0.5 s | See *Errors and reconnects*. |
| `replay_buffer` | 30 s | Uncommitted audio kept to re-send after a reconnect. |

### Events

| Server event | `STTEvent` |
|---|---|
| `conversation.item.input_audio_transcription.delta` | `INTERIM_TRANSCRIPT` with the item's text so far (`segment_id` = item id) |
| `conversation.item.input_audio_transcription.completed` | `FINAL_TRANSCRIPT`, in commit order; `language` from `languages` (`gpt-transcribe`), `confidence` from `logprobs` |
| `conversation.item.input_audio_transcription.failed` | `FINAL_TRANSCRIPT` with the partial text (the failure is logged) |
| `input_audio_buffer.speech_started` / `speech_stopped` | `START_OF_SPEECH` / `END_OF_SPEECH` (server VAD only) |
| `semantic_vad` commit | `FINAL_TRANSCRIPT` then `END_OF_TURN` (`capabilities.end_of_turn`) |

Every flush is answered by at least one `FINAL_TRANSCRIPT`, which is what the cascade's
endpointing waits for: the transcript of the committed audio or, when there was nothing to
commit, an empty final right away. The API rejects commits of less than 100 ms of audio, so
a shorter tail stays in the buffer for the next commit (`end_input()` pads it instead). A
flush while an earlier commit is still being transcribed is answered by that commit's final.
Completions of different commits may arrive out of order: finals are emitted in commit
order.

With server turn detection the server commits by itself, flushes only answer (the last one,
from `end_input()`, commits speech still in progress), and the cascade can run without a
VAD: `START_OF_SPEECH` drives barge-in and `END_OF_SPEECH` the endpointing.

### Batch transcription

`stt.transcribe(audio)` posts a WAV file to `/audio/transcriptions` (`gpt-transcribe`,
`gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `whisper-1`), with `stream=true` where the
model supports it, so long files never hit the read timeout. The streaming-only models
transcribe through a short realtime session instead.

### Errors and reconnects

| Failure | Result |
|---|---|
| handshake 401/403, `invalid_api_key` error event | `AuthenticationError`, the stream fails |
| handshake 429 | `RateLimitError`; an `insufficient_quota` error event fails the stream |
| rejected `session.update` (unknown model, bad language code, invalid keyword) | `ProviderError` with the server's message, when the stream starts |
| connection refused, handshake 5xx | `ProviderConnectionError` (retryable) |
| dropped connection, `session_expired` | reconnect (see below) |
| other server errors | logged; the stream goes on |
| batch: HTTP errors, network failures | as for the LLM (401/403, 429, 404 with a hint, 408/504 timeouts, 5xx retryable) |

After a dropped connection or an expired session (Realtime sessions last at most 60 minutes) the
stream reconnects with exponential backoff, configures the new session, **re-sends the
audio of commits that were not transcribed yet and commits them again**, then re-sends the
uncommitted audio (up to `replay_buffer` seconds): no utterance is lost. More than
`max_reconnect_attempts` reconnects per minute end the stream with
`ProviderConnectionError`, which the cascade reports as an `EngineErrorEvent`.

### Latency and cost notes

* Independent measurements put OpenAI's realtime transcription at 0.69–0.81 s from the end
  of speech to the final transcript
  ([research note 03](../research/03-stt-tts-llm-landscape.md)), slower than Deepgram,
  Soniox or Cartesia. The cascade waits `CascadeOptions.final_transcript_timeout` (1.0 s)
  for the final after flushing, then uses the interim text: with `gpt-live-transcribe` the
  deltas usually already cover the utterance. Raise the timeout (e.g. 1.5 s) if finals often
  arrive late.
* `delay="minimal"`/`"low"` gives earlier partial text on the streaming models.
* Streaming models are billed per minute of streamed audio, silence included: the
  cascade streams audio continuously, also while the agent speaks.
* A long pause mid-sentence makes the VAD flush, which commits the first half as its own
  item; `gpt-transcribe` uses earlier turns as context, which limits the accuracy cost.

## Text-to-speech

`tts="openai/gpt-4o-mini-tts"` posts each text to `/audio/speech` with
`response_format="pcm"` and streams the chunked response body (24 kHz s16le mono) as it
arrives. The endpoint takes complete text, so `tts.stream()` synthesizes sentence by
sentence (`SentenceStreamAdapter`: the next sentence is requested while the current one
plays), which gives exact text/audio alignment for truncation on barge-in.

```python
tts = create(
    "tts",
    "openai/gpt-4o-mini-tts",
    voice="cedar",
    instructions="Speak in a calm, friendly tone, a little faster than usual.",
)
tts = create("tts", "openai/tts-1", voice="nova", speed=1.1)
```

### Models and voices

| Model | Notes |
|---|---|
| `gpt-4o-mini-tts` (default; snapshot `gpt-4o-mini-tts-2025-12-15`) | `instructions` steer accent, emotion, intonation, speed, tone, whispering. $0.60 per 1M text tokens in, $12 per 1M audio tokens out. |
| `tts-1`, `tts-1-hd` | Lower latency / higher quality; no `instructions`, fewer voices. $15 / $30 per 1M characters. |

Voices: `alloy`, `ash`, `ballad`, `coral`, `echo`, `fable`, `nova`, `onyx`, `sage`,
`shimmer`, `verse`, `marin`, `cedar`. The default is `marin` (OpenAI recommends `marin` or
`cedar` for quality); the `tts-1` models support `alloy`, `ash`, `coral`, `echo`, `fable`,
`nova`, `onyx`, `sage` and `shimmer`, and default to `alloy`. Custom voices are passed by
id (`voice="voice_1234"` is sent as `{"id": "voice_1234"}`).

### Options

| Option | Default | Meaning |
|---|---|---|
| `model`, `voice` | see above | `Agent(voice=...)` wins over the constructor voice. |
| `api_key`, `base_url` | env | As for the LLM. |
| `instructions` | — | Voice steering (GPT-4o models only). |
| `speed` | `1.0` | 0.25–4.0. |
| `sample_rate` | `24000` | Output rate of the frames; other rates are resampled locally. |
| `response_format` | `pcm` | `pcm` (fastest) or `wav`; WAV headers are parsed and stripped. |
| `extra` | — | Extra JSON body fields (`None` removes a default one). |
| `timeout`, `connect_timeout` | 30 s, 10 s | Read timeout (the longest wait for the next audio chunk), connection timeout. |
| `max_retries` | `1` | Retries of a request that failed **before any audio arrived** (connection errors, timeouts, 429, 5xx). A request that fails mid-audio is not retried. |
| `keepalive_expiry` | 120 s | Idle connections stay open between turns, so the first sentence of a turn does not pay a TLS handshake. |
| `http_client` | — | An `httpx.AsyncClient` (not closed by `aclose()`). |

Texts longer than the API's 4,096-character limit are split at sentence boundaries into
several requests. Errors are mapped like the LLM's (401/403 `AuthenticationError`, 429
`RateLimitError`, 404 with a hint, network failures `ProviderConnectionError`).

### Latency notes

* `pcm` avoids decoding; OpenAI recommends `pcm` or `wav` for the fastest responses.
* `warmup()` (called by `CascadeEngine.warmup()`) opens the HTTP connection with
  `GET /models`, so the first sentence skips the TCP and TLS handshakes.
* The cascade speaks the first clause early (`CascadeOptions.first_sentence_max_chars`):
  with a request per sentence, a short first sentence is the fastest first audio.
* No word timestamps: truncation on barge-in uses the per-sentence alignment.

## Compatible speech servers

The STT and TTS classes are also the clients of OpenAI-compatible speech servers. Each
host is a subclass with its own defaults; every option above applies.

| Spec | Class | Default base URL | API key | Default model / voice |
|---|---|---|---|---|
| `stt="speaches"` | `SpeachesSTT` | `http://localhost:8000/v1` (`SPEACHES_BASE_URL`) | optional, `SPEACHES_API_KEY` | `Systran/faster-distil-whisper-small.en` |
| `tts="speaches"` | `SpeachesTTS` | same | same | `speaches-ai/Kokoro-82M-v1.0-ONNX`, `af_heart` |
| `stt="localai"` | `LocalAISTT` | `http://localhost:8080/v1` (`LOCALAI_BASE_URL`) | optional, `LOCALAI_API_KEY` | `whisper-1` |
| `tts="localai"` | `LocalAITTS` | same | same | `tts-1`, the model's voice |
| `stt="azure_openai/<deployment>"` | `AzureOpenAISTT` | `https://<resource>.openai.azure.com/openai/v1` (`AZURE_OPENAI_ENDPOINT`) | `AZURE_OPENAI_API_KEY` (`api-key` header) or an Entra ID token (`azure_ad_token=` / `AZURE_OPENAI_AD_TOKEN`) | `gpt-4o-mini-transcribe` |
| `tts="azure_openai/<deployment>"` | `AzureOpenAITTS` | same | same | `gpt-4o-mini-tts`, `alloy` |
| `tts="kokoro_fastapi"` (or `kokoro-fastapi/kokoro`) | `KokoroFastAPITTS` | `http://localhost:8880/v1` (`KOKORO_FASTAPI_BASE_URL`) | optional, `KOKORO_FASTAPI_API_KEY` | `kokoro`, `af_heart` |

Classes live in `voice_agent_next.providers.<spec>`; the base URL variables may hold the
`ws://` URL shared with the realtime engine.

* **Speaches** and **LocalAI** STT are batch-only (`realtime=False` by default): the cascade
  segments the audio with its VAD and posts each utterance to `/v1/audio/transcriptions`.
  Their realtime endpoints serve voice pipelines (see [OpenAI Realtime](openai-realtime.md)),
  and Speaches' transcription sessions cannot turn server VAD off. Both servers load
  models on first use, so `warmup()` sends one short request (0.5 s of silence, or "Hi.").
  The default Speaches model is English-only; use `Systran/faster-whisper-small` or larger
  for other languages.
* **Speaches TTS** asks for raw PCM at `sample_rate` (the server resamples, so Piper
  voices such as `speaches-ai/piper-en_US-amy-medium` work too). **LocalAI TTS** answers with
  WAV whatever the requested format, at the backend's rate (Piper: 22.05 kHz); the header is
  parsed and the audio resampled.
* **Azure OpenAI**: the model is your deployment name; name deployments after the model
  (`gpt-4o-mini-transcribe`...) so that model-specific options are sent correctly.
  Realtime transcription connects to
  `wss://<resource>.openai.azure.com/openai/v1/realtime?model=<deployment>&intent=transcription`
  (GA protocol); `realtime=False` uses `/openai/v1/audio/transcriptions`.
* **Kokoro-FastAPI** (`docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:latest`):
  voices are Kokoro voice names or mixes (`af_bella(2)+af_sky(1)`). Unlike the in-process
  `kokoro` provider, the model runs in the server.
* Any other server: `OpenAISTT(base_url=..., realtime=False)` / `OpenAITTS(base_url=...)`, or
  subclass `OpenAICompatibleSTT` / `OpenAICompatibleTTS` like the hosts above.

## Testing

* `tests/test_openai_stt.py` runs the STT against `tests/fake_transcription_server.py`, a
  scripted transcription server that replays OpenAI's GA events (live deltas for the
  streaming models, commits, out-of-order completions, failures, server VAD, dropped
  connections, expired sessions), and the batch endpoint against `httpx.MockTransport`.
  `tests/test_openai_tts.py` covers the TTS and the compatible hosts the same way.
* `uv run pytest -m integration tests/test_openai_stt.py tests/test_openai_tts.py` talks to
  the real API when `OPENAI_API_KEY` is set.

## References

OpenAI documentation, read on 2026-09-24:
[Realtime transcription](https://developers.openai.com/api/docs/guides/realtime-transcription),
[File transcription](https://developers.openai.com/api/docs/guides/speech-to-text),
[Transcription overview](https://developers.openai.com/api/docs/guides/transcription),
[Voice activity detection](https://developers.openai.com/api/docs/guides/realtime-vad),
[Text to speech](https://developers.openai.com/api/docs/guides/text-to-speech),
[Realtime client events](https://developers.openai.com/api/reference/resources/realtime/client-events),
[Realtime server events](https://developers.openai.com/api/reference/resources/realtime/server-events),
[Create speech](https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create),
[Create transcription](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create),
[Transcription streaming events](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/streaming-events),
[Deprecations](https://developers.openai.com/api/docs/deprecations) and the model pages of
[gpt-live-transcribe](https://developers.openai.com/api/docs/models/gpt-live-transcribe),
[gpt-transcribe](https://developers.openai.com/api/docs/models/gpt-transcribe),
[gpt-realtime-whisper](https://developers.openai.com/api/docs/models/gpt-realtime-whisper) and
[gpt-4o-mini-tts](https://developers.openai.com/api/docs/models/gpt-4o-mini-tts).
