# Google Gemini LLM and TTS (`llm: google`, `tts: google`)

`llm="google/gemini-3.8-flash"` and `tts="google/gemini-3.8-flash-tts"` run Gemini models in
a cascade through the official [`google-genai`](https://pypi.org/project/google-genai/) SDK.
Both are also available under the alias `gemini`. The Gemini Live speech-to-speech engine is
documented separately in [gemini-live.md](gemini-live.md).

* **LLM** (`providers/google/llm.py`): streaming `generateContent`. It streams text deltas,
  emits complete tool calls (parallel calls included), keeps thinking low by default (thoughts
  are never spoken), accepts audio and image input, and reports token usage, including
  cached and thinking tokens.
* **TTS** (`providers/google/tts.py`): streaming `generateContent` with audio output. It
  forwards 24 kHz PCM as it is generated and supports 30 prebuilt voices, designed or
  replicated voice ids, a delivery style, and an optional language code.

## Setup

```bash
pip install 'voice-agent-next[google]'   # or: uv sync --extra google  (google-genai >= 2.25)
export GOOGLE_API_KEY=...                # or GEMINI_API_KEY
```

**Gemini Developer API** (default). The key is taken from `api_key=`, then
`GOOGLE_API_KEY`, then `GEMINI_API_KEY`. It is sent in the `x-goog-api-key` header and is
never logged. If no key is found, the constructor raises `AuthenticationError`.

**Vertex AI.** Pass `vertexai=True` or set `GOOGLE_GENAI_USE_VERTEXAI=true`. Passing
`project`, `location` or `credentials` also selects Vertex AI.

```python
llm = create("llm", "google/gemini-3.8-flash", vertexai=True,
             project="my-project", location="global")        # Application Default Credentials
llm = create("llm", "google/gemini-3.8-flash", vertexai=True, api_key="...")  # express mode
```

`project` and `location` default to `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`, and
the SDK falls back to the ADC project and `global`. Credential errors from `google-auth` are
mapped to `AuthenticationError`.

## Usage

```python
from voice_agent_next import AgentSession, create

session = AgentSession(
    stt="deepgram/nova-3",
    llm="google/gemini-3.8-flash",
    tts="google/gemini-3.8-flash-tts",
)

# half-cascade: no STT, Gemini listens to the user's audio directly
session = AgentSession(llm="gemini/gemini-3.8-flash", tts="cartesia", vad="silero")

tts = create("tts", "gemini", voice="Puck", style="warm, relaxed, smiling")
```

```yaml
llm:
  provider: google/gemini-3.8-flash
  thinking_level: low
  extra_config: {top_p: 0.9}
tts:
  provider: google/gemini-3.8-flash-tts
  voice: Kore
  style: calm and friendly
```

## LLM

### Models

| Model | Notes for voice |
|---|---|
| `gemini-3.8-flash` (default) | Current stable Flash. It cannot go below `thinking_level="low"`, which is what `"auto"` selects. |
| `gemini-3.7-flash` | Same thinking levels as 3.8. |
| `gemini-3.6-flash`, `gemini-3.5-flash` | Support `"minimal"` thinking, which `"auto"` selects. |
| `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite` | Cheapest and fastest. Their default thinking level is already `minimal`. |
| `gemini-3.1-pro-preview` | Most capable, but too slow for most real-time turns (thinking cannot be disabled). |
| `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.5-pro` | Legacy models. `"auto"` sends `thinking_budget=0` (off) on Flash and Flash-Lite, and `128` on Pro (its minimum). |

The API accepts any model id, including tuned models and Gemma. The list above is only a
suggestion.

### Options

| Option | Default | Meaning |
|---|---|---|
| `model` | `gemini-3.8-flash` | Model id. |
| `api_key`, `vertexai`, `project`, `location`, `credentials` | env | Credentials (see Setup). |
| `thinking_level` | `"auto"` | `"auto"` picks the lowest-latency setting for the model. `"minimal"`, `"low"`, `"medium"` and `"high"` set it explicitly. `None` sends nothing and leaves the model default, which is `medium` or `high` on Gemini 3 and too slow for voice. |
| `thinking_budget` | `None` | Legacy token budget for Gemini 2.5 (`0` = off, `-1` = dynamic). It replaces `thinking_level`. |
| `temperature` | not sent | Google recommends leaving Gemini 3 at its default. |
| `max_tokens` | not sent | `max_output_tokens`. Thinking tokens count towards it. |
| `tool_choice` | `auto` | `"auto"`, `"required"` (`ANY`), `"none"` or a tool name (`ANY` restricted to that function). |
| `audio_input` | `True` | Send user `AudioContent` as WAV audio (half-cascade). `False` sends transcripts only. |
| `cached_content` | `None` | An explicit context cache (`cachedContents/...`) that holds the instructions and tools. They are then left out of each request. |
| `extra_config` | `{}` | Extra `GenerateContentConfig` fields in snake_case, e.g. `top_p`, `safety_settings`, `media_resolution` or `thinking_config: {include_thoughts: true}`. `tools` entries such as `[{"google_search": {}}]` are added next to the function declarations. The per-call `extra=` dict overrides these. |
| `fallback_thought_signature` | `"skip_thought_signature_validator"` | Signature sent with function calls that have none, such as history from another provider (see below). `None` disables it. |
| `timeout` | `30.0` | Seconds per request phase. It is also sent as the server-side deadline. |
| `max_retries` | `1` | SDK retries for connection errors and 408/429/5xx before the stream starts. |
| `keepalive_expiry` | `120.0` | Seconds an idle connection stays pooled, which saves a TLS handshake per turn. |
| `base_url`, `api_version`, `headers` | — | Endpoint overrides and extra HTTP headers. |
| `client` | — | A pre-built `google.genai.Client`. The connection options are then ignored, and `aclose()` does not close it. |
| `http_client` | — | An `httpx.AsyncClient` (proxies, custom transports, tests). `aclose()` does not close it. |

### How the conversation is sent

* **System and developer messages** become the `system_instruction`, joined with blank
  lines. Instructions placed before the conversation come first; later ones, such as
  per-response instructions, are appended.
* **`user` → `user`, `assistant` → `model`.** Consecutive items with the same role are merged.
  A conversation that starts with the model (a greeting) gets a short placeholder user turn
  in front, and one that ends with the model gets one at the end.
* **An interrupted assistant message** contributes exactly the text the user heard.
* **Tool calls** become `function_call` parts of the model turn. Their outputs become
  `function_response` parts (`{"output": ...}` or `{"error": ...}`) of the following user
  turn. Outputs that are JSON objects or arrays are sent as structured JSON. Calls without an
  output are dropped, and so are outputs without a call.
* **Call ids.** Gemini usually returns calls without an id, so the provider generates a
  `call_...` id for each one. When the API does return an id, it is kept and echoed back in
  the `function_response`.
* **Thought signatures.** Gemini 3 attaches an opaque `thought_signature` to the first
  function call of each step and checks it when the call is sent back within the same turn.
  The provider remembers each call's signature and puts it back on the right part. Calls it
  has no signature for get Google's documented placeholder
  [`skip_thought_signature_validator`](https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures).
* **Images** are sent as `inline_data` (for `data:` URLs) or `file_data` (for `https://` and
  `gs://` URLs). **User audio** is sent as `inline_data` WAV, or as its transcript when
  `audio_input=False`.
* **Thoughts** (parts with `thought: true`) are never emitted as text.

### Usage and errors

`ChatChunk.usage` is a `GeminiUsage` object:

* `prompt_tokens` includes cache hits and server-side tool results.
* `cached_tokens` counts implicit and explicit cache hits.
* `completion_tokens` includes the `thoughts_tokens`, which are billed as output.

A cancelled stream (barge-in) still reports the prompt tokens billed so far.

Finish reasons map as follows:

* `STOP` → `stop` (`tool_calls` when the response has calls)
* `MAX_TOKENS` → `length`
* `SAFETY`, `RECITATION`, blocked prompts and similar → `content_filter`

Errors are mapped to `voice_agent_next.errors`:

* 401, 403 and invalid or expired API keys (reported as 400 `API_KEY_INVALID`) →
  `AuthenticationError`
* 429 → `RateLimitError`
* 408 and 504 → `ProviderTimeoutError`
* 5xx and 409 → retryable `ProviderError`
* network failures → `ProviderConnectionError`

Error messages are truncated and never contain the key.

`warmup()` sends a free `models.get` for the model. This opens the HTTPS connection (and,
on Vertex AI, fetches the access token) before the first turn. It also reports a bad key or
model id as a logged warning.

## TTS

### Models and voices

| Model | Notes |
|---|---|
| `gemini-3.8-flash-tts` (default) | 130 languages; style via `speech_metadata`, inline tags such as `<sigh>`, `<short pause>`. |
| `gemini-3.8-flash-lite-tts` | Cheaper, 101 languages. |
| `gemini-3.1-flash-tts-preview`, `gemini-2.5-flash-preview-tts`, `gemini-2.5-pro-preview-tts` | Legacy previews. On the 2.5 models the style is prefixed to the text. |

All models output 24 kHz, mono, 16-bit PCM. Other rates are resampled, so `sample_rate` is
always 24000.

There are 30 prebuilt voices: Zephyr, Puck, Charon, Kore (the default), Fenrir, Leda,
Orus, Aoede, Callirrhoe, Autonoe, Enceladus, Iapetus, Umbriel, Algieba, Despina, Erinome,
Algenib, Rasalgethi, Laomedeia, Achernar, Alnilam, Schedar, Gacrux, Pulcherrima, Achird,
Zubenelgenubi, Vindemiatrix, Sadachbia, Sadaltager and Sulafat. Designed or replicated
voice ids (`voice_...`, `voicekey_...`) are passed through unchanged.

### Options

| Option | Default | Meaning |
|---|---|---|
| `voice` | `Kore` | A prebuilt voice name (case-insensitive) or a custom voice id. |
| `style` | `None` | Delivery instruction for every utterance, e.g. `"calm and friendly"` or `"fast-paced, excited"`. Keep the text itself a verbatim transcript, as Google recommends. |
| `language` | `None` | `speech_config.language_code`. By default the model detects the language. |
| `temperature` | not sent | Sampling temperature. |
| `extra_config` | `{}` | Extra `GenerateContentConfig` fields. A full `speech_config` replaces the voice settings, e.g. for `multi_speaker_voice_config` (at most 2 speakers). |
| `empty_retries` | `1` | Extra attempts when the model finishes without audio (TTS models occasionally return text or nothing). After that, a retryable `ProviderError` is raised. |
| `timeout`, `max_retries`, `keepalive_expiry`, credentials, `client`, `http_client` | | Same as for the LLM. |

### Streaming and latency

`generateContent` needs the complete text before synthesis starts; there is no incremental
text input. `GeminiTTS.stream()` therefore uses the `SentenceStreamAdapter`: every sentence
is one `streamGenerateContent` request whose audio is forwarded as it arrives, and the next
sentence is requested while the current one plays. For token-level text streaming with
word timestamps, use a WebSocket TTS such as Cartesia. For the lowest end-to-end latency
with Gemini voices, use the [Gemini Live engine](gemini-live.md).

Cloud Text-to-Speech (Chirp 3 HD voices) uses a different API and SDK
(`google-cloud-texttospeech`) and is not included here.

## Testing

```bash
uv sync --extra google
uv run pytest -q tests/providers/test_google_format.py tests/providers/test_google_llm.py \
    tests/providers/test_google_tts.py
GOOGLE_API_KEY=... uv run pytest -q -m integration tests/providers/test_google_llm.py \
    tests/providers/test_google_tts.py
```

The unit tests replay recorded `streamGenerateContent` SSE response shapes through
`httpx.MockTransport`, so the real SDK parses them. The integration tests call the real API.
`GEMINI_TEST_MODEL` and `GEMINI_TTS_TEST_MODEL` override the models they use.

## References (checked 2026-09-24)

* Models: <https://ai.google.dev/gemini-api/docs/models>
* Thinking levels and budgets: <https://ai.google.dev/gemini-api/docs/generate-content/thinking>
* Thought signatures: <https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures>
* Speech generation with `generateContent`: <https://ai.google.dev/gemini-api/docs/generate-content/speech-generation>
* Speech generation with the Interactions API: <https://ai.google.dev/gemini-api/docs/speech-generation>
