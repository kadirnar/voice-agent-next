# OpenAI-compatible LLM servers

Local servers (Ollama, llama.cpp, vLLM, LM Studio, mlx-lm) and fast hosted APIs (Groq,
Cerebras, Together, OpenRouter, DeepSeek, Fireworks, SambaNova) all speak OpenAI's Chat
Completions protocol. Each has a small preconfigured provider, a subclass of `OpenAICompatibleLLM`,
which is itself an [`OpenAILLM`](openai.md). Streaming, tool calls, usage, error mapping and
every option documented there apply to all of them.

```bash
pip install 'voice-agent-next[openai]'   # or: uv sync --extra openai
```

```python
from voice_agent_next import AgentSession

# fully local cascade
session = AgentSession(stt="faster-whisper", llm="ollama/qwen3.5:4b", tts="kokoro", vad="silero")
# fast cloud LLM
session = AgentSession(
    stt="deepgram/nova-3", llm="groq/llama-3.3-70b-versatile", tts="cartesia", vad="silero"
)
```

## Providers

| Spec | Class | Default base URL | API key | Default model |
|---|---|---|---|---|
| `ollama` | `OllamaLLM` | `http://127.0.0.1:11434/v1` | optional, `OLLAMA_API_KEY` | `qwen3.5:4b` |
| `llamacpp` | `LlamaCppLLM` | `http://127.0.0.1:8080/v1` | optional, `LLAMA_API_KEY` | the loaded model |
| `vllm` | `VllmLLM` | `http://127.0.0.1:8000/v1` | optional, `VLLM_API_KEY` | the served model |
| `vllm_omni` | `VllmOmniLLM` | `http://127.0.0.1:8091/v1` | optional, `VLLM_API_KEY` | the served model |
| `dashscope` | `DashScopeLLM` | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` | `qwen3.5-omni-flash` |
| `lmstudio` | `LMStudioLLM` | `http://127.0.0.1:1234/v1` | optional, `LM_API_TOKEN` | the first listed LLM |
| `mlx_lm` | `MLXLMServerLLM` | `http://127.0.0.1:8080/v1` | none | the server's `--model` |
| `groq` | `GroqLLM` | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| `cerebras` | `CerebrasLLM` | `https://api.cerebras.ai/v1` | `CEREBRAS_API_KEY` | `gpt-oss-120b` |
| `together` | `TogetherLLM` | `https://api.together.ai/v1` | `TOGETHER_API_KEY` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| `openrouter` | `OpenRouterLLM` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | `openai/gpt-4.1-mini` |
| `deepseek` | `DeepSeekLLM` | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` | `deepseek-flash` |
| `fireworks` | `FireworksLLM` | `https://api.fireworks.ai/inference/v1` | `FIREWORKS_API_KEY` | `accounts/fireworks/models/llama-v3p3-70b-instruct` |
| `sambanova` | `SambaNovaLLM` | `https://api.sambanova.ai/v1` | `SAMBANOVA_API_KEY` | `Meta-Llama-3.3-70B-Instruct` |

Classes live in `voice_agent_next.providers.<spec>`.

* **Model**: everything after the first `/` of the spec is the model id, so ids may
  contain `/` and `:` (`ollama/LiquidAI/lfm2.5-1.2b-instruct:latest`,
  `together/meta-llama/Llama-3.3-70B-Instruct-Turbo`). llama.cpp, vLLM and LM Studio have
  no default: the provider asks the server (`GET /v1/models`) and uses the first model it
  lists, skipping embedding models. Pass the model to choose one.
* **Base URL**: `base_url=`, else the `<SPEC>_BASE_URL` environment variable
  (`OLLAMA_BASE_URL`, `LLAMACPP_BASE_URL`, `VLLM_BASE_URL`, `LMSTUDIO_BASE_URL`,
  `GROQ_BASE_URL`...), else the default above. Ollama also honours its own `OLLAMA_HOST`.
* **API key**: `api_key=`, else the variable in the table, also when `base_url` points to
  a proxy. Hosted providers raise `ConfigurationError` without a key. Local servers work
  without one; their variables are the ones the servers themselves read, so a key set
  for the server is picked up by the client too.

Differences from `openai`: `max_tokens` is sent as `max_tokens`, and `developer`
messages are sent with the `system` role, which chat templates understand.

**System messages and strict chat templates.** Many chat templates (Qwen, Gemma, … as
rendered by llama.cpp `--jinja`, vLLM and LM Studio) reject a system message that is not
the first message, which is what per-response instructions (`generate_reply(instructions=...)`)
produce. `system_message_policy` controls this:

| policy | effect | default for |
|---|---|---|
| `"keep"` | system messages stay where they are | OpenAI, hosted APIs, Ollama |
| `"merge"` | all system messages are joined into one leading system message | `llamacpp`, `vllm`, `lmstudio` |
| `"as_user"` | the leading system prompt stays; later system messages are sent as user messages (adjacent user messages are joined, for templates that require alternating roles) | — |

## Local servers

Start the model before the first turn with `await llm.warmup()`, or
`await session.engine.warmup()` for a whole cascade. It opens the connection and, for
Ollama and LM Studio, which load models on demand, runs a 1-token completion so the model
is in memory. Measured on an RTX 5070 Ti with `LiquidAI/lfm2.5-1.2b-instruct` (Q4_K_M,
Ollama 0.32): **582 ms** time to first token when the first request has to load the
model, **9.8 ms** after `warmup()`, and 8.8 ms p50 over 20 warm requests.

### Ollama

```bash
ollama pull qwen3.5:4b
```

`llm="ollama/qwen3.5:4b"`. `OLLAMA_HOST` is read like Ollama reads it (`host`,
`host:port` or a URL). Wildcard bind addresses such as `0.0.0.0:11434` are turned into
`127.0.0.1`, which every OS can connect to. A missing model fails with
`ProviderError(status_code=404)` and a `ollama pull <model>` hint. Set
`OLLAMA_BASE_URL=https://ollama.com/v1` and `OLLAMA_API_KEY` for Ollama's cloud models.

Ollama's OpenAI API does not support `tool_choice`, accepts only base64 `data:` image
URLs and has no audio input.

### llama.cpp

```bash
llama-server -m model.gguf --jinja --port 8080
```

`llm="llamacpp"`. Tool calling needs `--jinja`. With `--api-key`, set `LLAMA_API_KEY`
(the variable `llama-server` itself reads) or pass `api_key=`. Reasoning from
`--reasoning-format` arrives in `reasoning_content` and is never spoken.

Audio models (libmtmd) take the user's audio directly, for a [half-cascade](#audio-input-half-cascades):

```bash
llama-server -hf ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF --port 8080   # or Voxtral Mini,
# Qwen2.5-Omni, Gemma 4 E2B/E4B, or -m LFM2.5-Audio-1.5B-Q4_0.gguf --mmproj mmproj-...gguf
```

`llm={"provider": "llamacpp", "audio_input": True}`. The model is discovered from the
server, so `audio_input=True` has to be explicit (a model id from the known-model table
turns it on by itself). `warmup()` reads `GET /props` and warns when the loaded model has
no audio support. Audio is sent as 16 kHz WAV.

### vLLM

```bash
vllm serve Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes
```

`llm="vllm"` or `llm="vllm/Qwen/Qwen3-8B"` (the served model name). vLLM's server runs on
Linux; the client works anywhere. Pass template options through `extra`, e.g.
`extra={"chat_template_kwargs": {"enable_thinking": False}}` to turn off Qwen3 thinking.
With `--api-key`, set `VLLM_API_KEY`.

vLLM also serves audio-in / text-out models (Qwen2-Audio, the thinker of Qwen2.5/3-Omni,
Ultravox, Voxtral, Gemma 3n, Phi-4-multimodal): `llm="vllm/Qwen/Qwen2.5-Omni-7B"` turns
`audio_input` on from the model id. Audio is sent as 16 kHz WAV.

### vLLM-Omni

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091
```

`llm="vllm_omni"` (or `vllm_omni/<served model>`) has `audio_input` on and asks for text
only (`modalities: ["text"]`): the cascade's TTS speaks the reply. Pass `voice=...` or
`extra={"modalities": ["text", "audio"]}` to hear the model's own voice instead
(experimental: the stream must carry pcm16 `delta.audio` like OpenAI's). Address:
`base_url=`, else `VLLM_OMNI_BASE_URL`; key: `VLLM_API_KEY`. The same server's Realtime
WebSocket is the [`vllm_realtime` engine](openai-realtime.md).

### DashScope (Qwen-Omni)

Alibaba Model Studio's OpenAI-compatible mode. `llm="dashscope/qwen3.5-omni-flash"`
(default), `qwen3.5-omni-plus`, `qwen3.8-omni-flash` (text output only, tool calling),
`qwen3-omni-flash`. Set `DASHSCOPE_API_KEY`; the endpoint is `base_url=`, else
`DASHSCOPE_BASE_URL`, else the workspace endpoint when `DASHSCOPE_WORKSPACE_ID` is set,
else the regional one (`DASHSCOPE_REGION`: `ap-southeast-1` default, `cn-beijing`,
`us-east-1`). Differences handled by the provider:

* requests are always streamed (the API refuses unstreamed calls to these models);
* audio goes as a `data:;base64,` URL (WAV, 16 kHz);
* system messages are merged into one leading message;
* spoken replies: `voice="Tina"` requests `modalities: ["text", "audio"]` with
  `audio: {"voice": "Tina", "format": "wav"}` (the only format accepted; the chunks are
  pcm16 at 24 kHz). Without a TTS the cascade plays that voice, see
  [omni models](../concepts/omni-models.md).

For the Realtime WebSocket use the [`qwen_omni` engine](openai-realtime.md).

## Audio-input half-cascades

With `stt=None`, the cascade sends the user's turn to the LLM as audio (`AudioContent`,
an `input_audio` part). The LLM hears tone and hesitation, and an STT round-trip
disappears. Every host above that serves an audio model works; the per-host format is
handled for you:

| host | audio sent as | notes |
|---|---|---|
| `openai` (`gpt-audio*`, `gpt-4o-audio*`) | WAV at the input rate | audio in and out |
| `vllm`, `vllm_omni` | WAV, 16 kHz | `vllm_omni`: `audio_input` on, text out by default |
| `llamacpp` | WAV, 16 kHz | `audio_input=True` for a discovered model; `/props` check |
| `dashscope` | WAV, 16 kHz, `data:;base64,` URL | always streamed |

* **Which models hear audio.** `audio_input=` on the LLM decides. Left unset, a table of
  known audio models decides from the model id (`gpt-audio`, `Qwen*-Omni`, `Qwen2-Audio`,
  `Ultravox`, `Voxtral`, `Gemma 3n/4 E2B-E4B`, `Phi-4-multimodal`, `MiniCPM-o`,
  `LFM2-Audio`, `Gemini`..., see `providers/openai/_models.py`).
* **Encoding.** `audio_format=AudioInputFormat(format="wav" | "pcm16", sample_rate=...,
  data_url=...)` (or a mapping in YAML) overrides the host's default, e.g. raw PCM for a
  server that takes it.
* **The user's words.** The model gets audio, so the history has no user text unless you
  ask for it: `CascadeOptions(input_transcriber="llm")` asks the same model for a
  transcript in a second, text-only request (`llm.transcribe()`), run alongside the reply;
  `input_transcriber="faster_whisper/tiny"` (any STT) transcribes locally instead. The
  transcript arrives as the turn's final `user_transcript` and fills the history.
* **Shorter requests.** Every earlier user turn is audio too, so each request grows.
  `audio_history=N` sends only the last N turns as audio and older ones as their
  transcripts (turns without a transcript stay audio).

```python
from voice_agent_next import AgentSession, CascadeOptions

session = AgentSession(
    llm={"provider": "llamacpp", "audio_input": True, "audio_history": 2},
    tts="kokoro",
    vad="silero",
    turn_detector="smart_turn",
    cascade_options=CascadeOptions(input_transcriber="llm"),
)  # no stt=...
```

Measured (T1, `latency-local-omni` scenario, RTX 5070 Ti, shared machine):
LFM2.5-Audio-1.5B Q4_0 on `llama-server` (CUDA, `-np 2`) + Kokoro (CPU) + Silero +
Smart Turn, 11 measured turns: see the table in [omni models](../concepts/omni-models.md#audio-input-half-cascade).

### LM Studio

Start the server in the app (Developer tab) or with `lms server start`, then use
`llm="lmstudio/<model key>"`. Models load on first use. If authentication is enabled in
the server settings, set `LM_API_TOKEN`.

### mlx-lm (Apple silicon)

`python -m mlx_lm.server --model mlx-community/Qwen3.5-4B-4bit`, then `llm="mlx_lm"` (the
server's model) or `llm="mlx_lm/<model>"` (loaded on demand). Thinking is off by default.
Set `MLX_LM_BASE_URL` for another address. See [MLX on Apple silicon](mlx.md#llm-mlx-lm-server-mlx_lm).

### Any other server

`openai` plus `base_url` works with any other compatible server: LocalAI, a LiteLLM proxy, other hosted APIs... The `OPENAI_API_KEY` is never sent to a custom
`base_url`, so pass `api_key=` when the server needs one:

```python
import os

from voice_agent_next import create

llm = create("llm", "openai/my-model", base_url="http://127.0.0.1:8080/v1")
llm = create(
    "llm",
    "openai/<model>",
    base_url="https://api.example.com/v1",
    api_key=os.environ["EXAMPLE_API_KEY"],
)
```

Turn off parameters a server rejects with `include_usage=False` (no `stream_options`) or
by setting them to `None` in `extra`.

## Hosted providers

* **Groq**: `llama-3.3-70b-versatile` is the default because it doesn't reason, so the
  first token comes quickly. `openai/gpt-oss-120b` was the fastest cloud configuration in
  the Pipecat voice benchmark (98 ms p50 time to first audio token, 86.3 % strict pass
  rate). `qwen/qwen3.8-27b` is also available.
* **Cerebras**: `gpt-oss-120b` runs at about 1,700 tokens/s. It is a reasoning model:
  `reasoning_effort="low"` keeps the first token fast. `qwen-3.8-27b` reasons at `high`
  effort by default.
* **DeepSeek**: the API has thinking on by default, which costs seconds before the first
  word, so this provider sends `thinking: {"type": "disabled"}`. Turn it back on with
  `extra={"thinking": {"type": "enabled"}}`. Cache hits (`prompt_cache_hit_tokens`) are
  reported as `cached_tokens`.
* **OpenRouter**: model ids are `vendor/model`. Attribution headers are optional:
  `headers={"HTTP-Referer": "https://your.app", "X-Title": "Your app"}`. Provider routing
  goes through `extra`, e.g. `extra={"provider": {"sort": "latency"}}`.
* **Fireworks** sends usage on the final chunk instead of a separate one; both are handled.
* **Together** and **SambaNova** need nothing special.

Voice benchmark figures and speeds are from [research note 03 §7](../research/03-stt-tts-llm-landscape.md).

## Reasoning models

For voice, reasoning is mostly latency: the model thinks before the first spoken word.
Whatever the server does with it, nothing but the answer is spoken:

* reasoning streamed in separate fields (`reasoning` on Ollama, `reasoning_content` on
  llama.cpp, vLLM and DeepSeek) is ignored;
* a leading `<think>…</think>` block in the text is removed (`strip_thinking=True`). This
  covers servers without a reasoning parser, and models that always think even when
  thinking is disabled.

To make the first token faster, turn reasoning off: `reasoning_effort="none"` (Ollama,
OpenAI and some hosted APIs), `chat_template_kwargs` (vLLM) or a provider-specific field
(DeepSeek's `thinking`).

## Adding a host

```python
from voice_agent_next import register_provider
from voice_agent_next.providers.openai.llm import OpenAICompatibleLLM


@register_provider(
    "llm",
    "myhost",
    default_model="my-model",
    env=("MYHOST_API_KEY",),
    extra="openai",
    requires=("openai",),
)
class MyHostLLM(OpenAICompatibleLLM):
    provider = "myhost"
    DEFAULT_MODEL = "my-model"  # None: ask the server
    DEFAULT_BASE_URL = "https://api.myhost.example/v1"
    BASE_URL_ENV = ("MYHOST_BASE_URL",)
    API_KEY_ENV = ("MYHOST_API_KEY",)
    API_KEY_REQUIRED = True  # False for local servers
    DEFAULT_EXTRA = {}  # request parameters sent by default
```

## Known limitations

* With `system_message_policy="keep"` on a strict chat template, per-response instructions
  fail the request; the local hosts that render model templates default to `"merge"`.
* One completion per request: `n` greater than 1 (via `extra`) is not supported.

## Speech servers

Speaches, LocalAI, Azure OpenAI and Kokoro-FastAPI also serve OpenAI's speech endpoints
(`/v1/audio/transcriptions`, `/v1/audio/speech`): `stt="speaches"`, `tts="speaches"`,
`stt="localai"`, `tts="localai"`, `stt="azure_openai/<deployment>"`,
`tts="azure_openai/<deployment>"`, `tts="kokoro_fastapi"`. See
[OpenAI: compatible speech servers](openai.md#compatible-speech-servers).
