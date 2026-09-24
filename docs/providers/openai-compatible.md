# OpenAI-compatible LLM servers

Local servers (Ollama, llama.cpp, vLLM, LM Studio) and fast hosted APIs (Groq, Cerebras,
Together, OpenRouter, DeepSeek, Fireworks, SambaNova) all speak OpenAI's Chat Completions
protocol. Each has a small preconfigured provider, a subclass of `OpenAICompatibleLLM`,
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
| `lmstudio` | `LMStudioLLM` | `http://127.0.0.1:1234/v1` | optional, `LM_API_TOKEN` | the first listed LLM |
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

### vLLM

```bash
vllm serve Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes
```

`llm="vllm"` or `llm="vllm/Qwen/Qwen3-8B"` (the served model name). vLLM's server runs on
Linux; the client works anywhere. Pass template options through `extra`, e.g.
`extra={"chat_template_kwargs": {"enable_thinking": False}}` to turn off Qwen3 thinking.
With `--api-key`, set `VLLM_API_KEY`.

### LM Studio

Start the server in the app (Developer tab) or with `lms server start`, then use
`llm="lmstudio/<model key>"`. Models load on first use. If authentication is enabled in
the server settings, set `LM_API_TOKEN`.

### Any other server

`openai` plus `base_url` works with any other compatible server: `mlx_lm.server`,
LocalAI, a LiteLLM proxy, other hosted APIs... The `OPENAI_API_KEY` is never sent to a custom
`base_url`, so pass `api_key=` when the server needs one:

```python
import os

from voice_agent_next import create

llm = create("llm", "openai/mlx-community/Qwen3-4B-4bit", base_url="http://127.0.0.1:8080/v1")
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
