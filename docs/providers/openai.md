# OpenAI (LLM)

`llm="openai/gpt-4.1-mini"` streams OpenAI's Chat Completions API through the official
[`openai`](https://pypi.org/project/openai/) SDK. It forwards text deltas as they arrive,
emits complete tool calls and reports token usage, including cached prompt tokens.

The same class, `OpenAILLM`, talks to any server that speaks the Chat Completions
protocol. Preconfigured providers exist for Ollama, llama.cpp, vLLM, LM Studio, Groq,
Cerebras, Together, OpenRouter, DeepSeek, Fireworks and SambaNova: see
[OpenAI-compatible servers](openai-compatible.md).

## Setup

```bash
pip install 'voice-agent-next[openai]'   # or: uv sync --extra openai
export OPENAI_API_KEY=sk-...
```

Credentials come from `api_key=`, then `OPENAI_API_KEY`. Without a key the constructor
raises `ConfigurationError`. `OPENAI_BASE_URL`, the SDK's own variable, points the
provider at a gateway, and the OpenAI key is sent there too. An explicit `base_url=`
means *another server*: `OPENAI_API_KEY` is then never sent, so pass `api_key=` if that
server needs one.

## Usage

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

## Models

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

## Options

| Option | Default | Meaning |
|---|---|---|
| `model` | `gpt-4.1-mini` | Model id. |
| `api_key`, `base_url` | env | See Setup. `base_url` includes the version path (`https://host/v1`). |
| `temperature`, `max_tokens` | not sent | Defaults for every request, overridable per `chat()`. `max_tokens` is sent as `max_completion_tokens` to OpenAI and as `max_tokens` to other servers. |
| `reasoning_effort` | not sent | `"none"`, `"minimal"`, `"low"`, ... for reasoning models. |
| `parallel_tool_calls` | not sent | `False` limits the model to one tool call per response. Only sent with tools. |
| `extra` | `{}` | Extra request parameters for every request; the per-call `extra=` wins. Parameters the SDK knows (`seed`, `stop`, `top_p`, `service_tier`, `prompt_cache_key`, `verbosity`...) are passed as such; anything else goes into the JSON body (`extra_body`). A `None` value removes a default. |
| `headers` | — | Extra HTTP headers on every request. |
| `capabilities` | tools, parallel tools, image input | Override the declared `LLMCapabilities`, e.g. `audio_input=True` for audio-input models. |
| `max_tokens_param` | auto | `"max_completion_tokens"` for OpenAI and Azure OpenAI, `"max_tokens"` elsewhere. |
| `developer_role` | auto | Role used for `developer` messages: `"developer"` for OpenAI, `"system"` elsewhere. |
| `include_usage` | `True` | Sends `stream_options={"include_usage": true}` to get token usage. Turn it off for servers that reject it. |
| `strip_thinking` | `True` | Drops a leading `<think>…</think>` block from the text (see below). |
| `timeout` | `60.0` | Seconds for connecting and for each read, i.e. the longest wait for the next streamed chunk. |
| `max_retries` | `1` | SDK retries for connection errors and 408/409/429/5xx, before the stream starts. The SDK obeys `retry-after`, which can be long: use `0` behind a failover chain. |
| `keepalive_expiry` | `120.0` | Seconds an idle connection stays pooled. httpx's default of 5 s would make most turns pay a new TLS handshake. |
| `client` | — | A pre-built `openai.AsyncOpenAI`-compatible client, e.g. `openai.AsyncAzureOpenAI`. The connection options above are then ignored and the client is not closed by `aclose()`. |
| `http_client` | — | An `httpx.AsyncClient` (or `httpx2.AsyncClient` with openai 3.x) for proxies or custom transports. Not closed by `aclose()`. |

## How the conversation is sent

`ChatContext` items map to Chat Completions messages as follows
(`providers/openai/_format.py`, `to_chat_messages()`):

* **System and developer messages** keep their position and role (`developer` becomes
  `system` for servers other than OpenAI).
* **User messages** are sent as a string, or as content parts when they carry media:
  `image_url` for `ImageContent` (`https://` or `data:` URLs) and `input_audio` (base64
  WAV) for `AudioContent`. Audio is sent only when `capabilities.audio_input` is set;
  otherwise its transcript is used, and audio without a transcript is dropped.
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

## What the stream yields

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

## Errors

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

## Latency notes

* **`warmup()`** opens the HTTP connection (`GET /models`) so the first turn skips the
  TCP and TLS handshakes; `CascadeEngine.warmup()` calls it. Idle connections are kept
  for `keepalive_expiry` (120 s) so the next turns reuse them. Failures are logged, not
  raised.
* **Prompt caching** is automatic on OpenAI for prompts of 1,024 tokens or more. The
  instructions and tools come first in every request, so keep them stable. `extra={"prompt_cache_key": ...}`
  improves hit rates across sessions. Cache hits appear in `LLMMetrics.cached_tokens`.
* **Reasoning** delays the first token: prefer non-reasoning models, or
  `reasoning_effort="none"`, for conversational turns.

## Azure OpenAI and custom clients

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

## Audio input

Chat models that accept audio (for example `gpt-audio`) can receive the user's audio
directly: construct the LLM with `capabilities=LLMCapabilities(audio_input=True)` and user
`AudioContent` is sent as WAV `input_audio` parts. Half-cascades built on this are tracked
in issue #14.
