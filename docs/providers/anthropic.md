# Anthropic Claude (`llm: anthropic`)

`llm="anthropic/claude-haiku-4-5"` runs Claude through the official
[`anthropic`](https://pypi.org/project/anthropic/) SDK and the streaming Messages API. It
streams text deltas, emits complete tool calls, uses prompt caching and reports token
usage, including cache reads and writes.

Claude Haiku 4.5 is the default model. In the Pipecat voice benchmark (30-turn scripted
tool-use conversations) it passed 98.0 % of turns with a 637 ms p50 time to first audio
token, the best cloud result within the ~700 ms budget ([research note 03 §7.1](../research/03-stt-tts-llm-landscape.md)).

## Setup

```bash
pip install 'voice-agent-next[anthropic]'   # or: uv sync --extra anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

Credentials are resolved in this order: `api_key=`, `ANTHROPIC_API_KEY`,
`ANTHROPIC_AUTH_TOKEN`, then whatever else the SDK finds (an `ant auth login` profile,
workload identity federation). If no credentials are found, the constructor raises
`AuthenticationError`. `ANTHROPIC_BASE_URL` overrides the endpoint.

## Usage

```python
from voice_agent_next import AgentSession, create

# as part of a cascade
session = AgentSession(stt="deepgram/nova-3", llm="anthropic/claude-haiku-4-5", tts="cartesia")

# or directly; "claude" is an alias of "anthropic"
llm = create("llm", "claude/claude-sonnet-5", max_tokens=512)
```

```yaml
# config file
llm:
  provider: anthropic/claude-haiku-4-5
  max_tokens: 512
  extra:
    metadata: {user_id: "caller-123"}
```

## Models

| Model | Notes for voice |
|---|---|
| `claude-haiku-4-5` (default) | Fastest Claude: 98.0 % pass at 637 ms p50 TTFAT. No thinking by default. $1 / $5 per 1M tokens. |
| `claude-sonnet-5` | 93.0 % at 1,204 ms. Adaptive thinking is on by default; lower latency with `extra={"output_config": {"effort": "low"}}`. |
| `claude-sonnet-4-6` | Accepts `temperature`. Thinking is off unless requested. |
| `claude-opus-5`, `claude-opus-4-8` | Most capable, but slow for real-time turns. |

Any model id the API accepts works; the list above is only a suggestion.

## Options

| Option | Default | Meaning |
|---|---|---|
| `model` | `claude-haiku-4-5` | Model id. |
| `api_key`, `base_url` | env | Credentials and endpoint (see Setup). |
| `max_tokens` | `1024` | Output cap per response. Thinking tokens count towards it, so raise it on thinking models. |
| `temperature` | not sent | Only sent when set (it goes in the request body; `anthropic` 1.x dropped the keyword). Claude Opus 4.7 and later, and Sonnet 5, reject it. Haiku 4.5 and the 4.6 models accept it. |
| `tool_choice` | `auto` | Default tool choice: `"auto"`, `"required"` (also `"any"`), `"none"` or a tool name. Can be overridden per `chat()` call. |
| `parallel_tool_calls` | `None` | `False` sets `disable_parallel_tool_use` (at most one call per turn). |
| `prompt_caching` | `True` | Adds cache breakpoints (see below). |
| `cache_ttl` | `"5m"` | `"1h"` keeps entries alive across long pauses but costs 2× base input to write (5-minute writes cost 1.25×). |
| `timeout` | `30.0` | Seconds per request phase (connect is capped at 5 s). Between SSE events this acts as an inactivity timeout. |
| `max_retries` | `1` | SDK retries (connection errors, 408/409/429/5xx) before the stream starts. The SDK obeys `retry-after`, which can be long; use `0` behind a failover chain. |
| `keepalive_expiry` | `120.0` | Seconds an idle connection stays pooled. httpx's default of 5 s would add a TLS handshake to most turns. |
| `extra` | `{}` | Extra request-body fields sent on every request, e.g. `{"output_config": {"effort": "low"}}`, `{"thinking": {...}}`, `{"metadata": {...}}`, `{"top_k": 5}`, `{"stop_sequences": [...]}`. The per-call `extra=` dict overrides them. (`extra_params` is a deprecated alias.) |
| `headers` | `{}` | Extra HTTP headers, e.g. `{"anthropic-beta": "..."}`. (`extra_headers` is a deprecated alias.) |
| `client` | — | A pre-built async SDK client (`AsyncAnthropicBedrockMantle`, `AsyncAnthropicVertex`, `AsyncAnthropicFoundry`, `AsyncAnthropicAWS`...). The connection options above are then ignored and the client is not closed by `aclose()`. Use that platform's model ids. |
| `http_client` | — | An `httpx2.AsyncClient` (proxies, custom transports). |

## How the conversation is sent

`ChatContext` items map onto the Messages API as follows (`to_anthropic_messages()`):

* **System and developer messages** go to the top-level `system` prompt, one text block each. Instructions placed before the conversation stay first; later ones, such as per-response instructions, come after them.
* **Consecutive items with the same role** merge into one turn.
* **Start and end with the user.** An empty conversation, or one that starts with the agent's greeting, gets a `(start of the conversation)` user turn at the front. A conversation that ends with an assistant turn gets a `(continue)` user turn at the end, because Claude 4.6+ models reject assistant prefill.
* **Tool calls** (`FunctionCall`) become assistant `tool_use` blocks. Each output (`FunctionCallOutput`) becomes a `tool_result` block at the start of the next user turn, keeping `is_error`. This still holds when the user spoke while a tool was running. Calls without an output, and outputs without a call, are dropped because the API rejects unmatched blocks. Ids from other providers are sanitized to `[a-zA-Z0-9_-]`.
* **History with tool calls but no tools this turn** (for example after a handoff): the history's tool names are sent as placeholder definitions with `tool_choice: none`, because the API requires tools to be defined whenever `tool_use` blocks are present.
* **Images** in user messages become `image` blocks from `https://` URLs or `data:` URLs. Images in system or assistant messages are dropped with a warning.
* **Audio** is sent as its transcript. Claude does not accept audio input, so a user message with untranscribed audio raises `ConfigurationError`: add an STT stage to the cascade.
* **Interrupted assistant messages** keep only the text the user actually heard.

## Tool calls

Streamed `input_json_delta` fragments are accumulated per content block. When a
`tool_use` block finishes (`content_block_stop`), one complete `FunctionCall` is emitted
whose `call_id` is the Anthropic `toolu_...` id and whose `arguments` is the JSON the model
produced (`"{}"` for tools without input). If the input JSON is incomplete, for example
because `max_tokens` cut it off, the call is dropped and logged, so a tool never runs on a
truncated input. The final chunk then reports `finish_reason="length"`.

`finish_reason` values use the framework's names: `stop` (from `end_turn` or
`stop_sequence`), `tool_calls`, `length` (from `max_tokens`),
`content_filter` (from `refusal`).

## Prompt caching

With `prompt_caching=True`, up to three `cache_control` breakpoints are placed:

1. on the last tool definition;
2. on the last *leading* system block (the agent instructions);
3. on the last block of the conversation (never on the `(continue)` placeholder).

Each turn then reads the previous prefix from the cache: cache reads bill at ~0.1× of
input and also lower TTFT. Only the newly appended turn is written, at 1.25×. Per-response
instructions come after breakpoint 2, so the tools and instructions stay cached even when
those instructions change. They do invalidate the conversation cache for that one
request.

A prefix shorter than the model's minimum is simply not cached. There is no error, and
`usage.cache_creation_tokens` is `0`. The minimums are 4,096 tokens on Haiku 4.5,
1,024 on Sonnet 5 and Sonnet 4.6, and 512 on Opus 5. For a short voice prompt on Haiku,
caching therefore starts once the conversation grows, which is also when the latency and
cost savings matter.

**Pre-warming.** `await llm.warmup()` sends a free `GET /v1/models/{model}`. That opens and
pools the TLS connection before the first turn, and a bad key or model id shows up as a
logged warning. If you also pass the prompt, a `max_tokens=0` request writes the cache
entry for the tools and instructions, so the first real turn reads it:

```python
ctx = ChatContext()
ctx.add_message("system", agent.instructions)
await llm.warmup(ctx, tools=agent.tools)  # the same system prompt and tools as real turns
```

The system prompt, tools, thinking and effort settings must match later requests exactly,
or the pre-warmed entry is never read. Warmup failures are logged, never raised.
Pre-warming is skipped with manual extended thinking (`thinking.type: "enabled"`), which
the API does not allow with `max_tokens=0`.

**Checking hits.** The final chunk's `usage` is an `AnthropicUsage`. `prompt_tokens` is the
whole prompt, `cached_tokens` the part read from the cache, `cache_creation_tokens` the
part written, and `uncached_prompt_tokens` the rest. `LLMMetrics` (and the session's
`UsageSummary`) carry all three, so cost estimates can price reads, writes and uncached input
separately.

## Thinking

Thinking and signature deltas are never forwarded, so the reasoning is never spoken. On
models that think by default (Sonnet 5, Opus 5 and newer), control latency with
`extra={"output_config": {"effort": "low"}}` rather than by turning thinking off.
Thinking tokens count towards `max_tokens`. See also [Limitations](#limitations).

## Errors

| Failure | Raised |
|---|---|
| 401, 403 | `AuthenticationError` |
| 429 (HTTP or a mid-stream `rate_limit_error` event) | `RateLimitError` (retryable) |
| 408, 504 `timeout_error`, SDK timeouts, stalled streams | `ProviderTimeoutError` (retryable) |
| Connection refused or dropped mid-stream | `ProviderConnectionError` (retryable) |
| 500 `api_error`, 529 `overloaded_error` (HTTP or mid-stream `error` event), 409 | `ProviderError(retryable=True)` |
| 400, 402, 404, 413 | `ProviderError(retryable=False)` |

Messages include the API error type, its message and the `request-id`, but never the
key. A mid-stream `error` event arrives with HTTP status 200; its status is derived from
the error type.

## Latency

* Text streams token by token. The cascade's sentence segmenter starts TTS on the first
  clause.
* `keepalive_expiry=120` and `warmup()` keep the TLS connection warm between turns.
* Keep the instructions and tool list stable during a call. Any change to them rewrites
  the cache.
* `max_retries=1` and `timeout=30` favour failing fast over stalling a turn.
* When a stream is cancelled (barge-in), `LLMMetrics` still reports the prompt tokens
  already billed.

## Limitations

* **Thinking blocks are not replayed.** `ChatContext` has no place for them. Adaptive
  thinking (on by default on Sonnet 5, Opus 5 and newer) does not need them, so tool loops
  work. Manual extended thinking (`thinking: {"type": "enabled", ...}`) with tools does not
  work: the API requires the thinking block with the tool result and returns a 400.
* **Late system messages go to the top-level `system` prompt** rather than into
  `messages`, so they invalidate the conversation cache for that request. Models that
  support mid-conversation `system` messages could avoid this; that is not used yet.
* **No audio input:** Claude reads transcripts only, so use an STT stage.
* **Unit tests need the SDK.** `tests/providers/test_anthropic.py` is skipped unless the
  `anthropic` extra is installed; CI's default `uv sync` does not install it. The
  conversion tests run everywhere.

## Tests

* `tests/providers/test_anthropic_convert.py` covers `ChatContext` conversion without the
  SDK.
* `tests/providers/test_anthropic.py` runs the provider against an `httpx2.MockTransport`
  that replays real SSE event streams split into small byte chunks. It covers text, tool
  use with fragmented JSON, parallel tools, usage with cache tokens, HTTP and mid-stream
  errors, cancellation, warmup, and a full `AgentSession` cascade turn with a tool call:
  `uv sync --extra anthropic && uv run pytest tests/providers/test_anthropic.py`.
* A real-API test (skipped unless `ANTHROPIC_API_KEY` is set):
  `uv run pytest -m integration tests/providers/test_anthropic.py`. Set
  `ANTHROPIC_TEST_MODEL` to change the model.
