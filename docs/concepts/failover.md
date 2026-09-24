# Provider failover

Cloud providers fail: connections drop, rate limits hit, a model stalls, a key expires.
`FallbackSTT`, `FallbackLLM` and `FallbackTTS` (in `voice_agent_next.fallback`) wrap
several providers of one kind and move a request to the next provider when the current one
fails. Each wrapper *is* an STT, LLM or TTS, so it works anywhere a single provider does:
in a `CascadeEngine`, an `AgentSession`, or on its own.

```python
from voice_agent_next import AgentSession, FallbackLLM, create

session = AgentSession(
    stt=["deepgram/nova-3", "elevenlabs"],  # a list = failover chain
    llm=["groq/llama-3.3-70b-versatile", "openai/gpt-4.1-mini"],
    tts=[{"provider": "cartesia/sonic-2", "voice": "<id>"}, "elevenlabs"],
    vad="silero",
)

# or build one explicitly, with options
llm = FallbackLLM(
    [create("llm", "anthropic/claude-haiku-4-5"), create("llm", "openai/gpt-4.1")],
    first_token_timeout=5.0,
    cooldown=60.0,
    probe_interval=15.0,
)
```

In a config file, write a list, or a mapping with a `fallback` key to set options:

```yaml
llm: [groq/llama-3.3-70b-versatile, openai/gpt-4.1-mini]
tts:
  fallback: [cartesia/sonic-2, kokoro]   # cloud first, local backup
  first_audio_timeout: 3.0
```

## When a request fails over

Providers are tried in order. The default policy (`is_failover_error`) fails over on
anything the next provider may not suffer from:

| Failure | Fails over? |
|---|---|
| connection errors, dropped sockets, unmapped SDK/network exceptions | yes (`connection` / `error`) |
| timeouts, including the wrapper's first-result timeouts (silent stalls) | yes (`timeout`) |
| a stream that ends on its own before its input did (silent disconnect) | yes (`stream_ended`) |
| rate limits (429), bad credentials (401/403), unknown model (404), 5xx | yes |
| missing optional dependency of a provider | yes |
| a request rejected as invalid (non-retryable 400/422) | no: every provider would reject it |
| the library's own configuration or usage errors | no |

Pass `failover_on=` to use your own predicate.

## Only switch while nobody can hear it

Switching mid-output would be audible, so each kind has a rule for how long switching is
still safe (see research note 05 §7.3 and §9.4):

* **LLM: before the first token.** `first_token_timeout` (default 10 s) catches a model
  that connects but never answers. After the first token (text or tool call) the answer is
  already on its way to the TTS and maybe to the listener's ears. A second model would not
  continue the same sentence: it would restart or diverge, and the history would hold
  text from two models. So an error after the first token propagates (the cascade ends
  the response and reports the error). The provider still enters its cooldown, so the
  *next* turn starts on a healthy one.
* **TTS: while none of the segment's audio exists.** `first_audio_timeout` (default 5 s)
  is measured from the request, or from a segment's flush. When the providers only accept
  whole texts (`capabilities.streaming` is False on any of them), the wrapper streams
  sentence by sentence and each sentence fails over on its own. When all of them stream
  text natively, the wrapper keeps the text of the segments that have not finished and
  replays it to the next provider's stream. A segment whose audio has started is never
  switched: the listener would hear another voice start the sentence again.
* **STT: at any time.** When the stream fails, the next provider's stream starts and the
  current utterance is replayed from a ring buffer (`replay_seconds`, default 5 s) so it
  is not lost. Audio already covered by a final transcript is dropped from the buffer and
  not replayed. A replayed utterance does not emit a second `START_OF_SPEECH`.
  `final_timeout` (off by default: some providers send no final for a flush without
  speech) fails over when no final transcript follows a flush in time.

When every provider has failed, the last error is raised.

## Health, cooldown and recovery

A failed provider is marked unavailable and skipped for `cooldown` seconds (default 30).
After the cooldown it is tried again in its normal priority order; if it answers, it is
available again and has recovered. With `probe_interval` set, a background task also
calls `warmup()` on unavailable providers and restores them as soon as a probe succeeds.
A failed probe restarts the cooldown. (A provider whose `warmup()` does nothing passes
every probe, so for such providers probing acts like a shorter cooldown.)

Providers in cooldown are still tried as a last resort, when every other provider has
failed for this request. That beats failing the request outright.

`warmup()` warms all providers and only marks the ones that fail as unavailable. It
raises only when every provider fails. `aclose()` closes all providers and stops the
probe.

`wrapper.health` lists a `ProviderHealth` per provider (`available`, `retry_at`, `served`,
`failures`, `last_error`).

## Observability

Every wrapper emits:

* `"metrics"`: each provider's own `STTMetrics` / `LLMMetrics` / `TTSMetrics`, one per
  attempt. Their `provider`/`model` fields show who served each request, and failed
  attempts carry `error`. The wrapper adds no metrics of its own, so a cascade does not
  double-count.
* `"provider_failover"` → `ProviderFailover(kind, from_provider, to_provider, reason,
  error, request_id)`.
* `"provider_availability_changed"` → `ProviderAvailabilityChanged(kind, provider,
  available, error)`, when a provider goes down or recovers.

`wrapper.stats` returns a `FailoverStats` snapshot: requests served and failures per
provider, the number of failovers, and failovers by reason. A stream returned by the
wrapper has a `served_by` attribute (`"provider/model"`).

```python
llm.on(
    "provider_failover",
    lambda ev: log.warning("%s -> %s (%s)", ev.from_provider, ev.to_provider, ev.reason),
)
```

## Details

* **Capabilities** are the intersection of the providers' capabilities. For example, tool
  calling stays on only if every LLM supports it, and a TTS streams text natively only if
  every provider does.
* **Audio formats.** `FallbackTTS` resamples every provider to one output rate (the first
  provider's rate, or `sample_rate=`). `FallbackSTT` accepts audio at its `sample_rate`,
  and each provider resamples it to its own rate.
* **Voices** are provider-specific. Configure each TTS provider's `voice`. A voice passed
  to the wrapper is forwarded to every provider.
* **Options** such as `temperature`, `max_tokens`, `tools` and `extra` are forwarded to
  whichever LLM serves the request.

## Limitations

* An LLM or TTS that fails after output started is not resumed on another provider.
  Continuing only the unspoken rest (resynthesizing from the playout cursor) is a possible
  follow-up.
* The STT buffer is cut at final transcripts. With provider-side endpointing (no flush),
  it is cut at the moment the final arrives, so speech that started right before the
  final could be missing from a replay.
