# Preemptive (speculative) generation

In a cascade, the agent's reply normally starts only when the user's turn is
**committed**: after the final transcript, the turn detector and the endpointing delay
(`min_endpointing_delay`: 0.4 s with a turn detector, 0.6 s with VAD only, measured from
the end of speech). The LLM's time to first token (TTFT) and the TTS time to first audio
then add up on top of it:

```
end of speech ─ VAD pause ─ final transcript ─ ··· silence ··· ─ commit ─ LLM TTFT ─ TTS ─ ♪
```

With `CascadeOptions(preemptive_generation=True)` the cascade starts the reply as soon as
the turn has *probably* ended and only the silence is left to wait for. The reply is
generated in the background and **held back**; when the turn is committed with the same
transcript it is released at once, otherwise it is thrown away:

```
end of speech ─ VAD pause ─ final transcript ─ ··· silence ··· ─ commit ─ TTS ─ ♪
                                     └─ LLM TTFT ─┘ (overlaps the wait)
```

Voice-to-voice latency then approaches *max(endpointing, transcript + LLM TTFT)* + TTS
instead of their sum. The price is an LLM call for every speculation that has to be
thrown away (the user went on talking).

```python
from voice_agent_next import AgentSession, CascadeOptions

session = AgentSession(
    stt="deepgram/nova-3",
    llm="openai/gpt-4.1-mini",
    tts="cartesia/sonic-2",
    vad="silero",
    cascade_options=CascadeOptions(preemptive_generation=True),
)
```

or in a config file:

```yaml
cascade:
  preemptive_generation: true
```

## When a speculation starts

* **VAD endpointing** (the default cascade): at a pause (VAD end of speech), once the
  STT has delivered the final transcript and the turn would be committed after the
  remaining silence. With a turn detector, only if its probability is at least
  `preemptive_threshold` (default: the detector's own `threshold`, i.e. only pauses that
  end the turn after `min_endpointing_delay`). Nothing starts when no time is left to
  gain (the transcript came after the endpointing delay).
* **STT turn detection** (Deepgram Flux, Cartesia Ink): on `EAGER_END_OF_TURN`
  (Flux `eager_eot_threshold`, Ink `turn_eager_end_threshold`). `TURN_RESUMED` cancels it.
* A late final transcript that changes the text restarts the speculation on the new text.

It does **not** start:

* while a reply is being generated or (probably) still playing — speech that ends then
  overlapped the agent, and the session's [interruption policy](interruptions.md)
  decides whether it is a turn at all;
* for turns longer than `preemptive_max_speech` (10 s) or after
  `preemptive_max_attempts` (3) speculations in the same turn;
* without an STT (a half-cascade with an audio LLM has no transcript to compare).

## What is held back

Until the turn is committed, a speculative reply is invisible:

* no `ResponseStarted`, `ResponseText`, `ResponseAudio` or `ResponseToolCall` event
  reaches the session (so nothing is played, transcribed or added to the session history);
* nothing is added to the engine's chat context;
* **tool calls are not executed**: they are reported at the commit, like any tool call;
* nothing reaches the TTS — unless `preemptive_tts=True`, which also synthesizes the start
  of the reply (its audio is held back too).

## At the commit

The speculation is kept if the committed turn has the same transcript (whitespace
normalized) and nothing else the reply depends on changed: history (including a
truncated agent message after a barge-in), instructions and tools. Its LLM input is
exactly the one a reply started at the commit would get (same history window).

A kept reply is released as an ordinary response: `ResponseStarted` and everything held
back are delivered at the commit, stamped with the commit time, so `TurnMetrics`
(`end_of_turn_delay`, `response_ttfb`, `voice_to_voice`), barge-in, truncation and word
alignment behave exactly as for any other reply. Otherwise it is cancelled and a normal
reply starts.

A speculation is discarded when:

| reason | when |
|---|---|
| `resumed` | the user speaks again (VAD / STT speech start, STT `TURN_RESUMED`) |
| `transcript` | the committed transcript differs |
| `context` | the history, instructions or tools changed (`update()`, `send_text()`, `send_tool_output()`, `truncate()`) |
| `cancelled` | `cancel_response()` or another response started |
| `cleared` | `clear_input()` (e.g. the session dropped a backchannel) |
| `failed` | the speculative generation failed before the commit (a normal reply is tried) |
| `closed` | the connection closed |

## Options

All are fields of `CascadeOptions` (the `cascade:` block in config files).

| Option | Default | Meaning |
|---|---|---|
| `preemptive_generation` | `False` | Start the reply speculatively (see above). |
| `preemptive_tts` | `False` | Also synthesize it before the commit: hides the TTS time to first audio too, wastes synthesis when discarded. |
| `preemptive_threshold` | `None` | Minimum turn-detector probability to speculate at a pause (`None` = the detector's `threshold`). Lower it to also speculate on pauses that wait `max_endpointing_delay`. |
| `preemptive_max_speech` | `10.0` | No speculation on turns longer than this (seconds). |
| `preemptive_max_attempts` | `3` | Maximum speculative LLM calls per user turn. |

## Metrics

Every speculation ends with one `SpeculationMetrics` event (`session.on("metrics")`):
`hit`, `reason` (see above), `lead` (seconds between its start and the commit or discard),
`output_tokens` (generated before the commit/discard), `request_id` (its `LLMMetrics`) and
`response_id` (hits). `session.usage` adds them up: `speculation_hits`,
`speculation_waste_calls`, `speculation_waste_tokens`. The discarded calls themselves
also appear as ordinary `LLMMetrics` (the cost of speculation is in the usage totals).

## How much it saves

The saving is *min(LLM TTFT, time between the final transcript and the commit)* — plus
the TTS time to first audio with `preemptive_tts`, if that fits in the window too. It is
largest with cloud LLMs (TTFT 200–800 ms), fast streaming STTs (the transcript arrives
early in the endpointing window), VAD-only endpointing (0.6 s window), and STTs with an
eager end of turn. It is zero when the final transcript only arrives after the
endpointing delay, e.g. a local batch STT (faster-whisper takes ~360 ms after the VAD's
0.25 s pause) with a confident turn detector (0.4 s).

MEASUREMENTS

## Why it is off by default

Every discarded speculation is a full LLM call (the whole prompt is sent again), and the
gain depends on the configuration (see above) — zero on the fully local stack. It also
changes what an LLM receives: apps that count or script LLM requests see the speculative
ones. Turn it on for cloud cascades, especially with VAD-only endpointing or an STT with an
eager end of turn, and watch `speculation_waste_calls` / `speculation_hits`.

## Limitations

* Speculation only covers the reply to a user turn committed by the engine (not
  `create_response()`, `say()` or tool follow-ups).
* A speculation started on a pause is not retried later in the same pause (e.g. after a
  barge-in was resolved), only on the next pause or transcript.
* With `preemptive_tts`, the whole reply may be synthesized before the commit (not just
  its first sentence), all of it wasted when discarded.
