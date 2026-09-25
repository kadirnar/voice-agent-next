# Session rotation and reconnects

Provider sessions do not last forever, and connections drop:

| Engine | Limit | What the library does |
|---|---|---|
| OpenAI Realtime (+ Azure, xAI, compatible servers) | 60 min per session (`expires_at`) | rotates to a new session prepared in the background, re-seeded with the conversation |
| Gemini Live | ~10 min per connection (`goAway`), 15 min audio without compression | resumes the server-side session with a resumption handle; re-seeds a fresh session if the handle expired |
| OpenAI GPT-Live | per session (`expires_at`, announced `expiry_warning` ahead) | `RotatingEngine`: a new session prepared in the background, seeded through `session.input` |
| Nova Sonic | 8 min | (engine not wired yet) `RotatingEngine` rotates any engine |
| any engine | network drops | reconnects with backoff and the same carry-over |

The goal: the user never notices. The conversation continues on the new connection with the
same instructions, tools, voice and history, and no user audio is lost.

## What happens during a rotation

1. **Schedule.** `RotationPolicy.lead` seconds before the provider limit (at most half of the
   limit), a server `goAway`, `EngineStatus("expiring")`, or an explicit `conn.rotate()` makes
   a rotation *pending*. The application sees `EngineStatus("expiring", time_left=...)`.
2. **Make before break.** The next connection is opened right away and configured with the
   current instructions, tools and voice. When the old session cannot be resumed, the
   conversation is carried over: the history is fitted by the carry-over strategy and seeded
   into the new session as text.
3. **Quiet moment.** The switch waits until nobody is talking: the user is not speaking, no
   response is being generated or requested, the agent's audio has finished playing
   (estimated from the audio received), no tool call is waiting for its result, and nothing
   happened for `quiet_period` (0.5 s). If the conversation moved on while the next connection
   was waiting, it is re-seeded first.
4. **Switch.** `EngineStatus("reconnecting")`. User audio is buffered while the connection is
   swapped; the recent audio of the turn the user may have started (at most `replay` = 1 s,
   never audio of an already committed turn) is sent again, then the buffered audio.
   Events of the old connection are ignored from now on and it is closed, so a response can
   never come twice. `EngineStatus("reconnected")` (or `"resumed"` when the provider kept the
   session, e.g. a Gemini resumption handle).
5. **Deadline.** If the conversation never goes quiet, the switch is forced
   `force_margin` (10 s) before the limit (for Gemini: `go_away_margin` before the `goAway`
   deadline). A response in flight then ends with `ResponseDone(status="failed")`.

A planned rotation that cannot open the next connection keeps the current one
(`EngineStatus("resumed", detail="kept the ...")` plus a recoverable `EngineErrorEvent`) and
tries again later.

## Reconnects after a drop

When the connection drops, the engine fails the response in flight
(`ResponseDone(status="failed")`), emits `EngineStatus("reconnecting")` and reconnects with
exponential backoff (`max_reconnect_attempts`, `backoff`). The new session gets the same
carry-over as a planned rotation. User audio sent during the outage is buffered (up to
`max_buffered_audio` = 30 s) and delivered afterwards; control calls made meanwhile
(`send_text`, tool results, `create_response`) are carried over or sent once the new session
is up. After the last failed attempt the engine emits a non-recoverable `EngineErrorEvent`.

Silent reconnects that lose work are how agents go quiet for a minute (research note 05,
Pipecat #5305). So a switch never hides a loss: responses it cut off end as `failed`, and
buffered audio that overflowed is reported as a recoverable `EngineErrorEvent` and in
`RotationMetrics.lost_audio`.

## What is carried over

The engine keeps its own record of the conversation *as the user heard it*
(`ConversationRecorder`): final user transcripts, the agent's text trimmed to the audio that
was played when the user interrupted (`interrupt()`/`truncate()`), tool calls and their
results, and text messages. Instructions and tools are carried separately (so `update()`
survives a rotation), and so is the voice.

The history is fitted into the new session by a **carry-over strategy** — any
`async def strategy(history: ChatContext) -> ChatContext`:

```python
from voice_agent_next import create
from voice_agent_next.engines import RotationPolicy, SummarizeHistory, TruncateHistory

# default: the most recent items within ~24k characters (tool calls kept with their results)
policy = RotationPolicy(carry_over=TruncateHistory(max_items=100, max_chars=24_000))

# summarize older turns with a (cheap) LLM, keep the last 12 items verbatim
policy = RotationPolicy(carry_over=SummarizeHistory(create("llm", "openai/gpt-4.1-mini")))

engine = create("engine", "openai/gpt-realtime-2.1", rotation=policy)
```

`SummarizeHistory` extends its summary incrementally (repeated rotations only summarize what
is new) and falls back to truncation when the LLM fails or times out. The summary is seeded
as a system message (Gemini: a user turn).

Gemini Live takes `carry_over=...` directly; its rotation timing is set by its own options
(`rotate_after`, `go_away_margin`, `resume_replay`, `max_buffered_audio`,
`max_reconnect_attempts`, see `docs/providers/gemini-live.md`), and it only needs the
carry-over when the server-side session cannot be resumed.

## Any engine: `RotatingEngine`

Engines without native rotation get the same behaviour from a wrapper. Each connection of the
wrapped engine is opened with `EngineOptions(chat_ctx=<carried history>)`, which every engine
already supports:

```python
from voice_agent_next import AgentSession, create
from voice_agent_next.engines import RotatingEngine, RotationPolicy

engine = RotatingEngine(create("engine", "mock"), policy=RotationPolicy(rotate_after=480))
session = AgentSession(engine)
```

The wrapper rotates before the wrapped engine's `max_session_duration`, on its
`EngineStatus("expiring")`, after `rotate_after`, or on `conn.rotate()`; it reconnects when
the wrapped connection closes or fails with a retryable error. Input speech positions are
mapped onto one continuous input stream, so latency metrics stay correct across switches.

## `RotationPolicy`

| Option | Default | Meaning |
|---|---|---|
| `rotate_after` | `None` | rotate after this connection age (overrides the limit-based schedule) |
| `lead` | 300 s | start looking for a quiet moment this long before the limit (at most half of it) |
| `force_margin` | 10 s | switch even if busy this long before the limit |
| `quiet_period` | 0.5 s | how long the conversation must have been quiet |
| `replay` | 1 s | recent uncommitted user audio re-sent to the new connection |
| `max_buffered_audio` | 30 s | cap on audio buffered during a switch |
| `max_reconnect_attempts`, `backoff`, `max_backoff` | 3, 0.5 s, 10 s | reconnects (`RotatingEngine`; OpenAI keeps its `max_reconnect_attempts`/`reconnect_backoff` options) |
| `carry_over` | `TruncateHistory()` | history strategy |
| `proactive` | `True` | `False`: only reconnect after failures |

## Events and metrics

| Signal | When |
|---|---|
| `EngineStatus("expiring", time_left=...)` | limit approaching / `goAway` |
| `EngineStatus("reconnecting", detail=<reason>)` | a switch starts (planned or after a drop) |
| `EngineStatus("reconnected")` / `("resumed")` | the new connection carries the conversation (re-seeded / resumed) |
| `ResponseDone(status="failed")` | a response the switch cut off |
| `EngineErrorEvent(recoverable=True)` | a failed planned rotation, or lost audio |
| `RotationMetrics` (`"metrics"`) | one per switch |

`RotationMetrics` fields: `reason`, `planned`, `resumed`, `rotation` (count), `gap` (seconds
user audio was held back), `attempts`, `buffered_audio`, `replayed_audio`, `lost_audio`,
`carried_items`, `failed_responses`. `AgentSession.usage` sums them up
(`engine_rotations`, `engine_rotation_gap`, `engine_lost_audio`).

## Limitations

* The carried history is text. A new session does not hear earlier audio (tone, names spelled
  out), and an answer played before the switch can no longer be truncated server-side.
* Quietness is judged from engine events; playback is estimated from the audio received
  (real-time playout), not measured at the transport.
* A forced switch (deadline) cuts off the response in flight.
