# Tools: fillers, non-blocking tools, progress and delegation

Tool calls are where voice agents stall. A 100-token call takes about 1.5 s to generate
before the tool even starts, and a slow backend makes it worse. `AgentSession` covers
this in four ways, and they work the same for cascades and native speech-to-speech
engines:

| Feature | What the user hears | API |
|---|---|---|
| Watchdog filler | "One moment, let me check that." when a tool round is slow | `SessionOptions.tool_filler_delay`, `function_tool(filler=...)` |
| Non-blocking tools | the conversation goes on; the result is announced later | `function_tool(blocking=False, scheduling=...)` |
| Progress updates | "Found three flights, comparing prices..." | `await ctx.report_progress(...)` |
| Delegation | background work that survives turns and interruptions | `session.delegate(coro)` |

## Blocking tools (the default)

```python
@function_tool
async def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return await weather_api(city)
```

The model calls the tool and waits for the result. All calls of one response run at the
same time. Once the response is done and every call has finished, the outputs go back to
the engine, and the last one triggers the follow-up response. `max_tool_steps` limits
how many rounds can follow each other.

Timeouts: `SessionOptions.tool_timeout` (30 s) and `function_tool(timeout=...)` both apply,
and the smaller one wins. A timeout, an exception or invalid arguments become an error
output (`FunctionCallOutput.is_error`), so the model can recover. Nothing is raised into
the session.

## Watchdog fillers

If a blocking round keeps the conversation silent for `tool_filler_delay` seconds (default
1.5 s), the session says a short filler with `say()`:

```python
SessionOptions(
    tool_filler_delay=1.5,  # None disables fillers
    tool_fillers=["One moment, let me check that.", "Just a second."],
    tool_filler_interruptible=True,  # False: the user cannot barge in on the filler
)
```

* The filler is only spoken for slow rounds. If the tools finish first, nothing is said.
* It is said at most once per round.
* The delay counts from the last moment someone spoke. If the model talked before
  calling the tool ("Sure, let me look"), or the user is talking, the watchdog waits for
  silence and then starts counting again.
* Phrases are picked at random from the list, without repeating one before all have
  been used.
* The filler does not belong to the user's turn. It does not change `voice_to_voice`,
  and it does not end the turn. The agent state goes back to `thinking` after the
  filler, not to `listening`.
* The tool outputs are sent only after the filler has been generated, so the follow-up
  answer does not cut it off.
* No filler is spoken when the engine's model keeps talking while tools run
  (`EngineCapabilities.tool_mode != "blocking"`, e.g. Gemini Live with non-blocking
  tools).

Per tool:

```python
@function_tool(filler="Checking your calendar.")          # one phrase
@function_tool(filler=["Let me see.", "Checking..."])     # picked without repeating
@function_tool(filler=lambda call: f"Looking up {json.loads(call.arguments)['city']}.")
@function_tool(filler=False)                              # never
```

`None` or `True` uses the session's `tool_fillers`. If several calls are running, the
filler comes from the first running call whose tool has one. The session emits
`tool_filler` (`ToolFiller(text, calls, waited)`).

## Non-blocking tools

```python
@function_tool(blocking=False, scheduling="when_idle")
async def book_hotel(city: str) -> str:
    """Book a hotel (takes a while)."""
    return await slow_booking(city)
```

The model does not wait for the result. The conversation goes on, and the result is
delivered when it is ready. How depends on the engine:

* **Engines with native asynchronous tools** (`tool_mode` `"non_blocking"`, e.g. Gemini
  Live): the result goes out as a normal function response through
  `EngineConnection.send_async_tool_output(output, scheduling=...)`. Gemini maps
  `scheduling` to `FunctionResponse.scheduling` (`INTERRUPT` / `WHEN_IDLE` / `SILENT`).
* **Other engines** (cascades, OpenAI Realtime, blocking Gemini models...): the model
  immediately gets an acknowledgement as the call's output (`function_tool(ack=...)`,
  default `DEFAULT_TOOL_ACK`: "The task is running in the background..."). It answers
  and the conversation continues. When the result arrives, the session adds it to the
  context as a message ("Result of the background task book_hotel (call ...): ...") at a
  moment that fits `scheduling`:

| `scheduling` | Delivered | Response |
|---|---|---|
| `"when_idle"` (default) | once the agent has finished speaking and the user is silent | yes |
| `"interrupt"` | right away (the agent's current speech is interrupted and truncated) | yes |
| `"silent"` | once the agent and the user are silent | no (the model uses it later) |

In every mode, the session waits while the user is speaking or a blocking tool round is
pending. The injected message is also added to `session.history`, with
`metadata["background_result"] = True` and `metadata["tool_call_id"]`. `tool_result`
fires when the result arrives, with `ToolResult.blocking=False`.

Non-blocking calls are not tied to the response that made them. An interruption does
not stop them. The engine can still withdraw them, see [Cancellation](#cancellation).

## Progress updates

A tool that takes a `ToolContext` can report progress:

```python
@function_tool(blocking=False)
async def search_flights(ctx: ToolContext, route: str) -> str:
    """Search flights."""
    offers = await first_pass(route)
    await ctx.report_progress(f"Found {len(offers)} flights, comparing prices.")
    await ctx.report_progress("Cheapest so far is 90 euros.", speak=False, to_model=True)
    return await compare(offers)
```

* `tool_progress` (`ToolProgress(call, message, spoken)`) is always emitted. Use it to
  update a UI or to send a data message to the client.
* `speak=True` (default): the agent says the message, unless the user or the agent is
  talking. In that case it is skipped and `report_progress` returns `False`. For a
  blocking round, spoken progress counts as the round's filler.
* `to_model=True`: the message is added to the model's context without a response.

## Delegation

`session.delegate(work, name=..., scheduling=..., timeout=...)` runs any coroutine in the
background (the "thinker"), independently of turns and interruptions. The conversation
(the "talker") goes on. When the work finishes, its result, error or timeout is
delivered like a non-blocking tool result:

```python
async def on_user_turn_completed(self, session, msg):
    if "research" in msg.text:
        session.delegate(deep_research(msg.text), name="research")
```

The returned task can be awaited or cancelled. If it is cancelled, the model is told
silently.

## Cancellation

* **Engine-withdrawn calls** (`ToolCallCancelled`, e.g. Gemini cancels pending calls when
  the user barges in, or after a reconnect): the running tool is cancelled
  (`CancelledError` inside the tool), `tool_cancelled` (`ToolCancelled(call, duration)`)
  is emitted, and no output is sent. The engine does not expect one. The other calls
  of the round still complete normally.
* **App-cancelled calls**: `session.cancel_tool_call(call_id)` does the same for any
  running call. For a non-blocking call on an engine that got an acknowledgement, the
  model is also told silently that the task was cancelled.
* Session shutdown cancels everything without emitting events.

## Events

| Event | Payload |
|---|---|
| `tool_call` | `ToolCalled(call)` |
| `tool_result` | `ToolResult(call, output, duration, blocking)` |
| `tool_filler` | `ToolFiller(text, calls, waited)` |
| `tool_progress` | `ToolProgress(call, message, spoken)` |
| `tool_cancelled` | `ToolCancelled(call, duration)` |

`TurnMetrics.tool_calls` counts the calls of the turn, non-blocking ones included. The
recorder (`record=...`) writes all of these events to the JSONL timeline.

## Limitations

* An engine without native asynchronous tools gets results as text messages in the
  conversation (as the user role, because the engine interface has no system-message
  injection). The model sees them as context, not as function outputs.
* `say()` on native engines asks the model to say the filler verbatim. Most models
  comply, but they are not guaranteed to be exact.
* Engines with `tool_mode="delegation"` ([GPT-Live](../providers/openai-live.md)) get
  *every* result, blocking tools included, through `send_async_tool_output` as soon as
  it is ready (the voice model does not wait, it keeps talking), and no fillers.
