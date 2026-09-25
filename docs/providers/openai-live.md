# OpenAI GPT-Live (`openai-live` engine)

GPT-Live (`gpt-live-1`) is OpenAI's **full-duplex** voice model. It listens while it
speaks, decides by itself when to talk, and **delegates** reasoning and tool use to a
backend while the conversation goes on. It uses its own protocol, the Live API
(`wss://api.openai.com/v1/live/sessions`, `session.*` events), not the Realtime API.
The code is in `voice_agent_next/providers/openai/live.py`.

```python
from voice_agent_next import Agent, AgentSession, function_tool


@function_tool
async def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return await weather_api(city)


session = AgentSession("openai-live/gpt-live-1")  # or "gpt-live"
await session.run(
    Agent("You are a friendly travel assistant. Keep answers short.", tools=[get_weather]),
    transport,
)
```

```yaml
# agent.yaml
engine:
  provider: openai-live/gpt-live-1
  responses_model: gpt-5.6-terra
  responses_instructions: "Check the weather with get_weather. Answer with the facts only."
```

## Setup

No extra is needed (the engine uses `websockets`, a core dependency). Set
`OPENAI_API_KEY` to a project key with GPT-Live access (every paid tier; the rate limit
counts concurrent sessions, from 25 on Tier 1 to 500 on Tier 5). Keep the key on the
server: this engine is a server-side WebSocket client.

**Pricing:** $0.05 per minute of voice session, billed per second. The Responses backend
(model and tools) is billed separately at its normal rates.

## How the conversation maps onto the engine events

GPT-Live has no request/response cycle: the model keeps the floor, backchannels and
yields by itself. The engine follows the [Moshi](moshi.md) precedent:

| Live protocol | Engine events |
|---|---|
| `session.output_audio.delta` above `speech_threshold_db` | `ResponseStarted`, `ResponseAudio` (a response is one stretch of agent speech) |
| no speech and no transcript for `response_gap` s (`yield_gap` while the user talks) | `ResponseDone` |
| `session.output_transcript.delta` | `ResponseText` |
| local energy VAD on the sent audio, while the agent is quiet | `InputSpeechStarted` / `InputSpeechStopped` |
| `session.input_transcript.delta` | `InputTranscript` (partial), final at the commit |
| first agent response after user speech | `InputCommitted` + `InputTranscript(is_final=True)` |
| `session.delegation.created` (client) / backend function call (Responses) | `ResponseToolCall` |
| `session.usage.updated`, `session.closed` | `connection.usage_seconds` |
| `expires_at` - `expiry_warning` | `EngineStatus("expiring")`, then a session rotation |
| `error` | `EngineErrorEvent` (recoverable) |

* **Overlaps belong to the model.** User speech *over* the agent is not reported, so the
  session's interruption policy never fights GPT-Live, which resolves overlaps itself.
  If the agent falls silent while the user keeps talking, the speech is reported then.
  `report_overlap=True` reports it anyway (the session then applies its policy).
* **Interrupt.** The Live API has no cancel. `session.interrupt()` ends the current
  response as `cancelled` and mutes the agent's audio locally until its next pause.
* **Transcripts** carry no item ids or turn boundaries. The engine groups the user's
  fragments into the turn it commits before the agent's next response. Fragments that
  arrive late but start before the commit update that turn.
* **Text in:** `say(text)` and `create_response(instructions)` send
  `session.instructions.append` (which can redirect speech in progress);
  `send_text(text)` sends `session.commentary.append` (the model says it, paraphrased)
  and `send_text(text, respond=False)` sends `session.thinking.append` (context that is
  not said right away). Appends longer than about 500 tokens are split.
* **Keep-alive.** The session timeline runs on input audio, so the engine streams
  silence while the transport delivers nothing (`keepalive=True`). A greeting then plays
  even before the caller has said anything.

## Delegation

`EngineCapabilities.tool_mode` is `"delegation"`: the voice model hands work to a
backend and keeps talking, and interrupting it never cancels backend work. The session
delivers **every** tool result as soon as it is ready, whatever the tool's `blocking`
flag, and says no fillers (the model covers the wait itself). Choose the mode with
`delegation=`; it is fixed for the session.

### Responses delegation (default)

GPT-Live runs a Responses model (`responses_model`, default `gpt-5.6-terra`) as the
backend. The agent's tools are registered as its function tools
(`delegation.responses.tools`), and the tools still run in your application:

1. the backend calls a function (a `response.output_item.done` inside a `response.event`
   envelope): the session runs the tool;
2. the output goes back as `response.item.create` (`function_call_output`);
3. once every call of that backend response has an output, the engine sends
   `response.create` and the backend continues. GPT-Live then says the result.

```python
from voice_agent_next.providers.openai.live import OpenAILiveEngine

engine = OpenAILiveEngine(
    responses_model="gpt-5.6-luna",  # cheaper backend
    responses_instructions="You check orders with lookup_order. Report the status only.",
    web_search=True,  # the hosted web_search tool as well
    responses={"reasoning": {"effort": "low"}, "service_tier": "priority"},
)
```

Keep the agent's instructions (the voice model's prompt) short and about conversation
style and when to delegate. Put business rules and tool workflows in
`responses_instructions`. An agent handoff updates the backend tools through
`session.update`, and `session.update_instructions()` appends the new instructions (the
startup prompt itself cannot change). Backend token usage is reported as `LLMMetrics` (one per
backend response).

### Client delegation

With `delegation="client"` GPT-Live asks *your application*. `session.delegation.created`
contains only metadata, so the engine turns it into a call of the agent's
`delegation_tool` (default `"delegate"`). The call has one argument, `request`: what the
user said since the previous delegation. Its `call_id` is the delegation id, and
`session.history` has the whole conversation. The tool's output goes back with that
delegation id as `session.commentary.append` (said aloud), or as
`session.thinking.append` for tools with `scheduling="silent"`. Failed tools send
"The task failed: ...".

```python
from voice_agent_next import Agent, AgentSession, ToolContext, function_tool


@function_tool
async def delegate(ctx: ToolContext, request: str) -> str:
    """Answer a request the voice model delegated."""
    return await my_agent.run(request, history=ctx.session.history)


session = AgentSession(OpenAILiveEngine(delegation="client"))
await session.run(Agent("Be warm and brief.", tools=[delegate]), transport)
```

The engine waits up to `delegation_wait` (0.6 s) for the user's transcript to catch up
with the delegation before it calls the tool.

## Mute, context and usage

`AgentSession.connection` is an `OpenAILiveConnection` with the Live controls:

```python
conn = session.connection
await conn.mute_input()  # session.input_audio.mute; waits for session.input_audio.muted
await conn.unmute_input()
await conn.append_thinking("The user is on the checkout page with two items in the cart.")
await conn.append_instructions("Stop talking about pricing.")
print(conn.session_id, conn.usage_seconds)
```

Muting stops the model from hearing the user. The agent keeps talking and backend work
goes on. The mute state carries over to the next session after a rotation.

## Session expiry and rotation

`session.started` reports `expires_at`. `expiry_warning` seconds before it (default
120 s), the engine emits `EngineStatus("expiring")`. The conversation then moves to a
fresh session at a quiet moment: nobody speaking, nothing playing, no delegated call
waiting for its result. The switch is make-before-break: the next session is started and
seeded before the old one is closed. See [Session rotation](../concepts/session-rotation.md).
A dropped connection, or `session.closed` with reason `expired` or `connection_lost`,
reconnects the same way. A `session.closed` with reason `content` (a safety filter) or
`remote_hangup` ends the conversation with a non-recoverable error.

The new session is seeded through `session.input` with the carried-over text history (at
most 128 messages and about 8k tokens; tool results become developer messages). The
Live API keeps no conversation between sessions unless you fork a stored one, and forking
is not implemented here. `aclose()` sends `session.close` and waits for `session.closed`
(up to `close_timeout`) so the final usage is confirmed.

## Options

| Option | Default | Meaning |
|---|---|---|
| `voice` | `marin` | `alloy`, `ash`, `ballad`, `beacon`, `bossa`, `cedar`, `cinder`, `coral`, `delta`, `echo`, `gleam`, `marin`, `meridian`, `quartz`, `ripple`, `sage`, `shimmer`, `stone`, `tempo`, `verse`, `vesper`, `willow`, or `{"id": ...}` (fixed per session; `Agent(voice=...)` wins) |
| `sample_rate` | `24000` | PCM16 rate of both directions (`16000` or `24000`) |
| `delegation` | `"responses"` | or `"client"` |
| `responses_model`, `responses_instructions`, `web_search`, `responses` | `gpt-5.6-terra`, none, `False`, `{}` | Responses backend |
| `delegation_tool`, `delegation_wait` | `"delegate"`, `0.6` | client delegation |
| `store` | server default (`False`) | keep the session for forking or recording download |
| `session` | `{}` | extra `session.start` fields, deep-merged last |
| `speech_threshold_db`, `response_gap`, `yield_gap`, `transcript_grace` | `-40`, `0.64`, `0.24`, `0.3` | agent speech segmentation |
| `report_overlap`, `handover_delay`, `user_vad_threshold_db`, `user_min_silence` | `False`, `0.3`, `-40`, `0.3` | user speech reporting |
| `keepalive` | `True` | stream silence while the transport is idle |
| `connect_timeout`, `close_timeout`, `expiry_warning` | `10`, `5`, `120` | seconds |
| `rotation` | `RotationPolicy()` | see [Session rotation](../concepts/session-rotation.md) |

`OpenAILiveSessionEngine` takes the same options without `rotation`: it runs one Live
session per connection, and its connection is `LiveSessionConnection`.

## Testing

`voice_agent_next.testing.openai_live.FakeLiveServer` is an offline fake of the Live
endpoint. It supports scripted turns, full-duplex yielding, both delegation modes, mute,
expiry and drops. `tests/test_openai_live.py` runs it through the engine and
`AgentSession`. The real-API test is marked `integration`:

```bash
OPENAI_API_KEY=... uv run pytest -m integration tests/test_openai_live.py
```

## Limitations

* WebSocket only. WebRTC, SIP, the sideband connection and forking are not implemented.
* G.711 (`audio/pcmu`, `audio/pcma`) is not offered: the engine uses PCM16. Telephony
  transports resample.
* The segmentation of agent speech into responses is inferred from the audio level and
  the transcript. The primary WebSocket sends no speech boundaries.
* A transcript fragment that arrives after its response ended (more than
  `transcript_grace` late) starts a new, text-only response.
* The protocol was implemented from the public docs (checked 2026-09-25):
  [Getting started](https://developers.openai.com/api/docs/guides/live),
  [WebSockets](https://developers.openai.com/api/docs/guides/voice-websockets?api=live),
  [Managing sessions](https://developers.openai.com/api/docs/guides/live-conversations),
  [Delegation and tools](https://developers.openai.com/api/docs/guides/live-delegation)
  and the [primary WebSocket reference](https://developers.openai.com/api/reference/resources/live/primary-websocket).
  The docs do not state the session duration limit, whether the server streams silence
  between utterances, or the exact moment audio and transcripts arrive relative to each
  other. The engine handles every combination, but its segmentation defaults were tuned
  on the fake server only.
