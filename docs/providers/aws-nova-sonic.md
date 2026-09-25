# Amazon Nova 2 Sonic (`aws` engine)

Amazon Nova 2 Sonic (`amazon.nova-2-sonic-v1:0`) is AWS's speech-to-speech model on
Amazon Bedrock. It hears 16 kHz audio and answers with 24 kHz speech, detects the end of
the user's turn itself, calls tools asynchronously (it keeps talking while a tool runs)
and accepts typed text in the middle of a voice session. One connection is an HTTP/2
bidirectional event stream (`InvokeModelWithBidirectionalStream`) that lasts at most
**8 minutes**. The engine rotates to a new one before then and carries the conversation
over. The code is in `voice_agent_next/providers/aws/nova_sonic.py`.

```python
from voice_agent_next import Agent, AgentSession, function_tool


@function_tool
async def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return await weather_api(city)


session = AgentSession("aws/nova-2-sonic")
await session.run(
    Agent("You are a friendly travel assistant. Keep answers short.", tools=[get_weather]),
    transport,
)
```

```yaml
# agent.yaml
engine:
  provider: aws/nova-2-sonic
  region: us-east-1
  voice: tiffany
  endpointing_sensitivity: MEDIUM
```

## Setup

```bash
pip install 'voice-agent-next[aws]'   # or: uv sync --extra aws
```

The `aws` extra installs `aws-sdk-bedrock-runtime`, AWS's new Python SDK built on Smithy.
It is **experimental** and needs **Python 3.12 or later**. On Python 3.11 the extra
installs nothing and the engine raises `MissingDependencyError` when it connects.

Why not boto3: Bedrock's bidirectional streaming is only in the new SDK. boto3 and
aiobotocore have no bidirectional event streams. AWS's own Nova Sonic samples use the new
SDK too. The SDK's API still changes between minor versions, so the extra pins
`>=0.11,<0.12`. The engine calls it only through a small `NovaStream` interface (`send`,
`receive`, `close`), so an SDK update only touches `BedrockStream`.

**Credentials** come from the standard AWS chain: environment variables
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`), the shared
`~/.aws/config` and `~/.aws/credentials` files (`AWS_PROFILE` or `profile=`), then
container and EC2 instance metadata. You can also pass them explicitly with
`aws_access_key_id=`, `aws_secret_access_key=` and `aws_session_token=`. The IAM
principal needs `bedrock:InvokeModelWithBidirectionalStream` on the model. You may also
have to enable model access in the Bedrock console.

**Region**: `region=`, else `AWS_REGION` or `AWS_DEFAULT_REGION`, else the profile's
region, else `us-east-1`. Nova 2 Sonic is available in us-east-1, us-west-2, eu-north-1
and ap-northeast-1. `endpoint_url=` overrides the endpoint, for example for a VPC
endpoint.

**Models**: `aws/nova-2-sonic` (default), `aws/nova-sonic` (the first-generation
`amazon.nova-sonic-v1:0`, which has no endpointing option), or any full model ID or
inference-profile ARN (`aws/<id>`).

## How the protocol maps onto the engine events

At the start of each connection the engine sends `sessionStart` (inference settings and
endpointing), then `promptStart` (the output voice and format, and the tools), then the
system prompt, then the carried-over chat history. After that it opens **one** audio
content block and streams `audioInput` into it until the connection ends.

| Nova Sonic | Engine events |
|---|---|
| local energy VAD on the sent audio, while the agent is quiet | `InputSpeechStarted` / `InputSpeechStopped` |
| `USER` text block (the transcript of the user's speech) | `InputTranscript` (partial) |
| the first assistant block after the user spoke, or the user block's `END_TURN` | `InputCommitted` + `InputTranscript(is_final=True)` |
| first `ASSISTANT` block (`SPECULATIVE` text, audio or tool) | `ResponseStarted` |
| `audioOutput` | `ResponseAudio` (24 kHz) |
| `SPECULATIVE` text (or `FINAL` with `transcript="final"`) | `ResponseText` |
| audio `contentEnd(END_TURN)` | `ResponseDone` (with the turn's token usage) |
| `contentEnd(INTERRUPTED)` or v1's `{"interrupted": true}` | `InputSpeechStarted` + `ResponseDone(status="cancelled")` (barge-in) |
| `toolUse` | `ResponseToolCall` |
| `usageEvent` | `ResponseDone.usage`, `EngineMetrics` tokens, `connection.usage` |
| stream error or end | `EngineErrorEvent`, then a reconnect if the error is retryable |

**Transcripts.** Nova sends two versions of what it says. The `SPECULATIVE` text is a
preview of each sentence, sent just before its audio. The `FINAL` text is what was
actually spoken, sent after the audio. `transcript="speculative"` (the default) streams
the preview, so the text arrives together with the speech. `transcript="final"` reports
what was really said, and each response then waits up to `final_grace` seconds after its
audio for that text.

**Barge-in.** The model detects when the user talks over it: it stops speaking and
reports `INTERRUPTED`. The engine turns that into `InputSpeechStarted` followed by
`ResponseDone(status="cancelled")`, so the session clears playback right away. The local
VAD reports user speech only while the agent is quiet. Speech over the agent is left to
the model, because the protocol cannot cancel a response. `report_overlap=True` also
reports overlapping speech, so the session's
[interruption policy](../concepts/interruptions.md) decides instead. In that case
`cancel_response()` drops the rest of the model's turn locally. The model's own context
still holds the whole reply.

**Text input.** `send_text()`, `say()` and `create_response()` send cross-modal text
(`contentStart` TEXT, `interactive: true`), and the model answers it. `say()` asks the
model to say a text verbatim. Most replies comply, but the wording is not guaranteed. If
the model repeats the text back as a user transcript, the engine does not report it as
user speech.

## Tools

Tools are declared in `promptStart` (`toolSpec` with the JSON schema). `tool_choice` can
be `"auto"` (the default), `"any"` or a tool name. Nova 2 Sonic calls tools
**asynchronously**: it keeps talking while a tool runs, so the engine reports
`tool_mode="non_blocking"`, like Gemini Live. When a result is ready, the session sends it
as a `toolResult` block. The content is a JSON object: the tool's own JSON object if it
returned one, otherwise `{"result": ...}` or `{"error": ...}`. The model then speaks
about the result on its own. Nova expects a result for every `toolUse`, and failed or
timed-out tools send an error result. A call cancelled with `session.cancel_tool_call()`
sends nothing, and the model may then keep waiting for it.

A tool's `scheduling="silent"` cannot be honored: Nova decides by itself whether to talk
about a result.

## Rotation before the 8-minute limit

`NovaSonicEngine` (what `"aws/..."` creates) is a
[`RotatingEngine`](../concepts/session-rotation.md) over the single-connection
`NovaSonicSessionEngine`. By default it starts looking for a quiet moment at minute 5
(`RotationPolicy(lead=180)`) and forces the switch 10 s before the limit. The next
connection opens in the background and receives the same system prompt, tools and voice,
plus the conversation *as the user heard it* as chat history. Rules the engine enforces:

- The history goes into `USER`/`ASSISTANT` blocks with `interactive: false`, after the
  system prompt and before any audio.
- Consecutive messages from the same role are merged, because the roles must alternate.
- Tool results become assistant text.
- A carry-over summary is appended to the system prompt.
- Only the newest messages that fit in ~190 KB are kept (Nova's limit is 200 KB).

The switch happens when nobody is talking. User audio is buffered while the connections
change. The new connection's 8 minutes count from when its stream opened, not from the
switch, so a connection that waited for a quiet moment rotates earlier. A dropped stream
reconnects the same way, with backoff.

```python
from voice_agent_next import create
from voice_agent_next.engines import RotationPolicy, SummarizeHistory

engine = create(
    "engine",
    "aws/nova-2-sonic",
    rotation=RotationPolicy(carry_over=SummarizeHistory(create("llm", "openai/gpt-4.1-mini"))),
)
```

The system prompt, voice and tools are fixed for a connection. `update(instructions=...,
tools=...)`, as used by agent handoffs, therefore moves the conversation to a fresh
connection that has the new ones, at the next quiet moment.

## Options

| Option | Default | Meaning |
|---|---|---|
| `region`, `profile`, `endpoint_url` | AWS chain, `us-east-1` | where to connect |
| `aws_access_key_id`, `aws_secret_access_key`, `aws_session_token` | AWS chain | explicit credentials |
| `voice` | `matthew` | `matthew`, `tiffany`, `amy`, `olivia`, `lupe`, `carlos`, `ambre`, `florian`, `lennart`, `beatrice`, `lorenzo`, `tina`, `carolina`, `leo`, `kiara`, `arjun` (fixed per connection; `Agent(voice=...)` wins) |
| `output_sample_rate` | `24000` | `24000`, `16000` or `8000` (telephony) |
| `endpointing_sensitivity` | service default | `HIGH` (fast turn ends), `MEDIUM`, `LOW` (patient) |
| `max_tokens`, `top_p`, `temperature` | `1024`, `0.9`, `0.7` | `inferenceConfiguration` (`EngineOptions.temperature` wins) |
| `tool_choice` | `auto` | `auto`, `any` or a tool name |
| `transcript` | `speculative` | or `final` (see above) |
| `final_grace`, `tool_grace` | `1.0`, `0.3` | seconds a response waits for its final text or after a tool call |
| `report_overlap`, `user_vad_threshold_db`, `user_min_silence` | `False`, `-40`, `0.3` | local user-speech reporting |
| `keepalive` | `True` | stream silence while the transport delivers no audio (Nova times out without continuous audio) |
| `connect_timeout`, `close_timeout` | `10`, `2` | seconds |
| `session_limit` | `480` | connection limit (seconds) used for the rotation schedule |
| `rotation` | `RotationPolicy(lead=180)` | see [Session rotation](../concepts/session-rotation.md) |
| `stream_factory` | the Bedrock SDK | a custom `NovaStream` opener (tests, proxies) |

`EngineOptions.extra["session_start"]` and `extra["prompt_start"]` add raw fields to
those events.

## Testing

`voice_agent_next.testing.nova_sonic.FakeNovaSonic` is an offline fake of the event
stream, passed as `stream_factory`. It checks the order and shape of the input events,
detects the end of the user's turn from audio energy, and answers scripted turns
(`FakeNovaTurn`): the user transcript, an optional tool call, and the reply sentence by
sentence (speculative text, real-time audio, final text), with usage events. It also
simulates barge-in (both Nova 2 and v1 signalling), cross-modal text, the connection limit
and failures. `tests/test_nova_sonic.py` runs it through the engine and through
`AgentSession` with `LoopbackTransport`, and tests the SDK adapter against fake SDK
modules. The real-API test is marked `integration` and costs money, so it also needs an
opt-in variable:

```bash
VAN_TEST_NOVA_SONIC=1 AWS_PROFILE=... uv run --extra aws pytest -m integration tests/test_nova_sonic.py
```

## Limitations

* The protocol was implemented from the public docs (checked 2026-09-25):
  [getting started](https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-getting-started.html),
  [input events](https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-input-events.html),
  [output events](https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-output-events.html),
  [tools](https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-tool-configuration.html),
  [chat history](https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-chat-history.html),
  [cross-modal input](https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-cross-modal.html),
  and SDK version 0.11.0. It has not been run against the real service yet (no AWS
  account in CI). The exact order of the `FINAL` text, `usageEvent` and `completionEnd`
  relative to the audio is not documented. The engine handles either order.
* No server-side cancel or truncation: when the session interrupts the agent (only with
  `report_overlap=True`), the model still remembers its whole reply. Barge-in that the
  model detects itself is exact.
* The carried-over history is text. A new connection does not hear the earlier audio.
* `send_text(respond=False)` and silent tool results are not possible: Nova answers every
  text message by itself.
* The SDK is experimental and needs Python 3.12 or later.
