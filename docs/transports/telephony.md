# Telephony transports (Twilio, Telnyx, Vonage, Plivo)

`voice_agent_next.transports.telephony` puts voice agents on phone calls. When a call
comes in, the provider opens a WebSocket to your server and streams the call's audio over
it (a *media stream*). The server answers each call with its own `AgentSession`.

Each provider speaks its own WebSocket dialect. A per-provider **serializer**
(`TelephonySerializer`) translates it to one common model:

| | Twilio | Telnyx | Vonage | Plivo |
|---|---|---|---|---|
| Framing | JSON, base64 | JSON, base64 RTP payload | **binary** PCM plus JSON control | JSON, base64 |
| Audio in | μ-law 8 kHz | PCMU/PCMA 8 kHz, L16 16 kHz | L16 8/16/24 kHz | μ-law 8 kHz, L16 8/16 kHz |
| Audio out | μ-law 8 kHz | the `bidirectional_codec` you configure | same as in, in 20 ms frames | same as in (`playAudio`) |
| Barge-in | `clear` | `clear` | `{"action": "clear"}` | `clearAudio` |
| Playback position | `mark` | `mark` | `notify` → `websocket:notify` | `checkpoint` → `playedStream` |
| DTMF | `dtmf` | `dtmf` | `websocket:dtmf` | `dtmf` |
| Caller hangs up | `stop`, then close | `stop`, then close | socket closes | socket closes |
| REST hang-up | ✓ (`account_sid`, `auth_token`) | ✓ (`api_key`) | follow-up (needs a JWT) | ✓ (`auth_id`, `auth_token`) |

The protocols were checked against the providers' documentation on 2026-09-24:
[Twilio](https://www.twilio.com/docs/voice/media-streams/websocket-messages),
[Telnyx](https://developers.telnyx.com/docs/voice/programmable-voice/media-streaming),
[Vonage](https://developer.vonage.com/en/voice/voice-api/concepts/websockets),
[Plivo](https://www.plivo.com/docs/voice-agents/audio-streaming/concepts/audio-streaming-reference).

## Quick start

```python
import asyncio
import os

from voice_agent_next import Agent, AgentSession
from voice_agent_next.transports.telephony import serve_telephony


def make_agent(transport) -> Agent:
    call = transport.call  # CallInfo: call_id, from_number, custom_parameters...
    return Agent(
        "You are a friendly phone assistant. Keep answers short.", greeting="Hello! How can I help?"
    )


async def main() -> None:
    server = await serve_telephony(
        lambda: AgentSession("openai/gpt-realtime"),  # or a cascade
        make_agent,
        provider="twilio",  # "telnyx" | "vonage" | "plivo"
        host="0.0.0.0",
        port=8765,
        stream_secret=os.environ["VAN_TELEPHONY_SECRET"],  # shared with your webhook
        serializer_options={"account_sid": "AC...", "auth_token": "..."},  # optional
    )
    await server.serve_forever()


asyncio.run(main())
```

Every media stream must prove that it belongs to a call you answered: your webhook puts
a **stream token** in the markup and the server checks it (see [Security](#security)).
Generate the secret once, for example with
`python -c "import secrets; print(secrets.token_hex(32))"`, and give it to both the
webhook and the server (`stream_secret=` or the `VAN_TELEPHONY_SECRET` environment
variable). Without a secret the server refuses to start.

Providers only connect to public `wss://` URLs. Put the server behind a TLS reverse proxy,
or use a tunnel (for example `ngrok http 8765`) during development. Factories may take the
call's `TelephonyTransport` as an argument (see `transport.call`).

For a single call without a server loop, create a standalone transport. It serves the first
call and ends when that call hangs up:

```python
transport = create_transport({"type": "twilio", "host": "0.0.0.0", "port": 8765})
await session.run(agent, transport)  # stream_secret from VAN_TELEPHONY_SECRET
```

The CLI does the same: `van run --transport twilio` (and `van serve -p twilio` for many
calls); both read the secret from `VAN_TELEPHONY_SECRET`. The transport types are `twilio`,
`telnyx`, `vonage`, `plivo`, and `telephony` (which takes a `provider` option).

## Provider setup

Each provider needs a voice webhook that returns call-control markup pointing at your
WebSocket. `voice_agent_next.transports.telephony` has helpers for this. Serve their output
from any HTTP framework. Pass them the stream `secret` and the `call_id` of the webhook
request: they add the call's stream token (`vanToken`). `stream_token(secret, call_id)`
computes it when you build the markup yourself.

### Twilio: TwiML `<Connect><Stream>`

```python
from voice_agent_next.transports.telephony import (
    twilio_stream_twiml,
    validate_twilio_signature,
)


def twiml_webhook(form: dict[str, str], signature: str | None) -> tuple[int, str]:
    """The "A call comes in" webhook (HTTP POST, form parameters)."""
    url = "https://agent.example.com/twiml"  # the public URL configured at Twilio
    if not validate_twilio_signature(TWILIO_AUTH_TOKEN, url, form, signature):
        return 403, "Forbidden"
    twiml = twilio_stream_twiml(
        "wss://agent.example.com/twilio",
        {"customer": "42"},
        secret=SECRET,
        call_id=form["CallSid"],
    )
    return 200, twiml


# <Response><Connect><Stream url="wss://agent.example.com/twilio">
#   <Parameter name="customer" value="42"/><Parameter name="vanToken" value="..."/>
# </Stream></Connect></Response>
```

Twilio also signs the WebSocket upgrade of the media stream with `X-Twilio-Signature`
(HMAC-SHA1 of the `wss://` URL with your auth token). Pass
`public_url="wss://agent.example.com"` (the URL from the TwiML, without the path) to
`TelephonyServer` together with the Twilio `auth_token`, and upgrades without a valid
signature get HTTP 403. The stream URL cannot carry a query string, so the token travels
as a `<Parameter>`.

Set the phone number's "A call comes in" webhook to a URL that returns this TwiML with
`Content-Type: text/xml`. `<Connect><Stream>` makes the stream bidirectional. Twilio
allows one bidirectional stream per call, carrying the inbound track only. The audio is
always μ-law 8 kHz. `<Parameter>` values arrive in `transport.call.custom_parameters`.
When the WebSocket closes, Twilio runs the TwiML that follows `<Connect>`. If nothing
follows, the call ends.

### Telnyx: TeXML `<Stream>` or Call Control

```python
from voice_agent_next.transports.telephony import telnyx_stream_texml

texml = telnyx_stream_texml(
    "wss://agent.example.com/telnyx",
    codec="PCMU",
    sample_rate=8000,
    secret=SECRET,
    call_id=call_control_id,  # CallControlId of the TeXML webhook request
)
```

With Call Control, pass `stream_url`, `stream_track: "inbound_track"`,
`stream_bidirectional_mode: "rtp"` and `stream_bidirectional_codec` to *answer* or
*streaming_start*, and the token as `stream_auth_token` (Telnyx sends it in the
`x-telnyx-streaming-auth-token` header) or as a `custom_parameters` entry:

```python
from voice_agent_next.transports.telephony import stream_token

token = stream_token(SECRET, call_control_id)
body = {"stream_url": "wss://agent.example.com/telnyx", "stream_auth_token": token}
# or: "custom_parameters": [{"name": "vanToken", "value": token}]
```

The token is bound to the `call_control_id`, so for an outbound *dial* start the stream
with *streaming_start* once the call exists. Telnyx does not repeat the outbound codec in its `start` message, so give
the serializer the same values:

```python
serve_telephony(
    ...,
    provider="telnyx",
    serializer_options={
        "outbound_encoding": "L16",
        "outbound_sample_rate": 16000,
        "api_key": "KEY...",
    },
)
```

L16 is 16 kHz linear PCM. RTP carries L16 in network byte order, which is the serializer's
default (`l16_byteorder="big"`).

### Vonage: NCCO `connect` to a `websocket` endpoint

```python
from voice_agent_next.transports.telephony import vonage_ncco

ncco = vonage_ncco(
    "wss://agent.example.com/vonage",
    sample_rate=16000,
    headers={"customer": "42"},
    secret=SECRET,
    call_id=answer["uuid"],  # the answer webhook's call uuid
)
# [{"action": "connect", "endpoint": [{"type": "websocket", "uri": "...",
#   "content-type": "audio/l16;rate=16000",
#   "headers": {"customer": "42", "uuid": "...", "vanToken": "..."}}]}]
```

Vonage does not identify the call in `websocket:connected`, so the helper adds the `uuid`
header that the token is bound to.

Return the NCCO from your answer webhook. Audio is 16-bit little-endian PCM in binary
frames. Vonage recommends 16 kHz for speech recognition. The transport sends whole 20 ms
frames, padding the last frame of a response with silence. The `headers` arrive in
`transport.call.custom_parameters`. To hang up, the transport closes the WebSocket.
REST hang-up needs an application JWT and is not implemented yet. Vonage recommends
ending the call leg over REST so that it does not raise a `disconnected` event.

### Plivo: XML `<Stream bidirectional="true">`

```python
from voice_agent_next.transports.telephony import plivo_stream_xml

xml = plivo_stream_xml(
    "wss://agent.example.com/plivo",
    content_type="audio/x-l16;rate=16000",
    extra_headers={"customer": "42"},
    secret=SECRET,
    call_id=request["CallUUID"],  # from the answer URL request
)
```

Plivo only allows letters and digits in `extraHeaders`; the token is hexadecimal.

`contentType` is `audio/x-mulaw;rate=8000` (the default), `audio/x-l16;rate=8000` or
`audio/x-l16;rate=16000`. The `playAudio` messages use the same format.
`keepCallAlive="true"` keeps the call up while the stream runs. `extraHeaders` values
(`k=v,k2=v2`) arrive in `transport.call.custom_parameters`.

## Security

The media-stream WebSocket is reachable by anyone who knows its URL, and the transport
acts on what the carrier says: it can hang up the call through the provider's REST API
with your credentials. So:

* **Stream tokens (on by default).** The start message must carry
  `stream_token(secret, call_id)` for the call it describes, as the `vanToken` custom
  parameter (Twilio/Telnyx `<Parameter>` or `custom_parameters`, Vonage header, Plivo
  `extraHeaders`) or as a Telnyx `stream_auth_token`. Otherwise the stream is closed with
  code 1008 before a session starts. The token is an HMAC-SHA256 of the call ID keyed with
  your secret, so it only authorizes that call. The transport removes it from
  `call.custom_parameters`. `authenticate=False` accepts any client (local development
  only).
* **Webhook signatures.** Check that the webhook request comes from the provider before
  you issue a token: `validate_twilio_signature` implements Twilio's
  `X-Twilio-Signature`. See the provider's docs for
  [Telnyx](https://developers.telnyx.com/docs/messaging/webhooks/receiving-webhooks),
  [Vonage](https://developer.vonage.com/en/getting-started/concepts/webhooks) and
  [Plivo](https://www.plivo.com/docs/voice/concepts/signature-validation) signatures.
* **Call IDs are validated.** A start message is refused unless its call ID has the
  provider's format: Twilio `CA` plus 32 hex digits, a Telnyx `v<n>:` call control ID
  (letters, digits, `-`, `_`), a Plivo UUID. The IDs are also percent-encoded in the
  REST URL, so a call ID such as `../Number/...` can never reach another resource.
* **Account IDs come from your configuration only.** The hang-up URL uses the configured
  Twilio account SID or Plivo Auth ID, never the one in the start message. A stream whose
  `accountSid` / `accountId` differs from the configured one is refused. For a Twilio or
  Plivo subaccount, configure the subaccount's own credentials.
* **Hang-up needs trust.** `hangup_on_close` defaults to hanging up authenticated streams
  only.

References (checked 2026-09-25):
[Twilio webhook security](https://www.twilio.com/docs/usage/security),
[Twilio `<Stream>`](https://www.twilio.com/docs/voice/twiml/stream),
[Twilio Call resource](https://www.twilio.com/docs/voice/api/call-resource),
[Telnyx streaming_start](https://developers.telnyx.com/api-reference/call-commands/streaming-start),
[Telnyx hangup](https://developers.telnyx.com/api-reference/call-commands/hangup-call),
[Vonage WebSockets](https://developer.vonage.com/en/voice/voice-api/concepts/websockets),
[Plivo audio streaming XML](https://www.plivo.com/docs/voice/xml/audio-streaming),
[Plivo stream protocol](https://www.plivo.com/docs/voice-agents/audio-streaming/concepts/audio-streaming-reference).

## What the transport does

* **Audio.** Caller audio is decoded to s16le at the stream's rate. The session resamples
  it for the engine. Agent audio is resampled to the stream's rate, encoded, and sent in
  20 ms messages (`frame_duration`). A partial frame waits for the next write. If playback
  is about to run dry first, the partial frame is flushed.
* **Playback position** (`capabilities.playback_position`). Every `mark_interval` (100 ms)
  of audio, the transport sends a mark after the media (`mark`, `notify` or `checkpoint`).
  The provider echoes a mark once playback reaches it.
  `buffered_duration()` is anchored at the last echoed mark and extrapolated in real time.
  The estimate never passes the next mark that has not come back. A barge-in therefore
  truncates the agent's reply (and the transcript kept in the history) to what the caller
  actually heard, including the carrier's jitter buffer and network delay. A mark that
  has not come back `mark_grace` (1 s) after it was due is ignored. A provider that never
  echoes marks therefore falls back to the wall-clock estimate.
* **Barge-in** (`clear_audio`). Audio still queued on our side is dropped, and the
  provider's clear message is sent. Marks that the clear flushes back are ignored.
* **DTMF** (`capabilities.dtmf`). The transport emits `"dtmf"` events carrying the
  digit: `transport.on("dtmf", lambda digit: ...)`.
* **Call start and end.** `"call_started"` is emitted with `CallInfo`. When the provider
  sends `stop` or closes the socket, the audio input ends and the session closes with
  reason `user_disconnected`. When our side ends the call first (the session closes),
  the transport calls the provider's REST hang-up if credentials are configured and
  `hangup_on_close` allows it (by default: when the stream was authenticated), then
  closes the stream. For transfer flows, set
  `transport_options={"hangup_on_close": False}` so that Twilio or Plivo continue with the
  markup after the stream.
* **Provider-specific messages.** `await transport.send_raw({...})` sends a message
  verbatim, for example Plivo's `{"event": "sendDTMF", "dtmf": "1234#"}`. Phone calls have
  no data channel, so `send_message()` (used for transcripts and state) is a no-op.

Credentials come from the serializer options or from the environment:
`TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN`, `TELNYX_API_KEY`, and `PLIVO_AUTH_ID` and
`PLIVO_AUTH_TOKEN`. The account SID in Twilio's `start` message is never used.

## Options

| `TelephonyTransport` option | Default | |
|---|---|---|
| `provider` | class default | `"twilio"`, `"telnyx"`, `"vonage"`, `"plivo"`, or a serializer instance |
| `frame_duration` | `0.02` | seconds of audio per outbound media message |
| `mark_interval` | `0.1` | minimum audio between two marks |
| `mark_grace` | `1.0` | seconds after a mark was due before it is ignored |
| `start_timeout` | `10.0` | seconds to wait for the provider's start message |
| `hangup_on_close` | `None` | REST hang-up when our side ends the call (`None`: only for authenticated streams) |
| `stream_secret` | `VAN_TELEPHONY_SECRET` | secret of the stream tokens; required unless `authenticate=False` |
| `authenticate` | `True` | refuse streams without a valid stream token |
| `http_client` | `None` | `httpx.AsyncClient` for the REST hang-up |
| `**serializer_options` | | `account_sid`/`auth_token` (Twilio), `api_key`/`outbound_encoding`/`outbound_sample_rate`/`l16_byteorder` (Telnyx), `sample_rate` (Vonage), `auth_id`/`auth_token`/`l16_byteorder` (Plivo), `api_base` |

`TelephonyServer` / `serve_telephony` take `provider`, `stream_secret`, `authenticate`,
`public_url` (Twilio handshake signatures), `serializer_options`, `transport_options`,
`max_sessions`, and the `websockets` serve options (`ssl`, `process_request`, and so on).

## Latency notes

The public telephone network adds about 230 ms per turn compared with in-platform audio
(research note 05, §4). Request 16 kHz L16 where the provider offers it (Telnyx, Vonage,
Plivo). Narrowband μ-law hurts recognition. The session resamples exactly once in each
direction.
