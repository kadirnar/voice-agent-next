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
        serializer_options={"account_sid": "AC...", "auth_token": "..."},  # optional
    )
    await server.serve_forever()


asyncio.run(main())
```

Providers only connect to public `wss://` URLs. Put the server behind a TLS reverse proxy,
or use a tunnel (for example `ngrok http 8765`) during development. Factories may take the
call's `TelephonyTransport` as an argument (see `transport.call`).

For a single call without a server loop, create a standalone transport. It serves the first
call and ends when that call hangs up:

```python
transport = create_transport({"type": "twilio", "host": "0.0.0.0", "port": 8765})
await session.run(agent, transport)
```

The CLI does the same: `van run --transport twilio`. The transport types are `twilio`,
`telnyx`, `vonage`, `plivo`, and `telephony` (which takes a `provider` option).

## Provider setup

Each provider needs a voice webhook that returns call-control markup pointing at your
WebSocket. `voice_agent_next.transports.telephony` has helpers for this. Serve their output
from any HTTP framework.

### Twilio: TwiML `<Connect><Stream>`

```python
from voice_agent_next.transports.telephony import twilio_stream_twiml

twiml = twilio_stream_twiml("wss://agent.example.com/twilio", {"customer": "42"})
# <Response><Connect><Stream url="wss://agent.example.com/twilio">
#   <Parameter name="customer" value="42"/></Stream></Connect></Response>
```

Set the phone number's "A call comes in" webhook to a URL that returns this TwiML with
`Content-Type: text/xml`. `<Connect><Stream>` makes the stream bidirectional. Twilio
allows one bidirectional stream per call, carrying the inbound track only. The audio is
always μ-law 8 kHz. `<Parameter>` values arrive in `transport.call.custom_parameters`.
When the WebSocket closes, Twilio runs the TwiML that follows `<Connect>`. If nothing
follows, the call ends.

### Telnyx: TeXML `<Stream>` or Call Control

```python
from voice_agent_next.transports.telephony import telnyx_stream_texml

texml = telnyx_stream_texml("wss://agent.example.com/telnyx", codec="PCMU", sample_rate=8000)
```

With Call Control, pass `stream_url`, `stream_track: "inbound_track"`,
`stream_bidirectional_mode: "rtp"` and `stream_bidirectional_codec` to *dial*, *answer* or
*streaming_start*. Telnyx does not repeat the outbound codec in its `start` message, so give
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

ncco = vonage_ncco("wss://agent.example.com/vonage", sample_rate=16000, headers={"customer": "42"})
# [{"action": "connect", "endpoint": [{"type": "websocket", "uri": "...",
#   "content-type": "audio/l16;rate=16000", "headers": {"customer": "42"}}]}]
```

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
)
```

`contentType` is `audio/x-mulaw;rate=8000` (the default), `audio/x-l16;rate=8000` or
`audio/x-l16;rate=16000`. The `playAudio` messages use the same format.
`keepCallAlive="true"` keeps the call up while the stream runs. `extraHeaders` values
(`k=v;k2=v2`) arrive in `transport.call.custom_parameters`.

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
  `hangup_on_close=True`, then closes the stream. For transfer flows, set
  `transport_options={"hangup_on_close": False}` so that Twilio or Plivo continue with the
  markup after the stream.
* **Provider-specific messages.** `await transport.send_raw({...})` sends a message
  verbatim, for example Plivo's `{"event": "sendDTMF", "dtmf": "1234#"}`. Phone calls have
  no data channel, so `send_message()` (used for transcripts and state) is a no-op.

Credentials come from the serializer options or from the environment:
`TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN` (Twilio also sends the account SID in
`start`), `TELNYX_API_KEY`, and `PLIVO_AUTH_ID` and `PLIVO_AUTH_TOKEN`.

## Options

| `TelephonyTransport` option | Default | |
|---|---|---|
| `provider` | class default | `"twilio"`, `"telnyx"`, `"vonage"`, `"plivo"`, or a serializer instance |
| `frame_duration` | `0.02` | seconds of audio per outbound media message |
| `mark_interval` | `0.1` | minimum audio between two marks |
| `mark_grace` | `1.0` | seconds after a mark was due before it is ignored |
| `start_timeout` | `10.0` | seconds to wait for the provider's start message |
| `hangup_on_close` | `True` | REST hang-up when our side ends the call |
| `http_client` | `None` | `httpx.AsyncClient` for the REST hang-up |
| `**serializer_options` | | `account_sid`/`auth_token` (Twilio), `api_key`/`outbound_encoding`/`outbound_sample_rate`/`l16_byteorder` (Telnyx), `sample_rate` (Vonage), `auth_id`/`auth_token`/`l16_byteorder` (Plivo), `api_base` |

`TelephonyServer` / `serve_telephony` take `provider`, `serializer_options`,
`transport_options`, `max_sessions`, and the `websockets` serve options (`ssl`,
`process_request` for authentication, and so on).

## Latency notes

The public telephone network adds about 230 ms per turn compared with in-platform audio
(research note 05, §4). Request 16 kHz L16 where the provider offers it (Telnyx, Vonage,
Plivo). Narrowband μ-law hurts recognition. The session resamples exactly once in each
direction.
